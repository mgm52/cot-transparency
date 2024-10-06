import argparse
import json
from pathlib import Path
from collections.abc import Sequence
from functools import lru_cache
from typing import Callable

from slist import Slist, identity

from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
    FineTuneParams,
    FinetuneSample,
    run_finetune_with_wandb,
)
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel
from scripts.num_tokens import num_tokens_for_finetuning_samples
from scripts.finetune_cot import (
    FormatSampler,
    FormatterOptions,
    NFormatsPerQuestionSampler,
    ParaphrasingSampler,
    apply_cot_bias,
    apply_non_cot_bias,
)


@lru_cache
def get_unbiased_training_data(cot: bool, model_name: str):
    return read_jsonl_file_into_basemodel(
        Path(f"data/training_{'cots' if cot else 'non_cots'}/{model_name}_unfiltered.jsonl"), TaskOutput
    )


@lru_cache
def get_alpaca_training_data(seed: str | None, n_train: int, n_val: int, model_name: str):
    alpaca_samples = (
        read_jsonl_file_into_basemodel(Path(f"data/instructions/{model_name}_temp_1.jsonl"), FinetuneSample)
        .shuffle(seed=seed)
        .take(n_train + n_val)
    )
    alpaca_train_samples, alpaca_val_samples = (
        (
            alpaca_samples[:-n_val],
            alpaca_samples[-n_val:],
        )
        # Need to check if >0 because [1,2,3][:-0] is [] and then it goes boom
        if n_val > 0
        else (alpaca_samples, Slist[FinetuneSample]([]))
    )
    assert (
        len(alpaca_train_samples) == n_train
    ), f"Not enough alpaca train samples, only {len(alpaca_train_samples)}, required {n_train}"
    return alpaca_train_samples, alpaca_val_samples, alpaca_samples


def fine_tune_with_bias_augmentation(
    hyperparams: FineTuneHyperParams = FineTuneHyperParams(n_epochs=1, batch_size=16, learning_rate_multiplier=1.6),
    exclude_tasks: Sequence[str] = [],
    project_name: str = "consistency-training",
    model_name: str = "gpt-4o-mini-2024-07-18",
    n_bct_samples: int = 10_000,  # does not include the (n_bct_samples * instruct_sample_proportion) alpaca samples
    instruct_sample_proportion: float = 1.0,
    post_hoc: bool = False,
    cot_percentage=0.5,  # so there are (n_bct_samples * cot_percentage) cot samples
    # CLI waits for user input to validate the training
    ask_to_validate_training: bool = True,
    # For now we recommend using NFormatsPerQuestionSampler=1, rather than RandomSampler
    sampler: FormatSampler = NFormatsPerQuestionSampler(
        # If FormatterOptions.control_only_unbiased, then we only use unbiased contexts for training
        n_formats_per_question=1,
        formatter_options=FormatterOptions.suggested_answer_all,
    ),
    val_sampler: FormatSampler = NFormatsPerQuestionSampler(
        n_formats_per_question=1, formatter_options=FormatterOptions.all_biased
    ),
    prepend_notes: str = "",
    # If true, we permute the verbalize instructions to have multiple variations
    permute_verbalize_instructions: bool = True,
    # Ensures that the cot and non cot questions do not overlap
    # This is useful so the chance of overlaps between the cot and non cot questions does not
    # change when we change the size of the training data
    no_overlap_cot_non_cot: bool = True,
    # Note that training validation is free in the openai api!
    n_val_samples: int = 0,  # does not include the (n_val_samples * instruct_sample_proportion) alpaca val samples
    # Run some postprocessing on the finetune
    post_processing_func: Callable[[Sequence[FinetuneSample]], Sequence[FinetuneSample]] = identity,
    cot_seed: str = "42",
    non_cot_seed: str = "1",
) -> str:
    """
    We use unbiased correct COTs, then replace the unbiased COT prompt with a biased COT formatter prompt
    """
    assert 0 <= cot_percentage <= 1
    assert 0 <= instruct_sample_proportion
    cot_limit = int(cot_percentage * n_bct_samples)
    # split of training data
    non_cot_percentage = 1 - cot_percentage
    non_cot_limit = int(non_cot_percentage * n_bct_samples)
    # split of val samples
    n_non_cot_val_samples = int(n_val_samples * (1 - cot_percentage))
    n_cot_val_samples = int(n_val_samples * cot_percentage)
    val_instruct_samples = int(n_val_samples * instruct_sample_proportion)

    ### GET UNBIASED BCT DATA ###
    cot_data = get_unbiased_training_data(True, model_name)
    non_cot_data = get_unbiased_training_data(False, model_name)

    ### CONVERT TO BIASED BCT DATA ###
    non_cot_samples, non_cot_val_samples, non_cot_hashes, val_task_hashes = apply_non_cot_bias(
        non_cot_data,
        non_cot_seed,
        exclude_tasks,
        non_cot_limit,
        sampler,
        permute_verbalize_instructions,
        val_sampler,
        n_non_cot_val_samples,
    )
    cot_samples, cot_val_samples, cot_hashes = apply_cot_bias(
        cot_data,
        cot_seed,
        exclude_tasks,
        cot_limit,
        sampler,
        permute_verbalize_instructions,
        val_sampler,
        val_task_hashes,
        non_cot_hashes,
        n_cot_val_samples,
        post_hoc,
        no_overlap_cot_non_cot,
    )

    ### GET INSTRUCTION-FOLLOWING (ALPACA) DATA ###
    n_instruct_samples = int(instruct_sample_proportion * (len(non_cot_samples) + len(cot_samples)))

    alpaca_train_samples, alpaca_val_samples, alpaca_samples = get_alpaca_training_data(
        cot_seed, n_instruct_samples, val_instruct_samples, model_name
    )

    ### COLLECT AND SAVE BCT + ALPACA TRAINING DATA ###
    # BCT Hashes
    combined_hashes = cot_hashes.union(non_cot_hashes)
    with open("finetune_cot_hashes.json", "w") as f:  # why is this called "cot_hashes" when it's combined...
        json.dump(list(combined_hashes), f)
    if no_overlap_cot_non_cot:
        assert non_cot_hashes.isdisjoint(cot_hashes), "cot and non cot hashes are not disjoint, this is a bug"
    # BCT + ALPACA Training Data
    samples = (non_cot_samples + cot_samples + alpaca_train_samples).shuffle("42")
    print("ESTIMATED NUMBER OF TOKENS", num_tokens_for_finetuning_samples(samples))
    write_jsonl_file_from_basemodel("bct_non_cot.jsonl", non_cot_samples)
    write_jsonl_file_from_basemodel("bct_cot.jsonl", cot_samples)
    write_jsonl_file_from_basemodel("instruct_samples", alpaca_samples)
    # BCT + ALPACA Validation Data
    val_samples = (non_cot_val_samples + cot_val_samples + alpaca_val_samples).shuffle("42")

    ### HYPERPARAMETER SETUP ###
    params = FineTuneParams(model=model_name, hyperparameters=hyperparams)
    control_only_unbiased = sampler.format_options_name == FormatterOptions.control_only_unbiased.value

    ### RUN FINETUNE ###
    more_config = {
        "instruct_sample_proportion": instruct_sample_proportion,
        "n_cots": len(cot_samples),
        "n_non_cots": len(non_cot_samples),
        "n_unique_cot_questions": len(cot_hashes),
        "n_unique_non_cot_questions": len(non_cot_hashes),
        "n_train_instruct_samples": len(alpaca_train_samples),
        "n_val_instruct_samples": len(alpaca_val_samples),
        "n_val_cots": len(cot_val_samples),
        "n_val_non_cots": len(non_cot_val_samples),
        "n_val_samples": len(val_samples),
        "excluded_formatters": list(Slist(sampler.excluded_formatters).map(lambda x: x.name())),
        "eligible_non_cot_formatters": [sorted(sampler.eligible_non_cot_formatters.map(lambda x: x.name()))],
        "eligible_cot_formatters": [sorted(sampler.eligible_cot_formatters.map(lambda x: x.name()))],
        "formatter_options": sampler.format_options_name,
        "post_hoc": post_hoc,
        "cot_percentage": cot_percentage,
        "control_only_unbiased": control_only_unbiased,
        "sampling_strategy": sampler,
        "permute_verbalize_instructions": permute_verbalize_instructions,
        "no_overlap_cot_non_cot": no_overlap_cot_non_cot,
        "cot_paraphrasings_from": sampler.cot_paraphrasings_file if isinstance(sampler, ParaphrasingSampler) else None,
        "non_cot_paraphrasings_from": (
            sampler.non_cot_paraphrasings_file if isinstance(sampler, ParaphrasingSampler) else None
        ),
        "non_cot_seed": non_cot_seed,
        "cot_seed": cot_seed,
    }
    cot_percentage_percentage = int(cot_percentage * 100)
    non_cot_percentage_percentage = int(non_cot_percentage * 100)
    bias_type_str = sampler.format_options_name + " bias formatters"
    notes = f"{prepend_notes}{bias_type_str} {cot_percentage_percentage}% cot {non_cot_percentage_percentage}% non cot, {n_bct_samples} samples"
    if post_hoc:
        notes = "post hoc " + notes
    _id = run_finetune_with_wandb(
        params=params,
        samples=post_processing_func(samples),
        notes=notes,
        more_config=more_config,
        project_name=project_name,
        ask_to_validate_training=ask_to_validate_training,
        val_samples=post_processing_func(val_samples),
    )
    return _id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Name of the model to use (e.g. gpt-3.5-turbo-0613).",
    )
    args = parser.parse_args()

    trained_model_id = fine_tune_with_bias_augmentation(
        model_name=args.model_name,
        hyperparams=FineTuneHyperParams(batch_size=16, n_epochs=1, learning_rate_multiplier=1.6),
        sampler=NFormatsPerQuestionSampler(1, formatter_options=FormatterOptions.suggested_answer_all),
        n_bct_samples=10_000,
        instruct_sample_proportion=1.0,
        n_val_samples=100,
        cot_percentage=0.5,
        prepend_notes="(Paper recreation - BCT) ",
        ask_to_validate_training=False,
    )

    print(f"Fine-tuned model ID: {trained_model_id}")
