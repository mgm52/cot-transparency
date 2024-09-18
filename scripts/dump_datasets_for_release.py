from pathlib import Path
from typing import Literal
from cot_transparency.formatters.more_biases.sarcasm_smart_bias import SarcasmSmartBias
from grugstream import Observable
from pydantic import BaseModel
from slist import Group, Slist
from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai.formatting import append_assistant_preferred_to_last_user
from cot_transparency.data_models.data.inverse_scaling import InverseScalingTask
from cot_transparency.data_models.example_base import MultipleChoiceAnswer
from cot_transparency.data_models.messages import StrictChatMessage
from cot_transparency.data_models.models import TaskSpec
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.inverse_scaling.no_few_shot import (
    ClearFewShotsThinkStepByStepCOT,
)
from cot_transparency.formatters.more_biases.anchor_initial_wrong import PostHocNoPlease
from cot_transparency.formatters.more_biases.distractor_fact import FirstLetterDistractor
from cot_transparency.formatters.more_biases.random_bias_formatter import RandomBiasedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import (
    ImprovedDistractorArgument,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import WrongFewShotMoreClearlyLabelledAtBottom
from cot_transparency.formatters.verbalize.formatters import BlackSquareBiasedFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.util import assert_not_none
from scripts.are_you_sure.eval_are_you_sure_second_cot import (
    OutputWithAreYouSure,
    run_are_you_sure_multi_model_second_round_cot,
)
from scripts.evaluate_judge_consistency.judge_consistency import (
    BothJudgements,
    ComparisonGenerationJudged,
    many_judge_obs,
)

from stage_one import create_stage_one_task_specs

FORMATTERS_TO_NAME = {
    RandomBiasedFormatter.name(): "suggested_answer",
    PostHocNoPlease.name(): "post_hoc",
    WrongFewShotMoreClearlyLabelledAtBottom.name(): "wrong_few_shot",
    BlackSquareBiasedFormatter.name(): "spurious_few_shot_squares",
    FirstLetterDistractor.name(): "distractor_fact",
    ImprovedDistractorArgument.name(): "distractor_argument",
    SarcasmSmartBias.name(): "sarcasm_smart",
    # DistractorAnswerWithoutInfluence.name(): "distractor_argument_2",
    # DistractorArgumentCorrectOrWrong.name(): "distractor_argument_3",
    # DistractorArgumentImportant.name(): "distractor_argument_4",
    # DistractorArgumentNotsure.name(): "distractor_argument_5",
    # DistractorArgumentNoTruthfullyAnswer.name(): "distractor_argument_6",
}

DEFAULT_FORMATTERS_LIST = [
    RandomBiasedFormatter.name(),
    PostHocNoPlease.name(),
    WrongFewShotMoreClearlyLabelledAtBottom.name(),
    BlackSquareBiasedFormatter.name(),
    FirstLetterDistractor.name(),
    ImprovedDistractorArgument.name(),
    SarcasmSmartBias.name(),
]


def rename_dataset_name(dataset_name: str) -> str:
    match dataset_name:
        case "truthful_qa":
            return "truthfulqa"
        case "mmlu_test":
            return "mmlu"
        case _:
            return dataset_name


class PositionalBiasDump(BaseModel):
    original_instruction: list[StrictChatMessage]
    gpt_35_response: str
    gpt_4_response: str
    first_judge_prompt: list[StrictChatMessage]
    # Same as the first_judge_prompt, but flipped the order of choices
    second_judge_prompt: list[StrictChatMessage]
    original_dataset: Literal["alpaca_cleaned"] = "alpaca_cleaned"
    bias_name: Literal["positional_bias"] = "positional_bias"

    @staticmethod
    def from_positional_bias(output: BothJudgements) -> "PositionalBiasDump":
        first_judgement: ComparisonGenerationJudged = assert_not_none(output.first_judgement)
        second_judgement: ComparisonGenerationJudged = assert_not_none(output.second_judgement)
        original_instruction = output.original_instruction_seq
        gpt_4_response = second_judgement.generation.a_response
        gpt_35_response = first_judgement.generation.b_response
        return PositionalBiasDump(
            original_instruction=[instruction.to_strict() for instruction in original_instruction],
            first_judge_prompt=[prompt.to_strict() for prompt in first_judgement.judge_prompt],
            second_judge_prompt=[prompt.to_strict() for prompt in second_judgement.judge_prompt],
            gpt_35_response=gpt_35_response,
            gpt_4_response=gpt_4_response,
        )


class StandardDatasetDump(BaseModel):
    original_question: str
    original_question_hash: str
    original_dataset: str
    unbiased_question: list[StrictChatMessage]
    biased_question: list[StrictChatMessage]
    bias_name: str
    ground_truth: MultipleChoiceAnswer
    biased_option: str

    @staticmethod
    def from_task_spec(task_spec: TaskSpec) -> "StandardDatasetDump":
        bias_messages = task_spec.messages
        assistant_on_user_side: list[StrictChatMessage] = append_assistant_preferred_to_last_user(bias_messages)

        unbiased_question: list[StrictChatMessage] = append_assistant_preferred_to_last_user(
            ZeroShotCOTUnbiasedFormatter.format_example(question=task_spec.get_data_example_obj())
        )
        biased_question = assistant_on_user_side
        bias_name: str = FORMATTERS_TO_NAME.get(task_spec.formatter_name, task_spec.formatter_name)
        biased_option = task_spec.biased_ans
        assert biased_option is not None, "Biased option should not be None"

        return StandardDatasetDump(
            original_question=task_spec.get_data_example_obj().get_parsed_input(),
            original_question_hash=task_spec.task_hash,
            original_dataset=rename_dataset_name(task_spec.task_name),
            unbiased_question=unbiased_question,
            biased_question=biased_question,
            bias_name=bias_name,
            ground_truth=task_spec.ground_truth,  # type: ignore
            biased_option=biased_option,
        )

    @staticmethod
    def from_are_you_sure(output: OutputWithAreYouSure) -> "StandardDatasetDump":
        task_spec = output.task_spec
        bias_messages = task_spec.messages
        assistant_on_user_side: list[StrictChatMessage] = append_assistant_preferred_to_last_user(bias_messages)

        unbiased_question: list[StrictChatMessage] = append_assistant_preferred_to_last_user(
            ZeroShotCOTUnbiasedFormatter.format_example(question=task_spec.get_data_example_obj())
        )
        biased_question = assistant_on_user_side
        biased_option = f"NOT {output.first_round_inference.parsed_response}"
        assert biased_option is not None, "Biased option should not be None"

        return StandardDatasetDump(
            original_question=task_spec.get_data_example_obj().get_parsed_input(),
            original_question_hash=task_spec.task_hash,
            original_dataset=rename_dataset_name(task_spec.task_name),
            unbiased_question=unbiased_question,
            biased_question=biased_question,
            bias_name="are_you_sure",
            ground_truth=task_spec.ground_truth,  # type: ignore
            biased_option=biased_option,
        )


async def dump_data(
    formatters_list: list[str] = DEFAULT_FORMATTERS_LIST,
    use_are_you_sure: bool = True,
    use_positional_bias: bool = True,
    use_hindsight_neglect: bool = True,
):
    # delete whole dataset_dumps folder if it exists
    tasks_to_run: Slist[TaskSpec] = Slist(
        create_stage_one_task_specs(
            dataset="cot_testing",
            models=["gpt-4o-mini-2024-07-18"],
            formatters=formatters_list,
            example_cap=2000,  # 2000 each dataset in "mmlu", "truthful_qa", ""
            temperature=0,
            raise_after_retries=False,
            max_tokens=1000,
            n_responses_per_request=1,
        )
    )

    if use_hindsight_neglect:
        hindsight_neglect = Slist(
            create_stage_one_task_specs(
                tasks=[InverseScalingTask.hindsight_neglect],
                models=["gpt-4o-mini-2024-07-18"],
                formatters=[
                    # ClearFewShotsCOT().name(),
                    # ClearFewShotsCOTVariant().name(),
                    ClearFewShotsThinkStepByStepCOT().name(),
                    # ClearFewShotsThinkStepByStepCOTVariant().name(),
                ],
                example_cap=1000,
                temperature=0,
                raise_after_retries=False,
                max_tokens=1000,
                n_responses_per_request=1,
            )
        ).map(
            # rename formatter to inverse_scaling
            lambda x: x.copy_update(formatter_name="spurious_few_shot_hindsight")
        )
    else:
        hindsight_neglect = Slist()

    stage_one_path = Path("experiments/grid_exp")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=600)

    # Are you sure actually depends on the model being used, but we'll just dump the data that the model gets right in the first turn

    if use_hindsight_neglect:
        standard_with_hindsight = tasks_to_run + hindsight_neglect
    else:
        standard_with_hindsight = tasks_to_run

    standard_dumps = standard_with_hindsight.map(StandardDatasetDump.from_task_spec)

    if use_are_you_sure:
        # are you sure function filters for bias_on_wrong_answer
        _are_you_sure_second_round_cot: Slist[OutputWithAreYouSure] = (
            await run_are_you_sure_multi_model_second_round_cot(
                models=["gpt-4o-mini-2024-07-18"], caller=stage_one_caller, example_cap=1000
            )
        )
        # dump are you sure
        are_you_sure_dump: Slist[StandardDatasetDump] = _are_you_sure_second_round_cot.map(
            StandardDatasetDump.from_are_you_sure
        )
        dumps = standard_dumps + are_you_sure_dump
    else:
        dumps = standard_dumps

    # put in a folder titled "dataset_dumps/test". The file will be named "{original_dataset}_{bias_name}.jsonl"
    # make the folder if it doesn't exist

    # group by this name
    dumps_grouped: Slist[Group[str, Slist[StandardDatasetDump]]] = dumps.group_by(
        lambda x: f"{x.bias_name}/{x.original_dataset}_{x.bias_name}.jsonl"
    )
    for group in dumps_grouped:
        write_jsonl_file_from_basemodel(f"dataset_dumps/test/{group.key}", group.values)

    # Positional bias is annoyingly non standard...
    if use_positional_bias:

        pipeline: Observable[BothJudgements] = many_judge_obs(
            judge_models=["gpt-4o-mini-2024-07-18"],
            caller=stage_one_caller,
            samples_to_judge=800,
            first_model="gpt-4o-2024-08-06",
            second_model="gpt-4o-mini-2024-07-18",
        ).filter(lambda x: x.first_judgement is not None and x.second_judgement is not None)
        results: Slist[BothJudgements] = await pipeline.to_slist()
        positional_bias_dumps: Slist[PositionalBiasDump] = results.map(PositionalBiasDump.from_positional_bias)
        write_jsonl_file_from_basemodel(
            "dataset_dumps/test/positional_bias/alpaca_positional_bias.jsonl", positional_bias_dumps
        )


def test_parse_one_file(formatter_name: str = "distractor_fact"):
    # open dataset_dumps/test/mmlu_distractor_fact.jsonl
    with open(f"dataset_dumps/test/{formatter_name}/mmlu_{formatter_name}.jsonl", "r") as f:
        for line in f.readlines():
            # read into the basemodel
            parsed = StandardDatasetDump.model_validate_json(line)
            print(parsed.biased_question)


if __name__ == "__main__":
    import asyncio

    asyncio.run(dump_data([SarcasmSmartBias.name()], False, False, False))
    test_parse_one_file(FORMATTERS_TO_NAME[SarcasmSmartBias.name()])
