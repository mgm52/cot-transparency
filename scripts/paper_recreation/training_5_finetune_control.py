import argparse


from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
)
from scripts.finetune_cot import FormatterOptions, NFormatsPerQuestionSampler
from scripts.paper_recreation.training_4_finetune_bct import fine_tune_with_bias_augmentation


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
        sampler=NFormatsPerQuestionSampler(1, formatter_options=FormatterOptions.control_only_unbiased),
        n_bct_samples=10_000,
        instruct_sample_proportion=1.0,
        n_val_samples=100,
        cot_percentage=0.5,
        prepend_notes="(Paper recreation - Control) ",
        ask_to_validate_training=False,
    )

    print(f"Fine-tuned model ID: {trained_model_id}")
