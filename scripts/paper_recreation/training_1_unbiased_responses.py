import argparse
from cot_transparency.formatters.core.unbiased import (
    ZeroShotUnbiasedFormatter,
    ZeroShotCOTUnbiasedFormatter,
)
from stage_one import main

# Writes to experiments/training_data_1_unfiltered/...
# Gets unbiased qs & generates unbiased as (for ARC, OpenBookQA, BIG-Bench Hard) (for both COT and non-COT)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="gpt-4o-mini-2024-07-18",
        help="Specify the model to use (e.g. gpt-3.5-turbo)",
    )
    args = parser.parse_args()

    # Script to replicate generating training data
    exp_dir = "experiments/training_data_1_unfiltered"
    main(
        dataset="cot_training",
        formatters=[ZeroShotCOTUnbiasedFormatter.name(), ZeroShotUnbiasedFormatter.name()],
        example_cap=5000,
        models=[args.model],
        temperature=1.0,
        exp_dir=exp_dir,
        batch=40,
        raise_after_retries=False,
        num_tries=1,
        # High max tokens so that it does not get truncated
        max_tokens=2000,
    )
