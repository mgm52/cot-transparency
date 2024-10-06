from scripts.unfiltered_data_experiments.dump_unfiltered_data import dump_correct_data
import argparse

# Writes to data/training_cots/... and data/training_non_cots/...
# Simply collects unbiased qs & as into more succinct jsonl format
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

    dump_correct_data(
        cot_data=True,
        exp_dir="experiments/training_data_1_unfiltered",
        model=args.model,
    )
    dump_correct_data(
        cot_data=False,
        exp_dir="experiments/training_data_1_unfiltered",
        model=args.model,
    )
