from typing import List, Dict, Any, Optional, cast
from pathlib import Path
import argparse
import openai

from cot_transparency.apis.openai.finetune import (
    read_jsonl_file_into_basemodel,
    run_finetune_with_wandb_from_file,
    FinetuneSample,
    FineTuneParams,
    FineTuneHyperParams,
    try_upload_until_success,
)

# Prepare arguments
parser = argparse.ArgumentParser(description="Fine-tune a model with custom datasets.")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to fine-tune (default: gpt-3.5-turbo)")
parser.add_argument(
    "--force-upload",
    action="store_true",
    default=False,
    help="Force upload a new dataset even if one exists (default: False).",
)
args = parser.parse_args()


# Check for existing jobs
def get_existing_job_ids() -> List[str]:
    finetune_jobs: Dict[str, Any] = cast(Dict[str, Any], openai.FineTuningJob.list(limit=100))
    return [job["id"] for job in finetune_jobs["data"] if job["status"] in ["running", "validating_files"]]


def cancel_existing_jobs(job_ids: List[str]) -> None:
    for job_id in job_ids:
        try:
            openai.FineTuningJob.cancel(job_id)
            print(f"Successfully canceled job: {job_id}")
        except Exception as e:
            print(f"Failed to cancel job {job_id}: {e}")


# Check for existing files on OpenAI
def get_existing_file_id(file_name: str) -> Optional[str]:
    files: Dict[str, Any] = cast(Dict[str, Any], openai.File.list(limit=100))
    for file in files["data"]:
        if file["filename"] == file_name and file["status"] == "processed":
            return file["id"]
    return None


# Upload a new dataset or reuse an existing one
def upload_or_reuse_file(file_path: Path) -> str:
    file_name: str = file_path.name
    file_id: Optional[str] = None

    if not args.force_upload:
        file_id = get_existing_file_id(file_name)
        if file_id:
            print(f"Reusing previously uploaded file: {file_name} (ID: {file_id})")
            return file_id
        else:
            print(f"No existing file found for {file_name}, uploading a new one.")

    # Upload the file if force_upload is True or if no existing file was found
    file_id = try_upload_until_success(
        samples=read_jsonl_file_into_basemodel(file_path, FinetuneSample), file_path=file_path
    )

    # Ensure file_id is not None
    assert file_id is not None, "File upload failed, file_id is None."

    return file_id


existing_job_ids: List[str] = get_existing_job_ids()

if existing_job_ids:
    print(f"Found {len(existing_job_ids)} existing in-progress jobs!")
    for jid in existing_job_ids:
        finetune_job = openai.FineTuningJob.retrieve(jid)
        print(finetune_job)

    # Ask for confirmation before canceling jobs
    cancel_confirmation = input(f"Do you want to cancel the {len(existing_job_ids)} running jobs? (yes/no): ")
    if cancel_confirmation.lower() == "yes":
        cancel_existing_jobs(existing_job_ids)
else:
    print("No existing jobs found.")

input("Are you sure you want to start a new finetuning job? (enter to continue)")

# Run fine-tuning with either a new upload or reused file
params = FineTuneParams(
    model=args.model,
    hyperparameters=FineTuneHyperParams(n_epochs=1, batch_size=16, learning_rate_multiplier=1.6),
)

dataset_path = Path("dataset_dumps/train/bct_10k_instruct_10k.jsonl")
file_id = upload_or_reuse_file(dataset_path)
model_id = run_finetune_with_wandb_from_file(
    params=params,
    file_path=dataset_path,
    project_name="bct-dump-training",
    notes=f"Fine-tuning {args.model} with custom datasets",
    ask_to_validate_training=True,
)

print(f"Fine-tuned model ID: {model_id}")
