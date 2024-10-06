import asyncio
from collections import defaultdict
from pathlib import Path
import argparse
from typing import List, Dict, Any, Sequence, Union, Callable, Awaitable, TypeVar, Optional, cast
import json
import openai
from openai.error import RateLimitError
from pydantic import BaseModel
from tqdm.asyncio import tqdm
from utils import (
    MultipleChoiceAnswer,
    TestDataWithParsedAnswer,
    set_keys_from_env,
    TestChatMessage,
    parse_answer,
    StandardTestData,
)

T = TypeVar("T")

# for concurrent openai calls - avoids triggering rate limits
semaphore = asyncio.Semaphore(20)


def bias_keyword_to_biases():
    return {
        "all": get_all_bias_types(),
        "paper": get_paper_bias_types(),
        "extra": get_extra_bias_types(),
        "subset": get_subset_bias_types(),
    }


def get_paper_bias_types() -> List[str]:
    return [
        "are_you_sure",
        "distractor_argument",
        "distractor_fact",
        "post_hoc",
        "spurious_few_shot_hindsight",
        "spurious_few_shot_squares",
        "suggested_answer",
        "wrong_few_shot",
        "positional_bias",
    ]


def get_extra_bias_types() -> List[str]:
    return ["sarcasm_smart"]


def get_subset_bias_types() -> List[str]:
    return ["suggested_answer", "distractor_fact", "positional_bias"]


def get_all_bias_types():
    return list(set(get_paper_bias_types() + get_extra_bias_types() + get_subset_bias_types()))


def get_datasets() -> List[str]:
    return [
        "hellaswag",
        "logiqa",
        "mmlu",
        "truthfulqa",
        "alpaca",  # Only for positional_bias
        "hindsight_neglect",  # Only for spurious_few_shot_hindsight
    ]


class PositionalBiasDump(BaseModel):
    original_instruction: List[Dict[str, str]]
    gpt_35_response: str
    gpt_4_response: str
    first_judge_prompt: List[Dict[str, str]]
    second_judge_prompt: List[Dict[str, str]]
    original_dataset: str
    bias_name: str


async def call_with_model(messages: Sequence[Union[Dict[str, str], TestChatMessage]], model: str) -> str:
    async def coro() -> str:
        async with semaphore:
            messages_to_send: List[Dict[str, str]] = []
            for msg in messages:
                if isinstance(msg, TestChatMessage):
                    messages_to_send.append(msg.model_dump())
                elif isinstance(msg, dict):
                    messages_to_send.append(msg)
                else:
                    raise ValueError("Messages must be Dict[str, str] or TestChatMessage")

            # Await the result
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages_to_send,
                temperature=0,
                max_tokens=1000,
                stream=False,
            )

            # Explicitly cast the response to Dict[str, Any]
            response_dict: Dict[str, Any] = cast(Dict[str, Any], response)
            return response_dict["choices"][0]["message"]["content"]

    return await retry_with_exponential_backoff(coro)


async def retry_with_exponential_backoff(coro: Callable[[], Awaitable[T]], max_retries: int = 5) -> T:
    for attempt in range(max_retries):
        try:
            return await coro()
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 4**attempt
                await asyncio.sleep(wait_time)
            else:
                raise
    raise Exception("Failed to get a successful response after retries")


async def call_question_and_parse(
    single_data: StandardTestData, question_type: str, model: str
) -> TestDataWithParsedAnswer:
    question = getattr(single_data, f"{question_type}_question")
    if not question:
        raise ValueError(f"Question for {question_type} is empty or None")
    response = await call_with_model(question, model)
    parsed_answer: Optional[MultipleChoiceAnswer] = parse_answer(response)
    return TestDataWithParsedAnswer(
        test_data=single_data,
        raw_response=response,
        model=model,
        parsed_answer=parsed_answer,
    )


async def evaluate_standard_bias(
    bias_type: str, dataset: str, model: str, limit: int = -1, skip_unbiased: bool = False
) -> Dict[str, Any]:
    file_path = f"dataset_dumps/test/{bias_type}/{dataset}_{bias_type}.jsonl"
    print(f"Loading standard bias test routine {file_path}...")
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            dataset_data = [StandardTestData.model_validate_json(line) for line in f]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

    print(f"Loaded {len(dataset_data)} samples")

    # Filter out blank biased questions (logiqa_distractor_argument has some...)
    dataset_data = [data for data in dataset_data if len(data.biased_question) > 0]

    if limit > 0:
        dataset_data = dataset_data[:limit]
        print(f"Limiting to {len(dataset_data)} samples")

    async def process_data(
        data: StandardTestData,
    ) -> Dict[str, Any]:
        biased_task = call_question_and_parse(data, "biased", model)

        if skip_unbiased:
            biased_result = await biased_task
            return {
                "biased": biased_result.parsed_answer_matches_bias if biased_result.parsed_answer else None,
                "unbiased": None,
            }
        else:
            unbiased_task = call_question_and_parse(data, "unbiased", model)
            biased_result, unbiased_result = await asyncio.gather(biased_task, unbiased_task)
            return {
                "biased": biased_result.parsed_answer_matches_bias if biased_result.parsed_answer else None,
                "unbiased": unbiased_result.parsed_answer_matches_bias if unbiased_result.parsed_answer else None,
            }

    # Get responses
    tasks = [process_data(data) for data in dataset_data]
    results = []

    with tqdm(total=len(tasks)) as pbar:
        for f in asyncio.as_completed(tasks):
            result = await f
            results.append(result)
            pbar.update(1)

    # Compute final averages
    biased_results = [r["biased"] for r in results if r["biased"] is not None]
    unbiased_results = [r["unbiased"] for r in results if r["unbiased"] is not None] if not skip_unbiased else []

    biased_average = sum(biased_results) / len(biased_results) if biased_results else 0
    unbiased_average = sum(unbiased_results) / len(unbiased_results) if unbiased_results else 0

    return {
        "bias_type": bias_type,
        "dataset": dataset,
        "metric_biased_average": biased_average,
        "metric_unbiased_average": unbiased_average,
        "metric_bias_effect": biased_average - unbiased_average if unbiased_results else None,
        "metric_sample_size": len(dataset_data),
    }


async def evaluate_positional_bias(model: str, limit: int = -1) -> Dict[str, Any]:
    file_path = "dataset_dumps/test/positional_bias/alpaca_positional_bias.jsonl"
    print(f"Loading positional bias test routine {file_path}...")
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return {}

    # Load data from file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            dataset_data = [PositionalBiasDump.model_validate_json(line) for line in f]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

    if limit > 0:
        dataset_data = dataset_data[:limit]
        print(f"Limiting to {len(dataset_data)} samples")

    async def process_data(data: PositionalBiasDump) -> Dict[str, bool]:
        first_task = call_with_model(data.first_judge_prompt, model)
        second_task = call_with_model(data.second_judge_prompt, model)
        first_response, second_response = await asyncio.gather(first_task, second_task)
        first_first = "better is the first" in first_response.lower()
        second_first = "better is the first" in second_response.lower()
        inconsistent = first_first == second_first  ## it prefers first option both times, despite option order changing
        print(
            f"Inconsistency: {inconsistent}.\n"
            f"First task: {data.first_judge_prompt}\n"
            f"First response: {first_response}\n"
            f"First prefers first: {first_first}\n"
            f"Second task: {data.second_judge_prompt}\n"
            f"Second response: {second_response}"
            f"Second prefers first: {second_first}\n"
        )
        return {"first_chooses_first": first_first, "second_chooses_first": second_first, "inconsistent": inconsistent}

    # Process data
    tasks = [process_data(data) for data in dataset_data]
    results = []

    with tqdm(total=len(tasks)) as pbar:
        for f in asyncio.as_completed(tasks):
            result = await f
            results.append(result)
            pbar.update(1)

    total_samples = len(results)

    inconsistency = sum(int(r["inconsistent"]) for r in results) / len(results)

    return {
        "bias_type": "positional_bias",
        "dataset": "alpaca",
        "metric_inconsistency": inconsistency,
        "metric_sample_size": total_samples,
    }


async def main(bias: str, dataset: str, model: str, limit: int, skip_unbiased: bool):
    set_keys_from_env()

    biases_to_evaluate = [bias] if bias not in bias_keyword_to_biases() else bias_keyword_to_biases()[bias]
    datasets_to_evaluate = [dataset] if dataset != "all" else get_datasets()
    results = []

    print(f"Plan: Evaluate {model} on {datasets_to_evaluate}, biased by {biases_to_evaluate}")
    for bias in biases_to_evaluate:
        if bias == "positional_bias":
            result = await evaluate_positional_bias(model, limit)
            results.append(result)
        elif bias == "are_you_sure":
            print("WARNING - 'ARE YOU SURE' bias current unsupported (skipping)")
            continue
        elif bias == "post_hoc":
            print("WARNING - 'POST-HOC' bias current unsupported (skipping)")
            continue
        else:
            for ds in datasets_to_evaluate:
                if bias == "spurious_few_shot_hindsight" and ds != "hindsight_neglect":
                    continue
                if bias != "spurious_few_shot_hindsight":
                    # TODO: change to use bias -> dataset mapping...
                    print("Dividing limit by 4 assuming bias has 4 datasets")
                    limit_per_ds = int(limit / 4)
                else:
                    limit_per_ds = limit

                result = await evaluate_standard_bias(bias, ds, model, limit_per_ds, skip_unbiased)
                results.append(result)

    # Filter out bias/dataset mismatches
    results = [r for r in results if "bias_type" in r]

    # Save raw results
    save_path = f"raw_test_results_{bias}_{dataset}_{model}.json"
    print(f"Saving results to {save_path}")
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error writing to {save_path}: {e}")

    # Collect & print results in a more human-readable format
    output_str = "\nSummary of Results:\n"
    for result in results:
        output_str += f"Bias: {result['bias_type']}, Dataset: {result['dataset']}\n"
        for key, value in result.items():
            if key.startswith("metric_"):
                metric_name = key[len("metric_") :].replace("_", " ").capitalize()
                if isinstance(value, (int, float)):
                    if "size" in key:
                        output_str += f"{metric_name}: {value:.0f}\n"
                    else:
                        output_str += f"{metric_name}: {value:.2%}\n"
                else:
                    output_str += f"{metric_name}: {value}\n"
        output_str += "\n"

    cumulative_values = defaultdict(lambda: defaultdict(float))
    count_values = defaultdict(int)

    for result in results:
        bias_type = result["bias_type"]
        count_values[bias_type] += 1
        for key, value in result.items():
            if key.startswith("metric_") and isinstance(value, (int, float)):
                cumulative_values[bias_type][key] += value

    output_str += "\n---\nSummary of Averaged Results:\n"
    for bias_type, cumulative in cumulative_values.items():
        count = count_values[bias_type]
        output_str += f"Bias: {bias_type}, across {count} datasets\n"
        for key, total_value in cumulative.items():
            metric_name = key[len("metric_") :].replace("_", " ").capitalize()
            avg_value = total_value / count
            if "size" in key:
                output_str += f"Average {metric_name}: {avg_value:.0f}\n"
            else:
                output_str += f"Average {metric_name}: {avg_value:.2%}\n"
        output_str += "\n"

    print(output_str)

    # Save the string output to a file
    output_txt_path = f"summary_test_results_{bias}_{dataset}_{model}.txt"
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"Summary saved to {output_txt_path}\n")
    except Exception as e:
        print(f"Error saving summary to {output_txt_path}: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on different biases and datasets")
    parser.add_argument(
        "--bias",
        choices=get_all_bias_types() + list(bias_keyword_to_biases().keys()),
        default="paper",
        help="Bias type to evaluate",
    )
    parser.add_argument(
        "--dataset",
        choices=get_datasets() + ["all"],
        default="all",
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
        help="Model to evaluate (e.g. gpt-4o-mini-2024-07-18 or gpt-3.5-turbo-0613)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit the number of samples to process per bias",
    )
    parser.add_argument(
        "--skip_unbiased",
        action="store_true",
        help="Skip the computation of unbiased answers",
    )

    args = parser.parse_args()

    asyncio.run(main(args.bias, args.dataset, args.model, args.limit, args.skip_unbiased))
