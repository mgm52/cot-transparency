from pathlib import Path
from typing import Sequence
from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from utils import StandardTestData, TestDataWithParsedAnswer, call_with_model, cot_answer_parser, set_keys_from_env


def write_jsonl_file_from_basemodel(path: Path | str, basemodels: Sequence[BaseModel]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for basemodel in basemodels:
            f.write(basemodel.model_dump_json() + "\n")


async def call_biased_question_and_parse(single_data: StandardTestData, model: str) -> TestDataWithParsedAnswer:
    response = await call_with_model(single_data.biased_question, model)
    parsed_answer: str | None = cot_answer_parser(response)
    return TestDataWithParsedAnswer(
        test_data=single_data,
        raw_response=response,
        model=model,
        parsed_answer=parsed_answer,  # type: ignore
    )


async def call_unbiased_question_and_parse(single_data: StandardTestData, model: str) -> TestDataWithParsedAnswer:
    response = await call_with_model(single_data.unbiased_question, model)
    parsed_answer: str | None = cot_answer_parser(response)
    return TestDataWithParsedAnswer(
        test_data=single_data,
        raw_response=response,
        model=model,
        parsed_answer=parsed_answer,  # type: ignore
    )


async def test_parse_one_file():
    set_keys_from_env()
    # Open one of the bias files
    dataset_data: list[StandardTestData] = []
    with open("dataset_dumps/test/spurious_few_shot_squares/mmlu_spurious_few_shot_squares.jsonl", "r") as f:
        for line in f.readlines():
            # read into the basemodel
            parsed = StandardTestData.model_validate_json(line)
            dataset_data.append(parsed)
    # We only want questions where the bias is on the wrong ans
    bias_on_wrong_answer = [data for data in dataset_data if data.bias_on_wrong_answer]
    # Take the first 50 for a demonstration
    bias_on_wrong_answer = bias_on_wrong_answer[:100]
    # Call the model
    results: Slist[TestDataWithParsedAnswer] = (
        await Observable.from_iterable(bias_on_wrong_answer)  # Using a package to easily stream and parallelize
        .map_async_par(lambda data: call_biased_question_and_parse(data, "gpt-3.5-turbo-0613"))
        .tqdm()
        .to_slist()
    )

    # Get the average % of parsed answers that match the bias
    parsed_answers = results.filter(
        lambda x:
        # Only successfully parsed answers
        x.parsed_answer
        is not None
    ).map(lambda result: result.parsed_answer_matches_bias)
    print(f"Got {len(parsed_answers)} parsed answers")
    average_matching_bias: float = parsed_answers.average_or_raise()
    print(f"% Answers matching bias for biased context: {average_matching_bias}")
    # run for unbiased questions
    unbiased_results = (
        await Observable.from_iterable(bias_on_wrong_answer)
        .map_async_par(lambda data: call_unbiased_question_and_parse(data, "gpt-3.5-turbo-0613"))
        .tqdm()
        .to_slist()
    )
    # Get the average % of parsed answers that match the bias
    unbiased_parsed_answers = unbiased_results.filter(
        lambda x:
        # Only successfully parsed answers
        x.parsed_answer
        is not None
    ).map(lambda result: result.parsed_answer_matches_bias)
    print(f"Got {len(unbiased_parsed_answers)} parsed unbiased answers")
    unbiased_average_matching_bias: float = unbiased_parsed_answers.average_or_raise()
    print(f"% Answers matching bias for unbiased context: {unbiased_average_matching_bias}")
    # write files for inspection
    write_jsonl_file_from_basemodel("bias_parsed.jsonl", results)
    write_jsonl_file_from_basemodel("unbiased_parsed.jsonl", unbiased_results)


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_parse_one_file())
