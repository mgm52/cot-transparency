import cot_transparency.data_models.data.mmlu as mmlu
from typing import Optional
import argparse
import asyncio
import re
from tqdm.asyncio import tqdm
import openai  # openai>=0.27.9,<1.0.0
from pydantic import BaseModel
from string import ascii_uppercase
from dotenv import load_dotenv
import os


### Copied from dataset_dumps to keep it standalone ###
class TestChatMessage(BaseModel):
    role: str
    content: str


BREAK_WORDS: list[str] = [
    "answer is (",
    "answer is  (",
    "answer is: (",
    "answer is:(",
    "answer is:  (",
    "answer is:\n(",
    "answer is: \n(",
    "answer is:\n\n(",
    "answer is: ",
    "option is: ",
    "option is ",
    "answer is ",
    "answer is $\\boxed{\\text{(",
    "answer is: $\\boxed{\\text{(",
    "choices is: " r"is: $\boxed{\textbf{(",
    "answer: ",
    "answer is ",
    r"answer is: \[\boxed{\text{",
    r"is: $\boxed{\textbf{(",
    "choices is: ",
    r"is: $\boxed{\textbf{(",
    r"is $\boxed{\textbf{(",
    r"is $\boxed{\text{(",
    r"is: \boxed{\text{(",
    r"is: $\boxed{\text{(",
    r"is: (\boxed{\text{(",
    "accurate answer would be",
    "is: $\boxed{\textbf{(",
]


def cot_answer_parser(model_answer: str) -> str | None:
    # This is a very simple parser that looks for the first instance of a letter in the answer
    for break_word in BREAK_WORDS:
        if break_word not in model_answer:
            continue
        tmp = model_answer.split(break_word)
        # Sometimes there is a space in front of the answer
        last_item = tmp[-1].lstrip()

        if not last_item:
            continue

        # also add lowercase variants
        possible_indicators = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        possible_indicators_lower = [indicator.lower() for indicator in possible_indicators]
        possible_indicators_re = "|".join(possible_indicators + possible_indicators_lower)

        pattern = rf"^(?:[Oo]ption |[Ss]tatement )?\(?({possible_indicators_re})\)?(\s|\)|\.|$)+.*$"

        match = re.search(pattern, last_item)
        if match:
            candidate_ans = match.group(1)
            if candidate_ans in possible_indicators:
                idx = possible_indicators.index(candidate_ans)
                return ascii_uppercase[idx]
            elif candidate_ans in possible_indicators_lower:
                idx = possible_indicators_lower.index(candidate_ans)
                return ascii_uppercase[idx]

        return None


async def call_with_model(messages: list[TestChatMessage], model: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=[msg.model_dump() if isinstance(msg, TestChatMessage) else msg for msg in messages],
        temperature=0,
        max_tokens=1000,
        stream=False,
    )
    first_response: str = response.choices[0].message.content  # type: ignore
    assert isinstance(first_response, str)
    return first_response


def set_keys_from_env():
    # take environment variables from .env so you don't have
    # to source .env in your shell
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key


### End of copy from dataset_dumps ###


# Construct a multiple-choice question prompt
def construct_prompt(sample: mmlu.MMLUExample):
    question = sample.question
    options = sample.options
    prompt = (
        "Please answer the following multiple-choice question. Give your answer immediately, in the format 'The best answer is: (X)'\n\n"
        f"Question:\n{question}\n\nOptions:\n"
    )
    for idx, option in enumerate(options):
        option_letter = chr(ord("A") + idx)
        prompt += f"{option_letter}. {option}\n"
    prompt += "\nAnswer:"
    return prompt


async def evaluate_model(model_name: str, num_samples: Optional[int] = None):
    # Load MMLU test samples
    samples = mmlu.test(questions_per_task=None)
    if num_samples is not None and num_samples < len(samples):
        samples = samples.sample(n=num_samples, seed="101")

    correct = 0
    total = 0
    tasks = []

    # Prepare tasks
    for sample in samples:
        prompt = construct_prompt(sample)
        tasks.append(call_with_model([TestChatMessage(role="user", content=prompt)], model_name))

    # Run tasks
    with tqdm(total=len(tasks)) as pbar:
        responses = await asyncio.gather(*tasks)
        pbar.update(len(tasks))

    # Evaluate the results!
    for idx, (sample, response) in enumerate(zip(samples, responses)):
        predicted_answer = cot_answer_parser(response)

        print(f"Question {idx + 1}:")
        print(construct_prompt(sample))
        print(f"Model's raw answer: '{response}'")
        print(f"Model's parsed answer: {predicted_answer}")
        print(f"Correct answer: {sample.correct_ans_letter}\n")

        if predicted_answer == sample.correct_ans_letter:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Model {model_name} accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpenAI model on MMLU samples.")
    parser.add_argument(
        "--model_name", type=str, default="gpt-4o-mini-2024-07-18", help="Name of the OpenAI model to evaluate."
    )
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate.")
    args = parser.parse_args()

    set_keys_from_env()  # Set API keys from environment

    asyncio.run(evaluate_model(args.model_name, args.num_samples))
