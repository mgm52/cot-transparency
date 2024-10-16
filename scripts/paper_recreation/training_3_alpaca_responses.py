import asyncio
import argparse
from pathlib import Path

from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from tqdm import tqdm

from cot_transparency.apis import OpenAIChatCaller, InferenceResponse
from cot_transparency.apis.base import ModelCaller, Prompt
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.apis.openai.formatting import append_assistant_preferred_to_next_message
from cot_transparency.data_models.config import OpenaiInferenceConfig

# from cot_transparency.data_models.data.gpt_35_instructions import gpt_35_instruct_path
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from scripts.load_alpaca_dataset import get_alpaca_training


def call_and_create_sample(
    prompt: Prompt,
    vanilla_caller: ModelCaller,
    config: OpenaiInferenceConfig,
) -> FinetuneSample:
    vanilla_response: InferenceResponse = vanilla_caller.call(messages=prompt.messages, config=config)
    first = vanilla_response.single_response
    new_prompt: Prompt = prompt + Prompt(messages=[ChatMessage(role=MessageRole.assistant, content=first)])
    to_sample = append_assistant_preferred_to_next_message(new_prompt.messages)
    return FinetuneSample(messages=to_sample)


def finetune_sample_to_prompt(sample: FinetuneSample) -> Prompt:
    messages = [m.to_chat_message() for m in sample.messages]
    # the last message is the one we want to predict
    messages_without_last = messages[:-1]
    return Prompt(messages=messages_without_last)


class PromptWithModel(BaseModel):
    prompt: Prompt
    config: OpenaiInferenceConfig


async def generate_instruction(model_name: str):
    # Originally, authors ran 100k alpaca completions, then sampled down to 10k in next stage... but that's expensive!
    samples: Slist[FinetuneSample] = get_alpaca_training(15000)
    print(f"Total testing samples: {len(samples)}")

    vanilla_caller = OpenAIChatCaller().with_file_cache(
        Path("experiments/alignment_tax/vanilla_completion.jsonl"), write_every_n=100
    )
    vanilla_config = OpenaiInferenceConfig(model=model_name, max_tokens=2000, temperature=1.0, top_p=1.0)
    prompts: Slist[Prompt] = samples.map(finetune_sample_to_prompt)
    instruct_path = Path(f"data/instructions/{model_name}_temp_1.jsonl")

    # Clear the existing file content before appending new data
    with open(instruct_path, "w") as f:
        f.write("")

    await (
        Observable.from_iterable(prompts)
        .map_blocking_par(
            lambda p: call_and_create_sample(
                prompt=p,
                vanilla_caller=vanilla_caller,
                config=vanilla_config,
            )
        )
        .tqdm(tqdm(total=prompts.length))
        .to_file_appending(
            file_path=instruct_path,
            serialize=lambda x: x.model_dump_json(),
        )
    )
    vanilla_caller.save_cache()


# Writes to data/instructions/{model_name}_temp_1.jsonl
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate instructions using a specified model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Name of the model to use (e.g. gpt-3.5-turbo-0613).",
    )
    args = parser.parse_args()

    asyncio.run(generate_instruction(model_name=args.model_name))
