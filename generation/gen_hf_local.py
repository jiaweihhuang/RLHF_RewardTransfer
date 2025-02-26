import json
from dataclasses import dataclass, field, fields
from typing import List, Optional
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
import os
import torch
from accelerate import Accelerator
import torch.distributed as dist
import random

tqdm.pandas()

def seed_everything(seed: int = 42):
    print('seed_everything with {}'.format(seed))
    random.seed(seed + 1)
    np.random.seed(seed + 20)
    os.environ["PYTHONHASHSEED"] = str(seed + 300)
    torch.manual_seed(seed + 4000)


'''
Data format:

{
    # existing elements during the generation
    "prompt": ...,
    "responses": ...,
    "responses_for_source_rm": ...,
    "summary": ...,         # the target summary, used to get the rouge score.
}
'''


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    url: Optional[str] = field(
        default="http://localhost",
        metadata={"help": "url of the model response"},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "the tokenizer to use"},
    )

    pretrained_model: Optional[str] = field(
        default=None,
        metadata={"help": "the pretrained_model to use"},
    )
    
    ports: List[str] = field(default_factory=lambda: ["3000"], metadata={"help": "ports of the model response"})
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})
    dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the output file"},
    )
    bos_format: Optional[str] = field(
        default="",
        metadata={"help": "the format of the beginning of the sentence"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    N: Optional[int] = field(
        default=0,
        metadata={"help": "Best of N to approximate the optimal policy regarding the source reward models"},
    )
    max_input_length: Optional[int] = field(
        default=10000,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_prompt_length: Optional[int] = field(
        default=8000,
        metadata={"help": "the maximum string length of the prompt"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    max_workers: Optional[int] = field(
        default=1024,
        metadata={"help": "the number of workers"},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "inference batch size"},
    )
    train_data_size: Optional[int] = field(
        default=10000,
        metadata={"help": "training data size; -1 means using all training data"},
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={"help": "the data type for loading model"},
    )
    set_eval: Optional[bool] = field(
        default=True, metadata={"help": "whether to set model.eval when doing inference"}
    )


accelerator = Accelerator()
device = accelerator.device

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ds_dir = script_args.dataset_name_or_path
output_dir = script_args.output_dir
K = script_args.K + script_args.N
ports = script_args.ports
temperature = script_args.temperature
batch_size = script_args.batch_size
train_data_size = script_args.train_data_size


local_rank = Accelerator().local_process_index
world_size = int(os.getenv("WORLD_SIZE", "1"))

seed_everything(script_args.seed + local_rank * 6666)

tokenizer_path = script_args.tokenizer

if script_args.dtype is not None:
    if script_args.dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif script_args.dtype == 'float16':
        torch_dtype = torch.float16
    elif script_args.dtype == 'float32':
        torch_dtype = torch.float32
    else:
        raise NotImplementedError
else:
    torch_type = torch.float32
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, device_map='cuda', torch_dtype=torch_dtype)
model = T5ForConditionalGeneration.from_pretrained(script_args.pretrained_model, device_map='cuda', torch_dtype=torch_dtype)

if script_args.set_eval:
    model.eval()

print('Load tokenizer from ', tokenizer_path)
print('Load model from ', script_args.pretrained_model)
print('model.dtype is ', model.dtype)

eos_token_id = [tokenizer.eos_token_id] + script_args.eos_ids


def batch_query_model(prompt_list, num_return_sequences, temp):
    tokenized_dataset = tokenizer(prompt_list, truncation=True, padding='max_length', max_length=1000, return_tensors='pt')
    
    source_ids = tokenized_dataset['input_ids'].to(device)
    source_mask = tokenized_dataset['attention_mask'].to(device)

    # *multinomial sampling* if `num_beams=1` and `do_sample=True`
    output = model.generate(input_ids=source_ids, attention_mask=source_mask, max_length=256, num_return_sequences=num_return_sequences, do_sample=True, eos_token_id=eos_token_id, temperature=temp, num_beams=1)

    response = tokenizer.batch_decode(output, skip_special_tokens=True)
    return response


ds = load_dataset(ds_dir, split='train')
data_size = len(ds["document"])
print('Dataset size is ', data_size)

if data_size % world_size == 0:
    share = data_size // world_size
else:
    share = data_size // world_size + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

train_data_size_per_device = train_data_size // world_size
if local_rank == world_size - 1:
    train_data_size_per_device = train_data_size - train_data_size_per_device * (world_size - 1)
print('My dataset size is {}. I will select {} from them.'.format(len(ds), train_data_size_per_device))
random_indices = np.random.choice(len(ds), train_data_size_per_device, replace=False)
ds = ds.select(indices=random_indices)
    
ds = ds.map(
    lambda x: {
        "prompt": "Summarize: {}".format(x[script_args.dataset_key])
    }
)
dataset_size = len(ds)
print("My id is {}. I'm going to generate responses for {} data on device {}".format(local_rank, len(ds), device))


data = []
with torch.no_grad():
    for i in tqdm(range(dataset_size // batch_size)):
        prompt_list = []
        prompt_index_list = []
        start = i * batch_size

        # filter out those data with lengthy prompt
        for j in range(batch_size):
            if start + j >= len(ds):
                break
            if len(ds[start + j]["prompt"]) > script_args.max_prompt_length:
                print('This prompt length is {}, which is too long, so we skip it.'.format(len(ds[start + j]["prompt"])))
                continue
            prompt_list.append(ds[start + j]["prompt"])
            prompt_index_list.append(start + j)
        if len(prompt_list) == 0:
            continue


        responses_by_index = {}
        for j in range(len(prompt_list)):
            responses_by_index[j] = []

        responses = batch_query_model(prompt_list, K, temperature)
        assert K * len(prompt_list) == len(responses)
        for j in range(len(prompt_list)):
            responses_by_index[j] += responses[j * K: (j+1) * K]

        for j in range(len(prompt_list)):
            prompt_index = prompt_index_list[j]
            assert K == len(responses_by_index[j])
            data.append(
                {"prompt": ds[prompt_index]["prompt"], "responses": responses_by_index[j][:script_args.K], "responses_for_source_rm": responses_by_index[j][script_args.K:], "summary": ds[prompt_index]["summary"]}
            )
            

'''
Collect data from each process
'''
all_process_list = [{}] * world_size

if world_size > 1:
    dist.all_gather_object(all_process_list, data)
else:
    all_process_list = [data]

print('My worker id is {}. I collect {} samples'.format(local_rank, len(data)))

if local_rank == 0:
    gathered_data = []
    for i in range(world_size):
        gathered_data += all_process_list[i]

    output_eval_dataset = {}
    output_eval_dataset["type"] = "text_only"
    output_eval_dataset["instances"] = gathered_data
    print("I collect ", len(gathered_data), "samples")

    with open(output_dir, "w", encoding="utf8") as f:
        json.dump(output_eval_dataset, f, ensure_ascii=False)