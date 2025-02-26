import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import random
from datasets import Dataset, load_dataset
from dpo import PreferenceTrainer
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from transformers import T5Tokenizer, T5ForConditionalGeneration
from accelerate import Accelerator


def seed_everything(seed: int = 42):
    random.seed(seed + 1)
    np.random.seed(seed + 20)
    os.environ["PYTHONHASHSEED"] = str(seed + 300)
    torch.manual_seed(seed + 4000)


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters, i.e., the KL penalty in the paper
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the model name or path"},
    )
    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    tokenizer: Optional[str] = field(
        default="",
        metadata={"help": "the location of the tokenizer or path"},
    )
    train_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the dataset name or path"},
    )
    eval_dir: Optional[str] = field(
        default=None,  
        metadata={"help": "the location of the evalset name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(
        default="constant_with_warmup", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    total_train_batch_size: Optional[int] = field(default=128, metadata={"help": "train batch size per device"})

    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    eos_padding: Optional[bool] = field(default=False, metadata={"help": "whether to pad with eos token"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "the margin scale"})

    max_prompt_length: Optional[int] = field(default=1000, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=20, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=2, metadata={"help": "the logging frequency"})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "the saving strategy"})
    save_steps: Optional[int] = field(default=50000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})
    run_name: Optional[str] = field(default="dpo_soft", metadata={"help": "the run name"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "the loss type"})
    output_dir: Optional[str] = field(default="./dpo_soft", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    choose_type: Optional[str] = field(default="max_min", metadata={"help": "the choose type"})

    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "the end of text token"})
    mask_prompt: Optional[bool] = field(default=False, metadata={"help": "mask prompt"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "the length penalty"})

    '''
    Additional hyper-parameters
    '''
    algorithm: Optional[str] = field(
        default="DPO",
        metadata={"help": "the algorithm to be used"},
    )
    iteration: Optional[int] = field(default=0, metadata={"help": "the current iteration number"})

    is_encoder_decoder: Optional[bool] = field(default=False, metadata={"help": "is encoder_decoder architecture?"})

    disable_dropout: Optional[bool] = field(default=True, metadata={"help": "disable dropout?"})

    use_deepspeed: Optional[bool] = field(default=True, metadata={"help": "use deep speed or not"})

    train_data_size: Optional[int] = field(default=-1, metadata={"help": "the dataset size for training; by default, -1 means use all the dataset"})

    dtype: Optional[str] = field(
        default=None,
        metadata={"help": "the data type for loading model"},
    )

    alpha: Optional[float] = field(default=1e-2, metadata={"help": "the choice of alpha"})

    deterministic: Optional[bool] = field(default=True, metadata={"help": "how to choose preferred response"})
    reward_scale: Optional[float] = field(default=1.0, metadata={"help": "the scaling of the reward"})

    K: Optional[int] = field(default=8, metadata={"help": "the number of responses to use"})
    N: Optional[int] = field(default=-1, metadata={"help": "the number of transfer responses to use"})

    seed: Optional[int] = field(default=42, metadata={"help": "the random seed"})


def prepare_data(
    data_dir: str = None,
    sanity_check: bool = False,
    margin_scale=1,
    choose_type="random",
    eot_token="",
    length_penalty=0,
    train_data_size=-1,
    deterministic=True,
    reward_scale=1.0,
    K=8,
) -> Dataset:
    """Prepare the dataset for DPO training by rejection sampling.
    We implement different strategies to select pairs, including
    max_min: best v.s. worst
    max_random: best v.s. random from the remaining;
    max_max: best v.s. second best
    max_min_p: best v.s. worst but we additionally add a length penalty in the reward value

    deterministic: 
        if True, choose the one with higher reward to be the preferred one;
        otherwise, sample preference following BT model.
    """
    ds = load_dataset("json", data_files=data_dir, split="train", field="instances")

    if sanity_check:
        sample_size = min(len(ds), 100)
    else:
        sample_size = train_data_size

    if sample_size == -1:
        pass
    elif sample_size >= len(ds):
        print('The train_data_size={} is larger than the dataset_size{}, will use all the data for training'.format(sample_size, len(ds)))
    else:
        # otherwise, select "train_data_size" samples from ds
        assert sample_size > 0 and sample_size < len(ds)
        random_indices = np.random.choice(len(ds), sample_size, replace=False)
        ds = ds.select(indices=random_indices)
        print('Select {} samples from the original dataset'.format(sample_size))

    pos = []
    neg = []
    prompts = []

    margin = []
    for sample in ds:
        
        sample["rewards"] = sample["rewards"][:K]
        sample["responses"] = sample["responses"][:K]

        if choose_type == "random":
            idx0 = 0
            idx1 = 1
        elif choose_type == "rand_rand":
            if sample["rewards"][0] > sample["rewards"][1]:
                idx0 = 0
                idx1 = 1
            else:
                idx0 = 1
                idx1 = 0
        elif choose_type == 'max_mid':
            # max_mid: pick the best one and the median one.
            sorted_indices = np.argsort(sample["rewards"])
            idx0 = sorted_indices[-1]
            idx1 = sorted_indices[len(sorted_indices) // 2]
        elif choose_type == "max_random":
            idx0 = np.argmax(sample["rewards"])
            if idx0 == 0:
                idx1 = 1
            else:
                idx1 = 0
        elif choose_type == "max_min":
            idx0 = np.argmax(sample["rewards"])
            idx1 = np.argmin(sample["rewards"])
        elif choose_type == "max_max":
            sorted_indices = np.argsort(sample["rewards"])
            idx0 = sorted_indices[-1]
            idx1 = sorted_indices[-2]
        elif choose_type == "max_min_p":
            r = [
                sample["rewards"][i] - length_penalty * len(sample["responses"][i])
                for i in range(len(sample["rewards"]))
            ]
            idx0 = np.argmax(r)
            idx1 = np.argmin(r)
        else:
            raise NotImplementedError
        
        if not deterministic:
            r0 = sample["rewards"][idx0]
            r1 = sample["rewards"][idx1]

            prob_r0_over_r1 = 1. / (1. + np.exp((r1 - r0) * reward_scale))
            if np.random.rand() > prob_r0_over_r1:
                idx0, idx1 = idx1, idx0
                
        if type(idx0) == np.ndarray or type(idx0) == list:
            assert 0 == 1, 'This should not be the case'
            assert len(idx0) == len(idx1)
            for i in range(len(idx0)):
                prompts.append(sample["prompt"])
                pos.append(sample["responses"][idx0[i]] + eot_token)
                neg.append(sample["responses"][idx1[i]] + eot_token)
                margin.append((sample["rewards"][idx0[i]] - sample["rewards"][idx1[i]]) * margin_scale)
        else:
            if sample["rewards"][idx0] > sample["rewards"][idx1]:
                prompts.append(sample["prompt"])
                pos.append(sample["responses"][idx0] + eot_token)
                neg.append(sample["responses"][idx1] + eot_token)
                margin.append((sample["rewards"][idx0] - sample["rewards"][idx1]) * margin_scale)
            elif sample["rewards"][idx0] < sample["rewards"][idx1]:
                prompts.append(sample["prompt"])
                pos.append(sample["responses"][idx1] + eot_token)
                neg.append(sample["responses"][idx0] + eot_token)
                margin.append((-sample["rewards"][idx0] + sample["rewards"][idx1]) * margin_scale)

    '''
        The size of the following dataset can be smaller than the previous ones.
        Because we do not append the prompts if the preferred and not preferred responses have the same reward.
    '''
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin})

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    print('world_size is ', world_size)
    local_rank = Accelerator().local_process_index
    print('local rank is ', local_rank)

    seed_everything(script_args.seed + local_rank * 8888)

    is_deepspeed_enabled = script_args.use_deepspeed
    assert is_deepspeed_enabled is False, 'For now, we do not use deepspeed'

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
        torch_dtype = torch.float32

    # 1. load a pretrained model
    assert 't5' in script_args.model_name_or_path or 'T5' in script_args.model_name_or_path, 'Only for T5 Models'
    model = T5ForConditionalGeneration.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = script_args.model_name_or_path

    assert 't5' in ref_name or 'T5' in ref_name, 'Only for T5 Models'
    model_ref = T5ForConditionalGeneration.from_pretrained(
        ref_name,
        torch_dtype=torch_dtype,
    )

    assert 't5' in script_args.tokenizer or 'T5' in script_args.tokenizer, 'Only for T5 Tokenizers'
    tokenizer = T5Tokenizer.from_pretrained(script_args.tokenizer, 
        torch_dtype=torch_dtype)


    # 2. Load the Stack-exchange paired dataset
    train_dataset = prepare_data(
        data_dir=script_args.train_dir,
        margin_scale=script_args.margin_scale,
        sanity_check=script_args.sanity_check,
        choose_type=script_args.choose_type,
        eot_token=script_args.eot_token,
        length_penalty=script_args.len_penalty,
        train_data_size=script_args.train_data_size,
        deterministic=script_args.deterministic,
        reward_scale=script_args.reward_scale,
        K=script_args.K,
    )

    assert script_args.max_training_samples == -1
    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    # 3. Load evaluation dataset
    eval_dataset = prepare_data(
        data_dir=script_args.eval_dir,
        sanity_check=True,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
        deterministic=script_args.deterministic,
        reward_scale=script_args.reward_scale,
        K=script_args.K,
    )

    # 4. initialize training arguments:
    total_train_batch_size = script_args.total_train_batch_size
    
    gradient_accumulation_steps = total_train_batch_size // world_size
    assert total_train_batch_size % world_size == 0

    assert script_args.dtype == 'bfloat16'
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs=script_args.num_train_epochs,
        save_strategy=script_args.save_strategy,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        # evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        # report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        # optim=script_args.optimizer_type,
        bf16=True if script_args.dtype == 'bfloat16' else False,
        remove_unused_columns=False,
        run_name=script_args.run_name,
    )

    # 5. initialize the DPO trainer
    if script_args.algorithm == 'DPO':
        trainer = PreferenceTrainer(
            model,
            model_ref,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            loss_type=script_args.loss_type,
            max_target_length=256,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
            mask_prompt=script_args.mask_prompt,
            len_penalty=script_args.len_penalty,
            is_encoder_decoder=script_args.is_encoder_decoder,
            disable_dropout=script_args.disable_dropout,
        )
    else:
        raise NotImplementedError


    # 6. train
    trainer.train()
    trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)