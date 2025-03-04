import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from src.rl_dataset import RLHFDataset


def seed_all(seed, deterministic_algos=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if deterministic_algos:
        torch.use_deterministic_algorithms(True)


def load_model(model_path, device="cuda"):
    # assert torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    actor_model_config._attn_implementation = "eager"
    if device == "cuda":
        with torch.device("cuda"):
            actor_model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
    else:
        actor_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
    actor_model.tokenizer = tokenizer
    return actor_model


def _collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


def get_dataloader(data_path, batch_size, max_prompt_length, valid_size, tokenizer):

    data = pd.read_parquet(data_path)
    dataset = RLHFDataset(
        data_path,
        tokenizer,
        prompt_key="prompt",
        max_prompt_length=max_prompt_length,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        return_raw_chat=False,
        truncation="error",
    )

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - valid_size, valid_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    return train_loader, valid_loader
