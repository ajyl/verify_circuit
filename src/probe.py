import os
import re
import json
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
import einops

from src.rl_dataset import RLHFDataset
from src.record_utils import record_activations


def collate_fn(data_list: list[dict]) -> dict:
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


def get_dataloader(config, tokenizer):
    import pandas as pd
    from datasets import Dataset
    from torch.utils.data import DataLoader

    data_path = config["data_path"]
    batch_size = config["batch_size"]
    max_prompt_length = config["max_prompt_length"]
    valid_size = config["valid_size"]
    data = pd.read_parquet(data_path)
    # dataset = Dataset.from_pandas(data)
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
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def save_config(config):
    """Save config"""
    save_path = os.path.join(config["save_path"], "config.json")
    with open(save_path, "w") as f:
        json.dump(config, f)


def unembed(vector, lm_head, k=10):
    dots = einsum("vocab d_model, d_model -> vocab", lm_head, vector)
    top_k = dots.topk(k).indices
    return top_k


def _build_inner_batch(resid_stream, timesteps_not, timesteps_this):
    _resid_streams = []
    _labels = []
    for b_idx in timesteps_not.keys():
        _not_timesteps = timesteps_not[b_idx].tolist()
        _resid_streams.append(
            resid_stream[
                b_idx,
                :,
                _not_timesteps,
            ]
        )

        _labels.extend([0] * len(_not_timesteps))

        _this_timesteps = timesteps_this[b_idx].tolist()
        _resid_streams.append(
            resid_stream[
                b_idx,
                :,
                _this_timesteps,
            ]
        )
        _labels.extend([1] * len(_this_timesteps))

    indices = list(range(len(_labels)))
    random.shuffle(indices)

    _resid_streams = torch.cat(_resid_streams, dim=1)
    assert len(_labels) == _resid_streams.shape[1]

    shuffled_labels = [_labels[idx] for idx in indices]
    shuffled_resid_streams = _resid_streams[:, indices]
    return shuffled_resid_streams, shuffled_labels


def cache_valid_data(model, tokenizer, data_loader, config):
    """
    Cache valid data.
    """
    generation_config = GenerationConfig(do_sample=False)

    max_new_tokens = config["max_response_length"]
    max_prompt_length = config["max_prompt_length"]
    record_module_names = config["record_module_names"]
    log_interval = config["log_interval"]
    eval_interval = config["eval_interval"]
    n_layers = config["n_layers"]
    d_model = config["d_model"]
    probe_timestep = config["probe_timestep"]
    assert probe_timestep in ["last_int", "("]

    all_resid_streams = []
    all_labels = []

    for batch_idx, batch in enumerate(data_loader):

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        with record_activations(model, record_module_names) as recording:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config=generation_config,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )

        # len(recording["model.layers.0"]): max_response_length
        # recording["model.layers.0"][0].shape: [batch, prompt_length, d_model]
        # recording["model.layers.0"][1].shape: [batch, 1, d_model]
        recording = {
            layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording.items()
        }

        # recording["model.layers.0"].shape:
        # [batch, prompt_length + max_new_tokens, d_model]
        seq = output.sequences
        response = seq[:, -max_new_tokens:]
        response_text = tokenizer.batch_decode(response, skip_special_tokens=True)

        # [batch, n_layers, response_length, d_model]
        resid_stream = torch.stack(
            [acts[:, -max_new_tokens:] for acts in recording.values()], dim=1
        )

        if probe_timestep == "(":
            token_open = tokenizer.encode(" (")[0]  # 320
            token_not = tokenizer.encode("not")[0]  # 1921
            token_this = tokenizer.encode("this")[0]  # 574

            mask_not = (response[:, :-1] == token_open) & (response[:, 1:] == token_not)
            mask_this = (response[:, :-1] == token_open) & (
                response[:, 1:] == token_this
            )
            batch_idx_not, timesteps_not = torch.where(mask_not)
            batch_idx_this, timesteps_this = torch.where(mask_this)

            overlap_batches = torch.tensor(
                sorted(
                    list(
                        set(batch_idx_not.tolist()).intersection(
                            set(batch_idx_this.tolist())
                        )
                    )
                )
            ).cuda()
            batch_mask_not = torch.isin(batch_idx_not, overlap_batches)
            batch_mask_this = torch.isin(batch_idx_this, overlap_batches)

            filtered_timesteps_not = {
                b_idx: timesteps_not[(batch_idx_not == b_idx)]
                for b_idx in overlap_batches.tolist()
            }
            filtered_timesteps_this = {
                b_idx: timesteps_this[(batch_idx_this == b_idx)]
                for b_idx in overlap_batches.tolist()
            }

        elif probe_timestep == "last_int":
            filtered_timesteps_not = {}
            filtered_timesteps_this = {}

            for b_idx, text in enumerate(response_text):
                tokens = tokenizer(text, return_offsets_mapping=True)
                tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
                offsets = tokens["offset_mapping"]

                matches = re.findall(r"=\s*(-?\d+)\s*\((not|this works)", text)

                results = []
                for match in matches:
                    number, label = match  # Extract the number and label type

                    # Find number position in text
                    number_index = text.find(number)
                    number_end = number_index + len(number)

                    # Find corresponding token index
                    token_indices = [
                        i
                        for i, (start, end) in enumerate(offsets)
                        if start >= number_index and end <= number_end
                    ]

                    # Store results
                    results.append(
                        {
                            "number": number,
                            "tokens": [tokenized_text[i] for i in token_indices],
                            "token_index": token_indices[-1],
                            "all_token_indices": token_indices,
                            "label": "not" if label == "not" else "this",
                        }
                    )
                not_matches = [x for x in results if x["label"] == "not"]
                this_matches = [x for x in results if x["label"] == "this"]

                if len(not_matches) > 0 and len(this_matches) > 0:
                    filtered_timesteps_not[b_idx] = torch.tensor(
                        [r["token_index"] for r in not_matches]
                    )
                    filtered_timesteps_this[b_idx] = torch.tensor(
                        [r["token_index"] for r in this_matches]
                    )

        # resid_stream: [batch, n_layers, response_length, d_model]
        resid_stream_sample, labels = _build_inner_batch(
            resid_stream, filtered_timesteps_not, filtered_timesteps_this
        )
        if len(resid_stream_sample.shape) != 3:
            print("In valid - no inner batch found. skipping.")
            continue
        if resid_stream_sample.shape[1] == 0:
            print("In valid - no inner batch found. skipping.")
            continue

        all_resid_streams.append(resid_stream_sample)
        all_labels.extend(labels)

    all_resid_streams = torch.cat(all_resid_streams, dim=1)
    assert len(all_labels) == all_resid_streams.shape[1]
    return all_resid_streams.cpu(), all_labels


@torch.no_grad
def run_eval(
    actor,
    probe,
    tokenizer,
    # valid_dataloader,
    cached_valid_resid,
    cached_labels,
    lowest_val_loss,
    curr_patience,
    patience,
    config,
):
    batch_size = config["batch_size"]
    valid_size = config["valid_size"]
    max_new_tokens = config["max_response_length"]
    max_prompt_length = config["max_prompt_length"]
    record_module_names = config["record_module_names"]
    n_layers = config["n_layers"]
    d_model = config["d_model"]
    n_layers = probe.shape[0]

    token_open = tokenizer.encode(" (")[0]  # 320
    token_not = tokenizer.encode("not")[0]  # 1921
    token_this = tokenizer.encode("this")[0]  # 574

    val_losses = []
    val_accuracies = []
    done_training = False
    generation_config = GenerationConfig(do_sample=False)
    valid_seen = 0
    valid_size = cached_valid_resid.shape[1]
    for batch_idx in range(0, valid_size, batch_size):
        resid_stream_sample = cached_valid_resid[
            :, batch_idx : batch_idx + batch_size
        ].clone()
        labels = torch.tensor(cached_labels[batch_idx : batch_idx + batch_size])
        resid_stream_sample = resid_stream_sample.cuda()
        labels = labels.cuda()

        curr_batch_size = resid_stream_sample.shape[1]
        labels_one_hot = F.one_hot(labels, num_classes=2)

        probe_out = einsum(
            "n_layers batch d_model, n_layers d_model options -> n_layers batch options",
            resid_stream_sample,
            probe,
        )

        _labels_one_hot = labels_one_hot.unsqueeze(0).repeat((n_layers, 1, 1))

        probe_log_probs = probe_out.log_softmax(-1)
        probe_correct_log_probs = (
            einops.reduce(
                probe_log_probs * _labels_one_hot.cuda(),
                "n_layers batch options -> n_layers options",
                "mean",
            )
            * 2
        )

        valid_loss = -probe_correct_log_probs.mean().item()
        val_losses.append(valid_loss * curr_batch_size)

        # [n_layers, batch]
        val_preds = probe_log_probs.argmax(-1)
        val_gold = torch.tensor(labels).unsqueeze(0).repeat((n_layers, 1))

        val_results = val_preds.cpu() == val_gold.cpu()
        _val_accuracy = (val_results.sum(dim=1) / val_results.shape[1]).cpu()

        val_accuracies.append(_val_accuracy * curr_batch_size)
        valid_seen += curr_batch_size

    val_loss = sum(val_losses) / valid_seen
    val_accuracy_per_layer = torch.stack(val_accuracies, dim=1).sum(dim=1) / valid_seen

    print(f"  Validation loss: {val_loss}")
    print(f"  Validation accuracy: {val_accuracy_per_layer}")
    if val_loss < lowest_val_loss:
        print(f"  New lowest valid loss: {val_loss}!")
        curr_patience = 0
        output_dir = config["save_path"]
        output_probe_path = os.path.join(output_dir, f"probe.pt")
        metric_path = os.path.join(output_dir, f"metrics.pt")
        torch.save(probe, output_probe_path)
        torch.save(
            {
                "val_accuracy": val_accuracy_per_layer,
            },
            metric_path,
        )
        lowest_val_loss = val_loss

    else:
        curr_patience += 1
        print(f"  Did not beat previous best ({lowest_val_loss})")
        print(f"  Current patience: {curr_patience}")
        if curr_patience >= patience:
            print("  Ran out of patience! Stopping training.")
            done_training = True

    return {
        "loss": val_loss,
        "accuracy": val_accuracy_per_layer,
        "done_training": done_training,
        "lowest_val_loss": lowest_val_loss,
        "curr_patience": curr_patience,
    }


def main(config):
    assert torch.cuda.is_available()

    save_path = config["save_path"]
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    log_filepath = os.path.join(save_path, "metrics.pt")
    save_config(config)

    model_path = config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with torch.device("cuda"):
        actor_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        # actor_model.to(torch.bfloat16)

    train_dataloader, valid_dataloader = get_dataloader(config, tokenizer)

    # [n_layers, valid_size, d_model]
    print("Caching validation set...")
    cached_valid_resid, cached_valid_labels = cache_valid_data(
        actor_model, tokenizer, valid_dataloader, config
    )
    print("Done.")

    generation_config = GenerationConfig(do_sample=False)
    actor_model.cuda()

    # [vocab, d_model]
    lm_head = actor_model.lm_head.weight

    max_new_tokens = config["max_response_length"]
    max_prompt_length = config["max_prompt_length"]
    record_module_names = config["record_module_names"]
    log_interval = config["log_interval"]
    eval_interval = config["eval_interval"]
    n_layers = config["n_layers"]
    d_model = config["d_model"]
    probe_timestep = config["probe_timestep"]
    assert probe_timestep in ["last_int", "("]
    options = 2

    probe_model = torch.randn(
        n_layers,
        d_model,
        options,
        requires_grad=False,
        device="cuda",
    ) / np.sqrt(d_model)
    probe_model.requires_grad = True

    wd = 0.01
    lr = 1e-4
    optimiser = torch.optim.AdamW(
        [probe_model], lr=lr, betas=(0.9, 0.99), weight_decay=wd
    )
    optimiser.zero_grad()

    train_seen = 0
    done_training = False
    lowest_val_loss = 1e10
    curr_patience = 0
    patience = config["patience"]

    all_responses = []
    for batch_idx, batch in enumerate(train_dataloader):

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        with record_activations(actor_model, record_module_names) as recording:
            output = actor_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config=generation_config,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )  # may OOM when use_cache = True

        # len(recording["model.layers.0"]): max_response_length
        # recording["model.layers.0"][0].shape: [batch, prompt_length, d_model]
        # recording["model.layers.0"][1].shape: [batch, 1, d_model]
        recording = {
            layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording.items()
        }

        # recording["model.layers.0"].shape:
        # [batch, prompt_length + max_new_tokens, d_model]
        seq = output.sequences
        response = seq[:, -max_new_tokens:]
        response_text = tokenizer.batch_decode(response, skip_special_tokens=True)

        # [batch, n_layers, response_length, d_model]
        resid_stream = torch.stack(
            [acts[:, -max_new_tokens:] for acts in recording.values()], dim=1
        )

        if probe_timestep == "(":
            token_open = tokenizer.encode(" (")[0]  # 320
            token_not = tokenizer.encode("not")[0]  # 1921
            token_this = tokenizer.encode("this")[0]  # 574

            mask_not = (response[:, :-1] == token_open) & (response[:, 1:] == token_not)
            mask_this = (response[:, :-1] == token_open) & (
                response[:, 1:] == token_this
            )
            batch_idx_not, timesteps_not = torch.where(mask_not)
            batch_idx_this, timesteps_this = torch.where(mask_this)

            batch_idx_not = batch_idx_not
            batch_idx_this = batch_idx_this

            overlap_batches = torch.tensor(
                sorted(
                    list(
                        set(batch_idx_not.tolist()).intersection(
                            set(batch_idx_this.tolist())
                        )
                    )
                )
            ).cuda()
            batch_mask_not = torch.isin(batch_idx_not, overlap_batches)
            batch_mask_this = torch.isin(batch_idx_this, overlap_batches)

            filtered_timesteps_not = {
                b_idx: timesteps_not[(batch_idx_not == b_idx)]
                for b_idx in overlap_batches.tolist()
            }
            filtered_timesteps_this = {
                b_idx: timesteps_this[(batch_idx_this == b_idx)]
                for b_idx in overlap_batches.tolist()
            }

        else:
            filtered_timesteps_not = {}
            filtered_timesteps_this = {}

            for b_idx, text in enumerate(response_text):
                tokens = tokenizer(text, return_offsets_mapping=True)
                tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
                offsets = tokens["offset_mapping"]

                matches = re.findall(r"=\s*(-?\d+)\s*\((not|this works)", text)

                results = []
                for match in matches:
                    number, label = match  # Extract the number and label type

                    # Find number position in text
                    number_index = text.find(number)
                    number_end = number_index + len(number)

                    # Find corresponding token index
                    token_indices = [
                        i
                        for i, (start, end) in enumerate(offsets)
                        if start >= number_index and end <= number_end
                    ]

                    # Store results
                    results.append(
                        {
                            "number": number,
                            "tokens": [tokenized_text[i] for i in token_indices],
                            "token_index": token_indices[-1],
                            "all_token_indices": token_indices,
                            "label": "not" if label == "not" else "this",
                        }
                    )
                not_matches = [x for x in results if x["label"] == "not"]
                this_matches = [x for x in results if x["label"] == "this"]

                if len(not_matches) > 0 and len(this_matches) > 0:
                    filtered_timesteps_not[b_idx] = torch.tensor(
                        [r["token_index"] for r in not_matches]
                    )
                    filtered_timesteps_this[b_idx] = torch.tensor(
                        [r["token_index"] for r in this_matches]
                    )

        # resid_stream_sample: [n_layers, batch, d_model]
        resid_stream_sample, labels = _build_inner_batch(
            resid_stream, filtered_timesteps_not, filtered_timesteps_this
        )
        if len(resid_stream_sample.shape) != 3:
            print("Valid inner batch not found. Skipping...")
            continue
        if resid_stream_sample.shape[1] == 0:
            print("Valid inner batch not found. Skipping...")
            continue
        labels_one_hot = F.one_hot(torch.tensor(labels), num_classes=2)

        probe_out = einsum(
            "n_layers batch d_model, n_layers d_model options -> n_layers batch options",
            resid_stream_sample,
            probe_model,
        )

        _labels_one_hot = labels_one_hot.unsqueeze(0).repeat((n_layers, 1, 1))

        probe_log_probs = probe_out.log_softmax(-1)
        probe_correct_log_probs = (
            einops.reduce(
                probe_log_probs * _labels_one_hot.cuda(),
                "n_layers batch options -> n_layers options",
                "mean",
            )
            * 2
        )

        train_loss = -probe_correct_log_probs.mean()
        train_loss.backward()
        optimiser.step()

        if batch_idx % log_interval == 0:
            print(f"Batch {batch_idx}/{len(train_dataloader.dataset)}: {train_loss}")

        if batch_idx % eval_interval == 0:
            eval_results = run_eval(
                actor_model,
                probe_model,
                tokenizer,
                # valid_dataloader,
                cached_valid_resid,
                cached_valid_labels,
                lowest_val_loss,
                curr_patience,
                patience,
                config,
            )

            lowest_val_loss = eval_results["lowest_val_loss"]
            curr_patience = eval_results["curr_patience"]
            done_training = eval_results["done_training"]
            if done_training:
                return done_training

    return done_training


if __name__ == "__main__":
    config = {
        "data_path": "data/train.parquet",
        "model_path": "checkpoints/TinyZero/v4/actor/global_step_300",
        "batch_size": 128,
        "valid_size": 256,
        "max_prompt_length": 256,
        "max_response_length": 200,
        "save_path": "probe_checkpoints/probe_attn_out_parenthesis",
        "n_layers": 36,
        "d_model": 2048,
        "patience": 10,
        #"probe_timestep": "last_int",
        "probe_timestep": "(",
        "log_interval": 5,
        "eval_interval": 50,
        "probe_module": "attn",
    }
    n_layers = config["n_layers"]
    probe_module = config["probe_module"]

    if probe_module == "resid_stream":
        config["record_module_names"] = [
            f"model.layers.{idx}" for idx in range(n_layers)
        ]
    elif probe_module == "attn":
        config["record_module_names"] = [
            f"model.layers.{idx}.self_attn" for idx in range(n_layers)
        ]
    elif probe_module == "attn_o_proj":
        print("Qwen attn output == attn_o_proj.")
        breakpoint()
        config["record_module_names"] = [
            f"model.layers.{idx}.self_attn.o_proj" for idx in range(n_layers)
        ]
    elif probe_module == "mlp":
        config["record_module_names"] = [
            f"model.layers.{idx}.mlp" for idx in range(n_layers)
        ]
    else:
        raise RuntimeError(f"Invalid probe module name: {probe_module}")
    main(config)
