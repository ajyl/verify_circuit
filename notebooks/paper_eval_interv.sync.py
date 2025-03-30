# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

import os
import json
import random
from tqdm import tqdm
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
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from src.record_utils import record_activations, get_module, untuple_tensor
from src.utils import load_model
from src.HookedQwen import convert_to_hooked_model
from src.rl_dataset import RLHFDataset

# %%

cos = F.cosine_similarity

# %%

base_dir = "/n/home01/ajyl/verify_circuit"

# %%


def seed_all(seed, deterministic_algos=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    if deterministic_algos:
        torch.use_deterministic_algorithms()


def unembed(vector, lm_head, k=10):
    dots = einsum("vocab d_model, d_model -> vocab", lm_head, vector)
    top_k = dots.topk(k).indices
    return top_k


def unembed_text(vector, lm_head, tokenizer, k=10):
    top_k = unembed(vector, lm_head, k=k)
    return tokenizer.batch_decode(top_k, skip_special_tokens=True)


# %%


def _add_o_proj_hook(model, layer_idx, head_idx):
    def hook(module, input, output):
        # output.shape: [batch, heads, seq, head_dim]
        output[:, :, head_idx, :] = 0
        return output

    module = model.model.layers[layer_idx].self_attn.hook_attn_out_per_head
    return module.register_forward_hook(hook)


# %%


def _turn_off_mlp(model, layer_idx, mlp_idxs):
    def hook(module, input, output):
        output[:, -1, mlp_idxs] = 0
        return output

    module = model.model.layers[layer_idx].mlp.hook_mlp_mid
    return module.register_forward_hook(hook)


# %%


@torch.no_grad()
def generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    block_size,
    eos_token_id,
):
    """
    Generate text using a transformer language model with greedy sampling.

    Args:
        model: The auto-regressive transformer model that outputs logits.
        input_ids: A tensor of shape (batch_size, sequence_length) representing the initial token indices.
        max_new_tokens: The number of new tokens to generate.
        block_size: The maximum sequence length (context window) the model can handle.
        device: The device on which computations are performed.

    Returns:
        A tensor containing the original context concatenated with the generated tokens.
    """
    model.eval()  # Set the model to evaluation mode
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")
    batch_size = input_ids.shape[0]

    finished = torch.zeros(batch_size, dtype=torch.bool).to("cuda")

    for _ in tqdm(range(max_new_tokens)):
        if finished.all():
            break

        if input_ids.shape[1] > block_size:
            idx_cond = input_ids[:, -block_size:]
            attn_mask_cond = attention_mask[:, -block_size:]
        else:
            idx_cond = input_ids
            attn_mask_cond = attention_mask

        position_ids = attn_mask_cond.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn_mask_cond == 0, 1)

        output = model(
            idx_cond.to(model.device),
            attention_mask=attn_mask_cond.to(model.device),
            position_ids=position_ids.to(model.device),
            return_dict=True,
        )
        logits = output["logits"]
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)

        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        new_finished = (~finished) & (next_token.squeeze(1) == eos_token_id)
        finished |= new_finished
        next_token[finished] = eos_token_id

        # Append the predicted token to the sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        new_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype).to("cuda")
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

    return input_ids


# %%


def add_hooks(model, hook_config):
    handles = []
    for hook_module, layer, head_idx in hook_config:
        if hook_module == "attn_out":
            hook_func = _add_o_proj_hook
        elif hook_module == "mlp":
            hook_func = _turn_off_mlp
        handles.append(hook_func(model, layer, head_idx))
    return handles


# %%


@torch.no_grad()
def generate_hooked(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    block_size,
    tokenizer,
    hook_config,
):
    """
    Generate text using a transformer language model with greedy sampling.

    Args:
        model: The auto-regressive transformer model that outputs logits.
        input_ids: A tensor of shape (batch_size, sequence_length) representing the initial token indices.
        max_new_tokens: The number of new tokens to generate.
        block_size: The maximum sequence length (context window) the model can handle.
        device: The device on which computations are performed.

    Returns:
        A tensor containing the original context concatenated with the generated tokens.
    """
    remove_all_hooks(model)

    device = "cuda"
    model.eval()  # Set the model to evaluation mode
    eos_token_id = tokenizer.eos_token_id

    input_ids = input_ids.clone().to(device)
    attention_mask = attention_mask.to(device)
    batch_size = input_ids.shape[0]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    token_open = tokenizer.encode(" (")[0]  # 320

    for _ in range(max_new_tokens):
        if finished.all():
            break

        if input_ids.shape[1] > block_size:
            idx_cond = input_ids[:, -block_size:]
            attn_mask_cond = attention_mask[:, -block_size:]
        else:
            idx_cond = input_ids
            attn_mask_cond = attention_mask

        position_ids = attn_mask_cond.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn_mask_cond == 0, 1)

        output = model(
            idx_cond,
            attention_mask=attn_mask_cond,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = output["logits"]
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape: (batch, 1)

        most_recent_token = [
            tokenizer.decode(idx_cond[batch_idx, -1]) for batch_idx in range(batch_size)
        ]

        interv_batch_idx = []
        for batch_idx in range(batch_size):
            if most_recent_token[batch_idx] == " (":
                interv_batch_idx.append(batch_idx)

        if len(interv_batch_idx) > 0:

            handles = add_hooks(model, hook_config)
            interv_output = model(
                idx_cond[interv_batch_idx],
                attention_mask=attn_mask_cond[interv_batch_idx],
                position_ids=position_ids[interv_batch_idx],
                return_dict=True,
            )
            logits = interv_output["logits"]
            logits = logits[:, -1, :]  # shape: (batch, vocab_size)
            interv_next_token = torch.argmax(logits, dim=-1, keepdim=True)
            next_token[interv_batch_idx] = interv_next_token

            for handle in handles:
                handle.remove()

        new_finished = (~finished) & (next_token.squeeze(1) == eos_token_id)
        finished |= new_finished
        next_token[finished] = eos_token_id

        # Append the predicted token to the sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        new_mask = torch.ones(
            (batch_size, 1), dtype=attention_mask.dtype, device=device
        )
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

    return input_ids


# %%

# %%


def get_mlp_value_vecs(model):
    mlp_value_vecs = [layer.mlp.down_proj.weight for layer in model.model.layers]
    # [n_layers, d_mlp (11008), d_model (2048)]
    return torch.stack(mlp_value_vecs, dim=0)


# %%


def remove_all_hooks(model):
    for (
        name,
        module,
    ) in model.named_modules():  # Recursively iterates through submodules
        if hasattr(module, "_forward_hooks"):
            for handle_id in list(module._forward_hooks.keys()):
                module._forward_hooks.pop(handle_id)


# %%

config = {
    "model_path": os.path.join(
        base_dir, "checkpoints/TinyZero/v4/actor/global_step_300"
    ),
    "probe_path": os.path.join(base_dir, "probe_checkpoints/v2/probe.pt"),
    "batch_size": 4,
    "max_prompt_length": 256,
    "max_response_length": 300,
    "n_layers": 36,
    "d_model": 2048,
    "seed": 42,
}

# %%

seed_all(config["seed"])
assert torch.cuda.is_available()

model_path = config["model_path"]
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# with torch.device("cuda"):
#    actor = AutoModelForCausalLM.from_pretrained(
#        model_path, trust_remote_code=True, device_map="auto"
#    )
actor = load_model(config["model_path"])
tokenizer = actor.tokenizer

# %%

convert_to_hooked_model(actor)

# %%

generation_config = GenerationConfig(do_sample=False)

# %%

token_this = tokenizer.encode("this")[0]  # 574
token_equals = tokenizer.encode("equals")[0]  
token_open = tokenizer.encode(" (")[0]  # 320
token_not = tokenizer.encode("not")[0]  # 1921

# %%

samples = torch.load(os.path.join(base_dir, "data/countdown/test_set2.pt"))

# %%

probe_model = torch.load(config["probe_path"]).detach().cuda()

# %%


def run(actor, samples, hook_config, batch_size, include_orig=False, test_size=None):
    prompt = "open_parenthesis"
    assert prompt in ["orig", "open_parenthesis"]
    max_gen_length = 300
    if prompt == "open_parenthesis":
        max_gen_length = 100

    generated_tokens = set()
    generated_tokens2 = set()
    this_timesteps = []
    all_generations = []
    odd_batches = []

    # Metrics:
    # 1) # of times the prediction changed from "this" to "not"
    num_not = 0
    total = 0

    # 2) # of times the model never realizes it has found a solution.
    this_count = 0

    if test_size is None:
        test_size = len(samples)

    for batch_idx in tqdm(range(0, test_size, batch_size)):
        curr_batch = samples[batch_idx : batch_idx + batch_size]
        input_ids = torch.stack(
            [curr_batch[_idx]["input_ids"] for _idx in range(len(curr_batch))], dim=0
        ).to("cuda")
        attention_mask = torch.stack(
            [curr_batch[_idx]["attention_mask"] for _idx in range(len(curr_batch))],
            dim=0,
        ).to("cuda")

        if include_orig:
            orig_output = actor.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config=generation_config,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )
            orig_output_text = tokenizer.batch_decode(
                orig_output.sequences, skip_special_tokens=True
            )

        _this_timestep = [sample["this_timestep"] + 1 for sample in curr_batch]
        this_timesteps.extend(_this_timestep)

        if prompt == "open_parenthesis":
            _input_ids = [
                curr_batch[_idx]["response"][: _this_timestep[_idx]]
                for _idx in range(len(curr_batch))
            ]
            max_length = max(seq.shape[0] for seq in _input_ids)
            padded_input_ids = []
            for seq in _input_ids:
                pad_length = max_length - seq.shape[0]
                padded = F.pad(seq, (pad_length, 0), value=tokenizer.pad_token_id)
                padded_input_ids.append(padded)
            input_ids = torch.stack(padded_input_ids, dim=0).to("cuda")
            attention_mask = input_ids != tokenizer.pad_token_id

        hooked_output = generate_hooked(
            actor,
            input_ids,
            attention_mask,
            max_gen_length,
            800,
            tokenizer,
            hook_config,
        )
        hooked_output_text = tokenizer.batch_decode(
            hooked_output, skip_special_tokens=True
        )
        all_generations.append(hooked_output_text)

        if prompt == "orig":
            preds = hooked_output[torch.arange(len(curr_batch)), _this_timestep]
        elif prompt == "open_parenthesis":
            preds = hooked_output[:, input_ids.shape[1]]
        else:
            raise ValueError("z")

        # preds.shape: [batch]
        num_not += (preds == token_not).sum().item()
        generated_tokens.update(preds.tolist())

        this_count += ((hooked_output == token_this).any(dim=1) | (
            hooked_output == token_equals
        ).any(dim=1)).sum().item()

        mask = hooked_output[:, :-1] == token_open
        tokens_after_parenthesis = hooked_output[:, 1:][mask]
        generated_tokens2.update(tokens_after_parenthesis.tolist())

        #if len(set(tokens_after_parenthesis.tolist())) > 1:
        #    print("Hmm.")
        #    print(tokenizer.batch_decode(tokens_after_parenthesis))
        #    odd_batches.append(batch_idx)

        total += len(curr_batch)

    return num_not / total, this_count / total


# %%


def build_mlp_hook_config(actor, probe_model, labels, layers, k):

    value_vecs = get_mlp_value_vecs(actor)
    hook_config = []
    for target_label in labels:
        for target_probe_layer in layers:
            target_probe = probe_model[target_probe_layer, :, target_label]
            _curr_value_vecs = value_vecs[target_probe_layer]

            cos_scores = cos(_curr_value_vecs, target_probe.unsqueeze(-1), dim=0)
            _topk = cos_scores.topk(k=k)
            _idxs = [x.item() for x in _topk.indices]
            for jj in range(k):
                hook_config.append(("mlp", target_probe_layer, _idxs[jj]))

    return hook_config


# %%


def build_attn_hook_config():
    heads = [
        (3, 13),
        (4, 5),
        (4, 0),
        (5, 9),
        (5, 14),
        (10, 0),
        (10, 5),
        (11, 8),
        (12, 3),
        (13, 6),
        (13, 3),
        (15, 8),
        (15, 4),
        (17, 14),
        (17, 13),
        (17, 11),
        (17, 10),
        (17, 9),
        (17, 3),
        (17, 1),
        (19, 13),
        (19, 8),
        (21, 7),
        (21, 14),
        (21, 2),
        (22, 14),
        (22, 12),
        (25, 14),
        (25, 11),
    ]
    heads = [("attn_out", layer, head_idx) for layer, head_idx in heads]
    return heads


# %%

batch_size = config["batch_size"]

# %%

# Orig:
hook_config = []
orig_not_perc, orig_this_perc = run(
    actor, samples, hook_config, batch_size, include_orig=False,
)
print(f"Orig not percentage: {orig_not_perc}")
print(f"Orig this percentage: {orig_this_perc}")

# %%

# MLP (only [1]):

print("Running MLP (only [1])")
hook_config = build_mlp_hook_config(actor, probe_model, [1], list(range(18, 36)), 50)
mlp_1_not_perc, mlp_1_this_perc = run(
    actor, samples, hook_config, batch_size, include_orig=False,
)
print(f"MLP 1 not percentage: {mlp_1_not_perc}")
print(f"MLP 1 this percentage: {mlp_1_this_perc}")

# %%

# MLP (Both [0, 1]):

print("Running MLP (both [0, 1])")
hook_config = build_mlp_hook_config(actor, probe_model, [0, 1], list(range(18, 36)), 50)
mlp_both_not_perc, mlp_both_this_perc = run(
    actor, samples, hook_config, batch_size, include_orig=False,
)
print(f"MLP both not percentage: {mlp_both_not_perc}")
print(f"MLP both this percentage: {mlp_both_this_perc}")

## %%

# Attention:

print("Running Attention")
hook_config = build_attn_hook_config()
attn_not_perc, attn_this_perc = run(
    actor, samples, hook_config, batch_size, include_orig=False,
)
print(f"Attn not percentage: {attn_not_perc}")
print(f"Attn this percentage: {attn_this_perc}")
