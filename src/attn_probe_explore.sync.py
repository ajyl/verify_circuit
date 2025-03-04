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
import einops
from transformers import (
    GenerationConfig,
)
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
import einops
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from src.record_utils import record_activations, get_module
from src.HookedQwen import convert_to_hooked_model
from src.utils import seed_all, load_model, get_dataloader

# %%

cos = F.cosine_similarity

# %%

config = {
    "data_path": "data/train.parquet",
    "model_path": "checkpoints/TinyZero/v4/actor/global_step_300",
    "batch_size": 4,
    "valid_size": 256,
    "max_prompt_length": 256,
    "max_response_length": 300,
    "n_layers": 36,
    "d_model": 2048,
    "seed": 42,
}

seed_all(config["seed"])


# %%

actor = load_model(config["model_path"])
generation_config = GenerationConfig(do_sample=False)
tokenizer = actor.tokenizer

# %%

convert_to_hooked_model(actor)

# %%

_, valid_dataloader = get_dataloader(
    config["data_path"],
    config["batch_size"],
    config["max_prompt_length"],
    config["valid_size"],
    actor.tokenizer,
)


# %%

probe_path = "probe_checkpoints/probe_attn_per_head_last_int/probe.pt"
probes = torch.load(probe_path).detach()
print(probes.shape)


# %%

n_layers = probes.shape[0]
n_heads = probes.shape[1]
d_model = probes.shape[2]

pos_probes = probes[..., 1]
neg_probes = probes[..., 0]

# %%

norms = torch.norm(pos_probes, dim=-1)  # Shape: [Layers, num_heads]
# Plot the heatmap
plt.figure(figsize=(12, 10))
plt.imshow(norms.cpu().numpy(), cmap="viridis", aspect="auto")
plt.colorbar(label="Norm")
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Norm of Vectors per Layer and Head")
plt.xticks(range(n_heads))
plt.yticks(range(n_layers))
plt.show()

# %%

token_this = tokenizer.encode("this")[0]  # 574
token_open = tokenizer.encode(" (")[0]  # 320
token_not = tokenizer.encode("not")[0]  # 1921

# %%


def get_samples(actor, valid_dataloader, sample_size=1):
    max_new_tokens = 300
    timestep_offset = 1
    samples = []
    for batch_idx, batch in enumerate(valid_dataloader):

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        output = actor.generate(
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

        # [batch, prompt_length + max_new_tokens, d_model]
        response = output.sequences
        response_text = tokenizer.batch_decode(response, skip_special_tokens=True)

        mask_not = (response[:, :-1] == token_open) & (response[:, 1:] == token_not)
        mask_this = (response[:, :-1] == token_open) & (response[:, 1:] == token_this)
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

        for b_idx in filtered_timesteps_this.keys():
            print("Found a match...")
            _this_timesteps = filtered_timesteps_this[b_idx]
            samples.append(
                {
                    "input_ids": input_ids[b_idx],
                    "attention_mask": attention_mask[b_idx],
                    "not_timesteps": filtered_timesteps_not[b_idx],
                    "this_timesteps": _this_timesteps,
                    "prompt": tokenizer.batch_decode(
                        input_ids[b_idx], skip_special_tokens=True
                    ),
                    "output": output.sequences[b_idx],
                }
            )

        if len(samples) >= sample_size:
            break

    return samples


# %%

samples = get_samples(actor, valid_dataloader, sample_size=1)

# %%

# %%

record_module_names = [
    f"model.layers.{idx}.self_attn.hook_attn_out_per_head" for idx in range(n_layers)
]

max_new_tokens = 300
with record_activations(actor, record_module_names) as recording:
    output = actor.generate(
        input_ids=samples[0]["input_ids"].unsqueeze(0),
        attention_mask=samples[0]["attention_mask"].unsqueeze(0),
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        output_scores=False,  # this is potentially very large
        return_dict_in_generate=True,
        use_cache=True,
    )  # may OOM when use_cache = True

# %%

response = output.sequences[:, -max_new_tokens:]
response_text = tokenizer.batch_decode(response, skip_special_tokens=True)
print(response_text)

# %%

num_timesteps = response.shape[1]
proj_norms = torch.zeros(n_layers, n_heads, num_timesteps)
print(f"proj_norm shape: {proj_norms.shape}")
for layer_idx in range(n_layers):

    # [batch, seq, heads, d_model]
    attn_out = torch.cat(
        recording[f"model.layers.{layer_idx}.self_attn.hook_attn_out_per_head"],
        dim=1,
    )
    attn_out = attn_out[:, -max_new_tokens:, ...]

    for h_idx in range(n_heads):
        _attn_out = attn_out[0, :, h_idx, :]
        curr_probe = probes[layer_idx, h_idx, :, 1]

        dots = einsum("seq d_model, d_model -> seq", _attn_out, curr_probe)
        proj = dots / torch.norm(curr_probe)
        proj_norms[layer_idx, h_idx] = proj

# %%

tokens = tokenizer.batch_decode(response[0])
proj_norm_np = proj_norms.numpy()
# Create a subplot per layer. Each subplot will have 'n_heads' curves.
fig, axes = plt.subplots(n_layers, 1, figsize=(18, 6 * n_layers), sharex=True)
if n_layers == 1:
    axes = [axes]  # Ensure axes is always a list for consistency

for l in range(n_layers):
    ax = axes[l]
    for h in range(n_heads):
        ax.plot(range(num_timesteps), proj_norm_np[l, h, :], label=f"Head {h}")
    ax.set_ylabel("Projection Norm")
    ax.set_title(f"Layer {l}")
    #ax.legend(loc="upper right")
    ax.set_xticks(range(num_timesteps))
    ax.set_xticklabels(tokens, rotation=90, ha="right")

axes[-1].set_xlabel("Timestep")
plt.tight_layout()
plt.show()
