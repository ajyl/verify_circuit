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

from record_utils import record_activations, get_module
from HookedQwen import convert_to_hooked_model
from explore_utils import *

# %%

cos = F.cosine_similarity


# %%


def unembed_resid_streams(vector, model, k=10):
    """
    vector: [batch, layers, vocab]
    """
    norm = model.model.norm
    lm_head = model.lm_head.weight

    vector = norm(vector)
    dots = einsum(
        "vocab d_model, batch layers d_model -> layers batch vocab", lm_head, vector
    )
    return dots


def plot_logit_lens(_resid_stream, actor, layer_names, output_path):
    """
    _resid_stream: [layers, d_model]
    """
    dots = unembed_resid_streams(_resid_stream.unsqueeze(0), actor)
    # [layers, batch, vocab]
    probs = dots.softmax(dim=-1)

    # [layers, batch, k]
    top_idxs = probs.topk(k=10).indices
    top_probs = probs.topk(k=10).values
    top_idxs = top_idxs.squeeze()
    top_probs = top_probs.squeeze()

    tokens = [
        tokenizer.batch_decode(top_idxs[_idx, :], skip_special_tokens=True)
        for _idx in range(top_idxs.shape[0])
    ]

    hover_text = [
        [f"{tokens[i][j]} ({top_probs[i, j]:.2f})" for j in range(top_idxs.shape[1])]
        for i in range(top_idxs.shape[0])
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=top_probs.detach().cpu().numpy(),
            y=layer_names,
            hoverinfo="text",
            text=hover_text,
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )

    fig.update_layout(
        width=1600,
        height=1200,
        xaxis_title="Top K",
    )

    fig.write_html(output_path)


# %%


def plot_logit_lens_stacked(
    resid_stream1, resid_stream2, actor, layer_names, output_path
):
    def _get_plot_data(_rs):
        dots = unembed_resid_streams(_rs.unsqueeze(0), actor)
        # [layers, batch, vocab]
        probs = dots.softmax(dim=-1)

        # [layers, batch, k]
        top_idxs = probs.topk(k=10).indices
        top_probs = probs.topk(k=10).values
        top_idxs = top_idxs.squeeze()
        top_probs = top_probs.squeeze()

        tokens = [
            tokenizer.batch_decode(top_idxs[_idx, :], skip_special_tokens=True)
            for _idx in range(top_idxs.shape[0])
        ]

        hover_text = [
            [
                f"{tokens[i][j]} ({top_probs[i, j]:.2f})"
                for j in range(top_idxs.shape[1])
            ]
            for i in range(top_idxs.shape[0])
        ]
        return top_probs.detach().cpu().numpy(), hover_text

    top_probs1, hover_text1 = _get_plot_data(resid_stream1)
    top_probs2, hover_text2 = _get_plot_data(resid_stream2)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=("Resid Stream 1", "Resid Stream 2"),
        vertical_spacing=0.05,
    )

    heatmap1 = go.Heatmap(
        z=top_probs1,
        y=layer_names,
        zmin=0,
        zmax=1,
        hoverinfo="text",
        text=hover_text1,
        texttemplate="%{text}",
        textfont={"size": 12},
    )
    fig.add_trace(heatmap1, row=1, col=1)

    heatmap2 = go.Heatmap(
        z=top_probs2,
        y=layer_names,
        zmin=0,
        zmax=1,
        hoverinfo="text",
        text=hover_text2,
        texttemplate="%{text}",
        textfont={"size": 12},
    )
    fig.add_trace(heatmap2, row=2, col=1)

    fig.update_layout(
        width=1800,
        height=2600,
        xaxis_title="Top K",
    )
    fig.write_html(output_path)


# %%


def unembed(vector, lm_head, k=10):
    dots = einsum("vocab d_model, d_model -> vocab", lm_head, vector)
    top_k = dots.topk(k).indices
    return top_k


def unembed_text(vector, model, tokenizer, k=10):
    norm = model.model.norm
    lm_head = model.lm_head.weight
    top_k = unembed(vector, lm_head, k=k)
    return tokenizer.batch_decode(top_k, skip_special_tokens=True)


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


def get_all_hooks(model):
    all_hooks = []
    for (
        name,
        module,
    ) in model.named_modules():  # Recursively iterates through submodules
        if hasattr(module, "_forward_hooks"):
            all_hooks.extend(module._forward_hooks.values())

    return all_hooks


# %%

config = {
    "data_path": "data/train.parquet",
    "model_path": "checkpoints/TinyZero/v4/actor/global_step_300",
    "probe_path": "probe_checkpoints/v2/probe.pt",
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

n_layers = config["n_layers"]
record_module_names = [
    (f"model.layers.{idx}.hook_resid_mid", f"model.layers.{idx}")
    for idx in range(n_layers)
]
record_module_names = [x for sublist in record_module_names for x in sublist]

# %%

token_this = tokenizer.encode("this")[0]  # 574
token_open = tokenizer.encode(" (")[0]  # 320
token_not = tokenizer.encode("not")[0]  # 1921

# %%


# %%

max_new_tokens = 300
sample_size = 4
timestep_offset = 1
samples = []
for batch_idx, batch in enumerate(valid_dataloader):

    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    with record_activations(actor, record_module_names) as recording:
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

    mask_not = (response[:, :-1] == token_open) & (response[:, 1:] == token_not)
    mask_this = (response[:, :-1] == token_open) & (response[:, 1:] == token_this)
    batch_idx_not, timesteps_not = torch.where(mask_not)
    batch_idx_this, timesteps_this = torch.where(mask_this)

    batch_idx_not = batch_idx_not
    batch_idx_this = batch_idx_this

    overlap_batches = torch.tensor(
        sorted(
            list(set(batch_idx_not.tolist()).intersection(set(batch_idx_this.tolist())))
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
        _resid_stream = resid_stream[
            b_idx,
            :,
            _this_timesteps : _this_timesteps + timestep_offset + 1,
        ]
        samples.append(
            {
                "input_ids": input_ids[b_idx],
                "attention_mask": attention_mask[b_idx],
                "resid_stream": _resid_stream,  # [n_layers, timesteps, d_model]
                "not_timesteps": filtered_timesteps_not[b_idx],
                "this_timesteps": _this_timesteps,
                "prompt": tokenizer.batch_decode(
                    input_ids[b_idx], skip_special_tokens=True
                ),
            }
        )

    if len(samples) >= sample_size:
        break


# %%

print(samples[0]["resid_stream"].shape)
plot_logit_lens(
    samples[0]["resid_stream"][:, 0], actor, record_module_names, "logitlens.html"
)


# %%


def _add_mlp_hook(model, layer_idx, mlp_idxs):
    def hook(module, input, output):
        print("zz")
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
    device = "cuda"
    model.eval()  # Set the model to evaluation mode
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    batch_size = input_ids.shape[0]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

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

        # Get logits from the model. Ensure your model's forward function accepts an attention mask.
        output = model(
            idx_cond,
            attn_mask=attn_mask_cond,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = output["logits"]
        # Focus only on the last time step's logits
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)

        # Greedy sampling: select the token with the highest logit
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape: (batch, 1)

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
    device = "cuda"
    model.eval()  # Set the model to evaluation mode
    remove_all_hooks(model)

    eos_token_id = tokenizer.eos_token_id

    input_ids = input_ids.clone().to(device)
    attention_mask = attention_mask.clone().to(device)
    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Currently only supports a single sample at a time."

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    hook_mlps = hook_config["hook_mlp"]
    hook_target_chars = hook_config["hook_target_chars"]
    hook_timesteps = hook_config["hook_timesteps"]
    hook_token_ids = tokenizer.encode(hook_target_chars)
    print(hook_token_ids)

    hook_batch_idxs = []
    resid_streams = {}
    just_hooked = False
    for timestep in tqdm(range(max_new_tokens)):
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

        if just_hooked:
            with record_activations(model, record_module_names) as recording:
                output = model(
                    idx_cond,
                    attn_mask=attn_mask_cond,
                    position_ids=position_ids,
                    return_dict=True,
                )
                resid_stream = torch.stack(
                    [acts[0][:, -1] for acts in recording.values()], dim=1
                )
                resid_streams[timestep] = resid_stream
                just_hooked = False
        else:
            output = model(
                idx_cond,
                attn_mask=attn_mask_cond,
                position_ids=position_ids,
                return_dict=True,
            )

        logits = output["logits"]
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape: (batch, 1)

        most_recent_token = [x[-1] for x in tokenizer.batch_decode(idx_cond)]

        interv_batch_idx = []
        # for batch_idx in range(batch_size):
        # if (
        #    most_recent_token[batch_idx].isdigit()
        #    and next_token[batch_idx].item() in hook_token_ids
        # ):
        #    interv_batch_idx.append(batch_idx)
        # elif most_recent_token[batch_idx] in hook_target_chars:
        #    print("Hooking ...")
        #    interv_batch_idx.append(batch_idx)
        if timestep in hook_timesteps:
            interv_batch_idx = [0]

        if len(interv_batch_idx) > 0:
            print(f"Hooking timestep {timestep}")
            handles = []
            for layer_idx, mlp_idxs in hook_mlps.items():
                handles.append(_add_mlp_hook(actor, layer_idx, mlp_idxs))

            with record_activations(actor, record_module_names) as recording:
                interv_output = model(
                    idx_cond[interv_batch_idx],
                    attn_mask=attn_mask_cond[interv_batch_idx],
                    position_ids=position_ids[interv_batch_idx],
                    return_dict=True,
                )

            # resid_stream: [batch (1), n_layers, d_model]
            resid_stream = torch.stack(
                [acts[0][:, -1] for acts in recording.values()], dim=1
            )
            resid_streams[timestep] = resid_stream
            just_hooked = True

            logits = interv_output["logits"]
            logits = logits[:, -1, :]  # shape: (batch, vocab_size)
            interv_next_token = torch.argmax(
                logits, dim=-1, keepdim=True
            )  # shape: (batch, 1)

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

    return input_ids, resid_streams


# %%

hook_config = {
    "hook_mlp": {},
    "hook_target_chars": [" (", "("],
    "hook_timesteps": [86, 87],
}


for target_token in ["Success", "SUCCESS", "OK", "!)", "yes", "bingo"]:
    target_token_id = tokenizer.encode(target_token)[0]
    target_token_embed = actor.model.embed_tokens.weight[target_token_id]
    value_vecs = get_mlp_value_vecs(actor)

    for layer in range(20, 32):
        hook_config["hook_mlp"][layer] = []
        curr_value_vecs = value_vecs[layer]
        _dot_prods = einsum(
            "d_model d_mlp, d_model -> d_mlp", curr_value_vecs, target_token_embed
        )
        top_idxs = _dot_prods.topk(k=20).indices
        for _idx in top_idxs:
            # hook_config["hook_mlp"].append((layer, _idx.item()))
            if _idx.item() not in hook_config["hook_mlp"][layer]:
                hook_config["hook_mlp"][layer].append(_idx.item())

hook_config["hook_mlp"][26].extend(
    [
        6475,
        3665,
        3665,
        4334,
        3655,
    ]
)
# hook_config["hook_mlp"] = {}

hooked_output, resid_streams = generate_hooked(
    actor,
    samples[0]["input_ids"].unsqueeze(0),
    samples[0]["attention_mask"].unsqueeze(0),
    300,
    800,
    tokenizer,
    hook_config,
)

# %%

print(tokenizer.batch_decode(hooked_output, skip_special_tokens=True))

# %%

plot_logit_lens(
    samples[0]["resid_stream"][:, 1], actor, record_module_names, "logitlens_this.html"
)

# plot_logit_lens(resid_streams[86].clone()[0], actor, record_module_names, "testingzxcv.html")

# %%


# %%

last_timesteps = sorted(resid_streams.keys())[-2:]
plot_logit_lens_stacked(
    resid_streams[86][0].clone(),
    resid_streams[87][0].clone(),
    actor,
    record_module_names,
    "hooked.html",
)

# %%

probe_model = torch.load(config["probe_path"]).detach().cuda()

# %%

_k = 20
top_cos_scores = {0: [], 1: []}
for target_label in [0, 1]:
    for target_probe_layer in range(18, 36):
        target_probe = probe_model[target_probe_layer, :, target_label]

        for layer_idx in range(0, target_probe_layer + 1):
            print(f"Layer {layer_idx}")
            cos_scores = cos(value_vecs[layer_idx], target_probe.unsqueeze(-1), dim=0)
            _topk = cos_scores.topk(k=_k)
            _values = [x.item() for x in _topk.values]
            _idxs = [x.item() for x in _topk.indices]
            topk = list(
                zip(
                    _values,
                    _idxs,
                    [target_probe_layer] * _topk.indices.shape[0],
                    [layer_idx] * _topk.indices.shape[0],
                )
            )
            top_cos_scores[target_label].extend(topk)

sorted_scores_0 = sorted(top_cos_scores[0], key=lambda x: x[0], reverse=True)
sorted_scores_1 = sorted(top_cos_scores[1], key=lambda x: x[0], reverse=True)

# %%

hook_config = {
    "hook_mlp": {},
    "hook_target_chars": [" (", "("],
    "hook_timesteps": [86, 87],
}

for entry in sorted_scores_1 + sorted_scores_0:
    layer = entry[3]
    idx = entry[1]
    if layer not in hook_config["hook_mlp"]:
        hook_config["hook_mlp"][layer] = []
    hook_config["hook_mlp"][layer].append(idx)

hooked_output, resid_streams = generate_hooked(
    actor,
    samples[0]["input_ids"].unsqueeze(0),
    samples[0]["attention_mask"].unsqueeze(0),
    300,
    800,
    tokenizer,
    hook_config,
)


# %%

# plot_logit_lens(resid_streams[87].clone()[0], actor, record_module_names, f"logitlens_interv_pos_value_vecs_{_k}.html")
plot_logit_lens(
    resid_streams[87].clone()[0],
    actor,
    record_module_names,
    f"logitlens_interv_pos_and_neg_value_vecs_{_k}.html",
)

# %%

pos_value_vecs = torch.stack(
    [value_vecs[entry[3], :, entry[1]] for entry in sorted_scores_1],
    dim=0,
)
neg_value_vecs = torch.stack(
    [value_vecs[entry[3], :, entry[1]] for entry in sorted_scores_0],
    dim=0,
)

# %%


def find_min_cosine_similarity_pairs_batched(matrix1, matrix2, top_k=1, batch_size=500):
    """
    Computes the lowest cosine similarity pairs between two matrices in a memory-efficient way using batching.
    
    Args:
        matrix1 (torch.Tensor): Tensor of shape [N, D].
        matrix2 (torch.Tensor): Tensor of shape [M, D].
        top_k (int): Number of lowest similarity pairs to return per row.
        batch_size (int): Number of rows to process at a time.
    
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): Indices from matrix1, indices from matrix2, and similarity values.
    """
    N, D = matrix1.shape
    M, _ = matrix2.shape

    # Normalize matrices
    matrix1_norm = matrix1 / matrix1.norm(dim=1, keepdim=True)
    matrix2_norm = matrix2 / matrix2.norm(dim=1, keepdim=True)

    # Store results
    all_row_indices = []  # Indices from matrix1
    all_col_indices = []  # Indices from matrix2
    all_min_values = []   # Corresponding cosine similarity values

    for i in range(0, N, batch_size):
        batch = matrix1_norm[i : i + batch_size]  # Take a batch from matrix1
        
        # Compute cosine similarity for the batch (batch_size x M)
        cosine_sim = torch.mm(batch, matrix2_norm.T)  # Shape: [batch_size, M]

        # Get the smallest top_k similarities and their indices (along dim=1)
        min_values, min_indices = torch.topk(cosine_sim, k=top_k, largest=False, dim=1)

        # Convert local batch indices to global indices for matrix1
        row_indices = torch.arange(i, min(i + batch_size, N)).repeat_interleave(top_k)

        # Store results
        all_row_indices.append(row_indices)
        all_col_indices.append(min_indices.flatten())  # Corresponding indices in matrix2
        all_min_values.append(min_values.flatten())  # Corresponding similarity scores

    # Concatenate results from batches
    row_indices = torch.cat(all_row_indices).cpu()  # Indices from matrix1
    col_indices = torch.cat(all_col_indices).cpu()  # Indices from matrix2
    min_values = torch.cat(all_min_values).cpu()    # Cosine similarity values

    sorted_indices = torch.argsort(min_values).cpu()  # Get sorted order

    return row_indices[sorted_indices], col_indices[sorted_indices], min_values[sorted_indices]

# %%

# Run the memory-efficient cosine similarity computation
row_indices, col_indices, min_values = find_min_cosine_similarity_pairs_batched(
    pos_value_vecs, neg_value_vecs, top_k=10, batch_size=256
)

# %%

# Print some results
print("Row Indices:", row_indices[:10])
print("Col Indices:", col_indices[:10])
print("Cosine Similarity Values:", min_values[:10])

# %%

for _idx in range(100):
    pos_idx = sorted_scores_1[row_indices[_idx]]
    neg_idx = sorted_scores_0[col_indices[_idx]]
    pos_idx = (pos_idx[3], pos_idx[1])
    neg_idx = (neg_idx[3], neg_idx[1])
    pos_value_vec = value_vecs[pos_idx[0], :, pos_idx[1]]
    neg_value_vec = value_vecs[neg_idx[0], :, neg_idx[1]]
    
    print(cos(pos_value_vec, neg_value_vec, dim=0))
    
    print(unembed_text(pos_value_vec, actor, tokenizer, k=10))
    print(unembed_text(neg_value_vec, actor, tokenizer, k=10))
    print("--")


# %%

# Get pairwise cosine similarity between positive and negative value vectors
cos_sim_matrix = torch.zeros((pos_value_vecs.shape[0], neg_value_vecs.shape[0]))
for i in tqdm(range(pos_value_vecs.shape[0])):
    for j in range(neg_value_vecs.shape[0]):
        cos_sim_matrix[i, j] = cos(pos_value_vecs[i], neg_value_vecs[j], dim=0)


# %%

cos_sim_matrix.argmin()


# %%

orig_output = generate(
    actor,
    test_data["input_ids"],
    test_data["attention_mask"],
    300,
    300,
    tokenizer.eos_token_id,
)

# %%

print(tokenizer.batch_decode(orig_output, skip_special_tokens=True))


# %%


W_O = actor.model.layers[26].self_attn.o_proj.weight
W_O = einops.rearrange(W_O, "m (n h)->n h m", n=actor.config.num_attention_heads)

for h_idx in range(W_O.shape[0]):
    for inner_idx in range(W_O.shape[1]):
        print(unembed_text(W_O[h_idx, inner_idx], actor, tokenizer, k=10))


# %%

value_vecs = get_mlp_value_vecs(actor)

# %%


# def load_model(model_path, device="cuda"):
#    #assert torch.cuda.is_available()
#
#    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#    actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#    if device == "cuda":
#        with torch.device("cuda"):
#            actor_model = AutoModelForCausalLM.from_pretrained(
#                model_path, trust_remote_code=True
#            )
#    else:
#        actor_model = AutoModelForCausalLM.from_pretrained(
#            model_path, trust_remote_code=True
#        )
#    actor_model.tokenizer = tokenizer
#    return actor_model
#
# orig_model = load_model("Qwen/Qwen2.5-3B", device="cpu")
# orig_value_vecs = get_mlp_value_vecs(orig_model)
#
## %%
#
# for layer_idx in range(orig_value_vecs.shape[0]):
#    diff = value_vecs[layer_idx] - orig_value_vecs[layer_idx].cuda()
#    #diff = orig_value_vecs[layer_idx].cuda() - value_vecs[layer_idx]
#    #diff = diff.mean(dim=1)
#    hmm = diff.norm(dim=1).topk(k=10).indices
#    for idx in hmm:
#        print(unembed_text(value_vecs[layer_idx, :, idx], actor, tokenizer, k=20))
