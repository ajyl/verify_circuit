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
from collections import Counter
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
import plotly.subplots as sp
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

from src.record_utils import record_activations, get_module, untuple_tensor
from src.HookedQwen import convert_to_hooked_model
from src.rl_dataset import RLHFDataset


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

    _, valid_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - valid_size, valid_size]
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return valid_loader


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
    eos_token_id = tokenizer.eos_token_id

    input_ids = input_ids.clone().to(device)
    attention_mask = attention_mask.to(device)
    batch_size = input_ids.shape[0]

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    hook_attn_heads = hook_config["heads"]

    token_open = tokenizer.encode(" (")[0]  # 320

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
            idx_cond,
            attention_mask=attn_mask_cond,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = output["logits"]
        logits = logits[:, -1, :]  # shape: (batch, vocab_size)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # shape: (batch, 1)

        most_recent_token = [
            tokenizer.decode(idx_cond[:, -1]) for batch_idx in range(batch_size)
        ]

        interv_batch_idx = []
        this_interv_batch_idx = []
        for batch_idx in range(batch_size):
            if (
                most_recent_token[batch_idx] == " ("
                and next_token[batch_idx].item() == token_this
            ):
                interv_batch_idx.append(batch_idx)

        if len(interv_batch_idx) > 0:

            handles = []
            for head_layer, head_idx in hook_attn_heads:
                # handles.append(_add_value_state_hook(model, head_layer, head_idx))
                handles.append(_add_o_proj_hook(model, head_layer, head_idx))

            interv_output = model(
                idx_cond[interv_batch_idx],
                attention_mask=attn_mask_cond[interv_batch_idx],
                position_ids=position_ids[interv_batch_idx],
                return_dict=True,
            )
            logits = interv_output["logits"]
            logits = logits[:, -1, :]  # shape: (batch, vocab_size)
            interv_next_token = torch.argmax(
                logits, dim=-1, keepdim=True
            )  # shape: (batch, 1)
            next_token[interv_batch_idx] = interv_next_token

            for handle in handles:
                handle.remove()

        # elif len(this_interv_batch_idx) > 0:
        #    handles = []
        #    for head_layer, head_idx in hook_attn_heads:
        #        # handles.append(_add_value_state_hook(model, head_layer, head_idx))
        #        handles.append(_add_o_proj_hook(model, head_layer, head_idx))

        #    interv_output = model(
        #        idx_cond[interv_batch_idx],
        #        attention_mask=attn_mask_cond[interv_batch_idx],
        #        position_ids=position_ids[interv_batch_idx],
        #        return_dict=True,
        #    )
        #    logits = interv_output["logits"]
        #    logits = logits[:, -1, :]  # shape: (batch, vocab_size)
        #    interv_next_token = torch.argmax(
        #        logits, dim=-1, keepdim=True
        #    )  # shape: (batch, 1)
        #    next_token[interv_batch_idx] = interv_next_token

        #    for handle in handles:
        #        handle.remove()

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


def make_tokens_unique(tokens):
    """
    Ensures tokens are unique by appending an index to duplicates.

    Args:
        tokens (list of str): Original token list.

    Returns:
        list of str: Modified token list with unique names.
    """
    token_counts = Counter()
    unique_tokens = []
    token_display_map = {}

    for token in tokens:
        token_counts[token] += 1
        unique_token = f"{token}__{token_counts[token]}"
        unique_tokens.append(unique_token)
        token_display_map[unique_token] = token

    return unique_tokens, token_display_map


def plot_interactive_attention_all_layers(attn_matrix, tokens, output_filepath):
    """
    Creates an interactive heatmap for attention visualization.

    Args:
        attn_matrix (torch.Tensor): A tensor of shape [n_heads, seq_len] (single attention head).
        tokens (list): List of tokens corresponding to sequence positions.
    """
    n_layers = attn_matrix.shape[0]
    n_heads = attn_matrix.shape[1]
    seq_len = attn_matrix.shape[2]
    uniq_tokens, display_tokens = make_tokens_unique(tokens)

    fig = sp.make_subplots(
        rows=n_layers,
        cols=1,
        subplot_titles=[f"Layer {i}" for i in range(n_layers)],
    )

    for layer in range(n_layers):
        attn = attn_matrix[layer]
        # Convert to a Pandas DataFrame for easy plotting
        df = pd.DataFrame(
            attn.cpu().numpy(),
            index=[f"Head {i}" for i in range(n_heads)],
            columns=uniq_tokens,
        )

        hover_text = [
            [
                f"Token: {display_tokens[df.columns[j]]}<br>Head: {i}<br>Index: {j}<br>Score: {df.iloc[i, j]:.2f}"
                for j in range(seq_len)
            ]
            for i in range(n_heads)
        ]

        # Create the heatmap
        heatmap = go.Heatmap(
            z=df.values,
            x=uniq_tokens,
            y=df.index,
            colorscale="viridis",
            text=hover_text,
            hoverinfo="text",
            zmin=0,
            zmax=1,
        )
        fig.add_trace(heatmap, row=layer + 1, col=1)

    height = n_layers * 350
    # Improve interactivity
    fig.update_layout(
        title="Interactive Attention Map",
        xaxis_title="Key Positions (Tokens Being Attended To)",
        yaxis_title="Attention Heads",
        hovermode="closest",
        height=height,
    )

    fig.write_html(output_filepath)


# %%


def _add_value_state_hook(model, layer_idx, head_idx):
    def hook(module, input, output):
        # output.shape: [batch, heads, seq, head_dim]
        output[:, head_idx, :, :] = output[:, head_idx, :, :] / 1e2
        return output

    module = model.model.layers[layer_idx].self_attn.hook_value_states_post_attn
    return module.register_forward_hook(hook)


def _add_o_proj_hook(model, layer_idx, head_idx):
    def hook(module, input, output):
        # output.shape: [batch, heads, seq, head_dim]
        # output[:, :, head_idx, :] = output[:, :, head_idx, :] / 1e10
        output[:, :, head_idx, :] = 0
        return output

    # module = model.model.layers[layer_idx].self_attn.hook_o_proj
    module = model.model.layers[layer_idx].self_attn.hook_attn_out_per_head
    return module.register_forward_hook(hook)


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

config = {
    "data_path": os.path.join(base_dir, "data/train.parquet"),
    "model_path": os.path.join(
        base_dir, "checkpoints/TinyZero/v4/actor/global_step_300"
    ),
    "probe_path": os.path.join(base_dir, "probe_checkpoints/v2/probe.pt"),
    "batch_size": 4,
    "valid_size": 256,
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
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
with torch.device("cuda"):
    actor_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto"
    )

# %%

convert_to_hooked_model(actor_model)

# %%

valid_dataloader = get_dataloader(config, tokenizer)

# %%

generation_config = GenerationConfig(do_sample=False)
actor_model.cuda()

# [vocab, d_model]
lm_head = actor_model.lm_head.weight

# %%

token_this = tokenizer.encode("this")[0]  # 574
token_open = tokenizer.encode(" (")[0]  # 320
token_not = tokenizer.encode("not")[0]  # 1921

# %%

# Load some samples that the model gets correct.
samples = torch.load("../toy_samples.pt")


# %%

orig_output = generate(
    actor_model,
    samples[0]["input_ids"].unsqueeze(0),
    samples[0]["attention_mask"].unsqueeze(0),
    300,
    800,
    tokenizer.eos_token_id,
)

orig_output_text = tokenizer.decode(orig_output[0], skip_special_tokens=True)
print(orig_output_text)

# %%

# cut off at "( this"
target_token_id = tokenizer.encode(" (")[0]  # 320
target_next_token_id = tokenizer.encode("this")[0]  # 574
print(target_token_id)
print(target_next_token_id)
mask = (orig_output[:, :-1] == target_token_id) & (
    orig_output[:, 1:] == target_next_token_id
)
batch_idxs, target_timestep = torch.where(mask)
print(target_timestep)
print(tokenizer.decode(orig_output[0][:target_timestep], skip_special_tokens=True))

# %%

_offset = 0
_input_ids = orig_output[0][: target_timestep + _offset].unsqueeze(0)
_attention_mask = _input_ids != tokenizer.pad_token_id
_position_ids = _attention_mask.long().cumsum(-1) - 1
_position_ids.masked_fill_(_attention_mask == 0, 1)

n_layers = 36
record_module_names = [
    (
        f"model.layers.{idx}.hook_resid_mid",
        f"model.layers.{idx}",
        f"model.layers.{idx}.self_attn.hook_attn_pattern",
    )
    for idx in range(n_layers)
]
record_module_names = [x for sublist in record_module_names for x in sublist]

with record_activations(actor_model, record_module_names) as recording:
    actor_model(
        _input_ids,
        attention_mask=_attention_mask,
        position_ids=_position_ids,
    )

# %%

_recording = {
    layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording.items()
}
_resid_stream = torch.stack(
    [acts for layer_name, acts in _recording.items() if "self_attn" not in layer_name],
    dim=1,
)

# %%

plot_logit_lens(
    _resid_stream[0, :, -1, :], actor_model, list(recording.keys()), "pre_hook.html"
)


# %%

# Let's look at some attention heads when it predicts " (".


def pad_and_concatenate(tensor_list):
    """
    Pads and concatenates a list of tensors along the given dimension.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to concatenate.

    Returns:
        torch.Tensor: Padded and concatenated tensor.
    """
    # Find the max size in the target dimension
    max_size = tensor_list[-1].shape[-1]

    # Pad each tensor to match max_size in the given dimension
    padded_tensors = []
    for tensor_idx, tensor in enumerate(tensor_list):
        if tensor_idx == 0:
            zeros = torch.zeros(
                tensor.shape[0],
                tensor.shape[1],
                tensor.shape[2],
                max_size,
                device=tensor_list[-1].device,
            )
        else:
            zeros = torch.zeros_like(tensor_list[-1])
        zeros[:, :, :, : tensor.shape[-1]] = tensor
        padded_tensors.append(zeros)

    attn_pattern = torch.cat(padded_tensors, dim=2)
    assert attn_pattern.shape[2] == attn_pattern.shape[3]
    return attn_pattern


# %%

attn_patterns = torch.stack(
    [
        pad_and_concatenate(
            recording[f"model.layers.{layer_idx}.self_attn.hook_attn_pattern"]
        )
        for layer_idx in range(n_layers)
    ],
    dim=1,
)

# %%


def first_true_indices(tensor):
    """
    Finds the first occurrence of True in each row of a [batch, seq] boolean tensor.

    Args:
        tensor (torch.Tensor): A boolean tensor of shape [batch, seq].

    Returns:
        torch.Tensor: A tensor of shape [batch] containing the index of the first True in each row.
                      If no True is found, returns -1 for that row.
    """
    batch_size, seq_len = tensor.shape

    # Create an index tensor [0, 1, 2, ..., seq_len-1] and expand it across batch dimension
    indices = torch.arange(seq_len, device=tensor.device).expand(batch_size, seq_len)

    # Mask out positions where tensor is False (set to a large number so they are ignored in min())
    masked_indices = torch.where(
        tensor, indices, torch.tensor(seq_len, device=tensor.device)
    )

    # Find the minimum index where True occurs, or return -1 if no True is found
    first_true = masked_indices.min(dim=1).values
    first_true[first_true == seq_len] = (
        -1
    )  # Replace seq_len with -1 for rows where no True exists

    return first_true


# %%

pad_offsets = first_true_indices(_attention_mask.bool())

# %%

prompt_offset = samples[0]["input_ids"].shape[0]
pad_offset = pad_offsets[0]
attn_at_timestep = attn_patterns[
    0,
    :,
    :,
    -1,
    pad_offset:,
]

tokens = tokenizer.batch_decode(_input_ids[0, pad_offset:])

# %%

plot_interactive_attention_all_layers(attn_at_timestep, tokens, "attn_patterns.html")

# %%

hook_config = {
    "heads": [
        (3, 13),
        (4, 5),
        (4, 0),
        (5, 9),
        (5, 14),
        # (6, 6), (maybe)
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
        # (18, 7 (maybe)),
        # (18, 3 (maybe)),
        (19, 13),
        (19, 8),
        # (19, 14 (maybe)),
        # (19, 12 (maybe)),
        # (19, 6 (maybe)),
        # (19, 0 (maybe)),
        # (20, 1 (attends to next token after "62")),
        # (20, 3 (attends to next token after "62")),
        # (20, 4 (attends to next token after "62")),
        # (20, 5 (attends to next token after "62")),
        # (20, 6 (attends to next token after "62")),
        (21, 7),
        (21, 14),
        (21, 2),
        (22, 14),
        (22, 12),
        (25, 14),
        (25, 11),
    ],
}

# %%

handles = []
for head_layer, head_idx in hook_config["heads"]:
    handles.append(_add_o_proj_hook(actor_model, head_layer, head_idx))

with record_activations(actor_model, record_module_names) as recording:
    actor_model(
        _input_ids,
        attention_mask=_attention_mask,
        position_ids=_position_ids,
    )

for handle in handles:
    handle.remove()

recording = {
    layer_name: torch.cat(acts, dim=1)
    for layer_name, acts in recording.items()
    if "self_attn" not in layer_name
}
_resid_stream = torch.stack([acts for acts in recording.values()], dim=1)

plot_logit_lens(
    _resid_stream[0, :, -1, :], actor_model, list(recording.keys()), "post_hook.html"
)

# %%


hooked_output = generate_hooked(
    actor_model,
    samples[0]["input_ids"].unsqueeze(0),
    samples[0]["attention_mask"].unsqueeze(0),
    300,
    800,
    tokenizer,
    hook_config,
)

# %%

hooked_output_text = tokenizer.batch_decode(hooked_output, skip_special_tokens=True)

print(hooked_output_text[0])

# %%

probe_model = torch.load(config["probe_path"]).detach().cuda()

# %%


def get_mlp_value_vecs(model):
    mlp_value_vecs = [layer.mlp.down_proj.weight for layer in model.model.layers]
    # [n_layers, d_mlp (11008), d_model (2048)]
    return torch.stack(mlp_value_vecs, dim=0)


# %%

value_vecs = get_mlp_value_vecs(actor_model)

# %%

_k = 20
top_cos_scores = {0: [], 1: []}
cos = F.cosine_similarity
for target_label in [0, 1]:
    for target_probe_layer in range(18, 36):
        target_probe = probe_model[target_probe_layer, :, target_label]

        for layer_idx in range(0, target_probe_layer + 1):
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

_sorted_scores_0 = sorted(top_cos_scores[0], key=lambda x: x[0], reverse=True)
_sorted_scores_1 = sorted(top_cos_scores[1], key=lambda x: x[0], reverse=True)

_unique = set()
sorted_scores_0 = []
for entry in _sorted_scores_0:
    _pair = (entry[3], entry[1])
    if _pair not in _unique:
        _unique.add(_pair)
        sorted_scores_0.append(entry)

_unique = set()
sorted_scores_1 = []
for entry in _sorted_scores_1:
    _pair = (entry[3], entry[1])
    if _pair not in _unique:
        _unique.add(_pair)
        sorted_scores_1.append(entry)

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

record_module_names = [
    f"model.layers.{idx}.mlp.hook_mlp_mid" for idx in range(n_layers)
]
with record_activations(actor_model, record_module_names) as recording_pre:
    actor_model(
        _input_ids,
        attention_mask=_attention_mask,
        position_ids=_position_ids,
    )
recording_pre = {
    layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording_pre.items()
}

# %%

handles = []
for head_layer, head_idx in hook_config["heads"]:
    handles.append(_add_o_proj_hook(actor_model, head_layer, head_idx))

with record_activations(actor_model, record_module_names) as recording_post:
    actor_model(
        _input_ids,
        attention_mask=_attention_mask,
        position_ids=_position_ids,
    )
recording_post = {
    layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording_post.items()
}

for handle in handles:
    handle.remove()

# %%

pre_acts = []
post_acts = []
value_vec_unembedded = []
for _idx in range(100):
    mlp_layer = sorted_scores_1[_idx][3]
    mlp_idx = sorted_scores_1[_idx][1]
    pre_acts.append(
        recording_pre[f"model.layers.{mlp_layer}.mlp.hook_mlp_mid"][
            0, -1, mlp_idx
        ].item()
    )
    post_acts.append(
        recording_post[f"model.layers.{mlp_layer}.mlp.hook_mlp_mid"][
            0, -1, mlp_idx
        ].item()
    )
    value_vec_unembedded.append(
        ", ".join(
            unembed_text(
                value_vecs[mlp_layer, :, mlp_idx],
                actor_model,
                tokenizer,
                k=10,
            )
        )
    )

# %%

labels = [f"{entry[3]}_{entry[1]}" for entry in sorted_scores_1[:100]]

df = pd.DataFrame(
    {
        "MLP Neuron": labels + labels,
        "Activation": np.concatenate([pre_acts, post_acts]),
        "Intervened?": ["No (Orig)"] * 100 + ["Yes"] * 100,
        "Tokens": value_vec_unembedded + value_vec_unembedded,
    }
)

fig = px.bar(
    df,
    x="MLP Neuron",
    y="Activation",
    color="Intervened?",
    hover_data={"Tokens": True},
    barmode="group",
    height=600,
    width=1200,
)

fig.update_layout(xaxis={"categoryorder": "category ascending"}, bargap=0.2)
fig.write_html("mlp_neurons.html")
fig.show()

# %%


