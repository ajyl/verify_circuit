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
import pandas as pd
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
import plotly.express as px
from plotly.subplots import make_subplots

from src.record_utils import record_activations, get_module, untuple_tensor

# from src.utils import load_model
from src.HookedQwen import convert_to_hooked_model
from src.rl_dataset import RLHFDataset

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


def get_mlp_value_vecs(model):
    mlp_value_vecs = [layer.mlp.down_proj.weight for layer in model.model.layers]
    # [n_layers, d_mlp (11008), d_model (2048)]
    return torch.stack(mlp_value_vecs, dim=0)


def get_mlp_key_vecs(model):
    mlp_key_vecs = [layer.mlp.up_proj.weight for layer in model.model.layers]
    # [n_layers, d_mlp (11008), d_model (2048)]
    return torch.stack(mlp_key_vecs, dim=0)


def get_mlp_gate_vecs(model):
    mlp_gate_vecs = [layer.mlp.gate_proj.weight for layer in model.model.layers]
    # [n_layers, d_mlp (11008), d_model (2048)]
    return torch.stack(mlp_gate_vecs, dim=0)


# %%

seed_all(42)

# %%

model_path = os.path.join(base_dir, "checkpoints/TinyZero/v4/actor/global_step_300")

tokenizer = AutoTokenizer.from_pretrained(model_path)

actor_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# %%

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", device_map="auto")

# %%

convert_to_hooked_model(actor_model)
convert_to_hooked_model(base_model)

# %%

actor_value_vecs = get_mlp_value_vecs(actor_model)
actor_key_vecs = get_mlp_key_vecs(actor_model)
actor_gate_vecs = get_mlp_gate_vecs(actor_model)
base_value_vecs = get_mlp_value_vecs(base_model)
base_key_vecs = get_mlp_key_vecs(base_model)
base_gate_vecs = get_mlp_gate_vecs(base_model)

# %%

probe_path = os.path.join(base_dir, "probe_checkpoints/v2/probe.pt")
probe_model = torch.load(probe_path).detach().cuda()

# %%

_k = 20
top_cos_scores = {0: [], 1: []}
cos = F.cosine_similarity
for target_label in [0, 1]:
    for target_probe_layer in range(18, 36):
        target_probe = probe_model[target_probe_layer, :, target_label]

        for layer_idx in range(0, target_probe_layer + 1):
            cos_scores = cos(
                actor_value_vecs[layer_idx], target_probe.unsqueeze(-1), dim=0
            )
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

k = 100
sorted_scores_1 = sorted_scores_1[:k]
sorted_scores_0 = sorted_scores_0[:k]

# %%

actor_pos_value_vecs = torch.stack(
    [actor_value_vecs[entry[3], :, entry[1]] for entry in sorted_scores_1],
    dim=0,
)
actor_neg_value_vecs = torch.stack(
    [actor_value_vecs[entry[3], :, entry[1]] for entry in sorted_scores_0],
    dim=0,
)
base_pos_value_vecs = torch.stack(
    [base_value_vecs[entry[3], :, entry[1]] for entry in sorted_scores_1],
    dim=0,
)
base_neg_value_vecs = torch.stack(
    [base_value_vecs[entry[3], :, entry[1]] for entry in sorted_scores_0],
    dim=0,
)

# %%

actor_pos_key_vecs = torch.stack(
    [actor_key_vecs[entry[3], entry[1]] for entry in sorted_scores_1],
    dim=0,
)
actor_neg_key_vecs = torch.stack(
    [actor_key_vecs[entry[3], entry[1]] for entry in sorted_scores_0],
    dim=0,
)
base_pos_key_vecs = torch.stack(
    [base_key_vecs[entry[3], entry[1]] for entry in sorted_scores_1],
    dim=0,
)
base_neg_key_vecs = torch.stack(
    [base_key_vecs[entry[3], entry[1]] for entry in sorted_scores_0],
    dim=0,
)


# %%

actor_pos_gate_vecs = torch.stack(
    [actor_gate_vecs[entry[3], entry[1]] for entry in sorted_scores_1],
    dim=0,
)
actor_neg_gate_vecs = torch.stack(
    [actor_gate_vecs[entry[3], entry[1]] for entry in sorted_scores_0],
    dim=0,
)
base_pos_gate_vecs = torch.stack(
    [base_gate_vecs[entry[3], entry[1]] for entry in sorted_scores_1],
    dim=0,
)
base_neg_gate_vecs = torch.stack(
    [base_gate_vecs[entry[3], entry[1]] for entry in sorted_scores_0],
    dim=0,
)


# %%

unembed_text(actor_pos_value_vecs[2], actor_model.lm_head.weight, tokenizer)

# %%

cos_scores = cos(actor_pos_value_vecs, base_pos_value_vecs, dim=1)
norm_diffs = actor_pos_value_vecs.norm(dim=1) - base_pos_value_vecs.norm(dim=1)

print(cos_scores)
print(norm_diffs)

# %%

cos_scores = cos(actor_pos_key_vecs, base_pos_key_vecs, dim=1)
norm_diffs = actor_pos_key_vecs.norm(dim=1) - base_pos_key_vecs.norm(dim=1)

print(cos_scores)
print(norm_diffs)

# %%

cos_scores = cos(actor_pos_gate_vecs, base_pos_gate_vecs, dim=1)
norm_diffs = actor_pos_gate_vecs.norm(dim=1) - base_pos_gate_vecs.norm(dim=1)

print(cos_scores)
print(norm_diffs)

# %%

heads = [
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
]

# %%

actor_WO = []
actor_WQ = []
actor_WK = []
actor_WV = []
base_WO = []
base_WQ = []
base_WK = []
base_WV = []
n_heads = actor_model.config.num_attention_heads
n_kv_heads = actor_model.config.num_key_value_heads
repeat_kv_heads = n_heads // n_kv_heads

for layer, head in heads:
    W_O = actor_model.model.layers[layer].self_attn.o_proj.weight
    # [heads, d_head, d_model]
    W_O = einops.rearrange(W_O, "m (n h)->n h m", n=n_heads)
    actor_WO.append(W_O[head])

    W_Q = actor_model.model.layers[layer].self_attn.q_proj.weight
    W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=n_heads)
    actor_WQ.append(W_Q[head])

    W_K = actor_model.model.layers[layer].self_attn.k_proj.weight
    W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
    W_K = torch.repeat_interleave(W_K, dim=0, repeats=repeat_kv_heads)
    actor_WK.append(W_K[head])

    W_V = actor_model.model.layers[layer].self_attn.v_proj.weight
    W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)
    W_V = torch.repeat_interleave(W_V, dim=0, repeats=repeat_kv_heads)
    actor_WV.append(W_V[head])

    W_O = base_model.model.layers[layer].self_attn.o_proj.weight
    W_O = einops.rearrange(W_O, "m (n h)->n h m", n=n_heads)
    base_WO.append(W_O[head])

    W_Q = base_model.model.layers[layer].self_attn.q_proj.weight
    W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=n_heads)
    base_WQ.append(W_Q[head])

    W_K = base_model.model.layers[layer].self_attn.k_proj.weight
    W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
    W_K = torch.repeat_interleave(W_K, dim=0, repeats=repeat_kv_heads)
    base_WK.append(W_K[head])

    W_V = base_model.model.layers[layer].self_attn.v_proj.weight
    W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)
    W_V = torch.repeat_interleave(W_V, dim=0, repeats=repeat_kv_heads)
    base_WV.append(W_V[head])


actor_WO = torch.stack(actor_WO, dim=0)
actor_WQ = torch.stack(actor_WQ, dim=0)
actor_WK = torch.stack(actor_WK, dim=0)
actor_WV = torch.stack(actor_WV, dim=0)
base_WO = torch.stack(base_WO, dim=0)
base_WQ = torch.stack(base_WQ, dim=0)
base_WK = torch.stack(base_WK, dim=0)
base_WV = torch.stack(base_WV, dim=0)

# %%

cos_scores = cos(actor_WO, base_WO, dim=2)
print(cos_scores)
cos_scores = cos(actor_WQ, base_WQ, dim=2)
print(cos_scores)
cos_scores = cos(actor_WK, base_WK, dim=2)
print(cos_scores)
cos_scores = cos(actor_WV, base_WV, dim=2)
print(cos_scores)

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

samples = torch.load(os.path.join(base_dir, "data/test_set2.pt"))

# %%

input_id = samples[0]["response"].unsqueeze(0)
this_timestep = samples[0]["this_timestep"] + 1
input_id = input_id[:, :this_timestep]
attention_mask = input_id != tokenizer.pad_token_id
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)

# %%

n_layers = 36
record_module_names = [
    (
        f"model.layers.{idx}.hook_resid_mid",
        f"model.layers.{idx}",
        f"model.layers.{idx}.self_attn.hook_attn_pattern",
        f"model.layers.{idx}.mlp.hook_mlp_mid",
    )
    for idx in range(n_layers)
]
record_module_names = [x for sublist in record_module_names for x in sublist]

# %%

with record_activations(actor_model, record_module_names) as actor_recording:
    actor_model(
        input_ids=input_id,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

# %%

with record_activations(base_model, record_module_names) as base_recording:
    base_model(
        input_ids=input_id,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

# %%

_actor_recording = {
    layer_name: torch.cat(acts, dim=1)
    for layer_name, acts in actor_recording.items()
    if ("self_attn" not in layer_name and "mlp" not in layer_name)
}
_base_recording = {
    layer_name: torch.cat(acts, dim=1)
    for layer_name, acts in base_recording.items()
    if ("self_attn" not in layer_name and "mlp" not in layer_name)
}

# [layers, seq, d_model]
actor_resid_stream = torch.stack(list(_actor_recording.values()), dim=1)[0]
base_resid_stream = torch.stack(
    list(_base_recording.values()),
    dim=1,
)[0]

# %%

plot_logit_lens(
    actor_resid_stream[:, -1], actor_model, list(_actor_recording.keys()), "actor_logitlens.html"
)
plot_logit_lens(
    base_resid_stream[:, -1], base_model, list(_base_recording.keys()), "base_logitlens.html"
)

# %%

actor_acts = []
base_acts = []
value_vec_unembedded = []
actor_mlp_recording = {
    layer_name: torch.cat(acts, dim=1) for layer_name, acts in actor_recording.items()
    if "mlp" in layer_name
}
base_mlp_recording = {
    layer_name: torch.cat(acts, dim=1) for layer_name, acts in base_recording.items()
    if "mlp" in layer_name
}
for _idx in range(100):
    mlp_layer = sorted_scores_1[_idx][3]
    mlp_idx = sorted_scores_1[_idx][1]
    actor_acts.append(
        actor_mlp_recording[f"model.layers.{mlp_layer}.mlp.hook_mlp_mid"][
            0, -1, mlp_idx
        ].item()
    )
    base_acts.append(
        base_mlp_recording[f"model.layers.{mlp_layer}.mlp.hook_mlp_mid"][
            0, -1, mlp_idx
        ].item()
    )
    value_vec_unembedded.append(
        ", ".join(
            unembed_text(
                actor_value_vecs[mlp_layer, :, mlp_idx],
                actor_model.lm_head.weight,
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
        "Activation": np.concatenate([actor_acts, base_acts]),
        "Model": ["R1"] * 100 + ["Base"] * 100,
        "Tokens": value_vec_unembedded + value_vec_unembedded,
    }
)

fig = px.bar(
    df,
    x="MLP Neuron",
    y="Activation",
    color="Model",
    hover_data={"Tokens": True},
    barmode="group",
    height=600,
    width=1200,
)

fig.update_layout(xaxis={"categoryorder": "category ascending"}, bargap=0.2)
fig.write_html("base_vs_r1_acts.html")
fig.show()



