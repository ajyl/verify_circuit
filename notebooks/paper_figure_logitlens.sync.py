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

base_dir = "/n/home01/ajyl/verify_circuit"

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


# %%

config = {
    "data_path": os.path.join(base_dir, "data/train.parquet"),
    "model_path": os.path.join(
        base_dir, "checkpoints/TinyZero/v4/actor/global_step_300"
    ),
    "probe_path": os.path.join(base_dir, "probe_checkpoints/v2/probe.pt"),
    "batch_size": 8,
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

n_layers = config["n_layers"]
record_module_names = [f"model.layers.{idx}" for idx in range(n_layers)]

# %%

token_this = tokenizer.encode("this")[0]  # 574
token_open = tokenizer.encode(" (")[0]  # 320
token_not = tokenizer.encode("not")[0]  # 1921


# %%

probe_model = torch.load(config["probe_path"]).detach().cuda()


# %%

# Paper plots.
samples = torch.load(
    os.path.join(base_dir, "data/countdown/test_set2.pt"), map_location="cpu"
)

# %%


def _get_resid_stream(actor, samples, record_module_names, config):
    test_size = len(samples)
    batch_size = config["batch_size"]

    _all_resid_streams = []
    for batch_idx in tqdm(range(0, test_size, batch_size)):
        curr_batch = samples[batch_idx : batch_idx + batch_size]
        input_ids = torch.stack(
            [curr_batch[_idx]["input_ids"] for _idx in range(len(curr_batch))], dim=0
        ).to("cuda")
        attention_mask = torch.stack(
            [curr_batch[_idx]["attention_mask"] for _idx in range(len(curr_batch))],
            dim=0,
        ).to("cuda")

        _this_timestep = [sample["this_timestep"] + 1 for sample in curr_batch]

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
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            with record_activations(actor, record_module_names) as recording:
                output = actor(
                    input_ids.to(actor.device),
                    attention_mask=attention_mask.to(actor.device),
                    position_ids=position_ids.to(actor.device),
                    return_dict=True,
                )

        recording = {
            layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording.items()
        }
        resid_stream = torch.stack([acts for acts in recording.values()], dim=1)[
            :, :, -1, :
        ].cpu()
        _all_resid_streams.append(resid_stream)

    all_resid_streams = torch.cat(_all_resid_streams, dim=0)
    return all_resid_streams


# %%


def get_orig_resid_streams(
    actor, samples, record_module_names, config
):
    test_size = len(samples)
    batch_size = config["batch_size"]

    _all_resid_streams_this = []
    _all_resid_streams_not = []
    for batch_idx in tqdm(range(0, test_size, batch_size)):
        curr_batch = samples[batch_idx : batch_idx + batch_size]
        input_ids = torch.stack(
            [curr_batch[_idx]["input_ids"] for _idx in range(len(curr_batch))], dim=0
        ).to("cuda")
        attention_mask = torch.stack(
            [curr_batch[_idx]["attention_mask"] for _idx in range(len(curr_batch))],
            dim=0,
        ).to("cuda")

        _this_timestep = [sample["this_timestep"] + 1 for sample in curr_batch]

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
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            with record_activations(actor, record_module_names) as recording:
                output = actor(
                    input_ids.to(actor.device),
                    attention_mask=attention_mask.to(actor.device),
                    position_ids=position_ids.to(actor.device),
                    return_dict=True,
                )

        recording = {
            layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording.items()
        }
        resid_stream = torch.stack([acts for acts in recording.values()], dim=1)[
            :, :, -1, :
        ].cpu()
        _all_resid_streams_this.append(resid_stream)

        response = torch.stack(
            [curr_batch[_idx]["response"][:450] for _idx in range(len(curr_batch))],
            dim=0,
        )
        mask_not = (response[:, :-1] == token_open) & (response[:, 1:] == token_not)
        batch_idx_not, timesteps_not = torch.where(mask_not)
        _input_ids = [
            curr_batch[_idx]["response"][: timesteps_not[_idx] + 1]
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
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            with record_activations(actor, _record_module_names) as recording:
                output = actor(
                    input_ids.to(actor.device),
                    attention_mask=attention_mask.to(actor.device),
                    position_ids=position_ids.to(actor.device),
                    return_dict=True,
                )

        recording = {
            layer_name: torch.cat(acts, dim=1) for layer_name, acts in recording.items()
        }
        resid_stream = torch.stack([acts for acts in recording.values()], dim=1)[
            :, :, -1, :
        ].cpu()
        _all_resid_streams_not.append(resid_stream)

    all_resid_streams_this = torch.cat(_all_resid_streams_this, dim=0)
    all_resid_streams_not = torch.cat(_all_resid_streams_not, dim=0)
    return all_resid_streams_this, all_resid_streams_not


# %%


def _turn_off_mlp(model, layer_idx, mlp_idxs):
    def hook(module, input, output):
        output[:, -1, mlp_idxs] = 0
        return output

    module = model.model.layers[layer_idx].mlp.hook_mlp_mid
    return module.register_forward_hook(hook)


# %%


def _set_mlp(model, layer_idx, mlp_idxs, constant):
    def hook(module, input, output):
        output[:, -1, mlp_idxs] = constant
        return output

    module = model.model.layers[layer_idx].mlp.hook_mlp_mid
    return module.register_forward_hook(hook)

# %%


def _add_o_proj_hook(model, layer_idx, head_idx):
    def hook(module, input, output):
        # output.shape: [batch, heads, seq, head_dim]
        output[:, :, head_idx, :] = 0
        return output

    module = model.model.layers[layer_idx].self_attn.hook_attn_out_per_head
    return module.register_forward_hook(hook)


# %%


def _get_top_value_vecs(actor, probe_model, k):
    value_vecs = get_mlp_value_vecs(actor)
    top_cos_scores = {0: {}, 1: {}}
    for target_label in [0, 1]:
        # for target_probe_layer in [35]:

        for target_probe_layer in range(18, 36):
            top_value_vecs = []
            target_probe = probe_model[target_probe_layer, :, target_label]
            _curr_value_vecs = value_vecs[target_probe_layer]

            cos_scores = cos(_curr_value_vecs, target_probe.unsqueeze(-1), dim=0)
            _topk = cos_scores.topk(k=k)
            _idxs = [x.item() for x in _topk.indices]
            top_cos_scores[target_label][target_probe_layer] = _idxs

    return top_cos_scores


def get_glu_intervene_resid_streams(
    actor, probe_model, samples, record_module_names, k, config
):
    remove_all_hooks(actor)

    top_cos_scores = _get_top_value_vecs(actor, probe_model, k)
    top_scores_0 = top_cos_scores[0]
    top_scores_1 = top_cos_scores[1]
    handles = []
    for layer, idx in top_scores_1.items():
        handles.append(_turn_off_mlp(actor, layer, idx))
    for layer, idx in top_scores_0.items():
        handles.append(_turn_off_mlp(actor, layer, idx))

    resid_stream = _get_resid_stream(
        actor, samples, record_module_names, config
    )
    print(resid_stream.shape)
    for hook in handles:
        hook.remove()
    return resid_stream


# %%


def get_attn_intervene_resid_streams(
    actor, samples, record_module_names, config
):
    remove_all_hooks(actor)
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
    handles = []
    for layer, head_idx in heads:
        handles.append(_add_o_proj_hook(actor, layer, head_idx))

    resid_stream = _get_resid_stream(
        actor, samples, record_module_names, config
    )
    for hook in handles:
        hook.remove()
    return resid_stream


# %%


def _get_top_idxs_and_probs(rs):
    dots = unembed_resid_streams(rs.cuda(), actor).cpu()
    probs = dots.softmax(dim=-1)
    probs = probs.mean(dim=1)

    top_idxs = probs.topk(k=4).indices
    top_probs = probs.topk(k=4).values
    top_idxs = top_idxs.squeeze()
    top_probs = top_probs.squeeze()
    return top_idxs, top_probs


def _get_topk_tokens(top_idxs, top_probs, tokenizer):
    translate_map = {
        "不符合": "Does not conform",
        "不行": "Not okay",
        "不含": "Does not contain",
        "不满意": "Not satisfied",
        "不合格": "Unqualified",
        "不合": "Incompatible",
        "这不是": "This is not",
        "据此": "According to this",
        "都不是": "None of these are",
        "这是一个": "This is a",
        "正好": "Exactly",
        "相符": "Matches",
        "符合": "Meets requirements",
        "符合条件": "Qualifies",
        "两位": "Two people",
        "俩": "Both",
        "重伤": "Serious injury",
        "不符": "Does not match",
    }
    tokens = [
        tokenizer.batch_decode(top_idxs[_idx, :], skip_special_tokens=True)
        for _idx in range(top_idxs.shape[0])
    ]
    hover_text = []
    for i in range(top_idxs.shape[0]):
        _toks = []
        for j in range(top_idxs.shape[1]):
            _tok = tokens[i][j].strip()
            tok = _tok + f"  ({top_probs[i, j]:.2f})"
            if _tok in translate_map:
                tok += f"<br>({translate_map.get(tokens[i][j], "")})"
            _toks.append(tok)
        hover_text.append(_toks)
    return hover_text


def _add_heatmap(fig, z, layer_names, hover_text, colorscale, row, col):
    fig.add_trace(
        go.Heatmap(
            z=z.detach().cpu().numpy(),
            zsmooth=False,
            y=layer_names,
            hoverinfo="text",
            text=hover_text,
            texttemplate="%{text}",
            textfont={"size": 22},
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            colorbar=dict(
                tickfont=dict(size=18),
            ),
        ),
        row=row,
        col=col,
    )


def plot_figure_1(
    rs_this,
    rs_not,
    hooked_this,
    hooked_not,
    actor,
    tokenizer,
    layer_names,
    output_filepath,
    k=4,
):
    top_idxs_this, top_probs_this = _get_top_idxs_and_probs(rs_this)
    hover_text_this = _get_topk_tokens(top_idxs_this, top_probs_this, tokenizer)

    top_idxs_not, top_probs_not = _get_top_idxs_and_probs(rs_not)
    hover_text_not = _get_topk_tokens(top_idxs_not, top_probs_not, tokenizer)

    top_idxs_hooked_this, top_probs_hooked_this = _get_top_idxs_and_probs(hooked_this)
    hover_text_hooked_this = _get_topk_tokens(
        top_idxs_hooked_this, top_probs_hooked_this, tokenizer
    )

    top_idxs_hooked_not, top_probs_hooked_not = _get_top_idxs_and_probs(hooked_not)
    hover_text_hooked_not = _get_topk_tokens(
        top_idxs_hooked_not, top_probs_hooked_not, tokenizer
    )

    green_colorscale = [
        [0.0, "rgb(255, 255, 255)"],
        [1.0, "rgb(0, 100, 0)"],
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            '(A) Original Averaged LogitLens ("this works")',
            '(B) Original Averaged LogitLens ("not X")',
            '(C) Intervene GLU Neurons: "this" to "not"',
            '(D) Intervene Attn Heads: "this" to "not"',
        ),
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
    )

    _add_heatmap(
        fig, top_probs_this, layer_names, hover_text_this, green_colorscale, 1, 1
    )
    _add_heatmap(
        fig, top_probs_not, layer_names, hover_text_not, green_colorscale, 1, 2
    )
    _add_heatmap(
        fig,
        top_probs_hooked_this,
        layer_names,
        hover_text_hooked_this,
        green_colorscale,
        2,
        1,
    )
    _add_heatmap(
        fig,
        top_probs_hooked_not,
        layer_names,
        hover_text_hooked_not,
        green_colorscale,
        2,
        2,
    )

    fig.update_layout(
        width=1800,
        height=1500,
        xaxis_title="",
        xaxis2_title="",
        xaxis3_title="Top K",
        xaxis4_title="Top K",
    )
    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=22), tickvals=[0, 1, 2, 3], ticktext=["1", "2", "3", "4"]
        ),
        xaxis2=dict(
            tickfont=dict(size=22), tickvals=[0, 1, 2, 3], ticktext=["1", "2", "3", "4"]
        ),
        xaxis3=dict(
            tickfont=dict(size=22), tickvals=[0, 1, 2, 3], ticktext=["1", "2", "3", "4"]
        ),
        xaxis4=dict(
            tickfont=dict(size=22), tickvals=[0, 1, 2, 3], ticktext=["1", "2", "3", "4"]
        ),
        yaxis=dict(tickfont=dict(size=22)),
        yaxis3=dict(tickfont=dict(size=22)),
        yaxis_title_font=dict(size=22),
        yaxis3_title_font=dict(size=22),
        yaxis2=dict(showticklabels=False),  # Hides tick labels for subplot 2
        yaxis4=dict(showticklabels=False),  # Hides tick labels for subplot 2
        xaxis3_title_font=dict(size=22),
        xaxis4_title_font=dict(size=22),
    )

    # Update subplot title font size
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=28)  # Adjust size here

    fig.write_html(output_filepath.replace(".pdf", ".html"))
    fig.write_image(output_filepath)


# %%

plot_layer = 27
_record_module_names = record_module_names[plot_layer:]

# %%

resid_streams_this, resid_streams_not = get_orig_resid_streams(
    actor, samples, _record_module_names, config
)

# %%

hooked_resid_streams_mlp = get_glu_intervene_resid_streams(
    actor, probe_model, samples, _record_module_names, 50, config
)

# %%

hooked_resid_streams_attn = get_attn_intervene_resid_streams(
    actor, samples, _record_module_names, config
)

# %%

layer_names = [
    "L " + str(int(x.replace("model.layers.", "")) + 1)
    for x in _record_module_names
]
plot_figure_1(
    resid_streams_this,
    resid_streams_not,
    hooked_resid_streams_mlp,
    hooked_resid_streams_attn,
    actor,
    tokenizer,
    layer_names,
    "paper_figure_logitlens.pdf",
    k=4,
)
