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
import re
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
import matplotlib.pyplot as plt

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


model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
with torch.device("cuda:0"):
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

# %%

model_path = os.path.join(base_dir, "checkpoints/TinyZero/v4/actor/global_step_300")
with torch.device("cuda:1"):
    actor = AutoModelForCausalLM.from_pretrained(model_path)

# %%

convert_to_hooked_model(base_model)
convert_to_hooked_model(actor)


# %%

samples = torch.load(os.path.join(base_dir, "data/countdown/test_set2.pt"))


# %%


def _add_o_proj_hook(model, layer_idx, head_idx):
    def hook(module, input, output):
        # output.shape: [batch, heads, seq, head_dim]
        output[:, :, head_idx, :] = 0
        return output

    module = model.model.layers[layer_idx].self_attn.hook_attn_out_per_head
    return module.register_forward_hook(hook)


# %%


def run_base_mechanism(base_model, sample_data, hook_attn_heads, pos="yes", neg="no"):
    yes_token = tokenizer.encode(" " + pos.capitalize())[0]
    yes_token2 = tokenizer.encode(pos.capitalize())[0]
    yes_token3 = tokenizer.encode(" " + pos)[0]
    yes_token4 = tokenizer.encode(pos)[0]
    no_token = tokenizer.encode(" " + neg.capitalize())[0]
    no_token2 = tokenizer.encode(neg.capitalize())[0]
    no_token3 = tokenizer.encode(" " + neg)[0]
    no_token4 = tokenizer.encode(neg)[0]

    all_yes_probs = []
    all_no_probs = []
    all_yes_probs_normalized = []
    all_no_probs_normalized = []
    all_yes_probs_hooked = []
    all_no_probs_hooked = []
    all_yes_probs_hooked_normalized = []
    all_no_probs_hooked_normalized = []
    preds = []
    for sample in tqdm(sample_data):
        nums = sample["nums"]
        target = sample["target"]

        # base_prompt = f"I have been asked to use use the numbers {nums}, to create an equation that equals {target}. I can use basic arithmetic operations (+, -, *, /) and each number can only be used once."
        # base_prompt += " Here is my attempt:\n"
        # base_prompt += " Let me solve this step by step.\n<think> We have the numbers {nums}. We need to use these numbers to make an equation that equals {target} using basic arithmetic operations. "
        base_prompt = f"A conversation between User and Assistant. "
        base_prompt += f"The User is given a set of numbers {nums} to create an equation that equals {target}. "
        base_prompt += f"The User can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        base_prompt += f"The following is the User's attempt:"

        output_text = sample["output_text"]
        attempts = re.findall(r"^- (.*)", output_text, re.MULTILINE)

        # print("=====================================")
        # print(f"Nums: {nums}")
        # print(f"Target: {target}")
        for attempt in [attempts[-1]]:
            attempt = re.sub(r"\s*\([^)]*\)", "", attempt)

            # prompt = base_prompt + f"\n- {attempt}"
            # prompt += "\n Is this attempt correct? Answer only in 'No' or 'Yes'."
            # prompt += "\n Answer:"
            prompt = base_prompt + f"\n{attempt}"
            #prompt += "\nYour job as an Assistant is to verify if this attempt is good ({pos}) or bad ({neg}). "
            prompt += "\nYour job as an Assistant is to verify if this attempt is correct or not. "
            prompt += f"Is the User's attempt correct? Answer only in '{pos.capitalize()}' or '{neg.capitalize()}'."
            prompt += "\nAssistant: Let me think about this step by step. "
            prompt += f"The User is given the set of numbers {nums} and must create an equation that equals {target}. "
            prompt += f"The User's attempt is {attempt}. "
            # prompt += f"The User's attempt is {attempt}. "
            prompt += f"Therefore, the final answer is"

            _input = tokenizer(prompt, return_tensors="pt").to(base_model.device)
            output = base_model(**_input)

            probs = output.logits[0, -1].softmax(dim=-1)
            next_pred = output.logits[0, -1].argmax().item()
            yes_prob = (
                probs[yes_token].item()
                + probs[yes_token2].item()
                + probs[yes_token3].item()
                + probs[yes_token4].item()
            )
            no_prob = (
                probs[no_token].item()
                + probs[no_token2].item()
                + probs[no_token3].item()
                + probs[no_token4].item()
            )
            # print(f"{yes_prob} vs. {no_prob}")
            all_yes_probs.append(yes_prob)
            all_no_probs.append(no_prob)

            all_yes_probs_normalized.append(yes_prob / (yes_prob + no_prob))
            all_no_probs_normalized.append(no_prob / (yes_prob + no_prob))

            handles = []
            for head_layer, head_idx in hook_attn_heads:
                handles.append(_add_o_proj_hook(base_model, head_layer, head_idx))

            output = base_model(**_input)

            for handle in handles:
                handle.remove()

            probs = output.logits[0, -1].softmax(dim=-1)
            yes_prob = (
                probs[yes_token].item()
                + probs[yes_token2].item()
                + probs[yes_token3].item()
                + probs[yes_token4].item()
            )
            no_prob = (
                probs[no_token].item()
                + probs[no_token2].item()
                + probs[no_token3].item()
                + probs[no_token4].item()
            )

            all_yes_probs_hooked.append(yes_prob)
            all_no_probs_hooked.append(no_prob)

            all_yes_probs_hooked_normalized.append(yes_prob / (yes_prob + no_prob))
            all_no_probs_hooked_normalized.append(no_prob / (yes_prob + no_prob))

            preds.append(probs.topk(k=10).indices)

    return (
        np.mean(all_yes_probs_normalized),
        np.mean(all_yes_probs_hooked_normalized),
        np.mean(all_no_probs_normalized),
        np.mean(all_no_probs_hooked_normalized),
    )


# %%


def plot(yes_before, yes_after, no_before, no_after, output_filename):

    categories = ["Original", "Intervened"]
    before = [yes_before, no_before]
    after = [yes_after, no_after]
    values = [yes_before, no_before, yes_after, no_after]
    bar_labels = ["Yes", "No", "Yes", "No"]
    group_labels = ["Original", "", "Intervened", ""]

    x = range(len(values))
    # width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3))

    width = 0.35
    x_positions = [0, width * 2 + 0.1, 2, width * 8]
    bars = ax.bar(
        x_positions, values, color=["tab:blue", "tab:orange", "tab:blue", "tab:orange"]
    )
    # bars1 = ax.bar([p - width / 2 for p in x], before, width, label="Before Intervention")
    # bars2 = ax.bar([p + width / 2 for p in x], after, width, label="After Intervention")

    for idx, bar in enumerate(bars):
        yval = bar.get_height()
        offset = 0
        if idx in [0, 3]:
            offset = -0.13
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + offset,
            round(float(yval), 2),
            ha="center",
            va="bottom",
            fontsize=16,
        )

    # Labels and styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, fontsize=16)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.tick_params(axis="x", length=0)

    ax2.spines["top"].set_visible(False)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.15))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.set_xticks([0.4, 2.35])
    ax2.set_xticklabels(["Original", "Intervened"], fontsize=16)
    # ax.legend()

    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=16)
    ax.set_title("Intervention on Base Model", fontsize=16)

    ax.set_ylabel("(Normalized) Probs.", fontsize=16)
    plt.tight_layout()
    fig.savefig(f"{output_filename}.png", dpi=300)
    fig.savefig(f"{output_filename}.pdf", dpi=300)
    plt.show()


# %%


def get_top_value_vecs(actor, probe_model, value_vecs, k):
    top_cos_scores = {0: [], 1: []}
    cos = F.cosine_similarity
    for target_label in [0, 1]:
        for target_probe_layer in range(18, 36):
            target_probe = probe_model[target_probe_layer, :, target_label]

            for layer_idx in range(0, target_probe_layer + 1):
                cos_scores = cos(
                    value_vecs[layer_idx].to(target_probe.device),
                    target_probe.unsqueeze(-1),
                    dim=0,
                )
                _topk = cos_scores.topk(k=k)
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
            sorted_scores_0.append(_pair)

    _unique = set()
    sorted_scores_1 = []
    for entry in _sorted_scores_1:
        _pair = (entry[3], entry[1])
        if _pair not in _unique:
            _unique.add(_pair)
            sorted_scores_1.append(_pair)

    return sorted_scores_0, sorted_scores_1

# %%

def get_prev_token_heads(actor, samples, batch_size, thresh=0.1):
    attn_pattern = get_attn_density_for_target(actor, samples, batch_size)
    top_values, top_idxs = torch.topk(attn_pattern.flatten(), 50)
    top_idxs = np.array(np.unravel_index(top_idxs.cpu().numpy(), attn_pattern.shape)).T
    prev_token_heads = top_idxs[
        (top_values > thresh).nonzero().squeeze().cpu()
    ].squeeze()
    return prev_token_heads, attn_pattern

# %%


def get_occurrence_idxs(hay, needle):
    window_size = needle.shape[0]
    hay = hay.unfold(0, window_size, 1)
    mask = (hay == needle).all(dim=1)
    offset = window_size - 1
    match_idxs = mask.nonzero(as_tuple=True)[0] + offset
    return match_idxs


@torch.no_grad()
def get_attn_density_for_target(actor, samples, batch_size):
    n_layers = 36
    record_module_names = [
        f"model.layers.{idx}.self_attn.hook_attn_pattern" for idx in range(n_layers)
    ]
    test_size = len(samples)
    _all_attn_pattern = []
    all_recording = {}
    cutoff = (
        tokenizer(" Let's try different", return_tensors="pt")["input_ids"]
        .squeeze()
        .to(actor.device)
    )
    all_attn_density = []
    for batch_idx in tqdm(range(0, test_size, batch_size)):
        curr_batch = samples[batch_idx : batch_idx + batch_size]
        input_ids = torch.stack(
            [curr_batch[_idx]["input_ids"] for _idx in range(len(curr_batch))], dim=0
        ).to(actor.device)
        attention_mask = torch.stack(
            [curr_batch[_idx]["attention_mask"] for _idx in range(len(curr_batch))],
            dim=0,
        ).to(actor.device)

        _this_timestep = [sample["this_timestep"] for sample in curr_batch]

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
        input_ids = torch.stack(padded_input_ids, dim=0).to(actor.device)
        attention_mask = input_ids != tokenizer.pad_token_id
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with record_activations(actor, record_module_names) as recording:
            output = actor(
                input_ids.to(actor.device),
                attention_mask=attention_mask.to(actor.device),
                position_ids=position_ids.to(actor.device),
                return_dict=True,
            )

        # [layers, batch, heads, seq]
        _attn_pattern = torch.stack(
            [
                recording[f"model.layers.{layer_idx}.self_attn.hook_attn_pattern"][0][
                    :, :, -1, :
                ]
                for layer_idx in range(n_layers)
            ]
        )
        attn_density = []
        for _idx in range(len(curr_batch)):
            target_tokens = tokenizer(
                str(curr_batch[_idx]["target"]), return_tensors="pt"
            )["input_ids"].squeeze()
            _context = input_ids[_idx]
            cutoff_idx = get_occurrence_idxs(_context, cutoff)
            curr_context = _context[: cutoff_idx[0]]
            match_idxs = get_occurrence_idxs(
                curr_context, target_tokens.to(curr_context.device)
            ).to(actor.device)

            # [layers, heads]
            density = _attn_pattern[:, _idx, :, match_idxs].sum(dim=-1)
            attn_density.append(density)

    all_attn_density = torch.stack(attn_density, dim=0)
    return all_attn_density.mean(dim=0)

# %%

probe_path = os.path.join(base_dir, "probe_checkpoints/v2/probe.pt")
probe_model = torch.load(probe_path).detach().cuda()

# %%

value_vecs_actor = get_mlp_value_vecs(actor)
value_vecs_base = get_mlp_value_vecs(base_model).cpu()

# %%

top_scores_0, top_scores_1 = get_top_value_vecs(
    actor, probe_model, value_vecs_actor, k=50
)



# %%

dev_size = 50
batch_size = 4

# %%

actor_prev_token_heads, actor_attn_pattern = get_prev_token_heads(
    actor, samples[:dev_size], batch_size, thresh=0.05
)

# %%

base_prev_token_heads, base_attn_pattern = get_prev_token_heads(
    base_model, samples[:dev_size], batch_size, thresh=0.1
)


# %%

word_pairs = [
    ["yes", "no"],
    ["success", "failure"],
    ["correct", "incorrect"],
    ["good", "bad"],
    ["true", "false"],
    ["right", "wrong"],
    ["valid", "invalid"],
    ["positive", "negative"],
    ["acceptable", "unacceptable"],
    ["appropriate", "inappropriate"],
    ["satisfactory", "unsatisfactory"],
    ["favorable", "unfavorable"],
]
interv_heads = actor_prev_token_heads
#interv_heads = [
#    [17, 14],
#    [17, 10],
#    [17, 11],
#]


# %%

all_pos_pre = []
all_pos_post = []
all_neg_pre = []
all_neg_post = []
for word_pair in word_pairs:
    pos = word_pair[0]
    neg = word_pair[1]

    pos_pre, pos_post, neg_pre, neg_post = run_base_mechanism(
        base_model, samples[:dev_size], interv_heads, pos=pos, neg=neg
    )
    print(f"{pos} vs. {neg}")
    print(f"Pos: {pos_pre} {pos_post}")
    print(f"Neg: {neg_pre} {neg_post}")
    all_pos_pre.append(pos_pre)
    all_pos_post.append(pos_post)
    all_neg_pre.append(neg_pre)
    all_neg_post.append(neg_post)

print("Final")
print(np.mean(all_pos_pre))
print(np.mean(all_pos_post))
print(np.mean(all_neg_pre))
print(np.mean(all_neg_post))


# %%

# Experiment for hacking the weights of base model.

with torch.device("cuda:0"):
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
convert_to_hooked_model(base_model)

base_state_dict = base_model.state_dict()
actor_state_dict = actor.state_dict()

alpha = 2
n_heads = base_model.config.num_attention_heads
n_kv_heads = base_model.config.num_key_value_heads
n_kv_groups = n_heads // n_kv_heads
hacked_state_dict = base_state_dict.copy()
interv_heads = [
    [17, 14],
    [17, 10],
    [17, 11],
]
for head in interv_heads:
    layer_idx = head[0]
    head_idx = head[1]

    W_O_actor = actor_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
    W_V_actor = actor_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
    W_K_actor = actor_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
    W_Q_actor = actor_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
    W_O_actor = einops.rearrange(W_O_actor, "m (n h)->n h m", n=n_heads)
    W_V_actor = einops.rearrange(W_V_actor, "(n h) m->n m h", n=n_kv_heads)
    W_K_actor = einops.rearrange(W_K_actor, "(n h) m->n m h", n=n_kv_heads)
    W_Q_actor = einops.rearrange(W_Q_actor, "(n h) m->n m h", n=n_heads)

    w_o = W_O_actor[head_idx].to(base_model.device)
    w_q = W_Q_actor[head_idx].to(base_model.device)
    w_v = W_V_actor[0].to(base_model.device)
    w_k = W_K_actor[0].to(base_model.device)
    if head_idx >= n_kv_groups:
        w_v = W_V_actor[1].to(base_model.device)
        w_k = W_K_actor[1].to(base_model.device)

    W_O_base = base_state_dict[
        f"model.layers.{layer_idx}.self_attn.o_proj.weight"
    ].clone()
    W_V_base = base_state_dict[
        f"model.layers.{layer_idx}.self_attn.v_proj.weight"
    ].clone()
    W_K_base = base_state_dict[
        f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    ].clone()
    W_Q_base = base_state_dict[
        f"model.layers.{layer_idx}.self_attn.q_proj.weight"
    ].clone()
    W_O_base = einops.rearrange(W_O_base, "m (n h)->n h m", n=n_heads)
    W_V_base = einops.rearrange(W_V_base, "(n h) m->n m h", n=n_kv_heads)
    W_K_base = einops.rearrange(W_K_base, "(n h) m->n m h", n=n_kv_heads)
    W_Q_base = einops.rearrange(W_Q_base, "(n h) m->n m h", n=n_heads)

    w_o_diff = w_o - W_O_base[head_idx]
    w_q_diff = w_q - W_Q_base[head_idx]
    insert_idx = 0
    if head_idx >= n_kv_groups:
        insert_idx = 1
    w_v_diff = w_v - W_V_base[insert_idx]
    w_k_diff = w_k - W_K_base[insert_idx]

    W_O_base[head_idx] = W_O_base[head_idx] + (alpha * w_o_diff)
    W_Q_base[head_idx] = W_Q_base[head_idx] + (alpha * w_q_diff)
    W_V_base[insert_idx] = W_V_base[insert_idx] + (alpha * w_v_diff)
    W_K_base[insert_idx] = W_K_base[insert_idx] + (alpha * w_k_diff)

    hacked_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = (
        einops.rearrange(W_O_base, "n h m -> m (n h)", n=n_heads)
    )
    hacked_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = (
        einops.rearrange(W_Q_base, "n m h -> (n h) m", n=n_heads)
    )
    hacked_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = (
        einops.rearrange(W_V_base, "n m h -> (n h) m", n=n_kv_heads)
    )
    hacked_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = (
        einops.rearrange(W_K_base, "n m h -> (n h) m", n=n_kv_heads)
    )

# %%

base_model.load_state_dict(hacked_state_dict)

# %%

convert_to_hooked_model(base_model)
all_pos_pre = []
all_pos_post = []
all_neg_pre = []
all_neg_post = []
for word_pair in word_pairs:
    pos = word_pair[0]
    neg = word_pair[1]

    pos_pre, pos_post, neg_pre, neg_post = run_base_mechanism(
        base_model, samples[:dev_size], interv_heads, pos=pos, neg=neg
    )
    print(f"Pos: {pos_pre} {pos_post}")
    print(f"Neg: {neg_pre} {neg_post}")
    all_pos_pre.append(pos_pre)
    all_pos_post.append(pos_post)
    all_neg_pre.append(neg_pre)
    all_neg_post.append(neg_post)

print("Final")
print(np.mean(all_pos_pre))
print(np.mean(all_pos_post))
print(np.mean(all_neg_pre))
print(np.mean(all_neg_post))


# %%
