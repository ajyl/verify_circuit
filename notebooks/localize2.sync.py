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
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
import torch
import torch.nn.functional as F
from fancy_einsum import einsum
import einops
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from src.record_utils import record_activations, get_module, untuple_tensor

# from src.utils import load_model
from src.HookedQwen import convert_to_hooked_model
from src.rl_dataset import RLHFDataset

from IPython.core.debugger import set_trace


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


def unembed_text(vector, actor, tokenizer, k=10):
    lm_head = actor.lm_head.weight
    norm = actor.model.norm
    top_k = unembed(norm(vector), lm_head, k=k)
    return tokenizer.batch_decode(top_k, skip_special_tokens=True)


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


def _add_o_proj_hook(model, layer_idx, head_idx):
    def hook(module, input, output):
        # output.shape: [batch, heads, seq, head_dim]
        # output[:, :, head_idx, :] = output[:, :, head_idx, :] / 1e10
        output[:, :, head_idx, :] = 0
        return output

    # module = model.model.layers[layer_idx].self_attn.hook_o_proj
    module = model.model.layers[layer_idx].self_attn.hook_attn_out_per_head
    return module.register_forward_hook(hook)


def add_hooks(model, hook_config):
    handles = []
    for hook_module, layer, head_idx in hook_config:
        if hook_module == "attn_out":
            hook_func = _add_o_proj_hook
        elif hook_module == "mlp":
            hook_func = _turn_off_mlp
        handles.append(hook_func(model, layer, head_idx))
    return handles


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

config = {
    "model_path": os.path.join(
        base_dir, "checkpoints/TinyZero/v4/actor/global_step_300"
    ),
    "batch_size": 4,
    "max_prompt_length": 256,
    "max_response_length": 300,
    "n_layers": 36,
    "d_model": 2048,
    "seed": 42,
    "n_heads": 16,
    "probe_path": os.path.join(base_dir, "probe_checkpoints/v2/probe.pt"),
}

# %%

attn_heads = [
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

seed_all(config["seed"])
assert torch.cuda.is_available()

model_path = config["model_path"]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
with torch.device("cuda"):
    actor = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

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


def get_mlp_value_vecs(model):
    mlp_value_vecs = [layer.mlp.down_proj.weight for layer in model.model.layers]
    # [n_layers, d_mlp (11008), d_model (2048)]
    return torch.stack(mlp_value_vecs, dim=0)


def _get_top_value_vecs(actor, probe_model, value_vecs, k):
    top_cos_scores = {0: [], 1: []}
    cos = F.cosine_similarity
    for target_label in [0, 1]:
        for target_probe_layer in range(18, 36):
            target_probe = probe_model[target_probe_layer, :, target_label]

            for layer_idx in range(0, target_probe_layer + 1):
                cos_scores = cos(
                    value_vecs[layer_idx], target_probe.unsqueeze(-1), dim=0
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

probe_model = torch.load(config["probe_path"]).detach().cuda()

# %%
value_vecs = get_mlp_value_vecs(actor)
# %%


# %%


def _get_mlp_acts_inner(actor, samples, record_module_names, batch_size):
    test_size = len(samples)
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


def get_mlp_acts(actor, samples, neurons, batch_size):
    n_layers = 36
    record_module_names = [
        f"model.layers.{idx}.mlp.hook_mlp_mid" for idx in range(n_layers)
    ]
    # [batch, layers, d_mlp]
    acts = _get_mlp_acts_inner(actor, samples, record_module_names, batch_size)
    mlp_acts = []
    for neuron_idx in neurons:
        _acts = acts[:, neuron_idx[0], neuron_idx[1]]
        mlp_acts.append(_acts)
    mlp_acts = torch.stack(mlp_acts, dim=1)
    return mlp_acts


# %%

num_neurons = 200
dev_size = 60


def run(actor, samples, hook_config, batch_size, test_size=None):
    prompt = "open_parenthesis"
    assert prompt in ["orig", "open_parenthesis"]
    max_gen_length = 5
    if prompt == "open_parenthesis":
        max_gen_length = 5

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

        this_count += (
            (
                (hooked_output == token_this).any(dim=1)
                | (hooked_output == token_equals).any(dim=1)
            )
            .sum()
            .item()
        )

        mask = hooked_output[:, :-1] == token_open
        tokens_after_parenthesis = hooked_output[:, 1:][mask]
        generated_tokens2.update(tokens_after_parenthesis.tolist())

        # if len(set(tokens_after_parenthesis.tolist())) > 1:
        #    print("Hmm.")
        #    print(tokenizer.batch_decode(tokens_after_parenthesis))
        #    odd_batches.append(batch_idx)

        total += len(curr_batch)

    return num_not / total, this_count / total


# %%

for _idx, (_diff, _attn_head) in enumerate(_diffs):
    interv_heads = [("attn_out", x[0], x[1]) for x in attn_heads[: _idx + 1]]

    print(f"Len interv_heads: {len(interv_heads)}")
    remove_all_hooks(actor)
    print(run(actor, samples[:dev_size], interv_heads, config["batch_size"]))


# %%


def pad_and_concatenate(tensor_list):
    """
    Pads and concatenates a list of tensors along the given dimension.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to concatenate.

    Returns:
        torch.Tensor: Padded and concatenated tensor.
    """
    # Find the max size in the target dimension
    max_size = max([x.shape[-1] for x in tensor_list])

    # Pad each tensor to match max_size in the given dimension
    padded_tensors = []
    for tensor_idx, tensor in enumerate(tensor_list):
        zeros = torch.zeros(
            tensor.shape[0],
            tensor.shape[1],
            max_size,
            device=tensor_list[-1].device,
        )
        zeros[:, :, max_size - tensor.shape[-1] :] = tensor
        padded_tensors.append(zeros)

    attn_pattern = torch.cat(padded_tensors, dim=0)
    return attn_pattern


# %%


@torch.no_grad()
def get_attn_pattern(actor, samples, batch_size):
    n_layers = 36
    record_module_names = [
        f"model.layers.{idx}.self_attn.hook_attn_pattern" for idx in range(n_layers)
    ]
    test_size = len(samples)
    _all_attn_pattern = []
    all_recording = {}
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

        with record_activations(actor, record_module_names) as recording:
            output = actor(
                input_ids.to(actor.device),
                attention_mask=attention_mask.to(actor.device),
                position_ids=position_ids.to(actor.device),
                return_dict=True,
            )
        for key in recording:
            if key not in all_recording:
                all_recording[key] = []
            assert len(recording[key]) == 1
            all_recording[key].append(recording[key][0][:, :, -1, :].cpu())

    attn_pattern = torch.stack(
        [
            pad_and_concatenate(
                all_recording[f"model.layers.{layer_idx}.self_attn.hook_attn_pattern"]
            )
            for layer_idx in range(n_layers)
        ],
        dim=1,
    )
    return attn_pattern


# %%


def get_occurrence_idxs(hay, needle):
    window_size = needle.shape[0]
    hay = hay.unfold(0, window_size, 1)
    mask = (hay == needle).all(dim=1)
    offset = window_size - 1
    match_idxs = mask.nonzero(as_tuple=True)[0] + offset
    return match_idxs


# %%


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
        .to("cuda")
    )
    all_attn_density = []
    for batch_idx in tqdm(range(0, test_size, batch_size)):
        curr_batch = samples[batch_idx : batch_idx + batch_size]
        input_ids = torch.stack(
            [curr_batch[_idx]["input_ids"] for _idx in range(len(curr_batch))], dim=0
        ).to("cuda")
        attention_mask = torch.stack(
            [curr_batch[_idx]["attention_mask"] for _idx in range(len(curr_batch))],
            dim=0,
        ).to("cuda")

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
        input_ids = torch.stack(padded_input_ids, dim=0).to("cuda")
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
            )

            # [layers, heads]
            density = _attn_pattern[:, _idx, :, match_idxs].sum(dim=-1)
            attn_density.append(density)

    all_attn_density = torch.stack(attn_density, dim=0)
    return all_attn_density.mean(dim=0)


# %%

dev_size = 10
attn_pattern = get_attn_density_for_target(
    actor, samples[:dev_size], config["batch_size"]
)

# %%

top_values, top_idxs = torch.topk(attn_pattern.flatten(), 10)
print(top_values)
top_idxs = np.array(np.unravel_index(top_idxs.cpu().numpy(), attn_pattern.shape)).T
print(top_idxs)

# %%

dev_size = 60
for _idx in range(top_idxs.shape[0]):
    curr = top_idxs[: _idx + 1]
    interv_heads = [("attn_out", x[0], x[1]) for x in curr]

    print(f"Len interv_heads: {len(interv_heads)}")
    print(interv_heads)
    remove_all_hooks(actor)
    print(run(actor, samples[:dev_size], interv_heads, config["batch_size"]))


# %%

interv_heads = [
    ("attn_out", 17, 14),
    ("attn_out", 17, 10),
    ("attn_out", 17, 11),
]
remove_all_hooks(actor)
print(run(actor, samples[:dev_size], interv_heads, config["batch_size"]))

# %%


top_scores_0, top_scores_1 = _get_top_value_vecs(actor, probe_model, value_vecs, k=50)


# %%

n_layers = 36
n_heads = actor.config.num_attention_heads
n_kv_heads = actor.config.num_key_value_heads
n_kv_groups = n_heads // n_kv_heads
W_O = []
W_V = []
#for idx in range(n_layers):

    _W_O = actor.model.layers[idx].self_attn.o_proj.weight
    _W_O = einops.rearrange(_W_O, "m (n h)->n h m", n=n_heads)
    W_O.append(_W_O)

    _W_V = actor.model.layers[idx].self_attn.v_proj.weight
    _W_V = einops.rearrange(_W_V, "(n h) m->n m h", n=n_kv_heads)
    _W_V = torch.repeat_interleave(_W_V, dim=0, repeats=n_kv_groups)
    W_V.append(_W_V)



# [layers, heads, d_head, d_model]
W_O = torch.stack(W_O, dim=0)
W_V = torch.stack(W_V, dim=0)
print(W_O.shape)
print(W_V.shape)


# %%

OV = einsum(
    "layers heads d_head d_model, layers heads d_model d_head -> layers heads d_model",
    W_O,
    W_V,
)

# %%

subset_attn_heads = []
for attn_head in attn_heads:
    layer_idx = attn_head[0]
    head_idx = attn_head[1]
    _OV = OV[layer_idx, head_idx]
    subset_attn_heads.append(_OV)
_OV = torch.stack(subset_attn_heads, dim=0)

# %%

num_mlp_vecs = 200

# [N, d_model]
key_vecs_gate = torch.stack(
    [
        actor.model.layers[x[0]].mlp.gate_proj.weight[x[1]]
        for x in top_scores_1[:num_mlp_vecs]
    ],
    dim=0,
)

# [N, d_model]
key_vecs_up = torch.stack(
    [
        actor.model.layers[x[0]].mlp.up_proj.weight[x[1]]
        for x in top_scores_1[:num_mlp_vecs]
    ],
    dim=0,
)

# %%


act_fn = actor.model.layers[0].mlp.act_fn
dots_gate = einsum("N d_model, M d_model -> N M", key_vecs_gate, _OV)
acts = act_fn(dots_gate)

dots_up = einsum("N d_model, M d_model -> N M", key_vecs_up, _OV)

weights = (acts * dots_up).mean(dim=0)

# %%

top_values, top_idxs = torch.topk(weights, 10)
print(top_idxs)
print([attn_heads[x] for x in top_idxs])

# %%

for _idx in range(3, 10):
    _idxs = [attn_heads[x] for x in top_idxs[:_idx]]
    interv_heads = [("attn_out", x[0], x[1]) for x in _idxs]
    print(f"Len interv_heads: {len(interv_heads)}")
    print(interv_heads)
    remove_all_hooks(actor)
    print(run(actor, samples[:dev_size], interv_heads, config["batch_size"]))
