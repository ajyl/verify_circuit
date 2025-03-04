from typing import Callable, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from fancy_einsum import einsum
import einops
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2MLP,
    apply_rotary_pos_emb,
    repeat_kv,
)


class HookPoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = None

    def forward(self, x):
        return x


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    # value_states.shape: [batch, heads, seq, head_dim]
    value_states = module.hook_value_states_post_attn(value_states)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def hooked_forward_attention(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    # attn_weights: [batch, heads, seq (query), seq (key)]
    # attn_output: [batch, seq, heads, head_dim]
    attn_weights = self.hook_attn_pattern(attn_weights)
    # attn_output_reshaped: [batch, seq, d_model (heads * head_dim)]
    #attn_output_reshaped = attn_output.reshape(*input_shape, -1).contiguous()

    W_O = self.o_proj.weight #.clone()
    # [heads, d_head, d_model]
    W_O = einops.rearrange(W_O, "m (n h)->n h m", n=self.config.num_attention_heads)
    # self.o_proj: [d_model, d_model]
    #attn_output_final = self.hook_o_proj(self.o_proj(attn_output_reshaped))
    attn_output_per_head = einsum(
        "batch seq heads d_head, heads d_head d_model -> batch seq heads d_model",
        attn_output,
        W_O,
    )
    # [batch seq n_heads d_model]
    attn_output_per_head = self.hook_attn_out_per_head(attn_output_per_head)
    attn_output_final = attn_output_per_head.sum(dim=2)
    return attn_output_final, attn_weights


def hooked_forward_mlp(self, x):
    self.mlp_mid = self.hook_mlp_mid(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    down_proj = self.down_proj(self.mlp_mid)
    return down_proj


def hooked_forward_decoder_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # necessary, but kept here for BC
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states
    hidden_states = self.hook_resid_mid(hidden_states)

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs


def _convert_to_hooked_model(module):
    for child in module.children():

        if isinstance(child, Qwen2Attention):
            child.forward = hooked_forward_attention.__get__(child, Qwen2Attention)

        if isinstance(child, Qwen2MLP):
            child.forward = hooked_forward_mlp.__get__(child, Qwen2MLP)

        if isinstance(child, Qwen2DecoderLayer):
            child.forward = hooked_forward_decoder_layer.__get__(
                child, Qwen2DecoderLayer
            )

        _convert_to_hooked_model(child)


def convert_to_hooked_model(model):
    """
    This function sets up a hook for the model's forward pass.
    It modifies the output of the model based on a steering vector.
    """
    for layer in model.model.layers:
        layer.hook_resid_mid = HookPoint()

        layer.self_attn.hook_attn_pattern = HookPoint()
        layer.self_attn.hook_value_states_post_attn = HookPoint()
        layer.self_attn.hook_o_proj = HookPoint()
        layer.self_attn.hook_attn_out_per_head = HookPoint()

        layer.mlp.hook_mlp_mid = HookPoint()

    _convert_to_hooked_model(model)
