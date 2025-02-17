from typing import Callable, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ...configuration_utils import PretrainedConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_principle import PrincipleConfig


logger = logging.get_logger(__name__)

class PrincipleMLP(nn.Module):
    def __init__(self, config: PrincipleConfig, layer_idx):
        super().__init__()
        self.config = config
        self.in_size = config.layer_sizes[layer_idx]
        # out dimension = next in dimension
        self.out_size = config.layer_sizes[layer_idx + 1]
        
        self.mlp_upscale_size = config.mlp_upscale_size
        self.gate_proj = nn.Linear(self.in_size, self.mlp_upscale_size, bias=False)
        self.up_proj = nn.Linear(self.in_size, self.mlp_upscale_size, bias=False)
        self.down_proj = nn.Linear(self.mlp_upscale_size, self.out_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        in: [b, l, in_size]
        out: [b, l, out_size]
        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class PrincipleAttention(nn.Module):
    def __init__(self, config: PrincipleConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.layer_sizes[layer_idx] // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.layer_sizes[layer_idx], config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.layer_sizes[layer_idx], config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.layer_sizes[layer_idx], config.num_key_value_heads * self.head_dim, bias=True)
        # Current design pattern is that, the dimension-changing operation is purely on MLP. so that this o_proj will remain the input/output dimension
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.layer_sizes[layer_idx], bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
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
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class PrincipleRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        PrincipleRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class PrincipleDecoderLayer(nn.Module):
    def __init__(self, config: PrincipleConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.layer_sizes[layer_idx]
        self.self_attn = PrincipleAttention(config=config, layer_idx=layer_idx)
        self.mlp = PrincipleMLP(config, layer_idx)
        self.input_layernorm = PrincipleRMSNorm(config.layer_sizes[layer_idx], eps=config.rms_norm_eps)
        self.post_attention_layernorm = PrincipleRMSNorm(config.layer_sizes[layer_idx], eps=config.rms_norm_eps)
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
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

        # Fully Connected(original version)
        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states

        # Fully Connected(principle version)
        residual = self.mlp(hidden_states) # map from input_dim to output_dim to enable residual connection
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class PrincipleModel(LlamaModel):
    def __init__(self, config: PrincipleConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.layer_sizes[0], self.padding_idx)
        self.norm = PrincipleRMSNorm(config.layer_sizes[0], eps=config.rms_norm_eps)


class PrincipleForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.layer_sizes[config.num_hidden_layers], config.vocab_size, bias=False)


class PrincipleForSequenceClassification(LlamaForSequenceClassification):
    pass


class PrincipleForTokenClassification(LlamaForTokenClassification):
    pass


class PrincipleForQuestionAnswering(LlamaForQuestionAnswering):
    pass
