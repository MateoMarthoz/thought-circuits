"""Cumulative attention suppression used by the greedy ablation experiments.

Extracted without semantic changes from the dissertation-era
``run_backward_ablation.py`` implementation.
"""

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from causal_cot.rope_utils import apply_rotary_pos_emb, repeat_kv


class CumulativeAttnSuppressionContext:
    """
    Monkey-patches Qwen2Attention.forward to suppress attention from multiple
    source spans simultaneously. For each span (src_start, src_end), all tokens
    at absolute positions >= src_end cannot attend to positions [src_start, src_end).

    Supports optional past_key_values (KV prefix caching) where queries cover
    only a suffix of the sequence.
    """

    def __init__(self, model: torch.nn.Module, suppression_spans: List[Tuple[int, int]]):
        self.model = model
        self.suppression_spans = suppression_spans
        self._patches: List[Tuple[Any, str, Any]] = []

    def _make_patched_forward(self, attn_module: torch.nn.Module, layer_idx: int):
        model = self.model
        suppression_spans = self.suppression_spans
        _layer_idx = layer_idx

        def patched_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.Tensor] = None,
            **kwargs: Any,
        ):
            config = attn_module.config
            device = hidden_states.device
            bsz, q_len, _ = hidden_states.size()
            num_heads = config.num_attention_heads
            num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
            head_dim = config.hidden_size // num_heads
            num_key_value_groups = num_heads // num_key_value_heads

            query_states = attn_module.q_proj(hidden_states)
            key_states = attn_module.k_proj(hidden_states)
            value_states = attn_module.v_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            if position_ids is None:
                position_ids = torch.arange(q_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
            else:
                position_ids = position_ids.to(device)

            rotary_emb = getattr(model.model, "rotary_emb", None)
            if rotary_emb is not None and callable(rotary_emb):
                cos, sin = rotary_emb(value_states, position_ids=position_ids)
                cos, sin = cos.to(device), sin.to(device)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                cached_k = past_key_value.layers[_layer_idx].keys
                cached_v = past_key_value.layers[_layer_idx].values
                key_states = torch.cat([cached_k.to(device), key_states], dim=2)
                value_states = torch.cat([cached_v.to(device), value_states], dim=2)

            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)
            kv_len = key_states.shape[2]
            prefix_len = kv_len - q_len

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            key_positions = torch.arange(kv_len, device=device)
            query_abs = torch.arange(prefix_len, prefix_len + q_len, device=device)
            causal_mask = key_positions.unsqueeze(0) <= query_abs.unsqueeze(1)
            attn_weights.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            for src_start, src_end in suppression_spans:
                q_from = max(0, src_end - prefix_len)
                attn_weights[:, :, q_from:, src_start:src_end] = -1e4

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            return (attn_module.o_proj(attn_output), None)

        return patched_forward

    def __enter__(self) -> "CumulativeAttnSuppressionContext":
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn_module = layer.self_attn
            if not hasattr(attn_module, "original_forward"):
                attn_module.original_forward = attn_module.forward
            patched = self._make_patched_forward(attn_module, layer_idx)
            self._patches.append((attn_module, "forward", attn_module.forward))
            attn_module.forward = patched
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for attn_module, attr, original in self._patches:
            setattr(attn_module, attr, original)
        self._patches.clear()
