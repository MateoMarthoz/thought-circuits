"""
Attention-level splice: clean forward (batch index 0) with perturbed K/V on
selected token spans from a parallel full-sequence perturbed run (batch index 1).

This is the same mechanism as ``rank_steps_see_corrupt``: post-source queries
attend with **perturbed** keys/values on the source span(s). It is **not** the
repo's *see clean* visibility regime, where intermediate positions between source
and target are computed as if the CoT stayed on the clean branch.
"""

from typing import Any, List, Set, Tuple

import math

import torch
import torch.nn.functional as F

from causal_cot.rope_utils import apply_rotary_pos_emb, repeat_kv


def query_mix_start_for_kv_substitution(
    candidate_step_idx: int,
    substituted_step_indices: Set[int],
    step_token_ranges: List[Tuple[int, int]],
) -> int:
    """
    First query position at which mixed (spliced) K/V attention is used.

    If only the candidate step is substituted (no prior prunes), returns that
    step's **end** token index so queries strictly after the source match
    ``rank_steps_see_corrupt`` (post-source reads corrupt K/V on the source).

    If prior-pruned steps are also substituted, returns the **minimum start**
    among all substituted spans so every query that can causally attend to any
    substituted key position uses mixed K/V (no clean reads of pruned steps).
    """
    if candidate_step_idx not in substituted_step_indices:
        raise ValueError(
            f"candidate_step_idx {candidate_step_idx} not in {substituted_step_indices}"
        )
    prior = substituted_step_indices - {candidate_step_idx}
    if not prior:
        return step_token_ranges[candidate_step_idx][1]
    return min(step_token_ranges[j][0] for j in substituted_step_indices)


class PerturbedKVPostSourceContext:
    """
    Batch size 2: index 0 = clean, index 1 = perturbed (full sequence, aligned length).

    For key positions in ``source_spans``, K/V are taken from the perturbed branch.
    Attention for queries in ``[0, query_mix_start)`` is fully clean; for
    ``[query_mix_start, q_len)`` it is recomputed with mixed K/V (all listed spans
    spliced). See ``query_mix_start_for_kv_substitution`` for multi-step policy.
    """

    def __init__(
        self,
        model,
        source_spans: List[Tuple[int, int]],
        query_mix_start: int,
    ):
        self.model = model
        self.source_spans = source_spans
        self.query_mix_start = query_mix_start
        self._patches: List[Tuple[Any, str, Any]] = []

    def _make_patched_forward(self, attn_module):
        model = self.model
        source_spans = self.source_spans
        query_mix_start = self.query_mix_start
        original_forward = attn_module.original_forward

        def patched_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ):
            config = attn_module.config
            device = hidden_states.device
            bsz, q_len, _ = hidden_states.size()
            if bsz != 2:
                return original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            num_heads = config.num_attention_heads
            num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
            head_dim = config.hidden_size // num_heads
            num_key_value_groups = num_heads // num_key_value_heads

            query_states = attn_module.q_proj(hidden_states)
            key_states = attn_module.k_proj(hidden_states)
            value_states = attn_module.v_proj(hidden_states)
            query_states = query_states.view(
                bsz, q_len, num_heads, head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, num_key_value_heads, head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, num_key_value_heads, head_dim
            ).transpose(1, 2)

            if position_ids is None:
                position_ids = torch.arange(
                    q_len, dtype=torch.long, device=device
                ).unsqueeze(0).expand(bsz, -1)
            else:
                position_ids = position_ids.to(device)
            rotary_emb = getattr(model.model, "rotary_emb", None)
            if rotary_emb is not None and callable(rotary_emb):
                cos, sin = rotary_emb(value_states, position_ids=position_ids)
                cos, sin = cos.to(device), sin.to(device)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            if past_key_value is not None:
                key_states = torch.cat(
                    [past_key_value[0].to(device), key_states], dim=2
                )
                value_states = torch.cat(
                    [past_key_value[1].to(device), value_states], dim=2
                )
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

            K_clean, K_pert = key_states[0:1], key_states[1:2]
            V_clean, V_pert = value_states[0:1], value_states[1:2]

            K_mixed = K_clean.clone()
            V_mixed = V_clean.clone()
            for src_start, src_end in source_spans:
                K_mixed[:, :, src_start:src_end, :] = K_pert[:, :, src_start:src_end, :]
                V_mixed[:, :, src_start:src_end, :] = V_pert[:, :, src_start:src_end, :]

            Q_clean = query_states[0:1]
            scale = math.sqrt(head_dim)

            attn_w_clean = torch.matmul(Q_clean, K_clean.transpose(2, 3)) / scale
            if attention_mask is not None:
                attn_w_clean = attn_w_clean + attention_mask[0:1]
            else:
                causal = torch.tril(
                    torch.ones((q_len, q_len), device=device, dtype=torch.bool)
                )
                attn_w_clean.masked_fill_(
                    ~causal.unsqueeze(0).unsqueeze(0), float("-inf")
                )
            attn_w_clean = F.softmax(attn_w_clean, dim=-1, dtype=torch.float32).to(
                Q_clean.dtype
            )
            attn_out_full = torch.matmul(attn_w_clean, V_clean)

            mix_start = max(0, min(query_mix_start, q_len))
            if mix_start < q_len:
                Q_tail = Q_clean[:, :, mix_start:q_len, :]
                attn_w_tail = torch.matmul(Q_tail, K_mixed.transpose(2, 3)) / scale
                if attention_mask is not None:
                    attn_w_tail = attn_w_tail + attention_mask[
                        0:1, :, mix_start:q_len, :
                    ]
                else:
                    full_causal = torch.tril(
                        torch.ones((q_len, q_len), device=device, dtype=torch.bool)
                    )
                    attn_w_tail.masked_fill_(
                        ~full_causal[mix_start:q_len, :].unsqueeze(0).unsqueeze(0),
                        float("-inf"),
                    )
                attn_w_tail = F.softmax(attn_w_tail, dim=-1, dtype=torch.float32).to(
                    Q_clean.dtype
                )
                attn_out_full[:, :, mix_start:q_len, :] = torch.matmul(
                    attn_w_tail, V_mixed
                )

            attn_out_full = attn_module.o_proj(
                attn_out_full.transpose(1, 2).contiguous().reshape(1, q_len, -1)
            )

            out_pert = original_forward(
                hidden_states[1:2],
                attention_mask=attention_mask[1:2] if attention_mask is not None else None,
                position_ids=position_ids[1:2] if position_ids is not None else None,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            return (
                torch.cat(
                    [attn_out_full, out_pert[0] if isinstance(out_pert, tuple) else out_pert],
                    dim=0,
                ),
                None,
            )

        return patched_forward

    def __enter__(self):
        for layer in self.model.model.layers:
            attn = layer.self_attn
            if not hasattr(attn, "original_forward"):
                attn.original_forward = attn.forward
            patched = self._make_patched_forward(attn)
            self._patches.append((attn, "forward", attn.forward))
            attn.forward = patched
        return self

    def __exit__(self, *_):
        for attn, attr, original in self._patches:
            setattr(attn, attr, original)
        self._patches.clear()
