#!/usr/bin/env python3
"""
Greedy ablation: for each method (attention suppression, resampling,
perturbation), repeatedly evaluate all remaining steps and prune the one with
the lowest loss delta on the final step, until all steps are pruned.  Produces
one sparsity-curve plot per method (relative perplexity vs fraction pruned).

Perturbation uses attention-level K/V splicing (``PerturbedKVPostSourceContext``):
clean token positions and final-step loss slice; perturbed keys/values on
selected step spans come from a full forward on the perturbed CoT.

Attention suppression uses the same per-step token ranges as perturbation
(``map_steps_to_tokens`` on the clean CoT). KV prefix caching: the clean prefix
KV is computed once and reused for each candidate (prefix tokens are unaffected
by suppression on later key positions).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import torch
from tqdm import tqdm

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from causal_cot import paths as rp
from causal_cot.interventions.attention_suppression import CumulativeAttnSuppressionContext
from causal_cot.interventions.kv_perturbation import (
    PerturbedKVPostSourceContext,
    query_mix_start_for_kv_substitution,
)
from causal_cot.load_model import load_model_and_tokenizer
from causal_cot.metrics import calculate_step_loss
from causal_cot.step_utils import (
    assert_matching_reasoning_step_counts,
    assert_reasoning_token_ranges_match_think_inner,
    extract_think_block,
    map_steps_to_char_ranges,
    map_steps_to_tokens,
    parse_reasoning_steps,
    think_inner_token_span,
)


"""The reusable attention-suppression context is imported above."""

class _ArchivedCumulativeAttnSuppressionContext:
    """
    Monkey-patches Qwen2Attention.forward to suppress attention from multiple
    source spans simultaneously.  For each span (src_start, src_end), all tokens
    at absolute positions >= src_end cannot attend to positions [src_start, src_end).

    Supports optional past_key_values (KV prefix caching) where queries cover
    only a suffix of the sequence.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        suppression_spans: List[Tuple[int, int]],
    ):
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
                position_ids = (
                    torch.arange(q_len, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
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
                cached_k = past_key_value.layers[_layer_idx].keys
                cached_v = past_key_value.layers[_layer_idx].values
                key_states = torch.cat([cached_k.to(device), key_states], dim=2)
                value_states = torch.cat([cached_v.to(device), value_states], dim=2)

            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)

            kv_len = key_states.shape[2]
            prefix_len = kv_len - q_len

            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3))
                / math.sqrt(head_dim)
            )

            # Always build our own causal mask: the model-provided attention_mask
            # covers the full sequence and has the wrong shape when using a sliced
            # KV prefix cache.
            key_positions = torch.arange(kv_len, device=device)
            query_abs = torch.arange(prefix_len, prefix_len + q_len, device=device)
            causal_mask = key_positions.unsqueeze(0) <= query_abs.unsqueeze(1)
            attn_weights.masked_fill_(
                ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

            for src_start, src_end in suppression_spans:
                q_from = max(0, src_end - prefix_len)
                attn_weights[:, :, q_from:, src_start:src_end] = -1e4

            attn_weights = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            return (attn_module.o_proj(attn_output), None)

        return patched_forward

    def __enter__(self) -> "_ArchivedCumulativeAttnSuppressionContext":
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


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def _slice_kv_cache(full_cache, prefix_len: int):
    """Slice a DynamicCache to keep only the first *prefix_len* positions."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    sliced = DynamicCache()
    for layer in full_cache.layers:
        new_layer = DynamicLayer()
        new_layer.keys = layer.keys[:, :, :prefix_len, :].contiguous()
        new_layer.values = layer.values[:, :, :prefix_len, :].contiguous()
        new_layer.dtype = layer.keys.dtype
        new_layer.device = layer.keys.device
        new_layer.is_initialized = True
        sliced.layers.append(new_layer)
    return sliced


# ---------------------------------------------------------------------------
# Text rebuild helpers
# ---------------------------------------------------------------------------

def _compute_gaps(full_text: str, char_ranges: List[Tuple[int, int]]) -> List[str]:
    """Extract delimiter text between steps.
    gaps[0] = text before first step, gaps[-1] = text after last step."""
    gaps = [full_text[: char_ranges[0][0]]]
    for i in range(1, len(char_ranges)):
        gaps.append(full_text[char_ranges[i - 1][1] : char_ranges[i][0]])
    gaps.append(full_text[char_ranges[-1][1] :])
    return gaps


def _rebuild_text(steps_list: List[str], gaps: List[str]) -> str:
    parts = [gaps[0]]
    for i, step in enumerate(steps_list):
        parts.append(step)
        parts.append(gaps[i + 1])
    return "".join(parts)


def _rebuilt_step_token_ranges(
    text: str,
    working: List[str],
    gaps: List[str],
    tokenizer,
) -> List[Tuple[int, int]]:
    """
    Token ranges for each reasoning step after ``_rebuild_text(working, gaps)``.

    Uses the same step boundaries as the rebuild (fixed ``gaps`` from the clean
    CoT) instead of ``parse_reasoning_steps`` on the rebuilt string. Hybrid
    clean/resampled bodies can change newline structure and make the parser merge
    or split steps; this path always returns exactly ``len(working)`` contiguous
    ranges that partition the think-inner token span.
    """
    n = len(working)
    if len(gaps) != n + 1:
        raise ValueError(f"gaps length {len(gaps)} != {n + 1} for {n} steps.")

    tt0, tt1 = think_inner_token_span(text, tokenizer)

    pos = len(gaps[0])
    char_starts: List[int] = []
    for i in range(n):
        char_starts.append(pos)
        pos += len(working[i])
        if i + 1 < n:
            pos += len(gaps[i + 1])

    token_starts: List[int] = []
    for cs in char_starts:
        token_starts.append(
            len(tokenizer.encode(text[:cs], add_special_tokens=False))
        )

    ranges: List[Tuple[int, int]] = []
    for i in range(n):
        te = token_starts[i + 1] if i + 1 < n else tt1
        ranges.append((token_starts[i], te))

    assert_reasoning_token_ranges_match_think_inner(text, tokenizer, ranges)
    return ranges


def _load_corrupt_steps(
    cot_path: Path, num_original_steps: int
) -> List[str]:
    """Load a corrupt (perturbed / resampled) full CoT and return per-step texts."""
    text = cot_path.read_text(encoding="utf-8")
    think = extract_think_block(text)
    csteps = parse_reasoning_steps(think)
    if len(csteps) != num_original_steps:
        raise ValueError(
            f"{cot_path.name} has {len(csteps)} steps, expected {num_original_steps}."
        )
    return csteps


# ---------------------------------------------------------------------------
# Text-level greedy ablation (shared by resample & perturb)
# ---------------------------------------------------------------------------

def _run_text_ablation(
    method_name: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    steps: List[str],
    gaps: List[str],
    corrupt_step_texts: List[List[str]],
    clean_final_loss: float,
    num_steps: int,
) -> Tuple[Set[int], List[float], List[float], List[dict], List[dict]]:
    pruned: Set[int] = set()
    candidates: Set[int] = set(range(num_steps - 1))  # last step is the target, never pruned
    xs: List[float] = [0.0]
    ys: List[float] = [1.0]
    all_round_deltas: List[dict] = []
    all_round_raw: List[dict] = []  # per-corruption raw deltas

    while candidates:
        step_deltas: dict = {}
        step_raw: dict = {}  # i -> [delta_per_corruption]
        for i in tqdm(sorted(candidates), desc=f"{method_name} round {len(pruned) + 1}"):
            deltas: List[float] = []
            for variant_steps in corrupt_step_texts:
                working = list(steps)
                working[i] = variant_steps[i]
                for j in pruned:
                    working[j] = variant_steps[j]
                text = _rebuild_text(working, gaps)
                ids = tokenizer(
                    text, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(device)
                with torch.no_grad():
                    out = model(input_ids=ids)
                logits = out.logits[0] if out.logits.dim() == 3 else out.logits
                new_ranges = _rebuilt_step_token_ranges(
                    text, working, gaps, tokenizer
                )
                if new_ranges[-1][1] > ids.shape[1]:
                    raise ValueError(
                        "Rebuilt CoT: reasoning end exceeds tokenized length (truncation/leakage)."
                    )
                f_start, f_end = new_ranges[-1]
                loss = calculate_step_loss(logits, ids[0], f_start, f_end).item()
                deltas.append(loss - clean_final_loss)
                del out, logits, ids
            step_deltas[i] = sum(deltas) / len(deltas)
            step_raw[i] = deltas

        best_i = min(step_deltas, key=step_deltas.get)
        all_round_deltas.append({"pruned_step": best_i, **step_deltas})
        all_round_raw.append({"pruned_step": best_i, "step_deltas": {str(i): v for i, v in step_raw.items()}})
        pruned.add(best_i)
        candidates.remove(best_i)
        xs.append(len(pruned) / (num_steps - 1))
        ys.append(math.exp(step_deltas[best_i]))
        print(f"  Pruned step {best_i} (delta={step_deltas[best_i]:.4f}, ppl_ratio={ys[-1]:.4f})")

    return pruned, xs, ys, all_round_deltas, all_round_raw


# ---------------------------------------------------------------------------
# Perturbation via K/V splice (aligned clean + full perturbed forward)
# ---------------------------------------------------------------------------

def _run_kv_perturb_ablation(
    method_name: str,
    model: torch.nn.Module,
    clean_ids: torch.Tensor,
    step_ranges: List[Tuple[int, int]],
    pert_ids_list: List[torch.Tensor],
    pert_token_ranges_list: List[List[Tuple[int, int]]],
    clean_final_loss: float,
    num_steps: int,
) -> Tuple[Set[int], List[float], List[float], List[dict], List[dict]]:
    """Same greedy game as ``_run_text_ablation`` but perturbation uses ``PerturbedKVPostSourceContext``."""
    pruned: Set[int] = set()
    candidates: Set[int] = set(range(num_steps - 1))
    xs: List[float] = [0.0]
    ys: List[float] = [1.0]
    all_round_deltas: List[dict] = []
    all_round_raw: List[dict] = []
    f_start, f_end = step_ranges[-1]

    while candidates:
        step_deltas: dict = {}
        step_raw: dict = {}
        for i in tqdm(sorted(candidates), desc=f"{method_name} round {len(pruned) + 1}"):
            deltas: List[float] = []
            for pert_idx, pert_ids in enumerate(pert_ids_list):
                pert_ranges = pert_token_ranges_list[pert_idx]
                subst = {i, *pruned}
                spans = [pert_ranges[j] for j in sorted(subst)]
                mix_start = query_mix_start_for_kv_substitution(
                    i, subst, step_ranges
                )
                batch = torch.cat([clean_ids, pert_ids], dim=0)
                with PerturbedKVPostSourceContext(model, spans, mix_start):
                    with torch.no_grad():
                        out = model(input_ids=batch)
                logits = out.logits[0] if out.logits.dim() == 3 else out.logits
                loss = calculate_step_loss(
                    logits, clean_ids[0], f_start, f_end
                ).item()
                deltas.append(loss - clean_final_loss)
                del out, logits, batch
            step_deltas[i] = sum(deltas) / len(deltas)
            step_raw[i] = deltas

        best_i = min(step_deltas, key=step_deltas.get)
        all_round_deltas.append({"pruned_step": best_i, **step_deltas})
        all_round_raw.append(
            {"pruned_step": best_i, "step_deltas": {str(i): v for i, v in step_raw.items()}}
        )
        pruned.add(best_i)
        candidates.remove(best_i)
        xs.append(len(pruned) / (num_steps - 1))
        ys.append(math.exp(step_deltas[best_i]))
        print(
            f"  Pruned step {best_i} (delta={step_deltas[best_i]:.4f}, ppl_ratio={ys[-1]:.4f})"
        )

    return pruned, xs, ys, all_round_deltas, all_round_raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy backward ablation: prune reasoning steps per method."
    )
    parser.add_argument(
        "--clean_cot_path",
        type=str,
        default=rp.CLEAN_COT_DEFAULT,
        help="Path to the original (clean) CoT text file.",
    )
    parser.add_argument(
        "--pert_cot_dir",
        type=str,
        default=rp.PERT_DIR_DEFAULT,
        help="Directory containing perturbed_*.txt files.",
    )
    parser.add_argument(
        "--resample_cot_dir",
        type=str,
        default=rp.RESAMPLE_DIR_DEFAULT,
        help="Directory containing resampled_*.txt files produced by run_resample.py.",
    )
    parser.add_argument(
        "--num_corruptions", type=int, default=1,
        help="Number of corruption variants to average over for resample/perturb "
             "(default: 1). Ignored for attention suppression.",
    )
    parser.add_argument(
        "--flash_attn", action="store_true",
        help="Use flash_attention_2 for the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=rp.BACKWARD_ABLATION_OUTPUT,
        help="Directory to save results JSON and plots.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(use_flash_attention_2=args.flash_attn)
    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Parse clean CoT
    # ------------------------------------------------------------------
    clean_cot = Path(args.clean_cot_path).read_text(encoding="utf-8")
    think_content = extract_think_block(clean_cot)
    steps = parse_reasoning_steps(think_content)
    if not steps:
        raise ValueError("No reasoning steps found in <think> block.")
    num_steps = len(steps)
    print(f"Parsed {num_steps} reasoning steps.")

    char_ranges = map_steps_to_char_ranges(clean_cot, steps)
    step_ranges = map_steps_to_tokens(clean_cot, steps, tokenizer)
    gaps = _compute_gaps(clean_cot, char_ranges)

    clean_ids = tokenizer(
        clean_cot, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    seq_len = clean_ids.shape[1]
    try:
        assert_reasoning_token_ranges_match_think_inner(
            clean_cot, tokenizer, step_ranges
        )
    except ValueError as e:
        raise ValueError(f"Think / step token alignment: {e}") from e
    if step_ranges[-1][1] > seq_len:
        raise ValueError(
            f"Reasoning spans end at token {step_ranges[-1][1]} but sequence length is {seq_len}."
        )

    # ------------------------------------------------------------------
    # Load corrupt step texts
    # ------------------------------------------------------------------
    pert_dir = Path(args.pert_cot_dir)
    pert_files = sorted(pert_dir.glob("perturbed_*.txt"))[: args.num_corruptions]
    if not pert_files:
        raise ValueError(f"No perturbed_*.txt found in {pert_dir}")
    pert_texts = [p.read_text(encoding="utf-8") for p in pert_files]
    for t in pert_texts:
        try:
            assert_matching_reasoning_step_counts(clean_cot, [t])
        except ValueError as e:
            raise ValueError(f"Perturbed CoT alignment: {e}") from e
    pert_ids_list: List[torch.Tensor] = []
    pert_token_ranges_list: List[List[Tuple[int, int]]] = []
    for p, t in zip(pert_files, pert_texts):
        pert_steps = parse_reasoning_steps(extract_think_block(t))
        pert_ranges = map_steps_to_tokens(t, pert_steps, tokenizer)
        try:
            assert_reasoning_token_ranges_match_think_inner(
                t, tokenizer, pert_ranges
            )
        except ValueError as e:
            raise ValueError(f"{p.name}: think / step token alignment: {e}") from e
        if len(pert_ranges) != len(step_ranges):
            raise ValueError(
                f"{p.name}: {len(pert_ranges)} step spans vs clean {len(step_ranges)}."
            )
        for j, (pr_span, cr_span) in enumerate(zip(pert_ranges, step_ranges)):
            if pr_span != cr_span:
                raise ValueError(
                    f"{p.name}: step {j} pert token span {pr_span} != clean {cr_span}."
                )
        ids = tokenizer(
            t, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        if pert_ranges[-1][1] > ids.shape[1]:
            raise ValueError(
                f"{p.name}: reasoning end token {pert_ranges[-1][1]} exceeds "
                f"sequence length {ids.shape[1]}."
            )
        if ids.shape[1] != seq_len:
            raise ValueError(
                f"{p.name}: token length {ids.shape[1]} != clean {seq_len}."
            )
        pert_ids_list.append(ids)
        pert_token_ranges_list.append(pert_ranges)
    print(f"Loaded {len(pert_ids_list)} perturbed variant(s).")

    resample_dir = Path(args.resample_cot_dir)
    resample_files = sorted(resample_dir.glob("resampled_*.txt"))[: args.num_corruptions]
    if not resample_files:
        raise ValueError(f"No resampled_*.txt found in {resample_dir}")
    resample_step_texts = [_load_corrupt_steps(p, num_steps) for p in resample_files]
    print(f"Loaded {len(resample_step_texts)} resampled variant(s).")

    # ------------------------------------------------------------------
    # Clean forward pass  (also capture KV cache for attn_supp)
    # ------------------------------------------------------------------
    print("Running clean forward pass...")
    with torch.no_grad():
        clean_out = model(input_ids=clean_ids, use_cache=True)
    clean_logits = (
        clean_out.logits[0] if clean_out.logits.dim() == 3 else clean_out.logits
    )
    full_kv_cache = clean_out.past_key_values

    final_start, final_end = step_ranges[-1]
    clean_final_loss = calculate_step_loss(
        clean_logits, clean_ids[0], final_start, final_end
    ).item()
    print(f"Clean final-step loss: {clean_final_loss:.4f}")
    del clean_logits

    results = {}

    # ==================================================================
    # Method 1: Attention Suppression  (with KV prefix caching)
    # ==================================================================
    print("\n--- Attention Suppression (greedy ablation) ---")
    pruned_attn: Set[int] = set()
    candidates_attn: Set[int] = set(range(num_steps - 1))
    xs_attn: List[float] = [0.0]
    ys_attn: List[float] = [1.0]
    all_round_deltas_attn: List[dict] = []

    while candidates_attn:
        step_deltas_attn: dict = {}
        for i in tqdm(sorted(candidates_attn), desc=f"attn_supp round {len(pruned_attn) + 1}"):
            spans = [step_ranges[j] for j in pruned_attn] + [step_ranges[i]]

            prefix_len = step_ranges[i][0]
            suffix_ids = clean_ids[:, prefix_len:]
            suffix_len = suffix_ids.shape[1]
            position_ids = torch.arange(
                prefix_len, prefix_len + suffix_len, device=device
            ).unsqueeze(0)
            prefix_cache = _slice_kv_cache(full_kv_cache, prefix_len)

            with CumulativeAttnSuppressionContext(model, spans):
                with torch.no_grad():
                    out = model(
                        input_ids=suffix_ids,
                        past_key_values=prefix_cache,
                        position_ids=position_ids,
                        use_cache=False,
                    )
            logits = out.logits[0] if out.logits.dim() == 3 else out.logits
            local_start = final_start - prefix_len
            local_end = final_end - prefix_len
            suffix_labels = clean_ids[0, prefix_len:]
            step_deltas_attn[i] = (
                calculate_step_loss(logits, suffix_labels, local_start, local_end).item()
                - clean_final_loss
            )
            del out, logits, prefix_cache

        best_i = min(step_deltas_attn, key=step_deltas_attn.get)
        all_round_deltas_attn.append({"pruned_step": best_i, **step_deltas_attn})
        pruned_attn.add(best_i)
        candidates_attn.remove(best_i)
        xs_attn.append(len(pruned_attn) / (num_steps - 1))
        ys_attn.append(math.exp(step_deltas_attn[best_i]))
        print(f"  Pruned step {best_i} (delta={step_deltas_attn[best_i]:.4f}, ppl_ratio={ys_attn[-1]:.4f})")

    print(f"attn_supp: pruned {len(pruned_attn)} steps at indices {sorted(pruned_attn)}")
    results["attn_supp"] = sorted(pruned_attn)
    torch.cuda.empty_cache()

    # ==================================================================
    # Method 2: Resampling  (text-level substitution)
    # ==================================================================
    print("\n--- Resampling (greedy ablation) ---")
    pruned_resample, xs_resample, ys_resample, all_round_deltas_resample, raw_resample = _run_text_ablation(
        "resample", model, tokenizer, device,
        steps, gaps, resample_step_texts,
        clean_final_loss, num_steps,
    )
    print(
        f"resample: pruned {len(pruned_resample)} steps "
        f"at indices {sorted(pruned_resample)}"
    )
    results["resample"] = sorted(pruned_resample)
    torch.cuda.empty_cache()

    # ==================================================================
    # Method 3: Perturbation  (K/V splice; full perturbed run for K/V)
    # ==================================================================
    print("\n--- Perturbation (greedy ablation, K/V substitution) ---")
    pruned_perturb, xs_perturb, ys_perturb, all_round_deltas_perturb, raw_perturb = _run_kv_perturb_ablation(
        "perturb",
        model,
        clean_ids,
        step_ranges,
        pert_ids_list,
        pert_token_ranges_list,
        clean_final_loss,
        num_steps,
    )
    print(
        f"perturb: pruned {len(pruned_perturb)} steps "
        f"at indices {sorted(pruned_perturb)}"
    )
    results["perturb"] = sorted(pruned_perturb)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n=== Summary ===")
    for method, pruned in results.items():
        print(f"  {method}: {len(pruned)} pruned — {pruned}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "backward_ablation_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_corruptions": args.num_corruptions,
                    "num_steps": num_steps,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved results to {out_path}")

        # Write per-method CSVs: rows=rounds, columns=step indices, values=loss delta
        step_cols = [str(s) for s in range(num_steps - 1)]
        csv_data = [
            ("attn_supp",  all_round_deltas_attn),
            ("resample",   all_round_deltas_resample),
            ("perturb",    all_round_deltas_perturb),
        ]
        for fname, round_deltas in csv_data:
            csv_path = out_dir / f"ablation_{fname}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "pruned_step"] + step_cols)
                for round_idx, rd in enumerate(round_deltas):
                    pruned_step = rd["pruned_step"]
                    row = [round_idx, pruned_step] + [
                        rd.get(int(s), "") for s in step_cols
                    ]
                    writer.writerow(row)
            print(f"CSV saved to {csv_path}")

        raw_path = out_dir / "ablation_raw_deltas.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump({"resample": raw_resample, "perturb": raw_perturb}, f, indent=2)
        print(f"Raw per-corruption deltas saved to {raw_path}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        curves = [
            ("attn_supp",  "Attention Suppression", xs_attn,     ys_attn),
            ("resample",   "Resampling",             xs_resample, ys_resample),
            ("perturb",    "Perturbation",            xs_perturb,  ys_perturb),
        ]
        for fname, title, xs, ys in curves:
            plt.figure(figsize=(8, 5))
            plt.scatter(xs, ys, s=20, color="orange")
            plt.axhline(1.0, color="gray", linestyle="--", lw=0.8)
            plt.xlabel("Sparsity (fraction of steps pruned)")
            plt.ylabel("Relative Perplexity")
            plt.title(f"{title}: Relative Perplexity vs Step Sparsity")
            plt.tight_layout()
            plot_path = out_dir / f"ablation_{fname}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
