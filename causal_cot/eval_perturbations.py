"""
Load multiple token-aligned perturbed CoT files for evaluation.

Each file must match the clean CoT's per-step token spans (``map_steps_to_tokens``)
and pass ``assert_reasoning_token_ranges_match_think_inner`` so there are no gaps
inside the think block. K/V interventions always splice from one perturbed
forward (batch row 1) at those spans.
"""

from pathlib import Path
from typing import List, Tuple

import torch

from .step_utils import (
    assert_matching_reasoning_step_counts,
    assert_reasoning_token_ranges_match_think_inner,
    extract_think_block,
    map_steps_to_tokens,
    parse_reasoning_steps,
)


def load_aligned_perturbed_cots(
    clean_cot: str,
    clean_reasoning_token_ranges: List[Tuple[int, int]],
    tokenizer,
    pert_dir: Path,
    num_perturbations: int,
    device: torch.device,
    seq_len: int,
) -> Tuple[List[torch.Tensor], List[List[Tuple[int, int]]], List[str]]:
    """
    Load the first ``num_perturbations`` files matching ``perturbed_*.txt`` in
    ``pert_dir`` (sorted by filename).

    Returns:
        pert_ids_list: one tokenized sequence per file (``add_special_tokens=False``).
        pert_token_ranges_list: per-file ``map_steps_to_tokens`` ranges (must match
            ``clean_reasoning_token_ranges`` elementwise after validation).
        pert_names: filenames for logging.
    """
    if num_perturbations < 1:
        raise ValueError("num_perturbations must be at least 1.")

    pert_dir = pert_dir.expanduser().resolve()
    files = sorted(pert_dir.glob("perturbed_*.txt"))[:num_perturbations]
    if not files:
        raise ValueError(f"No perturbed_*.txt files found in {pert_dir}")
    if len(files) < num_perturbations:
        raise ValueError(
            f"Requested {num_perturbations} perturbation files but only found {len(files)} in {pert_dir}."
        )

    n_steps = len(clean_reasoning_token_ranges)
    pert_ids_list: List[torch.Tensor] = []
    pert_token_ranges_list: List[List[Tuple[int, int]]] = []
    names: List[str] = []

    for p in files:
        text = p.read_text(encoding="utf-8")
        try:
            assert_matching_reasoning_step_counts(clean_cot, [text])
        except ValueError as e:
            raise ValueError(f"{p.name}: alignment: {e}") from e

        steps = parse_reasoning_steps(extract_think_block(text))
        pert_ranges = map_steps_to_tokens(text, steps, tokenizer)
        try:
            assert_reasoning_token_ranges_match_think_inner(
                text, tokenizer, pert_ranges
            )
        except ValueError as e:
            raise ValueError(f"{p.name}: think / step token alignment: {e}") from e

        if len(pert_ranges) != n_steps:
            raise ValueError(
                f"{p.name}: {len(pert_ranges)} reasoning steps vs clean {n_steps}."
            )
        for j, (a, b) in enumerate(zip(pert_ranges, clean_reasoning_token_ranges)):
            if a != b:
                raise ValueError(
                    f"{p.name}: step {j} token span {a} != clean {b} "
                    "(K/V splice requires identical indices per step)."
                )

        ids = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        if ids.shape[1] != seq_len:
            raise ValueError(
                f"{p.name}: token length {ids.shape[1]} != clean sequence length {seq_len}."
            )
        if pert_ranges[-1][1] > ids.shape[1]:
            raise ValueError(
                f"{p.name}: reasoning end token {pert_ranges[-1][1]} exceeds "
                f"sequence length {ids.shape[1]}."
            )
        pert_ids_list.append(ids)
        pert_token_ranges_list.append(pert_ranges)
        names.append(p.name)

    return pert_ids_list, pert_token_ranges_list, names
