"""
Reasoning step parsing for CoT traces.
Splits a chain-of-thought trace into discrete steps (double newline or colon/period rules).
Maps steps to token index ranges via get_chunk_ranges and get_chunk_token_ranges.
"""

import re
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from .text_utils import get_chunk_ranges, get_chunk_token_ranges

THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_think_block(full_text: str) -> str:
    """Extract the inner think-block string (between opening/closing think tags), without stripping.

    Leading/trailing characters inside the tags are preserved so parsing and character
    alignment match the full document tokenization. If no think block is found, returns
    ``full_text`` unchanged.
    """
    m = THINK_BLOCK_RE.search(full_text)
    if not m:
        return full_text
    return m.group(1)


def parse_reasoning_steps(cot_text: str) -> List[str]:
    """
    Split a chain-of-thought trace into reasoning steps.

    Rules:
    1. Paragraph breaks (double newline in the raw text): the ``\\n\\n`` between
       paragraphs is the **first** character(s) of the **next** step, not the tail
       of the previous step. (One blank line in ``split("\\n")`` = ``\\n\\n`` in text;
       consecutive blank lines add more ``\\n`` to that prefix.)
    2. If a line ends with a colon, blank lines do not split until the line before the
       next line that ends with a period or colon. When that next step starts, a single
       ``\\n`` between the flushed block and the new line is the **first** character of
       the new step (so the boundary newline is included in the next step for masking).
    3. The following line that ends with ``.`` or ``:`` begins the next step.

    Returns:
        List of step strings (each step is the concatenation of its lines with ``\\n``).
    """
    lines = cot_text.split("\n")
    steps: List[str] = []
    current: List[str] = []
    leading_empty: List[str] = []
    in_colon_block = False
    blank_run = 0  # consecutive gap newlines after a paragraph break (see loop)

    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped == "":
            if in_colon_block:
                current.append(line)
            elif current:
                has_future_non_empty = any(
                    lines[j].strip() for j in range(i + 1, len(lines))
                )
                if has_future_non_empty:
                    steps.append("\n".join(current))
                    current = []
                    blank_run = 1
                else:
                    current.append(line)
            elif blank_run > 0:
                blank_run += 1
            else:
                leading_empty.append(line)
        elif in_colon_block:
            if stripped.endswith(".") or stripped.endswith(":"):
                if current:
                    steps.append("\n".join(current))
                    current = ["\n" + line]
                else:
                    current = [line]
                in_colon_block = stripped.endswith(":")
            else:
                current.append(line)
                if stripped.endswith(":"):
                    in_colon_block = True
        else:
            prefix = "\n" * (blank_run + 1) if blank_run > 0 else ""
            blank_run = 0
            line_to_add = prefix + line
            if not current and leading_empty:
                current = leading_empty + [line_to_add]
                leading_empty = []
            else:
                if not current:
                    current = [line_to_add]
                else:
                    current.append(line_to_add)
            if stripped.endswith(":"):
                in_colon_block = True

    if current:
        steps.append("\n".join(current))
    elif leading_empty:
        steps.append("\n".join(leading_empty))

    return steps


def count_reasoning_steps_in_think(full_text: str) -> int:
    """Number of reasoning steps inside the think block (same parser as training)."""
    return len(parse_reasoning_steps(extract_think_block(full_text)))


def assert_matching_reasoning_step_counts(
    clean_full_text: str,
    perturbed_full_texts: List[str],
    *,
    label: str = "perturbed",
) -> int:
    """
    Ensure clean and every perturbed CoT yield the same ``parse_reasoning_steps`` count
    inside ``think`` so step indices align. Returns that shared count.
    """
    n_clean = count_reasoning_steps_in_think(clean_full_text)
    for i, pt in enumerate(perturbed_full_texts):
        n_p = count_reasoning_steps_in_think(pt)
        if n_p != n_clean:
            raise ValueError(
                f"{label} #{i}: {n_p} reasoning steps inside think != clean {n_clean}; "
                "splitting must match for token alignment."
            )
    return n_clean


def _think_inner_char_bounds(full_text: str) -> Optional[Tuple[int, int]]:
    m = THINK_BLOCK_RE.search(full_text)
    if not m:
        return None
    return (m.start(1), m.end(1))


def _expand_chunk_ranges_to_think_inner(
    full_text: str, chunk_ranges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Extend first/last step character spans to cover the entire think inner region."""
    bounds = _think_inner_char_bounds(full_text)
    if bounds is None or not chunk_ranges:
        return chunk_ranges
    lo, hi = bounds
    out = list(chunk_ranges)
    s0, e0 = out[0]
    out[0] = (lo if s0 > lo else s0, e0)
    sL, eL = out[-1]
    out[-1] = (sL, hi if eL < hi else eL)
    return out


def map_steps_to_char_ranges(full_text: str, steps: List[str]) -> List[Tuple[int, int]]:
    """Character ranges per step in ``full_text``, expanded to the full think inner span."""
    chunk_ranges = get_chunk_ranges(full_text, steps)
    return _expand_chunk_ranges_to_think_inner(full_text, chunk_ranges)


def map_steps_to_tokens(
    full_text: str,
    steps: List[str],
    tokenizer: AutoTokenizer,
) -> List[Tuple[int, int]]:
    """
    Map each reasoning step to (start_idx, end_idx) token index ranges in the full text.

    Uses get_chunk_ranges to find character ranges for each step in full_text,
    then get_chunk_token_ranges to convert those to token indices. Ranges are expanded
    so the union covers the full think-block inner span (no token gap inside the block).

    Args:
        full_text: The full CoT text (must contain the step substrings in order).
        steps: List of step strings from parse_reasoning_steps.
        tokenizer: Tokenizer used for the LLM (e.g. same as model).

    Returns:
        List of (start_idx, end_idx) tuples, one per step, in order.
    """
    return get_chunk_token_ranges(
        full_text, map_steps_to_char_ranges(full_text, steps), tokenizer
    )


def think_inner_token_span(full_text: str, tokenizer: AutoTokenizer) -> Tuple[int, int]:
    """
    Half-open token indices ``[t0, t1)`` covering the think-inner character span
    ``[lo, hi)`` in ``full_text`` (same tokenization as training, no special tokens).
    """
    bounds = _think_inner_char_bounds(full_text)
    if bounds is None:
        raise ValueError("No think block found; cannot compute think-inner token span.")
    lo, hi = bounds
    t0 = len(tokenizer.encode(full_text[:lo], add_special_tokens=False))
    t1 = len(tokenizer.encode(full_text[:hi], add_special_tokens=False))
    return t0, t1


def validate_step_ranges_contiguous(
    step_ranges: List[Tuple[int, int]], num_reasoning_steps: int
) -> Tuple[int, int]:
    """
    ``step_ranges[0]`` = prompt, ``step_ranges[1:N+1]`` = reasoning steps.
    Ensures the prompt ends where step 1 starts and reasoning spans are contiguous
    with no gaps or overlaps. Returns ``(t0, t1)`` = ``[step1.start, stepN.end)``.
    """
    N = num_reasoning_steps
    if len(step_ranges) != N + 1:
        raise ValueError(
            f"Expected {N + 1} step ranges (prompt + {N} steps), got {len(step_ranges)}."
        )
    if step_ranges[0][1] != step_ranges[1][0]:
        raise ValueError(
            f"Prompt end token {step_ranges[0][1]} must equal first reasoning start "
            f"{step_ranges[1][0]} (no tokens between prompt and step 1 inside the mask logic)."
        )
    for k in range(1, N):
        if step_ranges[k][1] != step_ranges[k + 1][0]:
            raise ValueError(
                f"Reasoning steps {k} and {k + 1} must meet with no gap/overlap: "
                f"end {step_ranges[k][1]} != start {step_ranges[k + 1][0]}."
            )
    return step_ranges[1][0], step_ranges[N][1]


def assert_reasoning_span_matches_think_inner(
    full_text: str,
    tokenizer: AutoTokenizer,
    step_ranges: List[Tuple[int, int]],
    num_reasoning_steps: int,
) -> None:
    """Every think-inner token must be exactly the union of reasoning step ranges."""
    t0, t1 = validate_step_ranges_contiguous(step_ranges, num_reasoning_steps)
    tt0, tt1 = think_inner_token_span(full_text, tokenizer)
    if (t0, t1) != (tt0, tt1):
        raise ValueError(
            f"Reasoning token span [{t0}, {t1}) must equal think-inner span [{tt0}, {tt1}) "
            "so no think token is left unassigned or masked as prompt."
        )


def assert_reasoning_token_ranges_match_think_inner(
    full_text: str,
    tokenizer: AutoTokenizer,
    reasoning_token_ranges: List[Tuple[int, int]],
) -> None:
    """
    Same think-inner coverage check for ``N`` ranges (steps 1..N only, no prompt row),
    e.g. from ``map_steps_to_tokens``.
    """
    if not reasoning_token_ranges:
        raise ValueError("reasoning_token_ranges is empty.")
    N = len(reasoning_token_ranges)
    for k in range(N - 1):
        if reasoning_token_ranges[k][1] != reasoning_token_ranges[k + 1][0]:
            raise ValueError(
                f"Reasoning token ranges {k} and {k + 1} must be contiguous: "
                f"end {reasoning_token_ranges[k][1]} != start {reasoning_token_ranges[k + 1][0]}."
            )
    t0, t1 = reasoning_token_ranges[0][0], reasoning_token_ranges[-1][1]
    tt0, tt1 = think_inner_token_span(full_text, tokenizer)
    if (t0, t1) != (tt0, tt1):
        raise ValueError(
            f"Reasoning token span [{t0}, {t1}) must equal think-inner span [{tt0}, {tt1})."
        )


def assign_token2step_tensor(
    step_ranges: List[Tuple[int, int]],
    num_reasoning_steps: int,
    q_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    ``token2step[q] = k`` for ``q`` in ``step_ranges[k]`` (reasoning ``k = 1..N``),
    else ``0`` (prompt / outside think). Raises if the reasoning union does not
    exactly fill ``[t0, t1)`` with no holes or if prompt/post regions pick up nonzero.
    """
    N = num_reasoning_steps
    t0, t1 = validate_step_ranges_contiguous(step_ranges, N)
    if t1 > q_len:
        raise RuntimeError(
            f"Sequence length {q_len} < reasoning end {t1}; truncated input would leak masks."
        )
    token2step = torch.zeros(q_len, dtype=torch.long, device=device)
    for k, (s, e) in enumerate(step_ranges[1:], start=1):
        if s < 0 or e > q_len or s >= e:
            raise RuntimeError(f"Invalid step range k={k}: [{s}, {e}) vs q_len={q_len}.")
        token2step[s:e] = k
    if bool((token2step[t0:t1] == 0).any().item()):
        raise RuntimeError(
            "Token leakage: some positions in the reasoning span [t0,t1) are not assigned a step."
        )
    if t0 > 0 and bool((token2step[0:t0] != 0).any().item()):
        raise RuntimeError("Token leakage: pre-reasoning positions must be prompt (step 0).")
    if t1 < q_len and bool((token2step[t1:q_len] != 0).any().item()):
        raise RuntimeError(
            "Token leakage: post-reasoning positions must be prompt (step 0), not a reasoning step."
        )
    return token2step


def get_step_ranges_with_prompt(full_text: str, tokenizer: AutoTokenizer) -> List[Tuple[int, int]]:
    """
    Token ranges: ``[(0, t_first), (s1,e1), …, (sN,eN)]`` where index 0 is the prompt
    prefix and 1..N are reasoning steps inside the think block.

    Validates that reasoning spans are contiguous and exactly cover the think-inner
    token span (no gap tokens assigned to prompt, no leakage).
    """
    if not THINK_BLOCK_RE.search(full_text):
        raise ValueError("No think block found in the CoT text.")
    think_content = extract_think_block(full_text)
    steps = parse_reasoning_steps(think_content)
    if not steps:
        raise ValueError("parse_reasoning_steps returned no steps inside the think block.")
    step_ranges_mapped = map_steps_to_tokens(full_text, steps, tokenizer)
    if len(step_ranges_mapped) != len(steps):
        raise ValueError(
            "map_steps_to_tokens returned fewer ranges than steps; "
            "some steps may not appear in full_text."
        )
    first_reasoning_start = step_ranges_mapped[0][0]
    step_ranges: List[Tuple[int, int]] = [(0, first_reasoning_start)] + step_ranges_mapped
    N = len(steps)
    assert_reasoning_span_matches_think_inner(full_text, tokenizer, step_ranges, N)
    return step_ranges
