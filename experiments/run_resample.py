#!/usr/bin/env python3
"""
Generate full-CoT resampled variants: for each variant, independently resample
every reasoning step via LLM generation and glue them back together with the
original delimiters.  Analogous to run_perturb.py but uses LLM generation
instead of MLM perturbation.
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from causal_cot import paths as rp
from causal_cot.load_model import load_model_and_tokenizer
from causal_cot.metrics import generate_resampled_continuation
from causal_cot.step_utils import (
    assert_reasoning_token_ranges_match_think_inner,
    extract_think_block,
    map_steps_to_char_ranges,
    map_steps_to_tokens,
    parse_reasoning_steps,
)


def _compute_gaps(full_text: str, char_ranges):
    """Extract delimiter text between steps.
    gaps[0] = text before first step, gaps[-1] = text after last step."""
    gaps = [full_text[: char_ranges[0][0]]]
    for i in range(1, len(char_ranges)):
        gaps.append(full_text[char_ranges[i - 1][1] : char_ranges[i][0]])
    gaps.append(full_text[char_ranges[-1][1] :])
    return gaps


def _rebuild_text(steps_list, gaps):
    parts = [gaps[0]]
    for i, step in enumerate(steps_list):
        parts.append(step)
        parts.append(gaps[i + 1])
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full-CoT resampled variants (all steps independently resampled)."
    )
    parser.add_argument(
        "--clean_cot_path",
        type=str,
        default=rp.CLEAN_COT_DEFAULT,
        help="Path to the original (clean) CoT text file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=rp.RESAMPLE_DIR_DEFAULT,
        help="Directory to save resampled_0.txt, resampled_1.txt, ...",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=1,
        help="Number of full-CoT resampled variants to produce (default: 1).",
    )
    parser.add_argument(
        "--num_continuations",
        type=int,
        default=20,
        help="Candidate continuations per step for similarity selection (default: 20).",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="Use flash_attention_2 for the model.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    clean_cot = Path(args.clean_cot_path).read_text(encoding="utf-8")
    think_content = extract_think_block(clean_cot)
    steps = parse_reasoning_steps(think_content)
    if not steps:
        raise ValueError("parse_reasoning_steps returned no steps inside <think> block.")
    num_steps = len(steps)

    char_ranges = map_steps_to_char_ranges(clean_cot, steps)
    if len(char_ranges) != num_steps:
        raise ValueError(
            f"map_steps_to_char_ranges returned {len(char_ranges)} ranges for {num_steps} steps."
        )
    gaps = _compute_gaps(clean_cot, char_ranges)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(use_flash_attention_2=args.flash_attn)
    device = next(model.parameters()).device

    step_token_ranges = map_steps_to_tokens(clean_cot, steps, tokenizer)
    try:
        assert_reasoning_token_ranges_match_think_inner(
            clean_cot, tokenizer, step_token_ranges
        )
    except ValueError as e:
        raise ValueError(f"Think / step token alignment: {e}") from e
    clean_seq_len = tokenizer(
        clean_cot, return_tensors="pt", add_special_tokens=False
    ).input_ids.shape[1]
    if step_token_ranges[-1][1] > clean_seq_len:
        raise ValueError(
            f"Reasoning end token {step_token_ranges[-1][1]} exceeds sequence length "
            f"{clean_seq_len}."
        )

    print("Pre-tokenizing step prefixes...")
    prefix_ids_list = []
    for i in range(num_steps):
        prefix_text = clean_cot[: char_ranges[i][0]]
        ids = tokenizer(
            prefix_text, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(device)
        prefix_ids_list.append(ids)

    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for v in tqdm(range(args.num_variants), desc="Variants"):
        resampled_steps = []
        out_path = output_dir / f"resampled_{v}.txt"
        for i in tqdm(range(num_steps), desc=f"  Steps (variant {v})", leave=False):
            MAX_RETRIES = 3
            chosen = None
            for attempt in range(MAX_RETRIES):
                candidate = generate_resampled_continuation(
                    prefix_text=clean_cot[: char_ranges[i][0]],
                    original_step_text=steps[i],
                    model=model,
                    tokenizer=tokenizer,
                    step_index=i,
                    num_continuations=args.num_continuations,
                    embedder=embedder,
                    prefix_ids=prefix_ids_list[i],
                )
                trial_steps = resampled_steps + [candidate] + list(steps[i + 1 :])
                trial_text = _rebuild_text(trial_steps, gaps)
                trial_think = extract_think_block(trial_text)
                trial_parsed = parse_reasoning_steps(trial_think)
                if len(trial_parsed) == num_steps:
                    chosen = candidate
                    break
            if chosen is not None:
                resampled_steps.append(chosen)
            else:
                print(
                    f"  Warning: step {i} failed alignment after {MAX_RETRIES} retries, "
                    f"keeping original step."
                )
                resampled_steps.append(steps[i])

            # Write the CoT so far (remaining steps filled with originals) so
            # progress is visible while the script is running.
            partial_steps = resampled_steps + list(steps[i + 1:])
            out_path.write_text(_rebuild_text(partial_steps, gaps), encoding="utf-8")
            print(f"  [{i + 1}/{num_steps}] Step {i}: {resampled_steps[-1][:80]!r}")

        final_think = extract_think_block(out_path.read_text(encoding="utf-8"))
        final_parsed = parse_reasoning_steps(final_think)
        if len(final_parsed) != num_steps:
            print(
                f"  Warning: variant {v} has {len(final_parsed)} steps "
                f"(expected {num_steps}). Saving anyway."
            )
        print(f"  Saved {out_path}")

    print(f"Wrote {args.num_variants} resampled CoT(s) to {output_dir}")


if __name__ == "__main__":
    main()
