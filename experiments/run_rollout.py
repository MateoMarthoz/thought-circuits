#!/usr/bin/env python3
"""
Generate one CoT rollout with DeepSeek-R1-Distill-Qwen-14B.
Uses the same prompt structure and sampling (temperature/top_p) as thought-anchors.
Saves the full CoT (prompt + generation) for use with run_perturb.py.
"""

import argparse
import random
import sys
from pathlib import Path

import torch

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from causal_cot import paths as rp
from causal_cot.load_model import load_model_and_tokenizer


DEFAULT_QUESTION = (
    "When the base-16 number 66666_{16} is written in base 2, "
    "how many base-2 digits (bits) does it have?"
)

PROMPT_TEMPLATE = (
    "Solve this math problem step by step. "
    "You MUST put your final answer in \\boxed{{}}. "
    "Problem: {question} "
    "Solution: \n<think>\n"
)


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def main():
    parser = argparse.ArgumentParser(
        description="Generate one chain-of-thought rollout (DeepSeek-R1-Distill-Qwen-14B)"
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Math problem to solve (default: base-16 to base-2 bits question)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=rp.CLEAN_COT_DEFAULT,
        help="Output file path for the full CoT (prompt + generation).",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6, same as thought-anchors)",
    )
    parser.add_argument(
        "-tp",
        "--top-p",
        type=float,
        default=0.95,
        dest="top_p",
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "-mt",
        "--max-tokens",
        type=int,
        default=16384,
        dest="max_tokens",
        help="Max new tokens (default: 16384)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    prompt = build_prompt(args.question)
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Generating CoT (this may take a while)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    full_cot = prompt + generated

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_cot, encoding="utf-8")
    print(f"Saved full CoT ({len(full_cot)} chars) to {out_path}")


if __name__ == "__main__":
    main()
