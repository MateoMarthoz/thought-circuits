"""
Load deepseek-ai/DeepSeek-R1-Distill-Qwen-14B and its tokenizer for causal CoT analysis.
Phase 4: model in bfloat16, device_map="auto"; optional flash_attention_2 for VRAM.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


def load_tokenizer_only():
    """Load only the tokenizer (no model). Use for run_perturb.py to avoid loading 14B params."""
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def load_model_and_tokenizer(
    *,
    model_id: str | None = None,
    use_flash_attention_2: bool = False,
) -> tuple:
    """
    Load DeepSeek-R1-Distill-Qwen-14B and matching tokenizer.

    Args:
        model_id: HuggingFace model id or local path (default: MODEL_ID).
        use_flash_attention_2: If True, pass attn_implementation="flash_attention_2"
            to reduce VRAM for long contexts (requires flash-attn installed).

    Returns:
        (model, tokenizer)
    """
    mid = model_id or MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)

    kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(mid, **kwargs)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Load DeepSeek-R1-Distill-Qwen-14B")
    args = parser.parse_args()

    print(f"Loading tokenizer and model: {MODEL_ID}")
    model, tokenizer = load_model_and_tokenizer()
    print(f"Model device map: {model.hf_device_map}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print("Load complete.")


if __name__ == "__main__":
    main()
