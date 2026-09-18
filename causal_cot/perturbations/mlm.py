import random
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from transformers import pipeline

MLM_MODEL = "FacebookAI/roberta-large"
mlm_pipeline = pipeline("fill-mask", model=MLM_MODEL)


def get_maskable_words(text: str) -> List[Tuple[str, int, int]]:
    """
    Returns all maskable words: alphanumeric, more than 3 characters long if not a number.
    Returns list of (word, start, end) in document order.
    """
    results: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"\b[A-Za-z0-9]+\b", text):
        word = m.group(0)
        if len(word) <= 3 and not word.isdigit():
            continue
        results.append((word, m.start(), m.end()))
    return results


def get_maskable_words_in_think_tags(text: str) -> List[Tuple[str, int, int]]:
    """
    Returns maskable words that lie strictly inside <think>...</think>.
    (word, start, end) are in full-document character coordinates.
    Used only for selecting which tokens to perturb; tokenization/alignment use the full CoT.
    """
    block = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not block:
        return []
    inner = block.group(1)
    offset = block.start(1)
    words_inner = get_maskable_words(inner)
    return [(word, offset + start, offset + end) for word, start, end in words_inner]


def get_maskable_words_with_token_slots(
    original_text: str,
    llm_tokenizer,
) -> List[Tuple[str, int, int]]:
    """
    Returns maskable words with their token-index slots from the original text.
    Only words inside <think>...</think> are considered for perturbation; token indices are from
    the full-document tokenization (full CoT). Uses full-document offset_mapping for
    correct token boundaries. Each element is (word, t_start, t_end).
    """
    words = get_maskable_words_in_think_tags(original_text)
    enc = llm_tokenizer(
        original_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offsets = enc["offset_mapping"]

    result: List[Tuple[str, int, int]] = []
    for word, start, end in words:
        t_start, t_end = None, None
        for i, (o_start, o_end) in enumerate(offsets):
            if o_end > start and o_start < end:
                if t_start is None:
                    t_start = i
                t_end = i + 1
        if t_start is not None and t_end is not None:
            result.append((word, t_start, t_end))
    return result


def character_span_for_token_slot(
    current_text: str,
    t_start: int,
    t_end: int,
    llm_tokenizer,
) -> Tuple[int, int]:
    """
    Returns (char_start, char_end) for the token span [t_start, t_end) in current_text,
    using the LLM tokenizer's offset mapping. Use this to get character positions
    in current text after replacements have changed the string.
    """
    enc = llm_tokenizer(
        current_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    offset_mapping = enc["offset_mapping"]
    if t_start >= len(offset_mapping) or t_end <= 0 or t_start >= t_end:
        return (0, 0)
    char_start = offset_mapping[t_start][0]
    char_end = offset_mapping[t_end - 1][1]
    return (char_start, char_end)


def get_context_span(
    current_text: str,
    word_tuple: Tuple[str, int, int],
    mlm_tokenizer,
    window_size: int = 250,
) -> List[int]:
    """
    Input is the full CoT (with one word masked). Returns only the token window
    (window_size before + mask + window_size after) for the MLM. This is the only
    place we use a substring; alignment and all other logic use the full CoT.
    """
    word, start, end = word_tuple
    tokens = mlm_tokenizer.encode(current_text, add_special_tokens=False)
    first_token_idx = len(mlm_tokenizer.encode(current_text[:start], add_special_tokens=False))
    end_token_idx = len(mlm_tokenizer.encode(current_text[:end], add_special_tokens=False))
    lo = max(0, first_token_idx - window_size)
    hi = min(len(tokens), end_token_idx + window_size)
    return tokens[lo:hi]


def candidate_matches_capitalization(original: str, candidate: str) -> bool:
    """
    Returns True immediately if candidate is 1 character long.
    Returns False if one word is capitalised and the other isn't, or one word is all caps
    and the other isn't. Else returns True.
    """
    if len(candidate) == 1:
        return True
    original_letters = [c for c in original if c.isalpha()]
    candidate_letters = [c for c in candidate if c.isalpha()]
    if original_letters and candidate_letters:
        if original_letters[0].isupper() != candidate_letters[0].isupper():
            return False
        if all(c.isupper() for c in original_letters) != all(c.isupper() for c in candidate_letters):
            return False
    return True


def candidate_at_same_token_indices(
    current_cot: str,
    candidate_cot: str,
    t_start: int,
    t_end: int,
    llm_tokenizer,
) -> bool:
    """
    Tokenizes current_cot and candidate_cot with the LLM tokenizer (full sequence,
    no truncation). Returns True iff lengths match and every token outside [t_start, t_end)
    is identical. The slot indices are the stable token range from the original text.
    """
    enc_current = llm_tokenizer(
        current_cot, add_special_tokens=False, truncation=False
    )
    enc_candidate = llm_tokenizer(
        candidate_cot, add_special_tokens=False, truncation=False
    )
    current_ids = enc_current["input_ids"]
    candidate_ids = enc_candidate["input_ids"]

    if len(candidate_ids) != len(current_ids):
        return False
    if candidate_ids[:t_start] != current_ids[:t_start]:
        return False
    if candidate_ids[t_end:] != current_ids[t_end:]:
        return False
    return True


def filter_mlm_predictions(
    predictions: List[Any],
    original_word: str,
    top_k: int,
    current_cot: str,
    char_start: int,
    char_end: int,
    t_start: int,
    t_end: int,
    llm_tokenizer,
) -> List[str]:
    """
    Takes the top k predictions from the MLM and the original word. Keeps only candidates
    that are not equal to the original (lowercase), pass the capitalization filter, and
    (final check) sit at the same token indices [t_start, t_end) in the LLM tokenization.
    Uses char_start/char_end to build candidate_cot; uses t_start/t_end for the alignment check.
    """
    def get_token_str(item: Any) -> str:
        if isinstance(item, dict):
            t = item.get("token_str", "") or item.get("sequence", "")
        else:
            t = str(item)
        return t.replace("Ġ", "").replace("Ċ", "").strip()

    result: List[str] = []
    for pred in predictions[:top_k]:
        raw = get_token_str(pred)
        if not raw:
            continue
        if not raw.isalnum():
            continue
        if raw.lower() == original_word.lower():
            continue
        if not candidate_matches_capitalization(original_word, raw):
            continue
        if original_word.lower() in raw.lower():
            continue
        if raw.lower() in original_word.lower():
            continue

        # Preserve original leading/trailing whitespace so the splice doesn't delete spaces
        original_slice = current_cot[char_start:char_end]
        leading_space = original_slice[: len(original_slice) - len(original_slice.lstrip())]
        trailing_space = original_slice[len(original_slice.rstrip()) :]
        raw_spaced = leading_space + raw + trailing_space

        candidate_cot = current_cot[:char_start] + raw_spaced + current_cot[char_end:]
        if not candidate_at_same_token_indices(
            current_cot, candidate_cot, t_start, t_end, llm_tokenizer
        ):
            continue
        seen_stripped = {r.strip() for r in result}
        if raw not in seen_stripped:
            # Print token context once for the first selected candidate only (same index range for both)
            if not result:
                enc_cur = llm_tokenizer(current_cot, add_special_tokens=False, truncation=False)
                enc_cand = llm_tokenizer(candidate_cot, add_special_tokens=False, truncation=False)
                cur_ids = enc_cur["input_ids"]
                cand_ids = enc_cand["input_ids"]
                lo = max(0, t_start - 5)
                hi = min(len(cur_ids), t_end + 5)

                def _tokens_line(ids: list) -> str:
                    parts = []
                    for i in range(lo, hi):
                        tok = llm_tokenizer.decode([ids[i]])
                        in_slot = t_start <= (lo + len(parts)) < t_end
                        if in_slot:
                            parts.append(f"'[ {tok}]'")
                        else:
                            parts.append(f"[ {tok}]")
                    # Same index range for both: prev [lo:t_start), slot [t_start:t_end), next [t_end:hi)
                    n = len(parts)
                    i_prev = min(n, max(0, t_start - lo))
                    i_slot = min(n, max(i_prev, t_end - lo))
                    prev = " ".join(parts[:i_prev])
                    slot = " ".join(parts[i_prev:i_slot])
                    next_ = " ".join(parts[i_slot:])
                    return " ".join(s for s in (prev, slot, next_) if s)

                print(f"    [token_align] Original:  {_tokens_line(cur_ids)}")
                print(f"    [token_align] Perturbed: {_tokens_line(cand_ids)}")
            result.append(raw_spaced)
    return result


def generate_perturbed_cots(
    original_cot: str,
    llm_tokenizer,
    num_variants: int = 5,
    max_replacements_per_token: int = 20,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> List[str]:
    """
    For each of num_variants iterations: shuffle maskable word order, then for each word
    get top 20 MLM replacements, filter them, print success (replacement made) or failure
    (reason). After each iteration save the result to output_dir/perturbed_{i}.txt.
    """
    if seed is not None:
        random.seed(seed)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    mlm_tok = mlm_pipeline.tokenizer
    slots = get_maskable_words_with_token_slots(original_cot, llm_tokenizer)

    variants: List[str] = []
    for v in range(num_variants):
        print(f"\n--- Variant {v + 1}/{num_variants} ---")
        current = original_cot
        word_order = list(range(len(slots)))
        random.shuffle(word_order)

        for idx in word_order:
            word, t_start, t_end = slots[idx]
            char_start, char_end = character_span_for_token_slot(
                current, t_start, t_end, llm_tokenizer
            )
            masked_text = (
                current[:char_start]
                + mlm_tok.mask_token
                + current[char_end:]
            )
            mask_end = char_start + len(mlm_tok.mask_token)
            context_ids = get_context_span(
                masked_text,
                (word, char_start, mask_end),
                mlm_tok,
                window_size=250,
            )
            context_str = mlm_tok.decode(context_ids)
            try:
                preds = mlm_pipeline(context_str, top_k=max_replacements_per_token)
            except Exception:
                print(f"  [FAILURE] '{word}' - MLM error")
                continue
            if isinstance(preds, list) and preds and isinstance(preds[0], list):
                preds = preds[0]

            valid = filter_mlm_predictions(
                preds,
                original_word=word,
                top_k=max_replacements_per_token,
                current_cot=current,
                char_start=char_start,
                char_end=char_end,
                t_start=t_start,
                t_end=t_end,
                llm_tokenizer=llm_tokenizer,
            )
            if valid:
                replacement = valid[0]
                current = current[:char_start] + replacement + current[char_end:]
                print(f"  [SUCCESS] ({v + 1}/{num_variants}) Replaced '{word}' -> '{replacement}'")
            else:
                print(f"  [FAILURE] '{word}' - no valid replacement (all filtered out)")

        variants.append(current)
        if output_dir is not None:
            (output_dir / f"perturbed_{v}.txt").write_text(current, encoding="utf-8")

    return variants
