"""
Chunk range and token range utilities for mapping reasoning steps to text/tokens.
Copied from thought-anchors/utils.py (get_chunk_ranges, get_chunk_token_ranges).
Dependencies: transformers.AutoTokenizer, standard library only.
"""

import re
from typing import List, Tuple

from transformers import AutoTokenizer


def get_chunk_ranges(full_text: str, chunks: List[str]) -> List[Tuple[int, int]]:
    # Get character ranges for each chunk in the full text
    chunk_ranges = []
    current_pos = 0

    for chunk in chunks:
        # Normalize the chunk for comparison (preserve length but standardize whitespace)
        normalized_chunk = re.sub(r"\s+", " ", chunk).strip()

        # Try to find the chunk in the full text
        chunk_start = -1

        # First try exact match from current position
        exact_match_pos = full_text.find(chunk, current_pos)
        if exact_match_pos != -1:
            chunk_start = exact_match_pos
        else:
            # If exact match fails, try with normalized text
            chunk_words = normalized_chunk.split()

            # Search for the sequence of words, allowing for different whitespace
            for i in range(current_pos, len(full_text) - len(normalized_chunk)):
                # Check if this could be the start of our chunk
                text_window = full_text[i : i + len(normalized_chunk) + 20]  # Add some buffer
                normalized_window = re.sub(r"\s+", " ", text_window).strip()

                if normalized_window.startswith(normalized_chunk):
                    chunk_start = i
                    break

                # If not found with window, try word by word matching
                if i == current_pos + 100:  # Limit detailed search to avoid performance issues
                    for j in range(current_pos, len(full_text) - 10):
                        # Try to match first word
                        if re.match(
                            r"\b" + re.escape(chunk_words[0]) + r"\b",
                            full_text[j : j + len(chunk_words[0]) + 5],
                        ):
                            # Check if subsequent words match
                            match_text = full_text[j : j + len(normalized_chunk) + 30]
                            normalized_match = re.sub(r"\s+", " ", match_text).strip()
                            if normalized_match.startswith(normalized_chunk):
                                chunk_start = j
                                break
                    break

        if chunk_start == -1:
            print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
            continue

        # For the end position, find where the content of the chunk ends in the full text
        chunk_content = re.sub(r"\s+", "", chunk)  # Remove all whitespace
        full_text_from_start = full_text[chunk_start:]
        full_text_content = re.sub(
            r"\s+", "", full_text_from_start[: len(chunk) + 50]
        )  # Remove all whitespace

        # Find how many characters of content match
        content_match_len = 0
        for i in range(min(len(chunk_content), len(full_text_content))):
            if chunk_content[i] == full_text_content[i]:
                content_match_len += 1
            else:
                break

        # Map content length back to original text with whitespace
        chunk_end = chunk_start
        content_chars_matched = 0
        for i in range(len(full_text_from_start)):
            if chunk_end + i >= len(full_text):
                break
            if not full_text[chunk_start + i].isspace():
                content_chars_matched += 1
            if content_chars_matched > content_match_len:
                break
            chunk_end = chunk_start + i

        chunk_end += 1  # Include the last character
        # Do not strip newlines: step delimiters (e.g. \n\n) must stay inside the step
        # so token ranges are contiguous with no gaps. Only trim horizontal spaces.
        while chunk_end > chunk_start and full_text[chunk_end - 1] in " \t\r\v\f":
            chunk_end -= 1
        current_pos = chunk_end

        chunk_ranges.append((chunk_start, chunk_end))

    return chunk_ranges


def get_chunk_token_ranges(
    text: str, chunk_ranges: List[Tuple[int, int]], tokenizer: AutoTokenizer
) -> List[Tuple[int, int]]:
    """Convert character positions to token indices"""
    chunk_token_ranges = []

    for chunk_start, chunk_end in chunk_ranges:
        chunk_start_token = tokenizer.encode(text[:chunk_start], add_special_tokens=False)
        chunk_start_token_idx = len(chunk_start_token)
        chunk_end_token = tokenizer.encode(text[:chunk_end], add_special_tokens=False)
        chunk_end_token_idx = len(chunk_end_token)
        chunk_token_ranges.append((chunk_start_token_idx, chunk_end_token_idx))

    return chunk_token_ranges
