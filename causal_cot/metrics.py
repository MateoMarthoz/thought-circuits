"""
Phase 7: Metrics and baseline interventions.
calculate_step_loss, generate_resampled_continuation.
"""

import math

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import StoppingCriteria, StoppingCriteriaList

from .step_utils import extract_think_block, parse_reasoning_steps

_SAFETY_MAX_NEW_TOKENS = 512


def calculate_step_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    step_start_idx: int,
    step_end_idx: int,
) -> torch.Tensor:
    """
    Mean cross-entropy over a reasoning step for a causal LM.

    The first token of the step span is a boundary token (e.g. ``\\n\\n`` prefix carried
    from the previous step); we do **not** take loss for predicting it. We **do** use
    logits at that position to predict the second step token onward: logits
    ``[start : end-1)`` vs labels ``[start+1 : end)``.
    """
    if step_end_idx <= step_start_idx + 1:
        return logits.sum() * 0.0
    step_logits = logits[step_start_idx : step_end_idx - 1].contiguous()
    step_labels = labels[step_start_idx + 1 : step_end_idx].contiguous()
    return F.cross_entropy(step_logits, step_labels, reduction="mean")


class _StepBoundaryStoppingCriteria(StoppingCriteria):
    """
    Stop when ``parse_reasoning_steps(extract_think_block(decoded_full))`` has
    more steps than after the prefix alone—i.e. the step at ``step_index`` is
    complete per the same rules as splitting a full CoT (``\\n\\n`` opens the
    next step; in a colon block blank lines do not split until a line ends with
    ``.`` or ``:``, and the following step's leading ``\\n`` is not part of the
    finished step).
    """

    def __init__(
        self,
        prefix_len: int,
        tokenizer,
        batch_size: int,
        step_index: int,
        prefix_ids: torch.LongTensor,
    ):
        self.prefix_len = prefix_len
        self.tokenizer = tokenizer
        self.done = [False] * batch_size
        self.step_index = step_index
        prefix_decoded = tokenizer.decode(
            prefix_ids[0], skip_special_tokens=True
        )
        n_prefix = len(parse_reasoning_steps(extract_think_block(prefix_decoded)))
        if n_prefix != step_index:
            raise ValueError(
                f"Resample prefix decodes to {n_prefix} reasoning steps inside think, "
                f"expected {step_index} (step_index for the step being generated)."
            )

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            full_decoded = self.tokenizer.decode(
                input_ids[i], skip_special_tokens=True
            )
            n_full = len(parse_reasoning_steps(extract_think_block(full_decoded)))
            if n_full > self.step_index:
                self.done[i] = True
        return all(self.done)


def generate_resampled_continuation(
    prefix_text: str,
    original_step_text: str,
    model,
    tokenizer,
    step_index: int,
    num_continuations: int = 20,
    embedder=None,
    prefix_ids: torch.Tensor | None = None,
) -> str:
    """
    Generate num_continuations from prefix, extract step ``step_index`` (0-based)
    from each full decoded string via ``parse_reasoning_steps(extract_think_block(·))``,
    return the candidate step with largest semantic distance from ``original_step_text``.

    Stopping uses the same step boundaries as splitting a full CoT (not
    ``parse_reasoning_steps`` on the raw continuation fragment, which would reset
    colon-block state).

    Args:
        step_index: Index of the reasoning step being resampled (0 .. N-1).
        embedder: Pre-loaded SentenceTransformer to avoid reloading per call.
        prefix_ids: Pre-tokenized prefix tensor (1, seq_len) on model device.
            When provided, ``prefix_text`` is still used only if needed for docs;
            boundaries are detected from decoded tensors.
    """
    if prefix_ids is not None:
        input_ids = prefix_ids
    else:
        input_ids = tokenizer(
            prefix_text,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(model.device)
    prefix_len = input_ids.shape[1]

    seqs_per_batch = 5
    num_batches = math.ceil(num_continuations / seqs_per_batch)
    chopped_continuations = []

    with torch.no_grad():
        for _ in range(num_batches):
            stopping_criteria = StoppingCriteriaList([
                _StepBoundaryStoppingCriteria(
                    prefix_len,
                    tokenizer,
                    seqs_per_batch,
                    step_index,
                    input_ids,
                )
            ])
            outputs = model.generate(
                input_ids,
                max_new_tokens=_SAFETY_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                num_return_sequences=seqs_per_batch,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            for i in range(outputs.shape[0]):
                full_decoded = tokenizer.decode(
                    outputs[i], skip_special_tokens=True
                )
                parsed_steps = parse_reasoning_steps(
                    extract_think_block(full_decoded)
                )
                if len(parsed_steps) <= step_index:
                    chopped_continuations.append(
                        tokenizer.decode(
                            outputs[i, prefix_len:], skip_special_tokens=True
                        )
                    )
                else:
                    chopped_continuations.append(parsed_steps[step_index])

    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    orig_emb = embedder.encode([original_step_text])
    cont_embs = embedder.encode(chopped_continuations)
    similarities = cosine_similarity(orig_emb, cont_embs)
    distances = 1.0 - similarities
    best_idx = int(distances.argmax())
    return chopped_continuations[best_idx]
