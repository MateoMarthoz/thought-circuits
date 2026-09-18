# Metrics and Intervention Methods (Code-Faithful Specification)

This document is an archive-era implementation note. During curation, reusable
modules were moved into `causal_cot/`, with gradient helpers under
`causal_cot/gradient/`; scientific behavior was not changed.

It is intended to mirror implementation behavior in:

- `causal_cot/step_utils.py`
- `causal_cot/text_utils.py`
- `causal_cot/metrics.py`
- archive-only sparse-logit evaluation utilities (not retained in this repository)
- `causal_cot/interventions/kv_perturbation.py`
- `causal_cot/gradient/pruning_ssep.py`
- `causal_cot/gradient/hooks_jsep.py`
- `causal_cot/gradient/l0_utils.py`
- `causal_cot/perturbations/mlm.py`
- `experiments/run_backward_ablation.py`
- `experiments/run_joint_baselines.py`

All details below are implementation-level, including edge-case behavior.

Thought Anchors-derived utility provenance is recorded in `docs/ATTRIBUTION.md`.
Those upstream references are not separate local directories in this repository.

---

## 1. Fundamental Infrastructure

### 1.1 Think Block Extraction (`extract_think_block`)

- Regex used: `THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)`.
- If a think block exists, the exact inner substring (`group(1)`) is returned.
- No whitespace stripping is done.
- If no think block exists, the full original string is returned unchanged.

This "exact inner text" behavior is required for stable parser boundaries and character/token alignment.

### 1.2 Reasoning Step Parser (`parse_reasoning_steps`)

The parser is line-oriented (`split("\n")`) and stateful. It tracks:

- `current`: list of lines for the active step
- `leading_empty`: blank lines before first non-empty line
- `in_colon_block`: whether parser is in colon continuation mode
- `blank_run`: count of consecutive post-flush blank lines

Implemented rules:

1. **Blank line while in colon block**
   - Appended to `current`.
   - Does not split steps.

2. **Blank line outside colon block with non-empty `current`**
   - If future non-empty line exists, flush current step and start `blank_run = 1`.
   - If no future non-empty line, blank line is appended to current step.

3. **Additional blank lines after flush**
   - Increment `blank_run`.
   - Those newlines are owned by the next non-empty line.

4. **Non-empty line while in colon block**
   - If line ends with `"."` or `":"`, flush current step and start next step at `"\n" + line`.
   - Colon mode remains active only if new line ends with `":"`.
   - Otherwise append line to current step.

5. **Non-empty line outside colon block**
   - If `blank_run > 0`, prefix line with `"\n" * (blank_run + 1)`.
   - Append to current step.
   - Enter colon mode if line ends with `":"`.

Finalization:

- If `current` is non-empty, append it as last step.
- Else if only `leading_empty` exists, that becomes one step.

### 1.3 Character and Token Step Mapping

Step mapping is performed as:

1. `get_chunk_ranges(full_text, chunks)`
   - sequentially finds each chunk from `current_pos` onward
   - tries exact match first
   - includes normalized whitespace fallback logic if exact match fails
   - preserves step delimiters (trims only horizontal trailing whitespace)

2. `_expand_chunk_ranges_to_think_inner`
   - expands first chunk start down to think-inner start
   - expands last chunk end up to think-inner end

3. `get_chunk_token_ranges`
   - converts each char span to token span by counting tokenizer output lengths for string prefixes (`add_special_tokens=False`)

Core invariants enforced by validation:

- contiguous step token spans (`end_k == start_{k+1}`)
- union equals think-inner token span exactly
- no token leakage into prompt/post ranges in `assign_token2step_tensor`

### 1.4 Perturbation Alignment Contract

The code assumes (and repeatedly checks):

- same number of parsed reasoning steps in clean and perturbed traces
- same per-step token ranges
- same total tokenized sequence length

Interpretation:

- perturbations may change token identity at selected slots,
- but cannot change token positions, global length, or step boundaries.

---

## 2. Metrics

### 2.1 Per-Step Mean Cross-Entropy (`calculate_step_loss`)

Function: `causal_cot/metrics.py`.

Given step span `[s, e)`:

- uses logits slice `logits[s : e-1]`
- predicts labels slice `labels[s+1 : e]`
- returns `F.cross_entropy(..., reduction="mean")`

If `e <= s + 1`, returns differentiable zero: `logits.sum() * 0.0`.

### 2.2 Per-Step Joint Log-Probability (`sum_logprob_at_step`)

Function: `causal_cot/joint_token_likelihood.py`.

Given `[s, e)`:

- logits slice: `[s : e-1]`
- labels slice: `[s+1 : e]`
- logits cast to `float64`
- compute `log_softmax`, gather gold indices, sum

Returns Python `float`.

### 2.3 Delta CE and Relative Perplexity

For a source-target intervention:

- `delta = CE_mixed(step_j) - CE_clean(step_j)`
- average over perturbed variants where needed
- relative perplexity form is `exp(mean_delta)`

### 2.4 Joint Probability Ratio and Baseline Normalization

Implemented in:

- `joint_metric_from_mean_ce_delta`
- `normalize_joint_from_logprobs`

If baseline table exists:

- approximate mixed log-joint:
  - `log_p_mixed_approx = log_p_clean - K * mean_delta`
- normalized joint:
  - `(exp(log_p_mixed_approx) - exp(log_p_base)) / (exp(log_p_clean) - exp(log_p_base))`

Numerics:

- exponential conversion performed using `float64`
- denominator must be finite and strictly greater than `1e-300`
- otherwise output is `NaN`

Fallback without baseline table:

- `exp(-K * mean_delta)` (legacy behavior)

### 2.5 Sparse KL Divergence (`calculate_kl_divergence_sparse`)

This archive-only utility is not retained in the cleaned repository.

Input:

- `baseline_data = (indices, logits)`
- `suppressed_data = (indices, logits)`

Algorithm:

1. Validate non-null and shape consistency.
2. Build union index set via `np.union1d`.
3. Fill dense union-space logits with `-1e9` default.
4. Insert provided logits at matching union positions.
5. Compute log-probs with temperature 0.6.
6. Compute KL terms `P * (logP - logQ)`.
7. Force terms with `P == 0` to 0.
8. Sum.

Edge handling:

- invalid values produce `np.nan`
- if result is slightly negative, clip to 0
- strong negative values print a warning before clipping

### 2.6 Top-p Logit Compression (`compress_logits_top_p`)

This archive-only utility is not retained in the cleaned repository.

For each sequence position:

1. compute probs from `softmax(logits / 0.6)`
2. sort descending
3. find smallest `k_p` where cumulative prob reaches `p`
4. set `k = min(k_p, max_k)`, then clamp to at least 1
5. store top-k token indices (`int32`) and raw logits (`float16`)

Output schema:

- `flat_indices`
- `flat_logits`
- `offsets` (length `seq_len + 1`)
- `cum_probs_retained`

Position `t` data is `offsets[t]:offsets[t+1]`.

---

## 3. Intervention Methods

### 3.1 Attention Suppression (`CumulativeAttnSuppressionContext`)

Implementation location: `causal_cot/interventions/attention_suppression.py`.

Per attention layer:

1. compute Q/K/V projections and apply RoPE
2. if cache exists, concatenate cached K/V with current K/V
3. compute attention score matrix
4. build causal mask manually from absolute positions:
   - `query_abs = [prefix_len, ..., prefix_len + q_len - 1]`
   - allow key `k` only if `k <= query_abs`
5. for each suppression span `(src_start, src_end)`:
   - `q_from = max(0, src_end - prefix_len)`
   - set `attn[:, :, q_from:, src_start:src_end] = -1e4`
6. softmax and value projection as normal

Interpretation:

- tokens at absolute positions `< src_end` can still read suppressed span
- tokens at absolute positions `>= src_end` cannot read suppressed span
- this is attention-weight suppression only; hidden states are not replaced

### 3.2 KV Substitution (`PerturbedKVPostSourceContext`)

Implementation location: `causal_cot/interventions/kv_perturbation.py`.

Batch behavior:

- `bsz == 2` required for splice mode
  - index 0: clean
  - index 1: perturbed
- if `bsz != 2`, layer uses original forward unchanged

Per-layer splice logic:

1. project Q/K/V and apply RoPE
2. append cache K/V if present
3. expand grouped KV heads using `repeat_kv`
4. split K/V into clean and perturbed branches
5. clone clean K/V to mixed K/V
6. overwrite only selected key/value token spans in mixed K/V with perturbed K/V
7. compute clean-branch attention output for all queries against clean K/V
8. recompute query tail `[query_mix_start:q_len)` against mixed K/V and overwrite that tail
9. run perturbed branch independently via original forward on batch index 1
10. concatenate branch outputs and return

`query_mix_start_for_kv_substitution`:

- if only candidate step substituted: `mix_start = end(candidate)`
- if multiple substituted steps: `mix_start = min(start(step) for step in substituted)`

Important behavioral clarification:

- splice is token-local in K/V space (only selected source spans overwritten),
- but it applies to all queries from `mix_start` onward,
- and because layers are patched throughout a forward pass, downstream hidden states are recursively affected across layers.

### 3.3 Single-Target L0 Interpolation (SSEP, `CausalEdgePrunerSSEP`)

Implementation: `causal_cot/gradient/pruning_ssep.py`.

Setup:

- optimize one target step `k`
- learn gate vector `z` for source steps `1..k-1`
- each source step has one scalar gate shared across all its key tokens

Hard-concrete utilities (from `l0_utils.py`):

- constants:
  - `LIMIT_LEFT = -0.1`
  - `LIMIT_RIGHT = 1.1`
  - `TEMPERATURE = 2/3`
- sampling:
  - sample `u ~ Uniform(EPS, 1-EPS)`
  - stretched-sigmoid transform
  - clamp with hardtanh to `[0,1]`
- deterministic extraction:
  - infer expected zero count from CDF at 0
  - zero the lowest soft mask entries

SSEP attention semantics (`bsz == 2`):

- token2step map assigns each key token its step id
- key token gets learnable gate only if key step is in `1..k-1`
- for query position `q`, gate is applied to that key only if `q >= end(key_step)`
- this means a source step, when gated, affects all subsequent query tokens (not only target-step tokens)

Computation form per chunk:

- scores:
  - `S_f = S_p + Z * (S_c - S_p)`
- outputs:
  - `A_Z = A_f * Z`
  - `O = A_Z @ V_clean + (A_f - A_Z) @ V_pert`

Perturbed branch:

- uses `scaled_dot_product_attention` directly on perturbed Q/K/V

Training objective per perturbed sample:

- `delta_loss = CE_mixed(step_k) - CE_clean(step_k)`
- `sparsity = 1 - mean(z)`
- `target` linearly ramped during warmup
- regularizer:
  - `lambda1 * (sparsity - target)`
  - `+ lambda2 * (sparsity - target)^2`
- total loss is averaged across number of perturbations

Checkpointing:

- interval-based on edge sparsity
- within each interval band, best (lowest faithfulness loss) checkpoint is kept

Edge extraction after optimization:

- deterministic `z` is computed
- edge `(i, k)` exists if deterministic gate for source step `i` is `> 0`

### 3.4 Joint L0 Interpolation (JSEP, `CausalEdgePrunerJSEP`)

Implementation: `causal_cot/gradient/hooks_jsep.py`.

Parameters:

- `log_alpha`: matrix `(N+1, N+1)` (prompt index 0 plus reasoning steps 1..N)
- active learnable edges are strict upper triangle
- row/col 0 and diagonal are forced to 1 via `torch.where`

Optional node sparsity:

- `node_log_alpha` for intermediate nodes `1..N-1`
- sampled node gates multiply outgoing rows before forced-1 masking

Attention mixing:

1. compute token2step mapping
2. per query chunk, build token-token gate block using:
   - `Z_c = Z.T[step_q, step_k]`
3. interpolate scores:
   - `S_f = S_p + Z_c * (S_c - S_p)`
4. interpolate outputs:
   - `O = (A_f * Z_c) @ V_clean + (A_f - A_f * Z_c) @ V_pert`

Implementation is chunked to reduce VRAM pressure and avoid full dense token-token gate materialization.

Usage probabilities:

- computed by backward recurrence in log space in `calculate_usage_probabilities`
- recurrence clamps inner argument to at least `1e-8` before `log`
- returns `U = 1 - exp(v)`

Faithfulness aggregation modes:

- `weighted_sum`: `sum(U_k * delta_k)`
- `squared_sum`: `sum(ReLU(U_k * delta_k)^2)`

Total objective:

- faithfulness term
- edge sparsity regularizer
- optional node sparsity regularizer
- edge and node targets are warmup-ramped separately

---

## 4. Text-Level Interventions

### 4.1 LLM Resampling (`generate_resampled_continuation`)

Implementation: `causal_cot/metrics.py`.

Procedure:

1. encode prefix ids (or use provided prefix ids)
2. generate in mini-batches of 5 sequences until desired count (default 20)
3. stopping criterion `_StepBoundaryStoppingCriteria`:
   - decodes full sequence at each generation step
   - computes `len(parse_reasoning_steps(extract_think_block(decoded)))`
   - marks sequence done once parsed step count exceeds `step_index`
4. for each generated sequence:
   - if parsed step exists, take `parsed_steps[step_index]`
   - else fallback to raw generated suffix decode
5. embed candidate steps and original step with `all-MiniLM-L6-v2`
6. pick candidate with maximum cosine distance

Generation parameters in code:

- `temperature=0.6`
- `top_p=0.95`
- `max_new_tokens=512`

### 4.2 MLM Token-Aligned Perturbation (`generate_perturbed_cots`)

Implementation: `causal_cot/perturbations/mlm.py`.

Model:

- HuggingFace fill-mask pipeline with `FacebookAI/roberta-large`

Candidate pool:

- words strictly inside think block
- regex `\b[A-Za-z0-9]+\b`
- keep if numeric or length > 3
- each word mapped once to stable LLM token slot `[t_start, t_end)` in full text

Per-variant loop:

1. start `current = original_cot`
2. randomize candidate order
3. for each candidate slot:
   - recompute current character span for slot using fresh offset mapping
   - replace span with MLM mask token
   - extract ±250 MLM-token context window
   - query top-k MLM predictions
   - filter predictions by:
     - alphanumeric only
     - not equal to original (case-insensitive)
     - capitalization compatibility
     - no substring containment in either direction
     - preserve original leading/trailing whitespace
     - strict token-index invariance test:
       - same total LLM-token length
       - identical token IDs outside `[t_start, t_end)`
   - apply first valid replacement
4. write variant to `perturbed_{v}.txt`

Alignment consequence:

- every accepted replacement is validated to preserve all outside-slot token IDs,
- so repeated substitutions maintain global token alignment needed by KV-based interventions.
# Metrics and Intervention Methods

## 1. Fundamental Infrastructure

### 1.1 Reasoning Step Parser (`parse_reasoning_steps`)

The function `parse_reasoning_steps(cot_text: str) -> List[str]` takes the raw string inside the `<think>...</think>` block and splits it into a list of step strings. The splitting rules are:

**Rule 1 — Paragraph breaks.** A double newline (`\n\n`) in the raw text opens the *next* step, not the tail of the current step. The `\n\n` prefix is therefore the first characters of the new step string. Consecutive blank lines accumulate additional `\n` characters on the prefix of the new step.

**Rule 2 — Colon-block continuation.** When a non-empty line ends with `:`, the parser enters a colon block. Inside a colon block, blank lines do *not* create a new step. The colon block persists until a line ends with `.` or `:`. At that point, the accumulated colon block is flushed as a step and the new step begins with a single leading `\n`.

**Rule 3 — Period/colon terminators outside a colon block.** A non-empty line ending with `.` or `:` signals the potential end of the current step. The next non-empty line begins a new step.

The implementation iterates line by line (split on `\n`). Non-empty leading `\n` runs are accumulated in `blank_run` and prepended to the next non-empty line as `"\n" * (blank_run + 1)`. Step strings are assembled by joining their constituent lines with `"\n"`.

**Invariant.** The concatenation of all step strings, joined with `"\n"`, reproduces the original `cot_text` exactly. Each step string therefore owns its leading delimiter characters.

### 1.2 Step-to-Token Mapping (`map_steps_to_tokens`)

Token ranges are computed from character ranges:

1. `get_chunk_ranges(full_text, steps)` locates each step's substring span `[char_start, char_end)` inside `full_text` by sequential forward-search (each step must appear after the previous one, with no backtracking).
2. `_expand_chunk_ranges_to_think_inner` extends the first step's `char_start` down to the `<think>` inner boundary and the last step's `char_end` up to the `</think>` inner boundary, ensuring no character inside the think block is unassigned.
3. `get_chunk_token_ranges` converts each character range to a token index range by encoding the prefix `full_text[:char_start]` and `full_text[:char_end]` and taking their lengths (using `add_special_tokens=False`).

The result is a list of half-open token index pairs `(s, e)`, one per step, satisfying:

```
step_ranges[k][1] == step_ranges[k+1][0]    for all k
step_ranges[0][0] == think_inner_token_start
step_ranges[-1][1] == think_inner_token_end
```

This contiguity is verified by `assert_reasoning_token_ranges_match_think_inner` before any intervention or metric computation.

### 1.3 Alignment Constraint for Perturbed Variants

Every perturbed or resampled CoT variant must satisfy:

- **Step count equality.** `parse_reasoning_steps(extract_think_block(text))` returns the same number of steps as the clean CoT.
- **Token span identity.** The per-step token ranges `(s_i, e_i)` computed for the variant match those of the clean CoT exactly.
- **Total sequence length identity.** The perturbed tokenized sequence has the same length as the clean sequence.

These constraints mean the only thing that changes between the clean and perturbed CoT, from the perspective of token indices, is the *identities* of the tokens at certain positions — never their positions or boundaries.

---

## 2. Metrics

### 2.1 Per-Step Mean Cross-Entropy Loss

**Function.** `calculate_step_loss(logits, labels, step_start_idx, step_end_idx) -> torch.Tensor`

**Inputs.**
- `logits`: shape `(seq_len, vocab_size)` — raw model output logits for the full sequence.
- `labels`: shape `(seq_len,)` — integer token ids for the full sequence.
- `step_start_idx` (s), `step_end_idx` (e): half-open token range of the step.

**Formula.**

$$\mathcal{L}_{\text{step}} = \frac{1}{e - s - 1} \sum_{t=s}^{e-2} \text{CrossEntropy}\!\left(\text{logits}[t],\; \text{labels}[t+1]\right)$$

Equivalently, the logit slice used for prediction is `logits[s : e-1]` and the label slice is `labels[s+1 : e]`, passed jointly to `F.cross_entropy(..., reduction="mean")`.

**Rationale for the offset.** The first token at position `s` is a boundary delimiter (e.g. a `\n\n` prefix) carried over from the previous step. The loss for *predicting* this boundary token was already charged to the previous step; here it serves only as context. We therefore start predicting from position `s+1` onward, using the logit at position `s` as its predictor.

If `e <= s + 1` (step contains only the boundary token), the function returns a zero tensor without propagating gradients into the loss.

### 2.2 Per-Step Sum of Log-Probabilities

**Function.** `sum_logprob_at_step(logits, labels, step_start, step_end) -> float`

**Inputs.** Same as `calculate_step_loss`.

**Formula.**

$$\Lambda_{\text{step}} = \sum_{t=s}^{e-2} \log P(x_{t+1} \mid x_{\leq t})
= \sum_{t=s}^{e-2} \left[\log\text{-softmax}(\text{logits}[t])\right]_{\text{labels}[t+1]}$$

where the log-softmax is computed in `float64` for numerical stability.

This is the log of the joint probability of all tokens in the step given their respective prefixes:

$$\Lambda_{\text{step}} = \log \prod_{t=s+1}^{e-1} P(x_t \mid x_{<t})$$

### 2.3 Loss Delta and Relative Perplexity

Given an intervention applied to source step $i$ and a target step $j > i$, the **cross-entropy delta** for one perturbed variant $p$ is:

$$\delta^{(p)}(i \to j) = \mathcal{L}_{\text{step}_j}(\text{mixed}_p) - \mathcal{L}_{\text{step}_j}(\text{clean})$$

where $\mathcal{L}_{\text{step}_j}(\cdot)$ is the mean cross-entropy on step $j$'s tokens under the respective forward pass.

Averaged over $P$ perturbed variants:

$$\bar\delta(i \to j) = \frac{1}{P} \sum_{p=1}^{P} \delta^{(p)}(i \to j)$$

The **relative perplexity** is:

$$\text{RelPPL}(i \to j) = \exp\!\left(\bar\delta(i \to j)\right)$$

A value greater than 1 means step $j$ becomes harder to predict when step $i$'s representations are corrupted.

### 2.4 Normalized Joint Probability Ratio

Let $K_j = e_j - s_j - 1$ be the number of prediction positions in step $j$. The **approximated mixed log-probability** is:

$$\log \hat{p}_{\text{mixed}}(j) \approx \Lambda_{\text{clean}}(j) - K_j \cdot \bar\delta(i \to j)$$

This approximation treats the mean cross-entropy delta as uniform across the $K_j$ positions, which is exact only in the limit of large steps. It is used when a direct mixed forward has not been run on step $j$.

**Precomputed baselines.** `run_joint_baselines.py` computes two reference values for each step $j \geq 1$:

- $\Lambda_{\text{clean}}(j)$: log joint probability of step $j$'s tokens under the unmodified clean CoT.
- $\Lambda_{\text{base}}(j)$: log joint probability under a forward pass where *all* steps strictly before $j$ have their K/V representations corrupted (i.e., substituted from a perturbed CoT using `PerturbedKVPostSourceContext` with the full span $\{0, 1, \ldots, j-1\}$). This serves as a lower bound — the expected log-probability when the model has no useful context from any earlier step.

**Normalized ratio.**

$$\text{JointNorm}(i \to j) = \frac{p_{\text{mixed}}(j) - p_{\text{base}}(j)}{p_{\text{clean}}(j) - p_{\text{base}}(j)}$$

where all probabilities are in linear space: $p = \exp(\Lambda)$. Computation is done in `float64` to handle the very small values that arise from long step sequences. The denominator is clamped away from zero by $\epsilon = 10^{-300}$; if the denominator is non-positive or the quantities are not finite, the function returns `NaN`.

A value of 1 means step $j$ is equally predictable with or without corruption of step $i$ (relative to the all-corrupted baseline). A value of 0 means step $j$ is as hard to predict as when all context is corrupted.

**Legacy (no-baseline) fallback.** When no baseline table is available, `joint_metric_from_mean_ce_delta` returns:

$$\text{JointLegacy}(i \to j) = \exp\!\left(-K_j \cdot \bar\delta(i \to j)\right) = \frac{\hat{p}_{\text{mixed}}(j)}{p_{\text{clean}}(j)}$$

### 2.5 KL Divergence on Sparse Top-p Logit Distributions

**Function.** `calculate_kl_divergence_sparse(baseline_data, suppressed_data, temperature=0.6) -> float`

**Inputs.** Each of `baseline_data` and `suppressed_data` is a tuple `(indices: np.int32, logits: np.float16)` representing a *sparse* distribution over the vocabulary: only a top-p subset of tokens are stored, and all other tokens are assigned logit $-10^9$.

**Algorithm.**

1. Compute the union $\mathcal{U}$ of all token indices present in either distribution.
2. Build dense float32 logit vectors over $\mathcal{U}$ by placing stored logits at their positions and $-10^9$ elsewhere.
3. Convert to log-probabilities: $\log P_k = \log\text{-softmax}(\mathbf{b} / \tau)_k$ and $\log Q_k = \log\text{-softmax}(\mathbf{s} / \tau)_k$ where $\tau = 0.6$.
4. Compute $P_k = \exp(\log P_k)$.
5. Compute KL terms: $\sum_k P_k (\log P_k - \log Q_k)$, zeroing terms where $P_k = 0$ to avoid $0 \cdot (-\infty)$.
6. Clip small negative values (numerical noise below $-10^{-6}$ triggers a warning) to 0.

**Returns.** $\text{KL}(P \| Q) \geq 0$, or `np.nan` if inputs are invalid or intermediate values are non-finite.

### 2.6 Top-p Logit Compression (`compress_logits_top_p`)

**Purpose.** Store only the top-$p$ nucleus tokens per sequence position to enable efficient on-disk storage and KL comparison.

**Inputs.**
- `logits`: shape `(1, seq_len, vocab_size)`, float32.
- `p`: cumulative probability cutoff (default 0.999).
- `max_k`: hard cap on tokens kept per position (default 100).

**Algorithm.** For each position $t$:

1. Compute $\text{probs}[t] = \text{softmax}(\text{logits}[t] / 0.6)$.
2. Sort probabilities descending to get `sorted_probs` and `sorted_indices`.
3. Find $k_p = \min\{k : \sum_{i=1}^{k} \text{sorted\_probs}[i] \geq p\}$ via binary search on the cumulative sum.
4. Retain $k = \min(k_p, \text{max\_k})$ tokens.
5. Append their original vocab indices and *raw* (unscaled) logits to flat arrays.

**Output format.**

```
{
    'flat_indices':        np.int32  of shape (total_kept,)
    'flat_logits':         np.float16 of shape (total_kept,)
    'offsets':             np.int32  of shape (seq_len + 1,)
    'cum_probs_retained':  np.float32 of shape (seq_len,)
}
```

Position $t$'s data lives at `flat_indices[offsets[t] : offsets[t+1]]`. Logits are stored as float16 to save memory; they are cast back to float32 before KL computation. The temperature $\tau = 0.6$ is applied during compression (for top-$p$ selection) and again during KL computation (for softmax normalization).

---

## 3. Intervention Methods

### 3.1 Attention Suppression (`CumulativeAttnSuppressionContext`)

**Mechanism.** A context manager that monkey-patches every attention layer's `forward` method. The patched forward replaces the model's built-in attention mask with a custom causal mask and then adds large negative values on selected source spans.

**Patch logic for one layer.** Given suppression spans $\{(s^{(r)}_{\text{start}}, s^{(r)}_{\text{end}})\}_{r=1}^R$ (token index ranges to suppress):

1. Compute queries, keys, and values using the layer's projection matrices. Apply rotary position embeddings.
2. If KV prefix caching is active (`past_key_value` is provided), concatenate cached keys/values before the current suffix keys/values. Let `prefix_len` denote the number of cached key positions.
3. Compute raw attention scores: $A = Q K^T / \sqrt{d_{\text{head}}}$.
4. Build a causal mask from absolute positions: `key_positions = range(kv_len)`, `query_abs = range(prefix_len, prefix_len + q_len)`. Position $q$ can attend to position $k$ iff $k \leq q$.
5. For each suppression span $(s_{\text{start}}, s_{\text{end}})$:
   - Compute `q_from = max(0, s_end - prefix_len)` — the first *local* query index (relative to the suffix) at or after the span's end.
   - Set `A[:, :, q_from:, s_start:s_end] = -1e4`.
   - This ensures only queries at absolute positions $\geq s_{\text{end}}$ have their attention to the suppressed span zeroed out; queries whose absolute position is inside or before the span are unaffected.
6. Apply softmax and multiply by values as usual.

**Interpretation.** Attention suppression prevents information in the suppressed step's token positions from being read by any later token. Because it only zeroes out attention weights (rather than patching hidden states), it does not require a second forward pass over the perturbed CoT. The intervention is purely at the attention weight level; residual stream contributions from the suppressed step tokens themselves (from earlier layers) are not removed.

**KV prefix caching.** The clean prefix KV cache can be computed once and reused: since the suppressed spans must lie within the suffix (they are candidate reasoning step spans, which always follow the prompt), the prefix keys and values are clean and unaffected. The suffix starting at `prefix_len = step_ranges[candidate][0]` is re-run with the patched attention.

### 3.2 K/V Substitution via Perturbed Forward (`PerturbedKVPostSourceContext`)

**Mechanism.** A context manager that monkey-patches every attention layer's `forward` method. It requires a batch of size exactly 2: index 0 is the clean sequence and index 1 is the token-aligned perturbed sequence. For each designated source span, the clean branch's keys and values are replaced by those from the perturbed branch.

**Inputs.**
- `source_spans`: list of `(src_start, src_end)` token ranges to splice.
- `query_mix_start`: the first query position at which the spliced K/V is used.

**Patch logic for one layer.**

1. Project and RoPE-encode queries, keys, and values for both batch elements. Let $K_c, V_c$ be clean and $K_p, V_p$ be perturbed (shape: `(1, num_heads, seq_len, head_dim)` each after GQA expansion).
2. Build mixed keys/values:
   $$K_m = K_c.\text{clone}(); \quad K_m[:, :, s_{\text{start}}:s_{\text{end}}, :] = K_p[:, :, s_{\text{start}}:s_{\text{end}}, :]$$
   and likewise for $V_m$. This splice is applied for every span in `source_spans`.
3. Compute attention for queries in `[0, query_mix_start)` against the fully clean K/V:
   $$O_{\text{clean}}[:, :, 0:\text{mix\_start}, :] = \text{softmax}\!\left(\frac{Q_c[:, :, 0:\text{mix\_start}, :] \cdot K_c^T}{\sqrt{d}}\right) \cdot V_c$$
4. Compute attention for queries in `[query_mix_start, q_len)` against the mixed K/V:
   $$O_{\text{mixed}}[:, :, \text{mix\_start}:, :] = \text{softmax}\!\left(\frac{Q_c[:, :, \text{mix\_start}:, :] \cdot K_m^T}{\sqrt{d}}\right) \cdot V_m$$
5. The final output for the clean branch is $O = [O_{\text{clean}}, O_{\text{mixed}}]$ (concatenated along the sequence dimension), then passed through `o_proj`.
6. The perturbed branch is computed independently via the original unpatched `forward` on batch index 1.
7. Return the concatenated output: `[clean_branch_output, perturbed_branch_output]`.

**`query_mix_start_for_kv_substitution` policy.** The value of `query_mix_start` is computed as follows. Let `candidate` be the step whose K/V is currently being substituted and `substituted` be the full set of steps being substituted (candidate plus any previously accumulated steps):

- If `substituted = {candidate}` (no prior substitutions): `query_mix_start = step_ranges[candidate][1]`. Only queries strictly after the source step use mixed K/V.
- If `substituted` contains prior steps: `query_mix_start = min(step_ranges[j][0] for j in substituted)`. Every query at or after the earliest substituted step uses mixed K/V, ensuring no query reads clean K/V from any already-substituted step.

**Interpretation.** This intervention models what the model "would have computed" had the content of certain steps come from the perturbed CoT while everything else — including the token positions, prompt, and any unaffected steps — remained from the clean CoT. The clean branch's residual stream carries only clean token embeddings; the perturbed information enters exclusively through the keys and values at the designated span positions.

### 3.3 L0-Learned K/V Interpolation — Single Target Step

**Overview.** For a fixed target step $k$, a scalar gate $z_i \in [0, 1]$ is learned for each source step $i < k$. The gate interpolates between clean and perturbed keys and values for all tokens in step $i$. The objective is to minimize the faithfulness loss (cross-entropy increase on step $k$) subject to a sparsity regularizer on the gates.

#### 3.3.1 Hard Concrete Distribution

The gates are reparameterized via the Stretched Hard Concrete distribution with fixed constants $\zeta = -0.1$ (LIMIT_LEFT), $\gamma = 1.1$ (LIMIT_RIGHT), $\beta = 2/3$ (TEMPERATURE):

**Sampling.** Given log-parameter $\log\alpha_i$, draw $u \sim \text{Uniform}(\epsilon, 1-\epsilon)$:

$$s_i = \sigma\!\left(\frac{\log u - \log(1-u) + \log\alpha_i}{\beta}\right)$$

$$\tilde{z}_i = (\gamma - \zeta) \cdot s_i + \zeta$$

$$z_i = \text{HardTanh}(\tilde{z}_i,\; 0,\; 1)$$

Probabilities: $P(z_i = 0) = \sigma\!\left(\beta \log\!\left(\frac{-\zeta}{\gamma}\right) - \log\alpha_i\right)$. High $\log\alpha_i$ yields $z_i$ close to 1 (clean K/V); low $\log\alpha_i$ yields $z_i$ close to 0 (perturbed K/V).

**Deterministic evaluation.** After training, `deterministic_z_from_log_alpha` computes expected number of zeros from the CDF at 0, identifies the $\lfloor\text{expected zeros}\rfloor$ entries with the smallest soft sigmoid values, and hard-zeros them. An edge $(i \to k)$ is declared present iff the deterministic $z_i > 0$.

#### 3.3.2 Attention Forward Pass with Z-Gated K/V

The patched forward (batch size 2: clean at index 0, perturbed at index 1) proceeds in chunks of 128 queries:

For query chunk $[q_{\text{lo}}, q_{\text{hi}})$ and key sequence length `kv_len`:

1. Compute `apply_mix[q, j]` = True iff key position $j$ belongs to a source step $i$ (i.e., $s_i \leq j < e_i$ for some $i \in \{1, \ldots, k-1\}$) **and** $q \geq e_i$ (query strictly after step $i$'s end).
2. Build per-position gate: $Z_{\text{chunk}}[q, j] = z_{\text{step}(j)}$ if `apply_mix[q,j]`, else $1.0$.
3. Compute pre-softmax scores: $S_f = S_p + Z_{\text{chunk}} \cdot (S_c - S_p)$, where $S_c = Q_c K_c^T / \sqrt{d}$ and $S_p = Q_c K_p^T / \sqrt{d}$.
4. Apply causal mask and softmax: $A_f = \text{softmax}(S_f)$.
5. Compute output: $O = A_f \cdot (Z_{\text{chunk}} \cdot V_c) + (A_f - A_f \cdot Z_{\text{chunk}}) \cdot V_p = A_f Z V_c + A_f(1-Z) V_p$.

This implements: for a given (query, key) pair, the key's value vector is $z \cdot V_c[j] + (1-z) \cdot V_p[j]$, and the attention weight $A_f[q,j]$ is computed from an interpolated score $z \cdot S_c[q,j] + (1-z) \cdot S_p[q,j]$.

The perturbed branch (index 1) uses standard `scaled_dot_product_attention` on the perturbed queries, keys, and values.

#### 3.3.3 Optimization

**Objective.** For each source perturbed variant, the total loss is:

$$\mathcal{L}_{\text{total}} = \underbrace{\left(\mathcal{L}_{\text{step}_k}^{\text{mixed}} - \mathcal{L}_{\text{step}_k}^{\text{clean}}\right)}_{\text{faithfulness (delta loss)}} + \underbrace{\lambda_1 (\hat{s} - \tau) + \lambda_2 (\hat{s} - \tau)^2}_{\text{sparsity regularizer}}$$

where $\hat{s} = 1 - \mathbb{E}[z]$ is the current edge sparsity (fraction of gates at zero) and $\tau$ is the target sparsity.

During a warmup period of `warmup_steps` steps, $\tau$ is linearly ramped from 0 to its final value. The Lagrange multipliers $\lambda_1, \lambda_2 \geq 0$ are learned parameters (updated via gradient ascent with respect to $-\mathcal{L}_{\text{total}}$).

**Sparsity-interval checkpointing.** Checkpoints are saved at each of $N-1$ evenly spaced sparsity targets, keeping the checkpoint with the lowest faithfulness delta at each interval for comparison across sparsity levels.

### 3.4 L0-Learned K/V Interpolation — Joint Across All Step Pairs

**Overview.** A global $(N+1) \times (N+1)$ matrix of log-parameters $\log\alpha_{ij}$ is optimized jointly over all source-target step pairs. Each off-diagonal upper-triangular entry $Z_{ij}$ (for $0 \leq i < j \leq N$) gates the K/V contribution from step $i$ to step $j$. The prompt (index 0) and diagonal entries are always fixed to 1.

#### 3.4.1 Z Matrix Sampling

```
Z_raw = sample_z_from_log_alpha(log_alpha)      # (N+1, N+1)
Z_upper = triu(Z_raw, diagonal=1)
if use_node_sparsity:
    z_nodes ~ Hard_Concrete(node_log_alpha)     # (N-1,) for steps 1..N-1
    node_gate = [1, z_nodes, 1]                 # step 0 (prompt) and step N always on
    Z_upper = Z_upper * node_gate.unsqueeze(1)  # gate entire rows
Z[row/col 0 or diagonal] = 1.0                  # force via torch.where
```

Node masks gate the *outgoing* edges from each intermediate reasoning step, implementing a form of step-level sparsity in addition to edge-level sparsity.

#### 3.4.2 Attention Forward Pass with Z Matrix

For each query chunk $[q_{\text{lo}}, q_{\text{hi}})$:

1. Look up the step id for each query position: `step_q = token2step[q]` for $q \in [q_{\text{lo}}, q_{\text{hi}})$.
2. Look up the step id for each key position: `step_k = token2step[j]` for all $j$.
3. Build the chunk gate: $Z_{\text{chunk}}[q, j] = Z^T[\text{step\_q}[q], \text{step\_k}[j]]$ (using the transpose because $Z[i,j]$ is indexed as source=row, target=col, and we want the weight for key step $i$, query step $j$).
4. Interpolated score: $S_f = S_p + Z_{\text{chunk}} \cdot (S_c - S_p)$.
5. Output: $O = A_f Z_{\text{chunk}} V_c + (A_f - A_f Z_{\text{chunk}}) V_p$.

The upper-triangular structure of $Z$ (combined with the causal mask) ensures that keys in step $i$ cannot affect queries in step $j \leq i$, so causality is preserved.

#### 3.4.3 Usage Probabilities and Faithfulness Loss

**Usage probability.** To weight the faithfulness loss by how "reachable" each step is from the output given the current $Z$, the model computes step-level usage probabilities $U_j$ for $j = 1, \ldots, N$.

Let $v_j$ denote the log-probability that step $j$ is *not* used (not reachable from the output). Working backward from $j = N-1$ down to $j = 1$:

$$v_j = \sum_{k=j+1}^{N} \log\!\left(1 - Z_{jk} (1 - e^{v_k})\right)$$

Then $U_j = 1 - e^{v_j}$.

**Global faithfulness loss.** The CE delta for each step $k$ is $\Delta_k = \mathcal{L}_{\text{step}_k}^{\text{mixed}} - \mathcal{L}_{\text{step}_k}^{\text{clean}}$. Two aggregation modes are supported:

- `weighted_sum`: $\mathcal{L}_{\text{faith}} = \sum_{k=1}^{N} U_k \cdot \Delta_k$
- `squared_sum`: $\mathcal{L}_{\text{faith}} = \sum_{k=1}^{N} \left(\text{ReLU}(U_k \cdot \Delta_k)\right)^2$

**Full objective.**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{faith}} + \lambda_1^e (\hat{s}_e - \tau_e) + \lambda_2^e (\hat{s}_e - \tau_e)^2 + \lambda_1^n (\hat{s}_n - \tau_n) + \lambda_2^n (\hat{s}_n - \tau_n)^2$$

where $\hat{s}_e$ is edge sparsity (mean of $1 - Z$ over the active upper triangle), $\hat{s}_n$ is node sparsity (mean of $1 - z_{\text{nodes}}$), and $\tau_e, \tau_n$ are their respective targets (also linearly ramped during warmup).

When `use_node_sparsity=False`, the node terms are dropped.

---

## 4. Text-Level Intervention Methods

### 4.1 LLM Resampling (`generate_resampled_continuation`)

**Purpose.** Generate an alternative text for a given reasoning step that is maximally semantically distant from the original, while being produced by the same model.

**Algorithm.**

1. **Tokenize the prefix.** The prefix is the full CoT up to (but not including) step `step_index`. Its token count is `prefix_len`.
2. **Generate $K$ continuations** (default $K = 20$, in batches of 5) using `model.generate` with:
   - Temperature $\tau = 0.6$, top-$p = 0.95$
   - `max_new_tokens = 512`
   - A custom `StoppingCriteria` that halts each sequence as soon as `parse_reasoning_steps(extract_think_block(decoded))` contains more than `step_index` steps — i.e., until step `step_index` is complete per the standard boundary rules.
3. **Extract step text.** For each generated sequence, decode the full output, parse reasoning steps inside the think block, and take `parsed_steps[step_index]`. If the step parser returns fewer steps than expected, the raw decoded suffix is used as a fallback.
4. **Select the most distant candidate.** Embed all $K$ candidates and the original step text using `all-MiniLM-L6-v2`. Compute cosine similarities. Return the candidate with maximum $1 - \text{cosine\_similarity}$.

**Stopping criteria detail.** The stopping criterion calls `parse_reasoning_steps(extract_think_block(tokenizer.decode(full_ids)))` on each live sequence at every generation step. The stopping fires when the number of parsed steps inside the think block exceeds `step_index`. This ensures that the stopping boundary is defined by exactly the same parser as used for token range computation.

### 4.2 MLM-Based Token-Aligned Perturbation (`generate_perturbed_cots`)

**Purpose.** Generate multiple semantically altered versions of the full CoT where each alteration preserves the exact tokenization at every position outside the replaced word.

**Model.** FacebookAI/roberta-large, used via HuggingFace `fill-mask` pipeline.

**Candidate words.** Words eligible for masking are alphanumeric tokens (`\b[A-Za-z0-9]+\b`) inside the `<think>...</think>` block that are either more than 3 characters long or are purely numeric. Each candidate word is pre-mapped to its token slot `[t_start, t_end)` in the LLM tokenizer's tokenization of the full CoT.

**Per-variant algorithm.** For each of the $V$ requested variants:

1. Initialize `current_cot = original_cot`.
2. Shuffle the candidate word list in a random order.
3. For each word in the shuffled order:
   a. Retrieve the character span `[char_start, char_end)` for token slot `[t_start, t_end)` in `current_cot` (re-computed via `character_span_for_token_slot` using the LLM tokenizer's `offset_mapping`).
   b. Replace the word with the MLM mask token to form `masked_text`.
   c. Extract a ±250-token context window (in MLM tokens) around the mask position.
   d. Query the MLM for the top-20 fill-mask predictions.
   e. Filter candidates with `filter_mlm_predictions`:
      - Strip tokenizer artifacts (`Ġ`, `Ċ`); require alphanumeric.
      - Reject if lowercase-equal to the original word.
      - Reject if capitalization pattern differs (first letter case, all-caps).
      - Reject if original is a substring of candidate or vice versa.
      - Preserve original leading/trailing whitespace in the replacement.
      - Reject if the candidate CoT (with the replacement spliced in at `[char_start, char_end)`) does not satisfy `candidate_at_same_token_indices`:
        - The LLM-tokenized candidate CoT must have the same total length as `current_cot`.
        - All token ids outside `[t_start, t_end)` must be identical.
   f. Apply the first valid candidate: `current_cot = current_cot[:char_start] + replacement + current_cot[char_end:]`.
4. Append `current_cot` to the variant list and write it to `perturbed_{v}.txt`.

**Alignment guarantee.** By enforcing token identity outside `[t_start, t_end)` after every replacement, and by re-computing character spans from the LLM tokenizer's offset mapping after each prior substitution, the final perturbed file is guaranteed to tokenize to the same total length as the original, with each step's token span `[s_i, e_i)` unchanged. This makes the perturbed files immediately usable by `PerturbedKVPostSourceContext` and the learned interpolation methods without any additional alignment step.
