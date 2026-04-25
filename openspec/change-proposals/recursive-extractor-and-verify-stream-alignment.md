# Recursive-Language-Model-augmented constraint extraction + `verify_stream` alignment

**Status:** Draft change proposal.
**Origin:** Zhang, Kraska, Khattab, *Recursive Language Models*, arXiv 2512.24601
  (December 2025).
**Target milestone:** 2026.04.64.
**Priority:** Medium-high. Two specific scaling bottlenecks line up cleanly
  with the RLM pattern; one downstream API decision (`verify_stream`) needs
  to be made *before* it ships, not after.
**Depends on:**
  - `LLMConstraintExtractor` (already in `python/carnot/pipeline/extract.py`).
  - Issue #7 streaming verification API
    (`openspec/change-proposals/issue-007-streaming-verification-api.md`),
    not yet implemented.
  - Issue #2 `budget_ms` (`issue-002-bounded-time-verification-api.md`),
    needed to enforce hard latency caps that exclude the generative-time
    safety gate from this proposal.

## Summary

Recursive Language Models (RLMs) let an LLM treat a long prompt as an
environment, **programmatically examine, decompose, and recursively call
itself** over snippets. The Zhang/Kraska/Khattab paper post-trains
RLM-Qwen3-8B, processes inputs up to **100× beyond model context windows**,
+28.3% average over base Qwen3-8B on long-context benchmarks. The paper is
not about safety, hallucination, or energy-based modelling — it is about
**scale**.

Two places in Carnot are scale-bottlenecked in the same way long-context
benchmarks are:

1. **`LLMConstraintExtractor`** truncates or silently drops claims when a
   reasoning chain exceeds the extracting model's context window
   (multi-step proofs, long agent traces, multi-thousand-token code
   reviews). The downstream verifier (Z3 / Hypothesis / EBM) then has
   nothing to check against for the dropped claims, so a long answer
   silently reduces verification recall.
2. **`verify_stream(candidates: list[dict], ...)`** (issue #7) was drafted
   as a priority-queue async iterator. The decomposition pattern in the
   issue #7 proposal is *implicit* — pulled from the consumer side
   (`top_k`, `early_stop_margin`). RLM makes the decomposition pattern
   *explicit* on the producer side. The two should be reconciled before
   `verify_stream` ships, not after.

Energy-as-ground-truth is preserved throughout — each recursive snippet's
claims still go through the same verifier stack and produce the same
energy score. RLM changes the *scope* over which the verifier operates,
not what "correct" means.

This proposal does **not** touch the generative-time safety gate. RLM
recursion amplifies LLM cost (more calls, deeper trees), which is fine
for offline verification but breaks the hard latency budgets the
generative-time gate requires (see `generative-time-safety-gate.md` and
issue #2 `budget_ms`).

## Proposed experiments

### Exp A — RLM-augmented `LLMConstraintExtractor` for long reasoning chains

**Deliverable:** `python/carnot/pipeline/extract_recursive.py` +
`results/experiment_<N>_recursive_extractor.json` +
unit tests in `tests/python/test_extract_recursive.py`.

**What it does:**

1. Add `extract_recursive(answer: str, max_tokens: int) -> ExtractedClaims`
   to the constraint extractor. When the input fits the model's window,
   delegates to existing `LLMConstraintExtractor.extract`. When it does
   not, the extractor decomposes the answer along reasoning-step boundaries
   (already detectable from the `_step_segmenter` used in CPMI work) and
   recursively extracts from each segment.
2. After per-segment extraction, merge claim graphs by attaching each
   segment's claims to a node labelled with the segment index, then
   re-running the cross-segment dependency edge inference on the
   *merged* graph. Cross-segment edges are inferred from claim-text
   overlap and from explicit "as shown in step N" references.
3. Cap recursion depth at 3 by default. Beyond that, fall back to
   non-recursive extraction on a single concatenation with the highest
   per-segment energy density (heuristic: the segments most likely to
   carry violations) and tag the result `recursion_depth_capped` in the
   honest verdict.

**Acceptance gates:**

1. On a synthetic 32K-token reasoning-chain corpus (mix of valid and
   intentionally-broken multi-step arithmetic), recursive extraction
   recovers ≥ 95% of ground-truth claims that non-recursive truncated
   extraction misses (i.e., the claims that are in the second half of
   the input). Targeted at the simplest decomposition primitive working
   correctly.
2. Per-claim energy scores produced by the verifier on the
   recursively-extracted claim set are within ±5% of the energy scores
   the same verifier would assign if the full claim set had been
   extracted in one shot at unbounded context. Validates that recursion
   doesn't introduce systematic bias.
3. Cost amplification factor is reported per-call: number of LLM calls
   for the recursive path divided by the single-call cost. Median
   amplification ≤ 4× on the test corpus; tails (95th percentile) ≤ 8×.
4. Honest-verdict enum: `recursive_extraction_lossless`,
   `recursive_extraction_lossy_below_gate`,
   `recursion_depth_capped`,
   `cost_amplification_above_gate`,
   `extraction_unchanged_short_input`.

### Exp B — `verify_stream` API audit against RLM primitives

**Deliverable:** an audit document
`openspec/change-proposals/issue-007-rlm-audit.md` (or amend issue #7
in place if the changes are small) +
`results/experiment_<N>_verify_stream_rlm_audit.json`.

**What it does:**

1. Read the issue #7 design (priority queue, `top_k`, `early_stop_margin`,
   `VerdictRecord` enrichment with `rank` and `is_final`).
2. Map the issue #7 primitives onto RLM's decomposition primitives —
   what RLM calls "examine, decompose, recurse" maps to "peek priority
   queue, schedule next chunk, push verdicts into queue" in our async
   iterator. Identify gaps: places where RLM's decomposition decisions
   are explicit on the producer side but where issue #7 leaves them
   implicit.
3. Decide one of three outcomes per identified gap:
   (a) issue #7's design absorbs the RLM primitive verbatim,
   (b) issue #7's design is intentionally simpler and the gap is
       documented as out-of-scope,
   (c) issue #7 needs an extension before shipping.
4. Update `issue-007-streaming-verification-api.md` accordingly.

**Acceptance gates:**

1. Every identified gap has a documented outcome (a/b/c). No
   "to-be-determined" items remain.
2. If outcome (c) is chosen for any gap, a follow-up Exp C is scheduled
   with concrete acceptance criteria. The audit must not silently
   expand `verify_stream`'s scope.
3. Honest-verdict enum: `verify_stream_aligned_with_rlm`,
   `verify_stream_extension_required`, `verify_stream_gaps_documented`.

### Exp C — RLM-decomposed `verify_stream` driver (gated on Exp B)

**Deliverable:** only if Exp B identifies an outcome-(c) gap that
warrants extending `verify_stream`. In that case:
`python/carnot/pipeline/verify_stream_rlm.py` (a thin driver wrapping
the issue #7 implementation) plus targeted unit tests.

**What it does:** wraps `verify_stream` with the explicit RLM
decomposition pattern when the consumer has handed in a candidate
pool whose total token count exceeds an RLM threshold. Inside the
threshold, the existing `verify_stream` runs unchanged.

**Acceptance gates** (only relevant if Exp C is triggered):

1. End-to-end latency on a 500-candidate pool with average 8K-token
   answers is no worse than 1.4× the non-RLM `verify_stream` latency on
   the same pool — recursion overhead is bounded.
2. Top-10 ranking agreement with the non-RLM path is ≥ 95% (Spearman
   rank correlation) on the same pool. Ranking quality must not degrade
   meaningfully under recursion.
3. Honest-verdict enum: `rlm_stream_driver_ships`,
   `rlm_stream_latency_above_gate`, `rlm_stream_ranking_drift_above_gate`.

## Explicitly out of scope

- **Generative-time safety gate.** RLM recursion multiplies LLM calls;
  the safety gate has hard millisecond budgets (issue #2 `budget_ms`).
  The two are incompatible. The safety gate continues to use direct
  inference and per-token logprob features (per the Cognometry
  reference); no recursion path is exposed to it.
- **Hallucination detection** as such. RLM is a scale technique. Our
  hallucination-detection signal still comes from the verifier stack
  (Z3 / Hypothesis / EBM energy) and from probe-level features
  (Cognometry-style logprob trajectories where applicable). RLM lets
  the extractor read more; it does not add a new truthfulness signal.
- **Training our own RLM.** The paper post-trains RLM-Qwen3-8B. If the
  authors release the weights we may consume them; if not, we use the
  RLM *pattern* with our existing models and accept the
  cost-amplification overhead measured in Exp A. We do not propose a
  Carnot-trained RLM in this milestone.

## Risks

- **Cost amplification compounds**. Median 4× on Exp A's test corpus
  is acceptable; if real workloads see 10×+ on extraction alone, the
  recursive path is offline-only and a flag-gated option, not a
  default. Mitigation: the Exp A acceptance gate caps median and
  95th-percentile amplification.
- **Decomposition fidelity is corpus-specific**. The
  reasoning-step segmenter is trained on the CPMI corpus shape. Long
  free-form answers (essays, narrative explanations) may segment
  poorly. Mitigation: Exp A includes a fall-back to fixed-size
  windowing when step-segmentation produces fewer than 2 segments on
  a 32K input.
- **Issue #7 ships before this proposal lands**. If `verify_stream` is
  implemented as drafted before Exp B's audit, we may have to
  retrofit the API. Mitigation: schedule Exp B *before* issue #7's
  Exp A in milestone .64. The audit is cheap (read-and-decide work,
  not implementation).
- **RLM-Qwen3-8B weights may not be public**. If they are not, we
  use prompted recursive decomposition with our existing models.
  This is strictly weaker than a post-trained RLM, but the *pattern*
  is reusable — the paper's value to us is the design, not the
  weights.
