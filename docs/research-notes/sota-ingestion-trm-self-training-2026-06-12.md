# SOTA ingestion 2026-06-12: TRM self-training with verifiers

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_trm_self_training_mapped`
- methods_mapped:
  - {name: `V-STaR keep-rejected verifier training`, arxiv_id: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `STaR / ReST generate-filter-improve loop`, arxiv_id: `2203.14465`, url: `https://arxiv.org/abs/2203.14465`}
  - {name: `TTA-TRM full fine-tuning with verifier admission`, arxiv_id: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `Imperfect-verifier forward correction`, arxiv_id: `2510.00915`, url: `https://arxiv.org/abs/2510.00915`}
  - {name: `Verifiable process rewards for recursive steps`, arxiv_id: `2605.10325`, url: `https://arxiv.org/abs/2605.10325`}
- flagged_for_v380: `vstar_rejected_trace_selector_for_trm_rft`

**Fresh-pass provenance**

Read the local verifier-RFT and self-training track in `research-studying.md`
and `research-references.md`, including the `.377` verifier-as-reward ingestion,
the `.378` precision-calibration ingestion, and the TRM recursive-refiner entries
around `arXiv:2511.02886`. Ran the required helpers:

- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_semscholar.py "self training verifier recursive reasoner TRM RFT V-STaR ReST process reward" --limit 8`
- `python3 scripts/sweep_semscholar.py "Tiny Recursive Models test time adaptation verifier reward self training" --limit 8`
- `python3 scripts/sweep_semscholar.py "imperfect verifier noisy verifiable rewards RLVR process reward self training" --limit 8`

Semantic Scholar rate-limited two of the focused queries and returned
`arXiv:2603.02203` plus `arXiv:2602.05570` for the TRM-adaptation query; neither
displaced the operator-specified verifier-RFT anchors. Low-concurrency
WebSearch/WebFetch then verified the primary arXiv pages for `arXiv:2402.06457`,
`arXiv:2203.14465`, `arXiv:2308.08998`, `arXiv:2511.02886`,
`arXiv:2510.00915`, `arXiv:2601.17223`, `arXiv:2605.10325`, and the fresh
adjacent verifier-training paper `arXiv:2605.30290`. The `/deep-research` loop
was not invoked.

## Current .379 TRM verifier-RFT anchor

The `.379` headline is no longer generic "verifier-as-reward." It is
verifier-certified RFT of a recursive reasoner: a `nano-trm`/TRM-style model
generates candidate grid transformations or recursive edit traces, the Carnot
verifier stack certifies or rejects them, and training must improve the recursive
model rather than only rerank a fixed candidate pool.

That makes the load-bearing question narrower than prior SOTA ingestions. The
method must answer: which traces enter full fine-tuning, how rejected traces are
used instead of thrown away, how verifier noise is corrected, and whether dense
per-recursion feedback can be trusted without losing hidden-test correctness.
`arXiv:2605.30290` is important adjacent evidence because it frames verifier
quality as the bottleneck for both test-time refinement and training-time
self-improvement, but the first `.380` candidate should stay closer to the
existing accepted/rejected TRM trace pool.

## V-STaR keep-rejected verifier training

**Method:** V-STaR, `arXiv:2402.06457`
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions rather than discarding failures, then uses
that verifier to select among candidates.

**Implementation over nano-trm + Carnot-verifier stack:** Keep every sampled
TRM trace from the same ARC/Sudoku task pool: verifier-certified pass,
verifier-rejected, hidden-fail, parser-fail, and timeout. Convert pairs from
the same prompt into contrastive selector data: accepted trace should score
above rejected trace when the downstream hidden label confirms the verifier.
Use the selector first as a reranker and then as a corpus-admission gate for a
small full-fine-tune RFT arm. This is the cleanest way to turn Carnot's
rejected evidence into training signal without immediately changing the TRM
generator.

**Pitfalls / where it fails:** V-STaR assumes the accept/reject labels contain
real ranking information. If the Carnot verifier's false-positive channel is
not below the `.378` precision floor, DPO-style contrast training will teach the
selector to prefer verifier artifacts rather than hidden-correct transformations.
It also needs trace diversity; if nano-trm emits near-duplicate wrong traces,
the selector learns surface features rather than semantic repair.

## STaR / ReST generate-filter-improve loop

**Method:** STaR, `arXiv:2203.14465`
(https://arxiv.org/abs/2203.14465), iteratively generates rationales, keeps
those that yield correct answers, fine-tunes, and repeats. ReST,
`arXiv:2308.08998` (https://arxiv.org/abs/2308.08998), gives the offline
generate/filter/improve cadence with reusable batches and stronger filtering.

**Implementation over nano-trm + Carnot-verifier stack:** Treat a TRM recursive
trace as the rationale analogue. Run bounded candidate generation, filter with
the Carnot verifier plus hidden labels where available, train a full-fine-tune
TRM arm on unique certified traces, then regenerate from the updated TRM. Cache
all batches so the next improve step can use a stricter acceptance threshold
without paying for new sampling immediately.

**Pitfalls / where it fails:** STaR/ReST improve support that already exists.
If a correct ARC transform never appears in the candidate pool, verifier
filtering cannot invent it. The method also wastes rejected traces unless
combined with the V-STaR selector, and it can overfit public ARC task variants
if augmentation families are not held out.

## TTA-TRM full fine-tuning

**Method:** Test-time Adaptation of Tiny Recursive Models,
`arXiv:2511.02886` (https://arxiv.org/abs/2511.02886), reports that public-task
pretraining plus bounded full fine-tuning can adapt a 7M TRM, and explicitly
notes that full fine-tuning outperformed LoRA or task-embedding-only adaptation
for that setting.

**Implementation over nano-trm + Carnot-verifier stack:** Use the public
nano-trm training tasks as the pretraining/adaptation split. Keep the
competition-like budget explicit: number of optimizer steps, task count, and
wall-clock. Apply Carnot verifier gates before a trace can enter the
fine-tuning set, and keep a no-RFT full-fine-tune control so any gain is not
misattributed to the verifier when it came from task adaptation alone.

**Pitfalls / where it fails:** This is the substrate method, not a verifier
method by itself. It can "win" by memorizing public task structure or spending
more adaptation compute, and it can erase the planned verifier contribution if
the experiment does not isolate full fine-tune, verifier admission, and
reranking arms.

## Imperfect-verifier correction

**Method:** Reinforcement Learning with Verifiable yet Noisy Rewards under
Imperfect Verifiers, `arXiv:2510.00915`
(https://arxiv.org/abs/2510.00915), models verifier rewards as an asymmetric
false-positive/false-negative channel and adds backward or forward correction
hooks; the forward correction is the lighter-weight candidate because it mainly
needs a false-negative estimate.

**Implementation over nano-trm + Carnot-verifier stack:** Every verifier
certificate should carry `fp_rate`, `fn_rate`, calibration split, confidence
interval, and source verifier. Use those rates to downweight or abstain on
borderline TRM traces before RFT, and reserve a small appeal path where a
stronger checker re-examines rule-based negatives. This belongs before any
policy-gradient RLVR attempt and also informs weighted SFT.

**Pitfalls / where it fails:** The correction only helps if the noise rates
match the current generator distribution. Once full fine-tuning changes the TRM
trace distribution, stale FP/FN rates can become actively misleading. It also
does not solve absent support: cleanly correcting verifier noise cannot train a
trace the generator never produced.

## Verifiable process rewards

**Method:** VPRM, `arXiv:2601.17223`
(https://arxiv.org/abs/2601.17223), and VPR for agentic reasoning,
`arXiv:2605.10325` (https://arxiv.org/abs/2605.10325), replace sparse
outcome-only rewards with deterministic step or turn checks where the task
structure permits objective intermediate verification.

**Implementation over nano-trm + Carnot-verifier stack:** Add per-recursion
telemetry to TRM traces: current grid, proposed edit, latent halt decision,
visible-example consistency, exact-state equivalence, mutation consistency, and
final hidden outcome when available. Start with process-reward-weighted SFT or
reranking; only promote to RLVR if dense rewards predict final hidden
correctness on a held-out calibration split.

**Pitfalls / where it fails:** ARC intermediate states are often
underdetermined. A locally consistent edit can preserve all public examples and
still fail the intended transformation. Dense reward should therefore be a
credit-assignment aid, not a replacement for final hidden-test calibration.

## Flagged for the .380 roadmap

`vstar_rejected_trace_selector_for_trm_rft` is the strongest single `.380`
candidate. It uses evidence the `.379` TRM verifier-RFT run already produces,
turns both successful and failed traces into a selector training set, and stays
compatible with TTA-TRM full fine-tuning, imperfect-verifier correction, and
later process rewards. The first `.380` experiment should build this selector
over the saved nano-trm candidate pool and require a rerank win before allowing
the selector to gate a second full-fine-tune RFT corpus.

