# SOTA ingestion 2026-06-13: recursive reasoner generator plus verifier-as-reward map

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_recursive_reasoner_verifier_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `GRAM stochastic-latent generator`, arxiv_id_or_url: `2605.19376`, url: `https://arxiv.org/abs/2605.19376`}
  - {name: `TRM thinking reward for RLVR/GRPO`, arxiv_id_or_url: `2602.08498`, url: `https://arxiv.org/abs/2602.08498`}
  - {name: `Weaver weak-verifier weighted ensemble`, arxiv_id_or_url: `2506.18203`, url: `https://arxiv.org/abs/2506.18203`}
  - principle: Each method/source MUST carry a real arXiv ID or canonical doc URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v384: `gram_as_generator_if_verifier_value_added_and_headroom_present_v384`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner (candidate: GRAM-as-generator IF verifier_value_added).

**Fresh-pass provenance**

Read the 2026-06-13 Post-.382 Planning Sweep in `research-references.md` and
the recursive-reasoner / verifier-as-reward track in `research-studying.md`,
including the Exp 4081, 4102, 4111, 4121, and 4130 ingestions. Ran the reliable
helpers, not `/deep-research`:

- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "Generative Recursive Reasoning Models GRAM Sudoku Extreme stochastic latent" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "Thinking Reward Model TRM GRPO RLVR verified-correct traces" --limit 8`

The arXiv cluster helper emitted the broadened verifier, EBM, and
active-inference query URLs. Semantic Scholar returned HTTP 429 for both
focused queries, so it did not displace the primary-paper anchors. Low-volume
WebSearch/WebFetch verified `arXiv:2605.19376`, `arXiv:2602.08498`, and
`arXiv:2506.18203`.

## Current .383 recursive-reasoner plus verifier anchor

Exp 4139 is the current graft receipt: `verifier_value_added=false`,
`headroom_present=false`, `graft_deferred=true`, and the honest verdict is
`complete: uninformative_no_headroom_false_negative_risk`. That means the next
planner should not treat the no-lift result as evidence against verifier
reward. It should also not jump to GRAM as a headline replacement unless the
next run first creates measurable oracle headroom and then shows that the
non-oracle verifier or the RFT label contrast captures some of it.

## GRAM stochastic-latent generator

**Method/source:** GRAM, `arXiv:2605.19376`
(https://arxiv.org/abs/2605.19376), turns deterministic recursive refinement
into probabilistic multi-trajectory latent computation. The paper reports
97.0% Sudoku-Extreme test accuracy with a 10M-parameter model, above TRM's
87.4% and HRM's 55.0% in the same table.

**Implementation over nano-trm + Carnot-verifier stack:** Treat GRAM as the
strongest .384 generator candidate only if the verifier side earns the right
to consume a stronger candidate distribution. The concrete graft is:
GRAM samples multiple latent trajectories per Sudoku puzzle, the executable
oracle is used only to measure best-of-K headroom, and the Carnot non-oracle
energy/text-stat ensemble tries to recover that headroom without seeing the
exact-validity label.

**Pitfalls / where it fails:** GRAM can reduce or erase the reranker headroom
that Carnot needs to measure verifier value. If `verifier_value_added` remains
false or `headroom_present` remains false, a GRAM run is a generator benchmark,
not a verifier-as-reward result.

## TRM thinking reward for RLVR/GRPO

**Method/source:** Characterizing, Evaluating, and Optimizing Complex
Reasoning, `arXiv:2602.08498` (https://arxiv.org/abs/2602.08498), trains a
Thinking Reward Model from verified-correct reasoning traces and integrates it
as an auxiliary thinking reward inside RLVR/GRPO. The key precedent for Carnot
is that reasoning-quality shaping is isolated from answer correctness by
filtering to verified-correct traces first.

**Implementation over nano-trm + Carnot-verifier stack:** This supports the
.383 RFT de-confound. Arm A should use verifier-certified labels, arm B should
use vote-certified labels, and both arms should share the same baseline
checkpoint, candidate pool, optimizer budget, and scheduler receipts. The
measured claim is not "training improved"; it is whether the verifier label
source adds held-out value beyond a vote label source under the same adaptation
compute.

**Pitfalls / where it fails:** If the pipeline mixes correctness filtering,
candidate diversity, and label-source effects, the result cannot isolate
verifier reward. A positive delta would be uninterpretable if arm B lacks the
same adaptation budget, and a null is uninformative if the candidate pool has
no best-of-K headroom.

## Weaver weak-verifier weighted ensemble

**Method/source:** Weaver, `arXiv:2506.18203`
(https://arxiv.org/abs/2506.18203), combines multiple weak verifiers with
weak-supervision-derived weights. Its repeated-sampling setting is directly
aligned with Carnot's candidate-pool rerank question: generated candidates are
scored, normalized, and selected by a combined verifier score rather than by a
single weak verifier or unweighted vote.

**Implementation over nano-trm + Carnot-verifier stack:** Use Weaver as the
peer baseline for the .383 non-oracle ensemble-rerank headline. The executable
Sudoku checker remains an oracle upper bound, not the transferable result.
The transferable result is whether weighted continuous Sudoku energy plus
text-stat/verifier features beats fixed vote when oracle(best-of-K) proves
there is selectable headroom.

**Pitfalls / where it fails:** Weak-verifier weighting assumes enough diversity
that errors are not fully correlated. If all weak features track the same
near-valid dead ends, weighting can amplify the wrong candidate. The mandatory
positive control is still oracle(best-of-K) versus vote before interpreting a
non-oracle null.

## Flagged for the .384 roadmap

`gram_as_generator_if_verifier_value_added_and_headroom_present_v384` is the
strongest .384 candidate. GRAM is the best next generator because its
stochastic-latent trajectories naturally produce a distribution for Carnot to
rerank, but it should be scheduled only behind a positive headroom/value gate,
not as an unconditional rerank claim.

