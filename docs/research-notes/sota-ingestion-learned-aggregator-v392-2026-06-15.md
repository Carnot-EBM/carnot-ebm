# SOTA ingestion 2026-06-15: learned aggregator map for .392

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_learned_aggregator_mapped_v392`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `AggLM review-and-reconcile solution aggregation`, arxiv_id_or_url: `2509.06870`, url: `https://arxiv.org/abs/2509.06870`}
  - {name: `AgentAuditor localized-evidence reasoning-tree audit`, arxiv_id_or_url: `2602.09341`, url: `https://arxiv.org/abs/2602.09341`}
  - {name: `GenSelect-BoN RL-trained generative selection`, arxiv_id_or_url: `2602.02143`, url: `https://arxiv.org/abs/2602.02143`}
  - {name: `MSV cross-candidate multi-sequence verification`, arxiv_id_or_url: `2603.03417`, url: `https://arxiv.org/abs/2603.03417`}
  - {name: `SR-TTRL and online CoT-verifier self-learning loop`, arxiv_id_or_url: `2603.03538`, url: `https://arxiv.org/abs/2603.03538`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v392: `agglm_style_arc_review_reconcile_aggregator_v392`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner (e.g. AggLM-style aggregator for ARC, or an AgentAuditor localized-evidence verifier).

## Fresh-pass provenance

Read `research-references.md` `.391 planning sweep`, `research-studying.md`,
`results/experiment_4220_oracle_distinct_arc_verifier_build_labeled.json`, and
`results/experiment_4221_oracle_distinct_arc_verifier_beats_vote.json`.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "AggLM solution aggregation AgentAuditor majority vote LLM judge GenSelect Best-of-N multi-sequence verifier" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "self reflective test time reinforcement learning chain of thought verifier online learnability" --limit 8`

The cluster helper emitted the broadened verifier/process-reward and
energy/verifier arXiv API URLs. Semantic Scholar returned HTTP 429 for both
focused queries, so no S2-only promotion is claimed. Low-concurrency
WebSearch/WebFetch verified arXiv:2509.06870, arXiv:2602.09341,
arXiv:2602.02143, arXiv:2603.03417, and arXiv:2603.03538. The SR-TTRL ICML
listing was also checked as the self-reflective pseudo-labeling companion to
the CoT-verifier theory anchor.

## Exp 4220 A2 status and Exp 4221 A3 status

Exp 4220 did train an oracle-distinct ARC verifier:
`selector_trained=true`, `oracle_distinct_auroc=0.778980279`, CI95
`[0.6146676853, 0.9174508427]`, `verifier_is_oracle=false`, and
`wrong_majority_n=5`. The sparse-positive warning remains load-bearing:
only 14 positive candidates were available out of 1796 in the stratified set.

Exp 4221 then ran the A3 gate and found headroom without a vote-beating
selector: `oracle_at_k=1.0`, `oracle_minus_vote=0.3571428571`,
`verifier_minus_vote_delta=-0.0714285714`, CI95 `[-0.2142857143, 0.0]`, and
`oracle_distinct_beats_vote=false`. This is a complete ingestion target, not a
green selector result: the next method must recover wrong-majority answers
that flat reranking missed.

## SOTA -> experiment mapping

## Review-and-reconcile aggregation

**Method/source:** AggLM, arXiv:2509.06870
(https://arxiv.org/abs/2509.06870), trains aggregation as an explicit
reasoning skill: review candidates, reconcile disagreements, and synthesize a
final answer. AgentAuditor, arXiv:2602.09341
(https://arxiv.org/abs/2602.09341), audits localized branch evidence and beats
both majority vote and LLM-as-judge.

**Carnot stack mapping:** Strengthen the A2 ARC verifier into an aggregator.
The aggregator should see the whole candidate set, vote prior, verifier scores,
localized grid disagreements, and any partial-correctness evidence. It should
review, reconcile, and synthesize or select, not merely rerank candidates by an
independent logistic score.

**Implication:** The .392 headline should target the wrong-majority slice from
Exp 4220/4221: correct answer present, vote wrong or insufficient, and flat
verifier not enough. AggLM is the closest precedent because it explicitly
recovers minority-correct answers; AgentAuditor supplies the localized evidence
and LLM-judge efficiency frame.

**Failure mode:** These sources do not prove ARC aggregation. AggLM is not an
ARC grid system, and AgentAuditor assumes reasoning-tree evidence that cached
ARC rows may not contain. Without region-level evidence, aggregation can become
an expensive judge or an overfit selector.

**Experiment mapping:** Flag
`agglm_style_arc_review_reconcile_aggregator_v392`. Build an ARC aggregator
that reviews all candidates and either synthesizes a final grid or chooses a
candidate after localized reconciliation. Compare aggregator@1 against vote@1,
flat verifier@1, conservative override, and LLM-as-judge on wrong-majority
tasks with bootstrap CI and matched cost.

## RL-trained generative selection

**Method/source:** GenSelect-BoN, arXiv:2602.02143
(https://arxiv.org/abs/2602.02143), trains small models with DAPO on generated
Best-of-N selection tasks and reports selection gains over prompting and
majority-voting baselines.

**Carnot stack mapping:** Convert Exp 4220 rows into ARC Best-of-N selection
episodes with verified correct and incorrect candidates, then train a
generative selector as the selection-only baseline against the aggregator.

**Implication:** A learned selector can still be useful, but it should be
treated as the selection recipe, not the full reconciliation recipe. It answers
whether RL selection improves on the logistic verifier before synthesis is
introduced.

**Failure mode:** GenSelect cannot recover an answer that is only partially
present across multiple candidates, and it can still overvalue popular
wrong-majority clusters if the reward design does not isolate minority-correct
cases.

**Experiment mapping:** Build ARC Best-of-N selection episodes, train a DAPO-like
selector on correct/incorrect candidates, and compare it with the AggLM-style
aggregator and the Exp 4221 flat verifier.

## Cross-candidate verification

**Method/source:** MSV, arXiv:2603.03417
(https://arxiv.org/abs/2603.03417), jointly processes multiple candidate
solutions and models interactions across them instead of scoring each candidate
in isolation.

**Carnot stack mapping:** Add candidate-set context to the ARC verifier:
vote_weight, self_consistency_margin, verifier margins, basin features,
localized disagreement summaries, and cross-candidate calibration should be
available in one scoring pass.

**Implication:** The Exp 4220 feature list already points this way with
cross-candidate self-consistency and per-cell confidence. .392 should make that
explicit and ablate isolated scoring against whole-set scoring.

**Failure mode:** Calibration over a candidate set can still track frequency
rather than truth. MSV-style context must be paired with oracle-distinct local
evidence, or it can preserve the wrong majority more confidently.

**Experiment mapping:** Add an MSV-style candidate-set encoder or explicit
whole-set summaries to A2, then report isolated verifier, cross-candidate
verifier, and review/reconcile aggregator on the same wrong-majority tasks.

## Self-learning verifier-as-reward loop

**Method/source:** SR-TTRL, checked through the ICML 2026 listing, frames
self-reflective verification as a way to create higher-fidelity pseudo-labels
than majority vote. Online Learnability of Chain-of-Thought Verifiers,
arXiv:2603.03538 (https://arxiv.org/abs/2603.03538), provides the
soundness/completeness lens for verifier feedback loops.

**Carnot stack mapping:** This is Phase B, not the .392 headline: use a
positive aggregator/verifier as reward only after it has shown external
selection value, and track false-accept versus false-reject errors explicitly.

**Implication:** If the AggLM-style aggregator clears the wrong-majority gate,
Carnot can try self-learning with verifier pseudo-labels. If it does not, vote
pseudo-labels and verifier labels are both unsafe training targets.

**Failure mode:** Self-training can amplify verifier mistakes. Soundness errors
are worse than ordinary selection misses because they poison the generator or
aggregator that will later create more traces.

**Experiment mapping:** Gate a verifier-as-reward self-learning run behind a
positive aggregator result, then compare verifier pseudo-labels, majority
pseudo-labels, and random-label controls under explicit soundness/completeness
accounting.

## Flagged for .392

`agglm_style_arc_review_reconcile_aggregator_v392` is the strongest single
method for the next planner. The reason is direct: Exp 4221 already showed
headroom but no flat selector lift, and AggLM is the closest verified precedent
for recovering minority-correct answers by review and reconciliation. Use
AgentAuditor as the localized-evidence and LLM-judge efficiency comparator, but
run the AggLM-style ARC aggregator before another flat rerank.

