# SOTA ingestion 2026-06-14: oracle-distinct learned-verifier map for .391

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_oracle_distinct_mapped_v391`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `ARBITER conservative override for wrong-majority recovery`, arxiv_id_or_url: `2605.26172`, url: `https://arxiv.org/abs/2605.26172`}
  - {name: `SCOPE fine-grained confidence features`, arxiv_id_or_url: `2512.15146`, url: `https://arxiv.org/abs/2512.15146`}
  - {name: `ThinkPRM generative process verifier`, arxiv_id_or_url: `2504.16828`, url: `https://arxiv.org/abs/2504.16828`}
  - {name: `PRM survey outcome-to-process taxonomy`, arxiv_id_or_url: `2510.08049`, url: `https://arxiv.org/abs/2510.08049`}
  - {name: `V-STaR accepted and rejected boundary`, arxiv_id_or_url: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `Calibrated Reasoning detector and abstention axis`, arxiv_id_or_url: `2509.19681`, url: `https://arxiv.org/abs/2509.19681`}
  - {name: `ExecVerify execution-reward baseline`, arxiv_id_or_url: `2603.11226`, url: `https://arxiv.org/abs/2603.11226`}
  - {name: `EVOM execution-verified optimization modeling`, arxiv_id_or_url: `2604.00442`, url: `https://arxiv.org/abs/2604.00442`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v391: `arbiter_conservative_override_arc_wrong_majority_v391`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner (e.g. ARBITER conservative-override, or a learned-ARC-energy distill).

## Fresh-pass provenance

Read `research-references.md` `.390 planning sweep`, `research-studying.md`,
`results/experiment_4210_oracle_distinct_arc_verifier_beats_vote.json`, and
`results/experiment_4208_verifier_as_detector_auroc.json`.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "ARBITER reasoning trajectory basins majority vote failures test time sampling SCOPE fine grained reward signal" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "oracle distinct learned verifier process reward model ThinkPRM V-STaR calibrated reasoning abstention ExecVerify EVOM" --limit 8`

The cluster helper emitted broadened verifier/process-reward and energy-model
arXiv API URLs. Semantic Scholar returned HTTP 429 for both focused queries,
so no S2-only promotion is claimed. Low-concurrency WebSearch/WebFetch verified
arXiv:2605.26172, arXiv:2512.15146, arXiv:2504.16828, arXiv:2402.06457,
arXiv:2509.19681, arXiv:2510.08049, arXiv:2603.11226, and arXiv:2604.00442.

## Exp 4210 A3 status and Exp 4208 detector context

Exp 4210 is not a completed oracle-distinct A3 result: it reports
`blocked_gate_check_failed` because `exp4209-oracle-distinct-arc-verifier-build.selector_trained`
was false. The A3 claim therefore remains open; no vote-beating learned ARC
verifier result should be inferred from the blocked artifact.

Exp 4208 is the complementary detector-axis evidence, not a selector win. It
reports ARC detector AUROC 0.9016 with CI95 [0.7828, 0.9984],
`verifier_is_oracle=false`, ARC selector headroom 0.129, and ARC base rate
0.0024. That supports abstention/detection value, but it does not close the
wrong-majority selection gate.

## SOTA -> experiment mapping

## Wrong-majority recovery

**Method/source:** ARBITER, arXiv:2605.26172
(https://arxiv.org/abs/2605.26172), names the exact headroom: majority vote
selects the largest reasoning basin, not necessarily the most accurate one, so
correct answers can exist in the pool and lose. SCOPE, arXiv:2512.15146
(https://arxiv.org/abs/2512.15146), replaces flat vote supervision with
step-wise confidence and dynamic subgroup partitioning.

**Carnot stack mapping:** Strengthen A2/A3 with a conservative override: keep
vote as the prior, override only when the learned ARC verifier has high learned margin,
and feed it per-region confidence and subgroup evidence rather than a single
global vote count.

**Implication:** The .391 test should target wrong-majority strata first:
oracle@K > vote@1, correct answer present, and candidate basins separable by
learned features.

**Failure mode:** ARBITER and SCOPE are precedents for recovering vote-discarded
answers; they do not prove Carnot has a trained ARC verifier. If A2 remains
untrained or candidate support is too sparse, A3 stays blocked or null.

**Experiment mapping:** Flag `arbiter_conservative_override_arc_wrong_majority_v391`.
Build the wrong-majority slice, train the V-STaR-style ARC boundary, and report
override@1 - vote@1 with bootstrap CI and a matched no-verifier control.

## Learned process verifier recipe

**Method/source:** ThinkPRM, arXiv:2504.16828
(https://arxiv.org/abs/2504.16828), is the high-quality generative process
verifier recipe. A Survey of Process Reward Models, arXiv:2510.08049
(https://arxiv.org/abs/2510.08049), supplies the taxonomy separating data,
modeling, selection, abstention, and reward usage.

**Carnot stack mapping:** Use ThinkPRM as the expensive teacher or quality
ceiling for difficult ARC process labels, then distill into cheap ARC energy.
Use the survey taxonomy to keep selector, detector, and reward claims separate.

**Implication:** A learned-ARC-energy distill is a plausible .391 fork after the
conservative override is specified, especially for region-level violations that
flat vote cannot score.

**Failure mode:** Process verifiers cost tokens, need labels or synthetic
verification traces, and can produce locally plausible explanations that do not
preserve global ARC transformations.

**Experiment mapping:** On the hard wrong-majority subset, compare cheap ARC
energy alone, ThinkPRM-style region labels, and distilled learned ARC energy.

## Accepted and rejected boundary

**Method/source:** V-STaR, arXiv:2402.06457
(https://arxiv.org/abs/2402.06457), trains with both accepted and rejected
solutions instead of discarding failures.

**Carnot stack mapping:** This is the in-repo verifier class for A2: accepted
and rejected ARC candidates should define the correctness boundary before A3 is
allowed to rerank.

**Implication:** Exp 4210's blocked gate is the right guardrail. The workflow
must first produce `selector_trained=true` and off-fold ARC AUROC before the
headline vote-beating gate can run.

**Failure mode:** A V-STaR-style boundary fails honestly if the pool has no
positives, weak wrong-majority support, or features that do not separate ARC
transformations.

**Experiment mapping:** Rebuild A2 with accepted/rejected ARC candidate rows,
wrong-majority support counts, and off-fold AUROC; only then rerun A3.

## Detector and abstention axis

**Method/source:** Calibrated Reasoning, arXiv:2509.19681
(https://arxiv.org/abs/2509.19681), trains an explanatory verifier with
calibrated confidence for efficient test-time reasoning and difficult failure
detection.

**Carnot stack mapping:** This maps to Exp 4208: detector AUROC and
accuracy-vs-coverage are valuable, but they are not the same claim as selecting
the outvoted correct ARC candidate.

**Implication:** .391 should report abstention curves beside selection deltas
and pre-register coverage targets for any deployment claim.

**Failure mode:** A calibrated detector can reject ambiguous cases and still be
unable to choose the correct candidate among wrong-majority basins.

**Experiment mapping:** Add a detector-gated conservative override row: apply
ARBITER/SCOPE override only inside coverage bands where calibration is
pre-registered.

## Execution reward baselines

**Method/source:** ExecVerify, arXiv:2603.11226
(https://arxiv.org/abs/2603.11226), turns execution traces into verifiable
stepwise rewards for code execution reasoning. EVOM, arXiv:2604.00442
(https://arxiv.org/abs/2604.00442), treats a solver backend as the deterministic
verifier in a generate-execute-feedback-update loop.

**Carnot stack mapping:** These frame B1 verifier-as-reward as execution-grounded
RL. They are valid baselines for code and solver-backed optimization, but the
verifier is the executable oracle.

**Implication:** A B1 positive must be reported against execution-reward
baselines with random-label controls, while the .391 oracle-distinct ARC claim
remains separate.

**Failure mode:** Execution-reward wins can be strong and still circular for a
moat claim. They cannot substitute for the non-executable ARC learned-verifier
gate.

**Experiment mapping:** Keep B1 in the execution-verified RL family and use
ExecVerify/EVOM as baselines; do not let B1 flip the oracle-distinct A3 gate.

## Flagged for .391

`arbiter_conservative_override_arc_wrong_majority_v391` is the strongest method
for the next planner. The reason is specific: ARBITER names the headroom Exp
4210 could not test, and SCOPE supplies the feature direction. Before attempting
a broader learned-ARC-energy distill, run the ARBITER conservative override over
ARC wrong-majority cases first.

