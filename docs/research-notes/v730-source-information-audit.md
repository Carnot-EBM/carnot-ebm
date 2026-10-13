# Source information audit — 2026-10-13

The fixed-panel comparison supports retrospective source information beyond the
registered q-only spline and source-permutation controls. This is exposed-label
evidence. Both generalization scores remain zero. No generator weights change.
No model loads, current LLM calls, CUDA calls or board calls occur.

The canonical result is `results/experiment_8488_v730_source_information_audit.json`.
Its raw invocation stores sealed predictions, evaluator targets, immutable input
hashes, exact authority snapshots, validation commands, logs and coverage shards.
The terminal sidecar binds validator receipts to the exact primary bytes.

The audit authenticates Exp8487 at
`sha256:094f758e2046efec7a93d22fcc297a56ac2af388a574fef78a638b15e61dcb4d`.
It reads inherited producers under sealed original V726 authority. Current V730
full-task authority governs this invocation. V729 cascade-skipped this comparison;
that absence remains unmeasured and supplies no evidence against the hypothesis.

All 7500 answer/arm/seed probabilities are sealed before this evaluator opens
A1/A2. The panel contains 100 questions, three sibling answers, five arms and
seeds17/29/43/71/101. Siblings and seeds average within each question before the
10000 paired whole-question draws at seed7278446. Two-sided97.5 percent intervals
apply Bonferroni adjustment to the two registered primary contrasts.

| Primary contrast | Brier gain | Adjusted interval | Full minus control typed loss |
|---|---:|---|---:|
| q-only cubic spline minus full spline | 0.0195580383 | [0.0083885004, 0.0318702655] | -0.0073333333 |
| source-permuted spline minus full spline | 0.0201741052 | [0.0087494673, 0.0326772612] | -0.0073333333 |

Both lower bounds exceed zero. Typed loss uses wrong=1, correct=0, escalate=0.2,
with accept p<=0.2 and reject p>=0.8. These are conditional comparisons on the
fixed exposed panel. A qualified positive is at most `circular_positive`.
The legacy `brier_gain_ci95` schema alias carries the registered97.5 percent
q-only interval here; `familywise_intervals` supplies the explicit primary fields.

| Descriptive arm | A1 Brier | Log loss | Typed loss | Coverage |
|---|---:|---:|---:|---:|
| raw p2 | 0.1336720000 | 0.4319920911 | 0.1560000000 | 0.9200000000 |
| scalar logistic | 0.1428961935 | 0.4832234903 | 0.1553333333 | 0.9733333333 |
| q-only cubic spline | 0.1529864194 | 0.6015379375 | 0.1626666667 | 0.9866666667 |
| full spline | 0.1334283811 | 0.4309839600 | 0.1553333333 | 0.9066666667 |
| source-permuted spline | 0.1536024863 | 0.6073336849 | 0.1626666667 | 0.9866666667 |

The full head barely improves raw p2 Brier. At matched coverage0.9066666667,
full and q-only risk both equal0.1507352941. This limits the practical interpretation
of the positive registered contrasts. Ranking uses confidence only. Fractional
boundary ties prevent metadata or evaluator labels from selecting favorable rows.

Every arm has one distinct probability vector across five deterministic fits:
duplicate rate0.8 and pairwise decision-flip rate0. Per-seed losses remain explicit.
Repeats do not add independent questions. Reorder and metadata-rename tests compare
exact numerical probabilities after inverse joins. This checks frozen fit outputs;
the audit does not perform new fits or stochastic stability experiments.

A1 covers all300 answers. A2 sensitivity has observed labels in31 question groups.
Missing sensitivity labels remain absent; they are never replaced with zero.
The fixed test strata have35 HotpotQA,33 TriviaQA and32 TruthfulQA questions.
Each question has Gemma-2-9b, Llama3-8b and Mistral-7b answers. Crossed strata retain
all cells. The reducer also emits empty cells and explicit missing denominators
when a cell or operand is unavailable. Seed-expanded false-accept counts are
reported alongside per-seed counts and their original denominators.

The original V726 full-spline versus nine-input-logistic gain remains
0.01969866346324697, CI95[0.009633442406249476,0.030576330671862523], with typed-loss
delta-0.0073333333333333315. It is a different historical contrast. V728 qualified
journal equivalence and its null complete-service speedup1.0447833367,
CI95[0.9942346768,1.1095895448], remain unchanged. V717 H1=-0.00390625/H2=0 stay closed.

[Decomposing Probabilistic Scores: Reliability, Information Loss and Uncertainty](https://arxiv.org/abs/2603.15232)
motivates separating scalar calibration from information retained by source features.
The finite panel does not estimate population grouping loss. No theorem transfers
from the paper to this evaluation. Test labels never fit a calibration curve.

[SIFT: Robust Meta-Faithfulness Verification of Chain-of-Thought Reasoning Under Distribution Shift](https://arxiv.org/abs/2610.04594)
motivates explicit seed, invariance and matched-coverage checks. This audit does
not implement SIFT or claim its guarantees.

Qualification requires scoped tests, genuine100 percent newly owned statement
coverage including private CLI children, unchanged E2E-018/021 consumers, Ruff,
strict mypy, spec coverage, fresh-process replay, missing/error controls, rehashed
primitive tamper rejection and unchanged adversarial/strict row consumers.
Unresolved findings or failed owned checks disqualify readiness. Missing external
inputs block readiness. A scientific null would retire the unchanged claim.
Independent reopening requires externally collected, previously unexposed
questions and independently adjudicated new labels, with frozen heads before
label access. Another split of this panel cannot establish that condition.

Ops and traceability reconciliation is reserved for the conductor, as requested.
