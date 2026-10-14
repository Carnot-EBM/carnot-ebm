# V732 strongest-baseline audit

Exp8513 reproduces the five V730 losses from original sealed predictions and
labels. It reports each test question's full-minus-raw Brier and typed decision
loss for A1 and observed A2. These are descriptive results on exposed history.
The nearly tied raw comparison cannot select another threshold or establish
independent generalization. All current model invocation counts are zero.

The frozen comparator manifest contains raw_p2, legacy_full,
identity_brier_scalar, identity_brier_full and identity_brier_permuted. It retains
900 answers, 300 questions and the original 150/50/100 fit/tune/test roles. The
original class support, knots, scaler, scalar basis and donor identities stay
bound to the V730 receipts. Every answer/arm/seed has a slot. Exp8514 is the
future producer of the Brier heads; no fit result is inferred here.

The existing scalar_logit already adds the raw-logit offset. The changed
mechanism registered for Exp8514 is Brier fitting at temperature1, followed by
tune-only shrinkage over alpha0/.25/.5/.75/1. Alpha0 preserves the raw forecast.
The registered full-batch fit remains200 steps with learning rate.01 and L2.001.
Fit/tune diagnostics at steps0/25/50/100/200 cannot select a checkpoint or read
test targets. An optimizer-step curve measures optimization, not sample efficiency.

## Inspected methods

| Primary source | Inspected scope | Adopt | Defer |
|---|---|---|---|
| [RAIM2609.39229](https://arxiv.org/html/2609.39229v1), [author analysis](https://github.com/eOnofri04/raim-analysis) | Comparator ladder and paired differences; README, weighting.py and bootstrap.py | Strongest-component controls and whole-question paired comparisons | Logistic cross-fitting, admissibility thresholds, ten-judge runs and replication |
| [KAN2610.10627](https://arxiv.org/html/2610.10627v1), [author repository](https://github.com/DerKevinRiehl/neurips26_kan_training) | Feynman/Gymnasium sample protocols; README and Feynman protocol source | Fixed fit/tune optimization prefixes | KAN/MLP sample-efficiency and RL replication; author default enables matched MLP only |
| [Calibeating2603.22167](https://arxiv.org/html/2603.22167v1) | Proper-loss online reductions and calibration conditions | Proper-loss comparisons and prediction before delayed feedback | Regret guarantees for Carnot's sparse nonlinear updates |
| [Online KAN2602.02056v3](https://arxiv.org/html/2602.02056v3) | Spline locality, update cost and fixed-point FPGA scope | Touched-coefficient and complete update cost accounting | Bitstream support, physical timing and FPGA replication |

Retrieval used concurrency one,20-second request timeouts and4-MiB response
limits. Raw source bytes, URLs, retrieval times and hashes are below
`results/raw/experiment_8513_v732_strongest_baseline_audit/literature/`.
The method mapping names the exact inspected scope and decisions. No external
repository code was executed or installed.

Public availability cannot establish who accessed questions, outputs or labels.
Fresh evidence requires independently adjudicated disjoint questions, generators
and heads frozen before access, and resolved exposure. This task neither re-splits
the exposed panel nor treats RAIM judge outputs as factual truth. V730 objective
publication and append timing remain unqualified; readiness here concerns only
comparator inputs. Optional CUDA and boards do not gate this CPU branch.

The production runner collects historical node IDs before disjoint serial shards.
Private E2E-018/021 retain original historical dates and isolated roots. It checks
100 percent newly owned statement coverage, including real children and rejection
paths, then cold replays, scans and publishes only validated terminal bytes.
Publication sidecars bind their primary hash, and the unchanged downstream
consumer checks the comparator readiness field. Ops and BMAD reconciliation are
owned by the conductor after this task exits.

## Reproduced descriptive A1 results

| Original arm | Brier | Fixed typed loss |
|---|---:|---:|
| raw_p2 | 0.133672 | 0.156 |
| scalar_logistic | 0.14289619348971933 | 0.15533333333333332 |
| scalar_cubic_spline | 0.1529864193582933 | 0.16266666666666665 |
| full_spline9 | 0.13342838109062669 | 0.15533333333333332 |
| source_permuted_spline9 | 0.15360248626043985 | 0.16266666666666665 |

Full-minus-raw mean Brier is about-0.000243619; fixed typed loss differs by
about-0.000666667. This small descriptive difference motivates retaining raw
as a comparator. It cannot justify a new threshold or independent factual gain.
The artifact retains all100 question differences, A2 missingness and the full
original metric dictionaries. Five deterministic repeats do not increase the
100 test-question independence count.
