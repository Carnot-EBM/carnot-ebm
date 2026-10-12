# Retrospective typed decisions, 2026-10-11

Exp8432 evaluates frozen Exp8431 heads against released human A1 judgments.
The intended test panel contains 300 answers from 100 questions. All five
arms and five seeds remain visible as 7,500 prediction slots. Seed duplicates
and the three sibling answers do not increase the independent question count.
The artifact and its terminal sidecar record the executed qualification outcome.

The primary conditional Brier contrast is logistic9 minus spline9.
Its measured value is 0.01969866346324697. The paired whole-question 95 percent
bootstrap interval is [0.009633442406249476, 0.030576330671862523]. The registered
bootstrap uses 10,000 draws and seed 7258419 after seed and sibling averaging.
Spline9 minus logistic9 decision loss is -0.0073333333333333315.
These values satisfy the registered exploratory numeric conditions.
Administrative readiness also requires the artifact's recorded publication checks.

| Arm | A1 Brier | A1 log loss | Decision loss | Decision coverage |
| --- | ---: | ---: | ---: | ---: |
| Released raw p2 | 0.133672 | 0.431992 | 0.156000 | 0.920000 |
| Scalar calibration | 0.142896 | 0.483223 | 0.155333 | 0.973333 |
| Nine-input logistic | 0.153127 | 0.601366 | 0.162667 | 0.986667 |
| Nine-input spline | 0.133428 | 0.430984 | 0.155333 | 0.906667 |
| Source-permuted spline | 0.153602 | 0.607334 | 0.162667 | 0.986667 |

The primary baseline is logistic9. Improvement over released raw p2 is much
smaller and has no separately registered positive inference. Source-permuted
predictions differ from spline9 predictions; maximum absolute difference is
0.18407358219547187. This control is descriptive and supplies no independent
factuality credit. Inert controls in analytic tests cannot earn benefit credit.

Several fixed strata have higher spline decision loss. The pooled gain does
not establish a uniform benefit. Every dataset, generator and crossed stratum
has metrics in the artifact, including calibration and conditional contrasts.

| Dataset | Generator | Brier gain versus logistic | Spline minus logistic decision loss |
| --- | --- | ---: | ---: |
| hotpotqa | gemma-2-9b | 0.015607 | 0.011429 |
| hotpotqa | llama3-8b | 0.001009 | 0.017143 |
| hotpotqa | mistral-7b | 0.044286 | -0.028571 |
| triviaqa | gemma-2-9b | 0.009047 | -0.024242 |
| triviaqa | llama3-8b | 0.006812 | -0.018182 |
| triviaqa | mistral-7b | 0.021662 | 0.018182 |
| truthfulqa | gemma-2-9b | 0.018732 | -0.012500 |
| truthfulqa | llama3-8b | 0.026374 | -0.043750 |
| truthfulqa | mistral-7b | 0.034265 | 0.012500 |

Decisions accept when hallucination probability is at most 0.2 and reject
when it is at least 0.8. Other probabilities escalate. Correct decisions cost
zero, wrong decisions cost one, and escalation costs 0.2. No threshold search
runs. Brier and log loss use individual seed predictions; paired contrasts
average their losses. False-accept counts in aggregate raw operands count
prediction slots; per-seed counts keep the answer denominator explicit.
Selective risk names a zero-definitive-decision denominator when undefined.
Calibration uses ten fixed equal-width probability bins.

A1 is the primary human target. A2 sensitivity uses its observed subset.
The artifact retains missing prediction slots and worst-case Brier/decision
bounds. A1 bounds also retain unknown target operands. A2 bounds condition on
observed A2 labels. Complete-case contrasts state their conditional question
denominator and omitted questions. This invocation completed every intended slot.

The runtime seals prediction bytes before its evaluator opens test targets.
Labels were already public and used in historical work. Setup inspected one
evaluator record before this invocation; the fixed predictor reads only public
numeric vectors. The new seal cannot reverse prior exposure. Old overlap
remains unknown. This is an oracle-referenced retrospective result, so an
exploratory positive uses circular_positive. Both generalization scores stay zero.
No current language model loads or generates text; MODEL_SPECS is empty.
GPT-5-mini and generator names are historical provenance only.

Cold replay reconstructs prediction rows and metric reductions using independently
pinned producer bytes. Real children exercise absent replay input, deliberate
errors, rehashed metric changes, changed prediction primitives, changed receipts
and reversed seal order. Private E2E-018 and E2E-021, scoped serial tests,
genuine child coverage, Ruff, strict mypy and spec coverage provide validation
receipts. Unchanged primary publication, adversarial verification and strict
row checks qualify terminal bytes. Findings stay in the record; unsupported
informational findings disqualify readiness while preserving measured science.

Required review found that log hashes alone did not authenticate receipt status.
A failing regression reproduced forged exit metadata. Replay now compares the
independent child record and derives pass status from exits, timeout and scanner
findings. The interrupted prepublication attempt remains under raw evidence.

No result establishes independent factuality, current-Qwen skill or hidden-game
generalization. Exp8432 does not measure continuous feedback learning. A null
or administrative failure here does not block that independent branch.
V717 H1=-0.00390625 and H2=0 remain closed. Historical artifacts and protocols
remain intact. The conductor owns ops and BMAD reconciliation.
