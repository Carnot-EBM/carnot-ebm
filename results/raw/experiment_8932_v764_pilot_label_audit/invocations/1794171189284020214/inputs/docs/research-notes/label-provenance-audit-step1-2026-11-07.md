# Label-provenance audit, step 1 of the consensus plan (2026-11-07)

Status: CPU-only audit. Read-only. No experiment artifact was written. This is a measurement of existing data, not a Carnot result.

Plan: `jepa-two-model-consensus-plan-2026-11-05.md`, step 1. Gate: pass or fail on "are the error labels independent of the detector?"

## Verdict

FAIL. Zero independently labeled errors are confirmed. The 114 error traces of exp4381 cannot feed the calibration pilot (step 4, needs at least 200).

## Method

1. Load `data/step_level_prm_training.jsonl` with `load_step_labeled_traces`. This is the file exp4381 used. It is not `data/fover_corpus.jsonl`.
2. Count error traces. Join each wrong question id to `data/fover_corpus.jsonl` by `question_id`.
3. Check how the step labels relate to the final-answer flag.

## Findings

| Item | Result |
|---|---|
| Traces / error traces | 6,548 / 114 (matches exp4381) |
| Rows in `step_level_prm_training.jsonl` | 7,349 (6,881 correct, 468 wrong) |
| `step_label == "wrong"` exactly when `full_cot_correct` is false | 7,349 of 7,349 rows |
| First-error index in the 114 error traces | 0 in all 114 |
| Every step after the first error is also wrong | 114 of 114 |
| Error traces with one step | 11 |
| exp4381 localization exact-match count | 11 of 114 (F1 0.096) |
| Source of the 114 (by `question_id` join) | math_z3_v3: 76, math_z3: 29, fover_v4: 9 |
| `verifier` field on all 114 joined rows | `heuristic` |

## What this means

- The "step labels" are outcome labels. The generator (exp1084) cuts each parent entry into prefixes. Every prefix inherits the parent label. The prefix `cascade_score` is a heuristic.
- The first error is therefore always step 0. A localizer that predicts step 0 is always right for a one-step trace. That matches the 11 exact matches. The F1 of 0.096 equals 11 of 114. The localization result measures trace length, not localization skill. This is an outer-loop inference from the numbers above. It is not yet re-run.
- All 114 wrong labels trace to the parent corpus. The parent `verifier` value is `heuristic`. For `fover_v4` the earlier audit found this value is a backfilled default. For `math_z3*` the label origin was not confirmed: the `verifier` field says `heuristic`, and the source name suggests Z3. Not resolved.
- So the count of independently labeled errors is 0 confirmed, and at most 114.

## Not checked

- The origin of the `math_z3` and `math_z3_v3` labels (Z3 check or answer heuristic).
- The 100 incorrect plus 100 correct label sample (needs a human or Z3 read).
- Whether `fover_corpus.jsonl` `sota_extension_v5` and `None`-verifier rows differ from the above. Earlier findings stand: 523 `sota_extension_v5` rows use an answer-derived fallback, and about 1,000 `None`-verifier rows carry `coherence_label` and `sc_energy_label`, probably circular.

## Consequences for the plan

1. Step 4 (calibration pilot) is blocked on labels. The ImplicitHop idea (exact breadth-first-search labels on random graphs) is now the only candidate source of independent labels. It still needs a generator run and a GPU slot through the conductor.
2. Any claim built on exp4381 localization F1 needs a corrigendum note. Append it. Do not edit the artifact.
3. Step 2 (cost-monotonicity audit) should include a trace-length control for the same reason.

Rule check: no weight update to the mandated generator. ARC stays suspended. No external publication.

## Addendum 2026-11-07: sibling experiments (append-only)

Status: code and verdict-string read only. Nothing was re-run. No artifact was edited.

Question: do exp4392, exp4393, exp4396, exp4403 and exp4407 share the exp4381 label defect?

Finding: all five read `data/step_level_prm_training.jsonl`. Per the findings above, that file gives first error = 0 on all 114 error traces.

| Experiment | Recorded verdict | Real-split target | Effect |
|---|---|---|---|
| exp4392 | success: synthetic localizer beats ensemble baseline | `trace.first_error_index` from the same loader | Real-split F1 measures trace length. Synthetic-injection part not checked. |
| exp4393 | complete: A1 win quarantined as artifact-confounded | position-only baseline built from counts of `first_error_index` | Baseline predicts 0. The quarantine was for a template confound, not this one. |
| exp4396 | complete: clean saturated null | same loader | Null is uninformative. |
| exp4403 | complete: clean powered null, position-only not beaten | own label builder sets `first_error_index=0` for every incorrect row | Index is 0 by construction. The reference-replacement check tests the same outcome label. Null is uninformative. |
| exp4407 | complete: clean null, position-bound or saturated | `predict(trace) == trace.first_error_index` | Null is uninformative. |

Consequences:

1. Real-split localization results in this family measure trace length, not localization skill.
2. The nulls in exp4396, exp4403 and exp4407 are not evidence that the method fails (FALSE_NEGATIVE_RISK rule).
3. exp4392 and exp4393 synthetic-injection parts use their own injected labels. Not audited here.

Not checked: artifact counts (`n_traces`, `n_errors`) against 114 and 6,548. Whether any of the five reports a nonzero positive control. The loader behaviour on each code path. Re-running any experiment.

Step 6 answer (check or prose): prose and a corrigendum entry for now. A check would be a lint that rejects a localization artifact whose real-split `first_error_index` is constant. Not built. It needs the label-provenance question settled first, or it fires on legitimate work.
