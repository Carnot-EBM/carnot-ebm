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
