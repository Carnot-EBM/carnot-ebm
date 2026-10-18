# Certified PWA-KAN Verifier Capability

**Capability:** kan-verifier  
**Version:** 0.1.0  
**Status:** Draft  
**Traces to:** Exp6977

## Requirements

### REQ-KAN-6977: Calibration-Only Certified PWA-KAN Residual Energy

Carnot SHALL provide
`python/carnot/experiment_6977_certified_pwa_kan_energy.py`. The command
`.venv/bin/python scripts/experiments/experiment_6977_certified_pwa_kan_energy.py --date 20260904`
SHALL write `results/experiment_6977_certified_pwa_kan_energy.json`.

The experiment SHALL check all preconditions before training. It SHALL require
`candidate_certification_complete_score=1`, nonconstant calibration features,
nonconstant calibration labels, replayable calibration and held-out split
hashes, a supported compact KAN implementation, and an executed mixed-integer
linear programming (MILP) solver smoke test. If any check fails, the experiment
SHALL write `blocked_certified_pwa_kan_energy`. The blocked artifact SHALL name
each failed check and its expected and observed values. It SHALL not train a
model or read held-out labels after a failed calibration gate.

Before training, the experiment SHALL freeze numeric features from parser
state, structural schema counts, normalized declared-domain and objective
diagnostics, bounded output length, schedule ID, formulation family, and model
family. The feature vector SHALL exclude exact witnesses, exact outcomes,
held-out labels, pair IDs, raw hashes, and names or values that directly encode
correctness. Split and opaque candidate metadata SHALL remain outside the
feature vector.

When all preconditions pass, the experiment SHALL train a seeded compact KAN
residual on calibration rows only. It SHALL compare a constant baseline and a
size-matched multilayer perceptron (MLP). Every optimizer epoch SHALL have a
terminal training row. The experiment SHALL save and hash one KAN checkpoint.
No training choice SHALL use a held-out metric.

The experiment SHALL construct sound lower and upper piecewise-affine (PWA)
bounds for each nonlinear KAN unit over finite input bounds. A unit-level
dynamic program SHALL enumerate feasible piece counts. A network-level
knapsack SHALL allocate one fixed global piece budget. The artifact SHALL keep
local unit error and propagated network error in separate rows.

The experiment SHALL use an actual MILP backend. It SHALL prove finite output
bounds, only preregistered monotonicity claims, stability under changes to
irrelevant fields, and preservation of hard infeasibility outside the learned
residual. An optimal solver result SHALL include integer variables, constraints,
an objective, and a witness or bound that is not copied from the claimed
property. Timeout, unknown, infeasible proof models, solver errors, and fallback
enumeration SHALL not produce a certificate. An infeasible candidate SHALL
remain infeasible for every residual value.

The held-out split SHALL be evaluated once after fitting and certification.
The KAN, PWA abstraction, constant baseline, and size-matched MLP rows SHALL
report AUROC, calibration error, pairwise ranking accuracy, latency, parameter
count, and deterministic uncertainty intervals. PWA rows SHALL also report
abstraction disagreement. The experiment SHALL not tune on these results.

`certified_pwa_energy_ready_score` SHALL be a bare integer. It SHALL equal one
only when the real MILP path executed, every claimed bound and invariant is
proved, every held-out row is terminal, and measured abstraction disagreement
is within the certified envelope. A positive held-out accuracy result SHALL not
be required for readiness. `pwa_energy_heldout_positive_score` SHALL be a
separate bare integer.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `random_seed`,
`feature_schema`, `feature_rows`, `split_isolation_rows`, `training_rows`,
`checkpoint_hash`, `model_config`, `model_parameter_count`, `pwa_unit_rows`,
`piece_budget_rows`, `local_error_bound_rows`,
`propagated_error_bound_rows`, `milp_query_rows`, `milp_solver_receipts`,
`invariant_certificate_rows`, `rows`, `per_candidate_rows`,
`heldout_comparison_rows`, `abstraction_disagreement_rows`, `latency_rows`,
`certified_pwa_energy_ready_score`, `pwa_energy_heldout_positive_score`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL contain one
scientific principle for each required field and the readiness score.

`inference_substrate` SHALL equal
`calibration_only_kan_training_plus_pwa_milp_certification`.
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL be one of
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`. `honest_verdict` SHALL use a terminal prefix consistent with the
class. A blocked verdict SHALL start with `blocked_`.

#### SCENARIO-KAN-6977-PRECONDITIONS: Constant Calibration Labels Block Training

Given a complete exact-candidate artifact whose calibration labels have one
unique value,
When Exp6977 checks its inputs,
Then the artifact records the failed label-variation gate and no training row
or checkpoint hash.

#### SCENARIO-KAN-6977-FEATURES: Features Exclude Exact And Identity Fields

Given candidate metadata, raw generated text, and exact certification rows,
When Exp6977 freezes features,
Then feature vectors contain only the registered schema and contain no exact
witness, label, pair ID, raw hash, correctness, or exact-outcome field.

#### SCENARIO-KAN-6977-TRAINING: Seeded Calibration Training Is Deterministic

Given the same nonconstant calibration tensor, labels, seed, and model config,
When two compact KAN fits run,
Then all epoch losses, model parameters, predictions, and checkpoint payload
hashes are equal. Held-out rows do not enter either fit.

#### SCENARIO-KAN-6977-PWA: Local Envelopes Bound Every Unit

Given a bounded nonlinear KAN unit and finite interval,
When Exp6977 builds lower and upper PWA abstractions,
Then dense independent probe values lie between both envelopes and the local
error row bounds the largest observed gap.

#### SCENARIO-KAN-6977-BUDGET: Dynamic Programming And Knapsack Respect Budget

Given unit error tables and one global piece budget,
When allocation runs,
Then each unit receives its minimum piece count, the total does not exceed the
budget, and the selected allocation minimizes propagated error with stable
tie-breaking.

#### SCENARIO-KAN-6977-MILP: A Non-Tautological Solver Path Executes

Given finite feature bounds and an encoded PWA network,
When the MILP backend maximizes and minimizes network output,
Then it reports an optimal mixed-integer solve, nonzero variable and constraint
counts, objective values, solver witnesses, and independently checked bounds.

#### SCENARIO-KAN-6977-INFEASIBILITY: Residuals Cannot Override Hard Failure

Given a candidate rejected by parsing, schema, or exact feasibility,
When any finite KAN or PWA residual score is attached,
Then the combined decision remains infeasible and the invariant certificate is
true.

#### SCENARIO-KAN-6977-BARE: Readiness Scores Remain Bare Integers

Given any blocked or completed artifact,
When schema validation runs,
Then both readiness fields are bare zero-or-one integers and not wrapped
objects or booleans.

## Implementation Status

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-KAN-6977 and SCENARIO-KAN-6977-* | Implemented (`python/carnot/experiment_6977_certified_pwa_kan_energy.py`; `scripts/experiments/experiment_6977_certified_pwa_kan_energy.py`) | Implemented (`tests/python/test_experiment_6977_certified_pwa_kan_energy.py`; feature isolation, deterministic training, PWA soundness, piece allocation, real MILP execution, infeasibility preservation, blocked preflight, bare fields, and 100% new-module statement coverage) |
