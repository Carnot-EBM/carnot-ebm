"""Tests for REQ-CL-7405 and SCENARIO-CL-7405-*.

The audit must finish the synthetic review even when the independent live
cohort is blocked. It must never pool those cohorts into one value result.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import pytest

from carnot import experiment_7405_v649_proof_audit as experiment


@pytest.fixture(scope="module")
def inputs() -> experiment.AuditInputs:
    """SCENARIO-CL-7405-COHORTS: Authenticate both planned cohort sources."""

    return experiment.collect_inputs(experiment.REPO_ROOT)


@pytest.fixture(scope="module")
def evidence(inputs: experiment.AuditInputs) -> experiment.AuditEvidence:
    """SCENARIO-CL-7405-TRUTH: Audit every synthetic request and arm once."""

    return experiment.audit_inputs(inputs)


def test_inputs_authenticate_each_cohort_without_a_whole_task_gate(
    inputs: experiment.AuditInputs,
) -> None:
    """SCENARIO-CL-7405-COHORTS: Live absence does not block synthetic reading."""

    assert inputs.synthetic["verdict_class"] == "circular_positive"
    assert inputs.live["honest_verdict"] == "blocked_gate_check_failed"
    assert inputs.hashes[experiment.SYNTHETIC_PATH.as_posix()] == (
        "sha256:7985849e93927d7fecaca23eac0cb185578b70e7b6f8a2d90777bc62eacd92b6"
    )
    assert inputs.hashes[experiment.LIVE_PATH.as_posix()] == (
        "sha256:f2efff28f8bd22ce6364866e5e867b7a8eee68c7388b907c452a6d05f12d8fe9"
    )
    assert all(row["passed"] for row in inputs.checks if row["cohort"] == "synthetic")
    assert any(not row["passed"] for row in inputs.checks if row["cohort"] == "live")
    assert all(row["terminal_blocking"] is False for row in inputs.checks)
    assert {row["original_verdict_class"] for row in inputs.historical_sources} == {
        "circular_positive",
        "blocked",
    }


def test_independent_truth_checker_rebuilds_formula_identity(
    inputs: experiment.AuditInputs,
) -> None:
    """SCENARIO-CL-7405-TRUTH: Canonical clauses, not producer answers, decide truth."""

    stream = inputs.protocol["evaluation_streams"][0]
    version = stream["versions"][0]
    assert experiment.formula_errors(version) == []
    assert experiment.independent_satisfiable(version, [1, -8]) is False
    assert experiment.independent_satisfiable(version, [-1, -8]) is True

    changed = deepcopy(version)
    changed["clauses"][0]["literals"] = [1, 8]
    assert "source_hash_mismatch" in experiment.formula_errors(changed)


def test_full_audit_recomputes_rows_proofs_costs_and_state(
    evidence: experiment.AuditEvidence,
) -> None:
    """SCENARIO-CL-7405-TRUTH: Every row and persistent state is independently checked."""

    assert evidence.errors == []
    assert len(evidence.rows) == 32 * 24 * 5
    assert len(evidence.witness_rows) == 431
    assert all(row["truth_match"] for row in evidence.rows)
    assert all(row["cost_complete"] for row in evidence.rows)
    assert evidence.continuity["incremental_state_persisted"] is True
    assert evidence.continuity["graph_cache_state_persisted"] is True
    assert evidence.continuity["reset_isolation_preserved"] is True
    assert evidence.metrics["unsafe_decisions"] == 0
    assert evidence.metrics["exact_decision_coverage"] == 1.0
    assert evidence.metrics["valid_erasure_witness_count"] == 431
    assert evidence.metrics["valid_erasure_witness_stream_count"] == 24
    assert evidence.metrics["bootstrap_draws"] == 10_000
    assert evidence.metrics["bootstrap_seed"] == 7_371_307
    assert evidence.metrics["paid_query_ratio_ci95_upper"] == {
        "persistent_incremental_exact_solver": pytest.approx(0.20833333333333334),
        "persistent_source_graph_reachability_cache": pytest.approx(0.2702702702702703),
    }
    assert evidence.metrics["full_cost_ratio_ci95_upper"] == {
        "persistent_incremental_exact_solver": pytest.approx(0.7784100495572027),
        "persistent_source_graph_reachability_cache": pytest.approx(0.5473049205061702),
    }


def test_seven_private_mutations_are_rejected(
    inputs: experiment.AuditInputs, evidence: experiment.AuditEvidence
) -> None:
    """SCENARIO-CL-7405-MUTATIONS: Each private evidence change fails closed."""

    rows = experiment.run_mutation_controls(inputs, evidence)

    assert {row["mutation"] for row in rows} == set(experiment.MUTATIONS)
    assert all(row["expected"] == "reject" for row in rows)
    assert all(row["observed"] == "reject" and row["passed"] for row in rows)
    assert all(row["rejecting_observation"] for row in rows)


def test_cohort_claims_preserve_synthetic_value_and_live_block(
    inputs: experiment.AuditInputs, evidence: experiment.AuditEvidence
) -> None:
    """SCENARIO-CL-7405-VALUE: Audit completion and combined value stay separate."""

    artifact = experiment.build_artifact_for_test(inputs, evidence)
    claims = {row["cohort"]: row for row in artifact["cohort_claim_rows"]}

    assert artifact["proof_audit_complete_score"] == 1
    assert artifact["proof_value_confirmed_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert claims["synthetic"]["verdict_class"] == "circular_positive"
    assert claims["synthetic"]["eligible"] is True
    assert claims["synthetic"]["confirmed_value"] == 1
    assert claims["live"]["verdict_class"] == "blocked"
    assert claims["live"]["eligible"] is False
    assert claims["live"]["confirmed_value"] == 0
    assert artifact["gate_check_summary"]["first_failure"] == {
        "upstream": experiment.LIVE_PATH.as_posix(),
        "path": experiment.LIVE_PATH.as_posix(),
        "check": "live_cohort_eligibility",
        "field": "honest_verdict",
        "expected": "eligible_complete_live_proof_rows",
        "observed": "blocked_gate_check_failed",
    }
    assert experiment.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(schema="bad"), "identity_mismatch"),
        (lambda value: value.update(MODEL_SPECS=["model"]), "model_declaration_mismatch"),
        (
            lambda value: value["invocation_counts"].update(model_loads_attempted=1),
            "invocation_counts_nonzero",
        ),
        (lambda value: value.update(inference_substrate={}), "substrate_mismatch"),
        (
            lambda value: value.update(inference_substrate_class="no_model_load"),
            "substrate_mismatch",
        ),
        (lambda value: value.update(execution_venue="host_cpu"), "substrate_mismatch"),
        (lambda value: value.update(verifier_is_oracle=False), "oracle_declaration_mismatch"),
        (lambda value: value.update(independent_metrics={}), "metrics_mismatch"),
        (lambda value: value.update(proof_value_confirmed_score=1), "reduction_mismatch"),
        (lambda value: value.update(promotion_score=1), "promotion_nonzero"),
        (lambda value: value.update(cohort_claim_rows=[]), "cohort_claim_mismatch"),
        (lambda value: value.update(mutation_rows=[]), "mutation_mismatch"),
        (lambda value: value.update(acceptance_gate_results=[]), "acceptance_gates_mismatch"),
        (lambda value: value.update(gate_check_summary={}), "gate_check_summary_mismatch"),
        (lambda value: value.update(field_principles={}), "field_principles_incomplete"),
        (
            lambda value: value.update(reproducibility_checksum="bad"),
            "reproducibility_checksum_mismatch",
        ),
        (lambda value: value.update(honest_verdict="complete_wrong"), "honest_verdict_mismatch"),
        (lambda value: value.update(verdict_class="null"), "verdict_class_mismatch"),
    ],
)
def test_artifact_validation_rejects_terminal_drift(
    inputs: experiment.AuditInputs,
    evidence: experiment.AuditEvidence,
    mutation,
    expected: str,
) -> None:
    """SCENARIO-CL-7405-ARTIFACT: Terminal fields cannot outrun audited evidence."""

    artifact = experiment.build_artifact_for_test(inputs, evidence)
    mutation(artifact)
    assert expected in experiment.validate_artifact(artifact)


def test_missing_synthetic_input_gets_its_own_blocked_cohort(tmp_path: Path) -> None:
    """SCENARIO-CL-7405-COHORTS: Missing source names an exact cohort failure."""

    inputs = experiment.collect_inputs(
        experiment.REPO_ROOT, synthetic_path=tmp_path / "missing.json"
    )
    evidence = experiment.audit_inputs(inputs)
    artifact = experiment.build_artifact_for_test(inputs, evidence)
    claims = {row["cohort"]: row for row in artifact["cohort_claim_rows"]}

    assert claims["synthetic"]["verdict_class"] == "blocked"
    assert claims["synthetic"]["missing_checks"]
    assert artifact["proof_audit_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert experiment.validate_artifact(artifact) == []


def test_scoped_plan_is_frozen_from_exp7358(tmp_path: Path) -> None:
    """SCENARIO-CL-7405-ARTIFACT: Exp7358 fixes the exact affected check set."""

    commands = experiment.scoped_command_plan(experiment.REPO_ROOT, tmp_path / "private")
    assert experiment.validate_scoped_command_plan(experiment.REPO_ROOT, commands) == []
    assert {command.name for command in commands} == set(experiment.REQUIRED_CHECK_NAMES)
    assert all("full_python_suite" not in command.name for command in commands)
    coverage = next(
        command for command in commands if command.name == "changed_module_coverage_report"
    )
    assert dict(getattr(coverage, "command_environment", ()))["COVERAGE_FILE"].startswith(
        str(tmp_path)
    )

    assert "required_command_names_changed" in experiment.validate_scoped_command_plan(
        experiment.REPO_ROOT, commands[:-1]
    )
    forbidden = [
        *commands,
        experiment.validation_scope.CommandSpec("full_python_suite", ("true",), "repository"),
    ]
    assert "full_python_suite_forbidden" in experiment.validate_scoped_command_plan(
        experiment.REPO_ROOT, forbidden
    )


def test_source_sidecar_and_cold_reload_preserve_producer_bytes(
    tmp_path: Path, inputs: experiment.AuditInputs, evidence: experiment.AuditEvidence
) -> None:
    """SCENARIO-CL-7405-ARTIFACT: Cold replay reads copies and preserves producers."""

    original = (experiment.REPO_ROOT / experiment.SYNTHETIC_PATH).read_bytes()
    source_hashes = experiment.write_source_sidecar(tmp_path, inputs)
    artifact = experiment.build_artifact_for_test(
        inputs, evidence, source_hashes={**inputs.hashes, **source_hashes}
    )
    assert experiment.cold_reload_errors(artifact, experiment.REPO_ROOT, check_sidecars=False) == []
    assert (experiment.REPO_ROOT / experiment.SYNTHETIC_PATH).read_bytes() == original

    changed = deepcopy(artifact)
    changed["rows"][0]["independent_satisfiable"] = not changed["rows"][0][
        "independent_satisfiable"
    ]
    experiment.finalize_artifact(changed)
    errors = experiment.cold_reload_errors(changed, experiment.REPO_ROOT, check_sidecars=False)
    assert "cold_row_replay_mismatch" in errors


def test_helpers_emit_progress_and_build_terminal_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-CL-7405-ARTIFACT: Boundaries and fresh readers stay explicit."""

    started = time.monotonic()
    experiment.progress(started, "test", "boundary", unit=1)
    assert "phase=test event=boundary" in capsys.readouterr().out
    assert experiment.utc_now().endswith("+00:00")
    span = experiment.phase_span("test", started, started, experiment.utc_now(), checkpoints=1)
    assert span["duration_s"] >= 0
    assert span["checkpoints"] == 1
    assert experiment.load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("[", encoding="utf-8")
    assert experiment.load_object(malformed) == {}

    terminal = experiment.terminal_commands(tmp_path / "candidate.json")
    assert {row.spec.name for row in terminal} == set(experiment.TERMINAL_CHECK_NAMES[:-1])
    args = experiment.parse_args(["--date", "20260919", "--output", str(tmp_path / "out.json")])
    assert args.date == "20260919"
    assert experiment.validate_artifact([]) == ["artifact_not_object"]


def test_defensive_source_checkers_reject_malformed_private_values(
    inputs: experiment.AuditInputs,
) -> None:
    """SCENARIO-CL-7405-MUTATIONS: Malformed clauses, proofs, costs, and order fail."""

    version = inputs.protocol["evaluation_streams"][0]["versions"][0]
    assert "formula_version_invalid" in experiment.formula_errors({"clauses": [{}]})
    assert "n_vars_invalid" in experiment.formula_errors({"version": "v", "clauses": [{}]})
    assert "clauses_invalid" in experiment.formula_errors({"version": "v", "n_vars": 1})
    bad_clause = deepcopy(version)
    bad_clause["clauses"][0]["clause_id"] = 4
    bad_clause["clauses"][1]["literals"] = [0]
    errors = experiment.formula_errors(bad_clause)
    assert {"clause_id_order", "clause_literal_invalid", "source_hash_mismatch"} <= set(errors)
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        experiment.independent_satisfiable(bad_clause, [])
    with pytest.raises(ValueError, match="assumption_literal_invalid"):
        experiment.independent_satisfiable(version, [999])

    witness = inputs.synthetic["erasure_witness_rows"][0]
    assert experiment.proof_errors(version, {}) == ["proof_missing"]
    changed = deepcopy(witness)
    changed["path"]["source_hash"] = "sha256:bad"
    assert "proof_source_hash_mismatch" in experiment.proof_errors(version, changed)
    changed = deepcopy(witness)
    changed["path"]["edges"] = []
    assert "proof_edges_missing" in experiment.proof_errors(version, changed)
    changed = deepcopy(witness)
    changed["path"]["edges"] = [None]
    assert "proof_edge_invalid" in experiment.proof_errors(version, changed)
    changed = deepcopy(witness)
    changed["path"]["edges"].append(deepcopy(changed["path"]["edges"][0]))
    changed["path"]["antecedent"] = -999
    changed["path"]["consequent"] = 999
    proof_errors = experiment.proof_errors(version, changed)
    assert {
        "proof_edge_gap",
        "proof_antecedent_mismatch",
        "proof_consequent_mismatch",
        "proof_path_id_mismatch",
    } <= set(proof_errors)

    assert experiment._cost_errors({}) == ["complete_service_cost_missing"]
    cost = deepcopy(inputs.synthetic["rows"][0])
    cost["complete_service_cost"]["checking"] = -1
    assert experiment._cost_errors(cost) == ["cost_phase_invalid"]
    cost = deepcopy(inputs.synthetic["rows"][0])
    cost["complete_service_cost"]["total"] += 1
    assert experiment._cost_errors(cost) == ["cost_total_mismatch"]

    protocol = deepcopy(inputs.protocol)
    protocol["evaluation_streams"].pop()
    assert experiment._protocol_request_errors(protocol) == ["stream_count_mismatch"]
    protocol = deepcopy(inputs.protocol)
    protocol["evaluation_streams"][0]["requests"].pop()
    assert "request_count_mismatch" in experiment._protocol_request_errors(protocol)
    protocol = deepcopy(inputs.protocol)
    protocol["evaluation_streams"][0]["requests"][0]["request_index"] = 4
    assert "request_order_mismatch" in experiment._protocol_request_errors(protocol)
    protocol = deepcopy(inputs.protocol)
    protocol["evaluation_streams"][1]["stream_id"] = protocol["evaluation_streams"][0]["stream_id"]
    assert "duplicate_stream" in experiment._protocol_request_errors(protocol)

    assert experiment._feedback_errors({}, {}) == ["feedback_schema_invalid"]
    feedback = deepcopy(inputs.synthetic["rows"][48])
    feedback["prior_verified_feedback_request_ids"] = [feedback["request_id"]]
    feedback["witnessed_earlier_feedback"] = ["not-prior"]
    feedback_errors = experiment._feedback_errors(
        feedback, {feedback["request_id"]: feedback["request_index"]}
    )
    assert {"feedback_not_earlier", "witness_not_in_verified_feedback"} <= set(feedback_errors)
    assert math.isinf(experiment._bootstrap_paired_upper((("only", 1.0, 0.0),)))


def test_corrupt_private_audit_exercises_all_owned_rejections(
    tmp_path: Path,
    inputs: experiment.AuditInputs,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-CL-7405-MUTATIONS: Row, witness, reset, and headline drift is visible."""

    synthetic = deepcopy(inputs.synthetic)
    synthetic["rows"].pop()
    rows = synthetic["rows"]
    rows[0]["request_id"] = "unknown-request"
    rows[1]["formula_version"] = "unknown-version"
    rows[2]["assumptions"] = [999]
    rows[3]["request_index"] = 99
    rows[4]["formula_source_hash"] = "sha256:bad"
    rows[5]["decision"] = "satisfiable"
    rows[6]["complete_service_cost"]["total"] += 1
    del rows[7]["complete_service_cost"]["verification"]
    synthetic["cost_rows"][8]["complete_service_cost"]["total"] += 1
    first_witness = synthetic["erasure_witness_rows"][0]
    first_witness["earlier_request_index"] = 99
    first_witness["path"]["antecedent"] = 999
    first_witness["changed_exact_work"] = False
    synthetic["erasure_witness_rows"][1]["formula_version"] = "unknown-version"
    synthetic["erasure_witness_rows"].append("bad-witness")
    synthetic["restart_rows"].pop()
    synthetic["independent_reduction"]["synthetic_metrics"]["unsafe_decisions"] = 99
    synthetic["synthetic_memory_value_score"] = 0
    corrupted = replace(inputs, synthetic=synthetic)

    evidence = experiment.audit_inputs(
        corrupted,
        emit_progress=True,
        started=time.monotonic(),
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert evidence.errors
    assert any(error.startswith("row:") for error in evidence.errors)
    assert any(error.startswith("witness:") for error in evidence.errors)
    assert "source_row_count_mismatch" in evidence.errors
    assert "source_cost_row_count_mismatch" in evidence.errors
    assert "persistent_state_or_reset_invalid" in evidence.errors
    assert "producer_metric_mismatch:unsafe_decisions" in evidence.errors
    assert "producer_headline_mismatch" in evidence.errors
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 32
    assert "stream_complete" in capsys.readouterr().out


def test_terminal_class_reducer_preserves_disqualified_null_and_positive(
    inputs: experiment.AuditInputs, evidence: experiment.AuditEvidence
) -> None:
    """SCENARIO-CL-7405-VALUE: All reachable closed terminal classes stay explicit."""

    base = experiment.build_artifact_for_test(inputs, evidence)
    disqualified = deepcopy(base)
    disqualified["audit_errors"] = ["private_failure"]
    experiment.finalize_artifact(disqualified)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["honest_verdict"].startswith("complete_disqualified")

    null = deepcopy(base)
    null["source_cohort_dispositions"]["live"].update(
        original_verdict_class="null", original_eligible=True
    )
    experiment.finalize_artifact(null)
    assert null["verdict_class"] == "null"
    assert null["honest_verdict"].startswith("complete_null")

    positive = deepcopy(base)
    positive["source_cohort_dispositions"]["live"].update(
        original_verdict_class="circular_positive", original_eligible=True
    )
    experiment.finalize_artifact(positive)
    assert positive["verdict_class"] == "circular_positive"
    assert positive["honest_verdict"].startswith("complete_circular_positive")
    positive["honest_verdict"] = "blocked_wrong"
    assert "honest_verdict_mismatch" in experiment.validate_artifact(positive)

    live_disqualified = deepcopy(base)
    live_disqualified["source_cohort_dispositions"]["live"].update(
        original_verdict_class="disqualified", original_eligible=False
    )
    experiment.finalize_artifact(live_disqualified)
    assert live_disqualified["verdict_class"] == "disqualified"


def test_sidecars_receipts_and_cold_reader_retain_failures(
    tmp_path: Path, inputs: experiment.AuditInputs, evidence: experiment.AuditEvidence
) -> None:
    """SCENARIO-CL-7405-ARTIFACT: Exact sidecars and cold mismatches remain observable."""

    mutations = experiment.run_mutation_controls(inputs, evidence)
    sidecars = experiment.write_audit_sidecars(tmp_path, inputs, evidence, mutations)
    assert len(sidecars) == 3
    normalized = experiment._normalized_receipt(
        {
            "name": "worktree_imports",
            "command": "python check",
            "command_argv": ["python", "check"],
            "command_environment": {"COVERAGE_FILE": "/tmp/cov"},
            "scope": "changed_modules",
            "exit_code": 0,
            "duration_s": 0.1,
            "log_path": "/tmp/log",
            "log_sha256": "sha256:" + "1" * 64,
            "passed": True,
            "timed_out": False,
            "resolved_imports": {"carnot.x": "/worktree/x.py"},
        }
    )
    assert normalized["resolved_imports"]
    assert normalized["environment"] == {"COVERAGE_FILE": "/tmp/cov"}

    artifact = experiment.build_artifact_for_test(inputs, evidence)
    artifact["source_artifact_hashes"][(experiment.RAW_DIR / "missing.json").as_posix()] = (
        "sha256:" + "2" * 64
    )
    artifact["source_artifact_hashes"][experiment.SYNTHETIC_PATH.as_posix()] = "sha256:" + "3" * 64
    artifact["erasure_witness_rows"].pop()
    artifact["independent_metrics"]["unsafe_decisions"] = 99
    experiment.finalize_artifact(artifact)
    errors = experiment.cold_reload_errors(artifact, experiment.REPO_ROOT)
    assert any(error.startswith("source_hash_mismatch:") for error in errors)
    assert any(error.startswith("audit_sidecar_hash_mismatch:") for error in errors)
    assert "cold_witness_replay_mismatch" in errors
    assert "cold_metric_replay_mismatch" in errors
