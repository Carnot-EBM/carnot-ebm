"""Verify the independent cold audit of committed predicate refinement.

Spec refs: REQ-CL-7214 and SCENARIO-CL-7214-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7214_v635_refinement_cold_audit as exp
from carnot.memory import transactional_constraint_memory as transactional


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fallback_state() -> dict[str, object]:
    """Build a small fallback that makes intervention effects easy to see."""

    return {
        "label_rows": [["lower_bound", 0, "reject"]],
        "defaults": {
            "cyclic_window": "reject",
            "lower_bound": "reject",
            "modular_equals": "reject",
            "upper_bound": "accept",
        },
        "evidence_ids": ["warmup"],
        "frozen": True,
    }


def _saved_state() -> dict[str, object]:
    """Build one reloadable state with two committed predicates."""

    controller = {
        "schema": "carnot.exp7212.controller.v1",
        "stream_id": "stream-1",
        "fallback": _fallback_state(),
        "reserved_validation": {
            "cyclic_window": [],
            "lower_bound": [],
            "modular_equals": [],
            "upper_bound": [],
        },
        "families": {
            family: {
                "hypotheses": [parameter],
                "support_ids": [f"support-{family}"],
                "validation_ids": [f"validation-{family}"],
                "candidate_parameter": parameter,
                "expected_parent_hash": "parent",
                "active_commit_receipt": None,
            }
            for family, parameter in {
                "cyclic_window": 2,
                "lower_bound": 5,
                "modular_equals": 7,
                "upper_bound": 9,
            }.items()
        },
        "fitting_queries": 4,
        "validation_queries": 4,
        "deployment_reads_private_hypotheses": False,
    }
    return {
        "unit_id": "1:witness_query_committed",
        "seed": 1,
        "arm": "witness_query_committed",
        "fallback": _fallback_state(),
        "controller_state": controller,
        "memory_states": {},
        "compiled_parameters": {"lower_bound": 5, "upper_bound": 9},
        "pending_public_state": [],
        "query_count": 8,
        "fitting_count": 4,
        "validation_count": 4,
        "state_hash": "saved",
        "reloadable": True,
    }


def _public_rows() -> list[dict[str, object]]:
    """Return public-only rows for direct replay tests."""

    return [
        {
            "event_id": "p1",
            "seed": 1,
            "chronology_index": 32,
            "family_id": "lower_bound",
            "numeric_value": 6,
            "public_input": "family=lower_bound;value=6",
        },
        {
            "event_id": "p2",
            "seed": 1,
            "chronology_index": 33,
            "family_id": "upper_bound",
            "numeric_value": 10,
            "public_input": "family=upper_bound;value=10",
        },
    ]


def test_fixed_contract_and_exact_principle_wrapper() -> None:
    """REQ-CL-7214: Freeze the audit substrate, fields, and wrapper contract."""

    assert exp.EXPERIMENT_ID == 7214
    assert exp.RUN_DATE == "20260911"
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert exp.INFERENCE_SUBSTRATE_CLASS == "cpu_exact_solver_or_simulator"
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.FIELD_PRINCIPLES["field_principles"] == (
        "Echo the reason for each field beside its actual evidence."
    )

    wrapped = {"principle": "why", "value": {"passed": True}}
    arbitrary = {"value": 1, "other": 2}
    extra = {"principle": "why", "value": 1, "evidence": "x"}
    assert exp.unwrap_principled(wrapped) == {"passed": True}
    assert exp.unwrap_principled(arbitrary) is arbitrary
    assert exp.unwrap_principled(extra) is extra
    assert exp.ExperimentPaths.defaults().artifact == exp.DEFAULT_ARTIFACT_PATH
    assert exp._sha256_path(Path("/tmp/exp7214-missing")) is None
    assert exp._percentile([], 0.5) == 0.0


def test_malformed_json_and_public_parse_mismatch_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7214-PRECONDITIONS: Malformed or changed input cannot pass."""

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    changed = _public_rows()
    changed[0]["numeric_value"] = 5
    with pytest.raises(ValueError, match="public_parse_mismatch"):
        exp.replay_public_decisions(_saved_state(), changed)


def test_preconditions_authenticate_real_producer_and_reject_quarantine(tmp_path: Path) -> None:
    """SCENARIO-CL-7214-PRECONDITIONS: Hashes and quarantine fail closed."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, producer, hashes = exp.collect_preconditions(
        REPO_ROOT, exp.DEFAULT_PRODUCER_PATH, paths
    )
    assert producer["refinement_run_complete_score"] == 1
    assert producer["refinement_value_score"] == 0
    assert all(row["passed"] for row in checks)
    assert hashes[str(exp.DEFAULT_PRODUCER_PATH)] is not None

    clean = exp.quarantine_state({}, "", "artifact.json", "exp-clean")
    flagged = exp.quarantine_state({"flagged_adversarial": True}, "", "artifact.json", "exp-test")
    listed = exp.quarantine_state({}, "- artifact.json\n", "artifact.json", "exp-test")
    assert clean["quarantined"] is False
    assert flagged["quarantined"] is True
    assert listed["quarantined"] is True

    forged = {
        "status": "complete",
        "run_date": exp.RUN_DATE,
        "refinement_run_complete_score": 1,
        "refinement_value_score": 0,
        "gate_check_summary": {"passed": True},
        "checkpoint_path": "results/checkpoints/missing.json",
        "checkpoint_hash": "sha256:" + "0" * 64,
        "source_artifact_hashes": {},
        "MODEL_SPECS": [],
        "model_invoked": False,
        "flagged_adversarial": True,
    }
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    failed, _producer, _hashes = exp.collect_preconditions(REPO_ROOT, forged_path, paths)
    quarantine = next(row for row in failed if row["check"] == "producer_not_quarantined")
    assert quarantine["passed"] is False


def test_event_reducer_ignores_producer_metrics_and_rebuilds_costs() -> None:
    """SCENARIO-CL-7214-RECOMPUTATION: Sealed actions own metrics and costs."""

    decisions = [
        {
            "unit_id": "1:witness_query_committed",
            "arm": "witness_query_committed",
            "seed": 1,
            "event_id": "e1",
            "chronology_index": 32,
            "phase": "drift",
            "prediction": "accept",
            "exact_label": "reject",
            "error": 0,
            "false_accept": 0,
            "abstention": 0,
            "prospective": True,
            "lookup_ns": 3,
            "selection_ns": 5,
            "score_ns": 7,
            "prediction_operation_index": 1,
            "query_decision_operation_index": 2,
            "authority_score_operation_index": 3,
            "future_label_visible_to_decision": False,
            "hidden_parameter_visible_to_decision": False,
            "hidden_audit_visible_to_decision": False,
        },
        {
            "unit_id": "1:witness_query_committed",
            "arm": "witness_query_committed",
            "seed": 1,
            "event_id": "e2",
            "chronology_index": 33,
            "phase": "recurrence",
            "prediction": "abstain",
            "exact_label": "accept",
            "error": 0,
            "false_accept": 0,
            "abstention": 0,
            "prospective": True,
            "lookup_ns": 11,
            "selection_ns": 13,
            "score_ns": 17,
            "prediction_operation_index": 4,
            "query_decision_operation_index": 5,
            "authority_score_operation_index": 6,
            "future_label_visible_to_decision": False,
            "hidden_parameter_visible_to_decision": False,
            "hidden_audit_visible_to_decision": False,
        },
    ]
    queries = [
        {
            "unit_id": "1:witness_query_committed",
            "arm": "witness_query_committed",
            "seed": 1,
            "event_id": "e1",
            "query_charged": True,
            "released": True,
            "role": "fitting",
            "request_index": 32,
            "release_index": 36,
            "pending_before": 0,
            "pending_after": 1,
            "validation_used_for_elimination": False,
            "queue_ns": 19,
            "update_ns": 23,
            "validation_ns": 0,
            "commit_ns": 29,
        }
    ]
    rows, receipts = exp.recompute_stream_metrics(decisions, queries)
    assert rows[0]["error"] == 2
    assert rows[0]["false_accept"] == 1
    assert rows[0]["abstention"] == 1
    assert rows[0]["drift_error"] == 1
    assert rows[0]["recurrence_error"] == 1
    assert rows[0]["total_cost_ns"] == 127
    assert receipts[0]["chronology_passed"] is True
    assert receipts[0]["max_pending"] == 1


def test_public_replay_and_interventions_use_distinct_state_parts() -> None:
    """SCENARIO-CL-7214-COLD/CAUSAL: Replay checks decisions, not hash changes."""

    state = _saved_state()
    public = _public_rows()
    baseline = exp.replay_public_decisions(state, public)
    assert [row["prediction"] for row in baseline] == ["accept", "reject"]
    assert all(row["authority_fields_present"] is False for row in baseline)

    deleted, deletion_receipt = exp.apply_state_intervention(state, "delete_templates")
    reset, reset_receipt = exp.apply_state_intervention(state, "full_reset")
    last, last_receipt = exp.apply_state_intervention(
        state, "remove_last_changed_predicate", family="upper_bound"
    )
    assert deleted["compiled_parameters"] == {}
    assert deleted["controller_state"] == state["controller_state"]
    assert deletion_receipt["fitting_state_preserved"] is True
    assert reset["compiled_parameters"] == {}
    assert reset["controller_state"] is None
    assert reset_receipt["full_learned_state_reset"] is True
    assert last["compiled_parameters"] == {"lower_bound": 5}
    assert last_receipt["removed_family"] == "upper_bound"
    assert exp.changed_decision_locations(baseline, exp.replay_public_decisions(last, public)) == [
        "p2"
    ]
    with pytest.raises(ValueError, match="unknown_state_intervention"):
        exp.apply_state_intervention(state, "unknown")

    memory_state = deepcopy(state)
    payload = {"schema": transactional.STATE_SCHEMA, "version": 1, "records": []}
    encoded = transactional.canonical_json_bytes(payload)
    memory_state["memory_states"] = {
        "lower_bound": {
            "state": payload,
            "state_bytes_b64": base64.b64encode(encoded).decode("ascii"),
            "state_hash": transactional.sha256_bytes(encoded),
        }
    }
    removed, _receipt = exp.apply_state_intervention(memory_state, "delete_templates")
    assert removed["memory_states"]["lower_bound"]["state"]["records"] == []


def test_feedback_controls_match_schedule_and_separate_phases() -> None:
    """SCENARIO-CL-7214-CONTROLS: Controls retain dates, budget, and phase rows."""

    baseline = [
        {
            "event_id": "e1",
            "seed": 1,
            "chronology_index": 32,
            "phase": "drift",
            "prediction": "accept",
            "exact_label": "accept",
        },
        {
            "event_id": "e2",
            "seed": 1,
            "chronology_index": 33,
            "phase": "recurrence",
            "prediction": "reject",
            "exact_label": "accept",
        },
    ]
    queries = [
        {
            "event_id": "e1",
            "seed": 1,
            "query_charged": True,
            "request_index": 32,
            "release_index": 36,
            "observed_label": "accept",
        },
        {
            "event_id": "e2",
            "seed": 1,
            "query_charged": True,
            "request_index": 33,
            "release_index": 40,
            "observed_label": "reject",
        },
    ]
    controls = exp.build_feedback_controls(baseline, queries, _saved_state())
    assert {row["control"] for row in controls} == {"shuffled_feedback", "no_feedback"}
    assert all(row["query_budget_matched"] for row in controls)
    assert all(row["request_and_release_dates_matched"] for row in controls)
    assert all(row["hidden_audit_label_access_count"] == 0 for row in controls)
    assert all(set(row["phase_errors"]) == {"drift", "recurrence"} for row in controls)


def test_stale_and_poisoned_transactions_restore_prior_behavior() -> None:
    """SCENARIO-CL-7214-ROLLBACK: Rejected changes preserve bytes and decisions."""

    rows = exp.rollback_probes(_saved_state(), _public_rows())
    assert {row["attack"] for row in rows} == {
        "stale_version_transaction",
        "poisoned_validation_response",
    }
    assert all(row["illegitimate_promotion_count"] == 0 for row in rows)
    assert all(row["byte_equal"] and row["hash_equal"] for row in rows)
    assert all(row["decision_parity"] and row["passed"] for row in rows)
    no_controller = deepcopy(_saved_state())
    no_controller["controller_state"] = None
    assert exp.rollback_probes(no_controller, _public_rows()) == []


def test_complete_null_finishes_without_promotion() -> None:
    """SCENARIO-CL-7214-TERMINAL: Complete mechanics retain producer null value."""

    complete, promotion, verdict_class, verdict = exp.derive_terminal_scores(
        producer_value_score=0,
        all_audit_checks_passed=True,
        equal_information=True,
    )
    assert (complete, promotion, verdict_class) == (1, 0, "null")
    assert verdict.startswith("complete_null:")
    positive = exp.derive_terminal_scores(
        producer_value_score=1,
        all_audit_checks_passed=True,
        equal_information=True,
    )
    assert positive[:3] == (1, 1, "circular_positive")
    disqualified = exp.derive_terminal_scores(
        producer_value_score=1,
        all_audit_checks_passed=True,
        equal_information=False,
    )
    assert disqualified[:3] == (1, 0, "disqualified")
    partial = exp.derive_terminal_scores(
        producer_value_score=0,
        all_audit_checks_passed=False,
        equal_information=True,
    )
    assert partial[:3] == (0, 0, "partial")


def test_missing_upstream_builds_terminal_row_free_block(tmp_path: Path) -> None:
    """SCENARIO-CL-7214-PRECONDITIONS: Missing external bytes are blocked."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, producer, hashes = exp.collect_preconditions(
        tmp_path / "missing-root", Path("missing.json"), paths
    )
    artifact = exp.build_blocked_artifact(
        checks,
        producer,
        source_hashes=hashes,
        paths=paths,
        duration_s=0.01,
    )
    assert artifact["status"] == artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_external:")
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["rows"] == artifact["cold_reload_rows"] == []
    assert artifact["gate_check_summary"]["failed_check"]
    assert exp.validate_artifact(artifact) == []


def test_command_refuses_invalid_worker_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7214: The command validates before its final atomic write."""

    monkeypatch.setattr(exp, "spawn_audit_worker", lambda _args: {"verdict_class": "null"})
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_error"])
    with pytest.raises(RuntimeError, match="cold audit artifact validation failed:forced_error"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)])
    assert not exp.ExperimentPaths.under(tmp_path).artifact.exists()


def test_validator_rejects_promotion_of_known_null() -> None:
    """SCENARIO-CL-7214-TERMINAL: Validator rejects a forged promotion."""

    artifact = exp.base_artifact(
        checks=[exp.gate_check("fixture", "test", "ready", True, True)],
        producer={"refinement_value_score": 0, "refinement_run_complete_score": 1},
        source_hashes={},
        paths=exp.ExperimentPaths(Path("checkpoint.json"), Path("artifact.json")),
        duration_s=1.0,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": exp.INFERENCE_SUBSTRATE,
            "inference_substrate_class": exp.INFERENCE_SUBSTRATE_CLASS,
            "refinement_audit_complete_score": 1,
            "memory_promotion_score": 1,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete_positive: forged",
        }
    )
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert "known_failed_value_promoted" in exp.validate_artifact(artifact)


def test_cold_worker_loads_one_checkpoint_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-CL-7214-COLD: The worker reads full state from checkpoint bytes."""

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "states": [_saved_state()],
                "public_probes": {"1": _public_rows()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CARNOT_7214_PARENT_PID", str(os.getppid()))
    args = exp.parse_args(
        ["--checkpoint-path", str(checkpoint), "--cold-seed", "1", "--cold-worker"]
    )
    result = exp.cold_reload_worker(args)
    assert len(result["decisions"]) == 2
    assert result["process_receipt"]["fresh_process"] is True


def test_spawn_and_main_dispatch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CL-7214: Worker, validation, and atomic-write dispatch paths are bounded."""

    args = exp.parse_args(["--date", exp.RUN_DATE])
    captured: dict[str, object] = {}

    def fake_stream(command: object, environment: object, **kwargs: object) -> dict[str, object]:
        captured.update(command=command, environment=environment, kwargs=kwargs)
        return {}

    monkeypatch.setattr(exp.cold_support, "_stream_subprocess", fake_stream)
    assert exp.spawn_audit_worker(args) == {}
    assert "--worker" in captured["command"]

    with pytest.raises(ValueError, match="run_date_must_equal"):
        exp.main(["--date", "20260910"])

    monkeypatch.setattr(exp, "cold_reload_worker", lambda _args: {"ok": True})
    assert exp.main(["--cold-worker", "--date", exp.RUN_DATE]) == 0
    monkeypatch.setattr(exp, "worker_artifact", lambda _args: {"ok": True})
    assert exp.main(["--worker", "--date", exp.RUN_DATE]) == 0

    missing = tmp_path / "missing-artifact.json"
    assert exp.main(["--validate", "--artifact-path", str(missing)]) == 1

    output_root = tmp_path / "output"
    monkeypatch.setattr(exp, "spawn_audit_worker", lambda _args: {})
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: [])
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(output_root)]) == 0
    assert exp.ExperimentPaths.under(output_root).artifact.exists()


def test_executable_wrapper_calls_module_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CL-7214: The requested executable wrapper owns the command boundary."""

    monkeypatch.setattr(exp, "main", lambda: 0)
    wrapper = REPO_ROOT / "scripts/experiments/experiment_7214_v635_refinement_cold_audit.py"
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert raised.value.code == 0


def test_worker_runs_complete_real_audit_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7214: Real files drive every audit phase before terminal classification."""

    paths = exp.ExperimentPaths.under(tmp_path)
    args = exp.parse_args(
        [
            "--date",
            exp.RUN_DATE,
            "--producer-artifact-path",
            str(exp.DEFAULT_PRODUCER_PATH),
            "--artifact-path",
            str(paths.artifact),
            "--checkpoint-path",
            str(paths.checkpoint),
        ]
    )
    monkeypatch.setenv("CARNOT_7214_PARENT_PID", str(os.getppid()))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "none")
    artifact = exp.worker_artifact(args)
    assert artifact["refinement_audit_complete_score"] == 1
    assert artifact["memory_promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert exp.validate_artifact(artifact) == []
    assert paths.checkpoint.exists()
