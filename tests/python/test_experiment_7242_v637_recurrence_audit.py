"""Verify the independent cold recurrence-memory audit.

Spec refs: REQ-CL-7242 and SCENARIO-CL-7242-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest

from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7242_v637_recurrence_audit as exp
from carnot.memory import transactional_constraint_memory as transactional


REPO_ROOT = Path(__file__).resolve().parents[2]


def _small_evidence() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Build two events for every arm with decisions sealed before labels."""

    public = [
        {
            "stream_id": "stream-01",
            "event_id": f"e{index}",
            "chronology_index": index,
            "family_id": "lower_bound",
            "numeric_value": 0,
            "public_input": "family=lower_bound;value=0",
        }
        for index in range(2)
    ]
    authority = [
        {
            "stream_id": "stream-01",
            "event_id": "e0",
            "chronology_index": 0,
            "stream_seed": 11,
            "family_id": "lower_bound",
            "numeric_value": 0,
            "exact_label": "reject",
            "drift_pattern": "unchanged_input_label_drift",
            "regime_id": "A",
            "hidden_parameter": 4,
        },
        {
            "stream_id": "stream-01",
            "event_id": "e1",
            "chronology_index": 1,
            "stream_seed": 11,
            "family_id": "lower_bound",
            "numeric_value": 0,
            "exact_label": "accept",
            "drift_pattern": "unchanged_input_label_drift",
            "regime_id": "B",
            "hidden_parameter": 0,
        },
    ]
    releases = [
        {
            "stream_id": "stream-01",
            "event_id": row["event_id"],
            "chronology_index": row["chronology_index"],
            "delay": 0,
            "observed_label": row["exact_label"],
        }
        for row in authority
    ]
    predictions = {
        "frozen_warmup": ("reject", "reject"),
        "destructive_packed_learner": ("reject", "accept"),
        "reset_relearn_no_archive": ("reject", "accept"),
        "unvalidated_stale_archive_reuse": ("reject", "reject"),
        "validation_selected_archive": ("reject", "accept"),
        "shuffled_nomination_validated": ("reject", "accept"),
    }
    decisions: list[dict[str, object]] = []
    for index, truth in enumerate(authority):
        for arm in exp7240.ARMS:
            prediction = predictions[arm][index]
            abstention = int(prediction == "abstain")
            classification_error = int(prediction != truth["exact_label"])
            decisions.append(
                {
                    "unit_id": f"stream-01:{arm}",
                    "stream_id": "stream-01",
                    "seed": 11,
                    "arm": arm,
                    "event_id": truth["event_id"],
                    "chronology_index": index,
                    "controller_input_fields": ["event_id", "family_id", "numeric_value"],
                    "held_out_label_visible_to_controller": False,
                    "prediction": prediction,
                    "prediction_energy": 0.0,
                    "prediction_receipt_completed_ns": 10 + index * 10,
                    "query_receipt_completed_ns": 11 + index * 10,
                    "label_accessed_ns": 12 + index * 10,
                    "later_released_label": truth["exact_label"],
                    "drift_pattern": truth["drift_pattern"],
                    "regime_id": truth["regime_id"],
                    "recurrence_eligible": index == 1,
                    "released_query_count_before": index,
                    "query_selected": index == 0,
                    "classification_error": classification_error,
                    "abstention": abstention,
                    "full_denominator_error": max(classification_error, abstention),
                    "false_accept": int(
                        prediction == "accept" and truth["exact_label"] == "reject"
                    ),
                    "later_changed_decision": int(index == 1 and prediction != "reject"),
                    "pre_release_difference": 0,
                    "lookup_cost_ns": 7,
                    "prediction_frozen_before_release": True,
                }
            )
    operations = [
        {
            "operation": "delayed_delivery",
            "stream_id": "stream-01",
            "arm": "shared_adaptive_feedback",
            "event_id": "e0",
            "cost_ns": 3,
            "source_sha256": transactional.sha256_json({"event_id": "e0"}),
        },
        {
            "operation": "validation_and_state_write",
            "stream_id": "stream-01",
            "arm": "validation_selected_archive",
            "event_id": "e0",
            "release_count": 1,
            "cost_ns": 5,
            "archive_reactivation_count": 1,
            "valid_reactivation_count": 1,
            "source_sha256": transactional.sha256_json(["e0"]),
        },
    ]
    return decisions, public, releases, authority, operations


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write test-owned evidence below ``tmp_path`` only."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _state_entry() -> tuple[dict[str, object], list[dict[str, object]]]:
    """Create one manifest entry and public probe from real controller bytes."""

    active = exp7226.PackedBeliefController.from_survivors(
        {family: {0} for family in exp7226.FAMILIES}
    )
    controller = exp7240.ArchivedBeliefController.from_active(active)
    state_bytes = controller.state_bytes()
    state = controller.state_dict()
    active_masks = {
        family: int(state["active"]["families"][family]["survivor_mask"])
        for family in exp7240.FAMILIES
    }
    archives: list[dict[str, object]] = []
    entry = {
        "unit_id": "stream-01:validation_selected_archive",
        "stream_id": "stream-01",
        "arm": "validation_selected_archive",
        "final_state_sha256": transactional.sha256_bytes(state_bytes),
        "final_state_bytes_b64": transactional.encode_bytes(state_bytes),
        "final_serialized_bytes": len(state_bytes),
        "active_survivor_masks": active_masks,
        "active_masks_sha256": transactional.sha256_json(active_masks),
        "archived_masks": archives,
        "archived_masks_sha256": transactional.sha256_json(archives),
        "operation_certificate_count": 0,
        "operation_certificates_sha256": transactional.sha256_json([]),
    }
    public = [
        {
            "stream_id": "stream-01",
            "event_id": "probe",
            "chronology_index": 0,
            "family_id": "lower_bound",
            "numeric_value": 0,
            "public_input": "family=lower_bound;value=0",
        }
    ]
    return entry, public


def test_req_cl_7242_contract_and_exact_principle_unwrap() -> None:
    """REQ-CL-7242: Bind the new task and keep invocation provenance empty."""

    assert exp.EXPERIMENT_ID == 7242
    assert exp.RUN_DATE == "20260912"
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert exp.INFERENCE_SUBSTRATE == "cpu_exact_solver_or_simulator"
    assert exp.INFERENCE_SUBSTRATE_CLASS == "cpu_exact_solver_or_simulator"
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.ExperimentPaths.defaults().artifact == exp.DEFAULT_ARTIFACT_PATH

    wrapped = {"principle": "why", "value": {"ready": True}}
    extra = {"principle": "why", "value": 1, "evidence": "not a wrapper"}
    missing = {"value": 1}
    assert exp.unwrap_principled(wrapped) == {"ready": True}
    assert exp.unwrap_principled(extra) is extra
    assert exp.unwrap_principled(missing) is missing


def test_scenario_cl_7242_preconditions_authenticate_both_upstreams(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7242-PRECONDITIONS: Exact complete producers can run."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, upstreams, hashes = exp.collect_preconditions(REPO_ROOT, paths)
    assert all(row["passed"] for row in checks)
    assert upstreams["exp7240"]["recurrence_fixture_ready_score"] == 1
    assert upstreams["exp7241"]["recurrence_run_complete_score"] == 1
    assert hashes[str(exp.DEFAULT_FIXTURE_ARTIFACT)] == exp.EXPECTED_FIXTURE_SHA256
    assert hashes[str(exp.DEFAULT_LEARNER_ARTIFACT)] == exp.EXPECTED_LEARNER_SHA256

    forged = deepcopy(upstreams["exp7241"])
    forged["flagged_adversarial"] = True
    forged_path = tmp_path / "quarantined.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    failed, _, _ = exp.collect_preconditions(
        REPO_ROOT,
        paths,
        learner_artifact=forged_path,
    )
    quarantine = next(row for row in failed if row["check"] == "exp7241_not_quarantined")
    assert quarantine["passed"] is False


def test_scenario_cl_7242_raw_reducer_joins_by_request_index() -> None:
    """SCENARIO-CL-7242-REDUCTION: Raw rows, not aggregates, define results."""

    decisions, public, releases, authority, operations = _small_evidence()
    result = exp.reduce_raw_evidence(
        decisions,
        public,
        releases,
        authority,
        operations,
        expected_stream_ids=("stream-01",),
        events_per_stream=2,
        warmup_count=1,
    )
    assert len(result["recomputed_seed_rows"]) == len(exp7240.ARMS)
    target = next(
        row for row in result["recomputed_seed_rows"] if row["arm"] == "validation_selected_archive"
    )
    assert target["future_error"] == 0
    assert target["recurrence_event_count"] == 1
    assert target["valid_reactivation_count"] == 1
    assert result["causal_summary"] == {
        "valid_reactivation_count": 1,
        "later_changed_decision_count": 1,
        "pre_release_difference_count": 0,
        "positive_control_changed_decision_count": 1,
    }
    assert all(row["passed"] for row in result["raw_check_rows"])
    assert result["control_summary"]["identical_input_label_drift_stream_count"] == 1
    assert result["control_summary"]["shuffled_candidate_unit_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_decision", "duplicate_decision"),
        ("premature_label", "premature_label"),
        ("label_mismatch", "request_index_join"),
        ("seed_mismatch", "seed_mismatch"),
        ("hidden_regime", "hidden_regime_input"),
        ("duplicate_feedback", "duplicate_feedback_credit"),
    ],
)
def test_scenario_cl_7242_raw_reducer_rejects_unsafe_evidence(
    mutation: str,
    message: str,
) -> None:
    """SCENARIO-CL-7242-CHRONOLOGY: Unsafe or mismatched rows fail closed."""

    decisions, public, releases, authority, operations = _small_evidence()
    if mutation == "duplicate_decision":
        decisions.append(deepcopy(decisions[0]))
    elif mutation == "premature_label":
        decisions[0]["label_accessed_ns"] = 9
    elif mutation == "label_mismatch":
        decisions[0]["later_released_label"] = "accept"
    elif mutation == "seed_mismatch":
        decisions[0]["seed"] = 12
    elif mutation == "hidden_regime":
        decisions[0]["controller_input_fields"] = ["event_id", "regime_id"]
    else:
        operations.append(deepcopy(operations[0]))
    with pytest.raises(exp.AuditEvidenceError, match=message):
        exp.reduce_raw_evidence(
            decisions,
            public,
            releases,
            authority,
            operations,
            expected_stream_ids=("stream-01",),
            events_per_stream=2,
            warmup_count=1,
        )


def test_scenario_cl_7242_independent_comparisons_and_terminal_scores() -> None:
    """SCENARIO-CL-7242-TERMINAL: Science and audit completion stay separate."""

    comparisons = [
        {
            "comparison_id": comparison_id,
            "ci95": [-0.2, -0.1],
            "estimate": 0.01 if comparison_id == "recurrence_error_vs_frozen" else -0.15,
        }
        for comparison_id, _, _ in exp.COMPARISON_SPECS
    ]
    causal = {
        "valid_reactivation_count": 1,
        "later_changed_decision_count": 2,
        "pre_release_difference_count": 0,
        "positive_control_changed_decision_count": 3,
    }
    gates = exp.score_acceptance_gates(comparisons, causal)
    assert all(row["pass"] for row in gates.values())
    assert exp.derive_terminal_scores(True, gates, True) == (
        1,
        1,
        "circular_positive",
        "complete_circular_positive: recurrence efficacy and every cold safety check reproduced",
    )
    failed = deepcopy(gates)
    failed["future_error_vs_reset_upper_ci95_lt_zero"]["pass"] = False
    assert exp.derive_terminal_scores(True, failed, True)[:3] == (1, 0, "null")
    assert exp.derive_terminal_scores(True, gates, False)[:3] == (1, 0, "null")
    assert exp.derive_terminal_scores(False, gates, True)[:3] == (0, 0, "null")
    assert exp._bootstrap_interval([], draws=4, salt="empty") == {
        "estimate": 0.0,
        "ci95": [0.0, 0.0],
    }


def test_scenario_cl_7242_restore_and_mutations_preserve_parent(tmp_path: Path) -> None:
    """SCENARIO-CL-7242-RESTORE/MUTATIONS: Cold restore and failures are exact."""

    entry, public = _state_entry()
    restore = exp.audit_state_entries([entry], public)
    assert len(restore) == 1
    assert restore[0]["action_mismatch_count"] == 0
    assert restore[0]["energy_mismatch_count"] == 0
    assert restore[0]["active_hash_byte_equal"] is True
    assert restore[0]["archive_hash_byte_equal"] is True
    assert restore[0]["passed"] is True

    mutation_rows = exp.run_transaction_controls(tmp_path)
    assert {row["control"] for row in mutation_rows} == {
        "withheld_update_future_only",
        "fresh_process_restore",
        "stale_invalid_archive",
        "capped_pending_queue",
        "out_of_order_delivery",
        "corrupted_certificate",
    }
    assert all(row["passed"] for row in mutation_rows)
    assert all(row["parent_state_unchanged"] for row in mutation_rows if row["rejected"])

    corrupt = deepcopy(entry)
    corrupt["final_state_bytes_b64"] = transactional.encode_bytes(b"{}")
    with pytest.raises(exp.AuditEvidenceError, match="invalid_restored_state"):
        exp.audit_state_entries([corrupt], public)


def test_scenario_cl_7242_worker_denies_authority_during_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-7242-RESTORE: State probes cannot reopen private labels."""

    decisions, public, releases, authority, operations = _small_evidence()
    entry, _ = _state_entry()
    inputs = {
        "decision_rows": tmp_path / "decisions.jsonl",
        "public_stream": tmp_path / "public.jsonl",
        "release_schedule": tmp_path / "releases.jsonl",
        "private_authority": tmp_path / "authority.jsonl",
        "operation_rows": tmp_path / "operations.jsonl",
        "state_manifest": tmp_path / "states.json",
    }
    for name, rows in (
        ("decision_rows", decisions),
        ("public_stream", public),
        ("release_schedule", releases),
        ("private_authority", authority),
        ("operation_rows", operations),
    ):
        _write_jsonl(inputs[name], rows)
    inputs["state_manifest"].write_text(
        json.dumps({"schema": "test", "entry_count": 1, "entries": [entry]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CARNOT_7242_PARENT_PID", str(os.getppid()))
    args = exp.parse_args(
        [
            "--reduce-worker",
            "--decision-rows",
            str(inputs["decision_rows"]),
            "--public-stream",
            str(inputs["public_stream"]),
            "--release-schedule",
            str(inputs["release_schedule"]),
            "--private-authority",
            str(inputs["private_authority"]),
            "--operation-rows",
            str(inputs["operation_rows"]),
            "--state-manifest",
            str(inputs["state_manifest"]),
            "--stream-ids",
            "stream-01",
            "--events-per-stream",
            "2",
            "--warmup-count",
            "1",
            "--bootstrap-draws",
            "9",
        ]
    )
    result = exp.reducer_worker(args)
    assert result["process_receipt"]["fresh_process"] is True
    assert result["process_receipt"]["authority_reopen_denied"] is True
    assert result["process_receipt"]["no_model_load"] is True
    assert len(result["recomputed_seed_rows"]) == 6


def test_scenario_cl_7242_blocked_and_complete_artifacts(tmp_path: Path) -> None:
    """SCENARIO-CL-7242-PRECONDITIONS/TERMINAL: Blocks are row-free and exact."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, upstreams, hashes = exp.collect_preconditions(REPO_ROOT, paths)
    failed = [dict(row) for row in checks]
    failed[0] = dict(failed[0], passed=False, observed_value=False)
    blocked = exp.build_blocked_artifact(failed, upstreams, hashes, paths, duration_s=0.1)
    assert blocked["status"] == blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == blocked["recomputed_seed_rows"] == []
    assert blocked["inference_substrate"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] is not None
    assert exp.validate_artifact(blocked) == []

    worker = {
        "recomputed_seed_rows": [],
        "comparison_rows": [],
        "acceptance_gate_results": {},
        "raw_check_rows": [],
        "restore_rows": [],
        "causal_summary": {},
        "control_summary": {},
        "process_receipt": {"fresh_process": True, "authority_reopen_denied": True},
    }
    with (
        patch.object(exp, "collect_preconditions", return_value=(failed, upstreams, hashes)),
        patch.object(exp, "spawn_reducer", return_value=worker) as spawn,
    ):
        stopped = exp.build_and_seal(REPO_ROOT, paths, progress=True)
    assert stopped["status"] == "blocked"
    spawn.assert_not_called()


def test_scenario_cl_7242_complete_build_hashes_mutation_sidecar(tmp_path: Path) -> None:
    """SCENARIO-CL-7242-TERMINAL: A completed build binds sidecar bytes and hash."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, upstreams, hashes = exp.collect_preconditions(REPO_ROOT, paths)
    decisions, public, releases, authority, operations = _small_evidence()
    worker = exp.reduce_raw_evidence(
        decisions,
        public,
        releases,
        authority,
        operations,
        expected_stream_ids=("stream-01",),
        events_per_stream=2,
        warmup_count=1,
    )
    worker["comparison_rows"] = exp.build_comparison_rows(
        worker["recomputed_seed_rows"], draws=9
    )
    worker["acceptance_gate_results"] = exp.score_acceptance_gates(
        worker["comparison_rows"], worker["causal_summary"]
    )
    entry, _ = _state_entry()
    worker["restore_rows"] = exp.audit_state_entries([entry], public)
    worker["mutation_rows"] = exp.run_transaction_controls(tmp_path / "controls")
    worker["process_receipt"] = {
        "fresh_process": True,
        "authority_reopen_denied": True,
        "gpu_disabled": True,
        "network_cache_offline": True,
        "no_model_load": True,
    }
    with (
        patch.object(exp, "collect_preconditions", return_value=(checks, upstreams, hashes)),
        patch.object(exp, "spawn_reducer", return_value=worker),
    ):
        artifact = exp.build_and_seal(
            REPO_ROOT,
            paths,
            stream_ids=("stream-01",),
            bootstrap_draws=9,
        )
    receipt = artifact["mutation_receipt_path"]
    assert receipt["sha256"] == exp._sha256_path(paths.mutation_receipts)
    assert receipt["bytes"] == paths.mutation_receipts.stat().st_size
    assert exp.validate_artifact(artifact) == []


def test_req_cl_7242_command_and_thin_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7242: The command validates before one atomic terminal write."""

    with pytest.raises(ValueError, match="run_date_must_equal_20260912"):
        exp.main(["--date", "20260911", "--output-root", str(tmp_path)])

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, upstreams, hashes = exp.collect_preconditions(REPO_ROOT, paths)
    checks[0] = dict(checks[0], passed=False, observed_value=False)
    artifact = exp.build_blocked_artifact(checks, upstreams, hashes, paths, duration_s=0.1)
    with patch.object(exp, "build_and_seal", return_value=artifact):
        assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)]) == 0
    assert json.loads(paths.artifact.read_text(encoding="utf-8"))["status"] == "blocked"
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path), "--validate"]) == 0

    monkeypatch.setattr(exp, "main", lambda: 0)
    monkeypatch.setattr(
        sys,
        "path",
        [item for item in sys.path if Path(item or ".").resolve() != REPO_ROOT / "python"],
    )
    wrapper = REPO_ROOT / "scripts/experiments/experiment_7242_v637_recurrence_audit.py"
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert raised.value.code == 0
