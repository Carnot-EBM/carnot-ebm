"""Verify the independent packed-belief cold and causal audit.

Spec refs: REQ-CL-7228 and SCENARIO-CL-7228-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7228_v636_belief_cold_audit as exp
from carnot.memory import transactional_constraint_memory as transactional


REPO_ROOT = Path(__file__).resolve().parents[2]


def _public(event_id: str = "probe-1", value: int = 16) -> dict[str, object]:
    """Return one public event with no correctness authority."""

    return {
        "event_id": event_id,
        "seed": exp.STREAM_SEEDS[0],
        "chronology_index": exp.EVENTS_PER_SEED,
        "family_id": "lower_bound",
        "numeric_value": value,
        "public_input": f"family=lower_bound;value={value}",
    }


def _state_record() -> dict[str, object]:
    """Build equivalent packed and reference states for direct cold probes."""

    packed = exp7226.PackedBeliefController.from_survivors(
        {family: {4, 5, 6} for family in exp7226.FAMILIES}
    )
    reference = exp.reference_from_survivors({family: {4, 5, 6} for family in exp7226.FAMILIES})
    return {
        "seed": exp.STREAM_SEEDS[0],
        "packed_online_memory": packed.state_dict(),
        "reference_online_version_space": reference.state_dict(),
    }


def _delayed_release() -> dict[str, object]:
    """Create one fixed correction that becomes eligible four events later."""

    return {
        "event_id": "audit-correction-1",
        "family_id": "lower_bound",
        "numeric_value": 16,
        "observed_label": "reject",
        "role": "support",
        "request_index": exp.EVENTS_PER_SEED,
        "release_index": exp.EVENTS_PER_SEED + 4,
    }


def test_req_cl_7228_contract_and_exact_wrapper() -> None:
    """REQ-CL-7228: Freeze the task identity, substrate, and required fields."""

    assert exp.EXPERIMENT_ID == 7228
    assert exp.RUN_DATE == "20260912"
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert exp.INFERENCE_SUBSTRATE == "cpu_exact_solver_or_simulator"
    assert exp.INFERENCE_SUBSTRATE_CLASS == "cpu_exact_solver_or_simulator"
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.FIELD_PRINCIPLES["field_principles"].startswith("Annotate actual values")

    wrapped = {"principle": "why", "value": {"passed": True}}
    arbitrary = {"value": 1, "other": 2}
    extra = {"principle": "why", "value": 1, "evidence": "x"}
    assert exp.unwrap_principled(wrapped) == {"passed": True}
    assert exp.unwrap_principled(arbitrary) is arbitrary
    assert exp.unwrap_principled(extra) is extra
    assert exp.ExperimentPaths.defaults().artifact == exp.DEFAULT_ARTIFACT_PATH


def test_scenario_cl_7228_preconditions_authenticate_completion_not_value(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7228-PRECONDITIONS: A complete null producer can run."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, producer, hashes = exp.collect_preconditions(REPO_ROOT, paths)
    assert all(row["passed"] for row in checks)
    assert producer["belief_run_complete_score"] == 1
    assert producer["belief_learning_value_score"] == 0
    assert hashes[str(exp.DEFAULT_PRODUCER_PATH)] == exp.EXPECTED_PRODUCER_SHA256

    forged = deepcopy(producer)
    forged["flagged_adversarial"] = True
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    failed, _, _ = exp.collect_preconditions(
        REPO_ROOT,
        paths,
        producer_path=forged_path,
    )
    assert (
        next(row for row in failed if row["check"] == "producer_not_quarantined")["passed"] is False
    )


def test_scenario_cl_7228_recomputation_uses_rows_and_refuses_denominator_loss() -> None:
    """SCENARIO-CL-7228-RECOMPUTATION: Row values override false aggregates."""

    rows = [
        {
            "unit_id": "1:packed_online_memory",
            "arm": "packed_online_memory",
            "seed": 1,
            "event_id": "e0",
            "chronology_index": 0,
            "prediction": "accept",
            "exact_label": "reject",
            "error": 0,
            "false_accept": 0,
            "abstention": 0,
            "prospective": False,
            "window": "warmup",
            "query_admitted": True,
            "pending_capacity_use": 1,
            "released_feedback_ids": [],
            "prediction_before_release": True,
            "future_label_visible_to_decision": False,
            "hidden_parameter_visible_to_decision": False,
        },
        {
            "unit_id": "1:packed_online_memory",
            "arm": "packed_online_memory",
            "seed": 1,
            "event_id": "e1",
            "chronology_index": 1,
            "prediction": "abstain",
            "exact_label": "accept",
            "error": 0,
            "false_accept": 1,
            "abstention": 0,
            "prospective": True,
            "window": "recurrence",
            "query_admitted": False,
            "pending_capacity_use": 0,
            "released_feedback_ids": ["e0"],
            "prediction_before_release": True,
            "future_label_visible_to_decision": False,
            "hidden_parameter_visible_to_decision": False,
        },
    ]
    rebuilt, receipts = exp.recompute_stream_metrics(
        rows,
        expected_seeds=(1,),
        expected_arms=("packed_online_memory",),
        events_per_seed=2,
        warmup_count=1,
    )
    assert rebuilt[0]["error"] == 1
    assert rebuilt[0]["false_accept"] == 0
    assert rebuilt[0]["abstention"] == 1
    assert rebuilt[0]["query_count"] == rebuilt[0]["released_query_count"] == 1
    assert receipts[0]["stored_outcome_mismatch_count"] == 5
    assert receipts[0]["passed"] is False

    with pytest.raises(exp.AuditEvidenceError, match="missing_denominator"):
        exp.recompute_stream_metrics(
            rows[:-1],
            expected_seeds=(1,),
            expected_arms=("packed_online_memory",),
            events_per_seed=2,
            warmup_count=1,
        )
    with pytest.raises(exp.AuditEvidenceError, match="unit_set"):
        exp.recompute_stream_metrics(
            [],
            expected_seeds=(1,),
            expected_arms=("packed_online_memory",),
            events_per_seed=2,
            warmup_count=1,
        )
    wrong_window = deepcopy(rows)
    wrong_window[1]["prospective"] = False
    with pytest.raises(exp.AuditEvidenceError, match="prospective"):
        exp.recompute_stream_metrics(
            wrong_window,
            expected_seeds=(1,),
            expected_arms=("packed_online_memory",),
            events_per_seed=2,
            warmup_count=1,
        )
    assert exp._percentile([], 0.5) == 0.0
    assert exp._bootstrap_interval([], 3, "empty") == {
        "estimate": 0.0,
        "ci95": [0.0, 0.0],
        "draws": 3,
    }


def test_scenario_cl_7228_rebuilds_real_comparisons_and_deletions() -> None:
    """SCENARIO-CL-7228-RECOMPUTATION: Independent reducers match raw evidence."""

    producer = json.loads((REPO_ROOT / exp.DEFAULT_PRODUCER_PATH).read_text())
    decisions = exp.read_jsonl(REPO_ROOT / producer["decision_rows_path"]["path"])
    one_seed = exp.STREAM_SEEDS[:1]
    selected = [row for row in decisions if int(row["seed"]) in one_seed]
    rebuilt, _ = exp.recompute_stream_metrics(selected, expected_seeds=one_seed)
    comparisons = exp.recompute_comparisons(rebuilt, draws=37)
    deletions = exp.recompute_deletions(selected)
    assert len(rebuilt) == len(exp.ARMS)
    assert len(comparisons) == 3
    assert all(row["independent_unit_count"] == 1 for row in comparisons)
    assert len(deletions) == 1
    assert deletions[0]["changed_decision_count"] > 0
    assert deletions[0]["pre_release_difference_count"] == 0
    with pytest.raises(exp.AuditEvidenceError, match="deletion"):
        exp.recompute_deletions(selected[:-1])


def test_scenario_cl_7228_cold_and_transaction_attacks() -> None:
    """SCENARIO-CL-7228-COLD/TRANSACTIONS: Reload, attacks, and rollback are exact."""

    result = exp.run_state_probe(_state_record(), [_public()], _delayed_release())
    assert result["cold_row"]["prediction_mismatch_count"] == 0
    assert result["cold_row"]["vote_mismatch_count"] == 0
    assert result["cold_row"]["next_update_mismatch_count"] == 0
    assert result["cold_row"]["passed"] is True
    assert {row["attack"] for row in result["rollback_rows"]} == {
        "delayed_correction",
        "stale_parent_commit",
        "corrupted_mask",
        "duplicate_release",
        "rollback",
    }
    assert all(row["passed"] for row in result["rollback_rows"])
    assert result["removal_row"]["labels_joined_after_decision"] is True
    with pytest.raises(exp.AuditEvidenceError, match="invalid_reference_families"):
        exp.reference_from_state({})


def test_scenario_cl_7228_isolated_worker_denies_future_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7228-ISOLATION: A real process policy denies future labels."""

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "states": [_state_record()],
                "public_probes": {str(exp.STREAM_SEEDS[0]): [_public()]},
                "delayed_releases": {str(exp.STREAM_SEEDS[0]): _delayed_release()},
            }
        ),
        encoding="utf-8",
    )
    sidecar = tmp_path / "future.jsonl"
    sidecar.write_text('{"exact_label":"accept"}\n', encoding="utf-8")
    monkeypatch.setenv("CARNOT_7228_PARENT_PID", str(os.getppid()))
    monkeypatch.setenv("CARNOT_7228_FORBIDDEN_SIDECAR", str(sidecar))
    monkeypatch.setenv("CARNOT_7228_VARIANT", "shifted")
    args = exp.parse_args(
        [
            "--cold-worker",
            "--checkpoint-path",
            str(checkpoint),
            "--cold-seed",
            str(exp.STREAM_SEEDS[0]),
        ]
    )
    result = exp.cold_reload_worker(args)
    receipt = result["process_receipt"]
    assert receipt["fresh_process"] is True
    assert receipt["future_sidecar_open_denied"] is True
    assert receipt["future_sidecar_read_success"] is False
    assert receipt["variant"] == "shifted"
    assert (
        exp.main(
            [
                "--cold-worker",
                "--checkpoint-path",
                str(checkpoint),
                "--cold-seed",
                str(exp.STREAM_SEEDS[0]),
            ]
        )
        == 0
    )


def test_scenario_cl_7228_history_terminal_and_blocked_contract(tmp_path: Path) -> None:
    """SCENARIO-CL-7228-HISTORY/TERMINAL: Nulls stay null and blocks stay row-free."""

    assert exp.derive_terminal_scores(0, True, True)[:3] == (1, 0, "null")
    assert exp.derive_terminal_scores(1, True, True)[:3] == (1, 1, "circular_positive")
    assert exp.derive_terminal_scores(1, True, False)[:3] == (1, 0, "disqualified")
    assert exp.derive_terminal_scores(0, False, True)[:3] == (0, 0, "partial")

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, producer, hashes = exp.collect_preconditions(
        tmp_path / "missing",
        paths,
        producer_path=Path("missing.json"),
    )
    artifact = exp.build_blocked_artifact(
        checks,
        producer,
        hashes,
        paths,
        duration_s=0.01,
    )
    assert artifact["status"] == artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == artifact["cold_reload_rows"] == []
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"] is not None
    assert exp.validate_artifact(artifact) == []
    blocked = exp.build_and_seal(
        tmp_path / "missing-root",
        paths,
        producer_path=Path("missing.json"),
        seeds=exp.STREAM_SEEDS[:1],
        progress=True,
    )
    assert blocked["status"] == "blocked"


def test_req_cl_7228_one_stream_build_and_command_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7228: Real producer bytes drive a complete private audit."""

    paths = exp.ExperimentPaths.under(tmp_path / "build")
    artifact = exp.build_and_seal(
        REPO_ROOT,
        paths,
        seeds=exp.STREAM_SEEDS[:1],
        progress=True,
    )
    assert artifact["belief_audit_complete_score"] == 1
    assert artifact["belief_promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["v635_history_receipt"]["memory_promotion_score"] == 0
    assert exp.validate_artifact(artifact, expected_seeds=exp.STREAM_SEEDS[:1]) == []

    with pytest.raises(ValueError, match="run_date_must_equal"):
        exp.main(["--date", "20260911", "--output-root", str(tmp_path / "bad")])

    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: [])
    destination = tmp_path / "command"
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(destination)]) == 0
    assert exp.ExperimentPaths.under(destination).artifact.is_file()
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(destination), "--validate"]) == 0
    monkeypatch.setattr(exp, "_load_object", lambda _path: artifact)
    assert exp.main(["--date", exp.RUN_DATE, "--validate"]) == 0

    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(exp.AuditEvidenceError, match="final_validation_failed:forced"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "invalid")])

    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: [])
    monkeypatch.setattr(exp, "main", lambda: 0)
    monkeypatch.setattr(
        sys,
        "path",
        [item for item in sys.path if Path(item or ".").resolve() != (REPO_ROOT / "python")],
    )
    wrapper = REPO_ROOT / "scripts/experiments/experiment_7228_v636_belief_cold_audit.py"
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert raised.value.code == 0


def test_scenario_cl_7228_defensive_parsers_and_validator(tmp_path: Path) -> None:
    """SCENARIO-CL-7228-PRECONDITIONS: Malformed bytes and forged promotion fail."""

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    assert exp._sha256_path(tmp_path / "absent") is None
    with pytest.raises(exp.AuditEvidenceError, match="invalid_jsonl"):
        exp.read_jsonl(malformed)
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(exp.AuditEvidenceError, match="invalid_jsonl"):
        exp.read_jsonl(malformed)
    assert exp.validate_artifact({})[0].startswith("missing_fields:")

    paths = exp.ExperimentPaths(Path("checkpoint.json"), Path("artifact.json"))
    artifact = exp.base_artifact([], {}, {}, paths, duration_s=1.0)
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": exp.INFERENCE_SUBSTRATE,
            "inference_substrate_class": exp.INFERENCE_SUBSTRATE_CLASS,
            "belief_audit_complete_score": 1,
            "belief_promotion_score": 1,
            "verdict_class": "circular_positive",
            "honest_verdict": "complete: forged",
            "producer_gate_receipt": {
                "belief_run_complete_score": 1,
                "belief_learning_value_score": 0,
            },
        }
    )
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert "null_promoted" in exp.validate_artifact(artifact)
    errors = exp._audit_error_names(
        (("empty", []), ("failed", [{"passed": False}])),
        0,
        1,
        {"all_workers_isolated": False, "protected_inputs_unchanged": False},
    )
    assert errors == [
        "empty",
        "failed",
        "state_count",
        "runtime_isolation",
        "protected_inputs_changed",
    ]
    with pytest.raises(exp.AuditEvidenceError, match="forced:a,b"):
        exp._require_valid(["a", "b"], "forced")
    exp._require_valid([], "unused")

    packed = exp7226.PackedBeliefController()
    changed = packed.state_dict()
    changed["schema"] = "wrong"
    with pytest.raises(ValueError, match="invalid_state_schema"):
        exp7226.PackedBeliefController.from_state(changed)
    assert transactional.sha256_json({"stable": True}).startswith("sha256:")
