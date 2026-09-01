"""Tests for the sealed risk-sensitive learning reduction.

Spec refs: REQ-CL-6856 and SCENARIO-CL-6856-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import runpy

import pytest

from carnot import experiment_6856_sealed_risk_sensitive_learning_audit as exp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def controller_source() -> dict:
    """Load the sealed chronological controller rows once."""

    return exp.load_source(REPO / exp.CONTROLLER_RELATIVE_PATH)


@pytest.fixture(scope="session")
def credit_source() -> dict:
    """Load the sealed per-write counterfactual rows once."""

    return exp.load_source(REPO / exp.CREDIT_RELATIVE_PATH)


@pytest.fixture(scope="session")
def current_artifact(controller_source: dict, credit_source: dict) -> dict:
    """Build the full audit without writing tracked state."""

    return exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        controller=controller_source,
        credit=credit_source,
    )


def _measured_null_arm(decision_id: str) -> dict:
    """Build one arm whose measured outcomes are all null."""

    return {
        "decision_id": decision_id,
        "action": "abstain",
        "loss": None,
        "reward": None,
        "regret": None,
        "false_positive_injection": None,
        "helpful_memory_selected": None,
        "missed_reuse": None,
        "abstained": None,
    }


def _metric_row(
    *,
    decision_id: str,
    arm: str,
    family: str,
    order_id: str,
    effect: float | None,
    no_headroom: bool,
) -> dict:
    """Build one normalized row for portability reducer tests."""

    return {
        "decision_id": decision_id,
        "decision_sequence_index": 1,
        "arm": arm,
        "attack": "held_future",
        "family": family,
        "order_id": order_id,
        "split": "held_future",
        "metric_value": 0.0 if effect is None else effect,
        "effect_vs_no_memory": effect,
        "no_headroom": no_headroom,
        "false_positive_injection": False,
        "abstained": arm == "contextual_bandit",
    }


def test_req_cl_6856_complete_artifact_has_required_row_supported_fields(
    current_artifact: dict,
) -> None:
    """REQ-CL-6856: a complete reduction emits every required audit field."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(current_artifact)
    assert current_artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert current_artifact["verifier_is_oracle"] is False
    assert current_artifact["honest_verdict"].startswith("complete_")
    assert current_artifact["verdict_class"] == "null"
    assert current_artifact["continuous_self_learning_ready_score"] == 0
    assert current_artifact["rows"]
    assert all(
        {"decision_id", "arm", "attack", "family", "order_id", "metric_value"} <= set(row)
        for row in current_artifact["rows"]
    )
    assert current_artifact["counterfactual_support_results"]["harmful_write_count"] > 0


@pytest.mark.parametrize(
    "failure",
    ("controller_gate", "credit_gate", "missing_arm", "changed_split", "overlap"),
)
def test_scenario_cl_6856_preconditions_fail_closed(
    controller_source: dict,
    credit_source: dict,
    failure: str,
) -> None:
    """SCENARIO-CL-6856-PRECONDITIONS: invalid sealed evidence blocks."""

    controller = controller_source
    credit = credit_source
    split_manifest = dict(exp.SEALED_SPLIT_MANIFEST)
    reducer_imports: tuple[str, ...] = ()
    if failure == "controller_gate":
        controller = dict(controller_source)
        controller["risk_sensitive_controller_complete_score"] = 0
    elif failure == "credit_gate":
        credit = dict(credit_source)
        credit["counterfactual_memory_audit_complete_score"] = 0
    elif failure == "missing_arm":
        controller = deepcopy(controller_source)
        del controller["rows"][0]["arm_metrics"]["always_memory"]
    elif failure == "changed_split":
        split_manifest["held_future_start"] += 1
    else:
        reducer_imports = ("carnot.experiment_6854_risk_sensitive_abstention_memory_controller",)

    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        controller=controller,
        credit=credit,
        split_manifest=split_manifest,
        reducer_imports=reducer_imports,
    )
    assert artifact["continuous_self_learning_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_checks"]
    assert artifact["gate_check_summary"]["observed"] is not None


def test_scenario_cl_6856_all_null_row_blocks_claim(
    controller_source: dict,
    credit_source: dict,
) -> None:
    """SCENARIO-CL-6856-ALL-NULL: null measurements cannot support a claim."""

    controller = deepcopy(controller_source)
    decision_id = controller["rows"][0]["decision_id"]
    controller["rows"][0]["arm_metrics"] = {
        arm: _measured_null_arm(decision_id) for arm in exp.POLICY_ARMS
    }
    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        controller=controller,
        credit=credit_source,
    )
    checks = {row["check"]: row for row in artifact["gate_check_summary"]["checks"]}
    assert checks["non_null_decision_metrics"]["passed"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["continuous_self_learning_ready_score"] == 0


def test_scenario_cl_6856_aggregate_contradiction_uses_fresh_rows(
    controller_source: dict,
    credit_source: dict,
    current_artifact: dict,
) -> None:
    """SCENARIO-CL-6856-AGGREGATE-CONTRADICTION: fresh rows control."""

    controller = dict(controller_source)
    controller["held_future_effect"] = 99.0
    controller["false_positive_injection_rate"] = 1.0
    artifact = exp.build_artifact(
        REPO,
        run_date="20260901",
        duration_s=0.1,
        controller=controller,
        credit=credit_source,
    )
    assert artifact["held_future_effect"] == current_artifact["held_future_effect"]
    assert (
        artifact["false_positive_injection_rate"]
        == current_artifact["false_positive_injection_rate"]
    )
    assert artifact["leakage_results"]["aggregate_contradiction_count"] == 2
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["continuous_self_learning_ready_score"] == 0


def test_scenarios_cl_6856_poison_and_rollback_restore_parent_bytes() -> None:
    """SCENARIO-CL-6856-POISON/ROLLBACK: poison leaves exact parent bytes."""

    state = exp.BoundedAuditState(capacity=2, tombstone_capacity=2, credit_bound=1.0)
    assert state.apply("write-a", 0.25, sequence=1, expires_at=5)["accepted"] is True
    parent = state.to_bytes()
    poison = state.apply("poison", math.nan, sequence=2, expires_at=5)
    assert poison == {"accepted": False, "reason": "nonfinite_credit"}
    assert state.to_bytes() == parent

    clamped = state.apply("write-b", 100.0, sequence=2, expires_at=5)
    assert clamped["accepted"] is True
    assert clamped["clamped"] is True
    state.restore(parent)
    assert state.to_bytes() == parent


def test_scenarios_cl_6856_stale_correction_capacity_and_tombstone() -> None:
    """SCENARIO-CL-6856-STALE-CORRECTION/CAPACITY: eviction stays removed."""

    state = exp.BoundedAuditState(capacity=2, tombstone_capacity=2, credit_bound=1.0)
    state.apply("write-a", 0.1, sequence=1, expires_at=2)
    assert state.correct("write-a", 0.1, sequence=3) == {
        "accepted": False,
        "reason": "expired_support",
    }
    state.apply("write-b", 0.2, sequence=3, expires_at=8)
    state.apply("write-c", 0.3, sequence=4, expires_at=8)
    state.apply("write-d", 0.4, sequence=5, expires_at=8)
    assert len(state.active) == 2
    assert len(state.tombstones) <= 2
    assert "write-b" in state.tombstones
    assert state.correct("write-b", 0.1, sequence=6) == {
        "accepted": False,
        "reason": "tombstoned_write",
    }


def test_scenario_cl_6856_restart_is_byte_identical() -> None:
    """REQ-CL-6856 durability: stable bytes survive a fresh load."""

    state = exp.BoundedAuditState(capacity=3, tombstone_capacity=2, credit_bound=1.0)
    state.apply("write-a", -0.2, sequence=1, expires_at=4)
    state.apply("write-b", 0.3, sequence=2, expires_at=5)
    raw = state.to_bytes()
    loaded = exp.BoundedAuditState.from_bytes(raw)
    assert loaded.to_bytes() == raw
    with pytest.raises(ValueError, match="checksum"):
        exp.BoundedAuditState.from_bytes(raw[:-1] + b"0")
    wrapper = json.loads(raw)
    wrapper["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        exp.BoundedAuditState.from_bytes(json.dumps(wrapper).encode())


def test_scenario_cl_6856_family_removal_keeps_null_headroom_separate(
    current_artifact: dict,
) -> None:
    """SCENARIO-CL-6856-FAMILY-REMOVAL: each family stays separate."""

    rows = current_artifact["leave_one_family_out_rows"]
    assert len(rows) == 3
    assert len({row["held_out_family"] for row in rows}) == 3

    synthetic = [
        _metric_row(
            decision_id="d1",
            arm="contextual_bandit",
            family="family-a",
            order_id="order_1",
            effect=None,
            no_headroom=True,
        )
    ]
    reduced = exp.leave_one_group_out(
        synthetic,
        group_field="family",
        output_field="held_out_family",
        groups=("family-a",),
    )
    assert reduced[0]["mean_effect_vs_no_memory"] is None
    assert reduced[0]["no_headroom_count"] == 1


def test_scenario_cl_6856_order_removal_reports_each_order(
    current_artifact: dict,
) -> None:
    """SCENARIO-CL-6856-ORDER-REMOVAL: chronological orders are not pooled."""

    rows = current_artifact["leave_one_order_out_rows"]
    assert len(rows) == 5
    assert {row["held_out_order"] for row in rows} == {
        "order_1",
        "order_2",
        "order_3",
        "order_4",
        "order_5",
    }


def test_scenario_cl_6856_gates_do_not_accept_persistence_as_benefit(
    current_artifact: dict,
) -> None:
    """SCENARIO-CL-6856-GATES: durability cannot replace learning benefit."""

    assert current_artifact["durability_results"]["passed"] is True
    assert current_artifact["restart_results"]["byte_identical"] is True
    assert current_artifact["abstention_calibration"]["calibrated"] is False
    assert current_artifact["continuous_self_learning_ready_score"] == 0
    failed = {
        row["check"]
        for row in current_artifact["gate_check_summary"]["readiness_checks"]
        if not row["passed"]
    }
    assert "calibrated_abstention" in failed


def test_req_cl_6856_manifest_and_validation_are_independent(
    current_artifact: dict,
) -> None:
    """REQ-CL-6856: manifest names fresh code and artifact checks are strict."""

    manifest = current_artifact["fresh_reducer_manifest"]
    assert manifest["producer_module_imports"] == []
    assert manifest["source_overlap"] is False
    assert manifest["headlines_recomputed_from_rows"] is True
    assert exp.validate_artifact(current_artifact) == []
    malformed = dict(current_artifact)
    malformed.pop("rows")
    assert "missing field rows" in exp.validate_artifact(malformed)
    malformed = dict(current_artifact)
    malformed["verdict_class"] = "celebratory"
    assert "invalid verdict_class" in exp.validate_artifact(malformed)
    malformed = dict(current_artifact)
    malformed.update(
        {
            "inference_substrate": "unknown",
            "verifier_is_oracle": True,
            "honest_verdict": "null_without_terminal_prefix",
        }
    )
    assert exp.validate_artifact(malformed) == [
        "invalid inference_substrate",
        "verifier_is_oracle must be false",
        "honest_verdict must start with complete_",
    ]


def test_req_cl_6856_checksum_ignores_runtime_and_writer_uses_stable_json(
    current_artifact: dict,
    tmp_path: Path,
) -> None:
    """REQ-CL-6856 reproducibility excludes wall time and writes stable bytes."""

    changed = dict(current_artifact)
    changed["duration_s"] = 9.0
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert exp.reproducibility_checksum(changed) == current_artifact["reproducibility_checksum"]
    output = tmp_path / "artifact.json"
    exp.write_artifact(output, current_artifact)
    assert json.loads(output.read_text()) == current_artifact
    assert output.read_bytes().endswith(b"\n")
    malformed = dict(current_artifact)
    malformed.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing field honest_verdict"):
        exp.write_artifact(tmp_path / "malformed.json", malformed)


def test_req_cl_6856_small_guard_branches_use_fail_closed_values(tmp_path: Path) -> None:
    """REQ-CL-6856: malformed identities, sources, and metrics fail closed."""

    assert exp.sha256_file(tmp_path / "missing.json") is None
    list_path = tmp_path / "list.json"
    list_path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        exp.load_source(list_path)
    assert exp._decision_dimensions("malformed") == ("invalid_order", "invalid_family")
    assert exp._abstention_calibration([])["calibrated"] is False
    assert exp._terminal_disposition(
        readiness=1,
        contradictions=[],
        harmful_write_count=0,
        calibrated=True,
    ) == ("positive", "complete_positive_continuous_self_learning_evidence")
    measured = {
        field: False if field in exp.MEASURED_ARM_FIELDS[3:] else 0.0
        for field in exp.MEASURED_ARM_FIELDS
    }
    checks = exp.validate_preconditions(
        {
            "risk_sensitive_controller_complete_score": 1,
            "rows": [
                {
                    "decision_id": "malformed",
                    "decision_sequence_index": 1,
                    "arm_metrics": {arm: dict(measured) for arm in exp.POLICY_ARMS},
                }
            ],
        },
        {"counterfactual_memory_audit_complete_score": 1, "rows": []},
    )
    dimensions = {
        row["check"]: row for row in checks["checks"]
    }["declared_family_and_order_dimensions"]
    assert dimensions["passed"] is False


def test_req_cl_6856_credit_poison_unknown_and_unsupported_are_visible() -> None:
    """REQ-CL-6856: raw per-write support never invents a usable credit."""

    controller = {
        "controller_schema": {"max_updates_per_action": 256},
        "update_rows": [{"update_receipt_sha256": "write-a"}],
    }
    decisions = [
        _metric_row(
            decision_id="known",
            arm="contextual_bandit",
            family="family-a",
            order_id="order_1",
            effect=0.1,
            no_headroom=False,
        )
    ]
    rows = [
        {
            "decision_id": "unknown",
            "write_id": "poison",
            "metric_value": math.nan,
            "counterfactual_metric": "marginal_value",
            "method": "poison",
        },
        {
            "decision_id": "known",
            "write_id": "write-a",
            "metric_value": 0.1,
            "counterfactual_metric": "substitution_effect",
            "method": "exact_observed_donor_transition",
        },
    ]
    reduced = exp.reduce_credit_rows(
        rows,
        controller=controller,
        normalized_decisions=decisions,
    )
    assert reduced["valid_counterfactual_support"] is False
    assert reduced["poison_metric_count"] == 1
    assert reduced["unknown_decision_count"] == 1
    assert reduced["unsupported_write_count"] == 1


def test_req_cl_6856_state_rejects_removed_missing_and_nonfinite_corrections() -> None:
    """REQ-CL-6856: bounded state rejects every unsupported update form."""

    state = exp.BoundedAuditState(capacity=1, tombstone_capacity=1, credit_bound=1.0)
    state.apply("write-a", 0.1, sequence=1, expires_at=5)
    state.apply("write-b", 0.2, sequence=2, expires_at=5)
    assert state.apply("write-a", 0.3, sequence=3, expires_at=5) == {
        "accepted": False,
        "reason": "tombstoned_write",
    }
    assert state.correct("missing", 0.1, sequence=3) == {
        "accepted": False,
        "reason": "missing_support",
    }
    assert state.correct("write-b", math.nan, sequence=3) == {
        "accepted": False,
        "reason": "nonfinite_correction",
    }
    state.apply("write-c", 0.3, sequence=3, expires_at=5)
    state.apply("write-d", 0.4, sequence=4, expires_at=5)
    assert len(state.tombstones) == 1


def test_req_cl_6856_cli_and_wrapper_write_only_requested_output(
    controller_source: dict,
    credit_source: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6856: the CLI supports isolated end-to-end output."""

    controller_path = tmp_path / "controller.json"
    credit_path = tmp_path / "credit.json"
    output_path = tmp_path / "result.json"
    controller_path.write_text(json.dumps(controller_source))
    credit_path.write_text(json.dumps(credit_source))
    assert (
        exp.main(
            [
                "--date",
                "20260901",
                "--controller",
                str(controller_path),
                "--credit",
                str(credit_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text())["honest_verdict"].startswith("complete_")

    wrapper = REPO / "scripts/experiments/experiment_6856_sealed_risk_sensitive_learning_audit.py"
    monkeypatch.setattr("sys.argv", [str(wrapper), "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert exc.value.code == 0

    package = REPO / "python/carnot/experiment_6856_sealed_risk_sensitive_learning_audit.py"
    monkeypatch.setattr("sys.argv", [str(package), "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(package), run_name="__main__")
    assert exc.value.code == 0
