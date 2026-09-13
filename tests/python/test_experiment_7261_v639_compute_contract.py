"""Tests for the V639 authenticated compute-floor contract.

Spec refs: REQ-SUBSTRATE-CLASS-1 and SCENARIO-SUBSTRATE-CLASS-9 through
SCENARIO-SUBSTRATE-CLASS-11.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest

from carnot import experiment_7261_v639_compute_contract as exp7261
from scripts import adversarial_verify as av


ROOT = Path(__file__).resolve().parents[2]


def test_scenario_substrate_class_9_boundary_and_contradiction_panel() -> None:
    """SCENARIO-SUBSTRATE-CLASS-9 exercises every independent fixture."""

    rows = exp7261.evaluate_boundary_fixtures()
    assert len(rows) >= 24
    assert len({row["fixture_id"] for row in rows}) == len(rows)
    assert all(row["passed"] is True for row in rows)
    assert sum(row["control_kind"] == "duration_boundary" for row in rows) >= 12
    assert sum(row["control_kind"] == "contradiction" for row in rows) >= 8
    assert any(row["fixture_id"] == "bounded_exp7237_59_466359" for row in rows)


def test_scenario_substrate_class_9_exp7237_uses_bounded_floor() -> None:
    """SCENARIO-SUBSTRATE-CLASS-9 reproduces the 59.466359/60-second conflict."""

    payload = {
        "run_date": "20260912",
        "honest_verdict": "complete_null_bounded_canary_fixture",
        "inference_substrate": "live_llm_inference",
        "inference_substrate_class": "model_bounded_generation",
        "duration_s": 59.46635937620886,
        "model_invoked": True,
        "generation_calls_attempted": 48,
        "generation_calls_completed": 48,
    }
    floor = av.duration_floor_for_artifact(payload)
    assert floor == {
        "substrate": "model_bounded_generation",
        "min_duration_s": 10.0,
        "reason": "substrate_class",
    }
    flags: list[av.Flag] = []
    av.check_duration_vs_claim(payload, flags)
    av.check_substrate_class(payload, flags)
    assert [flag.kind for flag in flags] == []


@pytest.mark.parametrize(
    ("class_value", "typed", "expected_floor"),
    [
        (None, {"model_invoked": True}, 60.0),
        ({"value": "model_bounded_generation"}, {"model_invoked": True}, 60.0),
        ("unknown_model_class", {"model_invoked": True}, 60.0),
        ("hardware_board", {"model_invoked": True}, 60.0),
        ("model_bounded_generation", {"model_invoked": False}, 60.0),
        ("model_bounded_generation", {"model_invoked": True, "llm_invoked": False}, 60.0),
        ("no_model_load", {"generation_calls_attempted": 1}, 60.0),
        ("aggregation", {"model_loads_completed": 1}, 60.0),
    ],
)
def test_scenario_substrate_class_10_invalid_class_never_lowers_legacy_floor(
    class_value: object,
    typed: dict[str, object],
    expected_floor: float,
) -> None:
    """SCENARIO-SUBSTRATE-CLASS-10 keeps malformed and contradictory claims strict."""

    payload: dict[str, object] = {
        "run_date": "20260913",
        "inference_substrate": "live_llm_inference",
        "duration_s": 30.0,
        **typed,
    }
    if class_value is not None:
        payload["inference_substrate_class"] = class_value
    floor = av.duration_floor_for_artifact(payload)
    assert floor is not None and floor["min_duration_s"] == expected_floor


def test_scenario_substrate_class_10_external_receipts_do_not_become_current_calls() -> None:
    """SCENARIO-SUBSTRATE-CLASS-10 keeps historical model fixtures in sidecars."""

    payload = {
        "run_date": "20260913",
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "duration_s": 0.01,
        "model_invoked": False,
        "historical_sidecar": {
            "scope": "historical",
            "model_invoked": True,
            "generation_calls_completed": 48,
        },
    }
    floor = av.duration_floor_for_artifact(payload)
    assert floor is not None and floor["min_duration_s"] == 0.0001
    flags: list[av.Flag] = []
    av.check_substrate_class(payload, flags)
    assert flags == []


def test_req_substrate_class_1_hermetic_exp7240_precondition_fixture(tmp_path: Path) -> None:
    """REQ-SUBSTRATE-CLASS-1 preserves the Exp7240 positive and negative assertions."""

    receipt = exp7261.build_hermetic_exp7240_fixture(tmp_path / "fixture")
    checks, hashes, upstream = exp7261.read_hermetic_exp7240_fixture(receipt)
    assert exp7261.exp7240.gate_summary(checks)["passed"] is True
    assert upstream["belief_run_complete_score"] == 1
    assert all(value and value.startswith("sha256:") for value in hashes.values())

    changed = deepcopy(upstream)
    changed["decision_rows_path"] = "not-a-receipt"
    with patch.object(exp7261.exp7240, "_load_object", return_value=changed):
        changed_checks, _, _ = exp7261.read_hermetic_exp7240_fixture(receipt)
    assert exp7261.exp7240.gate_summary(changed_checks)["passed"] is False


def test_scenario_substrate_class_9_actual_cli_and_summary_agree(tmp_path: Path) -> None:
    """SCENARIO-SUBSTRATE-CLASS-9 runs both public command-line readers end to end."""

    receipt = exp7261.run_floor_e2e(ROOT, tmp_path / "e2e")
    assert receipt["passed"] is True
    assert receipt["mismatch_count"] == 0
    assert {row["checker_floor_s"] for row in receipt["rows"]} == {2.0, 10.0, 60.0}
    assert all(row["checker_floor_s"] == row["summary_floor_s"] for row in receipt["rows"])


def test_req_substrate_class_1_artifact_and_mutations(tmp_path: Path) -> None:
    """REQ-SUBSTRATE-CLASS-1 publishes only a recomputable terminal null."""

    artifact = exp7261.build_artifact(
        ROOT,
        output_path=tmp_path / "result.json",
        raw_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "checkpoint.json",
        validation_receipts=exp7261.fixture_validation_receipts(),
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["compute_contract_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp7261.ZERO_INVOCATION_COUNTS
    assert artifact["rows"] == artifact["boundary_rows"]
    assert exp7261.validate_artifact(artifact, root=ROOT) == []
    assert json.loads((tmp_path / "result.json").read_text()) == artifact

    for field, value in (
        ("field_principles", None),
        ("schema", "bad"),
        ("status", "running"),
        ("run_date", "20260912"),
        ("model_invoked", True),
        ("MODEL_SPECS", [{"name": "forbidden"}]),
        ("inference_substrate_class", "aggregation"),
        ("execution_venue", "moon"),
        ("duration_s", 0.0),
        ("source_artifact_hashes", {"bad": "not-a-hash"}),
        ("rows", []),
        ("honest_verdict", "bad"),
        ("validation_receipts", []),
        ("fixture_repair_receipt", {}),
        ("e2e_floor_receipt", {}),
        ("raw_rows_receipt", {"path": "missing", "sha256": "sha256:missing"}),
        ("compute_contract_ready_score", 0),
        ("reproducibility_checksum", "sha256:forged"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert exp7261.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed.pop("field_principles")
    assert exp7261.validate_artifact(changed, root=ROOT)
    assert exp7261.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]

    with patch.object(
        exp7261, "_run_validations", return_value=exp7261.fixture_validation_receipts()
    ):
        internally_validated = exp7261.build_artifact(
            ROOT,
            output_path=tmp_path / "internally-validated.json",
            raw_dir=tmp_path / "internally-validated-raw",
            checkpoint_path=tmp_path / "internally-validated-checkpoint.json",
        )
    assert internally_validated["compute_contract_ready_score"] == 1


def test_req_substrate_class_1_missing_external_input_is_terminal_blocked(tmp_path: Path) -> None:
    """REQ-SUBSTRATE-CLASS-1 reports an absent prerequisite as blocked, never partial."""

    root = tmp_path / "repo"
    root.mkdir()
    artifact = exp7261.build_artifact(
        root,
        output_path=tmp_path / "blocked.json",
        raw_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "checkpoint.json",
        validation_receipts=[],
    )
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == "required_input"
    assert artifact["gate_check_summary"]["observed_value"] == "missing"
    assert exp7261.validate_artifact(artifact, root=root) == []
    changed = deepcopy(artifact)
    changed["rows"] = [{"unexpected": True}]
    changed["reproducibility_checksum"] = exp7261.reproducibility_checksum(changed)
    assert "blocked_contract_invalid" in exp7261.validate_artifact(changed, root=root)


def test_req_substrate_class_1_validation_scope_and_reducer_controls(tmp_path: Path) -> None:
    """REQ-SUBSTRATE-CLASS-1 keeps validation focused and reduction independent."""

    raw = tmp_path / "rows.json"
    raw.write_text("[]", encoding="utf-8")
    assert exp7261.independent_reduce(raw) == ["boundary_rows_missing"]
    rows = exp7261.evaluate_boundary_fixtures()
    rows[1]["fixture_id"] = rows[0]["fixture_id"]
    rows[2]["passed"] = False
    raw.write_text(json.dumps(rows), encoding="utf-8")
    assert exp7261.independent_reduce(raw) == [
        "fixture_ids_not_independent",
        "boundary_mismatch",
    ]

    commands = exp7261._validation_commands(ROOT, tmp_path / "candidate.json", raw)
    assert [name for name, _command in commands] == list(exp7261.VALIDATION_NAMES)
    joined = " ".join(shlex for _name, command in commands for shlex in command)
    assert "tests/python -q" not in joined
    assert "--fail-under=100" in joined
    assert exp7261._date(exp7261.RUN_DATE) == exp7261.RUN_DATE
    with pytest.raises(argparse.ArgumentTypeError, match="run date must be"):
        exp7261._date("20260912")


def test_req_substrate_class_1_thin_wrapper_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SUBSTRATE-CLASS-1 keeps the experiment-numbered entrypoint thin."""

    wrapper = ROOT / exp7261.WRAPPER_PATH
    monkeypatch.setattr(sys, "argv", [str(wrapper), "--date", exp7261.RUN_DATE])
    with patch.object(exp7261, "main", return_value=0) as delegated:
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(wrapper), run_name="__main__")
    assert stopped.value.code == 0
    delegated.assert_called_once_with()
