"""Tests for the V644 in-process acquired-constraint binding.

Spec refs: REQ-VERIFY-7339, SCENARIO-VERIFY-7339-*, REQ-PYBIND-7339,
and SCENARIO-PYBIND-7339-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot import experiment_7326_v643_constraint_kernel as exp7326
from carnot import experiment_7339_v644_native_binding as exp7339


ROOT = Path(__file__).resolve().parents[2]


class _FakeCompiledScheduleEvaluator:
    """Model the planned Rust ownership boundary without hiding Python control use."""

    def __init__(self, constraints: Any) -> None:
        self.constraints = deepcopy(constraints)

    def evaluate_batch(self, requests: Any) -> list[dict[str, Any]]:
        full = []
        for request in requests:
            restored = deepcopy(request)
            restored["constraints"] = deepcopy(self.constraints)
            full.append(restored)
        return exp7326.evaluate_batch(full)


FAKE_BINDING = SimpleNamespace(RustCompiledScheduleEvaluator=_FakeCompiledScheduleEvaluator)


def _request() -> dict[str, Any]:
    return {
        "schema": exp7326.REQUEST_SCHEMA,
        "executor_version": "executor-v1",
        "slot_min": 0,
        "slot_max": 3,
        "schedule": [
            {"activity": "a", "slot": 0},
            {"activity": "b", "slot": 1},
            {"activity": "c", "slot": 1},
        ],
        "constraints": [
            {
                "kind": "pairwise_separation",
                "constraint_id": "sep",
                "version": "executor-v1",
                "left": "a",
                "right": "b",
                "minimum": 2,
            },
            {
                "kind": "sliding_window_capacity",
                "constraint_id": "cap",
                "version": "executor-v1",
                "window_size": 2,
                "maximum": 1,
            },
        ],
    }


def test_req_verify_7339_authenticates_exact_v643_inputs(tmp_path: Path) -> None:
    """REQ-VERIFY-7339: exact retained inputs pass and absent copies fail closed."""

    checks, hashes = exp7339.collect_preconditions(ROOT)
    summary = exp7339.gate_check_summary(checks)
    assert summary["passed"] is True
    assert hashes["experiment_7325"] == exp7339.EXPECTED_EXP7325_SHA256
    assert hashes["experiment_7326"] == exp7339.EXPECTED_EXP7326_SHA256
    assert hashes["parity_fixtures"] == exp7339.EXPECTED_FIXTURE_SHA256

    missing_checks, _ = exp7339.collect_preconditions(tmp_path)
    missing = exp7339.gate_check_summary(missing_checks)
    assert missing["passed"] is False
    assert missing["first_failure"]["check"] == "exp7325_available"
    assert missing["first_failure"]["observed_value"] is False


def test_req_pbind_7339_compacts_and_groups_without_aliasing() -> None:
    """REQ-PYBIND-7339: compiled terms leave a detached compact request batch."""

    request = _request()
    compact = exp7339.compact_request(request)
    assert "constraints" not in compact
    assert compact["schedule"] == request["schedule"]
    compact["schedule"][0]["slot"] = 3
    assert request["schedule"][0]["slot"] == 0

    fixtures = [
        {"fixture_id": "one", "source": "test", "case": "valid", "request": request},
        {
            "fixture_id": "two",
            "source": "test",
            "case": "valid",
            "request": deepcopy(request),
        },
    ]
    groups = exp7339.group_fixtures(fixtures)
    assert len(groups) == 1
    assert groups[0]["constraints"] == request["constraints"]
    assert [row["fixture_id"] for row in groups[0]["fixtures"]] == ["one", "two"]


def test_scenario_pbind_7339_parity_preserves_errors_order_and_overflow() -> None:
    """SCENARIO-PYBIND-7339-ROUNDTRIP: all fixed adverse outputs match exactly."""

    fixtures = [
        {"fixture_id": "base", "source": "test", "case": "valid", "request": _request()},
        *exp7339.adverse_fixtures(),
    ]
    rows = exp7339.native_parity(FAKE_BINDING, fixtures, progress_every=2)
    reduced = exp7339.reduce_parity_rows(rows)
    assert reduced == {"rows": len(fixtures), "mismatches": 0, "all_matched": True}
    assert all(row["python"] == row["rust"] for row in rows)
    overflow = next(row for row in rows if row["case"] == "term_energy_overflow")
    assert overflow["rust"]["error"] == "energy_overflow:overflow-sep"
    total = next(row for row in rows if row["case"] == "total_energy_overflow")
    assert total["rust"]["error"] == "total_energy_overflow:overflow-total-b"
    assert rows[0]["rust"]["terms"][0]["constraint_id"] == "sep"


def test_scenario_pbind_7339_mutation_checks_owned_constraints_and_results() -> None:
    """SCENARIO-PYBIND-7339-MUTATION: inputs and outputs cannot alias native state."""

    rows = exp7339.mutation_parity(FAKE_BINDING)
    assert [row["case"] for row in rows] == [
        "constructor_input_mutation",
        "returned_result_mutation",
    ]
    assert all(row["matched"] for row in rows)
    assert rows[0]["rust"] == rows[0]["python"]
    assert rows[1]["rust"] == rows[1]["python"]


def test_scenario_pbind_7339_e2e003_uses_actual_imported_extension() -> None:
    """SCENARIO-PYBIND-7339-E2E003: runtime tests cross the compiled extension."""

    extension_value = os.environ.get("CARNOT_EXP7339_EXTENSION")
    assert extension_value is not None, "CARNOT_EXP7339_EXTENSION must name the task build"
    extension = Path(extension_value)
    assert extension.is_file()
    binding = exp7339.load_native_extension(extension)
    assert Path(binding.__file__).resolve() == extension.resolve()
    rows = exp7339.native_parity(
        binding,
        [{"fixture_id": "e2e", "source": "test", "case": "valid", "request": _request()}],
    )
    assert rows[0]["matched"] is True
    assert rows[0]["rust"]["terms"] == rows[0]["python"]["terms"]


def test_scenario_verify_7339_cost_reducer_requires_every_arm_and_block() -> None:
    """SCENARIO-VERIFY-7339-PROTOCOL: the sealed cost denominator cannot shrink."""

    rows = exp7339.synthetic_cost_rows()
    summary = exp7339.reduce_cost_rows(rows)
    assert summary["complete"] is True
    assert summary["paired_blocks"] == 90
    assert summary["row_count"] == 270
    assert summary["mismatches"] == 0
    assert set(summary["by_batch_size"]) == {"1", "32", "256"}

    rows.pop()
    incomplete = exp7339.reduce_cost_rows(rows)
    assert incomplete["complete"] is False
    assert incomplete["row_count"] == 269


def test_req_verify_7339_terminal_validator_rejects_tampering(tmp_path: Path) -> None:
    """REQ-VERIFY-7339: readiness needs exact runtime evidence and complete receipts."""

    artifact = exp7339.complete_artifact_fixture_for_test(tmp_path)
    assert exp7339.validate_artifact(artifact, check_files=True) == []

    wrong = deepcopy(artifact)
    wrong["native_binding_ready_score"] = 0
    assert "native_binding_ready_score" in exp7339.validate_artifact(wrong, check_files=True)

    wrong = deepcopy(artifact)
    wrong["binding_identity"]["compiled_execution"] = False
    wrong["reproducibility_checksum"] = exp7339.reproducibility_checksum(wrong)
    errors = exp7339.validate_artifact(wrong, check_files=True)
    assert "binding_compiled_execution" in errors

    blocked = exp7339.blocked_artifact_fixture_for_test()
    assert exp7339.validate_artifact(blocked) == []
    blocked["rows"] = [{"not": "allowed"}]
    blocked["reproducibility_checksum"] = exp7339.reproducibility_checksum(blocked)
    assert "blocked_rows" in exp7339.validate_artifact(blocked)


def test_req_verify_7339_evidence_helpers_and_defensive_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-VERIFY-7339: sidecar, identity, and defensive helper paths stay testable."""

    object_path = tmp_path / "object.json"
    object_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not_json_object"):
        exp7339.read_object(object_path)
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_jsonl_row"):
        exp7339.read_jsonl(jsonl_path)

    exp7339._atomic_jsonl(jsonl_path, [{"row": 1}])
    assert exp7339.read_jsonl(jsonl_path) == [{"row": 1}]
    broken = exp7339.complete_artifact_fixture_for_test(tmp_path)
    broken["schema"] = "wrong"
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp7339.write_artifact(tmp_path / "broken.json", broken)

    spans: list[dict[str, Any]] = []
    monkeypatch.setattr(exp7339.time, "monotonic", lambda: 5.0)
    exp7339._phase(spans, "fixture", 4.0, 3.0, 2)
    assert spans == [
        {
            "phase": "fixture",
            "start_s": 1.0,
            "end_s": 2.0,
            "completed_units": 2,
            "checkpoint_positions": [2],
            "pending_operations": [],
        }
    ]
    assert "phase=fixture event=end" in capsys.readouterr().out

    hashes = exp7339._current_source_hashes(ROOT)
    assert hashes[exp7339.RUST_BINDING_PATH.as_posix()].startswith("sha256:")
    suffix = exp7339._extension_destination(ROOT)
    assert suffix.name.startswith("_rust.")
    monkeypatch.setattr(exp7339.sysconfig, "get_config_var", lambda _name: None)
    with pytest.raises(RuntimeError, match="extension suffix"):
        exp7339._extension_destination(ROOT)

    extension = tmp_path / "native.so"
    extension.write_bytes(b"native")
    monkeypatch.undo()
    identity = exp7339._binding_identity(
        ROOT, extension, SimpleNamespace(__file__=str(extension)), {"command": "cargo"}
    )
    assert identity["module_sha256"] == exp7339.sha256_file(extension)
    assert identity["rust_source_identity"][exp7339.RUST_SCHEDULE_PATH.as_posix()].startswith(
        "sha256:"
    )

    selected, constraints = exp7339._benchmark_requests(
        [{"fixture_id": "one", "source": "test", "case": "valid", "request": _request()}]
    )
    assert len(selected) == 1
    assert constraints == _request()["constraints"]
    with pytest.raises(RuntimeError, match="no_valid_benchmark_requests"):
        exp7339._benchmark_requests([])

    receipt = exp7339._validation_receipt("fixture", "command", "scope", 0.1, jsonl_path, True)
    assert receipt["exit_code"] == 0
    assert receipt["log_sha256"] == exp7339.sha256_file(jsonl_path)

    checksum_artifact = exp7339.complete_artifact_fixture_for_test(tmp_path)
    monkeypatch.setattr(
        exp7339,
        "reproducibility_checksum",
        lambda _artifact: (_ for _ in ()).throw(TypeError("bad")),
    )
    assert "reproducibility_checksum" in exp7339.validate_artifact(checksum_artifact)


def test_scenario_verify_7339_atomic_write_and_main_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-VERIFY-7339-TERMINAL: publication follows cold validation."""

    artifact = exp7339.complete_artifact_fixture_for_test(tmp_path)
    output = tmp_path / "result.json"
    receipt = exp7339.write_artifact(output, artifact, check_files=True)
    assert receipt["sha256"] == exp7339.sha256_file(output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"

    monkeypatch.setattr(exp7339, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(exp7339, "run_experiment", lambda _root: artifact)
    monkeypatch.setattr(exp7339, "write_artifact", lambda *_args, **_kwargs: receipt)
    monkeypatch.setattr(
        exp7339,
        "_terminal_validators",
        lambda *_args: [{"name": "validator", "passed": True}],
    )
    assert exp7339.main(["--date", exp7339.RUN_DATE]) == 0
    assert "phase=terminal_write event=end" in capsys.readouterr().out

    failed = exp7339.complete_artifact_fixture_for_test(tmp_path)
    monkeypatch.setattr(exp7339, "run_experiment", lambda _root: failed)
    monkeypatch.setattr(
        exp7339,
        "_terminal_validators",
        lambda *_args: [{"name": "validator", "passed": False}],
    )
    assert exp7339.main(["--date", exp7339.RUN_DATE]) == 1
    assert failed["native_binding_ready_score"] == 0

    with pytest.raises(SystemExit, match="--date 20260916 is required"):
        exp7339.main([])
