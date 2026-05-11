"""Tests for the Exp 1877 ROCE/HILED artifact contract normalization.

Spec: REQ-REPORT-1877, SCENARIO-REPORT-1877.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.artifact_contract_normalization import (
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    _extract_roce_success_rate,
    _read_json,
    build_artifact,
    main,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _roce_payload() -> dict[str, object]:
    return {
        "dataset_size": 20,
        "successes": 16,
        "success_rate": 0.8,
        "results": [
            {
                "input": "must contain 'apple'",
                "output": {
                    "model": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "extracted_constraints": [{"type": "contains", "value": "apple"}],
                    "success": True,
                },
            },
            {
                "input": "no constraints",
                "output": {
                    "model": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "extracted_constraints": [],
                    "success": False,
                },
            },
        ],
    }


def _hiled_payload() -> dict[str, object]:
    return {
        "efficiency_gains_ms": 2.0737648010253906,
        "constraint_enforcement_rate": 1.0,
        "hiled_enabled": True,
        "simulated_steps": 2,
    }


def test_scenario_report_1877_normalizes_roce_and_hiled_gate_fields() -> None:
    """SCENARIO-REPORT-1877: malformed source metrics become gate-ready wrappers."""

    artifact = build_artifact(
        sources={"roce": _roce_payload(), "hiled": _hiled_payload()},
        missing_source_paths=[],
        tests_run=["targeted smoke"],
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gate_contract_normalization_ready"] is True
    assert artifact["roce_success_rate"] == 0.8
    assert artifact["hiled_simulator_ready"] is True
    assert artifact["tests_run"] == ["targeted smoke"]

    wrappers = {row["source_experiment_id"]: row for row in artifact["normalized_artifacts"]}
    roce = wrappers["exp1864"]
    hiled = wrappers["exp1869"]
    assert roce["status"] == "complete"
    assert roce["honest_verdict"].startswith("complete:")
    assert roce["roce_success_rate"] == 0.8
    assert roce["raw_metrics"] == _roce_payload()
    assert "status" in roce["schema"]
    assert "honest_verdict" in roce["schema"]

    assert hiled["status"] == "complete"
    assert hiled["honest_verdict"].startswith("complete:")
    assert hiled["hiled_simulator_ready"] is True
    assert hiled["raw_metrics"] == _hiled_payload()
    assert "status" in hiled["schema"]
    assert "honest_verdict" in hiled["schema"]


def test_req_report_1877_blocks_when_a_source_artifact_is_missing() -> None:
    """REQ-REPORT-1877: missing source evidence prevents a ready gate contract."""

    artifact = build_artifact(
        sources={"roce": _roce_payload()},
        missing_source_paths=["results/experiment_1869_hiled.json"],
        tests_run=[],
    )

    assert artifact["status"] == "blocked"
    assert artifact["gate_contract_normalization_ready"] is False
    assert artifact["roce_success_rate"] == 0.8
    assert artifact["hiled_simulator_ready"] is False
    assert "listed source artifacts are missing" in artifact["blocked_reasons"]
    assert artifact["source_inputs_read"]["results/experiment_1869_hiled.json"]["exists"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_req_report_1877_run_writes_terminal_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-1877: run writes in-progress and terminal JSON artifacts."""

    out_path = tmp_path / "results" / "experiment_1877_artifact_contract_normalization.json"
    bootstrap = write_in_progress_artifact(out_path)
    assert bootstrap["status"] == "in_progress"
    assert json.loads(out_path.read_text(encoding="utf-8"))["status"] == "in_progress"

    _write_json(tmp_path / "results" / SOURCE_FILES["roce"], _roce_payload())
    _write_json(tmp_path / "results" / SOURCE_FILES["hiled"], _hiled_payload())

    artifact = run(root=tmp_path, out_path=out_path, tests_run=["pytest normalization"])
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["source_inputs_read"]["results/experiment_1864_roce.json"]["exists"] is True
    assert written["source_inputs_read"]["results/experiment_1869_hiled.json"]["exists"] is True


def test_req_report_1877_helpers_keep_malformed_inputs_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-1877: helpers preserve absent and derived metric states."""

    assert _read_json(tmp_path / "missing.json") is None
    derived = {"dataset_size": 4, "successes": 3}
    assert _extract_roce_success_rate(derived) == 0.75
    assert _extract_roce_success_rate({"dataset_size": 0, "successes": 3}) is None

    artifact = build_artifact(
        sources={"roce": derived, "hiled": {"hiled_enabled": False}},
        missing_source_paths=[],
        tests_run=[],
    )

    assert artifact["status"] == "blocked"
    assert artifact["roce_success_rate"] == 0.75
    assert artifact["hiled_simulator_ready"] is False
    assert "hiled simulator gate is not ready" in artifact["blocked_reasons"]

    no_roce_rate = build_artifact(
        sources={"roce": {"dataset_size": 0, "successes": 3}, "hiled": _hiled_payload()},
        missing_source_paths=[],
        tests_run=[],
    )
    assert "roce success-rate gate is not numeric" in no_roce_rate["blocked_reasons"]

    out_path = tmp_path / "results" / "experiment_1877_artifact_contract_normalization.json"
    assert main(["--root", str(tmp_path), "--out", str(out_path), "--tests-run", "cli smoke"]) == 0
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["status"] == "blocked"
    assert written["tests_run"] == ["cli smoke"]
    assert written["source_inputs_read"]["results/experiment_1864_roce.json"]["exists"] is False
