"""Tests for Exp5758 Rust parity scalar bridge.

Spec refs: REQ-REPORT-5758, REQ-SAMPLE-5758,
SCENARIO-REPORT-5758, SCENARIO-REPORT-5758-GATE-REPLAY,
SCENARIO-REPORT-5758-FIELD-PRINCIPLES, SCENARIO-SAMPLE-5758.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts.conductor_gates import evaluate_gates


REPO = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO / "scripts/experiment_5758_rust_parity_scalar_bridge.py"
REPORT_SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
SAMPLER_SPEC = REPO / "openspec/capabilities/samplers/spec.md"
UPSTREAM_PATH = REPO / "results/experiment_5751_rust_restart_parity_repair.json"

sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("experiment_5758_bridge", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
bridge = importlib.util.module_from_spec(_spec)
sys.modules["experiment_5758_bridge"] = bridge
_spec.loader.exec_module(bridge)  # type: ignore[union-attr]


def _run_bridge(tmp_path: Path) -> dict[str, Any]:
    return bridge.run(input_repo_root=REPO, output_repo_root=tmp_path, write=True)


def _write_bridge(results_dir: Path, artifact: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / bridge.RESULT_RELATIVE_PATH.name
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _gate_check(artifact: dict[str, Any], tmp_path: Path):
    results_dir = tmp_path / "results"
    _write_bridge(results_dir, artifact)
    return evaluate_gates(bridge.planned_exp5764_task(), results_dir)


def _write_upstream_copy(tmp_path: Path, payload: dict[str, Any], name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_req_5758_specs_declare_bridge_contract() -> None:
    """REQ-REPORT-5758 and REQ-SAMPLE-5758: OpenSpec anchors the bridge."""

    report = REPORT_SPEC.read_text(encoding="utf-8")
    sampler = SAMPLER_SPEC.read_text(encoding="utf-8")
    report_section = report[report.index("### REQ-REPORT-5758") :]
    sampler_section = sampler[sampler.index("### REQ-SAMPLE-5758") :]

    for marker in (
        "REQ-REPORT-5758",
        "SCENARIO-REPORT-5758-GATE-REPLAY",
        "results/experiment_5758_rust_parity_scalar_bridge.json",
        "`distributional_parity_score`",
        "`fallback_equivalence_score`",
        "`production_backend_reachable_score`",
        "`rust_benchmark_gate_ready_score`",
        "producer_normalizer_receipts",
    ):
        assert marker in report_section
    for marker in (
        "REQ-SAMPLE-5758",
        "SCENARIO-SAMPLE-5758",
        "`scripts/conductor_gates.evaluate_gates()`",
        "`timing_claimed=false`",
        "`hardware_speedup_claimed=false`",
    ):
        assert marker in sampler_section


def test_scenario_5758_builds_lossless_bridge_from_exp5751(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5758: hash-verified Exp5751 evidence emits bare scalars."""

    artifact = _run_bridge(tmp_path)
    output = tmp_path / bridge.RESULT_RELATIVE_PATH
    upstream = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert bridge.validate_artifact(artifact, input_repo_root=REPO) is True
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["spec_refs"] == list(bridge.SPEC_REFS)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["upstream_modified"] is False
    assert artifact["sampler_code_modified"] is False
    assert artifact["timing_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["inference_substrate"] == "cached_fixture_replay_no_llm"
    assert artifact["unsafe_synthesis_count"] == 0
    assert artifact["upstream_artifact_hash"] == "sha256:d69c72ac75c5ad3f67b0367d5c6643b1f2a1be3e0683db030b46d1c4d8344202"
    assert artifact["repair_source_hashes"]["python/carnot/samplers/one_axis_rust_backend.py"] == "2b953f14aeb57bb6f20af21484d8926f45e02ee30f2ae9af15b458d70645b488"
    assert artifact["reproduced_failure_receipt_hash"] == bridge.sha256_json(
        upstream["reproduced_failure_receipts"]
    )
    assert artifact["first_divergence_receipt_hash"] == bridge.sha256_json(
        upstream["first_divergence_receipt"]
    )
    assert artifact["interruption_manifest_hash"] == bridge.sha256_json(
        upstream["interruption_injection_manifest"]
    )
    assert artifact["parity_case_count"] == 3
    assert artifact["restart_parity_ready_score"] == pytest.approx(1.0)
    assert artifact["distributional_parity_score"] == pytest.approx(1.0)
    assert artifact["fallback_equivalence_score"] == pytest.approx(1.0)
    assert artifact["production_backend_reachable_score"] == pytest.approx(1.0)
    assert artifact["rust_benchmark_gate_ready_score"] == pytest.approx(1.0)
    assert artifact["producer_normalizer_receipts"]["ready_for_gated_consumers"] is True
    assert artifact["producer_normalizer_receipts"]["producer_gate_fields"] == list(
        bridge.PRODUCER_GATE_FIELDS
    )

    for field in bridge.PRODUCER_GATE_FIELDS:
        assert field in artifact
        assert not isinstance(artifact[field], dict)


def test_scenario_5758_replays_planned_exp5764_conductor_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5758-GATE-REPLAY: evaluate_gates passes on bare fields."""

    artifact = _run_bridge(tmp_path)
    gate_check = _gate_check(artifact, tmp_path)

    assert gate_check.passed is True
    assert gate_check.summary == "4 gate(s) satisfied"
    assert [gate.artifact_field for gate in gate_check.gates_evaluated] == list(
        bridge.EXP5764_GATE_FIELDS
    )
    assert [gate.actual for gate in gate_check.gates_evaluated] == [
        artifact[field] for field in bridge.EXP5764_GATE_FIELDS
    ]


def test_scenario_5758_gate_negative_controls_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5758-GATE-REPLAY: bad gate shapes cannot pass."""

    artifact = _run_bridge(tmp_path)

    missing = deepcopy(artifact)
    del missing["distributional_parity_score"]
    assert _gate_check(missing, tmp_path).passed is False
    with pytest.raises(bridge.BridgeValidationError, match="missing required fields"):
        bridge.validate_artifact(missing, input_repo_root=REPO)

    wrapped = deepcopy(artifact)
    wrapped["rust_benchmark_gate_ready_score"] = {
        "value": 1.0,
        "principle": "wrapper objects are not gate scalars",
    }
    assert _gate_check(wrapped, tmp_path).passed is False
    with pytest.raises(bridge.BridgeValidationError, match="rust_benchmark_gate_ready_score"):
        bridge.validate_artifact(wrapped, input_repo_root=REPO)

    false_positive = deepcopy(artifact)
    false_positive["fallback_equivalence_score"] = 0.0
    false_positive["rust_benchmark_gate_ready_score"] = 1.0
    false_positive["reproducibility_checksum"] = bridge.reproducibility_checksum(false_positive)
    assert _gate_check(false_positive, tmp_path).passed is False
    with pytest.raises(bridge.BridgeValidationError, match="rust_benchmark_gate_ready_score"):
        bridge.validate_artifact(false_positive, input_repo_root=REPO)


def test_scenario_5758_upstream_adversarial_controls_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5758: missing, wrapped, contradictory, and stale inputs block."""

    upstream = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))

    absent_predicate = deepcopy(upstream)
    del absent_predicate["distributional_parity"]["passed"]
    absent_path = _write_upstream_copy(tmp_path, absent_predicate, "absent.json")
    with pytest.raises(bridge.BridgeValidationError, match="distributional_parity.passed"):
        bridge.build_bridge(
            input_repo_root=REPO,
            output_repo_root=tmp_path,
            upstream_artifact_path=absent_path,
        )

    wrapped_predicate = deepcopy(upstream)
    wrapped_predicate["production_backend_reachable"]["passed"] = {
        "value": True,
        "principle": "nested wrapper must not be accepted as a predicate",
    }
    wrapped_path = _write_upstream_copy(tmp_path, wrapped_predicate, "wrapped.json")
    with pytest.raises(bridge.BridgeValidationError, match="object-wrapped"):
        bridge.build_bridge(
            input_repo_root=REPO,
            output_repo_root=tmp_path,
            upstream_artifact_path=wrapped_path,
        )

    contradictory_case = deepcopy(upstream)
    contradictory_case["distributional_parity"]["cases"][1]["energy_histogram_tv"] = 0.25
    contradictory_path = _write_upstream_copy(tmp_path, contradictory_case, "contradictory.json")
    with pytest.raises(bridge.BridgeValidationError, match="distributional_parity case"):
        bridge.build_bridge(
            input_repo_root=REPO,
            output_repo_root=tmp_path,
            upstream_artifact_path=contradictory_path,
        )

    source_hash_drift = deepcopy(upstream)
    source_hash_drift["upstream_artifact_hashes"]["source_hashes"][
        "python/carnot/samplers/one_axis_rust_backend.py"
    ] = "0" * 64
    drift_path = _write_upstream_copy(tmp_path, source_hash_drift, "drift.json")
    with pytest.raises(bridge.BridgeValidationError, match="source hash drift"):
        bridge.build_bridge(
            input_repo_root=REPO,
            output_repo_root=tmp_path,
            upstream_artifact_path=drift_path,
        )
