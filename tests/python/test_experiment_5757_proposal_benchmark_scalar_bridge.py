"""Tests for Exp5757 proposal benchmark scalar bridge.

Spec refs: REQ-REPORT-5757, REQ-BENCH-5757,
SCENARIO-REPORT-5757, SCENARIO-REPORT-5757-GATE-REPLAY,
SCENARIO-REPORT-5757-AMBIGUITY, SCENARIO-REPORT-5757-FIELD-PRINCIPLES,
SCENARIO-BENCH-5757, SCENARIO-BENCH-5757-NEGATIVE-CONTROLS.
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
SCRIPT_PATH = REPO / "scripts/experiment_5757_proposal_benchmark_scalar_bridge.py"
REPORT_SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
BENCH_SPEC = REPO / "openspec/capabilities/benchmarks/spec.md"

sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("experiment_5757_bridge", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
bridge = importlib.util.module_from_spec(_spec)
sys.modules["experiment_5757_bridge"] = bridge
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
    return evaluate_gates(bridge.planned_exp5759_task(), results_dir)


def test_req_5757_specs_declare_bridge_contract() -> None:
    """REQ-REPORT-5757 and REQ-BENCH-5757: OpenSpec anchors the bridge."""

    report = REPORT_SPEC.read_text(encoding="utf-8")
    bench = BENCH_SPEC.read_text(encoding="utf-8")
    report_section = report[report.index("### REQ-REPORT-5757") :]
    bench_section = bench[bench.index("### REQ-BENCH-5757") :]

    for marker in (
        "REQ-REPORT-5757",
        "SCENARIO-REPORT-5757-GATE-REPLAY",
        "SCENARIO-REPORT-5757-AMBIGUITY",
        "results/experiment_5757_proposal_benchmark_scalar_bridge.json",
        "producer_gate_fields",
        "`benchmark_bridge_ready_score`",
        "`heldout_partition_disjoint_score`",
        "`adversarial_verification_clean_score`",
    ):
        assert marker in report_section
    for marker in (
        "REQ-BENCH-5757",
        "SCENARIO-BENCH-5757-NEGATIVE-CONTROLS",
        "`scripts/conductor_gates.evaluate_gates()`",
        "`benchmark_ready_score`",
        "`solution_receipt_failure_count`",
    ):
        assert marker in bench_section


def test_scenario_5757_builds_lossless_bridge_from_sealed_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5757: sealed Exp5746 evidence emits bare bridge scalars."""

    artifact = _run_bridge(tmp_path)
    output = tmp_path / bridge.RESULT_RELATIVE_PATH

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert bridge.validate_artifact(artifact, input_repo_root=REPO) is True
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["upstream_modified"] is False
    assert artifact["llm_inference_used"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == "cached_fixture_replay_no_llm"
    assert artifact["unsafe_synthesis_count"] == 0
    assert artifact["row_hash_count"] == 180
    assert artifact["benchmark_manifest_hash"] == "sha256:071e0bbe2b7498745a8ae52b20326af129f0ffb603aeea62220a10c3de57e17c"
    assert artifact["benchmark_ready_score"] == pytest.approx(1.0)
    assert artifact["structure_receipt_failure_count"] == 0
    assert artifact["solution_receipt_failure_count"] == 0
    assert artifact["validator_disagreement_count"] == 0
    assert artifact["heldout_partition_disjoint_score"] == pytest.approx(1.0)
    assert artifact["adversarial_verification_clean_score"] == pytest.approx(1.0)
    assert artifact["benchmark_bridge_ready_score"] == pytest.approx(1.0)
    assert artifact["producer_normalizer_receipts"]["ready_for_gated_consumers"] is True

    for field in bridge.PRODUCER_GATE_FIELDS:
        assert field in artifact
        assert not isinstance(artifact[field], dict)


def test_scenario_5757_replays_planned_exp5759_conductor_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5757-GATE-REPLAY: evaluate_gates passes on bare fields."""

    artifact = _run_bridge(tmp_path)
    gate_check = _gate_check(artifact, tmp_path)

    assert gate_check.passed is True
    assert gate_check.summary == "7 gate(s) satisfied"
    assert [gate.artifact_field for gate in gate_check.gates_evaluated] == list(
        bridge.PRODUCER_GATE_FIELDS
    )
    assert [gate.actual for gate in gate_check.gates_evaluated] == [
        artifact[field] for field in bridge.PRODUCER_GATE_FIELDS
    ]


def test_scenario_5757_gate_negative_controls_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5757-NEGATIVE-CONTROLS: bad gate shapes cannot pass."""

    artifact = _run_bridge(tmp_path)

    missing = deepcopy(artifact)
    del missing["benchmark_ready_score"]
    assert _gate_check(missing, tmp_path).passed is False

    wrapped = deepcopy(artifact)
    wrapped["benchmark_bridge_ready_score"] = {
        "value": 1.0,
        "principle": "wrapper objects are not gate scalars",
    }
    assert _gate_check(wrapped, tmp_path).passed is False

    false_ready = deepcopy(artifact)
    false_ready["solution_receipt_failure_count"] = 1
    false_ready["benchmark_bridge_ready_score"] = 1.0
    assert _gate_check(false_ready, tmp_path).passed is False
    with pytest.raises(bridge.BridgeValidationError, match="benchmark_bridge_ready_score"):
        bridge.validate_artifact(false_ready, input_repo_root=REPO)


def test_scenario_5757_hash_and_ambiguity_controls_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5757-AMBIGUITY: hash drift and ambiguous values block."""

    artifact = _run_bridge(tmp_path)

    altered_upstream_hash = deepcopy(artifact)
    altered_upstream_hash["upstream_artifact_hash"] = "sha256:" + "0" * 64
    altered_upstream_hash["reproducibility_checksum"] = bridge.reproducibility_checksum(
        altered_upstream_hash
    )
    with pytest.raises(bridge.BridgeValidationError, match="upstream_artifact_hash"):
        bridge.validate_artifact(altered_upstream_hash, input_repo_root=REPO)

    nested_source = {
        "receipt": {
            "benchmark_ready_score": {
                "value": 1.0,
                "principle": "single nested scalar fixture",
            }
        }
    }
    assert bridge.derive_required_scalar(nested_source, "benchmark_ready_score") == pytest.approx(
        1.0
    )

    ambiguous_source = {
        "nested_a": {"benchmark_ready_score": 1.0},
        "nested_b": {"benchmark_ready_score": 0.0},
    }
    with pytest.raises(bridge.BridgeValidationError, match="ambiguous"):
        bridge.derive_required_scalar(ambiguous_source, "benchmark_ready_score")

    manifest_lines = (
        REPO / "results/experiment_5746_exact_proposal_utility_benchmark.instances.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    first_row = json.loads(manifest_lines[0])
    first_row["row_hash"] = "sha256:" + "1" * 64
    manifest_lines[0] = json.dumps(first_row, sort_keys=True, ensure_ascii=True)
    bad_manifest = tmp_path / "bad.instances.jsonl"
    bad_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    upstream_copy = json.loads(
        (REPO / "results/experiment_5746_exact_proposal_utility_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    upstream_copy["benchmark_manifest_path"] = str(bad_manifest)
    upstream_copy["benchmark_manifest_hash"] = bridge.sha256_file(bad_manifest)
    upstream_copy["reproducibility_checksum"] = bridge.exp5746.reproducibility_checksum(
        upstream_copy
    )
    bad_upstream = tmp_path / "bad.upstream.json"
    bad_upstream.write_text(
        json.dumps(upstream_copy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(bridge.BridgeValidationError, match="row_hash"):
        bridge.build_bridge(
            input_repo_root=REPO,
            output_repo_root=tmp_path,
            upstream_artifact_path=bad_upstream,
            benchmark_manifest_path=bad_manifest,
        )
