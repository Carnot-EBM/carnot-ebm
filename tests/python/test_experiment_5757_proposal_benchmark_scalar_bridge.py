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
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO / "scripts" / "experiment_5757_proposal_benchmark_scalar_bridge.py"
REPORT_SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
BENCH_SPEC = REPO / "openspec/capabilities/benchmarks/spec.md"
UPSTREAM_ARTIFACT = REPO / "results/experiment_5746_exact_proposal_utility_benchmark.json"
UPSTREAM_MANIFEST = (
    REPO / "results/experiment_5746_exact_proposal_utility_benchmark.instances.jsonl"
)
UPSTREAM_PREFLIGHT = (
    REPO / "results/experiment_5746_exact_proposal_utility_benchmark.preflight.json"
)

sys.path.insert(0, str(REPO))

from carnot import experiment_5746_exact_proposal_utility_benchmark as exp5746  # noqa: E402
from scripts.conductor_gates import evaluate_gates  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_5757_bridge", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["experiment_5757_bridge"] = mod
spec.loader.exec_module(mod)  # type: ignore[union-attr]

TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5757_proposal_benchmark_scalar_bridge.py "
    "tests/python/test_experiment_template.py -q"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --include=scripts/experiment_5757_proposal_benchmark_scalar_bridge.py "
    "-m pytest tests/python/test_experiment_5757_proposal_benchmark_scalar_bridge.py -q "
    "&& .venv/bin/coverage report "
    "--include=scripts/experiment_5757_proposal_benchmark_scalar_bridge.py --fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5757_proposal_benchmark_scalar_bridge.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _run_bridge(tmp_path: Path) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        upstream_artifact_path=UPSTREAM_ARTIFACT,
        benchmark_manifest_path=UPSTREAM_MANIFEST,
        upstream_preflight_path=UPSTREAM_PREFLIGHT,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def _write_gate_artifact(results_dir: Path, artifact: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / mod.RESULT_RELATIVE_PATH.name).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _roadmap_task(task_id: str) -> dict[str, Any]:
    roadmap = yaml.safe_load((REPO / "research-roadmap.yaml").read_text(encoding="utf-8"))
    for task in roadmap["tasks"]:
        if task["id"] == task_id:
            return task
    raise AssertionError(f"missing roadmap task {task_id}")


def test_req_5757_specs_declare_scalar_bridge_contract() -> None:
    """REQ-REPORT-5757, REQ-BENCH-5757: OpenSpec anchors the bridge contract."""

    report = REPORT_SPEC.read_text(encoding="utf-8")
    bench = BENCH_SPEC.read_text(encoding="utf-8")
    report_section = report[report.index("### REQ-REPORT-5757") :]
    bench_section = bench[bench.index("### REQ-BENCH-5757") : bench.index("### REQ-BENCH-3389")]

    for marker in (
        "REQ-REPORT-5757",
        "SCENARIO-REPORT-5757-GATE-REPLAY",
        str(mod.RESULT_RELATIVE_PATH),
        "ExperimentTemplate",
        "producer_gate_fields",
        "benchmark_bridge_ready_score",
        "cached_fixture_replay_no_llm",
    ):
        assert marker in report_section
    for marker in (
        "REQ-BENCH-5757",
        "SCENARIO-BENCH-5757-NEGATIVE-CONTROLS",
        "row hash",
        "split hashes",
        "adversarial controls",
    ):
        assert marker in bench_section


def test_scenario_5757_hash_verified_bridge_emits_bare_scalars(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5757, SCENARIO-BENCH-5757: canonical Exp5746 replays cleanly."""

    before_hash = mod.sha256_file(UPSTREAM_ARTIFACT)
    artifact = _run_bridge(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) is True
    assert mod.sha256_file(UPSTREAM_ARTIFACT) == before_hash
    assert artifact["status"] == "complete"
    assert artifact["upstream_modified"] is False
    assert artifact["upstream_artifact_hash"] == before_hash
    assert artifact["benchmark_manifest_hash"] == mod.sha256_file(UPSTREAM_MANIFEST)
    assert artifact["row_hash_count"] == 180
    assert artifact["benchmark_ready_score"] == pytest.approx(1.0)
    assert artifact["structure_receipt_failure_count"] == 0
    assert artifact["solution_receipt_failure_count"] == 0
    assert artifact["validator_disagreement_count"] == 0
    assert artifact["heldout_partition_disjoint_score"] == pytest.approx(1.0)
    assert artifact["adversarial_verification_clean_score"] == pytest.approx(1.0)
    assert artifact["benchmark_bridge_ready_score"] == pytest.approx(1.0)
    assert artifact["llm_inference_used"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == "cached_fixture_replay_no_llm"
    assert artifact["unsafe_synthesis_count"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(artifact) == set(artifact["field_principles"])
    for field in mod.PRODUCER_GATE_FIELDS:
        assert not isinstance(artifact[field], dict)
    assert artifact["producer_normalizer_receipts"]["ready_for_gated_consumers"] is True
    assert artifact["producer_normalizer_receipts"]["unsafe_rejections"] == []
    assert artifact["gate_replay_receipts"]["passed"] is True


def test_scenario_5757_replays_exact_planned_exp5759_gates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5757-GATE-REPLAY: bridge gates match active Exp5759 predicates."""

    artifact = _run_bridge(tmp_path)
    task = _roadmap_task("exp5759-sota-exact-proposal-utility-panel")

    assert mod.EXP5759_GATE_TASK["gated_on"] == task["gated_on"]

    results_dir = tmp_path / "results"
    _write_gate_artifact(results_dir, artifact)
    gate_check = evaluate_gates(mod.EXP5759_GATE_TASK, results_dir=results_dir)

    assert gate_check.passed is True
    assert [gate.artifact_field for gate in gate_check.gates_evaluated] == [
        "benchmark_bridge_ready_score",
        "structure_receipt_failure_count",
        "validator_disagreement_count",
    ]


def test_scenario_5757_gate_negative_controls_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-BENCH-5757-NEGATIVE-CONTROLS: missing, wrapped, and false gates fail."""

    artifact = _run_bridge(tmp_path)

    missing = deepcopy(artifact)
    del missing["benchmark_bridge_ready_score"]
    missing_dir = tmp_path / "missing"
    _write_gate_artifact(missing_dir, missing)
    missing_check = evaluate_gates(mod.EXP5759_GATE_TASK, results_dir=missing_dir)

    wrapped = deepcopy(artifact)
    wrapped["structure_receipt_failure_count"] = {"value": 0, "principle": "wrapped gate"}
    wrapped_dir = tmp_path / "wrapped"
    _write_gate_artifact(wrapped_dir, wrapped)
    wrapped_check = evaluate_gates(mod.EXP5759_GATE_TASK, results_dir=wrapped_dir)

    false_ready = deepcopy(artifact)
    false_ready["structure_receipt_failure_count"] = 1
    false_ready["benchmark_bridge_ready_score"] = 1.0
    false_dir = tmp_path / "false_ready"
    _write_gate_artifact(false_dir, false_ready)
    false_check = evaluate_gates(mod.EXP5759_GATE_TASK, results_dir=false_dir)

    assert missing_check.passed is False
    assert missing_check.gates_evaluated[0].actual is None
    assert wrapped_check.passed is False
    assert wrapped_check.gates_evaluated[1].actual == {"value": 0, "principle": "wrapped gate"}
    assert false_check.passed is False
    assert false_check.gates_evaluated[1].actual == 1


def test_scenario_5757_hash_and_false_readiness_validation_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5757-AMBIGUITY: altered hashes and false readiness are rejected."""

    artifact = _run_bridge(tmp_path)

    bad_hash = deepcopy(artifact)
    bad_hash["upstream_artifact_hash"] = "sha256:" + "0" * 64
    with pytest.raises(mod.BridgeBlockedError, match="upstream_artifact_hash"):
        mod.validate_artifact(bad_hash)

    false_ready = deepcopy(artifact)
    false_ready["benchmark_ready_score"] = 0.0
    false_ready["benchmark_bridge_ready_score"] = 1.0
    false_ready["honest_verdict"] = "complete: false readiness"
    false_ready["reproducibility_checksum"] = mod.reproducibility_checksum(false_ready)
    with pytest.raises(mod.BridgeBlockedError, match="benchmark_bridge_ready_score"):
        mod.validate_artifact(false_ready)


def test_scenario_5757_ambiguous_nested_value_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5757-AMBIGUITY: conflicting nested scalar evidence is not used."""

    upstream = json.loads(UPSTREAM_ARTIFACT.read_text(encoding="utf-8"))
    upstream["derivation_receipts"] = {"heldout_partition_disjoint_score": 0.0}
    upstream["field_principles"]["derivation_receipts"] = "fixture nested ambiguity"
    upstream["reproducibility_checksum"] = exp5746.reproducibility_checksum(upstream)
    mutated = tmp_path / UPSTREAM_ARTIFACT.name
    mutated.write_text(json.dumps(upstream, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(mod.BridgeBlockedError, match="ambiguous_nested_value"):
        mod.build_bridge_evidence(
            upstream_artifact_path=mutated,
            benchmark_manifest_path=UPSTREAM_MANIFEST,
            upstream_preflight_path=UPSTREAM_PREFLIGHT,
        )
