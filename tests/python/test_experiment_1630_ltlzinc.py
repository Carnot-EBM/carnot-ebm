"""Tests for Exp 1630 LTLZinc temporal retention benchmark.

Spec: REQ-LEARN-1630, SCENARIO-LEARN-1630.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_1630_ltlzinc as mod


def test_req_learn_1630_generates_case_memory_records() -> None:
    """REQ-LEARN-1630-2/3: generated rows map to deterministic CaseRecords."""

    cases = mod.generate_benchmark_cases()

    assert len(cases) == mod.DEFAULT_BENCHMARK_SIZE
    assert {str(case["temporal_operator"]) for case in cases} == set(mod.SUPPORTED_OPERATORS)
    for case in cases:
        mod.validate_case_schema(case)
        assert mod.verify_temporal_case(case) is bool(case["expected_satisfied"])
        record = mod.case_to_record(case)
        assert record.benchmark == mod.BENCHMARK_NAME
        assert record.benchmark_slice == (
            f"{mod.BENCHMARK_SLICE_PREFIX}:{case['temporal_operator']}"
        )
        assert record.provenance.source_experiment == mod.EXPERIMENT_ID
        assert record.provenance.case_id == case["case_id"]
        assert "temporal" in record.violation_families


def test_scenario_learn_1630_retains_constraints_after_updates() -> None:
    """SCENARIO-LEARN-1630: later update rows do not forget old constraints."""

    cases = mod.generate_benchmark_cases()
    update_cases = mod.generate_update_cases()
    artifact = mod.build_artifact(
        benchmark_cases=cases,
        update_cases=update_cases,
        project_root="/repo",
        run_date="20260509",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["benchmark_size"] == len(cases)
    assert artifact["retained_case_count"] == len(cases)
    assert artifact["pass_rate"] == pytest.approx(1.0)
    assert artifact["memory_entry_count"] >= len(cases) + len(update_cases)
    assert artifact["honest_verdict"] == "ltlzinc_temporal_retention_benchmark_passed"
    assert all(result["retained"] is True for result in artifact["case_results"])
    assert {
        result["temporal_operator"] for result in artifact["case_results"]
    } == set(mod.SUPPORTED_OPERATORS)


def test_req_learn_1630_run_writes_json_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-1630-1/4: run writes benchmark_size and pass_rate."""

    output_path = tmp_path / "results" / mod.OUTPUT_FILE

    artifact = mod.run_experiment(
        output_path=output_path,
        project_root=tmp_path,
        run_date="20260509",
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["benchmark_size"] == mod.DEFAULT_BENCHMARK_SIZE
    assert artifact["pass_rate"] == pytest.approx(1.0)
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)


def test_req_learn_1630_blocks_mislabeled_temporal_case() -> None:
    """REQ-LEARN-1630-3/5: verifier disagreement prevents complete status."""

    cases = mod.generate_benchmark_cases()
    mislabeled = dict(cases[0], expected_satisfied=not bool(cases[0]["expected_satisfied"]))

    artifact = mod.build_artifact(
        benchmark_cases=[mislabeled],
        update_cases=(),
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["benchmark_size"] == 1
    assert artifact["retained_case_count"] == 0
    assert artifact["pass_rate"] == pytest.approx(0.0)
    assert artifact["case_results"][0]["local_verifier_matches_expected"] is False
    assert artifact["case_results"][0]["retained"] is False


def test_req_learn_1630_validation_rejects_bad_artifacts() -> None:
    """REQ-LEARN-1630-4/5: artifact validation enforces result consistency."""

    artifact = mod.build_artifact(
        benchmark_cases=mod.generate_benchmark_cases(),
        update_cases=mod.generate_update_cases(),
        project_root="/repo",
    )

    missing = dict(artifact)
    del missing["benchmark_size"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="unsupported schema"):
        mod.validate_artifact(dict(artifact, schema="wrong"))

    with pytest.raises(AssertionError, match="pass_rate"):
        mod.validate_artifact(dict(artifact, pass_rate=1.5))

    with pytest.raises(AssertionError, match="retained_case_count"):
        mod.validate_artifact(dict(artifact, retained_case_count=0))

    with pytest.raises(AssertionError, match="complete artifact"):
        mod.validate_artifact(dict(artifact, status="complete", pass_rate=0.5))
