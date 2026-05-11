"""Tests for Exp 1752 LTLZinc spatial benchmark expansion.

Spec: REQ-LEARN-1752, SCENARIO-LEARN-1752.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from scripts import experiment_1752_expand_ltlzinc as mod


def test_req_learn_1752_builds_stable_spatial_benchmark_schema() -> None:
    """REQ-LEARN-1752-2/4: benchmark JSON exposes 100 balanced spatial cases."""

    benchmark = mod.build_spatial_benchmark()

    mod.validate_spatial_benchmark(benchmark)
    assert benchmark["schema"] == mod.SCHEMA
    assert benchmark["benchmark_id"] == mod.BENCHMARK_ID
    assert benchmark["source"]["base_benchmark_path"] == "data/ltlzinc_benchmark.json"
    assert benchmark["source"]["base_benchmark_id"] == "ltlzinc_temporal_nonforgetting_v1"
    assert benchmark["case_count"] == len(benchmark["cases"]) == 100
    assert benchmark["map_count"] == 50
    assert benchmark["sat_case_count"] == 50
    assert benchmark["repair_hint_case_count"] == 50
    assert set(benchmark["supported_spatial_families"]) == set(mod.SPATIAL_FAMILIES)


def test_req_learn_1752_cases_are_verifiable_and_retention_ready() -> None:
    """REQ-LEARN-1752-3/6: every spatial case is locally verifiable."""

    benchmark = mod.build_spatial_benchmark()
    cases = benchmark["cases"]
    case_ids = {case["case_id"] for case in cases}

    assert len(case_ids) == len(cases)
    family_states: dict[str, set[str]] = {}
    for case in cases:
        mod.validate_spatial_case_schema(case)
        assert mod.verify_spatial_case(case) is bool(case["expected_satisfied"])
        assert case["evaluation"]["verifier_path"] == mod.VERIFIER_PATH
        assert case["source_benchmark"]["benchmark_id"] == benchmark["source"]["base_benchmark_id"]
        assert case["retention"]["phase"] == "spatial"
        assert case["retention"]["must_retrieve_after_updates"] is True
        assert case["route"][0] == case["start"]
        assert case["route"][-1] == case["goal"]
        assert case["topological_map"]["nodes"]
        assert case["topological_map"]["edges"]
        assert {"ltlzinc", "spatial", "topological-map"}.issubset(case["tags"])
        family_states.setdefault(str(case["spatial_family"]), set()).add(
            str(case["certificate_state"])
        )

    assert set(family_states) == set(mod.SPATIAL_FAMILIES)
    assert all(states == {"SAT", "REPAIR_HINT"} for states in family_states.values())


def test_req_learn_1752_write_spatial_benchmark_json(tmp_path: Path) -> None:
    """REQ-LEARN-1752-2: write_spatial_benchmark writes deterministic JSON."""

    output_path = tmp_path / "data" / "ltlzinc_spatial_benchmark.json"

    benchmark = mod.write_spatial_benchmark(output_path=output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == benchmark
    assert benchmark == mod.build_spatial_benchmark()


def test_req_learn_1752_run_writes_terminal_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-1752-1/5: run_experiment writes the terminal artifact."""

    benchmark_path = tmp_path / "data" / "ltlzinc_spatial_benchmark.json"
    output_path = tmp_path / "results" / "experiment_1752_spatial.json"

    artifact = mod.run_experiment(
        output_path=output_path,
        benchmark_path=benchmark_path,
        project_root=tmp_path,
        commands_run=["pytest targeted"],
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert benchmark_path.exists()
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["schema"] == mod.ARTIFACT_SCHEMA
    assert artifact["experiment_id"] == 1752
    assert artifact["benchmark_path"] == str(benchmark_path)
    assert artifact["spatial_case_count"] == 100
    assert artifact["validated_case_count"] == 100
    assert artifact["sat_case_count"] == 50
    assert artifact["repair_hint_case_count"] == 50
    assert artifact["commands_run"] == ["pytest targeted"]
    assert artifact["honest_verdict"] == "complete: ltlzinc_spatial_benchmark_ready"


def test_req_learn_1752_external_base_benchmark_path_is_preserved(tmp_path: Path) -> None:
    """REQ-LEARN-1752-3: non-repo source benchmark paths remain auditable."""

    base_path = tmp_path / "ltlzinc_benchmark.json"
    base_path.write_text(
        json.dumps(
            {
                "schema": "external.ltlzinc.v1",
                "benchmark_id": "external_spatial_seed",
                "case_count": 7,
                "spec": ["REQ-EXTERNAL"],
            }
        ),
        encoding="utf-8",
    )

    benchmark = mod.build_spatial_benchmark(base_benchmark_path=base_path)

    assert benchmark["source"]["base_benchmark_path"] == str(base_path)
    assert benchmark["source"]["base_benchmark_id"] == "external_spatial_seed"
    assert {case["source_benchmark"]["case_count"] for case in benchmark["cases"]} == {7}


def test_req_learn_1752_direct_script_bootstrap_adds_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-1752-1: direct execution can import repo-local modules."""

    repo_root = str(mod.REPO_ROOT)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != repo_root])

    namespace = runpy.run_path(str(mod.REPO_ROOT / "scripts" / "experiment_1752_expand_ltlzinc.py"))

    assert sys.path[0] == repo_root
    assert namespace["BENCHMARK_ID"] == mod.BENCHMARK_ID


def test_req_learn_1752_validation_rejects_bad_payloads() -> None:
    """REQ-LEARN-1752-6: validation protects counts and verifier agreement."""

    benchmark = mod.build_spatial_benchmark()

    with pytest.raises(AssertionError, match="case_count"):
        mod.validate_spatial_benchmark(dict(benchmark, case_count=99))

    bad_case = json.loads(json.dumps(benchmark["cases"][0]))
    del bad_case["topological_map"]
    with pytest.raises(AssertionError, match="missing spatial case fields"):
        mod.validate_spatial_case_schema(bad_case)

    mislabeled_case = json.loads(json.dumps(benchmark["cases"][0]))
    mislabeled_case["expected_satisfied"] = not bool(mislabeled_case["expected_satisfied"])
    mislabeled_case["evaluation"]["expected_verifier_result"] = bool(
        mislabeled_case["expected_satisfied"]
    )
    with pytest.raises(AssertionError, match="spatial verifier disagrees"):
        mod.validate_spatial_case_schema(mislabeled_case)

    artifact = mod.build_artifact(
        benchmark=benchmark,
        benchmark_path=mod.DEFAULT_SPATIAL_OUTPUT_PATH,
        commands_run=["pytest targeted"],
    )
    artifact["validated_case_count"] = 0
    with pytest.raises(AssertionError, match="validated_case_count"):
        mod.validate_artifact(artifact)
