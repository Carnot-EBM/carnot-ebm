"""Tests for the reusable LTLZinc temporal benchmark generator.

Spec: REQ-LEARN-1630-6, REQ-LEARN-1630-7, SCENARIO-LEARN-1630.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from carnot.reporting import ltlzinc_temporal_continual_learning_adapter as temporal
from scripts import generate_ltlzinc as mod


def test_req_learn_1630_6_builds_stable_top_level_schema() -> None:
    """REQ-LEARN-1630-6: benchmark JSON exposes counts and source provenance."""

    benchmark = mod.build_benchmark()

    mod.validate_benchmark(benchmark)
    assert benchmark["schema"] == mod.SCHEMA
    assert benchmark["benchmark_id"] == mod.BENCHMARK_ID
    assert benchmark["source"]["constraint_templates_path"] == "data/constraint_templates.json"
    assert benchmark["source"]["template_count"] == 13
    assert benchmark["case_count"] == len(benchmark["cases"]) == 32
    assert benchmark["anchor_case_count"] == 24
    assert benchmark["update_case_count"] == 8
    assert benchmark["sat_case_count"] == 16
    assert benchmark["repair_hint_case_count"] == 16
    assert set(benchmark["supported_operators"]) == set(temporal.SUPPORTED_OPERATORS)


def test_req_learn_1630_7_cases_are_verifiable_and_retention_ready() -> None:
    """REQ-LEARN-1630-7: every case carries temporal and retention fields."""

    benchmark = mod.build_benchmark()
    cases = benchmark["cases"]
    case_ids = {case["case_id"] for case in cases}

    assert len(case_ids) == len(cases)
    family_states: dict[str, set[str]] = {}
    phases = {case["nonforgetting_phase"] for case in cases}
    assert phases == {"anchor", "update"}
    for case in cases:
        mod.validate_case_schema(case)
        assert temporal.verify_temporal_case(case) is bool(case["expected_satisfied"])
        assert case["evaluation"]["verifier_path"] == mod.VERIFIER_PATH
        assert case["source_template"]["template_id"] in benchmark["source"]["template_ids"]
        assert case["retention"]["phase"] == case["nonforgetting_phase"]
        if case["nonforgetting_phase"] == "anchor":
            assert case["retention"]["must_retrieve_after_updates"] is True
            assert case["retention"]["anchor_case_id"] == case["case_id"]
        family_states.setdefault(str(case["temporal_operator"]), set()).add(
            str(case["certificate_state"])
        )

    assert set(family_states) == set(temporal.SUPPORTED_OPERATORS)
    assert all(states == {"SAT", "REPAIR_HINT"} for states in family_states.values())


def test_req_learn_1630_6_write_benchmark_json(tmp_path: Path) -> None:
    """REQ-LEARN-1630-6: write_benchmark writes stable JSON to the requested path."""

    output_path = tmp_path / "data" / "ltlzinc_benchmark.json"

    benchmark = mod.write_benchmark(output_path=output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == benchmark
    assert benchmark == mod.build_benchmark()


def test_req_learn_1630_6_external_template_path_is_preserved(tmp_path: Path) -> None:
    """REQ-LEARN-1630-6: non-repo template paths remain auditable in provenance."""

    template_path = tmp_path / "constraint_templates.json"
    template_path.write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "template_id": "external",
                        "violation_pattern": "external temporal seed",
                        "violation_type": "verified_repair",
                        "source_experiment": 1630,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    benchmark = mod.build_benchmark(template_path=template_path)

    assert benchmark["source"]["constraint_templates_path"] == str(template_path)
    assert benchmark["source"]["template_count"] == 1
    assert {case["source_template"]["template_id"] for case in benchmark["cases"]} == {
        "external"
    }


def test_req_learn_1630_6_direct_script_bootstrap_adds_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-1630-6: direct execution can import repo-local script modules."""

    repo_root = str(mod.REPO_ROOT)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != repo_root])

    namespace = runpy.run_path(str(mod.REPO_ROOT / "scripts" / "generate_ltlzinc.py"))

    assert sys.path[0] == repo_root
    assert namespace["BENCHMARK_ID"] == mod.BENCHMARK_ID


def test_req_learn_1630_7_validation_rejects_bad_payloads() -> None:
    """REQ-LEARN-1630-7: validation protects counts and required case fields."""

    benchmark = mod.build_benchmark()

    with pytest.raises(AssertionError, match="case_count"):
        mod.validate_benchmark(dict(benchmark, case_count=0))

    bad_case = json.loads(json.dumps(benchmark["cases"][0]))
    del bad_case["retention"]
    with pytest.raises(AssertionError, match="missing benchmark case fields"):
        mod.validate_case_schema(bad_case)
