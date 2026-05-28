"""Tests for Exp 3224 Logitext-style partial SMT coverage pilot.

Spec refs: REQ-VERIFY-3224, SCENARIO-VERIFY-3224.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import logitext_partial_smt_context_coverage_pilot_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path, rows: list[dict[str, Any]]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _synthetic_sources(root: Path) -> None:
    context_rows = [
        {
            "fixture_id": "ctx-smt-symbolic",
            "family": "symbolic_aliases",
            "context": "In this glossary, mercury means banana.",
            "question": "What does mercury mean here?",
            "expected_answer": "banana",
            "prior_bait_answer": "planet",
            "exact_checker_type": "exact_alias_string",
            "minimal_counterexample": {"failure_mode": "parametric_prior_shortcut"},
        },
        {
            "fixture_id": "ctx-smt-arithmetic",
            "family": "local_arithmetic_rules",
            "context": "In this worksheet, plus means multiply.",
            "question": "What is 3 plus 4?",
            "expected_answer": "12",
            "prior_bait_answer": "7",
            "exact_checker_type": "exact_integer_string",
            "minimal_counterexample": {"failure_mode": "parametric_prior_shortcut"},
        },
    ]
    constraint_rows = [
        {
            "row_id": "cb-smt-knapsack",
            "family": "knapsack",
            "checker_backend": "exact_knapsack_subset_enumerator",
            "instance_data": {
                "capacity": 3,
                "items": [
                    {"name": "kit", "weight": 1, "value": 3},
                    {"name": "map", "weight": 2, "value": 5},
                ],
                "required_items": ["kit"],
                "incompatible_pairs": [],
            },
            "constraints": [
                "selected_items must name only listed items",
                "total selected weight must be <= capacity",
                "all required_items must be selected",
            ],
            "objective": {"sense": "max", "name": "total_value"},
            "exact_reference": {
                "feasible": True,
                "objective_sense": "max",
                "objective_name": "total_value",
                "objective_value": 8,
                "solution": {"selected_items": ["kit", "map"]},
                "feasible_count": 2,
            },
        },
        {
            "row_id": "cb-smt-coloring",
            "family": "graph_coloring",
            "checker_backend": "exact_graph_coloring_enumerator",
            "instance_data": {
                "nodes": [0, 1],
                "edges": [[0, 1]],
                "colors": [0, 1],
            },
            "constraints": [
                "every listed node must receive exactly one listed color",
                "adjacent nodes must receive different colors",
            ],
            "objective": {"sense": "min", "name": "used_color_count"},
            "exact_reference": {
                "feasible": True,
                "objective_sense": "min",
                "objective_name": "used_color_count",
                "objective_value": 2,
                "solution": {"colors": {"0": 0, "1": 1}},
                "feasible_count": 2,
            },
        },
    ]
    _write_jsonl(root, mod.CONTEXT_FIXTURE_REL_PATH, context_rows)
    _write_jsonl(root, mod.CONSTRAINT_FIXTURE_REL_PATH, constraint_rows)
    _write_json(
        root,
        mod.CONTEXT_ARTIFACT_REL_PATH,
        {
            "schema_version": "carnot.context_cot_clbench_parametric_shortcut_fixtures.v1",
            "experiment_id": "exp3210",
            "fixture_path": mod.CONTEXT_FIXTURE_REL_PATH.as_posix(),
            "fixture_count": len(context_rows),
            "ready_for_clean_verifier": True,
            "honest_verdict": "complete: synthetic context fixture",
        },
    )
    _write_json(
        root,
        mod.CONSTRAINT_ARTIFACT_REL_PATH,
        {
            "schema_version": "carnot.constraintbench_feasibility_objective_pilot.v1",
            "experiment_id": "exp3211",
            "fixture_path": mod.CONSTRAINT_FIXTURE_REL_PATH.as_posix(),
            "fixture_count": len(constraint_rows),
            "ready_for_clean_verifier": True,
            "honest_verdict": "complete: synthetic constraint fixture",
        },
    )


def test_req_verify_3224_spec_declares_coverage_contract() -> None:
    """REQ-VERIFY-3224: OpenSpec declares the artifact, taxonomy, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3224" in spec
    assert "SCENARIO-VERIFY-3224" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.CONTEXT_ARTIFACT_REL_PATH.as_posix() in spec
    assert mod.CONSTRAINT_ARTIFACT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for fragment_type in mod.TAXONOMY_FRAGMENT_TYPES:
        assert fragment_type.replace("_", "-") in spec or fragment_type in spec


def test_scenario_verify_3224_builds_checked_in_partial_smt_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3224: all `.297` rows receive exact SMT coverage labels."""

    output = mod.write_artifact(
        REPO_ROOT,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.0,
        tests_run=["SCENARIO-VERIFY-3224 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3224"
    assert artifact["milestone"] == "2026.05.298"
    assert artifact["fixture_row_count"] == 45
    assert artifact["fully_formalizable_count"] == 35
    assert artifact["partially_formalizable_count"] == 10
    assert artifact["not_formalizable_count"] == 0
    assert artifact["exact_solver_row_count"] == 45
    assert artifact["partial_smt_coverage"] == pytest.approx(1.0)
    assert artifact["coverage_ready"] is True
    assert artifact["inference_substrate"] == "deterministic_artifact_replay_no_llm"
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["tests_run"] == ["SCENARIO-VERIFY-3224 focused"]
    assert artifact["honest_verdict"].startswith("complete:")

    taxonomy = {entry["fragment_type"] for entry in artifact["constraint_taxonomy"]}
    assert set(mod.TAXONOMY_FRAGMENT_TYPES) == taxonomy
    assert len(artifact["smt_rows"]) == artifact["fixture_row_count"]
    assert all(set(mod.SMT_ROW_FIELDS) <= set(row) for row in artifact["smt_rows"])
    assert all(source["exists"] is True for source in artifact["source_fixture_artifacts"])

    family = artifact["coverage_by_fixture_family"]
    assert family["symbolic_aliases"]["fully_formalizable"] == 10
    assert family["context_defined_entity_facts"]["fully_formalizable"] == 10
    assert family["local_arithmetic_rules"]["partially_formalizable"] == 10
    assert family["knapsack"]["fully_formalizable"] == 5
    assert family["assignment"]["fully_formalizable"] == 5
    assert family["graph_coloring"]["fully_formalizable"] == 5

    rows = {row["row_id"]: row for row in artifact["smt_rows"]}
    symbolic = rows["ctx3210-symbolic-01"]
    arithmetic = rows["ctx3210-arithmetic-01"]
    assignment = rows["cbpilot-3211-assignment-001"]
    coloring = rows["cbpilot-3211-coloring-001"]

    assert symbolic["formalizability"] == "fully_formalizable"
    assert symbolic["exact_solver_pointer"]["checker_type"] == "exact_alias_string"
    assert symbolic["solver_ready_representation"]["kind"] == "answer_string_equality"

    assert arithmetic["formalizability"] == "partially_formalizable"
    assert arithmetic["solver_ready_representation"]["kind"] == "answer_integer_equality"
    assert "natural_language_local_rule_semantics" in arithmetic["unformalized_requirements"]

    assignment_fragments = {fragment["fragment_type"] for fragment in assignment["formalized_fragments"]}
    assert assignment["formalizability"] == "fully_formalizable"
    assert "all_different" in assignment_fragments
    assert "objective_bound" in assignment_fragments
    assert assignment["exact_solver_pointer"]["checker_backend"] == "exact_assignment_permutation_enumerator"

    coloring_fragments = {fragment["fragment_type"] for fragment in coloring["formalized_fragments"]}
    assert coloring["formalizability"] == "fully_formalizable"
    assert "graph_relation" in coloring_fragments
    assert "objective_bound" in coloring_fragments

    assert artifact["highest_value_rows_for_exp3225"]
    assert all("exp3225" in row["use_for"].lower() for row in artifact["highest_value_rows_for_exp3225"])
    mod.validate_artifact(artifact)


def test_req_verify_3224_missing_and_synthetic_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-3224: missing sources block; synthetic rows preserve labels."""

    missing = mod.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.5,
        tests_run=["REQ-VERIFY-3224 missing"],
    )
    assert missing["coverage_ready"] is False
    assert missing["fixture_row_count"] == 0
    assert missing["honest_verdict"].startswith("blocked_")
    mod.validate_artifact(missing)

    _synthetic_sources(tmp_path)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=3.25,
        tests_run=["REQ-VERIFY-3224 synthetic"],
    )

    assert artifact["coverage_ready"] is True
    assert artifact["fixture_row_count"] == 4
    assert artifact["fully_formalizable_count"] == 3
    assert artifact["partially_formalizable_count"] == 1
    assert artifact["not_formalizable_count"] == 0
    assert artifact["partial_smt_coverage"] == pytest.approx(1.0)
    assert artifact["duration_s"] == pytest.approx(1.25)

    rows = {row["row_id"]: row for row in artifact["smt_rows"]}
    assert rows["ctx-smt-symbolic"]["formalizability"] == "fully_formalizable"
    assert rows["ctx-smt-arithmetic"]["formalizability"] == "partially_formalizable"
    assert rows["cb-smt-knapsack"]["solver_ready_representation"]["kind"] == "knapsack_bv_linear"
    assert rows["cb-smt-coloring"]["solver_ready_representation"]["kind"] == "graph_coloring_finite_domain"
    mod.validate_artifact(artifact)

    relative = mod.write_artifact(
        tmp_path,
        output_path=Path("relative-exp3224.json"),
        started_s=4.0,
        now_s=4.5,
        tests_run=["relative-output"],
    )
    assert relative == tmp_path / "relative-exp3224.json"


def test_req_verify_3224_validation_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-VERIFY-3224: validation rejects missing fields and unsafe overclaims."""

    _synthetic_sources(tmp_path)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=6.0,
        tests_run=["validation"],
    )

    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required artifact field"):
        mod.validate_artifact(missing_required)

    count_mismatch = dict(artifact, fully_formalizable_count=0)
    with pytest.raises(ValueError, match="formalizability counts"):
        mod.validate_artifact(count_mismatch)

    live_overclaim = dict(artifact, inference_substrate="live_llm")
    with pytest.raises(ValueError, match="deterministic artifact replay"):
        mod.validate_artifact(live_overclaim)

    modified_conductor = dict(artifact, conductor_file_modified=True)
    with pytest.raises(ValueError, match="conductor"):
        mod.validate_artifact(modified_conductor)

    broken_rows = dict(artifact)
    broken_rows["smt_rows"] = [dict(row) for row in artifact["smt_rows"]]
    broken_rows["smt_rows"][0]["exact_solver_pointer"] = None
    broken_rows["smt_rows"][0]["solver_ready_representation"] = None
    broken_rows["smt_rows"][0]["formalized_fragments"] = []
    broken_rows["exact_solver_row_count"] -= 1
    with pytest.raises(ValueError, match="formalizable row lacks"):
        mod.validate_artifact(broken_rows)

    unknown = mod.constraintbench_smt_row({"row_id": "unknown", "family": "mystery"})
    assert unknown["formalizability"] == "not_formalizable_without_extraction"
    assert unknown["unformalized_requirements"] == ["unknown_constraintbench_family"]

    assert mod.int_value("not-an-int") == 0
    assert mod.parse_int_or_string("not-an-int") == "not-an-int"
    assert mod.numeric_or_none("not-a-number") is None

    object_path = tmp_path / "not-object.json"
    object_path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        mod.read_json_object(object_path)

    jsonl_path = tmp_path / "bad.jsonl"
    jsonl_path.write_text("\n[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSONL object row"):
        mod.read_jsonl_objects(jsonl_path)
