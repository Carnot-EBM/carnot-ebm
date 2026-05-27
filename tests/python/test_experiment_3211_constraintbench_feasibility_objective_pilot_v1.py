"""Tests for Exp 3211 ConstraintBench feasibility/objective exact pilot.

Spec refs: REQ-BENCH-3211, SCENARIO-BENCH-3211.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import constraintbench_feasibility_objective_pilot_v1 as exp


def test_req_bench_3211_spec_declares_exact_pilot_contract() -> None:
    """REQ-BENCH-3211: OpenSpec records fixture, artifact, and metric fields."""

    spec = Path("openspec/capabilities/benchmarks/spec.md").read_text(encoding="utf-8")

    assert "REQ-BENCH-3211" in spec
    assert "SCENARIO-BENCH-3211" in spec
    assert exp.FIXTURE_REL_PATH.as_posix() in spec
    assert exp.ARTIFACT_REL_PATH.as_posix() in spec
    assert "hallucinated_entity_metric_defined" in spec


def test_req_bench_3211_fixture_has_15_exact_rows_and_checker_backends(
    tmp_path: Path,
) -> None:
    """REQ-BENCH-3211: fixture rows expose exact references and checker metadata."""

    rows = exp.build_fixture_rows()
    fixture_path = tmp_path / exp.FIXTURE_REL_PATH
    payload = exp.write_fixture(rows, fixture_path)
    persisted = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert payload["fixture_count"] == 15
    assert persisted == rows
    assert {row["family"] for row in rows} == {"assignment", "graph_coloring", "knapsack"}
    assert {row["checker_backend"] for row in rows} == {
        "exact_assignment_permutation_enumerator",
        "exact_graph_coloring_enumerator",
        "exact_knapsack_subset_enumerator",
    }
    assert all(row["exact_reference"]["feasible"] is True for row in rows)
    assert all(row["exact_reference"]["objective_value"] is not None for row in rows)
    assert all(row["constraints"] for row in rows)
    assert all(row["checker"]["authority"] == "local_exhaustive_enumeration" for row in rows)


def test_scenario_bench_3211_checker_separates_failure_modes() -> None:
    """SCENARIO-BENCH-3211: checker splits format, entity, constraint, and gap."""

    rows = exp.build_fixture_rows()
    knapsack = next(row for row in rows if row["family"] == "knapsack")
    assignment = next(row for row in rows if row["family"] == "assignment")
    graph = next(row for row in rows if row["family"] == "graph_coloring")

    optimal = exp.score_candidate(knapsack, json.dumps(knapsack["exact_reference"]["solution"]))
    suboptimal = exp.score_candidate(
        knapsack,
        json.dumps({"selected_items": exp.feasible_nonoptimal_solution(knapsack)["selected_items"]}),
    )
    hallucinated = exp.score_candidate(knapsack, '{"selected_items": ["ghost-item"]}')
    missing_constraint = exp.score_candidate(
        graph,
        json.dumps({"colors": {str(node): 0 for node in graph["instance_data"]["nodes"]}}),
    )
    invalid = exp.score_candidate(assignment, "not-json")
    unknown = exp.score_candidate({"family": "unknown", "row_id": "bad"}, "{}")
    not_object = exp.score_candidate(knapsack, "[]")
    bad_knapsack_schema = exp.score_candidate(knapsack, '{"selected_items": "all"}')
    bad_assignment_schema = exp.score_candidate(assignment, '{"assignment": ["amy"]}')
    bad_graph_schema = exp.score_candidate(graph, '{"colors": {"0": "red"}}')
    unknown_reference = exp.solve_row({"family": "unknown"})

    assert optimal["valid_format"] is True
    assert optimal["feasibility_pass"] is True
    assert optimal["objective_gap"] == 0.0
    assert optimal["hallucinated_entity"] is False
    assert optimal["missing_constraint"] is False

    assert suboptimal["valid_format"] is True
    assert suboptimal["feasibility_pass"] is True
    assert suboptimal["objective_gap"] > 0.0
    assert suboptimal["missing_constraint"] is False

    assert hallucinated["valid_format"] is True
    assert hallucinated["hallucinated_entity"] is True
    assert hallucinated["feasibility_pass"] is False
    assert hallucinated["objective_gap"] is None

    assert missing_constraint["valid_format"] is True
    assert missing_constraint["hallucinated_entity"] is False
    assert missing_constraint["missing_constraint"] is True
    assert any(reason.startswith("edge_conflict:") for reason in missing_constraint["reasons"])

    assert invalid["valid_format"] is False
    assert invalid["invalid_format"] is True
    assert invalid["reasons"] == ["invalid_json"]

    assert unknown["valid_format"] is False
    assert unknown["invalid_format"] is True
    assert unknown["reasons"] == ["unknown_family"]

    assert not_object["reasons"] == ["candidate_not_object"]
    assert bad_knapsack_schema["reasons"] == ["selected_items_not_string_list"]
    assert bad_assignment_schema["reasons"] == ["assignment_not_string_mapping"]
    assert bad_graph_schema["reasons"] == ["colors_not_node_integer_mapping"]
    assert unknown_reference["feasible"] is False


def test_scenario_bench_3211_artifact_writes_required_fields_and_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-3211: artifact records separate exact-pilot metrics."""

    artifact = exp.write_artifacts(root=tmp_path, tests_run=("pytest exp3211",))
    fixture_path = tmp_path / exp.FIXTURE_REL_PATH
    artifact_path = tmp_path / exp.ARTIFACT_REL_PATH
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    fixture_rows = fixture_path.read_text(encoding="utf-8").splitlines()

    assert persisted == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema_version"] == exp.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3211"
    assert artifact["fixture_path"] == exp.FIXTURE_REL_PATH.as_posix()
    assert artifact["fixture_count"] == 15
    assert len(fixture_rows) == 15
    assert artifact["optimization_families"] == ["assignment", "graph_coloring", "knapsack"]
    assert artifact["exact_solver_backends"] == [
        "exact_assignment_permutation_enumerator",
        "exact_graph_coloring_enumerator",
        "exact_knapsack_subset_enumerator",
    ]
    assert artifact["metric_summary"]["candidate_count"] == 15
    assert artifact["metric_summary"]["feasibility_pass_rate"] == pytest.approx(0.6)
    assert artifact["metric_summary"]["objective_gap_mean_feasible"] > 0.0
    assert artifact["metric_summary"]["hallucinated_entity_rate"] == pytest.approx(
        1 / 15, abs=1e-6
    )
    assert artifact["metric_summary"]["missing_constraint_rate"] == pytest.approx(1 / 5)
    assert artifact["metric_summary"]["invalid_format_rate"] == pytest.approx(1 / 15, abs=1e-6)
    assert artifact["optional_llm_smoke"] is None
    assert artifact["ready_for_clean_verifier"] is True
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_bench_3211_aggregate_handles_empty_and_min_objective_gap() -> None:
    """REQ-BENCH-3211: objective gaps are defined for max and min families."""

    rows = exp.build_fixture_rows()
    graph = next(row for row in rows if row["family"] == "graph_coloring")
    exact = graph["exact_reference"]
    worse_coloring = dict(exact["solution"]["colors"])
    spare_color = max(graph["instance_data"]["colors"])
    worse_coloring[str(graph["instance_data"]["nodes"][0])] = spare_color

    exact_score = exp.score_candidate(graph, json.dumps(exact["solution"]))
    worse_score = exp.score_candidate(graph, json.dumps({"colors": worse_coloring}))
    empty = exp.aggregate_scores([])

    assert exact_score["objective_gap"] == 0.0
    assert worse_score["feasibility_pass"] is True
    assert worse_score["objective_gap"] >= 0.0
    assert empty["candidate_count"] == 0
    assert empty["feasibility_pass_rate"] == 0.0
    assert empty["objective_gap_mean_feasible"] is None
