"""Tests for Exp 3223 exact-row uncertainty sidecar v2.

Spec refs: REQ-VERIFY-3223, SCENARIO-VERIFY-3223.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import distributional_ebm_exact_row_uncertainty_sidecar_v2 as mod


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
            "fixture_id": "ctx-synthetic-1",
            "family": "symbolic_aliases",
            "context": "In this glossary, mercury means banana.",
            "question": "What does mercury mean here?",
            "expected_answer": "banana",
            "prior_bait_answer": "planet",
            "exact_checker_type": "exact_alias_string",
            "minimal_counterexample": {
                "candidate_answer": "planet",
                "expected_answer": "banana",
                "failure_mode": "parametric_prior_shortcut",
            },
        },
        {
            "fixture_id": "ctx-synthetic-2",
            "family": "local_arithmetic_rules",
            "context": "In this worksheet, plus means multiply.",
            "question": "What is 3 plus 4?",
            "expected_answer": "12",
            "prior_bait_answer": "7",
            "exact_checker_type": "exact_integer_string",
            "minimal_counterexample": {
                "candidate_answer": "7",
                "expected_answer": "12",
                "failure_mode": "parametric_prior_shortcut",
            },
        },
    ]
    constraint_rows = [
        {
            "row_id": "cb-synthetic-1",
            "family": "knapsack",
            "checker_backend": "exact_knapsack_subset_enumerator",
            "constraints": ["weight <= capacity", "select required item"],
            "exact_reference": {
                "feasible": True,
                "feasible_count": 2,
                "objective_value": 10,
                "objective_sense": "max",
                "solution": {"selected_items": ["kit"]},
            },
        },
        {
            "row_id": "cb-synthetic-2",
            "family": "assignment",
            "checker_backend": "exact_assignment_permutation_enumerator",
            "constraints": ["one worker per task", "required task filled"],
            "exact_reference": {
                "feasible": True,
                "feasible_count": 1,
                "objective_value": 20,
                "objective_sense": "max",
                "solution": {"assignment": {"pack": "amy"}},
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
            "milestone": "2026.05.297",
            "fixture_path": mod.CONTEXT_FIXTURE_REL_PATH.as_posix(),
            "fixture_count": len(context_rows),
            "fixture_families": ["symbolic_aliases", "local_arithmetic_rules"],
            "exact_checker_types": ["exact_alias_string", "exact_integer_string"],
            "prior_bait_row_count": len(context_rows),
            "ready_for_clean_verifier": True,
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "honest_verdict": "complete: synthetic context fixture",
        },
    )
    _write_json(
        root,
        mod.CONSTRAINT_ARTIFACT_REL_PATH,
        {
            "schema_version": "carnot.constraintbench_feasibility_objective_pilot.v1",
            "experiment_id": "exp3211",
            "milestone": "2026.05.297",
            "fixture_path": mod.CONSTRAINT_FIXTURE_REL_PATH.as_posix(),
            "fixture_count": len(constraint_rows),
            "ready_for_clean_verifier": True,
            "candidate_scores": [
                {
                    "row_id": "cb-synthetic-1",
                    "family": "knapsack",
                    "checker_backend": "exact_knapsack_subset_enumerator",
                    "valid_format": True,
                    "feasibility_pass": True,
                    "hallucinated_entity": False,
                    "missing_constraint": False,
                    "invalid_format": False,
                    "objective_gap": 0.0,
                    "objective_value": 10,
                    "optimum_value": 10,
                },
                {
                    "row_id": "cb-synthetic-2",
                    "family": "assignment",
                    "checker_backend": "exact_assignment_permutation_enumerator",
                    "valid_format": True,
                    "feasibility_pass": False,
                    "hallucinated_entity": False,
                    "missing_constraint": True,
                    "invalid_format": False,
                    "objective_gap": None,
                    "objective_value": None,
                    "optimum_value": 20,
                },
            ],
            "conductor_file_modified": False,
            "active_roadmap_modified": False,
            "honest_verdict": "complete: synthetic constraint fixture",
        },
    )


def test_req_verify_3223_spec_declares_sidecar_contract() -> None:
    """REQ-VERIFY-3223: OpenSpec declares the artifact, sources, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3223" in spec
    assert "SCENARIO-VERIFY-3223" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.CONTEXT_ARTIFACT_REL_PATH.as_posix() in spec
    assert mod.CONSTRAINT_ARTIFACT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_3223_scores_exact_rows_without_model_calls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3223: checked-in exact rows receive triage-only metadata."""

    output = mod.write_artifact(
        REPO_ROOT,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.5,
        tests_run=["SCENARIO-VERIFY-3223 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3223"
    assert artifact["milestone"] == "2026.05.298"
    assert artifact["exact_row_count"] == 45
    assert len(artifact["sidecar_rows"]) == 45
    assert artifact["uncertainty_sidecar_ready"] is True
    assert artifact["abstention_threshold_defined"] is True
    assert artifact["exact_verifier_authority_preserved"] is True
    assert artifact["inference_substrate"] == "deterministic_artifact_replay_no_llm"
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["SCENARIO-VERIFY-3223 focused"]
    assert artifact["honest_verdict"].startswith("complete:")

    source_ids = {source["id"] for source in artifact["source_fixture_artifacts"]}
    assert source_ids == {
        "exp3210_context_artifact",
        "exp3210_context_fixture",
        "exp3211_constraint_artifact",
        "exp3211_constraint_fixture",
    }
    assert all(source["exists"] is True for source in artifact["source_fixture_artifacts"])

    row_by_id = {row["row_id"]: row for row in artifact["sidecar_rows"]}
    context_row = row_by_id["ctx3210-arithmetic-01"]
    constraint_bad = row_by_id["cbpilot-3211-knapsack-004"]
    constraint_clean = row_by_id["cbpilot-3211-knapsack-001"]

    for row in artifact["sidecar_rows"]:
        assert set(mod.ROW_SCORE_FIELDS) <= set(row)
        assert 0.0 <= row["uncertainty_score"] <= 1.0
        assert 0.0 <= row["abstention_risk"] <= 1.0
        assert 0.0 <= row["shortcut_risk"] <= 1.0
        assert 0.0 <= row["solver_disagreement_risk"] <= 1.0
        assert row["model_identity"] == "no_model_identity:deterministic_fixture"
        assert row["exact_verifier_authority"] in {
            "context_exact_checker",
            "constraintbench_exact_bounded_solver",
        }

    assert context_row["artifact_source"] == "exp3210_context"
    assert context_row["context_dependency_label"] == "context_required_prior_contradiction"
    assert context_row["shortcut_risk"] >= 0.7
    assert context_row["solver_disagreement_risk"] == 0.0

    assert constraint_bad["artifact_source"] == "exp3211_constraintbench"
    assert constraint_bad["solver_disagreement_risk"] > constraint_clean["solver_disagreement_risk"]
    assert constraint_bad["feature_branches"]["missing_constraint_flag"] == 1.0

    assert {row["artifact_source"] for row in artifact["shortcut_risk_rows"]} == {
        "exp3210_context"
    }
    assert any(
        row["row_id"].startswith("cbpilot-3211-")
        for row in artifact["solver_disagreement_risk_rows"]
    )

    audit = artifact["model_identity_shortcut_audit"]
    assert audit["model_identity_dominated"] is False
    assert audit["artifact_source_dominated"] is False
    assert audit["risk_dominated_by"] == "row_difficulty"
    assert audit["row_count"] == artifact["exact_row_count"]
    assert audit["model_identity_values"] == ["no_model_identity:deterministic_fixture"]

    assert "exp3225" in artifact["clean_verifier_consumption_plan"].lower()
    assert "exact verifier" in artifact["clean_verifier_consumption_plan"].lower()
    mod.validate_artifact(artifact)


def test_req_verify_3223_synthetic_sources_and_missing_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-3223: synthetic exact rows score, while missing sources block."""

    missing = mod.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.25,
        tests_run=["REQ-VERIFY-3223 missing"],
    )
    assert missing["uncertainty_sidecar_ready"] is False
    assert missing["exact_row_count"] == 0
    assert missing["abstention_threshold_defined"] is False
    assert missing["honest_verdict"].startswith("blocked_")
    mod.validate_artifact(missing)

    _synthetic_sources(tmp_path)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=3.0,
        tests_run=["REQ-VERIFY-3223 synthetic"],
    )

    assert artifact["uncertainty_sidecar_ready"] is True
    assert artifact["exact_row_count"] == 4
    assert artifact["abstention_threshold_defined"] is True
    assert len(artifact["sidecar_rows"]) == 4
    assert artifact["duration_s"] == pytest.approx(1.0)

    rows = {row["row_id"]: row for row in artifact["sidecar_rows"]}
    assert rows["ctx-synthetic-2"]["difficulty_score"] > rows["ctx-synthetic-1"][
        "difficulty_score"
    ]
    assert rows["cb-synthetic-2"]["solver_disagreement_risk"] > rows["cb-synthetic-1"][
        "solver_disagreement_risk"
    ]
    assert artifact["shortcut_risk_rows"][0]["row_id"].startswith("ctx-synthetic-")
    assert artifact["solver_disagreement_risk_rows"][0]["row_id"] == "cb-synthetic-2"
    assert artifact["model_identity_shortcut_audit"]["artifact_source_dominated"] is False
    mod.validate_artifact(artifact)

    rel_output = mod.write_artifact(
        tmp_path,
        output_path=Path("relative-exp3223.json"),
        started_s=4.0,
        now_s=4.5,
        tests_run=["relative-output"],
    )
    assert rel_output == tmp_path / "relative-exp3223.json"


def test_req_verify_3223_validation_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-VERIFY-3223: validation rejects missing fields and authority overclaims."""

    _synthetic_sources(tmp_path)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=6.0,
        tests_run=["validation"],
    )

    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")
    invalid_cases = [
        (missing_required, "missing required fields"),
        (
            artifact | {"inference_substrate": "live_llm_inference"},
            "deterministic artifact replay",
        ),
        (
            artifact | {"exact_verifier_authority_preserved": False},
            "exact verifier authority",
        ),
        (
            artifact | {"exact_row_count": artifact["exact_row_count"] + 1},
            "exact_row_count",
        ),
        (
            artifact | {"abstention_threshold_defined": False},
            "abstention threshold",
        ),
        (
            artifact
            | {
                "model_identity_shortcut_audit": artifact["model_identity_shortcut_audit"]
                | {"model_identity_dominated": True}
            },
            "model identity dominated",
        ),
        (
            artifact
            | {
                "model_identity_shortcut_audit": artifact["model_identity_shortcut_audit"]
                | {"artifact_source_dominated": True}
            },
            "artifact source dominated",
        ),
        (
            artifact
            | {
                "sidecar_rows": [
                    artifact["sidecar_rows"][0] | {"abstention_risk": 1.25},
                    *artifact["sidecar_rows"][1:],
                ]
            },
            "unit interval",
        ),
        (artifact | {"honest_verdict": "ready"}, "honest_verdict"),
    ]

    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad_artifact)


def test_req_verify_3223_metric_helpers_are_bounded(tmp_path: Path) -> None:
    """REQ-VERIFY-3223: helper metrics stay stable on degenerate inputs."""

    assert mod.unit_interval(-0.2) == 0.0
    assert mod.unit_interval(1.2) == 1.0
    assert mod.unit_interval(0.25) == pytest.approx(0.25)
    assert mod.safe_rate(0, 0) == 0.0
    assert mod.safe_rate(2, 4) == pytest.approx(0.5)
    assert mod.pearson_correlation([], []) == 0.0
    assert mod.pearson_correlation([1.0, 1.0], [0.2, 0.4]) == 0.0
    assert mod.pearson_correlation([0.0, 1.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert mod.candidate_score_by_id({}) == {}
    assert mod.group_delta({}) == 0.0
    assert mod.answer_delta_score("", "") == 0.0

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n{\"row_id\": \"covered\"}\n", encoding="utf-8")
    assert mod.read_jsonl_objects(blank_jsonl) == [{"row_id": "covered"}]

    fallback_row = mod.score_constraint_row(
        {
            "row_id": "fallback-objective",
            "family": "knapsack",
            "checker_backend": "exact_knapsack_subset_enumerator",
            "constraints": ["capacity"],
            "exact_reference": {
                "feasible": True,
                "feasible_count": 1,
                "objective_value": 8,
            },
        },
        {"fallback-objective": {"row_id": "fallback-objective", "objective_gap": 4}},
    )
    assert fallback_row["feature_branches"]["objective_gap_norm"] == pytest.approx(0.5)
