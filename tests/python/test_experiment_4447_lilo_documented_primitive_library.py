"""Tests for Exp 4447 LILO-style documented primitive library retrieval.

Spec refs: REQ-REPORT-4447, SCENARIO-REPORT-4447.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4447_lilo_documented_primitive_library as exp
from carnot.agentic import arc_primitive_library as lib
from carnot.agentic import arc_solve_learning as learning


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def test_req_report_4447_spec_declares_documented_library_contract() -> None:
    """REQ-REPORT-4447: OpenSpec declares the documented library and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4447" in spec
    assert "SCENARIO-REPORT-4447" in spec
    assert exp.RESULT_RELATIVE_PATH in spec
    assert "retrieve_primitives(digest)" in spec
    assert "library_coverage" in spec
    assert "constant_leak_violations" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4447_autodoc_entries_have_required_retrieval_fields() -> None:
    """REQ-REPORT-4447: each AutoDoc row has name, mechanic, derived games, and cues."""

    entries = lib.documented_primitive_library()
    by_name = {entry.name: entry for entry in entries}

    assert "glyph_rewrite_matcher" in by_name
    assert "config_rule_verifier" in by_name
    assert "object_motion_world_model" in by_name
    assert "mechanic_object_motion_reflect" in by_name
    for entry in entries:
        row = entry.as_dict()
        assert row["name"]
        assert row["mechanic_class"]
        assert row["description"]
        assert row["derived_from_games"]
        assert row["retrieval_cues"]
        assert row["operator"]
        assert row["source"] in {"consolidated_primitive", "registry_mechanic"}
    assert lib.constant_leak_violations(entries) == []


def test_scenario_report_4447_retrieval_ranks_documented_primitives_before_generator() -> None:
    """SCENARIO-REPORT-4447: digest retrieval identifies held-out mechanics."""

    ft09_digest = {
        "game": "ft09",
        "mechanic_class": "local_constraint_color_cycle",
        "rule_family": "local_constraint_color_cycle",
        "constraints": [{"pattern": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]}],
        "cells": [{"color": 1}],
        "color_cycle": [1, 2, 3],
    }
    ft09_retrieved = lib.retrieve_primitives(ft09_digest, exclude_games=("ft09",))
    assert ft09_retrieved[0]["operator"] == "config_rule_verifier"
    assert lib.retrieval_identifies_mechanic(ft09_digest, ft09_retrieved) is True

    ar25_digest = {
        "game": "ar25",
        "mechanic_class": "object_motion_reflect",
        "motion_family": "reflect",
        "world_model": "object slots translate selected shape and mirrored reflected shape",
    }
    ar25_retrieved = lib.retrieve_primitives(ar25_digest, exclude_games=("ar25",))
    assert ar25_retrieved[0]["operator"] == "object_motion_world_model"
    assert lib.retrieval_identifies_mechanic(ar25_digest, ar25_retrieved[:1]) is True


def test_scenario_report_4447_leave_one_out_metrics_clear_falsifiable_gate() -> None:
    """SCENARIO-REPORT-4447: LOO coverage clears the self-learning gate."""

    metrics = lib.measure_leave_one_out()

    assert metrics["target_count"] == 18
    assert metrics["library_coverage"] >= 0.5
    assert 0.0 <= metrics["retrieval_precision_at_1"] <= 1.0
    assert metrics["constant_leak_violations"] == []
    assert any(row["identified"] for row in metrics["per_game"])


def test_req_report_4447_literal_scan_flags_game_specific_constants() -> None:
    """REQ-REPORT-4447: constant leakage is detected and excluded from counted primitives."""

    leaky = lib.DocumentedPrimitive(
        name="leaky_coord_recipe",
        mechanic_class="bad_constant_recipe",
        operator="bad_operator",
        description="click display coordinate (24,49) on L1 with sprite tag 0064ocqkuqacti",
        derived_from_games=("sc25",),
        retrieval_cues=("cell0,1", "exact action 44445222222244444"),
        supported_mechanic_classes=("bad_constant_recipe",),
        source="registry_mechanic",
    )

    violations = lib.constant_leak_violations([leaky])

    assert {violation["kind"] for violation in violations} >= {
        "coordinate_literal",
        "level_id_literal",
        "sprite_tag_literal",
        "action_sequence_literal",
    }
    assert all(violation["entry"] == "leaky_coord_recipe" for violation in violations)


def test_req_report_4447_recommend_approach_exposes_ranked_library_primitives() -> None:
    """REQ-REPORT-4447: recommend_approach queries documented primitives first."""

    recommendation = learning.recommend_approach("ft09")

    assert "retrieved_primitives" in recommendation
    assert recommendation["retrieved_primitives"]
    assert recommendation["retrieved_primitives"][0]["operator"] == "config_rule_verifier"
    assert "selected_generic_operators" in recommendation
    assert "recommended" in recommendation


def test_req_report_4447_experiment_artifact_schema_and_write(tmp_path: Path) -> None:
    """REQ-REPORT-4447: experiment artifact carries required bare fields and checksum."""

    artifact = exp.run(root=REPO, write=False, no_regression=True)

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["inference_substrate"].startswith("aggregation_from_upstream_artifacts")
    assert artifact["library_coverage"] >= 0.5
    assert type(artifact["library_coverage"]) is float
    assert type(artifact["retrieval_precision_at_1"]) is float
    assert artifact["constant_leak_violations"] == []
    assert artifact["no_regression"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["random_seed"] == exp.RANDOM_SEED
    assert exp.artifact_schema_errors(artifact) == []

    path = exp.write_artifact(tmp_path, artifact)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_report_4447_schema_rejects_malformed_or_blocked_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-4447: schema catches type drift and blocked resources are honest."""

    blocked = exp.run(root=tmp_path, write=True, no_regression=False)
    assert blocked["honest_verdict"] == "complete: blocked_arc_solve_registry"
    assert blocked["library_coverage"] == 0.0
    assert exp.artifact_schema_errors(blocked) == []

    artifact = exp.run(root=REPO, write=False, no_regression=True)
    bad: dict[str, Any] = {
        **artifact,
        "honest_verdict": "partial: invalid",
        "inference_substrate": None,
        "library_coverage": "0.5",
        "retrieval_precision_at_1": "1.0",
        "primitives_documented": [],
        "constant_leak_violations": {},
        "no_regression": "true",
        "verifier_is_oracle": False,
        "random_seed": "4447",
        "reproducibility_checksum": "z" * 64,
        "submitted_to_leaderboard": True,
        "field_principles": {**exp.FIELD_PRINCIPLES, "honest_verdict": {"principle": "wrong"}},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "inference_substrate must not be None" in errors
    assert "library_coverage must be bare float" in errors
    assert "retrieval_precision_at_1 must be bare float" in errors
    assert "primitives_documented must be non-empty list" in errors
    assert "constant_leak_violations must be list" in errors
    assert "no_regression must be bare bool" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4447" in errors
