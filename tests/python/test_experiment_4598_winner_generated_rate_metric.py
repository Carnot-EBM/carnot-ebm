"""Tests for Exp 4598 winner-generated-rate co-headline.

Spec refs: REQ-CAPSTONE-4598, SCENARIO-CAPSTONE-4598,
SCENARIO-CAPSTONE-4598-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4550_honest_sprint_metric as exp4550
from carnot import experiment_4586_live_submittable_coheadline as exp4586
from carnot import experiment_4598_winner_generated_rate_metric as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _registry(total: int = 4) -> JsonDict:
    return {
        "schema_version": 1,
        "reproducible_total_levels": total,
        "games": [
            {
                "game": "with_traj",
                "reproducibility": "reproduced",
                "levels_reproduced": 3,
            },
            {
                "game": "adaptive",
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
            },
        ],
    }


def _package() -> JsonDict:
    return {
        "experiment": "fixture_package",
        "package_manifest": [
            {
                "game": "with_traj",
                "levels": 3,
                "offline_reproduced_level": 3,
                "trajectory_path": "results/arc3_live_banked_trajectories/with_traj.json",
                "action_count": 4,
                "env_matched": True,
            },
            {
                "game": "adaptive",
                "levels": 1,
                "offline_reproduced_level": 1,
                "adaptive_solver": "fixture_env_adaptive_solver",
                "env_matched": True,
            },
        ],
    }


def _b1_artifact() -> JsonDict:
    return {
        "reproducible_total_levels": 4,
        "generic_transfer_rate_over_variants": 0.25,
        "generic_transfer_ci": [0.0, 0.5],
        "action_efficiency_score": 0.75,
        "action_efficiency_ci": [0.5, 1.0],
        "median_actions_to_first_levelup": 6.0,
        "human_baseline_actions": 9.0,
    }


def _winner_source() -> JsonDict:
    return {
        "result_path": "results/fixture_winner_source.json",
        "winner_generated": {
            "generated_count": 2,
            "attempted_count": 4,
            "not_generated_count": 2,
        },
        "generic_transfer_rate_with_router": 0.25,
    }


def _preconditions() -> JsonDict:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4598": True,
        "offline_arcade": True,
        "winner_source_artifact_present": True,
        "live_submittable_coheadline_artifact_present": True,
        "capstone_423_artifact_present": True,
        "generation_wiring_artifact_present": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }


def test_req_capstone_4598_spec_declares_winner_generated_rate_contract() -> None:
    """REQ-CAPSTONE-4598: OpenSpec declares the co-headline metric contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4598",
        "SCENARIO-CAPSTONE-4598",
        "SCENARIO-CAPSTONE-4598-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4598_shared_helper_counts_generated_but_not_selected() -> None:
    """SCENARIO-CAPSTONE-4598: winner generation is counted separately from solved transfer."""

    attempts = [
        {
            "variant_signature": "g1~color01",
            "attempted": True,
            "solved": True,
            "winner_generated": False,
        },
        {
            "variant_signature": "g2~color01",
            "attempted": True,
            "solved": False,
            "winner_generated": True,
        },
        {
            "variant_signature": "g3~color01",
            "attempted": True,
            "solved": False,
            "winner_generated": False,
        },
        {
            "variant_signature": "not_attempted~color01",
            "attempted": False,
            "solved": False,
            "winner_generated": True,
        },
    ]

    metric = exp4550.measure_winner_generated_over_variants(attempts)

    assert metric["winner_generated_attempted_count"] == 3
    assert metric["winner_generated_count"] == 2
    assert metric["winner_generated_not_selected_count"] == 1
    assert metric["generic_transfer_solved_count"] == 1
    assert metric["winner_generated_rate"] == pytest.approx(2 / 3)
    assert metric["generic_transfer_rate_over_variants"] == pytest.approx(1 / 3)
    assert metric["generation_vs_ranking_gap"] == pytest.approx(1 / 3)
    assert metric["winner_generated_rate"] >= metric["generic_transfer_rate_over_variants"]


def test_req_capstone_4598_exp4582_baseline_reproduces_one_of_twenty_five() -> None:
    """REQ-CAPSTONE-4598: the recorded Exp4582 baseline is 1/25 winner-generated."""

    source = mod._read_json(REPO / mod.FEATURE_ROUTER_BASELINE_RELATIVE_PATH)
    metric = exp4550.winner_generated_metric_from_artifact(source)

    assert metric["winner_generated_attempted_count"] == 25
    assert metric["winner_generated_count"] == 1
    assert metric["winner_generated_rate"] == pytest.approx(0.04)
    assert metric["generic_transfer_rate_over_variants"] == pytest.approx(0.04)
    assert metric["generation_vs_ranking_gap"] == pytest.approx(0.0)


def test_scenario_capstone_4598_live_bridge_reports_all_five_coheadlines() -> None:
    """SCENARIO-CAPSTONE-4598: the capstone bridge lists all co-headline metrics."""

    metrics = exp4586.build_winner_generated_rate_coheadline_metrics(
        registry=_registry(),
        package=_package(),
        package_path="results/fixture_package.json",
        b1_artifact=_b1_artifact(),
        winner_source_artifact=_winner_source(),
    )

    assert metrics["reported_side_by_side"] == [
        "reproducible_total_levels",
        "live_submittable_level_count",
        "reproducible_vs_submittable_gap",
        "generic_transfer_rate_over_variants",
        "generic_transfer_ci",
        "action_efficiency_score",
        "action_efficiency_ci",
        "winner_generated_rate",
        "generation_vs_ranking_gap",
    ]
    assert metrics["reproducible_total_levels"] == 4
    assert metrics["live_submittable_level_count"] == 4
    assert metrics["generic_transfer_rate_over_variants"] == pytest.approx(0.25)
    assert metrics["action_efficiency_score"] == pytest.approx(0.75)
    assert metrics["winner_generated_rate"] == pytest.approx(0.5)
    assert metrics["generation_vs_ranking_gap"] == pytest.approx(0.25)


def test_scenario_capstone_4598_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4598-FIELD-PRINCIPLES: artifact exposes the generation gap."""

    artifact = mod.build_artifact(
        root=tmp_path,
        registry=_registry(),
        package=_package(),
        package_path="results/fixture_package.json",
        b1_artifact=_b1_artifact(),
        winner_source_artifact=_winner_source(),
        preconditions_checked=_preconditions(),
    )

    assert artifact["honest_verdict"] == "shipped: winner_generated_rate_coheadline_wired"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["reproducible_total_levels"] == 4
    assert artifact["live_submittable_level_count"] == 4
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(0.25)
    assert artifact["winner_generated_rate"] == pytest.approx(0.5)
    assert artifact["generation_vs_ranking_gap"] == pytest.approx(0.25)
    assert "can we GENERATE the winner at all" in artifact["honest_metric_framing"]
    assert artifact["metric_wired_into_capstone"]["reported_side_by_side"][-2:] == [
        "winner_generated_rate",
        "generation_vs_ranking_gap",
    ]
    assert artifact["tests_added_pass"]["passed"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) == []

    path = mod.write_artifact(tmp_path, artifact=artifact)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_scenario_capstone_4598_validation_rejects_winner_below_transfer(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4598: solving without generating the winner is invalid."""

    artifact = mod.build_artifact(
        root=tmp_path,
        registry=_registry(),
        package=_package(),
        package_path="results/fixture_package.json",
        b1_artifact=_b1_artifact(),
        winner_source_artifact=_winner_source(),
        preconditions_checked=_preconditions(),
    )
    bad = dict(artifact)
    bad["winner_generated_rate"] = 0.1

    errors = mod.validate_artifact(bad)

    assert "winner_generated_rate must be >= generic_transfer_rate_over_variants" in errors


def test_scenario_capstone_4598_partial_verdicts_and_missing_json(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4598: defensive helpers keep missing resources honest."""

    assert mod._read_json(tmp_path / "missing.json") == {}
    assert (
        mod._honest_verdict({"ok": False}, 0.5, 0.25)
        == "complete: winner_generated_rate_coheadline_partial_preconditions"
    )
    assert (
        mod._honest_verdict({"ok": True}, 0.1, 0.25)
        == "complete: winner_generated_rate_coheadline_partial_metric_invariant"
    )


def test_scenario_capstone_4598_validation_reports_malformed_fields(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4598-FIELD-PRINCIPLES: malformed fields fail validation."""

    artifact = mod.build_artifact(
        root=tmp_path,
        registry=_registry(),
        package=_package(),
        package_path="results/fixture_package.json",
        b1_artifact=_b1_artifact(),
        winner_source_artifact=_winner_source(),
        preconditions_checked=_preconditions(),
    )

    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "not_terminal",
            "inference_substrate": "wrong",
            "winner_generated_rate": "0.5",
            "generation_vs_ranking_gap": "0.25",
            "winner_generated_count": "2",
            "metric_wired_into_capstone": [],
            "leaderboard_submission": True,
            "tests_added_pass": [],
            "preconditions_checked": [],
            "field_principles": None,
            "reproducibility_checksum": "bad",
        }
    )

    errors = mod.validate_artifact(bad)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate mismatch" in errors
    assert "winner_generated_rate must be a bare float in [0,1]" in errors
    assert "generation_vs_ranking_gap must be a bare float" in errors
    assert "winner_generated_count must be bare int" in errors
    assert "metric_wired_into_capstone must be object" in errors
    assert "leaderboard_submission must be false" in errors
    assert "tests_added_pass must be object" in errors
    assert "preconditions_checked must be object" in errors
    assert "field_principles missing" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    bad_wiring = dict(artifact)
    bad_wiring["metric_wired_into_capstone"] = {"reported_side_by_side": []}
    assert (
        "metric_wired_into_capstone must report all co-headlines"
        in mod.validate_artifact(bad_wiring)
    )

    bad_principles = dict(artifact)
    principles = dict(artifact["field_principles"])
    principles.pop("tests_added_pass")
    bad_principles["field_principles"] = principles
    assert "missing field principle for tests_added_pass" in mod.validate_artifact(
        bad_principles
    )

    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, artifact=bad_wiring)


def test_scenario_capstone_4598_run_wrapper_validates_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-4598: run validates before writing the deliverable."""

    artifact = mod.build_artifact(
        root=tmp_path,
        registry=_registry(),
        package=_package(),
        package_path="results/fixture_package.json",
        b1_artifact=_b1_artifact(),
        winner_source_artifact=_winner_source(),
        preconditions_checked=_preconditions(),
    )
    monkeypatch.setattr(mod, "build_artifact", lambda _root: artifact)

    dry = mod.run(tmp_path, write=False)
    written = mod.run(tmp_path, write=True)

    assert dry["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    bad = dict(artifact)
    bad["winner_generated_rate"] = 0.1
    monkeypatch.setattr(mod, "build_artifact", lambda _root: bad)
    with pytest.raises(ValueError):
        mod.run(tmp_path, write=False)
