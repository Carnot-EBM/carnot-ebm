"""Tests for Exp 4424 single solved-game lookahead repair.

Spec refs: REQ-PHASE4-4424, SCENARIO-PHASE4-4424.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4424_deeper_solved_game as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _fixture_registry(root: Path, prior: int = 1) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "environment_files" / "sc25" / "dummy").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "games": [
                    {
                        "game": "sc25",
                        "levels_reproduced": prior,
                        "reproducibility": "reproduced",
                    }
                ],
                "reproducible_total_levels": 35,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _passing_check() -> dict[str, object]:
    return {
        "game": "sc25",
        "name": "l2_first_cast_hud_cleanup",
        "transition": "sc25:L2:first_cast_hud_cleanup",
        "mechanic": "hud_cleanup_on_first_l2_cast",
        "derived_from_rollout_trace": True,
        "passed": True,
        "expected_runs": [[0, 62, 64, 0]],
        "observed_runs": [[0, 62, 64, 0]],
    }


def _failing_check() -> dict[str, object]:
    return {
        "game": "sc25",
        "name": "l2_complete_route_reproduction",
        "transition": "sc25:L2:complete_route_reproduction",
        "mechanic": "plan_reaches_target_level",
        "derived_from_rollout_trace": False,
        "passed": False,
        "gap_class": "sc25_l2_route_search_still_missing_after_hud_cleanup",
        "expected": "arc_solver_kit.reproduce reaches L2",
        "observed": "prior L1 path still reaches only L1",
    }


def test_req_phase4_4424_spec_declares_required_artifact_fields() -> None:
    """REQ-PHASE4-4424: OpenSpec names the exact artifact and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-4424" in spec
    assert "SCENARIO-PHASE4-4424" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_4424_sc25_l2_hud_cleanup_mechanic_unit_passes() -> None:
    """REQ-PHASE4-4424: the L2 first-cast HUD cleanup has an executable unit check."""

    check = mod.sc25_l2_hud_cleanup_check(REPO)

    assert check["transition"] == "sc25:L2:first_cast_hud_cleanup"
    assert check["mechanic"] == "hud_cleanup_on_first_l2_cast"
    assert check["passed"] is True
    assert check["expected_runs"] == check["observed_runs"]
    assert [0, 62, 64, 0] in check["expected_runs"]
    assert [49, 29, 32, 14] in check["expected_runs"]


def test_scenario_phase4_4424_partial_separates_mechanic_pass_from_solve_claim(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-4424: passing mechanics do not become an ungrounded solve."""

    _fixture_registry(tmp_path, prior=1)

    def runner(root: Path, prior_best: int, target_level: int) -> dict[str, object]:
        return {
            "game": "sc25",
            "claimed_level": target_level,
            "reached_level": prior_best,
            "reproduced": False,
        }

    artifact = mod.run_experiment(
        root=tmp_path,
        reproduction_runner=runner,
        mechanic_checks=[_passing_check(), _failing_check()],
    )

    assert artifact["honest_verdict"] == "complete: sc25_L2_hud_cleanup_fixed_reproduction_gap"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 1
    assert artifact["prior_best_level"] == 1
    assert artifact["target_level"] == 2
    assert artifact["per_mechanic_tests"][0]["passed"] is True
    assert artifact["per_mechanic_tests"][1]["passed"] is False
    assert artifact["residual_failing_mechanic"] == "sc25_l2_route_search_still_missing_after_hud_cleanup"
    assert artifact["verifier_is_oracle"] is True
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))["offline_reproduced"] is False


def test_req_phase4_4424_success_requires_reproduce_beyond_prior(tmp_path: Path) -> None:
    """REQ-PHASE4-4424: +1 only counts when the reproduction runner reaches target."""

    _fixture_registry(tmp_path, prior=1)

    def runner(root: Path, prior_best: int, target_level: int) -> dict[str, object]:
        return {
            "game": "sc25",
            "claimed_level": target_level,
            "reached_level": target_level,
            "reproduced": True,
        }

    artifact = mod.run_experiment(
        root=tmp_path,
        reproduction_runner=runner,
        mechanic_checks=[_passing_check()],
    )

    assert artifact["honest_verdict"] == "success: sc25_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["residual_failing_mechanic"] == "none"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_phase4_4424_schema_errors_are_specific(tmp_path: Path) -> None:
    """REQ-PHASE4-4424: required artifact fields have typed validation."""

    _fixture_registry(tmp_path, prior=1)

    def runner(root: Path, prior_best: int, target_level: int) -> dict[str, object]:
        return {"game": "sc25", "claimed_level": target_level, "reached_level": prior_best, "reproduced": False}

    artifact = mod.run_experiment(
        root=tmp_path,
        reproduction_runner=runner,
        mechanic_checks=[_passing_check(), _failing_check()],
    )
    broken = {
        **artifact,
        "offline_reproduced": "false",
        "reproduced_levels": "1",
        "verifier_is_oracle": False,
        "honest_verdict": "maybe",
        "per_mechanic_tests": [],
        "reproducibility_checksum": "not-hex",
    }

    errors = mod.artifact_schema_errors(broken)

    assert "offline_reproduced must be a bool" in errors
    assert "reproduced_levels must be an int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "per_mechanic_tests must include at least one row" in errors
    assert "reproducibility_checksum must be a sha256 hex string" in errors

    missing_errors = mod.artifact_schema_errors({})
    assert "missing offline_reproduced" in missing_errors

    residual = mod.reproduction_gap_check(
        {"game": "sc25", "claimed_level": 2, "reached_level": 1, "reproduced": False},
        prior_best_level=1,
        target_level=2,
    )
    assert residual["passed"] is False
    assert residual["gap_class"] == "sc25_l2_route_search_still_missing_after_hud_cleanup"

    try:
        mod.write_artifact(tmp_path, broken)
    except ValueError as exc:
        assert "offline_reproduced must be a bool" in str(exc)
    else:  # pragma: no cover - write_artifact must reject invalid artifacts
        raise AssertionError("invalid artifact unexpectedly wrote")
