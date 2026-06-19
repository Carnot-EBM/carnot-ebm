"""Tests for Exp 4432 ARC leave-one-out generic-solver benchmark.

Spec refs: REQ-REPORT-4432, SCENARIO-REPORT-4432.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4432_loo_generic_solve_benchmark as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _registry_payload() -> dict[str, Any]:
    games = []
    for game in ("tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09"):
        games.append(
            {
                "game": game,
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
                "mechanic_class": f"{game}_mechanic",
                "solver": f"own recipe for {game}",
                "win_condition": f"{game} target mechanic",
                "action_model": "generic actions",
                "gotchas": [f"{game} private gotcha"],
            }
        )
    games.append(
        {
            "game": "vc33",
            "reproducibility": "unsolved",
            "levels_reproduced": 0,
            "solver": "not counted",
        }
    )
    return {
        "schema_version": 1,
        "updated": "2026-06-19",
        "general_gotchas": [{"id": "level_on_frame_not_game"}],
        "games": games,
        "reproducible_total_levels": 7,
        "reproducible_total_games": 7,
    }


def _write_fixture_repo(root: Path) -> str:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    for game in ("tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09", "vc33"):
        (root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    registry_text = yaml.safe_dump(_registry_payload(), sort_keys=False)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(registry_text, encoding="utf-8")
    return registry_text


def _ok_preconditions() -> dict[str, Any]:
    return {
        "offline_env_files_present": True,
        "offline_env_file_count": 8,
        "arc_solver_kit_import": True,
        "arc_solve_learning_import": True,
        "qwen_gguf_cached": True,
        "ok": True,
    }


def _route(target: str, registry_view: Mapping[str, Any], _root: Path) -> dict[str, Any]:
    games = [
        row["game"]
        for row in registry_view.get("games", [])
        if isinstance(row, Mapping) and row.get("reproducibility") == "reproduced"
    ]
    assert target not in games
    routed_to = games[0] if games else ""
    return {
        "target_game": target,
        "recommended": [{"game": routed_to, "solver": f"other recipe {routed_to}"}] if routed_to else [],
        "general_gotchas": registry_view.get("general_gotchas", []),
    }


def _attempt(target: str, _root: Path, _route_result: Mapping[str, Any]) -> dict[str, Any]:
    solved = target in {"tr87", "sc25"}
    return {
        "game": target,
        "mode": "standing_arc_loop_adapter_withheld",
        "solution_labels": [json.dumps({"action": 4})] if solved else [],
        "reproduction_gate": {
            "game": target,
            "claimed_level": 1,
            "reached_level": 1 if solved else 0,
            "reproduced": solved,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "offline_reproduced": solved,
        "reproduced_levels": 1 if solved else 0,
    }


def test_req_report_4432_spec_declares_leave_one_out_contract() -> None:
    """REQ-REPORT-4432: OpenSpec names the LOO benchmark and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4432" in spec
    assert "SCENARIO-REPORT-4432" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "leave-one-out" in spec
    assert "arc_solve_learning.recommend_approach" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4432_selects_reproduced_registry_targets_and_excludes_unsolved(tmp_path: Path) -> None:
    """REQ-REPORT-4432: held-out set comes from reproduced registry rows, not unsolved entries."""

    _write_fixture_repo(tmp_path)
    registry = mod.load_registry(tmp_path)

    selected = mod.select_heldout_games(registry, env_games=mod.environment_games(tmp_path))

    assert selected == ["tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09"]
    assert "vc33" not in selected
    assert len(selected) >= 6
    with pytest.raises(ValueError, match="at least 6"):
        mod.select_heldout_games({"games": registry["games"][:5]}, env_games=set(selected))


def test_scenario_report_4432_runs_loo_without_mutating_registry_and_counts_only_reproduced(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4432: own recipes are withheld and only reproduced attempts count."""

    registry_before = _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        route_fn=_route,
        attempt_fn=_attempt,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 10.0,
    )

    assert (tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8") == registry_before
    assert artifact["honest_verdict"] == "complete: generic_loo_solve_count_2_of_7_gate_passed"
    assert artifact["generic_loo_solve_count"] == 2
    assert artifact["offline_reproduced"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["loo_gate_passed"] is True
    assert artifact["heldout_games"] == ["tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09"]
    assert artifact["per_game"][0] == {
        "game": "tr87",
        "solved_without_own_recipe": True,
        "routed_to": "tu93",
        "residual_delta": "none",
    }
    failed = [row for row in artifact["per_game"] if not row["solved_without_own_recipe"]]
    assert failed
    assert all(row["residual_delta"].startswith("missing_") for row in failed)
    assert {gap["game"] for gap in artifact["missing_verifier_gaps"]} == {
        row["game"] for row in failed
    }
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["generic_loo_solve_count"] == 2
    assert len(written["reproducibility_checksum"]) == 64


def test_req_report_4432_precondition_block_writes_honest_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-4432: missing offline resources stop before solve attempts."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        route_fn=lambda game, registry, root: calls.append(game) or {},
        attempt_fn=lambda game, root, route: calls.append(game) or {},
        preconditions_checked={
            "offline_env_files_present": False,
            "arc_solver_kit_import": True,
            "arc_solve_learning_import": True,
            "qwen_gguf_cached": True,
            "ok": False,
        },
        now=lambda: 1.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_offline_env_files"
    assert artifact["generic_loo_solve_count"] == 0
    assert artifact["per_game"] == []
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4432_model_cache_blocks_only_llm_induction_targets(tmp_path: Path) -> None:
    """REQ-REPORT-4432: uncached model marks LLM-induction rows without stopping CPU-only rows."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        route_fn=_route,
        attempt_fn=_attempt,
        preconditions_checked={**_ok_preconditions(), "qwen_gguf_cached": False},
        llm_induction_games={"ka59"},
        now=lambda: 2.0,
    )

    ka59 = next(row for row in artifact["per_game"] if row["game"] == "ka59")
    assert ka59 == {
        "game": "ka59",
        "solved_without_own_recipe": False,
        "routed_to": "",
        "residual_delta": "blocked_model_not_cached",
    }
    assert artifact["generic_loo_solve_count"] == 2
    assert "blocked_model_not_cached" in {
        gap["residual_delta"] for gap in artifact["missing_verifier_gaps"]
    }
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4432_schema_rejects_fabricated_or_malformed_results(tmp_path: Path) -> None:
    """REQ-REPORT-4432: schema catches non-bare counts, ungated claims, and bad prefixes."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        route_fn=_route,
        attempt_fn=_attempt,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 10.0,
    )

    bad = {
        **artifact,
        "honest_verdict": "partial: not terminal for this benchmark",
        "generic_loo_solve_count": "2",
        "offline_reproduced": "yes",
        "verifier_is_oracle": False,
        "reproducibility_checksum": "bad",
        "per_game": [{"game": "x", "solved_without_own_recipe": True, "routed_to": "y"}],
        "missing_verifier_gaps": {},
        "random_seed": "4432",
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must be terminal-prefixed for Exp 4432" in errors
    assert "generic_loo_solve_count must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "per_game[0] missing residual_delta" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "random_seed must be bare int" in errors

    fabricated = {
        **artifact,
        "attempts": [
            {
                "game": "fake",
                "solved_without_own_recipe": True,
                "reproduction_gate": {"reproduced": False, "reached_level": 0},
            }
        ],
    }
    assert "solved attempts must have reproduced gate evidence" in mod.artifact_schema_errors(fabricated)
    with pytest.raises(ValueError, match="generic_loo_solve_count"):
        mod.write_artifact(tmp_path, bad)


def test_req_report_4432_defensive_helpers_and_blocked_target_count(tmp_path: Path) -> None:
    """REQ-REPORT-4432: defensive paths stay honest without live ARC resources."""

    assert mod._as_int("bad") == 0
    assert mod.load_registry(tmp_path) == {"games": []}
    assert mod.environment_games(tmp_path) == set()
    assert mod._registry_games({"games": "bad"}) == []
    assert mod._first_precondition_miss({"offline_env_files_present": True}) == "arc_solver_kit_import"
    assert (
        mod._first_precondition_miss(
            {"offline_env_files_present": True, "arc_solver_kit_import": True}
        )
        == "arc_solve_learning_import"
    )

    registry = {
        "games": [
            {"game": "new", "reproducibility": "reproduced", "levels_reproduced": 1, "mechanic_class": "Odd Mechanic"},
            {"game": "wm", "reproducibility": "reproduced", "levels_reproduced": 1, "world_model": "wm.py"},
            {"game": "plain", "reproducibility": "reproduced", "levels_reproduced": 1},
        ]
    }
    assert mod.residual_delta_for("new", registry, solved=False) == "missing_odd_mechanic_verifier_or_primitive"
    assert mod.residual_delta_for("wm", registry, solved=False) == "missing_executable_world_model_transfer"
    assert mod.residual_delta_for("plain", registry, solved=False) == "missing_generic_goal_discriminator"
    assert mod.routed_to({"recommended": []}) == ""

    primitive = tmp_path / "scripts" / "arc_loop_solve.py"
    primitive.parent.mkdir(parents=True, exist_ok=True)
    primitive.write_text("print('x')\n", encoding="utf-8")
    assert "scripts/arc_loop_solve.py" in mod._primitive_hashes(tmp_path)

    target_block = mod.run(
        root=tmp_path,
        route_fn=_route,
        attempt_fn=_attempt,
        preconditions_checked=_ok_preconditions(),
        now=lambda: 3.0,
    )
    assert target_block["honest_verdict"] == "blocked_reproducible_target_count"
    assert mod.artifact_schema_errors(target_block) == []

    malformed = {
        "honest_verdict": 5,
        "generic_loo_solve_count": 1,
        "per_game": "bad",
        "offline_reproduced": True,
        "missing_verifier_gaps": [],
        "random_seed": mod.RANDOM_SEED,
        "reproducibility_checksum": "0" * 64,
        "verifier_is_oracle": True,
        "attempts": "bad",
    }
    errors = mod.artifact_schema_errors(malformed)
    assert "honest_verdict must be terminal-prefixed for Exp 4432" in errors
    assert "per_game must be list" in errors
    assert "attempts must be list" in errors
    assert "missing honest_verdict" in mod.artifact_schema_errors({})

    malformed_rows = {
        **malformed,
        "honest_verdict": "complete: malformed",
        "per_game": ["bad", {"game": "x", "solved_without_own_recipe": "yes", "routed_to": "", "residual_delta": "x"}],
        "attempts": [],
    }
    row_errors = mod.artifact_schema_errors(malformed_rows)
    assert "per_game[0] must be dict" in row_errors
    assert "per_game[1].solved_without_own_recipe must be bare bool" in row_errors
    assert "generic_loo_solve_count must match solved per_game rows" in row_errors
