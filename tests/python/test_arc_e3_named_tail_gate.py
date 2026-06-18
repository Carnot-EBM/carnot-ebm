"""Tests for Exp 4395 named-tail ARC E3 fidelity gate.

Spec refs: REQ-PHASE4-4395, SCENARIO-PHASE4-4395.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.agentic import arc_e3_named_tail_gate as gate
from carnot.agentic.arc_e3_named_tail_gate import (
    ExperimentConfig,
    artifact_schema_errors,
    run_experiment,
)


TARGET_PRIORS = {"ar25": 1, "ka59": 1, "ft09": 1}


def _prior_card(game: str, fidelity: float) -> dict[str, object]:
    card: dict[str, object] = {
        "game": game,
        "prior_best_level": 1,
        "new_reproduced_level": 1,
        "lookahead_fidelity": fidelity,
        "lookahead_fidelity_per_round": [fidelity],
        "verifier_accuracy": fidelity,
        "verifier_accuracy_per_round": [fidelity],
        "active_transitions_collected": 12,
        "active_dataset_sha256": f"{game}_active_hash",
        "world_model_path": f"results/arc_e3/{game}/world_model.py",
        "plan": [],
        "reproduce_result": {
            "game": game,
            "claimed_level": 2,
            "reached_level": 1,
            "reproduced": False,
        },
    }
    if game == "ka59":
        card["targeted_gap_lemmas"] = [
            {
                "action": 6,
                "hud_count_before": 60,
                "hud_count_after": 59,
                "hud_count_predicted": 59,
                "changed_cells": 1,
            },
            {
                "action": 2,
                "hud_count_before": 59,
                "hud_count_after": 58,
                "hud_count_predicted": 57,
                "changed_cells": 19,
            },
        ]
    return card


def _write_fake_repo(tmp_path: Path, fidelities: dict[str, float]) -> ExperimentConfig:
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        "reproducible_total_levels: 34\n",
        encoding="utf-8",
    )
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09.json").write_text(
        json.dumps(
            {
                "per_game_scorecard": [
                    _prior_card(game, fidelity) for game, fidelity in fidelities.items()
                ],
                "new_levels_reproduced": 0,
                "reproducible_total_levels": 34,
            }
        ),
        encoding="utf-8",
    )
    for game in TARGET_PRIORS:
        env_dir = tmp_path / "environment_files" / game
        env_dir.mkdir(parents=True)
        (env_dir / "placeholder").write_text("offline env present\n", encoding="utf-8")
        wm_dir = tmp_path / "results" / "arc_e3" / game
        wm_dir.mkdir(parents=True)
        (wm_dir / "world_model.py").write_text("# fake model\n", encoding="utf-8")
    return ExperimentConfig.from_repo_root(tmp_path)


def _imports_ok() -> dict[str, bool]:
    return {
        "arc_solver_kit": True,
        "arc_executable_world_model": True,
    }


def test_req_phase4_4395_spec_declares_named_tail_contract() -> None:
    spec = Path("openspec/capabilities/phase4_active_inference/spec.md").read_text(encoding="utf-8")

    assert "REQ-PHASE4-4395" in spec
    assert "SCENARIO-PHASE4-4395" in spec
    assert "object-relevance discriminator" in spec
    assert "per-register fidelity" in spec


def test_scenario_phase4_4395_blocks_planning_below_register_fidelity_gate(
    tmp_path: Path,
) -> None:
    config = _write_fake_repo(
        tmp_path,
        {"ar25": 0.733333, "ka59": 0.112281, "ft09": 0.347518},
    )
    attempted: list[str] = []

    def reproduction_runner(game: str, _target_level: int) -> dict[str, object]:
        attempted.append(game)
        return {"game": game, "reached_level": 2, "reproduced": True}

    artifact = run_experiment(
        config,
        reproduction_runner=reproduction_runner,
        write_artifact=True,
        import_checker=_imports_ok,
    )

    assert attempted == []
    assert artifact["honest_verdict"] == "complete_e3_ar25_ka59_ft09_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["verifier_is_oracle"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact_schema_errors(artifact) == []
    assert Path(artifact["artifact_path"]).exists()
    for card in artifact["per_game_scorecard"]:
        assert card["fidelity_gate_passed"] is False
        assert card["offline_reproduced"] is False
        assert card["new_reproduced_level"] == card["prior_best_level"]


def test_scenario_phase4_4395_ka59_records_object_relevance_outcome(
    tmp_path: Path,
) -> None:
    config = _write_fake_repo(
        tmp_path,
        {"ar25": 0.733333, "ka59": 0.112281, "ft09": 0.347518},
    )

    artifact = run_experiment(config, write_artifact=False, import_checker=_imports_ok)

    cards = {card["game"]: card for card in artifact["per_game_scorecard"]}
    discriminator = cards["ka59"]["object_relevance_discriminator"]
    assert discriminator["provenance_commits"] == ["f0b078247", "6fba583c7"]
    assert discriminator["blocker_class"] == "object_relevance_not_clicks_or_multi_object_push"
    assert discriminator["selected_object_hypothesis"] == "agent_plus_second_movable_block"
    assert discriminator["planning_allowed"] is False
    assert cards["ka59"]["residual_gap_class"] == "ka59_l2_object_relevance_step_counter_hud_register_gap"


def test_req_phase4_4395_gate_pass_counts_only_reproduced_new_level(
    tmp_path: Path,
) -> None:
    config = _write_fake_repo(
        tmp_path,
        {"ar25": 0.96, "ka59": 0.112281, "ft09": 0.347518},
    )
    attempted: list[str] = []

    def reproduction_runner(game: str, target_level: int) -> dict[str, object]:
        attempted.append(game)
        return {
            "game": game,
            "claimed_level": target_level,
            "reached_level": target_level,
            "reproduced": True,
        }

    artifact = run_experiment(
        config,
        reproduction_runner=reproduction_runner,
        write_artifact=False,
        import_checker=_imports_ok,
    )

    cards = {card["game"]: card for card in artifact["per_game_scorecard"]}
    assert attempted == ["ar25"]
    assert cards["ar25"]["offline_reproduced"] is True
    assert cards["ar25"]["new_reproduced_level"] == 2
    assert artifact["honest_verdict"] == "success_e3_ar25_ka59_ft09_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 35
    assert artifact["reproducibility_checksum"].startswith("sha256:")


def test_scenario_phase4_4395_missing_env_blocks_one_game_and_continues(
    tmp_path: Path,
) -> None:
    config = _write_fake_repo(
        tmp_path,
        {"ar25": 0.96, "ka59": 0.96, "ft09": 0.96},
    )
    for child in (tmp_path / "environment_files" / "ft09").iterdir():
        child.unlink()

    artifact = run_experiment(config, write_artifact=False, import_checker=_imports_ok)

    statuses = {card["game"]: card["checkpoint_status"] for card in artifact["per_game_scorecard"]}
    assert statuses["ft09"] == "blocked_offline_env_missing_ft09"
    assert statuses["ar25"] == "honest_partial_reproduction_gate_not_proven"
    assert artifact["preconditions_checked"]["offline_envs"]["ft09"]["available"] is False
    config.registry_path.unlink()
    assert gate.read_registry_total(config.registry_path) == 34


def test_req_phase4_4395_schema_errors_report_missing_required_fields() -> None:
    errors = artifact_schema_errors({"honest_verdict": "complete_e3_ar25_ka59_ft09_partial"})

    assert "missing:per_game_scorecard" in errors
    assert "missing:reproducibility_checksum" in errors


def test_req_phase4_4395_schema_errors_report_order_and_ka59_discriminator() -> None:
    errors = artifact_schema_errors(
        {
            "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
            "per_game_scorecard": [{"game": "ka59"}],
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "world_model_paths": [
                "results/arc_e3/ar25/world_model.py",
                "results/arc_e3/ka59/world_model.py",
                "results/arc_e3/ft09/world_model.py",
            ],
            "verifier_is_oracle": True,
            "preconditions_checked": {},
            "random_seed": 4395,
            "reproducibility_checksum": "sha256:test",
        }
    )

    assert "per_game_scorecard_order_wrong" in errors
    assert "ka59_missing_object_relevance_discriminator" in errors


def test_req_phase4_4395_run_records_schema_errors_when_validator_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _write_fake_repo(
        tmp_path,
        {"ar25": 0.733333, "ka59": 0.112281, "ft09": 0.347518},
    )
    monkeypatch.setattr(gate, "artifact_schema_errors", lambda _artifact: ["forced_schema_error"])

    artifact = gate.run_experiment(config, write_artifact=False, import_checker=_imports_ok)

    assert artifact["schema_errors"] == ["forced_schema_error"]
