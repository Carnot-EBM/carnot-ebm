"""Tests for the Exp 4394 ARC E3 fidelity gate.

Spec refs: REQ-VERIFY-4394, SCENARIO-VERIFY-4394.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot.agentic import arc_e3_fidelity_gate as gate
from carnot.agentic.arc_e3_fidelity_gate import (
    ExperimentConfig,
    artifact_schema_errors,
    run_experiment,
)


TARGET_PRIORS = {"lp85": 5, "tu93": 4, "tn36": 7, "tr87": 6}


def _write_fake_repo(tmp_path: Path, fidelities: dict[str, float]) -> ExperimentConfig:
    registry = {
        "games": [
            {"game": game, "levels_reproduced": level}
            for game, level in TARGET_PRIORS.items()
        ]
    }
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(registry),
        encoding="utf-8",
    )
    (tmp_path / "results").mkdir()
    scorecard = [
        {
            "game": game,
            "prior_best_level": TARGET_PRIORS[game],
            "new_reproduced_level": TARGET_PRIORS[game],
            "lookahead_fidelity": fidelity,
            "lookahead_fidelity_per_round": [fidelity],
            "verifier_accuracy": fidelity,
            "verifier_accuracy_per_round": [fidelity],
            "residual_win_mechanic_gap_class": "lookahead_fidelity_below_gate",
            "world_model_path": "python/carnot/agentic/arc_game_adapters.py",
            "reproduce_result": {
                "game": game,
                "claimed_level": TARGET_PRIORS[game] + 1,
                "reached_level": TARGET_PRIORS[game],
                "reproduced": False,
            },
        }
        for game, fidelity in fidelities.items()
    ]
    (tmp_path / "results" / "experiment_4383_e3_deeper_high_headroom_lookahead.json").write_text(
        json.dumps(
            {
                "lookahead_k": 3,
                "per_target_scorecard": scorecard,
                "reproducible_total_levels": 34,
            }
        ),
        encoding="utf-8",
    )
    for game in TARGET_PRIORS:
        env_dir = tmp_path / "environment_files" / game
        env_dir.mkdir(parents=True)
        (env_dir / "placeholder").write_text("offline env present\n", encoding="utf-8")
    return ExperimentConfig.from_repo_root(tmp_path)


def test_req_verify_4394_spec_declares_fidelity_gate_contract() -> None:
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-4394" in spec
    assert "SCENARIO-VERIFY-4394" in spec
    assert "lookahead-fidelity gate" in spec
    assert "0.95" in spec


def test_scenario_verify_4394_gate_blocks_reproduction_when_fidelity_low(
    tmp_path: Path,
) -> None:
    config = _write_fake_repo(
        tmp_path,
        {"lp85": 0.833333, "tu93": 0.8, "tn36": 0.875, "tr87": 0.857143},
    )
    attempted: list[str] = []

    def reproduction_runner(game: str, _target_level: int) -> dict[str, object]:
        attempted.append(game)
        return {"game": game, "reached_level": 99, "reproduced": True}

    artifact = run_experiment(
        config,
        reproduction_runner=reproduction_runner,
        write_artifact=True,
        import_checker=lambda: {"arc_solver_kit": True, "arc_executable_world_model": True},
    )

    assert attempted == []
    assert artifact["honest_verdict"] == "complete_e3_deeper_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["verifier_is_oracle"] is True
    assert artifact_schema_errors(artifact) == []
    assert Path(artifact["artifact_path"]).exists()
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    for card in artifact["per_target_scorecard"]:
        assert card["fidelity_gate_passed"] is False
        assert card["offline_reproduced"] is False
        assert card["new_reproduced_level"] == card["prior_best_level"]
        assert Path(card["mind_studio_skill_file"]).exists()


def test_scenario_verify_4394_missing_env_blocks_one_target_and_continues(
    tmp_path: Path,
) -> None:
    config = _write_fake_repo(
        tmp_path,
        {"lp85": 0.833333, "tu93": 0.8, "tn36": 0.875, "tr87": 0.857143},
    )
    for child in (tmp_path / "environment_files" / "tu93").iterdir():
        child.unlink()

    artifact = run_experiment(
        config,
        write_artifact=False,
        import_checker=lambda: {"arc_solver_kit": True, "arc_executable_world_model": True},
    )

    statuses = {card["game"]: card["checkpoint_status"] for card in artifact["per_target_scorecard"]}
    assert len(statuses) == 4
    assert statuses["tu93"] == "blocked_offline_env_missing_tu93"
    assert statuses["lp85"] == "honest_partial_fidelity_gate_not_met"
    assert artifact["preconditions_checked"]["offline_envs"]["tu93"]["available"] is False


def test_req_verify_4394_gate_pass_counts_only_reproduced_new_level(
    tmp_path: Path,
) -> None:
    config = _write_fake_repo(
        tmp_path,
        {"lp85": 0.96, "tu93": 0.8, "tn36": 0.875, "tr87": 0.857143},
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
        import_checker=lambda: {"arc_solver_kit": True, "arc_executable_world_model": True},
    )

    cards = {card["game"]: card for card in artifact["per_target_scorecard"]}
    assert attempted == ["lp85"]
    assert cards["lp85"]["fidelity_gate_passed"] is True
    assert cards["lp85"]["offline_reproduced"] is True
    assert cards["lp85"]["new_reproduced_level"] == 6
    assert artifact["honest_verdict"] == "success_e3_deeper_lp85_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 35


def test_req_verify_4394_schema_errors_report_missing_required_fields() -> None:
    errors = artifact_schema_errors({"honest_verdict": "complete_e3_deeper_partial"})

    assert "missing:per_target_scorecard" in errors
    assert "missing:reproducibility_checksum" in errors


def test_req_verify_4394_gate_pass_without_runner_records_honest_partial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _write_fake_repo(
        tmp_path,
        {"lp85": 0.96, "tu93": 0.8, "tn36": 0.875, "tr87": 0.857143},
    )
    prior_path = tmp_path / "results" / "experiment_4383_e3_deeper_high_headroom_lookahead.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior["per_target_scorecard"][0].pop("lookahead_fidelity_per_round")
    prior["per_target_scorecard"][0].pop("verifier_accuracy_per_round")
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    monkeypatch.setattr(gate, "artifact_schema_errors", lambda _artifact: ["forced_schema_error"])

    artifact = run_experiment(
        config,
        write_artifact=False,
        import_checker=lambda: {"arc_solver_kit": True, "arc_executable_world_model": True},
    )

    cards = {card["game"]: card for card in artifact["per_target_scorecard"]}
    assert artifact["schema_errors"] == ["forced_schema_error"]
    assert cards["lp85"]["fidelity_gate_passed"] is True
    assert cards["lp85"]["offline_reproduced"] is False
    assert cards["lp85"]["lookahead_fidelity_per_round"] == [0.96]
    assert cards["lp85"]["reproduce_result"]["reason"] == "reproduction_runner_not_configured"
