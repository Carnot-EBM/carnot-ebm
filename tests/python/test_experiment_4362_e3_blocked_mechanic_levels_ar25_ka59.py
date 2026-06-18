"""Tests for Exp 4362 ar25/ka59 blocked-mechanic E3 deepen pass.

Spec refs: REQ-PHASE4-086, SCENARIO-PHASE4-086.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_4362_e3_blocked_mechanic_levels_ar25_ka59 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _row(
    game: str,
    *,
    prior: int = 1,
    reached: int = 1,
    accuracy: float = 0.5,
    advanced: bool = False,
) -> dict:
    return {
        "game": game,
        "prior_best_level": prior,
        "new_reproduced_level": reached,
        "verifier_accuracy": accuracy,
        "verifier_accuracy_per_round": [accuracy],
        "offline_reproduced": advanced,
        "reproduce_result": {
            "game": game,
            "reached_level": reached,
            "claimed_level": reached,
            "reproduced": advanced,
        },
        "plan": ["mock"],
        "checkpoint_status": "new_level_reproduced" if advanced else "honest_partial",
        "residual_gap_class": "none" if advanced else exp.NAMED_GAP_CLASSES[game],
        "world_model_path": exp.WORLD_MODEL_PATHS[game],
        "targeted_gap_lemmas": [],
    }


def test_req_phase4_086_spec_declares_exp4362_contract() -> None:
    """REQ-PHASE4-086: OpenSpec declares the two-game blocked-mechanic contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-086" in spec
    assert "SCENARIO-PHASE4-086" in spec
    assert "experiment_4362_e3_blocked_mechanic_levels_ar25_ka59.json" in spec
    assert "blocked_offline_env_missing_<game>" in spec
    assert "success_e3_ar25_ka59_<n>_reproduced" in spec
    assert "complete_e3_ar25_ka59_partial" in spec
    assert "ar25 ACTION7 undo-stack transitions" in spec
    assert "ka59 hidden StepCounter HUD register" in spec
    assert "ar25_l2_hidden_rule_delta_not_reproduced_action7_undo_stack_gap" in spec
    assert "ka59_l2_hidden_step_counter_hud_register_gap" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_086_checksum_binds_rows_paths_hashes_and_seed() -> None:
    """REQ-PHASE4-086: checksum binds scorecard, deliverables, and seed."""

    rows = [_row("ar25"), _row("ka59", reached=2, accuracy=1.0, advanced=True)]
    hashes = {"results/arc_e3/ar25/world_model.py": "a" * 64}
    base = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        path_hashes=hashes,
        random_seed=4362,
    )
    same = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        path_hashes=hashes,
        random_seed=4362,
    )
    changed = exp.compute_reproducibility_checksum(
        per_game_scorecard=[{**rows[1], "new_reproduced_level": 1}],
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        path_hashes=hashes,
        random_seed=4362,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_086_build_artifact_counts_only_new_reproduced_levels(tmp_path: Path) -> None:
    """REQ-PHASE4-086: only levels beyond the prior best count as progress."""

    ar25_model = tmp_path / exp.WORLD_MODEL_PATHS["ar25"]
    ka59_model = tmp_path / exp.WORLD_MODEL_PATHS["ka59"]
    ar25_model.parent.mkdir(parents=True, exist_ok=True)
    ka59_model.parent.mkdir(parents=True, exist_ok=True)
    ar25_model.write_text("# ar25\n", encoding="utf-8")
    ka59_model.write_text("# ka59\n", encoding="utf-8")

    rows = [_row("ar25"), _row("ka59", reached=2, accuracy=1.0, advanced=True)]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        reproducible_total_levels=33,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4362,
        duration_s=2.25,
    )

    assert artifact["honest_verdict"] == "success_e3_ar25_ka59_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 33
    assert artifact["verifier_is_oracle"] is True
    assert artifact["per_game_scorecard"][1]["residual_gap_class"] == "none"
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_086_partial_artifact_preserves_named_gap_rows(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-086: all-partial runs keep both named residual gaps."""

    rows = [_row("ar25", accuracy=0.8875), _row("ka59", accuracy=0.75)]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        reproducible_total_levels=32,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4362,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete_e3_ar25_ka59_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert [row["game"] for row in artifact["per_game_scorecard"]] == list(exp.TARGET_ORDER)
    assert artifact["per_game_scorecard"][0]["residual_gap_class"] == exp.NAMED_GAP_CLASSES["ar25"]
    assert artifact["per_game_scorecard"][1]["residual_gap_class"] == exp.NAMED_GAP_CLASSES["ka59"]
    assert isinstance(artifact["new_levels_reproduced"], int)
    assert isinstance(artifact["reproducible_total_levels"], int)
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_086_schema_errors_are_specific() -> None:
    """REQ-PHASE4-086: schema validation catches malformed gate fields and rows."""

    bad = {
        "honest_verdict": "complete_e3_ar25_ka59_partial",
        "per_game_scorecard": ["not-a-row", {"game": "ka59", "offline_reproduced": "yes"}],
        "new_levels_reproduced": "0",
        "reproducible_total_levels": {"value": 32},
        "world_model_paths": [123],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": {"value": 4362},
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "per_game_scorecard[0] must be dict" in errors
    assert "per_game_scorecard[1] missing prior_best_level" in errors
    assert "per_game_scorecard[1].offline_reproduced must be bare bool" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "per_game_scorecard must be list" in missing
    assert "field_principles missing" in missing

    non_bare = exp.artifact_schema_errors(
        {
            "honest_verdict": "complete_e3_ar25_ka59_partial",
            "per_game_scorecard": [],
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 32,
            "world_model_paths": [],
            "verifier_is_oracle": "true",
            "preconditions_checked": {},
            "random_seed": 4362,
            "reproducibility_checksum": "a" * 64,
            "field_principles": exp.REQUIRED_FIELD_PRINCIPLES,
        }
    )
    assert "verifier_is_oracle must be bare bool" in non_bare


def test_scenario_phase4_086_run_experiment_records_missing_envs_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-086: missing envs block per game without fabrication."""

    env = tmp_path / "environment_files" / "ka59"
    env.mkdir(parents=True)
    (env / "fixture").write_text("present", encoding="utf-8")
    ka59_model = tmp_path / exp.WORLD_MODEL_PATHS["ka59"]
    ka59_model.parent.mkdir(parents=True)
    ka59_model.write_text("# ka59\n", encoding="utf-8")

    calls: list[str] = []

    def fake_ka59_runner(_repo: Path, _random_seed: int, _round_budget: int) -> dict:
        calls.append("ka59")
        return _row("ka59", reached=2, accuracy=1.0, advanced=True)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "ka59", fake_ka59_runner)

    artifact = exp.run_experiment(random_seed=4362, round_budget=1)

    assert calls == ["ka59"]
    assert artifact["honest_verdict"] == "success_e3_ar25_ka59_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert [row["checkpoint_status"] for row in artifact["per_game_scorecard"]] == [
        "blocked_offline_env_missing_ar25",
        "new_level_reproduced",
    ]
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_req_phase4_086_prior_artifact_rows_keep_l1_as_partial(tmp_path: Path) -> None:
    """REQ-PHASE4-086: existing L1 artifacts do not count as L2 progress."""

    ar25_result = tmp_path / "results" / "experiment_4339_e3_explore_verify_plan_ar25.json"
    ka59_result = tmp_path / "results" / "experiment_4350_e3_explore_verify_plan_ka59.json"
    ar25_result.parent.mkdir(parents=True)
    ar25_result.write_text(
        json.dumps(
            {
                "verifier_accuracy_per_round": [0.8875],
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "plan_executed_detail": {"plan_result": {"solution": ["3"]}},
            }
        ),
        encoding="utf-8",
    )
    ka59_result.write_text(
        json.dumps(
            {
                "verifier_accuracy_per_round": [0.5625, 0.6375],
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "plan_executed_detail": {"plan_result": {"solution": ["4"]}},
            }
        ),
        encoding="utf-8",
    )

    ar25 = exp._prior_artifact_row(tmp_path, "ar25")
    ka59 = exp._prior_artifact_row(tmp_path, "ka59")

    assert ar25["verifier_accuracy"] == 0.8875
    assert ar25["new_reproduced_level"] == 1
    assert ar25["offline_reproduced"] is False
    assert ar25["residual_gap_class"] == exp.NAMED_GAP_CLASSES["ar25"]
    assert ka59["verifier_accuracy"] == 0.6375
    assert ka59["new_reproduced_level"] == 1
    assert ka59["offline_reproduced"] is False
    assert ka59["residual_gap_class"] == exp.NAMED_GAP_CLASSES["ka59"]

    missing = exp._prior_artifact_row(tmp_path / "empty_repo", "ar25")
    assert missing["checkpoint_status"] == "honest_partial_prior_artifact_missing"
    assert missing["verifier_accuracy"] == 0.0
    assert missing["plan"] == []


def test_scenario_phase4_086_gap_writer_replaces_existing_entry(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-086: partial residual gaps are stable and auditable."""

    gap_path = tmp_path / "ops" / "verifier_gaps.md"
    exp._write_gap(
        gap_path,
        game="ar25",
        best_accuracy=0.8,
        residual_gap_class="first",
        checksum="a" * 64,
    )
    exp._write_gap(
        gap_path,
        game="ar25",
        best_accuracy=0.9,
        residual_gap_class="second",
        checksum="b" * 64,
    )

    text = gap_path.read_text(encoding="utf-8")
    assert "Best verifier accuracy: 0.9000" in text
    assert "`second`" in text
    assert "`first`" not in text


def test_req_phase4_086_registry_total_parsing(tmp_path: Path) -> None:
    """REQ-PHASE4-086: registry total parsing is monotonic and optional."""

    assert exp._registry_total(tmp_path) is None

    registry = tmp_path / exp.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text("reproducible_total_levels: 41\n", encoding="utf-8")
    assert exp._registry_total(tmp_path) == 41

    registry.write_text("games: []\n", encoding="utf-8")
    assert exp._registry_total(tmp_path) is None


def test_scenario_phase4_086_run_experiment_writes_partial_gap_and_schema_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-086: partial rows write named gaps and schema failures abort."""

    env = tmp_path / "environment_files" / "ar25"
    env.mkdir(parents=True)
    (env / "fixture").write_text("present", encoding="utf-8")
    ar25_model = tmp_path / exp.WORLD_MODEL_PATHS["ar25"]
    ar25_model.parent.mkdir(parents=True)
    ar25_model.write_text("# ar25\n", encoding="utf-8")

    def fake_ar25_runner(_repo: Path, _random_seed: int, _round_budget: int) -> dict:
        return _row("ar25", accuracy=0.9, advanced=False)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "ar25", fake_ar25_runner)

    artifact = exp.run_experiment(random_seed=4362, round_budget=1)

    assert artifact["honest_verdict"] == "complete_e3_ar25_ka59_partial"
    assert (tmp_path / exp.GAP_RELATIVE_PATH).exists()
    assert exp.NAMED_GAP_CLASSES["ar25"] in (tmp_path / exp.GAP_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="Exp4362 artifact schema errors"):
        exp.run_experiment(random_seed=4362, round_budget=1)
