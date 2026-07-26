"""Tests for Exp5927 coordinate-router progress qualification.

Spec refs: REQ-ARC-FCP-5927, SCENARIO-ARC-FCP-5927-POWERED-PROGRESS-CORPUS,
SCENARIO-ARC-FCP-5927-CONTROLS-AND-LEAKAGE,
SCENARIO-ARC-FCP-5927-COMMITTED-OUTCOME-HOOK,
SCENARIO-ARC-FCP-5927-NO-PROMOTION-WITHOUT-GATE.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5927_coordinate_router_progress_qualification as mod
from carnot.agentic.arc_click_target_features import CLICK_TARGET_FEATURE_DIM


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-human-replay-frame-change/spec.md"


def _feature(signal: float, state: int, position: int) -> list[float]:
    vector = [0.0] * CLICK_TARGET_FEATURE_DIM
    vector[0] = float(signal)
    vector[1] = float((state % 5) / 5.0)
    vector[2] = float((position % 7) / 7.0)
    vector[19] = float(position / 10.0)
    vector[20] = float(state / 10.0)
    return vector


def _corpus(
    *,
    n_games: int = 3,
    n_states: int = 6,
    per_state: int = 10,
    positives_per_state: int = 2,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game_i in range(n_games):
        game = f"g{game_i:02d}"
        for state in range(n_states):
            for position in range(per_state):
                validated_progress = position < positives_per_state
                raw_frame_change = position < 6
                ui_animation = position == 6
                state_novelty = raw_frame_change and position in {0, 2, 4}
                rows.append(
                    {
                        "game": game,
                        "state_index": state,
                        "row_id": f"{game}-s{state}-p{position}",
                        "x": position,
                        "y": state,
                        "salience_rank": (position * 7 + state + game_i) % per_state,
                        "raw_frame_change": raw_frame_change,
                        "ui_animation": ui_animation,
                        "state_novelty": state_novelty,
                        "validated_progress": validated_progress,
                        "action_legal": True,
                        "features": _feature(1.0 if validated_progress else 0.0, state, position),
                        "blind_action_id": 6,
                    }
                )
    return rows


def _exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.TASK_OWNED_COMMANDS}


def test_req_5927_spec_declares_required_fields_and_principles() -> None:
    section = SPEC.read_text(encoding="utf-8")
    section = section[section.index("### REQ-ARC-FCP-5927") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-ARC-FCP-5927",
        "SCENARIO-ARC-FCP-5927-POWERED-PROGRESS-CORPUS",
        "SCENARIO-ARC-FCP-5927-CONTROLS-AND-LEAKAGE",
        "SCENARIO-ARC-FCP-5927-COMMITTED-OUTCOME-HOOK",
        "SCENARIO-ARC-FCP-5927-NO-PROMOTION-WITHOUT-GATE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_5927_manifest_separates_labels_and_power_gate() -> None:
    rows = mod.normalize_corpus_rows(_corpus())
    manifest = mod.games_states_rows_and_label_manifest(rows)
    power = mod.hard_progress_power_gate(rows)

    assert manifest["games"] == ["g00", "g01", "g02"]
    assert manifest["row_count"] == 180
    assert manifest["raw_frame_change_rows"] == 108
    assert manifest["ui_animation_only_rows"] == 18
    assert manifest["state_novelty_rows"] == 54
    assert manifest["validated_progress_rows"] == 36
    assert power["hard_progress_positive_count"] == 36
    assert power["powered"] is True

    underpowered_rows = mod.normalize_corpus_rows(_corpus(n_games=1, n_states=5))
    underpowered = mod.hard_progress_power_gate(underpowered_rows)
    assert underpowered["hard_progress_positive_count"] == 10
    assert underpowered["powered"] is False


def test_scenario_5927_controls_interval_and_random_sanity_on_powered_fixture() -> None:
    rows = mod.normalize_corpus_rows(_corpus())
    metrics = mod.evaluate_progress_controls(rows, seed=5927, n_bootstrap=300)

    controls = metrics["coordinate_static_blind_step_and_random_controls"]
    assert set(controls) == {
        "coordinate",
        "static_salience",
        "blind_action_id",
        "step_index",
        "random",
    }
    assert controls["blind_action_id"]["distinct_scores_per_state_max"] == 1
    assert controls["random"]["distinct_scores"] > 30

    within = metrics["within_state_and_leave_state_out_metrics"]["within_state"]
    leave = metrics["within_state_and_leave_state_out_metrics"]["leave_state_out"]
    assert within["blind_action_id"]["auroc"] == pytest.approx(0.5)
    assert within["step_index"]["auroc"] == pytest.approx(0.5)
    assert leave["coordinate"]["n_scored_rows"] > 0
    assert leave["coordinate"]["auroc"] > leave["static_salience"]["auroc"]

    delta = metrics["coordinate_over_static_delta_and_interval"]
    assert delta["delta"] > 0.0
    assert delta["ci95"][0] > 0.0
    assert metrics["random_control_sanity"]["passed"] is True
    assert (
        metrics["cross_game_isolation_and_leakage_checks"]["cross_game_checkpoint_loaded"] is False
    )
    assert (
        metrics["cross_game_isolation_and_leakage_checks"][
            "future_outcomes_used_as_current_features"
        ]
        is False
    )


def test_scenario_5927_artifact_ready_and_underpowered_paths_validate(tmp_path: Path) -> None:
    ready = mod.build_qualification_artifact(
        rows=mod.normalize_corpus_rows(_corpus()),
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_exit_codes=_exit_codes(),
    )
    assert mod.validate_artifact(ready) is True
    assert ready["status"] == "complete_ready"
    assert ready["honest_verdict"].startswith("complete_ready:")
    assert ready["coordinate_router_progress_ready_score"] == 1.0
    assert ready["default_enabled"] is False
    assert ready["cross_game_checkpoint_loaded"] is False
    assert ready["online_within_game_only"] is True
    assert ready["solve_provenance"] == "development_proxy"
    assert ready["no_level_solve_or_registry_update"]["no_level_solve_claimed"] is True
    assert ready["no_level_solve_or_registry_update"]["registry_update_performed"] is False
    assert ready["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert ready["verifier_is_oracle"] is True

    underpowered = mod.build_qualification_artifact(
        rows=mod.normalize_corpus_rows(_corpus(n_games=1, n_states=5)),
        output_path=tmp_path / "underpowered.json",
        duration_s=0.0,
        test_exit_codes=_exit_codes(),
    )
    assert mod.validate_artifact(underpowered) is True
    assert underpowered["status"] == "complete_underpowered"
    assert underpowered["honest_verdict"].startswith("complete_underpowered:")
    assert underpowered["coordinate_router_progress_ready_score"] == 0.0
    assert underpowered["default_enabled"] is False


def test_scenario_5927_artifact_validation_fails_closed() -> None:
    artifact = mod.build_qualification_artifact(
        rows=mod.normalize_corpus_rows(_corpus()),
        duration_s=0.0,
        test_exit_codes=_exit_codes(),
    )

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    bad_checkpoint = deepcopy(artifact)
    bad_checkpoint["cross_game_checkpoint_loaded"] = True
    bad_checkpoint["reproducibility_checksum"] = mod.reproducibility_checksum(bad_checkpoint)
    with pytest.raises(ValueError, match="cross_game_checkpoint_loaded"):
        mod.validate_artifact(bad_checkpoint)

    bad_default = deepcopy(artifact)
    bad_default["default_enabled"] = True
    bad_default["reproducibility_checksum"] = mod.reproducibility_checksum(bad_default)
    with pytest.raises(ValueError, match="default_enabled"):
        mod.validate_artifact(bad_default)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete_ready: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_score = deepcopy(artifact)
    bad_score["coordinate_router_progress_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_json({"wrong": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)
