"""Tests for Exp 4841 object-identity perception probe artifact.

Spec refs: REQ-ARC-WMTE-4841,
SCENARIO-ARC-WMTE-4841-REAL-FRAME-CORRESPONDENCE,
SCENARIO-ARC-WMTE-4841-TU93-POSITIVE-CONTROL,
SCENARIO-ARC-WMTE-4841-LIVE-PATH-REACHABLE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_4841_object_identity_perception_probe as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _fixture_frames() -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for step, color in enumerate((2, 7, 9)):
        grid = np.zeros((10, 10), dtype=np.int16)
        grid[2:4, 2 + step : 4 + step] = color
        grid[7:9, 7:9] = 5
        frames.append(grid)
    return frames


def _frame_source(game: str) -> mod.FrameSequence | None:
    if game == "r11l":
        return None
    return mod.FrameSequence(
        game=game,
        frames=_fixture_frames(),
        source_path=f"results/arc3_live_banked_trajectories/{game}.json",
        source_kind="banked_replay",
    )


def test_req_arc_wmte_4841_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4841: OpenSpec anchors the object-identity probe contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4841",
        "SCENARIO-ARC-WMTE-4841-REAL-FRAME-CORRESPONDENCE",
        "SCENARIO-ARC-WMTE-4841-TU93-POSITIVE-CONTROL",
        "SCENARIO-ARC-WMTE-4841-LIVE-PATH-REACHABLE",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4841_run_blocks_missing_preconditions(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4841: blocked resources never fabricate correspondence scores."""

    blocked_arcade = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: False,
        frame_sequence_provider=_frame_source,
        live_path_checker=lambda _root: True,
        now=iter([1.0, 1.1]).__next__,
        write=False,
    )
    assert blocked_arcade["honest_verdict"] == "blocked_offline_arcade_missing"
    assert blocked_arcade["measured_on_real_frames"] is False
    assert blocked_arcade["per_game_correspondence"] == {}

    blocked_frames = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        frame_sequence_provider=lambda _game: None,
        live_path_checker=lambda _root: True,
        now=iter([2.0, 2.1]).__next__,
        write=False,
    )
    assert blocked_frames["honest_verdict"] == "blocked_no_offline_frames"
    assert blocked_frames["preconditions_checked"]["available_games"] == []


def test_scenario_arc_wmte_4841_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4841-REAL-FRAME-CORRESPONDENCE: artifact fields are schema-gated."""

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        frame_sequence_provider=_frame_source,
        live_path_checker=lambda _root: True,
        now=iter([10.0, 10.2]).__next__,
        write=False,
    )

    assert artifact["measured_on_real_frames"] is True
    assert set(artifact["per_game_correspondence"]) == {"lp85", "tu93"}
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert mod.artifact_schema_errors(artifact) == []

    path = mod.write_artifact(artifact, root=tmp_path)
    assert path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(path.read_text(encoding="utf-8")) == artifact

    broken = dict(artifact)
    broken["measured_on_real_frames"] = False
    broken["reproducibility_checksum"] = "sha256:bad"
    errors = mod.artifact_schema_errors(broken)
    assert "measured_on_real_frames must be true for non-blocked artifacts" in errors
    assert "reproducibility_checksum must match artifact content" in errors


def test_scenario_arc_wmte_4841_tu93_positive_control_uses_stable_tracks() -> None:
    """SCENARIO-ARC-WMTE-4841-TU93-POSITIVE-CONTROL: player-like and goal-like tracks gate control."""

    result = mod.evaluate_tu93_positive_control(_fixture_frames())

    assert result["passed"] is True
    assert result["player_track_id"] != result["goal_track_id"]
    assert result["player_motion"] > 0.0
    assert result["goal_persistence"] == 1.0


def test_scenario_arc_wmte_4841_live_path_import_hook_is_reachable() -> None:
    """SCENARIO-ARC-WMTE-4841-LIVE-PATH-REACHABLE: live import closure reaches the module."""

    import scripts.arc_orphan_solver_lint as lint

    closure = lint._closure(lint.ENTRYPOINTS) | {entrypoint.stem for entrypoint in lint.ENTRYPOINTS}
    assert "arc_object_identity_perception" in closure
    assert lint.main() == 0


def test_req_arc_wmte_4841_delivered_result_json_is_real_frame_backed() -> None:
    """SCENARIO-ARC-WMTE-4841-REAL-FRAME-CORRESPONDENCE: final artifact is the requested deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["measured_on_real_frames"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert set(artifact["per_game_correspondence"]) == {"lp85", "r11l", "tu93"}
    assert artifact["preconditions_checked"]["offline_arcade"]["ok"] is True
    assert artifact["preconditions_checked"]["available_games"] == ["lp85", "r11l", "tu93"]
    assert artifact["reproducibility_checksum"] == "sha256:" + mod.payload_checksum(artifact)
    for row in artifact["per_game_correspondence"].values():
        assert row["n_frames"] > 1
        assert 0.0 <= row["shape_motion_score"] <= 1.0
        assert 0.0 <= row["color_centroid_baseline_score"] <= 1.0
