"""Tests for Exp5641 ARC counterexample-patched executable-model artifact.

Spec refs: REQ-ARC-WMTE-5641,
SCENARIO-ARC-WMTE-5641-COUNTEREXAMPLE-PATCH-REPLAY,
SCENARIO-ARC-WMTE-5641-CONTROLS-AND-ABSTENTION.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carnot import experiment_5641_arc_counterexample_executable_model as mod
from carnot.agentic.arc_counterexample_executable_model import TransitionReceipt


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _receipt(game: str, step: int, *, color: int, x: int, y: int) -> TransitionReceipt:
    before = np.zeros((8, 8), dtype=np.int16)
    before[y : y + 2, x : x + 2] = color
    after = before.copy()
    after[y : y + 2, x : x + 2] = 9
    return TransitionReceipt(
        trace_id=game,
        episode=f"{game}-ep-0",
        step=step,
        state=before,
        action=6,
        data={"x": x, "y": y},
        successor=after,
        provenance="agent_owned_runtime_observation",
    )


def _synthetic_game(game: str) -> list[TransitionReceipt]:
    coords = [(1, 1), (4, 1), (1, 4), (4, 4), (2, 2), (5, 2)]
    return [
        _receipt(game, step, color=step + 2, x=x, y=y)
        for step, (x, y) in enumerate(coords)
    ]


def _registry() -> dict[str, object]:
    return {
        "reproducible_total_levels": 177,
        "games": [
            {"game": "ga", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "gb", "reproducibility": "reproduced", "levels_reproduced": 3},
            {"game": "gc", "reproducibility": "reproduced", "levels_reproduced": 4},
        ],
    }


def test_req_arc_wmte_5641_spec_declares_patch_language_and_controls() -> None:
    """REQ-ARC-WMTE-5641: OpenSpec pins typed language, bounded patch operators,
    source exclusions, controls, and ready-score gates before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5641") :]

    for marker in (
        "SCENARIO-ARC-WMTE-5641-COUNTEREXAMPLE-PATCH-REPLAY",
        "SCENARIO-ARC-WMTE-5641-CONTROLS-AND-ABSTENTION",
        "game_adapter_used=false",
        "offline_ground_truth_bfs_used=false",
        "deterministic_counterexample_patched_executable_model",
        "contradictory active clauses",
    ):
        assert marker in section


def test_req_arc_wmte_5641_source_guard_blocks_game_source_before_trace_load() -> None:
    """REQ-ARC-WMTE-5641: source-access guards are recorded before traces load and
    forbid game source or per-game adapter paths."""

    guard = mod.source_access_guard(recorded_before_trace_load=True)

    assert guard["recorded_before_trace_load"] is True
    assert guard["source_read"] is False
    assert guard["game_adapter_used"] is False
    assert mod.path_allowed_by_source_guard("data/arc_transition_corpus/dc22.npz", guard) is True
    assert mod.path_allowed_by_source_guard("environment_files/dc22/game.py", guard) is False
    assert (
        mod.path_allowed_by_source_guard("python/carnot/agentic/arc_game_adapters.py", guard)
        is False
    )


def test_scenario_5641_build_artifact_reports_required_controls() -> None:
    """SCENARIO-ARC-WMTE-5641-CONTROLS-AND-ABSTENTION: build_artifact evaluates the
    patched arm, generic controls, replay gate, and adversarial controls."""

    traces = {game: _synthetic_game(game) for game in ("ga", "gb", "gc")}
    artifact = mod.build_artifact(
        transitions_by_game=traces,
        registry=_registry(),
        roster=("ga", "gb", "gc"),
        random_seed=5641,
    )

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["agent_owned_evidence_only"] is True
    assert artifact["source_read"] is False
    assert artifact["game_adapter_used"] is False
    assert artifact["offline_ground_truth_bfs_used"] is False
    assert artifact["model_specs"] == []
    assert artifact["registry_precheck_receipt"]["only_already_reproduced_levels"] is True
    assert artifact["patch_operator_set"] == ["add", "specialize", "relax", "retire"]
    assert artifact["accepted_patch_count"] > 0
    assert artifact["rejected_patch_count"] > 0
    assert artifact["counterexample_count"] >= artifact["accepted_patch_count"]
    assert artifact["all_receipt_replay_pass"] is True
    assert artifact["heldout_transition_error_by_arm"]["patched"] < artifact[
        "heldout_transition_error_by_arm"
    ]["unpatched"]
    assert artifact["abstention_calibration"]["unsupported_abstention_rate"] == 1.0
    assert artifact["mechanism_question_controls"]["informative"]["score"] > artifact[
        "mechanism_question_controls"
    ]["irrelevant"]["score"]
    assert artifact["unsafe_patch_accept_count"] == 0
    assert artifact["executable_model_ready_score"] == 1.0
    assert artifact["honest_verdict"].startswith("complete:")
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]


def test_req_arc_wmte_5641_checked_in_artifact_has_required_schema() -> None:
    """REQ-ARC-WMTE-5641: the checked-in JSON is the stable Exp5641 deliverable."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["agent_owned_evidence_only"] is True
    assert artifact["source_read"] is False
    assert artifact["game_adapter_used"] is False
    assert artifact["offline_ground_truth_bfs_used"] is False
    assert artifact["model_specs"] == []
    assert artifact["source_access_guards"]["recorded_before_trace_load"] is True
    assert artifact["hypothesis_language"]["forbidden_literals_present"] == []
    assert artifact["unsafe_patch_accept_count"] == 0
    assert artifact["all_receipt_replay_pass"] is True
    assert artifact["honest_verdict"].startswith(("complete:", "blocked:"))
    assert artifact["reproducibility_checksum"].startswith("sha256:")
