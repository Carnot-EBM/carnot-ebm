"""Tests for Exp 4773 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-ARC-WMTE-4773,
SCENARIO-ARC-WMTE-4773-CHECKPOINT-REFRESHED,
SCENARIO-ARC-WMTE-4773-RESIDUAL-NO-FABRICATION,
SCENARIO-ARC-WMTE-4773-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot import experiment_4763_self_play_verifier_checkpoint as exp4763
from carnot import experiment_4773_self_play_verifier_checkpoint as exp4773
from carnot.agentic.arc_value_learner import LearnedVerifier
import scripts.arc_loop_solve as arc_loop_solve


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(game: str, reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "states_expanded": 77,
        "learned_verifier_checkpoint": f"models/arc_verifier_{game}.json",
        "solve_provenance": "development_proxy",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": reproduced,
        },
    }


def _success_artifact(tmp_path: Path, monkeypatch) -> dict[str, object]:
    ckpt = tmp_path / "models" / "arc_verifier_re86.json"
    ckpt.parent.mkdir()
    ckpt.write_text('{"ok": true}\n', encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    monkeypatch.setattr(exp4773, "REPO", tmp_path)

    summary = exp4773.summarize_loop_result(
        game="re86",
        loop_result=_loop_result("re86"),
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    return exp4773.build_artifact(
        target_selection={
            "game": "re86",
            "prior_level": 2,
            "banked": True,
            "reason": "preferred_banked_target",
        },
        loop_summary=summary,
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True, "game": "re86"},
        },
    )


def test_req_arc_wmte_4773_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4773: OpenSpec declares the 4773 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4773.SPEC_REFS:
        assert ref in spec
    assert exp4773.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4773.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4773_checkpoint_refreshed_success(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4773-CHECKPOINT-REFRESHED: mtime advance plus gate green succeeds."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    assert artifact["experiment"] == exp4773.EXPERIMENT
    assert artifact["schema"] == exp4773.SCHEMA
    assert artifact["spec_refs"] == exp4773.SPEC_REFS
    assert artifact["random_seed"] == exp4773.RANDOM_SEED
    assert artifact["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["states_expanded"] == 77
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert artifact["checkpoint_mtime_after_ns"] > artifact["checkpoint_mtime_before_ns"]
    assert artifact["schema_errors"] == []
    assert exp4773.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4773_residual_does_not_fabricate_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4773-RESIDUAL-NO-FABRICATION: failed gates stay residuals."""

    monkeypatch.setattr(exp4773, "REPO", tmp_path)
    failed = _loop_result("re86", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp4773.summarize_loop_result(
        game="re86",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=123,
    )
    artifact = exp4773.build_artifact(
        target_selection={
            "game": "re86",
            "prior_level": 2,
            "banked": True,
            "reason": "preferred_banked_target",
        },
        loop_summary=summary,
        preconditions_checked={"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}},
    )

    assert artifact["honest_verdict"] == "complete_re86_self_play_residual_reproduction_gate_failed"
    assert artifact["verifier_checkpoint_refreshed"] is False
    assert artifact["checkpoint_path"] is None
    assert artifact["checkpoint_mtime_after_ns"] is None
    assert artifact["schema_errors"] == []


def test_scenario_arc_wmte_4773_blocked_precondition_artifact() -> None:
    """SCENARIO-ARC-WMTE-4773-BLOCKED-PRECONDITION: blocked runs remain auditable."""

    artifact = exp4773.build_blocked_artifact(
        reason="target_not_banked",
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": False, "game": "sb26", "prior_level": 0},
        },
        target_game="sb26",
    )

    assert artifact["experiment"] == exp4773.EXPERIMENT
    assert artifact["honest_verdict"] == "blocked_target_not_banked"
    assert artifact["target_game"] == "sb26"
    assert artifact["verifier_checkpoint_refreshed"] is False
    assert artifact["self_play_residual"] == "target_not_banked"
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4773_schema_rejects_stale_4763_metadata(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4773: schema prevents accidental reuse of the 4763 identity."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    stale_experiment = dict(artifact)
    stale_experiment["experiment"] = exp4763.EXPERIMENT
    stale_experiment["reproducibility_checksum"] = exp4773.stable_checksum(stale_experiment)
    assert "experiment_mismatch" in exp4773.artifact_schema_errors(stale_experiment)

    stale_schema = dict(artifact)
    stale_schema["schema"] = exp4763.SCHEMA
    stale_schema["reproducibility_checksum"] = exp4773.stable_checksum(stale_schema)
    assert "schema_mismatch" in exp4773.artifact_schema_errors(stale_schema)

    stale_refs = dict(artifact)
    stale_refs["spec_refs"] = exp4763.SPEC_REFS
    stale_refs["reproducibility_checksum"] = exp4773.stable_checksum(stale_refs)
    assert "spec_refs_mismatch" in exp4773.artifact_schema_errors(stale_refs)

    stale_seed = dict(artifact)
    stale_seed["random_seed"] = exp4763.RANDOM_SEED
    stale_seed["reproducibility_checksum"] = exp4773.stable_checksum(stale_seed)
    assert "random_seed_mismatch" in exp4773.artifact_schema_errors(stale_seed)


def test_req_arc_wmte_4773_main_writes_terminal_artifact(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4773: main writes the 4773 artifact path from loop output."""

    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    ckpt = tmp_path / "models" / "arc_verifier_re86.json"
    ckpt.write_text("{}", encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    loop_path = tmp_path / "results" / "arc_loop_solve_re86.json"
    loop_path.write_text(json.dumps(_loop_result("re86")), encoding="utf-8")

    monkeypatch.setattr(exp4773, "REPO", tmp_path)
    monkeypatch.setattr(exp4773, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4773, "ARTIFACT", tmp_path / exp4773.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4773,
        "check_preconditions",
        lambda game=None: {
            "ok": True,
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True, "game": "re86", "prior_level": 2},
            "target_selection": {
                "game": "re86",
                "prior_level": 2,
                "banked": True,
                "reason": "preferred_banked_target",
            },
        },
    )

    assert exp4773.main(["--game", "re86", "--checkpoint-mtime-before-ns", str(before_ns)]) == 0

    written = json.loads((tmp_path / exp4773.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["experiment"] == exp4773.EXPERIMENT
    assert written["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert written["schema_errors"] == []


def test_req_arc_wmte_4773_loop_warm_starts_saved_linear_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-ARC-WMTE-4773: arc_loop_solve warm-starts from models/arc_verifier_<game>.json."""

    def featurize(_game: object) -> list[float]:
        return [1.0]

    ckpt_dir = tmp_path / "models"
    ckpt_dir.mkdir()
    LearnedVerifier(featurize).fit([[1.0]], [3.0]).save(
        ckpt_dir / "arc_verifier_re86.json",
        meta={"feature_names": ["unit"], "trained_games": ["re86"]},
    )
    adapter = SimpleNamespace(featurize=featurize, hand_verifier=lambda _game: 99.0)

    monkeypatch.setattr(arc_loop_solve, "REPO", tmp_path)
    monkeypatch.setattr(arc_loop_solve, "CKPT_DIR", ckpt_dir)
    monkeypatch.setattr(arc_loop_solve, "load_live_spatial_value_head", lambda root, game: None)

    verifier, source = arc_loop_solve._live_verifier_for_adapter("re86", adapter)

    assert source == "learned_verifier_live_checkpoint"
    assert abs(verifier(object()) - 3.0) < 1e-9
