"""Tests for Exp 4783 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-ARC-WMTE-4783,
SCENARIO-ARC-WMTE-4783-CHECKPOINT-REFRESHED,
SCENARIO-ARC-WMTE-4783-RESIDUAL-NO-FABRICATION,
SCENARIO-ARC-WMTE-4783-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot import experiment_4773_self_play_verifier_checkpoint as exp4773
from carnot import experiment_4783_self_play_verifier_checkpoint as exp4783
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
        "states_expanded": 83,
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
    monkeypatch.setattr(exp4783, "REPO", tmp_path)

    summary = exp4783.summarize_loop_result(
        game="re86",
        loop_result=_loop_result("re86"),
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    return exp4783.build_artifact(
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


def test_req_arc_wmte_4783_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4783: OpenSpec declares the 4783 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4783.SPEC_REFS:
        assert ref in spec
    assert exp4783.RESULT_RELATIVE_PATH in spec
    assert "checkpoint_mtime_delta_ns" in spec
    for field, principle in exp4783.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4783_checkpoint_refreshed_success(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4783-CHECKPOINT-REFRESHED: mtime advance plus gate green succeeds."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    assert artifact["experiment"] == exp4783.EXPERIMENT
    assert artifact["schema"] == exp4783.SCHEMA
    assert artifact["spec_refs"] == exp4783.SPEC_REFS
    assert artifact["random_seed"] == exp4783.RANDOM_SEED
    assert artifact["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["states_expanded"] == 83
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert isinstance(artifact["checkpoint_mtime_before_ns"], str)
    assert isinstance(artifact["checkpoint_mtime_after_ns"], str)
    assert int(artifact["checkpoint_mtime_after_ns"]) > int(artifact["checkpoint_mtime_before_ns"])
    assert artifact["checkpoint_mtime_delta_ns"] > 0
    assert artifact["schema_errors"] == []
    assert exp4783.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4783_registry_helpers_and_bad_int(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4783: registry wrappers preserve banked-target prechecks."""

    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text(
        """schema_version: 1
games:
- game: re86
  levels_reproduced: 2
- game: sb26
  levels_reproduced: bad
""",
        encoding="utf-8",
    )

    registry = exp4783.load_registry(registry_file)

    assert exp4783.registry_levels(registry) == {"re86": 2, "sb26": 0}
    assert exp4783.select_banked_target(registry, preferred_game="re86") == {
        "game": "re86",
        "prior_level": 2,
        "banked": True,
        "reason": "preferred_banked_target",
    }
    assert exp4783._as_optional_int("not-an-int") is None


def test_scenario_arc_wmte_4783_residual_does_not_fabricate_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4783-RESIDUAL-NO-FABRICATION: failed gates stay residuals."""

    monkeypatch.setattr(exp4783, "REPO", tmp_path)
    failed = _loop_result("re86", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp4783.summarize_loop_result(
        game="re86",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=123,
    )
    artifact = exp4783.build_artifact(
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
    assert artifact["checkpoint_mtime_delta_ns"] is None
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4783_schema_requires_delta_field(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4783: the 4783 schema requires the mtime-delta field."""

    artifact = _success_artifact(tmp_path, monkeypatch)
    missing = dict(artifact)
    missing.pop("checkpoint_mtime_delta_ns")
    missing["reproducibility_checksum"] = exp4783.stable_checksum(missing)

    assert "missing_field:checkpoint_mtime_delta_ns" in exp4783.artifact_schema_errors(missing)


def test_scenario_arc_wmte_4783_blocked_precondition_artifact() -> None:
    """SCENARIO-ARC-WMTE-4783-BLOCKED-PRECONDITION: blocked runs remain auditable."""

    artifact = exp4783.build_blocked_artifact(
        reason="target_not_banked",
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": False, "game": "sb26", "prior_level": 0},
        },
        target_game="sb26",
    )

    assert artifact["experiment"] == exp4783.EXPERIMENT
    assert artifact["honest_verdict"] == "blocked_target_not_banked"
    assert artifact["target_game"] == "sb26"
    assert artifact["verifier_checkpoint_refreshed"] is False
    assert artifact["self_play_residual"] == "target_not_banked"
    assert artifact["checkpoint_mtime_delta_ns"] is None
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4783_schema_rejects_stale_4773_metadata(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4783: schema prevents accidental reuse of the 4773 identity."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    stale_experiment = dict(artifact)
    stale_experiment["experiment"] = exp4773.EXPERIMENT
    stale_experiment["reproducibility_checksum"] = exp4783.stable_checksum(stale_experiment)
    assert "experiment_mismatch" in exp4783.artifact_schema_errors(stale_experiment)

    stale_schema = dict(artifact)
    stale_schema["schema"] = exp4773.SCHEMA
    stale_schema["reproducibility_checksum"] = exp4783.stable_checksum(stale_schema)
    assert "schema_mismatch" in exp4783.artifact_schema_errors(stale_schema)

    stale_refs = dict(artifact)
    stale_refs["spec_refs"] = exp4773.SPEC_REFS
    stale_refs["reproducibility_checksum"] = exp4783.stable_checksum(stale_refs)
    assert "spec_refs_mismatch" in exp4783.artifact_schema_errors(stale_refs)

    stale_seed = dict(artifact)
    stale_seed["random_seed"] = exp4773.RANDOM_SEED
    stale_seed["reproducibility_checksum"] = exp4783.stable_checksum(stale_seed)
    assert "random_seed_mismatch" in exp4783.artifact_schema_errors(stale_seed)


def test_req_arc_wmte_4783_schema_rejects_bad_mtime_delta(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4783: refreshed success requires a positive mtime delta."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    missing_delta = dict(artifact)
    missing_delta["checkpoint_mtime_delta_ns"] = None
    missing_delta["reproducibility_checksum"] = exp4783.stable_checksum(missing_delta)
    assert "refreshed_checkpoint_missing_mtime_delta" in exp4783.artifact_schema_errors(missing_delta)

    stale_delta = dict(artifact)
    stale_delta["checkpoint_mtime_delta_ns"] = 0
    stale_delta["reproducibility_checksum"] = exp4783.stable_checksum(stale_delta)
    assert "refreshed_checkpoint_nonpositive_mtime_delta" in exp4783.artifact_schema_errors(stale_delta)


def test_req_arc_wmte_4783_main_writes_terminal_artifact(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4783: main writes the 4783 artifact path from loop output."""

    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    ckpt = tmp_path / "models" / "arc_verifier_re86.json"
    ckpt.write_text("{}", encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    loop_path = tmp_path / "results" / "arc_loop_solve_re86.json"
    loop_path.write_text(json.dumps(_loop_result("re86")), encoding="utf-8")

    monkeypatch.setattr(exp4783, "REPO", tmp_path)
    monkeypatch.setattr(exp4783, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4783, "ARTIFACT", tmp_path / exp4783.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4783,
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

    assert exp4783.main(["--game", "re86", "--checkpoint-mtime-before-ns", str(before_ns)]) == 0

    written = json.loads((tmp_path / exp4783.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["experiment"] == exp4783.EXPERIMENT
    assert written["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert written["checkpoint_mtime_delta_ns"] > 0
    assert written["schema_errors"] == []


def test_scenario_arc_wmte_4783_main_writes_blocked_artifact(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4783-BLOCKED-PRECONDITION: main writes blocked output."""

    monkeypatch.setattr(exp4783, "ARTIFACT", tmp_path / exp4783.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4783,
        "check_preconditions",
        lambda game=None: {
            "ok": False,
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": False, "game": "sb26", "prior_level": 0},
            "target_selection": {
                "game": "sb26",
                "prior_level": 0,
                "banked": False,
                "reason": "preferred_target_not_banked",
            },
        },
    )

    assert exp4783.main(["--game", "sb26"]) == 0

    written = json.loads((tmp_path / exp4783.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "blocked_target_not_banked"
    assert written["verifier_checkpoint_refreshed"] is False
    assert written["schema_errors"] == []


def test_req_arc_wmte_4783_loop_warm_starts_saved_linear_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-ARC-WMTE-4783: arc_loop_solve warm-starts from models/arc_verifier_<game>.json."""

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
