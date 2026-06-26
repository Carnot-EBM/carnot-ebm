"""Tests for Exp 4833 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-ARC-WMTE-4833,
SCENARIO-ARC-WMTE-4833-CHECKPOINT-REFRESHED,
SCENARIO-ARC-WMTE-4833-RESIDUAL-NO-FABRICATION,
SCENARIO-ARC-WMTE-4833-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4823_self_play_verifier_checkpoint as exp4823
from carnot import experiment_4833_self_play_verifier_checkpoint as exp4833


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(game: str, reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "states_expanded": 89,
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
    monkeypatch.setattr(exp4833, "REPO", tmp_path)

    summary = exp4833.summarize_loop_result(
        game="re86",
        loop_result=_loop_result("re86"),
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    return exp4833.build_artifact(
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


def test_req_arc_wmte_4833_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4833: OpenSpec declares the 4833 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4833.SPEC_REFS:
        assert ref in spec
    assert exp4833.RESULT_RELATIVE_PATH in spec
    assert "checkpoint_mtime_delta_ns" in spec
    assert "search_state_count" in spec
    for field, principle in exp4833.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4833_checkpoint_refreshed_success(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4833-CHECKPOINT-REFRESHED: mtime advance plus gate green succeeds."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    assert artifact["experiment"] == exp4833.EXPERIMENT
    assert artifact["schema"] == exp4833.SCHEMA
    assert artifact["spec_refs"] == exp4833.SPEC_REFS
    assert artifact["random_seed"] == exp4833.RANDOM_SEED
    assert artifact["field_principles"] == exp4833.FIELD_PRINCIPLES
    assert artifact["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["states_expanded"] == 89
    assert artifact["search_state_count"] == 89
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert int(artifact["checkpoint_mtime_after_ns"]) > int(artifact["checkpoint_mtime_before_ns"])
    assert artifact["checkpoint_mtime_delta_ns"] > 0
    assert artifact["schema_errors"] == []
    assert exp4833.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4833_residual_does_not_fabricate_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4833-RESIDUAL-NO-FABRICATION: failed gates stay residuals."""

    monkeypatch.setattr(exp4833, "REPO", tmp_path)
    failed = _loop_result("re86", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp4833.summarize_loop_result(
        game="re86",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=123,
    )
    artifact = exp4833.build_artifact(
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
    assert artifact["search_state_count"] == 89
    assert artifact["schema_errors"] == []


def test_scenario_arc_wmte_4833_missing_loop_result_records_residual() -> None:
    """SCENARIO-ARC-WMTE-4833-RESIDUAL-NO-FABRICATION: absent loop output is explicit."""

    summary = exp4833.summarize_loop_result(
        game="re86",
        loop_result=None,
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns="bad",
    )

    assert summary["self_play_residual"] == "loop_result_missing"
    assert summary["checkpoint_mtime_before_ns"] is None
    assert summary["search_state_count"] == 0


def test_scenario_arc_wmte_4833_blocked_precondition_artifact() -> None:
    """SCENARIO-ARC-WMTE-4833-BLOCKED-PRECONDITION: blocked runs remain auditable."""

    artifact = exp4833.build_blocked_artifact(
        reason="target_not_banked",
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": False, "game": "sb26", "prior_level": 0},
        },
        target_game="sb26",
    )

    assert artifact["experiment"] == exp4833.EXPERIMENT
    assert artifact["honest_verdict"] == "blocked_target_not_banked"
    assert artifact["target_game"] == "sb26"
    assert artifact["verifier_checkpoint_refreshed"] is False
    assert artifact["self_play_residual"] == "target_not_banked"
    assert artifact["search_state_count"] == 0
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4833_schema_rejects_stale_4823_metadata(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4833: schema prevents accidental reuse of the 4823 identity."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    stale_experiment = dict(artifact)
    stale_experiment["experiment"] = exp4823.EXPERIMENT
    stale_experiment["reproducibility_checksum"] = exp4833.stable_checksum(stale_experiment)
    assert "experiment_mismatch" in exp4833.artifact_schema_errors(stale_experiment)

    stale_schema = dict(artifact)
    stale_schema["schema"] = exp4823.SCHEMA
    stale_schema["reproducibility_checksum"] = exp4833.stable_checksum(stale_schema)
    assert "schema_mismatch" in exp4833.artifact_schema_errors(stale_schema)

    stale_refs = dict(artifact)
    stale_refs["spec_refs"] = exp4823.SPEC_REFS
    stale_refs["reproducibility_checksum"] = exp4833.stable_checksum(stale_refs)
    assert "spec_refs_mismatch" in exp4833.artifact_schema_errors(stale_refs)

    stale_seed = dict(artifact)
    stale_seed["random_seed"] = exp4823.RANDOM_SEED
    stale_seed["reproducibility_checksum"] = exp4833.stable_checksum(stale_seed)
    assert "random_seed_mismatch" in exp4833.artifact_schema_errors(stale_seed)


def test_req_arc_wmte_4833_io_helpers_cover_paths(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4833: helper IO paths are deterministic and fail closed."""

    monkeypatch.setattr(exp4833, "REPO", tmp_path)
    missing = tmp_path / "missing.json"
    present = tmp_path / "results" / "loop.json"
    present.parent.mkdir()
    present.write_text(json.dumps(_loop_result("re86")), encoding="utf-8")
    list_payload = tmp_path / "results" / "list.json"
    list_payload.write_text("[]", encoding="utf-8")

    assert exp4833._read_loop_result(missing) is None  # noqa: SLF001
    assert exp4833._read_loop_result(list_payload) is None  # noqa: SLF001
    assert exp4833._read_loop_result(present)["game"] == "re86"  # noqa: SLF001
    assert exp4833._relative_path(present) == "results/loop.json"  # noqa: SLF001
    assert exp4833._relative_path(Path("/outside.json")) == "/outside.json"  # noqa: SLF001
    assert exp4833._precondition_failure({"offline_arcade": {"ok": False}}) == "offline_arcade_missing"  # noqa: SLF001
    assert exp4833._precondition_failure(  # noqa: SLF001
        {"offline_arcade": {"ok": True}, "registry_loadable": {"ok": False}}
    ) == "registry_missing"
    assert exp4833._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": False},
        }
    ) == "target_not_banked"
    assert exp4833._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
        }
    ) is None


def test_req_arc_wmte_4833_main_writes_terminal_artifact(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4833: main writes the 4833 artifact path from loop output."""

    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    ckpt = tmp_path / "models" / "arc_verifier_re86.json"
    ckpt.write_text("{}", encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    loop_path = tmp_path / "results" / "arc_loop_solve_re86.json"
    loop_path.write_text(json.dumps(_loop_result("re86")), encoding="utf-8")

    monkeypatch.setattr(exp4833, "REPO", tmp_path)
    monkeypatch.setattr(exp4833, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4833, "ARTIFACT", tmp_path / exp4833.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4833,
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

    assert exp4833.main(["--game", "re86", "--checkpoint-mtime-before-ns", str(before_ns)]) == 0

    written = json.loads((tmp_path / exp4833.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["experiment"] == exp4833.EXPERIMENT
    assert written["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert written["search_state_count"] == 89
    assert written["schema_errors"] == []


def test_req_arc_wmte_4833_main_writes_blocked_artifact(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4833-BLOCKED-PRECONDITION: main writes blocked artifacts."""

    monkeypatch.setattr(exp4833, "ARTIFACT", tmp_path / exp4833.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4833,
        "check_preconditions",
        lambda game=None: {
            "ok": False,
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": False, "game": "missing", "prior_level": 0},
            "target_selection": {
                "game": "missing",
                "prior_level": 0,
                "banked": False,
                "reason": "preferred_target_not_banked",
            },
        },
    )

    assert exp4833.main(["--game", "missing"]) == 0

    written = json.loads((tmp_path / exp4833.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "blocked_target_not_banked"
    assert written["verifier_checkpoint_refreshed"] is False
    assert written["search_state_count"] == 0
