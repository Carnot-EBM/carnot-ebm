"""Tests for Exp 4960 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-LEARN-4960,
SCENARIO-LEARN-4960-CHECKPOINT-REFRESHED,
SCENARIO-LEARN-4960-SUBSTRATE-FIX,
SCENARIO-LEARN-4960-RESIDUAL-NO-FABRICATION,
SCENARIO-LEARN-4960-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.adversarial_verify as av
from carnot import experiment_4949_self_play_verifier_checkpoint as exp4949
from carnot import experiment_4960_self_play_verifier_checkpoint as exp4960


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _loop_result(game: str, reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "states_expanded": 74,
        "learned_verifier_checkpoint": f"models/arc_verifier_{game}.json",
        "solve_provenance": "live_agent_self_discovery",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
    }


def _preconditions(game: str = "dc22", *, ok: bool = True) -> dict[str, object]:
    return {
        "ok": ok,
        "offline_arcade": {"ok": True},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_banked": {"ok": True, "game": game, "prior_level": 2},
        "checkpoint_existing": {"ok": True, "path": f"models/arc_verifier_{game}.json"},
        "target_rotation": {
            "ok": True,
            "game": game,
            "rotated_off": list(exp4960.DISALLOWED_TARGET_GAMES),
        },
        "target_selection": {
            "game": game,
            "prior_level": 2,
            "banked": True,
            "reason": f"rotated_banked_target_warm_start_{game}",
        },
    }


def _success_artifact(tmp_path: Path, monkeypatch) -> dict[str, object]:
    ckpt = tmp_path / "models" / "arc_verifier_dc22.json"
    ckpt.parent.mkdir()
    ckpt.write_text('{"ok": true}\n', encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    monkeypatch.setattr(exp4960, "REPO", tmp_path)

    summary = exp4960.summarize_loop_result(
        game="dc22",
        loop_result=_loop_result("dc22"),
        loop_result_path="results/arc_loop_solve_dc22.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    return exp4960.build_artifact(
        target_selection={
            "game": "dc22",
            "prior_level": 2,
            "banked": True,
            "reason": "rotated_banked_target_warm_start_dc22",
        },
        loop_summary=summary,
        preconditions_checked=_preconditions("dc22"),
        duration_s=3.25,
        flag_resolved=True,
    )


def test_req_learn_4960_spec_declares_required_contract() -> None:
    """REQ-LEARN-4960: OpenSpec anchors the continued honest-substrate contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert exp4960._target_game(None) == "dc22"
    assert exp4960._target_game("sb26") == "sb26"
    assert exp4960._rotation_ok("dc22") is True
    assert exp4960._rotation_ok("lp85") is False
    for ref in exp4960.SPEC_REFS:
        assert ref in spec
    assert exp4960.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4960.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_learn_4960_checkpoint_refreshed_success(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-LEARN-4960-CHECKPOINT-REFRESHED: dc22 checkpoint refresh succeeds."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    assert artifact["experiment"] == exp4960.EXPERIMENT
    assert artifact["schema"] == exp4960.SCHEMA
    assert artifact["spec_refs"] == exp4960.SPEC_REFS
    assert artifact["random_seed"] == exp4960.RANDOM_SEED
    assert artifact["field_principles"] == exp4960.FIELD_PRINCIPLES
    assert artifact["honest_verdict"] == exp4960.SUCCESS_VERDICT
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["checkpoint_path"] == "models/arc_verifier_dc22.json"
    assert artifact["target_game"] == "dc22"
    assert artifact["inference_substrate"] == exp4960.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == 3.25
    assert artifact["flag_resolved"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["states_expanded"] == 74
    assert "search_state_count" not in artifact
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert int(artifact["checkpoint_mtime_after_ns"]) > int(artifact["checkpoint_mtime_before_ns"])
    assert artifact["checkpoint_mtime_delta_ns"] > 0
    assert artifact["schema_errors"] == []
    assert exp4960.artifact_schema_errors(artifact) == []


def test_scenario_learn_4960_substrate_fix_avoids_duration_too_short(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-LEARN-4960-SUBSTRATE-FIX: offline gate uses verifier substrate."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    floor = av.duration_floor_for_artifact(artifact)
    flags: list[object] = []
    av.check_duration_vs_claim(artifact, flags)

    assert floor == {
        "substrate": exp4960.INFERENCE_SUBSTRATE,
        "min_duration_s": 1.0,
        "reason": "verifier_scoring",
    }
    assert [getattr(flag, "kind", None) for flag in flags] == []

    out = tmp_path / "artifact.json"
    exp4960.write_artifact(artifact, out)
    report = av.verify_artifact(out)
    critical = [flag for flag in report["flags"] if flag.get("severity") == "critical"]
    assert [flag["kind"] for flag in critical] == []


def test_scenario_learn_4960_residual_and_blocked_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-LEARN-4960-RESIDUAL-NO-FABRICATION: failures stay residuals."""

    monkeypatch.setattr(exp4960, "REPO", tmp_path)
    failed = _loop_result("dc22", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp4960.summarize_loop_result(
        game="dc22",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_dc22.json",
        checkpoint_mtime_before_ns=123,
    )
    residual = exp4960.build_artifact(
        target_selection={
            "game": "dc22",
            "prior_level": 2,
            "banked": True,
            "reason": "rotated_banked_target_warm_start_dc22",
        },
        loop_summary=summary,
        preconditions_checked={**_preconditions("dc22"), "checkpoint_existing": {"ok": False}},
        duration_s=1.2,
        flag_resolved=True,
    )
    unresolved = exp4960.build_artifact(
        target_selection={
            "game": "dc22",
            "prior_level": 2,
            "banked": True,
            "reason": "rotated_banked_target_warm_start_dc22",
        },
        loop_summary={
            **summary,
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "learned_verifier_checkpoint": "models/arc_verifier_dc22.json",
            "checkpoint_mtime_before_ns": "1",
            "checkpoint_mtime_after_ns": "2",
            "checkpoint_mtime_delta_ns": 1,
            "self_play_residual": "checkpoint_refreshed_gate_passed",
        },
        preconditions_checked=_preconditions("dc22"),
        duration_s=1.2,
        flag_resolved=False,
    )
    blocked = exp4960.build_blocked_artifact(
        reason="checkpoint_missing",
        preconditions_checked={
            **_preconditions("dc22", ok=False),
            "checkpoint_existing": {"ok": False, "path": "models/arc_verifier_dc22.json"},
        },
        target_game="dc22",
        duration_s=1.0,
        flag_resolved=True,
    )
    missing = exp4960.summarize_loop_result(
        game="dc22",
        loop_result=None,
        loop_result_path="results/arc_loop_solve_dc22.json",
        checkpoint_mtime_before_ns="bad",
    )

    assert residual["honest_verdict"] == "complete_dc22_self_play_residual_reproduction_gate_failed"
    assert residual["verifier_checkpoint_refreshed"] is False
    assert residual["checkpoint_path"] is None
    assert residual["checkpoint_mtime_after_ns"] is None
    assert residual["checkpoint_mtime_delta_ns"] is None
    assert residual["flag_resolved"] is True
    assert residual["schema_errors"] == []
    assert unresolved["flag_resolved"] is False
    assert unresolved["honest_verdict"] != exp4960.SUCCESS_VERDICT
    assert "success_verdict_without_gate" not in unresolved["schema_errors"]
    assert blocked["honest_verdict"] == "blocked_checkpoint_missing"
    assert blocked["target_game"] == "dc22"
    assert blocked["verifier_checkpoint_refreshed"] is False
    assert blocked["schema_errors"] == []
    assert missing["self_play_residual"] == "loop_result_missing"
    assert missing["checkpoint_mtime_before_ns"] is None
    assert missing["search_state_count"] == 0


def test_req_learn_4960_schema_rejects_stale_metadata_and_bad_substrate(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-LEARN-4960: schema rejects stale identity, targets, and substrate drift."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    for field, old_value, expected in (
        ("experiment", exp4949.EXPERIMENT, "experiment_mismatch"),
        ("schema", exp4949.SCHEMA, "schema_mismatch"),
        ("spec_refs", exp4949.SPEC_REFS, "spec_refs_mismatch"),
        ("random_seed", exp4949.RANDOM_SEED, "random_seed_mismatch"),
    ):
        stale = dict(artifact)
        stale[field] = old_value
        stale["reproducibility_checksum"] = exp4960.stable_checksum(stale)
        assert expected in exp4960.artifact_schema_errors(stale)

    too_fast = dict(artifact, duration_s=0.25)
    too_fast["reproducibility_checksum"] = exp4960.stable_checksum(too_fast)
    assert "duration_too_short_for_verifier_ensemble" in exp4960.artifact_schema_errors(
        too_fast
    )

    live_too_fast = dict(artifact, inference_substrate="live_llm_inference", duration_s=2.5)
    live_too_fast["reproducibility_checksum"] = exp4960.stable_checksum(live_too_fast)
    live_errors = exp4960.artifact_schema_errors(live_too_fast)
    assert "inference_substrate_mismatch" in live_errors
    assert "duration_too_short_for_live_llm_inference" in live_errors

    unknown_substrate = dict(artifact, inference_substrate="deterministic_verifier")
    unknown_substrate["reproducibility_checksum"] = exp4960.stable_checksum(unknown_substrate)
    assert "unknown_inference_substrate" in exp4960.artifact_schema_errors(
        unknown_substrate
    )

    rotation_violation = dict(artifact, target_game="lp85")
    rotation_violation["reproducibility_checksum"] = exp4960.stable_checksum(
        rotation_violation
    )
    assert "success_target_rotation_violation" in exp4960.artifact_schema_errors(
        rotation_violation
    )


def test_scenario_learn_4960_main_writes_success_and_blocked_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-LEARN-4960-BLOCKED-PRECONDITION: CLI writes terminal artifacts."""

    ckpt = tmp_path / "models" / "arc_verifier_dc22.json"
    ckpt.parent.mkdir()
    ckpt.write_text('{"ok": true}\n', encoding="utf-8")
    loop_result = tmp_path / "results" / "arc_loop_solve_dc22.json"
    loop_result.parent.mkdir()
    loop_result.write_text(json.dumps(_loop_result("dc22")), encoding="utf-8")
    output = tmp_path / "results" / "experiment_4960_self_play_verifier_checkpoint.json"

    monkeypatch.setattr(exp4960, "REPO", tmp_path)
    monkeypatch.setattr(exp4960, "ARTIFACT", output)
    monkeypatch.setattr(exp4960, "check_preconditions", lambda game=None: _preconditions("dc22"))

    ok = exp4960.main(
        [
            "--game",
            "dc22",
            "--loop-result",
            str(loop_result),
            "--checkpoint-mtime-before-ns",
            str(ckpt.stat().st_mtime_ns - 1),
            "--duration-s",
            "4.0",
        ]
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert ok == 0
    assert written["honest_verdict"] == exp4960.SUCCESS_VERDICT
    assert written["result_path"] == exp4960.RESULT_RELATIVE_PATH

    monkeypatch.setattr(
        exp4960,
        "check_preconditions",
        lambda game=None: {
            **_preconditions("lp85", ok=False),
            "target_rotation": {"ok": False, "game": "lp85"},
            "target_selection": {"game": "lp85"},
        },
    )
    blocked_code = exp4960.main(["--game", "lp85", "--duration-s", "1.0"])
    blocked = json.loads(output.read_text(encoding="utf-8"))

    assert blocked_code == 0
    assert blocked["honest_verdict"].startswith("blocked_target_rotation_")
    assert blocked["target_game"] == "lp85"
