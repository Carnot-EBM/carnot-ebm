"""Tests for Exp 5011 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-LEARN-5011,
SCENARIO-LEARN-5011-CHECKPOINT-REFRESHED,
SCENARIO-LEARN-5011-SUBSTRATE-FIX,
SCENARIO-LEARN-5011-RESIDUAL-NO-FABRICATION,
SCENARIO-LEARN-5011-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.adversarial_verify as av
from carnot import experiment_4993_self_play_verifier_checkpoint as exp4993
from carnot import experiment_5011_self_play_verifier_checkpoint as exp5011
from carnot.agentic import arc_game_adapters as adapters
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_value_learner import LearnedVerifier


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _loop_result(game: str, reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "states_expanded": 24,
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


def _preconditions(game: str = "r11l", *, ok: bool = True) -> dict[str, object]:
    return {
        "ok": ok,
        "offline_arcade": {"ok": True},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_banked": {"ok": True, "game": game, "prior_level": 2},
        "checkpoint_existing": {"ok": True, "path": f"models/arc_verifier_{game}.json"},
        "target_rotation": {
            "ok": game not in exp5011.DISALLOWED_TARGET_GAMES,
            "game": game,
            "rotated_off": list(exp5011.DISALLOWED_TARGET_GAMES),
        },
        "target_selection": {
            "game": game,
            "prior_level": 2,
            "banked": True,
            "reason": "preferred_banked_target_r11l",
        },
    }


def _success_artifact(tmp_path: Path, monkeypatch) -> dict[str, object]:
    ckpt = tmp_path / "models" / "arc_verifier_r11l.json"
    ckpt.parent.mkdir()
    ckpt.write_text('{"ok": true}\n', encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    monkeypatch.setattr(exp5011, "REPO", tmp_path)

    summary = exp5011.summarize_loop_result(
        game="r11l",
        loop_result=_loop_result("r11l"),
        loop_result_path="results/arc_loop_solve_r11l.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    return exp5011.build_artifact(
        target_selection={
            "game": "r11l",
            "prior_level": 2,
            "banked": True,
            "reason": "preferred_banked_target_r11l",
        },
        loop_summary=summary,
        preconditions_checked=_preconditions("r11l"),
        duration_s=2.0,
        flag_resolved=True,
    )


def test_req_learn_5011_spec_declares_required_contract() -> None:
    """REQ-LEARN-5011: OpenSpec anchors the preferred r11l checkpoint contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert exp5011._target_game(None) == "r11l"
    assert exp5011._target_game("lf52") == "lf52"
    assert exp5011._rotation_ok("r11l") is True
    assert exp5011._rotation_ok("su15") is False
    assert exp5011._rotation_ok("ft09") is False
    assert exp5011._rotation_ok("ls20") is False
    for ref in exp5011.SPEC_REFS:
        assert ref in spec
    assert exp5011.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5011.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_learn_5011_r11l_adapter_loads_existing_checkpoint() -> None:
    """REQ-LEARN-5011: r11l exposes the 3 features required by its 4-weight head."""

    ad = adapters.get_adapter("r11l")
    assert ad is not None
    assert callable(ad.featurize)

    arc = kit.offline_arcade()
    env = arc.make("r11l", scorecard_id=arc.open_scorecard())
    env.reset()
    features = [float(value) for value in ad.featurize(env._game)]
    checkpoint = json.loads((REPO / "models" / "arc_verifier_r11l.json").read_text())
    verifier = LearnedVerifier.load(REPO / "models" / "arc_verifier_r11l.json", ad.featurize)

    assert features == [4084.0, 5.0, 4096.0]
    assert len(features) + 1 == len(checkpoint["weights"])
    assert verifier.n_samples == checkpoint["n_samples"]
    assert verifier(env._game) >= 0.0


def test_scenario_learn_5011_checkpoint_refreshed_success(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-LEARN-5011-CHECKPOINT-REFRESHED: r11l checkpoint refresh succeeds."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    assert artifact["experiment"] == exp5011.EXPERIMENT
    assert artifact["schema"] == exp5011.SCHEMA
    assert artifact["spec_refs"] == exp5011.SPEC_REFS
    assert artifact["random_seed"] == exp5011.RANDOM_SEED
    assert artifact["field_principles"] == exp5011.FIELD_PRINCIPLES
    assert artifact["honest_verdict"] == exp5011.SUCCESS_VERDICT
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["checkpoint_path"] == "models/arc_verifier_r11l.json"
    assert artifact["target_game"] == "r11l"
    assert artifact["inference_substrate"] == exp5011.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == 2.0
    assert artifact["flag_resolved"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["states_expanded"] == 24
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert int(artifact["checkpoint_mtime_after_ns"]) > int(artifact["checkpoint_mtime_before_ns"])
    assert artifact["checkpoint_mtime_delta_ns"] > 0
    assert artifact["schema_errors"] == []
    assert exp5011.artifact_schema_errors(artifact) == []


def test_scenario_learn_5011_substrate_fix_avoids_duration_too_short(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-LEARN-5011-SUBSTRATE-FIX: offline gate uses verifier substrate."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    floor = av.duration_floor_for_artifact(artifact)
    flags: list[object] = []
    av.check_duration_vs_claim(artifact, flags)

    assert floor == {
        "substrate": exp5011.INFERENCE_SUBSTRATE,
        "min_duration_s": 1.0,
        "reason": "verifier_scoring",
    }
    assert [getattr(flag, "kind", None) for flag in flags] == []

    out = tmp_path / "artifact.json"
    exp5011.write_artifact(artifact, out)
    report = av.verify_artifact(out)
    critical = [flag for flag in report["flags"] if flag.get("severity") == "critical"]
    assert [flag["kind"] for flag in critical] == []


def test_scenario_learn_5011_residual_and_blocked_paths(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-LEARN-5011-RESIDUAL-NO-FABRICATION: failures stay residuals."""

    monkeypatch.setattr(exp5011, "REPO", tmp_path)
    failed = _loop_result("r11l", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp5011.summarize_loop_result(
        game="r11l",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_r11l.json",
        checkpoint_mtime_before_ns=123,
    )
    residual = exp5011.build_artifact(
        target_selection={
            "game": "r11l",
            "prior_level": 2,
            "banked": True,
            "reason": "preferred_banked_target_r11l",
        },
        loop_summary=summary,
        preconditions_checked={**_preconditions("r11l"), "checkpoint_existing": {"ok": False}},
        duration_s=1.1,
        flag_resolved=True,
    )
    unresolved = exp5011.build_artifact(
        target_selection={
            "game": "r11l",
            "prior_level": 2,
            "banked": True,
            "reason": "preferred_banked_target_r11l",
        },
        loop_summary={
            **summary,
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "learned_verifier_checkpoint": "models/arc_verifier_r11l.json",
            "checkpoint_mtime_before_ns": "1",
            "checkpoint_mtime_after_ns": "2",
            "checkpoint_mtime_delta_ns": 1,
            "self_play_residual": "checkpoint_refreshed_gate_passed",
        },
        preconditions_checked=_preconditions("r11l"),
        duration_s=1.1,
        flag_resolved=False,
    )
    blocked = exp5011.build_blocked_artifact(
        reason="checkpoint_missing",
        preconditions_checked={
            **_preconditions("r11l", ok=False),
            "checkpoint_existing": {"ok": False, "path": "models/arc_verifier_r11l.json"},
        },
        target_game="r11l",
        duration_s=1.0,
        flag_resolved=True,
    )
    missing = exp5011.summarize_loop_result(
        game="r11l",
        loop_result=None,
        loop_result_path="results/arc_loop_solve_r11l.json",
        checkpoint_mtime_before_ns="bad",
    )

    assert residual["honest_verdict"] == "complete_r11l_self_play_residual_reproduction_gate_failed"
    assert residual["verifier_checkpoint_refreshed"] is False
    assert residual["checkpoint_path"] is None
    assert residual["checkpoint_mtime_after_ns"] is None
    assert residual["checkpoint_mtime_delta_ns"] is None
    assert residual["schema_errors"] == []
    assert unresolved["flag_resolved"] is False
    assert unresolved["honest_verdict"] != exp5011.SUCCESS_VERDICT
    assert blocked["honest_verdict"] == "blocked_checkpoint_missing"
    assert blocked["target_game"] == "r11l"
    assert blocked["verifier_checkpoint_refreshed"] is False
    assert blocked["schema_errors"] == []
    assert missing["self_play_residual"] == "loop_result_missing"
    assert missing["checkpoint_mtime_before_ns"] is None
    assert missing["search_state_count"] == 0


def test_req_learn_5011_schema_rejects_stale_metadata_and_bad_substrate(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-LEARN-5011: schema rejects stale identity, targets, and substrate drift."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    for field, old_value, expected in (
        ("experiment", exp4993.EXPERIMENT, "experiment_mismatch"),
        ("schema", exp4993.SCHEMA, "schema_mismatch"),
        ("spec_refs", exp4993.SPEC_REFS, "spec_refs_mismatch"),
        ("random_seed", exp4993.RANDOM_SEED, "random_seed_mismatch"),
    ):
        stale = dict(artifact)
        stale[field] = old_value
        stale["reproducibility_checksum"] = exp5011.stable_checksum(stale)
        assert expected in exp5011.artifact_schema_errors(stale)

    too_fast = dict(artifact, duration_s=0.25)
    too_fast["reproducibility_checksum"] = exp5011.stable_checksum(too_fast)
    assert "duration_too_short_for_verifier_ensemble" in exp5011.artifact_schema_errors(
        too_fast
    )

    live_too_fast = dict(artifact, inference_substrate="live_llm_inference", duration_s=2.5)
    live_too_fast["reproducibility_checksum"] = exp5011.stable_checksum(live_too_fast)
    live_errors = exp5011.artifact_schema_errors(live_too_fast)
    assert "inference_substrate_mismatch" in live_errors
    assert "duration_too_short_for_live_llm_inference" in live_errors

    unknown_substrate = dict(artifact, inference_substrate="deterministic_verifier")
    unknown_substrate["reproducibility_checksum"] = exp5011.stable_checksum(unknown_substrate)
    assert "unknown_inference_substrate" in exp5011.artifact_schema_errors(unknown_substrate)

    for target in ("su15", "ft09", "ls20"):
        rotation_violation = dict(artifact, target_game=target)
        rotation_violation["reproducibility_checksum"] = exp5011.stable_checksum(
            rotation_violation
        )
        assert "success_target_rotation_violation" in exp5011.artifact_schema_errors(
            rotation_violation
        )


def test_scenario_learn_5011_preconditions_and_main_write_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-LEARN-5011-BLOCKED-PRECONDITION: CLI writes terminal artifacts."""

    monkeypatch.setattr(exp5011.previous, "check_preconditions", lambda game=None: _preconditions("r11l"))
    ok_preconditions = exp5011.check_preconditions("r11l")
    blocked_preconditions = {**_preconditions("su15"), "ok": False}

    assert ok_preconditions["target_rotation"]["ok"] is True
    assert blocked_preconditions["ok"] is False
    assert exp5011._precondition_failure(blocked_preconditions).startswith("target_rotation_")

    ckpt = tmp_path / "models" / "arc_verifier_r11l.json"
    ckpt.parent.mkdir()
    ckpt.write_text('{"ok": true}\n', encoding="utf-8")
    loop_result = tmp_path / "results" / "arc_loop_solve_r11l.json"
    loop_result.parent.mkdir()
    loop_result.write_text(json.dumps(_loop_result("r11l")), encoding="utf-8")
    output = tmp_path / "results" / "experiment_5011_self_play_verifier_checkpoint.json"

    monkeypatch.setattr(exp5011, "REPO", tmp_path)
    monkeypatch.setattr(exp5011, "ARTIFACT", output)
    monkeypatch.setattr(exp5011, "check_preconditions", lambda game=None: _preconditions("r11l"))

    ok = exp5011.main(
        [
            "--game",
            "r11l",
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
    assert written["honest_verdict"] == exp5011.SUCCESS_VERDICT
    assert written["result_path"] == exp5011.RESULT_RELATIVE_PATH

    monkeypatch.setattr(
        exp5011,
        "check_preconditions",
        lambda game=None: {
            **_preconditions("su15", ok=False),
            "target_rotation": {"ok": False, "game": "su15"},
            "target_selection": {"game": "su15"},
        },
    )
    blocked_code = exp5011.main(["--game", "su15", "--duration-s", "1.0"])
    blocked = json.loads(output.read_text(encoding="utf-8"))

    assert blocked_code == 0
    assert blocked["honest_verdict"].startswith("blocked_target_rotation_")
    assert blocked["target_game"] == "su15"
