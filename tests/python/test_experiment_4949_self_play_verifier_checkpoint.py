"""Tests for Exp 4949 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-LEARN-4949,
SCENARIO-LEARN-4949-CHECKPOINT-REFRESHED,
SCENARIO-LEARN-4949-SUBSTRATE-FIX,
SCENARIO-LEARN-4949-RESIDUAL-NO-FABRICATION,
SCENARIO-LEARN-4949-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.adversarial_verify as av
from carnot import experiment_4938_self_play_verifier_checkpoint as exp4938
from carnot import experiment_4949_self_play_verifier_checkpoint as exp4949


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _loop_result(game: str, reached_level: int = 3, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "states_expanded": 91,
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


def _preconditions(game: str = "lp85", *, ok: bool = True) -> dict[str, object]:
    return {
        "ok": ok,
        "offline_arcade": {"ok": True},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_banked": {"ok": True, "game": game, "prior_level": 5},
        "checkpoint_existing": {"ok": True, "path": f"models/arc_verifier_{game}.json"},
        "target_rotation": {
            "ok": True,
            "game": game,
            "rotated_off": list(exp4949.DISALLOWED_TARGET_GAMES),
        },
        "target_selection": {
            "game": game,
            "prior_level": 5,
            "banked": True,
            "reason": f"rotated_banked_target_warm_start_{game}",
        },
    }


def _success_artifact(tmp_path: Path, monkeypatch) -> dict[str, object]:
    ckpt = tmp_path / "models" / "arc_verifier_lp85.json"
    ckpt.parent.mkdir()
    ckpt.write_text('{"ok": true}\n', encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    monkeypatch.setattr(exp4949, "REPO", tmp_path)

    summary = exp4949.summarize_loop_result(
        game="lp85",
        loop_result=_loop_result("lp85"),
        loop_result_path="results/arc_loop_solve_lp85.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    return exp4949.build_artifact(
        target_selection={
            "game": "lp85",
            "prior_level": 5,
            "banked": True,
            "reason": "rotated_banked_target_warm_start_lp85",
        },
        loop_summary=summary,
        preconditions_checked=_preconditions("lp85"),
        duration_s=2.5,
        flag_resolved=True,
    )


def test_req_learn_4949_spec_declares_required_contract() -> None:
    """REQ-LEARN-4949: OpenSpec anchors the honest-substrate checkpoint contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4949.SPEC_REFS:
        assert ref in spec
    assert exp4949.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4949.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_learn_4949_checkpoint_refreshed_success(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-LEARN-4949-CHECKPOINT-REFRESHED: lp85 checkpoint refresh succeeds."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    assert artifact["experiment"] == exp4949.EXPERIMENT
    assert artifact["schema"] == exp4949.SCHEMA
    assert artifact["spec_refs"] == exp4949.SPEC_REFS
    assert artifact["random_seed"] == exp4949.RANDOM_SEED
    assert artifact["field_principles"] == exp4949.FIELD_PRINCIPLES
    assert artifact["honest_verdict"] == exp4949.SUCCESS_VERDICT
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["checkpoint_path"] == "models/arc_verifier_lp85.json"
    assert artifact["target_game"] == "lp85"
    assert artifact["inference_substrate"] == exp4949.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == 2.5
    assert artifact["flag_resolved"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["states_expanded"] == 91
    assert "search_state_count" not in artifact
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert int(artifact["checkpoint_mtime_after_ns"]) > int(artifact["checkpoint_mtime_before_ns"])
    assert artifact["checkpoint_mtime_delta_ns"] > 0
    assert artifact["schema_errors"] == []
    assert exp4949.artifact_schema_errors(artifact) == []


def test_scenario_learn_4949_substrate_fix_avoids_duration_too_short(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-LEARN-4949-SUBSTRATE-FIX: offline gate uses verifier substrate."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    floor = av.duration_floor_for_artifact(artifact)
    flags: list[object] = []
    av.check_duration_vs_claim(artifact, flags)

    assert floor == {
        "substrate": exp4949.INFERENCE_SUBSTRATE,
        "min_duration_s": 1.0,
        "reason": "verifier_scoring",
    }
    assert [getattr(flag, "kind", None) for flag in flags] == []

    out = tmp_path / "artifact.json"
    exp4949.write_artifact(artifact, out)
    report = av.verify_artifact(out)
    critical = [flag for flag in report["flags"] if flag.get("severity") == "critical"]
    assert [flag["kind"] for flag in critical] == []


def test_scenario_learn_4949_residual_and_blocked_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-LEARN-4949-RESIDUAL-NO-FABRICATION: failures stay residuals."""

    monkeypatch.setattr(exp4949, "REPO", tmp_path)
    failed = _loop_result("lp85", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp4949.summarize_loop_result(
        game="lp85",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_lp85.json",
        checkpoint_mtime_before_ns=123,
    )
    residual = exp4949.build_artifact(
        target_selection={
            "game": "lp85",
            "prior_level": 5,
            "banked": True,
            "reason": "rotated_banked_target_warm_start_lp85",
        },
        loop_summary=summary,
        preconditions_checked={**_preconditions("lp85"), "checkpoint_existing": {"ok": False}},
        duration_s=1.2,
        flag_resolved=True,
    )
    blocked = exp4949.build_blocked_artifact(
        reason="checkpoint_missing",
        preconditions_checked={
            **_preconditions("lp85", ok=False),
            "checkpoint_existing": {"ok": False, "path": "models/arc_verifier_lp85.json"},
        },
        target_game="lp85",
        duration_s=1.0,
        flag_resolved=True,
    )
    missing = exp4949.summarize_loop_result(
        game="lp85",
        loop_result=None,
        loop_result_path="results/arc_loop_solve_lp85.json",
        checkpoint_mtime_before_ns="bad",
    )

    assert residual["honest_verdict"] == "complete_lp85_self_play_residual_reproduction_gate_failed"
    assert residual["verifier_checkpoint_refreshed"] is False
    assert residual["checkpoint_path"] is None
    assert residual["checkpoint_mtime_after_ns"] is None
    assert residual["checkpoint_mtime_delta_ns"] is None
    assert residual["flag_resolved"] is True
    assert residual["schema_errors"] == []
    assert blocked["honest_verdict"] == "blocked_checkpoint_missing"
    assert blocked["target_game"] == "lp85"
    assert blocked["verifier_checkpoint_refreshed"] is False
    assert blocked["schema_errors"] == []
    assert missing["self_play_residual"] == "loop_result_missing"
    assert missing["checkpoint_mtime_before_ns"] is None
    assert missing["search_state_count"] == 0


def test_req_learn_4949_schema_rejects_stale_metadata_and_bad_substrate(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-LEARN-4949: schema rejects stale identity, targets, and substrate drift."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    for field, old_value, expected in (
        ("experiment", exp4938.EXPERIMENT, "experiment_mismatch"),
        ("schema", exp4938.SCHEMA, "schema_mismatch"),
        ("spec_refs", exp4938.SPEC_REFS, "spec_refs_mismatch"),
        ("random_seed", exp4938.RANDOM_SEED, "random_seed_mismatch"),
    ):
        stale = dict(artifact)
        stale[field] = old_value
        stale["reproducibility_checksum"] = exp4949.stable_checksum(stale)
        assert expected in exp4949.artifact_schema_errors(stale)

    too_fast = dict(artifact, duration_s=0.25)
    too_fast["reproducibility_checksum"] = exp4949.stable_checksum(too_fast)
    assert "duration_too_short_for_verifier_ensemble" in exp4949.artifact_schema_errors(
        too_fast
    )

    live_too_fast = dict(artifact, inference_substrate="live_llm_inference", duration_s=2.5)
    live_too_fast["reproducibility_checksum"] = exp4949.stable_checksum(live_too_fast)
    live_errors = exp4949.artifact_schema_errors(live_too_fast)
    assert "inference_substrate_mismatch" in live_errors
    assert "duration_too_short_for_live_llm_inference" in live_errors

    unknown_substrate = dict(artifact, inference_substrate="deterministic_verifier")
    unknown_substrate["reproducibility_checksum"] = exp4949.stable_checksum(unknown_substrate)
    assert "unknown_inference_substrate" in exp4949.artifact_schema_errors(
        unknown_substrate
    )

    nonfinite_duration = dict(artifact, duration_s=float("inf"))
    nonfinite_duration["reproducibility_checksum"] = exp4949.stable_checksum(
        nonfinite_duration
    )
    assert "duration_s_not_finite" in exp4949.artifact_schema_errors(nonfinite_duration)

    unresolved = dict(artifact, flag_resolved=False)
    unresolved["reproducibility_checksum"] = exp4949.stable_checksum(unresolved)
    assert "success_without_flag_resolved" in exp4949.artifact_schema_errors(unresolved)

    flag_not_bool = dict(artifact, flag_resolved="yes")
    flag_not_bool["reproducibility_checksum"] = exp4949.stable_checksum(flag_not_bool)
    assert "flag_resolved_must_be_bool" in exp4949.artifact_schema_errors(flag_not_bool)

    stamped = dict(artifact, flagged_adversarial=True, true_live_recheck="critical")
    stamped["reproducibility_checksum"] = exp4949.stable_checksum(stamped)
    assert "flag_resolved_contradicts_critical_recheck" in exp4949.artifact_schema_errors(
        stamped
    )

    wrong_success = dict(artifact, honest_verdict="success_lp85_L3_checkpoint_refreshed")
    wrong_success["reproducibility_checksum"] = exp4949.stable_checksum(wrong_success)
    assert "success_verdict_mismatch" in exp4949.artifact_schema_errors(wrong_success)

    missing_preconditions = dict(artifact, preconditions_checked={})
    missing_preconditions["reproducibility_checksum"] = exp4949.stable_checksum(
        missing_preconditions
    )
    assert "preconditions_missing_checkpoint_check" in exp4949.artifact_schema_errors(
        missing_preconditions
    )

    wrong_result_path = dict(artifact, result_path="results/not_4949.json")
    wrong_result_path["reproducibility_checksum"] = exp4949.stable_checksum(
        wrong_result_path
    )
    assert "result_path_mismatch" in exp4949.artifact_schema_errors(wrong_result_path)

    missing_duration = dict(artifact)
    missing_duration.pop("duration_s")
    missing_duration["reproducibility_checksum"] = exp4949.stable_checksum(missing_duration)
    assert "missing_field:duration_s" in exp4949.artifact_schema_errors(missing_duration)

    for disallowed_target in exp4949.DISALLOWED_TARGET_GAMES:
        disallowed_success = dict(artifact, target_game=disallowed_target)
        disallowed_success["reproducibility_checksum"] = exp4949.stable_checksum(
            disallowed_success
        )
        assert "success_target_rotation_violation" in exp4949.artifact_schema_errors(
            disallowed_success
        )


def test_req_learn_4949_helpers_and_main_write_artifacts(tmp_path: Path, monkeypatch) -> None:
    """REQ-LEARN-4949: IO helpers and main write lp85 terminal artifacts."""

    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    ckpt = tmp_path / "models" / "arc_verifier_lp85.json"
    ckpt.write_text("{}", encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    loop_path = tmp_path / "results" / "arc_loop_solve_lp85.json"
    loop_path.write_text(json.dumps(_loop_result("lp85")), encoding="utf-8")
    list_payload = tmp_path / "results" / "list.json"
    list_payload.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(exp4949, "REPO", tmp_path)
    monkeypatch.setattr(exp4949, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4949, "ARTIFACT", tmp_path / exp4949.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp4949, "check_preconditions", lambda game=None: _preconditions("lp85"))

    assert exp4949._target_game(None) == "lp85"  # noqa: SLF001
    assert exp4949._target_game("cd82") == "cd82"  # noqa: SLF001
    assert exp4949._as_int("bad", 7) == 7  # noqa: SLF001
    assert exp4949._read_loop_result(tmp_path / "missing.json") is None  # noqa: SLF001
    assert exp4949._read_loop_result(list_payload) is None  # noqa: SLF001
    assert exp4949._read_loop_result(loop_path)["game"] == "lp85"  # noqa: SLF001
    assert exp4949._relative_path(loop_path) == "results/arc_loop_solve_lp85.json"  # noqa: SLF001
    assert exp4949._relative_path(Path("/outside.json")) == "/outside.json"  # noqa: SLF001
    assert exp4949._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": False},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": True},
            "checkpoint_existing": {"ok": True},
        }
    ) == "offline_arcade_missing"
    assert exp4949._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": False, "game": "ar25"},
            "checkpoint_existing": {"ok": True},
        }
    ) == "target_rotation_ar25_bp35_vc33_sk48_ls20_re86_disallowed"
    assert exp4949._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": True},
            "checkpoint_existing": {"ok": False},
        }
    ) == "checkpoint_missing"
    assert exp4949._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": True},
            "checkpoint_existing": {"ok": True},
        }
    ) is None
    monkeypatch.setattr(exp4949.previous, "_precondition_failure", lambda preconditions: None)
    assert exp4949._precondition_failure(  # noqa: SLF001
        {"target_rotation": {"ok": True}, "checkpoint_existing": {"ok": False}}
    ) == "checkpoint_missing"
    assert exp4949._precondition_failure(  # noqa: SLF001
        {"checkpoint_existing": {"ok": False}}
    ) == "target_rotation_ar25_bp35_vc33_sk48_ls20_re86_disallowed"

    assert exp4949.main(["--checkpoint-mtime-before-ns", str(before_ns), "--duration-s", "2.0"]) == 0
    written = json.loads((tmp_path / exp4949.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == exp4949.SUCCESS_VERDICT
    assert written["target_game"] == "lp85"
    assert written["checkpoint_path"] == "models/arc_verifier_lp85.json"
    assert written["duration_s"] == 2.0
    assert written["flag_resolved"] is True
    assert written["schema_errors"] == []

    blocked_path = tmp_path / "blocked" / exp4949.RESULT_RELATIVE_PATH
    monkeypatch.setattr(exp4949, "ARTIFACT", blocked_path)
    monkeypatch.setattr(
        exp4949,
        "check_preconditions",
        lambda game=None: {
            **_preconditions("lp85", ok=False),
            "checkpoint_existing": {"ok": False, "path": "models/arc_verifier_lp85.json"},
        },
    )

    assert exp4949.main(["--duration-s", "1.0", "--flag-unresolved"]) == 0
    blocked = json.loads(blocked_path.read_text(encoding="utf-8"))
    assert blocked["honest_verdict"] == "blocked_checkpoint_missing"
    assert blocked["target_game"] == "lp85"
    assert blocked["checkpoint_path"] is None
    assert blocked["flag_resolved"] is False
    assert blocked["schema_errors"] == []

    default_duration = exp4949.build_blocked_artifact(
        reason="offline_arcade_missing",
        preconditions_checked={
            **_preconditions("lp85", ok=False),
            "offline_arcade": {"ok": False},
        },
        target_game="lp85",
    )
    assert default_duration["duration_s"] == exp4949.VERIFIER_MIN_DURATION_S
