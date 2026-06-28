"""Tests for Exp 4927 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-LEARN-4927,
SCENARIO-LEARN-4927-CHECKPOINT-REFRESHED,
SCENARIO-LEARN-4927-RESIDUAL-NO-FABRICATION,
SCENARIO-LEARN-4927-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4916_self_play_verifier_checkpoint as exp4916
from carnot import experiment_4927_self_play_verifier_checkpoint as exp4927


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _loop_result(game: str, reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "states_expanded": 73,
        "learned_verifier_checkpoint": f"models/arc_verifier_{game}.json",
        "solve_provenance": "live_agent_self_discovery",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": reproduced,
        },
    }


def _preconditions(game: str = "lf52", *, ok: bool = True) -> dict[str, object]:
    return {
        "ok": ok,
        "offline_arcade": {"ok": True},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_banked": {"ok": True, "game": game, "prior_level": 2},
        "checkpoint_existing": {"ok": True, "path": f"models/arc_verifier_{game}.json"},
        "target_rotation": {
            "ok": True,
            "game": game,
            "rotated_off": list(exp4927.DISALLOWED_TARGET_GAMES),
        },
        "target_selection": {
            "game": game,
            "prior_level": 2,
            "banked": True,
            "reason": f"rotated_banked_target_warm_start_{game}",
        },
    }


def _success_artifact(tmp_path: Path, monkeypatch) -> dict[str, object]:
    ckpt = tmp_path / "models" / "arc_verifier_lf52.json"
    ckpt.parent.mkdir()
    ckpt.write_text('{"ok": true}\n', encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    monkeypatch.setattr(exp4927, "REPO", tmp_path)

    summary = exp4927.summarize_loop_result(
        game="lf52",
        loop_result=_loop_result("lf52"),
        loop_result_path="results/arc_loop_solve_lf52.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    return exp4927.build_artifact(
        target_selection={
            "game": "lf52",
            "prior_level": 2,
            "banked": True,
            "reason": "rotated_banked_target_warm_start_lf52",
        },
        loop_summary=summary,
        preconditions_checked=_preconditions("lf52"),
    )


def test_req_learn_4927_spec_declares_required_contract() -> None:
    """REQ-LEARN-4927: OpenSpec anchors the 4927 checkpoint contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4927.SPEC_REFS:
        assert ref in spec
    assert exp4927.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4927.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_learn_4927_checkpoint_refreshed_success(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-LEARN-4927-CHECKPOINT-REFRESHED: rotated target gate succeeds."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    assert artifact["experiment"] == exp4927.EXPERIMENT
    assert artifact["schema"] == exp4927.SCHEMA
    assert artifact["spec_refs"] == exp4927.SPEC_REFS
    assert artifact["random_seed"] == exp4927.RANDOM_SEED
    assert artifact["field_principles"] == exp4927.FIELD_PRINCIPLES
    assert artifact["honest_verdict"] == exp4927.SUCCESS_VERDICT
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["checkpoint_path"] == "models/arc_verifier_lf52.json"
    assert artifact["target_game"] == "lf52"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["states_expanded"] == 73
    assert artifact["search_state_count"] == 73
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert int(artifact["checkpoint_mtime_after_ns"]) > int(artifact["checkpoint_mtime_before_ns"])
    assert artifact["checkpoint_mtime_delta_ns"] > 0
    assert artifact["schema_errors"] == []
    assert exp4927.artifact_schema_errors(artifact) == []


def test_scenario_learn_4927_residual_and_blocked_paths(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-LEARN-4927-RESIDUAL-NO-FABRICATION: failures stay residuals."""

    monkeypatch.setattr(exp4927, "REPO", tmp_path)
    failed = _loop_result("lf52", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp4927.summarize_loop_result(
        game="lf52",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_lf52.json",
        checkpoint_mtime_before_ns=123,
    )
    residual = exp4927.build_artifact(
        target_selection={
            "game": "lf52",
            "prior_level": 2,
            "banked": True,
            "reason": "rotated_banked_target_warm_start_lf52",
        },
        loop_summary=summary,
        preconditions_checked={**_preconditions("lf52"), "checkpoint_existing": {"ok": False}},
    )
    blocked = exp4927.build_blocked_artifact(
        reason="checkpoint_missing",
        preconditions_checked={
            **_preconditions("lf52", ok=False),
            "checkpoint_existing": {"ok": False, "path": "models/arc_verifier_lf52.json"},
        },
        target_game="lf52",
    )
    missing = exp4927.summarize_loop_result(
        game="lf52",
        loop_result=None,
        loop_result_path="results/arc_loop_solve_lf52.json",
        checkpoint_mtime_before_ns="bad",
    )

    assert residual["honest_verdict"] == "complete_lf52_self_play_residual_reproduction_gate_failed"
    assert residual["verifier_checkpoint_refreshed"] is False
    assert residual["checkpoint_path"] is None
    assert residual["checkpoint_mtime_after_ns"] is None
    assert residual["checkpoint_mtime_delta_ns"] is None
    assert residual["schema_errors"] == []
    assert blocked["honest_verdict"] == "blocked_checkpoint_missing"
    assert blocked["target_game"] == "lf52"
    assert blocked["verifier_checkpoint_refreshed"] is False
    assert blocked["schema_errors"] == []
    assert missing["self_play_residual"] == "loop_result_missing"
    assert missing["checkpoint_mtime_before_ns"] is None
    assert missing["search_state_count"] == 0


def test_req_learn_4927_schema_rejects_stale_metadata_and_disallowed_targets(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-LEARN-4927: schema rejects stale identity and excluded targets."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    for field, old_value, expected in (
        ("experiment", exp4916.EXPERIMENT, "experiment_mismatch"),
        ("schema", exp4916.SCHEMA, "schema_mismatch"),
        ("spec_refs", exp4916.SPEC_REFS, "spec_refs_mismatch"),
        ("random_seed", exp4916.RANDOM_SEED, "random_seed_mismatch"),
    ):
        stale = dict(artifact)
        stale[field] = old_value
        stale["reproducibility_checksum"] = exp4927.stable_checksum(stale)
        assert expected in exp4927.artifact_schema_errors(stale)

    wrong_verdict = dict(artifact, honest_verdict="success_lf52_L2_checkpoint_refreshed")
    wrong_verdict["reproducibility_checksum"] = exp4927.stable_checksum(wrong_verdict)
    assert "success_verdict_mismatch" in exp4927.artifact_schema_errors(wrong_verdict)

    bad_principle = dict(artifact, field_principles={})
    bad_principle["reproducibility_checksum"] = exp4927.stable_checksum(bad_principle)
    assert "missing_principle:target_game" in exp4927.artifact_schema_errors(bad_principle)

    missing_result_path = dict(artifact)
    missing_result_path.pop("result_path")
    missing_result_path["reproducibility_checksum"] = exp4927.stable_checksum(
        missing_result_path
    )
    result_path_errors = exp4927.artifact_schema_errors(missing_result_path)
    assert "missing_field:result_path" in result_path_errors
    assert "result_path_mismatch" in result_path_errors

    monkeypatch.setattr(
        exp4927,
        "REQUIRED_FIELDS",
        exp4927.REQUIRED_FIELDS + ("exp4927_only_field",),
    )
    exp4927_only_missing = dict(artifact)
    exp4927_only_missing["reproducibility_checksum"] = exp4927.stable_checksum(
        exp4927_only_missing
    )
    assert "missing_field:exp4927_only_field" in exp4927.artifact_schema_errors(
        exp4927_only_missing
    )

    missing_checkpoint_check = dict(artifact, preconditions_checked={})
    missing_checkpoint_check["reproducibility_checksum"] = exp4927.stable_checksum(
        missing_checkpoint_check
    )
    assert "preconditions_missing_checkpoint_check" in exp4927.artifact_schema_errors(
        missing_checkpoint_check
    )

    success_without_gate = dict(artifact, reproduced_levels="bad")
    success_without_gate["reproducibility_checksum"] = exp4927.stable_checksum(
        success_without_gate
    )
    assert "success_verdict_without_gate" in exp4927.artifact_schema_errors(
        success_without_gate
    )

    for disallowed_target in exp4927.DISALLOWED_TARGET_GAMES:
        disallowed_success = dict(artifact, target_game=disallowed_target)
        disallowed_success["reproducibility_checksum"] = exp4927.stable_checksum(
            disallowed_success
        )
        assert "success_target_rotation_violation" in exp4927.artifact_schema_errors(
            disallowed_success
        )


def test_req_learn_4927_helpers_and_main_write_artifacts(tmp_path: Path, monkeypatch) -> None:
    """REQ-LEARN-4927: IO helpers and main write lf52 terminal artifacts."""

    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    ckpt = tmp_path / "models" / "arc_verifier_lf52.json"
    ckpt.write_text("{}", encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    loop_path = tmp_path / "results" / "arc_loop_solve_lf52.json"
    loop_path.write_text(json.dumps(_loop_result("lf52")), encoding="utf-8")
    list_payload = tmp_path / "results" / "list.json"
    list_payload.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(exp4927, "REPO", tmp_path)
    monkeypatch.setattr(exp4927, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4927, "ARTIFACT", tmp_path / exp4927.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp4927, "check_preconditions", lambda game=None: _preconditions("lf52"))

    assert exp4927._target_game(None) == "lf52"  # noqa: SLF001
    assert exp4927._target_game("ar25") == "ar25"  # noqa: SLF001
    assert exp4927._read_loop_result(tmp_path / "missing.json") is None  # noqa: SLF001
    assert exp4927._read_loop_result(list_payload) is None  # noqa: SLF001
    assert exp4927._read_loop_result(loop_path)["game"] == "lf52"  # noqa: SLF001
    assert exp4927._relative_path(loop_path) == "results/arc_loop_solve_lf52.json"  # noqa: SLF001
    assert exp4927._relative_path(Path("/outside.json")) == "/outside.json"  # noqa: SLF001
    assert exp4927._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": False},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": True},
            "checkpoint_existing": {"ok": True},
        }
    ) == "offline_arcade_missing"
    assert exp4927._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": False, "game": "bp35"},
            "checkpoint_existing": {"ok": True},
        }
    ) == "target_rotation_bp35_vc33_sk48_ls20_re86_sp80_su15_disallowed"
    assert exp4927._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": True},
            "checkpoint_existing": {"ok": False},
        }
    ) == "checkpoint_missing"
    assert exp4927._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": True},
            "checkpoint_existing": {"ok": True},
        }
    ) is None
    monkeypatch.setattr(exp4927.previous, "_precondition_failure", lambda preconditions: None)
    assert exp4927._precondition_failure(  # noqa: SLF001
        {"target_rotation": {"ok": True}, "checkpoint_existing": {"ok": False}}
    ) == "checkpoint_missing"
    assert exp4927._precondition_failure(  # noqa: SLF001
        {"checkpoint_existing": {"ok": False}}
    ) == "target_rotation_bp35_vc33_sk48_ls20_re86_sp80_su15_disallowed"

    assert exp4927.main(["--checkpoint-mtime-before-ns", str(before_ns)]) == 0
    written = json.loads((tmp_path / exp4927.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == exp4927.SUCCESS_VERDICT
    assert written["target_game"] == "lf52"
    assert written["checkpoint_path"] == "models/arc_verifier_lf52.json"
    assert written["schema_errors"] == []

    blocked_path = tmp_path / "blocked" / exp4927.RESULT_RELATIVE_PATH
    monkeypatch.setattr(exp4927, "ARTIFACT", blocked_path)
    monkeypatch.setattr(
        exp4927,
        "check_preconditions",
        lambda game=None: {
            **_preconditions("lf52", ok=False),
            "checkpoint_existing": {"ok": False, "path": "models/arc_verifier_lf52.json"},
        },
    )

    assert exp4927.main([]) == 0
    blocked = json.loads(blocked_path.read_text(encoding="utf-8"))
    assert blocked["honest_verdict"] == "blocked_checkpoint_missing"
    assert blocked["target_game"] == "lf52"
    assert blocked["checkpoint_path"] is None
    assert blocked["schema_errors"] == []
