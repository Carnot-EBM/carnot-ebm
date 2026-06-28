"""Tests for Exp 4906 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-ARC-WMTE-4906,
SCENARIO-ARC-WMTE-4906-CHECKPOINT-REFRESHED,
SCENARIO-ARC-WMTE-4906-RESIDUAL-NO-FABRICATION,
SCENARIO-ARC-WMTE-4906-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4895_self_play_verifier_checkpoint as exp4895
from carnot import experiment_4906_self_play_verifier_checkpoint as exp4906


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(game: str, reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "states_expanded": 17,
        "learned_verifier_checkpoint": f"models/arc_verifier_{game}.json",
        "solve_provenance": "live_agent_self_discovery",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": reproduced,
        },
    }


def _success_artifact(tmp_path: Path, monkeypatch) -> dict[str, object]:
    ckpt = tmp_path / "models" / "arc_verifier_vc33.json"
    ckpt.parent.mkdir()
    ckpt.write_text('{"ok": true}\n', encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    monkeypatch.setattr(exp4906, "REPO", tmp_path)

    summary = exp4906.summarize_loop_result(
        game="vc33",
        loop_result=_loop_result("vc33"),
        loop_result_path="results/arc_loop_solve_vc33.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    return exp4906.build_artifact(
        target_selection={
            "game": "vc33",
            "prior_level": 2,
            "banked": True,
            "reason": "rotated_banked_target_warm_start_vc33",
        },
        loop_summary=summary,
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True, "game": "vc33"},
            "checkpoint_existing": {"ok": True, "path": "models/arc_verifier_vc33.json"},
            "target_rotation": {
                "ok": True,
                "game": "vc33",
                "rotated_off": ["sk48", "ls20", "re86"],
            },
        },
    )


def test_req_arc_wmte_4906_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4906: OpenSpec anchors the 4906 checkpoint contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4906.SPEC_REFS:
        assert ref in spec
    assert exp4906.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4906.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4906_checkpoint_refreshed_success(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4906-CHECKPOINT-REFRESHED: rotated target gate succeeds."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    assert artifact["experiment"] == exp4906.EXPERIMENT
    assert artifact["schema"] == exp4906.SCHEMA
    assert artifact["spec_refs"] == exp4906.SPEC_REFS
    assert artifact["random_seed"] == exp4906.RANDOM_SEED
    assert artifact["field_principles"] == exp4906.FIELD_PRINCIPLES
    assert artifact["honest_verdict"] == "success_self_play_checkpoint_refreshed"
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["checkpoint_path"] == "models/arc_verifier_vc33.json"
    assert artifact["target_game"] == "vc33"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["states_expanded"] == 17
    assert artifact["search_state_count"] == 17
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert int(artifact["checkpoint_mtime_after_ns"]) > int(artifact["checkpoint_mtime_before_ns"])
    assert artifact["checkpoint_mtime_delta_ns"] > 0
    assert artifact["schema_errors"] == []
    assert exp4906.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4906_residual_and_blocked_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4906-RESIDUAL-NO-FABRICATION: failures stay residuals."""

    monkeypatch.setattr(exp4906, "REPO", tmp_path)
    failed = _loop_result("vc33", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp4906.summarize_loop_result(
        game="vc33",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_vc33.json",
        checkpoint_mtime_before_ns=123,
    )
    artifact = exp4906.build_artifact(
        target_selection={
            "game": "vc33",
            "prior_level": 2,
            "banked": True,
            "reason": "rotated_banked_target_warm_start_vc33",
        },
        loop_summary=summary,
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "checkpoint_existing": {"ok": False},
            "target_rotation": {"ok": True, "game": "vc33", "rotated_off": ["sk48", "ls20", "re86"]},
        },
    )
    blocked = exp4906.build_blocked_artifact(
        reason="checkpoint_missing",
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True, "game": "vc33", "prior_level": 2},
            "checkpoint_existing": {"ok": False, "path": "models/arc_verifier_vc33.json"},
            "target_rotation": {"ok": True, "game": "vc33", "rotated_off": ["sk48", "ls20", "re86"]},
        },
        target_game="vc33",
    )
    missing = exp4906.summarize_loop_result(
        game="vc33",
        loop_result=None,
        loop_result_path="results/arc_loop_solve_vc33.json",
        checkpoint_mtime_before_ns="bad",
    )

    assert artifact["honest_verdict"] == "complete_vc33_self_play_residual_reproduction_gate_failed"
    assert artifact["verifier_checkpoint_refreshed"] is False
    assert artifact["checkpoint_path"] is None
    assert artifact["checkpoint_mtime_after_ns"] is None
    assert artifact["checkpoint_mtime_delta_ns"] is None
    assert artifact["schema_errors"] == []
    assert blocked["honest_verdict"] == "blocked_checkpoint_missing"
    assert blocked["target_game"] == "vc33"
    assert blocked["verifier_checkpoint_refreshed"] is False
    assert blocked["schema_errors"] == []
    assert missing["self_play_residual"] == "loop_result_missing"
    assert missing["checkpoint_mtime_before_ns"] is None
    assert missing["search_state_count"] == 0


def test_req_arc_wmte_4906_schema_rejects_stale_metadata_and_recent_targets(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-ARC-WMTE-4906: schema rejects stale identity and recent-target success."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    for field, old_value, expected in (
        ("experiment", exp4895.EXPERIMENT, "experiment_mismatch"),
        ("schema", exp4895.SCHEMA, "schema_mismatch"),
        ("spec_refs", exp4895.SPEC_REFS, "spec_refs_mismatch"),
        ("random_seed", exp4895.RANDOM_SEED, "random_seed_mismatch"),
    ):
        stale = dict(artifact)
        stale[field] = old_value
        stale["reproducibility_checksum"] = exp4906.stable_checksum(stale)
        assert expected in exp4906.artifact_schema_errors(stale)

    wrong_verdict = dict(artifact, honest_verdict="success_vc33_L2_checkpoint_refreshed")
    wrong_verdict["reproducibility_checksum"] = exp4906.stable_checksum(wrong_verdict)
    assert "success_verdict_mismatch" in exp4906.artifact_schema_errors(wrong_verdict)

    bad_principle = dict(artifact, field_principles={})
    bad_principle["reproducibility_checksum"] = exp4906.stable_checksum(bad_principle)
    assert "missing_principle:target_game" in exp4906.artifact_schema_errors(bad_principle)

    missing_result_path = dict(artifact)
    missing_result_path.pop("result_path")
    missing_result_path["reproducibility_checksum"] = exp4906.stable_checksum(
        missing_result_path
    )
    result_path_errors = exp4906.artifact_schema_errors(missing_result_path)
    assert "missing_field:result_path" in result_path_errors
    assert "result_path_mismatch" in result_path_errors

    monkeypatch.setattr(
        exp4906,
        "REQUIRED_FIELDS",
        exp4906.REQUIRED_FIELDS + ("exp4906_only_field",),
    )
    exp4906_only_missing = dict(artifact)
    exp4906_only_missing["reproducibility_checksum"] = exp4906.stable_checksum(
        exp4906_only_missing
    )
    assert "missing_field:exp4906_only_field" in exp4906.artifact_schema_errors(
        exp4906_only_missing
    )

    missing_checkpoint_check = dict(artifact, preconditions_checked={})
    missing_checkpoint_check["reproducibility_checksum"] = exp4906.stable_checksum(
        missing_checkpoint_check
    )
    assert "preconditions_missing_checkpoint_check" in exp4906.artifact_schema_errors(
        missing_checkpoint_check
    )

    success_without_gate = dict(artifact, reproduced_levels="bad")
    success_without_gate["reproducibility_checksum"] = exp4906.stable_checksum(
        success_without_gate
    )
    assert "success_verdict_without_gate" in exp4906.artifact_schema_errors(
        success_without_gate
    )

    for recent_target in ("sk48", "ls20", "re86"):
        recent_success = dict(artifact, target_game=recent_target)
        recent_success["reproducibility_checksum"] = exp4906.stable_checksum(recent_success)
        assert "success_target_rotation_violation" in exp4906.artifact_schema_errors(
            recent_success
        )


def test_req_arc_wmte_4906_helpers_and_main_write_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-ARC-WMTE-4906: IO helpers and main write vc33 terminal artifacts."""

    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    ckpt = tmp_path / "models" / "arc_verifier_vc33.json"
    ckpt.write_text("{}", encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    loop_path = tmp_path / "results" / "arc_loop_solve_vc33.json"
    loop_path.write_text(json.dumps(_loop_result("vc33")), encoding="utf-8")
    list_payload = tmp_path / "results" / "list.json"
    list_payload.write_text("[]", encoding="utf-8")

    monkeypatch.setattr(exp4906, "REPO", tmp_path)
    monkeypatch.setattr(exp4906, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4906, "ARTIFACT", tmp_path / exp4906.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4906,
        "check_preconditions",
        lambda game=None: {
            "ok": True,
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True, "game": "vc33", "prior_level": 2},
            "checkpoint_existing": {"ok": True, "path": "models/arc_verifier_vc33.json"},
            "target_rotation": {"ok": True, "game": "vc33", "rotated_off": ["sk48", "ls20", "re86"]},
            "target_selection": {
                "game": "vc33",
                "prior_level": 2,
                "banked": True,
                "reason": "rotated_banked_target_warm_start_vc33",
            },
        },
    )

    assert exp4906._target_game(None) == "vc33"  # noqa: SLF001
    assert exp4906._target_game("sp80") == "sp80"  # noqa: SLF001
    assert exp4906._read_loop_result(tmp_path / "missing.json") is None  # noqa: SLF001
    assert exp4906._read_loop_result(list_payload) is None  # noqa: SLF001
    assert exp4906._read_loop_result(loop_path)["game"] == "vc33"  # noqa: SLF001
    assert exp4906._relative_path(loop_path) == "results/arc_loop_solve_vc33.json"  # noqa: SLF001
    assert exp4906._relative_path(Path("/outside.json")) == "/outside.json"  # noqa: SLF001
    assert exp4906._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": False},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": True},
            "checkpoint_existing": {"ok": True},
        }
    ) == "offline_arcade_missing"
    assert exp4906._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": False, "game": "sk48"},
            "checkpoint_existing": {"ok": True},
        }
    ) == "target_rotation_sk48_ls20_re86_disallowed"
    assert exp4906._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": True},
            "checkpoint_existing": {"ok": False},
        }
    ) == "checkpoint_missing"
    assert exp4906._precondition_failure(  # noqa: SLF001
        {
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True},
            "target_rotation": {"ok": True},
            "checkpoint_existing": {"ok": True},
        }
    ) is None
    monkeypatch.setattr(exp4906.previous, "_precondition_failure", lambda preconditions: None)
    assert exp4906._precondition_failure(  # noqa: SLF001
        {"target_rotation": {"ok": True}, "checkpoint_existing": {"ok": False}}
    ) == "checkpoint_missing"
    assert exp4906._precondition_failure(  # noqa: SLF001
        {"checkpoint_existing": {"ok": False}}
    ) == "target_rotation_sk48_ls20_re86_disallowed"

    assert exp4906.main(["--checkpoint-mtime-before-ns", str(before_ns)]) == 0
    written = json.loads((tmp_path / exp4906.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "success_self_play_checkpoint_refreshed"
    assert written["target_game"] == "vc33"
    assert written["checkpoint_path"] == "models/arc_verifier_vc33.json"
    assert written["schema_errors"] == []

    blocked_path = tmp_path / "blocked" / exp4906.RESULT_RELATIVE_PATH
    monkeypatch.setattr(exp4906, "ARTIFACT", blocked_path)
    monkeypatch.setattr(
        exp4906,
        "check_preconditions",
        lambda game=None: {
            "ok": False,
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True, "game": "vc33", "prior_level": 2},
            "checkpoint_existing": {"ok": False, "path": "models/arc_verifier_vc33.json"},
            "target_rotation": {"ok": True, "game": "vc33", "rotated_off": ["sk48", "ls20", "re86"]},
            "target_selection": {
                "game": "vc33",
                "prior_level": 2,
                "banked": True,
                "reason": "rotated_banked_target_warm_start_vc33",
            },
        },
    )

    assert exp4906.main([]) == 0
    blocked = json.loads(blocked_path.read_text(encoding="utf-8"))
    assert blocked["honest_verdict"] == "blocked_checkpoint_missing"
    assert blocked["target_game"] == "vc33"
    assert blocked["checkpoint_path"] is None
    assert blocked["schema_errors"] == []
