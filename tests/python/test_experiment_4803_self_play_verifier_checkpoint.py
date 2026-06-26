"""Tests for Exp 4803 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-ARC-WMTE-4803,
SCENARIO-ARC-WMTE-4803-CHECKPOINT-REFRESHED,
SCENARIO-ARC-WMTE-4803-RESIDUAL-NO-FABRICATION,
SCENARIO-ARC-WMTE-4803-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4793_self_play_verifier_checkpoint as exp4793
from carnot import experiment_4803_self_play_verifier_checkpoint as exp4803


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(game: str, reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "states_expanded": 103,
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
    monkeypatch.setattr(exp4803, "REPO", tmp_path)

    summary = exp4803.summarize_loop_result(
        game="re86",
        loop_result=_loop_result("re86"),
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    return exp4803.build_artifact(
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


def test_req_arc_wmte_4803_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4803: OpenSpec declares the 4803 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4803.SPEC_REFS:
        assert ref in spec
    assert exp4803.RESULT_RELATIVE_PATH in spec
    assert "checkpoint_mtime_delta_ns" in spec
    assert "search_state_count" in spec
    for field, principle in exp4803.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4803_checkpoint_refreshed_success(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4803-CHECKPOINT-REFRESHED: mtime advance plus gate green succeeds."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    assert artifact["experiment"] == exp4803.EXPERIMENT
    assert artifact["schema"] == exp4803.SCHEMA
    assert artifact["spec_refs"] == exp4803.SPEC_REFS
    assert artifact["random_seed"] == exp4803.RANDOM_SEED
    assert artifact["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["states_expanded"] == 103
    assert artifact["search_state_count"] == 103
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert int(artifact["checkpoint_mtime_after_ns"]) > int(artifact["checkpoint_mtime_before_ns"])
    assert artifact["checkpoint_mtime_delta_ns"] > 0
    assert artifact["schema_errors"] == []
    assert exp4803.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4803_residual_does_not_fabricate_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4803-RESIDUAL-NO-FABRICATION: failed gates stay residuals."""

    monkeypatch.setattr(exp4803, "REPO", tmp_path)
    failed = _loop_result("re86", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp4803.summarize_loop_result(
        game="re86",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=123,
    )
    artifact = exp4803.build_artifact(
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
    assert artifact["search_state_count"] == 103
    assert artifact["schema_errors"] == []


def test_scenario_arc_wmte_4803_blocked_precondition_artifact() -> None:
    """SCENARIO-ARC-WMTE-4803-BLOCKED-PRECONDITION: blocked runs remain auditable."""

    artifact = exp4803.build_blocked_artifact(
        reason="target_not_banked",
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": False, "game": "sb26", "prior_level": 0},
        },
        target_game="sb26",
    )

    assert artifact["experiment"] == exp4803.EXPERIMENT
    assert artifact["honest_verdict"] == "blocked_target_not_banked"
    assert artifact["target_game"] == "sb26"
    assert artifact["verifier_checkpoint_refreshed"] is False
    assert artifact["self_play_residual"] == "target_not_banked"
    assert artifact["search_state_count"] == 0
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4803_schema_rejects_stale_4793_metadata(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4803: schema prevents accidental reuse of the 4793 identity."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    stale_experiment = dict(artifact)
    stale_experiment["experiment"] = exp4793.EXPERIMENT
    stale_experiment["reproducibility_checksum"] = exp4803.stable_checksum(stale_experiment)
    assert "experiment_mismatch" in exp4803.artifact_schema_errors(stale_experiment)

    stale_schema = dict(artifact)
    stale_schema["schema"] = exp4793.SCHEMA
    stale_schema["reproducibility_checksum"] = exp4803.stable_checksum(stale_schema)
    assert "schema_mismatch" in exp4803.artifact_schema_errors(stale_schema)

    stale_refs = dict(artifact)
    stale_refs["spec_refs"] = exp4793.SPEC_REFS
    stale_refs["reproducibility_checksum"] = exp4803.stable_checksum(stale_refs)
    assert "spec_refs_mismatch" in exp4803.artifact_schema_errors(stale_refs)

    stale_seed = dict(artifact)
    stale_seed["random_seed"] = exp4793.RANDOM_SEED
    stale_seed["reproducibility_checksum"] = exp4803.stable_checksum(stale_seed)
    assert "random_seed_mismatch" in exp4803.artifact_schema_errors(stale_seed)


def test_req_arc_wmte_4803_schema_requires_search_state_count_and_delta(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-ARC-WMTE-4803: refreshed success requires search states and mtime delta."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    missing_count = dict(artifact)
    missing_count.pop("search_state_count")
    missing_count["reproducibility_checksum"] = exp4803.stable_checksum(missing_count)
    assert "missing_field:search_state_count" in exp4803.artifact_schema_errors(missing_count)

    zero_count = dict(artifact)
    zero_count["search_state_count"] = 0
    zero_count["reproducibility_checksum"] = exp4803.stable_checksum(zero_count)
    assert "success_without_search_state_count" in exp4803.artifact_schema_errors(zero_count)

    missing_delta = dict(artifact)
    missing_delta["checkpoint_mtime_delta_ns"] = None
    missing_delta["reproducibility_checksum"] = exp4803.stable_checksum(missing_delta)
    assert "refreshed_checkpoint_missing_mtime_delta" in exp4803.artifact_schema_errors(missing_delta)


def test_req_arc_wmte_4803_schema_defensive_branches(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4803: malformed terminal artifacts fail closed."""

    artifact = _success_artifact(tmp_path, monkeypatch)

    missing_principles = dict(artifact, field_principles={})
    missing_principles["reproducibility_checksum"] = exp4803.stable_checksum(missing_principles)
    assert "missing_principle:honest_verdict" in exp4803.artifact_schema_errors(
        missing_principles
    )

    bad_verdict = dict(artifact, honest_verdict="not_terminal")
    bad_verdict["reproducibility_checksum"] = exp4803.stable_checksum(bad_verdict)
    assert "honest_verdict_missing_terminal_prefix" in exp4803.artifact_schema_errors(
        bad_verdict
    )

    bad_substrate = dict(artifact, inference_substrate="cached")
    bad_substrate["reproducibility_checksum"] = exp4803.stable_checksum(bad_substrate)
    assert "inference_substrate_mismatch" in exp4803.artifact_schema_errors(bad_substrate)

    bad_provenance = dict(artifact, solve_provenance="development_proxy")
    bad_provenance["reproducibility_checksum"] = exp4803.stable_checksum(bad_provenance)
    assert "solve_provenance_mismatch" in exp4803.artifact_schema_errors(bad_provenance)

    bad_refreshed_type = dict(artifact, verifier_checkpoint_refreshed="yes")
    bad_refreshed_type["reproducibility_checksum"] = exp4803.stable_checksum(
        bad_refreshed_type
    )
    assert "verifier_checkpoint_refreshed_must_be_bool" in exp4803.artifact_schema_errors(
        bad_refreshed_type
    )

    invalid_checksum = dict(artifact, reproducibility_checksum="bad")
    assert "invalid_reproducibility_checksum" in exp4803.artifact_schema_errors(invalid_checksum)

    checksum_mismatch = dict(artifact, target_game="other")
    assert "checksum_mismatch" in exp4803.artifact_schema_errors(checksum_mismatch)

    not_refreshed = dict(artifact, verifier_checkpoint_refreshed=False)
    not_refreshed["reproducibility_checksum"] = exp4803.stable_checksum(not_refreshed)
    assert "success_without_refreshed_checkpoint" in exp4803.artifact_schema_errors(
        not_refreshed
    )

    not_reproduced = dict(artifact, offline_reproduced=False)
    not_reproduced["reproducibility_checksum"] = exp4803.stable_checksum(not_reproduced)
    assert "success_without_reproduction_gate" in exp4803.artifact_schema_errors(not_reproduced)

    no_states = dict(artifact, states_expanded=0)
    no_states["reproducibility_checksum"] = exp4803.stable_checksum(no_states)
    assert "success_without_search_states" in exp4803.artifact_schema_errors(no_states)

    missing_path = dict(artifact, checkpoint_path=None)
    missing_path["reproducibility_checksum"] = exp4803.stable_checksum(missing_path)
    assert "refreshed_checkpoint_missing_path" in exp4803.artifact_schema_errors(missing_path)

    missing_after = dict(artifact, checkpoint_mtime_after_ns=None)
    missing_after["reproducibility_checksum"] = exp4803.stable_checksum(missing_after)
    assert "refreshed_checkpoint_missing_mtime" in exp4803.artifact_schema_errors(missing_after)

    stale_after = dict(
        artifact,
        checkpoint_mtime_after_ns=artifact["checkpoint_mtime_before_ns"],
    )
    stale_after["reproducibility_checksum"] = exp4803.stable_checksum(stale_after)
    assert "refreshed_checkpoint_without_mtime_advance" in exp4803.artifact_schema_errors(
        stale_after
    )

    bad_delta = dict(artifact, checkpoint_mtime_delta_ns="bad")
    bad_delta["reproducibility_checksum"] = exp4803.stable_checksum(bad_delta)
    assert "refreshed_checkpoint_missing_mtime_delta" in exp4803.artifact_schema_errors(
        bad_delta
    )

    zero_delta = dict(artifact, checkpoint_mtime_delta_ns=0)
    zero_delta["reproducibility_checksum"] = exp4803.stable_checksum(zero_delta)
    assert "refreshed_checkpoint_nonpositive_mtime_delta" in exp4803.artifact_schema_errors(
        zero_delta
    )


def test_req_arc_wmte_4803_main_writes_terminal_artifact(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4803: main writes the 4803 artifact path from loop output."""

    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    ckpt = tmp_path / "models" / "arc_verifier_re86.json"
    ckpt.write_text("{}", encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    loop_path = tmp_path / "results" / "arc_loop_solve_re86.json"
    loop_path.write_text(json.dumps(_loop_result("re86")), encoding="utf-8")

    monkeypatch.setattr(exp4803, "REPO", tmp_path)
    monkeypatch.setattr(exp4803, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4803, "ARTIFACT", tmp_path / exp4803.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4803,
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

    assert exp4803.main(["--game", "re86", "--checkpoint-mtime-before-ns", str(before_ns)]) == 0

    written = json.loads((tmp_path / exp4803.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["experiment"] == exp4803.EXPERIMENT
    assert written["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert written["search_state_count"] == 103
    assert written["schema_errors"] == []


def test_req_arc_wmte_4803_main_writes_blocked_artifact(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4803-BLOCKED-PRECONDITION: main writes blocked artifacts."""

    monkeypatch.setattr(exp4803, "ARTIFACT", tmp_path / exp4803.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4803,
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

    assert exp4803.main(["--game", "missing"]) == 0

    written = json.loads((tmp_path / exp4803.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "blocked_target_not_banked"
    assert written["verifier_checkpoint_refreshed"] is False
    assert written["search_state_count"] == 0
