"""Tests for Exp 4763 ARC self-play verifier checkpoint refresh.

Spec refs: REQ-ARC-WMTE-4763,
SCENARIO-ARC-WMTE-4763-CHECKPOINT-REFRESHED,
SCENARIO-ARC-WMTE-4763-RESIDUAL-NO-FABRICATION,
SCENARIO-ARC-WMTE-4763-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4763_self_play_verifier_checkpoint as exp4763


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: re86
  reproducibility: reproduced
  levels_reproduced: 2
- game: sb26
  reproducibility: reproduced
  levels_reproduced: 0
- game: bp35
  reproducibility: reproduced
  levels_reproduced: bad
reproducible_total_levels: 65
"""


def _loop_result(game: str, reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "states_expanded": 42,
        "learned_verifier_checkpoint": f"models/arc_verifier_{game}.json",
        "solve_provenance": "development_proxy",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": reproduced,
        },
    }


def test_req_arc_wmte_4763_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4763: OpenSpec declares the 4763 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4763.SPEC_REFS:
        assert ref in spec
    assert exp4763.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4763.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4763_selects_only_banked_registry_target(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4763: target selection requires a banked registry level."""

    registry = yaml.safe_load(_registry_text())

    assert exp4763.registry_levels(registry) == {"re86": 2, "sb26": 0, "bp35": 0}
    assert exp4763.select_banked_target(registry, preferred_game="re86") == {
        "game": "re86",
        "prior_level": 2,
        "banked": True,
        "reason": "preferred_banked_target",
    }
    assert exp4763.select_banked_target(registry, preferred_game="sb26") == {
        "game": "sb26",
        "prior_level": 0,
        "banked": False,
        "reason": "preferred_target_not_banked",
    }
    assert exp4763.select_banked_target(registry, candidate_games=("re86",)) == {
        "game": "re86",
        "prior_level": 2,
        "banked": True,
        "reason": "first_banked_rotation_target",
    }
    assert exp4763.select_banked_target({"games": []}, candidate_games=("sb26",)) == {
        "game": None,
        "prior_level": 0,
        "banked": False,
        "reason": "no_banked_target",
    }

    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4763.load_registry(registry_file) == {}


def test_scenario_arc_wmte_4763_checkpoint_refreshed_success(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4763-CHECKPOINT-REFRESHED: mtime advance plus gate green succeeds."""

    ckpt = tmp_path / "models" / "arc_verifier_re86.json"
    ckpt.parent.mkdir()
    ckpt.write_text('{"ok": true}\n', encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    monkeypatch.setattr(exp4763, "REPO", tmp_path)

    summary = exp4763.summarize_loop_result(
        game="re86",
        loop_result=_loop_result("re86"),
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=before_ns,
    )
    artifact = exp4763.build_artifact(
        target_selection={"game": "re86", "prior_level": 2, "banked": True, "reason": "preferred_banked_target"},
        loop_summary=summary,
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": True, "game": "re86"},
        },
    )

    assert artifact["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert artifact["verifier_checkpoint_refreshed"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["states_expanded"] == 42
    assert artifact["self_play_residual"] == "checkpoint_refreshed_gate_passed"
    assert artifact["checkpoint_mtime_after_ns"] > artifact["checkpoint_mtime_before_ns"]
    assert artifact["schema_errors"] == []
    assert exp4763.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4763_residual_does_not_fabricate_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4763-RESIDUAL-NO-FABRICATION: failed gates stay complete residuals."""

    monkeypatch.setattr(exp4763, "REPO", tmp_path)
    failed = _loop_result("re86", reached_level=0, reproduced=False)
    failed["learned_verifier_checkpoint"] = None

    summary = exp4763.summarize_loop_result(
        game="re86",
        loop_result=failed,
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=123,
    )
    artifact = exp4763.build_artifact(
        target_selection={"game": "re86", "prior_level": 2, "banked": True, "reason": "preferred_banked_target"},
        loop_summary=summary,
        preconditions_checked={"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}},
    )

    assert artifact["honest_verdict"] == "complete_re86_self_play_residual_reproduction_gate_failed"
    assert artifact["verifier_checkpoint_refreshed"] is False
    assert artifact["checkpoint_path"] is None
    assert artifact["checkpoint_mtime_after_ns"] is None
    assert artifact["schema_errors"] == []


def test_scenario_arc_wmte_4763_residual_variants(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4763-RESIDUAL-NO-FABRICATION: every non-green cause is explicit."""

    monkeypatch.setattr(exp4763, "REPO", tmp_path)
    missing = exp4763.summarize_loop_result(
        game="re86",
        loop_result=None,
        loop_result_path="results/missing.json",
        checkpoint_mtime_before_ns="not-an-int",  # type: ignore[arg-type]
    )
    assert missing["self_play_residual"] == "loop_result_missing"
    assert missing["checkpoint_mtime_before_ns"] is None
    assert exp4763._read_loop_result(tmp_path / "missing.json") is None

    no_checkpoint = _loop_result("re86")
    no_checkpoint["learned_verifier_checkpoint"] = None
    assert (
        exp4763.summarize_loop_result(
            game="re86",
            loop_result=no_checkpoint,
            loop_result_path="results/arc_loop_solve_re86.json",
            checkpoint_mtime_before_ns=None,
        )["self_play_residual"]
        == "checkpoint_not_reported"
    )

    stale_ckpt = tmp_path / "models" / "arc_verifier_re86.json"
    stale_ckpt.parent.mkdir(exist_ok=True)
    stale_ckpt.write_text("{}", encoding="utf-8")
    stale_loop = _loop_result("re86")
    stale = exp4763.summarize_loop_result(
        game="re86",
        loop_result=stale_loop,
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=stale_ckpt.stat().st_mtime_ns,
    )
    assert stale["self_play_residual"] == "checkpoint_mtime_not_advanced"

    outside = tmp_path.parent / "outside_arc_verifier_re86.json"
    outside.write_text("{}", encoding="utf-8")
    outside_loop = _loop_result("re86")
    outside_loop["learned_verifier_checkpoint"] = str(outside)
    outside_summary = exp4763.summarize_loop_result(
        game="re86",
        loop_result=outside_loop,
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=outside.stat().st_mtime_ns - 1,
    )
    assert outside_summary["checkpoint_path"] == str(outside)


def test_scenario_arc_wmte_4763_blocked_precondition_artifact() -> None:
    """SCENARIO-ARC-WMTE-4763-BLOCKED-PRECONDITION: blocked runs remain auditable."""

    artifact = exp4763.build_blocked_artifact(
        reason="target_not_banked",
        preconditions_checked={
            "offline_arcade": {"ok": True},
            "registry_loadable": {"ok": True},
            "target_banked": {"ok": False, "game": "sb26", "prior_level": 0},
        },
        target_game="sb26",
    )

    assert artifact["honest_verdict"] == "blocked_target_not_banked"
    assert artifact["target_game"] == "sb26"
    assert artifact["verifier_checkpoint_refreshed"] is False
    assert artifact["self_play_residual"] == "target_not_banked"
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4763_schema_guards_success_and_checksum(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4763: schema rejects fabricated success and stale checksums."""

    ckpt = tmp_path / "models" / "arc_verifier_re86.json"
    ckpt.parent.mkdir()
    ckpt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(exp4763, "REPO", tmp_path)
    summary = exp4763.summarize_loop_result(
        game="re86",
        loop_result=_loop_result("re86"),
        loop_result_path="results/arc_loop_solve_re86.json",
        checkpoint_mtime_before_ns=ckpt.stat().st_mtime_ns - 1,
    )
    artifact = exp4763.build_artifact(
        target_selection={"game": "re86", "prior_level": 2, "banked": True, "reason": "preferred_banked_target"},
        loop_summary=summary,
        preconditions_checked={"offline_arcade": {"ok": True}, "registry_loadable": {"ok": True}},
    )

    missing = dict(artifact)
    missing.pop("honest_verdict")
    assert "missing_field:honest_verdict" in exp4763.artifact_schema_errors(missing)

    wrong_principle = dict(artifact)
    wrong_principle["field_principles"] = dict(artifact["field_principles"])
    wrong_principle["field_principles"]["honest_verdict"] = "wrong"
    assert "missing_principle:honest_verdict" in exp4763.artifact_schema_errors(wrong_principle)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    assert "invalid_reproducibility_checksum" in exp4763.artifact_schema_errors(bad_checksum)

    stale_checksum = dict(artifact)
    stale_checksum["reproducibility_checksum"] = "0" * 64
    assert "checksum_mismatch" in exp4763.artifact_schema_errors(stale_checksum)

    fabricated_success = dict(artifact)
    fabricated_success["verifier_checkpoint_refreshed"] = False
    fabricated_success["reproducibility_checksum"] = exp4763.stable_checksum(fabricated_success)
    assert "success_without_refreshed_checkpoint" in exp4763.artifact_schema_errors(fabricated_success)

    no_gate = dict(artifact)
    no_gate["offline_reproduced"] = False
    no_gate["reproducibility_checksum"] = exp4763.stable_checksum(no_gate)
    assert "success_without_reproduction_gate" in exp4763.artifact_schema_errors(no_gate)

    no_states = dict(artifact)
    no_states["states_expanded"] = 0
    no_states["reproducibility_checksum"] = exp4763.stable_checksum(no_states)
    assert "success_without_search_states" in exp4763.artifact_schema_errors(no_states)

    no_path = dict(artifact)
    no_path["checkpoint_path"] = None
    no_path["reproducibility_checksum"] = exp4763.stable_checksum(no_path)
    assert "refreshed_checkpoint_missing_path" in exp4763.artifact_schema_errors(no_path)

    no_after_mtime = dict(artifact)
    no_after_mtime["checkpoint_mtime_after_ns"] = None
    no_after_mtime["reproducibility_checksum"] = exp4763.stable_checksum(no_after_mtime)
    assert "refreshed_checkpoint_missing_mtime" in exp4763.artifact_schema_errors(no_after_mtime)

    fabricated_mtime = dict(artifact)
    fabricated_mtime["checkpoint_mtime_after_ns"] = fabricated_mtime["checkpoint_mtime_before_ns"]
    fabricated_mtime["reproducibility_checksum"] = exp4763.stable_checksum(fabricated_mtime)
    assert "refreshed_checkpoint_without_mtime_advance" in exp4763.artifact_schema_errors(fabricated_mtime)

    bad_refreshed_type = dict(artifact)
    bad_refreshed_type["verifier_checkpoint_refreshed"] = "true"
    bad_refreshed_type["reproducibility_checksum"] = exp4763.stable_checksum(bad_refreshed_type)
    assert "verifier_checkpoint_refreshed_must_be_bool" in exp4763.artifact_schema_errors(bad_refreshed_type)

    bad_prefix = dict(artifact)
    bad_prefix["honest_verdict"] = "partial_re86"
    assert "honest_verdict_missing_terminal_prefix" in exp4763.artifact_schema_errors(bad_prefix)

    bad_substrate = dict(artifact)
    bad_substrate["inference_substrate"] = "verifier_ensemble_against_cached_candidates"
    assert "inference_substrate_mismatch" in exp4763.artifact_schema_errors(bad_substrate)

    bad_provenance = dict(artifact)
    bad_provenance["solve_provenance"] = "outer_loop_re"
    assert "solve_provenance_mismatch" in exp4763.artifact_schema_errors(bad_provenance)


def test_scenario_arc_wmte_4763_precondition_failures_and_blocked_main(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4763-BLOCKED-PRECONDITION: main writes blocked artifacts."""

    assert (
        exp4763._precondition_failure({"offline_arcade": {"ok": False}})
        == "offline_arcade_missing"
    )
    assert (
        exp4763._precondition_failure(
            {"offline_arcade": {"ok": True}, "registry_loadable": {"ok": False}}
        )
        == "registry_missing"
    )
    assert (
        exp4763._precondition_failure(
            {
                "offline_arcade": {"ok": True},
                "registry_loadable": {"ok": True},
                "target_banked": {"ok": False},
            }
        )
        == "target_not_banked"
    )

    monkeypatch.setattr(exp4763, "ARTIFACT", tmp_path / exp4763.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4763,
        "check_preconditions",
        lambda game=None: {"offline_arcade": {"ok": False}, "ok": False},
    )

    assert exp4763.main(["--game", "sb26"]) == 0

    written = json.loads((tmp_path / exp4763.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "blocked_offline_arcade_missing"
    assert written["target_game"] == "sb26"
    assert written["schema_errors"] == []


def test_req_arc_wmte_4763_main_writes_terminal_artifact(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4763: main writes a stable terminal artifact from loop output."""

    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    ckpt = tmp_path / "models" / "arc_verifier_re86.json"
    ckpt.write_text("{}", encoding="utf-8")
    before_ns = ckpt.stat().st_mtime_ns - 1
    loop_path = tmp_path / "results" / "arc_loop_solve_re86.json"
    loop_path.write_text(json.dumps(_loop_result("re86")), encoding="utf-8")

    monkeypatch.setattr(exp4763, "REPO", tmp_path)
    monkeypatch.setattr(exp4763, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4763, "ARTIFACT", tmp_path / exp4763.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(
        exp4763,
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

    assert exp4763.main(["--game", "re86", "--checkpoint-mtime-before-ns", str(before_ns)]) == 0

    written = json.loads((tmp_path / exp4763.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "success_re86_L2_checkpoint_refreshed"
    assert written["schema_errors"] == []


def test_scenario_arc_wmte_4763_stable_artifact_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4763-CHECKPOINT-REFRESHED: writer emits sorted JSON."""

    out = tmp_path / exp4763.RESULT_RELATIVE_PATH
    payload = {"z": 1, "a": {"b": 2}}

    exp4763.write_artifact(payload, path=out)

    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert out.read_text(encoding="utf-8").startswith("{\n  \"a\"")
