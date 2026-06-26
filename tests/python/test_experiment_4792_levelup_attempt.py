"""Tests for Exp 4792 ARC level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4792,
SCENARIO-ARC-WMTE-4792-ROTATION-TARGET,
SCENARIO-ARC-WMTE-4792-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4792-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_4792_levelup_attempt as exp4792


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - ka59: residual_existing_depth
- game: lf52
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - lf52: residual_existing_depth
- game: re86
  reproducibility: reproduced
  levels_reproduced: 2
- game: sb26
  reproducibility: reproduced
  levels_reproduced: 2
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 1
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 2
reproducible_total_levels: 65
"""


def _loop_result(game: str = "sb26", reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
        "learned_verifier_checkpoint": f"models/arc_verifier_{game}.json",
        "reproduction_gate": {
            "game": game,
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": ["seed", "tail", "C:1"],
    }


def _preconditions(game: str = "sb26") -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade": {"ok": True, "check": "arc_solver_kit.offline_arcade()"},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_offline_env": {"game": game, "ok": True},
        "induction_needed": False,
        "qwen_igpu": {"needed": False, "ok": None},
    }


def _recommendation(game: str = "sb26") -> dict[str, object]:
    return {
        "game": game,
        "recommended": "reuse_color_match_slot_sequence_verifier",
        "selected_generic_operators": [{"operator": "color_match_slot_sequence_verifier"}],
        "guidance": ["derive only the per-level delta"],
    }


def test_req_arc_wmte_4792_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4792: OpenSpec declares the 4792 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4792.SPEC_REFS:
        assert ref in spec
    assert exp4792.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4792.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4792_selects_sb26_then_bp35() -> None:
    """SCENARIO-ARC-WMTE-4792-ROTATION-TARGET: retired games rotate to sb26 then bp35."""

    registry = yaml.safe_load(_registry_text())

    selection = exp4792.select_rotation_target(
        registry,
        adaptered_games={"sb26", "bp35", "g50t", "ka59", "lf52", "re86"},
        approach_recommendation=_recommendation("sb26"),
    )

    assert selection["game"] == "sb26"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["reason"] == "preferred_public_candidate_deepen"
    assert selection["approach_recommendation"] == _recommendation("sb26")
    assert selection["rotate_if_no_bank"] == [
        {"game": "bp35", "prior_level": 2, "target_level": 3, "reason": "preferred_public_candidate_deepen"}
    ]
    assert {row["game"] for row in selection["primary_rotation"]} == {"sb26", "bp35"}
    assert selection["retired_targets"] == [
        {"game": "ka59", "reason": "recent_no_bank_residual_existing_depth"},
        {"game": "lf52", "reason": "recent_no_bank_residual_existing_depth"},
        {"game": "re86", "reason": "recent_self_play_coverage"},
    ]


def test_scenario_arc_wmte_4792_unreproduced_primary_takes_priority() -> None:
    """SCENARIO-ARC-WMTE-4792-ROTATION-TARGET: primary first-contact wins when present."""

    registry = yaml.safe_load(_registry_text())
    for row in registry["games"]:
        if row["game"] == "sb26":
            row["levels_reproduced"] = 0

    selection = exp4792.select_rotation_target(registry, adaptered_games={"sb26", "bp35"})

    assert selection["game"] == "sb26"
    assert selection["prior_level"] == 0
    assert selection["target_level"] == 1
    assert selection["reason"] == "preferred_public_first_contact"


def test_scenario_arc_wmte_4792_fallback_excludes_retired_targets() -> None:
    """SCENARIO-ARC-WMTE-4792-ROTATION-TARGET: fallback skips retired games."""

    fallback_registry = {
        "games": [
            {"game": "ka59", "reproducibility": "reproduced", "levels_reproduced": 1},
            {"game": "g50t", "reproducibility": "reproduced", "levels_reproduced": 1},
        ]
    }

    fallback = exp4792.select_rotation_target(fallback_registry, adaptered_games={"ka59", "g50t"})
    assert fallback["game"] == "g50t"
    assert fallback["prior_level"] == 1
    assert fallback["target_level"] == 2
    assert fallback["reason"] == "shallowest_adaptered_fallback"

    no_fallback = exp4792.select_rotation_target(fallback_registry, adaptered_games=set())
    assert no_fallback["game"] == "none"
    assert no_fallback["reason"] == "no_reproduced_standing_loop_target"


def test_scenario_arc_wmte_4792_same_depth_reproduction_rotates() -> None:
    """SCENARIO-ARC-WMTE-4792-REPRODUCTION-GATE: same-depth gates retire with no bank."""

    selection = {
        "game": "sb26",
        "prior_level": 2,
        "target_level": 3,
        "reason": "preferred_public_candidate_deepen",
    }

    attempt = exp4792.summarize_loop_attempt(
        selection=selection,
        loop_result=_loop_result("sb26", 2),
        loop_result_path="results/arc_loop_solve_sb26.json",
    )

    assert attempt["offline_reproduced_existing_depth"] is True
    assert attempt["offline_reproduced_new_depth"] is False
    assert attempt["new_levels_banked"] == 0
    assert attempt["residual_cause"] == "reproduced_existing_or_lower_level"
    assert "same-depth" in attempt["dead_end"]
    assert attempt["target_selection_reason"] == "preferred_public_candidate_deepen"


def test_req_arc_wmte_4792_builds_no_bank_artifact_without_fabrication() -> None:
    """REQ-ARC-WMTE-4792: no-bank artifact preserves the registry total."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4792.select_rotation_target(
        registry,
        adaptered_games={"sb26", "bp35"},
        approach_recommendation=_recommendation("sb26"),
    )
    attempts = [
        exp4792.summarize_loop_attempt(
            selection=selection,
            loop_result=_loop_result("sb26", 2),
            loop_result_path="results/arc_loop_solve_sb26.json",
        ),
        exp4792.summarize_loop_attempt(
            selection=selection["rotate_if_no_bank"][0],
            loop_result=_loop_result("bp35", 2),
            loop_result_path="results/arc_loop_solve_bp35.json",
        ),
    ]

    artifact = exp4792.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("sb26"),
    )

    assert artifact["honest_verdict"] == "complete_sb26_no_new_level_residual_existing_depth"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["approach_recommendation"] == _recommendation("sb26")
    assert artifact["registry_update"]["updated"] is False
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 65
    assert artifact["schema_errors"] == []
    assert exp4792.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4792_success_can_come_from_second_rotation_target() -> None:
    """REQ-ARC-WMTE-4792: success requires a gate above prior registry depth."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4792.select_rotation_target(
        registry,
        adaptered_games={"sb26", "bp35"},
        approach_recommendation=_recommendation("sb26"),
    )
    attempts = [
        exp4792.summarize_loop_attempt(
            selection=selection,
            loop_result=_loop_result("sb26", 2),
            loop_result_path="results/arc_loop_solve_sb26.json",
        ),
        exp4792.summarize_loop_attempt(
            selection=selection["rotate_if_no_bank"][0],
            loop_result=_loop_result("bp35", 3),
            loop_result_path="results/arc_loop_solve_bp35.json",
        ),
    ]

    artifact = exp4792.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("sb26"),
    )

    assert artifact["honest_verdict"] == "success_bp35_L3_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["target_game"] == "bp35"
    assert artifact["registry_update"]["updated"] is True
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 66
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4792_blocks_missing_target_env() -> None:
    """REQ-ARC-WMTE-4792: missing target environments produce blocked artifacts."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4792.select_rotation_target(registry, adaptered_games={"sb26", "bp35"})
    attempt = exp4792.summarize_loop_attempt(
        selection=selection,
        loop_result=_loop_result("sb26", 3),
        loop_result_path="results/arc_loop_solve_sb26.json",
    )
    preconditions = _preconditions("sb26")
    preconditions["target_offline_env"] = {"game": "sb26", "ok": False}

    artifact = exp4792.build_artifact(
        registry=registry,
        selection=selection,
        attempts=[attempt],
        preconditions_checked=preconditions,
    )

    assert artifact["honest_verdict"] == "blocked_sb26_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_update"]["updated"] is False
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4792_schema_guards_required_fields() -> None:
    """REQ-ARC-WMTE-4792: schema validation rejects overclaims and bad checksums."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4792.select_rotation_target(registry, adaptered_games={"sb26", "bp35"})
    attempt = exp4792.summarize_loop_attempt(
        selection=selection,
        loop_result=_loop_result("sb26", 2),
        loop_result_path="results/arc_loop_solve_sb26.json",
    )
    artifact = exp4792.build_artifact(
        registry=registry,
        selection=selection,
        attempts=[attempt],
        preconditions_checked=_preconditions("sb26"),
    )

    missing = dict(artifact)
    missing.pop("approach_recommendation")
    assert "missing_field:approach_recommendation" in exp4792.artifact_schema_errors(missing)

    wrong_principle = dict(artifact)
    wrong_principle["field_principles"] = dict(artifact["field_principles"])
    wrong_principle["field_principles"]["solve_provenance"] = "wrong"
    assert "missing_principle:solve_provenance" in exp4792.artifact_schema_errors(wrong_principle)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    assert "invalid_reproducibility_checksum" in exp4792.artifact_schema_errors(bad_checksum)

    stale_checksum = dict(artifact)
    stale_checksum["reproducibility_checksum"] = "0" * 64
    assert "checksum_mismatch" in exp4792.artifact_schema_errors(stale_checksum)

    bad_prefix = dict(artifact)
    bad_prefix["honest_verdict"] = "partial_sb26"
    assert "honest_verdict_missing_terminal_prefix" in exp4792.artifact_schema_errors(bad_prefix)

    bad_provenance = dict(artifact)
    bad_provenance["solve_provenance"] = "outer_loop_re"
    assert "solve_provenance_mismatch" in exp4792.artifact_schema_errors(bad_provenance)

    bad_substrate = dict(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate_mismatch" in exp4792.artifact_schema_errors(bad_substrate)

    bad_oracle = dict(artifact)
    bad_oracle["verifier_is_oracle"] = False
    assert "verifier_is_oracle_must_be_true" in exp4792.artifact_schema_errors(bad_oracle)

    fabricated_bank = dict(artifact)
    fabricated_bank["new_levels_banked"] = 1
    fabricated_bank["reproducibility_checksum"] = exp4792.stable_checksum(fabricated_bank)
    assert "bank_without_offline_reproduction" in exp4792.artifact_schema_errors(fabricated_bank)

    fabricated_repro = dict(artifact)
    fabricated_repro["offline_reproduced"] = True
    fabricated_repro["reproducibility_checksum"] = exp4792.stable_checksum(fabricated_repro)
    assert "offline_reproduced_true_without_new_bank" in exp4792.artifact_schema_errors(fabricated_repro)

    fabricated_retire = dict(artifact)
    fabricated_retire["retire_if_same_verdict"] = False
    fabricated_retire["reproducibility_checksum"] = exp4792.stable_checksum(fabricated_retire)
    assert "retire_if_same_verdict_must_be_true" in exp4792.artifact_schema_errors(fabricated_retire)


def test_req_arc_wmte_4792_edge_paths_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4792: malformed registries and missing loop outputs do not fabricate progress."""

    registry = {"games": [{"game": "sb26", "levels_reproduced": "bad"}], "reproducible_total_levels": object()}

    assert exp4792.registry_levels(registry) == {"sb26": 0}
    assert exp4792.registry_total_levels(registry) == 0

    no_target = exp4792.select_rotation_target({"games": []}, adaptered_games=set())
    assert no_target["game"] == "none"
    assert no_target["reason"] == "no_reproduced_standing_loop_target"
    assert exp4792.collect_attempts(no_target, results_dir=tmp_path) == []

    missing = exp4792.collect_attempt(
        {"game": "sb26", "prior_level": 2, "target_level": 3, "reason": "unit"},
        results_dir=tmp_path,
    )
    assert missing["residual_cause"] == "missing_loop_result"
    assert missing["new_levels_banked"] == 0
    missing_artifact = exp4792.build_artifact(
        registry=yaml.safe_load(_registry_text()),
        selection={"game": "sb26", "prior_level": 2, "target_level": 3, "reason": "unit"},
        attempts=[missing],
        preconditions_checked=_preconditions("sb26"),
    )
    assert missing_artifact["honest_verdict"] == "complete_sb26_no_new_level_residual_missing_loop_result"

    no_attempt_artifact = exp4792.build_artifact(
        registry={},
        selection=no_target,
        attempts=[],
        preconditions_checked=_preconditions("none"),
    )
    assert no_attempt_artifact["honest_verdict"] == "complete_none_no_new_level_residual_no_attempts"

    (tmp_path / "arc_loop_solve_sb26.json").write_text(json.dumps(["bad"]), encoding="utf-8")
    malformed = exp4792.collect_attempt(
        {"game": "sb26", "prior_level": 2, "target_level": 3, "reason": "unit"},
        results_dir=tmp_path,
    )
    assert malformed["residual_cause"] == "offline_reproduction_failed"

    success_dir = tmp_path / "success"
    success_dir.mkdir()
    (success_dir / "arc_loop_solve_sb26.json").write_text(
        json.dumps(_loop_result("sb26", 3)),
        encoding="utf-8",
    )
    early_stop = exp4792.collect_attempts(
        {
            "game": "sb26",
            "prior_level": 2,
            "target_level": 3,
            "reason": "unit",
            "rotate_if_no_bank": [
                {"game": "bp35", "prior_level": 2, "target_level": 3, "reason": "unit"}
            ],
        },
        results_dir=success_dir,
    )
    assert [attempt["game"] for attempt in early_stop] == ["sb26"]
    assert early_stop[0]["new_levels_banked"] == 1

    registry_file = tmp_path / "registry.yaml"
    registry_file.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4792.load_registry(registry_file) == {}


def test_scenario_arc_wmte_4792_stable_artifact_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4792-STABLE-ARTIFACT: writer emits deterministic JSON."""

    payload = {"z": 1, "a": {"b": 2}}
    out = tmp_path / exp4792.RESULT_RELATIVE_PATH

    exp4792.write_artifact(payload, path=out)

    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert out.read_text(encoding="utf-8").startswith("{\n  \"a\"")


def test_req_arc_wmte_4792_main_writes_terminal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4792: main writes a stable terminal artifact from loop results."""

    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / exp4792.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "results" / "arc_loop_solve_sb26.json").write_text(
        json.dumps(_loop_result("sb26", 2)),
        encoding="utf-8",
    )
    (tmp_path / "results" / "arc_loop_solve_bp35.json").write_text(
        json.dumps(_loop_result("bp35", 2)),
        encoding="utf-8",
    )

    monkeypatch.setattr(exp4792, "REPO", tmp_path)
    monkeypatch.setattr(exp4792, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4792, "REGISTRY", tmp_path / exp4792.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(exp4792, "ARTIFACT", tmp_path / exp4792.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp4792, "_adaptered_games", lambda: {"sb26", "bp35"})
    monkeypatch.setattr(exp4792, "_recommend_approach", lambda game: _recommendation(game))
    monkeypatch.setattr(exp4792, "check_preconditions", lambda selection: _preconditions(selection["game"]))

    assert exp4792.main([]) == 0

    written = json.loads((tmp_path / exp4792.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "complete_sb26_no_new_level_residual_existing_depth"
    assert written["target_game"] == "sb26"
    assert written["approach_recommendation"] == _recommendation("sb26")
    assert written["schema_errors"] == []
