"""Tests for Exp 4832 ARC rotated level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4832,
SCENARIO-ARC-WMTE-4832-ROTATION-TARGET,
SCENARIO-ARC-WMTE-4832-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4832-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4832_levelup_attempt as exp4832


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
- game: sb26
  reproducibility: reproduced
  levels_reproduced: 2
- game: lf52
  reproducibility: reproduced
  levels_reproduced: 2
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
reproducible_total_levels: 65
"""


def _loop_result(game: str, reached_level: int, reproduced: bool = True) -> dict[str, object]:
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
        "solution_labels": ["seed", "tail"],
    }


def _preconditions(game: str = "ka59") -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade": {"ok": True, "check": "arc_solver_kit.offline_arcade()"},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_offline_env": {"game": game, "ok": True},
        "induction_needed": False,
        "qwen_igpu": {"needed": False, "ok": None},
    }


def _recommendation(game: str = "ka59") -> dict[str, object]:
    return {
        "game": game,
        "recommended": "reuse_standing_loop_delta",
        "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        "guidance": ["derive only the per-game delta"],
    }


def test_req_arc_wmte_4832_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4832: OpenSpec declares the 4832 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-4832") : spec.index("### REQ-ARC-WMTE-4763")]

    for ref in exp4832.SPEC_REFS:
        assert ref in section
    assert exp4832.RESULT_RELATIVE_PATH in section
    for field, principle in exp4832.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle in section


def test_scenario_arc_wmte_4832_selects_shallowest_after_public_contacts() -> None:
    """SCENARIO-ARC-WMTE-4832-ROTATION-TARGET: bp35/sb26/lf52 done means deepen ka59."""

    registry = yaml.safe_load(_registry_text())

    selection = exp4832.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "lf52", "ka59", "cd82"},
        approach_recommendation=_recommendation("ka59"),
    )

    assert selection["game"] == "ka59"
    assert selection["prior_level"] == 1
    assert selection["target_level"] == 2
    assert selection["reason"] == "shallowest_already_solved_deepen"
    assert selection["approach_recommendation"] == _recommendation("ka59")
    assert selection["public_rotation"] == [
        {"game": "bp35", "known": True, "prior_level": 2, "status": "already_reproduced"},
        {"game": "sb26", "known": True, "prior_level": 2, "status": "already_reproduced"},
        {"game": "lf52", "known": True, "prior_level": 2, "status": "already_reproduced"},
    ]
    assert selection["rotate_if_no_bank"][0] == {
        "game": "bp35",
        "prior_level": 2,
        "target_level": 3,
        "reason": "shallowest_already_solved_deepen",
    }


def test_scenario_arc_wmte_4832_public_first_contact_takes_priority() -> None:
    """SCENARIO-ARC-WMTE-4832-ROTATION-TARGET: unreproduced lf52 is selected for L1."""

    registry = yaml.safe_load(_registry_text())
    for row in registry["games"]:
        if row["game"] == "lf52":
            row["levels_reproduced"] = 0

    selection = exp4832.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "lf52", "ka59"},
    )

    assert selection["game"] == "lf52"
    assert selection["prior_level"] == 0
    assert selection["target_level"] == 1
    assert selection["reason"] == "preferred_public_first_contact"


def test_scenario_arc_wmte_4832_same_depth_attempt_does_not_bank() -> None:
    """SCENARIO-ARC-WMTE-4832-REPRODUCTION-GATE: same-depth gates retire with no bank."""

    attempt = exp4832.summarize_loop_attempt(
        selection={"game": "ka59", "prior_level": 1, "target_level": 2, "reason": "unit"},
        loop_result=_loop_result("ka59", 1),
        loop_result_path="results/arc_loop_solve_ka59.json",
    )

    assert attempt["offline_reproduced_existing_depth"] is True
    assert attempt["offline_reproduced_new_depth"] is False
    assert attempt["new_levels_banked"] == 0
    assert attempt["residual_cause"] == "reproduced_existing_or_lower_level"
    assert "same-depth" in attempt["dead_end"]


def test_req_arc_wmte_4832_builds_no_bank_artifact_without_fabrication() -> None:
    """REQ-ARC-WMTE-4832: no-bank artifact preserves the registry total."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4832.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "lf52", "ka59", "cd82"},
        approach_recommendation=_recommendation("ka59"),
    )
    attempts = [
        exp4832.summarize_loop_attempt(
            selection=selection,
            loop_result=_loop_result("ka59", 1),
            loop_result_path="results/arc_loop_solve_ka59.json",
        ),
        exp4832.summarize_loop_attempt(
            selection=selection["rotate_if_no_bank"][0],
            loop_result=_loop_result("bp35", 2),
            loop_result_path="results/arc_loop_solve_bp35.json",
        ),
    ]

    artifact = exp4832.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("ka59"),
    )

    assert artifact["honest_verdict"] == "complete_ka59_no_new_level_residual_existing_depth"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 65
    assert artifact["field_principles"]["honest_verdict"] == (
        "terminal prefix; banked is success_, no-bank is "
        "complete_<game>_no_new_level_residual_<cause>."
    )
    assert artifact["schema_errors"] == []
    assert exp4832.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4832_success_requires_new_reproduced_depth() -> None:
    """REQ-ARC-WMTE-4832: success requires a gate above prior registry depth."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4832.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "lf52", "ka59", "cd82"},
        approach_recommendation=_recommendation("ka59"),
    )
    attempts = [
        exp4832.summarize_loop_attempt(
            selection=selection,
            loop_result=_loop_result("ka59", 2),
            loop_result_path="results/arc_loop_solve_ka59.json",
        )
    ]

    artifact = exp4832.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("ka59"),
    )

    assert artifact["honest_verdict"] == "success_ka59_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 2
    assert artifact["new_levels_banked"] == 1
    assert artifact["target_game"] == "ka59"
    assert artifact["registry_update"]["updated"] is True
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 66
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4832_blocks_missing_target_env() -> None:
    """REQ-ARC-WMTE-4832: missing target environments produce blocked artifacts."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4832.select_rotation_target(registry, adaptered_games={"ka59"})
    preconditions = _preconditions("ka59")
    preconditions["target_offline_env"] = {"game": "ka59", "ok": False}

    artifact = exp4832.build_artifact(
        registry=registry,
        selection=selection,
        attempts=[],
        preconditions_checked=preconditions,
    )

    assert artifact["honest_verdict"] == "blocked_ka59_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_update"]["updated"] is False
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4832_schema_guards_required_fields() -> None:
    """REQ-ARC-WMTE-4832: schema validation rejects overclaims."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4832.select_rotation_target(registry, adaptered_games={"ka59"})
    artifact = exp4832.build_artifact(
        registry=registry,
        selection=selection,
        attempts=[
            exp4832.summarize_loop_attempt(
                selection=selection,
                loop_result=_loop_result("ka59", 1),
                loop_result_path="results/arc_loop_solve_ka59.json",
            )
        ],
        preconditions_checked=_preconditions("ka59"),
    )

    missing = dict(artifact)
    missing.pop("attempted_games")
    assert "missing_field:attempted_games" in exp4832.artifact_schema_errors(missing)

    bad_prefix = dict(artifact)
    bad_prefix["honest_verdict"] = "partial_ka59"
    assert "honest_verdict_missing_terminal_prefix" in exp4832.artifact_schema_errors(bad_prefix)

    fabricated_bank = dict(artifact)
    fabricated_bank["new_levels_banked"] = 1
    fabricated_bank["reproducibility_checksum"] = exp4832.stable_checksum(fabricated_bank)
    assert "bank_without_offline_reproduction" in exp4832.artifact_schema_errors(fabricated_bank)

    wrong_principle = dict(artifact)
    wrong_principle["field_principles"] = dict(artifact["field_principles"])
    wrong_principle["field_principles"]["solve_provenance"] = "wrong"
    assert "missing_principle:solve_provenance" in exp4832.artifact_schema_errors(wrong_principle)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    assert "invalid_reproducibility_checksum" in exp4832.artifact_schema_errors(bad_checksum)

    bad_provenance = dict(artifact)
    bad_provenance["solve_provenance"] = "outer_loop_re"
    assert "solve_provenance_mismatch" in exp4832.artifact_schema_errors(bad_provenance)

    bad_substrate = dict(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate_mismatch" in exp4832.artifact_schema_errors(bad_substrate)

    bad_oracle = dict(artifact)
    bad_oracle["verifier_is_oracle"] = False
    assert "verifier_is_oracle_must_be_true" in exp4832.artifact_schema_errors(bad_oracle)

    fabricated_repro = dict(artifact)
    fabricated_repro["offline_reproduced"] = True
    fabricated_repro["reproducibility_checksum"] = exp4832.stable_checksum(fabricated_repro)
    assert "offline_reproduced_true_without_new_bank" in exp4832.artifact_schema_errors(fabricated_repro)

    fabricated_retire = dict(artifact)
    fabricated_retire["retire_if_same_verdict"] = False
    fabricated_retire["reproducibility_checksum"] = exp4832.stable_checksum(fabricated_retire)
    assert "retire_if_same_verdict_must_be_true" in exp4832.artifact_schema_errors(fabricated_retire)


def test_req_arc_wmte_4832_collect_and_write_edges(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4832: malformed inputs do not fabricate progress."""

    no_target = exp4832.select_rotation_target({"games": []}, adaptered_games=set())
    assert no_target["game"] == "none"
    assert no_target["reason"] == "no_reproduced_standing_loop_target"
    assert exp4832.collect_attempts(no_target, results_dir=tmp_path) == []

    missing = exp4832.collect_attempt(
        {"game": "ka59", "prior_level": 1, "target_level": 2, "reason": "unit"},
        results_dir=tmp_path,
    )
    assert missing["residual_cause"] == "missing_loop_result"

    (tmp_path / "arc_loop_solve_ka59.json").write_text(json.dumps(["bad"]), encoding="utf-8")
    malformed = exp4832.collect_attempt(
        {"game": "ka59", "prior_level": 1, "target_level": 2, "reason": "unit"},
        results_dir=tmp_path,
    )
    assert malformed["residual_cause"] == "offline_reproduction_failed"

    missing_artifact = exp4832.build_artifact(
        registry=yaml.safe_load(_registry_text()),
        selection={"game": "ka59", "prior_level": 1, "target_level": 2, "reason": "unit"},
        attempts=[missing],
        preconditions_checked=_preconditions("ka59"),
    )
    assert missing_artifact["honest_verdict"] == "complete_ka59_no_new_level_residual_missing_loop_result"

    no_attempt_artifact = exp4832.build_artifact(
        registry={},
        selection=no_target,
        attempts=[],
        preconditions_checked=_preconditions("none"),
    )
    assert no_attempt_artifact["honest_verdict"] == "complete_none_no_new_level_residual_no_attempts"

    success_dir = tmp_path / "success"
    success_dir.mkdir()
    (success_dir / "arc_loop_solve_ka59.json").write_text(json.dumps(_loop_result("ka59", 2)), encoding="utf-8")
    early_stop = exp4832.collect_attempts(
        {
            "game": "ka59",
            "prior_level": 1,
            "target_level": 2,
            "reason": "unit",
            "rotate_if_no_bank": [{"game": "bp35", "prior_level": 2, "target_level": 3, "reason": "unit"}],
        },
        results_dir=success_dir,
    )
    assert [attempt["game"] for attempt in early_stop] == ["ka59"]

    payload = {"z": 1, "a": {"b": 2}}
    out = tmp_path / exp4832.RESULT_RELATIVE_PATH
    exp4832.write_artifact(payload, path=out)
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert exp4832.load_registry(tmp_path / "missing.yaml") == {}


def test_req_arc_wmte_4832_main_writes_terminal_artifact(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4832-STABLE-ARTIFACT: main writes a stable terminal artifact."""

    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / exp4832.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "results" / "arc_loop_solve_ka59.json").write_text(json.dumps(_loop_result("ka59", 1)), encoding="utf-8")
    (tmp_path / "results" / "arc_loop_solve_bp35.json").write_text(json.dumps(_loop_result("bp35", 2)), encoding="utf-8")

    monkeypatch.setattr(exp4832, "REPO", tmp_path)
    monkeypatch.setattr(exp4832, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4832, "REGISTRY", tmp_path / exp4832.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(exp4832, "ARTIFACT", tmp_path / exp4832.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp4832, "_adaptered_games", lambda: {"bp35", "sb26", "lf52", "ka59"})
    monkeypatch.setattr(exp4832, "_recommend_approach", lambda game: _recommendation(game))
    monkeypatch.setattr(exp4832, "check_preconditions", lambda selection: _preconditions(selection["game"]))

    assert exp4832.main([]) == 0

    written = json.loads((tmp_path / exp4832.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "complete_ka59_no_new_level_residual_existing_depth"
    assert written["target_game"] == "ka59"
    assert written["approach_recommendation"] == _recommendation("ka59")
    assert written["attempted_games"][0]["game"] == "ka59"
    assert written["attempted_games"][1]["game"] == "bp35"
    assert written["schema_errors"] == []
