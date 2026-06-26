"""Tests for Exp 4802 ARC rotated level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4802,
SCENARIO-ARC-WMTE-4802-ROTATION-TARGET,
SCENARIO-ARC-WMTE-4802-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4802-ADAPTER-FREE-SEEDING,
SCENARIO-ARC-WMTE-4802-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4802_levelup_attempt as exp4802


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
ARC_LOOP_PATH = REPO / "scripts" / "arc_loop_solve.py"


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: lf52
  reproducibility: reproduced
  levels_reproduced: 2
- game: re86
  reproducibility: reproduced
  levels_reproduced: 2
- game: sc25
  reproducibility: reproduced
  levels_reproduced: 5
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
- game: sb26
  reproducibility: reproduced
  levels_reproduced: 2
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 1
- game: r11l
  reproducibility: reproduced
  levels_reproduced: 1
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
- game: tr87
  reproducibility: reproduced
  levels_reproduced: 6
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


def _preconditions(game: str = "bp35") -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade": {"ok": True, "check": "arc_solver_kit.offline_arcade()"},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_offline_env": {"game": game, "ok": True},
        "induction_needed": False,
        "qwen_igpu": {"needed": False, "ok": None},
    }


def _recommendation(game: str = "bp35") -> dict[str, object]:
    return {
        "game": game,
        "recommended": "reuse_standing_loop_delta",
        "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        "guidance": ["derive only the per-game delta"],
    }


def test_req_arc_wmte_4802_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4802: OpenSpec declares the 4802 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4802.SPEC_REFS:
        assert ref in spec
    assert exp4802.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4802.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4802_selects_bp35_then_sb26() -> None:
    """SCENARIO-ARC-WMTE-4802-ROTATION-TARGET: bp35 rotates before sb26."""

    registry = yaml.safe_load(_registry_text())

    selection = exp4802.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "cd82", "tr87"},
        approach_recommendation=_recommendation("bp35"),
    )

    assert selection["game"] == "bp35"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["reason"] == "preferred_public_candidate_deepen"
    assert selection["approach_recommendation"] == _recommendation("bp35")
    assert selection["rotate_if_no_bank"][:2] == [
        {"game": "sb26", "prior_level": 2, "target_level": 3, "reason": "preferred_public_candidate_deepen"},
        {"game": "g50t", "prior_level": 1, "target_level": 2, "reason": "shallowest_literal_fallback"},
    ]
    assert {"game": "sc25", "reason": "recently_covered_existing_depth"} in selection["retired_targets"]
    assert selection["shallowest_adaptered_fallbacks"][0]["game"] == "cd82"


def test_scenario_arc_wmte_4802_first_contact_primary_takes_priority() -> None:
    """SCENARIO-ARC-WMTE-4802-ROTATION-TARGET: unreproduced primary gets L1 target."""

    registry = yaml.safe_load(_registry_text())
    for row in registry["games"]:
        if row["game"] == "bp35":
            row["levels_reproduced"] = 0

    selection = exp4802.select_rotation_target(registry, adaptered_games={"bp35", "sb26"})

    assert selection["game"] == "bp35"
    assert selection["prior_level"] == 0
    assert selection["target_level"] == 1
    assert selection["reason"] == "preferred_public_first_contact"


def test_scenario_arc_wmte_4802_timed_fallback_is_no_gate_dead_end() -> None:
    """SCENARIO-ARC-WMTE-4802-REPRODUCTION-GATE: timed fallbacks do not count."""

    timed = exp4802.summarize_timed_no_gate(
        game="g50t",
        prior_level=1,
        target_level=2,
        elapsed_s=120.0,
        loop_result_path="results/arc_loop_solve_g50t.json",
    )

    assert timed["residual_cause"] == "time_budget_no_terminal_gate"
    assert timed["new_levels_banked"] == 0
    assert timed["offline_reproduced_new_depth"] is False
    assert "timed no-gate residual" in timed["dead_end"]


def test_req_arc_wmte_4802_builds_no_bank_artifact_without_fabrication() -> None:
    """REQ-ARC-WMTE-4802: same-depth attempts preserve the registry total."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4802.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "cd82", "tr87"},
        approach_recommendation=_recommendation("bp35"),
    )
    attempts = [
        exp4802.summarize_loop_attempt(
            selection=selection,
            loop_result=_loop_result("bp35", 2),
            loop_result_path="results/arc_loop_solve_bp35.json",
        ),
        exp4802.summarize_loop_attempt(
            selection=selection["rotate_if_no_bank"][0],
            loop_result=_loop_result("sb26", 2),
            loop_result_path="results/arc_loop_solve_sb26.json",
        ),
        exp4802.summarize_timed_no_gate(
            game="g50t",
            prior_level=1,
            target_level=2,
            elapsed_s=120.0,
            loop_result_path="results/arc_loop_solve_g50t.json",
        ),
    ]

    artifact = exp4802.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("bp35"),
    )

    assert artifact["honest_verdict"] == "complete_bp35_no_new_level_residual_existing_depth"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["registry_update"]["updated"] is False
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 65
    assert artifact["schema_errors"] == []
    assert exp4802.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4802_success_can_come_from_later_rotation() -> None:
    """REQ-ARC-WMTE-4802: success requires a gate above prior registry depth."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4802.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "cd82", "tr87"},
        approach_recommendation=_recommendation("bp35"),
    )
    tr87_selection = {"game": "tr87", "prior_level": 6, "target_level": 7, "reason": "deeper_bank_probe"}
    attempts = [
        exp4802.summarize_loop_attempt(
            selection=selection,
            loop_result=_loop_result("bp35", 2),
            loop_result_path="results/arc_loop_solve_bp35.json",
        ),
        exp4802.summarize_loop_attempt(
            selection=tr87_selection,
            loop_result=_loop_result("tr87", 7),
            loop_result_path="results/arc_loop_solve_tr87.json",
        ),
    ]

    artifact = exp4802.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("bp35"),
    )

    assert artifact["honest_verdict"] == "success_tr87_L7_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 7
    assert artifact["new_levels_banked"] == 1
    assert artifact["target_game"] == "tr87"
    assert artifact["registry_update"]["updated"] is True
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 66
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4802_blocks_missing_target_env() -> None:
    """REQ-ARC-WMTE-4802: missing target environments produce blocked artifacts."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4802.select_rotation_target(registry, adaptered_games={"bp35", "sb26"})
    attempt = exp4802.summarize_loop_attempt(
        selection=selection,
        loop_result=_loop_result("bp35", 3),
        loop_result_path="results/arc_loop_solve_bp35.json",
    )
    preconditions = _preconditions("bp35")
    preconditions["target_offline_env"] = {"game": "bp35", "ok": False}

    artifact = exp4802.build_artifact(
        registry=registry,
        selection=selection,
        attempts=[attempt],
        preconditions_checked=preconditions,
    )

    assert artifact["honest_verdict"] == "blocked_bp35_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_update"]["updated"] is False
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4802_schema_guards_required_fields() -> None:
    """REQ-ARC-WMTE-4802: schema validation rejects overclaims."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4802.select_rotation_target(registry, adaptered_games={"bp35", "sb26"})
    artifact = exp4802.build_artifact(
        registry=registry,
        selection=selection,
        attempts=[
            exp4802.summarize_loop_attempt(
                selection=selection,
                loop_result=_loop_result("bp35", 2),
                loop_result_path="results/arc_loop_solve_bp35.json",
            )
        ],
        preconditions_checked=_preconditions("bp35"),
    )

    missing = dict(artifact)
    missing.pop("attempted_games")
    assert "missing_field:attempted_games" in exp4802.artifact_schema_errors(missing)

    wrong_principle = dict(artifact)
    wrong_principle["field_principles"] = dict(artifact["field_principles"])
    wrong_principle["field_principles"]["solve_provenance"] = "wrong"
    assert "missing_principle:solve_provenance" in exp4802.artifact_schema_errors(wrong_principle)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    assert "invalid_reproducibility_checksum" in exp4802.artifact_schema_errors(bad_checksum)

    stale_checksum = dict(artifact)
    stale_checksum["reproducibility_checksum"] = "0" * 64
    assert "checksum_mismatch" in exp4802.artifact_schema_errors(stale_checksum)

    bad_prefix = dict(artifact)
    bad_prefix["honest_verdict"] = "partial_bp35"
    assert "honest_verdict_missing_terminal_prefix" in exp4802.artifact_schema_errors(bad_prefix)

    bad_provenance = dict(artifact)
    bad_provenance["solve_provenance"] = "outer_loop_re"
    assert "solve_provenance_mismatch" in exp4802.artifact_schema_errors(bad_provenance)

    bad_substrate = dict(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate_mismatch" in exp4802.artifact_schema_errors(bad_substrate)

    bad_oracle = dict(artifact)
    bad_oracle["verifier_is_oracle"] = False
    assert "verifier_is_oracle_must_be_true" in exp4802.artifact_schema_errors(bad_oracle)

    fabricated_bank = dict(artifact)
    fabricated_bank["new_levels_banked"] = 1
    fabricated_bank["reproducibility_checksum"] = exp4802.stable_checksum(fabricated_bank)
    assert "bank_without_offline_reproduction" in exp4802.artifact_schema_errors(fabricated_bank)

    fabricated_repro = dict(artifact)
    fabricated_repro["offline_reproduced"] = True
    fabricated_repro["reproducibility_checksum"] = exp4802.stable_checksum(fabricated_repro)
    assert "offline_reproduced_true_without_new_bank" in exp4802.artifact_schema_errors(fabricated_repro)

    fabricated_retire = dict(artifact)
    fabricated_retire["retire_if_same_verdict"] = False
    fabricated_retire["reproducibility_checksum"] = exp4802.stable_checksum(fabricated_retire)
    assert "retire_if_same_verdict_must_be_true" in exp4802.artifact_schema_errors(fabricated_retire)


def test_req_arc_wmte_4802_collect_and_write_edges(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4802: malformed inputs do not fabricate progress."""

    assert exp4802.registry_levels({"games": [{"game": "bp35", "levels_reproduced": "bad"}]}) == {"bp35": 0}
    assert exp4802.registry_total_levels({"reproducible_total_levels": object()}) == 0

    literal_only = exp4802.select_rotation_target(
        {"games": [{"game": "g50t", "levels_reproduced": 1}, {"game": "r11l", "levels_reproduced": 1}]},
        adaptered_games=set(),
    )
    assert literal_only["game"] == "g50t"
    assert literal_only["rotate_if_no_bank"] == [
        {"game": "r11l", "prior_level": 1, "target_level": 2, "reason": "shallowest_literal_fallback"}
    ]

    no_target = exp4802.select_rotation_target({"games": []}, adaptered_games=set())
    assert no_target["game"] == "none"
    assert no_target["reason"] == "no_reproduced_standing_loop_target"
    assert exp4802.collect_attempts(no_target, results_dir=tmp_path) == []

    missing = exp4802.collect_attempt(
        {"game": "bp35", "prior_level": 2, "target_level": 3, "reason": "unit"},
        results_dir=tmp_path,
    )
    assert missing["residual_cause"] == "missing_loop_result"

    timed = exp4802.collect_attempt(
        {"game": "g50t", "prior_level": 1, "target_level": 2, "reason": "shallowest_literal_fallback"},
        results_dir=tmp_path,
    )
    assert timed["residual_cause"] == "time_budget_no_terminal_gate"

    (tmp_path / "arc_loop_solve_bp35.json").write_text(json.dumps(["bad"]), encoding="utf-8")
    malformed = exp4802.collect_attempt(
        {"game": "bp35", "prior_level": 2, "target_level": 3, "reason": "unit"},
        results_dir=tmp_path,
    )
    assert malformed["residual_cause"] == "offline_reproduction_failed"
    missing_artifact = exp4802.build_artifact(
        registry=yaml.safe_load(_registry_text()),
        selection={"game": "bp35", "prior_level": 2, "target_level": 3, "reason": "unit"},
        attempts=[missing],
        preconditions_checked=_preconditions("bp35"),
    )
    assert missing_artifact["honest_verdict"] == "complete_bp35_no_new_level_residual_missing_loop_result"

    no_attempt_artifact = exp4802.build_artifact(
        registry={},
        selection=no_target,
        attempts=[],
        preconditions_checked=_preconditions("none"),
    )
    assert no_attempt_artifact["honest_verdict"] == "complete_none_no_new_level_residual_no_attempts"

    success_dir = tmp_path / "success"
    success_dir.mkdir()
    (success_dir / "arc_loop_solve_bp35.json").write_text(json.dumps(_loop_result("bp35", 3)), encoding="utf-8")
    early_stop = exp4802.collect_attempts(
        {
            "game": "bp35",
            "prior_level": 2,
            "target_level": 3,
            "reason": "unit",
            "rotate_if_no_bank": [{"game": "sb26", "prior_level": 2, "target_level": 3, "reason": "unit"}],
        },
        results_dir=success_dir,
    )
    assert [attempt["game"] for attempt in early_stop] == ["bp35"]

    payload = {"z": 1, "a": {"b": 2}}
    out = tmp_path / exp4802.RESULT_RELATIVE_PATH
    exp4802.write_artifact(payload, path=out)
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert exp4802.load_registry(tmp_path / "missing.yaml") == {}


def test_req_arc_wmte_4802_main_writes_terminal_artifact(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-4802: main writes a stable terminal artifact from loop results."""

    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / exp4802.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / "results" / "arc_loop_solve_bp35.json").write_text(json.dumps(_loop_result("bp35", 2)), encoding="utf-8")
    (tmp_path / "results" / "arc_loop_solve_sb26.json").write_text(json.dumps(_loop_result("sb26", 2)), encoding="utf-8")

    monkeypatch.setattr(exp4802, "REPO", tmp_path)
    monkeypatch.setattr(exp4802, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(exp4802, "REGISTRY", tmp_path / exp4802.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(exp4802, "ARTIFACT", tmp_path / exp4802.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp4802, "_adaptered_games", lambda: {"bp35", "sb26"})
    monkeypatch.setattr(exp4802, "_recommend_approach", lambda game: _recommendation(game))
    monkeypatch.setattr(exp4802, "check_preconditions", lambda selection: _preconditions(selection["game"]))

    assert exp4802.main([]) == 0

    written = json.loads((tmp_path / exp4802.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == "complete_bp35_no_new_level_residual_existing_depth"
    assert written["target_game"] == "bp35"
    assert written["approach_recommendation"] == _recommendation("bp35")
    assert written["schema_errors"] == []


def test_scenario_arc_wmte_4802_adapter_free_seeding_imports_learned_verifier() -> None:
    """SCENARIO-ARC-WMTE-4802-ADAPTER-FREE-SEEDING: verifier seeding imports its dependency."""

    script = ARC_LOOP_PATH.read_text(encoding="utf-8")
    start = script.index("def solve_via_explore")
    end = script.index("\ndef needs_re", start)
    source = script[start:end]
    assert "from carnot.agentic.arc_value_learner import LearnedVerifier" in source
