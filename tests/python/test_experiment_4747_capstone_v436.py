"""Tests for Exp 4747 .436 ARC capstone scorecard.

Spec refs: REQ-CAPSTONE-4747, SCENARIO-CAPSTONE-4747,
SCENARIO-CAPSTONE-4747-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4747-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4747_capstone_v436 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _paper_gate(ready: bool = True) -> dict[str, Any]:
    return {
        "paper_ready": ready,
        "gates": {
            "G1": {"pass": ready, "detail": "FoVer measured", "source": "experiment_2850.json"},
            "G2": {"pass": ready, "detail": "independent reproducer"},
            "G3": {"pass": ready, "detail": "narrowing clean"},
            "G4": {"pass": ready, "detail": "traceable numbers", "source": "experiment_2850.json"},
        },
        "unmet_gates": [] if ready else ["G2"],
    }


def _previous() -> dict[str, Any]:
    return {
        "experiment": "experiment_4735_capstone_v435",
        "honest_verdict": "complete: capability_grew_63_to_64",
        "bridge_crossed_for_solve": False,
        "reproducible_total_levels": 64,
        "reproducible_total_levels_delta": 1,
        "verifier_is_oracle": False,
    }


def _a1(*, beat: bool = False, l2: bool = False, arms: bool = True, flagged: bool = False) -> dict[str, Any]:
    delta = 0.06 if beat else 0.0
    return {
        "experiment": "experiment_4737_goal_energy_candidate_generation_valid_test",
        "honest_verdict": (
            "success: goal_energy_generation_first_win_lift_0.06_or_l2_none"
            if beat
            else "complete: goal_energy_generation_no_first_win_lift_residual_goal_energy_does_not_up_weight_the_winner"
        ),
        "flagged_adversarial": flagged,
        "arms_non_degenerate": arms,
        "candidate_pool_differs_from_baseline": arms,
        "goal_energy_score_variance": 0.1 if arms else 0.0,
        "goal_energy_first_win": 0.10 if beat else 0.04,
        "baseline_first_win": 0.04,
        "goal_energy_vs_baseline_delta": delta,
        "goal_free_l2_reached": l2,
        "offline_reproduced": l2,
        "reproduced_levels": 2 if l2 else 0,
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": False,
        "positive_control_passed": arms,
        "bare_control_passed": True,
        "parity_test_green": True,
        "null_delta_methodology_note": "valid non-degenerate zero-lift null" if not beat else "",
    }


def _a2(*, generated: bool = False, l2: bool = False, arms: bool = True) -> dict[str, Any]:
    delta = 0.08 if generated else 0.0
    return {
        "experiment": "experiment_4738_energy_fitness_qd_generation_valid_test",
        "honest_verdict": (
            "success: energy_qd_generation_first_win_lift_0.08_or_l2_2"
            if l2
            else "complete: energy_qd_generation_no_first_win_lift_residual_winner_not_in_reachable_mutation_neighborhood"
        ),
        "arms_non_degenerate": arms,
        "arm_pool_jaccard": {
            "naive-search__random-mutation": 0.5,
            "naive-search__energy-QD": 0.25,
            "random-mutation__energy-QD": 0.75,
        },
        "novel_candidates_generated": 4 if arms else 0,
        "energy_qd_first_win": 0.12 if generated else 0.04,
        "naive_search_first_win": 0.04,
        "energy_qd_vs_naive_delta": delta,
        "winner_generated_by_energy_qd": generated,
        "naive_search_generated_winner": False if generated else None,
        "goal_free_l2_reached": l2,
        "offline_reproduced": l2,
        "reproduced_levels": 2 if l2 else 0,
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": False,
        "positive_control_passed": arms,
        "bare_control_passed": True,
        "parity_test_green": True,
        "null_delta_methodology_note": "valid non-degenerate zero-lift null" if not generated else "",
        "target_game": "lp85",
    }


def _a3(*, banked: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_4739_levelup_selfplay",
        "honest_verdict": "success: re86_L3_offline_reproduced" if banked else "complete: re86_delta_identified_no_bank",
        "offline_reproduced": True,
        "reproduced_levels": 3 if banked else 2,
        "new_levels_banked": 1 if banked else 0,
        "reproducible_total_levels_before": 64,
        "reproducible_total_levels": 65 if banked else 64,
        "target_game": "re86",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
    }


def _b1() -> dict[str, Any]:
    return {
        "experiment": "experiment_4743_adversarial_verify_carveout_hardening",
        "honest_verdict": "success: adversarial_verify_carveout_hardening_shipped_pinned.",
        "tautology_carveout_added": {"passed": True},
        "exercise_evidence_extension_added": {"passed": True},
        "a1_exemplar_downgraded_to_warn": {"passed": True},
        "a2_exemplar_flagged": {"passed": True},
        "positive_exercise_null_not_flagged": {"passed": True},
        "verifier_is_oracle": False,
    }


def _b2(*, ready: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_4744_submission_package_readiness",
        "honest_verdict": "success: submission_package_ready" if ready else "complete: submission_package_blocked_manifest_resources",
        "submission_package_ready": ready,
        "frozen_generator_confirmed": True,
        "parity_test_green": ready,
        "verifier_is_oracle": False,
    }


def _d() -> dict[str, Any]:
    return {
        "experiment": "experiment_4746_sota_ingestion_epistemic_mcts_causal_probe",
        "honest_verdict": "success: sota_ingestion_epistemic_mcts_causal_probe_matm_mapped",
        "verifier_is_oracle": False,
        "flagged_for_next_roadmap": [
            "flagged_for_v437: epistemic_object_model_mcts_probe_planner",
            "flagged_for_v437: factored_interaction_causal_probe_bank",
        ],
    }


def _simple(experiment: str) -> dict[str, Any]:
    return {"experiment": experiment, "honest_verdict": "complete: clean_supporting_artifact", "verifier_is_oracle": False}


def _artifacts(
    *,
    a1: Mapping[str, Any] | None = None,
    a2: Mapping[str, Any] | None = None,
    a3: Mapping[str, Any] | None = None,
    b1: Mapping[str, Any] | None = None,
    b2: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "PREVIOUS": _previous(),
        "A1": dict(a1 or _a1()),
        "A2": dict(a2 or _a2()),
        "A3": dict(a3 or _a3()),
        "A4": _simple("experiment_4740_held_out_first_win_readiness"),
        "A5": _simple("experiment_4741_primitive_persist_transfer"),
        "A6": _simple("experiment_4742_integration_gate"),
        "B1": dict(b1 or _b1()),
        "B2": dict(b2 or _b2()),
        "C": _simple("experiment_4745_kv260_continuity"),
        "D": _d(),
    }


def test_req_capstone_4747_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4747: OpenSpec declares the .436 scorecard fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4747" in spec
    assert "SCENARIO-CAPSTONE-4747" in spec
    assert "SCENARIO-CAPSTONE-4747-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4747-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4747_default_honest_null_scorecard() -> None:
    """SCENARIO-CAPSTONE-4747: clean non-degenerate nulls do not cross the bridge."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: no_bridge_crossed_capability_unchanged"
    assert artifact["bridge_crossed_for_solve"] is False
    assert artifact["reproducible_total_levels_delta"] == 0
    assert artifact["a1_goal_energy_result"]["arms_non_degenerate"] is True
    assert artifact["a1_goal_energy_result"]["beat_baseline_by_0_05"] is False
    assert artifact["a1_goal_energy_result"]["reason"] == "goal_energy_real_non_degenerate_zero_lift_null"
    assert artifact["a2_energy_qd_result"]["arms_non_degenerate"] is True
    assert artifact["a2_energy_qd_result"]["generated_winner_where_naive_missed"] is False
    assert artifact["a2_energy_qd_result"]["reason"] == "energy_qd_real_non_degenerate_zero_lift_null"
    assert artifact["a3_banked_level"]["banked"] is False
    assert artifact["b1_carveout_fix_confirmed"] is True
    assert artifact["submission_package_ready"] is False
    assert artifact["headline_decision"]["capability_delta"] == 0
    assert artifact["next_milestone_fallback"]["flagged_for_v437"] == [
        "flagged_for_v437: epistemic_object_model_mcts_probe_planner",
        "flagged_for_v437: factored_interaction_causal_probe_bank",
    ]
    assert artifact["next_milestone_fallback"]["deferred_reopens"] == [
        "P1_go_explore",
        "P4_subgoal",
        "A2_active_probe",
    ]
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["publication_gate"]["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4747_a1_a2_a3_decision_gates() -> None:
    """SCENARIO-CAPSTONE-4747: A1/A2 bridge gates require L2 reproduction."""

    lift_only = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(beat=True)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    a1_l2 = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(l2=True)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    a2_l2 = mod.build_artifact(
        artifacts=_artifacts(a2=_a2(generated=True, l2=True)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    a3_bank = mod.build_artifact(
        artifacts=_artifacts(a3=_a3(banked=True)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 65},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert lift_only["a1_goal_energy_result"]["beat_baseline_by_0_05"] is True
    assert lift_only["bridge_crossed_for_solve"] is False
    assert a1_l2["honest_verdict"] == "complete: bridge_crossed_for_solve_goal_energy_L2"
    assert a1_l2["bridge_crossed_for_solve"] is True
    assert a2_l2["honest_verdict"] == "complete: bridge_crossed_for_solve_lp85_L2"
    assert a2_l2["a2_energy_qd_result"]["generated_winner_where_naive_missed"] is True
    assert a2_l2["bridge_crossed_for_solve"] is True
    assert a3_bank["honest_verdict"] == "complete: capability_grew_64_to_65"
    assert a3_bank["a3_banked_level"]["banked"] is True
    assert a3_bank["reproducible_total_levels_delta"] == 1


def test_scenario_capstone_4747_missing_and_flagged_sources_fail_closed() -> None:
    """SCENARIO-CAPSTONE-4747-BLOCKED-PRECONDITION: missing and flagged inputs are not aggregated."""

    partial = _artifacts(a1=_a1(beat=True, l2=True, flagged=True), b2={**_b2(ready=True), "control_failed": True})
    partial.pop("A4")

    artifact = mod.build_artifact(
        artifacts=partial,
        live_flags_by_name={"A1": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"}]},
        registry={"reproducible_total_levels": 65},
        publication_gate=_paper_gate(False),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "blocked_upstream_artifacts"
    assert artifact["bridge_crossed_for_solve"] is False
    assert artifact["a1_goal_energy_result"]["included_in_headline"] is False
    assert artifact["a1_goal_energy_result"]["reason"] == "flagged_adversarial_or_live_critical"
    assert artifact["submission_package_ready"] is False
    assert artifact["missing_artifacts"] == ["results/experiment_4740_*.json"]
    assert "results/experiment_4737_*.json" in artifact["skipped_artifacts"]
    assert "results/experiment_4744_*.json" in artifact["skipped_artifacts"]
    assert "G2" in artifact["publication_gate"]["unmet_gates"]
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4747_run_writes_and_validation_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4747: run writes a validated JSON artifact and invalid payloads fail."""

    artifact = mod.run(
        tmp_path,
        artifacts=_artifacts(b2=_b2(ready=True)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        write=True,
        duration_s=0.001,
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    bad = dict(artifact)
    bad["verifier_is_oracle_confirmed_false"] = False

    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert artifact["submission_package_ready"] is True
    assert artifact["headline_decision"]["submission_package_ready"] is True
    assert mod._as_int(True, 7) == 7
    assert mod._as_int("bad", 9) == 9
    assert mod._as_float(False, 1.5) == pytest.approx(1.5)
    assert mod._as_float("bad", 2.5) == pytest.approx(2.5)
    assert mod._flagged_for_v437({"flagged_for_next_roadmap": "bad"}) == []
    assert "verifier_is_oracle_confirmed_false must be true" in mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="verifier_is_oracle_confirmed_false"):
        mod.write_artifact(path=tmp_path / "bad.json", artifact=bad)


def test_req_capstone_4747_defensive_edges_and_reader_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4747: readers, skip reasons, and fail-closed branches are covered."""

    json_path = tmp_path / "dict.json"
    json_path.write_text('{"ok": true}\n', encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_path.write_text("[1, 2]\n", encoding="utf-8")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("x: 1\n", encoding="utf-8")
    yaml_list = tmp_path / "list.yaml"
    yaml_list.write_text("- 1\n", encoding="utf-8")
    yaml_bad = tmp_path / "bad.yaml"
    yaml_bad.write_text("x: [\n", encoding="utf-8")
    globbed = tmp_path / "results" / "experiment_4740_fixture.json"
    globbed.parent.mkdir(parents=True, exist_ok=True)
    globbed.write_text('{"experiment": "fixture"}\n', encoding="utf-8")

    assert mod._read_json(json_path) == {"ok": True}
    assert mod._read_json(list_path) == {}
    assert mod._read_yaml(tmp_path / "missing.yaml") == {}
    assert mod._read_yaml(yaml_path) == {"x": 1}
    assert mod._read_yaml(yaml_list) == {}
    assert mod._read_yaml(yaml_bad) == {}
    assert mod._file_sha256(tmp_path / "missing") is None
    assert mod._resolve_source(tmp_path, mod.SourceSpec("A4", "results/experiment_4740_*.json", "fixture")) == globbed
    exact = tmp_path / "exact.json"
    exact.write_text('{"ok": true}\n', encoding="utf-8")
    assert mod._resolve_source(tmp_path, mod.SourceSpec("X", "exact.json", "fixture")) == exact

    monkeypatch.setattr(mod, "artifact_reader", None)
    assert mod._summarize_and_live_flags(json_path) == (None, [])
    monkeypatch.setattr(
        mod,
        "artifact_reader",
        type(
            "Reader",
            (),
            {
                "summarize": staticmethod(lambda _path: 2),
                "_live_flags": staticmethod(
                    lambda _path: [
                        {"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"},
                        "not-a-mapping",
                    ]
                ),
            },
        ),
    )
    summary_code, flags = mod._summarize_and_live_flags(json_path)
    summarized_status = mod._source_status(
        name="X",
        source=mod.SourceSpec("X", "dict.json", "fixture"),
        root=tmp_path,
        artifact={"honest_verdict": "complete: fixture", "verifier_is_oracle": False},
        path=json_path,
        live_flags_by_name=None,
    )
    failed_gate = mod._source_status(
        name="X",
        source=mod.SourceSpec("X", "dict.json", "fixture"),
        root=tmp_path,
        artifact={"honest_verdict": "complete: fixture", "gate_fixture": False},
        path=json_path,
        live_flags_by_name={},
    )
    ablation_missing = mod._source_status(
        name="X",
        source=mod.SourceSpec("X", "dict.json", "fixture"),
        root=tmp_path,
        artifact={"honest_verdict": "complete: fixture", "ablation_missing": True},
        path=json_path,
        live_flags_by_name={},
    )

    assert summary_code == 2
    assert flags == [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"}]
    assert summarized_status["reason"] == "flagged_adversarial_or_live_critical"
    assert failed_gate["reason"] == "failed_gate"
    assert ablation_missing["reason"] == "ablation_missing"

    monkeypatch.setattr(mod, "publication_gate_reader", None)
    assert mod._load_publication_gate()["unmet_gates"] == ["publication_gate_unavailable"]
    monkeypatch.setattr(mod, "publication_gate_reader", type("Gate", (), {"evaluate": staticmethod(lambda: _paper_gate())}))
    assert mod._load_publication_gate()["paper_ready"] is True

    loaded, statuses = mod._load_artifacts(tmp_path, artifacts=None, live_flags_by_name={})
    blocked = mod.check_preconditions(tmp_path, statuses=statuses, registry_payload={}, publication_gate_available=False)
    assert loaded["A4"] == {"experiment": "fixture"}
    assert blocked["ok"] is False
    assert blocked["blocked_resource"] == "registry_yaml"
    assert mod._target_game({"target_games": ["x"]}, "default") == "x"
    assert mod._target_game({}, "default") == "default"

    assert mod._a1_goal_energy_result(_a1(arms=False), True, {})["reason"] == "goal_energy_generation_arms_degenerate"
    a1_no_offline = _a1(l2=True)
    a1_no_offline["offline_reproduced"] = False
    assert mod._a1_goal_energy_result(a1_no_offline, True, {})["reason"] == "goal_energy_l2_not_offline_reproduced"
    a1_no_provenance = _a1(l2=True)
    a1_no_provenance["solve_provenance"] = ""
    assert mod._a1_goal_energy_result(a1_no_provenance, True, {})["reason"] == "solve_provenance_missing"
    a1_low_delta = _a1()
    a1_low_delta["null_delta_methodology_note"] = ""
    assert mod._a1_goal_energy_result(a1_low_delta, True, {})["reason"] == "goal_energy_delta_below_0_05"

    assert mod._a2_energy_qd_result(_a2(), False, {"reason": "flagged"})["reason"] == "flagged"
    assert mod._a2_energy_qd_result(_a2(arms=False), True, {})["reason"] == "energy_qd_generation_arms_degenerate"
    assert mod._a2_energy_qd_result(_a2(generated=True, l2=False), True, {})["reason"] == "energy_qd_generated_winner_but_no_l2_bank"
    a2_no_offline = _a2(generated=True, l2=True)
    a2_no_offline["offline_reproduced"] = False
    assert mod._a2_energy_qd_result(a2_no_offline, True, {})["reason"] == "energy_qd_l2_not_offline_reproduced"
    a2_no_provenance = _a2(generated=True, l2=True)
    a2_no_provenance["solve_provenance"] = ""
    assert mod._a2_energy_qd_result(a2_no_provenance, True, {})["reason"] == "solve_provenance_missing"
    a2_low_delta = _a2()
    a2_low_delta["null_delta_methodology_note"] = ""
    assert mod._a2_energy_qd_result(a2_low_delta, True, {})["reason"] == "energy_qd_delta_below_0_05_or_naive_not_missed"

    no_solve_provenance = _artifacts()
    no_solve_provenance["A3"]["solve_provenance"] = ""
    no_solve = mod.build_artifact(
        artifacts=no_solve_provenance,
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate={"paper_ready": True, "gates": "bad", "unmet_gates": "bad"},
        duration_s=0.001,
    )
    strongest_a1 = mod.build_artifact(
        artifacts=_artifacts(a2=_a2(generated=True)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    strongest_a3 = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(beat=True), a2=_a2(generated=True)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    strongest_b2 = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(beat=True), a2=_a2(generated=True), a3=_a3(banked=True)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 65},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    strongest_d = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(beat=True), a2=_a2(generated=True), a3=_a3(banked=True), b2=_b2(ready=True)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 65},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    bad_run = _artifacts()
    bad_run["A5"]["verifier_is_oracle"] = True

    assert no_solve["solve_provenance_confirmed"] is False
    assert no_solve["publication_gate"]["gates"] == {}
    assert strongest_a1["next_milestone_fallback"]["strongest_open_lever"] == "A1_goal_energy_generation"
    assert strongest_a3["next_milestone_fallback"]["strongest_open_lever"] == "A3_levelup_bank"
    assert strongest_b2["next_milestone_fallback"]["strongest_open_lever"] == "B2_submission_package"
    assert strongest_d["next_milestone_fallback"]["strongest_open_lever"] == "D_flagged_for_v437"

    invalid = dict(strongest_d)
    invalid.pop("scorecard")
    invalid["honest_verdict"] = "running"
    invalid["inference_substrate"] = "live"
    invalid["leaderboard_submission"] = True
    invalid["random_seed"] = 1
    invalid["publication_gate"] = {"frozen_fover_auroc": 0.1}
    invalid["reproducibility_checksum"] = "sha256:bad"
    validation = mod.validate_artifact(invalid)
    assert "missing required field: scorecard" in validation
    assert "honest_verdict must be terminal-prefixed" in validation
    assert "inference_substrate must declare aggregation_from_upstream_artifacts" in validation
    assert "leaderboard_submission must be false" in validation
    assert "random_seed mismatch" in validation
    assert "publication_gate must preserve frozen FoVer 0.9131" in validation
    assert "reproducibility_checksum mismatch" in validation
    with pytest.raises(ValueError, match="verifier_is_oracle_confirmed_false"):
        mod.run(
            tmp_path,
            artifacts=bad_run,
            live_flags_by_name={},
            registry={"reproducible_total_levels": 64},
            publication_gate=_paper_gate(),
            write=False,
            duration_s=0.001,
        )
