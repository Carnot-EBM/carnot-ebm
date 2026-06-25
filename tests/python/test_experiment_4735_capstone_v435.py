"""Tests for Exp 4735 .435 ARC capstone scorecard.

Spec refs: REQ-CAPSTONE-4735, SCENARIO-CAPSTONE-4735,
SCENARIO-CAPSTONE-4735-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4735-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4735_capstone_v435 as mod


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
        "experiment": "experiment_4723_capstone_v434",
        "honest_verdict": "complete: capability_grew_62_to_63",
        "bridge_crossed_for_solve": False,
        "reproducible_total_levels": 63,
        "reproducible_total_levels_delta": 1,
        "verifier_is_oracle": False,
    }


def _b1(reopens: list[str] | None = None) -> dict[str, Any]:
    return {
        "experiment": "experiment_4725_silent_bug_audit",
        "honest_verdict": "complete: silent_bug_audit_12_nulls_5_must_reopen",
        "verifier_is_oracle": False,
        "silent_bug_must_reopen": list(
            reopens
            or [
                "experiment_4701_amortized_exploration_prior_go_explore_live",
                "experiment_4715_online_action_learning_driver_corrected",
            ]
        ),
        "a4_tautology_verdict": "online_driver_arms_degenerate (no-op, must reopen)",
    }


def _a1(
    *,
    arms_non_degenerate: bool = True,
    beat: bool = False,
    l2: bool = False,
    flagged: bool = False,
) -> dict[str, Any]:
    delta = 0.06 if beat else 0.0
    return {
        "experiment": "experiment_4726_online_action_learning_driver_valid_test",
        "honest_verdict": (
            "success: online_warm_beats_frozen_0.06_or_l2_goal_free"
            if beat or l2
            else "complete: online_action_learning_no_first_win_lift_residual_online_signal_genuinely_too_sparse"
        ),
        "flagged_adversarial": flagged,
        "arms_non_degenerate": arms_non_degenerate,
        "per_arm_action_distribution_distinct": arms_non_degenerate,
        "online_train_steps_executed": 66 if arms_non_degenerate else 0,
        "online_warm_first_win": 0.10 if beat else 0.04,
        "frozen_first_win": 0.04,
        "online_warm_vs_frozen_delta": delta,
        "goal_free_l2_reached": l2,
        "offline_reproduced": l2,
        "reproduced_levels": 2 if l2 else 0,
        "solve_provenance": "live_agent_self_discovery" if l2 else "development_proxy",
        "verifier_is_oracle": False,
        "chosen_submitted_config": "online_warm_action_effect_controller" if beat else "unchanged",
        "positive_control_passed": arms_non_degenerate,
        "parity_test_green": True,
    }


def _a2(*, reached: int = 0, ablation_level: int = 0, probed: bool = True) -> dict[str, Any]:
    reached_new = reached > 0
    return {
        "experiment": "experiment_4727_active_probe_disambiguation",
        "honest_verdict": (
            f"success: active_probe_generic_agent_new_level_bp35_L{reached}"
            if reached_new
            else "complete: active_probe_no_new_level_residual_budget_insufficient"
        ),
        "target_game": "bp35",
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "live_path_reachable": True,
        "hypothesis_posterior_built": probed,
        "probe_actions_taken": 3 if probed else 0,
        "posterior_entropy_reduction": 0.25 if probed else 0.0,
        "generic_agent_reached_level": reached,
        "no_probe_ablation_reached_level": ablation_level,
        "offline_reproduced": reached_new,
        "reproduced_levels": reached,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "chosen_submitted_config": "active_probe_controller" if reached_new else "unchanged",
        "parity_test_green": True,
    }


def _a3(*, banked: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4728_levelup_selfplay",
        "honest_verdict": "success: ar25_L3_offline_reproduced" if banked else "complete: ar25_delta_identified_no_bank",
        "offline_reproduced": banked,
        "reproduced_levels": 3 if banked else 2,
        "new_levels_banked": 1 if banked else 0,
        "reproducible_total_levels_before": 63,
        "reproducible_total_levels": 64 if banked else 63,
        "target_game": "ar25",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "reproduction_gate": {"game": "ar25", "claimed_level": 3, "reached_level": 3, "reproduced": banked},
    }


def _a4() -> dict[str, Any]:
    return {
        "experiment": "experiment_4729_held_out_first_win_readiness",
        "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
        "first_win_rate_integrated": 0.04,
        "first_win_baseline": 0.04,
        "first_win_delta_vs_baseline": 0.0,
        "first_win_ci_lower": 0.0,
        "multi_level_deepen_rate_integrated": 0.0,
        "parity_test_green": True,
        "positive_control_passed": True,
        "null_delta_methodology_note": "flat first-win null with positive control",
        "verifier_is_oracle": False,
    }


def _simple(experiment: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "honest_verdict": "complete: clean_supporting_artifact",
        "verifier_is_oracle": False,
    }


def _d() -> dict[str, Any]:
    return {
        "experiment": "experiment_4734_sota_ingestion_epistemic_mcts_causal_probe",
        "honest_verdict": "success: sota_ingestion_epistemic_mcts_causal_probe_mapped",
        "verifier_is_oracle": False,
        "flagged_for_next_roadmap": [
            "flagged_for_v436: epistemic_object_model_mcts_probe_planner",
            "flagged_for_v436: factored_interaction_causal_probe_bank",
        ],
    }


def _artifacts(
    *,
    b1: Mapping[str, Any] | None = None,
    a1: Mapping[str, Any] | None = None,
    a2: Mapping[str, Any] | None = None,
    a3: Mapping[str, Any] | None = None,
    a4: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "PREVIOUS": _previous(),
        "B1": dict(b1 or _b1()),
        "A1": dict(a1 or _a1()),
        "A2": dict(a2 or _a2()),
        "A3": dict(a3 or _a3()),
        "A4": dict(a4 or _a4()),
        "A5": _simple("experiment_4730_primitive_persist_transfer"),
        "A6": _simple("experiment_4731_integration_gate"),
        "B2": _simple("experiment_4732_adversarial_verify_exercise_evidence_guard"),
        "C": _simple("experiment_4733_kv260_continuity"),
        "D": _d(),
    }


def test_req_capstone_4735_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4735: OpenSpec declares the .435 scorecard fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4735" in spec
    assert "SCENARIO-CAPSTONE-4735" in spec
    assert "SCENARIO-CAPSTONE-4735-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4735-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4735_missing_a4_blocks_and_skips_flagged_a1() -> None:
    """SCENARIO-CAPSTONE-4735-BLOCKED-PRECONDITION: missing upstreams block completion."""

    partial = _artifacts(a1=_a1(arms_non_degenerate=True, beat=True, l2=True, flagged=True))
    partial.pop("A4")

    artifact = mod.build_artifact(
        artifacts=partial,
        live_flags_by_name={"A1": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"}]},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "blocked_upstream_artifacts"
    assert artifact["bridge_crossed_for_solve"] is False
    assert artifact["a1_online_driver_result"]["included_in_headline"] is False
    assert artifact["a1_online_driver_result"]["reason"] == "flagged_adversarial_or_live_critical"
    assert artifact["a3_levelup_banked"]["banked"] is True
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["missing_artifacts"] == ["results/experiment_4729_*.json"]
    assert "results/experiment_4726_*.json" in artifact["skipped_artifacts"]
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["publication_gate"]["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert artifact["headline_decision"]["a1_arms_non_degenerate"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4735_a1_lift_is_characterized_but_l2_crosses_bridge() -> None:
    """SCENARIO-CAPSTONE-4735: A1 answers lift and L2 separately."""

    lift_only = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(arms_non_degenerate=True, beat=True, l2=False), a3=_a3(banked=False)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    l2 = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(arms_non_degenerate=True, beat=False, l2=True), a3=_a3(banked=False)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    degenerate = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(arms_non_degenerate=False), a3=_a3(banked=False)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert lift_only["a1_online_driver_result"]["arms_non_degenerate"] is True
    assert lift_only["a1_online_driver_result"]["beat_frozen_by_0_05"] is True
    assert lift_only["bridge_crossed_for_solve"] is False
    assert lift_only["honest_verdict"] == "complete: no_bridge_crossed_capability_unchanged"
    assert l2["honest_verdict"] == "complete: bridge_crossed_for_solve_online_driver_L2"
    assert l2["bridge_crossed_for_solve"] is True
    assert l2["a1_online_driver_result"]["crossed"] is True
    assert degenerate["a1_online_driver_result"]["reason"] == "online_driver_arms_degenerate"
    assert degenerate["headline_decision"]["a1_arms_non_degenerate"] is False


def test_scenario_capstone_4735_a2_bridge_requires_no_probe_ablation_failing() -> None:
    """SCENARIO-CAPSTONE-4735: A2 counts only with probing, reproduction, and ablation."""

    crossed = mod.build_artifact(
        artifacts=_artifacts(a2=_a2(reached=2, ablation_level=1), a3=_a3(banked=False)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    ablation_not_lower = mod.build_artifact(
        artifacts=_artifacts(a2=_a2(reached=2, ablation_level=2), a3=_a3(banked=False)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    no_probe = mod.build_artifact(
        artifacts=_artifacts(a2=_a2(reached=2, ablation_level=1, probed=False), a3=_a3(banked=False)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert crossed["honest_verdict"] == "complete: bridge_crossed_for_solve_bp35_L2"
    assert crossed["bridge_crossed_for_solve"] is True
    assert crossed["a2_active_probe_new_level"]["crossed"] is True
    assert crossed["scorecard"]["A2"]["surfaced"] is True
    assert ablation_not_lower["a2_active_probe_new_level"]["reason"] == "no_probe_ablation_not_lower"
    assert no_probe["a2_active_probe_new_level"]["reason"] == "probe_mechanism_did_not_run"


def test_req_capstone_4735_run_writes_and_validates_artifact(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4735: run writes a validated JSON artifact and bad payloads fail closed."""

    artifact = mod.run(
        tmp_path,
        artifacts=_artifacts(b1=_b1(["exp4715_a4", "exp4701_go_explore"])),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(False),
        write=True,
        duration_s=0.001,
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    bad = dict(artifact)
    bad["verifier_is_oracle_confirmed_false"] = False

    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert artifact["honest_verdict"] == "complete: capability_grew_63_to_64"
    assert artifact["b1_silent_bug_reopen_list"] == ["exp4715_a4", "exp4701_go_explore"]
    assert artifact["headline_decision"]["b1_reopened_434_a4"] is True
    assert artifact["next_milestone_fallback"]["flagged_for_v436"] == [
        "flagged_for_v436: epistemic_object_model_mcts_probe_planner",
        "flagged_for_v436: factored_interaction_causal_probe_bank",
    ]
    assert artifact["publication_gate"]["paper_ready"] is False
    assert "G2" in artifact["publication_gate"]["unmet_gates"]
    assert mod._as_int(True, 7) == 7
    assert mod._as_int("bad", 9) == 9
    assert mod._as_float(False, 1.5) == pytest.approx(1.5)
    assert mod._as_float("bad", 2.5) == pytest.approx(2.5)
    assert mod._extract_reopen_list({"must_reopen": "exp"}) == ["exp"]
    assert mod._extract_reopen_list({"silent_bug_nulls": [{"artifact": "a"}, {"lever": "b"}, {"id": "c"}]}) == [
        "a",
        "b",
        "c",
    ]
    assert "verifier_is_oracle_confirmed_false must be true" in mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="verifier_is_oracle_confirmed_false"):
        mod.write_artifact(path=tmp_path / "bad.json", artifact=bad)


def test_req_capstone_4735_defensive_edges_and_reader_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4735: defensive readers and fail-closed branches are covered."""

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
    globbed = tmp_path / "results" / "experiment_4729_fixture.json"
    globbed.parent.mkdir(parents=True, exist_ok=True)
    globbed.write_text('{"experiment": "fixture"}\n', encoding="utf-8")

    assert mod._read_json(json_path) == {"ok": True}
    assert mod._read_json(list_path) == {}
    assert mod._read_yaml(tmp_path / "missing.yaml") == {}
    assert mod._read_yaml(yaml_path) == {"x": 1}
    assert mod._read_yaml(yaml_list) == {}
    assert mod._read_yaml(yaml_bad) == {}
    assert mod._file_sha256(tmp_path / "missing") is None
    assert mod._resolve_source(tmp_path, mod.SourceSpec("A4", "results/experiment_4729_*.json", "fixture")) == globbed

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
    control_failed = mod._source_status(
        name="X",
        source=mod.SourceSpec("X", "dict.json", "fixture"),
        root=tmp_path,
        artifact={"honest_verdict": "complete: fixture", "control_failed": True},
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
    assert control_failed["reason"] == "control_failed"
    assert ablation_missing["reason"] == "ablation_missing"

    monkeypatch.setattr(mod, "publication_gate_reader", None)
    assert mod._load_publication_gate()["unmet_gates"] == ["publication_gate_unavailable"]
    monkeypatch.setattr(
        mod,
        "publication_gate_reader",
        type("Gate", (), {"evaluate": staticmethod(lambda: _paper_gate())}),
    )
    assert mod._load_publication_gate()["paper_ready"] is True

    loaded, statuses = mod._load_artifacts(tmp_path, artifacts=None, live_flags_by_name={})
    blocked = mod.check_preconditions(
        tmp_path,
        statuses=statuses,
        registry_payload={},
        publication_gate_available=False,
    )
    assert loaded["PREVIOUS"] == {}
    assert blocked["ok"] is False
    assert "registry_yaml" in blocked["blocked_resource"]

    assert mod._target_game({"target_games": ["x"]}, "default") == "x"
    assert mod._target_game({}, "default") == "default"
    assert mod._a1_online_driver_result(_a1(l2=True), False, {})["reason"] == "source_not_clean"
    no_offline_a1 = _a1(l2=True)
    no_offline_a1["offline_reproduced"] = False
    assert mod._a1_online_driver_result(no_offline_a1, True, {})["reason"] == "goal_free_l2_not_offline_reproduced"
    no_provenance_a1 = _a1(l2=True)
    no_provenance_a1["solve_provenance"] = ""
    assert mod._a1_online_driver_result(no_provenance_a1, True, {})["reason"] == "solve_provenance_missing"

    assert mod._a2_active_probe_new_level(_a2(reached=1), False)["reason"] == "source_not_clean"
    no_offline_a2 = _a2(reached=1)
    no_offline_a2["offline_reproduced"] = False
    assert mod._a2_active_probe_new_level(no_offline_a2, True)["reason"] == "offline_reproduction_missing"
    no_prov_a2 = _a2(reached=1)
    no_prov_a2["solve_provenance"] = ""
    assert mod._a2_active_probe_new_level(no_prov_a2, True)["reason"] == "solve_provenance_missing"
    oracle_a2 = _a2(reached=1)
    oracle_a2["verifier_is_oracle"] = True
    assert mod._a2_active_probe_new_level(oracle_a2, True)["reason"] == "verifier_oracle_not_false"

    no_bridge = mod.build_artifact(
        artifacts=_artifacts(a3=_a3(banked=False), b1=_b1([])),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate={"paper_ready": True, "gates": "bad", "unmet_gates": "bad"},
        duration_s=0.001,
    )
    no_solve_provenance = _artifacts()
    no_solve_provenance["A3"]["solve_provenance"] = ""
    no_solve = mod.build_artifact(
        artifacts=no_solve_provenance,
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    b1_strongest_open = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(l2=True),
            a2=_a2(reached=2, ablation_level=1),
            b1=_b1(["experiment_4715_online_action_learning_driver_corrected"]),
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 64},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    bad_run = _artifacts()
    bad_run["A5"]["verifier_is_oracle"] = True

    assert no_bridge["honest_verdict"] == "complete: no_bridge_crossed_capability_unchanged"
    assert no_bridge["publication_gate"]["gates"] == {}
    assert no_solve["solve_provenance_confirmed"] is False
    assert b1_strongest_open["next_milestone_fallback"]["strongest_open_lever"] == "B1_silent_bug_reopen"
    assert mod._flagged_for_v436({"flagged_for_next_roadmap": "bad"}) == []
    assert mod._extract_reopen_list({}) == []

    invalid = dict(no_bridge)
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
