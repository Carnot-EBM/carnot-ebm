"""Tests for Exp 4723 .434 ARC capstone scorecard.

Spec refs: REQ-CAPSTONE-4723, SCENARIO-CAPSTONE-4723,
SCENARIO-CAPSTONE-4723-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4723-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4723_capstone_v434 as mod


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
        "experiment": "experiment_4710_capstone_v433",
        "honest_verdict": "complete: capability_grew_61_to_62",
        "bridge_crossed_for_solve": False,
        "reproducible_total_levels": 62,
        "reproducible_total_levels_delta": 1,
        "verifier_is_oracle": False,
    }


def _a1(*, banked: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_4712_perception_grounded_l2_goal_lp85",
        "honest_verdict": (
            "success: perception_grounded_l2_goal_lp85_L2_offline_reproduced"
            if banked
            else "complete: l2_perception_goal_no_deepening_residual_alignment_under_determined"
        ),
        "target_game": "lp85",
        "generic_agent_reached_level": 2 if banked else 1,
        "reproduced_levels": 2 if banked else 0,
        "offline_reproduced": banked,
        "goal_predicate_satisfiable": banked,
        "l2_plan_reaches_goal": banked,
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": False,
    }


def _a2(*, surfaced: bool = False, ablation_level: int = 0) -> dict[str, Any]:
    return {
        "experiment": "experiment_4713_surface_present_winner_verifier_ranker",
        "honest_verdict": (
            "success: surface_present_winner_generic_new_level_lp85"
            if surfaced
            else "complete: surface_present_winner_no_new_level_residual_present_winner_not_separable_from_distractors"
        ),
        "target_game": "lp85",
        "winner_present_coverage": 1.0,
        "winner_rank_pre_surfacing": [59],
        "winner_rank_with_surfacing": [1 if surfaced else 59],
        "precision_at_k_no_surfacing": {"k": 10, "hits": 0, "total": 1, "precision": 0.0},
        "precision_at_k_with_surfacing": {
            "k": 10,
            "hits": 1 if surfaced else 0,
            "total": 1,
            "precision": 1.0 if surfaced else 0.0,
        },
        "precision_at_k_delta": 1.0 if surfaced else 0.0,
        "generic_agent_reached_level": 1 if surfaced else 0,
        "no_surfacing_ablation_reached_level": ablation_level,
        "offline_reproduced": surfaced,
        "reproduced_levels": 1 if surfaced else 0,
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": False,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
    }


def _a3(*, banked: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4714_levelup_selfplay",
        "honest_verdict": "success: bp35_L2_offline_reproduced" if banked else "complete: bp35_delta_identified_no_bank",
        "offline_reproduced": banked,
        "reproduced_levels": 2 if banked else 0,
        "new_levels_banked": 1 if banked else 0,
        "reproducible_total_levels_before": 62,
        "reproducible_total_levels": 63 if banked else 62,
        "target_game": "bp35",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "reproduction_gate": {"game": "bp35", "claimed_level": 2, "reached_level": 2, "reproduced": banked},
    }


def _a4(*, beat: bool = False, l2: bool = False, flagged: bool = False) -> dict[str, Any]:
    delta = 0.05 if beat else 0.0
    return {
        "experiment": "experiment_4715_online_action_learning_driver_corrected",
        "honest_verdict": (
            "success: online_warm_beats_frozen_+0.0500_l2_goal_free"
            if beat and l2
            else "complete: online_action_learning_no_first_win_lift_residual_online_signal_too_sparse"
        ),
        "flagged_adversarial": True if flagged else False,
        "online_warm_first_win": 0.09 if beat else 0.04,
        "frozen_first_win": 0.04,
        "online_warm_vs_frozen_delta": delta,
        "goal_free_l2_reached": l2,
        "offline_reproduced": l2,
        "reproduced_levels": 2 if l2 else 0,
        "solve_provenance": "live_agent_self_discovery" if l2 else "development_proxy",
        "verifier_is_oracle": False,
    }


def _a5() -> dict[str, Any]:
    return {
        "experiment": "experiment_4716_held_out_first_win_readiness",
        "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
        "first_win_rate_integrated": 0.04,
        "first_win_baseline": 0.04,
        "first_win_delta_vs_baseline": 0.0,
        "verifier_is_oracle": False,
    }


def _simple(name: str) -> dict[str, Any]:
    return {
        "experiment": f"experiment_{name}",
        "honest_verdict": "complete: clean_supporting_artifact",
        "verifier_is_oracle": False,
    }


def _b1(reopens: list[str] | None = None) -> dict[str, Any]:
    return {
        "experiment": "experiment_4719_silent_bug_audit",
        "honest_verdict": "success: silent_bug_audit_reopen_list_classified",
        "verifier_is_oracle": False,
        "silent_bug_must_reopen": list(reopens or []),
    }


def _d() -> dict[str, Any]:
    return {
        "experiment": "experiment_4722_sota_ingestion_active_probe_world_model",
        "honest_verdict": "success: sota_ingestion_active_probe_world_model_mapped",
        "verifier_is_oracle": False,
        "flagged_for_next_roadmap": [
            "flagged_for_v435: hypothesis_posterior_active_probe_controller",
            "flagged_for_v435: epistemic_object_model_mcts_probe_planner",
        ],
    }


def _artifacts(
    *,
    a1: Mapping[str, Any] | None = None,
    a2: Mapping[str, Any] | None = None,
    a3: Mapping[str, Any] | None = None,
    a4: Mapping[str, Any] | None = None,
    b1: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "PREVIOUS": _previous(),
        "A1": dict(a1 or _a1()),
        "A2": dict(a2 or _a2()),
        "A3": dict(a3 or _a3()),
        "A4": dict(a4 or _a4()),
        "A5": _a5(),
        "A6": _simple("4717_primitive_persist"),
        "A7": _simple("4718_integration_gate"),
        "B1": dict(b1 or _b1(["experiment_4710_online_action_learning_arms"])),
        "B2": _simple("4720_guard"),
        "C": _simple("4721_kv260"),
        "D": _d(),
    }


def test_req_capstone_4723_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4723: OpenSpec declares the .434 scorecard fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4723" in spec
    assert "SCENARIO-CAPSTONE-4723" in spec
    assert "SCENARIO-CAPSTONE-4723-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4723-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4723_current_missing_upstreams_blocks_but_preserves_available_adjudication() -> None:
    """SCENARIO-CAPSTONE-4723-BLOCKED-PRECONDITION: missing upstreams block completion."""

    partial = _artifacts(a4=_a4(flagged=True))
    for missing in ("A6", "A7", "B1", "B2", "C"):
        partial.pop(missing)

    artifact = mod.build_artifact(
        artifacts=partial,
        live_flags_by_name={"A5": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "attempt count equality"}]},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "blocked_upstream_artifacts"
    assert artifact["bridge_crossed_for_solve"] is False
    assert artifact["a1_lp85_l2_banked"]["banked"] is False
    assert artifact["a2_surfaced_present_winner"]["surfaced"] is False
    assert artifact["a3_levelup_banked"]["banked"] is True
    assert artifact["a4_online_driver_beat_frozen"]["included_in_headline"] is False
    assert artifact["a4_online_driver_beat_frozen"]["reason"] == "flagged_adversarial_or_live_critical"
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["publication_gate"]["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert artifact["missing_artifacts"] == [
        "results/experiment_4717_*.json",
        "results/experiment_4718_*.json",
        "results/experiment_4719_*.json",
        "results/experiment_4720_*.json",
        "results/experiment_4721_*.json",
    ]
    assert "results/experiment_4715_online_action_learning_driver_corrected.json" in artifact["skipped_artifacts"]
    assert "results/experiment_4716_held_out_first_win_readiness.json" in artifact["skipped_artifacts"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4723_a1_bridge_requires_l2_offline_controls_and_provenance() -> None:
    """SCENARIO-CAPSTONE-4723: A1 crosses only for lp85 L2 with controls and provenance."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(banked=True), b1=_b1([])),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    no_goal = _a1(banked=True)
    no_goal["goal_predicate_satisfiable"] = False
    blocked = mod.build_artifact(
        artifacts=_artifacts(a1=no_goal),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: bridge_crossed_for_solve_lp85_L2"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["a1_lp85_l2_banked"]["banked"] is True
    assert artifact["scorecard"]["A1"]["crossed"] is True
    assert blocked["bridge_crossed_for_solve"] is False
    assert blocked["a1_lp85_l2_banked"]["reason"] == "goal_or_plan_control_failed"


def test_scenario_capstone_4723_a2_and_a4_headline_gates_are_adjudicated() -> None:
    """SCENARIO-CAPSTONE-4723: A2 surfacing and A4 online-driver gates are explicit."""

    a2_bridge = mod.build_artifact(
        artifacts=_artifacts(a2=_a2(surfaced=True), b1=_b1([])),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 62},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    a2_control_failed = mod.build_artifact(
        artifacts=_artifacts(a2=_a2(surfaced=True, ablation_level=1)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 62},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    a4_lift = mod.build_artifact(
        artifacts=_artifacts(a4=_a4(beat=True, l2=True), b1=_b1([])),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 62},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert a2_bridge["honest_verdict"] == "complete: bridge_crossed_for_solve_lp85_L1"
    assert a2_bridge["a2_surfaced_present_winner"]["surfaced"] is True
    assert a2_bridge["scorecard"]["A2"]["surfaced"] is True
    assert a2_control_failed["a2_surfaced_present_winner"]["surfaced"] is False
    assert a2_control_failed["a2_surfaced_present_winner"]["reason"] == "no_surfacing_ablation_not_lower"
    assert a4_lift["a4_online_driver_beat_frozen"]["beat_frozen_by_0_05"] is True
    assert a4_lift["a4_online_driver_beat_frozen"]["goal_free_l2_banked"] is True
    assert a4_lift["bridge_crossed_for_solve"] is True


def test_req_capstone_4723_run_writes_artifact_and_validation_fails_closed(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4723: run writes a validated JSON artifact and bad payloads fail closed."""

    artifact = mod.run(
        tmp_path,
        artifacts=_artifacts(b1=_b1(["exp4710", "exp4701"])),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(False),
        write=True,
        duration_s=0.001,
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    bad = dict(artifact)
    bad["verifier_is_oracle_confirmed_false"] = False

    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert artifact["honest_verdict"] == "complete: capability_grew_62_to_63"
    assert artifact["b1_silent_bug_reopen_list"] == ["exp4710", "exp4701"]
    assert artifact["next_milestone_fallback"]["flagged_for_v435"] == [
        "flagged_for_v435: hypothesis_posterior_active_probe_controller",
        "flagged_for_v435: epistemic_object_model_mcts_probe_planner",
    ]
    assert artifact["publication_gate"]["paper_ready"] is False
    assert "G2" in artifact["publication_gate"]["unmet_gates"]
    assert mod._as_int(True, 7) == 7
    assert mod._as_float(False, 1.5) == pytest.approx(1.5)
    assert mod._as_float("bad", 2.5) == pytest.approx(2.5)
    assert mod._extract_reopen_list({"silent_bug_reopen_list": [{"artifact": "a"}, "b"]}) == ["a", "b"]
    assert mod._extract_reopen_list({"must_reopen": "exp"}) == ["exp"]
    assert "verifier_is_oracle_confirmed_false must be true" in mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="verifier_is_oracle_confirmed_false"):
        mod.write_artifact(path=tmp_path / "bad.json", artifact=bad)


def test_req_capstone_4723_defensive_edges_and_reader_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4723: defensive readers and fail-closed branches are covered."""

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
    globbed = tmp_path / "results" / "experiment_4717_fixture.json"
    globbed.parent.mkdir(parents=True, exist_ok=True)
    globbed.write_text('{"experiment": "fixture"}\n', encoding="utf-8")

    assert mod._read_json(json_path) == {"ok": True}
    assert mod._read_json(list_path) == {}
    assert mod._read_yaml(tmp_path / "missing.yaml") == {}
    assert mod._read_yaml(yaml_path) == {"x": 1}
    assert mod._read_yaml(yaml_list) == {}
    assert mod._read_yaml(yaml_bad) == {}
    assert mod._as_int("bad", 9) == 9
    assert mod._file_sha256(tmp_path / "missing") is None
    assert mod._resolve_source(tmp_path, mod.SourceSpec("A6", "results/experiment_4717_*.json", "fixture")) == globbed
    assert mod._resolve_source(tmp_path, mod.SourceSpec("X", "missing_*.json", "fixture")) is None

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
    assert mod._a1_lp85_l2_banked(_a1(banked=True), False)["reason"] == "source_not_clean"
    no_offline = _a1(banked=True)
    no_offline["offline_reproduced"] = False
    no_offline["reproduced_levels"] = 0
    assert mod._a1_lp85_l2_banked(no_offline, True)["reason"] == "offline_l2_reproduction_missing"
    no_provenance = _a1(banked=True)
    no_provenance["solve_provenance"] = ""
    assert mod._a1_lp85_l2_banked(no_provenance, True)["reason"] == "solve_provenance_missing"
    oracle_bad = _a1(banked=True)
    oracle_bad["verifier_is_oracle"] = True
    assert mod._a1_lp85_l2_banked(oracle_bad, True)["reason"] == "verifier_oracle_not_false"
    low_reach = _a1(banked=True)
    low_reach["generic_agent_reached_level"] = 1
    assert mod._a1_lp85_l2_banked(low_reach, True)["reason"] == "generic_agent_did_not_reach_l2"

    assert mod._precision_delta(
        {
            "precision_at_k_with_surfacing": {"precision": 0.75},
            "precision_at_k_no_surfacing": {"precision": 0.25},
        }
    ) == pytest.approx(0.5)
    assert mod._precision_delta({}) == pytest.approx(0.0)
    assert mod._a2_surfaced_present_winner(_a2(surfaced=True), False)["reason"] == "source_not_clean"
    not_present = _a2(surfaced=True)
    not_present["winner_present_coverage"] = 0.0
    assert mod._a2_surfaced_present_winner(not_present, True)["reason"] == "winner_not_present_in_candidate_pool"
    no_precision = _a2(surfaced=True)
    no_precision["precision_at_k_delta"] = 0.0
    assert mod._a2_surfaced_present_winner(no_precision, True)["reason"] == "precision_at_k_not_up"
    no_level = _a2(surfaced=True)
    no_level["generic_agent_reached_level"] = 0
    assert mod._a2_surfaced_present_winner(no_level, True)["reason"] == "generic_agent_no_new_level"
    no_offline_a2 = _a2(surfaced=True)
    no_offline_a2["offline_reproduced"] = False
    assert mod._a2_surfaced_present_winner(no_offline_a2, True)["reason"] == "offline_reproduction_missing"
    no_prov_a2 = _a2(surfaced=True)
    no_prov_a2["solve_provenance"] = ""
    assert mod._a2_surfaced_present_winner(no_prov_a2, True)["reason"] == "solve_provenance_missing"
    oracle_a2 = _a2(surfaced=True)
    oracle_a2["verifier_is_oracle"] = True
    assert mod._a2_surfaced_present_winner(oracle_a2, True)["reason"] == "verifier_oracle_not_false"

    no_reopens = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(banked=True), a2=_a2(surfaced=True), b1=_b1(["exp"])),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    no_bridge = mod.build_artifact(
        artifacts=_artifacts(a3=_a3(banked=False), b1=_b1([])),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 62},
        publication_gate={"paper_ready": True, "gates": "bad", "unmet_gates": "bad"},
        duration_s=0.001,
    )
    no_solve_provenance = _artifacts()
    no_solve_provenance["A3"]["solve_provenance"] = ""
    no_solve = mod.build_artifact(
        artifacts=no_solve_provenance,
        live_flags_by_name={},
        registry={"reproducible_total_levels": 63},
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )
    bad_run = _artifacts()
    bad_run["A6"]["verifier_is_oracle"] = True

    assert no_reopens["next_milestone_fallback"]["strongest_open_lever"] == "B1_silent_bug_reopen"
    assert no_bridge["honest_verdict"] == "complete: no_bridge_crossed_capability_unchanged"
    assert no_bridge["publication_gate"]["gates"] == {}
    assert no_bridge["publication_gate"]["unmet_gates"] == []
    assert no_solve["solve_provenance_confirmed"] is False
    assert mod._flagged_for_v435({"flagged_for_next_roadmap": "bad"}) == []
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
            registry={"reproducible_total_levels": 63},
            publication_gate=_paper_gate(),
            write=False,
            duration_s=0.001,
        )
