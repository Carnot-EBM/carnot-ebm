"""Tests for REQ-CAPSTONE-4923 / SCENARIO-CAPSTONE-4923."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4923_capstone_v453 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1(
    *,
    fork: str = "WALL_IS_HIDDEN_STATE",
    flagged: bool = False,
    lever: str = "visible_level_before",
) -> JsonDict:
    observable_gap = fork == "WALL_IS_OBSERVABLE_VARIABLE_GAP"
    return {
        "experiment_id": 4914,
        "honest_verdict": (
            "complete_causal_abstraction_observable_variable_gap_visible_level_before"
            if observable_gap
            else "complete_causal_abstraction_hidden_state_representation_invariant_closure"
            if fork == "WALL_IS_HIDDEN_STATE"
            else "complete_causal_abstraction_diagnostic_degenerate_retired"
        ),
        "flagged_adversarial": flagged,
        "fork_verdict": fork,
        "missing_observable_variable": lever if observable_gap else None,
        "minimal_abstraction_is_observable_subset": observable_gap,
        "is_decision_need_table_in_disguise": False,
        "positive_control_classifies_observable": True,
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "n_games_measured": 3,
        "per_game_causal_abstraction": {
            "cn04": {
                "classification": "OBSERVABLE_GAP" if observable_gap else "HIDDEN_STATE",
                "required_variables": ["visible_grid_hash", lever]
                if observable_gap
                else ["visible_grid_hash", "winning_prefix_order_state"],
                "observable_from_interface": {
                    "visible_grid_hash": True,
                    lever: True,
                    "winning_prefix_order_state": False,
                },
            }
        },
        "positive_control_games": ["tu93", "ar25"],
        "inference_substrate": "live_llm_inference",
        "duration_s": 60.001,
    }


def _b1(
    *,
    trustworthy: bool = True,
    fork: str = "WALL_IS_HIDDEN_STATE",
) -> JsonDict:
    return {
        "experiment": "experiment_4918_causal_abstraction_audit",
        "experiment_id": 4918,
        "honest_verdict": "complete_a1_causal_abstraction_audited",
        "flagged_adversarial": False,
        "a1_diagnostic_trustworthy": trustworthy,
        "a1_failure_reasons": [] if trustworthy else ["numbers_match_fork"],
        "checks": {
            "real_transitions": trustworthy,
            "not_value_table": trustworthy,
            "observable_claims_verified": trustworthy,
            "positive_control_observable": trustworthy,
            "oracle_distinct_planner_blind": trustworthy,
            "numbers_match_fork": trustworthy,
        },
        "numbers_match_fork_evidence": {
            "declared_fork_verdict": fork,
            "computed_fork_verdict": fork if trustworthy else "DIAGNOSTIC_DEGENERATE_RETIRED",
        },
        "not_value_table_evidence": {"classification_only": trustworthy},
        "observable_claims_spot_checked": [{"variable": "visible_grid_hash", "passed": trustworthy}],
        "positive_control_evidence": {"declared": trustworthy},
        "oracle_distinct_planner_blind_evidence": {
            "verifier_is_oracle": False,
            "planner_blind_to_banked_answer": trustworthy,
        },
        "transition_cross_checks": [{"game": "cn04", "passed": trustworthy}],
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "duration_s": 1.0,
    }


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4915_levelup_attempt",
        "honest_verdict": "success_cn04_levelup_banked",
        "new_levels_banked": 1,
        "offline_reproduced": True,
        "reproduced_levels": 3,
        "reproducible_total_levels_before": 68,
        "reproducible_total_levels_after": 69,
        "registry_update": {
            "prior_total_declared": 68,
            "new_total_declared": 69,
            "banked_levels": 1,
            "reason": "banked_offline_reproduced_level",
            "updated": True,
        },
        "reproduction_gate": {"claimed_level": 3, "reached_level": 3, "reproduced": True},
        "target_game": "cn04",
        "inference_substrate": "offline_arcade_reproduction_gate_no_llm",
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4916_self_play_verifier_checkpoint",
        "honest_verdict": "success_self_play_checkpoint_refreshed",
        "verifier_checkpoint_refreshed": True,
        "checkpoint_path": "models/arc_verifier_bp35.json",
        "checkpoint_mtime_delta_ns": 139466336536785,
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "search_state_count": 69,
        "target_game": "bp35",
        "inference_substrate": "live_llm_inference",
    }


def _heldout(*, flagged: bool = False) -> JsonDict:
    return {
        "experiment": "experiment_4917_heldout_first_win_readiness",
        "experiment_id": 4917,
        "honest_verdict": "complete: heldout_first_win_soft_budget_stop_partial_21_of_25_games_84_attempts_resume_to_finish",
        "flagged_adversarial": flagged,
        "heldout_first_win_rate": 0.047619,
        "heldout_first_win_ci": {"ci95": [0.0, 0.0]},
        "heldout_first_win_ci_lower": 0.0,
        "first_win_baseline": 0.04,
        "prior_best_heldout_first_win_rate": 0.0625,
        "heldout_first_win_delta_vs_baseline": 0.007619,
        "heldout_first_win_delta_vs_prior_best": -0.014881,
        "positive_control_passed": True,
        "parity_test_green": True,
        "live_agent_ran": True,
        "partial": True,
        "completed_games": ["ar25", "bp35", "cn04"],
        "remaining_games": ["tr87", "tu93"],
        "inference_substrate": "live_llm_inference",
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4919_submission_package_harden",
        "honest_verdict": "success_submission_package_ready_final_pre_deadline",
        "submission_package_ready": True,
        "submits": False,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "peak_vram_gb": 15.146,
        "frozen_stack_loads": True,
        "package_builds": True,
        "package_build_check": {"package_builds": True, "dry_build_ran": True},
        "ready_package_regression_ok": True,
        "ready_package_regression_check": {"ok": True},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _retro() -> JsonDict:
    return {
        "experiment": "experiment_4920_retro_timing_and_stamping_fix",
        "experiment_id": 4920,
        "honest_verdict": "success_retro_timing_mtime_fallback_and_stamping_shipped",
        "research_conductor_modified": False,
        "stamping_audit_missing_duration": [{"experiment_id": 4905}],
        "wiring_proposal_written": True,
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 4921,
        "honest_verdict": "success_kv260_continuity_ok",
        "kv260_ssh_reachable": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "xmutil_requires_sudo": True,
        "inference_substrate": "hardware_smoke",
    }


def _pivot() -> JsonDict:
    return {
        "honest_verdict": "success_distributional_energy_verifier_pivot_scaffolded",
        "pivot_executable_on_6_30": True,
        "no_headline_claim": True,
        "no_verifier_win_claimed": True,
        "comparison_stubbed": True,
        "self_consistency_saturated": False,
        "harness_skeleton_path": "python/carnot/experiment_4922_distributional_energy_verifier_scaffold.py",
        "domain_slice_path": "data/experiment_4922_travelplanner_structured_slice.jsonl",
        "dry_run_three_columns": {
            "columns": ["distributional_energy_verifier", "self_consistency", "llm_judge"],
            "n_rows": 3,
        },
        "validation_gate": {"oracle_distinct_required": True},
        "verifier_is_oracle": False,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _artifacts(
    *,
    a1: JsonDict | None = None,
    b1: JsonDict | None = None,
    heldout: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "A1": a1 or _a1(),
        "B1_AUDIT": b1 or _b1(),
        "A2_LEVELUP": _levelup(),
        "A3_SELF_PLAY": _self_play(),
        "A4_HELDOUT": heldout or _heldout(),
        "B2_PACKAGE": _package(),
        "B3_RETRO": _retro(),
        "C_KV260": _hardware(),
        "D_PIVOT": _pivot(),
    }


def _hashes() -> dict[str, str]:
    return {source: f"sha256:{source.lower()}" for source in mod.UPSTREAM_SOURCES}


def _summaries(
    *,
    a4_code: int = 0,
    a1_code: int = 0,
) -> dict[str, mod.SummarizerResult]:
    return {
        source: mod.SummarizerResult(
            ["summarize", spec.relative_path],
            a4_code if source == "A4_HELDOUT" else a1_code if source == "A1" else 0,
            "LIVE re-check: CRITICAL" if source == "A4_HELDOUT" and a4_code >= 2 else "clean",
            "",
        )
        for source, spec in mod.UPSTREAM_SOURCES.items()
    }


def test_req_capstone_4923_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4923: OpenSpec declares the .453 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-4923") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH in section
    for source in mod.UPSTREAM_SOURCES.values():
        assert source.relative_path in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_capstone_4923_default_hidden_state_arc_closure() -> None:
    """SCENARIO-CAPSTONE-4923: B1-trusted hidden-state verdict closes representation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == (
        "complete_capstone_v453_wall_is_hidden_state_arc_closure"
    )
    assert "representation-invariant by construction" in artifact["headline"]
    assert "Do not queue representation #5" in artifact["headline"]
    assert artifact["a1_closure_verdict_trusted"]["trusted"] is True
    assert artifact["a1_closure_verdict_trusted"]["closure_verdict"] == "WALL_IS_HIDDEN_STATE"
    assert artifact["a1_closure_verdict_trusted"]["trust_gate"] == {
        "a1_diagnostic_trustworthy": True,
        "real_transitions": True,
        "not_value_table": True,
        "observable_claims_verified": True,
        "positive_control_observable": True,
        "oracle_distinct_planner_blind": True,
        "numbers_match_fork": True,
    }
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["heldout_first_win_rate"] == 0.047619
    assert artifact["submission_package_ready"] is True
    assert artifact["skipped_flagged_adversarial"] == []

    pivot = artifact["post_sprint_pivot"]
    assert pivot["decision"] == "post_6_30_distributional_energy_verifier_pivot"
    assert pivot["paper_ready"] is True
    assert pivot["do_not_queue"] == "representation_5"
    assert "~0.05" in pivot["deliverable"]

    scorecard = artifact["milestone_scorecard"]
    assert scorecard["a2_levelup_bank"]["decision"] == "new_level_banked"
    assert scorecard["a2_levelup_bank"]["registry_authoritative_total"] == 69
    assert scorecard["a3_self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert scorecard["a4_heldout_go_no_go"]["flag_resolved"] is True
    assert scorecard["b2_submission_package"]["decision"] == "package_ready_operator_only"
    assert scorecard["b3_retro_timing_stamping_fix"]["decision"] == "retro_fix_shipped"
    assert scorecard["c_kv260"]["decision"] == "kv260_continuity_ok"
    assert scorecard["d_distributional_energy_verifier_pivot"]["decision"] == (
        "pivot_scaffold_executable"
    )
    assert {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]} == {
        4914,
        4918,
        4915,
        4916,
        4917,
        4919,
        4920,
        4921,
        4922,
    }
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4923_alternate_headlines_and_flagged_skip() -> None:
    """SCENARIO-CAPSTONE-4923: headline gates fallback and observable-gap paths."""

    observable = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(fork="WALL_IS_OBSERVABLE_VARIABLE_GAP"),
            b1=_b1(fork="WALL_IS_OBSERVABLE_VARIABLE_GAP"),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert observable["honest_verdict"] == (
        "complete_capstone_v453_observable_variable_gap_visible_level_before"
    )
    assert observable["a1_closure_verdict_trusted"]["fixable_observable_lever"] == (
        "visible_level_before"
    )
    assert ".454" in observable["headline"]
    assert observable["post_sprint_pivot"]["decision"] == "v454_observable_variable_candidate"

    untrusted = mod.build_artifact(
        artifacts=_artifacts(b1=_b1(trustworthy=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert untrusted["honest_verdict"] == (
        "complete_capstone_v453_diagnostic_inconclusive_escalate"
    )
    assert untrusted["a1_closure_verdict_trusted"]["trusted"] is False
    assert "B1 did not trust A1" in untrusted["headline"]
    assert "numbers_match_fork" in untrusted["a1_closure_verdict_trusted"][
        "trust_failure_reasons"
    ]

    degenerate = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(fork="DIAGNOSTIC_DEGENERATE_RETIRED"),
            b1=_b1(fork="DIAGNOSTIC_DEGENERATE_RETIRED"),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert degenerate["honest_verdict"] == (
        "complete_capstone_v453_diagnostic_inconclusive_escalate"
    )
    assert "diagnostic was degenerate" in degenerate["headline"]

    flagged = mod.build_artifact(
        artifacts=_artifacts(heldout=_heldout(flagged=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(a4_code=2),
        duration_s=0.001,
    )
    assert flagged["heldout_first_win_rate"] is None
    assert flagged["milestone_scorecard"]["a4_heldout_go_no_go"]["flag_resolved"] is False
    assert flagged["skipped_flagged_adversarial"] == [
        {
            "source": "A4_HELDOUT",
            "experiment_id": 4917,
            "path": mod.UPSTREAM_SOURCES["A4_HELDOUT"].relative_path,
            "reason": "flagged_adversarial",
            "sha256": "sha256:a4_heldout",
            "summarizer_exit_code": 2,
            "true_honest_verdict": _heldout()["honest_verdict"],
            "true_live_recheck": "critical",
        }
    ]
    assert 4917 not in {row["experiment_id"] for row in flagged["cited_upstream_artifacts"]}


def test_run_capstone_invokes_summarizer_and_blocks_missing_required(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4923-BLOCKED-PRECONDITION: runtime checks fail closed."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    for key, source in mod.AUXILIARY_UPSTREAM_SOURCES.items():
        _write_json(tmp_path / source.relative_path, {"games": {}, "source": key})
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 69\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4923\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")
    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        return mod.SummarizerResult(["summarize", relative_path], 0, "clean", "")

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)
    assert called == [spec.relative_path for spec in mod.UPSTREAM_SOURCES.values()] + [
        spec.relative_path for spec in mod.AUXILIARY_UPSTREAM_SOURCES.values()
    ]
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["capstone_ready"] is True
    assert artifact["preconditions_checked"]["auxiliary_upstream_artifacts"][
        "A4_HELDOUT_PARTIAL"
    ]["summarizer_exit_code"] == 0
    assert mod.validate_artifact(artifact) == []

    missing_root = tmp_path / "missing"
    for key, payload in _artifacts().items():
        if key != "B1_AUDIT":
            _write_json(missing_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    missing_registry = missing_root / mod.REGISTRY_RELATIVE_PATH
    missing_registry.parent.mkdir(parents=True, exist_ok=True)
    missing_registry.write_text(
        "schema_version: 1\nreproducible_total_levels: 69\n", encoding="utf-8"
    )
    missing_spec = missing_root / mod.SPEC_RELATIVE_PATH
    missing_spec.parent.mkdir(parents=True, exist_ok=True)
    missing_spec.write_text("REQ-CAPSTONE-4923\n", encoding="utf-8")
    missing_summarizer = missing_root / mod.SUMMARIZER_RELATIVE_PATH
    missing_summarizer.parent.mkdir(parents=True, exist_ok=True)
    missing_summarizer.write_text("# placeholder\n", encoding="utf-8")

    blocked = mod.run_capstone(root=missing_root, summarizer=summarizer)
    assert blocked["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert blocked["a1_closure_verdict_trusted"]["trusted"] is False
    assert blocked["milestone_scorecard"] == {}
    assert blocked["preconditions_checked"]["upstream_artifacts"]["B1_AUDIT"]["present"] is False
    assert mod.validate_artifact(blocked) == []

    bad_registry_root = tmp_path / "bad_registry"
    for key, payload in _artifacts().items():
        _write_json(bad_registry_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    bad_registry = bad_registry_root / mod.REGISTRY_RELATIVE_PATH
    bad_registry.parent.mkdir(parents=True, exist_ok=True)
    bad_registry.write_text("schema_version: [unterminated\n", encoding="utf-8")
    bad_spec = bad_registry_root / mod.SPEC_RELATIVE_PATH
    bad_spec.parent.mkdir(parents=True, exist_ok=True)
    bad_spec.write_text("REQ-CAPSTONE-4923\n", encoding="utf-8")
    bad_summarizer = bad_registry_root / mod.SUMMARIZER_RELATIVE_PATH
    bad_summarizer.parent.mkdir(parents=True, exist_ok=True)
    bad_summarizer.write_text("# placeholder\n", encoding="utf-8")

    registry_blocked = mod.run_capstone(root=bad_registry_root, summarizer=summarizer)
    assert registry_blocked["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert registry_blocked["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(registry_blocked) == []


def test_validate_artifact_rejects_schema_errors_and_helpers() -> None:
    """SCENARIO-CAPSTONE-4923-FIELD-PRINCIPLES: malformed scorecards fail validation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "maybe"}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "invalid_a1_closure_verdict_trusted" in mod.validate_artifact(
        {**artifact, "a1_closure_verdict_trusted": []}
    )
    assert "invalid_headline" in mod.validate_artifact({**artifact, "headline": []})
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "69"}
    )
    assert "invalid_heldout_first_win_rate" in mod.validate_artifact(
        {**artifact, "heldout_first_win_rate": "0.047619"}
    )
    assert "invalid_submission_package_ready" in mod.validate_artifact(
        {**artifact, "submission_package_ready": "yes"}
    )
    assert "invalid_post_sprint_pivot" in mod.validate_artifact(
        {**artifact, "post_sprint_pivot": []}
    )
    assert "invalid_preconditions_checked" in mod.validate_artifact(
        {**artifact, "preconditions_checked": []}
    )
    assert "invalid_capstone_ready" in mod.validate_artifact(
        {**artifact, "capstone_ready": "true"}
    )
    assert "invalid_milestone_scorecard" in mod.validate_artifact(
        {**artifact, "milestone_scorecard": []}
    )
    assert "invalid_skipped_flagged_adversarial" in mod.validate_artifact(
        {**artifact, "skipped_flagged_adversarial": [{"experiment_id": 4917}]}
    )
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4914}]}
    )
    assert "invalid_random_seed" in mod.validate_artifact({**artifact, "random_seed": "seed"})
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert "missing_principle:honest_verdict" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert mod._experiment_id("A1", {"experiment_id": True}) == 4914
    assert mod._experiment_id("A2_LEVELUP", {"experiment": 4915}) == 4915
    assert mod._experiment_id("B1_AUDIT", {"experiment": "experiment_4918"}) == 4918
    assert mod._live_recheck(None) == "not_run"
    assert mod._live_recheck(mod.SummarizerResult([], 1, "", "")) == "warn"
    assert mod._is_skipped(_a1(flagged=True), None) == "flagged_adversarial"
    assert mod._is_skipped(_a1(), mod.SummarizerResult([], 2, "", "")) == (
        "live_critical_recheck"
    )
    assert mod._hidden_variables({"per_game_causal_abstraction": {"bad_row": []}}) == []
    assert mod._observable_gap_lever(None) == "unknown_observable_variable"
    observable_without_direct_lever = _a1(fork="WALL_IS_OBSERVABLE_VARIABLE_GAP")
    observable_without_direct_lever.pop("missing_observable_variable")
    assert mod._observable_gap_lever(observable_without_direct_lever) == "visible_grid_hash"
    assert (
        mod._observable_gap_lever(
            {"per_game_causal_abstraction": {"cn04": {"classification": "HIDDEN_STATE"}}}
        )
        == "unknown_observable_variable"
    )
    assert (
        mod._observable_gap_lever(
            {
                "per_game_causal_abstraction": {
                    "cn04": {
                        "classification": "OBSERVABLE_GAP",
                        "required_variables": ["winning_prefix_order_state"],
                        "observable_from_interface": {"winning_prefix_order_state": False},
                    }
                }
            }
        )
        == "unknown_observable_variable"
    )
    assert mod._a1_closure_verdict_trusted(None, _b1())["trusted"] is False
    assert mod._a2_levelup_bank(None, 69) == {}
    assert mod._a3_self_play_checkpoint(None) == {}
    assert mod._a4_heldout_go_no_go({}, {}, {}) == {}
    assert mod._b2_submission_package(None) == {}
    assert mod._b3_retro_timing_stamping_fix(None) == {}
    assert mod._c_kv260(None) == {}
    assert mod._d_distributional_energy_verifier_pivot(None) == {}
    assert mod._b1_audit(None) == {}
    assert mod._first_blocker(
        summarizer_present=False,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=True,
        upstream_preconditions={},
    ) == "missing_summarizer"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=False,
        registry_loadable=True,
        spec_has_req=True,
        upstream_preconditions={},
    ) == "missing_registry"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=True,
        registry_loadable=False,
        spec_has_req=True,
        upstream_preconditions={},
    ) == "registry_not_yaml_loadable"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=False,
        upstream_preconditions={},
    ) == "spec_missing_req_4923"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=True,
        upstream_preconditions={"A1": {"present": False}, "B1_AUDIT": {"present": True}},
    ) == "upstream_artifact_missing"
