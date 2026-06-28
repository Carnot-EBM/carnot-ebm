"""Tests for REQ-CAPSTONE-4912 / SCENARIO-CAPSTONE-4912."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4912_capstone_v452 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1(
    *,
    fork: str = "WALL_DEEPER_THAN_VALUE_PREDICTION",
    flagged: bool = False,
) -> JsonDict:
    lifted = fork == "ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN"
    budget = fork == "SEARCH_BUDGET_BOUND"
    return {
        "experiment_id": 4903,
        "honest_verdict": (
            "success_env_grounded_search_first_win_unlocked_0.120000"
            if lifted
            else f"complete_env_grounded_search_no_first_win_lift_{fork}"
        ),
        "flagged_adversarial": flagged,
        "fork_verdict": fork,
        "value_grounded_first_win_delta_median": 0.12 if lifted else 0.02 if budget else -0.04,
        "value_grounded_first_win_delta_ci95": [0.1, 0.14]
        if lifted
        else [0.0, 0.04]
        if budget
        else [-0.04, -0.04],
        "median_actions_to_first_win": 120.0 if budget else 12.0 if lifted else None,
        "coverage_migration_count": 3 if lifted else 1 if budget else 0,
        "positive_control_non_degenerate": True,
        "planner_blind_to_banked_answer": True,
        "change_location_prior_used_not_value": True,
        "inference_substrate": "live_llm_inference",
        "duration_s": 60.0,
    }


def _a1b(*, flagged: bool = False) -> JsonDict:
    return {
        "experiment_id": 4904,
        "honest_verdict": "complete_latent_action_no_value_lift_representation_invariant_4_classes",
        "flagged_adversarial": flagged,
        "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES",
        "latent_action_value_accuracy_delta_median": -0.103162,
        "latent_action_value_accuracy_delta_ci95": [-0.231195, 0.025266],
        "ran_genuinely_live": True,
        "delta_on_truly_heldout_split": True,
        "inference_substrate": "live_llm_inference",
        "duration_s": 178.750196,
    }


def _b1(
    *,
    trusted: bool = True,
    a1b_live: bool = True,
    a1b_gate_skipped: bool = False,
    a1_fork: str = "WALL_DEEPER_THAN_VALUE_PREDICTION",
) -> JsonDict:
    return {
        "experiment": "experiment_4908_env_grounded_search_audit",
        "experiment_id": 4908,
        "honest_verdict": "complete_a1_a1b_audited",
        "flagged_adversarial": False,
        "a1_source_fork_verdict": a1_fork,
        "a1_trustworthy": trusted,
        "a1_value_from_real_env": trusted,
        "a1_planner_blind": trusted,
        "a1_positive_control_non_degenerate": trusted,
        "a1_numbers_match_fork": trusted,
        "a1b_ran_genuinely_live": a1b_live,
        "a1b_gate_skipped": a1b_gate_skipped,
        "a1b_source_fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES",
        "a1_failure_reasons": [] if trusted else ["a1_not_trustworthy"],
        "a1b_failure_reasons": [] if a1b_live or a1b_gate_skipped else ["a1b_not_live"],
        "adversarial_flags_found": False,
        "checks": {
            "a1_value_from_real_env": {"passed": trusted},
            "a1_planner_blind": {"passed": trusted},
            "a1_positive_control_non_degenerate": {"passed": trusted},
            "a1_numbers_match_fork": {
                "passed": trusted,
                "computed_fork_verdict": a1_fork,
            },
            "a1b_live_or_gate_skipped": {
                "passed": a1b_live or a1b_gate_skipped,
                "ran_genuinely_live": a1b_live,
                "status": "skipped" if a1b_gate_skipped else "ran",
            },
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4905_levelup_attempt",
        "honest_verdict": "complete_m0r0_no_new_level_residual_duplicate_depth",
        "target_game": "m0r0",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "registry_update": {
            "prior_total_declared": 68,
            "new_total_declared": 68,
            "reason": "duplicate_depth",
            "updated": True,
        },
        "reproduction_gate": {"claimed_level": 2, "reached_level": 2, "reproduced": True},
        "inference_substrate": "offline_arcade_reproduction_gate_no_llm",
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4906_self_play_verifier_checkpoint",
        "honest_verdict": "success_self_play_checkpoint_refreshed",
        "target_game": "vc33",
        "verifier_checkpoint_refreshed": True,
        "checkpoint_path": "models/arc_verifier_vc33.json",
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "search_state_count": 10,
        "inference_substrate": "live_llm_inference",
    }


def _heldout(*, flagged: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_4907_heldout_first_win_readiness",
        "experiment_id": 4907,
        "honest_verdict": "complete_heldout_first_win_0.05_ci_lower_0_soft_budget_partial_live",
        "flagged_adversarial": flagged,
        "heldout_first_win_rate": 0.05,
        "first_win_baseline": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.01,
        "heldout_first_win_delta_vs_prior_best": -0.0125,
        "positive_control_passed": True,
        "parity_test_green": True,
        "live_agent_ran": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "inference_substrate": "live_llm_inference",
        "duration_s": 3518.208351,
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4909_submission_package_harden",
        "honest_verdict": "success_submission_package_ready_final_pre_deadline",
        "submission_package_ready": True,
        "submits": False,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "peak_vram_gb": 15.146,
        "frozen_stack_loads": True,
        "package_builds": {"package_builds": True, "dry_build_ran": True},
        "agent_config_resolution": {"resolved": True},
        "model_path_resolution": {"resolved": True},
        "packaging_requirements_crosscheck": {"ok": True},
        "ready_package_regression_check": {"ok": True},
        "inference_substrate": "live_llm_inference",
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 4910,
        "honest_verdict": "success_kv260_continuity_ok",
        "kv260_ssh_reachable": True,
        "loaded_overlay": False,
        "preconditions_checked": [{"resource": "kv260_ssh", "available": True}],
        "inference_substrate": "hardware_smoke",
        "duration_s": 6.7759,
    }


def _handoff() -> JsonDict:
    return {
        "honest_verdict": "success_sota_ingestion_v453_frontier_mapped",
        "aimed_at_fork_verdict": "WALL_DEEPER_THAN_VALUE_PREDICTION",
        "a1b_fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES",
        "selected_branch": "wall_survives_four_representations_plus_env_grounding",
        "flagged_for_v453": [{"candidate": "causal_state_abstraction_wall_diagnostic"}],
        "post_sprint_pivot_methods": [
            {
                "method": "Distributional energy verifier for structured reasoning",
                "track": "distributional_energy_verifier",
                "oracle_distinct_verifier": True,
            }
        ],
        "sota_to_experiment_mapping_note": {
            "summary": "post-sprint verifier-moat pivot",
            "planner_instruction": "start the verifier-moat pivot",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 0.0001,
    }


def _artifacts(
    *,
    a1: JsonDict | None = None,
    a1b: JsonDict | None = None,
    b1: JsonDict | None = None,
    heldout: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "A1": a1 or _a1(),
        "A1B": a1b or _a1b(),
        "B1_AUDIT": b1 or _b1(),
        "A2_LEVELUP": _levelup(),
        "A3_SELF_PLAY": _self_play(),
        "A4_HELDOUT": heldout or _heldout(),
        "B2_PACKAGE": _package(),
        "C_HARDWARE": _hardware(),
        "D_HANDOFF": _handoff(),
    }


def _hashes() -> dict[str, str]:
    return {source: f"sha256:{source.lower()}" for source in mod.UPSTREAM_SOURCES}


def _summaries(
    *,
    heldout_code: int = 2,
    a1b_code: int = 0,
) -> dict[str, mod.SummarizerResult]:
    return {
        source: mod.SummarizerResult(
            ["summarize", spec.relative_path],
            heldout_code if source == "A4_HELDOUT" else a1b_code if source == "A1B" else 0,
            "LIVE re-check: CRITICAL" if source == "A4_HELDOUT" and heldout_code >= 2 else "clean",
            "",
        )
        for source, spec in mod.UPSTREAM_SOURCES.items()
    }


def test_req_capstone_4912_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4912: OpenSpec declares the .452 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-4912") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH in section
    for source in mod.UPSTREAM_SOURCES.values():
        assert source.relative_path in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_capstone_4912_default_escalates_wall_survival() -> None:
    """SCENARIO-CAPSTONE-4912: trusted wall plus A1b null becomes escalation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == (
        "complete_capstone_v452_escalate_wall_survives_four_representations_plus_env_grounding"
    )
    assert "FOUR world-model representations" in artifact["headline"]
    assert "Do not queue representation #5" in artifact["headline"]
    assert artifact["a1_fork_verdict_trusted"]["trusted"] is True
    assert artifact["a1_fork_verdict_trusted"]["fork_verdict"] == (
        "WALL_DEEPER_THAN_VALUE_PREDICTION"
    )
    assert artifact["a1_fork_verdict_trusted"]["trust_gate"] == {
        "a1_trustworthy": True,
        "a1_value_from_real_env": True,
        "a1_planner_blind": True,
        "a1_positive_control_non_degenerate": True,
        "a1_numbers_match_fork": True,
        "a1b_ran_genuinely_live": True,
        "a1b_gate_skipped": False,
        "a1b_live_or_gate_skipped": True,
    }
    assert artifact["reproducible_total_levels"] == 68
    assert artifact["heldout_first_win_rate"] is None
    assert artifact["submission_package_ready"] is True

    pivot = artifact["post_sprint_pivot"]
    assert pivot["decision"] == "post_6_30_verifier_moat_pivot"
    assert pivot["paper_ready"] is True
    assert pivot["do_not_queue"] == "representation_5"
    assert "~0.05" in pivot["deliverable"]

    scorecard = artifact["milestone_scorecard"]
    assert scorecard["a2_levelup_bank"]["new_levels_banked"] == 0
    assert scorecard["a2_levelup_bank"]["decision"] == "no_new_level_banked"
    assert scorecard["a3_self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert scorecard["a4_fresh_live_heldout"] == {
        "source": "A4_HELDOUT",
        "experiment_id": 4907,
        "status": "skipped_flagged_adversarial",
        "reason": "flagged_adversarial",
        "true_honest_verdict": _heldout()["honest_verdict"],
        "stale_false_flag": False,
    }
    assert scorecard["b2_submission_package"]["decision"] == "package_ready_operator_only"
    assert scorecard["c_hardware"]["decision"] == "kv260_continuity_ok"
    assert scorecard["d_v453_handoff"]["selected_branch"] == (
        "wall_survives_four_representations_plus_env_grounding"
    )
    assert artifact["skipped_flagged_adversarial"] == [
        {
            "source": "A4_HELDOUT",
            "experiment_id": 4907,
            "path": mod.UPSTREAM_SOURCES["A4_HELDOUT"].relative_path,
            "reason": "flagged_adversarial",
            "sha256": "sha256:a4_heldout",
            "summarizer_exit_code": 2,
            "true_honest_verdict": _heldout()["honest_verdict"],
            "stale_false_flag": False,
            "true_live_recheck": "critical",
        }
    ]
    assert 4907 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4912_alternate_headlines_and_stale_false_flag() -> None:
    """SCENARIO-CAPSTONE-4912: A1's trusted fork chooses the non-escalate headlines."""

    lifted = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(fork="ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN"),
            b1=_b1(
                a1_fork="ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN",
                a1b_live=False,
                a1b_gate_skipped=True,
            ),
            heldout=_heldout(flagged=False),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(heldout_code=0),
        duration_s=0.001,
    )
    assert lifted["honest_verdict"] == "complete_capstone_v452_env_grounded_search_unlocked"
    assert "unlocked first-wins" in lifted["headline"]
    assert lifted["heldout_first_win_rate"] == 0.05
    assert lifted["post_sprint_pivot"]["decision"] == "not_escalated"

    budget = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(fork="SEARCH_BUDGET_BOUND"),
            b1=_b1(a1_fork="SEARCH_BUDGET_BOUND"),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert budget["honest_verdict"] == "complete_capstone_v452_search_budget_bound"
    assert "too high an action cost" in budget["headline"]

    untrusted = mod.build_artifact(
        artifacts=_artifacts(b1=_b1(trusted=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert untrusted["honest_verdict"] == "complete_capstone_v452_untrusted_a1_fork_non_test"
    assert untrusted["a1_fork_verdict_trusted"]["trusted"] is False
    assert "a1_trustworthy" in untrusted["a1_fork_verdict_trusted"]["trust_failure_reasons"]

    stale_false_flag = mod.build_artifact(
        artifacts=_artifacts(heldout=_heldout(flagged=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(heldout_code=0),
        duration_s=0.001,
    )
    skipped = stale_false_flag["skipped_flagged_adversarial"][0]
    assert skipped["stale_false_flag"] is True
    assert skipped["true_live_recheck"] == "clean"
    assert skipped["true_honest_verdict"] == _heldout()["honest_verdict"]
    assert stale_false_flag["heldout_first_win_rate"] is None


def test_run_capstone_invokes_summarizer_and_blocks_missing_required(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4912-BLOCKED-PRECONDITION: runtime checks fail closed."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 68\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4912\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")
    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        code = 2 if relative_path == mod.UPSTREAM_SOURCES["A4_HELDOUT"].relative_path else 0
        return mod.SummarizerResult(["summarize", relative_path], code, "clean", "")

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)
    assert called == [spec.relative_path for spec in mod.UPSTREAM_SOURCES.values()]
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["capstone_ready"] is True
    assert mod.validate_artifact(artifact) == []

    missing_root = tmp_path / "missing"
    for key, payload in _artifacts().items():
        if key != "B1_AUDIT":
            _write_json(missing_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    missing_registry = missing_root / mod.REGISTRY_RELATIVE_PATH
    missing_registry.parent.mkdir(parents=True, exist_ok=True)
    missing_registry.write_text(
        "schema_version: 1\nreproducible_total_levels: 68\n", encoding="utf-8"
    )
    missing_spec = missing_root / mod.SPEC_RELATIVE_PATH
    missing_spec.parent.mkdir(parents=True, exist_ok=True)
    missing_spec.write_text("REQ-CAPSTONE-4912\n", encoding="utf-8")
    missing_summarizer = missing_root / mod.SUMMARIZER_RELATIVE_PATH
    missing_summarizer.parent.mkdir(parents=True, exist_ok=True)
    missing_summarizer.write_text("# placeholder\n", encoding="utf-8")

    blocked = mod.run_capstone(root=missing_root, summarizer=summarizer)
    assert blocked["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert blocked["a1_fork_verdict_trusted"]["trusted"] is False
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
    bad_spec.write_text("REQ-CAPSTONE-4912\n", encoding="utf-8")
    bad_summarizer = bad_registry_root / mod.SUMMARIZER_RELATIVE_PATH
    bad_summarizer.parent.mkdir(parents=True, exist_ok=True)
    bad_summarizer.write_text("# placeholder\n", encoding="utf-8")

    registry_blocked = mod.run_capstone(root=bad_registry_root, summarizer=summarizer)
    assert registry_blocked["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert registry_blocked["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(registry_blocked) == []


def test_validate_artifact_rejects_schema_errors_and_helpers() -> None:
    """SCENARIO-CAPSTONE-4912-FIELD-PRINCIPLES: malformed scorecards fail validation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
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
    assert "invalid_a1_fork_verdict_trusted" in mod.validate_artifact(
        {**artifact, "a1_fork_verdict_trusted": []}
    )
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "68"}
    )
    assert "invalid_submission_package_ready" in mod.validate_artifact(
        {**artifact, "submission_package_ready": "yes"}
    )
    assert "invalid_heldout_first_win_rate" in mod.validate_artifact(
        {**artifact, "heldout_first_win_rate": "0.05"}
    )
    assert "missing_principle:honest_verdict" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert "invalid_milestone_scorecard" in mod.validate_artifact(
        {**artifact, "milestone_scorecard": []}
    )
    assert "invalid_skipped_flagged_adversarial" in mod.validate_artifact(
        {**artifact, "skipped_flagged_adversarial": [{"experiment_id": 4907}]}
    )
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4903}]}
    )
    assert "invalid_post_sprint_pivot" in mod.validate_artifact(
        {**artifact, "post_sprint_pivot": []}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert mod._experiment_id("A1", {"experiment_id": True}) == 4903
    assert mod._experiment_id("A2_LEVELUP", {"experiment": 4905}) == 4905
    assert mod._experiment_id("B1_AUDIT", {"experiment": "experiment_4908"}) == 4908
    assert mod._is_skipped(_a1(flagged=True), None) == "flagged_adversarial"
    assert mod._is_skipped(_a1(), mod.SummarizerResult([], 2, "", "")) == (
        "live_critical_recheck"
    )
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
    ) == "spec_missing_req_4912"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=True,
        upstream_preconditions={"A1": {"present": False}, "B1_AUDIT": {"present": True}},
    ) == "upstream_artifact_missing"
