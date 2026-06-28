"""Tests for REQ-CAPSTONE-4934 / SCENARIO-CAPSTONE-4934."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4934_capstone_v454 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prev_capstone() -> JsonDict:
    return {
        "experiment_id": 4923,
        "honest_verdict": "complete_capstone_v453_wall_is_hidden_state_arc_closure",
        "a1_closure_verdict_trusted": {
            "closure_verdict": "WALL_IS_HIDDEN_STATE",
            "trusted": True,
            "trust_failure_reasons": [],
        },
        "post_sprint_pivot": {
            "decision": "post_6_30_distributional_energy_verifier_pivot",
            "deliverable": (
                "current ~0.05 first-win agent (operator-only package) + publishable "
                "FoVer verifier-ensemble paper"
            ),
            "do_not_queue": "representation_5",
        },
        "reproducible_total_levels": 69,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
    }


def _levelup(game: str) -> JsonDict:
    return {
        "honest_verdict": f"complete_{game}_no_new_level_residual_duplicate_depth",
        "target_game": game,
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "solve_provenance": "live_agent_self_discovery",
        "live_path_reachable": True,
        "reproduction_gate": {
            "game": game,
            "claimed_level": 2,
            "reached_level": 2,
            "reproduced": True,
        },
        "registry_update": {
            "prior_total_declared": 69,
            "new_total_declared": 69,
            "banked_levels": 0,
            "reason": "duplicate_depth",
            "target_game": game,
        },
        "inference_substrate": "offline_arcade_reproduction_gate_no_llm",
    }


def _self_play() -> JsonDict:
    return {
        "honest_verdict": "success_self_play_checkpoint_refreshed",
        "target_game": "lf52",
        "checkpoint_path": "models/arc_verifier_lf52.json",
        "checkpoint_mtime_delta_ns": 195669605570217,
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "inference_substrate": "live_llm_inference",
    }


def _heldout() -> JsonDict:
    return {
        "experiment_id": 4928,
        "honest_verdict": "complete_heldout_first_win_0.04_full25_live_flag_resolved",
        "flagged_adversarial": False,
        "heldout_first_win_rate": 0.04,
        "heldout_first_win_ci": {
            "method": "paired_percentile_bootstrap",
            "ci95": [0.0, 0.0],
            "point": 0.0,
        },
        "games_evaluated": 25,
        "games_remaining": 0,
        "flag_resolved": True,
        "live_agent_ran": True,
        "positive_control_passed": True,
        "parity_test_green": True,
        "partial": False,
        "generator_backend": "gpu0_cuda",
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
    }


def _audit(*, banks_trustworthy: bool = False, efficiency_trustworthy: bool = False) -> JsonDict:
    return {
        "experiment_id": 4929,
        "honest_verdict": "blocked_experiment_4933_matm_similarity_retrieval_efficiency_missing",
        "banks_trustworthy": banks_trustworthy,
        "efficiency_trustworthy": efficiency_trustworthy,
        "checks": {
            "reproduction_genuine": True,
            "not_duplicate": banks_trustworthy,
            "self_discovery_provenance": True,
            "live_path_reachable": True,
            "oracle_distinct": efficiency_trustworthy,
            "honest_ab": efficiency_trustworthy,
        },
        "audit_failure_reasons": [
            "A1_not_duplicate_failed_duplicate_depth_sp80_L2",
            "A2_not_duplicate_failed_duplicate_depth_su15_L2",
            "D_missing_experiment_4933_matm_similarity_retrieval_efficiency",
        ]
        if not banks_trustworthy and not efficiency_trustworthy
        else [],
        "bank_evidence": {
            "A1": {"game": "sp80", "claimed_reached_level": 2},
            "A2": {"game": "su15", "claimed_reached_level": 2},
        },
        "efficiency_evidence": {
            "present": efficiency_trustworthy,
            "checks": {"oracle_distinct": efficiency_trustworthy, "honest_ab": efficiency_trustworthy},
        },
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _package() -> JsonDict:
    return {
        "honest_verdict": "success_submission_package_ready_final_pre_deadline",
        "submission_package_ready": True,
        "submits": False,
        "operator_only": True,
        "peak_vram_gb": 15.146,
        "frozen_stack_loads": True,
        "package_builds": True,
        "ready_package_regression_ok": True,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
    }


def _stamping() -> JsonDict:
    return {
        "experiment_id": 4931,
        "honest_verdict": "blocked_insufficient_v454_mtime_window",
        "milestone": "2026.06.454",
        "stamping_backfilled_arms": [{"experiment_id": 4925}, {"experiment_id": 4930}],
        "mtime_fallback_window": {"n_arms": 7, "compute_bound_count": 3, "passed": False},
        "wiring_proposal_reconfirmed": True,
        "research_conductor_modified": False,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
    }


def _kv260() -> JsonDict:
    return {
        "honest_verdict": "success_kv260_continuity_ok",
        "kv260_ssh_reachable": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "xmutil_requires_sudo": True,
        "verifier_is_oracle": False,
        "inference_substrate": "hardware_smoke",
    }


def _d_efficiency() -> JsonDict:
    return {
        "honest_verdict": "complete_matm_similarity_retrieval_no_efficiency_gain_retired",
        "actions_to_first_levelup_delta": {
            "tu93": 0,
            "lp85": 0,
            "sp80": 0,
            "cn04": 0,
            "m0r0": 0,
        },
        "forward_walk_hit_rate_delta": {
            "tu93": 0.0,
            "lp85": 0.0,
            "sp80": 0.0,
            "cn04": 0.0,
            "m0r0": 0.0,
        },
        "reached_level_regression": False,
        "submitted_parity_test_green": True,
        "retire_if_same_verdict": True,
        "post_sprint_pivot_gate_noted": {
            "noted": True,
            "arxiv_id": "2605.18871",
            "validation_gate": (
                "distributional-energy-verifier beats self-consistency with CI95 excluding zero"
            ),
        },
        "arxiv_ids_cited": ["2605.18871"],
        "verifier_is_oracle": False,
        "inference_substrate": "honest_replay_scorecard_substrate",
    }


def _artifacts(
    *, banks_trustworthy: bool = False, efficiency_trustworthy: bool = False
) -> dict[str, JsonDict]:
    return {
        "PREV_CAPSTONE": _prev_capstone(),
        "A1_LEVELUP": _levelup("sp80"),
        "A2_LEVELUP": _levelup("su15"),
        "A3_SELF_PLAY": _self_play(),
        "A4_HELDOUT": _heldout(),
        "B1_AUDIT": _audit(
            banks_trustworthy=banks_trustworthy,
            efficiency_trustworthy=efficiency_trustworthy,
        ),
        "B2_PACKAGE": _package(),
        "B3_STAMPING": _stamping(),
        "C_KV260": _kv260(),
        "D_EFFICIENCY": _d_efficiency(),
    }


def _hashes() -> dict[str, str]:
    return {source: f"sha256:{source.lower()}" for source in mod.UPSTREAM_SOURCES}


def _summaries(*, self_play_code: int = 2, heldout_code: int = 1) -> dict[str, mod.SummarizerResult]:
    return {
        source: mod.SummarizerResult(
            ["summarize", spec.relative_path],
            self_play_code if source == "A3_SELF_PLAY" else heldout_code if source == "A4_HELDOUT" else 0,
            "LIVE re-check: CRITICAL" if source == "A3_SELF_PLAY" and self_play_code >= 2 else "clean",
            "",
        )
        for source, spec in mod.UPSTREAM_SOURCES.items()
    }


def test_req_capstone_4934_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4934: OpenSpec declares the .454 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-4934") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH in section
    for source in mod.UPSTREAM_SOURCES.values():
        assert source.relative_path in section
    for field in mod.REQUIRED_FIELDS:
        assert field in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_capstone_4934_default_submission_readiness_scorecard() -> None:
    """SCENARIO-CAPSTONE-4934: default read maximizes submission without inflating claims."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == (
        "complete_capstone_v454_submission_maximized_levels_69_heldout_0.04_package_ready_efficiency_null"
    )
    assert artifact["capstone_ready"] is True
    assert "69 reproducible levels" in artifact["headline"]
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["banks_counted"]["counted"] == []
    assert artifact["banks_counted"]["b1_banks_trustworthy"] is False
    assert artifact["banks_counted"]["candidate_banks"] == [
        {"source": "A1_LEVELUP", "game": "sp80", "new_levels_banked": 0},
        {"source": "A2_LEVELUP", "game": "su15", "new_levels_banked": 0},
    ]
    assert artifact["action_efficiency_result"]["decision"] == "honest_null_not_trusted_lift"
    assert artifact["action_efficiency_result"]["b1_efficiency_trustworthy"] is False
    assert artifact["action_efficiency_result"]["d_honest_verdict"].endswith("retired")
    assert artifact["action_efficiency_result"]["reported_lift"] is None
    assert artifact["heldout_first_win_rate"]["rate"] == 0.04
    assert artifact["heldout_first_win_rate"]["flag_resolved"] is True
    assert artifact["heldout_first_win_rate"]["games_evaluated"] == 25
    assert artifact["submission_package_ready"]["ready"] is True
    assert artifact["submission_package_ready"]["peak_vram_gb"] == 15.146
    assert artifact["arc_first_win_wall_closed"] is True
    assert artifact["milestone_scorecard"]["wall_closure"]["closure_verdict"] == "WALL_IS_HIDDEN_STATE"
    assert artifact["milestone_scorecard"]["reserved_lanes"]["b3_stamping"]["decision"] == (
        "reserved_lane_blocked_insufficient_v454_mtime_window"
    )
    assert artifact["milestone_scorecard"]["reserved_lanes"]["c_kv260"]["decision"] == (
        "reserved_lane_kv260_continuity_ok"
    )
    assert artifact["post_sprint_pivot"]["arxiv_id"] == "2605.18871"
    assert artifact["post_sprint_pivot"]["domain"] == "non_saturated_structured_reasoning_domain"
    assert "FoVer paper" in artifact["post_sprint_pivot"]["deliverable"]
    assert artifact["post_sprint_pivot"]["do_not_queue"] == "representation_5"

    skipped = {row["source"]: row for row in artifact["skipped_flagged_adversarial"]}
    assert skipped["A3_SELF_PLAY"]["reason"] == "live_critical_recheck"
    assert skipped["A3_SELF_PLAY"]["summarizer_exit_code"] == 2
    cited_sources = {row["source"] for row in artifact["cited_upstream_artifacts"]}
    assert "A3_SELF_PLAY" not in cited_sources
    assert {"PREV_CAPSTONE", "A4_HELDOUT", "B1_AUDIT", "B2_PACKAGE", "D_EFFICIENCY"} <= cited_sources
    assert all(row["sha256"].startswith("sha256:") for row in artifact["cited_upstream_artifacts"])
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4934_trusted_banks_and_efficiency_are_gated() -> None:
    """SCENARIO-CAPSTONE-4934: B1 trust is required before banks or D lift are counted."""

    artifacts = _artifacts(banks_trustworthy=True, efficiency_trustworthy=True)
    artifacts["A1_LEVELUP"]["new_levels_banked"] = 1
    artifacts["A2_LEVELUP"]["new_levels_banked"] = 1
    artifacts["D_EFFICIENCY"]["matm_similarity_retrieval_lift"] = 0.25
    artifacts["D_EFFICIENCY"]["honest_verdict"] = "success_matm_similarity_retrieval_efficiency_lift"

    artifact = mod.build_artifact(
        artifacts=artifacts,
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 71},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(self_play_code=0, heldout_code=0),
        duration_s=0.001,
    )

    assert artifact["reproducible_total_levels"] == 71
    assert artifact["banks_counted"]["b1_banks_trustworthy"] is True
    assert artifact["banks_counted"]["counted"] == [
        {"source": "A1_LEVELUP", "game": "sp80", "new_levels_banked": 1},
        {"source": "A2_LEVELUP", "game": "su15", "new_levels_banked": 1},
    ]
    assert artifact["action_efficiency_result"]["decision"] == "trusted_efficiency_lift"
    assert artifact["action_efficiency_result"]["reported_lift"] == 0.25
    assert "A3_SELF_PLAY" in {row["source"] for row in artifact["cited_upstream_artifacts"]}
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4934_runtime_blocks_missing_required_inputs(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4934-BLOCKED-PRECONDITION: runtime checks fail closed."""

    for key, payload in _artifacts().items():
        if key != "D_EFFICIENCY":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 69\n", encoding="utf-8")
    north_star = tmp_path / mod.NORTH_STAR_RELATIVE_PATH
    north_star.parent.mkdir(parents=True, exist_ok=True)
    north_star.write_text("## 1\nFoVer\n## 2\npaper_ready\n## 5\nverifier moat\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4934\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")
    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        return mod.SummarizerResult(["summarize", relative_path], 0, "clean", "")

    blocked = mod.run_capstone(root=tmp_path, summarizer=summarizer)

    assert called == [
        spec.relative_path
        for source, spec in mod.UPSTREAM_SOURCES.items()
        if source != "D_EFFICIENCY"
    ]
    assert blocked["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert blocked["capstone_ready"] is False
    assert blocked["preconditions_checked"]["upstream_artifacts"]["D_EFFICIENCY"]["present"] is False
    assert blocked["preconditions_checked"]["upstream_artifacts"]["A4_HELDOUT"][
        "summarizer_exit_code"
    ] == 0
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == blocked
    assert mod.validate_artifact(blocked) == []

    bad_registry_root = tmp_path / "bad_registry"
    for key, payload in _artifacts().items():
        _write_json(bad_registry_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    bad_registry = bad_registry_root / mod.REGISTRY_RELATIVE_PATH
    bad_registry.parent.mkdir(parents=True, exist_ok=True)
    bad_registry.write_text("schema_version: [unterminated\n", encoding="utf-8")
    (bad_registry_root / mod.NORTH_STAR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (bad_registry_root / mod.NORTH_STAR_RELATIVE_PATH).write_text("## 1\n## 2\n## 5\n", encoding="utf-8")
    (bad_registry_root / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (bad_registry_root / mod.SPEC_RELATIVE_PATH).write_text("REQ-CAPSTONE-4934\n", encoding="utf-8")
    (bad_registry_root / mod.SUMMARIZER_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (bad_registry_root / mod.SUMMARIZER_RELATIVE_PATH).write_text("# placeholder\n", encoding="utf-8")

    registry_blocked = mod.run_capstone(root=bad_registry_root, summarizer=summarizer)
    assert registry_blocked["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert registry_blocked["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(registry_blocked) == []

    success_root = tmp_path / "success"
    for key, payload in _artifacts().items():
        _write_json(success_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    success_registry = success_root / mod.REGISTRY_RELATIVE_PATH
    success_registry.parent.mkdir(parents=True, exist_ok=True)
    success_registry.write_text(
        "schema_version: 1\nreproducible_total_levels: 69\n", encoding="utf-8"
    )
    (success_root / mod.NORTH_STAR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (success_root / mod.NORTH_STAR_RELATIVE_PATH).write_text("## 1\n## 2\n## 5\n", encoding="utf-8")
    (success_root / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (success_root / mod.SPEC_RELATIVE_PATH).write_text("REQ-CAPSTONE-4934\n", encoding="utf-8")
    (success_root / mod.SUMMARIZER_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (success_root / mod.SUMMARIZER_RELATIVE_PATH).write_text("# placeholder\n", encoding="utf-8")

    success = mod.run_capstone(root=success_root, summarizer=summarizer)
    assert success["capstone_ready"] is True
    assert success["honest_verdict"].startswith("complete_capstone_v454_submission_maximized")
    assert mod.validate_artifact(success) == []


def test_scenario_capstone_4934_validation_rejects_schema_drift() -> None:
    """SCENARIO-CAPSTONE-4934-FIELD-PRINCIPLES: malformed scorecards fail validation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert "missing_field:headline" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "headline"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "maybe"}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "69"}
    )
    assert "invalid_banks_counted" in mod.validate_artifact({**artifact, "banks_counted": []})
    assert "invalid_action_efficiency_result" in mod.validate_artifact(
        {**artifact, "action_efficiency_result": []}
    )
    assert "invalid_heldout_first_win_rate" in mod.validate_artifact(
        {**artifact, "heldout_first_win_rate": 0.04}
    )
    assert "invalid_submission_package_ready" in mod.validate_artifact(
        {**artifact, "submission_package_ready": True}
    )
    assert "invalid_arc_first_win_wall_closed" in mod.validate_artifact(
        {**artifact, "arc_first_win_wall_closed": "true"}
    )
    assert "invalid_post_sprint_pivot" in mod.validate_artifact(
        {**artifact, "post_sprint_pivot": []}
    )
    assert "invalid_preconditions_checked" in mod.validate_artifact(
        {**artifact, "preconditions_checked": []}
    )
    assert "invalid_random_seed" in mod.validate_artifact({**artifact, "random_seed": "seed"})
    assert "invalid_capstone_ready" in mod.validate_artifact(
        {**artifact, "capstone_ready": "true"}
    )
    assert "invalid_milestone_scorecard" in mod.validate_artifact(
        {**artifact, "milestone_scorecard": []}
    )
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4928}]}
    )
    assert "invalid_skipped_flagged_adversarial" in mod.validate_artifact(
        {**artifact, "skipped_flagged_adversarial": [{"experiment_id": 4927}]}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert "missing_principle:headline" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert mod._experiment_id("PREV_CAPSTONE", {"experiment": "experiment_4923_capstone"}) == 4923
    assert mod._live_recheck(None) == "not_run"
    assert mod._live_recheck(mod.SummarizerResult([], 1, "", "")) == "warn"
    assert mod._live_recheck(mod.SummarizerResult([], 0, "", "")) == "clean"
    assert mod._is_skipped({"flagged_adversarial": True}, None) == "flagged_adversarial"
    assert mod._banks_counted({}, 69)["candidate_banks"] == [
        {"source": "A1_LEVELUP", "game": "", "new_levels_banked": 0},
        {"source": "A2_LEVELUP", "game": "", "new_levels_banked": 0},
    ]
    assert mod._action_efficiency_result({})["decision"] == "not_reported_missing_or_untrusted"
    assert mod._heldout_first_win_rate({})["status"] == "missing_or_skipped"
    assert mod._submission_package_ready({})["status"] == "missing_or_skipped"
    assert mod._rate_slug({}) == "unknown"
    assert mod._first_blocker(
        summarizer_present=False,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=True,
        north_star_present=True,
        upstream_preconditions={},
    ) == "missing_summarizer"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=False,
        registry_loadable=True,
        spec_has_req=True,
        north_star_present=True,
        upstream_preconditions={},
    ) == "missing_registry"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=False,
        north_star_present=True,
        upstream_preconditions={},
    ) == "spec_missing_req_4934"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=True,
        north_star_present=False,
        upstream_preconditions={},
    ) == "missing_north_star"
    assert (
        mod._first_blocker(
            summarizer_present=True,
            registry_present=True,
            registry_loadable=True,
            spec_has_req=True,
            north_star_present=True,
            upstream_preconditions={"A": {"present": True}},
        )
        is None
    )
