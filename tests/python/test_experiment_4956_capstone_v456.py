"""Tests for REQ-CAPSTONE-4956 / SCENARIO-CAPSTONE-4956."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4956_capstone_v456 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _transition() -> JsonDict:
    return {
        "experiment": "experiment_4946_archive_455_activate_456",
        "experiment_id": 4946,
        "honest_verdict": "blocked_research_roadmap_next_yaml_missing",
        "arc_first_win_wall_closed_hidden_state": True,
        "deliverable_locked_agent_plus_fover_paper": True,
        "reproducible_total_levels": 69,
        "close_state_455": {
            "wall_closure": {
                "closed": True,
                "closure_verdict": "WALL_IS_HIDDEN_STATE",
                "trusted_source": "B1",
            },
            "deliverable": "current ~0.05 agent + publishable FoVer paper",
            "do_not_queue": ["representation_5"],
        },
    }


def _levelup(game: str, exp_id: int, levels: int) -> JsonDict:
    return {
        "experiment": f"experiment_{exp_id}_levelup_attempt",
        "experiment_id": exp_id,
        "honest_verdict": f"complete_{game}_no_new_level_residual_no_grounded_delta",
        "target_game": game,
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": levels,
        "solve_provenance": "live_agent_self_discovery",
        "live_path_reachable": True,
        "registry_update": {"target_game": game, "banked_levels": 0},
        "reproduction_gate": {},
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4949_self_play_verifier_checkpoint",
        "experiment_id": 4949,
        "honest_verdict": "success_self_play_checkpoint_refreshed",
        "target_game": "lp85",
        "verifier_checkpoint_refreshed": True,
        "checkpoint_path": "models/arc_verifier_lp85.json",
        "offline_reproduced": True,
        "reproduced_levels": 3,
        "reproduction_gate": {
            "game": "lp85",
            "claimed_level": 3,
            "reached_level": 3,
            "reproduced": True,
        },
        "flag_resolved": True,
        "duration_s": 10.198359,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _heldout() -> JsonDict:
    return {
        "experiment": "experiment_4950_heldout_first_win_readiness",
        "experiment_id": 4950,
        "honest_verdict": "complete_heldout_first_win_0.04_full25_final_flag_resolved",
        "flagged_adversarial": False,
        "heldout_first_win_rate": 0.04,
        "heldout_first_win_ci": {
            "method": "paired_percentile_bootstrap",
            "ci95": [0.0, 0.0],
            "point": 0.0,
        },
        "games_evaluated": 25,
        "flag_resolved": True,
        "positive_control_passed": True,
        "model_specs": {"generator": "Qwen3.5-9B-MTP", "backend": "gpu0_cuda"},
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
    }


def _pivot() -> JsonDict:
    return {
        "experiment": "experiment_4951_distributional_energy_verifier_turnkey",
        "experiment_id": 4951,
        "honest_verdict": "success_distributional_energy_verifier_pivot_turnkey_ready",
        "pivot_executable_on_7_1": True,
        "pivot_turnkey": True,
        "three_column_dry_run_ok": True,
        "sc_not_saturated_domain": "TravelPlanner",
        "validation_gate": {
            "beats_self_consistency_ci95_excludes_zero_required": True,
            "oracle_distinct_required": True,
            "no_model_identity_shortcut_required": True,
            "claimed_met": False,
            "real_post_6_30_experiment_must_pass": (
                "distributional_energy_verifier beats self-consistency with CI95 excluding zero"
            ),
        },
        "verifier_is_oracle": False,
        "moat_proven_claimed": False,
        "arxiv_ids_cited": ["2605.18871", "2504.16828", "2502.01989"],
    }


def _audit(*, banks_trustworthy: bool = True, pivot_trustworthy: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_4952_bank_and_pivot_audit",
        "experiment_id": 4952,
        "honest_verdict": "complete_v456_banks_and_pivot_audited_trusted",
        "banks_trustworthy": banks_trustworthy,
        "pivot_readiness_trustworthy": pivot_trustworthy,
        "checks": {
            "reproduction_genuine": True,
            "not_duplicate": True,
            "self_discovery_provenance": True,
            "live_path_reachable": True,
            "oracle_distinct_design": pivot_trustworthy,
            "honest_readiness": pivot_trustworthy,
        },
        "audit_failure_reasons": [] if pivot_trustworthy else ["D_honest_readiness_failed"],
        "bank_evidence": {
            "A1": {"bank_claimed": False, "game": "ar25"},
            "A2": {"bank_claimed": False, "game": "vc33"},
        },
        "pivot_readiness_evidence": {
            "checks": {
                "oracle_distinct_design": pivot_trustworthy,
                "honest_readiness": pivot_trustworthy,
            }
        },
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4953_submission_package_harden",
        "experiment_id": 4953,
        "honest_verdict": "success_submission_package_ready_final_pre_deadline",
        "submission_package_ready": True,
        "submits": False,
        "operator_only": True,
        "peak_vram_gb": 15.146,
        "frozen_stack_loads": True,
        "package_builds": True,
        "ready_package_regression_ok": True,
    }


def _stamping() -> JsonDict:
    return {
        "experiment": "experiment_4954_stamping_backfill_and_wiring_readiness",
        "experiment_id": 4954,
        "honest_verdict": "success_v456_stamping_backfilled_and_mtime_window_confirmed",
        "stamping_backfilled_arms": "none missing",
        "mtime_fallback_window": {
            "n_arms": 8,
            "compute_bound_count": 4,
            "wall_minutes": 100.2,
            "window_start": "2026-06-28T23:09:01Z",
            "window_end": "2026-06-29T00:49:13Z",
        },
        "window_gate_relaxed": True,
        "wiring_proposal_reconfirmed": True,
        "research_conductor_modified": False,
    }


def _kv260() -> JsonDict:
    return {
        "experiment": 4955,
        "honest_verdict": "success_kv260_continuity_ok",
        "kv260_ssh_reachable": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "xmutil_requires_sudo": True,
        "verifier_is_oracle": False,
    }


def _artifacts() -> dict[str, JsonDict]:
    return {
        "TRANSITION": _transition(),
        "A1_LEVELUP": _levelup("ar25", 4947, 3),
        "A2_LEVELUP": _levelup("vc33", 4948, 2),
        "A3_SELF_PLAY": _self_play(),
        "A4_HELDOUT": _heldout(),
        "D_PIVOT": _pivot(),
        "B1_AUDIT": _audit(),
        "B2_PACKAGE": _package(),
        "B3_STAMPING": _stamping(),
        "C_KV260": _kv260(),
    }


def _hashes() -> dict[str, str]:
    return {source: f"sha256:{source.lower()}" for source in mod.UPSTREAM_SOURCES}


def _summaries(*, a3_code: int = 0, a4_code: int = 1) -> dict[str, mod.SummarizerResult]:
    results: dict[str, mod.SummarizerResult] = {}
    for source, spec in mod.UPSTREAM_SOURCES.items():
        exit_code = a4_code if source == "A4_HELDOUT" else a3_code if source == "A3_SELF_PLAY" else 0
        status = "CRITICAL" if exit_code >= 2 else "warn" if exit_code == 1 else "clean"
        results[source] = mod.SummarizerResult(
            ["summarize", spec.relative_path],
            exit_code,
            f"LIVE re-check: {status}",
            "",
        )
    return results


def test_req_capstone_4956_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4956: OpenSpec declares the final .456 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-4956") :]

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


def test_scenario_capstone_4956_submission_ready_and_turnkey_handoff() -> None:
    """SCENARIO-CAPSTONE-4956: scorecard reports readiness without a new moat claim."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == (
        "complete_capstone_v456_submission_ready_levels_69_heldout_0.04_"
        "package_ready_pivot_turnkey_7_1"
    )
    assert artifact["capstone_ready"] is True
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["banks_counted"]["b1_banks_trustworthy"] is True
    assert artifact["banks_counted"]["counted"] == []
    assert artifact["banks_counted"]["candidate_banks"] == [
        {"source": "A1_LEVELUP", "game": "ar25", "new_levels_banked": 0},
        {"source": "A2_LEVELUP", "game": "vc33", "new_levels_banked": 0},
    ]
    assert artifact["heldout_first_win_rate"]["rate"] == 0.04
    assert artifact["heldout_first_win_rate"]["flag_resolved"] is True
    assert artifact["submission_package_ready"]["ready"] is True
    assert artifact["submission_package_ready"]["peak_vram_gb"] == 15.146
    assert artifact["a3_substrate_flag_resolved"] is True
    assert artifact["milestone_scorecard"]["a3_substrate"]["true_live_recheck"] == "clean"
    assert artifact["b3_window_nonzero"] is True
    assert artifact["milestone_scorecard"]["reserved_lanes"]["b3_stamping"]["window_nonzero"] is True
    assert artifact["milestone_scorecard"]["reserved_lanes"]["c_kv260"]["decision"] == (
        "reserved_lane_kv260_continuity_ok"
    )
    assert artifact["arc_first_win_wall_closed"] is True
    assert artifact["post_sprint_pivot"]["pivot_turnkey"] is True
    assert artifact["post_sprint_pivot"]["pivot_executable_on_7_1"] is True
    assert artifact["post_sprint_pivot"]["moat_proven"] is False
    assert artifact["post_sprint_pivot"]["arxiv_id"] == "2605.18871"
    assert artifact["pivot_executable_on_7_1"] is True
    assert artifact["skipped_flagged_adversarial"] == []
    assert "post-6/30 verifier-moat pivot turnkey 7/1" in artifact["headline"]

    cited_sources = {row["source"] for row in artifact["cited_upstream_artifacts"]}
    assert set(mod.UPSTREAM_SOURCES) == cited_sources
    assert all(row["sha256"].startswith("sha256:") for row in artifact["cited_upstream_artifacts"])
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4956_banks_pivot_and_a3_are_gated_by_trust() -> None:
    """REQ-CAPSTONE-4956: B1 gates banks/pivot and live-critical A3 is skipped."""

    artifacts = _artifacts()
    artifacts["A1_LEVELUP"]["new_levels_banked"] = 1
    artifacts["A2_LEVELUP"]["new_levels_banked"] = 1

    untrusted = {**artifacts, "B1_AUDIT": _audit(banks_trustworthy=False, pivot_trustworthy=False)}
    untrusted_artifact = mod.build_artifact(
        artifacts=untrusted,
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert untrusted_artifact["reproducible_total_levels"] == 69
    assert untrusted_artifact["banks_counted"]["counted"] == []
    assert untrusted_artifact["pivot_executable_on_7_1"] is False
    assert untrusted_artifact["post_sprint_pivot"]["decision"] == "pivot_not_stated_untrusted"
    assert untrusted_artifact["capstone_ready"] is False

    trusted_artifact = mod.build_artifact(
        artifacts=artifacts,
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert trusted_artifact["reproducible_total_levels"] == 71
    assert trusted_artifact["banks_counted"]["counted"] == [
        {"source": "A1_LEVELUP", "game": "ar25", "new_levels_banked": 1},
        {"source": "A2_LEVELUP", "game": "vc33", "new_levels_banked": 1},
    ]
    assert trusted_artifact["pivot_executable_on_7_1"] is True

    a3_live_critical = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 69},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(a3_code=2),
        duration_s=0.001,
    )
    skipped = {row["source"]: row for row in a3_live_critical["skipped_flagged_adversarial"]}
    cited = {row["source"] for row in a3_live_critical["cited_upstream_artifacts"]}
    assert skipped["A3_SELF_PLAY"]["reason"] == "live_critical_recheck"
    assert "A3_SELF_PLAY" not in cited
    assert a3_live_critical["a3_substrate_flag_resolved"] is False
    assert a3_live_critical["capstone_ready"] is False


def test_scenario_capstone_4956_runtime_blocks_missing_required_inputs(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4956-BLOCKED-PRECONDITION: missing inputs fail closed."""

    for key, payload in _artifacts().items():
        if key != "B2_PACKAGE":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 69\n", encoding="utf-8")
    north_star = tmp_path / mod.NORTH_STAR_RELATIVE_PATH
    north_star.parent.mkdir(parents=True, exist_ok=True)
    north_star.write_text("## 0\nARC\n## 1\nFoVer\n## 5\nverifier moat\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4956\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")
    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        return mod.SummarizerResult(["summarize", relative_path], 0, "clean", "")

    blocked = mod.run_capstone(root=tmp_path, summarizer=summarizer)

    assert called == [
        spec.relative_path for key, spec in mod.UPSTREAM_SOURCES.items() if key != "B2_PACKAGE"
    ]
    assert blocked["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert blocked["capstone_ready"] is False
    assert blocked["preconditions_checked"]["upstream_artifacts"]["B2_PACKAGE"]["present"] is False
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == blocked
    assert mod.validate_artifact(blocked) == []

    bad_registry_root = tmp_path / "bad_registry"
    for key, payload in _artifacts().items():
        _write_json(bad_registry_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    bad_registry = bad_registry_root / mod.REGISTRY_RELATIVE_PATH
    bad_registry.parent.mkdir(parents=True, exist_ok=True)
    bad_registry.write_text("schema_version: [unterminated\n", encoding="utf-8")
    (bad_registry_root / mod.NORTH_STAR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (bad_registry_root / mod.NORTH_STAR_RELATIVE_PATH).write_text("## 0\n## 1\n## 5\n", encoding="utf-8")
    (bad_registry_root / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (bad_registry_root / mod.SPEC_RELATIVE_PATH).write_text("REQ-CAPSTONE-4956\n", encoding="utf-8")
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
    success_registry.write_text("schema_version: 1\nreproducible_total_levels: 69\n", encoding="utf-8")
    (success_root / mod.NORTH_STAR_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (success_root / mod.NORTH_STAR_RELATIVE_PATH).write_text("## 0\n## 1\n## 5\n", encoding="utf-8")
    (success_root / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (success_root / mod.SPEC_RELATIVE_PATH).write_text("REQ-CAPSTONE-4956\n", encoding="utf-8")
    (success_root / mod.SUMMARIZER_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (success_root / mod.SUMMARIZER_RELATIVE_PATH).write_text("# placeholder\n", encoding="utf-8")

    success = mod.run_capstone(root=success_root, summarizer=summarizer)
    assert success["capstone_ready"] is True
    assert success["honest_verdict"].startswith("complete_capstone_v456_submission_ready")
    assert mod.validate_artifact(success) == []


def test_scenario_capstone_4956_validation_rejects_schema_drift() -> None:
    """SCENARIO-CAPSTONE-4956-FIELD-PRINCIPLES: malformed scorecards fail validation."""

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
    assert "invalid_heldout_first_win_rate" in mod.validate_artifact(
        {**artifact, "heldout_first_win_rate": 0.04}
    )
    assert "invalid_submission_package_ready" in mod.validate_artifact(
        {**artifact, "submission_package_ready": True}
    )
    assert "invalid_a3_substrate_flag_resolved" in mod.validate_artifact(
        {**artifact, "a3_substrate_flag_resolved": "true"}
    )
    assert "invalid_b3_window_nonzero" in mod.validate_artifact(
        {**artifact, "b3_window_nonzero": "true"}
    )
    assert "invalid_arc_first_win_wall_closed" in mod.validate_artifact(
        {**artifact, "arc_first_win_wall_closed": "true"}
    )
    assert "invalid_post_sprint_pivot" in mod.validate_artifact(
        {**artifact, "post_sprint_pivot": []}
    )
    assert "invalid_pivot_executable_on_7_1" in mod.validate_artifact(
        {**artifact, "pivot_executable_on_7_1": "true"}
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
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4950}]}
    )
    assert "invalid_skipped_flagged_adversarial" in mod.validate_artifact(
        {**artifact, "skipped_flagged_adversarial": [{"experiment_id": 4949}]}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert "missing_principle:headline" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert mod._experiment_id("C_KV260", {"experiment": 4955}) == 4955
    assert mod._experiment_id("A4_HELDOUT", {"experiment": "experiment_4950_ready"}) == 4950
    assert mod._experiment_id("A1_LEVELUP", {}) == 4947
    assert mod._live_recheck(None) == "not_run"
    assert mod._live_recheck(mod.SummarizerResult([], 1, "", "")) == "warn"
    assert mod._live_recheck(mod.SummarizerResult([], 0, "", "")) == "clean"
    assert mod._is_skipped({"flagged_adversarial": True}, None) == "flagged_adversarial"
    assert mod._banks_counted({}, 69)["candidate_banks"] == [
        {"source": "A1_LEVELUP", "game": "", "new_levels_banked": 0},
        {"source": "A2_LEVELUP", "game": "", "new_levels_banked": 0},
    ]
    assert mod._a3_substrate({})["resolved"] is False
    assert mod._b3_window({})["window_nonzero"] is False
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
    ) == "spec_missing_req_4956"
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
