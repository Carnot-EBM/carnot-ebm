"""Tests for REQ-CAPSTONE-4859 / SCENARIO-CAPSTONE-4859."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4859_capstone_v447 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1(*, dominant: str = "NEVER_ENUMERATED", flagged: bool = False) -> JsonDict:
    buckets = {
        "lp85": "COVERED",
        "r11l": dominant,
        "wa30": dominant,
        "sk48": dominant,
    }
    return {
        "experiment": "experiment_4851_generation_coverage_diagnostic",
        "experiment_id": 4851,
        "honest_verdict": f"complete_generation_wall_{dominant.lower()}_dominant",
        "dominant_bucket": dominant,
        "positive_control_game": "tu93",
        "positive_control_covered": True,
        "positive_control_coverage": {
            "game": "tu93",
            "bucket": "COVERED",
            "adaptered": True,
            "reached_l1_win": True,
            "winning_prefix_len": 18,
            "matched_winning_prefix_len": 18,
        },
        "proposer_blind_to_banked_answer": True,
        "n_games_measured": len(buckets),
        "per_game_coverage": {
            game: {
                "game": game,
                "bucket": bucket,
                "winning_prefix_len": 5,
                "matched_winning_prefix_len": 1 if bucket != "COVERED" else 5,
                "pool_size": 100,
                "reached_l1_win": bucket == "COVERED",
                "budget_actions": 160,
            }
            for game, bucket in buckets.items()
        },
        "verifier_is_oracle": True,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "flagged_adversarial": flagged,
    }


def _b1(
    *,
    diagnostic: bool = True,
    proposer_blind: bool = True,
    positive_control: bool = True,
    buckets_match: bool = True,
) -> JsonDict:
    reasons = []
    if not proposer_blind:
        reasons.append("banked_answer_used_before_classification")
    if not positive_control:
        reasons.append("positive_control_not_covered")
    if not buckets_match:
        reasons.append("dominant_bucket_mismatch")
    if not diagnostic:
        reasons.append("a1_not_genuinely_diagnostic")
    return {
        "experiment": "experiment_4855_generation_diagnostic_audit",
        "experiment_id": 4855,
        "honest_verdict": (
            "complete_a1_generation_diagnostic_audited"
            if diagnostic and not reasons
            else "complete_a1_generation_diagnostic_non_test"
        ),
        "a1_genuinely_diagnostic": diagnostic,
        "proposer_blind_confirmed": proposer_blind,
        "positive_control_confirmed": positive_control,
        "buckets_match_claim": buckets_match,
        "live_path_reachable_confirmed": True,
        "solve_provenance_confirmed": True,
        "source_dominant_bucket": "NEVER_ENUMERATED",
        "non_diagnostic_reasons": reasons,
        "checks": {
            "proposer_blind_to_banked_answer": {"passed": proposer_blind},
            "positive_control": {"passed": positive_control},
            "bucket_distribution": {
                "passed": buckets_match,
                "bucket_counts": {"COVERED": 1, "NEVER_ENUMERATED": 3},
                "computed_dominant_bucket": "NEVER_ENUMERATED",
                "claimed_dominant_bucket": "NEVER_ENUMERATED",
            },
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4852_levelup_attempt",
        "experiment_id": 4852,
        "honest_verdict": "complete_s5i5_no_new_level_residual_needs_per_game_RE",
        "target_game": "s5i5",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "registry_update": {
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": 65,
            "updated": False,
            "reason": "no_new_level_banked",
        },
        "attempted_games": [{"game": "s5i5", "reached_level": 0}],
        "dead_ends": ["s5i5: needs_per_game_RE"],
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": True,
        "inference_substrate": "adapter_free_graph_explore_no_induction",
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4853_self_play_verifier_checkpoint",
        "experiment_id": 4853,
        "honest_verdict": "success_self_play_checkpoint_refreshed",
        "target_game": "re86",
        "verifier_checkpoint_refreshed": True,
        "checkpoint_path": "models/arc_verifier_re86.json",
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "reproduction_gate": {"game": "re86", "claimed_level": 2, "reproduced": True},
        "search_state_count": 56,
        "solve_provenance": "live_agent_self_discovery",
        "inference_substrate": "live_llm_inference",
    }


def _heldout() -> JsonDict:
    return {
        "experiment": "experiment_4854_heldout_first_win_readiness",
        "experiment_id": 4854,
        "honest_verdict": "complete_heldout_first_win_0.04_flat_genuine_null",
        "heldout_first_win_rate": 0.04,
        "first_win_baseline": 0.04,
        "prior_best_heldout_first_win_rate": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_first_win_delta_vs_prior_best": 0.0,
        "heldout_variant_attempts": 100,
        "positive_control_passed": True,
        "parity_test_green": True,
        "checkpoint_emitted": False,
        "live_agent_ran": False,
        "solve_provenance": "development_proxy",
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _package(*, flagged: bool = False) -> JsonDict:
    return {
        "experiment": "experiment_4856_submission_package_harden",
        "experiment_id": 4856,
        "honest_verdict": "success_submission_package_ready",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"dry_build_ran": True, "package_builds": True},
        "agent_config_resolution": {"resolved": True},
        "packaging_requirements_crosscheck": {"ok": True},
        "ready_package_regression_check": {"ok": True, "regressions": []},
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "flagged_adversarial": flagged,
    }


def _hardware() -> JsonDict:
    return {
        "experiment": "experiment_4857_kv260_continuity",
        "experiment_id": 4857,
        "honest_verdict": "success_kv260_continuity_ok",
        "kv260_ssh_reachable": True,
        "board_state": {"captured": True, "hostname": "kv260", "uio_device_count": 5},
        "next_forward_step": "continue continuity checks",
        "inference_substrate": "hardware_smoke",
    }


def _sota() -> JsonDict:
    return {
        "experiment": "experiment_4858_sota_ingestion_generation_expressibility",
        "experiment_id": 4858,
        "honest_verdict": "success_sota_ingestion_generation_expressibility_mapped",
        "aimed_at_dominant_bucket": "NEVER_ENUMERATED",
        "flagged_for_v448": [
            {"candidate": "dreamcoder_lilo_action_library"},
            {"candidate": "eg_nps_soar_arc_program_search"},
        ],
        "methods_mapped": [
            {"method": "DreamCoder/LILO action-primitive library learner"},
            {"method": "Neurally guided ARC DSL program search"},
        ],
        "arxiv_ids_cited": ["2006.08381", "2310.19791"],
        "generation_expressibility_mapping_note": {"root_cause": "generation expressibility"},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(
    *,
    a1: JsonDict | None = None,
    b1: JsonDict | None = None,
    package: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "A1": a1 or _a1(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": _self_play(),
        "HELDOUT": _heldout(),
        "B1_AUDIT": b1 or _b1(),
        "PACKAGE": package or _package(),
        "HARDWARE": _hardware(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {source: f"sha256:{source.lower()}" for source in mod.UPSTREAM_SOURCES}


def _summaries(*, a1_code: int = 0) -> dict[str, mod.SummarizerResult]:
    return {
        source: mod.SummarizerResult(
            ["summarize", spec.relative_path],
            a1_code if source == "A1" else 0,
            "LIVE re-check: CRITICAL" if source == "A1" and a1_code >= 2 else "clean",
            "",
        )
        for source, spec in mod.UPSTREAM_SOURCES.items()
    }


def test_req_capstone_4859_spec_declares_generation_wall_scorecard() -> None:
    """REQ-CAPSTONE-4859: OpenSpec declares the .447 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4859_default_trusted_never_enumerated_verdict() -> None:
    """SCENARIO-CAPSTONE-4859: B1-trusted A1 imports the dominant generation bucket."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    a1 = artifact["a1_generation_wall_verdict"]
    assert artifact["honest_verdict"] == (
        "complete_a1_generation_wall_never_enumerated_capstone_ready"
    )
    assert a1["verdict"] == "generation_wall_never_enumerated"
    assert a1["dominant_bucket"] == "NEVER_ENUMERATED"
    assert a1["b1_trusted"] is True
    assert a1["trust_checks"] == {
        "a1_genuinely_diagnostic": True,
        "proposer_blind_confirmed": True,
        "positive_control_covered": True,
        "positive_control_confirmed_by_b1": True,
        "buckets_match_claim": True,
    }
    assert a1["bucket_counts"] == {"COVERED": 1, "NEVER_ENUMERATED": 3}
    assert a1["next_448_pivot"] == "generation_expressibility_program_synthesis"
    assert artifact["scored_lever_state"] == {
        "level_up_banked": False,
        "heldout_first_win_rate": 0.04,
        "live_agent_ran": False,
        "submission_package_ready": True,
    }
    assert artifact["levelup_bank"]["level_up_banked"] is False
    assert artifact["self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert artifact["heldout_readiness"]["decision"] == "flat_baseline_first_win_null"
    assert artifact["submission_package_state"]["decision"] == "package_ready_operator_only"
    assert artifact["hardware_continuity"]["decision"] == "kv260_reachable"
    assert artifact["sota_handoff"]["decision"] == "generation_expressibility_handoff"
    assert artifact["reproducible_total_levels"] == 65
    assert len(artifact["cited_upstream_artifacts"]) == len(mod.UPSTREAM_SOURCES)
    assert artifact["flagged_artifacts_skipped"] == []
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4859_b1_failure_labels_a1_non_test() -> None:
    """SCENARIO-CAPSTONE-4859: B1 failure voids the A1 generation-wall headline."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(b1=_b1(proposer_blind=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete_a1_generation_wall_non_test_capstone_ready"
    a1 = artifact["a1_generation_wall_verdict"]
    assert a1["verdict"] == "non_test_b1_untrusted"
    assert a1["b1_trusted"] is False
    assert a1["dominant_bucket"] is None
    assert a1["untrusted_dominant_bucket_claim"] == "NEVER_ENUMERATED"
    assert a1["next_448_pivot"] == "do_not_use_a1_non_test"
    assert "proposer_blind_confirmed" in a1["trust_failure_reasons"]
    assert mod.validate_artifact(artifact) == []

    invalid_bucket = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(dominant="UNKNOWN")),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert invalid_bucket["honest_verdict"] == (
        "complete_a1_generation_wall_non_test_capstone_ready"
    )
    assert invalid_bucket["a1_generation_wall_verdict"]["verdict"] == "non_test_invalid_bucket"
    assert mod.validate_artifact(invalid_bucket) == []


def test_scenario_capstone_4859_flagged_upstream_is_skipped_not_aggregated() -> None:
    """REQ-CAPSTONE-4859: flagged_adversarial upstreams are excluded from aggregation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(package=_package(flagged=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["scored_lever_state"]["submission_package_ready"] is False
    assert artifact["submission_package_state"] == {}
    assert artifact["flagged_artifacts_skipped"] == [
        {
            "source": "PACKAGE",
            "experiment_id": 4856,
            "path": mod.UPSTREAM_SOURCES["PACKAGE"].relative_path,
            "reason": "flagged_adversarial",
            "sha256": "sha256:package",
        }
    ]
    cited_ids = {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    assert 4856 not in cited_ids
    assert mod.validate_artifact(artifact) == []


def test_run_capstone_invokes_summarizer_and_blocks_on_missing_upstreams(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4859-BLOCKED-PRECONDITION: runtime checks fail closed."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4859\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")
    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        return mod.SummarizerResult(["summarize", relative_path], 0, "clean", "")

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)
    assert sorted(called) == sorted(spec.relative_path for spec in mod.UPSTREAM_SOURCES.values())
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"]["registry"]["yaml_loadable"] is True
    assert mod.validate_artifact(artifact) == []

    missing_root = tmp_path / "missing"
    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(missing_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    missing_registry = missing_root / mod.REGISTRY_RELATIVE_PATH
    missing_registry.parent.mkdir(parents=True, exist_ok=True)
    missing_registry.write_text(
        "schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8"
    )
    missing_spec = missing_root / mod.SPEC_RELATIVE_PATH
    missing_spec.parent.mkdir(parents=True, exist_ok=True)
    missing_spec.write_text("REQ-CAPSTONE-4859\n", encoding="utf-8")
    missing_summarizer = missing_root / mod.SUMMARIZER_RELATIVE_PATH
    missing_summarizer.parent.mkdir(parents=True, exist_ok=True)
    missing_summarizer.write_text("# placeholder\n", encoding="utf-8")

    blocked = mod.run_capstone(root=missing_root, summarizer=summarizer)
    assert blocked["honest_verdict"] == "blocked_upstreams_missing"
    assert blocked["a1_generation_wall_verdict"] == {}
    assert blocked["cited_upstream_artifacts"] == []
    assert blocked["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(blocked) == []

    bad_registry_root = tmp_path / "bad_registry"
    for key, payload in _artifacts().items():
        _write_json(bad_registry_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    bad_registry = bad_registry_root / mod.REGISTRY_RELATIVE_PATH
    bad_registry.parent.mkdir(parents=True, exist_ok=True)
    bad_registry.write_text("schema_version: [unterminated\n", encoding="utf-8")
    bad_spec = bad_registry_root / mod.SPEC_RELATIVE_PATH
    bad_spec.parent.mkdir(parents=True, exist_ok=True)
    bad_spec.write_text("REQ-CAPSTONE-4859\n", encoding="utf-8")
    bad_summarizer = bad_registry_root / mod.SUMMARIZER_RELATIVE_PATH
    bad_summarizer.parent.mkdir(parents=True, exist_ok=True)
    bad_summarizer.write_text("# placeholder\n", encoding="utf-8")

    registry_blocked = mod.run_capstone(root=bad_registry_root, summarizer=summarizer)
    assert registry_blocked["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert registry_blocked["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(registry_blocked) == []


def test_validate_artifact_rejects_schema_violations_and_helpers() -> None:
    """SCENARIO-CAPSTONE-4859: malformed scorecards fail validation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    critical_summary = mod.SummarizerResult(["summarize"], 2, "CRITICAL", "")

    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "not terminal"}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_model"}
    )
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "65"}
    )
    assert "missing_principle:honest_verdict" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert "invalid_a1_generation_wall_verdict" in mod.validate_artifact(
        {**artifact, "a1_generation_wall_verdict": {"verdict": "maybe"}}
    )
    assert "invalid_scored_lever_state" in mod.validate_artifact(
        {**artifact, "scored_lever_state": {"level_up_banked": "no"}}
    )
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4851}]}
    )
    assert "invalid_flagged_artifacts_skipped" in mod.validate_artifact(
        {**artifact, "flagged_artifacts_skipped": [{"experiment_id": 4856}]}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert mod._experiment_id("A1", {"experiment_id": True}) == 4851
    assert mod._experiment_id("HARDWARE", {"experiment": 4857}) == 4857
    assert mod._is_skipped(_a1(flagged=True), None) == "flagged_adversarial"
    assert mod._is_skipped(_a1(), critical_summary) == "live_critical_recheck"
    assert mod._a1_generation_wall_verdict(None, None, None, None) == {}
    assert mod._levelup_bank(None) == {}
    assert mod._self_play_checkpoint(None) == {}
    assert mod._heldout_readiness(None) == {}
    assert mod._submission_package_state(None) == {}
    assert mod._hardware_continuity(None) == {}
    assert mod._sota_handoff(None) == {}
    assert mod._next_pivot("ENUMERATED_BUT_LOST") == "widen_search_budget_pruner"
    assert mod._next_pivot("COVERED") == "revive_learned_verifier_ranker"
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
    ) == "spec_missing_req_4859"
