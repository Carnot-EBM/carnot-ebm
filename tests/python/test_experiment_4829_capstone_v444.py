"""Tests for REQ-CAPSTONE-4829 / SCENARIO-CAPSTONE-4829."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4829_capstone_v444 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _s3(
    *,
    verdict: str = "complete_structural_energy_s3_bounded_no_generation_lift",
    delta: float = 0.0,
    ci: list[float] | None = None,
    headroom: int = 24,
    min_headroom: int = 5,
    oracle: bool = False,
    lambda0: float = 0.0,
    live_path: bool = True,
    positive_control: bool = True,
    new_levels: list[JsonDict] | None = None,
    game_results: list[JsonDict] | None = None,
) -> JsonDict:
    ci = [0.0, 0.0] if ci is None else ci
    if game_results is None:
        game_results = [
            {
                "game": f"g{i:02d}",
                "banked_by_E": False,
                "banked_by_bare": False,
                "positive_control_reachable": True,
                "was_already_in_bare_pool": False,
                "winner_newly_entered_pool": False,
            }
            for i in range(headroom)
        ]
    return {
        "experiment": "experiment_4821_structural_energy_s3_generation_lift",
        "experiment_id": 4821,
        "honest_verdict": verdict,
        "verifier_is_oracle": oracle,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "solve_provenance": "live_agent_self_discovery",
        "lambda0_control": {"lambda": lambda0, "matched_control": lambda0 == 0.0},
        "lambda_guidance": 1.0,
        "live_path_reachable": live_path,
        "positive_control_passed": positive_control,
        "min_headroom_games": min_headroom,
        "n_headroom_games": headroom,
        "new_levels_not_in_bare_pool": [] if new_levels is None else new_levels,
        "winners_newly_entering_pool_delta": delta,
        "winners_newly_entering_pool_delta_ci95": ci,
        "game_results": game_results,
        "source_artifacts": {
            "positive_control": "results/experiment_4640_goal_energy_generation_live.json",
            "matched_generation_measurement": (
                "results/experiment_4737_goal_energy_candidate_generation_valid_test.json"
            ),
        },
    }


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4822_levelup_attempt",
        "honest_verdict": "complete_ka59_no_new_level_residual_existing_depth",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "target_game": "ka59",
        "registry_update": {
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": 65,
            "updated": False,
        },
        "attempted_games": [{"game": "ka59", "reached_level": 1}],
        "dead_ends": ["existing depth only"],
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": True,
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4823_self_play_verifier_checkpoint",
        "honest_verdict": "success_re86_L2_checkpoint_refreshed",
        "verifier_checkpoint_refreshed": True,
        "target_game": "re86",
        "self_play_residual": "checkpoint_refreshed_gate_passed",
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "reproduction_gate": {"reproduced": True, "game": "re86", "claimed_level": 2},
        "solve_provenance": "live_agent_self_discovery",
        "inference_substrate": "live_llm_inference",
    }


def _heldout(*, changed: bool = False) -> JsonDict:
    return {
        "experiment": "experiment_4824_heldout_first_win_readiness",
        "experiment_id": 4824,
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": 0.08 if changed else 0.04,
        "first_win_baseline": 0.04,
        "prior_best_heldout_first_win_rate": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.04 if changed else 0.0,
        "heldout_first_win_delta_vs_prior_best": 0.04 if changed else 0.0,
        "heldout_variant_attempts": 100,
        "positive_control_passed": True,
        "parity_test_green": True,
        "null_delta_methodology_note": "genuine no-improvement result",
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _audit(
    *,
    controls: bool = True,
    matched_lambda0: bool = True,
    not_reranking: bool = True,
    reachable_control: bool = True,
    guidance_exercised: bool = True,
) -> JsonDict:
    return {
        "experiment": "experiment_4825_silent_bug_audit",
        "experiment_id": 4825,
        "honest_verdict": "complete_arc_null_silent_bug_audit_3_nulls_0_reopen",
        "nulls_audited": 3,
        "trusted_nulls": [
            "experiment_4821_structural_energy_s3_generation_lift",
            "experiment_4822_levelup_attempt",
            "experiment_4824_heldout_first_win_readiness",
        ],
        "silent_bugs_found": [],
        "s3_controls_verified": controls,
        "s3_guidance_exercised": guidance_exercised,
        "s3_control_check": {
            "s3_controls_verified": controls,
            "matched_lambda0_control": matched_lambda0,
            "new_levels_not_in_bare_pool": not_reranking,
            "positive_control_passed": reachable_control,
            "s3_guidance_exercised": guidance_exercised,
            "same_games_seeds_budget": True,
            "n_headroom_games": 24,
            "min_headroom_games": 5,
        },
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4826_submission_package_harden",
        "honest_verdict": "success_package_builds_vram_gate_green",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"dry_build_ran": True, "package_builds": True},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _hardware() -> JsonDict:
    return {
        "experiment": "experiment_4827_kv260_continuity",
        "honest_verdict": "blocked_kv260_ssh_unreachable",
        "kv260_ssh_reachable": False,
        "board_state": {"captured": False, "reason": "kv260_ssh_unreachable"},
        "next_forward_step": "restore SSH",
        "verifier_is_oracle": False,
        "inference_substrate": "hardware_smoke",
    }


def _sota() -> JsonDict:
    return {
        "honest_verdict": "success_sota_ingestion_cross_family_transfer_mapped",
        "methods_mapped": [{"method": "Leave-one-family reward/verifier transfer gate"}],
        "flagged_for_v445": [
            {"candidate": "anchor_leave_one_family_transfer_gate"},
            {"candidate": "worst_family_group_dro_s4_energy"},
        ],
        "s4_context": {"s4_cross_family_transfer_required": True, "roadmap_target": ".445"},
        "arxiv_ids_cited": ["2605.25629", "2311.14743"],
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(
    *,
    s3: JsonDict | None = None,
    heldout: JsonDict | None = None,
    audit: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "S3": s3 or _s3(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": _self_play(),
        "HELDOUT": heldout or _heldout(),
        "BUG_AUDIT": audit or _audit(),
        "PACKAGE": _package(),
        "HARDWARE": _hardware(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {key: f"sha256:{key.lower()}" for key in mod.UPSTREAM_SOURCES}


def _summaries(
    *, s3_code: int = 0, s3_text: str = "LIVE re-check: clean"
) -> dict[str, mod.SummarizerResult]:
    summaries = {
        key: mod.SummarizerResult(
            command=["python", "scripts/summarize_artifact.py", spec.relative_path],
            exit_code=0,
            stdout="LIVE re-check: clean",
            stderr="",
        )
        for key, spec in mod.UPSTREAM_SOURCES.items()
    }
    summaries["S3"] = mod.SummarizerResult(
        command=[
            "python",
            "scripts/summarize_artifact.py",
            mod.UPSTREAM_SOURCES["S3"].relative_path,
        ],
        exit_code=s3_code,
        stdout=s3_text,
        stderr="",
    )
    summaries["HELDOUT"] = mod.SummarizerResult(
        command=[
            "python",
            "scripts/summarize_artifact.py",
            mod.UPSTREAM_SOURCES["HELDOUT"].relative_path,
        ],
        exit_code=1,
        stdout="LIVE re-check: warn declared null delta",
        stderr="",
    )
    return summaries


def test_req_capstone_4829_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4829: OpenSpec declares the .444 scorecard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4829_default_s3_is_bounded_generation_null() -> None:
    """SCENARIO-CAPSTONE-4829: clean S3 with zero generation lift is bounded."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    verdict = artifact["s3_structural_energy_verdict"]
    assert artifact["honest_verdict"] == "complete_s3_bounded_no_generation_lift_capstone_ready"
    assert verdict["verdict"] == "bounded_no_generation_lift"
    assert verdict["s4_authorized"] is False
    assert verdict["bounded_no_generation_lift"] is True
    assert verdict["controls_verified_by_b1"] is True
    assert verdict["matched_lambda0_control"] is True
    assert verdict["new_levels_not_re_ranking"] is True
    assert verdict["banked_levels_already_in_bare_pool"] == []
    assert verdict["winners_newly_entering_pool_delta"] == pytest.approx(0.0)
    assert verdict["winners_newly_entering_pool_delta_ci95"] == [0.0, 0.0]
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["levelup_bank"]["moat_claim"] is False
    assert artifact["self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert artifact["heldout_readiness"]["decision"] == "flat_null_no_readiness_gain"
    assert artifact["hardware_continuity"]["decision"] == "blocked_kv260_ssh_unreachable"
    assert artifact["readiness"]["structural_energy_direction"] == (
        "bounded_at_real_offline_discriminator_no_live_value"
    )
    assert artifact["sota_handoff"]["flagged_for_v445_candidates"] == [
        "anchor_leave_one_family_transfer_gate",
        "worst_family_group_dro_s4_energy",
    ]
    assert artifact["cited_upstream_artifacts"][0]["fields_imported"] == [
        "honest_verdict",
        "verifier_is_oracle",
        "live_path_reachable",
        "lambda0_control",
        "lambda_guidance",
        "n_headroom_games",
        "min_headroom_games",
        "positive_control_passed",
        "new_levels_not_in_bare_pool",
        "winners_newly_entering_pool_delta",
        "winners_newly_entering_pool_delta_ci95",
        "game_results",
        "source_artifacts",
        "solve_provenance",
        "inference_substrate",
    ]
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4829_s3_win_and_inconclusive_paths() -> None:
    """SCENARIO-CAPSTONE-4829: S3 distinguishes generation win from invalid lifts."""

    win = mod.build_artifact(
        artifacts=_artifacts(
            s3=_s3(
                verdict="success_structural_energy_s3_generation_authorizes_s4",
                delta=0.18,
                ci=[0.04, 0.29],
                new_levels=[{"game": "r11l", "level": 1}],
                game_results=[
                    {
                        "game": "r11l",
                        "banked_by_E": True,
                        "banked_by_bare": False,
                        "positive_control_reachable": True,
                        "was_already_in_bare_pool": False,
                        "winner_newly_entered_pool": True,
                    }
                ],
            ),
            heldout=_heldout(changed=True),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    no_headroom = mod.build_artifact(
        artifacts=_artifacts(s3=_s3(headroom=3, min_headroom=5)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    controls_unverified = mod.build_artifact(
        artifacts=_artifacts(audit=_audit(controls=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    live_critical = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(s3_code=2, s3_text="LIVE re-check: CRITICAL"),
        duration_s=0.001,
    )
    live_path_unreachable = mod.build_artifact(
        artifacts=_artifacts(s3=_s3(live_path=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    reranking = mod.build_artifact(
        artifacts=_artifacts(
            s3=_s3(
                delta=0.2,
                ci=[0.05, 0.3],
                game_results=[
                    {
                        "game": "r11l",
                        "banked_by_E": True,
                        "banked_by_bare": True,
                        "positive_control_reachable": True,
                        "was_already_in_bare_pool": True,
                        "winner_newly_entered_pool": True,
                    }
                ],
            )
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    oracle = mod.build_artifact(
        artifacts=_artifacts(s3=_s3(delta=0.2, ci=[0.05, 0.3], oracle=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    clean_gate_miss = mod.build_artifact(
        artifacts=_artifacts(s3=_s3(delta=0.0, ci=[0.01, 0.2])),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert win["honest_verdict"] == "success_s3_generation_lift_s4_authorized"
    assert win["s3_structural_energy_verdict"]["verdict"] == "generation_win_s4_authorized"
    assert win["s3_structural_energy_verdict"]["s4_authorized"] is True
    assert win["readiness"]["s4_authorized"] is True
    assert no_headroom["s3_structural_energy_verdict"]["reason"] == "inconclusive_no_generation_headroom"
    assert controls_unverified["s3_structural_energy_verdict"]["reason"] == "s3_controls_unverified"
    assert live_critical["s3_structural_energy_verdict"]["reason"] == "live_critical_recheck"
    assert live_critical["cited_upstream_artifacts"][0]["fields_imported"] == [
        "live_critical_recheck"
    ]
    assert live_path_unreachable["s3_structural_energy_verdict"]["reason"] == (
        "live_path_unreachable"
    )
    assert reranking["s3_structural_energy_verdict"]["reason"] == (
        "banked_level_already_in_bare_pool"
    )
    assert reranking["s3_structural_energy_verdict"]["banked_levels_already_in_bare_pool"] == [
        "r11l"
    ]
    assert oracle["s3_structural_energy_verdict"]["reason"] == "oracle_not_moat"
    assert clean_gate_miss["s3_structural_energy_verdict"]["reason"] == (
        "s3_generation_lift_gate_requirements_not_met"
    )


def test_run_capstone_invokes_summarizer_for_every_upstream_and_writes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4829: runtime aggregation reads every upstream via the summarizer."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4829\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        if "heldout" in relative_path:
            return mod.SummarizerResult(["summarize", relative_path], 1, "warn", "")
        return mod.SummarizerResult(["summarize", relative_path], 0, "clean", "")

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)

    assert sorted(called) == sorted(spec.relative_path for spec in mod.UPSTREAM_SOURCES.values())
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"]["summarizer"]["present"] is True
    assert artifact["preconditions_checked"]["registry"]["yaml_loadable"] is True
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.validate_artifact(artifact) == []


def test_run_capstone_blocks_on_missing_required_upstream(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4829-BLOCKED-PRECONDITION: missing upstreams fail closed."""

    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4829\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    artifact = mod.run_capstone(
        root=tmp_path,
        summarizer=lambda _root, relative_path: mod.SummarizerResult(
            ["summarize", relative_path], 0, "clean", ""
        ),
    )

    assert artifact["honest_verdict"] == "blocked_missing_upstream:SOTA"
    assert artifact["s3_structural_energy_verdict"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations_and_helpers_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4829: malformed scorecards fail validation."""

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
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4821}]}
    )
    assert "invalid_s3_verdict" in mod.validate_artifact(
        {**artifact, "s3_structural_energy_verdict": {"verdict": "maybe"}}
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
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert mod._experiment_id("S3", {"experiment_id": True}) == 4821
    assert mod._int(True, 7) == 7
    assert mod._int("x", 9) == 9
    assert mod._float(True) is None
    assert mod._float("x") is None
    assert mod._mapping("x") == {}
    assert mod._banked_re_ranking_games({"game_results": "bad"}) == []
    assert mod._banked_re_ranking_games({"game_results": ["bad"]}) == []
    assert mod._s3_verdict(None, None, None) == {}
    assert mod._levelup_bank(None) == {}
    assert mod._self_play_checkpoint(None) == {}
    assert mod._heldout_readiness(None) == {}
    assert mod._silent_bug_audit(None) == {}
    assert mod._submission_package_state(None) == {}
    assert mod._hardware_continuity(None) == {}
    assert mod._sota_handoff(None) == {}
    assert mod._imported_fields("S3", _s3(), critical_summary) == ["live_critical_recheck"]
    assert mod._flagged_artifacts_skipped(
        {"HELDOUT": _heldout()},
        {"HELDOUT": "sha256:heldout"},
        {"HELDOUT": critical_summary},
    ) == [
        {
            "source": "HELDOUT",
            "experiment_id": 4824,
            "path": mod.UPSTREAM_SOURCES["HELDOUT"].relative_path,
            "reason": "live_critical_recheck",
            "sha256": "sha256:heldout",
        }
    ]
    assert mod._cited_artifacts({"S3": None, "LEVELUP": _levelup()}, {}, {}) == [
        {
            "experiment_id": 4822,
            "fields_imported": list(mod.CLEAN_IMPORT_FIELDS["LEVELUP"]),
            "sha256": "",
        }
    ]

    upstream_present = {
        key: {"present": True, "path": spec.relative_path}
        for key, spec in mod.UPSTREAM_SOURCES.items()
    }
    assert (
        mod._first_blocker(
            summarizer_present=False,
            registry_present=False,
            registry_loadable=False,
            spec_has_req=False,
            upstream_preconditions=upstream_present,
        )
        == "missing_summarizer"
    )
    assert (
        mod._first_blocker(
            summarizer_present=True,
            registry_present=False,
            registry_loadable=False,
            spec_has_req=False,
            upstream_preconditions=upstream_present,
        )
        == "missing_registry"
    )
    assert (
        mod._first_blocker(
            summarizer_present=True,
            registry_present=True,
            registry_loadable=False,
            spec_has_req=False,
            upstream_preconditions=upstream_present,
        )
        == "registry_not_yaml_loadable"
    )
    assert (
        mod._first_blocker(
            summarizer_present=True,
            registry_present=True,
            registry_loadable=True,
            spec_has_req=False,
            upstream_preconditions=upstream_present,
        )
        == "spec_missing_req_4829"
    )

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4829\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    blocked = mod.run_capstone(
        root=tmp_path,
        summarizer=lambda _root, relative_path: mod.SummarizerResult(
            ["summarize", relative_path], 0, "clean", ""
        ),
    )

    assert blocked["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert blocked["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(blocked) == []
