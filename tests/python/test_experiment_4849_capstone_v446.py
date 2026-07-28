"""Tests for REQ-CAPSTONE-4849 / SCENARIO-CAPSTONE-4849."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4849_capstone_v446 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1(*, recovered_games: tuple[str, ...] = ("r11l",), real_frames: bool = True) -> JsonDict:
    per_game = {
        "lp85": {
            "shape_motion_score": 0.892323,
            "color_centroid_baseline_score": 0.842473,
            "n_frames": 53,
            "n_transition_pairs": 52,
            "recovered": "lp85" in recovered_games,
            "source_kind": "banked_replay" if real_frames else "synthetic",
            "source_path": "results/arc3_live_banked_trajectories/lp85.json",
            "frame_checksum": "sha256:lp85",
        },
        "r11l": {
            "shape_motion_score": 0.596154,
            "color_centroid_baseline_score": 0.403846,
            "n_frames": 5,
            "n_transition_pairs": 4,
            "recovered": "r11l" in recovered_games,
            "source_kind": "banked_replay" if real_frames else "synthetic",
            "source_path": "results/arc3_live_banked_trajectories/r11l.json",
            "frame_checksum": "sha256:r11l",
        },
        "tu93": {
            "shape_motion_score": 0.924274,
            "color_centroid_baseline_score": 0.857884,
            "n_frames": 94,
            "n_transition_pairs": 93,
            "recovered": "tu93" in recovered_games,
            "source_kind": "banked_replay" if real_frames else "synthetic",
            "source_path": "results/arc3_live_banked_trajectories/tu93.json",
            "frame_checksum": "sha256:tu93",
        },
    }
    return {
        "experiment": "experiment_4841_object_identity_perception_probe",
        "honest_verdict": (
            "success_object_identity_perception_recovers_goal_grounding"
            if len(recovered_games) >= 2
            else "complete_object_identity_unrecoverable_from_rendered_grid_deeper_finding"
        ),
        "measured_on_real_frames": real_frames,
        "per_game_correspondence": per_game,
        "positive_control_tu93_passed": True,
        "positive_control_tu93": {
            "passed": True,
            "player_track_id": 9,
            "goal_track_id": 39,
            "player_motion": 220.624731,
            "goal_persistence": 0.5,
        },
        "games_with_recovery": len(recovered_games),
        "verifier_is_oracle": True,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _levelup(*, banked: bool = False) -> JsonDict:
    after = 66 if banked else 65
    return {
        "experiment": "experiment_4842_levelup_attempt",
        "honest_verdict": (
            "success_ka59_new_level_banked"
            if banked
            else "complete_ka59_no_new_level_residual_existing_depth"
        ),
        "target_game": "ka59",
        "new_levels_banked": 1 if banked else 0,
        "offline_reproduced": banked,
        "reproduced_levels": 2 if banked else 0,
        "registry_update": {
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": after,
            "updated": banked,
            "reason": "banked" if banked else "no_new_level_banked",
        },
        "attempted_games": [{"game": "ka59", "reached_level": 2 if banked else 1}],
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": True,
        "inference_substrate": "adapter_search_only_no_induction",
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4843_self_play_verifier_checkpoint",
        "honest_verdict": "success_re86_L2_checkpoint_refreshed",
        "target_game": "re86",
        "verifier_checkpoint_refreshed": True,
        "self_play_residual": "checkpoint_refreshed_gate_passed",
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "reproduction_gate": {"reproduced": True, "game": "re86", "claimed_level": 2},
        "checkpoint_path": "models/arc_verifier_re86.json",
        "checkpoint_mtime_delta_ns": 42,
        "search_state_count": 56,
        "solve_provenance": "live_agent_self_discovery",
        "inference_substrate": "live_llm_inference",
    }


def _heldout(*, rate: float = 0.04) -> JsonDict:
    return {
        "experiment": "experiment_4844_heldout_first_win_readiness",
        "experiment_id": 4844,
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": rate,
        "first_win_baseline": 0.04,
        "prior_best_heldout_first_win_rate": 0.04,
        "heldout_first_win_delta_vs_baseline": round(rate - 0.04, 6),
        "heldout_first_win_delta_vs_prior_best": round(rate - 0.04, 6),
        "heldout_variant_attempts": 100,
        "positive_control_passed": True,
        "parity_test_green": True,
        "null_delta_methodology_note": "flat no-improvement result",
        "live_agent_ran": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _b1(
    *,
    exercised: bool = True,
    real_frames: bool = True,
    tracker_changed: bool = True,
    positive_control: bool = True,
) -> JsonDict:
    nonzero_games = ["lp85", "r11l", "tu93"] if tracker_changed else []
    deltas = {"lp85": 0.04985, "r11l": 0.192308, "tu93": 0.06639}
    if not tracker_changed:
        deltas = {"lp85": 0.0, "r11l": 0.0, "tu93": 0.0}
    return {
        "experiment": "experiment_4845_perception_probe_audit",
        "experiment_id": 4845,
        "honest_verdict": (
            "complete_a1_perception_probe_audit_genuinely_exercised"
            if exercised
            else "complete_a1_perception_probe_non_test_synthetic_or_degenerate"
        ),
        "a1_genuinely_exercised": exercised,
        "non_test_reasons": [] if exercised else ["synthetic_only_or_degenerate"],
        "recovered_games_from_rows": 1,
        "claimed_recovery_matches_rows": True,
        "source_artifact_checksum": "sha256:a1",
        "checks": {
            "measured_on_real_frames": {
                "passed": real_frames,
                "artifact_measured_on_real_frames": real_frames,
                "bad_games": [] if real_frames else ["lp85"],
                "target_rows": {
                    game: {
                        "present": True,
                        "real_frame_backed": real_frames,
                        "source_kind": "banked_replay" if real_frames else "synthetic",
                        "n_frames": 5,
                        "enough_frames": True,
                    }
                    for game in ("lp85", "r11l", "tu93")
                },
            },
            "tracker_changed_vs_baseline": {
                "passed": tracker_changed,
                "deltas": deltas,
                "distinct_delta_count": 3 if tracker_changed else 0,
                "nonzero_delta_games": nonzero_games,
                "missing_numeric_games": [],
            },
            "positive_control_and_recovery_claim": {
                "passed": positive_control,
                "positive_control_passed": positive_control,
                "claimed_recovery_matches_rows": True,
                "recovered_count": 1,
                "recovered_games": ["r11l"],
                "verdict_matches_numbers": True,
                "success_claimed": False,
                "should_be_success": False,
            },
            "summarizer_and_adversarial_verify": {
                "passed": True,
                "summarizer_returncode": 0,
                "adversarial_flag_count": 0,
                "adversarial_loaded": True,
            },
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _package(*, ready: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_4846_submission_package_harden",
        "honest_verdict": "success_package_builds_vram_gate_green"
        if ready
        else "complete_package_not_ready",
        "submission_package_ready": ready,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"dry_build_ran": ready, "package_builds": ready},
        "packaging_requirements_crosscheck": {"ok": ready},
        "agent_config_resolution": {"resolved": ready, "model_id": "unsloth/gemma-4-31B-it-GGUF"},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 4847,
        "honest_verdict": "success: kv260_continuity_recorded",
        "kv260_ssh_reachable": True,
        "board_state": {"captured": True, "hostname": "kv260", "uio_device_count": 5},
        "next_forward_step": "continue SSH-only continuity checks",
        "inference_substrate": "hardware_smoke",
    }


def _sota() -> JsonDict:
    return {
        "experiment": "experiment_4848_sota_ingestion_object_world_model",
        "honest_verdict": "success_sota_ingestion_object_world_model_mapped",
        "flagged_for_v447": [
            {"candidate": "comet_object_mcts_planner"},
            {"candidate": "slot_mpc_object_action_optimizer"},
        ],
        "methods_mapped": [{"method": "Object-relation transition graph proposer"}],
        "arxiv_ids_cited": ["2606.14418"],
        "a1_perception_layer_input": {"target_roadmap": ".447"},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(
    *,
    a1: JsonDict | None = None,
    b1: JsonDict | None = None,
    levelup: JsonDict | None = None,
    heldout: JsonDict | None = None,
    package: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "A1": a1 or _a1(),
        "LEVELUP": levelup or _levelup(),
        "SELF_PLAY": _self_play(),
        "HELDOUT": heldout or _heldout(),
        "B1_AUDIT": b1 or _b1(),
        "PACKAGE": package or _package(),
        "HARDWARE": _hardware(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {key: f"sha256:{key.lower()}" for key in mod.UPSTREAM_SOURCES}


def _summaries(
    *,
    a1_code: int = 0,
    heldout_code: int = 1,
) -> dict[str, mod.SummarizerResult]:
    summaries = {
        key: mod.SummarizerResult(["summarize", spec.relative_path], 0, "LIVE re-check: clean", "")
        for key, spec in mod.UPSTREAM_SOURCES.items()
    }
    summaries["A1"] = mod.SummarizerResult(
        ["summarize", mod.UPSTREAM_SOURCES["A1"].relative_path],
        a1_code,
        "LIVE re-check: CRITICAL" if a1_code >= 2 else "LIVE re-check: clean",
        "",
    )
    summaries["HELDOUT"] = mod.SummarizerResult(
        ["summarize", mod.UPSTREAM_SOURCES["HELDOUT"].relative_path],
        heldout_code,
        "LIVE re-check: warn declared null delta",
        "",
    )
    return summaries


def test_req_capstone_4849_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4849: OpenSpec declares the .446 scorecard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4849_default_a1_is_genuine_rendered_grid_null() -> None:
    """SCENARIO-CAPSTONE-4849: one recovered game with B1 checks is a genuine null."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    verdict = artifact["a1_perception_probe_verdict"]
    scored = artifact["scored_lever_state"]
    assert artifact["honest_verdict"] == (
        "complete_a1_object_identity_genuine_null_rendered_grid_unrecoverable_capstone_ready"
    )
    assert verdict["verdict"] == "genuine_null_object_identity_unrecoverable_from_rendered_grid"
    assert verdict["measured_on_real_frames_confirmed_by_b1"] is True
    assert verdict["tracker_not_baseline_noop"] is True
    assert verdict["positive_control_tu93_passed"] is True
    assert verdict["games_with_recovery"] == 1
    assert verdict["recovered_games"] == ["r11l"]
    assert verdict["goal_grounding_feasible"] is False
    assert artifact["reproducible_total_levels"] == 65
    assert scored == {
        "level_up_banked": False,
        "heldout_first_win_rate": 0.04,
        "submission_package_ready": True,
    }
    assert artifact["levelup_bank"]["reproducible_total_levels_delta"] == 0
    assert artifact["heldout_readiness"]["decision"] == "flat_baseline_first_win_null"
    assert artifact["self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert artifact["submission_package_state"]["decision"] == "package_ready_operator_only"
    assert artifact["hardware_continuity"]["decision"] == "kv260_reachable"
    assert artifact["sota_handoff"]["flagged_for_v447_candidates"] == [
        "comet_object_mcts_planner",
        "slot_mpc_object_action_optimizer",
    ]
    assert artifact["cited_upstream_artifacts"][0]["fields_imported"] == list(
        mod.CLEAN_IMPORT_FIELDS["A1"]
    )
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4849_recovery_and_non_test_paths() -> None:
    """SCENARIO-CAPSTONE-4849: A1 recovery needs >=2 games and B1 real/no-op checks."""

    recovered = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(recovered_games=("lp85", "r11l")),
            b1={
                **_b1(),
                "recovered_games_from_rows": 2,
                "checks": {
                    **_b1()["checks"],
                    "positive_control_and_recovery_claim": {
                        **_b1()["checks"]["positive_control_and_recovery_claim"],
                        "recovered_count": 2,
                        "recovered_games": ["lp85", "r11l"],
                        "success_claimed": True,
                        "should_be_success": True,
                    },
                },
            },
            levelup=_levelup(banked=True),
            heldout=_heldout(rate=0.08),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 66},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    audit_non_test = mod.build_artifact(
        artifacts=_artifacts(b1=_b1(exercised=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    synthetic_only = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(real_frames=False), b1=_b1(real_frames=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    baseline_noop = mod.build_artifact(
        artifacts=_artifacts(b1=_b1(tracker_changed=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    positive_control_failed = mod.build_artifact(
        artifacts=_artifacts(b1=_b1(positive_control=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    recovery_claim_mismatch = mod.build_artifact(
        artifacts=_artifacts(
            b1={
                **_b1(),
                "claimed_recovery_matches_rows": False,
                "checks": {
                    **_b1()["checks"],
                    "positive_control_and_recovery_claim": {
                        **_b1()["checks"]["positive_control_and_recovery_claim"],
                        "claimed_recovery_matches_rows": False,
                    },
                },
            }
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    live_path_unreachable = mod.build_artifact(
        artifacts=_artifacts(a1={**_a1(), "live_path_reachable": False}),
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
        summarizer_results=_summaries(a1_code=2),
        duration_s=0.001,
    )

    assert recovered["honest_verdict"] == (
        "success_a1_object_identity_recovered_goal_grounding_feasible_capstone_ready"
    )
    assert recovered["a1_perception_probe_verdict"]["goal_grounding_feasible"] is True
    assert recovered["scored_lever_state"] == {
        "level_up_banked": True,
        "heldout_first_win_rate": 0.08,
        "submission_package_ready": True,
    }
    assert audit_non_test["a1_perception_probe_verdict"]["reason"] == "b1_audit_non_test"
    assert synthetic_only["a1_perception_probe_verdict"]["reason"] == "not_measured_on_real_frames"
    assert baseline_noop["a1_perception_probe_verdict"]["reason"] == "tracker_baseline_noop"
    assert positive_control_failed["a1_perception_probe_verdict"]["reason"] == (
        "positive_control_failed"
    )
    assert recovery_claim_mismatch["a1_perception_probe_verdict"]["reason"] == (
        "recovery_claim_mismatch"
    )
    assert live_path_unreachable["a1_perception_probe_verdict"]["reason"] == (
        "live_path_unreachable"
    )
    assert live_critical["a1_perception_probe_verdict"]["reason"] == "live_critical_recheck"
    assert live_critical["flagged_artifacts_skipped"][0]["source"] == "A1"


def test_run_capstone_invokes_summarizer_for_every_upstream_and_writes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4849: runtime aggregation reads every upstream via summarizer."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4849\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        if "4844" in relative_path:
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
    """SCENARIO-CAPSTONE-4849-BLOCKED-PRECONDITION: missing upstreams fail closed."""

    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4849\n", encoding="utf-8")
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
    assert artifact["a1_perception_probe_verdict"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations_and_helpers_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4849: malformed scorecards fail validation."""

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
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4841}]}
    )
    assert "invalid_a1_perception_probe_verdict" in mod.validate_artifact(
        {**artifact, "a1_perception_probe_verdict": {"verdict": "maybe"}}
    )
    assert "invalid_scored_lever_state" in mod.validate_artifact(
        {**artifact, "scored_lever_state": {"level_up_banked": "no"}}
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
    assert mod._experiment_id("A1", {"experiment_id": True}) == 4841
    assert mod._experiment_id("HARDWARE", {"experiment": 4847}) == 4847
    assert set(mod._per_game_summary({"per_game_correspondence": {"lp85": {}}})) == set()
    assert mod._target_rows_real({"passed": False, "target_rows": {}}) is False
    assert (
        mod._target_rows_real(
            {
                "passed": True,
                "target_rows": {
                    "lp85": {"present": False, "real_frame_backed": True},
                    "r11l": {"present": True, "real_frame_backed": True},
                    "tu93": {"present": True, "real_frame_backed": True},
                },
            }
        )
        is False
    )
    assert (
        mod._target_rows_real(
            {
                "passed": True,
                "target_rows": {
                    game: {
                        "present": True,
                        "real_frame_backed": True,
                        "source_kind": "synthetic" if game == "tu93" else "banked_replay",
                    }
                    for game in ("lp85", "r11l", "tu93")
                },
            }
        )
        is False
    )
    assert mod._a1_perception_verdict(None, None, None) == {}
    assert mod._levelup_bank(None) == {}
    assert mod._self_play_checkpoint(None) == {}
    assert mod._heldout_readiness(None) == {}
    assert mod._submission_package_state(None) == {}
    assert mod._hardware_continuity(None) == {}
    assert mod._sota_handoff(None) == {}
    assert mod._imported_fields("A1", _a1(), critical_summary) == ["live_critical_recheck"]
    assert mod._flagged_artifacts_skipped(
        {"A1": _a1()},
        {"A1": "sha256:a1"},
        {"A1": critical_summary},
    ) == [
        {
            "source": "A1",
            "experiment_id": 4841,
            "path": mod.UPSTREAM_SOURCES["A1"].relative_path,
            "reason": "live_critical_recheck",
            "sha256": "sha256:a1",
        }
    ]
    assert mod._cited_artifacts({"A1": None, "LEVELUP": _levelup()}, {}, {}) == [
        {
            "experiment_id": 4842,
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
        == "spec_missing_req_4849"
    )

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4849\n", encoding="utf-8")
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
