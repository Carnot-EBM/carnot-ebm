"""Tests for REQ-CAPSTONE-4839 / SCENARIO-CAPSTONE-4839."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4839_capstone_v445 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1(
    *,
    verdict: str = "complete_amortized_prior_no_first_win_lift_l1_wall_survives",
    archive_alive: bool = True,
    prior_changed: bool = True,
    with_prior: float = 0.0,
    no_prior: float = 0.0,
    ci: JsonDict | None = None,
    heldout_not_distilled: bool = True,
    imitation_lift_holds: bool = False,
    live_path: bool = True,
) -> JsonDict:
    return {
        "experiment": "experiment_4831_amortized_incontext_exploration_prior_live",
        "experiment_id": 4831,
        "honest_verdict": verdict,
        "go_explore_archive_alive": {
            "alive": archive_alive,
            "observations": 2 if archive_alive else 0,
            "stored_cells": 2 if archive_alive else 0,
            "prefixes_injected": 1 if archive_alive else 0,
            "actions_injected": 1 if archive_alive else 0,
            "verifier_is_oracle": False,
        },
        "prior_changed_proposals": prior_changed,
        "prior_change_diagnostics": {
            "changed": prior_changed,
            "no_prior_order": [1, 2, 3],
            "with_prior_order": [1, 3, 2] if prior_changed else [1, 2, 3],
        },
        "prior_diagnostics": {
            "proposal_changes": 1 if prior_changed else 0,
            "rank_calls": 12,
            "context_hits": 12,
            "distillation_mode": "in_context_exploration_prior",
            "game_id_features_used": False,
        },
        "baseline_first_win_rate": 0.04,
        "first_win_rate_with_prior": with_prior,
        "first_win_rate_no_prior_ablation": no_prior,
        "first_win_delta_ci95": ci or {"low": 0.0, "high": 0.0, "confidence": 0.95},
        "imitation_control_heldout_games": {
            "distillation_games": ["cd82", "cn04"],
            "heldout_games": ["bp35"],
            "heldout_not_in_distillation_set": heldout_not_distilled,
            "first_win_rate_with_prior": with_prior,
            "first_win_rate_no_prior_ablation": no_prior,
            "lift_holds": imitation_lift_holds,
        },
        "live_path_reachable": live_path,
        "solve_provenance": "live_agent_self_discovery",
        "inference_substrate": "live_llm_inference",
    }


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4832_levelup_attempt",
        "honest_verdict": "complete_ka59_no_new_level_residual_existing_depth",
        "target_game": "ka59",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "registry_update": {
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": 65,
            "updated": False,
        },
        "attempted_games": [
            {
                "game": "ka59",
                "reached_level": 1,
                "offline_reproduced_existing_depth": True,
                "offline_reproduced_new_depth": False,
            }
        ],
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": True,
        "inference_substrate": "adapter_search_only_no_induction",
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4833_self_play_verifier_checkpoint",
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
        "experiment": "experiment_4834_heldout_first_win_readiness",
        "experiment_id": 4834,
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": rate,
        "first_win_baseline": 0.04,
        "prior_best_heldout_first_win_rate": 0.04,
        "heldout_first_win_delta_vs_baseline": round(rate - 0.04, 6),
        "heldout_first_win_delta_vs_prior_best": round(rate - 0.04, 6),
        "heldout_variant_attempts": 100,
        "positive_control_passed": True,
        "parity_test_green": True,
        "null_delta_methodology_note": "genuine no-improvement result",
        "a1_amortized_prior_decision": {
            "exists": True,
            "passed": False,
            "included_in_measurement": False,
            "reason": "a1_prior_not_passed",
            "source_artifact_path": mod.UPSTREAM_SOURCES["A1"].relative_path,
        },
        "live_agent_ran": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _b1(
    *,
    archive_alive: bool = True,
    prior_changed: bool = True,
    imitation_control_confirmed: bool = True,
    imitation_lift_holds: bool = False,
    silent_bugs: list[str] | None = None,
) -> JsonDict:
    silent_bugs = [] if silent_bugs is None else silent_bugs
    return {
        "experiment": "experiment_4835_silent_bug_audit",
        "experiment_id": 4835,
        "honest_verdict": "complete_arc_null_silent_bug_audit_3_nulls_0_reopen",
        "nulls_audited": 3,
        "trusted_nulls": [
            "experiment_4831_amortized_incontext_exploration_prior_live",
            "experiment_4832_levelup_attempt",
            "experiment_4834_heldout_first_win_readiness",
        ],
        "silent_bugs_found": silent_bugs,
        "a1_archive_alive_and_prior_exercised": archive_alive and prior_changed,
        "a1_control_check": {
            "archive_alive": archive_alive,
            "observations": 2 if archive_alive else 0,
            "stored_cells": 2 if archive_alive else 0,
            "prefixes_injected": 1 if archive_alive else 0,
            "prior_changed": prior_changed,
            "proposal_order_changed": prior_changed,
            "proposal_changes": 1 if prior_changed else 0,
            "heldout_not_in_distillation_set": imitation_control_confirmed,
            "imitation_control_confirmed": imitation_control_confirmed,
            "imitation_lift_holds": imitation_lift_holds,
            "first_win_rate_with_prior": 0.0,
            "first_win_rate_no_prior_ablation": 0.0,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4836_submission_package_harden",
        "honest_verdict": "success_package_builds_vram_gate_green",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "a1_prior_inclusion": {"included": False, "reason": "not_included_a1_prior_did_not_pass"},
        "vram_estimate_gb": 15.146,
        "package_builds": {"dry_build_ran": True, "package_builds": True},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 4837,
        "honest_verdict": "success: kv260_continuity_recorded",
        "kv260_ssh_reachable": True,
        "board_state": {"captured": True, "hostname": "kv260", "uio_device_count": 4},
        "next_forward_step": "continue SSH-only continuity checks",
        "inference_substrate": "hardware_smoke",
    }


def _sota() -> JsonDict:
    return {
        "honest_verdict": "success_sota_ingestion_perception_representation_mapped",
        "l1_wall_context": {
            "wall": "L1-FIRST-CONTACT",
            "root_cause": "perception/representation",
            "exploration_strategy_class_retired": True,
            "roadmap_target": ".446",
        },
        "flagged_for_v446": [
            {"candidate": "loop_owm_slot_transition_proposer"},
            {"candidate": "object_relational_world_model_mcts"},
        ],
        "methods_mapped": [{"method": "Slotized ARC object-state proposal binder"}],
        "arxiv_ids_cited": ["2606.12316"],
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(
    *,
    a1: JsonDict | None = None,
    b1: JsonDict | None = None,
    heldout: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "A1": a1 or _a1(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": _self_play(),
        "HELDOUT": heldout or _heldout(),
        "B1_AUDIT": b1 or _b1(),
        "PACKAGE": _package(),
        "HARDWARE": _hardware(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {key: f"sha256:{key.lower()}" for key in mod.UPSTREAM_SOURCES}


def _summaries(*, a1_code: int = 1, a1_text: str = "LIVE re-check: warn") -> dict[str, mod.SummarizerResult]:
    summaries = {
        key: mod.SummarizerResult(["summarize", spec.relative_path], 0, "LIVE re-check: clean", "")
        for key, spec in mod.UPSTREAM_SOURCES.items()
    }
    summaries["A1"] = mod.SummarizerResult(
        ["summarize", mod.UPSTREAM_SOURCES["A1"].relative_path], a1_code, a1_text, ""
    )
    summaries["HELDOUT"] = mod.SummarizerResult(
        ["summarize", mod.UPSTREAM_SOURCES["HELDOUT"].relative_path],
        1,
        "LIVE re-check: warn declared null delta",
        "",
    )
    return summaries


def test_req_capstone_4839_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4839: OpenSpec declares the .445 scorecard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4839_default_a1_is_genuine_null() -> None:
    """SCENARIO-CAPSTONE-4839: archive-alive exercised A1 null closes exploration priors."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    verdict = artifact["a1_amortized_prior_verdict"]
    assert artifact["honest_verdict"] == (
        "complete_a1_genuine_null_l1_wall_survives_exploration_prior_closed_capstone_ready"
    )
    assert verdict["verdict"] == "genuine_null_l1_wall_survives_exploration_prior_closed"
    assert verdict["archive_alive"] is True
    assert verdict["prior_exercised_confirmed_by_b1"] is True
    assert verdict["imitation_control_confirmed"] is True
    assert verdict["first_win_rate_with_prior"] == pytest.approx(0.0)
    assert verdict["baseline_first_win_rate"] == pytest.approx(0.04)
    assert verdict["wall_moves"] is False
    assert verdict["exploration_prior_class_closed"] is True
    assert artifact["levelup_bank"]["reproducible_total_levels_delta"] == 0
    assert artifact["self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert artifact["heldout_readiness"]["decision"] == "flat_baseline_first_win_null"
    assert artifact["readiness"]["v446_frontier"] == "perception/representation"
    assert artifact["submission_package_state"]["decision"] == "package_ready_operator_only"
    assert artifact["hardware_continuity"]["decision"] == "kv260_reachable"
    assert artifact["sota_handoff"]["flagged_for_v446_candidates"] == [
        "loop_owm_slot_transition_proposer",
        "object_relational_world_model_mcts",
    ]
    assert artifact["cited_upstream_artifacts"][0]["fields_imported"] == list(
        mod.CLEAN_IMPORT_FIELDS["A1"]
    )
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4839_a1_lift_and_non_test_paths() -> None:
    """SCENARIO-CAPSTONE-4839: A1 lift needs B1 controls; no-op/archive/imitation fail."""

    lift = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(
                verdict="success_amortized_prior_raises_first_win_above_baseline",
                with_prior=0.08,
                no_prior=0.04,
                ci={"low": 0.01, "high": 0.09, "confidence": 0.95},
                imitation_lift_holds=True,
            ),
            b1=_b1(imitation_lift_holds=True),
            heldout=_heldout(rate=0.08),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(a1_code=0, a1_text="LIVE re-check: clean"),
        duration_s=0.001,
    )
    dead_archive = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(archive_alive=False), b1=_b1(archive_alive=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    prior_noop = mod.build_artifact(
        artifacts=_artifacts(a1=_a1(prior_changed=False), b1=_b1(prior_changed=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    imitation_only = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(
                verdict="success_amortized_prior_raises_first_win_above_baseline",
                with_prior=0.08,
                no_prior=0.04,
                ci={"low": 0.01, "high": 0.09, "confidence": 0.95},
                heldout_not_distilled=False,
                imitation_lift_holds=True,
            ),
            b1=_b1(imitation_control_confirmed=False, imitation_lift_holds=True),
        ),
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
        summarizer_results=_summaries(a1_code=2, a1_text="LIVE re-check: CRITICAL"),
        duration_s=0.001,
    )

    assert lift["honest_verdict"] == "success_a1_amortized_prior_lift_wall_moves_capstone_ready"
    assert lift["a1_amortized_prior_verdict"]["wall_moves"] is True
    assert lift["readiness"]["ready_for_operator_submit"] is True
    assert dead_archive["a1_amortized_prior_verdict"]["reason"] == "dead_go_explore_archive"
    assert prior_noop["a1_amortized_prior_verdict"]["reason"] == "prior_noop_not_exercised"
    assert imitation_only["a1_amortized_prior_verdict"]["reason"] == "imitation_control_failed"
    assert live_critical["a1_amortized_prior_verdict"]["reason"] == "live_critical_recheck"
    assert live_critical["flagged_artifacts_skipped"][0]["source"] == "A1"


def test_run_capstone_invokes_summarizer_for_every_upstream_and_writes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4839: runtime aggregation reads every upstream via summarizer."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4839\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        if "4831" in relative_path or "4834" in relative_path:
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
    """SCENARIO-CAPSTONE-4839-BLOCKED-PRECONDITION: missing upstreams fail closed."""

    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4839\n", encoding="utf-8")
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
    assert artifact["a1_amortized_prior_verdict"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations_and_helpers_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4839: malformed scorecards fail validation."""

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
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4831}]}
    )
    assert "invalid_a1_verdict" in mod.validate_artifact(
        {**artifact, "a1_amortized_prior_verdict": {"verdict": "maybe"}}
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
    assert mod._experiment_id("A1", {"experiment_id": True}) == 4831
    assert mod._experiment_id("HARDWARE", {"experiment": 4837}) == 4837
    assert mod._int(True, 7) == 7
    assert mod._int("x", 9) == 9
    assert mod._float(True) is None
    assert mod._float("x") is None
    assert mod._mapping("x") == {}
    assert mod._ci_low_positive({"low": "bad"}) is False
    assert mod._a1_verdict(None, None, None) == {}
    assert mod._levelup_bank(None) == {}
    assert mod._self_play_checkpoint(None) == {}
    assert mod._heldout_readiness(None) == {}
    assert mod._silent_bug_audit(None) == {}
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
            "experiment_id": 4831,
            "path": mod.UPSTREAM_SOURCES["A1"].relative_path,
            "reason": "live_critical_recheck",
            "sha256": "sha256:a1",
        }
    ]
    assert mod._cited_artifacts({"A1": None, "LEVELUP": _levelup()}, {}, {}) == [
        {
            "experiment_id": 4832,
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
        == "spec_missing_req_4839"
    )

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4839\n", encoding="utf-8")
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
