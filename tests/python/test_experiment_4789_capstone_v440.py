"""Tests for REQ-CAPSTONE-4789 / SCENARIO-CAPSTONE-4789."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4789_capstone_v440 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _s1(*, flagged: bool = False, loo: float = 0.7134961314270525, oracle: bool = False) -> JsonDict:
    artifact: JsonDict = {
        "experiment": "experiment_4781_structural_energy_s1_contrastive_landscape",
        "experiment_id": 4781,
        "honest_verdict": "success_structural_energy_s1_landscape_authorizes_s2",
        "verifier_is_oracle": oracle,
        "s1_gate_passed": True,
        "s2_authorized": True,
        "energy_ranking_loo_auroc_mean": loo,
        "energy_ranking_loo_auroc_ci95": [0.7133175599984811, 0.7137104171413382],
        "energy_ranking_loo_auroc_per_seed": [loo] * 10,
        "n_seeds": 10,
        "denoising_direction_agreement": 0.6223390275952694,
        "origin_probe_auroc": 0.5,
        "origin_probe": {
            "loo_auroc": 0.5,
            "origin_counts": {"induced": 463},
            "status": "origin_matched_single_origin_all_induced",
        },
        "shuffled_label_control_auroc": 0.49335645814441664,
        "controls": {
            "majority_class_control_auroc": 0.5,
            "shuffled_label_resamples": 16,
            "v2_frame_marginal_energy_ranking_loo_auroc_mean": 0.48397091893626004,
        },
        "per_family_loo": {
            "frame_delta": 0.7262429748613959,
            "object_relational": 0.6602487486471862,
        },
        "per_game_loo": {"structural": {"dc22": 0.8958333333333334}},
        "n_candidate_rows": 463,
        "n_pos": 186,
        "n_neg": 277,
        "n_held_out_games": 16,
        "random_seeds_used": list(range(4781, 4791)),
    }
    if flagged:
        artifact["flagged_adversarial"] = True
    return artifact


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4782_levelup_attempt",
        "honest_verdict": "complete_lf52_no_new_level_residual_existing_depth",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "target_game": "lf52",
        "registry_update": {
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": 65,
            "updated": False,
        },
        "attempted_games": [{"game": "lf52", "reached_level": 1}],
        "dead_ends": ["lf52 residual existing depth only"],
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": True,
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4783_self_play_verifier_checkpoint",
        "honest_verdict": "success_re86_L2_checkpoint_refreshed",
        "verifier_checkpoint_refreshed": True,
        "target_game": "re86",
        "self_play_residual": "checkpoint_refreshed_gate_passed",
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "reproduction_gate": {"reproduced": True, "game": "re86", "claimed_level": 2},
        "solve_provenance": "development_proxy",
    }


def _heldout() -> JsonDict:
    return {
        "experiment": "experiment_4784_heldout_first_win_readiness",
        "experiment_id": 4784,
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": 0.04,
        "first_win_baseline": 0.04,
        "prior_best_heldout_first_win_rate": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_first_win_delta_vs_prior_best": 0.0,
        "heldout_variant_attempts": 100,
        "positive_control_passed": True,
        "parity_test_green": True,
        "null_delta_methodology_note": "genuine no-improvement result",
    }


def _audit() -> JsonDict:
    checks = {
        "all_controls_fired": False,
        "origin_probe_auroc": 0.5,
        "shuffled_label_control_auroc": 0.49335645814441664,
        "shuffled_label_resamples": 16,
        "denoising_direction_executed": True,
        "denoising_direction_agreement": 0.6223390275952694,
        "seed_floor_met": True,
        "distinct_seed_count": 10,
        "origin_probe_refit_on_origin_matched_data": False,
    }
    return {
        "experiment": "experiment_4785_silent_bug_audit",
        "honest_verdict": "complete_arc_null_silent_bug_audit_3_nulls_1_reopen",
        "nulls_audited": 3,
        "trusted_nulls": [
            "experiment_4782_levelup_attempt",
            "experiment_4784_heldout_first_win_readiness",
        ],
        "silent_bugs_found": [
            {
                "null_id": "experiment_4781_structural_energy_s1_contrastive_landscape",
                "verdict": "silent_bug_must_reopen",
                "silent_bug_signatures": ["s1_origin_probe_not_refit"],
                "s1_controls_fired": False,
                "s1_control_checks": checks,
            }
        ],
        "per_null_verdicts": [],
        "s1_controls_fired": False,
        "s1_control_checks": checks,
        "verifier_is_oracle": False,
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4786_submission_package_harden",
        "honest_verdict": "success_package_builds_vram_gate_green",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"dry_build_ran": True},
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 4787,
        "honest_verdict": "success: kv260_continuity_recorded",
        "kv260_ssh_reachable": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "board_state": {"hostname": "kv260", "uio_device_count": 5},
        "verifier_is_oracle": False,
    }


def _sota() -> JsonDict:
    return {
        "experiment": "experiment_4788_sota_ingestion_energy_guided_search",
        "honest_verdict": "success_sota_ingestion_energy_guided_search_mapped",
        "flagged_for_v441": [
            {"candidate": "energy_value_guided_mcts_frontier_controller"},
            {"candidate": "ebm_poe_planner_for_s3_generation"},
        ],
        "methods_mapped": [{"method": "Energy/value-guided MCTS frontier controller"}],
        "s1_context": {
            "s1_gate_passed": True,
            "s2_authorized": True,
            "energy_ranking_loo_auroc_mean": 0.7134961314270525,
        },
        "arxiv_ids_cited": ["2309.15028", "2502.07202"],
    }


def _artifacts(*, s1: JsonDict | None = None) -> dict[str, JsonDict]:
    return {
        "S1": s1 or _s1(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": _self_play(),
        "HELDOUT": _heldout(),
        "BUG_AUDIT": _audit(),
        "PACKAGE": _package(),
        "HARDWARE": _hardware(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {key: f"sha256:{key.lower()}" for key in mod.UPSTREAM_SOURCES}


def _summaries(*, s1_code: int = 0, heldout_code: int = 1) -> dict[str, mod.SummarizerResult]:
    summaries = {
        key: mod.SummarizerResult(
            command=["python", "scripts/summarize_artifact.py", spec.relative_path],
            exit_code=0,
            stdout="LIVE re-check: clean",
            stderr="",
        )
        for key, spec in mod.UPSTREAM_SOURCES.items()
    }
    summaries["S1"] = mod.SummarizerResult(
        command=["python", "scripts/summarize_artifact.py", mod.UPSTREAM_SOURCES["S1"].relative_path],
        exit_code=s1_code,
        stdout="LIVE re-check: clean" if s1_code == 0 else "LIVE re-check: CRITICAL",
        stderr="",
    )
    summaries["HELDOUT"] = mod.SummarizerResult(
        command=[
            "python",
            "scripts/summarize_artifact.py",
            mod.UPSTREAM_SOURCES["HELDOUT"].relative_path,
        ],
        exit_code=heldout_code,
        stdout="LIVE re-check: warn",
        stderr="",
    )
    return summaries


def test_req_capstone_4789_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4789: OpenSpec declares the .440 scorecard before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4789_imports_stale_flagged_s1_pass() -> None:
    """SCENARIO-CAPSTONE-4789: a live-clean stale conductor flag does not erase S1."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(s1=_s1(flagged=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success_s1_structural_energy_usable_landscape_s2_authorized"
    verdict = artifact["s1_structural_energy_verdict"]
    assert verdict["verdict"] == "usable_landscape"
    assert verdict["s2_authorized"] is True
    assert verdict["energy_ranking_loo_auroc_mean"] == pytest.approx(0.7134961314270525)
    assert verdict["n_seeds"] == 10
    assert verdict["denoising_direction_passed"] is True
    assert verdict["leak_controls_hold"] is True
    assert artifact["stale_false_positive_notes"][0]["source"] == "S1"
    assert artifact["stale_false_positive_notes"][0]["status"] == "stale_false_positive"
    assert artifact["flagged_artifacts_skipped"] == []
    assert "energy_ranking_loo_auroc_mean" in artifact["cited_upstream_artifacts"][0]["fields_imported"]

    assert artifact["levelup_bank"]["target_game"] == "lf52"
    assert artifact["levelup_bank"]["moat_claim"] is False
    assert artifact["self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert artifact["readiness"]["ready_for_operator_submit"] is False
    assert artifact["silent_bug_audit"]["s1_audit_note"] == "audit_note_recorded_does_not_override_live_clean_s1_pass"
    assert artifact["submission_package_state"]["decision"] == "package_ready_operator_only"
    assert artifact["hardware_continuity"]["decision"] == "kv260_reachable"
    assert artifact["sota_handoff"]["flagged_for_v441_candidates"] == [
        "energy_value_guided_mcts_frontier_controller",
        "ebm_poe_planner_for_s3_generation",
    ]
    assert artifact["upstream_oracle_declarations"]["LEVELUP"]["verifier_is_oracle"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4789_bounded_when_gate_numbers_or_trust_fail() -> None:
    """SCENARIO-CAPSTONE-4789: S1 is bounded unless non-oracle gate numbers pass."""

    low_loo = mod.build_artifact(
        artifacts=_artifacts(s1=_s1(loo=0.69)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    oracle = mod.build_artifact(
        artifacts=_artifacts(s1=_s1(oracle=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    critical = mod.build_artifact(
        artifacts=_artifacts(s1=_s1(flagged=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(s1_code=2),
        duration_s=0.001,
    )

    assert low_loo["s1_structural_energy_verdict"]["verdict"] == "bounded"
    assert low_loo["s1_structural_energy_verdict"]["reason"] == "s1_gate_numbers_do_not_authorize_s2"
    assert oracle["s1_structural_energy_verdict"]["reason"] == "s1_oracle_not_moat"
    assert critical["s1_structural_energy_verdict"]["reason"] == "s1_artifact_skipped_live_or_genuine_flag"
    assert critical["cited_upstream_artifacts"][0]["fields_imported"] == ["flagged_adversarial"]
    assert critical["stale_false_positive_notes"][0]["status"] == "genuine_or_unresolved"


def test_run_capstone_invokes_summarizer_for_every_upstream_and_writes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4789: runtime aggregation reads every upstream via the summarizer."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4789\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        code = 1 if "heldout" in relative_path else 0
        return mod.SummarizerResult(["summarize", relative_path], code, "LIVE re-check: clean", "")

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)

    assert sorted(called) == sorted(spec.relative_path for spec in mod.UPSTREAM_SOURCES.values())
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"]["summarizer"]["present"] is True
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.validate_artifact(artifact) == []


def test_run_capstone_blocks_on_missing_required_upstream(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4789-BLOCKED-PRECONDITION: missing upstreams fail closed."""

    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4789\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    artifact = mod.run_capstone(
        root=tmp_path,
        summarizer=lambda _root, relative_path: mod.SummarizerResult(
            ["summarize", relative_path], 0, "LIVE re-check: clean", ""
        ),
    )

    assert artifact["honest_verdict"] == "blocked_missing_upstream:SOTA"
    assert artifact["s1_structural_energy_verdict"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations() -> None:
    """SCENARIO-CAPSTONE-4789: malformed scorecards fail validation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "not terminal"}
    )
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4781}]}
    )
    assert "flagged_artifact_imported_metrics:4781" in mod.validate_artifact(
        {
            **artifact,
            "flagged_artifacts_skipped": [{"experiment_id": 4781}],
            "cited_upstream_artifacts": [
                {
                    "experiment_id": 4781,
                    "fields_imported": ["energy_ranking_loo_auroc_mean"],
                    "sha256": "sha256:s1",
                }
            ],
        }
    )
    assert "invalid_s1_verdict" in mod.validate_artifact(
        {**artifact, "s1_structural_energy_verdict": {"verdict": "maybe"}}
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


def test_defensive_branches_and_registry_yaml_error_are_recorded(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4789-BLOCKED-PRECONDITION: defensive branches fail closed."""

    assert mod._int(True, 7) == 7
    assert mod._int("x", 9) == 9
    assert mod._float(True) is None
    assert mod._float("x") is None
    assert mod._skip_metrics(None, None) is True
    assert mod._cited_artifacts({"LEVELUP": _levelup()}, {}, {})[0]["experiment_id"] == 4782
    assert mod._numeric_list("bad") == []
    assert mod._s1_verdict(None, None) == {}
    assert mod._levelup_bank(None) == {}
    assert mod._self_play_checkpoint(None) == {}
    assert mod._heldout_readiness(None) == {}
    assert mod._silent_bug_audit(None) == {}
    assert mod._submission_package_state(None) == {}
    assert mod._hardware_continuity(None) == {}
    assert mod._sota_handoff(None) == {}

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
        == "spec_missing_req_4789"
    )

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4789\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    artifact = mod.run_capstone(
        root=tmp_path,
        summarizer=lambda _root, relative_path: mod.SummarizerResult(
            ["summarize", relative_path], 0, "LIVE re-check: clean", ""
        ),
    )

    assert artifact["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert artifact["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(artifact) == []
