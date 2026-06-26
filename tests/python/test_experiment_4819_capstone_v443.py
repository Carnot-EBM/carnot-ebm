"""Tests for REQ-CAPSTONE-4819 / SCENARIO-CAPSTONE-4819."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4819_capstone_v443 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _s2v3(
    *,
    verdict: str = "complete_structural_energy_s2v3_bounded_corpus_wide",
    delta: float = 0.09054944463551816,
    ci: list[float] | None = None,
    n_available: int = 25,
    n_attempted: int = 25,
    n_effective: int = 23,
    s3: bool = False,
    oracle: bool = False,
    live_path: bool = True,
    positive_control: bool = True,
) -> JsonDict:
    ci = [-0.06276362669828736, 0.26410130774644547] if ci is None else ci
    return {
        "experiment": "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
        "experiment_id": 4811,
        "honest_verdict": verdict,
        "verifier_is_oracle": oracle,
        "live_path_reachable": live_path,
        "energy_selected_offpath_cell_recall": 0.30662955385318924,
        "accuracy_gate_selected_offpath_cell_recall": 0.216080109217671,
        "energy_minus_accuracy_delta": delta,
        "energy_minus_accuracy_delta_ci95": ci,
        "n_available_games": n_available,
        "n_games_attempted": n_attempted,
        "n_effective_games": n_effective,
        "required_effective_games": 15,
        "min_heldout_games": n_effective,
        "positive_control_passed": positive_control,
        "false_negative_risk_checked": True,
        "candidates_genuinely_induced": True,
        "s3_authorized": s3,
        "candidate_pool_diversity": [
            {
                "game": f"g{i:02d}",
                "effective": i < n_effective,
                "n_candidates": 3,
                "distinct_heldout_cell_recall_count": 2 if i < n_effective else 1,
            }
            for i in range(n_attempted)
        ],
        "game_results": [
            {"game": f"g{i:02d}", "energy_minus_accuracy_delta": delta} for i in range(n_attempted)
        ],
    }


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4812_levelup_attempt",
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
        "experiment": "experiment_4813_self_play_verifier_checkpoint",
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
        "experiment": "experiment_4814_heldout_first_win_readiness",
        "experiment_id": 4814,
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


def _audit() -> JsonDict:
    return {
        "experiment": "experiment_4815_silent_bug_audit",
        "experiment_id": 4815,
        "honest_verdict": "complete_arc_null_silent_bug_audit_3_nulls_0_reopen",
        "nulls_audited": 3,
        "trusted_nulls": [
            "experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate",
            "experiment_4812_levelup_attempt",
            "experiment_4814_heldout_first_win_readiness",
        ],
        "silent_bugs_found": [],
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4816_submission_package_harden",
        "honest_verdict": "success_package_builds_vram_gate_green",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"dry_build_ran": True, "package_builds": True},
    }


def _sota() -> JsonDict:
    return {
        "honest_verdict": "success_sota_ingestion_energy_guided_generation_mapped",
        "methods_mapped": [{"method": "Energy-constrained sampler"}],
        "flagged_for_v444": [
            {"candidate": "bolt_cold_cfg_value_tree_generator_for_s3"},
            {"candidate": "bes_energy_fitness_pool_inserter"},
        ],
        "s3_context": {"s3_generation_allowed": True},
        "arxiv_ids_cited": ["2202.11705"],
    }


def _artifacts(
    *, s2v3: JsonDict | None = None, heldout: JsonDict | None = None
) -> dict[str, JsonDict]:
    return {
        "S2V3": s2v3 or _s2v3(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": _self_play(),
        "HELDOUT": heldout or _heldout(),
        "BUG_AUDIT": _audit(),
        "PACKAGE": _package(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {key: f"sha256:{key.lower()}" for key in mod.UPSTREAM_SOURCES}


def _summaries(
    *, s2_code: int = 0, s2_text: str = "LIVE re-check: clean"
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
    summaries["S2V3"] = mod.SummarizerResult(
        command=[
            "python",
            "scripts/summarize_artifact.py",
            mod.UPSTREAM_SOURCES["S2V3"].relative_path,
        ],
        exit_code=s2_code,
        stdout=s2_text,
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


def test_req_capstone_4819_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4819: OpenSpec declares the .443 scorecard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4819_clean_s2v3_is_genuine_corpus_bounded() -> None:
    """SCENARIO-CAPSTONE-4819: powered clean S2-v3 with CI crossing zero is bounded."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        corpus_game_count=25,
        duration_s=0.001,
    )

    verdict = artifact["s2v3_structural_energy_verdict"]
    assert (
        artifact["honest_verdict"]
        == "complete_s2v3_genuine_corpus_wide_bounded_null_pivot_to_s3_generation"
    )
    assert verdict["verdict"] == "genuine_corpus_wide_bounded_null"
    assert verdict["s3_authorized"] is False
    assert verdict["genuine_corpus_wide_bounded_null"] is True
    assert verdict["coverage_trustworthy"] is True
    assert verdict["n_available_matches_real_corpus"] is True
    assert verdict["degenerate_candidate_pool_flagged"] is False
    assert verdict["reported_energy_minus_accuracy_delta"] == pytest.approx(0.09054944463551816)
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["levelup_bank"]["moat_claim"] is False
    assert artifact["self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert artifact["heldout_readiness"]["decision"] == "flat_null_no_readiness_gain"
    assert artifact["readiness"]["pivot_energy_to_s3_generation"] is True
    assert artifact["submission_package_state"]["decision"] == "package_ready_operator_only"
    assert artifact["sota_handoff"]["flagged_for_v444_candidates"] == [
        "bolt_cold_cfg_value_tree_generator_for_s3",
        "bes_energy_fitness_pool_inserter",
    ]
    assert artifact["cited_upstream_artifacts"][0]["fields_imported"] == [
        "honest_verdict",
        "s3_authorized",
        "verifier_is_oracle",
        "live_path_reachable",
        "energy_selected_offpath_cell_recall",
        "accuracy_gate_selected_offpath_cell_recall",
        "energy_minus_accuracy_delta",
        "energy_minus_accuracy_delta_ci95",
        "n_available_games",
        "n_games_attempted",
        "n_effective_games",
        "required_effective_games",
        "min_heldout_games",
        "positive_control_passed",
        "false_negative_risk_checked",
        "candidates_genuinely_induced",
        "candidate_pool_diversity",
        "game_results",
    ]
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4819_win_and_inconclusive_paths() -> None:
    """SCENARIO-CAPSTONE-4819: S2-v3 distinguishes win from under-covered results."""

    win = mod.build_artifact(
        artifacts=_artifacts(
            s2v3=_s2v3(
                verdict="success_structural_energy_s2v3_trust_gate_authorizes_s3",
                delta=0.12,
                ci=[0.03, 0.2],
                s3=True,
            ),
            heldout=_heldout(changed=True),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        corpus_game_count=25,
        duration_s=0.001,
    )
    degenerate = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(s2_code=2, s2_text="DEGENERATE_CANDIDATE_POOL"),
        corpus_game_count=25,
        duration_s=0.001,
    )
    undercovered = mod.build_artifact(
        artifacts=_artifacts(s2v3=_s2v3(n_effective=9)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        corpus_game_count=25,
        duration_s=0.001,
    )
    mismatch = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        corpus_game_count=26,
        duration_s=0.001,
    )
    not_attempted = mod.build_artifact(
        artifacts=_artifacts(s2v3=_s2v3(n_attempted=24)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        corpus_game_count=25,
        duration_s=0.001,
    )
    clean_gate_miss = mod.build_artifact(
        artifacts=_artifacts(s2v3=_s2v3(ci=[-0.4, -0.2], positive_control=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        corpus_game_count=25,
        duration_s=0.001,
    )

    assert (
        win["honest_verdict"]
        == "success_s2v3_corpus_wide_energy_ranking_beats_accuracy_s3_authorized"
    )
    assert win["s2v3_structural_energy_verdict"]["verdict"] == "corpus_wide_trust_win"
    assert win["readiness"]["ready_for_operator_submit"] is True
    assert (
        degenerate["s2v3_structural_energy_verdict"]["reason"]
        == "degenerate_candidate_pool_live_check"
    )
    assert degenerate["cited_upstream_artifacts"][0]["fields_imported"] == [
        "honest_verdict",
        "DEGENERATE_CANDIDATE_POOL",
        "n_available_games",
        "n_games_attempted",
        "n_effective_games",
        "required_effective_games",
        "candidate_pool_diversity",
    ]
    assert (
        undercovered["s2v3_structural_energy_verdict"]["reason"] == "insufficient_corpus_diversity"
    )
    assert mismatch["s2v3_structural_energy_verdict"]["reason"] == "b1_corpus_count_mismatch"
    assert (
        not_attempted["s2v3_structural_energy_verdict"]["reason"]
        == "not_all_available_games_attempted"
    )
    assert (
        clean_gate_miss["s2v3_structural_energy_verdict"]["reason"]
        == "s2v3_live_clean_but_gate_requirements_not_met"
    )


def test_run_capstone_invokes_summarizer_for_every_upstream_and_writes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4819: runtime aggregation reads every upstream via the summarizer."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4819\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")
    env_dir = tmp_path / "environment_files"
    env_dir.mkdir()
    for index in range(25):
        (env_dir / f"g{index:02d}").write_text("", encoding="utf-8")

    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        if "heldout" in relative_path:
            return mod.SummarizerResult(["summarize", relative_path], 1, "LIVE re-check: warn", "")
        return mod.SummarizerResult(["summarize", relative_path], 0, "LIVE re-check: clean", "")

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)

    assert sorted(called) == sorted(spec.relative_path for spec in mod.UPSTREAM_SOURCES.values())
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"]["summarizer"]["present"] is True
    assert artifact["preconditions_checked"]["offline_corpus"]["game_count"] == 25
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.validate_artifact(artifact) == []


def test_run_capstone_blocks_on_missing_required_upstream(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4819-BLOCKED-PRECONDITION: missing upstreams fail closed."""

    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4819\n", encoding="utf-8")
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
    assert artifact["s2v3_structural_energy_verdict"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations_and_helpers_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4819: malformed scorecards fail validation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        corpus_game_count=25,
        duration_s=0.001,
    )
    critical_summary = mod.SummarizerResult(["summarize"], 2, "OTHER_CRITICAL", "")

    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "not terminal"}
    )
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4811}]}
    )
    assert "invalid_s2v3_verdict" in mod.validate_artifact(
        {**artifact, "s2v3_structural_energy_verdict": {"verdict": "maybe"}}
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
    assert mod._int(True, 7) == 7
    assert mod._int("x", 9) == 9
    assert mod._float(True) is None
    assert mod._float("x") is None
    assert mod._mapping("x") == {}
    assert mod._s2v3_verdict(None, None, corpus_game_count=25) == {}
    assert mod._levelup_bank(None) == {}
    assert mod._self_play_checkpoint(None) == {}
    assert mod._heldout_readiness(None) == {}
    assert mod._silent_bug_audit(None) == {}
    assert mod._submission_package_state(None) == {}
    assert mod._sota_handoff(None) == {}
    assert mod._imported_fields("HELDOUT", _heldout(), critical_summary, corpus_game_count=25) == [
        "live_critical_recheck"
    ]
    assert mod._cited_artifacts(
        {"S2V3": None, "LEVELUP": _levelup()},
        {},
        {},
        corpus_game_count=25,
    ) == [
        {
            "experiment_id": 4812,
            "fields_imported": list(mod.CLEAN_IMPORT_FIELDS["LEVELUP"]),
            "sha256": "",
        }
    ]
    assert mod._flagged_artifacts_skipped(
        {"HELDOUT": _heldout()},
        {"HELDOUT": "sha256:heldout"},
        {"HELDOUT": critical_summary},
    ) == [
        {
            "source": "HELDOUT",
            "experiment_id": 4814,
            "path": mod.UPSTREAM_SOURCES["HELDOUT"].relative_path,
            "reason": "live_critical_recheck",
            "sha256": "sha256:heldout",
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
        == "spec_missing_req_4819"
    )

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4819\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    blocked = mod.run_capstone(
        root=tmp_path,
        summarizer=lambda _root, relative_path: mod.SummarizerResult(
            ["summarize", relative_path], 0, "LIVE re-check: clean", ""
        ),
    )

    assert blocked["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert blocked["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(blocked) == []
