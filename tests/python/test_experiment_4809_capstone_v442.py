"""Tests for REQ-CAPSTONE-4809 / SCENARIO-CAPSTONE-4809."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4809_capstone_v442 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _s2v2(
    *,
    verdict: str = "complete_structural_energy_s2v2_bounded_diverse_pool",
    delta: float = -0.1,
    ci: list[float] | None = None,
    s3: bool = False,
    oracle: bool = False,
    positive_control: bool = True,
) -> JsonDict:
    ci = [-0.3, 0.02] if ci is None else ci
    return {
        "experiment": "experiment_4801_structural_energy_s2v2_diverse_trust_gate",
        "experiment_id": 4801,
        "honest_verdict": verdict,
        "verifier_is_oracle": oracle,
        "live_path_reachable": True,
        "energy_selected_offpath_cell_recall": 0.22,
        "accuracy_gate_selected_offpath_cell_recall": 0.32,
        "energy_minus_accuracy_delta": delta,
        "energy_minus_accuracy_delta_ci95": ci,
        "n_effective_games": 5,
        "min_heldout_games": 5,
        "positive_control_passed": positive_control,
        "false_negative_risk_checked": True,
        "s3_authorized": s3,
        "candidate_pool_diversity": [
            {
                "game": f"g{i}",
                "effective": True,
                "n_candidates": 3,
                "distinct_heldout_cell_recall_count": 2,
            }
            for i in range(5)
        ],
        "game_results": [{"game": f"g{i}", "energy_minus_accuracy_delta": delta} for i in range(5)],
    }


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4802_levelup_attempt",
        "honest_verdict": "complete_bp35_no_new_level_residual_existing_depth",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "target_game": "bp35",
        "registry_update": {
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": 65,
            "updated": False,
        },
        "attempted_games": [{"game": "bp35", "reached_level": 2}],
        "dead_ends": ["existing depth only"],
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": True,
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4803_self_play_verifier_checkpoint",
        "honest_verdict": "success_re86_L2_checkpoint_refreshed",
        "verifier_checkpoint_refreshed": True,
        "target_game": "re86",
        "self_play_residual": "checkpoint_refreshed_gate_passed",
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "reproduction_gate": {"reproduced": True, "game": "re86", "claimed_level": 2},
        "solve_provenance": "live_agent_self_discovery",
    }


def _heldout(*, changed: bool = False) -> JsonDict:
    return {
        "experiment": "experiment_4804_heldout_first_win_readiness",
        "experiment_id": 4804,
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
    }


def _audit(*, reopened: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_4805_silent_bug_audit",
        "experiment_id": 4805,
        "honest_verdict": "complete_arc_null_silent_bug_audit_3_nulls_1_reopen",
        "nulls_audited": 3,
        "trusted_nulls": [
            "experiment_4802_levelup_attempt",
            "experiment_4804_heldout_first_win_readiness",
        ],
        "s2v2_candidate_pool_diverse": not reopened,
        "s2v2_diversity_check": {
            "degenerate_candidate_pool_flagged": reopened,
            "flag_kinds": ["DEGENERATE_CANDIDATE_POOL"] if reopened else [],
            "n_effective_games": 5,
            "min_heldout_games": 5,
        },
        "silent_bugs_found": [
            {
                "null_id": "experiment_4801_structural_energy_s2v2_diverse_trust_gate",
                "verdict": "silent_bug_must_reopen",
                "silent_bug_signatures": ["s2v2_degenerate_candidate_pool"],
            }
        ]
        if reopened
        else [],
        "per_null_verdicts": [],
        "verifier_is_oracle": False,
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4806_submission_package_harden",
        "honest_verdict": "success_package_builds_vram_gate_green",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"dry_build_ran": True},
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 4807,
        "honest_verdict": "success: kv260_continuity_recorded",
        "kv260_ssh_reachable": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "board_state": {"hostname": "kv260", "uio_device_count": 5},
        "verifier_is_oracle": False,
    }


def _sota() -> JsonDict:
    return {
        "experiment": "experiment_4808_sota_ingestion_energy_guided_generation",
        "honest_verdict": "success_sota_ingestion_energy_guided_generation_mapped",
        "flagged_for_v443": [
            {"candidate": "bolt_cold_cfg_value_tree_generator_for_s3"},
            {"candidate": "bes_energy_fitness_pool_inserter"},
        ],
        "methods_mapped": [{"method": "Energy-constrained sampler"}],
        "s3_context": {"s2v2_verdict": "inconclusive"},
        "arxiv_ids_cited": ["2202.11705"],
    }


def _artifacts(*, s2v2: JsonDict | None = None, audit: JsonDict | None = None) -> dict[str, JsonDict]:
    return {
        "S2V2": s2v2 or _s2v2(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": _self_play(),
        "HELDOUT": _heldout(),
        "BUG_AUDIT": audit or _audit(),
        "PACKAGE": _package(),
        "HARDWARE": _hardware(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {key: f"sha256:{key.lower()}" for key in mod.UPSTREAM_SOURCES}


def _summaries(*, s2_code: int = 2, s2_text: str = "DEGENERATE_CANDIDATE_POOL") -> dict[str, mod.SummarizerResult]:
    summaries = {
        key: mod.SummarizerResult(
            command=["python", "scripts/summarize_artifact.py", spec.relative_path],
            exit_code=0,
            stdout="LIVE re-check: clean",
            stderr="",
        )
        for key, spec in mod.UPSTREAM_SOURCES.items()
    }
    summaries["S2V2"] = mod.SummarizerResult(
        command=[
            "python",
            "scripts/summarize_artifact.py",
            mod.UPSTREAM_SOURCES["S2V2"].relative_path,
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
        stdout="LIVE re-check: warn",
        stderr="",
    )
    return summaries


def test_req_capstone_4809_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4809: OpenSpec declares the .442 scorecard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4809_degenerate_s2v2_is_inconclusive() -> None:
    """SCENARIO-CAPSTONE-4809: DEGENERATE_CANDIDATE_POOL blocks bounded nulls."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    verdict = artifact["s2v2_structural_energy_verdict"]
    assert artifact["honest_verdict"] == "complete_s2v2_inconclusive_degenerate_pool_capstone_ready"
    assert verdict["verdict"] == "inconclusive"
    assert verdict["reason"] == "degenerate_candidate_pool_live_check"
    assert verdict["degenerate_candidate_pool_flagged"] is True
    assert verdict["s3_authorized"] is False
    assert verdict["genuine_bounded_null"] is False
    assert verdict["diversity_check_trustworthy"] is False
    assert verdict["reported_energy_minus_accuracy_delta"] == pytest.approx(-0.1)
    assert artifact["silent_bug_audit"]["s2v2_reopened"] is True
    assert artifact["readiness"]["ready_for_operator_submit"] is False
    assert artifact["levelup_bank"]["moat_claim"] is False
    assert artifact["self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert artifact["submission_package_state"]["decision"] == "package_ready_operator_only"
    assert artifact["hardware_continuity"]["decision"] == "kv260_reachable"
    assert artifact["sota_handoff"]["flagged_for_v443_candidates"] == [
        "bolt_cold_cfg_value_tree_generator_for_s3",
        "bes_energy_fitness_pool_inserter",
    ]
    assert artifact["cited_upstream_artifacts"][0]["fields_imported"] == [
        "honest_verdict",
        "DEGENERATE_CANDIDATE_POOL",
        "n_effective_games",
        "min_heldout_games",
        "candidate_pool_diversity",
    ]
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4809_clean_s2v2_win_or_genuine_bounded() -> None:
    """SCENARIO-CAPSTONE-4809: clean S2-v2 triages win versus genuine bounded."""

    win = mod.build_artifact(
        artifacts=_artifacts(
            s2v2=_s2v2(
                verdict="success_structural_energy_s2v2_trust_gate_authorizes_s3",
                delta=0.12,
                ci=[0.03, 0.2],
                s3=True,
            ),
            audit=_audit(reopened=False),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(s2_code=0, s2_text="LIVE re-check: clean"),
        duration_s=0.001,
    )
    bounded = mod.build_artifact(
        artifacts=_artifacts(audit=_audit(reopened=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(s2_code=0, s2_text="LIVE re-check: clean"),
        duration_s=0.001,
    )

    assert win["honest_verdict"] == "success_s2v2_energy_ranking_beats_accuracy_s3_authorized"
    assert win["s2v2_structural_energy_verdict"]["verdict"] == "win"
    assert win["s2v2_structural_energy_verdict"]["s3_authorized"] is True
    assert bounded["honest_verdict"] == "complete_s2v2_genuine_bounded_null_pivot_to_s3_generation"
    assert bounded["s2v2_structural_energy_verdict"]["verdict"] == "genuine_bounded"
    assert bounded["s2v2_structural_energy_verdict"]["genuine_bounded_null"] is True
    assert bounded["s2v2_structural_energy_verdict"]["pivot"] == "pivot_energy_to_s3_generation"


def test_run_capstone_invokes_summarizer_for_every_upstream_and_writes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4809: runtime aggregation reads every upstream via the summarizer."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4809\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")

    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        if "s2v2" in relative_path:
            return mod.SummarizerResult(["summarize", relative_path], 2, "DEGENERATE_CANDIDATE_POOL", "")
        if "heldout" in relative_path:
            return mod.SummarizerResult(["summarize", relative_path], 1, "LIVE re-check: warn", "")
        return mod.SummarizerResult(["summarize", relative_path], 0, "LIVE re-check: clean", "")

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)

    assert sorted(called) == sorted(spec.relative_path for spec in mod.UPSTREAM_SOURCES.values())
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"]["summarizer"]["present"] is True
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.validate_artifact(artifact) == []


def test_run_capstone_blocks_on_missing_required_upstream(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4809-BLOCKED-PRECONDITION: missing upstreams fail closed."""

    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4809\n", encoding="utf-8")
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
    assert artifact["s2v2_structural_energy_verdict"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations() -> None:
    """SCENARIO-CAPSTONE-4809: malformed scorecards fail validation."""

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
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4801}]}
    )
    assert "invalid_s2v2_verdict" in mod.validate_artifact(
        {**artifact, "s2v2_structural_energy_verdict": {"verdict": "maybe"}}
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


def test_defensive_helpers_and_registry_yaml_error_are_recorded(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4809-BLOCKED-PRECONDITION: defensive branches fail closed."""

    clean_summary = mod.SummarizerResult(["summarize"], 0, "LIVE re-check: clean", "")
    critical_summary = mod.SummarizerResult(["summarize"], 2, "OTHER_CRITICAL", "")

    assert mod._int(True, 7) == 7
    assert mod._int("x", 9) == 9
    assert mod._float(True) is None
    assert mod._float("x") is None
    assert mod._mapping("x") == {}
    assert mod._s2v2_verdict(None, None) == {}
    assert (
        mod._s2v2_verdict(
            _s2v2(ci=[-0.3, -0.1], positive_control=False),
            clean_summary,
        )["reason"]
        == "s2v2_live_clean_but_gate_requirements_not_met"
    )
    assert mod._levelup_bank(None) == {}
    assert mod._self_play_checkpoint(None) == {}
    assert mod._heldout_readiness(None) == {}
    assert mod._silent_bug_audit(None) == {}
    assert mod._submission_package_state(None) == {}
    assert mod._hardware_continuity(None) == {}
    assert mod._sota_handoff(None) == {}
    assert mod._imported_fields("HELDOUT", _heldout(), critical_summary) == [
        "live_critical_recheck"
    ]
    assert mod._cited_artifacts({"S2V2": None, "LEVELUP": _levelup()}, {}, {}) == [
        {
            "experiment_id": 4802,
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
            "experiment_id": 4804,
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
        == "spec_missing_req_4809"
    )

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4809\n", encoding="utf-8")
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
