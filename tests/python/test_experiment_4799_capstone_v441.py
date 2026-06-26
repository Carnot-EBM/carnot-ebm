"""Tests for REQ-CAPSTONE-4799 / SCENARIO-CAPSTONE-4799."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4799_capstone_v441 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _s2(
    *,
    flagged: bool = False,
    delta: float = 0.125,
    ci: list[float] | None = None,
    s3: bool = True,
    oracle: bool = False,
    live_path: bool = True,
) -> JsonDict:
    ci = [0.04, 0.21] if ci is None else ci
    artifact: JsonDict = {
        "experiment": "experiment_4791_structural_energy_s2_offpath_trust_gate",
        "experiment_id": 4791,
        "honest_verdict": "success_structural_energy_s2_trust_gate_authorizes_s3"
        if s3
        else "complete_structural_energy_s2_no_live_trust_value",
        "verifier_is_oracle": oracle,
        "live_path_reachable": live_path,
        "energy_selected_offpath_cell_recall": 0.56,
        "accuracy_gate_selected_offpath_cell_recall": 0.56 - delta,
        "energy_minus_accuracy_delta": delta,
        "energy_minus_accuracy_delta_ci95": ci,
        "n_heldout_games": 5,
        "min_heldout_games": 5,
        "s3_authorized": s3,
        "game_results": [{"game": f"g{i}", "energy_minus_accuracy_delta": delta} for i in range(5)],
        "candidate_sets": {"g0": ["a", "b"]},
    }
    if flagged:
        artifact["flagged_adversarial"] = True
    return artifact


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4792_levelup_attempt",
        "honest_verdict": "complete_sb26_no_new_level_residual_existing_depth",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "target_game": "sb26",
        "registry_update": {
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": 65,
            "updated": False,
        },
        "attempted_games": [{"game": "sb26", "reached_level": 2}],
        "dead_ends": ["existing depth only"],
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": True,
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4793_self_play_verifier_checkpoint",
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
        "experiment": "experiment_4794_heldout_first_win_readiness",
        "experiment_id": 4794,
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


def _audit() -> JsonDict:
    return {
        "experiment": "experiment_4795_silent_bug_audit",
        "honest_verdict": "complete_arc_null_silent_bug_audit_3_nulls_0_reopen",
        "nulls_audited": 3,
        "trusted_nulls": [
            "experiment_4791_structural_energy_s2_offpath_trust_gate",
            "experiment_4792_levelup_attempt",
            "experiment_4794_heldout_first_win_readiness",
        ],
        "silent_bugs_found": [],
        "per_null_verdicts": [
            {
                "null_id": "experiment_4791_structural_energy_s2_offpath_trust_gate",
                "verdict": "trustworthy_null",
                "s2_live_path_reachable_confirmed": True,
                "s2_control_check": {"checked_games": 5},
            }
        ],
        "verifier_is_oracle": False,
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4796_submission_package_harden",
        "honest_verdict": "success_package_builds_vram_gate_green",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"dry_build_ran": True},
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 4797,
        "honest_verdict": "success: kv260_continuity_recorded",
        "kv260_ssh_reachable": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "board_state": {"hostname": "kv260", "uio_device_count": 5},
        "verifier_is_oracle": False,
    }


def _sota() -> JsonDict:
    return {
        "honest_verdict": "success_sota_ingestion_energy_guided_generation_mapped",
        "flagged_for_v442": [
            {"candidate": "cold_cfg_value_tree_generator_for_s3"},
            {"candidate": "energy_fitness_qd_pool_inserter"},
        ],
        "methods_mapped": [{"method": "Energy-constrained Langevin candidate generator"}],
        "s1_s2_context": {
            "s2_energy_minus_accuracy_delta": 0.0,
            "s2_imported_honest_verdict": "complete_structural_energy_s2_no_live_trust_value",
        },
        "arxiv_ids_cited": ["2202.11705"],
    }


def _artifacts(
    *, s2: JsonDict | None = None, heldout: JsonDict | None = None
) -> dict[str, JsonDict]:
    return {
        "S2": s2 or _s2(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": _self_play(),
        "HELDOUT": heldout or _heldout(),
        "BUG_AUDIT": _audit(),
        "PACKAGE": _package(),
        "HARDWARE": _hardware(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {key: f"sha256:{key.lower()}" for key in mod.UPSTREAM_SOURCES}


def _summaries(*, s2_code: int = 0, heldout_code: int = 1) -> dict[str, mod.SummarizerResult]:
    summaries = {
        key: mod.SummarizerResult(
            command=["python", "scripts/summarize_artifact.py", spec.relative_path],
            exit_code=0,
            stdout="LIVE re-check: clean",
            stderr="",
        )
        for key, spec in mod.UPSTREAM_SOURCES.items()
    }
    summaries["S2"] = mod.SummarizerResult(
        command=[
            "python",
            "scripts/summarize_artifact.py",
            mod.UPSTREAM_SOURCES["S2"].relative_path,
        ],
        exit_code=s2_code,
        stdout="LIVE re-check: clean" if s2_code == 0 else "LIVE re-check: CRITICAL",
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


def test_req_capstone_4799_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4799: OpenSpec declares the .441 scorecard before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4799_imports_stale_flagged_s2_pass() -> None:
    """SCENARIO-CAPSTONE-4799: live-clean stale flags do not erase an S2 win."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(s2=_s2(flagged=True), heldout=_heldout(changed=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success_s2_structural_energy_trust_gate_s3_authorized"
    verdict = artifact["s2_structural_energy_verdict"]
    assert verdict["verdict"] == "trust_gate_win"
    assert verdict["s3_authorized"] is True
    assert verdict["energy_minus_accuracy_delta"] == pytest.approx(0.125)
    assert verdict["energy_minus_accuracy_delta_ci95"] == [0.04, 0.21]
    assert verdict["ci_excludes_zero"] is True
    assert artifact["stale_false_positive_notes"][0]["source"] == "S2"
    assert artifact["stale_false_positive_notes"][0]["status"] == "stale_false_positive"
    assert artifact["flagged_artifacts_skipped"] == []
    assert (
        "energy_minus_accuracy_delta" in artifact["cited_upstream_artifacts"][0]["fields_imported"]
    )
    assert artifact["levelup_bank"]["moat_claim"] is False
    assert artifact["self_play_checkpoint"]["decision"] == "checkpoint_refreshed"
    assert artifact["readiness"]["ready_for_operator_submit"] is True
    assert artifact["silent_bug_audit"]["s2_audit_note"] == "s2_null_audited_as_trustworthy"
    assert artifact["submission_package_state"]["decision"] == "package_ready_operator_only"
    assert artifact["hardware_continuity"]["decision"] == "kv260_reachable"
    assert artifact["sota_handoff"]["flagged_for_v442_candidates"] == [
        "cold_cfg_value_tree_generator_for_s3",
        "energy_fitness_qd_pool_inserter",
    ]
    assert artifact["upstream_oracle_declarations"]["LEVELUP"]["verifier_is_oracle"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4799_bounded_for_live_critical_or_failed_gate() -> None:
    """SCENARIO-CAPSTONE-4799: S2 is bounded unless non-oracle gate numbers pass."""

    zero_delta = mod.build_artifact(
        artifacts=_artifacts(s2=_s2(delta=0.0, ci=[0.0, 0.0], s3=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    oracle = mod.build_artifact(
        artifacts=_artifacts(s2=_s2(oracle=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    live_unreachable = mod.build_artifact(
        artifacts=_artifacts(s2=_s2(live_path=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    critical = mod.build_artifact(
        artifacts=_artifacts(s2=_s2(flagged=True, delta=0.0, ci=[0.0, 0.0], s3=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(s2_code=2),
        duration_s=0.001,
    )

    assert (
        zero_delta["s2_structural_energy_verdict"]["reason"]
        == "s2_gate_numbers_do_not_authorize_s3"
    )
    assert oracle["s2_structural_energy_verdict"]["reason"] == "s2_oracle_not_moat"
    assert (
        live_unreachable["s2_structural_energy_verdict"]["reason"]
        == "s2_gate_numbers_do_not_authorize_s3"
    )
    assert (
        critical["s2_structural_energy_verdict"]["reason"]
        == "s2_artifact_skipped_live_or_genuine_flag"
    )
    assert critical["s2_structural_energy_verdict"]["reported_energy_minus_accuracy_delta"] == 0.0
    assert critical["s2_structural_energy_verdict"]["metrics_imported"] is False
    assert critical["cited_upstream_artifacts"][0]["fields_imported"] == ["flagged_adversarial"]
    assert critical["stale_false_positive_notes"][0]["status"] == "genuine_or_unresolved"
    assert critical["flagged_artifacts_skipped"][0]["reason"] == "live_critical_recheck"


def test_run_capstone_invokes_summarizer_for_every_upstream_and_writes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4799: runtime aggregation reads every upstream via the summarizer."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4799\n", encoding="utf-8")
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
    """SCENARIO-CAPSTONE-4799-BLOCKED-PRECONDITION: missing upstreams fail closed."""

    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 65\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4799\n", encoding="utf-8")
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
    assert artifact["s2_structural_energy_verdict"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations() -> None:
    """SCENARIO-CAPSTONE-4799: malformed scorecards fail validation."""

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
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4791}]}
    )
    assert "flagged_artifact_imported_metrics:4791" in mod.validate_artifact(
        {
            **artifact,
            "flagged_artifacts_skipped": [{"experiment_id": 4791}],
            "cited_upstream_artifacts": [
                {
                    "experiment_id": 4791,
                    "fields_imported": ["energy_minus_accuracy_delta"],
                    "sha256": "sha256:s2",
                }
            ],
        }
    )
    assert "invalid_s2_verdict" in mod.validate_artifact(
        {**artifact, "s2_structural_energy_verdict": {"verdict": "maybe"}}
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
    """SCENARIO-CAPSTONE-4799-BLOCKED-PRECONDITION: defensive branches fail closed."""

    assert mod._int(True, 7) == 7
    assert mod._int("x", 9) == 9
    assert mod._float(True) is None
    assert mod._float("x") is None
    assert mod._ci_lower_positive("bad") is False
    assert mod._skip_metrics(None, None) is True
    assert mod._cited_artifacts({"LEVELUP": _levelup()}, {}, {})[0]["experiment_id"] == 4792
    assert mod._s2_verdict(None, None) == {}
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
        == "spec_missing_req_4799"
    )

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4799\n", encoding="utf-8")
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
