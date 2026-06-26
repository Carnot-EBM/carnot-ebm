"""Tests for REQ-CAPSTONE-4779 / SCENARIO-CAPSTONE-4779."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4779_capstone_v439 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _s0prime(*, flagged: bool = True, gate: bool = True, oracle: bool = False) -> JsonDict:
    artifact: JsonDict = {
        "experiment": "experiment_4771_structural_energy_s0prime_origin_matched",
        "experiment_id": 4771,
        "honest_verdict": "success_structural_energy_s0prime_reopens_s1",
        "verifier_is_oracle": oracle,
        "s0prime_gate_passed": gate,
        "loo_auroc_structural": 0.7386642861889572,
        "loo_auroc_ci95": [0.636412794237013, 0.8332008450933205],
        "loo_auroc_marginal_control": 0.510766138123749,
        "loo_auroc_majority_control": 0.5,
        "origin_probe_auroc": 0.5,
        "shuffled_label_control_auroc": 0.5033091959271814,
        "controls": {
            "shuffled_label_resamples": 128,
            "v2_frame_marginal_loo_auroc": 0.510766138123749,
        },
        "dataset_diagnostics": {"origin_matched": True},
        "n_candidate_rows": 463,
        "n_pos": 186,
        "n_neg": 277,
        "n_held_out_games": 16,
        "per_family_loo": {"frame_delta": 0.7392638081947291, "object_relational": 0.659731635551948},
        "origin_probe": {
            "loo_auroc": 0.5,
            "origin_counts": {"induced": 463},
            "status": "origin_matched_refit_complete",
            "refit_on_origin_matched_data": True,
        },
        "structural_minus_marginal_delta_ci95": [0.1271826145701076, 0.33248035484558097],
        "retire_if_same_verdict": True,
        "in_sample_auroc": 0.8479387446139514,
    }
    if flagged:
        artifact["flagged_adversarial"] = True
    return artifact


def _levelup(*, oracle: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_4772_levelup_attempt",
        "honest_verdict": "complete_ka59_no_new_level_residual_existing_depth",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "target_game": "ka59",
        "registry_update": {
            "path": "ops/arc_solve_registry.yaml",
            "reason": "no_new_level_banked",
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": 65,
            "updated": False,
        },
        "attempted_games": [
            {
                "game": "ka59",
                "prior_level": 1,
                "reached_level": 1,
                "target_level": 2,
                "offline_reproduced_existing_depth": True,
                "offline_reproduced_new_depth": False,
            }
        ],
        "dead_ends": ["ka59: same-depth reproduction reached L1"],
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": oracle,
    }


def _self_play(*, flagged: bool = True) -> JsonDict:
    artifact: JsonDict = {
        "experiment": "experiment_4773_self_play_verifier_checkpoint",
        "honest_verdict": "success_re86_L2_checkpoint_refreshed",
        "verifier_checkpoint_refreshed": True,
        "target_game": "re86",
        "self_play_residual": "checkpoint_refreshed_gate_passed",
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "reproduction_gate": {"reproduced": True, "game": "re86", "claimed_level": 2},
        "solve_provenance": "live_agent_self_discovery",
    }
    if flagged:
        artifact["flagged_adversarial"] = True
    return artifact


def _heldout() -> JsonDict:
    return {
        "experiment": "experiment_4774_heldout_first_win_readiness",
        "experiment_id": 4774,
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


def _audit(*, controls_fired: bool = False) -> JsonDict:
    checks = {
        "all_controls_fired": controls_fired,
        "class_balance_non_degenerate": True,
        "contributing_games_with_both_classes": ["ar25", "ka59"],
        "contributing_games_missing_a_class": [],
        "single_class_games_skipped": ["r11l", "sp80"],
        "single_class_games_not_skipped": [],
        "single_class_games_skipped_correctly": True,
        "origin_matching_real": True,
        "origin_probe_status": "origin_matched_refit_complete"
        if controls_fired
        else "origin_matched_single_origin_all_induced",
        "origin_probe_refit_on_origin_matched_data": controls_fired,
        "origin_probe_not_stale_s0_value": True,
        "origin_probe_auroc": 0.5,
        "shuffled_label_resamples": 128,
        "shuffled_label_control_auroc": 0.5033091959271814,
        "shuffled_label_module_permutation_loo": True,
        "shuffled_label_permuted_and_reran_loo": True,
    }
    verdict = "trustworthy_null" if controls_fired else "silent_bug_must_reopen"
    row = {
        "artifact_path": "results/experiment_4771_structural_energy_s0prime_origin_matched.json",
        "null_id": "experiment_4771_structural_energy_s0prime_origin_matched",
        "verdict": verdict,
        "silent_bug_signatures": [] if controls_fired else ["s0prime_origin_probe_not_refit"],
        "s0prime_leak_controls_fired": controls_fired,
        "s0prime_leak_control_checks": checks,
    }
    return {
        "experiment": "experiment_4775_silent_bug_audit",
        "honest_verdict": "complete_arc_null_silent_bug_audit_3_nulls_1_reopen",
        "nulls_audited": 3,
        "trusted_nulls": [
            "experiment_4772_levelup_attempt",
            "experiment_4774_heldout_first_win_readiness",
        ],
        "silent_bugs_found": [] if controls_fired else [row],
        "per_null_verdicts": [row],
        "s0prime_leak_controls_fired": controls_fired,
        "s0prime_leak_control_checks": checks,
        "verifier_is_oracle": False,
    }


def _package(*, flagged: bool = True) -> JsonDict:
    artifact: JsonDict = {
        "experiment": "experiment_4776_submission_package_harden",
        "honest_verdict": "success_package_builds_vram_gate_green",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"package_builds": True, "dry_build_ran": True},
    }
    if flagged:
        artifact["flagged_adversarial"] = True
    return artifact


def _hardware() -> JsonDict:
    return {
        "experiment": 4777,
        "honest_verdict": "success: kv260_continuity_recorded",
        "kv260_ssh_reachable": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "board_state": {"captured": True, "hostname": "kv260", "uio_device_count": 5},
        "verifier_is_oracle": False,
    }


def _sota() -> JsonDict:
    return {
        "honest_verdict": "success_sota_ingestion_structural_energy_mapped",
        "flagged_for_v440": [
            {"candidate": "slot_relational_contrastive_energy_s0prime_guarded"},
            {"candidate": "poe_code_world_model_trust_gate_after_s0prime"},
        ],
        "methods_mapped": [
            {"method": "Slot-relational contrastive transition energy"},
            {"method": "Executable PoE/code world-model trust energy"},
        ],
        "leak_robust_evaluation_note": {
            "roadmap_gate": "flagged_for_v440: leak_robust_eval_gate_for_all_structural_energy_continuations"
        },
        "arxiv_ids_cited": ["2006.15055", "2505.10819"],
    }


def _artifacts(
    *,
    s0prime: JsonDict | None = None,
    audit: JsonDict | None = None,
    self_play: JsonDict | None = None,
    package: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "S0PRIME": s0prime or _s0prime(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": self_play or _self_play(),
        "HELDOUT": _heldout(),
        "BUG_AUDIT": audit or _audit(),
        "PACKAGE": package or _package(),
        "HARDWARE": _hardware(),
        "SOTA": _sota(),
    }


def _hashes() -> dict[str, str]:
    return {key: f"sha256:{key.lower()}" for key in mod.UPSTREAM_SOURCES}


def _summaries() -> dict[str, mod.SummarizerResult]:
    return {
        key: mod.SummarizerResult(
            command=["python", "scripts/summarize_artifact.py", spec.relative_path],
            exit_code=0,
            stdout="clean",
            stderr="",
        )
        for key, spec in mod.UPSTREAM_SOURCES.items()
    }


def test_req_capstone_4779_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4779: OpenSpec declares the .439 scorecard before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4779_skips_flagged_s0prime_and_retires() -> None:
    """SCENARIO-CAPSTONE-4779: flagged S0' metrics are skipped before verdict aggregation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert (
        artifact["honest_verdict"]
        == "complete_s0prime_structural_energy_retires_v439_capstone_ready"
    )
    verdict = artifact["s0prime_structural_energy_verdict"]
    assert verdict["direction"] == "RETIRES"
    assert verdict["s1_queued"] is False
    assert verdict["reason"] == "s0prime_flagged_adversarial_skipped_controls_unfired"
    assert verdict["loo_auroc_structural"] is None
    assert verdict["control_numbers_source"] == "BUG_AUDIT"
    assert verdict["origin_probe_auroc"] == pytest.approx(0.5)
    assert verdict["shuffled_label_control_auroc"] == pytest.approx(0.5033091959271814)
    assert verdict["shuffled_label_resamples"] == 128
    assert verdict["origin_probe_refit_on_origin_matched_data"] is False
    assert artifact["reproducible_total_levels"] == 65

    assert artifact["levelup_bank"] == {
        "source": "LEVELUP",
        "experiment_id": 4772,
        "target_game": "ka59",
        "new_levels_banked": 0,
        "reproducible_total_levels_before": 65,
        "reproducible_total_levels_after": 65,
        "reproducible_total_levels_delta": 0,
        "registry_updated": False,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "verifier_is_oracle": True,
        "moat_claim": False,
    }
    assert artifact["self_play_checkpoint"] == {
        "source": "SELF_PLAY",
        "experiment_id": 4773,
        "decision": "skipped_flagged_adversarial",
    }
    assert artifact["submission_package_state"] == {
        "source": "PACKAGE",
        "experiment_id": 4776,
        "decision": "skipped_flagged_adversarial",
    }
    assert artifact["heldout_readiness"]["decision"] == "flat_null_no_readiness_gain"
    assert artifact["readiness"]["ready_for_operator_submit"] is False
    assert artifact["silent_bug_audit"]["s0prime_leak_controls_fired"] is False
    assert artifact["hardware_continuity"]["decision"] == "kv260_reachable"
    assert artifact["sota_handoff"]["flagged_for_v440_candidates"] == [
        "slot_relational_contrastive_energy_s0prime_guarded",
        "poe_code_world_model_trust_gate_after_s0prime",
    ]

    skipped = {row["experiment_id"]: row for row in artifact["flagged_artifacts_skipped"]}
    assert set(skipped) == {4771, 4773, 4776}

    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited[4771]["fields_imported"] == ["flagged_adversarial"]
    assert cited[4773]["fields_imported"] == ["flagged_adversarial"]
    assert cited[4776]["fields_imported"] == ["flagged_adversarial"]
    assert "loo_auroc_structural" not in cited[4771]["fields_imported"]
    assert "origin_probe_auroc" not in cited[4771]["fields_imported"]
    assert "s0prime_leak_control_checks" in cited[4775]["fields_imported"]
    assert artifact["upstream_oracle_declarations"]["LEVELUP"]["verifier_is_oracle"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4779_clean_non_oracle_s0prime_can_reopen() -> None:
    """SCENARIO-CAPSTONE-4779: clean S0' evidence reopens S1; oracle evidence cannot."""

    clean = mod.build_artifact(
        artifacts=_artifacts(s0prime=_s0prime(flagged=False), audit=_audit(controls_fired=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    oracle = mod.build_artifact(
        artifacts=_artifacts(
            s0prime=_s0prime(flagged=False, oracle=True),
            audit=_audit(controls_fired=True),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    collapsed = mod.build_artifact(
        artifacts=_artifacts(
            s0prime={**_s0prime(flagged=False), "loo_auroc_ci95": [0.49, 0.8]},
            audit=_audit(controls_fired=True),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert clean["honest_verdict"] == "success_s0prime_structural_energy_reopens_s1"
    assert clean["s0prime_structural_energy_verdict"]["direction"] == "REOPENS_TO_S1"
    assert clean["s0prime_structural_energy_verdict"]["s1_queued"] is True
    assert clean["s0prime_structural_energy_verdict"]["control_numbers_source"] == "S0PRIME"
    assert oracle["s0prime_structural_energy_verdict"]["direction"] == "RETIRES"
    assert oracle["s0prime_structural_energy_verdict"]["reason"] == "s0prime_oracle_not_moat"
    assert collapsed["s0prime_structural_energy_verdict"]["reason"] == (
        "s0prime_gate_failed_or_controls_failed"
    )


def test_run_capstone_invokes_summarizer_for_every_upstream_and_writes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4779: runtime aggregation reads every upstream via the summarizer."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        "schema_version: 1\nreproducible_total_levels: 65\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4779\n", encoding="utf-8")
    (tmp_path / mod.SUMMARIZER_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.SUMMARIZER_RELATIVE_PATH).write_text("# placeholder\n", encoding="utf-8")

    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        return mod.SummarizerResult(
            command=["summarize", relative_path],
            exit_code=0,
            stdout=f"summary for {relative_path}",
            stderr="",
        )

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)

    assert sorted(called) == sorted(spec.relative_path for spec in mod.UPSTREAM_SOURCES.values())
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"]["summarizer"]["present"] is True
    assert all(
        row["summarizer_exit_code"] == 0
        for row in artifact["preconditions_checked"]["upstream_artifacts"].values()
    )
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.validate_artifact(artifact) == []


def test_run_capstone_blocks_on_missing_required_upstream(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4779-BLOCKED-PRECONDITION: missing upstreams fail closed."""

    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        "schema_version: 1\nreproducible_total_levels: 65\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4779\n", encoding="utf-8")
    (tmp_path / mod.SUMMARIZER_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.SUMMARIZER_RELATIVE_PATH).write_text("# placeholder\n", encoding="utf-8")

    artifact = mod.run_capstone(
        root=tmp_path,
        summarizer=lambda _root, relative_path: mod.SummarizerResult(
            command=["summarize", relative_path],
            exit_code=0,
            stdout="clean",
            stderr="",
        ),
    )

    assert artifact["honest_verdict"] == "blocked_missing_upstream:SOTA"
    assert artifact["s0prime_structural_energy_verdict"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations() -> None:
    """SCENARIO-CAPSTONE-4779: malformed scorecards fail validation."""

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
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4771}]}
    )
    assert "flagged_artifact_imported_metrics:4771" in mod.validate_artifact(
        {
            **artifact,
            "cited_upstream_artifacts": [
                *artifact["cited_upstream_artifacts"],
                {
                    "experiment_id": 4771,
                    "fields_imported": ["loo_auroc_structural"],
                    "sha256": "sha256:bad",
                },
            ],
        }
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert "missing_principle:honest_verdict" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_model"}
    )
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "65"}
    )
    assert "invalid_s0prime_direction" in mod.validate_artifact(
        {**artifact, "s0prime_structural_energy_verdict": {"direction": "MAYBE"}}
    )


def test_defensive_scorecard_branches_are_explicit() -> None:
    """SCENARIO-CAPSTONE-4779-BLOCKED-PRECONDITION: defensive branches fail closed."""

    assert mod._int(True, 7) == 7
    assert mod._int("x", 9) == 9
    assert mod._float(True) is None
    assert mod._float("x") is None
    assert mod._ci_lower_gt([0.6, 0.8], 0.5) is True
    assert mod._ci_lower_gt([0.5], 0.5) is False
    assert mod._ci_lower_gt(None, 0.5) is False
    assert mod._control_leq(None, 0.55) is False

    assert mod._cited_artifacts({"S0PRIME": _s0prime()}, {})[0]["experiment_id"] == 4771
    assert mod._s0prime_verdict(None, _audit()) == {}
    assert mod._s0prime_verdict({**_s0prime(flagged=False), "flagged_adversarial": True}, {})[
        "reason"
    ] == "s0prime_flagged_adversarial_skipped"
    assert mod._s0prime_verdict(
        {
            **_s0prime(flagged=False),
            "origin_probe_auroc": 0.7,
            "shuffled_label_control_auroc": 0.7,
        },
        _audit(controls_fired=True),
    )["direction"] == "RETIRES"
    assert mod._audit_control_numbers(None) == {}
    assert mod._audit_control_numbers({"s0prime_leak_control_checks": "bad"}) == {}
    assert mod._audit_control_numbers({"per_null_verdicts": [{"null_id": "other"}]}) == {}
    per_null_only = {
        "per_null_verdicts": [
            "bad",
            {
                "null_id": "experiment_4771_structural_energy_s0prime_origin_matched",
                "s0prime_leak_controls_fired": True,
                "s0prime_leak_control_checks": {
                    "all_controls_fired": True,
                    "origin_probe_auroc": 0.5,
                    "shuffled_label_control_auroc": 0.503,
                },
            },
        ]
    }
    assert mod._audit_control_numbers(per_null_only)["s0prime_leak_controls_fired"] is True

    assert mod._levelup_bank(None) == {}
    assert mod._levelup_bank({**_levelup(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert mod._self_play_checkpoint(None) == {}
    assert mod._self_play_checkpoint({**_self_play(flagged=False), "verifier_checkpoint_refreshed": False})[
        "decision"
    ] == "checkpoint_not_refreshed"
    assert mod._heldout_readiness(None) == {}
    assert mod._heldout_readiness({**_heldout(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert mod._heldout_readiness(
        {
            **_heldout(),
            "heldout_first_win_delta_vs_baseline": 0.02,
            "heldout_first_win_delta_vs_prior_best": 0.01,
        }
    )["decision"] == "heldout_readiness_changed"
    assert mod._silent_bug_audit(None) == {}
    assert mod._silent_bug_audit({**_audit(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert mod._submission_package_state(None) == {}
    assert mod._submission_package_state({**_package(flagged=False), "submission_package_ready": False})[
        "decision"
    ] == "package_not_ready"
    assert mod._hardware_continuity(None) == {}
    assert mod._hardware_continuity({**_hardware(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert mod._hardware_continuity({"experiment": 4777})["decision"] == "kv260_unreachable"
    assert mod._sota_handoff(None) == {}
    assert mod._sota_handoff({**_sota(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert mod._sota_handoff({"honest_verdict": "complete: no methods"})["decision"] == (
        "sota_handoff_empty"
    )
    readiness = mod._readiness(_heldout_readiness := mod._heldout_readiness(_heldout()), {})
    assert readiness["heldout_decision"] == _heldout_readiness["decision"]
    assert readiness["ready_for_operator_submit"] is False


def test_blocker_order_and_registry_yaml_error_are_recorded(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4779-BLOCKED-PRECONDITION: blocked resources are named."""

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
        == "spec_missing_req_4779"
    )

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4779\n", encoding="utf-8")
    (tmp_path / mod.SUMMARIZER_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.SUMMARIZER_RELATIVE_PATH).write_text("# placeholder\n", encoding="utf-8")

    artifact = mod.run_capstone(
        root=tmp_path,
        summarizer=lambda _root, relative_path: mod.SummarizerResult(
            command=["summarize", relative_path],
            exit_code=0,
            stdout="clean",
            stderr="",
        ),
    )

    assert artifact["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert artifact["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(artifact) == []
