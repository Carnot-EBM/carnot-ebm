"""Tests for REQ-CAPSTONE-4769 / SCENARIO-CAPSTONE-4769."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4769_capstone_v438 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _s0(*, passed: bool = False, oracle: bool = False) -> JsonDict:
    return {
        "experiment": "experiment_4761_structural_energy_s0_core_bet_probe",
        "experiment_id": 4761,
        "honest_verdict": "complete: structural_energy_s0_retired_loo_0.746_null_or_leaky",
        "s0_gate_passed": passed,
        "retire_energy_guided_direction": not passed,
        "retire_if_same_verdict": not passed,
        "loo_auroc_structural": 0.7455881880622204,
        "loo_auroc_majority_control": 0.5,
        "loo_auroc_marginal_control": 0.4645783964623073,
        "origin_probe_auroc": 0.7327927210707903,
        "structural_minus_marginal_delta_ci95": [0.17481816435901296, 0.39015500191631836],
        "n_held_out_games": 16,
        "n_candidate_rows": 463,
        "verifier_is_oracle": oracle,
    }


def _levelup(*, oracle: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_4762_levelup_attempt",
        "honest_verdict": "complete_re86_no_new_level_residual_existing_depth",
        "new_levels_banked": 0,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "target_game": "re86",
        "registry_update": {
            "path": "ops/arc_solve_registry.yaml",
            "reason": "no_new_level_banked",
            "reproducible_total_levels_before": 65,
            "reproducible_total_levels_after": 65,
            "updated": False,
        },
        "verifier_is_oracle": oracle,
    }


def _self_play(*, flagged: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_4763_self_play_verifier_checkpoint",
        "honest_verdict": "success_re86_L2_checkpoint_refreshed",
        "flagged_adversarial": flagged,
        "verifier_checkpoint_refreshed": True,
        "self_play_residual": "checkpoint_refreshed_gate_passed",
        "offline_reproduced": True,
        "reproduced_levels": 2,
    }


def _heldout() -> JsonDict:
    return {
        "experiment": "experiment_4764_heldout_first_win_readiness",
        "experiment_id": 4764,
        "honest_verdict": "complete: heldout_first_win_flat_genuine_null",
        "heldout_first_win_rate": 0.04,
        "first_win_baseline": 0.04,
        "prior_best_heldout_first_win_rate": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.0,
        "heldout_first_win_delta_vs_prior_best": 0.0,
        "parity_test_green": True,
        "positive_control_passed": True,
    }


def _bug_audit() -> JsonDict:
    return {
        "experiment": "experiment_4765_silent_bug_audit",
        "honest_verdict": "complete_arc_null_silent_bug_audit_3_nulls_1_reopen",
        "nulls_audited": 3,
        "trusted_nulls": [
            "experiment_4762_levelup_attempt",
            "experiment_4764_heldout_first_win_readiness",
        ],
        "silent_bugs_found": [
            {
                "null_id": "experiment_4761_structural_energy_s0_core_bet_probe",
                "verdict": "silent_bug_must_reopen",
                "silent_bug_signatures": ["s0_origin_probe_leak"],
            }
        ],
        "verifier_is_oracle": False,
    }


def _package(*, flagged: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_4766_submission_package_harden",
        "honest_verdict": "success_package_builds_vram_gate_green",
        "flagged_adversarial": flagged,
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
    }


def _hardware() -> JsonDict:
    return {
        "experiment": 4767,
        "honest_verdict": "success: kv260_continuity_recorded",
        "kv260_ssh_reachable": True,
        "loaded_overlay": "carnot_ising_v2_n64",
        "verifier_is_oracle": False,
    }


def _sota() -> JsonDict:
    return {
        "honest_verdict": "success_sota_ingestion_structural_energy_mapped",
        "s0_context": {
            "imported_honest_verdict": "complete: structural_energy_s0_retired_loo_0.746_null_or_leaky",
            "loo_auroc_structural": 0.7455881880622204,
            "origin_probe_auroc": 0.7327927210707903,
        },
        "flagged_for_v439": [
            {"candidate": "slot_factor_transition_energy_rerun"},
            {"candidate": "poe_code_world_model_trust_gate"},
        ],
        "methods_mapped": [
            {"method": "Slot-factor contrastive transition energy"},
            {"method": "Product-of-experts executable world-model trust gate"},
        ],
    }


def _artifacts(
    *,
    s0: JsonDict | None = None,
    self_play: JsonDict | None = None,
    package: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "S0": s0 or _s0(),
        "LEVELUP": _levelup(),
        "SELF_PLAY": self_play or _self_play(),
        "HELDOUT": _heldout(),
        "BUG_AUDIT": _bug_audit(),
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


def test_req_capstone_4769_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4769: OpenSpec declares the .438 scorecard before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4769_skips_flagged_and_retires_s0() -> None:
    """SCENARIO-CAPSTONE-4769: flagged artifacts are skipped and S0 is RETIRED."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert (
        artifact["honest_verdict"] == "complete: s0_structural_energy_retired_v438_capstone_ready"
    )
    assert artifact["s0_structural_energy_verdict"]["direction"] == "RETIRED"
    assert artifact["s0_structural_energy_verdict"]["s1_queued"] is False
    assert artifact["s0_structural_energy_verdict"]["s0_gate_passed"] is False
    assert artifact["s0_structural_energy_verdict"]["loo_auroc_structural"] == pytest.approx(
        0.7455881880622204
    )
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["levelup_bank"] == {
        "source": "LEVELUP",
        "experiment_id": 4762,
        "target_game": "re86",
        "new_levels_banked": 0,
        "reproducible_total_levels_before": 65,
        "reproducible_total_levels_after": 65,
        "reproducible_total_levels_delta": 0,
        "registry_updated": False,
        "verifier_is_oracle": True,
        "moat_claim": False,
    }
    assert artifact["self_play_checkpoint"] == {
        "source": "SELF_PLAY",
        "experiment_id": 4763,
        "decision": "skipped_flagged_adversarial",
    }
    assert artifact["submission_package_state"] == {
        "source": "PACKAGE",
        "experiment_id": 4766,
        "decision": "skipped_flagged_adversarial",
    }
    assert artifact["heldout_readiness"]["decision"] == "flat_null_no_readiness_gain"
    assert artifact["silent_bug_audit"]["s0_reopened_for_origin_probe_leak"] is True

    skipped = {row["experiment_id"]: row for row in artifact["flagged_artifacts_skipped"]}
    assert set(skipped) == {4763, 4766}

    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited[4763]["fields_imported"] == ["flagged_adversarial"]
    assert cited[4766]["fields_imported"] == ["flagged_adversarial"]
    assert "submission_package_ready" not in cited[4766]["fields_imported"]
    assert "verifier_checkpoint_refreshed" not in cited[4763]["fields_imported"]
    assert "loo_auroc_structural" in cited[4761]["fields_imported"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4769_alive_requires_clean_non_oracle_s0() -> None:
    """SCENARIO-CAPSTONE-4769: only clean non-oracle S0 evidence can queue S1."""

    alive = mod.build_artifact(
        artifacts=_artifacts(s0=_s0(passed=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    oracle = mod.build_artifact(
        artifacts=_artifacts(s0=_s0(passed=True, oracle=True)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert alive["honest_verdict"] == "success: s0_structural_energy_alive_s1_queued"
    assert alive["s0_structural_energy_verdict"]["direction"] == "ALIVE"
    assert alive["s0_structural_energy_verdict"]["s1_queued"] is True
    assert oracle["s0_structural_energy_verdict"]["direction"] == "RETIRED"
    assert oracle["s0_structural_energy_verdict"]["reason"] == "s0_oracle_not_moat"
    assert oracle["upstream_oracle_declarations"]["S0"]["verifier_is_oracle"] is True
    assert oracle["upstream_oracle_declarations"]["S0"]["moat_claim_allowed"] is False


def test_run_capstone_invokes_summarizer_for_every_upstream_and_writes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4769: runtime aggregation reads every upstream via the summarizer."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        "schema_version: 1\nreproducible_total_levels: 65\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4769\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.SUMMARIZER_RELATIVE_PATH).write_text("# placeholder\n", encoding="utf-8")

    called: list[str] = []

    def summarizer(root: Path, relative_path: str) -> mod.SummarizerResult:
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
    """SCENARIO-CAPSTONE-4769-BLOCKED-PRECONDITION: missing upstreams fail closed."""

    for key, payload in _artifacts().items():
        if key != "SOTA":
            _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        "schema_version: 1\nreproducible_total_levels: 65\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4769\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
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
    assert artifact["s0_structural_energy_verdict"] == {}
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["upstream_artifacts"]["SOTA"]["present"] is False
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations() -> None:
    """SCENARIO-CAPSTONE-4769: malformed scorecards fail validation."""

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
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4761}]}
    )
    assert "flagged_artifact_imported_metrics:4766" in mod.validate_artifact(
        {
            **artifact,
            "cited_upstream_artifacts": [
                *artifact["cited_upstream_artifacts"],
                {
                    "experiment_id": 4766,
                    "fields_imported": ["submission_package_ready"],
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


def test_defensive_scorecard_branches_are_explicit() -> None:
    """SCENARIO-CAPSTONE-4769-BLOCKED-PRECONDITION: defensive branches fail closed."""

    assert mod._int(True, 7) == 7
    assert mod._int("x", 9) == 9
    assert mod._float(True) is None
    assert mod._float("x") is None
    assert mod._cited_artifacts({"S0": _s0()}, {})[0]["experiment_id"] == 4761

    flagged_s0 = mod._s0_verdict({**_s0(passed=True), "flagged_adversarial": True})
    assert flagged_s0["direction"] == "RETIRED"
    assert flagged_s0["reason"] == "s0_flagged_skipped"
    assert mod._s0_verdict(None) == {}

    assert mod._levelup_bank(None) == {}
    assert mod._levelup_bank({**_levelup(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert mod._self_play_checkpoint(None) == {}
    assert (
        mod._self_play_checkpoint(
            {
                "experiment": "experiment_4763_self_play_verifier_checkpoint",
                "verifier_checkpoint_refreshed": False,
            }
        )["decision"]
        == "checkpoint_not_refreshed"
    )
    assert mod._heldout_readiness(None) == {}
    assert mod._heldout_readiness({**_heldout(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    changed_heldout = {
        **_heldout(),
        "heldout_first_win_delta_vs_baseline": 0.02,
        "heldout_first_win_delta_vs_prior_best": 0.01,
    }
    assert mod._heldout_readiness(changed_heldout)["decision"] == "heldout_readiness_changed"
    assert mod._silent_bug_audit(None) == {}
    assert mod._silent_bug_audit({**_bug_audit(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert (
        mod._silent_bug_audit({"silent_bugs_found": {}, "trusted_nulls": {}})[
            "silent_bugs_found_count"
        ]
        == 0
    )
    assert mod._submission_package_state(None) == {}
    assert (
        mod._submission_package_state(
            {
                "experiment": "experiment_4766_submission_package_harden",
                "submission_package_ready": False,
            }
        )["decision"]
        == "package_not_ready"
    )
    assert mod._hardware_continuity(None) == {}
    assert mod._hardware_continuity({**_hardware(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert mod._hardware_continuity({"experiment": 4767})["decision"] == "kv260_unreachable"
    assert mod._sota_handoff(None) == {}
    assert mod._sota_handoff({**_sota(), "flagged_adversarial": True})["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert mod._sota_handoff({"honest_verdict": "complete: no methods"})["decision"] == (
        "sota_handoff_empty"
    )


def test_blocker_order_and_registry_yaml_error_are_recorded(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4769-BLOCKED-PRECONDITION: blocked resources are named."""

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
        == "spec_missing_req_4769"
    )

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: [unterminated\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4769\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
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
