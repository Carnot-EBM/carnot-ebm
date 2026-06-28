"""Tests for REQ-CAPSTONE-4901 / SCENARIO-CAPSTONE-4901."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4901_capstone_v451 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1(*, flagged: bool = False, ci: list[float] | None = None) -> JsonDict:
    return {
        "experiment_id": 4892,
        "honest_verdict": "complete_decision_need_no_value_lift_VALUE_GAP_REPRESENTATION_INVARIANT",
        "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT",
        "decision_need_value_accuracy_delta_median": -0.101866,
        "decision_need_value_accuracy_delta_ci95": ci or [-0.227708, 0.025266],
        "positive_control_non_degenerate": True,
        "delta_on_truly_heldout_split": True,
        "n_games_measured": 9,
        "engine_cell_recall_median": 0.727273,
        "flagged_adversarial": flagged,
        "inference_substrate": "live_llm_inference",
    }


def _a1b(*, flagged: bool = False, ran_live: bool = True) -> JsonDict:
    return {
        "experiment_id": 4893,
        "honest_verdict": "complete_action_prefix_latent_no_value_lift_representation_invariant_hard",
        "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_HARD",
        "action_prefix_value_accuracy_delta_median": 0.0,
        "action_prefix_value_accuracy_delta_ci95": [-0.134887, 0.025266],
        "positive_control_non_degenerate": True,
        "delta_on_truly_heldout_split": True,
        "ran_genuinely_live": ran_live,
        "n_games_measured": 9,
        "engine_cell_recall_median": 0.3125,
        "flagged_adversarial": flagged,
        "inference_substrate": "live_llm_inference",
    }


def _b1(*, diagnostic: bool = True, a1b_live: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_4897_value_gap_representation_audit",
        "experiment_id": 4897,
        "honest_verdict": "complete_a1_a1b_audited",
        "a1_source_fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT",
        "a1b_source_fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_HARD",
        "a1_genuinely_diagnostic": diagnostic,
        "a1_ran_live_on_gpu0": True,
        "a1_positive_control_non_degenerate_confirmed": True,
        "a1b_ran_genuinely_live": a1b_live,
        "numbers_match_fork": True,
        "a1_failure_reasons": [] if diagnostic else ["a1_not_genuinely_diagnostic"],
        "a1b_failure_reasons": [] if a1b_live else ["a1b_not_genuinely_live"],
        "checks": {
            "a1_numbers_match_fork": {
                "computed_fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT",
                "passed": True,
            },
            "a1b_live_and_split": {
                "status": "ran",
                "ran_genuinely_live": a1b_live,
                "passed": a1b_live,
            },
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _levelup() -> JsonDict:
    return {
        "experiment": "experiment_4894_levelup_attempt",
        "honest_verdict": "complete_dc22_no_new_level_residual_duplicate_depth",
        "target_game": "dc22",
        "new_levels_banked": 0,
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "registry_update": {
            "prior_total_declared": 68,
            "new_total_declared": 68,
            "reason": "duplicate_depth",
            "updated": True,
        },
        "reproduction_gate": {"claimed_level": 2, "reached_level": 2, "reproduced": True},
        "inference_substrate": "offline_arcade_reproduction_gate_no_llm",
    }


def _self_play() -> JsonDict:
    return {
        "experiment": "experiment_4895_self_play_verifier_checkpoint",
        "honest_verdict": "success_self_play_checkpoint_refreshed",
        "target_game": "sk48",
        "verifier_checkpoint_refreshed": True,
        "checkpoint_path": "models/arc_verifier_sk48.json",
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "search_state_count": 42,
        "inference_substrate": "live_llm_inference",
    }


def _heldout(*, flagged: bool = True) -> JsonDict:
    return {
        "experiment_id": 4896,
        "honest_verdict": "complete_heldout_first_win_0.052632_ci_lower_0_soft_budget_partial",
        "flagged_adversarial": flagged,
        "heldout_first_win_rate": 0.052632,
        "first_win_baseline": 0.04,
        "heldout_first_win_delta_vs_baseline": 0.012632,
        "positive_control_passed": True,
        "parity_test_green": True,
        "live_agent_ran": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "inference_substrate": "live_llm_inference",
    }


def _package() -> JsonDict:
    return {
        "experiment": "experiment_4898_submission_package_harden",
        "honest_verdict": "success_submission_package_ready_final_pre_deadline",
        "submission_package_ready": True,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "vram_estimate_gb": 15.146,
        "package_builds": {"package_builds": True, "dry_build_ran": True},
        "agent_config_resolution": {"resolved": True},
        "model_path_resolution": {"resolved": True},
        "packaging_requirements_crosscheck": {"ok": True},
        "ready_package_regression_check": {"ok": True},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(
    *,
    a1: JsonDict | None = None,
    a1b: JsonDict | None = None,
    b1: JsonDict | None = None,
    heldout: JsonDict | None = None,
) -> dict[str, JsonDict]:
    return {
        "A1": a1 or _a1(),
        "A1B": a1b or _a1b(),
        "B1_AUDIT": b1 or _b1(),
        "A2_LEVELUP": _levelup(),
        "A3_SELF_PLAY": _self_play(),
        "A4_HELDOUT": heldout or _heldout(),
        "B2_PACKAGE": _package(),
    }


def _hashes() -> dict[str, str]:
    return {source: f"sha256:{source.lower()}" for source in mod.UPSTREAM_SOURCES}


def _summaries(*, heldout_code: int = 2, a1b_code: int = 0) -> dict[str, mod.SummarizerResult]:
    return {
        source: mod.SummarizerResult(
            ["summarize", spec.relative_path],
            heldout_code if source == "A4_HELDOUT" else a1b_code if source == "A1B" else 0,
            "CRITICAL TAUTOLOGY" if source == "A4_HELDOUT" and heldout_code >= 2 else "clean",
            "",
        )
        for source, spec in mod.UPSTREAM_SOURCES.items()
    }


def test_req_capstone_4901_spec_declares_scorecard_contract() -> None:
    """REQ-CAPSTONE-4901: OpenSpec declares the .451 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-4901") :]

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH in section
    for source in mod.UPSTREAM_SOURCES.values():
        assert source.relative_path in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_capstone_4901_default_escalates_representation_invariant() -> None:
    """SCENARIO-CAPSTONE-4901: trusted A1/A1b nulls become the escalation headline."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete_capstone_v451_representation_invariant_escalate_operator"
    assert artifact["fork_verdict_trusted"] is True
    assert artifact["change_value_gap_representation_invariant"] is True
    assert "0.08" in artifact["operator_escalation_note"]
    assert ".452" in artifact["operator_escalation_note"]
    assert artifact["representation_fork_verdict"]["verdict"] == "representation_invariant_escalate_operator"
    assert artifact["representation_fork_verdict"]["trust_gate"] == {
        "a1_genuinely_diagnostic": True,
        "a1b_ran_genuinely_live": True,
        "a1b_required": True,
    }
    assert artifact["representation_fork_verdict"]["a1"]["failed_to_move_change_value_accuracy"] is True
    assert artifact["representation_fork_verdict"]["a1b"]["failed_to_move_change_value_accuracy"] is True
    assert artifact["representation_fork_verdict"]["a1"]["delta_ci95_includes_zero"] is True
    assert artifact["representation_fork_verdict"]["a1b"]["delta_ci95_includes_zero"] is True
    assert artifact["reproducible_total_levels"] == 68

    scorecard = artifact["deadline_lever_scorecard"]
    assert scorecard["a2_bank"]["new_levels_banked"] == 0
    assert scorecard["a2_bank"]["decision"] == "no_new_level_banked"
    assert scorecard["a3_self_play"]["decision"] == "checkpoint_refreshed"
    assert scorecard["a4_fresh_live_rate"] == {
        "source": "A4_HELDOUT",
        "experiment_id": 4896,
        "status": "skipped_flagged_adversarial",
        "reason": "flagged_adversarial",
    }
    assert scorecard["b2_package"]["decision"] == "package_ready_operator_only"
    assert scorecard["deadline_deliverable"] == "current_scored_agent_0.08"
    assert artifact["flagged_upstreams_skipped"] == [
        {
            "source": "A4_HELDOUT",
            "experiment_id": 4896,
            "path": mod.UPSTREAM_SOURCES["A4_HELDOUT"].relative_path,
            "reason": "flagged_adversarial",
            "sha256": "sha256:a4_heldout",
            "summarizer_exit_code": 2,
        }
    ]
    assert 4896 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4901_untrusted_or_flagged_a1b_is_not_escalation() -> None:
    """SCENARIO-CAPSTONE-4901: untrusted or skipped forks are non-tests."""

    untrusted = mod.build_artifact(
        artifacts=_artifacts(b1=_b1(diagnostic=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert untrusted["honest_verdict"] == "complete_capstone_v451_untrusted_fork_non_test"
    assert untrusted["fork_verdict_trusted"] is False
    assert untrusted["change_value_gap_representation_invariant"] is False
    assert untrusted["operator_escalation_note"] == ""

    skipped_a1b = mod.build_artifact(
        artifacts=_artifacts(a1b=_a1b(flagged=True), b1=_b1()),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(a1b_code=2),
        duration_s=0.001,
    )
    assert skipped_a1b["fork_verdict_trusted"] is True
    assert skipped_a1b["change_value_gap_representation_invariant"] is False
    assert skipped_a1b["representation_fork_verdict"]["a1b"] == {"status": "skipped"}
    assert skipped_a1b["operator_escalation_note"] == ""
    assert {row["source"] for row in skipped_a1b["flagged_upstreams_skipped"]} == {
        "A1B",
        "A4_HELDOUT",
    }

    a1b_not_live = mod.build_artifact(
        artifacts=_artifacts(b1=_b1(a1b_live=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )
    assert a1b_not_live["representation_fork_verdict"]["trust_failure_reasons"] == [
        "a1b_ran_genuinely_live"
    ]

    included_a4 = mod.build_artifact(
        artifacts=_artifacts(heldout=_heldout(flagged=False)),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(heldout_code=0),
        duration_s=0.001,
    )
    assert included_a4["deadline_lever_scorecard"]["a4_fresh_live_rate"]["status"] == "included"
    assert (
        included_a4["deadline_lever_scorecard"]["a4_fresh_live_rate"]["heldout_first_win_rate"]
        == 0.052632
    )


def test_run_capstone_invokes_summarizer_and_blocks_missing_required(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4901-BLOCKED-PRECONDITION: runtime checks fail closed."""

    for key, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 68\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4901\n", encoding="utf-8")
    summarizer_path = tmp_path / mod.SUMMARIZER_RELATIVE_PATH
    summarizer_path.parent.mkdir(parents=True, exist_ok=True)
    summarizer_path.write_text("# placeholder\n", encoding="utf-8")
    called: list[str] = []

    def summarizer(_root: Path, relative_path: str) -> mod.SummarizerResult:
        called.append(relative_path)
        code = 2 if relative_path == mod.UPSTREAM_SOURCES["A4_HELDOUT"].relative_path else 0
        return mod.SummarizerResult(["summarize", relative_path], code, "clean", "")

    artifact = mod.run_capstone(root=tmp_path, summarizer=summarizer)
    assert sorted(called) == sorted(spec.relative_path for spec in mod.UPSTREAM_SOURCES.values())
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["capstone_ready"] is True
    assert mod.validate_artifact(artifact) == []

    missing_root = tmp_path / "missing"
    for key, payload in _artifacts().items():
        if key != "B1_AUDIT":
            _write_json(missing_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    missing_registry = missing_root / mod.REGISTRY_RELATIVE_PATH
    missing_registry.parent.mkdir(parents=True, exist_ok=True)
    missing_registry.write_text(
        "schema_version: 1\nreproducible_total_levels: 68\n", encoding="utf-8"
    )
    missing_spec = missing_root / mod.SPEC_RELATIVE_PATH
    missing_spec.parent.mkdir(parents=True, exist_ok=True)
    missing_spec.write_text("REQ-CAPSTONE-4901\n", encoding="utf-8")
    missing_summarizer = missing_root / mod.SUMMARIZER_RELATIVE_PATH
    missing_summarizer.parent.mkdir(parents=True, exist_ok=True)
    missing_summarizer.write_text("# placeholder\n", encoding="utf-8")

    blocked = mod.run_capstone(root=missing_root, summarizer=summarizer)
    assert blocked["honest_verdict"] == "blocked_b1_artifact_missing"
    assert blocked["representation_fork_verdict"] == {}
    assert blocked["fork_verdict_trusted"] is False
    assert blocked["change_value_gap_representation_invariant"] is False
    assert blocked["deadline_lever_scorecard"] == {}
    assert blocked["preconditions_checked"]["upstream_artifacts"]["B1_AUDIT"]["present"] is False
    assert mod.validate_artifact(blocked) == []

    bad_registry_root = tmp_path / "bad_registry"
    for key, payload in _artifacts().items():
        _write_json(bad_registry_root / mod.UPSTREAM_SOURCES[key].relative_path, payload)
    bad_registry = bad_registry_root / mod.REGISTRY_RELATIVE_PATH
    bad_registry.parent.mkdir(parents=True, exist_ok=True)
    bad_registry.write_text("schema_version: [unterminated\n", encoding="utf-8")
    bad_spec = bad_registry_root / mod.SPEC_RELATIVE_PATH
    bad_spec.parent.mkdir(parents=True, exist_ok=True)
    bad_spec.write_text("REQ-CAPSTONE-4901\n", encoding="utf-8")
    bad_summarizer = bad_registry_root / mod.SUMMARIZER_RELATIVE_PATH
    bad_summarizer.parent.mkdir(parents=True, exist_ok=True)
    bad_summarizer.write_text("# placeholder\n", encoding="utf-8")

    registry_blocked = mod.run_capstone(root=bad_registry_root, summarizer=summarizer)
    assert registry_blocked["honest_verdict"] == "blocked_registry_not_yaml_loadable"
    assert registry_blocked["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert mod.validate_artifact(registry_blocked) == []


def test_validate_artifact_rejects_schema_errors_and_helpers() -> None:
    """SCENARIO-CAPSTONE-4901-FIELD-PRINCIPLES: malformed scorecards fail validation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 68},
        registry_sha256="sha256:registry",
        summarizer_results=_summaries(),
        duration_s=0.001,
    )

    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "maybe"}
    )
    assert "invalid_inference_substrate" in mod.validate_artifact(
        {**artifact, "inference_substrate": "live_llm_inference"}
    )
    assert "invalid_fork_verdict_trusted" in mod.validate_artifact(
        {**artifact, "fork_verdict_trusted": "yes"}
    )
    assert "invalid_change_value_gap_representation_invariant" in mod.validate_artifact(
        {**artifact, "change_value_gap_representation_invariant": "yes"}
    )
    assert "invalid_reproducible_total_levels" in mod.validate_artifact(
        {**artifact, "reproducible_total_levels": "68"}
    )
    assert "missing_principle:honest_verdict" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert "invalid_representation_fork_verdict" in mod.validate_artifact(
        {**artifact, "representation_fork_verdict": {"verdict": "maybe"}}
    )
    assert "invalid_representation_fork_verdict" in mod.validate_artifact(
        {**artifact, "representation_fork_verdict": []}
    )
    assert "invalid_deadline_lever_scorecard" in mod.validate_artifact(
        {**artifact, "deadline_lever_scorecard": []}
    )
    assert "invalid_flagged_upstreams_skipped" in mod.validate_artifact(
        {**artifact, "flagged_upstreams_skipped": [{"experiment_id": 4896}]}
    )
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 4892}]}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert mod._ci95_includes_zero(None) is False
    assert mod._ci95_includes_zero(["bad", 0.2]) is False
    assert mod._ci95_includes_zero([-0.1, 0.2]) is True
    assert mod._representation_fork_verdict(None, None, None) == {}
    assert mod._a2_bank(None, 68) == {}
    assert mod._a3_self_play(None) == {}
    assert mod._a4_fresh_live({}, {}, {}) == {}
    assert mod._b2_package(None) == {}
    assert mod._experiment_id("A1", {"experiment_id": True}) == 4892
    assert mod._experiment_id("A2_LEVELUP", {"experiment": 4894}) == 4894
    assert mod._experiment_id("B1_AUDIT", {"experiment": "experiment_4897"}) == 4897
    assert mod._is_skipped(_a1(flagged=True), None) == "flagged_adversarial"
    assert mod._is_skipped(_a1(), mod.SummarizerResult([], 2, "", "")) == "live_critical_recheck"
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
    ) == "spec_missing_req_4901"
    assert mod._first_blocker(
        summarizer_present=True,
        registry_present=True,
        registry_loadable=True,
        spec_has_req=True,
        upstream_preconditions={"A1": {"present": False}, "B1_AUDIT": {"present": True}},
    ) == "a1_artifact_missing"
