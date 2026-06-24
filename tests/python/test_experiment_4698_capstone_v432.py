"""Tests for Exp 4698 .432 directed-exploration capstone scorecard.

Spec refs: REQ-CAPSTONE-4698, SCENARIO-CAPSTONE-4698,
SCENARIO-CAPSTONE-4698-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4698-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4698_capstone_v432 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_novelty(
    *,
    reached_level: int = 0,
    offline: bool = False,
    no_novelty_level: int = 0,
    cosmetic_level: int = 0,
    verdict: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4688_controllable_novelty_proposal_policy_live",
        "honest_verdict": verdict
        or (
            f"success: controllable_novelty_generic_agent_new_level_bp35_L{reached_level}"
            if reached_level > 0 and offline
            else "complete: controllable_novelty_no_new_level_residual_winning_prefix_still_not_proposed"
        ),
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "target_game": "bp35",
        "generic_agent_reached_level": reached_level,
        "no_novelty_ablation_reached_level": no_novelty_level,
        "cosmetic_novelty_ablation_reached_level": cosmetic_level,
        "offline_reproduced": offline,
        "reproduced_levels": reached_level if offline else 0,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "duration_s": 60.000427,
        "inference_substrate": "live_llm_inference",
    }


def _a2_filter(
    *,
    filter_coverage: float = 0.0,
    blind_coverage: float | None = 0.0,
    first_win_delta: float = -0.04,
    heldout_rejected: int | None = 2,
    ci: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = {"first_win_rate": 0.04, "variant_attempts_count": 25}
    payload: dict[str, Any] = {
        "experiment": "experiment_4689_program_synthesis_action_effect_proposal_filter",
        "honest_verdict": (
            "success: program_synthesis_filter_coverage_up_heldout_firstwin_lift_bp35"
            if filter_coverage > 0.0 and first_win_delta > 0.0
            else "complete: program_synthesis_filter_no_coverage_gain_residual_logged"
        ),
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "target_games": ["bp35"],
        "candidate_generation_coverage_filter": filter_coverage,
        "coverage_delta": filter_coverage - (blind_coverage if blind_coverage is not None else 0.0),
        "live_baseline_blind_proposal": baseline,
        "live_first_win_rate_filter": baseline["first_win_rate"] + first_win_delta,
        "first_win_rate_delta": first_win_delta,
        "live_lift_ci": dict(ci or {"metric": "first_win_rate_delta", "low": -0.12, "high": 0.0}),
        "heldout_programs_kept": 0,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "offline_reproduced": False,
        "duration_s": 60.000101,
        "inference_substrate": "live_llm_inference",
    }
    if blind_coverage is not None:
        payload["candidate_generation_coverage_blind_baseline"] = blind_coverage
    if heldout_rejected is not None:
        payload["heldout_programs_rejected"] = heldout_rejected
    return payload


def _a3_bank() -> dict[str, Any]:
    return {
        "experiment": "experiment_4690_levelup_selfplay",
        "honest_verdict": "success: lf52_L2_offline_reproduced",
        "verifier_is_oracle": False,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "target_game": "lf52",
        "reproduction_gate": {"game": "lf52", "claimed_level": 2, "reached_level": 2, "reproduced": True},
    }


def _a4_readiness(*, flagged: bool = True, rate: float = 0.04, ci_lower: float = 0.0) -> dict[str, Any]:
    return {
        "experiment": "experiment_4691_held_out_first_win_readiness",
        "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
        "verifier_is_oracle": False,
        "flagged_adversarial": flagged,
        "first_win_rate_integrated": rate,
        "first_win_baseline": 0.04,
        "first_win_delta_vs_baseline": round(rate - 0.04, 6),
        "first_win_ci_lower": ci_lower,
        "multi_level_deepen_rate_integrated": 0.0,
        "held_out_first_win_readiness": bool(rate > 0.04 and ci_lower > 0.0),
        "ready_for_operator_submit": bool(rate > 0.04 and ci_lower > 0.0),
        "replay_count_is_not_the_score": True,
        "replay_floor": {"live_submittable_level_count": 60, "replay_package_floor_reproduced": True},
        "submitted_to_leaderboard": False,
    }


def _a5_transfer() -> dict[str, Any]:
    return {
        "experiment": "experiment_4692_primitive_persist_transfer",
        "honest_verdict": "complete: primitive_persisted_transfer_null_characterized",
        "verifier_is_oracle": False,
        "primitive_persisted": {"operator": "controllable_novelty_embedding_operator"},
        "offline_reproduced_new_level": False,
    }


def _a6_integration() -> dict[str, Any]:
    return {
        "experiment": "experiment_4693_integration_gate",
        "honest_verdict": "complete: integration_unchanged_both_levers_null",
        "verifier_is_oracle": False,
        "config_changed": False,
        "first_win_rate_integrated": 0.04,
        "first_win_rate_delta_vs_pre_integration": 0.0,
        "multi_level_deepen_rate_integrated": 0.0,
        "multi_level_deepen_rate_delta_vs_pre_integration": 0.0,
    }


def _b1_guard() -> dict[str, Any]:
    return {
        "experiment": "experiment_4694_proposal_coverage_cigate",
        "honest_verdict": "success: proposal_coverage_cigate_plus_honest_firstwin_floor_shipped_tests_green",
        "verifier_is_oracle": False,
        "proposal_coverage_metric_added": {"passed": True},
        "honest_firstwin_floor_added": {"passed": True},
        "tests_added": {"passed": True},
    }


def _b2_guard() -> dict[str, Any]:
    return {
        "experiment": "experiment_4695_adversarial_verify_hardening",
        "honest_verdict": "success: adversarial_verify_hardened_novelty_ablation_and_proposal_filter_heldout_guards_tests_green.",
        "verifier_is_oracle": False,
        "novelty_ablation_guard_added": True,
        "proposal_filter_heldout_guard_added": True,
        "tests_added": {"passed": True},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(
    *,
    a1: Mapping[str, Any] | None = None,
    a2: Mapping[str, Any] | None = None,
    a4: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "A1": dict(a1 or _a1_novelty()),
        "A2": dict(a2 or _a2_filter()),
        "A3": _a3_bank(),
        "A4": dict(a4 or _a4_readiness(flagged=True)),
        "A5": _a5_transfer(),
        "A6": _a6_integration(),
        "B1": _b1_guard(),
        "B2": _b2_guard(),
    }


def _preconditions(total: int = 61) -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4698": True,
        "registry_yaml_loadable": True,
        "registry_reproducible_total_levels": total,
        "summarize_artifact_py_available": True,
        "summarize_artifact_py_used_for_every_upstream": True,
        "upstream_artifacts_present": {name: True for name in mod.UPSTREAM_SOURCES},
        "missing_upstream_artifacts": [],
        "leaderboard_submission": False,
        "operator_only": True,
        "research_conductor_modified": False,
    }


def _paper_gate(ready: bool = True) -> dict[str, Any]:
    return {
        "paper_ready": ready,
        "frozen_fover_auroc": 0.9131,
        "gates": {
            "G1": {"pass": ready, "detail": "FoVer dual-condition AUROC artifact present"},
            "G2": {"pass": ready, "detail": "independent reproducer confirmed"},
            "G3": {"pass": ready, "detail": "narrowing-clean"},
            "G4": {"pass": ready, "detail": "numbers trace to artifact"},
        },
        "unmet_gates": [] if ready else ["G2"],
    }


def test_req_capstone_4698_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4698: OpenSpec declares the .432 scorecard fields and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4698" in spec
    assert "SCENARIO-CAPSTONE-4698" in spec
    assert "SCENARIO-CAPSTONE-4698-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4698-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4698_default_null_bridge_counts_registry_growth() -> None:
    """SCENARIO-CAPSTONE-4698: default .432 excludes null/flagged levers from headline."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 61},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: capability_grew_60_to_61"
    assert artifact["bridge_crossed_for_solve"] is False
    assert artifact["a1_controllable_novelty_new_level"]["headline_counted"] is False
    assert artifact["a2_program_synthesis_coverage_and_lift"]["coverage_delta"] == 0.0
    assert artifact["a2_program_synthesis_coverage_and_lift"]["headline_counted"] is False
    assert artifact["cited_upstream_artifacts"]["A4"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["held_out_first_win_readiness"]["headline_counted"] is False
    assert artifact["held_out_first_win_readiness"]["replay_count_is_not_the_score"] is True
    assert artifact["reproducible_total_levels"] == 61
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["paper_ready"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4698_a1_success_requires_two_lower_ablations() -> None:
    """SCENARIO-CAPSTONE-4698: A1 counts only with both novelty ablations lower."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1_novelty(reached_level=1, offline=True, no_novelty_level=0, cosmetic_level=0),
            a4=_a4_readiness(flagged=False),
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 60},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=60),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_generic_new_level_via_directed_exploration_bp35"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["a1_controllable_novelty_new_level"]["headline_counted"] is True
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A1_controllable_novelty"

    collapsed_cosmetic = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1_novelty(reached_level=1, offline=True, no_novelty_level=0, cosmetic_level=1),
            a4=_a4_readiness(flagged=False),
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 60},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=60),
        duration_s=0.001,
    )

    assert collapsed_cosmetic["bridge_crossed_for_solve"] is False
    assert collapsed_cosmetic["cited_upstream_artifacts"]["A1"]["reason"] == "control_failed"
    assert collapsed_cosmetic["flagged_artifacts_handled"]["control_failed_artifacts"] == [
        {"name": "A1", "artifact": "results/experiment_4688_controllable_novelty_proposal_policy_live.json"}
    ]


def test_scenario_capstone_4698_a2_success_requires_coverage_heldout_and_ci() -> None:
    """SCENARIO-CAPSTONE-4698: A2 counts only with coverage, held-out rejection, and CI."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(
            a2=_a2_filter(
                filter_coverage=1.0,
                blind_coverage=0.0,
                first_win_delta=0.12,
                heldout_rejected=2,
                ci={"metric": "first_win_rate_delta", "low": 0.03, "high": 0.15},
            ),
            a4=_a4_readiness(flagged=False),
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 60},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=60),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_generic_new_level_via_directed_exploration_bp35"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["a2_program_synthesis_coverage_and_lift"]["headline_counted"] is True
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A2_program_synthesis_filter"

    missing_heldout = mod.build_artifact(
        artifacts=_artifacts(
            a2=_a2_filter(
                filter_coverage=1.0,
                blind_coverage=0.0,
                first_win_delta=0.12,
                heldout_rejected=None,
                ci={"metric": "first_win_rate_delta", "low": 0.03, "high": 0.15},
            ),
            a4=_a4_readiness(flagged=False),
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 60},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=60),
        duration_s=0.001,
    )
    ci_includes_baseline = mod.build_artifact(
        artifacts=_artifacts(
            a2=_a2_filter(
                filter_coverage=1.0,
                blind_coverage=0.0,
                first_win_delta=0.12,
                heldout_rejected=2,
                ci={"metric": "first_win_rate_delta", "low": -0.01, "high": 0.15},
            ),
            a4=_a4_readiness(flagged=False),
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 60},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=60),
        duration_s=0.001,
    )
    blind_already_had_winner = mod.build_artifact(
        artifacts=_artifacts(
            a2=_a2_filter(
                filter_coverage=1.0,
                blind_coverage=0.5,
                first_win_delta=0.12,
                heldout_rejected=2,
                ci={"metric": "first_win_rate_delta", "low": 0.03, "high": 0.15},
            ),
            a4=_a4_readiness(flagged=False),
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 60},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=60),
        duration_s=0.001,
    )

    assert missing_heldout["cited_upstream_artifacts"]["A2"]["reason"] == "ablation_missing"
    assert missing_heldout["flagged_artifacts_handled"]["ablation_missing_artifacts"] == [
        {"name": "A2", "artifact": "results/experiment_4689_program_synthesis_action_effect_proposal_filter.json"}
    ]
    assert ci_includes_baseline["a2_program_synthesis_coverage_and_lift"]["reason"] == (
        "heldout_first_win_ci_includes_blind_baseline"
    )
    assert blind_already_had_winner["a2_program_synthesis_coverage_and_lift"]["reason"] == (
        "winner_already_in_blind_baseline"
    )


def test_req_capstone_4698_exclusion_guards_cover_oracle_controls_and_live_flags() -> None:
    """REQ-CAPSTONE-4698: dirty upstreams are excluded from clean headline claims."""

    artifacts = _artifacts(a2=_a2_filter(filter_coverage=1.0, blind_coverage=None, first_win_delta=0.12))
    artifacts["A3"]["verifier_is_oracle"] = True
    artifacts["A5"]["acceptance_gate_transfer"] = False
    artifacts["A6"]["false_negative_risk_checked"] = False

    artifact = mod.build_artifact(
        artifacts=artifacts,
        live_flags_by_name={
            "A4": [
                {
                    "kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": "first_win_baseline=0.04 and first_win_rate_integrated=0.04",
                }
            ],
            "B1": [
                {
                    "kind": "FALSE_NEGATIVE_RISK",
                    "severity": "warn",
                    "detail": "false_negative_risk_open: proposal coverage control absent",
                }
            ],
        },
        registry={"reproducible_total_levels": 61},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["cited_upstream_artifacts"]["A2"]["reason"] == "ablation_missing"
    assert artifact["cited_upstream_artifacts"]["A3"]["reason"] == "oracle_not_distinct"
    assert artifact["cited_upstream_artifacts"]["A4"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["cited_upstream_artifacts"]["A5"]["reason"] == "failed_acceptance_gate"
    assert artifact["cited_upstream_artifacts"]["A6"]["reason"] == "positive_control_failed"
    assert artifact["cited_upstream_artifacts"]["B1"]["reason"] == "false_negative_risk_open"
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".431-B2 subgoal-decomposition/coverage-baseline"] is True
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".432-B2 novelty-ablation"] is True
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".432-B2 proposal-filter-heldout"] is True


def test_req_capstone_4698_run_reads_injected_files_and_records_missing(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4698-BLOCKED-PRECONDITION: missing upstreams block clean headlines."""

    for name, payload in _artifacts().items():
        if name == "A5":
            continue
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[name].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("reproducible_total_levels: 61\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4698\n", encoding="utf-8")
    scripts_path = tmp_path / "scripts" / "summarize_artifact.py"
    scripts_path.parent.mkdir(parents=True, exist_ok=True)
    scripts_path.write_text("# test\n", encoding="utf-8")

    artifact = mod.run(tmp_path, live_flags_by_name={}, publication_gate=_paper_gate(), write=True, duration_s=0.001)

    assert artifact["honest_verdict"] == "blocked_upstream_artifacts"
    assert artifact["preconditions_checked"]["missing_upstream_artifacts"] == [
        "results/experiment_4692_primitive_persist_transfer.json"
    ]
    assert artifact["cited_upstream_artifacts"]["A5"]["exists"] is False
    assert artifact["bridge_crossed_for_solve"] is False
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_capstone_4698_validation_and_defensive_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-4698: schema validation, checksum, and defensive helpers fail closed."""

    blocked = mod.build_artifact(
        artifacts=_artifacts(a4=_a4_readiness(flagged=False)),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 61},
        publication_gate=_paper_gate(False),
        preconditions_checked={**_preconditions(), "ok": False, "blocked_resource": "registry_yaml"},
        duration_s=0.001,
    )
    bad = dict(blocked)
    bad["verifier_is_oracle"] = True

    def bad_build(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"honest_verdict": "complete: invalid"}

    assert blocked["honest_verdict"] == "blocked_registry_yaml"
    assert blocked["paper_ready"] is False
    assert mod._read_yaml(tmp_path / "missing.yaml") == {}
    malformed_yaml = tmp_path / "broken.yaml"
    malformed_yaml.write_text("not: [valid\n", encoding="utf-8")
    assert mod._read_yaml(malformed_yaml) == {}
    assert mod._as_float(True, 3.0) == pytest.approx(3.0)
    assert mod._as_float("not-a-number", 3.0) == pytest.approx(3.0)
    assert mod._as_int(False, 4) == 4
    assert mod._as_int("bad", 4) == 4
    assert mod._file_sha256(tmp_path / "missing.json") is None
    assert mod._positive_control_failed({"positive_control_passed": False}) is True
    assert mod._positive_control_failed({"bare_control_passed": False}) is True
    assert mod._ci_excludes_zero({"ci95": "bad"}) is False
    assert mod._ci_excludes_zero({"ci95": [0.01, 0.02]}) is True
    assert mod._has_required_number({"x": "bad"}, "x") is False
    assert mod._a1_controllable_novelty_new_level(
        _a1_novelty(reached_level=1, offline=False, no_novelty_level=0, cosmetic_level=0),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "offline_reproduction_missing"
    assert mod._a1_controllable_novelty_new_level(
        _a1_novelty(reached_level=1, offline=True, no_novelty_level=1, cosmetic_level=0),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "novelty_ablations_not_strictly_lower"
    assert mod._a2_program_synthesis_coverage_and_lift(
        _a2_filter(
            filter_coverage=1.0,
            blind_coverage=0.0,
            first_win_delta=0.0,
            heldout_rejected=2,
            ci={"metric": "first_win_rate_delta", "low": 0.01, "high": 0.02},
        ),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "no_positive_heldout_first_win_lift"
    fallback_delta = _a2_filter(
        filter_coverage=1.0,
        blind_coverage=0.0,
        first_win_delta=0.12,
        heldout_rejected=2,
        ci={"metric": "first_win_rate_delta", "low": 0.01, "high": 0.02},
    )
    fallback_delta.pop("first_win_rate_delta")
    assert mod._a2_program_synthesis_coverage_and_lift(
        fallback_delta,
        {"included_in_headline": True, "reason": "manual"},
    )["first_win_rate_delta"] == pytest.approx(0.12)
    assert mod._a2_program_synthesis_coverage_and_lift(
        _a2_filter(
            filter_coverage=0.0,
            blind_coverage=0.0,
            first_win_delta=0.12,
            heldout_rejected=2,
            ci={"metric": "first_win_rate_delta", "low": 0.01, "high": 0.02},
        ),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "coverage_delta_not_positive"
    assert mod._a2_program_synthesis_coverage_and_lift(
        _a2_filter(
            filter_coverage=1.0,
            blind_coverage=None,
            first_win_delta=0.12,
            heldout_rejected=2,
            ci={"metric": "first_win_rate_delta", "low": 0.01, "high": 0.02},
        ),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "blind_baseline_missing"
    assert mod._a2_program_synthesis_coverage_and_lift(
        _a2_filter(
            filter_coverage=1.0,
            blind_coverage=0.0,
            first_win_delta=0.12,
            heldout_rejected=None,
            ci={"metric": "first_win_rate_delta", "low": 0.01, "high": 0.02},
        ),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "heldout_programs_rejected_missing"
    assert mod._held_out_first_win_readiness(
        _a4_readiness(flagged=False, rate=0.08, ci_lower=0.01),
        {"included_in_headline": True, "reason": "manual"},
    )["headline_counted"] is True
    assert mod._held_out_first_win_readiness(
        _a4_readiness(flagged=False, rate=0.08, ci_lower=0.0),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "held_out_first_win_ci_lower_not_positive"
    monkeypatch.setattr(
        mod,
        "publication_gate_reader",
        type("PublicationGateFixture", (), {"evaluate": staticmethod(lambda: _paper_gate())}),
    )
    assert mod._load_publication_gate()["paper_ready"] is True
    monkeypatch.setattr(
        mod,
        "artifact_reader",
        type(
            "ArtifactReaderFixture",
            (),
            {
                "summarize": staticmethod(lambda _path: 1),
                "_live_flags": staticmethod(lambda _path: [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}]),
            },
        ),
    )
    summary_code, live_flags = mod._summarize_and_live_flags(tmp_path / "fixture.json")
    assert summary_code == 1
    assert live_flags == [{"kind": "METHODOLOGY_MISSING", "severity": "warn"}]
    monkeypatch.setattr(
        mod,
        "_summarize_and_live_flags",
        lambda _path: (
            0,
            [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"}],
        ),
    )
    status = mod._source_status(
        name="X",
        source=mod.SourceSpec("X", "results/existing.json", "fixture"),
        root=tmp_path,
        artifact={"honest_verdict": "complete: fixture"},
        exists=True,
        live_flags_by_name=None,
    )
    missing_status = mod._source_status(
        name="X",
        source=mod.SourceSpec("X", "results/missing.json", "fixture"),
        root=tmp_path,
        artifact={},
        exists=False,
        live_flags_by_name=None,
    )
    assert status["summary_exit_code"] == 0
    assert status["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert missing_status["reason"] == "missing"
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.write_artifact(path=tmp_path / "bad.json", artifact=bad)
    malformed = dict(blocked)
    malformed["honest_verdict"] = "still_running"
    malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
    assert "honest_verdict must be terminal-prefixed" in mod.validate_artifact(malformed)
    monkeypatch.setattr(mod, "build_artifact", bad_build)
    with pytest.raises(ValueError, match="missing required field"):
        mod.run(tmp_path, write=False, duration_s=0.001)
