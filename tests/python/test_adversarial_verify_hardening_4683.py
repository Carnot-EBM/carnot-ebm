"""Tests for Exp 4683 adversarial_verify .431 hardening.

Spec refs: REQ-ARC-WMTE-4683,
SCENARIO-ARC-WMTE-4683-SUBGOAL-DECOMPOSITION,
SCENARIO-ARC-WMTE-4683-COVERAGE-BASELINE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
A1_PATH = REPO / "results" / "experiment_4676_hierarchical_subgoal_search_live.json"
A2_PATH = REPO / "results" / "experiment_4677_poe_world_factored_subgoal_planner.json"


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report_for_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return av.verify_artifact(_write_payload(tmp_path, payload))


def _flag_kind(report: dict[str, Any], kind: str) -> list[dict[str, str]]:
    return [flag for flag in report["flags"] if flag["kind"] == kind]


def _flag_kinds(report: dict[str, Any], kinds: set[str]) -> list[dict[str, str]]:
    return [flag for flag in report["flags"] if flag["kind"] in kinds]


def _subgoal_search_overclaim_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4683_hierarchical_subgoal_search_overclaim_fixture",
        "game": "lp85",
        "headline": "generic agent reached L2 via hierarchical subgoal search",
        "honest_verdict": "success: hierarchical_subgoal_generic_agent_new_level_lp85_L2",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "generic_agent_reached_level": {"lp85": 2},
        "reproduced_levels": {"lp85": 1},
        "random_seed": 4683,
        "reproducibility_checksum": "sha256:" + "e" * 64,
    }
    payload.update(overrides)
    return payload


def _coverage_overclaim_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4683_generation_coverage_overclaim_fixture",
        "game": "ar25",
        "headline": "candidate-generation coverage up with factored subgoal planner",
        "honest_verdict": "success: poe_world_factored_planner_coverage_up_live_firstwin_lift_ar25",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "candidate_generation_coverage_factored": 0.60,
        "coverage_delta": 0.40,
        "random_seed": 4683,
        "reproducibility_checksum": "sha256:" + "f" * 64,
    }
    payload.update(overrides)
    return payload


def test_req_arc_wmte_4683_spec_declares_subgoal_and_coverage_guards() -> None:
    """REQ-ARC-WMTE-4683: OpenSpec declares both adversarial_verify guards."""

    from carnot import experiment_4683_adversarial_verify_hardening as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4683" in spec
    assert "SCENARIO-ARC-WMTE-4683-SUBGOAL-DECOMPOSITION" in spec
    assert "SCENARIO-ARC-WMTE-4683-COVERAGE-BASELINE" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4683_subgoal_evidence_omitted_warn_and_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4683-SUBGOAL-DECOMPOSITION: omitted evidence flags."""

    report = _report_for_payload(tmp_path, _subgoal_search_overclaim_fixture())
    critical = _flag_kind(
        report, av.SUBGOAL_SEARCH_WITHOUT_DECOMPOSITION_EVIDENCE_KIND
    )
    omitted = _flag_kind(report, av.SUBGOAL_DECOMPOSITION_EVIDENCE_OMITTED_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert omitted
    assert omitted[0]["severity"] == "warn"
    assert "subgoal_decomposition" in omitted[0]["detail"]
    assert "offline_reproduced" in critical[0]["detail"]


def test_scenario_arc_wmte_4683_subgoal_ablation_not_lower_is_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4683-SUBGOAL-DECOMPOSITION: ablations must be lower."""

    report = _report_for_payload(
        tmp_path,
        _subgoal_search_overclaim_fixture(
            subgoal_decomposition=["unlock left portal", "enter target frame"],
            per_subgoal_reachable=[True, True],
            no_subgoal_ablation_reached_level={"lp85": 2},
            random_subgoal_ablation_reached_level={"lp85": 1},
            offline_reproduced={"lp85": True},
        ),
    )
    flags = _flag_kind(report, av.SUBGOAL_SEARCH_WITHOUT_DECOMPOSITION_EVIDENCE_KIND)

    assert flags
    assert flags[0]["severity"] == "critical"
    assert "strictly lower" in flags[0]["detail"]
    assert _flag_kind(report, av.SUBGOAL_DECOMPOSITION_EVIDENCE_OMITTED_KIND) == []


def test_scenario_arc_wmte_4683_subgoal_evidence_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4683-SUBGOAL-DECOMPOSITION: honest evidence is clean."""

    report = _report_for_payload(
        tmp_path,
        _subgoal_search_overclaim_fixture(
            subgoal_decomposition=["unlock left portal", "enter target frame"],
            per_subgoal_reachable=[True, True],
            no_subgoal_ablation_reached_level={"lp85": 1},
            random_subgoal_ablation_reached_level={"lp85": 0},
            offline_reproduced={"lp85": True},
        ),
    )

    assert _flag_kind(report, av.SUBGOAL_SEARCH_WITHOUT_DECOMPOSITION_EVIDENCE_KIND) == []
    assert _flag_kind(report, av.SUBGOAL_DECOMPOSITION_EVIDENCE_OMITTED_KIND) == []


def test_scenario_arc_wmte_4683_coverage_baseline_omitted_warn_and_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4683-COVERAGE-BASELINE: omitted baseline flags."""

    report = _report_for_payload(tmp_path, _coverage_overclaim_fixture())
    critical = _flag_kind(report, av.GENERATION_COVERAGE_WITHOUT_BASELINE_KIND)
    omitted = _flag_kind(report, av.GENERATION_COVERAGE_BASELINE_OMITTED_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert omitted
    assert omitted[0]["severity"] == "warn"
    assert "candidate_generation_coverage_flat_baseline" in critical[0]["detail"]


def test_scenario_arc_wmte_4683_coverage_baseline_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4683-COVERAGE-BASELINE: matched baseline is clean."""

    report = _report_for_payload(
        tmp_path,
        _coverage_overclaim_fixture(candidate_generation_coverage_flat_baseline=0.20),
    )

    assert _flag_kind(report, av.GENERATION_COVERAGE_WITHOUT_BASELINE_KIND) == []
    assert _flag_kind(report, av.GENERATION_COVERAGE_BASELINE_OMITTED_KIND) == []


def test_req_arc_wmte_4683_honest_a1_a2_fixtures_not_false_flagged() -> None:
    """REQ-ARC-WMTE-4683: honest .431 A1/A2 artifacts do not fire new guards."""

    guarded_kinds = {
        av.SUBGOAL_SEARCH_WITHOUT_DECOMPOSITION_EVIDENCE_KIND,
        av.SUBGOAL_DECOMPOSITION_EVIDENCE_OMITTED_KIND,
        av.GENERATION_COVERAGE_WITHOUT_BASELINE_KIND,
        av.GENERATION_COVERAGE_BASELINE_OMITTED_KIND,
    }

    assert _flag_kinds(av.verify_artifact(A1_PATH), guarded_kinds) == []
    assert _flag_kinds(av.verify_artifact(A2_PATH), guarded_kinds) == []


def test_req_arc_wmte_4683_helper_edges_keep_guards_narrow() -> None:
    """REQ-ARC-WMTE-4683: helper edges ignore nulls and non-coverage claims."""

    subgoal_null_flags: list[av.Flag] = []
    av.check_subgoal_search_decomposition_overclaim(
        {
            "experiment": "experiment_4676_hierarchical_subgoal_search_live",
            "game": "lp85",
            "honest_verdict": "complete: hierarchical_subgoal_no_new_level_residual",
            "generic_agent_reached_level": 0,
            "reproduced_levels": 0,
        },
        subgoal_null_flags,
    )
    assert subgoal_null_flags == []

    coverage_null_flags: list[av.Flag] = []
    av.check_generation_coverage_baseline_overclaim(
        {
            "experiment": "experiment_4677_poe_world_factored_subgoal_planner",
            "game": "ar25",
            "honest_verdict": "complete: poe_world_factored_planner_no_coverage_gain_residual_logged",
            "candidate_generation_coverage_factored": 0.0,
            "coverage_delta": 0.0,
        },
        coverage_null_flags,
    )
    assert coverage_null_flags == []


def test_req_arc_wmte_4683_runner_builds_required_terminal_artifact() -> None:
    """REQ-ARC-WMTE-4683: Exp 4683 emits the required evidence fields."""

    from carnot import experiment_4683_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_import_ok": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4683": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert artifact["honest_verdict"] == (
        "success: "
        "adversarial_verify_hardened_subgoal_decomposition_and_coverage_baseline_guards_tests_green."
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["subgoal_decomposition_guard_added"] is True
    assert artifact["coverage_baseline_guard_added"] is True
    assert artifact["honest_artifacts_not_flagged"] is True
    assert artifact["tests_added"]["passed"] is True
    assert artifact["random_seed"] == 4683
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_4683_runner_validation_rejects_malformed_artifact() -> None:
    """REQ-ARC-WMTE-4683: artifact validation fails closed."""

    from carnot import experiment_4683_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_import_ok": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4683": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["inference_substrate"] = "wrong"
    bad["subgoal_decomposition_guard_added"] = False
    bad["coverage_baseline_guard_added"] = False
    bad["honest_artifacts_not_flagged"] = False
    bad["tests_added"] = {"passed": False}
    bad["random_seed"] = 0
    bad["preconditions_checked"] = {"ok": False}
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "sha256:bad"

    errors = mod.validate_artifact(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "subgoal_decomposition_guard_added" in errors
    assert "coverage_baseline_guard_added" in errors
    assert "honest_artifacts_not_flagged" in errors
    assert "tests_added.passed" in errors
    assert "random_seed" in errors
    assert "preconditions_checked.ok" in errors
    assert "field_principles.honest_verdict" in errors
    assert "reproducibility_checksum" in errors

    bad_shape = dict(artifact)
    bad_shape["tests_added"] = None
    bad_shape["preconditions_checked"] = None
    bad_shape["field_principles"] = None
    bad_shape["reproducibility_checksum"] = "sha256:bad"
    shape_errors = mod.validate_artifact(bad_shape)

    assert "tests_added" in shape_errors
    assert "preconditions_checked" in shape_errors
    assert "field_principles" in shape_errors
