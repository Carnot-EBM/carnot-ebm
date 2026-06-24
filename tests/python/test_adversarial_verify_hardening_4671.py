"""Tests for Exp 4671 adversarial_verify .430 hardening.

Spec refs: REQ-ARC-WMTE-4671,
SCENARIO-ARC-WMTE-4671-L2-GOAL-SATISFIABILITY,
SCENARIO-ARC-WMTE-4671-MULTI-LEVEL-NONDEGENERATE-METRIC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
A1_PATH = REPO / "results" / "experiment_4664_l2_goal_predicate_induction_live.json"


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


def _l2_goal_induction_overclaim_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4671_l2_goal_induction_overclaim_fixture",
        "game": "lp85",
        "headline": "generic agent reached L2 via L2 goal induction",
        "honest_verdict": "success: l2_goal_induction_generic_agent_reached_L2_lp85",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "generic_agent_reached_level": {"lp85": 2},
        "offline_reproduced": {"lp85": True},
        "reproduced_levels": {"lp85": 1},
        "random_seed": 4671,
        "reproducibility_checksum": "sha256:" + "c" * 64,
    }
    payload.update(overrides)
    return payload


def _multi_level_metric_overclaim_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4671_multilevel_metric_overclaim_fixture",
        "game": "sc25",
        "headline": "multi-level solve-rate improved on the live ARC agent",
        "honest_verdict": "success: live_multi_level_solve_rate_up",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "live_multi_level_solve_rate": 0.25,
        "random_seed": 4671,
        "reproducibility_checksum": "sha256:" + "d" * 64,
    }
    payload.update(overrides)
    return payload


def test_req_arc_wmte_4671_spec_declares_l2_and_multilevel_guards() -> None:
    """REQ-ARC-WMTE-4671: OpenSpec declares both adversarial_verify guards."""

    from carnot import experiment_4671_adversarial_verify_hardening as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4671" in spec
    assert "SCENARIO-ARC-WMTE-4671-L2-GOAL-SATISFIABILITY" in spec
    assert "SCENARIO-ARC-WMTE-4671-MULTI-LEVEL-NONDEGENERATE-METRIC" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4671_l2_goal_controls_omitted_warn_and_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4671-L2-GOAL-SATISFIABILITY: omitted controls flag."""

    report = _report_for_payload(tmp_path, _l2_goal_induction_overclaim_fixture())
    critical = _flag_kind(
        report, av.L2_GOAL_INDUCTION_WITHOUT_SATISFIABILITY_CHECK_KIND
    )
    omitted = _flag_kind(report, av.L2_GOAL_SATISFIABILITY_CHECK_OMITTED_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert omitted
    assert omitted[0]["severity"] == "warn"
    assert "goal_predicate_satisfiable" in critical[0]["detail"]
    assert "l2_plan_reaches_goal" in critical[0]["detail"]


def test_scenario_arc_wmte_4671_l2_goal_false_control_is_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4671-L2-GOAL-SATISFIABILITY: false control flags."""

    report = _report_for_payload(
        tmp_path,
        _l2_goal_induction_overclaim_fixture(
            goal_predicate_satisfiable={"lp85": False},
            l2_plan_reaches_goal={"lp85": True},
        ),
    )
    flags = _flag_kind(
        report, av.L2_GOAL_INDUCTION_WITHOUT_SATISFIABILITY_CHECK_KIND
    )

    assert flags
    assert flags[0]["severity"] == "critical"
    assert _flag_kind(report, av.L2_GOAL_SATISFIABILITY_CHECK_OMITTED_KIND) == []


def test_scenario_arc_wmte_4671_l2_goal_controls_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4671-L2-GOAL-SATISFIABILITY: true controls are clean."""

    report = _report_for_payload(
        tmp_path,
        _l2_goal_induction_overclaim_fixture(
            goal_predicate_satisfiable={"lp85": True},
            l2_plan_reaches_goal={"lp85": True},
        ),
    )

    assert _flag_kind(report, av.L2_GOAL_INDUCTION_WITHOUT_SATISFIABILITY_CHECK_KIND) == []
    assert _flag_kind(report, av.L2_GOAL_SATISFIABILITY_CHECK_OMITTED_KIND) == []


def test_scenario_arc_wmte_4671_multilevel_metric_omitted_harness_warn_and_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4671-MULTI-LEVEL-NONDEGENERATE-METRIC: omission flags."""

    report = _report_for_payload(tmp_path, _multi_level_metric_overclaim_fixture())
    critical = _flag_kind(report, av.MULTI_LEVEL_WITHOUT_NONDEGENERATE_METRIC_KIND)
    omitted = _flag_kind(report, av.MULTI_LEVEL_NONDEGENERATE_METRIC_OMITTED_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert omitted
    assert omitted[0]["severity"] == "warn"
    assert "target_levels>=2" in critical[0]["detail"]
    assert "break_at_first_win=false" in critical[0]["detail"]


def test_scenario_arc_wmte_4671_multilevel_metric_invalid_harness_is_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4671-MULTI-LEVEL-NONDEGENERATE-METRIC: degenerate harness flags."""

    report = _report_for_payload(
        tmp_path,
        _multi_level_metric_overclaim_fixture(
            metric_harness_fixed={"target_levels": 1, "break_at_first_win": True}
        ),
    )
    flags = _flag_kind(report, av.MULTI_LEVEL_WITHOUT_NONDEGENERATE_METRIC_KIND)

    assert flags
    assert flags[0]["severity"] == "critical"
    assert _flag_kind(report, av.MULTI_LEVEL_NONDEGENERATE_METRIC_OMITTED_KIND) == []


def test_scenario_arc_wmte_4671_multilevel_metric_fixed_harness_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4671-MULTI-LEVEL-NONDEGENERATE-METRIC: fixed harness is clean."""

    report = _report_for_payload(
        tmp_path,
        _multi_level_metric_overclaim_fixture(
            metric_harness_fixed={"target_levels": 2, "break_at_first_win": False}
        ),
    )

    assert _flag_kind(report, av.MULTI_LEVEL_WITHOUT_NONDEGENERATE_METRIC_KIND) == []
    assert _flag_kind(report, av.MULTI_LEVEL_NONDEGENERATE_METRIC_OMITTED_KIND) == []


def test_req_arc_wmte_4671_honest_a1_fixture_not_false_flagged() -> None:
    """REQ-ARC-WMTE-4671: honest .430 A1 artifact does not fire the new guards."""

    guarded_kinds = {
        av.L2_GOAL_INDUCTION_WITHOUT_SATISFIABILITY_CHECK_KIND,
        av.L2_GOAL_SATISFIABILITY_CHECK_OMITTED_KIND,
        av.MULTI_LEVEL_WITHOUT_NONDEGENERATE_METRIC_KIND,
        av.MULTI_LEVEL_NONDEGENERATE_METRIC_OMITTED_KIND,
    }

    report = av.verify_artifact(A1_PATH)

    assert _flag_kinds(report, guarded_kinds) == []


def test_req_arc_wmte_4671_helper_edges_keep_guards_narrow() -> None:
    """REQ-ARC-WMTE-4671: helper edges ignore non-win and zero-metric claims."""

    non_win_flags: list[av.Flag] = []
    av.check_l2_goal_induction_satisfiability_overclaim(
        {
            "game": "lp85",
            "headline": "L2 goal induction diagnostic",
            "honest_verdict": "complete: l2_goal_induction_no_deepening_residual",
            "generic_agent_reached_level": {"lp85": 1},
            "reproduced_levels": {"lp85": 1},
        },
        non_win_flags,
    )
    assert non_win_flags == []

    zero_metric_flags: list[av.Flag] = []
    av.check_multilevel_nondegenerate_metric_overclaim(
        {
            "game": "lp85",
            "headline": "multi-level solve-rate measured",
            "live_multi_level_solve_rate": 0.0,
        },
        zero_metric_flags,
    )
    assert zero_metric_flags == []


def test_req_arc_wmte_4671_runner_builds_required_terminal_artifact() -> None:
    """REQ-ARC-WMTE-4671: Exp 4671 emits the required evidence fields."""

    from carnot import experiment_4671_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_import_ok": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4671": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert artifact["honest_verdict"] == (
        "success: adversarial_verify_hardened_l2_goal_and_multilevel_metric_guards_tests_green."
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["l2_goal_satisfiability_guard_added"] is True
    assert artifact["multilevel_metric_guard_added"] is True
    assert artifact["honest_artifacts_not_flagged"] is True
    assert artifact["tests_added"]["passed"] is True
    assert artifact["random_seed"] == 4671
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_4671_runner_validation_rejects_malformed_artifact() -> None:
    """REQ-ARC-WMTE-4671: artifact validation fails closed."""

    from carnot import experiment_4671_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_import_ok": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4671": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["inference_substrate"] = "wrong"
    bad["l2_goal_satisfiability_guard_added"] = False
    bad["multilevel_metric_guard_added"] = False
    bad["honest_artifacts_not_flagged"] = False
    bad["tests_added"] = {"passed": False}
    bad["random_seed"] = 0
    bad["preconditions_checked"] = {"ok": False}
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "sha256:bad"

    errors = mod.validate_artifact(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "l2_goal_satisfiability_guard_added" in errors
    assert "multilevel_metric_guard_added" in errors
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
