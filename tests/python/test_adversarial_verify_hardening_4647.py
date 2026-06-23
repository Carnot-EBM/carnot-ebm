"""Tests for Exp 4647 adversarial_verify goal-energy ablation hardening.

Spec refs: REQ-ARC-WMTE-4647,
SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-ABLATION,
SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-DIAGNOSTIC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
EXP4640 = REPO / "results" / "experiment_4640_goal_energy_generation_live.json"
GOAL_KIND = "goal-energy-without-ablation"


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report_for_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return av.verify_artifact(_write_payload(tmp_path, payload))


def _flag_kind(report: dict[str, Any], kind: str) -> list[dict[str, str]]:
    return [flag for flag in report["flags"] if flag["kind"] == kind]


def _goal_energy_win_artifact(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4647_goal_energy_win_without_ablation_fixture",
        "game": "ar25",
        "headline": "energy-driven generation win: graded goal-energy raised live first-win",
        "honest_verdict": "success: goal_energy_live_generation_firstwin_up_4",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "live_solve_rate_baseline": 0.04,
        "live_solve_rate_goal_energy": 0.12,
        "solve_rate_delta": 0.08,
        "first_win_rate_delta": 0.08,
        "energy_on_beats_baseline": True,
        "random_seed": 4647,
        "reproducibility_checksum": "sha256:" + "e" * 64,
    }
    payload.update(overrides)
    return payload


def _goal_energy_diagnostic_artifact(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4647_goal_energy_diagnostic_fixture",
        "game": "ar25",
        "headline": "diagnostic: graded goal-energy magnitude logged during replay",
        "honest_verdict": "complete: goal_energy_magnitude_diagnostic_only_no_win_claim",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "mean_goal_energy": 0.31,
        "random_seed": 4647,
        "reproducibility_checksum": "sha256:" + "f" * 64,
    }
    payload.update(overrides)
    return payload


def test_req_arc_wmte_4647_spec_declares_goal_energy_ablation_contract() -> None:
    """REQ-ARC-WMTE-4647: OpenSpec declares the goal-energy ablation guard."""

    from carnot import experiment_4647_adversarial_verify_hardening as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4647" in spec
    assert "SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-ABLATION" in spec
    assert "SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-DIAGNOSTIC" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4647_goal_energy_generation_win_needs_ablation(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-ABLATION: baseline-only win warns."""

    report = _report_for_payload(tmp_path, _goal_energy_win_artifact())
    flags = _flag_kind(report, GOAL_KIND)

    assert flags
    assert flags[0]["severity"] == "warn"
    assert "uniform-energy ablation" in flags[0]["detail"]
    assert "energy-on beat the baseline" in flags[0]["detail"]


def test_scenario_arc_wmte_4647_uniform_ablation_field_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-ABLATION: honest ablation field is clean."""

    report = _report_for_payload(
        tmp_path,
        _goal_energy_win_artifact(uniform_energy_ablation_passed=True),
    )

    assert _flag_kind(report, GOAL_KIND) == []


def test_scenario_arc_wmte_4647_uniform_ablation_arm_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-ABLATION: ablation arm is evidence."""

    report = _report_for_payload(
        tmp_path,
        _goal_energy_win_artifact(
            uniform_measurement={"solve_rate": 0.04, "first_win_rate": 0.04},
        ),
    )

    assert _flag_kind(report, GOAL_KIND) == []


def test_scenario_arc_wmte_4647_named_ablation_arm_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-ABLATION: named arm evidence is clean."""

    report = _report_for_payload(
        tmp_path,
        _goal_energy_win_artifact(
            arms=[{"name": "uniform_energy", "solve_rate": 0.04, "first_win_rate": 0.04}],
        ),
    )

    assert _flag_kind(report, GOAL_KIND) == []


def test_scenario_arc_wmte_4647_paired_goal_energy_metric_win_warns(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-ABLATION: paired metric wins warn."""

    artifact = _goal_energy_win_artifact()
    artifact.pop("solve_rate_delta")
    artifact.pop("first_win_rate_delta")
    artifact.pop("energy_on_beats_baseline")
    report = _report_for_payload(tmp_path, artifact)
    flags = _flag_kind(report, GOAL_KIND)

    assert flags
    assert flags[0]["severity"] == "warn"


def test_scenario_arc_wmte_4647_energy_magnitude_diagnostic_not_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-DIAGNOSTIC: no win claim is clean."""

    report = _report_for_payload(tmp_path, _goal_energy_diagnostic_artifact())

    assert _flag_kind(report, GOAL_KIND) == []


def test_req_arc_wmte_4647_helper_edges_keep_guard_narrow() -> None:
    """REQ-ARC-WMTE-4647: helper edges ignore metadata and weak claim shapes."""

    assert not av._has_positive_goal_energy_baseline_win_evidence(
        {"field_principles": {"solve_rate_delta": {"principle": "metadata only"}}}
    )
    assert not av._claims_goal_energy_generation_win(
        {
            "game": "ar25",
            "headline": "live generation first-win up",
            "honest_verdict": "success: live_generation_first_win_up",
            "solve_rate_delta": 0.08,
        }
    )
    assert not av._claims_goal_energy_generation_win(
        {
            "game": "ar25",
            "headline": "goal-energy calibration up",
            "honest_verdict": "success: goal_energy_up",
            "solve_rate_delta": 0.08,
        }
    )
    assert not av._claims_goal_energy_generation_win(
        {
            "game": "ar25",
            "headline": "goal-energy live generation measurement",
            "honest_verdict": "complete: goal_energy_live_generation_observed",
            "solve_rate_delta": 0.08,
        }
    )
    assert not av._claims_goal_energy_generation_win(
        {
            "game": "ar25",
            "headline": "goal-energy live generation first-win up",
            "honest_verdict": "success: goal_energy_live_generation_firstwin_up",
        }
    )
    assert not av._has_uniform_energy_ablation_evidence(
        {
            "field_principles": {
                "uniform_energy_ablation_passed": {"principle": "metadata only"}
            }
        }
    )
    assert av._has_uniform_energy_ablation_evidence(
        {"ablation_report": {"uniform_energy": {"solve_rate": 0.04}}}
    )


def test_req_arc_wmte_4647_a1_fixture_with_ablation_not_false_flagged() -> None:
    """REQ-ARC-WMTE-4647: exp4640 carries ablation evidence and stays clean."""

    report = av.verify_artifact(EXP4640)

    assert _flag_kind(report, GOAL_KIND) == []


def test_req_arc_wmte_4647_runner_builds_required_terminal_artifact() -> None:
    """REQ-ARC-WMTE-4647: Exp 4647 emits the required evidence fields."""

    from carnot import experiment_4647_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4647": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert artifact["honest_verdict"] == (
        "success: adversarial_verify_hardened_goal_energy_ablation_guard_tests_green."
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["goal_energy_ablation_guard_added"] is True
    assert artifact["honest_ablation_not_flagged"] is True
    assert artifact["diagnostic_not_flagged"] is True
    assert artifact["tests_added"]["passed"] is True
    assert artifact["research_conductor_modified"] is False
    assert artifact["random_seed"] == 4647
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_4647_runner_validation_rejects_malformed_artifact() -> None:
    """REQ-ARC-WMTE-4647: artifact validation fails closed."""

    from carnot import experiment_4647_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4647": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["inference_substrate"] = "wrong"
    bad["goal_energy_ablation_guard_added"] = False
    bad["honest_ablation_not_flagged"] = False
    bad["diagnostic_not_flagged"] = False
    bad["tests_added"] = {"passed": False}
    bad["research_conductor_modified"] = True
    bad["random_seed"] = 0
    bad["preconditions_checked"] = {"ok": False}
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "sha256:bad"
    errors = mod.validate_artifact(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "goal_energy_ablation_guard_added" in errors
    assert "honest_ablation_not_flagged" in errors
    assert "diagnostic_not_flagged" in errors
    assert "tests_added.passed" in errors
    assert "research_conductor_modified" in errors
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
