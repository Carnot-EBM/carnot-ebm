"""Tests for Exp 4695 adversarial_verify .432 hardening.

Spec refs: REQ-ARC-WMTE-4695,
SCENARIO-ARC-WMTE-4695-NOVELTY-ABLATION,
SCENARIO-ARC-WMTE-4695-PROPOSAL-FILTER-HELDOUT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
A1_PATH = REPO / "results" / "experiment_4688_controllable_novelty_proposal_policy_live.json"
A2_PATH = (
    REPO
    / "results"
    / "experiment_4689_program_synthesis_action_effect_proposal_filter.json"
)


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


def _novelty_overclaim_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4695_controllable_novelty_overclaim_fixture",
        "game": "bp35",
        "headline": "generic agent reached L2 via controllable novelty",
        "honest_verdict": "success: controllable_novelty_generic_agent_new_level_bp35_L2",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "controllability_gate_on": True,
        "generic_agent_reached_level": {"bp35": 2},
        "reproduced_levels": {"bp35": 1},
        "random_seed": 4695,
        "reproducibility_checksum": "sha256:" + "a" * 64,
    }
    payload.update(overrides)
    return payload


def _proposal_filter_overclaim_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4695_program_synthesis_filter_overclaim_fixture",
        "game": "bp35",
        "headline": "program-synthesis proposal filter coverage up",
        "honest_verdict": "success: program_synthesis_filter_coverage_up_heldout_firstwin_lift_bp35",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "candidate_generation_coverage_filter": 0.60,
        "coverage_delta": 0.40,
        "random_seed": 4695,
        "reproducibility_checksum": "sha256:" + "b" * 64,
    }
    payload.update(overrides)
    return payload


def test_req_arc_wmte_4695_spec_declares_directed_exploration_guards() -> None:
    """REQ-ARC-WMTE-4695: OpenSpec declares both adversarial_verify guards."""

    from carnot import experiment_4695_adversarial_verify_hardening as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4695" in spec
    assert "SCENARIO-ARC-WMTE-4695-NOVELTY-ABLATION" in spec
    assert "SCENARIO-ARC-WMTE-4695-PROPOSAL-FILTER-HELDOUT" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4695_novelty_ablation_omitted_warn_and_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4695-NOVELTY-ABLATION: omitted evidence flags."""

    report = _report_for_payload(tmp_path, _novelty_overclaim_fixture())
    critical = _flag_kind(report, av.NOVELTY_PROPOSAL_WITHOUT_ABLATION_KIND)
    omitted = _flag_kind(report, av.NOVELTY_PROPOSAL_ABLATION_OMITTED_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert omitted
    assert omitted[0]["severity"] == "warn"
    assert "no_novelty_ablation_reached_level" in omitted[0]["detail"]
    assert "offline_reproduced" in critical[0]["detail"]


def test_scenario_arc_wmte_4695_novelty_ablation_not_lower_is_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4695-NOVELTY-ABLATION: ablations must be lower."""

    report = _report_for_payload(
        tmp_path,
        _novelty_overclaim_fixture(
            no_novelty_ablation_reached_level={"bp35": 2},
            cosmetic_novelty_ablation_reached_level={"bp35": 1},
            offline_reproduced={"bp35": True},
        ),
    )
    flags = _flag_kind(report, av.NOVELTY_PROPOSAL_WITHOUT_ABLATION_KIND)

    assert flags
    assert flags[0]["severity"] == "critical"
    assert "strictly lower" in flags[0]["detail"]
    assert _flag_kind(report, av.NOVELTY_PROPOSAL_ABLATION_OMITTED_KIND) == []


def test_scenario_arc_wmte_4695_novelty_ablation_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4695-NOVELTY-ABLATION: honest evidence is clean."""

    report = _report_for_payload(
        tmp_path,
        _novelty_overclaim_fixture(
            no_novelty_ablation_reached_level={"bp35": 1},
            cosmetic_novelty_ablation_reached_level={"bp35": 0},
            offline_reproduced={"bp35": True},
        ),
    )

    assert _flag_kind(report, av.NOVELTY_PROPOSAL_WITHOUT_ABLATION_KIND) == []
    assert _flag_kind(report, av.NOVELTY_PROPOSAL_ABLATION_OMITTED_KIND) == []


def test_scenario_arc_wmte_4695_proposal_filter_omitted_warn_and_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4695-PROPOSAL-FILTER-HELDOUT: omitted evidence flags."""

    report = _report_for_payload(tmp_path, _proposal_filter_overclaim_fixture())
    critical = _flag_kind(report, av.PROPOSAL_FILTER_WITHOUT_HELDOUT_REJECTION_KIND)
    omitted = _flag_kind(report, av.PROPOSAL_FILTER_HELDOUT_REJECTION_OMITTED_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert omitted
    assert omitted[0]["severity"] == "warn"
    assert "heldout_programs_rejected" in omitted[0]["detail"]
    assert "candidate_generation_coverage_blind_baseline" in critical[0]["detail"]


def test_scenario_arc_wmte_4695_proposal_filter_evidence_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4695-PROPOSAL-FILTER-HELDOUT: evidence is clean."""

    report = _report_for_payload(
        tmp_path,
        _proposal_filter_overclaim_fixture(
            heldout_programs_rejected=2,
            candidate_generation_coverage_blind_baseline=0.20,
        ),
    )

    assert _flag_kind(report, av.PROPOSAL_FILTER_WITHOUT_HELDOUT_REJECTION_KIND) == []
    assert _flag_kind(report, av.PROPOSAL_FILTER_HELDOUT_REJECTION_OMITTED_KIND) == []


def test_req_arc_wmte_4695_honest_a1_a2_fixtures_not_false_flagged() -> None:
    """REQ-ARC-WMTE-4695: honest .432 A1/A2 artifacts do not fire new guards."""

    guarded_kinds = {
        av.NOVELTY_PROPOSAL_WITHOUT_ABLATION_KIND,
        av.NOVELTY_PROPOSAL_ABLATION_OMITTED_KIND,
        av.PROPOSAL_FILTER_WITHOUT_HELDOUT_REJECTION_KIND,
        av.PROPOSAL_FILTER_HELDOUT_REJECTION_OMITTED_KIND,
    }

    assert _flag_kinds(av.verify_artifact(A1_PATH), guarded_kinds) == []
    assert _flag_kinds(av.verify_artifact(A2_PATH), guarded_kinds) == []


def test_req_arc_wmte_4695_helper_edges_keep_guards_narrow() -> None:
    """REQ-ARC-WMTE-4695: helper edges ignore nulls."""

    novelty_null_flags: list[av.Flag] = []
    av.check_novelty_proposal_ablation_overclaim(
        {
            "experiment": "experiment_4688_controllable_novelty_proposal_policy_live",
            "game": "bp35",
            "honest_verdict": "complete: controllable_novelty_no_new_level_residual",
            "generic_agent_reached_level": 0,
            "reproduced_levels": 0,
        },
        novelty_null_flags,
    )
    assert novelty_null_flags == []

    proposal_filter_null_flags: list[av.Flag] = []
    av.check_proposal_filter_heldout_rejection_overclaim(
        {
            "experiment": "experiment_4689_program_synthesis_action_effect_proposal_filter",
            "game": "bp35",
            "honest_verdict": "complete: program_synthesis_filter_no_coverage_gain_residual_logged",
            "candidate_generation_coverage_filter": 0.0,
            "coverage_delta": 0.0,
        },
        proposal_filter_null_flags,
    )
    assert proposal_filter_null_flags == []


def test_req_arc_wmte_4695_runner_builds_required_terminal_artifact() -> None:
    """REQ-ARC-WMTE-4695: Exp 4695 emits the required evidence fields."""

    from carnot import experiment_4695_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_import_ok": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4695": True,
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
        "adversarial_verify_hardened_novelty_ablation_and_proposal_filter_heldout_guards_tests_green."
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["novelty_ablation_guard_added"] is True
    assert artifact["proposal_filter_heldout_guard_added"] is True
    assert artifact["honest_artifacts_not_flagged"] is True
    assert artifact["tests_added"]["passed"] is True
    assert artifact["random_seed"] == 4695
    assert artifact["preconditions_checked"]["ok"] is True


def test_req_arc_wmte_4695_runner_validation_rejects_malformed_artifact() -> None:
    """REQ-ARC-WMTE-4695: artifact validation fails closed."""

    from carnot import experiment_4695_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_import_ok": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4695": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )

    artifact["reproducibility_checksum"] = "sha256:" + "0" * 64

    assert "reproducibility_checksum" in mod.validate_artifact(artifact)

    malformed = dict(artifact)
    malformed.pop("honest_verdict")
    malformed["inference_substrate"] = "wrong_substrate"
    malformed["novelty_ablation_guard_added"] = False
    malformed["random_seed"] = -1
    malformed["tests_added"] = "not_a_mapping"
    malformed["preconditions_checked"] = "not_a_mapping"
    malformed["field_principles"] = "not_a_mapping"

    errors = mod.validate_artifact(malformed)

    assert "missing required field honest_verdict" in errors
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "novelty_ablation_guard_added" in errors
    assert "random_seed" in errors
    assert "tests_added" in errors
    assert "preconditions_checked" in errors
    assert "field_principles" in errors

    failed_nested = dict(artifact)
    failed_nested["tests_added"] = {"passed": False}
    failed_nested["preconditions_checked"] = {"ok": False}
    failed_nested["field_principles"] = dict(mod.FIELD_PRINCIPLES)
    failed_nested["field_principles"].pop("honest_verdict")

    nested_errors = mod.validate_artifact(failed_nested)

    assert "tests_added.passed" in nested_errors
    assert "preconditions_checked.ok" in nested_errors
    assert "field_principles.honest_verdict" in nested_errors
