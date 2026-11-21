"""Tests for Exp 4707 adversarial_verify .433 hardening.

Spec refs: REQ-ARC-WMTE-4707,
SCENARIO-ARC-WMTE-4707-FIRSTWIN-NULLDELTA,
SCENARIO-ARC-WMTE-4707-PERCEPTION-OVERCLAIM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
A4_PATH = REPO / "results" / "experiment_4691_held_out_first_win_readiness.json"
A1_PATH = REPO / "results" / "experiment_4700_object_centric_perception_proposal_live.json"


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


def _firstwin_null_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4707_firstwin_null_fixture",
        "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "first_win_baseline": 0.04,
        "first_win_rate_integrated": 0.04,
        "first_win_delta_vs_baseline": 0.0,
        "random_seed": 4707,
        "reproducibility_checksum": "sha256:" + "c" * 64,
    }
    payload.update(overrides)
    return payload


def _perception_overclaim_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4707_perception_overclaim_fixture",
        "game": "bp35",
        "headline": "object-centric relational representation lifted first-win and reached L2",
        "honest_verdict": "success: object_centric_perception_generic_agent_new_level_bp35_L2",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
        "generic_agent_reached_level": {"bp35": 2},
        "reproduced_levels": {"bp35": 1},
        "random_seed": 4707,
        "reproducibility_checksum": "sha256:" + "d" * 64,
    }
    payload.update(overrides)
    return payload


def _honest_a4_payload() -> dict[str, Any]:
    payload = json.loads(A4_PATH.read_text(encoding="utf-8"))
    payload.pop("flagged_adversarial", None)
    payload.pop("corrigendum_pending", None)
    payload.setdefault(
        "null_delta_methodology_note",
        "Held-out first-win is flat vs baseline with a passing parity positive control.",
    )
    payload.setdefault("positive_control_passed", True)
    payload.setdefault("first_win_delta_vs_baseline", 0.0)
    return payload


def test_req_arc_wmte_4707_spec_declares_guards() -> None:
    """REQ-ARC-WMTE-4707: OpenSpec declares both adversarial_verify guards."""

    from carnot import experiment_4707_adversarial_verify_hardening as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4707" in spec
    assert "SCENARIO-ARC-WMTE-4707-FIRSTWIN-NULLDELTA" in spec
    assert "SCENARIO-ARC-WMTE-4707-PERCEPTION-OVERCLAIM" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4707_firstwin_unvalidated_flat_null_stays_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4707-FIRSTWIN-NULLDELTA: missing markers stay critical."""

    report = _report_for_payload(tmp_path, _firstwin_null_fixture())
    flags = _flag_kind(report, "TAUTOLOGY")

    assert flags
    assert any(flag["severity"] == "critical" for flag in flags)
    assert any("first_win_baseline" in flag["detail"] for flag in flags)


def test_scenario_arc_wmte_4707_firstwin_validated_flat_null_is_warn(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4707-FIRSTWIN-NULLDELTA: validated flat null downgrades."""

    report = _report_for_payload(
        tmp_path,
        _firstwin_null_fixture(
            null_delta_methodology_note=(
                "Flat held-out first-win: the integrated lever did not move the "
                "leaderboard-relevant metric, and the parity positive control passed."
            ),
            positive_control_passed=True,
        ),
    )
    flags = [
        flag
        for flag in _flag_kind(report, "TAUTOLOGY")
        if "first_win_baseline" in flag["detail"]
        and "first_win_rate_integrated" in flag["detail"]
    ]

    assert flags
    assert {flag["severity"] for flag in flags} == {"warn"}
    assert "declared_null_delta" in flags[0]["detail"]


def test_scenario_arc_wmte_4707_firstwin_failed_positive_control_is_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4707-FIRSTWIN-NULLDELTA: failed control is not excused."""

    report = _report_for_payload(
        tmp_path,
        _firstwin_null_fixture(
            null_delta_methodology_note="Flat first-win null, but parity failed.",
            positive_control_passed=False,
        ),
    )
    flags = _flag_kind(report, "TAUTOLOGY")

    assert flags
    assert any(flag["severity"] == "critical" for flag in flags)


def test_scenario_arc_wmte_4707_perception_omitted_warn_and_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4707-PERCEPTION-OVERCLAIM: omitted evidence flags."""

    report = _report_for_payload(tmp_path, _perception_overclaim_fixture())
    critical = _flag_kind(report, av.PERCEPTION_OVERCLAIM_KIND)
    omitted = _flag_kind(report, av.PERCEPTION_OVERCLAIM_OMITTED_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert omitted
    assert omitted[0]["severity"] == "warn"
    assert "order1_ablation_reached_level" in omitted[0]["detail"]
    assert "offline_reproduced" in critical[0]["detail"]


def test_scenario_arc_wmte_4707_perception_order1_not_lower_is_critical(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4707-PERCEPTION-OVERCLAIM: order-1 must be lower."""

    report = _report_for_payload(
        tmp_path,
        _perception_overclaim_fixture(
            order1_ablation_reached_level={"bp35": 2},
            offline_reproduced={"bp35": True},
        ),
    )
    critical = _flag_kind(report, av.PERCEPTION_OVERCLAIM_KIND)

    assert critical
    assert critical[0]["severity"] == "critical"
    assert _flag_kind(report, av.PERCEPTION_OVERCLAIM_OMITTED_KIND) == []


def test_scenario_arc_wmte_4707_perception_evidence_not_false_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4707-PERCEPTION-OVERCLAIM: honest evidence is clean."""

    report = _report_for_payload(
        tmp_path,
        _perception_overclaim_fixture(
            order1_ablation_reached_level={"bp35": 1},
            offline_reproduced={"bp35": True},
        ),
    )

    assert _flag_kind(report, av.PERCEPTION_OVERCLAIM_KIND) == []
    assert _flag_kind(report, av.PERCEPTION_OVERCLAIM_OMITTED_KIND) == []


def test_req_arc_wmte_4707_honest_a1_a4_fixtures_not_false_flagged(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4707: honest A1/A4 artifacts do not fire new guarded criticals."""

    guarded_kinds = {
        av.PERCEPTION_OVERCLAIM_KIND,
        av.PERCEPTION_OVERCLAIM_OMITTED_KIND,
    }

    a4_report = _report_for_payload(tmp_path, _honest_a4_payload())
    a4_tautologies = [
        flag
        for flag in _flag_kind(a4_report, "TAUTOLOGY")
        if "first_win_baseline" in flag["detail"]
        and "first_win_rate_integrated" in flag["detail"]
    ]

    assert a4_tautologies
    assert {flag["severity"] for flag in a4_tautologies} == {"warn"}
    assert _flag_kinds(av.verify_artifact(A1_PATH), guarded_kinds) == []


def test_req_arc_wmte_4707_helper_edges_keep_guards_narrow() -> None:
    """REQ-ARC-WMTE-4707: helper edges ignore null and non-perception artifacts."""

    perception_null_flags: list[av.Flag] = []
    av.check_perception_overclaim(
        {
            "experiment": "experiment_4700_object_centric_perception_proposal_live",
            "game": "bp35",
            "honest_verdict": "complete: object_centric_perception_no_new_level_residual_search_is",
            "generic_agent_reached_level": 0,
            "reproduced_levels": 0,
            "order1_ablation_reached_level": 0,
            "offline_reproduced": False,
        },
        perception_null_flags,
    )
    assert perception_null_flags == []

    non_perception_flags: list[av.Flag] = []
    av.check_perception_overclaim(
        {
            "experiment": "experiment_4707_search_only_fixture",
            "honest_verdict": "success: generic_agent_new_level_via_search",
            "generic_agent_reached_level": 2,
            "reproduced_levels": 1,
        },
        non_perception_flags,
    )
    assert non_perception_flags == []


def test_req_arc_wmte_4707_runner_builds_required_terminal_artifact() -> None:
    """REQ-ARC-WMTE-4707: Exp 4707 emits the required evidence fields."""

    from carnot import experiment_4707_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_import_ok": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4707": True,
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
        "adversarial_verify_hardened_firstwin_nulldelta_and_perception_overclaim_guards_tests_green."
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["firstwin_nulldelta_carveout_added"] is True
    assert artifact["perception_overclaim_guard_added"] is True
    assert artifact["honest_artifacts_not_flagged"] is True


def test_req_arc_wmte_4707_artifact_validation_covers_schema_edges() -> None:
    """REQ-ARC-WMTE-4707: artifact validator catches malformed receipts."""

    from carnot import experiment_4707_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_import_ok": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4707": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )
    assert mod.validate_artifact(artifact) == []

    empty_errors = mod.validate_artifact({})
    assert "honest_verdict_terminal_prefix" in empty_errors
    assert "inference_substrate" in empty_errors
    assert "tests_added" in empty_errors
    assert "preconditions_checked" in empty_errors
    assert "field_principles" in empty_errors
    assert "reproducibility_checksum" in empty_errors

    malformed = dict(artifact)
    malformed["tests_added"] = {"passed": False}
    malformed["preconditions_checked"] = {"ok": False}
    malformed["field_principles"] = {"honest_verdict": mod.FIELD_PRINCIPLES["honest_verdict"]}
    malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
    malformed_errors = mod.validate_artifact(malformed)

    assert "tests_added.passed" in malformed_errors
    assert "preconditions_checked.ok" in malformed_errors
    assert "field_principles.inference_substrate" in malformed_errors
