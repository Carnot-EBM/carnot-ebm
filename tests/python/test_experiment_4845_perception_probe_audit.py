"""Tests for Exp 4845 hostile A1 perception-probe audit.

Spec refs: REQ-ARC-WMTE-4845,
SCENARIO-ARC-WMTE-4845-A1-HOSTILE-AUDIT,
SCENARIO-ARC-WMTE-4845-NON-TEST-CLASSIFICATION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4845_perception_probe_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _good_a1_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_4841_object_identity_perception_probe",
        "honest_verdict": "complete_object_identity_unrecoverable_from_rendered_grid_deeper_finding",
        "measured_on_real_frames": True,
        "per_game_correspondence": {
            "lp85": {
                "shape_motion_score": 0.892323,
                "color_centroid_baseline_score": 0.842473,
                "n_frames": 53,
                "recovered": False,
                "source_kind": "banked_replay",
            },
            "r11l": {
                "shape_motion_score": 0.596154,
                "color_centroid_baseline_score": 0.403846,
                "n_frames": 5,
                "recovered": True,
                "source_kind": "banked_replay",
            },
            "tu93": {
                "shape_motion_score": 0.924274,
                "color_centroid_baseline_score": 0.857884,
                "n_frames": 94,
                "recovered": False,
                "source_kind": "banked_replay",
            },
        },
        "positive_control_tu93_passed": True,
        "positive_control_tu93": {
            "passed": True,
            "player_track_id": 9,
            "goal_track_id": 39,
            "player_motion": 220.624731,
            "goal_persistence": 0.5,
        },
        "games_with_recovery": 1,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
    }


def _clean_auxiliary_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {"returncode": 0, "stdout": "LIVE re-check: clean", "stderr": ""},
        {"loaded": True, "flag_count": 0, "flags": []},
        {"passed": True, "returncode": 0, "stdout_tail": "OK", "stderr_tail": ""},
    )


def test_req_arc_wmte_4845_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4845: OpenSpec anchors the hostile audit artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4845",
        "SCENARIO-ARC-WMTE-4845-A1-HOSTILE-AUDIT",
        "SCENARIO-ARC-WMTE-4845-NON-TEST-CLASSIFICATION",
        mod.RESULT_RELATIVE_PATH,
        mod.AUDIT_REPORT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4845_good_artifact_is_genuinely_exercised() -> None:
    """SCENARIO-ARC-WMTE-4845-A1-HOSTILE-AUDIT: real nulls with exercised tracker pass."""

    summary, adversarial, lint = _clean_auxiliary_results()
    audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )

    assert audit["a1_genuinely_exercised"] is True
    assert audit["non_test_reasons"] == []
    assert audit["checks"]["measured_on_real_frames"]["passed"] is True
    assert audit["checks"]["tracker_changed_vs_baseline"]["passed"] is True
    assert audit["checks"]["positive_control_and_recovery_claim"]["passed"] is True
    assert audit["checks"]["live_path_and_provenance"]["passed"] is True
    assert audit["per_game_correspondence_deltas"]["r11l"] == pytest.approx(0.192308)
    assert audit["recovered_games_from_rows"] == 1
    assert audit["claimed_recovery_matches_rows"] is True


def test_scenario_arc_wmte_4845_non_test_variants_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-4845-NON-TEST-CLASSIFICATION: hostile checks reject no-op probes."""

    summary, adversarial, lint = _clean_auxiliary_results()

    synthetic = _good_a1_artifact()
    synthetic["measured_on_real_frames"] = False
    synthetic["per_game_correspondence"]["lp85"]["source_kind"] = "synthetic_alignment"

    no_op = _good_a1_artifact()
    for row in no_op["per_game_correspondence"].values():
        row["shape_motion_score"] = row["color_centroid_baseline_score"]
        row["recovered"] = False
    no_op["games_with_recovery"] = 0

    celebratory = _good_a1_artifact()
    celebratory["honest_verdict"] = "success_object_identity_perception_recovers_goal_grounding"

    dishonest = _good_a1_artifact()
    dishonest["solve_provenance"] = "live_agent_self_discovery"
    dishonest["live_path_reachable"] = False
    bad_lint = {"passed": False, "returncode": 1, "stdout_tail": "", "stderr_tail": "boom"}

    cases = [
        (synthetic, lint, "synthetic_or_missing_real_frames"),
        (no_op, lint, "tracker_degenerate_to_baseline"),
        (celebratory, lint, "success_claim_without_two_recovered_games"),
        (dishonest, bad_lint, "live_path_unreachable"),
        (dishonest, lint, "solve_provenance_not_development_proxy"),
    ]

    for artifact, lint_result, reason in cases:
        audit = mod.audit_a1_artifact(
            artifact,
            summarizer_result=summary,
            adversarial_result=adversarial,
            live_lint_result=lint_result,
        )
        assert audit["a1_genuinely_exercised"] is False
        assert reason in audit["non_test_reasons"]
        assert audit["honest_verdict"].startswith("complete_a1_perception_probe_non_test_")


def test_req_arc_wmte_4845_build_schema_and_report_write(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4845: artifact and markdown writes are checksum-stable."""

    source = tmp_path / mod.SOURCE_ARTIFACT_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(_good_a1_artifact()), encoding="utf-8")
    summary, adversarial, lint = _clean_auxiliary_results()
    audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )

    artifact = mod.build_artifact(
        source_path=source,
        source_artifact=_good_a1_artifact(),
        audit=audit,
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
        preconditions_checked={"source_artifact_present": True, "spec_has_req_4845": True},
        duration_s=0.0,
    )
    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["duration_s"] == mod.DURATION_FLOOR_S
    assert artifact["source_artifact_checksum"] == mod.file_checksum(source)

    result_path = mod.write_artifact(artifact, root=tmp_path)
    report_path = mod.append_markdown_report(artifact, root=tmp_path)
    mod.append_markdown_report(artifact, root=tmp_path)

    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")
    assert loaded == artifact
    assert report_text.count("## Experiment 4845 .446 A1 Perception Probe Audit") == 1
    assert "a1_genuinely_exercised" in report_text

    broken = dict(artifact)
    broken["a1_genuinely_exercised"] = "yes"
    broken["reproducibility_checksum"] = "sha256:bad"
    errors = mod.artifact_schema_errors(broken)
    assert "a1_genuinely_exercised_must_be_bool" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_req_arc_wmte_4845_run_checked_in_artifact() -> None:
    """SCENARIO-ARC-WMTE-4845-A1-HOSTILE-AUDIT: checked-in Exp 4841 passes the audit."""

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_a1_perception_probe_audit_genuinely_exercised"
    assert artifact["a1_genuinely_exercised"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["checks"]["measured_on_real_frames"]["passed"] is True
    assert artifact["checks"]["tracker_changed_vs_baseline"]["passed"] is True
    assert artifact["checks"]["positive_control_and_recovery_claim"]["passed"] is True
    assert artifact["checks"]["live_path_and_provenance"]["passed"] is True
    assert artifact["summarizer_result"]["returncode"] == 0
    assert artifact["adversarial_result"]["flag_count"] == 0
    assert artifact["live_lint_result"]["passed"] is True


def test_scenario_arc_wmte_4845_blocked_preconditions_do_not_fabricate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4845: missing source artifacts produce blocked audit output."""

    artifact = mod.run(root=tmp_path, write=True)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "blocked_missing_exp4841_perception_artifact"
    assert artifact["a1_genuinely_exercised"] is False
    assert artifact["checks"] == {}
    assert "source_artifact_present" in artifact["preconditions_checked"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / mod.AUDIT_REPORT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4845_defensive_branch_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4845: malformed inputs fail closed without fabricated trust."""

    assert mod._finite_float(True) is None
    assert mod._finite_float("0.5") is None
    assert mod._finite_float(float("nan")) is None
    assert mod._safe_suffix([]) == "genuinely_exercised"

    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(not_object)

    summary, adversarial, lint = _clean_auxiliary_results()
    missing_numeric = _good_a1_artifact()
    missing_numeric["per_game_correspondence"]["lp85"]["shape_motion_score"] = None
    missing_numeric["per_game_correspondence"]["r11l"]["color_centroid_baseline_score"] = "bad"
    numeric_audit = mod.audit_a1_artifact(
        missing_numeric,
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    assert "tracker_degenerate_to_baseline" in numeric_audit["non_test_reasons"]
    assert numeric_audit["checks"]["tracker_changed_vs_baseline"]["missing_numeric_games"] == [
        "lp85",
        "r11l",
    ]

    bad_positive = _good_a1_artifact()
    bad_positive["positive_control_tu93_passed"] = False
    bad_positive["positive_control_tu93"] = {"passed": False}
    assert (
        "tu93_positive_control_failed"
        in mod.audit_a1_artifact(
            bad_positive,
            summarizer_result=summary,
            adversarial_result=adversarial,
            live_lint_result=lint,
        )["non_test_reasons"]
    )

    mismatch = _good_a1_artifact()
    mismatch["games_with_recovery"] = 2
    assert (
        "games_with_recovery_mismatch"
        in mod.audit_a1_artifact(
            mismatch,
            summarizer_result=summary,
            adversarial_result=adversarial,
            live_lint_result=lint,
        )["non_test_reasons"]
    )

    underclaim = _good_a1_artifact()
    underclaim["per_game_correspondence"]["tu93"]["recovered"] = True
    underclaim["games_with_recovery"] = 2
    assert (
        "verdict_does_not_match_recovery_numbers"
        in mod.audit_a1_artifact(
            underclaim,
            summarizer_result=summary,
            adversarial_result=adversarial,
            live_lint_result=lint,
        )["non_test_reasons"]
    )

    tool_failed = mod.audit_a1_artifact(
        _good_a1_artifact(),
        summarizer_result={"returncode": 2},
        adversarial_result={"loaded": True, "flag_count": 1},
        live_lint_result=lint,
    )
    assert "summarizer_reported_live_flags" in tool_failed["non_test_reasons"]
    assert "adversarial_verify_flagged" in tool_failed["non_test_reasons"]

    source = tmp_path / mod.SOURCE_ARTIFACT_RELATIVE_PATH
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(_good_a1_artifact()), encoding="utf-8")
    audit = mod.audit_a1_artifact(
        _good_a1_artifact(),
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
    )
    artifact = mod.build_artifact(
        source_path=source,
        source_artifact=_good_a1_artifact(),
        audit=audit,
        summarizer_result=summary,
        adversarial_result=adversarial,
        live_lint_result=lint,
        preconditions_checked={"ok": True},
        duration_s=1.0,
    )
    broken = dict(artifact)
    broken.update(
        {
            "honest_verdict": "bad",
            "field_principles": {},
            "inference_substrate": "live_llm_inference",
            "checks": [],
            "non_test_reasons": "none",
            "random_seed": 1,
            "duration_s": 0.0,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    errors = mod.artifact_schema_errors(broken)
    for expected in (
        "honest_verdict_missing_terminal_prefix",
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "checks_must_be_dict",
        "non_test_reasons_must_be_list",
        "random_seed_mismatch",
        "duration_below_aggregation_floor",
        "reproducibility_checksum_mismatch",
    ):
        assert expected in errors
    with pytest.raises(ValueError, match="honest_verdict_missing_terminal_prefix"):
        mod.write_artifact(broken, root=tmp_path)

    report_artifact = dict(artifact)
    report_artifact["checks"] = dict(artifact["checks"], malformed=[])
    assert "malformed" not in mod.render_markdown_section(report_artifact)

    monkeypatch.setattr(mod, "check_preconditions", lambda _root: {"ok": True})
    monkeypatch.setattr(mod, "run_summarizer", lambda _path: summary)
    monkeypatch.setattr(mod, "run_adversarial_verify", lambda _path: adversarial)
    monkeypatch.setattr(mod, "run_arc_orphan_solver_lint", lambda _root: lint)
    written = mod.run(root=tmp_path, write=True, now=iter([1.0, 1.25]).__next__)
    assert written["duration_s"] == 0.25
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["boom"])
    with pytest.raises(ValueError, match="boom"):
        mod.run(root=tmp_path, write=False, now=iter([2.0, 2.25]).__next__)
