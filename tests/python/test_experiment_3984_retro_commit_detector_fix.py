"""Tests for Exp 3984 operational retro commit-detector repair.

Spec refs: REQ-REPORT-3984, SCENARIO-REPORT-3984-REPRO,
SCENARIO-REPORT-3984-BACKFILL.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import retro_commit_detector_3984 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def test_req_report_3984_spec_anchor_exists() -> None:
    """REQ-REPORT-3984: OpenSpec declares the detector repair contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3984" in spec
    assert "SCENARIO-REPORT-3984-REPRO" in spec
    assert "SCENARIO-REPORT-3984-BACKFILL" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "detector_gap_suspected" in spec


def test_scenario_report_3984_reproduces_legacy_false_zero() -> None:
    """SCENARIO-REPORT-3984-REPRO: .367-style subjects are not literal `Exp `."""

    git_log = "\n".join(
        [
            "COMMIT\t246aa551\t2026-06-09T12:46:15+00:00\t[conductor] r11l INCREMENTAL solve (+1..+2 levels, per the mandatory rule)",
            "A\tresults/experiment_3964_r11l_incremental_l2.json",
            "",
            "COMMIT\t7631dc2\t2026-06-09T12:59:45+00:00\t[conductor] lp85 INCREMENTAL solve (+1 level)",
            "A\tresults/experiment_3965_lp85_incremental_l2.json",
            "",
            "COMMIT\tb90d24a\t2026-06-09T13:33:38+00:00\t[conductor] THIRD ARC-AGI-3 game first-solve",
            "A\tresults/experiment_3966_third_game_first_solve.json",
        ]
    )

    detection = mod.detect_from_git_log_text("2026.06.367", git_log)

    assert detection.legacy_experiment_commit_count == 0
    assert detection.direct_artifact_count == 3
    assert detection.corrected_experiment_count == 3
    assert detection.detector_gap_suspected is True
    assert detection.detector_gap_artifact_count == 3
    assert detection.artifact_paths == (
        "results/experiment_3964_r11l_incremental_l2.json",
        "results/experiment_3965_lp85_incremental_l2.json",
        "results/experiment_3966_third_game_first_solve.json",
    )


def test_req_report_3984_artifact_filter_counts_terminal_numeric_results_only() -> None:
    """REQ-REPORT-3984: auxiliary state and nonnumeric artifacts are not experiments."""

    git_log = "\n".join(
        [
            "COMMIT\tabc\t2026-06-07T20:13:55+00:00\t[conductor] FR-11 continuous self-learning v26",
            "A\tresults/experiment_3930_fr11_v26_cascade_band_online_learning.json",
            "A\tresults/experiment_3930_fr11_v26_cascade_band_online_learning_state.json",
            "A\tresults/experiment_arc_invariant_tiling_scaling.json",
            "A\tresults/operational_retro_2026_06_363.json",
        ]
    )

    detection = mod.detect_from_git_log_text("2026.06.363", git_log)

    assert detection.direct_artifact_count == 1
    assert detection.corrected_experiment_count == 1
    assert detection.artifact_paths == (
        "results/experiment_3930_fr11_v26_cascade_band_online_learning.json",
    )


def test_scenario_report_3984_backfill_uses_next_activation_bounds() -> None:
    """SCENARIO-REPORT-3984-BACKFILL: closed milestones end at next activation."""

    calls: list[list[str]] = []
    responses = {
        ("git", "log", "--format=%H%x09%aI%x09%s", "--grep=\\[conductor\\] Activate milestone 2026.06.367", "-n", "1"): (
            "start367\t2026-06-09T12:16:37+00:00\t[conductor] Activate milestone 2026.06.367\n"
        ),
        ("git", "log", "--format=%H%x09%aI%x09%s", "--grep=\\[conductor\\] Activate milestone 2026.06.368", "-n", "1"): (
            "start368\t2026-06-09T23:42:42+00:00\t[conductor] Activate milestone 2026.06.368\n"
        ),
        ("git", "log", "--format=COMMIT%x09%H%x09%aI%x09%s", "--name-status", "--diff-filter=A", "start367..start368", "--", "results/experiment_*.json"): (
            "COMMIT\tone\t2026-06-09T12:46:15+00:00\t[conductor] r11l solve\n"
            "A\tresults/experiment_3964_r11l_incremental_l2.json\n"
        ),
    }

    def runner(args: list[str], cwd: Path) -> str:
        calls.append(args)
        return responses[tuple(args)]

    backfill = mod.backfill_corrected_counts(
        Path("/repo"),
        ("2026.06.367",),
        run_git=runner,
    )

    assert backfill == {"2026.06.367": 1}
    assert [
        "git",
        "log",
        "--format=COMMIT%x09%H%x09%aI%x09%s",
        "--name-status",
        "--diff-filter=A",
        "start367..start368",
        "--",
        "results/experiment_*.json",
    ] in calls


def test_req_report_3984_payload_validation_requires_bare_fields() -> None:
    """REQ-REPORT-3984: terminal artifact fields are present and scalar."""

    payload = {
        "schema": mod.SCHEMA,
        "experiment": "experiment_3984_retro_commit_detector_fix",
        "detector_bug_reproduced": True,
        "root_cause": mod.ROOT_CAUSE,
        "detector_fixed": True,
        "self_check_added": True,
        "backfill_corrected_counts": "2026.06.367=11",
        "honest_verdict": "complete: detector fixed",
        "duration_s": 0.1,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "estimated_time_savings": 0,
    }

    mod.validate_payload(payload)

    bad = dict(payload)
    bad["detector_fixed"] = {"nested": True}
    try:
        mod.validate_payload(bad)
    except ValueError as exc:
        assert "detector_fixed" in str(exc)
    else:  # pragma: no cover - defensive assertion shape
        raise AssertionError("nested detector_fixed field was accepted")

    encoded = json.dumps(payload)
    assert "complete:" in encoded
