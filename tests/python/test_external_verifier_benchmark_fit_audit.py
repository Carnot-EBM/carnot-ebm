"""Tests for Exp 1465 external verifier benchmark fit audit.

Spec: REQ-REPORT-046, SCENARIO-REPORT-046.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.external_verifier_benchmark_fit_audit import (
    REQUIRED_ARTIFACT_FIELDS,
    REQUIRED_BENCHMARKS_REVIEWED,
    build_artifact,
    default_decisions,
    render_decision_note,
    run,
    write_in_progress_artifact,
)


def test_req_report_046_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-046: seed the Exp 1465 deliverable before audit work."""

    out_path = tmp_path / "results" / "experiment_1465_external_verifier_benchmark_fit_audit.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact == written
    assert artifact["status"] == "in_progress"
    assert artifact["benchmarks_reviewed"] == []
    assert artifact["benchmark_adoption_decision"] == "pending"
    assert artifact["adopted_benchmark"] is None
    assert artifact["deferred_benchmarks"] == []
    assert artifact["retired_benchmarks"] == []
    assert artifact["honest_verdict"] == "in_progress"


def test_scenario_report_046_selects_beaver_as_single_adoption() -> None:
    """SCENARIO-REPORT-046: exactly one benchmark family is adopted."""

    decisions = default_decisions()
    artifact = build_artifact(
        decisions=decisions,
        decision_table_path="docs/research-notes/external_verifier_benchmark_fit.md",
    )
    note = render_decision_note(decisions, artifact["next_minimal_benchmark_task"])

    assert [decision.benchmark for decision in decisions] == REQUIRED_BENCHMARKS_REVIEWED
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["benchmarks_reviewed"] == REQUIRED_BENCHMARKS_REVIEWED
    assert artifact["benchmark_adoption_decision"] == (
        "adopt_beaver_style_deterministic_bounds_smoke; "
        "defer_vnnlib_vnncomp_and_smaller_existing_benchmark"
    )
    assert artifact["adopted_benchmark"] == (
        "BEAVER-style deterministic bounds via existing BEAVER-lite smoke"
    )
    assert artifact["deferred_benchmarks"] == [
        "VNNLIB/VNN-COMP",
        "smaller existing benchmark",
    ]
    assert artifact["retired_benchmarks"] == []
    assert artifact["honest_verdict"] == "adopt_one_minimal_beaver_bounds_smoke"
    assert "VNNLIB/VNN-COMP | defer" in note
    assert "BEAVER-style deterministic bounds | adopt" in note
    assert "smaller existing benchmark | defer" in note


def test_req_report_046_next_minimal_task_is_exactly_scoped() -> None:
    """REQ-REPORT-046: the adopted benchmark defines one minimal future task."""

    artifact = build_artifact(
        decisions=default_decisions(),
        decision_table_path="docs/research-notes/external_verifier_benchmark_fit.md",
    )

    task = artifact["next_minimal_benchmark_task"]

    assert task["task_id"] == "exp_next_beaver_lite_external_bounds_smoke"
    assert task["benchmark_family"] == "BEAVER-style deterministic bounds"
    assert task["inputs"] == [
        "python/carnot/verify/beaver_lite.py",
        "tests/python/test_beaver_lite.py",
        "tests/python/test_beaver_lite_live_logprobs.py",
        "results/experiment_1142_beaver_lite_certificate_tier.json",
        "results/experiment_1158_beaver_lite_live_logprobs.json",
    ]
    assert task["expected_artifact_fields"] == [
        "status",
        "benchmark_family",
        "questions_evaluated",
        "prefix_closed_constraint",
        "unsafe_mass_bound",
        "empirical_violation_rate",
        "bound_is_sound",
        "mock_or_live_logprobs",
        "external_fit_verdict",
        "honest_verdict",
    ]
    assert task["e2e_check"] == (
        "run the existing BEAVER-lite bounder over three deterministic "
        "GSM8K-style arithmetic prompts and assert every reported unsafe "
        "mass bound is in [0, 1], sound, and labeled mock_or_live_logprobs"
    )


def test_req_report_046_run_writes_note_and_terminal_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-046: run writes the markdown note and terminal JSON."""

    out_path = tmp_path / "results" / "experiment_1465_external_verifier_benchmark_fit_audit.json"
    note_path = tmp_path / "docs" / "research-notes" / "external_verifier_benchmark_fit.md"

    artifact = run(root=tmp_path, out_path=out_path, decision_note_path=note_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    note = note_path.read_text(encoding="utf-8")

    assert artifact == written
    assert written["status"] == "complete"
    assert written["benchmark_decision_table_path"] == (
        "docs/research-notes/external_verifier_benchmark_fit.md"
    )
    assert written["adopted_benchmark"] == (
        "BEAVER-style deterministic bounds via existing BEAVER-lite smoke"
    )
    assert "## Next Minimal Benchmark Task" in note
    assert "exp_next_beaver_lite_external_bounds_smoke" in note
