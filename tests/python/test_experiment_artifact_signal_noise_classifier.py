"""Tests for the Exp 1454 experiment artifact signal/noise classifier.

Spec: REQ-REPORT-040, SCENARIO-REPORT-040.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from carnot.reporting.experiment_artifact_signal_noise_classifier import (
    REQUIRED_ARTIFACT_FIELDS,
    classify_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_req_report_040_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-040: the workflow seeds the deliverable before scanning."""

    out_path = tmp_path / "results" / "experiment_1454_experiment_artifact_signal_noise_classifier.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact == written
    assert artifact["status"] == "in_progress"
    assert artifact["artifacts_scanned"] == 0
    assert artifact["heuristic_version"] == "exp1454-v1"


def test_scenario_report_040_writes_conservative_signal_noise_ledger(tmp_path: Path) -> None:
    """SCENARIO-REPORT-040: scan all artifacts and keep environment blockers ambiguous."""

    out_path = tmp_path / "results" / "experiment_1454_experiment_artifact_signal_noise_classifier.json"
    table_path = tmp_path / "ops" / "experiment_signal_noise_classification.csv"
    summary_path = tmp_path / "ops" / "experiment_signal_noise_summary.md"
    write_in_progress_artifact(out_path)

    _write_json(
        tmp_path / "results" / "experiment_1001_headline_signal.json",
        {
            "experiment": "exp1001",
            "title": "Headline live verified improvement",
            "status": "complete",
            "honest_verdict": "complete_live_verified_headline_improvement",
            "headline_result_allowed": True,
            "accuracy_delta": 0.12,
            "gate_open": True,
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_1002_retired_noise.json",
        {
            "experiment": "exp1002",
            "title": "Retired no-improvement selector",
            "status": "complete",
            "honest_verdict": (
                "complete_prmv3_no_headline_improvement_prototype_candidate_pool_no_headline_claim"
            ),
            "headline_result_allowed": False,
            "retire_if_same_verdict": True,
            "improvement_pp": 0.0,
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_1003_environmental_blocker.json",
        {
            "experiment": "exp1003",
            "title": "Missing CUDA toolchain blocker",
            "status": "blocked",
            "honest_verdict": "blocked_missing_cuda_toolchain",
            "blocker": "missing CUDA compiler",
            "blocked_checks": ["cuda", "runtime", "cache", "model", "tool", "driver"],
            "gates_evaluated": ["cuda", "runtime"],
            "headline_result_allowed": False,
        },
    )
    (tmp_path / "results" / "experiment_1004_malformed.json").write_text(
        "{not json",
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_1005_non_object.json").write_text(
        "[1, 2, 3]",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "results" / "experiment_1006_no_improvement.json",
        {
            "experiment": "exp1006",
            "title": "No-improvement merit result",
            "status": "complete",
            "honest_verdict": "complete_no_improvement",
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_1007_regression.json",
        {
            "experiment": "exp1007",
            "title": "Negative regression result",
            "status": "complete",
            "honest_verdict": "negative_regression",
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_1008_failed_merit.json",
        {
            "experiment": "exp1008",
            "title": "Failed merit gate",
            "status": "failed",
            "honest_verdict": "failed_merit_gate",
        },
    )

    artifact = run(
        root=tmp_path,
        out_path=out_path,
        table_path=table_path,
        summary_path=summary_path,
    )
    rows = _read_rows(table_path)
    row_by_id = {row["experiment_id"]: row for row in rows}
    summary = summary_path.read_text(encoding="utf-8")

    assert artifact["status"] == "complete"
    assert artifact["artifacts_scanned"] == 9
    assert artifact["signal_count"] == 1
    assert artifact["noise_count"] == 4
    assert artifact["ambiguous_count"] == 4
    assert artifact["classification_table_path"] == "ops/experiment_signal_noise_classification.csv"
    assert artifact["summary_path"] == "ops/experiment_signal_noise_summary.md"
    assert [row["experiment_id"] for row in artifact["top_50_noise_candidates"]] == [
        "1002",
        "1006",
        "1007",
        "1008",
    ]
    assert row_by_id["1001"]["classification"] == "SIGNAL"
    assert "headline" in row_by_id["1001"]["reason"]
    assert row_by_id["1002"]["classification"] == "NOISE"
    assert "retirement" in row_by_id["1002"]["reason"]
    assert row_by_id["1003"]["classification"] == "AMBIGUOUS"
    assert "environmental" in row_by_id["1003"]["reason"]
    assert row_by_id["1004"]["classification"] == "AMBIGUOUS"
    assert "malformed" in row_by_id["1004"]["reason"]
    assert row_by_id["1005"]["classification"] == "AMBIGUOUS"
    assert "non_object_json" in row_by_id["1005"]["reason"]
    assert row_by_id["1006"]["classification"] == "NOISE"
    assert row_by_id["1007"]["classification"] == "NOISE"
    assert row_by_id["1008"]["classification"] == "NOISE"
    assert row_by_id["1454"]["classification"] == "AMBIGUOUS"
    assert "## Counts" in summary
    assert "Top 50 Noise Candidates" in summary


def test_req_report_040_covers_terminal_and_fallback_classifications() -> None:
    """REQ-REPORT-040: helper-level classification remains deterministic."""

    retired_blocker = classify_artifact(
        Path("results/experiment_2001_retired_blocker.json"),
        {
            "status": "blocked",
            "honest_verdict": "blocked_missing_tool_after_repeated_merit_gate",
            "blocker": "missing optional tool",
            "retirement_reason": "retired after repeated no_improvement merit gate",
        },
    )
    metadata_title = classify_artifact(
        Path("manual_artifact.json"),
            {
                "experiment": "exp2002",
                "status": "succeeded",
                "honest_verdict": "neutral",
                "artifact_metadata": {"title": "Nested metadata title"},
            },
        root=Path("elsewhere"),
    )
    verified = classify_artifact(
        Path("results/experiment_2003_verified.json"),
        {"status": "complete", "honest_verdict": "verified_positive"},
    )
    in_progress = classify_artifact(
        Path("results/experiment_2004_working.json"),
        {"status": "in_progress", "honest_verdict": "still_running"},
    )
    retired_status = classify_artifact(
        Path("results/experiment_2005_retired.json"),
        {"status": "retired", "honest_verdict": "governance_action"},
    )
    unknown = classify_artifact(
        Path("ops/experiment_2006_ops.json"),
        {"status": "unknown", "honest_verdict": None},
        root=Path("elsewhere"),
    )

    assert retired_blocker["classification"] == "NOISE"
    assert "retirement" in retired_blocker["reason"]
    assert metadata_title["experiment_id"] == "2002"
    assert metadata_title["path"] == "manual_artifact.json"
    assert metadata_title["title"] == "Nested metadata title"
    assert metadata_title["classification"] == "SIGNAL"
    assert "terminal success" in metadata_title["reason"]
    assert verified["classification"] == "SIGNAL"
    assert "verified" in verified["reason"]
    assert in_progress["classification"] == "AMBIGUOUS"
    assert "not terminal" in in_progress["reason"]
    assert retired_status["classification"] == "NOISE"
    assert unknown["path"] == "ops/experiment_2006_ops.json"
    assert unknown["honest_verdict"] == "null"
    assert unknown["classification"] == "AMBIGUOUS"
