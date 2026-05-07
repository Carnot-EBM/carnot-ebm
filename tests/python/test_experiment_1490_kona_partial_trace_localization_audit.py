"""Tests for Exp 1490 Kona/EBT partial-trace localization audit.

Spec refs: REQ-KONA-036, SCENARIO-KONA-036.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase3.kona_partial_trace_localization_audit import (
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    load_telemetry_rows,
    run_localization_audit,
    write_experiment_artifact,
)


def _row(
    *,
    case_id: str,
    expected_answer: str,
    adversarial_wrong_answer: str,
    wrong_energy: float,
) -> dict[str, object]:
    top_logprobs = [
        {"\n\n": -0.01, "\n": -4.0},
        {"<think>": -0.10, "1": -5.0},
        {"\n\n": -0.20, "\n": -0.6},
        {"</think>": 0.0, "</": -8.0},
        {"\n\n": -0.01, " ": -7.0},
        {expected_answer[-1]: -0.001, adversarial_wrong_answer[-1]: -wrong_energy},
    ]
    return {
        "case_id": case_id,
        "correct": True,
        "format_valid": True,
        "expected_answer": expected_answer,
        "adversarial_wrong_answer": adversarial_wrong_answer,
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "model_name": "Qwen3.6-35B-A3B",
        "model_family": "qwen_moe",
        "token_texts": ["\n\n", "<think>", "\n\n", "</think>", "\n\n", expected_answer[-1]],
        "token_logprobs": [-0.01, -0.10, -0.20, 0.0, -0.01, -0.001],
        "token_logprobs_available": True,
        "top_logprobs": top_logprobs,
        "topk_alternatives_available": True,
        "superficial_baselines": {"response_length": 22, "token_count": 6},
    }


def _rows() -> list[dict[str, object]]:
    return [
        _row(
            case_id="clean_001", expected_answer="1", adversarial_wrong_answer="0", wrong_energy=6.0
        ),
        _row(
            case_id="clean_002", expected_answer="3", adversarial_wrong_answer="4", wrong_energy=5.5
        ),
    ]


def test_injected_failure_span_ranks_above_clean_spans() -> None:
    """SCENARIO-KONA-036: injected bad spans rank first by local energy."""
    summary = run_localization_audit(_rows(), max_traces=2)

    assert summary.localization_audit_complete is True
    assert summary.traces_evaluated == 2
    assert summary.injected_failures == 2
    assert summary.localization_top1_rate == 1.0
    assert summary.localization_top3_rate == 1.0
    assert all(result.localization_rank == 1 for result in summary.results)
    assert all(
        sum(span.injected_failure for span in result.spans) == 1 for result in summary.results
    )


def test_random_and_span_length_baselines_are_span_count_derived() -> None:
    """REQ-KONA-036: baselines are computed from candidate spans, not constants."""
    summary = run_localization_audit(_rows(), max_traces=2)

    assert summary.random_baseline_rate == pytest.approx(1.0 / 6.0)
    assert summary.random_top3_baseline_rate == pytest.approx(3.0 / 6.0)
    assert summary.length_baseline_top1_rate == 0.0
    assert summary.localization_top1_rate > summary.random_baseline_rate


def test_loader_selects_clean_exp1480_style_rows(tmp_path: Path) -> None:
    """REQ-KONA-036: loader keeps only bounded clean rows with local telemetry."""
    manifest_path = tmp_path / "telemetry.jsonl"
    usable = _rows()[0]
    unusable = {**_rows()[1], "correct": False}
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in (unusable, usable)) + "\n",
        encoding="utf-8",
    )

    rows = load_telemetry_rows(manifest_path, max_traces=1)

    assert [row["case_id"] for row in rows] == ["clean_001"]


def test_missing_topk_wrong_answer_uses_conservative_energy() -> None:
    """REQ-KONA-036: unknown injected alternatives still produce a bounded score."""
    row = _row(
        case_id="clean_003",
        expected_answer="7",
        adversarial_wrong_answer="8",
        wrong_energy=5.0,
    )
    row["top_logprobs"] = row["top_logprobs"][:-1] + [{"7": -0.001, "9": -1.0}]

    summary = run_localization_audit([row], max_traces=1)

    assert summary.results[0].spans[-1].injected_text == "8"
    assert summary.results[0].spans[-1].local_energy > 6.0
    assert summary.localization_top1_rate == 1.0


def test_empty_input_builds_blocked_boundary_artifact() -> None:
    """REQ-KONA-036: no usable traces blocks without quality or Kona claims."""
    artifact = build_artifact(rows=[], tests_run=("pytest focused",))

    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["localization_audit_complete"] is False
    assert artifact["decoded_quality_claim_allowed"] is False
    assert artifact["kona_dependency_used"] is False
    assert artifact["traces_evaluated"] == 0
    assert "blocked" in artifact["honest_verdict"]


def test_build_artifact_contains_required_complete_fields() -> None:
    """SCENARIO-KONA-036: complete artifact records metrics and boundary flags."""
    artifact = build_artifact(
        rows=_rows(),
        tests_run=(
            "pytest tests/python/test_experiment_1490_kona_partial_trace_localization_audit.py -q",
        ),
    )

    assert artifact["schema"] == "carnot.phase3.kona_partial_trace_localization_audit.v1"
    assert artifact["experiment"] == "1490_kona_ebt_partial_trace_localization_audit"
    assert artifact["run_date"] == "20260507"
    assert artifact["spec_refs"] == ["REQ-KONA-036", "SCENARIO-KONA-036"]
    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["localization_audit_complete"] is True
    assert artifact["localization_top1_rate"] == 1.0
    assert artifact["random_baseline_rate"] == pytest.approx(1.0 / 6.0)
    assert artifact["decoded_quality_claim_allowed"] is False
    assert artifact["kona_dependency_used"] is False
    assert (
        artifact["audit_note_path"]
        == "docs/research-notes/kona_ebt_partial_trace_localization_audit.md"
    )
    assert "bounded" in artifact["honest_verdict"]
    json.dumps(artifact)


def test_write_experiment_artifact_round_trips_json(tmp_path: Path) -> None:
    """REQ-KONA-036: writer persists the measured audit artifact."""
    result_path = tmp_path / "experiment_1490.json"

    artifact = write_experiment_artifact(result_path, rows=_rows())

    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert loaded["status"] == "complete"
