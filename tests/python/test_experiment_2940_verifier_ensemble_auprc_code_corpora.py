"""Tests for Exp 2940 code-corpus AUPRC/base-rate audit.

Spec: REQ-REPORT-2940, SCENARIO-REPORT-2940.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any

import pytest

from carnot.reporting import verifier_ensemble_auprc_code_corpora_2940 as mod


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(
    *,
    passed: bool,
    extraction_success: bool = True,
    syntax_success: bool = True,
    runtime_success: bool = True,
    row_status: str = "candidate_failed",
) -> dict[str, Any]:
    return {
        "candidate_index": 0,
        "corpus": "MBPP",
        "error_type": None if passed else "AssertionError",
        "extraction_success": extraction_success,
        "passed": passed,
        "random_seed": 1,
        "raw_response_path": "results/raw/mock.txt",
        "row_status": "candidate_passed" if passed else row_status,
        "runtime_success": runtime_success,
        "stable_id": "mbpp-mock",
        "syntax_success": syntax_success,
        "timed_out": False,
    }


def test_req_report_2940_precision_recall_points_are_base_rate_aware() -> None:
    """REQ-REPORT-2940: AUPRC and thresholds use precision/recall, not AUROC."""

    rows = [
        _candidate(passed=True),
        _candidate(passed=True),
        _candidate(passed=True),
        _candidate(passed=False),
        _candidate(passed=False, runtime_success=False, row_status="candidate_failed"),
        _candidate(
            passed=False,
            syntax_success=False,
            runtime_success=False,
            row_status="candidate_syntax_failed",
        ),
        _candidate(
            passed=False,
            extraction_success=False,
            syntax_success=False,
            runtime_success=False,
            row_status="candidate_extraction_failed",
        ),
        _candidate(
            passed=False,
            extraction_success=False,
            syntax_success=False,
            runtime_success=False,
            row_status="candidate_extraction_failed",
        ),
    ]

    labels, scores, energies = mod.code_labels_scores_from_candidates(rows)
    summary = mod.summarize_precision_recall(labels, scores)

    assert labels == [1, 1, 1, 0, 0, 0, 0, 0]
    assert energies == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0]
    assert summary.auprc == pytest.approx(0.75)
    assert summary.max_f1.threshold == pytest.approx(1.0)
    assert summary.max_f1.ppv == pytest.approx(0.75)
    assert summary.max_f1.recall == pytest.approx(1.0)
    assert summary.ppv_50.threshold == pytest.approx(0.25)
    assert summary.ppv_50.ppv == pytest.approx(0.5)
    assert summary.recall_80.threshold == pytest.approx(1.0)
    assert summary.recall_80.recall == pytest.approx(1.0)

    with pytest.raises(ValueError, match="same-length"):
        mod.summarize_precision_recall([], [])
    with pytest.raises(ValueError, match="both classes"):
        mod.summarize_precision_recall([1, 1], [0.9, 0.8])
    with pytest.raises(ValueError, match="unsupported FoVer"):
        mod._correct_label("unknown")


def test_scenario_report_2940_writes_required_artifact_from_upstreams(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2940: artifact cites upstreams and gates paper-v6 recommendation."""

    exp2910_path = tmp_path / mod.EXP2910_REL_PATH
    exp2837_path = tmp_path / mod.EXP2837_REL_PATH
    _write_json(
        exp2910_path,
        {
            "artifact": "experiment_2910_sota_code_generation_corrigendum_v2",
            "codegen_corrigendum_ready": True,
            "candidate_results": [
                _candidate(passed=True),
                _candidate(passed=True),
                _candidate(passed=True),
                _candidate(passed=False),
                _candidate(passed=False, runtime_success=False),
                _candidate(passed=False, syntax_success=False, runtime_success=False),
                _candidate(
                    passed=False,
                    extraction_success=False,
                    syntax_success=False,
                    runtime_success=False,
                ),
                _candidate(
                    passed=False,
                    extraction_success=False,
                    syntax_success=False,
                    runtime_success=False,
                ),
            ],
            "k_candidates_per_task": 8,
            "per_task_results": [{"stable_id": "mbpp-mock"}],
        },
    )
    _write_json(
        exp2837_path,
        {
            "artifact": "experiment_2837_fover_memory_leakage_v3",
            "honest_verdict": "complete: mock fover scores present",
            "fover_candidate_scores": [
                "ignored-non-dict-row",
                {"approval_score": 0.95, "label": "correct"},
                {"score": 0.90, "label": "correct"},
                {"energy": 0.8, "label": "incorrect"},
                {"energy": 0.9, "label": "incorrect"},
            ],
        },
    )

    output_path = mod.OUTPUT_REL_PATH
    artifact = mod.write_artifact(
        tmp_path,
        output_path=output_path,
        started_s=10.0,
        now_s=12.5,
    )
    saved = json.loads((tmp_path / output_path).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["code_corpus_auprc"] == pytest.approx(0.75)
    assert artifact["code_corpus_baseline_random_auprc"] == {
        "value": 0.075,
        "principle": mod.RANDOM_BASELINE_PRINCIPLE,
    }
    assert artifact["fover_corpus_auprc"]["value"] == pytest.approx(1.0)
    assert artifact["paper_v6_recommendation"]["value"] == "retain"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["cited_upstream_artifacts"] == [
        {
            "experiment_id": "exp2910",
            "fields_imported": [
                "candidate_results",
                "codegen_corrigendum_ready",
                "k_candidates_per_task",
                "per_task_results",
            ],
            "sha256": _sha256(exp2910_path),
        },
        {
            "experiment_id": "exp2837",
            "fields_imported": ["fover_candidate_scores"],
            "sha256": _sha256(exp2837_path),
        },
    ]


def test_req_report_2940_blocks_when_required_upstream_is_missing(tmp_path: Path) -> None:
    """REQ-REPORT-2940: missing upstream artifacts produce an honest blocked record."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert artifact["honest_verdict"] == "blocked_required_upstream_artifact_missing"
    assert artifact["code_corpus_auprc"] is None
    assert artifact["fover_corpus_auprc"]["value"] is None
    assert artifact["paper_v6_recommendation"]["value"] == "retract"
    assert artifact["duration_s"] == pytest.approx(0.25)
