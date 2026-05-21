"""Tests for Exp 2832 cross-corpus verifier matrix.

Spec: REQ-VERIFY-2832, SCENARIO-VERIFY-2832,
SCENARIO-VERIFY-2832-BLOCKED-UPSTREAM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval.cross_corpus_verifier_matrix_v2 import (
    ARTIFACT_FILES,
    CORPORA,
    OUTPUT_FILENAME,
    build_matrix_artifact,
    classify_verifier,
    has_diversity_gap,
    normalize_auroc,
    run_analysis,
)


def _artifact(
    condition_a: dict[str, float | list[float]],
    condition_b: dict[str, float | list[float]],
    verdict: str = "complete: measured",
) -> dict[str, object]:
    return {
        "honest_verdict": verdict,
        "per_verifier_condition_a_auroc": condition_a,
        "per_verifier_condition_b_auroc": condition_b,
    }


def _measured_inputs() -> dict[str, dict[str, object]]:
    return {
        "FoVer": _artifact(
            {
                "transfer_all": [0.80, 0.84],
                "memory_one": 0.86,
                "low": 0.55,
            },
            {
                "transfer_all": [0.76, 0.78],
                "memory_one": 0.60,
                "low": 0.54,
            },
        ),
        "MBPP": _artifact(
            {
                "transfer_all": 0.81,
                "memory_one": 0.52,
                "specific_mbpp": [0.85, 0.83],
                "low": 0.55,
            },
            {
                "transfer_all": 0.79,
                "memory_one": 0.50,
                "specific_mbpp": [0.82, 0.80],
                "low": 0.55,
            },
        ),
        "HumanEval": _artifact(
            {
                "transfer_all": 0.84,
                "low": 0.50,
            },
            {
                "transfer_all": 0.80,
                "low": 0.52,
            },
        ),
        "TruthfulQA": _artifact(
            {
                "transfer_all": 0.83,
                "low": 0.51,
            },
            {
                "transfer_all": 0.81,
                "low": 0.53,
            },
        ),
    }


def test_req_verify_2832_normalizes_auroc_inputs() -> None:
    """REQ-VERIFY-2832: scalar and per-seed AUROC lists normalize to floats."""

    assert normalize_auroc(0.75) == pytest.approx(0.75)
    assert normalize_auroc([0.70, 0.80, 0.90]) == pytest.approx(0.80)
    assert normalize_auroc([]) is None
    assert normalize_auroc(None) is None
    with pytest.raises(TypeError, match="bool"):
        normalize_auroc(True)
    with pytest.raises(TypeError, match="unsupported AUROC value"):
        normalize_auroc({"bad": "shape"})


def test_scenario_verify_2832_builds_dual_matrix_and_categories() -> None:
    """SCENARIO-VERIFY-2832: measured verifiers get full 3-D matrix cells."""

    artifact = build_matrix_artifact(_measured_inputs(), duration_s=12.5)
    matrix = artifact["verifier_corpus_dual_matrix"]

    assert artifact["honest_verdict"].startswith("complete:")
    assert tuple(CORPORA) == ("FoVer", "MBPP", "HumanEval", "TruthfulQA")
    assert matrix["transfer_all"]["FoVer"] == {
        "production": pytest.approx(0.82),
        "architecture_only": pytest.approx(0.77),
        "delta": pytest.approx(0.05),
    }
    assert matrix["specific_mbpp"]["FoVer"] == {
        "production": None,
        "architecture_only": None,
        "delta": None,
    }
    assert matrix["specific_mbpp"]["MBPP"]["production"] == pytest.approx(0.84)
    assert matrix["specific_mbpp"]["MBPP"]["architecture_only"] == pytest.approx(0.81)
    assert artifact["architecture_transfer_verifiers"] == ["transfer_all"]
    assert artifact["memory_augmented_verifiers"] == ["memory_one"]
    assert artifact["corpus_specific_verifiers"] == ["specific_mbpp"]
    assert artifact["low_signal_verifiers"] == ["low"]
    assert artifact["diversity_gap_on_non_fover"] is True
    assert artifact["duration_s"] == pytest.approx(12.5)
    assert artifact["upstream_artifact_statuses"]["MBPP"]["n_condition_a_verifiers"] == 4

    bad_inputs = {"FoVer": {"per_verifier_condition_a_auroc": []}}
    with pytest.raises(TypeError, match="must be a mapping"):
        build_matrix_artifact(bad_inputs, duration_s=1.0)


def test_req_verify_2832_classifies_each_verifier_once() -> None:
    """REQ-VERIFY-2832: category precedence is deterministic and exhaustive."""

    transfer_cells = {
        corpus: {"production": 0.91, "architecture_only": 0.80, "delta": 0.11}
        for corpus in CORPORA
    }
    memory_cells = {
        corpus: {"production": 0.86, "architecture_only": 0.60, "delta": 0.26}
        for corpus in CORPORA
    }
    low_cells = {
        corpus: {"production": 0.64, "architecture_only": 0.50, "delta": 0.14}
        for corpus in CORPORA
    }
    specific_cells = {
        corpus: {"production": None, "architecture_only": None, "delta": None}
        for corpus in CORPORA
    }
    specific_cells["FoVer"] = {"production": 0.78, "architecture_only": 0.72, "delta": 0.06}

    assert classify_verifier(transfer_cells) == "ARCHITECTURE_TRANSFER"
    assert classify_verifier(memory_cells) == "MEMORY_AUGMENTED"
    assert classify_verifier(low_cells) == "LOW_SIGNAL"
    assert classify_verifier(specific_cells) == "CORPUS_SPECIFIC"
    assert classify_verifier({}) == "LOW_SIGNAL"

    three_transfer_matrix = {
        f"transfer_{idx}": transfer_cells for idx in range(3)
    }
    assert has_diversity_gap(three_transfer_matrix, list(three_transfer_matrix)) is False


def test_scenario_verify_2832_blocked_upstream_stays_empty() -> None:
    """SCENARIO-VERIFY-2832-BLOCKED-UPSTREAM: no fake verifier rows are inferred."""

    blocked = {
        corpus: _artifact({}, {}, verdict="blocked_cuda_unavailable")
        for corpus in CORPORA
    }

    artifact = build_matrix_artifact(blocked, duration_s=0.25)

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verifier_corpus_dual_matrix"] == {}
    assert artifact["architecture_transfer_verifiers"] == []
    assert artifact["memory_augmented_verifiers"] == []
    assert artifact["corpus_specific_verifiers"] == []
    assert artifact["low_signal_verifiers"] == []
    assert artifact["diversity_gap_on_non_fover"] is True
    assert "No synthetic verifier rows" in artifact["methodology_note"]


def test_scenario_verify_2832_run_analysis_writes_expected_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2832: run_analysis loads the four named artifact paths."""

    for corpus, relative_path in ARTIFACT_FILES.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_measured_inputs()[corpus]) + "\n",
            encoding="utf-8",
        )

    clock_values = iter([100.0, 107.0])
    artifact = run_analysis(tmp_path, write=True, clock=lambda: next(clock_values))

    output_path = tmp_path / "results" / OUTPUT_FILENAME
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == artifact
    assert saved["duration_s"] == pytest.approx(7.0)
    assert saved["verifier_corpus_dual_matrix"]["transfer_all"]["TruthfulQA"][
        "architecture_only"
    ] == pytest.approx(0.81)
    assert saved["upstream_artifact_statuses"]["FoVer"]["path"] == ARTIFACT_FILES["FoVer"]
