"""Tests for Exp 2840 cross-corpus verifier matrix v3.

Spec: REQ-VERIFY-MATRIX-2840,
      SCENARIO-VERIFY-MATRIX-2840-REAL,
      SCENARIO-VERIFY-MATRIX-2840-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval.cross_corpus_verifier_matrix_v3 import (
    ARTIFACT_CANDIDATES,
    CORPORA,
    OUTPUT_FILENAME,
    build_matrix_artifact,
    classify_verifier,
    has_diversity_gap,
    normalize_auroc,
    run_analysis,
    select_artifact_for_corpus,
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
                "fr11_session_memory": [0.86, 0.88],
                "low": 0.55,
                "specific_fover": 0.79,
                "transfer_all": [0.82, 0.84],
            },
            {
                "low": 0.54,
                "specific_fover": 0.72,
                "transfer_all": [0.76, 0.78],
            },
        ),
        "MBPP": _artifact(
            {
                "low": 0.52,
                "memory_drop": 0.86,
                "specific_mbpp": [0.79, 0.79],
                "transfer_all": 0.81,
            },
            {
                "low": 0.53,
                "memory_drop": 0.62,
                "specific_mbpp": [0.72, 0.72],
                "transfer_all": 0.79,
            },
        ),
        "HumanEval": _artifact(
            {
                "low": 0.51,
                "transfer_all": 0.84,
            },
            {
                "low": 0.50,
                "transfer_all": 0.80,
            },
        ),
        "TruthfulQA": _artifact(
            {
                "low": 0.50,
                "transfer_all": 0.83,
            },
            {
                "low": 0.52,
                "transfer_all": 0.81,
            },
        ),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_req_verify_matrix_2840_normalizes_auroc_inputs() -> None:
    """REQ-VERIFY-MATRIX-2840: scalars and per-seed AUROC lists normalize to means."""

    assert normalize_auroc(0.75) == pytest.approx(0.75)
    assert normalize_auroc([0.70, 0.80, 0.90]) == pytest.approx(0.80)
    assert normalize_auroc([]) is None
    assert normalize_auroc(None) is None
    with pytest.raises(TypeError, match="bool"):
        normalize_auroc(True)
    with pytest.raises(TypeError, match="unsupported AUROC value"):
        normalize_auroc({"bad": "shape"})


def test_scenario_verify_matrix_2840_builds_real_matrix_and_categories() -> None:
    """SCENARIO-VERIFY-MATRIX-2840-REAL: measured rows classify without imputation."""

    artifact = build_matrix_artifact(_measured_inputs(), duration_s=12.5)
    matrix = artifact["verifier_corpus_dual_matrix"]

    assert artifact["honest_verdict"].startswith("complete:")
    assert tuple(CORPORA) == ("FoVer", "MBPP", "HumanEval", "TruthfulQA")
    assert matrix["fr11_session_memory"]["FoVer"] == {
        "production": pytest.approx(0.87),
        "architecture_only": None,
        "delta": None,
    }
    assert matrix["specific_mbpp"]["FoVer"] == {
        "production": None,
        "architecture_only": None,
        "delta": None,
    }
    assert matrix["specific_mbpp"]["MBPP"]["production"] == pytest.approx(0.79)
    assert matrix["transfer_all"]["TruthfulQA"]["architecture_only"] == pytest.approx(0.81)
    assert artifact["architecture_transfer_verifiers"] == ["transfer_all"]
    assert artifact["memory_augmented_verifiers"] == ["fr11_session_memory", "memory_drop"]
    assert artifact["corpus_specific_verifiers"] == ["specific_fover", "specific_mbpp"]
    assert artifact["low_signal_verifiers"] == ["low"]
    assert artifact["diversity_gap_on_non_fover"] is True
    assert artifact["duration_s"] == pytest.approx(12.5)

    bad_inputs = {"FoVer": {"per_verifier_condition_a_auroc": []}}
    with pytest.raises(TypeError, match="must be a mapping"):
        build_matrix_artifact(bad_inputs, duration_s=1.0)


def test_req_verify_matrix_2840_classification_edges() -> None:
    """REQ-VERIFY-MATRIX-2840: sparse memory rows and transfer coverage are distinct."""

    fover_only_high = {
        corpus: {"production": None, "architecture_only": None, "delta": None} for corpus in CORPORA
    }
    fover_only_high["FoVer"] = {
        "production": 0.90,
        "architecture_only": 0.90,
        "delta": 0.0,
    }
    memory_missing_baseline = {
        corpus: {"production": None, "architecture_only": None, "delta": None} for corpus in CORPORA
    }
    memory_missing_baseline["FoVer"] = {
        "production": 0.88,
        "architecture_only": None,
        "delta": None,
    }
    low_cells = {
        corpus: {"production": 0.64, "architecture_only": 0.50, "delta": 0.14} for corpus in CORPORA
    }
    transfer_cells = {
        corpus: {"production": 0.91, "architecture_only": 0.80, "delta": 0.11} for corpus in CORPORA
    }

    assert classify_verifier(fover_only_high) == "CORPUS_SPECIFIC"
    assert classify_verifier(memory_missing_baseline) == "MEMORY_AUGMENTED"
    assert classify_verifier(low_cells) == "LOW_SIGNAL"
    assert classify_verifier(transfer_cells) == "ARCHITECTURE_TRANSFER"
    assert classify_verifier({}) == "LOW_SIGNAL"

    three_transfer_matrix = {f"transfer_{idx}": transfer_cells for idx in range(3)}
    assert has_diversity_gap(three_transfer_matrix, list(three_transfer_matrix)) is False


def test_scenario_verify_matrix_2840_blocked_upstream_stays_empty() -> None:
    """SCENARIO-VERIFY-MATRIX-2840-BLOCKED: no fake verifier rows are inferred."""

    blocked = {corpus: _artifact({}, {}, verdict="blocked_model_not_cached") for corpus in CORPORA}

    artifact = build_matrix_artifact(blocked, duration_s=0.25)

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verifier_corpus_dual_matrix"] == {}
    assert artifact["architecture_transfer_verifiers"] == []
    assert artifact["memory_augmented_verifiers"] == []
    assert artifact["corpus_specific_verifiers"] == []
    assert artifact["low_signal_verifiers"] == []
    assert artifact["diversity_gap_on_non_fover"] is True
    assert "No synthetic verifier rows" in artifact["methodology_note"]


def test_scenario_verify_matrix_2840_run_analysis_selects_measured_candidate(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-MATRIX-2840-REAL: run_analysis writes the v3 artifact."""

    for corpus, relative_paths in ARTIFACT_CANDIDATES.items():
        primary = tmp_path / relative_paths[0]
        _write_json(primary, _artifact({}, {}, verdict="blocked_primary"))
        if corpus == "FoVer":
            fallback = tmp_path / relative_paths[1]
            _write_json(fallback, _measured_inputs()[corpus])
        else:
            _write_json(primary, _measured_inputs()[corpus])

    selected_path, selected_payload = select_artifact_for_corpus(
        tmp_path,
        "FoVer",
        ARTIFACT_CANDIDATES["FoVer"],
    )
    assert selected_path == ARTIFACT_CANDIDATES["FoVer"][1]
    assert selected_payload["honest_verdict"] == "complete: measured"
    missing_path, missing_payload = select_artifact_for_corpus(
        tmp_path,
        "Missing",
        ("results/does_not_exist.json",),
    )
    assert missing_path == ""
    assert missing_payload["honest_verdict"] == "blocked_missing_artifact_missing"

    clock_values = iter([100.0, 107.0])
    artifact = run_analysis(tmp_path, write=True, clock=lambda: next(clock_values))

    output_path = tmp_path / "results" / OUTPUT_FILENAME
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == artifact
    assert saved["duration_s"] == pytest.approx(7.0)
    assert saved["source_artifacts"]["FoVer"]["selected_path"] == ARTIFACT_CANDIDATES["FoVer"][1]
    assert saved["source_artifacts"]["MBPP"]["selected_path"] == ARTIFACT_CANDIDATES["MBPP"][0]
    assert saved["verifier_corpus_dual_matrix"]["transfer_all"]["HumanEval"][
        "architecture_only"
    ] == pytest.approx(0.80)
