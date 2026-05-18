"""Tests for the Exp 2352 NSVIF neuro-symbolic Z3 extractor.

Spec: REQ-VERIFY-1996, SCENARIO-VERIFY-1996
"""

from __future__ import annotations

import json
from pathlib import Path

import z3

from carnot.extraction import NsvifExtractor
from carnot.extraction.nsvif_extractor import (
    RANDOM_SEED,
    build_experiment_2352_corpus,
    evaluate_nsvif_corpus,
    run_experiment_2352,
)


def test_nsvif_extract_steps_keeps_assertion_like_cot_sentences() -> None:
    """REQ-VERIFY-1996: assertion steps are isolated from unsupported prose."""

    extractor = NsvifExtractor()

    steps = extractor.extract_steps(
        "We need the total. 8 plus 4 equals 12. Thus 12 - 2 = 10. Done."
    )

    assert steps == ["8 plus 4 equals 12", "Thus 12 - 2 = 10"]


def test_nsvif_encode_z3_handles_equalities_and_inequalities() -> None:
    """REQ-VERIFY-1996: arithmetic assertions compile to Z3 BoolRef formulas."""

    extractor = NsvifExtractor()

    formulas = extractor.encode_z3(["8 plus 4 equals 12", "20 >= 19", "2.5 + 1.5 = 4"])

    assert len(formulas) == 3
    assert all(isinstance(formula, z3.BoolRef) for formula in formulas)
    solver = z3.Solver()
    solver.add(*formulas)
    assert solver.check() == z3.sat


def test_nsvif_verify_flags_incorrect_arithmetic_step() -> None:
    """SCENARIO-VERIFY-1996: unsatisfiable arithmetic creates a violation."""

    extractor = NsvifExtractor()

    result = extractor.verify("First compute 12 + 7 = 20. Therefore 20 + 3 = 23.")

    assert result["satisfiable"] is False
    assert result["verification_pass"] is False
    assert result["violations"] == ["First compute 12 + 7 = 20"]
    assert result["n_constraints"] >= 1


def test_nsvif_verify_passes_correct_arithmetic_trace() -> None:
    """SCENARIO-VERIFY-1996: satisfiable arithmetic passes verification."""

    extractor = NsvifExtractor()

    result = extractor.verify("100 divided by 4 equals 25. Therefore 25 + 10 = 35.")

    assert result["satisfiable"] is True
    assert result["verification_pass"] is True
    assert result["violations"] == []
    assert result["n_constraints"] >= 2


def test_nsvif_corpus_has_required_size_and_metrics() -> None:
    """REQ-VERIFY-1996: Exp 2352 corpus validates both correct and incorrect paths."""

    corpus = build_experiment_2352_corpus()
    metrics = evaluate_nsvif_corpus(corpus)

    assert sum(1 for case in corpus if case["expected_correct"]) == 10
    assert sum(1 for case in corpus if not case["expected_correct"]) == 10
    assert metrics["verification_pass_rate"] >= 0.60
    assert metrics["extraction_coverage"] == 1.0


def test_nsvif_run_experiment_2352_writes_corpus_and_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1996: Exp 2352 writes the terminal JSON deliverable."""

    corpus_path = tmp_path / "experiment_2352_nsvif_corpus.json"
    artifact_path = tmp_path / "experiment_2352_nsvif_extractor.json"

    payload = run_experiment_2352(corpus_path=corpus_path, artifact_path=artifact_path)
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    assert persisted == payload
    assert len(corpus) == 20
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["nsvif_extractor_validated"] is True
    assert payload["verification_pass_rate"] >= 0.60
    assert payload["extraction_coverage"] == 1.0
    assert payload["n_correct_examples"] == 10
    assert payload["n_incorrect_examples"] == 10
    assert payload["random_seed"] == RANDOM_SEED == 42
    assert payload["z3_version"] == z3.get_version_string()
