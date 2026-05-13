"""Tests for the Exp 1996 NSVIF/Z3 SMT CoT extractor.

Spec: REQ-VERIFY-1996, SCENARIO-VERIFY-1996
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.pipeline.nsvif_smt_extractor import (
    MODEL_SPECS,
    NsvifSmtExtractor,
    run_experiment_1996,
)


def test_it_prose_arithmetic_violation_is_z3_checked() -> None:
    """REQ-VERIFY-1996: IT-prose arithmetic contradictions are solver checked."""

    extractor = NsvifSmtExtractor()

    results = extractor.extract("the total is 47 plus 28, which gives 76")

    assert len(results) == 1
    result = results[0]
    assert result.constraint_type == "nsvif_smt_arithmetic"
    assert result.metadata["solver"] == "z3"
    assert result.metadata["verdict"] == "violation"
    assert result.metadata["satisfied"] is False
    assert result.metadata["correct_result"] == 75
    assert result.metadata["first_order_formula"] == "(= (+ 47 28) 76)"


def test_correct_it_prose_arithmetic_is_not_flagged() -> None:
    """SCENARIO-VERIFY-1996: Correct IT-prose arithmetic has zero false positives."""

    extractor = NsvifSmtExtractor()

    samples = [
        "47 plus 28 equals 75",
        "20% of 50 is 10",
        "subtracting 10 from 100 gives 90",
        "100 divided by 8 gives 12.5",
    ]

    for sample in samples:
        results = extractor.extract(sample)
        assert results
        assert all(result.metadata["satisfied"] is True for result in results)


def test_unsupported_text_fails_closed_without_constraints() -> None:
    """REQ-VERIFY-1996: Unsupported prose abstains instead of fabricating checks."""

    extractor = NsvifSmtExtractor()

    assert extractor.extract("Blue feels calmer than red in this design.") == []
    assert extractor.extract("47 plus something gives a useful number.") == []


def test_categorical_logic_entailment_and_contradiction() -> None:
    """REQ-VERIFY-1996: Bounded first-order categorical claims use Z3."""

    extractor = NsvifSmtExtractor()
    text = (
        "All cats are mammals. "
        "Felix is a cat. "
        "Therefore Felix is a mammal. "
        "Felix is not a mammal."
    )

    results = extractor.extract(text, domain="logic")

    assert [result.metadata["logic_kind"] for result in results] == [
        "universal",
        "membership",
        "conclusion",
        "negated_membership",
    ]
    assert [result.metadata["verdict"] for result in results] == [
        "asserted",
        "asserted",
        "entailed",
        "violation",
    ]
    assert results[-1].metadata["satisfied"] is False
    assert results[-1].metadata["solver_status"] == "unsat"


def test_non_entailed_conclusion_abstains_without_violation() -> None:
    """SCENARIO-VERIFY-1996: Missing entailment does not become a false positive."""

    extractor = NsvifSmtExtractor()
    text = "All cats are mammals. Felix is a cat. Therefore Felix is a reptile."

    results = extractor.extract(text, domain="logic")

    conclusion = results[-1]
    assert conclusion.metadata["logic_kind"] == "conclusion"
    assert conclusion.metadata["verdict"] == "abstain"
    assert "satisfied" not in conclusion.metadata
    assert not any(result.metadata.get("satisfied") is False for result in results)


def test_domain_filter_skips_unrelated_domains() -> None:
    """REQ-VERIFY-1996: The extractor obeys non-matching domain hints."""

    extractor = NsvifSmtExtractor()

    assert extractor.extract("47 plus 28 equals 75", domain="code") == []


def test_run_experiment_1996_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1996: The terminal artifact contains gate fields."""

    artifact_path = tmp_path / "experiment_1996_nsvif_smt_extractor.json"

    payload = run_experiment_1996(artifact_path)
    persisted = json.loads(artifact_path.read_text())

    assert persisted == payload
    assert payload["status"] == "complete"
    assert payload["success"] is True
    assert payload["experiment_id"] == 1996
    assert payload["model_specs"] == list(MODEL_SPECS)
    assert payload["false_positives"] == 0
    assert payload["false_accepts"] == 0
    assert payload["zero_false_positives_by_design"] is True
    assert payload["solver_checks"] == payload["constraints_extracted"]
    assert payload["honest_verdict"].startswith("complete:")
