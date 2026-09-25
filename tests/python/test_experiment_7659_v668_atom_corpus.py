"""Focused custody and scope checks for REQ-REPORT-7659."""

from __future__ import annotations

from copy import deepcopy

import pytest

from carnot.reporting.experiment_7659_atom_corpus import (
    build_role_rows,
    extract_group,
    validate_input,
)
from carnot.verify.tool_source_atoms import digest


def source_row(group: str, line: int = 1) -> dict:
    source = f"Tool output:\n```\n1: raise ValidationError()\n2: pass\n```"
    answer = f"`ValidationError` is raised at line {line}. The cause is complex."
    sentence = {
        "text": answer,
        "byte_start": 0,
        "byte_end": len(answer.encode()),
        "text_sha256": digest(answer),
    }
    return {
        "component_hash": group,
        "role": "fit",
        "learning_partition": "fit_optimization",
        "complete_source": source,
        "complete_answer": answer,
        "source_sha256": digest(source),
        "answer_sha256": digest(answer),
        "answer_sentences": [sentence],
        "historically_exposed": True,
        "official_split": "train",
        "source_role": "fit",
        "labels_accessible": False,
        "fresh_confirmatory_claim_allowed": False,
    }


def test_7659_scope_and_unknown_denominator() -> None:
    """SCENARIO-REPORT-7659-SCOPE checks only a visible raise statement."""
    row = source_row("g1")
    result = extract_group(row, row["complete_source"], "original_source", "g1")
    assert result["checked_structural_propositions"] == 1
    assert result["lexical_membership"] >= 1
    assert result["unsupported_semantic_spans"] >= 1
    assert result["whole_answer_certified"] is False
    assert result["denominator"] == 1
    erased = extract_group(row, "", "evidence_erasure", None)
    assert erased["checked_structural_propositions"] == 0
    assert erased["unknown_claims"] >= 1
    assert erased["denominator"] == 1


def test_7659_role_derangement_preserves_groups() -> None:
    """SCENARIO-REPORT-7659-ROSTER keeps one unit in every arm."""
    one, two = source_row("g1"), source_row("g2", 2)
    rows = build_role_rows([one, two], "fit", {"g1": "g2", "g2": "g1"})
    assert len(rows) == 6
    assert {row["unit_id"] for row in rows} == {"g1", "g2"}
    assert [row["source_group_id"] for row in rows[:3]] == ["g1", None, "g2"]
    assert all(row["answer_sha256"] == one["answer_sha256"] for row in rows[:3])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"role": "evaluation"}, "role_mismatch"),
        ({"source_sha256": "sha256:wrong"}, "source_hash_mismatch"),
        ({"dataset_hallucination_label": 1}, "label_access_rejected"),
        ({"labels_accessible": True}, "label_access_rejected"),
    ],
)
def test_7659_rejects_custody_and_label_mutations(change: dict, message: str) -> None:
    """SCENARIO-REPORT-7659-ISOLATION rejects drift before extraction."""
    row = source_row("g1")
    row.update(change)
    with pytest.raises(ValueError, match=message):
        validate_input(row, source_row("g1"), "fit")


def test_7659_rejects_offset_mutation() -> None:
    """SCENARIO-REPORT-7659-ISOLATION binds original sentence bytes."""
    row = source_row("g1")
    altered = deepcopy(row)
    altered["answer_sentences"][0]["byte_end"] -= 1
    with pytest.raises(ValueError, match="answer_offset_mismatch"):
        validate_input(altered, row, "fit")


def test_7659_rejects_expected_text_drift_and_unknown_arm() -> None:
    """SCENARIO-REPORT-7659-ISOLATION compares exact original text."""
    row = source_row("g1")
    expected = deepcopy(row)
    expected["complete_source"] += " changed"
    with pytest.raises(ValueError, match="complete_source_mismatch"):
        validate_input(row, expected, "fit")
    expected = deepcopy(row)
    expected["learning_partition"] = "fit_old_distribution_anchor"
    with pytest.raises(ValueError, match="learning_partition_mismatch"):
        validate_input(row, expected, "fit")
    with pytest.raises(ValueError, match="unknown_arm"):
        extract_group(row, row["complete_source"], "bad", "g1")


def test_7659_rejects_duplicate_and_non_deranged_roles() -> None:
    """SCENARIO-REPORT-7659-ROSTER keeps independent units unique."""
    one, two = source_row("g1"), source_row("g2")
    with pytest.raises(ValueError, match="duplicate_or_wrong_role"):
        build_role_rows([one, one], "fit", {"g1": "g1"})
    with pytest.raises(ValueError, match="derangement_roster_mismatch"):
        build_role_rows([one, two], "fit", {"g1": "g2"})
    with pytest.raises(ValueError, match="derangement_fixed_point"):
        build_role_rows([one, two], "fit", {"g1": "g1", "g2": "g2"})


def test_7659_lexical_presence_is_ambiguous() -> None:
    """SCENARIO-REPORT-7659-SCOPE keeps mentions below structural truth."""
    row = source_row("g1")
    source = "Tool output:\n```\n1: result = ValidationError()\n```"
    result = extract_group(row, source, "original_source", "g1")
    assert result["ambiguity"] == 1
    assert result["checked_structural_propositions"] == 0
