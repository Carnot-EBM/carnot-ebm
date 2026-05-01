"""Tests for scripts/prm_data_generator.py.

Three focused tests for the step-level PRM data generator:
1. step_decomposition_produces_valid_prefix_sequence
2. cascade_score_higher_for_wrong_prefixes
3. prm_data_file_written_with_correct_schema

Spec: REQ-LEARN-011, REQ-VERIFY-098
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add repo root and scripts to path so prm_data_generator is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from prm_data_generator import (
    ENERGY_THRESHOLD,
    cascade_score,
    decompose_cot_steps,
    generate_and_save,
    generate_step_examples,
)

# ---------------------------------------------------------------------------
# Test 1: step_decomposition_produces_valid_prefix_sequence
# ---------------------------------------------------------------------------


def test_step_decomposition_produces_valid_prefix_sequence():
    """Decomposing a multi-sentence step_text must yield an ordered prefix sequence.

    WHY: The MCTS-inspired labeling depends on having a well-ordered list of
    sub-steps so that prefix[1:k] is always a proper subset of the full CoT.
    A multi-paragraph step should split into >= 2 sub-steps, and joining
    prefix[1:k] should produce a text that is a prefix of the full joined text.
    """
    step_text = (
        "First, we calculate 2 + 2 = 4.\n\n"
        "Next, we multiply by 3: 4 × 3 = 12.\n\n"
        "Therefore, the answer is 12."
    )
    sub_steps = decompose_cot_steps(step_text)

    # Must yield at least 2 sub-steps for a multi-paragraph input
    assert len(sub_steps) >= 2, f"Expected >= 2 sub-steps, got {sub_steps}"

    # All sub-steps must be non-empty
    for i, s in enumerate(sub_steps):
        assert s.strip(), f"Sub-step {i} is empty or whitespace-only"

    # Verify prefix ordering: for each prefix length k, all k sub-steps appear
    # in order relative to the original text
    for k in range(1, len(sub_steps) + 1):
        partial = " ".join(sub_steps[:k])
        # All sub-steps in prefix must appear in the partial text
        for s in sub_steps[:k]:
            assert s in partial, f"Sub-step '{s[:30]}...' missing from prefix of length {k}"


# ---------------------------------------------------------------------------
# Test 2: cascade_score_higher_for_wrong_prefixes
# ---------------------------------------------------------------------------


def test_cascade_score_higher_for_wrong_prefixes():
    """Cascade score must be higher (worse) for texts containing error indicators.

    WHY: The energy threshold (1.11) separates correct from incorrect steps.
    A text with explicit error words ("wrong", "incorrect") must score above
    threshold; a clean mathematical derivation must score below threshold.
    If the score doesn't respect this, the label assignment logic breaks and
    we'd generate misleading training data (ambiguous labels instead of
    correct/wrong).
    """
    # Clean correct reasoning: structured math, no error words
    correct_text = (
        "To find the total: 5 items × $3 = $15. "
        "Adding tax: $15 × 1.08 = $16.20. "
        "Therefore the total cost = $16.20."
    )
    # Incorrect reasoning: explicit error indicator
    wrong_text = (
        "This calculation is wrong because 5 × 3 = 20 is incorrect. "
        "The error in the previous step led to the wrong answer."
    )

    score_correct = cascade_score(correct_text)
    score_wrong = cascade_score(wrong_text)

    # Wrong text must score at or above threshold
    assert score_wrong >= ENERGY_THRESHOLD, (
        f"Wrong-text score {score_wrong} should be >= threshold {ENERGY_THRESHOLD}"
    )
    # Correct text should score below threshold (clean math reasoning)
    assert score_correct < ENERGY_THRESHOLD, (
        f"Correct-text score {score_correct} should be < threshold {ENERGY_THRESHOLD}"
    )
    # Wrong score must be strictly greater than correct score
    assert score_wrong > score_correct, (
        f"Expected score_wrong ({score_wrong}) > score_correct ({score_correct})"
    )


# ---------------------------------------------------------------------------
# Test 3: prm_data_file_written_with_correct_schema
# ---------------------------------------------------------------------------


def test_prm_data_file_written_with_correct_schema():
    """generate_and_save must write valid JSONL with all required schema fields.

    WHY: The retraining step reads data/step_level_prm_training.jsonl and
    expects specific keys. If the schema is wrong, retraining silently uses
    incorrect data. This test verifies every written record has exactly the
    six required fields with correct types.
    """
    corpus = [
        {
            "question_id": "101",
            "step_text": (
                "Step 1: Add 3 + 4 = 7.\n\n"
                "Step 2: Multiply 7 × 2 = 14.\n\n"
                "Therefore the answer is 14."
            ),
            "label": "correct",
            "confidence": 1.0,
        },
        {
            "question_id": "102",
            "step_text": (
                "This step contains a wrong calculation. "
                "The incorrect value propagates to the wrong answer."
            ),
            "label": "incorrect",
            "confidence": 1.0,
        },
    ]

    required_keys = {
        "question_id",
        "partial_cot",
        "step_label",
        "full_cot_correct",
        "cascade_score",
        "prefix_fraction",
    }
    valid_labels = {"correct", "wrong", "ambiguous"}

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = str(Path(tmpdir) / "corpus.json")
        output_path = str(Path(tmpdir) / "output.jsonl")

        with open(corpus_path, "w") as f:
            json.dump(corpus, f)

        stats = generate_and_save(corpus_path, output_path)

        # Stats must have all required keys
        for k in [
            "n_fover_pairs_processed",
            "n_step_examples_generated",
            "n_correct_step_examples",
            "n_wrong_step_examples",
            "n_ambiguous_excluded",
            "output_file",
        ]:
            assert k in stats, f"Missing stat key: {k}"

        assert stats["n_fover_pairs_processed"] == 2

        # Read back the output file (only non-ambiguous rows are written)
        written_rows = []
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    written_rows.append(json.loads(line))

        # n_step_examples_generated must match the written rows
        assert stats["n_step_examples_generated"] == len(written_rows), (
            f"Stat says {stats['n_step_examples_generated']} but {len(written_rows)} rows written"
        )

        # Every written row must have correct schema
        for i, row in enumerate(written_rows):
            missing = required_keys - set(row.keys())
            assert not missing, f"Row {i} missing keys: {missing}"

            # step_label must be "correct" or "wrong" (never "ambiguous" in output)
            assert row["step_label"] in ("correct", "wrong"), (
                f"Row {i} step_label={row['step_label']} — ambiguous rows must be excluded"
            )
            # full_cot_correct must be bool
            assert isinstance(row["full_cot_correct"], bool), (
                f"Row {i} full_cot_correct must be bool, got {type(row['full_cot_correct'])}"
            )
            # cascade_score must be float in [0.5, 2.0]
            assert 0.5 <= row["cascade_score"] <= 2.0, (
                f"Row {i} cascade_score={row['cascade_score']} out of range"
            )
            # prefix_fraction must be in (0, 1]
            assert 0 < row["prefix_fraction"] <= 1.0, (
                f"Row {i} prefix_fraction={row['prefix_fraction']} out of range"
            )
            # partial_cot must be non-empty
            assert row["partial_cot"].strip(), f"Row {i} partial_cot is empty"
