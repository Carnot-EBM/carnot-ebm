"""Apple adversarial GSM8K dataset generator — Exp 281 tests.

Tests that the dataset generator in
scripts/experiment_281_apple_adversarial_dataset.py:
  - produces exactly 400 rows (200 cohort questions × 2 variants)
  - produces both variant types (number_swap, irrelevant_sentence)
  - number_swap changes the answer for the majority of rows
  - irrelevant_sentence preserves the answer for ALL rows
  - inserted irrelevant sentence contains a number absent from original
  - output schema is valid (all required fields present)
  - seeds do not collide with Exp 119 (base seed 119) or Exp 279 (279_000+)
  - output is reproducible across two calls

Spec: REQ-VERIFY-063, SCENARIO-VERIFY-078, SCENARIO-VERIFY-079
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Module loading helper
# ---------------------------------------------------------------------------

def _load_module() -> Any:
    """Load experiment_281_apple_adversarial_dataset without executing main()."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_281_apple_adversarial_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_281_apple_adversarial_dataset", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _extract_numbers(text: str) -> set[int]:
    """Extract all integer numbers from text."""
    return {int(m) for m in re.findall(r"\b\d+\b", text)}


# ===========================================================================
# Row count
# ===========================================================================

# REQ-VERIFY-063 — dataset must have 200 × 2 = 400 rows
def test_generate_dataset_row_count() -> None:
    """SCENARIO-VERIFY-078: generator produces exactly 400 rows (200 × 2 variants)."""
    mod = _load_module()
    rows = mod.generate_dataset()
    assert len(rows) == 400, f"Expected 400 rows, got {len(rows)}"


# ===========================================================================
# Both variant types present
# ===========================================================================

# REQ-VERIFY-063
def test_both_variant_types_present() -> None:
    """SCENARIO-VERIFY-078/079: both number_swap and irrelevant_sentence variants exist."""
    mod = _load_module()
    rows = mod.generate_dataset()
    types = {r["variant_type"] for r in rows}
    assert "number_swap" in types, "number_swap variant missing"
    assert "irrelevant_sentence" in types, "irrelevant_sentence variant missing"


# REQ-VERIFY-063
def test_variant_type_counts_equal() -> None:
    """REQ-VERIFY-063: exactly 200 rows of each variant type."""
    mod = _load_module()
    rows = mod.generate_dataset()
    ns_count = sum(1 for r in rows if r["variant_type"] == "number_swap")
    ir_count = sum(1 for r in rows if r["variant_type"] == "irrelevant_sentence")
    assert ns_count == 200, f"Expected 200 number_swap rows, got {ns_count}"
    assert ir_count == 200, f"Expected 200 irrelevant_sentence rows, got {ir_count}"


# ===========================================================================
# number_swap: answer changes
# ===========================================================================

# REQ-VERIFY-063, SCENARIO-VERIFY-078
def test_number_swap_changes_at_least_one_number() -> None:
    """SCENARIO-VERIFY-078: every number_swap row has a different number set."""
    mod = _load_module()
    rows = mod.generate_dataset()
    ns_rows = [r for r in rows if r["variant_type"] == "number_swap"]
    for row in ns_rows:
        orig_nums = _extract_numbers(row["original_question"])
        var_nums = _extract_numbers(row["variant_question"])
        assert orig_nums != var_nums, (
            f"question_id={row['question_id']}: number_swap produced identical number set"
        )


# SCENARIO-VERIFY-078
def test_number_swap_changes_answer_for_majority() -> None:
    """SCENARIO-VERIFY-078: number_swap variant_answer != original_answer for majority."""
    mod = _load_module()
    rows = mod.generate_dataset()
    ns_rows = [r for r in rows if r["variant_type"] == "number_swap"]
    changed = sum(1 for r in ns_rows if r["variant_answer"] != r["original_answer"])
    # At least 50% of swaps should produce a different answer
    assert changed >= len(ns_rows) // 2, (
        f"Only {changed}/{len(ns_rows)} number_swap rows changed the answer"
    )


# ===========================================================================
# irrelevant_sentence: answer preserved
# ===========================================================================

# REQ-VERIFY-063, SCENARIO-VERIFY-079
def test_irrelevant_sentence_preserves_answer() -> None:
    """SCENARIO-VERIFY-079: irrelevant_sentence variant_answer == original_answer for ALL rows."""
    mod = _load_module()
    rows = mod.generate_dataset()
    ir_rows = [r for r in rows if r["variant_type"] == "irrelevant_sentence"]
    for row in ir_rows:
        assert row["variant_answer"] == row["original_answer"], (
            f"question_id={row['question_id']}: irrelevant_sentence changed the answer "
            f"({row['original_answer']} → {row['variant_answer']})"
        )


# SCENARIO-VERIFY-079
def test_irrelevant_sentence_inserts_new_sentence() -> None:
    """SCENARIO-VERIFY-079: variant_question is longer than original_question."""
    mod = _load_module()
    rows = mod.generate_dataset()
    ir_rows = [r for r in rows if r["variant_type"] == "irrelevant_sentence"]
    for row in ir_rows:
        assert len(row["variant_question"]) > len(row["original_question"]), (
            f"question_id={row['question_id']}: irrelevant_sentence did not extend the question"
        )


# SCENARIO-VERIFY-079
def test_irrelevant_sentence_contains_distractor_number() -> None:
    """SCENARIO-VERIFY-079: inserted sentence contains a number absent from original."""
    mod = _load_module()
    rows = mod.generate_dataset()
    ir_rows = [r for r in rows if r["variant_type"] == "irrelevant_sentence"]
    # At least 90% of rows should have a new distractor number (some sentences
    # may reuse an existing number by coincidence, so allow a small tolerance)
    has_new_num = 0
    for row in ir_rows:
        orig_nums = _extract_numbers(row["original_question"])
        var_nums = _extract_numbers(row["variant_question"])
        if var_nums - orig_nums:
            has_new_num += 1
    assert has_new_num >= int(len(ir_rows) * 0.9), (
        f"Only {has_new_num}/{len(ir_rows)} irrelevant_sentence rows added a new number"
    )


# ===========================================================================
# Schema validation
# ===========================================================================

# REQ-VERIFY-063
def test_row_schema() -> None:
    """REQ-VERIFY-063: every row has all required fields with correct types."""
    mod = _load_module()
    rows = mod.generate_dataset()
    required_fields = {
        "question_id", "original_question", "original_answer",
        "variant_type", "variant_question", "variant_answer", "provenance",
    }
    for i, row in enumerate(rows):
        missing = required_fields - set(row.keys())
        assert not missing, f"Row {i} missing fields: {missing}"
        assert isinstance(row["question_id"], str), f"Row {i}: question_id must be str"
        assert isinstance(row["original_question"], str), f"Row {i}: original_question must be str"
        assert isinstance(row["original_answer"], int), f"Row {i}: original_answer must be int"
        assert row["variant_type"] in ("number_swap", "irrelevant_sentence"), (
            f"Row {i}: unknown variant_type {row['variant_type']!r}"
        )
        assert isinstance(row["variant_question"], str), f"Row {i}: variant_question must be str"
        assert isinstance(row["variant_answer"], int), f"Row {i}: variant_answer must be int"
        assert isinstance(row["provenance"], dict), f"Row {i}: provenance must be dict"


# ===========================================================================
# No seed collision with Exp 119
# ===========================================================================

# REQ-VERIFY-063
def test_no_seed_collision_with_exp119() -> None:
    """REQ-VERIFY-063: Exp 281 seeds (281_000+) do not collide with Exp 119 base (119)."""
    mod = _load_module()
    # The base seed for Exp 281 must be >= 281_000
    assert mod.BASE_SEED >= 281_000, (
        f"BASE_SEED={mod.BASE_SEED} collides with Exp 119 seed range"
    )


# ===========================================================================
# Reproducibility
# ===========================================================================

# REQ-VERIFY-063
def test_generate_dataset_is_reproducible() -> None:
    """REQ-VERIFY-063: two calls produce identical output."""
    mod = _load_module()
    rows_a = mod.generate_dataset()
    rows_b = mod.generate_dataset()
    assert rows_a == rows_b, "generate_dataset() is not reproducible across two calls"


# ===========================================================================
# Provenance fields
# ===========================================================================

# REQ-VERIFY-063
def test_provenance_contains_experiment_id() -> None:
    """REQ-VERIFY-063: provenance dict identifies the source experiment."""
    mod = _load_module()
    rows = mod.generate_dataset()
    for row in rows[:5]:  # spot-check first 5
        assert "experiment" in row["provenance"], (
            f"question_id={row['question_id']}: provenance missing 'experiment' key"
        )
        assert "281" in str(row["provenance"]["experiment"]), (
            f"question_id={row['question_id']}: provenance 'experiment' does not reference Exp 281"
        )
