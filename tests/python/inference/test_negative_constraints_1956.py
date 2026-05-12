"""Tests for Exp 1956 NCO-style negative constraint decoding.

Spec: REQ-INFER-1956, SCENARIO-INFER-1956.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.inference.negative_constraints import (
    NegativeConstraint,
    NegativeConstraintRegistry,
    PositiveMaskTrie,
    REQUIRED_ARTIFACT_FIELDS,
    benchmark_negative_vs_positive_trie,
    decode_with_negative_constraints,
    run_experiment,
    validate_artifact,
)


def test_req_infer_1956_registry_rejects_literal_and_regex_candidates() -> None:
    """REQ-INFER-1956: registry tracks literal and regex token exclusion patterns."""

    registry = NegativeConstraintRegistry()
    literal = registry.add_literal("profanity", "badword")
    regex = registry.add_regex("email", r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", window=64)
    token_text_by_id = {
        1: "WORD",
        2: " safe",
        3: " user@example.com",
    }

    report = registry.rejection_report("prefix bad", token_text_by_id)

    assert literal.kind == "literal"
    assert regex.kind == "regex"
    assert registry.max_lookback == 64
    assert registry.rejected_token_ids("prefix bad", token_text_by_id) == {1, 3}
    assert report[1].constraint_names == ("profanity",)
    assert report[3].constraint_names == ("email",)
    assert registry.matching_constraints("clean", " token") == ()


def test_req_infer_1956_registry_validation_paths() -> None:
    """REQ-INFER-1956: invalid constraints fail before entering the registry."""

    registry = NegativeConstraintRegistry()
    constraint = NegativeConstraint("strict", "Case", "literal", case_sensitive=True)
    registry.register(constraint)

    assert registry.matching_constraints("ca", "se") == ()
    assert registry.matching_constraints("Ca", "se") == (constraint,)

    with pytest.raises(ValueError, match="name"):
        NegativeConstraint("", "x", "literal")
    with pytest.raises(ValueError, match="pattern"):
        NegativeConstraint("empty", "", "literal")
    with pytest.raises(ValueError, match="kind"):
        NegativeConstraint("bad-kind", "x", "glob")
    with pytest.raises(ValueError, match="window"):
        NegativeConstraint("bad-window", "x", "literal", window=0)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(constraint)
    with pytest.raises(ValueError, match="unterminated"):
        registry.add_regex("broken", "(")


def test_scenario_infer_1956_decode_rejects_split_token_negative_constraint() -> None:
    """SCENARIO-INFER-1956: online rejection blocks the token completing a pattern."""

    registry = NegativeConstraintRegistry()
    registry.add_literal("blocked-word", "badword")
    token_text_by_id = {
        1: " bad",
        2: "word",
        3: " clean",
        4: " ok",
    }
    score_rows = [
        {1: 0.9, 3: 0.1},
        {2: 0.95, 4: 0.4},
    ]

    result = decode_with_negative_constraints("", token_text_by_id, score_rows, registry)

    assert result.completed is True
    assert result.text == " bad ok"
    assert result.token_ids == (1, 4)
    assert result.rejected_count == 1
    assert result.steps[1].selected_token_id == 4
    assert result.steps[1].rejections[2].constraint_names == ("blocked-word",)


def test_req_infer_1956_decode_reports_all_rejected_and_positive_trie_masks() -> None:
    """REQ-INFER-1956: positive trie masking composes with negative rejection."""

    registry = NegativeConstraintRegistry([NegativeConstraint("no-x", "x", "literal")])
    token_text_by_id = {1: "a", 2: "x", 3: "z"}
    trie = PositiveMaskTrie.from_token_sequences([[1, 2], [1, 3]])
    first = decode_with_negative_constraints(
        "",
        token_text_by_id,
        [{1: 0.2, 2: 0.9}, {2: 0.8, 3: 0.7}],
        registry,
        positive_trie=trie,
    )
    stopped = decode_with_negative_constraints(
        "",
        token_text_by_id,
        [{2: 1.0}],
        registry,
        positive_trie=PositiveMaskTrie.from_token_sequences([[2]]),
    )

    assert trie.allowed_next(()) == {1}
    assert trie.allowed_next((1, 2)) == set()
    assert trie.allowed_next((9,)) == set()
    assert first.text == "az"
    assert first.steps[0].positive_allowed_count == 1
    assert first.steps[1].positive_allowed_count == 2
    assert stopped.completed is False
    assert stopped.stopped_reason == "all_candidates_rejected"
    assert stopped.rejected_count == 1


def test_req_infer_1956_benchmark_reports_overhead_vs_positive_trie() -> None:
    """REQ-INFER-1956: benchmark reports NCO overhead against positive-mask trie."""

    metrics = benchmark_negative_vs_positive_trie(repeats=2)

    assert metrics["tokens_evaluated"] > 0
    assert metrics["candidate_checks"] > 0
    assert metrics["nco_ns_per_token"] > 0
    assert metrics["positive_trie_ns_per_token"] > 0
    assert metrics["overhead_ratio"] > 0
    assert metrics["nco_rejected_count"] >= 1


def test_req_infer_1956_experiment_artifact_schema(tmp_path: Path) -> None:
    """REQ-INFER-1956: runner writes the required Exp 1956 result fields."""

    output_path = tmp_path / "experiment_1956_nco_negative_constraints.json"
    artifact = run_experiment(
        output_path=output_path,
        run_date="20260512",
        tests_run=[".venv/bin/pytest tests/python/inference/test_negative_constraints_1956.py -q"],
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert validate_artifact(artifact) is True
    assert artifact["status"] == "complete"
    assert artifact["nco_negative_constraint_layer_ready"] is True
    assert artifact["negative_constraints_upheld"] is True
    assert artifact["overhead_vs_positive_trie"]["nco_rejected_count"] >= 1
    assert artifact["tests_run"]
    assert artifact["honest_verdict"].startswith("complete:")

    broken = dict(artifact)
    broken.pop("status")
    with pytest.raises(ValueError, match="missing"):
        validate_artifact(broken)

    broken = dict(artifact)
    broken["honest_verdict"] = "partial"
    with pytest.raises(ValueError, match="complete"):
        validate_artifact(broken)
