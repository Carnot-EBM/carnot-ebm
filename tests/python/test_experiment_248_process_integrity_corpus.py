"""Tests for Experiment 248: process integrity corpus generation.

Spec: REQ-VERIFY-060 (process integrity labeling from checked-in traces)
SCENARIO-VERIFY-070 (corpus schema shape),
SCENARIO-VERIFY-071 (deterministic generation),
SCENARIO-VERIFY-072 (all five process labels covered),
SCENARIO-VERIFY-073 (provenance links to all source experiments)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_248_process_integrity_corpus.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_248_process_integrity_corpus", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


REQUIRED_FIELDS = {
    "corpus_id",
    "run_date",
    "experiment",
    "source_experiment",
    "source_artifact",
    "benchmark",
    "domain",
    "model",
    "case_id",
    "iteration",
    "outcome_label",
    "process_label",
    "process_evidence",
    "steps",
    "final_answer",
}

VALID_PROCESS_LABELS = {
    "right_answer_wrong_process",
    "wrong_answer_partially_sound_process",
    "unsupported_step",
    "repair_fixed_outcome_only",
    "repair_fixed_process_and_outcome",
    "clean",
}

VALID_OUTCOME_LABELS = {"correct", "incorrect"}
VALID_DOMAINS = {"reasoning", "code"}


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-070: schema shape
# ---------------------------------------------------------------------------


def test_schema_shape():
    """Every corpus row must carry required fields with valid enum values."""
    module = load_module()
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "process_integrity_corpus_248.jsonl"
    assert corpus_path.exists(), f"Corpus not found at {corpus_path}"
    rows = read_jsonl(corpus_path)
    assert len(rows) > 0, "Corpus must not be empty"

    for i, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row.keys())
        assert not missing, f"Row {i} missing fields: {missing}"
        assert row["process_label"] in VALID_PROCESS_LABELS, (
            f"Row {i} has invalid process_label: {row['process_label']!r}"
        )
        assert row["outcome_label"] in VALID_OUTCOME_LABELS, (
            f"Row {i} has invalid outcome_label: {row['outcome_label']!r}"
        )
        assert row["domain"] in VALID_DOMAINS, (
            f"Row {i} has invalid domain: {row['domain']!r}"
        )
        assert row["run_date"] == "20260413", f"Row {i} wrong run_date: {row['run_date']!r}"
        assert row["experiment"] == 248, f"Row {i} wrong experiment: {row['experiment']}"
        assert isinstance(row["steps"], list), f"Row {i} steps must be list"
        assert isinstance(row["process_evidence"], dict), f"Row {i} process_evidence must be dict"
        assert row["iteration"] >= 0, f"Row {i} iteration must be >= 0"
        assert isinstance(row["corpus_id"], str) and row["corpus_id"], (
            f"Row {i} corpus_id must be non-empty string"
        )


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-071: deterministic generation
# ---------------------------------------------------------------------------


def test_deterministic_generation(tmp_path: Path):
    """Running the corpus builder twice produces byte-identical output."""
    module = load_module()
    repo_root = Path(__file__).resolve().parents[2]

    out1 = tmp_path / "run1" / "corpus.jsonl"
    out2 = tmp_path / "run2" / "corpus.jsonl"

    module.build_and_write(repo_root=repo_root, corpus_path=out1, summary_path=tmp_path / "run1" / "summary.json")
    module.build_and_write(repo_root=repo_root, corpus_path=out2, summary_path=tmp_path / "run2" / "summary.json")

    rows1 = read_jsonl(out1)
    rows2 = read_jsonl(out2)
    assert len(rows1) == len(rows2), "Row count must be stable across runs"
    for i, (r1, r2) in enumerate(zip(rows1, rows2)):
        assert r1 == r2, f"Row {i} differs between runs: {r1!r} vs {r2!r}"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-072: all five process labels present
# ---------------------------------------------------------------------------


def test_all_five_process_labels_present():
    """The corpus must include at least one entry for each of the five process labels."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "process_integrity_corpus_248.jsonl"
    rows = read_jsonl(corpus_path)
    labels_found = {row["process_label"] for row in rows}
    required_labels = {
        "right_answer_wrong_process",
        "wrong_answer_partially_sound_process",
        "unsupported_step",
        "repair_fixed_outcome_only",
        "repair_fixed_process_and_outcome",
    }
    missing = required_labels - labels_found
    assert not missing, (
        f"Corpus missing required process labels: {missing}\n"
        f"Labels found: {labels_found}"
    )


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-073: provenance coverage across source experiments
# ---------------------------------------------------------------------------


def test_provenance_covers_all_source_experiments():
    """Each source experiment (235, 238) must appear in the corpus provenance."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "process_integrity_corpus_248.jsonl"
    rows = read_jsonl(corpus_path)
    source_experiments_found = {row["source_experiment"] for row in rows}
    required = {235, 238}
    missing = required - source_experiments_found
    assert not missing, (
        f"Corpus missing entries from source experiments: {missing}\n"
        f"Source experiments found: {source_experiments_found}"
    )


# ---------------------------------------------------------------------------
# Unit tests for classification functions
# ---------------------------------------------------------------------------


def test_classify_reasoning_right_answer_wrong_process():
    """REQ-VERIFY-060: correct answer with unsupported step → right_answer_wrong_process."""
    module = load_module()
    # correct=True, one claim has premise_support < 0.3 (unsupported)
    claim_results = [
        {"claim_id": "cl1", "is_final": False, "premise_support": 0.1, "missing_clause_ids": ["p1"]},
        {"claim_id": "cl2", "is_final": False, "premise_support": 0.9, "missing_clause_ids": []},
        {"claim_id": "final_answer", "is_final": True, "premise_support": 1.0, "missing_clause_ids": []},
    ]
    label = module.classify_reasoning(
        is_correct=True,
        verdict="abstain",
        claim_results=claim_results,
        is_repair=False,
        prior_correct=False,
    )
    assert label == "right_answer_wrong_process", f"Expected right_answer_wrong_process, got {label!r}"


def test_classify_reasoning_wrong_answer_partially_sound():
    """REQ-VERIFY-060: wrong answer with majority of steps sound → wrong_answer_partially_sound_process."""
    module = load_module()
    claim_results = [
        {"claim_id": "cl1", "is_final": False, "premise_support": 0.9, "missing_clause_ids": []},
        {"claim_id": "cl2", "is_final": False, "premise_support": 0.8, "missing_clause_ids": []},
        {"claim_id": "cl3", "is_final": False, "premise_support": 0.7, "missing_clause_ids": []},
        {"claim_id": "final_answer", "is_final": True, "premise_support": 0.0, "missing_clause_ids": []},
    ]
    label = module.classify_reasoning(
        is_correct=False,
        verdict="violated",
        claim_results=claim_results,
        is_repair=False,
        prior_correct=False,
    )
    assert label == "wrong_answer_partially_sound_process", f"Got {label!r}"


def test_classify_reasoning_unsupported_step():
    """REQ-VERIFY-060: wrong answer with unsupported step → unsupported_step."""
    module = load_module()
    claim_results = [
        {"claim_id": "cl1", "is_final": False, "premise_support": 0.1, "missing_clause_ids": ["p1"]},
        {"claim_id": "cl2", "is_final": False, "premise_support": 0.2, "missing_clause_ids": ["p2"]},
    ]
    label = module.classify_reasoning(
        is_correct=False,
        verdict="violated",
        claim_results=claim_results,
        is_repair=False,
        prior_correct=False,
    )
    assert label == "unsupported_step", f"Got {label!r}"


def test_classify_reasoning_repair_fixed_process_and_outcome():
    """REQ-VERIFY-060: repair iteration that fixed both outcome and process."""
    module = load_module()
    claim_results = [
        {"claim_id": "cl1", "is_final": False, "premise_support": 1.0, "missing_clause_ids": []},
        {"claim_id": "cl2", "is_final": False, "premise_support": 0.9, "missing_clause_ids": []},
    ]
    label = module.classify_reasoning(
        is_correct=True,
        verdict="abstain",
        claim_results=claim_results,
        is_repair=True,
        prior_correct=False,
    )
    assert label == "repair_fixed_process_and_outcome", f"Got {label!r}"


def test_classify_reasoning_repair_fixed_outcome_only():
    """REQ-VERIFY-060: repair iteration that fixed outcome but not process."""
    module = load_module()
    claim_results = [
        {"claim_id": "cl1", "is_final": False, "premise_support": 0.1, "missing_clause_ids": ["p1"]},
    ]
    label = module.classify_reasoning(
        is_correct=True,
        verdict="violated",
        claim_results=claim_results,
        is_repair=True,
        prior_correct=False,
    )
    assert label == "repair_fixed_outcome_only", f"Got {label!r}"


def test_classify_code_right_answer_wrong_process():
    """REQ-VERIFY-060: code passes official tests but has spec violations → right_answer_wrong_process."""
    module = load_module()
    label = module.classify_code(
        is_correct=True,
        n_pbt_failures=0,
        n_spec_violations=3,
        pbt_verified=True,
        n_derived_props=2,
        is_repair=False,
        prior_correct=False,
    )
    assert label == "right_answer_wrong_process", f"Got {label!r}"


def test_classify_code_unsupported_step():
    """REQ-VERIFY-060: official fails, PBT verified, but spec has violations → unsupported_step."""
    module = load_module()
    label = module.classify_code(
        is_correct=False,
        n_pbt_failures=0,
        n_spec_violations=5,
        pbt_verified=True,
        n_derived_props=2,
        is_repair=False,
        prior_correct=False,
    )
    assert label == "unsupported_step", f"Got {label!r}"


def test_classify_code_wrong_answer_partially_sound():
    """REQ-VERIFY-060: some PBT props pass on wrong answer → wrong_answer_partially_sound_process."""
    module = load_module()
    label = module.classify_code(
        is_correct=False,
        n_pbt_failures=1,
        n_spec_violations=0,
        pbt_verified=False,
        n_derived_props=3,
        is_repair=False,
        prior_correct=False,
    )
    assert label == "wrong_answer_partially_sound_process", f"Got {label!r}"


def test_classify_code_repair_fixed_process_and_outcome():
    """REQ-VERIFY-060: repair iteration fixes both outcome and process for code."""
    module = load_module()
    label = module.classify_code(
        is_correct=True,
        n_pbt_failures=0,
        n_spec_violations=0,
        pbt_verified=True,
        n_derived_props=2,
        is_repair=True,
        prior_correct=False,
    )
    assert label == "repair_fixed_process_and_outcome", f"Got {label!r}"


def test_classify_code_repair_fixed_outcome_only():
    """REQ-VERIFY-060: repair fixes outcome but spec violations remain → repair_fixed_outcome_only."""
    module = load_module()
    label = module.classify_code(
        is_correct=True,
        n_pbt_failures=0,
        n_spec_violations=2,
        pbt_verified=True,
        n_derived_props=2,
        is_repair=True,
        prior_correct=False,
    )
    assert label == "repair_fixed_outcome_only", f"Got {label!r}"


# ---------------------------------------------------------------------------
# Summary artifact shape
# ---------------------------------------------------------------------------


def test_summary_artifact_shape():
    """The companion summary must report label counts by source benchmark and model."""
    repo_root = Path(__file__).resolve().parents[2]
    summary_path = repo_root / "results" / "experiment_248_results.json"
    assert summary_path.exists(), f"Summary not found at {summary_path}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary.get("experiment") == 248
    assert summary.get("run_date") == "20260413"
    assert "label_counts" in summary, "Summary must include label_counts"
    assert "by_source_benchmark" in summary, "Summary must include by_source_benchmark"
    assert "by_model" in summary, "Summary must include by_model"
    assert "source_artifacts" in summary, "Summary must include source_artifacts"
    assert "corpus_path" in summary, "Summary must include corpus_path"
    assert "total_rows" in summary, "Summary must include total_rows"

    # Every required label must appear in label_counts
    required_labels = {
        "right_answer_wrong_process",
        "wrong_answer_partially_sound_process",
        "unsupported_step",
        "repair_fixed_outcome_only",
        "repair_fixed_process_and_outcome",
    }
    lc = summary["label_counts"]
    for label in required_labels:
        assert label in lc and lc[label] > 0, (
            f"Summary label_counts missing or zero for {label!r}"
        )
