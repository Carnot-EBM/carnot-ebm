"""Tests for Experiment 252: predictive verification corpus generation.

Spec: REQ-VERIFY-252 (predictive verification corpus from live artifacts)
SCENARIO-VERIFY-252-A (corpus schema shape and required fields),
SCENARIO-VERIFY-252-B (deterministic generation — two runs produce identical output),
SCENARIO-VERIFY-252-C (provenance completeness — all source experiments covered),
SCENARIO-VERIFY-252-D (both semantic and code traces present),
SCENARIO-VERIFY-252-E (memory-hit metadata present on held-out decisions),
SCENARIO-VERIFY-252-F (accepted-repair field populated for successful repairs)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_252_predictive_verification_corpus.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_252_predictive_verification_corpus", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-252-A: schema shape and required fields
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "corpus_id",
    "run_date",
    "experiment",
    "source_experiment",
    "source_artifact",
    "benchmark",
    "benchmark_slice",
    "domain",
    "model",
    "case_id",
    "sample_position",
    "partial_response",
    "final_response",
    "violation_family",
    "process_label",
    "outcome_label",
    "verifier_outcome",
    "confidence",
    "baseline_latency_seconds",
    "repair_latency_seconds",
    "downstream_repair_outcome",
    "memory_hit",
    "memory_match_metadata",
    "policy_context",
    "accepted_repair",
    "provenance",
}

VALID_DOMAINS = {"reasoning", "code"}
VALID_OUTCOME_LABELS = {"correct", "incorrect"}
VALID_VERIFIER_OUTCOMES = {"verified", "violated", "abstain", "unknown"}
VALID_REPAIR_OUTCOMES = {"accepted", "rejected", "not_attempted"}


def test_schema_shape():
    """Every corpus row must carry all required fields with valid enum values."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "predictive_verification_corpus_252.jsonl"
    assert corpus_path.exists(), f"Corpus not found at {corpus_path}"
    rows = read_jsonl(corpus_path)
    assert len(rows) >= 50, f"Corpus must have at least 50 rows, got {len(rows)}"

    for i, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row.keys())
        assert not missing, f"Row {i} ({row.get('corpus_id', '?')}) missing fields: {missing}"

        assert row["run_date"] == "20260413", f"Row {i} wrong run_date: {row['run_date']!r}"
        assert row["experiment"] == 252, f"Row {i} wrong experiment: {row['experiment']}"
        assert row["domain"] in VALID_DOMAINS, f"Row {i} invalid domain: {row['domain']!r}"
        assert row["outcome_label"] in VALID_OUTCOME_LABELS, (
            f"Row {i} invalid outcome_label: {row['outcome_label']!r}"
        )
        assert row["verifier_outcome"] in VALID_VERIFIER_OUTCOMES, (
            f"Row {i} invalid verifier_outcome: {row['verifier_outcome']!r}"
        )
        assert row["downstream_repair_outcome"] in VALID_REPAIR_OUTCOMES, (
            f"Row {i} invalid downstream_repair_outcome: {row['downstream_repair_outcome']!r}"
        )
        assert isinstance(row["memory_hit"], bool), f"Row {i} memory_hit must be bool"
        assert isinstance(row["memory_match_metadata"], dict), (
            f"Row {i} memory_match_metadata must be dict"
        )
        assert isinstance(row["violation_family"], list), f"Row {i} violation_family must be list"
        assert isinstance(row["corpus_id"], str) and row["corpus_id"], (
            f"Row {i} corpus_id must be non-empty string"
        )
        assert isinstance(row["case_id"], str) and row["case_id"], (
            f"Row {i} case_id must be non-empty string"
        )
        assert isinstance(row["sample_position"], int), (
            f"Row {i} sample_position must be int"
        )
        assert isinstance(row["provenance"], dict), f"Row {i} provenance must be dict"

        # provenance must carry source tracing fields
        prov = row["provenance"]
        for pf in ("source_experiment", "source_artifact", "model", "benchmark", "case_id"):
            assert pf in prov, f"Row {i} provenance missing field: {pf!r}"

        # confidence must be float or null
        conf = row["confidence"]
        assert conf is None or isinstance(conf, float), (
            f"Row {i} confidence must be float or null, got {type(conf)}"
        )

        # latency fields must be float or null
        for lat_field in ("baseline_latency_seconds", "repair_latency_seconds"):
            lat = row[lat_field]
            assert lat is None or isinstance(lat, (int, float)), (
                f"Row {i} {lat_field} must be numeric or null"
            )


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-252-B: deterministic generation
# ---------------------------------------------------------------------------


def test_deterministic_generation(tmp_path: Path):
    """Running build_and_write twice produces byte-identical output."""
    module = load_module()
    repo_root = Path(__file__).resolve().parents[2]

    out1 = tmp_path / "run1" / "corpus.jsonl"
    summary1 = tmp_path / "run1" / "summary.json"
    out2 = tmp_path / "run2" / "corpus.jsonl"
    summary2 = tmp_path / "run2" / "summary.json"

    module.build_and_write(
        repo_root=repo_root,
        corpus_path=out1,
        summary_path=summary1,
    )
    module.build_and_write(
        repo_root=repo_root,
        corpus_path=out2,
        summary_path=summary2,
    )

    rows1 = read_jsonl(out1)
    rows2 = read_jsonl(out2)
    assert len(rows1) == len(rows2), f"Row counts differ: {len(rows1)} vs {len(rows2)}"
    for i, (r1, r2) in enumerate(zip(rows1, rows2)):
        assert r1 == r2, (
            f"Row {i} differs between runs:\n  run1: {r1!r}\n  run2: {r2!r}"
        )

    s1 = json.loads(summary1.read_text())
    s2 = json.loads(summary2.read_text())
    assert s1 == s2, "Summaries differ between runs"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-252-C: provenance completeness — all source experiments covered
# ---------------------------------------------------------------------------


def test_provenance_covers_all_source_experiments():
    """Source experiments 235, 238, 241 must each appear in the corpus provenance."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "predictive_verification_corpus_252.jsonl"
    rows = read_jsonl(corpus_path)
    found = {row["source_experiment"] for row in rows}
    required = {235, 238, 241}
    missing = required - found
    assert not missing, (
        f"Corpus missing entries from source experiments: {missing}\n"
        f"Found: {found}"
    )


def test_provenance_covers_all_source_artifacts():
    """Each row's provenance.source_artifact must point to a real file in the repo."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "predictive_verification_corpus_252.jsonl"
    rows = read_jsonl(corpus_path)
    for i, row in enumerate(rows):
        artifact = row["provenance"]["source_artifact"]
        full_path = repo_root / artifact
        assert full_path.exists(), (
            f"Row {i} provenance.source_artifact does not exist: {artifact!r}"
        )


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-252-D: both semantic and code traces present
# ---------------------------------------------------------------------------


def test_semantic_and_code_traces_present():
    """The corpus must include both reasoning (semantic) and code domain traces."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "predictive_verification_corpus_252.jsonl"
    rows = read_jsonl(corpus_path)
    domains = {row["domain"] for row in rows}
    assert "reasoning" in domains, "No reasoning-domain rows found"
    assert "code" in domains, "No code-domain rows found"

    # at least 10 rows in each domain
    reasoning_count = sum(1 for r in rows if r["domain"] == "reasoning")
    code_count = sum(1 for r in rows if r["domain"] == "code")
    assert reasoning_count >= 10, f"Too few reasoning rows: {reasoning_count}"
    assert code_count >= 10, f"Too few code rows: {code_count}"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-252-E: memory-hit metadata present on held-out decisions
# ---------------------------------------------------------------------------


def test_memory_hit_metadata_present_on_held_out():
    """Rows from Exp 241 held-out decisions must have non-empty memory_match_metadata."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "predictive_verification_corpus_252.jsonl"
    rows = read_jsonl(corpus_path)
    held_out_rows = [r for r in rows if r["source_experiment"] == 241]
    assert len(held_out_rows) > 0, "No rows with source_experiment=241 found"
    for i, row in enumerate(held_out_rows):
        mm = row["memory_match_metadata"]
        assert isinstance(mm, dict), f"Held-out row {i} memory_match_metadata must be dict"
        # must contain at least the candidate keys field
        assert "candidate_case_keys" in mm, (
            f"Held-out row {i} memory_match_metadata missing candidate_case_keys"
        )
        assert "matched_case_keys" in mm, (
            f"Held-out row {i} memory_match_metadata missing matched_case_keys"
        )


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-252-F: accepted_repair populated for successful repairs
# ---------------------------------------------------------------------------


def test_accepted_repair_populated_for_successes():
    """Rows with downstream_repair_outcome='accepted' must have non-null accepted_repair."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "predictive_verification_corpus_252.jsonl"
    rows = read_jsonl(corpus_path)
    accepted = [r for r in rows if r["downstream_repair_outcome"] == "accepted"]
    assert len(accepted) > 0, "No rows with downstream_repair_outcome='accepted' found"
    for i, row in enumerate(accepted):
        assert row["accepted_repair"] is not None, (
            f"Accepted-repair row {i} ({row['corpus_id']}) has null accepted_repair"
        )
        assert isinstance(row["accepted_repair"], str) and row["accepted_repair"].strip(), (
            f"Accepted-repair row {i} has empty accepted_repair string"
        )


# ---------------------------------------------------------------------------
# Unit tests for pure helper functions
# ---------------------------------------------------------------------------


def test_corpus_id_uniqueness():
    """All corpus_id values must be unique within the corpus."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "predictive_verification_corpus_252.jsonl"
    rows = read_jsonl(corpus_path)
    ids = [r["corpus_id"] for r in rows]
    assert len(ids) == len(set(ids)), (
        f"Duplicate corpus_ids found: {len(ids) - len(set(ids))} duplicates"
    )


def test_violation_family_matches_domain():
    """Code rows must have code-specific violation families; semantic rows must not."""
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "data" / "research" / "predictive_verification_corpus_252.jsonl"
    rows = read_jsonl(corpus_path)
    code_families = {"syntax_error", "humaneval_failure", "deterministic", "no_exception",
                     "syntax", "pbt_failure", "spec_failure"}
    semantic_families = {"question_grounding_failures", "answer_target_mismatch",
                         "missing_entity_coverage", "unit_aggregation_errors",
                         "missing_quantity_coverage", "omitted_premises",
                         "entity_quantity_binding_errors"}
    for i, row in enumerate(rows):
        vf_set = set(row["violation_family"])
        if row["domain"] == "code":
            # code rows may have empty violation_family (clean) or code families
            cross = vf_set & semantic_families
            assert not cross, (
                f"Code row {i} has semantic violation families: {cross}"
            )
        elif row["domain"] == "reasoning":
            cross = vf_set & code_families
            assert not cross, (
                f"Reasoning row {i} has code violation families: {cross}"
            )


def test_summary_artifact_label_counts():
    """The summary artifact must include label_counts and source_breakdown fields."""
    repo_root = Path(__file__).resolve().parents[2]
    summary_path = repo_root / "results" / "experiment_252_results.json"
    assert summary_path.exists(), f"Summary not found at {summary_path}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "label_counts" in summary, "Summary missing label_counts"
    assert "source_breakdown" in summary, "Summary missing source_breakdown"
    assert "total_records" in summary, "Summary missing total_records"
    assert summary["total_records"] >= 50, (
        f"Summary reports too few records: {summary['total_records']}"
    )
    # label_counts must break down by process_label and outcome_label
    lc = summary["label_counts"]
    assert "by_process_label" in lc, "label_counts missing by_process_label"
    assert "by_outcome_label" in lc, "label_counts missing by_outcome_label"
    assert "by_domain" in lc, "label_counts missing by_domain"
    assert "by_verifier_outcome" in lc, "label_counts missing by_verifier_outcome"
    assert "by_repair_outcome" in lc, "label_counts missing by_repair_outcome"
    # source_breakdown must reference known source experiments
    sb = summary["source_breakdown"]
    assert len(sb) >= 2, "source_breakdown must list at least 2 source experiments"
