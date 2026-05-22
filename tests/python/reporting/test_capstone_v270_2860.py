"""Test the Exp 2860 milestone .270 capstone synthesis.

References:
- REQ-REPORT-2860
- SCENARIO-REPORT-2860

These tests verify that the synthesis module:
  1. Classifies each upstream artifact as clean/blocked/skipped/missing/flagged.
  2. Emits a terminal-prefix honest_verdict.
  3. Refuses paper_ready=true until the cross-corpus matrix has FoVer plus at
     least one non-FoVer clean row and no headline row is adversarially flagged.
  4. Refuses to invent metric values for blocked or missing corpora.
  5. Surfaces the Exp 2849 vs Exp 2854 manifest-naming mismatch when present.
  6. Writes a valid JSON artifact with every required schema field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v270_2860 as cap


def _write(directory: Path, filename: str, data: dict[str, Any]) -> Path:
    p = directory / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_realistic_artifacts(tmp_path: Path) -> None:
    """Write a realistic set of .270 result files into ``tmp_path``.

    Mirrors the actual milestone state at the time the capstone was written:
    FoVer clean, BEAVER/EPR clean, SOTA runtime + HaluEval/FEVER blocked,
    MBPP/HumanEval/TruthfulQA/2856 missing, 2857 and 2859 blocked downstream.
    """
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)

    _write(
        results,
        "experiment_2847_archive_v269_activate_v270.json",
        {
            "honest_verdict": "complete: archive_ready=true",
            "duration_s": 1.2,
        },
    )
    _write(
        results,
        "experiment_2848_sota_runtime_evidence_v2.json",
        {
            "honest_verdict": "blocked_llama_cpp_gpu_offload: precondition c failed",
            "sota_runtime_ready_v2": False,
            "blocked_resources": ["llama_cpp_gpu_offload", "cached_sota_pair"],
            "duration_s": 1.59,
        },
    )
    _write(
        results,
        "experiment_2849_local_dataset_materialization_v1.json",
        {
            "honest_verdict": "complete: local benchmark manifests materialized",
            "fever_ready": True,
            "halueval_ready": True,
            "humaneval_ready": True,
            "mbpp_ready": True,
            "truthfulqa_ready": True,
            "manifest_counts": {
                "fever": 500,
                "halueval": 500,
                "humaneval": 164,
                "mbpp": 100,
                "truthfulqa": 200,
            },
            "manifest_paths": {
                "fever": "/abs/path/data/eval_manifests/fever_20260522.jsonl",
                "halueval": "/abs/path/data/eval_manifests/halueval_20260522.jsonl",
                "humaneval": "/abs/path/data/eval_manifests/humaneval_20260522.jsonl",
                "mbpp": "/abs/path/data/eval_manifests/mbpp_20260522.jsonl",
                "truthfulqa": "/abs/path/data/eval_manifests/truthfulqa_20260522.jsonl",
            },
            "synthetic_rows_created": False,
            "duration_s": 4.0,
        },
    )
    _write(
        results,
        "experiment_2850_fover_dual_condition_integrity_v4.json",
        {
            "honest_verdict": "complete: FoVer dual-condition integrity rerun measured",
            "condition_a_production_auroc_mean": 0.9131336,
            "condition_a_production_auroc_std": 0.0074942,
            "condition_b_architecture_only_auroc_mean": 0.8946624,
            "condition_b_architecture_only_auroc_std": 0.0075385,
            "learning_contribution": 0.01847,
            "n_examples": 1000,
            "n_seeds": 5,
            "live_model_invoked": False,
            "adversarial_verify_passed": True,
            "duration_s": 17.5,
        },
    )
    _write(
        results,
        "experiment_2854_halueval_fever_full_calibration_v2.json",
        {
            "honest_verdict": "blocked_missing_eval_manifests",
            "full_benchmark_ready": False,
            "live_model_invoked": False,
            "adversarial_verify_passed": False,
            "duration_s": 0.4,
        },
    )
    _write(
        results,
        "experiment_2855_cross_corpus_matrix_v4.json",
        {
            "honest_verdict": "complete: cross-corpus matrix not built; clean=1; blocked=1",
            "cross_corpus_matrix_built": False,
            "clean_corpus_count": 1,
            "blocked_corpus_count": 1,
            "flagged_corpus_count": 0,
            "missing_corpus_count": 3,
            "row_status_by_corpus": {
                "FoVer": "clean",
                "HaluEval/FEVER": "blocked",
                "HumanEval": "missing",
                "MBPP": "missing",
                "TruthfulQA": "missing",
            },
            "paper_eligible_rows": ["FoVer"],
            "duration_s": 0.05,
        },
    )
    _write(
        results,
        "experiment_2857_loopus_fr11_self_learning_v2.json",
        {
            "honest_verdict": "blocked_missing_exp2856_artifact",
            "fr11_self_learning_ready": False,
            "n_examples": 0,
            "max_loops": 0,
            "correctness_delta": 0.0,
            "energy_delta_mean": 0.0,
            "recurrence_success_rate": 0.0,
            "no_model_weight_mutation": True,
            "duration_s": 0.1,
        },
    )
    _write(
        results,
        "experiment_2858_beaver_epr_clean_bounded_proxy_v2.json",
        {
            "honest_verdict": "complete: clean bounded-prefix/EPR proxy evaluated",
            "beaver_exact": False,
            "entropy_production_measured": True,
            "bounded_prefix_proxy_auc": 0.74,
            "entropy_production_auc": 0.5811688,
            "n_examples": 100,
            "live_model_invoked": False,
            "adversarial_verify_passed": True,
            "duration_s": 0.3,
        },
    )
    _write(
        results,
        "experiment_2859_drift_mus_conflict_prioritizer.json",
        {
            "honest_verdict": "blocked_cross_corpus_matrix_not_built",
            "drift_mus_diagnostic_ready": False,
            "n_failure_rows": 0,
            "failure_class_counts": {},
            "hypergraph_nodes": 0,
            "hypergraph_hyperedges": 0,
            "hgnn_inspired_heuristic_name": "min_hitting_set_proxy",
            "baseline_random_checks_to_conflict": 0.0,
            "baseline_degree_checks_to_conflict": 0.0,
            "heuristic_checks_to_conflict": 0.0,
            "heuristic_improvement_vs_best_baseline": 0.0,
            "duration_s": 0.05,
        },
    )
    # exp2851, exp2852, exp2853, exp2856 are intentionally absent (missing).


# ---------------------------------------------------------------------------
# Unit tests for utility predicates
# ---------------------------------------------------------------------------


def test_is_terminal_success_complete_colon() -> None:
    """REQ-REPORT-2860: complete: prefix counts as terminal success."""
    assert cap.is_terminal_success("complete: something measured")


def test_is_terminal_success_complete_underscore() -> None:
    """REQ-REPORT-2860: complete_ underscore form counts as terminal success."""
    assert cap.is_terminal_success("complete_fover_measured")


def test_is_terminal_success_success_colon() -> None:
    assert cap.is_terminal_success("success: all preconditions passed")


def test_is_terminal_success_rejects_blocked() -> None:
    """REQ-REPORT-2860: blocked_ verdicts are not terminal success."""
    assert not cap.is_terminal_success("blocked_missing_manifests")


def test_is_terminal_success_rejects_non_string() -> None:
    """REQ-REPORT-2860: non-string inputs are not terminal success."""
    assert not cap.is_terminal_success(None)
    assert not cap.is_terminal_success(0)


def test_is_blocked_verdict_positive() -> None:
    assert cap.is_blocked_verdict("blocked_missing_manifests")


def test_is_blocked_verdict_negative() -> None:
    assert not cap.is_blocked_verdict("complete: measured")


def test_is_blocked_verdict_non_string() -> None:
    assert not cap.is_blocked_verdict(None)


def test_is_skipped_verdict_positive() -> None:
    assert cap.is_skipped_verdict("skipped_quota_exhausted")
    assert cap.is_skipped_verdict("retired_lineage_v3")


def test_is_skipped_verdict_negative() -> None:
    assert not cap.is_skipped_verdict("complete: measured")
    assert not cap.is_skipped_verdict(123)


def test_is_adversarially_flagged_via_flag() -> None:
    """REQ-REPORT-2860: explicit flagged_adversarial=true marks adversarial."""
    assert cap.is_adversarially_flagged({"flagged_adversarial": True})


def test_is_adversarially_flagged_via_corrigendum() -> None:
    """REQ-REPORT-2860: a non-empty corrigendum_pending also marks adversarial."""
    assert cap.is_adversarially_flagged({"corrigendum_pending": ["fix duration"]})


def test_is_adversarially_flagged_negative() -> None:
    """REQ-REPORT-2860: precondition-fail blocked verdicts are NOT adversarial."""
    payload = {
        "honest_verdict": "blocked_missing_manifests",
        "adversarial_verify_passed": False,
    }
    assert not cap.is_adversarially_flagged(payload)


def test_read_json_missing_file_returns_empty(tmp_path: Path) -> None:
    """REQ-REPORT-2860: a missing path returns an empty dict, never crashes."""
    assert cap.read_json(tmp_path / "nope.json") == {}


def test_read_json_malformed_file_returns_empty(tmp_path: Path) -> None:
    """REQ-REPORT-2860: malformed JSON returns an empty dict, never crashes."""
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert cap.read_json(p) == {}


def test_read_json_non_object_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert cap.read_json(p) == {}


def test_classify_artifact_missing_when_absent() -> None:
    assert cap.classify_artifact({}, present=False) == "missing"


def test_classify_artifact_missing_when_empty() -> None:
    assert cap.classify_artifact({}, present=True) == "missing"


def test_classify_artifact_blocked() -> None:
    assert cap.classify_artifact({"honest_verdict": "blocked_foo"}, present=True) == "blocked"


def test_classify_artifact_clean() -> None:
    assert (
        cap.classify_artifact({"honest_verdict": "complete: measured"}, present=True)
        == "clean"
    )


def test_classify_artifact_skipped() -> None:
    assert (
        cap.classify_artifact({"honest_verdict": "skipped_quota"}, present=True)
        == "skipped"
    )


def test_classify_artifact_adversarially_flagged_wins() -> None:
    """REQ-REPORT-2860: a flagged artifact is flagged regardless of verdict."""
    payload = {"honest_verdict": "complete: measured", "flagged_adversarial": True}
    assert cap.classify_artifact(payload, present=True) == "adversarially_flagged"


def test_classify_artifact_unknown_verdict_is_missing() -> None:
    """Non-terminal, non-blocked, non-skipped verdicts default to missing."""
    payload = {"honest_verdict": "partial_no_idea"}
    assert cap.classify_artifact(payload, present=True) == "missing"


# ---------------------------------------------------------------------------
# Integration tests over a realistic tmp_path snapshot of .270
# ---------------------------------------------------------------------------


def test_build_artifact_has_all_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2860: every required schema field is present."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    required = [
        "honest_verdict",
        "milestone",
        "paper_ready",
        "sota_runtime_ready_v2",
        "dataset_materialization_summary",
        "primary_corpus_results",
        "self_learning_summary",
        "beaver_epr_summary",
        "drift_mus_summary",
        "clean_artifacts",
        "blocked_artifacts",
        "skipped_artifacts",
        "missing_artifacts",
        "adversarially_flagged_artifacts",
        "headline_eligible_rows",
        "excluded_from_headline",
        "top_3_next_actions",
        "claim_boundary_notes",
        "duration_s",
        "run_date",
    ]
    missing = [f for f in required if f not in art]
    assert not missing, f"Missing required fields: {missing}"


def test_milestone_and_run_date_are_pinned(tmp_path: Path) -> None:
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert art["milestone"] == "2026.05.270"
    assert art["run_date"] == "20260522"


def test_honest_verdict_starts_with_terminal_prefix(tmp_path: Path) -> None:
    """REQ-REPORT-2860: the capstone's own verdict must be terminal."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert cap.is_terminal_success(art["honest_verdict"])


def test_paper_ready_is_false_when_matrix_not_built(tmp_path: Path) -> None:
    """REQ-REPORT-2860: paper_ready stays false unless the matrix is built."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert art["paper_ready"] is False


def test_sota_runtime_ready_v2_is_false_when_blocked(tmp_path: Path) -> None:
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert art["sota_runtime_ready_v2"] is False


def test_classification_lists_partition_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-2860: clean/blocked/skipped/missing/flagged partition the set."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    total = (
        len(art["clean_artifacts"])
        + len(art["blocked_artifacts"])
        + len(art["skipped_artifacts"])
        + len(art["missing_artifacts"])
        + len(art["adversarially_flagged_artifacts"])
    )
    assert total == len(cap.EXPECTED_ARTIFACTS)


def test_specific_classification(tmp_path: Path) -> None:
    """REQ-REPORT-2860: clean/blocked/missing assignments match the fixture state."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert "exp2850" in art["clean_artifacts"]
    assert "exp2858" in art["clean_artifacts"]
    assert "exp2848" in art["blocked_artifacts"]
    assert "exp2854" in art["blocked_artifacts"]
    assert "exp2857" in art["blocked_artifacts"]
    assert "exp2859" in art["blocked_artifacts"]
    assert "exp2851" in art["missing_artifacts"]
    assert "exp2852" in art["missing_artifacts"]
    assert "exp2853" in art["missing_artifacts"]
    assert "exp2856" in art["missing_artifacts"]
    assert art["adversarially_flagged_artifacts"] == []


def test_fover_row_carries_real_values(tmp_path: Path) -> None:
    """REQ-REPORT-2860: FoVer row echoes upstream metric fields, not invented ones."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    fover = art["primary_corpus_results"]["FoVer"]
    assert fover["status"] == "clean"
    assert fover["production_auroc_mean"] == pytest.approx(0.9131336, abs=1e-6)
    assert fover["architecture_only_auroc_mean"] == pytest.approx(0.8946624, abs=1e-6)
    assert fover["n_examples"] == 1000
    assert fover["n_seeds"] == 5
    assert fover["headline_eligible"] is True


def test_blocked_corpus_rows_have_null_metrics(tmp_path: Path) -> None:
    """REQ-REPORT-2860: blocked corpora must not carry fabricated AUROC values."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    halueval = art["primary_corpus_results"]["HaluEval"]
    assert halueval["status"] == "blocked"
    assert halueval["production_auroc_mean"] is None
    assert halueval["architecture_only_auroc_mean"] is None
    assert halueval["headline_eligible"] is False


def test_missing_corpus_rows_have_null_metrics(tmp_path: Path) -> None:
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    for corpus in ("MBPP", "HumanEval", "TruthfulQA"):
        row = art["primary_corpus_results"][corpus]
        assert row["status"] == "missing", corpus
        assert row["production_auroc_mean"] is None, corpus
        assert row["headline_eligible"] is False, corpus


def test_headline_eligible_rows_is_only_fover(tmp_path: Path) -> None:
    """REQ-REPORT-2860: only FoVer is paper-eligible at .270."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert art["headline_eligible_rows"] == ["FoVer"]


def test_excluded_from_headline_has_reason_per_corpus(tmp_path: Path) -> None:
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    excluded = art["excluded_from_headline"]
    assert excluded["HaluEval"] == "blocked_precondition"
    assert excluded["FEVER"] == "blocked_precondition"
    assert excluded["MBPP"] == "missing_artifact"
    assert excluded["HumanEval"] == "missing_artifact"
    assert excluded["TruthfulQA"] == "missing_artifact"


def test_self_learning_summary_reports_block(tmp_path: Path) -> None:
    """REQ-REPORT-2860: 2857 stays blocked when 2856 adapter is missing."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    self_l = art["self_learning_summary"]
    assert self_l["exp2856_status"] == "missing"
    assert self_l["exp2857_status"] == "blocked"
    assert self_l["fr11_self_learning_ready"] is False
    assert self_l["measured_improvement"] is False


def test_beaver_summary_records_proxy_not_exact(tmp_path: Path) -> None:
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    beaver = art["beaver_epr_summary"]
    assert beaver["status"] == "clean"
    assert beaver["beaver_exact"] is False
    assert beaver["bounded_prefix_proxy_auc"] == pytest.approx(0.74)
    assert beaver["headline_eligible"] is False


def test_drift_mus_summary_records_block(tmp_path: Path) -> None:
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    drift = art["drift_mus_summary"]
    assert drift["status"] == "blocked"
    assert drift["drift_mus_diagnostic_ready"] is False


def test_dataset_naming_mismatch_detected(tmp_path: Path) -> None:
    """REQ-REPORT-2860: date-suffixed manifest paths surface as a mismatch."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert art["dataset_naming_mismatch_detected"] is True


def test_dataset_naming_mismatch_not_detected_when_plain_paths(tmp_path: Path) -> None:
    """REQ-REPORT-2860: plain paths do not trigger the mismatch flag."""
    _make_realistic_artifacts(tmp_path)
    # Overwrite 2849 with plain paths.
    _write(
        tmp_path / "results",
        "experiment_2849_local_dataset_materialization_v1.json",
        {
            "honest_verdict": "complete: local benchmark manifests materialized",
            "fever_ready": True,
            "halueval_ready": True,
            "humaneval_ready": True,
            "mbpp_ready": True,
            "truthfulqa_ready": True,
            "manifest_counts": {"halueval": 500, "fever": 500},
            "manifest_paths": {
                "halueval": "/abs/path/data/eval_manifests/halueval.jsonl",
                "fever": "/abs/path/data/eval_manifests/fever.jsonl",
            },
        },
    )
    art = cap.build_artifact(tmp_path)
    assert art["dataset_naming_mismatch_detected"] is False


def test_top_3_next_actions_has_exactly_three_items(tmp_path: Path) -> None:
    """REQ-REPORT-2860: planner expects exactly three next actions."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert len(art["top_3_next_actions"]) == 3
    for action in art["top_3_next_actions"]:
        assert isinstance(action, str) and action.strip()


def test_top_3_next_actions_when_all_clean_still_three(tmp_path: Path) -> None:
    """REQ-REPORT-2860: even with no blockers the action list must have 3 slots."""
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    # Minimal clean state: SOTA ready, naming clean, nothing missing.
    _write(
        results,
        "experiment_2848_sota_runtime_evidence_v2.json",
        {"honest_verdict": "complete: sota runtime ready", "sota_runtime_ready_v2": True},
    )
    _write(
        results,
        "experiment_2849_local_dataset_materialization_v1.json",
        {
            "honest_verdict": "complete: materialized",
            "manifest_paths": {"halueval": "data/eval_manifests/halueval.jsonl"},
        },
    )
    # All other expected artifacts absent.
    art = cap.build_artifact(tmp_path)
    assert len(art["top_3_next_actions"]) == 3


def test_claim_boundary_notes_mentions_missing_corpora(tmp_path: Path) -> None:
    """REQ-REPORT-2860: missing corpora appear in claim_boundary_notes."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    blob = " ".join(art["claim_boundary_notes"])
    assert "MBPP" in blob
    assert "HumanEval" in blob
    assert "TruthfulQA" in blob


def test_duration_is_non_negative_float(tmp_path: Path) -> None:
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert isinstance(art["duration_s"], float)
    assert art["duration_s"] >= 0.0


def test_duration_with_explicit_times(tmp_path: Path) -> None:
    """REQ-REPORT-2860: explicit started_s/now_s control the reported duration."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path, started_s=0.0, now_s=2.5)
    assert art["duration_s"] == pytest.approx(2.5, abs=1e-6)


def test_duration_clamped_to_non_negative(tmp_path: Path) -> None:
    """REQ-REPORT-2860: a negative elapsed window is clamped to 0.0."""
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path, started_s=5.0, now_s=2.0)
    assert art["duration_s"] == 0.0


def test_write_artifact_creates_file(tmp_path: Path) -> None:
    """REQ-REPORT-2860: write_artifact produces a readable JSON deliverable."""
    _make_realistic_artifacts(tmp_path)
    out = cap.write_artifact(tmp_path)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["milestone"] == "2026.05.270"
    assert loaded["paper_ready"] is False
    assert loaded["headline_eligible_rows"] == ["FoVer"]
    assert cap.is_terminal_success(loaded["honest_verdict"])


def test_write_artifact_supports_absolute_output_path(tmp_path: Path) -> None:
    """REQ-REPORT-2860: absolute output_path is honored as-is."""
    _make_realistic_artifacts(tmp_path)
    target = tmp_path / "elsewhere" / "capstone.json"
    out = cap.write_artifact(tmp_path, output_path=target)
    assert out == target
    assert target.exists()


def test_paper_ready_true_when_matrix_built_and_two_clean_rows(tmp_path: Path) -> None:
    """REQ-REPORT-2860: paper_ready flips true once a non-FoVer row is clean."""
    _make_realistic_artifacts(tmp_path)
    results = tmp_path / "results"
    # Upgrade 2855 to "matrix built" and 2851 (MBPP) to clean.
    _write(
        results,
        "experiment_2855_cross_corpus_matrix_v4.json",
        {
            "honest_verdict": "complete: cross-corpus matrix built",
            "cross_corpus_matrix_built": True,
            "clean_corpus_count": 2,
            "blocked_corpus_count": 0,
            "flagged_corpus_count": 0,
            "missing_corpus_count": 0,
            "paper_eligible_rows": ["FoVer", "MBPP"],
        },
    )
    _write(
        results,
        "experiment_2851_mbpp_dual_condition_v4.json",
        {
            "honest_verdict": "complete: MBPP dual-condition measured",
            "condition_a_production_auroc_mean": 0.78,
            "condition_b_architecture_only_auroc_mean": 0.71,
            "learning_contribution": 0.07,
            "n_examples": 100,
            "n_seeds": 5,
        },
    )
    art = cap.build_artifact(tmp_path)
    assert art["paper_ready"] is True
    assert "MBPP" in art["headline_eligible_rows"]


def test_paper_ready_false_if_a_headline_row_is_flagged(tmp_path: Path) -> None:
    """REQ-REPORT-2860: an adversarial flag on any headline source blocks paper_ready."""
    _make_realistic_artifacts(tmp_path)
    results = tmp_path / "results"
    _write(
        results,
        "experiment_2855_cross_corpus_matrix_v4.json",
        {
            "honest_verdict": "complete: cross-corpus matrix built",
            "cross_corpus_matrix_built": True,
            "clean_corpus_count": 2,
            "blocked_corpus_count": 0,
            "flagged_corpus_count": 0,
            "missing_corpus_count": 0,
            "paper_eligible_rows": ["FoVer", "MBPP"],
        },
    )
    _write(
        results,
        "experiment_2851_mbpp_dual_condition_v4.json",
        {
            "honest_verdict": "complete: measured",
            "condition_a_production_auroc_mean": 0.78,
            "condition_b_architecture_only_auroc_mean": 0.71,
        },
    )
    # Now flag FoVer adversarially.
    _write(
        results,
        "experiment_2850_fover_dual_condition_integrity_v4.json",
        {
            "honest_verdict": "complete: FoVer measured",
            "condition_a_production_auroc_mean": 0.913,
            "condition_b_architecture_only_auroc_mean": 0.895,
            "flagged_adversarial": True,
        },
    )
    art = cap.build_artifact(tmp_path)
    assert art["paper_ready"] is False
    assert "exp2850" in art["adversarially_flagged_artifacts"]


def test_source_artifact_status_covers_every_expected_id(tmp_path: Path) -> None:
    _make_realistic_artifacts(tmp_path)
    art = cap.build_artifact(tmp_path)
    assert set(art["source_artifact_status"]) == set(cap.EXPECTED_ARTIFACTS)


def test_number_or_none_rejects_bool() -> None:
    """REQ-REPORT-2860: bool inputs are not numeric and must not coerce to float."""
    assert cap._number_or_none(True) is None
    assert cap._number_or_none(False) is None
    assert cap._number_or_none(1) == 1.0
    assert cap._number_or_none("0.5") is None


def test_dataset_naming_mismatch_with_non_dict_paths_returns_false() -> None:
    """REQ-REPORT-2860: defensive branch when manifest_paths is malformed."""
    assert cap._dataset_naming_mismatch({"manifest_paths": "not a dict"}) is False
    assert cap._dataset_naming_mismatch({}) is False
