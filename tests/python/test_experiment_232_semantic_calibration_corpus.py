"""Spec: REQ-VERIFY-042, REQ-VERIFY-043, SCENARIO-VERIFY-043, SCENARIO-VERIFY-044."""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_232_semantic_calibration_corpus.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_232_semantic_calibration_corpus",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def write_live_source_fixtures(repo: Path) -> None:
    source_root = Path(__file__).resolve().parents[2] / "results"
    target_root = repo / "results"
    target_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "experiment_219_results.json",
        "experiment_221_results.json",
    ):
        (target_root / name).write_text(
            (source_root / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# REQ-VERIFY-042, SCENARIO-VERIFY-043
def test_build_live_calibration_rows_preserve_live_outcomes_and_provenance(tmp_path: Path):
    module = load_module()
    repo = make_repo(tmp_path)
    write_live_source_fixtures(repo)

    rows = module.build_live_calibration_rows(repo)

    assert len(rows) == 562
    assert {row["source_type"] for row in rows} == {"live_artifact"}
    assert {row["source_artifact"] for row in rows} == {"exp219_live", "exp221_live"}

    semantic_rows = [row for row in rows if row["source_artifact"] == "exp219_live"]
    prompt_rows = [row for row in rows if row["source_artifact"] == "exp221_live"]
    assert len(semantic_rows) == 400
    assert len(prompt_rows) == 162
    assert {row["labels"]["outcome_bucket"] for row in semantic_rows} == {
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    }
    assert {row["labels"]["outcome_bucket"] for row in prompt_rows} == {
        "true_positive",
        "true_negative",
    }

    qwen_tp = next(
        row
        for row in rows
        if row["example_id"] == "exp232-live-219-qwen3-5-0-8b-gsm8k-249"
    )
    assert qwen_tp["labels"]["actual_error"] is True
    assert qwen_tp["labels"]["verifier_detected"] is True
    assert qwen_tp["labels"]["outcome_bucket"] == "true_positive"
    assert qwen_tp["labels"]["gold_violation_family"] == "question_grounding_failures"
    assert qwen_tp["labels"]["answer_target_alignment"] == "misaligned"
    assert qwen_tp["provenance"]["source_experiment"] == 219
    assert qwen_tp["provenance"]["source_case_id"] == "gsm8k-249"
    assert qwen_tp["source_refs"] == ["exp219:gsm8k-249"]
    assert qwen_tp["calibration"]["score"] > 0.5

    prompt_tn = next(
        row
        for row in rows
        if row["example_id"] == "exp232-live-221-qwen3-5-0-8b-exp211-instruction-grounded-3"
    )
    assert prompt_tn["labels"]["outcome_bucket"] == "true_negative"
    assert prompt_tn["labels"]["claim_granularity"] == "constraint_bundle"
    assert prompt_tn["labels"]["repairability_hint"] == "no_repair_needed"
    assert prompt_tn["provenance"]["source_experiment"] == 221
    assert prompt_tn["source_refs"] == ["vifbench-inspired", "constraintbench-inspired"]


# REQ-VERIFY-042, SCENARIO-VERIFY-044
def test_build_follow_up_rows_only_fill_missing_prompt_side_buckets(tmp_path: Path):
    module = load_module()
    repo = make_repo(tmp_path)
    write_live_source_fixtures(repo)

    rows = module.build_follow_up_rows(repo)

    assert len(rows) == 6
    assert {row["source_type"] for row in rows} == {"targeted_follow_up"}
    assert {row["source_artifact"] for row in rows} == {"exp232_followup"}
    assert {row["benchmark"] for row in rows} == {"constraint_ir"}
    assert {row["labels"]["outcome_bucket"] for row in rows} == {
        "false_positive",
        "false_negative",
    }
    assert {row["task_slice"] for row in rows} == {
        "code_typed_properties",
        "instruction_surface_only",
        "instruction_grounded",
    }
    assert all(row["provenance"]["follow_up_gap"] in {"false_positive", "false_negative"} for row in rows)
    assert all(str(row["source_refs"][0]).startswith("exp221:") for row in rows)

    code_fn = next(
        row
        for row in rows
        if row["example_id"] == "exp232-followup-code-typed-properties-fn-1"
    )
    assert code_fn["labels"]["actual_error"] is True
    assert code_fn["labels"]["verifier_detected"] is False
    assert code_fn["labels"]["repairability_hint"] == "detect_before_repair"
    assert "sorted(set(items))" in code_fn["response"]


# REQ-VERIFY-042, REQ-VERIFY-043, SCENARIO-VERIFY-043, SCENARIO-VERIFY-044
def test_build_corpus_and_results_summary_match_expected_counts(tmp_path: Path):
    module = load_module()
    repo = make_repo(tmp_path)
    write_live_source_fixtures(repo)

    rows = module.build_corpus(repo)
    results = module.build_results(rows)

    assert len(rows) == 568
    assert len({row["example_id"] for row in rows}) == 568
    assert results["experiment"] == "Exp 232"
    assert results["run_date"] == "20260413"
    assert results["summary"]["n_examples"] == 568
    assert results["summary"]["by_source_type"] == {
        "live_artifact": 562,
        "targeted_follow_up": 6,
    }
    assert results["summary"]["by_source_artifact"] == {
        "exp219_live": 400,
        "exp221_live": 162,
        "exp232_followup": 6,
    }
    assert results["summary"]["by_benchmark"] == {
        "gsm8k_semantic": 400,
        "constraint_ir": 168,
    }
    assert results["summary"]["by_outcome_bucket"] == {
        "true_positive": 155,
        "false_positive": 33,
        "false_negative": 221,
        "true_negative": 159,
    }
    assert results["summary"]["coverage_checks"]["semantic_live_has_all_outcomes"] is True
    assert results["summary"]["coverage_checks"]["prompt_side_live_missing_outcomes"] == [
        "false_positive",
        "false_negative",
    ]
    assert results["summary"]["coverage_checks"]["follow_ups_fill_prompt_gaps_only"] is True
    assert results["summary"]["coverage_checks"]["has_threshold_score_fields"] is True


# REQ-VERIFY-042, SCENARIO-VERIFY-043
def test_write_jsonl_writes_one_json_object_per_line(tmp_path: Path):
    module = load_module()
    path = tmp_path / "nested" / "semantic_calibration_corpus_232.jsonl"

    module.write_jsonl(
        path,
        [
            {"example_id": "exp232-a", "prompt": "A"},
            {"example_id": "exp232-b", "prompt": "B"},
        ],
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["example_id"] == "exp232-a"
    assert json.loads(lines[1])["prompt"] == "B"


# REQ-VERIFY-042, REQ-VERIFY-043, SCENARIO-VERIFY-043
def test_helper_branches_cover_validation_and_label_edge_cases(tmp_path: Path):
    module = load_module()
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        module.load_json(bad_json)

    assert module.primary_221_family([{"status": "violated", "family": "custom"}]) == "custom"
    assert module.answer_target_alignment_219(True, "omitted_premises", []) == "partially_aligned"
    assert module.claim_granularity_219(
        {"typed_reasoning_parse_status": "fallback_text", "response_mode": "fallback_text"},
        {"claims": [{"claim_id": "c1"}]},
    ) == "final_answer_only"
    assert module.premise_coverage_221(
        actual_error=True,
        constraint_coverage=1.0,
        gold_family="semantic",
        semantic_violation_count=0,
    ) == "partial"


# REQ-VERIFY-042, REQ-VERIFY-043, SCENARIO-VERIFY-043, SCENARIO-VERIFY-044
def test_main_writes_corpus_and_summary_idempotently(tmp_path: Path, monkeypatch):
    module = load_module()
    repo = make_repo(tmp_path)
    write_live_source_fixtures(repo)

    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        module,
        "CORPUS_PATH",
        repo / "data" / "research" / "semantic_calibration_corpus_232.jsonl",
    )
    monkeypatch.setattr(module, "RESULTS_PATH", repo / "results" / "experiment_232_results.json")

    assert module.main([]) == 0
    assert module.main([]) == 0

    corpus = read_jsonl(repo / "data" / "research" / "semantic_calibration_corpus_232.jsonl")
    results = json.loads(
        (repo / "results" / "experiment_232_results.json").read_text(encoding="utf-8")
    )

    assert len(corpus) == 568
    assert corpus[0]["example_id"] == "exp232-live-219-qwen3-5-0-8b-gsm8k-178"
    assert results["summary"]["n_examples"] == 568
    assert results["run_date"] == "20260413"


# REQ-VERIFY-043, SCENARIO-VERIFY-044
def test_cli_entrypoint_honors_repo_override(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path)
    write_live_source_fixtures(repo)
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "experiment_232_semantic_calibration_corpus.py"
    )

    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.setattr("sys.argv", [str(module_path)])

    try:
        runpy.run_path(str(module_path), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    corpus_path = repo / "data" / "research" / "semantic_calibration_corpus_232.jsonl"
    results_path = repo / "results" / "experiment_232_results.json"
    assert corpus_path.exists()
    assert results_path.exists()
    assert os.environ["CARNOT_REPO_ROOT"] == str(repo)
