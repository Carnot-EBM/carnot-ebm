"""Spec: REQ-VERIFY-018, REQ-VERIFY-019, SCENARIO-VERIFY-018, SCENARIO-VERIFY-019."""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_214_semantic_failure_corpus.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_214_semantic_failure_corpus", module_path
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
        "experiment_203_results.json",
        "experiment_206_results.json",
        "experiment_207_results.json",
    ):
        (target_root / name).write_text(
            (source_root / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# REQ-VERIFY-018, SCENARIO-VERIFY-018
def test_build_live_trace_examples_extracts_curated_unique_failures():
    module = load_module()

    examples = module.build_live_trace_examples()

    assert len(examples) == 8
    assert {example["source_type"] for example in examples} == {"live_trace"}
    assert {example["source_artifact"] for example in examples} == {"exp203_live", "exp206_live"}

    lana = next(example for example in examples if example["example_id"] == "exp214-live-923")
    assert lana["gold_diagnosis"]["taxonomy_label"] == "omitted_premises"
    assert lana["expected_verifier_signal"]["verifier_path"] == "question_grounding.quantity_graph"
    assert lana["source_refs"] == ["exp203:923", "exp206:923", "exp207:923"]
    assert "27 cups" in lana["prompt"]
    assert "27 / 3 = 9" in lana["response"]


# REQ-VERIFY-018, SCENARIO-VERIFY-018
def test_build_targeted_follow_up_examples_cover_taxonomy_and_exp208_code_slice():
    module = load_module()

    examples = module.build_targeted_follow_up_examples()

    assert len(examples) == 52
    assert {example["source_type"] for example in examples} == {"targeted_follow_up"}
    assert "exp208_followup" in {example["source_artifact"] for example in examples}
    assert {example["gold_diagnosis"]["taxonomy_label"] for example in examples} == {
        "question_grounding_failures",
        "omitted_premises",
        "entity_quantity_binding_errors",
        "unit_aggregation_errors",
        "genuine_arithmetic_slips",
        "code_specific_oracle_property_misses",
    }

    code_case = next(
        example for example in examples if example["example_id"] == "exp214-followup-code-dedupe-1"
    )
    assert code_case["domain"] == "code"
    assert code_case["gold_diagnosis"]["taxonomy_label"] == "code_specific_oracle_property_misses"
    assert code_case["expected_verifier_signal"]["verifier_path"] == (
        "code_ir.typed_contracts_plus_execution"
    )
    assert code_case["source_refs"] == ["exp208:HumanEval/26"]
    assert "sorted(set(items))" in code_case["response"]


# REQ-VERIFY-018, REQ-VERIFY-019, SCENARIO-VERIFY-018
def test_build_corpus_and_results_summary_match_expected_counts():
    module = load_module()

    examples = module.build_corpus()
    results = module.build_results(examples)

    assert len(examples) == 60
    assert len({example["example_id"] for example in examples}) == 60
    assert results["experiment"] == "Exp 214"
    assert results["run_date"] == "20260412"
    assert results["summary"]["n_examples"] == 60
    assert results["summary"]["by_source_type"] == {
        "live_trace": 8,
        "targeted_follow_up": 52,
    }
    assert results["summary"]["by_domain"] == {"word_problem": 50, "code": 10}
    assert results["summary"]["by_taxonomy"] == {
        "question_grounding_failures": 10,
        "omitted_premises": 10,
        "entity_quantity_binding_errors": 10,
        "unit_aggregation_errors": 10,
        "genuine_arithmetic_slips": 10,
        "code_specific_oracle_property_misses": 10,
    }
    assert results["summary"]["coverage_checks"]["min_example_count_met"] is True
    assert results["summary"]["coverage_checks"]["has_live_trace_examples"] is True
    assert results["summary"]["coverage_checks"]["has_targeted_follow_up_examples"] is True
    assert results["summary"]["coverage_checks"]["covers_required_taxonomy"] is True
    assert results["summary"]["coverage_checks"]["has_exp208_informed_code_followups"] is True


# REQ-VERIFY-019, SCENARIO-VERIFY-019
def test_write_jsonl_writes_one_json_object_per_line(tmp_path: Path):
    module = load_module()
    path = tmp_path / "nested" / "semantic_failure_corpus_214.jsonl"

    module.write_jsonl(
        path,
        [
            {"example_id": "exp214-a", "prompt": "A"},
            {"example_id": "exp214-b", "prompt": "B"},
        ],
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["example_id"] == "exp214-a"
    assert json.loads(lines[1])["prompt"] == "B"


# REQ-VERIFY-018, REQ-VERIFY-019, SCENARIO-VERIFY-018, SCENARIO-VERIFY-019
def test_main_writes_corpus_and_summary_idempotently(tmp_path: Path, monkeypatch):
    module = load_module()
    repo = make_repo(tmp_path)
    write_live_source_fixtures(repo)

    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        module,
        "CORPUS_PATH",
        repo / "data" / "research" / "semantic_failure_corpus_214.jsonl",
    )
    monkeypatch.setattr(module, "RESULTS_PATH", repo / "results" / "experiment_214_results.json")

    assert module.main() == 0
    assert module.main() == 0

    corpus = read_jsonl(repo / "data" / "research" / "semantic_failure_corpus_214.jsonl")
    results = json.loads(
        (repo / "results" / "experiment_214_results.json").read_text(encoding="utf-8")
    )

    assert len(corpus) == 60
    assert corpus[0]["example_id"] == "exp214-live-923"
    assert results["summary"]["n_examples"] == 60
    assert results["run_date"] == "20260412"


# REQ-VERIFY-019, SCENARIO-VERIFY-019
def test_cli_entrypoint_honors_repo_override(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path)
    write_live_source_fixtures(repo)
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "experiment_214_semantic_failure_corpus.py"
    )

    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.setattr("sys.argv", [str(module_path)])

    try:
        runpy.run_path(str(module_path), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    corpus_path = repo / "data" / "research" / "semantic_failure_corpus_214.jsonl"
    results_path = repo / "results" / "experiment_214_results.json"
    assert corpus_path.exists()
    assert results_path.exists()
    assert os.environ["CARNOT_REPO_ROOT"] == str(repo)
