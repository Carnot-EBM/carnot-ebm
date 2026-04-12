"""Spec: REQ-VERIFY-011, REQ-VERIFY-012, SCENARIO-VERIFY-011, SCENARIO-VERIFY-012."""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_211_constraint_ir_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_211_constraint_ir_benchmark", module_path
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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# REQ-VERIFY-011, SCENARIO-VERIFY-011
def test_build_live_gsm8k_examples_covers_curated_semantic_failures():
    module = load_module()

    examples = module.build_live_gsm8k_examples()

    assert len(examples) == 9
    assert {example["source_family"] for example in examples} == {"live_gsm8k_semantic_failure"}
    assert all(example["free_form_reasoning_monitorable"] is False for example in examples)

    lana = next(example for example in examples if example["example_id"] == "exp211-live-gsm8k-923")
    assert lana["expected_verifier_path"] == "question_grounding.quantity_graph"
    assert lana["expected_answer_schema"] == {"type": "number", "numeric_type": "integer"}
    assert lana["source_refs"] == ["exp203:923", "exp206:923", "exp207:923"]
    assert "semantic_grounding" in lana["constraint_types"]
    assert lana["gold_atomic_constraints"][0]["target"] == "chamomile_and_mint_total"


# REQ-VERIFY-011, SCENARIO-VERIFY-011
def test_instruction_and_code_examples_cover_required_modes():
    module = load_module()

    instruction_examples = module.build_instruction_examples()
    code_examples = module.build_code_examples()

    assert len(instruction_examples) == 36
    assert len(code_examples) == 36
    assert {example["source_family"] for example in instruction_examples} == {
        "instruction_following"
    }
    assert {example["source_family"] for example in code_examples} == {"code_typed_properties"}
    assert any(
        example["expected_answer_schema"]["type"] == "json_object"
        for example in instruction_examples
    )
    assert any(
        example["expected_answer_schema"]["type"] == "bullet_list"
        for example in instruction_examples
    )
    assert any(
        example["free_form_reasoning_monitorable"] is True for example in instruction_examples
    )
    assert all(
        example["expected_answer_schema"]["type"] == "python_function" for example in code_examples
    )
    assert all("typed_property" in example["constraint_types"] for example in code_examples)
    assert any(
        example["expected_verifier_path"] == "code_ir.typed_contracts_plus_execution"
        for example in code_examples
    )
    assert all(example["free_form_reasoning_monitorable"] is False for example in code_examples)


# REQ-VERIFY-011, REQ-VERIFY-012, SCENARIO-VERIFY-011
def test_build_benchmark_and_results_summary_match_expected_counts():
    module = load_module()

    examples = module.build_benchmark()
    results = module.build_results(examples)

    assert len(examples) == 81
    assert len({example["example_id"] for example in examples}) == 81
    assert results["experiment"] == "Exp 211"
    assert results["run_date"] == "20260412"
    assert results["summary"]["n_examples"] == 81
    assert results["summary"]["by_source_family"] == {
        "live_gsm8k_semantic_failure": 9,
        "instruction_following": 36,
        "code_typed_properties": 36,
    }
    assert results["summary"]["coverage_checks"]["example_count_in_range"] is True
    assert results["summary"]["coverage_checks"]["has_literal_constraints"] is True
    assert results["summary"]["coverage_checks"]["has_compositional_constraints"] is True
    assert results["summary"]["coverage_checks"]["has_semantic_grounding_constraints"] is True


# REQ-VERIFY-012, SCENARIO-VERIFY-012
def test_write_jsonl_writes_one_json_object_per_line(tmp_path: Path):
    module = load_module()
    path = tmp_path / "nested" / "benchmark.jsonl"

    module.write_jsonl(
        path,
        [
            {"example_id": "exp211-a", "prompt": "A"},
            {"example_id": "exp211-b", "prompt": "B"},
        ],
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["example_id"] == "exp211-a"
    assert json.loads(lines[1])["prompt"] == "B"


# REQ-VERIFY-011, REQ-VERIFY-012, SCENARIO-VERIFY-011, SCENARIO-VERIFY-012
def test_main_writes_benchmark_and_summary_idempotently(tmp_path: Path, monkeypatch):
    module = load_module()
    repo = make_repo(tmp_path)

    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        module,
        "BENCHMARK_PATH",
        repo / "data" / "research" / "constraint_ir_benchmark_211.jsonl",
    )
    monkeypatch.setattr(module, "RESULTS_PATH", repo / "results" / "experiment_211_results.json")

    assert module.main() == 0
    assert module.main() == 0

    benchmark = read_jsonl(repo / "data" / "research" / "constraint_ir_benchmark_211.jsonl")
    results = json.loads(
        (repo / "results" / "experiment_211_results.json").read_text(encoding="utf-8")
    )

    assert len(benchmark) == 81
    assert benchmark[0]["example_id"] == "exp211-live-gsm8k-923"
    assert results["summary"]["n_examples"] == 81
    assert results["run_date"] == "20260412"


# REQ-VERIFY-012, SCENARIO-VERIFY-012
def test_cli_entrypoint_honors_repo_override(tmp_path: Path, monkeypatch):
    repo = make_repo(tmp_path)
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "experiment_211_constraint_ir_benchmark.py"
    )

    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.setattr("sys.argv", [str(module_path)])

    try:
        runpy.run_path(str(module_path), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    benchmark_path = repo / "data" / "research" / "constraint_ir_benchmark_211.jsonl"
    results_path = repo / "results" / "experiment_211_results.json"
    assert benchmark_path.exists()
    assert results_path.exists()
    assert os.environ["CARNOT_REPO_ROOT"] == str(repo)
