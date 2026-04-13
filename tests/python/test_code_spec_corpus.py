"""Spec: REQ-CODE-023, REQ-CODE-024, SCENARIO-CODE-020, SCENARIO-CODE-021."""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

import pytest

from carnot.pipeline import code_spec_corpus as module


def load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_236_code_spec_corpus.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_236_code_spec_corpus",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def write_source_fixtures(repo: Path) -> None:
    source_root = Path(__file__).resolve().parents[2] / "results"
    target_root = repo / "results"
    target_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "experiment_226_results.json",
        "experiment_227_results.json",
    ):
        (target_root / name).write_text(
            (source_root / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# REQ-CODE-023, SCENARIO-CODE-020
def test_build_corpus_is_deterministic_and_uses_expected_row_schema() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    rows = module.build_corpus(repo_root)
    rerun = module.build_corpus(repo_root)

    assert rows == rerun
    assert len(rows) == 164
    assert rows[0]["row_id"] == "exp236-humaneval-0"
    assert rows[-1]["row_id"] == "exp236-humaneval-163"

    first = rows[0]
    assert first["schema_version"] == "carnot.code_spec_corpus.v1"
    assert first["run_date"] == "20260413"
    assert first["task_id"] == "HumanEval/0"
    assert first["case_id"] == "humaneval-0"
    assert first["entry_point"] == "has_close_elements"
    assert first["signature"].startswith("has_close_elements(")
    assert first["preconditions"]
    assert first["postconditions"]
    assert first["invariants"]
    assert first["oracle_hints"]
    assert isinstance(first["mutation_constraints"], list)
    assert isinstance(first["source_traces"], list)
    assert first["trace_summary"]["source_trace_count"] == 1
    assert first["trace_summary"]["source_refs"] == ["exp226:humaneval-0"]
    assert first["trace_summary"]["artifacts"] == ["results/experiment_226_results.json"]

    for family in module.SPEC_FAMILIES:
        assert all(
            sorted(item) == ["kind", "sources", "text", "trace_refs"] for item in first[family]
        )


# REQ-CODE-023, REQ-CODE-024, SCENARIO-CODE-021
def test_shared_trace_rows_preserve_multi_artifact_provenance_and_miss_counts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rows = module.build_corpus(repo_root)

    row = next(item for item in rows if item["task_id"] == "HumanEval/29")

    assert row["case_id"] == "humaneval-29"
    assert row["trace_summary"]["source_trace_count"] == 2
    assert row["trace_summary"]["official_test_miss_trace_count"] == 2
    assert row["trace_summary"]["repaired_trace_count"] == 0
    assert row["trace_summary"]["source_refs"] == [
        "exp226:humaneval-29",
        "exp227:humaneval-29",
    ]
    assert row["trace_summary"]["artifacts"] == [
        "results/experiment_226_results.json",
        "results/experiment_227_results.json",
    ]
    assert {trace["experiment"] for trace in row["source_traces"]} == {226, 227}
    assert all("annotated_return_type" in trace["failure_properties"] for trace in row["source_traces"])


# REQ-CODE-023, REQ-CODE-024
def test_repaired_rows_keep_trace_backed_repair_evidence() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rows = module.build_corpus(repo_root)

    row = next(item for item in rows if item["task_id"] == "HumanEval/66")

    assert row["trace_summary"]["source_trace_count"] == 1
    assert row["trace_summary"]["repaired_trace_count"] == 1
    assert row["trace_summary"]["source_refs"] == ["exp226:humaneval-66"]
    assert row["source_traces"][0]["repaired"] is True
    assert row["source_traces"][0]["repair_iterations"] == 1
    assert row["source_traces"][0]["failure_properties"] == [
        "deterministic",
        "no_exception",
    ]
    assert any(
        "exp226:humaneval-66" in item["trace_refs"] for item in row["invariants"] + row["postconditions"]
    )


# REQ-CODE-024, SCENARIO-CODE-020, SCENARIO-CODE-021
def test_build_results_reports_family_and_source_trace_counts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rows = module.build_corpus(repo_root)

    results = module.build_results(rows)
    by_family = {
        family: sum(len(row[family]) for row in rows)
        for family in module.SPEC_FAMILIES
    }

    assert results["experiment"] == "Exp 236"
    assert results["run_date"] == "20260413"
    assert results["summary"]["n_rows"] == 164
    assert results["summary"]["n_source_traces"] == 194
    assert results["summary"]["n_rows_with_multi_trace_provenance"] == 30
    assert results["summary"]["n_rows_with_official_test_miss"] == 6
    assert results["summary"]["n_official_test_miss_traces"] == 8
    assert results["summary"]["n_rows_with_repairs"] == 5
    assert results["summary"]["n_repaired_traces"] == 5
    assert results["summary"]["by_source_trace"] == {
        "results/experiment_226_results.json": 164,
        "results/experiment_227_results.json": 30,
    }
    assert results["summary"]["by_spec_family"] == by_family


# REQ-CODE-024
def test_helper_branches_cover_json_validation_and_jsonl_writing(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        module.load_json(bad_json)

    path = tmp_path / "nested" / "code_spec_corpus_236.jsonl"
    module.write_jsonl(
        path,
        [
            {"row_id": "exp236-a", "task_id": "HumanEval/A"},
            {"row_id": "exp236-b", "task_id": "HumanEval/B"},
        ],
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["row_id"] == "exp236-a"
    assert json.loads(lines[1])["task_id"] == "HumanEval/B"


# REQ-CODE-023, REQ-CODE-024
def test_helper_edge_branches_cover_fallback_paths_sparse_payloads_and_trace_ignores(
    tmp_path: Path,
) -> None:
    outside = Path("/tmp/exp236-outside.json")

    assert module._cli_default_path(outside) == outside
    assert module._display_path(tmp_path, outside) == str(outside)
    assert module._cohort_lookup({}) == {}
    assert module._find_function_node("def other():\n    return 1\n", "wanted") is None
    assert module._extract_signature_data("def broken(", "broken") == ("broken(...)", [], None)

    signature, params, return_annotation = module._extract_signature_data(
        "def f(self, x: int) -> int:\n    return x\n",
        "f",
    )
    assert signature == "f(x: int) -> int"
    assert params == [("x", "int", "int")]
    assert return_annotation == "int"

    families, zero_sig = module._base_family_map(
        prompt="def zero() -> int:\n    return 0\n",
        official_tests="",
        entry_point="zero",
    )
    assert zero_sig == "zero() -> int"
    assert families["preconditions"][
        ("declared_arity", "call uses the declared zero-argument input contract")
    ]["sources"] == ["signature"]

    trace_families = module._empty_family_map()
    module._apply_trace_support(
        trace_families,
        [
            {
                "failure_properties": ["unknown_property"],
                "official_test_miss": False,
                "repaired": False,
                "repair_iterations": 0,
                "source_ref": "exp999:case-x",
            }
        ],
    )
    assert trace_families["oracle_hints"] == {}

    repo = make_repo(tmp_path)
    results_dir = repo / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_226_results.json").write_text(
        json.dumps(
            {
                "experiment": 226,
                "metadata": {"model_name": "test-model"},
                "cohort": {"cases": []},
                "per_problem_results": "bad",
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "experiment_227_results.json").write_text(
        json.dumps(
            {
                "experiment": 227,
                "metadata": {"model_name": "test-model"},
                "cohort": {
                    "cases": [
                        {
                            "case_id": "case-a",
                            "dataset_idx": 0,
                            "task_id": "",
                            "prompt": "def zero() -> int:\n    return 0\n",
                            "test": "",
                            "entry_point": "zero",
                        }
                    ]
                },
                "per_problem_results": [
                    "bad",
                    {"case_id": "missing", "task_id": "HumanEval/missing"},
                    {"case_id": "case-a", "task_id": ""},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert module.build_corpus(repo) == []


# REQ-CODE-023, REQ-CODE-024, SCENARIO-CODE-020
def test_main_writes_corpus_and_summary_idempotently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path)
    write_source_fixtures(repo)

    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        module,
        "CORPUS_PATH",
        repo / "data" / "research" / "code_spec_corpus_236.jsonl",
    )
    monkeypatch.setattr(module, "RESULTS_PATH", repo / "results" / "experiment_236_results.json")

    assert module.main([]) == 0
    assert module.main([]) == 0

    corpus = read_jsonl(repo / "data" / "research" / "code_spec_corpus_236.jsonl")
    results = json.loads((repo / "results" / "experiment_236_results.json").read_text(encoding="utf-8"))

    assert len(corpus) == 164
    assert corpus[0]["row_id"] == "exp236-humaneval-0"
    assert corpus[-1]["row_id"] == "exp236-humaneval-163"
    assert results["summary"]["n_rows"] == 164
    assert results["summary"]["by_source_trace"]["results/experiment_227_results.json"] == 30


# REQ-CODE-023, SCENARIO-CODE-020
def test_script_entrypoint_honors_repo_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path)
    write_source_fixtures(repo)
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "experiment_236_code_spec_corpus.py"
    )

    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    argv_before = sys.argv[:]
    try:
        sys.argv = [str(script_path)]
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = argv_before
        monkeypatch.delenv("CARNOT_REPO_ROOT", raising=False)

    assert excinfo.value.code == 0
    assert (repo / "data" / "research" / "code_spec_corpus_236.jsonl").exists()
    results = json.loads((repo / "results" / "experiment_236_results.json").read_text(encoding="utf-8"))
    assert results["run_date"] == "20260413"


# REQ-CODE-023
def test_script_module_reexports_main() -> None:
    loaded = load_script_module()

    assert callable(loaded.main)
