"""Tests for Exp 2911 code-hallucination taxonomy verifier.

Spec: REQ-CODE-2911, SCENARIO-CODE-2911.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import code_hallucination_taxonomy_verifier as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate(
    *,
    stable_id: str,
    candidate_index: int,
    code: str,
    passed: bool,
    error_type: str | None,
    corpus: str = "MBPP",
    model: str = "local/model-a",
    row_status: str | None = None,
) -> dict[str, Any]:
    raw_rel = (
        "results/raw/experiment_2910_sota_code_generation_corrigendum_v2/"
        f"{corpus.lower()}_{stable_id}_{candidate_index}.txt"
    )
    return {
        "candidate_index": candidate_index,
        "corpus": corpus,
        "error_message": "" if passed else (error_type or "failed"),
        "error_type": error_type,
        "executed": True,
        "extracted_code": code,
        "extraction_status": "python_fence",
        "extraction_success": True,
        "model_hf_id": model,
        "n_tests": 2,
        "passed": passed,
        "random_seed": 2910 + candidate_index,
        "raw_response": code,
        "raw_response_path": raw_rel,
        "row_status": row_status or ("candidate_passed" if passed else "candidate_failed"),
        "runtime_success": passed or error_type == "AssertionError",
        "stable_id": stable_id,
        "syntax_success": error_type != "SyntaxError",
        "timed_out": False,
    }


def _stage_upstream(tmp_path: Path) -> None:
    rows = [
        _candidate(
            stable_id="task-a",
            candidate_index=0,
            code="def solve(x):\n    return x + 1\n",
            passed=True,
            error_type=None,
        ),
        _candidate(
            stable_id="task-a",
            candidate_index=1,
            code="def solve(x):\n    return x\n",
            passed=False,
            error_type="AssertionError",
        ),
        _candidate(
            stable_id="task-b",
            candidate_index=0,
            code="def broken(:\n    return 1\n",
            passed=False,
            error_type="SyntaxError",
            row_status="candidate_syntax_failed",
        ),
        _candidate(
            stable_id="task-b",
            candidate_index=1,
            code=(
                "import json\n"
                "import not_a_real_codegen_module\n"
                "def helper(x):\n"
                "    return x\n"
                "def solve(items):\n"
                "    value = missing_value + 1\n"
                "    parsed = json.parse('{}')\n"
                "    return helper(1, 2) + len(items, 0) + value\n"
            ),
            passed=False,
            error_type="TypeError",
        ),
    ]
    for row in rows:
        raw_path = tmp_path / row["raw_response_path"]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(row["raw_response"], encoding="utf-8")

    _write_json(
        tmp_path / exp.UPSTREAM_CODEGEN_ARTIFACT,
        {
            "artifact": "experiment_2910_sota_code_generation_corrigendum_v2",
            "codegen_corrigendum_ready": True,
            "candidate_results": rows,
            "per_task_results": [
                {
                    "candidate_count": 2,
                    "corpus": "MBPP",
                    "pass_vector": [True, False],
                    "stable_id": "task-a",
                },
                {
                    "candidate_count": 2,
                    "corpus": "MBPP",
                    "pass_vector": [False, False],
                    "stable_id": "task-b",
                },
            ],
        },
    )


def test_req_code_2911_static_taxonomy_labels_are_orthogonal() -> None:
    """REQ-CODE-2911: one AST pass can assign multiple independent hallucination labels."""

    result = exp.classify_source(
        "import json\n"
        "import not_a_real_codegen_module\n"
        "def helper(x):\n"
        "    return x\n"
        "def solve(items):\n"
        "    value = missing_value + 1\n"
        "    parsed = json.parse('{}')\n"
        "    return helper(1, 2) + len(items, 0) + value\n"
    )

    assert result.syntax_success is True
    assert set(result.labels) >= {
        "invented_import",
        "undefined_name",
        "invented_attribute_or_method",
        "invalid_argument",
    }
    by_category = {finding["category"] for finding in result.findings}
    assert {
        "invented_import",
        "undefined_name",
        "invented_attribute_or_method",
        "invalid_argument",
    } <= by_category


def test_req_code_2911_static_helper_edges_cover_resolvable_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-2911: static helpers cover import-from, type, and signature edges."""

    source = (
        "from json import loads, parse\n"
        "from json import *\n"
        "from not_a_real_codegen_module import thing\n"
        "TOP = 1\n"
        "ANN: int = 2\n"
        "class Demo:\n"
        "    pass\n"
        "def helper(x):\n"
        "    return x\n"
        "def kwonly(x=1):\n"
        "    return x\n"
        "def variadic(x, *args, **kwargs):\n"
        "    return x\n"
        "def solve(items: list[int], text: str, *rest, **kw):\n"
        "    a, b = (1, 2)\n"
        "    local_text = 'x'\n"
        "    local_dict: dict[str, int] = {}\n"
        "    plain_dict = {}\n"
        "    local_list = []\n"
        "    local_set = set()\n"
        "    plain_set = {1}\n"
        "    local_tuple = ()\n"
        "    number = 1\n"
        "    payload = bytes()\n"
        "    vals = [seen for seen in items]\n"
        "    for item in items:\n"
        "        vals.append(item)\n"
        "    with open(__file__) as handle:\n"
        "        data = handle.read()\n"
        "    try:\n"
        "        int('bad')\n"
        "    except ValueError as exc:\n"
        "        data = str(exc)\n"
        "    helper()\n"
        "    helper(1, 2)\n"
        "    helper(y=1)\n"
        "    kwonly(y=1)\n"
        "    variadic(1, 2, z=3)\n"
        "    len(vals)\n"
        "    int(1)\n"
        "    factory()[0]()\n"
        "    local_list.append(1, 2)\n"
        "    items.push(3)\n"
        "    'literal'.no_such_method()\n"
        "    local_text.no_such_method()\n"
        "    local_dict.no_such_method()\n"
        "    plain_dict.no_such_method()\n"
        "    local_set.no_such_method()\n"
        "    plain_set.no_such_method()\n"
        "    local_tuple.no_such_method()\n"
        "    number.no_such_method()\n"
        "    payload.no_such_method()\n"
        "    unknown_object.method()\n"
        "    return data\n"
    )

    result = exp.classify_source(source)

    assert result.syntax_success is True
    assert "invented_import" in result.labels
    assert "invented_attribute_or_method" in result.labels
    assert "invalid_argument" in result.labels
    assert "undefined_name" in result.labels
    symbols = {finding["symbol"] for finding in result.findings}
    assert "json.parse" in symbols
    assert "not_a_real_codegen_module" in symbols
    assert "list.push" in symbols
    assert "helper" in symbols

    assert exp.run_experiment(exp.ExperimentConfig(repo_root=tmp_path))["honest_verdict"] == (
        "blocked_codegen_corrigendum_missing"
    )
    assert exp._module_available("") is False
    assert exp._safe_import_module("pytest") is None
    assert exp._imported_member_available("json", "*") is True
    assert exp._load_raw_response(tmp_path, {"raw_response": "fallback"}) == ("fallback", False)
    assert exp._line_snippet("one\n", 0) == ""
    subscript_call = ast.parse("factory()[0]()").body[0].value.func
    assert exp._call_name(subscript_call) == "Subscript"
    attr_target = ast.parse("obj.x = 1").body[0].targets[0]
    assert exp._target_names(attr_target) == set()

    exp._MODULE_CACHE.pop("json", None)

    def fail_import(_module_name: str) -> Any:
        raise RuntimeError("forced")

    monkeypatch.setattr(exp.importlib, "import_module", fail_import)
    assert exp._safe_import_module("json") is None
    exp._MODULE_CACHE.pop("json", None)

    timeout_row = {"passed": False, "executed": True, "timed_out": True, "error_type": None}
    clean_static = exp.StaticTaxonomyResult(labels=(), findings=(), syntax_success=True)
    assert exp._is_runtime_error(timeout_row, clean_static) is True


def test_scenario_code_2911_writes_taxonomy_artifact(tmp_path: Path) -> None:
    """SCENARIO-CODE-2911: artifact separates true test failures from hallucinations."""

    _stage_upstream(tmp_path)
    artifact = exp.write_experiment_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            tests_run=("focused-pytest",),
            started_at=10.0,
            clock=lambda: 13.5,
        )
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["code_hallucination_verifier_ready"] is True
    assert artifact["upstream_codegen_artifact"] == str(exp.UPSTREAM_CODEGEN_ARTIFACT)
    assert artifact["taxonomy_categories"] == list(exp.TAXONOMY_CATEGORIES)
    assert len(artifact["per_candidate_labels"]) == 4
    assert artifact["invented_import_rate"] == pytest.approx(0.25)
    assert artifact["undefined_name_rate"] == pytest.approx(0.25)
    assert artifact["invented_attribute_or_method_rate"] == pytest.approx(0.25)
    assert artifact["invalid_argument_rate"] == pytest.approx(0.25)
    assert artifact["syntax_error_rate"] == pytest.approx(0.25)
    assert artifact["runtime_error_rate"] == pytest.approx(0.25)
    assert artifact["true_test_failure_rate"] == pytest.approx(0.25)
    assert artifact["pass_rate_after_taxonomy_filter"] == pytest.approx(0.5)
    assert artifact["inference_substrate"] == "deterministic_verifier"
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["run_date"] == "20260523"

    multi = artifact["per_candidate_labels"][3]
    assert set(multi["labels"]) >= {
        "invented_import",
        "undefined_name",
        "invented_attribute_or_method",
        "invalid_argument",
        "runtime_error",
    }
    assert multi["raw_response_loaded"] is True
    assert artifact["summary_by_model"]["local/model-a"]["n_candidates"] == 4
    assert artifact["summary_by_corpus"]["MBPP"]["category_counts"]["true_test_failure"] == 1
    assert artifact["summary_by_task"]["MBPP:task-b"]["category_counts"]["syntax_error"] == 1
    assert artifact["summary_by_pass_status"]["passed"]["n_candidates"] == 1
    assert artifact["summary_by_pass_status"]["failed"]["n_candidates"] == 3


def test_req_code_2911_blocks_without_ready_upstream(tmp_path: Path) -> None:
    """REQ-CODE-2911: missing or unready Exp 2910 writes the blocked taxonomy artifact."""

    missing = exp.build_experiment_artifact(
        exp.ExperimentConfig(repo_root=tmp_path, started_at=1.0, clock=lambda: 2.0)
    )
    assert missing["honest_verdict"] == "blocked_codegen_corrigendum_missing"
    assert missing["code_hallucination_verifier_ready"] is False
    assert missing["per_candidate_labels"] == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(missing)

    _write_json(
        tmp_path / exp.UPSTREAM_CODEGEN_ARTIFACT,
        {"codegen_corrigendum_ready": False, "candidate_results": [], "per_task_results": []},
    )
    unready = exp.write_experiment_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            started_at=1.0,
            clock=lambda: 2.0,
        )
    )
    assert unready["honest_verdict"] == "blocked_codegen_corrigendum_missing"
    assert json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text()) == unready
