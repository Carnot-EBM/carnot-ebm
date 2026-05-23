"""Tests for Exp 2890 MBPP/HumanEval structural dependency verifier.

Spec: REQ-CODE-2890, SCENARIO-CODE-2890.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import code_structural_dependency_verifier as exp


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mbpp_row(stable_id: str = "mbpp-a") -> dict[str, Any]:
    return {
        "canonical_code": "def add_one(x):\n    return x + 1\n",
        "dataset": "MBPP",
        "prompt": "Write a function that adds one.",
        "stable_id": stable_id,
        "test_imports": [],
        "tests": ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
    }


def _humaneval_row(stable_id: str = "HumanEval/0") -> dict[str, Any]:
    return {
        "canonical_solution": "    return x * 2\n",
        "dataset": "HumanEval",
        "entry_point": "double",
        "prompt": "def double(x: int) -> int:\n    \"\"\"Return x doubled.\"\"\"\n",
        "stable_id": stable_id,
        "tests": "\ndef check(candidate):\n    assert candidate(3) == 6\n",
    }


def test_req_code_2890_contract_schema_parses_manifest_row() -> None:
    """REQ-CODE-2890: contracts expose signature, inputs, edges, tests, and obligations."""

    contract = exp.build_contract_from_manifest_row("mbpp", _mbpp_row(), manifest_path="mbpp.jsonl")
    payload = exp.contract_to_json(contract)
    result = exp.verify_candidate_source(contract, _mbpp_row()["canonical_code"], "reference")

    assert payload["contract_schema_version"] == exp.CONTRACT_SCHEMA_VERSION
    assert payload["function_signature"] == {"name": "add_one", "parameters": ["x"]}
    assert payload["required_inputs"] == ["x"]
    assert ["input:x", "function:add_one", "argument_to_function"] in payload["dependency_edges"]
    assert payload["forbidden_imports"]
    assert payload["test_prerequisites"]["n_tests"] == 2
    assert "defines_entry_point" in payload["output_obligations"]
    assert exp.validate_contract_json(payload) == []
    assert result["passed"] is True
    assert result["violations"] == []
    assert result["unsupported_reasons"] == []

    invalid_payload = dict(payload)
    invalid_payload["contract_schema_version"] = "wrong"
    invalid_payload["function_signature"] = {}
    invalid_payload["dependency_edges"] = []
    assert exp.validate_contract_json(invalid_payload) == [
        "invalid:contract_schema_version",
        "invalid:function_signature",
        "invalid:dependency_edges",
    ]


def test_scenario_code_2890_localizes_static_violations() -> None:
    """SCENARIO-CODE-2890: malformed generated code localizes deterministic failures."""

    contract = exp.build_contract_from_manifest_row(
        "humaneval",
        _humaneval_row(),
        manifest_path="humaneval.jsonl",
    )

    syntax_result = exp.verify_candidate_source(contract, "def double(x)\n    return x\n", "generated")
    missing_result = exp.verify_candidate_source(contract, "from typing import List\n", "generated")
    side_effect_result = exp.verify_candidate_source(
        contract,
        "def double(x):\n    eval('x')\n",
        "generated",
    )
    signature_result = exp.verify_candidate_source(
        contract,
        "def double():\n    return 2\n",
        "generated",
    )
    forbidden_import_result = exp.verify_candidate_source(
        contract,
        "import os\n\ndef double(x):\n    return x\n",
        "generated",
    )
    empty_result = exp.verify_candidate_source(contract, "", "generated")

    assert syntax_result["passed"] is False
    assert syntax_result["violations"][0]["violation_type"] == "parse_error"
    assert syntax_result["violations"][0]["contract_field"] == "function_signature"
    assert syntax_result["violations"][0]["line"] == 1

    missing = missing_result["violations"][0]
    assert missing["violation_type"] == "missing_function_definition"
    assert missing["contract_field"] == "function_signature"
    assert missing["line"] == 1
    assert "double" in missing["message"]

    violation_types = {v["violation_type"] for v in side_effect_result["violations"]}
    assert {"forbidden_side_effect", "missing_return_obligation"} <= violation_types
    by_type = {v["violation_type"]: v for v in side_effect_result["violations"]}
    assert by_type["forbidden_side_effect"]["line"] == 2
    assert by_type["missing_return_obligation"]["line"] == 1

    signature_types = {v["violation_type"] for v in signature_result["violations"]}
    assert {"signature_mismatch", "missing_dependency_edge"} <= signature_types
    assert forbidden_import_result["violations"][0]["violation_type"] == "forbidden_import"
    assert empty_result["violations"] == []
    assert empty_result["unsupported_reasons"] == ["empty_candidate_source"]


def test_req_code_2890_handles_contract_fallbacks_and_side_effect_variants() -> None:
    """REQ-CODE-2890: fallback parsing remains deterministic for odd manifest rows."""

    string_test_contract = exp.build_contract_from_manifest_row(
        "mbpp",
        {
            "canonical_code": "def add_one(x):\n    return x + 1\n",
            "dataset": "MBPP",
            "prompt": "Tests stored as a string.",
            "stable_id": "mbpp-string-tests",
            "tests": "assert add_one(1) == 2",
        },
        manifest_path="mbpp.jsonl",
    )
    fallback_contract = exp.build_contract_from_manifest_row(
        "mbpp",
        {
            "canonical_code": "def inferred_name(*args, **kwargs):\n    return args\n",
            "dataset": "MBPP",
            "prompt": "No parseable test call.",
            "stable_id": "mbpp-fallback",
            "tests": ["assert broken("],
        },
        manifest_path="mbpp.jsonl",
    )
    fallback_to_first_contract = exp.build_contract_from_manifest_row(
        "mbpp",
        {
            "canonical_code": "def actual_name(x):\n    return x\n",
            "dataset": "MBPP",
            "prompt": "Tests name a function that canonical code does not define.",
            "stable_id": "mbpp-fallback-first",
            "tests": ["assert expected_name(1) == 1"],
        },
        manifest_path="mbpp.jsonl",
    )
    no_function_contract = exp.build_contract_from_manifest_row(
        "mbpp",
        {
            "canonical_code": "VALUE = 1\n",
            "dataset": "MBPP",
            "prompt": "No function in canonical code.",
            "stable_id": "mbpp-no-function",
            "tests": [],
        },
        manifest_path="mbpp.jsonl",
    )
    no_tests_contract = exp.build_contract_from_manifest_row(
        "mbpp",
        {
            "canonical_code": "def orphan(x):\n    return x\n",
            "dataset": "MBPP",
            "prompt": "A row with no usable local tests.",
            "stable_id": "mbpp-no-tests",
            "tests": [],
        },
        manifest_path="mbpp.jsonl",
    )
    base_contract = exp.build_contract_from_manifest_row(
        "mbpp",
        _mbpp_row(),
        manifest_path="mbpp.jsonl",
    )
    attr_side_effect = exp.verify_candidate_source(
        base_contract,
        "def add_one(x):\n    return __builtins__.eval('x')\n",
        "generated",
    )
    wrong_name = exp.verify_candidate_source(
        base_contract,
        "def add_two(x):\n    return x + 2\n",
        "generated",
    )
    no_tests = exp.verify_candidate_source(
        no_tests_contract,
        "def orphan(x):\n    return x\n",
        "generated",
    )
    unknown_call = exp.verify_candidate_source(
        base_contract,
        "def add_one(x):\n    return (lambda y: y)(x)\n",
        "generated",
    )
    comment_only = exp.verify_candidate_source(base_contract, "# comment\n", "generated")
    custom_contract = exp.build_contract_from_manifest_row(
        "custom",
        _mbpp_row("custom-a"),
        manifest_path="custom.jsonl",
    )

    assert string_test_contract.function_name == "add_one"
    assert fallback_contract.function_name == "inferred_name"
    assert fallback_contract.required_inputs == ("*args", "**kwargs")
    assert fallback_to_first_contract.function_name == "expected_name"
    assert fallback_to_first_contract.required_inputs == ("x",)
    assert no_function_contract.required_inputs == ()
    assert attr_side_effect["violations"][0]["violation_type"] == "forbidden_side_effect"
    assert wrong_name["violations"][0]["violation_type"] == "missing_function_definition"
    assert no_tests["violations"][0]["violation_type"] == "missing_test_prerequisite"
    assert unknown_call["passed"] is True
    assert comment_only["violations"][0]["line"] is None
    assert custom_contract.corpus == "custom"


def test_req_code_2890_defensive_localization_helpers(tmp_path: Path) -> None:
    """REQ-CODE-2890: localization helper edge cases stay deterministic."""

    rows = [
        {
            "corpus": "MBPP",
            "stable_id": f"row-{idx}",
            "candidate_kind": "generated",
            "violations": [
                {
                    "violation_type": "parse_error",
                    "contract_field": "function_signature",
                    "line": idx + 1,
                    "message": "bad",
                    "snippet": "x",
                }
            ],
        }
        for idx in range(2)
    ]

    assert exp._localization_examples(rows, limit=1) == [
        {
            "corpus": "MBPP",
            "stable_id": "row-0",
            "candidate_kind": "generated",
            "violation_type": "parse_error",
            "contract_field": "function_signature",
            "line": 1,
            "message": "bad",
            "snippet": "x",
        }
    ]
    assert exp._line_snippet("only one line", 3) == ""
    assert exp._source_name(tmp_path, Path("/definitely/outside/carnot.json")) == (
        "/definitely/outside/carnot.json"
    )
    assert exp._first_function_name("def bad(") == ""


def test_scenario_code_2890_builds_artifact_from_2879_and_2889_rows(tmp_path: Path) -> None:
    """SCENARIO-CODE-2890: Exp 2879 contracts and Exp 2889 outputs become matrix metadata."""

    mbpp_a = _mbpp_row("mbpp-a")
    mbpp_b = {
        "canonical_code": "def negate(x):\n    return -x\n",
        "dataset": "MBPP",
        "prompt": "Write a function that negates the input.",
        "stable_id": "mbpp-b",
        "test_imports": [],
        "tests": ["assert negate(3) == -3"],
    }
    humaneval = _humaneval_row("HumanEval/0")
    mbpp_path = tmp_path / "data" / "eval_manifests" / "mbpp_20260522.jsonl"
    humaneval_path = tmp_path / "data" / "eval_manifests" / "humaneval_20260522.jsonl"
    _write_jsonl(mbpp_path, [mbpp_a, mbpp_b])
    _write_jsonl(humaneval_path, [humaneval])

    _write_json(
        tmp_path / exp.EXP2879_REL_PATH,
        {
            "artifact": "experiment_2879_code_corpus_manifest_execution_pilot_v1",
            "honest_verdict": "complete: MBPP/HumanEval manifest-only execution pilot ready",
            "manifest_paths": {"mbpp": str(mbpp_path), "humaneval": str(humaneval_path)},
            "pilot_rows": [
                {
                    "corpus": "MBPP",
                    "stable_id": "mbpp-a",
                    "row_sha256": exp.stable_json_sha256(mbpp_a),
                    "passed": True,
                },
                {
                    "corpus": "HumanEval",
                    "stable_id": "HumanEval/0",
                    "row_sha256": exp.stable_json_sha256(humaneval),
                    "passed": True,
                },
            ],
        },
    )
    _write_json(
        tmp_path / exp.CROSS_CORPUS_MATRIX_V6_REL_PATH,
        {"artifact": "experiment_2880_cross_corpus_matrix_v6", "honest_verdict": "complete"},
    )
    _write_json(
        tmp_path / exp.EXP2889_REL_PATH,
        {
            "artifact": "experiment_2889_mbpp_humaneval_generated_code_clean_row_v1",
            "honest_verdict": "complete: bounded SOTA GGUF generation executed cleanly",
            "manifest_paths": {"mbpp": str(mbpp_path), "humaneval": str(humaneval_path)},
            "row_results": [
                {
                    "corpus": "MBPP",
                    "stable_id": "mbpp-b",
                    "row_sha256": exp.stable_json_sha256(mbpp_b),
                    "extracted_code": "def negate(x)\n    return -x\n",
                    "generated_text_sha256": "badsyntax",
                },
                {
                    "corpus": "HumanEval",
                    "stable_id": "HumanEval/0",
                    "row_sha256": exp.stable_json_sha256(humaneval),
                    "extracted_code": "from typing import List\n",
                    "generated_text_sha256": "missing",
                },
                {
                    "corpus": "MBPP",
                    "stable_id": "mbpp-missing",
                    "row_sha256": "missing",
                    "extracted_code": "def missing(x):\n    return x\n",
                    "generated_text_sha256": "not-in-manifest",
                },
            ],
        },
    )

    artifact = exp.write_experiment_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            started_at=100.0,
            clock=lambda: 103.25,
            tests_run=("focused-pytest",),
        )
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["structural_dependency_verifier_ready"] is True
    assert artifact["contract_schema_version"] == exp.CONTRACT_SCHEMA_VERSION
    assert artifact["n_contracts_built"] == 3
    assert artifact["n_rows_verified"] == 5
    assert artifact["generated_outputs_consumed"] is True
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["tests_run"] == ["focused-pytest"]
    assert artifact["source_artifacts"] == [
        "results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json",
        "results/experiment_2880_cross_corpus_matrix_v6.json",
        "results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json",
        "data/eval_manifests/mbpp_20260522.jsonl",
        "data/eval_manifests/humaneval_20260522.jsonl",
    ]
    assert artifact["violation_types"]["parse_error"] == 1
    assert artifact["violation_types"]["missing_function_definition"] == 1
    assert artifact["unsupported_contracts"] == [
        {
            "corpus": "MBPP",
            "stable_id": "mbpp-missing",
            "unsupported_reason": "manifest_row_not_found",
        }
    ]
    assert artifact["localization_examples"]
    assert artifact["field_principles"]["generated_outputs_consumed"]

    (tmp_path / exp.EXP2889_REL_PATH).unlink()
    no_generated = exp.build_experiment_artifact(
        exp.ExperimentConfig(repo_root=tmp_path, started_at=200.0, clock=lambda: 201.0)
    )
    assert no_generated["generated_outputs_consumed"] is False
