"""Tests for Exp 1607 HumanEval prompt extraction with the NSVIF DSL.

Spec: REQ-CODE-034, SCENARIO-CODE-032.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from carnot.verifiers import dsl


PROMPT_HAS_CLOSE = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    """
'''


PROMPT_TRUNCATE = '''def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    Return the decimal part of the number.
    """
'''


def _case(task_id: str, prompt: str, entry_point: str) -> dict[str, object]:
    return {"task_id": task_id, "prompt": prompt, "entry_point": entry_point}


def test_req_code_034_extracts_humaneval_prompt_constraints_into_safe_dsl() -> None:
    """REQ-CODE-034: HumanEval prompts produce schema-valid code-output constraints."""

    pack = dsl.extract_humaneval_prompt_constraints(_case("HumanEval/0", PROMPT_HAS_CLOSE, ""))
    payload = pack.to_dict()

    assert dsl.validate_constraint_pack(payload) == []
    assert [constraint["op"] for constraint in payload["constraints"]] == [
        "contains",
        "contains",
        "not_contains",
        "not_contains",
        "not_contains",
        "not_contains",
    ]
    assert payload["constraints"][0]["value"] == "def has_close_elements"
    assert payload["constraints"][1]["value"] == "return"
    assert [constraint["value"] for constraint in payload["constraints"][2:]] == [
        "```",
        "TODO",
        "pass",
        "NotImplementedError",
    ]

    validator = dsl.compile_constraint_pack(pack)
    accepted = validator.validate(
        "def has_close_elements(numbers, threshold):\n"
        "    return any(abs(a - b) < threshold for a in numbers for b in numbers if a != b)\n"
    )
    rejected = validator.validate(
        "```python\n"
        "def has_close_elements(numbers, threshold):\n"
        "    pass\n"
        "```\n"
    )

    assert accepted.accepted is True
    assert rejected.accepted is False
    assert {"c002-contains", "c003-not_contains", "c005-not_contains"} <= set(
        rejected.failure_ids
    )


def test_req_code_034_reports_valid_extraction_rates_for_model_specs() -> None:
    """REQ-CODE-034: aggregate rates and mandated model specs are deterministic."""

    cases = [
        _case("HumanEval/0", PROMPT_HAS_CLOSE, "has_close_elements"),
        _case("HumanEval/2", PROMPT_TRUNCATE, "truncate_number"),
        {"task_id": "bad", "prompt": "no function here"},
    ]

    artifact = dsl.evaluate_humaneval_dsl_extraction(cases, sample_rows_limit=2)

    assert artifact["status"] == "partial"
    assert artifact["experiment_id"] == "experiment_1607_dsl_humaneval"
    assert artifact["model_specs"] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    assert artifact["prompts_attempted"] == 3
    assert artifact["valid_extractions"] == 2
    assert artifact["validators_compiled"] == 2
    assert artifact["constraints_extracted"] == 12
    assert artifact["valid_extraction_rate"] == pytest.approx(0.666667)
    assert artifact["compiled_validator_rate"] == pytest.approx(0.666667)
    assert artifact["arbitrary_code_execution_path_introduced"] is False
    assert len(artifact["sample_rows"]) == 2
    assert artifact["case_rows"][2]["error"].startswith("missing HumanEval entry point")
    assert [row["model_spec"] for row in artifact["model_spec_results"]] == artifact["model_specs"]
    assert all(row["valid_extraction_rate"] == artifact["valid_extraction_rate"] for row in artifact["model_spec_results"])


def test_scenario_code_032_writes_terminal_humaneval_dsl_artifact(tmp_path: Path) -> None:
    """SCENARIO-CODE-032: Exp 1607 writes complete artifact fields."""

    output_path = tmp_path / "experiment_1607_dsl_humaneval.json"
    artifact = dsl.write_humaneval_dsl_artifact(
        output_path=output_path,
        cases=[
            _case("HumanEval/0", PROMPT_HAS_CLOSE, "has_close_elements"),
            _case("HumanEval/2", PROMPT_TRUNCATE, "truncate_number"),
        ],
        tests_run=[".venv/bin/pytest tests/python/test_experiment_1607_dsl_humaneval.py -q"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert artifact["status"] == "complete"
    assert artifact["dataset_source"] == "openai_humaneval"
    assert artifact["dsl_schema_version"] == dsl.DSL_SCHEMA_VERSION
    assert artifact["prompts_attempted"] == 2
    assert artifact["valid_extraction_rate"] == pytest.approx(1.0)
    assert artifact["compiled_validator_rate"] == pytest.approx(1.0)
    assert artifact["tests_run"] == [
        ".venv/bin/pytest tests/python/test_experiment_1607_dsl_humaneval.py -q"
    ]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_code_034_loads_humaneval_cases_with_sample_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CODE-034: loader normalizes official HumanEval rows for extraction."""

    class FakeDataset:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, str]:
            rows = [
                {
                    "task_id": "HumanEval/0",
                    "prompt": PROMPT_HAS_CLOSE,
                    "entry_point": "has_close_elements",
                },
                {
                    "task_id": "HumanEval/2",
                    "prompt": PROMPT_TRUNCATE,
                    "entry_point": "truncate_number",
                },
            ]
            return rows[index]

    fake_module = types.SimpleNamespace(load_dataset=lambda name, split: FakeDataset())
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    cases = dsl.load_humaneval_prompt_cases(sample_size=1)

    assert cases == [
        {
            "dataset_idx": 0,
            "task_id": "HumanEval/0",
            "prompt": PROMPT_HAS_CLOSE,
            "entry_point": "has_close_elements",
        }
    ]
