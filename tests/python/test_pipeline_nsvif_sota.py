"""Tests for the NSVIF SOTA GGUF output adapter.

Spec: REQ-VERIFY-1641, SCENARIO-VERIFY-1641.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import nsvif_sota as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"


def test_req_verify_1641_adapts_instruction_json_to_dsl_input_and_carnot_rows() -> None:
    """REQ-VERIFY-1641: raw SOTA JSON instruction output becomes NSVIF DSL input."""

    row = {
        "case_id": "json-text-bound",
        "model_hf_id": QWEN,
        "model_name": "Qwen3.6-35B-A3B",
        "model_path": "/models/qwen.gguf",
        "output_text": (
            'model preface {"instruction": "Respond in JSON with keys answer and '
            'confidence. Include \\"approved\\". Do not mention \\"secret\\". '
            'Use at most 12 words."} trailing text'
        ),
        "known_good": '{"answer": "approved", "confidence": "high"}',
        "known_bad": '{"answer": "secret", "extra": true}',
    }

    result = mod.adapt_sota_output(row)

    assert result["adapter_success"] is True
    assert result["dsl_input"]["schema_version"] == mod.dsl.DSL_SCHEMA_VERSION
    assert [constraint["op"] for constraint in result["dsl_input"]["constraints"]] == [
        "json_object",
        "json_has_keys",
        "contains",
        "not_contains",
        "max_words",
    ]
    assert result["validator_compiled"] is True
    assert result["known_good"]["accepted"] is True
    assert result["known_bad"]["accepted"] is False
    assert result["raw_output_sha256"] == mod.sha256_text(str(row["output_text"]))

    first_constraint = result["carnot_constraints"][0]
    assert first_constraint["constraint_type"] == "instruction_constraint"
    assert first_constraint["metadata"]["model_hf_id"] == QWEN
    assert first_constraint["metadata"]["case_id"] == "json-text-bound"
    assert first_constraint["metadata"]["spec_traces"] == mod.SPEC_TRACES


def test_req_verify_1641_accepts_constraint_list_and_existing_pack_shapes() -> None:
    """REQ-VERIFY-1641: supported constraints and existing DSL packs normalize safely."""

    constraint_row = {
        "case_id": "direct-constraints",
        "model_hf_id": QWEN,
        "output_text": json.dumps(
            {
                "instruction": 'Answer one of "yes", "no".',
                "constraints": [
                    {
                        "id": "choice",
                        "op": "enum",
                        "field": "text",
                        "value": ["yes", "no"],
                        "source_text": "bounded choice",
                    }
                ],
            }
        ),
        "known_good": "yes",
        "known_bad": "maybe",
    }
    packed_row = {
        "case_id": "existing-pack",
        "model_hf_id": QWEN,
        "output_text": json.dumps(
            {
                "dsl_pack": {
                    "schema_version": mod.dsl.DSL_SCHEMA_VERSION,
                    "instruction": 'Include "north". Use at least 2 words.',
                    "constraints": [
                        {
                            "id": "c001-contains",
                            "op": "contains",
                            "field": "text",
                            "value": "north",
                        },
                        {"id": "c002-min_words", "op": "min_words", "field": "text", "value": 2},
                    ],
                }
            }
        ),
        "known_good": "north star",
        "known_bad": "north",
    }

    constraint_result = mod.adapt_sota_output(constraint_row)
    packed_result = mod.adapt_sota_output(packed_row)

    assert constraint_result["adapter_success"] is True
    assert constraint_result["dsl_input"]["constraints"][0]["op"] == "enum"
    assert constraint_result["known_bad"]["failure_ids"] == ["choice"]
    assert packed_result["adapter_success"] is True
    assert [item["op"] for item in packed_result["dsl_input"]["constraints"]] == [
        "contains",
        "min_words",
    ]
    assert packed_result["known_bad"]["failure_ids"] == ["c002-min_words"]


def test_req_verify_1641_fails_closed_for_unsafe_unparseable_or_unsupported_output() -> None:
    """REQ-VERIFY-1641: unsafe, unparseable, and unsupported SOTA outputs do not compile."""

    rows = [
        {"case_id": "no-json", "model_hf_id": QWEN, "output_text": "no json here"},
        {"case_id": "malformed-json", "model_hf_id": QWEN, "output_text": "before {bad json"},
        {
            "case_id": "unsafe",
            "model_hf_id": QWEN,
            "output_text": '{"instruction": "Include \\"__import__\\"."}',
        },
        {
            "case_id": "unsupported",
            "model_hf_id": QWEN,
            "output_text": '{"instruction": "Answer carefully."}',
        },
        {
            "case_id": "bad-op",
            "model_hf_id": QWEN,
            "output_text": json.dumps(
                {
                    "instruction": "Unsupported direct op.",
                    "constraints": [{"op": "regex", "field": "text", "value": ".*"}],
                }
            ),
        },
    ]

    failures = [mod.adapt_sota_output(row) for row in rows]

    assert [failure["adapter_success"] for failure in failures] == [
        False,
        False,
        False,
        False,
        False,
    ]
    assert [failure["validator_compiled"] for failure in failures] == [
        False,
        False,
        False,
        False,
        False,
    ]
    assert failures[0]["error"] == "no_json_object"
    assert failures[1]["error"] == "no_json_object"
    assert failures[2]["error"].startswith("unsafe token:")
    assert failures[3]["error"] == "no supported constraints"
    assert "unsupported" in failures[4]["error"]


def test_scenario_verify_1641_builds_complete_artifact_with_zero_false_accepts() -> None:
    """SCENARIO-VERIFY-1641: artifact records complete adapter metrics and zero false accepts."""

    artifact = mod.build_artifact(
        rows=mod.default_sota_output_cases(),
        tests_run=[".venv/bin/pytest tests/python/test_pipeline_nsvif_sota.py -q"],
    )

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["spec_traces"] == mod.SPEC_TRACES
    assert artifact["adapter_schema_version"] == mod.ADAPTER_SCHEMA_VERSION
    assert artifact["sota_outputs_seen"] == 3
    assert artifact["dsl_inputs_emitted"] == 3
    assert artifact["validators_compiled"] == 3
    assert artifact["known_good_pass_rate"] == pytest.approx(1.0)
    assert artifact["known_bad_reject_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["arbitrary_code_execution_path_introduced"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1641_run_experiment_writes_json_deliverable(tmp_path: Path) -> None:
    """REQ-VERIFY-1641: run_experiment writes the stable JSON deliverable."""

    output_path = tmp_path / "results" / "experiment_1641_nsvif_sota.json"

    artifact = mod.run_experiment(
        output_path=output_path,
        tests_run=["focused"],
        rows=mod.default_sota_output_cases(),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["artifact_path"] == str(output_path)
    assert artifact["tests_run"] == ["focused"]


def test_req_verify_1641_artifact_validation_catches_inconsistent_shapes() -> None:
    """REQ-VERIFY-1641: artifact validation rejects missing fields and false accepts."""

    artifact = mod.build_artifact(rows=mod.default_sota_output_cases(), tests_run=[])
    partial = mod.build_artifact(
        rows=[{"case_id": "no-json", "model_hf_id": QWEN, "output_text": "no json"}],
        tests_run=[],
    )

    missing = dict(artifact)
    del missing["dsl_inputs_emitted"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="false_accept_rate"):
        mod.validate_artifact(dict(artifact, false_accept_rate=1.0))

    with pytest.raises(AssertionError, match="validators_compiled"):
        mod.validate_artifact(dict(artifact, validators_compiled=2))

    mod.validate_artifact(partial)
    assert partial["status"] == "partial"
    assert partial["honest_verdict"].startswith("partial:")
