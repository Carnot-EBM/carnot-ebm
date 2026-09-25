"""Focused checks for the bounded source-witness challenge.

Spec: REQ-REPORT-7651 and SCENARIO-REPORT-7651-*.
"""

import json
from pathlib import Path

import pytest

from carnot import experiment_7651_v667_qwen_witness_challenge as challenge


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl"


def test_paired_requests_keep_complete_source_and_response() -> None:
    """SCENARIO-REPORT-7651-PAIRED: both arms see the same full input."""

    record = json.loads(PILOT.read_text().splitlines()[0])
    explicit = challenge.build_request(record, challenge.EXPLICIT_ARM)
    grammar = challenge.build_request(record, challenge.GRAMMAR_ARM)
    for request in (explicit, grammar):
        assert request["max_tokens"] == 512
        assert request["seed"] == 7651
        assert request["temperature"] == 0
        assert record["complete_source"] == challenge.schema_protocol.reconstruct_canonical_text(
            challenge.schema_protocol.build_canonical_input(record), "source"
        )
        assert record["complete_answer"] == challenge.schema_protocol.reconstruct_canonical_text(
            challenge.schema_protocol.build_canonical_input(record), "response"
        )
        assert request["messages"] == explicit["messages"]
    assert "grammar" not in explicit
    assert grammar["grammar"]
    with pytest.raises(ValueError, match="invalid_arm"):
        challenge.build_request(record, "other")


def test_structural_predicate_counts_only_checkable_truth() -> None:
    """SCENARIO-REPORT-7651-PREDICATE: unsupported prose has no truth label."""

    source = "```python file=a.py\n1 | def found():\n2 |     pass\n```\n"
    sentences = [
        {"sentence_id": "a1", "text": "In `a.py`, `missing` exists."},
        {"sentence_id": "a2", "text": "The function is safe."},
    ]
    evidence = [
        {"response_sentence_id": "a1", "relation": "supports", "source_sentence_ids": ["s1"]},
        {"response_sentence_id": "a2", "relation": "supports", "source_sentence_ids": ["s1"]},
    ]
    result = challenge.predicate_metrics(source, sentences, evidence, closed_files=["a.py"])
    assert result["structurally_checkable_denominator"] == 1
    assert result["false_support_numerator"] == 1
    assert result["false_support_denominator"] == 1
    assert result["unknown_numerator"] == 1
    assert result["unknown_denominator"] == 2
    assert result["predicates"][0]["source_offset"] is None
    assert result["predicates"][1]["independent_truth"] is None


def test_blocked_capacity_has_exact_operands_and_zero_calls() -> None:
    """SCENARIO-REPORT-7651-BLOCKED: external absence is terminal."""

    gate = challenge.gate(
        "owned_cuda_capacity",
        "nvidia-smi_and_exp7630",
        "/tmp/inventory",
        "exclusive_device",
        "eq",
        True,
        False,
    )
    artifact = challenge.blocked_artifact([gate], duration_s=0.4)
    assert artifact["honest_verdict"] == "complete_blocked_owned_cuda_capacity"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["model_invoked"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["invocation_counts"]["generation_calls_attempted"] == 0
    assert artifact["gate_check_summary"]["first_failure"] == gate


def test_cold_reducer_detects_changed_raw_response(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7651-TERMINAL: raw bytes bind the reduction."""

    path = tmp_path / "response.json"
    path.write_text('{"choices":[]}', encoding="utf-8")
    row = {"raw_response_path": str(path), "raw_response_sha256": challenge.sha256_file(path)}
    assert challenge.raw_rows_valid([row])
    path.write_text('{"choices":[1]}', encoding="utf-8")
    assert not challenge.raw_rows_valid([row])


def test_terminal_builder_and_blocked_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7651-TERMINAL: gates keep benefit separate."""

    artifact = challenge.blocked_artifact(
        [challenge.gate("capacity", "gpu", "/tmp", "free", "eq", True, False)],
        duration_s=0.2,
    )
    challenge._complete_artifact(artifact, started=0, preconditions=[], sources={}, spans=[])
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert challenge.cold_replay(path)["passed"]
    assert artifact["acceptance_gate_results"]["readiness"]["passed"] is False
    assert artifact["acceptance_gate_results"]["probability_benefit"]["passed"] is None
    artifact["model_invoked"] = True
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert not challenge.cold_replay(path)["passed"]


def test_independent_paired_reduction_rechecks_raw_tokens(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7651-PREDICATE: raw tokens own all sixteen rows."""

    records = [json.loads(line) for line in PILOT.read_text().splitlines()]
    rows = []
    for index, record in enumerate(records):
        for arm in (challenge.EXPLICIT_ARM, challenge.GRAMMAR_ARM):
            path = tmp_path / f"{index}-{arm}.json"
            path.write_text(
                json.dumps({"choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}]}),
                encoding="utf-8",
            )
            parsed = challenge.schema_protocol.validate_evidence_output(
                record, "[]", finish_reason="stop"
            )
            rows.append(
                {
                    "component_hash": record["component_hash"],
                    "arm": arm,
                    "input_record": record,
                    "raw_response_path": str(path),
                    "raw_response_sha256": challenge.sha256_file(path),
                    "parser_result": parsed,
                    "closed_files": [],
                    "predicate_metrics": challenge.predicate_metrics(
                        record["complete_source"],
                        record["answer_sentences"],
                        parsed["evidence"],
                        closed_files=[],
                    ),
                }
            )
    candidate = tmp_path / "paired.json"
    candidate.write_text(json.dumps({"rows": rows, "verdict_class": "null"}), encoding="utf-8")
    assert challenge.independent_reduce(candidate)["passed"]
    rows[0]["parser_result"]["accepted"] = False
    candidate.write_text(json.dumps({"rows": rows, "verdict_class": "null"}), encoding="utf-8")
    assert challenge.independent_reduce(candidate)["reason"] == "pointer_validation_changed"
    rows[0]["parser_result"]["accepted"] = True
    rows[0]["predicate_metrics"]["unknown_numerator"] = -1
    candidate.write_text(json.dumps({"rows": rows, "verdict_class": "null"}), encoding="utf-8")
    assert challenge.independent_reduce(candidate)["reason"] == "predicate_reduction_changed"
    rows.pop()
    candidate.write_text(json.dumps({"rows": rows, "verdict_class": "null"}), encoding="utf-8")
    assert challenge.independent_reduce(candidate)["reason"] == "raw_rows_invalid"
