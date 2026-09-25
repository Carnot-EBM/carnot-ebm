"""REQ-REPORT-7665: bounded, byte-scoped grounded-claim comparison."""

import json
import time

import pytest

from carnot.experiment_7665_v668_qwen_grounded_claims import (
    _artifact,
    blocked_artifact,
    freeze_panel,
    gate,
    make_request,
    reduce_response,
)
from carnot.verify.grounded_claims import check_pointers, parse_pointers


def test_panel_freezes_two_populations_before_response():
    """SCENARIO-REPORT-7665-PANEL: a fixture view cannot enlarge the sample."""
    pilot = [
        {
            "component_hash": f"pilot-{index}",
            "complete_source": "`````\ndef f():\n    pass\n`````",
            "complete_answer": "`f` is defined at line 1.",
        }
        for index in range(8)
    ]
    panel = freeze_panel(pilot)
    assert len(panel) == 24
    assert [row["population"] for row in panel].count("pilot") == 8
    assert [row["population"] for row in panel].count("synthetic_control") == 16
    assert len({row["unit_id"] for row in panel}) == 24
    assert all("truth" in row for row in panel)


def test_requests_preserve_bytes_and_hide_truth():
    """SCENARIO-REPORT-7665-PANEL: both prompts use original bytes and one schema."""
    panel = freeze_panel(
        [
            {
                "component_hash": f"pilot-{index}",
                "complete_source": "```\ndef f():\n    pass\n```",
                "complete_answer": "`f` is defined at line 1.",
            }
            for index in range(8)
        ]
    )
    row = panel[0]
    source_only = make_request(row, "source_only")
    indexed = make_request(row, "typed_index")
    assert source_only["max_tokens"] == indexed["max_tokens"] == 256
    assert source_only["seed"] == indexed["seed"]
    assert (
        source_only["chat_template_kwargs"]
        == indexed["chat_template_kwargs"]
        == {"enable_thinking": False}
    )
    for request in (source_only, indexed):
        visible = json.dumps(request)
        user_content = json.loads(request["messages"][1]["content"])
        assert user_content["complete_source"] == row["source"]
        assert user_content["complete_answer"] == row["answer"]
        assert "scoped_contradiction" not in visible
        assert "independent_truth" not in visible
    assert "typed_atom_index" not in json.dumps(source_only)
    assert "typed_atom_index" in json.dumps(indexed)


def test_pointer_checker_rejects_overlap_and_erasure():
    """SCENARIO-REPORT-7665-POINTER: exact source bytes and typed truth govern support."""
    source = "```\ndef alpha():\n    pass\n```"
    answer = "`alpha` is defined at line 1."
    panel = freeze_panel(
        [
            {
                "component_hash": f"pilot-{index}",
                "complete_source": source,
                "complete_answer": answer,
            }
            for index in range(8)
        ]
    )
    row = panel[0]
    span = row["truth"]["witnesses"][0]["evidence_span"]
    good = {
        "claim_index": 0,
        "source_byte_start": span[0],
        "source_byte_end": span[1],
        "relation": "supports",
    }
    metrics = check_pointers(row, [good])
    assert metrics["exact_supported"] == 1
    assert metrics["invalid_pointers"] == 0
    assert (
        check_pointers(row, [{**good, "source_byte_start": span[0] + 1}])["invalid_pointers"] == 1
    )
    assert check_pointers({**row, "source": ""}, [good])["exact_supported"] == 0


def test_schema_and_response_reduction_fail_closed():
    """SCENARIO-REPORT-7665-POINTER: malformed and unsupported support stays visible."""
    assert parse_pointers('{"pointers":[]}') == []
    with pytest.raises(ValueError):
        parse_pointers('{"pointers":[{"claim_index":0}]}')
    row = freeze_panel(
        [
            {
                "component_hash": f"pilot-{index}",
                "complete_source": "```\ndef f():\n    pass\n```",
                "complete_answer": "`f` is defined at line 9.",
            }
            for index in range(8)
        ]
    )[0]
    result = reduce_response(row, '{"pointers":[]}', "stop")
    assert result["truncated"] is False
    assert result["claim_count"] == 1
    assert reduce_response(row, "bad", "length")["truncated"] is True


def test_external_absence_is_complete_blocked():
    """SCENARIO-REPORT-7665-RESOURCE: a missing model cannot be a scientific null."""
    failure = {
        "check": "model",
        "upstream": "cache",
        "path": "/missing",
        "field": "gguf",
        "operator": "eq",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    result = blocked_artifact([failure])
    assert result["honest_verdict"].startswith("complete_blocked_")
    assert result["verdict_class"] == "blocked"
    assert result["gate_check_summary"]["first_failure"] == failure
    assert result["model_invoked"] is False
    assert result["MODEL_SPECS"] == []


def test_contract_rejects_bad_panel_request_and_pointer(monkeypatch):
    """SCENARIO-REPORT-7665-PANEL: malformed schedules and pointers stop early."""
    with pytest.raises(ValueError, match="eight_distinct"):
        freeze_panel([])
    pilots = [
        {
            "component_hash": f"pilot-{index}",
            "complete_source": "```\ndef f():\n pass\n```",
            "complete_answer": "`f` is defined at line 1.",
        }
        for index in range(8)
    ]
    from carnot import experiment_7665_v668_qwen_grounded_claims as module

    original_cases = module.fixtures.fixture_cases
    monkeypatch.setattr(module.fixtures, "fixture_cases", lambda: [])
    with pytest.raises(ValueError, match="sixteen_fixture"):
        freeze_panel(pilots)
    monkeypatch.setattr(module.fixtures, "fixture_cases", original_cases)
    with pytest.raises(ValueError, match="invalid_arm"):
        make_request(freeze_panel(pilots)[0], "unplanned")
    for response in (
        "[]",
        '{"pointers":{}}',
        '{"pointers":[{"claim_index":0,"source_byte_start":0,"source_byte_end":1,"relation":"wrong"}]}',
    ):
        with pytest.raises(ValueError):
            parse_pointers(response)
    assert gate("x", "y", "/z", "field", True, False)["passed"] is False


def test_artifact_keeps_gates_and_fixture_scope_separate():
    """SCENARIO-REPORT-7665-TERMINAL: complete accounting is separate from benefit."""
    rows = [
        {
            "unit_id": f"unit-{index // 2}",
            "population": "pilot" if index < 16 else "synthetic_control",
            "arm": "source_only" if index % 2 == 0 else "typed_index",
            "metrics": {
                "exact_supported": 0,
                "unsupported_certifications": 0,
                "invalid_pointers": 0,
                "claim_count": 1,
                "proposition_rows": [],
            },
            "prompt_tokens": 1,
            "output_tokens": 1,
            "censored": False,
            "generation_s": 0.1,
        }
        for index in range(48)
    ]
    hashes = {
        "producers": {"pilot": "sha256:test"},
        "pre_gate_receipts": {},
        "missing_inputs": [],
        "planned_outputs": [],
    }
    runtime = {
        "model_load_attempted": 1,
        "model_load_completed": 1,
        "generation_attempted": 48,
        "device_uuid": "GPU-test",
        "server_pid": 123,
        "model_path": "/cache/snapshots/revision/model.gguf",
        "model_sha256": "sha256:model",
    }
    result = _artifact(rows, [], [], hashes, [], runtime, time.monotonic(), "20260925")
    assert result["verdict_class"] == "null"
    assert result["grounded_claim_measurement_complete_score"] == 1
    assert result["acceptance_gate_results"]["readiness"]["passed"] is False
    assert result["acceptance_gate_results"]["probability_benefit"]["passed"] is None
    assert result["population_arm_metrics"]["pilot"]["source_only"]["coverage"] == 0
    assert result["model_runtime_receipt"]["model_sha256"] == "sha256:model"
    assert result["model_revision"] == "revision"
    failure = gate("missing", "upstream", "/missing", "file", True, False)
    blocked = _artifact([], [failure], [failure], hashes, [], {}, time.monotonic(), "20260925")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["grounded_claim_measurement_complete_score"] == 0
