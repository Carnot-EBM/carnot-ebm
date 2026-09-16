"""Tests for REQ-CL-7347 and SCENARIO-CL-7347-*.

The tests use public fixture bytes and synthetic transport receipts. They do
not load a model or write the repository result.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_7347_v645_plan_canary as mod


ROOT = Path(__file__).resolve().parents[2]


def _request() -> dict[str, object]:
    """Return one small request that exposes every public plan check."""

    return {
        "request_id": "development-canary-00",
        "version_token": "opaque-public-version",
        "activities": ["a", "b"],
        "allowed_starts": {"a": [0, 2], "b": [1, 3]},
        "durations": {"a": 1, "b": 1},
        "weights": {"a": 2, "b": 1},
        "horizon": 4,
        "public_revision": 0,
        "warmup": True,
    }


def _response_row(index: int, raw_reply: str, *, owned: bool = True) -> dict[str, object]:
    """Build one terminal response with task-bound runner evidence."""

    request = _request()
    request["request_id"] = f"development-canary-{index:02d}"
    return mod.build_call_row(
        call_index=index,
        request=request,
        prompt=mod.render_public_prompt(request),
        response={
            "raw_reply": raw_reply.replace("development-canary-00", str(request["request_id"])),
            "raw_response": {"model": "/cache/qwen.gguf", "choices": [{}]},
            "prompt_tokens": 40,
            "completion_tokens": 20,
            "latency_s": 1.25,
            "finish_reason": "stop",
            "error": None,
        },
        runtime_identity={
            "pid": 1234,
            "start_time_ticks": 99,
            "owned_by_task": owned,
            "command": ["llama-server", "--model", "/cache/qwen.gguf"],
            "model_path": "/cache/qwen.gguf",
            "model_sha256": "sha256:" + "a" * 64,
            "served_model": "/cache/qwen.gguf",
            "gpu_uuid": "GPU-test",
            "lease_id": "lease-test",
            "cuda_provenance_ok": True,
        },
    )


def test_scenario_cl_7347_parser_accepts_only_exact_public_plan_fields() -> None:
    """SCENARIO-CL-7347-PARSER rejects repair, drift, and invalid assignments."""

    request = _request()
    valid = json.dumps({"request_id": request["request_id"], "assignments": {"a": 0, "b": 1}})
    parsed = mod.decode_public_plan(valid, request)
    assert parsed == {
        "parse_status": "valid",
        "parse_errors": [],
        "plan": {"request_id": request["request_id"], "assignments": {"a": 0, "b": 1}},
    }

    invalid = {
        "markdown": f"```json\n{valid}\n```",
        "partial": json.dumps({"request_id": request["request_id"]}),
        "extra": json.dumps(
            {
                "request_id": request["request_id"],
                "assignments": {"a": 0, "b": 1},
                "score": 1,
            }
        ),
        "identity": json.dumps({"request_id": "other", "assignments": {"a": 0, "b": 1}}),
        "activity_set": json.dumps({"request_id": request["request_id"], "assignments": {"a": 0}}),
        "domain": json.dumps(
            {"request_id": request["request_id"], "assignments": {"a": 1, "b": 1}}
        ),
        "boolean": json.dumps(
            {"request_id": request["request_id"], "assignments": {"a": False, "b": 1}}
        ),
    }
    observed = {name: mod.decode_public_plan(raw, request) for name, raw in invalid.items()}
    assert all(
        row["parse_status"] == "invalid" and row["plan"] is None for row in observed.values()
    )
    assert observed["markdown"]["parse_errors"] == ["json_object"]
    assert observed["partial"]["parse_errors"] == ["top_level_fields"]
    assert observed["extra"]["parse_errors"] == ["top_level_fields"]
    assert observed["identity"]["parse_errors"] == ["request_id"]
    assert observed["activity_set"]["parse_errors"] == ["assignment_fields"]
    assert observed["domain"]["parse_errors"] == ["assignment_domain:a"]
    assert observed["boolean"]["parse_errors"] == ["assignment_type:a"]
    assert mod.decode_public_plan("[]", request)["parse_errors"] == ["json_object"]


def test_req_cl_7347_selects_four_development_requests_and_public_prompt() -> None:
    """REQ-CL-7347 freezes four source requests without private evaluator data."""

    fixture = json.loads(mod.PUBLIC_MANIFEST_PATH.read_text(encoding="utf-8"))
    selected = mod.select_development_requests(fixture)
    assert len(selected) == 4
    assert selected == fixture["development_streams"][0]["requests"][:4]
    assert len({row["request_id"] for row in selected}) == 4
    prompt = mod.render_public_prompt(selected[0])
    assert prompt == mod.render_public_prompt(selected[0])
    assert '"request_id"' in prompt and '"assignments"' in prompt
    assert '"private_rules"' not in prompt and "evaluator" not in prompt.lower()
    assert mod.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]

    with pytest.raises(ValueError, match="development_streams"):
        mod.select_development_requests({})
    with pytest.raises(ValueError, match="development_request_count"):
        mod.select_development_requests({"development_streams": [{"requests": []}]})
    duplicate = deepcopy(fixture)
    duplicate["development_streams"][0]["requests"][1]["request_id"] = duplicate[
        "development_streams"
    ][0]["requests"][0]["request_id"]
    with pytest.raises(ValueError, match="development_request_identity"):
        mod.select_development_requests(duplicate)


def test_scenario_cl_7347_transport_is_independent_of_semantic_usability() -> None:
    """SCENARIO-CL-7347-TRANSPORT keeps malformed replies as healthy transport rows."""

    valid = '{"request_id":"development-canary-00","assignments":{"a":0,"b":1}}'
    rows = [
        _response_row(0, valid),
        _response_row(1, valid),
        _response_row(2, "not-json"),
        _response_row(3, '{"request_id":"development-canary-00"}'),
    ]
    reduced = mod.reduce_raw_calls(
        rows,
        load_receipt={"attempted": True, "completed": True, "failed": False, "cancelled": False},
    )
    assert reduced["plan_transport_ready_score"] == 1
    assert reduced["usable_plan_count"] == 2
    assert reduced["invocation_counts"] == {
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "model_loads_failed": 0,
        "model_loads_cancelled": 0,
        "model_loads_in_flight": 0,
        "generation_calls_attempted": 4,
        "generation_calls_completed": 4,
        "generation_calls_failed": 0,
        "generation_calls_cancelled": 0,
        "generation_calls_in_flight": 0,
    }
    assert [row["parse_status"] for row in rows] == ["valid", "valid", "invalid", "invalid"]

    unowned = deepcopy(rows)
    unowned[0]["runtime_identity_receipt"]["owned_by_task"] = False
    failed = mod.reduce_raw_calls(
        unowned,
        load_receipt={"attempted": True, "completed": True, "failed": False, "cancelled": False},
    )
    assert failed["plan_transport_ready_score"] == 0
    assert failed["usable_plan_count"] == 2

    missing_model = deepcopy(rows)
    missing_model[0]["runtime_identity_receipt"]["served_model"] = None
    assert (
        mod.reduce_raw_calls(
            missing_model,
            load_receipt={
                "attempted": True,
                "completed": True,
                "failed": False,
                "cancelled": False,
            },
        )["plan_transport_ready_score"]
        == 0
    )

    content_addressed = deepcopy(rows)
    for row in content_addressed:
        row["runtime_identity_receipt"]["model_path"] = "/cache/blobs/" + "a" * 64
    assert (
        mod.reduce_raw_calls(
            content_addressed,
            load_receipt={
                "attempted": True,
                "completed": True,
                "failed": False,
                "cancelled": False,
            },
        )["plan_transport_ready_score"]
        == 1
    )
    assert mod._served_identity_sound({}, {}) is False


def test_scenario_cl_7347_terminal_reducer_rejects_raw_row_drift() -> None:
    """SCENARIO-CL-7347-TERMINAL recomputes counts and hashes from retained calls."""

    valid = '{"request_id":"development-canary-00","assignments":{"a":0,"b":1}}'
    rows = [_response_row(index, valid) for index in range(4)]
    reduced = mod.reduce_raw_calls(
        rows,
        load_receipt={"attempted": True, "completed": True, "failed": False, "cancelled": False},
    )
    artifact = {
        "rows": rows,
        "raw_call_manifest": {"calls": deepcopy(rows)},
        "load_receipt": {"attempted": True, "completed": True, "failed": False, "cancelled": False},
        **reduced,
    }
    assert mod.independent_reduce(artifact) == []

    changed = deepcopy(artifact)
    changed["rows"][0]["raw_reply"] = "changed"
    assert "rows_manifest_mismatch" in mod.independent_reduce(changed)

    changed = deepcopy(artifact)
    changed["usable_plan_count"] = 0
    assert "usable_plan_count_mismatch" in mod.independent_reduce(changed)

    assert mod.independent_reduce({}) == ["raw_calls_unavailable"]
    changed = deepcopy(artifact)
    changed["plan_transport_ready_score"] = 0
    assert "plan_transport_ready_score_mismatch" in mod.independent_reduce(changed)
    changed = deepcopy(artifact)
    changed["invocation_counts"] = {}
    assert "invocation_counts_mismatch" in mod.independent_reduce(changed)

    disqualified = deepcopy(artifact)
    disqualified["observed_plan_transport_ready_score"] = 1
    disqualified["plan_transport_ready_score"] = 0
    disqualified["verdict_class"] = "disqualified"
    assert mod.independent_reduce(disqualified) == []


def test_req_cl_7347_cpu_controls_use_the_production_parser() -> None:
    """REQ-CL-7347 keeps valid and invalid format controls outside model counts."""

    controls = mod.cpu_parser_controls(_request())
    assert [row["expected"] for row in controls] == ["valid", "invalid"]
    assert all(row["passed"] for row in controls)
    assert all(row["inference_substrate"] == "cpu_parser_control" for row in controls)


def test_req_cl_7347_tokenizer_probe_cannot_consume_the_gpu_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7347 keeps the tokenizer preflight from creating CUDA conflicts."""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    observed: list[str | None] = []

    def probe(_path: str) -> tuple[bool, str]:
        observed.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return True, "embedded tokenizer ok"

    assert mod.cpu_embedded_tokenizer_check("/cache/qwen.gguf", probe=probe) == (
        True,
        "embedded tokenizer ok",
    )
    assert observed == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert mod.cpu_embedded_tokenizer_check("/cache/qwen.gguf", probe=probe)[0] is True
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
