"""REQ-REPORT-7702 and REQ-VERIFY-7702: bounded record proposal pilot."""

import json
from pathlib import Path

import pytest

from carnot.experiment_7672_v669_bound_relations import fixture_cases
from carnot.experiment_7702_v671_qwen_record_pilot import (
    build_artifact,
    freeze_panel,
    gate,
    make_request,
)
from carnot.verify.record_proposals import reduce_response
from carnot.verify.tool_source_atoms import digest


def _pilots():
    source, answer = "```\n at run0 (pkg0/a.js:5:2)\n```", "`run0` at `pkg0/a.js:5`."
    return [
        {
            "component_hash": f"pilot-{index}",
            "complete_source": source,
            "complete_answer": answer,
            "source_sha256": digest(source),
            "answer_sha256": digest(answer),
        }
        for index in range(8)
    ]


def _proposal(quote="run0", **changes):
    return {
        "sentence_id": "S001",
        "proposition_id": "P001",
        "kind": "stack_frame",
        "arguments": {"function": "run0", "path": "pkg0/a.js", "line": 5},
        "polarity": "positive",
        "modifiers": [],
        "relation": "supports",
        "source_quote": quote,
        **changes,
    }


def test_paired_panel_and_requests_hide_truth():
    """SCENARIO-REPORT-7702-PAIRED: freeze 24 groups and identical input bytes."""
    panel = freeze_panel(_pilots())
    assert len(panel) == 24
    assert len({row["unit_id"] for row in panel}) == 24
    assert sum(row["population"] == "fixture" for row in panel) == 16
    assert {case["id"] for case in fixture_cases()}.issuperset(
        row["unit_id"] for row in panel if row["population"] == "fixture"
    )
    for row in panel:
        old = make_request(row, "opaque_index")
        new = make_request(row, "explicit_record")
        assert old["messages"][1]["content"] != new["messages"][1]["content"]
        for request in (old, new):
            assert request["max_tokens"] == 256
            assert request["temperature"] == 0
            assert request["seed"] == 7702
            visible = json.loads(request["messages"][1]["content"])
            assert visible["complete_source"] == row["source"]
            assert visible["complete_answer"] == row["answer"]
            assert "truth" not in request["messages"][1]["content"]


def test_substring_location_and_ambiguous_duplicate():
    """SCENARIO-VERIFY-7702-ADDRESS: a subquote locates, a duplicate does not."""
    row = freeze_panel(_pilots())[0]
    result = reduce_response(row, "explicit_record", json.dumps({"proposal": _proposal()}), "stop")
    assert result["schema_valid"]
    assert result["exact_span"]
    assert result["unique_containing_record"]
    assert result["correct_binding"]
    assert result["tuple_truth"] == "supported"
    assert result["supported"]
    changed = {**row, "source": row["source"] + "\n" + row["source"]}
    ambiguous = reduce_response(
        changed, "explicit_record", json.dumps({"proposal": _proposal()}), "stop"
    )
    assert not ambiguous["unique_containing_record"]
    assert not ambiguous["supported"]
    wrong = reduce_response(
        row,
        "explicit_record",
        json.dumps({"proposal": _proposal(arguments={"function": "wrong"})}),
        "stop",
    )
    assert wrong["exact_span"] and not wrong["correct_binding"]
    assert not wrong["supported"]


def test_malformed_truncated_and_blocked_accounting():
    """SCENARIO-REPORT-7702-BLOCKED: absent capacity is a terminal external block."""
    row = freeze_panel(_pilots())[0]
    invalid = reduce_response(row, "opaque_index", "bad", "length")
    assert not invalid["schema_valid"] and invalid["truncated"]
    assert invalid["unknown_remainder"]
    failure = {
        "check": "owned_cuda_capacity",
        "upstream_id": "exp7630",
        "artifact_path": "/dev/nvidia0",
        "field": "exclusive_idle_device",
        "operator": "==",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    artifact = build_artifact([], [failure], {}, {}, [], 0.0)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["MODEL_SPECS"] == []
    assert artifact["gate_check_summary"][0] == failure


@pytest.mark.parametrize(
    "change",
    [
        {"claim_index": "zero"},
        {"kind": 4},
        {"arguments": []},
        {"arguments": {"bad": []}},
        {"polarity": "maybe"},
        {"relation": "certain"},
        {"modifiers": "always"},
        {"modifiers": [4]},
        {"source_quote": 2},
    ],
)
def test_invalid_opaque_proposal_fields(change):
    """REQ-VERIFY-7702: generated fields require exact schema and types."""
    row = freeze_panel(_pilots())[0]
    proposal = _proposal()
    proposal.pop("sentence_id")
    proposal.pop("proposition_id")
    proposal["claim_index"] = 0
    proposal.update(change)
    result = reduce_response(row, "opaque_index", json.dumps({"proposal": proposal}), "stop")
    assert not result["schema_valid"]
    assert not result["supported"]


def test_wrong_references_and_request_arm():
    """REQ-VERIFY-7702: wrong IDs and malformed containers stay unknown."""
    row = freeze_panel(_pilots())[0]
    for change in ({"sentence_id": "S999"}, {"proposition_id": 5}):
        result = reduce_response(
            row, "explicit_record", json.dumps({"proposal": _proposal(**change)}), "stop"
        )
        assert not result["supported"]
    for text in ("[]", '{"proposal": []}', '{"proposal": {}}'):
        assert not reduce_response(row, "explicit_record", text, "stop")["schema_valid"]
    with pytest.raises(ValueError):
        make_request(row, "other")
    with pytest.raises(ValueError):
        reduce_response(row, "other", "{}", "stop")
    assert not gate("x", "u", "p", "f", True, False)["passed"]
    proposal = _proposal()
    del proposal["sentence_id"]
    del proposal["proposition_id"]
    proposal["claim_index"] = 0
    assert reduce_response(row, "opaque_index", json.dumps({"proposal": proposal}), "stop")[
        "supported"
    ]


def test_paired_artifact_difference_is_one_group():
    """SCENARIO-REPORT-7702-PAIRED: paired views never enlarge independent n."""
    row = freeze_panel(_pilots())[0]
    metrics = reduce_response(row, "explicit_record", json.dumps({"proposal": _proposal()}), "stop")
    common = {
        "unit_id": row["unit_id"],
        "population": "pilot",
        "censored": False,
        "raw_response_sha256": "sha256:raw",
        "request_path": "request.json",
        "request_sha256": "sha256:request",
        "raw_response_path": "response.json",
        "metrics": metrics,
    }
    rows = [{**common, "arm": arm} for arm in ("opaque_index", "explicit_record")]
    artifact = build_artifact(rows, [], {}, {"model_load_attempted": 1}, [], 11.0)
    assert artifact["sample_size_budget"]["observed"] == 1
    assert artifact["sample_size_budget"]["effective_blocks"] == 1
    assert artifact["paired_group_differences"][0]["support_delta"] == 0
    assert artifact["MODEL_SPECS"] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert artifact["verdict_class"] == "partial"


def test_preflight_exact_gates_and_owned_capacity(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7702-BLOCKED: absent, unqualified and owned inputs differ."""
    from carnot import experiment_7630_v666_cuda_ownership as ownership
    from carnot import experiment_7702_v671_qwen_record_pilot as exp
    from carnot.inference import sota_models
    from llama_cpp import llama_cpp

    monkeypatch.setattr(exp.protocol, "RESULT", Path("upstream.json"))
    monkeypatch.setattr(exp.protocol, "PILOT", Path("pilot.jsonl"))
    monkeypatch.setattr(exp.protocol, "RAW", Path("schema"))
    monkeypatch.setattr(exp.custody, "sha256_file", lambda _: "sha256:fixture")
    checks, hashes, context = exp.preflight(tmp_path, 0.0)
    assert context == {}
    assert hashes["missing_custody"]
    assert {item["upstream_id"] for item in checks if not item["passed"]} >= {"exp7700"}

    (tmp_path / "pilot.jsonl").write_text("{}\n")
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/exclusion_manifest.yaml").write_text("retired: []\n")
    (tmp_path / "upstream.json").write_text(json.dumps({"record_protocol_ready_score": 0}))
    checks, _, context = exp.preflight(tmp_path, 0.0)
    assert context == {}
    assert any(
        item["field"] == "record_protocol_ready_score" and not item["passed"] for item in checks
    )

    (tmp_path / "schema").mkdir()
    (tmp_path / "schema/feature_schema.json").write_text("{}")
    (tmp_path / "upstream.json").write_text(
        json.dumps(
            {
                "record_protocol_ready_score": 1,
                "verdict_class": "circular_positive",
                "flagged_adversarial": False,
                "feature_schema_path": "schema/feature_schema.json",
            }
        )
    )
    monkeypatch.setattr(sota_models, "cached_current_model", lambda **_: None)
    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)
    checks, _, context = exp.preflight(tmp_path, 0.0)
    assert context == {}
    assert {item["check"] for item in checks if not item["passed"]} == {
        "cached_qwen_identity",
        "cuda_offload_support",
    }

    model_path = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    with model_path.open("wb") as model_file:
        model_file.truncate(15_000_000_001)
    monkeypatch.setattr(
        sota_models,
        "cached_current_model",
        lambda **_: {"hf_id": exp.MODEL_ID, "model_path": str(model_path)},
    )
    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", lambda: True)
    monkeypatch.setattr(ownership.ProcessRegistry, "current", lambda: object())
    monkeypatch.setattr(ownership, "_current_inventory", lambda: [])
    monkeypatch.setattr(ownership, "select_owned_capacity", lambda *_: (None, []))
    checks, hashes, context = exp.preflight(tmp_path, 0.0)
    assert context["selected"] is None
    assert hashes["producers"][str(model_path)] == "sha256:fixture"
    assert checks[-1]["check"] == "owned_cuda_capacity" and not checks[-1]["passed"]


def test_cold_replay_raw_pair_and_tamper(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7702-PAIRED: replay both raw arms, then reject changed custody."""
    from carnot import experiment_7702_v671_qwen_record_pilot as exp

    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(exp, "RAW", Path("raw"))
    (tmp_path / "raw").mkdir()
    base = freeze_panel(_pilots())[0]
    panel = [{**base, "unit_id": f"unit-{index}"} for index in range(24)]
    (tmp_path / "raw/frozen_panel.json").write_text(json.dumps(panel))
    rows = []
    content = json.dumps({"proposal": _proposal()})
    for group in panel:
        for arm in ("opaque_index", "explicit_record"):
            number = len(rows)
            request = make_request(group, arm)
            response = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
            request_path = tmp_path / f"request-{number}.json"
            response_path = tmp_path / f"response-{number}.json"
            request_path.write_text(json.dumps(request))
            response_path.write_text(json.dumps(response))
            rows.append(
                {
                    "unit_id": group["unit_id"],
                    "arm": arm,
                    "source": group["source"],
                    "answer": group["answer"],
                    "request_path": str(request_path),
                    "request_sha256": exp.custody.sha256_file(request_path),
                    "raw_response_path": str(response_path),
                    "raw_response_sha256": exp.custody.sha256_file(response_path),
                    "response_text": content,
                    "finish_reason": "stop",
                    "metrics": reduce_response(group, arm, content, "stop"),
                }
            )
    candidate = tmp_path / "candidate.json"
    artifact = {"rows": rows, "verdict_class": "null", "model_invoked": True}
    candidate.write_text(json.dumps(artifact))
    assert exp.independent_reduce(candidate) == {"passed": True, "calls": 48}
    assert exp.cold_replay(candidate) == {"passed": True, "calls": 48}
    assert len(exp.terminal_commands(tmp_path, candidate)) == 4
    for key, value, reason in (
        ("request_sha256", "bad", "raw_hash_mismatch"),
        ("source", "changed", "request_source_answer_mismatch"),
        ("response_text", "changed", "raw_response_mismatch"),
        ("metrics", {}, "reduction_mismatch"),
    ):
        changed = json.loads(json.dumps(artifact))
        changed["rows"][0][key] = value
        candidate.write_text(json.dumps(changed))
        assert exp.independent_reduce(candidate)["reason"] == reason
        assert not exp.cold_replay(candidate)["passed"]
    candidate.write_text(
        json.dumps({"rows": [], "verdict_class": "blocked", "model_invoked": False})
    )
    assert exp.cold_replay(candidate) == {"passed": True, "calls": 0}
