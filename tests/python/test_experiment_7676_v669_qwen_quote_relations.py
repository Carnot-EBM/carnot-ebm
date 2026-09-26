"""REQ-REPORT-7676 and REQ-VERIFY-7676: paired quote relation evidence."""

import json
from pathlib import Path

import pytest

from carnot import experiment_7676_v669_qwen_quote_relations as experiment
from carnot.experiment_7676_v669_qwen_quote_relations import (
    ARMS,
    build_artifact,
    freeze_panel,
    make_request,
)
from carnot.verify.quote_relations import bind_quote, reduce_proposal


def _pilots():
    source = "```\nclass A:\n    def café(self):\n        pass\n```"
    return [
        {
            "component_hash": f"pilot-{index}",
            "complete_source": source,
            "complete_answer": "`café` is defined in `A` at line 2.",
            "source_sha256": __import__(
                "carnot.verify.tool_source_atoms", fromlist=["digest"]
            ).digest(source),
            "answer_sha256": __import__(
                "carnot.verify.tool_source_atoms", fromlist=["digest"]
            ).digest("`café` is defined in `A` at line 2."),
        }
        for index in range(8)
    ]


def _proposal(quote, **updates):
    proposal = {
        "claim_index": 0,
        "source_quote": quote,
        "kind": "definition_in_scope",
        "arguments": {"name": "café", "scope": "A", "line": 2},
        "polarity": "positive",
        "modifiers": [],
        "relation": "supports",
    }
    return {**proposal, **updates}


def test_quote_binding_exact_utf8_and_ambiguity():
    """SCENARIO-VERIFY-7676-BIND: Unicode offsets and duplicate bytes matter."""
    assert bind_quote("xé café!", "café") == [4, 9]
    assert bind_quote("café café", "café") is None
    assert bind_quote("café", "Cafe") is None
    assert bind_quote("café", "") is None


def test_relation_checks_tuple_scope_polarity_and_qualifier():
    """SCENARIO-VERIFY-7676-RELATION: quotes do not rewrite answer meaning."""
    row = freeze_panel(_pilots())[0]
    quote = "    def café(self):"
    response = json.dumps({"proposals": [_proposal(quote)]})
    good = reduce_proposal(row, "exact_quote", response, "stop")
    assert good["unique_evidence"] == 1
    assert good["full_proposition_supported"] == 1
    assert good["proposition_rows"][0]["original_answer_span"]
    for change in (
        {"arguments": {"name": "café", "scope": "B", "line": 2}},
        {"arguments": {"name": "wrong", "scope": "A", "line": 2}},
        {"polarity": "negative"},
        {"modifiers": ["always"]},
    ):
        result = reduce_proposal(
            row, "exact_quote", json.dumps({"proposals": [_proposal(quote, **change)]}), "stop"
        )
        assert result["unique_evidence"] == 1
        assert result["full_proposition_supported"] == 0
    qualified = {**row, "answer": row["answer"] + " It always works."}
    result = reduce_proposal(qualified, "exact_quote", response, "stop")
    assert result["full_proposition_supported"] == 0
    assert result["qualifier_retained"] == 0


def test_schema_truncation_and_numeric_arm():
    """SCENARIO-REPORT-7676-PAIRS: malformed and cut output stays counted."""
    row = freeze_panel(_pilots())[0]
    span = bind_quote(row["source"], "    def café(self):")
    numeric = _proposal("")
    numeric.pop("source_quote")
    numeric.update(source_byte_start=span[0], source_byte_end=span[1])
    metric = reduce_proposal(row, "numeric_offset", json.dumps({"proposals": [numeric]}), "stop")
    assert metric["full_proposition_supported"] == 1
    assert reduce_proposal(row, "exact_quote", "bad", "length")["truncated"]
    assert not reduce_proposal(row, "exact_quote", "bad", "stop")["schema_valid"]
    with pytest.raises(ValueError):
        reduce_proposal(row, "unplanned", "{}", "stop")


def test_panel_and_requests_do_not_expose_oracle():
    """SCENARIO-REPORT-7676-PAIRS: one roster and identical original bytes."""
    panel = freeze_panel(_pilots())
    assert len(panel) == 24
    assert len({row["unit_id"] for row in panel}) == 24
    assert sum(row["population"] == "pilot" for row in panel) == 8
    for arm in ARMS:
        request = make_request(panel[0], arm)
        assert request["max_tokens"] == 256
        visible = json.loads(request["messages"][1]["content"])
        assert panel[0]["source"] == visible["complete_source"]
        assert panel[0]["answer"] == visible["complete_answer"]
        assert "fixture_truth" not in visible
    with pytest.raises(ValueError):
        freeze_panel([])


def test_blocked_and_completion_accounting():
    """SCENARIO-REPORT-7676-ACCOUNTING: absence differs from semantic null."""
    failure = {
        "check": "cached_model",
        "upstream": "model_cache",
        "path": "/missing",
        "field": "gguf_file",
        "operator": "eq",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    blocked = build_artifact([], [failure], {}, {}, [], 0.1)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["gate_check_summary"]["first_failure"] == failure
    assert blocked["quote_pilot_measurement_complete_score"] == 0
    rows = [
        {
            "unit_id": f"unit-{index // 2}",
            "population": "pilot" if index < 16 else "fixture",
            "arm": ARMS[index % 2],
            "metrics": {
                "schema_valid": True,
                "unique_evidence": 0,
                "full_proposition_supported": 0,
                "qualifier_retained": 0,
                "unknown_remainder": 1,
                "proposition_rows": [],
            },
            "generation_s": 0.1,
            "output_tokens": 1,
            "prompt_tokens": 1,
            "censored": False,
            "raw_response_sha256": "sha256:fixture-receipt",
        }
        for index in range(48)
    ]
    artifact = build_artifact(rows, [], {}, {"generation_attempted": 48}, [], 10.1)
    assert artifact["quote_pilot_measurement_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["acceptance_gate_results"]["readiness"]["passed"] is False


def test_bad_input_and_schema_are_rejected(monkeypatch):
    """REQ-VERIFY-7676: model supplied structure never bypasses the reader."""
    pilots = _pilots()
    pilots[0]["answer_sha256"] = "sha256:wrong"
    with pytest.raises(ValueError, match="authentication"):
        freeze_panel(pilots)
    monkeypatch.setattr(experiment.fixtures, "fixture_cases", lambda: [])
    with pytest.raises(ValueError, match="sixteen_relation"):
        freeze_panel(_pilots())
    with pytest.raises(ValueError, match="invalid_arm"):
        make_request({"source": "", "answer": ""}, "bad")
    assert experiment._gate("x", "y", "/z", "field", 1, 0)["passed"] is False


@pytest.mark.parametrize(
    "proposals",
    [
        [],
        {},
        [{"claim_index": 0}],
        [_proposal("x", claim_index=True)],
        [_proposal("x", kind="invented")],
        [_proposal("x", arguments={"line": []})],
        [_proposal("x", polarity="maybe")],
        [_proposal("x", modifiers="always")],
        [_proposal("x", source_quote=12)],
    ],
)
def test_malformed_quote_proposals_stay_invalid(proposals):
    """SCENARIO-VERIFY-7676-BIND: malformed responses cannot gain evidence."""
    row = freeze_panel(_pilots())[0]
    result = reduce_proposal(row, "exact_quote", json.dumps({"proposals": proposals}), "stop")
    if proposals == []:
        assert result["schema_valid"]
    else:
        assert not result["schema_valid"]
        assert result["full_proposition_supported"] == 0


def test_numeric_type_and_unmatched_quote():
    """SCENARIO-VERIFY-7676-BIND: neither malformed offsets nor absent text bind."""
    row = freeze_panel(_pilots())[0]
    wrong = _proposal("not in source")
    result = reduce_proposal(row, "exact_quote", json.dumps({"proposals": [wrong]}), "stop")
    assert result["unique_evidence"] == 0
    numeric = _proposal("")
    numeric.pop("source_quote")
    numeric.update(source_byte_start="0", source_byte_end=4)
    result = reduce_proposal(row, "numeric_offset", json.dumps({"proposals": [numeric]}), "stop")
    assert not result["schema_valid"]
    assert not reduce_proposal(row, "exact_quote", "[]", "stop")["schema_valid"]


def test_cold_replay_detects_raw_and_semantic_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7676-TERMINAL: raw bytes govern a saved decision."""
    from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "RAW", Path("raw"))
    monkeypatch.setattr(experiment, "CAPABILITY", "capability.py")
    (tmp_path / "capability.py").write_text("exact reducer")
    (tmp_path / "raw").mkdir()
    original = freeze_panel(_pilots())[0]
    (tmp_path / "raw" / "frozen_panel.json").write_text(json.dumps([original]))
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    response = {"choices": [{"message": {"content": '{"proposals":[]}'}, "finish_reason": "stop"}]}
    request_path.write_text(json.dumps(make_request(original, "exact_quote")))
    response_path.write_text(json.dumps(response))
    metric = reduce_proposal(original, "exact_quote", '{"proposals":[]}', "stop")
    row = {
        "unit_id": original["unit_id"],
        "arm": "exact_quote",
        "source": original["source"],
        "answer": original["answer"],
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
        "raw_response_path": str(response_path),
        "raw_response_sha256": sha256_file(response_path),
        "metrics": metric,
    }
    artifact = {
        "rows": [row],
        "verdict_class": "null",
        "model_invoked": True,
        "source_artifact_hashes": {"producers": {"source": original["source_sha256"]}},
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {
            "input_hashes": artifact["source_artifact_hashes"]["producers"],
            "seed": experiment.SEED,
            "arms": experiment.ARMS,
            "max_tokens": 256,
            "reducer_sha256": sha256_file(tmp_path / "capability.py"),
        }
    )
    candidate = tmp_path / "candidate.json"

    def saved():
        candidate.write_text(json.dumps(artifact))
        return experiment.independent_reduce(candidate)

    assert saved()["passed"]
    assert experiment.cold_replay(candidate)["passed"]
    artifact["rows"][0]["source"] = "changed"
    assert saved()["reason"] == "source_or_answer_drift"
    artifact["rows"][0]["source"] = original["source"]
    response_path.write_text("changed")
    assert saved()["reason"] == "raw_receipt_drift"
    response_path.write_text(json.dumps(response))
    request_path.write_text("{}")
    artifact["rows"][0]["request_sha256"] = sha256_file(request_path)
    assert saved()["reason"] == "request_drift"
    request_path.write_text(json.dumps(make_request(original, "exact_quote")))
    artifact["rows"][0]["request_sha256"] = sha256_file(request_path)
    artifact["rows"][0]["metrics"]["full_proposition_supported"] = 1
    assert saved()["reason"] == "semantic_reduction_drift"
    artifact["rows"][0]["metrics"] = metric
    artifact["reproducibility_checksum"] = "wrong"
    candidate.write_text(json.dumps(artifact))
    assert not experiment.cold_replay(candidate)["passed"]
    artifact["verdict_class"] = "blocked"
    artifact["rows"] = []
    artifact["model_invoked"] = False
    assert saved() == {"passed": True, "calls": 0}
    artifact["verdict_class"] = "null"
    (tmp_path / "raw" / "frozen_panel.json").unlink()
    assert saved()["reason"] == "frozen_panel_absent"


def test_terminal_command_names(tmp_path):
    """SCENARIO-REPORT-7676-TERMINAL: every reader targets one candidate."""
    commands = experiment._terminal_commands(experiment.ROOT, tmp_path / "candidate.json")
    assert [command.name for command in commands] == [
        "fresh_process_cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    assert all(str(tmp_path / "candidate.json") in command.argv for command in commands)
