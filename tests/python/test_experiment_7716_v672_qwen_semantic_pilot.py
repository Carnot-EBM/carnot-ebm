"""REQ-REPORT-7716 and REQ-VERIFY-7716 contract tests."""

import json

from carnot.verify import semantic_evidence as evidence
from carnot import experiment_7716_v672_qwen_semantic_pilot as pilot


def test_req_verify_7716_windows_keep_original_bytes():
    source = "First fact.  Second fact!\nThird fact?"
    windows = evidence.sentence_windows(source)
    assert "".join(item["text"] for item in windows) == source
    assert [item["start"] for item in windows] == [0, 13, 26]
    row = {"source": source, "answer": "An original answer.", "family_id": "f"}
    whole = pilot.make_request(row, "whole_source")
    indexed = pilot.make_request(row, "indexed_windows")
    assert whole["temperature"] == indexed["temperature"] == 0
    assert whole["max_tokens"] == indexed["max_tokens"] == 128
    visible_whole = json.loads(whole["messages"][1]["content"])
    visible_indexed = json.loads(indexed["messages"][1]["content"])
    assert visible_whole["answer"] == visible_indexed["answer"] == row["answer"]
    assert "".join(item["text"] for item in visible_indexed["source_windows"]) == source
    assert "labels" not in json.dumps(whole) + json.dumps(indexed)


def test_scenario_verify_7716_address_is_not_entailment():
    source = "One claim. Another claim."
    output = '{"decision":"support","quote":"One claim"}'
    result = evidence.reduce_response(source, output, "stop", "Evident Conflict")
    assert result["schema_valid"] is True
    assert result["address_valid"] is True
    assert result["human_label_agreement"] is False
    assert result["semantic_verified"] is False
    assert (
        evidence.reduce_response("X X", '{"decision":"support","quote":"X"}', "stop", None)[
            "address_valid"
        ]
        is False
    )
    assert evidence.reduce_response(source, "{broken", "length", None)["censored"] is True


def test_scenario_report_7716_pair_counts_unknown_and_missing():
    rows = [
        {
            "family_id": "a",
            "arm": "whole_source",
            "metrics": {"decision": "unknown", "schema_valid": True},
            "latency_s": 2,
        },
        {
            "family_id": "a",
            "arm": "indexed_windows",
            "metrics": {"decision": "support", "schema_valid": True},
            "latency_s": 3,
        },
        {
            "family_id": "b",
            "arm": "whole_source",
            "metrics": {"decision": None, "schema_valid": False},
            "latency_s": 4,
        },
    ]
    result = evidence.reduce_pairs(rows)
    assert result["paired_families"] == 1
    assert result["unknown_calls"] == 1
    assert result["schema_valid_calls"] == 2
    assert result["pairs"][0]["latency_delta_s"] == 1
