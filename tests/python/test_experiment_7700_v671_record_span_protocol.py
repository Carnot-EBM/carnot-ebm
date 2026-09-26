"""REQ-REPORT-7700 and REQ-VERIFY-7700: record addresses and narrow claims."""

import json
from pathlib import Path

import pytest

from carnot.experiment_7700_v671_record_span_protocol import (
    PILOT,
    ROOT,
    build_artifact,
    build_rows,
    cold_reduce,
    feature_schema,
    freeze_panel,
    preconditions,
)
from carnot.experiment_7672_v669_bound_relations import fixture_cases
from carnot.verify.record_addresses import (
    SourceRecord,
    analyze_answer,
    index_records,
    replay_analysis,
    resolve_address,
)


CASES = fixture_cases()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_frozen_fixture_panel(case):
    """SCENARIO-REPORT-7700-FIXTURES: each source group is one independent case."""
    assert len(CASES) == 72
    assert sum(row["split"] == "held_out" for row in CASES) == 24
    result = analyze_answer(case["source"], case["answer"])
    assert result["source_sha256"].startswith("sha256:")
    assert result["answer_sha256"].startswith("sha256:")
    assert result["sentences"]
    assert {row["sentence_id"] for row in result["propositions"]} <= {
        row["sentence_id"] for row in result["sentences"]
    }
    assert len({row["proposition_id"] for row in result["propositions"]}) == len(
        result["propositions"]
    )
    if case["truth"] == "unknown":
        assert result["whole_answer_status"] != "supported"
    replay_analysis(case["source"], case["answer"], json.loads(json.dumps(result)))


def test_subquote_and_numeric_span_address_only():
    """SCENARIO-VERIFY-7700-ADDRESS: a subquote locates a record, not a claim."""
    source = "```\n at café (src/a.js:4:2)\n at other (src/b.js:8:1)\n```"
    records = index_records(source)
    quote = resolve_address(source, records, quote="café (src/a.js")
    assert quote.valid and quote.record_id == records[0].record_id
    assert quote.unique_containment and quote.exact_span
    numeric = resolve_address(source, records, span=quote.span)
    assert numeric == quote
    answer = "`café` at `src/a.js:4` because it crashes."
    result = analyze_answer(source, answer)
    assert result["propositions"][0]["narrow_status"] == "supported"
    assert result["whole_answer_status"] == "unknown"
    assert result["residual_unknown_text"]
    assert resolve_address(source, records).reason == "one_address_required"


@pytest.mark.parametrize(
    ("source", "kwargs", "reason"),
    [
        ("```\na/x.py:7: 'same'\nb/x.py:8: 'same'\n```", {"quote": "same"}, "duplicate_quote"),
        ("```\na/x.py:7: 'same'\n```", {"quote": "missing"}, "quote_absent"),
        ("```\na/x.py:7: 'é'\n```", {"span": [0, 2]}, "outside_record"),
        ("```\na/x.py:7: 'é'\n```", {"span": [16, 17]}, "utf8_split"),
        ("```\na/x.py:7: 'x'\n```", {"span": [-1, 4]}, "invalid_span"),
        ("```\na/x.py:7: 'x'\n```", {"span": [0, 999]}, "invalid_span"),
    ],
)
def test_invalid_address_reasons(source, kwargs, reason):
    """SCENARIO-VERIFY-7700-ADDRESS: invalid bytes fail closed."""
    assert resolve_address(source, index_records(source), **kwargs).reason == reason


def test_cross_record_and_overlap_rejected():
    """SCENARIO-VERIFY-7700-ADDRESS: one span must have exactly one owner."""
    source = "```\na/x.py:7: 'x'\nb/x.py:8: 'y'\n```"
    records = index_records(source)
    cross = resolve_address(source, records, span=[records[0].byte_start, records[1].byte_end])
    assert not cross.valid and cross.reason == "cross_record"
    overlap = SourceRecord(
        record_id="overlap",
        byte_start=records[0].byte_start,
        byte_end=records[0].byte_end,
        source_bytes=records[0].source_bytes,
        dialect=records[0].dialect,
        source_id=records[0].source_id,
        line=records[0].line,
        tuples=records[0].tuples,
        polarity="positive",
        complete=False,
    )
    ambiguous = resolve_address(
        source, [*records, overlap], span=[records[0].byte_start, records[0].byte_end]
    )
    assert not ambiguous.valid and ambiguous.reason == "ambiguous_overlap"


def test_tuple_binding_and_atom_certificate():
    """SCENARIO-VERIFY-7700-CLAIM: pair arguments in one clause and one record."""
    source = "```\n at café (src/a.js:4:2)\n at other (src/b.js:8:1)\n```"
    answer = "`café` at `src/a.js:4`. `café` at `src/b.js:8`."
    result = analyze_answer(source, answer)
    assert [row["narrow_status"] for row in result["propositions"]] == ["supported", "unknown"]
    assert all(row["certificate_type"] == "bound_tuple" for row in result["propositions"])
    assert [row["sentence_id"] for row in result["propositions"]] == ["S001", "S002"]
    atom = analyze_answer("```\na/x.py:7: value\n```", "See `a/x.py:7`.")
    assert atom["propositions"][0]["certificate_type"] == "atom_membership"
    assert atom["propositions"][0]["narrow_status"] == "supported"


def test_unicode_replay_and_authentication(tmp_path: Path):
    """SCENARIO-VERIFY-7700-CLAIM: serialization keeps original UTF-8 bytes."""
    source = "```\na/x.py:7: 'café'\n```"
    answer = "É: `a/x.py:7` contains 'café'."
    value = analyze_answer(source, answer)
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    replay_analysis(source, answer, json.loads(path.read_text(encoding="utf-8")))
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["propositions"][0]["answer_span"][0] += 1
    with pytest.raises(ValueError, match="replay"):
        replay_analysis(source, answer, changed)
    empty = analyze_answer(source, "   ")
    assert empty["sentences"] == []
    assert empty["whole_answer_status"] == "unknown"


def test_pilot_authentication_and_frozen_rows():
    """SCENARIO-REPORT-7700-FIXTURES: exposed pilots are diagnostic only."""
    pilots = [json.loads(line) for line in (ROOT / PILOT).read_text().splitlines()]
    panel = freeze_panel(pilots)
    assert len(panel) == 80
    assert sum(row["population"] == "pilot" for row in panel) == 8
    assert sum(row["split"] == "held_out" for row in panel) == 24
    rows = build_rows(panel)
    assert len(rows) == 160
    assert all(row["population"] != "pilot" or row["truth"] is None for row in rows)
    assert all("source_sha256" in row and "answer_sha256" in row for row in rows)
    assert {row["arm"] for row in rows} == {"old_relation", "record_span"}
    changed = json.loads(json.dumps(pilots))
    changed[0]["complete_source"] += "x"
    with pytest.raises(ValueError, match="source_authentication"):
        freeze_panel(changed)
    changed = json.loads(json.dumps(pilots))
    changed[0]["complete_answer"] += "x"
    with pytest.raises(ValueError, match="answer_authentication"):
        freeze_panel(changed)
    with pytest.raises(ValueError, match="eight_distinct"):
        freeze_panel(pilots[:7])


def test_artifact_cold_replay_and_gates(tmp_path: Path):
    """SCENARIO-REPORT-7700-TERMINAL: raw rows and terminal checks govern readiness."""
    pilots = [json.loads(line) for line in (ROOT / PILOT).read_text().splitlines()]
    panel = freeze_panel(pilots)
    rows = build_rows(panel)
    checks = preconditions(ROOT)
    assert all(item["passed"] for item in checks)
    receipts = [{"name": "all_required", "passed": True, "exit_code": 0}]
    artifact = build_artifact(rows, panel, checks, receipts, "20260926", 1.0)
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["record_protocol_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["feature_schema_path"].endswith("feature_schema.json")
    assert {gate["gate"] for gate in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "coverage",
        "freshness",
        "probability",
        "utility",
        "retention",
        "efficiency",
    }
    assert set(feature_schema()["certificate_types"]) == {
        "bound_tuple",
        "atom_membership",
        "atom_definition",
    }
    candidate = tmp_path / "candidate.json"
    panel_path = tmp_path / "panel.jsonl"
    panel_path.write_text("".join(json.dumps(row) + "\n" for row in panel))
    candidate.write_text(json.dumps(artifact))
    assert cold_reduce(candidate, panel_path)["fixture_groups"] == 72
    changed = json.loads(candidate.read_text())
    changed["rows"][0]["observed"] = "contradicted"
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="row_reduction"):
        cold_reduce(candidate, panel_path)
    candidate.write_text(json.dumps(artifact))
    panel_path.write_text("".join(json.dumps(row) + "\n" for row in panel[:-1]))
    with pytest.raises(ValueError, match="row_reduction_count"):
        cold_reduce(candidate, panel_path)
    failed = build_artifact(rows, panel, checks, [{"name": "x", "passed": False}], "20260926", 1.0)
    assert failed["verdict_class"] == "disqualified"
    assert failed["record_protocol_ready_score"] == 0
    missing = [
        *checks,
        {
            "check": "input_exists",
            "upstream_id": "pilot",
            "artifact_path": "x",
            "field": "bytes",
            "operator": "!=",
            "expected": None,
            "observed": None,
            "passed": False,
        },
    ]
    blocked = build_artifact([], [], missing, receipts, "20260926", 1.0)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["gate_check_summary"]
