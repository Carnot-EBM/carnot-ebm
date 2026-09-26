"""REQ-REPORT-7672 and REQ-VERIFY-7672: bound source relations."""

import json
from pathlib import Path
import time

import pytest

from carnot.experiment_7672_v669_bound_relations import (
    PILOT,
    ROOT,
    _artifact,
    _hash,
    _progress,
    cold_reduce,
    fixture_cases,
    fixture_rows,
    main,
    pilot_findings,
    pilot_rows,
)
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES
from carnot.verify.tool_source_relations import index_relations, replay_relations, verify_relations


def test_stack_tuple_and_partial_mismatch():
    """SCENARIO-VERIFY-7672-TUPLE: one frame must hold all arguments."""
    source = "```\n at café (src/a.js:4:2)\n at other (src/b.js:8:1)\n```"
    yes = verify_relations(source, "`café` at `src/a.js:4`.")
    wrong = verify_relations(source, "`café` at `src/b.js:8`.")
    assert yes["status"] == "supported"
    assert wrong["status"] == "unknown"
    assert yes["findings"][0]["answer_span"] == [0, len("`café` at `src/a.js:4`".encode())]
    assert any(row["arguments"]["function"] == "café" for row in index_relations(source))


def test_grep_quote_binding_and_duplicate_quotes():
    """SCENARIO-VERIFY-7672-TUPLE: a quoted token in another row gives no support."""
    source = "```\na/x.py:7: 'café'\nb/x.py:7: 'other'\n```"
    assert verify_relations(source, "`a/x.py:7` contains 'café'.")["status"] == "supported"
    assert verify_relations(source, "`b/x.py:7` contains 'café'.")["status"] == "unknown"
    duplicate = "```\na/x.py:7: 'café'\na/x.py:7: 'other'\n```"
    assert verify_relations(duplicate, "`a/x.py:7` contains 'café'.")["status"] == "unknown"


def test_ast_scope_contradiction_and_negation():
    """SCENARIO-VERIFY-7672-QUALIFIER: complete scope may decide a narrow absence."""
    source = "```\nclass A:\n    def run(self):\n        pass\nclass B:\n    def stop(self):\n        pass\n```"
    assert verify_relations(source, "`run` is defined in `A` at line 2.")["status"] == "supported"
    assert (
        verify_relations(source, "`run` is defined in `B` at line 2.")["status"] == "contradicted"
    )
    assert (
        verify_relations(source, "`run` is not defined in `A` at line 2.")["status"]
        == "contradicted"
    )
    assert (
        verify_relations(source, "`run` is not defined in `B` at line 2.")["status"] == "supported"
    )


@pytest.mark.parametrize("extra", ["because of a bug", "might", "always", "and breaks startup"])
def test_qualifiers_remain_unknown(extra):
    """SCENARIO-VERIFY-7672-QUALIFIER: a tuple cannot certify added meaning."""
    source = "```\n at run (src/a.js:4:2)\n```"
    result = verify_relations(source, f"`run` at `src/a.js:4` {extra}.")
    assert result["status"] == "unknown"
    assert result["residual_unverified_text"]


def test_truncation_and_utf8_replay():
    """SCENARIO-VERIFY-7672-REPLAY: offsets are bytes and drift fails closed."""
    source = "```\n at café (src/a.js:4:2)"
    answer = "é: `café` at `src/a.js:4`."
    result = verify_relations(source, answer)
    assert result["status"] == "unknown"
    assert index_relations(source) == []
    complete = source + "\n```"
    result = verify_relations(complete, answer)
    saved = json.loads(json.dumps(result))
    replay_relations(complete, answer, saved)
    assert saved["findings"][0]["answer_span"][0] == len("é: ".encode())
    saved["findings"][0]["answer_span"][0] += 1
    with pytest.raises(ValueError, match="replay"):
        replay_relations(complete, answer, saved)


def test_fixture_panel_truth_and_split():
    """SCENARIO-REPORT-7672-FIXTURES: 72 independent oracle groups, 24 held out."""
    cases = fixture_cases()
    assert len(cases) >= 72
    assert len({row["id"] for row in cases}) == len(cases)
    assert sum(row["split"] == "held_out" for row in cases) == 24
    assert {row["dialect"] for row in cases} == {"stack", "grep", "ast"}
    rows = fixture_rows(cases)
    assert len(rows) == 2 * len(cases)
    assert not [
        row
        for row in rows
        if row["arm"] == "bound_relation"
        and row["observed"] == "supported"
        and row["truth"] != "supported"
    ]
    assert all(row["provenance"] == "exact_fixture_oracle" for row in rows)


def test_pilot_custody_and_cold_reduction(tmp_path, capsys):
    """SCENARIO-VERIFY-7672-REPLAY: exposed pilots cannot change after custody."""
    inputs = [json.loads(line) for line in (ROOT / PILOT).read_text().splitlines()]
    findings = pilot_findings(inputs)
    assert len(findings) == 8
    assert all(not row["fresh_accuracy_claim"] for row in findings)
    assert len(pilot_rows(findings)) == 16
    changed = json.loads(json.dumps(inputs))
    changed[0]["complete_source"] += "x"
    with pytest.raises(ValueError, match="source_authentication"):
        pilot_findings(changed)
    changed = json.loads(json.dumps(inputs))
    changed[0]["complete_answer"] += "x"
    with pytest.raises(ValueError, match="answer_authentication"):
        pilot_findings(changed)
    checks = [{"check": "input_exists", "passed": True}]
    receipts = [{"name": name, "passed": True, "exit_code": 0} for name in REQUIRED_CHECK_NAMES]
    hashes = {
        "producers": {str(PILOT): sha256_file(ROOT / PILOT)},
        "pre_gate_receipts": {},
        "missing_evidence": [],
    }
    rows = fixture_rows(fixture_cases()) + pilot_rows(findings)
    artifact = _artifact(
        rows, findings, checks, receipts, [], time.monotonic(), "20260926", hashes, {}
    )
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["relation_protocol_ready_score"] == 1
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    assert cold_reduce(path)["fixture_groups"] == 72
    assert main(["--cold-reduce", str(path)]) == 0
    assert "fixture_groups" in capsys.readouterr().out
    bad = json.loads(path.read_text())
    bad["rows"][0]["observed"] = "unknown"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="row_reduction"):
        cold_reduce(path)
    bad = json.loads(json.dumps(artifact))
    bad["pilot_findings"][0]["bound_decision"] = "supported"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="pilot_reduction"):
        cold_reduce(path)
    bad = json.loads(json.dumps(artifact))
    bad["source_artifact_hashes"]["producers"][str(PILOT)] = "sha256:changed"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="source_hash"):
        cold_reduce(path)
    assert _hash({"x": 1}) == _hash({"x": 1})
    _progress("test", "done", time.monotonic())
    assert "test done" in capsys.readouterr().out


def test_artifact_invalid_gates_and_main_dispatch(monkeypatch, tmp_path):
    """SCENARIO-REPORT-7672-TERMINAL: blocked and failed checks cannot open readiness."""
    inputs = [json.loads(line) for line in (ROOT / PILOT).read_text().splitlines()]
    findings = pilot_findings(inputs)
    rows = fixture_rows(fixture_cases()) + pilot_rows(findings)
    checks = [{"check": "input_exists", "passed": False}]
    receipts = [{"name": name, "passed": True, "exit_code": 0} for name in REQUIRED_CHECK_NAMES]
    args = (
        rows,
        findings,
        checks,
        receipts,
        [],
        time.monotonic(),
        "20260926",
        {"producers": {}},
        {},
    )
    blocked = _artifact(*args)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"] == checks
    checks = [{"check": "pilot_byte_authentication", "passed": False}]
    assert (
        _artifact(
            rows,
            findings,
            checks,
            receipts,
            [],
            time.monotonic(),
            "20260926",
            {"producers": {}},
            {},
        )["verdict_class"]
        == "disqualified"
    )
    receipts[0]["passed"] = False
    assert (
        _artifact(
            rows, findings, [], receipts, [], time.monotonic(), "20260926", {"producers": {}}, {}
        )["relation_protocol_ready_score"]
        == 0
    )
    import carnot.experiment_7672_v669_bound_relations as module

    called = []
    monkeypatch.setattr(module, "run_experiment", lambda date, path: called.append((date, path)))
    assert main(["--date", "20260926", "--output", str(tmp_path / "out.json")]) == 0
    assert called == [("20260926", tmp_path / "out.json")]
