"""REQ-REPORT-7658: native evidence atoms and independently labeled fixtures."""

import hashlib
import json
from pathlib import Path

import pytest

from carnot.experiment_7658_v668_evidence_atoms import (
    _artifact,
    _progress,
    cold_reduce,
    fixture_cases,
    fixture_rows,
    pilot_rows,
)
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES
from carnot.verify.tool_source_atoms import parse_atoms, replay, verify_answer


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl"


def digest(value):
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def test_fixture_truth_and_exact_spans():
    """SCENARIO-REPORT-7658-DIALECTS: 64 oracle labels remain independent."""
    cases = fixture_cases()
    assert len(cases) >= 64
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["dialect"] for case in cases} == {"plain", "numbered", "grep", "stack"}
    for case in cases:
        atoms = parse_atoms(case["source"])
        checked = verify_answer(case["source"], case["answer"])
        assert checked["status"] == case["truth"], case["id"]
        for atom in atoms:
            encoded = case["source"].encode()
            assert encoded[atom["byte_start"] : atom["byte_end"]].decode() == atom["text"]
        replay(
            case["source"],
            case["answer"],
            digest(case["source"]),
            digest(case["answer"]),
            atoms,
            checked,
        )


def test_mutations_fail_closed():
    """SCENARIO-REPORT-7658-MUTATIONS: hash, offset and sidecar violations fail."""
    case = fixture_cases()[0]
    atoms = parse_atoms(case["source"])
    checked = verify_answer(case["source"], case["answer"])
    args = (case["source"], case["answer"], digest(case["source"]), digest(case["answer"]))
    with pytest.raises(ValueError, match="source_hash"):
        replay(*args[:2], digest("changed"), args[3], atoms, checked)
    with pytest.raises(ValueError, match="answer_hash"):
        replay(*args[:3], digest("changed"), atoms, checked)
    damaged = [dict(atom) for atom in atoms]
    damaged[0]["byte_start"] += 1
    with pytest.raises(ValueError, match="offset"):
        replay(*args, damaged, checked)
    with pytest.raises(ValueError, match="sidecar"):
        replay(*args, atoms, checked, evaluator_sidecar={"label": "supports"})


def test_partial_search_and_listing_abstain():
    """SCENARIO-REPORT-7658-SCOPE: omitted rows and listings prove no absence."""
    search = "Tool output:\n```\na/main.py:7: value = 1\n```"
    assert verify_answer(search, "`a/main.py` at line 9")["status"] == "unknown"
    listing = "Tool output:\n```\n-rw-r--r-- 1 user staff 12 foo.py\n```"
    assert parse_atoms(listing) == []
    assert verify_answer(listing, "`foo.py` at line 12")["status"] == "unknown"
    assert verify_answer(search, "`value` did not cause the failure")["status"] == "unknown"


def test_eight_pilots_reproduce_miss_then_cover_four():
    """REQ-REPORT-7658: old grammar misses all pilots; native atoms check four."""
    from carnot.verify.source_claim_witness import verify_claim

    inputs = [json.loads(line) for line in PILOT.read_text().splitlines()]
    assert len(inputs) == 8
    assert all(
        verify_claim(row["complete_source"], part["text"])["status"] == "unknown"
        for row in inputs
        for part in row["answer_sentences"]
    )
    rows = pilot_rows(inputs)
    assert len(rows) == 8
    assert sum(row["checked_atoms"] > 0 for row in rows) >= 4
    assert all(row["whole_answer_certified"] is False for row in rows)
    assert all(
        row["answer_sha256"] == digest(inputs[i]["complete_answer"]) for i, row in enumerate(rows)
    )


def test_cold_reduction_and_terminal_classes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7658-TERMINAL: replay and checks govern class."""
    import time

    inputs = [json.loads(line) for line in PILOT.read_text().splitlines()]
    pilots = pilot_rows(inputs)
    fixtures = fixture_rows(fixture_cases())
    checks = [{"passed": True, "check": "source"}]
    receipts = [{"name": name, "passed": True, "exit_code": 0} for name in REQUIRED_CHECK_NAMES]
    artifact = _artifact(
        pilots + fixtures, fixtures, checks, receipts, [], time.monotonic(), "20260925", {}
    )
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["atom_protocol_ready_score"] == 1
    assert artifact["inference_substrate"].endswith("_no_llm")
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    assert cold_reduce(path)["fixture_groups"] >= 64
    altered = json.loads(path.read_text())
    altered["rows"].pop()
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="row_reduction"):
        cold_reduce(path)
    altered = json.loads(json.dumps(artifact))
    producer = altered["source_artifact_hashes"]["producers"]
    producer[next(iter(producer))] = digest("changed")
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="source_hash"):
        cold_reduce(path)
    blocked = _artifact(
        pilots + fixtures,
        fixtures,
        [{"passed": False}],
        receipts,
        [],
        time.monotonic(),
        "20260925",
        {},
    )
    assert blocked["verdict_class"] == "blocked"
    invalid = _artifact(
        pilots + fixtures, fixtures, checks, receipts[:-1], [], time.monotonic(), "20260925", {}
    )
    assert invalid["verdict_class"] == "disqualified"
    path.write_text(json.dumps({**artifact, "rows": pilots}))
    monkeypatch.setattr("carnot.experiment_7658_v668_evidence_atoms.fixture_rows", lambda _: [])
    with pytest.raises(ValueError, match="sample_size"):
        cold_reduce(path)
    _progress("test", "complete", time.monotonic())


def test_parser_abstentions_and_replay_mutations():
    """SCENARIO-REPORT-7658-MUTATIONS: unsupported spans and drift reject."""
    assert parse_atoms("no fence") == []
    assert parse_atoms("```\n\n```") == []
    malformed = "```\n1: def broken(:\n```"
    assert parse_atoms(malformed)[0]["definitions"] == []
    source = "```\n1: def alpha():\n2:     pass\n```"
    answer = "`alpha` is defined at line 1."
    atoms = parse_atoms(source)
    checked = verify_answer(source, answer)
    changed = [dict(atom) for atom in atoms]
    changed[0]["source_id"] = "swapped"
    with pytest.raises(ValueError, match="atom_replay"):
        replay(source, answer, digest(source), digest(answer), changed, checked)
    altered = dict(checked)
    altered["status"] = "unknown"
    with pytest.raises(ValueError, match="claim_replay"):
        replay(source, answer, digest(source), digest(answer), atoms, altered)
    assert verify_answer(source, "`alpha` is not defined at line 1.")["status"] == "unknown"
