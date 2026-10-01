"""REQ-VERIFY-7995, REQ-REPORT-7995: owned calls and independent support gates."""

import copy
import json
from pathlib import Path

import pytest

from carnot.verify import qwen_development_capture_7995 as c
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from scripts.adversarial_verify import _classify_current_task_inference_claim


def views():
    return {
        role: dict(
            request_rows=[
                dict(
                    family_id=f"{role}-{i}",
                    source_bytes=f"{role} source {i}.".encode().hex(),
                    answer_bytes=b"Answer.".hex(),
                )
                for i in range(n)
            ],
            features=[
                dict(family_id=f"{role}-{i}", abstention=None, source_normalized_hash=f"{role}-{i}")
                for i in range(n)
            ],
        )
        for role, n in c.ROLES.items()
    }


class Runtime:
    def count(self, text):
        return 30

    def generate(self, request):
        return dict(
            model=c.risk.MODEL,
            choices=[
                dict(
                    message=dict(
                        content=json.dumps(
                            dict(unsupported_probability=0.25, source_sentence_id=None)
                        )
                    ),
                    finish_reason="stop",
                )
            ],
            usage=dict(completion_tokens=18, prompt_tokens=30),
        )


def test_frozen_protocol_and_independent_roles():
    frozen = c.freeze(views())
    assert len(frozen) == 384
    assert frozen[0]["request"]["max_tokens"] == 96
    assert frozen[0]["request"]["messages"][0]["content"] == c.risk.SYSTEM
    altered = views()
    altered["stream"]["features"][0]["source_normalized_hash"] = "calibration-0"
    with pytest.raises(ValueError, match="cross_role_overlap"):
        c.freeze(altered)
    with pytest.raises(ValueError, match="role_roster"):
        c.freeze({})
    altered = views()
    altered["calibration"]["request_rows"].pop()
    with pytest.raises(ValueError, match="role_count"):
        c.freeze(altered)


def test_capture_four_calls_and_final_provenance(tmp_path):
    slots = c.freeze(views())[:4]
    ledger = c.Ledger(tmp_path / "ledger.json")
    ledger.start("model_load", "private-load", {})
    ledger.finish("private-load", "completed", {})
    rows = c.capture(slots, Runtime(), tmp_path / "slots", "identity", ledger=ledger)
    value = c.provenance(ledger, rows, live=True)
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 4
    assert value["model_invocation_counts"]["generation_calls_completed"] == 4
    assert (
        _classify_current_task_inference_claim(json.loads(json.dumps(value)))["state"]
        == "live_inference"
    )
    assert c.reduce(rows)["sample_size_budget"]["intended"] == 384
    assert all(row["denominator"] == 1 for row in rows)
    assert sum(row["numerator"] for row in rows) == 4
    assert c.reduce(rows)["capture_ready_score"] == 0
    broken = copy.deepcopy(rows)
    broken[0]["parsed"] = {}
    with pytest.raises(ValueError, match="parse_drift"):
        c.reduce(broken)
    with pytest.raises(ValueError, match="slot_roster"):
        c.reduce(rows + rows)


def test_interrupted_resumed_and_mixed_scope(tmp_path):
    slot = c.freeze(views())[0]
    ledger = c.Ledger(tmp_path / "ledger.json")
    row = dict(
        slot,
        capture_identity="identity",
        started=True,
        status="running",
        raw_response={},
        reserved_tokens=96,
        input_tokens=30,
        duration_s=0,
    )
    ledger.start("generation", slot["family_id"], slot["request"])
    atomic_json(tmp_path / "slots/slot-000.json", row)
    rows = c.capture([slot], Runtime(), tmp_path / "slots", "identity", ledger=ledger)
    assert rows[0]["error"] == "interrupted_uncertain_no_retry"
    assert ledger.counts()["generation_calls_failed"] == 1
    assert len(c.capture([slot], Runtime(), tmp_path / "slots", "identity", ledger=ledger)) == 1
    with pytest.raises(ValueError, match="checkpoint_identity"):
        c.capture([slot], Runtime(), tmp_path / "slots", "other", ledger=ledger)
    with pytest.raises(ValueError, match="duplicate_call"):
        ledger.start("generation", slot["family_id"], {})
    changed = ledger.rows[0].copy()
    changed["scope"] = "historical"
    ledger.rows.append(changed)
    with pytest.raises(ValueError, match="ledger_scope"):
        ledger.counts()


def test_censoring_failure_and_token_admission(tmp_path):
    slots = c.freeze(views())[:4]
    ledger = c.Ledger(tmp_path / "ledger.json")
    rows = c.capture(slots, Runtime(), tmp_path / "censored", "id", ledger=ledger, deadline_s=0)
    assert all(r["status"] == "censored" for r in rows)
    assert ledger.counts()["generation_calls_attempted"] == 0
    slots[0]["public_eligible"] = False
    runtime = Runtime()
    runtime.count = lambda _: 6001
    rows = c.capture(slots, runtime, tmp_path / "excluded", "id", ledger=ledger)
    assert all(r["status"] == "excluded" for r in rows)
    runtime.count = lambda _: (_ for _ in ()).throw(ValueError("tokenizer"))
    rows = c.capture(slots[1:2], runtime, tmp_path / "tokenfail", "id", ledger=ledger)
    assert rows[0]["status"] == "failed"
    runtime.count = lambda _: 30
    runtime.generate = lambda _: (_ for _ in ()).throw(TimeoutError("uncertain"))
    rows = c.capture(slots[1:2], runtime, tmp_path / "fail", "id", ledger=ledger)
    assert rows[0]["status"] == "failed"
    assert ledger.counts()["generation_calls_failed"] == 1
    assert c.reduce(rows)["censor_rows"]


def test_role_floors_are_support_only(tmp_path):
    ledger = c.Ledger(tmp_path / "ledger.json")
    rows = c.capture(c.freeze(views()), Runtime(), tmp_path / "slots", "id", ledger=ledger)
    reduced = c.reduce(rows)
    assert reduced["capture_ready_score"] == 1
    assert all(reduced[f"{role}_capture_ready_score"] == 1 for role in c.ROLES)
    rows[0]["raw_response"] = {}
    rows[0]["parsed"] = c.risk.transport.parse_response({}, rows[0]["visible_ids"])
    rows[0]["status"] = "censored"
    assert c.reduce(rows)["capture_ready_score"] == 1
    assert c.config()["output_tokens"] == 36864
    assert canonical_hash(c.config()).startswith("sha256:")


def test_receipt_bindings_and_terminal_mutations(tmp_path):
    """REQ-REPORT-7995: changed request/response bytes disqualify custody."""
    ledger = c.Ledger(tmp_path / "ledger.json")
    rows = c.capture(c.freeze(views())[:1], Runtime(), tmp_path / "slots", "id", ledger=ledger)
    with pytest.raises(ValueError, match="duplicate_terminal"):
        ledger.finish(rows[0]["family_id"], "completed", {})
    rows[0]["started"] = False
    with pytest.raises(ValueError, match="ledger_raw_count"):
        c.provenance(ledger, rows, live=True)
    rows[0]["started"] = True
    rows[0]["raw_response"]["usage"]["completion_tokens"] = 19
    with pytest.raises(ValueError, match="ledger_response_binding"):
        c.provenance(ledger, rows, live=True)


def test_request_and_terminal_bindings(tmp_path):
    ledger = c.Ledger(tmp_path / "ledger.json")
    rows = c.capture(c.freeze(views())[:1], Runtime(), tmp_path / "slots", "id", ledger=ledger)
    original = rows[0]["request"]["max_tokens"]
    rows[0]["request"]["max_tokens"] = 95
    with pytest.raises(ValueError, match="ledger_request_binding"):
        c.provenance(ledger, rows, live=True)
    rows[0]["request"]["max_tokens"] = original
    rows[0]["status"] = "failed"
    with pytest.raises(ValueError, match="ledger_terminal_binding"):
        c.provenance(ledger, rows, live=True)
