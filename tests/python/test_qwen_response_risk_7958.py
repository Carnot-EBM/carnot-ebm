"""REQ-VERIFY-7958: measure the same complete event with independent labels."""

import json
from pathlib import Path

import pytest

from carnot.verify import qwen_response_risk_7958 as risk
from carnot.verify import qwen_completion_7932 as transport


def public(n=64):
    return [
        dict(
            family_id=str(i),
            source_bytes=f"Source {i}. Café.".encode().hex(),
            answer_bytes=b"First. Last unsupported span.".hex(),
        )
        for i in range(n)
    ]


def reply(p=0.1, finish="stop", tokens=20):
    return dict(
        model=risk.MODEL,
        choices=[
            dict(
                finish_reason=finish,
                message=dict(
                    content=json.dumps(dict(unsupported_probability=p, source_sentence_id=None))
                ),
            )
        ],
        usage=dict(prompt_tokens=30, completion_tokens=tokens),
    )


class Runtime:
    def __init__(self, response=None):
        self.calls = []
        self.response = reply() if response is None else response

    def generate(self, payload):
        self.calls.append(payload)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def labels(n=64):
    return [
        dict(
            family_id=str(i),
            source_cluster_id=str(i),
            y=i % 2,
            implicit_true_excluded_y=i % 2,
            annotation_count=i % 2,
        )
        for i in range(n)
    ]


def test_full_response_fixed_event_and_public_admission():
    """SCENARIO-VERIFY-7958-QUERY: no sentence selection or outcome tuning."""
    frozen = risk.freeze(public(), lambda s: 30)
    assert len(frozen) == 64
    for row in frozen:
        for arm in risk.ARMS:
            request = row["requests"][arm]
            assert request["grammar"] == transport.GRAMMAR
            assert request["seed"] == 69058 and request["max_tokens"] == 96
            body = json.loads(request["messages"][1]["content"])
            assert body["original_answer"] == "First. Last unsupported span."
            assert body["complete_source"] == (
                bytes.fromhex(row["source_bytes"]).decode() if arm == "full_source" else ""
            )
    assert risk.freeze(public(1), lambda s: 6001)[0]["eligible"] is False
    with pytest.raises(ValueError, match="public_fields"):
        risk.freeze([{**public(1)[0], "y": 1}], len)
    with pytest.raises(ValueError, match="duplicate"):
        risk.freeze(public(1) * 2, len)


def test_capture_seals_every_pair_and_never_retries(tmp_path):
    """SCENARIO-VERIFY-7958-QUERY: all intended cells keep raw evidence."""
    runtime = Runtime()
    rows = risk.capture(risk.freeze(public(2), len), runtime, tmp_path)
    assert len(rows) == len(runtime.calls) == 4
    assert len(list(tmp_path.glob("pair-*.json"))) == 2
    assert all(r["parsed"]["completed"] for r in rows)
    for response in ({}, reply(finish="length"), reply(tokens=97), RuntimeError("failed")):
        runtime = Runtime(response)
        rows = risk.capture(risk.freeze(public(1), len), runtime, tmp_path)
        assert len(runtime.calls) == 2
        assert not any(r["parsed"]["completed"] for r in rows)
    for kw in (dict(token_budget=0), dict(deadline_s=-1)):
        runtime = Runtime()
        rows = risk.capture(risk.freeze(public(1), len), runtime, tmp_path, **kw)
        assert len(rows) == 2 and not runtime.calls
    runtime = Runtime()
    rows = risk.capture(risk.freeze(public(1), lambda s: 6001), runtime, tmp_path)
    assert not runtime.calls and all(r["status"] == "excluded" for r in rows)


@pytest.mark.parametrize(
    "p,decision",
    [(0.0, "accept"), (0.05, "escalate"), (0.5, "escalate"), (0.75, "escalate"), (1.0, "reject")],
)
def test_typed_cost_ties(p, decision):
    """SCENARIO-VERIFY-7958-REDUCTION: ties preserve the escalation policy."""
    assert risk.decision(p) == decision
    assert risk.decision(None) == "escalate"


def test_cluster_comparison_capacity_and_null(tmp_path):
    """SCENARIO-VERIFY-7958-REDUCTION: syntax does not give accuracy."""
    rows = risk.capture(risk.freeze(public(), len), Runtime(), tmp_path)
    result = risk.reduce(rows, labels())
    assert result["sample_size_budget"]["intended"] == 128
    assert result["probability_metrics"]["complete_pairs"] == 64
    assert result["qwen_response_benefit_score"] == 0
    assert result["confidence_intervals"]["brier"] == [0.0, 0.0]
    assert risk.reduce(rows, labels(2))["comparison_status"] == "insufficient_data"
    assert risk.reduce([], [])["comparison_status"] == "insufficient_data"
    rows[0]["raw_response"] = reply(finish="length")
    rows[0]["parsed"] = transport.parse_response(rows[0]["raw_response"], rows[0]["visible_ids"])
    reduced = risk.reduce(rows, labels())
    assert reduced["probability_metrics"]["complete_pairs"] == 63
    assert reduced["typed_cost_rows"][0]["decision"] == "escalate"
    with pytest.raises(ValueError, match="duplicate"):
        risk.reduce(rows + rows[:1], labels())
    rows[0]["parsed"]["probability"] = 0.9
    with pytest.raises(ValueError, match="parse_drift"):
        risk.reduce(rows, labels())


def test_natural_benefit_and_cluster_averaging(tmp_path):
    """SCENARIO-VERIFY-7958-REDUCTION: all registered gates are necessary."""
    rows = risk.capture(risk.freeze(public(), len), Runtime(), tmp_path)
    for row in rows:
        p = int(row["family_id"]) % 2 if row["arm"] == "full_source" else 0.5
        row["raw_response"] = reply(p)
        row["parsed"] = transport.parse_response(row["raw_response"], row["visible_ids"])
    result = risk.reduce(rows, labels())
    assert result["qwen_response_benefit_score"] == 1
    assert result["probability_metrics"]["brier_gain"] == 0.25
    grouped = labels()
    for row in grouped:
        row["source_cluster_id"] = str(int(row["family_id"]) // 2)
    assert risk.reduce(rows, grouped)["sample_size_budget"]["independent"] == 32
