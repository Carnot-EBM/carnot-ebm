"""REQ-VERIFY-8166: lossless private sentence and semantic transport checks."""

import json

import pytest

from carnot.verify import sentence_evidence_8166 as s


def test_partition_reconstructs_qualifiers_and_utf8():
    """SCENARIO-VERIFY-8166-PARSER: no qualifier or whitespace disappears."""
    answer = "  Dr. Éva may agree.\nHowever, only if measured! “Yes?”  ".encode()
    rows = s.partition(answer)
    assert len(rows) == 3
    assert b"".join(bytes.fromhex(r["sentence_bytes"]) for r in rows) == answer
    assert rows[0]["byte_start"] == 0 and rows[-1]["byte_end"] == len(answer)
    assert all(r["complete"] for r in rows)
    assert len(s.partition(b"1. Measure 3.14 units. Next result.")) == 2
    assert s.partition(b"No final punctuation")[0]["complete"] is False
    assert s.partition(b"  ") == []
    with pytest.raises(UnicodeError):
        s.partition(b"\xff")


def test_requests_keep_full_text_and_escalate():
    """REQ-VERIFY-8166: never truncate to fit a request or fabricate a label."""
    row = dict(source_bytes="évidence fact.".encode().hex(), answer_bytes=b"One. Two.".hex())
    result = s.requests(row)
    assert result["status"] == "completed"
    prompt = json.loads(result["requests"][0]["payload"])
    assert prompt["source"] == "évidence fact." and prompt["answer"] == "One. Two."
    assert len(prompt["sentences"]) == 2
    assert result["requests"][0]["maximum_output_tokens"] == 256
    assert s.requests(row, lambda _: 6000)["status"] == "completed"
    assert s.requests(row, lambda _: 6001)["exclusion_reason"] == "input_token_limit"
    for answer, reason in [
        (b"", "missing_sentences"),
        (b"Fragment", "incomplete_sentence"),
        (b"Fact. " * 9, "sentence_capacity"),
    ]:
        assert s.requests(dict(row, answer_bytes=answer.hex()))["exclusion_reason"] == reason
    assert len(s.requests(dict(row, answer_bytes=b"Fact. ".hex() * 5))["requests"]) == 2
    assert s.requests(dict(row, source_bytes=""))["exclusion_reason"] == "missing_source"


def test_parser_checks_semantics_and_quote_custody_separately():
    """REQ-VERIFY-8166: a valid substring cannot establish entailment."""
    source = "évidence fact.".encode()
    item = dict(
        sentence_index=0,
        p_unsupported=0.2,
        relation="entailed",
        quote="fact",
        byte_start=10,
        byte_end=14,
    )
    parsed = s.parse(json.dumps([item]), source, [0])
    assert parsed["status"] == "completed" and parsed["rows"][0]["quote_valid"]
    assert parsed["quote_validity_is_entailment"] is False
    absent = dict(item, quote=None, byte_start=None, byte_end=None)
    assert s.parse(json.dumps([absent]), source, [0])["status"] == "completed"
    for patch in [
        dict(p_unsupported=True),
        dict(p_unsupported=2),
        dict(p_unsupported=float("nan")),
        dict(sentence_index=True),
        dict(sentence_index=1),
        dict(relation="unknown"),
    ]:
        assert s.parse(json.dumps([dict(item, **patch)]), source, [0])["status"] == "escalated"
    for payload in ["bad", "{}", "[]", json.dumps([item, item])]:
        assert s.parse(payload, source, [0])["status"] == "escalated"
    missing_quote = dict(item)
    missing_quote.pop("quote")
    assert s.parse(json.dumps([missing_quote]), source, [0])["status"] == "escalated"
    assert s.parse(json.dumps([item, item]), source, [0, 0])["status"] == "escalated"
    for patch in [
        dict(byte_start=9),
        dict(quote="fake"),
        dict(byte_end=True),
        dict(quote=None),
        dict(quote=""),
    ]:
        got = s.parse(json.dumps([dict(item, **patch)]), source, [0])
        assert got["status"] == "escalated"
    assert s.parse(json.dumps([item]), b"altered source", [0])["status"] == "escalated"


def test_local_features_require_complete_original_sentences():
    """SCENARIO-VERIFY-8166-PARSER: relation predictions never inherit targets."""
    rows = [
        dict(sentence_index=i, p_unsupported=p, relation=r)
        for i, p, r in [(0, 0.1, "entailed"), (1, 0.9, "contradicted"), (2, 0.5, "baseless")]
    ]
    assert s.local_features(rows, 3) == [0.5, 0.9, 1 / 3, 1 / 3]
    assert s.local_features(rows[:1], 3) is None
    assert s.local_features([], 0) is None
    assert s.local_features([rows[0], rows[0]], 2) is None
