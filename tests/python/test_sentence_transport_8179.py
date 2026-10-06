"""REQ-VERIFY-8179: private transport checks do not supply semantic labels."""

import json

from carnot.verify import sentence_transport_8179 as t


def source(answer="Même fait. Même fait. Tail qualifier applies."):
    """Keep repeated and Unicode text so losing bytes changes the assertion."""
    return dict(
        source_bytes="Même fait. More evidence!".encode().hex(), answer_bytes=answer.encode().hex()
    )


def test_lossless_requests_and_exact_grammar():
    """SCENARIO-VERIFY-8179-TRANSPORT: preserve qualifiers and repeat identities."""
    row = t.requests(source(), len)
    assert row["status"] == "completed"
    for field, parts in [("source_bytes", "source_segments"), ("answer_bytes", "sentences")]:
        data = bytes.fromhex(source()[field])
        assert b"".join(data[p["byte_start"] : p["byte_end"]] for p in row[parts]) == data
    payload = json.loads(row["requests"][0]["payload"])
    assert payload["source"] == bytes.fromhex(source()["source_bytes"]).decode()
    assert payload["answer"] == bytes.fromhex(source()["answer_bytes"]).decode()
    from llama_cpp import LlamaGrammar

    LlamaGrammar.from_string(row["requests"][0]["grammar"], verbose=False)
    assert row["requests"][0]["sentence_indices"] == [0, 1, 2]
    assert (
        t.parse("0|E|0.00|[0]\n1|C|1.00|[0,1]\n2|B|0.32|[]", [0, 1, 2], 2)["status"] == "completed"
    )


def test_escalations_never_shorten_input():
    """REQ-VERIFY-8179: capacity and incomplete text escalate the whole source."""
    row = t.requests(source("Fact. " * 16), len)
    assert len(row["requests"]) == 2
    assert [len(x["sentence_indices"]) for x in row["requests"]] == [8, 8]
    for answer, reason in [
        ("", "missing_sentences"),
        ("Unfinished", "incomplete_sentence"),
        ("Fact. " * 17, "sentence_capacity"),
    ]:
        row = t.requests(source(answer), len)
        assert row["exclusion_reason"] == reason and not row["requests"]
    empty = source()
    empty["source_bytes"] = b" ".hex()
    assert t.requests(empty, len)["exclusion_reason"] == "missing_source"
    assert t.requests(source(), lambda _: 6001)["exclusion_reason"] == "input_token_limit"
    assert t.requests(source(), lambda _: -1)["exclusion_reason"] == "input_token_limit"
    assert t.requests(source(), lambda _: True)["exclusion_reason"] == "input_token_limit"
    assert t.requests(source())["requests"][0]["token_count_measured"] is False


def test_parser_rejects_partial_truncated_or_invalid_records():
    """E2E-010/015: output syntax never substitutes evidence truth."""
    for text in [
        "0|E|0.00|[2]",
        "0|E|0.0|[0]",
        "0|E|1.01|[]",
        "1|E|0.00|[]",
        "0|E|0.00|[-1]",
        "0|E|0.00|[0,1,0]",
        "0|E|0.00|[0",
        "0|E|0.00|[]\n",
        "0|X|0.00|[]",
        "0|E|0.00|[true]",
    ]:
        assert t.parse(text, [0], 2)["status"] == "escalated"
    assert t.parse("0|E|0.00|[]", [0, 1], 2)["status"] == "escalated"
    assert t.parse("", [], 2)["status"] == "escalated"
    parsed = t.parse("0|B|0.99|[]", [0], 2)
    assert parsed["rows"][0]["source_indices"] == []
    assert parsed["semantic_gold"] is False


def test_tokenizer_enumeration_bounds_every_grammar_string():
    """REQ-VERIFY-8179: maximum byte length also bounds noncanonical token paths."""
    budget = t.output_budget(list(range(8, 16)), 120, len)
    assert budget["maximum_encoded_output_tokens"] <= 256
    assert budget["enumerated_count"] == 303
    assert budget["maximum_measured_tokens"] + 1 == budget["maximum_encoded_output_tokens"]
    assert budget["bound_includes_eos"]
    assert t.output_budget(list(range(8)), 10**20, len)["maximum_encoded_output_tokens"] > 256
