"""REQ-VERIFY-7980: public bytes, reservation and fit-only predicate selection."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import evidence_features_7980 as f
from carnot.verify import source_alignment as a


def public(
    fid="one", source="A has 12 units. It is not final. A has 12 units. ", answer="A has 13 units."
):
    return dict(
        family_id=fid, source_bytes=source.encode().hex(), answer_bytes=answer.encode().hex()
    )


def test_features_complete_deduplicated_and_public_only():
    row = public()
    result = f.extract(row)
    view = a.prepare(bytes.fromhex(row["source_bytes"]), bytes.fromhex(row["answer_bytes"]))
    windows = {w.strip(): w for w in view["windows"]}
    pairs = np.array([a.pair_features(w, view["answer_units"])[-4:] for w in windows.values()])
    assert result["values"] == [
        v for i in range(4) for v in (pairs[:, i].mean(), pairs[:, i].max())
    ]
    assert result["window_count"] == len(windows)
    for key in ("y", "role", "quality", "annotations"):
        with pytest.raises(ValueError, match="public_fields"):
            f.extract(dict(row, **{key: 1}))
    for row, reason in [
        (public(source=""), "empty_source"),
        (public(answer=""), "empty_answer"),
        (public(answer="A. " * 17), "answer_units_over_budget"),
        (public(source="A. " * 100), "source_windows_over_budget"),
    ]:
        assert f.extract(row)["abstention"] == reason
        assert f.extract(row)["values"] is None
    with pytest.raises(ValueError, match="public_bytes"):
        f.extract(dict(public(), source_bytes="zz"))
    assert f.normalized("Ａ  ONE\nTwo".encode()) == f.normalized(b"a one two")


def test_reservation_train_group_order_and_no_outcome_dependence():
    sources = [dict(source_id=str(i), source_info=f"Source {i}.") for i in range(110)]
    responses = [
        dict(
            id=f"{i}-{j}",
            source_id=str(i),
            split="train",
            response="Answer.",
            y=j,
            quality="good" if j else "bad",
        )
        for i in range(110)
        for j in range(2)
    ]
    responses.append(dict(id="test", source_id="0", split="test", response="Never select."))
    roster, rows = f.reserve(sources, responses, {f.normalized(b"Source 0.")}, {"1"})
    assert len(roster) == len(rows) == 96
    assert not {"0", "1"} & {r["source_id"] for r in roster}
    assert [r["selection_hash"] for r in roster] == sorted(r["selection_hash"] for r in roster)
    changed = copy.deepcopy(responses)
    for row in changed:
        row["quality"], row["y"], row["labels"] = "truncated", 999, [dict(text="secret")]
    assert f.reserve(sources, changed, {f.normalized(b"Source 0.")}, {"1"}) == (roster, rows)
    assert f.reserve(sources[:1], responses, set(), set())[0][0]["response_id"] == min(
        ("0-0", "0-1"), key=lambda rid: hashlib.sha256(b"seed69280" + rid.encode()).hexdigest()
    )
    assert f.reserve(sources[:1], responses, {hashlib.sha256(b"Source 0.").hexdigest()}, set()) == (
        [],
        [],
    )
    duplicate = sources + [dict(source_id="duplicate", source_info="SOURCE 2. ")]
    extra = responses + [dict(id="dup", source_id="duplicate", split="train", response="Other.")]
    short, _ = f.reserve(duplicate, extra, set(), set(), size=200)
    assert len(short) == 110


def test_predicate_bank_inert_slots_vacancies_and_lexical_ties():
    features = [
        dict(
            family_id=str(i),
            values=[
                i / 30,
                (i % 7) / 7,
                (i % 11) / 11,
                (i % 13) / 13,
                i / 30,
                0.0,
                (i % 5) / 5,
                (i % 3) / 3,
            ],
            abstention=None,
        )
        for i in range(32)
    ]
    fit = [dict(family_id=str(i), y=i % 2, q=0.5, status="completed") for i in range(32)]
    heads = [dict(arm="raw_qwen")]
    bank = f.predicates(features, fit, heads)
    assert len(bank["unary"]) == 16 and len(bank["conjunctions"]) == 8
    assert any(r["inert_reason"] == "duplicate" for r in bank["unary"])
    assert any(r["inert_reason"] == "constant" for r in bank["unary"])
    vectors = [tuple(r["fit_truth"]) for r in bank["selection_trace"]]
    assert len(set(vectors)) == len(vectors)
    assert bank == f.predicates(features, fit, heads)
    fit[0]["y"], fit[1]["q"], fit[2]["status"] = None, None, "excluded"
    assert f.predicates(features, fit, heads)["fit_residual_count"] == 29
    inert = f.predicates([dict(r, values=[0.0] * 8) for r in features], fit, heads)
    assert inert["conjunctions"] == [None] * 8
    empty = f.predicates([], [], heads)
    assert len(empty["unary"]) == 16 and empty["fit_residual_count"] == 0


def test_public_extract_and_cold_replay(tmp_path):
    from carnot.reporting.current_work_receipt import atomic_json

    src, out = tmp_path / "public.json", tmp_path / "features.json"
    atomic_json(src, dict(request_rows=[public(str(i)) for i in range(33)]))
    f.extract_file(src, out)
    f.replay_features(src, out)
    value = json.loads(out.read_text())
    value["rows"][0]["values"][0] += 1
    atomic_json(out, value)
    with pytest.raises(ValueError, match="feature_drift"):
        f.replay_features(src, out)
    atomic_json(src, dict(request_rows=[public(), public()]))
    with pytest.raises(ValueError, match="duplicate_family"):
        f.extract_file(src, out)
