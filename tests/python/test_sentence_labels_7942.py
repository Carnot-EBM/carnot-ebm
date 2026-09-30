"""Sentence transport checks for REQ-VERIFY-7942 and REQ-REPORT-7942."""

from copy import deepcopy
import hashlib
import json

import pytest

from carnot.verify import sentence_labels_7942 as labels


def example(text="Café. Wrong. Last.", spans=None, quality="good"):
    """Keep public bytes separate from the human evaluator fixture."""
    public = {
        "family_id": "opaque",
        "source_bytes": "Café. Last.".encode().hex(),
        "answer_bytes": text.encode().hex(),
    }
    response = {
        "id": "r",
        "source_id": "s",
        "response": text,
        "quality": quality,
        "labels": [] if spans is None else spans,
        "model": "cached-human-corpus",
    }
    evaluator = {"family_id": "opaque", "response_id": "r", "role": "evaluation"}
    role = {
        "family_id": "opaque",
        "role": "evaluation",
        "status": "completed",
        "source_cluster_id": labels.digest(bytes.fromhex(public["source_bytes"])),
    }
    source = {"source_id": "s", "source_info": "Café. Last."}
    return public, response, evaluator, role, source


def span(start, end, text, implicit=False, null=False):
    """Use the original annotation fields, including sensitivity metadata."""
    return dict(
        start=start,
        end=end,
        text=text,
        implicit_true=implicit,
        due_to_null=null,
        label_type="Evident Baseless Info",
        meta="human",
    )


def joined(data):
    """Freeze the public query before handing the evaluator to the join."""
    p, r, e, role, s = data
    frozen, bounds = labels.freeze([p])
    return labels.join(frozen, bounds, [e], [role], [r], [s])


def test_unicode_cross_boundary_and_overlapping_annotations():
    """SCENARIO-VERIFY-7942-OFFSETS: character offsets become exact bytes."""
    data = example(spans=[span(3, 9, "é. Wro"), span(7, 11, "rong")])
    rows, offsets = joined(data)
    assert rows[0]["sentence_labels"] == [1, 1, 0]
    assert offsets[0]["start_byte"] == 3
    assert offsets[0]["end_byte"] == 10
    assert offsets[0]["text_equal"] is True
    assert offsets[0]["start_char"] == 3


def test_boundary_whitespace_no_spans_and_sensitivity():
    """SCENARIO-VERIFY-7942-OFFSETS: no response-wide label broadcast."""
    assert joined(example())[0][0]["sentence_labels"] == [0, 0, 0]
    assert joined(example(spans=[span(5, 6, " ")]))[0][0]["sentence_labels"] == [0, 0, 0]
    rows, _ = joined(example(spans=[span(6, 11, "Wrong", implicit=True, null=True)]))
    assert rows[0]["sentence_labels"] == [0, 1, 0]
    assert rows[0]["implicit_true_excluded_sentence_labels"] == [0, 0, 0]
    rows, _ = joined(example(spans=[span(6, 11, "Wrong", null=True)]))
    assert rows[0]["implicit_true_excluded_sentence_labels"] == [0, 1, 0]
    nullable_note = {**span(6, 11, "Wrong"), "meta": None}
    assert joined(example(spans=[nullable_note]))[0][0]["sentence_labels"] == [0, 1, 0]
    for quality in ("incorrect_refusal", "truncated"):
        rows, _ = joined(example(quality=quality))
        assert rows[0]["y"] is None and rows[0]["status"] == "excluded"
    p, r, e, role, s = example()
    role["status"] = "excluded"
    rows, _ = joined((p, r, e, role, s))
    assert rows[0]["status"] == "excluded" and rows[0]["y"] is None


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"quality": None}, "quality"),
        ({"quality": "bad"}, "quality"),
        ({"labels": None}, "annotations"),
        ({"response": "changed"}, "response_equality"),
        ({"labels": [{}]}, "span_shape"),
        ({"labels": [span(-1, 1, "C")]}, "span_offsets"),
        ({"labels": [span(True, 1, "C")]}, "span_offsets"),
        ({"labels": [span(1, 0, "")]}, "span_offsets"),
        ({"labels": [span(0, 999, "C")]}, "span_offsets"),
        ({"labels": [span(0, 1, "X")]}, "span_text"),
        ({"labels": [span(0, 0, "")]}, "span_offsets"),
        ({"labels": [span(0, 1, "C", implicit=1)]}, "span_shape"),
    ],
)
def test_fail_closed_spans(mutation, reason):
    """REQ-VERIFY-7942: unknown or malformed labels never become negatives."""
    p, r, e, role, s = example()
    r.update(mutation)
    with pytest.raises(ValueError, match=reason):
        joined((p, r, e, role, s))


def test_freeze_exact_rule_metadata_mutation_and_public_rejection():
    """REQ-VERIFY-7942: all metadata mutations preserve bytes and features."""
    p, r, e, role, s = example()
    frozen, bounds = labels.freeze([p])
    intervals = bounds[0]["intervals"]
    expected = min(
        intervals,
        key=lambda interval: hashlib.sha256(
            p["family_id"].encode()
            + json.dumps(interval, separators=(",", ":")).encode()
            + b"v689-sentence-1"
        ).digest(),
    )
    assert frozen[0]["sentence_interval"] == expected
    assert set(frozen[0]) == labels.PREDICTOR_KEYS
    metadata = {**r, **e, **role, "y": 999, "annotations": [{"start": -99}]}
    for key in set(metadata) - labels.PUBLIC_KEYS:
        changed = {**p, key: deepcopy(metadata[key])}
        assert labels.freeze([labels.public_only(changed)]) == (frozen, bounds)
    assert labels.features(frozen[0]) == labels.features(labels.freeze([p])[0][0])
    with pytest.raises(ValueError, match="public_fields"):
        labels.freeze([{**p, "quality": "good"}])
    with pytest.raises(ValueError, match="duplicate"):
        labels.freeze([p, p])
    with pytest.raises(ValueError, match="empty_answer"):
        labels.freeze([{**p, "answer_bytes": ""}])


def test_identity_rosters_clusters_and_source_bytes():
    """REQ-VERIFY-7942: ambiguous identities and role leaks fail custody."""
    p, r, e, role, s = example()
    frozen, bounds = labels.freeze([p])
    for which in ("response", "evaluator", "source", "role"):
        args = [frozen, bounds, [e], [role], [r], [s]]
        index = {"response": 4, "evaluator": 2, "source": 5, "role": 3}[which]
        args[index] = args[index] * 2
        with pytest.raises(ValueError, match="duplicate"):
            labels.join(*args)
    for index in (2, 3, 4, 5):
        args = [frozen, bounds, [e], [role], [r], [s]]
        args[index] = []
        with pytest.raises(ValueError, match="missing_join|roster"):
            labels.join(*args)
    with pytest.raises(ValueError, match="source_equality"):
        labels.join(frozen, bounds, [e], [role], [r], [{**s, "source_info": "changed"}])
    p2 = {**p, "family_id": "other"}
    e2 = {**e, "family_id": "other", "response_id": "r2", "role": "fit"}
    r2 = {**r, "id": "r2"}
    role2 = {**role, "family_id": "other", "role": "fit"}
    frozen2, bounds2 = labels.freeze([p, p2])
    with pytest.raises(ValueError, match="source_cluster_role"):
        labels.join(frozen2, bounds2, [e, e2], [role, role2], [r, r2], [s])
    with pytest.raises(ValueError, match="response_reused"):
        labels.join(
            frozen2,
            bounds2,
            [e, {**e2, "response_id": "r"}],
            [role, {**role2, "role": "evaluation"}],
            [r],
            [s],
        )
    with pytest.raises(ValueError, match="role_drift"):
        labels.join(frozen, bounds, [{**e, "role": "fit"}], [role], [r], [s])
    with pytest.raises(ValueError, match="source_cluster_hash"):
        labels.join(frozen, bounds, [e], [{**role, "source_cluster_id": "bad"}], [r], [s])
    with pytest.raises(ValueError, match="boundary_drift"):
        labels.join(frozen, [{**bounds[0], "intervals": [[0, 1]]}], [e], [role], [r], [s])
    # Structured source data uses the pinned corpus serialization convention.
    obj = {"passages": ["Café."], "question": "why?"}
    p["source_bytes"] = (
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode().hex()
    )
    role["source_cluster_id"] = labels.digest(bytes.fromhex(p["source_bytes"]))
    assert joined((p, r, e, role, {**s, "source_info": obj}))[0]


def test_reduce_cohort_counts_and_failed_operands():
    """SCENARIO-VERIFY-7942-COHORT: repeated rows are not independent samples."""
    row = joined(example())[0][0]
    cohort = [
        {**row, "family_id": str(i), "source_cluster_id": str(i), "y": i % 2} for i in range(64)
    ]
    reduced = labels.reduce_rows(cohort)
    assert reduced["sentence_labels_ready_score"] == 1
    assert reduced["label_counts"] == {"0": 32, "1": 32}
    assert reduced["sample_size_budget"]["independent"] == 64
    assert labels.reduce_rows([])["sentence_labels_ready_score"] == 0
    for mutation in (
        [{**r, "y": 0} for r in cohort],
        [{**r, "source_cluster_id": "one"} for r in cohort],
        cohort[:32],
        [{**r, "status": "excluded", "y": None} for r in cohort],
    ):
        reduced = labels.reduce_rows(mutation)
        assert reduced["sentence_labels_ready_score"] == 0
        assert reduced["failed_operands"]
    with pytest.raises(ValueError, match="duplicate"):
        labels.reduce_rows([row, row])
