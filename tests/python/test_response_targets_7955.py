"""Whole-response custody and capacity checks for REQ-VERIFY-7955."""

from copy import deepcopy

import pytest

from carnot.verify import response_targets_7955 as targets
from test_sentence_labels_7942 import example, span


def data(value=None):
    """Use the established exact-byte fixture without its sentence selection."""
    p, r, e, role, s = example() if value is None else value
    e["human_label"] = int(bool(r.get("labels")))
    return dict(public=[p], responses=[r], evaluators=[e], roles=[role], sources=[s])


def test_whole_response_union_and_all_sentence_audit():
    """SCENARIO-VERIFY-7955-CUSTODY: every span contributes to the union."""
    value = data(example(spans=[span(3, 9, "é. Wro"), span(7, 11, "rong")]))
    public, audit = targets.freeze(value["public"])
    rows, annotations = targets.join(public, audit, value)
    assert public == value["public"] and len(audit[0]["intervals"]) == 3
    assert rows[0]["y"] == 1 and rows[0]["annotation_count"] == 2
    assert annotations[0]["end_byte"] == 10
    for spans, y, sensitivity in (
        ([], 0, 0),
        ([span(6, 11, "Wrong", implicit=True, null=True)], 1, 0),
        ([span(6, 11, "Wrong", null=True)], 1, 1),
        ([span(5, 6, " ")], 1, 1),
    ):
        d = data(example(spans=spans))
        p, a = targets.freeze(d["public"])
        row = targets.join(p, a, d)[0][0]
        assert (row["y"], row["implicit_true_excluded_y"]) == (y, sensitivity)
    d["evaluators"][0]["human_label"] = 0
    assert targets.join(p, a, d)[0][0]["original_label_disagreement"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        {"quality": "truncated"},
        {"quality": "incorrect_refusal"},
        {"quality": "bad"},
        {"labels": None},
        {"response": "changed"},
        {"labels": [span(-1, 1, "C")]},
        {"labels": [span(0, 1, "X")]},
    ],
)
def test_missing_bad_quality_or_offsets_never_supply_negative(mutation):
    """SCENARIO-VERIFY-7955-CUSTODY: incomplete annotation means unknown."""
    d = data()
    d["responses"][0].update(mutation)
    p, a = targets.freeze(d["public"])
    row = targets.join(p, a, d)[0][0]
    assert row["y"] is None and row["status"] == "excluded"


def test_public_mutations_budgets_and_identity_failures():
    """SCENARIO-VERIFY-7955-CUSTODY: metadata cannot admit another response."""
    d = data()
    p, a = targets.freeze(d["public"])
    for key in ("labels", "response_id", "offsets", "annotation_order", "human_label"):
        changed = [{**p[0], key: "mutated"}]
        assert targets.freeze([targets.public_only(r) for r in changed]) == (p, a)
    with pytest.raises(ValueError, match="public_fields"):
        targets.freeze(changed)
    with pytest.raises(ValueError, match="duplicate"):
        targets.freeze(p + p)
    for key in ("responses", "evaluators", "sources", "roles"):
        changed = deepcopy(d)
        changed[key] *= 2
        with pytest.raises(ValueError, match="duplicate"):
            targets.join(p, a, changed)
    changed = deepcopy(d)
    changed["responses"] = []
    assert targets.join(p, a, changed)[0][0]["y"] is None
    changed = deepcopy(d)
    changed["sources"][0]["source_info"] = "changed"
    assert targets.join(p, a, changed)[0][0]["custody_passed"] is False
    changed = deepcopy(d)
    changed["roles"][0]["source_cluster_id"] = "wrong"
    with pytest.raises(ValueError, match="source_cluster"):
        targets.join(p, a, changed)
    changed = deepcopy(d)
    changed["evaluators"][0]["role"] = "fit"
    with pytest.raises(ValueError, match="role"):
        targets.join(p, a, changed)
    with pytest.raises(ValueError, match="public_drift"):
        targets.join(p, [], d)
    with pytest.raises(ValueError, match="roster"):
        targets.join(p, a, {**d, "evaluators": []})
    with pytest.raises(ValueError, match="roster"):
        targets.check_roles(p, [])
    p2 = {**p[0], "family_id": "another"}
    r2 = {**d["roles"][0], "family_id": "another", "role": "fit"}
    with pytest.raises(ValueError, match="source_cluster_role"):
        targets.check_roles([p[0], p2], [d["roles"][0], r2])
    over = {**p[0], "answer_bytes": ("A. " * 17).encode().hex()}
    assert targets.freeze([over])[1][0]["public_eligible"] is False
    over = {**p[0], "source_bytes": ("A. " * 66).encode().hex()}
    assert targets.freeze([over])[1][0]["public_eligible"] is False
    assert targets.freeze([{**p[0], "answer_bytes": ""}])[1][0]["public_eligible"] is False


def test_response_capacity_and_independent_cluster_counts():
    """SCENARIO-VERIFY-7955-CAPACITY: sentences cannot inflate the sample."""
    d = data()
    row = targets.join(*targets.freeze(d["public"]), d)[0][0]
    rows = [
        {**row, "family_id": str(i), "source_cluster_id": str(i), "y": i % 2} for i in range(64)
    ]
    result = targets.reduce_rows(rows)
    assert result["response_targets_ready_score"] == 1
    assert result["class_counts"] == {"0": 32, "1": 32}
    assert result["sample_size_budget"]["unit"] == "complete_evaluation_response"
    assert targets.reduce_rows([])["response_targets_ready_score"] == 0
    assert targets.reduce_rows([{**r, "y": 0} for r in rows])["response_targets_ready_score"] == 0
    assert (
        targets.reduce_rows([{**r, "source_cluster_id": "one"} for r in rows])[
            "response_targets_ready_score"
        ]
        == 0
    )
