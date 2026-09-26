"""REQ-REPORT-7673 and REQ-VERIFY-7673 cohort boundary tests."""

from copy import deepcopy

import pytest

from carnot.reporting import fresh_relation_cohort as cohort


def row(split: str, source: str, answer: str, instance: str) -> dict:
    """Build a public row without evaluator-only label columns."""
    return {
        "official_split": split,
        "context": source,
        "question": "Where is the named definition?",
        "answer": answer,
        "metadata": {"instance_id": instance, "tool_type": "code"},
    }


def test_scenario_report_7673_custody_clusters_siblings_and_split_collision():
    # SCENARIO-REPORT-7673-CUSTODY: answer siblings and source templates stay together.
    public = [
        row("train", "1 def apple(): pass", "`apple` at line 1", "a"),
        row("train", "1  def apple(): pass", "`apple` on line 1", "b"),
        row("train", "1 def pear(): pass", "`pear` at line 1", "c"),
        row("test", "1 def pear(): pass", "`pear` at line 1", "d"),
    ]
    families, collisions = cohort.cluster_public_rows(public)
    assert len(families) == 1
    assert families[0]["member_count"] == 2
    assert collisions[0]["reason"] == "cross_official_split_family"
    assert collisions[0]["member_count"] == 2


def test_scenario_report_7673_custody_excludes_missing_question():
    # SCENARIO-REPORT-7673-CUSTODY: a source without a question cannot become a view.
    incomplete = row("train", "source", "answer", "missing")
    incomplete["question"] = None
    families, exclusions = cohort.cluster_public_rows([incomplete])
    assert families == []
    assert exclusions[0]["reason"] == "complete_question_missing"


def test_scenario_report_7673_custody_exposure_and_salted_roles():
    # REQ-REPORT-7673: exposure is subtracted before fixed role assignment.
    families = [
        {"family_id": str(i), "official_split": "train", "source_hashes": [str(i)]}
        for i in range(5)
    ] + [{"family_id": "test", "official_split": "test", "source_hashes": ["test"]}]
    eligible, excluded = cohort.subtract_exposure(
        families,
        {"source_hashes": {"0"}, "answer_hashes": set(), "family_ids": set()},
    )
    assert len(eligible) == 5
    assert excluded == [{"family_id": "0", "reason": "prior_exposure"}]
    counts = {"fit": 2, "tune": 1, "evaluation": 1}
    a = cohort.assign_roles(eligible, counts, "v669-test-salt")
    b = cohort.assign_roles(list(reversed(eligible)), counts, "v669-test-salt")
    assert a == b
    assert {x["role"] for x in a if x["official_split"] == "test"} == {"evaluation"}
    with pytest.raises(ValueError, match="underfilled"):
        cohort.assign_roles(eligible, {"fit": 5, "evaluation": 1}, "v669-test-salt")


def test_scenario_report_7673_isolation_rejects_labels_and_role_drift():
    # SCENARIO-REPORT-7673-ISOLATION: no label-bearing metadata reaches features.
    item = cohort.predictor_view(
        row("train", "1 def apple(): pass", "`apple` at line 1", "a"), "family", "fit"
    )
    cohort.validate_predictor(item, "fit")
    for key, value in (("label", 1), ("metadata", {"is_hallucinated": True})):
        broken = deepcopy(item)
        broken[key] = value
        with pytest.raises(ValueError, match="label|field"):
            cohort.validate_predictor(broken, "fit")
    with pytest.raises(ValueError, match="role"):
        cohort.validate_predictor(item, "evaluation")
    broken = deepcopy(item)
    broken["complete_source"] += "\nextra"
    with pytest.raises(ValueError, match="hash"):
        cohort.validate_predictor(broken, "fit")


def test_scenario_verify_7673_controls_keep_one_denominator():
    # SCENARIO-VERIFY-7673-CONTROLS: all views retain one answer and family.
    a = cohort.predictor_view(
        row("train", "1 def apple(): pass", "`apple` at line 1", "a"), "a", "fit"
    )
    b = cohort.predictor_view(
        row("train", "1 def pear(): pass", "`apple` at line 1", "b"), "b", "fit"
    )
    rows = cohort.feature_rows([a, b], "fit")
    assert len(rows) == 6
    assert {x["unit_id"] for x in rows} == {"a", "b"}
    assert all(x["answer_sha256"] == a["answer_sha256"] for x in rows if x["unit_id"] == "a")
    erased = next(x for x in rows if x["unit_id"] == "a" and x["arm"] == "evidence_erasure")
    deranged = next(
        x for x in rows if x["unit_id"] == "a" and x["arm"] == "within_role_derangement"
    )
    assert erased["source_atom_count"] == 0 and erased["censored"]
    assert deranged["source_group_id"] == "b"
    with pytest.raises(ValueError, match="derangement"):
        cohort.feature_rows([a], "fit")


def test_scenario_verify_7673_rejects_all_boundary_drift():
    # SCENARIO-VERIFY-7673-CONTROLS: every rejected identity has its own witness.
    base = row("train", "source", "answer", "a")
    text_metadata = deepcopy(base)
    text_metadata["metadata"] = '{"instance_id": "a", "is_hallucinated": true}'
    assert cohort.cluster_public_rows([text_metadata])[0][0]["instance_ids"] == ["a"]
    absent_metadata = deepcopy(base)
    absent_metadata["metadata"] = None
    assert cohort.cluster_public_rows([absent_metadata])[0][0]["instance_ids"] == []
    bad_public = deepcopy(base)
    bad_public["context"] = None
    with pytest.raises(ValueError, match="public_row_incomplete"):
        cohort.cluster_public_rows([bad_public])
    with pytest.raises(ValueError, match="duplicate_family"):
        cohort.assign_roles([{"family_id": "x"}, {"family_id": "x"}], {}, "salt")
    with pytest.raises(ValueError, match="role_split"):
        cohort.predictor_view(base, "x", "evaluation")
    with pytest.raises(ValueError, match="role_invalid"):
        cohort.predictor_view(base, "x", "unknown")
    a = cohort.predictor_view(base, "a", "fit")
    b = cohort.predictor_view(row("train", "other", "answer", "b"), "b", "fit")
    for key, value, message in (
        ("labels_accessible", True, "label"),
        ("answer_sha256", "wrong", "answer_hash"),
    ):
        broken = deepcopy(a)
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            cohort.validate_predictor(broken, "fit")
    with pytest.raises(ValueError, match="duplicate_family"):
        cohort.feature_rows([a, a], "fit")
    assert len(cohort.feature_rows([a, b], "fit")) == 6
