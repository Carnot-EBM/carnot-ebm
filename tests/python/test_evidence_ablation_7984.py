"""REQ-VERIFY-7984: public mapping, matched fitting and original-target inference."""

import copy

import numpy as np
import pytest

from carnot.verify import evidence_ablation_7984 as a
from carnot.verify import evidence_features_7980 as f


def fixture():
    """Disjoint public sources retain qualifiers; synthetic outcomes test wiring."""
    public, data = {}, {}
    for role, count in dict(fit=128, tune=32, policy_design=32, evaluation=64).items():
        public[role], data[role] = [], []
        for i in range(count):
            source = f"Only {role} item {i:03d} has {i % 2} units. No other item does."
            answer = f"Only {role} item {i:03d} has {i % 2} units."
            row = dict(
                family_id=f"{role}-{i}",
                source_bytes=source.encode().hex(),
                answer_bytes=answer.encode().hex(),
            )
            public[role].append(row)
            data[role].append(
                dict(
                    family_id=row["family_id"],
                    source_cluster_id=f.normalized(source.encode()),
                    q=0.5,
                    features=f.extract(row)["values"],
                    y=i % 2,
                    status="completed",
                )
            )
    return public, data


def test_mapping_preserves_bytes_and_excludes_self():
    public, _ = fixture()
    manifest = a.interventions(public)
    assert manifest == a.interventions(copy.deepcopy(public))
    for role, rows in manifest.items():
        assert len(rows) == len(public[role])
        assert len({r["donor_family_id"] for r in rows}) == len(rows)
        for original, r in zip(public[role], rows, strict=True):
            assert r["original"] == original == r["duplicate"]
            assert r["donor"]["answer_bytes"] == original["answer_bytes"]
            assert r["donor_family_id"] != original["family_id"]
            assert r["erased"]["source_bytes"] == ""
            assert r["target_scope"] == "original_source_only"
    singleton = a.interventions({"fit": [public["fit"][0]]})["fit"][0]
    assert singleton["donor"] is None
    assert singleton["donor_exclusion"] == "no_within_bin_derangement"
    with pytest.raises(ValueError, match="public_fields"):
        a.interventions({"fit": [dict(public["fit"][0], y=1)]})
    same = copy.deepcopy(public["fit"][:2])
    same[1]["source_bytes"] = same[0]["source_bytes"]
    assert all(r["donor"] is None for r in a.interventions({"fit": same})["fit"])


def test_feature_masks_and_null_location():
    public, data = fixture()
    views = a.interventions(public)
    r, v = data["fit"][0], views["fit"][0]
    for arm in a.ARMS:
        inputs = a.arm_rows([r], [v], arm)
        if arm == "q_only":
            assert inputs[0]["features"] == [0.0] * 8
        if arm == "source_only":
            assert inputs[0]["q"] == 0.5
        assert inputs[0]["y"] == r["y"]
    assert a.source_features(v["erased"])[-2:] == [1.0, 1.0]
    assert a.arm_rows([r], [dict(v, donor=None)], "q_donor")[0]["features"] is None


def test_matched_fit_and_original_only_reduction():
    public, data = fixture()
    views = a.interventions(public)
    fitted = a.fit(data, views)
    assert fitted["optimizer_work"]["total_steps"] == 2400
    assert all(len(h["parameters"]) == 97 for arm in fitted["arms"].values() for h in arm["heads"])
    policies = a.design(fitted, data["policy_design"], views["policy_design"])
    result = a.evaluate(fitted, policies, data["evaluation"], views["evaluation"])
    assert result["duplicate_parity"]["passed"]
    assert result["evaluation_support"]["passed"]
    assert len(result["paired_comparisons"]) == 4
    assert all(
        r["y"] is None and r["actual_cost"] is None
        for r in result["rows"]
        if r["intervention"] != "original"
    )
    missing = copy.deepcopy(data["evaluation"])
    for r in missing:
        r["q"] = r["y"] = None
    null = a.evaluate(fitted, policies, missing, views["evaluation"])
    assert not null["added_information_score"] and not null["evaluation_support"]["passed"]
    assert null["duplicate_parity"]["passed"]
    assert all(r["gain"] is None for r in null["paired_comparisons"].values())
    with pytest.raises(ValueError, match="support_floor"):
        a.fit(dict(data, fit=data["fit"][:4]), views)
    assert not a.added_information(dict(passed=False), {})
    c = dict(q_only_brier=dict(gain=0.02, interval=[0.01, 0.03], adjusted_p=0.001))
    assert a.added_information(dict(passed=True), c)
    c["q_only_brier"]["gain"] = 0.009
    assert not a.added_information(dict(passed=True), c)


def test_inference_failure_and_policy_missing():
    public, data = fixture()
    views = a.interventions(public)
    fitted = a.fit(data, views)
    r = dict(data["evaluation"][0], features=None)
    assert a.probabilities(fitted, "full", [r]) == [None] * 3
    assert a.design(fitted, [], [])["full"]["cutoff"] == 0
    assert a.action(None, 0.1) == "escalate"
    assert a.action(0.99, 0.1) == "reject"
    assert a.action(0.001, 0.1) == "accept"
    assert a.action(0.5, 0.1) == "escalate"
    assert a.cost("accept", 1) == 5 and a.cost("reject", 0) == 1
    assert a.cost("escalate", 1) == 0.25
    assert a.cost("escalate", None) is None
    assert np.isfinite(a.source_features(views["evaluation"][0]["erased"])).all()


def test_public_feature_cache_preserves_qualifiers():
    """REQ-VERIFY-7984: byte-keyed reuse avoids repeated extraction without rewriting text."""
    from unittest.mock import patch

    row = dict(
        family_id="cache",
        source_bytes=b"Only cache probe 7984 has 2 units.".hex(),
        answer_bytes=b"Only cache probe 7984 has 2 units.".hex(),
    )
    with patch.object(f, "extract", wraps=f.extract) as extract:
        first = a.source_features(row)
        second = a.source_features(dict(row, family_id="duplicate"))
        assert first == second and extract.call_count == 1
        changed = dict(
            row, source_bytes=b"Only cache probe 7984 has 2 units. Unless it rains.".hex()
        )
        a.source_features(changed)
        assert extract.call_count == 2


def test_original_feature_custody_reuse():
    """REQ-VERIFY-7984: original and duplicate inputs reuse qualified original features."""
    from unittest.mock import patch

    public, data = fixture()
    views = a.interventions(public)
    with patch.object(a, "source_features", side_effect=AssertionError("unexpected extraction")):
        assert (
            a.arm_rows(data["fit"][:1], views["fit"], "full")[0]["features"]
            == data["fit"][0]["features"]
        )
        assert (
            a.arm_rows(data["fit"][:1], views["fit"], "full", "duplicate")[0]["features"]
            == data["fit"][0]["features"]
        )


def test_same_training_targets_across_ablation_arms():
    """REQ-VERIFY-7984: donor abstentions exclude the same fit target in every arm."""
    public, data = fixture()
    views = a.interventions(public)
    views["fit"][0]["donor"] = None
    matched = a.matched_data(data, views)
    assert all(len(current["fit"]) == 127 for current in matched.values())
    assert len({tuple(r["family_id"] for r in current["fit"]) for current in matched.values()}) == 1
    assert len({tuple(r["y"] for r in current["fit"]) for current in matched.values()}) == 1
