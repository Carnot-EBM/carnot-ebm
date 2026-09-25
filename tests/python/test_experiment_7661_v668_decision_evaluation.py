"""Frozen evaluation tests for REQ-REPORT-7661."""

import copy
import json

import pytest

from carnot.reporting.experiment_7661_decision_evaluation import build_rows, reduce_rows


def _fixture():
    ids = [f"group-{i}" for i in range(4)]
    features = []
    labels = []
    for i, unit in enumerate(ids):
        labels.append(
            {
                "component_hash": unit,
                "role": "evaluation",
                "learning_partition": "evaluation_only",
                "training_allowed": False,
                "evaluator_only": True,
                "raw_probability": 0.2 if i % 2 == 0 else 0.8,
                "label": i % 2,
            }
        )
        for arm in ("original_source", "evidence_erasure", "within_role_derangement"):
            features.append(
                {
                    "unit_id": unit,
                    "source_group_id": ids[(i + 1) % 4]
                    if arm == "within_role_derangement"
                    else unit,
                    "role": "evaluation",
                    "partition": "evaluation_only",
                    "arm": arm,
                    "denominator": 2,
                    "checked_structural_propositions": 1 if arm == "original_source" else 0,
                    "scoped_contradictions": 0,
                    "ambiguity": 1,
                    "source_sha256": f"sha256:{i}",
                    "censored": True,
                    "excluded": False,
                    "unknown_claims": 1,
                    "whole_answer_certified": False,
                }
            )
    head = {
        "schema": "carnot.exp7660.head.v1",
        "feature_order": [
            "bias",
            "checked_fraction",
            "contradiction_fraction",
            "ambiguity_fraction",
        ],
        "weights": [0.0, 0.0, 0.0, 0.0],
        "clip": 1e-6,
        "normalization": "log1p_counts_and_sentence_fraction",
        "residual_clip": 2.0,
        "state_order": ["correct", "error"],
    }
    bundle = {
        "heads": {
            name: head for name in ("identity", "scalar", "cheap_atom", "atom", "source_erased")
        },
        "selected": "atom",
        "thresholds": [0.1, 0.9],
    }
    return ids, features, labels, bundle


def test_scenario_report_7661_metrics_recompute_paired_rows():
    """SCENARIO-REPORT-7661-METRICS keeps unknowns and paired denominator."""
    ids, features, labels, bundle = _fixture()
    bundle["thresholds"] = [0.3, 0.7]
    rows = build_rows(
        features, labels, bundle, ids, {u: ids[(i + 1) % 4] for i, u in enumerate(ids)}
    )
    assert len(rows) == 24
    assert all(r["censored"] and r["counts"]["independent_group"] == 1 for r in rows)
    result = reduce_rows(rows, seed=7661, draws=10000)
    json.dumps(result)  # REQ-REPORT-7661: the terminal reduction must serialize.
    assert result["sample_size_budget"]["observed"] == 4
    assert result["metrics"]["atom"]["brier"] == pytest.approx(0.04)
    assert result["metrics"]["atom"]["auroc"] == 1.0
    assert result["probability_benefit"] is False


def test_scenario_report_7661_custody_rejects_sidecar_change():
    """SCENARIO-REPORT-7661-CUSTODY rejects a changed label or roster."""
    ids, features, labels, bundle = _fixture()
    derangement = {u: ids[(i + 1) % 4] for i, u in enumerate(ids)}
    changed = copy.deepcopy(labels)
    changed[0]["role"] = "fit"
    with pytest.raises(ValueError, match="label_custody"):
        build_rows(features, changed, bundle, ids, derangement)
    with pytest.raises(ValueError, match="derangement"):
        build_rows(features, labels, bundle, ids, {u: u for u in ids})


def test_scenario_report_7661_terminal_rejects_row_tamper():
    """SCENARIO-REPORT-7661-TERMINAL derives metrics from row operands."""
    ids, features, labels, bundle = _fixture()
    rows = build_rows(
        features, labels, bundle, ids, {u: ids[(i + 1) % 4] for i, u in enumerate(ids)}
    )
    rows[0]["brier"] += 0.1
    with pytest.raises(ValueError, match="row_loss"):
        reduce_rows(rows, seed=7661, draws=10000)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("short_roster", "role_roster"),
        ("wrong_label_id", "role_roster"),
        ("wrong_arm", "feature_arms"),
        ("short_derangement", "derangement"),
        ("bad_thresholds", "thresholds"),
        ("bad_baseline", "baseline_probability"),
        ("bad_feature_role", "feature_custody"),
    ],
)
def test_scenario_report_7661_custody_fails_closed(mutation, error):
    """SCENARIO-REPORT-7661-CUSTODY checks each frozen join operand."""
    ids, features, labels, bundle = _fixture()
    derangement = {u: ids[(i + 1) % 4] for i, u in enumerate(ids)}
    if mutation == "short_roster":
        ids.pop()
    elif mutation == "wrong_label_id":
        labels[0]["component_hash"] = "wrong"
    elif mutation == "wrong_arm":
        features[0]["arm"] = "wrong"
    elif mutation == "short_derangement":
        derangement.pop(ids[0])
    elif mutation == "bad_thresholds":
        bundle["thresholds"] = [0.9, 0.1]
    elif mutation == "bad_baseline":
        labels[0]["raw_probability"] = 2.0
    else:
        features[0]["role"] = "fit"
    with pytest.raises(ValueError, match=error):
        build_rows(features, labels, bundle, ids, derangement)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("empty", "row_roster"),
        ("duplicate", "row_roster"),
        ("paired_label", "paired_label"),
        ("invalid_probability", "probability"),
    ],
)
def test_scenario_report_7661_metrics_rejects_invalid_rows(mutation, error):
    """SCENARIO-REPORT-7661-METRICS does not trust retained row claims."""
    ids, features, labels, bundle = _fixture()
    rows = build_rows(
        features, labels, bundle, ids, {u: ids[(i + 1) % 4] for i, u in enumerate(ids)}
    )
    if mutation == "empty":
        rows = []
    elif mutation == "duplicate":
        rows[1] = rows[0]
    elif mutation == "paired_label":
        rows[1]["label"] = 1 - rows[1]["label"]
    else:
        rows[1]["probability"] = 2.0
    with pytest.raises(ValueError, match=error):
        reduce_rows(rows, draws=10)


def test_scenario_report_7661_metrics_undefined_auc():
    """SCENARIO-REPORT-7661-METRICS uses null AUROC for one-class groups."""
    ids, features, labels, bundle = _fixture()
    for label in labels:
        label["label"] = 0
    rows = build_rows(
        features, labels, bundle, ids, {u: ids[(i + 1) % 4] for i, u in enumerate(ids)}
    )
    assert reduce_rows(rows, draws=10)["metrics"]["atom"]["auroc"] is None
