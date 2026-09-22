"""Tests for REQ-VERIFY-7504 and SCENARIO-VERIFY-7504-*.

Tiny fixtures exercise arithmetic and access boundaries before the read-only
production capture pass. No test loads a model or trains an optimizer.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7504_v657_evidence_interface as exp


ROOT = Path(__file__).resolve().parents[2]


def _fixture() -> tuple[list[exp.JsonDict], list[exp.JsonDict], list[exp.JsonDict]]:
    """Build one exact source, response, window, and two-order call panel."""

    source = "Source says the drug is not approved."
    response = "The drug is not approved. However, trials continue."
    predictor = {
        "group_id": "g-1",
        "role": "training",
        "source_text": source,
        "response_text": response,
        "source_hash": "sha256:normalized-source",
        "response_hash": exp.sha256_text(response),
        "corpus": "fixture",
    }
    cut = len(b"The drug is not approved.")
    windows = [
        {
            "group_id": "g-1",
            "window_index": 0,
            "byte_start": 0,
            "byte_end": cut,
            "window_sha256": exp.sha256_bytes(response.encode()[:cut]),
        },
        {
            "group_id": "g-1",
            "window_index": 1,
            "byte_start": cut,
            "byte_end": len(response.encode()),
            "window_sha256": exp.sha256_bytes(response.encode()[cut:]),
        },
    ]
    calls: list[exp.JsonDict] = []
    cells = [
        ("whole_response", None, (0.2, 0.4)),
        ("focused_window", 0, (0.1, 0.3)),
        ("focused_window", 1, (0.7, 0.9)),
    ]
    for arm, index, probabilities in cells:
        for order, probability in zip(exp.OPTION_ORDERS, probabilities, strict=True):
            calls.append(
                {
                    "call_id": f"{arm}-{index}-{order[0]}",
                    "group_id": "g-1",
                    "source_group_id": "g-1",
                    "role": "training",
                    "arm": arm,
                    "window_index": index,
                    "option_order": list(order),
                    "label_to_option_id": {" A": order[0], " B": order[1]},
                    "probabilities_by_option_id": {
                        "supported": 1.0 - probability,
                        "contains_unsupported": probability,
                    },
                    "source_sha256": exp.sha256_text(source),
                    "response_sha256": exp.sha256_text(response),
                    "window_sha256": None if index is None else windows[index]["window_sha256"],
                    "byte_start": None if index is None else windows[index]["byte_start"],
                    "byte_end": None if index is None else windows[index]["byte_end"],
                    "eligible": True,
                    "disposition": "complete",
                    "gold_label": None,
                }
            )
    return calls, [predictor], windows


def test_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7504 fixes every boundary before code exists."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7504" in text
    for suffix in ("IDENTITY", "FEATURES", "ACCESS", "EVALUATE", "REAL-ROWS", "READY", "E2E"):
        assert f"SCENARIO-VERIFY-7504-{suffix}" in text


def test_two_orders_produce_frozen_analytic_features() -> None:
    """SCENARIO-VERIFY-7504-FEATURES checks all ten equations and controls."""

    calls, predictors, windows = _fixture()
    row = exp.build_feature_rows(calls, predictors, windows)[0]
    assert len(row["features"]) == len(exp.FEATURE_NAMES) == 10
    assert row["raw_whole_expectation"] == pytest.approx(0.3)
    assert row["raw_max_window_probability"] == pytest.approx(0.8)
    assert row["features"][1:4] == pytest.approx([0.2, 0.5, 0.8])
    assert row["features"][4] == pytest.approx(0.3)
    assert row["features"][5:7] == pytest.approx([0.2, 0.2])
    assert row["window_count"] == 2
    assert row["source_hash"] == "sha256:normalized-source"
    assert row["proposed_features_not_findings"] is True


def test_option_order_reversal_is_semantic_not_positional() -> None:
    """SCENARIO-VERIFY-7504-IDENTITY maps both display orders by option ID."""

    calls, predictors, windows = _fixture()
    expected = exp.build_feature_rows(calls, predictors, windows)
    reversed_rows = list(reversed(calls))
    assert exp.build_feature_rows(reversed_rows, predictors, windows) == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda calls, _predictors, _windows: calls.pop(), "missing_option_order"),
        (lambda calls, _predictors, _windows: calls.append(deepcopy(calls[0])), "duplicate_call"),
        (
            lambda calls, _predictors, _windows: calls[0].__setitem__(
                "source_sha256", "sha256:wrong"
            ),
            "source_hash_mismatch",
        ),
        (
            lambda calls, predictors, _windows: predictors[0].__setitem__("role", "test"),
            "role_mismatch",
        ),
    ],
)
def test_join_mutations_fail_closed(mutation: object, message: str) -> None:
    """SCENARIO-VERIFY-7504-IDENTITY rejects incomplete or leaking joins."""

    calls, predictors, windows = _fixture()
    mutation(calls, predictors, windows)  # type: ignore[operator]
    with pytest.raises(exp.EvidenceInterfaceError, match=message):
        exp.build_feature_rows(calls, predictors, windows)


def _write_mode_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write physically separate features and labels for access tests."""

    features = [
        {
            "group_id": f"g-{role}",
            "role": role,
            "source_hash": f"sha256:{role}",
            "features": [float(index)] * 10,
        }
        for index, role in enumerate(("training", "calibration_tuning", "test", "online"), 1)
    ]
    labels = [
        {"group_id": row["group_id"], "role": row["role"], "label": index % 2}
        for index, row in enumerate(features)
    ]
    feature_path = tmp_path / "features.jsonl"
    evaluator_path = tmp_path / "evaluators.jsonl"
    exp.write_jsonl(feature_path, features)
    exp.write_jsonl(evaluator_path, labels)
    return feature_path, evaluator_path


def test_fit_and_predict_are_invariant_to_held_out_label_mutation(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7504-ACCESS keeps test and online labels inaccessible."""

    features, evaluators = _write_mode_fixture(tmp_path)
    first_fit = exp.read_mode(features, evaluators, mode="fit")
    first_predict = exp.read_mode(features, evaluators, mode="predict")
    labels = exp.load_jsonl(evaluators)
    for row in labels:
        if row["role"] in {"test", "online"}:
            row["label"] = 1 - row["label"]
    exp.write_jsonl(evaluators, labels)
    assert exp.read_mode(features, evaluators, mode="fit") == first_fit
    assert exp.read_mode(features, evaluators, mode="predict") == first_predict
    assert all(row["role"] in {"training", "calibration_tuning"} for row in first_fit["rows"])
    assert all("label" not in row for row in first_predict["rows"])
    assert first_fit["access_receipt"]["held_out_labels_opened"] is False


def test_evaluate_requires_an_existing_exact_prediction_hash(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7504-EVALUATE opens labels only after freeze identity."""

    features, evaluators = _write_mode_fixture(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    exp.write_jsonl(predictions, [{"group_id": "g-test", "probability": 0.5}])
    digest = exp.sha256_file(predictions)
    result = exp.read_mode(
        features,
        evaluators,
        mode="evaluate",
        prediction_path=predictions,
        prediction_sha256=digest,
    )
    assert [row["role"] for row in result["rows"]] == ["test"]
    assert result["rows"][0]["label"] in {0, 1}
    assert result["access_receipt"]["prediction_freeze_sha256"] == digest
    with pytest.raises(exp.EvidenceInterfaceError, match="prediction_hash_mismatch"):
        exp.read_mode(
            features,
            evaluators,
            mode="evaluate",
            prediction_path=predictions,
            prediction_sha256="sha256:wrong",
        )
    with pytest.raises(exp.EvidenceInterfaceError, match="prediction_freeze_required"):
        exp.read_mode(features, evaluators, mode="evaluate")


def test_normalization_is_fit_on_training_only() -> None:
    """REQ-VERIFY-7504 prevents calibration or held-out rows changing scaling."""

    rows = [
        {"group_id": "a", "role": "training", "features": [0.0] * 10},
        {"group_id": "b", "role": "training", "features": [2.0] * 10},
        {"group_id": "c", "role": "calibration_tuning", "features": [100.0] * 10},
    ]
    first = exp.fit_normalization(rows)
    rows[-1]["features"] = [-100.0] * 10
    assert exp.fit_normalization(rows) == first
    assert first["mean"] == pytest.approx([1.0] * 10)
    assert first["population_sd"] == pytest.approx([1.0] * 10)
    with pytest.raises(exp.EvidenceInterfaceError, match="training_rows_required"):
        exp.fit_normalization(rows[-1:])


def test_real_features_are_complete_and_label_blind() -> None:
    """SCENARIO-VERIFY-7504-REAL-ROWS reads captures without evaluator labels."""

    result = exp.build_real_feature_rows(ROOT)
    rows = result["rows"]
    assert len(rows) == 511
    assert result["planned_groups"] == 520
    assert result["excluded_groups"] == 9
    assert result["role_counts"] == {
        "training": 176,
        "calibration_tuning": 60,
        "test": 116,
        "online": 159,
    }
    assert len({row["source_hash"] for row in rows}) == 511
    assert all("label" not in row for row in rows)
    assert result["evaluation_labels_opened"] is False


def test_fixture_artifact_validates_and_reduces_sidecars(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7504-READY keeps structural readiness separate."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []
    assert artifact["evidence_ready_score"] == 1
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["verdict_class"] == "null"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["small_ebm_training"]["performed"] is False
    assert set(artifact["field_principles"]) == set(artifact)
    assert (
        exp.independent_reduce(artifact, root=tmp_path, require_validation=False)[
            "evidence_ready_score"
        ]
        == 1
    )

    changed = deepcopy(artifact)
    changed["role_manifest"]["eligible_role_counts"]["test"] = 117
    assert "independent_reduction_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, require_validation=False
    )


def test_frozen_settings_and_validation_scope() -> None:
    """SCENARIO-VERIFY-7504-E2E fixes gates and only three code paths."""

    assert exp.FROZEN_SETTINGS["online_primary"] == {
        "delay": 8,
        "audit_fraction": 0.25,
        "block_release": 8,
        "seeds": [656201, 656202, 656203, 656204, 656205],
    }
    assert exp.FROZEN_SETTINGS["static_primary"]["controls"] == [
        "identical_feature_logistic",
        "whole_only_gibbs",
    ]
    manifest = exp.VALIDATION_MANIFEST
    assert manifest.test_paths == (exp.TEST_PATH.as_posix(),)
    assert manifest.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert manifest.static_paths == (exp.WRAPPER_PATH.as_posix(),)
