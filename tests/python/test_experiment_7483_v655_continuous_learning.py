"""Tests for REQ-KAN-7483 and SCENARIO-KAN-7483-*.

The fixtures use synthetic labels to test replay mechanics. They do not claim
scientific value. The experiment entrypoint supplies the sealed human labels.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7483_v655_continuous_learning as exp


def _rows(count: int = 24) -> list[exp.JsonDict]:
    families = ("qa", "review", "article")
    rows = []
    for index in range(count):
        label = index % 2
        logit = (-1.0 if label == 0 else 1.0) + 0.1 * ((index % 5) - 2)
        rows.append(
            {
                "group_id": f"group-{index:03d}",
                "source_family": families[index % len(families)],
                "role": "online",
                "label": label,
                "native_log_odds": logit,
                "features": [logit, index / count, (index % 3) / 2.0, (index % 7) / 6.0],
            }
        )
    return rows


# REQ-KAN-7483; SCENARIO-KAN-7483-01 and SCENARIO-KAN-7483-04.
def test_protocol_freezes_selection_and_control_laws() -> None:
    protocol = exp.protocol()
    assert protocol["order_seeds"] == list(exp.ORDER_SEEDS)
    assert protocol["delays"] == [0, 8]
    assert protocol["label_audit_probability"] == 0.5
    assert protocol["selection_roles"] == ["training", "calibration_tuning"]
    assert protocol["heldout_selection_allowed"] is False
    assert protocol["arms"] == list(exp.ARMS)
    assert protocol["bootstrap_draws"] == 10_000
    assert protocol["replicate_unit"] == "mean_within_group_before_inference"
    assert protocol["baseline"] == "temperature_calibrated_native_readout"
    assert exp.MODEL_SPECS == []


# REQ-KAN-7483; SCENARIO-KAN-7483-02.
def test_source_families_orders_and_audit_masks_ignore_labels() -> None:
    assert exp.infer_source_family('{"passages":"p","question":"q"}') == "qa"
    assert exp.infer_source_family('{"address":"a","attributes":{}}') == "review"
    assert exp.infer_source_family("plain article text") == "article"
    assert exp.infer_source_family("[]") == "article"
    rows = _rows()
    changed = deepcopy(rows)
    for row in changed:
        row["label"] = 1 - row["label"]
    orders = []
    for order_seed, audit_seed in zip(exp.ORDER_SEEDS, exp.AUDIT_SEEDS, strict=True):
        left = exp.build_stream_order(rows, order_seed)
        right = exp.build_stream_order(changed, order_seed)
        assert [row["group_id"] for row in left] == [row["group_id"] for row in right]
        assert exp.audit_mask(left, audit_seed) == exp.audit_mask(right, audit_seed)
        assert all(value in {True, False} for value in exp.audit_mask(left, audit_seed).values())
        orders.append(tuple(row["group_id"] for row in left))
    assert len(set(orders)) == len(exp.ORDER_SEEDS)


# REQ-KAN-7483; SCENARIO-KAN-7483-03, -04, -06, and -07.
def test_controlled_replay_preserves_chronology_fairness_retention_and_restart(
    tmp_path: Path,
) -> None:
    training = [{**row, "role": "training"} for row in _rows(32)]
    calibration = [{**row, "role": "calibration_tuning"} for row in _rows(24)]
    online = _rows(24)
    retention = [{**row, "role": "internal_test"} for row in _rows(18)]
    selected = exp.select_settings(training, calibration)
    evidence = exp.run_controlled_replay(
        training,
        calibration,
        online,
        retention,
        selected,
        output_dir=tmp_path,
        order_seeds=exp.ORDER_SEEDS[:2],
        audit_seeds=exp.AUDIT_SEEDS[:2],
        delays=(0, 8),
    )
    events = exp.load_jsonl(Path(evidence["feedback_ledger"]["path"]))
    assert events
    assert all(row["prediction_time"] <= row["feedback_time"] for row in events)
    assert all(row["propensity"] == 0.5 for row in events)
    assert all(row["prediction_event_hash"] for row in events)
    assert all(row["state_hash_before"] and row["state_hash_after"] for row in events)
    by_opportunity: dict[tuple[int, int, str], set[bool]] = {}
    for row in events:
        key = (row["order_seed"], row["delay"], row["group_id"])
        by_opportunity.setdefault(key, set()).add(row["label_revealed"])
    assert all(len(values) == 1 for values in by_opportunity.values())
    assert evidence["chronology_violations"] == 0
    assert evidence["crash_replay"]["passed"] is True
    assert evidence["retention_label_uses_for_updates"] == 0
    assert evidence["service_cost_rows"]
    assert {row["operation"] for row in evidence["service_cost_rows"]} >= {
        "prediction",
        "feedback_processing",
        "update",
        "replay_guard",
        "serialization",
        "fsync",
        "restart",
        "no_op",
    }
    assert all(row["denominator_events"] > 0 for row in evidence["service_cost_rows"])


# REQ-KAN-7483; SCENARIO-KAN-7483-05.
def test_reducer_averages_replicates_before_block_bootstrap() -> None:
    rows = []
    for group in range(12):
        label = group % 2
        for replicate in range(5):
            for arm, probability in (
                ("importance_anchor", 0.8 if label else 0.2),
                ("frozen", 0.6 if label else 0.4),
                ("affine", 0.58 if label else 0.42),
                ("unanchored_residual", 0.55 if label else 0.45),
            ):
                rows.append(
                    {
                        "group_id": f"g{group}",
                        "source_family": f"f{group % 3}",
                        "order_seed": replicate,
                        "delay": 0,
                        "arm": arm,
                        "label": label,
                        "probability": probability,
                    }
                )
    reduced = exp.reduce_online_predictions(rows, draws=200, bootstrap_seed=19)
    assert reduced["delays"]["0"]["independent_group_count"] == 12
    assert reduced["delays"]["0"]["replicates_per_group"] == 5
    comparisons = reduced["delays"]["0"]["comparisons"]
    assert set(comparisons) == {"frozen", "affine", "unanchored_residual"}
    assert all(row["delta"] < 0.0 for row in comparisons.values())
    assert all("holm_upper_delta" in row for row in comparisons.values())


# REQ-KAN-7483; SCENARIO-KAN-7483-08 and SCENARIO-KAN-7483-09.
def test_fixture_artifact_is_reducible_and_mutations_fail_closed(tmp_path: Path) -> None:
    artifact = exp.build_fixture_artifact(tmp_path)
    assert exp.validate_artifact(artifact, verify_sources=False) == []
    assert artifact["online_complete_score"] == 1
    assert artifact["verdict_class"] in {"positive", "null"}
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    assert set(exp.REQUIRED_FIELDS) <= set(artifact)
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["verifier_is_oracle"] is False

    for field, replacement in (
        ("schema", "wrong"),
        ("run_date", "19000101"),
        ("MODEL_SPECS", [{"model_id": "wrong"}]),
        ("model_invoked", True),
        ("invocation_counts", {}),
        ("verifier_is_oracle", True),
        ("online_complete_score", 0),
        ("reproducibility_checksum", "sha256:wrong"),
    ):
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert exp.validate_artifact(changed, verify_sources=False)

    shard_change = deepcopy(artifact)
    ledger_path = Path(shard_change["feedback_ledger"]["path"])
    original_ledger = ledger_path.read_text(encoding="utf-8")
    ledger_path.write_text(original_ledger + json.dumps({"bad": 1}) + "\n")
    assert any(
        error.startswith("shard_hash_invalid:")
        for error in exp.validate_artifact(shard_change, verify_sources=False)
    )
    ledger_path.write_text(original_ledger, encoding="utf-8")

    source_file = tmp_path / "source.json"
    source_file.write_text("{}\n", encoding="utf-8")
    sourced = deepcopy(artifact)
    sourced["source_artifact_hashes"] = {
        "source": {"path": str(source_file), "sha256": exp.sha256_file(source_file)}
    }
    exp._finalize(sourced)
    assert exp.validate_artifact(sourced) == []
    malformed_source = deepcopy(sourced)
    malformed_source["source_artifact_hashes"] = {"source": "bad"}
    malformed_source["reproducibility_checksum"] = exp.reproducibility_checksum(malformed_source)
    assert "source_hash_row_invalid:source" in exp.validate_artifact(malformed_source)
    absent_source = deepcopy(sourced)
    absent_source["source_artifact_hashes"]["source"]["path"] = str(tmp_path / "absent")
    absent_source["reproducibility_checksum"] = exp.reproducibility_checksum(absent_source)
    assert "source_hash_invalid:source" in exp.validate_artifact(absent_source)


# REQ-KAN-7483; SCENARIO-KAN-7483-01.
def test_baseline_fallback_uses_fit_roles_only() -> None:
    training = [{**row, "role": "training"} for row in _rows(32)]
    calibration = [{**row, "role": "calibration_tuning"} for row in _rows(24)]
    state = exp.resolve_baseline_state({}, training, calibration)
    assert state["branch"] == "independent_fit_from_fit_shard"
    assert state["roles_consumed"] == ["training", "calibration_tuning"]
    assert state["temperature"] in exp.TEMPERATURE_GRID
    existing = {
        "calibration_complete_score": 1,
        "flagged_adversarial": False,
        "initial_states_for_exp7483": [
            {"base_arm": "temperature", "base_temperature": 1.5, "seed": 655101}
        ],
    }
    loaded = exp.resolve_baseline_state(existing, training, calibration)
    assert loaded["branch"] == "exp7481_frozen_initial_state"
    assert loaded["temperature"] == 1.5


# REQ-KAN-7483; SCENARIO-KAN-7483-01, -02, -05, and -09.
def test_malformed_inputs_and_terminal_classes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = [{**row, "role": "training"} for row in _rows(12)]
    calibration = [{**row, "role": "calibration_tuning"} for row in _rows(12)]
    with pytest.raises(ValueError, match="baseline_training_role_invalid"):
        exp.resolve_baseline_state({}, calibration, calibration)
    with pytest.raises(ValueError, match="baseline_calibration_role_invalid"):
        exp.resolve_baseline_state({}, training, training)
    bad_feature = deepcopy(training[0])
    bad_feature["features"] = [1.0]
    with pytest.raises(ValueError, match="residual_feature_shape_invalid"):
        exp._feature_array(bad_feature)
    non_object = tmp_path / "non-object.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_row_not_object"):
        exp.load_jsonl(non_object)
    settings = exp.select_settings(training, calibration)
    with pytest.raises(ValueError, match="seed_count_mismatch"):
        exp.run_controlled_replay(
            training,
            calibration,
            _rows(6),
            [{**row, "role": "internal_test"} for row in _rows(6)],
            settings,
            output_dir=tmp_path / "bad-seeds",
            order_seeds=(1,),
            audit_seeds=(),
        )
    with pytest.raises(ValueError, match="bootstrap_input_invalid"):
        exp._hierarchical_bootstrap([], draws=10, seed=1, alpha=0.05)
    inconsistent = [
        {
            "group_id": "g",
            "source_family": "qa",
            "order_seed": 1,
            "delay": 0,
            "arm": "importance_anchor",
            "label": 0,
            "probability": 0.5,
        },
        {
            "group_id": "g",
            "source_family": "qa",
            "order_seed": 2,
            "delay": 0,
            "arm": "importance_anchor",
            "label": 1,
            "probability": 0.5,
        },
    ]
    with pytest.raises(ValueError, match="replicate_identity_disagreement"):
        exp.reduce_online_predictions(inconsistent, draws=10, bootstrap_seed=1)

    delay_row = {
        "independent_group_count": 120,
        "importance_improvement_vs_frozen": 0.02,
        "comparisons": {name: {"holm_upper_delta": -0.01} for name in exp.COMPARATORS},
    }
    retention_row = {"upper_brier_delta": 0.0, "passed": True}
    base_reduction = {
        "shards_valid": True,
        "chronology_violations": 0,
        "fair_feedback_opportunities": True,
        "online_reduction": {"delays": {"0": delay_row, "8": delay_row}},
        "retention_reduction": {
            "delays": {"0": retention_row, "8": retention_row},
            "all_delays_passed": True,
        },
        "online_complete_score": 1,
        "online_benefit_score": 1,
    }
    shell = {
        "service_cost_rows": [
            {"operation": name}
            for name in (
                "prediction",
                "feedback_processing",
                "update",
                "replay_guard",
                "serialization",
                "fsync",
                "restart",
                "no_op",
            )
        ],
        "crash_replay": {"passed": True},
    }
    monkeypatch.setattr(exp, "independent_reduce", lambda _value: base_reduction)
    assert exp._finalize(deepcopy(shell))["verdict_class"] == "positive"
    partial = {**base_reduction, "online_complete_score": 0, "online_benefit_score": 0}
    monkeypatch.setattr(exp, "independent_reduce", lambda _value: partial)
    assert exp._finalize(deepcopy(shell))["verdict_class"] == "partial"
    invalid = {**base_reduction, "shards_valid": False, "online_benefit_score": 0}
    monkeypatch.setattr(exp, "independent_reduce", lambda _value: invalid)
    assert exp._finalize(deepcopy(shell))["verdict_class"] == "disqualified"
