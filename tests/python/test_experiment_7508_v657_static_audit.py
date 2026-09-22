"""Tests for REQ-REPORT-7508 and SCENARIO-REPORT-7508-*.

The fixtures are private synthetic evidence. They never write a tracked result
or change an upstream V657 artifact.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7508_v657_static_audit as exp


def _decision_cells(probability: float, label: int) -> list[exp.Json]:
    """Build the registered nine cells from the same raw probability."""

    return [
        exp.decision_cell(
            probability,
            label,
            false_accept_cost=false_accept,
            escalation_cost=escalation,
        )
        for false_accept in exp.FALSE_ACCEPT_COSTS
        for escalation in exp.ESCALATION_COSTS
    ]


def _evaluation_rows(groups: int = 100) -> list[exp.Json]:
    """Create complete source rows with five fitted seeds per learned arm."""

    rows: list[exp.Json] = []
    for index in range(groups):
        label = index % 2
        source = f"sha256:source-{index:03d}"
        candidate = 0.9 if label else 0.1
        probabilities = {
            "window_gibbs": candidate,
            "whole_only_gibbs": 0.65 if label else 0.35,
            "identical_ten_feature_logistic": 0.6 if label else 0.4,
        }
        for arm, base in probabilities.items():
            for seed_index, seed in enumerate(exp.FIT_SEEDS):
                probability = base + (seed_index - 2) * 0.001
                losses = exp.metric_losses(probability, label)
                rows.append(
                    {
                        "group_id": f"group-{index:03d}",
                        "source_hash": source,
                        "role": "test",
                        "arm": arm,
                        "fit_seed": seed,
                        "probability": probability,
                        "label": label,
                        **losses,
                        "decision_costs": _decision_cells(probability, label),
                        "status": "complete",
                        "failed": False,
                        "censored": False,
                    }
                )
        for arm, probability in (
            ("temperature_whole", 0.7 if label else 0.3),
            ("raw_whole_expectation", 0.65 if label else 0.35),
            ("raw_max_window_probability", 0.6 if label else 0.4),
        ):
            rows.append(
                {
                    "group_id": f"group-{index:03d}",
                    "source_hash": source,
                    "role": "test",
                    "arm": arm,
                    "fit_seed": None,
                    "probability": probability,
                    "label": label,
                    **exp.metric_losses(probability, label),
                    "decision_costs": _decision_cells(probability, label),
                    "status": "complete",
                    "failed": False,
                    "censored": False,
                }
            )
    return rows


def test_metric_reduction_clips_only_log_loss() -> None:
    """SCENARIO-REPORT-7508-REDUCTION fixes the audit clipping rule."""

    low = exp.metric_losses(0.0, 1)
    high = exp.metric_losses(1.0, 0)
    assert low["brier"] == high["brier"] == 1.0
    assert low["log_loss"] == pytest.approx(-math.log(1e-6))
    assert high["log_loss"] == pytest.approx(-math.log(1e-6))
    with pytest.raises(ValueError, match="probability_or_label_invalid"):
        exp.metric_losses(float("nan"), 0)


def test_independent_probability_and_nine_cell_reduction() -> None:
    """REQ-REPORT-7508 counts sources and preserves both Holm families."""

    rows = _evaluation_rows()
    policy = exp.project_policy_rows(rows)
    reduced = exp.reduce_static_rows(rows, policy, exp.fixture_settings(draws=64))
    assert reduced["source_support"] == {
        "groups": 100,
        "supported": 50,
        "contains_unsupported": 50,
        "passed": True,
    }
    assert reduced["seed_rows_per_learned_arm"] == 5
    assert set(reduced["probability_contrasts"]["brier"]) == set(exp.PROBABILITY_CONTROLS)
    assert reduced["probability_holm_family_size"] == 2
    assert len(reduced["decision_cells"]) == reduced["decision_holm_family_size"] == 9
    assert reduced["static_probability_value_score"] == 1
    assert reduced["sample_size_budget"]["completed"] == 100


def test_option_order_and_claim_mutations_fail_closed() -> None:
    """SCENARIO-REPORT-7508-MUTATIONS rejects order, checkpoint, and promotion drift."""

    plans = [
        {"request_id": "a", "option_order": ["supported", "contains_unsupported"]},
        {"request_id": "b", "option_order": ["contains_unsupported", "supported"]},
    ]
    observed = [
        {
            **row,
            "label_to_option_id": {" A": row["option_order"][0], " B": row["option_order"][1]},
            "order_remapping": {" A": row["option_order"][0], " B": row["option_order"][1]},
            "disposition": "complete",
        }
        for row in plans
    ]
    assert exp.verify_option_order_rows(plans, observed) == []
    swapped = deepcopy(observed)
    swapped[0]["option_order"] = list(reversed(swapped[0]["option_order"]))
    assert "option_order_mismatch:a" in exp.verify_option_order_rows(plans, swapped)
    contract = exp.fixture_claim_contract()
    assert exp.claim_contract_errors(contract) == []
    wrong = deepcopy(contract)
    wrong["fit_bundle_sha256"] = "sha256:wrong"
    assert "checkpoint_hash_mismatch" in exp.claim_contract_errors(wrong)
    wrong = deepcopy(contract)
    wrong["probability_holm_family_size"] = 3
    assert "probability_holm_family_invalid" in exp.claim_contract_errors(wrong)
    wrong = deepcopy(contract)
    wrong["producer_probability_score"] = 1
    assert "descriptive_result_promoted" in exp.claim_contract_errors(wrong)


def test_row_mutations_reject_label_source_seed_and_escalation() -> None:
    """SCENARIO-REPORT-7508-MUTATIONS rejects four raw-row corruptions."""

    rows = _evaluation_rows()
    policy = exp.project_policy_rows(rows)
    settings = exp.fixture_settings(draws=16)
    flipped = deepcopy(rows)
    flipped[0]["label"] = 1 - flipped[0]["label"]
    with pytest.raises(ValueError, match="metric_mismatch"):
        exp.reduce_static_rows(flipped, policy, settings)
    duplicated = deepcopy(rows)
    duplicated[-1]["source_hash"] = duplicated[0]["source_hash"]
    with pytest.raises(ValueError, match="source_identity_not_bijective"):
        exp.reduce_static_rows(duplicated, policy, settings)
    selected = [
        row
        for row in rows
        if not (
            row["group_id"] == "group-000"
            and row["arm"] == "window_gibbs"
            and row["fit_seed"] == exp.FIT_SEEDS[0]
        )
    ]
    with pytest.raises(ValueError, match="fit_seed_roster_invalid"):
        exp.reduce_static_rows(selected, policy, settings)
    omitted = deepcopy(policy)
    del omitted[0]
    with pytest.raises(ValueError, match="policy_projection_mismatch"):
        exp.reduce_static_rows(rows, omitted, settings)


def test_inventory_distinguishes_missing_invalid_and_complete(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7508-INVENTORY keeps external states distinct."""

    inventory = exp.inventory_upstreams(tmp_path)
    assert [row["state"] for row in inventory] == ["absent", "absent", "absent"]
    terminal = exp.classify_inventory(inventory, reduction_errors=[])
    assert terminal == {
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_missing_v657_static_inputs",
        "static_audit_complete_score": 1,
        "static_claims_qualified_score": 0,
    }
    present = deepcopy(inventory)
    present[0]["state"] = "invalid"
    assert exp.classify_inventory(present, reduction_errors=[])["verdict_class"] == "disqualified"
    complete = deepcopy(inventory)
    for row in complete:
        row["state"] = "valid"
    classified = exp.classify_inventory(complete, reduction_errors=[])
    assert classified["verdict_class"] == "null"
    assert classified["static_claims_qualified_score"] == 1


def test_real_upstreams_authenticate_and_reproduce_null() -> None:
    """REQ-REPORT-7508 authenticates the present V657 evidence without model work."""

    preconditions = exp.collect_preconditions(exp.REPO_ROOT)
    assert preconditions["missing_external"] == []
    assert preconditions["invalid_present"] == []
    assert all(row["passed"] for row in preconditions["rows"])
    loaded = exp.load_static_inputs(exp.REPO_ROOT, verify_option_rows=False)
    reduced = exp.reduce_static_rows(
        loaded["evaluation_rows"],
        loaded["policy_rows"],
        loaded["evaluation"]["evaluator_settings"],
    )
    assert reduced["source_support"]["groups"] == 116
    assert reduced["probability_holm_family_size"] == 2
    assert reduced["decision_holm_family_size"] == 9
    assert reduced["static_probability_value_score"] == 0
    assert reduced["selective_decision_value_score"] == 0
    assert exp.compare_producer_reduction(loaded["evaluation"], reduced) == []


def test_artifact_contract_and_private_mutation_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7508-E2E binds fields without storing corrupt fixtures."""

    artifact = exp.fixture_artifact()
    assert exp.validate_artifact(artifact, verify_sources=False) == []
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["static_audit_complete_score"] == 1
    assert artifact["static_claims_qualified_score"] == 1
    controls = exp.run_private_mutations()
    assert set(controls) == set(exp.MUTATION_NAMES)
    assert all(controls.values())
    with monkeypatch.context() as patcher:
        patcher.setattr(exp, "_validated_groups", lambda *_args: {})
        assert exp.run_private_mutations()["duplicate_source"] is False
    assert "private_mutation_fixtures" not in artifact
    changed = deepcopy(artifact)
    changed["qualified_static_probability_value_score"] = 1
    assert "qualified_probability_exceeds_recomputed" in exp.validate_artifact(
        changed, verify_sources=False
    )
    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    assert "field_principles_incomplete" in exp.validate_artifact(changed, verify_sources=False)
    changed = deepcopy(artifact)
    changed["inference_substrate"] = "wrong"
    assert "current_provenance_invalid" in exp.validate_artifact(changed, verify_sources=False)


def test_parser_and_checksum_are_stable() -> None:
    """REQ-REPORT-7508 freezes the run identity for fresh-process replay."""

    assert exp.parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE
    with pytest.raises(SystemExit):
        exp.parse_args([])
    value = {"schema": "fixture", "reproducibility_checksum": ""}
    value["reproducibility_checksum"] = exp.reproducibility_checksum(value)
    assert value["reproducibility_checksum"] == exp.reproducibility_checksum(value)


def test_defensive_readers_receipts_and_option_contract(tmp_path: Path) -> None:
    """REQ-REPORT-7508 fails closed on malformed objects and receipt drift."""

    object_path = tmp_path / "object.json"
    object_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp.load_json(object_path)
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        exp.load_jsonl(rows_path)
    assert exp._required_receipts_pass(7504, {}) is False
    actual = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATHS[7504])
    missing = deepcopy(actual)
    missing["validation_receipts"] = missing["validation_receipts"][1:]
    assert exp._required_receipts_pass(7504, missing) is False
    failed = deepcopy(actual)
    failed["validation_receipts"][0]["passed"] = False
    assert exp._required_receipts_pass(7504, failed) is False
    malformed_root = tmp_path / "malformed"
    for path in exp.UPSTREAM_PATHS.values():
        target = malformed_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("[]", encoding="utf-8")
    assert all(row["state"] == "invalid" for row in exp.inventory_upstreams(malformed_root))

    plans = [
        {"request_id": "a", "option_order": ["supported", "contains_unsupported"]},
        {"request_id": "b", "option_order": ["contains_unsupported", "supported"]},
    ]
    observed = [
        {
            **row,
            "label_to_option_id": {" A": row["option_order"][0], " B": row["option_order"][1]},
            "order_remapping": {" A": row["option_order"][0], " B": row["option_order"][1]},
            "raw_logits_by_option_id": {"supported": 1.0, "contains_unsupported": 0.0},
            "disposition": "complete",
        }
        for row in plans
    ]
    assert "duplicate_planned_request" in exp.verify_option_order_rows(
        [plans[0], plans[0]], observed
    )
    assert "duplicate_observed_request" in exp.verify_option_order_rows(
        plans, [observed[0], observed[0]]
    )
    assert "option_request_roster_mismatch" in exp.verify_option_order_rows(plans, observed[:1])
    bad = deepcopy(observed)
    bad[0]["label_to_option_id"] = {}
    assert "option_mapping_mismatch:a" in exp.verify_option_order_rows(plans, bad)
    bad = deepcopy(observed)
    bad[0]["raw_logits_by_option_id"] = {"supported": 1.0}
    assert "option_logits_invalid:a" in exp.verify_option_order_rows(plans, bad)
    assert "both_option_orders_not_observed" in exp.verify_option_order_rows(
        plans[:1], observed[:1]
    )
    absent = exp.verify_option_order_rows(plans, [observed[1]])
    assert "option_request_roster_mismatch" in absent


def test_defensive_row_contract_branches() -> None:
    """SCENARIO-REPORT-7508-MUTATIONS names malformed row boundaries."""

    rows = _evaluation_rows()
    policy = exp.project_policy_rows(rows)
    with pytest.raises(ValueError, match="row_identity_invalid"):
        exp._validated_groups([{**rows[0], "group_id": ""}], policy)
    changed = deepcopy(rows)
    changed[-1]["role"] = "training"
    with pytest.raises(ValueError, match="evaluation_status_or_role_invalid"):
        exp._validated_groups(changed, policy)
    changed = deepcopy(rows)
    changed[-1]["failed"] = True
    with pytest.raises(ValueError, match="evaluation_failure_or_censoring_present"):
        exp._validated_groups(changed, policy)
    changed = deepcopy(rows)
    changed[-1]["decision_costs"] = []
    with pytest.raises(ValueError, match="decision_cell_roster_invalid"):
        exp._validated_groups(changed, policy)
    changed = deepcopy(rows)
    changed[-1]["decision_costs"][0]["cost"] = 99
    with pytest.raises(ValueError, match="decision_cell_mismatch"):
        exp._validated_groups(changed, policy)
    with pytest.raises(ValueError, match="evaluation_rows_missing"):
        exp._validated_groups([], [])
    changed = [row for row in rows if row["arm"] != "raw_max_window_probability"]
    with pytest.raises(ValueError, match="arm_roster_invalid"):
        exp._validated_groups(changed, exp.project_policy_rows(changed))
    changed = [deepcopy(rows[0]), deepcopy(rows[-1])]
    changed[1]["source_hash"] = changed[0]["source_hash"]
    with pytest.raises(ValueError, match="source_identity_not_bijective"):
        exp._validated_groups(changed, exp.project_policy_rows(changed))

    changed = deepcopy(rows)
    duplicate_simple = deepcopy(next(row for row in changed if row["arm"] == "temperature_whole"))
    changed.append(duplicate_simple)
    with pytest.raises(ValueError, match="simple_arm_roster_invalid"):
        exp._validated_groups(changed, exp.project_policy_rows(changed))
    changed = deepcopy(rows)
    target = next(
        row
        for row in changed
        if row["group_id"] == "group-000" and row["arm"] == "temperature_whole"
    )
    target["label"] = 1
    target.update(exp.metric_losses(target["probability"], 1))
    target["decision_costs"] = _decision_cells(target["probability"], 1)
    with pytest.raises(ValueError, match="group_label_disagreement"):
        exp._validated_groups(changed, exp.project_policy_rows(changed))
    with pytest.raises(ValueError, match="paired_bootstrap_shape_invalid"):
        exp._paired([1.0], np.zeros((2, 2), dtype=int), 1)
    with pytest.raises(ValueError, match="decision_cell_roster_invalid"):
        exp._cell({"decision_costs": []}, "missing")
    with pytest.raises(ValueError, match="decision_label_invalid"):
        exp.decision_cell(0.5, 2, false_accept_cost=1.0, escalation_cost=0.1)
    with pytest.raises(ValueError, match="decision_probability_invalid"):
        exp.decision_cell(float("nan"), 0, false_accept_cost=1.0, escalation_cost=0.1)


def test_settings_and_producer_mismatch_guards() -> None:
    """REQ-REPORT-7508 refuses threshold and producer-headline drift."""

    rows = _evaluation_rows()
    policy = exp.project_policy_rows(rows)
    changed = exp.fixture_settings(draws=64)
    changed["expected_groups"] = 101
    with pytest.raises(ValueError, match="evaluator_setting_invalid:bootstrap_draws"):
        exp.reduce_static_rows(rows, policy, changed)
    changed = exp.fixture_settings(draws=64)
    changed["holm_alpha"] = 0.1
    with pytest.raises(ValueError, match="evaluator_setting_invalid:holm_alpha"):
        exp.reduce_static_rows(rows, policy, changed)
    changed = exp.fixture_settings(draws=64)
    changed["bootstrap_draws"] = exp.BOOTSTRAP_DRAWS
    changed["expected_groups"] = 99
    with pytest.raises(ValueError, match="expected_group_count_mismatch"):
        exp.reduce_static_rows(rows, policy, changed)

    loaded = exp.load_static_inputs(exp.REPO_ROOT, verify_option_rows=False)
    reduced = exp.reduce_static_rows(
        loaded["evaluation_rows"],
        loaded["policy_rows"],
        loaded["evaluation"]["evaluator_settings"],
    )
    producer = loaded["evaluation"]
    mutations = (
        ("probability_contrasts", {}, "producer_probability_contrasts_mismatch"),
        ("probability_metrics", {}, "producer_probability_metrics_mismatch"),
        (
            "policy_evaluation",
            {**producer["policy_evaluation"], "holm_family_size": 8},
            "producer_decision_holm_family_mismatch",
        ),
        (
            "policy_evaluation",
            {**producer["policy_evaluation"], "cells": []},
            "producer_decision_cells_mismatch",
        ),
        ("static_probability_value_score", 1, "producer_score_mismatch"),
    )
    for field, replacement, expected in mutations:
        changed_producer = deepcopy(producer)
        changed_producer[field] = replacement
        assert any(
            error.startswith(expected)
            for error in exp.compare_producer_reduction(changed_producer, reduced)
        )
    assert exp._close({"a": [1.0]}, {"a": [1.0]}) is True
    assert exp._close({"a": 1}, {"b": 1}) is False
    assert exp._close([1], [1, 2]) is False
    assert exp._close("a", "b") is False


def test_access_checkpoint_and_prediction_guard_details() -> None:
    """SCENARIO-REPORT-7508-INVENTORY reports each access-order defect."""

    loaded = exp.load_static_inputs(exp.REPO_ROOT, verify_option_rows=False)
    base = (
        loaded["evidence"],
        loaded["fit"],
        loaded["evaluation"],
        loaded["access"],
        loaded["checkpoints"],
    )
    mutations = (
        (1, "checkpoint_manifest", "bundle_sha256", "wrong", "checkpoint_bundle_hash_mismatch"),
        (1, "checkpoint_manifest", "policy_sha256", "wrong", "checkpoint_policy_hash_mismatch"),
        (
            1,
            "checkpoint_manifest",
            "transform_sha256",
            "wrong",
            "checkpoint_transform_hash_mismatch",
        ),
        (
            1,
            "checkpoint_manifest",
            "frozen_before_heldout_label_access",
            False,
            "checkpoint_not_frozen_before_labels",
        ),
        (1, "label_access_receipt", "held_out_labels_opened", True, "fit_accessed_heldout_labels"),
        (
            3,
            "evaluator_separation",
            "evaluator_store_parsed_during_feature_build",
            True,
            "future_evaluator_influenced_features",
        ),
        (3, "equal_access", "identical_roles", False, "unequal_arm_access"),
        (3, "exposure_audit", "claim_scope", "confirmatory", "exposure_scope_invalid"),
        (0, "role_manifest", "independent_unit", "window", "test_role_manifest_invalid"),
        (2, "evaluator_settings", "confirmatory_allowed", True, "descriptive_result_promoted"),
    )
    for index, container, field, replacement, expected in mutations:
        values = [deepcopy(item) for item in base]
        values[index][container][field] = replacement
        assert expected in exp._checkpoint_errors(*values)

    evaluation = loaded["evaluation"]
    predictions = loaded["predictions"]
    rows = loaded["evaluation_rows"]
    mutations2 = (
        ("prediction_written_before_label_access", False, "prediction_not_frozen_before_labels"),
        ("held_out_labels_opened", False, "evaluation_label_access_invalid"),
        ("prediction_freeze_sha256", "wrong", "prediction_freeze_hash_mismatch"),
        ("prediction_row_count", 0, "prediction_row_count_mismatch"),
    )
    for field, replacement, expected in mutations2:
        changed = deepcopy(evaluation)
        changed["label_access_receipt"][field] = replacement
        assert expected in exp._prediction_errors(changed, predictions, rows)
    labeled = deepcopy(predictions)
    labeled[0]["label"] = 0
    assert "prediction_contains_label" in exp._prediction_errors(evaluation, labeled, rows)
    changed_rows = deepcopy(rows)
    changed_rows[0]["probability"] += 0.01
    assert "prediction_evaluation_projection_mismatch" in exp._prediction_errors(
        evaluation, predictions, changed_rows
    )


def test_source_reference_and_load_input_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7508 authenticates each referenced file before parsing it."""

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    reference = {
        "source_artifact_hashes": [{"path": "source.json", "sha256": exp.sha256_file(source)}]
    }
    assert exp._reference(reference, Path("source.json"))["sha256"] == exp.sha256_file(source)
    assert exp._authenticate_file(tmp_path, reference, Path("source.json"))["bytes"] == 2
    with pytest.raises(ValueError, match="source_reference_invalid"):
        exp._reference({}, Path("source.json"))
    duplicate = {
        **reference,
        "raw_sidecars": {"same": reference["source_artifact_hashes"][0]},
    }
    with pytest.raises(ValueError, match="source_reference_invalid"):
        exp._reference(duplicate, Path("source.json"))
    source.unlink()
    with pytest.raises(ValueError, match="source_missing"):
        exp._authenticate_file(tmp_path, reference, Path("source.json"))
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        exp._authenticate_file(tmp_path, reference, Path("source.json"))
    with pytest.raises(ValueError, match="upstream_inventory_not_valid"):
        exp.load_static_inputs(tmp_path)

    artifacts = {
        7504: {"source_artifact_hashes": []},
        7505: {"source_artifact_hashes": []},
        7507: {"source_artifact_hashes": []},
    }
    monkeypatch.setattr(
        exp,
        "inventory_upstreams",
        lambda _root: [{"state": "valid"}] * 3,
    )
    monkeypatch.setattr(exp, "_authenticate_file", lambda *_args: {})
    monkeypatch.setattr(
        exp,
        "load_json",
        lambda path: artifacts[7504] if path == tmp_path / exp.UPSTREAM_PATHS[7504] else {},
    )
    monkeypatch.setattr(exp, "load_jsonl", lambda _path: [])
    monkeypatch.setattr(exp, "_checkpoint_errors", lambda *_args: [])
    monkeypatch.setattr(exp, "_prediction_errors", lambda *_args: [])
    with pytest.raises(ValueError, match="option_capture_reference_roster_invalid"):
        exp.load_static_inputs(tmp_path)

    artifacts[7504] = {
        "source_artifact_hashes": [
            {
                "path": f"raw/experiment_7494_v656_window_eval_capture/plan-{index}.json",
                "expected_sha256": "sha256:expected",
                "observed_sha256": "sha256:expected",
            }
            for index in range(4)
        ]
    }
    with pytest.raises(ValueError, match="option_capture_hash_invalid"):
        exp.load_static_inputs(tmp_path)


def test_claim_contract_and_artifact_validation_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7508-E2E rejects score, gate, source, and identity drift."""

    contract = exp.fixture_claim_contract()
    changed_contract = deepcopy(contract)
    changed_contract["decision_holm_family_size"] = 8
    assert "decision_holm_family_invalid" in exp.claim_contract_errors(changed_contract)
    changed_contract = deepcopy(contract)
    changed_contract["producer_decision_score"] = 1
    errors = exp.claim_contract_errors(changed_contract)
    assert "descriptive_result_promoted" in errors
    assert "producer_decision_exceeds_recomputed" in errors

    artifact = exp.fixture_artifact()
    mutations = (
        ("schema", "wrong", "artifact_identity_invalid"),
        ("verdict_class", "wrong", "verdict_class_invalid"),
        ("honest_verdict", "unfinished", "honest_verdict_not_terminal"),
        ("static_audit_complete_score", 2, "score_not_bare_binary"),
        (
            "qualified_selective_decision_value_score",
            1,
            "qualified_decision_exceeds_recomputed",
        ),
        ("acceptance_gate_results", [], "gate_contract_invalid"),
        ("validation_receipts", [], "required_validation_failed"),
    )
    for field, replacement, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert any(
            error.startswith(expected)
            for error in exp.validate_artifact(changed, root=tmp_path, verify_sources=False)
        )
    missing = deepcopy(artifact)
    missing.pop("schema")
    assert exp.validate_artifact(missing, verify_sources=False)[0].startswith(
        "required_fields_missing"
    )
    unqualified = deepcopy(artifact)
    unqualified["static_claims_qualified_score"] = 0
    unqualified["qualified_static_probability_value_score"] = 1
    unqualified["independent_reduction"]["static_probability_value_score"] = 1
    assert "unqualified_claim_has_value" in exp.validate_artifact(unqualified, verify_sources=False)

    source = tmp_path / "bound.txt"
    source.write_text("bound", encoding="utf-8")
    with_source = deepcopy(artifact)
    with_source["source_artifact_hashes"] = [
        exp._source_row(source, tmp_path, evidence_class="fixture")
    ]
    with_source["reproducibility_checksum"] = exp.reproducibility_checksum(with_source)
    source.write_text("changed", encoding="utf-8")
    assert "source_hash_mismatch" in " ".join(exp.validate_artifact(with_source, root=tmp_path))
    source.unlink()
    assert "source_missing" in " ".join(exp.validate_artifact(with_source, root=tmp_path))


def test_positive_builder_empty_reduction_and_real_reducer() -> None:
    """REQ-REPORT-7508 keeps positive classification and empty accounting explicit."""

    fixture = exp.fixture_artifact()
    reduced = deepcopy(fixture["independent_reduction"])
    reduced["static_probability_value_score"] = 1
    reduced["producer_probability_value_score"] = 1
    positive = exp.build_artifact(
        preconditions={
            "rows": [{"check": "fixture", "passed": True}],
            "inventory": [
                {"producer": number, "path": path.as_posix(), "state": "valid"}
                for number, path in exp.UPSTREAM_PATHS.items()
            ],
        },
        reduced=reduced,
        reduction_errors=[],
        source_hashes=[],
        validation_receipts=fixture["validation_receipts"],
        mutation_results={name: True for name in exp.MUTATION_NAMES},
        phase_spans=[],
        started_at_utc="2026-09-22T00:00:00+00:00",
        ended_at_utc="2026-09-22T00:00:01+00:00",
        duration_s=1.0,
    )
    assert positive["verdict_class"] == "positive"
    assert positive["qualified_static_probability_value_score"] == 1
    failed_receipts = deepcopy(fixture["validation_receipts"])
    guard = next(row for row in failed_receipts if row["name"] == "verdict_row_consistency_strict")
    guard.update(
        {
            "passed": False,
            "exit_code": 1,
            "output_tail": "NO_HEADROOM_MAJORITY",
        }
    )
    disqualified = exp.build_artifact(
        preconditions={
            "rows": [{"check": "fixture", "passed": True}],
            "inventory": [
                {"producer": number, "path": path.as_posix(), "state": "valid"}
                for number, path in exp.UPSTREAM_PATHS.items()
            ],
        },
        reduced=reduced,
        reduction_errors=[],
        source_hashes=[],
        validation_receipts=failed_receipts,
        mutation_results={name: True for name in exp.MUTATION_NAMES},
        phase_spans=[],
        started_at_utc="2026-09-22T00:00:00+00:00",
        ended_at_utc="2026-09-22T00:00:01+00:00",
        duration_s=1.0,
    )
    assert disqualified["honest_verdict"] == "complete_disqualified_required_validation"
    assert disqualified["static_claims_qualified_score"] == 0
    assert disqualified["flagged_adversarial"] is True
    assert exp.validate_artifact(disqualified, verify_sources=False) == []
    assert exp._empty_reduction()["rows"] == []
    loaded, actual, errors = exp._reduce_real(exp.REPO_ROOT, verify_option_rows=False)
    assert loaded["option_order_checked"] is False
    assert actual["producer_probability_value_score"] == 0
    assert errors == []
    assert len(exp._provisional_terminal_receipts()) == len(exp.TERMINAL_CHECK_NAMES)
    assert exp.load_static_inputs(exp.REPO_ROOT)["option_order_checked"] is True
