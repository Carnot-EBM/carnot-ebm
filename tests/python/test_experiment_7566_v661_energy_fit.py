"""Tests for REQ-VERIFY-7566 and SCENARIO-VERIFY-7566-*.

The fixtures use circular labels only to test structure. They do not support an
empirical benefit claim.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7566_v661_energy_fit as exp


def _native_group(role: str, index: int, label: int) -> tuple[list[exp.JsonDict], exp.JsonDict]:
    """Build six mapped option rows and one separate evaluator label."""

    component = f"sha256:{role}-component-{index}"
    group = f"sha256:{role}-group-{index}"
    base = 1.6 if label else -1.4
    rows: list[exp.JsonDict] = []
    for order_index, order in enumerate(exp.OPTION_ORDERS):
        order_bias = 0.12 * order_index
        for condition, shift in (("original", 0.0), ("absent", -0.8), ("mismatched", -0.35)):
            z_value = base + shift + order_bias + 0.01 * index
            semantic = {"supported": 0.25, "contains_unsupported": 0.25 + z_value}
            display = {" A": semantic[order[0]], " B": semantic[order[1]]}
            rows.append(
                {
                    "component_hash": component,
                    "group_hash": group,
                    "role": role,
                    "donor_role": role,
                    "condition": condition,
                    "option_order": list(order),
                    "display_logits": display,
                    "full_logits_by_option_id": semantic,
                    "disposition": "complete",
                    "generated_tokens": 0,
                    "readout_kind": "option_logits",
                }
            )
    return rows, {"component_hash": component, "role": role, "label": label}


def _role_rows(role: str, count: int, offset: int = 0) -> list[exp.JsonDict]:
    """Create deterministic joined groups with both binary labels."""

    output = []
    for index in range(count):
        native, label = _native_group(role, index + offset, (index + offset) % 2)
        output.append(exp.build_source_group(native, label))
    return output


def test_source_features_preserve_both_orders_and_contrasts() -> None:
    """SCENARIO-VERIFY-7566-FEATURES keeps all contrasts inside one group."""

    native, label = _native_group("fit", 2, 1)
    row = exp.build_source_group(native, label)
    assert row["role"] == "fit"
    assert row["label"] == 1
    assert set(row["features_by_order"]) == {"supported_first", "unsupported_first"}
    for values in row["features_by_order"].values():
        assert len(values) == 3
        assert values[1] == pytest.approx(0.8)
        assert values[2] == pytest.approx(0.35)
    swapped = deepcopy(native)
    swapped[0]["option_order"] = list(exp.OPTION_ORDERS[1])
    with pytest.raises(ValueError, match="readout_(cells|mapping)_invalid"):
        exp.build_source_group(swapped, label)
    leaked = deepcopy(native)
    leaked[-1]["donor_role"] = "policy"
    with pytest.raises(ValueError, match="donor_role_leakage"):
        exp.build_source_group(leaked, label)


def test_binary_energy_and_local_basis_are_exact() -> None:
    """SCENARIO-VERIFY-7566-ENERGY checks 25 parameters and normalization."""

    logits = np.asarray([-1000.0, -1.0, 0.0, 2.0, 1000.0])
    energies, probabilities = exp.binary_energy(logits)
    assert np.array_equal(energies[:, 0], np.zeros(5))
    assert np.array_equal(energies[:, 1], -logits)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-15)
    features = np.asarray(
        [row["features_by_order"]["supported_first"] for row in _role_rows("fit", 12)]
    )
    knots = exp.fit_local_knots(features)
    design = exp.local_design_matrix(features, knots)
    assert knots.shape == (3, exp.KNOT_VECTOR_SIZE)
    assert design.shape == (12, 24)
    assert design.shape[1] + 1 == 25
    assert np.allclose(design.reshape(12, 3, 8).sum(axis=2), 1.0)
    with pytest.raises(ValueError, match="logits_nonfinite"):
        exp.binary_energy([float("nan")])
    with pytest.raises(ValueError, match="feature_matrix_shape_invalid"):
        exp.fit_local_knots(np.ones((4, 2)))


def test_training_retains_nine_candidates_per_family() -> None:
    """SCENARIO-VERIFY-7566-BUDGET retains every fixed search cell."""

    fit = _role_rows("fit", 20)
    tune = _role_rows("tune", 10, 20)
    bundle = exp.fit_energy_bundle(fit, tune, steps=8)
    assert bundle["roles_consumed"] == ["fit", "tune"]
    assert bundle["heldout_labels_consumed"] is False
    assert bundle["seed"] == exp.FIT_SEED
    assert bundle["learning_rate"] == 0.03
    assert len(bundle["candidate_rows"]) == 9 * len(exp.TRAINABLE_FAMILIES)
    for family in exp.TRAINABLE_FAMILIES:
        candidates = [row for row in bundle["candidate_rows"] if row["family"] == family]
        assert {(row["consistency_lambda"], row["l2"]) for row in candidates} == {
            (weight, l2) for weight in exp.CONSISTENCY_GRID for l2 in exp.L2_GRID
        }
        assert all(row["optimizer_steps"] == 8 for row in candidates)
        assert all(row["loss_trace"] for row in candidates)
        assert bundle["selected_heads"][family]["selection_metric"] == "tune_brier"
    assert bundle["selected_heads"]["source_contrast_energy"]["parameter_count"] == 25
    assert bundle["controls"]["constant_feature"]["status"] == "complete"
    assert bundle["controls"]["shuffled_label"]["status"] in {
        "complete",
        "failed_loss_increase",
    }
    assert bundle["bundle_sha256"] == exp.bundle_hash(bundle)


def test_policy_role_is_descriptive_after_tune_freeze() -> None:
    """SCENARIO-VERIFY-7566-FREEZE forbids policy-role model selection."""

    bundle = exp.fit_energy_bundle(_role_rows("fit", 20), _role_rows("tune", 10, 20), steps=4)
    report = exp.build_policy_report(bundle, _role_rows("policy", 10, 30))
    assert report["selection_performed"] is False
    assert report["role"] == "policy"
    assert report["group_count"] == 10
    assert report["frozen_bundle_sha256"] == bundle["bundle_sha256"]
    assert report["policy_sha256"] == exp.policy_hash(report)
    assert set(report["probability_rows"]) >= {
        "source_contrast_energy",
        "raw_original",
        "temperature_original",
    }
    assert exp.expected_action(0.0) == "accept"
    assert exp.expected_action(1.0) == "reject"
    assert exp.expected_action(0.04) == "escalate"
    assert exp.expected_action(0.8) == "escalate"
    with pytest.raises(ValueError, match="probability_invalid"):
        exp.expected_action(float("nan"))


def test_registered_challenges_fail_or_stay_non_claiming() -> None:
    """SCENARIO-VERIFY-7566-CONTROLS covers all five registered challenges."""

    controls = exp.run_challenge_controls()
    assert controls["passed"] is True
    by_name = {row["challenge"]: row for row in controls["rows"]}
    assert by_name["option_swap"]["observed"] == "rejected"
    assert by_name["donor_role_leakage"]["observed"] == "rejected"
    assert by_name["failed_normalization"]["observed"] == "rejected"
    assert by_name["constant_inputs"]["empirical_claim"] is False
    assert by_name["shuffled_training_labels"]["empirical_claim"] is False


def test_real_preconditions_and_reader_authenticate_all_roles() -> None:
    """REQ-VERIFY-7566 checks Exp7564 and sealed label shards before fitting."""

    preconditions = exp.collect_preconditions(exp.REPO_ROOT)
    assert preconditions["passed"] is True
    assert all(row["passed"] for row in preconditions["rows"])
    roles = exp.load_fit_roles(exp.REPO_ROOT)
    assert {name: len(rows) for name, rows in roles.items()} == {
        "fit": 160,
        "tune": 40,
        "policy": 40,
    }
    assert {row["label"] for rows in roles.values() for row in rows} == {0, 1}
    assert all(len(row["features_by_order"]) == 2 for rows in roles.values() for row in rows)


def test_sidecars_and_fixture_artifact_are_cold_replayable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7566-E2E binds frozen heads, rows, and policy bytes."""

    bundle = exp.fit_energy_bundle(_role_rows("fit", 16), _role_rows("tune", 8, 16), steps=2)
    policy = exp.build_policy_report(bundle, _role_rows("policy", 8, 24))
    sidecars = exp.write_fit_sidecars(tmp_path, bundle, policy)
    assert {"frozen_heads", "training_rows", "policy_report"} <= set(sidecars)
    assert all((tmp_path / row["path"]).is_file() for row in sidecars.values())
    artifact = exp.fixture_artifact()
    assert exp.validate_artifact(artifact, verify_sources=False) == []
    assert exp.independent_reduce(artifact) == {
        "energy_fit_ready_score": 1,
        "baseline_ready_score": 1,
    }
    assert artifact["predictive_benefit_measured"] is False
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert set(exp.REQUIRED_FIELDS) <= set(artifact)
    assert set(artifact) <= set(artifact["field_principles"])
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(candidate, verify_sources=False) == []


def test_blocked_artifact_names_exact_external_absence(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7566-BLOCKED keeps external absence out of partial work."""

    failed = exp.precondition_row(
        "upstream_ready",
        "exp7564-fit-capture",
        "fit_capture_ready_score",
        1,
        None,
        "results/experiment_7564_v661_fit_capture.json",
        False,
    )
    artifact = exp.blocked_artifact(failed, root=tmp_path)
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "upstream": "exp7564-fit-capture",
        "path": "results/experiment_7564_v661_fit_capture.json",
        "field": "fit_capture_ready_score",
        "expected": 1,
        "observed": None,
    }
    assert artifact["energy_fit_ready_score"] == artifact["baseline_ready_score"] == 0
    assert artifact["sample_size_budget"]["unstarted"] == 240


def test_validator_rejects_provenance_and_readiness_mutations() -> None:
    """REQ-VERIFY-7566 keeps no-load, checksum, and independent reduction strict."""

    base = exp.fixture_artifact()
    mutations = (
        ("schema", "wrong", "identity_mismatch"),
        ("MODEL_SPECS", ["wrong"], "model_specs_must_be_empty"),
        ("model_specs", ["wrong"], "model_specs_must_be_empty"),
        ("model_invoked", True, "current_model_calls_nonzero"),
        ("inference_substrate_class", "aggregation", "inference_substrate_class_mismatch"),
        ("predictive_benefit_measured", True, "heldout_benefit_claimed"),
        ("energy_fit_ready_score", 0, "independent_reduction_mismatch"),
    )
    for field, replacement, error in mutations:
        changed = deepcopy(base)
        changed[field] = replacement
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert error in exp.validate_artifact(changed, verify_sources=False)
    changed = deepcopy(base)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, verify_sources=False
    )


def test_malformed_inputs_and_cli_modes_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-7566 rejects invalid roles, labels, matrices, and candidate grids."""

    fit = _role_rows("fit", 8)
    tune = _role_rows("tune", 6, 8)
    bad = deepcopy(fit)
    bad[0]["role"] = "test"
    with pytest.raises(ValueError, match="fit_role_invalid"):
        exp.fit_energy_bundle(bad, tune, steps=2)
    bad = deepcopy(tune)
    bad[0]["label"] = None
    with pytest.raises(ValueError, match="binary_label_invalid"):
        exp.fit_energy_bundle(fit, bad, steps=2)
    with pytest.raises(ValueError, match="optimizer_steps_invalid"):
        exp.fit_energy_bundle(fit, tune, steps=301)
    with pytest.raises(ValueError, match="candidate_grid_incomplete"):
        exp.select_head([], "source_contrast_energy")
    with pytest.raises(ValueError, match="knots_shape_invalid"):
        exp.local_design_matrix(np.ones((2, 3)), np.ones((3, 2)))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert exp.collect_preconditions(empty)["passed"] is False
    parsed = exp.parse_args(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"])
    assert parsed.date == exp.RUN_DATE
    assert parsed.cold_replay == Path("candidate.json")


def test_source_and_numeric_guard_branches_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7566-FEATURES rejects each malformed source shape."""

    native, label = _native_group("fit", 0, 0)
    with pytest.raises(ValueError, match="feature_matrix_nonfinite"):
        exp.fit_local_knots(np.asarray([[0.0, 1.0, np.nan], [1.0, 2.0, 3.0]]))
    knots = exp.fit_local_knots(np.asarray([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError, match="feature_matrix_shape_invalid"):
        exp.local_design_matrix(np.ones((2, 2)), knots)
    with pytest.raises(ValueError, match="spline_input_nonfinite"):
        exp.local_design_matrix(np.asarray([[0.0, 1.0, np.nan]]), knots)
    bad = deepcopy(native[0])
    bad["option_order"] = ["wrong", "order"]
    with pytest.raises(ValueError, match="readout_mapping_invalid"):
        exp._semantic_logit(bad)
    bad = deepcopy(native[0])
    bad["display_logits"] = {" A": 0.0}
    with pytest.raises(ValueError, match="readout_mapping_invalid"):
        exp._semantic_logit(bad)
    bad = deepcopy(native[0])
    bad["display_logits"][" A"] = np.nan
    with pytest.raises(ValueError, match="readout_nonfinite"):
        exp._semantic_logit(bad)
    with pytest.raises(ValueError, match="readout_cells_invalid"):
        exp.build_source_group(native[:-1], label)
    bad_rows = deepcopy(native)
    bad_rows[0]["component_hash"] = "sha256:changed"
    with pytest.raises(ValueError, match="source_group_identity_invalid"):
        exp.build_source_group(bad_rows, label)
    with pytest.raises(ValueError, match="evaluator_identity_mismatch"):
        exp.build_source_group(native, {**label, "component_hash": "sha256:changed"})
    with pytest.raises(ValueError, match="binary_label_invalid"):
        exp.build_source_group(native, {**label, "label": None})
    bad_rows = deepcopy(native)
    bad_rows[0]["generated_tokens"] = 1
    with pytest.raises(ValueError, match="native_readout_incomplete"):
        exp.build_source_group(bad_rows, label)
    bad_rows = deepcopy(native)
    bad_rows[0]["condition"] = "wrong"
    with pytest.raises(ValueError, match="readout_cells_invalid"):
        exp.build_source_group(bad_rows, label)
    bad_rows = deepcopy(native)
    bad_rows[1]["condition"] = bad_rows[0]["condition"]
    with pytest.raises(ValueError, match="readout_cells_invalid"):
        exp.build_source_group(bad_rows, label)
    rows = _role_rows("fit", 4)
    rows[0]["features_by_order"] = {}
    with pytest.raises(ValueError, match="feature_views_invalid"):
        exp._role_matrices(rows, "fit")
    rows = _role_rows("fit", 4)
    rows[0]["features_by_order"]["supported_first"] = [0.0]
    with pytest.raises(ValueError, match="feature_vector_invalid"):
        exp._role_matrices(rows, "fit")
    with pytest.raises(ValueError, match="binary_support_invalid"):
        exp._role_matrices([{**row, "label": 1} for row in _role_rows("fit", 4)], "fit")
    with pytest.raises(ValueError, match="family_invalid"):
        exp._family_design("wrong", np.ones((2, 3)), knots)
    failed_grid = [
        {
            "family": "source_contrast_energy",
            "consistency_lambda": weight,
            "l2": l2,
            "status": "failed_loss_increase",
        }
        for weight in exp.CONSISTENCY_GRID
        for l2 in exp.L2_GRID
    ]
    with pytest.raises(ValueError, match="candidate_grid_no_complete"):
        exp.select_head(failed_grid, "source_contrast_energy")
    document = tmp_path / "not-object.json"
    document.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp._load_object(document)
    jsonl = tmp_path / "not-object.jsonl"
    jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        exp._read_jsonl(jsonl)


def test_policy_sidecar_and_reduction_guards(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7566-FREEZE detects bundle and persisted-byte drift."""

    bundle = exp.fit_energy_bundle(_role_rows("fit", 12), _role_rows("tune", 8, 12), steps=1)
    policy_rows = _role_rows("policy", 8, 20)
    changed = deepcopy(bundle)
    changed["bundle_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="fit_bundle_hash_invalid"):
        exp.build_policy_report(changed, policy_rows)
    policy = exp.build_policy_report(bundle, policy_rows)
    sidecars = exp.write_fit_sidecars(tmp_path, bundle, policy)
    assert exp.reload_fit_sidecars(tmp_path, sidecars)["passed"] is True
    missing = deepcopy(sidecars)
    del missing["policy_report"]
    with pytest.raises(ValueError, match="sidecar_receipt_missing"):
        exp.reload_fit_sidecars(tmp_path, missing)
    changed_receipt = deepcopy(sidecars)
    changed_receipt["policy_report"]["sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="sidecar_reload_hash_mismatch"):
        exp.reload_fit_sidecars(tmp_path, changed_receipt)
    policy_path = tmp_path / sidecars["policy_report"]["path"]
    policy_value = json.loads(policy_path.read_text())
    policy_value["frozen_bundle_sha256"] = "sha256:changed"
    policy_path.write_text(json.dumps(policy_value), encoding="utf-8")
    changed_identity = deepcopy(sidecars)
    changed_identity["policy_report"]["sha256"] = exp.sha256_file(policy_path)
    with pytest.raises(ValueError, match="sidecar_reload_identity_mismatch"):
        exp.reload_fit_sidecars(tmp_path, changed_identity)
    assert exp.independent_reduce({}) == {
        "energy_fit_ready_score": 0,
        "baseline_ready_score": 0,
    }
    assert exp.independent_reduce(
        {
            "frozen_head_manifest": {},
            "policy_calibration_report": {},
            "challenge_controls": {},
            "validation_summary": {},
        }
    ) == {
        "energy_fit_ready_score": 0,
        "baseline_ready_score": 0,
    }
    summary = exp._gate_summary([exp._gate("x", "validity", True, False, "==", False)])
    assert summary["failed_checks"] == ["x"]


def test_manifest_and_role_reader_corruptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7566 rejects missing shards and each invalid join shape."""

    outside = tmp_path / "outside.txt"
    outside.write_text("bound", encoding="utf-8")
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    assert exp.source_hash_row(outside, other_root)["path"] == str(outside.resolve())

    original_loader = exp._load_object
    upstream = original_loader(exp.REPO_ROOT / exp.UPSTREAM_PATH)
    protocol = original_loader(exp.REPO_ROOT / exp.PROTOCOL_PATH)
    missing_protocol = deepcopy(protocol)
    del missing_protocol["sealed_shards"]["policy_labels"]

    def missing_loader(path: Path) -> exp.JsonDict:
        if path.name == exp.PROTOCOL_PATH.name:
            return missing_protocol
        return original_loader(path)

    monkeypatch.setattr(exp, "_load_object", missing_loader)
    preconditions = exp.collect_preconditions(exp.REPO_ROOT)
    assert preconditions["passed"] is False
    assert any(row["field"] == "sealed_shards.policy_labels" for row in preconditions["rows"])

    fake_upstream = {"raw_manifest": {"native_row_shards": [{"path": "capture"}]}}
    fake_protocol = {
        "sealed_shards": {
            "fit_labels": {"path": "fit_labels"},
            "tune_labels": {"path": "tune_labels"},
            "policy_labels": {"path": "policy_labels"},
        }
    }

    def fake_loader(path: Path) -> exp.JsonDict:
        return fake_upstream if path.name == exp.UPSTREAM_PATH.name else fake_protocol

    monkeypatch.setattr(exp, "_load_object", fake_loader)
    duplicate = {"component_hash": "sha256:duplicate", "role": "fit", "label": 0}
    monkeypatch.setattr(
        exp,
        "_read_jsonl",
        lambda path: [] if path.name == "capture" else [deepcopy(duplicate)],
    )
    with pytest.raises(ValueError, match="evaluator_identity_duplicate"):
        exp.load_fit_roles(tmp_path)

    native, _label = _native_group("fit", 0, 0)
    monkeypatch.setattr(exp, "_read_jsonl", lambda path: native if path.name == "capture" else [])
    with pytest.raises(ValueError, match="authorized_label_missing"):
        exp.load_fit_roles(tmp_path)

    other_native, other_label = _native_group("other", 0, 0)
    monkeypatch.setattr(
        exp,
        "_read_jsonl",
        lambda path: (
            other_native
            if path.name == "capture"
            else ([other_label] if path.name == "fit_labels" else [])
        ),
    )
    with pytest.raises(ValueError, match="fit_role_invalid"):
        exp.load_fit_roles(tmp_path)

    monkeypatch.setattr(exp, "_read_jsonl", lambda _path: [])
    with pytest.raises(ValueError, match="role_counts_invalid"):
        exp.load_fit_roles(tmp_path)


def test_validator_covers_all_terminal_failure_classes(tmp_path: Path) -> None:
    """REQ-VERIFY-7566 cold validation names every terminal corruption class."""

    assert exp.validate_artifact(None, verify_sources=False) == ["artifact_not_object"]
    base = exp.fixture_artifact()

    def errors(field: str, value: object) -> list[str]:
        changed = deepcopy(base)
        if value is _DELETE:
            del changed[field]
        else:
            changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return exp.validate_artifact(changed, verify_sources=False)

    assert any(item.startswith("required_fields_missing") for item in errors("milestone", _DELETE))
    assert "identity_mismatch" in errors("milestone", "wrong")
    assert "inference_substrate_mismatch" in errors("inference_substrate", "wrong")
    assert "execution_venue_invalid" in errors("execution_venue", "host_cpu")
    assert "verdict_class_invalid" in errors("verdict_class", "wrong")
    assert "honest_verdict_prefix_invalid" in errors("honest_verdict", "wrong")

    changed = deepcopy(base)
    changed["policy_calibration_report"]["policy_sha256"] = "sha256:changed"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "policy_hash_mismatch" in exp.validate_artifact(changed, verify_sources=False)

    changed = deepcopy(base)
    del changed["field_principles"]["milestone"]
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "field_principles_incomplete" in exp.validate_artifact(changed, verify_sources=False)
    assert "acceptance_gates_invalid" in errors("acceptance_gate_results", [{}])

    source = tmp_path / "source.txt"
    source.write_text("bound", encoding="utf-8")
    changed = deepcopy(base)
    changed["source_artifact_hashes"] = [exp.source_hash_row(source, tmp_path)]
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert exp.validate_artifact(changed, root=tmp_path) == []
    source.write_text("changed", encoding="utf-8")
    assert any(
        item.startswith("source_hash_mismatch")
        for item in exp.validate_artifact(changed, root=tmp_path)
    )


_DELETE = object()
