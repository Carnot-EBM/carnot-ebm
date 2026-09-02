"""Tests for REQ-VERIFY-6869 and SCENARIO-VERIFY-6869-*.

The tests keep generated artifacts in memory or under ``tmp_path``. They do
not write the tracked research record.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6869_calibration_only_paired_semantic_rule as exp


def _control_cell(
    *,
    model: str = "model-a",
    model_family: str = "architecture-a",
    family: str = "family-a",
    group: str = "group-a",
) -> dict:
    return {
        "accepted": True,
        "model_hf_id": model,
        "model_family": model_family,
        "semantic_family": family,
        "semantic_group_identity": group,
        "candidate_ids": ["valid", "invalid"],
        "candidate_token_counts": {"slot_0": 2, "slot_1": 2},
        "candidate_character_counts": {"slot_0": 8, "slot_1": 8},
        "paired_prompt_token_ids": {"base": [1, 2], "label_swap": [1, 2]},
        "presentation_order": {"base": [0, 1], "label_swap": [1, 0]},
        "normalization": "NFC",
        "surface_control": {
            "raw_prompt_template_sha256": "sha256:prompt",
            "candidate_sequence_template_sha256": "sha256:candidate",
        },
        "rejection_reasons": [],
    }


def _score_rows(
    *,
    base: tuple[float, float] = (0.40, 0.10),
    swapped: tuple[float, float] = (0.35, 0.15),
    model: str = "model-a",
    model_family: str = "architecture-a",
    family: str = "family-a",
    group: str = "group-a",
) -> list[dict]:
    rows: list[dict] = []
    for transform, values, order in (
        ("base", base, (0, 1)),
        ("label_swap", swapped, (1, 0)),
    ):
        for slot, candidate in enumerate(("valid", "invalid")):
            row = {
                "model_hf_id": model,
                "model_family": model_family,
                "semantic_family": family,
                "semantic_group_identity": group,
                "candidate_id": candidate,
                "candidate_slot": slot,
                "presentation_position": order.index(slot),
                "nuisance_transform_identity": transform,
                "token_logprobs": [values[slot] - 0.02, values[slot] + 0.02],
                "finite": True,
                "row_hash": "",
            }
            row["row_hash"] = exp.score_row_hash(row)
            rows.append(row)
    return rows


def _summary(
    model: str,
    estimate: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> dict:
    return {
        "model_hf_id": model,
        "estimate": estimate,
        "lower_bound": estimate if lower is None else lower,
        "upper_bound": estimate if upper is None else upper,
    }


def test_req_verify_6869_spec_declares_the_complete_contract() -> None:
    """REQ-VERIFY-6869 exists before the implementation module."""

    text = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("## REQ-VERIFY-6869:", 1)[1]
    for scenario in (
        "HELD-LABEL-ACCESS",
        "SPLIT-COLLISION",
        "SIGN-REVERSAL",
        "NUISANCE-WIN",
        "MISSING-CELL",
        "FAMILY-ONLY-WIN",
        "POOLED-SIMPSON-REVERSAL",
        "BOOTSTRAP-DRIFT",
    ):
        assert f"SCENARIO-VERIFY-6869-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_6869_held_label_access_is_denied_before_loading() -> None:
    """SCENARIO-VERIFY-6869-HELD-LABEL-ACCESS denies held identities."""

    store = exp.CalibrationLabelStore(
        {"calibration": {"valid": True, "invalid": False}},
        {"held"},
        source_path="preregistration.json",
    )
    assert store.load("calibration") == {"valid": True, "invalid": False}
    with pytest.raises(exp.HeldLabelAccessError, match="held label access denied"):
        store.load("held")
    assert store.held_label_access_count == 0
    assert store.held_label_access_attempt_count == 1
    assert store.access_log[-1]["outcome"] == "denied_before_value_load"


def test_scenario_6869_split_collision_blocks_preconditions() -> None:
    """SCENARIO-VERIFY-6869-SPLIT-COLLISION rejects shared groups."""

    assert exp.split_collisions(["cal-a"], ["held-a"]) == []
    assert exp.split_collisions(["same", "cal-a"], ["held-a", "same"]) == ["same"]
    summary = exp.evaluate_preconditions(
        exp6867={
            "calibration_group_manifest": [{"semantic_group_identity": "same"}],
            "sealed_held_group_manifest": [
                {"semantic_group_identity": "same", "label_commitment": "sha256:seal"}
            ],
        },
        exp6868={
            "semantic_contrast_stream_v2_complete_score": 1,
            "calibration_score_manifest": {
                "sha256": "sha256:cal",
                "payload_sha256": "sha256:payload",
            },
            "sealed_held_score_manifest": {
                "sha256": "sha256:held",
                "labels_present": False,
                "effect_reduced": False,
            },
        },
        exp6867_sha256=exp.EXPECTED_EXP6867_SHA256,
        exp6868_sha256=exp.EXPECTED_EXP6868_SHA256,
        calibration_sidecar_sha256="sha256:cal",
        calibration_payload_sha256="sha256:payload",
        held_sidecar_sha256="sha256:held",
    )
    assert summary["passed"] is False
    assert "split_group_disjointness" in summary["failed_checks"]


def test_req_6869_preconditions_check_exact_hashes_and_seals() -> None:
    """REQ-VERIFY-6869 fails closed on source drift and plaintext held labels."""

    exp6867 = {
        "calibration_group_manifest": [
            {
                "semantic_group_identity": "cal",
                "candidate_ids": ["valid", "invalid"],
                "candidate_labels_by_slot": [True, False],
            }
        ],
        "sealed_held_group_manifest": [
            {"semantic_group_identity": "held", "label_commitment": "sha256:seal"}
        ],
    }
    exp6868 = {
        "semantic_contrast_stream_v2_complete_score": 1,
        "calibration_score_manifest": {
            "sha256": "sha256:cal",
            "payload_sha256": "sha256:payload",
        },
        "sealed_held_score_manifest": {
            "sha256": "sha256:held",
            "labels_present": False,
            "effect_reduced": False,
        },
    }
    ready = exp.evaluate_preconditions(
        exp6867=exp6867,
        exp6868=exp6868,
        exp6867_sha256=exp.EXPECTED_EXP6867_SHA256,
        exp6868_sha256=exp.EXPECTED_EXP6868_SHA256,
        calibration_sidecar_sha256="sha256:cal",
        calibration_payload_sha256="sha256:payload",
        held_sidecar_sha256="sha256:held",
    )
    assert ready["passed"] is True

    changed = deepcopy(exp6867)
    changed["sealed_held_group_manifest"][0]["expected_label"] = True
    blocked = exp.evaluate_preconditions(
        exp6867=changed,
        exp6868=exp6868,
        exp6867_sha256="sha256:changed",
        exp6868_sha256=exp.EXPECTED_EXP6868_SHA256,
        calibration_sidecar_sha256="sha256:cal",
        calibration_payload_sha256="sha256:payload",
        held_sidecar_sha256="sha256:held",
    )
    assert blocked["failed_check"] == "exact_exp6867_hash"
    assert "sealed_held_manifest" in blocked["failed_checks"]


def test_req_6869_recomputes_valid_minus_invalid_and_every_nuisance() -> None:
    """REQ-VERIFY-6869 recomputes token means and seven paired controls."""

    rows, missing = exp.reduce_group_rows(
        _score_rows(),
        labels={"valid": True, "invalid": False},
        control_cell=_control_cell(),
    )
    assert missing == []
    assert len(rows) == len(exp.NUISANCE_CONTROLS) == 7
    by_control = {row["nuisance_control"]: row for row in rows}
    assert by_control["identifier"]["semantic_effect"] == pytest.approx(0.25)
    assert by_control["identifier"]["matched_nuisance_effect"] == 0.0
    assert by_control["label_position"]["matched_nuisance_effect"] == pytest.approx(0.05)
    assert by_control["order"]["matched_nuisance_effect"] == pytest.approx(0.0)
    assert by_control["label_position"]["nuisance_difference_in_differences"] == pytest.approx(0.20)
    assert all(row["orientation"] == "valid_minus_invalid" for row in rows)
    assert all(row["normalization"] == "mean_token_log_probability" for row in rows)


def test_scenario_6869_sign_reversal_is_preserved_and_fails() -> None:
    """SCENARIO-VERIFY-6869-SIGN-REVERSAL keeps opposed transform signs."""

    rows, missing = exp.reduce_group_rows(
        _score_rows(base=(0.4, 0.1), swapped=(0.1, 0.2)),
        labels={"valid": True, "invalid": False},
        control_cell=_control_cell(),
    )
    assert missing == []
    assert len(rows) == 7
    assert all(row["sign_reversal"] is True for row in rows)
    assert any(row["nuisance_wins"] is True for row in rows)


def test_scenario_6869_missing_duplicate_and_nonfinite_cells_are_not_imputed() -> None:
    """SCENARIO-VERIFY-6869-MISSING-CELL preserves typed failures."""

    incomplete = _score_rows()[:-1]
    rows, missing = exp.reduce_group_rows(
        incomplete,
        labels={"valid": True, "invalid": False},
        control_cell=_control_cell(),
    )
    assert rows == []
    assert missing[0]["reason"] == "required_score_cell_missing"
    assert missing[0]["imputed"] is False

    duplicate = _score_rows()
    duplicate.append(deepcopy(duplicate[0]))
    assert (
        exp.reduce_group_rows(
            duplicate,
            labels={"valid": True, "invalid": False},
            control_cell=_control_cell(),
        )[1][0]["reason"]
        == "duplicate_score_cell"
    )

    nonfinite = _score_rows()
    nonfinite[0]["token_logprobs"] = [float("nan")]
    nonfinite[0]["row_hash"] = exp.score_row_hash(nonfinite[0])
    assert (
        exp.reduce_group_rows(
            nonfinite,
            labels={"valid": True, "invalid": False},
            control_cell=_control_cell(),
        )[1][0]["reason"]
        == "nonfinite_token_logprob"
    )

    bad_hash = _score_rows()
    bad_hash[0]["row_hash"] = "sha256:wrong"
    assert (
        exp.reduce_group_rows(
            bad_hash,
            labels={"valid": True, "invalid": False},
            control_cell=_control_cell(),
        )[1][0]["reason"]
        == "score_row_hash_invalid"
    )

    no_tokens = _score_rows()
    no_tokens[0]["token_logprobs"] = []
    no_tokens[0]["row_hash"] = exp.score_row_hash(no_tokens[0])
    assert (
        exp.reduce_group_rows(
            no_tokens,
            labels={"valid": True, "invalid": False},
            control_cell=_control_cell(),
        )[1][0]["reason"]
        == "token_logprobs_missing"
    )


def test_scenario_6869_all_structural_nuisance_receipts_fail_closed() -> None:
    """SCENARIO-VERIFY-6869-MISSING-CELL checks every matched control."""

    cell = _control_cell()
    cell.update(
        {
            "accepted": False,
            "rejection_reasons": ["producer rejection"],
            "candidate_ids": ["same", "same"],
            "candidate_token_counts": {"slot_0": 1, "slot_1": 2},
            "candidate_character_counts": {"slot_0": 1, "slot_1": 2},
            "paired_prompt_token_ids": {"base": [1], "label_swap": [2]},
            "presentation_order": {"base": [1, 0], "label_swap": [0, 1]},
            "normalization": "NFD",
            "surface_control": {},
        }
    )
    rows, missing = exp.reduce_group_rows(
        _score_rows(),
        labels={"valid": True, "invalid": False},
        control_cell=cell,
    )
    assert rows == []
    assert set(missing[0]["detail"]) == {
        "accepted_cell",
        "character_length",
        "identifier",
        "label_position",
        "normalization",
        "order",
        "surface_form",
        "token_count",
    }
    assert (
        exp.reduce_group_rows(
            _score_rows(),
            labels={"valid": True},
            control_cell=_control_cell(),
        )[1][0]["reason"]
        == "calibration_label_pair_invalid"
    )


def test_scenario_6869_family_only_win_and_simpson_reversal_fail() -> None:
    """SCENARIO-VERIFY-6869-FAMILY-ONLY-WIN keeps each family visible."""

    family_rows = [
        {"model_hf_id": "a", "semantic_family": "f1", "estimate": 2.0},
        {"model_hf_id": "a", "semantic_family": "f2", "estimate": -0.1},
        {"model_hf_id": "b", "semantic_family": "f1", "estimate": 2.0},
        {"model_hf_id": "b", "semantic_family": "f2", "estimate": 2.0},
    ]
    replication = exp.family_replication_result(
        family_rows,
        required_models=("a", "b"),
        required_families=("f1", "f2"),
    )
    assert replication["passed"] is False
    assert replication["failed_cells"] == [{"model_hf_id": "a", "semantic_family": "f2"}]

    decision = exp.rule_readiness(
        per_model_effects=[_summary("a", 1.5, lower=1.0), _summary("b", 1.0, lower=0.5)],
        per_family_effects=family_rows,
        nuisance_effects=[
            {"model_hf_id": "a", "absolute_upper_bound": 0.1},
            {"model_hf_id": "b", "absolute_upper_bound": 0.1},
        ],
        missing_cell_rows=[],
        sign_reversal_count=0,
        preconditions_passed=True,
        dry_run_without_held_access=True,
        required_models=("a", "b"),
        required_families=("f1", "f2"),
    )
    assert decision["semantic_contrast_rule_ready_score"] == 0
    assert decision["pooled_or_model_win_cannot_override_family_failure"] is True


def test_scenario_6869_nuisance_win_fails_readiness() -> None:
    """SCENARIO-VERIFY-6869-NUISANCE-WIN rejects equal nuisance effects."""

    decision = exp.rule_readiness(
        per_model_effects=[_summary("a", 0.2, lower=0.1)],
        per_family_effects=[{"model_hf_id": "a", "semantic_family": "f1", "estimate": 0.2}],
        nuisance_effects=[{"model_hf_id": "a", "absolute_upper_bound": 0.2}],
        missing_cell_rows=[],
        sign_reversal_count=0,
        preconditions_passed=True,
        dry_run_without_held_access=True,
        required_models=("a",),
        required_families=("f1",),
    )
    assert decision["semantic_contrast_rule_ready_score"] == 0
    assert "semantic_effect_exceeds_nuisance" in decision["failed_checks"]


def test_scenario_6869_bootstrap_is_order_stable_and_contract_drift_disqualifies() -> None:
    """SCENARIO-VERIFY-6869-BOOTSTRAP-DRIFT fixes every interval setting."""

    values = [0.1, 0.3, 0.2, 0.5, 0.4]
    first = exp.bca_mean_interval(values, random_seed=6867, resamples=400)
    second = exp.bca_mean_interval(reversed(values), random_seed=6867, resamples=400)
    assert first == second
    assert first["method"] == "BCa_cluster_bootstrap"
    assert first["lower_bound"] <= first["estimate"] <= first["upper_bound"]

    manifest = deepcopy(exp.FROZEN_BOOTSTRAP_CONTRACT)
    assert exp.bootstrap_contract_errors(manifest) == []
    manifest["random_seed"] = 1
    assert exp.bootstrap_contract_errors(manifest) == ["bootstrap.random_seed"]

    assert exp.bca_mean_interval([0.5], resamples=10)["lower_bound"] == 0.5
    assert exp._percentile([0.5], 0.25) == 0.5
    lower, upper = exp._bca_bounds(0.2, [0.1, 0.2, 0.3], [0.2], confidence_level=0.95)
    assert lower <= upper
    with pytest.raises(ValueError, match="bootstrap_values_empty"):
        exp.bca_mean_interval([], resamples=10)
    with pytest.raises(ValueError, match="aggregate_rows_empty"):
        exp._equal_weight_estimate([], "semantic_effect")
    with pytest.raises(ValueError, match="bootstrap_rows_empty"):
        exp.stratified_bca_interval([], value_field="semantic_effect", resamples=10)


def test_req_6869_blocked_artifact_has_complete_schema_and_failed_values() -> None:
    """REQ-VERIFY-6869 keeps blocked evidence machine-readable."""

    summary = exp.gate_summary([exp.gate_check("exact_exp6868_hash", "expected", "changed")])
    artifact = exp.build_blocked_artifact(
        run_date="20260902",
        duration_s=0.1,
        preconditions=summary,
        access_log=[],
    )
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "exact_exp6868_hash"
    assert artifact["gate_check_summary"]["expected"] == "expected"
    assert artifact["gate_check_summary"]["observed"] == "changed"
    assert exp.validate_artifact(artifact) == []

    mutations = {
        "inference_substrate": "wrong",
        "held_label_access_count": 1,
        "verifier_is_oracle": True,
        "verdict_class": "unknown",
        "honest_verdict": "blocked_without_terminal_prefix",
        "field_principles": {},
        "reproducibility_checksum": "sha256:wrong",
        "gate_check_summary": {},
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert field in exp.validate_artifact(changed) or (
            field == "gate_check_summary"
            and "blocked_gate_check_summary" in exp.validate_artifact(changed)
        )
    missing_field = deepcopy(artifact)
    del missing_field["bootstrap_rows"]
    assert "bootstrap_rows" in exp.validate_artifact(missing_field)
    invalid_row = deepcopy(artifact)
    invalid_row["rows"] = [{"row_hash": "bad"}]
    assert "rows" in exp.validate_artifact(invalid_row)


def test_req_6869_end_to_end_reduces_only_real_calibration_rows() -> None:
    """REQ-VERIFY-6869 runs the complete calibration-only CPU path."""

    artifact = exp.build_artifact(root=exp.REPO_ROOT, run_date="20260902", resamples=600)
    assert artifact["held_label_access_count"] == 0
    assert artifact["rows"]
    assert len(artifact["rows"]) == 112 * 7
    assert artifact["per_model_calibration_effects"]
    assert artifact["per_family_calibration_effects"]
    assert artifact["pooled_calibration_effect"]
    assert artifact["frozen_held_reducer_hash"].startswith("sha256:")
    assert artifact["semantic_contrast_rule_ready_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["held_acceptance_contract"]["bootstrap_interval"]["resamples"] == 10000
    assert all(
        row.get("access_scope") != "held_json_fields" for row in artifact["calibration_access_log"]
    )
    assert exp.validate_artifact(artifact) == []


def test_req_6869_atomic_writer_uses_only_requested_path(tmp_path: Path) -> None:
    """REQ-VERIFY-6869 writes tests only below their temporary directory."""

    target = tmp_path / "result.json"
    exp.write_json_atomic(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}

    missing, raw, error = exp._read_json_object(tmp_path / "absent.json")
    assert missing == {} and raw == b"" and error and error.startswith("FileNotFoundError:")
    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    assert exp._read_json_object(array_path)[2] == "ValueError:top-level JSON object required"

    blocked = exp.build_artifact(root=tmp_path, run_date="20260902", resamples=10)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "required_files_readable"
    assert exp.validate_artifact(blocked) == []
