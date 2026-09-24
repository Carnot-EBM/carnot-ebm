"""Tests for the V665 independent evidence and learning audit.

Spec refs: REQ-REPORT-7624 and SCENARIO-REPORT-7624-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7624_v665_evidence_audit as exp


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_source_inventory_distinguishes_producer_gate_and_absence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7624-CUSTODY keeps each actual custody state literal."""

    _write_json(
        tmp_path / "producer.json",
        {
            "honest_verdict": "complete_null_measured",
            "verdict_class": "null",
            "flagged_adversarial": False,
        },
    )
    _write_json(tmp_path / "gate.json", {"honest_verdict": "blocked_gate_check_failed"})
    producer = exp.classify_source(
        tmp_path, exp.SourceSpec("producer", Path("producer.json"), None)
    )
    gate = exp.classify_source(
        tmp_path, exp.SourceSpec("gate", Path("missing.json"), Path("gate.json"))
    )
    missing = exp.classify_source(tmp_path, exp.SourceSpec("missing", Path("absent.json"), None))
    assert producer["disposition"] == "authenticated_producer"
    assert producer["eligible_for_science"] is True
    assert gate["disposition"] == "conductor_pre_gate"
    assert gate["eligible_for_science"] is False
    assert missing["disposition"] == "missing_producer"
    assert missing["sha256"] is None


def test_static_reducer_rebuilds_probabilities_costs_and_cluster_interval() -> None:
    """SCENARIO-REPORT-7624-STATIC rebuilds static claims from raw operands."""

    fixture = exp.private_fixture()
    result = exp.reduce_static_branch(
        fixture["static_rows"],
        checkpoint_bytes=fixture["static_checkpoint_bytes"],
        checkpoint_sha256=fixture["static_checkpoint_sha256"],
        role_map=fixture["role_map"],
        bootstrap_draws=128,
        bootstrap_seed=7624001,
    )
    assert result["eligible"] is True
    assert result["independent_unit_count"] == 2
    assert result["row_count"] == 6
    assert result["controls_distinct"] is True
    assert result["sample_count_uses_views_or_seeds"] is False
    assert set(result["arm_metrics"]) == {"factual", "erased", "deranged"}
    assert result["contrasts"]["factual_vs_erased_brier"]["raw_denominator"] == 2
    assert all(0.0 < row["probability"] < 1.0 for row in result["rows"])


def test_learning_reducer_replays_only_released_update_labels() -> None:
    """SCENARIO-REPORT-7624-LEARNING enforces causal roles and retention."""

    fixture = exp.private_fixture()
    result = exp.reduce_learning_branch(
        fixture["learning_rows"],
        checkpoint_bytes=fixture["learning_checkpoint_bytes"],
        checkpoint_sha256=fixture["learning_checkpoint_sha256"],
        role_map=fixture["role_map"],
    )
    assert result["eligible"] is True
    assert result["independent_unit_count"] == 2
    assert result["causal_release_order"] is True
    assert result["optimization_roles"] == ["update"]
    assert result["evaluation_or_admission_labels_used"] is False
    assert result["checkpoint_bytes_match"] is True
    assert result["retention_measured"] is True


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda f: f.update(static_checkpoint_bytes="{"), "checkpoint_hash_mismatch"),
        (lambda f: f["static_rows"][0].update(arm="bad"), "static_arm_invalid"),
        (lambda f: f["static_rows"][0].update(label_role="fit"), "static_label_role_invalid"),
        (
            lambda f: f["static_rows"][0].update(label_accessed_during_optimization=True),
            "label_leak",
        ),
        (
            lambda f: f["static_rows"][0].update(checkpoint_sha256="sha256:bad"),
            "row_checkpoint_mismatch",
        ),
        (lambda f: f["static_rows"][0].update(model_identity="wrong"), "model_identity_mismatch"),
        (lambda f: f["static_rows"][0].update(included=False), "excluded_observed_row"),
        (lambda f: f["static_rows"][0].update(features=[1.0]), "static_feature_width_invalid"),
        (lambda f: f["static_rows"][0].update(label=2), "static_outcome_invalid"),
        (lambda f: f["static_rows"].pop(), "static_arm_roster_invalid"),
    ],
)
def test_static_reader_rejects_other_invalid_operands(change: object, expected: str) -> None:
    """SCENARIO-REPORT-7624-STATIC rejects malformed raw static operands."""

    fixture = exp.private_fixture()
    change(fixture)  # type: ignore[operator]
    with pytest.raises(ValueError, match=expected):
        exp.reduce_static_branch(
            fixture["static_rows"],
            checkpoint_bytes=fixture["static_checkpoint_bytes"],
            checkpoint_sha256=fixture["static_checkpoint_sha256"],
            role_map=fixture["role_map"],
            bootstrap_draws=16,
            bootstrap_seed=1,
        )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda f: f["learning_rows"][0].update(group_id="bad"), "corrupt_group_id"),
        (lambda f: f["learning_rows"][0].update(arm="bad"), "learning_arm_invalid"),
        (lambda f: f["learning_rows"].append(deepcopy(f["learning_rows"][0])), "duplicated_group"),
        (
            lambda f: f["learning_rows"][0].update(checkpoint_sha256="bad"),
            "row_checkpoint_mismatch",
        ),
        (lambda f: f["learning_rows"][0].update(evaluator_access=True), "label_leak"),
        (
            lambda f: f["learning_rows"][0].update(optimizer_input_roles=["admission"]),
            "optimizer_role_leak",
        ),
        (lambda f: f["learning_rows"][0].update(release_index=0), "causal_release_order_invalid"),
        (lambda f: f["learning_rows"][0].update(features=[1.0]), "learning_feature_width_invalid"),
        (
            lambda f: f["learning_rows"][0].update(prediction_probability=0.9),
            "learning_probability_mismatch",
        ),
        (
            lambda f: f["learning_rows"][0].update(accepted_update=True),
            "learning_update_role_invalid",
        ),
        (lambda f: f["learning_rows"][1].update(feedback_label=3), "learning_label_invalid"),
        (
            lambda f: f["learning_rows"][0].update(state_after_sha256="bad"),
            "learning_state_mismatch",
        ),
        (
            lambda f: f["learning_rows"][0].update(restart_state_sha256="bad"),
            "restart_checkpoint_mismatch",
        ),
        (
            lambda f: f["learning_rows"][0].update(retention_label_role="update"),
            "retention_role_invalid",
        ),
        (
            lambda f: f["learning_rows"][0].update(retention_used_for_update=True),
            "optimizer_role_leak",
        ),
        (
            lambda f: f["learning_rows"][0].update(retention_loss=float("nan")),
            "retention_loss_invalid",
        ),
        (lambda f: f["learning_rows"].pop(), "learning_arm_roster_invalid"),
    ],
)
def test_learning_reader_rejects_other_invalid_operands(change: object, expected: str) -> None:
    """SCENARIO-REPORT-7624-LEARNING rejects malformed causal operands."""

    fixture = exp.private_fixture()
    change(fixture)  # type: ignore[operator]
    with pytest.raises(ValueError, match=expected):
        exp.reduce_learning_branch(
            fixture["learning_rows"],
            checkpoint_bytes=fixture["learning_checkpoint_bytes"],
            checkpoint_sha256=fixture["learning_checkpoint_sha256"],
            role_map=fixture["role_map"],
        )


@pytest.mark.parametrize("mutation", exp.MUTATIONS)
def test_required_mutations_fail_closed(mutation: str) -> None:
    """SCENARIO-REPORT-7624-MUTATIONS rejects every required corruption."""

    fixture = exp.private_fixture()
    before = exp.canonical_hash(fixture)
    changed_path = exp.mutate_private_fixture(fixture, mutation)
    failures = exp.validate_private_fixture(fixture)
    assert changed_path
    assert exp.canonical_hash(fixture) != before
    assert exp.MUTATION_FAILURES[mutation] in failures


def test_mutation_receipts_bind_changed_private_bytes() -> None:
    """SCENARIO-REPORT-7624-MUTATIONS retains exact mutation outcomes."""

    receipts = exp.run_private_mutations()
    assert {row["mutation"] for row in receipts} == set(exp.MUTATIONS)
    assert all(row["passed"] is True for row in receipts)
    assert all(row["before_sha256"] != row["after_sha256"] for row in receipts)
    assert all(row["corrupted_fixture_published"] is False for row in receipts)


def test_blocked_artifact_keeps_static_and_learning_outcomes_separate() -> None:
    """SCENARIO-REPORT-7624-TERMINAL keeps readiness distinct from benefit."""

    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    assert exp.validate_artifact(artifact, root=exp.REPO_ROOT) == {"valid": True}
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["flagged_adversarial"] is False
    assert artifact["static_audit_eligible_score"] == 0
    assert artifact["learning_audit_eligible_score"] == 0
    assert artifact["audited_evidence_benefit_score"] is None
    assert artifact["audited_learning_benefit_score"] is None
    assert {row["branch"] for row in artifact["branch_dispositions"]} == {
        "static",
        "learning",
    }
    first = artifact["gate_check_summary"]["first_failure"]
    assert set(first) == exp.GATE_OPERAND_FIELDS
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False


def test_artifact_rejects_false_benefit_and_checksum_drift() -> None:
    """SCENARIO-REPORT-7624-TERMINAL prevents readiness from becoming benefit."""

    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    false_benefit = deepcopy(artifact)
    false_benefit["audited_evidence_benefit_score"] = 1
    false_benefit["reproducibility_checksum"] = exp.reproducibility_checksum(false_benefit)
    with pytest.raises(ValueError, match="blocked_benefit_must_be_null"):
        exp.validate_artifact(false_benefit, root=exp.REPO_ROOT)
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "complete_blocked_changed"
    with pytest.raises(ValueError, match="checksum_mismatch"):
        exp.validate_artifact(changed, root=exp.REPO_ROOT)
    custody_drift = deepcopy(artifact)
    receipt = next(
        row for row in custody_drift["source_artifact_hashes"] if row.get("sha256") is not None
    )
    receipt["bytes"] += 1
    custody_drift["reproducibility_checksum"] = exp.reproducibility_checksum(custody_drift)
    with pytest.raises(ValueError, match="source_size_mismatch"):
        exp.validate_artifact(custody_drift, root=exp.REPO_ROOT)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda a: a.update(schema="bad"), "identity_mismatch"),
        (lambda a: a.update(milestone="bad"), "run_identity_mismatch"),
        (lambda a: a.update(honest_verdict="blocked"), "terminal_prefix_missing"),
        (lambda a: a.update(verdict_class="partial"), "blocked_class_required"),
        (lambda a: a.update(flagged_adversarial=True), "terminal_adversarial_outcome_invalid"),
        (lambda a: a.update(MODEL_SPECS=["model"]), "model_specs_not_empty"),
        (lambda a: a.update(model_invoked=True), "current_model_invoked"),
        (lambda a: a["invocation_counts"].update(model_loads=1), "current_model_counts_nonzero"),
        (lambda a: a.update(inference_substrate="live"), "substrate_mismatch"),
        (lambda a: a.update(inference_substrate_class="no_model_load"), "substrate_class_mismatch"),
        (lambda a: a["field_principles"].pop("rows"), "field_principles_incomplete"),
        (lambda a: a.update(static_audit_eligible_score=1), "blocked_static_eligibility_nonzero"),
        (
            lambda a: a.update(learning_audit_eligible_score=1),
            "blocked_learning_eligibility_nonzero",
        ),
        (lambda a: a.update(audited_learning_benefit_score=1), "blocked_benefit_must_be_null"),
        (lambda a: a.update(gate_check_summary={}), "blocked_gate_summary_invalid"),
        (lambda a: a["acceptance_gate_results"].pop(), "acceptance_gate_categories_incomplete"),
        (
            lambda a: a["acceptance_gate_results"][0].update(condition=""),
            "acceptance_gate_explanations_incomplete",
        ),
        (lambda a: a["branch_dispositions"].pop(), "branch_dispositions_incomplete"),
        (
            lambda a: a["branch_dispositions"][0].update(scientific_hypothesis_retired=True),
            "external_block_retired_hypothesis",
        ),
        (lambda a: a["mutation_rows"][0].update(passed=False), "mutation_rows_invalid"),
        (lambda a: a.update(rows=[{"fabricated": True}]), "blocked_rows_must_not_be_fabricated"),
        (lambda a: a["validation_receipts"].pop(), "validation_receipts_failed"),
        (lambda a: a["source_artifact_hashes"].pop(), "producer_custody_incomplete"),
        (
            lambda a: a.update(
                phase_spans=[
                    {"start_offset_s": 1.0, "end_offset_s": 2.0},
                    {"start_offset_s": 1.5, "end_offset_s": 3.0},
                ]
            ),
            "phase_spans_overlap",
        ),
    ],
)
def test_artifact_contract_rejects_each_invalid_field(change: object, expected: str) -> None:
    """SCENARIO-REPORT-7624-TERMINAL rejects each governed field violation."""

    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    change(artifact)  # type: ignore[operator]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    with pytest.raises(ValueError, match=expected):
        exp.validate_artifact(artifact, root=exp.REPO_ROOT)


def test_checkpoint_and_builder_guards_fail_closed() -> None:
    """SCENARIO-REPORT-7624-STATIC authenticates checkpoint identity and bytes."""

    fixture = exp.private_fixture()
    with pytest.raises(ValueError, match="checkpoint_json_invalid"):
        exp._checkpoint("{", exp._bytes_sha256("{"), "static")
    wrong_branch = exp._json_bytes(
        {"branch": "wrong", "model_identity": exp.HISTORICAL_MODEL_IDENTITY}
    )
    with pytest.raises(ValueError, match="checkpoint_identity_mismatch"):
        exp._checkpoint(wrong_branch, exp._bytes_sha256(wrong_branch), "static")
    wrong_model = exp._json_bytes({"branch": "static", "model_identity": "wrong"})
    with pytest.raises(ValueError, match="model_identity_mismatch"):
        exp._checkpoint(wrong_model, exp._bytes_sha256(wrong_model), "static")
    for checkpoint, expected in (
        (
            {
                "branch": "static",
                "model_identity": exp.HISTORICAL_MODEL_IDENTITY,
                "weights": [],
                "probability_sign": 1,
            },
            "checkpoint_weights_invalid",
        ),
        (
            {
                "branch": "static",
                "model_identity": exp.HISTORICAL_MODEL_IDENTITY,
                "weights": [1.0],
                "probability_sign": -1,
            },
            "checkpoint_probability_sign_invalid",
        ),
    ):
        checkpoint_bytes = exp._json_bytes(checkpoint)
        with pytest.raises(ValueError, match=expected):
            exp.reduce_static_branch(
                fixture["static_rows"],
                checkpoint_bytes=checkpoint_bytes,
                checkpoint_sha256=exp._bytes_sha256(checkpoint_bytes),
                role_map=fixture["role_map"],
                bootstrap_draws=8,
                bootstrap_seed=1,
            )
    invalid_learning = exp._json_bytes(
        {
            "branch": "learning",
            "model_identity": exp.HISTORICAL_MODEL_IDENTITY,
            "initial_weights": [],
            "learning_rate": 0.0,
        }
    )
    with pytest.raises(ValueError, match="learning_checkpoint_invalid"):
        exp.reduce_learning_branch(
            fixture["learning_rows"],
            checkpoint_bytes=invalid_learning,
            checkpoint_sha256=exp._bytes_sha256(invalid_learning),
            role_map=fixture["role_map"],
        )
    with pytest.raises(ValueError, match="unknown_mutation"):
        exp.mutate_private_fixture(fixture, "unknown")
    passed_check = exp.check_row("x", "x", "x", "x", True, True, "eq")
    with pytest.raises(ValueError, match="blocked_artifact_requires_external_failure"):
        exp.build_blocked_artifact(
            exp.REPO_ROOT,
            [passed_check],
            [],
            validation_receipts=exp.provisional_validation_receipts(exp.REPO_ROOT),
            duration_s=0.1,
            phase_spans=[],
            run_date="20260924",
        )


def test_cold_and_independent_replays_use_exact_candidate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7624-TERMINAL reopens exact artifact and source bytes."""

    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    path = tmp_path / "candidate.json"
    _write_json(path, artifact)
    assert exp.cold_replay(path, root=exp.REPO_ROOT) == {"valid": True}
    replay = exp.independent_replay(path, root=exp.REPO_ROOT)
    assert replay["source_count"] == 8
    assert replay["mutation_count"] == 6
    empty = tmp_path / "empty.json"
    _write_json(empty, [])
    with pytest.raises(ValueError, match="artifact_not_object"):
        exp.cold_replay(empty, root=exp.REPO_ROOT)

    drift = deepcopy(artifact)
    source = next(
        row
        for row in drift["source_artifact_hashes"]
        if row["upstream"] == "exp7616-evidence-schema"
    )
    source["disposition"] = "changed"
    drift["reproducibility_checksum"] = exp.reproducibility_checksum(drift)
    _write_json(path, drift)
    with pytest.raises(ValueError, match="independent_source_mismatch"):
        exp.independent_replay(path, root=exp.REPO_ROOT)

    mutation_drift = deepcopy(artifact)
    mutation_drift["mutation_rows"][0]["observed_failures"] = ["changed"]
    mutation_drift["reproducibility_checksum"] = exp.reproducibility_checksum(mutation_drift)
    _write_json(path, mutation_drift)
    with pytest.raises(ValueError, match="independent_mutation_replay_mismatch"):
        exp.independent_replay(path, root=exp.REPO_ROOT)


def test_cli_contract_requires_fixed_date() -> None:
    """SCENARIO-REPORT-7624-TERMINAL fixes the dated execution contract."""

    arguments = exp.parse_args(["--date", "20260924"])
    assert arguments.date == "20260924"
    with pytest.raises(ValueError, match="run_date_must_equal_20260924"):
        exp.parse_args(["--date", "20260925"])
