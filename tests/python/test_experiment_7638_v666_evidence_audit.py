"""Tests for the V666 evidence and retained-learning audit.

Spec refs: REQ-REPORT-7638 and SCENARIO-REPORT-7638-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7638_v666_evidence_audit as exp


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _static_result() -> dict[str, object]:
    fixture = exp.private_fixture()
    return exp.reduce_static_evidence(
        fixture["static_rows"],
        checkpoint_bytes=fixture["static_checkpoint_bytes"],
        checkpoint_sha256=fixture["static_checkpoint_sha256"],
        role_map=fixture["role_map"],
        expected_attempted_rows=len(fixture["static_rows"]),
        bootstrap_draws=128,
        bootstrap_seed=7638001,
    )


def _learning_result() -> dict[str, object]:
    fixture = exp.private_fixture()
    return exp.reduce_learning_events(
        fixture["learning_rows"],
        checkpoint_bytes=fixture["learning_checkpoint_bytes"],
        checkpoint_sha256=fixture["learning_checkpoint_sha256"],
        role_map=fixture["role_map"],
        expected_attempted_rows=len(fixture["learning_rows"]),
    )


def test_source_inventory_preserves_producer_pregate_blocked_and_missing(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7638-CUSTODY keeps actual custody states literal."""

    _write_json(
        tmp_path / "producer.json",
        {
            "honest_verdict": "complete_null_measured",
            "verdict_class": "null",
            "flagged_adversarial": False,
        },
    )
    _write_json(
        tmp_path / "blocked.json",
        {
            "honest_verdict": "complete_blocked_external",
            "verdict_class": "blocked",
            "flagged_adversarial": False,
        },
    )
    _write_json(tmp_path / "gate.json", {"honest_verdict": "blocked_gate_check_failed"})
    producer = exp.classify_source(
        tmp_path, exp.SourceSpec("producer", Path("producer.json"), None)
    )
    blocked = exp.classify_source(tmp_path, exp.SourceSpec("blocked", Path("blocked.json"), None))
    gate = exp.classify_source(
        tmp_path, exp.SourceSpec("gate", Path("absent.json"), Path("gate.json"))
    )
    missing = exp.classify_source(tmp_path, exp.SourceSpec("missing", Path("missing.json"), None))
    assert producer["disposition"] == "authenticated_producer"
    assert producer["eligible_for_science"] is True
    assert blocked["disposition"] == "authenticated_blocked_producer"
    assert blocked["eligible_for_science"] is False
    assert gate["disposition"] == "conductor_pre_gate"
    assert gate["sha256"].startswith("sha256:")
    assert missing["disposition"] == "missing_producer"
    assert missing["sha256"] is None


def test_static_reduction_rebuilds_registered_absolute_metrics() -> None:
    """SCENARIO-REPORT-7638-STATIC recomputes every registered static metric."""

    result = _static_result()
    assert result["eligible"] is True
    assert result["independent_unit_count"] == 2
    assert result["attempted_row_count"] == 6
    assert result["sample_count_uses_views_or_seeds"] is False
    assert set(result["arm_metrics"]) == {"factual", "erased", "deranged"}
    for metrics in result["arm_metrics"].values():
        assert metrics["brier_denominator"] == 2
        assert metrics["log_loss_denominator"] == 2
        assert metrics["cost_denominator"] == 2
        assert metrics["false_accept_denominator"] == 2
    assert set(result["control_differences"]) == {"erased", "deranged"}
    assert all(row["raw_denominator"] == 1 for row in result["rows"])
    assert all(row["raw_provenance"] for row in result["rows"])


def test_learning_reduction_reconstructs_events_gradients_and_retention() -> None:
    """SCENARIO-REPORT-7638-LEARNING replays causal event rows."""

    result = _learning_result()
    assert result["eligible"] is True
    assert result["independent_unit_count"] == 2
    assert result["attempted_row_count"] == 4
    assert result["future_labels_used"] is False
    assert result["admission_labels_used_for_gradient"] is False
    assert result["evaluation_labels_used_for_gradient"] is False
    assert result["checkpoint_changes_reconstructed"] is True
    assert result["retention_measured"] is True
    assert result["repeated_orders_multiply_samples"] is False
    guarded = [row for row in result["rows"] if row["arm"] == "guarded"]
    assert all(row["checkpoint_changed"] is True for row in guarded)
    assert all(any(abs(value) > 0.0 for value in row["gradient"]) for row in guarded)


@pytest.mark.parametrize("mutation", exp.MUTATIONS)
def test_registered_negative_mutations_fail_closed(mutation: str) -> None:
    """SCENARIO-REPORT-7638-MUTATIONS rejects every registered corruption."""

    fixture = exp.private_fixture()
    before = exp.canonical_hash(fixture)
    changed_path = exp.mutate_private_fixture(fixture, mutation)
    failures = exp.validate_private_fixture(fixture)
    assert changed_path
    assert exp.canonical_hash(fixture) != before
    assert exp.MUTATION_FAILURES[mutation] in failures


def test_mutation_receipts_bind_changed_bytes_and_fixed_tolerances() -> None:
    """SCENARIO-REPORT-7638-MUTATIONS records exact rejection evidence."""

    receipts = exp.run_private_mutations()
    assert {row["mutation"] for row in receipts} == set(exp.MUTATIONS)
    assert all(row["passed"] is True for row in receipts)
    assert all(row["before_sha256"] != row["after_sha256"] for row in receipts)
    assert all(row["tolerances_weakened"] is False for row in receipts)
    assert all(row["corrupted_fixture_published"] is False for row in receipts)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("shuffled_future_labels", "future_label_shuffle"),
        ("duplicate_groups", "duplicated_group"),
        ("dropped_malformed_rows", "attempted_denominator_mismatch"),
        ("unchanged_evidence_controls", "identical_control"),
        ("swapped_fit_evaluation_roles", "corrupt_group_id"),
        ("checkpoint_reuse", "checkpoint_reuse"),
    ],
)
def test_mutation_names_bind_expected_reader_failures(mutation: str, expected: str) -> None:
    """SCENARIO-REPORT-7638-MUTATIONS keeps each corruption branch-specific."""

    assert exp.MUTATION_FAILURES[mutation] == expected


def test_real_preconditions_account_for_each_v666_producer() -> None:
    """SCENARIO-REPORT-7638-CUSTODY inventories all six planned producers."""

    checks, receipts = exp.collect_preconditions(exp.REPO_ROOT)
    producer_receipts = [
        row
        for row in receipts
        if row.get("upstream") in {spec.upstream for spec in exp.SOURCE_SPECS}
    ]
    required = [row for row in checks if row["check"] == "required_scientific_producer"]
    assert len(producer_receipts) == 6
    assert len(required) == 6
    assert {row["disposition"] for row in producer_receipts} == {
        "conductor_pre_gate",
        "missing_producer",
    }
    assert all(row["passed"] is False for row in required)


def test_blocked_artifact_is_complete_without_fabricated_science() -> None:
    """SCENARIO-REPORT-7638-TERMINAL keeps completion distinct from benefit."""

    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    assert exp.validate_artifact(artifact, root=exp.REPO_ROOT) == {"valid": True}
    assert artifact["honest_verdict"] == "complete_blocked_v666_scientific_producers_unavailable"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["flagged_adversarial"] is False
    assert artifact["rows"] == []
    assert artifact["static_audit_eligible_score"] == 0
    assert artifact["learning_audit_eligible_score"] == 0
    assert artifact["audited_evidence_benefit_score"] is None
    assert artifact["audited_learning_benefit_score"] is None
    assert artifact["MODEL_SPECS"] == []
    assert artifact["planned_MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["gate_check_summary"]["first_failure"].keys() == exp.GATE_OPERAND_FIELDS
    assert {row["category"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "probability_benefit",
        "utility",
        "retention",
        "freshness",
    }


def test_blocked_builder_requires_a_real_external_failure() -> None:
    """SCENARIO-REPORT-7638-CUSTODY forbids a fabricated blocked disposition."""

    passed = exp.check_row("x", "x", "x", "x", True, True, "eq")
    with pytest.raises(ValueError, match="blocked_artifact_requires_external_failure"):
        exp.build_blocked_artifact(
            exp.REPO_ROOT,
            [passed],
            [],
            validation_receipts=exp.provisional_validation_receipts(exp.REPO_ROOT),
            duration_s=0.1,
            phase_spans=[],
            run_date="20260925",
        )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda a: a.update(schema="bad"), "identity_mismatch"),
        (lambda a: a.update(run_date="bad"), "run_identity_mismatch"),
        (lambda a: a.update(honest_verdict="blocked"), "terminal_prefix_missing"),
        (lambda a: a.update(verdict_class="partial"), "blocked_class_required"),
        (lambda a: a.update(flagged_adversarial=True), "terminal_adversarial_outcome_invalid"),
        (lambda a: a.update(MODEL_SPECS=["model"]), "model_specs_not_empty"),
        (lambda a: a.update(model_invoked=True), "current_model_invoked"),
        (lambda a: a["invocation_counts"].update(model_loads=1), "current_model_counts_nonzero"),
        (lambda a: a.update(inference_substrate_class="model_load"), "substrate_class_mismatch"),
        (lambda a: a["field_principles"].pop("rows"), "field_principles_incomplete"),
        (lambda a: a.update(static_audit_eligible_score=1), "blocked_static_eligibility_nonzero"),
        (
            lambda a: a.update(learning_audit_eligible_score=1),
            "blocked_learning_eligibility_nonzero",
        ),
        (lambda a: a.update(audited_evidence_benefit_score=1), "blocked_benefit_must_be_null"),
        (lambda a: a.update(gate_check_summary={}), "blocked_gate_summary_invalid"),
        (lambda a: a["acceptance_gate_results"].pop(), "acceptance_gate_categories_incomplete"),
        (lambda a: a.update(rows=[{"fabricated": True}]), "blocked_rows_must_not_be_fabricated"),
        (lambda a: a["mutation_rows"][0].update(passed=False), "mutation_rows_invalid"),
        (lambda a: a["validation_receipts"].pop(), "validation_receipts_failed"),
        (
            lambda a: a["source_artifact_hashes"].pop(
                next(
                    index
                    for index, row in enumerate(a["source_artifact_hashes"])
                    if row.get("upstream") == "exp7632-fit-evidence"
                )
            ),
            "producer_custody_incomplete",
        ),
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
def test_terminal_contract_rejects_invalid_fields(change: object, expected: str) -> None:
    """SCENARIO-REPORT-7638-TERMINAL rejects every governed field violation."""

    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    change(artifact)  # type: ignore[operator]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    with pytest.raises(ValueError, match=expected):
        exp.validate_artifact(artifact, root=exp.REPO_ROOT)


def test_checksum_and_source_custody_drift_fail_closed() -> None:
    """SCENARIO-REPORT-7638-TERMINAL binds claims to exact source bytes."""

    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "complete_blocked_changed"
    with pytest.raises(ValueError, match="checksum_mismatch"):
        exp.validate_artifact(changed, root=exp.REPO_ROOT)
    custody = deepcopy(artifact)
    receipt = next(row for row in custody["source_artifact_hashes"] if row.get("sha256"))
    receipt["bytes"] += 1
    custody["reproducibility_checksum"] = exp.reproducibility_checksum(custody)
    with pytest.raises(ValueError, match="source_size_mismatch"):
        exp.validate_artifact(custody, root=exp.REPO_ROOT)


def test_cold_and_independent_replay_reopen_exact_bytes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7638-TERMINAL independently reopens source dispositions."""

    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    candidate = tmp_path / "candidate.json"
    _write_json(candidate, artifact)
    assert exp.cold_replay(candidate, root=exp.REPO_ROOT) == {"valid": True}
    replay = exp.independent_replay(candidate, root=exp.REPO_ROOT)
    assert replay == {"valid": True, "source_count": 6, "row_count": 0, "mutation_count": 6}
    drift = deepcopy(artifact)
    source = next(
        row
        for row in drift["source_artifact_hashes"]
        if row.get("upstream") == "exp7632-fit-evidence"
    )
    source["disposition"] = "changed"
    drift["reproducibility_checksum"] = exp.reproducibility_checksum(drift)
    _write_json(candidate, drift)
    with pytest.raises(ValueError, match="independent_source_mismatch"):
        exp.independent_replay(candidate, root=exp.REPO_ROOT)
    _write_json(candidate, [])
    with pytest.raises(ValueError, match="artifact_not_object"):
        exp.cold_replay(candidate, root=exp.REPO_ROOT)


def test_reducer_guards_reject_unregistered_inputs() -> None:
    """SCENARIO-REPORT-7638-STATIC and LEARNING keep fixed denominators."""

    fixture = exp.private_fixture()
    with pytest.raises(ValueError, match="attempted_denominator_mismatch"):
        exp.reduce_static_evidence(
            fixture["static_rows"],
            checkpoint_bytes=fixture["static_checkpoint_bytes"],
            checkpoint_sha256=fixture["static_checkpoint_sha256"],
            role_map=fixture["role_map"],
            expected_attempted_rows=99,
            bootstrap_draws=8,
            bootstrap_seed=1,
        )
    bad_learning = deepcopy(fixture["learning_rows"])
    bad_learning[0]["gradient"] = [99.0, 99.0]
    with pytest.raises(ValueError, match="gradient_mismatch"):
        exp.reduce_learning_events(
            bad_learning,
            checkpoint_bytes=fixture["learning_checkpoint_bytes"],
            checkpoint_sha256=fixture["learning_checkpoint_sha256"],
            role_map=fixture["role_map"],
            expected_attempted_rows=len(bad_learning),
        )
    with pytest.raises(ValueError, match="unknown_mutation"):
        exp.mutate_private_fixture(fixture, "unknown")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda rows: rows[0].update(gradient_label_role="admission"), "gradient_role_invalid"),
        (lambda rows: rows[0].update(admission_used_for_gradient=True), "admission_label_leak"),
        (lambda rows: rows[0].update(evaluation_used_for_gradient=True), "evaluation_label_leak"),
        (
            lambda rows: rows[1].update(admission_example_id=rows[0]["admission_example_id"]),
            "admission_reuse",
        ),
        (lambda rows: rows[0].update(checkpoint_before_sha256="bad"), "checkpoint_before_mismatch"),
        (lambda rows: rows[0].update(checkpoint_after_sha256="bad"), "checkpoint_after_mismatch"),
    ],
)
def test_learning_event_custody_guards_fail_closed(change: object, expected: str) -> None:
    """SCENARIO-REPORT-7638-LEARNING rejects invalid event custody operands."""

    fixture = exp.private_fixture()
    rows = deepcopy(fixture["learning_rows"])
    change(rows)  # type: ignore[operator]
    with pytest.raises(ValueError, match=expected):
        exp.reduce_learning_events(
            rows,
            checkpoint_bytes=fixture["learning_checkpoint_bytes"],
            checkpoint_sha256=fixture["learning_checkpoint_sha256"],
            role_map=fixture["role_map"],
            expected_attempted_rows=len(rows),
        )


def test_learning_attempted_denominator_is_fixed() -> None:
    """SCENARIO-REPORT-7638-LEARNING keeps attempted rows in the denominator."""

    fixture = exp.private_fixture()
    with pytest.raises(ValueError, match="attempted_denominator_mismatch"):
        exp.reduce_learning_events(
            fixture["learning_rows"],
            checkpoint_bytes=fixture["learning_checkpoint_bytes"],
            checkpoint_sha256=fixture["learning_checkpoint_sha256"],
            role_map=fixture["role_map"],
            expected_attempted_rows=99,
        )


def test_other_terminal_guards_and_mutation_replay_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7638-TERMINAL covers remaining exact-reader guards."""

    for field, value, expected in (
        ("inference_substrate", "live", "substrate_mismatch"),
        ("audited_learning_benefit_score", 1, "blocked_benefit_must_be_null"),
    ):
        artifact = exp.build_test_artifact(exp.REPO_ROOT)
        artifact[field] = value
        artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(artifact, root=exp.REPO_ROOT)
    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    artifact["acceptance_gate_results"][0]["principle"] = ""
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    with pytest.raises(ValueError, match="acceptance_gate_explanations_incomplete"):
        exp.validate_artifact(artifact, root=exp.REPO_ROOT)
    artifact = exp.build_test_artifact(exp.REPO_ROOT)
    artifact["mutation_rows"][0]["observed_failures"] = ["changed"]
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    candidate = tmp_path / "candidate.json"
    _write_json(candidate, artifact)
    with pytest.raises(ValueError, match="independent_mutation_replay_mismatch"):
        exp.independent_replay(candidate, root=exp.REPO_ROOT)


def test_cli_contract_uses_fixed_date_and_absolute_root() -> None:
    """SCENARIO-REPORT-7638-TERMINAL fixes the dated execution contract."""

    arguments = exp.parse_args(["--date", "20260925"])
    assert arguments.date == "20260925"
    assert arguments.root.is_absolute()
    with pytest.raises(ValueError, match="run_date_must_equal_20260925"):
        exp.parse_args(["--date", "20260924"])
