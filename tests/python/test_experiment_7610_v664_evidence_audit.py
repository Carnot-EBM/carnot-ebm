"""Tests for the V664 independent evidence audit.

Spec refs: REQ-REPORT-7610 and SCENARIO-REPORT-7610-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7610_v664_evidence_audit as exp


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _segments(text: str, prefix: str) -> list[dict[str, object]]:
    encoded = text.encode("utf-8")
    return [
        {
            "sentence_id": f"{prefix}001",
            "byte_start": 0,
            "byte_end": len(encoded),
            "text": text,
            "text_sha256": exp.text_sha256(text),
        }
    ]


def _model_row(component: str) -> dict[str, object]:
    source = "Source café."
    question = "Question?"
    answer = "Answer."
    return {
        "component_hash": component,
        "role": "pilot",
        "complete_source": source,
        "complete_question": question,
        "complete_answer": answer,
        "source_sha256": exp.text_sha256(source),
        "question_sha256": exp.text_sha256(question),
        "answer_sha256": exp.text_sha256(answer),
        "source_sentences": _segments(source, "S"),
        "question_sentences": _segments(question, "S"),
        "answer_sentences": _segments(answer, "R"),
        "evidence_feature_names": list(exp.EVIDENCE_FEATURE_NAMES),
        "labels_accessible": False,
        "raw_probability_accessible": False,
    }


def test_mixed_source_custody_stays_blocked(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7610-MIXED-CUSTODY keeps each source class distinct."""

    producer = tmp_path / "producer.json"
    gate = tmp_path / "gate.json"
    _write_json(
        producer,
        {
            "honest_verdict": "complete_null_measured",
            "verdict_class": "null",
            "flagged_adversarial": False,
        },
    )
    _write_json(gate, {"honest_verdict": "blocked_gate_check_failed"})
    valid = exp.classify_source(tmp_path, exp.SourceSpec("valid", Path("producer.json"), None))
    pre_gate = exp.classify_source(
        tmp_path, exp.SourceSpec("gate", Path("missing.json"), Path("gate.json"))
    )
    missing = exp.classify_source(tmp_path, exp.SourceSpec("missing", Path("absent.json"), None))
    assert valid["disposition"] == "authenticated_producer"
    assert pre_gate["disposition"] == "conductor_pre_gate"
    assert missing["disposition"] == "missing_producer"
    checks = [
        exp.check_row(
            "required",
            row["upstream"],
            row["path"],
            "eligible",
            True,
            row["eligible_for_science"],
            "eq",
        )
        for row in (pre_gate, missing)
    ]
    summary = exp.blocked_summary(checks)
    assert summary["failed_count"] == 2
    assert set(summary["first_failure"]) == exp.GATE_OPERAND_FIELDS


def test_raw_pilot_rows_rebuild_identity_pointers_and_label_join(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7610-MIXED-CUSTODY rebuilds available raw pilot rows."""

    component = "sha256:component"
    model_path = tmp_path / "raw/model.jsonl"
    labels_path = tmp_path / "raw/labels.jsonl"
    request_path = tmp_path / "raw/request.json"
    response_path = tmp_path / "raw/response.json"
    model = _model_row(component)
    _write_jsonl(model_path, [model])
    _write_jsonl(
        labels_path,
        [
            {
                "component_hash": component,
                "role": "pilot",
                "label": 1,
                "raw_probability": 0.75,
                "evaluator_only": True,
                "training_allowed": False,
            }
        ],
    )
    _write_json(request_path, {"request": True})
    _write_json(response_path, {"content": "[]"})
    protocol = {
        "raw_sidecars": {
            "model_inputs": {"pilot": exp.file_receipt(model_path, tmp_path)},
            "evaluator_stores": {"pilot": exp.file_receipt(labels_path, tmp_path)},
        }
    }
    pilot = {
        "pilot_rows": [
            {
                "component_hash": component,
                "input_record": model,
                "request_path": str(request_path),
                "request_sha256": exp.sha256_file(request_path),
                "response_path": str(response_path),
                "response_sha256": exp.sha256_file(response_path),
                "parser_outcome": "invalid_output",
                "censoring": "invalid_output",
                "usable_schema": False,
                "lossless_input": True,
                "evidence": [],
                "seed": 7604001,
            }
        ]
    }
    rows, receipts = exp.audit_pilot_rows(tmp_path, protocol, pilot)
    assert len(rows) == 1
    assert rows[0]["unit_id"] == component
    assert rows[0]["absolute_metrics"]["raw_probability"] == 0.75
    assert rows[0]["absolute_metrics"]["extraction_censor_indicator"] == 1.0
    assert rows[0]["censored"] is True
    assert rows[0]["raw_denominator"] == 1
    assert len(receipts) == 4

    duplicate_protocol = deepcopy(protocol)
    _write_jsonl(labels_path, [{"component_hash": "other"}])
    duplicate_protocol["raw_sidecars"]["evaluator_stores"]["pilot"] = exp.file_receipt(
        labels_path, tmp_path
    )
    with pytest.raises(ValueError, match="label_join_identity_mismatch"):
        exp.audit_pilot_rows(tmp_path, duplicate_protocol, pilot)
    _write_jsonl(
        labels_path,
        [
            {
                "component_hash": component,
                "role": "bad",
                "label": 1,
                "raw_probability": 0.75,
                "evaluator_only": True,
                "training_allowed": False,
            }
        ],
    )
    bad_role_protocol = deepcopy(protocol)
    bad_role_protocol["raw_sidecars"]["evaluator_stores"]["pilot"] = exp.file_receipt(
        labels_path, tmp_path
    )
    with pytest.raises(ValueError, match="pilot_label_role_invalid"):
        exp.audit_pilot_rows(tmp_path, bad_role_protocol, pilot)
    missing = deepcopy(pilot)
    missing["pilot_rows"][0]["component_hash"] = "missing"
    with pytest.raises(ValueError, match="pilot_component_missing"):
        exp.audit_pilot_rows(tmp_path, bad_role_protocol, missing)
    changed_input = deepcopy(pilot)
    changed_input["pilot_rows"][0]["input_record"]["complete_answer"] = "changed"
    with pytest.raises(ValueError, match="pilot_input_record_mismatch"):
        exp.audit_pilot_rows(tmp_path, bad_role_protocol, changed_input)
    with pytest.raises(ValueError, match="pilot_row_count_mismatch"):
        exp.audit_pilot_rows(tmp_path, bad_role_protocol, {"pilot_rows": []})


def test_pointer_and_label_mutations_fail_closed() -> None:
    """SCENARIO-REPORT-7610-STATIC rejects changed bytes and outcome leakage."""

    row = _model_row("u0")
    exp.validate_model_row(row)
    changed = deepcopy(row)
    changed["source_sentences"][0]["byte_end"] = 1
    with pytest.raises(ValueError, match="segment_roundtrip_invalid"):
        exp.validate_model_row(changed)
    leaked = deepcopy(row)
    leaked["label"] = 1
    with pytest.raises(ValueError, match="predictor_label_access"):
        exp.validate_model_row(leaked)


def test_static_probabilities_losses_costs_and_intervals_recompute() -> None:
    """SCENARIO-REPORT-7610-STATIC recomputes every operand without producer reducers."""

    fixture = exp.private_fixture()
    rows = exp.reconstruct_static_rows(
        fixture["static_inputs"], fixture["frozen_head"], seed=7610001
    )
    reduction = exp.reduce_static_rows(rows, draws=128, seed=7610002)
    assert len(rows) == 6
    assert reduction["independent_unit_count"] == 2
    assert reduction["metrics"]["factual"]["brier_denominator"] == 2
    assert reduction["controls_distinct"] is True
    assert all(0.0 < row["probability"] < 1.0 for row in rows)
    assert all(row["raw_brier_denominator"] == 1 for row in rows)
    assert set(reduction["contrasts"]) == {"factual_vs_erased", "factual_vs_deranged"}


def test_online_weights_roles_updates_and_restarts_recompute() -> None:
    """SCENARIO-REPORT-7610-ONLINE replays released feedback for every arm and order."""

    fixture = exp.private_fixture()
    replay = exp.audit_online_rows(
        fixture["online_rows"],
        initial_weights=fixture["initial_weights"],
        learning_rate=fixture["learning_rate"],
    )
    assert replay["qualified"] is True
    assert replay["order_count"] == 5
    assert replay["arms"] == sorted(exp.ONLINE_ARMS)
    assert replay["exactly_once_updates"] is True
    assert replay["restart_hashes_match"] is True
    assert replay["evaluator_denied"] is True


@pytest.mark.parametrize("mutation", exp.MUTATIONS)
def test_private_mutations_close_the_applicable_reader(mutation: str) -> None:
    """SCENARIO-REPORT-7610-MUTATIONS rejects each mandated private corruption."""

    fixture = exp.private_fixture()
    before = exp.canonical_hash(fixture)
    changed_path = exp.mutate_private_fixture(fixture, mutation)
    errors = exp.validate_private_fixture(fixture)
    assert exp.canonical_hash(fixture) != before
    assert changed_path
    assert exp.MUTATION_FAILURES[mutation] in errors


def test_private_mutation_receipts_bind_changed_bytes() -> None:
    """SCENARIO-REPORT-7610-MUTATIONS records hashes and exact failed checks."""

    receipts = exp.run_private_mutations()
    assert {row["mutation"] for row in receipts} == set(exp.MUTATIONS)
    assert all(row["passed"] is True for row in receipts)
    assert all(row["before_sha256"] != row["after_sha256"] for row in receipts)


def test_blocked_artifact_schema_checksum_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7610-TERMINAL keeps a complete block distinct from a null."""

    sources = [exp.classify_source(tmp_path, spec) for spec in exp.SOURCE_SPECS]
    checks = [
        exp.check_row(
            "required_scientific_producer",
            row["upstream"],
            row["path"],
            "exists_and_eligible",
            True,
            row["eligible_for_science"],
            "eq",
        )
        for row in sources[-3:]
    ]
    artifact = exp.build_blocked_artifact(
        tmp_path,
        checks,
        sources,
        pilot_rows=[],
        validation_receipts=exp.provisional_validation_receipts(tmp_path),
        duration_s=0.25,
        phase_spans=[],
        run_date="20260924",
    )
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["static_audit_ready_score"] == 0
    assert artifact["online_audit_ready_score"] == 0
    assert all(row["scientific_hypothesis_retired"] is False for row in artifact["retirement_rows"])
    assert exp.validate_artifact(artifact, root=tmp_path) == {"valid": True}
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path, root=tmp_path) == {"valid": True}
    replay = exp.independent_replay(path, root=tmp_path)
    assert replay["valid"] is True
    assert replay["row_count"] == 0


def test_artifact_rejects_false_positive_and_receipt_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7610-MUTATIONS blocks claim and terminal-receipt promotion."""

    artifact = exp.build_test_artifact(tmp_path)
    promoted = deepcopy(artifact)
    promoted["verdict_class"] = "positive"
    promoted["reproducibility_checksum"] = exp.reproducibility_checksum(promoted)
    with pytest.raises(ValueError, match="blocked_class_required"):
        exp.validate_artifact(promoted, root=tmp_path)
    drifted = deepcopy(artifact)
    drifted["validation_receipts"][0]["exit_code"] = 1
    drifted["reproducibility_checksum"] = exp.reproducibility_checksum(drifted)
    with pytest.raises(ValueError, match="validation_receipts_failed"):
        exp.validate_artifact(drifted, root=tmp_path)


def test_validation_plan_and_cli_modes_are_bounded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7610-TERMINAL freezes scoped and fresh-process readers."""

    commands = exp.build_validation_commands(tmp_path, tmp_path / "private")
    assert {command.name for command in commands} == set(exp.AFFECTED_CHECK_NAMES)
    terminal = exp.terminal_commands(tmp_path / "candidate.json", tmp_path)
    assert {command.name for command in terminal} == set(exp.TERMINAL_CHECK_NAMES)
    artifact = exp.build_test_artifact(tmp_path)
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert (
        exp.main(["--root", str(tmp_path), "--date", "20260924", "--cold-replay", str(path)]) == 0
    )
    assert '"mode": "cold_replay"' in capsys.readouterr().out
    assert (
        exp.main(["--root", str(tmp_path), "--date", "20260924", "--independent-reduce", str(path)])
        == 0
    )
    assert '"mode": "independent_reduction"' in capsys.readouterr().out
    with pytest.raises(ValueError, match="run_date_must_equal_20260924"):
        exp.parse_args(["--root", str(tmp_path), "--date", "20260923"])


def test_source_and_raw_reader_guards_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7610-MIXED-CUSTODY rejects invalid, flagged, and changed bytes."""

    invalid = tmp_path / "invalid.json"
    flagged = tmp_path / "flagged.json"
    _write_json(invalid, {"honest_verdict": "partial"})
    _write_json(
        flagged,
        {"honest_verdict": "complete_positive", "flagged_adversarial": True},
    )
    assert (
        exp.classify_source(tmp_path, exp.SourceSpec("invalid", Path("invalid.json"), None))[
            "disposition"
        ]
        == "invalid_producer"
    )
    assert (
        exp.classify_source(tmp_path, exp.SourceSpec("flagged", Path("flagged.json"), None))[
            "disposition"
        ]
        == "flagged_producer"
    )
    assert Path(exp.file_receipt(invalid, tmp_path / "other")["path"]).is_absolute()

    scalar = tmp_path / "scalar.jsonl"
    scalar.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        exp.load_jsonl(scalar)
    with pytest.raises(ValueError, match="sidecar_missing"):
        exp._resolve_receipt(tmp_path, {"path": "none", "bytes": 0, "sha256": "x"})
    receipt = exp.file_receipt(invalid, tmp_path)
    with pytest.raises(ValueError, match="sidecar_size_mismatch"):
        exp._resolve_receipt(tmp_path, {**receipt, "bytes": 0})
    with pytest.raises(ValueError, match="sidecar_hash_mismatch"):
        exp._resolve_receipt(tmp_path, {**receipt, "sha256": "sha256:bad"})
    with pytest.raises(ValueError, match="raw_pointer_hash_mismatch"):
        exp._audit_file_pointer(tmp_path, invalid, "sha256:bad")


def test_model_static_and_interval_guards_fail_closed() -> None:
    """SCENARIO-REPORT-7610-STATIC rejects malformed pointers and numeric operands."""

    row = _model_row("u0")
    bad_segment = deepcopy(row)
    bad_segment["source_sentences"][0]["byte_start"] = "zero"
    with pytest.raises(ValueError, match="segment_roundtrip_invalid"):
        exp.validate_model_row(bad_segment)
    with pytest.raises(ValueError, match="segment_roundtrip_invalid"):
        exp._roundtrip_segments("x", [])
    for field, value, error in (
        ("labels_accessible", True, "predictor_label_access"),
        ("raw_probability_accessible", True, "predictor_probability_access"),
        ("evidence_feature_names", [], "evidence_feature_roster_invalid"),
        ("complete_source", "", "complete_source_absent"),
        ("source_sha256", "bad", "source_hash_invalid"),
        ("component_hash", "", "component_hash_absent"),
    ):
        changed = deepcopy(row)
        changed[field] = value
        with pytest.raises(ValueError, match=error):
            exp.validate_model_row(changed)
    with pytest.raises(ValueError, match="static_logit_invalid"):
        exp._probability_from_logit(float("inf"))
    fixture = exp.private_fixture()
    with pytest.raises(ValueError, match="frozen_head_weights_invalid"):
        exp.reconstruct_static_rows(fixture["static_inputs"], {"weights": []}, seed=1)
    invalid_input = deepcopy(fixture["static_inputs"])
    invalid_input[0]["arm"] = "bad"
    with pytest.raises(ValueError, match="static_input_invalid"):
        exp.reconstruct_static_rows(invalid_input, fixture["frozen_head"], seed=1)
    duplicate = deepcopy(fixture["static_inputs"])
    duplicate.append(deepcopy(duplicate[0]))
    with pytest.raises(ValueError, match="static_duplicate_arm"):
        exp.reconstruct_static_rows(duplicate, fixture["frozen_head"], seed=1)
    incomplete = fixture["static_inputs"][:-1]
    with pytest.raises(ValueError, match="static_arm_roster_invalid"):
        exp.reconstruct_static_rows(incomplete, fixture["frozen_head"], seed=1)
    with pytest.raises(ValueError, match="interval_inputs_invalid"):
        exp._interval([], draws=0, seed=1)
    static_rows = deepcopy(fixture["static_rows"])
    static_rows.append(deepcopy(static_rows[0]))
    with pytest.raises(ValueError, match="static_duplicate_arm"):
        exp.reduce_static_rows(static_rows, draws=4, seed=1)
    static_rows = deepcopy(fixture["static_rows"])
    static_rows[0]["raw_brier_denominator"] = 0
    with pytest.raises(ValueError, match="static_denominator_invalid"):
        exp.reduce_static_rows(static_rows, draws=4, seed=1)
    with pytest.raises(ValueError, match="static_arm_roster_invalid"):
        exp.reduce_static_rows(fixture["static_rows"][:-1], draws=4, seed=1)


def test_online_reader_reports_all_lifecycle_corruptions() -> None:
    """SCENARIO-REPORT-7610-ONLINE names every chronology and state failure."""

    fixture = exp.private_fixture()
    rows = deepcopy(fixture["online_rows"])
    target = next(row for row in rows if row["accepted_update"] is True)
    target.update(
        prediction_probability=0.9,
        label_available_at_prediction=True,
        evaluator_access=True,
        feedback_origin="wrong",
        state_hash_before="bad",
        feedback_label=3,
        state_hash_after="bad",
        restart_hash_before="a",
        restart_hash_after="b",
        anchor_hash="changed",
    )
    width_target = next(
        row for row in rows if row["accepted_update"] is False and row["arm"] != "frozen"
    )
    width_target["features"] = []
    deranged = next(row for row in rows if row["arm"] == "guarded_deranged")
    deranged["released_component_ids"] = []
    deranged["prediction_index"] = deranged["release_index"]
    frozen = next(row for row in rows if row["arm"] == "frozen")
    frozen["accepted_update"] = True
    frozen["update_id"] = target["update_id"]
    replay = exp.audit_online_rows(
        rows, initial_weights=fixture["initial_weights"], learning_rate=fixture["learning_rate"]
    )
    assert {
        "online_feature_width_invalid",
        "evaluator_role_leak",
        "deranged_feedback_origin_invalid",
        "frozen_arm_updated",
        "state_hash_before_mismatch",
        "state_hash_after_mismatch",
        "restart_hash_mismatch",
        "anchor_provenance_changed",
    } <= set(replay["errors"])
    reduced_roster = exp.audit_online_rows(
        rows[:1], initial_weights=fixture["initial_weights"], learning_rate=fixture["learning_rate"]
    )
    assert "online_order_or_arm_roster_invalid" in reduced_roster["errors"]
    with pytest.raises(ValueError, match="unknown_mutation"):
        exp.mutate_private_fixture(fixture, "unknown")


def test_artifact_guard_matrix_rejects_each_claim_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7610-TERMINAL rejects identity, custody, and claim drift."""

    base = exp.build_test_artifact(tmp_path)

    def expect(mutator: object, error: str, *, refresh: bool = True) -> None:
        changed = deepcopy(base)
        assert callable(mutator)
        mutator(changed)
        if refresh:
            changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed, root=tmp_path)

    cases = (
        (lambda value: value.update(schema="bad"), "identity_mismatch"),
        (lambda value: value.update(run_date="bad"), "run_identity_mismatch"),
        (lambda value: value.update(honest_verdict="blocked"), "terminal_prefix_missing"),
        (
            lambda value: value.update(flagged_adversarial=True),
            "terminal_adversarial_outcome_invalid",
        ),
        (lambda value: value.update(model_specs=["bad"]), "model_specs_not_empty"),
        (lambda value: value.update(model_invoked=True), "current_model_calls_nonzero"),
        (lambda value: value.update(inference_substrate="bad"), "substrate_mismatch"),
        (lambda value: value.update(inference_substrate_class="bad"), "substrate_class_mismatch"),
        (lambda value: value["field_principles"].pop("rows"), "field_principles_incomplete"),
        (lambda value: value.update(static_audit_ready_score=1), "blocked_readiness_nonzero"),
        (lambda value: value.update(gate_check_summary={}), "blocked_gate_summary_invalid"),
        (lambda value: value["branch_conclusions"].pop(), "branch_conclusions_incomplete"),
        (
            lambda value: value["retirement_rows"][0].update(scientific_hypothesis_retired=True),
            "blocked_retirement_invalid",
        ),
        (
            lambda value: value["mutation_receipts"][0].update(passed=False),
            "mutation_receipts_invalid",
        ),
        (lambda value: value.update(rows=[{"bad": True}]), "pilot_row_schema_invalid"),
        (lambda value: value["source_artifact_hashes"].pop(), "producer_custody_incomplete"),
        (
            lambda value: next(
                row
                for row in value["source_artifact_hashes"]
                if row["upstream"] == "exp7604-evidence-pilot"
            ).update(disposition="authenticated_producer"),
            "available_pilot_rows_missing",
        ),
    )
    for mutator, error in cases:
        expect(mutator, error)
    expect(
        lambda value: value.update(submitted_externally=True), "checksum_mismatch", refresh=False
    )

    source = tmp_path / "bound.json"
    source.write_text("{}", encoding="utf-8")
    receipt = {**exp.file_receipt(source, tmp_path), "upstream": "extra"}
    expect(
        lambda value: value["source_artifact_hashes"].append({**receipt, "sha256": "sha256:bad"}),
        "source_hash_mismatch",
    )


def test_real_worktree_inventory_and_raw_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7610-MIXED-CUSTODY replays the current eight pilot rows."""

    checks, sources = exp.collect_preconditions(exp.REPO_ROOT)
    failures = [row for row in checks if row["passed"] is not True]
    assert failures
    by_source = {row["upstream"]: row for row in sources}
    protocol = exp.load_json(
        exp.authenticate_source_receipt(
            by_source["exp7602-evidence-requalification"], exp.REPO_ROOT
        )
    )
    pilot = exp.load_json(
        exp.authenticate_source_receipt(by_source["exp7604-evidence-pilot"], exp.REPO_ROOT)
    )
    rows, _raw = exp.audit_pilot_rows(exp.REPO_ROOT, protocol, pilot)
    assert len(rows) == 8
    artifact = exp.build_blocked_artifact(
        exp.REPO_ROOT,
        failures,
        sources,
        pilot_rows=rows,
        validation_receipts=exp.provisional_validation_receipts(exp.REPO_ROOT),
        duration_s=0.5,
        phase_spans=[],
        run_date="20260924",
    )
    path = tmp_path / "real-candidate.json"
    exp.atomic_json(path, artifact)
    replay = exp.independent_replay(path, root=exp.REPO_ROOT)
    assert replay["row_count"] == 8

    mismatch = deepcopy(artifact)
    mismatch["rows"][0]["raw_numerator"] = 1
    mismatch["reproducibility_checksum"] = exp.reproducibility_checksum(mismatch)
    exp.atomic_json(path, mismatch)
    with pytest.raises(ValueError, match="independent_pilot_reduction_mismatch"):
        exp.independent_replay(path, root=exp.REPO_ROOT)
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_not_object"):
        exp.cold_replay(empty, root=tmp_path)
