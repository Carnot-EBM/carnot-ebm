"""Tests for REQ-VERIFY-7602 and SCENARIO-VERIFY-7602-*.

The fixtures cover only V664 behavior. V663 owns sentence segmentation,
evidence reduction, and the original fixed-role selector.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7588_v663_evidence_protocol as v663
from carnot import experiment_7602_v664_evidence_requalification as exp


def _prompt(source: str, question: str, answer: str) -> str:
    return (
        v663._PROMPT_PREFIX
        + source
        + v663._QUESTION_MARKER
        + question
        + v663._ANSWER_MARKER
        + answer
        + v663._OPTION_MARKER
        + v663._PROMPT_SUFFIX
    )


def _group(role: str, index: int) -> dict:
    source = f"Source {role} {index}.\n\n| key | value |\n| --- | --- |\n| n | {index} |\n"
    question = f"Is claim {index} supported?"
    answer = f"Claim {index} is supported. However, it is not unlimited.\n"
    prompt = _prompt(source, question, answer)
    return {
        "source_id": f"sha256:{role}-{index:03d}",
        "group_id": f"sha256:group-{role}-{index:03d}",
        "role": role,
        "official_split": "test" if role == "test" else "train",
        "probability": (index + 1) / 200,
        "label": index % 2,
        "context": prompt,
        "response": "",
        "context_sha256": v663.canonical_hash(prompt),
        "response_sha256": v663.canonical_hash(answer),
        "request_hashes": [f"sha256:req-{role}-{index:03d}"],
    }


def _roles() -> dict[str, list[dict]]:
    return {
        role: [_group(role, index) for index in range(count)]
        for role, count in v663.SOURCE_ROLE_COUNTS.items()
    }


def _selected() -> dict:
    return exp.select_requalified_roles(_roles())


def _passing_receipts() -> list[dict]:
    return [
        {
            "name": "focused_pytest",
            "command": ".venv/bin/pytest tests/python/test_experiment_7602.py",
            "command_argv": [".venv/bin/pytest", "tests/python/test_experiment_7602.py"],
            "scope": "explicit_tests",
            "exit_code": 0,
            "duration_s": 1.0,
            "log_path": "/tmp/exp7602/pytest.log",
            "log_sha256": "sha256:" + "1" * 64,
            "passed": True,
            "timed_out": False,
            "worktree": str(exp.REPO_ROOT),
        }
    ]


def test_exact_selector_and_learning_schedule_are_frozen() -> None:
    """SCENARIO-VERIFY-7602-CUSTODY freezes roles, anchors, and online access."""

    selected = _selected()
    schedule = exp.build_learning_schedule(selected)
    assert {role: len(rows) for role, rows in selected["scored"].items()} == v663.ROLE_COUNTS
    assert len(selected["pilot"]) == 8
    assert len(schedule["fit_optimization_ids"]) == 64
    assert len(schedule["fit_anchor_ids"]) == 16
    assert set(schedule["fit_optimization_ids"]).isdisjoint(schedule["fit_anchor_ids"])
    assert schedule["tune_role"] == "hyperparameter_selection_only"
    assert schedule["policy_role"] == "strongest_comparator_selection_only"
    assert schedule["evaluation_role"] == "evaluator_only"
    assert len(schedule["online_blocks"]) == 10
    for block_index, block in enumerate(schedule["online_blocks"]):
        assert block["block_index"] == block_index
        assert len(block["update_ids"]) == 4
        assert len(block["admission_ids"]) == 4
        assert set(block["update_ids"]).isdisjoint(block["admission_ids"])
        assert block["label_release_lag"] == 8
    assert schedule["admission_labels_trainable"] is False
    assert schedule["evaluation_labels_trainable"] is False


def test_parser_rejects_empty_answer_and_delimiter_collision() -> None:
    """SCENARIO-VERIFY-7602-PARSER rejects ambiguous or absent historical bytes."""

    empty = _group("fit", 1)
    empty["context"] = _prompt("Source.", "Question?", "")
    with pytest.raises(ValueError, match="original_text_contract_invalid"):
        exp.restore_authenticated_group(empty)

    collision = _group("fit", 2)
    collision["context"] = _prompt(
        "Source.", "Question?", f"Answer before{v663._ANSWER_MARKER}answer after"
    )
    with pytest.raises(ValueError, match="parser_delimiter_collision"):
        exp.restore_authenticated_group(collision)


def test_model_record_rejects_qualifier_omission_and_label_access() -> None:
    """SCENARIO-VERIFY-7602-PARSER keeps exact bytes and blocks predictor labels."""

    restored = exp.restore_authenticated_group(_group("fit", 3))
    record = exp.build_model_record(restored, partition="fit_optimization")
    assert exp.validate_model_record(record, restored) is True
    assert "label" not in record
    assert "probability" not in record
    assert "raw_probability_offset" not in record

    omitted = deepcopy(record)
    omitted["complete_answer"] = omitted["complete_answer"].replace(
        " However, it is not unlimited.", ""
    )
    with pytest.raises(ValueError, match="answer_byte_identity_mismatch"):
        exp.validate_model_record(omitted, restored)

    leaked = deepcopy(record)
    leaked["label"] = restored["label"]
    with pytest.raises(ValueError, match="predictor_label_access"):
        exp.validate_model_record(leaked, restored)


def test_duplicate_group_and_bad_label_join_fail_closed() -> None:
    """SCENARIO-VERIFY-7602-CUSTODY rejects duplicate identity and bad labels."""

    roles = _roles()
    roles["tune"][0]["source_id"] = roles["fit"][0]["source_id"]
    with pytest.raises(ValueError, match="component_duplicate"):
        exp.select_requalified_roles(roles)

    selected = _selected()
    model_rows, evaluator_rows = exp.build_role_records(selected)
    bad = deepcopy(evaluator_rows)
    bad["fit"][0]["component_hash"] = "sha256:not-the-selected-group"
    with pytest.raises(ValueError, match="label_join_identity_mismatch"):
        exp.validate_label_join(model_rows, bad)

    bad = deepcopy(evaluator_rows)
    bad["online"][0]["label"] = 2
    with pytest.raises(ValueError, match="label_join_value_invalid"):
        exp.validate_label_join(model_rows, bad)


def test_role_sidecars_separate_predictor_and_evaluator_bytes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7602-SIDECARS freezes isolated role-separated stores."""

    selected = _selected()
    bundle = exp.freeze_requalification_files(tmp_path, selected, root=tmp_path)
    protocol = bundle["protocol"]
    assert Path(bundle["protocol_path"]).is_file()
    assert protocol["selection_salt"] == v663.SELECTION_SALT
    assert protocol["role_counts"] == {**v663.ROLE_COUNTS, "pilot": 8}
    assert protocol["evidence_feature_names"] == list(v663.EVIDENCE_FEATURE_NAMES)
    assert protocol["output_contract"]["relations"] == [
        "contradicts",
        "supports",
        "unknown",
    ]
    assert protocol["output_contract"]["maximum_proposed_links"] == 6
    assert protocol["fresh_confirmatory_claim_allowed"] is False
    assert protocol["fit_partition_counts"] == {"optimization": 64, "old_distribution_anchor": 16}
    assert set(bundle["raw_sidecars"]["model_inputs"]) == {
        "fit",
        "tune",
        "policy",
        "online",
        "evaluation",
        "pilot",
    }
    assert set(bundle["raw_sidecars"]["evaluator_stores"]) == {
        "fit",
        "tune",
        "policy",
        "online",
        "evaluation",
        "pilot",
    }

    for receipt in bundle["raw_sidecars"]["model_inputs"].values():
        rows = [json.loads(line) for line in (tmp_path / receipt["path"]).read_text().splitlines()]
        assert rows
        assert all("label" not in row and "probability" not in row for row in rows)
        assert all("raw_probability_offset" not in row for row in rows)
        assert all(row["complete_source"] and row["complete_question"] for row in rows)
        assert all(row["complete_answer"] for row in rows)
        assert all(row["fresh_confirmatory_claim_allowed"] is False for row in rows)

    online_store = bundle["evaluator_rows"]["online"]
    assert sum(row["online_phase"] == "update" for row in online_store) == 40
    assert sum(row["online_phase"] == "admission" for row in online_store) == 40
    assert all(
        row["training_allowed"] is False
        for row in online_store
        if row["online_phase"] == "admission"
    )
    assert all(row["label_release_lag"] == 8 for row in online_store)
    assert all(row["training_allowed"] is False for row in bundle["evaluator_rows"]["evaluation"])
    assert bundle["row_reduction"]["unique_units"] == 248
    assert bundle["row_reduction"]["row_count"] == 248 * len(v663.PROTOCOL_ARMS)


def test_historical_failure_record_preserves_bytes_without_inventing_cause(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7602-AUTH separates preserved bytes from lost process state."""

    path = tmp_path / "historical.json"
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_blocked_selected_role_roster",
                "verdict_class": "blocked",
                "gate_check_summary": {
                    "first_failure": {
                        "check": "selected_role_roster",
                        "observed": "source_group_incomplete",
                    }
                },
            }
        )
    )
    record = exp.historical_failure_record(path, root=tmp_path)
    assert record["terminal_artifact_bytes_preserved"] is True
    assert record["exact_runtime_bytes_reconstructable"] is False
    assert record["exact_process_state_available"] is False
    assert record["historical_cause"] is None
    assert record["preserved_verdict"] == "complete_blocked_selected_role_roster"
    assert record["preserved_observed_failure"] == "source_group_incomplete"
    assert record["artifact_sha256"] == exp.sha256_file(path)


def test_ready_artifact_is_null_and_cold_reducible(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7602-E2E keeps readiness separate from benefit."""

    selected = _selected()
    bundle = exp.freeze_requalification_files(tmp_path, selected, root=tmp_path)
    checks = [
        exp.precondition(
            "authenticated_inputs",
            "v662_and_v663",
            tmp_path / "input.json",
            "all_exact_bytes",
            True,
            True,
        ),
        exp.precondition(
            "fresh_process_selector",
            "committed_v663_module",
            tmp_path / "selector.json",
            "selected_ids_sha256",
            bundle["selected_ids_sha256"],
            bundle["selected_ids_sha256"],
        ),
    ]
    historical = {
        "terminal_artifact_bytes_preserved": True,
        "exact_runtime_bytes_reconstructable": False,
        "exact_process_state_available": False,
        "historical_cause": None,
        "preserved_verdict": "complete_blocked_selected_role_roster",
        "preserved_observed_failure": "source_group_incomplete",
        "artifact_sha256": "sha256:" + "a" * 64,
    }
    artifact = exp.build_artifact(
        bundle,
        preconditions=checks,
        source_hashes=[
            {
                "path": "results/experiment_7588_v663_evidence_protocol.json",
                "sha256": "sha256:" + "a" * 64,
                "bytes": 10,
                "producer": "v663_failure_receipt",
                "source_class": "authenticated_producer",
            }
        ],
        validation_receipts=_passing_receipts(),
        duration_s=2.5,
        historical_failure=historical,
        selector_replay={
            "all_preconditions_passed": True,
            "source_role_counts": v663.SOURCE_ROLE_COUNTS,
            "scored_role_counts": v663.ROLE_COUNTS,
            "pilot_count": 8,
            "selected_unique": 248,
            "selected_ids_sha256": bundle["selected_ids_sha256"],
            "salt": v663.SELECTION_SALT,
        },
        require_validation=True,
    )
    assert artifact["honest_verdict"] == "complete_null_evidence_requalification_ready"
    assert artifact["verdict_class"] == "null"
    assert artifact["flagged_adversarial"] is False
    assert artifact["evidence_protocol_ready_score"] == 1
    assert artifact["fixture_control_verdict_class"] == "circular_positive"
    assert artifact["empirical_benefit_measured"] is False
    assert artifact["fresh_confirmatory_claim_allowed"] is False
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["role_counts"] == {**v663.ROLE_COUNTS, "pilot": 8}
    assert artifact["sample_size_budget"]["observed_independent_units"] == 248
    assert {row["category"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])

    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)
    assert exp.cold_replay(candidate, root=tmp_path)["ready"] is True
    reduced = exp.independent_reduce_artifact(candidate, root=tmp_path)
    assert reduced["passed"] is True
    assert reduced["row_count"] == 744

    changed = deepcopy(artifact)
    changed["rows"][0]["raw_numerator"] += 1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="row_reduction_invalid"):
        exp.validate_artifact(changed, root=tmp_path)


def test_blocked_artifact_is_complete_and_names_exact_operands(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7602-BLOCKED reports external drift, never partial work."""

    check = exp.precondition(
        "v663_module_committed_bytes",
        "git_commit",
        tmp_path / "module.py",
        "worktree_sha256",
        "sha256:expected",
        "sha256:observed",
    )
    artifact = exp.build_blocked_artifact(
        [check],
        [],
        duration_s=0.25,
        historical_failure={
            "terminal_artifact_bytes_preserved": True,
            "exact_runtime_bytes_reconstructable": False,
            "historical_cause": None,
        },
    )
    assert artifact["honest_verdict"] == "complete_blocked_v663_module_committed_bytes"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["rows"] == []
    assert artifact["validation_receipts"] == []
    assert artifact["sample_size_budget"]["unstarted_independent_units"] == 248
    first = artifact["gate_check_summary"]["first_failure"]
    assert set(first) >= {"check", "upstream", "path", "field", "op", "expected", "observed"}
    assert Path(first["path"]).is_absolute()
    assert exp.validate_artifact(artifact, root=tmp_path)["blocked"] is True


def test_precondition_operator_and_principles_are_closed() -> None:
    """REQ-VERIFY-7602 keeps gate operators and terminal classes explicit."""

    row = exp.precondition("check", "source", Path("relative"), "field", (1, 2), 2, op="in")
    assert row["passed"] is True
    with pytest.raises(ValueError, match="precondition_operator_invalid"):
        exp.precondition("check", "source", Path("relative"), "field", 1, 1, op="lt")
    assert exp.field_principles()["verdict_class"].startswith("Exactly positive")


def test_defensive_custody_branches_reject_invalid_inputs(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7602-PARSER exercises V664-only fail-closed controls."""

    with pytest.raises(ValueError, match="original_text_contract_invalid"):
        exp.restore_authenticated_group({"context": None})

    selected = _selected()
    incomplete = deepcopy(selected)
    incomplete["scored"]["online"] = incomplete["scored"]["online"][:-1]
    with pytest.raises(ValueError, match="online_block_incomplete"):
        exp.build_learning_schedule(incomplete)

    restored = exp.restore_authenticated_group(_group("fit", 4))
    record = exp.build_model_record(restored, partition="fit_optimization")
    record["labels_accessible"] = True
    with pytest.raises(ValueError, match="predictor_label_access"):
        exp.validate_model_record(record, restored)

    with pytest.raises(ValueError, match="label_join_identity_mismatch"):
        exp.validate_label_join({"fit": []}, {"online": []})
    with pytest.raises(ValueError, match="predictor_label_access"):
        exp.validate_label_join(
            {"fit": [{"component_hash": "a", "label": 0}]},
            {"fit": [{"component_hash": "a", "label": 0}]},
        )

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    assert exp._load_object(malformed) == {}


def test_committed_v663_bytes_and_selector_snapshot_are_exact() -> None:
    """SCENARIO-VERIFY-7602-AUTH binds the Git blob and selected roster."""

    check, record = exp._committed_source_record(exp.REPO_ROOT, exp.V663_MODULE_PATH, "v663_module")
    assert check["passed"] is True
    assert record["sha256"] == exp.sha256_file(exp.REPO_ROOT / exp.V663_MODULE_PATH)
    assert record["git_blob_sha1"]
    assert len(record["commit"]) == 40

    selected = _selected()
    snapshot = exp.build_selector_snapshot(
        [exp.precondition("input", "v662", exp.REPO_ROOT / exp.V662_PATH, "available", True, True)],
        selected,
    )
    assert snapshot["all_preconditions_passed"] is True
    assert snapshot["source_role_counts"] == v663.SOURCE_ROLE_COUNTS
    assert snapshot["scored_role_counts"] == v663.ROLE_COUNTS
    assert snapshot["pilot_count"] == 8
    assert snapshot["selected_unique"] == 248
    assert snapshot["selected_ids_sha256"] == v663.canonical_hash(selected["selected_ids"])


def _ready_fixture(tmp_path: Path) -> dict:
    selected = _selected()
    bundle = exp.freeze_requalification_files(tmp_path, selected, root=tmp_path)
    checks = [
        exp.precondition("authenticated_inputs", "v662", tmp_path / "source", "exact", True, True)
    ]
    historical = {
        "terminal_artifact_bytes_preserved": True,
        "exact_runtime_bytes_reconstructable": False,
        "exact_process_state_available": False,
        "historical_cause": None,
        "preserved_verdict": "complete_blocked_selected_role_roster",
        "preserved_observed_failure": "source_group_incomplete",
        "artifact_sha256": "sha256:" + "a" * 64,
    }
    return exp.build_artifact(
        bundle,
        preconditions=checks,
        source_hashes=[],
        validation_receipts=_passing_receipts(),
        duration_s=1.0,
        historical_failure=historical,
        selector_replay=exp.build_selector_snapshot(checks, selected),
    )


def test_cold_reader_rejects_terminal_field_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7602-E2E authenticates every readiness-bearing field."""

    artifact = _ready_fixture(tmp_path)

    def rejected(field: str, value: object, error: str, *, recompute: bool = True) -> None:
        changed = deepcopy(artifact)
        changed[field] = value
        if recompute:
            changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed, root=tmp_path)

    with pytest.raises(ValueError, match="artifact_object_required"):
        exp.validate_artifact([])
    rejected("schema", "wrong", "identity_invalid")
    rejected("duration_s", 9.0, "checksum_invalid", recompute=False)
    rejected("MODEL_SPECS", ["model"], "model_specs_not_empty")
    rejected("no_model_load", False, "model_invocation_claim_invalid")
    rejected("invocation_counts", {}, "invocation_counts_nonzero")
    rejected("fresh_confirmatory_claim_allowed", True, "freshness_invalid")
    rejected("field_principles", {}, "field_principles_invalid")
    rejected("verdict_class", "unknown", "verdict_class_invalid")
    rejected("verdict_class", "positive", "terminal_verdict_invalid")
    rejected("inference_substrate_class", "live", "substrate_invalid")
    rejected("protocol_sha256", "sha256:wrong", "protocol_hash_invalid")
    rejected("role_counts", {}, "role_counts_invalid")
    rejected("rows", [], "rows_missing")
    rejected("independent_row_reduction", {}, "row_reduction_mismatch")
    rejected("raw_sidecars", [], "raw_sidecar_invalid")
    rejected(
        "raw_sidecars",
        {"model_inputs": {}, "evaluator_stores": {}},
        "raw_sidecar_invalid",
    )
    bad_sidecars = deepcopy(artifact["raw_sidecars"])
    bad_sidecars["model_inputs"]["fit"]["sha256"] = "sha256:wrong"
    rejected("raw_sidecars", bad_sidecars, "raw_sidecar_invalid")
    rejected("evidence_protocol_ready_score", 2, "ready_score_invalid")
    rejected("historical_failure_reconstruction", {}, "historical_failure_not_preserved")
    overread = deepcopy(artifact["historical_failure_reconstruction"])
    overread["historical_cause"] = "invented"
    rejected("historical_failure_reconstruction", overread, "historical_failure_overinterpreted")
    rejected("validation_receipts", [], "validation_invalid")

    fewer = deepcopy(artifact)
    removed_id = fewer["rows"][0]["unit_id"]
    fewer["rows"] = [row for row in fewer["rows"] if row["unit_id"] != removed_id]
    fewer["independent_row_reduction"] = v663.reduce_protocol_rows(fewer["rows"])
    fewer["reproducibility_checksum"] = exp.reproducibility_checksum(fewer)
    with pytest.raises(ValueError, match="row_unit_count_invalid"):
        exp.validate_artifact(fewer, root=tmp_path)


def test_cold_reader_rejects_protocol_and_blocked_summary_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7602-BLOCKED keeps protocol and blocker operands exact."""

    artifact = _ready_fixture(tmp_path)
    protocol_path = tmp_path / str(artifact["protocol_path"])
    original = json.loads(protocol_path.read_text(encoding="utf-8"))

    for field, value, error in (
        ("selection_salt", "changed", "selection_salt_invalid"),
        ("model_records_include_labels", True, "reader_isolation_invalid"),
        ("fresh_confirmatory_claim_allowed", True, "protocol_freshness_invalid"),
    ):
        changed_protocol = deepcopy(original)
        changed_protocol[field] = value
        exp.atomic_json(protocol_path, changed_protocol)
        changed_artifact = deepcopy(artifact)
        changed_artifact["protocol_sha256"] = exp.sha256_file(protocol_path)
        changed_artifact["reproducibility_checksum"] = exp.reproducibility_checksum(
            changed_artifact
        )
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed_artifact, root=tmp_path)
    exp.atomic_json(protocol_path, original)

    check = exp.precondition(
        "external", "source", tmp_path / "source", "field", "expected", "observed"
    )
    blocked = exp.build_blocked_artifact(
        [check],
        [],
        duration_s=1.0,
        historical_failure={
            "terminal_artifact_bytes_preserved": True,
            "exact_runtime_bytes_reconstructable": False,
            "historical_cause": None,
        },
    )

    def rejected(mutate: object, error: str) -> None:
        changed = deepcopy(blocked)
        assert callable(mutate)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed, root=tmp_path)

    rejected(lambda value: value.update(honest_verdict="blocked"), "blocked_verdict_invalid")
    rejected(
        lambda value: value["gate_check_summary"].update(first_failure={}),
        "blocked_gate_summary_invalid",
    )
    rejected(lambda value: value.update(rows=[{"invented": True}]), "blocked_measurement_invalid")
