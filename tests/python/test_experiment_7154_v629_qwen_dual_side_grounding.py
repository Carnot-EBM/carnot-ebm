"""Focused tests for REQ-VERIFY-7154 and SCENARIO-VERIFY-7154-*.

The tests use frozen checked-in inputs and temporary output paths. They do not
start a model or change the checked-in result artifact.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7154_v629_qwen_dual_side_grounding as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verification/spec.md"
RUNTIME_PATH = REPO / "results/experiment_7153_v629_grounding_runtime.json"
FIXTURE_PATH = REPO / "results/experiment_7138_v627_relational_fixture.json"


def _upstreams() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load frozen test inputs without writing any repository state."""

    return (
        json.loads(RUNTIME_PATH.read_text(encoding="utf-8")),
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
    )


def _span(document: str, text: str) -> dict[str, Any]:
    """Build one exact provenance span at the start of a test document."""

    token = text.split()[0]
    return {
        "document": document,
        "start": 0,
        "end": len(token),
        "text": token,
        "sha256": exp.sha256_text(token),
    }


def _bundle(source: str, response: str, *, unknown: bool) -> dict[str, Any]:
    """Build one valid relation with source or response provenance."""

    provenance = _span("response" if unknown else "source", response if unknown else source)
    return {
        "entities": [{"entity_id": "e001", "entity_type": "other", "canonical_name": "subject"}],
        "relations": [
            {
                "relation_id": "r001",
                "subject_entity_id": "e001",
                "predicate": "has_attribute",
                "object": {
                    "type": "unknown" if unknown else "string",
                    "value": None if unknown else "supported",
                    "unit": None,
                    "unknown_reason": "not_stated_in_source" if unknown else None,
                },
                "provenance": [provenance],
            }
        ],
    }


def _model_evidence(
    runtime: dict[str, Any], fixture: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create deterministic raw rows for every frozen call opportunity."""

    schedule = exp.expected_call_schedule(runtime)
    model_by_id = {row["fixture_id"]: row for row in fixture["model_view_rows"]}
    truth_by_id = {
        row["fixture_id"]: row["response_label"] for row in fixture["sealed_scorer_rows"]
    }
    first_outputs: dict[tuple[str, str], str] = {}
    prompt_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for index, call in enumerate(schedule):
        fixture_id = call["fixture_id"]
        arm = call["arm"]
        pass_index = call["pass_index"]
        prompt = call["prompt"]
        if pass_index == 2:
            prompt = prompt.replace("{{FIRST_PASS_OUTPUT}}", first_outputs[(fixture_id, arm)])
        prompt_rows.append(
            {
                "call_id": call["call_id"],
                "fixture_id": fixture_id,
                "arm": arm,
                "pass_index": pass_index,
                "output_token_limit": call["output_token_limit"],
                "template_prompt_sha256": call["prompt_sha256"],
                "prompt_sha256": exp.sha256_text(prompt),
            }
        )
        truth = truth_by_id[fixture_id]
        if arm in {"relational_sql", "dual_side"} and pass_index == 1:
            view = model_by_id[fixture_id]
            raw_output = json.dumps(
                _bundle(
                    view["source_text"],
                    view["response_text"],
                    unknown=truth == "hallucinated",
                )
            )
        elif arm in {"relational_sql", "dual_side"}:
            raw_output = json.dumps({"query": exp.REQUIRED_SQL_QUERY})
        else:
            # The direct control has headroom. The self-check final pass is correct.
            direct_wrong = index % 5 == 0
            prediction = (
                ("clean" if truth == "hallucinated" else "hallucinated")
                if arm == "direct" and direct_wrong
                else truth
            )
            status = "supported" if prediction == "clean" else "unsupported"
            raw_output = json.dumps({"status": status, "confidence": 0.9})
        if pass_index == 1:
            first_outputs[(fixture_id, arm)] = raw_output
        raw_rows.append(
            {
                "call_id": call["call_id"],
                "fixture_id": fixture_id,
                "arm": arm,
                "pass_index": pass_index,
                "raw_output": raw_output,
                "raw_output_sha256": exp.sha256_text(raw_output),
                "raw_response_sha256": exp.sha256_text("{}"),
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "latency_s": 0.1,
                "terminal_state": "complete",
                "error": None,
            }
        )
    return schedule, prompt_rows, raw_rows


def _complete_artifact(n_boot: int = 64) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build one complete four-arm artifact through the public reducers."""

    runtime, fixture = _upstreams()
    schedule, prompt_rows, raw_rows = _model_evidence(runtime, fixture)
    seal = exp.model_evidence_seal(prompt_rows, raw_rows, schedule)
    parse_rows, structure_rows, execution_rows, arm_rows = exp.score_frozen_outputs(
        runtime, fixture, prompt_rows, raw_rows, seal
    )
    artifact = exp.base_artifact(exp.RUN_DATE)
    spec = deepcopy(runtime["MODEL_SPECS"][0])
    artifact.update(
        {
            "status": "preconditions_passed",
            "source_artifact_hashes": {
                str(exp.RUNTIME_PATH): exp.sha256_file(RUNTIME_PATH),
                str(exp.FIXTURE_PATH): exp.sha256_file(FIXTURE_PATH),
            },
            "rows": arm_rows,
            "MODEL_SPECS": [spec],
            "model_identity_rows": deepcopy(runtime["model_identity_rows"]),
            "gpu_rows": [
                {"phase": "before", "ok": True, "gpu_count": 2},
                {"phase": "model_loaded", "ok": True, "gpu_count": 2},
                {"phase": "after_teardown", "ok": True, "gpu_count": 2, "compute_apps": []},
            ],
            "model_load_receipts": [
                {
                    "model_id": exp.QWEN_MODEL_ID,
                    "sha256": spec["sha256"],
                    "health": {"ok": True},
                    "gpu_offload_confirmed": True,
                    "process_returncode": -15,
                    "cleanup": {"leak_free": True},
                    "duration_s": 70.0,
                }
            ],
            "frozen_fixture_ids": list(exp.FROZEN_FIXTURE_IDS),
            "frozen_schedule_hash": runtime["frozen_schedule_hash"],
            "prompt_hash_rows": prompt_rows,
            "raw_output_rows": raw_rows,
            "parse_rows": parse_rows,
            "source_structure_rows": structure_rows,
            "sql_execution_rows": execution_rows,
            "label_exposure_count": seal["label_exposure_count"],
            "schedule_identity_score": 1,
        }
    )
    checks = [
        exp.gate_row(name, True, True, upstream="synthetic", field=name)
        for name in exp.REQUIRED_PRECONDITION_CHECKS
    ]
    final = exp.finalize_artifact(artifact, checks, duration_s=120.0, n_boot=n_boot)
    return final, runtime, fixture


def test_req_verify_7154_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7154 owns each focused scenario and required field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7154") :]
    for scenario in (
        "FIRST-WRITE",
        "SEALING",
        "SCHEDULE",
        "STRUCTURE",
        "SQL",
        "METRICS",
        "VERDICT",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7154-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_7154_first_write_has_exact_block_receipt(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7154-FIRST-WRITE keeps a full blocked shape."""

    path = tmp_path / "artifact.json"
    running = exp.initialize_artifact(path, exp.RUN_DATE)
    assert set(running) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(running["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    check = exp.gate_row(
        "runtime_ready",
        1,
        0,
        False,
        upstream="results/experiment_7153_v629_grounding_runtime.json",
        field="grounding_runtime_ready_score",
    )
    blocked = exp.finish_blocked(running, path, [check], duration_s=0.5)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"] == {
        "upstream": "results/experiment_7153_v629_grounding_runtime.json",
        "field": "grounding_runtime_ready_score",
        "failed_check": "runtime_ready",
        "expected_value": 1,
        "observed_value": 0,
        "passed": False,
    }
    assert blocked["honest_verdict"] == "blocked_runtime_ready"
    assert exp.validate_artifact(blocked) == []


def test_scenario_verify_7154_schedule_and_label_seal_are_exact() -> None:
    """SCENARIO-VERIFY-7154-SEALING and SCHEDULE reject identity drift."""

    runtime, fixture = _upstreams()
    schedule, prompt_rows, raw_rows = _model_evidence(runtime, fixture)
    assert len(schedule) == 168
    assert exp.schedule_identity_errors(schedule, runtime) == []
    assert {arm: sum(row["arm"] == arm for row in schedule) for arm in exp.ARM_IDS} == {
        "direct": 24,
        "self_check": 48,
        "relational_sql": 48,
        "dual_side": 48,
    }
    seal = exp.model_evidence_seal(prompt_rows, raw_rows, schedule)
    assert seal["sealed"] is True
    assert seal["label_exposure_count"] == 0
    assert len(exp.open_exact_labels(fixture, seal)) == 24

    broken_schedule = deepcopy(schedule)
    broken_schedule[0]["prompt_sha256"] = "sha256:bad"
    assert "schedule_prompt_hash_mismatch" in exp.schedule_identity_errors(broken_schedule, runtime)
    leaked = deepcopy(raw_rows)
    leaked[0]["response_label"] = "clean"
    rejected = exp.model_evidence_seal(prompt_rows, leaked, schedule)
    assert rejected["sealed"] is False
    assert rejected["label_exposure_count"] == 1
    with pytest.raises(ValueError, match="model evidence is not sealed"):
        exp.open_exact_labels(fixture, rejected)


def test_scenario_verify_7154_structure_precedes_exact_sql() -> None:
    """SCENARIO-VERIFY-7154-STRUCTURE and SQL fail closed before execution."""

    source = "Parcel is blue."
    response = "Parcel is green."
    documents = {"source": source, "response": response}
    relation = exp.parse_relation_output(
        json.dumps(_bundle(source, response, unknown=True)), documents
    )
    query = exp.parse_sql_output(json.dumps({"query": exp.REQUIRED_SQL_QUERY}))
    structure = exp.check_source_structure(relation, query)
    assert structure == {
        "relation_schema_valid": True,
        "source_provenance_valid": True,
        "table_valid": True,
        "fields_valid": True,
        "join_valid": True,
        "filter_valid": True,
        "aggregation_valid": True,
        "comparison_valid": True,
        "structure_valid": True,
        "errors": [],
    }
    execution = exp.execute_checked_sql(relation, query, documents, structure)
    assert execution["status"] == "ok"
    assert execution["row_count"] == 1
    assert exp.reduce_sql_prediction(execution)["prediction"] == "hallucinated"

    altered = exp.parse_sql_output(
        json.dumps({"query": "SELECT * FROM grounded_relations LIMIT 32"})
    )
    rejected_structure = exp.check_source_structure(relation, altered)
    assert rejected_structure["structure_valid"] is False
    assert set(rejected_structure["errors"]) >= {
        "sql_query_invalid",
        "fields_invalid",
        "filter_invalid",
        "comparison_invalid",
    }
    assert exp.execute_checked_sql(relation, altered, documents, rejected_structure) == {
        "status": "rejected",
        "reason": "source_structure_invalid",
    }

    bad_provenance = deepcopy(relation)
    bad_provenance["bundle"]["relations"][0]["provenance"][0]["document"] = "source"
    provenance_check = exp.check_source_structure(bad_provenance, query)
    assert provenance_check["source_provenance_valid"] is False
    assert provenance_check["structure_valid"] is False


def test_scenario_verify_7154_metrics_pair_rows_and_bootstrap() -> None:
    """SCENARIO-VERIFY-7154-METRICS keeps four complete paired arms."""

    final, _runtime, _fixture = _complete_artifact()
    rows = final["rows"]
    assert exp.arm_accounting_errors(rows) == []
    assert len(rows) == 96
    assert {row["arm"] for row in rows} == set(exp.ARM_IDS)
    assert {row["row_count"] for row in final["arm_metric_rows"]} == {24}
    assert len(final["source_family_metric_rows"]) == 16
    assert {row["row_count"] for row in final["source_family_metric_rows"]} == {6}
    assert len(final["paired_comparison_rows"]) == 3
    assert len(final["bootstrap_rows"]) == 15
    assert all(row["n_boot"] == 64 for row in final["bootstrap_rows"])
    assert len(final["harmful_flip_rows"]) == 72
    assert len(final["abstention_rows"]) == 96
    assert len(final["latency_rows"]) == 4
    assert len(final["token_rows"]) == 4
    assert final["qwen_dual_side_pilot_complete_score"] == 1
    assert final["verdict_class"] == "null"

    missing = rows[:-1]
    errors = exp.arm_accounting_errors(missing)
    assert any(error.startswith("arm_row_count:") for error in errors)
    assert any(error.startswith("fixture_arm_set:") for error in errors)


def test_scenario_verify_7154_verdict_disqualifies_uninformative_runs() -> None:
    """SCENARIO-VERIFY-7154-VERDICT separates completion from value."""

    final, _runtime, _fixture = _complete_artifact()
    rows = deepcopy(final["rows"])
    for row in rows:
        row["prediction"] = row["truth_label"]
        row["correct"] = True
        row["abstained"] = False
    score, verdict_class, verdict, reasons = exp.select_verdict(
        rows,
        schedule_identity_score=1,
        label_exposure_count=0,
        bootstrap_rows=exp.build_bootstrap_rows(rows, seed=exp.RANDOM_SEED, n_boot=32),
    )
    assert score == 1
    assert verdict_class == "disqualified"
    assert verdict.startswith("disqualified_")
    assert set(reasons) >= {"control_headroom_absent", "all_arms_identical"}

    score, verdict_class, _verdict, reasons = exp.select_verdict(
        rows[:-1],
        schedule_identity_score=0,
        label_exposure_count=1,
        bootstrap_rows=[],
    )
    assert score == 0
    assert verdict_class == "disqualified"
    assert set(reasons) >= {"schedule_identity_failed", "label_leak", "rows_missing"}


def test_scenario_verify_7154_artifact_rejects_row_and_hash_drift() -> None:
    """SCENARIO-VERIFY-7154-ARTIFACT cold-checks row completeness and hashes."""

    final, runtime, fixture = _complete_artifact()
    assert exp.validate_artifact(final, runtime_artifact=runtime, fixture_artifact=fixture) == []

    changed = deepcopy(final)
    changed["raw_output_rows"][0]["raw_output"] = "changed"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "raw_output_hash_mismatch" in exp.validate_artifact(
        changed, runtime_artifact=runtime, fixture_artifact=fixture
    )

    changed = deepcopy(final)
    changed["rows"].pop()
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert any(
        error.startswith("arm_row_count:")
        for error in exp.validate_artifact(
            changed, runtime_artifact=runtime, fixture_artifact=fixture
        )
    )

    changed = deepcopy(final)
    changed["qwen_dual_side_pilot_complete_score"] = 0
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, runtime_artifact=runtime, fixture_artifact=fixture
    )


def test_req_verify_7154_defensive_parsers_seals_and_accounting(tmp_path: Path) -> None:
    """REQ-VERIFY-7154 preserves malformed inputs instead of repairing them."""

    runtime, fixture = _upstreams()
    schedule, prompt_rows, raw_rows = _model_evidence(runtime, fixture)
    assert exp._load_value(3) is None
    missing = tmp_path / "missing.json"
    assert exp._load_value(missing) is None
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    assert exp._load_value(invalid) is None
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp._load_value(array) is None
    assert "flagged_adversarial" not in exp._runtime_without_annotations(runtime)

    broken = deepcopy(schedule)
    broken.pop()
    broken[0]["call_id"] = "wrong"
    broken[1]["fixture_id"] = "wrong"
    broken[2]["arm"] = "wrong"
    broken[3]["output_token_limit"] = 1
    broken[4]["prompt"] = "changed"
    broken_runtime = deepcopy(runtime)
    broken_runtime["frozen_schedule_hash"] = "sha256:bad"
    assert set(exp.schedule_identity_errors(broken, broken_runtime)) >= {
        "schedule_call_count_mismatch",
        "schedule_call_order_mismatch",
        "schedule_fixture_order_mismatch",
        "schedule_arm_order_mismatch",
        "schedule_output_limit_mismatch",
        "schedule_prompt_content_mismatch",
        "runtime_schedule_hash_mismatch",
    }

    broken_prompts = deepcopy(prompt_rows)
    broken_raw = deepcopy(raw_rows)
    broken_prompts.pop()
    broken_raw.pop()
    broken_prompts[0]["template_prompt_sha256"] = "sha256:bad"
    broken_prompts[1]["prompt"] = "changed"
    broken_raw[0]["raw_output"] = "changed"
    broken_raw[1]["terminal_state"] = "unknown"
    seal = exp.model_evidence_seal(broken_prompts, broken_raw, schedule)
    assert set(seal["errors"]) >= {
        "prompt_receipt_identity_mismatch",
        "raw_receipt_identity_mismatch",
        "template_prompt_hash_mismatch",
        "executed_prompt_hash_mismatch",
        "raw_output_hash_mismatch",
        "raw_terminal_state_invalid",
    }

    valid_seal = exp.model_evidence_seal(prompt_rows, raw_rows, schedule)
    invalid_labels = deepcopy(fixture)
    selected = next(
        row
        for row in invalid_labels["sealed_scorer_rows"]
        if row["fixture_id"] == exp.FROZEN_FIXTURE_IDS[0]
    )
    selected["response_label"] = "unknown"
    with pytest.raises(ValueError, match="exact label authority invalid"):
        exp.open_exact_labels(invalid_labels, valid_seal)
    missing_labels = deepcopy(fixture)
    missing_labels["sealed_scorer_rows"] = missing_labels["sealed_scorer_rows"][1:]
    with pytest.raises(ValueError, match="IDs differ"):
        exp.open_exact_labels(missing_labels, valid_seal)

    malformed_relation = exp.parse_relation_output("not json", {"source": "a", "response": "b"})
    malformed_sql = exp.parse_sql_output("not json")
    structure = exp.check_source_structure(malformed_relation, malformed_sql)
    assert set(structure["errors"]) >= {
        "relation_schema_invalid",
        "source_provenance_invalid",
        "sql_query_invalid",
        "table_invalid",
        "fields_invalid",
        "filter_invalid",
        "comparison_invalid",
    }
    assert exp.reduce_sql_prediction({"status": "rejected"})["prediction"] == "unknown"
    assert exp._decision_or_abstention({"status": "rejected"})["prediction"] == "unknown"
    with pytest.raises(ValueError, match="changed after sealing"):
        exp.score_frozen_outputs(
            runtime, fixture, prompt_rows, raw_rows, {**valid_seal, "sealed": False}
        )

    rejected_raw = deepcopy(raw_rows)
    for row in rejected_raw:
        if row["arm"] == "relational_sql" and row["pass_index"] == 1:
            row["raw_output"] = "not json"
            row["raw_output_sha256"] = exp.sha256_text(row["raw_output"])
            break
    for row in rejected_raw:
        if row["arm"] == "dual_side" and row["pass_index"] == 2:
            row["raw_output"] = json.dumps({"query": "SELECT * FROM grounded_relations LIMIT 32"})
            row["raw_output_sha256"] = exp.sha256_text(row["raw_output"])
            break
    rejected_seal = exp.model_evidence_seal(prompt_rows, rejected_raw, schedule)
    _parses, _structures, executions, rows = exp.score_frozen_outputs(
        runtime, fixture, prompt_rows, rejected_raw, rejected_seal
    )
    assert any(row["reason"] == "relation_schema_invalid" for row in executions)
    assert any(row["reason"] == "exact_sql_invalid" for row in executions)
    assert any(row["abstained"] for row in rows)
    assert "fixture_row_order" in exp.arm_accounting_errors(list(reversed(rows)))
    assert exp._quantile([0.5], 0.5) == 0.5
    assert exp.build_bootstrap_rows(rows[:-1], seed=1, n_boot=10) == []
    assert exp.build_bootstrap_rows(rows, seed=1, n_boot=0) == []


def test_req_verify_7154_positive_and_defensive_terminal_paths(tmp_path: Path) -> None:
    """REQ-VERIFY-7154 covers positive selection and every terminal rejection."""

    final, runtime, fixture = _complete_artifact(n_boot=32)
    winning = deepcopy(final["rows"])
    for row in winning:
        if row["arm"] != "dual_side":
            row["prediction"] = "clean" if row["truth_label"] == "hallucinated" else "hallucinated"
            row["correct"] = False
            row["detected"] = False
        else:
            row["prediction"] = row["truth_label"]
            row["correct"] = True
            row["detected"] = row["truth_label"] == "hallucinated"
            row["harmful_flip"] = False
    bootstrap = exp.build_bootstrap_rows(winning, seed=exp.RANDOM_SEED, n_boot=32)
    assert exp.select_verdict(
        winning,
        schedule_identity_score=1,
        label_exposure_count=0,
        bootstrap_rows=bootstrap,
    )[:3] == (1, "positive", "positive_external_label_dual_side_improvement")

    failed_check = exp.gate_row(
        "runtime_gate", 1, 0, False, upstream="runtime", field="ready_score"
    )
    blocked = exp.finalize_artifact(exp.base_artifact(exp.RUN_DATE), [failed_check], duration_s=1)
    assert blocked["verdict_class"] == "blocked"

    malformed_receipts = deepcopy(final)
    malformed_receipts["MODEL_SPECS"] = []
    malformed_receipts["model_identity_rows"] = []
    malformed_receipts["model_load_receipts"] = []
    malformed_receipts["gpu_rows"] = []
    assert set(exp._completed_model_evidence_errors(malformed_receipts)) == {
        "MODEL_SPECS_mismatch",
        "model_identity_mismatch",
        "model_load_receipt_count",
        "gpu_phase_receipts_missing",
    }
    malformed_receipts = deepcopy(final)
    load = malformed_receipts["model_load_receipts"][0]
    load.update(
        {
            "model_id": "wrong",
            "sha256": "sha256:wrong",
            "health": {"ok": False},
            "gpu_offload_confirmed": False,
            "cleanup": {"leak_free": False},
        }
    )
    assert set(exp._completed_model_evidence_errors(malformed_receipts)) >= {
        "model_load_identity_mismatch",
        "model_load_health_failed",
        "model_gpu_offload_unconfirmed",
        "model_cleanup_failed",
    }

    assert exp.validate_artifact([]) == ["artifact_not_object"]
    assert exp.validate_artifact({})[0].startswith("artifact_fields_mismatch")
    contradictory = deepcopy(final)
    contradictory.update(
        {
            "field_principles": {},
            "inference_substrate": "wrong",
            "execution_venue": "remote",
            "random_seed": 0,
            "verifier_is_oracle": True,
            "verdict_class": "invalid",
            "honest_verdict": "wrong",
            "gate_check_summary": {},
            "status": "running",
            "inference_substrate_class": "blocked_no_run",
            "preconditions_checked": [],
        }
    )
    contradictory["reproducibility_checksum"] = exp.artifact_checksum(contradictory)
    assert set(
        exp.validate_artifact(contradictory, runtime_artifact=runtime, fixture_artifact=fixture)
    ) >= {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "execution_venue_mismatch",
        "random_seed_mismatch",
        "verifier_is_oracle_mismatch",
        "verdict_class_invalid",
        "honest_verdict_prefix_mismatch",
        "gate_check_summary_mismatch",
        "completed_status_mismatch",
        "completed_substrate_class_mismatch",
        "preconditions_incomplete",
    }

    malformed_block = deepcopy(blocked)
    malformed_block.update(
        {
            "status": "completed",
            "inference_substrate_class": "model_full_generation",
            "qwen_dual_side_pilot_complete_score": 1,
        }
    )
    malformed_block["preconditions_checked"][0]["upstream"] = None
    malformed_block["gate_check_summary"] = exp._gate_summary(
        malformed_block["preconditions_checked"]
    )
    malformed_block["reproducibility_checksum"] = exp.artifact_checksum(malformed_block)
    assert set(exp.validate_artifact(malformed_block)) >= {
        "blocked_status_mismatch",
        "blocked_substrate_class_mismatch",
        "blocked_completion_score_mismatch",
        "blocked_gate_detail_missing",
    }

    assert "runtime_artifact_missing" in exp.validate_artifact(
        final, runtime_artifact=tmp_path / "missing-runtime.json", fixture_artifact=fixture
    )
    assert "fixture_artifact_missing" in exp.validate_artifact(
        final, runtime_artifact=runtime, fixture_artifact=tmp_path / "missing-fixture.json"
    )

    replay_drift = deepcopy(final)
    replay_drift.update(
        {
            "frozen_fixture_ids": [],
            "frozen_schedule_hash": "bad",
            "schedule_identity_score": 0,
            "label_exposure_count": 1,
            "parse_rows": [],
            "source_structure_rows": [],
            "sql_execution_rows": [],
            "arm_metric_rows": [],
            "source_family_metric_rows": [],
            "paired_comparison_rows": [],
            "bootstrap_rows": deepcopy(final["bootstrap_rows"]),
            "harmful_flip_rows": [],
            "abstention_rows": [],
            "latency_rows": [],
            "token_rows": [],
            "qwen_dual_side_pilot_complete_score": 1,
            "verdict_class": "null",
            "honest_verdict": "null_wrong",
        }
    )
    replay_drift["bootstrap_rows"][0]["observed_delta"] = 999
    replay_drift["reproducibility_checksum"] = exp.artifact_checksum(replay_drift)
    drift_errors = exp.validate_artifact(
        replay_drift, runtime_artifact=runtime, fixture_artifact=fixture
    )
    assert set(drift_errors) >= {
        "frozen_fixture_ids_mismatch",
        "frozen_schedule_hash_mismatch",
        "schedule_identity_score_mismatch",
        "label_exposure_count_mismatch",
        "parse_rows_mismatch",
        "source_structure_rows_mismatch",
        "sql_execution_rows_mismatch",
        "arm_metric_rows_mismatch",
        "source_family_metric_rows_mismatch",
        "paired_comparison_rows_mismatch",
        "bootstrap_rows_mismatch",
        "harmful_flip_rows_mismatch",
        "abstention_rows_mismatch",
        "latency_rows_mismatch",
        "token_rows_mismatch",
        "completion_score_mismatch",
        "terminal_verdict_mismatch",
    }

    broken_fixture = {"sealed_scorer_rows": fixture["sealed_scorer_rows"]}
    assert any(
        error.startswith("scoring_replay_failed:")
        for error in exp.validate_artifact(
            final, runtime_artifact=runtime, fixture_artifact=broken_fixture
        )
    )
