"""Tests for REQ-VERIFY-7139 and SCENARIO-VERIFY-7139-*.

The tests use small private fixtures. They never start a model or write to the
research record.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7139_v627_symbolic_grounding_ab as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verification/spec.md"


def _model_rows(tmp_path: Path) -> list[dict[str, Any]]:
    """Create one small cached-file receipt for each required family."""

    rows = []
    for index, model_id in enumerate(exp.REQUIRED_MODEL_IDS):
        path = tmp_path / f"model-{index}.Q4_K_M.gguf"
        path.write_bytes(model_id.encode("utf-8"))
        rows.append(
            {
                "name": exp.MODEL_NAMES[model_id],
                "hf_id": model_id,
                "model_path": str(path),
                "gpu": index % 2,
                "preferred_quant": "Q4_K_M",
                "resolution_method": "cached_sota_pair",
                "remote_allowed": False,
            }
        )
    return rows


def _model_views() -> list[dict[str, Any]]:
    """Return two label-free source and response rows."""

    return [
        {
            "fixture_id": "unit-001",
            "task_type": "Summary",
            "split": "test",
            "source_text": "Anne arrived in Paris in 1945.",
            "response_text": "Anne arrived in Paris.",
            "source_text_sha256": exp.sha256_text("Anne arrived in Paris in 1945."),
            "response_text_sha256": exp.sha256_text("Anne arrived in Paris."),
            "prompt": "Extract relations.",
        },
        {
            "fixture_id": "unit-002",
            "task_type": "Summary",
            "split": "test",
            "source_text": "Ben lives in Rome.",
            "response_text": "Ben lives in Oslo.",
            "source_text_sha256": exp.sha256_text("Ben lives in Rome."),
            "response_text_sha256": exp.sha256_text("Ben lives in Oslo."),
            "prompt": "Extract relations.",
        },
    ]


def _sealed_rows() -> list[dict[str, Any]]:
    """Return the external labels that open only after raw output freezes."""

    return [
        {"fixture_id": "unit-001", "response_label": "clean", "span_labels": []},
        {
            "fixture_id": "unit-002",
            "response_label": "hallucinated",
            "span_labels": [{"start": 13, "end": 17}],
        },
    ]


def _bundle(unknown: bool = False) -> dict[str, Any]:
    """Build one Exp7138-compatible bundle."""

    source = _model_views()[0]["source_text"]
    text = "Anne"
    return {
        "entities": [{"entity_id": "e001", "entity_type": "person", "canonical_name": "Anne"}],
        "relations": [
            {
                "relation_id": "r001",
                "subject_entity_id": "e001",
                "predicate": "located_in",
                "object": (
                    {
                        "type": "unknown",
                        "value": None,
                        "unit": None,
                        "unknown_reason": "not_stated_in_source",
                    }
                    if unknown
                    else {
                        "type": "string",
                        "value": "Paris",
                        "unit": None,
                        "unknown_reason": None,
                    }
                ),
                "provenance": [
                    {
                        "document": "source",
                        "start": source.index(text),
                        "end": source.index(text) + len(text),
                        "text": text,
                        "sha256": exp.sha256_text(text),
                    }
                ],
            }
        ],
    }


def _arm_rows() -> list[dict[str, Any]]:
    """Build a complete small matrix with one useful SQL catch."""

    predictions = {
        ("unit-001", "direct"): ("clean", 0.1),
        ("unit-001", "self_verification"): ("clean", 0.1),
        ("unit-001", "relational_sql"): ("clean", 0.0),
        ("unit-002", "direct"): ("clean", 0.2),
        ("unit-002", "self_verification"): ("unknown", 0.5),
        ("unit-002", "relational_sql"): ("hallucinated", 1.0),
    }
    rows = []
    labels = {"unit-001": "clean", "unit-002": "hallucinated"}
    families = {"unit-001": "CNN/DM", "unit-002": "MARCO"}
    for model_id in exp.REQUIRED_MODEL_IDS:
        for fixture_id in ("unit-001", "unit-002"):
            for arm in exp.ARM_IDS:
                prediction, score = predictions[(fixture_id, arm)]
                rows.append(
                    {
                        "arm_row_id": f"{model_id}|{fixture_id}|{arm}",
                        "fixture_id": fixture_id,
                        "model_id": model_id,
                        "model_family": exp.MODEL_FAMILIES[model_id],
                        "source_family": families[fixture_id],
                        "arm": arm,
                        "prediction": prediction,
                        "hallucination_score": score,
                        "truth_label": labels[fixture_id],
                        "correct": prediction == labels[fixture_id],
                        "invalid_sql": False,
                        "unknown": prediction == "unknown",
                        "latency_s": 1.0,
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "cost_usd": 0.0,
                    }
                )
    return rows


def _receipt_rows(tmp_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Build complete synthetic call and SQL receipt sets."""

    schedule = exp.build_schedule(_model_views(), _model_rows(tmp_path))
    prompt_rows = [
        {
            "call_id": row["call_id"],
            "prompt": row["prompt"],
            "prompt_sha256": exp.sha256_text(row["prompt"]),
        }
        for row in schedule
    ]
    raw_output_rows = [
        {
            "call_id": row["call_id"],
            "raw_output": "{}",
            "raw_output_sha256": exp.sha256_text("{}"),
        }
        for row in schedule
    ]
    parse_rows = [{"call_id": row["call_id"], "status": "ok"} for row in schedule]
    sql_cells = [
        {
            "cell_id": f"{model}|{fixture}",
            "model_id": model,
            "fixture_id": fixture,
        }
        for model in exp.REQUIRED_MODEL_IDS
        for fixture in ("unit-001", "unit-002")
    ]
    return {
        "prompt_rows": prompt_rows,
        "raw_output_rows": raw_output_rows,
        "parse_rows": parse_rows,
        "relation_rows": [dict(row, status="ok", relation_errors=[]) for row in sql_cells],
        "sql_query_rows": [
            dict(row, status="ok", query=exp.REQUIRED_SQL_QUERY) for row in sql_cells
        ],
        "sql_execution_rows": [dict(row, status="ok", row_count=0) for row in sql_cells],
    }


def test_req_verify_7139_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7139 owns all focused scenarios and artifact fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7139") :]
    for scenario in ("FIRST-WRITE", "MATRIX", "BLINDING", "SQL", "METRICS", "VERDICT", "ARTIFACT"):
        assert f"SCENARIO-VERIFY-7139-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_7139_first_write_and_block_are_schema_complete(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7139-FIRST-WRITE preserves shape before gates."""

    path = tmp_path / "artifact.json"
    running = exp.initialize_artifact(path, exp.RUN_DATE)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(running)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(json.loads(path.read_text()))
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(running["field_principles"])
    assert running["inference_substrate_class"] == "blocked_no_run"

    failed = exp.gate_row("gpu_available", True, False)
    blocked = exp.finish_blocked(running, path, [failed], duration_s=0.25)
    assert exp.validate_artifact(blocked) == []
    assert blocked["gate_check_summary"] == {
        "failed_check": "gpu_available",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_gpu_available"
    assert blocked["prompt_rows"] == blocked["raw_output_rows"] == []


def test_scenario_verify_7139_resolves_exact_cached_q4_roster(tmp_path: Path) -> None:
    """REQ-VERIFY-7139 rejects model substitutions and remote fallback."""

    paths = {row["hf_id"]: row["model_path"] for row in _model_rows(tmp_path)}

    def pair(**_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "hf_id": exp.REQUIRED_MODEL_IDS[0],
                "model_path": paths[exp.REQUIRED_MODEL_IDS[0]],
                "gpu": 0,
            },
            {
                "hf_id": exp.REQUIRED_MODEL_IDS[2],
                "model_path": paths[exp.REQUIRED_MODEL_IDS[2]],
                "gpu": 1,
            },
        ]

    specs = exp.resolve_model_specs(
        pair_provider=pair,
        resolver=lambda model_id, _quant: paths.get(model_id),
    )
    assert [row["hf_id"] for row in specs] == list(exp.REQUIRED_MODEL_IDS)
    assert exp.model_spec_errors(specs) == []
    assert all(row["preferred_quant"] == "Q4_K_M" for row in specs)
    assert all(row["remote_allowed"] is False for row in specs)

    broken = deepcopy(specs)
    broken.reverse()
    broken[0]["model_path"] = ""
    errors = exp.model_spec_errors(broken)
    assert "model_ids_mismatch" in errors
    assert any(error.startswith("model_path_missing:") for error in errors)


def test_scenario_verify_7139_matrix_is_matched_and_label_free(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7139-MATRIX and BLINDING freeze every call."""

    schedule = exp.build_schedule(_model_views(), _model_rows(tmp_path))
    assert len(schedule) == 2 * 3 * 5
    assert exp.schedule_errors(schedule, ["unit-001", "unit-002"]) == []
    counts = {(row["arm"], row["pass_index"]) for row in schedule}
    assert counts == {
        ("direct", 1),
        ("self_verification", 1),
        ("self_verification", 2),
        ("relational_sql", 1),
        ("relational_sql", 2),
    }
    assert {row["output_token_limit"] for row in schedule} == {exp.OUTPUT_TOKEN_LIMIT}
    assert exp.prompt_exposure_errors([row["prompt"] for row in schedule]) == []

    broken = deepcopy(schedule)
    broken.pop()
    broken[0]["output_token_limit"] += 1
    broken[1]["prompt"] += "\nresponse_label=clean"
    errors = exp.schedule_errors(broken, ["unit-001", "unit-002"])
    assert "schedule_identity_mismatch" in errors
    assert any(error.startswith("output_limit_mismatch:") for error in errors)
    assert any(error.startswith("prompt_exposure:") for error in errors)


def test_scenario_verify_7139_parsers_preserve_invalid_and_unknown() -> None:
    """SCENARIO-VERIFY-7139-SQL parses without repair or label access."""

    direct = exp.parse_decision_output(
        'prefix\n```json\n{"status":"unsupported","confidence":0.8}\n```'
    )
    assert direct == {
        "status": "ok",
        "prediction": "hallucinated",
        "hallucination_score": 0.8,
    }
    assert (
        exp.parse_decision_output('{"status":"supported","confidence":0.9}')["prediction"]
        == "clean"
    )
    assert (
        exp.parse_decision_output('{"status":"unknown","confidence":0.9}')["prediction"]
        == "unknown"
    )
    assert exp.parse_decision_output("not json")["status"] == "rejected"
    assert exp.parse_decision_output('{"status":"bad","confidence":2}')["status"] == "rejected"

    documents = {
        "source": _model_views()[0]["source_text"],
        "response": _model_views()[0]["response_text"],
    }
    parsed = exp.parse_relation_output(json.dumps(_bundle()), documents)
    assert parsed["status"] == "ok"
    assert parsed["bundle"] == _bundle()
    assert exp.parse_relation_output("[]", documents)["status"] == "rejected"

    query = exp.REQUIRED_SQL_QUERY
    assert exp.parse_sql_output(json.dumps({"query": query})) == {"status": "ok", "query": query}
    assert (
        exp.parse_sql_output('{"query":"SELECT * FROM grounded_relations LIMIT 32"}')["status"]
        == "rejected"
    )
    assert exp.parse_sql_output("bad")["status"] == "rejected"


def test_scenario_verify_7139_sql_executes_only_exp7138_sandbox() -> None:
    """SCENARIO-VERIFY-7139-SQL never turns invalid or unknown into clean."""

    documents = {
        "source": _model_views()[0]["source_text"],
        "response": _model_views()[0]["response_text"],
    }
    clean_execution = exp.execute_sql_candidate(_bundle(), documents, exp.REQUIRED_SQL_QUERY)
    assert clean_execution["status"] == "ok"
    assert exp.reduce_sql_prediction([], {"status": "ok"}, clean_execution) == {
        "prediction": "clean",
        "hallucination_score": 0.0,
        "invalid_sql": False,
        "reason": "no_unsupported_relation_returned",
    }

    unknown_execution = exp.execute_sql_candidate(_bundle(True), documents, exp.REQUIRED_SQL_QUERY)
    assert unknown_execution["row_count"] == 1
    assert (
        exp.reduce_sql_prediction([], {"status": "ok"}, unknown_execution)["prediction"]
        == "hallucinated"
    )

    for relation_errors, query_parse, execution in (
        (["bad relation"], {"status": "ok"}, clean_execution),
        ([], {"status": "rejected"}, clean_execution),
        ([], {"status": "ok"}, {"status": "rejected", "reason": "unsupported_sql"}),
    ):
        reduced = exp.reduce_sql_prediction(relation_errors, query_parse, execution)
        assert reduced["prediction"] == "unknown"
        assert reduced["invalid_sql"] is True


def test_scenario_verify_7139_metrics_keep_model_and_source_losses_visible() -> None:
    """SCENARIO-VERIFY-7139-METRICS computes every declared projection."""

    rows = _arm_rows()
    metrics = exp.build_metric_rows(rows)
    assert len(metrics) == 3 * (1 + 3 + 2)
    sql_overall = next(
        row for row in metrics if row["scope"] == "overall" and row["arm"] == "relational_sql"
    )
    assert sql_overall["accuracy"] == 1.0
    assert sql_overall["auroc"] == 1.0
    assert sql_overall["false_positive_rate"] == 0.0
    assert sql_overall["hallucination_catch_rate"] == 1.0
    assert exp.binary_metric_row([])["auroc"] is None

    source_rows = exp.build_source_family_rows(metrics)
    model_rows = exp.build_model_family_rows(metrics)
    assert {(row["source_family"], row["arm"]) for row in source_rows} == {
        (family, arm) for family in ("CNN/DM", "MARCO") for arm in exp.ARM_IDS
    }
    assert {(row["model_id"], row["arm"]) for row in model_rows} == {
        (model, arm) for model in exp.REQUIRED_MODEL_IDS for arm in exp.ARM_IDS
    }
    assert exp.build_token_rows(rows)[0]["total_tokens"] == 30
    assert exp.build_latency_rows(rows)[0]["total_latency_s"] == 6.0
    assert exp.build_cost_rows(rows)[0]["cost_usd"] == 0.0


def test_scenario_verify_7139_bootstrap_is_paired_by_frozen_id() -> None:
    """SCENARIO-VERIFY-7139-METRICS resamples whole fixture IDs."""

    first = exp.build_bootstrap_rows(_arm_rows(), seed=7139, n_boot=100)
    second = exp.build_bootstrap_rows(_arm_rows(), seed=7139, n_boot=100)
    assert first == second
    assert len(first) == 2 * (1 + 3 + 2)
    assert all(row["resampling_unit"] == "fixture_id" for row in first)
    assert all(row["paired"] is True for row in first)
    deltas = {
        (row["scope"], row["scope_value"], row["comparator"]): row["sql_minus_control_accuracy"]
        for row in first
    }
    assert deltas[("overall", "all", "direct")] == 0.5
    assert deltas[("source_family", "CNN/DM", "direct")] == 0.0
    assert deltas[("source_family", "MARCO", "direct")] == 1.0
    useful = exp.build_useful_detection_rows(_arm_rows())
    assert all(row["sql_unique_hallucination_catches"] == 1 for row in useful)


def test_scenario_verify_7139_verdict_separates_completion_from_uplift() -> None:
    """SCENARIO-VERIFY-7139-VERDICT needs supported external-label uplift."""

    complete, verdict_class, verdict = exp.select_verdict(
        matrix_complete=True,
        bootstrap_rows=[
            {"comparator": arm, "ci95": [0.1, 0.5], "scope": "overall"}
            for arm in ("direct", "self_verification")
        ],
        family_loss=False,
    )
    assert (complete, verdict_class) == (1, "positive")
    assert verdict.startswith("positive_")

    complete, verdict_class, verdict = exp.select_verdict(
        matrix_complete=True,
        bootstrap_rows=[
            {"comparator": "direct", "ci95": [-0.1, 0.5], "scope": "overall"},
            {"comparator": "self_verification", "ci95": [0.1, 0.5], "scope": "overall"},
        ],
        family_loss=False,
    )
    assert (complete, verdict_class) == (1, "null")
    assert verdict.startswith("null_")
    assert exp.select_verdict(matrix_complete=False, bootstrap_rows=[], family_loss=True) == (
        0,
        "partial",
        "partial_incomplete_symbolic_grounding_matrix",
    )


def test_scenario_verify_7139_artifact_validation_recomputes_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7139-ARTIFACT rejects row and checksum drift."""

    rows = _arm_rows()
    receipts = _receipt_rows(tmp_path)
    artifact = exp.finalize_artifact(
        exp.base_artifact(exp.RUN_DATE),
        arm_rows=rows,
        prompt_rows=receipts["prompt_rows"],
        raw_output_rows=receipts["raw_output_rows"],
        parse_rows=receipts["parse_rows"],
        relation_rows=receipts["relation_rows"],
        sql_query_rows=receipts["sql_query_rows"],
        sql_execution_rows=receipts["sql_execution_rows"],
        model_specs=_model_rows(tmp_path),
        model_identity_rows=[
            {"model_id": model, "passed": True} for model in exp.REQUIRED_MODEL_IDS
        ],
        model_load_receipts=[
            {"model_id": model, "passed": True} for model in exp.REQUIRED_MODEL_IDS
        ],
        preconditions_checked=[exp.gate_row("all", True, True)],
        source_artifact_hashes={"fixture": "sha256:" + "1" * 64},
        duration_s=61.0,
        expected_fixture_ids=("unit-001", "unit-002"),
        n_boot=40,
    )
    path = tmp_path / "complete.json"
    exp.write_artifact(path, artifact)
    assert exp.validate_artifact(path) == []
    assert artifact["symbolic_grounding_complete_score"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["per_game_results"] == []

    changed = deepcopy(artifact)
    changed["arm_rows"][0]["prediction"] = "hallucinated"
    assert "metric_rows mismatch" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["raw_output_rows"][0]["raw_output_sha256"] = "changed"
    assert any("raw_output_hash" in error for error in exp.validate_artifact(changed))

    changed = deepcopy(artifact)
    changed.pop("cost_rows")
    assert any(
        error.startswith("artifact fields mismatch") for error in exp.validate_artifact(changed)
    )


@pytest.mark.parametrize("value", [None, [], "bad"])
def test_req_verify_7139_validator_rejects_non_artifacts(value: object) -> None:
    """REQ-VERIFY-7139 rejects absent, unreadable, and non-object artifacts."""

    assert exp.validate_artifact(value) == ["artifact_not_object"]


def test_scenario_verify_7139_adversarial_branches_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7139-ARTIFACT exercises malformed receipt branches."""

    specs = _model_rows(tmp_path)
    broken_specs = deepcopy(specs)
    broken_specs[0].update(
        {
            "model_path": str(tmp_path / "mmproj.Q5.gguf"),
            "preferred_quant": "Q5_K_M",
            "resolution_method": "remote",
            "remote_allowed": True,
        }
    )
    spec_errors = exp.model_spec_errors(broken_specs)
    assert any(error.startswith("model_path_not_primary_gguf") for error in spec_errors)
    assert any(error.startswith("model_quantization_mismatch") for error in spec_errors)
    assert any(error.startswith("model_resolution_mismatch") for error in spec_errors)
    assert any(error.startswith("remote_fallback_enabled") for error in spec_errors)

    schedule = exp.build_schedule(_model_views(), specs)
    schedule[0]["source_text_sha256"] = "changed"
    schedule[1]["response_text_sha256"] = "changed"
    schedule_errors = exp.schedule_errors(schedule, ["unit-001", "unit-002"])
    assert any(error.startswith("source_hash_mismatch") for error in schedule_errors)
    assert any(error.startswith("response_hash_mismatch") for error in schedule_errors)
    assert (
        exp.parse_relation_output("{bad", {"source": "x", "response": "y"})["status"] == "rejected"
    )
    assert exp.parse_relation_output("{}", {"source": "x", "response": "y"})["status"] == "rejected"
    assert exp.execute_sql_candidate({}, {}, exp.REQUIRED_SQL_QUERY)["status"] == "rejected"
    assert exp.build_token_rows([]) == []
    assert exp.build_latency_rows([]) == []
    assert exp.build_useful_detection_rows([]) == []
    assert exp._quantile([1.0], 0.5) == 1.0
    assert exp._call_seed("same") == exp._call_seed("same")

    harmful_rows = _arm_rows()
    for row in harmful_rows:
        if row["fixture_id"] == "unit-001" and row["arm"] == "relational_sql":
            row["prediction"] = "hallucinated"
            row["correct"] = False
    assert all(
        row["sql_unique_false_positives"] == 1
        for row in exp.build_useful_detection_rows(harmful_rows)
    )

    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert exp._load_artifact_value(missing) is None
    assert exp._load_artifact_value(invalid) is None
    assert exp._load_artifact_value(non_object) is None
    assert exp._load_artifact_value(json.dumps({"one": 1})) == {"one": 1}

    failed = exp.gate_row("gpu", True, False)
    blocked = exp.finish_blocked(
        exp.base_artifact(exp.RUN_DATE), tmp_path / "blocked.json", [failed], duration_s=0.1
    )
    blocked.update(
        {
            "gate_check_summary": {},
            "inference_substrate_class": "model_full_generation",
            "verdict_class": "null",
            "honest_verdict": "null_mutated_block",
            "prompt_rows": [{"prompt": "x"}],
            "symbolic_grounding_complete_score": 1,
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    for expected in (
        "blocked gate_check_summary mismatch",
        "blocked inference_substrate_class mismatch",
        "blocked verdict_class mismatch",
        "blocked arm generation rows present",
        "blocked completion score mismatch",
    ):
        assert expected in blocked_errors

    missing_check = exp.base_artifact(exp.RUN_DATE)
    missing_check.update(
        {"verdict_class": "blocked", "honest_verdict": "blocked_missing_precondition"}
    )
    missing_check["reproducibility_checksum"] = exp.artifact_checksum(missing_check)
    assert "blocked precondition missing" in exp.validate_artifact(missing_check)

    rows = _arm_rows()
    receipts = _receipt_rows(tmp_path)
    complete = exp.finalize_artifact(
        exp.base_artifact(exp.RUN_DATE),
        arm_rows=rows,
        prompt_rows=receipts["prompt_rows"],
        raw_output_rows=receipts["raw_output_rows"],
        parse_rows=receipts["parse_rows"],
        relation_rows=receipts["relation_rows"],
        sql_query_rows=receipts["sql_query_rows"],
        sql_execution_rows=receipts["sql_execution_rows"],
        model_specs=specs,
        model_identity_rows=[
            {"model_id": model, "passed": True} for model in exp.REQUIRED_MODEL_IDS
        ],
        model_load_receipts=[
            {"model_id": model, "passed": True} for model in exp.REQUIRED_MODEL_IDS
        ],
        preconditions_checked=[exp.gate_row("all", True, True)],
        source_artifact_hashes={"fixture": "sha256:" + "2" * 64},
        duration_s=61.0,
        expected_fixture_ids=("unit-001", "unit-002"),
        n_boot=20,
    )
    complete.update(
        {
            "field_principles": {},
            "run_date": "19000101",
            "inference_substrate": "wrong",
            "execution_venue": "remote",
            "random_seed": 0,
            "verifier_is_oracle": True,
            "per_game_results": ["game"],
            "duration_s": -1,
            "verdict_class": "invalid",
            "honest_verdict": "wrong_mutation",
            "inference_substrate_class": "wrong",
            "rows": [],
            "metric_rows": [],
            "source_family_rows": [],
            "model_family_rows": [],
            "token_rows": [],
            "latency_rows": [],
            "cost_rows": [],
            "useful_detection_rows": [],
            "invalid_sql_rate": 1.0,
            "unknown_rate": 1.0,
            "bootstrap_rows": [{"n_boot": 0}],
            "symbolic_grounding_complete_score": 0,
            "gate_check_summary": {},
            "exact_label_blinding_passed": True,
        }
    )
    complete["prompt_rows"][0]["prompt"] += " response_label=clean"
    complete["raw_output_rows"][0]["raw_output_sha256"] = "changed"
    complete["reproducibility_checksum"] = exp.artifact_checksum(complete)
    errors = exp.validate_artifact(complete)
    for prefix in (
        "field_principles mismatch",
        "run_date mismatch",
        "inference_substrate mismatch",
        "execution_venue mismatch",
        "random_seed mismatch",
        "verifier_is_oracle mismatch",
        "per_game_results mismatch",
        "duration_s invalid",
        "verdict_class invalid",
        "honest_verdict mismatch",
        "inference_substrate_class mismatch",
        "rows mismatch",
        "prompt_hash mismatch",
        "raw_output_hash mismatch",
        "exact_label_blinding mismatch",
        "metric_rows mismatch",
        "source_family_rows mismatch",
        "model_family_rows mismatch",
        "token_rows mismatch",
        "latency_rows mismatch",
        "cost_rows mismatch",
        "useful_detection_rows mismatch",
        "invalid_sql_rate mismatch",
        "unknown_rate mismatch",
        "bootstrap_rows mismatch",
        "symbolic_grounding_complete_score mismatch",
        "terminal verdict mismatch",
        "gate_check_summary mismatch",
        "exact_label_blinding_passed mismatch",
    ):
        assert any(error.startswith(prefix) for error in errors)
