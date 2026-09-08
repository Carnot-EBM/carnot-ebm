"""Tests for REQ-HARNESS-7138 and SCENARIO-HARNESS-7138-*.

The tests use the checked-in RAGTruth cache as read-only input. Every writer
uses ``tmp_path`` so the test run cannot change the research record.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_7138_v627_relational_fixture as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"
SOURCE_INFO = REPO / "data/ragtruth/source_info.jsonl"
RESPONSES = REPO / "data/ragtruth/response.jsonl"
LINEAGE = REPO / "data/real_factual_corpus_ragtruth.jsonl"


def _artifact(tmp_path: Path) -> dict[str, object]:
    """Build a complete fixture at a private output path."""

    return exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=tmp_path / "artifact.json",
        duration_s=0.25,
    )


def _relation_bundle() -> tuple[dict[str, object], dict[str, str]]:
    """Return one small valid relation bundle and its exact documents."""

    source = "Anne arrived in Paris in 1945."
    response = "Anne arrived in Paris."
    documents = {"source": source, "response": response}
    span_text = "Anne"
    bundle = {
        "entities": [
            {
                "entity_id": "e001",
                "entity_type": "person",
                "canonical_name": "Anne",
            }
        ],
        "relations": [
            {
                "relation_id": "r001",
                "subject_entity_id": "e001",
                "predicate": "located_in",
                "object": {
                    "type": "string",
                    "value": "Paris",
                    "unit": None,
                    "unknown_reason": None,
                },
                "provenance": [
                    {
                        "document": "source",
                        "start": 0,
                        "end": len(span_text),
                        "text": span_text,
                        "sha256": exp.sha256_text(span_text),
                    }
                ],
            }
        ],
    }
    return bundle, documents


def test_req_harness_7138_spec_precedes_implementation() -> None:
    """REQ-HARNESS-7138 owns the focused scenarios and artifact fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-HARNESS-7138") :]
    for scenario in (
        "SELECTION",
        "BLINDING",
        "RELATIONS",
        "SQL",
        "REPLAY",
        "BOOTSTRAP",
        "ARTIFACT",
    ):
        assert f"SCENARIO-HARNESS-7138-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_harness_7138_selection_is_group_first_and_balanced() -> None:
    """SCENARIO-HARNESS-7138-SELECTION fixes groups before class quotas."""

    sources = exp.load_source_records(SOURCE_INFO)
    membership = exp.load_response_membership(RESPONSES)
    selected_groups = exp.select_source_groups(sources, membership)
    assert list(selected_groups) == list(exp.SOURCE_FAMILIES)
    assert all(len(ids) == exp.SOURCE_GROUPS_PER_FAMILY for ids in selected_groups.values())

    responses = exp.load_response_records(RESPONSES)
    fixture = exp.build_fixture_views(sources, responses, selected_groups)
    flipped = [dict(row, labels=[] if row["labels"] else [{"hidden": True}]) for row in responses]
    assert exp.select_source_groups(sources, membership) == selected_groups
    assert (
        exp.select_source_groups(sources, exp.response_membership_rows(flipped)) == selected_groups
    )
    assert len(fixture["fixture_rows"]) == 72
    assert fixture["split_rows"] == [{"split": "test", "row_count": 72}]
    assert {row["source_family"] for row in fixture["source_family_rows"]} == set(
        exp.SOURCE_FAMILIES
    )
    assert all(row["source_group_count"] == 12 for row in fixture["source_family_rows"])
    assert all(row["row_count"] == 18 for row in fixture["source_family_rows"])
    assert {
        (row["source_family"], row["response_label"], row["row_count"])
        for row in fixture["class_balance_rows"]
    } == {
        (family, response_label, 9)
        for family in exp.SOURCE_FAMILIES
        for response_label in exp.RESPONSE_LABELS
    }
    assert [row["fixture_id"] for row in fixture["fixture_rows"]] == [
        f"unit-{index:03d}" for index in range(1, 73)
    ]


def test_scenario_harness_7138_model_view_is_label_free() -> None:
    """SCENARIO-HARNESS-7138-BLINDING rejects added scorer state."""

    fixture = exp.build_fixture_views(
        exp.load_source_records(SOURCE_INFO),
        exp.load_response_records(RESPONSES),
        exp.select_source_groups(
            exp.load_source_records(SOURCE_INFO),
            exp.load_response_membership(RESPONSES),
        ),
    )
    model_rows = fixture["model_view_rows"]
    assert exp.model_view_exposure_errors(model_rows) == []
    assert exp.count_label_exposures(model_rows) == 0
    assert all(set(row) == set(exp.MODEL_VIEW_FIELDS) for row in model_rows)
    assert all(row["prompt"] == exp.render_model_prompt(row) for row in model_rows)

    exposed = deepcopy(model_rows)
    exposed[0]["response_label"] = "hallucinated"
    assert "model_view_fields:unit-001" in exp.model_view_exposure_errors(exposed)

    changed_prompt = deepcopy(model_rows)
    changed_prompt[0]["prompt"] += "\nScorer label: clean"
    assert "model_prompt_mismatch:unit-001" in exp.model_view_exposure_errors(changed_prompt)

    changed_hashes = deepcopy(model_rows)
    changed_hashes[0]["source_text_sha256"] = "changed"
    changed_hashes[0]["response_text_sha256"] = "changed"
    hash_errors = exp.model_view_exposure_errors(changed_hashes)
    assert "model_source_hash:unit-001" in hash_errors
    assert "model_response_hash:unit-001" in hash_errors


def test_scenario_harness_7138_relation_schema_checks_spans_and_unknowns() -> None:
    """SCENARIO-HARNESS-7138-RELATIONS enforces closed typed payloads."""

    bundle, documents = _relation_bundle()
    assert exp.validate_relation_bundle(bundle, documents) == []

    unknown = deepcopy(bundle)
    unknown["relations"][0]["object"] = {
        "type": "unknown",
        "value": None,
        "unit": None,
        "unknown_reason": "not_stated_in_source",
    }
    assert exp.validate_relation_bundle(unknown, documents) == []

    bad_unknown = deepcopy(unknown)
    bad_unknown["relations"][0]["object"]["value"] = "guessed"
    assert "relation_object_invalid:r001" in exp.validate_relation_bundle(bad_unknown, documents)

    bad_span = deepcopy(bundle)
    bad_span["relations"][0]["provenance"][0]["text"] = "Else"
    assert "provenance_text_mismatch:r001:0" in exp.validate_relation_bundle(bad_span, documents)

    bad_type = deepcopy(bundle)
    bad_type["entities"][0]["entity_type"] = "invented_type"
    assert "entity_invalid:e001" in exp.validate_relation_bundle(bad_type, documents)


def test_scenario_harness_7138_sql_accepts_only_bounded_selects() -> None:
    """SCENARIO-HARNESS-7138-SQL permits one closed read-only grammar."""

    bundle, documents = _relation_bundle()
    connection = exp.create_relation_database(bundle, documents)
    accepted = exp.execute_bounded_select(
        connection,
        "SELECT subject_name, predicate, object_value FROM grounded_relations "
        "WHERE predicate = 'located_in' ORDER BY relation_id ASC LIMIT 4",
    )
    assert accepted == {
        "status": "ok",
        "columns": ["subject_name", "predicate", "object_value"],
        "rows": [["Anne", "located_in", "Paris"]],
        "row_count": 1,
    }
    before = connection.total_changes
    rejected = {
        query: exp.execute_bounded_select(connection, query)["reason"]
        for query in (
            "UPDATE grounded_relations SET object_value = 'Rome'",
            "PRAGMA query_only = OFF",
            "ATTACH DATABASE '/tmp/x' AS x",
            "SELECT load_extension('x') FROM grounded_relations LIMIT 1",
            "SELECT * FROM grounded_relations; SELECT 1",
            "SELECT * FROM grounded_relations UNION SELECT * FROM grounded_relations LIMIT 2",
            "SELECT * FROM grounded_relations JOIN grounded_relations g LIMIT 2",
            "SELECT missing FROM grounded_relations LIMIT 1",
            "SELECT * FROM grounded_relations LIMIT 33",
        )
    }
    assert set(rejected.values()) == {"unsupported_sql"}
    assert connection.total_changes == before
    assert (
        exp.execute_bounded_select(
            connection,
            "SELECT * FROM grounded_relations LIMIT 1",
            max_steps=0,
        )["reason"]
        == "step_budget_exhausted"
    )
    assert (
        exp.execute_bounded_select(
            connection,
            "SELECT * FROM grounded_relations LIMIT 1",
            timeout_s=0.0,
        )["reason"]
        == "time_budget_exhausted"
    )
    connection.close()


def test_scenario_harness_7138_independent_replay_detects_drift(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7138-REPLAY checks bytes, order, labels, and hashes."""

    artifact = _artifact(tmp_path)
    receipts, errors = exp.independent_replay(artifact, SOURCE_INFO, RESPONSES)
    assert errors == []
    assert receipts == artifact["independent_loader_rows"]
    assert len(receipts) == 72
    assert all(row["exact_match"] is True for row in receipts)

    reordered = deepcopy(artifact)
    reordered["fixture_rows"][0], reordered["fixture_rows"][1] = (
        reordered["fixture_rows"][1],
        reordered["fixture_rows"][0],
    )
    assert (
        "fixture_row_order_mismatch" in exp.independent_replay(reordered, SOURCE_INFO, RESPONSES)[1]
    )

    changed_label = deepcopy(artifact)
    changed_label["sealed_scorer_rows"][0]["response_label"] = (
        "clean"
        if changed_label["sealed_scorer_rows"][0]["response_label"] == "hallucinated"
        else "hallucinated"
    )
    assert any(
        error.startswith("response_label_mismatch:")
        for error in exp.independent_replay(changed_label, SOURCE_INFO, RESPONSES)[1]
    )

    changed_source = tmp_path / "changed-source.jsonl"
    changed_source.write_bytes(SOURCE_INFO.read_bytes() + b"\n")
    assert (
        "source_info_hash_mismatch"
        in exp.independent_replay(artifact, changed_source, RESPONSES)[1]
    )


def test_scenario_harness_7138_mutation_receipts_are_executed(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7138-REPLAY records four effective mutations."""

    artifact = _artifact(tmp_path)
    assert [row["mutation"] for row in artifact["mutation_rows"]] == [
        "label_exposure",
        "unsupported_sql",
        "changed_source_bytes",
        "reordered_rows",
    ]
    assert all(row["detected"] is True and row["detection"] for row in artifact["mutation_rows"])


def test_scenario_harness_7138_bootstrap_precedes_load_and_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HARNESS-7138-BOOTSTRAP writes first and blocks exactly."""

    result = tmp_path / "missing.json"
    missing = tmp_path / "no-source.jsonl"
    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=result,
        corpus_paths={
            "source_info": missing,
            "responses": RESPONSES,
            "lineage": LINEAGE,
        },
        duration_s=0.1,
    )
    assert result.is_file()
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"] == {
        "failed_check": "source_info_path",
        "expected_value": "readable_file",
        "observed_value": "missing_or_unreadable",
        "passed": False,
    }

    observed: list[dict[str, object]] = []
    original_loader = exp.load_source_records

    def observing_loader(path: Path) -> list[dict[str, object]]:
        running = json.loads(result.read_text(encoding="utf-8"))
        observed.append(running)
        assert running["status"] == "running"
        assert set(running) == set(exp.REQUIRED_ARTIFACT_FIELDS)
        return original_loader(path)

    monkeypatch.setattr(exp, "load_source_records", observing_loader)
    complete = exp.build_artifact(REPO, exp.RUN_DATE, result_path=result, duration_s=0.2)
    assert observed and complete["status"] == "complete"


def test_scenario_harness_7138_hash_and_schema_failures_are_exact(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7138-BOOTSTRAP separates hash and schema failures."""

    changed = tmp_path / "source.jsonl"
    changed.write_bytes(SOURCE_INFO.read_bytes() + b"\n")
    hash_block = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=tmp_path / "hash.json",
        corpus_paths={"source_info": changed, "responses": RESPONSES, "lineage": LINEAGE},
        duration_s=0.1,
    )
    assert hash_block["gate_check_summary"]["failed_check"] == "source_info_hash"
    assert (
        hash_block["gate_check_summary"]["expected_value"]
        == exp.PINNED_CORPUS_HASHES["source_info"]
    )
    assert hash_block["gate_check_summary"]["observed_value"] == exp.sha256_file(changed)

    malformed = tmp_path / "responses.jsonl"
    malformed.write_text("{}\n", encoding="utf-8")
    custom_hashes = dict(exp.PINNED_CORPUS_HASHES)
    custom_hashes["responses"] = exp.sha256_file(malformed)
    schema_block = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=tmp_path / "schema.json",
        corpus_paths={"source_info": SOURCE_INFO, "responses": malformed, "lineage": LINEAGE},
        expected_hashes=custom_hashes,
        duration_s=0.1,
    )
    assert schema_block["gate_check_summary"]["failed_check"] == "response_schema"
    assert schema_block["gate_check_summary"]["expected_value"] == list(
        exp.RESPONSE_REQUIRED_FIELDS
    )
    assert schema_block["gate_check_summary"]["observed_value"] == {
        "line": 1,
        "missing_fields": list(exp.RESPONSE_REQUIRED_FIELDS),
    }


def test_scenario_harness_7138_artifact_validates_without_value_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-HARNESS-7138-ARTIFACT gates only fixture readiness."""

    artifact = _artifact(tmp_path)
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"
    assert artifact["source_grounding_fixture_ready_score"] == 1
    assert artifact["fixture_row_count"] == 72
    assert artifact["label_exposure_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert "verifier_value" not in artifact
    assert "accuracy" not in artifact
    assert "auroc" not in artifact
    assert artifact["rows"] == artifact["fixture_rows"]
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact, SOURCE_INFO, RESPONSES) == []

    for field, value in (
        ("field_principles", {}),
        ("inference_substrate", "live model"),
        ("inference_substrate_class", "model_full_generation"),
        ("execution_venue", "gpu"),
        ("duration_s", -1.0),
        ("label_exposure_count", 1),
        ("fixture_row_count", 71),
        ("source_grounding_fixture_ready_score", 0),
        ("verifier_is_oracle", True),
        ("verdict_class", "circular_positive"),
        ("honest_verdict", "complete_circular_fixture"),
        ("reproducibility_checksum", "sha256:bad"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert exp.validate_artifact(changed, SOURCE_INFO, RESPONSES), field

    output = tmp_path / "cli.json"
    assert exp.main(["--date", exp.RUN_DATE, "--result-path", str(output)]) == 0
    assert '"source_grounding_fixture_ready_score": 1' in capsys.readouterr().out
    assert exp.main(["--validate", str(output)]) == 0
    assert exp.main(["--date", "bad", "--result-path", str(output)]) == 2


def test_hash_helpers_are_byte_exact() -> None:
    """REQ-HARNESS-7138 hashes exact UTF-8 bytes and canonical JSON."""

    assert exp.sha256_text("x") == "sha256:" + hashlib.sha256(b"x").hexdigest()
    assert exp.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_scenario_harness_7138_defensive_loader_and_quota_paths(tmp_path: Path) -> None:
    """REQ-HARNESS-7138 reports malformed rows and unavailable fixed quotas."""

    rows = tmp_path / "rows.jsonl"
    rows.write_text('\n["not", "an", "object"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_row_not_object:2"):
        exp.load_source_records(rows)
    with pytest.raises(ValueError, match="source_group_quota_unavailable:CNN/DM"):
        exp.select_source_groups([], [])

    sources = exp.load_source_records(SOURCE_INFO)
    groups = exp.select_source_groups(sources, exp.load_response_membership(RESPONSES))
    with pytest.raises(ValueError, match="class_quota_unavailable:CNN/DM:clean"):
        exp.build_fixture_views(sources, [], groups)


def test_scenario_harness_7138_relation_rejection_matrix() -> None:
    """SCENARIO-HARNESS-7138-RELATIONS covers each closed value and span branch."""

    bundle, documents = _relation_bundle()
    entity_ids = {"e001"}
    assert exp._valid_relation_object(None, entity_ids) is False
    assert (
        exp._valid_relation_object(
            {"type": "other", "value": "x", "unit": None, "unknown_reason": None},
            entity_ids,
        )
        is False
    )
    assert (
        exp._valid_relation_object(
            {"type": "string", "value": "x", "unit": None, "unknown_reason": "guess"},
            entity_ids,
        )
        is False
    )
    for value in (
        {"type": "entity", "value": "e001", "unit": None, "unknown_reason": None},
        {"type": "date", "value": "1945", "unit": None, "unknown_reason": None},
        {"type": "number", "value": 3, "unit": "years", "unknown_reason": None},
        {"type": "boolean", "value": True, "unit": None, "unknown_reason": None},
    ):
        assert exp._valid_relation_object(value, entity_ids) is True

    assert exp.validate_relation_bundle({"entities": []}, documents) == ["relation_bundle_fields"]
    assert exp.validate_relation_bundle({"entities": "bad", "relations": []}, documents) == [
        "relation_bundle_collections"
    ]

    invalid_span = deepcopy(bundle)
    invalid_span["relations"][0]["provenance"][0]["start"] = "0"
    assert "provenance_invalid:r001:0" in exp.validate_relation_bundle(invalid_span, documents)
    outside = deepcopy(bundle)
    outside["relations"][0]["provenance"][0]["end"] = 999
    assert "provenance_bounds:r001:0" in exp.validate_relation_bundle(outside, documents)
    bad_relation = deepcopy(bundle)
    bad_relation["relations"] = [None]
    assert "relation_invalid:unknown" in exp.validate_relation_bundle(bad_relation, documents)
    with pytest.raises(ValueError, match="relation_bundle_fields"):
        exp.create_relation_database({"entities": []}, {})


def test_scenario_harness_7138_sql_runtime_rejection_matrix() -> None:
    """SCENARIO-HARNESS-7138-SQL covers grammar, authorizer, and runtime budgets."""

    connection = exp.create_relation_database({"entities": [], "relations": []}, {})
    for query in (
        "not a select",
        "SELECT * FROM grounded_relations WHERE missing = 1 LIMIT 1",
        "SELECT * FROM grounded_relations ORDER BY missing LIMIT 1",
    ):
        assert exp.execute_bounded_select(connection, query)["reason"] == "unsupported_sql"
    assert (
        exp.execute_bounded_select(
            connection,
            "SELECT * FROM grounded_relations LIMIT 1",
            max_steps=1,
        )["reason"]
        == "step_budget_exhausted"
    )
    assert (
        exp.execute_bounded_select(
            connection,
            "SELECT * FROM grounded_relations LIMIT 1",
            timeout_s=1e-15,
        )["reason"]
        == "time_budget_exhausted"
    )
    connection.set_authorizer(lambda *_args: exp.sqlite3.SQLITE_DENY)
    assert (
        exp.execute_bounded_select(connection, "SELECT * FROM grounded_relations LIMIT 1")["reason"]
        == "unsupported_sql"
    )
    connection.close()


def test_scenario_harness_7138_schema_observation_matrix(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7138-BOOTSTRAP preserves the first schema failure."""

    passed, observed = exp._schema_observation(tmp_path / "missing", ())
    assert passed is False and observed["line"] == 0
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    assert exp._schema_observation(empty, ()) == (False, {"row_count": 0})
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{\n", encoding="utf-8")
    assert exp._schema_observation(invalid, ())[1]["error"] == "invalid_json"
    sequence = tmp_path / "sequence.jsonl"
    sequence.write_text("[]\n", encoding="utf-8")
    assert exp._schema_observation(sequence, ())[1]["error"] == "row_not_object"


def test_scenario_harness_7138_replay_field_mutation_matrix(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7138-REPLAY identifies every independently loaded field."""

    artifact = _artifact(tmp_path)
    changed = deepcopy(artifact)
    fixture = changed["fixture_rows"][0]
    model = changed["model_view_rows"][0]
    sealed = changed["sealed_scorer_rows"][0]
    fixture.update(
        {
            "source_id": "changed",
            "response_id": "changed",
            "split": "train",
            "source_text": "changed",
            "source_text_sha256": "changed",
            "response_text": "changed",
            "response_text_sha256": "changed",
        }
    )
    model.update(
        {
            "split": "train",
            "source_text": "changed",
            "source_text_sha256": "changed",
            "response_text": "changed",
            "response_text_sha256": "changed",
        }
    )
    sealed.update(
        {
            "response_label": "changed",
            "span_labels": [{"changed": True}],
            "span_labels_sha256": "changed",
        }
    )
    changed["model_view_rows"].pop()
    changed["sealed_scorer_rows"].pop()
    receipts, errors = exp.independent_replay(changed, SOURCE_INFO, RESPONSES)
    assert receipts[0]["exact_match"] is False
    for prefix in (
        "fixture_view_count_mismatch",
        "source_id_mismatch:unit-001",
        "response_id_mismatch:unit-001",
        "split_mismatch:unit-001",
        "response_label_mismatch:unit-001",
        "span_labels_mismatch:unit-001",
        "span_labels_hash_mismatch:unit-001",
        "source_text_mismatch:unit-001",
        "response_text_mismatch:unit-001",
    ):
        assert prefix in errors

    changed_responses = tmp_path / "responses.jsonl"
    changed_responses.write_bytes(RESPONSES.read_bytes() + b"\n")
    assert exp.independent_replay(artifact, SOURCE_INFO, changed_responses)[1] == [
        "response_hash_mismatch"
    ]


def test_scenario_harness_7138_build_and_validation_failure_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HARNESS-7138-ARTIFACT rejects every derived terminal state."""

    relative = exp.build_artifact(
        tmp_path,
        exp.RUN_DATE,
        result_path=Path("relative.json"),
        corpus_paths={"source_info": SOURCE_INFO, "responses": RESPONSES, "lineage": LINEAGE},
        duration_s=0.1,
    )
    assert relative["status"] == "complete"

    monkeypatch.setattr(
        exp,
        "build_fixture_views",
        lambda *_args: (_ for _ in ()).throw(ValueError("forced_balance_failure")),
    )
    blocked_balance = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=tmp_path / "balance.json",
        duration_s=0.1,
    )
    assert blocked_balance["gate_check_summary"]["failed_check"] == "fixture_balance"

    changed = deepcopy(relative)
    changed["status"] = "running"
    changed["rows"] = []
    changed["fixture_row_count"] = 71
    changed["model_view_rows"].pop()
    changed["sealed_scorer_rows"][0]["response_label"] = "changed"
    changed["fixture_rows"][0]["source_family"] = "changed"
    changed["split_rows"] = []
    changed["relation_schema"] = {}
    changed["sql_sandbox_contract"] = {}
    changed["mutation_rows"] = []
    changed["independent_loader_rows"] = []
    changed["source_grounding_fixture_ready_score"] = 0
    changed["gate_check_summary"] = {}
    changed["verdict_class"] = "null"
    changed["honest_verdict"] = "null_changed"
    changed["run_date"] = "20260909"
    changed["random_seed"] = 0
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    errors = exp.validate_artifact(changed, SOURCE_INFO, RESPONSES)
    for expected in (
        "run_date mismatch",
        "random_seed mismatch",
        "status mismatch",
        "rows mismatch",
        "fixture_row_count mismatch",
        "view row count mismatch",
        "source family balance mismatch",
        "class balance mismatch",
        "split_rows mismatch",
        "relation_schema mismatch",
        "sql_sandbox_contract mismatch",
        "mutation_rows mismatch",
        "independent_loader_rows mismatch",
        "source_grounding_fixture_ready_score mismatch",
        "gate_check_summary mismatch",
        "verdict_class mismatch",
        "honest_verdict mismatch",
    ):
        assert expected in errors

    blocked = deepcopy(blocked_balance)
    blocked["status"] = "complete"
    blocked["inference_substrate_class"] = exp.INFERENCE_SUBSTRATE_CLASS
    blocked["gate_check_summary"] = {}
    blocked["verdict_class"] = "partial"
    blocked["honest_verdict"] = "partial_changed"
    blocked["source_grounding_fixture_ready_score"] = 1
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    assert {
        "blocked status mismatch",
        "blocked inference_substrate_class mismatch",
        "blocked gate_check_summary mismatch",
        "blocked verdict_class mismatch",
        "blocked honest_verdict mismatch",
        "blocked readiness score mismatch",
    }.issubset(blocked_errors)


def test_scenario_harness_7138_artifact_input_rejections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-HARNESS-7138-ARTIFACT rejects unreadable and malformed inputs."""

    missing = tmp_path / "missing.json"
    assert exp.validate_artifact(missing) == ["artifact_missing"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp.validate_artifact(invalid) == ["artifact_unreadable"]
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert exp.validate_artifact(sequence) == ["artifact_not_object"]
    assert exp.validate_artifact(1) == ["artifact_not_object"]
    assert exp.validate_artifact({})[0].startswith("artifact fields mismatch")
    assert exp.main(["--validate", str(missing)]) == 1
    assert '"valid": false' in capsys.readouterr().out
