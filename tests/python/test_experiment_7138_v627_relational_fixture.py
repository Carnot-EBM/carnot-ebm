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
    assert exp.select_source_groups(sources, exp.response_membership_rows(flipped)) == selected_groups
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
    assert "relation_object_invalid:r001" in exp.validate_relation_bundle(
        bad_unknown, documents
    )

    bad_span = deepcopy(bundle)
    bad_span["relations"][0]["provenance"][0]["text"] = "Else"
    assert "provenance_text_mismatch:r001:0" in exp.validate_relation_bundle(
        bad_span, documents
    )

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
    assert exp.execute_bounded_select(
        connection,
        "SELECT * FROM grounded_relations LIMIT 1",
        max_steps=0,
    )["reason"] == "step_budget_exhausted"
    assert exp.execute_bounded_select(
        connection,
        "SELECT * FROM grounded_relations LIMIT 1",
        timeout_s=0.0,
    )["reason"] == "time_budget_exhausted"
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
    assert "fixture_row_order_mismatch" in exp.independent_replay(
        reordered, SOURCE_INFO, RESPONSES
    )[1]

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
    assert "source_info_hash_mismatch" in exp.independent_replay(
        artifact, changed_source, RESPONSES
    )[1]


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
    assert hash_block["gate_check_summary"]["expected_value"] == exp.PINNED_CORPUS_HASHES[
        "source_info"
    ]
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
