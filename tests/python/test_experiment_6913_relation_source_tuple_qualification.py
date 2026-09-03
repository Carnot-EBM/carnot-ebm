"""Tests for deterministic source and tuple qualification.

Spec refs: REQ-CONSTRAINT-6913 and SCENARIO-CONSTRAINT-6913-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6913_relation_source_tuple_qualification as mod


MODEL_ARM = "gguf:example/model"
RULE_ARM = "rule:anchored_lexical_v1"
ENOKI_ARM = "enoki:pinned_openie_encoder"


def _source(*, negative: bool = False) -> dict[str, Any]:
    text = "Café evidence: Node n0 has color red."
    if negative:
        text += " Node n0 does not have color red."
    return {
        "fixture_id": "graph_coloring_00",
        "group_id": "relation_group_00",
        "family": "graph_coloring",
        "split": "calibration",
        "source_order": 0,
        "source_text": text,
        "source_text_hash": mod.sha256_bytes(text.encode("utf-8")),
        "relation_schema_version": "anchored_relation_v1",
        "allowed_predicates": ["has_color"],
    }


def _span(text: str, phrase: str, *, last: bool = False) -> tuple[int, int]:
    index = text.rindex(phrase) if last else text.index(phrase)
    start = len(text[:index].encode("utf-8"))
    return start, start + len(phrase.encode("utf-8"))


def _line(
    source: dict[str, Any],
    *,
    subject: str = "n0",
    object_text: str = "red",
    polarity: str = "positive",
    predicate: str = "has_color",
    last_object: bool = False,
) -> str:
    subject_start, subject_end = _span(source["source_text"], subject)
    object_start, object_end = _span(source["source_text"], object_text, last=last_object)
    return "\t".join(
        (
            "REL",
            str(subject_start),
            str(subject_end),
            predicate,
            str(object_start),
            str(object_end),
            polarity,
        )
    )


def _cell(
    source: dict[str, Any],
    raw_output: str,
    *,
    arm: str = MODEL_ARM,
    identity: str = "example/model::7::graph_coloring_00",
    parser_attempted: bool = True,
) -> dict[str, Any]:
    payload = raw_output.encode("utf-8")
    digest = mod.sha256_bytes(payload)
    model_id = arm.removeprefix("gguf:") if arm.startswith("gguf:") else None
    return {
        "cell_identity": identity,
        "arm": arm,
        "hf_id": model_id,
        "model_family": "example_family" if model_id else arm.split(":", 1)[0],
        "seed": 7 if model_id else None,
        "fixture_id": source["fixture_id"],
        "group_id": source["group_id"],
        "family": source["family"],
        "split": source["split"],
        "source_order": source["source_order"],
        "source_text_hash": source["source_text_hash"],
        "raw_output_b64": base64.b64encode(payload).decode("ascii"),
        "raw_output_sha256": digest,
        "output_byte_count": len(payload),
        "parser_attempted": parser_attempted,
        "parser_input_sha256": digest,
        "parse_rows": [],
        "terminal": True,
        "timed_out": False,
        "truncated": False,
        "stop_reason": "stop",
    }


def _qualified(source: dict[str, Any], raw_output: str, **cell_kwargs: Any) -> dict[str, Any]:
    return mod.qualify_cell(
        _cell(source, raw_output, **cell_kwargs),
        source,
        mod.source_relation_contract(source),
    )


def test_req_constraint_6913_spec_declares_all_failure_boundaries() -> None:
    """REQ-CONSTRAINT-6913 owns the source-only contract before implementation."""

    text = Path("openspec/capabilities/constraint-verification/spec.md").read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6913") :]
    for suffix in (
        "PRECONDITIONS",
        "BYTES",
        "PARSER",
        "ENTITIES",
        "COMPLETENESS",
        "DENOMINATORS",
        "AGGREGATES",
        "READINESS",
    ):
        assert f"SCENARIO-CONSTRAINT-6913-{suffix}" in section


def test_utf8_offsets_partial_substrings_and_normalized_text_fail() -> None:
    """SCENARIO-CONSTRAINT-6913-BYTES requires exact complete UTF-8 entities."""

    source = _source()
    good = _qualified(source, _line(source))
    assert good["source_grounding_label"] == "source_grounded_correct"

    character_offset = source["source_text"].index("n0")
    object_start, object_end = _span(source["source_text"], "red")
    drift_line = (
        f"REL\t{character_offset}\t{character_offset + 2}\thas_color\t"
        f"{object_start}\t{object_end}\tpositive"
    )
    drift = _qualified(source, drift_line)
    assert drift["source_offset_check"]["passed"] is False
    assert drift["source_grounded_correct"] is False

    partial = _qualified(source, _line(source, subject="Node"))
    assert partial["source_byte_identity_check"]["passed"] is True
    assert partial["entity_anchor_check"]["passed"] is False

    text = source["source_text"]
    start, end = _span(text, "Café")
    normalized = mod.check_source_span(text.encode("utf-8"), start, end, "Cafe\u0301")
    assert normalized["offset_valid"] is True
    assert normalized["byte_identity"] is False
    assert base64.b64decode(normalized["exact_source_bytes_b64"]) == "Café".encode()


def test_parser_bypass_malformed_protocol_and_invalid_tuple_types_fail() -> None:
    """SCENARIO-CONSTRAINT-6913-PARSER preserves parser and tuple failures."""

    source = _source()
    bypass = _qualified(source, _line(source), parser_attempted=False)
    assert bypass["parser_check"]["state"] == "parser_bypass"
    assert bypass["source_grounded_correct"] is False

    mismatch_cell = _cell(source, _line(source))
    mismatch_cell["parser_input_sha256"] = "sha256:changed"
    mismatch = mod.qualify_cell(mismatch_cell, source, mod.source_relation_contract(source))
    assert mismatch["parser_check"]["state"] == "parser_input_mismatch"

    malformed = _qualified(source, "REL\t1\t2")
    assert malformed["parser_check"]["state"] == "malformed"
    assert malformed["proposal_covered"] is False

    assert mod.check_tuple_types(["node_0", "has_color", "red"])["arity_valid"] is False
    wrong_type = mod.check_tuple_types(["node_0", "has_color", "red", 1])
    assert wrong_type["field_types_valid"] is False
    assert wrong_type["passed"] is False


def test_unknown_entities_and_reversed_relations_fail_direction() -> None:
    """SCENARIO-CONSTRAINT-6913-ENTITIES requires typed entities and direction."""

    source = _source()
    source["source_text"] += " Blue is only a note."
    source["source_text_hash"] = mod.sha256_bytes(source["source_text"].encode())
    unknown = _qualified(source, _line(source, object_text="Blue"))
    assert unknown["entity_anchor_check"]["passed"] is False
    assert unknown["relation_direction_check"]["passed"] is False

    reversed_line = _line(source, subject="red", object_text="n0")
    reversed_row = _qualified(source, reversed_line)
    assert reversed_row["source_byte_identity_check"]["passed"] is True
    assert reversed_row["relation_direction_check"]["passed"] is False

    unsupported = _qualified(source, _line(source, predicate="unknown_relation"))
    assert unsupported["relation_direction_check"]["passed"] is False


def test_duplicates_omissions_and_false_abstentions_remain_terminal() -> None:
    """SCENARIO-CONSTRAINT-6913-COMPLETENESS keeps all proposal failures."""

    source = _source(negative=True)
    positive = _line(source)
    duplicate = _qualified(source, f"{positive}\n{positive}")
    assert duplicate["duplicate_check"]["duplicate_count"] == 1
    assert duplicate["omission_check"]["omitted_count"] == 1
    assert duplicate["source_grounded_correct"] is False

    complete = _qualified(
        source,
        f"{positive}\n{_line(source, polarity='negative', last_object=True)}",
    )
    assert complete["omission_check"]["omitted_count"] == 0
    assert complete["source_grounding_label"] == "source_grounded_correct"

    abstention = _qualified(source, "ABSTAIN")
    assert abstention["abstention_check"] == {
        "abstained": True,
        "required_relation_count": 2,
        "false_abstention": True,
        "passed": False,
    }
    assert abstention["source_grounding_label"] == "source_grounding_failed_false_abstention"

    empty = _qualified(source, "")
    assert empty["parser_check"]["state"] == "empty"
    assert empty["omission_check"]["omitted_count"] == 2


def test_enoki_saved_json_is_reparsed_without_accepting_substrings() -> None:
    """SCENARIO-CONSTRAINT-6913-BYTES applies to the Enoki control too."""

    source = _source()
    result = {
        "sentence": source["source_text"],
        "triples": [
            {"subject": "Node", "relation": "has color", "object": "red", "confidence": 0.9}
        ],
    }
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"))
    row = _qualified(
        source,
        raw,
        arm=ENOKI_ARM,
        identity="enoki:pinned_openie_encoder::deterministic::graph_coloring_00",
    )
    assert row["parser_check"]["state"] == "parsed"
    assert row["entity_anchor_check"]["passed"] is False
    assert row["model_id"] == ENOKI_ARM


def test_row_summaries_keep_control_and_model_denominators_separate() -> None:
    """SCENARIO-CONSTRAINT-6913-DENOMINATORS never fills a GGUF row from controls."""

    source = _source()
    model_good = _qualified(source, _line(source))
    model_bad = _qualified(
        source,
        "ABSTAIN",
        identity="example/model::8::graph_coloring_00",
    )
    model_bad["seed"] = 8
    rule = _qualified(
        source,
        _line(source),
        arm=RULE_ARM,
        identity="rule:anchored_lexical_v1::deterministic::graph_coloring_00",
    )
    enoki_payload = json.dumps(
        {"triples": [{"subject": "n0", "relation": "has color", "object": "red"}]}
    )
    enoki = _qualified(
        source,
        enoki_payload,
        arm=ENOKI_ARM,
        identity="enoki:pinned_openie_encoder::deterministic::graph_coloring_00",
    )

    summaries = mod.summarize_rows([model_good, model_bad, rule, enoki])
    by_arm = {row["value"]: row for row in summaries["arm_summary_rows"]}
    assert by_arm[MODEL_ARM]["denominator"] == 2
    assert by_arm[MODEL_ARM]["source_grounded_numerator"] == 1
    assert by_arm[RULE_ARM]["denominator"] == 1
    assert by_arm[ENOKI_ARM]["denominator"] == 1
    assert {row["value"] for row in summaries["seed_summary_rows"]} == {"7", "8", "deterministic"}
    assert len(summaries["wilson_interval_rows"]) == 2 * sum(
        len(summaries[name])
        for name in (
            "arm_summary_rows",
            "model_summary_rows",
            "family_summary_rows",
            "seed_summary_rows",
        )
    )


def test_wilson_intervals_and_aggregate_disagreement_are_exact() -> None:
    """SCENARIO-CONSTRAINT-6913-AGGREGATES detects changed reported metrics."""

    assert mod.wilson_interval(0, 0) == {"confidence": 0.95, "lower": None, "upper": None}
    interval = mod.wilson_interval(1, 2)
    assert interval["lower"] == pytest.approx(0.09452865)
    assert interval["upper"] == pytest.approx(0.90547135)

    source = _source()
    rows = [_qualified(source, _line(source))]
    summaries = mod.summarize_rows(rows)
    comparison = mod.compare_reported_metrics(
        rows,
        summaries["proposal_coverage_by_arm"],
        summaries["source_grounded_correctness_by_arm"],
    )
    assert comparison["agreement"] is True

    changed = deepcopy(summaries["proposal_coverage_by_arm"])
    changed[MODEL_ARM]["numerator"] = 0
    disagreement = mod.compare_reported_metrics(
        rows, changed, summaries["source_grounded_correctness_by_arm"]
    )
    assert disagreement["agreement"] is False
    assert disagreement["comparisons"][0]["passed"] is False


def test_preconditions_block_hash_identity_and_held_access_drift() -> None:
    """SCENARIO-CONSTRAINT-6913-PRECONDITIONS fails before scoring unsafe input."""

    expected_ids = {"cell-a"}
    receipt = {
        "clean_relation_corpus_ready_score": 1,
        "cell_identity_rows": [
            {"cell_identity": "cell-a", "occurrence_count": 1, "expected": True}
        ],
        "source_artifact_hashes": {"exp6900": {"observed_sha256": "sha256:source"}},
    }
    cells = [{"cell_identity": "cell-a", "terminal": True}]
    report = mod.validate_preconditions(
        receipt=receipt,
        observed_hashes={"exp6886": "sha256:fixture", "exp6900": "sha256:source"},
        expected_hashes={"exp6886": "sha256:fixture", "exp6900": "sha256:source"},
        cells=cells,
        expected_cell_ids=expected_ids,
        held_sidecar_access_count=0,
    )
    assert report["passed"] is True

    wrong_hash = mod.validate_preconditions(
        receipt=receipt,
        observed_hashes={"exp6886": "changed", "exp6900": "sha256:source"},
        expected_hashes={"exp6886": "sha256:fixture", "exp6900": "sha256:source"},
        cells=cells,
        expected_cell_ids=expected_ids,
        held_sidecar_access_count=1,
    )
    assert wrong_hash["failed_check"] == "source_hash:exp6886"
    assert {row["check"] for row in wrong_hash["failed_checks"]} >= {
        "source_hash:exp6886",
        "held_sidecar_access_count",
    }


def test_artifact_readiness_means_complete_not_universally_correct() -> None:
    """SCENARIO-CONSTRAINT-6913-READINESS permits complete measured failures."""

    source = _source()
    rows = [
        _qualified(source, _line(source)),
        _qualified(source, "ABSTAIN", identity="example/model::8::graph_coloring_00"),
    ]
    rows[1]["seed"] = 8
    artifact = mod.build_artifact(
        rows=rows,
        date="20260903",
        duration_s=1.25,
        source_artifact_hashes={"exp6900": {"observed_sha256": "sha256:test"}},
        preconditions_checked={"passed": True, "checks": []},
        expected_cell_ids={row["cell_identity"] for row in rows},
        expected_arms={MODEL_ARM},
        expected_families={"graph_coloring"},
    )
    assert artifact["source_tuple_shard_ready_score"] == 1
    assert artifact["source_grounded_correctness_by_arm"][MODEL_ARM]["rate"] == 0.5
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["held_sidecar_access_count"] == 0
    assert artifact["model_inference_call_count"] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert all(
        f"gate:{row['check']}" in artifact["field_principles"]
        for row in artifact["gate_check_summary"]["checks"]
    )
    assert mod.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["proposal_coverage_by_arm"][MODEL_ARM]["numerator"] = 0
    assert "aggregate_disagreement" in mod.validate_artifact(changed)


def test_blocked_artifact_has_complete_gate_and_required_schema() -> None:
    """SCENARIO-CONSTRAINT-6913-PRECONDITIONS emits a complete blocked receipt."""

    check = mod.gate_check("clean_relation_corpus_ready_score", 1, 0)
    artifact = mod.blocked_artifact(
        date="20260903",
        duration_s=0.1,
        source_artifact_hashes={},
        preconditions_checked=mod.gate_summary([check]),
    )
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["verdict_class"] == "blocked"
    assert artifact["source_tuple_shard_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "clean_relation_corpus_ready_score"
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)


def test_extract_sources_requires_hash_verified_saved_rule_bytes() -> None:
    """SCENARIO-CONSTRAINT-6913-PRECONDITIONS binds source text to saved bytes."""

    source = _source()
    cell = _cell(
        source,
        _line(source),
        arm=RULE_ARM,
        identity="rule:anchored_lexical_v1::deterministic::graph_coloring_00",
    )
    source_bytes = source["source_text"].encode()
    cell.update(
        {
            "raw_request_b64": base64.b64encode(source_bytes).decode("ascii"),
            "raw_request_sha256": mod.sha256_bytes(source_bytes),
            "request_byte_count": len(source_bytes),
        }
    )
    fixture_rows = [
        {
            "row_type": "fixture",
            "fixture_id": source["fixture_id"],
            "group_id": source["group_id"],
            "family": source["family"],
            "split": source["split"],
            "source_text_hash": source["source_text_hash"],
        }
    ]
    extracted = mod.extract_source_records([cell], fixture_rows)
    assert extracted == [source]

    cell["raw_request_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="source_request_hash_mismatch"):
        mod.extract_source_records([cell], fixture_rows)


def test_main_forwards_date_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CONSTRAINT-6913 exposes the required command-line surface."""

    output = tmp_path / "artifact.json"

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        output.write_text(kwargs["date"], encoding="utf-8")
        return {}

    monkeypatch.setattr(mod, "run", fake_run)
    assert mod.main(["--date", "20260903", "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == "20260903"
