"""Tests for the lossless claim-span protocol.

Spec refs: REQ-VERIFY-7437 and SCENARIO-VERIFY-7437-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7437_v652_span_protocol as experiment


REPO_ROOT = Path(__file__).resolve().parents[2]


def _predictor_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(32):
        split = "train" if index < 6 else "test"
        response = (
            f"Group {index} reported exactly {index + 2} confirmed cases before 2026. "
            "The unchanged response keeps this explicit count and time qualifier."
        )
        rows.append(
            {
                "group_id": f"source-{index:02d}",
                "response_id": f"response-{index:02d}",
                "split": split,
                "source_text": f"Source context {index}",
                "response_text": response,
                "task_type": "Summary",
            }
        )
    return rows


def test_parser_reconstructs_unicode_and_verbatim_claims() -> None:
    """SCENARIO-VERIFY-7437-PARSER: both arms reconstruct exact Unicode text."""

    paragraph = "Café prices rose by €5 only in 2026."
    span_reply = json.dumps({"claims": [[0, len(paragraph)]]})
    span = experiment.parse_claim_output(span_reply, arm="span", paragraph=paragraph)
    verbatim = experiment.parse_claim_output(
        json.dumps({"claims": [paragraph]}), arm="verbatim", paragraph=paragraph
    )

    assert span["disposition"] == "completed_valid"
    assert span["claims"] == [paragraph]
    assert span["claim_spans"] == [[0, len(paragraph)]]
    assert span["literal_span_reconstruction"] is True
    assert verbatim["claims"] == span["claims"]
    assert verbatim["claim_spans"] == span["claim_spans"]


def test_parser_accepts_explicit_empty_and_rejects_ambiguous_verbatim() -> None:
    """SCENARIO-VERIFY-7437-PARSER: empty is explicit and duplicate text fails closed."""

    empty = experiment.parse_claim_output(
        '{"claims":[]}', arm="span", paragraph="No factual proposition."
    )
    duplicate = experiment.parse_claim_output(
        '{"claims":["It rained."]}',
        arm="verbatim",
        paragraph="It rained. It rained.",
    )

    assert empty["disposition"] == "completed_valid_empty"
    assert empty["claims"] == []
    assert duplicate["disposition"] == "duplicate_text_ambiguous"
    assert duplicate["parse_valid"] is False
    assert duplicate["repair_attempted"] is False


@pytest.mark.parametrize(
    ("reply", "arm", "expected"),
    [
        ('{"claims":[[0,4]]', "span", "partial_json_object"),
        ("not-json", "span", "malformed_json"),
        ('{"claims":[],"extra":1}', "span", "top_level_schema"),
        ('{"claims":[[true,4]]}', "span", "span_type"),
        ('{"claims":[[0,99]]}', "span", "span_bounds"),
        ('{"claims":["absent"]}', "verbatim", "claim_not_found"),
        ('{"claims":[3]}', "verbatim", "verbatim_claim_type"),
        ('{"claims":"bad"}', "span", "claims_not_list"),
        ('```json\n{"claims":[]}\n```', "span", "markdown_fence_forbidden"),
    ],
)
def test_parser_has_explicit_malformed_dispositions(reply: str, arm: str, expected: str) -> None:
    """SCENARIO-VERIFY-7437-PARSER: malformed output is not repaired."""

    parsed = experiment.parse_claim_output(reply, arm=arm, paragraph="text")
    assert parsed["disposition"] == expected
    assert parsed["repair_attempted"] is False
    assert parsed["retry_count"] == 0


def test_parser_retains_missing_modifier_as_a_failed_endpoint() -> None:
    """SCENARIO-VERIFY-7437-PARSER: the parser does not add an omitted modifier."""

    paragraph = "The launch did not happen before 2024."
    claim = "The launch did"
    parsed = experiment.parse_claim_output(
        json.dumps({"claims": [claim]}),
        arm="verbatim",
        paragraph=paragraph,
        required_modifiers=("not", "before 2024"),
    )

    assert parsed["parse_valid"] is True
    assert parsed["qualifier_retention"] is False
    assert parsed["missing_modifiers"] == ["not", "before 2024"]
    assert parsed["claims"] == [claim]


def test_paragraph_seal_preserves_offsets_and_marks_clipping() -> None:
    """SCENARIO-VERIFY-7437-PARSER: clipped text cannot claim full-response coverage."""

    short = experiment.seal_response_paragraph("First claim.\n\nSecond claim.")
    long = experiment.seal_response_paragraph("é" * 513)

    assert short["paragraph"] == "First claim."
    assert short["paragraph_start"] == 0
    assert short["paragraph_end"] == len("First claim.")
    assert short["complete_response_coverage_eligible"] is False
    assert long["paragraph"] == "é" * 512
    assert long["paragraph_end"] == 512
    assert long["clipped"] is True
    assert long["complete_response_coverage_eligible"] is False


def test_label_blind_panel_and_schedule_freeze_48_paired_units() -> None:
    """SCENARIO-VERIFY-7437-PANEL: 24 distinct groups share exact paired input."""

    panel = experiment.seal_panel(_predictor_rows())
    schedule = experiment.build_evaluation_schedule(panel["evaluation"])

    assert len(panel["development"]) == 4
    assert len(panel["evaluation"]) == 24
    assert len({row["group_id"] for row in panel["evaluation"]}) == 24
    assert len(schedule) == 48
    for index in range(0, len(schedule), 2):
        pair = schedule[index : index + 2]
        assert {row["arm"] for row in pair} == {"span", "verbatim"}
        assert pair[0]["paragraph"] == pair[1]["paragraph"]
        assert pair[0]["paragraph_sha256"] == pair[1]["paragraph_sha256"]
        assert all(row["max_new_tokens"] == 256 for row in pair)
        assert all(row["temperature"] == 0.0 for row in pair)
        assert all(row["generation_count"] == 1 for row in pair)
        assert all("annotation" not in row["prompt"].lower() for row in pair)


def test_panel_selection_does_not_accept_evaluator_fields() -> None:
    """SCENARIO-VERIFY-7437-PANEL: evaluator authority cannot enter selection."""

    rows = _predictor_rows()
    rows[0]["labels"] = [{"start": 0, "end": 5}]
    with pytest.raises(ValueError, match="predictor_contains_evaluator_field:labels"):
        experiment.seal_panel(rows)


def test_constructed_pairs_are_separate_exact_qualifier_controls() -> None:
    """SCENARIO-VERIFY-7437-SCOPE: twelve pairs test modifiers, not entailment."""

    pairs = experiment.constructed_qualifier_pairs()
    controls = experiment.reduce_constructed_pairs(pairs)

    assert len(pairs) == 12
    assert len({row["pair_id"] for row in pairs}) == 12
    assert len(controls) == 24
    assert {row["arm"] for row in controls} == {"span", "verbatim"}
    assert all(row["parse_valid"] for row in controls)
    assert all(row["qualifier_retention"] for row in controls)
    assert all(row["scope"] == "constructed_exact_check" for row in controls)


def test_archive_reduction_keeps_two_failures_distinct() -> None:
    """SCENARIO-VERIFY-7437-ARCHIVE: truncation and receipt absence remain separate."""

    source = REPO_ROOT / experiment.ARCHIVE_PATH
    artifact = json.loads(source.read_text(encoding="utf-8"))
    reduced = experiment.reduce_archived_capture(artifact)

    assert len(reduced["archive_diagnosis_rows"]) == 4
    assert {row["finish_reason"] for row in reduced["archive_diagnosis_rows"]} == {"length"}
    assert {row["actual_token_ceiling"] for row in reduced["archive_diagnosis_rows"]} == {64}
    assert {row["tokenizer_identity"] for row in reduced["archive_diagnosis_rows"]} == {
        "embedded_gguf"
    }
    assert all(
        row["parse_status"] == "invalid_truncated_json" for row in reduced["archive_diagnosis_rows"]
    )
    assert reduced["producer_validation_diagnosis"]["missing_receipt"] == "adversarial_verify"
    assert reduced["producer_validation_diagnosis"]["cause"] == "missing_passing_terminal_receipt"


def test_terminal_receipt_names_reject_missing_adversarial_then_accept_complete() -> None:
    """SCENARIO-VERIFY-7437-TERMINAL: the repaired sequence fails then passes."""

    incomplete = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in experiment.TERMINAL_CHECK_NAMES
        if name != "adversarial_verify"
    ]
    complete = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in experiment.TERMINAL_CHECK_NAMES
    ]

    assert experiment.validate_terminal_receipts(incomplete) == [
        "missing_terminal_receipt:adversarial_verify"
    ]
    assert experiment.validate_terminal_receipts(complete) == []
    duplicate = [*complete, deepcopy(complete[0])]
    assert experiment.validate_terminal_receipts(duplicate) == [
        f"duplicate_terminal_receipt:{complete[0]['name']}"
    ]


def test_fixture_artifact_is_no_model_and_independently_replayable(tmp_path: Path) -> None:
    """REQ-VERIFY-7437: readiness binds parser, provenance, panel, and receipts."""

    artifact = experiment.build_fixture_artifact(tmp_path)
    errors = experiment.validate_artifact(artifact, root=tmp_path, require_terminal=True)

    assert errors == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["span_protocol_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert len(artifact["rows"]) == 48
    assert all(row["status"] == "unstarted" for row in artifact["rows"])
    assert artifact["sample_size_budget"]["planned"] == 48
    assert artifact["sample_size_budget"]["unstarted"] == 48
    assert artifact["verifier_is_oracle"] is True
    assert artifact["human_annotations_certify_extraction"] is False
    assert artifact["claim_span_is_entailment_verifier"] is False
    assert experiment.independent_reduce_artifact(artifact, root=tmp_path) == []


def test_validator_rejects_checksum_rows_and_receipt_mutations(tmp_path: Path) -> None:
    """REQ-VERIFY-7437: cold validation fails on bound evidence drift."""

    artifact = experiment.build_fixture_artifact(tmp_path)

    changed = deepcopy(artifact)
    changed["rows"][0]["max_new_tokens"] = 255
    assert "evaluation_rows_mismatch" in experiment.validate_artifact(
        changed, root=tmp_path, require_terminal=True
    )
    assert "evaluation_rows_mismatch" in experiment.independent_reduce_artifact(
        changed, root=tmp_path, require_terminal=True
    )

    changed = deepcopy(artifact)
    changed["validation_receipts"] = [
        row for row in changed["validation_receipts"] if row["name"] != "adversarial_verify"
    ]
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert "missing_terminal_receipt:adversarial_verify" in experiment.validate_artifact(
        changed, root=tmp_path, require_terminal=True
    )

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["unsloth/Qwen3.8-27B-GGUF"]
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert "current_model_provenance_invalid" in experiment.validate_artifact(
        changed, root=tmp_path, require_terminal=True
    )


def test_date_argument_and_arm_validation() -> None:
    """REQ-VERIFY-7437: closed identities reject silent protocol drift."""

    assert experiment.date_argument("20260920") == "20260920"
    with pytest.raises(ValueError, match="date must be 20260920"):
        experiment.date_argument("20260919")
    with pytest.raises(ValueError, match="unsupported_arm"):
        experiment.parse_claim_output('{"claims":[]}', arm="triples", paragraph="x")


def test_defensive_parser_and_panel_boundaries(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7437-PARSER: closed inputs reject every ambiguous boundary."""

    assert experiment._load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert experiment._load_object(malformed) == {}
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert experiment._load_object(scalar) == {}
    assert (
        experiment.parse_claim_output('{"claims":[""]}', arm="verbatim", paragraph="text")[
            "disposition"
        ]
        == "verbatim_claim_empty"
    )
    with pytest.raises(ValueError, match="response_text_required"):
        experiment.seal_response_paragraph("")
    leading = experiment.seal_response_paragraph("\n\nKept paragraph.")
    assert leading["paragraph"] == "Kept paragraph."
    assert leading["paragraph_start"] == 2

    with pytest.raises(ValueError, match="predictor_fields_missing"):
        experiment.seal_panel([{"group_id": "x"}])
    with pytest.raises(ValueError, match="insufficient_distinct_groups"):
        experiment.seal_panel(_predictor_rows()[:8])
    overlapping = _predictor_rows()
    overlapping[6]["group_id"] = overlapping[0]["group_id"]
    for row in overlapping[7:]:
        row["group_id"] = f"test-{row['group_id']}"
    with pytest.raises(ValueError, match="development_evaluation_group_overlap"):
        experiment.seal_panel(overlapping)
    with pytest.raises(ValueError, match="evaluation_group_count"):
        experiment.build_evaluation_schedule([])
    with pytest.raises(ValueError, match="unsupported_arm"):
        experiment._prompt("triples", "text")


def test_archive_and_receipt_defensive_branches() -> None:
    """SCENARIO-VERIFY-7437-ARCHIVE: malformed archives and failed receipts stay explicit."""

    with pytest.raises(ValueError, match="archive_development_row_count"):
        experiment.reduce_archived_capture({"development_rows": []})
    bad_rows: list[object] = [{}, {}, {}, None]
    with pytest.raises(ValueError, match="archive_development_row_shape"):
        experiment.reduce_archived_capture({"development_rows": bad_rows})
    valid_rows = [
        {
            "call_id": str(index),
            "finish_reason": "stop",
            "completion_tokens": 1,
            "raw_reply": "{}",
            "raw_request": {"max_tokens": 64},
            "raw_response": {},
        }
        for index in range(4)
    ]
    reduced = experiment.reduce_archived_capture(
        {
            "MODEL_SPECS": [{"native_tokenizer": "fixture"}],
            "development_rows": valid_rows,
            "validation_receipts": [{"name": "adversarial_verify", "passed": True, "exit_code": 0}],
        }
    )
    assert {row["parse_status"] for row in reduced["archive_diagnosis_rows"]} == {"valid_json"}
    assert reduced["producer_validation_diagnosis"]["cause"] is None
    failed = [
        {"name": name, "passed": name != "adversarial_verify", "exit_code": 0}
        for name in experiment.TERMINAL_CHECK_NAMES
    ]
    assert experiment.validate_terminal_receipts(failed) == [
        "failed_terminal_receipt:adversarial_verify"
    ]


def test_sidecar_reference_evaluator_and_private_receipt_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7437-TERMINAL: sidecars and private receipt candidates are bound."""

    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("{}", encoding="utf-8")
    reference = experiment._reference(outside, root=tmp_path)
    assert Path(reference["path"]).is_absolute()
    assert experiment._serialize_source("plain") == "plain"
    assert experiment._serialize_source({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'

    panel = experiment.seal_panel(_predictor_rows())
    evaluators = {
        row["response_id"]: {
            "labels": [{"start": 0, "end": 5}],
            "quality": "good",
            "model": "archived",
            "temperature": 0.0,
        }
        for row in panel["evaluation"]
    }
    view = experiment._evaluator_view(panel, evaluators)
    assert len(view["rows"]) == 24
    assert all(row["human_annotation_count"] == 1 for row in view["rows"])
    with pytest.raises(ValueError, match="evaluator_missing"):
        experiment._evaluator_view(panel, {})

    fixture = experiment.build_fixture_artifact(tmp_path / "fixture")
    affected = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in experiment.AFFECTED_CHECK_NAMES
    ]
    controls = experiment._receipt_controls(fixture, root=tmp_path, affected=affected)
    assert [row["passed"] for row in controls] == [True, True]


def test_bound_reader_and_source_hash_boundaries(tmp_path: Path) -> None:
    """REQ-VERIFY-7437: missing or changed sidecars and source bytes fail closed."""

    missing = {"path": "missing.json", "sha256": "sha256:nope"}
    with pytest.raises(ValueError, match="sidecar_hash_mismatch"):
        experiment._read_bound_json(tmp_path, missing)
    empty = tmp_path / "empty.json"
    empty.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar_invalid"):
        experiment._read_bound_json(
            tmp_path,
            {"path": "empty.json", "sha256": experiment.sha256_file(empty)},
        )

    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"flagged_adversarial":true}\n', encoding="utf-8")
    hashes = experiment._source_hashes(tmp_path, [Path("artifact.json"), Path("missing.json")])
    assert list(hashes) == ["artifact.json"]
    assert hashes["artifact.json"]["original_flagged_adversarial"] is True


def test_validator_reports_each_cold_boundary(tmp_path: Path) -> None:
    """REQ-VERIFY-7437: each required cold boundary has a specific error."""

    artifact = experiment.build_fixture_artifact(tmp_path)

    mutations = [
        ("field_principles", {}, "field_principles_mismatch"),
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("honest_verdict", "null_without_prefix", "terminal_verdict_prefix_invalid"),
        ("span_protocol_manifest", None, "span_protocol_manifest_invalid"),
        ("parser_control_rows", [], "constructed_parser_controls_invalid"),
        ("archive_diagnosis_rows", [], "archive_diagnosis_invalid"),
        ("producer_validation_diagnosis", {}, "archive_producer_diagnosis_invalid"),
    ]
    for field, replacement, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = replacement
        changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
        assert expected in experiment.validate_artifact(
            changed, root=tmp_path, require_terminal=True
        )

    blocked = deepcopy(artifact)
    blocked.update(
        {
            "verdict_class": "blocked",
            "honest_verdict": "wrong",
            "span_protocol_ready_score": 1,
        }
    )
    blocked["reproducibility_checksum"] = experiment.artifact_checksum(blocked)
    blocked_errors = experiment.validate_artifact(blocked, root=tmp_path, require_terminal=True)
    assert "blocked_verdict_prefix_invalid" in blocked_errors
    assert "blocked_readiness_invalid" in blocked_errors


def test_validator_reports_manifest_and_sidecar_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-7437: panel, schedule, evaluator, and prompt drift fail closed."""

    artifact = experiment.build_fixture_artifact(tmp_path)
    manifest = deepcopy(artifact["span_protocol_manifest"])
    manifest["evaluation_unit_count"] = 47
    changed = deepcopy(artifact)
    changed["span_protocol_manifest"] = manifest
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert "span_protocol_manifest_hash_mismatch" in experiment.validate_artifact(
        changed, root=tmp_path, require_terminal=True
    )

    for reference_name, mutation, expected in [
        ("panel", lambda value: value.update({"development": []}), "panel_counts_invalid"),
        (
            "panel",
            lambda value: value["evaluation"].__setitem__(1, deepcopy(value["evaluation"][0])),
            "panel_groups_not_distinct",
        ),
        ("schedule", lambda value: value.update({"rows": []}), "schedule_identity_mismatch"),
        (
            "evaluator",
            lambda value: value.update({"annotation_scope": "wrong"}),
            "evaluator_scope_invalid",
        ),
    ]:
        case_root = tmp_path / expected
        case = experiment.build_fixture_artifact(case_root)
        ref = case["span_protocol_manifest"][reference_name]
        path = experiment._resolve_reference(case_root, ref)
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)
        experiment.atomic_json(path, value)
        ref["sha256"] = experiment.sha256_file(path)
        copied = deepcopy(case["span_protocol_manifest"])
        copied.pop("manifest_hash")
        case["span_protocol_manifest"]["manifest_hash"] = experiment.canonical_hash(copied)
        case["reproducibility_checksum"] = experiment.artifact_checksum(case)
        assert expected in experiment.validate_artifact(case, root=case_root, require_terminal=True)

    leak_root = tmp_path / "prompt-leak"
    leaked = experiment.build_fixture_artifact(leak_root)
    schedule_ref = leaked["span_protocol_manifest"]["schedule"]
    schedule_path = experiment._resolve_reference(leak_root, schedule_ref)
    schedule_value = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_value["rows"][0]["prompt"] += " evaluator annotation"
    experiment.atomic_json(schedule_path, schedule_value)
    schedule_ref["sha256"] = experiment.sha256_file(schedule_path)
    leaked["span_protocol_manifest"]["schedule_identity"] = experiment.canonical_hash(
        schedule_value["rows"]
    )
    copied = deepcopy(leaked["span_protocol_manifest"])
    copied.pop("manifest_hash")
    leaked["span_protocol_manifest"]["manifest_hash"] = experiment.canonical_hash(copied)
    leaked["reproducibility_checksum"] = experiment.artifact_checksum(leaked)
    assert "evaluator_annotation_prompt_leak" in experiment.validate_artifact(
        leaked, root=leak_root, require_terminal=True
    )


def test_independent_reducer_handles_missing_manifest_and_sidecar(tmp_path: Path) -> None:
    """REQ-VERIFY-7437: independent reduction preserves cold-reader failures."""

    artifact = experiment.build_fixture_artifact(tmp_path)
    missing_manifest = deepcopy(artifact)
    missing_manifest["span_protocol_manifest"] = None
    missing_manifest["reproducibility_checksum"] = experiment.artifact_checksum(missing_manifest)
    assert "span_protocol_manifest_invalid" in experiment.independent_reduce_artifact(
        missing_manifest, root=tmp_path
    )

    missing_sidecar = deepcopy(artifact)
    missing_sidecar["span_protocol_manifest"]["schedule"]["path"] = "absent.json"
    copied = deepcopy(missing_sidecar["span_protocol_manifest"])
    copied.pop("manifest_hash")
    missing_sidecar["span_protocol_manifest"]["manifest_hash"] = experiment.canonical_hash(copied)
    missing_sidecar["reproducibility_checksum"] = experiment.artifact_checksum(missing_sidecar)
    errors = experiment.independent_reduce_artifact(missing_sidecar, root=tmp_path)
    assert any(error.startswith("sidecar_hash_mismatch") for error in errors)
