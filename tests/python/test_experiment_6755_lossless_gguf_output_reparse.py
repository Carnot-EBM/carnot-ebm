"""Focused tests for the lossless frozen GGUF output reparse.

Spec refs: REQ-VERIFY-6755, SCENARIO-VERIFY-6755-*, REQ-REPORT-6755,
and SCENARIO-REPORT-6755-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from carnot import experiment_6755_lossless_gguf_output_reparse as exp
from carnot.inference import gguf_output_text as boundary


SAT_CNF = {"n_vars": 2, "clauses": [[1], [-2, 1]]}


def _source(row_id: str = "source-1", cnf: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "row_sha256": f"sha256:{row_id}",
        "family": "fixture",
        "cnf": deepcopy(cnf or SAT_CNF),
        "label": "must_not_be_read",
    }


def _proposal(
    raw_output: str = "b'SAT x1=1 x2=0'", row_id: str = "model|source-1"
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "row_sha256": f"sha256:{row_id}",
        "model_family_id": "model",
        "model_hf_id": "unsloth/model-GGUF",
        "model_role": "fixture",
        "family": "fixture",
        "source_row_id": row_id.split("|", 1)[-1],
        "source_row_sha256": "sha256:source",
        "raw_output": raw_output,
        "raw_output_sha256": exp.sha256_text(raw_output),
        "raw_api_response_sha256": "sha256:api",
        "diagnosis": "malformed_certificate",
        "parser_status": "malformed",
        "parse_failure": "unknown_claim",
    }


def _precondition_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [_proposal("b'ABSTAIN'", f"model|source-{index}") for index in range(216)]
    proposal = {
        "rows": rows,
        "frozen_manifest": {"stream_checksum": "sha256:stream"},
    }
    stream = {
        "hardness_stream_ready": True,
        "deterministic_replay_receipt": {"first_stream_sha256": "sha256:stream"},
        "rows": [_source(f"source-{index}") for index in range(216)],
    }
    return proposal, stream


@pytest.mark.parametrize(
    ("value", "expected", "kind"),
    [
        (b"SAT x1=1", "SAT x1=1", "utf8_bytes"),
        ("SAT x1=1", "SAT x1=1", "text"),
        ("b'SAT x1=1'", "SAT x1=1", "legacy_python_bytes_literal"),
        ('b"UNSAT c1,c2"', "UNSAT c1,c2", "legacy_python_bytes_literal"),
        (r"b'SAT x1=1\n'", "SAT x1=1\n", "legacy_python_bytes_literal"),
        ("banana begins with b", "banana begins with b", "text"),
        ("bSAT is ordinary text", "bSAT is ordinary text", "text"),
        ("", "", "text"),
    ],
)
def test_scenario_verify_6755_boundary_accepts_lossless_values(
    value: str | bytes, expected: str, kind: str
) -> None:
    """SCENARIO-VERIFY-6755-BOUNDARY decodes only proven transport forms."""

    receipt = boundary.normalize_gguf_output_text(value, unwrap_legacy_envelope=True)

    assert receipt["text"] == expected
    assert receipt["normalization_kind"] == kind
    assert receipt["semantic_edits_performed"] == 0
    assert receipt["source_bytes_sha256"].startswith("sha256:")


@pytest.mark.parametrize(
    ("value", "failure"),
    [
        (b"\xff", "invalid_utf8"),
        (r"b'\xff'", "invalid_utf8"),
        ("b'SAT' + b' x1=1'", "ambiguous_bytes_literal"),
        ("b\"b'SAT x1=1'\"", "nested_bytes_literal"),
        ("b'''SAT x1=1'''", "ambiguous_bytes_literal"),
        ("b'not closed", "ambiguous_bytes_literal"),
    ],
)
def test_scenario_verify_6755_boundary_rejects_ambiguous_or_lossy_values(
    value: str | bytes, failure: str
) -> None:
    """SCENARIO-VERIFY-6755-BOUNDARY rejects expressions and lossy decoding."""

    with pytest.raises(boundary.OutputTextNormalizationError, match=failure):
        boundary.normalize_gguf_output_text(value, unwrap_legacy_envelope=True)


def test_scenario_verify_6755_boundary_is_idempotent() -> None:
    """SCENARIO-VERIFY-6755-BOUNDARY keeps normalized text stable on replay."""

    first = boundary.normalize_gguf_output_text(r"b'SAT x1=1 x2=0'", unwrap_legacy_envelope=True)
    second = boundary.normalize_gguf_output_text(first["text"], unwrap_legacy_envelope=True)

    assert second["text"] == first["text"]
    assert second["source_bytes_sha256"] == first["normalized_text_sha256"]
    with pytest.raises(boundary.OutputTextNormalizationError, match="unsupported_output_type"):
        boundary.normalize_gguf_output_text(1, unwrap_legacy_envelope=True)  # type: ignore[arg-type]


def test_scenario_verify_6755_boundary_checks_canonical_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-6755-BOUNDARY rejects a failed canonical bytes replay."""

    monkeypatch.setattr(boundary.ast, "literal_eval", lambda _value: b"different")
    with pytest.raises(boundary.OutputTextNormalizationError, match="ambiguous_bytes_literal"):
        boundary.normalize_gguf_output_text("b'SAT x1=1'", unwrap_legacy_envelope=True)


def test_req_verify_6755_preconditions_cover_each_frozen_input_gate() -> None:
    """SCENARIO-VERIFY-6755-BLOCKED checks count, identity, hashes, and stream link."""

    proposal, stream = _precondition_inputs()
    ready = exp.evaluate_preconditions(proposal, stream)
    assert ready["all_passed"] is True
    assert exp.first_failed_check(ready)["check"] == "all_preconditions"

    cases = [
        ("exp6745_row_count", lambda value: value["rows"].pop()),
        (
            "exp6745_unique_row_ids",
            lambda value: value["rows"].__setitem__(1, deepcopy(value["rows"][0])),
        ),
        (
            "exp6745_original_output_hashes",
            lambda value: value["rows"][0].__setitem__("raw_output_sha256", ""),
        ),
    ]
    for check, mutate in cases:
        changed = deepcopy(proposal)
        mutate(changed)
        assert exp.first_failed_check(exp.evaluate_preconditions(changed, stream))["check"] == check

    changed_stream = deepcopy(stream)
    changed_stream["hardness_stream_ready"] = False
    assert exp.first_failed_check(exp.evaluate_preconditions(proposal, changed_stream))[
        "check"
    ] == ("exp6744_ready_stream_link")


def test_req_verify_6755_grammar_flags_are_row_local() -> None:
    """REQ-VERIFY-6755 derives typed, domain, uniqueness, and completeness failures."""

    flags = exp.analyze_grammar_failures("SAT x1=2 x1=0 x3=1 prose", SAT_CNF)
    clause_flags = exp.analyze_grammar_failures(
        "UNSAT c1,c1,c3", {"n_vars": 1, "clauses": [[1], [-1]]}
    )

    assert flags["invalid_variable_reference"] is True
    assert flags["invalid_clause_reference"] is False
    assert flags["non_binary_value"] is True
    assert flags["duplicate"] is True
    assert flags["incomplete_evidence"] is True
    assert flags["invalid_typed_symbol"] is True
    assert flags["environment_grammar_targetable"] is True
    assert clause_flags["invalid_clause_reference"] is True
    assert clause_flags["duplicate"] is True
    assert exp.analyze_grammar_failures("UNSAT c1 nope", SAT_CNF)["invalid_typed_symbol"] is True


def test_scenario_verify_6755_replay_preserves_receipts_and_checks_semantics() -> None:
    """SCENARIO-VERIFY-6755-REPLAY separates envelope recovery from exact validity."""

    row = exp.replay_row(_proposal(), _source())
    false_row = exp.replay_row(_proposal("b'SAT x1=0 x2=0'", "model|source-2"), _source("source-2"))

    assert row["original_output_text"] == "b'SAT x1=1 x2=0'"
    assert row["original_output_sha256"] == exp.sha256_text(row["original_output_text"])
    assert row["normalized_output_text"] == "SAT x1=1 x2=0"
    assert row["normalization_kind"] == "legacy_python_bytes_literal"
    assert row["pre_diagnosis"] == "malformed_certificate"
    assert row["post_diagnosis"] == "exact_valid"
    assert row["encoder_agreement"] is True
    assert row["exact_outcome"] == "exact_valid"
    assert row["semantic_edits_performed"] == 0
    assert row["evidence_preserved"] is True
    assert false_row["post_diagnosis"] == "reasoning_error"
    assert false_row["exact_outcome"] == "false_parseable_proof"
    assert false_row["failure_reason"] == "clause_unsatisfied"


def test_req_verify_6755_replay_handles_parser_and_boundary_failures() -> None:
    """REQ-VERIFY-6755 retains malformed and rejected transport rows without repair."""

    malformed = exp.replay_row(_proposal("b'SAT x1=2'"), _source())
    rejected = exp.replay_row(_proposal("b\"b'SAT x1=1'\""), _source())
    abstention = exp.replay_row(_proposal("b'ABSTAIN'"), _source())

    assert malformed["post_parse_result"]["parse_failure"] == "invalid_sat_term"
    assert malformed["post_diagnosis"] == "malformed_certificate"
    assert malformed["grammar_failures"]["non_binary_value"] is True
    assert rejected["post_diagnosis"] == "transport_normalization_failed"
    assert rejected["normalized_output_text"] is None
    assert rejected["failure_reason"] == "nested_bytes_literal"
    assert rejected["evidence_preserved"] is False
    assert abstention["post_diagnosis"] == "abstention"
    assert abstention["failure_reason"] == "model_abstention"


def test_req_verify_6755_replay_retains_encoder_and_authority_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6755 keeps disagreement and unavailable exact checks explicit."""

    original = exp.encoder_b.encode_certificate

    def disagree(parsed: dict[str, Any]) -> dict[str, Any]:
        result = original(parsed)
        result["normalized_constraints"]["bindings"][0]["values"] = [False]
        return result

    monkeypatch.setattr(exp.encoder_b, "encode_certificate", disagree)
    disagreement = exp.replay_row(_proposal(), _source())
    assert disagreement["post_diagnosis"] == "translation_disagreement"
    assert disagreement["failure_reason"] == "normalized_constraints_disagree"

    monkeypatch.setattr(exp.encoder_b, "encode_certificate", original)

    def unavailable(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("authority unavailable")

    monkeypatch.setattr(exp.frozen, "exact_check_constraints", unavailable)
    failed = exp.replay_row(_proposal(), _source())
    assert failed["post_diagnosis"] == "exact_authority_failed"
    assert failed["exact_outcome"] == "authority_failure"
    assert failed["encoder_a"]["exact_check"]["authority_available"] is False


def test_req_report_6755_blocked_input_is_atomic(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6755-BLOCKED writes the failed parse check and stops."""

    results = tmp_path / "results"
    results.mkdir()
    (results / exp.UPSTREAM_PROPOSAL_PATH.name).write_text("not json", encoding="utf-8")
    artifact = exp.run("20260829", tmp_path)

    assert artifact["status"] == "complete_blocked_reparse_input"
    assert artifact["honest_verdict"].startswith("complete_blocked_reparse_input")
    assert artifact["gate_check_summary"]["failed_check"] == "exp6745_json_object"
    assert artifact["rows"] == []
    assert artifact["verdict_class"] == "blocked"
    assert exp.validate_artifact(artifact) == []
    written = json.loads((results / exp.RESULT_PATH.name).read_text())
    assert written == artifact

    stream_root = tmp_path / "bad-stream"
    stream_results = stream_root / "results"
    stream_results.mkdir(parents=True)
    proposal, stream = _precondition_inputs()
    (stream_results / exp.UPSTREAM_PROPOSAL_PATH.name).write_text(json.dumps(proposal))
    (stream_results / exp.UPSTREAM_STREAM_PATH.name).write_text("[]")
    stream_block = exp.run("20260829", stream_root)
    assert stream_block["gate_check_summary"]["failed_check"] == "exp6744_json_object"

    count_root = tmp_path / "bad-count"
    count_results = count_root / "results"
    count_results.mkdir(parents=True)
    proposal["rows"].pop()
    (count_results / exp.UPSTREAM_PROPOSAL_PATH.name).write_text(json.dumps(proposal))
    (count_results / exp.UPSTREAM_STREAM_PATH.name).write_text(json.dumps(stream))
    count_block = exp.run("20260829", count_root)
    assert count_block["gate_check_summary"]["failed_check"] == "exp6745_row_count"

    assert exp.sha256_file(tmp_path / "absent") == "missing"
    array_path = tmp_path / "array.json"
    array_path.write_text("[]")
    with pytest.raises(TypeError, match="JSON object required"):
        exp.load_json_object(array_path)


def test_req_report_6755_actual_216_row_replay_is_recomputable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6755-ATOMIC replays every paid-for proposal without an LLM."""

    results = tmp_path / "results"
    results.mkdir()
    for source in (exp.UPSTREAM_PROPOSAL_PATH, exp.UPSTREAM_STREAM_PATH):
        shutil.copy2(exp.REPO_ROOT / source, results / source.name)

    artifact = exp.run("20260829", tmp_path)
    reduction = exp.recompute_aggregates(artifact["rows"])

    assert artifact["transport_reparse_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["bytes_envelope_rows"] == 216
    assert artifact["semantic_edits_performed"] == 0
    assert artifact["post_diagnosis_counts"] == {
        "exact_valid": 11,
        "malformed_certificate": 21,
        "reasoning_error": 184,
    }
    assert artifact["invalid_variable_reference_rows"] == 0
    assert artifact["invalid_clause_reference_rows"] == 0
    assert artifact["non_binary_value_rows"] == 20
    assert artifact["duplicate_rows"] == 0
    assert artifact["incomplete_evidence_rows"] == 20
    assert artifact["invalid_typed_symbol_rows"] == 1
    assert artifact["false_parseable_proof_rows"] == 184
    assert artifact["environment_grammar_targetable_rows"] == 21
    assert artifact["exact_valid_rows"] == 11
    assert artifact["verdict_class"] == "positive"
    assert "11/216 exact-valid" in artifact["honest_verdict"]
    assert reduction == {key: artifact[key] for key in exp.ROW_DERIVED_FIELDS}
    assert set(artifact) == set(artifact["field_principles"])
    assert exp.validate_artifact(artifact) == []

    bad_total = deepcopy(artifact)
    bad_total["exact_valid_rows"] += 1
    assert "aggregate_recomputation_mismatch" in exp.validate_artifact(bad_total)
    bad_gate = deepcopy(artifact)
    bad_gate["gate_check_summary"]["all_passed"] = False
    assert "readiness_gate_mismatch" in exp.validate_artifact(bad_gate)
    bad_class = deepcopy(artifact)
    bad_class["verdict_class"] = "partial"
    assert "ready_verdict_class_mismatch" in exp.validate_artifact(bad_class)
    bad_receipt = deepcopy(artifact)
    bad_receipt["reproducibility_receipt"] = []
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(bad_receipt)


def test_req_report_6755_validation_rejects_drift_and_bad_schema(tmp_path: Path) -> None:
    """REQ-REPORT-6755 rejects missing fields, aggregate drift, and checksum drift."""

    assert exp.validate_artifact({}) == [
        "missing_required_fields:" + ",".join(sorted(exp.ARTIFACT_FIELDS))
    ]
    artifact = exp.build_blocked_artifact(
        date="20260829",
        duration_s=0.1,
        failed_check="fixture",
        expected=True,
        observed=False,
        source_artifact_sha256="sha256:source",
    )
    broken = deepcopy(artifact)
    broken["field_principles"].pop("title")
    broken["inference_substrate"] = "wrong"
    broken["verifier_is_oracle"] = True
    broken["verdict_class"] = "wrong"
    broken["honest_verdict"] = "wrong"
    broken["rows"] = [{}]
    assert set(exp.validate_artifact(broken)) == {
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "verifier_is_oracle_mismatch",
        "verdict_class_invalid",
        "reproducibility_checksum_mismatch",
        "blocked_verdict_class_mismatch",
        "blocked_verdict_prefix_mismatch",
        "blocked_rows_invalid",
    }
    with pytest.raises(ValueError):
        exp.write_json_atomic(tmp_path / "bad.json", broken)


def test_req_report_6755_spec_anchors_and_cli_contract() -> None:
    """REQ-REPORT-6755 keeps both capability anchors and the fixed planning date."""

    verify_spec = (exp.REPO_ROOT / "openspec/capabilities/verifiable-reasoning/spec.md").read_text()
    report_spec = (exp.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text()

    assert "REQ-VERIFY-6755" in verify_spec
    assert "SCENARIO-VERIFY-6755-REPLAY" in verify_spec
    assert "REQ-REPORT-6755" in report_spec
    assert "SCENARIO-REPORT-6755-ATOMIC" in report_spec
    assert exp.parse_args([]).date == "20260829"
    assert exp.parse_args(["--date", "20260830"]).date == "20260830"
