"""Tests for Exp 5223 GAP-4 flagged-pool authenticity audit.

Spec refs: REQ-REPORT-5223, SCENARIO-REPORT-5223-QUARANTINE,
SCENARIO-REPORT-5223-PREFLIGHT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5223_gap4_flagged_pool_authenticity_audit_v478 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _canonical_record(**overrides: Any) -> JsonDict:
    row: JsonDict = {
        "candidate_id": "gap4:unit:0001",
        "source_task_id": "human_replay:unit:1",
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "model_path_or_digest": "sha256:" + "a" * 64,
        "prompt_digest": "sha256:" + "b" * 64,
        "random_seed": 5223,
        "generation_started_at": "2026-07-04T00:00:00Z",
        "generation_duration_s": 61.0,
        "decoding_protocol": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 384,
            "stop": ["\n\n\n"],
        },
        "pass_at_1_fields": {
            "vote_top1": False,
            "gated_top1": False,
            "scoring_protocol": "experiment_5161_5177_5197_gap4",
        },
        "pass_at_2_fields": {
            "vote_top2": False,
            "gated_top2": True,
            "scoring_protocol": "experiment_5161_5177_5197_gap4",
        },
        "validation_inputs_digest": "sha256:" + "c" * 64,
        "provenance_kind": "live_llm_generation",
    }
    row.update(overrides)
    return row


def _legacy_pool_artifact(**overrides: Any) -> JsonDict:
    row = {
        "accepted": True,
        "task_id": "human_replay:unit:0",
        "model_hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "model_path": "/models/qwen.gguf",
        "live_prompted": False,
        "repair_strategy": "demo_lookup_same_shape",
        "repair_attempts": 1,
        "guard_status": "accepted",
        "demo_perfect": True,
        "output_shape_matches": True,
        "code": "def transform(grid):\n    return [list(row) for row in grid]\n",
    }
    artifact: JsonDict = {
        "experiment": "experiment_5211_gap4_sota_local_candidate_expansion_v477",
        "candidate_pool_n": 120,
        "accepted_rows": 120,
        "repair_attempts": 120,
        "gap4_expansion_usable": True,
        "models_used": [],
        "duration_s": 48.6,
        "inference_substrate": "live_llm_generation_with_deterministic_execution_guard",
        "honest_verdict": "complete_gap4_sota_local_candidate_expansion_v477_n120_pool_ready_for_exp5212",
        "candidate_rows": [dict(row) for _ in range(120)],
    }
    artifact.update(overrides)
    return artifact


def test_req_report_5223_spec_and_schema_declare_canonical_fields() -> None:
    """REQ-REPORT-5223: OpenSpec and JSON schema declare the canonical contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    schema_path = REPO / mod.CANONICAL_SCHEMA_RELATIVE_PATH
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    for marker in (
        "REQ-REPORT-5223",
        "SCENARIO-REPORT-5223-QUARANTINE",
        "SCENARIO-REPORT-5223-PREFLIGHT",
        mod.RESULT_RELATIVE_PATH,
        mod.CANONICAL_SCHEMA_RELATIVE_PATH,
    ):
        assert marker in spec

    for field in mod.CANONICAL_CANDIDATE_REQUIRED_FIELDS:
        assert field in schema["required"]
        assert field in schema["properties"]


def test_req_report_5223_canonical_record_guard_requires_provenance_and_protocol() -> None:
    """REQ-REPORT-5223: canonical rows require model, seed, and pass fields."""

    assert mod.canonical_candidate_record_errors(_canonical_record()) == []

    bad = _canonical_record(model_id="", random_seed=None)
    bad.pop("pass_at_2_fields")
    bad["pass_at_1_fields"] = {"vote_top1": True}

    assert mod.canonical_candidate_record_errors(bad) == [
        "missing_model_provenance",
        "missing_pass_at_1_fields",
        "missing_pass_at_2_fields",
        "missing_random_seed",
    ]

    missing = mod.canonical_candidate_record_errors(
        {"generation_duration_s": True, "decoding_protocol": {}}
    )
    for reason in (
        "missing_candidate_id",
        "missing_decoding_protocol",
        "missing_generation_duration_s",
        "missing_model_provenance",
        "missing_pass_at_1_fields",
        "missing_pass_at_2_fields",
        "missing_random_seed",
        "missing_validation_inputs_digest",
    ):
        assert reason in missing


def test_scenario_report_5223_preflight_rejects_flagged_gap4_pool_shapes() -> None:
    """SCENARIO-REPORT-5223-PREFLIGHT: flagged v477 shapes cannot pass."""

    validation = {
        "n_scored": {"value": 0, "principle": "unit"},
        "scored_rows": [],
    }
    result = mod.preflight_gap4_validation(
        pool_artifact=_legacy_pool_artifact(),
        validation_artifact=validation,
    )

    assert result.passed is False
    assert result.validated_pool_n == 0
    assert result.protocol_fields_complete is False
    assert result.gap4_pool_repairable is False
    for reason in (
        "missing_models_used",
        "missing_random_seed",
        "missing_protocol_pass1_fields",
        "missing_protocol_pass2_fields",
        "validation_n_scored_zero",
        "generation_duration_too_short",
        "tautology_shaped_pool_ready",
    ):
        assert reason in result.reasons

    try:
        mod.require_gap4_pool_preflight(result)
    except ValueError as exc:
        assert "missing_models_used" in str(exc)
    else:  # pragma: no cover - the assertion above must raise.
        raise AssertionError("preflight unexpectedly passed")


def test_scenario_report_5223_preflight_accepts_canonical_scored_rows() -> None:
    """SCENARIO-REPORT-5223-PREFLIGHT: canonical rows can pass the guard."""

    pool = {
        "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
        "candidate_rows": [_canonical_record()],
        "duration_s": None,
        "inference_substrate": "artifact_provenance_audit",
    }
    validation = {"scored_rows": [{"candidate_id": "gap4:unit:0001"}]}

    result = mod.preflight_gap4_validation(pool_artifact=pool, validation_artifact=validation)

    assert result.passed is True
    assert result.reasons == []
    assert result.validated_pool_n == 1
    assert result.protocol_fields_complete is True
    assert result.gap4_pool_repairable is True
    mod.require_gap4_pool_preflight(result)


def test_req_report_5223_preflight_handles_empty_and_malformed_inputs() -> None:
    """REQ-REPORT-5223: malformed inputs fail closed without synthetic counts."""

    assert mod._scored_rows(None) == []

    result = mod.preflight_gap4_validation(
        pool_artifact={
            "models_used": ["unit-model"],
            "candidate_rows": "not-a-list",
            "duration_s": True,
        },
        validation_artifact={"n_scored": "bad", "scored_rows": "not-a-list"},
    )

    assert result.passed is False
    assert result.reasons == ["validation_n_scored_zero"]
    assert result.validated_pool_n == 0
    assert result.checked_candidate_n == 0

    no_validation = mod.preflight_gap4_validation(
        pool_artifact={"models_used": ["unit-model"], "events": []}
    )
    assert no_validation.passed is True
    assert no_validation.validated_pool_n == 0

    missing_model = _canonical_record(model_path_or_digest="")
    missing_model_result = mod.preflight_gap4_validation(
        pool_artifact={
            "models_used": ["unit-model"],
            "candidate_rows": [missing_model],
        },
        validation_artifact={"scored_rows": [{"candidate_id": "gap4:unit:0001"}]},
    )
    assert missing_model_result.reasons == ["missing_model_provenance"]


def test_scenario_report_5223_actual_v477_artifacts_are_quarantined() -> None:
    """SCENARIO-REPORT-5223-QUARANTINE: checked-in v477 artifacts stay out of headlines."""

    artifact = mod.build_audit_artifact(
        root=REPO,
        tests_run=["unit: pass"],
        guard_tests_added=True,
        duration_s=0.25,
    )

    assert artifact["gap4_pool_repairable"] is False
    assert artifact["validated_pool_n"] == 0
    assert artifact["protocol_fields_complete"] is False
    assert artifact["quarantined_artifacts"] == list(mod.QUARANTINED_ARTIFACTS)
    assert artifact["canonical_schema_path"] == mod.CANONICAL_SCHEMA_RELATIVE_PATH
    assert artifact["guard_tests_added"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert "must be regenerated" in artifact["honest_verdict"]
    assert artifact["inference_substrate"] == "artifact_provenance_audit"
    assert artifact["artifact_findings"][mod.EXP5211_RELATIVE_PATH]["headline_eligible"] is False
    assert artifact["artifact_findings"][mod.EXP5212_RELATIVE_PATH]["validated_pool_n"] == 0
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5223_run_writes_artifact_and_success_path_for_canonical_pool(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-5223: the writer preserves bare top-level audit fields."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (tmp_path / mod.EXP5211_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "candidate_pool_n": 1,
                "accepted_rows": 1,
                "repair_attempts": 0,
                "gap4_expansion_usable": False,
                "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
                "candidate_rows": [_canonical_record()],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / mod.EXP5211_CHECKPOINT_RELATIVE_PATH).write_text(
        json.dumps({"events": [_canonical_record()]}),
        encoding="utf-8",
    )
    (tmp_path / mod.EXP5212_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "n_scored": {"value": 1, "principle": "unit"},
                "scored_rows": [{"candidate_id": "gap4:unit:0001"}],
            }
        ),
        encoding="utf-8",
    )
    ticks = iter([10.0, 11.5])

    artifact = mod.run(
        root=tmp_path,
        tests_run=["unit: pass"],
        guard_tests_added=True,
        now=lambda: next(ticks),
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["gap4_pool_repairable"] is True
    assert artifact["validated_pool_n"] == 1
    assert artifact["protocol_fields_complete"] is True
    assert artifact["honest_verdict"].startswith("success:")
    assert mod.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["tests_run"] = "unit"
    with pytest.raises(ValueError, match="tests_run"):
        mod.write_artifact(tmp_path, bad)


def test_req_report_5223_artifact_schema_rejects_wrapped_or_overclaimed_fields() -> None:
    """REQ-REPORT-5223: terminal fields stay bare and cannot overclaim repairability."""

    artifact = mod.build_audit_artifact(
        root=REPO,
        tests_run=["unit: pass"],
        guard_tests_added=True,
        duration_s=0.25,
    )

    bad = dict(artifact)
    bad["gap4_pool_repairable"] = {"value": False}
    bad["protocol_fields_complete"] = True
    bad["validated_pool_n"] = "0"
    bad["honest_verdict"] = "blocked_without_terminal_prefix"
    bad["reproducibility_checksum"] = "sha256:bad"

    assert mod.artifact_schema_errors(bad) == [
        "gap4_pool_repairable_bare_bool",
        "honest_verdict_terminal_prefix",
        "protocol_fields_complete",
        "reproducibility_checksum",
        "validated_pool_n_bare_int",
    ]

    malformed = dict(artifact)
    malformed.pop("experiment")
    malformed["field_principles"] = {}
    malformed["protocol_fields_complete"] = "false"
    malformed["quarantined_artifacts"] = []
    malformed["canonical_schema_path"] = "bad.json"
    malformed["guard_tests_added"] = "true"
    malformed["tests_run"] = "unit"
    malformed["inference_substrate"] = "live_llm_generation"
    malformed["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(malformed)
    for reason in (
        "missing required field experiment",
        "field_principles",
        "protocol_fields_complete_bare_bool",
        "quarantined_artifacts",
        "canonical_schema_path",
        "guard_tests_added_bare_bool",
        "tests_run",
        "inference_substrate",
        "reproducibility_checksum",
    ):
        assert reason in errors


def test_req_report_5223_missing_source_artifacts_build_blocked_audit(tmp_path: Path) -> None:
    """REQ-REPORT-5223: absent source artifacts produce an explicit nonrepairable audit."""

    artifact = mod.build_audit_artifact(
        root=tmp_path,
        tests_run=[],
        guard_tests_added=False,
        duration_s=0.0,
    )

    assert artifact["gap4_pool_repairable"] is False
    assert artifact["validated_pool_n"] == 0
    assert artifact["artifact_findings"][mod.EXP5211_RELATIVE_PATH]["exists"] is False
    assert artifact["artifact_findings"][mod.EXP5211_CHECKPOINT_RELATIVE_PATH]["candidate_rows"] == 0
    assert artifact["artifact_findings"][mod.EXP5212_RELATIVE_PATH]["excluded_rows"] == 0
