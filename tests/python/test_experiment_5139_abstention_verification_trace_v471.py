"""Tests for Exp 5139 abstention and verification trace evaluation.

Spec refs: REQ-INFER-SOTA-032, REQ-PIPELINE-5138,
SCENARIO-INFER-SOTA-032-POOL, SCENARIO-PIPELINE-5138.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5136_receipt_structured_pool_v2_v471 as pool_mod
from carnot import experiment_5139_abstention_verification_trace_v471 as mod
from scripts import experiment_5139_abstention_verification_trace_v471 as script_mod


REPO = Path(__file__).resolve().parents[2]
LLM_SPEC_PATH = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"
PIPELINE_SPEC_PATH = REPO / "openspec/capabilities/pipeline/spec.md"


def _fake_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "loader": "llama.cpp",
            "model_path": "/models/qwen3.6-35b-a3b-q4.gguf",
        },
        {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "gpu": None,
            "loader": "llama.cpp",
            "model_path": "/models/gemma-4-31b-q4.gguf",
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "loader": "llama.cpp",
            "model_path": "/models/gemma-4-26b-a4b-q4.gguf",
        },
    ]


def _write_ready_upstream(
    root: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    artifact_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source_rows = rows
    if source_rows is None:
        source_rows, receipts = pool_mod.build_pool_rows(
            pool_mod.build_task_bank(), _fake_specs(), run_date="20260702"
        )
    else:
        _, receipts = pool_mod.build_pool_rows(
            pool_mod.build_task_bank()[: len(source_rows)], _fake_specs(), run_date="20260702"
        )
    pool_path = root / pool_mod.POOL_RELATIVE_PATH
    pool_mod.write_jsonl(pool_path, source_rows)
    artifact = {
        "experiment_id": pool_mod.EXPERIMENT_ID,
        "milestone": pool_mod.MILESTONE,
        "honest_verdict": pool_mod.SUCCESS_VERDICT,
        "inference_substrate": pool_mod.INFERENCE_SUBSTRATE,
        "duration_s": 143.366125,
        "MODEL_SPECS": _fake_specs(),
        "model_specs": _fake_specs(),
        "structured_pool_v2_clean": True,
        "pool_path": pool_mod.POOL_RELATIVE_PATH,
        "pool_sha256": pool_mod.sha256_file(pool_path),
        "pool_n": len(source_rows),
        "receipt_records": receipts,
        "receipt_record_count": len(receipts),
        "fover_scope_used": False,
        "conductor_modified": False,
        "tests_run": ["fixture"],
    }
    if artifact_overrides:
        artifact.update(artifact_overrides)
    pool_mod.write_json(root / pool_mod.RESULT_RELATIVE_PATH, artifact)
    return source_rows


def test_req_anchored_specs_cover_clean_pool_and_exact_validator_authority() -> None:
    """REQ-INFER-SOTA-032/REQ-PIPELINE-5138: specs anchor Exp 5139 inputs."""

    llm_spec = LLM_SPEC_PATH.read_text(encoding="utf-8")
    pipeline_spec = PIPELINE_SPEC_PATH.read_text(encoding="utf-8")

    assert "### REQ-INFER-SOTA-032" in llm_spec
    assert "structured_pool_v2_clean" in llm_spec
    assert "deterministic exact validators are the only ground truth" in llm_spec
    assert "### REQ-PIPELINE-5138" in pipeline_spec
    assert "exact_validator_authority" in pipeline_spec


def test_trace_records_validate_schema_and_score_against_exact_validator_only() -> None:
    """SCENARIO-INFER-SOTA-032-POOL: self-check fields are scored features."""

    rows, receipts = pool_mod.build_pool_rows(
        pool_mod.build_task_bank(), _fake_specs(), run_date="20260702"
    )
    traces = mod.build_trace_records(rows, receipts, _fake_specs())
    metrics = mod.evaluate_trace_records(traces)

    assert len(traces) == len(rows)
    assert metrics["schema_validity_rate"] == pytest.approx(1.0)
    assert metrics["evidence_validity_rate"] == pytest.approx(1.0)
    assert metrics["answer_correctness"]["direct_answer_accuracy"] < 1.0
    assert metrics["repair_correctness"]["attempt_rate"] > 0.0
    assert metrics["repair_correctness"]["exact_correct_rate"] > 0.0

    trace = traces[1]
    score = mod.score_trace(trace)
    mutated = copy.deepcopy(trace)
    mutated["structured_output"]["self_check"]["claimed_correct"] = not mutated[
        "structured_output"
    ]["self_check"]["claimed_correct"]
    mutated_score = mod.score_trace(mutated)
    assert mutated_score["answer_correct"] == score["answer_correct"]
    assert mutated_score["self_check_calibrated"] is not score["self_check_calibrated"]

    bad_evidence = copy.deepcopy(trace)
    bad_evidence["structured_output"]["evidence"]["raw_response_hash"] = "sha256:bad"
    assert mod.score_trace(bad_evidence)["evidence_valid"] is False

    bad_schema = copy.deepcopy(trace)
    del bad_schema["structured_output"]["answer"]
    assert mod.validate_trace_schema(bad_schema) is False
    assert mod.validate_trace_schema({"structured_output": []}) is False
    assert mod.validate_trace_schema({"structured_output": {"answer": None}}) is False
    for mutate in [
        lambda item: item | {"evidence": {}},
        lambda item: item | {"self_check": {"result": "maybe", "claimed_correct": True}},
        lambda item: (
            item | {"self_check": {"result": "pass", "claimed_correct": "yes", "confidence": 0.9}}
        ),
        lambda item: (
            item | {"self_check": {"result": "pass", "claimed_correct": True, "confidence": 2.0}}
        ),
        lambda item: item | {"uncertainty": {}},
        lambda item: item | {"abstention": {"decision": "defer"}},
    ]:
        broken = copy.deepcopy(trace["structured_output"])
        assert mod.validate_trace_schema({"structured_output": mutate(broken)}) is False

    assert mod._evidence_matches({}, None) is False
    assert mod.build_trace_records([{"task_id": "empty", "candidates": []}], [], []) == []


def test_write_artifact_reports_abstention_metrics_and_required_fields(tmp_path: Path) -> None:
    """REQ-PIPELINE-5138: artifact reports exact-validator abstention utility."""

    rows = _write_ready_upstream(tmp_path)

    artifact = mod.write_artifact(
        root=tmp_path,
        run_date="20260702",
        tests_run=["focused"],
        current_duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == mod.SUCCESS_READY_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(143.366125)
    assert artifact["MODEL_SPECS"] == _fake_specs()
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]
    assert artifact["upstream_pool_artifact"] == pool_mod.RESULT_RELATIVE_PATH
    assert artifact["trace_schema"]["required"] == list(mod.TRACE_REQUIRED_FIELDS)
    assert artifact["schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["exact_validator_authority"]["llm_judge_used_as_ground_truth"] is False
    assert artifact["exact_validator_authority"]["self_check_used_as_ground_truth"] is False
    assert artifact["harmful_answer_reduction"] > 0.0
    assert artifact["false_abstain_rate"] < mod.FALSE_ABSTAIN_RATE_MAX
    assert artifact["abstention_delta"]["delta"] >= 0.0
    assert artifact["strongest_baseline"]["baseline"] in artifact["baseline_metrics"]
    assert artifact["verification_trace_ready"] is True
    assert artifact["conductor_modified"] is False
    assert artifact["tests_run"] == ["focused"]
    assert artifact["trace_count"] == len(rows)
    assert (
        artifact["coverage_risk_curve"][0]["coverage"]
        >= artifact["coverage_risk_curve"][-1]["coverage"]
    )

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact


def test_dirty_missing_or_incomplete_upstream_blocks_without_traces(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-032-POOL: closed upstream gates fail closed."""

    missing = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    mod.validate_artifact(missing)
    assert missing["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert missing["verification_trace_ready"] is False
    assert missing["trace_records"] == []
    assert "missing upstream" in missing["preconditions_checked"]["upstream_error"]
    assert mod.read_jsonl(tmp_path / "missing.jsonl") == []

    upstream_path = tmp_path / pool_mod.RESULT_RELATIVE_PATH
    upstream_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path.write_text("{", encoding="utf-8")
    malformed = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert malformed["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert "JSONDecodeError" in malformed["preconditions_checked"]["upstream_error"]

    upstream_path.write_text("[]", encoding="utf-8")
    non_object = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert non_object["honest_verdict"] == mod.BLOCKED_UPSTREAM_VERDICT
    assert "not a JSON object" in non_object["preconditions_checked"]["upstream_error"]

    _write_ready_upstream(tmp_path, artifact_overrides={"structured_pool_v2_clean": False})
    dirty = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert dirty["honest_verdict"] == mod.BLOCKED_POOL_VERDICT
    assert dirty["preconditions_checked"]["structured_pool_v2_clean"] is False

    _write_ready_upstream(tmp_path)
    (tmp_path / pool_mod.POOL_RELATIVE_PATH).unlink()
    missing_rows = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert missing_rows["honest_verdict"] == mod.BLOCKED_ROWS_VERDICT

    _write_ready_upstream(tmp_path, artifact_overrides={"MODEL_SPECS": _fake_specs()[:2]})
    missing_model = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])
    assert missing_model["honest_verdict"] == mod.BLOCKED_MODEL_VERDICT


def test_baselines_curve_validation_and_cli_edges(tmp_path: Path) -> None:
    """REQ-PIPELINE-5138: baselines, coverage-risk, validation, and CLI are deterministic."""

    _write_ready_upstream(tmp_path)
    artifact = mod.write_artifact(root=tmp_path, run_date="20260702", tests_run=["focused"])

    baselines = mod.evaluate_baselines(artifact["trace_records"])
    curve = mod.coverage_risk_curve(artifact["trace_records"])
    assert set(baselines) == {
        "non_abstaining_direct_answer",
        "confidence_threshold",
        "exact_constraint_only_filter",
    }
    assert baselines["non_abstaining_direct_answer"]["coverage"] == pytest.approx(1.0)
    assert baselines["exact_constraint_only_filter"]["harmful_answer_rate"] == pytest.approx(0.0)
    assert curve == sorted(curve, key=lambda row: row["threshold"])

    for mutate, message in [
        (
            lambda item: {key: value for key, value in item.items() if key != "duration_s"},
            "missing",
        ),
        (lambda item: item | {"experiment_id": "bad"}, "experiment_id"),
        (lambda item: item | {"milestone": "2026.07.470"}, "milestone"),
        (lambda item: item | {"honest_verdict": "bad"}, "honest_verdict"),
        (
            lambda item: item | {"honest_verdict": "complete_wrong_success"},
            "verification_trace_ready",
        ),
        (lambda item: item | {"inference_substrate": "bad"}, "substrate"),
        (lambda item: item | {"MODEL_SPECS": []}, "MODEL_SPECS"),
        (
            lambda item: item | {"MODEL_SPECS": [], "model_specs": []},
            "MODEL_SPECS",
        ),
        (lambda item: item | {"model_specs": []}, "model_specs"),
        (lambda item: item | {"upstream_pool_artifact": "bad"}, "upstream"),
        (lambda item: item | {"trace_schema": {}}, "trace_schema"),
        (lambda item: item | {"schema_validity_rate": 0.5}, "schema validity"),
        (lambda item: item | {"exact_validator_authority": {}}, "validator authority"),
        (lambda item: item | {"coverage_risk_curve": []}, "coverage_risk_curve"),
        (lambda item: item | {"abstention_delta": {}}, "abstention_delta"),
        (lambda item: item | {"harmful_answer_reduction": -0.1}, "harmful_answer_reduction"),
        (lambda item: item | {"false_abstain_rate": 1.0}, "false_abstain_rate"),
        (lambda item: item | {"strongest_baseline": {}}, "strongest_baseline"),
        (
            lambda item: (
                item
                | {
                    "exact_validator_authority": item["exact_validator_authority"]
                    | {"authority_intact": False}
                }
            ),
            "authority",
        ),
        (lambda item: item | {"verification_trace_ready": False}, "verification_trace_ready"),
        (lambda item: item | {"conductor_modified": True}, "conductor_modified"),
        (lambda item: item | {"tests_run": []}, "tests_run"),
    ]:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(mutate(artifact))

    assert script_mod.main(["--root", str(tmp_path), "--date", "20260702"]) == 0
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
