"""Tests for Exp 2978 first-step and semantic-energy telemetry panel.

Spec refs: REQ-VERIFY-2978, SCENARIO-VERIFY-2978.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import first_step_semantic_energy_repair_telemetry_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
REQUIRED_FIELDS = {
    "honest_verdict",
    "telemetry_panel_ready",
    "first_step_signal_usable",
    "semantic_energy_signal_usable",
    "logprob_unavailable",
    "no_headline_verifier_claim",
    "models_used",
    "mandatory_headline_model_ids",
    "candidate_rows",
    "calibration_metrics",
    "triage_examples",
    "failure_modes_explained",
    "inference_substrate",
    "duration_s",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "model_path": "/models/gemma.gguf",
        }
    ]


def _config(tmp_path: Path) -> exp.TelemetryPanelConfig:
    return exp.TelemetryPanelConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        started_at=10.0,
        clock=lambda: 16.25,
    )


def _write_sources(root: Path, *, include_2977: bool = True) -> None:
    raw2964 = root / "results" / "raw" / "experiment_2964"
    _write_text(raw2964 / "ok.py", "public public public def misleading_prefix(x):\n    return x\n")
    _write_text(raw2964 / "bad.py", "def looks_ok_but_fails(x):\n    return x - 1\n")
    _write_json(
        root / "results" / exp.EXP2964_FILENAME,
        {
            "honest_verdict": "complete: source",
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "model_specs": _model_specs(),
            "candidate_evaluations": [
                {
                    "task_id": "MBPP:ok",
                    "mode": "taxonomy_guided",
                    "sample_index": 0,
                    "seed": 296400,
                    "model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "schema_valid": True,
                    "syntax_success": True,
                    "passed": True,
                    "verifier_accepted": True,
                    "verifier_score": 1.0,
                    "raw_response_ref": "results/raw/experiment_2964/ok.py",
                    "tokens_generated": 16,
                },
                {
                    "task_id": "MBPP:bad",
                    "mode": "baseline_no_taxonomy",
                    "sample_index": 1,
                    "seed": 296401,
                    "model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "schema_valid": True,
                    "syntax_success": True,
                    "passed": False,
                    "verifier_accepted": False,
                    "verifier_score": 0.25,
                    "raw_response_ref": "results/raw/experiment_2964/bad.py",
                    "test_status": "failed",
                    "execution_error_type": "AssertionError",
                    "tokens_generated": 17,
                },
                "malformed-candidate-row-is-skipped",
            ],
        },
    )

    raw2967 = root / "results" / "sota_nl_to_z3_dccd_2967_raw"
    _write_json(raw2967 / "ok.json", {"draft_text": "{", "structured_output": "(check-sat)"})
    _write_json(raw2967 / "bad.json", {"draft_text": "no json object", "structured_output": ""})
    _write_json(
        root / "results" / exp.EXP2967_FILENAME,
        {
            "honest_verdict": "complete: source",
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "model_specs": _model_specs(),
            "per_item_results": [
                {
                    "item_id": "lf-ok",
                    "model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "parseable": True,
                    "z3_executed": True,
                    "answer_correct": True,
                    "solver_formula_correct": True,
                    "failure_category": "solver_verified_correct",
                    "raw_response_path": str(raw2967 / "ok.json"),
                },
                {
                    "item_id": "lf-bad",
                    "model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "parseable": False,
                    "z3_executed": False,
                    "answer_correct": False,
                    "solver_formula_correct": False,
                    "failure_category": "unparseable",
                    "parse_error": "no_json_object",
                    "raw_response_path": str(raw2967 / "bad.json"),
                },
                "malformed-formalization-row-is-skipped",
            ],
        },
    )

    if include_2977:
        raw2977 = root / "results" / "raw" / "experiment_2977"
        _write_text(raw2977 / "repair.py", "def trace_aware(x):\n    return x + 1\n")
        _write_text(raw2977 / "schema.txt", '{"repaired_code": "unterminated"')
        _write_json(
            root / "results" / exp.EXP2977_FILENAME,
            {
                "honest_verdict": "blocked_cached_sota_pair_unavailable_cpu_smoke_only",
                "models_used": ["Qwen/Qwen3.5-0.8B"],
                "mandatory_headline_model_ids": list(exp.MANDATORY_HEADLINE_MODEL_IDS),
                "model_specs": _model_specs()
                + [{"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"}],
                "candidate_evaluations": [
                    {
                        "task_id": "MBPP:smoke-ok",
                        "mode": "intent_preserving_trace_aware_repair",
                        "sample_index": 0,
                        "seed": 297700,
                        "model_hf_id": "Qwen/Qwen3.5-0.8B",
                        "schema_valid": True,
                        "syntax_success": True,
                        "passed": True,
                        "verifier_accepted": True,
                        "raw_candidate_ref": "results/raw/experiment_2977/repair.py",
                        "runtime_trace_present": True,
                        "tokens_generated": 12,
                    },
                    {
                        "task_id": "MBPP:smoke-bad",
                        "mode": "schema_only_dccd",
                        "sample_index": 1,
                        "seed": 297701,
                        "model_hf_id": "Qwen/Qwen3.5-0.8B",
                        "schema_valid": False,
                        "syntax_success": False,
                        "passed": False,
                        "verifier_accepted": False,
                        "raw_candidate_ref": "results/raw/experiment_2977/schema.txt",
                        "schema_diagnostics": {"schema_errors": ["unterminated string"]},
                        "tokens_generated": 8,
                    },
                ],
            },
        )

    _write_json(
        root / "results" / exp.EXP2968_FILENAME,
        {
            "partial_monitor_harness_ready": True,
            "monitor_results": [
                {"checks_passed": True, "trace_kind": "code"},
                {"checks_passed": False, "trace_kind": "solver"},
            ],
        },
    )


def test_req_verify_2978_spec_anchor_exists() -> None:
    """REQ-VERIFY-2978: the telemetry panel is anchored in OpenSpec."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-2978" in spec
    assert "SCENARIO-VERIFY-2978" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert 'no_headline_verifier_claim=true' in spec


def test_scenario_verify_2978_extracts_rows_and_proxy_calibration(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2978: available artifacts become calibrated triage rows."""
    _write_sources(tmp_path)

    artifact = exp.build_artifact(_config(tmp_path))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["telemetry_panel_ready"] is True
    assert artifact["candidate_rows"] == 6
    assert artifact["logprob_unavailable"] is True
    assert artifact["no_headline_verifier_claim"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert "Qwen/Qwen3.5-0.8B" in artifact["models_used"]
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" in artifact["models_used"]
    assert artifact["mandatory_headline_model_ids"] == list(exp.MANDATORY_HEADLINE_MODEL_IDS)

    rows = artifact["candidate_level_rows"]
    assert rows[0]["final_verifier_outcome"] in {True, False}
    assert {"schema_status", "syntax_status", "failure_category"} <= set(rows[0])
    assert any(row["candidate_kind"] == "nl_to_z3_formalization" for row in rows)
    assert any(row["candidate_kind"] == "code_repair" for row in rows)

    metrics = artifact["calibration_metrics"]
    assert metrics["semantic_energy_proxy_failure_score"]["auroc"] is not None
    assert metrics["first_step_proxy_failure_score"]["sample_count"] == 6
    assert metrics["direct_logprob_features"]["available"] is False

    example_types = {example["example_type"] for example in artifact["triage_examples"]}
    assert {"false_positive", "false_negative"} <= example_types
    assert artifact["failure_modes_explained"]["schema_failure"]["count"] == 2
    assert artifact["failure_modes_explained"]["syntax_failure"]["count"] == 2


def test_req_verify_2978_writes_terminal_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-2978: write_artifact persists the required terminal JSON."""
    _write_sources(tmp_path, include_2977=False)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text())

    assert saved == artifact
    assert artifact["candidate_rows"] == 4
    assert artifact["source_artifacts"][f"results/{exp.EXP2977_FILENAME}"]["available"] is False
    assert artifact["duration_s"] == pytest.approx(6.25)
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_2978_missing_candidate_sources_blocks(tmp_path: Path) -> None:
    """REQ-VERIFY-2978: no repair or solver rows produces a blocked artifact."""
    artifact = exp.write_artifact(_config(tmp_path))

    assert artifact["honest_verdict"] == "blocked_no_candidate_artifacts_available"
    assert artifact["telemetry_panel_ready"] is False
    assert artifact["candidate_rows"] == 0
    assert artifact["calibration_metrics"]["direct_logprob_features"]["available"] is False
    assert artifact["no_headline_verifier_claim"] is True
    assert artifact["triage_examples"] == []


def test_req_verify_2978_metric_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-2978: edge helpers fail closed and keep proxy semantics explicit."""
    default_config = exp.TelemetryPanelConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 1.0)
    assert default_config.resolved_output_path() == tmp_path / "results" / exp.ARTIFACT_FILENAME

    assert exp.compute_auroc([1], [0.2]) is None
    with pytest.raises(ValueError, match="same length"):
        exp.compute_auroc([1], [0.1, 0.2])

    assert exp._code_failure_category({}, True, False, False) == "syntax_failure"
    assert exp._code_failure_category({"false_accept": True}, True, True, False) == "false_accept"
    assert (
        exp._code_failure_category({"original_failure_categories": ["name_error"]}, True, True, False)
        == "name_error"
    )
    assert exp._code_failure_category({}, True, True, False) == "verifier_rejected"

    assert exp._first_step_features("Z3 plan: declare-fun x", "nl_to_z3_formalization")[
        "first_step_proxy_failure_score"
    ] < 0.5
    assert exp._first_step_features("plain prose", "code_repair")["first_step_proxy_failure_score"] > 0.5
    assert (
        exp._semantic_energy_proxy(
            schema_valid=True,
            syntax_success=True,
            first_step_failure=0.0,
            tokens_generated=384,
        )
        > 0.0
    )
    assert exp._artifact_confidence_proxy({"verifier_output": {"score": "0.75"}}, False) == pytest.approx(
        0.75
    )

    compact_base = {
        "source_artifact": "results/source.json",
        "candidate_kind": "code_repair",
        "task_id": "task",
        "downstream_failed": False,
        "schema_status": "valid",
        "syntax_status": "valid",
        "failure_category": "passed",
    }
    rows = [
        {"candidate_id": f"ok-{idx}", "first_step_proxy_failure_score": 0.9, **compact_base}
        for idx in range(4)
    ]
    assert len(exp._threshold_examples(rows, "first_step_proxy_failure_score", False)) == 3
    assert exp._threshold_examples([{**compact_base, "candidate_id": "none"}], "missing", False) == []

    assert exp._has_logprobs({"token_logprobs": [-0.1]}) is True
    assert exp._has_logprobs({"first_token_logprob": -0.2}) is True
    assert exp._raw_text(tmp_path, "") == ""
    assert exp._raw_text(tmp_path, "missing.txt") == ""
    assert exp._finite_float(True) is None
    assert exp._bounded01(-0.5) == 0.0
    assert exp._bounded01(2.0) == 1.0
    assert exp._repetition_ratio([]) == 1.0
