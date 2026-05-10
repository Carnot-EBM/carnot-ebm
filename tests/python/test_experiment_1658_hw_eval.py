"""Tests for Exp 1658 CPU vs KV260 EBRM trace-scoring evaluation.

Spec: REQ-VERIFY-1658, SCENARIO-VERIFY-1658.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import experiment_1658_hw_eval as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA = "unsloth/gemma-4-31B-it-GGUF"


def _sota_row(
    case_id: str,
    *,
    correct: bool,
    format_valid: bool = True,
    hf_id: str = QWEN,
    expected_answer: str = "1",
) -> dict[str, Any]:
    response = expected_answer if correct else "0"
    return {
        "case_id": case_id,
        "correct": correct,
        "format_valid": format_valid,
        "expected_answer": expected_answer,
        "response_text": f"<think></think>\n{response}",
        "response_text_available": True,
        "generation_source": "live_sota_llamacpp",
        "hf_id": hf_id,
        "model_name": hf_id.split("/", 1)[1].removesuffix("-GGUF"),
        "prompt": f"Case {case_id}: return {expected_answer}.",
        "token_logprobs": [-0.01, -0.02, -0.03],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_gate_artifacts(tmp_path: Path, *, hardware: bool = False) -> tuple[Path, Path]:
    ebrm_path = tmp_path / "experiment_1656_ebrm_trace_scorer.json"
    kv260_path = tmp_path / "experiment_1657_kv260_ebrm_binding.json"
    ebrm_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "experiment_id": 1656,
                "ebrm_trace_scorer_ready": True,
            }
        ),
        encoding="utf-8",
    )
    kv260_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "experiment_id": 1657,
                "kv260_ebrm_binding_ready": True,
                "hardware_execution_available": hardware,
                "software_fallback_used": not hardware,
                "potts_q_states": 3,
            }
        ),
        encoding="utf-8",
    )
    return ebrm_path, kv260_path


def test_req_verify_1658_converts_live_sota_rows_to_logical_traces(tmp_path: Path) -> None:
    """REQ-VERIFY-1658: live SOTA rows become CPU/KV260-compatible traces."""

    manifest = tmp_path / "sota.jsonl"
    _write_jsonl(
        manifest,
        [
            _sota_row("correct", correct=True, hf_id=QWEN),
            _sota_row("wrong", correct=False, hf_id=GEMMA),
            _sota_row("bad-format", correct=True, format_valid=False, hf_id=QWEN),
        ],
    )

    cases = mod.load_sota_trace_cases(manifest, max_cases=3)

    assert [case.case_id for case in cases] == ["correct", "wrong", "bad-format"]
    assert cases[0].trace.expected_inconsistent is False
    assert cases[1].trace.expected_inconsistent is True
    assert cases[1].trace.steps[1].contradicts == ("expected",)
    assert cases[2].trace.steps[-1].supports == ("missing-format-proof",)
    assert {case.model_id for case in cases} == {QWEN, GEMMA}

    coherent_only = tmp_path / "coherent-only.jsonl"
    _write_jsonl(
        coherent_only,
        [
            _sota_row("coherent-1", correct=True, hf_id=QWEN),
            _sota_row("coherent-2", correct=True, hf_id=QWEN),
            _sota_row("coherent-3", correct=True, hf_id=QWEN),
        ],
    )
    assert [case.case_id for case in mod.load_sota_trace_cases(coherent_only, max_cases=4)] == [
        "coherent-1",
        "coherent-2",
        "coherent-3",
    ]


def test_scenario_verify_1658_writes_latency_and_delta_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1658: CPU and KV260 scores match with latency evidence."""

    manifest = tmp_path / "sota.jsonl"
    _write_jsonl(
        manifest,
        [
            _sota_row("correct-qwen", correct=True, hf_id=QWEN),
            _sota_row("correct-gemma", correct=True, hf_id=GEMMA),
            _sota_row("wrong-qwen", correct=False, hf_id=QWEN),
            _sota_row("format-gemma", correct=True, format_valid=False, hf_id=GEMMA),
        ],
    )
    ebrm_path, kv260_path = _write_gate_artifacts(tmp_path)
    output_path = tmp_path / "experiment_1658_hw_eval.json"
    ticks = iter([1.0, 1.012, 2.0, 2.007])

    artifact = mod.run_experiment(
        output_path=output_path,
        sota_manifest_path=manifest,
        ebrm_artifact_path=ebrm_path,
        kv260_artifact_path=kv260_path,
        max_cases=4,
        tests_run=["tests/python/test_experiment_1658_hw_eval.py"],
        timer=lambda: next(ticks),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1658
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["hardware_execution_available"] is False
    assert artifact["software_fallback_used"] is True
    assert artifact["cpu_latency_ms"] == pytest.approx(12.0)
    assert artifact["kv260_latency_ms"] == pytest.approx(7.0)
    assert artifact["kv260_speedup_vs_cpu"] == pytest.approx(1.714286)
    assert artifact["max_score_delta"] == pytest.approx(0.0)
    assert artifact["mean_abs_score_delta"] == pytest.approx(0.0)
    assert artifact["scoring_delta_within_tolerance"] is True
    assert artifact["cpu_score_accuracy"] == artifact["kv260_score_accuracy"] == 1.0
    assert artifact["spec_traces"] == ["REQ-VERIFY-1658", "SCENARIO-VERIFY-1658"]
    assert len(artifact["case_scores"]) == 4
    assert {row["kv260_backend"] for row in artifact["case_scores"]} == {
        "software-kv260-potts"
    }
    assert all(row["potts_q_states"] == 3 for row in artifact["case_scores"])


def test_req_verify_1658_blocks_without_completed_gates(tmp_path: Path) -> None:
    """REQ-VERIFY-1658: incomplete upstream artifacts prevent complete results."""

    manifest = tmp_path / "sota.jsonl"
    _write_jsonl(manifest, [_sota_row("correct", correct=True)])
    ebrm_path = tmp_path / "experiment_1656_ebrm_trace_scorer.json"
    ebrm_path.write_text(
        json.dumps({"status": "blocked", "experiment_id": 1656}),
        encoding="utf-8",
    )

    artifact = mod.run_experiment(
        output_path=tmp_path / "blocked.json",
        sota_manifest_path=manifest,
        ebrm_artifact_path=ebrm_path,
        kv260_artifact_path=tmp_path / "missing_1657.json",
        max_cases=1,
    )

    assert artifact["status"] == "blocked"
    assert artifact["cases_total"] == 0
    assert artifact["blockers"]
    assert "Exp 1656" in artifact["blockers"][0]
    mod.validate_artifact(artifact)

    with pytest.raises(AssertionError, match="spec_traces"):
        mod.validate_artifact({**artifact, "spec_traces": []})

    invalid_complete = {
        **artifact,
        "status": "complete",
        "cases_total": 1,
        "blockers": [],
        "scoring_delta_within_tolerance": False,
    }
    with pytest.raises(AssertionError, match="delta tolerance"):
        mod.validate_artifact(invalid_complete)

    assert mod._speedup(1.0, 0.0) is None
