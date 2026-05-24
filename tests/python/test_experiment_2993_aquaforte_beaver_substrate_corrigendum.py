"""Tests for Exp 2993 AquaForte/BEAVER substrate corrigendum.

Spec: REQ-VERIFY-2993, SCENARIO-VERIFY-2993.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import aquaforte_beaver_reformulation_pipeline as exp2934
from carnot.eval import aquaforte_beaver_substrate_corrigendum as exp
from carnot.eval import constraintbench_constrained_output_rerun as exp2926
from carnot.eval import constraintbench_mini_direct_optimization as base


HEADLINE_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _clock(*values: float):
    ticks = iter(values)
    return lambda: next(ticks)


def _write_fixture(
    root: Path,
    *,
    headline_available: bool = True,
    known_issue: bool = True,
) -> tuple[exp.ExperimentConfig, base.OptimizationTask, Path]:
    task = exp2926.build_task_manifest()[0]
    raw_dir = root / "results" / "constraintbench_constrained_output_rerun_2926_raw"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / f"{task.task_id}__Gemma4-26B-A4B-it.json"
    raw_text = "not json"
    raw_payload = {
        "task_id": task.task_id,
        "raw_response": raw_text,
        "raw_response_sha256": exp2926.sha256_text(raw_text),
    }
    raw_path.write_text(json.dumps(raw_payload, indent=2, sort_keys=True), encoding="utf-8")
    exp2926_row = exp2926.evaluate_raw_output(
        task,
        raw_text,
        generation_metadata={
            "model_hf_id": HEADLINE_MODEL,
            "model_name": "Gemma4-26B-A4B-it",
            "model_path": "/tmp/gemma.gguf",
            "gpu_index": 0,
            "generation_source": "live_sota_llamacpp_prompt_schema",
            "raw_response_path": str(raw_path),
            "raw_response_sha256": exp2926.sha256_text(raw_text),
            "elapsed_seconds": 0.25,
            "blocker": None,
        },
    )
    exp2926_path = root / "results" / exp.EXP2926_FILENAME
    exp2926_path.write_text(
        json.dumps(
            {
                "constraintbench_corrigendum_ready": True,
                "model_specs": [
                    {
                        "name": "Gemma4-26B-A4B-it",
                        "hf_id": HEADLINE_MODEL,
                        "model_path": "/tmp/gemma.gguf",
                        "gpu": 0,
                    }
                ],
                "models_used": [HEADLINE_MODEL],
                "per_task_results": [exp2926_row],
                "raw_response_dir": str(raw_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    retry_prompt = exp2934.build_retry_prompt(task, exp2926_row)
    exp2934_path = root / "results" / exp.EXP2934_FILENAME
    exp2934_path.write_text(
        json.dumps(
            {
                "artifact": "experiment_2934_aquaforte_beaver_reformulation_pipeline_v1",
                "duration_s": 0.046,
                "honest_verdict": "complete: exp2926 live GGUF proposals reformulated and exact-verified",
                "inference_substrate": "live_llm_inference_plus_exact_verifier",
                "selected_task_ids": [task.task_id],
                "per_task_results": [
                    {
                        "task_id": task.task_id,
                        "initial_proposal_text": raw_text,
                        "raw_response_path": str(raw_path),
                        "exact_verifier_type": task.exact_verifier_type,
                        "retry": {
                            "attempted": True,
                            "cheap": True,
                            "prompt": retry_prompt,
                        },
                    }
                ],
                "corrigendum_pending": ["DURATION_TOO_SHORT"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    model_path = root / "models" / "gemma.gguf"
    if headline_available:
        model_path.parent.mkdir(parents=True)
        model_path.write_text("gguf fixture", encoding="utf-8")
    exp2989_path = root / "results" / exp.EXP2989_FILENAME
    exp2989_path.write_text(
        json.dumps(
            {
                "sota_headline_ready": headline_available,
                "sota_models_available": (
                    [{"hf_id": HEADLINE_MODEL, "path": str(model_path), "status": "cache_resolved"}]
                    if headline_available
                    else []
                ),
                "precondition_evidence": {
                    "torch_cuda": {"cuda_available": headline_available},
                    "llama_cpp": {
                        "llama_cpp_import_ok": headline_available,
                        "llama_cpp_supports_gpu_offload": headline_available,
                    },
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    known_path = root / "ops" / "known-issues.md"
    if known_issue:
        known_path.parent.mkdir(parents=True)
        known_path.write_text(
            "exp2934 AquaForte/BEAVER Reformulation Pipeline\n"
            "duration_s = 0.046s\n"
            "DURATION_TOO_SHORT\n"
            "mandatory substrate-corrigendum issue\n",
            encoding="utf-8",
        )

    config = exp.ExperimentConfig(
        output_path=root / "results" / exp.OUTPUT_FILENAME,
        exp2926_path=exp2926_path,
        exp2934_path=exp2934_path,
        exp2989_path=exp2989_path,
        known_issues_path=known_path,
        raw_transcript_dir=root / "results" / "raw" / exp.ARTIFACT_NAME,
        selected_count=1,
        started_at=10.0,
        clock=_clock(14.0),
        monotonic=_clock(100.0, 100.25),
        tests_run=("REQ-VERIFY-2993 focused pytest",),
    )
    return config, task, model_path


def test_req_verify_2993_spec_anchor_exists() -> None:
    """REQ-VERIFY-2993: the substrate corrigendum is spec-anchored first."""

    spec = Path("openspec/capabilities/verifiable-reasoning/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-2993" in spec
    assert "SCENARIO-VERIFY-2993" in spec
    assert exp.OUTPUT_FILENAME in spec


def test_scenario_verify_2993_measures_live_and_fallback_separately(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2993: live retry and enumerator fallback get separate labels."""

    config, task, model_path = _write_fixture(tmp_path)
    calls: list[exp.LiveRetryRequest] = []

    def live_runner(request: exp.LiveRetryRequest) -> dict[str, Any]:
        calls.append(request)
        return {
            "attempted": True,
            "truly_live": True,
            "hf_id": request.model["hf_id"],
            "model_path": request.model["path"],
            "prompt": request.prompt,
            "response_text": base.compliant_answer_for_task(request.task),
            "tokens_generated": 9,
            "duration_seconds": 2.5,
            "inference_substrate": "llama_cpp_gpu",
            "load_status": "loaded",
            "generation_status": "generated",
        }

    artifact = exp.run_experiment(config, live_retry_runner=live_runner)

    assert json.loads(config.output_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert len(calls) == 1
    assert calls[0].task == task
    assert calls[0].model["path"] == str(model_path)
    assert calls[0].prompt.startswith("Exclude the rejected candidate")
    assert artifact["substrate_corrigendum_complete"] is True
    assert artifact["live_llm_retry_measured"] is True
    assert artifact["enumerator_only_fallback_measured"] is True
    assert artifact["substrate_labels_corrected"] is True
    assert artifact["no_impossible_duration_claims"] is True
    assert artifact["live_retry_duration_seconds"] == pytest.approx(2.5)
    assert artifact["fallback_duration_seconds"] == pytest.approx(0.25)
    assert artifact["inference_substrate"] == (
        "live_llm_inference_plus_exact_verifier_and_enumerator_fallback"
    )
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["corrected_claim_labels"]["exp2934_retry_substrate"] == (
        "enumerator_only_fallback_plus_exact_verifier"
    )
    live = artifact["verifier_results_by_condition"]["live_llm_retry"]
    fallback = artifact["verifier_results_by_condition"]["enumerator_only_fallback"]
    assert live["measured"] is True
    assert live["pass_rate"] == pytest.approx(1.0)
    assert live["per_task_results"][0]["verifier"]["accepted"] is True
    assert fallback["measured"] is True
    assert fallback["substrate_label"] == "enumerator_only_fallback_plus_exact_verifier"
    assert fallback["per_task_results"][0]["transcript"]["llm_disabled"] is True
    assert Path(live["per_task_results"][0]["transcript_path"]).is_file()


def test_req_verify_2993_blocks_live_without_promoting_fallback(tmp_path: Path) -> None:
    """REQ-VERIFY-2993-2: unavailable headline GGUF blocks only the live condition."""

    config, _task, _model_path = _write_fixture(tmp_path, headline_available=False)

    artifact = exp.run_experiment(
        config,
        live_retry_runner=lambda request: pytest.fail(f"unexpected live call: {request}"),
    )

    assert artifact["substrate_corrigendum_complete"] is True
    assert artifact["live_llm_retry_measured"] is False
    assert artifact["enumerator_only_fallback_measured"] is True
    assert artifact["substrate_labels_corrected"] is True
    assert artifact["no_impossible_duration_claims"] is True
    assert artifact["inference_substrate"] == "enumerator_only_fallback_with_live_retry_blocked"
    assert artifact["honest_verdict"].startswith("complete:")
    live = artifact["verifier_results_by_condition"]["live_llm_retry"]
    assert live["measured"] is False
    assert "no mandated headline model" in live["blocked_reason"]
    assert live["substrate_label"] == "blocked_live_llm_retry"


def test_req_verify_2993_missing_preconditions_write_blocked_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-2993-1: missing issue evidence blocks without fake measurements."""

    config, _task, _model_path = _write_fixture(tmp_path, known_issue=False)

    artifact = exp.run_experiment(
        config,
        live_retry_runner=lambda request: pytest.fail(f"unexpected live call: {request}"),
    )

    assert json.loads(config.output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["substrate_corrigendum_complete"] is False
    assert artifact["live_llm_retry_measured"] is False
    assert artifact["enumerator_only_fallback_measured"] is False
    assert artifact["substrate_labels_corrected"] is False
    assert artifact["no_impossible_duration_claims"] is True
    assert artifact["honest_verdict"].startswith("blocked_preconditions:")
    assert artifact["preconditions"]["known_issue_confirmed"]["ok"] is False


def test_req_verify_2993_rejects_impossible_live_duration(tmp_path: Path) -> None:
    """REQ-VERIFY-2993-4: sub-second live retry evidence cannot close the corrigendum."""

    config, _task, _model_path = _write_fixture(tmp_path)

    artifact = exp.run_experiment(
        config,
        live_retry_runner=lambda request: {
            "attempted": True,
            "truly_live": True,
            "hf_id": request.model["hf_id"],
            "model_path": request.model["path"],
            "prompt": request.prompt,
            "response_text": base.compliant_answer_for_task(request.task),
            "tokens_generated": 9,
            "duration_seconds": 0.2,
            "inference_substrate": "llama_cpp_gpu",
            "load_status": "loaded",
            "generation_status": "generated",
        },
    )

    assert artifact["substrate_corrigendum_complete"] is False
    assert artifact["live_llm_retry_measured"] is True
    assert artifact["enumerator_only_fallback_measured"] is True
    assert artifact["substrate_labels_corrected"] is True
    assert artifact["no_impossible_duration_claims"] is False
    assert artifact["honest_verdict"].startswith("blocked_impossible_duration:")


def test_req_verify_2993_helper_edges_are_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-2993: helper edge cases stay explicit and deterministic."""

    assert exp._load_json(tmp_path / "missing.json") == {}
    assert exp._summarize("abcdef", limit=3).endswith("<truncated>")
    assert (
        exp._honest_verdict(
            complete=False,
            live_measured=False,
            no_impossible_duration=True,
        )
        == "blocked: substrate corrigendum incomplete"
    )
    assert exp._first_blocker({"ok1": {"ok": True}, "ok2": {"ok": True}}) == (
        "unknown_precondition"
    )
    assert exp._selected_python()

    skipped = {
        "per_task_results": [
            "not a dict",
            {"task_id": "unknown", "retry": {"attempted": True}},
        ]
    }
    with pytest.raises(ValueError, match="at least one"):
        exp._select_task_inputs(skipped, selected_count=1)
