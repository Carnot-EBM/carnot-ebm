"""Tests for Exp 3004 AquaForte/BEAVER live retry provenance repair.

Spec: REQ-VERIFY-3004, SCENARIO-VERIFY-3004.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import aquaforte_beaver_live_retry_provenance as exp
from carnot.eval import aquaforte_beaver_reformulation_pipeline as exp2934
from carnot.eval import constraintbench_constrained_output_rerun as exp2926
from carnot.eval import constraintbench_mini_direct_optimization as base


HEADLINE_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _clock(*values: float):
    ticks = iter(values)
    return lambda: next(ticks)


def _write_fixture(
    root: Path,
    *,
    headline_ready: bool = True,
) -> tuple[exp.ExperimentConfig, base.OptimizationTask, Path]:
    task = exp2926.build_task_manifest()[0]
    rejected = exp2926.evaluate_raw_output(
        task,
        "not json",
        generation_metadata={
            "model_hf_id": HEADLINE_MODEL,
            "model_name": "Gemma4-26B-A4B-it",
            "generation_source": "live_sota_llamacpp_prompt_schema",
            "raw_response_sha256": exp2926.sha256_text("not json"),
        },
    )
    retry_prompt = exp2934.build_retry_prompt(task, rejected)

    exp2934_path = root / "results" / exp.EXP2934_FILENAME
    exp2934_path.parent.mkdir(parents=True)
    exp2934_path.write_text(
        json.dumps(
            {
                "artifact": "experiment_2934_aquaforte_beaver_reformulation_pipeline_v1",
                "duration_s": 0.046,
                "inference_substrate": "live_llm_inference_plus_exact_verifier",
                "selected_task_ids": [task.task_id],
                "per_task_results": [
                    {
                        "task_id": task.task_id,
                        "initial_proposal_text": "not json",
                        "exact_verifier_type": task.exact_verifier_type,
                        "retry": {
                            "attempted": True,
                            "cheap": True,
                            "prompt": retry_prompt,
                            "retry_solution": base.compliant_answer_for_task(task),
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

    exp2993_path = root / "results" / exp.EXP2993_FILENAME
    exp2993_path.write_text(
        json.dumps(
            {
                "artifact": "experiment_2993_aquaforte_beaver_substrate_corrigendum_v1",
                "selected_task_ids": [task.task_id],
                "live_llm_retry_measured": True,
                "enumerator_only_fallback_measured": True,
                "verifier_results_by_condition": {
                    "live_llm_retry": {"substrate_label": "live_llm_inference_plus_exact_verifier"},
                    "enumerator_only_fallback": {
                        "substrate_label": "enumerator_only_fallback_plus_exact_verifier"
                    },
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    model_path = root / "models" / "gemma.gguf"
    if headline_ready:
        model_path.parent.mkdir(parents=True)
        model_path.write_text("gguf fixture", encoding="utf-8")

    exp3001_path = root / "results" / exp.EXP3001_FILENAME
    exp3001_path.write_text(
        json.dumps(
            {
                "artifact": "experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1",
                "sota_headline_ready": headline_ready,
                "preconditions_checked": True,
                "model_specs": {
                    "headline_models": list(exp.HEADLINE_MODEL_IDS),
                    "smoke_only_models": list(exp.SMOKE_ONLY_MODEL_IDS),
                },
                "sota_models_available": (
                    [{"hf_id": HEADLINE_MODEL, "path": str(model_path), "status": "cache_resolved"}]
                    if headline_ready
                    else []
                ),
                "model_checksums": {
                    HEADLINE_MODEL: {
                        "status": "available" if headline_ready else "missing",
                        "path": str(model_path) if headline_ready else None,
                        "bounded_sha256": "abcd1234",
                        "checksum_algorithm": "sha256_head_tail_1mib_plus_size_mtime",
                    }
                },
                "precondition_evidence": {
                    "torch_cuda": {"cuda_available": headline_ready},
                    "llama_cpp": {
                        "llama_cpp_import_ok": headline_ready,
                        "llama_cpp_supports_gpu_offload": headline_ready,
                    },
                    "gpu_inventory": {"available": headline_ready},
                    "checksum_feasibility": {"feasible": headline_ready},
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    config = exp.ExperimentConfig(
        output_path=root / "results" / exp.OUTPUT_FILENAME,
        exp2934_path=exp2934_path,
        exp2993_path=exp2993_path,
        exp3001_path=exp3001_path,
        raw_transcript_dir=root / "results" / "raw" / exp.ARTIFACT_NAME,
        duration_provenance_path=root / "results" / "raw" / exp.ARTIFACT_NAME / "duration.json",
        selected_count=1,
        selected_python="/repo/.venv/bin/python",
        monotonic=_clock(100.0, 103.0, 200.0, 200.25),
        tests_run=("REQ-VERIFY-3004 focused pytest",),
    )
    return config, task, model_path


def test_req_verify_3004_spec_anchor_exists() -> None:
    """REQ-VERIFY-3004: the provenance repair is spec-anchored first."""

    spec = Path("openspec/capabilities/verifiable-reasoning/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-3004" in spec
    assert "SCENARIO-VERIFY-3004" in spec
    assert exp.OUTPUT_FILENAME in spec


def test_scenario_verify_3004_clean_live_provenance_promotes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3004: clean live retry evidence opens the promotion gate."""

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
    assert artifact["preconditions_checked"] is True
    assert artifact["live_retry_provenance_clean"] is True
    assert artifact["substrate_corrigendum_promotable"] is True
    assert artifact["headline_models_used"] == [HEADLINE_MODEL]
    assert artifact["model_checksums"][HEADLINE_MODEL]["bounded_sha256"] == "abcd1234"
    assert artifact["duration_seconds_live"] == pytest.approx(3.0)
    assert artifact["enumerator_fallback_separated"] is True
    assert artifact["impossible_duration_flag"] is False
    assert artifact["honest_verdict"].startswith("clean:")
    assert artifact["live_transcript_paths"]
    assert artifact["enumerator_fallback_paths"]
    assert set(artifact["live_transcript_paths"]).isdisjoint(artifact["enumerator_fallback_paths"])

    duration = json.loads(Path(artifact["duration_provenance_path"]).read_text(encoding="utf-8"))
    assert duration["live_started_monotonic"] == pytest.approx(100.0)
    assert duration["live_finished_monotonic"] == pytest.approx(103.0)
    assert duration["duration_seconds_live"] == pytest.approx(3.0)
    assert duration["transcript_write_timestamps"]

    live_transcript = json.loads(Path(artifact["live_transcript_paths"][0]).read_text(encoding="utf-8"))
    assert live_transcript["condition"] == "live_retry"
    assert live_transcript["substrate_label"] == "live_llm_inference_plus_exact_verifier"
    assert live_transcript["prompt"].startswith("Exclude the rejected candidate")
    assert live_transcript["raw_output_sha256"]
    assert live_transcript["verifier"]["accepted"] is True

    fallback_transcript = json.loads(
        Path(artifact["enumerator_fallback_paths"][0]).read_text(encoding="utf-8")
    )
    assert fallback_transcript["condition"] == "enumerator_fallback"
    assert fallback_transcript["llm_disabled"] is True
    assert fallback_transcript["verifier"]["accepted"] is True


def test_req_verify_3004_refuses_promotion_on_live_fallback_contamination(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3004-4: fallback-labeled live evidence cannot be promoted."""

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
            "duration_seconds": 2.5,
            "inference_substrate": "enumerator_only_fallback_plus_exact_verifier",
            "load_status": "loaded",
            "generation_status": "generated",
        },
    )

    assert artifact["live_retry_provenance_clean"] is False
    assert artifact["substrate_corrigendum_promotable"] is False
    assert artifact["enumerator_fallback_separated"] is True
    assert artifact["contamination_detected"] is True
    assert artifact["honest_verdict"].startswith("flagged:")


def test_req_verify_3004_blocks_failed_preconditions_without_live_call(tmp_path: Path) -> None:
    """REQ-VERIFY-3004-1: setup failure blocks without fabricated live evidence."""

    config, _task, _model_path = _write_fixture(tmp_path, headline_ready=False)

    artifact = exp.run_experiment(
        config,
        live_retry_runner=lambda request: pytest.fail(f"unexpected live call: {request}"),
    )

    assert json.loads(config.output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["preconditions_checked"] is True
    assert artifact["live_retry_provenance_clean"] is False
    assert artifact["substrate_corrigendum_promotable"] is False
    assert artifact["headline_models_used"] == []
    assert artifact["duration_seconds_live"] == 0.0
    assert artifact["live_transcript_paths"] == []
    assert artifact["enumerator_fallback_paths"] == []
    assert artifact["enumerator_fallback_separated"] is False
    assert artifact["impossible_duration_flag"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["preconditions"]["exp3001_sota_headline_ready"]["ok"] is False


def test_req_verify_3004_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3004: helper edge cases stay conservative and auditable."""

    assert exp._load_json(tmp_path / "missing.json") == {}
    assert exp._model_specs({}) == {
        "headline_models": list(exp.HEADLINE_MODEL_IDS),
        "smoke_only_models": list(exp.SMOKE_ONLY_MODEL_IDS),
        "preferred_quantization": "Q4_K_M",
    }
    assert exp._source_task_ids({}, {"selected_task_ids": ["from-exp2934"]}) == [
        "from-exp2934"
    ]
    assert exp._source_task_ids(
        {},
        {"per_task_results": [{"task_id": "from-row", "retry": {"attempted": True}}]},
    ) == ["from-row"]

    with pytest.raises(ValueError, match="at least one"):
        exp._reconstruct_retry_items(
            {"per_task_results": [{"task_id": "unknown", "retry": {"attempted": True}}]},
            {},
            selected_count=1,
        )

    assert exp._headline_models_used({"per_task_results": [{"truly_live": False}]}, False) == []
    assert exp._detect_contamination({"transcript_paths": ["same"]}, {"transcript_paths": ["same"]})
    assert exp._detect_contamination(
        {
            "transcript_paths": ["live"],
            "per_task_results": [
                {"runner_inference_substrate": "llama_cpp_gpu", "transcript_path": "live/fallback.json"}
            ],
        },
        {"transcript_paths": ["fallback"]},
    )
    assert exp._enumerator_fallback_separated([], []) is False
    assert exp._enumerator_fallback_separated(["same"], ["same"]) is False
    assert exp._impossible_duration_flag({"measured": False}, []) is False
    assert exp._impossible_duration_flag(
        {"measured": True, "live_started_monotonic": None, "live_finished_monotonic": 1.0},
        [],
    )
    assert exp._impossible_duration_flag(
        {
            "measured": True,
            "live_started_monotonic": 1.0,
            "live_finished_monotonic": 1.2,
            "duration_seconds": 0.2,
        },
        [],
    )
    assert exp._impossible_duration_flag(
        {
            "measured": True,
            "live_started_monotonic": 1.0,
            "live_finished_monotonic": 3.0,
            "duration_seconds": 2.0,
        },
        [str(tmp_path / "absent.json")],
    )
    assert exp._honest_verdict(
        preconditions={},
        provenance_clean=False,
        contamination=False,
        impossible_duration=True,
        live_measured=True,
    ).startswith("flagged: impossible")
    assert exp._honest_verdict(
        preconditions={},
        provenance_clean=False,
        contamination=False,
        impossible_duration=False,
        live_measured=False,
    ).startswith("flagged: live retry attempted")
    assert exp._honest_verdict(
        preconditions={},
        provenance_clean=False,
        contamination=False,
        impossible_duration=False,
        live_measured=True,
    ).startswith("flagged: live retry provenance incomplete")
    assert exp._path_timestamp_row(str(tmp_path / "absent.json")) == {
        "path": str(tmp_path / "absent.json"),
        "exists": False,
        "mtime_ns": None,
    }
