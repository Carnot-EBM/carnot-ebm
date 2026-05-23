"""Tests for Exp 2917 spilled-energy logit detector micro-panel.

Spec: REQ-INFER-SOTA-018,
      SCENARIO-INFER-SOTA-018-001,
      SCENARIO-INFER-SOTA-018-002
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import spilled_energy_logit_detector_micro_panel_v1 as exp


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _model_specs(tmp_path: Path) -> list[dict[str, Any]]:
    qwen_path = tmp_path / "models" / "qwen.gguf"
    gemma_path = tmp_path / "models" / "gemma.gguf"
    qwen_path.parent.mkdir(parents=True, exist_ok=True)
    qwen_path.write_bytes(b"qwen")
    gemma_path.write_bytes(b"gemma")
    return [
        {"name": "Qwen3.6-35B-A3B", "hf_id": QWEN, "gpu": 0, "model_path": str(qwen_path)},
        {"name": "Gemma4-26B-A4B-it", "hf_id": GEMMA26, "gpu": 1, "model_path": str(gemma_path)},
    ]


def _stage_repo(tmp_path: Path) -> None:
    candidates: list[dict[str, Any]] = []
    raw_dir = tmp_path / "results" / "raw" / "experiment_2910_sota_code_generation_corrigendum_v2"
    for index in range(12):
        passed = index % 2 == 0
        raw_response = (
            "def solve(x):\n    return x + 1\n"
            if passed
            else "def solve(x):\n    return x - 1\n"
        )
        raw_path = raw_dir / f"candidate_{index}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_response, encoding="utf-8")
        candidates.append(
            {
                "candidate_index": index,
                "corpus": "MBPP",
                "model_hf_id": GEMMA26,
                "model_path": "models/gemma.gguf",
                "passed": passed,
                "prompt_sha256": f"code-prompt-{index}",
                "raw_response": raw_response,
                "raw_response_path": str(raw_path.relative_to(tmp_path)),
                "stable_id": f"mbpp-{index}",
                "tokens_generated": 17,
            }
        )
    _write_json(
        tmp_path / "results" / "experiment_2910_sota_code_generation_corrigendum_v2.json",
        {
            "honest_verdict": "complete: fixture",
            "candidate_results": candidates,
            "model_specs": _model_specs(tmp_path),
            "models_used": [GEMMA26],
        },
    )
    _write_jsonl(
        tmp_path / "data" / "eval_manifests" / "halueval_20260522.jsonl",
        [
            {
                "candidate": f"Factual candidate {index}",
                "dataset": "HaluEval",
                "label": index % 2,
                "prompt": f"Context and question {index}",
                "reference": f"Reference {index}",
                "stable_id": f"halueval-{index}",
            }
            for index in range(12)
        ],
    )


def _config(tmp_path: Path, output_name: str = exp.OUTPUT_FILENAME) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / output_name,
        run_date="20260523",
        random_seed=2917,
        target_examples=24,
        max_tokens=8,
        tests_run=("pytest tests/python/test_experiment_2917_spilled_energy_logit_detector_micro_panel.py",),
        started_at=10.0,
        clock=lambda: 15.25,
    )


def _runner_with_logprobs(
    *,
    prompt: str,
    example: exp.PanelExample,
    model_spec: dict[str, Any],
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    del prompt, seed
    assert max_tokens == 8
    is_risky = example.verification_label == "hallucination_like"
    response = "HALLUCINATION-LIKE" if is_risky else "VERIFIED"
    if is_risky:
        token_logprobs = [math.log(0.55), math.log(0.52)]
        top_logprobs = [{" HALLUCINATION": math.log(0.52), " VERIFIED": math.log(0.48)}]
    else:
        token_logprobs = [math.log(0.98), math.log(0.97)]
        top_logprobs = [{" VERIFIED": math.log(0.97), " HALLUCINATION": math.log(0.03)}]
    return {
        "raw_response": response,
        "token_count": len(token_logprobs),
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "model_id": model_spec["hf_id"],
    }


def test_scenario_infer_sota_018_ready_artifact_has_required_diagnostic_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-018-001: logprob rows produce diagnostic-only metrics."""
    _stage_repo(tmp_path)

    artifact = exp.write_experiment_artifact(
        _config(tmp_path),
        inference_runner=_runner_with_logprobs,
        cached_pair_provider=lambda gpu_indices=(0, 1): _model_specs(tmp_path),
        mandated_model_resolver=lambda: [],
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == "complete: spilled_energy_micro_panel_diagnostic_ready"
    assert artifact["spilled_energy_micro_panel_ready"] is True
    assert artifact["benchmark_claim_made"] is False
    assert artifact["claim_boundary"] == "diagnostic_only_no_benchmark_claim"
    assert artifact["cached_sota_pair_used"] is True
    assert artifact["models_used"] == [QWEN]
    assert artifact["random_seed"] == 2917
    assert artifact["run_date"] == "20260523"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["duration_s"] == pytest.approx(5.25)
    assert len(artifact["examples"]) == 24
    assert {row["source_type"] for row in artifact["examples"]} == {"code", "factuality"}
    assert {row["verification_label"] for row in artifact["examples"]} == {
        "hallucination_like",
        "verified",
    }
    assert all(row["raw_response"] for row in artifact["examples"])
    assert all(row["prompt_hash"] for row in artifact["examples"])
    assert all(row["logprob_or_logits_available"] for row in artifact["examples"])
    assert all(row["model_id"] == QWEN for row in artifact["examples"])
    assert artifact["spilled_energy_features"]["detector_trained"] is False
    assert artifact["separation_summary"]["detector_trained"] is False
    assert artifact["separation_summary"]["n_examples"] == 24
    assert (
        artifact["separation_summary"]["features"]["final_token_spilled_energy"]["auroc"]
        == pytest.approx(1.0)
    )


def test_scenario_infer_sota_018_blocks_when_runtime_omits_logprobs_and_logits(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFER-SOTA-018-002: text-only runtime exits without fake metrics."""
    _stage_repo(tmp_path)

    def text_only_runner(
        *,
        prompt: str,
        example: exp.PanelExample,
        model_spec: dict[str, Any],
        seed: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        del prompt, example, seed, max_tokens
        return {"raw_response": "VERIFIED", "token_count": 1, "model_id": model_spec["hf_id"]}

    artifact = exp.write_experiment_artifact(
        _config(tmp_path, "blocked.json"),
        inference_runner=text_only_runner,
        cached_pair_provider=lambda gpu_indices=(0, 1): _model_specs(tmp_path),
        mandated_model_resolver=lambda: [],
    )

    assert artifact["honest_verdict"] == "complete: blocked_logprob_runtime_unavailable"
    assert artifact["blocked_reason"] == "blocked_logprob_runtime_unavailable"
    assert artifact["spilled_energy_micro_panel_ready"] is False
    assert artifact["logprob_or_logits_available"] is False
    assert artifact["examples"] == []
    assert artifact["spilled_energy_features"]["available"] is False
    assert artifact["separation_summary"]["available"] is False
    assert artifact["benchmark_claim_made"] is False


def test_req_infer_sota_018_tries_cached_pair_before_single_model_fallback(
    tmp_path: Path,
) -> None:
    """REQ-INFER-SOTA-018: cached_sota_pair is attempted before fallback resolution."""
    _stage_repo(tmp_path)
    calls: list[str] = []

    def cached_pair_provider(gpu_indices: tuple[int, int] = (0, 1)) -> None:
        calls.append(f"cached:{gpu_indices}")
        return None

    def resolver() -> list[dict[str, Any]]:
        calls.append("resolver")
        return [_model_specs(tmp_path)[1]]

    artifact = exp.write_experiment_artifact(
        _config(tmp_path, "fallback.json"),
        inference_runner=_runner_with_logprobs,
        cached_pair_provider=cached_pair_provider,
        mandated_model_resolver=resolver,
    )

    assert calls == ["cached:(0, 1)", "resolver"]
    assert artifact["cached_sota_pair_used"] is False
    assert artifact["models_used"] == [GEMMA26]
    assert artifact["model_specs"][0]["hf_id"] == GEMMA26
    assert artifact["spilled_energy_micro_panel_ready"] is True


def test_req_infer_sota_018_feature_helpers_compute_final_token_energy() -> None:
    """REQ-INFER-SOTA-018: final-token top-k logprobs drive spilled/marginal scores."""
    features = exp.compute_energy_features(
        {
            "token_logprobs": [math.log(0.8), math.log(0.6)],
            "top_logprobs": [{" A": math.log(0.25), " B": math.log(0.75)}],
        }
    )

    assert features["logprob_or_logits_available"] is True
    assert features["token_count"] == 2
    assert features["sequence_spilled_energy"] == pytest.approx(0.3)
    assert features["sequence_marginal_energy"] == pytest.approx(-math.log(0.8 * 0.6) / 2)
    assert features["final_token_spilled_energy"] == pytest.approx(0.25)
    assert features["final_token_marginal_energy"] == pytest.approx(-math.log(0.25))


def test_req_infer_sota_018_defensive_paths_remain_non_claiming(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-018: missing inputs and alternate telemetry shapes stay diagnostic."""
    no_model_artifact = exp.write_experiment_artifact(
        _config(tmp_path, "no-model.json"),
        inference_runner=_runner_with_logprobs,
        cached_pair_provider=lambda gpu_indices=(0, 1): None,
        mandated_model_resolver=lambda: [],
    )
    assert no_model_artifact["blocked_reason"] == "blocked_no_mandated_sota_gguf_cached"
    assert no_model_artifact["benchmark_claim_made"] is False

    model_only_artifact = exp.write_experiment_artifact(
        _config(tmp_path, "no-panel.json"),
        inference_runner=_runner_with_logprobs,
        cached_pair_provider=lambda gpu_indices=(0, 1): _model_specs(tmp_path),
        mandated_model_resolver=lambda: [],
    )
    assert model_only_artifact["blocked_reason"] == "blocked_insufficient_micro_panel_examples"
    assert model_only_artifact["examples"] == []

    token_only = exp.compute_energy_features({"token_logprobs": [math.log(0.4)]})
    assert token_only["final_token_top1_probability"] == pytest.approx(0.4)
    assert token_only["token_count"] == 1

    nested_logits = exp.compute_energy_features({"final_logits": [[0.0, 1.0]], "tokens": ["a", "b"]})
    assert nested_logits["logprob_or_logits_available"] is True
    assert nested_logits["token_count"] == 2

    flat_logits = exp.compute_energy_features({"logits": [0.0, 0.0]})
    assert flat_logits["final_token_spilled_energy"] == pytest.approx(0.5)

    invalid_values = exp.compute_energy_features({"token_logprobs": [True, "not-a-number"]})
    assert invalid_values["logprob_or_logits_available"] is False
    assert exp._softmax_log_values([]) == []

    one_label = exp.build_separation_summary(
        [
            {
                "verification_label": "verified",
                "verification_label_int": 0,
                "final_token_spilled_energy": 0.1,
                "final_token_marginal_energy": 0.2,
                "sequence_spilled_energy": 0.3,
                "sequence_marginal_energy": 0.4,
            }
        ]
    )
    assert one_label["features"]["final_token_spilled_energy"]["auroc"] is None

    assert exp._read_optional_text(tmp_path, None) == ""
    assert exp._read_optional_text(tmp_path, "missing.txt") == ""
