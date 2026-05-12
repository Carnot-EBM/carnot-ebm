"""Tests for Exp 1880 live/cache SOTA ROCE validator evaluation.

Spec: REQ-VERIFY-1880, SCENARIO-VERIFY-1880.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.verify import sota_roce_validator_eval as mod


def _resolved_specs() -> list[dict[str, Any]]:
    return [
        {**spec, "model_path": f"/cache/{index}.gguf"}
        for index, spec in enumerate(mod.MODEL_SPECS)
    ]


def _good_generator(model: Mapping[str, Any], case: Mapping[str, Any]) -> str:
    assert model["hf_id"] in mod.MANDATED_HF_IDS
    return str(case["known_good"])


def test_req_verify_1880_model_specs_and_prompt_suite_are_mandated() -> None:
    """REQ-VERIFY-1880: MODEL_SPECS and prompt suite cover the mandated scope."""

    assert [spec["hf_id"] for spec in mod.MODEL_SPECS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]

    cases = mod.default_prompt_cases()
    coverage = mod.summarize_prompt_coverage(cases)

    assert len(cases) >= 30
    assert coverage["prompt_count"] == len(cases)
    assert coverage["constraint_coverage_rate"] == pytest.approx(1.0)
    assert {"format", "arithmetic", "lexical", "conditional"} <= set(
        coverage["constraint_families"]
    )


def test_scenario_verify_1880_injected_outputs_are_gated_and_bounded(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1880: SOTA-shaped outputs are gated by ROCE validators."""

    output_path = tmp_path / "experiment_1880_sota_roce_validator_eval.json"
    tests_run = [".venv/bin/pytest tests/python/test_sota_roce_validator_eval.py -q"]

    artifact = mod.run_experiment(
        output_path=output_path,
        model_specs=_resolved_specs(),
        generator_fn=_good_generator,
        tests_run=tests_run,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["sota_roce_eval_ready"] is True
    assert artifact["inference_mode"] == "injected"
    assert artifact["models_used"] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert artifact["zero_false_accepts"] is True
    assert artifact["false_accept_count"] == 0
    assert artifact["constraint_coverage_rate"] == pytest.approx(1.0)
    assert artifact["tests_run"] == tests_run
    assert artifact["prompt_count"] == 30
    assert artifact["output_rows"] == 90
    assert artifact["case_results"][0]["beaver_lite_bounds"]["beaver_lite_bounds_ready"] is True
    assert artifact["generation_rows"][0]["validation"]["accepted"] is True
    assert artifact["generation_rows"][0]["provenance"]["mode"] == "injected"


def test_req_verify_1880_blocks_without_all_mandated_ggufs(tmp_path: Path) -> None:
    """REQ-VERIFY-1880: unavailable mandated GGUFs block headline accuracy."""

    partial_specs = [
        {**mod.MODEL_SPECS[0], "model_path": "/cache/qwen.gguf"},
        dict(mod.MODEL_SPECS[1]),
        dict(mod.MODEL_SPECS[2]),
    ]

    artifact = mod.run_experiment(
        output_path=tmp_path / "blocked.json",
        model_specs=partial_specs,
        generator_fn=_good_generator,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["sota_roce_eval_ready"] is False
    assert artifact["inference_mode"] == "blocked_missing_mandated_gguf"
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert artifact["zero_false_accepts"] is False
    assert artifact["headline_accuracy_claimed"] is False
    assert "unavailable" in artifact["honest_verdict"]
    assert artifact["missing_models"] == [
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]


def test_req_verify_1880_resolver_uses_cached_pair_then_direct_lookup() -> None:
    """REQ-VERIFY-1880: cached_sota_pair is used before direct GGUF resolution."""

    pair_calls: list[tuple[int, int]] = []
    resolver_calls: list[str] = []

    def cached_pair_fn(**kwargs: Any) -> list[dict[str, Any]] | None:
        model_indices = tuple(kwargs["model_indices"])
        pair_calls.append(model_indices)
        if model_indices == (0, 2):
            return [
                {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "model_path": "/pair/qwen.gguf",
                },
                {
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "model_path": "/pair/gemma31.gguf",
                },
            ]
        return None

    def resolver_fn(hf_id: str, preferred_quant: str) -> str | None:
        resolver_calls.append(f"{hf_id}:{preferred_quant}")
        if hf_id == "unsloth/gemma-4-26B-A4B-it-GGUF":
            return "/direct/gemma26.gguf"
        return None

    resolved = mod.resolve_mandated_model_specs(
        cached_pair_fn=cached_pair_fn,
        resolver_fn=resolver_fn,
    )

    assert pair_calls == [(0, 2), (0, 1)]
    assert [spec["model_path"] for spec in resolved] == [
        "/pair/qwen.gguf",
        "/pair/gemma31.gguf",
        "/direct/gemma26.gguf",
    ]
    assert resolver_calls == ["unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M"]


def test_req_verify_1880_partial_when_adversarial_control_is_accepted(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1880: false accepts prevent complete headline artifacts."""

    case = mod.default_prompt_cases()[0]
    bad_case = dict(case, known_bad=[case["known_good"]])

    artifact = mod.run_experiment(
        output_path=tmp_path / "partial.json",
        model_specs=_resolved_specs(),
        generator_fn=_good_generator,
        prompt_cases=[bad_case],
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "partial"
    assert artifact["sota_roce_eval_ready"] is False
    assert artifact["zero_false_accepts"] is False
    assert artifact["false_accept_count"] == 1
    assert artifact["case_results"][0]["false_accept_count"] == 1
    assert artifact["honest_verdict"].startswith("partial:")


def test_req_verify_1880_schema_validation_rejects_invalid_complete() -> None:
    """REQ-VERIFY-1880: complete artifacts require ready zero-false-accept runs."""

    artifact = mod.build_artifact(
        model_specs=_resolved_specs(),
        generation_rows=[],
        case_results=[],
        tests_run=[],
        inference_mode="injected",
    )

    with pytest.raises(AssertionError, match="complete requires ready"):
        mod.validate_artifact(dict(artifact, status="complete", sota_roce_eval_ready=False))
    with pytest.raises(AssertionError, match="complete requires zero false accepts"):
        mod.validate_artifact(
            dict(artifact, status="complete", sota_roce_eval_ready=True, zero_false_accepts=False)
        )
    with pytest.raises(AssertionError, match="coverage out of range"):
        mod.validate_artifact(dict(artifact, constraint_coverage_rate=1.2))
