"""Tests for Exp5500 SOTA GGUF concept/claim evidence panel.

Spec refs: REQ-VERIFY-5500, SCENARIO-VERIFY-5500.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5500_sota_concept_claim_panel_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5500_sota_concept_claim_panel_v499.py")


class FakeRuntime:
    """Minimal llama.cpp-like runtime used by tests instead of live GGUF loading."""

    def __init__(self, spec: dict[str, Any], output_text: str) -> None:
        self.spec = spec
        self.output_text = output_text
        self.load_receipt = {
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "llama_cpp_binding": "llama_cpp.Llama",
            "n_gpu_layers": -1,
            "gpu_memory_delta_mb": 1536.0,
            "gpu_offload_verified": True,
            "load_wall_time_s": 0.25,
        }
        self.closed = False

    def generate(self, prompt: str) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "output_text": self.output_text,
            "wall_time_s": 0.5,
            "prompt_tokens": 101,
            "completion_tokens": 37,
            "total_tokens": 138,
            "llama_cpp_command_or_binding": "llama_cpp.Llama.create_completion",
        }

    def close(self) -> None:
        self.closed = True


class FakeNoOffloadRuntime:
    """Runtime fixture that loads but fails the GPU offload receipt gate."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.load_receipt = {
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "llama_cpp_binding": "llama_cpp.Llama",
            "n_gpu_layers": -1,
            "gpu_memory_delta_mb": 0.0,
            "gpu_offload_verified": False,
        }
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _touch_cached_models(tmp_path: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for index, hf_id in enumerate(mod.MANDATED_HEADLINE_MODEL_IDS):
        filename = hf_id.rsplit("/", 1)[-1].replace("-GGUF", "-UD-Q4_K_M.gguf")
        path = tmp_path / f"model_{index}" / filename
        path.parent.mkdir(parents=True)
        path.write_text("fake gguf", encoding="utf-8")
        paths[hf_id] = str(path)
    return paths


def _resolver(paths: dict[str, str]):
    return lambda hf_id, preferred_quant="Q4_K_M": paths.get(hf_id)


def _ready_probe() -> dict[str, Any]:
    return {
        "cuda_available": True,
        "cuda_device_count": 2,
        "llama_cpp_import_ok": True,
        "llama_cpp_cuda_available": True,
        "gpu_offload_supported": True,
        "runtime_ready": True,
        "blocked_reasons": [],
        "system_info": "CUDA : ARCHS = 860",
    }


def _blocked_probe() -> dict[str, Any]:
    return {
        "cuda_available": True,
        "cuda_device_count": 2,
        "llama_cpp_import_ok": True,
        "llama_cpp_cuda_available": False,
        "gpu_offload_supported": False,
        "runtime_ready": False,
        "blocked_reasons": ["llama_cpp_gpu_offload_unavailable"],
        "system_info": "CPU only",
    }


def _optimal_output() -> str:
    return json.dumps(
        {
            "instances": [
                {
                    "instance_id": "claim_support_preference",
                    "assignment": {
                        "support": "entailed",
                        "source_quality": "primary",
                        "scope": "bounded",
                    },
                    "explanation": "Hard support and scope pass; primary source wins the soft tie.",
                },
                {
                    "instance_id": "claim_safety_conflict",
                    "assignment": {
                        "safety": "safe",
                        "citation": "present",
                        "action": "accept",
                    },
                    "explanation": "Hard safety and citation pass; accepting is the best soft action.",
                },
                {
                    "instance_id": "claim_infeasible_negative_control",
                    "abstain": True,
                    "explanation": "The hard verdict constraints conflict, so no assignment is valid.",
                },
            ]
        }
    )


def test_req_verify_5500_spec_declares_panel_contract() -> None:
    """REQ-VERIFY-5500: OpenSpec anchors required model and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5500") : spec.index("### REQ-VERIFY-5462")
    ]

    assert "SCENARIO-VERIFY-5500" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.FIXTURE_ARTIFACT_RELATIVE_PATH) in section
    for hf_id in mod.MANDATED_HEADLINE_MODEL_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_verify_5500_model_specs_use_mandated_local_gguf_ids(tmp_path: Path) -> None:
    """REQ-VERIFY-5500: MODEL_SPECS contains only mandated headline GGUF ids."""

    paths = _touch_cached_models(tmp_path)
    specs = mod.resolve_model_specs(cache_resolver=_resolver(paths))

    assert [row["hf_id"] for row in mod.MODEL_SPECS] == list(mod.MANDATED_HEADLINE_MODEL_IDS)
    assert [row["hf_id"] for row in specs] == list(mod.MANDATED_HEADLINE_MODEL_IDS)
    assert all(row["local_model_present"] is True for row in specs)
    assert all(row["quant"] == "UD-Q4_K_M" for row in specs)
    assert all("AutoTokenizer" not in json.dumps(row) for row in specs)


def test_scenario_verify_5500_live_fake_runtime_scores_exact_validators(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5500: generated claim states are scored by Exp5499 validators."""

    paths = _touch_cached_models(tmp_path)
    runtime_holder: list[FakeRuntime] = []

    def factory(spec: dict[str, Any]) -> FakeRuntime:
        runtime = FakeRuntime(spec, _optimal_output())
        runtime_holder.append(runtime)
        return runtime

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_resolver=_resolver(paths),
        runtime_probe=_ready_probe,
        runtime_factory=factory,
        pair_resolver=lambda: [
            {"hf_id": mod.MANDATED_HEADLINE_MODEL_IDS[0], "model_path": paths[mod.MANDATED_HEADLINE_MODEL_IDS[0]]}
        ],
        max_headline_models=1,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert runtime_holder and runtime_holder[0].closed is True
    assert artifact["headline_models_used"] == [mod.MANDATED_HEADLINE_MODEL_IDS[0]]
    assert artifact["legacy_smoke_models_used"] == []
    assert artifact["cached_models_missing"] == []
    assert artifact["llama_cpp_cuda_available"] is True
    assert artifact["gpu_offload_verified"] is True
    assert artifact["gpu_memory_delta_mb"] == pytest.approx(1536.0)
    assert artifact["exact_validator_accuracy"] == pytest.approx(1.0)
    assert artifact["hard_constraint_violation_rate"] == pytest.approx(0.0)
    assert artifact["preference_optimality_rate"] == pytest.approx(1.0)
    assert artifact["concept_claim_telemetry_rows"] == 3
    assert artifact["guided_decoding_used"] is False
    assert artifact["token_steering_used"] is False
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["abstention_count"] == 1
    assert artifact["token_counts"] == {
        "prompt_tokens": 101,
        "completion_tokens": 37,
        "total_tokens": 138,
    }
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_verify_5500_missing_cache_blocks_without_cpu_headline(tmp_path: Path) -> None:
    """REQ-VERIFY-5500: no mandated cache writes a blocked artifact, not CPU fallback."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_resolver=lambda hf_id, preferred_quant="Q4_K_M": None,
        runtime_probe=_ready_probe,
        runtime_factory=lambda spec: pytest.fail("runtime must not load without cache"),
        pair_resolver=lambda: None,
        max_headline_models=1,
    )

    assert artifact["headline_models_used"] == []
    assert artifact["legacy_smoke_models_used"] == []
    assert artifact["cached_models_missing"] == list(mod.MANDATED_HEADLINE_MODEL_IDS)
    assert artifact["gpu_offload_verified"] is False
    assert artifact["gpu_memory_delta_mb"] == pytest.approx(0.0)
    assert artifact["concept_claim_telemetry_rows"] == 0
    assert artifact["exact_validator_accuracy"] == pytest.approx(0.0)
    assert artifact["blocked_reasons"] == ["no_cached_mandated_sota_gguf"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_req_verify_5500_missing_offload_blocks_even_with_cache(tmp_path: Path) -> None:
    """REQ-VERIFY-5500: cached weights cannot become CPU headline inference."""

    paths = _touch_cached_models(tmp_path)
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_resolver=_resolver(paths),
        runtime_probe=_blocked_probe,
        runtime_factory=lambda spec: pytest.fail("runtime must not load without offload"),
        pair_resolver=lambda: None,
        max_headline_models=1,
    )

    assert artifact["headline_models_used"] == []
    assert artifact["cached_models_missing"] == []
    assert artifact["llama_cpp_cuda_available"] is False
    assert artifact["gpu_offload_verified"] is False
    assert "llama_cpp_gpu_offload_unavailable" in artifact["blocked_reasons"]
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_scenario_verify_5500_bad_generation_is_measured_not_promoted(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5500: exact validators expose hard violations and abstentions."""

    paths = _touch_cached_models(tmp_path)
    bad_output = (
        "model explanation before JSON "
        + json.dumps(
            {
                "instances": [
                    {
                        "instance_id": "claim_support_preference",
                        "assignment": {
                            "support": "unsupported",
                            "source_quality": "primary",
                            "scope": "bounded",
                        },
                    },
                    {
                        "instance_id": "claim_safety_conflict",
                        "assignment": {
                            "safety": "safe",
                            "citation": "present",
                            "action": "reject",
                        },
                    },
                    {
                        "instance_id": "claim_infeasible_negative_control",
                        "assignment": {"verdict": "accept", "evidence": "present"},
                    },
                ]
            }
        )
    )

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_resolver=_resolver(paths),
        runtime_probe=_ready_probe,
        runtime_factory=lambda spec: FakeRuntime(spec, bad_output),
        pair_resolver=lambda: None,
        max_headline_models=1,
    )

    assert artifact["headline_models_used"] == [mod.MANDATED_HEADLINE_MODEL_IDS[0]]
    assert artifact["exact_validator_accuracy"] == pytest.approx(0.0)
    assert artifact["hard_constraint_violation_rate"] == pytest.approx(2 / 3)
    assert artifact["preference_optimality_rate"] == pytest.approx(0.0)
    assert artifact["abstention_count"] == 0
    assert "accuracy_0.0" in artifact["honest_verdict"]
    assert {row["exact_validator_verdict"] for row in artifact["concept_claim_telemetry"]} == {
        "hard_constraint_violation",
        "soft_suboptimal",
    }
    mod.validate_artifact(artifact)


def test_req_verify_5500_defensive_parsers_classify_malformed_rows() -> None:
    """REQ-VERIFY-5500: malformed free-form outputs become explicit abstentions."""

    fixture = mod.load_fixture_artifact()["fixture"]
    instance = fixture["instances"][0]

    assert mod.quant_from_filename("model-no-quant.gguf") == "unknown"
    assert mod.parse_candidate_payload("{not valid json") == {}
    assert mod.candidate_rows_by_instance({"instances": {"bad": "shape"}}) == {}
    assert mod.coerce_assignment(instance, None) == (None, True, "missing_candidate")
    assert mod.coerce_assignment(instance, {"instance_id": instance["instance_id"]}) == (
        None,
        True,
        "missing_assignment",
    )
    assert mod.coerce_assignment(instance, {"assignment": {"support": "entailed"}}) == (
        None,
        True,
        "invalid_assignment_keys",
    )
    invalid_domain = {
        "assignment": {
            "support": "entailed",
            "source_quality": "tertiary",
            "scope": "bounded",
        }
    }
    assert mod.coerce_assignment(instance, invalid_domain) == (
        None,
        True,
        "invalid_assignment_domain",
    )
    assert mod.command_or_binding([{"llama_cpp_binding": "llama_cpp.Llama"}], []) == "llama_cpp.Llama"


def test_req_verify_5500_runtime_load_failures_block_with_diagnostics(tmp_path: Path) -> None:
    """REQ-VERIFY-5500: load failures and unverified offload stay non-headline."""

    paths = _touch_cached_models(tmp_path)
    calls = 0

    def factory(spec: dict[str, Any]) -> FakeNoOffloadRuntime:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("load failed")
        return FakeNoOffloadRuntime(spec)

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_resolver=_resolver(paths),
        runtime_probe=_ready_probe,
        runtime_factory=factory,
        pair_resolver=lambda: None,
        max_headline_models=2,
    )

    assert artifact["headline_models_used"] == []
    assert artifact["gpu_offload_verified"] is False
    assert artifact["blocked_reasons"] == ["no_gpu_offloaded_headline_model_completed"]
    assert artifact["runtime_errors"] == [
        {
            "model_hf_id": mod.MANDATED_HEADLINE_MODEL_IDS[0],
            "error": "gpu_offload_not_verified_after_load",
        },
        {
            "model_hf_id": mod.MANDATED_HEADLINE_MODEL_IDS[1],
            "error": "RuntimeError: load failed",
        },
    ]
    assert artifact["llama_cpp_command_or_binding"] == "llama_cpp.Llama"
    mod.validate_artifact(artifact)


def test_req_verify_5500_artifact_validation_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5500: artifact validator rejects schema drift and forbidden decoding."""

    paths = _touch_cached_models(tmp_path)
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_resolver=_resolver(paths),
        runtime_probe=_ready_probe,
        runtime_factory=lambda spec: FakeRuntime(spec, _optimal_output()),
        pair_resolver=lambda: None,
        max_headline_models=1,
    )
    mod.validate_artifact(artifact)

    missing = deepcopy(artifact)
    missing.pop("model_specs")
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(missing)

    bad_guidance = deepcopy(artifact)
    bad_guidance["guided_decoding_used"] = True
    with pytest.raises(ValueError, match="guided_decoding_used"):
        mod.validate_artifact(bad_guidance)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "verifier_ensemble_against_cached_candidates"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_fixture = deepcopy(artifact)
    bad_fixture["fixture_artifact"] = "results/wrong.json"
    with pytest.raises(ValueError, match="fixture_artifact"):
        mod.validate_artifact(bad_fixture)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)
