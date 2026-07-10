"""Tests for Exp5513 SOTA GGUF hard/soft structured panel.

Spec refs: REQ-VERIFY-5513, SCENARIO-VERIFY-5513.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5513_sota_hard_soft_structured_panel as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5513_sota_hard_soft_structured_panel.py")


def _ready_gate(tmp_path: Path, *, ready: bool = True) -> Path:
    path = tmp_path / positive.RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "structured_output_positive_control_ready": ready,
                "sota_panel_gate_open": ready,
                "honest_verdict": "complete: ready" if ready else "blocked: closed",
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_cache(tmp_path: Path) -> dict[str, str]:
    paths = {}
    for hf_id in mod.MANDATED_HEADLINE_MODEL_IDS:
        model_path = tmp_path / f"{hf_id.rsplit('/', 1)[-1]}-Q4_K_M.gguf"
        model_path.write_text("fake gguf", encoding="utf-8")
        paths[hf_id] = str(model_path)
    return paths


def _runtime(*, cuda: bool = True, offload: bool = True) -> dict[str, object]:
    return {
        "llama_cpp_cuda_available": cuda,
        "gpu_offload_verified": offload,
        "gpu_memory_delta_mb": 1536.5 if offload else 0.0,
        "offload_diagnostics": [
            {
                "resource": "llama_cpp_gpu_offload",
                "available": offload,
                "detail": "injected test runtime",
            }
        ],
    }


def _telemetry(raw_output: str) -> dict[str, object]:
    return {
        "raw_output": raw_output,
        "llama_cpp_binding": "llama_cpp.Llama.create_completion",
        "llama_cpp_command": None,
        "n_gpu_layers": -1,
        "gpu_memory_before_mb": 1024.0,
        "gpu_memory_after_mb": 2560.5,
        "gpu_memory_delta_mb": 1536.5,
        "wall_time_s": 1.25,
        "prompt_tokens": 321,
        "completion_tokens": 654,
    }


def _wrapper_output(rows: list[dict], proof_claim: str = "exact") -> str:
    claims = []
    for row in rows:
        verdict = "correct_abstention" if row["conclusion"]["status"] == "abstain" else "exact_match"
        claims.append(
            {
                "candidate_id": row["candidate_id"],
                "claimed_exact_validator_verdict": verdict if proof_claim == "exact" else proof_claim,
            }
        )
    return (
        "Brief reasoning: I compare each assignment to the hard constraints first.\n"
        + json.dumps({"candidate_rows": rows, "proof_claims": claims})
    )


def test_req_verify_5513_spec_declares_panel_contract() -> None:
    """REQ-VERIFY-5513: OpenSpec anchors preconditions, fields, and model IDs."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5513") : spec.index("### REQ-VERIFY-5501")
    ]

    assert "SCENARIO-VERIFY-5513" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.STRUCTURED_POSITIVE_CONTROL_ARTIFACT_RELATIVE_PATH) in section
    assert "AutoTokenizer.from_pretrained" in section
    for hf_id in mod.MANDATED_HEADLINE_MODEL_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_verify_5513_gate_closed_blocks_before_inference(tmp_path: Path) -> None:
    """REQ-VERIFY-5513: Exp5512 readiness must be true before model inference."""

    gate = _ready_gate(tmp_path, ready=False)
    cache = _fake_cache(tmp_path)
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        structured_positive_control_path=gate,
        runtime_status=_runtime(),
        cache_resolver=lambda hf_id, _quant="Q4_K_M": cache[hf_id],
        panel_runner=lambda _spec, _prompt: pytest.fail("runner must not be called"),
    )

    assert artifact["headline_models_used"] == []
    assert artifact["cached_models_missing"] == []
    assert artifact["sota_rows_emitted"] == 0
    assert artifact["sota_structured_panel_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "structured_positive_control_not_ready" in artifact["readiness_blockers"]
    mod.validate_artifact(artifact)


def test_req_verify_5513_missing_cache_or_offload_writes_blocked_artifact(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5513: no cache or no GPU offload is terminal diagnostic evidence."""

    gate = _ready_gate(tmp_path)
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        structured_positive_control_path=gate,
        runtime_status=_runtime(cuda=False, offload=False),
        cache_resolver=lambda _hf_id, _quant="Q4_K_M": None,
        panel_runner=lambda _spec, _prompt: pytest.fail("runner must not be called"),
    )

    assert artifact["headline_models_used"] == []
    assert artifact["cached_models_missing"] == list(mod.MANDATED_HEADLINE_MODEL_IDS)
    assert artifact["llama_cpp_cuda_available"] is False
    assert artifact["gpu_offload_verified"] is False
    assert artifact["gpu_memory_delta_mb"] == pytest.approx(0.0)
    assert artifact["exact_validator_accuracy"] == pytest.approx(0.0)
    assert artifact["sota_structured_panel_ready"] is False
    assert "no_cached_mandated_gguf" in artifact["readiness_blockers"]
    assert "llama_cpp_cuda_unavailable" in artifact["readiness_blockers"]
    assert "gpu_offload_unverified" in artifact["readiness_blockers"]
    mod.validate_artifact(artifact)


def test_scenario_verify_5513_scores_injected_headline_model_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5513: schema rows score through Exp5499 exact validators."""

    gate = _ready_gate(tmp_path)
    cache = _fake_cache(tmp_path)
    rows = positive.build_fixture_candidate_payloads()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        structured_positive_control_path=gate,
        runtime_status=_runtime(),
        cache_resolver=lambda hf_id, _quant="Q4_K_M": cache[hf_id],
        panel_runner=lambda _spec, _prompt: _telemetry(_wrapper_output(rows)),
        max_headline_models=1,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["headline_models_used"] == [mod.MANDATED_HEADLINE_MODEL_IDS[0]]
    assert artifact["legacy_smoke_models_used"] == []
    assert artifact["exact_validator_accuracy"] == pytest.approx(1.0)
    assert artifact["hard_constraint_violation_rate"] == pytest.approx(0.0)
    assert artifact["preference_optimality_rate"] == pytest.approx(1.0)
    assert artifact["schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["abstention_rate"] == pytest.approx(1 / 3)
    assert artifact["missing_candidate_rows"] == 0
    assert artifact["sota_rows_emitted"] == 3
    assert artifact["sota_structured_panel_ready"] is True
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["model_runs"][0]["model_file"].endswith(".gguf")
    assert artifact["model_runs"][0]["quant"] == "Q4_K_M"
    assert artifact["model_runs"][0]["n_gpu_layers"] == -1
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_verify_5513_missing_rows_parse_failures_and_proof_mismatch_are_visible(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5513: missing rows and proof mismatches are evidence, not drops."""

    gate = _ready_gate(tmp_path)
    cache = _fake_cache(tmp_path)
    partial_rows = positive.build_fixture_candidate_payloads()[:2]
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        structured_positive_control_path=gate,
        runtime_status=_runtime(),
        cache_resolver=lambda hf_id, _quant="Q4_K_M": cache[hf_id],
        panel_runner=lambda _spec, _prompt: _telemetry(
            _wrapper_output(partial_rows, proof_claim="exact_match")
        ),
        max_headline_models=1,
    )

    assert artifact["sota_structured_panel_ready"] is False
    assert artifact["sota_rows_emitted"] == 2
    assert artifact["missing_candidate_rows"] == 1
    assert artifact["schema_validity_rate"] == pytest.approx(2 / 3)
    assert artifact["exact_validator_accuracy"] == pytest.approx(2 / 3)
    assert artifact["proof_claim_consistency_rate"] == pytest.approx(1.0)
    assert artifact["model_runs"][0]["missing_instance_ids"] == [
        "claim_infeasible_negative_control"
    ]

    no_json = mod.run(
        result_path=tmp_path / "no_json.json",
        structured_positive_control_path=gate,
        runtime_status=_runtime(),
        cache_resolver=lambda hf_id, _quant="Q4_K_M": cache[hf_id],
        panel_runner=lambda _spec, _prompt: _telemetry("reasoning only, no JSON"),
        max_headline_models=1,
    )
    assert no_json["parse_failure_counts"] == {"no_json_payload": 1}
    assert no_json["missing_candidate_rows"] == 3
    assert no_json["schema_validity_rate"] == pytest.approx(0.0)
    assert no_json["sota_structured_panel_ready"] is False

    proof_bad = mod.run(
        result_path=tmp_path / "proof_bad.json",
        structured_positive_control_path=gate,
        runtime_status=_runtime(),
        cache_resolver=lambda hf_id, _quant="Q4_K_M": cache[hf_id],
        panel_runner=lambda _spec, _prompt: _telemetry(
            _wrapper_output(positive.build_fixture_candidate_payloads(), proof_claim="soft_suboptimal")
        ),
        max_headline_models=1,
    )
    assert proof_bad["proof_claim_consistency_rate"] == pytest.approx(0.0)
    assert proof_bad["sota_structured_panel_ready"] is False
    mod.validate_artifact(proof_bad)


def test_req_verify_5513_validation_and_parser_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5513: artifact and parser helpers fail closed."""

    gate = _ready_gate(tmp_path)
    cache = _fake_cache(tmp_path)
    assert mod.load_structured_positive_control(tmp_path / "missing_5512.json")[
        "structured_output_positive_control_ready"
    ] is False

    single_payload = positive.build_fixture_candidate_payloads()[0]
    assert mod.extract_candidate_payloads(json.dumps(single_payload))["candidate_payloads"] == [
        single_payload
    ]
    assert mod.extract_candidate_payloads(json.dumps({"not_candidate_rows": []}))[
        "parse_failures"
    ] == [{"parse_status": "candidate_rows_missing"}]
    recovered = mod.extract_candidate_payloads("bad {not json} " + json.dumps(single_payload))
    assert recovered["candidate_payloads"] == [single_payload]
    preferred = mod.extract_candidate_payloads(
        json.dumps({"candidate_id": "...", "claimed_exact_validator_verdict": "..."})
        + json.dumps({"candidate_rows": [single_payload]})
    )
    assert preferred["candidate_payloads"] == [single_payload]
    assert mod._payload_has_candidate_rows("not json") is False

    payload = deepcopy(positive.build_fixture_candidate_payloads()[0])
    payload.pop("premises")
    artifact = mod.run(
        result_path=tmp_path / "schema_bad.json",
        structured_positive_control_path=gate,
        runtime_status=_runtime(),
        cache_resolver=lambda hf_id, _quant="Q4_K_M": cache[hf_id],
        panel_runner=lambda _spec, _prompt: _telemetry(json.dumps([payload])),
        max_headline_models=1,
        pair_resolver=lambda: None,
    )
    assert artifact["parse_failure_counts"] == {"schema_invalid": 1}
    assert artifact["schema_validity_rate"] == pytest.approx(0.0)
    assert artifact["candidate_rows"][0]["parse_status"] == "schema_invalid"

    hard_bad = deepcopy(positive.build_fixture_candidate_payloads()[0])
    hard_bad["conclusion"]["assignment"]["support"] = "unsupported"
    hard_violation = mod.run(
        result_path=tmp_path / "hard_bad.json",
        structured_positive_control_path=gate,
        runtime_status=_runtime(),
        cache_resolver=lambda hf_id, _quant="Q4_K_M": cache[hf_id],
        panel_runner=lambda _spec, _prompt: _wrapper_output([hard_bad]),
        max_headline_models=1,
    )
    assert hard_violation["hard_constraint_violation_rate"] == pytest.approx(1 / 3)
    assert "hard_constraint_violation" in hard_violation["readiness_blockers"]

    runtime_error = mod.run(
        result_path=tmp_path / "runtime_error.json",
        structured_positive_control_path=gate,
        runtime_status=_runtime(),
        cache_resolver=lambda hf_id, _quant="Q4_K_M": cache[hf_id],
        panel_runner=lambda _spec, _prompt: (_ for _ in ()).throw(RuntimeError("boom")),
        max_headline_models=1,
    )
    assert runtime_error["parse_failure_counts"] == {
        "no_json_payload": 1,
        "runtime_error": 1,
    }
    assert runtime_error["model_runs"][0]["runtime_error"] == "RuntimeError: boom"

    prompt = mod.build_reason_then_structure_prompt(positive.build_fixture_candidate_payloads())
    assert "brief reasoning" in prompt
    assert positive.CANDIDATE_SCHEMA_VERSION in prompt

    missing = deepcopy(artifact)
    missing.pop("model_specs")
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "fixture"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_model = deepcopy(artifact)
    bad_model["headline_models_used"] = ["legacy/model"]
    with pytest.raises(ValueError, match="headline_models_used"):
        mod.validate_artifact(bad_model)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)
