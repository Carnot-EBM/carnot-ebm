"""Tests for Exp 5263 neuron/attention/logit hallucination-energy probe.

Spec refs: REQ-VERIFY-5263, SCENARIO-VERIFY-5263.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5263_neuron_attention_energy_hallucination_probe_v481 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def _ready_preflight() -> dict[str, Any]:
    return {
        "sota_runtime_ready": True,
        "sota_runtime_ready_principle": "sota_runtime_ready=true; ready through flagship_moe",
        "gpu_offload_receipts": {"value": {"llama_cpp": {"version": "0.3.29"}}},
        "model_receipts": {
            "value": {
                "flagship_moe": {
                    "role": "flagship_moe",
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "preferred_quant": "Q4_K_M",
                    "path": "/models/qwen.gguf",
                    "runtime_ready": True,
                    "status": "runtime_ready",
                    "size_bytes": 123,
                    "checksum_head_1m_sha256": "abc",
                },
                "flagship_dense": {
                    "role": "flagship_dense",
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "preferred_quant": "Q4_K_M",
                    "path": "/models/gemma31.gguf",
                    "runtime_ready": True,
                    "status": "runtime_ready",
                    "size_bytes": 456,
                    "checksum_head_1m_sha256": "def",
                },
                "middle_moe": {
                    "role": "middle_moe",
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "preferred_quant": "Q4_K_M",
                    "path": "/models/gemma26.gguf",
                    "runtime_ready": False,
                    "status": "optional_not_loaded",
                    "size_bytes": 789,
                    "checksum_head_1m_sha256": "ghi",
                },
            }
        },
    }


def _logprob_signal_surface() -> dict[str, Any]:
    return {
        "hidden_states": False,
        "attention_tensors": False,
        "logits": True,
        "token_logprobs": True,
        "generated_text": True,
        "api_receipts": {"Llama.__call__": "logprobs parameter present"},
    }


def test_req_verify_5263_spec_declares_signal_probe_contract() -> None:
    """REQ-VERIFY-5263: OpenSpec anchors the internal-signal contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5263") :]

    for marker in (
        "REQ-VERIFY-5263",
        "SCENARIO-VERIFY-5263",
        str(mod.RESULT_RELATIVE_PATH),
        "blocked_internal_signal_unavailable",
        "live_llm_inference_local_gguf_sota",
        "llama_cpp_runtime_preflight_no_quality_claim",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "external_text_scorer_used.value=false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5263_fixture_checksums_are_independent_and_deterministic() -> None:
    """REQ-VERIFY-5263: curated labels and prompt checksums are local receipts."""

    fixtures = mod.default_fixtures()
    checksums = mod.fixture_checksums(fixtures)

    assert len(fixtures) == 6
    assert {fixture.unsupported_label for fixture in fixtures} == {False, True}
    assert checksums["fixture_count"] == 6
    assert len(checksums["prompt_checksums"]) == 6
    assert checksums == mod.fixture_checksums(mod.default_fixtures())
    for fixture in fixtures:
        prompt = mod.render_prompt(fixture)
        assert fixture.claim in prompt
        assert fixture.label_source == "curated_local_evidence_label"
        assert "expected" not in prompt.lower()


def test_req_verify_5263_energy_feature_math_and_threshold_metrics() -> None:
    """REQ-VERIFY-5263: pre-registered logprob/logit features are deterministic."""

    features = mod.compute_energy_features(
        {
            "token_logprobs": [math.log(0.8), math.log(0.5)],
            "top_logprobs": [{" SAFE": math.log(0.7), " RISK": math.log(0.3)}],
            "final_logits": [0.0, 1.0],
        }
    )

    assert features["signal_available"] is True
    assert features["sequence_spilled_energy"] == pytest.approx(0.35)
    assert features["sequence_marginal_energy"] == pytest.approx(-math.log(0.8 * 0.5) / 2)
    assert features["final_token_spilled_energy"] == pytest.approx(0.3)
    assert features["full_logit_spilled_energy"] == pytest.approx(1.0 / (1.0 + math.e))
    assert features["primary_energy"] == pytest.approx(features["sequence_marginal_energy"])

    rows = [
        {"unsupported_label": False, "energy_features": {"primary_energy": 0.1}},
        {"unsupported_label": False, "energy_features": {"primary_energy": 0.2}},
        {"unsupported_label": True, "energy_features": {"primary_energy": 0.8}},
        {"unsupported_label": True, "energy_features": {"primary_energy": 0.9}},
    ]
    summary = mod.summarize_energy(rows)

    assert summary["signal_delta"] == pytest.approx(0.7)
    assert summary["auroc"] == pytest.approx(1.0)
    assert summary["false_accepts_at_threshold"] == 0
    assert summary["precision_at_threshold"] == pytest.approx(1.0)
    assert mod.deterministic_baselines(mod.default_fixtures())["always_supported"]["false_accepts"] == 3


def test_scenario_verify_5263_blocks_text_only_runtime_without_external_scorer(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5263: text-only local runtime exits with a blocked artifact."""

    artifact = mod.run_pilot(
        result_path=tmp_path / "blocked.json",
        preflight_artifact=_ready_preflight(),
        signal_surface={
            "hidden_states": False,
            "attention_tensors": False,
            "logits": False,
            "token_logprobs": False,
            "generated_text": True,
            "api_receipts": {},
        },
        generation_runner=lambda fixture, model_spec, seed: {"raw_response": "SUPPORTED"},
        commands_run=[{"command": "unit blocked", "outcome": "passed"}],
    )

    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("blocked_internal_signal_unavailable")
    assert artifact["inference_substrate"]["value"] == mod.PREFLIGHT_SUBSTRATE
    assert artifact["internal_signal_available"] is False
    assert artifact["hidden_energy_probe_signal_delta"] == 0.0
    assert artifact["false_accepts_at_threshold"]["value"] == 0
    assert artifact["external_text_scorer_used"]["value"] is False
    assert artifact["pilot_rows"] == []
    mod.validate_artifact(artifact)


def test_scenario_verify_5263_runs_injected_logprob_pilot(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5263: available logprob receipts produce a bounded pilot."""

    def runner(
        fixture: mod.HallucinationFixture,
        model_spec: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        assert model_spec["slot"] == "flagship_moe"
        base = math.log(0.35 if fixture.unsupported_label else 0.95)
        return {
            "raw_response": "UNSUPPORTED" if fixture.unsupported_label else "SUPPORTED",
            "token_logprobs": [base, base],
            "top_logprobs": [
                {
                    " UNSUPPORTED": math.log(0.55 if fixture.unsupported_label else 0.05),
                    " SUPPORTED": math.log(0.45 if fixture.unsupported_label else 0.95),
                }
            ],
            "token_count": 2,
            "seed": seed,
            "logit_receipt": {"steps": 1, "vocab_size": 4},
        }

    artifact = mod.run_pilot(
        result_path=tmp_path / "ready.json",
        preflight_artifact=_ready_preflight(),
        signal_surface=_logprob_signal_surface(),
        generation_runner=runner,
        commands_run=[{"command": "unit ready", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / "ready.json").read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("complete: signal")
    assert artifact["inference_substrate"]["value"] == mod.LIVE_SUBSTRATE
    assert artifact["internal_signal_available"] is True
    assert artifact["hidden_energy_probe_signal_delta"] > 0.0
    assert artifact["separation_summary"]["auroc"] == pytest.approx(1.0)
    assert artifact["false_accepts_at_threshold"]["value"] == 0
    assert artifact["MODEL_SPECS"]["value"]["flagship_moe"]["selected_for_pilot"] is True
    assert len(artifact["pilot_rows"]) == len(mod.default_fixtures())
    assert all(row["prompt_checksum"] for row in artifact["pilot_rows"])
    assert artifact["external_text_scorer_used"]["value"] is False


def test_req_verify_5263_live_rows_without_receipts_block(tmp_path: Path) -> None:
    """REQ-VERIFY-5263: API promise is insufficient if live rows are text-only."""

    artifact = mod.run_pilot(
        result_path=tmp_path / "no-receipts.json",
        preflight_artifact=_ready_preflight(),
        signal_surface=_logprob_signal_surface(),
        generation_runner=lambda fixture, model_spec, seed: {"raw_response": "SUPPORTED"},
        commands_run=[],
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_internal_signal_unavailable")
    assert artifact["internal_signal_available"] is False
    assert artifact["preconditions_checked"]["value"]["live_signal_receipts_found"] is False
    mod.validate_artifact(artifact)


def test_req_verify_5263_precondition_and_schema_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5263: missing gates and malformed artifacts are rejected."""

    blocked = mod.run_pilot(
        result_path=tmp_path / "preflight-blocked.json",
        preflight_artifact={"sota_runtime_ready": False},
        signal_surface=_logprob_signal_surface(),
        generation_runner=lambda fixture, model_spec, seed: {"token_logprobs": [-0.1]},
        commands_run=[],
    )

    assert blocked["honest_verdict"]["value"].startswith("blocked_sota_runtime_unavailable")
    assert blocked["internal_signal_available"] is False
    mod.validate_artifact(blocked)

    valid = mod.run_pilot(
        result_path=tmp_path / "valid.json",
        preflight_artifact=_ready_preflight(),
        signal_surface={
            "hidden_states": False,
            "attention_tensors": False,
            "logits": False,
            "token_logprobs": False,
            "generated_text": True,
            "api_receipts": {},
        },
        generation_runner=lambda fixture, model_spec, seed: {},
        commands_run=[],
    )

    for mutation, message in (
        (lambda art: {key: value for key, value in art.items() if key != "honest_verdict"}, "missing required field"),
        (lambda art: art | {"honest_verdict": {"value": "pending", "principle": mod.FIELD_PRINCIPLES["honest_verdict"]}}, "honest_verdict"),
        (lambda art: art | {"internal_signal_available": "false"}, "bare bool"),
        (lambda art: art | {"hidden_energy_probe_signal_delta": "0.0"}, "bare float"),
        (
            lambda art: art
            | {
                "external_text_scorer_used": {
                    "value": True,
                    "principle": mod.FIELD_PRINCIPLES["external_text_scorer_used"],
                }
            },
            "external_text_scorer_used",
        ),
        (
            lambda art: art
            | {
                "inference_substrate": {
                    "value": "cached_text_scorer",
                    "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                }
            },
            "inference_substrate",
        ),
    ):
        with pytest.raises(AssertionError, match=message):
            mod.validate_artifact(mutation(valid))


def test_req_verify_5263_defensive_helper_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-5263: edge helpers remain explicit rather than silently claiming signal."""

    summary = mod.summarize_energy(
        [
            {"unsupported_label": False, "energy_features": {"primary_energy": None}},
            {"unsupported_label": False, "energy_features": {"primary_energy": 0.2}},
        ]
    )
    assert summary["n_scored"] == 1
    assert summary["threshold"] is None
    assert summary["auroc"] is None

    tied = mod.summarize_energy(
        [
            {"unsupported_label": False, "energy_features": {"primary_energy": 0.5}},
            {"unsupported_label": True, "energy_features": {"primary_energy": 0.5}},
        ]
    )
    assert tied["auroc"] == pytest.approx(0.5)
    assert mod._honest_verdict(-0.1).startswith("complete: harmful")
    assert mod._honest_verdict(0.0).startswith("complete: null")

    assert mod._first_choice({"choices": [{"text": "x", "logprobs": {}}]})["text"] == "x"
    assert mod._first_choice("raw") == {"text": "raw", "logprobs": {}}
    assert mod._full_logit_summary([]) == {}
    assert mod._softmax_log_values([]) == []
    assert mod._nested_value("not mapping", "field") is None
    assert mod._optional_float(True) is None

    nested_logits = mod.compute_energy_features({"logits": [[0.0, 1.0]], "tokens": ["a", "b"]})
    assert nested_logits["signal_available"] is True
    assert nested_logits["token_count"] == 2

    invalid_receipt_specs = mod._model_specs_from_preflight({"model_receipts": {"value": {"flagship_moe": []}}})
    assert invalid_receipt_specs["flagship_moe"]["runtime_status"] == "missing_receipt"

    artifact = mod.run_pilot(
        result_path=tmp_path / "not-written.json",
        preflight_artifact=_ready_preflight(),
        signal_surface={
            "hidden_states": False,
            "attention_tensors": False,
            "logits": False,
            "token_logprobs": False,
            "generated_text": True,
            "api_receipts": {},
        },
        generation_runner=lambda fixture, model_spec, seed: {},
        commands_run=[],
        write=False,
    )
    assert artifact["honest_verdict"]["value"].startswith("blocked_internal_signal_unavailable")
    assert not (tmp_path / "not-written.json").exists()
