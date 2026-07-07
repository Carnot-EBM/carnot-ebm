"""Tests for Exp 5331 stable local internal-energy receipt harness.

Spec refs: REQ-VERIFY-5331, SCENARIO-VERIFY-5331.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5331_internal_energy_receipt_harness_v486 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 16)
    return path


def _model_specs(model_path: Path) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for spec in mod.MODEL_SPECS:
        role = str(spec["role"])
        specs[role] = {
            "role": role,
            "hf_id": spec["hf_id"],
            "quantization": "Q4_K_M",
            "model_path": str(model_path) if role == "flagship_dense" else None,
            "status": "local_gguf_resolved" if role == "flagship_dense" else "missing_local_gguf",
            "autotokenizer_used": False,
        }
    return specs


def _stable_prior(model_path: Path, binary_path: Path) -> dict[str, Any]:
    command = [str(binary_path), "-m", str(model_path), "-p", "Return exactly OK.", "-n", "2"]
    return {
        "experiment_id": {"value": "experiment_5324_runtime_receipt_stabilization_v486"},
        "status": {"value": "complete"},
        "honest_verdict": {"value": "complete: local_native_llama_cpp_stability_receipts"},
        "sota_runtime_unblocked_stable": True,
        "MODEL_SPECS": {"value": _model_specs(model_path)},
        "selected_model_spec": {"value": _model_specs(model_path)["flagship_dense"]},
        "selected_backend_command": {
            "value": {
                "backend_kind": "llama-cli",
                "backend_variant": "llama-cli-single-turn-batch512",
                "command": command,
                "model_path": str(model_path),
                "model_role": "flagship_dense",
                "prompt": "Return exactly OK.",
                "n_predict": 2,
                "context": 512,
                "batch": 512,
                "ubatch": 128,
                "gpu_layers": "all",
                "timeout_s": 120.0,
            }
        },
    }


def _write_prior(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _preconditions(binary_path: Path, *, gpu_visible: bool = True) -> dict[str, Any]:
    return {
        "gpu_visible": gpu_visible,
        "nvidia_smi": {
            "ok": gpu_visible,
            "stdout": "0, NVIDIA RTX 3090, 610.43.02, 24576, 24120, 0",
        },
        "free_vram_mb": 24120 if gpu_visible else 0,
        "binary_paths": {
            "llama-cli": str(binary_path),
            "llama-server": str(binary_path.with_name("llama-server")),
        },
        "binary_versions": {"llama-cli": {"ok": True, "stderr": "version: 9606 CUDA"}},
        "cuda_backend_evidence": gpu_visible,
    }


def _option_surface(*, n_probs: bool) -> dict[str, Any]:
    server_help = "--props\n--slots\n--metrics\n" + (
        "completion n_probs top_logprobs\n" if n_probs else ""
    )
    return mod.summarize_backend_options(
        {
            "llama-cli": {"ok": True, "stdout": "--perf\n--show-timings\n--logit-bias\n"},
            "llama-server": {"ok": True, "stdout": server_help},
        }
    )


def _token_probability_probe(**kwargs: Any) -> dict[str, Any]:
    assert kwargs["selected_model_spec"]["role"] == "flagship_dense"
    return {
        "status": "completed",
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "prompt": kwargs["prompt"],
        "response_json": {
            "content": "\n\nOK",
            "tokens_predicted": 2,
            "tokens_evaluated": 5,
            "timings": {
                "prompt_n": 5,
                "predicted_n": 2,
                "predicted_per_token_ms": 30.563,
            },
            "completion_probabilities": [
                {
                    "id": 108,
                    "token": "\n\n",
                    "logprob": -0.22484834492206573,
                    "top_logprobs": [
                        {"id": 108, "token": "\n\n", "logprob": -0.22484834492206573},
                        {"id": 107, "token": "\n", "logprob": -2.7778267860412598},
                    ],
                },
                {
                    "id": 16067,
                    "token": "OK",
                    "logprob": -0.001,
                    "top_logprobs": [{"id": 16067, "token": "OK", "logprob": -0.001}],
                },
            ],
        },
        "wall_clock_s": 18.2,
    }


def _raw_only_probe(**kwargs: Any) -> dict[str, Any]:
    return {
        "status": "completed",
        "backend_kind": "llama-cli",
        "endpoint": None,
        "prompt": kwargs["prompt"],
        "response_json": {
            "content": "OK",
            "tokens_predicted": 2,
            "timings": {"predicted_n": 2, "predicted_per_token_ms": 31.0},
        },
        "wall_clock_s": 18.2,
    }


def test_req_verify_5331_spec_declares_internal_receipt_contract() -> None:
    """REQ-VERIFY-5331: OpenSpec anchors the internal receipt harness."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5331") : spec.index("### REQ-VERIFY-5326")]

    for marker in (
        "REQ-VERIFY-5331",
        "SCENARIO-VERIFY-5331",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "internal_signal_receipt_ready",
        "external_text_scorer_reopened=false",
        "no_quality_claim=true",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_verify_5331_blocks_when_exp5324_is_not_stable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5331: missing stable Exp5324 blocks before probing signals."""

    calls: list[str] = []
    prior_path = _write_prior(
        tmp_path / "results" / "experiment_5324_runtime_receipt_stabilization_v486.json",
        {"sota_runtime_unblocked_stable": False},
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        prior_artifact_path=prior_path,
        preconditions_provider=lambda: _preconditions(tmp_path / "llama-cli"),
        option_surface_provider=lambda _preconditions: calls.append("options") or {},
        signal_probe=lambda **_kwargs: calls.append("signal") or {},
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert calls == []
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["internal_signal_receipt_ready"] is False
    assert artifact["logits_available"] is False
    assert artifact["token_probability_available"] is False
    assert artifact["attention_available"] is False
    assert artifact["hidden_state_proxy_available"] is False
    assert artifact["external_text_scorer_reopened"] is False
    assert artifact["no_quality_claim"] is True
    assert (
        "exp5324_stable_backend_unavailable"
        in artifact["preconditions_checked"]["value"]["blocked_preconditions"]
    )
    assert Path(artifact["receipt_schema_path"]["value"]).is_file()


def test_scenario_verify_5331_saves_token_probability_receipt(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5331: native token probabilities open the internal receipt gate."""

    model_path = _write_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.with_name("llama-server").write_text("#!/bin/sh\n", encoding="utf-8")
    prior_path = _write_prior(
        tmp_path / "results" / "experiment_5324_runtime_receipt_stabilization_v486.json",
        _stable_prior(model_path, binary),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        prior_artifact_path=prior_path,
        preconditions_provider=lambda: _preconditions(binary),
        option_surface_provider=lambda _preconditions: _option_surface(n_probs=True),
        signal_probe=_token_probability_probe,
        tests_run=[{"command": "unit complete", "outcome": "passed"}],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["selected_model_spec"]["value"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert artifact["MODEL_SPECS"]["value"]["flagship_dense"]["hf_id"] == (
        "unsloth/gemma-4-31B-it-GGUF"
    )
    assert artifact["logits_available"] is False
    assert artifact["token_probability_available"] is True
    assert artifact["attention_available"] is False
    assert artifact["hidden_state_proxy_available"] is False
    assert artifact["token_timing_available"] is True
    assert artifact["raw_output_receipt_available"] is True
    assert artifact["internal_signal_receipt_ready"] is True
    tiny = json.loads(Path(artifact["tiny_receipt_path"]["value"]).read_text(encoding="utf-8"))
    assert tiny["receipt_kind"] == "token_probability"
    assert tiny["completion_probabilities"][0]["top_logprobs"][0]["logprob"] < 0
    schema = json.loads(Path(artifact["receipt_schema_path"]["value"]).read_text(encoding="utf-8"))
    assert schema["internal_signal_receipt_ready"] is True
    assert "completion_probabilities" in schema["receipt_fields"]


def test_scenario_verify_5331_raw_text_and_timing_do_not_open_gate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5331: raw output plus aggregate timing is still blocked."""

    model_path = _write_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.with_name("llama-server").write_text("#!/bin/sh\n", encoding="utf-8")
    prior_path = _write_prior(
        tmp_path / "results" / "experiment_5324_runtime_receipt_stabilization_v486.json",
        _stable_prior(model_path, binary),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "raw_only.json",
        prior_artifact_path=prior_path,
        preconditions_provider=lambda: _preconditions(binary),
        option_surface_provider=lambda _preconditions: _option_surface(n_probs=False),
        signal_probe=_raw_only_probe,
        tests_run=[],
        write=False,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["internal_signal_receipt_ready"] is False
    assert artifact["raw_output_receipt_available"] is True
    assert artifact["token_timing_available"] is True
    assert "token_probability_metadata_unavailable" in artifact["missing_backend_features"]["value"]
    assert artifact["tiny_receipt_path"]["value"] is None


def test_req_verify_5331_signal_summary_handles_all_internal_surfaces() -> None:
    """REQ-VERIFY-5331: helper normalization records each internal surface distinctly."""

    receipt = mod.normalise_signal_receipt(
        {
            "response_json": {
                "content": "x",
                "completion_probabilities": [{"logprob": -0.1, "top_logprobs": []}],
                "timings": {"predicted_per_token_ms": 1.2},
            },
            "logits": {"top_logits": [{"token_id": 1, "logit": 2.0}]},
            "attention": {"heads": [{"layer": 0, "mean": 0.5}]},
            "hidden_state_proxy": {"embedding": [0.1, 0.2]},
        },
        prompt="p",
        backend_kind="unit",
    )
    availability = mod.signal_availability(receipt)

    assert availability["logits_available"] is True
    assert availability["token_probability_available"] is True
    assert availability["attention_available"] is True
    assert availability["hidden_state_proxy_available"] is True
    assert availability["token_timing_available"] is True
    assert availability["raw_output_receipt_available"] is True
    assert mod._receipt_kind(availability) == "multi_internal_signal"


def test_req_verify_5331_schema_errors_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5331: artifact validation rejects scorer and quality-claim drift."""

    model_path = _write_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.with_name("llama-server").write_text("#!/bin/sh\n", encoding="utf-8")
    prior_path = _write_prior(
        tmp_path / "results" / "experiment_5324_runtime_receipt_stabilization_v486.json",
        _stable_prior(model_path, binary),
    )
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "complete.json",
        prior_artifact_path=prior_path,
        preconditions_provider=lambda: _preconditions(binary),
        option_surface_provider=lambda _preconditions: _option_surface(n_probs=True),
        signal_probe=_token_probability_probe,
        write=False,
    )

    bad = dict(artifact)
    bad["external_text_scorer_reopened"] = True
    bad["no_quality_claim"] = False
    bad["honest_verdict"] = {"value": "complete without prefix", "principle": "wrong"}
    errors = mod.artifact_schema_errors(bad)

    assert any("external_text_scorer_reopened must be bare false" in error for error in errors)
    assert any("no_quality_claim must be bare true" in error for error in errors)
    assert any("honest_verdict must be principle-wrapped" in error for error in errors)
    assert any("honest_verdict must start with complete: or blocked_" in error for error in errors)


def test_req_verify_5331_helper_edges_and_schema_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-5331: defensive helper and schema branches fail closed."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json(tmp_path / "missing.json") == {}
    assert mod._read_json(bad_json) == {}
    assert (
        mod._selected_model_from_prior({"selected_model_spec": {"value": {"role": "bad"}}}) is None
    )
    assert (
        mod._selected_command_from_prior({"selected_backend_command": {"value": {"command": []}}})
        is None
    )
    assert mod._normalise_top_logprobs("bad") == []
    assert len(mod._normalise_top_logprobs([1, {"token": "x", "logprob": -1.0}])) == 1
    assert mod._normalise_completion_probabilities("bad") == []
    assert len(mod._normalise_completion_probabilities([1, {"top_logprobs": "bad"}])) == 1
    assert mod._surface_available({"availability": "available"}) is True
    assert mod._surface_available([1]) is True
    assert mod._surface_available(0) is False
    assert mod._receipt_kind({"logits_available": True}) == "logits"
    assert mod._receipt_kind({"token_probability_available": True}) == "token_probability"
    assert mod._receipt_kind({"attention_available": True}) == "attention"
    assert mod._receipt_kind({"hidden_state_proxy_available": True}) == "hidden_state_proxy"
    assert mod._receipt_kind({}) == "none"

    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    missing_model = {"model_path": str(tmp_path / "missing.gguf")}
    assert "selected_model_file_missing" in mod._precondition_blockers(
        prior_stable=True,
        selected_model_spec=missing_model,
        selected_backend_command={"command": [str(binary)]},
        preconditions={"gpu_visible": True},
    )
    blockers = mod._precondition_blockers(
        prior_stable=True,
        selected_model_spec=None,
        selected_backend_command={"command": []},
        preconditions={"gpu_visible": False},
    )
    assert "selected_mandated_model_unavailable" in blockers
    assert "gpu_not_visible" in blockers
    assert "selected_backend_command_malformed" in blockers
    assert "selected_backend_binary_missing" in mod._precondition_blockers(
        prior_stable=True,
        selected_model_spec={"model_path": str(binary)},
        selected_backend_command={"command": [str(tmp_path / "missing-bin")]},
        preconditions={"gpu_visible": True},
    )

    model_path = _write_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    binary.with_name("llama-server").write_text("#!/bin/sh\n", encoding="utf-8")
    prior_path = _write_prior(
        tmp_path / "results" / "experiment_5324_runtime_receipt_stabilization_v486.json",
        _stable_prior(model_path, binary),
    )
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "complete.json",
        prior_artifact_path=prior_path,
        preconditions_provider=lambda: _preconditions(binary),
        option_surface_provider=lambda _preconditions: _option_surface(n_probs=True),
        signal_probe=_token_probability_probe,
        write=False,
    )

    missing = copy.deepcopy(artifact)
    del missing["token_probability_available"]
    assert any("missing required fields" in error for error in mod.artifact_schema_errors(missing))

    bad = copy.deepcopy(artifact)
    bad["milestone"] = "not wrapped"
    bad["experiment_id"]["value"] = "wrong"
    bad["status"]["value"] = "pending"
    bad["inference_substrate"]["value"] = "wrong"
    bad["logits_available"] = "yes"
    bad["MODEL_SPECS"]["value"] = "not object"
    bad["receipt_schema_path"]["value"] = None
    bad["tests_run"]["value"] = "not list"
    errors = mod.artifact_schema_errors(bad)
    assert any("milestone must be principle-wrapped" in error for error in errors)
    assert any("experiment_id mismatch" in error for error in errors)
    assert any("status must be complete or blocked" in error for error in errors)
    assert any("inference_substrate mismatch" in error for error in errors)
    assert any("logits_available must be a bare boolean" in error for error in errors)
    assert any("MODEL_SPECS must be an object" in error for error in errors)
    assert any("receipt_schema_path must be" in error for error in errors)
    assert any("tests_run must be a list" in error for error in errors)

    bad_specs = copy.deepcopy(artifact)
    del bad_specs["MODEL_SPECS"]["value"]["middle_moe"]
    bad_specs["MODEL_SPECS"]["value"]["flagship_dense"]["hf_id"] = "wrong"
    bad_specs["MODEL_SPECS"]["value"]["flagship_moe"]["autotokenizer_used"] = True
    spec_errors = mod.artifact_schema_errors(bad_specs)
    assert any("MODEL_SPECS roles mismatch" in error for error in spec_errors)
    assert any("MODEL_SPECS hf_id mismatch" in error for error in spec_errors)
    assert any("autotokenizer_used must stay false" in error for error in spec_errors)

    bad_ready = copy.deepcopy(artifact)
    bad_ready["status"]["value"] = "blocked"
    bad_ready["logits_available"] = False
    bad_ready["token_probability_available"] = False
    bad_ready["tiny_receipt_path"]["value"] = None
    ready_errors = mod.artifact_schema_errors(bad_ready)
    assert any("ready artifact must have complete status" in error for error in ready_errors)
    assert any("ready artifact must expose" in error for error in ready_errors)
    assert any("ready artifact must include tiny_receipt_path" in error for error in ready_errors)

    bad_not_ready = copy.deepcopy(artifact)
    bad_not_ready["internal_signal_receipt_ready"] = False
    assert any(
        "not-ready artifact must have blocked status" in error
        for error in mod.artifact_schema_errors(bad_not_ready)
    )
    try:
        mod.validate_artifact(bad_not_ready)
    except AssertionError as exc:
        assert "not-ready artifact must have blocked status" in str(exc)
    else:
        raise AssertionError("validate_artifact should reject not-ready complete artifact")
