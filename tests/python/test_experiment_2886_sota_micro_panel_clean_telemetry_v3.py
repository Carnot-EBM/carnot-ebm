"""Tests for Exp 2886 SOTA micro-panel clean telemetry v3.

Spec: REQ-INFER-SOTA-017,
      SCENARIO-INFER-SOTA-017-001,
      SCENARIO-INFER-SOTA-017-002,
      SCENARIO-INFER-SOTA-017-003
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import sota_micro_panel_clean_telemetry_v3 as mod
from carnot.reporting.sota_micro_panel_clean_telemetry_v3 import (
    ExperimentConfig,
    REQUIRED_ARTIFACT_FIELDS,
)


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31 = "unsloth/gemma-4-31B-it-GGUF"
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


def _exp2874_payload(model_path: Path, *, clean: bool = True) -> dict[str, Any]:
    return {
        "sota_runtime_clean": clean,
        "sota_runtime_ready_v4": clean,
        "selected_model_hf_id": GEMMA26,
        "selected_model_path": str(model_path),
        "llama_cpp_gpu_offload_verified": True,
        "cached_sota_pair_returned_two_loadable_specs": False,
        "model_specs": [
            {"hf_id": QWEN, "legacy_smoke_only": False},
            {"hf_id": GEMMA31, "legacy_smoke_only": False},
            {"hf_id": GEMMA26, "legacy_smoke_only": False, "model_path": str(model_path)},
        ],
    }


def _manifest_rows(count: int = 6) -> list[dict[str, Any]]:
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "dataset": "FEVER",
                "stable_id": f"exp2886-fixture-{index:02d}",
                "prompt": f"Evidence sentence {index}.",
                "claim": f"Fixture claim {index}.",
                "label_text": labels[index % len(labels)],
            }
        )
    return rows


def _prepare_inputs(tmp_path: Path, *, clean: bool = True) -> Path:
    model_path = tmp_path / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("tiny gguf fixture", encoding="utf-8")
    _write_json(tmp_path / "results" / mod.EXP2874_FILENAME, _exp2874_payload(model_path, clean=clean))
    _write_jsonl(tmp_path / "data" / "eval_manifests" / "fever_20260522.jsonl", _manifest_rows())
    return model_path


def _telemetry_ok(**extra: Any) -> dict[str, Any]:
    row = {
        "response_text": "SUPPORTS",
        "logprobs_available": True,
        "substitute_telemetry_used": False,
        "token_logprobs": [math.log(0.9)],
        "tokens": ["SUPPORTS"],
        "top_logprobs": [{"SUPPORTS": math.log(0.9), "REFUTES": math.log(0.1)}],
        "telemetry_source": "injected_logprobs",
    }
    row.update(extra)
    return row


def _other_answer(expected: str) -> str:
    for answer in ("SUPPORTS", "REFUTES", "UNKNOWN"):
        if answer != expected:
            return answer
    raise AssertionError("unreachable")


def _panel_runner(*, with_logprobs: bool) -> mod.PanelRunnerFn:
    def run(
        *,
        model_spec: dict[str, Any],
        examples: list[Any],
        random_seed: int,
        max_tokens: int = mod.DEFAULT_MAX_TOKENS,
    ) -> list[dict[str, Any]]:
        del random_seed, max_tokens
        rows: list[dict[str, Any]] = []
        for index, example in enumerate(examples):
            correct = index % 2 == 0
            answer = example.expected_answer if correct else _other_answer(example.expected_answer)
            row: dict[str, Any] = {
                "example_id": example.example_id,
                "model_hf_id": model_spec["hf_id"],
                "model_path": model_spec["model_path"],
                "response_text": answer,
                "tokens_generated": 1,
                "duration_s": 0.05,
            }
            if with_logprobs:
                probability = 0.9 if correct else 0.1
                row.update(
                    {
                        "tokens": [answer],
                        "token_logprobs": [math.log(probability)],
                        "top_logprobs": [
                            {answer: math.log(probability), "other": math.log(1.0 - probability)}
                        ],
                    }
                )
            rows.append(row)
        return rows

    return run


def _fake_gpu_memory() -> dict[str, Any]:
    return {
        "available": True,
        "gpus": [
            {"index": 0, "memory_used_mib": 5, "memory_free_mib": 24121, "memory_total_mib": 24126},
        ],
    }


def _fake_adversarial_clean(path: Path) -> dict[str, Any]:
    return {"artifact": str(path), "loaded": True, "flags": [], "flag_count": 0}


def _config(tmp_path: Path, output_name: str = mod.OUTPUT_FILENAME) -> ExperimentConfig:
    return ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / output_name,
        run_date="20260522",
        n_prompts=6,
        max_tokens=24,
        started_at=10.0,
        clock=lambda: 75.0,
        tests_run=[
            "pytest tests/python/test_experiment_2886_sota_micro_panel_clean_telemetry_v3.py"
        ],
    )


def test_scenario_infer_sota_017_clean_panel_records_provenance(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-017-001: clean telemetry survives adversarial verify."""
    _prepare_inputs(tmp_path)

    artifact = mod.run_experiment(
        _config(tmp_path),
        telemetry_probe_fn=lambda **_: _telemetry_ok(),
        panel_runner_fn=_panel_runner(with_logprobs=True),
        gpu_memory_fn=_fake_gpu_memory,
        adversarial_verify_fn=_fake_adversarial_clean,
    )

    on_disk = json.loads((_config(tmp_path).output_path).read_text(encoding="utf-8"))
    assert on_disk == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["micro_panel_clean"] is True
    assert artifact["micro_panel_downgraded_to_non_benchmark"] is False
    assert artifact["blocked_reason"] == ""
    assert artifact["selected_model_hf_id"] == GEMMA26
    assert artifact["selected_model_fingerprint"]
    assert artifact["cached_sota_pair_returned_two_loadable_specs"] is False
    assert artifact["n_prompts"] == 6
    assert artifact["n_nonempty_responses"] == 6
    assert artifact["logprobs_available"] is True
    assert artifact["adversarial_verify_invoked"] is True
    assert artifact["adversarial_verify_result"]["flag_count"] == 0
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert isinstance(artifact["reproducibility_checksum"], str)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["gpu_memory_evidence"]["before_panel"]["available"] is True
    assert artifact["gpu_memory_evidence"]["after_panel"]["available"] is True
    assert artifact["duration_s"] == pytest.approx(65.0)
    assert artifact["auroc_if_computable"] == pytest.approx(1.0)
    assert artifact["benchmark_claim_made"] is False


def test_scenario_infer_sota_017_missing_telemetry_downgrades(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-017-002: missing telemetry produces a downgraded note."""
    _prepare_inputs(tmp_path)

    artifact = mod.run_experiment(
        _config(tmp_path, "downgrade.json"),
        telemetry_probe_fn=lambda **_: _telemetry_ok(),
        panel_runner_fn=_panel_runner(with_logprobs=False),
        gpu_memory_fn=_fake_gpu_memory,
        adversarial_verify_fn=_fake_adversarial_clean,
    )

    assert artifact["micro_panel_clean"] is False
    assert artifact["micro_panel_downgraded_to_non_benchmark"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["blocked_reason"] == "downgraded_logprobs_unavailable_non_benchmark_telemetry_note"
    assert artifact["benchmark_claim_made"] is False
    assert artifact["n_prompts"] == 6
    assert artifact["logprobs_available"] is False


def test_scenario_infer_sota_017_precondition_failure_blocks(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-017-003: precondition failure blocks before panel."""
    _prepare_inputs(tmp_path, clean=False)

    artifact = mod.run_experiment(
        _config(tmp_path, "blocked.json"),
        telemetry_probe_fn=lambda **_: pytest.fail("telemetry probe must not run"),
        panel_runner_fn=lambda **_: pytest.fail("panel must not run"),
        gpu_memory_fn=_fake_gpu_memory,
        adversarial_verify_fn=_fake_adversarial_clean,
    )

    assert artifact["honest_verdict"] == "blocked_exp2874_sota_runtime_not_clean"
    assert artifact["blocked_reason"] == "blocked_exp2874_sota_runtime_not_clean"
    assert artifact["micro_panel_clean"] is False
    assert artifact["micro_panel_downgraded_to_non_benchmark"] is False
    assert artifact["n_prompts"] == 0
    assert artifact["adversarial_verify_invoked"] is True
    assert artifact["reproducibility_checksum"]


def test_req_infer_sota_017_probe_missing_logprobs_blocks(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-017: probe without logprobs blocks before panel run."""
    _prepare_inputs(tmp_path)

    artifact = mod.run_experiment(
        _config(tmp_path, "probe-blocked.json"),
        telemetry_probe_fn=lambda **_: {
            "response_text": "SUPPORTS",
            "token_logprobs": [],
            "substitute_telemetry_used": False,
            "blocked_reason": "blocked_logprobs_unavailable",
        },
        panel_runner_fn=lambda **_: pytest.fail("panel must not run without telemetry"),
        gpu_memory_fn=_fake_gpu_memory,
        adversarial_verify_fn=_fake_adversarial_clean,
    )

    assert artifact["honest_verdict"] == "blocked_logprobs_unavailable"
    assert artifact["blocked_reason"] == "blocked_logprobs_unavailable"
    assert artifact["preconditions_checked"][-1]["resource"] == "llama_cpp_logprob_or_substitute_telemetry"
    assert artifact["preconditions_checked"][-1]["available"] is False


def test_req_infer_sota_017_missing_manifest_blocks(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-017: zero panel rows raises an insufficient-rows block."""
    model_path = tmp_path / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("tiny gguf fixture", encoding="utf-8")
    _write_json(tmp_path / "results" / mod.EXP2874_FILENAME, _exp2874_payload(model_path))

    artifact = mod.run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / "no-manifest.json",
            run_date="20260522",
            n_prompts=4,
            max_tokens=16,
            started_at=1.0,
            clock=lambda: 2.0,
            manifest_paths=(Path("data/eval_manifests/missing.jsonl"),),
        ),
        telemetry_probe_fn=lambda **_: _telemetry_ok(),
        panel_runner_fn=lambda **_: pytest.fail("panel must not run without labels"),
        gpu_memory_fn=_fake_gpu_memory,
        adversarial_verify_fn=_fake_adversarial_clean,
    )

    assert artifact["honest_verdict"] == "blocked_insufficient_micro_panel_rows"
    assert artifact["n_prompts"] == 0


def test_req_infer_sota_017_helpers_and_defaults(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-017: helper functions return well-typed provenance."""
    missing = mod._model_fingerprint(tmp_path / "missing.gguf")
    assert missing.startswith("missing:")

    real = tmp_path / "tiny.gguf"
    real.write_text("fixture", encoding="utf-8")
    fingerprint = mod._model_fingerprint(real)
    assert "size_bytes=" in fingerprint
    assert "resolved_path=" in fingerprint

    sha_named = tmp_path / ("a" * 64)
    sha_named.write_text("hex-named blob", encoding="utf-8")
    assert mod._model_fingerprint(sha_named).startswith("sha256:")

    checksum_a = mod._reproducibility_checksum(
        selected_model_hf_id=GEMMA26,
        selected_model_path=str(real),
        fingerprint=fingerprint,
        panel_prompts=["alpha", "beta"],
        random_seed=2886,
        max_tokens=24,
    )
    checksum_b = mod._reproducibility_checksum(
        selected_model_hf_id=GEMMA26,
        selected_model_path=str(real),
        fingerprint=fingerprint,
        panel_prompts=["alpha", "beta", "gamma"],
        random_seed=2886,
        max_tokens=24,
    )
    assert checksum_a != checksum_b
    assert len(checksum_a) == 64

    default_config = ExperimentConfig(repo_root=tmp_path)
    assert default_config.resolved_output_path() == tmp_path / "results" / mod.OUTPUT_FILENAME
    assert default_config.resolved_exp2874_path() == tmp_path / "results" / mod.EXP2874_FILENAME


def test_req_infer_sota_017_write_false_still_invokes_verifier(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-017: write=False still embeds the adversarial verify result."""
    _prepare_inputs(tmp_path)
    captured: list[Path] = []

    def fake_verify(path: Path) -> dict[str, Any]:
        captured.append(Path(path))
        return {"flags": [], "flag_count": 0}

    artifact = mod.run_experiment(
        _config(tmp_path, "no-write.json"),
        telemetry_probe_fn=lambda **_: _telemetry_ok(),
        panel_runner_fn=_panel_runner(with_logprobs=True),
        gpu_memory_fn=_fake_gpu_memory,
        adversarial_verify_fn=fake_verify,
        write=False,
    )

    assert captured == [_config(tmp_path, "no-write.json").output_path]
    assert artifact["adversarial_verify_invoked"] is True
    assert not (_config(tmp_path, "no-write.json").output_path).exists()
