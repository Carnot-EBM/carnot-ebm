"""Tests for Exp 3043 transcript fingerprint replay preflight.

Spec: REQ-INFER-SOTA-022,
      SCENARIO-INFER-SOTA-022-001,
      SCENARIO-INFER-SOTA-022-002,
      SCENARIO-INFER-SOTA-022-003
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_3043_verified_speculation_transcript_fingerprint as exp3043


MANDATED_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"


class DeterministicFakeLlama:
    """Small callable stand-in for llama.cpp that returns stable text per prompt."""

    instances: list["DeterministicFakeLlama"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self.instances.append(self)

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((prompt, kwargs))
        prompt_digest = exp3043.sha256_text(prompt)[:12]
        return {
            "choices": [{"text": f"repair fingerprint response {prompt_digest}"}],
            "usage": {"completion_tokens": 4},
        }

    def close(self) -> None:
        self.closed = True


class DivergentFakeLlama:
    """Small callable stand-in for a runtime whose repeated outputs diverge."""

    def __init__(self, **kwargs: Any) -> None:
        self.count = 0

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.count += 1
        return {
            "choices": [{"text": f"response {self.count} for {exp3043.sha256_text(prompt)[:8]}"}],
            "usage": {"completion_tokens": 3},
        }

    def close(self) -> None:
        return None


class FailingFakeLlama:
    """Small callable stand-in for a model load/generation failure."""

    def __init__(self, **kwargs: Any) -> None:
        return None

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")


def _config(tmp_path: Path, model_path: Path) -> exp3043.ExperimentConfig:
    return exp3043.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "result.json",
        raw_dir=tmp_path / "raw",
        prompts=[
            "Repair prompt A: fix clamp_score lower-bound behavior.",
            "Repair prompt B: keep first-seen order in unique_preserve_order.",
            "Repair prompt C: count vowels case-insensitively.",
        ],
        model_checksum_full_limit_bytes=1024,
        selected_model_path_for_tests=model_path,
    )


def _resolver(model_path: Path):
    def resolve(
        hf_id: str, preferred_quant: str = "Q4_K_M", cache_root: str | None = None
    ) -> str | None:
        if hf_id == MANDATED_MODEL:
            return str(model_path)
        return None

    return resolve


def test_missing_sota_gguf_blocks_without_legacy_headline(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-022-003: missing mandated cache writes blocked artifact."""

    config = exp3043.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "result.json",
        raw_dir=tmp_path / "raw",
    )
    artifact = exp3043.build_artifact(
        config,
        cached_pair_func=lambda **_: None,
        resolve_gguf_func=lambda *_, **__: None,
        llama_factory=lambda **_: DeterministicFakeLlama(),
        monotonic=iter([10.0, 11.25]).__next__,
    )

    assert artifact["fingerprint_live_ready"] is False
    assert artifact["deterministic_replay_passed"] is False
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["models_used"] == []
    assert artifact["n_prompts"] == 0
    assert artifact["honest_verdict"].startswith("blocked_sota_gguf_unavailable")
    assert artifact["inference_substrate"]["gguf_cache_resolution"][MANDATED_MODEL] is None


def test_deterministic_replay_artifact_contains_required_hashes(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-022 / SCENARIO-INFER-SOTA-022-001: stable replay opens gate."""

    model_path = tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.write_text("small fake gguf bytes", encoding="utf-8")
    config = _config(tmp_path, model_path)

    artifact = exp3043.build_artifact(
        config,
        cached_pair_func=lambda **_: None,
        resolve_gguf_func=_resolver(model_path),
        llama_factory=DeterministicFakeLlama,
        monotonic=iter([20.0, 22.5]).__next__,
        repo_commit_func=lambda root: "abc123",
    )

    assert artifact["fingerprint_live_ready"] is True
    assert artifact["deterministic_replay_passed"] is True
    assert artifact["models_used"] == [MANDATED_MODEL]
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["n_prompts"] == 3
    assert len(artifact["prompt_hashes"]) == 3
    assert len(artifact["output_hashes"]) == 3
    assert all(
        row["run_1_raw_output_hash"] == row["run_2_raw_output_hash"]
        for row in artifact["output_hashes"]
    )
    assert all(
        row["run_1_normalized_output_hash"] == row["run_2_normalized_output_hash"]
        for row in artifact["output_hashes"]
    )
    assert len(artifact["batch_context_hash"]) == 64
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["model_specs"][0]["model_hash_or_cache_path"].startswith("sha256:")
    assert artifact["inference_substrate"]["repo_commit"] == "abc123"
    assert artifact["inference_substrate"]["wall_clock_duration_s"] == 2.5
    assert artifact["honest_verdict"].startswith("complete:")

    fake = DeterministicFakeLlama.instances[-1]
    assert fake.kwargs["model_path"] == str(model_path)
    assert fake.closed is True
    assert fake.calls[0][1]["seed"] == config.seed


def test_replay_divergence_is_recorded_without_ready_claim(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-022-002: divergent repeated hashes stay explicit."""

    model_path = tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.write_text("small fake gguf bytes", encoding="utf-8")
    config = _config(tmp_path, model_path)

    artifact = exp3043.build_artifact(
        config,
        cached_pair_func=lambda **_: [
            {
                "hf_id": MANDATED_MODEL,
                "model_path": str(model_path),
                "gpu": 0,
                "name": "Gemma4-26B-A4B-it",
            }
        ],
        resolve_gguf_func=_resolver(model_path),
        llama_factory=DivergentFakeLlama,
        monotonic=iter([1.0, 4.0]).__next__,
    )

    assert artifact["fingerprint_live_ready"] is False
    assert artifact["deterministic_replay_passed"] is False
    assert len(artifact["replay_divergences"]) == 3
    assert artifact["honest_verdict"].startswith("complete_replay_diverged:")


def test_write_artifact_persists_sorted_json(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-022: terminal artifact is written at the configured path."""

    model_path = tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.write_text("small fake gguf bytes", encoding="utf-8")
    config = _config(tmp_path, model_path)

    artifact = exp3043.write_artifact(
        config,
        cached_pair_func=lambda **_: None,
        resolve_gguf_func=_resolver(model_path),
        llama_factory=DeterministicFakeLlama,
        monotonic=iter([30.0, 31.0]).__next__,
    )

    loaded = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert loaded == artifact
    assert loaded["schema"] == exp3043.SCHEMA


def test_runtime_failure_blocks_with_explicit_blocker(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-022: generation failures are terminal blockers, not fake replays."""

    model_path = tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.write_text("small fake gguf bytes", encoding="utf-8")
    config = _config(tmp_path, model_path)

    artifact = exp3043.build_artifact(
        config,
        cached_pair_func=lambda **_: None,
        resolve_gguf_func=_resolver(model_path),
        llama_factory=FailingFakeLlama,
        monotonic=iter([40.0, 41.0]).__next__,
    )

    assert artifact["status"] == "blocked"
    assert artifact["fingerprint_live_ready"] is False
    assert artifact["runtime_blocker"] == "RuntimeError: boom"
    assert artifact["honest_verdict"].startswith("blocked_sota_gguf_unavailable:")
    assert artifact["raw_transcript_paths"] == []


def test_helper_fallbacks_and_alternate_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFER-SOTA-022: helper branches preserve auditable fallback metadata."""

    model_path = tmp_path / "fallback.gguf"
    model_path.write_text("tiny", encoding="utf-8")
    config = exp3043.ExperimentConfig(
        repo_root=tmp_path,
        selected_model_path_for_tests=model_path,
        decode_config={"max_tokens": 7},
        load_config={"main_gpu": 1},
    )

    resolved = exp3043._resolve_cache(lambda *_, **__: None, config)
    assert resolved[exp3043.MANDATED_MODEL_IDS[0]] == str(model_path)
    assert config.effective_decode_config()["max_tokens"] == 7
    assert config.effective_load_config()["main_gpu"] == 1

    missing = exp3043._file_evidence(tmp_path / "missing.gguf", full_limit_bytes=1)
    assert missing["exists"] is False

    big_path = tmp_path / "large.gguf"
    big_path.write_bytes(b"x" * (exp3043.BOUNDED_HASH_BYTES + 3))
    bounded = exp3043._file_evidence(big_path, full_limit_bytes=1)
    assert bounded["model_hash_or_cache_path"].startswith("bounded_sha256:")

    assert exp3043._int_or_none("not-an-int") is None
    assert exp3043._extract_text("plain") == "plain"
    assert exp3043._extract_text(123) == ""
    assert exp3043._extract_text({}) == ""
    assert exp3043._extract_text({"choices": [5]}) == ""
    assert exp3043._extract_text({"choices": [{"message": {"content": "hello"}}]}) == "hello"
    assert exp3043._extract_text({"choices": [{"finish_reason": "stop"}]}) == ""
    assert exp3043._write_transcripts(config, []) == []

    class FakeCompleted:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(
        exp3043.subprocess,
        "run",
        lambda *args, **kwargs: FakeCompleted(1, stdout="", stderr="probe failed"),
    )
    assert exp3043._cuda_probe()["torch_error"] == "probe failed"
    assert exp3043._gpu_inventory()["error"] == "probe failed"

    monkeypatch.setattr(
        exp3043.subprocess,
        "run",
        lambda *args, **kwargs: FakeCompleted(
            0,
            stdout="malformed\n0, GPU, 10, 20, 1, 595.71.05\n",
        ),
    )
    inventory = exp3043._gpu_inventory()
    assert inventory["available"] is True
    assert inventory["free_vram_mib_total"] == 10
