"""Tests for Exp 2399 FST live PATH A/B/C reporting.

Spec: REQ-FST-2399, SCENARIO-FST-2399.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import fst_live_path_ab as mod


class _FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, prompt: str, **kwargs: Any) -> dict:
        assert prompt
        assert kwargs["max_tokens"] == 64
        return {"choices": [{"text": " 1"}]}

    def close(self) -> None:
        self.closed = True


def _fake_cache(root: Path) -> Path:
    snapshot = root / "models--unsloth--gemma-4-26B-A4B-it-GGUF" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf").write_bytes(b"fake gguf")
    (snapshot / "mmproj-F16.gguf").write_bytes(b"not the language model")
    return root


def _telemetry(path: Path) -> Path:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "case_id": f"cached_{index}",
                    "prompt": f"Verify claim: {index} + 1 = {index + 1}.",
                    "response_text": "1",
                }
            )
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_req_fst_2399_attempt_path_a_runs_fake_llama_through_fst(tmp_path: Path) -> None:
    """REQ-FST-2399: PATH A loads a mandated GGUF path before FST validation."""

    gguf = _fake_cache(tmp_path / "hf") / (
        "models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/abc123/"
        "gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf"
    )
    preconditions = {
        "gguf_cache_check": {"available": True},
        "llama_cpp_check": {"available": True},
        "resolved_gguf_models": [
            {
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "path": str(gguf),
                "filename": gguf.name,
                "size_bytes": gguf.stat().st_size,
            }
        ],
    }

    attempt = mod.attempt_path_a(
        preconditions=preconditions,
        n_test_prompts=3,
        llama_factory=_FakeLlama,
    )

    assert attempt.attempted is True
    assert attempt.success is True
    assert attempt.path_used == "A_gguf"
    assert (
        attempt.model_used == "unsloth/gemma-4-26B-A4B-it-GGUF:gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf"
    )
    assert attempt.first_text == "1"
    assert len(attempt.rows) == 3
    assert all(row["fst_terminal_prefix_present"] for row in attempt.rows)


def test_scenario_fst_2399_cached_telemetry_fallback_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-FST-2399: PATH C writes a complete artifact when forced."""

    output = tmp_path / mod.OUTPUT_FILE
    telemetry = _telemetry(tmp_path / "telemetry.jsonl")

    artifact = mod.run_experiment(
        output_path=output,
        telemetry_path=telemetry,
        cache_root=tmp_path / "empty_hf",
        n_test_prompts=3,
        force_path_c=True,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["fst_live_validated"] is True
    assert artifact["live_path_used"] == "C_cached"
    assert artifact["model_used"] is None
    assert artifact["path_a_attempted"] is False
    assert artifact["path_a_blocked_reason"] == "forced_path_c"
    assert artifact["path_b_attempted"] is False
    assert artifact["n_test_prompts"] == 3
    assert artifact["first_live_generation_text"] == "1"


def test_req_fst_2399_preconditions_resolve_fake_gguf_and_telemetry(tmp_path: Path) -> None:
    """REQ-FST-2399: preconditions record GGUF cache and PATH C telemetry checks."""

    cache = _fake_cache(tmp_path / "hf")
    telemetry = _telemetry(tmp_path / "telemetry.jsonl")

    preconditions = mod.check_preconditions(cache_root=cache, telemetry_path=telemetry)

    assert preconditions["gguf_cache_check"]["available"] is True
    assert preconditions["path_c_telemetry_check"]["exists"] is True
    assert preconditions["resolved_gguf_models"][0]["hf_id"] == ("unsloth/gemma-4-26B-A4B-it-GGUF")
    assert preconditions["resolved_gguf_models"][0]["filename"].endswith("UD-IQ2_XXS.gguf")


def test_req_fst_2399_validation_rejects_missing_terminal_prefix(tmp_path: Path) -> None:
    """REQ-FST-2399: validation enforces terminal-prefix FST evidence."""

    telemetry = _telemetry(tmp_path / "telemetry.jsonl")
    artifact = mod.run_experiment(
        output_path=tmp_path / mod.OUTPUT_FILE,
        telemetry_path=telemetry,
        cache_root=tmp_path / "empty_hf",
        force_path_c=True,
    )
    artifact["fst_rows"][0]["fst_terminal_prefix_present"] = False

    try:
        mod.validate_artifact(artifact)
    except AssertionError as exc:
        assert "terminal prefix" in str(exc)
    else:  # pragma: no cover - defensive assertion for clearer failures.
        raise AssertionError("expected validation to reject missing FST prefix")
