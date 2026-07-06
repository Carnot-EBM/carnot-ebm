"""Tests for adversarial_verify.py's local_native_llama_cpp_gguf_backend_bisect substrate.

Origin: 2026-07-06. exp5323 (results/experiment_5323_native_gguf_backend_flag_bisect_v486.json)
resolved the multi-milestone native-llama-cli-hang blocker (see CLAUDE.md's Build Environment
section and the exp5297/exp5309 incidents) by independently bisecting to the -st/--single-turn
flag, confirming completed_load_first_token_and_8_tokens=True with authenticated GPU offload in
20.8s. Got flagged DURATION_TOO_SHORT under the generic 60s live_llm_inference floor because no
substrate category recognized "a bounded bisect that stops at the first working flag combination,
not a full generation benchmark" -- the seventh time this exact bug class has appeared this week.

Spec refs: none (operational lint fix, no OpenSpec capability).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report_for_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return av.verify_artifact(_write_payload(tmp_path, payload))


def _flag_kinds(report: dict[str, Any]) -> set[str]:
    return {flag["kind"] for flag in report["flags"]}


class TestIsNativeGgufBackendBisect:
    def test_recognizes_canonical_substrate_value(self) -> None:
        d = {"inference_substrate": "local_native_llama_cpp_gguf_backend_bisect"}
        assert av._is_native_gguf_backend_bisect(d) is True

    def test_does_not_match_generic_live_llm_inference(self) -> None:
        d = {"inference_substrate": "live_llm_inference"}
        assert av._is_native_gguf_backend_bisect(d) is False

    def test_does_not_match_missing_field(self) -> None:
        assert av._is_native_gguf_backend_bisect({}) is False


class TestDurationFloorForArtifact:
    def test_returns_the_5s_floor(self) -> None:
        d = {"inference_substrate": "local_native_llama_cpp_gguf_backend_bisect"}
        floor = av.duration_floor_for_artifact(d)
        assert floor == {
            "substrate": "local_native_llama_cpp_gguf_backend_bisect",
            "min_duration_s": 5.0,
            "reason": "native_gguf_backend_bisect",
        }


class TestExp5323IncidentReproduction:
    def test_21s_bisect_no_longer_flags_duration_too_short(self, tmp_path: Path) -> None:
        payload = {
            "experiment": "exp5323_repro",
            "honest_verdict": (
                "complete: native_llama_cpp_backend_candidate_ready=flagship_dense:llama-cli"
            ),
            "inference_substrate": "local_native_llama_cpp_gguf_backend_bisect",
            "duration_s": 20.830088,
            "MODEL_SPECS": {"flagship_dense": {"hf_id": "unsloth/gemma-4-31B-it-GGUF"}},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_genuinely_too_fast_claim_is_still_caught(self, tmp_path: Path) -> None:
        payload = {
            "experiment": "exp_fabricated_bisect",
            "honest_verdict": "complete: fake fast bisect result",
            "inference_substrate": "local_native_llama_cpp_gguf_backend_bisect",
            "duration_s": 0.5,
            "MODEL_SPECS": {"flagship_dense": {"hf_id": "unsloth/gemma-4-31B-it-GGUF"}},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)

    def test_full_generation_claim_still_needs_the_60s_floor(self, tmp_path: Path) -> None:
        """Regression guard: declaring the new substrate must not become a loophole for
        artifacts that actually claim full generative inference."""
        payload = {
            "experiment": "exp_full_generation",
            "honest_verdict": "complete: full generation run",
            "inference_substrate": "live_llm_inference",
            "duration_s": 10.0,
            "model_specs": {"gguf_path": "/some/path/model.gguf"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)
