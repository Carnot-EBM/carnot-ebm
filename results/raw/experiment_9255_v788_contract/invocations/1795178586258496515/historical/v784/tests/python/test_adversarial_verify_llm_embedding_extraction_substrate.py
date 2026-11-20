"""Tests for adversarial_verify.py's live_llm_embedding_extraction substrate category.

Origin: 2026-07-03. exp5178's hidden-state verifier pilot (results/experiment_5178_
hidden_state_verifier_pilot_v474.json) loaded a real local GGUF model
(gemma-4-26B-A4B-it, Q4_K_M) via `llama_cpp.Llama(embedding=True, pooling_type=LAST).embed`
to extract final-token hidden-state vectors for 48 candidates across 6 questions, then
trained a small centroid probe. Total wall-clock: 35.28s -- plausible for model load
(~10-15s for a 26B quantized model) plus 48 single-pass embedding extractions (no
iterative token decoding), implausible under the generic 60s live_llm_inference floor
(calibrated for full generative inference).

The project's substrate taxonomy (CLAUDE.md "Inference-Substrate Declaration Discipline")
previously had 5 legal values, none of which fit: live_llm_inference's 60s floor assumes
full generation; verifier_ensemble_against_cached_candidates's 1s floor explicitly
requires the LLM NOT be loaded at all (this workload DOES load it, just for embeddings).
Added live_llm_embedding_extraction as the 6th legal value with a 2.0s floor -- enough
to catch a genuinely fabricated near-instant claim, permissive enough for real embedding-
only work even on the smallest SOTA GGUF models.

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


class TestIsLlmEmbeddingExtraction:
    def test_recognizes_canonical_substrate_value(self) -> None:
        d = {"inference_substrate": "live_llm_embedding_extraction"}
        assert av._is_llm_embedding_extraction(d) is True

    def test_recognizes_canonical_value_with_trailing_note(self) -> None:
        d = {"inference_substrate": "live_llm_embedding_extraction; embedding-only, no decode"}
        assert av._is_llm_embedding_extraction(d) is True

    def test_does_not_match_plain_live_llm_inference(self) -> None:
        d = {"inference_substrate": "live_llm_inference"}
        assert av._is_llm_embedding_extraction(d) is False

    def test_does_not_match_missing_field(self) -> None:
        assert av._is_llm_embedding_extraction({}) is False


class TestDurationFloorForArtifact:
    def test_returns_the_2s_floor(self) -> None:
        d = {"inference_substrate": "live_llm_embedding_extraction"}
        floor = av.duration_floor_for_artifact(d)
        assert floor == {
            "substrate": "live_llm_embedding_extraction",
            "min_duration_s": 2.0,
            "reason": "llm_embedding_extraction",
        }

    def test_takes_priority_over_generic_compute_bound_marker(self) -> None:
        """A GGUF path in model_specs must not pull this back to the 60s
        live_llm_inference floor once the embedding-extraction substrate is declared."""
        d = {
            "inference_substrate": "live_llm_embedding_extraction",
            "model_specs": {"gguf_path": "/some/path/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"},
        }
        floor = av.duration_floor_for_artifact(d)
        assert floor["min_duration_s"] == 2.0


class TestExp5178IncidentReproduction:
    """End-to-end reproduction of the exp5178 case, verified fixed with the corrected substrate."""

    def test_35s_embedding_extraction_run_no_longer_flags_duration_too_short(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "experiment": "exp5178_repro",
            "honest_verdict": "complete_hidden_state_verifier_ties_tuned_sc",
            "inference_substrate": "live_llm_embedding_extraction",
            "duration_s": 35.279684,
            "random_seed": 5178,
            "reproducibility_checksum": "sha256:" + "0" * 64,
            "model_specs": {"gguf_path": "/some/path/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_genuinely_too_fast_embedding_claim_is_still_caught(self, tmp_path: Path) -> None:
        """Regression guard: the new floor must not be so permissive it lets a
        fabricated near-instant embedding-extraction claim through."""
        payload = {
            "experiment": "exp_fabricated_embedding",
            "honest_verdict": "complete_fake_embedding_run",
            "inference_substrate": "live_llm_embedding_extraction",
            "duration_s": 0.5,
            "random_seed": 1,
            "reproducibility_checksum": "sha256:" + "0" * 64,
            "model_specs": {"gguf_path": "/some/path/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)

    def test_full_generation_claim_still_needs_the_60s_floor(self, tmp_path: Path) -> None:
        """Regression guard: declaring the new substrate must not become a loophole for
        artifacts that actually claim full generative inference -- only artifacts that
        genuinely declare embedding-extraction get the lower floor."""
        payload = {
            "experiment": "exp_full_generation",
            "honest_verdict": "complete_full_generation_run",
            "inference_substrate": "live_llm_inference",
            "duration_s": 5.0,
            "random_seed": 1,
            "reproducibility_checksum": "sha256:" + "0" * 64,
            "model_specs": {"gguf_path": "/some/path/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)
