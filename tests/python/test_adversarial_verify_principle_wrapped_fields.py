"""Tests for adversarial_verify.py's principle-wrapped field normalization.

Origin: 2026-07-02. exp5161's GAP-4 pilot was HARD-flagged DURATION_TOO_SHORT for a reason
unrelated to its actual duration: its `inference_substrate` field was written as
`{"principle": "...", "value": "verifier_ensemble_against_cached_candidates"}` (the CLAUDE.md
"Principle-Annotated Artifact Fields" convention), but `_inference_substrate_text()` did
`str(d.get("inference_substrate"))`, stringifying a DICT to a Python repr that matches no
canonical substrate string -- so the duration-floor check fell through to the generic 60s
compute-bound fallback regardless of what substrate was actually declared.

A corpus-wide scan at fix time found 176 artifacts wrap `inference_substrate` this way, plus
smaller counts for `honest_verdict` (9), `duration_s` (12), `random_seed` (14), and
`reproducibility_checksum` (14) -- every field read via a bare `d.get(...)` anywhere in this
file's checks was exposed to the same class of silent misbehavior. Fixed via
`_normalize_principle_wrapped_fields`, applied once in `verify_artifact` immediately after
`_flatten_metrics`, so every check (existing and future) sees the bare value automatically.

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


class TestNormalizePrincipleWrappedFields:
    """Unit tests for the pure normalization function."""

    def test_unwraps_value_principle_dict(self) -> None:
        d = {
            "inference_substrate": {"principle": "why this matters", "value": "live_llm_inference"}
        }
        out = av._normalize_principle_wrapped_fields(d)
        assert out["inference_substrate"] == "live_llm_inference"

    def test_leaves_bare_string_unchanged(self) -> None:
        d = {"inference_substrate": "live_llm_inference"}
        out = av._normalize_principle_wrapped_fields(d)
        assert out["inference_substrate"] == "live_llm_inference"

    def test_leaves_dict_without_principle_key_unchanged(self) -> None:
        """A dict with 'value' but no 'principle' is NOT the wrapping convention --
        could be genuine domain data for some other field. Must not be touched."""
        d = {"model_specs": {"value": "gemma-4-12B", "quantization": "Q4_K_M"}}
        out = av._normalize_principle_wrapped_fields(d)
        assert out["model_specs"] == {"value": "gemma-4-12B", "quantization": "Q4_K_M"}

    def test_leaves_dict_without_value_key_unchanged(self) -> None:
        d = {"model_specs": {"principle": "must name the model", "model": "gemma-4-12B"}}
        out = av._normalize_principle_wrapped_fields(d)
        assert out["model_specs"] == {"principle": "must name the model", "model": "gemma-4-12B"}

    def test_does_not_recurse_into_nested_structures(self) -> None:
        """Only top-level fields are unwrapped -- nested dicts (e.g. inside model_specs)
        keep their own shape even if they happen to have value+principle keys."""
        d = {
            "model_specs": {
                "generator": {"principle": "nested, not top-level", "value": "should stay wrapped"}
            }
        }
        out = av._normalize_principle_wrapped_fields(d)
        assert out["model_specs"]["generator"] == {
            "principle": "nested, not top-level",
            "value": "should stay wrapped",
        }

    def test_unwraps_numeric_fields_too(self) -> None:
        """The convention wraps ANY field type, not just strings -- duration_s, random_seed,
        etc. are commonly numbers."""
        d = {
            "duration_s": {"principle": "real compute takes wall-clock time", "value": 5.59},
            "random_seed": {"principle": "determinism", "value": 5161},
        }
        out = av._normalize_principle_wrapped_fields(d)
        assert out["duration_s"] == 5.59
        assert out["random_seed"] == 5161

    def test_preserves_all_other_fields(self) -> None:
        d = {"experiment": "exp1", "n": 40, "flat": "already bare"}
        out = av._normalize_principle_wrapped_fields(d)
        assert out == d


class TestExp5161IncidentReproduction:
    """End-to-end reproduction of the exact exp5161 false-positive, verified fixed."""

    def test_wrapped_verifier_scoring_substrate_no_longer_flags_duration_too_short(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "experiment": "exp5161_repro",
            "honest_verdict": "complete_gap4_pilot_n60_direction_replicated",
            "inference_substrate": {
                "principle": "Substrate honesty: this task invokes live Codex/LLM calls for the sandbox smoke.",
                "value": "verifier_ensemble_against_cached_candidates",
            },
            "duration_s": 5.58995,
            "random_seed": 5161,
            "reproducibility_checksum": "sha256:" + "0" * 64,
            "model_specs": {"local_generator_arm_cache_check": "unsloth/gemma-4-12B-it-GGUF"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_same_payload_bare_string_substrate_also_clean(self, tmp_path: Path) -> None:
        """Sanity: the bare-string form (the common case) was never broken -- confirms the
        fix doesn't regress the already-working path."""
        payload = {
            "experiment": "exp5161_repro_bare",
            "honest_verdict": "complete_gap4_pilot_n60_direction_replicated",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "duration_s": 5.58995,
            "random_seed": 5161,
            "reproducibility_checksum": "sha256:" + "0" * 64,
            "model_specs": {"local_generator_arm_cache_check": "unsloth/gemma-4-12B-it-GGUF"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_wrapped_live_llm_inference_substrate_still_enforces_60s_floor(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: unwrapping must not accidentally WEAKEN the check -- a
        wrapped live_llm_inference substrate with an implausibly short duration must
        still be caught, exactly as the bare-string form already is."""
        payload = {
            "experiment": "exp_genuinely_too_fast",
            "honest_verdict": "complete_fake_gguf_run",
            "inference_substrate": {
                "principle": "loads a real GGUF model",
                "value": "live_llm_inference",
            },
            "duration_s": 3.4,
            "model_specs": {"model": "some-35B-GGUF"},
            "random_seed": 1,
            "reproducibility_checksum": "sha256:" + "0" * 64,
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)
