"""Tests for adversarial_verify.py's deterministic_smt_hint_validation_no_llm substrate.

Origin: 2026-07-06. exp5318 (results/experiment_5318_smt_hint_validation_protocol_v485.json)
declared inference_substrate=deterministic_smt_hint_validation_no_llm -- explicitly naming itself
LLM-free -- and validated SMT hints deterministically (valid_hint_acceptance_rate=1.0, a real,
non-suspicious result for a formal solver check) in duration_s=0.03. Got flagged
DURATION_TOO_SHORT under the generic 60s live_llm_inference floor because no substrate category
recognized it, the sixth time this exact bug class has appeared this week. Mirrors the same
pattern as the five prior additions: a genuine, LLM-free substrate whose own name discloses the
gap, needing its own near-zero floor rather than the compute-bound fallback.

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


class TestIsDeterministicSmtHintValidation:
    def test_recognizes_canonical_substrate_value(self) -> None:
        d = {"inference_substrate": "deterministic_smt_hint_validation_no_llm"}
        assert av._is_deterministic_smt_hint_validation(d) is True

    def test_does_not_match_generic_live_llm_inference(self) -> None:
        d = {"inference_substrate": "live_llm_inference"}
        assert av._is_deterministic_smt_hint_validation(d) is False

    def test_does_not_match_missing_field(self) -> None:
        assert av._is_deterministic_smt_hint_validation({}) is False


class TestDurationFloorForArtifact:
    def test_returns_the_near_zero_floor(self) -> None:
        d = {"inference_substrate": "deterministic_smt_hint_validation_no_llm"}
        floor = av.duration_floor_for_artifact(d)
        assert floor == {
            "substrate": "deterministic_smt_hint_validation_no_llm",
            "min_duration_s": 0.0001,
            "reason": "deterministic_smt_hint_validation",
        }


class TestExp5318IncidentReproduction:
    def test_003s_hint_validation_no_longer_flags_duration_too_short(self, tmp_path: Path) -> None:
        payload = {
            "experiment": "exp5318_repro",
            "honest_verdict": "complete: deterministic SMT hint validation protocol is ready",
            "inference_substrate": "deterministic_smt_hint_validation_no_llm",
            "duration_s": 0.029515,
            "valid_hint_acceptance_rate": 1.0,
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_model_specs_not_required_for_this_substrate(self, tmp_path: Path) -> None:
        """This substrate genuinely has no model to name -- matches the ARC-no-LLM and
        log-analysis exemption pattern."""
        payload = {
            "experiment": "exp5318_repro2",
            "honest_verdict": "complete: deterministic SMT hint validation protocol is ready",
            "inference_substrate": "deterministic_smt_hint_validation_no_llm",
            "duration_s": 0.03,
            "random_seed": 1,
            "reproducibility_checksum": "abc123",
        }
        report = _report_for_payload(tmp_path, payload)
        flags_by_kind = {f["kind"]: f for f in report["flags"]}
        assert "METHODOLOGY_MISSING" not in flags_by_kind or "model_specs" not in flags_by_kind[
            "METHODOLOGY_MISSING"
        ].get("detail", "")

    def test_full_generation_claim_still_needs_the_60s_floor(self, tmp_path: Path) -> None:
        """Regression guard: declaring the new substrate must not become a loophole for
        artifacts that actually claim full generative inference."""
        payload = {
            "experiment": "exp_full_generation",
            "honest_verdict": "complete: full generation run",
            "inference_substrate": "live_llm_inference",
            "duration_s": 5.0,
            "model_specs": {"gguf_path": "/some/path/model.gguf"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)
