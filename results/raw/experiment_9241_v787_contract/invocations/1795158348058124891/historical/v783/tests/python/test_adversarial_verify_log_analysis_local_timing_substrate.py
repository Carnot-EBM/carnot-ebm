"""Tests for adversarial_verify.py's arc_log_analysis_plus_local_timing substrate category.

Origin: 2026-07-04. exp5215's ARC PAW amortization gate (results/experiment_5215_arc_paw_
amortization_gate_v477.json) read 9 arc_loop_solve_*.json logs and ran one small, local, non-LLM
timing measurement (a tiny CUDA kernel dispatch to sanity-check a compile-cost estimate, clamped to
a conservative floor rather than trusted raw) -- genuine compute, no fabrication, and it landed a
real, decisive negative result (paw_amortization_viable=False: median/p75 remaining actions fall
short of break-even). Total wall-clock 4.10s is plausible for reading several JSON logs plus one
bounded local timing check, but got CRITICAL-flagged under the generic 60s live_llm_inference floor
because no existing substrate category recognized this shape (a "torch_cuda:0" backend string in
the timing evidence pulled it toward the compute-bound-marker fallback).

Mirrors the same pattern as the two prior substrate-category additions this project has made
(verifier_ensemble_against_cached_candidates's exp5161 incident, live_llm_embedding_extraction's
exp5178 incident): rather than force-fitting the artifact into an existing category, add a genuine
new one with its own calibrated floor.

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


class TestIsLogAnalysisLocalTiming:
    def test_recognizes_canonical_substrate_value(self) -> None:
        d = {"inference_substrate": "arc_log_analysis_plus_local_timing"}
        assert av._is_log_analysis_local_timing(d) is True

    def test_recognizes_canonical_value_with_trailing_note(self) -> None:
        d = {
            "inference_substrate": "arc_log_analysis_plus_local_timing; reads logs + one timing check"
        }
        assert av._is_log_analysis_local_timing(d) is True

    def test_does_not_match_plain_live_llm_inference(self) -> None:
        d = {"inference_substrate": "live_llm_inference"}
        assert av._is_log_analysis_local_timing(d) is False

    def test_does_not_match_missing_field(self) -> None:
        assert av._is_log_analysis_local_timing({}) is False


class TestDurationFloorForArtifact:
    def test_returns_the_1s_floor(self) -> None:
        d = {"inference_substrate": "arc_log_analysis_plus_local_timing"}
        floor = av.duration_floor_for_artifact(d)
        assert floor == {
            "substrate": "arc_log_analysis_plus_local_timing",
            "min_duration_s": 1.0,
            "reason": "log_analysis_local_timing",
        }

    def test_takes_priority_over_generic_compute_bound_marker(self) -> None:
        """A CUDA backend string in timing evidence must not pull this back to the
        60s live_llm_inference floor once the substrate is declared."""
        d = {
            "inference_substrate": "arc_log_analysis_plus_local_timing",
            "timing_evidence": {"cheap_step": {"backend": "torch_cuda:0"}},
        }
        floor = av.duration_floor_for_artifact(d)
        assert floor["min_duration_s"] == 1.0


class TestExp5215IncidentReproduction:
    """End-to-end reproduction of the exp5215 case, verified fixed with the corrected substrate."""

    def test_4s_log_analysis_run_no_longer_flags_duration_too_short(self, tmp_path: Path) -> None:
        payload = {
            "experiment": "exp5215_repro",
            "honest_verdict": "complete_paw_amortization_gate_not_viable_no_arc_solve_claim",
            "inference_substrate": "arc_log_analysis_plus_local_timing",
            "duration_s": 4.09946,
            "timing_evidence": {"cheap_step": {"backend": "torch_cuda:0"}},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_genuinely_too_fast_claim_is_still_caught(self, tmp_path: Path) -> None:
        """Regression guard: the new floor must not be so permissive it lets a
        fabricated near-instant claim through. Includes a compute-bound marker
        (matching the real exp5215 shape, timing_evidence.cheap_step.backend) so the
        duration check actually fires -- without any marker present, the check
        early-returns regardless of substrate, per check_duration_vs_claim's own gate."""
        payload = {
            "experiment": "exp_fabricated_log_analysis",
            "honest_verdict": "complete_fake_gate_run",
            "inference_substrate": "arc_log_analysis_plus_local_timing",
            "duration_s": 0.05,
            "timing_evidence": {"cheap_step": {"backend": "torch_cuda:0"}},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)

    def test_full_generation_claim_still_needs_the_60s_floor(self, tmp_path: Path) -> None:
        """Regression guard: declaring the new substrate must not become a loophole for
        artifacts that actually claim full generative inference."""
        payload = {
            "experiment": "exp_full_generation",
            "honest_verdict": "complete_full_generation_run",
            "inference_substrate": "live_llm_inference",
            "duration_s": 5.0,
            "model_specs": {"gguf_path": "/some/path/model.gguf"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)
