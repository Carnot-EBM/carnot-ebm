"""Tests for adversarial_verify.py's broadened _is_precondition_check_only_blocked exemption.

Origin: 2026-07-05. exp5274 (results/experiment_5274_solver_constraint_extraction_retry_gated_
v482.json) checked its own preconditions, found llama_cpp_gpu_offload_unavailable, and correctly
wrote an honest blocked_preconditions verdict (extraction_results=[], rows_total=0) without
attempting any inference -- exactly the Pre-Launch Preconditions Discipline's intended behavior.
It still got flagged DURATION_TOO_SHORT (duration_s=0.14) because the pre-existing
_is_precondition_check_only_blocked exemption required the artifact to ALSO declare
inference_substrate="precondition_check_only" specifically -- but this script hardcodes the same
INFERENCE_SUBSTRATE constant (live_llm_inference_local_gguf_sota) on both its live and blocked
branches, a design choice several conductor-authored scripts share.

The fix trusts the honest_verdict's blocked_ prefix on its own: per this project's own Verdict
Terminal-Prefix Discipline + Pre-Launch Preconditions Discipline, a blocked_* verdict is already
a mandated, structured admission that the compute-bound work did not happen -- requiring one
specific substrate string in addition to that was over-narrow. Both existing call sites
(duration_floor_for_artifact, check_duration_vs_claim) are duration-related only, so this does not
weaken the tautology, methodology, or gate checks.

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


class TestIsPreconditionCheckOnlyBlocked:
    def test_bare_blocked_verdict_is_sufficient_regardless_of_substrate(self) -> None:
        d = {
            "honest_verdict": "blocked_preconditions: llama_cpp_gpu_offload_unavailable",
            "inference_substrate": "live_llm_inference_local_gguf_sota",
        }
        assert av._is_precondition_check_only_blocked(d) is True

    def test_explicit_precondition_check_only_substrate_still_recognized(self) -> None:
        d = {
            "honest_verdict": "blocked_model_not_cached",
            "inference_substrate": "precondition_check_only",
        }
        assert av._is_precondition_check_only_blocked(d) is True

    def test_non_blocked_verdict_is_not_exempted(self) -> None:
        d = {
            "honest_verdict": "complete: real result",
            "inference_substrate": "live_llm_inference_local_gguf_sota",
        }
        assert av._is_precondition_check_only_blocked(d) is False

    def test_missing_verdict_is_not_exempted(self) -> None:
        assert av._is_precondition_check_only_blocked({}) is False


class TestExp5274IncidentReproduction:
    def test_014s_blocked_precondition_no_longer_flags_duration_too_short(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "experiment": "exp5274_repro",
            "honest_verdict": (
                "blocked_preconditions: llama_cpp_gpu_offload_unavailable; retry was unmeasured"
            ),
            "inference_substrate": "live_llm_inference_local_gguf_sota",
            "duration_s": 0.143891,
            "extraction_results": [],
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_duration_floor_is_none_for_a_blocked_artifact(self) -> None:
        d = {
            "honest_verdict": "blocked_preconditions: llama_cpp_gpu_offload_unavailable",
            "inference_substrate": "live_llm_inference_local_gguf_sota",
        }
        assert av.duration_floor_for_artifact(d) is None

    def test_non_blocked_artifact_with_same_substrate_still_needs_the_10s_floor(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: a non-blocked artifact declaring the same substrate must
        still be held to its floor -- the exemption only applies to honest blocked_*
        verdicts, not to every artifact that shares this substrate string."""
        payload = {
            "experiment": "exp_not_blocked",
            "honest_verdict": "complete: fabricated fast claim",
            "inference_substrate": "live_llm_inference_local_gguf_sota",
            "duration_s": 0.1,
            "timing_evidence": {"backend": "torch_cuda:0"},
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)
