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

2026-07-14 follow-up (TestExp5713TerminalPrefixedBlockedVerdict below): the original fix only
recognized a BARE `blocked_` prefix, missing the `complete: blocked_<resource>` form CLAUDE.md's
own Verdict Terminal-Prefix Discipline actually mandates every terminal verdict use. Surfaced by
exp5713 (Qwen3.6-27B Q4_K_M + Q8 KV-cache with MTP enabled -- a real, fast precondition block on
a hard GPU-memory OOM, written with the mandated terminal prefix and false-flagged before this
follow-up fix). See REQ-ARC-WMTE-5599-4 in
openspec/capabilities/arc-human-replay-frame-change/spec.md for the full incident this operational
lint fix traces to (this file itself has no OpenSpec capability of its own -- it tests a script,
not a product requirement).

Spec refs: REQ-ARC-WMTE-5599-4 (operational lint fix motivated by that experiment's incident),
REQ-INFER-SOTA-6102 (blocked colon verdict required by the sequential VRAM recovery artifact).
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


class TestExp5713TerminalPrefixedBlockedVerdict:
    """2026-07-14: the ORIGINAL fix above only recognized a BARE `blocked_` prefix, missing
    the `complete: blocked_<resource>` form CLAUDE.md's Verdict Terminal-Prefix Discipline
    actually mandates every terminal verdict use. exp5713 (Qwen3.6-27B Q4_K_M + Q8 KV-cache
    with MTP enabled) hit exactly this: its precondition check found the self-speculative
    MTP dual-load (target + draft, same GGUF loaded twice) needs ~32.6GB, exceeding the
    single RTX 3090's 24GB -- a real, fast (duration_s=0.0), honest precondition block,
    written as `complete: blocked_gpu1_free_vram_sufficient_for_mtp_dual_load` per the
    terminal-prefix discipline, and false-flagged DURATION_TOO_SHORT before this fix."""

    def test_complete_colon_prefixed_blocked_verdict_is_recognized(self) -> None:
        d = {
            "honest_verdict": "complete: blocked_gpu1_free_vram_sufficient_for_mtp_dual_load",
            "inference_substrate": "live_llm_inference",
        }
        assert av._is_precondition_check_only_blocked(d) is True

    def test_complete_underscore_prefixed_blocked_verdict_is_recognized(self) -> None:
        d = {"honest_verdict": "complete_blocked_model_not_cached"}
        assert av._is_precondition_check_only_blocked(d) is True

    def test_success_and_passed_and_shipped_prefixes_also_recognized(self) -> None:
        for prefix in ("success:", "passed:", "shipped:"):
            d = {"honest_verdict": f"{prefix} blocked_resource_x"}
            assert av._is_precondition_check_only_blocked(d) is True, prefix

    def test_terminal_prefixed_non_blocked_verdict_still_not_exempted(self) -> None:
        d = {"honest_verdict": "complete: qwen27b_mtp_plans_more_reliably_than_current_9b"}
        assert av._is_precondition_check_only_blocked(d) is False

    def test_blocked_colon_verdict_is_recognized_for_exp6102(self, tmp_path: Path) -> None:
        """REQ-INFER-SOTA-6102: blocked: preflight verdicts do not claim model execution."""

        payload = {
            "experiment": "exp6102_repro",
            "honest_verdict": "blocked: insufficient_free_vram",
            "inference_substrate": "live_local_sota_gguf_cuda_representation_extraction",
            "duration_s": 0.5,
            "runtime_cuda_vram_thermal_and_pid_lease_receipts": {
                "capacity_verdicts": {
                    "unsloth/Qwen3.6-35B-A3B-GGUF": {
                        "fits": False,
                        "reason": "insufficient_free_vram",
                    }
                }
            },
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
        assert "METHODOLOGY_MISSING" not in _flag_kinds(report)

    def test_exp5713_style_artifact_no_longer_flags_duration_too_short(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "experiment": "exp5713_repro",
            "honest_verdict": "complete: blocked_gpu1_free_vram_sufficient_for_mtp_dual_load",
            "inference_substrate": "live_llm_inference",
            "duration_s": 0.0,
            "per_draw_results": [],
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)


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
