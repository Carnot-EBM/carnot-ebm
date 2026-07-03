"""Tests for adversarial_verify.py's _flips_gate word-boundary fix.

Origin: 2026-07-02. exp5173's honest_verdict `blocked_diffusiongemma_meta_tensor_bug_
unresolved` falsely triggered CIRCULAR_MOAT_OVERCLAIM. `_flips_gate` did a plain
`"diffusiongemma_met" in hv.lower()` substring check, which matched inside
"diffusiongemma_META_tensor" (a PyTorch technical term for placeholder tensors during
device-mapped model loading -- unrelated to any gate being MET). The artifact was
honestly BLOCKED (empty arm_rows, all-zero pass@1s, verdict correctly prefixed
`blocked_`) yet got CRITICAL-flagged as if it had flipped a circular
(verifier_is_oracle=True) moat gate.

Fixed with negative-lookahead regexes (`_GATE_MET_RE`, `_DIFFUSIONGEMMA_MET_RE`)
requiring "met" not be immediately followed by another lowercase letter/digit --
excludes "meta"/"method"/"metric"/"metadata" while still matching the genuine word
"met" at a token boundary (followed by "_" or end-of-string).

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


class TestFlipsGateWordBoundary:
    """Unit tests for the pure detection function."""

    def test_meta_tensor_does_not_false_positive(self) -> None:
        d = {"honest_verdict": "blocked_diffusiongemma_meta_tensor_bug_unresolved"}
        assert av._flips_gate(d) is False

    def test_method_does_not_false_positive(self) -> None:
        d = {"honest_verdict": "blocked_diffusiongemma_method_lookup_failed"}
        assert av._flips_gate(d) is False

    def test_metric_does_not_false_positive_on_gate_met(self) -> None:
        d = {"honest_verdict": "complete_gate_metric_recorded_no_decision"}
        assert av._flips_gate(d) is False

    def test_genuine_diffusiongemma_met_at_underscore_boundary_still_detected(self) -> None:
        d = {"honest_verdict": "success_diffusiongemma_met_gate_condition"}
        assert av._flips_gate(d) is True

    def test_genuine_diffusiongemma_met_at_string_end_still_detected(self) -> None:
        d = {"honest_verdict": "success_the_diffusiongemma_met"}
        assert av._flips_gate(d) is True

    def test_genuine_gate_met_still_detected(self) -> None:
        d = {
            "honest_verdict": "success_archived_v401_v402_active_moat_replicated_leak_robust_gate_MET_arc21"
        }
        assert av._flips_gate(d) is True

    def test_dict_diffusiongemma_gate_status_still_detected(self) -> None:
        d = {"diffusiongemma_gate_status": "MET"}
        assert av._flips_gate(d) is True

    def test_dict_diffusiongemma_gate_object_still_detected(self) -> None:
        d = {"diffusiongemma_gate": {"met": True}}
        assert av._flips_gate(d) is True

    def test_no_verdict_field_returns_false(self) -> None:
        assert av._flips_gate({}) is False


class TestExp5173IncidentReproduction:
    """End-to-end reproduction of the exact exp5173 false-positive, verified fixed."""

    def test_blocked_diffusiongemma_pilot_no_longer_flags_circular_moat_overclaim(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "experiment": "exp5173_repro",
            "honest_verdict": "blocked_diffusiongemma_meta_tensor_bug_unresolved",
            "verifier_is_oracle": True,
            "arm_rows": [],
            "pass_at_1_ar_baseline": 0.0,
            "pass_at_1_guided": 0.0,
            "pass_at_1_unguided": 0.0,
            "inference_substrate": "blocked_preflight",
        }
        report = _report_for_payload(tmp_path, payload)
        assert "CIRCULAR_MOAT_OVERCLAIM" not in _flag_kinds(report)

    def test_genuinely_circular_gate_flip_is_still_caught(self, tmp_path: Path) -> None:
        """Regression guard: the fix must not weaken the check -- a real circular
        gate-flip claim (verifier_is_oracle=True, honest_verdict genuinely says the
        diffusiongemma gate is MET) must still be caught."""
        payload = {
            "experiment": "exp_genuinely_circular",
            "honest_verdict": "success_diffusiongemma_met_gate_condition_moat_proven",
            "verifier_is_oracle": True,
        }
        report = _report_for_payload(tmp_path, payload)
        assert "CIRCULAR_MOAT_OVERCLAIM" in _flag_kinds(report)
