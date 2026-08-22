"""Tests for adversarial_verify.py's web_and_bibliographic_search_only substrate.

Origin: 2026-07-20 (outer-loop, surveying overnight conductor activity). Two source-delta/SOTA-
ingestion artifacts (results/experiment_5718_v511_source_delta_ingestion.json,
results/experiment_5732_v512_source_delta_ingestion.json) declared
inference_substrate=web_and_bibliographic_search_only -- pure WebSearch/WebFetch + a
research-references.md read, no model load, no GPU -- and were flagged DURATION_TOO_SHORT under
the generic 60s live_llm_inference floor because no substrate category recognized the value. Two
sibling v510/v513 ingestion artifacts with the same substrate value but content that didn't trip
the compute-bound-marker detector passed clean, confirming the false positive was
substrate-recognition-gap-driven (the value simply wasn't wired in), not artifact-specific.
Mirrors the deterministic_smt_hint_validation_no_llm / artifact_qa_lint_tests precedent: a genuine,
LLM-free substrate whose own name discloses the gap, needing its own near-zero floor rather than
the compute-bound fallback.

Spec refs: REQ-ARC-FCP-5732 (operational lint fix found while surveying overnight conductor
artifacts adjacent to that experiment's lineage; no dedicated OpenSpec capability of its own).
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


class TestIsWebBibliographicSearchOnly:
    def test_recognizes_canonical_substrate_value(self) -> None:
        d = {"inference_substrate": "web_and_bibliographic_search_only"}
        assert av._is_web_bibliographic_search_only(d) is True

    def test_recognizes_source_receipt_method_preregistration_alias(self) -> None:
        d = {"inference_substrate": "source_receipts_and_local_method_preregistration_no_llm"}
        assert av._is_web_bibliographic_search_only(d) is True

    def test_does_not_match_generic_live_llm_inference(self) -> None:
        d = {"inference_substrate": "live_llm_inference"}
        assert av._is_web_bibliographic_search_only(d) is False

    def test_does_not_match_missing_field(self) -> None:
        assert av._is_web_bibliographic_search_only({}) is False


class TestDurationFloorForArtifact:
    def test_returns_the_near_zero_floor(self) -> None:
        d = {"inference_substrate": "web_and_bibliographic_search_only"}
        floor = av.duration_floor_for_artifact(d)
        assert floor == {
            "substrate": "web_and_bibliographic_search_only",
            "min_duration_s": 0.0001,
            "reason": "web_bibliographic_search_only",
        }

    def test_returns_the_near_zero_floor_for_source_receipt_method_contract(self) -> None:
        d = {"inference_substrate": "source_receipts_and_local_method_preregistration_no_llm"}
        floor = av.duration_floor_for_artifact(d)
        assert floor == {
            "substrate": "source_receipts_and_local_method_preregistration_no_llm",
            "min_duration_s": 0.0001,
            "reason": "web_bibliographic_search_only",
        }


class TestExp5718Exp5732IncidentReproduction:
    def test_exp5732_duration_no_longer_flags_duration_too_short(self, tmp_path: Path) -> None:
        payload = {
            "experiment": "exp5732_v512_source_delta_ingestion",
            "honest_verdict": (
                "complete: accepted 3 non-duplicate actionable V512 source deltas; "
                "no roadmap ID, gate, benchmark, or hardware claim changed"
            ),
            "inference_substrate": "web_and_bibliographic_search_only",
            "duration_s": 0.049658,
            # Realistic trigger: the ingested-literature prose mentions GGUF (about a proposal
            # signal, not a claim this task itself ran a model) -- this is what makes
            # _has_compute_bound_marker fire on the real artifact and route it into the
            # duration-vs-claim check at all. A payload with no such text never reaches the
            # check in the first place, so this must be present for a realistic reproduction.
            "accepted_deltas": [
                {"note": "style feasibility validators admit rows; GGUF label scores remain..."}
            ],
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" not in _flag_kinds(report)

    def test_model_specs_not_required_for_this_substrate(self, tmp_path: Path) -> None:
        """This substrate genuinely has no model to name -- matches the SMT-hint-validation and
        log-analysis exemption pattern."""
        payload = {
            "experiment": "exp5732_repro",
            "honest_verdict": "complete: accepted non-duplicate actionable source deltas",
            "inference_substrate": "web_and_bibliographic_search_only",
            "duration_s": 0.05,
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

    def test_implausibly_low_duration_still_flags_even_under_the_new_floor(
        self, tmp_path: Path
    ) -> None:
        """exp5718's actual duration_s (2.7e-07s, 270 nanoseconds) is implausible for ANY task
        involving a WebSearch network round-trip -- a genuine duration-measurement anomaly in
        that artifact, not the substrate-recognition gap this fix closes. The new 0.0001s floor
        must not be loosened further to paper over it; it should stay flagged. Reproduces the
        exact trigger from the real artifact: ingested-literature prose mentioning GGUF (about
        FR-11's architecture, not a claim this task itself ran a model) is what makes
        _has_compute_bound_marker fire at all -- a payload with no such text never reaches the
        duration check in the first place, so the marker text must be present for this to be a
        realistic regression guard."""
        payload = {
            "experiment": "exp5718_v511_source_delta_ingestion",
            "honest_verdict": (
                "complete: accepted 1 non-duplicate actionable V511 source delta; "
                "no roadmap ID or gate change"
            ),
            "inference_substrate": "web_and_bibliographic_search_only",
            "duration_s": 2.6996713131666183e-07,
            "accepted_deltas": [
                {
                    "source_id": "gate_zero_growth",
                    "note": "does not fit immutable GGUF FR-11 sidecar.",
                }
            ],
        }
        report = _report_for_payload(tmp_path, payload)
        assert "DURATION_TOO_SHORT" in _flag_kinds(report)
