"""Tests for exp3435 Capstone v316 (Depth-Over-Breadth II).

Covers REQ-REPORT-3435 and SCENARIO-REPORT-3435.

These tests drive the capstone module in isolation by injecting a fake
results directory so no real artifact files are required. They lock down:

  * the schema / required-field contract,
  * the honest_verdict terminal-prefix requirement,
  * G1-G4 gate passthrough from the gate-synthesis artifact (exp3434),
  * Paper-v6 Narrowing Discipline: safe_claims present, forbidden_claims
    present, the 4-verifier/k=15 conflation guard, and the P0.1-v2
    null-candidate framing guard,
  * the fabrication-gate skip (flagged_adversarial artifacts are omitted),
  * depth_forcing_function_can_relax behaviour,
  * deterministic reproducibility_checksum,
  * missing gate-synthesis artifact fallback.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from carnot.reporting.capstone_v316_3435 import run_capstone, _FLAGGED_ADVERSARIAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(directory: Path, filename: str, data: dict) -> None:
    """Write a JSON file into *directory* for use as a fake upstream artifact."""
    p = directory / filename
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _gate_artifact(
    *,
    g1: bool = True,
    g2: bool = False,
    g3: bool = True,
    g4: bool = True,
    depth_relax: bool = False,
    p0_1_v2_is_clean: bool = True,
) -> dict:
    """Return a minimal gate-synthesis artifact (exp3434 shape)."""
    return {
        "experiment": 3434,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "unmet_gates": [k for k, v in {"G1": g1, "G2": g2, "G3": g3, "G4": g4}.items() if not v],
        "p0_1_v2_verdict": "complete: energy_matches_but_does_not_beat_self_consistency_at_equal_compute",
        "p0_1_v2_is_clean": p0_1_v2_is_clean,
        "depth_forcing_function_can_relax": depth_relax,
        "depth_forcing_function_relax_rationale": "P0.1 v2 clean but null candidates; G2 CI failed.",
        "gate_status_v316_ready": True,
        "honest_verdict": "complete: gate_status_v316_synthesized",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 0.07,
    }


# ---------------------------------------------------------------------------
# Tests: schema / required fields
# ---------------------------------------------------------------------------

class TestSchema:
    """REQ-REPORT-3435: capstone emits a complete, schema-valid artifact."""

    REQUIRED_FIELDS = [
        "schema",
        "experiment",
        "experiment_id",
        "task_id",
        "milestone",
        "inference_substrate",
        "duration_s",
        "random_seed",
        "reproducibility_checksum",
        "g1",
        "g2",
        "g3",
        "g4",
        "unmet_gates",
        "paper_ready",
        "p0_1_v2_verdict",
        "p0_1_v2_is_clean",
        "p0_1_v2_summary",
        "depth_forcing_function_can_relax",
        "next_depth_focus",
        "paper_v6_safe_claims",
        "paper_v6_forbidden_claims",
        "upstreams",
        "capstone_v316_ready",
        "honest_verdict",
        "cited_upstream_artifacts",
        "field_provenance",
    ]

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name)
        _write_json(self.results, "experiment_3434_gate_status.json", _gate_artifact())

    def teardown_method(self):
        self.tmp.cleanup()

    def test_all_required_fields_present(self):
        result = run_capstone(results_dir=self.results)
        for field in self.REQUIRED_FIELDS:
            assert field in result, f"Missing required field: {field}"

    def test_schema_string(self):
        result = run_capstone(results_dir=self.results)
        assert result["schema"] == "carnot.milestone_capstone.v316.v1"

    def test_inference_substrate_is_aggregation(self):
        result = run_capstone(results_dir=self.results)
        assert result["inference_substrate"] == "aggregation_from_upstream_artifacts"

    def test_capstone_v316_ready_true(self):
        result = run_capstone(results_dir=self.results)
        assert result["capstone_v316_ready"] is True

    def test_milestone_string(self):
        result = run_capstone(results_dir=self.results)
        assert result["milestone"] == "2026.05.316"

    def test_experiment_id(self):
        result = run_capstone(results_dir=self.results)
        assert result["experiment"] == 3435
        assert result["experiment_id"] == "exp3435"


# ---------------------------------------------------------------------------
# Tests: honest_verdict terminal-prefix
# ---------------------------------------------------------------------------

class TestHonestVerdict:
    """REQ-REPORT-3435 / SCENARIO-REPORT-3435: terminal-prefix discipline."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name)
        _write_json(self.results, "experiment_3434_gate_status.json", _gate_artifact())

    def teardown_method(self):
        self.tmp.cleanup()

    def test_verdict_starts_with_complete(self):
        result = run_capstone(results_dir=self.results)
        assert result["honest_verdict"].startswith("complete:")

    def test_verdict_contains_capstone_ready(self):
        result = run_capstone(results_dir=self.results)
        assert "capstone_v316_ready=true" in result["honest_verdict"]


# ---------------------------------------------------------------------------
# Tests: G1-G4 gate passthrough
# ---------------------------------------------------------------------------

class TestGatePassthrough:
    """REQ-REPORT-3435: G-gate values are read from exp3434."""

    def _run(self, **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_json(p, "experiment_3434_gate_status.json", _gate_artifact(**kwargs))
            return run_capstone(results_dir=p)

    def test_g1_true_propagates(self):
        result = self._run(g1=True)
        assert result["g1"] is True

    def test_g2_false_propagates(self):
        result = self._run(g2=False)
        assert result["g2"] is False
        assert "G2" in result["unmet_gates"]

    def test_paper_ready_false_when_g2_unmet(self):
        result = self._run(g1=True, g2=False, g3=True, g4=True)
        assert result["paper_ready"] is False

    def test_paper_ready_true_when_all_met(self):
        result = self._run(g1=True, g2=True, g3=True, g4=True)
        assert result["paper_ready"] is True
        assert result["unmet_gates"] == []

    def test_unmet_gates_list(self):
        result = self._run(g1=True, g2=False, g3=True, g4=True)
        assert result["unmet_gates"] == ["G2"]


# ---------------------------------------------------------------------------
# Tests: Paper-v6 Narrowing Discipline
# ---------------------------------------------------------------------------

class TestPaperV6NarrowingDiscipline:
    """SCENARIO-REPORT-3435: safe_claims and forbidden_claims content."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name)
        _write_json(self.results, "experiment_3434_gate_status.json", _gate_artifact())

    def teardown_method(self):
        self.tmp.cleanup()

    def test_safe_claims_nonempty(self):
        result = run_capstone(results_dir=self.results)
        assert len(result["paper_v6_safe_claims"]) > 0

    def test_forbidden_claims_nonempty(self):
        result = run_capstone(results_dir=self.results)
        assert len(result["paper_v6_forbidden_claims"]) > 0

    def test_safe_claims_include_fover_headline(self):
        result = run_capstone(results_dir=self.results)
        combined = " ".join(result["paper_v6_safe_claims"])
        assert "0.9131" in combined

    def test_safe_claims_cite_4verifier_not_k15(self):
        """FRAMING GUARD: safe claims must name 4-verifier, not k=15."""
        result = run_capstone(results_dir=self.results)
        combined = " ".join(result["paper_v6_safe_claims"])
        assert "4-verifier" in combined
        # Must NOT claim k=15 for the FoVer headline score
        assert "k=15" not in combined

    def test_forbidden_claims_include_k15_conflation_guard(self):
        result = run_capstone(results_dir=self.results)
        combined = " ".join(result["paper_v6_forbidden_claims"])
        assert "k=15" in combined or "4-verifier" in combined

    def test_p0_1_v2_null_candidate_caveat_in_summary(self):
        """P0.1 v2 summary must disclose the null-candidate limitation."""
        result = run_capstone(results_dir=self.results)
        summary = result["p0_1_v2_summary"].lower()
        assert "null" in summary or "uninterpretable" in summary

    def test_p0_1_v2_summary_does_not_claim_energy_beats_sc(self):
        """Must not assert energy beats/matches SC without the null-candidate caveat."""
        result = run_capstone(results_dir=self.results)
        summary = result["p0_1_v2_summary"].lower()
        # If "beats" or "validates" appear they must be qualified
        if "beats" in summary and "self-consistency" in summary:
            assert "null" in summary or "uninterpretable" in summary or "caveat" in summary

    def test_forbidden_claims_include_thermalization(self):
        result = run_capstone(results_dir=self.results)
        combined = " ".join(result["paper_v6_forbidden_claims"])
        assert "thermalization" in combined.lower() or "#2" in combined

    def test_forbidden_claims_include_hardware_speedup(self):
        result = run_capstone(results_dir=self.results)
        combined = " ".join(result["paper_v6_forbidden_claims"])
        assert "speedup" in combined.lower() or "hardware" in combined.lower()

    def test_forbidden_claims_include_p0_1_null_candidate_guard(self):
        """Forbidden claims must block misinterpretation of exp3426 delta=0.0."""
        result = run_capstone(results_dir=self.results)
        combined = " ".join(result["paper_v6_forbidden_claims"])
        assert "null" in combined.lower() or "exp3426" in combined or "delta=0.0" in combined


# ---------------------------------------------------------------------------
# Tests: P0.1 v2 passthrough
# ---------------------------------------------------------------------------

class TestP01V2:
    """SCENARIO-REPORT-3435: P0.1 v2 fields are passed through correctly."""

    def _run(self, **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_json(p, "experiment_3434_gate_status.json", _gate_artifact(**kwargs))
            return run_capstone(results_dir=p)

    def test_p0_1_v2_is_clean_true(self):
        result = self._run(p0_1_v2_is_clean=True)
        assert result["p0_1_v2_is_clean"] is True

    def test_p0_1_v2_is_clean_false(self):
        result = self._run(p0_1_v2_is_clean=False)
        assert result["p0_1_v2_is_clean"] is False

    def test_p0_1_v2_verdict_propagates(self):
        result = self._run()
        assert "energy_matches" in result["p0_1_v2_verdict"] or "complete:" in result["p0_1_v2_verdict"]


# ---------------------------------------------------------------------------
# Tests: fabrication-gate skip
# ---------------------------------------------------------------------------

class TestFabricationGate:
    """SCENARIO-REPORT-3435: flagged_adversarial artifacts are skipped."""

    def test_flagged_adversarial_set_contains_exp3397_and_exp3405(self):
        assert "exp3397" in _FLAGGED_ADVERSARIAL
        assert "exp3405" in _FLAGGED_ADVERSARIAL

    def test_flagged_artifact_skipped_in_upstreams(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_json(p, "experiment_3434_gate_status.json", _gate_artifact())
            # Write a fake artifact with flagged_adversarial=True
            _write_json(p, "experiment_3397_ebm_cot_benchmark.json", {
                "experiment": 3397,
                "honest_verdict": "complete: auroc=1.0",
                "flagged_adversarial": True,
            })
            result = run_capstone(results_dir=p)
        # exp3397 is in _FLAGGED_ADVERSARIAL and not in _UPSTREAM_IDS,
        # so the constant guard is validated
        assert "exp3397" not in result["upstreams"] or (
            "SKIPPED" in result["upstreams"].get("exp3397", "")
        )


# ---------------------------------------------------------------------------
# Tests: depth forcing function
# ---------------------------------------------------------------------------

class TestDepthForcingFunction:
    """SCENARIO-REPORT-3435: depth_can_relax controls next_depth_focus."""

    def _run(self, depth_relax: bool) -> dict:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            _write_json(p, "experiment_3434_gate_status.json", _gate_artifact(depth_relax=depth_relax))
            return run_capstone(results_dir=p)

    def test_relax_true_recommends_g2_closure(self):
        result = self._run(depth_relax=True)
        assert result["depth_forcing_function_can_relax"] is True
        assert "G2" in result["next_depth_focus"] or "g2" in result["next_depth_focus"].lower()

    def test_relax_false_recommends_p01_rerun(self):
        result = self._run(depth_relax=False)
        assert result["depth_forcing_function_can_relax"] is False
        assert "P0.1" in result["next_depth_focus"] or "p0_1" in result["next_depth_focus"].lower()

    def test_relax_false_mentions_answer_extraction_fix(self):
        """When depth is still active, next focus must name the root cause."""
        result = self._run(depth_relax=False)
        focus = result["next_depth_focus"].lower()
        assert "extraction" in focus or "parser" in focus or "regex" in focus or "answer" in focus


# ---------------------------------------------------------------------------
# Tests: reproducibility_checksum
# ---------------------------------------------------------------------------

class TestReproducibilityChecksum:
    """SCENARIO-REPORT-3435: checksum is deterministic and non-empty."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name)
        _write_json(self.results, "experiment_3434_gate_status.json", _gate_artifact())

    def teardown_method(self):
        self.tmp.cleanup()

    def test_checksum_nonempty(self):
        result = run_capstone(results_dir=self.results)
        assert result["reproducibility_checksum"]
        assert len(result["reproducibility_checksum"]) == 64  # SHA-256 hex

    def test_checksum_is_deterministic(self):
        r1 = run_capstone(results_dir=self.results)
        r2 = run_capstone(results_dir=self.results)
        assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]

    def test_checksum_excludes_duration_s(self):
        """Checksum must not depend on the mutable duration_s field."""
        result = run_capstone(results_dir=self.results)
        stable = {k: v for k, v in result.items() if k not in ("reproducibility_checksum", "duration_s")}
        expected = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
        assert result["reproducibility_checksum"] == expected


# ---------------------------------------------------------------------------
# Tests: missing gate-synthesis artifact fallback
# ---------------------------------------------------------------------------

class TestMissingGateArtifact:
    """SCENARIO-REPORT-3435: capstone must not crash if exp3434 is absent."""

    def test_runs_without_crash(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_capstone(results_dir=Path(d))
        assert result["capstone_v316_ready"] is True
        assert result["honest_verdict"].startswith("complete:")

    def test_fallback_g2_false(self):
        """Without gate artifact, G2 defaults to unmet (conservative)."""
        with tempfile.TemporaryDirectory() as d:
            result = run_capstone(results_dir=Path(d))
        assert result["g2"] is False
        assert "G2" in result["unmet_gates"]
