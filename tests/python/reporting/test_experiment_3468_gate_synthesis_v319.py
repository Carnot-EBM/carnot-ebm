"""Tests for scripts/experiment_3468_g_gate_status_synthesis_v319.py.

REQ-GATE-001: G1–G4 gate status synthesis must read depth-block artifacts,
  skip flagged_adversarial ones for numeric aggregation, and emit a
  structured record with all required schema fields.
SCENARIO-GATE-001: exp3460 flagged_adversarial → p0_1_v5_verdict contains
  "flagged_adversarial" and trained_energy_vs_self_consistency_delta is None.
SCENARIO-GATE-002: clean exp3460 → delta is set from artifact.
SCENARIO-GATE-003: exp3461 (clean) → trained_energy_correctness_auroc populated.
SCENARIO-GATE-004: exp3463 (clean) → g2_ci_status from artifact.
SCENARIO-GATE-005: exp3462 flagged → fr11_collapse_consequence_deflagged notes
  flag but still records directional finding string.
SCENARIO-GATE-006: depth_forcing_function_can_relax only when BOTH P0.1
  clean AND external reproducer in motion.
SCENARIO-GATE-007: All required schema fields present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import experiment_3468_g_gate_status_synthesis_v319 as synth  # noqa: E402


# ─── fixture helpers ─────────────────────────────────────────────────────────


def _make_gate(*, g2: bool = False) -> dict:
    """Minimal publication_gate.evaluate() result."""
    return {
        "paper_ready": False,
        "gates": {
            "G1": {"pass": True, "detail": "FoVer 0.9131 present", "source": "exp2850.json"},
            "G2": {"pass": g2, "detail": "no independent reproducer" if not g2 else "confirmed"},
            "G3": {"pass": True, "detail": "no forbidden phrasings", "hits": []},
            "G4": {"pass": True, "detail": "seed+checksum present", "source": "exp2850.json"},
        },
        "unmet_gates": ["G2"] if not g2 else [],
        "note": "stable gate",
    }


def _exp3460_flagged() -> dict:
    """exp3460 with flagged_adversarial=True (TAUTOLOGY on tied metrics)."""
    return {
        "experiment": 3460,
        "honest_verdict": "complete: trained_energy_matches_but_does_not_beat_self_consistency_at_equal_compute",
        "self_consistency_non_degenerate": True,
        "delta_trained_energy_vs_self_consistency": 0.0,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "SC==trained_energy acc"}],
    }


def _exp3460_clean() -> dict:
    """Hypothetical clean exp3460 that beats SC."""
    return {
        "experiment": 3460,
        "honest_verdict": "complete: trained_energy_beats_self_consistency_delta_0.025",
        "self_consistency_non_degenerate": True,
        "delta_trained_energy_vs_self_consistency": 0.025,
        "flagged_adversarial": False,
    }


def _exp3461_clean() -> dict:
    """Clean exp3461 with AUROC above 0.55."""
    return {
        "experiment": 3461,
        "honest_verdict": "complete: trained_or_fover_energy_tracks_correctness_lift_over_untrained_reported",
        "trained_energy_correctness_auroc": 0.629401,
        "fover_energy_correctness_auroc": 0.605838,
        "flagged_adversarial": False,
    }


def _exp3462_flagged() -> dict:
    """exp3462 flagged (TAUTOLOGY on pass_rate fields)."""
    return {
        "experiment": 3462,
        "honest_verdict": "complete: residual_diversity_holds_no_collapse_in_fr11_loop_deflagged",
        "grounding_collapse_consequence": "ARM A did NOT collapse over 50 iterations: residual diversity sufficient.",
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "pass_rate fields tied"}],
    }


def _exp3462_clean() -> dict:
    """Hypothetical clean exp3462."""
    return {
        "experiment": 3462,
        "honest_verdict": "complete: residual_diversity_holds_no_collapse",
        "grounding_collapse_consequence": "ARM A did NOT collapse; ARM B stable too.",
        "flagged_adversarial": False,
    }


def _exp3463_clean() -> dict:
    """Clean exp3463: CI dry-run green, handoff ready."""
    return {
        "experiment": 3463,
        "honest_verdict": "complete: fover_g2_ci_dryrun_green_handoff_ready_external_run_pending",
        "g2_status": "ci_dryrun_green_handoff_ready_external_run_pending",
        "g2_ci_dryrun_green": True,
        "g2_handoff_package_ready": True,
        "g2_independent_reproducer": False,
        "flagged_adversarial": False,
    }


def _exp3464_clean() -> dict:
    """Clean exp3464: no lift on Kona hybrid."""
    return {
        "experiment": 3464,
        "honest_verdict": "complete: trained_energy_no_lift_over_untrained_kona_hybrid",
        "delta_trained_vs_untrained_hybrid": 0.0,
        "flagged_adversarial": False,
    }


def _run_synthesise(
    *,
    exp3460: dict | None = "default_flagged",
    exp3461: dict | None = "default_clean",
    exp3462: dict | None = "default_flagged",
    exp3463: dict | None = "default_clean",
    exp3464: dict | None = "default_clean",
) -> dict:
    """Run synth.synthesise() with controlled artifact loading and gate."""
    # Resolve sentinel defaults
    if exp3460 == "default_flagged":
        exp3460 = _exp3460_flagged()
    if exp3461 == "default_clean":
        exp3461 = _exp3461_clean()
    if exp3462 == "default_flagged":
        exp3462 = _exp3462_flagged()
    if exp3463 == "default_clean":
        exp3463 = _exp3463_clean()
    if exp3464 == "default_clean":
        exp3464 = _exp3464_clean()

    artifacts = {
        3460: exp3460,
        3461: exp3461,
        3462: exp3462,
        3463: exp3463,
        3464: exp3464,
    }

    def _fake_load(exp_id: int) -> dict | None:
        return artifacts.get(exp_id)

    class _FakeGateMod:
        @staticmethod
        def evaluate() -> dict:
            return _make_gate()

    with (
        patch.object(synth, "load_artifact", side_effect=_fake_load),
        patch.object(synth, "_gate_eval", return_value=_make_gate()),
    ):
        return synth.synthesise()


# ─── is_flagged ──────────────────────────────────────────────────────────────


def test_is_flagged_true():
    # REQ-GATE-001 SCENARIO-GATE-001
    assert synth.is_flagged({"flagged_adversarial": True}) is True


def test_is_flagged_false():
    assert synth.is_flagged({"flagged_adversarial": False}) is False


def test_is_flagged_missing_key():
    assert synth.is_flagged({}) is False


def test_is_flagged_none_artifact():
    assert synth.is_flagged(None) is False


# ─── load_artifact ────────────────────────────────────────────────────────────


def test_load_artifact_exists(tmp_path, monkeypatch):
    """load_artifact reads and parses a real artifact file."""
    payload = {"experiment": 3460, "x": 1}
    fname = synth.DEPTH_BLOCK[3460]
    (tmp_path / fname).write_text(json.dumps(payload))
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(3460) == payload


def test_load_artifact_missing(tmp_path, monkeypatch):
    """File absent → None (not a crash)."""
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(3460) is None


def test_load_artifact_corrupt_json(tmp_path, monkeypatch):
    """Corrupt JSON → None."""
    fname = synth.DEPTH_BLOCK[3460]
    (tmp_path / fname).write_text("{not valid json")
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(3460) is None


def test_load_artifact_unknown_id(tmp_path, monkeypatch):
    """Unknown exp_id → None (not in DEPTH_BLOCK)."""
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(9999) is None


# ─── synthesise: p0_1_v5_verdict ─────────────────────────────────────────────


def test_p0_1_verdict_flagged_contains_marker():
    # SCENARIO-GATE-001: flagged exp3460 → verdict string notes the flag
    result = _run_synthesise()
    assert "flagged_adversarial" in result["p0_1_v5_verdict"]


def test_p0_1_verdict_flagged_delta_null():
    # SCENARIO-GATE-001: flagged → numeric delta excluded
    result = _run_synthesise()
    assert result["trained_energy_vs_self_consistency_delta"] is None


def test_p0_1_verdict_clean_propagates_verdict():
    # SCENARIO-GATE-002: clean exp3460 → verdict and delta from artifact
    result = _run_synthesise(exp3460=_exp3460_clean())
    assert "flagged_adversarial" not in result["p0_1_v5_verdict"]
    assert result["trained_energy_vs_self_consistency_delta"] == pytest.approx(0.025)


def test_p0_1_verdict_missing_artifact():
    result = _run_synthesise(exp3460=None)
    assert "artifact_missing" in result["p0_1_v5_verdict"]
    assert result["trained_energy_vs_self_consistency_delta"] is None


# ─── synthesise: trained_energy_correctness_auroc ────────────────────────────


def test_auroc_from_clean_exp3461():
    # SCENARIO-GATE-003: clean exp3461 → AUROC populated
    result = _run_synthesise()
    assert result["trained_energy_correctness_auroc"] == pytest.approx(0.629401)


def test_auroc_crosses_055_threshold():
    result = _run_synthesise()
    assert result["trained_energy_crosses_055_threshold"] is True


def test_auroc_none_when_exp3461_missing():
    result = _run_synthesise(exp3461=None)
    assert result["trained_energy_correctness_auroc"] is None
    assert result["trained_energy_crosses_055_threshold"] is False


def test_auroc_none_when_exp3461_flagged():
    flagged = {**_exp3461_clean(), "flagged_adversarial": True}
    result = _run_synthesise(exp3461=flagged)
    assert result["trained_energy_correctness_auroc"] is None


# ─── synthesise: g2_ci_status ────────────────────────────────────────────────


def test_g2_ci_status_from_clean_exp3463():
    # SCENARIO-GATE-004: clean exp3463 → g2_status propagated
    result = _run_synthesise()
    assert result["g2_ci_status"] == "ci_dryrun_green_handoff_ready_external_run_pending"


def test_g2_ci_status_artifact_missing():
    result = _run_synthesise(exp3463=None)
    assert "artifact_missing" in result["g2_ci_status"]


def test_g2_ci_status_flagged():
    flagged = {**_exp3463_clean(), "flagged_adversarial": True}
    result = _run_synthesise(exp3463=flagged)
    assert "flagged_adversarial" in result["g2_ci_status"]


# ─── synthesise: fr11_collapse_consequence_deflagged ─────────────────────────


def test_fr11_collapse_notes_flagged_status():
    # SCENARIO-GATE-005: flagged exp3462 → field notes flag + directional finding
    result = _run_synthesise()
    cons = result["fr11_collapse_consequence_deflagged"]
    assert "flagged_adversarial" in cons
    assert "collapse" in cons.lower()


def test_fr11_collapse_directional_finding_in_flagged():
    # Even when flagged, the directional consequence text is preserved
    result = _run_synthesise()
    assert "NOT collapse" in result["fr11_collapse_consequence_deflagged"]


def test_fr11_collapse_clean():
    # Clean exp3462 → uses grounding_collapse_consequence field directly
    result = _run_synthesise(exp3462=_exp3462_clean())
    assert "ARM A did NOT collapse" in result["fr11_collapse_consequence_deflagged"]
    assert "flagged_adversarial" not in result["fr11_collapse_consequence_deflagged"]


def test_fr11_collapse_missing():
    result = _run_synthesise(exp3462=None)
    assert "artifact_missing" in result["fr11_collapse_consequence_deflagged"]


# ─── synthesise: kona_trained_hybrid_delta ───────────────────────────────────


def test_kona_delta_from_clean_exp3464():
    result = _run_synthesise()
    assert result["kona_trained_hybrid_delta"] == pytest.approx(0.0)


def test_kona_delta_none_when_missing():
    result = _run_synthesise(exp3464=None)
    assert result["kona_trained_hybrid_delta"] is None


def test_kona_delta_none_when_flagged():
    flagged = {**_exp3464_clean(), "flagged_adversarial": True}
    result = _run_synthesise(exp3464=flagged)
    assert result["kona_trained_hybrid_delta"] is None


# ─── synthesise: depth_forcing_function_can_relax ────────────────────────────


def test_depth_forcing_false_when_p01_flagged():
    # SCENARIO-GATE-006: exp3460 flagged → not clean → cannot relax
    result = _run_synthesise()
    assert result["depth_forcing_function_can_relax"] is False


def test_depth_forcing_false_when_p01_missing():
    result = _run_synthesise(exp3460=None)
    assert result["depth_forcing_function_can_relax"] is False


def test_depth_forcing_false_even_with_clean_p01_no_external_ask():
    # Even with a clean P0.1, if no external ask confirmed in motion → False.
    # exp3463 has g2_handoff_package_ready=True + g2_ci_dryrun_green=True
    # but external_ask_confirmed is still False (hardcoded in synthesise()).
    result = _run_synthesise(exp3460=_exp3460_clean())
    assert result["depth_forcing_function_can_relax"] is False


# ─── synthesise: required schema fields ──────────────────────────────────────


REQUIRED_FIELDS = {
    "honest_verdict",
    "g1",
    "g2",
    "g3",
    "g4",
    "unmet_gates",
    "p0_1_v5_verdict",
    "trained_energy_vs_self_consistency_delta",
    "trained_energy_correctness_auroc",
    "g2_ci_status",
    "fr11_collapse_consequence_deflagged",
    "kona_trained_hybrid_delta",
    "depth_forcing_function_can_relax",
    "gate_status_v319_ready",
}


def test_all_required_fields_present():
    # SCENARIO-GATE-007
    result = _run_synthesise()
    missing = REQUIRED_FIELDS - set(result)
    assert not missing, f"Missing schema fields: {missing}"


def test_honest_verdict_has_terminal_prefix():
    result = _run_synthesise()
    v = result["honest_verdict"]
    assert any(
        v.startswith(p)
        for p in (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    ), f"Non-terminal prefix in honest_verdict: {v!r}"


def test_gate_status_v319_ready_true():
    result = _run_synthesise()
    assert result["gate_status_v319_ready"] is True


def test_unmet_gates_is_list_contains_g2():
    result = _run_synthesise()
    assert isinstance(result["unmet_gates"], list)
    assert "G2" in result["unmet_gates"]


def test_g1_g3_g4_met():
    result = _run_synthesise()
    assert result["g1"] is True
    assert result["g3"] is True
    assert result["g4"] is True


def test_g2_not_met():
    result = _run_synthesise()
    assert result["g2"] is False


# ─── integration: output artifact valid ──────────────────────────────────────


def test_output_artifact_valid():
    """The actual artifact on disk has all required fields and correct values."""
    artifact_path = (
        PROJECT_ROOT / "results" / "experiment_3468_g_gate_status_synthesis_v319.json"
    )
    if not artifact_path.exists():
        pytest.skip("artifact not yet written — run the script first")
    data = json.loads(artifact_path.read_text())
    missing = REQUIRED_FIELDS - set(data)
    assert not missing, f"Artifact missing fields: {missing}"
    assert data["gate_status_v319_ready"] is True
    assert data["honest_verdict"].startswith("complete:")
    assert data["g2"] is False
    assert data["depth_forcing_function_can_relax"] is False
