"""Tests for scripts/experiment_3480_g_gate_status_synthesis_v320.py.

REQ-GATE-001: G1–G4 gate status synthesis must read depth-block artifacts,
  skip flagged_adversarial ones for numeric aggregation, and emit a
  structured record with all required schema fields.
SCENARIO-GATE-001: exp3472 blocked (corpus too small) → p0_1_v6_verdict
  contains "blocked" and process_energy_vs_self_consistency_delta is None.
SCENARIO-GATE-002: exp3472 hypothetically clean → delta is set from artifact.
SCENARIO-GATE-003: exp3473 flagged → minority_correct_recovery_rate is None.
SCENARIO-GATE-004: exp3473 clean → minority_correct_recovery_rate populated.
SCENARIO-GATE-005: exp3474 clean → fr11_depth_collapse_consequence from artifact.
SCENARIO-GATE-006: exp3476 clean → g2_package_status from artifact g2_status.
SCENARIO-GATE-007: depth_forcing_function_can_relax is False while P0.1 blocked.
SCENARIO-GATE-008: All required schema fields present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import experiment_3480_g_gate_status_synthesis_v320 as synth  # noqa: E402


# ─── fixture helpers ─────────────────────────────────────────────────────────


def _make_gate(*, g2: bool = False) -> dict:
    """Minimal publication_gate._gate_eval() result."""
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


def _exp3472_blocked() -> dict:
    """exp3472 as it actually ran: blocked due to corpus too small (n=21)."""
    return {
        "experiment": 3472,
        "honest_verdict": "complete: blocked_p01_corpus_too_small_n=21",
        "n_problems_heldout": 21,
        "flip_count_optimal_vs_sc": None,
        "delta_optimal_vs_self_consistency": None,
        "flagged_adversarial": False,
    }


def _exp3472_clean() -> dict:
    """Hypothetical clean exp3472 that beats SC."""
    return {
        "experiment": 3472,
        "honest_verdict": "complete: process_energy_beats_sc_delta_0.032",
        "n_problems_heldout": 50,
        "flip_count_optimal_vs_sc": 8,
        "delta_optimal_vs_self_consistency": 0.032,
        "flagged_adversarial": False,
    }


def _exp3473_flagged() -> dict:
    """exp3473 with flagged_adversarial=True (TAUTOLOGY on recovery metrics)."""
    return {
        "experiment": 3473,
        "honest_verdict": "complete: energy_fails_to_recover_minority_correct",
        "minority_correct_recovery_rate_process": 0.041667,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "minority_correct_recovery_rate_process and _trained match to >5 sig figs.",
            }
        ],
    }


def _exp3473_clean() -> dict:
    """Hypothetical clean exp3473."""
    return {
        "experiment": 3473,
        "honest_verdict": "complete: energy_recovers_minority_correct_rate_0.60",
        "minority_correct_recovery_rate_process": 0.60,
        "flagged_adversarial": False,
    }


def _exp3474_clean() -> dict:
    """exp3474 as actually run: ARM A collapsed at N=200, ARM B stable."""
    return {
        "experiment": 3474,
        "honest_verdict": "complete: at_risk_grounding_causes_collapse_at_depth_entropy_reg_prevents_it_deflagged",
        "arm_a_mode_collapse_detected": True,
        "arm_b_mode_collapse_detected": False,
        "grounding_collapse_consequence": (
            "ARM A COLLAPSED at depth N=200 (onset iteration 138): "
            "entropy→0.9901, mode_mass→0.6056, pass_rate=1.0000 while "
            "true_accuracy=0.000000 (gap=1.0000 — null-space gaming). "
            "ARM B (entropy_beta=0.50) PREVENTED collapse: entropy=4.9067."
        ),
        "flagged_adversarial": False,
    }


def _exp3475_blocked() -> dict:
    """exp3475 as actually run: blocked — Kona instances saturated."""
    return {
        "experiment": 3475,
        "honest_verdict": "complete: blocked_kona_instances_saturated_no_headroom",
        "n_instances": 0,
        "delta_process_vs_untrained_hybrid": None,
        "flagged_adversarial": False,
    }


def _exp3476_clean() -> dict:
    """exp3476 as actually run: self-contained package verified, external pending."""
    return {
        "experiment": 3476,
        "honest_verdict": "complete: fover_g2_self_contained_package_verified_external_run_pending",
        "g2_status": "self_contained_package_verified_external_run_pending",
        "package_verified_reproduces": True,
        "g2_independent_reproducer": False,
        "flagged_adversarial": False,
    }


def _run_synthesise(
    *,
    exp3472: dict | None = "default_blocked",
    exp3473: dict | None = "default_flagged",
    exp3474: dict | None = "default_clean",
    exp3475: dict | None = "default_blocked",
    exp3476: dict | None = "default_clean",
) -> dict:
    """Run synth.synthesise() with controlled artifact loading and gate."""
    if exp3472 == "default_blocked":
        exp3472 = _exp3472_blocked()
    if exp3473 == "default_flagged":
        exp3473 = _exp3473_flagged()
    if exp3474 == "default_clean":
        exp3474 = _exp3474_clean()
    if exp3475 == "default_blocked":
        exp3475 = _exp3475_blocked()
    if exp3476 == "default_clean":
        exp3476 = _exp3476_clean()

    artifacts = {
        3472: exp3472,
        3473: exp3473,
        3474: exp3474,
        3475: exp3475,
        3476: exp3476,
    }

    def _fake_load(exp_id: int) -> dict | None:
        return artifacts.get(exp_id)

    with (
        patch.object(synth, "load_artifact", side_effect=_fake_load),
        patch.object(synth, "_gate_eval", return_value=_make_gate()),
    ):
        return synth.synthesise()


# ─── is_flagged ──────────────────────────────────────────────────────────────


def test_is_flagged_true():
    # REQ-GATE-001 SCENARIO-GATE-003
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
    payload = {"experiment": 3472, "x": 1}
    fname = synth.DEPTH_BLOCK[3472]
    (tmp_path / fname).write_text(json.dumps(payload))
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(3472) == payload


def test_load_artifact_missing(tmp_path, monkeypatch):
    """File absent → None (not a crash)."""
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(3472) is None


def test_load_artifact_corrupt_json(tmp_path, monkeypatch):
    """Corrupt JSON → None."""
    fname = synth.DEPTH_BLOCK[3472]
    (tmp_path / fname).write_text("{not valid json")
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(3472) is None


def test_load_artifact_unknown_id(tmp_path, monkeypatch):
    """Unknown exp_id → None (not in DEPTH_BLOCK)."""
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(9999) is None


# ─── p0_1_v6_verdict + process_energy delta ──────────────────────────────────


def test_p0_1_verdict_blocked_contains_blocked():
    # SCENARIO-GATE-001: blocked exp3472 → verdict string contains "blocked"
    result = _run_synthesise()
    assert "blocked" in result["p0_1_v6_verdict"].lower()


def test_p0_1_verdict_blocked_delta_null():
    # SCENARIO-GATE-001: blocked → numeric delta excluded
    result = _run_synthesise()
    assert result["process_energy_vs_self_consistency_delta"] is None


def test_flip_count_null_when_blocked():
    # SCENARIO-GATE-001: blocked → flip_count null
    result = _run_synthesise()
    assert result["flip_count"] is None


def test_p0_1_verdict_clean_propagates_verdict():
    # SCENARIO-GATE-002: clean exp3472 → verdict and delta from artifact
    result = _run_synthesise(exp3472=_exp3472_clean())
    assert "blocked" not in result["p0_1_v6_verdict"].lower()
    assert result["process_energy_vs_self_consistency_delta"] == pytest.approx(0.032)


def test_flip_count_set_when_clean():
    # SCENARIO-GATE-002: clean exp3472 → flip_count populated
    result = _run_synthesise(exp3472=_exp3472_clean())
    assert result["flip_count"] == 8


def test_p0_1_verdict_missing_artifact():
    result = _run_synthesise(exp3472=None)
    assert "artifact_missing" in result["p0_1_v6_verdict"]
    assert result["process_energy_vs_self_consistency_delta"] is None


def test_p0_1_verdict_flagged():
    flagged = {**_exp3472_blocked(), "flagged_adversarial": True}
    result = _run_synthesise(exp3472=flagged)
    assert "flagged_adversarial" in result["p0_1_v6_verdict"]
    assert result["process_energy_vs_self_consistency_delta"] is None


# ─── minority_correct_recovery_rate (exp3473) ────────────────────────────────


def test_minority_recovery_null_when_flagged():
    # SCENARIO-GATE-003: flagged exp3473 → recovery rate excluded
    result = _run_synthesise()
    assert result["minority_correct_recovery_rate"] is None


def test_minority_recovery_populated_when_clean():
    # SCENARIO-GATE-004: clean exp3473 → rate populated
    result = _run_synthesise(exp3473=_exp3473_clean())
    assert result["minority_correct_recovery_rate"] == pytest.approx(0.60)


def test_minority_recovery_null_when_missing():
    result = _run_synthesise(exp3473=None)
    assert result["minority_correct_recovery_rate"] is None


# ─── fr11_depth_collapse_consequence (exp3474) ───────────────────────────────


def test_fr11_collapse_from_clean_artifact():
    # SCENARIO-GATE-005: clean exp3474 → consequence string from artifact
    result = _run_synthesise()
    assert "ARM A COLLAPSED" in result["fr11_depth_collapse_consequence"]


def test_fr11_collapse_contains_arm_b():
    result = _run_synthesise()
    assert "ARM B" in result["fr11_depth_collapse_consequence"]


def test_fr11_collapse_missing():
    result = _run_synthesise(exp3474=None)
    assert "artifact_missing" in result["fr11_depth_collapse_consequence"]


def test_fr11_collapse_flagged_notes_flag():
    flagged = {**_exp3474_clean(), "flagged_adversarial": True}
    result = _run_synthesise(exp3474=flagged)
    assert "flagged_adversarial" in result["fr11_depth_collapse_consequence"]


# ─── kona_process_hybrid_delta (exp3475) ─────────────────────────────────────


def test_kona_delta_null_when_saturated():
    # exp3475 blocked (saturated) → delta null
    result = _run_synthesise()
    assert result["kona_process_hybrid_delta"] is None


def test_kona_delta_null_when_missing():
    result = _run_synthesise(exp3475=None)
    assert result["kona_process_hybrid_delta"] is None


def test_kona_delta_null_when_flagged():
    flagged = {**_exp3475_blocked(), "flagged_adversarial": True}
    result = _run_synthesise(exp3475=flagged)
    assert result["kona_process_hybrid_delta"] is None


def test_kona_delta_set_when_clean_unblocked():
    clean = {
        "experiment": 3475,
        "honest_verdict": "complete: process_energy_lifts_kona_hybrid_delta_0.05",
        "delta_process_vs_untrained_hybrid": 0.05,
        "flagged_adversarial": False,
    }
    result = _run_synthesise(exp3475=clean)
    assert result["kona_process_hybrid_delta"] == pytest.approx(0.05)


# ─── g2_package_status (exp3476) ─────────────────────────────────────────────


def test_g2_package_status_from_clean_artifact():
    # SCENARIO-GATE-006: clean exp3476 → g2_status propagated
    result = _run_synthesise()
    assert result["g2_package_status"] == "self_contained_package_verified_external_run_pending"


def test_g2_package_status_artifact_missing():
    result = _run_synthesise(exp3476=None)
    assert "artifact_missing" in result["g2_package_status"]


def test_g2_package_status_flagged():
    flagged = {**_exp3476_clean(), "flagged_adversarial": True}
    result = _run_synthesise(exp3476=flagged)
    assert "flagged_adversarial" in result["g2_package_status"]


# ─── depth_forcing_function_can_relax ────────────────────────────────────────


def test_depth_forcing_false_when_p01_blocked():
    # SCENARIO-GATE-007: blocked corpus → cannot relax
    result = _run_synthesise()
    assert result["depth_forcing_function_can_relax"] is False


def test_depth_forcing_false_when_p01_missing():
    result = _run_synthesise(exp3472=None)
    assert result["depth_forcing_function_can_relax"] is False


def test_depth_forcing_false_even_with_clean_p01_no_external():
    # Even with a clean P0.1, if no external ask confirmed in motion → False.
    # external_ask_in_motion is hardcoded False in synthesise().
    result = _run_synthesise(exp3472=_exp3472_clean())
    assert result["depth_forcing_function_can_relax"] is False


# ─── G1–G4 gate passthrough ──────────────────────────────────────────────────


def test_g1_g3_g4_met():
    result = _run_synthesise()
    assert result["g1"] is True
    assert result["g3"] is True
    assert result["g4"] is True


def test_g2_not_met():
    result = _run_synthesise()
    assert result["g2"] is False


def test_unmet_gates_contains_g2():
    result = _run_synthesise()
    assert "G2" in result["unmet_gates"]


# ─── required schema fields ──────────────────────────────────────────────────


REQUIRED_FIELDS = {
    "honest_verdict",
    "g1",
    "g2",
    "g3",
    "g4",
    "unmet_gates",
    "p0_1_v6_verdict",
    "process_energy_vs_self_consistency_delta",
    "flip_count",
    "minority_correct_recovery_rate",
    "g2_package_status",
    "fr11_depth_collapse_consequence",
    "kona_process_hybrid_delta",
    "depth_forcing_function_can_relax",
    "gate_status_v320_ready",
}


def test_all_required_fields_present():
    # SCENARIO-GATE-008
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


def test_gate_status_v320_ready_true():
    result = _run_synthesise()
    assert result["gate_status_v320_ready"] is True


# ─── integration: output artifact valid ──────────────────────────────────────


def test_output_artifact_valid():
    """The actual artifact on disk has all required fields and correct values."""
    artifact_path = (
        PROJECT_ROOT / "results" / "experiment_3480_g_gate_status_synthesis_v320.json"
    )
    if not artifact_path.exists():
        pytest.skip("artifact not yet written — run the script first")
    data = json.loads(artifact_path.read_text())
    missing = REQUIRED_FIELDS - set(data)
    assert not missing, f"Artifact missing fields: {missing}"
    assert data["gate_status_v320_ready"] is True
    assert data["honest_verdict"].startswith("complete:")
    assert data["g2"] is False
    assert data["depth_forcing_function_can_relax"] is False
