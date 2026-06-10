"""Tests for scripts/experiment_3456_g_gate_status_synthesis_v318.py.

REQ-GATE-001: G1–G4 gate status synthesis must read depth-block artifacts,
  skip flagged_adversarial ones, and emit a structured record with all
  required schema fields.
SCENARIO-GATE-001: exp3449 flagged_adversarial → p0_1_v4_verdict contains
  "flagged_adversarial" and energy_vs_self_consistency_delta is null.
SCENARIO-GATE-002: clean exp3449 → delta is set from artifact.
SCENARIO-GATE-003: exp3450 (clean) → energy_correctness_auroc populated.
SCENARIO-GATE-004: exp3451 (clean) → g2_ci_status from artifact.
SCENARIO-GATE-005: exp3452 flagged → fr11_collapse_consequence notes flag
  but still records directional verdict string.
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

# Add scripts/ to path so we can import the module under test.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import experiment_3456_g_gate_status_synthesis_v318 as synth  # noqa: E402


# ─── fixture helpers ────────────────────────────────────────────────────────


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


def _exp3449_flagged() -> dict:
    return {
        "experiment": 3449,
        "honest_verdict": "complete: energy_matches_but_does_not_beat_self_consistency_at_equal_compute",
        "self_consistency_non_degenerate": True,
        "delta_energy_vs_self_consistency": 0.0,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "..."}],
    }


def _exp3449_clean() -> dict:
    return {
        "experiment": 3449,
        "honest_verdict": "complete: energy_beats_self_consistency_delta_0.032",
        "self_consistency_non_degenerate": True,
        "delta_energy_vs_self_consistency": 0.032,
        "flagged_adversarial": False,
    }


def _exp3450_clean() -> dict:
    return {
        "experiment": 3450,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "honest_verdict": "complete: energy_does_not_track_correctness_explains_p01_ceiling",
        "energy_as_correctness_auroc": 0.516,
    }


def _exp3451_clean() -> dict:
    return {
        "experiment": 3451,
        "honest_verdict": "complete: fover_g2_ci_and_docker_cleanroom_ready_external_run_pending",
        "g2_status": "ci_and_docker_ready_external_run_pending",
        "g2_independent_reproducer": False,
    }


def _exp3452_flagged() -> dict:
    return {
        "experiment": 3452,
        "honest_verdict": "complete: at_risk_grounding_causes_collapse_entropy_reg_prevents_it",
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "..."}],
    }


def _exp3452_clean() -> dict:
    return {
        "experiment": 3452,
        "honest_verdict": "complete: at_risk_grounding_causes_collapse_entropy_reg_prevents_it",
        "grounding_collapse_consequence": "ARM A collapsed; ARM B did not.",
        "flagged_adversarial": False,
    }


_DEFAULT_ARTIFACT = object()


def _artifact_loader(
    *,
    exp3449: dict | None = None,
    exp3450: dict | None | object = _DEFAULT_ARTIFACT,
    exp3451: dict | None | object = _DEFAULT_ARTIFACT,
    exp3452: dict | None | object = _DEFAULT_ARTIFACT,
) -> dict:
    """Return artifacts dict for patching load_artifact."""
    return {
        3448: {
            "experiment": 3448,
            "honest_verdict": "complete: p01_generation_corpus_partial_resumable",
        },
        3449: exp3449,
        3450: _exp3450_clean() if exp3450 is _DEFAULT_ARTIFACT else exp3450,
        3451: _exp3451_clean() if exp3451 is _DEFAULT_ARTIFACT else exp3451,
        3452: _exp3452_flagged() if exp3452 is _DEFAULT_ARTIFACT else exp3452,
    }


def _run_synthesise(**kwargs) -> dict:
    """Run synth.synthesise() with controlled artifact loading and gate."""
    artifacts = _artifact_loader(**kwargs)

    def _fake_load(exp_id: int) -> dict | None:
        return artifacts.get(exp_id)

    class _FakeGateMod:
        @staticmethod
        def evaluate() -> dict:
            return _make_gate()

    with (
        patch.object(synth, "load_artifact", side_effect=_fake_load),
        patch.dict(sys.modules, {"publication_gate": _FakeGateMod()}),
    ):
        return synth.synthesise()


# ─── is_flagged ─────────────────────────────────────────────────────────────


def test_is_flagged_true():
    # REQ-GATE-001 SCENARIO-GATE-001
    assert synth.is_flagged({"flagged_adversarial": True}) is True


def test_is_flagged_false():
    assert synth.is_flagged({"flagged_adversarial": False}) is False


def test_is_flagged_missing_key():
    assert synth.is_flagged({}) is False


def test_is_flagged_none_artifact():
    assert synth.is_flagged(None) is False


# ─── load_artifact ───────────────────────────────────────────────────────────


def test_load_artifact_exists(tmp_path, monkeypatch):
    # Write a real artifact and confirm load_artifact reads it.
    payload = {"experiment": 3448, "x": 1}
    fname = synth.DEPTH_BLOCK[3448]
    (tmp_path / fname).write_text(json.dumps(payload))
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(3448) == payload


def test_load_artifact_missing(tmp_path, monkeypatch):
    # File absent → None.
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(3448) is None


def test_load_artifact_corrupt_json(tmp_path, monkeypatch):
    # Corrupt JSON → None.
    fname = synth.DEPTH_BLOCK[3448]
    (tmp_path / fname).write_text("{not valid json")
    monkeypatch.setattr(synth, "RESULTS_DIR", tmp_path)
    assert synth.load_artifact(3448) is None


# ─── synthesise: p0_1_v4_verdict ─────────────────────────────────────────────


def test_p0_1_verdict_flagged(monkeypatch):
    # SCENARIO-GATE-001: flagged exp3449 → p0_1_v4_verdict contains "flagged"
    result = _run_synthesise(exp3449=_exp3449_flagged())
    assert "flagged_adversarial" in result["p0_1_v4_verdict"]


def test_p0_1_verdict_flagged_delta_null(monkeypatch):
    # SCENARIO-GATE-001: flagged exp3449 → delta is null (numbers excluded)
    result = _run_synthesise(exp3449=_exp3449_flagged())
    assert result["energy_vs_self_consistency_delta"] is None


def test_p0_1_verdict_clean(monkeypatch):
    # SCENARIO-GATE-002: clean exp3449 → delta taken from artifact
    result = _run_synthesise(exp3449=_exp3449_clean())
    assert "flagged" not in result["p0_1_v4_verdict"]
    assert result["energy_vs_self_consistency_delta"] == pytest.approx(0.032)


def test_p0_1_verdict_missing_artifact():
    # Missing exp3449 → "artifact_missing" in verdict
    result = _run_synthesise(exp3449=None)
    assert "artifact_missing" in result["p0_1_v4_verdict"]


# ─── synthesise: energy_correctness_auroc ────────────────────────────────────


def test_energy_correctness_auroc_from_clean_exp3450():
    # SCENARIO-GATE-003
    result = _run_synthesise(exp3449=_exp3449_flagged(), exp3450=_exp3450_clean())
    assert result["energy_correctness_auroc"] == pytest.approx(0.516)


def test_energy_correctness_auroc_none_when_missing():
    result = _run_synthesise(exp3449=_exp3449_flagged(), exp3450=None)
    assert result["energy_correctness_auroc"] is None


# ─── synthesise: g2_ci_status ────────────────────────────────────────────────


def test_g2_ci_status_from_clean_exp3451():
    # SCENARIO-GATE-004
    result = _run_synthesise(exp3449=_exp3449_flagged())
    assert result["g2_ci_status"] == "ci_and_docker_ready_external_run_pending"


def test_g2_ci_status_missing_artifact():
    result = _run_synthesise(exp3449=_exp3449_flagged(), exp3451=None)
    assert "missing" in result["g2_ci_status"] or "artifact" in result["g2_ci_status"]


# ─── synthesise: fr11_collapse_consequence ───────────────────────────────────


def test_fr11_collapse_consequence_notes_flagged():
    # SCENARIO-GATE-005: exp3452 flagged → notes it but records directional verdict
    result = _run_synthesise(exp3449=_exp3449_flagged(), exp3452=_exp3452_flagged())
    cons = result["fr11_collapse_consequence"]
    assert "flagged_adversarial" in cons
    assert "collapse" in cons.lower()


def test_fr11_collapse_consequence_clean():
    # Clean exp3452 → uses grounding_collapse_consequence field
    result = _run_synthesise(exp3449=_exp3449_flagged(), exp3452=_exp3452_clean())
    assert "ARM A collapsed" in result["fr11_collapse_consequence"]


def test_fr11_collapse_consequence_missing():
    result = _run_synthesise(exp3449=_exp3449_flagged(), exp3452=None)
    assert "missing" in result["fr11_collapse_consequence"]


# ─── synthesise: depth_forcing_function_can_relax ────────────────────────────


def test_depth_forcing_false_when_p01_flagged():
    # SCENARIO-GATE-006: exp3449 flagged → no clean verdict → cannot relax
    result = _run_synthesise(exp3449=_exp3449_flagged())
    assert result["depth_forcing_function_can_relax"] is False


def test_depth_forcing_false_when_p01_missing():
    result = _run_synthesise(exp3449=None)
    assert result["depth_forcing_function_can_relax"] is False


def test_depth_forcing_false_even_with_clean_p01_no_external_ask():
    # Even with clean P0.1, if external ask not in motion → False
    # (this models the current state: CI/Docker ready but no external ask confirmed)
    result = _run_synthesise(exp3449=_exp3449_clean())
    # The current synthesise() has external_ask_in_motion=False hardcoded
    # until a non-operator confirms a run.
    assert result["depth_forcing_function_can_relax"] is False


# ─── synthesise: required schema fields ──────────────────────────────────────


REQUIRED_FIELDS = {
    "honest_verdict",
    "g1",
    "g2",
    "g3",
    "g4",
    "unmet_gates",
    "p0_1_v4_verdict",
    "energy_vs_self_consistency_delta",
    "energy_correctness_auroc",
    "g2_ci_status",
    "fr11_collapse_consequence",
    "depth_forcing_function_can_relax",
    "gate_status_v318_ready",
}


def test_all_required_fields_present():
    # SCENARIO-GATE-007
    result = _run_synthesise(exp3449=_exp3449_flagged())
    missing = REQUIRED_FIELDS - set(result)
    assert not missing, f"Missing schema fields: {missing}"


def test_honest_verdict_has_terminal_prefix():
    result = _run_synthesise(exp3449=_exp3449_flagged())
    v = result["honest_verdict"]
    assert any(
        v.startswith(p) for p in ("complete:", "complete_", "success:", "success_",
                                   "passed:", "passed_", "shipped:", "shipped_")
    ), f"Non-terminal prefix in honest_verdict: {v!r}"


def test_gate_status_v318_ready_true():
    result = _run_synthesise(exp3449=_exp3449_flagged())
    assert result["gate_status_v318_ready"] is True


def test_unmet_gates_is_list():
    result = _run_synthesise(exp3449=_exp3449_flagged())
    assert isinstance(result["unmet_gates"], list)
    assert "G2" in result["unmet_gates"]


def test_g1_true_g3_true_g4_true():
    result = _run_synthesise(exp3449=_exp3449_flagged())
    assert result["g1"] is True
    assert result["g3"] is True
    assert result["g4"] is True


# ─── integration: output file is valid JSON with required fields ─────────────


def test_output_artifact_valid(tmp_path, monkeypatch):
    """The real output artifact (already on disk) is valid JSON with required fields."""
    # REQ-GATE-001: validate the actual artifact written by the script.
    artifact_path = (
        PROJECT_ROOT / "results" / "experiment_3456_g_gate_status_synthesis_v318.json"
    )
    if not artifact_path.exists():
        pytest.skip("artifact not yet written (run the script first)")
    data = json.loads(artifact_path.read_text())
    missing = REQUIRED_FIELDS - set(data)
    assert not missing, f"Artifact missing fields: {missing}"
    assert data["gate_status_v318_ready"] is True
    assert data["honest_verdict"].startswith("complete:")
