"""Tests for scripts/publication_gate.py — the stable G1-G4 publication gate.

Focus: the G3 narrowing lint's retraction-context whitelist (the false-positive
class that the first implementation tripped on — flagging the project's own
honest retraction narrative), and the evaluate() composition.

Spec: ops/north-star.md §2 (stable gate replacing publication_blocker_count).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load():
    root = Path(__file__).resolve().parents[2]
    p = root / "scripts" / "publication_gate.py"
    spec = importlib.util.spec_from_file_location("publication_gate", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["publication_gate"] = m
    spec.loader.exec_module(m)
    return m


_M = _load()


class TestG3RetractionContextWhitelist:
    """A forbidden phrasing in retraction context must NOT fail G3."""

    def _g3_on_text(self, text: str, monkeypatch, tmp_path):
        doc = tmp_path / "technical-report.md"
        doc.write_text(text)
        monkeypatch.setattr(_M, "TECH_REPORT", doc)
        monkeypatch.setattr(_M, "PAPER_TEX", tmp_path / "nonexistent.tex")
        return _M.check_g3()

    def test_live_forbidden_claim_fails(self, monkeypatch, tmp_path):
        # 0.9857 asserted as a live headline — no retraction marker nearby
        g3 = self._g3_on_text(
            "The ensemble reaches AUROC 0.9857 on the FoVer corpus, our headline result.",
            monkeypatch, tmp_path,
        )
        assert g3["pass"] is False
        assert any("0.9857" in h for h in g3["hits"])

    def test_retraction_disclosure_passes(self, monkeypatch, tmp_path):
        # same number, but explaining the retraction — must be allowed
        g3 = self._g3_on_text(
            "This repins the earlier v2 headline of 0.9857 downward to 0.9131 after audit.",
            monkeypatch, tmp_path,
        )
        assert g3["pass"] is True, f"retraction context should pass; hits={g3['hits']}"

    def test_thermalization_negation_passes(self, monkeypatch, tmp_path):
        g3 = self._g3_on_text(
            "These are deterministic samples, not Boltzmann-thermalized samples; "
            "we remove the implication of equilibrium.",
            monkeypatch, tmp_path,
        )
        assert g3["pass"] is True, f"negated thermalization should pass; hits={g3['hits']}"

    def test_live_thermalization_claim_fails(self, monkeypatch, tmp_path):
        g3 = self._g3_on_text(
            "The KV260 sampler produces thermalized equilibrium samples at 24 microseconds.",
            monkeypatch, tmp_path,
        )
        assert g3["pass"] is False

    def test_unsupported_humaneval_claim_fails(self, monkeypatch, tmp_path):
        g3 = self._g3_on_text(
            "Carnot lifts HumanEval pass@1 from 0% to 36% on a 35B model.",
            monkeypatch, tmp_path,
        )
        assert g3["pass"] is False

    def test_clean_prose_passes(self, monkeypatch, tmp_path):
        g3 = self._g3_on_text(
            "The verifier ensemble reaches AUROC 0.9131 (5-seed, dual-condition).",
            monkeypatch, tmp_path,
        )
        assert g3["pass"] is True


class TestG2ExternalState:
    """G2 is read from the manual state file; defaults to UNMET."""

    def test_g2_unmet_by_default(self):
        assert _M.check_g2({})["pass"] is False

    def test_g2_met_when_recorded(self):
        g2 = _M.check_g2({"g2_independent_reproducer": True, "g2_evidence": "CG re-ran exp2837"})
        assert g2["pass"] is True
        assert "CG" in g2["detail"]


class TestEvaluateComposition:
    """paper_ready is the AND of all four gates; unmet_gates lists the failures."""

    def test_paper_ready_requires_all_four(self, monkeypatch):
        # force a known mix: G1 pass, G2 unmet, G3 pass, G4 pass
        monkeypatch.setattr(_M, "check_g1", lambda: {"pass": True, "detail": "x"})
        monkeypatch.setattr(_M, "check_g2", lambda s: {"pass": False, "detail": "no reproducer"})
        monkeypatch.setattr(_M, "check_g3", lambda: {"pass": True, "detail": "clean"})
        monkeypatch.setattr(_M, "check_g4", lambda: {"pass": True, "detail": "x"})
        monkeypatch.setattr(_M, "_load_state", lambda: {})
        r = _M.evaluate()
        assert r["paper_ready"] is False
        assert r["unmet_gates"] == ["G2"]

    def test_paper_ready_true_when_all_pass(self, monkeypatch):
        for g in ("check_g1", "check_g3", "check_g4"):
            monkeypatch.setattr(_M, g, lambda: {"pass": True, "detail": "x"})
        monkeypatch.setattr(_M, "check_g2", lambda s: {"pass": True, "detail": "reproduced"})
        monkeypatch.setattr(_M, "_load_state", lambda: {})
        r = _M.evaluate()
        assert r["paper_ready"] is True
        assert r["unmet_gates"] == []
