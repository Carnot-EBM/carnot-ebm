"""Tests for Exp 733 — Tier 2.1 JEPAReasonerProbe cascade integration.

Each test traces to at least one REQ-VER-03x requirement.

Coverage targets (code added by Exp 733 only):
  - python/carnot/cascade/tier21_probe.py   (Tier21ProbeWrapper, ViolationEventStub)
  - python/carnot/cascade/cascade_router.py  (Tier 2.1 routing paths added in Exp 733)
  - scripts/experiment_733_tier21_cascade.py (gate-blocked path, gate file schema)

Spec: REQ-VER-035, REQ-VER-036, REQ-VER-037,
      SCENARIO-VER-044, SCENARIO-VER-045, SCENARIO-VER-046
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HIDDEN_DIM = 1024


def _make_trained_probe(seed: int = 0):
    """Return a JEPAReasonerProbe with a trained _probe (NumPy MLP).

    We train on tiny synthetic data so the probe weights are non-trivial and
    the predict() method actually runs the linear layers.
    """
    from carnot.samplers.jepa_reasoner_probe import JEPAReasonerProbe

    rng = np.random.default_rng(seed)
    n = 100
    X = rng.standard_normal((n, HIDDEN_DIM)).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    # Make first half clearly violation, second half clearly correct.
    X[:50, :256] += 3.0
    y[:50] = 1.0
    y[50:] = 0.0

    probe = JEPAReasonerProbe(device="cpu")
    probe.train_probe(X, y, n_epochs=20)
    return probe


# ---------------------------------------------------------------------------
# Tests for Tier21ProbeWrapper (tier21_probe.py)
# ---------------------------------------------------------------------------


class TestTier21ProbeWrapper:
    """Unit tests for Tier21ProbeWrapper.  Spec: REQ-VER-035, REQ-VER-036, REQ-VER-037."""

    def test_likely_correct_verdict_below_threshold(self):
        """Probe score <= threshold returns "likely_correct".

        Spec: REQ-VER-035-3, REQ-VER-036, SCENARIO-VER-044
        """
        from carnot.cascade.tier21_probe import Tier21ProbeWrapper

        probe = _make_trained_probe()
        # Use a threshold that is definitely above the probe output for a zero vector.
        threshold = 0.99
        wrapper = Tier21ProbeWrapper(probe=probe, threshold=threshold)

        hs = np.zeros(HIDDEN_DIM, dtype=np.float32)
        score, verdict = wrapper.score(hs)
        assert verdict == "likely_correct", f"Expected likely_correct, got {verdict}"
        assert 0.0 <= score <= 1.0

    def test_likely_violation_verdict_above_threshold(self):
        """Probe score > threshold returns "likely_violation".

        Spec: REQ-VER-035-3, REQ-VER-037, SCENARIO-VER-045
        """
        from carnot.cascade.tier21_probe import Tier21ProbeWrapper

        probe = _make_trained_probe()
        # Use a threshold of 0.0 so any positive score is a violation.
        threshold = 0.0
        wrapper = Tier21ProbeWrapper(probe=probe, threshold=threshold)

        # A large violation-pattern hidden state should produce a score above 0.0.
        hs = np.ones(HIDDEN_DIM, dtype=np.float32) * 3.0
        hs[:256] += 3.0
        score, verdict = wrapper.score(hs)
        assert verdict == "likely_violation", f"Expected likely_violation, got {verdict}"

    def test_emit_violation_stub_appends_to_log(self):
        """emit_violation_stub appends a ViolationEventStub to violation_log.

        Spec: REQ-VER-037-1, REQ-VER-037-3, SCENARIO-VER-045
        """
        from carnot.cascade.tier21_probe import Tier21ProbeWrapper, ViolationEventStub

        probe = _make_trained_probe()
        violation_log: list = []
        wrapper = Tier21ProbeWrapper(probe=probe, threshold=0.5, violation_log=violation_log)

        wrapper.emit_violation_stub(query_id="q_0000001", probe_score=0.87)

        assert len(violation_log) == 1
        event = violation_log[0]
        assert isinstance(event, ViolationEventStub)
        assert event.query_id == "q_0000001"
        assert abs(event.probe_score - 0.87) < 1e-6
        assert event.timestamp_utc  # non-empty ISO timestamp

    def test_emit_violation_stub_calls_custom_event_bus(self):
        """emit_violation_stub calls the user-supplied event_bus callable.

        Spec: REQ-VER-037-2
        """
        from carnot.cascade.tier21_probe import Tier21ProbeWrapper, ViolationEventStub

        probe = _make_trained_probe()
        received: list = []
        wrapper = Tier21ProbeWrapper(probe=probe, threshold=0.5, event_bus=received.append)

        wrapper.emit_violation_stub(query_id="q_test", probe_score=0.99)

        assert len(received) == 1
        assert isinstance(received[0], ViolationEventStub)
        assert received[0].query_id == "q_test"

    def test_violation_event_stub_to_dict(self):
        """ViolationEventStub.to_dict() returns all required fields.

        Spec: REQ-VER-037-3
        """
        from carnot.cascade.tier21_probe import ViolationEventStub

        event = ViolationEventStub(query_id="q_007", probe_score=0.77, timestamp_utc="2026-04-22T00:00:00Z")
        d = event.to_dict()
        assert d["query_id"] == "q_007"
        assert abs(d["probe_score"] - 0.77) < 1e-6
        assert d["timestamp_utc"] == "2026-04-22T00:00:00Z"

    def test_missing_hidden_state_fn_raises(self):
        """CascadeRouter raises ValueError when tier21_probe supplied without hidden_state_fn.

        Spec: REQ-VER-035-2 (misconfiguration must be caught at construction time)
        """
        from carnot.cascade.cascade_router import CascadeRouter
        from carnot.cascade.tier21_probe import Tier21ProbeWrapper

        probe = _make_trained_probe()
        wrapper = Tier21ProbeWrapper(probe=probe, threshold=0.5)
        with pytest.raises(ValueError, match="hidden_state_fn is required"):
            CascadeRouter(
                eorm_fn=lambda q: 0.5,
                ising_fn=lambda q: True,
                tier21_probe=wrapper,
                hidden_state_fn=None,
            )


# ---------------------------------------------------------------------------
# Tests for cascade_router.py Tier 2.1 paths
# ---------------------------------------------------------------------------


class TestCascadeRouterTier21:
    """Tests for the Tier 2.1 integration in CascadeRouter.

    Spec: REQ-VER-035, REQ-VER-036, REQ-VER-037,
          SCENARIO-VER-044, SCENARIO-VER-045
    """

    def _make_router_with_tier21(self, probe_score: float, threshold: float):
        """Build a CascadeRouter with a stub probe that returns a fixed score."""
        from carnot.cascade.cascade_router import CascadeRouter
        from carnot.cascade.tier21_probe import Tier21ProbeWrapper

        probe = _make_trained_probe()
        violation_log: list = []
        wrapper = Tier21ProbeWrapper(probe=probe, threshold=threshold, violation_log=violation_log)

        # Patch wrapper.score to return deterministic (probe_score, verdict).
        def _fixed_score(hs):
            verdict = "likely_correct" if probe_score <= threshold else "likely_violation"
            return probe_score, verdict

        wrapper.score = _fixed_score  # type: ignore[method-assign]

        router = CascadeRouter(
            eorm_fn=lambda q: 0.5,  # below EORM gate → reaches Tier 2.1
            ising_fn=lambda q: True,
            tier21_probe=wrapper,
            hidden_state_fn=lambda q: np.zeros(HIDDEN_DIM, dtype=np.float32),
        )
        return router, violation_log

    def test_tier21_likely_correct_skips_downstream(self):
        """When probe says likely_correct, route returns "likely_correct" and tier21_skip=True.

        Spec: REQ-VER-036-1, REQ-VER-036-2, SCENARIO-VER-044
        """
        router, _ = self._make_router_with_tier21(probe_score=0.1, threshold=0.5)
        result = router.route("test_query")
        assert result.verdict == "likely_correct"
        assert result.metadata["tier21_skip"] is True
        assert "probe_score" in result.metadata

    def test_tier21_likely_violation_continues_cascade(self):
        """When probe says likely_violation, cascade continues to Ising and tier21_skip=False.

        Spec: REQ-VER-036-1, SCENARIO-VER-045
        """
        router, _ = self._make_router_with_tier21(probe_score=0.9, threshold=0.5)
        result = router.route("test_query")
        # Ising fn returns True → "verified_full"
        assert result.verdict == "verified_full"
        assert result.metadata["tier21_skip"] is False
        assert "probe_score" in result.metadata

    def test_tier21_violation_emits_stub(self):
        """When probe says likely_violation, emit_violation_stub is called.

        Spec: REQ-VER-037-1, SCENARIO-VER-045
        """
        router, violation_log = self._make_router_with_tier21(probe_score=0.9, threshold=0.5)
        router.route("test_query")
        assert len(violation_log) == 1, "Expected exactly one ViolationEventStub in log"

    def test_no_tier21_probe_follows_original_path(self):
        """Without tier21_probe, router follows original EORM+Ising path unchanged.

        Spec: REQ-INFRA-046 — existing behaviour must not regress.
        """
        from carnot.cascade.cascade_router import CascadeRouter

        router = CascadeRouter(
            eorm_fn=lambda q: 0.5,
            ising_fn=lambda q: True,
        )
        result = router.route("test_query")
        assert result.verdict == "verified_full"
        assert result.metadata == {}

    def test_probe_score_always_in_metadata(self):
        """probe_score is in metadata regardless of skip decision (REQ-VER-036-2).

        Spec: REQ-VER-036-2
        """
        for probe_score in (0.01, 0.99):
            router, _ = self._make_router_with_tier21(probe_score=probe_score, threshold=0.5)
            result = router.route("q")
            assert "probe_score" in result.metadata, (
                f"probe_score missing from metadata for probe_score={probe_score}"
            )


# ---------------------------------------------------------------------------
# Tests for experiment_733_tier21_cascade.py — gate-blocked path and gate schema
# ---------------------------------------------------------------------------


class TestExp733GateAndSchema:
    """Tests for the gate-blocked path and the gate file schema.

    Spec: REQ-VER-035 (gate check), SCENARIO-VER-046 (gate file schema)
    """

    def test_gated_blocked_when_gate_fails(self, tmp_path, monkeypatch):
        """When tier21_gate.json has gate=fail, write gated_blocked artifact and stop.

        Spec: REQ-VER-035 (pre-condition: xval must pass before wiring)
        """
        gate_file = tmp_path / "tier21_gate.json"
        gate_file.write_text(json.dumps({"gate": "fail", "reason": "low_auc"}))

        deliverable = tmp_path / "experiment_733_tier21_cascade.json"

        # Patch constants in the script module.
        repo_root = Path(__file__).resolve().parent.parent.parent
        script_path = str(repo_root / "scripts")
        if script_path not in sys.path:
            sys.path.insert(0, script_path)

        import importlib
        import experiment_733_tier21_cascade as exp733  # noqa: PLC0415

        monkeypatch.setattr(exp733, "GATE_SOURCE_FILE", str(gate_file))
        monkeypatch.setattr(exp733, "DELIVERABLE", str(deliverable))
        monkeypatch.setattr(exp733, "CASCADE_GATE_FILE", str(tmp_path / "cascade_gate.json"))

        exp733.main()

        assert deliverable.exists(), "Deliverable must be written even on gate fail"
        with open(deliverable) as f:
            artifact = json.load(f)
        assert artifact["status"] == "gated_blocked"
        assert artifact["gate_source"] == "exp732"
        assert artifact["honest_verdict"] == "gated_blocked_probe_xval_failed"
        assert artifact["schema"] == "carnot.result.v1"

    def test_gate_file_schema_on_success(self, tmp_path, monkeypatch):
        """On success, tier21_cascade_gate.json must have correct schema.

        Spec: SCENARIO-VER-046
        """
        gate_file = tmp_path / "tier21_gate.json"
        gate_file.write_text(json.dumps({
            "gate": "pass",
            "mean_auc": 0.99,
            "std_auc": 0.005,
        }))

        deliverable = tmp_path / "experiment_733_tier21_cascade.json"
        cascade_gate_file = tmp_path / "tier21_cascade_gate.json"

        repo_root = Path(__file__).resolve().parent.parent.parent
        script_path = str(repo_root / "scripts")
        if script_path not in sys.path:
            sys.path.insert(0, script_path)

        import importlib
        import experiment_733_tier21_cascade as exp733  # noqa: PLC0415

        monkeypatch.setattr(exp733, "GATE_SOURCE_FILE", str(gate_file))
        monkeypatch.setattr(exp733, "DELIVERABLE", str(deliverable))
        monkeypatch.setattr(exp733, "CASCADE_GATE_FILE", str(cascade_gate_file))

        exp733.main()

        assert cascade_gate_file.exists(), "Cascade gate file must be written"
        with open(cascade_gate_file) as f:
            cg = json.load(f)

        # SCENARIO-VER-046: all four fields required.
        for field in ("gate", "skip_rate_symcode", "fn_delta", "probe_latency_p99_ms"):
            assert field in cg, f"Missing required field '{field}' in cascade gate"

        assert cg["gate"] in ("pass", "fail")
        assert 0.0 <= cg["skip_rate_symcode"] <= 1.0
        assert isinstance(cg["fn_delta"], float)
        assert isinstance(cg["probe_latency_p99_ms"], float)
