"""Tests for Exp 2394 HALT cached-logprob Tier 0j probe.

Spec: REQ-TIER0-008, SCENARIO-TIER0-007
"""

from __future__ import annotations

import math

from carnot.verify.halt_probe import HaltProbeDetector, oof_halt_risk_scores


def _entry(scale: float) -> dict:
    return {
        "token_logprobs": [-0.05 * scale, -0.15 * scale, -0.4 * scale],
        "top_logprobs": [
            {" yes": -0.05 * scale, " no": -1.0 * scale, " maybe": -2.0 * scale},
            {" 1": -0.15 * scale, " 2": -1.5 * scale, " 3": -3.0 * scale},
            {"</think>": -0.4 * scale, " answer": -1.8 * scale, "\n": -4.0 * scale},
        ],
    }


def test_verify_returns_finite_halt_risk_score() -> None:
    """REQ-TIER0-008-1/4: verify exposes a finite score and proxy name."""
    result = HaltProbeDetector().verify(_entry(scale=1.0))

    assert math.isfinite(result["halt_risk_score"])
    assert 0.0 <= result["halt_risk_score"] <= 1.0
    assert result["proxy_used"] in {"A+B", "A", "B", "token_logprobs"}


def test_proxy_c_oof_scores_are_nontrivial_probabilities() -> None:
    """REQ-TIER0-008-3: proxy C produces deterministic non-constant risk scores."""
    entries = [_entry(0.5), _entry(0.7), _entry(2.0), _entry(2.5)]
    labels = [0, 0, 1, 1]

    scores = oof_halt_risk_scores(entries, labels, random_seed=42, n_splits=2)

    assert len(set(round(score, 6) for score in scores)) > 1
    assert all(0.0 <= score <= 1.0 for score in scores)
