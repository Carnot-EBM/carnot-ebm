"""Tests for Q11 TSS sign-bottleneck instrumentation.

Spec refs: REQ-KONA-026, SCENARIO-KONA-026.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from carnot.phase3.continuous_ebm import ContinuousEBM


def test_req_kona_026_tss_diagnose_returns_required_fields() -> None:
    """REQ-KONA-026: ContinuousEBM reports the Q11 TSS diagnostic fields."""

    model = ContinuousEBM(
        variables=6,
        coupling=np.eye(6, dtype=np.float64) * 0.1,
        bias=np.linspace(-0.2, 0.2, 6),
    )
    triples = [
        ("What is 2 + 2?", "2 + 2 = 4, so the answer is 4.", True),
        ("What is 3 + 5?", "3 + 5 = 9, so the answer is 9.", False),
        ("What is 7 - 2?", "7 - 2 = 5, so the answer is 5.", True),
        ("What is 9 - 4?", "9 - 4 = 8, so the answer is 8.", False),
        ("Name a primary color.", "Blue is a primary color.", True),
    ]

    artifact = model.tss_diagnose(triples, n_steps=8, seed=1264)

    assert {
        "sc_energy_z3_correlation",
        "optimal_transversal_k",
        "tss_vulnerability_score",
        "tss_instrumented",
        "sign_z_bottleneck_diagnosed",
        "ste_pipeline_risk",
        "honest_verdict",
    } <= set(artifact)
    assert math.isfinite(artifact["sc_energy_z3_correlation"])
    assert math.isfinite(artifact["tss_vulnerability_score"])
    assert artifact["optimal_transversal_k"] == 2
    assert artifact["tss_instrumented"] is True
    assert artifact["sign_z_bottleneck_diagnosed"] is True
    assert artifact["tss_vulnerability_score"] == pytest.approx(
        round(1.0 - abs(artifact["sc_energy_z3_correlation"]), 4),
        abs=1e-4,
    )
    assert re.fullmatch(
        r"tss_instrumented_corr_-?\d+\.\d{3}_vuln_\d+\.\d{3}",
        artifact["honest_verdict"],
    )
