"""Tests for Exp 3344 constrained output extractor llguidance smoke test.

REQ-INFER-SOTA-3344
SCENARIO-INFER-SOTA-3344-001
"""

from pathlib import Path
from unittest import mock

import pytest

from carnot.reporting.constrained_output_extractor_llguidance_smoke_v1_3344 import run_experiment


def test_run_experiment_missing_dependencies(tmp_path: Path):
    """Test that missing llguidance/xgrammar results in a blocked artifact."""
    
    # Mock find_spec to simulate missing dependencies
    with mock.patch("carnot.reporting.constrained_output_extractor_llguidance_smoke_v1_3344.find_spec", return_value=None):
        
        # We also need to mock cached_sota_pair since tests shouldn't hit real cache if possible,
        # but let's see what happens. The prompt says "require at least one mandated model".
        with mock.patch("carnot.reporting.constrained_output_extractor_llguidance_smoke_v1_3344.cached_sota_pair", return_value=[{"name": "mock_model"}]):
            
            artifact = run_experiment(tmp_path)
            
            assert artifact["constrained_tool"] == "none"
            assert not artifact["constrained_extractor_ready"]
            assert len(artifact["blocked_reasons"]) > 0
            assert "neither llguidance nor xgrammar" in artifact["blocked_reasons"][0]
            assert artifact["honest_verdict"].startswith("blocked:")
            assert artifact["duration_s"] >= 0.0

def test_run_experiment_missing_sota_pair(tmp_path: Path):
    """Test behavior when SOTA models are missing."""
    
    with mock.patch("carnot.reporting.constrained_output_extractor_llguidance_smoke_v1_3344.find_spec", return_value=None):
        with mock.patch("carnot.reporting.constrained_output_extractor_llguidance_smoke_v1_3344.cached_sota_pair", return_value=None):
            artifact = run_experiment(tmp_path)
            assert not artifact["constrained_extractor_ready"]
            
            reasons = " ".join(artifact["blocked_reasons"])
            assert "blocked_sota_gguf_unavailable" in reasons
