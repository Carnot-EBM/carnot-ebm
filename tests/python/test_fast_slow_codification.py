import json
import os
import pytest
from carnot.codification.fast_slow import get_fast_slow_codified_metrics

# REQ-FAST-SLOW-CODIFICATION: The Fast-Slow variant metrics must match exp1811 and exp1909.
def test_fast_slow_codified_metrics():
    metrics = get_fast_slow_codified_metrics()
    assert metrics["exp1811"]["sample_efficiency_ratio"] == 3.1
    assert metrics["exp1811"]["kl_drift_ratio"] == 0.25
    assert metrics["exp1909"]["confirmation_sample_efficiency_ratio"] == 3.0

def test_paper_v6_section_3_exists():
    path = "openspec/papers/paper-v6/section-3-architecture.md"
    assert os.path.exists(path)
    with open(path, "r") as f:
        content = f.read()
        assert "3.1x" in content
        assert "172911" in content
        assert "3.0x" in content
        assert "192737" in content

def test_research_references_updated():
    with open("research-references.md", "r") as f:
        content = f.read()
        assert "arXiv:2605.12484" in content
        assert "arXiv:2602.23681" in content
        assert "arXiv:2602.02991" in content

def test_known_issues_updated():
    with open("ops/known-issues.md", "r") as f:
        content = f.read()
        assert "Fast-Slow Variant CONFIRMED" in content
