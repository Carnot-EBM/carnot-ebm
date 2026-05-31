"""Tests for FR-11 Conservative-Default Beta Deploy Non-Degenerate Corpus v3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11.conservative_default_deploy_nondegenerate_corpus_v3 import (
    CONSERVATIVE_DEFAULT_BETA,
    FRESH_DEPLOY_CONFIG,
    RANDOM_SEED,
    run_arm_closed_loop,
    run_conservative_default_deploy_nondegenerate_corpus_v3,
)

@pytest.fixture()
def small_traces() -> list[dict]:
    """30-trace corpus: every 5th trace is correct."""
    return [
        {
            "question_id": f"q{i:03d}",
            "prompt": f"Q{i}",
            "completion": f"A{i}",
            "is_correct": (i % 5 == 0),
            "trace_metadata": {},
        }
        for i in range(30)
    ]

@pytest.fixture()
def traces_jsonl(small_traces, tmp_path) -> str:
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for t in small_traces:
            f.write(json.dumps(t) + "\n")
    return str(path)

def test_run_arm_closed_loop():
    # Construct synthetic inputs for the loop
    n = 20
    traces = [{"is_correct": i % 2 == 0} for i in range(n)]
    at_risk_scores = np.random.RandomState(42).uniform(-0.1, 0.1, n)
    
    res = run_arm_closed_loop(
        traces=traces,
        at_risk_scores=at_risk_scores,
        n_iterations=5,
        entropy_beta=0.5,
        config_name="test_config",
        arm_label="DEPLOY"
    )
    
    assert "collapse_detected" in res
    assert "final_entropy" in res
    assert "final_pass_rate" in res
    assert "final_true_accuracy" in res
    assert "entropy_drop_ratio" in res

def test_run_conservative_default_deploy_nondegenerate_corpus_v3(traces_jsonl):
    # To pass the initial_true_acc in [0.3, 0.6] gate, our 30 trace fixture with every 5th correct = 6/30 = 0.2.
    # We need to increase correctness to get >= 0.3. Let's make every 2nd trace correct (0.5).
    traces = []
    with open(traces_jsonl) as f:
        for line in f:
            if not line.strip(): continue
            t = json.loads(line)
            # Override is_correct
            traces.append(t)
    
    for i, t in enumerate(traces):
        t["is_correct"] = (i % 2 == 0)
        
    with open(traces_jsonl, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    res = run_conservative_default_deploy_nondegenerate_corpus_v3(
        traces_path=traces_jsonl,
        n_iterations=5,
        seed=RANDOM_SEED,
        fresh_config={"name": "test", "active_weight": 0.045}
    )
    
    assert "honest_verdict" in res
    assert "inference_substrate" in res
    assert "deploy_arm_final_true_accuracy" in res
    assert "quality_maintained" in res
    assert "conservative_default_beta" in res
    assert "pass_rate_vs_true_accuracy_distinct_assert" in res

def test_run_blocked_no_traces(tmp_path):
    res = run_conservative_default_deploy_nondegenerate_corpus_v3(
        traces_path=str(tmp_path / "nonexistent.jsonl"),
        n_iterations=1,
    )
    assert res["honest_verdict"] == "complete: blocked_fr11_module_or_traces_unavailable"
