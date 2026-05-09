"""Tests for Exp 1590 STATIC-style CSR mask prototype for DSL."""

import json
from pathlib import Path

import pytest

from carnot.samplers import dsl_csr_mask as exp


def test_req_verify_1590_csr_mask_matches() -> None:
    """REQ-VERIFY-1590: CSR mask exact matching works correctly."""
    mask = exp.build_dsl_csr_mask(["json", "JSON"])
    assert mask.accepts("json") is True
    assert mask.accepts("JSON") is True
    assert mask.accepts("not json") is False
    assert mask.state_count == 9


def test_req_verify_1590_csr_mask_empty_raises() -> None:
    """REQ-VERIFY-1590: Cannot build mask from empty strings."""
    with pytest.raises(ValueError, match="at least one string"):
        exp.build_dsl_csr_mask([])


def test_req_verify_1590_benchmark_repeats() -> None:
    """REQ-VERIFY-1590: Benchmark repeats must be positive."""
    mask = exp.build_dsl_csr_mask(["json"])
    with pytest.raises(ValueError, match="positive"):
        exp.benchmark_acceptors(["json"], mask, repeats=0)


def test_req_verify_1590_regex_path() -> None:
    """REQ-VERIFY-1590: Regex path matching works correctly."""
    assert exp.dsl_regex_path_accepts("json") is True
    assert exp.dsl_regex_path_accepts("JSON") is True
    assert exp.dsl_regex_path_accepts("not matching") is False


def test_req_verify_1590_clean_instruction() -> None:
    """REQ-VERIFY-1590: clean instruction strips whitespace."""
    assert exp._clean_instruction("  json  ") == "json"
    with pytest.raises(TypeError, match="must be string"):
        exp._clean_instruction(123)  # type: ignore


def test_req_verify_1590_smoke_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1590: artifact is generated and persisted."""
    output_path = tmp_path / "experiment_1590_csr_mask.json"
    
    class DummyTimer:
        def __init__(self) -> None:
            self.val = 0
            self.step = 1000
        def __call__(self) -> int:
            self.val += self.step
            return self.val

    persisted = exp.run_experiment(
        output_path=output_path,
        repeats=2,
        tests_run=["test_req_verify_1590_smoke_artifact"],
        timer=DummyTimer(),
    )
    
    assert persisted["experiment"] == "1590_csr_mask"
    assert persisted["schema"] == "dsl_csr_mask_v1"
    assert persisted["status"] == "complete"
    assert "csr_automaton_path" in persisted
    assert persisted["csr_state_count"] > 0
    assert persisted["csr_transition_count"] > 0
    assert "csr_latency_ms_p50" in persisted
    assert "existing_path_latency_ms_p50" in persisted
    assert persisted["tests_run"] == ["test_req_verify_1590_smoke_artifact"]
    
    # verify written to disk
    data = json.loads(output_path.read_text("utf-8"))
    assert data["experiment"] == "1590_csr_mask"
