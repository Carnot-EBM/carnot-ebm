"""Tests for CPU-based HILED simulator.

Spec: REQ-SAMPLE-1869
"""

import json
from unittest.mock import patch

from carnot.inference.hiled_simulator import HiledSimulator
from carnot.inference.llm_solver import logprob_rejection_sample, LLMSolverConfig


def test_hiled_simulator_logic():
    simulator = HiledSimulator(penalty=5.0, constraints=["bad", "unsafe"])

    # Should not penalize
    score1 = simulator.score_candidate("this is a good response", -1.0)
    assert score1 == -1.0

    # Should penalize
    score2 = simulator.score_candidate("this is an unsafe response", -1.0)
    assert score2 == -6.0


def test_hiled_simulator_integration_and_experiment(tmp_path):
    config = LLMSolverConfig(model="mock-model")
    simulator = HiledSimulator(penalty=10.0, constraints=["hallucinate"])

    # We will patch _generate_with_logprobs
    responses = [
        ("this response will hallucinate heavily", -2.0),
        ("this response is safe and correct", -3.0),
    ]

    with patch("carnot.inference.llm_solver._generate_with_logprobs") as mock_gen:
        mock_gen.side_effect = responses

        result = logprob_rejection_sample(
            config=config,
            prompt="test",
            n_candidates=2,
            hiled_simulator=simulator,
            model="mock",
            tokenizer="mock",
        )

    # Without HILED, the first candidate (-2.0) would win because it's higher than -3.0.
    # With HILED, the first is penalized to -12.0, so the second (-3.0) wins.
    assert result.best_response == "this response is safe and correct"

    # Serialise the same artifact shape the experiment emits, but into pytest's
    # tmp_path -- NEVER into the repo's results/ directory.
    #
    # WHY THIS MUST NOT WRITE INTO results/: `efficiency_gains_ms` below is
    # `simulator.latency_ms`, a LIVE wall-clock measurement that differs on every
    # run. Writing it to results/experiment_1869_hiled.json meant that merely
    # RUNNING THE TEST SUITE silently rewrote a historical experiment artifact
    # with a fresh number -- the research record mutating as a side effect of
    # `pytest`, with no experiment having been re-run. That is exactly the
    # "never rewrite a historical artifact" rule being violated by accident: on
    # 2026-07-28 this halved the published value (4.2827ms -> 2.1133ms) and the
    # rewrite was staged for commit unnoticed. The committed artifact is the
    # record of the ORIGINAL run and stays frozen; this test verifies the
    # artifact SHAPE, which is what it can honestly check.
    output_path = tmp_path / "experiment_1869_hiled.json"

    results = {
        "efficiency_gains_ms": simulator.latency_ms,
        "constraint_enforcement_rate": 1.0,
        "hiled_enabled": True,
        "simulated_steps": simulator.simulated_steps,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Assert the artifact shape, so this write is a checked behaviour rather
    # than an unverified side effect (a test that only writes asserts nothing).
    written = json.loads(output_path.read_text())
    assert set(written) == {
        "efficiency_gains_ms",
        "constraint_enforcement_rate",
        "hiled_enabled",
        "simulated_steps",
    }
    assert written["hiled_enabled"] is True
    assert written["constraint_enforcement_rate"] == 1.0
    assert written["efficiency_gains_ms"] >= 0.0
    assert written["simulated_steps"] >= 0
