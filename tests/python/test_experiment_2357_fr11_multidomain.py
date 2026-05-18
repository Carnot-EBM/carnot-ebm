"""Tests for Exp 2357 FR-11 multidomain FST retention.

Spec: REQ-LEARN-2357, SCENARIO-LEARN-2357.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.learning.fast_slow import FastSlowTrainer
from carnot.reporting import fr11_multidomain_fst_retention as mod


def test_req_learn_2357_fast_slow_trainer_updates_fast_cache_only() -> None:
    """REQ-LEARN-2357: FST trainer exposes slow/fast weights, update, and predict."""

    corpus = mod.build_multidomain_corpus()
    trainer = FastSlowTrainer()
    before_slow = dict(trainer.slow_weights.constraint_weights)
    case = corpus["arithmetic"]["train"][0]

    observed = trainer.update_fast(
        case.question,
        {
            "verified": True,
            "domain": case.domain,
            "constraints": case.constraints,
        },
    )

    assert observed
    assert trainer.slow_weights.constraint_weights == before_slow
    assert "operation:addition" in trainer.fast_weights.cache
    assert (
        trainer.predict(corpus["arithmetic"]["holdout"][0].question)
        == corpus["arithmetic"]["holdout"][0].answer
    )

    trainer.clear_query_context()
    assert trainer.fast_weights.query_context == {}
    assert "operation:addition" in trainer.fast_weights.cache


def test_scenario_learn_2357_writes_valid_retention_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2357: 3-domain FST retention passes the FR-11 gate."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.run_experiment(output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["fr11_multidomain_passed"] is True
    assert artifact["continuous_self_learning_validated"] is True
    assert artifact["cross_domain_retention_rate"] >= 0.75
    assert artifact["n_domains"] == 3
    assert artifact["random_seed"] == 42
    assert len(artifact["retention_measurements"]) == 3
    assert artifact["slow_weights_mutated"] is False


def test_req_learn_2357_validation_rejects_gate_mismatch(tmp_path: Path) -> None:
    """REQ-LEARN-2357: validation enforces the retention pass boolean."""

    artifact = mod.run_experiment(output_path=tmp_path / mod.OUTPUT_FILE)
    artifact["fr11_multidomain_passed"] = False

    try:
        mod.validate_artifact(artifact)
    except AssertionError as exc:
        assert "fr11_multidomain_passed" in str(exc)
    else:  # pragma: no cover - defensive assertion for clearer failures.
        raise AssertionError("expected validation to reject gate mismatch")
