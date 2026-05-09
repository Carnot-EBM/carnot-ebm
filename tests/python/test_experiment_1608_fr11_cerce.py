"""Tests for Exp 1608 FR-11 continuous self-learning CerCE scale.

Spec: REQ-LEARN-1608, SCENARIO-LEARN-1608, SCENARIO-LEARN-1609.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.training import fr11_cerce_scale as mod


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _prior_ledger(*, unsafe: bool = False) -> dict[str, object]:
    accepted = bool(unsafe)
    return {
        "status": "blocked" if unsafe else "complete",
        "schema": "cerce_certificate_ledger_v1",
        "cerce_ledger_ready": not unsafe,
        "policy_certificates": [
            {
                "policy_update_id": "prior-policy",
                "promotion_safe": not unsafe,
                "accepted_violation_count": int(accepted),
                "false_accept_delta": int(accepted),
                "soundness_mistakes": int(accepted),
            }
        ],
        "ledger_rows": [
            {
                "accepted_violation": accepted,
                "baseline_violation": False,
                "constraint_id": "prior:case-001",
                "constraint_type": "runtime_contract",
                "false_accept_delta": int(accepted),
                "policy_update_id": "prior-policy",
                "promoted_violation": accepted,
                "soundness_mistake": accepted,
                "source": "prior-fixture",
            }
        ],
    }


def _bounds(*, passed: bool = True) -> dict[str, object]:
    return {
        "status": "complete",
        "schema": "cerce_bounds_check_v1",
        "continuous_self_learning_task": "exp1595-cerce-bounds",
        "bounds_check_passed": passed,
        "simulated_updates_run": 2,
        "rejected_updates": [] if passed else ["unsafe"],
        "honest_verdict": "complete: cerce_bounds_checked",
    }


def _write_sources(
    tmp_path: Path, *, unsafe_prior: bool = False, bounds_passed: bool = True
) -> dict[str, Path]:
    results = tmp_path / "results"
    prior = results / "experiment_1594_cerce_ledger.json"
    bounds = results / "experiment_1595_cerce_bounds.json"
    output = results / mod.OUTPUT_FILE
    _write_json(prior, _prior_ledger(unsafe=unsafe_prior))
    _write_json(bounds, _bounds(passed=bounds_passed))
    return {"prior": prior, "bounds": bounds, "output": output}


def test_req_learn_1608_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1608-1/6: bootstrap artifact exposes the scale-run contract."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(
        output,
        project_root=tmp_path,
        run_date="20260509",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "in_progress"
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["examples_requested"] == 1000
    assert artifact["examples_processed"] == 0
    assert artifact["no_model_weight_mutation"] is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    mod.validate_artifact(artifact)


def test_scenario_learn_1608_runs_1000_examples_and_preserves_cerce(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1608: the scale run improves utility without forgetting."""

    paths = _write_sources(tmp_path)

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=paths["output"],
        prior_ledger_path=paths["prior"],
        bounds_artifact_path=paths["bounds"],
        batch_size=125,
        run_date="20260509",
        tests_run=["tests/python/test_experiment_1608_fr11_cerce.py"],
    )

    assert json.loads(paths["output"].read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["examples_processed"] == 1000
    assert artifact["training_loop_batches"] == 8
    assert artifact["current_update_rows"] == 1000
    assert artifact["past_certificates_checked"] == 1
    assert artifact["past_certificates_violated"] == 0
    assert artifact["utility_delta"] == pytest.approx(0.75)
    assert artifact["cerce_ledger_ready"] is True
    assert artifact["bounds_check_passed"] is True
    assert artifact["accepted_violation_count"] == 0
    assert artifact["false_accept_delta"] == 0
    assert artifact["soundness_mistakes"] == 0
    assert artifact["nonforgetting_certificate_rate"] == 1.0
    assert artifact["blocked_policy_updates"] == []
    assert len(artifact["ledger_rows"]) == 1001
    assert artifact["training_summary"]["improved_examples"] == 750
    assert artifact["tests_run"] == ["tests/python/test_experiment_1608_fr11_cerce.py"]
    mod.validate_artifact(artifact)


def test_scenario_learn_1609_unsafe_new_update_blocks_scale_run(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1609: new accepted violations fail the CerCE gate."""

    paths = _write_sources(tmp_path)
    examples = mod.generate_training_examples(1000, unsafe_indices={7})

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=paths["output"],
        prior_ledger_path=paths["prior"],
        bounds_artifact_path=paths["bounds"],
        examples=examples,
    )

    assert artifact["status"] == "blocked"
    assert artifact["examples_processed"] == 1000
    assert artifact["utility_delta"] > 0
    assert artifact["cerce_ledger_ready"] is False
    assert artifact["accepted_violation_count"] == 1
    assert artifact["false_accept_delta"] == 1
    assert artifact["soundness_mistakes"] == 1
    assert "accepted_constraint_violation" in artifact["blockers"]
    assert "cerce_bounds_worsened" in artifact["blockers"]
    assert "fr11-1608-batch-000" in artifact["blocked_policy_updates"]
    mod.validate_artifact(artifact)


def test_req_learn_1608_prior_or_bounds_failures_block_without_training(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1608-3/5: missing or unsafe prerequisites fail closed."""

    unsafe_paths = _write_sources(tmp_path / "unsafe-prior", unsafe_prior=True)
    unsafe_prior = mod.run_experiment(
        project_root=tmp_path / "unsafe-prior",
        output_path=unsafe_paths["output"],
        prior_ledger_path=unsafe_paths["prior"],
        bounds_artifact_path=unsafe_paths["bounds"],
    )
    failing_bounds_paths = _write_sources(tmp_path / "bad-bounds", bounds_passed=False)
    failing_bounds = mod.run_experiment(
        project_root=tmp_path / "bad-bounds",
        output_path=failing_bounds_paths["output"],
        prior_ledger_path=failing_bounds_paths["prior"],
        bounds_artifact_path=failing_bounds_paths["bounds"],
    )
    missing = mod.run_experiment(
        project_root=tmp_path / "missing",
        output_path=Path("results") / mod.OUTPUT_FILE,
        prior_ledger_path=Path("results/missing-ledger.json"),
        bounds_artifact_path=Path("results/missing-bounds.json"),
    )
    not_ready_prior = dict(_prior_ledger(), cerce_ledger_ready=False)
    ledger = mod.cerce.CerCECertificateLedger(run_date="20260509")

    assert unsafe_prior["status"] == "blocked"
    assert unsafe_prior["past_certificates_violated"] == 1
    assert "past_certificate_violation" in unsafe_prior["blockers"]
    assert failing_bounds["status"] == "blocked"
    assert failing_bounds["examples_processed"] == 1000
    assert "cerce_bounds_check_failed" in failing_bounds["blockers"]
    assert missing["status"] == "blocked"
    assert missing["examples_processed"] == 0
    assert "missing_past_cerce_ledger" in missing["blockers"]
    assert "missing_cerce_bounds_artifact" in missing["blockers"]
    assert mod.replay_past_certificate_rows(ledger, not_ready_prior) == (1, 1)


def test_req_learn_1608_cli_and_validation_edges(tmp_path: Path) -> None:
    """REQ-LEARN-1608-4/5/6: CLI and schema validation stay strict."""

    paths = _write_sources(tmp_path)
    output = tmp_path / "results" / "cli.json"

    assert (
        mod.main(
            [
                "--project-root",
                str(tmp_path),
                "--output",
                str(output),
                "--prior-ledger",
                str(paths["prior"]),
                "--bounds-artifact",
                str(paths["bounds"]),
            ]
        )
        == 0
    )
    complete = json.loads(output.read_text(encoding="utf-8"))
    assert complete["status"] == "complete"

    missing = dict(complete)
    del missing["status"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="unsupported schema"):
        mod.validate_artifact(dict(complete, schema="wrong"))

    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(dict(complete, status="unknown"))

    with pytest.raises(AssertionError, match="continuous_self_learning_task"):
        mod.validate_artifact(dict(complete, continuous_self_learning_task=False))

    invalid_complete = dict(
        complete,
        examples_processed=999,
        utility_delta=0.0,
        cerce_ledger_ready=False,
        bounds_check_passed=False,
        no_model_weight_mutation=False,
        accepted_violation_count=1,
        false_accept_delta=1,
        soundness_mistakes=1,
        nonforgetting_certificate_rate=0.5,
        blockers=["x"],
    )
    with pytest.raises(AssertionError, match="complete artifact is invalid"):
        mod.validate_artifact(invalid_complete)
