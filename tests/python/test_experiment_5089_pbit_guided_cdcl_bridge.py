"""Tests for Exp 5089 p-bit guided CDCL bridge.

Spec coverage: REQ-VERIFY-5089, SCENARIO-VERIFY-5089.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify.pbit_cdcl_bridge import (
    ARTIFACT_FILENAME,
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    ExactSatAuthority,
    PBitConsensusSampler,
    build_declared_family,
    run_diagnostic,
    validate_artifact,
    write_artifact,
)


def test_declared_family_labels_are_exact_solver_verified() -> None:
    """REQ-VERIFY-5089: declared CNFs have known satisfiability from Z3."""

    family = build_declared_family()
    authority = ExactSatAuthority()

    assert family
    assert {instance.family for instance in family} == {"planted_consensus_3sat_v1"}
    for instance in family:
        result = authority.solve(instance)
        assert result.status in {"sat", "unsat"}
        assert (result.status == "sat") is instance.expected_satisfiable
        if result.assignment is not None:
            assert authority.verify_assignment(instance, result.assignment)


def test_pbit_sampler_only_proposes_assumptions() -> None:
    """SCENARIO-VERIFY-5089: stochastic samples guide literals, not truth labels."""

    instance = build_declared_family()[0]
    sampler = PBitConsensusSampler(seed=17, n_samples=48, burn_in=12, consensus_threshold=0.55)
    assumptions = sampler.propose_assumptions(instance)
    authority = ExactSatAuthority()

    assert assumptions
    assert all(1 <= abs(literal) <= instance.n_vars for literal in assumptions)
    assert authority.verify_assignment(instance, instance.planted_assignment)
    assert not authority.verify_assignment(
        instance,
        [not value for value in instance.planted_assignment],
    )
    assert not authority.verify_assignment(instance, instance.planted_assignment[:-1])


def test_run_diagnostic_preserves_correctness_and_reports_effort() -> None:
    """REQ-VERIFY-5089: all arms are checked by the exact solver."""

    artifact = run_diagnostic()

    validate_artifact(artifact)
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["inference_substrate"] != "live_llm_inference"
    assert artifact["exact_solver_used"] == "z3"
    assert artifact["correctness_preserved"] is True
    assert artifact["n_instances"] == len(build_declared_family())
    assert artifact["fallback_rate"] >= 0.0
    assert set(artifact["delta_effort_vs_pure"]) == {"random_assumption", "pbit_guided"}

    for row in artifact["per_instance_results"]:
        assert row["known_satisfiable"] == row["exact_status"]
        for arm_name in ("pure_solver", "random_assumption", "pbit_guided"):
            arm = row[arm_name]
            if arm["status"] == "sat":
                assert arm["solution_verified"] is True


def test_write_artifact_schema_and_principle_annotations(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5089: terminal JSON contains the required fields."""

    output_path = tmp_path / "results" / ARTIFACT_FILENAME
    artifact = write_artifact(output_path=output_path)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    validate_artifact(loaded)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(loaded)
    assert loaded["honest_verdict"].startswith(
        (
            "success_pbit_guided_cdcl_effort_reduction_",
            "complete_pbit_guided_cdcl_distribution_sensitive_no_win",
        )
    )
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(loaded["field_principles"])
    assert loaded["field_principles"]["inference_substrate"].endswith("not live_llm_inference.")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "optimistic", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "live_llm_inference"),
        ("inference_substrate", "deterministic_solver_only", "inference_substrate"),
        ("exact_solver_used", "sampler", "exact_solver_used"),
        ("correctness_preserved", "true", "correctness_preserved"),
        ("fallback_rate", 2.0, "fallback_rate"),
    ],
)
def test_validate_artifact_rejects_schema_violations(field: str, value: object, message: str) -> None:
    """REQ-VERIFY-5089: invalid terminal artifacts are rejected."""

    artifact = run_diagnostic()
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        validate_artifact(artifact)


def test_validate_artifact_requires_all_fields_and_principles() -> None:
    """SCENARIO-VERIFY-5089: required field annotations are enforced."""

    artifact = run_diagnostic()
    artifact.pop("duration_s")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_artifact(artifact)

    artifact = run_diagnostic()
    artifact["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(artifact)
