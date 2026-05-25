"""Tests for Exp 3062 KAN/PWA locality verification audit.

Spec refs: REQ-LEARN-3062, SCENARIO-LEARN-3062,
SCENARIO-LEARN-3062-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_kan_pwa_locality_verification_audit_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3062_kan_pwa_locality_verification_audit_v1.py"
SOURCE_FILES = (exp.EXP3047_ARTIFACT_REL_PATH, exp.EXP3061_ARTIFACT_REL_PATH)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        exp3047_artifact_path=tmp_path / exp.EXP3047_ARTIFACT_REL_PATH,
        exp3061_artifact_path=tmp_path / exp.EXP3061_ARTIFACT_REL_PATH,
        started_at=100.0,
        clock=lambda: 101.25,
        tests_or_checks_run=("focused-req-3062",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for rel_path in SOURCE_FILES:
        source = REPO_ROOT / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_learn_3062_spec_and_script_anchor_exists() -> None:
    """REQ-LEARN-3062: audit artifact is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3062" in spec
    assert "SCENARIO-LEARN-3062" in spec
    assert "SCENARIO-LEARN-3062-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "kan_pwa_verification_ready" in spec
    assert "approximation_error_bound" in spec
    assert "controller_locality_evidence_only" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_learn_3062_writes_controller_only_audit_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3062: exact controller anchor audit stays bounded."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["kan_pwa_verification_ready"] is False
    assert artifact["locality_bound"] == pytest.approx(0.75)
    assert artifact["prior_retention_bound"] == pytest.approx(0.0)
    assert artifact["approximation_error_bound"] == pytest.approx(0.0)
    assert artifact["verification_path"] == "exact_controller_anchor_audit"
    assert artifact["promotion_decision"] == "controller_locality_evidence_only"
    assert artifact["claim_promotion_useful"] is False
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["tests_or_checks_run"] == ["focused-req-3062"]

    locality_object = artifact["smallest_locality_object"]
    assert locality_object["object_type"] == "controller_anchor_delta_set"
    assert locality_object["total_anchor_count"] == 8
    assert locality_object["changed_anchor_count"] == 2
    assert locality_object["anchored_prior_count"] == 2
    assert locality_object["nonlocal_change_bound"] == pytest.approx(0.25)
    assert locality_object["changed_anchor_keys"] == [
        "solver_residual::nonzero",
        "solver_residual::zero",
    ]

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "exact_cpu_controller_anchor_audit"
    assert substrate["live_llm_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["kan_model_weight_training"] is False
    assert substrate["trained_kan_weight_verification"] is False

    assert artifact["source_artifacts"]["exp3047_ready"] is True
    assert artifact["source_artifacts"]["exp3061_ready"] is True
    exp.validate_artifact(artifact)


def test_req_learn_3062_bounds_are_derived_from_source_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3062-2/3/4: bounds and path are source-derived."""

    _copy_sources(tmp_path)
    config = _config(tmp_path)
    sources = exp.load_source_bundle(config)
    locality_object = exp.identify_smallest_locality_object(sources)
    audit = exp.measure_verification_audit(sources)

    assert exp.precondition_blocker(sources) is None
    assert locality_object.object_type == "controller_anchor_delta_set"
    assert locality_object.locality_bound == pytest.approx(0.75)
    assert locality_object.nonlocal_change_bound == pytest.approx(0.25)
    assert audit.locality_bound == pytest.approx(locality_object.locality_bound)
    assert audit.prior_retention_bound == pytest.approx(0.0)
    assert audit.approximation_error_bound == pytest.approx(0.0)
    assert audit.exact_controller_anchor_bound_available is True
    assert audit.kan_pwa_verification_ready is False
    assert audit.verification_path == "exact_controller_anchor_audit"
    assert audit.promotion_decision == "controller_locality_evidence_only"
    assert audit.claim_promotion_useful is False


def test_scenario_learn_3062_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3062-BLOCKED: missing sources fail closed."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["kan_pwa_verification_ready"] is False
    assert artifact["locality_bound"] == 0.0
    assert artifact["prior_retention_bound"] == 0.0
    assert artifact["approximation_error_bound"] == 0.0
    assert artifact["verification_path"] == "blocked_missing_source"
    assert artifact["promotion_decision"] == "blocked"
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["blocked_reason"] == "exp3047_artifact_missing_or_empty"
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    exp.validate_artifact(artifact)


def test_req_learn_3062_precondition_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-3062-1: unsafe source artifacts are rejected."""

    _copy_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    exp3047_ready = dict(sources.exp3047_artifact)
    exp3061_ready = dict(sources.exp3061_artifact)

    assert exp.precondition_blocker(sources) is None

    cases = [
        ({}, exp3061_ready, "exp3047_artifact_missing_or_empty"),
        ({"_malformed": True}, exp3061_ready, "exp3047_artifact_malformed"),
        (exp3047_ready | {"honest_verdict": "waiting"}, exp3061_ready, "exp3047_not_terminal"),
        (
            exp3047_ready | {"kan_locality_probe_ready": False},
            exp3061_ready,
            "exp3047_locality_not_ready",
        ),
        (
            exp3047_ready
            | {"inference_substrate": {"live_llm_inference": True, "model_weight_training": False}},
            exp3061_ready,
            "exp3047_live_llm_inference_claimed",
        ),
        (
            exp3047_ready | {"inference_substrate": "missing"},
            exp3061_ready,
            "exp3047_inference_substrate_missing",
        ),
        (exp3047_ready, {}, "exp3061_artifact_missing_or_empty"),
        (exp3047_ready, {"_malformed": True}, "exp3061_artifact_malformed"),
        (exp3047_ready, exp3061_ready | {"honest_verdict": "waiting"}, "exp3061_not_terminal"),
        (
            exp3047_ready | {"honest_verdict": "success_locality"},
            exp3061_ready | {"fr11_delayed_regression_ready": False},
            "exp3061_delayed_regression_not_ready",
        ),
        (
            exp3047_ready,
            exp3061_ready
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": False,
                    "model_weight_mutation": True,
                }
            },
            "exp3061_model_weight_learning_claimed",
        ),
    ]
    for exp3047_artifact, exp3061_artifact, expected in cases:
        assert (
            exp.precondition_blocker(exp.SourceBundle(exp3047_artifact, exp3061_artifact))
            == expected
        )

    with pytest.raises(ValueError, match="cannot audit locality"):
        exp.measure_verification_audit(exp.SourceBundle({}, exp3061_ready))
    with pytest.raises(ValueError, match="without controller anchors"):
        exp.identify_smallest_locality_object(
            exp.SourceBundle(
                exp3047_ready | {"locality_report": {"total_anchor_count": 0}},
                exp3061_ready,
            )
        )


def test_req_learn_3062_validation_rejects_invalid_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3062-5: validation blocks promotion and substrate drift."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")

    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"honest_verdict": "waiting"}, "honest_verdict"),
        (artifact | {"locality_bound": -0.1}, "locality_bound"),
        (artifact | {"prior_retention_bound": -0.1}, "prior_retention_bound"),
        (artifact | {"approximation_error_bound": -0.1}, "approximation_error_bound"),
        (artifact | {"verification_path": ""}, "verification_path"),
        (artifact | {"promotion_decision": "model_weight_learning_ready"}, "promotion_decision"),
        (artifact | {"claim_promotion_useful": True}, "claim_promotion_useful"),
        (artifact | {"inference_substrate": "bad"}, "inference_substrate"),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_llm_inference": True}
            },
            "live LLM inference",
        ),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_training": True}
            },
            "model weights",
        ),
        (
            artifact
            | {
                "kan_pwa_verification_ready": True,
                "inference_substrate": artifact["inference_substrate"]
                | {"trained_kan_weight_verification": False},
            },
            "trained KAN",
        ),
    ]
    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(bad_artifact)

    malformed_json = tmp_path / "bad.json"
    malformed_json.write_text("{", encoding="utf-8")
    assert exp._read_json(malformed_json) == {"_malformed": True}
    assert exp._sha256_file(tmp_path / "missing.json") == ""
    assert exp._relative_to(tmp_path, tmp_path / "results" / "x.json") == Path("results/x.json")
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")
