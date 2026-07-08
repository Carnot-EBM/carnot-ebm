"""Tests for Exp5399 bounded KAN/KANDy dynamic counterexample certificate.

Spec refs: REQ-KAN-5399, SCENARIO-KAN-5399.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5399_kan_dynamic_counterexample_certificate_v491 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_kan_5399_spec_declares_dynamic_certificate_contract() -> None:
    """REQ-KAN-5399: OpenSpec anchors the bounded dynamics certificate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5399") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-KAN-5399",
        "SCENARIO-KAN-5399",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.EXP5395_RESULT_RELATIVE_PATH),
        "KAN/KANDy-style lifted-feature model",
        "held-out true properties",
        "held-out false-property perturbations",
        "`broad_kan_verification_claim` MUST be false",
        "`scripts/research_conductor.py`",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_kan_5399_selects_exp5395_trace_family_when_available() -> None:
    """REQ-KAN-5399: Exp5395 is the preferred bounded routing trace source."""

    source, samples = exp.select_trace_family(root=REPO)

    assert source == {
        "source_type": "exp5395",
        "path": str(exp.EXP5395_RESULT_RELATIVE_PATH),
        "fallback_reason": None,
    }
    assert len(samples) == 36
    assert samples[0].event_id == "t01-e1-retrieve"
    assert samples[0].previous_tier_rank == 0
    assert samples[0].risk_velocity == pytest.approx(0.0)
    assert {row.memory_variant for row in samples} >= {"clean", "stale", "poisoned"}
    assert {row.selected_verifier_tier for row in samples} == {
        "cheap_deterministic",
        "local_sota",
        "rich_deterministic",
    }


def test_req_kan_5399_synthetic_fallback_records_reason(tmp_path: Path) -> None:
    """REQ-KAN-5399: missing Exp5395 data uses an explicit synthetic fallback."""

    source, samples = exp.select_trace_family(root=tmp_path)
    model = exp.fit_lifted_dynamics_model(samples[: exp.TRAIN_SAMPLE_COUNT])

    assert source["source_type"] == "synthetic"
    assert source["path"] == str(exp.EXP5395_RESULT_RELATIVE_PATH)
    assert source["fallback_reason"] == "Exp5395 artifact unavailable"
    assert len(samples) == exp.SYNTHETIC_SAMPLE_COUNT
    assert model.lifted_feature_names == exp.LIFTED_FEATURE_NAMES


def test_scenario_kan_5399_lifted_model_matches_heldout_routing_dynamics() -> None:
    """SCENARIO-KAN-5399: lifted dynamics preserve held-out true routing properties."""

    _source, samples = exp.select_trace_family(root=REPO)
    train, heldout = exp.split_train_heldout(samples)
    model = exp.fit_lifted_dynamics_model(train)

    assert len(train) == exp.TRAIN_SAMPLE_COUNT
    assert len(heldout) == len(samples) - exp.TRAIN_SAMPLE_COUNT
    assert model.lifted_feature_names == exp.LIFTED_FEATURE_NAMES
    assert len(model.lifted_feature_names) == 10
    assert model.candidate_dynamics["cheap_low_risk_cell"]["tier"] == "cheap_deterministic"
    assert model.candidate_dynamics["severe_memory_onset_cell"]["tier"] == "local_sota"
    assert model.candidate_dynamics["constraint_or_novelty_cell"]["tier"] == "rich_deterministic"

    heldout_predictions = [model.predict_tier(row) for row in heldout]
    assert heldout_predictions == [row.selected_verifier_tier for row in heldout]

    lifted = model.lift(heldout[-1])
    assert list(lifted) == list(exp.LIFTED_FEATURE_NAMES)
    assert lifted["risk_velocity"] == pytest.approx(heldout[-1].risk_velocity)
    assert lifted["previous_tier_rank"] == heldout[-1].previous_tier_rank


def test_scenario_kan_5399_rejects_false_properties_with_certificate_regions() -> None:
    """SCENARIO-KAN-5399: false routing-dynamics properties produce cells."""

    diagnostic = exp.evaluate_dynamic_certificate(root=REPO)

    assert diagnostic["sample_count"] == 36
    assert diagnostic["lifted_feature_count"] == len(exp.LIFTED_FEATURE_NAMES)
    assert diagnostic["true_property_count"] == 3
    assert diagnostic["false_property_count"] == 4
    assert diagnostic["false_property_rejection_rate"] == 1.0
    assert diagnostic["true_property_preservation_rate"] == 1.0
    assert diagnostic["counterexample_region_count"] == 4
    assert diagnostic["broad_kan_verification_claim"] is False
    assert diagnostic["dynamic_counterexample_certificate_ready"] is True
    assert diagnostic["trace_source"]["source_type"] == "exp5395"

    for check in diagnostic["true_property_checks"]:
        assert check["preserved"] is True
        assert check["heldout_only"] is True

    for check in diagnostic["false_property_checks"]:
        assert check["rejected"] is True
        assert check["heldout_only"] is True
        assert check["model_tier"] != check["false_claimed_tier"]
        assert check["counterexample_cell_id"].startswith("dyn_cell_")

    for region in diagnostic["counterexample_regions"]:
        assert region["bounded_fixture_only"] is True
        assert region["feature_bounds"]
        assert region["rejects_false_property"] is True
        assert region["model_tier"] in exp.VERIFIER_TIER_RANKS

    assert any("bounded Exp5395" in limit for limit in diagnostic["claim_limits"])
    assert any("no broad KAN verification claim" in limit for limit in diagnostic["claim_limits"])


def test_req_kan_5399_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-KAN-5399: run() writes the bounded certificate artifact."""

    tests_run = [
        {
            "command": (
                ".venv/bin/pytest "
                "tests/python/test_experiment_5399_kan_dynamic_counterexample_certificate_v491.py "
                "-q"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5399_kan_dynamic_counterexample_certificate_v491.py "
                "-m pytest "
                "tests/python/test_experiment_5399_kan_dynamic_counterexample_certificate_v491.py "
                "-q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage report "
                "--include=python/carnot/experiment_5399_kan_dynamic_counterexample_certificate_v491.py "
                "--fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_FIELDS) <= set(artifact)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["trace_source"]["source_type"] == "exp5395"
    assert artifact["sample_count"] == 36
    assert artifact["lifted_feature_count"] == len(exp.LIFTED_FEATURE_NAMES)
    assert artifact["true_property_count"] == 3
    assert artifact["false_property_count"] == 4
    assert artifact["false_property_rejection_rate"] == 1.0
    assert artifact["true_property_preservation_rate"] == 1.0
    assert artifact["counterexample_region_count"] == 4
    assert artifact["broad_kan_verification_claim"] is False
    assert artifact["dynamic_counterexample_certificate_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == tests_run
    assert artifact["spec_refs"] == list(exp.SPEC_REFS)
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    exp.validate_artifact(artifact)


def test_req_kan_5399_repository_artifact_matches_replay() -> None:
    """REQ-KAN-5399: committed result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["dynamic_counterexample_certificate_ready"] is True
    assert result["broad_kan_verification_claim"] is False


def test_req_kan_5399_validation_rejects_claim_drift() -> None:
    """REQ-KAN-5399: validation fails closed on schema and claim drift."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5399", "outcome": "passed"}],
    )

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(missing)

    bad_principle = deepcopy(artifact)
    bad_principle["field_principles"]["status"] = "changed"
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_principle)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.490"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_source = deepcopy(artifact)
    bad_source["trace_source"] = {"source_type": "synthetic", "path": "x", "fallback_reason": None}
    with pytest.raises(ValueError, match="trace_source"):
        exp.validate_artifact(bad_source)

    bad_bool = deepcopy(artifact)
    bad_bool["broad_kan_verification_claim"] = True
    with pytest.raises(ValueError, match="broad_kan_verification_claim"):
        exp.validate_artifact(bad_bool)

    bad_ready = deepcopy(artifact)
    bad_ready["dynamic_counterexample_certificate_ready"] = False
    with pytest.raises(ValueError, match="ready"):
        exp.validate_artifact(bad_ready)

    bad_false_rate = deepcopy(artifact)
    bad_false_rate["false_property_rejection_rate"] = 0.75
    with pytest.raises(ValueError, match="false_property_rejection_rate"):
        exp.validate_artifact(bad_false_rate)

    bad_true_rate = deepcopy(artifact)
    bad_true_rate["true_property_preservation_rate"] = 0.5
    with pytest.raises(ValueError, match="true_property_preservation_rate"):
        exp.validate_artifact(bad_true_rate)

    bad_regions = deepcopy(artifact)
    bad_regions["counterexample_regions"] = []
    with pytest.raises(ValueError, match="counterexample"):
        exp.validate_artifact(bad_regions)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    assert exp._json_ready(Path("results/example.json")) == "results/example.json"
    assert exp._json_ready(("REQ-KAN-5399",)) == ["REQ-KAN-5399"]
