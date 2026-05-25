"""Tests for Exp 3047 KAN-style locality/nonforgetting probe.

Spec refs: REQ-LEARN-3047, SCENARIO-LEARN-3047,
SCENARIO-LEARN-3047-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_kan_locality_nonforgetting_probe_v2 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3047_kan_locality_nonforgetting_probe_v2.py"
SOURCE_FILES = (
    exp.EXP3046_ARTIFACT_REL_PATH,
    exp.EXP3044_ARTIFACT_REL_PATH,
    exp.EXP3045_ARTIFACT_REL_PATH,
)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        locality_report_path=tmp_path / exp.LOCALITY_REPORT_REL_PATH,
        exp3046_artifact_path=tmp_path / exp.EXP3046_ARTIFACT_REL_PATH,
        exp3044_artifact_path=tmp_path / exp.EXP3044_ARTIFACT_REL_PATH,
        exp3045_artifact_path=tmp_path / exp.EXP3045_ARTIFACT_REL_PATH,
        started_at=100.0,
        clock=lambda: 102.5,
        tests_run=("focused-req-3047",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for rel_path in SOURCE_FILES:
        source = REPO_ROOT / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_req_learn_3047_spec_and_template_script_anchor_exists() -> None:
    """REQ-LEARN-3047: locality probe is OpenSpec anchored and runnable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3047" in spec
    assert "SCENARIO-LEARN-3047" in spec
    assert "SCENARIO-LEARN-3047-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "kan_locality_probe_ready" in spec
    assert "changed_anchor_count" in spec
    assert "irrelevant_control_delta" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_learn_3047_writes_complete_locality_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3047: governed update changes local anchors and retains priors."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))
    report_rows = _jsonl_rows(tmp_path / exp.LOCALITY_REPORT_REL_PATH)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["kan_locality_probe_ready"] is True
    assert artifact["locality_metric"] == pytest.approx(0.75)
    assert artifact["changed_anchor_count"] == 2
    assert artifact["anchored_prior_count"] == 2
    assert artifact["heldout_delta"] == pytest.approx(0.5)
    assert artifact["prior_retention_delta"] == pytest.approx(0.0)
    assert artifact["irrelevant_control_delta"] == pytest.approx(0.0)
    assert artifact["promotion_decision"] == "controller_locality_evidence_only"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["focused-req-3047"]

    locality = artifact["locality_report"]
    assert locality["total_anchor_count"] == 8
    assert locality["changed_anchor_keys"] == [
        "solver_residual::nonzero",
        "solver_residual::zero",
    ]
    assert locality["anchored_prior_keys"] == [
        "prior_guard::invalid",
        "prior_guard::valid",
    ]
    assert locality["changed_update_trace_count"] == 2

    comparator = artifact["comparator_summary"]
    assert comparator["available"] is True
    assert comparator["comparator_type"] == "shuffled_control_update"
    assert comparator["heldout_delta"] == pytest.approx(-0.5)
    assert comparator["changed_anchor_count"] == 2
    assert comparator["promoted"] is False

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "cached_exp3046_controller_locality_probe"
    assert substrate["live_llm_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["controller_weight_update"] is True
    assert substrate["kan_model_weight_training"] is False

    assert artifact["source_trace_counts"]["exp3046_train_update_case_count"] == 2
    assert artifact["source_trace_counts"]["exp3046_family_holdout_case_count"] == 4
    assert artifact["source_trace_counts"]["exp3046_prior_exact_case_count"] == 2
    assert {row["section"] for row in report_rows} == {
        "locality",
        "retention",
        "comparator",
    }
    exp.validate_artifact(artifact)


def test_req_learn_3047_reuses_exp3046_split_and_feedback_traces(tmp_path: Path) -> None:
    """REQ-LEARN-3047-2/3/4: split reuse and anchor accounting are auditable."""

    _copy_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    probe = exp.measure_locality(sources)

    assert exp.precondition_blocker(sources) is None
    assert probe.split_reused_from_exp3046 is True
    assert probe.locality_metric == pytest.approx(0.75)
    assert probe.changed_anchor_count == 2
    assert probe.anchored_prior_count == 2
    assert probe.heldout_delta > 0.0
    assert probe.prior_retention_delta >= 0.0
    assert probe.irrelevant_control_delta == 0.0
    assert probe.comparator_summary["available"] is True
    assert probe.comparator_summary["heldout_delta"] <= 0.0
    assert not (set(probe.changed_anchor_keys) & set(probe.anchored_prior_keys))

    artifact = exp.complete_artifact(_config(tmp_path), sources, probe, 2.5)
    assert artifact["split_report"]["reused_train_update_ids"] == [
        "train-exp3044-verified",
        "train-exp3044-correction",
    ]
    assert artifact["split_report"]["matches_exp3046_artifact"] is True


def test_scenario_learn_3047_blocked_without_solver_feedback_source(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3047-BLOCKED: missing Exp 3046 evidence fails closed."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["kan_locality_probe_ready"] is False
    assert artifact["locality_metric"] == 0.0
    assert artifact["changed_anchor_count"] == 0
    assert artifact["anchored_prior_count"] == 0
    assert artifact["heldout_delta"] == 0.0
    assert artifact["prior_retention_delta"] == 0.0
    assert artifact["irrelevant_control_delta"] == 0.0
    assert artifact["promotion_decision"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_missing_solver_feedback_locality_source"
    assert artifact["blocked_reason"] == "exp3046_artifact_missing_or_empty"
    assert artifact["comparator_summary"]["available"] is False
    assert artifact["comparator_summary"]["reason"] == "blocked_before_comparator"
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    assert not (tmp_path / exp.LOCALITY_REPORT_REL_PATH).exists()
    exp.validate_artifact(artifact)


def test_req_learn_3047_precondition_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-3047-1: source blockers name the failed precondition."""

    _copy_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    exp3046_ready = dict(sources.exp3046_artifact)
    exp3044_ready = dict(sources.exp3044_artifact)
    exp3045_ready = dict(sources.exp3045_artifact)

    assert exp.precondition_blocker(sources) is None

    cases = [
        ({}, exp3044_ready, exp3045_ready, "exp3046_artifact_missing_or_empty"),
        ({"_malformed": True}, exp3044_ready, exp3045_ready, "exp3046_artifact_malformed"),
        (
            exp3046_ready | {"honest_verdict": "waiting"},
            exp3044_ready,
            exp3045_ready,
            "exp3046_not_terminal",
        ),
        (
            exp3046_ready | {"fr11_solver_feedback_ready": False},
            exp3044_ready,
            exp3045_ready,
            "exp3046_solver_feedback_not_ready",
        ),
        (
            exp3046_ready | {"source_trace_counts": {}},
            exp3044_ready,
            exp3045_ready,
            "exp3046_source_trace_counts_missing",
        ),
        (
            exp3046_ready
            | {
                "inference_substrate": {
                    "live_llm_inference": True,
                    "model_weight_training": False,
                    "model_weight_mutation": False,
                }
            },
            exp3044_ready,
            exp3045_ready,
            "exp3046_live_llm_inference_claimed",
        ),
        (
            exp3046_ready
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": True,
                    "model_weight_mutation": False,
                }
            },
            exp3044_ready,
            exp3045_ready,
            "exp3046_model_weight_training_claimed",
        ),
        (
            exp3046_ready
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": False,
                    "model_weight_mutation": True,
                }
            },
            exp3044_ready,
            exp3045_ready,
            "exp3046_model_weight_mutation_claimed",
        ),
        (
            exp3046_ready,
            {},
            exp3045_ready,
            "exp3044_artifact_missing_or_empty",
        ),
        (
            exp3046_ready,
            {"_malformed": True},
            exp3045_ready,
            "exp3044_artifact_malformed",
        ),
        (
            exp3046_ready,
            exp3044_ready | {"honest_verdict": "waiting"},
            exp3045_ready,
            "exp3044_not_terminal",
        ),
        (
            exp3046_ready,
            exp3044_ready | {"validator_tree_exactness_ready": False},
            exp3045_ready,
            "exp3044_exact_feedback_not_ready",
        ),
        (
            exp3046_ready,
            exp3044_ready | {"correction_sets": []},
            exp3045_ready,
            "exp3044_correction_sets_missing",
        ),
        (
            exp3046_ready,
            exp3044_ready | {"inference_substrate": "missing"},
            exp3045_ready,
            "exp3044_inference_substrate_missing",
        ),
        (
            exp3046_ready,
            exp3044_ready,
            {},
            "exp3045_artifact_missing_or_empty",
        ),
        (
            exp3046_ready,
            exp3044_ready,
            {"_malformed": True},
            "exp3045_artifact_malformed",
        ),
        (
            exp3046_ready,
            exp3044_ready,
            exp3045_ready | {"honest_verdict": "waiting"},
            "exp3045_not_terminal",
        ),
        (
            exp3046_ready,
            exp3044_ready,
            exp3045_ready | {"fr11_governance_ready": False},
            "exp3045_governance_not_ready",
        ),
    ]
    for exp3046_artifact, exp3044_artifact, exp3045_artifact, expected in cases:
        assert (
            exp.precondition_blocker(
                exp.SourceBundle(exp3046_artifact, exp3044_artifact, exp3045_artifact)
            )
            == expected
        )

    with pytest.raises(ValueError, match="cannot measure locality"):
        exp.measure_locality(exp.SourceBundle({}, exp3044_ready, exp3045_ready))


def test_req_learn_3047_validation_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3047-5: readiness requires locality, retention, and substrate gates."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete_incomplete"})
    with pytest.raises(ValueError, match="terminal success prefix"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="locality_metric"):
        exp.validate_artifact(artifact | {"locality_metric": 0.0})
    with pytest.raises(ValueError, match="changed_anchor_count"):
        exp.validate_artifact(artifact | {"changed_anchor_count": 0})
    with pytest.raises(ValueError, match="anchored_prior_count"):
        exp.validate_artifact(artifact | {"anchored_prior_count": 0})
    with pytest.raises(ValueError, match="heldout_delta"):
        exp.validate_artifact(artifact | {"heldout_delta": 0.0})
    with pytest.raises(ValueError, match="prior_retention_delta"):
        exp.validate_artifact(artifact | {"prior_retention_delta": -0.1})
    with pytest.raises(ValueError, match="irrelevant_control_delta"):
        exp.validate_artifact(artifact | {"irrelevant_control_delta": 0.2})
    with pytest.raises(ValueError, match="comparator_summary"):
        exp.validate_artifact(artifact | {"comparator_summary": {}})
    with pytest.raises(ValueError, match="comparator_summary"):
        exp.validate_artifact(
            artifact | {"comparator_summary": artifact["comparator_summary"] | {"promoted": True}}
        )
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(artifact | {"inference_substrate": "cached"})
    with pytest.raises(ValueError, match="live LLM"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_llm_inference": True}
            }
        )
    with pytest.raises(ValueError, match="model weights"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_training": True}
            }
        )
    with pytest.raises(ValueError, match="blocked artifacts cannot be ready"):
        exp.validate_artifact(
            artifact
            | {
                "kan_locality_probe_ready": False,
                "honest_verdict": "complete_not_ready",
            }
        )

    malformed_json = tmp_path / "bad.json"
    malformed_json.write_text("{", encoding="utf-8")
    assert exp._read_json(malformed_json) == {"_malformed": True}
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")
