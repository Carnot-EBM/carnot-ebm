"""Tests for Exp 3060 solver self-model trace schema.

Spec refs: REQ-LEARN-3060, SCENARIO-LEARN-3060,
SCENARIO-LEARN-3060-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_solver_self_model_trace_schema_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3060_fr11_solver_self_model_trace_schema_v1.py"
SOURCE_FILES = (
    exp.EXP3045_ARTIFACT_REL_PATH,
    exp.EXP3046_ARTIFACT_REL_PATH,
    exp.EXP3047_ARTIFACT_REL_PATH,
)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        exp3045_artifact_path=tmp_path / exp.EXP3045_ARTIFACT_REL_PATH,
        exp3046_artifact_path=tmp_path / exp.EXP3046_ARTIFACT_REL_PATH,
        exp3047_artifact_path=tmp_path / exp.EXP3047_ARTIFACT_REL_PATH,
        started_at=50.0,
        clock=lambda: 51.5,
        tests_run=("focused-req-3060",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for rel_path in SOURCE_FILES:
        source = REPO_ROOT / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_learn_3060_spec_and_script_anchor_exists() -> None:
    """REQ-LEARN-3060: trace schema is OpenSpec anchored and runnable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3060" in spec
    assert "SCENARIO-LEARN-3060" in spec
    assert "SCENARIO-LEARN-3060-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "solver_self_model_trace_ready" in spec
    assert "solver_prompt_input" in spec
    assert "missing_delayed_regression_evaluation" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_learn_3060_writes_directly_consumable_schema(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3060: ready sources produce an Exp 3061 trace schema."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["solver_self_model_trace_ready"] is True
    assert artifact["honest_verdict"] == "complete_solver_self_model_trace_schema_ready"
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["tests_run"] == ["focused-req-3060"]

    trace_schema = artifact["trace_schema"]
    assert trace_schema["schema_id"] == exp.TRACE_SCHEMA_ID
    assert trace_schema["schema_version"] == "1.0"
    assert trace_schema["exp3061_consumable"] is True
    assert trace_schema["for_process_level_solver_feedback"] is True
    assert {field["name"] for field in trace_schema["fields"]} == exp.REQUIRED_TRACE_FIELDS
    assert trace_schema["fields"][0]["name"] == "trace_id"
    assert trace_schema["fields"][1]["name"] == "solver_prompt_input"
    assert trace_schema["fields"][1]["required"] is True
    assert trace_schema["fields"][2]["name"] == "exact_constraint_family"

    edit_names = {row["name"] for row in artifact["allowed_edit_targets"]}
    assert edit_names == exp.ALLOWED_EDIT_TARGET_NAMES
    assert "model_weights" not in edit_names
    assert all(row["scope"] == "controller_side_only" for row in artifact["allowed_edit_targets"])

    assert any("model-weight learning" in claim for claim in artifact["forbidden_claims"])
    assert any("live LLM inference" in claim for claim in artifact["forbidden_claims"])
    assert {rule["name"] for rule in artifact["validation_rules"]} == exp.REQUIRED_RULE_NAMES
    assert all(rule["automatic"] is True for rule in artifact["validation_rules"])

    window = artifact["delayed_regression_window"]
    assert window["evaluation_required"] is True
    assert window["metric_name"] == "delayed_regression_delta"
    assert window["failure_threshold"] == "delta < 0.0"
    assert window["min_delay_cycles"] >= 1
    assert window["replay_case_source"] == "exp3046.delayed_regression"

    sources = {row["experiment_id"]: row for row in artifact["source_artifacts"]}
    assert set(sources) == {"exp3045", "exp3046", "exp3047"}
    assert all(row["ready"] is True for row in sources.values())
    assert sources["exp3046"]["flagged_adversarial"] is True
    assert sources["exp3046"]["limits"]["model_weight_mutation"] is False
    assert sources["exp3047"]["evidence_type"] == "controller_locality_nonforgetting_probe"

    evidence = artifact["controller_only_evidence_summary"]
    assert evidence["controller_only"] is True
    assert evidence["model_weight_learning_evidence"] is False
    assert evidence["exp3046"]["edit_targets_used"] == ["controller_weights", "trace_memory"]
    assert evidence["exp3047"]["locality_metric"] == pytest.approx(0.75)
    assert "process-level trace schema" in evidence["interpretation"]

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "cached_artifact_schema_definition"
    assert substrate["live_llm_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["live_model_inference"] is False
    assert substrate["schema_work_only"] is True

    exp.validate_artifact(artifact)


def test_req_learn_3060_schema_components_are_constructed_from_sources(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-3060-2/5/6: schema, source summary, and readiness are auditable."""

    _copy_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    summary = exp.controller_only_evidence_summary(sources)
    trace_schema = exp.trace_schema()
    rules = exp.validation_rules()

    assert exp.precondition_blocker(sources) is None
    assert summary["controller_only"] is True
    assert summary["model_weight_learning_evidence"] is False
    assert summary["exp3045"]["governance_ready"] is True
    assert summary["exp3046"]["solver_feedback_ready"] is True
    assert summary["exp3046"]["flagged_adversarial"] is True
    assert summary["exp3047"]["locality_probe_ready"] is True
    assert exp.schema_is_directly_consumable(
        trace_schema,
        exp.allowed_edit_targets(),
        rules,
        exp.delayed_regression_window(),
        exp.source_artifacts(sources, _config(tmp_path)),
        exp.inference_substrate(),
    )
    assert exp._relative_to(tmp_path, tmp_path / "results" / "x.json") == Path("results/x.json")
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")
    assert exp._round(1.2345678) == pytest.approx(1.234568)


def test_scenario_learn_3060_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3060-BLOCKED: missing sources fail closed."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["solver_self_model_trace_ready"] is False
    assert artifact["honest_verdict"] == "blocked_missing_solver_self_model_trace_sources"
    assert artifact["blocked_reason"] == "exp3045_artifact_missing_or_empty"
    assert artifact["trace_schema"]["exp3061_consumable"] is True
    assert artifact["source_artifacts"] == []
    assert artifact["controller_only_evidence_summary"]["controller_only"] is False
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    exp.validate_artifact(artifact)


def test_req_learn_3060_precondition_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-3060-1: source blockers name the failed precondition."""

    _copy_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    exp3045_ready = dict(sources.exp3045_artifact)
    exp3046_ready = dict(sources.exp3046_artifact)
    exp3047_ready = dict(sources.exp3047_artifact)

    assert exp.precondition_blocker(sources) is None

    cases = [
        ({}, exp3046_ready, exp3047_ready, "exp3045_artifact_missing_or_empty"),
        ({"_malformed": True}, exp3046_ready, exp3047_ready, "exp3045_artifact_malformed"),
        (
            exp3045_ready | {"honest_verdict": "waiting"},
            exp3046_ready,
            exp3047_ready,
            "exp3045_not_terminal",
        ),
        (
            exp3045_ready | {"fr11_governance_ready": False},
            exp3046_ready,
            exp3047_ready,
            "exp3045_not_ready",
        ),
        (
            exp3045_ready,
            {},
            exp3047_ready,
            "exp3046_artifact_missing_or_empty",
        ),
        (
            exp3045_ready,
            {"_malformed": True},
            exp3047_ready,
            "exp3046_artifact_malformed",
        ),
        (
            exp3045_ready,
            exp3046_ready | {"honest_verdict": "waiting"},
            exp3047_ready,
            "exp3046_not_terminal",
        ),
        (
            exp3045_ready,
            exp3046_ready | {"fr11_solver_feedback_ready": False},
            exp3047_ready,
            "exp3046_not_ready",
        ),
        (
            exp3045_ready,
            exp3046_ready,
            {},
            "exp3047_artifact_missing_or_empty",
        ),
        (
            exp3045_ready,
            exp3046_ready,
            {"_malformed": True},
            "exp3047_artifact_malformed",
        ),
        (
            exp3045_ready,
            exp3046_ready,
            exp3047_ready | {"honest_verdict": "waiting"},
            "exp3047_not_terminal",
        ),
        (
            exp3045_ready,
            exp3046_ready,
            exp3047_ready | {"kan_locality_probe_ready": False},
            "exp3047_not_ready",
        ),
        (
            exp3045_ready
            | {
                "inference_substrate": {
                    "live_llm_inference": True,
                    "model_weight_training": False,
                    "model_weight_mutation": False,
                }
            },
            exp3046_ready,
            exp3047_ready,
            "exp3045_live_llm_inference_claimed",
        ),
        (
            exp3045_ready,
            exp3046_ready
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": True,
                    "model_weight_mutation": False,
                }
            },
            exp3047_ready,
            "exp3046_model_weight_training_claimed",
        ),
        (
            exp3045_ready,
            exp3046_ready,
            exp3047_ready
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": False,
                    "model_weight_mutation": True,
                }
            },
            "exp3047_model_weight_mutation_claimed",
        ),
        (
            exp3045_ready | {"inference_substrate": "missing"},
            exp3046_ready,
            exp3047_ready,
            "exp3045_inference_substrate_missing",
        ),
    ]
    for exp3045_artifact, exp3046_artifact, exp3047_artifact, expected in cases:
        blocker = exp.precondition_blocker(
            exp.SourceBundle(exp3045_artifact, exp3046_artifact, exp3047_artifact)
        )
        assert blocker == expected


def test_req_learn_3060_validation_rejects_inconsistent_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3060-6: readiness requires schema, rules, sources, and substrate."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete_incomplete"})
    with pytest.raises(ValueError, match="terminal success prefix"):
        exp.validate_artifact(artifact | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked_ prefix"):
        exp.validate_artifact(
            artifact | {"solver_self_model_trace_ready": False, "honest_verdict": "waiting"}
        )
    with pytest.raises(ValueError, match="trace_schema"):
        exp.validate_artifact(artifact | {"trace_schema": "missing"})
    with pytest.raises(ValueError, match="required trace fields"):
        bad_schema = dict(artifact["trace_schema"])
        bad_schema["fields"] = bad_schema["fields"][:-1]
        exp.validate_artifact(artifact | {"trace_schema": bad_schema})
    with pytest.raises(ValueError, match="Exp 3061 consumable"):
        exp.validate_artifact(
            artifact | {"trace_schema": artifact["trace_schema"] | {"exp3061_consumable": False}}
        )
    with pytest.raises(ValueError, match="allowed_edit_targets"):
        exp.validate_artifact(artifact | {"allowed_edit_targets": "controller"})
    with pytest.raises(ValueError, match="model_weights"):
        exp.validate_artifact(
            artifact
            | {
                "allowed_edit_targets": artifact["allowed_edit_targets"]
                + [{"name": "model_weights"}]
            }
        )
    with pytest.raises(ValueError, match="forbidden_claims"):
        exp.validate_artifact(artifact | {"forbidden_claims": []})
    with pytest.raises(ValueError, match="validation_rules"):
        exp.validate_artifact(artifact | {"validation_rules": []})
    with pytest.raises(ValueError, match="delayed_regression_window"):
        exp.validate_artifact(
            artifact
            | {
                "delayed_regression_window": artifact["delayed_regression_window"]
                | {"evaluation_required": False}
            }
        )
    with pytest.raises(ValueError, match="source_artifacts"):
        exp.validate_artifact(
            artifact
            | {
                "source_artifacts": [
                    artifact["source_artifacts"][0] | {"ready": False},
                    *artifact["source_artifacts"][1:],
                ]
            }
        )
    with pytest.raises(ValueError, match="continuous_self_learning_scope"):
        exp.validate_artifact(artifact | {"continuous_self_learning_scope": "model weights ok"})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(artifact | {"inference_substrate": "cached"})
    with pytest.raises(ValueError, match="live model inference"):
        exp.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_model_inference": True}
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

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert exp._read_json(bad_json) == {"_malformed": True}
    not_json = tmp_path / "not_json.txt"
    not_json.write_text("[]", encoding="utf-8")
    assert exp._read_json(not_json) == {"_malformed": True}
