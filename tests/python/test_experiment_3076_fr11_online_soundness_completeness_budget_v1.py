"""Tests for Exp 3076 FR-11 online soundness/completeness budget.

Spec refs: REQ-LEARN-3076, SCENARIO-LEARN-3076,
SCENARIO-LEARN-3076-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import fr11_online_soundness_completeness_budget_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "self-learning" / "spec.md"
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "experiment_3076_fr11_online_soundness_completeness_budget_v1.py"
)
SOURCE_FILES = (
    exp.EXP3046_ARTIFACT_REL_PATH,
    exp.EXP3060_ARTIFACT_REL_PATH,
    exp.EXP3061_ARTIFACT_REL_PATH,
)


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        exp3046_artifact_path=tmp_path / exp.EXP3046_ARTIFACT_REL_PATH,
        exp3060_artifact_path=tmp_path / exp.EXP3060_ARTIFACT_REL_PATH,
        exp3061_artifact_path=tmp_path / exp.EXP3061_ARTIFACT_REL_PATH,
        started_at=200.0,
        clock=lambda: 202.25,
        tests_run=("focused-req-3076",),
    )


def _copy_sources(tmp_path: Path) -> None:
    for rel_path in SOURCE_FILES:
        source = REPO_ROOT / rel_path
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_req_learn_3076_spec_and_script_anchor_exists() -> None:
    """REQ-LEARN-3076: protocol artifact is OpenSpec anchored and runnable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3076" in spec
    assert "SCENARIO-LEARN-3076" in spec
    assert "SCENARIO-LEARN-3076-BLOCKED" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "soundness_completeness_budget_ready" in spec
    assert "max_soundness_mistakes" in spec
    assert "native KAN/EBT integration" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_learn_3076_writes_exp3077_consumable_protocol(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3076: ready sources produce a budget Exp 3077 can consume."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["soundness_completeness_budget_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["honest_verdict"] == "complete_fr11_soundness_completeness_budget_ready"
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["tests_run"] == ["focused-req-3076"]

    soundness = artifact["soundness_mistake_definition"]
    assert soundness["mistake_type"] == "soundness"
    assert soundness["decision_under_audit"] == "controller_accept"
    assert soundness["unsafe_condition"] == "independent_exact_authority_rejects_candidate"
    assert soundness["count_key"] == "soundness_mistakes"
    assert soundness["rate_key"] == "soundness_mistake_rate"
    assert soundness["severity"] == "promotion_blocking"
    assert soundness["rollback_trigger"]["trigger_key"] == "soundness_mistakes"
    assert soundness["independent_exact_authority_required"] is True

    completeness = artifact["completeness_mistake_definition"]
    assert completeness["mistake_type"] == "completeness"
    assert completeness["decision_under_audit"] == "controller_reject_or_abstain"
    assert completeness["unsafe_condition"] == "independent_exact_authority_accepts_candidate"
    assert completeness["count_key"] == "completeness_mistakes"
    assert completeness["rate_key"] == "completeness_mistake_rate"
    assert completeness["rollback_trigger"]["trigger_key"] == "completeness_mistakes"

    delayed = artifact["delayed_regression_window"]
    assert delayed["evaluation_required"] is True
    assert delayed["metric_name"] == "delayed_regression_mistakes"
    assert delayed["min_delay_cycles"] >= 1
    assert delayed["max_allowed_mistakes"] == 0
    assert delayed["rollback_trigger"]["trigger_key"] == "delayed_regressions"

    contradiction = artifact["contradiction_mistake_definition"]
    assert contradiction["mistake_type"] == "contradiction"
    assert contradiction["count_key"] == "contradiction_mistakes"
    assert contradiction["rollback_trigger"]["trigger_key"] == "contradiction_mistakes"

    rollback_triggers = artifact["rollback_triggers"]
    assert {row["name"] for row in rollback_triggers} == exp.REQUIRED_ROLLBACK_TRIGGER_NAMES
    assert all(row["action"] == "rollback_candidate_update" for row in rollback_triggers)

    budget = artifact["mistake_budget"]
    assert budget["pilot_id"] == "exp3077_tiny_online_controller_pilot"
    assert budget["max_soundness_mistakes"] == 0
    assert budget["max_completeness_mistakes"] == 0
    assert budget["max_delayed_regressions"] == 0
    assert budget["no_feedback_max_delta"] == 0.0
    assert budget["shuffled_feedback_max_delta"] == 0.0
    assert budget["prior_retention_floor"] == 1.0
    assert all(isinstance(budget[key], int | float) for key in exp.NUMERIC_BUDGET_KEYS)

    controls = {row["name"]: row for row in artifact["required_controls"]}
    assert set(controls) == exp.REQUIRED_CONTROL_NAMES
    assert controls["no_feedback_control"]["promotion_rule"] == "learning_delta_must_exceed_control"
    assert controls["shuffled_feedback_control"]["promotion_rule"] == (
        "learning_delta_must_exceed_control"
    )
    assert controls["prior_retention_floor"]["promotion_rule"] == "retention_score_must_meet_floor"

    claims = artifact["forbidden_claims"]
    assert any("model-weight self-learning" in claim for claim in claims)
    assert any("autonomous production self-modification" in claim for claim in claims)
    assert any("native KAN integration" in claim for claim in claims)
    assert any("native EBT integration" in claim for claim in claims)

    sources = {row["source_experiment_id"]: row for row in artifact["source_artifacts"]}
    assert set(sources) == {"exp3046", "exp3060", "exp3061"}
    assert all(row["ready"] is True for row in sources.values())
    assert all(row["claim_classification"]["controller_only"] is True for row in sources.values())
    assert sources["exp3046"]["claim_classification"]["flagged_adversarial"] is True
    assert sources["exp3061"]["claim_classification"]["flagged_adversarial"] is True
    assert (
        "model_weight_self_learning" in sources["exp3060"]["claim_classification"]["out_of_scope"]
    )

    summary = artifact["prior_artifact_claims_summary"]
    assert summary["all_sources_controller_only"] is True
    assert summary["flagged_source_experiments"] == ["exp3046", "exp3061"]
    assert "model_weight_self_learning" in summary["out_of_scope_claims"]
    assert "Exp 3076 defines accounting only" in summary["interpretation"]

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "cached_artifact_protocol_definition"
    assert substrate["protocol_work_only"] is True
    assert substrate["live_llm_inference"] is False
    assert substrate["live_model_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False

    assert exp.protocol_is_exp3077_consumable(
        artifact["soundness_mistake_definition"],
        artifact["completeness_mistake_definition"],
        artifact["delayed_regression_window"],
        artifact["mistake_budget"],
        artifact["required_controls"],
        artifact["forbidden_claims"],
        artifact["source_artifacts"],
        artifact["inference_substrate"],
    )
    exp.validate_artifact(artifact)


def test_req_learn_3076_components_are_machine_readable(tmp_path: Path) -> None:
    """REQ-LEARN-3076-2/3/4/5/6: accounting components are auditable."""

    _copy_sources(tmp_path)
    config = _config(tmp_path)
    sources = exp.load_source_bundle(config)
    source_rows = exp.source_artifacts(sources, config)

    assert exp.precondition_blocker(sources) is None
    assert exp.soundness_mistake_definition()["rollback_trigger"]["comparison"] == ">"
    assert exp.completeness_mistake_definition()["rollback_trigger"]["threshold_key"] == (
        "max_completeness_mistakes"
    )
    assert exp.delayed_regression_window()["source_metric"] == "exp3061.delayed_regression_delta"
    assert exp.contradiction_mistake_definition()["unsafe_condition"] == (
        "candidate_update_increases_exact_contradiction_rate"
    )
    assert exp.mistake_budget()["prior_retention_floor"] == pytest.approx(1.0)
    assert {row["name"] for row in exp.required_controls()} == exp.REQUIRED_CONTROL_NAMES
    assert exp.prior_artifact_claims_summary(source_rows)["flagged_source_experiments"] == [
        "exp3046",
        "exp3061",
    ]
    assert exp.protocol_is_exp3077_consumable(
        exp.soundness_mistake_definition(),
        exp.completeness_mistake_definition(),
        exp.delayed_regression_window(),
        exp.mistake_budget(),
        exp.required_controls(),
        exp.forbidden_claims(),
        source_rows,
        exp.inference_substrate(),
    )
    assert exp._relative_to(tmp_path, tmp_path / "results" / "x.json") == Path("results/x.json")
    assert exp._relative_to(tmp_path, Path("/outside/root.json")) == Path("/outside/root.json")
    assert exp._round(1.2345678) == pytest.approx(1.234568)


def test_scenario_learn_3076_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3076-BLOCKED: missing sources fail closed."""

    artifact = exp.run_experiment(_config(tmp_path))

    assert artifact["soundness_completeness_budget_ready"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["blocked_reason"] == "exp3046_artifact_missing_or_empty"
    assert artifact["source_artifacts"][0]["source_experiment_id"] == "exp3046"
    assert artifact["source_artifacts"][0]["ready"] is False
    assert artifact["prior_artifact_claims_summary"]["all_sources_controller_only"] is False
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["inference_substrate"]["model_weight_mutation"] is False
    assert (tmp_path / "results" / exp.ARTIFACT_FILENAME).is_file()
    exp.validate_artifact(artifact)


def test_req_learn_3076_precondition_blockers_are_explicit(tmp_path: Path) -> None:
    """REQ-LEARN-3076-1: source blockers name unsafe prior evidence."""

    _copy_sources(tmp_path)
    sources = exp.load_source_bundle(_config(tmp_path))
    exp3046_ready = dict(sources.exp3046_artifact)
    exp3060_ready = dict(sources.exp3060_artifact)
    exp3061_ready = dict(sources.exp3061_artifact)

    assert exp.precondition_blocker(sources) is None

    cases = [
        ({}, exp3060_ready, exp3061_ready, "exp3046_artifact_missing_or_empty"),
        ({"_malformed": True}, exp3060_ready, exp3061_ready, "exp3046_artifact_malformed"),
        (
            exp3046_ready | {"honest_verdict": "waiting"},
            exp3060_ready,
            exp3061_ready,
            "exp3046_not_terminal",
        ),
        (
            exp3046_ready | {"fr11_solver_feedback_ready": False},
            exp3060_ready,
            exp3061_ready,
            "exp3046_not_ready",
        ),
        (
            exp3046_ready | {"inference_substrate": "bad"},
            exp3060_ready,
            exp3061_ready,
            "exp3046_inference_substrate_missing",
        ),
        (
            exp3046_ready | {"inference_substrate": {"live_model_inference": True}},
            exp3060_ready,
            exp3061_ready,
            "exp3046_live_model_inference_claimed",
        ),
        (exp3046_ready, {}, exp3061_ready, "exp3060_artifact_missing_or_empty"),
        (exp3046_ready, {"_malformed": True}, exp3061_ready, "exp3060_artifact_malformed"),
        (
            exp3046_ready,
            exp3060_ready | {"honest_verdict": "waiting"},
            exp3061_ready,
            "exp3060_not_terminal",
        ),
        (
            exp3046_ready,
            exp3060_ready | {"solver_self_model_trace_ready": False},
            exp3061_ready,
            "exp3060_not_ready",
        ),
        (
            exp3046_ready,
            exp3060_ready | {"inference_substrate": {"model_weight_training": True}},
            exp3061_ready,
            "exp3060_model_weight_training_claimed",
        ),
        (exp3046_ready, exp3060_ready, {}, "exp3061_artifact_missing_or_empty"),
        (exp3046_ready, exp3060_ready, {"_malformed": True}, "exp3061_artifact_malformed"),
        (
            exp3046_ready,
            exp3060_ready,
            exp3061_ready | {"honest_verdict": "waiting"},
            "exp3061_not_terminal",
        ),
        (
            exp3046_ready,
            exp3060_ready,
            exp3061_ready | {"fr11_delayed_regression_ready": False},
            "exp3061_not_ready",
        ),
        (
            exp3046_ready,
            exp3060_ready,
            exp3061_ready | {"inference_substrate": {"model_weight_mutation": True}},
            "exp3061_model_weight_mutation_claimed",
        ),
    ]
    for exp3046_artifact, exp3060_artifact, exp3061_artifact, expected in cases:
        assert (
            exp.precondition_blocker(
                exp.SourceBundle(exp3046_artifact, exp3060_artifact, exp3061_artifact)
            )
            == expected
        )

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._read_json(malformed) == {"_malformed": True}


def test_req_learn_3076_validation_rejects_invalid_ready_artifacts(tmp_path: Path) -> None:
    """REQ-LEARN-3076-7: readiness rejects missing gates and unsafe claims."""

    _copy_sources(tmp_path)
    artifact = exp.run_experiment(_config(tmp_path))
    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")

    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"continuous_self_learning_task": False}, "continuous_self_learning_task"),
        (artifact | {"honest_verdict": "waiting"}, "honest_verdict"),
        (artifact | {"soundness_mistake_definition": {}}, "soundness_mistake_definition"),
        (artifact | {"completeness_mistake_definition": {}}, "completeness_mistake_definition"),
        (artifact | {"delayed_regression_window": {}}, "delayed_regression_window"),
        (artifact | {"mistake_budget": {"max_soundness_mistakes": 0}}, "mistake_budget"),
        (artifact | {"required_controls": []}, "required_controls"),
        (artifact | {"forbidden_claims": ["live LLM inference"]}, "forbidden_claims"),
        (artifact | {"source_artifacts": []}, "source_artifacts"),
        (artifact | {"source_artifacts": "bad"}, "source_artifacts"),
        (artifact | {"inference_substrate": "bad"}, "inference_substrate"),
        (
            artifact
            | {
                "inference_substrate": {
                    "live_llm_inference": True,
                    "model_weight_training": False,
                    "model_weight_mutation": False,
                }
            },
            "inference_substrate",
        ),
        (
            artifact
            | {
                "inference_substrate": {
                    "live_llm_inference": False,
                    "model_weight_training": False,
                    "model_weight_mutation": False,
                }
            },
            "soundness_completeness_budget_ready",
        ),
        (
            artifact
            | {
                "soundness_completeness_budget_ready": False,
                "honest_verdict": "complete_bad",
            },
            "blocked artifacts",
        ),
    ]
    for invalid, expected in invalid_cases:
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(invalid)
