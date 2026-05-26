"""Tests for Exp 3128 FR-11 EvoEnv verifiable environment synthesis.

Spec refs: REQ-LEARN-3128, SCENARIO-LEARN-3128,
SCENARIO-LEARN-3128-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_evoenv_verifiable_environment_synthesis_v1 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no fake verifier claims\n", encoding="utf-8")
    spec = root / "openspec/capabilities/self-learning/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3128\nSCENARIO-LEARN-3128\nSCENARIO-LEARN-3128-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(
        root,
        mod.EXP3123_REL_PATH,
        {
            "sota_cache_manifest_v2_ready": True,
            "mandatory_headline_model_ids": list(mod.MANDATED_MODEL_SPECS),
            "present_model_ids": [GEMMA26],
            "missing_model_ids": [
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "unsloth/gemma-4-31B-it-GGUF",
            ],
            "selected_headline_model_ids": [GEMMA26],
            "inference_substrate": {"live_model_calls": 0, "no_live_llm_inference": True},
        },
    )
    _write_json(
        root,
        mod.EXP3116_REL_PATH,
        {
            "fr11_unsolvable_curriculum_ready": True,
            "guarded_decisions": [
                {"fixture_id": "prior-1", "decision_label": "correct"},
                {"fixture_id": "prior-2", "decision_label": "correct"},
                {"fixture_id": "prior-3", "decision_label": "correct"},
            ],
            "hard_family_count": 2,
            "soundness_mistakes": 0,
            "completeness_mistakes": 0,
        },
    )


def test_req_learn_3128_spec_anchor_exists() -> None:
    """REQ-LEARN-3128: OpenSpec declares the EvoEnv admission artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3128" in spec
    assert "SCENARIO-LEARN-3128" in spec
    assert "SCENARIO-LEARN-3128-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "fr11_evoenv_pilot_v1_ready" in spec
    assert "live_model_environment_synthesis" in spec
    assert "no_answer_leakage_pass_rate" in spec
    assert "no_weight_update_claim" in spec


def test_req_learn_3128_environment_schema_samples_solves_and_scores() -> None:
    """REQ-LEARN-3128-1: environments sample, solve exactly, and score cheaply."""

    family = mod.default_environment_families()[0]
    first = family.sample_instances(count=2, seed=3128)
    second = family.sample_instances(count=2, seed=3128)

    assert first == second
    assert first[0].environment_id.startswith("modular_balance")
    reference = first[0].compute_reference()

    assert reference.authority == "exact_enumeration"
    assert reference.solution_count > 0
    assert reference.solver_evaluations > reference.verify_checks
    assert first[0].score_response(reference.canonical_assignment).accepted is True

    bad = dict(reference.canonical_assignment)
    variable = next(iter(bad))
    bad[variable] = max(first[0].domains[variable])
    if bad == reference.canonical_assignment:
        bad[variable] = min(first[0].domains[variable])
    bad_score = first[0].score_response(bad)
    assert bad_score.accepted is False
    assert bad_score.score == 0.0
    assert bad_score.violations


def test_req_learn_3128_admission_gates_reject_leaks_and_bad_difficulty() -> None:
    """REQ-LEARN-3128-2/3: only validated, novel, nonleaky environments admit."""

    summary = mod.evaluate_admission(
        mod.sample_candidate_environments(seed=3128),
        prior_signatures={"legacy-fixture-signature"},
    )
    by_id = {record.environment_id: record for record in summary.records}

    assert summary.candidate_count == 5
    assert summary.admitted_count == 3
    assert summary.soundness_errors == 0
    assert summary.completeness_errors == 0
    assert summary.solve_verify_asymmetry_pass_rate == pytest.approx(0.8)
    assert summary.novelty_pass_rate == pytest.approx(1.0)
    assert summary.no_answer_leakage_pass_rate == pytest.approx(0.8)

    assert by_id["modular_balance-3128-0"].admitted is True
    assert by_id["interval_order-3128-0"].admitted is True
    assert by_id["graph_coloring-3128-0"].admitted is True
    assert by_id["leaky_modular-3128-0"].admitted is False
    assert "answer_leakage" in by_id["leaky_modular-3128-0"].rejection_reasons
    assert by_id["too_easy-3128-0"].admitted is False
    assert "difficulty_calibration" in by_id["too_easy-3128-0"].rejection_reasons
    assert "solve_verify_asymmetry" in by_id["too_easy-3128-0"].rejection_reasons


def test_scenario_learn_3128_writes_complete_solver_only_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3128: bounded admission writes the reusable artifact."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.5,
        tests_run=["REQ-LEARN-3128 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_evoenv_pilot_v1_ready"] is True
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["model_specs"] == list(mod.MANDATED_MODEL_SPECS)
    assert artifact["selected_model_ids"] == []
    assert artifact["live_call_count"] == 0
    assert artifact["live_model_environment_synthesis"] is False
    assert artifact["candidate_environment_count"] == 5
    assert artifact["admitted_environment_count"] == 3
    assert artifact["solve_verify_asymmetry_pass_rate"] == pytest.approx(0.8)
    assert artifact["novelty_pass_rate"] == pytest.approx(1.0)
    assert artifact["no_answer_leakage_pass_rate"] == pytest.approx(0.8)
    assert artifact["retention_delta"] == pytest.approx(0.0)
    assert artifact["soundness_errors"] == 0
    assert artifact["completeness_errors"] == 0
    assert artifact["no_weight_update_claim"] is True
    assert artifact["tests_run"] == ["REQ-LEARN-3128 focused"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")

    admitted = artifact["admitted_environments"]
    assert [row["environment_id"] for row in admitted] == [
        "modular_balance-3128-0",
        "interval_order-3128-0",
        "graph_coloring-3128-0",
    ]
    assert all(row["reference"]["solution_count"] > 0 for row in admitted)
    assert all(row["prompt_leaks_answer"] is False for row in admitted)

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "solver_only_environment_admission"
    assert substrate["present_mandated_model_ids"] == [GEMMA26]
    assert substrate["legacy_small_model_headline_used"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["live_model_environment_synthesis"] is False
    mod.validate_artifact(artifact)


def test_scenario_learn_3128_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3128-BLOCKED: missing sources fail closed."""

    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=1.0,
        now_s=1.25,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_evoenv_pilot_v1_ready"] is False
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["selected_model_ids"] == []
    assert artifact["live_call_count"] == 0
    assert artifact["live_model_environment_synthesis"] is False
    assert artifact["candidate_environment_count"] == 0
    assert artifact["admitted_environment_count"] == 0
    assert artifact["retention_delta"] == 0.0
    assert artifact["soundness_errors"] == 0
    assert artifact["completeness_errors"] == 0
    assert artifact["no_weight_update_claim"] is True
    assert artifact["blocked_reason"] == "exp3123_precondition_manifest_missing_or_empty"
    assert artifact["honest_verdict"].startswith("blocked_precondition_failed")
    mod.validate_artifact(artifact)


def test_req_learn_3128_validation_rejects_overclaims(tmp_path: Path) -> None:
    """REQ-LEARN-3128-4/5/6: validation blocks live and weight-learning overclaims."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    mod.validate_artifact(artifact)

    all_valid = mod.ConstraintEnvironment(
        environment_id="all-valid",
        family_id="unit",
        variables=("u",),
        domains={"u": (0,)},
        constraints=(),
        prompt="Return any u.",
    )
    assert mod.first_invalid_assignment(all_valid) is None
    assert mod.pass_rate(()) == 0.0
    assert (
        mod.precondition_blocker({"sota_cache_manifest_v2_ready": True}, {})
        == "exp3116_retention_guard_missing_or_not_ready"
    )

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="live_call_count"):
        mod.validate_artifact(
            artifact
            | {
                "live_model_environment_synthesis": True,
                "live_call_count": 0,
            }
        )
    with pytest.raises(ValueError, match="no_weight_update_claim"):
        mod.validate_artifact(artifact | {"no_weight_update_claim": False})
    with pytest.raises(ValueError, match="soundness_errors"):
        mod.validate_artifact(artifact | {"soundness_errors": 1})
    with pytest.raises(ValueError, match="completeness_errors"):
        mod.validate_artifact(artifact | {"completeness_errors": 1})
    with pytest.raises(ValueError, match="retention_delta"):
        mod.validate_artifact(artifact | {"retention_delta": -0.1})
    with pytest.raises(ValueError, match="admitted_environment_count"):
        mod.validate_artifact(artifact | {"admitted_environment_count": 0})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: not terminal"})
