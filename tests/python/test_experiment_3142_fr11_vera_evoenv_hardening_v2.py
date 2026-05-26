"""Tests for Exp 3142 FR-11 VeRA EvoEnv hardening v2.

Spec refs: REQ-LEARN-3142, SCENARIO-LEARN-3142,
SCENARIO-LEARN-3142-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fragment_time_monitor_satisfiable_drift_audit_v1 as monitor
from carnot.eval import fr11_evoenv_verifiable_environment_synthesis_v1 as evo
from carnot.eval import fr11_vera_evoenv_hardening_v2 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"
GEMMA26 = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    target = root / Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp3128_payload() -> dict[str, Any]:
    summary = evo.evaluate_admission(evo.sample_candidate_environments(seed=3128))
    return {
        "artifact": "experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1",
        "fr11_evoenv_pilot_v1_ready": True,
        "admitted_environment_count": summary.admitted_count,
        "admitted_environments": evo.admitted_environment_rows(summary),
        "admission_records": [record.to_dict() for record in summary.records],
        "soundness_errors": 0,
        "completeness_errors": 0,
        "no_weight_update_claim": True,
    }


def _prior_monitor_events() -> list[dict[str, Any]]:
    return [
        {
            "event_type": "candidate_final_answer",
            "fixture_id": "prior-ok",
            "payload": {
                "has_returned_answer": True,
                "final_answer_consistent_with_ledger": True,
            },
        },
        {
            "event_type": "candidate_final_answer",
            "fixture_id": "prior-bad",
            "payload": {
                "has_returned_answer": True,
                "final_answer_consistent_with_ledger": False,
            },
        },
    ]


def _exp3129_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3129_fr11_constraint_memory_retention_drift_audit_v1",
        "fr11_constraint_memory_audit_v1_ready": True,
        "admitted_environment_count": 3,
        "ledger_consistency_rate": 0.5,
        "soundness_errors": 0,
        "completeness_errors": 0,
        "no_weight_update_claim": True,
        "promotion_recommendation": (
            "promote_controller_environment_memory_only_"
            "block_model_weight_learning_until_ledger_consistency_is_1.0"
        ),
    }


def _write_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no fake verifier claims\n", encoding="utf-8")
    (root / "research-program.md").write_text("continuous self-learning\n", encoding="utf-8")
    (root / "_bmad").mkdir(parents=True, exist_ok=True)
    (root / "_bmad/prd.md").write_text("FR-11 Autonomous Self-Learning Loop\n", encoding="utf-8")
    spec = root / "openspec/capabilities/self-learning/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3142\nSCENARIO-LEARN-3142\nSCENARIO-LEARN-3142-BLOCKED\n",
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
            "inference_substrate": {"live_model_calls": 0},
        },
    )
    _write_json(root, mod.EXP3128_REL_PATH, _exp3128_payload())
    _write_json(root, mod.EXP3129_REL_PATH, _exp3129_payload())
    _write_json(
        root,
        mod.EXP3126_REL_PATH,
        {
            "fragment_time_monitor_v1_ready": True,
            "monitor_events": _prior_monitor_events(),
            "ledger_consistency_rate": monitor.replay_monitor_events(_prior_monitor_events())[
                "ledger_consistency_rate"
            ],
        },
    )


def test_req_learn_3142_spec_anchor_exists() -> None:
    """REQ-LEARN-3142: OpenSpec declares the VeRA hardening artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3142" in spec
    assert "SCENARIO-LEARN-3142" in spec
    assert "SCENARIO-LEARN-3142-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "fr11_vera_evoenv_v2_ready" in spec
    assert "live_model_variant_generation" in spec
    assert "equivalent_variant_count" in spec
    assert "hardened_variant_count" in spec
    assert "ledger_consistency_rate" in spec


def test_req_learn_3142_generates_equivalent_and_hardened_variants() -> None:
    """REQ-LEARN-3142-1/2/3: variants are executable, novel, and calibrated."""

    admitted = mod.load_admitted_environments(_exp3128_payload())
    summary = mod.generate_and_validate_variants(admitted)
    sources = {environment.environment_id: environment for environment in admitted}

    assert len(admitted) == 3
    assert summary.equivalent_variant_count == 3
    assert summary.hardened_variant_count == 3
    assert summary.soundness_errors == 0
    assert summary.completeness_errors == 0
    assert summary.solve_verify_asymmetry_pass_rate == pytest.approx(1.0)
    assert summary.no_answer_leakage_pass_rate == pytest.approx(1.0)
    assert summary.novelty_pass_rate == pytest.approx(1.0)
    assert summary.difficulty_pass_rate == pytest.approx(1.0)

    for record in summary.records:
        source = sources[record.source_environment_id]
        source_reference = source.compute_reference()
        assert record.determinism_passed is True
        assert record.exact_replay_passed is True
        assert record.no_answer_leakage_passed is True
        assert record.novelty_passed is True
        assert source_reference.solution_count > 0
        if record.variant_kind == "equivalent":
            assert record.reference.solution_count == source_reference.solution_count
            assert record.solution_density_delta == pytest.approx(0.0)
        else:
            assert record.reference.solution_count < source_reference.solution_count
            assert record.solution_density_delta < 0.0


def test_scenario_learn_3142_writes_complete_solver_only_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3142: solver-only variants replay through exact ledgers."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=13.25,
        tests_run=["REQ-LEARN-3142 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_vera_evoenv_v2_ready"] is True
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["model_specs"] == list(mod.MANDATED_MODEL_SPECS)
    assert artifact["selected_model_ids"] == []
    assert artifact["live_call_count"] == 0
    assert artifact["live_model_variant_generation"] is False
    assert artifact["admitted_environment_count"] == 3
    assert artifact["equivalent_variant_count"] == 3
    assert artifact["hardened_variant_count"] == 3
    assert artifact["solve_verify_asymmetry_pass_rate"] == pytest.approx(1.0)
    assert artifact["no_answer_leakage_pass_rate"] == pytest.approx(1.0)
    assert artifact["ledger_consistency_rate"] == pytest.approx(0.875)
    assert artifact["soundness_errors"] == 0
    assert artifact["completeness_errors"] == 0
    assert artifact["no_weight_update_claim"] is True
    assert artifact["promotion_recommendation"].endswith(
        "block_model_weight_learning_until_ledger_consistency_is_1.0"
    )
    assert artifact["tests_run"] == ["REQ-LEARN-3142 focused"]
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["honest_verdict"].startswith("complete:")

    ledger = artifact["ledger_replay_summary"]
    assert ledger["prior_ledger_consistency_rate"] == pytest.approx(0.5)
    assert ledger["new_variant_ledger_consistency_rate"] == pytest.approx(1.0)
    assert ledger["prior_observed_final_answer_count"] == 2
    assert ledger["new_variant_observed_final_answer_count"] == 6
    assert artifact["inference_substrate"]["mode"] == "solver_only_vera_variant_generation"
    assert artifact["inference_substrate"]["model_weight_training"] is False
    assert artifact["inference_substrate"]["live_model_variant_generation"] is False
    mod.validate_artifact(artifact)


def test_scenario_learn_3142_blocked_without_source_evidence(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3142-BLOCKED: missing sources fail closed."""

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_vera_evoenv_v2_ready"] is False
    assert artifact["continuous_self_learning_targeted"] is True
    assert artifact["selected_model_ids"] == []
    assert artifact["live_call_count"] == 0
    assert artifact["live_model_variant_generation"] is False
    assert artifact["admitted_environment_count"] == 0
    assert artifact["equivalent_variant_count"] == 0
    assert artifact["hardened_variant_count"] == 0
    assert artifact["ledger_consistency_rate"] == 0.0
    assert artifact["soundness_errors"] == 0
    assert artifact["completeness_errors"] == 0
    assert artifact["no_weight_update_claim"] is True
    assert artifact["blocked_reason"] == "exp3123_precondition_manifest_missing_or_empty"
    assert artifact["honest_verdict"].startswith("blocked_precondition_failed")
    assert (
        mod.precondition_blocker({"sota_cache_manifest_v2_ready": True}, {}, {})
        == "exp3128_evoenv_artifact_missing_or_not_ready"
    )
    assert (
        mod.precondition_blocker(
            {"sota_cache_manifest_v2_ready": True},
            {"fr11_evoenv_pilot_v1_ready": True},
            {},
        )
        == "exp3129_constraint_memory_audit_missing_or_not_ready"
    )
    mod.validate_artifact(artifact)


def test_req_learn_3142_validation_rejects_overclaims_and_regressions(tmp_path: Path) -> None:
    """REQ-LEARN-3142-4/5/6: validation blocks leakage and weight overclaims."""

    _write_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=4.0)
    mod.validate_artifact(artifact)

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.rate(1, 0) == 0.0
    assert mod.rate(1, 2) == 0.5
    assert mod.round_float(1 / 3) == pytest.approx(0.333333)
    assert mod.prior_ledger_counts({"ledger_consistency_rate": 0.25}, {})[
        "ledger_consistency_rate"
    ] == pytest.approx(0.25)
    assert mod.promotion_recommendation(False, 0, 0, 1.0).startswith("block_fr11")
    assert mod.promotion_recommendation(True, 1, 0, 1.0).startswith("block_fr11")
    assert mod.promotion_recommendation(True, 0, 0, 1.0) == (
        "promote_controller_environment_memory_only"
    )
    assert mod.honest_verdict(False, "blocked").startswith("blocked_precondition_failed")

    first_variant = mod.generate_and_validate_variants(
        mod.load_admitted_environments(_exp3128_payload())
    ).records[0]
    leaky = first_variant.environment.__class__(
        environment_id="leaky",
        family_id="leaky",
        variables=first_variant.environment.variables,
        domains=first_variant.environment.domains,
        constraints=first_variant.environment.constraints,
        prompt=(
            first_variant.environment.prompt
            + " "
            + evo.answer_text(first_variant.reference.canonical_assignment)
        ),
    )
    assert mod.no_answer_leakage_passed(leaky, leaky.compute_reference()) is False

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="live_call_count"):
        mod.validate_artifact(
            artifact
            | {
                "live_model_variant_generation": True,
                "live_call_count": 0,
            }
        )
    with pytest.raises(ValueError, match="no_weight_update_claim"):
        mod.validate_artifact(artifact | {"no_weight_update_claim": False})
    with pytest.raises(ValueError, match="model_weight_mutation"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_mutation": True}
            }
        )
    with pytest.raises(ValueError, match="soundness_errors"):
        mod.validate_artifact(artifact | {"soundness_errors": 1})
    with pytest.raises(ValueError, match="completeness_errors"):
        mod.validate_artifact(artifact | {"completeness_errors": 1})
    with pytest.raises(ValueError, match="equivalent_variant_count"):
        mod.validate_artifact(artifact | {"equivalent_variant_count": 0})
    with pytest.raises(ValueError, match="hardened_variant_count"):
        mod.validate_artifact(artifact | {"hardened_variant_count": 0})
    with pytest.raises(ValueError, match="ledger_consistency_rate"):
        mod.validate_artifact(artifact | {"ledger_consistency_rate": 1.5})
    with pytest.raises(ValueError, match="admitted_environment_count"):
        mod.validate_artifact(artifact | {"admitted_environment_count": 0})
    with pytest.raises(ValueError, match="solve_verify_asymmetry_pass_rate"):
        mod.validate_artifact(artifact | {"solve_verify_asymmetry_pass_rate": 0.5})
    with pytest.raises(ValueError, match="no_answer_leakage_pass_rate"):
        mod.validate_artifact(artifact | {"no_answer_leakage_pass_rate": 0.5})
    with pytest.raises(ValueError, match="source_artifacts"):
        mod.validate_artifact(
            artifact
            | {"source_artifacts": [{"path": "missing", "required": True, "exists": False}]}
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready: not terminal"})
