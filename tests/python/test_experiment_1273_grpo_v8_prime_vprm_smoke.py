"""Tests for Exp 1273 GRPO v8 PRIME/VPRM bounded smoke.

Spec: REQ-LEARN-1273, SCENARIO-LEARN-1273, SCENARIO-LEARN-1274.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.training import grpo_v8_prime_vprm_smoke as exp


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exp1272_payload(weights: dict[str, float] | None = None) -> dict:
    return {
        "experiment": "1272_prime_verifier_selection_audit",
        "status": "complete",
        "verifier_weight_vector": weights
        or {
            "Z3MathVerifier": 0.15,
            "SymCodeVerifier": 0.10,
            "CausalReasoningVerifier": 0.15,
            "SemEnergyProbe": 0.35,
            "SOSKANEnergyV3": 0.10,
            "k5_ensemble_summary": 0.15,
        },
    }


def test_in_progress_artifact_written_first_for_req1273(tmp_path: Path) -> None:
    """REQ-LEARN-1273-1: skeleton artifact is parseable before the run finishes."""

    output_path = tmp_path / "experiment_1273_grpo_v8_prime_vprm_smoke.json"

    artifact = exp.write_in_progress_artifact(output_path, run_date="20260504")

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["experiment"] == "1273_grpo_v8_prime_vprm_smoke"
    assert artifact["schema"] == "grpo_v8_prime_vprm_smoke_v1"
    assert artifact["status"] == "in_progress"
    assert artifact["honest_verdict"] == "in_progress"


def test_prime_vprm_reward_uses_exp1272_weights_for_req1273(tmp_path: Path) -> None:
    """REQ-LEARN-1273-2/4: VPRM and PRIME signals become a weighted reward."""

    exp1272_path = tmp_path / "experiment_1272_prime_verifier_selection_audit.json"
    _write_json(exp1272_path, _exp1272_payload())
    weights = exp.load_verifier_weights(exp1272_path)
    item = exp.build_smoke_slices(n_train=10, n_eval=20)["eval"][0]

    correct = exp.synthesise_response(item, phase="after")
    wrong = exp.synthesise_response(item, phase="before")
    correct_score = exp.score_weighted_prime_vprm_reward(item, correct, weights)
    wrong_score = exp.score_weighted_prime_vprm_reward(item, wrong, weights)

    assert correct_score["weighted_reward"] > wrong_score["weighted_reward"]
    assert wrong_score["signals"]["SemEnergyProbe"] == 1.0
    assert wrong_score["signals"]["k5_ensemble_summary"] == 1.0
    assert correct_score["signals"]["SemEnergyProbe"] == 0.0
    assert exp._mean([]) == 0.0


def test_missing_sota_cache_writes_smoke_only_artifact_for_scenario1273(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1273: missing cached_sota_pair() is non-headline smoke."""

    exp1272_path = tmp_path / "experiment_1272_prime_verifier_selection_audit.json"
    output_path = tmp_path / "experiment_1273_grpo_v8_prime_vprm_smoke.json"
    _write_json(exp1272_path, _exp1272_payload())

    artifact = exp.run_experiment(
        exp1272_path=exp1272_path,
        output_path=output_path,
        cached_pair_fn=lambda: None,
        run_date="20260504",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        wall_budget_s=90.0,
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert written == artifact
    exp.validate_artifact(artifact)
    assert artifact["execution_mode"] == "smoke_only"
    assert artifact["terminal_status"] == "smoke_only_no_sota_gguf"
    assert artifact["honest_verdict"] == "smoke_only_not_headline"
    assert artifact["headline_result_allowed"] is False
    assert artifact["n_train_items"] == 16
    assert artifact["n_eval_items"] == 24
    assert artifact["wall_budget_s"] == 90.0
    assert artifact["grpo_v8_delta_pp"] == pytest.approx(
        100.0 * artifact["self_learning_delta_overall"]
    )
    assert artifact["verifier_weights_used"] == _exp1272_payload()["verifier_weight_vector"]
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in {
        model["hf_id"] for model in artifact["models_used"]
    }


def test_live_sota_specs_can_claim_headline_for_scenario1274(tmp_path: Path) -> None:
    """SCENARIO-LEARN-1274: live_sota mode requires cached GGUF model specs."""

    exp1272_path = tmp_path / "experiment_1272_prime_verifier_selection_audit.json"
    output_path = tmp_path / "experiment_1273_grpo_v8_prime_vprm_smoke.json"
    _write_json(exp1272_path, _exp1272_payload())
    specs = [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/qwen.gguf",
        }
    ]

    artifact = exp.run_experiment(
        exp1272_path=exp1272_path,
        output_path=output_path,
        cached_pair_fn=lambda: specs,
        live_response_provider=lambda items, phase, _specs: [
            exp.synthesise_response(item, phase=phase) for item in items
        ],
        run_date="20260504",
    )

    exp.validate_artifact(artifact)
    assert artifact["execution_mode"] == "live_sota"
    assert artifact["terminal_status"] == "live_sota_complete"
    assert artifact["headline_result_allowed"] is True
    assert artifact["MODEL_SPECS"] == specs
    assert artifact["models_used"][0]["used_for_generation"] is True
    assert artifact["honest_verdict"].startswith("live_sota_delta_pp_")


def test_missing_weights_blocks_before_smoke_for_req1273(tmp_path: Path) -> None:
    """REQ-LEARN-1273-2/5: no Exp 1272 weights means blocked, not headline."""

    exp1272_path = tmp_path / "experiment_1272_prime_verifier_selection_audit.json"
    output_path = tmp_path / "experiment_1273_grpo_v8_prime_vprm_smoke.json"
    _write_json(exp1272_path, {"verifier_weight_vector": {}})

    artifact = exp.run_experiment(
        exp1272_path=exp1272_path,
        output_path=output_path,
        cached_pair_fn=lambda: None,
        run_date="20260504",
    )

    exp.validate_artifact(artifact)
    assert artifact["execution_mode"] == "blocked"
    assert artifact["status"] == "blocked"
    assert artifact["terminal_status"] == "blocked_missing_verifier_weights"
    assert artifact["honest_verdict"] == "blocked_missing_verifier_weights"
    assert artifact["headline_result_allowed"] is False
    assert artifact["grpo_v8_delta_pp"] == 0.0


def test_req1273_validation_and_edges_are_strict(tmp_path: Path) -> None:
    """REQ-LEARN-1273-3/4/5: schema validation rejects dishonest artifacts."""

    with pytest.raises(ValueError, match="verifier_weight_vector"):
        exp.load_verifier_weights(tmp_path / "missing.json")
    zero_weights = tmp_path / "zero_weights.json"
    _write_json(zero_weights, _exp1272_payload({"SemEnergyProbe": 0.0}))
    with pytest.raises(ValueError, match="positive weight"):
        exp.load_verifier_weights(zero_weights)
    with pytest.raises(ValueError, match="10-20 train"):
        exp.build_smoke_slices(n_train=9, n_eval=20)

    resolution = exp.resolve_model_specs(cached_pair_fn=lambda: [])
    assert resolution["cached_sota_available"] is False
    assert resolution["MODEL_SPECS"] == []

    item = exp.build_smoke_slices(n_train=10, n_eval=20)["eval"][0]
    no_answer = exp.score_weighted_prime_vprm_reward(
        item,
        "No numeric answer was provided.",
        _exp1272_payload()["verifier_weight_vector"],
    )
    assert no_answer["final_answer"] == ""

    with pytest.raises(ValueError, match="phase"):
        exp.synthesise_response(item, phase="later")

    artifact = exp.run_experiment(
        exp1272_path=tmp_path / "missing.json",
        output_path=tmp_path / "artifact.json",
        cached_pair_fn=lambda: None,
    )
    assert exp.derive_honest_verdict("blocked", artifact["terminal_status"], 0.0) == artifact["terminal_status"]
    del artifact["models_used"]
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact(artifact)

    valid = exp.run_experiment(
        exp1272_path=zero_weights.with_name("valid_weights.json"),
        output_path=tmp_path / "valid_artifact.json",
        cached_pair_fn=lambda: None,
    )
    assert valid["status"] == "blocked"

    exp1272_path = tmp_path / "weights_for_mutation.json"
    _write_json(exp1272_path, _exp1272_payload())
    good = exp.run_experiment(
        exp1272_path=exp1272_path,
        output_path=tmp_path / "good_artifact.json",
        cached_pair_fn=lambda: None,
    )
    bad_models = dict(good)
    bad_models["models_used"] = []
    with pytest.raises(AssertionError, match="mandated SOTA"):
        exp.validate_artifact(bad_models)
    bad_headline = dict(good)
    bad_headline["headline_result_allowed"] = True
    with pytest.raises(AssertionError, match="non-live_sota"):
        exp.validate_artifact(bad_headline)
    bad_delta = dict(good)
    bad_delta["grpo_v8_delta_pp"] = -999.0
    with pytest.raises(AssertionError, match="100 \\* self_learning"):
        exp.validate_artifact(bad_delta)

    live_bad = dict(good)
    live_bad["execution_mode"] = "live_sota"
    live_bad["MODEL_SPECS"] = [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}]
    live_bad["headline_result_allowed"] = False
    with pytest.raises(AssertionError, match="allow headline"):
        exp.validate_artifact(live_bad)
    live_bad_specs = dict(live_bad)
    live_bad_specs["headline_result_allowed"] = True
    live_bad_specs["MODEL_SPECS"] = []
    with pytest.raises(AssertionError, match="require MODEL_SPECS"):
        exp.validate_artifact(live_bad_specs)
