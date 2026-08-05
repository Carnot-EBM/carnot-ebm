"""Tests for Exp6148 shifted-family admission held evaluation.

Spec refs: REQ-VERIFY-6148, REQ-VERIFY-6148-1, REQ-VERIFY-6148-2,
REQ-VERIFY-6148-3, REQ-VERIFY-6148-4, REQ-VERIFY-6148-5,
REQ-VERIFY-6148-6, REQ-VERIFY-6148-7, REQ-VERIFY-6148-8,
REQ-VERIFY-6148-9, SCENARIO-VERIFY-6148-ONE-SHOT,
SCENARIO-VERIFY-6148-PAIRED, SCENARIO-VERIFY-6148-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6148_shifted_family_admission_held as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _run_artifact(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.25,
        write=write,
    )


def test_req_6148_spec_declares_held_contract_and_fields() -> None:
    """REQ-VERIFY-6148/SCENARIO-VERIFY-6148-ONE-SHOT: spec anchors the work."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6148") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6148-1",
        "REQ-VERIFY-6148-2",
        "REQ-VERIFY-6148-3",
        "REQ-VERIFY-6148-4",
        "REQ-VERIFY-6148-5",
        "REQ-VERIFY-6148-6",
        "REQ-VERIFY-6148-7",
        "REQ-VERIFY-6148-8",
        "REQ-VERIFY-6148-9",
        "SCENARIO-VERIFY-6148-ONE-SHOT",
        "SCENARIO-VERIFY-6148-PAIRED",
        "SCENARIO-VERIFY-6148-ATTACKS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6148_one_shot_guard_on_synthetic_sealed_rows() -> None:
    """SCENARIO-VERIFY-6148-ONE-SHOT: synthetic held labels materialize once."""

    rows_by_model = {
        "model-a": [
            {
                "event_id": "e1",
                "partition": "future_known",
                "base_template_id": "b1",
                "current_validator_result": "accepted",
                "invalid_output": False,
            },
            {
                "event_id": "e2",
                "partition": "sealed_shifted_family",
                "base_template_id": "b2",
                "current_validator_result": "rejected",
                "invalid_output": False,
            },
            {
                "event_id": "e3",
                "partition": "calibration",
                "base_template_id": "b3",
                "current_validator_result": "accepted",
                "invalid_output": False,
            },
            {
                "event_id": "e4",
                "partition": "ignored_sidecar_partition",
                "base_template_id": "b4",
                "current_validator_result": "accepted",
                "invalid_output": False,
            },
        ]
    }
    guard = mod.HeldAccessGuard(prior_receipt_seen=False)

    held_rows, receipt = guard.unseal(rows_by_model, expected_event_ids_by_partition={})

    assert guard.access_count == 1
    assert receipt["held_access_count"] == 1
    assert receipt["future_known_label_read_count"] == 1
    assert receipt["sealed_shifted_family_label_read_count"] == 1
    assert receipt["calibration_label_read_count"] == 0
    assert receipt["held_label_read_count"] == 2
    assert {row["partition"] for row in held_rows["model-a"]} == set(mod.HELD_PARTITIONS)

    with pytest.raises(mod.HeldAccessError, match="exactly one"):
        guard.unseal(rows_by_model, expected_event_ids_by_partition={})
    with pytest.raises(mod.HeldAccessError, match="prior held-access"):
        mod.HeldAccessGuard(prior_receipt_seen=True).unseal(
            rows_by_model, expected_event_ids_by_partition={}
        )
    assert mod._exp6147_receipt_hash({}, Path("missing.py")) is None
    assert mod._shuffled_family_for_attack("new_family") == "new_family"


def test_scenario_6148_grouped_paired_metrics_on_synthetic_scores() -> None:
    """SCENARIO-VERIFY-6148-PAIRED: grouped deltas stay separated by held group."""

    selection = {
        "threshold": 0.5,
        "abstention_rule": {"margin": 0.0},
    }
    entries = [
        {
            "model_hf_id": "model-a",
            "partition": "future_known",
            "event_id": "e1",
            "base_template_id": "b1",
            "family": "known",
            "variant_kind": "canonical",
            "unsafe_label": 0,
            "scores": {"global_energy": 0.2, "task_aware_energy": 0.1},
        },
        {
            "model_hf_id": "model-a",
            "partition": "future_known",
            "event_id": "e2",
            "base_template_id": "b2",
            "family": "known",
            "variant_kind": "strategy_poison",
            "unsafe_label": 1,
            "scores": {"global_energy": 0.3, "task_aware_energy": 1.4},
        },
        {
            "model_hf_id": "model-a",
            "partition": "sealed_shifted_family",
            "event_id": "e3",
            "base_template_id": "b3",
            "family": "shifted",
            "variant_kind": "alias",
            "unsafe_label": 0,
            "scores": {"global_energy": 0.9, "task_aware_energy": 0.0},
        },
        {
            "model_hf_id": "model-a",
            "partition": "sealed_shifted_family",
            "event_id": "e4",
            "base_template_id": "b4",
            "family": "shifted",
            "variant_kind": "malformed_proposal",
            "unsafe_label": 1,
            "scores": {"global_energy": 0.4, "task_aware_energy": 1.5},
        },
    ]

    evaluated = mod.evaluate_scored_entries(
        entries,
        selection=selection,
        model_ids=("model-a",),
        bootstrap_replicates=20,
    )

    metrics = evaluated["per_model_future_known_and_shifted_metrics"]["by_model"]["model-a"]
    assert set(metrics) == set(mod.HELD_PARTITIONS)
    assert metrics["future_known"]["scores"]["task_aware_energy"]["auroc"] == 1.0
    assert metrics["sealed_shifted_family"]["scores"]["global_energy"]["auroc"] == 0.0
    assert metrics["sealed_shifted_family"]["scores"]["task_aware_energy"]["auroc"] == 1.0

    intervals = evaluated["paired_task_aware_minus_global_intervals"]["by_model"]["model-a"]
    assert intervals["sealed_shifted_family"]["auroc_delta"]["observed"] == 1.0
    assert intervals["sealed_shifted_family"]["auroc_delta"]["ci95"][0] >= 0.0

    matrices = evaluated["unsafe_acceptance_and_abstention_matrices"]["by_model"]["model-a"]
    assert (
        matrices["future_known"]["task_aware_energy"]["confusion_matrix"]["false_unsafe_acceptance"]
        == 0
    )
    assert evaluated["safe_acceptance_noninferiority"]["passed"] is True
    assert (
        evaluated["alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks"][
            "all_required_attacks_present"
        ]
        is True
    )


def test_req_6148_real_artifact_is_complete_null_with_one_held_read(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6148-3/8/9: real held read produces a strict null artifact."""

    artifact = _run_artifact(tmp_path, write=True)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert "shifted_primary_metric_lower_ci_not_positive" in artifact["honest_verdict"]
    assert artifact["shifted_family_admission_ready_score"] == 0.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["selector_refit_count"] == 0
    assert all(value == 0 for value in artifact["prompt_retry_and_llm_invocation_counts"].values())
    assert artifact["first_and_only_held_access_receipt"]["held_access_count"] == 1
    assert artifact["first_and_only_held_access_receipt"]["held_label_read_count"] == 240
    assert artifact["held_group_row_conservation"]["all_models_conserved"] is True
    assert artifact["safe_acceptance_noninferiority"]["passed"] is True
    assert (
        artifact["alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks"][
            "any_attack_wins"
        ]
        is False
    )
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    for model_id in mod.MANDATED_MODEL_IDS:
        model_metrics = artifact["per_model_future_known_and_shifted_metrics"]["by_model"][model_id]
        assert set(model_metrics) == set(mod.HELD_PARTITIONS)
        shifted = model_metrics["sealed_shifted_family"]
        assert shifted["scores"]["task_aware_energy"]["unsafe_count"] == 12
        assert shifted["scores"]["global_energy"]["auroc"] == 1.0
        assert shifted["scores"]["task_aware_energy"]["auroc"] == 1.0


def test_req_6148_validation_blocks_prior_receipt_selector_mismatch_and_refit(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6148-1/2/8: selector drift and refits fail closed."""

    artifact = _run_artifact(tmp_path)

    bad_access = deepcopy(artifact)
    bad_access["first_and_only_held_access_receipt"]["held_access_count"] = 2
    bad_access["reproducibility_checksum"] = mod.reproducibility_checksum(bad_access)
    assert "held_access_count_not_one" in mod._blocked_reasons(bad_access)
    with pytest.raises(ValueError, match="first_and_only_held_access_receipt"):
        mod.validate_artifact(bad_access)

    no_access = deepcopy(artifact)
    no_access["first_and_only_held_access_receipt"]["held_access_count"] = 0
    assert "first_and_only_held_access_receipt" in mod._blocked_reasons(no_access)

    bad_refit = deepcopy(artifact)
    bad_refit["selector_refit_count"] = 1
    bad_refit["shifted_family_admission_ready_score"] = mod.ready_score(bad_refit)
    bad_refit["status"] = mod.status(bad_refit)
    bad_refit["honest_verdict"] = mod.honest_verdict(bad_refit)
    bad_refit["reproducibility_checksum"] = mod.reproducibility_checksum(bad_refit)
    assert bad_refit["shifted_family_admission_ready_score"] == 0.0
    with pytest.raises(ValueError, match="selector_refit_count"):
        mod.validate_artifact(bad_refit)

    bad_safe = deepcopy(artifact)
    bad_safe["safe_acceptance_noninferiority"]["passed"] = False
    assert "future_known_safe_acceptance_noninferiority" in mod._blocked_reasons(bad_safe)

    bad_attack = deepcopy(artifact)
    bad_attack["alias_shuffle_frequency_poison_duplicate_identity_and_boundary_attacks"][
        "any_attack_wins"
    ] = True
    assert "attack_wins" in mod._blocked_reasons(bad_attack)

    bad_prompt_reason = deepcopy(artifact)
    bad_prompt_reason["prompt_retry_and_llm_invocation_counts"]["llm_invocation_count"] = 1
    assert "prompt_retry_or_llm_invocation" in mod._blocked_reasons(bad_prompt_reason)

    positive = deepcopy(artifact)
    for model_id in mod.MANDATED_MODEL_IDS:
        delta = positive["paired_task_aware_minus_global_intervals"]["by_model"][model_id][
            "sealed_shifted_family"
        ]["auroc_delta"]
        delta["observed"] = 0.1
        delta["ci95"] = [0.01, 0.2]
        delta["positive_lower_95"] = True
    positive["shifted_family_admission_ready_score"] = mod.ready_score(positive)
    assert positive["shifted_family_admission_ready_score"] == 1.0
    assert mod.status(positive) == "complete_positive"
    assert mod.honest_verdict(positive).startswith("complete_positive:")

    retired = deepcopy(artifact)
    retired["retirement_triggered"] = True
    assert mod.status(retired) == "retired"
    assert mod.honest_verdict(retired).startswith("retired:")

    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps({"first_and_only_held_access_receipt": {"held_access_count": 1}}),
        encoding="utf-8",
    )
    prior = mod.run(
        result_path=prior_path,
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.5,
    )
    assert prior["status"] == "blocked"
    assert prior["first_and_only_held_access_receipt"]["held_access_count"] == 0
    assert "prior_held_access_receipt" in prior["honest_verdict"]
    assert mod.validate_artifact(prior) is True

    freeze = mod.load_json(mod.REPO_ROOT / mod.EXP6147_RESULT_RELATIVE_PATH)
    freeze["selection_manifest_hash"] = mod.sha256_text("mismatch")
    mismatch = mod.run(
        result_path=tmp_path / "mismatch.json",
        exp6147_artifact=freeze,
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.5,
    )
    assert mismatch["status"] == "blocked"
    assert mismatch["first_and_only_held_access_receipt"]["held_access_count"] == 0
    assert "selection_manifest_mismatch" in mismatch["honest_verdict"]
    assert mod.validate_artifact(mismatch) is True

    mismatch_access = deepcopy(mismatch)
    mismatch_access["first_and_only_held_access_receipt"]["held_access_count"] = 2
    mismatch_access["reproducibility_checksum"] = mod.reproducibility_checksum(mismatch_access)
    with pytest.raises(ValueError, match="first_and_only_held_access_receipt"):
        mod.validate_artifact(mismatch_access)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_prompt = deepcopy(artifact)
    bad_prompt["prompt_retry_and_llm_invocation_counts"]["llm_invocation_count"] = 1
    bad_prompt["reproducibility_checksum"] = mod.reproducibility_checksum(bad_prompt)
    with pytest.raises(ValueError, match="prompt_retry_and_llm_invocation_counts"):
        mod.validate_artifact(bad_prompt)

    bad_score = deepcopy(artifact)
    bad_score["shifted_family_admission_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="shifted_family_admission_ready_score"):
        mod.validate_artifact(bad_score)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete_positive"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete_null: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "simulation"
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_is_oracle"] = True
    bad_verifier["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verifier)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_verifier)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)


def test_req_6148_adversarial_verify_accepts_sealed_cached_substrate(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6148-9: sealed cached evaluation is not live GGUF inference."""

    artifact = _run_artifact(tmp_path, write=True)
    report = adversarial_verify.verify_artifact(tmp_path / mod.RESULT_RELATIVE_PATH.name)
    kinds = {flag["kind"] for flag in report["flags"]}

    assert adversarial_verify._classify_inference_substrate(artifact)["kind"] == "no_llm"
    assert "DURATION_TOO_SHORT" not in kinds
    assert "METHODOLOGY_MISSING" not in kinds
