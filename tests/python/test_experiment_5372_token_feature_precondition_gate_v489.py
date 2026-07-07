"""Tests for Exp5372 token/internal-feature precondition gate.

Spec refs: REQ-VERIFY-5372, SCENARIO-VERIFY-5372.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5372_token_feature_precondition_gate_v489 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _wrap(value: Any) -> dict[str, Any]:
    return {"principle": "fixture principle", "value": value}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp5353_artifact() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5353.tokenprob_feature_audit_corrigendum.v488",
        "status": _wrap("complete"),
        "honest_verdict": _wrap("complete: tokenprob_feature_rows_ready"),
        "inference_substrate": _wrap("live_llm_inference"),
        "duration_s": 18.988115,
        "methodology_duration_s": 18.988115,
        "feature_audit_duration_s": 0.95139,
        "per_token_logprob_available": True,
        "topk_alternatives_available": True,
        "logits_available": False,
        "hidden_states_available": False,
        "attention_available": False,
        "prompt_completion_token_split_available": True,
        "token_timing_available": True,
        "tokenprob_feature_row_count": 3,
        "tokenprob_feature_rows_ready": True,
        "missing_feature_names": [],
        "feature_audit": {
            "per_token_logprob_available": True,
            "topk_alternatives_available": True,
            "logits_available": False,
            "hidden_states_available": False,
            "attention_available": False,
            "prompt_completion_token_split_available": True,
            "token_timing_available": True,
            "missing_feature_names": ["logits", "attention", "hidden_states"],
        },
        "preconditions_checked": _wrap(
            {
                "live_probe_attempted": True,
                "selected_backend_kind": "llama-server",
                "selected_backend_path": "/tmp/llama-server",
                "selected_runtime_backend_kind": "llama-cli",
                "selected_model_hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "selected_model_path": "/tmp/gemma-4-31B-it-Q4_K_M.gguf",
                "token_probability_api_available": True,
                "external_text_scorer_reopened": False,
            }
        ),
        "selected_model_spec": _wrap(
            {
                "role": "flagship_dense",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "model_path": "/tmp/gemma-4-31B-it-Q4_K_M.gguf",
            }
        ),
    }


def _exp5354_artifact() -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5354.arithmetic_carry_token_energy.v488",
        "status": _wrap("blocked"),
        "honest_verdict": _wrap(
            "blocked_carry_token_features_incomplete:no_carry_12_23:"
            "perturbed_target_logprob,correct_vs_perturbed_margin_nonpositive"
        ),
        "inference_substrate": _wrap("live_llm_inference"),
        "duration_s": 82.932366,
        "methodology_duration_s": 82.932366,
        "diagnostic_case_count": 16,
        "carry_case_count": 8,
        "feature_complete_rate": 0.5625,
        "correct_vs_perturbed_margin": 0.0,
        "unsafe_false_accepts": 0,
        "carry_token_energy_signal_ready": False,
        "missing_feature_names": [
            "no_carry_12_23:perturbed_target_logprob",
            "correct_vs_perturbed_margin_nonpositive",
        ],
        "carry_token_energy_feature_rows": _wrap(
            [
                {
                    "case_id": "no_carry_12_23",
                    "category": "no_carry",
                    "feature_complete": False,
                    "missing_features": ["perturbed_target_logprob"],
                    "correct_vs_perturbed_margin": None,
                },
                {
                    "case_id": "single_carry_46_37",
                    "category": "single_carry",
                    "feature_complete": True,
                    "missing_features": [],
                    "correct_vs_perturbed_margin": 11.046974491,
                },
            ]
        ),
    }


def _capstone_artifact() -> dict[str, Any]:
    return {
        "tokenprob_feature_rows_ready": True,
        "carry_token_energy_signal_ready": False,
        "missing_blocked_flagged_or_skipped_artifacts": _wrap(
            [
                {
                    "experiment_number": 5353,
                    "classification": "flagged",
                    "corrigendum_pending": [
                        {
                            "kind": "TAUTOLOGY",
                            "severity": "critical",
                            "detail": "duration_s=18.988115 and methodology_duration_s=18.988115 agree",
                        },
                        {
                            "kind": "DURATION_TOO_SHORT",
                            "severity": "critical",
                            "detail": "duration_s=18.988115 but live model takes >=60.0s minimum",
                        },
                    ],
                },
                {
                    "experiment_number": 5354,
                    "classification": "blocked_and_flagged",
                    "corrigendum_pending": [
                        {
                            "kind": "TAUTOLOGY",
                            "severity": "critical",
                            "detail": "duration_s=82.932366 and methodology_duration_s=82.932366 agree",
                        }
                    ],
                },
            ]
        ),
    }


def _make_repo(root: Path) -> Path:
    _write_json(root / exp.EXP5353_RELATIVE_PATH, _exp5353_artifact())
    _write_json(root / exp.EXP5354_RELATIVE_PATH, _exp5354_artifact())
    _write_json(root / exp.CAPSTONE_RELATIVE_PATH, _capstone_artifact())
    (root / exp.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(
        "\n".join(
            [
                "HalluField uses token-path thermodynamic instability from logits.",
                "FLaG-style latent probes need hidden states and attention evidence.",
                "Attention energy claims need attention tensors.",
            ]
        ),
        encoding="utf-8",
    )
    return root


def test_req_verify_5372_spec_declares_precondition_gate_contract() -> None:
    """REQ-VERIFY-5372: OpenSpec anchors token/internal-feature no-go decisions."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5372") : spec.index("### REQ-VERIFY-5345")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5372",
        "SCENARIO-VERIFY-5372",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp 5353",
        "Exp 5354",
        "future_signal_allowed",
        "carry_token_energy_continue",
        "retire_recommendation",
        "methodology_min_duration_s >= 60",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5372_extracts_missing_and_tautological_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5372: flagged source fields are preserved exactly."""

    root = _make_repo(tmp_path)
    artifact = exp.build_artifact(
        root=root,
        tests_run=[{"command": "unit exp5372", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    missing = artifact["missing_or_tautological_fields"]

    assert missing["exp5353"]["feature_audit_missing_feature_names"] == [
        "logits",
        "attention",
        "hidden_states",
    ]
    assert missing["exp5353"]["top_level_missing_feature_names"] == []
    assert missing["exp5353"]["top_level_omits_nested_latent_missing"] is True
    assert missing["exp5353"]["duration_tautology"] is True
    assert "DURATION_TOO_SHORT" in missing["exp5353"]["adversarial_flag_kinds"]
    assert missing["exp5354"]["missing_feature_names"] == [
        "no_carry_12_23:perturbed_target_logprob",
        "correct_vs_perturbed_margin_nonpositive",
    ]
    assert missing["exp5354"]["incomplete_row_missing_features"] == {
        "no_carry_12_23": ["perturbed_target_logprob"]
    }
    assert missing["exp5354"]["duration_tautology"] is True
    assert "TAUTOLOGY" in missing["exp5354"]["adversarial_flag_kinds"]


def test_scenario_verify_5372_retires_carry_lane_for_logprob_only_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5372: logprob-only evidence cannot open signal claims."""

    root = _make_repo(tmp_path)
    artifact = exp.run(
        root=root,
        result_path=root / exp.RESULT_RELATIVE_PATH,
        tests_run=[{"command": "unit exp5372", "outcome": "passed"}],
    )

    assert json.loads((root / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["token_feature_gate_ready"] is True
    assert artifact["tokenprob_rows_available"] is True
    assert artifact["logits_available"] is False
    assert artifact["hidden_states_available"] is False
    assert artifact["attention_available"] is False
    assert artifact["completion_split_available"] is True
    assert artifact["methodology_min_duration_s"] == 60.0
    assert artifact["future_signal_allowed"] is False
    assert artifact["carry_token_energy_continue"] is False
    assert artifact["retire_recommendation"] is True
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["tests_run"] == [{"command": "unit exp5372", "outcome": "passed"}]
    assert artifact["honest_verdict"].startswith("complete:")
    assert any("HalluField" in claim for claim in artifact["forbidden_claims"])
    assert any("carry-token energy margin" in claim for claim in artifact["forbidden_claims"])
    assert artifact["bounded_continuation_allowed"] == [
        "feature-surface receipt refresh only",
        "backend upgrade preflight for logits/hidden states/attention",
    ]


def test_req_verify_5372_validation_rejects_unsafe_continuation(tmp_path: Path) -> None:
    """REQ-VERIFY-5372: validation fails closed on unsupported signal claims."""

    artifact = exp.build_artifact(
        root=_make_repo(tmp_path),
        tests_run=[{"command": "unit exp5372", "outcome": "passed"}],
    )

    assert exp._numeric("not numeric") is None

    bad_missing = deepcopy(artifact)
    del bad_missing["status"]
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(bad_missing)

    bad_duration = deepcopy(artifact)
    bad_duration["methodology_min_duration_s"] = 30.0
    with pytest.raises(ValueError, match="methodology_min_duration_s"):
        exp.validate_artifact(bad_duration)

    bad_signal = deepcopy(artifact)
    bad_signal["future_signal_allowed"] = True
    with pytest.raises(ValueError, match="future_signal_allowed"):
        exp.validate_artifact(bad_signal)

    bad_taut_signal = deepcopy(artifact)
    bad_taut_signal["future_signal_allowed"] = True
    bad_taut_signal["logits_available"] = True
    with pytest.raises(ValueError, match="flagged or tautological"):
        exp.validate_artifact(bad_taut_signal)

    bad_continue = deepcopy(artifact)
    bad_continue["carry_token_energy_continue"] = True
    with pytest.raises(ValueError, match="carry_token_energy_continue"):
        exp.validate_artifact(bad_continue)

    bad_claims = deepcopy(artifact)
    bad_claims["forbidden_claims"] = []
    with pytest.raises(ValueError, match="forbidden_claims"):
        exp.validate_artifact(bad_claims)

    bad_false_accepts = deepcopy(artifact)
    bad_false_accepts["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_false_accepts)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    bad_ready = deepcopy(artifact)
    bad_ready["token_feature_gate_ready"] = False
    with pytest.raises(ValueError, match="token_feature_gate_ready"):
        exp.validate_artifact(bad_ready)


def test_req_verify_5372_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5372: checked-in result is stable under deterministic replay."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=artifact["tests_run"])

    assert artifact == replay
    assert artifact["status"] == "complete"
    assert artifact["token_feature_gate_ready"] is True
    assert artifact["future_signal_allowed"] is False
    assert artifact["carry_token_energy_continue"] is False
    assert artifact["retire_recommendation"] is True
    exp.validate_artifact(artifact)
