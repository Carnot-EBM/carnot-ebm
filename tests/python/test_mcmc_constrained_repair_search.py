"""Tests for Exp 1429 bounded constrained repair candidate search.

Spec: REQ-VERIFY-1429, SCENARIO-VERIFY-1429
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from carnot.pipeline.certificate_repair_executor import CertificateRepairRequest
from carnot.pipeline.mcmc_constrained_repair_search import (
    BoundedConstrainedRepairCandidateSearch,
    CandidateSearchConfig,
    verifier_energy_score,
)


def _request() -> CertificateRepairRequest:
    return CertificateRepairRequest(
        case_id="case_1429",
        original_prompt="Question: compute 2 + 2. Candidate reasoning says 5.",
        current_certificate="<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: repair step.",
        repair_hint="Repair the localized FoVer reasoning step before accepting.",
        validator_error="semantic_result=REPAIR_HINT; repair_required=True",
    )


def _dccd_payload(*, variant: str, final_text: str = "<CARNOT_CERT_STATE:SAT>\nSAT") -> str:
    return json.dumps(
        {
            "draft_certificate": {
                "certificate_text": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: repair step.",
                "state": "REPAIR_HINT",
            },
            "repair_action": {
                "action_type": "STEP_REWRITE",
                "target": f"{variant} localized step",
                "rationale": "Bounded candidate search proposes a concrete repair.",
            },
            "final_certificate": {
                "certificate_text": final_text,
                "state": "SAT",
            },
            "validator_metadata": {"variant": variant},
        }
    )


def test_scenario1429_best_of_n_accepts_later_semantic_valid_candidate() -> None:
    """SCENARIO-VERIFY-1429: best-of-N succeeds when the first candidate fails."""

    outputs = iter(
        [
            "not json",
            _dccd_payload(variant="semantic_fail"),
            _dccd_payload(variant="semantic_pass"),
        ]
    )
    validator_calls: list[str] = []

    def validator(_request: CertificateRepairRequest, candidate: Any) -> dict[str, Any]:
        validator_calls.append(candidate.repair_target)
        accepted = candidate.validator_metadata["variant"] == "semantic_pass"
        validation = {
            "constraint_passed": accepted,
            "semantic_result": "SAT" if accepted else "REPAIR_HINT",
            "repair_required": not accepted,
            "false_acceptance": False,
        }
        if accepted:
            validation["energy"] = 0.5
        return validation

    search = BoundedConstrainedRepairCandidateSearch(
        generator=lambda _prompt: next(outputs),
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=validator,
        config=CandidateSearchConfig(candidates_per_case=3),
    )

    result = search.search(_request())

    assert validator_calls == ["semantic_fail localized step", "semantic_pass localized step"]
    assert result.candidates_evaluated == 3
    assert result.one_candidate_success is False
    assert result.best_of_n_success is True
    assert result.mcmc_acceptance_rate == pytest.approx(1 / 3)
    assert result.best_candidate_index == 2
    assert result.energy_rerank_improved is True
    assert result.candidate_results[0].schema_valid is False
    assert result.candidate_results[0].rejection_reason == "schema_validation_failed"
    assert result.candidate_results[2].to_dict()["accepted"] is True
    assert search.model_id == "unsloth/Qwen3.6-35B-A3B-GGUF"


def test_req1429_energy_rerank_selects_lowest_energy_accepted_candidate() -> None:
    """REQ-VERIFY-1429: accepted candidates are ranked by verifier energy."""

    outputs = iter(
        [
            _dccd_payload(variant="high_energy", final_text="<CARNOT_CERT_STATE:SAT>\nHIGH"),
            _dccd_payload(variant="low_energy", final_text="<CARNOT_CERT_STATE:SAT>\nLOW"),
        ]
    )

    def validator(_request: CertificateRepairRequest, candidate: Any) -> dict[str, Any]:
        return {
            "constraint_passed": True,
            "semantic_result": "SAT",
            "repair_required": False,
            "false_acceptance": False,
            "energy": 5.0 if candidate.validator_metadata["variant"] == "high_energy" else 1.0,
        }

    search = BoundedConstrainedRepairCandidateSearch(
        generator=lambda _prompt: next(outputs),
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        validator=validator,
        config=CandidateSearchConfig(candidates_per_case=2),
    )

    result = search.search(_request())

    assert result.one_candidate_success is True
    assert result.best_of_n_success is True
    assert result.best_candidate_index == 1
    assert result.best_candidate_energy == pytest.approx(1.0)
    assert result.energy_rerank_improved is True
    assert result.selected_candidate is not None
    assert result.selected_candidate.final_certificate == "<CARNOT_CERT_STATE:SAT>\nLOW"


def test_req1429_candidate_budget_is_bounded() -> None:
    """REQ-VERIFY-1429: candidate search rejects unbounded proposal counts."""

    with pytest.raises(ValueError, match="candidates_per_case"):
        CandidateSearchConfig(candidates_per_case=0)

    with pytest.raises(ValueError, match="max_candidates_per_case"):
        CandidateSearchConfig(max_candidates_per_case=0)

    with pytest.raises(ValueError, match="max_candidates_per_case"):
        CandidateSearchConfig(candidates_per_case=5, max_candidates_per_case=4)


def test_req1429_default_verifier_energy_penalizes_failed_signals() -> None:
    """REQ-VERIFY-1429: missing explicit energy falls back to verifier penalties."""

    request = _request()
    search = BoundedConstrainedRepairCandidateSearch(
        generator=lambda _prompt: _dccd_payload(variant="penalty"),
        model_spec={"name": "fallback-model"},
        validator=lambda _request, _candidate: {
            "constraint_passed": False,
            "semantic_result": "REPAIR_HINT",
            "repair_required": True,
            "false_acceptance": True,
        },
        config=CandidateSearchConfig(candidates_per_case=1),
    )
    result = search.search(request)

    assert result.best_of_n_success is False
    assert result.candidate_results[0].energy == pytest.approx(45.0)
    assert search.model_id == "fallback-model"
    assert verifier_energy_score(
        request,
        result.candidate_results[0].candidate,
        {"semantic_energy": 3},
    ) == pytest.approx(3.0)
