"""Bounded constrained candidate search around repair executor v2.

Exp 1428 proved that the repair path needs a strict DCCD object before semantic
validation. Exp 1429 adds the next layer: evaluate a small, bounded set of
schema-first candidates per repair-hint case, score the semantic-valid repairs
with verifier energy, and compare the first candidate path against best-of-N.

Spec: REQ-VERIFY-1429, SCENARIO-VERIFY-1429
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from carnot.pipeline.certificate_repair_executor import (
    CertificateRepairRequest,
    validation_accepts_repair,
)
from carnot.pipeline.dccd_schema_constrained_repair import (
    DCCDRepairCandidate,
    DCCDRepairConfig,
    DCCDRepairOutputSchemaError,
    build_dccd_repair_prompt,
    classify_dccd_rejection,
    parse_dccd_repair_model_output,
)


GeneratorFn = Callable[[str], str]
ValidatorFn = Callable[[CertificateRepairRequest, DCCDRepairCandidate], Mapping[str, Any]]
EnergyScorerFn = Callable[
    [CertificateRepairRequest, DCCDRepairCandidate, Mapping[str, Any]],
    float,
]


@dataclass(frozen=True)
class CandidateSearchConfig:
    """Runtime bounds for best-of-N constrained repair exploration."""

    candidates_per_case: int = 4
    max_candidates_per_case: int = 8
    max_field_chars: int = 1800
    max_output_chars: int = 4000

    def __post_init__(self) -> None:
        if self.candidates_per_case < 1:
            raise ValueError("candidates_per_case must be at least 1")
        if self.max_candidates_per_case < 1:
            raise ValueError("max_candidates_per_case must be at least 1")
        if self.candidates_per_case > self.max_candidates_per_case:
            raise ValueError("candidates_per_case must not exceed max_candidates_per_case")


@dataclass(frozen=True)
class CandidateEvaluation:
    """Audit record for one constrained repair proposal."""

    candidate_index: int
    accepted: bool
    schema_valid: bool
    semantic_accepted: bool
    energy: float | None
    validation_result: dict[str, Any]
    fallback_reason: str | None
    rejection_reason: str | None
    runtime_s: float
    candidate: DCCDRepairCandidate | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible proposal details for experiment artifacts."""

        return asdict(self)


@dataclass(frozen=True)
class CandidateSearchResult:
    """Best-of-N search outcome for one repair-hint case."""

    case_id: str
    candidates_evaluated: int
    accepted_candidate_count: int
    mcmc_acceptance_rate: float
    one_candidate_success: bool
    best_of_n_success: bool
    energy_rerank_improved: bool
    best_candidate_index: int | None
    best_candidate_energy: float | None
    selected_candidate: DCCDRepairCandidate | None
    candidate_results: list[CandidateEvaluation]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible case-level search details."""

        return asdict(self)


class BoundedConstrainedRepairCandidateSearch:
    """Generate, validate, score, and select bounded repair v2 candidates."""

    def __init__(
        self,
        *,
        generator: GeneratorFn,
        model_spec: Mapping[str, Any],
        validator: ValidatorFn,
        energy_scorer: EnergyScorerFn | None = None,
        config: CandidateSearchConfig | None = None,
    ) -> None:
        self._generator = generator
        self._model_spec = dict(model_spec)
        self._validator = validator
        self._energy_scorer = energy_scorer or verifier_energy_score
        self._config = config or CandidateSearchConfig()

    @property
    def model_id(self) -> str | None:
        """Return the auditable local model identifier selected for search."""

        value = self._model_spec.get("hf_id") or self._model_spec.get("name")
        return str(value) if value else None

    def search(self, request: CertificateRepairRequest) -> CandidateSearchResult:
        """Evaluate a bounded candidate set and return the best accepted repair."""

        evaluations: list[CandidateEvaluation] = []
        best_candidate: DCCDRepairCandidate | None = None
        best_candidate_index: int | None = None
        best_energy = math.inf

        for candidate_index in range(self._config.candidates_per_case):
            previous_best = best_candidate
            evaluation = self._evaluate_candidate(
                request=request,
                candidate_index=candidate_index,
                previous_best=previous_best,
            )
            evaluations.append(evaluation)
            if (
                evaluation.accepted
                and evaluation.energy is not None
                and evaluation.energy < best_energy
            ):
                best_candidate = evaluation.candidate
                best_candidate_index = candidate_index
                best_energy = evaluation.energy

        accepted_count = sum(1 for evaluation in evaluations if evaluation.accepted)
        first = evaluations[0]
        best_of_n_success = best_candidate is not None
        energy_rerank_improved = best_of_n_success and (
            not first.accepted
            or (
                best_candidate_index != 0
                and best_energy < (first.energy if first.energy is not None else math.inf)
            )
        )

        return CandidateSearchResult(
            case_id=request.case_id,
            candidates_evaluated=len(evaluations),
            accepted_candidate_count=accepted_count,
            mcmc_acceptance_rate=_rate(accepted_count, len(evaluations)),
            one_candidate_success=first.accepted,
            best_of_n_success=best_of_n_success,
            energy_rerank_improved=energy_rerank_improved,
            best_candidate_index=best_candidate_index,
            best_candidate_energy=None if best_candidate_index is None else round(best_energy, 6),
            selected_candidate=best_candidate,
            candidate_results=evaluations,
        )

    def _evaluate_candidate(
        self,
        *,
        request: CertificateRepairRequest,
        candidate_index: int,
        previous_best: DCCDRepairCandidate | None,
    ) -> CandidateEvaluation:
        started = time.perf_counter()
        try:
            prompt = build_mcmc_repair_prompt(
                request=request,
                candidate_index=candidate_index,
                previous_best=previous_best,
                config=self._config,
            )
            raw_output = self._generator(prompt)
            candidate = parse_dccd_repair_model_output(
                raw_output,
                DCCDRepairConfig(
                    max_field_chars=self._config.max_field_chars,
                    max_output_chars=self._config.max_output_chars,
                ),
            )
        except DCCDRepairOutputSchemaError as exc:
            return self._rejected_evaluation(
                candidate_index=candidate_index,
                started=started,
                fallback_reason="schema_validation_failed",
                validation_result={"error": str(exc)},
                schema_valid=False,
                candidate=None,
            )

        validation = dict(self._validator(request, candidate))
        energy = float(self._energy_scorer(request, candidate, validation))
        accepted = validation_accepts_repair(validation)
        if not accepted:
            return self._rejected_evaluation(
                candidate_index=candidate_index,
                started=started,
                fallback_reason="semantic_validation_failed",
                validation_result=validation,
                schema_valid=True,
                candidate=candidate,
                energy=energy,
            )

        return CandidateEvaluation(
            candidate_index=candidate_index,
            accepted=True,
            schema_valid=True,
            semantic_accepted=True,
            energy=round(energy, 6),
            validation_result=validation,
            fallback_reason=None,
            rejection_reason=None,
            runtime_s=round(time.perf_counter() - started, 6),
            candidate=candidate,
        )

    def _rejected_evaluation(
        self,
        *,
        candidate_index: int,
        started: float,
        fallback_reason: str,
        validation_result: Mapping[str, Any],
        schema_valid: bool,
        candidate: DCCDRepairCandidate | None,
        energy: float | None = None,
    ) -> CandidateEvaluation:
        validation = dict(validation_result)
        return CandidateEvaluation(
            candidate_index=candidate_index,
            accepted=False,
            schema_valid=schema_valid,
            semantic_accepted=False,
            energy=None if energy is None else round(energy, 6),
            validation_result=validation,
            fallback_reason=fallback_reason,
            rejection_reason=classify_dccd_rejection(fallback_reason, validation),
            runtime_s=round(time.perf_counter() - started, 6),
            candidate=candidate,
        )


def build_mcmc_repair_prompt(
    *,
    request: CertificateRepairRequest,
    candidate_index: int,
    previous_best: DCCDRepairCandidate | None,
    config: CandidateSearchConfig | None = None,
) -> str:
    """Build a DCCD repair prompt annotated with bounded search state."""

    cfg = config or CandidateSearchConfig()
    base_prompt = build_dccd_repair_prompt(
        request,
        DCCDRepairConfig(
            max_field_chars=cfg.max_field_chars,
            max_output_chars=cfg.max_output_chars,
        ),
    )
    search_state = {
        "spec": "REQ-VERIFY-1429",
        "candidate_index": candidate_index,
        "candidates_per_case": cfg.candidates_per_case,
        "previous_best": _candidate_summary(previous_best),
    }
    return (
        f"{base_prompt}\n"
        "Bounded constrained candidate-search state:\n"
        f"{json.dumps(search_state, sort_keys=True, indent=2)}\n"
        "Propose one schema-valid alternative repair candidate for this index.\n"
    )


def verifier_energy_score(
    _request: CertificateRepairRequest,
    _candidate: DCCDRepairCandidate,
    validation: Mapping[str, Any],
) -> float:
    """Convert existing verifier signals into a lower-is-better energy score."""

    for key in ("energy", "semantic_energy", "verification_energy"):
        value = validation.get(key)
        if isinstance(value, int | float):
            return float(value)

    penalty = 0.0
    if validation.get("constraint_passed") is not True:
        penalty += 10.0
    if validation.get("semantic_result") != "SAT":
        penalty += 10.0
    if validation.get("repair_required") is True:
        penalty += 5.0
    if validation.get("false_acceptance") is True:
        penalty += 20.0
    return penalty


def _candidate_summary(candidate: DCCDRepairCandidate | None) -> dict[str, str] | None:
    if candidate is None:
        return None
    return {
        "repair_action_type": candidate.repair_action_type,
        "repair_target": candidate.repair_target,
        "final_state": candidate.final_state,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0
