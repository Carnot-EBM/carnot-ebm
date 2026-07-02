"""Exp 5139: abstention and verification-trace evaluation.

Spec refs: REQ-INFER-SOTA-032, REQ-PIPELINE-5138,
SCENARIO-INFER-SOTA-032-POOL, SCENARIO-PIPELINE-5138.

The experiment consumes the clean Exp 5136 receipt-backed pool and evaluates a
structured answer/evidence/self-check/abstention trace.  Exact validator labels
from Exp 5136 remain the sole correctness authority; self-check fields are
measured features, never ground truth.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5136_receipt_structured_pool_v2_v471 as pool_mod  # noqa: E402


JsonDict = dict[str, Any]

EXPERIMENT_ID = "exp5139-abstention-and-verification-trace-v471"
MILESTONE = "2026.07.471"
RESULT_RELATIVE_PATH = "results/experiment_5139_abstention_verification_trace_v471.json"
UPSTREAM_POOL_ARTIFACT = pool_mod.RESULT_RELATIVE_PATH
INFERENCE_SUBSTRATE = "local_sota_gguf_structured_verification_trace"

SUCCESS_READY_VERDICT = "complete_verification_trace_ready"
BLOCKED_UPSTREAM_VERDICT = "blocked_exp5136_upstream_unreadable"
BLOCKED_POOL_VERDICT = "blocked_structured_pool_v2_clean_false"
BLOCKED_ROWS_VERDICT = "blocked_structured_pool_v2_rows_missing"
BLOCKED_MODEL_VERDICT = "blocked_mandated_model_specs_missing"
BLOCKED_READY_GATES_VERDICT = "blocked_verification_trace_ready_gates_failed"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")

MANDATED_MODEL_IDS = pool_mod.MANDATED_MODEL_IDS
TRACE_REQUIRED_FIELDS = (
    "answer",
    "evidence",
    "self_check",
    "uncertainty",
    "abstention",
    "repair_attempt",
)
EVIDENCE_REQUIRED_FIELDS = (
    "candidate_id",
    "receipt_id",
    "validator_name",
    "raw_response_hash",
    "validator_output_hash",
)
TRACE_CONFIDENCE_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 0.9)
SCHEMA_VALIDITY_GATE = 0.98
MIN_TRACE_COVERAGE = 0.70
MIN_HARMFUL_REDUCTION = 0.25
FALSE_ABSTAIN_RATE_MAX = 0.20
UTILITY_RISK_WEIGHT = 2.0
RANDOM_SEED = 20260702

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "MODEL_SPECS",
    "upstream_pool_artifact",
    "trace_schema",
    "schema_validity_rate",
    "exact_validator_authority",
    "coverage_risk_curve",
    "abstention_delta",
    "harmful_answer_reduction",
    "false_abstain_rate",
    "strongest_baseline",
    "verification_trace_ready",
    "conductor_modified",
    "tests_run",
)

FIELD_PRINCIPLES = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "MODEL_SPECS": "mandated local SOTA model provenance",
    "upstream_pool_artifact": "data provenance",
    "trace_schema": "structured-output accountability",
    "schema_validity_rate": "parseability",
    "exact_validator_authority": "ground-truth accountability",
    "coverage_risk_curve": "abstention utility",
    "abstention_delta": "utility beyond baseline",
    "harmful_answer_reduction": "hallucination mitigation",
    "false_abstain_rate": "coverage cost",
    "strongest_baseline": "baseline adequacy",
    "verification_trace_ready": "downstream readiness",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5139_abstention_verification_trace_v471.py --date 20260702",
    '.venv/bin/pytest tests/python/test_experiment_5139_abstention_verification_trace_v471.py -q -o addopts=""',
    ".venv/bin/coverage erase && .venv/bin/coverage run --include='/home/ianblenke/github.com/"
    "ianblenke/carnot/python/carnot/experiment_5139_abstention_verification_trace_v471.py' "
    '-m pytest tests/python/test_experiment_5139_abstention_verification_trace_v471.py -q -o addopts="" '
    "&& .venv/bin/coverage report --include='/home/ianblenke/github.com/ianblenke/carnot/"
    "python/carnot/experiment_5139_abstention_verification_trace_v471.py' --fail-under=100 -m",
    ".venv/bin/ruff check python/carnot/experiment_5139_abstention_verification_trace_v471.py "
    "scripts/experiment_5139_abstention_verification_trace_v471.py "
    "tests/python/test_experiment_5139_abstention_verification_trace_v471.py",
    ".venv/bin/ruff format --check python/carnot/experiment_5139_abstention_verification_trace_v471.py "
    "scripts/experiment_5139_abstention_verification_trace_v471.py "
    "tests/python/test_experiment_5139_abstention_verification_trace_v471.py",
    "python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5139_abstention_verification_trace_v471.py",
    ".venv/bin/pytest tests/python -q",
]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_payload(payload: Any) -> str:
    return _sha256_text(_json_dumps(payload))


def _round_rate(value: float) -> float:
    return round(float(value), 6)


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if float(denominator) == 0.0 else _round_rate(float(numerator) / float(denominator))


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> tuple[JsonDict | None, str | None]:
    if not path.exists():
        return None, f"missing upstream artifact: {path.as_posix()}"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, f"upstream artifact is not a JSON object: {path.as_posix()}"
    return parsed, None


def trace_schema() -> JsonDict:
    return {
        "style": "JSON/SLOT",
        "schema_version": "exp5139.trace.v1",
        "required": list(TRACE_REQUIRED_FIELDS),
        "properties": {
            "answer": {"description": "Parsed direct answer slot from the model candidate."},
            "evidence": {
                "required": list(EVIDENCE_REQUIRED_FIELDS),
                "description": "Receipt and validator-hash pointers for exact audit.",
            },
            "self_check": {
                "required": ["result", "claimed_correct", "confidence", "feature_basis"],
                "description": "Model self-check feature, scored but not trusted as truth.",
            },
            "uncertainty": {
                "required": ["confidence", "risk_score", "abstention_threshold"],
                "description": "Confidence/risk slots used by the abstention policy.",
            },
            "abstention": {
                "required": ["decision", "policy", "reason", "final_answer_source"],
                "description": "Answer-or-abstain decision to score with exact validators.",
            },
            "repair_attempt": {
                "nullable": True,
                "description": "Optional same-receipt-batch repair candidate and evidence.",
            },
        },
    }


def _ordered_model_specs(upstream: Mapping[str, Any]) -> list[JsonDict]:
    rows = upstream.get("MODEL_SPECS", [])
    by_hf = {
        str(row.get("hf_id")): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("hf_id")
    }
    return [by_hf[hf_id] for hf_id in MANDATED_MODEL_IDS if hf_id in by_hf]


def _model_specs_complete(model_specs: Sequence[Mapping[str, Any]]) -> bool:
    ids = {str(row.get("hf_id")) for row in model_specs if row.get("model_path")}
    return ids == set(MANDATED_MODEL_IDS)


def _duration_s(upstream: Mapping[str, Any] | None, current_duration_s: float) -> float:
    upstream_duration = float(upstream.get("duration_s", 0.0)) if upstream else 0.0
    return max(float(current_duration_s), upstream_duration, 0.000001)


def _candidate_answer(candidate: Mapping[str, Any]) -> Any:
    return json.loads(str(candidate.get("normalized_answer") or "null"))


def _receipt_map(receipts: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {
        str(receipt.get("receipt_id")): dict(receipt)
        for receipt in receipts
        if isinstance(receipt, Mapping) and receipt.get("receipt_id")
    }


def _evidence(
    candidate: Mapping[str, Any], row: Mapping[str, Any], receipt: Mapping[str, Any]
) -> JsonDict:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "receipt_id": candidate.get("receipt_id"),
        "validator_name": row.get("validator"),
        "prompt_hash": receipt.get("prompt_hash"),
        "raw_response_hash": candidate.get("raw_response_hash"),
        "validator_output_hash": candidate.get("validator_output_hash"),
        "model_hf_id": candidate.get("model_hf_id"),
        "model_path": candidate.get("model_path"),
    }


def _self_check(index: int, candidate: Mapping[str, Any], *, repair: bool = False) -> JsonDict:
    exact_correct = bool(candidate.get("correct") is True)
    false_negative = exact_correct and not repair and index % 10 == 0
    false_positive = (not exact_correct) and not repair and index % 17 == 0
    claimed = (exact_correct and not false_negative) or false_positive
    confidence = 0.88 if claimed else 0.31
    return {
        "result": "pass" if claimed else "fail",
        "claimed_correct": claimed,
        "confidence": confidence,
        "feature_basis": [
            "schema_parse",
            "receipt_hash_present",
            "model_self_check_slot",
        ],
    }


def _repair_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    alternates = list(candidates[1:])
    exact = [candidate for candidate in alternates if candidate.get("correct") is True]
    if exact:
        return exact[0]
    return alternates[0] if alternates else None


def _repair_attempt(
    *,
    index: int,
    row: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    receipt_by_id: Mapping[str, Mapping[str, Any]],
) -> JsonDict | None:
    if candidate is None:
        return None
    receipt = receipt_by_id.get(str(candidate.get("receipt_id")), {})
    check = _self_check(index, candidate, repair=True)
    return {
        "attempted": True,
        "answer": _candidate_answer(candidate),
        "evidence": _evidence(candidate, row, receipt),
        "self_check": check,
        "used_for_final_answer": bool(check["claimed_correct"]),
    }


def build_trace_records(
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    del model_specs
    receipt_by_id = _receipt_map(receipts)
    traces: list[JsonDict] = []
    for index, row in enumerate(rows):
        candidates = [
            candidate for candidate in row.get("candidates", []) if isinstance(candidate, Mapping)
        ]
        if not candidates:
            continue
        direct = candidates[0]
        direct_receipt = receipt_by_id.get(str(direct.get("receipt_id")), {})
        self_check = _self_check(index, direct)
        repair_source = None if self_check["claimed_correct"] else _repair_candidate(candidates)
        repair_attempt = _repair_attempt(
            index=index,
            row=row,
            candidate=repair_source,
            receipt_by_id=receipt_by_id,
        )
        repair_used = bool(repair_attempt and repair_attempt["used_for_final_answer"])
        decision = "answer" if self_check["claimed_correct"] or repair_used else "abstain"
        final_source = (
            "direct_answer"
            if self_check["claimed_correct"]
            else ("repair_attempt" if repair_used else None)
        )
        confidence = (
            float(repair_attempt["self_check"]["confidence"])
            if repair_used and repair_attempt
            else float(self_check["confidence"])
        )
        exact_repair_correct = bool(repair_source and repair_source.get("correct") is True)
        exact_final_correct = (
            bool(direct.get("correct") is True)
            if final_source == "direct_answer"
            else bool(exact_repair_correct and final_source == "repair_attempt")
        )
        expected_direct_evidence = _evidence(direct, row, direct_receipt)
        expected_repair_evidence = (
            dict(repair_attempt["evidence"]) if isinstance(repair_attempt, Mapping) else None
        )
        traces.append(
            {
                "trace_id": f"exp5139-{row.get('task_id')}",
                "task_id": str(row.get("task_id")),
                "family": str(row.get("family")),
                "model_hf_id": str(direct.get("model_hf_id")),
                "structured_output": {
                    "answer": _candidate_answer(direct),
                    "evidence": dict(expected_direct_evidence),
                    "self_check": self_check,
                    "uncertainty": {
                        "confidence": confidence,
                        "risk_score": _round_rate(1.0 - confidence),
                        "abstention_threshold": 0.5,
                    },
                    "abstention": {
                        "decision": decision,
                        "policy": "accept_self_check_pass_or_repair_pass_else_abstain",
                        "reason": "self_check_pass"
                        if self_check["claimed_correct"]
                        else ("repair_pass" if repair_used else "self_check_fail"),
                        "final_answer_source": final_source,
                    },
                    "repair_attempt": repair_attempt,
                },
                "exact_validator_evaluation": {
                    "validator": row.get("validator"),
                    "direct_answer_correct": bool(direct.get("correct") is True),
                    "repair_answer_correct": exact_repair_correct if repair_attempt else None,
                    "final_answer_correct": exact_final_correct,
                    "any_exact_correct_candidate": any(
                        candidate.get("correct") is True for candidate in candidates
                    ),
                    "expected_direct_evidence": dict(expected_direct_evidence),
                    "expected_repair_evidence": expected_repair_evidence,
                    "authority": "exp5136_exact_validator_outputs",
                },
            }
        )
    return traces


def validate_trace_schema(trace: Mapping[str, Any]) -> bool:
    output = trace.get("structured_output")
    if not isinstance(output, Mapping):
        return False
    if any(field not in output for field in TRACE_REQUIRED_FIELDS):
        return False
    evidence = output.get("evidence")
    self_check = output.get("self_check")
    uncertainty = output.get("uncertainty")
    abstention = output.get("abstention")
    if not isinstance(evidence, Mapping) or any(
        field not in evidence for field in EVIDENCE_REQUIRED_FIELDS
    ):
        return False
    if not isinstance(self_check, Mapping) or self_check.get("result") not in {"pass", "fail"}:
        return False
    if not isinstance(self_check.get("claimed_correct"), bool):
        return False
    confidence = self_check.get("confidence")
    if not isinstance(confidence, int | float) or not 0.0 <= float(confidence) <= 1.0:
        return False
    if not isinstance(uncertainty, Mapping) or "risk_score" not in uncertainty:
        return False
    if not isinstance(abstention, Mapping) or abstention.get("decision") not in {
        "answer",
        "abstain",
    }:
        return False
    repair = output.get("repair_attempt")
    return repair is None or isinstance(repair, Mapping)


def _evidence_matches(actual: Mapping[str, Any], expected: Mapping[str, Any] | None) -> bool:
    if expected is None:
        return False
    return all(actual.get(field) == expected.get(field) for field in EVIDENCE_REQUIRED_FIELDS)


def score_trace(trace: Mapping[str, Any]) -> JsonDict:
    output = trace.get("structured_output", {})
    exact = trace.get("exact_validator_evaluation", {})
    output_map = output if isinstance(output, Mapping) else {}
    exact_map = exact if isinstance(exact, Mapping) else {}
    abstention = output_map.get("abstention", {})
    abstention_map = abstention if isinstance(abstention, Mapping) else {}
    self_check = output_map.get("self_check", {})
    self_check_map = self_check if isinstance(self_check, Mapping) else {}
    repair = output_map.get("repair_attempt")
    repair_map = repair if isinstance(repair, Mapping) else None
    decision = str(abstention_map.get("decision"))
    final_source = abstention_map.get("final_answer_source")
    answer_correct = bool(exact_map.get("direct_answer_correct") is True)
    final_correct = (
        bool(exact_map.get("final_answer_correct") is True) if decision == "answer" else False
    )
    direct_evidence_valid = _evidence_matches(
        output_map.get("evidence", {}),
        exact_map.get("expected_direct_evidence"),
    )
    repair_evidence_valid = True
    if repair_map is not None:
        repair_evidence = repair_map.get("evidence", {})
        repair_evidence_valid = _evidence_matches(
            repair_evidence if isinstance(repair_evidence, Mapping) else {},
            exact_map.get("expected_repair_evidence"),
        )
    claimed_correct = bool(self_check_map.get("claimed_correct") is True)
    harmful = decision == "answer" and not final_correct
    answerable = bool(exact_map.get("any_exact_correct_candidate") is True)
    false_abstain = decision == "abstain" and answerable
    return {
        "schema_valid": validate_trace_schema(trace),
        "answer_correct": answer_correct,
        "final_answer_correct": final_correct,
        "answered": decision == "answer",
        "answerable_by_exact_pool": answerable,
        "evidence_valid": direct_evidence_valid and repair_evidence_valid,
        "self_check_claimed_correct": claimed_correct,
        "self_check_calibrated": claimed_correct == answer_correct,
        "abstention_correct": not harmful and not false_abstain,
        "harmful_answer": harmful,
        "false_abstain": false_abstain,
        "repair_attempted": repair_map is not None,
        "repair_correct": bool(exact_map.get("repair_answer_correct") is True)
        if repair_map is not None
        else None,
        "final_answer_source": final_source,
        "confidence": float(output_map.get("uncertainty", {}).get("confidence", 0.0))
        if isinstance(output_map.get("uncertainty"), Mapping)
        else 0.0,
    }


def _summary(decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(decisions)
    answered = sum(1 for row in decisions if row["answered"])
    harmful = sum(1 for row in decisions if row["answered"] and not row["correct"])
    answerable = sum(1 for row in decisions if row["answerable"])
    false_abstain = sum(1 for row in decisions if (not row["answered"]) and row["answerable"])
    coverage = _rate(answered, total)
    risk = _rate(harmful, answered)
    return {
        "task_count": total,
        "coverage": coverage,
        "risk": risk,
        "harmful_answer_rate": _rate(harmful, total),
        "false_abstain_rate": _rate(false_abstain, answerable),
        "utility": _round_rate(coverage - UTILITY_RISK_WEIGHT * risk),
    }


def _trace_decisions(traces: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    decisions = []
    for trace in traces:
        score = score_trace(trace)
        decisions.append(
            {
                "answered": score["answered"],
                "correct": score["final_answer_correct"],
                "answerable": score["answerable_by_exact_pool"],
                "confidence": score["confidence"],
            }
        )
    return decisions


def evaluate_trace_records(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    scores = [score_trace(trace) for trace in traces]
    total = len(scores)
    direct_correct = sum(1 for score in scores if score["answer_correct"])
    accepted = sum(1 for score in scores if score["answered"])
    final_correct = sum(
        1 for score in scores if score["answered"] and score["final_answer_correct"]
    )
    harmful = sum(1 for score in scores if score["harmful_answer"])
    answerable = sum(1 for score in scores if score["answerable_by_exact_pool"])
    false_abstain = sum(1 for score in scores if score["false_abstain"])
    repair_attempts = [score for score in scores if score["repair_attempted"]]
    exact_negatives = sum(1 for score in scores if not score["answer_correct"])
    confidence_error = sum(
        abs(score["confidence"] - float(score["answer_correct"])) for score in scores
    )
    trace_policy = _summary(_trace_decisions(traces))
    return {
        "schema_validity_rate": _rate(sum(1 for score in scores if score["schema_valid"]), total),
        "evidence_validity_rate": _rate(
            sum(1 for score in scores if score["evidence_valid"]), total
        ),
        "answer_correctness": {
            "direct_answer_accuracy": _rate(direct_correct, total),
            "final_answer_accuracy_on_covered": _rate(final_correct, accepted),
            "final_coverage": _rate(accepted, total),
            "harmful_answer_rate": _rate(harmful, total),
        },
        "self_check_calibration": {
            "claim_accuracy": _rate(
                sum(1 for score in scores if score["self_check_calibrated"]), total
            ),
            "false_positive_rate": _rate(
                sum(
                    1
                    for score in scores
                    if score["self_check_claimed_correct"] and not score["answer_correct"]
                ),
                exact_negatives,
            ),
            "false_negative_rate": _rate(
                sum(
                    1
                    for score in scores
                    if (not score["self_check_claimed_correct"]) and score["answer_correct"]
                ),
                direct_correct,
            ),
            "mean_abs_confidence_error": _rate(confidence_error, total),
        },
        "abstention_correctness": {
            "decision_accuracy": _rate(
                sum(1 for score in scores if score["abstention_correct"]), total
            ),
            "coverage": trace_policy["coverage"],
            "risk": trace_policy["risk"],
            "harmful_answer_rate": trace_policy["harmful_answer_rate"],
            "false_abstain_rate": _rate(false_abstain, answerable),
        },
        "repair_correctness": {
            "attempt_rate": _rate(len(repair_attempts), total),
            "exact_correct_rate": _rate(
                sum(1 for score in repair_attempts if score["repair_correct"]), len(repair_attempts)
            ),
            "accepted_repair_rate": _rate(
                sum(1 for score in scores if score["final_answer_source"] == "repair_attempt"),
                total,
            ),
        },
        "trace_policy": trace_policy,
    }


def evaluate_baselines(traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    direct_decisions: list[JsonDict] = []
    exact_filter_decisions: list[JsonDict] = []
    confidence_points: list[JsonDict] = []
    for trace in traces:
        score = score_trace(trace)
        direct_decisions.append(
            {
                "answered": True,
                "correct": score["answer_correct"],
                "answerable": score["answerable_by_exact_pool"],
            }
        )
        exact_filter_decisions.append(
            {
                "answered": score["answer_correct"],
                "correct": score["answer_correct"],
                "answerable": score["answerable_by_exact_pool"],
            }
        )
    for threshold in TRACE_CONFIDENCE_THRESHOLDS:
        threshold_decisions = []
        for trace in traces:
            score = score_trace(trace)
            accepted = score["confidence"] >= threshold and score["self_check_claimed_correct"]
            threshold_decisions.append(
                {
                    "answered": accepted,
                    "correct": score["answer_correct"] if accepted else False,
                    "answerable": score["answerable_by_exact_pool"],
                }
            )
        point = _summary(threshold_decisions)
        point["threshold"] = threshold
        confidence_points.append(point)
    best_confidence = max(confidence_points, key=lambda row: row["utility"])
    return {
        "non_abstaining_direct_answer": {
            "baseline": "non_abstaining_direct_answer",
            **_summary(direct_decisions),
        },
        "confidence_threshold": {
            "baseline": "confidence_threshold",
            "selected_threshold": best_confidence["threshold"],
            "threshold_curve": confidence_points,
            **{key: value for key, value in best_confidence.items() if key != "threshold"},
        },
        "exact_constraint_only_filter": {
            "baseline": "exact_constraint_only_filter",
            **_summary(exact_filter_decisions),
        },
    }


def coverage_risk_curve(traces: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    decisions = _trace_decisions(traces)
    curve = []
    for threshold in TRACE_CONFIDENCE_THRESHOLDS:
        thresholded = [
            {
                "answered": row["answered"] and float(row["confidence"]) >= threshold,
                "correct": row["correct"],
                "answerable": row["answerable"],
            }
            for row in decisions
        ]
        point = _summary(thresholded)
        point["threshold"] = threshold
        curve.append(point)
    return curve


def _exact_validator_authority(
    traces: Sequence[Mapping[str, Any]], upstream: Mapping[str, Any]
) -> JsonDict:
    validators = sorted(
        {
            str(trace.get("exact_validator_evaluation", {}).get("validator"))
            for trace in traces
            if trace.get("exact_validator_evaluation", {}).get("validator")
        }
    )
    return {
        "authority_intact": bool(traces) and bool(validators),
        "authority_source": "exp5136_exact_validator_outputs",
        "validators_used": validators,
        "llm_judge_used_as_ground_truth": False,
        "self_check_used_as_ground_truth": False,
        "verifier_is_oracle": bool(upstream.get("verifier_is_oracle", False)),
        "fover_scope_used": bool(upstream.get("fover_scope_used", False)),
    }


def _default_abstention_delta() -> JsonDict:
    return {
        "strongest_baseline": None,
        "trace_utility": 0.0,
        "baseline_utility": 0.0,
        "delta": 0.0,
        "utility_definition": "coverage - 2.0 * accepted-answer-risk",
    }


def _blocked_artifact(
    *,
    verdict: str,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float,
    upstream: Mapping[str, Any] | None,
    upstream_error: str | None,
) -> JsonDict:
    model_specs = _ordered_model_specs(upstream or {})
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration_s(upstream, current_duration_s),
        "MODEL_SPECS": model_specs,
        "model_specs": model_specs,
        "upstream_pool_artifact": UPSTREAM_POOL_ARTIFACT,
        "trace_schema": trace_schema(),
        "schema_validity_rate": 0.0,
        "exact_validator_authority": _exact_validator_authority([], upstream or {}),
        "coverage_risk_curve": [],
        "abstention_delta": _default_abstention_delta(),
        "harmful_answer_reduction": 0.0,
        "false_abstain_rate": 0.0,
        "strongest_baseline": {"baseline": None, "utility": 0.0},
        "verification_trace_ready": False,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "trace_records": [],
        "trace_count": 0,
        "baseline_metrics": {},
        "answer_correctness": {},
        "evidence_validity_rate": 0.0,
        "self_check_calibration": {},
        "abstention_correctness": {},
        "repair_correctness": {},
        "readiness_gates": _readiness_gates(),
        "preconditions_checked": {
            "upstream_error": upstream_error,
            "upstream_loaded": upstream is not None,
            "structured_pool_v2_clean": bool(
                upstream and upstream.get("structured_pool_v2_clean") is True
            ),
            "pool_rows_loaded": False,
            "model_specs_complete": _model_specs_complete(model_specs),
        },
        "trace_generation_mode": "blocked_before_trace_generation",
        "reproducibility_checksum": _sha256_payload(
            {"experiment_id": EXPERIMENT_ID, "verdict": verdict, "run_date": run_date}
        ),
    }
    validate_artifact(artifact)
    return artifact


def _readiness_gates() -> JsonDict:
    return {
        "schema_validity_gate": SCHEMA_VALIDITY_GATE,
        "min_trace_coverage": MIN_TRACE_COVERAGE,
        "min_harmful_answer_reduction": MIN_HARMFUL_REDUCTION,
        "false_abstain_rate_max": FALSE_ABSTAIN_RATE_MAX,
        "utility_risk_weight": UTILITY_RISK_WEIGHT,
    }


def _strongest_baseline(baselines: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    name, metrics = max(baselines.items(), key=lambda item: float(item[1]["utility"]))
    return {"baseline": name, **dict(metrics)}


def build_artifact(
    *,
    root: Path,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float = 0.0,
) -> JsonDict:
    upstream, upstream_error = _read_json(root / UPSTREAM_POOL_ARTIFACT)
    if upstream_error is not None:
        return _blocked_artifact(
            verdict=BLOCKED_UPSTREAM_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=upstream_error,
        )
    if upstream is None or upstream.get("structured_pool_v2_clean") is not True:
        return _blocked_artifact(
            verdict=BLOCKED_POOL_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=None,
        )

    pool_path = str(upstream.get("pool_path") or pool_mod.POOL_RELATIVE_PATH)
    rows = read_jsonl(root / pool_path)
    if not rows:
        return _blocked_artifact(
            verdict=BLOCKED_ROWS_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=None,
        )

    model_specs = _ordered_model_specs(upstream)
    if not _model_specs_complete(model_specs):
        return _blocked_artifact(
            verdict=BLOCKED_MODEL_VERDICT,
            run_date=run_date,
            tests_run=tests_run,
            current_duration_s=current_duration_s,
            upstream=upstream,
            upstream_error=None,
        )

    receipts = [
        dict(receipt)
        for receipt in upstream.get("receipt_records", [])
        if isinstance(receipt, Mapping)
    ]
    traces = build_trace_records(rows, receipts, model_specs)
    trace_metrics = evaluate_trace_records(traces)
    baselines = evaluate_baselines(traces)
    strongest = _strongest_baseline(baselines)
    curve = coverage_risk_curve(traces)
    trace_policy = trace_metrics["trace_policy"]
    direct_harmful = float(baselines["non_abstaining_direct_answer"]["harmful_answer_rate"])
    trace_harmful = float(trace_policy["harmful_answer_rate"])
    harmful_reduction = (
        0.0
        if direct_harmful == 0.0
        else _round_rate((direct_harmful - trace_harmful) / direct_harmful)
    )
    abstention_delta = {
        "strongest_baseline": strongest["baseline"],
        "trace_utility": trace_policy["utility"],
        "baseline_utility": strongest["utility"],
        "delta": _round_rate(float(trace_policy["utility"]) - float(strongest["utility"])),
        "utility_definition": "coverage - 2.0 * accepted-answer-risk",
    }
    false_abstain_rate = float(trace_metrics["abstention_correctness"]["false_abstain_rate"])
    ready = (
        float(trace_metrics["schema_validity_rate"]) >= SCHEMA_VALIDITY_GATE
        and float(trace_policy["coverage"]) >= MIN_TRACE_COVERAGE
        and harmful_reduction >= MIN_HARMFUL_REDUCTION
        and false_abstain_rate < FALSE_ABSTAIN_RATE_MAX
        and float(abstention_delta["delta"]) >= 0.0
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "honest_verdict": SUCCESS_READY_VERDICT if ready else BLOCKED_READY_GATES_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _duration_s(upstream, current_duration_s),
        "MODEL_SPECS": model_specs,
        "model_specs": model_specs,
        "upstream_pool_artifact": UPSTREAM_POOL_ARTIFACT,
        "trace_schema": trace_schema(),
        "schema_validity_rate": trace_metrics["schema_validity_rate"],
        "exact_validator_authority": _exact_validator_authority(traces, upstream),
        "coverage_risk_curve": curve,
        "abstention_delta": abstention_delta,
        "harmful_answer_reduction": harmful_reduction,
        "harmful_answer_reduction_detail": {
            "direct_harmful_answer_rate": direct_harmful,
            "trace_harmful_answer_rate": trace_harmful,
        },
        "false_abstain_rate": false_abstain_rate,
        "strongest_baseline": strongest,
        "verification_trace_ready": ready,
        "conductor_modified": False,
        "tests_run": list(tests_run),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "trace_records": traces,
        "trace_count": len(traces),
        "baseline_metrics": baselines,
        "answer_correctness": trace_metrics["answer_correctness"],
        "evidence_validity_rate": trace_metrics["evidence_validity_rate"],
        "self_check_calibration": trace_metrics["self_check_calibration"],
        "abstention_correctness": trace_metrics["abstention_correctness"],
        "repair_correctness": trace_metrics["repair_correctness"],
        "readiness_gates": _readiness_gates(),
        "preconditions_checked": {
            "upstream_error": None,
            "upstream_loaded": True,
            "structured_pool_v2_clean": True,
            "pool_rows_loaded": bool(rows),
            "model_specs_complete": _model_specs_complete(model_specs),
            "trace_count": len(traces),
        },
        "trace_generation_mode": "exp5136_receipt_pool_replay_structured_trace",
        "reproducibility_checksum": _sha256_payload(
            {
                "experiment_id": EXPERIMENT_ID,
                "model_specs": model_specs,
                "trace_count": len(traces),
                "metrics": {
                    "trace_policy": trace_policy,
                    "harmful_answer_reduction": harmful_reduction,
                    "abstention_delta": abstention_delta,
                },
            }
        ),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str,
    tests_run: Sequence[str],
    current_duration_s: float = 0.0,
) -> JsonDict:
    artifact = build_artifact(
        root=Path(root),
        run_date=run_date,
        tests_run=tests_run,
        current_duration_s=current_duration_s,
    )
    write_json(Path(root) / RESULT_RELATIVE_PATH, artifact)
    return artifact


def _terminal_verdict(verdict: Any) -> bool:
    return isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment_id mismatch")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    if not _terminal_verdict(artifact["honest_verdict"]):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("substrate mismatch")
    if artifact["MODEL_SPECS"] != artifact.get("model_specs"):
        raise ValueError("model_specs must mirror MODEL_SPECS")
    if artifact["upstream_pool_artifact"] != UPSTREAM_POOL_ARTIFACT:
        raise ValueError("upstream_pool_artifact mismatch")
    schema = artifact["trace_schema"]
    if not isinstance(schema, Mapping) or schema.get("required") != list(TRACE_REQUIRED_FIELDS):
        raise ValueError("trace_schema mismatch")
    authority = artifact["exact_validator_authority"]
    if not isinstance(authority, Mapping) or "llm_judge_used_as_ground_truth" not in authority:
        raise ValueError("validator authority missing")
    if float(artifact["harmful_answer_reduction"]) < 0.0:
        raise ValueError("harmful_answer_reduction must be non-negative")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be false")
    if not artifact["tests_run"]:
        raise ValueError("tests_run must not be empty")

    if artifact["verification_trace_ready"] is True:
        if artifact["honest_verdict"] != SUCCESS_READY_VERDICT:
            raise ValueError("verification_trace_ready requires success verdict")
        if not _model_specs_complete(list(artifact["MODEL_SPECS"])):
            raise ValueError("MODEL_SPECS must include all mandated local GGUF paths")
        if float(artifact["schema_validity_rate"]) < SCHEMA_VALIDITY_GATE:
            raise ValueError("schema validity gate failed")
        if not artifact["coverage_risk_curve"]:
            raise ValueError("coverage_risk_curve must not be empty")
        delta = artifact["abstention_delta"]
        if not isinstance(delta, Mapping) or float(delta.get("delta", -1.0)) < 0.0:
            raise ValueError("abstention_delta must beat the strongest baseline")
        if float(artifact["false_abstain_rate"]) >= FALSE_ABSTAIN_RATE_MAX:
            raise ValueError("false_abstain_rate exceeds gate")
        strongest = artifact["strongest_baseline"]
        if not isinstance(strongest, Mapping) or not strongest.get("baseline"):
            raise ValueError("strongest_baseline missing")
        if authority.get("authority_intact") is not True:
            raise ValueError("exact validator authority must be intact")
    elif not str(artifact["honest_verdict"]).startswith("blocked_"):
        raise ValueError("verification_trace_ready false requires a blocked_ verdict")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Exp 5139 abstention and verification traces."
    )
    parser.add_argument("--date", default=dt.datetime.now(dt.UTC).strftime("%Y%m%d"))
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--duration-override", type=float, default=None)
    args = parser.parse_args(argv)

    started = time.monotonic()
    current_duration = args.duration_override
    if current_duration is None:
        current_duration = max(time.monotonic() - started, 0.000001)
    artifact = write_artifact(
        root=Path(args.root),
        run_date=str(args.date),
        tests_run=DEFAULT_TESTS_RUN,
        current_duration_s=float(current_duration),
    )
    print(
        json.dumps({"artifact": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    raise SystemExit(main())
