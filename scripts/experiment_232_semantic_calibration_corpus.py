#!/usr/bin/env python3
"""Experiment 232: semantic calibration corpus from live semantic artifacts.

Writes:
- ``data/research/semantic_calibration_corpus_232.jsonl``
- ``results/experiment_232_results.json``

Spec: REQ-VERIFY-042, REQ-VERIFY-043,
SCENARIO-VERIFY-043, SCENARIO-VERIFY-044
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

RUN_DATE = "20260413"
REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = Path("data/research/semantic_calibration_corpus_232.jsonl")
RESULTS_PATH = Path("results/experiment_232_results.json")
SOURCE_ARTIFACTS = (
    Path("results/experiment_219_results.json"),
    Path("results/experiment_221_results.json"),
)
ALL_OUTCOMES = ("true_positive", "false_positive", "false_negative", "true_negative")

FOLLOW_UP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "example_id": "exp232-followup-instruction-surface-only-fp-1",
        "case_id": "exp211-instruction-json-3",
        "response_mode": "structured_json",
        "response": '{"channel":"ops","status":"green","escalation":"no"}',
        "actual_error": False,
        "verifier_detected": True,
        "gold_violation_family": "none",
        "evidence_violation_family": "literal",
        "answer_target_alignment": "aligned",
        "premise_coverage": "complete",
        "claim_granularity": "structured_constraint_bundle",
        "signal_type": "prompt",
        "signal_features": {
            "evidence_count": 1.0,
            "total_constraints": 3.0,
            "semantic_violation_count": 0.0,
            "constraint_coverage": 1.0,
            "answer_alignment_penalty": 0.0,
            "premise_gap_penalty": 0.0,
            "structured_bonus": 0.1,
        },
    },
    {
        "example_id": "exp232-followup-instruction-surface-only-fn-1",
        "case_id": "exp211-instruction-two-sentence-1",
        "response_mode": "answer_only_terse",
        "response": "Status green. Channel ops. Escalation no.",
        "actual_error": True,
        "verifier_detected": False,
        "gold_violation_family": "literal",
        "evidence_violation_family": "none",
        "answer_target_alignment": "aligned_format_gap",
        "premise_coverage": "complete",
        "claim_granularity": "constraint_bundle",
        "signal_type": "prompt",
        "signal_features": {
            "evidence_count": 0.0,
            "total_constraints": 2.0,
            "semantic_violation_count": 0.0,
            "constraint_coverage": 1.0,
            "answer_alignment_penalty": 0.15,
            "premise_gap_penalty": 0.0,
            "structured_bonus": 0.0,
        },
    },
    {
        "example_id": "exp232-followup-instruction-grounded-fp-1",
        "case_id": "exp211-instruction-grounded-3",
        "response_mode": "answer_only_terse",
        "response": "S1, S4",
        "actual_error": False,
        "verifier_detected": True,
        "gold_violation_family": "none",
        "evidence_violation_family": "semantic",
        "answer_target_alignment": "aligned",
        "premise_coverage": "complete",
        "claim_granularity": "constraint_bundle",
        "signal_type": "prompt",
        "signal_features": {
            "evidence_count": 1.0,
            "total_constraints": 3.0,
            "semantic_violation_count": 1.0,
            "constraint_coverage": 1.0,
            "answer_alignment_penalty": 0.0,
            "premise_gap_penalty": 0.0,
            "structured_bonus": 0.0,
        },
    },
    {
        "example_id": "exp232-followup-instruction-grounded-fn-1",
        "case_id": "exp211-instruction-grounded-3",
        "response_mode": "answer_only_terse",
        "response": "S1,S3",
        "actual_error": True,
        "verifier_detected": False,
        "gold_violation_family": "semantic",
        "evidence_violation_family": "none",
        "answer_target_alignment": "misaligned",
        "premise_coverage": "partial",
        "claim_granularity": "constraint_bundle",
        "signal_type": "prompt",
        "signal_features": {
            "evidence_count": 0.0,
            "total_constraints": 3.0,
            "semantic_violation_count": 0.0,
            "constraint_coverage": 1.0,
            "answer_alignment_penalty": 0.8,
            "premise_gap_penalty": 0.2,
            "structured_bonus": 0.0,
        },
    },
    {
        "example_id": "exp232-followup-code-typed-properties-fp-1",
        "case_id": "exp211-code-dedupe-1",
        "response_mode": "code_only",
        "response": (
            "def dedupe_preserve_order(items: list[str]) -> list[str]:\n"
            "    return list(dict.fromkeys(items))"
        ),
        "actual_error": False,
        "verifier_detected": True,
        "gold_violation_family": "none",
        "evidence_violation_family": "search_optimization_limited",
        "answer_target_alignment": "aligned",
        "premise_coverage": "complete",
        "claim_granularity": "code_property_bundle",
        "signal_type": "prompt",
        "signal_features": {
            "evidence_count": 1.0,
            "total_constraints": 5.0,
            "semantic_violation_count": 0.0,
            "constraint_coverage": 1.0,
            "answer_alignment_penalty": 0.0,
            "premise_gap_penalty": 0.0,
            "structured_bonus": 0.05,
        },
    },
    {
        "example_id": "exp232-followup-code-typed-properties-fn-1",
        "case_id": "exp211-code-dedupe-1",
        "response_mode": "code_only",
        "response": (
            "def dedupe_preserve_order(items: list[str]) -> list[str]:\n"
            "    return sorted(set(items))"
        ),
        "actual_error": True,
        "verifier_detected": False,
        "gold_violation_family": "search_optimization_limited",
        "evidence_violation_family": "none",
        "answer_target_alignment": "partially_aligned",
        "premise_coverage": "complete",
        "claim_granularity": "code_property_bundle",
        "signal_type": "prompt",
        "signal_features": {
            "evidence_count": 0.0,
            "total_constraints": 5.0,
            "semantic_violation_count": 0.0,
            "constraint_coverage": 1.0,
            "answer_alignment_penalty": 0.45,
            "premise_gap_penalty": 0.0,
            "structured_bonus": 0.05,
        },
    },
)


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return REPO_ROOT


def resolve_path(repo_root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else repo_root / candidate


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(content, encoding="utf-8")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def outcome_bucket(actual_error: bool, verifier_detected: bool) -> str:
    if actual_error and verifier_detected:
        return "true_positive"
    if (not actual_error) and verifier_detected:
        return "false_positive"
    if actual_error and (not verifier_detected):
        return "false_negative"
    return "true_negative"


def repairability_hint(actual_error: bool, verifier_detected: bool, repaired: bool) -> str:
    if (not actual_error) and (not verifier_detected):
        return "no_repair_needed"
    if (not actual_error) and verifier_detected:
        return "tighten_threshold"
    if actual_error and (not verifier_detected):
        return "detect_before_repair"
    if repaired:
        return "repair_from_live_history"
    return "needs_retrieval_upgrade"


def domain_for_task_slice(task_slice: str) -> str:
    if task_slice == "code_typed_properties":
        return "code"
    if task_slice.startswith("instruction_"):
        return "instruction_following"
    return "word_problem"


def primary_219_evidence_family(violations: list[dict[str, Any]]) -> str:
    families: Counter[str] = Counter()
    for violation in violations:
        family = str(violation.get("metadata", {}).get("taxonomy_hint") or "unknown")
        families[family] += 1
    if not families:
        return "none"
    return families.most_common(1)[0][0]


def primary_221_family(constraint_results: list[dict[str, Any]]) -> str:
    families: Counter[str] = Counter()
    for result in constraint_results:
        if result.get("status") != "violated":
            continue
        family = str(result.get("family") or "unknown")
        families[family] += 1
    if not families:
        return "none"
    for preferred in ("semantic", "search_optimization_limited", "literal"):
        if preferred in families:
            return preferred
    return families.most_common(1)[0][0]


def answer_target_alignment_219(
    actual_error: bool,
    evidence_family: str,
    violations: list[dict[str, Any]],
) -> str:
    if not actual_error:
        return "aligned"
    violation_types = {str(item.get("violation_type") or "") for item in violations}
    if "answer_target_mismatch" in violation_types or evidence_family == "question_grounding_failures":
        return "misaligned"
    if evidence_family in {"entity_quantity_binding_errors", "unit_aggregation_errors"}:
        return "partially_aligned"
    if evidence_family == "omitted_premises":
        return "partially_aligned"
    return "uncertain"


def premise_coverage_219(
    actual_error: bool,
    prompt_clause_count: int,
    claim_count: int,
    violations: list[dict[str, Any]],
) -> str:
    if not actual_error:
        return "complete"
    missing_count = sum(
        1
        for item in violations
        if str(item.get("violation_type") or "").startswith("missing_")
    )
    if missing_count == 0 and claim_count > 0:
        return "complete"
    if missing_count >= max(1, prompt_clause_count // 2) or claim_count == 0:
        return "missing"
    return "partial"


def claim_granularity_219(case: dict[str, Any], semantic: dict[str, Any]) -> str:
    claims = semantic.get("claims", [])
    parse_status = str(case.get("typed_reasoning_parse_status") or "")
    response_mode = str(case.get("response_mode") or "")
    if not claims:
        return "no_claim_signal"
    if parse_status == "direct_json" or response_mode == "structured_json":
        return "structured_claim_trace"
    if len(claims) <= 1:
        return "final_answer_only"
    return "multi_claim_trace"


def score_semantic_row(
    *,
    prompt_clause_count: int,
    claim_count: int,
    evidence_count: int,
    answer_alignment_penalty: float,
    premise_gap_penalty: float,
    structured_bonus: float,
) -> tuple[float, dict[str, float]]:
    evidence_density = evidence_count / max(prompt_clause_count, 1)
    claim_bonus = 0.1 if claim_count > 1 else 0.0
    score = clamp(
        0.15 * evidence_density
        + 0.4 * answer_alignment_penalty
        + 0.25 * premise_gap_penalty
        + claim_bonus
        + structured_bonus
    )
    components = {
        "evidence_count": float(evidence_count),
        "prompt_clause_count": float(prompt_clause_count),
        "claim_count": float(claim_count),
        "evidence_density": round(evidence_density, 6),
        "answer_alignment_penalty": answer_alignment_penalty,
        "premise_gap_penalty": premise_gap_penalty,
        "structured_bonus": structured_bonus,
    }
    return round(score, 6), components


def answer_target_alignment_221(
    *,
    actual_error: bool,
    gold_family: str,
    semantic_violation_count: int,
) -> str:
    if not actual_error:
        return "aligned"
    if gold_family == "semantic" or semantic_violation_count > 0:
        return "misaligned"
    if gold_family == "search_optimization_limited":
        return "partially_aligned"
    if gold_family == "literal":
        return "aligned_format_gap"
    return "uncertain"


def premise_coverage_221(
    *,
    actual_error: bool,
    constraint_coverage: float,
    gold_family: str,
    semantic_violation_count: int,
) -> str:
    if not actual_error:
        return "complete"
    if constraint_coverage < 0.75:
        return "missing"
    if constraint_coverage < 1.0 or semantic_violation_count > 0:
        return "partial"
    if gold_family == "semantic":
        return "partial"
    return "complete"


def claim_granularity_221(case: dict[str, Any], task_slice: str) -> str:
    if task_slice == "code_typed_properties":
        return "code_property_bundle"
    if str(case.get("output_style") or "") == "structured_json":
        return "structured_constraint_bundle"
    return "constraint_bundle"


def score_prompt_row(
    *,
    evidence_count: float,
    total_constraints: float,
    semantic_violation_count: float,
    constraint_coverage: float,
    answer_alignment_penalty: float,
    premise_gap_penalty: float,
    structured_bonus: float,
) -> tuple[float, dict[str, float]]:
    violation_fraction = evidence_count / max(total_constraints, 1.0)
    semantic_fraction = semantic_violation_count / max(total_constraints, 1.0)
    score = clamp(
        0.55 * violation_fraction
        + 0.2 * semantic_fraction
        + 0.15 * (1.0 - constraint_coverage)
        + 0.05 * answer_alignment_penalty
        + 0.03 * premise_gap_penalty
        + structured_bonus
    )
    components = {
        "evidence_count": evidence_count,
        "total_constraints": total_constraints,
        "semantic_violation_count": semantic_violation_count,
        "constraint_coverage": round(constraint_coverage, 6),
        "violation_fraction": round(violation_fraction, 6),
        "semantic_fraction": round(semantic_fraction, 6),
        "answer_alignment_penalty": answer_alignment_penalty,
        "premise_gap_penalty": premise_gap_penalty,
        "structured_bonus": structured_bonus,
    }
    return round(score, 6), components


def build_row(
    *,
    example_id: str,
    source_type: str,
    source_artifact: str,
    source_refs: list[str],
    benchmark: str,
    task_slice: str,
    domain: str,
    model_name: str,
    response_mode: str,
    prompt: str,
    response: str,
    actual_error: bool,
    verifier_detected: bool,
    gold_violation_family: str,
    evidence_violation_family: str,
    answer_target_alignment: str,
    premise_coverage: str,
    claim_granularity: str,
    repaired: bool,
    score: float,
    score_components: dict[str, float],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "source_type": source_type,
        "source_artifact": source_artifact,
        "source_refs": source_refs,
        "benchmark": benchmark,
        "task_slice": task_slice,
        "domain": domain,
        "model_name": model_name,
        "response_mode": response_mode,
        "prompt": prompt,
        "response": response,
        "labels": {
            "actual_error": actual_error,
            "verifier_detected": verifier_detected,
            "target_label": int(actual_error),
            "predicted_label": int(verifier_detected),
            "outcome_bucket": outcome_bucket(actual_error, verifier_detected),
            "gold_violation_family": gold_violation_family,
            "evidence_violation_family": evidence_violation_family,
            "answer_target_alignment": answer_target_alignment,
            "premise_coverage": premise_coverage,
            "claim_granularity": claim_granularity,
            "repairability_hint": repairability_hint(actual_error, verifier_detected, repaired),
            "repair_observed": repaired,
        },
        "calibration": {
            "score": score,
            "score_direction": "higher_is_more_likely_error",
            "score_components": score_components,
        },
        "provenance": provenance,
    }


def build_exp219_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cohort_cases = {str(case["case_id"]): case for case in payload.get("cohort", {}).get("cases", [])}
    repair_map: dict[tuple[str, str], bool] = {}
    for run in payload.get("paired_runs", []):
        if run.get("mode") != "verify_repair":
            continue
        model_name = str(run.get("model_name") or "")
        for case in run.get("cases", []):
            repair_map[(model_name, str(case.get("case_id") or ""))] = bool(case.get("repaired"))

    rows: list[dict[str, Any]] = []
    for run in payload.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_name = str(run.get("model_name") or "")
        model_slug = slugify(model_name)
        for case in run.get("cases", []):
            case_id = str(case.get("case_id") or "")
            cohort = cohort_cases[case_id]
            semantic = case.get("verification", {}).get("semantic_grounding", {})
            violations = semantic.get("violations", [])
            evidence_family = primary_219_evidence_family(violations)
            actual_error = not bool(case.get("correct"))
            verifier_detected = bool(case.get("flagged"))
            gold_family = evidence_family if actual_error and evidence_family != "none" else (
                "unlabeled_live_error" if actual_error else "none"
            )
            prompt_clause_count = len(semantic.get("question_profile", {}).get("prompt_clauses", []))
            claim_count = len(semantic.get("claims", []))
            answer_target_alignment = answer_target_alignment_219(
                actual_error=actual_error,
                evidence_family=evidence_family,
                violations=violations,
            )
            premise_coverage = premise_coverage_219(
                actual_error=actual_error,
                prompt_clause_count=prompt_clause_count,
                claim_count=claim_count,
                violations=violations,
            )
            structured_bonus = (
                0.1
                if str(case.get("typed_reasoning_parse_status") or "") == "direct_json"
                or str(case.get("response_mode") or "") == "structured_json"
                else 0.0
            )
            alignment_penalty = {
                "aligned": 0.0,
                "partially_aligned": 0.5,
                "misaligned": 1.0,
                "aligned_format_gap": 0.15,
                "uncertain": 0.2,
            }[answer_target_alignment]
            premise_penalty = {
                "complete": 0.0,
                "partial": 0.5,
                "missing": 1.0,
                "uncertain": 0.2,
            }[premise_coverage]
            score, score_components = score_semantic_row(
                prompt_clause_count=prompt_clause_count,
                claim_count=claim_count,
                evidence_count=len(violations),
                answer_alignment_penalty=alignment_penalty,
                premise_gap_penalty=premise_penalty,
                structured_bonus=structured_bonus,
            )
            rows.append(
                build_row(
                    example_id=f"exp232-live-219-{model_slug}-{case_id}",
                    source_type="live_artifact",
                    source_artifact="exp219_live",
                    source_refs=[f"exp219:{case_id}"],
                    benchmark="gsm8k_semantic",
                    task_slice=str(cohort.get("task_slice") or "live_gsm8k_semantic_failure"),
                    domain="word_problem",
                    model_name=model_name,
                    response_mode=str(case.get("response_mode") or "unknown"),
                    prompt=str(cohort.get("question") or ""),
                    response=str(case.get("response") or ""),
                    actual_error=actual_error,
                    verifier_detected=verifier_detected,
                    gold_violation_family=gold_family,
                    evidence_violation_family=evidence_family,
                    answer_target_alignment=answer_target_alignment,
                    premise_coverage=premise_coverage,
                    claim_granularity=claim_granularity_219(case, semantic),
                    repaired=repair_map.get((model_name, case_id), False),
                    score=score,
                    score_components=score_components,
                    provenance={
                        "calibration_artifact_run_date": RUN_DATE,
                        "source_run_date": str(payload.get("run_date") or ""),
                        "source_experiment": 219,
                        "source_case_id": case_id,
                        "sample_position": int(cohort.get("sample_position") or 0),
                        "dataset_idx": int(cohort.get("dataset_idx") or 0),
                        "prompt_seed": int(cohort.get("prompt_seeds", {}).get("verify_only") or 0),
                        "paired_mode": "verify_only",
                        "follow_up_gap": None,
                    },
                )
            )
    return rows


def build_exp221_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cohort_cases = {str(case["case_id"]): case for case in payload.get("cohort", {}).get("cases", [])}
    repair_map: dict[tuple[str, str], bool] = {}
    for run in payload.get("paired_runs", []):
        if run.get("mode") != "verify_repair":
            continue
        model_name = str(run.get("model_name") or "")
        for case in run.get("cases", []):
            repair_map[(model_name, str(case.get("case_id") or ""))] = bool(case.get("repaired"))

    rows: list[dict[str, Any]] = []
    for run in payload.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_name = str(run.get("model_name") or "")
        model_slug = slugify(model_name)
        for case in run.get("cases", []):
            case_id = str(case.get("case_id") or "")
            cohort = cohort_cases[case_id]
            evaluation = case.get("evaluation", {})
            constraint_results = evaluation.get("constraint_results", [])
            gold_family = primary_221_family(constraint_results)
            actual_error = not bool(case.get("exact_satisfaction"))
            verifier_detected = bool(case.get("flagged"))
            evidence_family = gold_family if verifier_detected else "none"
            task_slice = str(evaluation.get("task_slice") or cohort.get("task_slice") or "")
            semantic_violation_count = int(case.get("semantic_violation_count") or 0)
            answer_target_alignment = answer_target_alignment_221(
                actual_error=actual_error,
                gold_family=gold_family,
                semantic_violation_count=semantic_violation_count,
            )
            constraint_coverage = float(
                evaluation.get("constraint_extraction_coverage")
                or case.get("constraint_extraction_coverage")
                or 0.0
            )
            premise_coverage = premise_coverage_221(
                actual_error=actual_error,
                constraint_coverage=constraint_coverage,
                gold_family=gold_family,
                semantic_violation_count=semantic_violation_count,
            )
            alignment_penalty = {
                "aligned": 0.0,
                "partially_aligned": 0.5,
                "misaligned": 1.0,
                "aligned_format_gap": 0.15,
                "uncertain": 0.2,
            }[answer_target_alignment]
            premise_penalty = {
                "complete": 0.0,
                "partial": 0.5,
                "missing": 1.0,
                "uncertain": 0.2,
            }[premise_coverage]
            structured_bonus = (
                0.1
                if str(case.get("output_style") or "") == "structured_json"
                else 0.05
                if task_slice == "code_typed_properties"
                else 0.0
            )
            score, score_components = score_prompt_row(
                evidence_count=float(sum(item.get("status") == "violated" for item in constraint_results)),
                total_constraints=float(len(constraint_results)),
                semantic_violation_count=float(semantic_violation_count),
                constraint_coverage=constraint_coverage,
                answer_alignment_penalty=alignment_penalty,
                premise_gap_penalty=premise_penalty,
                structured_bonus=structured_bonus,
            )
            rows.append(
                build_row(
                    example_id=f"exp232-live-221-{model_slug}-{case_id}",
                    source_type="live_artifact",
                    source_artifact="exp221_live",
                    source_refs=list(cohort.get("source_refs") or []),
                    benchmark="constraint_ir",
                    task_slice=task_slice,
                    domain=domain_for_task_slice(task_slice),
                    model_name=model_name,
                    response_mode=str(case.get("response_mode") or "unknown"),
                    prompt=str(cohort.get("prompt") or ""),
                    response=str(case.get("response") or ""),
                    actual_error=actual_error,
                    verifier_detected=verifier_detected,
                    gold_violation_family=gold_family if actual_error else "none",
                    evidence_violation_family=evidence_family,
                    answer_target_alignment=answer_target_alignment,
                    premise_coverage=premise_coverage,
                    claim_granularity=claim_granularity_221(case, task_slice),
                    repaired=repair_map.get((model_name, case_id), False),
                    score=score,
                    score_components=score_components,
                    provenance={
                        "calibration_artifact_run_date": RUN_DATE,
                        "source_run_date": str(payload.get("run_date") or ""),
                        "source_experiment": 221,
                        "source_case_id": case_id,
                        "sample_position": int(cohort.get("sample_position") or 0),
                        "dataset_idx": int(cohort.get("dataset_idx") or 0),
                        "prompt_seed": int(cohort.get("prompt_seeds", {}).get("verify_only") or 0),
                        "paired_mode": "verify_only",
                        "follow_up_gap": None,
                    },
                )
            )
    return rows


def build_live_calibration_rows(repo_root: Path) -> list[dict[str, Any]]:
    exp219 = load_json(resolve_path(repo_root, SOURCE_ARTIFACTS[0]))
    exp221 = load_json(resolve_path(repo_root, SOURCE_ARTIFACTS[1]))
    return build_exp219_rows(exp219) + build_exp221_rows(exp221)


def build_follow_up_rows(repo_root: Path) -> list[dict[str, Any]]:
    exp221 = load_json(resolve_path(repo_root, SOURCE_ARTIFACTS[1]))
    cohort_cases = {str(case["case_id"]): case for case in exp221.get("cohort", {}).get("cases", [])}
    rows: list[dict[str, Any]] = []
    for spec in FOLLOW_UP_SPECS:
        cohort = cohort_cases[str(spec["case_id"])]
        task_slice = str(cohort.get("task_slice") or "")
        score, score_components = score_prompt_row(**spec["signal_features"])
        rows.append(
            build_row(
                example_id=str(spec["example_id"]),
                source_type="targeted_follow_up",
                source_artifact="exp232_followup",
                source_refs=[f"exp221:{spec['case_id']}"],
                benchmark="constraint_ir",
                task_slice=task_slice,
                domain=domain_for_task_slice(task_slice),
                model_name="follow_up_gap_fill",
                response_mode=str(spec["response_mode"]),
                prompt=str(cohort.get("prompt") or ""),
                response=str(spec["response"]),
                actual_error=bool(spec["actual_error"]),
                verifier_detected=bool(spec["verifier_detected"]),
                gold_violation_family=str(spec["gold_violation_family"]),
                evidence_violation_family=str(spec["evidence_violation_family"]),
                answer_target_alignment=str(spec["answer_target_alignment"]),
                premise_coverage=str(spec["premise_coverage"]),
                claim_granularity=str(spec["claim_granularity"]),
                repaired=False,
                score=score,
                score_components=score_components,
                provenance={
                    "calibration_artifact_run_date": RUN_DATE,
                    "source_run_date": str(exp221.get("run_date") or ""),
                    "source_experiment": 221,
                    "source_case_id": str(spec["case_id"]),
                    "sample_position": int(cohort.get("sample_position") or 0),
                    "dataset_idx": int(cohort.get("dataset_idx") or 0),
                    "prompt_seed": int(cohort.get("prompt_seeds", {}).get("verify_only") or 0),
                    "paired_mode": "gap_follow_up",
                    "follow_up_gap": outcome_bucket(
                        bool(spec["actual_error"]),
                        bool(spec["verifier_detected"]),
                    ),
                    "reference_source_refs": list(cohort.get("source_refs") or []),
                },
            )
        )
    return rows


def build_corpus(repo_root: Path) -> list[dict[str, Any]]:
    return build_live_calibration_rows(repo_root) + build_follow_up_rows(repo_root)


def counter_dict(rows: list[dict[str, Any]], key_fn: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(key_fn(row))] += 1
    return dict(counter)


def build_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    semantic_live = [
        row
        for row in rows
        if row["source_type"] == "live_artifact" and row["source_artifact"] == "exp219_live"
    ]
    prompt_live = [
        row
        for row in rows
        if row["source_type"] == "live_artifact" and row["source_artifact"] == "exp221_live"
    ]
    follow_ups = [row for row in rows if row["source_type"] == "targeted_follow_up"]
    prompt_live_outcomes = {row["labels"]["outcome_bucket"] for row in prompt_live}
    missing_prompt_outcomes = [
        bucket for bucket in ALL_OUTCOMES if bucket not in prompt_live_outcomes
    ]
    follow_up_outcomes = {row["labels"]["outcome_bucket"] for row in follow_ups}
    scores = [float(row["calibration"]["score"]) for row in rows]
    return {
        "experiment": "Exp 232",
        "run_date": RUN_DATE,
        "title": "Semantic calibration corpus from live semantic and prompt-side artifacts",
        "metadata": {
            "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS],
            "corpus_path": str(CORPUS_PATH),
            "output_path": str(RESULTS_PATH),
        },
        "summary": {
            "n_examples": len(rows),
            "by_source_type": counter_dict(rows, lambda row: row["source_type"]),
            "by_source_artifact": counter_dict(rows, lambda row: row["source_artifact"]),
            "by_benchmark": counter_dict(rows, lambda row: row["benchmark"]),
            "by_task_slice": counter_dict(rows, lambda row: row["task_slice"]),
            "by_model": counter_dict(rows, lambda row: row["model_name"]),
            "by_outcome_bucket": counter_dict(
                rows, lambda row: row["labels"]["outcome_bucket"]
            ),
            "by_gold_violation_family": counter_dict(
                rows, lambda row: row["labels"]["gold_violation_family"]
            ),
            "score_range": {
                "min": round(min(scores), 6),
                "max": round(max(scores), 6),
                "mean": round(mean(scores), 6),
            },
            "coverage_checks": {
                "semantic_live_has_all_outcomes": {
                    row["labels"]["outcome_bucket"] for row in semantic_live
                }
                == set(ALL_OUTCOMES),
                "prompt_side_live_missing_outcomes": missing_prompt_outcomes,
                "follow_ups_fill_prompt_gaps_only": follow_up_outcomes == set(missing_prompt_outcomes),
                "has_threshold_score_fields": all(
                    "score" in row["calibration"]
                    and "score_components" in row["calibration"]
                    and "target_label" in row["labels"]
                    and "predicted_label" in row["labels"]
                    for row in rows
                ),
            },
            "source_breakdown": {
                "live_rows": len(rows) - len(follow_ups),
                "follow_up_rows": len(follow_ups),
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Exp 232 semantic calibration corpus artifacts."
    )
    parser.add_argument(
        "--output",
        default=str(CORPUS_PATH),
        help="Relative or absolute output path for the JSONL corpus.",
    )
    parser.add_argument(
        "--results-output",
        default=str(RESULTS_PATH),
        help="Relative or absolute output path for the summary JSON artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = get_repo_root()
    rows = build_corpus(repo_root)
    write_jsonl(resolve_path(repo_root, Path(args.output)), rows)
    write_json(resolve_path(repo_root, Path(args.results_output)), build_results(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
