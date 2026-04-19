#!/usr/bin/env python3
"""Experiment 252: predictive verification corpus from checked-in live artifacts.

Writes:
- ``data/research/predictive_verification_corpus_252.jsonl``
- ``results/experiment_252_results.json``

Spec: REQ-VERIFY-252,
SCENARIO-VERIFY-252-A (schema shape),
SCENARIO-VERIFY-252-B (deterministic generation),
SCENARIO-VERIFY-252-C (provenance completeness),
SCENARIO-VERIFY-252-D (semantic and code traces),
SCENARIO-VERIFY-252-E (memory-hit metadata),
SCENARIO-VERIFY-252-F (accepted repairs)

Background
----------
Exp 241 demonstrated that the self-learning loop is too passive: the tracker,
case memory, and compiled policy only update *after* a violation is observed,
never *before* a response is committed.  To support predictive verification —
routing a response early to a stricter verifier, or injecting constraints
before generation completes — we need a corpus that bundles:

- ``partial_response``:   the baseline (first-iteration) response or candidate
  code, i.e. what was available at prediction time.
- ``final_response``:     the accepted repair if one existed, otherwise null.
- ``violation_family``:   the ordered list of observed violation families
  (empty for clean cases).
- ``process_label``:      from the Exp 248 / Exp 250 process integrity labels
  where available, else inferred from outcome.
- ``verifier_outcome``:   "verified", "violated", or "abstain" at the baseline
  iteration.
- ``downstream_repair_outcome``: whether a subsequent repair was accepted,
  rejected, or not attempted.
- ``memory_hit``:         whether the Exp 241 case-memory strategy found a
  matched case key.
- ``memory_match_metadata``: candidate and matched case key lists from
  Exp 241 strategy decisions.
- ``policy_context``:     the compiled-policy context if present.
- ``accepted_repair``:    the text or code of the accepted repair (non-null only
  when downstream_repair_outcome == "accepted").

Sources
-------
- Exp 241 (held-out decisions): 116 records spanning gsm8k-semantic and
  HumanEval-code.  Each carries the richest strategy metadata including memory
  hits and policy context.
- Exp 235 (GSM8K semantic baseline): 200 cases × 2 models providing response
  text and latency for cross-referencing Exp 241.
- Exp 238 (HumanEval code): 30 cases × 2 models providing candidate code.
- Exp 248 (process integrity corpus): 849 rows providing process labels for
  Exp 235 / Exp 238 cases.
- Exp 246 checkpoint (gsm8k verify-only): partial/truncated responses with
  formal-claim violation counts — the clearest "partial response" signal.
- Exp 250 checkpoint (code process-aware): per-case baseline code with
  process_flags carrying richer process labels and defect evidence.

Design decisions
----------------
Records from Exp 241 form the backbone because they carry the most complete
strategy metadata.  Exp 246 and Exp 250 checkpoint records are added for
benchmark slices that Exp 241 did not cover (verify-only and process-aware
stages), providing the code-path coverage needed to train the constraint-addition
predictor without being benchmark-specific.

All records include a ``sample_position`` derived from the original experiment
ordering so that the predictor and calibrator can reproduce the exact held-out
split from Exp 241.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

RUN_DATE = "20260413"
EXPERIMENT = 252
SCHEMA_VERSION = "carnot.predictive_verification_corpus.v1"

# ---------------------------------------------------------------------------
# Source artifact paths (relative to repo root)
# ---------------------------------------------------------------------------

_SOURCE_241 = Path("results/experiment_241_results.json")
_SOURCE_235 = Path("results/experiment_235_results.json")
_SOURCE_238 = Path("results/experiment_238_results.json")
_SOURCE_248_CORPUS = Path("data/research/process_integrity_corpus_248.jsonl")
_SOURCE_246_CKPT_QWEN = Path(
    "results/checkpoints/experiment_246/gsm8k_semantic__qwen3_5-0_8b__verify_only.json"
)
_SOURCE_250_CKPT_QWEN = Path("results/checkpoints/experiment_250/exp250_qwen3_5_0_8b.json")
_SOURCE_250_CKPT_GEMMA = Path("results/checkpoints/experiment_250/exp250_gemma4_e4b_it.json")

CORPUS_RELATIVE = Path("data/research/predictive_verification_corpus_252.jsonl")
RESULTS_RELATIVE = Path("results/experiment_252_results.json")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def get_repo_root() -> Path:
    """Return the repository root, respecting the CARNOT_REPO_ROOT override."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def _resolve(repo_root: Path, candidate: Path) -> Path:
    return candidate if candidate.is_absolute() else repo_root / candidate


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Model name slug helpers (deterministic, no I/O)
# ---------------------------------------------------------------------------


def _model_slug(model_name: str) -> str:
    """Convert a model name to a safe filename slug."""
    return (
        model_name.lower()
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


# ---------------------------------------------------------------------------
# Index builders (pure, no I/O)
# ---------------------------------------------------------------------------


def _build_exp235_index(
    payload: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index Exp 235 cases by (model_name, case_id, mode).

    Returns a dict mapping (model, case_id, mode) → case dict carrying
    ``response``, ``correct``, ``latency_seconds``, and optional
    ``typed_reasoning``.
    """
    idx: dict[tuple[str, str, str], dict[str, Any]] = {}
    for run in payload.get("paired_runs", []):
        model = run["model_name"]
        mode = run["mode"]
        for case in run.get("cases", []):
            idx[(model, case["case_id"], mode)] = case
    return idx


def _build_exp238_direct_index(
    payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index Exp 238 per-problem results by (model_name, case_id) with ordered position.

    Returns a dict mapping (model, case_id) → problem dict, preserving the
    original dataset_idx for sample_position.
    """
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for model, run_data in payload.get("model_runs", {}).items():
        for problem in run_data.get("per_problem_results", []):
            idx[(model, problem["case_id"])] = problem
    return idx


def _build_exp238_index(
    payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index Exp 238 per-problem results by (model_name, case_id).

    Returns a dict mapping (model, case_id) → problem dict carrying
    ``baseline.candidate_code``, ``verify_repair``, and ``history``.
    """
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for model, run_data in payload.get("model_runs", {}).items():
        for problem in run_data.get("per_problem_results", []):
            idx[(model, problem["case_id"])] = problem
    return idx


def _build_exp248_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Index Exp 248 process-integrity corpus rows by (model, case_id, iteration).

    Returns a dict mapping (model, case_id, iteration) → row dict carrying
    ``process_label``, ``outcome_label``, and ``process_evidence``.
    """
    idx: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["case_id"], row["iteration"])
        idx[key] = row
    return idx


# ---------------------------------------------------------------------------
# Violation-family extraction helpers (pure)
# ---------------------------------------------------------------------------


def _semantic_violation_families(error_types: list[str]) -> list[str]:
    """Deduplicate and sort semantic violation family strings.

    Exp 241 error_types often include composite strings like
    ``"question_grounding_failures:answer_target_mismatch"`` alongside the
    atomic ``"answer_target_mismatch"``.  We strip composite prefixes and
    return only the family-level labels.
    """
    families: set[str] = set()
    for et in error_types:
        # "family:sub" → keep "family"
        top = et.split(":")[0].strip()
        if top:
            families.add(top)
    return sorted(families)


def _code_violation_families(error_types: list[str]) -> list[str]:
    """Return sorted code-specific violation family labels from Exp 241 error_types."""
    # Code error_types are already atomic (e.g. "syntax", "humaneval_failure")
    return sorted({et.strip() for et in error_types if et.strip()})


def _infer_verifier_outcome(
    *,
    detected: bool,
    error_types: list[str],
    baseline_success: bool,
) -> str:
    """Infer a verifier_outcome string from Exp 241 decision fields.

    - If the case was detected with non-empty error_types → "violated"
    - If baseline was correct and not detected → "verified"
    - Otherwise → "abstain" (the verifier did not confidently fire)
    """
    if detected and error_types:
        return "violated"
    if baseline_success:
        return "verified"
    return "abstain"


def _repair_outcome_from_decision(dec: dict[str, Any]) -> str:
    """Map Exp 241 held-out decision fields to a downstream_repair_outcome label."""
    if dec.get("repair_success"):
        return "accepted"
    # Repair was attempted if repair_latency_seconds > 0.0 (non-trivial attempt)
    repair_lat = dec.get("repair_latency_seconds", 0.0) or 0.0
    if repair_lat > 0.001:
        return "rejected"
    return "not_attempted"


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def _build_record_from_exp241_semantic(
    dec: dict[str, Any],
    exp235_idx: dict[tuple[str, str, str], dict[str, Any]],
    exp248_idx: dict[tuple[str, str, int], dict[str, Any]],
    source_241_path: str,
    source_235_path: str,
) -> dict[str, Any]:
    """Build one corpus record from an Exp 241 semantic held-out decision.

    Cross-references Exp 235 for the response text and Exp 248 for the
    process label.  When a cross-reference is unavailable the field is filled
    with a sentinel that preserves the schema contract.
    """
    model = dec["model_name"]
    case_id = dec["case_id"]
    sample_position = dec["sample_position"]

    # --- response text from Exp 235 ---
    exp235_case = exp235_idx.get((model, case_id, "baseline"))
    partial_response: str | None = None
    baseline_latency: float | None = None
    if exp235_case:
        partial_response = exp235_case.get("response") or ""
        baseline_latency = exp235_case.get("latency_seconds")

    # Repair response: Exp 235 stores verify_repair mode cases with
    # initial_response and final_response fields.  We use final_response when
    # the decision's repair_success is True.
    vr_case = exp235_idx.get((model, case_id, "verify_repair"))
    final_response: str | None = None
    if vr_case and dec.get("repair_success"):
        fr = vr_case.get("final_response")
        if fr and fr != vr_case.get("initial_response"):
            final_response = fr
        elif fr:
            # repair matched initial — still record it
            final_response = fr

    # --- process label from Exp 248 ---
    exp248_row = exp248_idx.get((model, case_id, 0))
    if exp248_row:
        process_label = exp248_row["process_label"]
    else:
        # Infer a minimal process label from outcome fields
        if dec.get("baseline_success"):
            process_label = "clean"
        elif dec.get("repair_success"):
            process_label = "repair_fixed_outcome_only"
        else:
            process_label = "wrong_answer_partially_sound_process"

    # --- strategy metadata ---
    cm_strat = dec.get("strategies", {}).get("case_memory", {})
    policy_strat = dec.get("strategies", {}).get("case_memory_plus_policy", {})
    memory_hit = bool(cm_strat.get("matched_case_keys"))
    memory_match_metadata = {
        "candidate_error_types": cm_strat.get("candidate_error_types", []),
        "matched_error_types": cm_strat.get("matched_error_types", []),
        "candidate_case_keys": cm_strat.get("candidate_case_keys", []),
        "matched_case_keys": cm_strat.get("matched_case_keys", []),
        "support_models": cm_strat.get("support_models", []),
    }
    policy_context = policy_strat.get("policy_context") or {}

    # --- accepted repair text ---
    accepted_repair: str | None = None
    repair_outcome = _repair_outcome_from_decision(dec)
    if repair_outcome == "accepted" and final_response:
        accepted_repair = final_response

    corpus_id = (
        f"pvc252-241-gsm8k_semantic"
        f"-{_model_slug(model)}-{case_id}"
    )

    return {
        "corpus_id": corpus_id,
        "run_date": RUN_DATE,
        "experiment": EXPERIMENT,
        "source_experiment": 241,
        "source_artifact": source_241_path,
        "benchmark": dec.get("benchmark", "gsm8k_semantic"),
        "benchmark_slice": "held_out_decision",
        "domain": "reasoning",
        "model": model,
        "case_id": case_id,
        "sample_position": sample_position,
        "partial_response": partial_response or "",
        "final_response": final_response,
        "violation_family": _semantic_violation_families(dec.get("error_types", [])),
        "process_label": process_label,
        "outcome_label": "correct" if dec.get("baseline_success") else "incorrect",
        "verifier_outcome": _infer_verifier_outcome(
            detected=dec.get("detected", False),
            error_types=dec.get("error_types", []),
            baseline_success=dec.get("baseline_success", False),
        ),
        "confidence": None,  # Exp 241 does not carry per-case confidence scores
        "baseline_latency_seconds": dec.get("baseline_latency_seconds"),
        "repair_latency_seconds": dec.get("repair_latency_seconds"),
        "downstream_repair_outcome": repair_outcome,
        "memory_hit": memory_hit,
        "memory_match_metadata": memory_match_metadata,
        "policy_context": policy_context,
        "accepted_repair": accepted_repair,
        "provenance": {
            "source_experiment": 241,
            "source_artifact": source_241_path,
            "cross_ref_artifact": source_235_path,
            "model": model,
            "benchmark": dec.get("benchmark", "gsm8k_semantic"),
            "benchmark_slice": "held_out_decision",
            "case_id": case_id,
            "sample_position": sample_position,
        },
    }


def _build_record_from_exp241_code(
    dec: dict[str, Any],
    exp238_idx: dict[tuple[str, str], dict[str, Any]],
    source_241_path: str,
    source_238_path: str,
) -> dict[str, Any]:
    """Build one corpus record from an Exp 241 code held-out decision.

    Cross-references Exp 238 for the candidate code and repair history.
    """
    model = dec["model_name"]
    case_id = dec["case_id"]
    sample_position = dec["sample_position"]

    # --- candidate code from Exp 238 ---
    exp238_problem = exp238_idx.get((model, case_id))
    partial_response: str = ""
    final_response: str | None = None
    if exp238_problem:
        baseline_info = exp238_problem.get("baseline", {})
        partial_response = baseline_info.get("candidate_code") or ""
        # Accepted repair: check verify_repair.accepted and take final_body
        vr = exp238_problem.get("verify_repair", {})
        if vr.get("accepted") and vr.get("final_body"):
            final_response = vr["final_body"]

    # --- strategy metadata ---
    cm_strat = dec.get("strategies", {}).get("case_memory", {})
    policy_strat = dec.get("strategies", {}).get("case_memory_plus_policy", {})
    memory_hit = bool(cm_strat.get("matched_case_keys"))
    memory_match_metadata = {
        "candidate_error_types": cm_strat.get("candidate_error_types", []),
        "matched_error_types": cm_strat.get("matched_error_types", []),
        "candidate_case_keys": cm_strat.get("candidate_case_keys", []),
        "matched_case_keys": cm_strat.get("matched_case_keys", []),
        "support_models": cm_strat.get("support_models", []),
    }
    policy_context = policy_strat.get("policy_context") or {}

    repair_outcome = _repair_outcome_from_decision(dec)
    accepted_repair: str | None = None
    if repair_outcome == "accepted" and final_response:
        accepted_repair = final_response

    # Code baseline_success means the official tests passed
    baseline_success = dec.get("baseline_success", False)
    corpus_id = (
        f"pvc252-241-humaneval_code"
        f"-{_model_slug(model)}-{case_id}"
    )

    return {
        "corpus_id": corpus_id,
        "run_date": RUN_DATE,
        "experiment": EXPERIMENT,
        "source_experiment": 241,
        "source_artifact": source_241_path,
        "benchmark": dec.get("benchmark", "humaneval_dual_model_spec"),
        "benchmark_slice": "held_out_decision",
        "domain": "code",
        "model": model,
        "case_id": case_id,
        "sample_position": sample_position,
        "partial_response": partial_response,
        "final_response": final_response,
        "violation_family": _code_violation_families(dec.get("error_types", [])),
        "process_label": "clean" if baseline_success else "wrong_answer_wrong_process",
        "outcome_label": "correct" if baseline_success else "incorrect",
        "verifier_outcome": _infer_verifier_outcome(
            detected=dec.get("detected", False),
            error_types=dec.get("error_types", []),
            baseline_success=baseline_success,
        ),
        "confidence": None,
        "baseline_latency_seconds": dec.get("baseline_latency_seconds"),
        "repair_latency_seconds": dec.get("repair_latency_seconds"),
        "downstream_repair_outcome": repair_outcome,
        "memory_hit": memory_hit,
        "memory_match_metadata": memory_match_metadata,
        "policy_context": policy_context,
        "accepted_repair": accepted_repair,
        "provenance": {
            "source_experiment": 241,
            "source_artifact": source_241_path,
            "cross_ref_artifact": source_238_path,
            "model": model,
            "benchmark": dec.get("benchmark", "humaneval_dual_model_spec"),
            "benchmark_slice": "held_out_decision",
            "case_id": case_id,
            "sample_position": sample_position,
        },
    }


def _build_record_from_exp235_baseline(
    case: dict[str, Any],
    model: str,
    exp248_idx: dict[tuple[str, str, int], dict[str, Any]],
    source_235_path: str,
    position: int,
) -> dict[str, Any]:
    """Build one corpus record directly from an Exp 235 baseline case.

    These records supply the bulk of semantic training examples with full
    response text and typed-reasoning provenance.
    """
    case_id = case["case_id"]

    # Process label from Exp 248 index
    exp248_row = exp248_idx.get((model, case_id, 0))
    if exp248_row:
        process_label = exp248_row["process_label"]
        process_evidence = exp248_row.get("process_evidence", {})
        verifier_outcome_raw = process_evidence.get("verifier_verdict", "unknown")
        # Normalise to our enum
        if verifier_outcome_raw in ("verified", "supported"):
            verifier_outcome = "verified"
        elif verifier_outcome_raw == "violated":
            verifier_outcome = "violated"
        elif verifier_outcome_raw in ("abstain", "unknown"):
            verifier_outcome = "abstain"
        else:
            verifier_outcome = "unknown"
        confidence_raw = process_evidence.get("semantic_error_probability")
        confidence: float | None = float(confidence_raw) if confidence_raw is not None else None
    else:
        is_correct = case.get("correct", False)
        process_label = "clean" if is_correct else "wrong_answer_partially_sound_process"
        verifier_outcome = "verified" if is_correct else "abstain"
        confidence = None

    is_correct = case.get("correct", False)
    corpus_id = (
        f"pvc252-235-gsm8k_semantic"
        f"-{_model_slug(model)}-{case_id}"
    )

    return {
        "corpus_id": corpus_id,
        "run_date": RUN_DATE,
        "experiment": EXPERIMENT,
        "source_experiment": 235,
        "source_artifact": source_235_path,
        "benchmark": "gsm8k_semantic",
        "benchmark_slice": "baseline",
        "domain": "reasoning",
        "model": model,
        "case_id": case_id,
        "sample_position": position,
        "partial_response": case.get("response") or "",
        "final_response": None,  # Exp 235 baseline has no repair
        "violation_family": [],  # Exp 235 does not carry per-case violation families
        "process_label": process_label,
        "outcome_label": "correct" if is_correct else "incorrect",
        "verifier_outcome": verifier_outcome,
        "confidence": confidence,
        "baseline_latency_seconds": case.get("latency_seconds"),
        "repair_latency_seconds": None,
        "downstream_repair_outcome": "not_attempted",
        "memory_hit": False,
        "memory_match_metadata": {
            "candidate_error_types": [],
            "matched_error_types": [],
            "candidate_case_keys": [],
            "matched_case_keys": [],
            "support_models": [],
        },
        "policy_context": {},
        "accepted_repair": None,
        "provenance": {
            "source_experiment": 235,
            "source_artifact": source_235_path,
            "model": model,
            "benchmark": "gsm8k_semantic",
            "benchmark_slice": "baseline",
            "case_id": case_id,
            "sample_position": position,
        },
    }


def _build_record_from_exp246_verify_only(
    case_id: str,
    case: dict[str, Any],
    model: str,
    exp248_idx: dict[tuple[str, str, int], dict[str, Any]],
    source_246_path: str,
    position: int,
) -> dict[str, Any]:
    """Build one corpus record from an Exp 246 verify-only checkpoint case.

    These records are the clearest source of "partial response" data: the model
    was generating a step-by-step solution and the checkpoint captures the
    response at the point where the verifier fired (or the model stopped), often
    mid-computation.

    ``formal_claims`` from the checkpoint give the per-claim violation signal
    without requiring the full typed-reasoning IR.
    """
    is_correct = case.get("correct", False)
    n_violated = case.get("n_violated", 0)
    flagged = case.get("flagged", False)

    # Verifier outcome from formal claim counts
    if n_violated > 0 or flagged:
        verifier_outcome = "violated"
    elif case.get("n_supported", 0) > 0:
        verifier_outcome = "verified"
    else:
        verifier_outcome = "abstain"

    # Confidence: fraction of claims violated (higher → more likely error)
    n_claims = case.get("n_claims", 0) or 0
    if n_claims > 0:
        confidence = float(n_violated) / float(n_claims)
    else:
        confidence = None

    # Process label from Exp 248
    exp248_row = exp248_idx.get((model, case_id, 0))
    if exp248_row:
        process_label = exp248_row["process_label"]
    else:
        if is_correct:
            process_label = "clean"
        elif n_violated > 0:
            process_label = "wrong_answer_partially_sound_process"
        else:
            process_label = "wrong_answer_partially_sound_process"

    corpus_id = (
        f"pvc252-246-gsm8k_semantic_vo"
        f"-{_model_slug(model)}-{case_id}"
    )

    return {
        "corpus_id": corpus_id,
        "run_date": RUN_DATE,
        "experiment": EXPERIMENT,
        "source_experiment": 246,
        "source_artifact": source_246_path,
        "benchmark": "gsm8k_semantic",
        "benchmark_slice": "verify_only",
        "domain": "reasoning",
        "model": model,
        "case_id": case_id,
        "sample_position": position,
        "partial_response": case.get("response") or "",
        "final_response": None,
        "violation_family": [],  # formal_claims route does not produce named families
        "process_label": process_label,
        "outcome_label": "correct" if is_correct else "incorrect",
        "verifier_outcome": verifier_outcome,
        "confidence": confidence,
        "baseline_latency_seconds": case.get("latency_seconds"),
        "repair_latency_seconds": None,
        "downstream_repair_outcome": "not_attempted",
        "memory_hit": False,
        "memory_match_metadata": {
            "candidate_error_types": [],
            "matched_error_types": [],
            "candidate_case_keys": [],
            "matched_case_keys": [],
            "support_models": [],
        },
        "policy_context": {},
        "accepted_repair": None,
        "provenance": {
            "source_experiment": 246,
            "source_artifact": source_246_path,
            "model": model,
            "benchmark": "gsm8k_semantic",
            "benchmark_slice": "verify_only",
            "case_id": case_id,
            "sample_position": position,
            "formal_claims": case.get("formal_claims", []),
        },
    }


def _build_record_from_exp238_direct(
    problem: dict[str, Any],
    model: str,
    source_238_path: str,
) -> dict[str, Any]:
    """Build one corpus record directly from an Exp 238 per-problem result.

    These records cover the non-held-out slice of the HumanEval code benchmark
    (positions 0–22 per model) that Exp 241 did not include in its held-out
    decisions.  They carry baseline candidate code and verify-repair outcomes.
    """
    case_id = problem["case_id"]
    sample_position = problem.get("dataset_idx", 0)

    baseline_info = problem.get("baseline", {})
    partial_response = baseline_info.get("candidate_code") or ""

    # Accepted repair from verify_repair
    vr = problem.get("verify_repair", {})
    final_response: str | None = None
    accepted_repair: str | None = None
    repair_outcome: str = "not_attempted"
    if vr.get("accepted"):
        fb = vr.get("final_body") or ""
        final_response = fb
        accepted_repair = fb
        repair_outcome = "accepted"
    elif vr.get("repaired") is True and not vr.get("accepted"):
        repair_outcome = "rejected"

    # Verifier outcome from official_tests_verify_only
    official_passed = baseline_info.get("official_passed", False)
    if official_passed:
        verifier_outcome = "verified"
    elif problem.get("pbt_verify_only", {}).get("accepted") is False:
        verifier_outcome = "violated"
    else:
        verifier_outcome = "abstain"

    is_correct = official_passed
    process_label = "clean" if is_correct else "wrong_answer_wrong_process"

    corpus_id = (
        f"pvc252-238-humaneval_code"
        f"-{_model_slug(model)}-{case_id}"
    )

    return {
        "corpus_id": corpus_id,
        "run_date": RUN_DATE,
        "experiment": EXPERIMENT,
        "source_experiment": 238,
        "source_artifact": source_238_path,
        "benchmark": "humaneval_dual_model_spec",
        "benchmark_slice": "code_spec",
        "domain": "code",
        "model": model,
        "case_id": case_id,
        "sample_position": sample_position,
        "partial_response": partial_response,
        "final_response": final_response,
        "violation_family": [],
        "process_label": process_label,
        "outcome_label": "correct" if is_correct else "incorrect",
        "verifier_outcome": verifier_outcome,
        "confidence": None,
        "baseline_latency_seconds": None,
        "repair_latency_seconds": None,
        "downstream_repair_outcome": repair_outcome,
        "memory_hit": False,
        "memory_match_metadata": {
            "candidate_error_types": [],
            "matched_error_types": [],
            "candidate_case_keys": [],
            "matched_case_keys": [],
            "support_models": [],
        },
        "policy_context": {},
        "accepted_repair": accepted_repair,
        "provenance": {
            "source_experiment": 238,
            "source_artifact": source_238_path,
            "model": model,
            "benchmark": "humaneval_dual_model_spec",
            "benchmark_slice": "code_spec",
            "case_id": case_id,
            "sample_position": sample_position,
        },
    }


def _build_record_from_exp250_code(
    case_id: str,
    case: dict[str, Any],
    model: str,
    source_250_path: str,
    position: int,
) -> dict[str, Any]:
    """Build one corpus record from an Exp 250 process-aware code checkpoint case.

    Exp 250 adds ``process_flags`` (process_label, defects, right_for_wrong_reasons)
    to the code pipeline, making it the best source of labeled code traces.

    We take the *baseline* candidate code as ``partial_response`` and the final
    repair body (if accepted) as ``accepted_repair`` / ``final_response``.
    """
    baseline_info = case.get("baseline", {})
    partial_response = baseline_info.get("candidate_code") or ""

    # Process label and verifier outcome from process_flags.baseline
    pf_baseline = case.get("process_flags", {}).get("baseline", {})
    process_label_raw = pf_baseline.get("process_label", "")
    # Map Exp 250 process labels to our corpus vocabulary
    _LABEL_MAP = {
        "clean": "clean",
        "wrong_answer_wrong_process": "wrong_answer_partially_sound_process",
        "right_for_wrong_reasons": "right_answer_wrong_process",
        "repair_improved_outcome_only": "repair_fixed_outcome_only",
        "repair_improved_process_and_outcome": "repair_fixed_process_and_outcome",
    }
    process_label = _LABEL_MAP.get(process_label_raw, process_label_raw or "wrong_answer_partially_sound_process")

    # Verifier outcome from defect kinds
    defects = pf_baseline.get("defects", [])
    has_contradiction = any(d.get("kind") == "contradictory_intermediate" for d in defects)
    if has_contradiction:
        verifier_outcome = "violated"
    elif pf_baseline.get("outcome_correct"):
        verifier_outcome = "verified"
    else:
        verifier_outcome = "abstain"

    is_correct = pf_baseline.get("outcome_correct", False)

    # Violation families from defect kinds
    violation_family = sorted({d.get("kind", "") for d in defects if d.get("kind")})
    # Filter to code-domain families
    code_families = {
        "unsupported_step", "missing_premise_jump", "contradictory_intermediate",
        "repair_stall", "syntax_error", "execution_error",
    }
    violation_family = [f for f in violation_family if f in code_families]

    # Accepted repair
    vr = case.get("verify_repair", {})
    final_response: str | None = None
    accepted_repair: str | None = None
    repair_outcome: str = "not_attempted"
    if vr.get("accepted"):
        final_body = vr.get("final_body") or ""
        final_response = final_body
        accepted_repair = final_body
        repair_outcome = "accepted"
    elif case.get("process_flags", {}).get("final", {}).get("outcome_correct") is False:
        repair_outcome = "rejected"

    # Confidence: fraction of defects that are contradictory_intermediate
    n_defects = len(defects)
    if n_defects > 0:
        n_contradiction = sum(1 for d in defects if d.get("kind") == "contradictory_intermediate")
        confidence = float(n_contradiction) / float(n_defects)
    else:
        confidence = None

    corpus_id = (
        f"pvc252-250-humaneval_code_pa"
        f"-{_model_slug(model)}-{case_id}"
    )

    return {
        "corpus_id": corpus_id,
        "run_date": RUN_DATE,
        "experiment": EXPERIMENT,
        "source_experiment": 250,
        "source_artifact": source_250_path,
        "benchmark": "humaneval_code",
        "benchmark_slice": "process_aware",
        "domain": "code",
        "model": model,
        "case_id": case_id,
        "sample_position": position,
        "partial_response": partial_response,
        "final_response": final_response,
        "violation_family": violation_family,
        "process_label": process_label,
        "outcome_label": "correct" if is_correct else "incorrect",
        "verifier_outcome": verifier_outcome,
        "confidence": confidence,
        "baseline_latency_seconds": None,  # Exp 250 checkpoints do not carry per-case latency
        "repair_latency_seconds": None,
        "downstream_repair_outcome": repair_outcome,
        "memory_hit": False,
        "memory_match_metadata": {
            "candidate_error_types": [],
            "matched_error_types": [],
            "candidate_case_keys": [],
            "matched_case_keys": [],
            "support_models": [],
        },
        "policy_context": {},
        "accepted_repair": accepted_repair,
        "provenance": {
            "source_experiment": 250,
            "source_artifact": source_250_path,
            "model": model,
            "benchmark": "humaneval_code",
            "benchmark_slice": "process_aware",
            "case_id": case_id,
            "sample_position": position,
        },
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove rows with duplicate corpus_ids, keeping the first occurrence.

    Exp 241 held-out decisions may overlap with Exp 235 baseline cases for the
    same (model, case_id) pair.  We keep the Exp 241 record (richer metadata)
    by processing it first.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        cid = row["corpus_id"]
        if cid not in seen:
            seen.add(cid)
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the results summary artifact from the finalized corpus rows."""
    by_process: dict[str, int] = defaultdict(int)
    by_outcome: dict[str, int] = defaultdict(int)
    by_domain: dict[str, int] = defaultdict(int)
    by_verifier: dict[str, int] = defaultdict(int)
    by_repair: dict[str, int] = defaultdict(int)
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0})

    memory_hits = 0

    for row in rows:
        by_process[row["process_label"]] += 1
        by_outcome[row["outcome_label"]] += 1
        by_domain[row["domain"]] += 1
        by_verifier[row["verifier_outcome"]] += 1
        by_repair[row["downstream_repair_outcome"]] += 1
        src_key = f"exp{row['source_experiment']}"
        by_source[src_key]["count"] += 1
        if row["memory_hit"]:
            memory_hits += 1

    return {
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "schema_version": SCHEMA_VERSION,
        "title": "Predictive Verification Corpus — Exp 252",
        "total_records": len(rows),
        "memory_hit_count": memory_hits,
        "label_counts": {
            "by_process_label": dict(sorted(by_process.items())),
            "by_outcome_label": dict(sorted(by_outcome.items())),
            "by_domain": dict(sorted(by_domain.items())),
            "by_verifier_outcome": dict(sorted(by_verifier.items())),
            "by_repair_outcome": dict(sorted(by_repair.items())),
        },
        "source_breakdown": {k: v for k, v in sorted(by_source.items())},
        "source_artifacts": {
            "exp241": str(_SOURCE_241),
            "exp235": str(_SOURCE_235),
            "exp238": str(_SOURCE_238),
            "exp248_corpus": str(_SOURCE_248_CORPUS),
            "exp246_ckpt_qwen": str(_SOURCE_246_CKPT_QWEN),
            "exp250_ckpt_qwen": str(_SOURCE_250_CKPT_QWEN),
            "exp250_ckpt_gemma": str(_SOURCE_250_CKPT_GEMMA),
        },
    }


# ---------------------------------------------------------------------------
# Main build function (pure I/O wrapper — testable)
# ---------------------------------------------------------------------------


def build_corpus(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the corpus rows and summary from checked-in artifacts.

    Returns (rows, summary) without writing any files.  This separation makes
    the generation logic fully unit-testable.
    """
    # Load all source artifacts
    payload_241 = _load_json(_resolve(repo_root, _SOURCE_241))
    payload_235 = _load_json(_resolve(repo_root, _SOURCE_235))
    payload_238 = _load_json(_resolve(repo_root, _SOURCE_238))
    exp248_rows = _load_jsonl(_resolve(repo_root, _SOURCE_248_CORPUS))

    # Build indices
    exp235_idx = _build_exp235_index(payload_235)
    exp238_idx = _build_exp238_index(payload_238)
    exp238_direct_idx = _build_exp238_direct_index(payload_238)
    exp248_idx = _build_exp248_index(exp248_rows)

    # Relative paths for provenance strings
    src_241 = str(_SOURCE_241)
    src_235 = str(_SOURCE_235)
    src_238 = str(_SOURCE_238)
    src_246_qwen = str(_SOURCE_246_CKPT_QWEN)
    src_250_qwen = str(_SOURCE_250_CKPT_QWEN)
    src_250_gemma = str(_SOURCE_250_CKPT_GEMMA)

    rows: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Source 1: Exp 241 held-out decisions (richest strategy metadata)
    # -----------------------------------------------------------------------
    for dec in payload_241.get("held_out_decisions", []):
        if dec.get("source_experiment") == 235:
            # Semantic case
            row = _build_record_from_exp241_semantic(
                dec, exp235_idx, exp248_idx, src_241, src_235
            )
            rows.append(row)
        elif dec.get("source_experiment") == 238:
            # Code case
            row = _build_record_from_exp241_code(
                dec, exp238_idx, src_241, src_238
            )
            rows.append(row)

    # -----------------------------------------------------------------------
    # Source 1b: Exp 238 direct records — non-held-out code slice
    # -----------------------------------------------------------------------
    held_out_ids_238: set[tuple[str, str]] = {
        (dec["model_name"], dec["case_id"])
        for dec in payload_241.get("held_out_decisions", [])
        if dec.get("source_experiment") == 238
    }
    for (model, case_id), problem in sorted(
        exp238_direct_idx.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        if (model, case_id) in held_out_ids_238:
            # Already captured from Exp 241 with richer metadata — skip
            continue
        row = _build_record_from_exp238_direct(problem, model, src_238)
        rows.append(row)

    # -----------------------------------------------------------------------
    # Source 2: Exp 235 baseline cases (full response text + calibration)
    # Only non-Exp-241-held-out positions (positions 0–150 per model).
    # -----------------------------------------------------------------------
    held_out_ids_235: set[tuple[str, str]] = {
        (dec["model_name"], dec["case_id"])
        for dec in payload_241.get("held_out_decisions", [])
        if dec.get("source_experiment") == 235
    }
    for run in payload_235.get("paired_runs", []):
        if run["mode"] != "baseline":
            continue
        model = run["model_name"]
        for pos, case in enumerate(run.get("cases", [])):
            key = (model, case["case_id"])
            if key in held_out_ids_235:
                # Already captured from Exp 241 with richer metadata — skip
                continue
            row = _build_record_from_exp235_baseline(
                case, model, exp248_idx, src_235, pos
            )
            rows.append(row)

    # -----------------------------------------------------------------------
    # Source 3: Exp 246 verify-only checkpoint (partial/truncated responses)
    # -----------------------------------------------------------------------
    ckpt_246_path = _resolve(repo_root, _SOURCE_246_CKPT_QWEN)
    if ckpt_246_path.exists():
        ckpt_246 = _load_json(ckpt_246_path)
        model_246 = ckpt_246.get("model_name", "Qwen3.5-0.8B")
        rbc = ckpt_246.get("results_by_case", {})
        for pos, (case_id, case) in enumerate(
            sorted(rbc.items(), key=lambda kv: kv[0])
        ):
            row = _build_record_from_exp246_verify_only(
                case_id, case, model_246, exp248_idx, src_246_qwen, pos
            )
            rows.append(row)

    # -----------------------------------------------------------------------
    # Source 4: Exp 250 process-aware code checkpoints (both models)
    # -----------------------------------------------------------------------
    for src_250, model_250 in (
        (_SOURCE_250_CKPT_QWEN, "Qwen3.5-0.8B"),
        (_SOURCE_250_CKPT_GEMMA, "Gemma4-E4B-it"),
    ):
        ckpt_250_path = _resolve(repo_root, src_250)
        if not ckpt_250_path.exists():
            continue
        ckpt_250 = _load_json(ckpt_250_path)
        rbc = ckpt_250.get("results_by_case", {})
        for pos, (case_id, case) in enumerate(
            sorted(rbc.items(), key=lambda kv: kv[0])
        ):
            row = _build_record_from_exp250_code(
                case_id, case, model_250, str(src_250), pos
            )
            rows.append(row)

    # -----------------------------------------------------------------------
    # Deduplication and stable sort (by corpus_id for determinism)
    # -----------------------------------------------------------------------
    rows = _dedupe(rows)
    rows.sort(key=lambda r: r["corpus_id"])

    summary = _build_summary(rows)
    return rows, summary


def build_and_write(
    repo_root: Path,
    corpus_path: Path,
    summary_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the corpus, write both output files, and return (rows, summary)."""
    rows, summary = build_corpus(repo_root)
    _write_jsonl(corpus_path, rows)
    _write_json(summary_path, summary)
    return rows, summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 252: build the predictive verification corpus."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repository root (default: auto-detected from script location).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root or get_repo_root()
    corpus_path = _resolve(repo_root, CORPUS_RELATIVE)
    summary_path = _resolve(repo_root, RESULTS_RELATIVE)

    print(f"[exp252] Building predictive verification corpus …")
    print(f"[exp252] Repo root: {repo_root}")

    rows, summary = build_and_write(
        repo_root=repo_root,
        corpus_path=corpus_path,
        summary_path=summary_path,
    )

    print(f"[exp252] Wrote {len(rows)} records → {corpus_path}")
    print(f"[exp252] Summary → {summary_path}")
    print(f"[exp252] Domain breakdown: {summary['label_counts']['by_domain']}")
    print(f"[exp252] Repair outcomes: {summary['label_counts']['by_repair_outcome']}")
    print(f"[exp252] Memory hits: {summary['memory_hit_count']}")


if __name__ == "__main__":
    main()


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
