#!/usr/bin/env python3
# ruff: noqa: E501
"""Experiment 260: Solver-routed semantic benchmark with GPU acceleration.

This script extends the Exp 246 solver-semantic runner (scripts/experiment_246_solver_semantic_live.py)
to use the DualGPUBenchmarkHarness from Exp 258. It is the PRIMARY deliverable of milestone
2026.04.19: the first statistically complete solver-routed semantic benchmark on real IT models.

Background
----------
Exp 247 was blocked at 18/200 GSM8K cases at ~21 s/case on CPU. With the DualGPUBenchmarkHarness
introduced in Exp 258 (targeting ≤3 s/case), this script aims to run all 843 cells to completion:

    200 GSM8K × 3 modes × 2 models  =  1 200 cells
     81 constraint_ir × 3 modes × 2 models  =  486 cells

Checkpoint compatibility
------------------------
Checkpoint files are read from and written to ``results/checkpoints/experiment_246/`` using
the same filename scheme as Exp 246/247 (``<benchmark>__<model>__<mode>.json``). Existing
Exp 246 checkpoints with completed cases are reused automatically so the run resumes cleanly.

GPU / CPU fallback
------------------
The script attempts to initialize the DualGPUBenchmarkHarness. If CUDA is unavailable or VRAM
is insufficient, it falls back transparently to CPU inference (matching the Exp 246 path). The
``metadata.gpu_fallback`` field in the artifact records which path was taken.

Comparison
----------
The artifact includes a ``comparison`` block that benchmarks this run against:
- Exp 235 (semantic_verifier_v2 baseline)
- Exp 247 (partial 18/200 Qwen baseline)

Spec: REQ-VERIFY-058, REQ-VERIFY-059, REQ-VERIFY-041, REQ-VERIFY-036,
      SCENARIO-VERIFY-042, SCENARIO-VERIFY-036, SCENARIO-VERIFY-037
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from time import perf_counter
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_DATE = "20260413"
EXPERIMENT = 260
SCHEMA_ARTIFACT = "carnot.solver_semantic_gpu_results.v1"
DEFAULT_MAX_REPAIRS = 3
MODE_ORDER = ("baseline", "verify_only", "verify_repair")

MODEL_SPECS: list[dict[str, str]] = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
]

# Source artifacts — reuse cohorts verbatim from prior experiments for comparability
_SOURCE_ARTIFACT_GSM8K = "results/experiment_235_results.json"
_SOURCE_ARTIFACT_CONSTRAINT_IR = "results/experiment_221_results.json"

# Exp 246/247 checkpoint directory — Exp 260 reads existing checkpoints from here
_EXP246_CHECKPOINT_DIR = "results/checkpoints/experiment_246"

# Constraint type buckets for solver routing (shared with Exp 246)
_CARDINALITY_TYPES = frozenset(
    {"count_exact", "word_count_range", "sentence_count", "step_count", "sentence_count_per_section"}
)
_SET_MEMBERSHIP_TYPES = frozenset(
    {"must_include_token", "must_include_phrase", "forbidden_token", "forbidden_phrase",
     "enum_membership", "grounded_selection"}
)
# Matches explicit arithmetic equations of the form A OP B = C in response text.
_ARITHMETIC_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[+\-*×÷/]\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)

# ---------------------------------------------------------------------------
# Import DualGPUBenchmarkHarness from Exp 258 (avoids a package install requirement)
# ---------------------------------------------------------------------------

_EXP258_SCRIPT = Path(__file__).parent / "experiment_258_dual_gpu_harness.py"
_exp258_spec = importlib.util.spec_from_file_location("experiment_258_dual_gpu_harness", _EXP258_SCRIPT)
assert _exp258_spec is not None and _exp258_spec.loader is not None
_exp258_mod = importlib.util.module_from_spec(_exp258_spec)
sys.modules.setdefault("experiment_258_dual_gpu_harness", _exp258_mod)
_exp258_spec.loader.exec_module(_exp258_mod)  # type: ignore[union-attr]

DualGPUBenchmarkHarness = _exp258_mod.DualGPUBenchmarkHarness
ThroughputMeasurement = _exp258_mod.ThroughputMeasurement


# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------


def get_repo_root() -> Path:
    """Return the repository root, honouring ``CARNOT_REPO_ROOT`` when set."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def safe_slug(text: str) -> str:
    """Convert a label into a filesystem-safe slug."""
    cleaned = text.strip().lower().replace("/", "_").replace(" ", "_")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in cleaned)


def default_output_path() -> Path:
    """Return the default Exp 260 artifact path."""
    return get_repo_root() / "results" / "experiment_260_results.json"


def default_checkpoint_dir() -> Path:
    """Return the Exp 246 checkpoint directory (checkpoints are reused across experiments)."""
    return get_repo_root() / _EXP246_CHECKPOINT_DIR


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write an artifact with parent-dir creation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Cohort loading — verbatim reuse of Exp 235 and Exp 221 cohorts
# ---------------------------------------------------------------------------


def _display_path(path: Path) -> str:
    """Return a repo-relative path string when possible."""
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path)


def load_gsm8k_cohort(path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the GSM8K cohort from the checked-in Exp 235 artifact.

    Returns (cases, meta). The cohort is identical to the one used in Exp 246/247
    so results are directly comparable without re-sampling.

    Args:
        path: Override path for the Exp 235 artifact.

    Raises:
        ValueError: When the artifact does not contain benchmark='gsm8k_semantic'.
    """
    artifact_path = path or (get_repo_root() / _SOURCE_ARTIFACT_GSM8K)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if str(payload.get("benchmark")) != "gsm8k_semantic":
        raise ValueError(
            f"Expected benchmark='gsm8k_semantic'; got {payload.get('benchmark')!r}"
        )
    cohort_block = payload.get("cohort", {})
    cases = [dict(c) for c in list(cohort_block.get("cases", []))]
    metadata = payload.get("metadata", {})
    return cases, {
        "benchmark": "gsm8k_semantic",
        "source_artifact": _display_path(artifact_path),
        "sample_seed": int(metadata.get("sample_seed", 218)),
        "sample_size": int(metadata.get("sample_size", len(cases))),
        "case_count": int(cohort_block.get("case_count", len(cases))),
        "source_experiment": int(payload.get("experiment", 235)),
    }


def load_constraint_ir_cohort(path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the constraint-IR cohort from the checked-in Exp 221 artifact.

    Returns (cases, meta) with the same shape as ``load_gsm8k_cohort``.

    Args:
        path: Override path for the Exp 221 artifact.

    Raises:
        ValueError: When the artifact does not contain benchmark='constraint_ir'.
    """
    artifact_path = path or (get_repo_root() / _SOURCE_ARTIFACT_CONSTRAINT_IR)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if str(payload.get("benchmark")) != "constraint_ir":
        raise ValueError(
            f"Expected benchmark='constraint_ir'; got {payload.get('benchmark')!r}"
        )
    cohort_block = payload.get("cohort", {})
    cases = [dict(c) for c in list(cohort_block.get("cases", []))]
    metadata = payload.get("metadata", {})
    return cases, {
        "benchmark": "constraint_ir",
        "source_artifact": _display_path(artifact_path),
        "sample_seed": int(metadata.get("sample_seed", 218)),
        "sample_size": int(metadata.get("sample_size", len(cases))),
        "case_count": int(cohort_block.get("case_count", len(cases))),
        "source_experiment": int(payload.get("experiment", 221)),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers — Exp 218 / Exp 246 compatible interface
# ---------------------------------------------------------------------------


def checkpoint_path(
    checkpoint_dir: Path,
    *,
    benchmark: str,
    model_name: str,
    mode: str,
) -> Path:
    """Return the per-cell checkpoint path matching the Exp 246 naming scheme."""
    return checkpoint_dir / (
        f"{safe_slug(benchmark)}__{safe_slug(model_name)}__{safe_slug(mode)}.json"
    )


def load_checkpoint(path: Path, expected_case_ids: list[str]) -> dict[str, Any]:
    """Load a checkpoint when the cohort metadata still matches.

    Returns a fresh empty state when the file is missing, unreadable, or the
    case-id list has changed — preventing stale checkpoint reuse.
    """
    fresh: dict[str, Any] = {"case_ids": list(expected_case_ids), "results_by_case": {}}
    if not path.exists():
        return fresh
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return fresh
    if payload.get("case_ids") != expected_case_ids:
        return fresh
    results_by_case = payload.get("results_by_case", {})
    if not isinstance(results_by_case, dict):
        return fresh
    return {
        **payload,
        "case_ids": list(expected_case_ids),
        "results_by_case": dict(results_by_case),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically via a .tmp rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Formal claim extraction — pure, deterministic, no model calls
# (copied from Exp 246 verbatim to preserve route-classification parity)
# ---------------------------------------------------------------------------


def extract_formal_claims_from_response(
    response: str,
    *,
    case: dict[str, Any],
    benchmark: str,
) -> list[dict[str, Any]]:
    """Extract formal claim dicts from a model response for route verification.

    For ``gsm8k_semantic``: scans for explicit arithmetic equations (A OP B = C).
    For ``constraint_ir``: converts gold_atomic_constraints into claim dicts.

    Returns a list of claim dicts compatible with FormalClaim / normalize_claim.
    """
    if benchmark == "gsm8k_semantic":
        return _extract_arithmetic_claims(response)
    if benchmark == "constraint_ir":
        return _extract_constraint_ir_claims(case)
    return []


def _extract_arithmetic_claims(response: str) -> list[dict[str, Any]]:
    """Extract arithmetic equations (A OP B = C) from response text."""
    claims: list[dict[str, Any]] = []
    for match_idx, match in enumerate(_ARITHMETIC_RE.finditer(response)):
        try:
            a = float(match.group(1))
            b = float(match.group(2))
            result = float(match.group(3))
        except ValueError:
            continue
        claims.append({
            "claim_id": f"arith-{match_idx}",
            "claim_text": match.group(0),
            "candidate_solver_route": "arithmetic",
            "formalization_status": "formalized",
            "relation_type": "equation",
            "operands": [a, b, result],
            "target": "arithmetic_result",
            "bound_variables": [],
        })
    return claims


def _extract_constraint_ir_claims(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert gold_atomic_constraints into formal claim dicts."""
    claims: list[dict[str, Any]] = []
    for idx, constraint in enumerate(list(case.get("gold_atomic_constraints", []))):
        claim = _constraint_to_formal_claim(dict(constraint), idx)
        if claim is not None:
            claims.append(claim)
    return claims


def _constraint_to_formal_claim(constraint: dict[str, Any], idx: int) -> dict[str, Any] | None:
    """Map one gold_atomic_constraint to a formal claim dict, or None.

    Returns None for constraint types that cannot be normalized into any of the
    five supported solver routes (arithmetic, cardinality, set_membership, smt, abstain).
    """
    constraint_type = str(constraint.get("type", ""))
    value = constraint.get("value")
    target = str(constraint.get("target", ""))
    claim_id = str(constraint.get("constraint_id", f"constraint-{idx}"))

    # --- Cardinality route ---------------------------------------------------
    if constraint_type in _CARDINALITY_TYPES:
        if isinstance(value, int):
            return {
                "claim_id": claim_id,
                "claim_text": f"{target} count equals {value}",
                "candidate_solver_route": "cardinality",
                "formalization_status": "formalized",
                "relation_type": "equals",
                "operands": [float(value), 0.0],
                "target": target,
                "bound_variables": [],
            }
        if isinstance(value, list) and len(value) == 2:
            lo, hi = int(value[0]), int(value[1])
            return {
                "claim_id": claim_id,
                "claim_text": f"{target} count between {lo} and {hi}",
                "candidate_solver_route": "cardinality",
                "formalization_status": "formalized",
                "relation_type": "between",
                "operands": [float(lo), float(hi), 0.0],
                "target": target,
                "bound_variables": [],
            }
        return None

    # --- Set-membership route ------------------------------------------------
    if constraint_type in _SET_MEMBERSHIP_TYPES:
        if constraint_type in {"must_include_token", "must_include_phrase"}:
            phrase = str(value) if value is not None else ""
            if not phrase:
                return None
            return {
                "claim_id": claim_id,
                "claim_text": f"response contains '{phrase}'",
                "candidate_solver_route": "set_membership",
                "formalization_status": "formalized",
                "relation_type": "contains",
                "operands": [],
                "target": target,
                "bound_variables": [phrase],
            }
        if constraint_type in {"forbidden_token", "forbidden_phrase"}:
            phrase = str(value) if value is not None else ""
            if not phrase:
                return None
            return {
                "claim_id": claim_id,
                "claim_text": f"response does not contain '{phrase}'",
                "candidate_solver_route": "set_membership",
                "formalization_status": "formalized",
                "relation_type": "not_contains",
                "operands": [],
                "target": target,
                "bound_variables": [phrase],
            }
        if constraint_type in {"enum_membership", "grounded_selection"}:
            choices: list[str] = (
                [str(v) for v in value] if isinstance(value, list)
                else ([str(value)] if value is not None else [])
            )
            if not choices:
                return None
            return {
                "claim_id": claim_id,
                "claim_text": f"{target} in {choices!r}",
                "candidate_solver_route": "set_membership",
                "formalization_status": "formalized",
                "relation_type": "in",
                "operands": [],
                "target": choices[0],
                "bound_variables": choices,
            }
        return None

    return None


# ---------------------------------------------------------------------------
# Route-summary aggregation
# ---------------------------------------------------------------------------


def build_route_summary(claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-claim route decisions into a compact summary.

    Args:
        claims: Flat list of per-claim dicts, each with ``route`` and ``verdict``.

    Returns:
        Dict with ``by_route``, ``by_verdict``, ``total_claims``, and ``abstain_rate``.
    """
    by_route: dict[str, int] = {}
    by_verdict: dict[str, int] = {"supported": 0, "violated": 0, "abstain": 0}
    total_claims = 0
    for claim in claims:
        route = str(claim.get("route", "abstain"))
        verdict = str(claim.get("verdict", "abstain"))
        by_route[route] = by_route.get(route, 0) + 1
        normalized = verdict if verdict in by_verdict else "abstain"
        by_verdict[normalized] += 1
        total_claims += 1
    abstain_rate = (
        round(by_verdict["abstain"] / total_claims, 6) if total_claims > 0 else 0.0
    )
    return {
        "by_route": dict(sorted(by_route.items())),
        "by_verdict": dict(sorted(by_verdict.items())),
        "total_claims": total_claims,
        "abstain_rate": abstain_rate,
    }


def collect_all_claims_from_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-case ``formal_claims`` lists from a list of run dicts."""
    return [dict(claim) for run in runs for claim in run.get("formal_claims", [])]


# ---------------------------------------------------------------------------
# Per-run statistics
# ---------------------------------------------------------------------------


def _round_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def summarize_benchmark_runs(
    *,
    baseline_runs: list[dict[str, Any]],
    verify_only_runs: list[dict[str, Any]],
    verify_repair_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build per-mode summary statistics for one benchmark + model cell.

    Records baseline accuracy, verify-only flagging, and verify-repair yield
    plus paired deltas. The output schema matches Exp 246/247 for direct comparison.

    Args:
        baseline_runs: Per-case baseline results with ``correct`` (bool).
        verify_only_runs: Per-case verify-only results with ``flagged`` and ``accepted_correct``.
        verify_repair_runs: Per-case verify-repair results with ``correct``, ``repaired``, ``n_repairs``.

    Returns:
        Dict with ``baseline``, ``verify_only``, ``verify_repair``, and ``paired_deltas`` blocks.
    """
    n_cases = len(baseline_runs)
    if n_cases == 0:
        return {
            "baseline": {"n_cases": 0, "accuracy": 0.0},
            "verify_only": {"n_cases": 0, "n_flagged": 0},
            "verify_repair": {"n_cases": 0, "repair_yield": 0.0},
            "paired_deltas": {"verify_only_minus_baseline": 0.0, "repair_minus_baseline": 0.0},
        }

    n_correct_baseline = sum(1 for r in baseline_runs if r.get("correct"))
    baseline_accuracy = n_correct_baseline / n_cases
    n_wrong = n_cases - n_correct_baseline

    n_flagged = sum(1 for r in verify_only_runs if r.get("flagged"))
    n_accepted_correct = sum(1 for r in verify_only_runs if r.get("accepted_correct"))
    verify_accuracy = n_accepted_correct / n_cases

    n_repaired = sum(1 for r in verify_repair_runs if r.get("repaired"))
    repair_accuracy = sum(1 for r in verify_repair_runs if r.get("correct")) / n_cases
    repair_yield = round(n_repaired / n_wrong, 6) if n_wrong > 0 else 0.0

    # False positive analysis: cases flagged that were actually correct
    n_fp = sum(
        1 for base, vo in zip(baseline_runs, verify_only_runs)
        if base.get("correct") and vo.get("flagged")
    )
    fp_rate = round(n_fp / n_flagged, 6) if n_flagged > 0 else 0.0

    return {
        "baseline": {
            "n_cases": n_cases,
            "accuracy": baseline_accuracy,
            "n_correct": n_correct_baseline,
            "mean_latency_seconds": _round_mean([float(r.get("latency_seconds", 0.0)) for r in baseline_runs]),
            "mean_prompt_tokens": _round_mean([float(r.get("prompt_tokens", 0.0)) for r in baseline_runs]),
            "mean_response_tokens": _round_mean([float(r.get("response_tokens", 0.0)) for r in baseline_runs]),
            "mean_total_tokens": _round_mean([float(r.get("total_tokens", 0.0)) for r in baseline_runs]),
        },
        "verify_only": {
            "n_cases": n_cases,
            "n_flagged": n_flagged,
            "accuracy": verify_accuracy,
            "false_positives": n_fp,
            "false_positive_rate": fp_rate,
            "mean_latency_seconds": _round_mean([float(r.get("latency_seconds", 0.0)) for r in verify_only_runs]),
            "mean_total_tokens": _round_mean([float(r.get("total_tokens", 0.0)) for r in verify_only_runs]),
        },
        "verify_repair": {
            "n_cases": n_cases,
            "accuracy": repair_accuracy,
            "n_repaired": n_repaired,
            "repair_yield": repair_yield,
            "avg_repairs": _round_mean([float(r.get("n_repairs", 0)) for r in verify_repair_runs]),
            "mean_latency_seconds": _round_mean([float(r.get("latency_seconds", 0.0)) for r in verify_repair_runs]),
            "mean_total_tokens": _round_mean([float(r.get("total_tokens", 0.0)) for r in verify_repair_runs]),
        },
        "paired_deltas": {
            "verify_only_minus_baseline": round(verify_accuracy - baseline_accuracy, 6),
            "repair_minus_baseline": round(repair_accuracy - baseline_accuracy, 6),
        },
    }


# ---------------------------------------------------------------------------
# Comparison block against Exp 235 and Exp 247
# ---------------------------------------------------------------------------


# Exp 235 reference data (extracted from results/experiment_247_results.json comparison block)
_EXP235_REF: dict[str, Any] = {
    "Qwen3.5-0.8B": {
        "baseline_accuracy": 0.14,
        "verify_only_accuracy": 0.12,
        "verify_only_delta": -0.02,
        "repair_accuracy": 0.15,
        "repair_delta": 0.01,
        "semantic_verifier_v2": {
            "abstain_count": 153,
            "supported_count": 14,
            "violated_count": 33,
            "verdict_level_abstain_rate": 0.765,
            "n_flagged": 34,
            "n_wrong_detected": 30,
            "wrong_detection_rate": 0.174419,
            "false_positives": 4,
            "false_positive_rate": 0.142857,
        },
    },
    "Gemma4-E4B-it": {
        "baseline_accuracy": 0.465,
        "verify_only_accuracy": 0.335,
        "verify_only_delta": -0.13,
        "repair_accuracy": 0.475,
        "repair_delta": 0.01,
        "semantic_verifier_v2": {
            "abstain_count": 131,
            "supported_count": 30,
            "violated_count": 39,
            "verdict_level_abstain_rate": 0.655,
            "n_flagged": 54,
            "n_wrong_detected": 28,
            "wrong_detection_rate": 0.261682,
            "false_positives": 26,
            "false_positive_rate": 0.279570,
        },
    },
}

_EXP247_REF: dict[str, Any] = {
    "cells_completed": 1,
    "cells_total": 12,
    "best_cell": {
        "benchmark": "gsm8k_semantic",
        "model": "Qwen3.5-0.8B",
        "mode": "baseline",
        "cases_done": 18,
        "cases_total": 200,
    },
    "measured_throughput_seconds_per_case": 21.132,
    "blocker": "runtime_budget (CPU only)",
}


def build_comparison_block(
    *,
    gsm8k_statistics: dict[str, Any],
    constraint_ir_statistics: dict[str, Any],
    gsm8k_route_summary: dict[str, Any],
    constraint_ir_route_summary: dict[str, Any],
    exp247_cells_completed: int,
    exp247_cells_total: int,
) -> dict[str, Any]:
    """Build the comparison block benchmarking Exp 260 against Exp 235 and Exp 247.

    The comparison is unambiguous about progress: it reports deltas vs. Exp 235 numbers
    and counts how many blocked Exp 247 cells are now completed.

    Args:
        gsm8k_statistics: Per-model stats dict from ``summarize_benchmark_runs`` for GSM8K.
        constraint_ir_statistics: Same for constraint_ir.
        gsm8k_route_summary: Aggregated route summary for GSM8K verify runs.
        constraint_ir_route_summary: Aggregated route summary for constraint_ir verify runs.
        exp247_cells_completed: Number of cells completed in Exp 247 (was 1).
        exp247_cells_total: Total expected cells across both benchmarks (12 = 2 models × 3 modes × 2 benchmarks).

    Returns:
        JSON-serialisable dict for the ``comparison`` artifact key.
    """
    # Count how many cells Exp 260 has completed
    exp260_cells_completed = 0
    for model_name in [s["name"] for s in MODEL_SPECS]:
        for bench_stats in [gsm8k_statistics, constraint_ir_statistics]:
            if model_name in bench_stats:
                s = bench_stats[model_name]
                if s.get("baseline", {}).get("n_cases", 0) > 0:
                    exp260_cells_completed += 1

    # Per-model delta vs Exp 235 semantic_verifier_v2
    per_model_vs_235: dict[str, Any] = {}
    for model_name, ref in _EXP235_REF.items():
        exp260_gsm8k = gsm8k_statistics.get(model_name, {})
        if exp260_gsm8k.get("baseline", {}).get("n_cases", 0) > 0:
            exp260_baseline = exp260_gsm8k["baseline"]["accuracy"]
            exp260_vo = exp260_gsm8k["verify_only"]["accuracy"]
            exp260_fp_rate = exp260_gsm8k["verify_only"].get("false_positive_rate", None)
            per_model_vs_235[model_name] = {
                "exp235_baseline_accuracy": ref["baseline_accuracy"],
                "exp260_baseline_accuracy": round(exp260_baseline, 6),
                "baseline_delta": round(exp260_baseline - ref["baseline_accuracy"], 6),
                "exp235_verify_only_accuracy": ref["verify_only_accuracy"],
                "exp260_verify_only_accuracy": round(exp260_vo, 6),
                "verify_only_delta": round(exp260_vo - ref["verify_only_accuracy"], 6),
                "exp235_fp_rate": ref["semantic_verifier_v2"]["false_positive_rate"],
                "exp260_fp_rate": exp260_fp_rate,
                "fp_rate_delta": (
                    round(exp260_fp_rate - ref["semantic_verifier_v2"]["false_positive_rate"], 6)
                    if exp260_fp_rate is not None else None
                ),
                "exp235_verdict_abstain_rate": ref["semantic_verifier_v2"]["verdict_level_abstain_rate"],
                "exp260_gsm8k_claim_abstain_rate": gsm8k_route_summary.get("abstain_rate", None),
            }
        else:
            per_model_vs_235[model_name] = {"status": "no_data"}

    # verify_only non-harmful finding: is verify_only_delta >= 0 for at least one model+benchmark?
    non_harmful_cases: list[dict[str, Any]] = []
    for model_name in [s["name"] for s in MODEL_SPECS]:
        for bench_name, bench_stats in [
            ("gsm8k_semantic", gsm8k_statistics),
            ("constraint_ir", constraint_ir_statistics),
        ]:
            s = bench_stats.get(model_name, {})
            delta = s.get("paired_deltas", {}).get("verify_only_minus_baseline", None)
            if delta is not None and delta >= 0.0:
                non_harmful_cases.append({"model": model_name, "benchmark": bench_name, "delta": delta})

    return {
        "vs_exp235_semantic_verifier_v2": {
            "source_artifact": "results/experiment_235_results.json",
            "exp235_benchmark": "gsm8k_semantic",
            "exp235_cohort_size": 200,
            "exp235_sample_seed": 218,
            "cohort_identity": "identical — same 200 case_ids from experiment_219",
            "per_model": per_model_vs_235,
            "note": (
                "Exp 235 used semantic_verifier_v2 (confidence-gated LLM verification). "
                "Exp 260 uses solver-routed FormalClaimVerifier (deterministic, no confidence threshold). "
                "Lower abstain rates and lower FP rates would constitute a material improvement."
            ),
        },
        "vs_exp247_partial": {
            "exp247_status": "partial_blocked",
            "exp247_cells_completed": exp247_cells_completed,
            "exp247_cells_total": exp247_cells_total,
            "exp247_blocker": _EXP247_REF["blocker"],
            "exp247_throughput_seconds_per_case": _EXP247_REF["measured_throughput_seconds_per_case"],
            "exp260_cells_completed": exp260_cells_completed,
            "exp260_cells_total": exp247_cells_total,
            "progress_note": (
                f"Exp 247 completed {exp247_cells_completed}/{exp247_cells_total} cells. "
                f"Exp 260 completed {exp260_cells_completed}/{exp247_cells_total} cells."
            ),
        },
        "verify_only_non_harmful_finding": {
            "question": "Is verify-only non-harmful for at least one model on at least one benchmark slice?",
            "non_harmful_cases": non_harmful_cases,
            "finding": (
                "CONFIRMED: verify-only is non-harmful" if non_harmful_cases
                else "NOT CONFIRMED: verify-only reduces accuracy for all evaluated model+benchmark combinations"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_artifact_payload(
    *,
    output_path: Path,
    gsm8k_cohort: list[dict[str, Any]],
    gsm8k_cohort_meta: dict[str, Any],
    constraint_ir_cohort: list[dict[str, Any]],
    constraint_ir_cohort_meta: dict[str, Any],
    gsm8k_paired_runs: list[dict[str, Any]],
    constraint_ir_paired_runs: list[dict[str, Any]],
    gsm8k_route_summary: dict[str, Any],
    constraint_ir_route_summary: dict[str, Any],
    gsm8k_statistics: dict[str, Any],
    constraint_ir_statistics: dict[str, Any],
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
    checkpoint_dir: Path,
    max_repairs: int,
    inference_mode: str,
    gpu_fallback: bool,
    throughput_report: dict[str, Any],
    comparison_block: dict[str, Any],
) -> dict[str, Any]:
    """Build the Exp 260 artifact payload.

    Extends the Exp 246 schema with GPU throughput metrics, false-positive budgets,
    and direct comparison against Exp 235 and Exp 247.

    Args:
        output_path: Where the artifact will be written.
        gsm8k_cohort: Case list from the Exp 235 cohort.
        gsm8k_cohort_meta: Metadata from ``load_gsm8k_cohort``.
        constraint_ir_cohort: Case list from the Exp 221 cohort.
        constraint_ir_cohort_meta: Metadata from ``load_constraint_ir_cohort``.
        gsm8k_paired_runs: Ordered paired-run dicts for the GSM8K slice.
        constraint_ir_paired_runs: Ordered paired-run dicts for the constraint-IR slice.
        gsm8k_route_summary: Aggregated route summary for GSM8K verify runs.
        constraint_ir_route_summary: Same for constraint-IR.
        gsm8k_statistics: Per-model summary stats for the GSM8K slice.
        constraint_ir_statistics: Per-model summary stats for the constraint-IR slice.
        started_at: ISO-8601 UTC start timestamp.
        finished_at: ISO-8601 UTC end timestamp.
        runtime_seconds: Wall-clock seconds for the complete run.
        checkpoint_dir: Directory holding per-cell checkpoint files.
        max_repairs: Maximum verify-repair iterations per case.
        inference_mode: ``"simulated"``, ``"live_cpu"``, or ``"live_gpu"``.
        gpu_fallback: True when the run fell back from GPU to CPU.
        throughput_report: ``ThroughputMeasurement.report()`` dict.
        comparison_block: Output of ``build_comparison_block()``.

    Returns:
        Dict ready for ``write_artifact()``.
    """
    checkpoint_pattern = (
        "results/checkpoints/experiment_246/<benchmark>__<model>__<mode>.json"
    )

    def _cohort_block(cohort: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_count": len(cohort),
            "case_ids": [str(c["case_id"]) for c in cohort],
            "source_artifact": str(meta.get("source_artifact", "")),
            "source_experiment": int(meta.get("source_experiment", 0)),
            "sample_seed": int(meta.get("sample_seed", 218)),
        }

    return {
        "experiment": EXPERIMENT,
        "title": "Solver-routed semantic benchmark with GPU acceleration (Exp 260)",
        "run_date": RUN_DATE,
        "schema": {
            "artifact": SCHEMA_ARTIFACT,
            "benchmark_slices": ["gsm8k_semantic", "constraint_ir"],
            "consumes": ["carnot.solver_semantic_live.v1", "carnot.dual_gpu_harness_report.v1"],
        },
        "metadata": {
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": round(runtime_seconds, 3),
            "models": [dict(m) for m in MODEL_SPECS],
            "modes": list(MODE_ORDER),
            "source_artifacts": {
                "gsm8k_semantic": str(gsm8k_cohort_meta.get("source_artifact", "")),
                "constraint_ir": str(constraint_ir_cohort_meta.get("source_artifact", "")),
            },
            "output_path": str(output_path),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_pattern": checkpoint_pattern,
            "exp246_checkpoint_reuse": True,
            "max_repairs": max_repairs,
            "inference_mode": inference_mode,
            "gpu_fallback": gpu_fallback,
            "force_live": os.environ.get("CARNOT_FORCE_LIVE") == "1",
            "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
            "throughput": throughput_report,
            "target_seconds_per_case": 3.0,
        },
        "benchmarks": {
            "gsm8k_semantic": {
                "cohort": _cohort_block(gsm8k_cohort, gsm8k_cohort_meta),
                "paired_runs": list(gsm8k_paired_runs),
                "route_summary": dict(gsm8k_route_summary),
                "statistics": dict(gsm8k_statistics),
            },
            "constraint_ir": {
                "cohort": _cohort_block(constraint_ir_cohort, constraint_ir_cohort_meta),
                "paired_runs": list(constraint_ir_paired_runs),
                "route_summary": dict(constraint_ir_route_summary),
                "statistics": dict(constraint_ir_statistics),
            },
        },
        "comparison": comparison_block,
    }


# ---------------------------------------------------------------------------
# Live execution helpers
# ---------------------------------------------------------------------------


def _live_inference_mode(gpu_active: bool) -> str:
    """Return the inference mode string for the artifact metadata."""
    if os.environ.get("CARNOT_FORCE_LIVE") == "1":
        return "live_gpu" if gpu_active else "live_cpu"
    return "simulated"


def _verify_formal_claims_batch(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:  # pragma: no cover
    """Run FormalClaimVerifier over a list of claim dicts.

    Returns a list of verdict dicts suitable for embedding under the ``formal_claims``
    key in per-case results.
    """
    from carnot.pipeline.formal_claim_verifier import (  # type: ignore[import-untyped]
        FormalClaimVerifier,
        normalize_claim,
    )
    verifier = FormalClaimVerifier()
    verdicts: list[dict[str, Any]] = []
    for raw in claims:
        formal_claim = normalize_claim(raw)
        verdict = verifier.verify_claim(formal_claim)
        verdicts.append(verdict.to_dict())
    return verdicts


def _execute_gsm8k_case_live(  # pragma: no cover
    case: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    mode: str,
    max_repairs: int,
) -> dict[str, Any]:
    """Execute one GSM8K semantic case in live mode.

    Generates a response, extracts formal claims, verifies them, and records
    baseline / verify-only / verify-repair outcomes.
    """
    import time as _time
    from carnot.inference.model_loader import generate  # type: ignore[import-untyped]

    question = str(case.get("question", ""))
    ground_truth = int(case.get("ground_truth", -9999))
    prompt = f"Solve this math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
    t0 = _time.time()
    response = str(generate(model, tokenizer, prompt, max_new_tokens=256))
    latency = _time.time() - t0

    raw_claims = extract_formal_claims_from_response(response, case=case, benchmark="gsm8k_semantic")
    formal_claims = _verify_formal_claims_batch(raw_claims)

    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    predicted = int(float(numbers[-1].replace(",", ""))) if numbers else None
    correct = predicted == ground_truth

    base_result: dict[str, Any] = {
        "case_id": str(case["case_id"]),
        "mode": mode,
        "question": question,
        "ground_truth": ground_truth,
        "response": response,
        "predicted": predicted,
        "correct": correct,
        "formal_claims": formal_claims,
        "n_claims": len(formal_claims),
        "n_supported": sum(1 for c in formal_claims if c.get("verdict") == "supported"),
        "n_violated": sum(1 for c in formal_claims if c.get("verdict") == "violated"),
        "n_abstained": sum(1 for c in formal_claims if c.get("verdict") == "abstain"),
        "latency_seconds": round(latency, 3),
        "prompt_tokens": len(prompt.split()),
        "response_tokens": len(response.split()),
        "total_tokens": len(prompt.split()) + len(response.split()),
    }

    if mode == "baseline":
        return base_result

    flagged = any(c.get("verdict") == "violated" for c in formal_claims)
    accepted_correct = correct and not flagged
    vo_result = {**base_result, "flagged": flagged, "accepted_correct": accepted_correct}
    if mode == "verify_only":
        return vo_result

    # verify_repair: attempt repair when flagged and wrong
    repaired = False
    n_repairs = 0
    if flagged and not correct:
        for _ in range(max_repairs):
            repair_prompt = (
                f"{prompt}\n\nPrevious answer had arithmetic errors. "
                "Please recompute carefully.\n\nAnswer:"
            )
            response = str(generate(model, tokenizer, repair_prompt, max_new_tokens=256))
            numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
            predicted = int(float(numbers[-1].replace(",", ""))) if numbers else None
            correct = predicted == ground_truth
            n_repairs += 1
            if correct:
                repaired = True
                break
    return {**vo_result, "response": response, "predicted": predicted, "correct": correct,
            "repaired": repaired, "n_repairs": n_repairs}


def _execute_constraint_ir_case_live(  # pragma: no cover
    case: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    mode: str,
    max_repairs: int,
) -> dict[str, Any]:
    """Execute one constraint-IR case in live mode.

    Generates a response, extracts formal claims from gold constraints, verifies
    them, and records outcomes.
    """
    import time as _time
    from carnot.inference.model_loader import generate  # type: ignore[import-untyped]

    prompt = str(case.get("prompt", ""))
    t0 = _time.time()
    response = str(generate(model, tokenizer, prompt, max_new_tokens=256))
    latency = _time.time() - t0

    raw_claims = extract_formal_claims_from_response(response, case=case, benchmark="constraint_ir")
    formal_claims = _verify_formal_claims_batch(raw_claims)
    flagged = any(c.get("verdict") == "violated" for c in formal_claims)

    base_result: dict[str, Any] = {
        "case_id": str(case["case_id"]),
        "mode": mode,
        "response": response,
        "formal_claims": formal_claims,
        "n_claims": len(formal_claims),
        "n_supported": sum(1 for c in formal_claims if c.get("verdict") == "supported"),
        "n_violated": sum(1 for c in formal_claims if c.get("verdict") == "violated"),
        "n_abstained": sum(1 for c in formal_claims if c.get("verdict") == "abstain"),
        "flagged": flagged,
        "accepted": not flagged,
        "latency_seconds": round(latency, 3),
        "prompt_tokens": len(prompt.split()),
        "response_tokens": len(response.split()),
        "total_tokens": len(prompt.split()) + len(response.split()),
    }

    if mode == "baseline":
        result = dict(base_result)
        del result["flagged"]
        del result["accepted"]
        return result

    if mode == "verify_only":
        return {**base_result, "accepted_correct": not flagged}

    # verify_repair
    repaired = False
    n_repairs = 0
    if flagged:
        violated_details = [
            c.get("failure_detail", "") for c in formal_claims if c.get("verdict") == "violated"
        ]
        for _ in range(max_repairs):
            repair_prompt = (
                f"{prompt}\n\nViolated constraints: {violated_details!r}. "
                "Please revise your answer.\n\nAnswer:"
            )
            response = str(generate(model, tokenizer, repair_prompt, max_new_tokens=256))
            raw_claims = extract_formal_claims_from_response(response, case=case, benchmark="constraint_ir")
            formal_claims = _verify_formal_claims_batch(raw_claims)
            flagged = any(c.get("verdict") == "violated" for c in formal_claims)
            n_repairs += 1
            if not flagged:
                repaired = True
                break
    return {
        **base_result,
        "response": response,
        "formal_claims": formal_claims,
        "repaired": repaired,
        "n_repairs": n_repairs,
        "accepted": not flagged,
        "accepted_correct": not flagged,
    }


def _run_benchmark_for_model_cpu(  # pragma: no cover
    *,
    benchmark: str,
    model_spec: dict[str, str],
    cases: list[dict[str, Any]],
    checkpoint_dir: Path,
    max_repairs: int,
    harness: DualGPUBenchmarkHarness,
    gpu_active: bool = False,
) -> list[dict[str, Any]]:
    """Run all three modes for one model on one benchmark.

    Uses the DualGPUBenchmarkHarness run_mode() for checkpoint compatibility and
    throughput tracking. Loads models on GPU when ``gpu_active`` is True, otherwise CPU.

    Args:
        benchmark: Benchmark name.
        model_spec: Dict with ``name`` and ``hf_id``.
        cases: Ordered case list.
        checkpoint_dir: Checkpoint directory (Exp 246-compatible).
        max_repairs: Max verify-repair iterations per case.
        harness: Harness instance for checkpoint + throughput tracking.
        gpu_active: When True, loads models on CUDA; otherwise CPU.
    """
    from carnot.inference.model_loader import load_model  # type: ignore[import-untyped]

    hf_id = model_spec["hf_id"]
    device = "cuda" if gpu_active else "cpu"
    model, tokenizer = load_model(hf_id, device=device)

    execute_fn = (
        _execute_gsm8k_case_live if benchmark == "gsm8k_semantic"
        else _execute_constraint_ir_case_live
    )

    paired: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        def _make_executor(m: str) -> Any:
            def _execute(case: dict[str, Any]) -> dict[str, Any]:
                return execute_fn(case, model=model, tokenizer=tokenizer, mode=m, max_repairs=max_repairs)
            return _execute

        mode_results = harness.run_mode(
            benchmark=benchmark,
            model_name=model_spec["name"],
            mode=mode,
            cases=cases,
            checkpoint_dir=checkpoint_dir,
            execute_case=_make_executor(mode),
        )
        paired.append({
            "benchmark": benchmark,
            "model_name": model_spec["name"],
            "hf_id": model_spec["hf_id"],
            "mode": mode,
            "cases": mode_results,
            "summary": {"n_cases": len(mode_results)},
        })

    del model, tokenizer
    gc.collect()
    return paired


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the Exp 260 CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Exp 260: solver-routed semantic benchmark with GPU acceleration. "
            "Resumes from existing Exp 246 checkpoints."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Artifact output path (default: results/experiment_260_results.json).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        type=Path,
        default=default_checkpoint_dir(),
        help="Checkpoint directory — defaults to Exp 246's dir for seamless resume.",
    )
    parser.add_argument(
        "--max-repairs",
        dest="max_repairs",
        type=int,
        default=DEFAULT_MAX_REPAIRS,
        help="Maximum verify-repair iterations per case (default: 3).",
    )
    parser.add_argument(
        "--gsm8k-artifact",
        dest="gsm8k_artifact",
        type=Path,
        default=None,
        help="Override path for the Exp 235 GSM8K cohort artifact.",
    )
    parser.add_argument(
        "--constraint-ir-artifact",
        dest="constraint_ir_artifact",
        type=Path,
        default=None,
        help="Override path for the Exp 221 constraint-IR cohort artifact.",
    )
    return parser


def main() -> None:  # pragma: no cover
    """CLI entry point for the Exp 260 GPU-accelerated solver-semantic benchmark.

    In live mode (``CARNOT_FORCE_LIVE=1``):
    - Attempts GPU execution via DualGPUBenchmarkHarness
    - Falls back to CPU inference if GPU is unavailable (VRAM OOM or no CUDA)
    - Records ``metadata.gpu_fallback`` in the artifact

    Existing checkpoints from Exp 246 in ``results/checkpoints/experiment_246/``
    are reused automatically so the run continues from where Exp 247 stopped.
    """
    from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: PLC0415
    from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: PLC0415
    from scripts.experiment_template import ExperimentTemplate  # noqa: PLC0415

    apply_env_autofix()

    parser = _build_parser()
    args = parser.parse_args()

    output_path: Path = args.output
    checkpoint_dir: Path = args.checkpoint_dir
    max_repairs: int = args.max_repairs

    started_at = utc_now()
    t_start = time.time()

    _tmpl = ExperimentTemplate(
        exp_id=EXPERIMENT,
        title="Solver-routed semantic benchmark with GPU acceleration (Exp 260)",
        deliverable="results/experiment_260_results.json",
        requires_gpu=False,
        repo_root=get_repo_root(),
    )
    _tmpl.setup()

    _watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXPERIMENT,
        timeout_minutes=40,
        result_path=str(output_path),
    )
    _watchdog.start()

    print(f"[Exp 260] Starting at {started_at}")
    print(f"[Exp 260] Output: {output_path}")
    print(f"[Exp 260] Checkpoint dir: {checkpoint_dir}")
    print(f"[Exp 260] CARNOT_FORCE_LIVE={os.environ.get('CARNOT_FORCE_LIVE', '0')}")

    # ------------------------------------------------------------------
    # Load cohorts
    # ------------------------------------------------------------------
    print("[Exp 260] Loading cohorts...")
    gsm8k_cohort, gsm8k_cohort_meta = load_gsm8k_cohort(args.gsm8k_artifact)
    constraint_ir_cohort, constraint_ir_cohort_meta = load_constraint_ir_cohort(
        args.constraint_ir_artifact
    )
    print(f"[Exp 260] GSM8K cohort: {len(gsm8k_cohort)} cases")
    print(f"[Exp 260] Constraint-IR cohort: {len(constraint_ir_cohort)} cases")

    gsm8k_paired_runs: list[dict[str, Any]] = []
    gsm8k_statistics: dict[str, Any] = {}
    gsm8k_all_claims: list[dict[str, Any]] = []

    constraint_ir_paired_runs: list[dict[str, Any]] = []
    constraint_ir_statistics: dict[str, Any] = {}
    constraint_ir_all_claims: list[dict[str, Any]] = []

    gpu_fallback = False

    if os.environ.get("CARNOT_FORCE_LIVE") == "1":
        # ------------------------------------------------------------------
        # Try GPU; fall back to CPU on any failure
        # ------------------------------------------------------------------
        harness = DualGPUBenchmarkHarness(model_specs=MODEL_SPECS)
        gpu_active = False

        try:
            import torch
            harness._torch = torch  # wire real torch for VRAM check
            harness.verify_gpu_assignments()
            gpu_active = True
            print("[Exp 260] GPU verification passed — running in GPU mode")
        except Exception as exc:
            print(f"[Exp 260] GPU unavailable ({exc}); falling back to CPU")
            gpu_fallback = True
            gpu_active = False

        inference_mode = _live_inference_mode(gpu_active)

        for model_spec in MODEL_SPECS:
            print(f"[Exp 260] Running GSM8K for {model_spec['name']} on {'GPU' if gpu_active else 'CPU'}...")
            model_paired = _run_benchmark_for_model_cpu(
                benchmark="gsm8k_semantic",
                model_spec=model_spec,
                cases=gsm8k_cohort,
                checkpoint_dir=checkpoint_dir,
                max_repairs=max_repairs,
                harness=harness,
                gpu_active=gpu_active,
            )
            gsm8k_paired_runs.extend(model_paired)

            print(f"[Exp 260] Running constraint_ir for {model_spec['name']} on {'GPU' if gpu_active else 'CPU'}...")
            ir_paired = _run_benchmark_for_model_cpu(
                benchmark="constraint_ir",
                model_spec=model_spec,
                cases=constraint_ir_cohort,
                checkpoint_dir=checkpoint_dir,
                max_repairs=max_repairs,
                harness=harness,
                gpu_active=gpu_active,
            )
            constraint_ir_paired_runs.extend(ir_paired)

        harness.empty_cache_between_runs()

        # Aggregate statistics per model
        for model_spec in MODEL_SPECS:
            name = model_spec["name"]
            _runs = {
                m: next(
                    (r["cases"] for r in gsm8k_paired_runs if r["model_name"] == name and r["mode"] == m),
                    [],
                )
                for m in MODE_ORDER
            }
            gsm8k_statistics[name] = summarize_benchmark_runs(
                baseline_runs=_runs["baseline"],
                verify_only_runs=_runs["verify_only"],
                verify_repair_runs=_runs["verify_repair"],
            )
            gsm8k_all_claims.extend(
                collect_all_claims_from_runs(_runs["verify_only"] + _runs["verify_repair"])
            )

            _ir_runs = {
                m: next(
                    (r["cases"] for r in constraint_ir_paired_runs if r["model_name"] == name and r["mode"] == m),
                    [],
                )
                for m in MODE_ORDER
            }
            constraint_ir_statistics[name] = summarize_benchmark_runs(
                baseline_runs=_ir_runs["baseline"],
                verify_only_runs=_ir_runs["verify_only"],
                verify_repair_runs=_ir_runs["verify_repair"],
            )
            constraint_ir_all_claims.extend(
                collect_all_claims_from_runs(_ir_runs["verify_only"] + _ir_runs["verify_repair"])
            )

        throughput_report = harness.throughput.report()

    else:
        # Simulated mode — produce artifact schema without live data
        print("[Exp 260] Simulated mode (CARNOT_FORCE_LIVE not set) — no live inference")
        inference_mode = "simulated"
        throughput_report = {"target_seconds_per_case": 3.0, "per_model": {}}

    # ------------------------------------------------------------------
    # Route summaries
    # ------------------------------------------------------------------
    gsm8k_route_summary = build_route_summary(gsm8k_all_claims)
    constraint_ir_route_summary = build_route_summary(constraint_ir_all_claims)

    # ------------------------------------------------------------------
    # Comparison block
    # ------------------------------------------------------------------
    comparison_block = build_comparison_block(
        gsm8k_statistics=gsm8k_statistics,
        constraint_ir_statistics=constraint_ir_statistics,
        gsm8k_route_summary=gsm8k_route_summary,
        constraint_ir_route_summary=constraint_ir_route_summary,
        exp247_cells_completed=1,   # only 18/200 gsm8k Qwen baseline was done
        exp247_cells_total=12,      # 2 models × 3 modes × 2 benchmarks
    )

    finished_at = utc_now()
    runtime_seconds = time.time() - t_start

    # ------------------------------------------------------------------
    # Build and write artifact
    # ------------------------------------------------------------------
    payload = build_artifact_payload(
        output_path=output_path,
        gsm8k_cohort=gsm8k_cohort,
        gsm8k_cohort_meta=gsm8k_cohort_meta,
        constraint_ir_cohort=constraint_ir_cohort,
        constraint_ir_cohort_meta=constraint_ir_cohort_meta,
        gsm8k_paired_runs=gsm8k_paired_runs,
        constraint_ir_paired_runs=constraint_ir_paired_runs,
        gsm8k_route_summary=gsm8k_route_summary,
        constraint_ir_route_summary=constraint_ir_route_summary,
        gsm8k_statistics=gsm8k_statistics,
        constraint_ir_statistics=constraint_ir_statistics,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=runtime_seconds,
        checkpoint_dir=checkpoint_dir,
        max_repairs=max_repairs,
        inference_mode=inference_mode,
        gpu_fallback=gpu_fallback,
        throughput_report=throughput_report,
        comparison_block=comparison_block,
    )

    write_artifact(output_path, payload)
    print(f"[Exp 260] Artifact written to {output_path}")
    print(f"[Exp 260] Runtime: {runtime_seconds:.1f}s  inference_mode={inference_mode}  gpu_fallback={gpu_fallback}")
    _watchdog.stop()
    _tmpl.assert_deliverable_written()


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
