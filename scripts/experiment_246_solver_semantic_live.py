#!/usr/bin/env python3
# ruff: noqa: E501
"""Experiment 246: solver-semantic live benchmark runner.

Creates a checkpointed benchmark runner that measures route quality from the
solver-routed formal claim verifier (REQ-VERIFY-058) across two distinct
reasoning shapes:

- ``gsm8k_semantic``  — live GSM8K semantic-failure cases reused from the
  checked-in Exp 235 cohort (200 cases, paired to Exp 219)
- ``constraint_ir``   — prompt-side constraint-following cases reused from
  the checked-in Exp 221 cohort (≈ 81 cases)

This script is the **runner only** (Exp 246).  Full live execution with real
model weights happens in Exp 247.  Stopping here keeps the conductor out of
test-fix loops on the harness while Exp 247 can focus entirely on honest
live results.

Checkpointing and cohort discipline follow the Exp 218 / Exp 235 pattern:
- One checkpoint file per (benchmark, model, mode) cell
- Checkpoint invalidated when the sampled case-id list changes
- Cohorts are loaded verbatim from the checked-in prior artifacts to preserve
  exact pairing

For each response the verifier runs ``FormalClaimVerifier`` over extracted
claims and records per-claim route, verdict, and failure_detail so Exp 247
can aggregate route quality without post-hoc schema changes.

Spec: REQ-VERIFY-058, REQ-VERIFY-059
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

RUN_DATE = "20260413"
EXPERIMENT = 246
SCHEMA_ARTIFACT = "carnot.solver_semantic_live.v1"
DEFAULT_MAX_REPAIRS = 3
MODE_ORDER = ("baseline", "verify_only", "verify_repair")

MODEL_SPECS: list[dict[str, str]] = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
]

# Source artifacts for cohort reuse
_SOURCE_ARTIFACT_GSM8K = "results/experiment_235_results.json"
_SOURCE_ARTIFACT_CONSTRAINT_IR = "results/experiment_221_results.json"

# Constraint types that map to the cardinality solver route
_CARDINALITY_TYPES = frozenset(
    {"count_exact", "word_count_range", "sentence_count", "step_count", "sentence_count_per_section"}
)
# Constraint types that map to the set_membership solver route
_SET_MEMBERSHIP_TYPES = frozenset(
    {"must_include_token", "must_include_phrase", "forbidden_token", "forbidden_phrase",
     "enum_membership", "grounded_selection"}
)
# Arithmetic equation regex: captures A OP B = C
_ARITHMETIC_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[+\-*×÷/]\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)


# ---------------------------------------------------------------------------
# Repo root
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
    """Return the default Exp 246 artifact path."""
    return get_repo_root() / "results" / "experiment_246_results.json"


def default_checkpoint_dir() -> Path:
    """Return the default Exp 246 checkpoint directory."""
    return get_repo_root() / "results" / "checkpoints" / "experiment_246"


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write an artifact with parent-dir creation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Cohort loading — reuse checked-in prior artifacts verbatim
# ---------------------------------------------------------------------------


def load_gsm8k_cohort(
    path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the GSM8K cohort from the checked-in Exp 235 artifact.

    Returns (cases, meta) where cases preserves case_id and prompt_seeds from
    the prior run so the Exp 246 benchmark stays paired to Exp 219/235.

    Args:
        path: Override path for the Exp 235 artifact (default:
            ``results/experiment_235_results.json`` under the repo root).

    Raises:
        ValueError: When the artifact does not contain a ``gsm8k_semantic``
            benchmark block.
    """
    artifact_path = path or (get_repo_root() / _SOURCE_ARTIFACT_GSM8K)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if str(payload.get("benchmark")) != "gsm8k_semantic":
        raise ValueError(
            f"Expected {artifact_path} to contain benchmark='gsm8k_semantic'; "
            f"got {payload.get('benchmark')!r}"
        )
    cohort_block = payload.get("cohort", {})
    cases = [dict(c) for c in list(cohort_block.get("cases", []))]
    metadata = payload.get("metadata", {})
    display_path = _display_path(artifact_path)
    return cases, {
        "benchmark": "gsm8k_semantic",
        "source_artifact": display_path,
        "sample_seed": int(metadata.get("sample_seed", 218)),
        "sample_size": int(metadata.get("sample_size", len(cases))),
        "case_count": int(cohort_block.get("case_count", len(cases))),
        "source_experiment": int(payload.get("experiment", 235)),
    }


def load_constraint_ir_cohort(
    path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the constraint-IR cohort from the checked-in Exp 221 artifact.

    Returns (cases, meta) with the same shape as ``load_gsm8k_cohort``.

    Args:
        path: Override path for the Exp 221 artifact (default:
            ``results/experiment_221_results.json`` under the repo root).

    Raises:
        ValueError: When the artifact does not contain a ``constraint_ir``
            benchmark block.
    """
    artifact_path = path or (get_repo_root() / _SOURCE_ARTIFACT_CONSTRAINT_IR)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if str(payload.get("benchmark")) != "constraint_ir":
        raise ValueError(
            f"Expected {artifact_path} to contain benchmark='constraint_ir'; "
            f"got {payload.get('benchmark')!r}"
        )
    cohort_block = payload.get("cohort", {})
    cases = [dict(c) for c in list(cohort_block.get("cases", []))]
    metadata = payload.get("metadata", {})
    display_path = _display_path(artifact_path)
    return cases, {
        "benchmark": "constraint_ir",
        "source_artifact": display_path,
        "sample_seed": int(metadata.get("sample_seed", 218)),
        "sample_size": int(metadata.get("sample_size", len(cases))),
        "case_count": int(cohort_block.get("case_count", len(cases))),
        "source_experiment": int(payload.get("experiment", 221)),
    }


def _display_path(path: Path) -> str:
    """Return a repo-relative path string when possible."""
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Checkpoint helpers — identical interface to Exp 218 / Exp 235
# ---------------------------------------------------------------------------


def checkpoint_path(
    checkpoint_dir: Path,
    *,
    benchmark: str,
    model_name: str,
    mode: str,
) -> Path:
    """Return the per-cell checkpoint path for (benchmark, model, mode)."""
    return checkpoint_dir / (
        f"{safe_slug(benchmark)}__{safe_slug(model_name)}__{safe_slug(mode)}.json"
    )


def load_checkpoint(path: Path, expected_case_ids: list[str]) -> dict[str, Any]:
    """Load a checkpoint when the cohort metadata still matches.

    Returns a fresh empty state when the file is missing or the case-id list
    has changed so stale checkpoints are never silently reused.
    """
    fresh: dict[str, Any] = {
        "case_ids": list(expected_case_ids),
        "results_by_case": {},
    }
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
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Resume-aware mode runner — identical interface to Exp 218 / Exp 235
# ---------------------------------------------------------------------------


def run_mode(
    *,
    benchmark: str,
    model_name: str,
    mode: str,
    cases: list[dict[str, Any]],
    checkpoint_dir: Path,
    execute_case: Any,
) -> list[dict[str, Any]]:
    """Execute one benchmark/model/mode cell with checkpoint resume support.

    Already-completed cases are loaded from the checkpoint rather than
    re-executed, so interrupted runs resume cleanly without scrambling the
    case order or duplicating results.

    Args:
        benchmark: Benchmark slice name (``gsm8k_semantic`` or
            ``constraint_ir``).
        model_name: Short model display name (e.g. ``"Qwen3.5-0.8B"``).
        mode: One of ``baseline``, ``verify_only``, or ``verify_repair``.
        cases: Ordered list of case dicts from the loaded cohort.
        checkpoint_dir: Directory where per-cell checkpoint files are stored.
        execute_case: Callable ``(case) -> dict`` returning a result dict for
            one case.  Must not raise; should record any errors in the result.

    Returns:
        Ordered list of per-case result dicts aligned with the input ``cases``
        list.
    """
    case_ids = [str(case["case_id"]) for case in cases]
    ckpt_path = checkpoint_path(
        checkpoint_dir,
        benchmark=benchmark,
        model_name=model_name,
        mode=mode,
    )
    checkpoint = load_checkpoint(ckpt_path, case_ids)
    results_by_case: dict[str, Any] = dict(checkpoint["results_by_case"])

    for case in cases:
        case_id = str(case["case_id"])
        if case_id in results_by_case:
            continue
        result = dict(execute_case(case))
        result.setdefault("case_id", case_id)
        result.setdefault("mode", mode)
        results_by_case[case_id] = result
        save_checkpoint(
            ckpt_path,
            {
                "benchmark": benchmark,
                "model_name": model_name,
                "mode": mode,
                "case_ids": case_ids,
                "results_by_case": results_by_case,
            },
        )

    return [dict(results_by_case[case_id]) for case_id in case_ids]


# ---------------------------------------------------------------------------
# Formal claim extraction — pure, deterministic, no model calls
# ---------------------------------------------------------------------------


def extract_formal_claims_from_response(
    response: str,
    *,
    case: dict[str, Any],
    benchmark: str,
) -> list[dict[str, Any]]:
    """Extract formal claim dicts from a model response for route verification.

    For ``gsm8k_semantic``: scans the response text for explicit arithmetic
    equations of the form ``A OP B = C`` and produces one arithmetic claim per
    match.

    For ``constraint_ir``: converts each entry in ``gold_atomic_constraints``
    into the narrowest deterministic solver route that covers its type.
    Constraint types without a safe deterministic normalization are skipped
    rather than invented.

    Args:
        response: Raw model response text.
        case: Case dict including ``gold_atomic_constraints`` (constraint_ir)
            or question metadata (gsm8k_semantic).
        benchmark: ``"gsm8k_semantic"`` or ``"constraint_ir"``.

    Returns:
        List of claim dicts compatible with
        ``FormalClaim`` / ``normalize_claim``.  Each dict has at minimum:
        ``claim_id``, ``claim_text``, ``candidate_solver_route``,
        ``formalization_status``, ``relation_type``, ``operands``,
        ``target``, and ``bound_variables``.
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
        claims.append(
            {
                "claim_id": f"arith-{match_idx}",
                "claim_text": match.group(0),
                "candidate_solver_route": "arithmetic",
                "formalization_status": "formalized",
                "relation_type": "equation",
                "operands": [a, b, result],
                "target": "arithmetic_result",
                "bound_variables": [],
            }
        )
    return claims


def _extract_constraint_ir_claims(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert gold_atomic_constraints into formal claim dicts."""
    claims: list[dict[str, Any]] = []
    for idx, constraint in enumerate(list(case.get("gold_atomic_constraints", []))):
        claim = _constraint_to_formal_claim(dict(constraint), idx)
        if claim is not None:
            claims.append(claim)
    return claims


def _constraint_to_formal_claim(
    constraint: dict[str, Any],
    idx: int,
) -> dict[str, Any] | None:
    """Map one gold_atomic_constraint to a formal claim dict, or None.

    Returns ``None`` for constraint types that cannot be safely normalized
    into any of the five supported solver routes.
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
                # operands: [required_count, observed_count_placeholder]
                # observed count is 0 here; live runner fills it in
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
        # value shape not normalizable
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

    # Constraint type not normalizable into a supported solver route
    return None


# ---------------------------------------------------------------------------
# Route-summary aggregation
# ---------------------------------------------------------------------------


def build_route_summary(claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-claim route decisions into a compact summary.

    Args:
        claims: Flat list of per-claim dicts from run results.  Each dict
            must have at minimum a ``route`` and a ``verdict`` key.

    Returns:
        Dict with keys ``by_route``, ``by_verdict``, ``total_claims``, and
        ``abstain_rate`` (fraction in [0, 1]).  This is the schema Exp 247
        will consume directly.
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
    all_claims: list[dict[str, Any]] = []
    for run in runs:
        for claim in list(run.get("formal_claims", [])):
            all_claims.append(dict(claim))
    return all_claims


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
    plus paired deltas.  The same shape is consumed by Exp 247 artifact
    merging without post-hoc reshaping.

    Args:
        baseline_runs: Per-case baseline results.  Each dict must have
            ``correct`` (bool).
        verify_only_runs: Per-case verify-only results.  Each dict must have
            ``flagged`` (bool) and ``accepted_correct`` (bool).
        verify_repair_runs: Per-case verify-repair results.  Each dict must
            have ``correct`` (bool), ``repaired`` (bool), and ``n_repairs``
            (int).

    Returns:
        Dict with ``baseline``, ``verify_only``, ``verify_repair``, and
        ``paired_deltas`` blocks.
    """
    n_cases = len(baseline_runs)
    if n_cases == 0:
        empty: dict[str, Any] = {
            "baseline": {"n_cases": 0, "accuracy": 0.0},
            "verify_only": {"n_cases": 0, "n_flagged": 0},
            "verify_repair": {"n_cases": 0, "repair_yield": 0.0},
            "paired_deltas": {"verify_only_minus_baseline": 0.0, "repair_minus_baseline": 0.0},
        }
        return empty

    n_correct_baseline = sum(1 for r in baseline_runs if r.get("correct"))
    baseline_accuracy = n_correct_baseline / n_cases

    n_wrong = n_cases - n_correct_baseline
    n_flagged = sum(1 for r in verify_only_runs if r.get("flagged"))
    n_accepted_correct = sum(1 for r in verify_only_runs if r.get("accepted_correct"))
    verify_accuracy = n_accepted_correct / n_cases

    n_repaired = sum(1 for r in verify_repair_runs if r.get("repaired"))
    repair_accuracy = sum(1 for r in verify_repair_runs if r.get("correct")) / n_cases
    repair_yield = round(n_repaired / n_wrong, 6) if n_wrong > 0 else 0.0

    return {
        "baseline": {
            "n_cases": n_cases,
            "accuracy": baseline_accuracy,
            "n_correct": n_correct_baseline,
            "mean_latency_seconds": _round_mean(
                [float(r.get("latency_seconds", 0.0)) for r in baseline_runs]
            ),
            "mean_prompt_tokens": _round_mean(
                [float(r.get("prompt_tokens", 0.0)) for r in baseline_runs]
            ),
            "mean_response_tokens": _round_mean(
                [float(r.get("response_tokens", 0.0)) for r in baseline_runs]
            ),
            "mean_total_tokens": _round_mean(
                [float(r.get("total_tokens", 0.0)) for r in baseline_runs]
            ),
        },
        "verify_only": {
            "n_cases": n_cases,
            "n_flagged": n_flagged,
            "accuracy": verify_accuracy,
            "mean_latency_seconds": _round_mean(
                [float(r.get("latency_seconds", 0.0)) for r in verify_only_runs]
            ),
            "mean_total_tokens": _round_mean(
                [float(r.get("total_tokens", 0.0)) for r in verify_only_runs]
            ),
        },
        "verify_repair": {
            "n_cases": n_cases,
            "accuracy": repair_accuracy,
            "n_repaired": n_repaired,
            "repair_yield": repair_yield,
            "avg_repairs": _round_mean(
                [float(r.get("n_repairs", 0)) for r in verify_repair_runs]
            ),
            "mean_latency_seconds": _round_mean(
                [float(r.get("latency_seconds", 0.0)) for r in verify_repair_runs]
            ),
            "mean_total_tokens": _round_mean(
                [float(r.get("total_tokens", 0.0)) for r in verify_repair_runs]
            ),
        },
        "paired_deltas": {
            "verify_only_minus_baseline": round(verify_accuracy - baseline_accuracy, 6),
            "repair_minus_baseline": round(repair_accuracy - baseline_accuracy, 6),
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
) -> dict[str, Any]:
    """Build the stable Exp 246 artifact payload for Exp 247 consumption.

    The schema is intentionally minimal and self-describing.  Every field that
    Exp 247 might need for honest reporting is present; nothing is implied.

    Args:
        output_path: Where the artifact will be written.
        gsm8k_cohort: Case list from the Exp 235 cohort.
        gsm8k_cohort_meta: Metadata from ``load_gsm8k_cohort``.
        constraint_ir_cohort: Case list from the Exp 221 cohort.
        constraint_ir_cohort_meta: Metadata from
            ``load_constraint_ir_cohort``.
        gsm8k_paired_runs: Ordered paired-run dicts for the GSM8K slice.
        constraint_ir_paired_runs: Ordered paired-run dicts for the
            constraint-IR slice.
        gsm8k_route_summary: Aggregated route summary from
            ``build_route_summary`` over GSM8K verify runs.
        constraint_ir_route_summary: Same for constraint-IR verify runs.
        gsm8k_statistics: Per-model summary stats for the GSM8K slice.
        constraint_ir_statistics: Per-model summary stats for the
            constraint-IR slice.
        started_at: ISO-8601 UTC start timestamp string.
        finished_at: ISO-8601 UTC end timestamp string.
        runtime_seconds: Wall-clock seconds for the run.
        checkpoint_dir: Directory holding per-cell checkpoint files.
        max_repairs: Maximum verify-repair iterations per case.
        inference_mode: ``"simulated"``, ``"live_cpu"``, or ``"live_gpu"``.

    Returns:
        Dict ready for ``json.dumps`` / ``write_artifact``.
    """
    checkpoint_pattern = (
        "results/checkpoints/experiment_246/<benchmark>__<model>__<mode>.json"
    )

    def _cohort_block(
        cohort: list[dict[str, Any]],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "case_count": len(cohort),
            "case_ids": [str(c["case_id"]) for c in cohort],
            "source_artifact": str(meta.get("source_artifact", "")),
            "source_experiment": int(meta.get("source_experiment", 0)),
            "sample_seed": int(meta.get("sample_seed", 218)),
        }

    return {
        "experiment": EXPERIMENT,
        "title": "Solver-semantic live benchmark runner (Exp 246)",
        "run_date": RUN_DATE,
        "schema": {
            "artifact": SCHEMA_ARTIFACT,
            "benchmark_slices": ["gsm8k_semantic", "constraint_ir"],
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
            "max_repairs": max_repairs,
            "inference_mode": inference_mode,
            "force_live": os.environ.get("CARNOT_FORCE_LIVE") == "1",
            "force_cpu": os.environ.get("CARNOT_FORCE_CPU") == "1",
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
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the Exp 246 CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Exp 246 solver-semantic live benchmark runner.  "
            "Creates checkpoints and the artifact schema for Exp 247 execution."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Artifact output path (default: results/experiment_246_results.json).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        type=Path,
        default=default_checkpoint_dir(),
        help="Checkpoint directory (default: results/checkpoints/experiment_246).",
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


# ---------------------------------------------------------------------------
# Live execution helpers (pragma: no cover — Exp 247 path)
# ---------------------------------------------------------------------------


def _live_inference_mode() -> str:  # pragma: no cover
    if os.environ.get("CARNOT_FORCE_LIVE") == "1":
        return "live_cpu" if os.environ.get("CARNOT_FORCE_CPU") == "1" else "live_gpu"
    return "simulated"


def _verify_formal_claims_batch(
    claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:  # pragma: no cover
    """Run FormalClaimVerifier over a list of claim dicts.

    Returns a list of verdict dicts suitable for embedding in per-case
    results under the ``formal_claims`` key.
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
    pipeline: Any,
    max_repairs: int,
    prompt_seed: int,
) -> dict[str, Any]:
    """Execute one GSM8K semantic case in live mode (Exp 247 path).

    Generates a response, extracts formal claims, verifies them, and
    records baseline / verify-only / verify-repair outcomes.
    """
    import time as _time
    from carnot.inference.model_loader import generate  # type: ignore[import-untyped]

    question = str(case.get("question", ""))
    ground_truth = int(case.get("ground_truth", -9999))
    prompt = f"Solve this math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
    t0 = _time.time()
    response = str(generate(model, tokenizer, prompt, max_new_tokens=256))
    latency = _time.time() - t0

    # Extract and verify formal claims
    raw_claims = extract_formal_claims_from_response(
        response, case=case, benchmark="gsm8k_semantic"
    )
    formal_claims = _verify_formal_claims_batch(raw_claims)

    # Parse answer
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

    # verify_only: flag when any claim is violated
    flagged = any(c.get("verdict") == "violated" for c in formal_claims)
    accepted_correct = correct and not flagged
    vo_result = {**base_result, "flagged": flagged, "accepted_correct": accepted_correct}
    if mode == "verify_only":
        return vo_result

    # verify_repair: attempt repair when flagged
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
    return {
        **vo_result,
        "response": response,
        "predicted": predicted,
        "correct": correct,
        "repaired": repaired,
        "n_repairs": n_repairs,
    }


def _execute_constraint_ir_case_live(  # pragma: no cover
    case: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    mode: str,
    pipeline: Any,
    max_repairs: int,
    prompt_seed: int,
) -> dict[str, Any]:
    """Execute one constraint-IR case in live mode (Exp 247 path).

    Generates a response, extracts formal claims from gold constraints,
    verifies them with the formal claim verifier, and records outcomes.
    """
    import time as _time
    from carnot.inference.model_loader import generate  # type: ignore[import-untyped]

    prompt = str(case.get("prompt", ""))
    t0 = _time.time()
    response = str(generate(model, tokenizer, prompt, max_new_tokens=256))
    latency = _time.time() - t0

    raw_claims = extract_formal_claims_from_response(
        response, case=case, benchmark="constraint_ir"
    )
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
            c.get("failure_detail", "")
            for c in formal_claims
            if c.get("verdict") == "violated"
        ]
        for _ in range(max_repairs):
            repair_prompt = (
                f"{prompt}\n\nViolated constraints: {violated_details!r}. "
                "Please revise your answer.\n\nAnswer:"
            )
            response = str(generate(model, tokenizer, repair_prompt, max_new_tokens=256))
            raw_claims = extract_formal_claims_from_response(
                response, case=case, benchmark="constraint_ir"
            )
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


def _run_benchmark_for_model_live(  # pragma: no cover
    *,
    benchmark: str,
    model_spec: dict[str, str],
    cases: list[dict[str, Any]],
    checkpoint_dir: Path,
    max_repairs: int,
) -> list[dict[str, Any]]:
    """Run all three modes for one model on one benchmark (Exp 247 path)."""
    import gc as _gc
    from carnot.inference.model_loader import load_model  # type: ignore[import-untyped]

    hf_id = model_spec["hf_id"]
    device = "cpu" if os.environ.get("CARNOT_FORCE_CPU") == "1" else "cuda"
    model, tokenizer = load_model(hf_id, device=device)
    pipeline = None  # placeholder; Exp 247 wires in the VerifyRepairPipeline

    prompt_seed_map = {
        str(case["case_id"]): int(
            (case.get("prompt_seeds") or {}).get("baseline", 0)
        )
        for case in cases
    }

    execute_fn: Any
    if benchmark == "gsm8k_semantic":
        execute_fn = _execute_gsm8k_case_live
    else:
        execute_fn = _execute_constraint_ir_case_live

    paired: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        def _make_executor(m: str) -> Any:
            def _execute(case: dict[str, Any]) -> dict[str, Any]:
                return execute_fn(
                    case,
                    model=model,
                    tokenizer=tokenizer,
                    mode=m,
                    pipeline=pipeline,
                    max_repairs=max_repairs,
                    prompt_seed=prompt_seed_map.get(str(case["case_id"]), 0),
                )
            return _execute

        mode_results = run_mode(
            benchmark=benchmark,
            model_name=model_spec["name"],
            mode=mode,
            cases=cases,
            checkpoint_dir=checkpoint_dir,
            execute_case=_make_executor(mode),
        )
        paired.append(
            {
                "benchmark": benchmark,
                "model_name": model_spec["name"],
                "hf_id": model_spec["hf_id"],
                "mode": mode,
                "cases": mode_results,
                "summary": {"n_cases": len(mode_results)},
            }
        )

    del model, tokenizer
    _gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return paired


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """CLI entry point for the Exp 246 runner.

    In simulated mode (``CARNOT_FORCE_LIVE`` not set) this writes an artifact
    with empty paired_runs so Exp 247 can take over live execution without
    redefining the schema.
    """
    parser = build_parser()
    args = parser.parse_args()

    output_path: Path = args.output
    checkpoint_dir: Path = args.checkpoint_dir
    max_repairs: int = args.max_repairs
    inference_mode = _live_inference_mode()

    started_at = utc_now()
    t_start = time.time()

    # Load cohorts from checked-in prior artifacts
    gsm8k_cohort, gsm8k_cohort_meta = load_gsm8k_cohort(args.gsm8k_artifact)
    constraint_ir_cohort, constraint_ir_cohort_meta = load_constraint_ir_cohort(
        args.constraint_ir_artifact
    )

    gsm8k_paired_runs: list[dict[str, Any]] = []
    gsm8k_statistics: dict[str, Any] = {}
    gsm8k_all_claims: list[dict[str, Any]] = []

    constraint_ir_paired_runs: list[dict[str, Any]] = []
    constraint_ir_statistics: dict[str, Any] = {}
    constraint_ir_all_claims: list[dict[str, Any]] = []

    if inference_mode != "simulated":
        for model_spec in MODEL_SPECS:
            # GSM8K slice
            model_paired = _run_benchmark_for_model_live(
                benchmark="gsm8k_semantic",
                model_spec=model_spec,
                cases=gsm8k_cohort,
                checkpoint_dir=checkpoint_dir,
                max_repairs=max_repairs,
            )
            gsm8k_paired_runs.extend(model_paired)

            # Constraint-IR slice
            ir_paired = _run_benchmark_for_model_live(
                benchmark="constraint_ir",
                model_spec=model_spec,
                cases=constraint_ir_cohort,
                checkpoint_dir=checkpoint_dir,
                max_repairs=max_repairs,
            )
            constraint_ir_paired_runs.extend(ir_paired)

        # Aggregate statistics per model
        for model_spec in MODEL_SPECS:
            name = model_spec["name"]
            _runs = {
                m: next(
                    (
                        r["cases"]
                        for r in gsm8k_paired_runs
                        if r["model_name"] == name and r["mode"] == m
                    ),
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
                collect_all_claims_from_runs(
                    _runs["verify_only"] + _runs["verify_repair"]
                )
            )

            _ir_runs = {
                m: next(
                    (
                        r["cases"]
                        for r in constraint_ir_paired_runs
                        if r["model_name"] == name and r["mode"] == m
                    ),
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
                collect_all_claims_from_runs(
                    _ir_runs["verify_only"] + _ir_runs["verify_repair"]
                )
            )

    gsm8k_route_summary = build_route_summary(gsm8k_all_claims)
    constraint_ir_route_summary = build_route_summary(constraint_ir_all_claims)

    finished_at = utc_now()
    runtime_seconds = time.time() - t_start

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
    )
    write_artifact(output_path, payload)
    print(f"Exp 246 artifact written to {output_path}")
    print(f"  GSM8K cohort: {gsm8k_cohort_meta['case_count']} cases")
    print(f"  Constraint-IR cohort: {constraint_ir_cohort_meta['case_count']} cases")
    print(f"  Inference mode: {inference_mode}")


if __name__ == "__main__":
    main()
