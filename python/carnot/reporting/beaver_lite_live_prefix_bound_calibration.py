"""Exp 1482 BEAVER-lite live-prefix bound calibration.

This module stays deliberately narrow.  It reuses existing telemetry manifests
as fixed live prefixes, computes terminal-prefix unsafe mass from recorded
top-k logprobs, and falls back to the existing deterministic BEAVER-lite mock
path when the live rows are unavailable.  It does not generate new model text
or implement a broad external benchmark runner.

Spec: REQ-VERIFY-1482, SCENARIO-VERIFY-1482.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.verify.beaver_lite import (
    BEAVERLiteBounder,
    FinalIntegerConstraint,
    MockLogprobProvider,
    logsumexp,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
BENCHMARK_FAMILY = "BEAVER-style deterministic bounds"
SCHEMA = "beaver_lite_live_prefix_bound_calibration_v1"
EXPERIMENT = "1482_beaver_lite_live_prefix_bound_calibration"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1482_beaver_lite_live_prefix_bound_calibration.json"
)
DEFAULT_EXP1480_ARTIFACT_PATH = (
    REPO_ROOT / "results" / "experiment_1480_live_sota_balanced_telemetry_v2.json"
)
DEFAULT_EXP1480_MANIFEST_PATH = (
    REPO_ROOT / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl"
)
DEFAULT_EXP1468_ARTIFACT_PATH = (
    REPO_ROOT / "results" / "experiment_1468_live_sota_logprob_telemetry_preflight.json"
)
DEFAULT_EXP1468_MANIFEST_PATH = REPO_ROOT / "results" / "live_sota_telemetry_manifest_1468.jsonl"

MANDATED_SOTA_GGUF_MODELS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
}
MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "preferred_live_prefix_logprob_source",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "optional_live_prefix_logprob_source",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "optional_live_prefix_logprob_source",
    },
)
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "benchmark_family",
    "model_specs",
    "constraints_evaluated",
    "prefix_closed_constraints",
    "unsafe_mass_bounds",
    "empirical_violation_rates",
    "bound_is_sound",
    "bound_violations",
    "calibration_tightness_summary",
    "mock_or_live_logprobs",
    "broad_benchmark_deferred",
    "honest_verdict",
}
LIVE_LINEAGES = {
    "live_exp1480",
    "live_exp1480_plus_exp1468",
    "live_exp1468",
    "mock_logprobs",
}
SOURCE_ARTIFACTS = {
    "live_exp1480": ["results/experiment_1480_live_sota_balanced_telemetry_v2.json"],
    "live_exp1480_plus_exp1468": [
        "results/experiment_1480_live_sota_balanced_telemetry_v2.json",
        "results/experiment_1468_live_sota_logprob_telemetry_preflight.json",
    ],
    "live_exp1468": ["results/experiment_1468_live_sota_logprob_telemetry_preflight.json"],
    "mock_logprobs": [],
}
SUPPORTED_LIVE_FAMILIES = {
    "arithmetic_word_problem",
    "constraint_check",
    "fover_claim",
    "fover_style",
    "gsm8k_style",
}


def write_in_progress_artifact(path: str | Path = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-VERIFY-1482-1: write the durable startup artifact before row loading."""

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "spec": ["REQ-VERIFY-1482", "SCENARIO-VERIFY-1482"],
        "status": "in_progress",
        "benchmark_family": BENCHMARK_FAMILY,
        "model_specs": [dict(item) for item in MODEL_SPECS],
        "constraints_evaluated": 0,
        "prefix_closed_constraints": [],
        "unsafe_mass_bounds": [],
        "empirical_violation_rates": [],
        "bound_is_sound": False,
        "bound_violations": [],
        "calibration_tightness_summary": {"p50_slack": None, "p90_slack": None, "max_slack": None},
        "mock_or_live_logprobs": "pending",
        "broad_benchmark_deferred": True,
        "honest_verdict": "in_progress",
    }
    return _write_json(Path(path), artifact)


def compatible_exp1480_rows(
    artifact_path: str | Path = DEFAULT_EXP1480_ARTIFACT_PATH,
    manifest_path: str | Path = DEFAULT_EXP1480_MANIFEST_PATH,
    limit: int = 18,
) -> list[dict[str, Any]]:
    """Return compatible Exp 1480 live-prefix rows without fresh inference."""

    return _compatible_live_rows(artifact_path, manifest_path, limit)


def evaluate_live_prefix_row(row: Mapping[str, Any], constraint_index: int) -> dict[str, Any]:
    """Score one recorded live prefix using terminal top-k unsafe mass.

    Spec: REQ-VERIFY-1482-3, REQ-VERIFY-1482-4.
    """

    constraint = FinalIntegerConstraint()
    prefix = "".join(str(token) for token in row["token_texts"][:-1])
    final_top_logprobs = dict(row["top_logprobs"][-1])
    unsafe_logprobs = [
        float(logprob)
        for token, logprob in final_top_logprobs.items()
        if constraint.prefix_violates(prefix + str(token), terminal=True)
    ]
    unsafe_mass_bound = 0.0 if not unsafe_logprobs else math.exp(logsumexp(unsafe_logprobs))
    unsafe_mass_bound = max(0.0, min(1.0, unsafe_mass_bound))
    empirical_violation_rate = (
        1.0 if constraint.prefix_violates(str(row["response_text"]), terminal=True) else 0.0
    )
    bound_gap = unsafe_mass_bound - empirical_violation_rate
    return {
        "constraint": _constraint_record(constraint_index, row),
        "unsafe_mass_bound": unsafe_mass_bound,
        "empirical_violation_rate": empirical_violation_rate,
        "bound_gap": bound_gap,
        "n_live_topk_alternatives": len(final_top_logprobs),
    }


def build_artifact(
    *,
    constraints: Sequence[Mapping[str, Any]],
    unsafe_mass_bounds: Sequence[float],
    empirical_violation_rates: Sequence[float],
    mock_or_live_logprobs: str,
    models_used: Sequence[str],
    n_live_topk_alternatives: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build and validate the terminal Exp 1482 artifact."""

    bounds = [float(value) for value in unsafe_mass_bounds]
    rates = [float(value) for value in empirical_violation_rates]
    slacks = [bound - rate for bound, rate in zip(bounds, rates)]
    violations = [
        {
            "constraint_id": str(constraint.get("constraint_id", index)),
            "unsafe_mass_bound": bound,
            "empirical_violation_rate": rate,
        }
        for index, (constraint, bound, rate) in enumerate(zip(constraints, bounds, rates))
        if rate > bound + 1e-12
    ]
    sound = not violations and all(0.0 <= bound <= 1.0 for bound in bounds)
    if not sound:
        honest_verdict = "bound_violated_bug"
    elif mock_or_live_logprobs == "mock_logprobs":
        honest_verdict = "sound_bound_mock_logprobs_calibrated"
    else:
        honest_verdict = f"sound_bound_{mock_or_live_logprobs}_calibrated"

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "spec": ["REQ-VERIFY-1482", "SCENARIO-VERIFY-1482"],
        "status": "complete",
        "benchmark_family": BENCHMARK_FAMILY,
        "model_specs": [dict(item) for item in MODEL_SPECS],
        "models_used": list(dict.fromkeys(str(model) for model in models_used)),
        "source_artifacts": SOURCE_ARTIFACTS.get(mock_or_live_logprobs, []),
        "constraints_evaluated": len(constraints),
        "prefix_closed_constraints": [dict(item) for item in constraints],
        "unsafe_mass_bounds": bounds,
        "empirical_violation_rates": rates,
        "bound_is_sound": sound,
        "bound_violations": violations,
        "calibration_tightness_summary": _tightness_summary(slacks),
        "mock_or_live_logprobs": mock_or_live_logprobs,
        "n_live_topk_alternatives": (
            list(n_live_topk_alternatives) if n_live_topk_alternatives is not None else []
        ),
        "broad_benchmark_deferred": True,
        "honest_verdict": honest_verdict,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the REQ-VERIFY-1482 schema and soundness gate."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["mock_or_live_logprobs"] not in LIVE_LINEAGES:
        raise ValueError("mock_or_live_logprobs must label live_exp1480, live_exp1468, or mock")
    if artifact["broad_benchmark_deferred"] is not True:
        raise ValueError("broad_benchmark_deferred must remain true")
    n_constraints = int(artifact["constraints_evaluated"])
    constraints = list(artifact["prefix_closed_constraints"])
    bounds = [float(value) for value in artifact["unsafe_mass_bounds"]]
    rates = [float(value) for value in artifact["empirical_violation_rates"]]
    if not 12 <= n_constraints <= 20:
        raise ValueError("constraints_evaluated must be in the 12 to 20 bounded range")
    if not (len(constraints) == len(bounds) == len(rates) == n_constraints):
        raise ValueError("constraint, bound, and empirical-rate counts must match")
    for bound, rate in zip(bounds, rates):
        if not 0.0 <= bound <= 1.0 or not 0.0 <= rate <= 1.0:
            raise ValueError("unsafe mass bounds and empirical rates must be in [0, 1]")
        if rate > bound + 1e-12:
            raise ValueError("empirical violation rate exceeds bound")
    if bool(artifact["bound_violations"]):
        raise ValueError("bound_violations must be empty for a sound terminal artifact")
    if artifact["bound_is_sound"] is not True:
        raise ValueError("bound_is_sound must be true for a validated calibration artifact")


def run(
    output_path: str | Path = DEFAULT_OUT_PATH,
    exp1480_artifact_path: str | Path = DEFAULT_EXP1480_ARTIFACT_PATH,
    exp1480_manifest_path: str | Path = DEFAULT_EXP1480_MANIFEST_PATH,
    exp1468_artifact_path: str | Path = DEFAULT_EXP1468_ARTIFACT_PATH,
    exp1468_manifest_path: str | Path = DEFAULT_EXP1468_MANIFEST_PATH,
    limit: int = 18,
    mock_top_k: int = 10,
) -> dict[str, Any]:
    """Run the bounded calibration and write the terminal artifact."""

    write_in_progress_artifact(output_path)
    lineage, rows = _select_live_rows(
        exp1480_artifact_path,
        exp1480_manifest_path,
        exp1468_artifact_path,
        exp1468_manifest_path,
        limit,
    )
    if lineage == "mock_logprobs":
        evaluations = [_evaluate_mock_constraint(index, mock_top_k) for index in range(limit)]
        models_used: list[str] = []
    else:
        evaluations = [
            evaluate_live_prefix_row(row, constraint_index=index) for index, row in enumerate(rows)
        ]
        models_used = [str(row["hf_id"]) for row in rows]

    artifact = build_artifact(
        constraints=[evaluation["constraint"] for evaluation in evaluations],
        unsafe_mass_bounds=[float(evaluation["unsafe_mass_bound"]) for evaluation in evaluations],
        empirical_violation_rates=[
            float(evaluation["empirical_violation_rate"]) for evaluation in evaluations
        ],
        mock_or_live_logprobs=lineage,
        models_used=models_used,
        n_live_topk_alternatives=[
            int(evaluation["n_live_topk_alternatives"]) for evaluation in evaluations
        ],
    )
    return _write_json(Path(output_path), artifact)


def _select_live_rows(
    exp1480_artifact_path: str | Path,
    exp1480_manifest_path: str | Path,
    exp1468_artifact_path: str | Path,
    exp1468_manifest_path: str | Path,
    limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    exp1480_rows = compatible_exp1480_rows(exp1480_artifact_path, exp1480_manifest_path, limit)
    if len(exp1480_rows) >= 12:
        return "live_exp1480", exp1480_rows[:limit]
    exp1468_rows = _compatible_live_rows(exp1468_artifact_path, exp1468_manifest_path, limit)
    combined = (exp1480_rows + exp1468_rows)[:limit]
    if len(combined) >= 12 and exp1480_rows:
        return "live_exp1480_plus_exp1468", combined
    if len(exp1468_rows) >= 12:
        return "live_exp1468", exp1468_rows[:limit]
    return "mock_logprobs", []


def _compatible_live_rows(
    artifact_path: str | Path,
    manifest_path: str | Path,
    limit: int,
) -> list[dict[str, Any]]:
    artifact_file = Path(artifact_path)
    manifest_file = Path(manifest_path)
    if not artifact_file.exists() or not manifest_file.exists():
        return []
    summary = json.loads(artifact_file.read_text(encoding="utf-8"))
    if not _summary_has_compatible_logprobs(summary):
        return []
    rows: list[dict[str, Any]] = []
    for line in manifest_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if _row_has_compatible_live_prefix_logprobs(row):
            rows.append(row)
        if len(rows) == limit:
            break
    return rows


def _summary_has_compatible_logprobs(summary: Mapping[str, Any]) -> bool:
    return (
        summary.get("status") == "complete"
        and summary.get("live_sota_model_inference_used") is True
        and summary.get("topk_logprobs_available") is True
        and any(model in MANDATED_SOTA_GGUF_MODELS for model in summary.get("models_used", []))
    )


def _row_has_compatible_live_prefix_logprobs(row: Mapping[str, Any]) -> bool:
    token_logprobs = row.get("token_logprobs")
    top_logprobs = row.get("top_logprobs")
    token_texts = row.get("token_texts")
    constraint = FinalIntegerConstraint()
    return (
        row.get("generation_source") == "live_sota_llamacpp"
        and row.get("hf_id") in MANDATED_SOTA_GGUF_MODELS
        and str(row.get("family")) in SUPPORTED_LIVE_FAMILIES
        and row.get("format_valid", True) is True
        and row.get("response_text_available") is True
        and row.get("token_logprobs_available") is True
        and row.get("topk_alternatives_available") is True
        and isinstance(token_logprobs, list)
        and len(token_logprobs) > 0
        and isinstance(top_logprobs, list)
        and len(top_logprobs) > 0
        and isinstance(top_logprobs[-1], dict)
        and len(top_logprobs[-1]) > 0
        and isinstance(token_texts, list)
        and len(token_texts) > 0
        and constraint.is_satisfied(str(row.get("response_text", "")))
    )


def _evaluate_mock_constraint(index: int, top_k: int) -> dict[str, Any]:
    result = BEAVERLiteBounder(provider=MockLogprobProvider(), top_k=top_k).bound_prefix_violation(
        f"Mock Exp 1482 terminal-prefix calibration case {index + 1}."
    )
    return {
        "constraint": _mock_constraint_record(index),
        "unsafe_mass_bound": result.upper_bound,
        "empirical_violation_rate": result.empirical_rate,
        "bound_gap": result.bound_gap,
        "n_live_topk_alternatives": 0,
    }


def _constraint_record(index: int, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "constraint_id": f"exp1482_case_{index + 1:03d}",
        "constraint_type": _constraint_type(row),
        "description": "terminal live prefix must end with an integer in [0, 9999]",
        "prefix_closed": True,
        "terminal_only": True,
        "source_case_id": str(row["case_id"]),
        "source_family": str(row["family"]),
        "expected_answer": str(row["expected_answer"]),
        "prompt": str(row["prompt"]),
    }


def _mock_constraint_record(index: int) -> dict[str, Any]:
    return {
        "constraint_id": f"exp1482_mock_case_{index + 1:03d}",
        "constraint_type": "arithmetic",
        "description": "terminal mock prefix must end with an integer in [0, 9999]",
        "prefix_closed": True,
        "terminal_only": True,
        "source_case_id": f"mock_case_{index + 1:03d}",
        "source_family": "mock_arithmetic",
        "expected_answer": str((index + 7) % 10000),
        "prompt": f"Mock Exp 1482 terminal-prefix calibration case {index + 1}.",
    }


def _constraint_type(row: Mapping[str, Any]) -> str:
    family = str(row.get("family", ""))
    if family in {"arithmetic_word_problem", "gsm8k_style"}:
        return "arithmetic"
    return "certificate"


def _tightness_summary(slacks: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in slacks)
    return {
        "p50_slack": _quantile(ordered, 0.5),
        "p90_slack": _quantile(ordered, 0.9),
        "max_slack": max(ordered),
    }


def _quantile(ordered_values: Sequence[float], quantile: float) -> float:
    position = (len(ordered_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered_values[int(position)])
    lower_value = ordered_values[lower]
    upper_value = ordered_values[upper]
    return float(lower_value + (upper_value - lower_value) * (position - lower))


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "BENCHMARK_FAMILY",
    "DEFAULT_OUT_PATH",
    "MANDATED_SOTA_GGUF_MODELS",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "compatible_exp1480_rows",
    "evaluate_live_prefix_row",
    "run",
    "validate_artifact",
    "write_in_progress_artifact",
]
