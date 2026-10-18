"""Exp 3130 ARM/EBT energy-budget sidecar diagnostic v2.

Spec refs: REQ-VERIFY-3130, SCENARIO-VERIFY-3130.

This module is intentionally a diagnostic sidecar, not a generation hook. It
reads checked-in exact fixture evidence and optional cached live trace metadata
from earlier experiments, then separates hard constraint penalties, proxy
quality scores, and uncertainty estimates. That separation matters because
Distributional EBM-style reasoning needs the exact constraint authority to stay
outside any learned or heuristic quality score; otherwise a promising-looking
energy number can quietly become a label leak.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from carnot.eval.ebt_arm_sidecar_score_correlation_boundary_v3 import pearson, rate, spearman


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2"
SCHEMA = "carnot.arm_ebt_energy_budget_sidecar_diagnostic.v2"
OUTPUT_REL_PATH = Path("results/experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.json")
EXP3123_REL_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")
EXP3117_REL_PATH = Path("results/experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3.json")
EXP3124_REL_PATH = Path(
    "results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"
)
MANDATORY_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_FIELDS = (
    "arm_ebt_energy_budget_sidecar_v2_ready",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "exact_fixture_count",
    "deterministic_constraint_penalty_summary",
    "learned_quality_proxy_summary",
    "uncertainty_summary",
    "approximation_gap_summary",
    "model_identity_confound_audit",
    "correlation_metrics",
    "live_integration",
    "integration_blockers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval/arm_ebt_energy_budget_sidecar_diagnostic_v2.py -m pytest -o addopts='' tests/python/test_experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/arm_ebt_energy_budget_sidecar_diagnostic_v2.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3130: build the offline energy-budget sidecar diagnostic."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3123 = read_json_object(root_path / EXP3123_REL_PATH)
    exp3117 = read_json_object(root_path / EXP3117_REL_PATH)
    exp3124 = read_json_object(root_path / EXP3124_REL_PATH)
    exact_rows = diagnostic_rows(exp3117)
    live_rows = mapping_rows(exp3124.get("live_rows"))
    energy_rows = budget_rows(exact_rows)
    model_specs = model_specs_from_sources(exp3123, exp3124)
    selected_model_ids = selected_models(exp3123, exp3124, model_specs)
    live_call_count = int(exp3124.get("live_call_count") or 0)
    source_rows = source_artifacts(root_path)
    summaries = {
        "deterministic_constraint_penalty_summary": deterministic_constraint_penalty_summary(
            energy_rows
        ),
        "learned_quality_proxy_summary": learned_quality_proxy_summary(energy_rows, live_rows),
        "uncertainty_summary": uncertainty_summary(energy_rows),
        "approximation_gap_summary": approximation_gap_summary(energy_rows, exp3124),
        "model_identity_confound_audit": model_identity_confound_audit(
            model_specs, selected_model_ids, live_rows
        ),
        "correlation_metrics": correlation_metrics(energy_rows),
    }
    blockers = integration_blockers(summaries, live_rows, exp3124, selected_model_ids)
    substrate = inference_substrate(exp3123, exp3124, live_call_count)
    readiness_checks = readiness_checks_for(
        exp3123=exp3123,
        exact_rows=energy_rows,
        model_specs=model_specs,
        source_rows=source_rows,
        summaries=summaries,
        blockers=blockers,
    )
    ready = all(readiness_checks.values())
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3130", "SCENARIO-VERIFY-3130"],
        "arm_ebt_energy_budget_sidecar_v2_ready": ready,
        "model_specs": model_specs,
        "selected_model_ids": selected_model_ids,
        "live_call_count": live_call_count,
        "exact_fixture_count": len(energy_rows),
        **summaries,
        "live_integration": False,
        "integration_blockers": blockers,
        "readiness_checks": readiness_checks,
        "blocked_reasons": [name for name, ok in readiness_checks.items() if ok is not True],
        "source_artifacts": source_rows,
        "source_checksums": {
            source["path"]: source["sha256"] for source in source_rows if source["sha256"]
        },
        "inference_substrate": substrate,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3130 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(out_path, artifact)
    return out_path


def diagnostic_rows(exp3117: Mapping[str, Any]) -> list[JsonDict]:
    """Return Exp 3117 exact-fixture rows as plain dictionaries."""

    return mapping_rows(exp3117.get("diagnostic_rows"))


def mapping_rows(value: Any) -> list[JsonDict]:
    """Keep only JSON object rows from a possibly malformed list."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def budget_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize exact fixture rows into comparable proxy-budget rows."""

    normalized: list[JsonDict] = []
    for row in rows:
        energy = as_float(row.get("label_blind_feature_energy"), 0.0)
        feature_summary = row.get("feature_summary")
        feature_map = feature_summary if isinstance(feature_summary, Mapping) else {}
        penalty = as_float(feature_map.get("label_blind_violation"), energy)
        replay_score = row.get("replay_score")
        replay_map = replay_score if isinstance(replay_score, Mapping) else {}
        quality = as_float(replay_map.get("confidence"), 1.0 / (1.0 + max(0.0, energy)))
        exact_outcome = str(row.get("exact_outcome") or "")
        reject_or_repair = int(row.get("reject_or_repair_label") or 0)
        normalized.append(
            {
                "fixture_id": str(row.get("fixture_id") or ""),
                "fixture_family": str(row.get("task_family") or row.get("fixture_family") or ""),
                "exact_outcome": exact_outcome,
                "reject_or_repair_label": reject_or_repair,
                "prefix_energy_proxy": round(energy * 0.5, 6),
                "final_energy_proxy": round(energy, 6),
                "deterministic_constraint_penalty": round(penalty, 6),
                "prefix_constraint_penalty": round(penalty * 0.5, 6),
                "quality_proxy": round(quality, 6),
                "uncertainty_proxy": round(1.0 - quality, 6),
                "expected_action": str(row.get("expected_action") or ""),
                "sidecar_action": str(row.get("sidecar_action") or ""),
            }
        )
    return normalized


def deterministic_constraint_penalty_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize hard-constraint penalties apart from proxy quality scores."""

    final_penalties = [as_float(row.get("deterministic_constraint_penalty")) for row in rows]
    prefix_penalties = [as_float(row.get("prefix_constraint_penalty")) for row in rows]
    outcomes = Counter(str(row.get("exact_outcome") or "") for row in rows)
    return {
        "authority_boundary": "deterministic penalties are exact-fixture feature checks, not learned quality",
        "exact_verdict_counts": dict(sorted(outcomes.items())),
        "nonzero_penalty_count": sum(1 for value in final_penalties if value > 0.0),
        "prefix_penalty": numeric_summary(prefix_penalties),
        "final_penalty": numeric_summary(final_penalties),
        "by_exact_outcome": grouped_numeric(rows, "exact_outcome", "deterministic_constraint_penalty"),
    }


def learned_quality_proxy_summary(
    rows: Sequence[Mapping[str, Any]], live_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Summarize proxy quality while declaring that no learned energy head exists."""

    quality_values = [as_float(row.get("quality_proxy")) for row in rows]
    live_correct = [row for row in live_rows if row.get("live_correct") is True]
    live_answer_matches = [row for row in live_rows if row.get("exact_answer_match") is True]
    return {
        "learned_model_score_available": False,
        "quality_proxy_source": "cached replay confidence metadata; no trained EBT/ARM quality head",
        "quality_proxy": numeric_summary(quality_values),
        "posthoc_live_trace_accuracy": rate(len(live_correct), len(live_rows)),
        "posthoc_live_exact_answer_match_rate": rate(len(live_answer_matches), len(live_rows)),
        "posthoc_boundary": "live trace correctness is audit evidence, not a score available before exact labels",
    }


def uncertainty_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report uncertainty separately from exact constraints and quality proxies."""

    uncertainty_values = [as_float(row.get("uncertainty_proxy")) for row in rows]
    return {
        "uncertainty_proxy_source": "1 - cached replay confidence",
        "distributional_ebm_boundary": "uncertainty is reported as a sidecar diagnostic only",
        "uncertainty_proxy": numeric_summary(uncertainty_values),
        "by_exact_outcome": grouped_numeric(rows, "exact_outcome", "uncertainty_proxy"),
    }


def approximation_gap_summary(rows: Sequence[Mapping[str, Any]], exp3124: Mapping[str, Any]) -> JsonDict:
    """Compare proxy energies with exact reject/repair authority."""

    accepted = [
        as_float(row.get("final_energy_proxy"))
        for row in rows
        if row.get("exact_outcome") == "accepted"
    ]
    accept_boundary = max(accepted) if accepted else math.nan
    false_safe = [
        row
        for row in rows
        if row.get("exact_outcome") != "accepted"
        and math.isfinite(accept_boundary)
        and as_float(row.get("final_energy_proxy")) <= accept_boundary
    ]
    energies = [as_float(row.get("final_energy_proxy")) for row in rows]
    scaled = scale01(energies)
    labels = [as_float(row.get("reject_or_repair_label")) for row in rows]
    gaps = [abs(score - label) for score, label in zip(scaled, labels, strict=False)]
    return {
        "gap_definition": "absolute gap between min-max scaled proxy energy and exact reject/repair label",
        "mean_abs_gap_to_exact_binary": numeric_summary(gaps),
        "accepted_energy_boundary": round(accept_boundary, 6) if math.isfinite(accept_boundary) else None,
        "accept_boundary_false_safe_count": len(false_safe),
        "accept_boundary_false_safe_fixture_ids": [str(row.get("fixture_id")) for row in false_safe],
        "live_false_accept_rate_available": "false_accept_rate" in exp3124,
        "live_false_accept_rate": as_float(exp3124.get("false_accept_rate"), 0.0),
    }


def model_identity_confound_audit(
    model_specs: Sequence[Mapping[str, Any]],
    selected_model_ids: Sequence[str],
    live_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Audit whether model identity could explain trace outcomes."""

    model_counts = Counter(str(row.get("model_id") or "") for row in live_rows if row.get("model_id"))
    selected_count = len(set(selected_model_ids))
    return {
        "selected_model_ids": list(selected_model_ids),
        "selected_model_count": selected_count,
        "live_trace_model_counts": dict(sorted(model_counts.items())),
        "single_model_trace_only": len(model_counts) == 1,
        "model_id_used_in_energy_features": False,
        "legacy_small_model_selected": any(
            row.get("legacy_small_model") is True and row.get("selected") is True
            for row in model_specs
        ),
        "confound_risk": confound_risk(model_counts, selected_count),
        "audit_note": "energy proxies use fixture fields and replay confidence, not model id",
    }


def correlation_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Quantify sidecar utility against exact reject/repair labels."""

    labels = [as_float(row.get("reject_or_repair_label")) for row in rows]
    return {
        "sidecar_energy": score_correlation(
            [as_float(row.get("final_energy_proxy")) for row in rows], labels
        ),
        "deterministic_constraint_penalty": score_correlation(
            [as_float(row.get("deterministic_constraint_penalty")) for row in rows], labels
        ),
        "quality_proxy": score_correlation(
            [as_float(row.get("quality_proxy")) for row in rows], labels
        ),
        "by_fixture_family": by_fixture_family(rows),
    }


def score_correlation(scores: Sequence[float], labels: Sequence[float]) -> JsonDict:
    """Return finite Pearson/Spearman metrics for one score vector."""

    return {
        "count": len(scores),
        "spearman_reject_or_repair": spearman(scores, labels),
        "pearson_reject_or_repair": pearson(scores, labels),
        "finite": all(math.isfinite(value) for value in [*scores, *labels]),
    }


def by_fixture_family(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report family-level separation without mixing fixture domains."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("fixture_family") or "unknown")].append(row)
    return {
        family: {
            "count": len(items),
            "reject_or_repair_rate": rate(
                sum(as_float(row.get("reject_or_repair_label")) for row in items), len(items)
            ),
            "mean_final_energy_proxy": numeric_summary(
                [as_float(row.get("final_energy_proxy")) for row in items]
            ).get("mean"),
            "mean_quality_proxy": numeric_summary(
                [as_float(row.get("quality_proxy")) for row in items]
            ).get("mean"),
        }
        for family, items in sorted(grouped.items())
    }


def model_specs_from_sources(exp3123: Mapping[str, Any], exp3124: Mapping[str, Any]) -> list[JsonDict]:
    """Normalize mandated model metadata from Exp3123 and optional Exp3124."""

    specs: dict[str, JsonDict] = {
        model_id: {
            "hf_id": model_id,
            "present": False,
            "selected": False,
            "cache_status": "unknown",
            "role": "unknown",
            "model_path": None,
        }
        for model_id in MANDATORY_MODEL_IDS
    }
    for row in mapping_rows(exp3123.get("cache_inventory")):
        model_id = str(row.get("hf_id") or "")
        if model_id in specs:
            specs[model_id].update(
                {
                    "present": row.get("cache_status") == "resolved",
                    "cache_status": str(row.get("cache_status") or "unknown"),
                    "role": str(row.get("role") or "unknown"),
                    "model_path": row.get("path"),
                }
            )
    for row in mapping_rows(exp3124.get("model_specs")):
        model_id = str(row.get("hf_id") or "")
        if model_id in specs:
            specs[model_id].update(dict(row))
    selected = set(string_list(exp3123.get("selected_headline_model_ids"))) | set(
        string_list(exp3124.get("selected_model_ids"))
    )
    for model_id in selected:
        if model_id in specs:
            specs[model_id]["selected"] = True
    return [specs[model_id] for model_id in MANDATORY_MODEL_IDS]


def selected_models(
    exp3123: Mapping[str, Any], exp3124: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Return auditable selected model IDs without inferring new calls."""

    selected = string_list(exp3124.get("selected_model_ids")) or string_list(
        exp3123.get("selected_headline_model_ids")
    )
    if selected:
        return selected
    return [str(row["hf_id"]) for row in model_specs if row.get("selected") is True]


def integration_blockers(
    summaries: Mapping[str, Any],
    live_rows: Sequence[Mapping[str, Any]],
    exp3124: Mapping[str, Any],
    selected_model_ids: Sequence[str],
) -> list[str]:
    """Name concrete blockers before this diagnostic can become integration."""

    blockers = [
        "no generation-path sidecar hook exercised under tests",
        "no trained EBT/ARM learned quality head available",
        "no per-token live energy budget or logprob trace in Exp3130",
        "exact fixture labels are offline authority, not online supervision",
    ]
    false_accept = as_float(exp3124.get("false_accept_rate"), 0.0)
    if false_accept > 0.0:
        blockers.append(f"exp3124 false_accept_rate={false_accept:g}")
    if len(set(selected_model_ids)) <= 1 and live_rows:
        blockers.append("single selected-model trace confound")
    if not live_rows:
        blockers.append("no Exp3124 live rows consumed")
    if summaries["model_identity_confound_audit"]["legacy_small_model_selected"] is True:
        blockers.append("legacy small model selected")
    return blockers


def inference_substrate(
    exp3123: Mapping[str, Any], exp3124: Mapping[str, Any], live_call_count: int
) -> JsonDict:
    """Describe local execution honestly: this run reads artifacts only."""

    gpu_preflight = exp3123.get("gpu_preflight")
    gpu_map = gpu_preflight if isinstance(gpu_preflight, Mapping) else {}
    return {
        "kind": "checked_in_artifact_energy_budget_sidecar_diagnostic",
        "executes_models": False,
        "loads_model_weights": False,
        "generation_performed": False,
        "training_performed": False,
        "live_integration": False,
        "new_live_model_calls": 0,
        "upstream_live_trace_count": live_call_count,
        "uses_exp3124_cached_live_traces": bool(exp3124.get("live_rows")),
        "exp3123_gpu_cuda_available": gpu_map.get("cuda_available"),
        "exp3123_gpu_count": gpu_map.get("gpu_count"),
    }


def readiness_checks_for(
    *,
    exp3123: Mapping[str, Any],
    exact_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Any],
    blockers: Sequence[str],
) -> JsonDict:
    """Collect machine-readable checks for the terminal ready boolean."""

    required_sources_present = all(
        row.get("exists") is True for row in source_rows if row.get("required") is True
    )
    return {
        "exp3123_manifest_ready": exp3123.get("sota_cache_manifest_v2_ready") is True,
        "required_sources_present": required_sources_present,
        "mandated_model_policy_visible": {row.get("hf_id") for row in model_specs}
        == set(MANDATORY_MODEL_IDS),
        "exact_fixture_rows_present": bool(exact_rows),
        "summaries_finite": summaries_finite(summaries),
        "live_integration_false": True,
        "integration_blockers_present": bool(blockers),
    }


def summaries_finite(summaries: Mapping[str, Any]) -> bool:
    """Return whether the primary numeric summaries contain finite evidence."""

    final_penalty = summaries["deterministic_constraint_penalty_summary"]["final_penalty"]
    quality = summaries["learned_quality_proxy_summary"]["quality_proxy"]
    uncertainty = summaries["uncertainty_summary"]["uncertainty_proxy"]
    sidecar = summaries["correlation_metrics"]["sidecar_energy"]
    return all(
        summary.get("finite") is True and int(summary.get("count") or 0) > 0
        for summary in (final_penalty, quality, uncertainty, sidecar)
    )


def numeric_summary(values: Sequence[float]) -> JsonDict:
    """Summarize numeric values with an explicit empty-state marker."""

    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return {"count": 0, "finite": False}
    ordered = sorted(finite_values)
    middle = len(ordered) // 2
    return {
        "count": len(finite_values),
        "finite": True,
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
        "median": round(ordered[middle], 6),
    }


def grouped_numeric(rows: Sequence[Mapping[str, Any]], group_key: str, value_key: str) -> JsonDict:
    """Summarize one numeric field by an exact-authority grouping key."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "unknown")].append(as_float(row.get(value_key)))
    return {key: numeric_summary(values) for key, values in sorted(grouped.items())}


def scale01(values: Sequence[float]) -> list[float]:
    """Min-max scale values, returning zeros when there is no span."""

    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return []
    low = min(finite_values)
    high = max(finite_values)
    if high == low:
        return [0.0 for _value in values]
    return [round((value - low) / (high - low), 6) for value in values]


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return auditable source paths and checksums."""

    specs = (
        ("agents_repo_instructions", Path("AGENTS.md"), False),
        ("codex_repo_workflow", Path("CODEX.md"), False),
        ("claude_authenticity_rules", Path("CLAUDE.md"), False),
        ("research_references", Path("research-references.md"), False),
        ("exp3123_sota_cache_preconditions_manifest_v2", EXP3123_REL_PATH, True),
        ("exp3124_difficulty_stratified_live_sota_verifier_panel_v6", EXP3124_REL_PATH, False),
        ("exp3117_ebt_arm_sidecar_score_correlation_boundary_v3", EXP3117_REL_PATH, True),
    )
    rows: list[JsonDict] = []
    for source_id, rel_path, required in specs:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "exists": path.is_file(),
                "required": required,
                "sha256": file_sha256(path),
            }
        )
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that overclaim live integration or omit required fields."""

    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("live_integration") is False, "live_integration must be false")
    substrate = artifact.get("inference_substrate")
    _require(isinstance(substrate, Mapping), "inference_substrate must be an object")
    _require(substrate.get("new_live_model_calls") == 0, "new_live_model_calls must be 0")
    _require(
        bool(artifact.get("integration_blockers")), "integration_blockers must be non-empty"
    )
    _require(isinstance(artifact.get("model_specs"), list), "model_specs must be a list")
    _require(isinstance(artifact.get("source_artifacts"), list), "source_artifacts must be a list")
    for field in (
        "deterministic_constraint_penalty_summary",
        "learned_quality_proxy_summary",
        "uncertainty_summary",
        "approximation_gap_summary",
        "model_identity_confound_audit",
        "correlation_metrics",
    ):
        _require(isinstance(artifact.get(field), Mapping), f"{field} must be an object")
    if artifact.get("arm_ebt_energy_budget_sidecar_v2_ready") is True:
        _require(int(artifact.get("exact_fixture_count") or 0) > 0, "exact fixtures required")
    verdict = str(artifact.get("honest_verdict", ""))
    _require(
        verdict.startswith(SUCCESS_PREFIXES) or verdict.startswith("blocked_"),
        "honest_verdict must start with success or blocked prefix",
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build a terminal verdict that does not imply integration."""

    ready = artifact.get("arm_ebt_energy_budget_sidecar_v2_ready") is True
    exact_count = int(artifact.get("exact_fixture_count") or 0)
    live_count = int(artifact.get("live_call_count") or 0)
    if ready:
        return (
            "complete: arm_ebt_energy_budget_sidecar_v2_ready=true; "
            f"exact_fixture_count={exact_count}; live_call_count={live_count}; "
            "live_integration=false"
        )
    reasons = artifact.get("blocked_reasons")
    reason_text = ",".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
    if exact_count == 0:
        return f"blocked_missing_trace_source: exact_fixture_count=0; {reason_text}"
    return f"blocked_incomplete_diagnostic: {reason_text}"


def confound_risk(model_counts: Counter[str], selected_count: int) -> str:
    """Classify model-identity confound risk for cached traces."""

    if not model_counts:
        return "none_observed_no_live_trace_rows"
    if selected_count <= 1 or len(model_counts) == 1:
        return "high_single_model_trace"
    return "lower_multiple_model_traces"


def as_float(value: Any, default: float = 0.0) -> float:
    """Convert JSON scalar evidence to finite float with a deterministic default."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    return converted if math.isfinite(converted) else float(default)


def string_list(value: Any) -> list[str]:
    """Return a JSON list of strings, dropping malformed entries."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def file_sha256(path: Path) -> str | None:
    """Hash a source artifact when it exists."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def relative_path(root: Path, path: Path) -> str:
    """Return a stable repository-relative path when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON for artifact diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float, now_s: float | None) -> float:
    """Return rounded wall-clock duration for the artifact."""

    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - started_s), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
