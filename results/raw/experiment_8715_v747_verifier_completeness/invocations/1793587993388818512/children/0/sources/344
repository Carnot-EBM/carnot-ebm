"""Exp 3144 EBT/ARM sidecar calibration against .291 false accepts.

Spec refs: REQ-VERIFY-3144, SCENARIO-VERIFY-3144.

This module is an offline boundary check, not a live verifier integration. It
joins the `.291` false-accept row IDs to the existing ARM/EBT sidecar rows and
asks a deliberately narrower question: would the sidecar fields have identified
the known false accepts as abstention-worthy, and what still blocks using that
signal in the generation path? Keeping that question narrow prevents replay
diagnostics from being mistaken for trained EBT/ARM evidence.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3144_ebt_arm_false_accept_calibration_boundary_v3"
SCHEMA = "carnot.ebt_arm_false_accept_calibration_boundary.v3"
OUTPUT_REL_PATH = Path("results/experiment_3144_ebt_arm_false_accept_calibration_boundary_v3.json")
EXP3117_REL_PATH = Path(
    "results/experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3.json"
)
EXP3124_REL_PATH = Path(
    "results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"
)
EXP3130_REL_PATH = Path("results/experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.json")
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
MANDATORY_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_FIELDS = (
    "ebt_arm_false_accept_calibration_v3_ready",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "false_accept_rows_evaluated",
    "abstention_feature_candidates",
    "false_accept_separation_metrics",
    "approximation_gap_summary",
    "model_identity_confound_audit",
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
    ".venv/bin/pytest tests/python/test_experiment_3144_ebt_arm_false_accept_calibration_boundary_v3.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval/ebt_arm_false_accept_calibration_boundary_v3.py -m pytest -o addopts='' tests/python/test_experiment_3144_ebt_arm_false_accept_calibration_boundary_v3.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/ebt_arm_false_accept_calibration_boundary_v3.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
CALIBRATION_FIELDS = (
    ("deterministic_constraint_penalty", ">="),
    ("final_energy_proxy", ">="),
    ("quality_proxy", "<="),
    ("uncertainty_proxy", ">="),
    ("approximation_gap_to_exact_binary", ">="),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one checked-in JSON object while treating bad inputs as absent evidence."""

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
    """REQ-VERIFY-3144: build the offline false-accept calibration artifact."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    exp3117 = read_json_object(root_path / EXP3117_REL_PATH)
    exp3124 = read_json_object(root_path / EXP3124_REL_PATH)
    exp3130 = read_json_object(root_path / EXP3130_REL_PATH)
    exp3136 = read_json_object(root_path / EXP3136_REL_PATH)
    source_rows = source_artifacts(root_path)
    false_ids = sorted(string_list(exp3136.get("false_accept_row_ids")))
    live_rows = mapping_rows(exp3124.get("live_rows"))
    sidecar_by_id = sidecar_rows_by_id(mapping_rows(exp3117.get("diagnostic_rows")))
    joined_rows = apply_approximation_gaps(
        joined_calibration_rows(live_rows, sidecar_by_id, false_ids)
    )
    false_joined = [row for row in joined_rows if row["false_accept"] is True]
    models = model_specs(exp3130, exp3124)
    selected = selected_model_ids(exp3130, exp3124, models)
    live_count = int(
        as_float(exp3124.get("live_call_count"), as_float(exp3130.get("live_call_count")))
    )
    separation = false_accept_separation_metrics(joined_rows)
    candidates = abstention_feature_candidates(separation, joined_rows, len(false_ids))
    gap_summary = approximation_gap_summary(joined_rows)
    confounds = model_identity_confound_audit(models, selected, joined_rows)
    blockers = integration_blockers(candidates, confounds, joined_rows, false_ids)
    substrate = inference_substrate(exp3124, exp3130, live_count)
    checks = readiness_checks(
        source_rows=source_rows,
        live_rows=joined_rows,
        false_accept_ids=false_ids,
        false_accept_rows=false_joined,
        separation=separation,
        model_specs=models,
        blockers=blockers,
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3144", "SCENARIO-VERIFY-3144"],
        "ebt_arm_false_accept_calibration_v3_ready": all(checks.values()),
        "model_specs": models,
        "selected_model_ids": selected,
        "live_call_count": live_count,
        "false_accept_rows_evaluated": len(false_joined),
        "false_accept_row_ids": [row["row_id"] for row in false_joined],
        "source_false_accept_row_ids": false_ids,
        "live_row_count": len(joined_rows),
        "calibration_rows": joined_rows,
        "abstention_feature_candidates": candidates,
        "false_accept_separation_metrics": separation,
        "approximation_gap_summary": gap_summary,
        "model_identity_confound_audit": confounds,
        "live_integration": False,
        "integration_blockers": blockers,
        "readiness_checks": checks,
        "blocked_reasons": [name for name, ok in checks.items() if ok is not True],
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row["sha256"] is not None
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
    """Build, validate, and persist the Exp 3144 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(out_path, artifact)
    return out_path


def mapping_rows(value: Any) -> list[JsonDict]:
    """Keep JSON object rows and drop malformed list members."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def sidecar_rows_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Index sidecar diagnostic rows by fixture ID for row-level calibration."""

    return {str(row.get("fixture_id")): row for row in rows if row.get("fixture_id")}


def joined_calibration_rows(
    live_rows: Sequence[Mapping[str, Any]],
    sidecar_by_id: Mapping[str, Mapping[str, Any]],
    false_accept_ids: Sequence[str],
) -> list[JsonDict]:
    """Join live verifier rows to existing sidecar fields without recomputing inference."""

    false_set = set(false_accept_ids)
    joined: list[JsonDict] = []
    for live in live_rows:
        row_id = str(live.get("fixture_id") or live.get("row_id") or "")
        sidecar = sidecar_by_id.get(row_id)
        if not row_id or sidecar is None:
            continue
        feature_summary = sidecar.get("feature_summary")
        features = feature_summary if isinstance(feature_summary, Mapping) else {}
        replay_score = sidecar.get("replay_score")
        replay = replay_score if isinstance(replay_score, Mapping) else {}
        energy = as_float(sidecar.get("label_blind_feature_energy"))
        confidence = as_float(replay.get("confidence"), 1.0 / (1.0 + max(0.0, energy)))
        expected = str(live.get("expected_action") or sidecar.get("expected_action") or "")
        live_decision = str(live.get("live_decision") or "")
        joined.append(
            {
                "row_id": row_id,
                "false_accept": row_id in false_set,
                "expected_action": expected,
                "live_decision": live_decision,
                "exact_label": str(live.get("exact_label") or ""),
                "exact_outcome": str(sidecar.get("exact_outcome") or ""),
                "fixture_family": str(
                    live.get("fixture_family") or sidecar.get("task_family") or "unknown"
                ),
                "model_id": str(live.get("model_id") or ""),
                "model_hash": str(live.get("model_hash") or ""),
                "sidecar_action": str(sidecar.get("sidecar_action") or ""),
                "reject_or_repair_label": int(as_float(sidecar.get("reject_or_repair_label"))),
                "deterministic_constraint_penalty": round(
                    as_float(features.get("label_blind_violation"), energy), 6
                ),
                "final_energy_proxy": round(energy, 6),
                "quality_proxy": round(confidence, 6),
                "uncertainty_proxy": round(1.0 - confidence, 6),
                "uses_exact_label_reference_for_score": bool(
                    features.get("uses_exact_label_reference_for_score")
                ),
            }
        )
    return sorted(joined, key=lambda row: row["row_id"])


def apply_approximation_gaps(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Attach min-max energy gaps to each row against exact reject/repair labels."""

    energies = [as_float(row.get("final_energy_proxy")) for row in rows]
    scaled = scale01(energies)
    joined: list[JsonDict] = []
    for row, scaled_energy in zip(rows, scaled, strict=False):
        enriched = dict(row)
        label = as_float(row.get("reject_or_repair_label"))
        enriched["scaled_final_energy_proxy"] = scaled_energy
        enriched["approximation_gap_to_exact_binary"] = round(abs(scaled_energy - label), 6)
        joined.append(enriched)
    return joined


def false_accept_separation_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compare every sidecar field on known false accepts versus other live rows."""

    return {
        field: separation_for_field(rows, field, direction)
        for field, direction in CALIBRATION_FIELDS
    }


def separation_for_field(
    rows: Sequence[Mapping[str, Any]], field: str, threshold_direction: str
) -> JsonDict:
    """Summarize how one field separates false accepts from non-false-accept rows."""

    false_values = [as_float(row.get(field)) for row in rows if row.get("false_accept") is True]
    non_false_values = [
        as_float(row.get(field)) for row in rows if row.get("false_accept") is not True
    ]
    threshold = (
        min(false_values)
        if threshold_direction == ">=" and false_values
        else max(false_values)
        if false_values
        else math.nan
    )
    stats = threshold_stats(false_values, non_false_values, threshold, threshold_direction)
    false_summary = numeric_summary(false_values)
    non_false_summary = numeric_summary(non_false_values)
    return {
        "field": field,
        "threshold_direction": threshold_direction,
        "candidate_threshold": round(threshold, 6) if math.isfinite(threshold) else None,
        "false_accept": false_summary,
        "non_false_accept": non_false_summary,
        "mean_delta_false_minus_non_false": mean_delta(false_summary, non_false_summary),
        **stats,
    }


def threshold_stats(
    false_values: Sequence[float],
    non_false_values: Sequence[float],
    threshold: float,
    threshold_direction: str,
) -> JsonDict:
    """Measure how many rows one replay threshold would abstain on."""

    if not math.isfinite(threshold):
        return {
            "false_accept_recall_at_threshold": 0.0,
            "flagged_false_accept_count": 0,
            "non_false_flagged_at_threshold_count": 0,
        }
    if threshold_direction == ">=":
        false_flagged = sum(1 for value in false_values if value >= threshold)
        non_false_flagged = sum(1 for value in non_false_values if value >= threshold)
    else:
        false_flagged = sum(1 for value in false_values if value <= threshold)
        non_false_flagged = sum(1 for value in non_false_values if value <= threshold)
    return {
        "false_accept_recall_at_threshold": rate(false_flagged, len(false_values)),
        "flagged_false_accept_count": false_flagged,
        "non_false_flagged_at_threshold_count": non_false_flagged,
    }


def abstention_feature_candidates(
    separation: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], false_accept_count: int
) -> list[JsonDict]:
    """Report which replay fields are plausible abstention features but still blocked."""

    candidates: list[JsonDict] = []
    for field, _direction in CALIBRATION_FIELDS:
        metrics = separation.get(field)
        metric_map = metrics if isinstance(metrics, Mapping) else {}
        flagged_false = int(metric_map.get("flagged_false_accept_count") or 0)
        candidates.append(
            {
                "field": field,
                "would_flag_false_accept_rows": false_accept_count > 0
                and flagged_false == false_accept_count,
                "flagged_false_accept_count": flagged_false,
                "known_false_accept_count": false_accept_count,
                "non_false_flagged_at_threshold_count": int(
                    metric_map.get("non_false_flagged_at_threshold_count") or 0
                ),
                "candidate_threshold": metric_map.get("candidate_threshold"),
                "threshold_direction": metric_map.get("threshold_direction"),
                "exact_safe_contract": exact_safe_contract_for(field, rows),
                "live_contract_eligible": False,
                "blocking_contract_requirements": [
                    "missing_generation_path_integration_test",
                    "missing_online_abstention_threshold_contract",
                    "missing_regression_test_for_exp3136_false_accept_rows",
                    "single_model_trace_confound_not_retired",
                ],
            }
        )
    return candidates


def exact_safe_contract_for(field: str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Describe the exact-safe limit for a replay-only abstention field."""

    if field in {"deterministic_constraint_penalty", "final_energy_proxy"}:
        label_reference_used = any(row.get("uses_exact_label_reference_for_score") for row in rows)
        return (
            "candidate_label_blind_replay_signal; exact labels remain the authority; "
            f"uses_exact_label_reference_for_score={label_reference_used}"
        )
    return "proxy_only_replay_signal; cannot replace exact authority under the safe contract"


def approximation_gap_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize proxy-to-exact approximation gaps for false and non-false rows."""

    accepted = [
        as_float(row.get("final_energy_proxy"))
        for row in rows
        if row.get("exact_outcome") == "accepted"
    ]
    boundary = max(accepted) if accepted else math.nan
    false_rows = [row for row in rows if row.get("false_accept") is True]
    non_false_rows = [row for row in rows if row.get("false_accept") is not True]
    below_boundary = [
        row
        for row in false_rows
        if math.isfinite(boundary) and as_float(row.get("final_energy_proxy")) <= boundary
    ]
    return {
        "gap_definition": "absolute gap between min-max scaled proxy energy and exact reject/repair label",
        "false_accept": numeric_summary(
            [as_float(row.get("approximation_gap_to_exact_binary")) for row in false_rows]
        ),
        "non_false_accept": numeric_summary(
            [as_float(row.get("approximation_gap_to_exact_binary")) for row in non_false_rows]
        ),
        "accepted_energy_boundary": round(boundary, 6) if math.isfinite(boundary) else None,
        "false_accept_below_accepted_boundary_count": len(below_boundary),
        "false_accept_below_accepted_boundary_row_ids": [row["row_id"] for row in below_boundary],
        "row_count": len(rows),
    }


def model_specs(exp3130: Mapping[str, Any], exp3124: Mapping[str, Any]) -> list[JsonDict]:
    """Normalize mandated model metadata so trace model use is auditable."""

    specs: dict[str, JsonDict] = {
        model_id: {
            "hf_id": model_id,
            "present": False,
            "selected": False,
            "cache_status": "unknown",
            "role": "unknown",
            "legacy_small_model": False,
        }
        for model_id in MANDATORY_MODEL_IDS
    }
    for source in (exp3130, exp3124):
        for row in mapping_rows(source.get("model_specs")):
            model_id = str(row.get("hf_id") or "")
            if model_id in specs:
                specs[model_id].update(dict(row))
    selected = set(string_list(exp3130.get("selected_model_ids"))) | set(
        string_list(exp3124.get("selected_model_ids"))
    )
    for model_id in selected:
        if model_id in specs:
            specs[model_id]["selected"] = True
    return [specs[model_id] for model_id in MANDATORY_MODEL_IDS]


def selected_model_ids(
    exp3130: Mapping[str, Any], exp3124: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Return selected model IDs from upstream artifacts without inferring new calls."""

    selected = string_list(exp3124.get("selected_model_ids")) or string_list(
        exp3130.get("selected_model_ids")
    )
    if selected:
        return selected
    return [str(row["hf_id"]) for row in specs if row.get("selected") is True]


def model_identity_confound_audit(
    specs: Sequence[Mapping[str, Any]],
    selected: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Audit whether model identity could shortcut the false-accept split."""

    model_counts = Counter(str(row.get("model_id") or "") for row in rows if row.get("model_id"))
    false_model_counts = Counter(
        str(row.get("model_id") or "")
        for row in rows
        if row.get("false_accept") is True and row.get("model_id")
    )
    hash_counts = Counter(str(row.get("model_hash") or "") for row in rows if row.get("model_hash"))
    mandatory_visible = {str(row.get("hf_id")) for row in specs} == set(MANDATORY_MODEL_IDS)
    return {
        "selected_model_ids": list(selected),
        "selected_model_count": len(set(selected)),
        "live_trace_model_counts": dict(sorted(model_counts.items())),
        "false_accept_model_counts": dict(sorted(false_model_counts.items())),
        "live_trace_model_hash_counts": dict(sorted(hash_counts.items())),
        "single_model_trace_only": len(model_counts) == 1,
        "false_accepts_all_from_single_model": len(false_model_counts) == 1,
        "model_id_used_in_sidecar_features": False,
        "model_hash_used_in_sidecar_features": False,
        "legacy_small_model_selected": any(
            row.get("legacy_small_model") is True and row.get("selected") is True for row in specs
        ),
        "mandated_model_policy_visible": mandatory_visible,
        "confound_risk": confound_risk(model_counts, selected),
    }


def integration_blockers(
    candidates: Sequence[Mapping[str, Any]],
    confounds: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    false_accept_ids: Sequence[str],
) -> list[str]:
    """Name concrete work needed before replay calibration can become integration."""

    blockers = [
        "no generation-path sidecar hook exercised under tests",
        "no Exp3144 live generation or abstention integration test",
        "no trained EBT/ARM learned quality head available",
        "no per-token live energy budget or logprob trace in Exp3144",
        "exact-safe threshold not validated on unseen live rows",
    ]
    if false_accept_ids and not candidates:
        blockers.append("no abstention feature candidate evaluated")
    if len([row for row in rows if row.get("false_accept") is True]) != len(false_accept_ids):
        blockers.append("not all Exp3136 false-accept row IDs joined to sidecar diagnostics")
    if confounds.get("single_model_trace_only") is True:
        blockers.append("single selected-model trace confound")
    if confounds.get("legacy_small_model_selected") is True:
        blockers.append("legacy small model selected")
    return blockers


def inference_substrate(
    exp3124: Mapping[str, Any], exp3130: Mapping[str, Any], live_call_count: int
) -> JsonDict:
    """Declare that this run only reads artifacts and performs no new inference."""

    upstream = exp3124.get("inference_substrate")
    upstream_map = upstream if isinstance(upstream, Mapping) else {}
    sidecar_substrate = exp3130.get("inference_substrate")
    sidecar_map = sidecar_substrate if isinstance(sidecar_substrate, Mapping) else {}
    return {
        "kind": "checked_in_artifact_false_accept_sidecar_calibration",
        "executes_models": False,
        "loads_model_weights": False,
        "generation_performed": False,
        "training_performed": False,
        "live_integration": False,
        "new_live_model_calls": 0,
        "upstream_live_trace_count": live_call_count,
        "upstream_exp3124_executes_models": upstream_map.get("executes_models"),
        "upstream_exp3124_loads_model_weights": upstream_map.get("loads_model_weights"),
        "upstream_exp3130_new_live_model_calls": sidecar_map.get("new_live_model_calls"),
    }


def readiness_checks(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    live_rows: Sequence[Mapping[str, Any]],
    false_accept_ids: Sequence[str],
    false_accept_rows: Sequence[Mapping[str, Any]],
    separation: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
) -> JsonDict:
    """Collect the explicit gates behind the ready boolean."""

    required_sources = [row for row in source_rows if row.get("required") is True]
    return {
        "required_sources_present": all(row.get("exists") is True for row in required_sources),
        "live_rows_joined_to_sidecar": bool(live_rows),
        "false_accept_ids_present": bool(false_accept_ids),
        "all_false_accept_rows_evaluated": len(false_accept_rows) == len(false_accept_ids)
        and bool(false_accept_rows),
        "separation_metrics_finite": separation_metrics_finite(separation),
        "mandated_model_policy_visible": {row.get("hf_id") for row in model_specs}
        == set(MANDATORY_MODEL_IDS),
        "live_integration_false": True,
        "integration_blockers_present": bool(blockers),
    }


def separation_metrics_finite(separation: Mapping[str, Any]) -> bool:
    """Return true when every field has finite false and non-false summaries."""

    for metrics in separation.values():
        metric_map = metrics if isinstance(metrics, Mapping) else {}
        false_summary = metric_map.get("false_accept")
        non_false_summary = metric_map.get("non_false_accept")
        if not (
            isinstance(false_summary, Mapping)
            and false_summary.get("finite") is True
            and isinstance(non_false_summary, Mapping)
            and non_false_summary.get("finite") is True
        ):
            return False
    return bool(separation)


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return source provenance for the offline calibration."""

    specs = (
        ("agents_repo_instructions", Path("AGENTS.md"), False),
        ("codex_repo_workflow", Path("CODEX.md"), False),
        ("claude_authenticity_rules", Path("CLAUDE.md"), False),
        ("research_references", Path("research-references.md"), False),
        ("exp3124_live_verifier_rows", EXP3124_REL_PATH, True),
        ("exp3130_energy_budget_sidecar", EXP3130_REL_PATH, True),
        ("exp3136_false_accept_autopsy", EXP3136_REL_PATH, True),
        ("exp3117_sidecar_row_diagnostics", EXP3117_REL_PATH, True),
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
    """Reject artifacts that overclaim integration or omit the required schema."""

    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("live_integration") is False, "live_integration must be false")
    substrate = artifact.get("inference_substrate")
    _require(isinstance(substrate, Mapping), "inference_substrate must be an object")
    _require(substrate.get("new_live_model_calls") == 0, "new_live_model_calls must be 0")
    _require(bool(artifact.get("integration_blockers")), "integration_blockers must be non-empty")
    for field in (
        "false_accept_separation_metrics",
        "approximation_gap_summary",
        "model_identity_confound_audit",
    ):
        _require(isinstance(artifact.get(field), Mapping), f"{field} must be an object")
    _require(isinstance(artifact.get("abstention_feature_candidates"), list), "candidates list")
    _require(isinstance(artifact.get("model_specs"), list), "model_specs must be a list")
    verdict = str(artifact.get("honest_verdict", ""))
    _require(
        verdict.startswith(SUCCESS_PREFIXES) or verdict.startswith("blocked_"),
        "honest_verdict must start with success or blocked prefix",
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build a terminal verdict that preserves the offline-only boundary."""

    false_count = int(artifact.get("false_accept_rows_evaluated") or 0)
    live_count = int(artifact.get("live_call_count") or 0)
    if artifact.get("ebt_arm_false_accept_calibration_v3_ready") is True:
        return (
            "complete: ebt_arm_false_accept_calibration_v3_ready=true; "
            f"false_accept_rows_evaluated={false_count}; live_call_count={live_count}; "
            "live_integration=false"
        )
    reasons = artifact.get("blocked_reasons")
    reason_text = ",".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
    if false_count == 0:
        return f"blocked_missing_trace_source: false_accept_rows_evaluated=0; {reason_text}"
    return f"blocked_incomplete_calibration: {reason_text}"


def numeric_summary(values: Sequence[float]) -> JsonDict:
    """Summarize numeric evidence and mark empty summaries explicitly."""

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


def mean_delta(
    false_summary: Mapping[str, Any], non_false_summary: Mapping[str, Any]
) -> float | None:
    """Return false-minus-non-false mean delta when both summaries are finite."""

    if false_summary.get("finite") is not True or non_false_summary.get("finite") is not True:
        return None
    return round(as_float(false_summary.get("mean")) - as_float(non_false_summary.get("mean")), 6)


def scale01(values: Sequence[float]) -> list[float]:
    """Min-max scale row energies for approximation-gap accounting."""

    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return []
    low = min(finite_values)
    high = max(finite_values)
    if high == low:
        return [0.0 for _value in values]
    return [round((value - low) / (high - low), 6) for value in values]


def rate(numerator: float, denominator: float) -> float:
    """Return a rounded rate with a deterministic zero denominator fallback."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def confound_risk(model_counts: Counter[str], selected: Sequence[str]) -> str:
    """Classify model-identity shortcut risk for the cached traces."""

    if not model_counts:
        return "none_observed_no_live_trace_rows"
    if len(set(selected)) <= 1 or len(model_counts) == 1:
        return "high_single_model_trace"
    return "lower_multiple_model_traces"


def as_float(value: Any, default: float = 0.0) -> float:
    """Convert JSON scalars into finite floats without throwing on bad evidence."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    return converted if math.isfinite(converted) else float(default)


def string_list(value: Any) -> list[str]:
    """Return string members from a JSON list."""

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
    """Persist deterministic JSON so result diffs remain reviewable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float, now_s: float | None) -> float:
    """Return rounded wall-clock duration for the artifact."""

    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - started_s), 6)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
