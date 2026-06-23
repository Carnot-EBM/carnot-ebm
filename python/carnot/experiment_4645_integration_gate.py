"""Experiment 4645: ARC sprint integration gate for the scored submitted agent.

Spec refs: REQ-ARC-WMTE-4645, SCENARIO-ARC-WMTE-4645.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]
ArtifactPair = tuple[Mapping[str, Any], Mapping[str, Any]]
SummarizeRunner = Callable[[Path], Mapping[str, Any]]
GateCheck = Callable[[Path | str], Mapping[str, Any]]
OfflineArcadeChecker = Callable[[], bool]

EXPERIMENT = "experiment_4645_integration_gate"
SCHEMA = "carnot.exp4645.integration_gate.v1"
RESULT_RELATIVE_PATH = "results/experiment_4645_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4640_goal_energy_generation_live.json"
A2_RELATIVE_PATH = "results/experiment_4641_action_effect_expansion_prior_live.json"
A3_RELATIVE_PATH = "results/experiment_4642_levelup_selfplay.json"
A4_RELATIVE_PATH = "results/experiment_4643_refresh_submission_package.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4645
LIVE_SUBMITTABLE_BASELINE = 33
REFRESHED_PACKAGE_PATH = "results/experiment_4643_submission_package_operator_resubmit.json"
REFRESHED_PACKAGE_SOURCE = "experiment_4643_refresh_submission_package"
TERMINAL_PREFIXES = (
    "success:",
    "success_",
    "complete:",
    "complete_",
    "blocked_",
    "passed:",
    "shipped:",
)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: integrated_<metric>_raised_config_shipped OR "
            "complete: integration_no_clean_metric_bare_config_kept_honest_null."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false for every aggregated value claim -- the integrated levers are "
            "oracle-distinct."
        )
    },
    "levers_integrated": {
        "principle": (
            "names the upstream levers (A1/A2/A3) admitted into SUBMITTED_AGENT_CONFIG -- "
            "the audit trail."
        )
    },
    "flagged_artifacts_excluded": {
        "principle": (
            "names any flagged_adversarial / positive-control-failed / "
            "uniform-energy-ablation-failed upstream artifact NOT aggregated (the fabrication "
            "gate + FALSE_NEGATIVE_RISK compliance)."
        )
    },
    "live_solve_rate_integrated": {
        "principle": "the integrated held-out live solve-rate on the shipped config (A1's effect)."
    },
    "live_multi_level_solve_rate_integrated": {
        "principle": (
            "the integrated >=2-level live solve-rate (A2's deeper-solve effect) -- the new wall "
            "metric."
        )
    },
    "action_efficiency_integrated": {
        "principle": (
            "the integrated median actions-to-first-levelup + efficiency term on the shipped "
            "config."
        )
    },
    "offline_to_live_transfer_ratio_integrated": {
        "principle": (
            "the integrated bridge co-metric on the shipped config -- did the offline signal "
            "transfer to the SCORED agent."
        )
    },
    "live_submittable_level_count_integrated": {
        "principle": "the integrated live-submittable count (must stay > 33)."
    },
    "parity_test_green": {
        "principle": (
            "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config is "
            "the single source of truth."
        )
    },
    "orphan_lint_green": {
        "principle": (
            "HARD gate -- arc_orphan_solver_lint passes; the graduated A1/A2 modules stay "
            "live-path-reachable."
        )
    },
    "submitted_config_raised_metric_clean": {
        "principle": (
            "True only if a CLEAN (non-flagged, control-passed, uniform-energy-ablation-passed "
            "for A1) lever raised a real metric on the SCORED config; false -> honest null, "
            "bare config kept."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "present where any integrated delta == 0 -- states the equality is an honest "
            "no-value null, not a bug (the .427 TAUTOLOGY-false-flag fix)."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, upstream artifacts present); pre-empts "
            "missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "live_solve_rate_delta_vs_bare",
    "live_multi_level_solve_rate_delta_vs_bare",
    "offline_to_live_transfer_ratio_delta_vs_baseline",
    "live_submittable_delta_vs_baseline",
    "metric_delta_audit",
    "upstream_lever_audit",
    "quarantined_artifacts",
    "package_artifact",
    "submitted_agent_config",
    "submitted_config_expected_patch",
    "config_action",
    "parity_test",
    "orphan_lint",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)
SPEC_REFS = ["REQ-ARC-WMTE-4645", "SCENARIO-ARC-WMTE-4645"]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def _load_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _success_verdict(artifact: Mapping[str, Any]) -> bool:
    return str(artifact.get("honest_verdict") or "").startswith(("success:", "success_"))


def _live_status(summary: Mapping[str, Any]) -> str:
    status = str(summary.get("live_status") or "")
    if status:
        return status
    text = f"{summary.get('stdout') or ''}\n{summary.get('stderr') or ''}"
    if "LIVE re-check: CRITICAL" in text:
        return "CRITICAL"
    if "LIVE re-check: warn" in text:
        return "warn"
    return "clean" if int(summary.get("returncode") or 0) == 0 else "CRITICAL"


def _is_flagged(artifact: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    stamped = artifact.get("flagged_adversarial")
    return stamped not in (None, False) or _live_status(summary).upper() == "CRITICAL"


def _positive_control_passed(lever: str, artifact: Mapping[str, Any]) -> bool:
    if artifact.get("positive_control_passed") is True:
        return True
    if lever in {"A1", "A2"}:
        return (
            artifact.get("bare_control_passed") is True
            and artifact.get("false_negative_risk_checked") is True
        )
    if lever == "A3":
        gate = artifact.get("reproduction_gate")
        return (
            artifact.get("offline_reproduced") is True
            and isinstance(gate, Mapping)
            and gate.get("reproduced") is True
        )
    return False


def _uniform_gate_passed(lever: str, artifact: Mapping[str, Any]) -> bool:
    return lever != "A1" or artifact.get("uniform_energy_ablation_passed") is True


def _lever_metric_delta(lever: str, artifact: Mapping[str, Any]) -> float:
    if lever == "A1":
        return max(
            _as_float(artifact.get("solve_rate_delta")),
            _as_float(artifact.get("first_win_rate_delta")),
        )
    if lever == "A2":
        return max(
            _as_float(artifact.get("solve_rate_delta")),
            _as_float(artifact.get("depth_of_live_solve_delta")),
        )
    if lever == "A3":
        return float(_as_int(artifact.get("reproduced_levels")))
    return 0.0


def _lever_name(lever: str) -> str:
    return {
        "A1": "A1_graded_goal_energy_generation",
        "A2": "A2_action_effect_expansion_prior",
        "A3": "A3_level_bank_refreshed_package",
    }[lever]


def _audit_lever(lever: str, artifact: Mapping[str, Any], summary: Mapping[str, Any]) -> JsonDict:
    status = _live_status(summary)
    flagged = _is_flagged(artifact, summary)
    positive_control = _positive_control_passed(lever, artifact)
    uniform_ok = _uniform_gate_passed(lever, artifact)
    metric_delta = _lever_metric_delta(lever, artifact)
    if flagged:
        reason = "flagged_adversarial"
    elif not _success_verdict(artifact):
        reason = "honest_verdict_not_success"
    elif not positive_control:
        reason = "positive_control_failed"
    elif not uniform_ok:
        reason = "uniform_energy_ablation_failed"
    elif metric_delta <= 0.0:
        reason = "no_positive_metric_delta"
    else:
        reason = "admitted_clean_metric_raiser"
    integrated = reason == "admitted_clean_metric_raiser"
    return {
        "lever": lever,
        "artifact_honest_verdict": artifact.get("honest_verdict"),
        "flagged_adversarial": artifact.get("flagged_adversarial") not in (None, False),
        "summarize_returncode": int(summary.get("returncode") or 0),
        "live_status": status,
        "positive_control_passed": bool(positive_control),
        "uniform_energy_ablation_passed": bool(uniform_ok),
        "metric_delta": float(metric_delta),
        "integrated": bool(integrated),
        "reason": reason,
    }


def audit_upstream_levers(upstreams: Mapping[str, ArtifactPair]) -> JsonDict:
    rows = {
        lever: _audit_lever(lever, upstreams[lever][0], upstreams[lever][1])
        for lever in ("A1", "A2", "A3")
        if lever in upstreams
    }
    levers_integrated = [
        _lever_name(lever)
        for lever in ("A1", "A2", "A3")
        if rows.get(lever, {}).get("integrated") is True
    ]
    flagged_excluded = [
        {
            "lever": lever,
            "reason": row["reason"],
            "live_status": row["live_status"],
            "honest_verdict": row["artifact_honest_verdict"],
        }
        for lever, row in rows.items()
        if row["reason"]
        in {"flagged_adversarial", "positive_control_failed", "uniform_energy_ablation_failed"}
    ]
    quarantined = [
        {
            "lever": lever,
            "reason": row["reason"],
            "honest_verdict": row["artifact_honest_verdict"],
        }
        for lever, row in rows.items()
        if row["integrated"] is not True
    ]
    return {
        "levers_integrated": levers_integrated,
        "submitted_config_raised_metric_clean": bool(levers_integrated),
        "flagged_artifacts_excluded": flagged_excluded,
        "quarantined_artifacts": quarantined,
        "upstream_lever_audit": rows,
    }


def _lever_integrated(audit: Mapping[str, Any], lever: str) -> bool:
    rows = audit.get("upstream_lever_audit")
    return isinstance(rows, Mapping) and rows.get(lever, {}).get("integrated") is True


def _action_delta(median_bare: Any, median_integrated: Any) -> float:
    bare = _as_float(median_bare)
    integrated = _as_float(median_integrated)
    if bare <= 0.0 or integrated <= 0.0:
        return 0.0
    return _rounded(max(0.0, bare - integrated))


def _efficiency_term(median_bare: Any, median_integrated: Any) -> float:
    bare = _as_float(median_bare)
    integrated = _as_float(median_integrated)
    if bare <= 0.0 or integrated <= 0.0:
        return 0.0
    return _rounded(max(0.0, (bare - integrated) / bare))


def _bridge_ratio(a1_artifact: Mapping[str, Any], a2_artifact: Mapping[str, Any]) -> tuple[float, float]:
    for artifact in (a2_artifact, a1_artifact):
        ratio = _as_float(artifact.get("offline_to_live_transfer_ratio"))
        baseline = _as_float(artifact.get("offline_to_live_transfer_ratio_baseline"), 1.0)
        if ratio > 0.0:
            return _rounded(ratio), _rounded(baseline)
    return 0.0, 0.0


def measure_integrated_metrics(
    *,
    audit: Mapping[str, Any],
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    a4_artifact: Mapping[str, Any],
) -> JsonDict:
    a1_live = _lever_integrated(audit, "A1")
    a2_live = _lever_integrated(audit, "A2")
    a3_live = _lever_integrated(audit, "A3")

    live_solve_bare = _as_float(a1_artifact.get("live_solve_rate_baseline"))
    live_solve_integrated = (
        _as_float(a1_artifact.get("live_solve_rate_goal_energy"), live_solve_bare)
        if a1_live
        else live_solve_bare
    )

    live_multi_bare = _as_float(a2_artifact.get("live_solve_rate_ranker_baseline"))
    live_multi_integrated = (
        _as_float(a2_artifact.get("live_solve_rate_expansion"), live_multi_bare)
        if a2_live
        else live_multi_bare
    )

    median_bare = a2_artifact.get("median_actions_to_win_ranker_baseline")
    median_integrated = (
        a2_artifact.get("median_actions_to_win_expansion") if a2_live else median_bare
    )
    actions_delta = _action_delta(median_bare, median_integrated) if a2_live else 0.0
    efficiency = _efficiency_term(median_bare, median_integrated) if a2_live else 0.0

    ratio, ratio_baseline = _bridge_ratio(a1_artifact, a2_artifact) if a2_live or a1_live else (0.0, 0.0)

    live_count = _as_int(a4_artifact.get("live_submittable_level_count"))
    live_baseline = _as_int(a4_artifact.get("live_submittable_count_prev"), live_count)
    if live_count <= 0:
        live_count = LIVE_SUBMITTABLE_BASELINE
        live_baseline = LIVE_SUBMITTABLE_BASELINE
    integrated_count = live_count if a3_live else live_baseline
    return {
        "live_solve_rate_integrated": _rounded(live_solve_integrated),
        "live_solve_rate_bare": _rounded(live_solve_bare),
        "live_solve_rate_delta_vs_bare": _rounded(live_solve_integrated - live_solve_bare),
        "live_multi_level_solve_rate_integrated": _rounded(live_multi_integrated),
        "live_multi_level_solve_rate_bare": _rounded(live_multi_bare),
        "live_multi_level_solve_rate_delta_vs_bare": _rounded(
            live_multi_integrated - live_multi_bare
        ),
        "action_efficiency_integrated": {
            "median_actions_to_first_levelup": median_integrated,
            "median_actions_to_first_levelup_bare": median_bare,
            "actions_delta_vs_bare": actions_delta,
            "efficiency_score_term": efficiency,
        },
        "offline_to_live_transfer_ratio_integrated": ratio,
        "offline_to_live_transfer_ratio_baseline": ratio_baseline,
        "offline_to_live_transfer_ratio_delta_vs_baseline": _rounded(ratio - ratio_baseline),
        "live_submittable_level_count_integrated": integrated_count,
        "live_submittable_level_count_baseline": live_baseline,
        "live_submittable_delta_vs_baseline": integrated_count - live_baseline,
    }


def _action_efficiency_delta(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("action_efficiency_integrated")
    if isinstance(value, Mapping):
        return _as_float(value.get("actions_delta_vs_bare"))
    return 0.0


def _verdict(
    audit: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    parity_green: bool,
    orphan_green: bool,
) -> str:
    if (
        not parity_green
        or not orphan_green
        or audit.get("submitted_config_raised_metric_clean") is not True
    ):
        return "complete: integration_no_clean_metric_bare_config_kept_honest_null"
    if _as_float(metrics.get("live_solve_rate_delta_vs_bare")) > 0.0:
        return "success: integrated_live_solve_rate_raised_config_shipped"
    if _as_float(metrics.get("live_multi_level_solve_rate_delta_vs_bare")) > 0.0:
        return "success: integrated_live_multi_level_solve_rate_raised_config_shipped"
    if _action_efficiency_delta(metrics) > 0.0:
        return "success: integrated_action_efficiency_raised_config_shipped"
    if _as_int(metrics.get("live_submittable_delta_vs_baseline")) > 0:
        return "success: integrated_live_submittable_raised_config_shipped"
    if _as_float(metrics.get("offline_to_live_transfer_ratio_delta_vs_baseline")) > 0.0:
        return "success: integrated_offline_to_live_transfer_raised_config_shipped"
    return "complete: integration_no_clean_metric_bare_config_kept_honest_null"


def _null_delta_note(metrics: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    zero_fields = [
        name
        for name, value in (
            ("live_solve_rate_delta_vs_bare", metrics.get("live_solve_rate_delta_vs_bare")),
            (
                "live_multi_level_solve_rate_delta_vs_bare",
                metrics.get("live_multi_level_solve_rate_delta_vs_bare"),
            ),
            (
                "offline_to_live_transfer_ratio_delta_vs_baseline",
                metrics.get("offline_to_live_transfer_ratio_delta_vs_baseline"),
            ),
            ("actions_delta_vs_bare", _action_efficiency_delta(metrics)),
            (
                "live_submittable_delta_vs_baseline",
                metrics.get("live_submittable_delta_vs_baseline"),
            ),
        )
        if _as_float(value) == 0.0
    ]
    if audit.get("submitted_config_raised_metric_clean") is not True:
        return (
            "No clean upstream lever passed the summarize-artifact, positive-control, "
            "uniform-energy-ablation (for A1), FALSE_NEGATIVE_RISK, and non-adversarial "
            "gates; zero deltas are an honest no-value null, not a bug."
        )
    if not zero_fields:
        return "No integrated delta was zero."
    return (
        "Zero integrated deltas in "
        + ", ".join(zero_fields)
        + " are honest no-value nulls for those metrics, not measurement bugs; a "
        "bare==integrated equality is not a fabricated matched-pair improvement."
    )


def _expected_config_patch(
    audit: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    package_path: str,
) -> JsonDict:
    return {
        "goal_energy_enabled": bool(
            _lever_integrated(audit, "A1")
            or submitted_agent_config.get("goal_energy_enabled") is True
        ),
        "action_effect_expansion_prior_enabled": bool(
            _lever_integrated(audit, "A2")
            or submitted_agent_config.get("action_effect_expansion_prior_enabled") is True
        ),
        "action_effect_expansion_prior_mode": "persistent_aem_plus_optional_cnn_frontier_prior",
        "live_submit_package_path": package_path,
        "live_submit_source": REFRESHED_PACKAGE_SOURCE,
    }


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    audit: Mapping[str, Any],
    metrics: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    orphan_lint: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    parity_green = bool(parity_test.get("passed"))
    orphan_green = bool(orphan_lint.get("passed"))
    package_path = str(submitted_agent_config.get("live_submit_package_path") or REFRESHED_PACKAGE_PATH)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": _verdict(
            audit,
            metrics,
            parity_green=parity_green,
            orphan_green=orphan_green,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "levers_integrated": list(audit.get("levers_integrated") or []),
        "flagged_artifacts_excluded": list(audit.get("flagged_artifacts_excluded") or []),
        "live_solve_rate_integrated": metrics.get("live_solve_rate_integrated"),
        "live_solve_rate_delta_vs_bare": metrics.get("live_solve_rate_delta_vs_bare"),
        "live_multi_level_solve_rate_integrated": metrics.get(
            "live_multi_level_solve_rate_integrated"
        ),
        "live_multi_level_solve_rate_delta_vs_bare": metrics.get(
            "live_multi_level_solve_rate_delta_vs_bare"
        ),
        "action_efficiency_integrated": dict(metrics.get("action_efficiency_integrated") or {}),
        "offline_to_live_transfer_ratio_integrated": metrics.get(
            "offline_to_live_transfer_ratio_integrated"
        ),
        "offline_to_live_transfer_ratio_delta_vs_baseline": metrics.get(
            "offline_to_live_transfer_ratio_delta_vs_baseline"
        ),
        "live_submittable_level_count_integrated": metrics.get(
            "live_submittable_level_count_integrated"
        ),
        "live_submittable_delta_vs_baseline": metrics.get("live_submittable_delta_vs_baseline"),
        "metric_delta_audit": {
            "live_solve_rate": {
                "bare": metrics.get("live_solve_rate_bare"),
                "integrated": metrics.get("live_solve_rate_integrated"),
                "delta_vs_bare": metrics.get("live_solve_rate_delta_vs_bare"),
            },
            "live_multi_level_solve_rate": {
                "bare": metrics.get("live_multi_level_solve_rate_bare"),
                "integrated": metrics.get("live_multi_level_solve_rate_integrated"),
                "delta_vs_bare": metrics.get("live_multi_level_solve_rate_delta_vs_bare"),
            },
            "offline_to_live_transfer_ratio": {
                "baseline": metrics.get("offline_to_live_transfer_ratio_baseline"),
                "integrated": metrics.get("offline_to_live_transfer_ratio_integrated"),
                "delta_vs_baseline": metrics.get(
                    "offline_to_live_transfer_ratio_delta_vs_baseline"
                ),
            },
            "live_submittable_level_count": {
                "baseline": metrics.get("live_submittable_level_count_baseline"),
                "integrated": metrics.get("live_submittable_level_count_integrated"),
                "delta_vs_baseline": metrics.get("live_submittable_delta_vs_baseline"),
            },
        },
        "parity_test_green": parity_green,
        "orphan_lint_green": orphan_green,
        "submitted_config_raised_metric_clean": bool(
            audit.get("submitted_config_raised_metric_clean")
        ),
        "null_delta_methodology_note": _null_delta_note(metrics, audit),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "upstream_lever_audit": dict(audit.get("upstream_lever_audit") or {}),
        "quarantined_artifacts": list(audit.get("quarantined_artifacts") or []),
        "package_artifact": {
            "path": package_path,
            "source_result_path": A4_RELATIVE_PATH,
            "live_submittable_level_count": metrics.get("live_submittable_level_count_integrated"),
        },
        "submitted_agent_config": json.loads(json.dumps(submitted_agent_config, default=str)),
        "submitted_config_expected_patch": _expected_config_patch(
            audit, submitted_agent_config, package_path
        ),
        "config_action": (
            "ship_clean_metric_levers"
            if audit.get("submitted_config_raised_metric_clean") is True
            else "unchanged_bare_config_kept"
        ),
        "parity_test": dict(parity_test),
        "orphan_lint": dict(orphan_lint),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field {field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    verdict = str(artifact.get("honest_verdict") or "")
    blocked = verdict.startswith("blocked_")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not blocked and artifact.get("parity_test_green") is not True:
        errors.append("parity_test_green")
    if not blocked and artifact.get("orphan_lint_green") is not True:
        errors.append("orphan_lint_green")
    if (
        not blocked
        and _as_int(artifact.get("live_submittable_level_count_integrated"))
        <= LIVE_SUBMITTABLE_BASELINE
    ):
        errors.append("live_submittable_level_count_integrated")
    if "null_delta_methodology_note" not in artifact:
        errors.append("null_delta_methodology_note")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_false")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - ARC SDK boundary.
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return True


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    offline_arcade_checker: OfflineArcadeChecker | None = None,
) -> JsonDict:
    root_path = Path(root)
    checker = offline_arcade_checker or _default_offline_arcade_checker
    try:
        offline_ok = bool(checker())
        offline_error = ""
    except Exception as exc:
        offline_ok = False
        offline_error = f"{type(exc).__name__}: {exc}"
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": offline_ok,
        "offline_arcade_error": offline_error,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "a3_artifact_present": (root_path / A3_RELATIVE_PATH).exists(),
        "a4_artifact_present": (root_path / A4_RELATIVE_PATH).exists(),
        "spec_has_req_4645": "REQ-ARC-WMTE-4645" in spec_text,
    }
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "a3_artifact_present",
        "a4_artifact_present",
        "spec_has_req_4645",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next((key for key in required if not checks[key]), "precondition")
    return checks


def run_summarize_artifact(path: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    cmd = [sys.executable, "scripts/summarize_artifact.py", str(path)]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "live_status": _live_status({"returncode": proc.returncode, "stdout": proc.stdout}),
        "command": " ".join(cmd),
    }


def run_parity_check(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/python/test_arc_submitted_agent_parity.py",
        "-q",
        "--no-cov",
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def run_orphan_lint(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess.
    cmd = [sys.executable, "scripts/arc_orphan_solver_lint.py"]
    proc = subprocess.run(
        cmd,
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _blocked_artifact(
    checks: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    audit = {
        "levers_integrated": [],
        "submitted_config_raised_metric_clean": False,
        "flagged_artifacts_excluded": [],
        "quarantined_artifacts": [],
        "upstream_lever_audit": {},
    }
    metrics = {
        "live_solve_rate_integrated": 0.0,
        "live_solve_rate_bare": 0.0,
        "live_solve_rate_delta_vs_bare": 0.0,
        "live_multi_level_solve_rate_integrated": 0.0,
        "live_multi_level_solve_rate_bare": 0.0,
        "live_multi_level_solve_rate_delta_vs_bare": 0.0,
        "action_efficiency_integrated": {
            "median_actions_to_first_levelup": None,
            "median_actions_to_first_levelup_bare": None,
            "actions_delta_vs_bare": 0.0,
            "efficiency_score_term": 0.0,
        },
        "offline_to_live_transfer_ratio_integrated": 0.0,
        "offline_to_live_transfer_ratio_baseline": 0.0,
        "offline_to_live_transfer_ratio_delta_vs_baseline": 0.0,
        "live_submittable_level_count_integrated": 0,
        "live_submittable_level_count_baseline": LIVE_SUBMITTABLE_BASELINE,
        "live_submittable_delta_vs_baseline": -LIVE_SUBMITTABLE_BASELINE,
    }
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        metrics=metrics,
        parity_test={"passed": False, "blocked": True},
        orphan_lint={"passed": False, "blocked": True},
        submitted_agent_config=submitted_agent_config,
        duration_s=duration_s,
    )
    blocked = str(checks.get("blocked_resource") or "precondition")
    artifact["honest_verdict"] = f"blocked_{blocked}"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    *,
    offline_arcade_checker: OfflineArcadeChecker | None = None,
    summarize_runner: SummarizeRunner | None = None,
    parity_check: GateCheck | None = None,
    orphan_lint: GateCheck | None = None,
    now: Callable[[], float] = time.time,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    submitted_config = json.loads(json.dumps(SUBMITTED_AGENT_CONFIG, default=str))
    checks = check_preconditions(root_path, offline_arcade_checker=offline_arcade_checker)
    if not checks["ok"]:
        artifact = _blocked_artifact(checks, submitted_config, duration_s=max(1.0, now() - start))
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    summarize = summarize_runner or run_summarize_artifact
    paths = {
        "A1": root_path / A1_RELATIVE_PATH,
        "A2": root_path / A2_RELATIVE_PATH,
        "A3": root_path / A3_RELATIVE_PATH,
    }
    upstreams: dict[str, ArtifactPair] = {}
    for lever, path in paths.items():
        artifact = _load_json(path)
        summary = summarize(path) if path.exists() else {"returncode": 2, "live_status": "missing"}
        upstreams[lever] = (artifact, summary)
    audit = audit_upstream_levers(upstreams)
    metrics = measure_integrated_metrics(
        audit=audit,
        a1_artifact=upstreams["A1"][0],
        a2_artifact=upstreams["A2"][0],
        a4_artifact=_load_json(root_path / A4_RELATIVE_PATH),
    )
    parity = (parity_check or run_parity_check)(root_path)
    lint = (orphan_lint or run_orphan_lint)(root_path)
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        metrics=metrics,
        parity_test=parity,
        orphan_lint=lint,
        submitted_agent_config=submitted_config,
        duration_s=max(1.0, now() - start),
    )
    _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    errors = artifact_schema_errors(artifact)
    if errors:
        print(json.dumps({"result": RESULT_RELATIVE_PATH, "schema_errors": errors}, indent=2))
        return 1
    print(
        json.dumps({"result": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
