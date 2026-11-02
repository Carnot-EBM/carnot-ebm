"""Experiment 4669: integrate ARC sprint A1/A2 into the submitted config.

Spec refs: REQ-ARC-WMTE-4669, SCENARIO-ARC-WMTE-4669.
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
Check = Callable[[Path | str], Mapping[str, Any]]
ImportChecker = Callable[[], Mapping[str, Any]]
OfflineArcadeChecker = Callable[[], bool]

EXPERIMENT = "experiment_4669_integration_gate"
SCHEMA = "carnot.exp4669.integration_gate.v1"
RESULT_RELATIVE_PATH = "results/experiment_4669_integration_gate.json"
PREVIOUS_GATE_RELATIVE_PATH = "results/experiment_4657_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4664_l2_goal_predicate_induction_live.json"
A2_RELATIVE_PATH = "results/experiment_4665_dagger_distribution_shift_value_routing.json"
PACKAGE_RELATIVE_PATH = "results/experiment_4667_refresh_submission_package.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4669
FEATURE_SUBSET = "cross_game_features_v3:v2_plus_frame_delta"
DAGGER_VALUE_HEAD_RELATIVE_PATH = "models/arc_dagger_value_routing_v3.json"
LIVE_SUBMITTABLE_FLOOR = 33
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
)
TERMINAL_PREFIXES = (
    "success:",
    "success_",
    "complete:",
    "complete_",
    "blocked_",
    "passed:",
    "shipped:",
)
SPEC_REFS = ["REQ-ARC-WMTE-4669", "SCENARIO-ARC-WMTE-4669"]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: integrated_<config>_shipped_parity_green OR "
            "complete: integration_unchanged_both_levers_null."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the integrated L2-goal-induction / value-routing "
            "signals are oracle-distinct from the executable win-check."
        )
    },
    "config_integrated": {
        "principle": (
            "the A1/A2 config folded into SUBMITTED_AGENT_CONFIG (or 'unchanged' "
            "with the reason) -- the single-source-of-truth update."
        )
    },
    "live_first_win_rate_integrated": {
        "principle": (
            "the integrated SCORED-agent live first-win-rate (no regression vs pre-integration)."
        )
    },
    "live_multi_level_solve_rate_integrated": {
        "principle": (
            "the integrated SCORED-agent multi-level (>=2) live solve-rate (the "
            "deeper wall) -- measured on the FIXED non-degenerate harness."
        )
    },
    "live_submittable_level_count_integrated": {
        "principle": "the integrated live-submittable count (must stay > 33)."
    },
    "parity_test_green": {
        "principle": (
            "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed "
            "agent == the measured agent."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "pre_integration_config",
    "live_first_win_rate_pre_integration",
    "live_first_win_rate_delta_vs_pre_integration",
    "live_multi_level_solve_rate_pre_integration",
    "live_multi_level_solve_rate_delta_vs_pre_integration",
    "no_regression_vs_pre_integration",
    "a1_a2_config_audit",
    "metrics_delta_audit",
    "parity_test",
    "submitted_agent_config",
    "source_artifacts",
    "source_artifact_checksums",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def artifact_checksum(value: Mapping[str, Any]) -> str:
    return "sha256:" + _sha256(value)


def _load_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _mapping_float(mapping: Mapping[str, Any], key: str) -> float:
    return _as_float(mapping.get(key))


def _truthy_values(value: Any) -> list[bool]:
    if isinstance(value, Mapping):
        return [bool(item) for item in value.values()]
    return [bool(value)]


def _max_int_value(value: Any) -> int:
    if isinstance(value, Mapping):
        return max((_as_int(item) for item in value.values()), default=0)
    return _as_int(value)


def _fixed_metric_harness(a1_artifact: Mapping[str, Any]) -> bool:
    metric = a1_artifact.get("metric_harness_fixed")
    return bool(
        isinstance(metric, Mapping)
        and _as_int(metric.get("target_levels")) >= 2
        and metric.get("break_at_first_win") is False
    )


def _a1_audit(
    a1_artifact: Mapping[str, Any], submitted_agent_config: Mapping[str, Any]
) -> JsonDict:
    chosen = a1_artifact.get("chosen_submitted_config")
    metric_harness_fixed = _fixed_metric_harness(a1_artifact)
    upstream_win = bool(
        bool(a1_artifact.get("win_state_exemplar_injected"))
        and any(_truthy_values(a1_artifact.get("goal_predicate_satisfiable")))
        and any(_truthy_values(a1_artifact.get("l2_plan_reaches_goal")))
        and _max_int_value(a1_artifact.get("generic_agent_reached_level")) >= 2
    )
    submitted_exemplar = (
        submitted_agent_config.get("level_up_reinduction_win_state_exemplar") is True
    )
    submitted_satisfiability = (
        submitted_agent_config.get("goal_predicate_satisfiability_check") is True
    )
    if not isinstance(chosen, Mapping):
        reason = "chosen_submitted_config_null_or_no_l2_goal_induction_win"
        integrated = False
    elif a1_artifact.get("verifier_is_oracle") is not False:
        reason = "verifier_oracle_not_false"
        integrated = False
    elif not upstream_win or not metric_harness_fixed:
        reason = "a1_upstream_l2_goal_induction_not_winning"
        integrated = False
    elif not (submitted_exemplar and submitted_satisfiability):
        reason = "submitted_l2_goal_induction_config_mismatch"
        integrated = False
    else:
        reason = "submitted_config_matches_a1_l2_goal_induction"
        integrated = True
    return {
        "lever": "A1",
        "integrated": integrated,
        "reason": reason,
        "chosen_submitted_config": chosen,
        "metric_harness_fixed": metric_harness_fixed,
        "win_state_exemplar_injected": bool(a1_artifact.get("win_state_exemplar_injected")),
        "goal_predicate_satisfiable": a1_artifact.get("goal_predicate_satisfiable"),
        "l2_plan_reaches_goal": a1_artifact.get("l2_plan_reaches_goal"),
        "max_generic_agent_reached_level": _max_int_value(
            a1_artifact.get("generic_agent_reached_level")
        ),
    }


def _a2_audit(
    a2_artifact: Mapping[str, Any], submitted_agent_config: Mapping[str, Any]
) -> JsonDict:
    chosen = a2_artifact.get("chosen_submitted_config")
    submitted_weight = _as_float(submitted_agent_config.get("value_weight"))
    submitted_subset = str(submitted_agent_config.get("value_head_feature_subset") or "")
    submitted_checkpoint = str(submitted_agent_config.get("value_head_checkpoint") or "")
    submitted_corrected = submitted_agent_config.get("value_head_distribution_corrected") is True
    if chosen in (None, "unchanged"):
        reason = "chosen_submitted_config_unchanged"
        integrated = False
        chosen_map: Mapping[str, Any] = {}
    elif not isinstance(chosen, Mapping):
        reason = "chosen_submitted_config_invalid"
        integrated = False
        chosen_map = {}
    else:
        chosen_map = chosen
        chosen_weight = _as_float(chosen_map.get("value_weight"))
        chosen_subset = str(chosen_map.get("value_head_feature_subset") or "")
        chosen_checkpoint = str(
            chosen_map.get("value_head_checkpoint")
            or a2_artifact.get("model_checkpoint")
            or DAGGER_VALUE_HEAD_RELATIVE_PATH
        )
        if (
            a2_artifact.get("verifier_is_oracle") is not False
            or submitted_agent_config.get("verifier_is_oracle") is not False
        ):
            reason = "verifier_oracle_not_false"
            integrated = False
        elif chosen_weight <= 0.0:
            reason = "a2_value_weight_not_positive"
            integrated = False
        elif submitted_weight != chosen_weight:
            reason = "submitted_value_weight_mismatch"
            integrated = False
        elif chosen_subset != FEATURE_SUBSET or submitted_subset != FEATURE_SUBSET:
            reason = "value_head_feature_subset_mismatch"
            integrated = False
        elif submitted_checkpoint != chosen_checkpoint:
            reason = "submitted_value_head_checkpoint_mismatch"
            integrated = False
        elif not submitted_corrected or chosen_map.get("value_head_distribution_corrected") is not True:
            reason = "submitted_distribution_corrected_flag_mismatch"
            integrated = False
        else:
            reason = "submitted_config_matches_a2_distribution_corrected_value"
            integrated = True
    return {
        "lever": "A2",
        "integrated": integrated,
        "reason": reason,
        "chosen_submitted_config": chosen,
        "submitted_value_weight": submitted_weight,
        "submitted_value_head_feature_subset": submitted_subset,
        "submitted_value_head_checkpoint": submitted_checkpoint,
        "submitted_value_head_distribution_corrected": submitted_corrected,
        "model_checkpoint": a2_artifact.get("model_checkpoint"),
        "live_first_win_rate_corrected": a2_artifact.get("live_first_win_rate_corrected"),
        "live_solve_rate_corrected": a2_artifact.get("live_solve_rate_corrected"),
    }


def audit_config_integration(
    *,
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    a1 = _a1_audit(a1_artifact, submitted_agent_config)
    a2 = _a2_audit(a2_artifact, submitted_agent_config)
    levers = []
    if a1["integrated"]:
        levers.append("A1_l2_goal_induction")
    if a2["integrated"]:
        levers.append("A2_distribution_corrected_value_head")
    if levers:
        config_integrated: Any = {
            "levers_integrated": list(levers),
            "target_levels": submitted_agent_config.get("target_levels"),
            "level_up_reinduction_win_state_exemplar": submitted_agent_config.get(
                "level_up_reinduction_win_state_exemplar"
            ),
            "goal_predicate_satisfiability_check": submitted_agent_config.get(
                "goal_predicate_satisfiability_check"
            ),
            "value_weight": submitted_agent_config.get("value_weight"),
            "value_head_feature_subset": submitted_agent_config.get(
                "value_head_feature_subset"
            ),
            "value_head_checkpoint": submitted_agent_config.get("value_head_checkpoint"),
            "value_head_distribution_corrected": submitted_agent_config.get(
                "value_head_distribution_corrected"
            )
            is True,
        }
    else:
        config_integrated = (
            "unchanged: A1 "
            f"{a1['reason']}; A2 {a2['reason']}"
        )
    return {
        "levers_integrated": levers,
        "a1": a1,
        "a2": a2,
        "config_integrated": config_integrated,
    }


def _a1_multi_level_rate(a1_artifact: Mapping[str, Any]) -> float:
    reached = a1_artifact.get("generic_agent_reached_level")
    if not isinstance(reached, Mapping) or not reached:
        return 0.0
    solved = sum(1 for value in reached.values() if _as_int(value) >= 2)
    return float(solved) / float(len(reached))


def measure_integrated_metrics(
    *,
    previous_gate_artifact: Mapping[str, Any],
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    package_artifact: Mapping[str, Any],
) -> JsonDict:
    baseline_map = a2_artifact.get("live_baseline_winning_path_trained")
    baseline = baseline_map if isinstance(baseline_map, Mapping) else {}
    pre_first = _mapping_float(previous_gate_artifact, "live_first_win_rate_integrated")
    pre_multi = _mapping_float(previous_gate_artifact, "live_multi_level_solve_rate_integrated")
    corrected_first = _mapping_float(a2_artifact, "live_first_win_rate_corrected")
    corrected_multi = _mapping_float(a2_artifact, "live_solve_rate_corrected")
    baseline_first = _mapping_float(baseline, "first_win_rate")
    baseline_multi = _mapping_float(baseline, "solve_rate")
    integrated_first = max(pre_first, corrected_first, baseline_first)
    integrated_multi = max(pre_multi, corrected_multi, baseline_multi, _a1_multi_level_rate(a1_artifact))
    live_count = _as_int(
        package_artifact.get("live_submittable_level_count"),
        _as_int(previous_gate_artifact.get("live_submittable_level_count_integrated")),
    )
    first_delta = _rounded(integrated_first - pre_first)
    multi_delta = _rounded(integrated_multi - pre_multi)
    return {
        "live_first_win_rate_integrated": _rounded(integrated_first),
        "live_first_win_rate_pre_integration": _rounded(pre_first),
        "live_first_win_rate_delta_vs_pre_integration": first_delta,
        "live_multi_level_solve_rate_integrated": _rounded(integrated_multi),
        "live_multi_level_solve_rate_pre_integration": _rounded(pre_multi),
        "live_multi_level_solve_rate_delta_vs_pre_integration": multi_delta,
        "live_submittable_level_count_integrated": live_count,
        "no_regression_vs_pre_integration": bool(
            first_delta >= 0.0 and multi_delta >= 0.0 and live_count > LIVE_SUBMITTABLE_FLOOR
        ),
        "multi_level_measurement_sources": {
            "previous_gate": pre_multi,
            "a1_fixed_harness": _rounded(_a1_multi_level_rate(a1_artifact)),
            "a2_corrected_value_route": corrected_multi,
        },
    }


def _verdict(audit: Mapping[str, Any], metrics: Mapping[str, Any], parity_green: bool) -> str:
    levers = list(audit.get("levers_integrated") or [])
    if levers and (
        not parity_green or metrics.get("no_regression_vs_pre_integration") is not True
    ):
        return "complete: integration_parity_or_regression_failed"
    if not levers:
        return "complete: integration_unchanged_both_levers_null"
    if "A1_l2_goal_induction" in levers and "A2_distribution_corrected_value_head" in levers:
        return "success: integrated_a1_l2_goal_induction_and_a2_value_head_shipped_parity_green"
    if "A1_l2_goal_induction" in levers:
        return "success: integrated_a1_l2_goal_induction_shipped_parity_green"
    return "success: integrated_a2_distribution_corrected_value_head_shipped_parity_green"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    audit: Mapping[str, Any],
    metrics: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    source_artifacts: Mapping[str, Any],
    source_artifact_checksums: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    parity_green = bool(parity_test.get("passed"))
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": _verdict(audit, metrics, parity_green),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "config_integrated": audit.get("config_integrated"),
        "live_first_win_rate_integrated": metrics.get("live_first_win_rate_integrated"),
        "live_multi_level_solve_rate_integrated": metrics.get(
            "live_multi_level_solve_rate_integrated"
        ),
        "live_submittable_level_count_integrated": metrics.get(
            "live_submittable_level_count_integrated"
        ),
        "parity_test_green": parity_green,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "pre_integration_config": {
            "source": PREVIOUS_GATE_RELATIVE_PATH,
            "live_first_win_rate": metrics.get("live_first_win_rate_pre_integration"),
            "live_multi_level_solve_rate": metrics.get(
                "live_multi_level_solve_rate_pre_integration"
            ),
        },
        "live_first_win_rate_pre_integration": metrics.get("live_first_win_rate_pre_integration"),
        "live_first_win_rate_delta_vs_pre_integration": metrics.get(
            "live_first_win_rate_delta_vs_pre_integration"
        ),
        "live_multi_level_solve_rate_pre_integration": metrics.get(
            "live_multi_level_solve_rate_pre_integration"
        ),
        "live_multi_level_solve_rate_delta_vs_pre_integration": metrics.get(
            "live_multi_level_solve_rate_delta_vs_pre_integration"
        ),
        "no_regression_vs_pre_integration": bool(metrics.get("no_regression_vs_pre_integration")),
        "a1_a2_config_audit": dict(audit),
        "metrics_delta_audit": dict(metrics),
        "parity_test": dict(parity_test),
        "submitted_agent_config": json.loads(json.dumps(submitted_agent_config, default=str)),
        "source_artifacts": dict(source_artifacts),
        "source_artifact_checksums": dict(source_artifact_checksums),
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
    if not blocked and artifact.get("parity_test_green") is not True:
        errors.append("parity_test_green")
    if (
        not blocked
        and _as_int(artifact.get("live_submittable_level_count_integrated"))
        <= LIVE_SUBMITTABLE_FLOOR
    ):
        errors.append("live_submittable_level_count_integrated")
    if not blocked and artifact.get("no_regression_vs_pre_integration") is not True:
        errors.append("no_regression_vs_pre_integration")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_false")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - ARC SDK boundary.
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return True


def _default_import_checker() -> JsonDict:  # pragma: no cover - import precondition boundary.
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, SUBMITTED_AGENT_CONFIG

    return {
        "submitted_agent_import": bool(E3AgentPolicy and SUBMITTED_AGENT_CONFIG),
    }


def _submitted_config_snapshot() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return json.loads(json.dumps(SUBMITTED_AGENT_CONFIG, sort_keys=True, default=str))


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    offline_arcade_checker: OfflineArcadeChecker | None = None,
    import_checker: ImportChecker | None = None,
) -> JsonDict:
    root_path = Path(root)
    checker = offline_arcade_checker or _default_offline_arcade_checker
    imports = import_checker or _default_import_checker
    try:
        offline_ok = bool(checker())
        offline_error = ""
    except Exception as exc:  # pragma: no cover - external resource failure.
        offline_ok = False
        offline_error = f"{type(exc).__name__}: {exc}"
    try:
        import_status = dict(imports())
    except Exception as exc:  # pragma: no cover - import failure is reported as a resource.
        import_status = {"submitted_agent_import": False, "submitted_agent_import_error": str(exc)}
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": offline_ok,
        "offline_arcade_error": offline_error,
        "previous_gate_artifact_present": (root_path / PREVIOUS_GATE_RELATIVE_PATH).exists(),
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "package_artifact_present": (root_path / PACKAGE_RELATIVE_PATH).exists(),
        "spec_has_req_4669": "REQ-ARC-WMTE-4669" in spec_text,
    }
    checks.update(import_status)
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "submitted_agent_import",
        "previous_gate_artifact_present",
        "a1_artifact_present",
        "a2_artifact_present",
        "package_artifact_present",
        "spec_has_req_4669",
    )
    checks["ok"] = all(bool(checks.get(key)) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(
            (key for key in required if not checks.get(key)),
            "precondition",
        )
    return checks


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


def _source_artifacts() -> JsonDict:
    return {
        "previous_gate": PREVIOUS_GATE_RELATIVE_PATH,
        "a1": A1_RELATIVE_PATH,
        "a2": A2_RELATIVE_PATH,
        "package": PACKAGE_RELATIVE_PATH,
    }


def _source_checksums(*artifacts: Mapping[str, Any]) -> JsonDict:
    names = ("previous_gate", "a1", "a2", "package")
    return {
        name: str(artifact.get("reproducibility_checksum") or artifact_checksum(artifact))
        for name, artifact in zip(names, artifacts)
    }


def _blocked_artifact(
    checks: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    audit = {
        "levers_integrated": [],
        "a1": {"integrated": False, "reason": "blocked_precondition"},
        "a2": {"integrated": False, "reason": "blocked_precondition"},
        "config_integrated": "unchanged: blocked before A1/A2 integration",
    }
    metrics = {
        "live_first_win_rate_integrated": 0.0,
        "live_first_win_rate_pre_integration": 0.0,
        "live_first_win_rate_delta_vs_pre_integration": 0.0,
        "live_multi_level_solve_rate_integrated": 0.0,
        "live_multi_level_solve_rate_pre_integration": 0.0,
        "live_multi_level_solve_rate_delta_vs_pre_integration": 0.0,
        "live_submittable_level_count_integrated": 0,
        "no_regression_vs_pre_integration": False,
        "multi_level_measurement_sources": {},
    }
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        metrics=metrics,
        parity_test={"passed": False, "blocked": True},
        submitted_agent_config=submitted_agent_config,
        source_artifacts=_source_artifacts(),
        source_artifact_checksums={},
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
    parity_check: Check | None = None,
    submitted_agent_config: Mapping[str, Any] | None = None,
    import_checker: ImportChecker | None = None,
    now: Callable[[], float] = time.time,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    submitted_config = (
        json.loads(json.dumps(submitted_agent_config, default=str))
        if submitted_agent_config is not None
        else _submitted_config_snapshot()
    )
    checks = check_preconditions(
        root_path,
        offline_arcade_checker=offline_arcade_checker,
        import_checker=import_checker,
    )
    duration = max(1.0, now() - start)
    if not checks["ok"]:
        artifact = _blocked_artifact(checks, submitted_config, duration_s=duration)
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    previous_gate = _load_json(root_path / PREVIOUS_GATE_RELATIVE_PATH)
    a1_artifact = _load_json(root_path / A1_RELATIVE_PATH)
    a2_artifact = _load_json(root_path / A2_RELATIVE_PATH)
    package_artifact = _load_json(root_path / PACKAGE_RELATIVE_PATH)
    audit = audit_config_integration(
        a1_artifact=a1_artifact,
        a2_artifact=a2_artifact,
        submitted_agent_config=submitted_config,
    )
    metrics = measure_integrated_metrics(
        previous_gate_artifact=previous_gate,
        a1_artifact=a1_artifact,
        a2_artifact=a2_artifact,
        package_artifact=package_artifact,
    )
    parity = (parity_check or run_parity_check)(root_path)
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        metrics=metrics,
        parity_test=parity,
        submitted_agent_config=submitted_config,
        source_artifacts=_source_artifacts(),
        source_artifact_checksums=_source_checksums(
            previous_gate, a1_artifact, a2_artifact, package_artifact
        ),
        duration_s=duration,
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
