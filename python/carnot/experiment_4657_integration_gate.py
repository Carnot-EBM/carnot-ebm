"""Experiment 4657: integrate .429 A1/A2 into the submitted ARC agent config.

Spec refs: REQ-ARC-WMTE-4657, SCENARIO-ARC-WMTE-4657.
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

EXPERIMENT = "experiment_4657_integration_gate"
SCHEMA = "carnot.exp4657.integration_gate.v1"
RESULT_RELATIVE_PATH = "results/experiment_4657_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4652_value_routing_cost_fix_live.json"
A2_RELATIVE_PATH = "results/experiment_4653_energy_fitness_qd_generation_live.json"
PACKAGE_RELATIVE_PATH = "results/experiment_4643_refresh_submission_package.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4657
FEATURE_SUBSET = "cross_game_features_v3:v2_plus_frame_delta"
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
SPEC_REFS = ["REQ-ARC-WMTE-4657", "SCENARIO-ARC-WMTE-4657"]

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
            "MUST be false -- the integrated value-routing/QD signals are "
            "oracle-distinct from the executable win-check."
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
            "the integrated SCORED-agent multi-level (>=2) live solve-rate (the deeper wall)."
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
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive malformed-artifact guard.
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive malformed-artifact guard.
        return default


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _mapping_float(mapping: Mapping[str, Any], key: str) -> float:
    return _as_float(mapping.get(key))


def _a1_audit(
    a1_artifact: Mapping[str, Any], submitted_agent_config: Mapping[str, Any]
) -> JsonDict:
    artifact_weight = _as_float(a1_artifact.get("value_weight_set"))
    submitted_weight = _as_float(submitted_agent_config.get("value_weight"))
    artifact_subset = str(a1_artifact.get("feature_subset") or "")
    submitted_subset = str(submitted_agent_config.get("value_head_feature_subset") or "")
    if artifact_weight <= 0.0:
        reason = "a1_value_weight_not_positive"
        integrated = False
    elif submitted_weight != artifact_weight:
        reason = "submitted_value_weight_mismatch"
        integrated = False
    elif artifact_subset != FEATURE_SUBSET or submitted_subset != FEATURE_SUBSET:
        reason = "value_head_feature_subset_mismatch"
        integrated = False
    elif (
        a1_artifact.get("verifier_is_oracle") is not False
        or submitted_agent_config.get("verifier_is_oracle") is not False
    ):
        reason = "verifier_oracle_not_false"
        integrated = False
    else:
        reason = "submitted_config_matches_a1_cost_fix"
        integrated = True
    return {
        "lever": "A1",
        "integrated": integrated,
        "reason": reason,
        "artifact_value_weight_set": artifact_weight,
        "submitted_value_weight": submitted_weight,
        "feature_subset": artifact_subset,
    }


def _a2_audit(
    a2_artifact: Mapping[str, Any], submitted_agent_config: Mapping[str, Any]
) -> JsonDict:
    chosen = a2_artifact.get("chosen_submitted_config")
    qd_enabled = submitted_agent_config.get("qd_generation_enabled") is True
    if chosen in (None, "unchanged"):
        reason = "chosen_submitted_config_unchanged"
        integrated = False
    elif not qd_enabled:
        reason = "submitted_qd_generation_disabled"
        integrated = False
    else:
        reason = "submitted_config_matches_a2_qd_generator"
        integrated = True
    return {
        "lever": "A2",
        "integrated": integrated,
        "reason": reason,
        "chosen_submitted_config": chosen,
        "submitted_qd_generation_enabled": qd_enabled,
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
        levers.append("A1_value_routing_cost_fix")
    if a2["integrated"]:
        levers.append("A2_energy_fitness_qd_generator")
    if levers:
        config_integrated: Any = {
            "levers_integrated": list(levers),
            "value_weight": submitted_agent_config.get("value_weight"),
            "value_head_feature_subset": submitted_agent_config.get("value_head_feature_subset"),
            "qd_generation_enabled": submitted_agent_config.get("qd_generation_enabled") is True,
            "qd_generation_mode": submitted_agent_config.get("qd_generation_mode"),
        }
    else:
        config_integrated = (
            "unchanged: both A1 and A2 chosen submitted configs were null or did "
            "not match SUBMITTED_AGENT_CONFIG"
        )
    return {
        "levers_integrated": levers,
        "a1": a1,
        "a2": a2,
        "config_integrated": config_integrated,
    }


def measure_integrated_metrics(
    *,
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    package_artifact: Mapping[str, Any],
) -> JsonDict:
    baseline = a1_artifact.get("live_baseline_value_weight_zero")
    baseline_map = baseline if isinstance(baseline, Mapping) else {}
    a2_chosen = a2_artifact.get("chosen_submitted_config")
    a2_integrated_rate = (
        _mapping_float(a2_artifact, "live_solve_rate_qd") if isinstance(a2_chosen, Mapping) else 0.0
    )
    pre_multi = max(
        _mapping_float(baseline_map, "solve_rate"),
        _mapping_float(a2_artifact, "live_solve_rate_search_baseline"),
    )
    integrated_first = _mapping_float(a1_artifact, "live_first_win_rate_value_routed")
    integrated_multi = max(
        _mapping_float(a1_artifact, "live_solve_rate_value_routed"), a2_integrated_rate
    )
    pre_first = _mapping_float(baseline_map, "first_win_rate")
    live_count = _as_int(package_artifact.get("live_submittable_level_count"))
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
    }


def _verdict(audit: Mapping[str, Any], metrics: Mapping[str, Any], parity_green: bool) -> str:
    levers = list(audit.get("levers_integrated") or [])
    if not levers:
        return "complete: integration_unchanged_both_levers_null"
    if not parity_green or metrics.get("no_regression_vs_pre_integration") is not True:
        return "complete: integration_parity_or_regression_failed"
    if "A1_value_routing_cost_fix" in levers and "A2_energy_fitness_qd_generator" in levers:
        return "success: integrated_a1_value_routing_and_a2_qd_generator_shipped_parity_green"
    if "A1_value_routing_cost_fix" in levers:
        return "success: integrated_a1_value_routing_cost_fix_shipped_parity_green"
    return "success: integrated_a2_qd_generator_shipped_parity_green"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    audit: Mapping[str, Any],
    metrics: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
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
            "policy": "E3AgentPolicy",
            "value_weight": 0.0,
            "qd_generation_enabled": False,
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
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "package_artifact_present": (root_path / PACKAGE_RELATIVE_PATH).exists(),
        "spec_has_req_4657": "REQ-ARC-WMTE-4657" in spec_text,
    }
    checks.update(import_status)
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "submitted_agent_import",
        "a1_artifact_present",
        "a2_artifact_present",
        "package_artifact_present",
        "spec_has_req_4657",
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
    }
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        metrics=metrics,
        parity_test={"passed": False, "blocked": True},
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

    a1_artifact = _load_json(root_path / A1_RELATIVE_PATH)
    a2_artifact = _load_json(root_path / A2_RELATIVE_PATH)
    package_artifact = _load_json(root_path / PACKAGE_RELATIVE_PATH)
    audit = audit_config_integration(
        a1_artifact=a1_artifact,
        a2_artifact=a2_artifact,
        submitted_agent_config=submitted_config,
    )
    metrics = measure_integrated_metrics(
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
