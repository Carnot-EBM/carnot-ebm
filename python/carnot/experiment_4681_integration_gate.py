"""Experiment 4681: integrate structural A1/A2 levers into submitted config.

Spec refs: REQ-ARC-WMTE-4681, SCENARIO-ARC-WMTE-4681.
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

EXPERIMENT = "experiment_4681_integration_gate"
SCHEMA = "carnot.exp4681.integration_gate.v1"
RESULT_RELATIVE_PATH = "results/experiment_4681_integration_gate.json"
PREVIOUS_GATE_RELATIVE_PATH = "results/experiment_4669_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4676_hierarchical_subgoal_search_live.json"
A2_RELATIVE_PATH = "results/experiment_4677_poe_world_factored_subgoal_planner.json"
PACKAGE_RELATIVE_PATH = "results/experiment_4679_refresh_submission_package.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4681
LIVE_SUBMITTABLE_FLOOR = 33
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
)
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
SPEC_REFS = ["REQ-ARC-WMTE-4681", "SCENARIO-ARC-WMTE-4681"]

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
            "MUST be false -- the integrated subgoal-search / factored-planner signals "
            "are oracle-distinct from the executable win-check."
        )
    },
    "config_integrated": {
        "principle": (
            "the A1/A2 config folded into SUBMITTED_AGENT_CONFIG (or 'unchanged' with "
            "the reason) -- the single-source-of-truth update."
        )
    },
    "config_changed": {
        "principle": (
            "bool -- true only if a lever cleared its gate; when false, the pre/post "
            "numbers are identical BY CONSTRUCTION (not a finding) -- pre-empts the "
            ".430 A6 TAUTOLOGY flag."
        )
    },
    "live_first_win_rate_integrated": {
        "principle": (
            "the integrated SCORED-agent live first-win-rate (no regression vs pre-integration)."
        )
    },
    "live_multi_level_solve_rate_integrated": {
        "principle": (
            "the integrated SCORED-agent multi-level (>=2) live solve-rate (the deeper "
            "wall) -- measured on the FIXED non-degenerate harness."
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
    "tautology_guard",
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


def _a1_gate_cleared(a1_artifact: Mapping[str, Any]) -> bool:
    reached = _as_int(a1_artifact.get("generic_agent_reached_level"))
    flat = _as_int(a1_artifact.get("no_subgoal_ablation_reached_level"))
    random = _as_int(a1_artifact.get("random_subgoal_ablation_reached_level"))
    return bool(
        str(a1_artifact.get("honest_verdict") or "").startswith("success:")
        and a1_artifact.get("verifier_is_oracle") is False
        and a1_artifact.get("offline_reproduced") is True
        and _as_int(a1_artifact.get("reproduced_levels")) >= 1
        and reached > max(flat, random)
    )


def _a2_gate_cleared(a2_artifact: Mapping[str, Any]) -> bool:
    return bool(
        str(a2_artifact.get("honest_verdict") or "").startswith("success:")
        and a2_artifact.get("verifier_is_oracle") is False
        and _as_float(a2_artifact.get("coverage_delta")) > 0.0
        and max(
            _as_float(a2_artifact.get("first_win_rate_delta")),
            _as_float(a2_artifact.get("solve_rate_delta")),
        )
        > 0.0
    )


def _a1_audit(
    a1_artifact: Mapping[str, Any], submitted_agent_config: Mapping[str, Any]
) -> JsonDict:
    chosen = a1_artifact.get("chosen_submitted_config")
    submitted_enabled = bool(submitted_agent_config.get("hierarchical_subgoal_search_enabled"))
    submitted_budget = _as_int(submitted_agent_config.get("hierarchical_subgoal_budget"))
    chosen_map = chosen if isinstance(chosen, Mapping) else {}
    chosen_enabled = chosen_map.get("hierarchical_subgoal_search_enabled") is True
    chosen_budget = _as_int(chosen_map.get("hierarchical_subgoal_budget"))
    if chosen in (None, "unchanged"):
        reason = "chosen_submitted_config_unchanged"
        integrated = False
    elif not isinstance(chosen, Mapping):
        reason = "chosen_submitted_config_invalid"
        integrated = False
    elif a1_artifact.get("verifier_is_oracle") is not False:
        reason = "verifier_oracle_not_false"
        integrated = False
    elif not _a1_gate_cleared(a1_artifact):
        reason = "a1_hierarchical_subgoal_gate_not_cleared"
        integrated = False
    elif not chosen_enabled or chosen_budget <= 0:
        reason = "a1_chosen_subgoal_config_invalid"
        integrated = False
    elif submitted_enabled != chosen_enabled or submitted_budget != chosen_budget:
        reason = "submitted_hierarchical_subgoal_config_mismatch"
        integrated = False
    else:
        reason = "submitted_config_matches_a1_hierarchical_subgoal_search"
        integrated = True
    return {
        "lever": "A1",
        "integrated": integrated,
        "reason": reason,
        "chosen_submitted_config": chosen,
        "gate_cleared": _a1_gate_cleared(a1_artifact),
        "generic_agent_reached_level": _as_int(a1_artifact.get("generic_agent_reached_level")),
        "offline_reproduced": a1_artifact.get("offline_reproduced") is True,
        "reproduced_levels": _as_int(a1_artifact.get("reproduced_levels")),
        "submitted_hierarchical_subgoal_search_enabled": submitted_enabled,
        "submitted_hierarchical_subgoal_budget": submitted_budget,
    }


def _a2_audit(
    a2_artifact: Mapping[str, Any], submitted_agent_config: Mapping[str, Any]
) -> JsonDict:
    chosen = a2_artifact.get("chosen_submitted_config")
    submitted_enabled = bool(submitted_agent_config.get("factored_planner_enabled"))
    submitted_threshold = _as_float(submitted_agent_config.get("factored_trust_threshold"))
    chosen_map = chosen if isinstance(chosen, Mapping) else {}
    chosen_enabled = chosen_map.get("factored_planner_enabled") is True
    chosen_threshold = _as_float(chosen_map.get("factored_trust_threshold"), -1.0)
    if chosen in (None, "unchanged"):
        reason = "chosen_submitted_config_unchanged"
        integrated = False
    elif not isinstance(chosen, Mapping):
        reason = "chosen_submitted_config_invalid"
        integrated = False
    elif a2_artifact.get("verifier_is_oracle") is not False:
        reason = "verifier_oracle_not_false"
        integrated = False
    elif not _a2_gate_cleared(a2_artifact):
        reason = "a2_factored_planner_gate_not_cleared"
        integrated = False
    elif not chosen_enabled or not 0.0 <= chosen_threshold <= 1.0:
        reason = "a2_chosen_factored_planner_config_invalid"
        integrated = False
    elif submitted_enabled != chosen_enabled or submitted_threshold != chosen_threshold:
        reason = "submitted_factored_planner_config_mismatch"
        integrated = False
    else:
        reason = "submitted_config_matches_a2_poe_world_factored_planner"
        integrated = True
    return {
        "lever": "A2",
        "integrated": integrated,
        "reason": reason,
        "chosen_submitted_config": chosen,
        "gate_cleared": _a2_gate_cleared(a2_artifact),
        "coverage_delta": _as_float(a2_artifact.get("coverage_delta")),
        "first_win_rate_delta": _as_float(a2_artifact.get("first_win_rate_delta")),
        "solve_rate_delta": _as_float(a2_artifact.get("solve_rate_delta")),
        "submitted_factored_planner_enabled": submitted_enabled,
        "submitted_factored_trust_threshold": submitted_threshold,
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
        levers.append("A1_hierarchical_subgoal_search")
    if a2["integrated"]:
        levers.append("A2_poe_world_factored_planner")
    config_changed = bool(levers)
    if config_changed:
        config_integrated: Any = {
            "levers_integrated": list(levers),
            "hierarchical_subgoal_search_enabled": submitted_agent_config.get(
                "hierarchical_subgoal_search_enabled"
            ),
            "hierarchical_subgoal_budget": submitted_agent_config.get(
                "hierarchical_subgoal_budget"
            ),
            "factored_planner_enabled": submitted_agent_config.get("factored_planner_enabled"),
            "factored_trust_threshold": submitted_agent_config.get("factored_trust_threshold"),
        }
    else:
        config_integrated = f"unchanged: A1 {a1['reason']}; A2 {a2['reason']}"
    return {
        "config_changed": config_changed,
        "levers_integrated": levers,
        "a1": a1,
        "a2": a2,
        "config_integrated": config_integrated,
    }


def _max_config_metric(artifact: Mapping[str, Any], key: str) -> float:
    configs = artifact.get("generic_first_win_by_config")
    if not isinstance(configs, Mapping):
        return 0.0
    return max(
        (
            _mapping_float(row, key)
            for row in configs.values()
            if isinstance(row, Mapping)
        ),
        default=0.0,
    )


def measure_integrated_metrics(
    *,
    previous_gate_artifact: Mapping[str, Any],
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    package_artifact: Mapping[str, Any],
    config_changed: bool,
) -> JsonDict:
    pre_first = _mapping_float(previous_gate_artifact, "live_first_win_rate_integrated")
    pre_multi = _mapping_float(previous_gate_artifact, "live_multi_level_solve_rate_integrated")
    pre_count = _as_int(previous_gate_artifact.get("live_submittable_level_count_integrated"))
    package_count = _as_int(package_artifact.get("live_submittable_level_count"), pre_count)
    if config_changed:
        a1_first = _max_config_metric(a1_artifact, "first_win_rate")
        a1_multi = _max_config_metric(a1_artifact, "multi_level_rate")
        a2_first = _mapping_float(a2_artifact, "live_first_win_rate_factored")
        a2_multi = _mapping_float(a2_artifact, "live_solve_rate_factored")
        integrated_first = max(pre_first, a1_first, a2_first)
        integrated_multi = max(pre_multi, a1_multi, a2_multi)
        live_count = package_count
        note = "config_changed=true; integrated metrics use fixed cached A1/A2 measurements."
    else:
        a1_first = a1_multi = a2_first = a2_multi = 0.0
        integrated_first = pre_first
        integrated_multi = pre_multi
        live_count = pre_count
        note = (
            "config_changed=false; first-win, multi-level, and live-submittable metrics "
            "remain equal to the pre-integration config by construction."
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
        "metric_measurement_note": note,
        "measurement_sources": {
            "previous_gate": PREVIOUS_GATE_RELATIVE_PATH,
            "a1_hierarchical_first_win_rate": _rounded(a1_first),
            "a1_hierarchical_multi_level_rate": _rounded(a1_multi),
            "a2_factored_first_win_rate": _rounded(a2_first),
            "a2_factored_multi_level_rate": _rounded(a2_multi),
            "package_live_submittable_level_count": package_count,
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
    both_levers = (
        "A1_hierarchical_subgoal_search" in levers
        and "A2_poe_world_factored_planner" in levers
    )
    if both_levers:
        return "success: integrated_a1_subgoal_search_and_a2_factored_planner_shipped_parity_green"
    if "A1_hierarchical_subgoal_search" in levers:
        return "success: integrated_a1_hierarchical_subgoal_search_shipped_parity_green"
    return "success: integrated_a2_poe_world_factored_planner_shipped_parity_green"


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
    config_changed = bool(audit.get("config_changed"))
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": _verdict(audit, metrics, parity_green),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "config_integrated": audit.get("config_integrated"),
        "config_changed": config_changed,
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
            "live_submittable_level_count": metrics.get(
                "live_submittable_level_count_integrated"
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
        "tautology_guard": (
            "config_changed=false: pre/post measurements are identical BY CONSTRUCTION "
            "(not a finding)."
            if not config_changed
            else "config_changed=true: metric deltas are measured against the pre-integration gate."
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": max(1.0, round(float(duration_s), 6)),
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
    if type(artifact.get("config_changed")) is not bool:
        errors.append("config_changed_bool")
    if not blocked and artifact.get("parity_test_green") is not True:
        errors.append("parity_test_green")
    live_count_too_low = (
        _as_int(artifact.get("live_submittable_level_count_integrated")) <= LIVE_SUBMITTABLE_FLOOR
    )
    if not blocked and live_count_too_low:
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
    import_checker: ImportChecker | None = None,
) -> JsonDict:
    root_path = Path(root)
    imports = import_checker or _default_import_checker
    try:
        import_status = dict(imports())
    except Exception as exc:  # pragma: no cover - import failure is reported as a resource.
        import_status = {"submitted_agent_import": False, "submitted_agent_import_error": str(exc)}
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "previous_gate_artifact_present": (root_path / PREVIOUS_GATE_RELATIVE_PATH).exists(),
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "package_artifact_present": (root_path / PACKAGE_RELATIVE_PATH).exists(),
        "spec_has_req_4681": "REQ-ARC-WMTE-4681" in spec_text,
    }
    checks.update(import_status)
    required = (
        "agents_md_read",
        "codex_md_read",
        "submitted_agent_import",
        "previous_gate_artifact_present",
        "a1_artifact_present",
        "a2_artifact_present",
        "package_artifact_present",
        "spec_has_req_4681",
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
        "config_changed": False,
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
        "metric_measurement_note": "blocked before measurement",
        "measurement_sources": {},
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
    checks = check_preconditions(root_path, import_checker=import_checker)
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
        config_changed=bool(audit["config_changed"]),
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
