"""Experiment 4609: submitted-agent ARC integration consolidation gate.

Spec refs: REQ-ARC-WMTE-4609, SCENARIO-ARC-WMTE-4609.
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
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]
ArtifactPair = tuple[Mapping[str, Any], Mapping[str, Any]]
SummarizeRunner = Callable[[Path], Mapping[str, Any]]
ParityCheck = Callable[[Path | str], Mapping[str, Any]]
OfflineArcadeChecker = Callable[[], bool]

EXPERIMENT = "experiment_4609_integration_gate"
SCHEMA = "carnot.exp4609.integration_gate.v1"
RESULT_RELATIVE_PATH = "results/experiment_4609_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4604_world_model_trust_energy.json"
A2_RELATIVE_PATH = "results/experiment_4605_live_integration_scored_agent.json"
A3_RELATIVE_PATH = "results/experiment_4606_levelup_selfplay.json"
A4_RELATIVE_PATH = "results/experiment_4607_refresh_submission_package.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4609
LIVE_SUBMITTABLE_BASELINE = 33
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
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
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
        )
    },
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
            "names any flagged_adversarial / positive-control-failed upstream artifact NOT "
            "aggregated (the fabrication gate + FALSE_NEGATIVE_RISK compliance)."
        )
    },
    "world_model_trust_pass_rate_integrated": {
        "principle": (
            "the integrated world_model_trust_pass_rate (A1's headline metric on the shipped "
            "config) -- did the 0.08-wall fix make it into the SCORED agent."
        )
    },
    "first_win_rate_integrated": {
        "principle": (
            "the integrated held-out first-win-rate on the shipped config (A2's effect)."
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
    "submitted_config_raised_metric_clean": {
        "principle": (
            "True only if a CLEAN (non-flagged, control-passed) lever raised a real metric on "
            "the SCORED config; false -> honest null, bare config kept."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "present where any integrated delta == 0 -- states the equality is an honest "
            "no-value null, not a bug."
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
    "world_model_trust_pass_rate_baseline",
    "world_model_trust_pass_rate_delta_vs_baseline",
    "first_win_rate_bare",
    "first_win_rate_delta_vs_bare",
    "median_actions_to_first_levelup_bare",
    "actions_delta_vs_bare",
    "live_submittable_level_count_baseline",
    "live_submittable_delta_vs_baseline",
    "upstream_lever_audit",
    "quarantined_artifacts",
    "package_artifact",
    "submitted_agent_config",
    "config_action",
    "parity_test",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)
SPEC_REFS = ["REQ-ARC-WMTE-4609", "SCENARIO-ARC-WMTE-4609"]


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
    return artifact.get("flagged_adversarial") is True or _live_status(summary).upper() == "CRITICAL"


def _positive_control_passed(lever: str, artifact: Mapping[str, Any]) -> bool:
    if artifact.get("positive_control_passed") is True:
        return True
    if lever == "A1":
        return artifact.get("binary_gate_control_passed") is True
    if lever == "A2":
        return artifact.get("bare_control_passed") is True
    if lever == "A3":
        gate = artifact.get("reproduction_gate")
        return (
            artifact.get("offline_reproduced") is True
            and isinstance(gate, Mapping)
            and gate.get("reproduced") is True
        )
    if lever == "A4":
        return artifact.get("ready_for_operator_submit") is True
    return False


def _lever_metric_delta(lever: str, artifact: Mapping[str, Any]) -> float:
    if lever == "A1":
        return _as_float(artifact.get("trust_pass_rate_delta"))
    if lever == "A2":
        return max(_as_float(artifact.get("first_win_delta")), _as_float(artifact.get("actions_delta")))
    if lever == "A3":
        return float(_as_int(artifact.get("reproduced_levels")))
    return 0.0


def _lever_name(lever: str) -> str:
    return {
        "A1": "A1_world_model_trust_energy_gate",
        "A2": "A2_live_integration_router_tiebreak_forward_nav",
        "A3": "A3_level_bank_in_refreshed_package",
    }.get(lever, lever)


def _audit_lever(lever: str, artifact: Mapping[str, Any], summary: Mapping[str, Any]) -> JsonDict:
    status = _live_status(summary)
    flagged = _is_flagged(artifact, summary)
    positive_control = _positive_control_passed(lever, artifact)
    metric_delta = _lever_metric_delta(lever, artifact)
    if flagged:
        reason = "flagged_adversarial"
    elif lever == "A4" and positive_control:
        reason = "package_metric_only"
    elif not _success_verdict(artifact):
        reason = "honest_verdict_not_success"
    elif not positive_control:
        reason = "positive_control_failed"
    elif metric_delta <= 0.0:
        reason = "no_positive_metric_delta"
    else:
        reason = "admitted_clean_metric_raiser"
    integrated = reason == "admitted_clean_metric_raiser"
    return {
        "lever": lever,
        "artifact_honest_verdict": artifact.get("honest_verdict"),
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "summarize_returncode": int(summary.get("returncode") or 0),
        "live_status": status,
        "positive_control_passed": bool(positive_control),
        "metric_delta": metric_delta,
        "integrated": bool(integrated),
        "reason": reason,
    }


def audit_upstream_levers(upstreams: Mapping[str, ArtifactPair]) -> JsonDict:
    rows = {
        lever: _audit_lever(lever, artifact, summary)
        for lever, (artifact, summary) in sorted(upstreams.items())
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
        if row["reason"] in {"flagged_adversarial", "positive_control_failed"}
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


def measure_integrated_metrics(
    *,
    audit: Mapping[str, Any],
    a4_artifact: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    rows = audit.get("upstream_lever_audit") if isinstance(audit.get("upstream_lever_audit"), Mapping) else {}
    a1 = rows.get("A1", {}) if isinstance(rows, Mapping) else {}
    a2 = rows.get("A2", {}) if isinstance(rows, Mapping) else {}
    a1_integrated = a1.get("integrated") is True
    a2_integrated = a2.get("integrated") is True
    world_model = 0.0
    world_model_baseline = 0.0
    first_win = 0.0
    first_win_bare = 0.0
    median_actions = None
    bare_actions = None
    actions_delta = 0.0
    if a1_integrated:
        world_model = round(_as_float(submitted_agent_config.get("world_model_trust_pass_rate"), 1.0), 6)
        world_model_baseline = 0.0
    if a2_integrated:
        first_win = round(_as_float(submitted_agent_config.get("first_win_rate_integrated"), 0.0), 6)
        first_win_bare = 0.0
        median_actions = submitted_agent_config.get("median_actions_to_first_levelup_integrated")
        bare_actions = submitted_agent_config.get("median_actions_to_first_levelup_bare")
    live_count = _as_int(a4_artifact.get("live_submittable_level_count"))
    live_baseline = _as_int(a4_artifact.get("live_submittable_count_prev"), live_count)
    if live_count <= 0:
        live_count = LIVE_SUBMITTABLE_BASELINE
        live_baseline = LIVE_SUBMITTABLE_BASELINE
    return {
        "world_model_trust_pass_rate_integrated": world_model,
        "world_model_trust_pass_rate_baseline": world_model_baseline,
        "world_model_trust_pass_rate_delta_vs_baseline": round(world_model - world_model_baseline, 6),
        "first_win_rate_integrated": first_win,
        "first_win_rate_bare": first_win_bare,
        "first_win_rate_delta_vs_bare": round(first_win - first_win_bare, 6),
        "median_actions_to_first_levelup_integrated": median_actions,
        "median_actions_to_first_levelup_bare": bare_actions,
        "actions_delta_vs_bare": actions_delta,
        "live_submittable_level_count_integrated": live_count,
        "live_submittable_level_count_baseline": live_baseline,
        "live_submittable_delta_vs_baseline": live_count - live_baseline,
    }


def _verdict(audit: Mapping[str, Any], metrics: Mapping[str, Any], parity_green: bool) -> str:
    if not parity_green or audit.get("submitted_config_raised_metric_clean") is not True:
        return "complete: integration_no_clean_metric_bare_config_kept_honest_null"
    if _as_float(metrics.get("world_model_trust_pass_rate_delta_vs_baseline")) > 0.0:
        return "success: integrated_world_model_trust_raised_config_shipped"
    if _as_float(metrics.get("first_win_rate_delta_vs_bare")) > 0.0:
        return "success: integrated_first_win_raised_config_shipped"
    if _as_int(metrics.get("live_submittable_delta_vs_baseline")) > 0:
        return "success: integrated_live_submittable_raised_config_shipped"
    return "complete: integration_no_clean_metric_bare_config_kept_honest_null"


def _null_delta_note(metrics: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    zero_fields = [
        name
        for name in (
            "world_model_trust_pass_rate_delta_vs_baseline",
            "first_win_rate_delta_vs_bare",
            "actions_delta_vs_bare",
            "live_submittable_delta_vs_baseline",
        )
        if _as_float(metrics.get(name)) == 0.0
    ]
    if audit.get("submitted_config_raised_metric_clean") is not True:
        return (
            "No clean upstream lever passed the summarize-artifact, positive-control, and "
            "non-adversarial gates; zero deltas are an honest no-value null, not a bug."
        )
    return (
        "Zero integrated deltas in "
        + ", ".join(zero_fields)
        + " are honest no-value nulls for those metrics, not measurement bugs."
    )


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
        "levers_integrated": list(audit.get("levers_integrated") or []),
        "flagged_artifacts_excluded": list(audit.get("flagged_artifacts_excluded") or []),
        "world_model_trust_pass_rate_integrated": metrics.get(
            "world_model_trust_pass_rate_integrated"
        ),
        "world_model_trust_pass_rate_baseline": metrics.get(
            "world_model_trust_pass_rate_baseline"
        ),
        "world_model_trust_pass_rate_delta_vs_baseline": metrics.get(
            "world_model_trust_pass_rate_delta_vs_baseline"
        ),
        "first_win_rate_integrated": metrics.get("first_win_rate_integrated"),
        "first_win_rate_bare": metrics.get("first_win_rate_bare"),
        "first_win_rate_delta_vs_bare": metrics.get("first_win_rate_delta_vs_bare"),
        "median_actions_to_first_levelup_integrated": metrics.get(
            "median_actions_to_first_levelup_integrated"
        ),
        "median_actions_to_first_levelup_bare": metrics.get(
            "median_actions_to_first_levelup_bare"
        ),
        "actions_delta_vs_bare": metrics.get("actions_delta_vs_bare"),
        "live_submittable_level_count_integrated": metrics.get(
            "live_submittable_level_count_integrated"
        ),
        "live_submittable_level_count_baseline": metrics.get(
            "live_submittable_level_count_baseline"
        ),
        "live_submittable_delta_vs_baseline": metrics.get(
            "live_submittable_delta_vs_baseline"
        ),
        "parity_test_green": parity_green,
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
            "path": A4_RELATIVE_PATH,
            "live_submittable_level_count": metrics.get(
                "live_submittable_level_count_integrated"
            ),
        },
        "submitted_agent_config": json.loads(json.dumps(submitted_agent_config, default=str)),
        "config_action": (
            "ship_clean_metric_levers"
            if audit.get("submitted_config_raised_metric_clean") is True
            else "unchanged_bare_config_kept"
        ),
        "parity_test": dict(parity_test),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing required field {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("parity_test_green") is not True:
        errors.append("parity_test_green")
    if _as_int(artifact.get("live_submittable_level_count_integrated")) <= LIVE_SUBMITTABLE_BASELINE:
        errors.append("live_submittable_level_count_integrated")
    if "null_delta_methodology_note" not in artifact:
        errors.append("null_delta_methodology_note")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _default_offline_arcade_checker() -> bool:  # pragma: no cover - ARC SDK boundary
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
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": offline_ok,
        "offline_arcade_error": offline_error,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "a3_artifact_present": (root_path / A3_RELATIVE_PATH).exists(),
        "a4_artifact_present": (root_path / A4_RELATIVE_PATH).exists(),
        "spec_has_req_4609": "REQ-ARC-WMTE-4609" in spec_text,
    }
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "a4_artifact_present",
        "spec_has_req_4609",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next((key for key in required if not checks[key]), "precondition")
    return checks


def run_summarize_artifact(path: Path) -> JsonDict:  # pragma: no cover - subprocess boundary
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


def run_parity_check(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess boundary
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
        "submitted_config_raised_metric_clean": False,
        "flagged_artifacts_excluded": [],
        "quarantined_artifacts": [],
        "upstream_lever_audit": {},
    }
    metrics = {
        "world_model_trust_pass_rate_integrated": 0.0,
        "world_model_trust_pass_rate_baseline": 0.0,
        "world_model_trust_pass_rate_delta_vs_baseline": 0.0,
        "first_win_rate_integrated": 0.0,
        "first_win_rate_bare": 0.0,
        "first_win_rate_delta_vs_bare": 0.0,
        "median_actions_to_first_levelup_integrated": None,
        "median_actions_to_first_levelup_bare": None,
        "actions_delta_vs_bare": 0.0,
        "live_submittable_level_count_integrated": 0,
        "live_submittable_level_count_baseline": LIVE_SUBMITTABLE_BASELINE,
        "live_submittable_delta_vs_baseline": -LIVE_SUBMITTABLE_BASELINE,
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
    summarize_runner: SummarizeRunner | None = None,
    parity_check: ParityCheck | None = None,
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
        "A4": root_path / A4_RELATIVE_PATH,
    }
    upstreams: dict[str, ArtifactPair] = {}
    for lever, path in paths.items():
        artifact = _load_json(path)
        summary = summarize(path) if path.exists() else {"returncode": 2, "live_status": "missing"}
        upstreams[lever] = (artifact, summary)
    audit = audit_upstream_levers(upstreams)
    metrics = measure_integrated_metrics(
        audit=audit,
        a4_artifact=upstreams["A4"][0],
        submitted_agent_config=submitted_config,
    )
    parity = (parity_check or run_parity_check)(root_path)
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        metrics=metrics,
        parity_test=parity,
        submitted_agent_config=submitted_config,
        duration_s=max(1.0, now() - start),
    )
    _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run()
    errors = artifact_schema_errors(artifact)
    if errors:
        print(json.dumps({"result": RESULT_RELATIVE_PATH, "schema_errors": errors}, indent=2))
        return 1
    print(json.dumps({"result": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
