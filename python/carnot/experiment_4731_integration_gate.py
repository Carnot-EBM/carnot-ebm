"""Experiment 4731: integrate .435 A1/A2 submitted-agent changes.

Spec refs: REQ-ARC-WMTE-4731,
SCENARIO-ARC-WMTE-4731-HONEST-NULL-INTEGRATION.
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
PreconditionChecker = Callable[[Path], Mapping[str, Any]]
ScoredMeasure = Callable[[Path | str], Mapping[str, Any]]

EXPERIMENT = "experiment_4731_integration_gate"
SCHEMA = "carnot.exp4731.integration_gate.v1"
RESULT_RELATIVE_PATH = "results/experiment_4731_integration_gate.json"
PREVIOUS_GATE_RELATIVE_PATH = "results/experiment_4705_integration_gate.json"
A1_RELATIVE_PATH = "results/experiment_4726_online_action_learning_driver_valid_test.json"
A2_RELATIVE_PATH = "results/experiment_4727_active_probe_disambiguation.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4731
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- scores the integrated config over cached "
    "variants (1s floor)."
)
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
SPEC_REFS = ["REQ-ARC-WMTE-4731", "SCENARIO-ARC-WMTE-4731-HONEST-NULL-INTEGRATION"]

LEVER_LABELS = {
    "a1": "A1_online_action_learning_driver",
    "a2": "A2_active_probe_controller",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: integrated_<change>_first_win_<delta> OR complete: "
            "integration_no_change_all_levers_unchanged."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "integrated_change": {
        "principle": (
            "the change integrated into SUBMITTED_AGENT_CONFIG (or 'none' if all A1/A2 "
            "selected unchanged)."
        )
    },
    "live_first_win_rate_integrated": {
        "principle": "the held-out first-win of the integrated config -- the scored-lane proxy."
    },
    "live_first_win_rate_pre_integration": {
        "principle": "the pre-integration baseline -- the no-regression control."
    },
    "parity_test_green": {
        "principle": (
            "HARD gate -- the integrated agent is byte-for-byte the SUBMITTED_AGENT_CONFIG; "
            "a parity miss invalidates the integration."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when both deltas ~0; the TAUTOLOGY carve-out reads it (honest "
            "no-change, not a measurement bug)."
        )
    },
    "positive_control_passed": {
        "principle": (
            "bool(parity_test_green AND no_regression) -- GATES the TAUTOLOGY exemption; "
            "an unvalidated integration is NOT excused."
        )
    },
    "verifier_is_oracle": {
        "principle": "false -- the integration gate measures the agent; no oracle is invoked."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, A1/A2 artifacts present); "
            "pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "spec_refs",
    "result_path",
    "live_first_win_rate_delta_vs_pre_integration",
    "no_regression_vs_pre_integration",
    "config_changed",
    "selected_submitted_config",
    "a1_a2_config_audit",
    "metrics_delta_audit",
    "parity_test",
    "submitted_agent_config",
    "source_artifacts",
    "source_artifact_checksums",
    "field_principles",
    "duration_s",
    "submitted_to_leaderboard",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {}


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


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _first_present_float(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in mapping:
            return _as_float(mapping.get(key))
    return 0.0


def _held_out_lift(lever_key: str, artifact: Mapping[str, Any]) -> float:
    common_keys = ("held_out_first_win_lift", "first_win_rate_delta", "first_win_delta")
    if lever_key == "a1":
        return _rounded(
            _first_present_float(
                artifact,
                ("online_warm_vs_frozen_delta", "online_warm_delta_vs_frozen") + common_keys,
            )
        )
    return _rounded(
        _first_present_float(
            artifact,
            ("active_probe_first_win_delta", "probe_first_win_delta") + common_keys,
        )
    )


def _config_matches(chosen: Mapping[str, Any], submitted: Mapping[str, Any]) -> bool:
    return all(key in submitted and submitted.get(key) == value for key, value in chosen.items())


def _lever_audit(
    lever_key: str,
    artifact: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    chosen = artifact.get("chosen_submitted_config")
    lift = _held_out_lift(lever_key, artifact)
    if chosen in (None, "unchanged"):
        reason = "chosen_submitted_config_unchanged"
        eligible = False
    elif not isinstance(chosen, Mapping):
        reason = "chosen_submitted_config_invalid"
        eligible = False
    elif artifact.get("verifier_is_oracle") is not False:
        reason = "verifier_oracle_not_false"
        eligible = False
    elif not str(artifact.get("honest_verdict") or "").startswith("success:"):
        reason = "upstream_not_success"
        eligible = False
    elif lift <= 0.0:
        reason = "held_out_lift_not_positive"
        eligible = False
    elif not _config_matches(chosen, submitted_agent_config):
        reason = "submitted_config_mismatch"
        eligible = False
    else:
        reason = "eligible_non_unchanged_held_out_lift"
        eligible = True
    return {
        "lever": lever_key.upper(),
        "change_label": LEVER_LABELS[lever_key],
        "integrated": False,
        "eligible": eligible,
        "reason": reason,
        "chosen_submitted_config": chosen,
        "held_out_first_win_lift": lift,
        "honest_verdict": artifact.get("honest_verdict", ""),
        "parity_test_green": artifact.get("parity_test_green") is True,
        "verifier_is_oracle": artifact.get("verifier_is_oracle"),
    }


def audit_config_integration(
    *,
    a1_artifact: Mapping[str, Any],
    a2_artifact: Mapping[str, Any],
    submitted_agent_config: Mapping[str, Any],
) -> JsonDict:
    a1 = _lever_audit("a1", a1_artifact, submitted_agent_config)
    a2 = _lever_audit("a2", a2_artifact, submitted_agent_config)
    candidates = [row for row in (a1, a2) if row["eligible"]]
    selected = max(candidates, key=lambda row: row["held_out_first_win_lift"], default=None)
    if selected is None:
        integrated_change = "none"
        selected_config: Any = None
    else:
        selected["integrated"] = True
        selected["reason"] = "selected_strongest_held_out_lift"
        integrated_change = str(selected["change_label"])
        selected_config = dict(selected["chosen_submitted_config"])
        for row in (a1, a2):
            if row is not selected and row["eligible"]:
                row["reason"] = "not_strongest_lift"
    return {
        "config_changed": selected is not None,
        "integrated_change": integrated_change,
        "selected_submitted_config": selected_config,
        "a1": a1,
        "a2": a2,
    }


def _previous_first_win(previous_gate_artifact: Mapping[str, Any]) -> float:
    return _rounded(
        _first_present_float(
            previous_gate_artifact,
            (
                "live_first_win_rate_integrated",
                "first_win_rate_integrated",
                "first_win_rate",
            ),
        )
    )


def _scored_first_win(scored_measurement: Mapping[str, Any] | None) -> float:
    if not isinstance(scored_measurement, Mapping):
        return 0.0
    return _rounded(
        _first_present_float(
            scored_measurement,
            (
                "live_first_win_rate_integrated",
                "first_win_rate_integrated",
                "first_win_rate",
            ),
        )
    )


def measure_integrated_metrics(
    *,
    previous_gate_artifact: Mapping[str, Any],
    scored_measurement: Mapping[str, Any] | None,
    config_changed: bool,
) -> JsonDict:
    pre_first = _previous_first_win(previous_gate_artifact)
    if config_changed:
        integrated_first = _scored_first_win(scored_measurement)
        note = (
            "config_changed=true; integrated held-out first-win is measured against "
            "the pre-integration gate."
        )
    else:
        integrated_first = pre_first
        note = (
            "config_changed=false; all .435 A1/A2 chosen submitted configs were "
            "unchanged or lacked held-out lift, so integrated==pre by construction."
        )
    delta = _rounded(integrated_first - pre_first)
    return {
        "live_first_win_rate_integrated": integrated_first,
        "live_first_win_rate_pre_integration": pre_first,
        "live_first_win_rate_delta_vs_pre_integration": delta,
        "no_regression_vs_pre_integration": bool(delta >= 0.0),
        "metric_measurement_note": note,
        "scored_measurement": dict(scored_measurement) if isinstance(scored_measurement, Mapping) else None,
        "pre_integration_measurement": {
            "source": PREVIOUS_GATE_RELATIVE_PATH,
            "live_first_win_rate": pre_first,
            "honest_verdict": previous_gate_artifact.get("honest_verdict", ""),
        },
    }


def _is_flat(metrics: Mapping[str, Any]) -> bool:
    return abs(_as_float(metrics.get("live_first_win_rate_delta_vs_pre_integration"))) <= 1e-12


def _null_delta_note(
    *,
    audit: Mapping[str, Any],
    metrics: Mapping[str, Any],
    positive_control_passed: bool,
) -> str:
    if not _is_flat(metrics):
        return ""
    control = "passed" if positive_control_passed else "failed"
    if audit.get("integrated_change") == "none":
        reason = "all .435 A1/A2 selected submitted configs were unchanged or had no held-out lift"
    else:
        reason = "the selected .435 integration measured equal to the pre-integration baseline"
    return (
        "Honest no-change: live_first_win_rate_integrated equals "
        "live_first_win_rate_pre_integration (delta=0.0) because "
        f"{reason}. This is the expected no-op integration equality, not a measurement bug; "
        f"positive_control_passed {control} and gates the TAUTOLOGY carve-out."
    )


def _verdict(audit: Mapping[str, Any], metrics: Mapping[str, Any], parity_green: bool) -> str:
    integrated_change = str(audit.get("integrated_change") or "none")
    if integrated_change != "none" and (
        not parity_green or metrics.get("no_regression_vs_pre_integration") is not True
    ):
        return "complete: integration_parity_or_regression_failed"
    if integrated_change == "none":
        return "complete: integration_no_change_all_levers_unchanged"
    delta = _as_float(metrics.get("live_first_win_rate_delta_vs_pre_integration"))
    return f"success: integrated_{integrated_change.lower()}_first_win_{delta:g}"


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
    no_regression = bool(metrics.get("no_regression_vs_pre_integration"))
    positive_control = bool(parity_green and no_regression)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _verdict(audit, metrics, parity_green),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "integrated_change": str(audit.get("integrated_change") or "none"),
        "live_first_win_rate_integrated": metrics.get("live_first_win_rate_integrated"),
        "live_first_win_rate_pre_integration": metrics.get(
            "live_first_win_rate_pre_integration"
        ),
        "live_first_win_rate_delta_vs_pre_integration": metrics.get(
            "live_first_win_rate_delta_vs_pre_integration"
        ),
        "parity_test_green": parity_green,
        "null_delta_methodology_note": _null_delta_note(
            audit=audit,
            metrics=metrics,
            positive_control_passed=positive_control,
        ),
        "positive_control_passed": positive_control,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "no_regression_vs_pre_integration": no_regression,
        "config_changed": bool(audit.get("config_changed")),
        "selected_submitted_config": audit.get("selected_submitted_config"),
        "a1_a2_config_audit": dict(audit),
        "metrics_delta_audit": dict(metrics),
        "parity_test": dict(parity_test),
        "submitted_agent_config": json.loads(json.dumps(submitted_agent_config, default=str)),
        "source_artifacts": dict(source_artifacts),
        "source_artifact_checksums": dict(source_artifact_checksums),
        "field_principles": FIELD_PRINCIPLES,
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
    if not blocked and artifact.get("parity_test_green") is not True:
        errors.append("parity_test_green")
    if not blocked and artifact.get("no_regression_vs_pre_integration") is not True:
        errors.append("no_regression_vs_pre_integration")
    expected_positive_control = bool(
        artifact.get("parity_test_green") is True
        and artifact.get("no_regression_vs_pre_integration") is True
    )
    if artifact.get("positive_control_passed") is not expected_positive_control:
        errors.append("positive_control_passed")
    flat = abs(_as_float(artifact.get("live_first_win_rate_delta_vs_pre_integration"))) <= 1e-12
    if flat and not str(artifact.get("null_delta_methodology_note") or "").strip():
        errors.append("null_delta_methodology_note")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_false")
    if "tautology_guard" in artifact:
        errors.append("tautology_guard_forbidden")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _default_import_checker() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, SUBMITTED_AGENT_CONFIG

    return {"submitted_agent_import": bool(E3AgentPolicy and SUBMITTED_AGENT_CONFIG)}


def _offline_arcade_checker() -> bool:  # pragma: no cover - environment precondition boundary.
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return True


def _submitted_config_snapshot() -> JsonDict:  # pragma: no cover - import boundary.
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    return json.loads(json.dumps(SUBMITTED_AGENT_CONFIG, sort_keys=True, default=str))


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    import_checker: Callable[[], Mapping[str, Any]] | None = None,
    offline_arcade_checker: Callable[[], bool] | None = None,
) -> JsonDict:  # pragma: no cover - environment precondition boundary.
    root_path = Path(root)
    try:
        import_status = dict((import_checker or _default_import_checker)())
    except Exception as exc:
        import_status = {"submitted_agent_import": False, "submitted_agent_import_error": str(exc)}
    try:
        offline_ok = bool((offline_arcade_checker or _offline_arcade_checker)())
    except Exception as exc:
        offline_ok = False
        import_status["offline_arcade_error"] = str(exc)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_importable": offline_ok,
        "previous_gate_artifact_present": (root_path / PREVIOUS_GATE_RELATIVE_PATH).exists(),
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "spec_has_req_4731": "REQ-ARC-WMTE-4731" in spec_text,
    }
    checks.update(import_status)
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade_importable",
        "submitted_agent_import",
        "previous_gate_artifact_present",
        "a1_artifact_present",
        "a2_artifact_present",
        "spec_has_req_4731",
    )
    checks["ok"] = all(bool(checks.get(key)) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next((key for key in required if not checks.get(key)), "precondition")
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


def measure_scored_lane(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - ARC runtime.
    from carnot.experiment_4693_integration_gate import measure_scored_lane as measure

    return dict(measure(root))


def _source_artifacts() -> JsonDict:
    return {
        "pre_integration": PREVIOUS_GATE_RELATIVE_PATH,
        "a1": A1_RELATIVE_PATH,
        "a2": A2_RELATIVE_PATH,
        "scored_lane": "python/carnot/experiment_4605_live_integration_scored_agent.py",
    }


def _source_checksums(*artifacts: Mapping[str, Any]) -> JsonDict:
    names = ("pre_integration", "a1", "a2")
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
        "integrated_change": "none",
        "selected_submitted_config": None,
        "a1": {"integrated": False, "reason": "blocked_precondition"},
        "a2": {"integrated": False, "reason": "blocked_precondition"},
    }
    metrics = {
        "live_first_win_rate_integrated": 0.0,
        "live_first_win_rate_pre_integration": 0.0,
        "live_first_win_rate_delta_vs_pre_integration": 0.0,
        "no_regression_vs_pre_integration": False,
        "metric_measurement_note": "blocked before measurement",
        "scored_measurement": None,
        "pre_integration_measurement": None,
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
    precondition_checker: PreconditionChecker | None = None,
    parity_check: Check | None = None,
    submitted_agent_config: Mapping[str, Any] | None = None,
    measure_scored_lane: ScoredMeasure | None = None,
    now: Callable[[], float] = time.time,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    submitted_config = (
        json.loads(json.dumps(submitted_agent_config, default=str))
        if submitted_agent_config is not None
        else _submitted_config_snapshot()
    )
    checks = dict((precondition_checker or check_preconditions)(root_path))
    duration = max(1.0, now() - start)
    if not checks.get("ok"):
        artifact = _blocked_artifact(checks, submitted_config, duration_s=duration)
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    previous_gate = _load_json(root_path / PREVIOUS_GATE_RELATIVE_PATH)
    a1_artifact = _load_json(root_path / A1_RELATIVE_PATH)
    a2_artifact = _load_json(root_path / A2_RELATIVE_PATH)
    audit = audit_config_integration(
        a1_artifact=a1_artifact,
        a2_artifact=a2_artifact,
        submitted_agent_config=submitted_config,
    )
    scored = (
        dict((measure_scored_lane or globals()["measure_scored_lane"])(root_path))
        if audit["config_changed"]
        else None
    )
    metrics = measure_integrated_metrics(
        previous_gate_artifact=previous_gate,
        scored_measurement=scored,
        config_changed=bool(audit["config_changed"]),
    )
    parity = dict((parity_check or run_parity_check)(root_path))
    artifact = build_artifact(
        preconditions_checked=checks,
        audit=audit,
        metrics=metrics,
        parity_test=parity,
        submitted_agent_config=submitted_config,
        source_artifacts=_source_artifacts(),
        source_artifact_checksums=_source_checksums(previous_gate, a1_artifact, a2_artifact),
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
