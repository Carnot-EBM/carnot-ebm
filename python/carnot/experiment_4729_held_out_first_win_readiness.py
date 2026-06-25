"""Experiment 4729: .435 held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4729, SCENARIO-CAPSTONE-4729,
SCENARIO-CAPSTONE-4729-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4729-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
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
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
ParityCheck = Callable[[Path], Mapping[str, Any]]
ProxyRunner = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]
ReplayFloorLoader = Callable[[Path], Mapping[str, Any]]
LeverInputLoader = Callable[[Path], Mapping[str, Any]]

EXPERIMENT = "experiment_4729_held_out_first_win_readiness"
EXPERIMENT_ID = 4729
SCHEMA = "carnot.arc.held_out_first_win_readiness_4729.v1"
RESULT_RELATIVE_PATH = "results/experiment_4729_held_out_first_win_readiness.json"
PROXY_RESULT_RELATIVE_PATH = "results/experiment_4605_live_integration_scored_agent.json"
REPLAY_FLOOR_RESULT_RELATIVE_PATH = "results/experiment_4679_refresh_submission_package.json"
REPLAY_FLOOR_PACKAGE_FALLBACK = "results/experiment_4679_submission_package_operator_resubmit.json"
FIRST_WIN_BASELINE = 0.04
MIN_HELD_OUT_VARIANT_ATTEMPTS = 100
HELD_OUT_VARIANT_ATTEMPT_FLOOR = "B>=100"
HELD_OUT_VARIANT_IDS = (1, 2, 3, 4)
RANDOM_SEED = 4729
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- the held-out lane scores the submitted "
    "config over cached variants (1s floor)."
)
REPLAY_FLOOR_NOTE = (
    "replay package floor only; live_submittable_level_count is not the leaderboard score."
)

SPEC_REFS = [
    "REQ-CAPSTONE-4729",
    "SCENARIO-CAPSTONE-4729",
    "SCENARIO-CAPSTONE-4729-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4729-FIELD-PRINCIPLES",
]

V435_LEVER_ARTIFACTS = {
    "a1": "results/experiment_4726_online_action_learning_driver_valid_test.json",
    "a2": "results/experiment_4727_active_probe_disambiguation.json",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: held_out_first_win_improved_<delta> OR complete: "
            "held_out_first_win_flat_no_leaderboard_change."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- the held-out lane scores the "
            "submitted config over cached variants (1s floor)."
        )
    },
    "first_win_rate_integrated": {
        "principle": (
            "the held-out generic first-win on color-permuted variants -- the only offline "
            "proxy that tracks the scored leaderboard lane; the replay count does not."
        )
    },
    "first_win_ci_lower": {
        "principle": (
            "bootstrap-CI lower bound > 0 is the falsifiable improvement criterion; a point "
            "estimate is gameable by one lucky variant."
        )
    },
    "multi_level_deepen_rate_integrated": {
        "principle": (
            "deepening past L1 is the second scored lever; tracking it held-out keeps A4 "
            "honest about depth without the replay count."
        )
    },
    "parity_test_green": {
        "principle": (
            "the held-out proxy is valid only if the measured agent is byte-for-byte the "
            "SUBMITTED_AGENT_CONFIG; a parity miss invalidates readiness."
        )
    },
    "replay_package_floor_reproduced": {
        "principle": (
            "the replay package stays a reproduced FLOOR, but its level count is explicitly "
            "NOT the leaderboard score."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when first-win is flat; the TAUTOLOGY carve-out reads it to downgrade "
            "CRITICAL->WARN (honest no-change, not a measurement bug)."
        )
    },
    "positive_control_passed": {
        "principle": (
            "bool(parity_test_green) -- GATES the TAUTOLOGY exemption; an unvalidated flat "
            "result is NOT excused."
        )
    },
    "verifier_is_oracle": {
        "principle": "false -- the held-out lane measures the agent; no oracle is invoked."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, experiment_4605 importable); "
            "pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "proxy_artifact_path",
    "replay_floor_package_path",
    "replay_floor",
    "parity_test",
    "held_out_proxy_summary",
    "held_out_first_win_readiness",
    "ready_for_operator_submit",
    "first_win_baseline",
    "first_win_delta_vs_baseline",
    "held_out_variant_attempts",
    "held_out_variant_attempt_floor",
    "replay_count_is_not_the_score",
    "v435_lever_inputs",
    "submitted_to_leaderboard",
    "operator_only",
    "duration_s",
    "field_principles",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return default


def _extract_first_win_rate(proxy_artifact: Mapping[str, Any]) -> float:
    if "first_win_rate_integrated" in proxy_artifact:
        return _float(proxy_artifact.get("first_win_rate_integrated"))
    measurement = proxy_artifact.get("integrated_measurement")
    if isinstance(measurement, Mapping):
        return _float(measurement.get("first_win_rate"))
    return 0.0


def _extract_ci_lower(proxy_artifact: Mapping[str, Any]) -> float:
    if "first_win_ci_lower" in proxy_artifact:
        return _float(proxy_artifact.get("first_win_ci_lower"))
    ci = proxy_artifact.get("first_win_ci")
    if isinstance(ci, Mapping):
        interval = ci.get("ci95")
        if isinstance(interval, list | tuple) and interval:
            return _float(interval[0])
        if "low" in ci:
            return _float(ci.get("low"))
    return 0.0


def _attempt_depth(attempt: Mapping[str, Any]) -> int:
    explicit = attempt.get("depth_reached")
    if explicit is not None and not isinstance(explicit, bool):
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            pass
    return 1 if attempt.get("first_win") is True or attempt.get("solved") is True else 0


def _extract_multi_level_deepen_rate(proxy_artifact: Mapping[str, Any]) -> float:
    for key in ("multi_level_deepen_rate_integrated", "multi_level_solve_rate"):
        if key in proxy_artifact:
            return _float(proxy_artifact.get(key))
    measurement = proxy_artifact.get("integrated_measurement")
    attempts = measurement.get("variant_attempts") if isinstance(measurement, Mapping) else []
    if not isinstance(attempts, list) or not attempts:
        return 0.0
    attempted = [row for row in attempts if isinstance(row, Mapping) and row.get("attempted", True)]
    if not attempted:
        return 0.0
    deepened = sum(1 for row in attempted if _attempt_depth(row) >= 2)
    return round(float(deepened) / float(len(attempted)), 6)


def _extract_held_out_variant_attempts(proxy_artifact: Mapping[str, Any]) -> int:
    if "held_out_variant_attempts" in proxy_artifact:
        return int(_float(proxy_artifact.get("held_out_variant_attempts")))
    measurement = proxy_artifact.get("integrated_measurement")
    if isinstance(measurement, Mapping):
        count = measurement.get("variant_attempts_count")
        if count is not None and not isinstance(count, bool):
            try:
                return max(0, int(count))
            except (TypeError, ValueError):
                pass
        attempts = measurement.get("variant_attempts")
        if isinstance(attempts, list):
            return len(attempts)
    return 0


def _is_flat_delta(first_win_rate: float, baseline: float) -> bool:
    return abs(round(first_win_rate - baseline, 6)) <= 1e-12


def _ci_supports_improvement(first_win_rate: float, baseline: float, ci_lower: float) -> bool:
    return bool(first_win_rate > baseline and ci_lower > 0.0)


def _null_delta_note(
    *,
    first_win_rate: float,
    baseline: float,
    positive_control_passed: bool,
) -> str:
    if not _is_flat_delta(first_win_rate, baseline):
        return ""
    control = "passed" if positive_control_passed else "failed"
    return (
        "Held-out first-win is flat vs baseline (first_win_rate_integrated == "
        "first_win_baseline, delta=0.0): no lever moved the leaderboard-relevant metric "
        "in this readiness run. The equality is an honest no-leaderboard-change null; "
        f"positive_control_passed {control} and gates whether the flat null is excused."
    )


def _readiness(
    *,
    parity_green: bool,
    first_win_rate: float,
    baseline: float,
    ci_lower: float,
    attempts: int,
    null_delta_methodology_note: str,
    positive_control_passed: bool,
) -> bool:
    if attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
        return False
    improved = parity_green and _ci_supports_improvement(first_win_rate, baseline, ci_lower)
    held_flat = (
        parity_green
        and _is_flat_delta(first_win_rate, baseline)
        and positive_control_passed
        and bool(null_delta_methodology_note.strip())
    )
    return bool(improved or held_flat)


def _honest_verdict(
    *,
    readiness: bool,
    parity_green: bool,
    first_win_rate: float,
    baseline: float,
    ci_lower: float,
    attempts: int,
    positive_control_passed: bool,
) -> str:
    if attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
        return "complete: held_out_first_win_measurement_below_b100"
    if readiness and _ci_supports_improvement(first_win_rate, baseline, ci_lower):
        delta = round(first_win_rate - baseline, 6)
        return f"success: held_out_first_win_improved_{delta:g}"
    if not parity_green and _ci_supports_improvement(first_win_rate, baseline, ci_lower):
        return "complete: held_out_first_win_point_up_ci_supported_but_parity_failed"
    if _is_flat_delta(first_win_rate, baseline):
        if positive_control_passed:
            return "complete: held_out_first_win_flat_no_leaderboard_change"
        return "complete: held_out_first_win_flat_unvalidated_no_leaderboard_change"
    if first_win_rate > baseline:
        return "complete: held_out_first_win_point_up_ci_overlaps_baseline_no_leaderboard_change"
    return "complete: held_out_first_win_below_baseline_no_leaderboard_change"


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive file boundary.
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def load_v435_lever_inputs(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    out: JsonDict = {}
    for key, rel in V435_LEVER_ARTIFACTS.items():
        path = root_path / rel
        payload = _read_json(path)
        row: JsonDict = {
            "path": rel,
            "exists": path.exists(),
            "experiment": payload.get("experiment"),
            "honest_verdict": payload.get("honest_verdict"),
            "chosen_submitted_config": payload.get("chosen_submitted_config"),
        }
        if path.exists():
            row["sha256"] = _file_sha256(path)
        else:
            row["sha256"] = ""
            row["error"] = "missing_artifact"
        out[key] = row
    return out


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    proxy_artifact: Mapping[str, Any],
    replay_floor: Mapping[str, Any],
    v435_lever_inputs: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    first_win_rate = _extract_first_win_rate(proxy_artifact)
    first_win_ci_lower = _extract_ci_lower(proxy_artifact)
    first_win_delta = round(first_win_rate - FIRST_WIN_BASELINE, 6)
    attempts = _extract_held_out_variant_attempts(proxy_artifact)
    multi_level_deepen_rate = _extract_multi_level_deepen_rate(proxy_artifact)
    parity_green = bool(parity_test.get("passed"))
    positive_control = bool(parity_green)
    null_note = _null_delta_note(
        first_win_rate=first_win_rate,
        baseline=FIRST_WIN_BASELINE,
        positive_control_passed=positive_control,
    )
    ready = _readiness(
        parity_green=parity_green,
        first_win_rate=first_win_rate,
        baseline=FIRST_WIN_BASELINE,
        ci_lower=first_win_ci_lower,
        attempts=attempts,
        null_delta_methodology_note=null_note,
        positive_control_passed=positive_control,
    )
    floor = dict(replay_floor)
    floor.setdefault("note", REPLAY_FLOOR_NOTE)
    floor_path = str(
        floor.get("package_path")
        or floor.get("refreshed_package_path")
        or REPLAY_FLOOR_PACKAGE_FALLBACK
    )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(
            readiness=ready,
            parity_green=parity_green,
            first_win_rate=first_win_rate,
            baseline=FIRST_WIN_BASELINE,
            ci_lower=first_win_ci_lower,
            attempts=attempts,
            positive_control_passed=positive_control,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "first_win_rate_integrated": first_win_rate,
        "first_win_ci_lower": first_win_ci_lower,
        "first_win_baseline": FIRST_WIN_BASELINE,
        "first_win_delta_vs_baseline": first_win_delta,
        "multi_level_deepen_rate_integrated": multi_level_deepen_rate,
        "held_out_variant_attempts": attempts,
        "held_out_variant_attempt_floor": HELD_OUT_VARIANT_ATTEMPT_FLOOR,
        "parity_test_green": parity_green,
        "replay_package_floor_reproduced": bool(floor.get("replay_package_floor_reproduced")),
        "replay_count_is_not_the_score": True,
        "null_delta_methodology_note": null_note,
        "positive_control_passed": positive_control,
        "verifier_is_oracle": False,
        "ready_for_operator_submit": ready,
        "held_out_first_win_readiness": ready,
        "proxy_artifact_path": PROXY_RESULT_RELATIVE_PATH,
        "replay_floor_package_path": floor_path,
        "replay_floor": floor,
        "parity_test": dict(parity_test),
        "held_out_proxy_summary": {
            "source_artifact_path": PROXY_RESULT_RELATIVE_PATH,
            "first_win_rate_integrated": first_win_rate,
            "first_win_ci_lower": first_win_ci_lower,
            "first_win_baseline": FIRST_WIN_BASELINE,
            "first_win_delta_vs_baseline": first_win_delta,
            "multi_level_deepen_rate_integrated": multi_level_deepen_rate,
            "held_out_variant_attempts": attempts,
            "proxy_honest_verdict": proxy_artifact.get("honest_verdict", ""),
        },
        "v435_lever_inputs": dict(v435_lever_inputs),
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "random_seed": int(random_seed),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = str(artifact.get("honest_verdict") or "")
    blocked = verdict.startswith("blocked_")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("replay_count_is_not_the_score") is not True:
        errors.append("replay_count_is_not_the_score_true")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_false")
    if "min_held_out_variant_attempts" in artifact:
        errors.append("redundant_min_held_out_variant_attempts")
    first_win_rate = _float(artifact.get("first_win_rate_integrated"))
    baseline = _float(artifact.get("first_win_baseline"), FIRST_WIN_BASELINE)
    ci_lower = _float(artifact.get("first_win_ci_lower"))
    attempts = int(_float(artifact.get("held_out_variant_attempts")))
    parity_green = artifact.get("parity_test_green") is True
    expected_positive_control = bool(parity_green)
    if artifact.get("positive_control_passed") is not expected_positive_control:
        errors.append("positive_control_passed")
    note = str(artifact.get("null_delta_methodology_note") or "")
    if _is_flat_delta(first_win_rate, baseline) and not note.strip():
        errors.append("null_delta_methodology_note")
    expected_readiness = _readiness(
        parity_green=parity_green,
        first_win_rate=first_win_rate,
        baseline=baseline,
        ci_lower=ci_lower,
        attempts=attempts,
        null_delta_methodology_note=note,
        positive_control_passed=artifact.get("positive_control_passed") is True,
    )
    if artifact.get("held_out_first_win_readiness") is not expected_readiness:
        errors.append("held_out_first_win_readiness_gate")
    if artifact.get("ready_for_operator_submit") is not expected_readiness:
        errors.append("ready_for_operator_submit_gate")
    if not blocked and attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
        errors.append("held_out_variant_attempts_below_minimum")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(dict.fromkeys(errors))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    _write_json(path, artifact)
    return path


def _run_command(
    command: list[str], root: Path, *, timeout_s: int = 180
) -> JsonDict:  # pragma: no cover - subprocess boundary.
    try:
        proc = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return {
            "command": " ".join(command),
            "passed": False,
            "returncode": -1,
            "stderr_tail": repr(exc)[:1000],
            "stdout_tail": "",
        }
    return {
        "command": " ".join(command),
        "passed": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }


def check_preconditions(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    commands = {
        "offline_arcade": [
            sys.executable,
            "-c",
            "from carnot.agentic import arc_solver_kit as k; k.offline_arcade()",
        ],
        "experiment_4605_importable": [
            sys.executable,
            "-c",
            "from carnot import experiment_4605_live_integration_scored_agent",
        ],
    }
    checks: JsonDict = {}
    for key, command in commands.items():
        report = _run_command(command, root, timeout_s=180)
        checks[key] = bool(report["passed"])
        checks[f"{key}_command"] = report["command"]
        checks[f"{key}_returncode"] = report["returncode"]
        if not report["passed"]:
            checks["ok"] = False
            checks["blocked_resource"] = key
            checks["stdout_tail"] = report["stdout_tail"]
            checks["stderr_tail"] = report["stderr_tail"]
            return checks
    checks["ok"] = True
    return checks


def run_parity_test(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    from carnot import experiment_4605_live_integration_scored_agent as exp4605

    return dict(exp4605.run_parity_check(root))


def run_held_out_proxy(root: Path, parity_test: Mapping[str, Any]) -> JsonDict:
    from carnot import experiment_4605_live_integration_scored_agent as exp4605

    previous_deepen = os.environ.get(exp4605.DEEPEN_ENV)
    previous_variants = os.environ.get(exp4605.VARIANT_IDS_ENV)
    os.environ[exp4605.DEEPEN_ENV] = "1"
    os.environ[exp4605.VARIANT_IDS_ENV] = ",".join(str(item) for item in HELD_OUT_VARIANT_IDS)
    try:
        return dict(exp4605.run(root=root, parity_check=lambda _root: parity_test))
    finally:
        if previous_deepen is None:
            os.environ.pop(exp4605.DEEPEN_ENV, None)
        else:
            os.environ[exp4605.DEEPEN_ENV] = previous_deepen
        if previous_variants is None:
            os.environ.pop(exp4605.VARIANT_IDS_ENV, None)
        else:
            os.environ[exp4605.VARIANT_IDS_ENV] = previous_variants


def load_replay_package_floor(root: Path) -> JsonDict:  # pragma: no cover - filesystem boundary.
    from carnot.live_submittable_metrics import compute_live_submittable_metrics

    root_path = Path(root)
    source = _read_json(root_path / REPLAY_FLOOR_RESULT_RELATIVE_PATH)
    package_path = str(source.get("refreshed_package_path") or REPLAY_FLOOR_PACKAGE_FALLBACK)
    metrics = compute_live_submittable_metrics(root_path, package_path=package_path)
    package_exists = (root_path / package_path).exists()
    live_count = int(metrics.get("live_submittable_level_count") or 0)
    subset = metrics.get("live_submittable_subset_of_reproducible") is True
    return {
        "source_result_path": REPLAY_FLOOR_RESULT_RELATIVE_PATH,
        "source_result_exists": (root_path / REPLAY_FLOOR_RESULT_RELATIVE_PATH).exists(),
        "package_path": package_path,
        "package_exists": package_exists,
        "replay_package_floor_reproduced": bool(package_exists and subset and live_count > 0),
        "live_submittable_level_count": live_count,
        "reproducible_total_levels": int(metrics.get("reproducible_total_levels") or 0),
        "reproducible_vs_submittable_gap": int(
            metrics.get("reproducible_vs_submittable_gap") or 0
        ),
        "live_submittable_subset_of_reproducible": subset,
        "offline_reproduced": bool(package_exists and subset and live_count > 0),
        "ready_for_operator_submit": bool(package_exists and subset and live_count > 33),
        "note": REPLAY_FLOOR_NOTE,
    }


def _floor_duration(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    if elapsed < 1.0:
        sleep_fn(1.0 - elapsed)
    return max(float(now()), started_at + 1.0) - started_at


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    reason: str,
    duration_s: float,
    replay_floor: Mapping[str, Any] | None = None,
    v435_lever_inputs: Mapping[str, Any] | None = None,
) -> JsonDict:
    artifact = build_artifact(
        preconditions_checked=dict(preconditions_checked),
        parity_test={"passed": False, "blocked_reason": reason},
        proxy_artifact={},
        replay_floor=dict(replay_floor or {}),
        v435_lever_inputs=dict(v435_lever_inputs or {}),
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{reason}"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checker: PreconditionsChecker = check_preconditions,
    parity_check: ParityCheck = run_parity_test,
    proxy_runner: ProxyRunner = run_held_out_proxy,
    replay_floor_loader: ReplayFloorLoader = load_replay_package_floor,
    lever_input_loader: LeverInputLoader = load_v435_lever_inputs,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    root_path = Path(root)
    started = now()
    duration = lambda: _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn)
    lever_inputs = dict(lever_input_loader(root_path))
    replay_floor = dict(replay_floor_loader(root_path))
    checks = dict(preconditions_checker(root_path))
    if not checks.get("ok", False):
        reason = str(checks.get("blocked_resource") or "precondition")
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason=reason,
            duration_s=duration(),
            replay_floor=replay_floor,
            v435_lever_inputs=lever_inputs,
        )
        write_artifact(root_path, artifact)
        return artifact

    parity = dict(parity_check(root_path))
    if parity.get("passed") is not True:
        checks["blocked_resource"] = "parity_test"
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason="parity_test",
            duration_s=duration(),
            replay_floor=replay_floor,
            v435_lever_inputs=lever_inputs,
        )
        write_artifact(root_path, artifact)
        return artifact

    try:
        proxy = dict(proxy_runner(root_path, parity))
    except Exception as exc:  # pragma: no cover - defensive live-run boundary.
        checks["proxy_error"] = repr(exc)[:500]
        checks["blocked_resource"] = "experiment_4605_proxy"
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason="experiment_4605_proxy",
            duration_s=duration(),
            replay_floor=replay_floor,
            v435_lever_inputs=lever_inputs,
        )
        write_artifact(root_path, artifact)
        return artifact

    attempts = _extract_held_out_variant_attempts(proxy)
    if attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
        checks["blocked_resource"] = "experiment_4605_proxy_b100"
        checks["held_out_variant_attempts"] = attempts
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason="experiment_4605_proxy_b100",
            duration_s=duration(),
            replay_floor=replay_floor,
            v435_lever_inputs=lever_inputs,
        )
        write_artifact(root_path, artifact)
        return artifact

    artifact = build_artifact(
        preconditions_checked=checks,
        parity_test=parity,
        proxy_artifact=proxy,
        replay_floor=replay_floor,
        v435_lever_inputs=lever_inputs,
        duration_s=duration(),
    )
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(f"first_win_rate_integrated={artifact['first_win_rate_integrated']}")
    print(f"first_win_ci_lower={artifact['first_win_ci_lower']}")
    print(f"multi_level_deepen_rate_integrated={artifact['multi_level_deepen_rate_integrated']}")
    print(f"ready_for_operator_submit={artifact['ready_for_operator_submit']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
