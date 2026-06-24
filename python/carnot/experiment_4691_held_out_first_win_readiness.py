"""Experiment 4691: retarget A4 readiness to held-out scored first-win.

Spec refs: REQ-CAPSTONE-4691, SCENARIO-CAPSTONE-4691,
SCENARIO-CAPSTONE-4691-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4691-FIELD-PRINCIPLES.
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

EXPERIMENT = "experiment_4691_held_out_first_win_readiness"
EXPERIMENT_ID = 4691
SCHEMA = "carnot.arc.held_out_first_win_readiness_4691.v1"
RESULT_RELATIVE_PATH = "results/experiment_4691_held_out_first_win_readiness.json"
PROXY_RESULT_RELATIVE_PATH = "results/experiment_4605_live_integration_scored_agent.json"
REPLAY_FLOOR_RESULT_RELATIVE_PATH = "results/experiment_4679_refresh_submission_package.json"
REPLAY_FLOOR_PACKAGE_FALLBACK = "results/experiment_4679_submission_package_operator_resubmit.json"
FIRST_WIN_BASELINE = 0.04
RANDOM_SEED = 4691
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- the held-out proxy runs the scored agent "
    "over cached color-permuted variants offline, no live game server (1s floor)."
)

SPEC_REFS = [
    "REQ-CAPSTONE-4691",
    "SCENARIO-CAPSTONE-4691",
    "SCENARIO-CAPSTONE-4691-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4691-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: held_out_first_win_improved_ci_excludes_baseline OR "
            "complete: held_out_first_win_flat_no_leaderboard_change."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- the held-out proxy runs the scored "
            "agent over cached color-permuted variants offline, no live game server (1s floor)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the held-out proxy measures the scored agent's first-win, "
            "oracle-distinct from the executable win-check."
        )
    },
    "first_win_rate_integrated": {
        "principle": (
            "the held-out generic first-win on color-permuted variants is the only offline "
            "proxy that tracks the scored leaderboard lane; the replay count does not."
        )
    },
    "first_win_ci_lower": {
        "principle": (
            "bootstrap-CI lower bound > 0 is the falsifiable improvement criterion; a point "
            "estimate alone is gameable by a single lucky variant."
        )
    },
    "first_win_baseline": {
        "principle": (
            "the last-submission baseline first-win (0.04) -- the apples-to-apples comparison "
            "the readiness gate uses."
        )
    },
    "multi_level_deepen_rate_integrated": {
        "principle": (
            "deepening past L1 is the second scored lever (the L2 goal-predicate wall); "
            "tracking it held-out keeps A4 honest about depth without using the replay count."
        )
    },
    "parity_test_green": {
        "principle": (
            "the held-out proxy is only valid if the measured agent is byte-for-byte the "
            "SUBMITTED_AGENT_CONFIG; a parity miss invalidates any readiness claim."
        )
    },
    "replay_package_floor_reproduced": {
        "principle": (
            "the replay package stays a reproduced FLOOR artifact, but its level count is "
            "explicitly NOT the leaderboard score (the retargeted framing)."
        )
    },
    "replay_count_is_not_the_score": {
        "principle": (
            "MUST be true -- explicitly records that live_submittable_level_count is NOT the "
            "leaderboard score (the retired dead-end framing the 2026-06-24 directive strips)."
        )
    },
    "ready_for_operator_submit": {
        "principle": (
            "True if parity green AND the held-out first-win readiness gate passes; the task "
            "NEVER submits (operator-only)."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (SUBMITTED_AGENT_CONFIG importable, experiment_4605 "
            "importable); pre-empts missing-resource fabrication."
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
    "first_win_delta_vs_baseline",
    "leaderboard_relevant_change_note",
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


def _first_win_readiness(
    *,
    parity_green: bool,
    first_win_rate: float,
    first_win_baseline: float,
    first_win_ci_lower: float,
) -> bool:
    return bool(
        parity_green
        and first_win_rate > first_win_baseline
        and first_win_ci_lower > 0.0
    )


def _honest_verdict(
    *,
    readiness: bool,
    parity_green: bool,
    first_win_rate: float,
    first_win_baseline: float,
    first_win_ci_lower: float,
) -> str:
    if readiness:
        return "success: held_out_first_win_improved_ci_excludes_baseline"
    if not parity_green and first_win_rate > first_win_baseline and first_win_ci_lower > 0.0:
        return "complete: held_out_first_win_point_up_ci_supported_but_parity_failed"
    if abs(first_win_rate - first_win_baseline) <= 1e-12:
        return "complete: held_out_first_win_flat_no_leaderboard_change"
    if first_win_rate > first_win_baseline:
        return "complete: held_out_first_win_point_up_ci_overlaps_baseline_no_leaderboard_change"
    return "complete: held_out_first_win_below_baseline_no_leaderboard_change"


def _change_note(readiness: bool) -> str:
    if readiness:
        return "held-out first-win improved with bootstrap-CI lower bound above zero."
    return (
        "no leaderboard-relevant change this milestone: held-out first-win did not improve "
        "with bootstrap-CI lower bound above zero."
    )


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    proxy_artifact: Mapping[str, Any],
    replay_floor: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    first_win_rate = _extract_first_win_rate(proxy_artifact)
    first_win_ci_lower = _extract_ci_lower(proxy_artifact)
    first_win_delta = round(first_win_rate - FIRST_WIN_BASELINE, 6)
    multi_level_deepen_rate = _extract_multi_level_deepen_rate(proxy_artifact)
    parity_green = bool(parity_test.get("passed"))
    readiness = _first_win_readiness(
        parity_green=parity_green,
        first_win_rate=first_win_rate,
        first_win_baseline=FIRST_WIN_BASELINE,
        first_win_ci_lower=first_win_ci_lower,
    )
    floor_path = str(
        replay_floor.get("package_path")
        or replay_floor.get("refreshed_package_path")
        or REPLAY_FLOOR_PACKAGE_FALLBACK
    )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(
            readiness=readiness,
            parity_green=parity_green,
            first_win_rate=first_win_rate,
            first_win_baseline=FIRST_WIN_BASELINE,
            first_win_ci_lower=first_win_ci_lower,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "first_win_rate_integrated": first_win_rate,
        "first_win_ci_lower": first_win_ci_lower,
        "first_win_baseline": FIRST_WIN_BASELINE,
        "first_win_delta_vs_baseline": first_win_delta,
        "multi_level_deepen_rate_integrated": multi_level_deepen_rate,
        "parity_test_green": parity_green,
        "replay_package_floor_reproduced": bool(
            replay_floor.get("replay_package_floor_reproduced")
        ),
        "replay_count_is_not_the_score": True,
        "ready_for_operator_submit": readiness,
        "held_out_first_win_readiness": readiness,
        "leaderboard_relevant_change_note": _change_note(readiness),
        "proxy_artifact_path": PROXY_RESULT_RELATIVE_PATH,
        "replay_floor_package_path": floor_path,
        "replay_floor": dict(replay_floor),
        "parity_test": dict(parity_test),
        "held_out_proxy_summary": {
            "source_artifact_path": PROXY_RESULT_RELATIVE_PATH,
            "first_win_rate_integrated": first_win_rate,
            "first_win_ci_lower": first_win_ci_lower,
            "first_win_baseline": FIRST_WIN_BASELINE,
            "first_win_delta_vs_baseline": first_win_delta,
            "multi_level_deepen_rate_integrated": multi_level_deepen_rate,
            "proxy_honest_verdict": proxy_artifact.get("honest_verdict", ""),
        },
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
    if artifact.get("ready_for_operator_submit") is True and artifact.get(
        "held_out_first_win_readiness"
    ) is not True:
        errors.append("ready_for_operator_submit_requires_held_out_first_win_readiness")
    expected_readiness = _first_win_readiness(
        parity_green=artifact.get("parity_test_green") is True,
        first_win_rate=_float(artifact.get("first_win_rate_integrated")),
        first_win_baseline=_float(artifact.get("first_win_baseline"), FIRST_WIN_BASELINE),
        first_win_ci_lower=_float(artifact.get("first_win_ci_lower")),
    )
    if artifact.get("held_out_first_win_readiness") is not expected_readiness:
        errors.append("held_out_first_win_readiness_gate")
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


def check_preconditions(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    checks: JsonDict = {
        "submitted_agent_config_importable": False,
        "experiment_4605_importable": False,
    }
    commands = {
        "submitted_agent_config_importable": [
            sys.executable,
            "-c",
            (
                "from carnot.agentic.arc_competition_agent import "
                "E3AgentPolicy, SUBMITTED_AGENT_CONFIG"
            ),
        ],
        "experiment_4605_importable": [
            sys.executable,
            "-c",
            "from carnot import experiment_4605_live_integration_scored_agent",
        ],
    }
    for key, command in commands.items():
        proc = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        checks[key] = proc.returncode == 0
        checks[f"{key}_command"] = " ".join(command)
        checks[f"{key}_returncode"] = int(proc.returncode)
        if proc.returncode != 0:
            checks["ok"] = False
            checks["blocked_resource"] = key.replace("_importable", "_import")
            checks["stderr_tail"] = proc.stderr[-1000:]
            return checks
    checks["ok"] = True
    return checks


def run_parity_test(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    from carnot import experiment_4605_live_integration_scored_agent as exp4605

    return dict(exp4605.run_parity_check(root))


def run_held_out_proxy(root: Path, parity_test: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    from carnot import experiment_4605_live_integration_scored_agent as exp4605

    previous_deepen = os.environ.get(exp4605.DEEPEN_ENV)
    os.environ[exp4605.DEEPEN_ENV] = "1"
    try:
        return dict(
            exp4605.run(
                root=root,
                parity_check=lambda _root: parity_test,
            )
        )
    finally:
        if previous_deepen is None:
            os.environ.pop(exp4605.DEEPEN_ENV, None)
        else:
            os.environ[exp4605.DEEPEN_ENV] = previous_deepen


def load_replay_package_floor(root: Path) -> JsonDict:  # pragma: no cover - filesystem boundary.
    result_path = root / REPLAY_FLOOR_RESULT_RELATIVE_PATH
    floor: JsonDict = {
        "source_result_path": REPLAY_FLOOR_RESULT_RELATIVE_PATH,
        "package_path": REPLAY_FLOOR_PACKAGE_FALLBACK,
        "source_result_exists": result_path.exists(),
        "package_exists": False,
        "replay_package_floor_reproduced": False,
        "live_submittable_level_count": 0,
        "offline_reproduced": False,
        "ready_for_operator_submit": False,
        "note": "replay package floor only; not the scored leaderboard lane.",
    }
    if not result_path.exists():
        return floor
    try:
        source = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        floor["error"] = f"json_decode_error: {exc}"
        return floor
    package_path = str(source.get("refreshed_package_path") or REPLAY_FLOOR_PACKAGE_FALLBACK)
    package_exists = (root / package_path).exists()
    floor.update(
        {
            "package_path": package_path,
            "package_exists": package_exists,
            "live_submittable_level_count": int(source.get("live_submittable_level_count") or 0),
            "offline_reproduced": source.get("offline_reproduced") is True,
            "ready_for_operator_submit": source.get("ready_for_operator_submit") is True,
            "source_honest_verdict": source.get("honest_verdict", ""),
        }
    )
    floor["replay_package_floor_reproduced"] = bool(
        floor["package_exists"] and floor["offline_reproduced"]
    )
    return floor


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
) -> JsonDict:
    artifact = build_artifact(
        preconditions_checked=dict(preconditions_checked),
        parity_test={"passed": False, "blocked_reason": reason},
        proxy_artifact={},
        replay_floor={},
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
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    root_path = Path(root)
    started = now()
    checks = dict(preconditions_checker(root_path))
    duration = lambda: _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn)
    if not checks.get("ok", False):
        reason = str(checks.get("blocked_resource") or "precondition")
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason=reason,
            duration_s=duration(),
        )
        write_artifact(root_path, artifact)
        return artifact

    parity = dict(parity_check(root_path))
    if parity.get("passed") is True:
        try:
            proxy = dict(proxy_runner(root_path, parity))
        except Exception as exc:  # pragma: no cover - defensive live-run boundary.
            checks["proxy_error"] = repr(exc)[:500]
            checks["blocked_resource"] = "experiment_4605_proxy"
            artifact = _blocked_artifact(
                preconditions_checked=checks,
                reason="experiment_4605_proxy",
                duration_s=duration(),
            )
            write_artifact(root_path, artifact)
            return artifact
    else:
        proxy = {}

    replay_floor = dict(replay_floor_loader(root_path))
    artifact = build_artifact(
        preconditions_checked=checks,
        parity_test=parity,
        proxy_artifact=proxy,
        replay_floor=replay_floor,
        duration_s=duration(),
    )
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(f"first_win_rate_integrated={artifact['first_win_rate_integrated']}")
    print(f"first_win_ci_lower={artifact['first_win_ci_lower']}")
    print(f"ready_for_operator_submit={artifact['ready_for_operator_submit']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
