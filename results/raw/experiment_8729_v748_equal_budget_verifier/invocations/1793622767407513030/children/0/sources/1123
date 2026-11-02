"""Experiment 4729: .435 held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4729, SCENARIO-CAPSTONE-4729,
SCENARIO-CAPSTONE-4729-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4729-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
# Sibling checkpoint file. The held-out proxy runs the offline ARC arcade over every public game x
# {color variant 1..4} x {integrated, bare} -- minutes per game, and the whole sweep exceeds codex's
# 4800s hard wall-clock cap, so codex KILLS the run before it can write the final artifact and the
# task FAILs every milestone producing NOTHING. We persist per-game progress to this partial file so a
# capped/killed run leaves usable work on disk, and the NEXT run resumes from it instead of restarting
# the whole sweep. The partial file holds the already-computed integrated/bare attempt rows keyed by
# game; it is the resume ledger, NOT the scored artifact.
PARTIAL_RESULT_RELATIVE_PATH = "results/experiment_4729_held_out_first_win_readiness.partial.json"
PROXY_RESULT_RELATIVE_PATH = "results/experiment_4605_live_integration_scored_agent.json"
REPLAY_FLOOR_RESULT_RELATIVE_PATH = "results/experiment_4679_refresh_submission_package.json"
REPLAY_FLOOR_PACKAGE_FALLBACK = "results/experiment_4679_submission_package_operator_resubmit.json"
FIRST_WIN_BASELINE = 0.04
MIN_HELD_OUT_VARIANT_ATTEMPTS = 100
HELD_OUT_VARIANT_ATTEMPT_FLOOR = "B>=100"
HELD_OUT_VARIANT_IDS = (1, 2, 3, 4)
RANDOM_SEED = 4729
# Soft wall-clock budget (seconds). Checked BETWEEN per-game units. When exceeded the run stops
# GRACEFULLY -- it flushes the partial file and emits a clean partial: true artifact (exit 0), keeping
# the total wall-clock well under codex's 4800s HARD cap so codex never kills the process mid-write.
# Default 3500s (lowered from 4200s on 2026-06-27): the .449 A4 first attempt still died at the 4800s
# hard cap because 4200s left only ~600s for the agent's artifact-write+commit; 3500s leaves ~1300s of
# margin so a clean partial is rescued even on a slow run. (With the conductor's GPU-0 CUDA generator
# per the 2026-06-27 allocation the full harness now runs in ~1137s, so this budget rarely binds; it is
# the slow-path safety net.) The operator can override via EXP4729_SOFT_BUDGET_S (e.g. raise it on an
# uncapped direct invocation, or lower it for a faster partial). The PARTIAL artifact accumulates across
# resumed runs until COMPLETE.
SOFT_BUDGET_ENV = "EXP4729_SOFT_BUDGET_S"
DEFAULT_SOFT_BUDGET_S = 3500.0
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
    # A blocked OR a checkpoint-partial run is a legitimately-incomplete state: the B>=100 floor
    # cannot be met yet (a partial soft-budget stop ran only some games), so the below-minimum rule is
    # exempt for it -- the partial: true flag + completed/remaining_variants record the incompleteness.
    partial = artifact.get("partial") is True
    if not blocked and not partial and attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
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


class _BudgetExceeded(Exception):
    """Raised by the checkpointed proxy when the soft wall-clock budget is hit mid-sweep.

    It is NOT an error -- it is the graceful-stop signal. The run() handler catches it, treats the
    accumulated (partial) attempts as the run's output, and writes a partial: true artifact + exit 0.
    Carrying the per-game progress on the exception keeps the resume ledger flush-then-stop atomic.
    """

    def __init__(self, *, done_games: Sequence[str], remaining_games: Sequence[str]) -> None:
        self.done_games = list(done_games)
        self.remaining_games = list(remaining_games)
        super().__init__(
            f"soft budget exceeded after {len(self.done_games)} games; "
            f"{len(self.remaining_games)} remaining"
        )


def resolve_soft_budget_s(env: Mapping[str, str] | None = None) -> float:
    """Soft wall-clock budget in seconds: EXP4729_SOFT_BUDGET_S overrides DEFAULT_SOFT_BUDGET_S.

    A non-positive or unparseable override falls back to the default so a typo can never DISABLE the
    graceful stop (which would re-expose the run to codex's hard kill). The budget is the only knob;
    WHAT is measured is unchanged."""

    source = os.environ if env is None else env
    raw = str(source.get(SOFT_BUDGET_ENV, "")).strip()
    if not raw:
        return DEFAULT_SOFT_BUDGET_S
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_SOFT_BUDGET_S
    return parsed if parsed > 0.0 else DEFAULT_SOFT_BUDGET_S


def _partial_path(root: Path) -> Path:
    return Path(root) / PARTIAL_RESULT_RELATIVE_PATH


def load_partial(root: Path) -> JsonDict:
    """Load the resume ledger, or an empty skeleton when no partial file exists.

    The ledger records, per completed game, the integrated + bare attempt ROWS exactly as exp4605
    produced them, so a resumed run can rebuild the SAME aggregated measurement it would have built in
    one shot. We key by game (the per-unit checkpoint granularity) so resume skips whole completed
    games. A corrupt/unreadable file is treated as no-progress (defensive: never crash on resume)."""

    path = _partial_path(root)
    if not path.exists():
        return {"games": {}}
    raw = _read_json(path)
    games = raw.get("games")
    if not isinstance(games, Mapping):
        return {"games": {}}
    out: dict[str, JsonDict] = {}
    for game, rows in games.items():
        if not isinstance(rows, Mapping):
            continue
        integrated = rows.get("integrated_attempts")
        bare = rows.get("bare_attempts")
        if isinstance(integrated, list) and isinstance(bare, list):
            # Coerce only Mapping rows; a scalar/garbage row would make dict(row) raise, which would
            # defeat the "never crash on resume" guarantee. We keep only well-formed attempt rows.
            out[str(game)] = {
                "integrated_attempts": [dict(row) for row in integrated if isinstance(row, Mapping)],
                "bare_attempts": [dict(row) for row in bare if isinstance(row, Mapping)],
            }
    return {"games": out}


def _write_partial(root: Path, ledger: Mapping[str, Any]) -> None:
    """Flush the resume ledger after each per-game unit completes (incremental durability).

    ATOMIC write: serialize to a sibling temp file then os.replace() onto the real path. A naive
    truncate-then-write (Path.write_text) leaves a TRUNCATED partial file if codex SIGKILLs the process
    mid-flush -- the next resume would then read corrupt JSON, fall back to no-progress, and silently
    re-run every already-done game. os.replace is atomic on POSIX, so a kill leaves EITHER the old
    complete ledger OR the new complete ledger, never a half-written one -- exactly the durability the
    checkpoint exists to provide."""

    path = _partial_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(dict(ledger), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def clear_partial(root: Path) -> None:
    """Remove the resume ledger once the FULL sweep is complete (its job is done)."""

    path = _partial_path(root)
    if path.exists():  # pragma: no branch - trivial guard
        path.unlink()


def run_held_out_proxy_checkpointed(
    root: Path,
    parity_test: Mapping[str, Any],
    *,
    now: Callable[[], float] = time.time,
    soft_budget_s: float | None = None,
    public_games: Sequence[str] | None = None,
) -> JsonDict:
    """Checkpoint/resume variant of run_held_out_proxy that survives the codex 4800s hard cap.

    This drives exp4605's per-GAME building blocks directly (variant_specs -> per-game variant attempts
    -> measurement_from_attempts -> build_artifact) instead of calling exp4605.run() as one opaque
    blackbox, so it can (a) persist progress after EACH game and (b) skip games already on disk.

    CRITICAL -- the SCORE is byte-identical to a full exp4605.run(): we accumulate the integrated and
    bare attempt rows in the SAME game-major order (sorted games x sorted variants) that
    measure_policy_pair uses, then aggregate the FULL accumulated lists with the SAME
    measurement_from_attempts + build_artifact. Per-game checkpointing only changes WHEN rows are
    persisted, never which rows are produced or how they are scored.

    Raises _BudgetExceeded when the soft budget is hit between games -- the run() handler then writes a
    partial artifact from whatever games are done. Returns the full proxy artifact when ALL games run.
    """

    from carnot import experiment_4605_live_integration_scored_agent as exp4605

    root_path = Path(root)
    budget = resolve_soft_budget_s() if soft_budget_s is None else float(soft_budget_s)
    # The budget is an ELAPSED-TIME duration, so anchor to the start and compare now()-started, NOT the
    # absolute clock. (A naive now()>=budget would compare a ~1.7e9 Unix timestamp against 4200s and
    # always trip, stopping before game 1 every time.)
    started = float(now())

    # The held-out lane is exp4605 with deepening ON and the 4 color variants. We set the same env the
    # non-checkpointed run_held_out_proxy sets so the per-variant attempt rows (which read _deepen_enabled
    # / the variant ids) are byte-identical to today's behavior.
    previous_deepen = os.environ.get(exp4605.DEEPEN_ENV)
    previous_variants = os.environ.get(exp4605.VARIANT_IDS_ENV)
    os.environ[exp4605.DEEPEN_ENV] = "1"
    os.environ[exp4605.VARIANT_IDS_ENV] = ",".join(str(item) for item in HELD_OUT_VARIANT_IDS)
    try:
        games = list(
            public_games if public_games is not None else exp4605._public_games(root_path)
        )
        variant_ids = HELD_OUT_VARIANT_IDS
        budget_per_game = exp4605.DEFAULT_BUDGET
        integrated_factory = exp4605.default_variant_runner_factory("integrated")
        bare_factory = exp4605.default_variant_runner_factory("bare")

        ledger = load_partial(root_path)
        done = dict(ledger.get("games", {}))

        # Rebuild accumulated attempts in deterministic game-major order so the final aggregation equals
        # the all-at-once run regardless of how many resumes it took to get here.
        ordered_games = sorted(str(game) for game in games)
        remaining = [game for game in ordered_games if game not in done]

        for index, game in enumerate(remaining):
            # Soft-budget check BETWEEN games: if the ELAPSED time means we cannot afford another
            # (minutes-long) game, stop gracefully NOW with whatever is already flushed, well under the
            # 4800s hard cap. Elapsed = now()-started so the comparison is a duration-vs-duration.
            if float(now()) - started >= budget:
                already = [g for g in ordered_games if g in done]
                still = [g for g in ordered_games if g not in done]
                raise _BudgetExceeded(done_games=already, remaining_games=still)

            specs = exp4605.variant_specs([game], variant_ids)
            integrated_attempts = [
                dict(integrated_factory(str(spec["game"]), spec, int(budget_per_game)))
                for spec in specs
            ]
            bare_attempts = [
                dict(bare_factory(str(spec["game"]), spec, int(budget_per_game))) for spec in specs
            ]
            done[game] = {
                "integrated_attempts": integrated_attempts,
                "bare_attempts": bare_attempts,
            }
            # INCREMENTAL CHECKPOINT: flush after EACH game so a kill loses at most one game.
            _write_partial(root_path, {"games": done})
            _ = index  # explicit: progress is the loop position, kept for readability

        # All games are done -- assemble the full proxy artifact with the SAME aggregation exp4605.run
        # uses, then drop the ledger.
        proxy = _assemble_proxy_from_ledger(
            exp4605=exp4605,
            done=done,
            ordered_games=ordered_games,
            parity_test=parity_test,
        )
        clear_partial(root_path)
        return proxy
    finally:
        if previous_deepen is None:
            os.environ.pop(exp4605.DEEPEN_ENV, None)
        else:
            os.environ[exp4605.DEEPEN_ENV] = previous_deepen
        if previous_variants is None:
            os.environ.pop(exp4605.VARIANT_IDS_ENV, None)
        else:
            os.environ[exp4605.VARIANT_IDS_ENV] = previous_variants


def _assemble_proxy_from_ledger(
    *,
    exp4605: Any,
    done: Mapping[str, Any],
    ordered_games: Sequence[str],
    parity_test: Mapping[str, Any],
) -> JsonDict:
    """Aggregate the per-game ledger rows into the exp4605 proxy artifact, SCORE-identical to one run.

    Concatenating the per-game attempt rows in sorted-game order reproduces the exact list that
    measure_policy_pair builds in one shot (it iterates variant_specs == sorted games x sorted
    variants). We then call the unchanged exp4605.measurement_from_attempts + build_artifact, so the
    first-win rate, CI, deepen rate, and verdict are computed by the SAME code on the SAME rows."""

    integrated_attempts: list[JsonDict] = []
    bare_attempts: list[JsonDict] = []
    for game in ordered_games:
        rows = done.get(game)
        if not isinstance(rows, Mapping):
            continue
        integrated_attempts.extend(dict(row) for row in rows.get("integrated_attempts", []))
        bare_attempts.extend(dict(row) for row in rows.get("bare_attempts", []))

    integrated = exp4605.measurement_from_attempts(integrated_attempts)
    bare = exp4605.measurement_from_attempts(bare_attempts)
    artifact = exp4605.build_artifact(
        preconditions_checked={"ok": True},
        integrated_measurement=integrated,
        bare_measurement=bare,
        parity_test=dict(parity_test),
        duration_s=1.0,
        random_seed=exp4605.RANDOM_SEED,
    )
    return dict(artifact)


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


def _partial_artifact(
    *,
    root: Path,
    preconditions_checked: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    budget_exceeded: _BudgetExceeded,
    replay_floor: Mapping[str, Any],
    v435_lever_inputs: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Emit a CLEAN partial artifact (exit 0, NOT a crash) when the soft budget stops the sweep.

    We assemble the proxy from ONLY the games already flushed to the resume ledger and score it with
    the SAME build_artifact path as a full run, so the numbers in a partial are honest (just over fewer
    variants). The verdict carries a terminal prefix (complete_partial_...) so the conductor reconciler
    treats it as a terminal-but-incomplete state, not a failed/partial-token retry. completed_variants
    / remaining_variants record exactly which color variants ran so the NEXT resumed run finishes the
    rest. ready_for_operator_submit stays False (a sub-B100 partial can never be ready)."""

    from carnot import experiment_4605_live_integration_scored_agent as exp4605

    root_path = Path(root)
    ledger = load_partial(root_path)
    done = dict(ledger.get("games", {}))
    ordered_games = sorted(set(done) | set(budget_exceeded.done_games))
    proxy = _assemble_proxy_from_ledger(
        exp4605=exp4605,
        done=done,
        ordered_games=ordered_games,
        parity_test=parity_test,
    )
    artifact = build_artifact(
        preconditions_checked=dict(preconditions_checked),
        parity_test=dict(parity_test),
        proxy_artifact=proxy,
        replay_floor=dict(replay_floor),
        v435_lever_inputs=dict(v435_lever_inputs),
        duration_s=duration_s,
    )
    completed_variants = [
        _variant_signature(game, variant)
        for game in budget_exceeded.done_games
        for variant in HELD_OUT_VARIANT_IDS
    ]
    remaining_variants = [
        _variant_signature(game, variant)
        for game in budget_exceeded.remaining_games
        for variant in HELD_OUT_VARIANT_IDS
    ]
    attempts = _extract_held_out_variant_attempts(proxy)
    artifact["partial"] = True
    artifact["completed_variants"] = completed_variants
    artifact["remaining_variants"] = remaining_variants
    artifact["completed_games"] = list(budget_exceeded.done_games)
    artifact["remaining_games"] = list(budget_exceeded.remaining_games)
    # Terminal prefix 'complete:' (a member of TERMINAL_PREFIXES) so the conductor reconciler reads this
    # as a terminal-but-incomplete state, NOT a partial-token retry. 'partial' here is the descriptive
    # tail, not a partial-failure token -- the partial: true field is the structured signal.
    artifact["honest_verdict"] = (
        f"complete: held_out_first_win_soft_budget_stop_partial_"
        f"{len(budget_exceeded.done_games)}_of_"
        f"{len(budget_exceeded.done_games) + len(budget_exceeded.remaining_games)}_games_"
        f"{attempts}_attempts_resume_to_finish"
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _variant_signature(game: str, variant_id: int) -> str:
    return f"{game}~color{int(variant_id):02d}"


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checker: PreconditionsChecker = check_preconditions,
    parity_check: ParityCheck = run_parity_test,
    proxy_runner: ProxyRunner = run_held_out_proxy_checkpointed,
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
    except _BudgetExceeded as budget_exc:
        # GRACEFUL soft-budget stop: the per-game sweep ran out of soft budget BEFORE finishing. Whatever
        # games completed are already flushed to the resume ledger; emit a clean partial: true artifact
        # (exit 0, NOT a crash) so codex sees a successful run well under its 4800s hard cap, and the
        # NEXT run resumes from the ledger. The partial file is intentionally NOT cleared here.
        checks["soft_budget_partial"] = True
        artifact = _partial_artifact(
            root=root_path,
            preconditions_checked=checks,
            parity_test=parity,
            budget_exceeded=budget_exc,
            replay_floor=replay_floor,
            v435_lever_inputs=lever_inputs,
            duration_s=duration(),
        )
        write_artifact(root_path, artifact)
        return artifact
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
    # COMPLETE: all games ran (possibly across multiple resumed runs). The final artifact is the SAME
    # schema/values as today's full run; partial: false marks it complete. The resume ledger was already
    # cleared by the checkpointed proxy runner once the last game finished; clear defensively in case a
    # custom (non-checkpointing) proxy_runner was injected.
    artifact["partial"] = False
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    clear_partial(root_path)
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
