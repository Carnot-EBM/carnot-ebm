"""Experiment 4740: .436 held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4740, SCENARIO-CAPSTONE-4740,
SCENARIO-CAPSTONE-4740-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4740-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4729_held_out_first_win_readiness as base


JsonDict = dict[str, Any]
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
ParityCheck = Callable[[Path], Mapping[str, Any]]
ProxyRunner = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]
ReplayFloorLoader = Callable[[Path], Mapping[str, Any]]
LeverInputLoader = Callable[[Path], Mapping[str, Any]]

EXPERIMENT = "experiment_4740_held_out_first_win_readiness"
EXPERIMENT_ID = 4740
SCHEMA = "carnot.arc.held_out_first_win_readiness_4740.v1"
RESULT_RELATIVE_PATH = "results/experiment_4740_held_out_first_win_readiness.json"
PARTIAL_RESULT_RELATIVE_PATH = "results/experiment_4740_held_out_first_win_readiness.partial.json"
PROXY_RESULT_RELATIVE_PATH = base.PROXY_RESULT_RELATIVE_PATH
REPLAY_FLOOR_RESULT_RELATIVE_PATH = base.REPLAY_FLOOR_RESULT_RELATIVE_PATH
REPLAY_FLOOR_PACKAGE_FALLBACK = base.REPLAY_FLOOR_PACKAGE_FALLBACK
FIRST_WIN_BASELINE = base.FIRST_WIN_BASELINE
MIN_HELD_OUT_VARIANT_ATTEMPTS = base.MIN_HELD_OUT_VARIANT_ATTEMPTS
HELD_OUT_VARIANT_ATTEMPT_FLOOR = base.HELD_OUT_VARIANT_ATTEMPT_FLOOR
HELD_OUT_VARIANT_IDS = base.HELD_OUT_VARIANT_IDS
RANDOM_SEED = 4740
SOFT_BUDGET_ENV = "EXP4740_SOFT_BUDGET_S"
DEFAULT_SOFT_BUDGET_S = base.DEFAULT_SOFT_BUDGET_S
TERMINAL_PREFIXES = base.TERMINAL_PREFIXES
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- the held-out lane scores the submitted "
    "config over cached variants (1s floor)."
)
REPLAY_FLOOR_NOTE = base.REPLAY_FLOOR_NOTE

SPEC_REFS = [
    "REQ-CAPSTONE-4740",
    "SCENARIO-CAPSTONE-4740",
    "SCENARIO-CAPSTONE-4740-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4740-FIELD-PRINCIPLES",
]

V436_LEVER_ARTIFACTS = {
    "a1": "results/experiment_4737_goal_energy_candidate_generation_valid_test.json",
    "a2": "results/experiment_4738_energy_fitness_qd_generation_valid_test.json",
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
            "verifier_ensemble_against_cached_candidates -- the held-out lane scores the submitted "
            "config over cached variants (1s floor)."
        )
    },
    "first_win_rate_integrated": {
        "principle": (
            "the held-out generic first-win on color-permuted variants -- the only offline proxy "
            "that tracks the scored leaderboard lane; the replay count does not."
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
            "deepening past L1 is the second scored lever; tracking it held-out keeps A4 honest "
            "about depth."
        )
    },
    "submitted_config_current": {
        "principle": (
            "true if the SUBMITTED_AGENT_CONFIG measured here is the one that would be submitted -- "
            "the deadline-readiness confirm (5 days out)."
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
            "the replay package stays a reproduced FLOOR, but its level count is explicitly NOT "
            "the leaderboard score."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when first-win is flat; the TAUTOLOGY carve-out reads it to downgrade "
            "CRITICAL->WARN (honest no-change)."
        )
    },
    "positive_control_passed": {
        "principle": (
            "bool(parity_test_green) -- GATES the TAUTOLOGY exemption; an unvalidated flat result "
            "is NOT excused."
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
    "v436_lever_inputs",
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


def _neutral_config_choice(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "unchanged"}
    return False


def _submitted_config_snapshot() -> JsonDict:  # pragma: no cover - live import boundary.
    from carnot import experiment_4605_live_integration_scored_agent as exp4605

    return dict(exp4605._submitted_config_snapshot())


def _submitted_config_current(
    *,
    v436_lever_inputs: Mapping[str, Any],
    parity_green: bool,
    submitted_config_snapshot: Mapping[str, Any] | None = None,
) -> bool:
    if not parity_green:
        return False
    snapshot: Mapping[str, Any] | None = submitted_config_snapshot
    for row in v436_lever_inputs.values():
        if not isinstance(row, Mapping):
            continue
        chosen = row.get("chosen_submitted_config")
        if _neutral_config_choice(chosen):
            continue
        if not isinstance(chosen, Mapping):
            return False
        if snapshot is None:
            snapshot = _submitted_config_snapshot()
        for key, expected in chosen.items():
            if snapshot.get(key) != expected:
                return False
    return True


def _readiness(
    *,
    parity_green: bool,
    submitted_config_current: bool,
    first_win_rate: float,
    baseline: float,
    ci_lower: float,
    attempts: int,
    null_delta_methodology_note: str,
    positive_control_passed: bool,
) -> bool:
    if not submitted_config_current:
        return False
    return base._readiness(
        parity_green=parity_green,
        first_win_rate=first_win_rate,
        baseline=baseline,
        ci_lower=ci_lower,
        attempts=attempts,
        null_delta_methodology_note=null_delta_methodology_note,
        positive_control_passed=positive_control_passed,
    )


def _honest_verdict(
    *,
    submitted_config_current: bool,
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
    if not submitted_config_current:
        return "complete: submitted_config_not_current_no_leaderboard_change"
    return base._honest_verdict(
        readiness=readiness,
        parity_green=parity_green,
        first_win_rate=first_win_rate,
        baseline=baseline,
        ci_lower=ci_lower,
        attempts=attempts,
        positive_control_passed=positive_control_passed,
    )


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    proxy_artifact: Mapping[str, Any],
    replay_floor: Mapping[str, Any],
    v436_lever_inputs: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
    submitted_config_snapshot: Mapping[str, Any] | None = None,
) -> JsonDict:
    first_win_rate = base._extract_first_win_rate(proxy_artifact)
    first_win_ci_lower = base._extract_ci_lower(proxy_artifact)
    first_win_delta = round(first_win_rate - FIRST_WIN_BASELINE, 6)
    attempts = base._extract_held_out_variant_attempts(proxy_artifact)
    multi_level_deepen_rate = base._extract_multi_level_deepen_rate(proxy_artifact)
    parity_green = bool(parity_test.get("passed"))
    positive_control = bool(parity_green)
    current = _submitted_config_current(
        v436_lever_inputs=v436_lever_inputs,
        parity_green=parity_green,
        submitted_config_snapshot=submitted_config_snapshot,
    )
    null_note = base._null_delta_note(
        first_win_rate=first_win_rate,
        baseline=FIRST_WIN_BASELINE,
        positive_control_passed=positive_control,
    )
    ready = _readiness(
        parity_green=parity_green,
        submitted_config_current=current,
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
            submitted_config_current=current,
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
        "submitted_config_current": current,
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
        "v436_lever_inputs": dict(v436_lever_inputs),
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
    first_win_rate = base._float(artifact.get("first_win_rate_integrated"))
    baseline = base._float(artifact.get("first_win_baseline"), FIRST_WIN_BASELINE)
    ci_lower = base._float(artifact.get("first_win_ci_lower"))
    attempts = int(base._float(artifact.get("held_out_variant_attempts")))
    parity_green = artifact.get("parity_test_green") is True
    expected_positive_control = bool(parity_green)
    if artifact.get("positive_control_passed") is not expected_positive_control:
        errors.append("positive_control_passed")
    note = str(artifact.get("null_delta_methodology_note") or "")
    if base._is_flat_delta(first_win_rate, baseline) and not note.strip():
        errors.append("null_delta_methodology_note")
    current = artifact.get("submitted_config_current") is True
    expected_readiness = _readiness(
        parity_green=parity_green,
        submitted_config_current=current,
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
    partial = artifact.get("partial") is True
    if not blocked and not partial and attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
        errors.append("held_out_variant_attempts_below_minimum")
    if not current and artifact.get("ready_for_operator_submit") is True:
        errors.append("submitted_config_current_gate")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(dict.fromkeys(errors))


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:  # pragma: no cover.
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    base._write_json(path, artifact)
    return path


def load_v436_lever_inputs(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover.
    root_path = Path(root)
    out: JsonDict = {}
    for key, rel in V436_LEVER_ARTIFACTS.items():
        path = root_path / rel
        payload = base._read_json(path)
        row: JsonDict = {
            "path": rel,
            "exists": path.exists(),
            "experiment": payload.get("experiment"),
            "honest_verdict": payload.get("honest_verdict"),
            "chosen_submitted_config": payload.get("chosen_submitted_config"),
        }
        if path.exists():
            row["sha256"] = base._file_sha256(path)
        else:
            row["sha256"] = ""
            row["error"] = "missing_artifact"
        out[key] = row
    return out


def check_preconditions(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    return dict(base.check_preconditions(root))


def run_parity_test(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    return dict(base.run_parity_test(root))


def _with_base_checkpoint_constants(func: Callable[[], JsonDict]) -> JsonDict:  # pragma: no cover.
    old_partial = base.PARTIAL_RESULT_RELATIVE_PATH
    old_budget_env = base.SOFT_BUDGET_ENV
    old_default_budget = base.DEFAULT_SOFT_BUDGET_S
    try:
        base.PARTIAL_RESULT_RELATIVE_PATH = PARTIAL_RESULT_RELATIVE_PATH
        base.SOFT_BUDGET_ENV = SOFT_BUDGET_ENV
        base.DEFAULT_SOFT_BUDGET_S = DEFAULT_SOFT_BUDGET_S
        return func()
    finally:
        base.PARTIAL_RESULT_RELATIVE_PATH = old_partial
        base.SOFT_BUDGET_ENV = old_budget_env
        base.DEFAULT_SOFT_BUDGET_S = old_default_budget


def run_held_out_proxy_checkpointed(
    root: Path,
    parity_test: Mapping[str, Any],
    *,
    now: Callable[[], float] = time.time,
    soft_budget_s: float | None = None,
) -> JsonDict:  # pragma: no cover - live ARC boundary.
    return _with_base_checkpoint_constants(
        lambda: dict(
            base.run_held_out_proxy_checkpointed(
                root,
                parity_test,
                now=now,
                soft_budget_s=soft_budget_s,
            )
        )
    )


def _cached_proxy_after_unchanged_levers(
    root: Path,
    v436_lever_inputs: Mapping[str, Any],
) -> JsonDict | None:  # pragma: no cover - filesystem boundary.
    if not _submitted_config_current(
        v436_lever_inputs=v436_lever_inputs,
        parity_green=True,
        submitted_config_snapshot={},
    ):
        return None
    path = root / PROXY_RESULT_RELATIVE_PATH
    proxy = base._read_json(path)
    if base._extract_held_out_variant_attempts(proxy) < MIN_HELD_OUT_VARIANT_ATTEMPTS:
        return None
    if "multi_level_solve_rate" not in proxy and "multi_level_deepen_rate_integrated" not in proxy:
        return None
    proxy["proxy_cache_used"] = True
    proxy["proxy_cache_reason"] = (
        "A1/A2 chosen_submitted_config were unchanged, so the existing Exp4605 submitted-config "
        "held-out lane remains current."
    )
    return proxy


def load_cached_or_run_held_out_proxy(
    root: Path,
    parity_test: Mapping[str, Any],
    *,
    v436_lever_inputs: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - live/cache boundary.
    cached = _cached_proxy_after_unchanged_levers(root, v436_lever_inputs)
    if cached is not None:
        return cached
    return run_held_out_proxy_checkpointed(root, parity_test)


def load_replay_package_floor(root: Path) -> JsonDict:  # pragma: no cover - filesystem boundary.
    return dict(base.load_replay_package_floor(root))


def _floor_duration(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:  # pragma: no cover - timing boundary.
    return base._floor_duration(started_at=started_at, now=now, sleep_fn=sleep_fn)


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    reason: str,
    duration_s: float,
    replay_floor: Mapping[str, Any] | None = None,
    v436_lever_inputs: Mapping[str, Any] | None = None,
) -> JsonDict:  # pragma: no cover - defensive run boundary.
    artifact = build_artifact(
        preconditions_checked=dict(preconditions_checked),
        parity_test={"passed": False, "blocked_reason": reason},
        proxy_artifact={},
        replay_floor=dict(replay_floor or {}),
        v436_lever_inputs=dict(v436_lever_inputs or {}),
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
    proxy_runner: ProxyRunner | None = None,
    replay_floor_loader: ReplayFloorLoader = load_replay_package_floor,
    lever_input_loader: LeverInputLoader = load_v436_lever_inputs,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:  # pragma: no cover - orchestration boundary.
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
            v436_lever_inputs=lever_inputs,
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
            v436_lever_inputs=lever_inputs,
        )
        write_artifact(root_path, artifact)
        return artifact

    selected_proxy_runner = proxy_runner
    if selected_proxy_runner is None:
        selected_proxy_runner = lambda r, p: load_cached_or_run_held_out_proxy(
            r, p, v436_lever_inputs=lever_inputs
        )

    try:
        proxy = dict(selected_proxy_runner(root_path, parity))
    except base._BudgetExceeded as budget_exc:
        checks["soft_budget_partial"] = True
        artifact = base._partial_artifact(
            root=root_path,
            preconditions_checked=checks,
            parity_test=parity,
            budget_exceeded=budget_exc,
            replay_floor=replay_floor,
            v435_lever_inputs=lever_inputs,
            duration_s=duration(),
        )
        artifact = build_artifact(
            preconditions_checked=artifact["preconditions_checked"],
            parity_test=parity,
            proxy_artifact=artifact["held_out_proxy_summary"],
            replay_floor=replay_floor,
            v436_lever_inputs=lever_inputs,
            duration_s=artifact["duration_s"],
        )
        artifact["partial"] = True
        artifact["honest_verdict"] = (
            f"complete: held_out_first_win_soft_budget_stop_partial_"
            f"{len(budget_exc.done_games)}_of_"
            f"{len(budget_exc.done_games) + len(budget_exc.remaining_games)}_games_"
            f"{artifact['held_out_variant_attempts']}_attempts_resume_to_finish"
        )
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
        write_artifact(root_path, artifact)
        return artifact
    except Exception as exc:
        checks["proxy_error"] = repr(exc)[:500]
        checks["blocked_resource"] = "experiment_4605_proxy"
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason="experiment_4605_proxy",
            duration_s=duration(),
            replay_floor=replay_floor,
            v436_lever_inputs=lever_inputs,
        )
        write_artifact(root_path, artifact)
        return artifact

    attempts = base._extract_held_out_variant_attempts(proxy)
    if attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
        checks["blocked_resource"] = "experiment_4605_proxy_b100"
        checks["held_out_variant_attempts"] = attempts
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason="experiment_4605_proxy_b100",
            duration_s=duration(),
            replay_floor=replay_floor,
            v436_lever_inputs=lever_inputs,
        )
        write_artifact(root_path, artifact)
        return artifact

    artifact = build_artifact(
        preconditions_checked=checks,
        parity_test=parity,
        proxy_artifact=proxy,
        replay_floor=replay_floor,
        v436_lever_inputs=lever_inputs,
        duration_s=duration(),
    )
    artifact["partial"] = False
    artifact["held_out_proxy_summary"]["proxy_cache_used"] = bool(proxy.get("proxy_cache_used"))
    if proxy.get("proxy_cache_reason"):
        artifact["held_out_proxy_summary"]["proxy_cache_reason"] = proxy.get("proxy_cache_reason")
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    _with_base_checkpoint_constants(lambda: (base.clear_partial(root_path), {} )[1])
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(f"first_win_rate_integrated={artifact['first_win_rate_integrated']}")
    print(f"first_win_ci_lower={artifact['first_win_ci_lower']}")
    print(f"multi_level_deepen_rate_integrated={artifact['multi_level_deepen_rate_integrated']}")
    print(f"submitted_config_current={artifact['submitted_config_current']}")
    print(f"ready_for_operator_submit={artifact['ready_for_operator_submit']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
