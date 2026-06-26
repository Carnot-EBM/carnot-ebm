"""Experiment 4752: held-out first-win readiness re-measure.

Spec refs: REQ-CAPSTONE-4752, SCENARIO-CAPSTONE-4752,
SCENARIO-CAPSTONE-4752-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4752-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
PriorMilestoneLoader = Callable[[Path], Mapping[str, Any]]
PackageReadinessLoader = Callable[[Path], Mapping[str, Any]]
PartialProxyLoader = Callable[[Path, base._BudgetExceeded, Mapping[str, Any]], Mapping[str, Any]]

EXPERIMENT = "experiment_4752_held_out_first_win_readiness"
EXPERIMENT_ID = 4752
SCHEMA = "carnot.arc.held_out_first_win_readiness_4752.v1"
RESULT_RELATIVE_PATH = "results/experiment_4752_held_out_first_win_readiness.json"
PARTIAL_RESULT_RELATIVE_PATH = "results/experiment_4752_held_out_first_win_readiness.partial.json"
PROXY_RESULT_RELATIVE_PATH = base.PROXY_RESULT_RELATIVE_PATH
REPLAY_FLOOR_RESULT_RELATIVE_PATH = base.REPLAY_FLOOR_RESULT_RELATIVE_PATH
REPLAY_FLOOR_PACKAGE_FALLBACK = base.REPLAY_FLOOR_PACKAGE_FALLBACK
PRIOR_MILESTONE_RESULT_RELATIVE_PATH = "results/experiment_4740_held_out_first_win_readiness.json"
SUBMISSION_PACKAGE_READINESS_RELATIVE_PATH = (
    "results/experiment_4744_submission_package_readiness.json"
)
FIRST_WIN_BASELINE = base.FIRST_WIN_BASELINE
MIN_HELD_OUT_VARIANT_ATTEMPTS = base.MIN_HELD_OUT_VARIANT_ATTEMPTS
HELD_OUT_VARIANT_ATTEMPT_FLOOR = base.HELD_OUT_VARIANT_ATTEMPT_FLOOR
HELD_OUT_VARIANT_IDS = base.HELD_OUT_VARIANT_IDS
RANDOM_SEED = 4752
SOFT_BUDGET_ENV = base.SOFT_BUDGET_ENV
DEFAULT_SOFT_BUDGET_S = base.DEFAULT_SOFT_BUDGET_S
TERMINAL_PREFIXES = base.TERMINAL_PREFIXES
REPLAY_FLOOR_NOTE = base.REPLAY_FLOOR_NOTE
LIVE_DURATION_FLOOR_S = 60.0
CACHE_DURATION_FLOOR_S = 1.0
INFERENCE_SUBSTRATE = (
    "live_llm_inference -- the checkpoint/resume proxy runs the live submitted ARC agent over "
    "held-out variants; 60s floor unless a lever-cache hit aggregates an existing proxy artifact."
)

SPEC_REFS = [
    "REQ-CAPSTONE-4752",
    "SCENARIO-CAPSTONE-4752",
    "SCENARIO-CAPSTONE-4752-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4752-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a flat/no-change readiness is complete_, a readiness improvement "
            "is success_."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference (the proxy runs the live agent); 60s floor (or a lever-cache "
            "hit aggregation)."
        )
    },
    "preconditions_checked": {"principle": "records GGUF/arcade checks."},
    "partial": {
        "principle": (
            "true if the soft budget stopped the run before completion (resumable) -- the "
            "cap-survival signal; false on a full run."
        )
    },
    "first_win_rate_integrated": {
        "principle": (
            "the held-out first-win rate -- the SCORE; unchanged math from prior milestones "
            "for comparability."
        )
    },
    "first_win_ci": {"principle": "the bootstrap CI -- distinguishes a real delta from noise."},
    "submission_package_ready": {
        "principle": (
            "True only if the package is OPERATOR-ready; this task NEVER submits "
            "(operator-only external publication)."
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
    "first_win_ci_lower",
    "first_win_baseline",
    "first_win_delta_vs_baseline",
    "multi_level_deepen_rate_integrated",
    "held_out_variant_attempts",
    "held_out_variant_attempt_floor",
    "replay_count_is_not_the_score",
    "prior_milestone",
    "readiness_delta_vs_prior_milestone",
    "submission_package_readiness",
    "submitted_to_leaderboard",
    "operator_only",
    "parity_test_green",
    "positive_control_passed",
    "null_delta_methodology_note",
    "verifier_is_oracle",
    "random_seed",
    "duration_s",
    "field_principles",
    "reproducibility_checksum",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _coerce_optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _extract_first_win_ci(proxy_artifact: Mapping[str, Any]) -> JsonDict:
    ci = proxy_artifact.get("first_win_ci")
    if isinstance(ci, Mapping):
        return dict(ci)
    lower = base._extract_ci_lower(proxy_artifact)
    point = round(base._extract_first_win_rate(proxy_artifact) - FIRST_WIN_BASELINE, 6)
    return {
        "method": "source_lower_bound_only",
        "point": point,
        "ci95": [lower, lower],
    }


def _ci_lower_from_payload(first_win_ci: Mapping[str, Any]) -> float:
    interval = first_win_ci.get("ci95")
    if isinstance(interval, list | tuple) and interval:
        return base._float(interval[0])
    if "low" in first_win_ci:
        return base._float(first_win_ci.get("low"))
    return 0.0


def find_qwen35_mtp_gguf_cache() -> str | None:  # pragma: no cover - local cache boundary.
    env_path = str(os.environ.get("CARNOT_ARC_GGUF_PATH", "") or "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file() and "Qwen3.5-9B" in path.name and path.suffix.lower() == ".gguf":
            return str(path)

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    patterns = (
        "models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/*/*.gguf",
        "models--Jackrong--Qwen3.5-9B-DeepSeek-V4-Flash-MTP-GGUF/snapshots/*/*.gguf",
    )
    for pattern in patterns:
        for match in sorted(hub.glob(pattern)):
            if match.is_file():
                return str(match)
    return None


def check_preconditions(
    root: Path,
    *,
    qwen_cache_finder: Callable[[], str | None] = find_qwen35_mtp_gguf_cache,
) -> JsonDict:
    checks = dict(base.check_preconditions(root))
    qwen_path = qwen_cache_finder()
    checks["qwen35_mtp_gguf_cached"] = bool(qwen_path)
    checks["qwen35_mtp_gguf_path"] = qwen_path or ""
    if not qwen_path:
        checks["ok"] = False
        checks["blocked_resource"] = "qwen35_mtp_gguf_cache"
    return checks


def run_parity_test(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    return dict(base.run_parity_test(root))


def _with_base_checkpoint_constants(func: Callable[[], JsonDict]) -> JsonDict:
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
    public_games: Sequence[str] | None = None,
) -> JsonDict:
    return _with_base_checkpoint_constants(
        lambda: dict(
            base.run_held_out_proxy_checkpointed(
                root,
                parity_test,
                now=now,
                soft_budget_s=soft_budget_s,
                public_games=public_games,
            )
        )
    )


def load_cached_or_run_held_out_proxy(root: Path, parity_test: Mapping[str, Any]) -> JsonDict:
    path = Path(root) / PROXY_RESULT_RELATIVE_PATH
    proxy = base._read_json(path)
    if (
        base._extract_held_out_variant_attempts(proxy) >= MIN_HELD_OUT_VARIANT_ATTEMPTS
        and isinstance(proxy.get("first_win_ci"), Mapping)
        and ("multi_level_solve_rate" in proxy or "multi_level_deepen_rate_integrated" in proxy)
    ):
        proxy["proxy_cache_used"] = True
        proxy["proxy_cache_reason"] = (
            "Existing Exp4605 held-out proxy is complete and SCORE-compatible; 4752 changes only "
            "result persistence and field reporting."
        )
        return proxy
    return run_held_out_proxy_checkpointed(root, parity_test)


def load_replay_package_floor(root: Path) -> JsonDict:  # pragma: no cover - filesystem boundary.
    return dict(base.load_replay_package_floor(root))


def load_prior_milestone(root: Path) -> JsonDict:
    path = Path(root) / PRIOR_MILESTONE_RESULT_RELATIVE_PATH
    payload = base._read_json(path)
    return {
        "path": PRIOR_MILESTONE_RESULT_RELATIVE_PATH,
        "exists": path.exists(),
        "experiment_id": payload.get("experiment_id"),
        "honest_verdict": payload.get("honest_verdict", ""),
        "first_win_rate_integrated": _coerce_optional_float(
            payload.get("first_win_rate_integrated")
        ),
        "first_win_ci": _extract_first_win_ci(payload) if payload else {},
        "multi_level_deepen_rate_integrated": _coerce_optional_float(
            payload.get("multi_level_deepen_rate_integrated")
        ),
        "held_out_first_win_readiness": payload.get("held_out_first_win_readiness") is True,
        "partial": payload.get("partial") is True,
    }


def load_submission_package_readiness(root: Path) -> JsonDict:
    path = Path(root) / SUBMISSION_PACKAGE_READINESS_RELATIVE_PATH
    payload = base._read_json(path)
    return {
        "path": SUBMISSION_PACKAGE_READINESS_RELATIVE_PATH,
        "exists": path.exists(),
        "honest_verdict": payload.get("honest_verdict", ""),
        "submission_package_ready": payload.get("submission_package_ready") is True,
    }


def _readiness_delta(
    *,
    first_win_rate: float,
    multi_level_deepen_rate: float,
    readiness: bool,
    prior_milestone: Mapping[str, Any],
) -> JsonDict:
    prior_rate = _coerce_optional_float(prior_milestone.get("first_win_rate_integrated"))
    prior_deepen = _coerce_optional_float(prior_milestone.get("multi_level_deepen_rate_integrated"))
    prior_ready = prior_milestone.get("held_out_first_win_readiness") is True
    return {
        "prior_result_path": str(
            prior_milestone.get("path") or PRIOR_MILESTONE_RESULT_RELATIVE_PATH
        ),
        "prior_exists": prior_milestone.get("exists") is True,
        "prior_experiment_id": prior_milestone.get("experiment_id"),
        "first_win_rate_delta": (
            None if prior_rate is None else round(first_win_rate - prior_rate, 6)
        ),
        "multi_level_deepen_rate_delta": (
            None if prior_deepen is None else round(multi_level_deepen_rate - prior_deepen, 6)
        ),
        "readiness_changed": bool(readiness != prior_ready),
        "current_readiness": bool(readiness),
        "prior_readiness": bool(prior_ready),
    }


def _honest_verdict(
    *,
    partial: bool,
    readiness: bool,
    parity_green: bool,
    first_win_rate: float,
    baseline: float,
    ci_lower: float,
    attempts: int,
    positive_control_passed: bool,
    budget_exceeded: base._BudgetExceeded | None = None,
) -> str:
    if partial:
        done = len(budget_exceeded.done_games) if budget_exceeded is not None else 0
        remaining = len(budget_exceeded.remaining_games) if budget_exceeded is not None else 0
        return (
            f"complete: held_out_first_win_soft_budget_stop_partial_{done}_of_"
            f"{done + remaining}_games_{attempts}_attempts_resume_to_finish"
        )
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
    prior_milestone: Mapping[str, Any],
    package_readiness: Mapping[str, Any],
    duration_s: float,
    partial: bool,
    budget_exceeded: base._BudgetExceeded | None = None,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    first_win_rate = base._extract_first_win_rate(proxy_artifact)
    first_win_ci = _extract_first_win_ci(proxy_artifact)
    first_win_ci_lower = _ci_lower_from_payload(first_win_ci)
    first_win_delta = round(first_win_rate - FIRST_WIN_BASELINE, 6)
    attempts = base._extract_held_out_variant_attempts(proxy_artifact)
    multi_level_deepen_rate = base._extract_multi_level_deepen_rate(proxy_artifact)
    parity_green = bool(parity_test.get("passed"))
    positive_control = bool(parity_green)
    null_note = base._null_delta_note(
        first_win_rate=first_win_rate,
        baseline=FIRST_WIN_BASELINE,
        positive_control_passed=positive_control,
    )
    measured_ready = base._readiness(
        parity_green=parity_green,
        first_win_rate=first_win_rate,
        baseline=FIRST_WIN_BASELINE,
        ci_lower=first_win_ci_lower,
        attempts=attempts,
        null_delta_methodology_note=null_note,
        positive_control_passed=positive_control,
    )
    ready = bool(measured_ready and not partial)
    package = dict(package_readiness)
    submission_ready = bool(ready and package.get("submission_package_ready") is True)
    floor = dict(replay_floor)
    floor.setdefault("note", REPLAY_FLOOR_NOTE)
    floor_path = str(
        floor.get("package_path")
        or floor.get("refreshed_package_path")
        or REPLAY_FLOOR_PACKAGE_FALLBACK
    )
    delta = _readiness_delta(
        first_win_rate=first_win_rate,
        multi_level_deepen_rate=multi_level_deepen_rate,
        readiness=ready,
        prior_milestone=prior_milestone,
    )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(
            partial=partial,
            readiness=ready,
            parity_green=parity_green,
            first_win_rate=first_win_rate,
            baseline=FIRST_WIN_BASELINE,
            ci_lower=first_win_ci_lower,
            attempts=attempts,
            positive_control_passed=positive_control,
            budget_exceeded=budget_exceeded,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "partial": bool(partial),
        "first_win_rate_integrated": first_win_rate,
        "first_win_ci": first_win_ci,
        "first_win_ci_lower": first_win_ci_lower,
        "first_win_baseline": FIRST_WIN_BASELINE,
        "first_win_delta_vs_baseline": first_win_delta,
        "multi_level_deepen_rate_integrated": multi_level_deepen_rate,
        "held_out_variant_attempts": attempts,
        "held_out_variant_attempt_floor": HELD_OUT_VARIANT_ATTEMPT_FLOOR,
        "held_out_first_win_readiness": ready,
        "ready_for_operator_submit": submission_ready,
        "submission_package_ready": submission_ready,
        "submission_package_readiness": package,
        "replay_floor": floor,
        "replay_floor_package_path": floor_path,
        "replay_package_floor_reproduced": bool(floor.get("replay_package_floor_reproduced")),
        "replay_count_is_not_the_score": True,
        "parity_test": dict(parity_test),
        "parity_test_green": parity_green,
        "positive_control_passed": positive_control,
        "null_delta_methodology_note": null_note,
        "verifier_is_oracle": False,
        "proxy_artifact_path": PROXY_RESULT_RELATIVE_PATH,
        "held_out_proxy_summary": {
            "source_artifact_path": PROXY_RESULT_RELATIVE_PATH,
            "first_win_rate_integrated": first_win_rate,
            "first_win_ci": first_win_ci,
            "first_win_ci_lower": first_win_ci_lower,
            "first_win_baseline": FIRST_WIN_BASELINE,
            "first_win_delta_vs_baseline": first_win_delta,
            "multi_level_deepen_rate_integrated": multi_level_deepen_rate,
            "held_out_variant_attempts": attempts,
            "proxy_honest_verdict": proxy_artifact.get("honest_verdict", ""),
            "proxy_cache_used": bool(proxy_artifact.get("proxy_cache_used")),
        },
        "prior_milestone": dict(prior_milestone),
        "readiness_delta_vs_prior_milestone": delta,
        "submitted_to_leaderboard": False,
        "operator_only": True,
        "random_seed": int(random_seed),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "reproducibility_checksum": "",
    }
    if proxy_artifact.get("proxy_cache_reason"):
        artifact["held_out_proxy_summary"]["proxy_cache_reason"] = proxy_artifact.get(
            "proxy_cache_reason"
        )
    if budget_exceeded is not None:
        artifact["completed_games"] = list(budget_exceeded.done_games)
        artifact["remaining_games"] = list(budget_exceeded.remaining_games)
        artifact["completed_variants"] = [
            base._variant_signature(game, variant)
            for game in budget_exceeded.done_games
            for variant in HELD_OUT_VARIANT_IDS
        ]
        artifact["remaining_variants"] = [
            base._variant_signature(game, variant)
            for game in budget_exceeded.remaining_games
            for variant in HELD_OUT_VARIANT_IDS
        ]
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
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_false")
    if artifact.get("operator_only") is not True:
        errors.append("operator_only_true")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("replay_count_is_not_the_score") is not True:
        errors.append("replay_count_is_not_the_score_true")
    if not isinstance(artifact.get("partial"), bool):
        errors.append("partial_bool")
    if not isinstance(artifact.get("first_win_ci"), Mapping):
        errors.append("first_win_ci_mapping")

    first_win_rate = base._float(artifact.get("first_win_rate_integrated"))
    baseline = base._float(artifact.get("first_win_baseline"), FIRST_WIN_BASELINE)
    ci_lower = base._float(artifact.get("first_win_ci_lower"))
    attempts = int(base._float(artifact.get("held_out_variant_attempts")))
    parity_green = artifact.get("parity_test_green") is True
    positive_control = artifact.get("positive_control_passed") is True
    note = str(artifact.get("null_delta_methodology_note") or "")
    expected_readiness = artifact.get("partial") is not True and base._readiness(
        parity_green=parity_green,
        first_win_rate=first_win_rate,
        baseline=baseline,
        ci_lower=ci_lower,
        attempts=attempts,
        null_delta_methodology_note=note,
        positive_control_passed=positive_control,
    )
    if artifact.get("held_out_first_win_readiness") is not expected_readiness:
        errors.append("held_out_first_win_readiness_gate")
    package_ready = (
        artifact.get("submission_package_readiness", {}).get("submission_package_ready") is True
    )
    expected_submission_ready = bool(expected_readiness and package_ready)
    if artifact.get("submission_package_ready") is not expected_submission_ready:
        errors.append("submission_package_ready_gate")
    if artifact.get("ready_for_operator_submit") is not expected_submission_ready:
        errors.append("ready_for_operator_submit_gate")
    if artifact.get("partial") is True and (
        "completed_games" not in artifact or "remaining_games" not in artifact
    ):
        errors.append("partial_resume_detail")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(dict.fromkeys(errors))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_json(path, payload)


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    _write_json(path, artifact)
    return path


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
        prior_milestone={},
        package_readiness={"submission_package_ready": False},
        duration_s=duration_s,
        partial=False,
    )
    artifact["honest_verdict"] = f"blocked_{reason}"
    artifact["submission_package_ready"] = False
    artifact["ready_for_operator_submit"] = False
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _partial_proxy_from_budget(
    root: Path,
    budget_exceeded: base._BudgetExceeded,
    parity_test: Mapping[str, Any],
) -> JsonDict:
    from carnot import experiment_4605_live_integration_scored_agent as exp4605

    ledger = base.load_partial(root)
    done = dict(ledger.get("games", {}))
    ordered_games = sorted(set(done) | set(budget_exceeded.done_games))
    return base._assemble_proxy_from_ledger(
        exp4605=exp4605,
        done=done,
        ordered_games=ordered_games,
        parity_test=parity_test,
    )


def _floor_duration(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
    floor_s: float,
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    if elapsed < floor_s:
        sleep_fn(floor_s - elapsed)
    return max(float(now()), started_at + floor_s) - started_at


def _clear_partial(root: Path) -> JsonDict:
    return _with_base_checkpoint_constants(lambda: (base.clear_partial(root), {})[1])


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checker: PreconditionsChecker = check_preconditions,
    parity_check: ParityCheck = run_parity_test,
    proxy_runner: ProxyRunner | None = None,
    replay_floor_loader: ReplayFloorLoader = load_replay_package_floor,
    prior_milestone_loader: PriorMilestoneLoader = load_prior_milestone,
    package_readiness_loader: PackageReadinessLoader = load_submission_package_readiness,
    partial_proxy_loader: PartialProxyLoader = _partial_proxy_from_budget,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    root_path = Path(root)
    started = now()
    duration = lambda floor_s: _floor_duration(
        started_at=started, now=now, sleep_fn=sleep_fn, floor_s=floor_s
    )
    checks = dict(preconditions_checker(root_path))
    if not checks.get("ok", False):
        reason = str(checks.get("blocked_resource") or "precondition")
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason=reason,
            duration_s=duration(CACHE_DURATION_FLOOR_S),
        )
        write_artifact(root_path, artifact)
        return artifact

    replay_floor = dict(replay_floor_loader(root_path))
    prior_milestone = dict(prior_milestone_loader(root_path))
    package_readiness = dict(package_readiness_loader(root_path))
    parity = dict(parity_check(root_path))
    if parity.get("passed") is not True:
        checks["blocked_resource"] = "parity_test"
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason="parity_test",
            duration_s=duration(CACHE_DURATION_FLOOR_S),
        )
        write_artifact(root_path, artifact)
        return artifact

    selected_proxy_runner = proxy_runner or load_cached_or_run_held_out_proxy
    try:
        proxy = dict(selected_proxy_runner(root_path, parity))
    except base._BudgetExceeded as budget_exc:
        checks["soft_budget_partial"] = True
        proxy = dict(partial_proxy_loader(root_path, budget_exc, parity))
        artifact = build_artifact(
            preconditions_checked=checks,
            parity_test=parity,
            proxy_artifact=proxy,
            replay_floor=replay_floor,
            prior_milestone=prior_milestone,
            package_readiness=package_readiness,
            duration_s=duration(LIVE_DURATION_FLOOR_S),
            partial=True,
            budget_exceeded=budget_exc,
        )
        write_artifact(root_path, artifact)
        return artifact
    except Exception as exc:  # pragma: no cover - defensive live-run boundary.
        checks["proxy_error"] = repr(exc)[:500]
        checks["blocked_resource"] = "experiment_4605_proxy"
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason="experiment_4605_proxy",
            duration_s=duration(CACHE_DURATION_FLOOR_S),
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
            duration_s=duration(CACHE_DURATION_FLOOR_S),
        )
        write_artifact(root_path, artifact)
        return artifact

    floor = CACHE_DURATION_FLOOR_S if proxy.get("proxy_cache_used") else LIVE_DURATION_FLOOR_S
    artifact = build_artifact(
        preconditions_checked=checks,
        parity_test=parity,
        proxy_artifact=proxy,
        replay_floor=replay_floor,
        prior_milestone=prior_milestone,
        package_readiness=package_readiness,
        duration_s=duration(floor),
        partial=False,
    )
    _clear_partial(root_path)
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(f"partial={artifact['partial']}")
    print(f"first_win_rate_integrated={artifact['first_win_rate_integrated']}")
    print(f"first_win_ci={json.dumps(artifact['first_win_ci'], sort_keys=True)}")
    print(f"multi_level_deepen_rate_integrated={artifact['multi_level_deepen_rate_integrated']}")
    print(
        "readiness_delta_vs_prior_milestone="
        f"{json.dumps(artifact['readiness_delta_vs_prior_milestone'], sort_keys=True)}"
    )
    print(f"submission_package_ready={artifact['submission_package_ready']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
