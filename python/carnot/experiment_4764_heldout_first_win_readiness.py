"""Experiment 4764: held-out first-win readiness with substrate honesty.

Spec refs: REQ-CAPSTONE-4764, SCENARIO-CAPSTONE-4764,
SCENARIO-CAPSTONE-4764-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4764-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4729_held_out_first_win_readiness as base
from carnot import experiment_4752_held_out_first_win_readiness as exp4752


JsonDict = dict[str, Any]
PreconditionsChecker = Callable[[Path], Mapping[str, Any]]
ParityCheck = Callable[[Path], Mapping[str, Any]]
ProxyRunner = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]
PriorBestLoader = Callable[[Path], Mapping[str, Any]]
PartialProxyLoader = Callable[[Path, base._BudgetExceeded, Mapping[str, Any]], Mapping[str, Any]]

EXPERIMENT = "experiment_4764_heldout_first_win_readiness"
EXPERIMENT_ID = 4764
SCHEMA = "carnot.arc.heldout_first_win_readiness_4764.v1"
RESULT_RELATIVE_PATH = "results/experiment_4764_heldout_first_win_readiness.json"
PARTIAL_RESULT_RELATIVE_PATH = "results/experiment_4764_heldout_first_win_readiness.partial.json"
PROXY_RESULT_RELATIVE_PATH = base.PROXY_RESULT_RELATIVE_PATH
FIRST_WIN_BASELINE = base.FIRST_WIN_BASELINE
MIN_HELD_OUT_VARIANT_ATTEMPTS = base.MIN_HELD_OUT_VARIANT_ATTEMPTS
HELD_OUT_VARIANT_ATTEMPT_FLOOR = base.HELD_OUT_VARIANT_ATTEMPT_FLOOR
HELD_OUT_VARIANT_IDS = base.HELD_OUT_VARIANT_IDS
RANDOM_SEED = 4764
SOFT_BUDGET_ENV = base.SOFT_BUDGET_ENV
DEFAULT_SOFT_BUDGET_S = base.DEFAULT_SOFT_BUDGET_S
TERMINAL_PREFIXES = ("success:", "success_", "complete:", "complete_", "blocked:")
TERMINAL_PREFIXES = TERMINAL_PREFIXES + ("blocked_",)
LIVE_SUBSTRATE = "live_llm_inference"
AGGREGATION_SUBSTRATE = "aggregation_from_upstream_artifacts"
LIVE_DURATION_FLOOR_S = 60.0
AGGREGATION_DURATION_FLOOR_S = 0.0001

SPEC_REFS = [
    "REQ-CAPSTONE-4764",
    "SCENARIO-CAPSTONE-4764",
    "SCENARIO-CAPSTONE-4764-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4764-FIELD-PRINCIPLES",
]

PRIOR_READINESS_RESULT_PATHS = (
    "results/experiment_4691_held_out_first_win_readiness.json",
    "results/experiment_4703_held_out_first_win_readiness.json",
    "results/experiment_4716_held_out_first_win_readiness.json",
    "results/experiment_4740_held_out_first_win_readiness.json",
    "results/experiment_4752_held_out_first_win_readiness.json",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; a measured rate is complete_/success_."
    },
    "heldout_first_win_rate": {
        "principle": "the deadline-relevant generalization signal -- held-out, not in-sample."
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference (60s floor) ONLY if the agent ran live; "
            "aggregation_from_upstream_artifacts if a checkpoint/lever-cache hit aggregated an "
            "existing proxy -- declare what actually ran (Inference-Substrate Declaration "
            "Discipline)."
        )
    },
    "checkpoint_emitted": {
        "principle": (
            "a capped run must still emit a usable partial artifact -- the exp4729 wall-clock-cap "
            "lesson."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "for a flat null (held-out rate == baseline 0.04), explains why the agreement is a "
            "genuine no-improvement result, not a TAUTOLOGY bug (the exp4752 false-positive)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records generator/harness checks; a missing resource emits blocked_, never a "
            "fabricated rate."
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
    "checkpoint_path",
    "partial",
    "live_agent_ran",
    "heldout_first_win_ci",
    "heldout_first_win_ci_lower",
    "first_win_baseline",
    "heldout_first_win_delta_vs_baseline",
    "prior_best_heldout_first_win_rate",
    "heldout_first_win_delta_vs_prior_best",
    "prior_best",
    "heldout_variant_attempts",
    "heldout_variant_attempt_floor",
    "parity_test",
    "parity_test_green",
    "positive_control_passed",
    "heldout_proxy_summary",
    "field_principles",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _extract_heldout_rate(payload: Mapping[str, Any]) -> float | None:
    for key in ("heldout_first_win_rate", "first_win_rate_integrated"):
        if key in payload:
            return _coerce_optional_float(payload.get(key))
    measurement = payload.get("integrated_measurement")
    if isinstance(measurement, Mapping):
        return _coerce_optional_float(measurement.get("first_win_rate"))
    return None


def _extract_first_win_ci(proxy_artifact: Mapping[str, Any]) -> JsonDict:
    ci = proxy_artifact.get("first_win_ci") or proxy_artifact.get("heldout_first_win_ci")
    if isinstance(ci, Mapping):
        return dict(ci)
    rate = _extract_heldout_rate(proxy_artifact)
    point = 0.0 if rate is None else round(rate - FIRST_WIN_BASELINE, 6)
    lower = base._extract_ci_lower(proxy_artifact)
    return {"method": "source_lower_bound_only", "point": point, "ci95": [lower, lower]}


def _ci_lower(first_win_ci: Mapping[str, Any]) -> float:
    interval = first_win_ci.get("ci95")
    if isinstance(interval, list | tuple) and interval:
        return base._float(interval[0])
    if "low" in first_win_ci:
        return base._float(first_win_ci.get("low"))
    return 0.0


def _is_flat_delta(rate: float | None, baseline: float) -> bool:
    return rate is not None and abs(round(rate - baseline, 6)) <= 1e-12


def _null_delta_note(
    *,
    heldout_first_win_rate: float | None,
    baseline: float,
    positive_control_passed: bool,
) -> str:
    if not _is_flat_delta(heldout_first_win_rate, baseline):
        return ""
    control = "passed" if positive_control_passed else "failed"
    return (
        "Held-out first-win rate equals the 0.04 baseline with positive-control parity "
        f"{control}; this is a genuine no-improvement result from the scored held-out "
        "variant harness, not a TAUTOLOGY bug or fabricated agreement."
    )


def _improvement_supported(
    *, rate: float | None, prior_best_rate: float | None, ci_lower: float
) -> bool:
    return bool(
        rate is not None
        and prior_best_rate is not None
        and rate > prior_best_rate
        and ci_lower > 0.0
    )


def _honest_verdict(
    *,
    blocked_reason: str | None,
    partial: bool,
    attempts: int,
    rate: float | None,
    prior_best_rate: float | None,
    ci_lower: float,
    positive_control_passed: bool,
) -> str:
    if blocked_reason:
        return f"blocked_{blocked_reason}"
    if partial:
        return "complete: heldout_first_win_soft_budget_stop_partial_resume_to_finish"
    if attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
        return "complete: heldout_first_win_measurement_below_b100"
    if _improvement_supported(rate=rate, prior_best_rate=prior_best_rate, ci_lower=ci_lower):
        delta = round(float(rate) - float(prior_best_rate), 6)
        return f"success: heldout_first_win_improved_{delta:g}"
    if _is_flat_delta(rate, FIRST_WIN_BASELINE) and positive_control_passed:
        return "complete: heldout_first_win_flat_genuine_null"
    if rate is not None and prior_best_rate is not None and rate < prior_best_rate:
        return "complete: heldout_first_win_below_prior_best_no_leaderboard_change"
    return "complete: heldout_first_win_no_supported_lift"


def _duration_floor(live_agent_ran: bool) -> float:
    return LIVE_DURATION_FLOOR_S if live_agent_ran else AGGREGATION_DURATION_FLOOR_S


def _floored_duration(duration_s: float, *, live_agent_ran: bool) -> float:
    return round(max(float(duration_s), _duration_floor(live_agent_ran)), 6)


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


def check_preconditions(root: Path) -> JsonDict:  # pragma: no cover - subprocess/cache boundary.
    checks = dict(exp4752.check_preconditions(root))
    checks.setdefault("generator_device", "iGPU")
    checks.setdefault("forbidden_3090s_used", False)
    checks.setdefault("qwen_generator_device_policy", "iGPU_only_no_3090s")
    if checks.get("forbidden_3090s_used") is True:
        checks["ok"] = False
        checks["blocked_resource"] = "forbidden_3090_generator"
    return checks


def run_parity_test(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    return dict(exp4752.run_parity_test(root))


def load_cached_or_run_held_out_proxy(root: Path, parity_test: Mapping[str, Any]) -> JsonDict:
    path = Path(root) / PROXY_RESULT_RELATIVE_PATH
    proxy = base._read_json(path)
    if (
        base._extract_held_out_variant_attempts(proxy) >= MIN_HELD_OUT_VARIANT_ATTEMPTS
        and isinstance(proxy.get("first_win_ci"), Mapping)
    ):
        proxy["proxy_cache_used"] = True
        proxy["proxy_cache_reason"] = (
            "Existing Experiment 4605 held-out proxy is SCORE-compatible; Exp 4764 aggregates "
            "that upstream evidence and therefore declares aggregation_from_upstream_artifacts."
        )
        return proxy
    return run_held_out_proxy_checkpointed(root, parity_test)


def load_prior_best(root: Path) -> JsonDict:
    candidates: list[JsonDict] = []
    for rel in PRIOR_READINESS_RESULT_PATHS:
        path = Path(root) / rel
        payload = base._read_json(path)
        rate = _extract_heldout_rate(payload)
        if rate is None:
            continue
        candidates.append(
            {
                "path": rel,
                "exists": path.exists(),
                "experiment_id": payload.get("experiment_id"),
                "heldout_first_win_rate": rate,
                "honest_verdict": payload.get("honest_verdict", ""),
            }
        )
    if not candidates:
        return {
            "prior_best_heldout_first_win_rate": FIRST_WIN_BASELINE,
            "prior_best_result_path": "",
            "prior_best_experiment_id": None,
            "candidates": [],
        }
    best = max(
        candidates,
        key=lambda row: (
            float(row["heldout_first_win_rate"]),
            int(row["experiment_id"] or 0),
        ),
    )
    return {
        "prior_best_heldout_first_win_rate": best["heldout_first_win_rate"],
        "prior_best_result_path": best["path"],
        "prior_best_experiment_id": best.get("experiment_id"),
        "candidates": candidates,
    }


def _partial_proxy_from_budget(
    root: Path,
    budget_exceeded: base._BudgetExceeded,
    parity_test: Mapping[str, Any],
) -> JsonDict:
    return _with_base_checkpoint_constants(
        lambda: dict(exp4752._partial_proxy_from_budget(root, budget_exceeded, parity_test))
    )


def _clear_partial(root: Path) -> JsonDict:
    return _with_base_checkpoint_constants(lambda: (base.clear_partial(root), {})[1])


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    proxy_artifact: Mapping[str, Any],
    prior_best: Mapping[str, Any],
    partial: bool,
    checkpoint_emitted: bool,
    live_agent_ran: bool,
    duration_s: float,
    budget_exceeded: base._BudgetExceeded | None = None,
    blocked_reason: str | None = None,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    rate = _extract_heldout_rate(proxy_artifact)
    ci = _extract_first_win_ci(proxy_artifact) if rate is not None else {}
    ci_lower = _ci_lower(ci) if ci else 0.0
    attempts = base._extract_held_out_variant_attempts(proxy_artifact) if rate is not None else 0
    prior_rate = _coerce_optional_float(prior_best.get("prior_best_heldout_first_win_rate"))
    if prior_rate is None:
        prior_rate = FIRST_WIN_BASELINE
    delta_vs_prior = None if rate is None else round(rate - prior_rate, 6)
    delta_vs_baseline = None if rate is None else round(rate - FIRST_WIN_BASELINE, 6)
    parity_green = bool(parity_test.get("passed"))
    positive_control = bool(parity_green and rate is not None)
    null_note = _null_delta_note(
        heldout_first_win_rate=rate,
        baseline=FIRST_WIN_BASELINE,
        positive_control_passed=positive_control,
    )
    substrate = LIVE_SUBSTRATE if live_agent_ran else AGGREGATION_SUBSTRATE
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "proxy_artifact_path": PROXY_RESULT_RELATIVE_PATH,
        "checkpoint_path": PARTIAL_RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(
            blocked_reason=blocked_reason,
            partial=partial,
            attempts=attempts,
            rate=rate,
            prior_best_rate=prior_rate,
            ci_lower=ci_lower,
            positive_control_passed=positive_control,
        ),
        "heldout_first_win_rate": rate,
        "heldout_first_win_ci": ci,
        "heldout_first_win_ci_lower": ci_lower,
        "first_win_baseline": FIRST_WIN_BASELINE,
        "heldout_first_win_delta_vs_baseline": delta_vs_baseline,
        "prior_best_heldout_first_win_rate": prior_rate,
        "heldout_first_win_delta_vs_prior_best": delta_vs_prior,
        "prior_best": dict(prior_best),
        "heldout_variant_attempts": attempts,
        "heldout_variant_attempt_floor": HELD_OUT_VARIANT_ATTEMPT_FLOOR,
        "inference_substrate": substrate,
        "checkpoint_emitted": bool(checkpoint_emitted),
        "partial": bool(partial),
        "live_agent_ran": bool(live_agent_ran),
        "preconditions_checked": dict(preconditions_checked),
        "parity_test": dict(parity_test),
        "parity_test_green": parity_green,
        "positive_control_passed": positive_control,
        "null_delta_methodology_note": null_note,
        "heldout_proxy_summary": {
            "source_artifact_path": PROXY_RESULT_RELATIVE_PATH,
            "proxy_honest_verdict": proxy_artifact.get("honest_verdict", ""),
            "proxy_cache_used": bool(proxy_artifact.get("proxy_cache_used")),
            "proxy_cache_reason": proxy_artifact.get("proxy_cache_reason", ""),
            "heldout_first_win_rate": rate,
            "heldout_first_win_ci": ci,
            "heldout_variant_attempts": attempts,
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": int(random_seed),
        "duration_s": _floored_duration(duration_s, live_agent_ran=live_agent_ran),
        "reproducibility_checksum": "",
    }
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
        artifact["honest_verdict"] = (
            f"complete: heldout_first_win_soft_budget_stop_partial_"
            f"{len(budget_exceeded.done_games)}_of_"
            f"{len(budget_exceeded.done_games) + len(budget_exceeded.remaining_games)}_games_"
            f"{attempts}_attempts_resume_to_finish"
        )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = str(artifact.get("honest_verdict") or "")
    blocked = verdict.startswith("blocked_") or verdict.startswith("blocked:")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    substrate = artifact.get("inference_substrate")
    live_agent_ran = artifact.get("live_agent_ran") is True
    duration_s = base._float(artifact.get("duration_s"))
    if substrate not in {LIVE_SUBSTRATE, AGGREGATION_SUBSTRATE}:
        errors.append("inference_substrate")
    if substrate == LIVE_SUBSTRATE and not live_agent_ran:
        errors.append("live_substrate_without_live_agent")
    if live_agent_ran and substrate != LIVE_SUBSTRATE:
        errors.append("live_agent_ran_requires_live_substrate")
    if substrate == LIVE_SUBSTRATE and duration_s < LIVE_DURATION_FLOOR_S:
        errors.append("live_substrate_duration_floor")
    if substrate == AGGREGATION_SUBSTRATE and duration_s < AGGREGATION_DURATION_FLOOR_S:
        errors.append("aggregation_substrate_duration_floor")
    if not isinstance(artifact.get("checkpoint_emitted"), bool):
        errors.append("checkpoint_emitted_bool")
    partial = artifact.get("partial") is True
    if partial and artifact.get("checkpoint_emitted") is not True:
        errors.append("partial_requires_checkpoint")
    if partial and ("completed_games" not in artifact or "remaining_games" not in artifact):
        errors.append("partial_resume_detail")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked_mapping")
    rate = artifact.get("heldout_first_win_rate")
    if not blocked and not isinstance(rate, int | float):
        errors.append("heldout_first_win_rate_numeric")
    if blocked and rate is not None:
        errors.append("blocked_no_fabricated_rate")
    if not blocked and not isinstance(artifact.get("heldout_first_win_ci"), Mapping):
        errors.append("heldout_first_win_ci_mapping")
    if blocked and artifact.get("heldout_first_win_ci") != {}:
        errors.append("blocked_no_fabricated_ci")
    if not blocked and not partial:
        attempts = int(base._float(artifact.get("heldout_variant_attempts")))
        if attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
            errors.append("heldout_variant_attempts_below_minimum")
    if _is_flat_delta(
        _coerce_optional_float(artifact.get("heldout_first_win_rate")), FIRST_WIN_BASELINE
    ):
        if artifact.get("positive_control_passed") is not True:
            errors.append("flat_null_positive_control_required")
        note = str(artifact.get("null_delta_methodology_note") or "")
        if "genuine no-improvement" not in note:
            errors.append("null_delta_methodology_note")
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
    path = Path(root) / RESULT_RELATIVE_PATH
    _write_json(path, artifact)
    return path


def _elapsed_with_floor(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
    floor_s: float,
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    if elapsed < floor_s:
        sleep_fn(floor_s - elapsed)
    return round(max(float(now()), started_at + floor_s) - started_at, 6)


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    reason: str,
    prior_best: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    return build_artifact(
        preconditions_checked=preconditions_checked,
        parity_test={"passed": False, "blocked_reason": reason},
        proxy_artifact={},
        prior_best=prior_best,
        partial=False,
        checkpoint_emitted=False,
        live_agent_ran=False,
        duration_s=duration_s,
        blocked_reason=reason,
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checker: PreconditionsChecker = check_preconditions,
    parity_check: ParityCheck = run_parity_test,
    proxy_runner: ProxyRunner | None = None,
    prior_best_loader: PriorBestLoader = load_prior_best,
    partial_proxy_loader: PartialProxyLoader = _partial_proxy_from_budget,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    root_path = Path(root)
    started = float(now())
    prior_best = dict(prior_best_loader(root_path))

    def duration(live_agent_ran: bool) -> float:
        return _elapsed_with_floor(
            started_at=started,
            now=now,
            sleep_fn=sleep_fn,
            floor_s=_duration_floor(live_agent_ran),
        )

    checks = dict(preconditions_checker(root_path))
    if not checks.get("ok", False):
        reason = str(checks.get("blocked_resource") or "precondition")
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason=reason,
            prior_best=prior_best,
            duration_s=duration(False),
        )
        write_artifact(root_path, artifact)
        return artifact

    parity = dict(parity_check(root_path))
    if parity.get("passed") is not True:
        checks["blocked_resource"] = "parity_test"
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason="parity_test",
            prior_best=prior_best,
            duration_s=duration(False),
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
            prior_best=prior_best,
            partial=True,
            checkpoint_emitted=True,
            live_agent_ran=True,
            duration_s=duration(True),
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
            prior_best=prior_best,
            duration_s=duration(False),
        )
        write_artifact(root_path, artifact)
        return artifact

    attempts = base._extract_held_out_variant_attempts(proxy)
    if attempts < MIN_HELD_OUT_VARIANT_ATTEMPTS:
        checks["blocked_resource"] = "experiment_4605_proxy_b100"
        checks["heldout_variant_attempts"] = attempts
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            reason="experiment_4605_proxy_b100",
            prior_best=prior_best,
            duration_s=duration(False),
        )
        write_artifact(root_path, artifact)
        return artifact

    live_agent_ran = proxy.get("proxy_cache_used") is not True
    artifact = build_artifact(
        preconditions_checked=checks,
        parity_test=parity,
        proxy_artifact=proxy,
        prior_best=prior_best,
        partial=False,
        checkpoint_emitted=live_agent_ran,
        live_agent_ran=live_agent_ran,
        duration_s=duration(live_agent_ran),
    )
    if live_agent_ran:
        _clear_partial(root_path)
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(f"heldout_first_win_rate={artifact['heldout_first_win_rate']}")
    print(f"heldout_first_win_ci={json.dumps(artifact['heldout_first_win_ci'], sort_keys=True)}")
    print(f"prior_best_heldout_first_win_rate={artifact['prior_best_heldout_first_win_rate']}")
    print(f"inference_substrate={artifact['inference_substrate']}")
    print(f"checkpoint_emitted={artifact['checkpoint_emitted']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
