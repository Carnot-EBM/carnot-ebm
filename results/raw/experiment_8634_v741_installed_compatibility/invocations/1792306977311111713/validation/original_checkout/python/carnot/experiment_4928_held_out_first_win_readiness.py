"""Experiment 4928: final resume-live held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4928, SCENARIO-CAPSTONE-4928,
SCENARIO-CAPSTONE-4928-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4928-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot import experiment_4917_held_out_first_win_readiness as previous


JsonDict = dict[str, Any]
PreconditionsChecker = previous.PreconditionsChecker
ParityCheck = previous.ParityCheck
ProxyRunner = previous.ProxyRunner
PriorBestLoader = previous.PriorBestLoader
PartialProxyLoader = previous.PartialProxyLoader
A1PriorLoader = previous.A1PriorLoader

base = previous.base
REPO_ROOT = previous.REPO_ROOT
PYTHON_ROOT = REPO_ROOT / "python"
for _path in (REPO_ROOT, PYTHON_ROOT):  # pragma: no cover - direct script import guard.
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
EXPERIMENT = "experiment_4928_heldout_first_win_readiness"
EXPERIMENT_ID = 4928
SCHEMA = "carnot.arc.heldout_first_win_readiness_4928.v1"
RESULT_RELATIVE_PATH = "results/experiment_4928_heldout_first_win_readiness.json"
PARTIAL_RESULT_RELATIVE_PATH = "results/experiment_4928_heldout_first_win_readiness.partial.json"
SOURCE_EXPERIMENT_ID = previous.EXPERIMENT_ID
SOURCE_RESULT_RELATIVE_PATH = previous.RESULT_RELATIVE_PATH
SOURCE_PARTIAL_RESULT_RELATIVE_PATH = previous.PARTIAL_RESULT_RELATIVE_PATH
PROXY_RESULT_RELATIVE_PATH = previous.PROXY_RESULT_RELATIVE_PATH
A1_PRIOR_RESULT_RELATIVE_PATH = previous.A1_PRIOR_RESULT_RELATIVE_PATH
FIRST_WIN_BASELINE = previous.FIRST_WIN_BASELINE
MIN_HELD_OUT_VARIANT_ATTEMPTS = previous.MIN_HELD_OUT_VARIANT_ATTEMPTS
HELD_OUT_VARIANT_ATTEMPT_FLOOR = previous.HELD_OUT_VARIANT_ATTEMPT_FLOOR
HELD_OUT_VARIANT_IDS = previous.HELD_OUT_VARIANT_IDS
TARGET_GAMES = 25
RANDOM_SEED = 4928
SOFT_BUDGET_ENV = previous.SOFT_BUDGET_ENV
DEFAULT_SOFT_BUDGET_S = previous.DEFAULT_SOFT_BUDGET_S
TERMINAL_PREFIXES = previous.TERMINAL_PREFIXES
LIVE_SUBSTRATE = previous.LIVE_SUBSTRATE
AGGREGATION_SUBSTRATE = previous.AGGREGATION_SUBSTRATE
LIVE_DURATION_FLOOR_S = previous.LIVE_DURATION_FLOOR_S
AGGREGATION_DURATION_FLOOR_S = previous.AGGREGATION_DURATION_FLOOR_S
SOLVE_PROVENANCE = previous.SOLVE_PROVENANCE
GENERATOR_BACKENDS = previous.GENERATOR_BACKENDS

SPEC_REFS = [
    "REQ-CAPSTONE-4928",
    "SCENARIO-CAPSTONE-4928",
    "SCENARIO-CAPSTONE-4928-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4928-FIELD-PRINCIPLES",
]

PRIOR_READINESS_RESULT_PATHS = previous.PRIOR_READINESS_RESULT_PATHS + (
    SOURCE_RESULT_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a clean countable null is "
            "complete_heldout_first_win_<rate>_full25_live_flag_resolved; a lift is "
            "success_heldout_first_win_<rate>."
        )
    },
    "heldout_first_win_rate": {
        "principle": (
            "the live held-out first-win rate (full 25 games if complete) -- the 6/30 "
            "go/no-go number."
        )
    },
    "heldout_first_win_ci": {
        "principle": "bootstrap CI of the rate; a CI-lower-0 result is an honest null, not a failure."
    },
    "games_evaluated": {
        "principle": (
            "the count of games scored (target 25); a partial records games_remaining for "
            "the next resume."
        )
    },
    "live_agent_ran": {
        "principle": (
            "true -- a LIVE resume (not resume-from-cache-only), the precondition for a "
            "countable readiness number."
        )
    },
    "flag_resolved": {
        "principle": (
            "true iff the fresh fully-stamped artifact is NOT flagged true_live_recheck=critical -- "
            "the recurring-flag fix holds at full-25."
        )
    },
    "triggering_rule_if_flagged": {
        "principle": (
            "if still flagged, the exact summarize_artifact/adversarial_verify rule that "
            "fires plus the minimal documented source fix."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "for a flat null, why the ~0.04-agreement is genuine no-improvement, not a "
            "TAUTOLOGY/fabrication."
        )
    },
    "positive_control_passed": {
        "principle": (
            "true -- a non-degenerate positive control distinguishes a real null from a "
            "broken harness."
        )
    },
    "model_specs": {
        "principle": (
            "Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server -- the methodology stamp whose "
            "absence caused the historical flag."
        )
    },
    "random_seed": {
        "principle": "determinism plus part of the methodology stamp the live-recheck requires."
    },
    "generator_backend": {
        "principle": "gpu0_cuda | igpu_hip -- proves the GPU fix and a genuine live run."
    },
    "inference_substrate": {
        "principle": "live_llm_inference (60s floor)."
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- a readiness measurement on the variant harness, NOT a "
            "registry bank."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/variant-harness/.453-ledger/generator checks; a missing resource "
            "emits blocked_."
        )
    },
    "games_remaining": {
        "principle": "remaining target games after this resume; zero for the countable full-25 result."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(
    dict.fromkeys(tuple(FIELD_PRINCIPLES) + tuple(previous.REQUIRED_ARTIFACT_FIELDS))
)

_PATCHED_PREVIOUS_CONSTANTS: dict[str, Any] = {
    "EXPERIMENT": EXPERIMENT,
    "EXPERIMENT_ID": EXPERIMENT_ID,
    "SCHEMA": SCHEMA,
    "RESULT_RELATIVE_PATH": RESULT_RELATIVE_PATH,
    "PARTIAL_RESULT_RELATIVE_PATH": PARTIAL_RESULT_RELATIVE_PATH,
    "RANDOM_SEED": RANDOM_SEED,
    "SPEC_REFS": SPEC_REFS,
    "PRIOR_READINESS_RESULT_PATHS": PRIOR_READINESS_RESULT_PATHS,
    "FIELD_PRINCIPLES": FIELD_PRINCIPLES,
}


@contextmanager
def _patched_previous_constants() -> Iterator[None]:
    old_constants = {name: getattr(previous, name) for name in _PATCHED_PREVIOUS_CONSTANTS}
    old_patched_previous_constants = previous._PATCHED_PREVIOUS_CONSTANTS
    try:
        for name, value in _PATCHED_PREVIOUS_CONSTANTS.items():
            setattr(previous, name, value)
        previous._PATCHED_PREVIOUS_CONSTANTS = dict(_PATCHED_PREVIOUS_CONSTANTS)
        yield
    finally:
        previous._PATCHED_PREVIOUS_CONSTANTS = old_patched_previous_constants
        for name, value in old_constants.items():
            setattr(previous, name, value)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    with _patched_previous_constants():
        return previous.payload_checksum(artifact)


def _rate_label(value: Any) -> str:
    return previous._rate_label(value)


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _checkpoint_game_count(payload: Mapping[str, Any]) -> int:
    games = payload.get("games")
    return len(games) if isinstance(games, Mapping) else 0


def ensure_453_resume_checkpoint(root: Path | str = REPO_ROOT) -> JsonDict:
    """Make Exp4928 resume from Exp4917's 21/25 ledger when no 4928 ledger exists."""

    root_path = Path(root)
    source = root_path / SOURCE_PARTIAL_RESULT_RELATIVE_PATH
    destination = root_path / PARTIAL_RESULT_RELATIVE_PATH
    source_payload = _read_json(source)
    destination_payload = _read_json(destination)
    destination_exists_before = destination.exists()
    source_games = _checkpoint_game_count(source_payload)
    destination_games_before = _checkpoint_game_count(destination_payload)
    copied = False

    if not destination_exists_before and source_games > 0:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        copied = True
        destination_payload = source_payload

    destination_games_after = _checkpoint_game_count(destination_payload)
    ok = destination_games_after > 0
    return {
        "ok": ok,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_path": SOURCE_PARTIAL_RESULT_RELATIVE_PATH,
        "destination_path": PARTIAL_RESULT_RELATIVE_PATH,
        "source_exists": source.exists(),
        "destination_exists_before": destination_exists_before,
        "destination_exists_after": destination.exists(),
        "source_games": source_games,
        "destination_games_before": destination_games_before,
        "destination_games_after": destination_games_after,
        "copied_from_source": copied,
        "expected_source_checkpoint": "experiment_4917_21_of_25_games",
    }


def check_preconditions(root: Path) -> JsonDict:  # pragma: no cover - subprocess/cache boundary.
    resume = ensure_453_resume_checkpoint(root)
    if not resume.get("ok"):
        return {
            "ok": False,
            "blocked_resource": "experiment_4917_checkpoint_ledger",
            "experiment_4917_checkpoint_ledger": resume,
        }
    with _patched_previous_constants():
        checks = dict(previous.check_preconditions(root))
    checks["experiment_4917_checkpoint_ledger"] = resume
    checks["experiment_4917_21of25_ledger_present"] = bool(
        resume.get("source_games", 0) >= 21 or resume.get("destination_games_after", 0) >= 21
    )
    if checks.get("ok", False) and not checks["experiment_4917_21of25_ledger_present"]:
        checks["ok"] = False
        checks["blocked_resource"] = "experiment_4917_checkpoint_ledger"
    return checks


def _games_evaluated(artifact: Mapping[str, Any]) -> int:
    completed_games = artifact.get("completed_games")
    if isinstance(completed_games, list):
        return min(TARGET_GAMES, len(completed_games))
    attempts = int(base._float(artifact.get("heldout_variant_attempts")))
    if attempts <= 0:
        return 0
    return min(TARGET_GAMES, attempts // len(HELD_OUT_VARIANT_IDS))


def _games_remaining(artifact: Mapping[str, Any], games_evaluated: int) -> int:
    remaining_games = artifact.get("remaining_games")
    if isinstance(remaining_games, list):
        return max(0, len(remaining_games))
    return max(0, TARGET_GAMES - games_evaluated)


def _normalise_4928_verdict(artifact: Mapping[str, Any]) -> str:
    verdict = str(artifact.get("honest_verdict") or "")
    if not artifact.get("live_agent_ran") or verdict.startswith(("blocked_", "blocked:")):
        return verdict
    if artifact.get("partial") is True or "soft_budget_stop_partial" in verdict:
        return verdict
    if verdict.startswith("success_heldout_first_win_"):
        return verdict
    rate = _rate_label(artifact.get("heldout_first_win_rate"))
    if artifact.get("flag_resolved") is True:
        return f"complete_heldout_first_win_{rate}_full25_live_flag_resolved"
    return f"complete_heldout_first_win_{rate}_full25_live_flagged_recheck"


def _augment_4928_fields(artifact: Mapping[str, Any]) -> JsonDict:
    out = dict(artifact)
    checks = dict(out.get("preconditions_checked") or {})
    if "experiment_4917_checkpoint_ledger" not in checks:
        checks["experiment_4917_checkpoint_ledger"] = ensure_453_resume_checkpoint(REPO_ROOT)
    evaluated = _games_evaluated(out)
    out.update(
        {
            "experiment": EXPERIMENT,
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "spec_refs": list(SPEC_REFS),
            "result_path": RESULT_RELATIVE_PATH,
            "checkpoint_path": PARTIAL_RESULT_RELATIVE_PATH,
            "games_evaluated": evaluated,
            "games_remaining": _games_remaining(out, evaluated),
            "field_principles": dict(FIELD_PRINCIPLES),
            "preconditions_checked": checks,
            "random_seed": RANDOM_SEED,
        }
    )
    out["honest_verdict"] = _normalise_4928_verdict(out)
    out["reproducibility_checksum"] = payload_checksum(out)
    return out


def _critical_flags(path: Path) -> list[JsonDict]:
    return previous._critical_flags(path)


def _triggering_rule(flags: Sequence[Mapping[str, Any]]) -> str:
    return previous._triggering_rule(flags)


def _record_live_recheck(root: Path, artifact: Mapping[str, Any]) -> JsonDict:
    out = dict(artifact)
    if out.get("live_agent_ran") is not True:
        out["flag_resolved"] = False
        out["triggering_rule_if_flagged"] = "not_evaluated_non_live_or_blocked"
        out["flagged_adversarial"] = False
        out["honest_verdict"] = _normalise_4928_verdict(out)
        out["reproducibility_checksum"] = payload_checksum(out)
        return out

    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        out["reproducibility_checksum"] = payload_checksum(out)
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        flags = _critical_flags(path)
        out["flag_resolved"] = not flags
        out["triggering_rule_if_flagged"] = _triggering_rule(flags)
        out["flagged_adversarial"] = bool(flags)
        out["honest_verdict"] = _normalise_4928_verdict(out)
    out["reproducibility_checksum"] = payload_checksum(out)
    return out


def _interrupted_live_artifact(root: Path) -> JsonDict:
    artifact = _read_json(root / RESULT_RELATIVE_PATH)
    if (
        artifact.get("experiment_id") == EXPERIMENT_ID
        and artifact.get("live_agent_ran") is True
        and artifact.get("partial") is False
        and int(base._float(artifact.get("heldout_variant_attempts"))) >= MIN_HELD_OUT_VARIANT_ATTEMPTS
        and artifact.get("triggering_rule_if_flagged") == "pending_live_recheck"
    ):
        return artifact
    return {}


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    proxy_artifact: Mapping[str, Any],
    prior_best: Mapping[str, Any],
    a1_prior_decision: Mapping[str, Any],
    partial: bool,
    checkpoint_emitted: bool,
    live_agent_ran: bool,
    duration_s: float,
    budget_exceeded: base._BudgetExceeded | None = None,
    blocked_reason: str | None = None,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    with _patched_previous_constants():
        artifact = dict(
            previous.build_artifact(
                preconditions_checked=preconditions_checked,
                parity_test=parity_test,
                proxy_artifact=proxy_artifact,
                prior_best=prior_best,
                a1_prior_decision=a1_prior_decision,
                partial=partial,
                checkpoint_emitted=checkpoint_emitted,
                live_agent_ran=live_agent_ran,
                duration_s=duration_s,
                budget_exceeded=budget_exceeded,
                blocked_reason=blocked_reason,
                random_seed=random_seed,
            )
        )
    return _augment_4928_fields(artifact)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    with _patched_previous_constants():
        errors.extend(previous.artifact_schema_errors(artifact))

    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    evaluated = artifact.get("games_evaluated")
    remaining = artifact.get("games_remaining")
    if not isinstance(evaluated, int) or not 0 <= evaluated <= TARGET_GAMES:
        errors.append("games_evaluated")
    if not isinstance(remaining, int) or not 0 <= remaining <= TARGET_GAMES:
        errors.append("games_remaining")
    if isinstance(evaluated, int) and isinstance(remaining, int):
        if artifact.get("partial") is not True and artifact.get("live_agent_ran") is True:
            if evaluated != TARGET_GAMES or remaining != 0:
                errors.append("full25_games_accounting")
        if artifact.get("partial") is True and evaluated + remaining != TARGET_GAMES:
            errors.append("partial_games_accounting")
    verdict = str(artifact.get("honest_verdict") or "")
    if (
        artifact.get("live_agent_ran") is True
        and artifact.get("partial") is not True
        and artifact.get("flag_resolved") is True
        and "_full25_live_flag_resolved" not in verdict
        and not verdict.startswith("success_heldout_first_win_")
    ):
        errors.append("honest_verdict_full25_flag_resolved")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(dict.fromkeys(errors))


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_held_out_proxy_checkpointed(
    root: Path,
    parity_test: Mapping[str, Any],
    *,
    now: Callable[[], float] = time.time,
    soft_budget_s: float | None = None,
    public_games: Sequence[str] | None = None,
) -> JsonDict:
    with _patched_previous_constants():
        return dict(
            previous.run_held_out_proxy_checkpointed(
                root,
                parity_test,
                now=now,
                soft_budget_s=soft_budget_s,
                public_games=public_games,
            )
        )


def run_parity_test(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary.
    with _patched_previous_constants():
        return dict(previous.run_parity_test(root))


def load_prior_best(root: Path) -> JsonDict:
    with _patched_previous_constants():
        return dict(previous.load_prior_best(root))


def _partial_proxy_from_budget(
    root: Path,
    budget_exceeded: base._BudgetExceeded,
    parity_test: Mapping[str, Any],
) -> JsonDict:
    with _patched_previous_constants():
        return dict(previous._partial_proxy_from_budget(root, budget_exceeded, parity_test))


def load_a1_amortized_prior_decision(root: Path | str = REPO_ROOT) -> JsonDict:
    return dict(previous.load_a1_amortized_prior_decision(root))


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checker: PreconditionsChecker = check_preconditions,
    parity_check: ParityCheck = run_parity_test,
    proxy_runner: ProxyRunner | None = None,
    prior_best_loader: PriorBestLoader = load_prior_best,
    partial_proxy_loader: PartialProxyLoader = _partial_proxy_from_budget,
    a1_prior_loader: A1PriorLoader = load_a1_amortized_prior_decision,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    root_path = Path(root)
    interrupted = _interrupted_live_artifact(root_path)
    if interrupted:
        checks = dict(interrupted.get("preconditions_checked") or {})
        checks["finalized_existing_live_artifact_after_import_path_fix"] = True
        interrupted["preconditions_checked"] = checks
        artifact = _augment_4928_fields(interrupted)
        artifact = _record_live_recheck(root_path, artifact)
        write_artifact(root_path, artifact)
        return artifact

    with _patched_previous_constants():
        artifact = dict(
            previous.run(
                root=root_path,
                preconditions_checker=preconditions_checker,
                parity_check=parity_check,
                proxy_runner=proxy_runner,
                prior_best_loader=prior_best_loader,
                partial_proxy_loader=partial_proxy_loader,
                a1_prior_loader=a1_prior_loader,
                now=now,
                sleep_fn=sleep_fn,
            )
        )
    artifact = _augment_4928_fields(artifact)
    artifact = _record_live_recheck(root_path, artifact)
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(f"heldout_first_win_rate={artifact['heldout_first_win_rate']}")
    print(f"heldout_first_win_ci={json.dumps(artifact['heldout_first_win_ci'], sort_keys=True)}")
    print(f"heldout_first_win_delta_vs_baseline={artifact['heldout_first_win_delta_vs_baseline']}")
    print(f"prior_best_heldout_first_win_rate={artifact['prior_best_heldout_first_win_rate']}")
    print(f"heldout_first_win_delta_vs_prior_best={artifact['heldout_first_win_delta_vs_prior_best']}")
    print(f"games_evaluated={artifact['games_evaluated']}")
    print(f"games_remaining={artifact['games_remaining']}")
    print(f"generator_backend={artifact['generator_backend']}")
    print(f"inference_substrate={artifact['inference_substrate']}")
    print(f"live_agent_ran={artifact['live_agent_ran']}")
    print(f"flag_resolved={artifact['flag_resolved']}")
    print(f"triggering_rule_if_flagged={artifact['triggering_rule_if_flagged']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
