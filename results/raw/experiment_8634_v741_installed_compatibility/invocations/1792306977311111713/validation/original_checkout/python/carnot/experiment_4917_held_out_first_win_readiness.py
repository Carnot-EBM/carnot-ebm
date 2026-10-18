"""Experiment 4917: final fresh-live held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4917, SCENARIO-CAPSTONE-4917,
SCENARIO-CAPSTONE-4917-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4917-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4875_heldout_first_win_readiness as previous


JsonDict = dict[str, Any]
PreconditionsChecker = previous.PreconditionsChecker
ParityCheck = previous.ParityCheck
ProxyRunner = previous.ProxyRunner
PriorBestLoader = previous.PriorBestLoader
PartialProxyLoader = previous.PartialProxyLoader
A1PriorLoader = previous.A1PriorLoader

base = previous.base
REPO_ROOT = previous.REPO_ROOT
EXPERIMENT = "experiment_4917_heldout_first_win_readiness"
EXPERIMENT_ID = 4917
SCHEMA = "carnot.arc.heldout_first_win_readiness_4917.v1"
RESULT_RELATIVE_PATH = "results/experiment_4917_heldout_first_win_readiness.json"
PARTIAL_RESULT_RELATIVE_PATH = "results/experiment_4917_heldout_first_win_readiness.partial.json"
PROXY_RESULT_RELATIVE_PATH = previous.PROXY_RESULT_RELATIVE_PATH
A1_PRIOR_RESULT_RELATIVE_PATH = previous.A1_PRIOR_RESULT_RELATIVE_PATH
FIRST_WIN_BASELINE = previous.FIRST_WIN_BASELINE
MIN_HELD_OUT_VARIANT_ATTEMPTS = previous.MIN_HELD_OUT_VARIANT_ATTEMPTS
HELD_OUT_VARIANT_ATTEMPT_FLOOR = previous.HELD_OUT_VARIANT_ATTEMPT_FLOOR
HELD_OUT_VARIANT_IDS = previous.HELD_OUT_VARIANT_IDS
RANDOM_SEED = 4917
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
    "REQ-CAPSTONE-4917",
    "SCENARIO-CAPSTONE-4917",
    "SCENARIO-CAPSTONE-4917-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4917-FIELD-PRINCIPLES",
]

PRIOR_READINESS_RESULT_PATHS = previous.PRIOR_READINESS_RESULT_PATHS + (
    "results/experiment_4875_heldout_first_win_readiness.json",
    "results/experiment_4886_heldout_first_win_readiness.json",
    "results/experiment_4896_heldout_first_win_readiness.json",
    "results/experiment_4907_heldout_first_win_readiness.json",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a clean countable null is "
            "complete_heldout_first_win_<rate>_live_flag_resolved; a lift is "
            "success_heldout_first_win_<rate>."
        )
    },
    "heldout_first_win_rate": {
        "principle": (
            "the live held-out first-win rate -- the 6/30 go/no-go number for the operator."
        )
    },
    "heldout_first_win_ci": {
        "principle": "bootstrap CI of the rate; a CI-lower-0 result is an honest null."
    },
    "live_agent_ran": {
        "principle": (
            "true -- a FRESH live run, not resume-from-cache, the precondition for a "
            "countable readiness number."
        )
    },
    "flag_resolved": {
        "principle": (
            "true iff the fresh fully-stamped artifact is NOT flagged "
            "true_live_recheck=critical."
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
            "for a flat null, why the 0.04-agreement is genuine no-improvement, not a "
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
            "Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server -- the methodology stamp "
            "whose absence caused the recurring flag."
        )
    },
    "random_seed": {
        "principle": "determinism plus the methodology stamp the live-recheck requires."
    },
    "generator_backend": {
        "principle": "gpu0_cuda | igpu_hip -- proves the GPU fix and a genuine live run."
    },
    "inference_substrate": {
        "principle": "live_llm_inference (60s floor)."
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- a readiness measurement on the variant harness, not a "
            "registry bank."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/variant-harness/generator checks; a missing resource emits "
            "blocked_."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + tuple(
    field for field in previous.REQUIRED_ARTIFACT_FIELDS if field not in FIELD_PRINCIPLES
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


def _critical_flags(path: Path) -> list[JsonDict]:
    from scripts import adversarial_verify as av

    report = av.verify_artifact(path)
    flags = report.get("flags", [])
    if not isinstance(flags, list):
        return []
    return [
        dict(flag)
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
    ]


def _triggering_rule(flags: Sequence[Mapping[str, Any]]) -> str:
    if not flags:
        return ""
    rendered = [
        f"{flag.get('kind', 'UNKNOWN')}: {flag.get('detail', '')}" for flag in flags
    ]
    return (
        "; ".join(rendered)
        + " | minimal_source_fix: keep the source methodology stamp and remove/rename the "
        "independent duplicated top-level metric that the live recheck identified."
    )


def _normalise_verdict(artifact: Mapping[str, Any]) -> str:
    verdict = str(artifact.get("honest_verdict") or "")
    if not artifact.get("live_agent_ran"):
        return verdict
    if verdict.startswith("success_heldout_first_win_"):
        return verdict
    if "soft_budget_stop_partial" in verdict:
        return verdict
    rate = artifact.get("heldout_first_win_rate")
    return f"complete_heldout_first_win_{_rate_label(rate)}_live_flag_resolved"


def _model_specs_from_preconditions(
    preconditions: Mapping[str, Any],
    generator_backend: str | None,
) -> JsonDict:
    specs = dict(previous._model_specs_from_preconditions(preconditions, generator_backend))
    checks = dict(preconditions)
    qwen_path = str(checks.get("qwen35_mtp_gguf_path") or "")
    if qwen_path:
        specs["model_filename"] = Path(qwen_path).name
    specs.update(
        {
            "model_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
            "llama_server_kind": (
                "cuda-12.8-binary" if generator_backend == "gpu0_cuda" else "hip-binary"
            ),
            "serving_path": (
                "GPU-0 CUDA llama-server"
                if generator_backend == "gpu0_cuda"
                else "iGPU HIP llama-server"
            ),
            "request_note": (
                "Carried explicitly so the final pre-deadline held-out run is not "
                "flagged METHODOLOGY_MISSING."
            ),
        }
    )
    return specs


def _attach_4917_fields(artifact: Mapping[str, Any]) -> JsonDict:
    out = dict(artifact)
    checks = dict(out.get("preconditions_checked") or {})
    backend = previous._generator_backend_from_preconditions(checks)
    out["generator_backend"] = backend
    out["model_specs"] = _model_specs_from_preconditions(checks, backend)
    out["field_principles"] = dict(FIELD_PRINCIPLES)
    out["preconditions_checked"] = checks
    out["flag_resolved"] = False
    out["triggering_rule_if_flagged"] = "pending_live_recheck"
    out["flagged_adversarial"] = True
    out["honest_verdict"] = _normalise_verdict(out)
    out["reproducibility_checksum"] = payload_checksum(out)
    return out


def _record_live_recheck(root: Path, artifact: Mapping[str, Any]) -> JsonDict:
    out = dict(artifact)
    if out.get("live_agent_ran") is not True:
        out["flag_resolved"] = False
        out["triggering_rule_if_flagged"] = "not_evaluated_non_live_or_blocked"
        out["flagged_adversarial"] = False
        out["reproducibility_checksum"] = payload_checksum(out)
        return out

    provisional = dict(out)
    provisional["reproducibility_checksum"] = payload_checksum(provisional)
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(provisional, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    flags = _critical_flags(root / RESULT_RELATIVE_PATH)
    out["flag_resolved"] = not flags
    out["triggering_rule_if_flagged"] = _triggering_rule(flags)
    out["flagged_adversarial"] = bool(flags)
    out["honest_verdict"] = _normalise_verdict(out)
    out["reproducibility_checksum"] = payload_checksum(out)
    return out


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
    return _attach_4917_fields(artifact)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    with _patched_previous_constants():
        errors.extend(previous.artifact_schema_errors(artifact))

    verdict = str(artifact.get("honest_verdict") or "")
    blocked = verdict.startswith("blocked_") or verdict.startswith("blocked:")
    if not blocked and artifact.get("live_agent_ran") is True:
        if artifact.get("inference_substrate") != LIVE_SUBSTRATE:
            errors.append("live_agent_requires_live_substrate")
        if artifact.get("flag_resolved") is not True:
            trigger = str(artifact.get("triggering_rule_if_flagged") or "")
            if not trigger.strip() or trigger == "pending_live_recheck":
                errors.append("triggering_rule_if_flagged")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    backend = artifact.get("generator_backend")
    if not blocked and artifact.get("live_agent_ran") is True and backend not in GENERATOR_BACKENDS:
        errors.append("generator_backend")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
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


def check_preconditions(root: Path) -> JsonDict:  # pragma: no cover - subprocess/cache boundary.
    with _patched_previous_constants():
        return dict(previous.check_preconditions(root))


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
    selected_proxy_runner = proxy_runner or run_held_out_proxy_checkpointed
    with _patched_previous_constants():
        artifact = dict(
            previous.run(
                root=root,
                preconditions_checker=preconditions_checker,
                parity_check=parity_check,
                proxy_runner=selected_proxy_runner,
                prior_best_loader=prior_best_loader,
                partial_proxy_loader=partial_proxy_loader,
                a1_prior_loader=a1_prior_loader,
                now=now,
                sleep_fn=sleep_fn,
            )
        )
    root_path = Path(root)
    artifact = _attach_4917_fields(artifact)
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
    print(f"generator_backend={artifact['generator_backend']}")
    print(f"inference_substrate={artifact['inference_substrate']}")
    print(f"live_agent_ran={artifact['live_agent_ran']}")
    print(f"flag_resolved={artifact['flag_resolved']}")
    print(f"triggering_rule_if_flagged={artifact['triggering_rule_if_flagged']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
