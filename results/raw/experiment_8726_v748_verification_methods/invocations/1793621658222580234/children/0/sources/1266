"""Experiment 4875: fresh-live held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4875, SCENARIO-CAPSTONE-4875,
SCENARIO-CAPSTONE-4875-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4875-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot import experiment_4752_held_out_first_win_readiness as exp4752
from carnot import experiment_4864_heldout_first_win_readiness as previous


JsonDict = dict[str, Any]
PreconditionsChecker = previous.PreconditionsChecker
ParityCheck = previous.ParityCheck
ProxyRunner = previous.ProxyRunner
PriorBestLoader = previous.PriorBestLoader
PartialProxyLoader = previous.PartialProxyLoader
A1PriorLoader = previous.A1PriorLoader
GeneratorChecker = Callable[[], Mapping[str, Any] | bool]

base = previous.base
REPO_ROOT = previous.REPO_ROOT
EXPERIMENT = "experiment_4875_heldout_first_win_readiness"
EXPERIMENT_ID = 4875
SCHEMA = "carnot.arc.heldout_first_win_readiness_4875.v1"
RESULT_RELATIVE_PATH = "results/experiment_4875_heldout_first_win_readiness.json"
PARTIAL_RESULT_RELATIVE_PATH = "results/experiment_4875_heldout_first_win_readiness.partial.json"
PROXY_RESULT_RELATIVE_PATH = previous.PROXY_RESULT_RELATIVE_PATH
A1_PRIOR_RESULT_RELATIVE_PATH = previous.A1_PRIOR_RESULT_RELATIVE_PATH
FIRST_WIN_BASELINE = previous.FIRST_WIN_BASELINE
MIN_HELD_OUT_VARIANT_ATTEMPTS = previous.MIN_HELD_OUT_VARIANT_ATTEMPTS
HELD_OUT_VARIANT_ATTEMPT_FLOOR = previous.HELD_OUT_VARIANT_ATTEMPT_FLOOR
HELD_OUT_VARIANT_IDS = previous.HELD_OUT_VARIANT_IDS
RANDOM_SEED = 4875
SOFT_BUDGET_ENV = previous.SOFT_BUDGET_ENV
DEFAULT_SOFT_BUDGET_S = previous.DEFAULT_SOFT_BUDGET_S
TERMINAL_PREFIXES = previous.TERMINAL_PREFIXES
LIVE_SUBSTRATE = previous.LIVE_SUBSTRATE
AGGREGATION_SUBSTRATE = previous.AGGREGATION_SUBSTRATE
LIVE_DURATION_FLOOR_S = previous.LIVE_DURATION_FLOOR_S
AGGREGATION_DURATION_FLOOR_S = previous.AGGREGATION_DURATION_FLOOR_S
SOLVE_PROVENANCE = previous.SOLVE_PROVENANCE
GENERATOR_BACKENDS = ("gpu0_cuda", "igpu_hip")

SPEC_REFS = [
    "REQ-CAPSTONE-4875",
    "SCENARIO-CAPSTONE-4875",
    "SCENARIO-CAPSTONE-4875-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4875-FIELD-PRINCIPLES",
]

PRIOR_READINESS_RESULT_PATHS = previous.PRIOR_READINESS_RESULT_PATHS + (
    "results/experiment_4864_heldout_first_win_readiness.json",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a measured rate is complete_heldout_first_win_<rate> "
            "or success_ if it beats prior-best."
        )
    },
    "heldout_first_win_rate": {
        "principle": "the deadline-relevant generalization signal -- held-out, fresh."
    },
    "live_agent_ran": {
        "principle": (
            "true for the .449 fresh live run requirement; false only with an honest "
            "blocking-reason declaration."
        )
    },
    "heldout_first_win_delta_vs_baseline": {
        "principle": "delta vs the 0.04 baseline -- the go/no-go signal."
    },
    "generator_backend": {
        "principle": (
            "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a "
            "genuine live run."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference (60s floor) if live; aggregation_from_upstream_artifacts "
            "only on an honest cache-hit."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "for a flat null (rate==0.04), why the agreement is a genuine "
            "no-improvement, not a TAUTOLOGY bug."
        )
    },
    "positive_control_passed": {
        "principle": "a positive control so a flat null is not a harness artifact."
    },
    "checkpoint_emitted": {
        "principle": (
            "a capped run must still emit a usable partial (the 2026-06-25 wall-clock fix)."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- a held-out first-win proxy measurement, declared honestly "
            "(not a banked level)."
        )
    },
    "preconditions_checked": {
        "principle": "records generator/harness checks; a missing resource emits blocked_."
    },
    "model_specs": {
        "principle": (
            "names Qwen3.5-9B-MTP and the served backend for live_llm_inference methodology."
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


def _optional_float(value: Any) -> float | None:
    return previous._optional_float(value)


def _rate_label(value: Any) -> str:
    return previous._rate_label(value)


def _normalized_honest_verdict(artifact: Mapping[str, Any]) -> str:
    with _patched_previous_constants():
        return previous._normalized_honest_verdict(artifact)


def _ci_low(ci: Mapping[str, Any]) -> float:
    return previous._ci_low(ci)


def _normalise_generator_result(result: Any) -> JsonDict:
    if isinstance(result, Mapping):
        out = dict(result)
        out["ok"] = bool(out.get("ok"))
        backend = out.get("generator_backend") or out.get("backend")
        if backend in GENERATOR_BACKENDS:
            out["generator_backend"] = str(backend)
            out["backend"] = str(backend)
        return out
    return {"ok": bool(result)}


def _selected_generator_backend(
    server: Path | str,
    launch_env: Mapping[str, str] | None,
) -> str | None:
    server_text = str(server)
    launch_cuda = None if launch_env is None else launch_env.get("CUDA_VISIBLE_DEVICES")
    if "build-hip" in server_text:
        return "igpu_hip"
    if launch_cuda in (None, "0"):
        return "gpu0_cuda"
    return None


def _generator_backend_from_preconditions(preconditions: Mapping[str, Any]) -> str | None:
    direct = preconditions.get("generator_backend")
    if direct in GENERATOR_BACKENDS:
        return str(direct)
    generator = preconditions.get("generator")
    if isinstance(generator, Mapping):
        backend = generator.get("generator_backend") or generator.get("backend")
        if backend in GENERATOR_BACKENDS:
            return str(backend)
    return None


def _model_specs_from_preconditions(
    preconditions: Mapping[str, Any],
    generator_backend: str | None,
) -> JsonDict:
    generator = preconditions.get("generator")
    gen = generator if isinstance(generator, Mapping) else {}
    cuda_visible = gen.get("launch_env_cuda_visible_devices")
    if cuda_visible is None:
        cuda_visible = gen.get("ambient_cuda_visible_devices")
    return {
        "name": "Qwen3.5-9B-MTP",
        "repo_substr": "Qwen3.5-9B-MTP",
        "backend": generator_backend,
        "server": gen.get("server"),
        "port": gen.get("port"),
        "cuda_visible_devices": cuda_visible,
        "mtp": True,
        "kv_quant": "q8_0",
        "no_think_prefix": True,
    }


def generator_available(*, proposer: Any | None = None) -> JsonDict:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.experiment_4871_generation_wall_fork_probe_gpu_fixed import make_live_qwen_proposer

    server, launch_env = e3._generator_server_and_env()
    backend = _selected_generator_backend(server, launch_env)
    detail: JsonDict = {
        "server": str(server),
        "launch_env_cuda_visible_devices": (
            None if launch_env is None else launch_env.get("CUDA_VISIBLE_DEVICES")
        ),
        "ambient_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": "Qwen3.5-9B-MTP",
        "igpu_required": False,
        "allowed_backends": list(GENERATOR_BACKENDS),
        "generator_backend": backend,
        "backend": backend,
    }
    if backend not in GENERATOR_BACKENDS:
        return {**detail, "ok": False, "detail": "generator_backend_not_allowed"}
    prop = proposer or make_live_qwen_proposer()
    detail["port"] = getattr(prop, "port", None)
    ensure = getattr(prop, "_ensure_server", None)
    if not callable(ensure):
        return {**detail, "ok": False, "detail": "generator_missing_ensure_server"}
    ok = bool(ensure())
    return {**detail, "ok": ok, "detail": "ok" if ok else "qwen_llama_server_unhealthy"}


def check_preconditions(
    root: Path,
    *,
    qwen_cache_finder: Callable[[], str | None] = exp4752.find_qwen35_mtp_gguf_cache,
    generator_checker: GeneratorChecker = generator_available,
) -> JsonDict:  # pragma: no cover - subprocess/cache/llama boundary, unit-tested via injection.
    checks = dict(exp4752.check_preconditions(root, qwen_cache_finder=qwen_cache_finder))
    checks["gpu_generator_device_policy"] = "igpu_hip_or_gpu0_cuda_no_igpu_pin"
    checks["ambient_cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not checks.get("ok", False):
        return checks

    generator = _normalise_generator_result(generator_checker())
    checks["generator"] = generator
    backend = generator.get("generator_backend")
    checks["generator_backend"] = backend
    checks["generator_device"] = backend or ""
    if generator.get("ok") is not True:
        checks["ok"] = False
        checks["blocked_resource"] = str(generator.get("detail") or "qwen_llama_server")
        return checks
    checks["ok"] = True
    return checks


def load_a1_amortized_prior_decision(root: Path | str = REPO_ROOT) -> JsonDict:
    return dict(previous.load_a1_amortized_prior_decision(root))


def _attach_4875_fields(artifact: Mapping[str, Any]) -> JsonDict:
    out = dict(artifact)
    checks = dict(out.get("preconditions_checked") or {})
    backend = _generator_backend_from_preconditions(checks)
    out["generator_backend"] = backend
    out["model_specs"] = _model_specs_from_preconditions(checks, backend)
    out["field_principles"] = dict(FIELD_PRINCIPLES)
    out["preconditions_checked"] = checks
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
    if proxy_artifact.get("live_blocking_reason"):
        summary = dict(artifact.get("heldout_proxy_summary") or {})
        summary["live_blocking_reason"] = proxy_artifact.get("live_blocking_reason")
        artifact["heldout_proxy_summary"] = summary
    return _attach_4875_fields(artifact)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    with _patched_previous_constants():
        errors.extend(previous.artifact_schema_errors(artifact))

    verdict = str(artifact.get("honest_verdict") or "")
    blocked = verdict.startswith("blocked_") or verdict.startswith("blocked:")
    backend = artifact.get("generator_backend")
    live_agent_ran = artifact.get("live_agent_ran") is True
    if not blocked and live_agent_ran and backend not in GENERATOR_BACKENDS:
        errors.append("generator_backend")
    if not blocked and backend is not None and backend not in GENERATOR_BACKENDS:
        errors.append("generator_backend")

    summary = artifact.get("heldout_proxy_summary")
    proxy_cache_used = isinstance(summary, Mapping) and summary.get("proxy_cache_used") is True
    checks = artifact.get("preconditions_checked")
    live_blocking_reason = ""
    if isinstance(checks, Mapping):
        live_blocking_reason = str(checks.get("live_blocking_reason") or "")
    if isinstance(summary, Mapping) and not live_blocking_reason:
        live_blocking_reason = str(summary.get("live_blocking_reason") or "")
    if (
        proxy_cache_used
        and not live_agent_ran
        and not blocked
        and artifact.get("inference_substrate") == AGGREGATION_SUBSTRATE
        and not live_blocking_reason.strip()
    ):
        errors.append("cache_aggregation_requires_blocking_reason")

    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
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


def load_cached_or_run_held_out_proxy(root: Path, parity_test: Mapping[str, Any]) -> JsonDict:
    with _patched_previous_constants():
        return dict(previous.load_cached_or_run_held_out_proxy(root, parity_test))


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
    artifact = _attach_4875_fields(artifact)
    write_artifact(Path(root), artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(f"heldout_first_win_rate={artifact['heldout_first_win_rate']}")
    print(f"heldout_first_win_ci={json.dumps(artifact['heldout_first_win_ci'], sort_keys=True)}")
    print(f"heldout_first_win_delta_vs_baseline={artifact['heldout_first_win_delta_vs_baseline']}")
    print(f"prior_best_heldout_first_win_rate={artifact['prior_best_heldout_first_win_rate']}")
    print(f"generator_backend={artifact['generator_backend']}")
    print(f"inference_substrate={artifact['inference_substrate']}")
    print(f"live_agent_ran={artifact['live_agent_ran']}")
    print(f"checkpoint_emitted={artifact['checkpoint_emitted']}")
    print(f"solve_provenance={artifact['solve_provenance']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
