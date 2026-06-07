#!/usr/bin/env python3
"""Exp 3905 cost-instrumented verifier harness.

Spec refs: REQ-VERIFY-3905, SCENARIO-VERIFY-3905,
SCENARIO-VERIFY-3905-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.cost_instrumented_verification import (
    build_cost_fixture,
    measure_verification_cost,
    model_params_for_path,
    run_energy_verifier,
    run_llm_judge_verifier,
)


OUTPUT_REL_PATH = Path("results/experiment_3905_cost_instrumented_verify_harness.json")
HARNESS_MODULE_PATH = "python/carnot/verify/cost_instrumented_verification.py"
UNIT_TEST_PATH = "tests/python/test_cost_instrumented_verification.py"
PRIMARY_MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
FALLBACK_MODEL_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
RANDOM_SEED = 3905
INFERENCE_SUBSTRATE = "live_llama_cpp_judge_cpu_forward_plus_cpu_verifier_ensemble"
REQUIRED_FIELDS = {
    "harness_module_path",
    "fixture_cost_ratio",
    "fixture_energy_per_item_ms",
    "fixture_llm_per_item_ms",
    "unit_test_path",
    "unit_test_passed",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "honest_verdict",
}
FIELD_PRINCIPLES = {
    "harness_module_path": "Where exp3906 imports the tested cost harness from.",
    "fixture_cost_ratio": (
        "BARE FLOAT - llm_per_item / energy_per_item on the fixture; must be >1."
    ),
    "fixture_energy_per_item_ms": "BARE FLOAT - energy verifier per-item wall-clock.",
    "fixture_llm_per_item_ms": "BARE FLOAT - LLM judge per-item wall-clock.",
    "unit_test_path": "Where the live timing test is anchored.",
    "unit_test_passed": "BARE BOOL - passing test guards against a fabricated stub.",
    "preconditions_checked": "Pre-launch resource checks before live timing.",
    "model_specs": "SOTA GGUF path and llama.cpp runtime parameters.",
    "random_seed": "Fixed seed for llama.cpp invocation.",
    "reproducibility_checksum": "SHA256 over fixture, model specs, and measured costs.",
    "duration_s": "Real monotonic wall-clock duration; no sleep padding.",
    "inference_substrate": "Declares the exact local inference substrate.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One hard resource checked before the Exp 3905 live fixture run."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"resource": self.resource, "available": self.available, "detail": self.detail}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the Exp 3905 runner."""

    repo_root: Path
    output_path: Path | None = None
    started_monotonic_s: float | None = None
    clock: Callable[[], float] = time.perf_counter
    random_seed: int = RANDOM_SEED
    max_tokens: int = 96
    precondition_timeout_s: int = 60
    run_unit_test: bool = True

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_monotonic_s is None else self.started_monotonic_s

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"


def _checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "blocked_"))


def _probe_command(config: ExperimentConfig, resource: str, code: str) -> PreconditionCheck:
    try:
        proc = subprocess.run(
            [str(config.venv_python()), "-c", code],
            capture_output=True,
            text=True,
            timeout=config.precondition_timeout_s,
            check=False,
        )
    except Exception as exc:
        return PreconditionCheck(resource, False, repr(exc))
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    return PreconditionCheck(resource, proc.returncode == 0, detail)


def _cached_hub_ggufs(hf_id: str) -> list[Path]:
    model_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{hf_id.replace('/', '--')}"
    if not model_dir.is_dir():
        return []
    return [
        path
        for path in sorted(model_dir.rglob("*.gguf"))
        if ".no_exist" not in path.parts
        and "BF16" not in path.parts
        and not path.name.startswith("mmproj-")
        and "imatrix" not in path.name.lower()
        and path.is_file()
        and path.stat().st_size > 0
    ]


def _pick_cached_gguf(hf_id: str, preferred_tokens: Sequence[str]) -> str | None:
    ggufs = _cached_hub_ggufs(hf_id)
    for token in preferred_tokens:
        token_lower = token.lower()
        for path in ggufs:
            if token_lower in path.name.lower():
                return str(path)
    return str(ggufs[0]) if ggufs else None


def _resolve_model() -> tuple[dict[str, object], list[PreconditionCheck]]:
    selected_hf_id = PRIMARY_MODEL_HF_ID
    selected_path = _pick_cached_gguf(PRIMARY_MODEL_HF_ID, ("UD-Q4_K_M", "Q4_K_M"))
    selected_available = (
        selected_path is not None
        and Path(selected_path).is_file()
        and Path(selected_path).stat().st_size > 0
    )
    checks = [
        PreconditionCheck(
            "qwen3_6_35b_a3b_gguf_cached",
            selected_available,
            str(selected_path) if selected_path else "missing; checking gemma fallback",
        )
    ]
    fallback_used = False
    if not selected_available:
        fallback_used = True
        selected_hf_id = FALLBACK_MODEL_HF_ID
        selected_path = _pick_cached_gguf(
            FALLBACK_MODEL_HF_ID,
            ("UD-IQ2_M", "IQ2_M", "UD-IQ2_XXS", "IQ2_XXS", "UD-Q3_K_M", "UD-Q4_K_M"),
        )
        selected_available = (
            selected_path is not None
            and Path(selected_path).is_file()
            and Path(selected_path).stat().st_size > 0
        )
        checks.append(
            PreconditionCheck(
                "gemma_4_26b_a4b_it_gguf_cached",
                selected_available,
                str(selected_path) if selected_path else "missing",
            )
        )

    model_specs = {
        "hf_id": selected_hf_id,
        "model_path": selected_path if selected_available else None,
        "fallback_used": fallback_used,
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": 0,
        "n_ctx": 1024,
        "n_batch": 64,
        "offload_kqv": False,
        "max_tokens": 96,
        "parameter_count_for_flop_estimate": (
            model_params_for_path(selected_path) if selected_available else None
        ),
    }
    return model_specs, checks


def _probe_gguf_tokenizer(model_path: str | None) -> PreconditionCheck:
    if not model_path:
        return PreconditionCheck("gguf_tokenizer_load", False, "model path missing")
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        n_tokens = len(llm.tokenize(b"Exp 3905 tokenizer preflight", add_bos=True))
    except Exception as exc:
        return PreconditionCheck("gguf_tokenizer_load", False, repr(exc))
    return PreconditionCheck("gguf_tokenizer_load", n_tokens > 0, f"tokens={n_tokens}")


def probe_preconditions(
    config: ExperimentConfig,
) -> tuple[str | None, list[PreconditionCheck], dict[str, object]]:
    """Check live resources before loading full GGUF weights."""

    checks = [
        _probe_command(config, "cuda_available", "import torch; assert torch.cuda.is_available()"),
        _probe_command(config, "carnot_verify_import", "import carnot.verify; print('ok')"),
    ]
    model_specs, model_checks = _resolve_model()
    checks.extend(model_checks)
    checks.append(_probe_command(config, "llama_cpp_import", "import llama_cpp; print('ok')"))
    checks.append(_probe_gguf_tokenizer(model_specs.get("model_path")))

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available.get("qwen3_6_35b_a3b_gguf_cached", False) and not available.get(
        "gemma_4_26b_a4b_it_gguf_cached", False
    ):
        blocked_reason = "blocked_model_not_cached"
    elif not available.get("carnot_verify_import", False):
        blocked_reason = "blocked_carnot_verify_import"
    elif not available.get("llama_cpp_import", False):
        blocked_reason = "blocked_llama_cpp_import"
    elif not available.get("gguf_tokenizer_load", False):
        blocked_reason = "blocked_gguf_tokenizer_load"
    return blocked_reason, checks, model_specs


def _cost_ratio(energy_cost: dict[str, object], llm_cost: dict[str, object]) -> float | None:
    energy_ms = float(energy_cost["per_item_wall_ms"])
    llm_ms = float(llm_cost["per_item_wall_ms"])
    if energy_ms <= 0.0:
        return None
    return llm_ms / energy_ms


def _run_cost_fixture(
    *,
    model_specs: dict[str, object],
    config: ExperimentConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    fixture = build_cost_fixture()
    energy_cost = measure_verification_cost(run_energy_verifier, fixture, "energy_verifier")
    model_path = str(model_specs["model_path"])
    llm_cost = measure_verification_cost(
        lambda rows: run_llm_judge_verifier(
            rows,
            model_path=model_path,
            model_params=int(model_specs["parameter_count_for_flop_estimate"]),
            max_tokens=config.max_tokens,
            n_gpu_layers=int(model_specs["n_gpu_layers"]),
            n_ctx=int(model_specs["n_ctx"]),
            n_batch=int(model_specs["n_batch"]),
            offload_kqv=bool(model_specs["offload_kqv"]),
            random_seed=config.random_seed,
        ),
        fixture,
        "llm_judge",
    )
    return energy_cost, llm_cost


def _ready(
    *,
    unit_test_passed: bool,
    ratio: float | None,
    duration_s: float,
) -> bool:
    return unit_test_passed and ratio is not None and ratio > 1.0 and duration_s >= 60.0


def build_artifact(
    *,
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    energy_cost: dict[str, object],
    llm_cost: dict[str, object],
    unit_test_passed: bool,
) -> dict[str, object]:
    """Build the Exp 3905 artifact from measured fixture costs."""

    duration_s = config.clock() - config.start_time()
    ratio = _cost_ratio(energy_cost, llm_cost)
    rendered_ratio = "nan" if ratio is None else f"{ratio:.2f}"
    if _ready(unit_test_passed=unit_test_passed, ratio=ratio, duration_s=duration_s):
        verdict = (
            "complete: "
            f"cost_harness_READY_ratio{rendered_ratio}_efficiency_head_to_head_can_run"
        )
    else:
        verdict = f"complete: cost_harness_NOT_READY_ratio{rendered_ratio}_unit_test{unit_test_passed}"

    checksum_payload = {
        "experiment": 3905,
        "fixture": build_cost_fixture(),
        "model_specs": model_specs,
        "energy_cost": energy_cost,
        "llm_cost": llm_cost,
        "random_seed": config.random_seed,
    }
    artifact: dict[str, object] = {
        "experiment": 3905,
        "title": "cost_instrumented_verify_harness",
        "honest_verdict": verdict,
        "status": verdict,
        "harness_module_path": HARNESS_MODULE_PATH,
        "fixture_cost_ratio": ratio,
        "fixture_energy_per_item_ms": float(energy_cost["per_item_wall_ms"]),
        "fixture_llm_per_item_ms": float(llm_cost["per_item_wall_ms"]),
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": bool(unit_test_passed),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "energy_cost": energy_cost,
        "llm_cost": llm_cost,
        "n_items": int(energy_cost["n_items"]),
        "fixture_label_balance": {
            "gold_error": sum(int(item["gold_error"]) for item in build_cost_fixture()),
            "gold_correct": sum(1 - int(item["gold_error"]) for item in build_cost_fixture()),
        },
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    model_specs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a blocked artifact without fabricated cost measurements."""

    artifact: dict[str, object] = {
        "experiment": 3905,
        "title": "cost_instrumented_verify_harness",
        "honest_verdict": reason,
        "status": reason,
        "harness_module_path": HARNESS_MODULE_PATH,
        "fixture_cost_ratio": None,
        "fixture_energy_per_item_ms": None,
        "fixture_llm_per_item_ms": None,
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": False,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs or {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _checksum(
            {
                "experiment": 3905,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
                "model_specs": model_specs or {},
            }
        ),
        "duration_s": duration_s,
        "inference_substrate": "none_blocked_preflight",
        "field_principles": FIELD_PRINCIPLES,
        "energy_cost": None,
        "llm_cost": None,
        "n_items": 0,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate required Exp 3905 fields and bare-scalar discipline."""

    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not _terminal_verdict(verdict):
        raise ValueError(f"honest_verdict lacks terminal prefix: {verdict}")
    for key in ("harness_module_path", "unit_test_path", "inference_substrate"):
        if not isinstance(artifact[key], str):
            raise ValueError(f"{key} must be a bare string")
    if artifact["fixture_cost_ratio"] is not None and not isinstance(
        artifact["fixture_cost_ratio"],
        float,
    ):
        raise ValueError("fixture_cost_ratio must be a bare float or null")
    for key in ("fixture_energy_per_item_ms", "fixture_llm_per_item_ms"):
        if artifact[key] is not None and not isinstance(artifact[key], float):
            raise ValueError(f"{key} must be a bare float or null")
    if not isinstance(artifact["unit_test_passed"], bool):
        raise ValueError("unit_test_passed must be a bare bool")
    if not isinstance(artifact["duration_s"], (int, float)):
        raise ValueError("duration_s must be a bare number")
    if len(str(artifact["reproducibility_checksum"])) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")


def write_artifact(output_path: Path, artifact: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_unit_test(config: ExperimentConfig, artifact_path: Path) -> bool:
    if not config.run_unit_test:
        return False
    env = os.environ.copy()
    env["CARNOT_EXP3905_LIVE_ARTIFACT"] = str(artifact_path)
    command = [
        str(config.venv_python()),
        "-m",
        "pytest",
        UNIT_TEST_PATH,
        "-q",
        "--no-cov",
        "-n",
        "0",
    ]
    proc = subprocess.run(command, cwd=config.repo_root, env=env, check=False)
    return proc.returncode == 0


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp 3905 end to end, or write a blocked artifact on failed gates."""

    config = config or ExperimentConfig(repo_root=REPO_ROOT)
    started = config.start_time()
    active_config = ExperimentConfig(
        repo_root=config.repo_root,
        output_path=config.output_path,
        started_monotonic_s=started,
        clock=config.clock,
        random_seed=config.random_seed,
        max_tokens=config.max_tokens,
        precondition_timeout_s=config.precondition_timeout_s,
        run_unit_test=config.run_unit_test,
    )
    output_path = active_config.resolved_output_path()
    blocked_reason, checks, model_specs = probe_preconditions(active_config)
    if blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=blocked_reason,
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=model_specs,
        )
        if write:
            write_artifact(output_path, artifact)
        return artifact

    energy_cost, llm_cost = _run_cost_fixture(model_specs=model_specs, config=active_config)
    preliminary = build_artifact(
        config=active_config,
        preconditions_checked=checks,
        model_specs=model_specs,
        energy_cost=energy_cost,
        llm_cost=llm_cost,
        unit_test_passed=False,
    )
    if write:
        write_artifact(output_path, preliminary)

    unit_test_passed = _run_unit_test(active_config, output_path)
    artifact = build_artifact(
        config=active_config,
        preconditions_checked=checks,
        model_specs=model_specs,
        energy_cost=energy_cost,
        llm_cost=llm_cost,
        unit_test_passed=unit_test_passed,
    )
    if write:
        write_artifact(output_path, artifact)
    return artifact


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--no-unit-test", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=args.repo_root,
            output_path=args.output_path,
            run_unit_test=not args.no_unit_test,
        ),
        write=True,
    )
    output_path = args.output_path if args.output_path is not None else args.repo_root / OUTPUT_REL_PATH
    print(f"{output_path.name} wrote {artifact['honest_verdict']}")
    return 0 if str(artifact["honest_verdict"]).startswith("complete:") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
