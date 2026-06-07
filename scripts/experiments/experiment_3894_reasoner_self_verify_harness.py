#!/usr/bin/env python3
"""Exp 3894 reasoner self-verification harness positive control.

Spec refs: REQ-VERIFY-3894, SCENARIO-VERIFY-3894,
SCENARIO-VERIFY-3894-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.reasoner_self_verification import (
    build_positive_control_fixture,
    reasoner_self_verify,
)


OUTPUT_REL_PATH = Path("results/experiment_3894_reasoner_self_verify_harness.json")
HARNESS_MODULE_PATH = "python/carnot/verify/reasoner_self_verification.py"
UNIT_TEST_PATH = "tests/python/test_reasoner_self_verification.py"
PRIMARY_MODEL_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
FALLBACK_MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
RANDOM_SEED = 3894
INFERENCE_SUBSTRATE = "live_llm_inference"
DEGENERACY_ROOT_CAUSE = (
    "Exp3827/Exp3885 reused a brittle YES/NO parser: `if \"no\" in response` "
    "classified errors, while every empty, JSON, yes/correct, or otherwise "
    "unparsed response silently defaulted to correct. Exp3827 also assigned "
    "`duration_s = max(duration, 61.0)`, fabricating the duration floor. That "
    "combination can yield 0 caught errors and AUROC 0.5 without surfacing a "
    "parser failure."
)
REQUIRED_FIELDS = {
    "harness_module_path",
    "fixture_auroc",
    "fixture_n_caught",
    "unit_test_path",
    "unit_test_passed",
    "degeneracy_root_cause",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "honest_verdict",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One hard resource checked before the Exp 3894 live fixture run."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the Exp 3894 runner."""

    repo_root: Path
    output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    random_seed: int = RANDOM_SEED
    max_tokens: int = 96
    cuda_probe_timeout_s: int = 60
    run_unit_test: bool = True

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"


def _checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "blocked_"))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _probe_cuda(config: ExperimentConfig) -> PreconditionCheck:
    command = [
        str(config.venv_python()),
        "-c",
        "import torch; assert torch.cuda.is_available()",
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.cuda_probe_timeout_s,
            check=False,
        )
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, repr(exc))
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    return PreconditionCheck("cuda_available", proc.returncode == 0, detail)


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
        and "imatrix" not in path.name
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
    checks: list[PreconditionCheck] = []
    selected_hf_id = PRIMARY_MODEL_HF_ID
    selected_path = _pick_cached_gguf(
        PRIMARY_MODEL_HF_ID,
        ("UD-IQ2_M", "IQ2_M", "UD-IQ2_XXS", "IQ2_XXS", "UD-Q3_K_M", "UD-Q4_K_M"),
    )
    selected_available = (
        selected_path is not None and Path(selected_path).is_file() and Path(selected_path).stat().st_size > 0
    )
    checks.append(
        PreconditionCheck(
            "gemma_4_26b_a4b_it_gguf_cached",
            selected_available,
            str(selected_path) if selected_path else "missing; checking qwen fallback",
        )
    )
    fallback_used = False
    if not selected_available:
        fallback_used = True
        selected_hf_id = FALLBACK_MODEL_HF_ID
        selected_path = _pick_cached_gguf(FALLBACK_MODEL_HF_ID, ("UD-Q4_K_M", "Q4_K_M"))
        selected_available = (
            selected_path is not None
            and Path(selected_path).is_file()
            and Path(selected_path).stat().st_size > 0
        )
        checks.append(
            PreconditionCheck(
                "qwen3_6_35b_a3b_gguf_cached",
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
    }
    return model_specs, checks


def probe_preconditions(config: ExperimentConfig) -> tuple[str | None, list[PreconditionCheck], dict[str, object]]:
    """Check live resources before loading any GGUF weights."""

    checks = [_probe_cuda(config)]
    model_specs, model_checks = _resolve_model()
    checks.extend(model_checks)
    try:
        __import__("llama_cpp")
        checks.append(PreconditionCheck("llama_cpp_import", True, "import llama_cpp OK"))
    except Exception as exc:
        checks.append(PreconditionCheck("llama_cpp_import", False, repr(exc)))

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not model_specs.get("model_path"):
        blocked_reason = "blocked_model_not_cached"
    elif not available.get("llama_cpp_import", False):
        blocked_reason = "blocked_llama_cpp_not_installed"
    return blocked_reason, checks, model_specs


def _run_unit_test(config: ExperimentConfig) -> bool:
    if not config.run_unit_test:
        return True
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
    proc = subprocess.run(command, cwd=config.repo_root, check=False)
    return proc.returncode == 0


def _fixture_payload() -> list[dict[str, object]]:
    return [dict(item) for item in build_positive_control_fixture()]


def _ready(harness_result: dict[str, object], unit_test_passed: bool) -> bool:
    auroc = harness_result.get("auroc")
    n_caught = int(harness_result.get("n_caught") or 0)
    return (
        unit_test_passed
        and isinstance(auroc, (int, float))
        and float(auroc) > 0.6
        and n_caught > 0
        and harness_result.get("parser_constant_prediction") is False
    )


def build_artifact(
    *,
    harness_result: dict[str, object],
    config: ExperimentConfig,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    unit_test_passed: bool,
) -> dict[str, object]:
    """Build the terminal Exp 3894 artifact from a completed fixture run."""

    started_at = config.start_time()
    finished_at = config.clock()
    fixture = _fixture_payload()
    auroc = harness_result.get("auroc")
    fixture_auroc = float(auroc) if isinstance(auroc, (int, float)) else None
    fixture_n_caught = int(harness_result.get("n_caught") or 0)
    if _ready(harness_result, unit_test_passed):
        verdict = (
            "complete: "
            f"reasoner_self_verify_harness_READY_fixture_auroc{fixture_auroc:.4f}_"
            f"ncaught{fixture_n_caught}_moat_scissor_can_run"
        )
    else:
        rendered_auroc = "nan" if fixture_auroc is None else f"{fixture_auroc:.4f}"
        verdict = (
            "complete: "
            f"reasoner_self_verify_harness_NOT_READY_fixture_auroc{rendered_auroc}_"
            "judge_still_degenerate"
        )
    checksum_payload = {
        "experiment": 3894,
        "fixture": fixture,
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "harness_result": {
            "per_step_pred": harness_result.get("per_step_pred"),
            "per_step_score": harness_result.get("per_step_score"),
            "parsed_count": harness_result.get("parsed_count"),
            "unparsed_count": harness_result.get("unparsed_count"),
            "auroc": fixture_auroc,
            "n_caught": fixture_n_caught,
        },
    }
    artifact: dict[str, object] = {
        "experiment": 3894,
        "title": "reasoner_self_verify_harness",
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "honest_verdict": verdict,
        "status": verdict,
        "harness_module_path": HARNESS_MODULE_PATH,
        "fixture_auroc": fixture_auroc,
        "fixture_n_caught": fixture_n_caught,
        "fixture_n_items": len(fixture),
        "fixture_n_errors": sum(int(item["gold_error"]) for item in fixture),
        "parser_constant_prediction": bool(harness_result.get("parser_constant_prediction")),
        "parsed_count": int(harness_result.get("parsed_count") or 0),
        "unparsed_count": int(harness_result.get("unparsed_count") or 0),
        "per_step_pred": list(harness_result.get("per_step_pred") or []),
        "per_step_score": list(harness_result.get("per_step_score") or []),
        "raw_responses": list(harness_result.get("raw_responses") or []),
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": bool(unit_test_passed),
        "degeneracy_root_cause": DEGENERACY_ROOT_CAUSE,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs,
        "random_seed": config.random_seed,
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": finished_at - started_at,
        "inference_substrate": INFERENCE_SUBSTRATE,
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
    """Build a blocked artifact without fabricated positive-control metrics."""

    artifact: dict[str, object] = {
        "experiment": 3894,
        "title": "reasoner_self_verify_harness",
        "honest_verdict": reason,
        "status": reason,
        "harness_module_path": HARNESS_MODULE_PATH,
        "fixture_auroc": None,
        "fixture_n_caught": 0,
        "fixture_n_items": 0,
        "fixture_n_errors": 0,
        "parser_constant_prediction": True,
        "parsed_count": 0,
        "unparsed_count": 0,
        "per_step_pred": [],
        "per_step_score": [],
        "raw_responses": [],
        "unit_test_path": UNIT_TEST_PATH,
        "unit_test_passed": False,
        "degeneracy_root_cause": DEGENERACY_ROOT_CAUSE,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs or {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _checksum(
            {
                "experiment": 3894,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
                "model_specs": model_specs or {},
            }
        ),
        "duration_s": duration_s,
        "inference_substrate": "none_blocked_preflight",
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate required Exp 3894 fields and terminal-prefix discipline."""

    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not _terminal_verdict(verdict):
        raise ValueError(f"honest_verdict lacks terminal prefix: {verdict}")
    if not isinstance(artifact["harness_module_path"], str):
        raise ValueError("harness_module_path must be a bare string")
    if artifact["fixture_auroc"] is not None and not isinstance(artifact["fixture_auroc"], float):
        raise ValueError("fixture_auroc must be a bare float or null")
    if not isinstance(artifact["fixture_n_caught"], int):
        raise ValueError("fixture_n_caught must be a bare int")
    if not isinstance(artifact["unit_test_passed"], bool):
        raise ValueError("unit_test_passed must be a bare bool")
    if not isinstance(artifact["duration_s"], (int, float)):
        raise ValueError("duration_s must be a bare number")
    if len(str(artifact["reproducibility_checksum"])) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")


def write_artifact(output_path: Path, artifact: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp 3894 end to end, or write a blocked artifact on failed gates."""

    config = config or ExperimentConfig(repo_root=REPO_ROOT)
    started = config.start_time()
    active_config = ExperimentConfig(
        repo_root=config.repo_root,
        output_path=config.output_path,
        started_at=started,
        clock=config.clock,
        random_seed=config.random_seed,
        max_tokens=config.max_tokens,
        cuda_probe_timeout_s=config.cuda_probe_timeout_s,
        run_unit_test=config.run_unit_test,
    )
    blocked_reason, checks, model_specs = probe_preconditions(active_config)
    if blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=blocked_reason,
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=model_specs,
        )
        if write:
            write_artifact(active_config.resolved_output_path(), artifact)
        return artifact

    unit_test_passed = _run_unit_test(active_config)
    fixture = build_positive_control_fixture()
    harness_result = reasoner_self_verify(
        [str(item["step"]) for item in fixture],
        model_path=str(model_specs["model_path"]),
        gold_labels=[int(item["gold_error"]) for item in fixture],
        max_tokens=active_config.max_tokens,
        n_gpu_layers=0,
        offload_kqv=False,
        random_seed=active_config.random_seed,
    )
    artifact = build_artifact(
        harness_result=harness_result,
        config=active_config,
        preconditions_checked=checks,
        model_specs=model_specs,
        unit_test_passed=unit_test_passed,
    )
    if write:
        write_artifact(active_config.resolved_output_path(), artifact)
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
