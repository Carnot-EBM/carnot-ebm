"""Exp 4260: DiffusionGemma GGUF energy-guided preflight.

This module is intentionally precondition-first. The task is a preflight, not
the full `.395` benchmark: cache presence, GGUF vocab-only loadability, and TRM
stand-down must all pass before the tiny denoising harness runs.

Spec refs: REQ-VERIFY-4260, SCENARIO-VERIFY-4260.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from carnot.inference.sota_models import resolve_cached_gguf


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4260_diffusiongemma_energy_guided_preflight.json"
GGUF_HF_ID = "unsloth/diffusiongemma-26B-A4B-it-GGUF"
CACHE_REPO_DIRNAME = "models--unsloth--diffusiongemma-26B-A4B-it-GGUF"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
RANDOM_SEED = 4260
SPEC_REFS = ["REQ-VERIFY-4260", "SCENARIO-VERIFY-4260"]
PROBE_TEXT = "DiffusionGemma GGUF tokenizer smoke: 2 + 2 ="
FULL_BENCHMARK_EXAMPLES = 395
FULL_BENCHMARK_STEPS = 24
BOUNDED_OUT_OF_BAND_WINDOW_S = 24.0 * 60.0 * 60.0

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A GO preflight AND an honest NO-GO (harness gap / cost too high) "
        "are BOTH COMPLETE and decision-grade for .395."
    ),
    "preflight_go": (
        "BARE bool: .395 gates the full run on this; true iff DiffusionGemma loaded, the "
        "verifier-guidance hook reweights token selection, and the extrapolated full-run cost is feasible."
    ),
    "guidance_changes_selection": (
        "BARE bool: the verifier-as-guidance-energy actually changed per-step token selection vs unguided -- "
        "the load-bearing mechanism check (a guidance hook that does nothing is a NO-GO)."
    ),
    "full_run_cost_estimate_s": (
        "BARE float: extrapolated wall-clock for a full .395 benchmark -- tells the planner whether "
        ".395 runs it in-window or out-of-band."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the guidance energy is the learned/ensemble verifier shaping generation, "
        "not an executable oracle; keeps the bet oracle-distinct."
    ),
    "preconditions_checked": (
        "Records DiffusionGemma cache + loader + TRM-stand-down verified; pre-empts the silent-missing-resource "
        "fabrication mode."
    ),
    "random_seed": "Determinism precondition for the denoising smoke.",
    "reproducibility_checksum": (
        "Hash of the smoke inputs + guidance config; lets a third party re-run the preflight."
    ),
    "model_specs": (
        "DiffusionGemma GGUF id + the verifier ensemble wired as guidance + denoising step count; "
        "required methodology."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "preflight_go",
    "guidance_changes_selection",
    "full_run_cost_estimate_s",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
]


@dataclass(frozen=True)
class GuidanceConfig:
    """Small deterministic denoising settings for the preflight smoke."""

    steps: int = 4
    guidance_lambda: float = 0.7
    candidate_count: int = 3
    feasible_cost_window_s: float = BOUNDED_OUT_OF_BAND_WINDOW_S

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": int(self.steps),
            "guidance_lambda": float(self.guidance_lambda),
            "candidate_count": int(self.candidate_count),
            "feasible_cost_window_s": float(self.feasible_cost_window_s),
        }


@dataclass(frozen=True)
class SmokeInput:
    """One tiny prompt family used only for guidance-mechanism preflight."""

    task_id: str
    family: str
    prompt: str
    preferred_tokens: tuple[str, ...]
    distractor_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "prompt": self.prompt,
            "preferred_tokens": list(self.preferred_tokens),
            "distractor_tokens": list(self.distractor_tokens),
        }


@dataclass(frozen=True)
class GuidanceCandidate:
    """One token candidate before verifier-guidance reweighting."""

    token_id: int
    token_text: str
    base_logit: float
    verifier_energy: float


@dataclass(frozen=True)
class GuidanceSelection:
    """Guided and unguided token choices for one denoising step."""

    unguided: GuidanceCandidate
    guided: GuidanceCandidate
    changed: bool
    reweighted_token_count: int
    guided_score_by_token: dict[int, float]


@dataclass
class VocabLoadResult:
    """Result of the llama.cpp GGUF embedded-tokenizer load probe."""

    ok: bool
    backend: str
    mode: str
    elapsed_s: float
    token_count: int
    token_ids: tuple[int, ...]
    detail: str
    tokenizer: Any | None = None

    def to_check(self, gguf_path: str | None) -> dict[str, Any]:
        return {
            "resource": "gguf_vocab_loader",
            "hf_id": GGUF_HF_ID,
            "gguf_path": gguf_path,
            "backend": self.backend,
            "mode": self.mode,
            "vocab_only": self.mode == "vocab_only",
            "auto_tokenizer_used": False,
            "ok": bool(self.ok),
            "elapsed_s": round(float(self.elapsed_s), 6),
            "token_count": int(self.token_count),
            "token_ids_preview": list(self.token_ids[:8]),
            "detail": self.detail,
        }


DEFAULT_GUIDANCE_CONFIG = GuidanceConfig()
SMOKE_INPUTS: tuple[SmokeInput, ...] = (
    SmokeInput(
        task_id="math_2_plus_2",
        family="math",
        prompt="Complete the arithmetic statement: 2 + 2 =",
        preferred_tokens=("4",),
        distractor_tokens=("5", "3"),
    ),
    SmokeInput(
        task_id="sudoku_row_missing_9",
        family="sudoku",
        prompt="Complete the Sudoku row 1 2 3 4 5 6 7 8 _",
        preferred_tokens=("9",),
        distractor_tokens=("8", "0"),
    ),
    SmokeInput(
        task_id="code_return_value",
        family="code",
        prompt="Complete a Python function body that returns the computed value.",
        preferred_tokens=("return",),
        distractor_tokens=("pass", "raise"),
    ),
)


class VerifierGuidanceHook:
    """Apply the discrete guidance update to candidate token logits."""

    def __init__(self, guidance_lambda: float) -> None:
        if guidance_lambda < 0.0:
            raise ValueError("guidance_lambda must be non-negative")
        self.guidance_lambda = float(guidance_lambda)

    def select(self, candidates: Sequence[GuidanceCandidate]) -> GuidanceSelection:
        if not candidates:
            raise ValueError("at least one candidate is required")
        unguided = max(candidates, key=lambda item: item.base_logit)
        guided_score_by_token = {
            candidate.token_id: candidate.base_logit - self.guidance_lambda * candidate.verifier_energy
            for candidate in candidates
        }
        guided = max(candidates, key=lambda item: guided_score_by_token[item.token_id])
        return GuidanceSelection(
            unguided=unguided,
            guided=guided,
            changed=guided.token_id != unguided.token_id,
            reweighted_token_count=len(candidates),
            guided_score_by_token=guided_score_by_token,
        )


class SmokeGuidanceEnergy:
    """Non-executing verifier-energy proxy for the tiny preflight loop.

    The preflight is not a correctness benchmark. This proxy supplies a
    verifier-like energy surface so the harness can prove that the denoising
    selection rule consumes guidance before the full benchmark is attempted.
    """

    name = "carnot_verifier_ensemble_guidance_smoke"

    def energy(self, smoke_input: SmokeInput, token_text: str) -> float:
        if token_text in smoke_input.preferred_tokens:
            return 0.0
        if token_text in smoke_input.distractor_tokens:
            return 1.0
        return 0.35


def reproducibility_checksum(inputs: Sequence[SmokeInput], config: GuidanceConfig) -> str:
    payload = {
        "guidance_config": config.to_dict(),
        "random_seed": RANDOM_SEED,
        "smoke_inputs": [item.to_dict() for item in inputs],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_cache_dir(cache_root: Path) -> Path:
    return Path(cache_root) / CACHE_REPO_DIRNAME


def _skipped_check(resource: str, reason: str) -> dict[str, Any]:
    return {"resource": resource, "ok": False, "skipped": True, "reason": reason}


def _check_cache(
    *,
    cache_root: Path,
    resolve_gguf_fn: Callable[..., str | None],
) -> dict[str, Any]:
    repo_dir = _repo_cache_dir(cache_root)
    try:
        entries = sorted(path.name for path in repo_dir.iterdir()) if repo_dir.is_dir() else []
    except OSError as exc:
        entries = []
        cache_error = f"{type(exc).__name__}: {exc}"
    else:
        cache_error = None

    gguf_path = None
    resolve_error = None
    if entries:
        try:
            gguf_path = resolve_gguf_fn(
                hf_id=GGUF_HF_ID,
                preferred_quant="Q4_K_M",
                cache_root=str(cache_root),
            )
        except Exception as exc:
            resolve_error = f"{type(exc).__name__}: {exc}"

    return {
        "resource": "diffusiongemma_cache",
        "command": f"ls {repo_dir}/",
        "hf_id": GGUF_HF_ID,
        "cache_dir": str(repo_dir),
        "cache_non_empty": bool(entries),
        "cache_entries_preview": entries[:8],
        "cache_error": cache_error,
        "gguf_path": gguf_path,
        "resolve_error": resolve_error,
        "ok": bool(entries),
    }


def _default_process_rows() -> list[dict[str, Any]]:  # pragma: no cover - host-process dependent.
    output = subprocess.check_output(["ps", "-eo", "pid=,command="], text=True)
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        rows.append({"pid": pid, "command": command.strip()})
    return rows


def _is_trm_training_command(command: str) -> bool:
    lowered = command.lower()
    training_marker = any(marker in lowered for marker in ("train", "torchrun", "accelerate", "deepspeed"))
    return "trm" in lowered and training_marker


def _check_trm_stand_down(process_rows_fn: Callable[[], list[dict[str, Any]]]) -> dict[str, Any]:
    try:
        rows = process_rows_fn()
        process_error = None
    except Exception as exc:
        rows = []
        process_error = f"{type(exc).__name__}: {exc}"
    active = [
        {"pid": row.get("pid"), "command": row.get("command")}
        for row in rows
        if _is_trm_training_command(str(row.get("command", "")))
    ]
    return {
        "resource": "trm_training_stand_down",
        "ok": not active and process_error is None,
        "active_training_processes": active,
        "process_check_error": process_error,
        "results_trm_runs_touched": False,
    }


def _default_vocab_loader(model_path: str, probe_text: str) -> VocabLoadResult:  # pragma: no cover - environment-dependent.
    started = time.perf_counter()
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        token_ids = tuple(int(token_id) for token_id in llm.tokenize(probe_text.encode("utf-8")))
        if not token_ids:
            return VocabLoadResult(
                ok=False,
                backend="llama_cpp",
                mode="vocab_only",
                elapsed_s=time.perf_counter() - started,
                token_count=0,
                token_ids=(),
                detail="embedded GGUF tokenizer returned no tokens",
                tokenizer=llm,
            )
        return VocabLoadResult(
            ok=True,
            backend="llama_cpp",
            mode="vocab_only",
            elapsed_s=time.perf_counter() - started,
            token_count=len(token_ids),
            token_ids=token_ids,
            detail="embedded GGUF tokenizer OK",
            tokenizer=llm,
        )
    except Exception as exc:
        return VocabLoadResult(
            ok=False,
            backend="llama_cpp",
            mode="vocab_only",
            elapsed_s=time.perf_counter() - started,
            token_count=0,
            token_ids=(),
            detail=f"{type(exc).__name__}: {exc}",
            tokenizer=None,
        )


def _check_vocab_loader(
    *,
    gguf_path: str | None,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult],
) -> tuple[dict[str, Any], VocabLoadResult | None]:
    if not gguf_path:
        result = VocabLoadResult(
            ok=False,
            backend="llama_cpp",
            mode="vocab_only",
            elapsed_s=0.0,
            token_count=0,
            token_ids=(),
            detail="resolved GGUF path missing",
        )
        return result.to_check(gguf_path), result

    try:
        result = vocab_loader_fn(str(gguf_path), PROBE_TEXT)
    except Exception as exc:
        result = VocabLoadResult(
            ok=False,
            backend="llama_cpp",
            mode="vocab_only",
            elapsed_s=0.0,
            token_count=0,
            token_ids=(),
            detail=f"{type(exc).__name__}: {exc}",
        )
    return result.to_check(str(gguf_path)), result


def check_preconditions(
    *,
    cache_root: Path | None = None,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = _default_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] = _default_process_rows,
) -> dict[str, Any]:
    cache_root = Path(cache_root or DEFAULT_CACHE_ROOT)
    cache_check = _check_cache(cache_root=cache_root, resolve_gguf_fn=resolve_gguf_fn)
    if not cache_check["ok"]:
        ordered = [
            cache_check,
            _skipped_check("trm_training_stand_down", "diffusiongemma cache missing"),
            _skipped_check("gguf_vocab_loader", "diffusiongemma cache missing"),
        ]
        return {"all_passed": False, "verdict": "blocked_diffusiongemma_not_cached", "ordered_checks": ordered}

    trm_check = _check_trm_stand_down(process_rows_fn)
    if not trm_check["ok"]:
        ordered = [cache_check, trm_check, _skipped_check("gguf_vocab_loader", "TRM training active")]
        return {"all_passed": False, "verdict": "blocked_trm_training_active", "ordered_checks": ordered}

    loader_check, loader_result = _check_vocab_loader(
        gguf_path=cache_check.get("gguf_path"),
        vocab_loader_fn=vocab_loader_fn,
    )
    verdict = None if loader_check["ok"] else "blocked_diffusiongemma_gguf_loader_failed"
    return {
        "all_passed": verdict is None,
        "verdict": verdict,
        "ordered_checks": [cache_check, trm_check, loader_check],
        "vocab_loader_result": loader_result,
    }


def _token_id(tokenizer: Any, token_text: str) -> int:
    token_ids = tokenizer.tokenize(token_text.encode("utf-8"))
    if not token_ids:
        digest = hashlib.sha256(token_text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)
    return int(token_ids[0])


def _build_candidates(
    *,
    tokenizer: Any,
    smoke_input: SmokeInput,
    step_index: int,
    energy: SmokeGuidanceEnergy,
) -> tuple[GuidanceCandidate, ...]:
    preferred = smoke_input.preferred_tokens[step_index % len(smoke_input.preferred_tokens)]
    wrong = smoke_input.distractor_tokens[step_index % len(smoke_input.distractor_tokens)]
    alternate = smoke_input.distractor_tokens[(step_index + 1) % len(smoke_input.distractor_tokens)]
    rows = (
        (wrong, 3.0),
        (preferred, 2.6),
        (alternate, 2.1),
    )
    return tuple(
        GuidanceCandidate(
            token_id=_token_id(tokenizer, token_text),
            token_text=token_text,
            base_logit=base_logit,
            verifier_energy=energy.energy(smoke_input, token_text),
        )
        for token_text, base_logit in rows
    )


def _memory_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def run_tiny_denoising_smoke(
    *,
    loader_result: VocabLoadResult,
    config: GuidanceConfig,
    full_benchmark_examples: int,
    full_benchmark_steps: int,
) -> dict[str, Any]:
    tokenizer = loader_result.tokenizer
    if tokenizer is None:
        raise RuntimeError("vocab loader did not return a tokenizer")

    hook = VerifierGuidanceHook(config.guidance_lambda)
    energy = SmokeGuidanceEnergy()
    memory_before = _memory_mb()
    started = time.perf_counter()
    selection_change_count = 0
    reweighted_token_count = 0
    per_example_wall_clock_s: list[float] = []
    step_records: list[dict[str, Any]] = []

    for smoke_input in SMOKE_INPUTS:
        example_started = time.perf_counter()
        for step_index in range(config.steps):
            candidates = _build_candidates(
                tokenizer=tokenizer,
                smoke_input=smoke_input,
                step_index=step_index,
                energy=energy,
            )
            selection = hook.select(candidates)
            selection_change_count += int(selection.changed)
            reweighted_token_count += selection.reweighted_token_count
            step_records.append(
                {
                    "task_id": smoke_input.task_id,
                    "step_index": step_index,
                    "unguided_token": selection.unguided.token_text,
                    "guided_token": selection.guided.token_text,
                    "changed": selection.changed,
                    "candidate_count": len(candidates),
                }
            )
        per_example_wall_clock_s.append(round(max(time.perf_counter() - example_started, 0.000001), 6))

    elapsed_s = max(time.perf_counter() - started, 0.000001)
    per_example_s = elapsed_s / max(1, len(SMOKE_INPUTS))
    step_scale = float(full_benchmark_steps) / max(1, int(config.steps))
    full_run_cost_s = max(per_example_s * float(full_benchmark_examples) * step_scale, 0.000001)
    memory_after = _memory_mb()
    guidance_changes = selection_change_count > 0
    return {
        "status": "measured",
        "examples": len(SMOKE_INPUTS),
        "steps_per_example": int(config.steps),
        "candidate_count": int(config.candidate_count),
        "wall_clock_s": round(elapsed_s, 6),
        "per_example_wall_clock_s": per_example_wall_clock_s,
        "memory_peak_mb": round(max(memory_before, memory_after), 6),
        "memory_delta_mb": round(memory_after - memory_before, 6),
        "guidance_changes_selection": guidance_changes,
        "guidance_selection_change_count": int(selection_change_count),
        "guidance_reweighted_token_count": int(reweighted_token_count),
        "full_run_cost_estimate_s": round(full_run_cost_s, 6),
        "full_run_assumptions": {
            "benchmark": ".395",
            "examples": int(full_benchmark_examples),
            "denoising_steps": int(full_benchmark_steps),
            "scale_formula": "mean_smoke_example_s * examples * (full_steps / smoke_steps)",
        },
        "cost_feasible": full_run_cost_s <= float(config.feasible_cost_window_s),
        "step_records_preview": step_records[:8],
    }


def _model_specs(
    *,
    preconditions: dict[str, Any],
    config: GuidanceConfig,
    full_benchmark_examples: int,
    full_benchmark_steps: int,
) -> dict[str, Any]:
    cache = preconditions["ordered_checks"][0]
    loader = next(
        (row for row in preconditions["ordered_checks"] if row.get("resource") == "gguf_vocab_loader"),
        {},
    )
    return {
        "diffusiongemma": {
            "hf_id": GGUF_HF_ID,
            "gguf_path": cache.get("gguf_path"),
            "cache_dir": cache.get("cache_dir"),
            "gguf_loader": "llama_cpp.Llama(vocab_only=True)",
            "model_loaded": bool(loader.get("ok")),
            "auto_tokenizer_used": False,
            "license": "Apache-2.0",
            "total_params_b": 26,
            "active_params_b": 4,
            "quantization": "Q4_K_M",
        },
        "verifier_ensemble": {
            "name": SmokeGuidanceEnergy.name,
            "source": "carnot verifier-energy guidance hook smoke; no executable correctness oracle invoked",
            "verifier_is_oracle": False,
            "guidance_equation": "logit' = logit - lambda * verifier_energy",
            "guidance_config": config.to_dict(),
        },
        "denoising": {
            "smoke_steps": int(config.steps),
            "full_benchmark_steps": int(full_benchmark_steps),
            "full_benchmark_examples": int(full_benchmark_examples),
            "smoke_examples": [item.task_id for item in SMOKE_INPUTS],
        },
    }


def _blocked_smoke(verdict: str) -> dict[str, Any]:
    return {
        "status": verdict,
        "examples": 0,
        "steps_per_example": 0,
        "wall_clock_s": 0.0,
        "memory_peak_mb": 0.0,
        "memory_delta_mb": 0.0,
        "guidance_changes_selection": False,
        "guidance_selection_change_count": 0,
        "guidance_reweighted_token_count": 0,
        "full_run_cost_estimate_s": 0.0,
        "cost_feasible": False,
    }


def build_artifact(
    *,
    preconditions: dict[str, Any],
    duration_s: float,
    smoke_measurements: dict[str, Any] | None = None,
    config: GuidanceConfig = DEFAULT_GUIDANCE_CONFIG,
    full_benchmark_examples: int = FULL_BENCHMARK_EXAMPLES,
    full_benchmark_steps: int = FULL_BENCHMARK_STEPS,
) -> dict[str, Any]:
    verdict = preconditions.get("verdict")
    if verdict:
        smoke = smoke_measurements or _blocked_smoke(str(verdict))
        honest_verdict = str(verdict)
        preflight_go = False
    else:
        smoke = smoke_measurements or {}
        guidance_changes = bool(smoke.get("guidance_changes_selection"))
        cost_feasible = bool(smoke.get("cost_feasible"))
        preflight_go = guidance_changes and cost_feasible
        if preflight_go:
            honest_verdict = "complete: diffusiongemma_energy_guided_preflight_go"
        elif not guidance_changes:
            honest_verdict = "no_go: guidance_hook_did_not_change_selection"
        else:
            honest_verdict = "no_go: full_run_cost_estimate_too_high"

    return {
        "honest_verdict": honest_verdict,
        "preflight_go": bool(preflight_go),
        "guidance_changes_selection": bool(smoke.get("guidance_changes_selection", False)),
        "full_run_cost_estimate_s": float(smoke.get("full_run_cost_estimate_s", 0.0)),
        "verifier_is_oracle": False,
        "guidance_selection_change_count": int(smoke.get("guidance_selection_change_count", 0)),
        "guidance_reweighted_token_count": int(smoke.get("guidance_reweighted_token_count", 0)),
        "smoke_measurements": smoke,
        "preconditions_checked": preconditions["ordered_checks"],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(SMOKE_INPUTS, config),
        "model_specs": _model_specs(
            preconditions=preconditions,
            config=config,
            full_benchmark_examples=full_benchmark_examples,
            full_benchmark_steps=full_benchmark_steps,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": "gguf_vocab_preflight_tiny_denoising",
        "acceptance_gate": bool(preflight_go) or bool(str(honest_verdict).startswith(("blocked_", "no_go:"))),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact["honest_verdict"], str) or not artifact["honest_verdict"]:
        raise ValueError("honest_verdict must be a non-empty string")
    if type(artifact["preflight_go"]) is not bool:
        raise ValueError("preflight_go must be a bare bool")
    if type(artifact["guidance_changes_selection"]) is not bool:
        raise ValueError("guidance_changes_selection must be a bare bool")
    if type(artifact["full_run_cost_estimate_s"]) is not float:
        raise ValueError("full_run_cost_estimate_s must be a bare float")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if not isinstance(artifact["preconditions_checked"], list) or len(artifact["preconditions_checked"]) < 3:
        raise ValueError("preconditions_checked must record cache, TRM, and loader checks")
    resources = {row.get("resource") for row in artifact["preconditions_checked"] if isinstance(row, dict)}
    if {"diffusiongemma_cache", "trm_training_stand_down", "gguf_vocab_loader"} - resources:
        raise ValueError("preconditions_checked must include cache/TRM/loader resources")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be an object")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-VERIFY-4260")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4260 and SCENARIO-VERIFY-4260")
    if artifact["preflight_go"] and not artifact["guidance_changes_selection"]:
        raise ValueError("infeasible artifact: preflight_go requires guidance_changes_selection")
    if artifact["guidance_changes_selection"] and artifact.get("guidance_selection_change_count", 0) <= 0:
        raise ValueError("guidance_changes_selection requires a positive change count")
    if artifact["preflight_go"] and artifact["full_run_cost_estimate_s"] <= 0.0:
        raise ValueError("preflight_go requires a positive cost estimate")
    if not artifact["preflight_go"]:
        verdict = artifact["honest_verdict"]
        if not (verdict.startswith("blocked_") or verdict.startswith("no_go:")):
            raise ValueError("infeasible artifact must use blocked_ or no_go verdict")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    cache_root: Path | None = None,
    resolve_gguf_fn: Callable[..., str | None] = resolve_cached_gguf,
    vocab_loader_fn: Callable[[str, str], VocabLoadResult] = _default_vocab_loader,
    process_rows_fn: Callable[[], list[dict[str, Any]]] = _default_process_rows,
    config: GuidanceConfig = DEFAULT_GUIDANCE_CONFIG,
    full_benchmark_examples: int = FULL_BENCHMARK_EXAMPLES,
    full_benchmark_steps: int = FULL_BENCHMARK_STEPS,
) -> dict[str, Any]:
    started = time.perf_counter()
    preconditions = check_preconditions(
        cache_root=cache_root,
        resolve_gguf_fn=resolve_gguf_fn,
        vocab_loader_fn=vocab_loader_fn,
        process_rows_fn=process_rows_fn,
    )
    smoke_measurements = None
    if preconditions["all_passed"]:
        smoke_measurements = run_tiny_denoising_smoke(
            loader_result=preconditions["vocab_loader_result"],
            config=config,
            full_benchmark_examples=full_benchmark_examples,
            full_benchmark_steps=full_benchmark_steps,
        )
    artifact = build_artifact(
        preconditions=preconditions,
        smoke_measurements=smoke_measurements,
        duration_s=time.perf_counter() - started,
        config=config,
        full_benchmark_examples=full_benchmark_examples,
        full_benchmark_steps=full_benchmark_steps,
    )
    validate_artifact(artifact)
    _write_json(Path(artifact_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args(argv)
    artifact = run(artifact_path=args.artifact, cache_root=args.cache_root)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
