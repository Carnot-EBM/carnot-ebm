#!/usr/bin/env python3
"""Exp 4189: verifier-guided DiffusionGemma decoding feasibility.

This runner is deliberately precondition-first. DiffusionGemma guidance is only
decision-grade if the actual model loads, the Hugging Face denoising loop
accepts a per-step logits processor, and that processor fires before
low-entropy token commit. When the weight shards are absent, the correct
deliverable is a blocked artifact, not a simulated smoke benchmark.

Spec refs: REQ-VERIFY-4189, SCENARIO-VERIFY-4189.
"""

import argparse
import hashlib
import inspect
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
for _path in (PYTHON_DIR, ROOT):
    if str(_path) not in sys.path:  # pragma: no cover - import bootstrap.
        sys.path.insert(0, str(_path))

try:  # pragma: no cover - exercised through the processor tests when present.
    import torch
except Exception:  # pragma: no cover - lets preflight report missing torch honestly.
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - import availability is environment-specific.
    from transformers.generation import LogitsProcessor
except Exception:  # pragma: no cover

    class LogitsProcessor:  # type: ignore[no-redef]
        """Fallback base so tests can import the module without transformers."""


ARTIFACT_PATH = ROOT / "results" / "experiment_4189_diffusiongemma_verifier_guided_decoding.json"
REPO_ID = "google/diffusiongemma-26B-A4B-it"
RANDOM_SEED = 4189
SMOKE_N = 20
SPEC_REFS = ["REQ-VERIFY-4189", "SCENARIO-VERIFY-4189"]

DEFAULT_GUIDANCE_CONFIG = {
    "domain": "executable_python_smoke",
    "guidance_lambda": 0.7,
    "top_k": 4,
    "max_new_tokens": 128,
    "max_denoising_steps": 24,
    "smoke_n": SMOKE_N,
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean blocked_diffusiongemma_* (model/hook/CUDA unavailable) "
        "is a COMPLETE feasibility verdict that de-risks the .389 scale-up."
    ),
    "diffusiongemma_feasible": (
        "Bare bool: the model loaded AND the per-step guidance hook fired end-to-end; "
        "the precondition for any .389 headline benchmark."
    ),
    "guided_vs_unguided_delta": (
        "guided - unguided pass-rate with bootstrap CI95 on the smoke n; the first "
        "directional read on whether verifier guidance helps at LLM scale."
    ),
    "model_specs": (
        "DiffusionGemma + the verifier ensemble invoked; required methodology for a live-LLM artifact."
    ),
    "random_seed": "Determinism precondition; the denoising + guidance must be reproducible.",
    "reproducibility_checksum": (
        "Hash of the prompt set + guidance config; catches silent drift."
    ),
    "preconditions_checked": (
        "Records the cache / per-step-logit / CUDA checks; pre-empts the silent-missing-resource fabrication mode."
    ),
}

REQUIRED_FIELDS = [
    "honest_verdict",
    "diffusiongemma_feasible",
    "guided_vs_unguided_delta",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
]


@dataclass(frozen=True)
class SmokeTask:
    """One tiny executable-code prompt plus tests for the optional live smoke."""

    task_id: str
    prompt: str
    entry_point: str
    tests: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "tests": list(self.tests),
        }


SMOKE_TASKS: tuple[SmokeTask, ...] = (
    SmokeTask("add_one", "Write Python function add_one(x) returning x + 1.", "add_one", ("add_one(2) == 3", "add_one(-1) == 0")),
    SmokeTask("double", "Write Python function double(x) returning twice x.", "double", ("double(4) == 8", "double(-3) == -6")),
    SmokeTask("square", "Write Python function square(x) returning x squared.", "square", ("square(5) == 25", "square(-4) == 16")),
    SmokeTask("is_even", "Write Python function is_even(x) returning True iff x is even.", "is_even", ("is_even(2) is True", "is_even(3) is False")),
    SmokeTask("first_char", "Write Python function first_char(s) returning the first character.", "first_char", ("first_char('abc') == 'a'", "first_char('z') == 'z'")),
    SmokeTask("last_char", "Write Python function last_char(s) returning the last character.", "last_char", ("last_char('abc') == 'c'", "last_char('q') == 'q'")),
    SmokeTask("abs_diff", "Write Python function abs_diff(a, b) returning absolute difference.", "abs_diff", ("abs_diff(7, 2) == 5", "abs_diff(2, 7) == 5")),
    SmokeTask("max2", "Write Python function max2(a, b) returning the larger value.", "max2", ("max2(1, 9) == 9", "max2(8, 3) == 8")),
    SmokeTask("min2", "Write Python function min2(a, b) returning the smaller value.", "min2", ("min2(1, 9) == 1", "min2(8, 3) == 3")),
    SmokeTask("negate", "Write Python function negate(x) returning -x.", "negate", ("negate(5) == -5", "negate(-2) == 2")),
    SmokeTask("cube", "Write Python function cube(x) returning x cubed.", "cube", ("cube(3) == 27", "cube(-2) == -8")),
    SmokeTask("nonempty", "Write Python function nonempty(s) returning True iff s is not empty.", "nonempty", ("nonempty('x') is True", "nonempty('') is False")),
    SmokeTask("concat", "Write Python function concat(a, b) returning a followed by b.", "concat", ("concat('a', 'b') == 'ab'", "concat('', 'x') == 'x'")),
    SmokeTask("head2", "Write Python function head2(xs) returning the first two elements.", "head2", ("head2([1,2,3]) == [1,2]", "head2(['a','b']) == ['a','b']")),
    SmokeTask("tail", "Write Python function tail(xs) returning all but the first element.", "tail", ("tail([1,2,3]) == [2,3]", "tail(['a']) == []")),
    SmokeTask("sum2", "Write Python function sum2(a, b) returning a + b.", "sum2", ("sum2(1, 2) == 3", "sum2(-2, 5) == 3")),
    SmokeTask("is_positive", "Write Python function is_positive(x) returning True iff x > 0.", "is_positive", ("is_positive(1) is True", "is_positive(0) is False")),
    SmokeTask("repeat_twice", "Write Python function repeat_twice(s) returning s repeated twice.", "repeat_twice", ("repeat_twice('ab') == 'abab'", "repeat_twice('') == ''")),
    SmokeTask("floor_half", "Write Python function floor_half(x) returning x // 2.", "floor_half", ("floor_half(5) == 2", "floor_half(4) == 2")),
    SmokeTask("bool_not", "Write Python function bool_not(x) returning not x.", "bool_not", ("bool_not(True) is False", "bool_not(False) is True")),
)


class VerifierEnergyLogitsProcessor(LogitsProcessor):
    """Apply executable verifier energy to DiffusionGemma per-step logits.

    The DiffusionGemma sampler operates in logit space before converting to a
    categorical distribution and accepting low-entropy tokens. Multiplying a
    candidate token probability by ``exp(-lambda * energy)`` is therefore the
    same as subtracting ``lambda * energy`` from that candidate's logit. The
    processor only evaluates the top-k candidates per canvas position so an
    executable verifier can be used during a bounded smoke run.
    """

    def __init__(
        self,
        verifier_energy_fn: Callable[..., Any],
        *,
        guidance_lambda: float,
        top_k: int,
        max_energy: float = 8.0,
    ) -> None:
        if guidance_lambda < 0:
            raise ValueError("guidance_lambda must be non-negative")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.verifier_energy_fn = verifier_energy_fn
        self.guidance_lambda = float(guidance_lambda)
        self.top_k = int(top_k)
        self.max_energy = float(max_energy)
        self.call_count = 0
        self.guidance_applied = False

    def __call__(self, input_ids: Any, scores: Any, cur_step: Any | None = None, **_: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch is required for verifier-guided logits processing")
        if scores.ndim != 3:
            raise ValueError("DiffusionGemma logits must have shape [batch, canvas, vocab]")

        k = min(self.top_k, int(scores.shape[-1]))
        _, candidate_token_ids = torch.topk(scores, k=k, dim=-1)
        energies = self.verifier_energy_fn(
            input_ids=input_ids,
            candidate_token_ids=candidate_token_ids,
            scores=scores,
            cur_step=cur_step,
        )
        if not torch.is_tensor(energies):
            energies = torch.as_tensor(energies, device=scores.device)
        energies = energies.to(device=scores.device, dtype=scores.dtype)
        if tuple(energies.shape) != tuple(candidate_token_ids.shape):
            raise ValueError("verifier energy shape must match candidate_token_ids")
        energies = torch.nan_to_num(
            energies,
            nan=self.max_energy,
            posinf=self.max_energy,
            neginf=0.0,
        ).clamp(min=0.0, max=self.max_energy)

        adjusted = scores.clone()
        guided_values = adjusted.gather(-1, candidate_token_ids) - (self.guidance_lambda * energies)
        adjusted.scatter_(-1, candidate_token_ids, guided_values)
        self.call_count += 1
        self.guidance_applied = True
        return adjusted


class ExecutablePythonVerifierEnergy:
    """Score candidate tokens by executing the completed Python task when possible.

    This is intentionally simple and expensive. The smoke run is only n=20 with
    a tiny top-k because the goal is feasibility of guidance, not a production
    decoder. Partial candidates that cannot yet form executable code receive a
    neutral energy so the verifier does not fabricate certainty mid-denoising.
    """

    def __init__(self, tokenizer: Any, tasks: Sequence[SmokeTask], *, neutral_energy: float = 0.5) -> None:
        self.tokenizer = tokenizer
        self.tasks = list(tasks)
        self.neutral_energy = float(neutral_energy)

    def __call__(self, *, input_ids: Any, candidate_token_ids: Any, scores: Any, cur_step: Any | None) -> Any:  # pragma: no cover - live model path.
        if torch is None:
            raise RuntimeError("torch is required for executable verifier energy")
        base_canvas = torch.argmax(scores, dim=-1)
        out = torch.empty(candidate_token_ids.shape, device=candidate_token_ids.device, dtype=torch.float32)
        for batch_idx in range(candidate_token_ids.shape[0]):
            task = self.tasks[min(batch_idx, len(self.tasks) - 1)]
            for pos_idx in range(candidate_token_ids.shape[1]):
                for cand_idx in range(candidate_token_ids.shape[2]):
                    canvas = base_canvas[batch_idx].clone()
                    canvas[pos_idx] = candidate_token_ids[batch_idx, pos_idx, cand_idx]
                    seq = torch.cat([input_ids[batch_idx].detach().cpu(), canvas.detach().cpu()])
                    text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                    out[batch_idx, pos_idx, cand_idx] = self.score_text(text, task)
        return out

    def score_text(self, text: str, task: SmokeTask) -> float:  # pragma: no cover - live model path.
        code = extract_python_code(text, task.entry_point)
        if not code:
            return self.neutral_energy
        result = execute_code_task(code, task)
        if result == "pass":
            return 0.0
        if result == "fail":
            return 2.0
        return self.neutral_energy


def extract_python_code(text: str, entry_point: str) -> str | None:  # pragma: no cover - live model path.
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced or [text]
    pattern = re.compile(rf"def\s+{re.escape(entry_point)}\s*\(.*", flags=re.DOTALL)
    for candidate in candidates:
        match = pattern.search(candidate)
        if match:
            return match.group(0).strip()
    return None


def execute_code_task(code: str, task: SmokeTask) -> str:  # pragma: no cover - live model path.
    forbidden = ("__import__", "open(", "exec(", "eval(", "compile(", "subprocess", "socket", "os.", "sys.")
    if any(token in code for token in forbidden):
        return "fail"
    namespace: dict[str, Any] = {"__builtins__": {"abs": abs, "bool": bool, "len": len, "max": max, "min": min, "sum": sum, "range": range, "list": list, "str": str, "int": int}}
    try:
        exec(code, namespace)
        fn = namespace.get(task.entry_point)
        if not callable(fn):
            return "partial"
        for assertion in task.tests:
            if not bool(eval(assertion, namespace)):
                return "fail"
    except SyntaxError:
        return "partial"
    except Exception:
        return "fail"
    return "pass"


def reproducibility_checksum(tasks: Sequence[SmokeTask], guidance_config: dict[str, Any]) -> str:
    payload = {
        "guidance_config": guidance_config,
        "tasks": [task.to_dict() for task in tasks],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def bootstrap_delta_ci(
    guided_passed: Sequence[bool],
    unguided_passed: Sequence[bool],
    *,
    seed: int,
    resamples: int = 1000,
) -> list[float]:
    if len(guided_passed) != len(unguided_passed):
        raise ValueError("guided and unguided result lengths must match")
    if not guided_passed:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(guided_passed)
    deltas: list[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        guided_mean = sum(1.0 for i in indices if guided_passed[i]) / n
        unguided_mean = sum(1.0 for i in indices if unguided_passed[i]) / n
        deltas.append(guided_mean - unguided_mean)
    deltas.sort()
    return [round(_percentile(deltas, 0.025), 6), round(_percentile(deltas, 0.975), 6)]


def _default_model_info(repo_id: str) -> Any:  # pragma: no cover - live network path.
    from huggingface_hub import HfApi  # noqa: PLC0415

    return HfApi().model_info(repo_id, files_metadata=True)


def _default_cuda_info() -> dict[str, Any]:  # pragma: no cover - hardware path.
    if torch is None:
        return {"available": False, "device_count": 0, "devices": [], "error": "torch_import_failed"}
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        return {
            "available": available,
            "device_count": count,
            "devices": [torch.cuda.get_device_name(i) for i in range(count)],
        }
    except Exception as exc:
        return {"available": False, "device_count": 0, "devices": [], "error": type(exc).__name__}


def inspect_per_step_logit_hook() -> dict[str, Any]:  # pragma: no cover - installed-package path.
    try:
        from transformers.models.diffusion_gemma.generation_diffusion_gemma import (  # noqa: PLC0415
            DiffusionGemmaGenerationMixin,
        )

        source = inspect.getsource(DiffusionGemmaGenerationMixin._denoising_step)
        has_processor = "processed_logits = logits_processor(input_ids, raw_logits, cur_step=cur_step)" in source
        has_sampler = "sampler.accept_canvas(current_canvas, denoiser_canvas, processed_logits, cur_step)" in source
        return {
            "available": bool(has_processor and has_sampler),
            "surface": "DiffusionGemmaGenerationMixin._denoising_step",
            "evidence": "logits_processor(input_ids, raw_logits, cur_step=cur_step) before sampler.accept_canvas"
            if has_processor and has_sampler
            else "expected logits_processor -> sampler.accept_canvas ordering not found",
        }
    except Exception as exc:
        return {
            "available": False,
            "surface": "DiffusionGemmaGenerationMixin._denoising_step",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _repo_cache_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def _expected_weight_files(model_info: Any | None) -> list[dict[str, Any]]:
    siblings = getattr(model_info, "siblings", []) if model_info is not None else []
    out: list[dict[str, Any]] = []
    for sibling in siblings:
        name = str(getattr(sibling, "rfilename", ""))
        if name.endswith(".safetensors") and name.startswith("model-"):
            out.append({"name": name, "size": int(getattr(sibling, "size", 0) or 0)})
    return sorted(out, key=lambda item: item["name"])


def _present_weight_files(cache_dir: Path, repo_id: str) -> set[str]:
    repo_dir = _repo_cache_dir(cache_dir, repo_id)
    if not repo_dir.exists():
        return set()
    present: set[str] = set()
    snapshots = repo_dir / "snapshots"
    if snapshots.exists():
        for path in snapshots.rglob("model-*.safetensors"):
            try:
                if path.is_file() and path.stat().st_size > 0:
                    present.add(path.name)
            except OSError:  # pragma: no cover - filesystem race defense.
                continue
    return present


def _cache_grep_matches(cache_dir: Path) -> list[str]:
    if not cache_dir.exists():
        return []
    return sorted(path.name for path in cache_dir.iterdir() if "diffusion" in path.name.lower())


def check_preconditions(
    *,
    repo_id: str = REPO_ID,
    cache_dir: Path | None = None,
    model_info_fn: Callable[[str], Any] | None = None,
    cuda_info_fn: Callable[[], dict[str, Any]] | None = None,
    hook_info_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir or Path.home() / ".cache" / "huggingface" / "hub")
    model_info_fn = model_info_fn or _default_model_info
    cuda_info_fn = cuda_info_fn or _default_cuda_info
    hook_info_fn = hook_info_fn or inspect_per_step_logit_hook

    repo_error = None
    model_info = None
    try:
        model_info = model_info_fn(repo_id)
        repo_accessible = True
    except Exception as exc:
        repo_accessible = False
        repo_error = f"{type(exc).__name__}: {exc}"

    expected = _expected_weight_files(model_info)
    present = _present_weight_files(cache_dir, repo_id)
    missing = [item["name"] for item in expected if item["name"] not in present]
    full_weights_cached = bool(expected) and not missing

    cache_check = {
        "resource": "diffusiongemma_cache",
        "repo_id": repo_id,
        "cache_dir": str(cache_dir),
        "repo_cache_dir": str(_repo_cache_dir(cache_dir, repo_id)),
        "cache_grep_diffusion_matches": _cache_grep_matches(cache_dir),
        "repo_accessible": repo_accessible,
        "repo_error": repo_error,
        "repo_sha": getattr(model_info, "sha", None),
        "gated": bool(getattr(model_info, "gated", False)) if model_info is not None else None,
        "private": bool(getattr(model_info, "private", False)) if model_info is not None else None,
        "expected_weight_shards": len(expected),
        "present_weight_shards": len(present & {item["name"] for item in expected}),
        "missing_weight_shards": missing,
        "full_weights_cached": full_weights_cached,
        "bounded_metadata_download_ok": repo_accessible,
    }
    hook_check = {"resource": "per_step_logit_hook", **hook_info_fn()}
    cuda_check = {"resource": "cuda", **cuda_info_fn()}

    verdict = None
    if not repo_accessible or not full_weights_cached:
        verdict = "blocked_diffusiongemma_not_cached"
    elif not bool(hook_check.get("available")):
        verdict = "blocked_diffusiongemma_no_perstep_logit_hook"
    elif not bool(cuda_check.get("available")):
        verdict = "blocked_cuda_unavailable"

    checks = {
        "diffusiongemma_cache": cache_check,
        "per_step_logit_hook": hook_check,
        "cuda": cuda_check,
    }
    return {
        "all_passed": verdict is None,
        "verdict": verdict,
        "checks": checks,
        "ordered_checks": [cache_check, hook_check, cuda_check],
    }


def _blocked_delta(verdict: str) -> dict[str, Any]:
    return {
        "status": verdict,
        "n": 0,
        "guided_pass_rate": None,
        "unguided_pass_rate": None,
        "delta": None,
        "ci95": None,
    }


def _model_specs(preconditions: dict[str, Any], *, hook_fired: bool = False) -> dict[str, Any]:
    cache = preconditions["checks"]["diffusiongemma_cache"]
    return {
        "diffusiongemma": {
            "repo_id": cache["repo_id"],
            "repo_sha": cache.get("repo_sha"),
            "weights_cached": cache["full_weights_cached"],
            "expected_weight_shards": cache["expected_weight_shards"],
            "present_weight_shards": cache["present_weight_shards"],
            "per_step_logit_hook": bool(preconditions["checks"]["per_step_logit_hook"].get("available")),
            "guidance_hook_fired": bool(hook_fired),
        },
        "verifier_ensemble": {
            "name": "carnot_executable_code_verifier_energy",
            "source": "python execution smoke over HumanEval-style micro tasks",
            "guidance_equation": "logit' = logit - lambda * verifier_energy",
            "guidance_config": dict(DEFAULT_GUIDANCE_CONFIG),
        },
    }


def build_blocked_artifact(
    *,
    verdict: str,
    preconditions: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "honest_verdict": verdict,
        "diffusiongemma_feasible": False,
        "guided_vs_unguided_delta": _blocked_delta(verdict),
        "model_specs": _model_specs(preconditions, hook_fired=False),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(SMOKE_TASKS, DEFAULT_GUIDANCE_CONFIG),
        "preconditions_checked": preconditions["ordered_checks"],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": "precondition_probe_only",
        "acceptance_gate": True,
    }


def _pass_rate(values: Sequence[bool]) -> float:
    return sum(1 for value in values if value) / max(1, len(values))


def _smoke_delta(
    guided_passed: Sequence[bool],
    unguided_passed: Sequence[bool],
    *,
    seed: int,
) -> dict[str, Any]:
    guided = _pass_rate(guided_passed)
    unguided = _pass_rate(unguided_passed)
    return {
        "status": "measured",
        "n": len(guided_passed),
        "guided_pass_rate": round(guided, 6),
        "unguided_pass_rate": round(unguided, 6),
        "delta": round(guided - unguided, 6),
        "ci95": bootstrap_delta_ci(guided_passed, unguided_passed, seed=seed),
    }


def run_live_smoke(preconditions: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - requires 53GB model cache.
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
    from transformers.generation import LogitsProcessorList

    if torch is None:
        raise RuntimeError("torch is required for live smoke")

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    processor = AutoProcessor.from_pretrained(REPO_ID, local_files_only=True)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        REPO_ID,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device

    tasks = list(SMOKE_TASKS[:SMOKE_N])
    unguided_passed: list[bool] = []
    guided_passed: list[bool] = []
    guided_processor = VerifierEnergyLogitsProcessor(
        ExecutablePythonVerifierEnergy(tokenizer, tasks),
        guidance_lambda=float(DEFAULT_GUIDANCE_CONFIG["guidance_lambda"]),
        top_k=int(DEFAULT_GUIDANCE_CONFIG["top_k"]),
    )

    for task in tasks:
        chat = [{"role": "user", "content": task.prompt + "\nReturn only Python code."}]
        input_ids = processor.apply_chat_template(chat, tokenize=True, return_tensors="pt").to(device)
        unguided = model.generate(
            input_ids,
            max_new_tokens=int(DEFAULT_GUIDANCE_CONFIG["max_new_tokens"]),
            max_denoising_steps=int(DEFAULT_GUIDANCE_CONFIG["max_denoising_steps"]),
        )
        guided = model.generate(
            input_ids,
            max_new_tokens=int(DEFAULT_GUIDANCE_CONFIG["max_new_tokens"]),
            max_denoising_steps=int(DEFAULT_GUIDANCE_CONFIG["max_denoising_steps"]),
            logits_processor=LogitsProcessorList([guided_processor]),
        )
        unguided_text = tokenizer.decode(unguided.sequences[0].detach().cpu().tolist(), skip_special_tokens=True)
        guided_text = tokenizer.decode(guided.sequences[0].detach().cpu().tolist(), skip_special_tokens=True)
        unguided_passed.append(execute_code_task(extract_python_code(unguided_text, task.entry_point) or "", task) == "pass")
        guided_passed.append(execute_code_task(extract_python_code(guided_text, task.entry_point) or "", task) == "pass")

    feasible = guided_processor.guidance_applied and guided_processor.call_count > 0
    return {
        "honest_verdict": "complete: diffusiongemma_guided_smoke_measured" if feasible else "blocked_diffusiongemma_no_perstep_logit_hook",
        "diffusiongemma_feasible": bool(feasible),
        "guided_vs_unguided_delta": _smoke_delta(guided_passed, unguided_passed, seed=RANDOM_SEED),
        "model_specs": _model_specs(preconditions, hook_fired=feasible),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(SMOKE_TASKS, DEFAULT_GUIDANCE_CONFIG),
        "preconditions_checked": preconditions["ordered_checks"],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": 0.0,
        "inference_substrate": "live_diffusiongemma_cuda_with_executable_verifier_guidance",
        "ar_baseline": {"status": "not_run_in_this_guarded_live_path"},
        "acceptance_gate": bool(feasible),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not isinstance(artifact["honest_verdict"], str) or not artifact["honest_verdict"]:
        raise ValueError("honest_verdict must be a non-empty string")
    if not isinstance(artifact["diffusiongemma_feasible"], bool):
        raise ValueError("diffusiongemma_feasible must be a bare bool")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4189")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4189 and SCENARIO-VERIFY-4189")
    if not isinstance(artifact["preconditions_checked"], list) or len(artifact["preconditions_checked"]) < 3:
        raise ValueError("preconditions_checked must record cache, hook, and CUDA checks")
    resources = {row.get("resource") for row in artifact["preconditions_checked"] if isinstance(row, dict)}
    if {"diffusiongemma_cache", "per_step_logit_hook", "cuda"} - resources:
        raise ValueError("preconditions_checked missing cache/hook/cuda resources")
    if not isinstance(artifact["guided_vs_unguided_delta"], dict):
        raise ValueError("guided_vs_unguided_delta must be an object")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be an object")
    if artifact["diffusiongemma_feasible"]:
        delta = artifact["guided_vs_unguided_delta"]
        if delta.get("status") != "measured" or delta.get("n", 0) < SMOKE_N:
            raise ValueError("feasible artifact must include measured smoke n >= 20")
        if not artifact["model_specs"].get("diffusiongemma", {}).get("guidance_hook_fired"):
            raise ValueError("diffusiongemma_feasible requires guidance hook fired")
    else:
        verdict = artifact["honest_verdict"]
        allowed = (
            verdict.startswith("blocked_diffusiongemma_")
            or verdict == "blocked_cuda_unavailable"
        )
        if not allowed:
            raise ValueError("infeasible artifact must use a blocked_diffusiongemma_* or CUDA verdict")
        if artifact["guided_vs_unguided_delta"].get("status") != verdict:
            raise ValueError("blocked delta status must match honest_verdict")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    artifact_path: Path = ARTIFACT_PATH,
    cache_dir: Path | None = None,
    model_info_fn: Callable[[str], Any] | None = None,
    cuda_info_fn: Callable[[], dict[str, Any]] | None = None,
    hook_info_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.time()
    preconditions = check_preconditions(
        cache_dir=cache_dir,
        model_info_fn=model_info_fn,
        cuda_info_fn=cuda_info_fn,
        hook_info_fn=hook_info_fn,
    )
    if preconditions["verdict"] is not None:
        artifact = build_blocked_artifact(
            verdict=preconditions["verdict"],
            preconditions=preconditions,
            duration_s=time.time() - started,
        )
        validate_artifact(artifact)
        _write_json(Path(artifact_path), artifact)
        return artifact

    artifact = run_live_smoke(preconditions)  # pragma: no cover - live 53GB model path.
    artifact["duration_s"] = round(time.time() - started, 6)  # pragma: no cover
    validate_artifact(artifact)  # pragma: no cover
    _write_json(Path(artifact_path), artifact)  # pragma: no cover
    return artifact  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by integration command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    args = parser.parse_args(argv)
    artifact = run(artifact_path=args.artifact, cache_dir=args.cache_dir)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
