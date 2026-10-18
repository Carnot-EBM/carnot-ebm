"""Exp 2399 FST live-path A/B/C runner.

This module is deliberately local-first.  It tries the mandated GGUF
repositories through llama.cpp, then a cached transformers model, then the
validated live telemetry JSONL.  Every successful path still runs Carnot's
FST verifier-output prefix flow so the artifact records the terminal-prefix
behavior rather than only raw text generation.

Spec: REQ-FST-2399, SCENARIO-FST-2399.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from carnot.pipeline.verify_repair import VerifyRepairPipeline
from carnot.training.fast_slow import FastSlowTrainer

JsonDict = dict[str, Any]
LlamaFactory = Callable[..., Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = "experiment_2399_fst_live_path_ab.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_TELEMETRY_PATH = REPO_ROOT / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl"
HF_CACHE_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

EXPERIMENT = "2399_fst_live_path_ab"
SCHEMA = "fst_live_path_ab_v1"
RUN_DATE = "20260518"
MIN_TEST_PROMPTS = 3
MAX_TEST_PROMPTS = 5

MANDATED_SOTA_GGUF_MODELS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "fst_live_validated",
    "live_path_used",
    "first_live_generation_text",
    "path_a_attempted",
    "path_a_blocked_reason",
    "path_b_attempted",
    "model_used",
    "n_test_prompts",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix required. complete: even if only PATH C succeeded.",
    "fst_live_validated": "True if FST pipeline ran end-to-end on ANY path (A, B, or C).",
    "live_path_used": "Which path succeeded: 'A_gguf', 'B_transformers', or 'C_cached'.",
    "first_live_generation_text": (
        "First 200 chars of generated text (PATH A/B) or cached response (PATH C)."
    ),
    "path_a_attempted": "True if PATH A was tried.",
    "path_a_blocked_reason": (
        "Why PATH A failed or was skipped (model_not_cached, llama_cpp_missing, etc)."
    ),
    "path_b_attempted": "True if PATH B was tried.",
    "model_used": "Which GGUF or transformers model was loaded (null if PATH C).",
    "n_test_prompts": "Number of prompts run through FST pipeline.",
    "duration_s": "Guards against fabrication -- live inference takes significant wall time.",
    "preconditions_checked": (
        "Records GGUF cache check, llama_cpp check, transformers check, PATH C check."
    ),
}


@dataclass(frozen=True)
class FSTCase:
    """One prompt/response pair to feed through the FST verifier-prefix flow."""

    case_id: str
    prompt: str
    response_text: str
    source: str


@dataclass(frozen=True)
class PathAttempt:
    """Outcome for one live or cached path attempt."""

    attempted: bool
    success: bool
    path_used: str | None
    model_used: str | None
    blocked_reason: str | None
    first_text: str
    rows: tuple[JsonDict, ...]
    details: JsonDict


class _NoopAndComposeVerifier:
    """Small verifier stub that avoids constructing the default k=5 ensemble."""

    def verify(self, _question: str, _response: str) -> SimpleNamespace:
        return SimpleNamespace(
            verified=True,
            k=0,
            per_verifier_scores={},
            per_verifier_verified={},
        )


def build_live_test_prompts(n_prompts: int = MIN_TEST_PROMPTS) -> tuple[str, ...]:
    """REQ-FST-2399: build the bounded prompt set for PATH A and PATH B."""

    prompts = (
        "Verify claim: 2 + 5 = 7. Return 1 if true, 0 if false. Return the final integer only.",
        "Mia has 1 marble and gets 2 more. What is the final answer? Return the final integer only.",
        "Constraint x=2 satisfies x+3=5. Return 1 if true, 0 if false. Return the final integer only.",
        "Verify claim: 4 * 3 = 11. Return 1 if true, 0 if false. Return the final integer only.",
        "A bus has 9 riders, 4 get off, and 6 get on. What is the final answer? Return the final integer only.",
    )
    return prompts[: _bounded_prompt_count(n_prompts)]


def check_preconditions(
    *,
    cache_root: Path | str = HF_CACHE_ROOT,
    telemetry_path: Path | str = DEFAULT_TELEMETRY_PATH,
) -> JsonDict:
    """Check the exact PATH A/B/C availability signals requested for Exp 2399."""

    cache = Path(cache_root)
    telemetry = Path(telemetry_path)
    cache_matches = _cached_model_entries(cache)
    llama_available, llama_detail = _import_check("llama_cpp", "Llama")
    transformers_available, transformers_detail = _transformers_check()
    resolved_ggufs = _resolved_mandated_ggufs(cache)
    transformers_candidate = _find_cached_transformers_model(cache)

    return {
        "gguf_cache_check": {
            "command": (
                "ls ~/.cache/huggingface/hub/ | grep -i "
                '"qwen3.6\\|gemma-4-26b\\|gemma-4-31b" | head -5'
            ),
            "cache_root": str(cache),
            "matches": cache_matches,
            "available": bool(cache_matches),
        },
        "resolved_gguf_models": resolved_ggufs,
        "llama_cpp_check": {
            "command": (
                '.venv/bin/python -c "from llama_cpp import Llama; '
                'print(\'llama_cpp OK\')" 2>/dev/null || echo "llama_cpp_missing"'
            ),
            "available": llama_available,
            **llama_detail,
        },
        "transformers_check": {
            "command": (
                '.venv/bin/python -c "import transformers; print(transformers.__version__)" '
                '2>/dev/null || echo "transformers_missing"'
            ),
            "available": transformers_available,
            **transformers_detail,
            "cached_model_candidate": transformers_candidate,
        },
        "path_c_telemetry_check": {
            "command": f"ls {telemetry}",
            "path": str(telemetry),
            "exists": telemetry.is_file(),
        },
    }


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    telemetry_path: Path | str = DEFAULT_TELEMETRY_PATH,
    cache_root: Path | str = HF_CACHE_ROOT,
    n_test_prompts: int = MIN_TEST_PROMPTS,
    llama_factory: LlamaFactory | None = None,
    force_path_c: bool = False,
) -> JsonDict:
    """Run PATH A, then PATH B, then PATH C, and write the terminal artifact."""

    started = time.perf_counter()
    preconditions = check_preconditions(cache_root=cache_root, telemetry_path=telemetry_path)
    attempts: list[JsonDict] = []

    path_a = PathAttempt(
        attempted=False,
        success=False,
        path_used=None,
        model_used=None,
        blocked_reason="forced_path_c" if force_path_c else None,
        first_text="",
        rows=(),
        details={},
    )
    if not force_path_c:
        path_a = attempt_path_a(
            preconditions=preconditions,
            n_test_prompts=n_test_prompts,
            llama_factory=llama_factory,
        )
    attempts.append(_attempt_to_dict("A_gguf", path_a))

    path_b = PathAttempt(
        attempted=False,
        success=False,
        path_used=None,
        model_used=None,
        blocked_reason=None,
        first_text="",
        rows=(),
        details={},
    )
    if not path_a.success and not force_path_c:
        path_b = attempt_path_b(preconditions=preconditions, n_test_prompts=n_test_prompts)
    attempts.append(_attempt_to_dict("B_transformers", path_b))

    if path_a.success:
        selected = path_a
    elif path_b.success:
        selected = path_b
    else:
        path_c = attempt_path_c(telemetry_path=telemetry_path, n_test_prompts=n_test_prompts)
        attempts.append(_attempt_to_dict("C_cached", path_c))
        selected = path_c

    duration_s = round(time.perf_counter() - started, 3)
    artifact = build_artifact(
        selected=selected,
        path_a=path_a,
        path_b=path_b,
        attempts=attempts,
        preconditions_checked=preconditions,
        duration_s=duration_s,
    )
    validate_artifact(artifact)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def attempt_path_a(
    *,
    preconditions: Mapping[str, Any],
    n_test_prompts: int,
    llama_factory: LlamaFactory | None = None,
) -> PathAttempt:
    """Try PATH A with llama.cpp and the first resolved mandated GGUF."""

    cache_ok = bool(preconditions["gguf_cache_check"]["available"])
    llama_ok = bool(preconditions["llama_cpp_check"]["available"])
    resolved = list(preconditions.get("resolved_gguf_models", []))
    if not cache_ok:
        return _blocked_attempt(False, "model_not_cached")
    if not llama_ok:
        return _blocked_attempt(False, "llama_cpp_missing")
    if not resolved:
        return _blocked_attempt(False, "gguf_file_not_resolved")

    model = resolved[0]
    model_path = str(model["path"])
    model_used = f"{model['hf_id']}:{Path(model_path).name}"
    prompts = build_live_test_prompts(n_test_prompts)

    try:
        _ensure_cuda_library_path()
        if llama_factory is None:  # pragma: no cover - real live path.
            from llama_cpp import Llama  # type: ignore[import]  # noqa: PLC0415

            llama_factory = Llama
        llm = llama_factory(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=512,
            verbose=False,
        )
        try:
            cases = tuple(
                FSTCase(
                    case_id=f"path_a_{index + 1:02d}",
                    prompt=prompt,
                    response_text=_completion_text(
                        llm(
                            prompt,
                            max_tokens=64,
                            temperature=0.0,
                            echo=False,
                            stop=["</s>", "<eos>"],
                        )
                    ),
                    source="llama_cpp_live_gguf",
                )
                for index, prompt in enumerate(prompts)
            )
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()

        rows = run_fst_cases(cases)
        return PathAttempt(
            attempted=True,
            success=_fst_rows_valid(rows),
            path_used="A_gguf",
            model_used=model_used,
            blocked_reason=None if _fst_rows_valid(rows) else "fst_pipeline_failed",
            first_text=_clamp(cases[0].response_text, 200) if cases else "",
            rows=tuple(rows),
            details={"model_path": model_path, "n_generation_prompts": len(prompts)},
        )
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
        return PathAttempt(
            attempted=True,
            success=False,
            path_used=None,
            model_used=model_used,
            blocked_reason=f"llama_cpp_generation_failed:{type(exc).__name__}: {exc}",
            first_text="",
            rows=(),
            details={"model_path": model_path},
        )


def attempt_path_b(
    *,
    preconditions: Mapping[str, Any],
    n_test_prompts: int,
) -> PathAttempt:
    """Try PATH B with cached transformers AutoModelForCausalLM."""

    transformers_ok = bool(preconditions["transformers_check"]["available"])
    if not transformers_ok:
        return _blocked_attempt(False, "transformers_missing")

    candidate = preconditions["transformers_check"].get("cached_model_candidate")
    if not candidate:
        return _blocked_attempt(True, "no_cached_transformers_model")

    prompts = build_live_test_prompts(n_test_prompts)
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        model_ref = str(candidate["path"])
        tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_ref, local_files_only=True)
        model.eval()
        cases: list[FSTCase] = []
        for index, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            cases.append(
                FSTCase(
                    case_id=f"path_b_{index + 1:02d}",
                    prompt=prompt,
                    response_text=str(text),
                    source="transformers_live_automodel",
                )
            )
        rows = run_fst_cases(cases)
        return PathAttempt(
            attempted=True,
            success=_fst_rows_valid(rows),
            path_used="B_transformers",
            model_used=str(candidate.get("model_id") or candidate["path"]),
            blocked_reason=None if _fst_rows_valid(rows) else "fst_pipeline_failed",
            first_text=_clamp(cases[0].response_text, 200) if cases else "",
            rows=tuple(rows),
            details={"model_path": model_ref, "n_generation_prompts": len(prompts)},
        )
    except Exception as exc:  # pragma: no cover - optional model dependent.
        return PathAttempt(
            attempted=True,
            success=False,
            path_used=None,
            model_used=str(candidate.get("model_id") or candidate.get("path")),
            blocked_reason=f"transformers_generation_failed:{type(exc).__name__}: {exc}",
            first_text="",
            rows=(),
            details={"candidate": candidate},
        )


def attempt_path_c(
    *,
    telemetry_path: Path | str,
    n_test_prompts: int,
) -> PathAttempt:
    """Run the FST flow on cached telemetry rows when live inference is blocked."""

    telemetry = Path(telemetry_path)
    if not telemetry.is_file():
        return _blocked_attempt(False, "telemetry_missing")
    cases = load_cached_telemetry_cases(telemetry, n_test_prompts=n_test_prompts)
    if not cases:
        return _blocked_attempt(True, "telemetry_has_no_usable_rows")
    rows = run_fst_cases(cases)
    return PathAttempt(
        attempted=True,
        success=_fst_rows_valid(rows),
        path_used="C_cached",
        model_used=None,
        blocked_reason=None if _fst_rows_valid(rows) else "fst_pipeline_failed",
        first_text=_clamp(cases[0].response_text, 200),
        rows=tuple(rows),
        details={"telemetry_path": str(telemetry), "n_cached_rows": len(cases)},
    )


def load_cached_telemetry_cases(
    telemetry_path: Path | str,
    *,
    n_test_prompts: int = MIN_TEST_PROMPTS,
) -> tuple[FSTCase, ...]:
    """SCENARIO-FST-2399: read cached live telemetry as PATH C prompt pairs."""

    cases: list[FSTCase] = []
    for line in Path(telemetry_path).read_text(encoding="utf-8").splitlines():
        if len(cases) >= _bounded_prompt_count(n_test_prompts):
            break
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = row.get("prompt")
        response = row.get("response_text") or row.get("model_output")
        if not isinstance(prompt, str) or not isinstance(response, str) or not response.strip():
            continue
        cases.append(
            FSTCase(
                case_id=str(row.get("case_id") or f"cached_{len(cases) + 1:02d}"),
                prompt=prompt,
                response_text=response,
                source="cached_live_sota_telemetry",
            )
        )
    return tuple(cases)


def run_fst_cases(cases: Sequence[FSTCase]) -> list[JsonDict]:
    """Run Carnot verification plus FST terminal-prefix construction."""

    pipeline = VerifyRepairPipeline(
        timeout_seconds=10.0,
        max_repairs=1,
        and_compose_verifier=_NoopAndComposeVerifier(),
    )
    trainer = FastSlowTrainer.from_pipeline(pipeline)
    rows: list[JsonDict] = []
    for index, case in enumerate(cases, 1):
        verification = pipeline.verify(
            case.prompt,
            case.response_text,
            use_fst=True,
            fst_trainer=trainer,
        )
        base_repair_prompt = (
            f"Question: {case.prompt}\n\n"
            f"Your previous answer:\n{case.response_text}\n\n"
            "Please provide a corrected answer that fixes any verifier issues."
        )
        prefixed_prompt = trainer.next_repair_prompt(
            verification_result=verification,
            base_prompt=base_repair_prompt,
            iteration=index,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "source": case.source,
                "prompt": case.prompt,
                "response_text": case.response_text,
                "verified": bool(verification.verified),
                "energy": float(verification.energy),
                "n_violations": len(verification.violations),
                "fst_terminal_prefix_present": prefixed_prompt.startswith(
                    "FST verifier-output summary:"
                ),
                "fst_prompt_prefix_text": prefixed_prompt[:200],
                "fst_certificate": trainer.certificate(),
            }
        )
    return rows


def build_artifact(
    *,
    selected: PathAttempt,
    path_a: PathAttempt,
    path_b: PathAttempt,
    attempts: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Build the Exp 2399 deliverable payload."""

    validated = bool(selected.success and selected.rows)
    path_used = selected.path_used if validated else None
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "status": "complete" if validated else "blocked",
        "title": "FST live inference PATH A/B with cached telemetry fallback",
        "honest_verdict": (
            f"complete: fst_live_validated_via_{path_used}"
            if validated
            else "blocked: blocked_all_paths"
        ),
        "fst_live_validated": validated,
        "live_path_used": path_used,
        "first_live_generation_text": selected.first_text,
        "path_a_attempted": bool(path_a.attempted),
        "path_a_blocked_reason": path_a.blocked_reason,
        "path_b_attempted": bool(path_b.attempted),
        "path_b_blocked_reason": path_b.blocked_reason,
        "model_used": selected.model_used,
        "n_test_prompts": len(selected.rows),
        "duration_s": duration_s,
        "preconditions_checked": preconditions_checked,
        "path_attempts": list(attempts),
        "fst_rows": list(selected.rows),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gates": {
            "fst_live_validated == true": validated,
        },
        "source_notes": {
            "requested_existing_pipeline": "python/carnot/pipeline/fst_pipeline.py",
            "requested_exp2365_artifact": "results/experiment_2365_fst_live_gen.json",
            "local_checkout_note": (
                "Neither requested file exists in this checkout; Exp 2399 uses "
                "VerifyRepairPipeline plus carnot.training.fast_slow.FastSlowTrainer."
            ),
        },
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 2399 schema fields and acceptance gate."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required artifact fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:", "failed:")):
        raise AssertionError("honest_verdict must use a terminal prefix")
    if artifact["fst_live_validated"] is not True:
        raise AssertionError("fst_live_validated must be true for the Exp 2399 acceptance gate")
    if artifact["live_path_used"] not in {"A_gguf", "B_transformers", "C_cached"}:
        raise AssertionError("live_path_used must name the successful path")
    if artifact["live_path_used"] == "C_cached" and artifact["model_used"] is not None:
        raise AssertionError("PATH C artifact must set model_used to null")
    if not isinstance(artifact["first_live_generation_text"], str):
        raise AssertionError("first_live_generation_text must be a string")
    if len(artifact["first_live_generation_text"]) > 200:
        raise AssertionError("first_live_generation_text must be capped at 200 chars")
    rows = artifact.get("fst_rows")
    if not isinstance(rows, list) or not rows:
        raise AssertionError("fst_rows must contain at least one FST run")
    if artifact["n_test_prompts"] != len(rows):
        raise AssertionError("n_test_prompts must equal len(fst_rows)")
    if not all(row.get("fst_terminal_prefix_present") is True for row in rows):
        raise AssertionError("every FST row must include the terminal prefix")


def _attempt_to_dict(label: str, attempt: PathAttempt) -> JsonDict:
    return {
        "path": label,
        "attempted": attempt.attempted,
        "success": attempt.success,
        "model_used": attempt.model_used,
        "blocked_reason": attempt.blocked_reason,
        "n_rows": len(attempt.rows),
        "details": attempt.details,
    }


def _blocked_attempt(attempted: bool, reason: str) -> PathAttempt:
    return PathAttempt(
        attempted=attempted,
        success=False,
        path_used=None,
        model_used=None,
        blocked_reason=reason,
        first_text="",
        rows=(),
        details={},
    )


def _bounded_prompt_count(n_prompts: int) -> int:
    return max(MIN_TEST_PROMPTS, min(MAX_TEST_PROMPTS, int(n_prompts)))


def _cached_model_entries(cache_root: Path) -> list[str]:
    if not cache_root.is_dir():
        return []
    needles = ("qwen3.6", "gemma-4-26b", "gemma-4-31b")
    matches = [
        path.name for path in cache_root.iterdir() if any(n in path.name.lower() for n in needles)
    ]
    return sorted(matches)[:5]


def _resolved_mandated_ggufs(cache_root: Path) -> list[JsonDict]:
    resolved: list[JsonDict] = []
    for hf_id in MANDATED_SOTA_GGUF_MODELS:
        model_dir = cache_root / f"models--{hf_id.replace('/', '--')}"
        candidates = _usable_gguf_candidates(model_dir)
        if not candidates:
            continue
        selected = _pick_gguf(candidates)
        resolved.append(
            {
                "hf_id": hf_id,
                "path": str(selected),
                "filename": selected.name,
                "size_bytes": selected.stat().st_size,
            }
        )
    return resolved


def _usable_gguf_candidates(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        return []
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    candidates: list[Path] = []
    for path in snapshots.rglob("*.gguf"):
        name = path.name.lower()
        if name.startswith("mmproj") or name.startswith("imatrix"):
            continue
        if "-of-" in name:
            continue
        if path.is_file():
            candidates.append(path)
    return candidates


def _pick_gguf(candidates: Sequence[Path]) -> Path:
    preference = (
        "UD-IQ2_XXS",
        "UD-IQ2_M",
        "UD-Q2_K_XL",
        "UD-Q3_K_M",
        "UD-Q4_K_M",
        "Q4_K_M",
        "Q8_0",
    )
    for token in preference:
        for candidate in candidates:
            if token.lower() in candidate.name.lower():
                return candidate
    return min(candidates, key=lambda path: path.stat().st_size)


def _import_check(module_name: str, symbol: str) -> tuple[bool, JsonDict]:
    try:
        module = __import__(module_name, fromlist=[symbol])
        getattr(module, symbol)
    except Exception as exc:
        return False, {"output": f"{module_name}_missing", "error": f"{type(exc).__name__}: {exc}"}
    return True, {"output": f"{module_name} OK"}


def _transformers_check() -> tuple[bool, JsonDict]:
    try:
        import transformers  # noqa: PLC0415
    except Exception as exc:
        return False, {"output": "transformers_missing", "error": f"{type(exc).__name__}: {exc}"}
    return True, {"output": str(transformers.__version__), "version": str(transformers.__version__)}


def _find_cached_transformers_model(cache_root: Path) -> JsonDict | None:
    env_model = os.environ.get("CARNOT_TRANSFORMERS_MODEL")
    if env_model:
        return {"model_id": env_model, "path": env_model, "source": "CARNOT_TRANSFORMERS_MODEL"}
    if not cache_root.is_dir():
        return None

    preferred_model_ids = (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3.5-0.8B",
        "LiquidAI/LFM2.5-350M",
        "microsoft/Phi-tiny-MoE-instruct",
        "google/gemma-4-E2B-it",
    )
    for model_id in preferred_model_ids:
        hit = _cached_transformers_snapshot(cache_root, model_id)
        if hit is not None:
            return hit

    candidates: list[tuple[int, str, Path]] = []
    for model_dir in cache_root.glob("models--*"):
        if model_dir.name.endswith("-GGUF") or "gguf" in model_dir.name.lower():
            continue
        lowered_name = model_dir.name.lower()
        if any(
            skip in lowered_name
            for skip in ("sentence-transformers", "cross-encoder", "roberta", "deberta", "whisper")
        ):
            continue
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
                continue
            if not _looks_like_causal_lm(snapshot / "config.json"):
                continue
            weights = list(snapshot.glob("*.safetensors")) + list(
                snapshot.glob("pytorch_model*.bin")
            )
            if not weights:
                continue
            size = sum(path.stat().st_size for path in weights if path.is_file())
            model_id = model_dir.name.removeprefix("models--").replace("--", "/")
            candidates.append((size, model_id, snapshot))
    if not candidates:
        return None
    size, model_id, snapshot = min(candidates, key=lambda item: item[0])
    return {
        "model_id": model_id,
        "path": str(snapshot),
        "size_bytes": size,
        "source": "huggingface_cache_snapshot",
    }


def _cached_transformers_snapshot(cache_root: Path, model_id: str) -> JsonDict | None:
    model_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for snapshot in snapshots.iterdir():
        if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
            continue
        weights = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("pytorch_model*.bin"))
        if not weights:
            continue
        size = sum(path.stat().st_size for path in weights if path.is_file())
        candidates.append((size, snapshot))
    if not candidates:
        return None
    size, snapshot = min(candidates, key=lambda item: item[0])
    return {
        "model_id": model_id,
        "path": str(snapshot),
        "size_bytes": size,
        "source": "preferred_huggingface_cache_snapshot",
    }


def _looks_like_causal_lm(config_path: Path) -> bool:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    architectures = [str(item).lower() for item in config.get("architectures", [])]
    model_type = str(config.get("model_type", "")).lower()
    if any("causallm" in item or "forcausallm" in item for item in architectures):
        return True
    return model_type in {"qwen2", "qwen3", "gemma", "gpt2", "phi", "lfm2"}


def _completion_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if not isinstance(result, Mapping):
        return ""
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    text = first.get("text")
    return text.strip() if isinstance(text, str) else ""


def _clamp(text: str, max_chars: int) -> str:
    text = str(text)
    return text[:max_chars]


def _fst_rows_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(row.get("fst_terminal_prefix_present") is True for row in rows)


def _ensure_cuda_library_path() -> None:
    site_packages = sorted((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))
    candidates: list[str] = []
    for site in site_packages:
        candidates.extend(
            [
                str(site / "nvidia" / "cuda_runtime" / "lib"),
                str(site / "nvidia" / "cublas" / "lib"),
            ]
        )
    current_parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    repaired: list[str] = []
    seen: set[str] = set()
    for path in [*candidates, *current_parts]:
        if path in seen or not Path(path).is_dir():
            continue
        seen.add(path)
        repaired.append(path)
    if repaired:
        os.environ["LD_LIBRARY_PATH"] = ":".join(repaired)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY_PATH)
    parser.add_argument("--n-test-prompts", type=int, default=MIN_TEST_PROMPTS)
    parser.add_argument("--force-path-c", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(
        output_path=args.output,
        telemetry_path=args.telemetry,
        n_test_prompts=args.n_test_prompts,
        force_path_c=args.force_path_c,
    )
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "fst_live_validated": artifact["fst_live_validated"],
                "live_path_used": artifact["live_path_used"],
                "n_test_prompts": artifact["n_test_prompts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
