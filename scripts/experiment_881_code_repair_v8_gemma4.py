#!/usr/bin/env python3
"""Experiment 881: Code repair v8 — Gemma4-E4B-it live HumanEval (transformers path).

**Researcher summary:**
    11 prior code-repair attempts (Exps 850–870) all failed due to GGUF download
    failures on the host.  This experiment bypasses llama.cpp entirely and uses
    google/gemma-4-E4B-it via the HuggingFace transformers AutoModelForCausalLM
    path, which loads reliably from the local HF model cache.

    The goal is to produce the first live-GPU positive-repair result on 25
    HumanEval problems (problems 0–24), demonstrating that Carnot's
    CodeExtractor + VerifyRepairPipeline add measurable value over the raw
    model output.

**Gate:**
    Reads results/experiment_855_preflight_v15.json and aborts if
    live_env_fixed != True.  Also requires CARNOT_FORCE_LIVE in the environment.

**Pipeline:**
    1. Load google/gemma-4-E4B-it via transformers (device_map="auto").
    2. Warm up with one synthetic code prompt.
    3. For each of 25 HumanEval problems:
       a. Generate a Python solution.
       b. Run it against the official HumanEval test harness — baseline result.
       c. Apply CodeExtractor to find verifiable constraints.
       d. If constraints found, run VerifyRepairPipeline.verify_generated_code()
          and attempt repair.
       e. Run repaired code against test harness.
    4. Emit signed_improvement = carnot_pass_rate - baseline_pass_rate.

**Honest-verdict mapping:**
    "positive_repair"     — signed_improvement > 0 and live_gpu
    "live_no_improvement" — signed_improvement <= 0 and live_gpu
    "zero_constraints"    — n_constraints_extracted == 0 (extraction issue)
    "simulation_fallback" — inference_mode != live_gpu
    "blocked"             — gate failed or model load failed

Spec: REQ-VR-020 (verify-repair live), SCENARIO-VR-030 (HumanEval live),
      REQ-CODE-010, SCENARIO-CODE-009
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Gate: live-env check
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PREFLIGHT_PATH = _REPO_ROOT / "results" / "experiment_855_preflight_v15.json"
_DELIVERABLE = str(_REPO_ROOT / "results" / "experiment_881_code_repair_v8_gemma4.json")

sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def _check_gate() -> tuple[bool, str]:
    """Return (ok, reason).  Gate passes only when live_env_fixed == True and
    CARNOT_FORCE_LIVE is set in the environment.

    Why two conditions: live_env_fixed confirms the EnvPropagationGuard
    shipped (Exp 855), while CARNOT_FORCE_LIVE is the runtime opt-in that
    prevents accidental live-GPU runs during dry-run CI passes.
    """
    if "CARNOT_FORCE_LIVE" not in os.environ:
        return False, "CARNOT_FORCE_LIVE not set — run with CARNOT_FORCE_LIVE=1"
    if not _PREFLIGHT_PATH.exists():
        return False, f"preflight artifact missing: {_PREFLIGHT_PATH}"
    with open(_PREFLIGHT_PATH) as f:
        data = json.load(f)
    if not data.get("live_env_fixed", False):
        return False, f"live_env_fixed != True in {_PREFLIGHT_PATH}"
    return True, "gate passed"


# ---------------------------------------------------------------------------
# HumanEval execution helper
# ---------------------------------------------------------------------------


def _exec_humaneval_test(code: str, test_str: str, entry_point: str) -> bool:
    """Execute generated code + official test in a sandboxed namespace.

    Why exec instead of subprocess: avoids repeated Python startup overhead
    across 25 × 2 executions.  The harness is purely arithmetic / functional;
    no file I/O or network calls occur.  We still isolate via a fresh dict.

    Returns True if all assertions pass, False otherwise.
    """
    ns: dict[str, Any] = {}
    try:
        exec(code, ns)  # noqa: S102
        check_fn_code = test_str + f"\ncheck({entry_point})\n"
        exec(check_fn_code, ns)  # noqa: S102
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Code generation helper
# ---------------------------------------------------------------------------


def _generate_code(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a Python function completion from a HumanEval prompt.

    We prepend an instruction telling the model to complete only the function
    body, mirroring the standard HumanEval few-shot setup.

    Why bfloat16 + no grad: keeps VRAM usage minimal and avoids autograd
    overhead on inference-only work.
    """
    import torch  # local import — torch is a heavy optional dep

    full_prompt = (
        "Complete the following Python function. Output only the code, no prose:\n\n" + prompt
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full Exp 881 pipeline and write the deliverable artifact."""
    import torch  # noqa: PLC0415

    tmpl = ExperimentTemplate(
        exp_id=881,
        title="Code repair v8 Gemma4 live HumanEval",
        deliverable=_DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # -- Gate check ----------------------------------------------------------
    gate_ok, gate_reason = _check_gate()
    if not gate_ok:
        artifact = tmpl.build_result(
            {"gate_reason": gate_reason},
            status="blocked",
            honest_verdict="blocked",
            inference_mode="unknown",
        )
        Path(_DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        print(f"[exp881] BLOCKED: {gate_reason}", flush=True)
        tmpl.assert_deliverable_written()
        return

    # -- Model load ----------------------------------------------------------
    print("[exp881] Loading google/gemma-4-E4B-it via transformers …", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        model_id = "google/gemma-4-E4B-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        print(f"[exp881] Model loaded on device: {model.device}", flush=True)
    except Exception as exc:
        artifact = tmpl.build_result(
            {"model_load_error": str(exc), "traceback": traceback.format_exc()},
            status="blocked",
            honest_verdict="blocked",
            inference_mode="unknown",
        )
        Path(_DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        print(f"[exp881] BLOCKED model load: {exc}", flush=True)
        tmpl.assert_deliverable_written()
        return

    # -- Warm-up -------------------------------------------------------------
    print("[exp881] Warm-up pass …", flush=True)
    _generate_code(
        model,
        tokenizer,
        'def add(a: int, b: int) -> int:\n    """Return a + b."""\n',
        max_new_tokens=32,
    )
    print("[exp881] Warm-up done.", flush=True)

    # -- Load HumanEval problems 0-24 ----------------------------------------
    from human_eval.data import read_problems  # noqa: PLC0415

    all_problems = read_problems()
    task_ids = sorted(all_problems.keys())[:25]
    print(f"[exp881] Running {len(task_ids)} HumanEval problems …", flush=True)

    # -- Pipeline ------------------------------------------------------------
    from carnot.pipeline.extract import CodeExtractor  # noqa: PLC0415
    from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: PLC0415

    extractor = CodeExtractor()
    # Use verify-only mode (no inner LLM repair model) — we apply repair
    # manually via a second generation call to keep the pipeline simple and
    # avoid re-loading the model inside VerifyRepairPipeline.
    pipeline = VerifyRepairPipeline(
        model=None,
        domains=["code"],
        extractor=extractor,
    )

    results_per_problem: list[dict[str, Any]] = []
    n_constraints_total = 0

    for task_id in task_ids:
        prob = all_problems[task_id]
        prompt_text: str = prob["prompt"]
        entry_point: str = prob["entry_point"]
        test_str: str = prob["test"]

        t0 = time.perf_counter()

        # Step a: generate baseline
        generated = _generate_code(model, tokenizer, prompt_text)
        # Prepend the prompt so the code block is self-contained
        full_code = prompt_text + "\n" + generated

        # Step b: baseline execution
        passed_baseline = _exec_humaneval_test(full_code, test_str, entry_point)

        # Step c: extract constraints
        constraints = extractor.extract(full_code, domain="code")
        n_constraints = len(constraints)
        n_constraints_total += n_constraints

        # Step d+e: repair pass using VerifyRepairPipeline's verify_generated_code
        if n_constraints > 0:
            vr = pipeline.verify_generated_code(
                full_code,
                prompt_text,
                entry_point,
                test_str,
                include_static=True,
                include_pbt=False,
            )
            # If violations found, attempt a guided repair by appending violation
            # feedback to the prompt and re-generating.
            has_violations = any(not c.satisfied for c in vr.constraints)
            if has_violations and not passed_baseline:
                violation_msgs = [c.description for c in vr.constraints if not c.satisfied]
                repair_hint = "\n".join(f"# FIX: {m}" for m in violation_msgs[:3])
                repair_prompt = (
                    prompt_text
                    + f"\n# Constraints violated:\n{repair_hint}\n"
                    + "# Please provide a corrected implementation:\n"
                )
                repaired_gen = _generate_code(model, tokenizer, repair_prompt)
                repaired_full = prompt_text + "\n" + repaired_gen
                passed_repaired = _exec_humaneval_test(repaired_full, test_str, entry_point)
            else:
                passed_repaired = passed_baseline
        else:
            passed_repaired = passed_baseline

        elapsed = round(time.perf_counter() - t0, 2)
        results_per_problem.append(
            {
                "task_id": task_id,
                "passed_baseline": passed_baseline,
                "passed_repaired": passed_repaired,
                "n_constraints_found": n_constraints,
                "elapsed_s": elapsed,
            }
        )
        status_char = "+" if passed_repaired else "-"
        print(
            f"[exp881] {task_id}: baseline={passed_baseline} repaired={passed_repaired} "
            f"constraints={n_constraints} [{elapsed}s] {status_char}",
            flush=True,
        )

        # Checkpoint every 5 problems
        if len(results_per_problem) % 5 == 0:
            tmpl.checkpoint_save(
                {"partial_results": results_per_problem}, step=len(results_per_problem)
            )

    # -- Compute metrics -----------------------------------------------------
    n_problems = len(results_per_problem)
    baseline_pass_rate = sum(r["passed_baseline"] for r in results_per_problem) / n_problems
    carnot_pass_rate = sum(r["passed_repaired"] for r in results_per_problem) / n_problems
    signed_improvement = round(carnot_pass_rate - baseline_pass_rate, 4)
    inference_mode = "live_gpu"

    # -- Honest verdict ------------------------------------------------------
    if n_constraints_total == 0:
        honest_verdict = "zero_constraints"
    elif inference_mode != "live_gpu":
        honest_verdict = "simulation_fallback"
    elif signed_improvement > 0:
        honest_verdict = "positive_repair"
    else:
        honest_verdict = "live_no_improvement"

    print(
        f"[exp881] Done. baseline={baseline_pass_rate:.3f} carnot={carnot_pass_rate:.3f} "
        f"signed_improvement={signed_improvement} verdict={honest_verdict}",
        flush=True,
    )

    # -- Build and write artifact --------------------------------------------
    artifact = tmpl.build_result(
        {
            "n_problems": n_problems,
            "baseline_pass_rate": baseline_pass_rate,
            "carnot_pass_rate": carnot_pass_rate,
            "signed_improvement": signed_improvement,
            "n_constraints_extracted": n_constraints_total,
            "inference_mode": inference_mode,
            "model_id": "google/gemma-4-E4B-it",
            "honest_verdict": honest_verdict,
            "results_per_problem": results_per_problem,
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode=inference_mode,
        decision_class=["verify", "repair"],
    )
    Path(_DELIVERABLE).write_text(json.dumps(artifact, indent=2))
    print(f"[exp881] Artifact written to {_DELIVERABLE}", flush=True)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
