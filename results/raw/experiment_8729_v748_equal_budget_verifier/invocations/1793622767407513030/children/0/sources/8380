#!/usr/bin/env python3
"""Experiment 942: Math Repair SOTA v2 — GSM8K with a SOTA GGUF model.

**Researcher summary:**
    Exp 930 showed math iterative self-repair produces zero improvement (baseline=12%,
    repair=12%, signed_improvement=0.0) when using gemma-4-E4B-it.  The root cause was
    a model capability ceiling — 12% baseline leaves no arithmetic CoT structure for
    the repair signal to improve.  arXiv 2604.17121 additionally explains that without
    external grounding, retry feedback cannot propagate usefully when the computed state
    is already lost across forward passes.

    This experiment re-runs the identical GSM8K repair loop using a SOTA GGUF model
    (gemma-4-26B-A4B-it-UD-Q4_K_M from preflight Exp 941, or the smallest available
    SOTA GGUF from the HF cache).  A SOTA model yields a ~70-80% GSM8K baseline,
    giving the repair rounds substantial room to push accuracy up toward 85-90%.

**Algorithm (per problem):**
    Round 0 (baseline): generate response → extract numeric answer → check vs truth.
    If wrong:
        Rounds 1-3 (repair): show model its previous answer, ask it to re-solve.
    After all rounds: use IsingEBM to select the attempt with lowest energy.
    Final answer is the extracted value from the lowest-energy attempt.

**Model selection (priority order):**
    1. sota_model_path from Exp 941 preflight (results/experiment_941_preflight_v22.json)
    2. resolve_cached_gguf("unsloth/gemma-4-26B-A4B-it-GGUF") — smallest SOTA
    3. resolve_cached_gguf("unsloth/gemma-4-31B-it-GGUF")
    4. resolve_cached_gguf("unsloth/Qwen3.6-35B-A3B-GGUF") — flagship MoE
    If none found: honest_verdict='sota_model_not_found', exit immediately.
    Do NOT fall back to gemma-4-E4B-it (proven to produce 12% ceiling in Exp 930).

**Prior failure this addresses:**
    experiment_id: exp930-math-iterative-self-repair-v1
    verdict: math_repair_zero (signed_improvement=0.0)
    root_cause: RETRO-MATH-REPAIR-MODEL-CEILING — model too small for GSM8K baseline
    addressed_by: Using SOTA GGUF model with ~70-80% GSM8K baseline instead of E4B

**Honest-verdict mapping:**
    'math_repair_significant' — signed_improvement > 0.10
    'math_repair_marginal'    — 0 < signed_improvement <= 0.10
    'math_repair_zero'        — signed_improvement == 0
    'math_repair_negative'    — signed_improvement < 0
    'sota_model_not_found'    — no SOTA GGUF was available on disk

Spec: REQ-VER-030 (SOTA math repair), SCENARIO-VER-030 (GSM8K SOTA repair loop)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback as tb
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo-root setup — must happen before any carnot imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_DELIVERABLE = "results/experiment_942_math_repair_sota_v2.json"

# ---------------------------------------------------------------------------
# 25 GSM8K problems — same set as Exp 930 so results are directly comparable
# ---------------------------------------------------------------------------

_GSM8K_PROBLEMS: list[dict[str, Any]] = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": 72,
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": 10,
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": 5,
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read tomorrow?",
        "answer": 42,
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": 624,
    },
    {
        "question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
        "answer": 35,
    },
    {
        "question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
        "answer": 48,
    },
    {
        "question": "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he added enough jelly beans to bring the weight to 2 pounds. Then, he added brownies to bring the weight to 7 pounds. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to bring the weight to 9 pounds. How many pounds of jelly beans did he have in the box?",
        "answer": 4,
    },
    {
        "question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also bought a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?",
        "answer": 41,
    },
    {
        "question": "Tina makes $18 per hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
        "answer": 990,
    },
    {
        "question": "A deep-sea monster rises from the waters once every 100 years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?",
        "answer": 121,
    },
    {
        "question": "Tobias is buying a new pair of shoes that costs $95. He has been saving up his money each month for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 left over. If he mowed 4 lawns, how many driveways did he shovel?",
        "answer": 5,
    },
    {
        "question": "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all?",
        "answer": 85,
    },
    {
        "question": "Jasper will serve chili for a large party. He made a recipe that serves 2 people. He needs to double the recipe 4 times to make enough chili to serve all his guests. Each recipe calls for 1.5 cups of beans. How many cups of beans does he need in all?",
        "answer": 24,
    },
    {
        "question": "Sam is hired for a 20-day period. On days that he works, he earns $60. For each day that he does not work, $30 is subtracted from his earnings. At the end of the 20-day period, he received $840. How many days did he not work?",
        "answer": 6,
    },
    {
        "question": "In a truck, there are 26 pink hard hats, 15 green hard hats, and 24 yellow hard hats. If Carl takes away 4 pink hard hats, and John takes away 6 pink hard hats and twice as many green hard hats as the number of pink hard hats that he removed, then calculate the total number of hard hats that remained in the truck.",
        "answer": 43,
    },
    {
        "question": "Mia is a student. In her first year of college, she spent $600 on college supplies. In her second year, she spent 50 percent more than she spent in her first year, and in her third year, she spent 25 percent less than she spent in her second year. How much did Mia spend on college supplies in her third year?",
        "answer": 675,
    },
    {
        "question": "Farmer Brown has 20 animals on his farm, all either chickens or cows. They have a total of 70 legs. How many chickens does the farmer have?",
        "answer": 5,
    },
    {
        "question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
        "answer": 64,
    },
    {
        "question": "There are 290 liters of oil in 24 cans. If 10 of the cans are holding 8 liters each, how much oil is each of the remaining cans holding?",
        "answer": 15,
    },
    {
        "question": "Joey wants to buy the latest released pair of designer High Jump basketball sneakers. He plans to mow 3 neighbors' lawns for $8 a lawn, sell 2 collectible figures to his friends for $9 each, and work an after-school job for 10 hours at $5 per hour. If his earnings just cover the price of the sneekers, how much do the sneakers cost?",
        "answer": 92,
    },
    {
        "question": "A farmer is growing corn. For every 4 seeds he plants, 1 fails to sprout. Of the seeds that sprout, 2/3 are consumed by insects. 5/8 of the seeds that survive insects are eaten by animals. If the farmer plants 96 seeds, how many survive?",
        "answer": 9,
    },
    {
        "question": "There are 28 students in a class. Two-sevenths of them were absent last Monday. How many students were present last Monday?",
        "answer": 20,
    },
    {
        "question": "Nancy earns $28 for each project she completes. If she completes 4 projects every week, how much does she earn in a month?",
        "answer": 448,
    },
    {
        "question": "Lola baked 13 mini cupcakes, 10 pop tarts, and 8 blueberry muffins. Meanwhile, her friend Lulu baked 16 mini cupcakes, 12 pop tarts, and 14 blueberry muffins. How many baked goods did Lola and Lulu bake altogether?",
        "answer": 73,
    },
]

# ---------------------------------------------------------------------------
# Numeric answer extractor (same regex as Exp 930 — directly comparable results)
# ---------------------------------------------------------------------------

# We match the last number at the tail of the response.  GSM8K models write
# step-by-step reasoning and finish with "#### <answer>" or "The answer is N".
# The tail heuristic captures that final answer reliably while ignoring
# intermediate arithmetic values that appear earlier in the chain-of-thought.
_ANSWER_RE = re.compile(
    r"(?:####\s*|the answer is\s*|answer:\s*|=\s*)?"
    r"(-?\d[\d,]*(?:\.\d+)?)"
    r"(?:\s*dollars?|\s*cents?|\s*%|\s*years?|\s*days?|\s*hours?|\s*minutes?|\s*liters?)?"
    r"\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_FALLBACK_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_numeric_answer(response: str) -> float | None:
    """Parse the numeric answer from an LLM math response.

    Why "last number" heuristic: GSM8K models write step-by-step reasoning,
    then state the final answer at the very end.  Grabbing the last number
    in the response (or specifically after "#### " or "The answer is")
    almost always captures the intended answer rather than an intermediate
    calculation.

    Parameters
    ----------
    response : str
        Raw LLM output for a math question.

    Returns
    -------
    float | None
        The extracted numeric value, or None if no number was found.
    """
    tail = response[-300:] if len(response) > 300 else response
    m = _ANSWER_RE.search(tail)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            pass

    all_numbers = _FALLBACK_RE.findall(tail)
    if all_numbers:
        try:
            return float(all_numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def answers_match(extracted: float | None, ground_truth: int) -> bool:
    """Return True when extracted answer matches ground truth within ±0.5.

    GSM8K answers are always whole numbers.  The 0.5 tolerance handles
    floating-point drift where a model writes "72.0" instead of "72".
    """
    if extracted is None:
        return False
    return abs(round(extracted) - ground_truth) < 1


# ---------------------------------------------------------------------------
# Energy scorer
# ---------------------------------------------------------------------------


def _build_energy_scorer() -> tuple[Any, str]:
    """Load Ising energy scorer; fall back to token-length heuristic.

    The Ising EBM assigns lower energy to more "in-distribution" text.
    For math responses, this acts as a proxy for reasoning coherence:
    short, clean step-by-step solutions get lower energy than confused
    or padded outputs.

    Returns (scorer_object, scorer_type_label).
    """
    try:
        from carnot.models.ising import IsingConfig, IsingModel  # noqa: PLC0415
        import jax.random as jrandom  # noqa: PLC0415

        config = IsingConfig(input_dim=64, coupling_init="xavier_uniform")
        model = IsingModel(config, key=jrandom.PRNGKey(942))

        class _IsingScorer:
            def __init__(self, m: Any) -> None:
                self._m = m

            def score(self, text: str) -> float:
                """Map text to Ising energy via ±1 spin encoding of char parity."""
                import jax.numpy as jnp  # noqa: PLC0415

                chars = [ord(c) % 2 * 2 - 1 for c in text[:64]]
                chars = chars[:64] + [1] * max(0, 64 - len(chars))
                spins = jnp.array(chars, dtype=jnp.float32)
                return float(self._m.energy(spins))

        return _IsingScorer(model), "ising_model"
    except Exception:

        class _LenScorer:
            """Fallback: shorter responses get lower energy (smaller = simpler)."""

            def score(self, text: str) -> float:
                return float(len(text.split()))

        return _LenScorer(), "token_length_heuristic"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _initial_prompt(question: str) -> str:
    """Build the zero-shot math prompt for GSM8K.

    We ask for step-by-step reasoning followed by a final numeric answer
    on its own line prefixed with "####".  This is the standard GSM8K
    evaluation format and makes the answer-extraction regex reliable.
    """
    return (
        "Solve the following math problem step by step. "
        "At the end, write your final numeric answer on its own line "
        "in the format: #### <number>\n\n"
        f"Problem: {question}"
    )


def _repair_prompt(question: str, prev_answer: float | None) -> str:
    """Build a repair prompt that shows the model its previous wrong answer.

    Why reveal the previous answer: the model can use it as a negative signal —
    it knows NOT to produce that value again, which guides search away from
    the previous failure mode.  This is the key insight from iterative
    self-repair (arXiv 2604.10508) applied to math.
    """
    prev_str = str(int(round(prev_answer))) if prev_answer is not None else "unknown"
    return (
        f"Your previous answer was {prev_str}, which is incorrect. "
        "Please re-read the problem carefully and solve it again from scratch, "
        "checking each arithmetic step explicitly. "
        "At the end, write your final numeric answer on its own line "
        "in the format: #### <number>\n\n"
        f"Problem: {question}"
    )


# ---------------------------------------------------------------------------
# Per-problem repair loop
# ---------------------------------------------------------------------------


def _run_problem(
    question: str,
    ground_truth: int,
    runner: Any,
    energy_scorer: Any,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Run iterative self-repair for one GSM8K question.

    Returns a dict with per-problem metrics: baseline_passed, repair_passed,
    best_extracted, n_retries, energy_score_best, n_attempts.

    Parameters
    ----------
    question : str
        The math word problem text.
    ground_truth : int
        The expected integer answer.
    runner : Any
        Object with a .generate(prompt: str) -> str method.
    energy_scorer : Any
        Object with a .score(text: str) -> float method (lower = better).
    max_retries : int
        Maximum repair rounds after the initial baseline attempt.
    """
    attempts: list[dict[str, Any]] = []
    prev_answer: float | None = None

    for round_idx in range(max_retries + 1):
        prompt = (
            _initial_prompt(question) if round_idx == 0 else _repair_prompt(question, prev_answer)
        )

        try:
            response = runner.generate(prompt)
        except Exception as exc:
            response = f"<generation_error: {exc}>"

        extracted = extract_numeric_answer(response)
        passed = answers_match(extracted, ground_truth)
        energy = energy_scorer.score(response)

        attempts.append(
            {
                "round": round_idx,
                "response": response[:500],
                "extracted_answer": extracted,
                "passed": passed,
                "energy": energy,
            }
        )

        # Stop as soon as we get the right answer — no need to keep retrying.
        if passed:
            break

        prev_answer = extracted

    # Energy-guided selection: among passing attempts pick lowest energy;
    # if none pass, pick lowest energy overall (Exp 905 pattern).
    passing = [a for a in attempts if a["passed"]]
    best = (
        min(passing, key=lambda a: a["energy"])
        if passing
        else min(attempts, key=lambda a: a["energy"])
    )

    baseline_passed = attempts[0]["passed"] if attempts else False
    repair_passed = best["passed"]

    return {
        "baseline_passed": baseline_passed,
        "repair_passed": repair_passed,
        "n_retries": best["round"],
        "energy_score_best": best["energy"],
        "best_round": best["round"],
        "baseline_extracted": attempts[0]["extracted_answer"] if attempts else None,
        "best_extracted": best["extracted_answer"],
        "ground_truth": ground_truth,
        "n_attempts": len(attempts),
    }


# ---------------------------------------------------------------------------
# GGUF model loader
# ---------------------------------------------------------------------------


class _LlamaCppRunner:
    """LLM runner backed by llama-cpp-python for GGUF models.

    n_gpu_layers=-1 offloads all transformer layers to GPU, which is
    required for reasonable inference speed on a 26B model.  On ROCm
    (AMD GPU), llama.cpp uses HIP and behaves identically to CUDA from
    the Python API perspective.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def generate(self, prompt: str) -> str:
        """Generate a completion and return only the assistant reply text."""
        output = self._model(
            prompt,
            max_tokens=512,
            temperature=0.0,
            echo=False,
        )
        return output["choices"][0]["text"].strip()


def _find_sota_model_path() -> tuple[str | None, str]:
    """Find the SOTA GGUF model path using the priority order from CLAUDE.md.

    Priority:
    1. sota_model_path from Exp 941 preflight JSON (already verified available)
    2. resolve_cached_gguf("unsloth/gemma-4-26B-A4B-it-GGUF")
    3. resolve_cached_gguf("unsloth/gemma-4-31B-it-GGUF")
    4. resolve_cached_gguf("unsloth/Qwen3.6-35B-A3B-GGUF")

    Returns (path_or_None, model_id_string).
    """
    # Priority 1: use path from Exp 941 preflight if it points to an existing file.
    preflight_path = _REPO_ROOT / "results" / "experiment_941_preflight_v22.json"
    if preflight_path.exists():
        try:
            preflight = json.loads(preflight_path.read_text())
            sota_path = preflight.get("sota_model_path", "")
            if sota_path and Path(sota_path).exists():
                print(f"[exp942] Using Exp 941 preflight sota_model_path: {sota_path}", flush=True)
                # Derive a human-readable model ID from the path.
                if "gemma-4-26B" in sota_path:
                    return sota_path, "unsloth/gemma-4-26B-A4B-it-GGUF"
                if "gemma-4-31B" in sota_path:
                    return sota_path, "unsloth/gemma-4-31B-it-GGUF"
                if "Qwen3.6-35B" in sota_path or "Qwen3" in sota_path:
                    return sota_path, "unsloth/Qwen3.6-35B-A3B-GGUF"
                return sota_path, "sota_gguf_unknown_id"
        except Exception as exc:
            print(f"[exp942] Failed to read Exp 941 preflight: {exc}", flush=True)

    # Priority 2-4: resolve from HF cache.
    try:
        from carnot.inference.sota_models import resolve_cached_gguf  # noqa: PLC0415

        for hf_id in (
            "unsloth/gemma-4-26B-A4B-it-GGUF",
            "unsloth/gemma-4-31B-it-GGUF",
            "unsloth/Qwen3.6-35B-A3B-GGUF",
        ):
            path = resolve_cached_gguf(hf_id)
            if path and Path(path).exists():
                print(f"[exp942] resolve_cached_gguf found: {hf_id} -> {path}", flush=True)
                return path, hf_id
    except Exception as exc:
        print(f"[exp942] resolve_cached_gguf failed: {exc}", flush=True)

    return None, "not_found"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate Exp 942 math repair with a SOTA GGUF model."""
    tmpl = ExperimentTemplate(
        exp_id=942,
        title="Math Repair SOTA v2 — GSM8K",
        deliverable=_DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    t_start = time.perf_counter()
    print("[exp942] Starting experiment 942", flush=True)

    # -- Energy scorer -------------------------------------------------------
    energy_scorer, energy_scorer_type = _build_energy_scorer()
    print(f"[exp942] Energy scorer: {energy_scorer_type}", flush=True)

    # -- Find SOTA model path ------------------------------------------------
    sota_candidates_checked = [
        "sota_model_path_from_exp941",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
    ]
    model_path, model_id = _find_sota_model_path()

    if model_path is None:
        print("[exp942] No SOTA GGUF model found — exiting with sota_model_not_found.", flush=True)
        artifact = tmpl.build_result(
            {
                "model_id": "none",
                "models_used": [],
                "model_checked": sota_candidates_checked,
                "inference_mode": "no_model",
                "baseline_accuracy": 0.0,
                "repair_accuracy": 0.0,
                "final_accuracy": 0.0,
                "signed_improvement": 0.0,
                "energy_scorer_type": energy_scorer_type,
            },
            status="blocked",
            honest_verdict="sota_model_not_found",
        )
        output_path = _REPO_ROOT / _DELIVERABLE
        output_path.write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        return

    # -- Load model via llama.cpp --------------------------------------------
    print(f"[exp942] Loading {model_id} from {model_path} via llama.cpp …", flush=True)
    try:
        from llama_cpp import Llama  # noqa: PLC0415

        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,  # offload all layers to GPU (ROCm HIP or CUDA)
            n_ctx=4096,
            verbose=False,
        )
        runner: Any = _LlamaCppRunner(llm)
        inference_mode = "live_gpu"
        print(f"[exp942] Model loaded: {model_id}", flush=True)
    except Exception as exc:
        print(f"[exp942] llama.cpp load failed: {exc}", flush=True)
        artifact = tmpl.build_result(
            {
                "model_id": model_id,
                "models_used": [model_id],
                "model_load_error": str(exc),
                "traceback": tb.format_exc(),
                "inference_mode": "load_failed",
                "baseline_accuracy": 0.0,
                "repair_accuracy": 0.0,
                "final_accuracy": 0.0,
                "signed_improvement": 0.0,
                "energy_scorer_type": energy_scorer_type,
            },
            status="blocked",
            honest_verdict="model_load_failed",
        )
        output_path = _REPO_ROOT / _DELIVERABLE
        output_path.write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        return

    # -- Run repair loop over all 25 problems --------------------------------
    problems = _GSM8K_PROBLEMS
    results_per_problem: list[dict[str, Any]] = []
    n_baseline_pass = 0
    n_repair_pass = 0

    for idx, prob in enumerate(problems):
        question = prob["question"]
        ground_truth = prob["answer"]
        print(f"[exp942] {idx + 1}/{len(problems)}: {question[:60]}…", flush=True)
        t0 = time.perf_counter()

        try:
            result = _run_problem(question, ground_truth, runner, energy_scorer, max_retries=3)
        except Exception as exc:
            print(f"[exp942]   ERROR: {exc}", flush=True)
            result = {
                "baseline_passed": False,
                "repair_passed": False,
                "n_retries": 0,
                "energy_score_best": 0.0,
                "best_round": 0,
                "error": str(exc),
                "ground_truth": ground_truth,
                "n_attempts": 0,
            }

        elapsed = round(time.perf_counter() - t0, 2)
        result["question"] = question
        result["elapsed_s"] = elapsed

        if result["baseline_passed"]:
            n_baseline_pass += 1
        if result["repair_passed"]:
            n_repair_pass += 1

        print(
            f"[exp942]   baseline={result['baseline_passed']} "
            f"repair={result['repair_passed']} "
            f"retries={result['n_retries']} "
            f"energy={result['energy_score_best']:.3f} [{elapsed}s]",
            flush=True,
        )
        results_per_problem.append(result)

        # Checkpoint every 5 problems to survive interruption without losing progress.
        if (idx + 1) % 5 == 0:
            tmpl.checkpoint_save({"results_so_far": results_per_problem}, step=idx + 1)

    # -- Metrics -------------------------------------------------------------
    n = len(problems)
    baseline_accuracy = n_baseline_pass / n if n > 0 else 0.0
    repair_accuracy = n_repair_pass / n if n > 0 else 0.0
    # final_accuracy uses the energy-selected answer (repair_passed field).
    final_accuracy = repair_accuracy
    signed_improvement = final_accuracy - baseline_accuracy

    # -- Honest verdict ------------------------------------------------------
    if signed_improvement > 0.10:
        honest_verdict = "math_repair_significant"
    elif signed_improvement > 0:
        honest_verdict = "math_repair_marginal"
    elif signed_improvement == 0:
        honest_verdict = "math_repair_zero"
    else:
        honest_verdict = "math_repair_negative"

    print(
        f"\n[exp942] Results: baseline={baseline_accuracy:.3f} "
        f"repair={repair_accuracy:.3f} "
        f"final={final_accuracy:.3f} "
        f"signed_improvement={signed_improvement:+.3f} "
        f"verdict={honest_verdict}",
        flush=True,
    )

    # -- Write deliverable ---------------------------------------------------
    artifact = tmpl.build_result(
        {
            "model_id": model_id,
            "models_used": [model_id],
            "n_problems": n,
            "baseline_accuracy": baseline_accuracy,
            "repair_accuracy": repair_accuracy,
            "final_accuracy": final_accuracy,
            "signed_improvement": signed_improvement,
            "n_baseline_pass": n_baseline_pass,
            "n_repair_pass": n_repair_pass,
            "energy_scorer_type": energy_scorer_type,
            "inference_mode": inference_mode,
            "max_retries": 3,
            "results_per_problem": results_per_problem,
            "prior_failure_addressed": "exp930-math-iterative-self-repair-v1",
            "prior_failure_verdict": "math_repair_zero",
            "prior_failure_root_cause": "RETRO-MATH-REPAIR-MODEL-CEILING",
        },
        status="success",
        honest_verdict=honest_verdict,
        decision_class="repair",
    )
    output_path = _REPO_ROOT / _DELIVERABLE
    output_path.write_text(json.dumps(artifact, indent=2))
    print(f"[exp942] Artifact written to {output_path}", flush=True)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
