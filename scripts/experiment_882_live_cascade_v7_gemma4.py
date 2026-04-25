#!/usr/bin/env python3
"""Experiment 882 — Live Cascade v7: Gemma4-E4B-it + Full Cascade, 50 GSM8K Questions.

**Researcher summary:**
    Prior live cascade benchmarks (Exps 871, 858) reported simulation_fallback or
    were blocked.  This experiment uses google/gemma-4-E4B-it via the HuggingFace
    transformers path (not GGUF / llama.cpp), which reliably loads from the local
    HF model cache as demonstrated in Exp 881.  The full Tier 0-3 cascade is
    exercised: Tiers 0a/0b/0c/0d/0e/0g (StreamingCoT, Exp 874) all advisory,
    Tier 2.5 (SymCodeVerifier AUC=0.804 live), Tier 2.7 (CausalReasoningVerifier)
    wired, and Tier 3 (Ising via VerifyRepairPipeline) for uncleaned responses.

    This is the first cascade benchmark expected to run on real GPU output.

**Gate:**
    Reads results/experiment_855_preflight_v15.json and aborts if
    live_env_fixed != True.  Also requires CARNOT_FORCE_LIVE in the environment.

**Cascade tiers exercised (Tiers 0-3):**
    Tier 0 — ThreeTierPipeline early-exit (text-mode probes)
    Tier 1 — SinkProbe (skipped when attention_matrix=None, which is our case)
    Tier 2 — EORM energy scoring
    Tier 3 — Ising via VerifyRepairPipeline for cases not cleared by 0-2

**StreamingCoT (Tier 0g):**
    Enabled when CARNOT_STREAMING_COT=1.  Uses StreamingCoTHalluDetector to
    emit an advisory streaming_cot_unstable flag per response.  Does not
    change the cascade decision; purely observational.

**Metrics reported:**
    - baseline_accuracy:    fraction correct before repair (raw Gemma4 output)
    - carnot_accuracy:      fraction correct after cascade + repair
    - signed_improvement:   carnot_accuracy - baseline_accuracy
    - cascade_skip_rate:    fraction cleared by Tiers 0-2 (no Ising needed)
    - cascade_tiers_active: count of distinct tiers that fired at least once
    - inference_mode:       "live_gpu" when CARNOT_FORCE_LIVE=1 and model healthy

**Honest verdict mapping:**
    "positive_improvement"  signed_improvement > 0 AND inference_mode=live_gpu
    "live_no_improvement"   inference_mode=live_gpu AND signed_improvement <= 0
    "cascade_running"       inference_mode=live_gpu AND cascade_tiers_active >= 3
    "simulation_fallback"   inference_mode != live_gpu
    "blocked"               gate failed or model load failed

Spec: REQ-BENCH-015 (live cascade benchmark), SCENARIO-BENCH-034
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo root wiring — allow running as standalone script from any cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP_ID = 882
TITLE = "Live Cascade v7: Gemma4-E4B-it + Full Cascade, 50 GSM8K"
DELIVERABLE = "results/experiment_882_live_cascade_v7_gemma4.json"
GATE_ARTIFACT = "results/experiment_855_preflight_v15.json"
MODEL_ID = "google/gemma-4-E4B-it"
N_GSM8K = 50

# 50 GSM8K-style arithmetic word problems with ground-truth answers.
# Identical corpus to Exp 871 for comparability.
_GSM8K_PROBLEMS: list[dict[str, Any]] = [
    {"id": f"gsm8k_{i}", "question": q, "answer": a}
    for i, (q, a) in enumerate([
        ("Janet has 3 apples and buys 5 more. How many does she have?", "8"),
        ("A car travels at 60 mph for 2 hours. How many miles?", "120"),
        ("If 4 shirts cost $48, how much does 1 shirt cost?", "$12"),
        ("Tom has 7 cats. He gives away 3. How many remain?", "4"),
        ("A rectangle is 6 cm by 4 cm. What is the area?", "24"),
        ("There are 5 rows of 8 chairs. How many chairs total?", "40"),
        ("Sara read 15 pages Monday and 22 Tuesday. Total pages?", "37"),
        ("A dozen eggs costs $3. How much for 3 dozen?", "$9"),
        ("A tank holds 120 litres. It is 3/4 full. How many litres?", "90"),
        ("John is 12. His father is 3 times his age. Father's age?", "36"),
        ("A train leaves at 9am and arrives at 1pm. Journey hours?", "4"),
        ("A pizza has 8 slices. 3 are eaten. Slices left?", "5"),
        ("15% of 200 is what number?", "30"),
        ("A square has side 7 cm. What is the perimeter?", "28"),
        ("60 students, 40% are girls. How many girls?", "24"),
        ("A shop sells 50 items per day. Items in 7 days?", "350"),
        ("Two numbers sum to 20 and one is 8. Other number?", "12"),
        ("A bag weighs 2.5 kg. 4 bags weigh how much?", "10"),
        ("If you earn $15/hr for 8 hrs, total pay?", "$120"),
        ("A box has 24 chocolates split among 6 kids equally. Each gets?", "4"),
        ("Temperature drops from 72F to 59F. Drop in degrees?", "13"),
        ("A recipe uses 2 cups of flour for 12 cookies. For 36 cookies?", "6"),
        ("A pool is 25m long. 8 laps = how many metres?", "200"),
        ("There are 100 seats; 63 are taken. Seats available?", "37"),
        ("5 friends share $75 equally. Each gets?", "$15"),
        ("A book has 320 pages. You read 80. Pages left?", "240"),
        ("3 + 4 x 2 = ?", "11"),
        ("A triangle has angles 45 and 60 degrees. Third angle?", "75"),
        ("$200 saved, spend $35.50. Amount left?", "$164.50"),
        ("A car travels 300 km on 30 L. Km per litre?", "10"),
        ("6 workers build 1 wall in 10 days. 1 worker takes how many days?", "60"),
        ("25 x 4 = ?", "100"),
        ("Largest prime less than 20?", "19"),
        ("A cube has side 3 cm. Volume?", "27"),
        ("Perimeter of a rectangle 9m by 5m?", "28"),
        ("Discount 20% off $50. Final price?", "$40"),
        ("LCM of 4 and 6?", "12"),
        ("GCD of 12 and 18?", "6"),
        ("A cistern fills in 6 hours. Fraction filled in 2 hours?", "1/3"),
        ("Distance = speed x time. Speed=50, time=3. Distance?", "150"),
        ("Average of 4, 8, 12, 16?", "10"),
        ("Angle in semicircle subtended at circumference?", "90"),
        ("Simple interest: P=1000, R=5%, T=2 years?", "$100"),
        ("Perimeter of equilateral triangle with side 9?", "27"),
        ("2^8 = ?", "256"),
        ("3 apples + 2 oranges = 5 fruits. 10 fruits if same ratio: apples?", "6"),
        ("A store has 5 red, 3 blue, 2 green balls. P(red)?", "0.5"),
        ("If 2x = 14, x = ?", "7"),
        ("Sum of first 10 natural numbers?", "55"),
        ("Area of circle radius 7 (use pi=22/7)?", "154"),
    ])
]

assert len(_GSM8K_PROBLEMS) == N_GSM8K, (
    f"Expected {N_GSM8K} problems, got {len(_GSM8K_PROBLEMS)}"
)


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------

def _check_gate() -> tuple[bool, str]:
    """Return (ok, reason).  Gate passes when live_env_fixed==True and CARNOT_FORCE_LIVE set.

    Why two conditions: live_env_fixed confirms EnvPropagationGuard shipped (Exp 855),
    while CARNOT_FORCE_LIVE is the runtime opt-in that prevents accidental live-GPU
    runs during dry-run CI passes.
    """
    if "CARNOT_FORCE_LIVE" not in os.environ:
        return False, "CARNOT_FORCE_LIVE not set — run with CARNOT_FORCE_LIVE=1"
    gate_path = _REPO_ROOT / GATE_ARTIFACT
    if not gate_path.exists():
        return False, f"preflight artifact missing: {gate_path}"
    with open(gate_path) as f:
        data = json.load(f)
    if not data.get("live_env_fixed", False):
        return False, f"live_env_fixed != True in {gate_path}"
    return True, "gate passed"


# ---------------------------------------------------------------------------
# Answer generation helper (real Gemma4 inference)
# ---------------------------------------------------------------------------

def _generate_answer(model: Any, tokenizer: Any, question: str, max_new_tokens: int = 128) -> str:
    """Generate a concise math answer from Gemma4-E4B-it given a question.

    We ask the model for a short numeric answer.  The instruction format mirrors
    standard GSM8K few-shot prompts: direct, no explanatory prose requested.

    Why max_new_tokens=128: GSM8K answers are short (single number or expression).
    Larger budgets increase latency without improving answer quality for these prompts.

    Args:
        model:          Loaded transformers AutoModelForCausalLM (bfloat16, device_map="auto").
        tokenizer:      Matching AutoTokenizer for the model.
        question:       Plain-text arithmetic word problem.
        max_new_tokens: Maximum tokens the model may generate.

    Returns:
        The decoded model output (newly generated tokens only, skip_special_tokens=True).
    """
    import torch  # local import — heavy optional dep

    prompt = (
        "Solve the following math problem. "
        "Give only the final numeric answer, nothing else.\n\n"
        f"Problem: {question}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Answer extraction / comparison
# ---------------------------------------------------------------------------

def _extract_final_answer(raw: str) -> str:
    """Extract the first number-like token from a model response.

    GSM8K ground-truth answers are short strings like "8", "$12", "1/3".
    This extractor normalises whitespace and returns the first token that
    looks like a number, fraction, or dollar-amount so minor formatting
    differences don't count as wrong.

    Why strip and lowercase: model may add trailing newlines or vary case on
    tokens like "None" or "Zero".

    Args:
        raw: The raw decoded model output string.

    Returns:
        Extracted answer string, or the original stripped string as fallback.
    """
    text = raw.strip()
    # Try to match a leading dollar-amount like $12 or $164.50
    dollar_match = re.search(r"\$[\d,]+(?:\.\d+)?", text)
    if dollar_match:
        return dollar_match.group(0)
    # Match fractions like 1/3
    frac_match = re.search(r"\b\d+/\d+\b", text)
    if frac_match:
        return frac_match.group(0)
    # Match decimals and integers, including leading minus
    num_match = re.search(r"-?\d+(?:\.\d+)?", text)
    if num_match:
        return num_match.group(0)
    # Fallback: first whitespace-separated token
    parts = text.split()
    return parts[0] if parts else text


def _answers_match(predicted: str, reference: str) -> bool:
    """Return True when predicted and reference answers are semantically equal.

    Handles minor differences in spacing and currency symbol normalisation.

    Why normalise: Gemma4 may produce "$ 12" while reference is "$12"; both
    are correct answers and should not be penalised.

    Args:
        predicted: Model-generated answer (post-extraction).
        reference: Ground-truth answer string.

    Returns:
        True if the answers are equivalent.
    """
    def norm(s: str) -> str:
        # Remove spaces around $ and normalise whitespace
        return re.sub(r"\s+", "", s.lower().replace("$ ", "$").replace(" $", "$"))

    return norm(predicted) == norm(reference)


# ---------------------------------------------------------------------------
# Per-question cascade runner
# ---------------------------------------------------------------------------

def _run_cascade(
    problem: dict[str, Any],
    model: Any,
    tokenizer: Any,
    three_tier: Any,
    verify_repair: Any,
    inference_mode: str,
    streaming_cot_detector: Any,
) -> dict[str, Any]:
    """Run Tiers 0-3 for one GSM8K problem and return a per-question result dict.

    Why separate from main(): keeping cascade logic in its own function makes
    it independently testable without real GPU hardware — callers mock model,
    three_tier, and verify_repair.

    Pipeline per question:
        1. Generate CoT response with Gemma4 (live) or stub (simulation).
        2. Optionally run StreamingCoT advisory probe (Tier 0g).
        3. Run ThreeTierPipeline (Tiers 0-2) with early-exit.
        4. If not cleared by Tiers 0-2, run VerifyRepairPipeline (Tier 3).
        5. Extract final answer from each stage and compare to ground truth.

    Returns:
        dict with keys: id, tier_exited_at, was_correct_baseline,
        was_correct_carnot, repaired, latency_ms, streaming_cot_unstable.
    """
    ref_answer = problem.get("answer", "")
    question = problem["question"]
    t0 = time.perf_counter()

    # Step a: generate baseline response
    if inference_mode == "live_gpu" and model is not None and tokenizer is not None:
        raw_response = _generate_answer(model, tokenizer, question)
    else:
        # Simulation: deterministic stub for CI / blocked runs
        idx = int(problem["id"].split("_")[-1])
        raw_response = ref_answer if idx % 10 >= 3 else "WRONG"

    baseline_extracted = _extract_final_answer(raw_response)
    was_correct_baseline = _answers_match(baseline_extracted, ref_answer)

    # Step b: StreamingCoT advisory probe (Tier 0g) — observational only
    streaming_cot_unstable: bool | None = None
    if streaming_cot_detector is not None:
        try:
            steps = [s for s in raw_response.split("\n") if s.strip()]
            detect_result = streaming_cot_detector.detect(steps)
            streaming_cot_unstable = detect_result.is_streaming_unstable
        except Exception:
            streaming_cot_unstable = None

    # Step c/d: ThreeTierPipeline (Tiers 0-2) with early-exit
    tier_exited_at: int | None = None
    final_response = raw_response
    repaired = False

    if inference_mode == "live_gpu" and three_tier is not None:
        try:
            verified, tier_used, _energy = three_tier.verify(
                response=raw_response,
                question=question,
                attention_matrix=None,   # no attention matrix — offline mode
                hidden_states=None,      # no hidden states — CI-safe
            )
            # Map tier_used string to integer tier index for the artifact
            _TIER_MAP = {
                "spilled_energy": 0,
                "nup_probe_v4": 0,
                "basin_detector": 0,
                "sink_probe": 1,
                "eorm": 2,
                "vg_skip": 2,
            }
            if tier_used in _TIER_MAP:
                tier_exited_at = _TIER_MAP[tier_used]
            # If cascade cleared the response, accept the baseline answer as final
            if verified:
                final_response = raw_response
        except Exception as exc:
            print(f"[exp882] ThreeTierPipeline.verify() failed: {exc}", flush=True)
            # Fall through to Tier 3

    # Step e: Tier 3 (Ising) via VerifyRepairPipeline if not cleared
    if tier_exited_at is None and inference_mode == "live_gpu" and verify_repair is not None:
        try:
            repair_result = verify_repair.verify_and_repair(
                question=question,
                response=raw_response,
                domain="math",
            )
            repaired_response = getattr(repair_result, "repaired_response", None)
            if repaired_response and repaired_response.strip() != raw_response.strip():
                repaired = True
                final_response = repaired_response
        except Exception as exc:
            print(f"[exp882] VerifyRepairPipeline.verify_and_repair() failed: {exc}", flush=True)

    final_extracted = _extract_final_answer(final_response)
    was_correct_carnot = _answers_match(final_extracted, ref_answer)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "id": problem["id"],
        "tier_exited_at": tier_exited_at,
        "was_correct_baseline": was_correct_baseline,
        "was_correct_carnot": was_correct_carnot,
        "repaired": repaired,
        "latency_ms": latency_ms,
        "streaming_cot_unstable": streaming_cot_unstable,
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    per_question: list[dict[str, Any]],
    inference_mode: str,
) -> dict[str, Any]:
    """Aggregate per-question results into experiment-level metrics.

    Why a standalone function: the conductor retrospective script imports metric
    helpers directly, so keeping them pure (no side effects, no I/O) makes them
    composable.

    Honest verdict precedence (order matters):
        1. simulation_fallback — inference_mode not live_gpu
        2. cascade_running     — live_gpu AND cascade_tiers_active >= 3
        3. positive_improvement — live_gpu AND signed_improvement > 0
        4. live_no_improvement — catch-all for live_gpu runs

    Returns:
        baseline_accuracy, carnot_accuracy, signed_improvement,
        cascade_skip_rate, cascade_tiers_active, honest_verdict
    """
    n = len(per_question)
    if n == 0:
        return {
            "baseline_accuracy": 0.0,
            "carnot_accuracy": 0.0,
            "signed_improvement": 0.0,
            "cascade_skip_rate": 0.0,
            "cascade_tiers_active": 0,
            "honest_verdict": "blocked",
        }

    baseline_correct = sum(1 for r in per_question if r["was_correct_baseline"])
    carnot_correct = sum(1 for r in per_question if r["was_correct_carnot"])
    skipped = sum(
        1 for r in per_question
        if r.get("tier_exited_at") is not None
    )

    # Track distinct tiers that fired: 0/1/2 from tier_exited_at; 3 when repaired
    tiers_fired: set[int] = set()
    for r in per_question:
        tee = r.get("tier_exited_at")
        if isinstance(tee, int):
            tiers_fired.add(tee)
        if r.get("repaired") and tee is None:
            tiers_fired.add(3)

    baseline_accuracy = round(baseline_correct / n, 4)
    carnot_accuracy = round(carnot_correct / n, 4)
    signed_improvement = round(carnot_accuracy - baseline_accuracy, 4)
    cascade_skip_rate = round(skipped / n, 4)
    cascade_tiers_active = len(tiers_fired)

    if inference_mode != "live_gpu":
        honest_verdict = "simulation_fallback"
    elif cascade_tiers_active >= 3:
        honest_verdict = "cascade_running"
    elif signed_improvement > 0:
        honest_verdict = "positive_improvement"
    else:
        honest_verdict = "live_no_improvement"

    return {
        "baseline_accuracy": baseline_accuracy,
        "carnot_accuracy": carnot_accuracy,
        "signed_improvement": signed_improvement,
        "cascade_skip_rate": cascade_skip_rate,
        "cascade_tiers_active": cascade_tiers_active,
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate: gate check → model load → 50-question cascade → artifact."""
    import torch  # noqa: PLC0415 — heavy dep, local import

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # -------------------------------------------------------------------------
    # GATE CHECK
    # -------------------------------------------------------------------------
    gate_ok, gate_reason = _check_gate()
    if not gate_ok:
        artifact = tmpl.build_result(
            {"honest_verdict": "blocked", "gate_reason": gate_reason,
             "inference_mode": "blocked"},
            status="blocked",
            honest_verdict="blocked",
            inference_mode="blocked",
        )
        (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        print(f"[exp882] BLOCKED: {gate_reason}", flush=True)
        tmpl.assert_deliverable_written()
        return

    inference_mode = "live_gpu"

    # -------------------------------------------------------------------------
    # MODEL LOAD
    # -------------------------------------------------------------------------
    print(f"[exp882] Loading {MODEL_ID} via transformers …", flush=True)
    model: Any = None
    tokenizer: Any = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        print(f"[exp882] Model loaded on device: {model.device}", flush=True)
    except Exception as exc:
        tb = traceback.format_exc()
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked",
                "model_load_error": str(exc),
                "traceback": tb,
                "inference_mode": "blocked",
            },
            status="blocked",
            honest_verdict="blocked",
            inference_mode="blocked",
        )
        (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        print(f"[exp882] BLOCKED model load: {exc}", flush=True)
        tmpl.assert_deliverable_written()
        return

    # -------------------------------------------------------------------------
    # WARM-UP
    # -------------------------------------------------------------------------
    print("[exp882] Warm-up pass …", flush=True)
    _generate_answer(model, tokenizer, "What is 2 + 2?", max_new_tokens=16)
    print("[exp882] Warm-up done.", flush=True)

    # -------------------------------------------------------------------------
    # PIPELINE INSTANTIATION
    # -------------------------------------------------------------------------
    three_tier: Any = None
    verify_repair_pipeline: Any = None
    streaming_cot_detector: Any = None

    try:
        from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline  # noqa: PLC0415
        three_tier = ThreeTierPipeline()
        print("[exp882] ThreeTierPipeline loaded.", flush=True)
    except Exception as exc:
        print(f"[exp882] WARNING: ThreeTierPipeline unavailable: {exc}", flush=True)

    try:
        from carnot.pipeline.verify_repair import VerifyRepairPipeline  # noqa: PLC0415
        verify_repair_pipeline = VerifyRepairPipeline(
            model=None,
            domains=["math"],
            max_repairs=2,
            extractor=None,
            semantic_grounding_verifier=None,
            semantic_verifier_v2=None,
            timeout_seconds=60,
            memory=None,
            template_library=None,
            session_memory=None,
            constraint_memory=None,
            nup_probe=None,
            nup_probe_threshold=0.5,
            enable_constraint_accumulation=False,
            second_model_spec=None,
        )
        print("[exp882] VerifyRepairPipeline loaded.", flush=True)
    except Exception as exc:
        print(f"[exp882] WARNING: VerifyRepairPipeline unavailable: {exc}", flush=True)

    # Wire StreamingCoT (Tier 0g) if requested
    if os.environ.get("CARNOT_STREAMING_COT") == "1":
        try:
            from carnot.pipeline.streaming_cot import StreamingCoTHalluDetector  # noqa: PLC0415
            streaming_cot_detector = StreamingCoTHalluDetector(alpha=0.3, threshold=0.35)
            if three_tier is not None:
                three_tier.wire_tier_0g(streaming_cot_detector)
            print("[exp882] StreamingCoT (Tier 0g) wired.", flush=True)
        except Exception as exc:
            print(f"[exp882] WARNING: StreamingCoT unavailable: {exc}", flush=True)

    # -------------------------------------------------------------------------
    # RUN CASCADE ON 50 GSM8K QUESTIONS
    # -------------------------------------------------------------------------
    print(f"[exp882] Running cascade on {N_GSM8K} GSM8K questions …", flush=True)
    per_question: list[dict[str, Any]] = []

    for problem in _GSM8K_PROBLEMS:
        result = _run_cascade(
            problem,
            model,
            tokenizer,
            three_tier,
            verify_repair_pipeline,
            inference_mode,
            streaming_cot_detector,
        )
        per_question.append(result)

        # Checkpoint every 10 questions
        if len(per_question) % 10 == 0:
            tmpl.checkpoint_save({"partial_results": per_question}, step=len(per_question))

        q_num = len(per_question)
        tier = result.get("tier_exited_at")
        print(
            f"[exp882] Q{q_num:02d}/{N_GSM8K} tier={tier} "
            f"baseline={result['was_correct_baseline']} "
            f"carnot={result['was_correct_carnot']} "
            f"repaired={result['repaired']} [{result['latency_ms']:.0f}ms]",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # COMPUTE METRICS
    # -------------------------------------------------------------------------
    metrics = _compute_metrics(per_question, inference_mode)
    print(
        f"[exp882] Done. baseline={metrics['baseline_accuracy']:.3f} "
        f"carnot={metrics['carnot_accuracy']:.3f} "
        f"improvement={metrics['signed_improvement']:+.4f} "
        f"skip_rate={metrics['cascade_skip_rate']:.3f} "
        f"tiers_active={metrics['cascade_tiers_active']} "
        f"verdict={metrics['honest_verdict']}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # BUILD AND WRITE ARTIFACT
    # -------------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "honest_verdict": metrics["honest_verdict"],
            "inference_mode": inference_mode,
            "model_id": MODEL_ID,
            "n_gsm8k": N_GSM8K,
            "baseline_accuracy": metrics["baseline_accuracy"],
            "carnot_accuracy": metrics["carnot_accuracy"],
            "signed_improvement": metrics["signed_improvement"],
            "cascade_skip_rate": metrics["cascade_skip_rate"],
            "cascade_tiers_active": metrics["cascade_tiers_active"],
            "streaming_cot_enabled": os.environ.get("CARNOT_STREAMING_COT") == "1",
            "per_question": per_question,
        },
        status="success",
        honest_verdict=metrics["honest_verdict"],
        inference_mode=inference_mode,
        decision_class=["verify", "repair"],
    )

    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"[exp882] Artifact written to {out_path}", flush=True)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
