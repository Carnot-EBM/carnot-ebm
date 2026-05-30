#!/usr/bin/env python3
"""Exp 3437 (P0.1 v3): Energy-vote vs Self-Consistency on a REPAIRED harness.

**The load-bearing repair of exp3426.** exp3426 ran clean wall-time (642 s) but
its multi-sample harness was broken: every k-sample condition (self-consistency,
self-certainty BoN, energy-argmin, energy-weighted vote) returned 0.0 accuracy
while greedy AR scored 0.75. The "energy matches self-consistency" verdict it
emitted was a degenerate 0.0-vs-0.0 tie, NOT a measurement.

Root cause (mechanical, not scientific): exp3426 requested per-token logprobs
(`logprobs=1`) from a llama.cpp model created with `logits_all=False`. llama.cpp
raises ``ValueError: logprobs is not supported for models created with
logits_all=False`` in that case, and exp3426's broad ``except Exception``
swallowed it, returning an empty string for every sampled candidate. Empty text
extracts to a ``None`` answer, so every vote collapsed to nothing. The greedy
condition survived only because it did not request logprobs.

This v3 FIXES the harness:
  1. The llama.cpp model is created with ``logits_all=True`` so per-token
     logprobs actually work for the sampled candidates.
  2. A NON-DEGENERATE-SC gate (step 0e) runs on a warm-up batch BEFORE any energy
     comparison is reported: majority-vote self-consistency must be at least
     greedy accuracy AND strictly above an absolute floor. A 0.0-vs-0.0 tie is
     impossible to ship.
  3. Per-sample answer extraction is an explicit, unit-tested step that runs on
     EACH sampled generation.

PRIMARY comparison (unchanged from exp3426): energy-weighted vote vs
majority-vote self-consistency at MATCHED compute. The sharpened bar
(arXiv:2410.12608): a program-verifier already beats SC on math, so the honest
target is "energy beats SC," not merely "energy beats greedy AR."

Conditions (greedy AR is the 1-sample floor; conditions 2-4 all aggregate the
SAME k sampled generations — only the aggregation rule differs, so energy never
gets extra samples):

  1. greedy AR              — 1 greedy generation/problem (the exp3312 baseline).
  2. self-consistency       — majority vote over k samples (PRIMARY control).
  3. self-certainty BoN     — pick the most-confident sample (arXiv:2502.18581).
  4a. energy-argmin         — min-energy candidate after bounded latent descent.
  4b. energy-weighted vote  — softmax(-E/T) over answers (EBM-CoT,
                              arXiv:2511.07124). The headline condition.

Spec: REQ-KONA-3437, SCENARIO-KONA-3437, SCENARIO-KONA-3437-DEGENERATE.
Run: .venv/bin/python scripts/experiment_3437_energy_vote_vs_self_consistency_real_harness_premise_v3.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.inference.sota_models import resolve_cached_gguf  # noqa: E402
from carnot.phase3.energy_premise_v3 import (  # noqa: E402
    derive_premise_v3_verdict,
    energy_descent_select,
    energy_weighted_vote,
    evaluate_sc_non_degenerate,
    extract_candidate_answers,
    extract_final_answer,
    is_correct,
    load_gsm8k_subset,
    majority_vote,
    mcnemar_test,
    paired_bootstrap_ci,
    reproducibility_checksum,
    self_certainty_select,
)

DELIVERABLE_PATH = (
    "results/experiment_3437_energy_vote_vs_self_consistency_real_harness_premise_v3.json"
)
CORPUS_PATH = project_root / "data" / "research" / "gsm8k_adversarial_281.jsonl"
AR_MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"

# Tunables (env-overridable so a smoke run is cheap; defaults are the real run).
# SEED is deliberately NOT the experiment id (3437) — exp3312 was flagged by
# adversarial_verify when random_seed==experiment id read as a tautology.
N_PROBLEMS = int(os.environ.get("EXP3437_N", "200"))
K_SAMPLES = int(os.environ.get("EXP3437_K", "5"))
MAX_TOKENS = int(os.environ.get("EXP3437_MAXTOK", "512"))
DESCENT_STEPS = int(os.environ.get("EXP3437_DESCENT_STEPS", "8"))
SEED = int(os.environ.get("EXP3437_SEED", "20260531"))
WARMUP_N = int(os.environ.get("EXP3437_WARMUP", "20"))
# Scale-free energy-vote sharpness (applied to per-problem z-scored energies).
ENERGY_VOTE_T = float(os.environ.get("EXP3437_VOTE_T", "0.5"))
VISIBLE_DIM = 16
FOVER_TRAIN_EPOCHS = 60

PROMPT_TEMPLATE = (
    "Solve this math word problem. Show brief reasoning, then on the final line "
    "write the answer as: #### <number>\n\nProblem: {q}\n\nSolution:"
)


def _field_principles() -> dict[str, str]:
    """Per-field principle annotations (CLAUDE.md Principle-Annotated Fields)."""
    return {
        "honest_verdict": "Terminal verdict must start with complete:/success:/passed:/shipped_.",
        "inference_substrate": "live_llm_inference: candidates really load + run the GGUF.",
        "task_name": "Name the real benchmark + split; toy/synthetic is not acceptable.",
        "n_problems": ">=200 for a CLT-valid accuracy delta.",
        "k_samples": "Generations per problem for conditions 2-4; the matched-compute budget.",
        "self_consistency_non_degenerate": "Boolean from step 0e: SC accuracy >= greedy AND "
        "> 0.30 on the warm-up batch — the gate that makes the exp3426 0.0-tie impossible "
        "to ship.",
        "ar_greedy_accuracy": "1-sample greedy control (the exp3312/exp3426 baseline).",
        "self_consistency_accuracy": "Majority vote over k samples — the PRIMARY control "
        "energy must beat; MUST be non-degenerate per step 0e.",
        "self_certainty_bon_accuracy": "Self-certainty Best-of-N (arXiv:2502.18581) — the "
        "strongest cheap selector.",
        "energy_argmin_accuracy": "Energy-argmin selection over the same k samples.",
        "energy_weighted_vote_accuracy": "Energy-weighted vote (EBM-CoT) — the premise under "
        "test; the headline condition.",
        "delta_energy_vs_self_consistency": "energy_weighted_vote minus self_consistency at "
        "MATCHED compute — THE headline; does energy add value over majority vote?",
        "delta_energy_vs_greedy_ar": "energy minus greedy AR — reported for continuity with "
        "exp3312/exp3426.",
        "paired_significance": "McNemar exact p + paired bootstrap CI95 for the PRIMARY "
        "(energy vs SC) delta; an unpaired or n<200 delta is gameable.",
        "compute_parity_note": "State per-condition generation budget + param count so energy "
        "does not win by spending more compute.",
        "random_seed": "Determinism precondition for reproducibility.",
        "reproducibility_checksum": "Content hash of corpus + substrate + seed.",
        "duration_s": "Real live 35B inference over 200 problems x k samples takes minutes; "
        "60s floor — a sub-60s duration is the fabrication signal that flagged exp3312-class "
        "artifacts.",
    }


def _train_energy_substrate():
    """Train the Boltzmann-GPT continuous-latent energy on FoVer (correct->low E).

    Returns (model, signature, gap) or raises if the substrate is untrainable.
    Contrastive training pushes correct reasoning traces to low energy and
    incorrect ones to high energy, so the minimum-energy candidate is the one the
    learned manifold judges most correct. The model is frozen after training: the
    per-candidate latent (not the model) is what descends.
    """
    from carnot.data.fover import FoVerDataset  # noqa: PLC0415
    from carnot.phase3.boltzmann_gpt import (  # noqa: PLC0415
        BoltzmannGPTLayer,
        evaluate_energy_gap,
        split_dataset,
        train_contrastive,
    )

    dataset = FoVerDataset()
    train, test = split_dataset(dataset, test_fraction=0.2, seed=SEED)
    model = BoltzmannGPTLayer(visible_dim=VISIBLE_DIM, hidden_dim=VISIBLE_DIM, seed=SEED)
    train_contrastive(
        model, train, n_epochs=FOVER_TRAIN_EPOCHS, lr=1e-2, visible_dim=VISIBLE_DIM, seed=SEED
    )
    gap = evaluate_energy_gap(model, test, visible_dim=VISIBLE_DIM)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    signature = f"boltzmann_gpt_v{VISIBLE_DIM}_e{FOVER_TRAIN_EPOCHS}_gap{gap:.4f}"
    return model, signature, float(gap)


def _generate(llm, prompt: str, *, temperature: float, seed: int, want_logprobs: bool):
    """One bounded llama.cpp generation. Returns (text, token_logprobs|None).

    For the sampled candidates we request logprobs so the self-certainty Best-of-N
    selector has the per-token chosen-token logprobs it needs. THIS IS THE
    exp3426 BUG-FIX SITE: the caller MUST have created the ``Llama`` with
    ``logits_all=True`` (see ``main``), otherwise this call raises
    ``ValueError: logprobs is not supported for models created with
    logits_all=False`` and the candidate is silently lost. Returns ('', None) on
    any failure so a single bad generation scores as a miss rather than crashing
    the 200-problem run — but with the logits_all fix that path should no longer
    fire for the whole sampled set.
    """
    try:
        out = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            seed=seed,
            stop=["\nProblem:", "\n\nProblem"],
            logprobs=1 if want_logprobs else None,
        )
        choice = out["choices"][0]
        text = choice["text"]
        token_logprobs = None
        if want_logprobs and choice.get("logprobs"):
            token_logprobs = [
                lp for lp in choice["logprobs"].get("token_logprobs", []) if lp is not None
            ]
        return text, token_logprobs
    except Exception:  # pragma: no cover - inference-environment-dependent
        return "", None


def _zscore(values: list[float]) -> list[float]:
    """Standardise a short energy vector so the vote temperature is scale-free.

    The Boltzmann-GPT energies have an arbitrary scale; z-scoring the per-problem
    energy vector before the softmax makes the vote temperature ``ENERGY_VOTE_T``
    a dimensionless sharpness knob that means the same thing regardless of how
    large the raw energies are. With a single candidate or a zero-variance
    vector we return zeros (uniform weighting), the only sane fallback.
    """
    if len(values) < 2:
        return [0.0 for _ in values]
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    if sd == 0.0:
        return [0.0 for _ in values]
    return [(v - mean) / sd for v in values]


def _score_problem(llm, energy_model, problem):
    """Run all four conditions on ONE problem; return the per-condition outcomes.

    This is the per-problem unit the warm-up gate and the full run both reuse, so
    the warm-up batch is scored by the EXACT same code path as the final batch —
    any asymmetry there would itself be a harness bug. Returns a dict with the
    correctness booleans and the raw extracted answers (the latter so a degenerate
    run can dump real per-sample answers for diagnosis).
    """
    prompt = PROMPT_TEMPLATE.format(q=problem.question)

    # Condition 1: greedy AR (1 generation, the matched-compute floor).
    ar_text, _ = _generate(llm, prompt, temperature=0.0, seed=SEED, want_logprobs=False)
    ar_pred = extract_final_answer(ar_text)

    # k sampled candidates shared by conditions 2-4 (the matched budget).
    cand_texts: list[str] = []
    cand_logprobs: list[list[float] | None] = []
    for k in range(K_SAMPLES):
        text, lps = _generate(
            llm, prompt, temperature=0.8, seed=SEED + 1000 * (k + 1), want_logprobs=True
        )
        cand_texts.append(text)
        cand_logprobs.append(lps)
    cand_preds = extract_candidate_answers(cand_texts)

    # Condition 2: self-consistency (majority vote over the k samples).
    sc_pred = majority_vote(cand_preds)

    # Condition 3: self-certainty Best-of-N (most-confident single sample).
    scert_idx = self_certainty_select(cand_logprobs)

    # Condition 4: energy. Bounded latent descent -> final per-candidate energy.
    descent = energy_descent_select(
        cand_texts, energy_model, visible_dim=VISIBLE_DIM, n_steps=DESCENT_STEPS, lr=0.05
    )
    eargmin_pred = cand_preds[descent.selected_index]
    evote_pred = energy_weighted_vote(
        cand_preds, _zscore(descent.final_energies), temperature=ENERGY_VOTE_T
    )

    return {
        "problem_id": problem.problem_id,
        "gold": problem.answer,
        "ar_pred": ar_pred,
        "ar_correct": is_correct(ar_pred, problem.answer),
        "sc_pred": sc_pred,
        "sc_correct": is_correct(sc_pred, problem.answer),
        "self_certainty_pred": cand_preds[scert_idx],
        "self_certainty_index": scert_idx,
        "scert_correct": is_correct(cand_preds[scert_idx], problem.answer),
        "energy_argmin_pred": eargmin_pred,
        "energy_argmin_index": descent.selected_index,
        "eargmin_correct": is_correct(eargmin_pred, problem.answer),
        "energy_weighted_vote_pred": evote_pred,
        "evote_correct": is_correct(evote_pred, problem.answer),
        "candidate_preds": cand_preds,
        "candidate_final_energies": [round(e, 6) for e in descent.final_energies],
    }


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=3437,
        title="Energy-vote vs Self-Consistency on a repaired harness (Kona premise v3)",
        deliverable=DELIVERABLE_PATH,
        requires_gpu=True,
    )
    tmpl.setup()
    start = time.time()
    principles = _field_principles()

    def _emit_block(verdict: str, detail: str, extra: dict | None = None) -> None:
        payload = {
            "honest_verdict": verdict,
            "inference_substrate": "live_llm_inference",
            "task_name": "GSM8K (original split) — premise v3",
            "block_detail": detail,
            "preconditions_checked": extra.get("preconditions_checked", []) if extra else [],
            "random_seed": SEED,
            "duration_s": round(time.time() - start, 3),
            "field_provenance": principles,
        }
        if extra:
            payload.update({k: v for k, v in extra.items() if k != "preconditions_checked"})
        artifact = tmpl.build_result(payload, status="blocked")
        Path(DELIVERABLE_PATH).write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        print(f"BLOCKED: {verdict} — {detail}")

    # ----- Step 0: PRECONDITIONS (before any inference) -----
    pre: list[dict] = []

    # (a) CUDA
    try:
        import torch  # noqa: PLC0415

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    pre.append({"resource": "cuda", "available": cuda_ok})
    if not cuda_ok:
        _emit_block(
            "blocked_cuda_unavailable",
            "torch.cuda.is_available() is False",
            {"preconditions_checked": pre},
        )
        return

    # (c) real corpus (cheap to check before the substrate train)
    corpus_ok = CORPUS_PATH.exists()
    pre.append({"resource": "real_task_corpus", "available": corpus_ok})
    if not corpus_ok:
        _emit_block(
            "blocked_real_task_corpus_missing",
            f"corpus not found at {CORPUS_PATH}",
            {"preconditions_checked": pre},
        )
        return

    # (b) energy-descent substrate trainable
    try:
        energy_model, substrate_signature, energy_gap = _train_energy_substrate()
    except Exception as exc:  # pragma: no cover - substrate-environment-dependent
        pre.append({"resource": "energy_descent_substrate", "available": False})
        _emit_block(
            "blocked_energy_descent_substrate_unavailable",
            f"Boltzmann-GPT training failed: {exc}",
            {"preconditions_checked": pre},
        )
        return
    pre.append({"resource": "energy_descent_substrate", "available": True})

    # (d) SOTA GGUF loads via the GGUF path (embedded tokenizer; NOT AutoTokenizer)
    model_path = resolve_cached_gguf(AR_MODEL_HF_ID)
    gguf_ok = model_path is not None and os.path.exists(model_path)
    if gguf_ok:
        try:
            import llama_cpp  # noqa: PLC0415

            probe = llama_cpp.Llama(model_path=model_path, vocab_only=True, verbose=False)
            probe.tokenize(b"2 + 2 = 4")
            del probe
        except Exception as exc:  # pragma: no cover - inference-environment-dependent
            pre.append({"resource": "sota_gguf_tokenizer", "available": False})
            _emit_block(
                "blocked_sota_gguf_tokenizer_unavailable",
                f"GGUF embedded-tokenizer probe failed for {AR_MODEL_HF_ID}: {exc}",
                {"preconditions_checked": pre},
            )
            return
    pre.append({"resource": "sota_gguf_tokenizer", "available": gguf_ok})
    if not gguf_ok:
        _emit_block(
            "blocked_sota_gguf_tokenizer_unavailable",
            f"GGUF for {AR_MODEL_HF_ID} not cached",
            {"preconditions_checked": pre},
        )
        return

    try:
        import llama_cpp  # noqa: PLC0415

        # THE exp3426 FIX: logits_all=True so per-token logprobs work for the
        # sampled candidates. Without it, every `logprobs=1` call raises and the
        # whole sampled set is silently lost (the exp3426 0.0-vs-0.0 tie).
        llm = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            seed=SEED,
            logits_all=True,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - inference-environment-dependent
        _emit_block(
            "blocked_sota_gguf_tokenizer_unavailable",
            f"llama.cpp failed to load {AR_MODEL_HF_ID}: {exc}",
            {"preconditions_checked": pre},
        )
        return

    # ----- Steps 1-6: paired four-condition scoring over the same problems -----
    problems = load_gsm8k_subset(CORPUS_PATH, n=N_PROBLEMS, seed=SEED)
    warmup_n = min(WARMUP_N, len(problems))
    per_problem: list[dict] = []
    gate_checked = False
    sc_gate = None

    for idx, problem in enumerate(problems):
        per_problem.append(_score_problem(llm, energy_model, problem))

        # ----- Step 0e: NON-DEGENERATE-SC gate (runs once, after warm-up) -----
        if not gate_checked and (idx + 1) >= warmup_n:
            gate_checked = True
            warm = per_problem[:warmup_n]
            sc_gate = evaluate_sc_non_degenerate(
                [r["ar_correct"] for r in warm],
                [r["sc_correct"] for r in warm],
            )
            print(
                f"[step0e] warm-up n={warmup_n} "
                f"SC={sc_gate.self_consistency_accuracy:.3f} "
                f"greedy={sc_gate.ar_greedy_accuracy:.3f} "
                f"passed={sc_gate.passed}"
            )
            if not sc_gate.passed:
                # Dump raw per-sample extracted answers for 3 example warm-up
                # problems so the bug is diagnosable, then STOP — do NOT report
                # an energy comparison against a broken self-consistency baseline.
                examples = [
                    {
                        "problem_id": r["problem_id"],
                        "gold": r["gold"],
                        "ar_pred": r["ar_pred"],
                        "candidate_preds": r["candidate_preds"],
                    }
                    for r in warm[:3]
                ]
                _emit_block(
                    "complete: blocked_self_consistency_harness_degenerate_"
                    "per_sample_extraction_broken",
                    sc_gate.reason,
                    {
                        "preconditions_checked": pre,
                        "self_consistency_non_degenerate": False,
                        "warmup_n": warmup_n,
                        "warmup_self_consistency_accuracy": round(
                            sc_gate.self_consistency_accuracy, 4
                        ),
                        "warmup_ar_greedy_accuracy": round(sc_gate.ar_greedy_accuracy, 4),
                        "raw_per_sample_examples": examples,
                    },
                )
                return

        if (idx + 1) % 25 == 0:
            ar = sum(1 for r in per_problem if r["ar_correct"]) / len(per_problem)
            sc = sum(1 for r in per_problem if r["sc_correct"]) / len(per_problem)
            ev = sum(1 for r in per_problem if r["evote_correct"]) / len(per_problem)
            print(f"[{idx + 1}/{len(problems)}] AR={ar:.3f} SC={sc:.3f} Evote={ev:.3f}")

    # ----- Step 7: accuracies + PRIMARY paired significance (energy vs SC) -----
    n = len(problems)
    ar_correct = [r["ar_correct"] for r in per_problem]
    sc_correct = [r["sc_correct"] for r in per_problem]
    scert_correct = [r["scert_correct"] for r in per_problem]
    eargmin_correct = [r["eargmin_correct"] for r in per_problem]
    evote_correct = [r["evote_correct"] for r in per_problem]

    ar_acc = sum(ar_correct) / n
    sc_acc = sum(sc_correct) / n
    scert_acc = sum(scert_correct) / n
    eargmin_acc = sum(eargmin_correct) / n
    evote_acc = sum(evote_correct) / n

    delta_vs_sc = evote_acc - sc_acc
    delta_vs_ar = evote_acc - ar_acc
    # PRIMARY paired test: energy-weighted vote vs self-consistency.
    mcnemar = mcnemar_test(sc_correct, evote_correct)
    ci_lo, ci_hi = paired_bootstrap_ci(sc_correct, evote_correct, n_boot=2000, seed=SEED)
    verdict = derive_premise_v3_verdict(
        sc_gate,
        sc_acc,
        evote_acc,
        mcnemar["p_value"],
        (ci_lo, ci_hi),
        direction=mcnemar["direction"],
    )

    checksum = reproducibility_checksum(
        corpus_path=CORPUS_PATH,
        n_problems=n,
        seed=SEED,
        substrate_signature=substrate_signature,
    )

    compute_parity_note = (
        f"All conditions use the same base model {AR_MODEL_HF_ID} (Qwen3.6-35B-A3B "
        f"MoE, ~3B active params, Q4_K_M) on the same {n} paired problems. Greedy AR "
        f"= 1 greedy generation/problem (the matched-compute floor). Conditions 2-4 "
        f"(self-consistency, self-certainty BoN, energy-argmin, energy-weighted vote) "
        f"ALL consume the SAME {K_SAMPLES} sampled generations/problem (temp 0.8); "
        f"only the selection/aggregation differs — energy gets NO extra samples. The "
        f"energy conditions add only a {VISIBLE_DIM}-dim latent gradient descent "
        f"({DESCENT_STEPS} steps, negligible FLOPs, no token sampling in-loop). The "
        f"model is loaded with logits_all=True so per-token logprobs work (the exp3426 "
        f"fix). PRIMARY comparison is energy_weighted_vote vs self_consistency at "
        f"matched compute; delta_energy_vs_greedy_ar is reported only for continuity "
        f"with exp3312/exp3426."
    )

    artifact_data = {
        "honest_verdict": verdict.verdict,
        "inference_substrate": "live_llm_inference",
        "task_name": "GSM8K (original questions, exp281 corpus), held-out shuffled split",
        "n_problems": n,
        "k_samples": K_SAMPLES,
        "self_consistency_non_degenerate": bool(sc_gate.passed),
        "ar_greedy_accuracy": round(ar_acc, 4),
        "self_consistency_accuracy": round(sc_acc, 4),
        "self_certainty_bon_accuracy": round(scert_acc, 4),
        "energy_argmin_accuracy": round(eargmin_acc, 4),
        "energy_weighted_vote_accuracy": round(evote_acc, 4),
        "delta_energy_vs_self_consistency": round(delta_vs_sc, 4),
        "delta_energy_vs_greedy_ar": round(delta_vs_ar, 4),
        "paired_significance": {
            "comparison": "energy_weighted_vote_vs_self_consistency",
            "test": "mcnemar_exact_binomial + paired_bootstrap_ci",
            "energy_wins": int(mcnemar["energy_descent_wins"]),
            "self_consistency_wins": int(mcnemar["ar_wins"]),
            "p_value": round(mcnemar["p_value"], 6),
            "direction": int(mcnemar["direction"]),
            "bootstrap_delta_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        },
        "warmup_n": warmup_n,
        "warmup_self_consistency_accuracy": round(sc_gate.self_consistency_accuracy, 4),
        "warmup_ar_greedy_accuracy": round(sc_gate.ar_greedy_accuracy, 4),
        "g1_energy_non_inferior": verdict.g1_energy_non_inferior,
        "g2_energy_adds_value": verdict.g2_energy_adds_value,
        "compute_parity_note": compute_parity_note,
        "ar_model": AR_MODEL_HF_ID,
        "energy_substrate": substrate_signature,
        "energy_substrate_test_gap": round(energy_gap, 4),
        "energy_vote_temperature": ENERGY_VOTE_T,
        "descent_steps": DESCENT_STEPS,
        "max_tokens": MAX_TOKENS,
        "random_seed": SEED,
        "reproducibility_checksum": checksum,
        "duration_s": round(time.time() - start, 3),
        "preconditions_checked": pre,
        "model_specs": [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": AR_MODEL_HF_ID,
                "model_path": model_path,
                "quantization": "Q4_K_M",
            }
        ],
        "field_provenance": principles,
        "per_problem": per_problem,
    }

    artifact = tmpl.build_result(artifact_data, status="success")
    Path(DELIVERABLE_PATH).write_text(json.dumps(artifact, indent=2))
    tmpl.assert_deliverable_written()
    print(
        f"DONE: {verdict.verdict}\n"
        f"  AR(greedy)={ar_acc:.4f} SC={sc_acc:.4f} SCert={scert_acc:.4f} "
        f"Eargmin={eargmin_acc:.4f} Evote={evote_acc:.4f}\n"
        f"  PRIMARY delta(Evote-SC)={delta_vs_sc:+.4f} p={mcnemar['p_value']:.4f} "
        f"CI=[{ci_lo:.4f},{ci_hi:.4f}] dur={artifact_data['duration_s']}s"
    )
    _ = artifact


if __name__ == "__main__":
    main()
