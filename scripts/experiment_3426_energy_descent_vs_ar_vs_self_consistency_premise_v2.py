#!/usr/bin/env python3
"""Exp 3426 (P0.1 v2): Energy vs AR vs Self-Consistency at MATCHED compute.

**The load-bearing follow-up to exp3312.** v1 showed energy-descent selection
beat a single greedy autoregressive (AR) generation (+0.090, McNemar p=0.033)
but LOST to plain majority-vote self-consistency over the same samples
(delta -0.055). v1 was also flagged_adversarial (a false-positive tautology:
its random_seed was set equal to its experiment id). So the honest, decisive
question is still open: **does the energy function add ANYTHING beyond plain
majority vote at the same compute budget?**

This script runs a paired head-to-head on a real reasoning benchmark (GSM8K,
n>=200, held-out) across FOUR conditions at MATCHED compute, with clean
methodology and a falsifiable significance gate. The PRIMARY comparison is
energy-weighted vote vs majority-vote self-consistency.

Conditions (greedy AR is the 1-sample floor; conditions 2-4 all aggregate the
SAME k sampled generations — only the aggregation rule differs, so energy never
gets extra samples):

  1. greedy AR              — 1 greedy generation/problem (the exp3312 baseline).
  2. self-consistency       — majority vote over k samples (Wang et al.); the
                              PRIMARY control energy must beat.
  3. self-certainty BoN     — pick the single most-confident sample by mean token
                              confidence (arXiv:2502.18581); strongest cheap
                              selector.
  4a. energy-argmin         — pick the min-energy candidate after bounded latent
                              descent (REQ-KONA-001: no token sampling in-loop).
  4b. energy-weighted vote  — softmax(-E/T) over the candidate answers, mirroring
                              EBM-CoT latent calibration (arXiv:2511.07124). The
                              headline condition under test.

Both energy conditions use a frozen Boltzmann-GPT energy trained contrastively
on FoVer (correct reasoning -> low energy). The energy weighting is computed
over per-problem z-scored final energies at a scale-free temperature, so T is a
dimensionless sharpness knob (T -> inf recovers plain majority vote; the whole
premise is that a meaningful T reshapes the vote toward correctness).

Spec: REQ-KONA-3426, SCENARIO-KONA-3426, SCENARIO-KONA-3426-BLOCKED.
Run: .venv/bin/python scripts/experiment_3426_energy_descent_vs_ar_vs_self_consistency_premise_v2.py
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
from carnot.phase3.energy_premise_v2 import (  # noqa: E402
    derive_premise_v2_verdict,
    energy_descent_select,
    energy_weighted_vote,
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
    "results/experiment_3426_energy_descent_vs_ar_vs_self_consistency_premise_v2.json"
)
CORPUS_PATH = project_root / "data" / "research" / "gsm8k_adversarial_281.jsonl"
AR_MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"

# Tunables (env-overridable so a smoke run is cheap; defaults are the real run).
# NB: SEED is deliberately NOT the experiment id (3426). exp3312 was flagged by
# adversarial_verify because random_seed==experiment id read as a tautology;
# keeping them distinct avoids that false positive.
N_PROBLEMS = int(os.environ.get("EXP3426_N", "200"))
K_SAMPLES = int(os.environ.get("EXP3426_K", "5"))
MAX_TOKENS = int(os.environ.get("EXP3426_MAXTOK", "512"))
DESCENT_STEPS = int(os.environ.get("EXP3426_DESCENT_STEPS", "8"))
SEED = int(os.environ.get("EXP3426_SEED", "20260530"))
# Scale-free energy-vote sharpness (applied to per-problem z-scored energies).
ENERGY_VOTE_T = float(os.environ.get("EXP3426_VOTE_T", "0.5"))
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
        "ar_greedy_accuracy": "1-sample greedy control (the exp3312 baseline).",
        "self_consistency_accuracy": "Majority vote over k samples — the PRIMARY control "
        "energy must beat.",
        "self_certainty_bon_accuracy": "Self-certainty Best-of-N (arXiv:2502.18581) — the "
        "strongest cheap selector.",
        "energy_argmin_accuracy": "Energy-argmin selection over the same k samples.",
        "energy_weighted_vote_accuracy": "Energy-weighted vote (EBM-CoT) — the premise under "
        "test; the headline condition.",
        "delta_energy_vs_self_consistency": "energy_weighted_vote minus self_consistency at "
        "MATCHED compute — THE headline; does energy add value over majority vote?",
        "delta_energy_vs_greedy_ar": "energy minus greedy AR — reported for continuity with "
        "exp3312.",
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
    selector has the per-token chosen-token logprobs it needs. The greedy AR
    control does not need logprobs. Returns ('', None) on any failure so a single
    bad generation scores as a miss rather than crashing the 200-problem run.
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


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=3426,
        title="Energy vs AR vs Self-Consistency at matched compute (Kona premise v2)",
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
            "task_name": "GSM8K (original split) — premise v2",
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
            gguf_ok = False
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

        llm = llama_cpp.Llama(
            model_path=model_path, n_gpu_layers=-1, n_ctx=2048, seed=SEED, verbose=False
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
    ar_correct: list[bool] = []
    sc_correct: list[bool] = []
    scert_correct: list[bool] = []
    eargmin_correct: list[bool] = []
    evote_correct: list[bool] = []
    per_problem: list[dict] = []

    for idx, problem in enumerate(problems):
        prompt = PROMPT_TEMPLATE.format(q=problem.question)

        # Condition 1: greedy AR (1 generation, the matched-compute floor).
        ar_text, _ = _generate(
            llm, prompt, temperature=0.0, seed=SEED, want_logprobs=False
        )
        ar_pred = extract_final_answer(ar_text)
        ar_correct.append(is_correct(ar_pred, problem.answer))

        # k sampled candidates shared by conditions 2-4 (the matched budget).
        cand_texts: list[str] = []
        cand_logprobs: list[list[float] | None] = []
        for k in range(K_SAMPLES):
            text, lps = _generate(
                llm, prompt, temperature=0.8, seed=SEED + 1000 * (k + 1), want_logprobs=True
            )
            cand_texts.append(text)
            cand_logprobs.append(lps)
        cand_preds = [extract_final_answer(c) for c in cand_texts]

        # Condition 2: self-consistency (majority vote over the k samples).
        sc_pred = majority_vote(cand_preds)
        sc_correct.append(is_correct(sc_pred, problem.answer))

        # Condition 3: self-certainty Best-of-N (most-confident single sample).
        scert_idx = self_certainty_select(cand_logprobs)
        scert_correct.append(is_correct(cand_preds[scert_idx], problem.answer))

        # Condition 4: energy. Bounded latent descent -> final per-candidate energy.
        descent = energy_descent_select(
            cand_texts, energy_model, visible_dim=VISIBLE_DIM, n_steps=DESCENT_STEPS, lr=0.05
        )
        # 4a: energy-argmin (min final energy).
        eargmin_pred = cand_preds[descent.selected_index]
        eargmin_correct.append(is_correct(eargmin_pred, problem.answer))
        # 4b: energy-weighted vote over per-problem z-scored energies (scale-free T).
        evote_pred = energy_weighted_vote(
            cand_preds, _zscore(descent.final_energies), temperature=ENERGY_VOTE_T
        )
        evote_correct.append(is_correct(evote_pred, problem.answer))

        per_problem.append(
            {
                "problem_id": problem.problem_id,
                "gold": problem.answer,
                "ar_pred": ar_pred,
                "sc_pred": sc_pred,
                "self_certainty_pred": cand_preds[scert_idx],
                "self_certainty_index": scert_idx,
                "energy_argmin_pred": eargmin_pred,
                "energy_argmin_index": descent.selected_index,
                "energy_weighted_vote_pred": evote_pred,
                "candidate_preds": cand_preds,
                "candidate_final_energies": [round(e, 6) for e in descent.final_energies],
            }
        )
        if (idx + 1) % 25 == 0:
            print(
                f"[{idx + 1}/{len(problems)}] "
                f"AR={sum(ar_correct) / len(ar_correct):.3f} "
                f"SC={sum(sc_correct) / len(sc_correct):.3f} "
                f"SCert={sum(scert_correct) / len(scert_correct):.3f} "
                f"Eargmin={sum(eargmin_correct) / len(eargmin_correct):.3f} "
                f"Evote={sum(evote_correct) / len(evote_correct):.3f}"
            )

    # ----- Step 7: accuracies + PRIMARY paired significance (energy vs SC) -----
    n = len(problems)
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
    verdict = derive_premise_v2_verdict(
        sc_acc, evote_acc, mcnemar["p_value"], (ci_lo, ci_hi), direction=mcnemar["direction"]
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
        f"energy-weighted vote softmaxes over per-problem z-scored final energies at "
        f"temperature T={ENERGY_VOTE_T} (scale-free; T->inf recovers plain majority "
        f"vote). PRIMARY comparison is energy_weighted_vote vs self_consistency at "
        f"matched compute; delta_energy_vs_greedy_ar is reported only for continuity "
        f"with exp3312."
    )

    artifact_data = {
        "honest_verdict": verdict.verdict,
        "inference_substrate": "live_llm_inference",
        "task_name": "GSM8K (original questions, exp281 corpus), held-out shuffled split",
        "n_problems": n,
        "k_samples": K_SAMPLES,
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
