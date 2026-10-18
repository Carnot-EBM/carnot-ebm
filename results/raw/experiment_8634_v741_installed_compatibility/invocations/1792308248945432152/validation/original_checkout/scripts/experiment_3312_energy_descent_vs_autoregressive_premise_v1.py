#!/usr/bin/env python3
"""Exp 3312 (P0.1): Energy-Descent Reasoning vs Autoregressive Baseline.

**The single most important experiment in the project.** The entire Phase-3 /
Kona endgame assumes that *energy-descent reasoning on continuous latents* beats
*autoregressive (AR) token sampling*. That premise has never been tested on a
real task against a real AR baseline — only toy 5x5 puzzles (exp1222) and a
downgraded BFS tie (exp1210). This script runs the paired head-to-head on a real
reasoning benchmark (GSM8K) with a falsifiable significance gate. Either outcome
is high value: validation greenlights Phase 3; refutation honestly retires the
foundation-model endgame and saves years.

How the two conditions are made apples-to-apples
------------------------------------------------
Both conditions use the SAME base GGUF model (Qwen3.6-35B-A3B) on the SAME
paired problems:

  * **AR condition** — the base model answers each problem autoregressively with
    greedy decoding (one generation per problem). This is the literal AR
    baseline.
  * **Energy-descent condition** — the base model produces N sampled candidate
    chains; each candidate's reasoning is projected into a *continuous latent*
    and refined by a bounded number of gradient-descent steps under a verifier
    energy trained contrastively on FoVer (correct reasoning -> low energy). No
    tokens are sampled inside the refinement loop (REQ-KONA-001/002); the answer
    is decoded only at the coda by selecting the minimum-energy candidate.

Because the energy-descent condition consumes an N-sample generation budget, we
ALSO report self-consistency (majority vote over the same N samples) as the
equal-compute AR control, and the ``compute_parity_note`` discloses the
asymmetry plainly so the comparison cannot be read as a bigger-budget win in
disguise.

Spec: REQ-KONA-3312, SCENARIO-KONA-3312, SCENARIO-KONA-3312-BLOCKED.
Run: .venv/bin/python scripts/experiment_3312_energy_descent_vs_autoregressive_premise_v1.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.inference.sota_models import resolve_cached_gguf  # noqa: E402
from carnot.phase3.energy_descent_premise import (  # noqa: E402
    derive_premise_verdict,
    energy_descent_select,
    extract_final_answer,
    is_correct,
    load_gsm8k_subset,
    majority_vote,
    mcnemar_test,
    paired_bootstrap_ci,
    reproducibility_checksum,
)

DELIVERABLE_PATH = "results/experiment_3312_energy_descent_vs_autoregressive_premise_v1.json"
CORPUS_PATH = project_root / "data" / "research" / "gsm8k_adversarial_281.jsonl"
AR_MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"

# Tunables (env-overridable so a smoke run is cheap; defaults are the real run).
N_PROBLEMS = int(os.environ.get("EXP3312_N", "200"))
N_CANDIDATES = int(os.environ.get("EXP3312_NCAND", "3"))
MAX_TOKENS = int(os.environ.get("EXP3312_MAXTOK", "512"))
DESCENT_STEPS = int(os.environ.get("EXP3312_DESCENT_STEPS", "8"))
SEED = int(os.environ.get("EXP3312_SEED", "3312"))
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
        "inference_substrate": "live_llm_inference: the AR baseline really loads + runs the GGUF.",
        "task_name": "Name the real benchmark + split; toy/synthetic is not acceptable here.",
        "n_problems": ">=200 for a CLT-valid accuracy delta.",
        "ar_baseline_accuracy": "The autoregressive control; same problems, greedy decode.",
        "energy_descent_accuracy": "The non-AR condition; the premise under test.",
        "accuracy_delta": "energy_descent minus AR — the headline.",
        "paired_significance": "McNemar exact p + paired bootstrap CI; unpaired/n<200 is gameable.",
        "compute_parity_note": "State param-count + per-condition generation budget so the "
        "comparison is apples-to-apples, not a bigger-budget win.",
        "random_seed": "Determinism precondition for reproducibility.",
        "reproducibility_checksum": "Content hash of corpus + substrate + seed.",
        "duration_s": "Real training+inference takes wall time; 60s live-inference floor.",
    }


def _train_energy_substrate():
    """Train the Boltzmann-GPT continuous-latent energy on FoVer (correct->low E).

    Returns (model, substrate_signature) or raises if the substrate is untrainable.
    The energy is what makes the energy-descent condition a *learned* reasoning
    judge rather than an arbitrary scoring function: contrastive training pushes
    correct reasoning traces to low energy and incorrect ones to high energy, so
    descending the energy moves a candidate latent toward the correct manifold.
    """
    import torch  # noqa: PLC0415
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
    # Freeze parameters: the latent (not the model) is what descends per candidate.
    for param in model.parameters():
        param.requires_grad_(False)
    signature = f"boltzmann_gpt_v{VISIBLE_DIM}_e{FOVER_TRAIN_EPOCHS}_gap{gap:.4f}"
    return model, signature, float(gap)


def _generate(llm, prompt: str, *, temperature: float, seed: int) -> str:
    """One bounded llama.cpp generation. Returns text ('' on any failure)."""
    try:
        out = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            seed=seed,
            stop=["\nProblem:", "\n\nProblem"],
        )
        return out["choices"][0]["text"]
    except Exception:  # pragma: no cover - inference-environment-dependent
        return ""


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=3312,
        title="Energy-Descent Reasoning vs Autoregressive Baseline (the Kona premise test)",
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
            "task_name": "GSM8K (original split) — premise test",
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
        _ = artifact

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

    # (c) real corpus (checked before substrate train so a missing corpus is cheap)
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
        substrate_ok = True
    except Exception as exc:  # pragma: no cover - substrate-environment-dependent
        substrate_ok = False
        substrate_signature, energy_gap = "", 0.0
        pre.append({"resource": "energy_descent_substrate", "available": False})
        _emit_block(
            "blocked_energy_descent_substrate_unavailable",
            f"Boltzmann-GPT training failed: {exc}",
            {"preconditions_checked": pre},
        )
        return
    pre.append({"resource": "energy_descent_substrate", "available": substrate_ok})

    # (d) AR baseline GGUF runnable
    model_path = resolve_cached_gguf(AR_MODEL_HF_ID)
    baseline_ok = model_path is not None and os.path.exists(model_path)
    pre.append({"resource": "ar_baseline_gguf", "available": baseline_ok})
    if not baseline_ok:
        _emit_block(
            "blocked_ar_baseline_unavailable",
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
            "blocked_ar_baseline_unavailable",
            f"llama.cpp failed to load {AR_MODEL_HF_ID}: {exc}",
            {"preconditions_checked": pre},
        )
        return

    # ----- Steps 1-3: paired AR vs energy-descent over the same problems -----
    problems = load_gsm8k_subset(CORPUS_PATH, n=N_PROBLEMS, seed=SEED)
    ar_correct: list[bool] = []
    ed_correct: list[bool] = []
    sc_correct: list[bool] = []  # self-consistency (equal-compute control)
    per_problem: list[dict] = []

    for idx, problem in enumerate(problems):
        prompt = PROMPT_TEMPLATE.format(q=problem.question)

        # AR condition: single greedy generation.
        ar_text = _generate(llm, prompt, temperature=0.0, seed=SEED)
        ar_pred = extract_final_answer(ar_text)
        ar_hit = is_correct(ar_pred, problem.answer)
        ar_correct.append(ar_hit)

        # Energy-descent condition: N sampled candidates -> latent descent -> select.
        candidates = [
            _generate(llm, prompt, temperature=0.8, seed=SEED + 1000 * (k + 1))
            for k in range(N_CANDIDATES)
        ]
        descent = energy_descent_select(
            candidates,
            energy_model,
            visible_dim=VISIBLE_DIM,
            n_steps=DESCENT_STEPS,
            lr=0.05,
        )
        ed_pred = extract_final_answer(candidates[descent.selected_index])
        ed_hit = is_correct(ed_pred, problem.answer)
        ed_correct.append(ed_hit)

        # Equal-compute control: majority vote over the SAME N candidate answers.
        cand_preds = [extract_final_answer(c) for c in candidates]
        sc_pred = majority_vote(cand_preds)
        sc_correct.append(is_correct(sc_pred, problem.answer))

        per_problem.append(
            {
                "problem_id": problem.problem_id,
                "gold": problem.answer,
                "ar_pred": ar_pred,
                "ar_correct": ar_hit,
                "ed_pred": ed_pred,
                "ed_correct": ed_hit,
                "ed_selected_index": descent.selected_index,
                "candidate_preds": cand_preds,
            }
        )
        if (idx + 1) % 25 == 0:
            print(
                f"[{idx + 1}/{len(problems)}] "
                f"AR={sum(ar_correct) / len(ar_correct):.3f} "
                f"ED={sum(ed_correct) / len(ed_correct):.3f} "
                f"SC={sum(sc_correct) / len(sc_correct):.3f}"
            )

    # ----- Step 4: accuracies + paired significance -----
    n = len(problems)
    ar_acc = sum(ar_correct) / n
    ed_acc = sum(ed_correct) / n
    sc_acc = sum(sc_correct) / n
    delta = ed_acc - ar_acc
    mcnemar = mcnemar_test(ar_correct, ed_correct)
    ci_lo, ci_hi = paired_bootstrap_ci(ar_correct, ed_correct, n_boot=2000, seed=SEED)
    verdict = derive_premise_verdict(
        ar_acc, ed_acc, mcnemar["p_value"], (ci_lo, ci_hi), direction=mcnemar["direction"]
    )

    checksum = reproducibility_checksum(
        corpus_path=CORPUS_PATH,
        n_problems=n,
        seed=SEED,
        substrate_signature=substrate_signature,
    )

    compute_parity_note = (
        f"Both conditions use the same base model {AR_MODEL_HF_ID} "
        f"(Qwen3.6-35B-A3B MoE, ~3B active params, Q4_K_M) on the same {n} paired "
        f"problems. AR baseline = 1 greedy generation/problem. Energy-descent = "
        f"{N_CANDIDATES} sampled generations/problem (temp 0.8) + {DESCENT_STEPS} "
        f"latent gradient-descent steps under a frozen FoVer-trained Boltzmann-GPT "
        f"energy (16-dim, negligible FLOPs). Energy-descent therefore spends "
        f"{N_CANDIDATES}x the generation budget of the greedy AR baseline; the "
        f"equal-compute AR control (self-consistency majority vote over the same "
        f"{N_CANDIDATES} samples) scored {sc_acc:.4f}. Read the headline delta "
        f"against ar_baseline_accuracy for the literal AR comparison and against "
        f"self_consistency_accuracy for the equal-budget comparison."
    )

    artifact_data = {
        "honest_verdict": verdict.verdict,
        "inference_substrate": "live_llm_inference",
        "task_name": "GSM8K (original questions, exp281 corpus), held-out shuffled split",
        "n_problems": n,
        "ar_baseline_accuracy": round(ar_acc, 4),
        "energy_descent_accuracy": round(ed_acc, 4),
        "self_consistency_accuracy": round(sc_acc, 4),
        "accuracy_delta": round(delta, 4),
        "accuracy_delta_vs_self_consistency": round(ed_acc - sc_acc, 4),
        "paired_significance": {
            "test": "mcnemar_exact_binomial + paired_bootstrap_ci",
            "energy_descent_wins": int(mcnemar["energy_descent_wins"]),
            "ar_wins": int(mcnemar["ar_wins"]),
            "p_value": round(mcnemar["p_value"], 6),
            "direction": int(mcnemar["direction"]),
            "bootstrap_delta_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        },
        "g1_premise_viable": verdict.g1_premise_viable,
        "g2_premise_validated": verdict.g2_premise_validated,
        "compute_parity_note": compute_parity_note,
        "ar_model": AR_MODEL_HF_ID,
        "energy_substrate": substrate_signature,
        "energy_substrate_test_gap": round(energy_gap, 4),
        "n_candidates": N_CANDIDATES,
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
        f"DONE: {verdict.verdict} | AR={ar_acc:.4f} ED={ed_acc:.4f} SC={sc_acc:.4f} "
        f"delta={delta:+.4f} p={mcnemar['p_value']:.4f} "
        f"CI=[{ci_lo:.4f},{ci_hi:.4f}] dur={artifact_data['duration_s']}s"
    )
    _ = artifact


if __name__ == "__main__":
    main()
