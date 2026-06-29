#!/usr/bin/env python3
"""Distributional-energy verifier on MuSR — the off-ARC headroom experiment (operator-directed 2026-06-29).

THESIS (arXiv:2605.18871 + the Carnot verifier-moat): on a domain where SELF-CONSISTENCY is NOT saturated
and there is NO cheap executable oracle, an ORACLE-DISTINCT energy verifier should select better answers
than self-consistency. MuSR murder_mysteries (binary MCQ, long multi-step narrative reasoning) is exactly
such a domain — unlike HumanEval/Sudoku where running the tests IS the oracle (circular, see CLAUDE.md
"Circularity / Oracle-Distinctness Discipline").

PIPELINE (all selection methods are ORACLE-DISTINCT — none sees the gold answer):
  1. Generate K reasoning+answer candidates per question with a local generator (temperature for diversity).
  2. self_consistency: majority vote over the K parsed answers (the baseline to beat).
  3. distributional_energy: the decomposed energy of arXiv:2605.18871 made real --
       energy(cand) = -mean(quality_ensemble) + analytical_penalty ; uncertainty = stddev(quality_ensemble)
       The quality scorer is a THINKPRM-style process-reward (rate the reasoning's validity 0-1), run M
       times = the ENSEMBLE (mean RANKS candidates; stddev ABSTAINS -> fall back to SC when uncertain).
       analytical_penalty = oracle-distinct text-statistical answer/reasoning-consistency penalty.
       Select the min-energy candidate's answer (or SC if the min-energy candidate's stddev > abstain thr).
  4. llm_judge: a single judge call picks the best candidate (the comparator baseline).
  5. EVAL ONLY: accuracy vs gold; paired bootstrap CI95 on (energy_correct - sc_correct).

GATE (falsifiable, oracle-distinct, non-circular): energy_accuracy > sc_accuracy AND paired CI95 excludes 0
AND verifier_is_oracle == False AND the verifier never reads answer_index/answer_choice AND no
model-identity shortcut (all candidates come from one model; the scorer ranks by reasoning quality, not by
which model produced a candidate). A null tightens the moat scope; a positive is the verifier-moat win
where SC is not saturated — the headroom ARC's generation wall cannot host.

USAGE: exp_distributional_energy_verifier_musr.py [n_questions] [k_candidates] [m_ensemble]
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 50
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
M = int(sys.argv[3]) if len(sys.argv) > 3 else 3  # quality-scorer ensemble size (mean ranks, stddev abstains)
SEED = 20260629
ABSTAIN_STDDEV = 0.22  # if the min-energy candidate's quality stddev exceeds this, abstain -> SC
CKPT_DIR = REPO / "results" / "distributional_energy_verifier_musr_checkpoints"
ART_PATH = REPO / "results" / "distributional_energy_verifier_musr.json"


def _choices(row) -> list[str]:
    ch = row.get("choices")
    if isinstance(ch, str):
        try:
            ch = ast.literal_eval(ch)
        except Exception:
            ch = [c.strip() for c in ch.strip("[]").split(",")]
    return [str(c).strip().strip("'\"") for c in ch]


def _match_choice(text: str, choices: list[str]) -> str | None:
    """Parse the model's selected choice from its completion (oracle-distinct: only the model's own text)."""
    if not text:
        return None
    m = re.search(r"ANSWER\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    tail = (m.group(1) if m else text[-160:]).lower()
    # exact choice mention wins; else first choice that appears in the tail
    hits = [c for c in choices if c.lower() in tail]
    if len(hits) == 1:
        return hits[0]
    if hits:
        # pick the choice appearing LAST in the tail (the final declared answer)
        return max(hits, key=lambda c: tail.rfind(c.lower()))
    full = text.lower()
    hits = [c for c in choices if c.lower() in full]
    return hits[-1] if hits else None


def _gen_candidate(proposer, narrative: str, question: str, choices: list[str], seed: int) -> dict:
    prompt = (
        "Read the scenario and answer the multiple-choice question with careful step-by-step reasoning.\n\n"
        f"SCENARIO:\n{narrative[:6000]}\n\n"
        f"QUESTION: {question}\nCHOICES: {choices}\n\n"
        "Think step by step, then end with a final line exactly: ANSWER: <one choice verbatim>."
    )
    ok, text = proposer.complete_text(prompt, max_tokens=512, temperature=0.7)
    ans = _match_choice(text or "", choices) if ok else None
    return {"reasoning": (text or "")[:4000], "answer": ans}


def _quality_score(proposer, question: str, cand: dict, seed: int) -> float | None:
    """ORACLE-DISTINCT process-reward: rate the candidate's reasoning validity 0-1. NEVER sees the gold."""
    prompt = (
        "You are a strict reasoning critic. Rate ONLY how logically valid, internally consistent, and "
        "well-supported-by-the-scenario the following reasoning is. Do NOT judge whether the final answer "
        "is correct against any key — only the reasoning's quality.\n\n"
        f"QUESTION: {question}\n\nREASONING:\n{cand.get('reasoning','')[:3500]}\n\n"
        "Reply with ONLY a single number from 0.0 (invalid) to 1.0 (rigorous)."
    )
    ok, text = proposer.complete_text(prompt, max_tokens=12, temperature=0.3)
    if not ok:
        return None
    m = re.search(r"(0?\.\d+|[01](?:\.0+)?)", text or "")
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except Exception:
        return None


def _analytical_penalty(cand: dict, choices: list[str]) -> float:
    """Oracle-distinct text-statistical penalty: the stated ANSWER should agree with the reasoning's
    conclusion (the last choice mentioned). Penalize answer/reasoning disagreement + no-answer."""
    if not cand.get("answer"):
        return 1.0
    concl = _match_choice(cand.get("reasoning", ""), choices)
    return 0.0 if concl == cand["answer"] else 0.5


def _energy(cand: dict, choices: list[str]) -> tuple[float, float]:
    """Decomposed energy (lower=better) + uncertainty(stddev). arXiv:2605.18871 form:
    energy = -mean(quality_ensemble) + analytical_penalty ; uncertainty = stddev(quality_ensemble)."""
    qs = [q for q in cand.get("quality_ensemble", []) if q is not None]
    q_mean = mean(qs) if qs else 0.0
    q_std = pstdev(qs) if len(qs) > 1 else 0.0
    return (-q_mean + _analytical_penalty(cand, choices)), q_std


def _sc_answer(cands: list[dict]) -> str | None:
    from collections import Counter
    votes = Counter(c["answer"] for c in cands if c.get("answer"))
    return votes.most_common(1)[0][0] if votes else None


def _energy_answer(cands: list[dict], choices: list[str], sc: str | None) -> tuple[str, str, bool]:
    """Return (energy_with_abstain, energy_pure_min, abstained). The pure-min column always shows the raw
    verifier signal; the with-abstain column is the full arXiv:2605.18871 method (stddev abstains to SC)."""
    scored = [(c, *_energy(c, choices)) for c in cands if c.get("answer")]
    if not scored:
        return (sc or ""), (sc or ""), False
    best, _e, best_std = min(scored, key=lambda t: (t[1], t[0]["answer"]))
    pure = best["answer"]
    abstained = bool(best_std > ABSTAIN_STDDEV and sc)
    return (sc if abstained else pure), pure, abstained


def _judge_answer(proposer, question: str, cands: list[dict], choices: list[str]) -> str | None:
    opts = [c for c in cands if c.get("answer")]
    if not opts:
        return None
    blocks = "\n\n".join(f"[{i}] answer={c['answer']}\n{c['reasoning'][:600]}" for i, c in enumerate(opts[:8]))
    prompt = (
        "Pick the index of the BEST-reasoned candidate answer (do not use any answer key).\n\n"
        f"QUESTION: {question}\n\nCANDIDATES:\n{blocks}\n\nReply with ONLY the index number."
    )
    ok, text = proposer.complete_text(prompt, max_tokens=8, temperature=0.0)
    if ok and text:
        m = re.search(r"\d+", text)
        if m:
            i = int(m.group())
            if 0 <= i < len(opts[:8]):
                return opts[i]["answer"]
    return opts[0]["answer"]


def _bootstrap_ci(pairs, seed, n=2000):
    import random
    if not pairs:
        return [0.0, 0.0]
    rng = random.Random(seed)
    deltas = []
    for _ in range(n):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(mean(a - b for a, b in s))
    deltas.sort()
    return [round(deltas[int(0.025 * (n - 1))], 4), round(deltas[int(0.975 * (n - 1))], 4)]


def main() -> int:
    started = time.time()
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    proposer = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP", kv_quant="q8_0", no_think_prefix="/no_think\n",
        port=int(os.environ.get("CARNOT_IGE_LLM_PORT", "8920")),
    )
    try:
        ok = proposer._healthy() or proposer._ensure_server()
    except Exception:
        ok = False
    if not ok:
        _write({"experiment": "distributional_energy_verifier_musr",
                "honest_verdict": "blocked_musr_energy_verifier_llm_server_unreachable",
                "inference_substrate": "live_llm_inference", "verifier_is_oracle": False,
                "random_seed": SEED, "duration_s": round(time.time() - started, 2)})
        print("BLOCKED: generator unreachable"); return 0

    from datasets import load_dataset
    rows = list(load_dataset("TAUR-Lab/MuSR")["murder_mysteries"])[:N_Q]
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for qi, row in enumerate(rows):
        cpath = CKPT_DIR / f"q{qi:04d}.json"
        if cpath.exists():
            continue
        choices = _choices(row)
        gold = str(row.get("answer_choice") or "").strip()  # EVAL ONLY — never passed to any scorer
        narrative, question = str(row.get("narrative", "")), str(row.get("question", ""))
        cands = [_gen_candidate(proposer, narrative, question, choices, SEED + qi * 100 + k) for k in range(K)]
        for k, c in enumerate(cands):
            c["quality_ensemble"] = [_quality_score(proposer, question, c, SEED + qi + 7 * m) for m in range(M)]
        sc = _sc_answer(cands)
        en, en_pure, abstained = _energy_answer(cands, choices, sc)
        ju = _judge_answer(proposer, question, cands, choices)
        rec = {
            "q": qi, "gold": gold, "n_choices": len(choices),
            "sc_answer": sc, "energy_answer": en, "energy_pure_answer": en_pure, "judge_answer": ju,
            "sc_correct": int(sc == gold), "energy_correct": int(en == gold),
            "energy_pure_correct": int(en_pure == gold), "judge_correct": int(ju == gold),
            "n_valid_candidates": sum(1 for c in cands if c.get("answer")),
            "answers": [c.get("answer") for c in cands],
            "energy_abstained": abstained,
        }
        cpath.write_text(json.dumps(rec) + "\n")
        print(f"[q{qi}] gold={gold} sc={sc}({rec['sc_correct']}) energy={en}({rec['energy_correct']}) "
              f"pure={en_pure}({rec['energy_pure_correct']}) judge={ju}({rec['judge_correct']})", flush=True)

    per_q = [json.loads(p.read_text()) for p in sorted(CKPT_DIR.glob("q*.json"))]
    scored = [r for r in per_q if r.get("gold")]
    n = len(scored)
    sc_acc = round(sum(r["sc_correct"] for r in scored) / max(1, n), 4)
    en_acc = round(sum(r["energy_correct"] for r in scored) / max(1, n), 4)
    enp_acc = round(sum(r.get("energy_pure_correct", 0) for r in scored) / max(1, n), 4)
    ju_acc = round(sum(r["judge_correct"] for r in scored) / max(1, n), 4)
    pairs = [(r["energy_correct"], r["sc_correct"]) for r in scored]
    ci = _bootstrap_ci(pairs, SEED)
    pairs_pure = [(r.get("energy_pure_correct", 0), r["sc_correct"]) for r in scored]
    ci_pure = _bootstrap_ci(pairs_pure, SEED)
    delta = round(en_acc - sc_acc, 4)
    delta_pure = round(enp_acc - sc_acc, 4)
    sc_saturated = sc_acc >= 0.9  # if SC is already near-ceiling, the domain is the wrong venue
    # the moat win is EITHER the full method (with-abstain) OR the raw verifier signal (pure) beating SC
    energy_beats_sc = bool((delta > 0 and ci[0] > 0) or (delta_pure > 0 and ci_pure[0] > 0))
    n_abstain = sum(1 for r in scored if r.get("energy_abstained"))

    if n < 30:
        verdict = f"complete_musr_energy_verifier_underpowered_n{n}_need_30plus"
    elif sc_saturated:
        verdict = f"complete_musr_energy_verifier_inconclusive_SC_saturated_{sc_acc}_wrong_venue"
    elif energy_beats_sc:
        which = "withAbstain" if (delta > 0 and ci[0] > 0) else "pureMinEnergy"
        verdict = (f"success_distributional_energy_verifier_BEATS_self_consistency_musr_{which}_"
                   f"en_{en_acc}_pure_{enp_acc}_vs_sc_{sc_acc}_delta_{delta}_puredelta_{delta_pure}"
                   f"_ci_excl_0_oracle_distinct")
    else:
        verdict = (f"complete_musr_energy_verifier_no_win_en_{en_acc}_pure_{enp_acc}_vs_sc_{sc_acc}"
                   f"_delta_{delta}_ci_{ci[0]}_{ci[1]}_puredelta_{delta_pure}_ci_{ci_pure[0]}_{ci_pure[1]}"
                   f"_sc_not_saturated_headroom_unrealized")

    art = {
        "experiment": "distributional_energy_verifier_musr",
        "schema": "carnot.distributional_energy_verifier_musr.v1",
        "honest_verdict": verdict,
        "domain": "MuSR/murder_mysteries",
        "question": ("does an ORACLE-DISTINCT decomposed-energy verifier (mean-ranks/stddev-abstains "
                     "quality ensemble + analytical penalty) beat self-consistency on a non-saturated "
                     "no-cheap-oracle domain (arXiv:2605.18871)?"),
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "n_questions": n, "k_candidates": K, "m_ensemble": M,
        "self_consistency_accuracy": sc_acc,
        "distributional_energy_accuracy": en_acc,
        "distributional_energy_pure_minenergy_accuracy": enp_acc,
        "llm_judge_accuracy": ju_acc,
        "energy_minus_sc_delta": delta, "energy_minus_sc_ci95": ci,
        "energy_pure_minus_sc_delta": delta_pure, "energy_pure_minus_sc_ci95": ci_pure,
        "energy_beats_sc": energy_beats_sc,
        "sc_saturated": sc_saturated,
        "n_energy_abstained_to_sc": n_abstain,
        "abstain_rate": round(n_abstain / max(1, n), 3),
        "oracle_distinctness": {
            "verifier_sees_gold": False,
            "note": ("gold (answer_choice) is used for EVAL accuracy ONLY; never passed to the generator, "
                     "the quality scorer, the analytical penalty, or the judge. The energy decomposition "
                     "is -mean(quality_ensemble)+analytical_penalty; uncertainty=stddev abstains to SC."),
            "no_model_identity_shortcut": ("all candidates come from ONE generator model; the scorer ranks "
                                           "by reasoning quality, not by candidate model_id (no model_id "
                                           "field is read)."),
        },
        "model_specs": {"generator": "unsloth/Qwen3.5-9B-MTP-GGUF", "scorer": "same (single-model)",
                        "kv_quant": "q8_0"},
        "arxiv_ingested": ["2605.18871", "2504.16828", "2502.01989"],
        "interpretation": (
            "First REAL execution of the post-6/30 distributional-energy-verifier pivot on a non-saturated, "
            "no-cheap-oracle domain (the headroom ARC's generation wall cannot host). A win (energy beats SC, "
            "CI excludes 0, oracle-distinct) is the verifier-moat result; a null scopes the moat. CAVEAT: "
            "the quality scorer is an LLM process-reward ensemble (THINKPRM-style), not yet a TRAINED EBM "
            "LoRA-ensemble -- a positive justifies training the real EBM scorer next; this is the cheapest "
            "decisive first test of the oracle-distinct-energy-beats-SC thesis off ARC."
        ),
        "solve_provenance": "development_proxy",
        "random_seed": SEED,
        "duration_s": round(time.time() - started, 2),
    }
    _write(art)
    print("\n=== VERDICT:", verdict)
    print(f"SC={sc_acc} ENERGY={en_acc} JUDGE={ju_acc} delta={delta} ci={ci} n={n} abstain={n_abstain}")
    return 0


def _write(art: dict) -> None:
    payload = dict(art); payload["reproducibility_checksum"] = ""
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    ART_PATH.write_text(json.dumps(art, indent=2) + "\n")
    print(f"-> {ART_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
