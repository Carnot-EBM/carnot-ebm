"""Fresh, genuinely-real multi-candidate headroom pre-check on MMLU-Pro (outer-loop, item 2).

CONTEXT: the oracle-distinct-headroom-present moat corpus search has hit three dead ends this
session: MuSR (SC near-ceiling, no headroom), FoVer (headroom claim was a construction artifact --
no natural multi-candidate structure), and ConstraintBench/exp5044 (candidates are NOT real LLM
samples -- generator_kind="deterministic_solver_backed_variant", generation_model=None, alternating
correct/incorrect by construction). A repo-wide grep for any non-deterministic, real-LLM-generated
multi-candidate pool came back EMPTY -- no reusable corpus exists.

This script generates a SMALL, GENUINELY REAL corpus from scratch: real LLM sampling (temperature>0,
varying seeds -> genuine epistemic diversity, not synthetic templates), on MMLU-Pro (specifically
constructed to be harder than base MMLU -- 10-way multiple choice, more reasoning-heavy, a domain
where self-consistency is NOT expected to be near-ceiling the way it is on MuSR). Ground truth is
MMLU-Pro's own answer_index (real human-curated multiple-choice labels, exact match scoring --
non-circular for a text-based verifier since the verifier would not itself BE the labeling process).

Purpose: compute oracle@K vs self-consistency-vote HONESTLY on real generations, before investing in
building any verifier. If headroom is near-zero (SC already near-ceiling, like MuSR) or if oracle@K
barely exceeds SC (little room for a verifier to add value), that is itself the honest, valuable
result -- report it rather than proceed to build something with no headroom to capture.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "results" / "experiment_mmlu_pro_fresh_headroom_check.json"
MODEL_PATH = str(
    Path.home()
    / ".cache/huggingface/hub/models--unsloth--gemma-4-12B-it-GGUF/snapshots/"
    "3f09de26549e6d7ea54f1b83755149f840fcd333/gemma-4-12b-it-Q4_K_M.gguf"
)
N_QUESTIONS = 40
K_SAMPLES = 6
TEMPERATURE = 0.8
SEED = 20260701
GPU_DEVICE = 1  # outer-loop's dedicated GPU per CLAUDE.md's GPU-allocation rule


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_prompt(question: str, options: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    opts_block = "\n".join(f"{letter}. {opt}" for letter, opt in zip(letters, options))
    return (
        "Answer the following multiple-choice question. Think briefly, then give your final answer "
        "as a single capital letter on the last line in the exact form 'ANSWER: X'.\n\n"
        f"Question: {question}\n\n{opts_block}\n\nANSWER:"
    )


def parse_letter(text: str, n_options: int) -> str | None:
    valid = {chr(ord("A") + i) for i in range(n_options)}
    # look for the LAST "ANSWER: X" occurrence (in case the model repeats the prompt format while reasoning)
    idx = text.rfind("ANSWER:")
    tail = text[idx:] if idx >= 0 else text
    for ch in tail:
        if ch in valid:
            return ch
    # fallback: scan the whole text for the last standalone valid letter
    for ch in reversed(text):
        if ch in valid:
            return ch
    return None


def main() -> int:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)
    from datasets import load_dataset
    from llama_cpp import Llama

    _log(f"loading MMLU-Pro test split (sampling {N_QUESTIONS} questions, seed={SEED})")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.shuffle(seed=SEED).select(range(N_QUESTIONS))
    questions = [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],  # letter, e.g. "C"
            "category": row["category"],
        }
        for row in ds
    ]
    _log(f"loaded {len(questions)} questions, categories: {sorted({q['category'] for q in questions})}")

    _log(f"loading gemma-4-12B-it-GGUF from {MODEL_PATH} on GPU {GPU_DEVICE}")
    t0 = time.time()
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)
    load_duration_s = time.time() - t0
    _log(f"model loaded in {load_duration_s:.1f}s")

    rows = []
    t_gen_start = time.time()
    for qi, q in enumerate(questions):
        prompt = build_prompt(q["question"], q["options"])
        candidates = []
        for k in range(K_SAMPLES):
            out = llm(
                prompt,
                max_tokens=200,
                temperature=TEMPERATURE,
                seed=SEED + 1000 * qi + k,
                stop=["</s>", "<end_of_turn>"],
            )
            text = out["choices"][0]["text"]
            letter = parse_letter(text, len(q["options"]))
            candidates.append({"k": k, "raw_text": text[-120:], "parsed_letter": letter})
        n_parsed = sum(1 for c in candidates if c["parsed_letter"] is not None)
        rows.append({**q, "candidates": candidates, "n_parsed": n_parsed})
        if (qi + 1) % 5 == 0:
            elapsed = time.time() - t_gen_start
            _log(f"  {qi + 1}/{len(questions)} questions done ({elapsed:.0f}s elapsed)")

    gen_duration_s = time.time() - t_gen_start
    _log(f"generation done: {gen_duration_s:.1f}s for {len(questions) * K_SAMPLES} calls")

    # score: oracle@K = fraction of questions where AT LEAST ONE parsed candidate matches gold.
    # sc_vote = fraction where the MAJORITY parsed answer matches gold (ties broken by first-seen).
    oracle_hits = 0
    sc_hits = 0
    scored_questions = 0
    unparseable_questions = 0
    for row in rows:
        letters = [c["parsed_letter"] for c in row["candidates"] if c["parsed_letter"] is not None]
        if not letters:
            unparseable_questions += 1
            continue
        scored_questions += 1
        gold = row["answer"]
        if gold in letters:
            oracle_hits += 1
        vote = Counter(letters).most_common(1)[0][0]
        if vote == gold:
            sc_hits += 1

    oracle_at_k = oracle_hits / scored_questions if scored_questions else 0.0
    sc_vote = sc_hits / scored_questions if scored_questions else 0.0
    headroom = oracle_at_k - sc_vote

    artifact = {
        "experiment": "mmlu_pro_fresh_headroom_check",
        "corpus": "TIGER-Lab/MMLU-Pro (test split, random sample)",
        "n_questions_sampled": N_QUESTIONS,
        "n_questions_scored": scored_questions,
        "n_questions_unparseable": unparseable_questions,
        "k_samples_per_question": K_SAMPLES,
        "temperature": TEMPERATURE,
        "oracle_at_k": round(oracle_at_k, 4),
        "sc_vote": round(sc_vote, 4),
        "headroom": round(headroom, 4),
        "headroom_definition": "oracle_at_k - sc_vote = selectable headroom a verifier could capture",
        "oracle_distinct": True,
        "oracle_distinct_note": (
            "Ground truth is MMLU-Pro's own human-curated answer_index, an exact-match multiple-"
            "choice label -- NOT an executable oracle a verifier would replicate. A text-based "
            "verifier scoring candidate reasoning/answers is oracle-distinct by construction here."
        ),
        "genuinely_real_candidates": True,
        "genuinely_real_candidates_note": (
            "Candidates are real llama.cpp generations (temperature=0.8, distinct seed per sample) "
            "from a real cached GGUF model -- NOT deterministic templates or synthetic constructions. "
            "This directly addresses the failure mode found in ConstraintBench/exp5044 (candidates "
            "there were generator_kind='deterministic_solver_backed_variant', not real LLM output)."
        ),
        "model_id": "unsloth/gemma-4-12B-it-GGUF",
        "model_path": MODEL_PATH,
        "categories_sampled": sorted({q["category"] for q in questions}),
        "per_question_results": [
            {
                "question_id": r["question_id"],
                "category": r["category"],
                "gold": r["answer"],
                "parsed_letters": [c["parsed_letter"] for c in r["candidates"]],
                "n_parsed": r["n_parsed"],
            }
            for r in rows
        ],
        "load_duration_s": round(load_duration_s, 2),
        "generation_duration_s": round(gen_duration_s, 2),
        "gpu_device": GPU_DEVICE,
        "inference_substrate": "live_llm_inference",
        "model_specs": {"model": "unsloth/gemma-4-12B-it-GGUF", "quantization": "Q4_K_M"},
        "target_model": "unsloth/gemma-4-12B-it-GGUF",
        "random_seed": SEED,
    }
    total_duration_s = load_duration_s + gen_duration_s
    artifact["duration_s"] = round(total_duration_s, 2)

    if oracle_at_k <= sc_vote + 0.03:
        verdict = (
            f"complete_mmlu_pro_no_meaningful_headroom_oracle_{oracle_at_k:.3f}_sc_{sc_vote:.3f}_"
            f"headroom_{headroom:.3f}"
        )
    else:
        verdict = (
            f"complete_mmlu_pro_headroom_present_oracle_{oracle_at_k:.3f}_sc_{sc_vote:.3f}_"
            f"headroom_{headroom:.3f}_candidate_for_verifier_build"
        )
    artifact["honest_verdict"] = verdict

    checksum_payload = {k: v for k, v in artifact.items() if k not in ("duration_s",)}
    artifact["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(checksum_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    RESULT.write_text(json.dumps(artifact, indent=2))
    print(
        json.dumps(
            {
                k: artifact[k]
                for k in (
                    "n_questions_scored",
                    "n_questions_unparseable",
                    "oracle_at_k",
                    "sc_vote",
                    "headroom",
                    "honest_verdict",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
