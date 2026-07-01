"""Few-shot MMLU-Pro headroom check: does improving the generator raise SC-vote off the
random-chance floor before re-testing whether a verifier adds value? (outer-loop follow-up,
operator: "let's improve the generator first (few-shot prompting) to help SC-vote land
somewhere meaningful")

CONTEXT: exp_mmlu_pro_fresh_headroom_check.py and exp_mmlu_pro_verifier_vs_cheap_baseline.py both
used ZERO-SHOT prompting with gemma-4-12B-it, and got sc_vote=0.075 -- close to the 10-way
random-chance floor (0.10). The honest caveat filed there: a verifier "win" against a
near-random-guessing baseline would be confounded (beating a coin-flip isn't evidence of real
verifier value). This script uses the STANDARD MMLU-Pro 5-shot chain-of-thought protocol (the
paper's own evaluation format) to raise generator quality first, so any future verifier test is
against a competent baseline, not a confound.

Few-shot exemplars: TIGER-Lab/MMLU-Pro's `validation` split, 70 rows, exactly 5 per category, each
with real worked chain-of-thought reasoning (`cot_content`) ending "The answer is (X)." -- this IS
the dataset's own designated few-shot source, not a held-out leak (validation is disjoint from the
test split questions this script samples from). All 5 exemplars for the TARGET question's own
category are used (matching the standard protocol), so the model sees genuine worked examples in
the same subject domain before answering.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
POOL_PATH = REPO / "results" / "experiment_mmlu_pro_fewshot_candidate_pool.jsonl"
RESULT = REPO / "results" / "experiment_mmlu_pro_fewshot_headroom_check.json"
N_QUESTIONS = 40
K_SAMPLES = 6
TEMPERATURE = 0.8
SEED = 20260701
GPU_DEVICE = 1
SERVER_URL = "http://127.0.0.1:8712/v1/chat/completions"
MAX_TOKENS = 500  # a bit more than the zero-shot run's 400: few-shot CoT exemplars encourage longer reasoning

_ANSWER_RE = re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE)


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _letters(options: list[str]) -> list[str]:
    return [chr(ord("A") + i) for i in range(len(options))]


def _format_question_block(question: str, options: list[str]) -> str:
    letters = _letters(options)
    opts_block = "\n".join(f"({letter}) {opt}" for letter, opt in zip(letters, options))
    return f"{question}\n{opts_block}"


def build_fewshot_exemplar_bank(validation_ds) -> dict[str, list[dict]]:
    """Group the validation split's 70 rows (5/category) by category."""
    bank: dict[str, list[dict]] = {}
    for row in validation_ds:
        bank.setdefault(row["category"], []).append(row)
    return bank


def build_fewshot_prompt(
    question: str, options: list[str], category: str, exemplar_bank: dict[str, list[dict]]
) -> str:
    exemplars = exemplar_bank.get(category, [])
    parts = [
        "The following are multiple choice questions (with answers) about "
        f"{category}. Think step by step and then finish your answer with "
        "'the answer is (X)' where X is the correct letter choice.\n"
    ]
    for ex in exemplars:
        q_block = _format_question_block(ex["question"], ex["options"])
        # cot_content already starts with "A: " (verified 70/70 in the validation split) -- don't
        # double-prefix it.
        cot = ex["cot_content"].strip()
        if cot.startswith("A:"):
            cot = cot[2:].strip()
        parts.append(f"Q: {q_block}\nA: {cot}\n")
    target_block = _format_question_block(question, options)
    parts.append(f"Q: {target_block}\nA: Let's think step by step.")
    return "\n".join(parts)


def parse_letter(text: str, n_options: int) -> str | None:
    valid = {chr(ord("A") + i) for i in range(n_options)}
    matches = [m.group(1).upper() for m in _ANSWER_RE.finditer(text)]
    for letter in reversed(matches):
        if letter in valid:
            return letter
    # fallback: last standalone valid letter in the text
    for ch in reversed(text):
        if ch in valid:
            return ch
    return None


def call_server(prompt: str, seed: int) -> str:
    resp = requests.post(
        SERVER_URL,
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "seed": seed,
        },
        timeout=120,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    return (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")


def generate_pool(exemplar_bank: dict[str, list[dict]], questions: list[dict]) -> tuple[list[dict], float]:
    """Resumable: appends each question's K candidates to POOL_PATH as soon as they're
    generated -- matches exp_mmlu_pro_verifier_vs_cheap_baseline.py's checkpoint pattern (this
    environment's background-task lifecycle killed unattended servers mid-run multiple times)."""
    from collections import Counter as _Counter

    rows: list[dict] = []
    if POOL_PATH.exists():
        for line in POOL_PATH.open():
            rows.append(json.loads(line))
    counts = _Counter(r["question_index"] for r in rows)
    already_done_qi = {qi for qi, n in counts.items() if n >= K_SAMPLES}
    partial_qi = {qi for qi, n in counts.items() if 0 < n < K_SAMPLES}
    if partial_qi:
        _log(f"dropping partial rows for questions {sorted(partial_qi)}; will regenerate")
        rows = [r for r in rows if r["question_index"] not in partial_qi]
        POOL_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if already_done_qi:
        _log(f"resuming: {len(already_done_qi)} fully-generated questions already done")

    t0 = time.time()
    with POOL_PATH.open("a") as f:
        for qi, q in enumerate(questions):
            if qi in already_done_qi:
                continue
            prompt = build_fewshot_prompt(q["question"], q["options"], q["category"], exemplar_bank)
            for k in range(K_SAMPLES):
                text = call_server(prompt, seed=SEED + 1000 * qi + k)
                letter = parse_letter(text, len(q["options"]))
                row = {
                    "question_index": qi,
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "k": k,
                    "gold": q["answer"],
                    "parsed_letter": letter,
                    "correct": letter == q["answer"] if letter is not None else False,
                    "full_text": text,
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
            if (qi + 1) % 5 == 0:
                elapsed = time.time() - t0
                _log(f"  {qi + 1}/{len(questions)} questions done ({elapsed:.0f}s elapsed)")
    gen_duration_s = time.time() - t0
    _log(f"generation done: {gen_duration_s:.1f}s for {len(rows)} total candidates in pool")
    return rows, gen_duration_s


def main() -> int:
    from datasets import load_dataset

    _log(f"loading MMLU-Pro test split (sampling {N_QUESTIONS} questions, seed={SEED})")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.shuffle(seed=SEED).select(range(N_QUESTIONS))
    questions = [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "answer": row["answer"],
            "category": row["category"],
        }
        for row in ds
    ]

    _log("loading MMLU-Pro validation split for 5-shot CoT exemplars")
    val_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    exemplar_bank = build_fewshot_exemplar_bank(val_ds)
    exemplar_counts = {cat: len(rows) for cat, rows in exemplar_bank.items()}
    _log(f"exemplar bank: {exemplar_counts}")

    n_before = sum(1 for _ in POOL_PATH.open()) if POOL_PATH.exists() else 0
    rows, gen_duration_s = generate_pool(exemplar_bank, questions)
    pool_reused = n_before >= N_QUESTIONS * K_SAMPLES

    n_questions = len(set(r["question_index"] for r in rows))
    oracle_hits = sum(
        1
        for qi in set(r["question_index"] for r in rows)
        if any(r["correct"] for r in rows if r["question_index"] == qi)
    )
    oracle_at_k = oracle_hits / n_questions
    sc_hits = 0
    for qi in sorted(set(r["question_index"] for r in rows)):
        letters = [r["parsed_letter"] for r in rows if r["question_index"] == qi and r["parsed_letter"]]
        if not letters:
            continue
        vote = Counter(letters).most_common(1)[0][0]
        gold = next(r["gold"] for r in rows if r["question_index"] == qi)
        if vote == gold:
            sc_hits += 1
    sc_vote = sc_hits / n_questions
    n_correct_rows = sum(1 for r in rows if r["correct"])
    n_unparseable = sum(1 for r in rows if r["parsed_letter"] is None)

    artifact = {
        "experiment": "mmlu_pro_fewshot_headroom_check",
        "corpus": "TIGER-Lab/MMLU-Pro (same 40-question test-split sample as the zero-shot runs)",
        "prompting": "5-shot chain-of-thought, MMLU-Pro's own standard evaluation protocol -- 5 exemplars per category drawn from the validation split (disjoint from the sampled test questions), each with real worked cot_content ending 'the answer is (X)'.",
        "n_questions": n_questions,
        "n_candidates": len(rows),
        "n_correct_candidates": n_correct_rows,
        "n_unparseable_candidates": n_unparseable,
        "oracle_at_k": round(oracle_at_k, 4),
        "sc_vote": round(sc_vote, 4),
        "headroom": round(oracle_at_k - sc_vote, 4),
        "headroom_definition": "oracle_at_k - sc_vote = selectable headroom a verifier could capture",
        "comparison_to_zero_shot": (
            "Zero-shot (exp_mmlu_pro_fresh_headroom_check.py, 2nd run): oracle_at_k=0.350, "
            "sc_vote=0.075. This 5-shot CoT run is the SAME 40 questions, same model, same K, "
            "only the prompting changed."
        ),
        "generator_quality_verdict": (
            "sc_vote landed at a meaningful, non-floor level -- 5-shot CoT is a genuine generator "
            "improvement over zero-shot."
            if sc_vote > 0.20
            else (
                "sc_vote is still close to the random-chance floor even with 5-shot CoT -- the "
                "generator-weakness confound from the zero-shot run is NOT resolved by prompting "
                "alone; a stronger/larger model would be the next lever, not more prompt engineering."
            )
        ),
        "inference_substrate": "live_llm_inference",
        "model_specs": {"model": "unsloth/gemma-4-12B-it-GGUF", "quantization": "Q4_K_M"},
        "target_model": "unsloth/gemma-4-12B-it-GGUF",
        "gpu_device": GPU_DEVICE,
        "pool_reused": pool_reused,
        "generation_duration_s": round(gen_duration_s, 2),
        "random_seed": SEED,
        "k_samples_per_question": K_SAMPLES,
        "temperature": TEMPERATURE,
    }
    total_duration_s = gen_duration_s
    artifact["duration_s"] = round(total_duration_s, 2)

    if oracle_at_k <= sc_vote + 0.03:
        verdict_tag = "no_meaningful_headroom"
    else:
        verdict_tag = "headroom_present"
    artifact["honest_verdict"] = (
        f"complete_mmlu_pro_fewshot_{verdict_tag}_oracle_{oracle_at_k:.3f}_sc_{sc_vote:.3f}_"
        f"headroom_{oracle_at_k - sc_vote:.3f}"
    )

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
                    "n_questions",
                    "oracle_at_k",
                    "sc_vote",
                    "headroom",
                    "generator_quality_verdict",
                    "honest_verdict",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
