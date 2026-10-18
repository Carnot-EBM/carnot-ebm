"""Exp 4002 GAP-4 LOCAL open-weight generator arm — does a sovereign model drive the proven verifier?

WHY THIS, WHY NOW (the decentralization-clean headline path)
------------------------------------------------------------
The GAP-4 positive (results/arc3_gap4_rule_exec_verifier.json) proved that a program-induction +
execution-consistency VERIFIER beats TRM frequency vote on the 31-entry ARC-1 rerank pool (vote
pass@2 0.4516 -> gated 0.5806, oracle ceiling 0.6129, bootstrap CI above zero). But that positive
leans on a CLOSED-WEIGHT generator: gpt-5.5 via the codex CLI writes `def transform(grid)`. The
verifier side is already fully local and model-free (demo-fit + restricted-namespace execution +
content-hash candidate match), yet the GENERATOR violates the project's sovereignty story
(CLAUDE.md Decentralization Rule 1: "Local-first using open models, always"). The GAP-4 forward
protocol explicitly owes "a local open-weight generator arm (Gemma-4/Qwen3.6) for the
decentralization tier."

This experiment pays that debt. It swaps ONLY the inducer — a SOTA local GGUF
(Qwen3.6-35B-A3B / gemma-4-26B-A4B / gemma-4-12B) loaded via llama.cpp replaces the codex
subprocess — and re-runs the IDENTICAL verifier on the IDENTICAL pool. Everything downstream of
the generated `transform` function is byte-for-byte the same code, imported unchanged from
`arc3_gap4_rule_exec_verifier`. That isolation is the whole point: any change in the rerank result
is attributable to the GENERATOR, because the verifier is held fixed.

WHAT THE NUMBERS MEAN
---------------------
  * local_induction_demo_perfect_rate — the binding constraint. The verifier only ever overrides
    vote when the program reproduces every demo exactly; if the open model cannot induce a
    demo-perfect program, the verifier has nothing to act on. Codex hit 29/31 = 0.9355. The open
    model's rate is the open-vs-closed induction-capability datum.
  * local_gated_pass2 — does an OPEN model reach the moat? Compared head-to-head against vote
    0.4516, oracle 0.6129, and the codex tier 0.5806.
  * local_beats_vote — the sovereign accuracy claim: local gated pass@2 > vote pass@2 with a
    bootstrap 95% CI lower bound that strictly excludes 0.

Two outcomes are BOTH publishable, neither is a verifier failure:
  - success: a local model reaches the moat -> a decentralization-clean headline.
  - complete: a local model induces but lands below the codex tier -> the open-vs-closed induction
    gap quantified (a real, citable finding about generator capability, NOT about the verifier).

THE exp3975 LESSON (do NOT repeat): the .368 DSL-only build (results/experiment_3975_*) FAILED
because no real LLM proposer was invoked (`llm_proposer_used=false`, coverage 0.0). The fix is a
real, PRECONDITION-gated GGUF proposer. If the GGUF is not cached we emit
`blocked_local_gguf_not_cached` and STOP — we never silently degrade to DSL-only or to codex.

NO-ORACLE INVARIANT (inherited unchanged): the prompt contains demo pairs + the test INPUT only —
never the test gold, never the candidate pool. `correct` labels score rankings post-hoc.

  # plumbing smoke (2 entries):
  .venv/bin/python scripts/experiments/experiment_4002_gap4_local_generator_arm.py --limit 2
  # full run (gemma-4-26B-A4B-it inducer over the 31-entry pool, <=3 demo-feedback iters):
  .venv/bin/python scripts/experiments/experiment_4002_gap4_local_generator_arm.py --model gemma26

Inducer choice: gemma-4-26B-A4B-it (instruct MoE) is the default — it emits a clean
```python def transform``` block in ~8-11 s/call. Qwen3.6-35B-A3B is a *reasoning* model: at
greedy decoding it spends thousands of tokens thinking and frequently exhausts the token budget
before emitting code (measured ~111 s/call, no code block), so it is selectable but not the
default for this rerank venue.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# --- the GAP-4 VERIFIER PRIMITIVES, imported UNCHANGED (this is the "verifier_side_unchanged" claim).
#     Only the generator is new; everything below `transform` is the codex-tier code, byte-identical.
from arc3_gap3_stage2_transition_ebm import (  # noqa: E402
    POOL,
    SEED,
    _grouped_loto_union,
    _pass,
    ghash,
)
from arc3_gap4_rule_exec_verifier import (  # noqa: E402
    _extract_code,
    _failing_demos,
    build_rankers,
    demo_fit,
    induction_prompt,
    safe_transform_from_code,
)
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
REPO_ROOT = repo_root()
OUTPUT = REPO_ROOT / "results" / "experiment_4002_gap4_local_generator_arm.json"
CODEX_REF = REPO_ROOT / "results" / "arc3_gap4_rule_exec_verifier.json"

# The three SOTA local GGUFs sanctioned for the inducer arm. Keys are CLI-friendly; values are the
# unsloth HF repo ids resolved to a concrete .gguf path via resolve_cached_gguf (the GGUF tokenizer
# rule: load by .gguf path, never AutoTokenizer on the repo id).
LOCAL_MODELS: dict[str, dict[str, str]] = {
    "qwen35": {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "name": "Qwen3.6-35B-A3B"},
    "gemma26": {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "name": "gemma-4-26B-A4B"},
    "gemma12": {"hf_id": "unsloth/gemma-4-12B-it-GGUF", "name": "gemma-4-12B"},
}

INFERENCE_SUBSTRATE = "live_llm_inference"

# Reference numbers from the codex-tier positive (the comparison baselines for the headline).
VOTE_PASS2_REF = 0.4516
ORACLE_PASS2_REF = 0.6129
CODEX_GATED_PASS2_REF = 0.5806

REQUIRED_FIELDS = [
    "local_induction_demo_perfect_rate",
    "local_gated_pass2",
    "local_beats_vote",
    "local_model_used",
    "cost_local_seconds",
    "cost_codex_seconds_ref",
    "cost_verifier_seconds",
    "ci95_local_minus_vote",
    "verifier_side_unchanged",
    "missing_verifier_gaps",
    "preconditions_checked",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "local_induction_demo_perfect_rate": (
        "BARE FLOAT - fraction of pool entries the LOCAL GGUF induced a demo-perfect program for "
        "(the open-model induction capability; the binding constraint vs codex 0.9355)."
    ),
    "local_gated_pass2": (
        "BARE FLOAT - the LOCAL-inducer gated rerank pass@2 (vs vote 0.4516, oracle 0.6129, "
        "codex tier 0.5806 - does an OPEN model reach the moat)."
    ),
    "local_beats_vote": (
        "BARE BOOL - local gated rerank beats vote pass@2 with a bootstrap CI excluding 0 "
        "(the sovereign accuracy claim)."
    ),
    "local_model_used": (
        "Which SOTA local GGUF was the inducer - the decentralization provenance."
    ),
    "cost_local_seconds": (
        "Per-task wall-cost of the LOCAL inducer (sovereign, on-box; should beat codex network "
        "calls per the decentralized-efficiency datum)."
    ),
    "cost_codex_seconds_ref": (
        "Per-task wall-cost of the codex tier (reference, from the saved codex artifact)."
    ),
    "cost_verifier_seconds": (
        "Per-task wall-cost of the model-free verifier (rerank + scoring); the cheap local layer."
    ),
    "ci95_local_minus_vote": (
        "Bootstrap 95% CI of (local gated - vote) pass@2; non-overlap with 0 is significance."
    ),
    "verifier_side_unchanged": (
        "BARE BOOL - the model-free verifier primitives were reused unchanged (only the inducer "
        "was swapped; isolates the generator's contribution)."
    ),
    "missing_verifier_gaps": (
        "What the local inducer could NOT synthesize that codex could (the open-vs-closed "
        "induction gap, per the Missing-Verifier Gap Logging mandate)."
    ),
    "preconditions_checked": (
        "list of {resource, available} - the GGUF cache + llama_cpp checks (pre-empts the "
        "exp3975 silent-DSL-fallback failure)."
    ),
    "random_seed": "Shared GAP-4 substrate seed for reproducible bootstrap + greedy decoding.",
    "honest_verdict": "Terminal-prefix verdict (success/complete/blocked).",
    "duration_s": "Wall-clock seconds for this runner.",
    "inference_substrate": "live_llm_inference - the local GGUF is loaded and run for real.",
}


# --------------------------------------------------------------------------- preconditions
def resolve_local_gguf(hf_id: str) -> str | None:
    """Resolve a SOTA GGUF repo id to a concrete on-disk .gguf path, or None if it isn't cached.

    Thin wrapper around carnot.inference.sota_models.resolve_cached_gguf so the precondition check
    and the proposer agree on exactly one path. Returns None (not a fabricated path) when absent so
    the caller emits blocked_local_gguf_not_cached.
    """
    from carnot.inference.sota_models import resolve_cached_gguf

    return resolve_cached_gguf(hf_id)


def check_preconditions(
    hf_id: str,
    pool_path: Path | str,
    *,
    gguf_path_override: str | None = None,
    llama_available_override: bool | None = None,
) -> list[dict[str, Any]]:
    """Verify the three resources the LOCAL arm needs BEFORE any inference (Pre-Launch Preconditions
    Discipline). Overrides exist purely so unit tests can exercise every blocked branch without a
    22 GB model on disk."""
    if gguf_path_override is not None:
        gguf_ok = bool(gguf_path_override) and Path(gguf_path_override).exists()
    else:
        gguf_ok = resolve_local_gguf(hf_id) is not None

    if llama_available_override is not None:
        llama_ok = bool(llama_available_override)
    else:
        try:
            import llama_cpp  # noqa: F401

            llama_ok = True
        except Exception:
            llama_ok = False

    try:
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            json.load(handle)
        pool_ok = True
    except Exception:
        pool_ok = False

    return [
        {"resource": "local_gguf_cached", "available": gguf_ok},
        {"resource": "llama_cpp", "available": llama_ok},
        {"resource": "eval_pool", "available": pool_ok},
    ]


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    """Map the first failed precondition to its blocked_<resource> verdict. Order matters: the GGUF
    cache is checked first because it is the point of the experiment (never degrade to DSL/codex)."""
    by_resource = {row["resource"]: bool(row["available"]) for row in preconditions}
    if not by_resource.get("local_gguf_cached", False):
        return "blocked_local_gguf_not_cached"
    if not by_resource.get("llama_cpp", False):
        return "blocked_llama_cpp_unavailable"
    if not by_resource.get("eval_pool", False):
        return "blocked_eval_pool_unreadable"
    return None


# --------------------------------------------------------------------------- the LOCAL generator
class LocalGGUFProposer:
    """The ONLY new piece of generator machinery: a callable that, given an induction prompt, asks a
    local open-weight GGUF for one ```python def transform(grid)``` block and returns
    (raw_text, wall_seconds) — the exact contract of the codex `ask_codex` it replaces.

    The model is injected (not constructed) so tests can pass a fake llama object exposing
    `create_chat_completion`. The real factory `load_local_llama` builds the llama.cpp model with
    all GPU layers offloaded and greedy decoding for reproducibility.
    """

    SYSTEM = (
        "You are an expert Python programmer solving ARC (Abstraction and Reasoning Corpus) "
        "puzzles. You infer the single transformation rule from the demonstration pairs and "
        "implement it generically. Output ONLY one ```python code block containing "
        "def transform(grid)."
    )

    def __init__(self, llama: Any, max_tokens: int = 2048, temperature: float = 0.0):
        self._llama = llama
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, prompt: str) -> tuple[str, float]:
        t0 = time.time()
        try:
            out = self._llama.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            text = out["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # a model crash on one prompt must not sink the whole run
            text = f"__local_error__:{type(exc).__name__}"
        return text, round(time.time() - t0, 2)


def load_local_llama(
    gguf_path: str, n_ctx: int = 16384, seed: int = SEED
) -> Any:  # pragma: no cover
    """Build the real llama.cpp model. Excluded from unit coverage because it loads a multi-GB GGUF
    onto the GPU; the proposer contract around it is covered with a fake llama in the tests.

    n_gpu_layers=-1 offloads every layer; with two RTX 3090s llama.cpp splits a 35B model across
    both by default (LLAMA_SPLIT_MODE_LAYER), so the 22 GB Qwen + KV cache fits the 48 GB pool.
    """
    from llama_cpp import Llama

    return Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        seed=seed,
        verbose=False,
    )


# --------------------------------------------------------------------------- induction (generator)
def induce_program_local(
    task_name: str,
    demos: list[dict[str, Any]],
    test_input: Any,
    proposer: Callable[[str], tuple[str, float]],
    iters: int = 3,
) -> dict[str, Any]:
    """Mirror of arc3_gap4_rule_exec_verifier.induce_program with ONE substitution: the codex
    subprocess call is replaced by `proposer` (the local GGUF). Every other step — prompt assembly,
    code extraction, sandbox compilation, demo-fit grading, failure-feedback refinement — calls the
    UNCHANGED verifier primitives. Returns the same record shape so the downstream rerank is
    generator-agnostic.
    """
    best_fit, best_code, best_fn = -1.0, None, None
    history: list[dict[str, Any]] = []
    prior_code, failures = None, None
    for it in range(iters):
        raw, dt = proposer(induction_prompt(demos, test_input, prior_code, failures))
        code = _extract_code(raw)
        if code is None:
            history.append({"iter": it, "status": "no_code", "local_s": dt})
            continue
        fn = safe_transform_from_code(code)
        if fn is None:
            history.append({"iter": it, "status": "unsafe_or_uncompilable", "local_s": dt})
            continue
        fit = demo_fit(fn, demos)
        history.append(
            {
                "iter": it,
                "status": "graded",
                "demo_fit": round(fit, 4),
                "local_s": dt,
                "code_len": len(code),
            }
        )
        if fit > best_fit:
            best_fit, best_code, best_fn = fit, code, fn
        if best_fit >= 1.0:
            break
        prior_code = best_code
        failures = _failing_demos(best_fn, demos) if best_fn else None
    pred = best_fn(test_input) if (best_fn is not None and best_fit >= 1.0) else None
    return {
        "task": task_name,
        "demo_fit": round(max(best_fit, 0.0), 4),
        "demo_perfect": bool(best_fit >= 1.0),
        "pred_hash": ghash(pred) if pred is not None else None,
        "pred_grid": pred.tolist() if pred is not None else None,
        "n_calls": len(history),
        "local_seconds": round(sum(h["local_s"] for h in history), 2),
        "history": history,
        "code": best_code,
    }


def induce_pool(
    entries: list[dict[str, Any]],
    proposer: Callable[[str], tuple[str, float]],
    iters: int = 3,
) -> dict[int, dict[str, Any]]:
    """Induce once per UNIQUE task (a task with several test entries shares one rule), then
    re-execute the demo-perfect program on each extra entry's test input. Returns id(entry) -> record
    so the scorer can attach the right prediction to each pool entry. Mirrors the codex `_induce_for`.
    """
    by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_task.setdefault(entry["task"], []).append(entry)

    prog_by_entry: dict[int, dict[str, Any]] = {}
    for task_name in sorted(by_task):
        ents = by_task[task_name]
        rec = induce_program_local(
            task_name, ents[0]["demos"], ents[0]["test_input"], proposer, iters
        )
        prog_by_entry[id(ents[0])] = rec
        for extra in ents[1:]:
            fn = safe_transform_from_code(rec["code"]) if rec["code"] else None
            pred = fn(extra["test_input"]) if (fn is not None and rec["demo_perfect"]) else None
            prog_by_entry[id(extra)] = {
                **rec,
                "pred_hash": ghash(pred) if pred is not None else None,
                "pred_grid": pred.tolist() if pred is not None else None,
                "n_calls": 0,
                "local_seconds": 0.0,
            }
    return prog_by_entry


# --------------------------------------------------------------------------- verifier (UNCHANGED)
def _boot_ci(tasks: list[dict[str, Any]], key_a, key_b, n: int, seed: int) -> list[float]:
    """Percentile bootstrap 95% CI of pass@2(A) - pass@2(B), using the same LCG resampling as the
    codex run so the methodology is bit-identical (only the induced programs differ)."""

    def _lcg(s: int):
        x = s
        while True:
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            yield x

    def _p2(sample, key):
        return sum(
            int(any(c["correct"] for c in sorted(t["cands"], key=key)[:2])) for t in sample
        ) / len(sample)

    gen, deltas = _lcg(seed), []
    for _ in range(1000):
        samp = [tasks[next(gen) % n] for _ in range(n)]
        deltas.append(_p2(samp, key_a) - _p2(samp, key_b))
    deltas.sort()
    return [round(deltas[25], 4), round(deltas[974], 4)]


def score_pool(
    entries: list[dict[str, Any]],
    prog_by_entry_id: dict[int, dict[str, Any]],
    seed: int = SEED,
) -> dict[str, Any]:
    """Run the UNCHANGED GAP-4 rerank over the pool given the locally-induced programs. Every actual
    verifier computation is delegated to an imported primitive — build_rankers (the gate),
    _grouped_loto_union (the union baseline), _pass (pass@k), ghash (content match). This function is
    only data marshalling; it adds no new verifier logic, which is what lets verifier_side_unchanged
    be reported True."""
    tasks: list[dict[str, Any]] = []
    for e in entries:
        tot = sum(c["votes"] for c in e["candidates"])
        cands = [
            {
                "votes": c["votes"],
                "q_mean": c["q_mean"],
                "correct": c["correct"],
                "grid": c["grid"],
                "vote_share": c["votes"] / max(1, tot),
            }
            for c in e["candidates"]
        ]
        tasks.append({"task": e["task"], "cands": cands, "prog": prog_by_entry_id.get(id(e))})

    rankers = build_rankers(tasks)  # sets _exec_match / _exec_hamming (the gate features)
    _grouped_loto_union(
        tasks, lambda c: np.array([np.log1p(c["votes"]), c["vote_share"], c["q_mean"]])
    )
    for t in tasks:
        for c in t["cands"]:
            c["_union_noX"] = c["_u"]
    _grouped_loto_union(
        tasks,
        lambda c: np.array(
            [
                np.log1p(c["votes"]),
                c["vote_share"],
                c["q_mean"],
                c["_exec_match"],
                min(c["_exec_hamming"], 2.0),
            ]
        ),
    )
    for t in tasks:
        for c in t["cands"]:
            c["_union_withX"] = c["_u"]
    rankers["UNION_votes_qmean_voteshare"] = lambda c: (c["_union_noX"], -c["votes"])
    rankers["UNION_plus_exec"] = lambda c: (c["_union_withX"], -c["votes"])

    res = {name: _pass(tasks, key) for name, key in rankers.items()}
    n = len(tasks)
    n_oracle = sum(1 for t in tasks if any(c["correct"] for c in t["cands"]))
    oracle2 = round(n_oracle / n, 4)

    kv, kg = rankers["TRM_VOTE"], rankers["GAP4_GATED"]
    vote_hits = {
        i for i, t in enumerate(tasks) if any(c["correct"] for c in sorted(t["cands"], key=kv)[:2])
    }
    gated_hits = {
        i for i, t in enumerate(tasks) if any(c["correct"] for c in sorted(t["cands"], key=kg)[:2])
    }
    vote_wins_lost = sorted(vote_hits - gated_hits)
    headroom_recovered = sorted(
        i for i in gated_hits - vote_hits if any(c["correct"] for c in tasks[i]["cands"])
    )
    vote2 = res["TRM_VOTE"]["pass@2"]
    g2 = res["GAP4_GATED"]["pass@2"]
    n_perfect = sum(1 for t in tasks if t["prog"] and t["prog"]["demo_perfect"])

    per_task = []
    for i, t in enumerate(tasks):
        prog = t["prog"]
        per_task.append(
            {
                "i": i,
                "task": t["task"],
                "n_cands": len(t["cands"]),
                "oracle_hit": any(c["correct"] for c in t["cands"]),
                "vote_top2": i in vote_hits,
                "gated_top2": i in gated_hits,
                "demo_fit": prog["demo_fit"] if prog else None,
                "demo_perfect": bool(prog and prog["demo_perfect"]),
                "pred_in_pool": bool(t["_gate_hit_pool"]),
            }
        )

    return {
        "rankers": res,
        "n": n,
        "n_oracle": n_oracle,
        "oracle2": oracle2,
        "vote2": vote2,
        "g2": g2,
        "n_perfect": n_perfect,
        "vote_wins_lost": [tasks[i]["task"] for i in vote_wins_lost],
        "headroom_recovered": [tasks[i]["task"] for i in headroom_recovered],
        "ci95_gated_vs_vote": _boot_ci(tasks, kg, kv, n, seed),
        "gates": {
            "selection_beats_vote": bool(g2 > vote2),
            "selection_beats_union": bool(g2 > res["UNION_votes_qmean_voteshare"]["pass@2"]),
            "vote_wins_lost": len(vote_wins_lost),
            "headroom_recovered": len(headroom_recovered),
            "coverage_demo_perfect": round(n_perfect / n, 4),
        },
        "per_task": per_task,
    }


# --------------------------------------------------------------------------- artifact assembly
def codex_reference_cost(codex_ref_path: Path = CODEX_REF) -> float:
    """Per-task codex wall-seconds from the saved codex-tier artifact (the cost reference). Falls
    back to the published 1387.2 s / 30 unique-task number if the artifact is unreadable."""
    try:
        ref = json.loads(codex_ref_path.read_text(encoding="utf-8"))
        total = float(ref["generator"]["total_codex_seconds"])
        n_unique = int(ref.get("n_unique_tasks") or 30)
        return round(total / max(1, n_unique), 2)
    except Exception:
        return round(1387.2 / 30, 2)


def compute_missing_verifier_gaps(
    entries: list[dict[str, Any]],
    prog_by_entry_id: dict[int, dict[str, Any]],
    codex_ref_path: Path = CODEX_REF,
) -> str:
    """The open-vs-closed induction gap: tasks codex induced demo-perfect that the local model did
    not. This is the verifier-gap datum (CLAUDE.md Missing-Verifier Gap Logging) — every such task
    is a candidate for a stronger local inducer or a richer prompt, not a verifier defect."""
    local_perfect = {
        prog_by_entry_id[id(e)]["task"]
        for e in entries
        if prog_by_entry_id.get(id(e)) and prog_by_entry_id[id(e)]["demo_perfect"]
    }
    try:
        ref = json.loads(codex_ref_path.read_text(encoding="utf-8"))
        codex_perfect = {row["task"] for row in ref.get("per_task", []) if row.get("demo_perfect")}
    except Exception:
        codex_perfect = set()
    gap_tasks = sorted(codex_perfect - local_perfect)
    if not gap_tasks:
        if codex_perfect:
            return (
                "No induction gap vs codex on demo-perfect coverage: the local inducer matched or "
                "exceeded codex's demo-perfect set on this pool."
            )
        return "Codex reference unavailable; local-only demo-perfect coverage recorded in per_task."
    return (
        "Local inducer could NOT synthesize a demo-perfect program for "
        + str(len(gap_tasks))
        + " task(s) that codex could: "
        + ", ".join(gap_tasks)
        + ". These are open-vs-closed induction-capability gaps (stronger local model / richer "
        "few-shot prompt), not verifier defects."
    )


def _fmt(value: float) -> str:
    """Compact float for the verdict string (0.5806 -> '0.5806', 0.5 -> '0.5')."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _verdict(local_beats_vote: bool, demo_perfect_rate: float, g2: float, model_short: str) -> str:
    if local_beats_vote:
        return f"success: gap4_local_generator_beats_vote_pass2{_fmt(g2)}_inducer{model_short}"
    return (
        "complete: gap4_local_induction"
        + _fmt(demo_perfect_rate)
        + "_pass2"
        + _fmt(g2)
        + "_below_codex"
    )


def _is_bare_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Schema gate (mirrors exp4000): every required field present, bare scalar types, terminal
    verdict prefix. Called on both the blocked and complete artifacts so a malformed artifact can
    never reach disk."""
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in ("local_beats_vote", "verifier_side_unchanged"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in (
        "local_induction_demo_perfect_rate",
        "local_gated_pass2",
        "cost_local_seconds",
        "cost_codex_seconds_ref",
        "cost_verifier_seconds",
        "duration_s",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    if not (
        isinstance(artifact["random_seed"], int) and not isinstance(artifact["random_seed"], bool)
    ):
        raise ValueError("random_seed must be a bare int")
    if not (
        isinstance(artifact["ci95_local_minus_vote"], list)
        and len(artifact["ci95_local_minus_vote"]) == 2
    ):
        raise ValueError("ci95_local_minus_vote must be a 2-element list")
    for field in ("local_model_used", "missing_verifier_gaps", "inference_substrate"):
        if not isinstance(artifact[field], str):
            raise ValueError(f"{field} must be a string")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")


def blocked_artifact(
    verdict: str,
    model_name: str,
    preconditions: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4002_gap4_local_generator_arm",
        "schema": "carnot.experiment_4002_gap4_local_generator_arm.v1",
        "title": "GAP-4 local open-weight generator arm",
        "local_induction_demo_perfect_rate": 0.0,
        "local_gated_pass2": 0.0,
        "local_beats_vote": False,
        "local_model_used": model_name,
        "cost_local_seconds": 0.0,
        "cost_codex_seconds_ref": codex_reference_cost(),
        "cost_verifier_seconds": 0.0,
        "ci95_local_minus_vote": [0.0, 0.0],
        "verifier_side_unchanged": True,
        "missing_verifier_gaps": (
            "Run blocked before induction; no local-vs-codex induction comparison produced."
        ),
        "preconditions_checked": preconditions,
        "random_seed": SEED,
        "honest_verdict": verdict,
        "duration_s": round(duration_s, 2),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def build_complete_artifact(
    entries: list[dict[str, Any]],
    prog_by_entry_id: dict[int, dict[str, Any]],
    scored: dict[str, Any],
    model_name: str,
    model_path: str,
    preconditions: list[dict[str, Any]],
    verifier_seconds: float,
    started_s: float,
    now_s: float,
    codex_ref_path: Path = CODEX_REF,
) -> dict[str, Any]:
    n = scored["n"]
    n_unique = len({e["task"] for e in entries})
    demo_perfect_rate = round(scored["n_perfect"] / n, 4)
    g2 = scored["g2"]
    ci = scored["ci95_gated_vs_vote"]
    local_beats_vote = bool(g2 > scored["vote2"] and ci[0] > 0.0)

    total_local_s = sum(
        prog_by_entry_id[id(e)]["local_seconds"] for e in entries if prog_by_entry_id.get(id(e))
    )
    cost_local = round(total_local_s / max(1, n_unique), 2)
    total_calls = sum(
        prog_by_entry_id[id(e)]["n_calls"] for e in entries if prog_by_entry_id.get(id(e))
    )

    # reproducibility checksum over the induced programs (content-addresses the generator output so a
    # third party can confirm the rerank re-derives from these exact programs).
    progs_blob = json.dumps(
        sorted(
            (rec["task"], rec["code"] or "")
            for rec in {
                id(e): prog_by_entry_id[id(e)] for e in entries if prog_by_entry_id.get(id(e))
            }.values()
        ),
        sort_keys=True,
    )
    repro = hashlib.sha256(progs_blob.encode()).hexdigest()[:16]

    verdict = _verdict(local_beats_vote, demo_perfect_rate, g2, model_name.replace("/", "_"))
    artifact = {
        "experiment": "experiment_4002_gap4_local_generator_arm",
        "schema": "carnot.experiment_4002_gap4_local_generator_arm.v1",
        "title": "GAP-4 local open-weight generator arm vs the model-free verifier",
        "local_induction_demo_perfect_rate": demo_perfect_rate,
        "local_gated_pass2": g2,
        "local_beats_vote": local_beats_vote,
        "local_model_used": model_name,
        "cost_local_seconds": cost_local,
        "cost_codex_seconds_ref": codex_reference_cost(codex_ref_path),
        "cost_verifier_seconds": round(verifier_seconds / max(1, n), 4),
        "ci95_local_minus_vote": ci,
        "verifier_side_unchanged": True,
        "missing_verifier_gaps": compute_missing_verifier_gaps(
            entries, prog_by_entry_id, codex_ref_path
        ),
        "preconditions_checked": preconditions,
        "random_seed": SEED,
        "honest_verdict": verdict,
        "duration_s": round(now_s - started_s, 2),
        "inference_substrate": INFERENCE_SUBSTRATE,
        # --- context (not required, but the comparison + provenance + audit trail) ---
        "n_entries": n,
        "n_unique_tasks": n_unique,
        "n_local_demo_perfect": scored["n_perfect"],
        "total_local_calls": total_calls,
        "total_local_seconds": round(total_local_s, 2),
        "references": {
            "vote_pass2": VOTE_PASS2_REF,
            "oracle_pass2": ORACLE_PASS2_REF,
            "codex_gated_pass2": CODEX_GATED_PASS2_REF,
        },
        "rankers": scored["rankers"],
        "gates": scored["gates"],
        "headroom_recovered_tasks": scored["headroom_recovered"],
        "vote_wins_lost_tasks": scored["vote_wins_lost"],
        "per_task": scored["per_task"],
        "model_specs": {
            "generator_model": model_name,
            "generator_gguf_path": model_path,
            "verifier": (
                "model-free: demo-fit exact-reproduction gate + restricted-namespace execution + "
                "content-hash candidate match (imported unchanged from arc3_gap4_rule_exec_verifier)"
            ),
        },
        "reproducibility_checksum": repro,
        "no_gpu_used": False,
        "decentralization_note": (
            "Inducer is a LOCAL open-weight GGUF (" + model_name + ") loaded via llama.cpp; the "
            "verifier side is fully local and model-free. This is the decentralization-clean "
            "deployment-tier path (CLAUDE.md Decentralization Rule 1) — no closed-weight call in "
            "the headline arm."
        ),
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- driver
def run(
    model_key: str = "gemma26",
    pool_path: Path | str = POOL,
    output_path: Path = OUTPUT,
    codex_ref_path: Path = CODEX_REF,
    iters: int = 3,
    limit: int = 0,
    n_ctx: int = 16384,
    proposer: Callable[[str], tuple[str, float]] | None = None,
    gguf_path_override: str | None = None,
    llama_available_override: bool | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Swap the generator, hold the verifier fixed, score the pool. `proposer` is injectable so the
    unit tests drive the whole pipeline with a fake model; production passes None and the real
    GGUF-backed LocalGGUFProposer is built after the preconditions pass."""
    started = time.time()
    spec = LOCAL_MODELS.get(model_key, LOCAL_MODELS["qwen35"])
    model_name = spec["name"]

    preconditions = check_preconditions(
        spec["hf_id"],
        pool_path,
        gguf_path_override=gguf_path_override,
        llama_available_override=llama_available_override,
    )
    blocker = blocker_from_preconditions(preconditions)
    if blocker:
        artifact = blocked_artifact(blocker, model_name, preconditions, time.time() - started)
        if write:
            _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
        pool = json.load(handle)
    entries = pool["entries"]
    if limit:
        entries = entries[:limit]

    model_path = gguf_path_override or resolve_local_gguf(spec["hf_id"]) or ""
    if proposer is None:  # pragma: no cover - real model path exercised by the script command
        proposer = LocalGGUFProposer(load_local_llama(model_path, n_ctx=n_ctx))

    print(
        f"[exp4002] LOCAL inducer={model_name} over {len(entries)} entries "
        f"({len({e['task'] for e in entries})} unique tasks, iters<={iters})",
        flush=True,
    )
    prog_by_entry_id = induce_pool(entries, proposer, iters)

    verifier_t0 = time.time()
    scored = score_pool(entries, prog_by_entry_id, seed=SEED)
    verifier_seconds = time.time() - verifier_t0

    artifact = build_complete_artifact(
        entries=entries,
        prog_by_entry_id=prog_by_entry_id,
        scored=scored,
        model_name=model_name,
        model_path=model_path,
        preconditions=preconditions,
        verifier_seconds=verifier_seconds,
        started_s=started,
        now_s=time.time(),
        codex_ref_path=codex_ref_path,
    )
    if write:
        _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        f"   demo_perfect={artifact['local_induction_demo_perfect_rate']} "
        f"gated_pass2={artifact['local_gated_pass2']} vote={VOTE_PASS2_REF} "
        f"oracle={ORACLE_PASS2_REF} codex={CODEX_GATED_PASS2_REF} "
        f"beats_vote={artifact['local_beats_vote']} CI={artifact['ci95_local_minus_vote']}",
        flush=True,
    )
    print(
        f"   cost/task local={artifact['cost_local_seconds']}s codex_ref="
        f"{artifact['cost_codex_seconds_ref']}s verifier={artifact['cost_verifier_seconds']}s",
        flush=True,
    )
    return artifact


def main() -> None:  # pragma: no cover - exercised by the required script command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(LOCAL_MODELS), default="gemma26")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="cap entries for a plumbing smoke")
    parser.add_argument("--n-ctx", type=int, default=16384)
    args = parser.parse_args()
    run(model_key=args.model, iters=args.iters, limit=args.limit, n_ctx=args.n_ctx)


if __name__ == "__main__":  # pragma: no cover
    main()
