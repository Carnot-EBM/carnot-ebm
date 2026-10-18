"""GAP-4: rule-execution verifier — induce the task's rule as a PROGRAM from the demo pairs, verify
it reproduces the demos (oracle-free), execute it on the test input, and gate the TRM rerank on
executed-rule consistency.

WHY THIS, WHY NOW (the GAP-3 closure, ops/verifier_gaps.md): four adversarially-confirmed stages
showed that scalar (q_halt), latent (z_H), and trained-content-energy selectors all sit at chance on
the dominant TRM error class — same-shape plausible-but-wrong rule applications (59.1% of real errors).
The v2 panel's verdict: "synthesizing the missing negative class IS program synthesis" — the only
signal that can distinguish 'right transformation' from 'coherent wrong transformation' is actually
INDUCING the transformation and EXECUTING it. This is the ARC-AGI-3 M2-v3/v4 stack (codex program
synthesis + consistency verification) transplanted to the ARC-AGI-1 rerank venue.

THE DIVISION OF LABOR (Carnot thesis): codex is the GENERATOR (writes `def transform(grid)` from the
demo pairs); the VERIFIER is mechanical and oracle-free — (a) demo-fit: the program must reproduce
every demo output exactly (demos are public context, never the test gold); (b) execution: the
demo-perfect program's output on the test input is the rule-consistency anchor for ranking.

THE GATED RERANK (vote-primary, provably-bounded downside): vote order is overridden ONLY when
  (1) the induced program reproduces ALL demos exactly (demo_fit == 1.0), AND
  (2) its executed test-output EXACTLY matches one of TRM's candidates (content hash).
In that case the matched candidate is promoted to rank 1 and everything else keeps vote order;
otherwise the ranker IS vote (abstain == no-op). The only way to lose a vote-win is a demo-perfect
program whose test execution is wrong yet coincides with a non-gold candidate — measured and reported
as `vote_wins_lost` (the safety count, expected 0).

NO-ORACLE INVARIANT: the codex prompt contains the demo pairs + the test INPUT only — never the test
gold, never the candidate pool. `correct` labels are used exclusively to score rankings post-hoc.
Candidates are never shown to the generator, so the program cannot pattern-match the pool.

DECENTRALIZATION NOTE (CLAUDE.md rules 1/2): codex (gpt-5.5) is the escalation-tier generator per the
M2-v3 precedent; data_handling_class=minimize (prompt = public ARC demo grids + test input, no user
data). A local open-weight generator (Gemma-4/Qwen3.6 via llama.cpp) is the deployment-tier path;
this experiment measures the CONCEPT's headroom with the strongest available inducer first.

  # plumbing smoke (2 tasks, ~2-4 codex calls):
  ~/trm_venv/bin/python scripts/experiments/arc3_gap4_rule_exec_verifier.py --limit 2
  # full run (30 unique tasks, <=3 iters each, 4 parallel codex workers):
  ~/trm_venv/bin/python scripts/experiments/arc3_gap4_rule_exec_verifier.py --workers 4
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
CARNOT = str(repo_root())
sys.path.insert(0, f"{CARNOT}/scripts/experiments")

from arc3_gap3_stage2_transition_ebm import (  # noqa: E402  (shared substrate)
    POOL,
    SEED,
    _grouped_loto_union,
    _pass,
    ghash,
)

ARTIFACT = f"{CARNOT}/results/arc3_gap4_rule_exec_verifier.json"
PROGRAMS = f"{CARNOT}/results/arc3_gap4_induced_programs.json"

CODEX = [
    "codex",
    "exec",
    "--color",
    "never",
    "--model",
    "gpt-5.5",
    "-c",
    "model_reasoning_effort=medium",
    "--dangerously-bypass-approvals-and-sandbox",
    "--cd",
    "/tmp",
    "--ephemeral",
]
_FORBIDDEN = (
    "__import__",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    "subprocess",
    "os.",
    "sys.",
    "import os",
    "import sys",
    "import subprocess",
    "socket",
    "shutil",
    "Path(",
    "getattr(",
    "setattr(",
    "globals(",
    "locals(",
    # 2026-06-10 hardening per the GAP-4 adversarial panel (corrigendum_2026_06_10_gap4):
    "type(",
    "np.load",
    "np.save",
    "np.fromfile",
    "np.memmap",
    "np.DataSource",
    "numpy.load",
    "numpy.save",
    ".tofile",
)
EXEC_TIMEOUT_S = (
    5.0  # per-call wall-clock cap (panel hardening); ARC transforms run in microseconds
)
_SAFE_BUILTIN_NAMES = [
    "range",
    "len",
    "min",
    "max",
    "abs",
    "enumerate",
    "zip",
    "sum",
    "sorted",
    "list",
    "dict",
    "set",
    "tuple",
    "int",
    "float",
    "bool",
    "map",
    "filter",
    "reversed",
    "any",
    "all",
    "round",
    "isinstance",
    "str",
]


def _numpy_only_import(name, *args, **kwargs):
    """numpy lazily __import__s its own submodules inside ops like np.unique/np.where — a bare
    builtins dict makes EVERY such program crash (KeyError: '__import__'; the smoke-run bug that
    zeroed demo_fit). Allow numpy's internal imports ONLY; codex code-level imports stay blocked
    (import statements are stripped and the __import__ token is in _FORBIDDEN)."""
    if name == "numpy" or name.startswith("numpy."):
        return __import__(name, *args, **kwargs)
    raise ImportError(f"import of {name!r} is blocked in the GAP-4 sandbox")


def _safe_builtins():
    import builtins as _b

    out = {k: getattr(_b, k) for k in _SAFE_BUILTIN_NAMES if hasattr(_b, k)}
    out["__import__"] = _numpy_only_import
    return out


def safe_transform_from_code(code: str):
    """Compile codex's transform() in a restricted namespace. Unlike the M2-v3 wrapper, the output
    SHAPE MAY DIFFER from the input (ARC rules crop/tile/grow); a crash or illegal output -> None
    (treated as abstention, never trusted)."""
    # Word-boundary matching (2026-06-10 fix, ARC-2 round): bare substring matching false-rejected
    # legitimate numpy code — 'type(' matched 'astype(' and 'os.' matched 'pos.' (5/5 blacklist hits
    # on the ARC-2 transcripts were false positives, costing one demo-perfect program). The negative
    # lookbehind requires the token NOT to be preceded by an identifier character.
    for tok in _FORBIDDEN:
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(tok), code):
            return None
    body = "\n".join(
        ln for ln in code.splitlines() if not ln.strip().startswith(("import ", "from "))
    )
    ns = {"np": np, "numpy": np, "__builtins__": _safe_builtins()}
    try:
        exec(body, ns)  # defines transform; body not executed yet
    except Exception:
        return None
    fn = ns.get("transform")
    if not callable(fn):
        return None

    def _call(grid):
        return np.asarray(fn(np.asarray(grid, dtype=np.int64).copy()), dtype=np.int64)

    def wrapped(grid):
        # Timeout per the panel hardening: a runaway loop in an induced program must abandon the
        # call, not hang the run. Thread-abandonment is a CONTAINMENT measure, not isolation (the
        # stuck thread leaks until process exit) — full subprocess isolation is the 400-task-run
        # protocol; legit ARC transforms finish in microseconds so 5s is generous.
        from concurrent.futures import ThreadPoolExecutor as _TPE
        from concurrent.futures import TimeoutError as _Timeout

        _ex = _TPE(max_workers=1)
        try:
            out = _ex.submit(_call, grid).result(timeout=EXEC_TIMEOUT_S)
            if out.ndim != 2 or out.size == 0 or out.shape[0] > 30 or out.shape[1] > 30:
                return None
            if out.min() < 0 or out.max() > 9:
                return None
            return out
        except _Timeout:
            # NOTE: shutdown(wait=False) — a `with` block would join the stuck thread and defeat
            # the timeout entirely.
            return None  # runaway program -> abandon -> abstain
        except Exception:
            return None  # crash -> abstain; the gated ranker falls back to vote
        finally:
            _ex.shutdown(wait=False)

    return wrapped


def _fmt_grid(g):
    return "[" + ",\n ".join(str(list(map(int, row))) for row in np.asarray(g)) + "]"


def induction_prompt(demos, test_input, prior_code=None, failures=None):
    lines = [
        "You are solving an ARC (Abstraction and Reasoning Corpus) puzzle. The demonstration pairs "
        "below all share ONE transformation rule.",
        "",
    ]
    for i, p in enumerate(demos):
        a, b = np.asarray(p["input"]), np.asarray(p["output"])
        lines.append(f"Demo {i + 1} INPUT ({a.shape[0]}x{a.shape[1]}):\n{_fmt_grid(a)}")
        lines.append(f"Demo {i + 1} OUTPUT ({b.shape[0]}x{b.shape[1]}):\n{_fmt_grid(b)}\n")
    t = np.asarray(test_input)
    lines.append(f"TEST INPUT ({t.shape[0]}x{t.shape[1]}):\n{_fmt_grid(t)}\n")
    lines.append(
        "Write exactly one Python function:\n"
        "    def transform(grid):\n"
        "        # grid: 2D numpy int array (colors 0-9). Return the transformed grid as a numpy\n"
        "        # int array. THE OUTPUT SHAPE MAY DIFFER from the input shape.\n"
        "Infer the rule from the demos and implement it GENERICALLY (np is provided; no imports, no "
        "file/network access). Your function MUST reproduce every demo output exactly from its demo "
        "input, and must then generalize to the test input by applying the SAME rule. Do NOT hardcode "
        "demo-specific coordinates/colors or anything specific to the test input. Output ONLY one "
        "```python code block."
    )
    base = "\n".join(lines)
    if prior_code and failures:
        base += (
            "\n\nYour PREVIOUS function failed these demos:\n"
            + failures
            + "\n\nPrevious code:\n```python\n"
            + prior_code
            + "\n```\nFix the rule inference and output the corrected function as one ```python block."
        )
    return base


def _extract_code(text: str):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    for b in reversed(blocks):
        if "def transform" in b:
            return b.strip()
    return None


def ask_codex(prompt, timeout=300, transcript_path=None):
    """transcript_path (panel hardening): archive the FULL raw codex stdout per call — not just the
    extracted code — so the no-oracle invariant is auditable post-hoc (grep transcripts for any
    file-access attempt / solution lookup)."""
    t0 = time.time()
    try:
        r = subprocess.run(CODEX, input=prompt, capture_output=True, text=True, timeout=timeout)
        out = r.stdout or ""
    except Exception as e:
        out = f"__codex_error__:{type(e).__name__}"
    if transcript_path is not None:
        Path(transcript_path).write_text(
            "===== PROMPT =====\n" + prompt + "\n===== RAW OUTPUT =====\n" + out
        )
    return out, round(time.time() - t0, 1)


def demo_fit(fn, demos):
    """Fraction of demo pairs the program reproduces EXACTLY. The oracle-free verification: demos are
    public context. 1.0 is the gate for any vote override."""
    hits = 0
    for p in demos:
        out = fn(p["input"])
        if out is not None and np.array_equal(out, np.asarray(p["output"])):
            hits += 1
    return hits / max(1, len(demos))


def _failing_demos(fn, demos, k=2, cap=20):
    out = []
    for i, p in enumerate(demos):
        if len(out) >= k:
            break
        got = fn(p["input"])
        if got is None or not np.array_equal(got, np.asarray(p["output"])):
            exp = np.asarray(p["output"])
            got_s = "(crash / illegal output)" if got is None else _fmt_grid(got[:cap, :cap])
            out.append(
                f"Demo {i + 1}: expected OUTPUT ({exp.shape[0]}x{exp.shape[1]}):\n"
                f"{_fmt_grid(exp[:cap, :cap])}\nyour function returned:\n{got_s}"
            )
    return "\n\n".join(out)


def induce_program(task_name, demos, test_input, iters=3, timeout=300, transcripts_dir=None):
    """Codex induction with refactor-from-best. Returns a record with the best program, its demo_fit,
    the executed test prediction (hash + grid), and the call history."""
    best_fit, best_code, best_fn = -1.0, None, None
    history = []
    prior_code, failures = None, None
    for it in range(iters):
        tp = f"{transcripts_dir}/{task_name}_iter{it}.txt" if transcripts_dir else None
        raw, dt = ask_codex(induction_prompt(demos, test_input, prior_code, failures), timeout, tp)
        code = _extract_code(raw)
        if code is None:
            history.append({"iter": it, "status": "no_code", "codex_s": dt})
            continue
        fn = safe_transform_from_code(code)
        if fn is None:
            history.append({"iter": it, "status": "unsafe_or_uncompilable", "codex_s": dt})
            continue
        fit = demo_fit(fn, demos)
        history.append(
            {
                "iter": it,
                "status": "graded",
                "demo_fit": round(fit, 4),
                "codex_s": dt,
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
        "codex_seconds": round(sum(h["codex_s"] for h in history), 1),
        "history": history,
        "code": best_code,
    }


# ----------------------------------------------------------------------------------------- rerank
def norm_hamming(cand, pred):
    """Graded execution-consistency energy: 0 = exact match; same-shape -> fraction of differing
    cells; shape mismatch -> 1 + relative size difference (always worse than any same-shape)."""
    c, p = np.asarray(cand), np.asarray(pred)
    if c.shape == p.shape:
        return float((c != p).mean())
    return 1.0 + abs(c.size - p.size) / max(c.size, p.size)


def build_rankers(tasks):
    """Attach per-candidate exec features + return the ranker dict. tasks[i]['prog'] holds the
    induction record for that entry's task (or None)."""
    for t in tasks:
        prog = t["prog"]
        gated_hash = prog["pred_hash"] if (prog and prog["demo_perfect"]) else None
        pred = np.asarray(prog["pred_grid"]) if (prog and prog["pred_grid"] is not None) else None
        for c in t["cands"]:
            c["_exec_match"] = (
                1.0 if (gated_hash is not None and ghash(c["grid"]) == gated_hash) else 0.0
            )
            c["_exec_hamming"] = norm_hamming(c["grid"], pred) if pred is not None else 2.0
        t["_has_gate"] = gated_hash is not None
        t["_gate_hit_pool"] = any(c["_exec_match"] for c in t["cands"])
    return {
        "TRM_VOTE": lambda c: (-c["votes"],),
        "GAP4_GATED": lambda c: (-c["_exec_match"], -c["votes"]),
        "GAP4_GRADED": lambda c: (c["_exec_hamming"], -c["votes"]),
        "EXEC_PURE": lambda c: (-c["_exec_match"], c["_exec_hamming"]),
    }


def run(
    limit=0,
    iters=3,
    workers=4,
    timeout=300,
    write=True,
    pool_path=POOL,
    artifact_path=ARTIFACT,
    programs_path=PROGRAMS,
    transcripts_dir=None,
    experiment_name="arc3_gap4_rule_exec_verifier",
):
    started = time.time()
    pre = {
        "codex_cli": subprocess.run(
            ["bash", "-lc", "command -v codex"], capture_output=True, text=True
        ).returncode
        == 0,
        "eval_pool": Path(pool_path).exists(),
    }
    if not all(pre.values()):
        art = {
            "experiment": experiment_name,
            "honest_verdict": "blocked_" + "_".join(k for k, v in pre.items() if not v),
            "preconditions_checked": [{"resource": k, "available": v} for k, v in pre.items()],
        }
        if write:
            Path(artifact_path).write_text(json.dumps(art, indent=2) + "\n")
        print(f"-> {art['honest_verdict']}")
        return art

    with gzip.open(pool_path, "rt") as f:
        pool = json.load(f)
    entries = pool["entries"]
    if limit:
        entries = entries[:limit]

    # one induction per UNIQUE task (f3e62deb has two test entries -> same rule, separate executions)
    by_task = {}
    for e in entries:
        by_task.setdefault(e["task"], []).append(e)

    def _induce_for(task_name):
        ents = by_task[task_name]
        # induce once from demos; execute per entry test_input
        rec = induce_program(
            task_name, ents[0]["demos"], ents[0]["test_input"], iters, timeout, transcripts_dir
        )
        recs = [rec]
        for extra in ents[1:]:  # extra test entries of the same task: reuse the program, re-execute
            fn = safe_transform_from_code(rec["code"]) if rec["code"] else None
            pred = fn(extra["test_input"]) if (fn is not None and rec["demo_perfect"]) else None
            recs.append(
                {
                    **rec,
                    "pred_hash": ghash(pred) if pred is not None else None,
                    "pred_grid": pred.tolist() if pred is not None else None,
                    "n_calls": 0,
                    "codex_seconds": 0.0,
                }
            )
        return task_name, recs

    if transcripts_dir:
        Path(transcripts_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"[gap4] inducing programs for {len(by_task)} unique tasks "
        f"({len(entries)} entries, iters<={iters}, workers={workers})",
        flush=True,
    )
    prog_by_entry = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for task_name, recs in ex.map(_induce_for, sorted(by_task)):
            for ent, rec in zip(by_task[task_name], recs):
                prog_by_entry[id(ent)] = rec
            fit = recs[0]["demo_fit"]
            print(
                f"  {task_name}: demo_fit={fit} perfect={recs[0]['demo_perfect']} "
                f"calls={recs[0]['n_calls']} ({recs[0]['codex_seconds']}s)",
                flush=True,
            )

    tasks = []
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
        tasks.append({"task": e["task"], "cands": cands, "prog": prog_by_entry[id(e)]})

    rankers = build_rankers(tasks)
    # union baselines (grouped LOTO): no-exec control vs +exec features
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
    rand_terms = []
    for t in tasks:
        ng = sum(1 for c in t["cands"] if c["correct"])
        nc = len(t["cands"])
        rand_terms.append(
            0.0
            if ng == 0
            else (
                1.0 - math.comb(nc - ng, min(2, nc)) / math.comb(nc, min(2, nc)) if nc >= 2 else 1.0
            )
        )
    random_baseline = round(float(np.mean(rand_terms)), 4)

    # safety + headroom accounting for the GATED ranker (the headline)
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
    per_task = []
    for i, t in enumerate(tasks):
        prog = t["prog"]
        gold_pred = bool(
            prog
            and prog["pred_hash"]
            and any(c["correct"] and ghash(c["grid"]) == prog["pred_hash"] for c in t["cands"])
        )
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
                "pred_is_gold": gold_pred,  # post-hoc scoring only
            }
        )

    def _lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            yield x

    def _boot(kA, kB):
        gen, deltas = _lcg(SEED), []

        def p2(sample, key):
            return sum(
                int(any(c["correct"] for c in sorted(t["cands"], key=key)[:2])) for t in sample
            ) / len(sample)

        for _ in range(1000):
            samp = [tasks[next(gen) % n] for _ in range(n)]
            deltas.append(p2(samp, kA) - p2(samp, kB))
        deltas.sort()
        return [round(deltas[25], 4), round(deltas[974], 4)]

    vote2 = res["TRM_VOTE"]["pass@2"]
    g2 = res["GAP4_GATED"]["pass@2"]
    n_perfect = sum(1 for t in tasks if t["prog"] and t["prog"]["demo_perfect"])
    gates = {
        "selection_beats_vote": bool(g2 > vote2),
        "selection_beats_union": bool(g2 > res["UNION_votes_qmean_voteshare"]["pass@2"]),
        "vote_wins_lost": len(vote_wins_lost),
        "headroom_recovered": len(headroom_recovered),
        "headroom_capture_fraction": round((g2 - vote2) / max(1e-9, oracle2 - vote2), 4),
        "coverage_demo_perfect": round(n_perfect / n, 4),
    }
    total_calls = sum(t["prog"]["n_calls"] for t in tasks if t["prog"])
    total_codex_s = round(sum(t["prog"]["codex_seconds"] for t in tasks if t["prog"]), 1)
    verdict = (
        "complete: gap4_rule_exec_"
        + ("BEATS_vote" if gates["selection_beats_vote"] else "does_not_beat_vote")
        + f"_n{n}_vote_{vote2}_gated_{g2}_recovered_{len(headroom_recovered)}"
        + f"_lost_{len(vote_wins_lost)}_demoperfect_{n_perfect}of{n}"
    )
    art = {
        "experiment": experiment_name,
        "title": "GAP-4: program-induction + execution-consistency verifier vs TRM frequency vote",
        "honest_verdict": verdict,
        "inference_substrate": "codex_program_induction_plus_offline_trm_candidate_rerank_no_oracle",
        "n_tasks": n,
        "n_unique_tasks": len(by_task),
        "n_oracle_hit": n_oracle,
        "oracle_pass2_ceiling": oracle2,
        "random_ranker_pass2_baseline": random_baseline,
        "rankers": res,
        "gates": gates,
        "vote_wins_lost_tasks": [per_task[i]["task"] for i in vote_wins_lost],
        "headroom_recovered_tasks": [per_task[i]["task"] for i in headroom_recovered],
        "bootstrap": {
            "gated_vs_vote_pass2_ci95": _boot(kg, kv),
            "gated_vs_union_pass2_ci95": _boot(kg, rankers["UNION_votes_qmean_voteshare"]),
            "B": 1000,
        },
        "per_task": per_task,
        "generator": {
            "model": "gpt-5.5 via codex exec (reasoning_effort=medium)",
            "total_codex_calls": total_calls,
            "total_codex_seconds": total_codex_s,
            "iters_per_task_max": iters,
            "data_handling_class": "minimize (public ARC demo grids + test input only)",
        },
        "no_oracle_audit": (
            "The codex prompt contains demo pairs + the test INPUT only — never the test gold and "
            "never the candidate pool. Programs are verified on demos (public context); the gated "
            "ranker overrides vote ONLY on demo_fit==1.0 AND exact candidate match of the executed "
            "output; otherwise it IS vote. 'correct' labels score rankings post-hoc."
        ),
        "decentralization_note": (
            "Closed-weight generator (gpt-5.5) used as the escalation-tier inducer per the M2-v3 "
            "precedent; deployment tier targets a local open-weight inducer (Gemma-4/Qwen3.6 GGUF). "
            "The verifier side (demo-fit + execution + rerank) is fully local and model-free."
        ),
        "preconditions_checked": [{"resource": k, "available": v} for k, v in pre.items()],
        "random_seed": SEED,
        "no_gpu_used": True,
        "duration_s": round(time.time() - started, 1),
    }
    if write:
        Path(artifact_path).write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
        progs = [
            {k: v for k, v in t["prog"].items() if k != "history"} | {"entry_i": i}
            for i, t in enumerate(tasks)
            if t["prog"]
        ]
        Path(programs_path).write_text(
            json.dumps(
                {
                    "experiment": "arc3_gap4_induced_programs",
                    "programs": progs,
                    "histories": [
                        {"task": t["task"], "history": t["prog"]["history"]}
                        for t in tasks
                        if t["prog"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(f"\n-> {verdict}")
    for r in rankers:
        print(f"   {r:30s} pass@1={res[r]['pass@1']} pass@2={res[r]['pass@2']}")
    print(
        f"   oracle={oracle2} random={random_baseline} demo_perfect={n_perfect}/{n} gates={gates}"
    )
    print(f"   recovered={art['headroom_recovered_tasks']} lost={art['vote_wins_lost_tasks']}")
    print(
        f"   bootstrap gated-vote={art['bootstrap']['gated_vs_vote_pass2_ci95']} "
        f"gated-union={art['bootstrap']['gated_vs_union_pass2_ci95']}"
    )
    print(f"   codex: {total_calls} calls / {total_codex_s}s")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap entries for a plumbing smoke")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--artifact", default=ARTIFACT)
    ap.add_argument("--programs", default=PROGRAMS)
    ap.add_argument(
        "--transcripts", default=None, help="dir to archive full per-call codex transcripts"
    )
    ap.add_argument("--name", default="arc3_gap4_rule_exec_verifier")
    a = ap.parse_args()
    run(
        limit=a.limit,
        iters=a.iters,
        workers=a.workers,
        timeout=a.timeout,
        pool_path=a.pool,
        artifact_path=a.artifact,
        programs_path=a.programs,
        transcripts_dir=a.transcripts,
        experiment_name=a.name,
    )
