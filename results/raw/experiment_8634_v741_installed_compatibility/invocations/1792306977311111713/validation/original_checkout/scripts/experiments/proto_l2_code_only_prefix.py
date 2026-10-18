"""PROTO-L2-CODE-ONLY-PREFIX: can a STRONGER code-only prefix make the lp85 L2 induction emit a
valid engine + is_level_complete at the SHIPPING max_tokens=4096 -- without raising the budget?

Background: proto_l2_proposer_truncation_check proved the 4096-token L2 induction call burns the
ENTIRE budget on win-state chain-of-thought (CoT) prose (stop_type='limit', tokens_predicted=4096,
ZERO code emitted). The '/no_think' prefix did NOT suppress the analysis -- the model reasons about
the WIN STATE grid until the budget is exhausted, never reaching the code block. So the L2
goal-induction nulls (goal_predicate_satisfiable=False) for ~10 milestones.

This proto tests the operator-directed CHEAP FIX: a forceful CODE-ONLY directive (Arm A = pure
prefix; Arm B = prefix + function-signature PREFILL that starts the model inside the code) that
stops the model analyzing and emits code directly, so the code fits in 4096 generated tokens with
NO budget/latency hit. If CoT is irrepressible at 4096, FALL BACK to raise-budget (8192) + a long
timeout (Arm C).

Design for an AIRTIGHT, WITHIN-RUN controlled comparison:
  - BASELINE arm reproduces the truncation on the EXACT prompt bytes used by the test arms
    (the historical artifact's prompt was built by an earlier version of the truncation-check
    script -- different synthetic-grid bytes / key names -- so we re-establish the control here
    rather than trust a stale hash). The prompt is substantively identical: lp85 L2 induce_prompt
    WITH a win-state exemplar, WIN STATE block present.
  - All arms run against the SAME warm Qwen3.5-9B-MTP server on :8920 (no startup, no 2nd GPU
    resident, 4 idle slots confirmed). cache_prompt=True; temperature=0.2 (shipping fidelity).

SHORT-CIRCUIT: run BASELINE, then Arm A (4096). If A yields valid code -> done (record, then test
goal satisfiability). Else run Arm B (4096 prefill). If B works -> done. Else run Arm C (8192
fallback). Whichever works is the recommended fix for the live induce path.

Goal SATISFIABILITY test on the winning code: exec it, check is_level_complete(win_exemplar)==True
AND non-degenerate (returns False on >=1 other grid). The downstream graded-bias fix needs a
satisfiable, non-degenerate goal predicate.

Integrity: inference_substrate=live_llm_inference, solve_provenance=development_proxy,
verifier_is_oracle=false. Does NOT edit shipped files. Does NOT submit anything.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

PORT = int(os.environ.get("L2_PREFIX_PORT", "8920"))
URL = f"http://127.0.0.1:{PORT}"
QWEN_GGUF = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/"
    "9716a636ee4bddc3fed678220b7a33dd2a4160ae/Qwen3.5-9B-Q4_K_M.gguf"
)
RESULT_PATH = REPO / "results" / "proto_l2_code_only_prefix.json"

from carnot.agentic.arc_executable_world_model import (  # noqa: E402
    Transition,
    induce_prompt,
    _extract_python,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# PRECONDITIONS (step 0): GGUF cached + server healthy + serving Qwen + idle
# ---------------------------------------------------------------------------
def preconditions() -> dict:
    pc = {"qwen_gguf": os.path.isfile(QWEN_GGUF)}
    try:
        with urllib.request.urlopen(URL + "/health", timeout=4) as r:
            pc["server_health"] = json.load(r).get("status") == "ok"
    except Exception as ex:
        pc["server_health"] = False
        pc["health_err"] = str(ex)[:120]
    try:
        with urllib.request.urlopen(URL + "/props", timeout=5) as r:
            props = json.load(r)
        mp = props.get("model_path", "")
        pc["serving_qwen"] = "Qwen" in mp or "qwen" in mp.lower()
        pc["n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
        pc["model_path"] = mp
    except Exception as ex:
        pc["serving_qwen"] = False
        pc["props_err"] = str(ex)[:120]
    return pc


# ---------------------------------------------------------------------------
# Reconstruct the lp85 L2 induce prompt + win-state exemplar (deterministic)
# ---------------------------------------------------------------------------
def build_prompt() -> tuple[str, str, np.ndarray, list]:
    rng = np.random.default_rng(42)
    trans = []
    for i in range(6):
        grid = rng.integers(0, 3, size=(5, 5), dtype=np.uint8)
        ng = grid.copy()
        if i % 2 == 0 and ng[2, 2] < 2:
            ng[2, 2] += 1
        trans.append(
            Transition(grid=grid, action=i % 4, data={}, next_grid=ng,
                       level_before=0, level_after=1 if i == 5 else 0)
        )
    prev = rng.integers(0, 3, size=(5, 5), dtype=np.uint8)
    base = induce_prompt("lp85", trans, 1, previous_level_complete_grid=np.asarray(prev))
    weak_suffix = "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n"
    captured = base + weak_suffix
    return base, captured, np.asarray(prev), trans


# The STRONGER code-only directive (the operator's "stronger code-only prefix").
STRONG_DIRECTIVE = (
    "/no_think\n"
    "CRITICAL OUTPUT RULES -- obey EXACTLY:\n"
    "1. Output ONLY one ```python code block. NOTHING before it. NOTHING after it.\n"
    "2. Do NOT analyze the grids. Do NOT describe or reason about the win state. Do NOT write\n"
    "   step-by-step analysis, explanation, or commentary -- not even as comments.\n"
    "3. Your response MUST begin with the characters ```python and end with ```.\n"
    "4. Induce SIMPLE, GENERAL rules and write the two functions directly. Skip all reasoning.\n"
    "\n"
)


def call_completion(prompt: str, n_predict: int, timeout: int) -> dict:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.2,
        "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(
        URL + "/completion", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    resp["_duration_s"] = round(time.time() - t0, 2)
    return resp


def parse_fenced(content: str, required=("engine", "is_level_complete")) -> dict:
    """For arms whose prompt ENDS with an opened ```python fence: prepend the opener, but also
    try the bare content (in case the model wrote its own fenced block after prose)."""
    candidates = []
    c1 = _extract_python("```python\n" + content)
    if c1:
        candidates.append(c1)
    c2 = _extract_python(content)
    if c2:
        candidates.append(c2)
    return _best_parse(candidates, required)


def parse_prefill(content: str, prefill_code: str, required=("engine", "is_level_complete")) -> dict:
    """For the prefill arm: the model continues from inside `prefill_code`. Cut at the first
    closing fence, prepend the prefill, parse."""
    raw = content
    idx = raw.find("```")
    if idx != -1:
        raw = raw[:idx]
    code = prefill_code + raw
    return _best_parse([code], required)


def _best_parse(candidates: list[str], required) -> dict:
    best = {"parse_ok": False, "code": "", "missing": list(required), "syntax_error": None,
            "code_chars": 0}
    for code in candidates:
        if not code:
            continue
        missing = [fn for fn in required if f"def {fn}" not in code]
        syn = None
        if not missing:
            try:
                ast.parse(code)
            except SyntaxError as se:
                syn = f"SyntaxError line {se.lineno}: {se.msg}"
        ok = (not missing) and syn is None
        rec = {"parse_ok": ok, "code": code, "missing": missing, "syntax_error": syn,
               "code_chars": len(code)}
        if ok:
            return rec
        # keep the most-complete failing candidate
        if len(required) - len(missing) > len(required) - len(best["missing"]):
            best = rec
    return best


def satisfiability(code: str, win_exemplar: np.ndarray, init_grid: np.ndarray) -> dict:
    """Exec the induced code; test is_level_complete on the win exemplar (expect True) and on
    several other grids (expect >=1 False, i.e. non-degenerate). Also check engine runs."""
    out = {"exec_ok": False, "win_state_complete": None, "non_degenerate": None,
           "engine_runs": None, "goal_predicate_satisfiable": False, "error": None,
           "is_level_complete_on": {}}
    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<induced>", "exec"), ns)  # noqa: S102 (sandboxless dev proxy)
        ilc = ns.get("is_level_complete")
        eng = ns.get("engine")
        out["exec_ok"] = ilc is not None and eng is not None
        if ilc is None or eng is None:
            out["error"] = "missing engine or is_level_complete after exec"
            return out
        # win exemplar
        win = bool(ilc(win_exemplar.copy()))
        out["win_state_complete"] = win
        out["is_level_complete_on"]["win_exemplar"] = win
        # other grids (init + 3 random) -> need >=1 False for non-degeneracy
        rng = np.random.default_rng(7)
        others = [("init", init_grid.copy())] + [
            (f"rand{i}", rng.integers(0, 3, size=win_exemplar.shape, dtype=np.uint8)) for i in range(3)
        ]
        false_seen = False
        for name, g in others:
            try:
                v = bool(ilc(g))
            except Exception as ex:
                v = f"err:{str(ex)[:40]}"
            out["is_level_complete_on"][name] = v
            if v is False:
                false_seen = True
        out["non_degenerate"] = false_seen
        # engine runs without raising and returns same-shape grid
        try:
            ng = np.asarray(eng(init_grid.copy(), 1, None))
            out["engine_runs"] = ng.shape == init_grid.shape
        except Exception as ex:
            out["engine_runs"] = False
            out["engine_err"] = str(ex)[:120]
        out["goal_predicate_satisfiable"] = bool(win and false_seen)
    except Exception as ex:
        out["error"] = str(ex)[:200]
    return out


def summarize_call(resp: dict) -> dict:
    content = resp.get("content", "")
    return {
        "duration_s": resp.get("_duration_s"),
        "stop_type": resp.get("stop_type"),
        "tokens_predicted": resp.get("tokens_predicted"),
        "content_chars": len(content),
        "content_tail_300": content[-300:],
    }


def main() -> int:
    t_start = time.time()
    pc = preconditions()
    log(f"PRECONDITIONS: {pc}")
    if not (pc.get("qwen_gguf") and pc.get("server_health") and pc.get("serving_qwen")):
        RESULT_PATH.write_text(json.dumps({
            "experiment_id": "proto_l2_code_only_prefix",
            "honest_verdict": "blocked_preconditions",
            "preconditions_checked": pc,
            "inference_substrate": "live_llm_inference",
            "solve_provenance": "development_proxy",
            "verifier_is_oracle": False,
            "duration_s": round(time.time() - t_start, 2),
        }, indent=2))
        log("BLOCKED: preconditions not met")
        return 0

    base, captured, win_exemplar, trans = build_prompt()
    init_grid = np.asarray(trans[0].grid)
    captured_sha = hashlib.sha256(captured.encode()).hexdigest()[:16]
    log(f"prompt rebuilt: base={len(base)}c captured={len(captured)}c sha={captured_sha} "
        f"(WIN STATE block present={'WIN STATE' in base})")

    arms: dict = {}
    winner = None
    PREFILL_CODE = "import numpy as np\n\ndef engine(grid, action, data):\n"

    # ---- BASELINE: exact captured form, /no_think only (reproduce truncation control) ----
    log("BASELINE: /no_think + weak suffix @4096 (expect truncation)...")
    try:
        r = call_completion(captured, n_predict=4096, timeout=850)
        parsed = parse_fenced(r.get("content", ""))
        arms["baseline_noThink_4096"] = {**summarize_call(r), "parse_ok": parsed["parse_ok"],
                                         "missing_fns": parsed["missing"],
                                         "syntax_error": parsed["syntax_error"],
                                         "code_chars": parsed["code_chars"]}
        log(f"  baseline: stop={r.get('stop_type')} toks={r.get('tokens_predicted')} "
            f"parse_ok={parsed['parse_ok']} dur={r.get('_duration_s')}s")
    except Exception as ex:
        arms["baseline_noThink_4096"] = {"error": str(ex)[:200]}
        log(f"  baseline ERROR: {ex}")

    # ---- ARM A: strong code-only prefix (pure prefix), fenced suffix @4096 ----
    log("ARM A: STRONG_DIRECTIVE prefix + fenced suffix @4096...")
    promptA = STRONG_DIRECTIVE + base + "\n```python\n"
    try:
        r = call_completion(promptA, n_predict=4096, timeout=850)
        parsed = parse_fenced(r.get("content", ""))
        arms["A_strong_prefix_4096"] = {**summarize_call(r), "parse_ok": parsed["parse_ok"],
                                        "missing_fns": parsed["missing"],
                                        "syntax_error": parsed["syntax_error"],
                                        "code_chars": parsed["code_chars"]}
        log(f"  A: stop={r.get('stop_type')} toks={r.get('tokens_predicted')} "
            f"parse_ok={parsed['parse_ok']} dur={r.get('_duration_s')}s")
        if parsed["parse_ok"]:
            winner = ("A_strong_prefix_4096", parsed["code"])
    except Exception as ex:
        arms["A_strong_prefix_4096"] = {"error": str(ex)[:200]}
        log(f"  A ERROR: {ex}")

    # ---- ARM B: strong prefix + signature PREFILL @4096 (only if A failed) ----
    if winner is None:
        log("ARM B: STRONG_DIRECTIVE + signature PREFILL @4096...")
        promptB = (STRONG_DIRECTIVE + base +
                   "\n\nWrite the code now -- continue directly from this opening:\n\n```python\n"
                   + PREFILL_CODE)
        try:
            r = call_completion(promptB, n_predict=4096, timeout=850)
            parsed = parse_prefill(r.get("content", ""), PREFILL_CODE)
            arms["B_prefill_4096"] = {**summarize_call(r), "parse_ok": parsed["parse_ok"],
                                      "missing_fns": parsed["missing"],
                                      "syntax_error": parsed["syntax_error"],
                                      "code_chars": parsed["code_chars"]}
            log(f"  B: stop={r.get('stop_type')} toks={r.get('tokens_predicted')} "
                f"parse_ok={parsed['parse_ok']} dur={r.get('_duration_s')}s")
            if parsed["parse_ok"]:
                winner = ("B_prefill_4096", parsed["code"])
        except Exception as ex:
            arms["B_prefill_4096"] = {"error": str(ex)[:200]}
            log(f"  B ERROR: {ex}")

    # ---- ARM C: raise-budget fallback @8192 + long timeout (only if A and B failed) ----
    if winner is None:
        log("ARM C (FALLBACK): STRONG_DIRECTIVE + prefill @8192, long timeout...")
        promptC = (STRONG_DIRECTIVE + base +
                   "\n\nWrite the code now -- continue directly from this opening:\n\n```python\n"
                   + PREFILL_CODE)
        try:
            r = call_completion(promptC, n_predict=8192, timeout=1500)
            parsed = parse_prefill(r.get("content", ""), PREFILL_CODE)
            arms["C_raise_budget_8192"] = {**summarize_call(r), "parse_ok": parsed["parse_ok"],
                                           "missing_fns": parsed["missing"],
                                           "syntax_error": parsed["syntax_error"],
                                           "code_chars": parsed["code_chars"]}
            log(f"  C: stop={r.get('stop_type')} toks={r.get('tokens_predicted')} "
                f"parse_ok={parsed['parse_ok']} dur={r.get('_duration_s')}s")
            if parsed["parse_ok"]:
                winner = ("C_raise_budget_8192", parsed["code"])
        except Exception as ex:
            arms["C_raise_budget_8192"] = {"error": str(ex)[:200]}
            log(f"  C ERROR: {ex}")

    # ---- Satisfiability test on the winner ----
    sat = None
    if winner is not None:
        log(f"WINNER={winner[0]} -- testing goal satisfiability...")
        sat = satisfiability(winner[1], win_exemplar, init_grid)
        log(f"  satisfiability: {sat}")

    # ---- Verdict ----
    baseline_truncated = (
        arms.get("baseline_noThink_4096", {}).get("stop_type") == "limit"
        and not arms.get("baseline_noThink_4096", {}).get("parse_ok", False)
    )
    if winner is not None and winner[0].endswith("4096"):
        if sat and sat.get("goal_predicate_satisfiable"):
            verdict = "complete_code_only_prefix_fixes_it_satisfiable"
            summary = (f"{winner[0]} emits valid engine+is_level_complete at 4096 (no budget hit) "
                       f"AND the goal predicate is SATISFIABLE+non-degenerate. Chain to graded-bias.")
        else:
            verdict = "complete_code_only_prefix_emits_code_but_goal_unsatisfiable"
            summary = (f"{winner[0]} emits valid code at 4096, but is_level_complete is NOT "
                       f"satisfiable/non-degenerate on the win exemplar (sat={sat}).")
    elif winner is not None and winner[0].startswith("C"):
        if sat and sat.get("goal_predicate_satisfiable"):
            verdict = "complete_raise_budget_fixes_it_satisfiable"
            summary = ("CoT is irrepressible at 4096; the 8192 fallback emits valid+satisfiable "
                       "code. Fix = raise L2 max_tokens to 8192 (acknowledged latency hit).")
        else:
            verdict = "complete_raise_budget_emits_code_but_goal_unsatisfiable"
            summary = ("8192 fallback emits valid code but goal not satisfiable/non-degenerate.")
    else:
        verdict = "complete_no_fix_generation_quality_wall"
        summary = ("Neither the strong code-only prefix (4096) nor the 8192 fallback produced "
                   "valid engine+is_level_complete. The wall is generation quality, not budget.")

    artifact = {
        "experiment_id": "proto_l2_code_only_prefix",
        "honest_verdict": verdict,
        "verdict_summary": summary,
        "baseline_truncation_reproduced_on_this_prompt": baseline_truncated,
        "winner_arm": winner[0] if winner else None,
        "goal_satisfiability": sat,
        "arms": arms,
        "prompt_info": {
            "game": "lp85",
            "base_chars": len(base),
            "captured_chars": len(captured),
            "captured_sha256_prefix": captured_sha,
            "note": ("sha differs from the historical truncation-check artifact (4db52dc7) because "
                     "that artifact was written by an earlier proto version with different synthetic "
                     "bytes/keys; the prompt is substantively identical -- lp85 L2 induce + win-state "
                     "exemplar -- and truncation is re-confirmed here as the matched within-run control."),
            "win_exemplar": win_exemplar.tolist(),
        },
        "strong_directive": STRONG_DIRECTIVE,
        "preconditions_checked": pc,
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "random_seed": 42,
        "duration_s": round(time.time() - t_start, 2),
        "model_port": PORT,
        "model_path": QWEN_GGUF,
        # adversarial_verify honest-null markers (no positive-control needed: this is not a null claim)
    }
    if winner is not None:
        artifact["winning_code"] = winner[1]
    RESULT_PATH.write_text(json.dumps(artifact, indent=2))
    log(f"VERDICT: {verdict}")
    log(f"SUMMARY: {summary}")
    log(f"DONE in {artifact['duration_s']}s -> {RESULT_PATH}")
    # chain marker for the operator-directed next step
    if winner is not None and sat and sat.get("goal_predicate_satisfiable"):
        log("CHAIN: goal SATISFIABLE -> proceed to graded-bias re-test with the winning fix wired in.")
    else:
        log("CHAIN: NOT satisfiable -> do NOT chain to graded-bias; report the wall.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
