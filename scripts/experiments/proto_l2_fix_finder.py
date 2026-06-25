"""Find the FIX for the real lp85 L2 reinduction code-gen failure, using the CAPTURED real prompt.

Root cause (from proto_l2_capture): on the real lp85 L2 prompt the model rambles analysis into code
COMMENTS inside engine(), exhausts max_tokens=2560, and never writes is_level_complete (missing def
-> proposer_failed). Test candidate fixes on the EXACT captured prompt:

  B (raise budget):    same directive, n_predict=4096  -> is it just budget?
  C (goal-first):      a directive ordering is_level_complete BEFORE engine, n_predict=2560 ->
                       does writing the short goal first protect it from the engine ramble?
  D (separate goal):   a FOCUSED goal-only call (write ONLY is_level_complete from the WIN STATE),
                       n_predict=768 -> the structural fix (engine ramble cannot starve the goal).

Success = the result contains a parseable def is_level_complete (B/C: + def engine).
Warm Qwen :8920. inference_substrate=live_llm_inference; solve_provenance=development_proxy.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from carnot.agentic.arc_executable_world_model import _extract_python, _L2_CODEONLY_DIRECTIVE  # noqa

URL = "http://127.0.0.1:8920"
RESULT = REPO / "results" / "proto_l2_fix_finder.json"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def call(prompt: str, n_predict: int, stop=None, timeout: int = 900) -> dict:
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.2, "cache_prompt": True}
    if stop:
        body["stop"] = stop
    req = urllib.request.Request(
        URL + "/completion", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    resp["_dur"] = round(time.time() - t0, 1)
    return resp


def analyze(content: str, need=("engine", "is_level_complete")) -> dict:
    code = _extract_python("```python\n" + content + "\n```") or _extract_python(content) or content.strip()
    missing = [fn for fn in need if f"def {fn}" not in code]
    syn = None
    if not missing:
        try:
            ast.parse(code)
        except SyntaxError as se:
            syn = f"line {se.lineno}: {se.msg}"
    return {"parse_ok": (not missing and syn is None), "missing": missing, "syntax_error": syn,
            "code_chars": len(code)}


def main() -> int:
    # load captured real L2 reinduction prompt
    rec = None
    for line in open(REPO / "results" / "l2_capture.jsonl"):
        r = json.loads(line)
        if "WIN STATE" in r["prompt"]:
            rec = r
            break
    if rec is None:
        log("BLOCKED: no captured WIN STATE prompt")
        return 0
    captured = rec["prompt"]
    # recover the base induce prompt (strip the code-only directive prefix + the opened fence suffix)
    base = captured
    if base.startswith(_L2_CODEONLY_DIRECTIVE):
        base = base[len(_L2_CODEONLY_DIRECTIVE):]
    base = base.rsplit("\n```python\n", 1)[0]
    # recover the WIN STATE grid block for the focused goal call (D)
    win_idx = base.find("WIN STATE")
    win_block = base[win_idx: win_idx + 220] if win_idx >= 0 else ""
    log(f"captured prompt {len(captured)}c | base {len(base)}c | win_block present={bool(win_block)}")

    arms = {}

    # B: raise budget
    log("ARM B: same directive, n_predict=4096 ...")
    promptB = _L2_CODEONLY_DIRECTIVE + base + "\n```python\n"
    rB = call(promptB, 4096, stop=["```"])
    aB = analyze(rB["content"])
    arms["B_raise_budget_4096"] = {"stop_type": rB.get("stop_type"), "tokens": rB.get("tokens_predicted"),
                                   "dur": rB["_dur"], **aB}
    log(f"  B: stop={rB.get('stop_type')} toks={rB.get('tokens_predicted')} parse_ok={aB['parse_ok']} missing={aB['missing']} dur={rB['_dur']}s")

    # C: goal-first ordering (still 2560)
    log("ARM C: goal-first directive, n_predict=2560 ...")
    goal_first = (
        "/no_think\n"
        "CRITICAL: Output ONLY one ```python code block, NO prose, NO comments, NO analysis.\n"
        "Write the TWO functions in THIS ORDER -- is_level_complete FIRST, then engine:\n"
        "  import numpy as np\n"
        "  def is_level_complete(grid): ...   # write this FIRST, keep it short\n"
        "  def engine(grid, action, data): ...\n"
        "Begin immediately with ```python.\n\n"
    )
    promptC = goal_first + base + "\n```python\n"
    rC = call(promptC, 2560, stop=["```"])
    aC = analyze(rC["content"])
    arms["C_goal_first_2560"] = {"stop_type": rC.get("stop_type"), "tokens": rC.get("tokens_predicted"),
                                 "dur": rC["_dur"], **aC}
    log(f"  C: stop={rC.get('stop_type')} toks={rC.get('tokens_predicted')} parse_ok={aC['parse_ok']} missing={aC['missing']} dur={rC['_dur']}s")

    # D: separate focused goal-only call
    log("ARM D: separate focused is_level_complete-only call, n_predict=768 ...")
    promptD = (
        "/no_think\n"
        "Output ONLY one ```python code block with EXACTLY one function and NO comments/prose:\n"
        "    import numpy as np\n"
        "    def is_level_complete(grid):\n"
        "        # return True iff `grid` is the WIN STATE shown below, else False\n"
        f"The level is complete at this WIN STATE:\n{win_block}\n"
        "Write def is_level_complete now. Begin with ```python.\n\n```python\n"
    )
    rD = call(promptD, 768, stop=["```"])
    aD = analyze(rD["content"], need=("is_level_complete",))
    arms["D_separate_goal_768"] = {"stop_type": rD.get("stop_type"), "tokens": rD.get("tokens_predicted"),
                                   "dur": rD["_dur"], **aD}
    log(f"  D: stop={rD.get('stop_type')} toks={rD.get('tokens_predicted')} parse_ok={aD['parse_ok']} missing={aD['missing']} dur={rD['_dur']}s")

    winners = [k for k, v in arms.items() if v["parse_ok"]]
    if "D_separate_goal_768" in winners:
        verdict = "complete_fix_separate_goal_call"
        summary = ("The separate focused is_level_complete call (D) produces a valid goal predicate "
                   "where the combined engine+goal prompt fails -- the engine ramble can no longer "
                   "starve the goal. Recommended structural fix: induce the goal in its own call.")
    elif "B_raise_budget_4096" in winners or "C_goal_first_2560" in winners:
        verdict = "complete_fix_" + ("raise_budget" if "B_raise_budget_4096" in winners else "goal_first")
        summary = f"Working fix(es): {winners}. (D status: {arms['D_separate_goal_768']['parse_ok']})"
    else:
        verdict = "complete_no_simple_fix"
        summary = ("None of raise-budget / goal-first / separate-goal produced a valid result on the "
                   "real prompt -- deeper redesign needed.")

    artifact = {
        "experiment_id": "proto_l2_fix_finder",
        "honest_verdict": verdict,
        "verdict_summary": summary,
        "working_fixes": winners,
        "root_cause": ("real lp85 L2 prompt -> model rambles analysis into engine() comments -> "
                       "exhausts max_tokens=2560 -> is_level_complete never written (missing def)"),
        "captured_prompt_chars": len(captured),
        "arms": arms,
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "random_seed": 4729,
        "model_port": 8920,
    }
    RESULT.write_text(json.dumps(artifact, indent=2))
    log(f"VERDICT: {verdict}")
    log(f"SUMMARY: {summary}")
    log(f"WORKING FIXES: {winners}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
