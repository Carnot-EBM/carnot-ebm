#!/usr/bin/env python3
"""PHASE 1 -- does raising the induce COMPLETION BUDGET make ft09 emit a usable engine()?

THE PRIOR THIS MUST BEAT. `arc_executable_world_model.induce()` carries a comment from
proto_l2_fix_finder (2026-06-25) recording that this exact lever was already tried:

    "the focused goal call is valid in ~3.5s where the combined call fails
     (a budget bump does NOT help; the model just rambles more)."

So Phase 1 is a RERUN of a measured-negative lever and needs a stated reason to expect a
different answer (CLAUDE.md Failed-Experiment Rerun Discipline). Two things changed since:
  1. THE GENERATOR. That measurement was taken before the 2026-07-28 switch to
     gemma-4-31B-it. "This 9B rambles more when given more room" is not evidence about a
     different, 3.4x larger model.
  2. THE CALL. The 2026-06-25 note is about the COMBINED engine+goal call. ft09's observed
     comment wall came from the SPLIT path's ENGINE-ONLY call, which did not exist in the
     shape being described. Both are swept here so the two are never conflated again.
If the answer comes back "no help", that is the finding, and no default is changed.

WHAT IS MEASURED PER CALL, and why each is needed to tell the three failure modes apart:
  * `predicted_n` + `stop_type`     -- budget-limit vs pool-truncation vs natural stop.
  * `ramble_frac`                   -- share of emitted lines that are bare `#`. THE metric
                                       for "the model just rambles more": if the wall grows
                                       in proportion to the budget, budget is not the lever.
  * `has_engine` / `parses`         -- what `generate()` itself gates on.
  * `engine_returns_on_all_paths`   -- THE Phase-1 acceptance question. ft09's banked engine
                                       has 11 lines of code and a 1061-line `#` wall, and
                                       falls off the end of the `action == 6` branch, so it
                                       returns None on every click. A bigger budget that
                                       yields a longer comment wall and still no return is
                                       not progress, and only this check can say so.

PINS (each one paid for in an earlier incident, see the harness notes in p4/llmab_cell.py):
  n_ctx=32768 (81920 does not fit a 24 GiB card and the 31B silently falls through to the
  iGPU HIP build, then runs LLM-OFF while REPORTING LLM-ON), ffn_cpu_layers=0, MTP off,
  CUDA_VISIBLE_DEVICES="" on this parent, a NON-DEFAULT port (8919 is the default and a
  stale server there is silently adopted), and the CUDA build PROVEN from /proc/<pid>/exe
  plus a per-PID VRAM row rather than inferred from what we asked for.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = os.path.dirname(os.path.abspath(__file__))
# Per-lane output dir. Two lanes run concurrently on the two idle 3090s and BOTH would
# otherwise write `sweep/sweep.json`, so the second lane's rows would silently overwrite the
# first lane's -- a missing observation dressed up as a complete one.
OUT = os.path.join(HERE, os.environ.get("SWEEP_OUT", "sweep"))
os.makedirs(OUT, exist_ok=True)

PORT = int(os.environ.get("SWEEP_PORT", "8933"))
GPU = os.environ.get("SWEEP_GPU", "1")
BUDGETS = [int(x) for x in os.environ.get("SWEEP_BUDGETS", "4096,8192,16384").split(",")]
PROMPTS = os.environ.get("SWEEP_PROMPTS", "engine,combined").split(",")
ATTEMPTS = int(os.environ.get("SWEEP_ATTEMPTS", "3"))
CALL_TIMEOUT_S = float(os.environ.get("SWEEP_CALL_TIMEOUT_S", "1200"))
REPO_SUBSTR = "gemma-4-31B-it"

os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = GPU
os.environ["CARNOT_ARC_INDUCE_N_CTX"] = "32768"
os.environ["CARNOT_ARC_FFN_CPU_LAYERS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CARNOT_ARC_GENERATOR_SEED", "3003")

sys.path.insert(0, os.path.join(REPO, "python"))

_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{round(time.monotonic() - _T0, 1):>8}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------
# THE ACCEPTANCE CHECK
# ---------------------------------------------------------------------------------------
def _terminates(stmts: list) -> bool:
    """Does this statement list ALWAYS leave via return/raise, i.e. never fall off its end?

    Conservative on purpose: anything it cannot prove terminating is reported as
    non-terminating. For the Phase-1 question a false "does not return" is a
    re-examinable finding, while a false "returns fine" would hide the exact defect being
    measured.
    """
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.If):
        return _terminates(last.body) and _terminates(last.orelse)
    if isinstance(last, ast.While):
        # `while True:` with no `break` never falls through.
        const_true = isinstance(last.test, ast.Constant) and bool(last.test.value)
        has_break = any(isinstance(n, ast.Break) for n in ast.walk(last))
        return bool(const_true and not has_break)
    if isinstance(last, ast.Try):
        handlers_ok = all(_terminates(h.body) for h in last.handlers)
        if last.finalbody and _terminates(last.finalbody):
            return True
        return _terminates(last.body) and handlers_ok and (
            _terminates(last.orelse) if last.orelse else True
        )
    if isinstance(last, ast.With):
        return _terminates(last.body)
    if isinstance(last, ast.Match):
        return bool(last.cases) and all(_terminates(c.body) for c in last.cases)
    return False


def analyse_code(code: str) -> dict:
    """Everything `generate()` gates on, PLUS the defect it does not look at."""
    out: dict = {
        "has_engine": "def engine" in code,
        "has_is_level_complete": "def is_level_complete" in code,
        "parses": False,
        "engine_returns_on_all_paths": None,
        "engine_body_stmts": None,
        "syntax_error": None,
    }
    try:
        tree = ast.parse(code)
        out["parses"] = True
    except SyntaxError as se:
        out["syntax_error"] = f"line {se.lineno}: {se.msg}"
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "engine":
            out["engine_body_stmts"] = len(node.body)
            out["engine_returns_on_all_paths"] = _terminates(node.body)
            break
    return out


_BARE_COMMENT = re.compile(r"^\s*#\s*$")


def ramble_stats(text: str) -> dict:
    """How much of the completion is content-free comment padding.

    `bare_comment_lines` is the ft09 signature exactly (1061 consecutive `    #` lines). The
    LONGEST RUN is reported alongside the total because a model that sprinkles blank comments
    is a different phenomenon from one that gets stuck in a repetition loop, and only the
    second is what a budget increase would feed.
    """
    lines = text.split("\n")
    bare = [i for i, ln in enumerate(lines) if _BARE_COMMENT.match(ln)]
    run = longest = 1 if bare else 0
    for a, b in zip(bare, bare[1:]):
        run = run + 1 if b == a + 1 else 1
        longest = max(longest, run)
    return {
        "n_lines": len(lines),
        "bare_comment_lines": len(bare),
        "longest_bare_comment_run": longest,
        "ramble_frac": round(len(bare) / max(1, len(lines)), 4),
        "code_lines": sum(
            1 for ln in lines if ln.strip() and not ln.strip().startswith("#")
        ),
    }


def main() -> int:
    import urllib.request

    from carnot.agentic import arc_executable_world_model as e3

    assert e3.__file__.startswith(REPO), e3.__file__

    prop = e3.LocalGGUFProposer(
        repo_substr=REPO_SUBSTR,
        port=PORT,
        mtp=False,
        n_ctx=32768,
        ffn_cpu_layers=0,
        kv_quant="q8_0",
    )
    log("starting server ...")
    if not prop._ensure_server():
        json.dump(
            {"status": "blocked_generator_server_not_started",
             "selection_log": list(e3.GENERATOR_SELECTION_LOG)[-25:]},
            open(os.path.join(OUT, "sweep.json"), "w"), indent=2)
        return 3

    witness: dict = {
        "port": prop.port,
        "observed_model_path": prop.observed_model_path(),
        "observed_n_ctx": prop.observed_n_ctx(),
        "observed_total_slots": prop.observed_total_slots(),
        "gpu_requested": GPU,
    }
    # BUILD + CARD IDENTITY, OBSERVED. `prop._proc.args[0]` reads "reused_existing" whenever a
    # server started by an earlier process is adopted, so it cannot be the check.
    pid_out = subprocess.run(
        ["ss", "-lptnH", f"sport = :{PORT}"], capture_output=True, text=True, timeout=5
    ).stdout
    m = re.search(r"pid=(\d+)", pid_out)
    spid = int(m.group(1)) if m else None
    witness["server_pid"] = spid
    if spid is not None:
        witness["server_exe"] = os.path.realpath(f"/proc/{spid}/exe")
        witness["server_exe_is_cuda_build"] = "build-hip" not in witness["server_exe"]
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_bus_id",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5).stdout
    witness["vram_rows_mine"] = [
        r.strip() for r in smi.splitlines()
        if spid is not None and r.strip().startswith(str(spid))
    ]
    log(f"witness: {json.dumps(witness)}")
    for bad, why in (
        (witness.get("server_exe_is_cuda_build") is not True,
         "blocked_generator_not_proven_cuda_build"),
        (not witness.get("vram_rows_mine"), "blocked_generator_no_vram_residency"),
        (witness.get("observed_model_path")
         and REPO_SUBSTR not in str(witness["observed_model_path"]),
         "blocked_generator_wrong_model"),
    ):
        if bad:
            json.dump({"status": why, "witness": witness},
                      open(os.path.join(OUT, "sweep.json"), "w"), indent=2)
            return 4

    rows: list[dict] = []
    for pname in PROMPTS:
        path = os.path.join(HERE, "prompts", f"prompt2_{pname}.txt")
        prompt = open(path).read()
        required = ("engine",) if pname == "engine" else ("engine", "is_level_complete")
        for budget in BUDGETS:
            for attempt in range(ATTEMPTS):
                payload = {
                    "prompt": prompt,
                    "n_predict": budget,
                    "temperature": 0.2 + 0.1 * attempt,
                    "cache_prompt": True,
                    "stop": ["```"],  # codeonly_eligible=True on every induce call
                }
                seed = e3.LocalGGUFProposer.sampling_seed(attempt)
                if seed is not None:
                    payload["seed"] = seed
                t = time.monotonic()
                row: dict = {
                    "prompt": pname, "budget": budget, "attempt": attempt,
                    "temperature": payload["temperature"], "seed": seed,
                    "required": list(required),
                }
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{PORT}/completion",
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
                        resp = json.load(r)
                except Exception as exc:
                    row.update(status=f"http_error:{type(exc).__name__}",
                               error=str(exc)[:300],
                               wall_s=round(time.monotonic() - t, 1))
                    rows.append(row)
                    log(f"  {pname} b={budget} a={attempt} :: {row['status']}")
                    continue
                text = resp.get("content", "")
                timings = resp.get("timings") or {}
                # generate()'s own extraction, including its codeonly fallback: the stop
                # sequence eats the closing fence and the opener was in the prompt, so the
                # raw completion IS the block body.
                code = e3._extract_python(text) or text.strip()
                row.update(
                    status="ok",
                    stop_type=resp.get("stop_type"),
                    prompt_truncated=bool(resp.get("truncated")),
                    predicted_n=timings.get("predicted_n"),
                    prompt_n=timings.get("prompt_n"),
                    predict_per_second=round(timings.get("predicted_per_second") or 0, 2),
                    wall_s=round(time.monotonic() - t, 1),
                    n_chars=len(text),
                    **ramble_stats(text),
                    **analyse_code(code),
                )
                # generate() ACCEPTS when: code non-empty, every required `def` present, parses.
                row["generate_would_accept"] = bool(
                    code
                    and all(f"def {fn}" in code for fn in required)
                    and row["parses"]
                )
                # THE PHASE-1 ACCEPTANCE GATE: accepted AND the engine actually returns.
                row["usable_engine"] = bool(
                    row["generate_would_accept"] and row["engine_returns_on_all_paths"]
                )
                fn = f"{pname}_b{budget}_a{attempt}.txt"
                with open(os.path.join(OUT, fn), "w") as fh:
                    fh.write(text)
                row["completion_file"] = fn
                rows.append(row)
                log(
                    f"  {pname} b={budget} a={attempt} :: stop={row['stop_type']} "
                    f"pred_n={row['predicted_n']} ramble={row['ramble_frac']} "
                    f"accept={row['generate_would_accept']} "
                    f"returns={row['engine_returns_on_all_paths']} "
                    f"USABLE={row['usable_engine']} wall={row['wall_s']}s"
                )
                with open(os.path.join(OUT, "sweep.json"), "w") as fh:
                    json.dump({"status": "partial", "witness": witness, "rows": rows},
                              fh, indent=2, sort_keys=True)

    payload = {
        "status": "ok",
        "witness": witness,
        "budgets": BUDGETS,
        "prompts": PROMPTS,
        "attempts": ATTEMPTS,
        "rows": rows,
        "wall_s": round(time.monotonic() - _T0, 1),
    }
    with open(os.path.join(OUT, "sweep.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    log(f"DONE {len(rows)} rows in {payload['wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
