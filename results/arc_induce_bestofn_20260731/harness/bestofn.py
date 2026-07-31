#!/usr/bin/env python3
"""BEST-OF-N, STEP 3 -- generate N candidate world models per STALL induction. Live, one GPU.

THE QUESTION. The shipped induce path asks the model ONCE and takes the first candidate that
clears a MECHANICAL bar (non-empty, defines the functions, parses, does not raise on the shown
transitions). Best-of-N asks N times and lets the VERIFIER choose. That is this project's core
thesis -- verification selecting among candidates -- applied to the one decision the mediation
analysis says the pipeline actually makes badly.

WHY THE COMBINED PROMPT AND NOT THE ENGINE-ONLY ONE. The confirm phase measured
`prompt{i}_engine.txt` because it was about engine VALIDITY. Best-of-N has to grade the GOAL
predicate too -- criteria (ii) and (iii) are functions of `is_level_complete`, which the
engine-only prompt never asks for. `prompt1_combined.txt` is also the SHIPPED happy path:
`LocalGGUFProposer.induce` issues the combined call first and only falls back to two focused
calls when it fails.

WHY call_index 1 AND NOT 2. Both induce calls belong to the SAME stall attempt -- verified by
stack capture, not assumed: call 1 comes from `execute_bounded_llm_reinduction`
(arc_competition_agent.py:5770, the stall branch) and call 2 from the plain single-shot path
(arc_competition_agent.py:5889) further down the same `_induce_and_plan`. Call 1 is the one this
phase measures because all THREE gates the criteria name live on it and run by default:

    heldout verifier   ->  arc_llm_reinduction.py:1443   (criterion (i))
    goal gate          ->  arc_llm_reinduction.py:1511   (criterion (ii))
    plan_in_model      ->  arc_llm_reinduction.py:1739   (criterion (iii))

The plain path's goal check is opt-in and DEFAULT OFF (`CARNOT_ARC_PLAIN_PATH_GOAL_
SATISFIABILITY_CHECK`), so grading criterion (ii) there would grade a gate the live agent does
not run. Call 1's split is also strictly richer: its `full` list is the real 25-row set, so the
live suffix is inside the held-out set instead of being lost (at call 2 `_proposal_prefix` was
never applied, so full == prefix and the suffix is unrecoverable). Measured consequence: ft09
goes from 0 held-out CHANGING rows at call 2 to 4 at call 1 -- an engine that is graded only on
no-ops cannot be told apart from the identity function.

TEMPERATURE IS HELD FIXED AT THE SHIPPED FIRST-ATTEMPT VALUE (0.2); only the seed varies. The
shipped `generate()` walks a ladder (0.2 + 0.1*attempt) as it retries, so an 8-long ladder would
run out to 0.9 -- far outside anything this path has ever been measured at -- and would confound
"more samples" with "hotter samples": every marginal candidate at N=8 would be hotter than every
candidate at N=1, and a gain could not be attributed to the N axis. Fixed temperature makes N a
pure sampling axis. It is also the PESSIMISTIC choice for diversity, so a gain measured here is
a floor. If it turns out the 8 candidates are near-identical, that is not a silent confound: the
scorer reports distinct-code-sha and distinct-behaviour counts, and a degenerate pool is reported
as the mechanism rather than left to look like "sampling does not help".

EVERYTHING ELSE IS THE SHIPPED PAYLOAD, read from the repo constants rather than retyped:
`repeat_penalty` = `_induce_repeat_penalty()` (wired 2026-07-31, so this IS the shipped sampler
now), `repeat_last_n` = `_INDUCE_REPEAT_LAST_N`, `n_predict` = `_INDUCE_DEFAULT_MAX_TOKENS`,
`cache_prompt` True, `stop` ["```"] (every induce call is codeonly_eligible).

CANDIDATE-MAJOR ACROSS GAMES, deliberately, and for the same reason the confirm phase was
attempt-major: a wall-clock cutoff at any point leaves every game with the SAME number of
candidates, so the grid is reportable at whatever N completed rather than being an unbalanced
set that cannot be pooled. Partial JSON is rewritten after EVERY candidate.

NOTHING HERE SCORES ANYTHING. Held-out rows are not opened, the goal gate is not run, the
planner is not run. Generation and scoring are separate processes so that no scoring decision
can leak back into what gets generated, and so the expensive GPU step never has to be repeated
when a scoring bug is found.

PINS, each paid for in an earlier incident: n_ctx=32768 (the shipped 81920 does not fit a 24 GiB
card and the 31B then silently binds the iGPU HIP build and runs LLM-OFF while REPORTING LLM-ON),
ffn_cpu_layers=0, MTP off, `CUDA_VISIBLE_DEVICES=""` on this parent, a NON-DEFAULT port (8919 is
the default and a stale server there is silently adopted), and the CUDA build PROVEN from
`/proc/<pid>/exe` plus a per-PID VRAM row rather than inferred from what was requested.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = os.path.dirname(os.path.abspath(__file__))

GAMES = [g for g in os.environ.get("BON_GAMES", "ft09,tn36,vc33").split(",") if g]
PORT = int(os.environ.get("BON_PORT", "8961"))
GPU = os.environ.get("BON_GPU", "0")
CALL_INDEX = int(os.environ.get("BON_CALL_INDEX", "1"))
N_CANDIDATES = int(os.environ.get("BON_N", "8"))
SEED_BASE = int(os.environ.get("BON_SEED_BASE", "7100"))
TEMPERATURE = float(os.environ.get("BON_TEMPERATURE", "0.2"))
CALL_TIMEOUT_S = float(os.environ.get("BON_CALL_TIMEOUT_S", "1800"))
DEADLINE_S = float(os.environ.get("BON_DEADLINE_S", "10800"))  # 3h hard stop
REPO_SUBSTR = "gemma-4-31B-it"
TAG = os.environ.get("BON_TAG", f"gpu{GPU}")

OUT = os.path.join(HERE, "bon", TAG)
os.makedirs(OUT, exist_ok=True)

os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = GPU
os.environ["CARNOT_ARC_INDUCE_N_CTX"] = "32768"
os.environ["CARNOT_ARC_FFN_CPU_LAYERS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CARNOT_ARC_E3_DIR"] = os.path.join(OUT, "e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)

sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, HERE)

_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{TAG} {round(time.monotonic() - _T0, 1):>7}s] {msg}", flush=True)


def main() -> int:  # noqa: C901
    import urllib.request

    from carnot.agentic import arc_engine_static_validation as sv
    from carnot.agentic import arc_executable_world_model as e3
    from split import load_split

    for mod in (e3, sv):
        assert mod.__file__.startswith(REPO), f"wrong repo code imported: {mod.__file__}"

    # THE SHIPPED SAMPLER, read from the repo rather than retyped. If someone changes the wired
    # default, this run follows it and the artifact records what it actually sent.
    sampler: dict = {}
    _rp = e3._induce_repeat_penalty()
    if _rp != 1.0:
        sampler["repeat_penalty"] = _rp
        sampler["repeat_last_n"] = e3._INDUCE_REPEAT_LAST_N
    budget = e3._INDUCE_DEFAULT_MAX_TOKENS

    cells: dict = {}
    for game in GAMES:
        prompt_path = os.path.join(HERE, "capture", game, f"prompt{CALL_INDEX}_combined.txt")
        if not os.path.exists(prompt_path):
            log(f"SKIP {game}: no capture at {prompt_path}")
            continue
        s = load_split(game, CALL_INDEX)
        if not s["split_proven"]:
            log(f"SKIP {game}: split not proven -> {s['checks']}")
            continue
        cells[game] = {
            "prompt": open(prompt_path).read(),
            # ONLY the shown rows reach the mechanical defect check -- the same leak discipline
            # the confirm phase ran under. Held-out rows are opened for the first time by the
            # scorer, in a different process.
            "shown": s["_shown"],
            "split": {k: v for k, v in s.items() if not k.startswith("_")},
        }
        log(
            f"{game}: prompt sha {s['prompt_sha256_16']} shown={s['n_shown']} "
            f"heldout={s['n_heldout']} (chg {s['heldout_n_changing']})"
        )
    if not cells:
        json.dump(
            {"status": "blocked_no_usable_capture", "games": GAMES},
            open(os.path.join(OUT, "bon.json"), "w"),
            indent=2,
        )
        return 2

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
            {
                "status": "blocked_generator_server_not_started",
                "selection_log": list(e3.GENERATOR_SELECTION_LOG)[-25:],
            },
            open(os.path.join(OUT, "bon.json"), "w"),
            indent=2,
        )
        return 3

    witness: dict = {
        "port": prop.port,
        "observed_model_path": prop.observed_model_path(),
        "observed_n_ctx": prop.observed_n_ctx(),
        "gpu_requested": GPU,
    }
    pid_out = subprocess.run(
        ["ss", "-lptnH", f"sport = :{PORT}"], capture_output=True, text=True, timeout=5
    ).stdout
    m = re.search(r"pid=(\d+)", pid_out)
    spid = int(m.group(1)) if m else None
    witness["server_pid"] = spid
    if spid is not None:
        witness["server_exe"] = os.path.realpath(f"/proc/{spid}/exe")
        witness["server_exe_is_cuda_build"] = "build-hip" not in witness["server_exe"]
        witness["server_cmdline"] = (
            open(f"/proc/{spid}/cmdline", "rb").read().replace(b"\0", b" ").decode()[:400]
        )
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,gpu_bus_id",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    witness["vram_rows_mine"] = [
        r.strip() for r in smi.splitlines() if spid is not None and r.strip().startswith(str(spid))
    ]
    log(f"witness: {json.dumps(witness)}")
    for bad, why in (
        (
            witness.get("server_exe_is_cuda_build") is not True,
            "blocked_generator_not_proven_cuda_build",
        ),
        (not witness.get("vram_rows_mine"), "blocked_generator_no_vram_residency"),
        (
            witness.get("observed_model_path")
            and REPO_SUBSTR not in str(witness["observed_model_path"]),
            "blocked_generator_wrong_model",
        ),
    ):
        if bad:
            json.dump(
                {"status": why, "witness": witness},
                open(os.path.join(OUT, "bon.json"), "w"),
                indent=2,
            )
            prop.stop()
            return 4

    rows: list[dict] = []

    def call(game: str, k: int) -> dict:
        prompt = cells[game]["prompt"]
        seed = SEED_BASE + k
        payload = {
            "prompt": prompt,
            "n_predict": budget,
            "temperature": TEMPERATURE,
            "cache_prompt": True,
            "stop": ["```"],  # codeonly_eligible=True on every induce call
            "seed": seed,
            **sampler,
        }
        t = time.monotonic()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/completion",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
                resp = json.load(r)
        except Exception as exc:  # noqa: BLE001
            return {
                "game": game,
                "candidate": k,
                "seed": seed,
                "status": f"http_error:{type(exc).__name__}",
                "error": str(exc)[:300],
                "wall_s": round(time.monotonic() - t, 1),
            }
        text = resp.get("content", "")
        timings = resp.get("timings") or {}
        code = e3._extract_python(text) or text.strip()
        shown = cells[game]["shown"]
        # `required` matches what the SHIPPED combined induce call passes to generate().
        defects = sv.validate_engine_code(
            code,
            transitions=shown,  # SHOWN ONLY -- no held-out row informs this
            stop_type=resp.get("stop_type"),
            required=("engine", "is_level_complete"),
            budget=budget,
        )
        try:
            import ast as _ast

            _ast.parse(code)
            parses = True
        except SyntaxError:
            parses = False
        # The SHIPPED accept-first bar for the combined call: both functions present and parsing.
        accepted = bool(code and "def engine" in code and "def is_level_complete" in code and parses)
        changes = sv.engine_changes_anything(code, shown)
        fn = f"{game}_k{k}.txt"
        with open(os.path.join(OUT, fn), "w") as fh:
            fh.write(text)
        row = {
            "game": game,
            "candidate": k,
            "seed": seed,
            "temperature": TEMPERATURE,
            "status": "ok",
            "completion_file": fn,
            "code_sha256_16": hashlib.sha256(code.encode()).hexdigest()[:16],
            "code_chars": len(code),
            "sampler": dict(sampler),
            "n_predict_requested": budget,
            "stop_type": resp.get("stop_type"),
            "predicted_n": timings.get("predicted_n"),
            "prompt_n": timings.get("prompt_n"),
            "prompt_truncated": bool(resp.get("truncated")),
            "wall_s": round(time.monotonic() - t, 1),
            "generate_would_accept": accepted,
            "defect_kinds": sorted({d.kind for d in defects}),
            "defect_details": [d.detail[:240] for d in defects],
            "engine_changes_anything": changes,
            # The mechanical bar only. NOT a quality claim -- criteria (i)/(ii)/(iii) are the
            # scorer's job and are computed in a separate process against held-out evidence.
            "usable": bool(accepted and not defects and changes is True),
        }
        log(
            f"  {game} k{k}: stop={row['stop_type']} pred_n={row['predicted_n']} "
            f"accept={accepted} defects={row['defect_kinds']} changes={changes} "
            f"USABLE={row['usable']} wall={row['wall_s']}s sha={row['code_sha256_16']}"
        )
        return row

    def flush(status: str) -> None:
        payload = {
            "status": status,
            "tag": TAG,
            "games": list(cells),
            "call_index": CALL_INDEX,
            "n_candidates_requested": N_CANDIDATES,
            "seed_base": SEED_BASE,
            "temperature": TEMPERATURE,
            "budget": budget,
            "sampler": sampler,
            "sampler_source": "repo constants: _induce_repeat_penalty() / _INDUCE_REPEAT_LAST_N",
            "witness": witness,
            "splits": {g: cells[g]["split"] for g in cells},
            "rows": rows,
            "wall_s": round(time.monotonic() - _T0, 1),
        }
        with open(os.path.join(OUT, "bon.json"), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    stopped = None
    # CANDIDATE-MAJOR: every game gets candidate k before any game gets k+1, so a cutoff leaves a
    # balanced grid that is reportable at whatever N completed.
    for k in range(N_CANDIDATES):
        for game in list(cells):
            if time.monotonic() - _T0 > DEADLINE_S:
                stopped = "deadline_reached"
                break
            rows.append(call(game, k))
            flush("partial")
        if stopped:
            break

    flush(stopped or "ok")
    log(f"DONE rows={len(rows)} stopped={stopped}")
    prop.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
