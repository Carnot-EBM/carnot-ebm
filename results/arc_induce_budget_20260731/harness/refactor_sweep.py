#!/usr/bin/env python3
"""PHASE 1, second axis -- the REFACTOR call, which is where ft09's truncation actually fired.

WHY THIS IS A SEPARATE SWEEP FROM `budget_sweep.py`. ft09 has TWO distinct defects and the
diagnosis conflates them:

  round 1 (INDUCE)   -- succeeded structurally; produced an engine() with 11 lines of code, a
                        1061-line `#` wall, and NO return on the `action == 6` path. It was not
                        rejected for truncation; it was rejected for being wrong (heldout 0.125).
  round 2 (REFACTOR) -- 3/3 tries returned `missing ('engine', 'is_level_complete') in output
                        [HIT n_predict=4096 OUTPUT LIMIT before completing]`. THIS is the
                        truncation.

The refactor call also has DIRECT prior evidence pointing the other way from the induce call's:
REQ-ARC-FCP-5699-34 (2026-07-16) replayed exactly this call shape on a 27B model and measured
that `max_tokens=8192` + the structural reminder turns `[HIT n_predict=4096 OUTPUT LIMIT]` into
`stop_type='eos'` with both required functions present. REQ-ARC-FCP-5699-35 then declined to
graduate 8192 to the live default, and said why in as many words: "the 8192 requirement
REQ-ARC-FCP-5699-34 found was specific to a 3x larger, non-live candidate, not this one" -- where
"this one" was the 9B live generator. **The live generator became that 3x-larger class on
2026-07-28** (gemma-4-31B-it). The stated reason for keeping 4096 expired with the model switch
and nothing moved the default. That is the changed-since-the-prior-failure this rerun needs.

THE PROMPT IS REBUILT, NOT INVENTED. `refactor_prompt()` reads only `vr.n`, `vr.n_correct`,
`vr.accuracy` and `vr.mismatches`, and all four are recorded verbatim in the banked ft09 `on`
cell's round-1 counterexample (`real_n=25`, `real_n_correct=6`, `real_accuracy=0.24`, 8
mismatches) -- the same replay-from-the-real-counterexample pattern REQ-34 used. The structural
reminder is left at its shipped default (ON), because that is what the live path runs today; the
REQ-34 confound was measuring with it accidentally OFF.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("SWEEP_OUT", "sweep_refactor"))
os.makedirs(OUT, exist_ok=True)

CELL = os.environ.get(
    "REFACTOR_CELL",
    os.path.join(os.path.dirname(HERE), "p4", "cells", "on__ft09__s1.json"),
)
PORT = int(os.environ.get("SWEEP_PORT", "8933"))
GPU = os.environ.get("SWEEP_GPU", "1")
BUDGETS = [int(x) for x in os.environ.get("SWEEP_BUDGETS", "4096,8192,16384").split(",")]
ATTEMPTS = int(os.environ.get("SWEEP_ATTEMPTS", "3"))
CALL_TIMEOUT_S = float(os.environ.get("SWEEP_CALL_TIMEOUT_S", "1800"))
REPO_SUBSTR = "gemma-4-31B-it"

os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = GPU
os.environ["CARNOT_ARC_INDUCE_N_CTX"] = "32768"
os.environ["CARNOT_ARC_FFN_CPU_LAYERS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CARNOT_ARC_GENERATOR_SEED", "3003")

sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, HERE)

_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{round(time.monotonic() - _T0, 1):>8}s] {msg}", flush=True)


def main() -> int:
    import urllib.request

    from budget_sweep import analyse_code, ramble_stats  # the validated analysers
    from carnot.agentic import arc_executable_world_model as e3

    assert e3.__file__.startswith(REPO), e3.__file__

    cell = json.load(open(CELL))
    ce = cell["result"]["induction_events"][0]["refinement_rounds"][0]["counterexample"]
    game = cell["anon_game_id"]  # the live path builds the prompt with the ANONYMIZED id
    vr = e3.VerifyResult(
        n=int(ce["real_n"]),
        n_correct=int(ce["real_n_correct"]),
        accuracy=float(ce["real_accuracy"]),
        mismatches=list(ce["real_mismatches"]),
    )
    prompt = (
        e3.refactor_prompt(game, vr)
        + "\n\nReturn ONLY the corrected ```python file.\n```python\n"
    )
    with open(os.path.join(OUT, "prompt_refactor.txt"), "w") as fh:
        fh.write(prompt)
    log(f"refactor prompt rebuilt: {len(prompt)} chars, game={game}, "
        f"vr={vr.n_correct}/{vr.n} acc={vr.accuracy}, "
        f"reminder_on={os.environ.get('CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER', '1') != '0'}")

    prop = e3.LocalGGUFProposer(
        repo_substr=REPO_SUBSTR, port=PORT, mtp=False, n_ctx=32768,
        ffn_cpu_layers=0, kv_quant="q8_0",
    )
    if not prop._ensure_server():
        json.dump({"status": "blocked_generator_server_not_started"},
                  open(os.path.join(OUT, "sweep.json"), "w"), indent=2)
        return 3

    witness = {
        "port": prop.port,
        "observed_model_path": prop.observed_model_path(),
        "observed_n_ctx": prop.observed_n_ctx(),
        "observed_total_slots": prop.observed_total_slots(),
        "gpu_requested": GPU,
    }
    pid_out = subprocess.run(["ss", "-lptnH", f"sport = :{PORT}"],
                             capture_output=True, text=True, timeout=5).stdout
    m = re.search(r"pid=(\d+)", pid_out)
    spid = int(m.group(1)) if m else None
    witness["server_pid"] = spid
    if spid is not None:
        witness["server_exe"] = os.path.realpath(f"/proc/{spid}/exe")
        witness["server_exe_is_cuda_build"] = "build-hip" not in witness["server_exe"]
    smi = subprocess.run(["nvidia-smi",
                          "--query-compute-apps=pid,used_memory,gpu_bus_id",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5).stdout
    witness["vram_rows_mine"] = [r.strip() for r in smi.splitlines()
                                 if spid is not None and r.strip().startswith(str(spid))]
    log(f"witness: {json.dumps(witness)}")
    if witness.get("server_exe_is_cuda_build") is not True or not witness["vram_rows_mine"]:
        json.dump({"status": "blocked_generator_unproven", "witness": witness},
                  open(os.path.join(OUT, "sweep.json"), "w"), indent=2)
        return 4

    required = ("engine", "is_level_complete")
    rows: list[dict] = []
    for budget in BUDGETS:
        for attempt in range(ATTEMPTS):
            # refactor() is codeonly_eligible=False, so NO code-only directive and NO stop
            # sequence -- it is a reasoning task and the directive would degrade exactly that.
            payload = {
                "prompt": prompt,
                "n_predict": budget,
                "temperature": 0.2 + 0.1 * attempt,
                "cache_prompt": True,
            }
            seed = e3.LocalGGUFProposer.sampling_seed(attempt)
            if seed is not None:
                payload["seed"] = seed
            t = time.monotonic()
            row: dict = {"prompt": "refactor", "budget": budget, "attempt": attempt,
                         "temperature": payload["temperature"], "seed": seed,
                         "required": list(required)}
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/completion",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
                    resp = json.load(r)
            except Exception as exc:
                row.update(status=f"http_error:{type(exc).__name__}", error=str(exc)[:300],
                           wall_s=round(time.monotonic() - t, 1))
                rows.append(row)
                log(f"  refactor b={budget} a={attempt} :: {row['status']}")
                continue
            text = resp.get("content", "")
            timings = resp.get("timings") or {}
            code = e3._extract_python(text)  # NOT codeonly: no raw-body fallback here
            row.update(
                status="ok", stop_type=resp.get("stop_type"),
                prompt_truncated=bool(resp.get("truncated")),
                predicted_n=timings.get("predicted_n"), prompt_n=timings.get("prompt_n"),
                predict_per_second=round(timings.get("predicted_per_second") or 0, 2),
                wall_s=round(time.monotonic() - t, 1), n_chars=len(text),
                **ramble_stats(text), **analyse_code(code or ""),
            )
            row["generate_would_accept"] = bool(
                code and all(f"def {fn}" in code for fn in required) and row["parses"])
            row["usable_engine"] = bool(
                row["generate_would_accept"] and row["engine_returns_on_all_paths"])
            fn = f"refactor_b{budget}_a{attempt}.txt"
            with open(os.path.join(OUT, fn), "w") as fh:
                fh.write(text)
            row["completion_file"] = fn
            rows.append(row)
            log(f"  refactor b={budget} a={attempt} :: stop={row['stop_type']} "
                f"pred_n={row['predicted_n']} ramble={row['ramble_frac']} "
                f"accept={row['generate_would_accept']} "
                f"returns={row['engine_returns_on_all_paths']} "
                f"USABLE={row['usable_engine']} wall={row['wall_s']}s")
            with open(os.path.join(OUT, "sweep.json"), "w") as fh:
                json.dump({"status": "partial", "witness": witness, "rows": rows},
                          fh, indent=2, sort_keys=True)

    with open(os.path.join(OUT, "sweep.json"), "w") as fh:
        json.dump({"status": "ok", "witness": witness, "budgets": BUDGETS,
                   "attempts": ATTEMPTS, "cell": CELL, "game": game,
                   "vr": {"n": vr.n, "n_correct": vr.n_correct, "accuracy": vr.accuracy,
                          "n_mismatches": len(vr.mismatches)},
                   "rows": rows, "wall_s": round(time.monotonic() - _T0, 1)},
                  fh, indent=2, sort_keys=True)
    log(f"DONE {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
