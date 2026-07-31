#!/usr/bin/env python3
"""PHASE 1, control arm -- is the comment wall a BUDGET problem or a SAMPLER problem?

WHY THIS ARM EXISTS. Reading the 4096 completions shows the failure is not "the model ran out
of room mid-thought". It reaches a genuine impasse on ft09's progress-bar mechanic and then
emits the SAME comment line verbatim, dozens of times, until the budget is gone:

    # Let's just find the first pair of 2 cells from the right that are not 11.
    # Let's just find the first pair of 2 cells from the right that are not 11.
    ... x N

That is textbook decode-level degeneration, and the server this path talks to has EVERY
repetition control switched off -- read from its own /props, not assumed:

    repeat_penalty 1.0   repeat_last_n 64   dry_multiplier 0.0
    frequency_penalty 0.0   presence_penalty 0.0

`LocalGGUFProposer.generate()` sends only {prompt, n_predict, temperature, cache_prompt,
seed, stop}, so those defaults are what every induce call in the scored path runs under. If a
repetition penalty converts the wall into finished code, then "raise max_tokens" was always
the wrong lever -- more budget just buys a longer loop -- and Phase 2's repair layer should be
built on top of a generator that is not looping in the first place.

THIS ARM ONLY MEASURES. Changing the scored agent's sampler is a behaviour change, not a
measurement change, and this file does not make it (the same line `sampling_seed()`'s docstring
draws about seeding). The output is evidence for the operator, nothing more.
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
OUT = os.path.join(HERE, os.environ.get("SWEEP_OUT", "sweep_sampler"))
os.makedirs(OUT, exist_ok=True)

PORT = int(os.environ.get("SWEEP_PORT", "8933"))
GPU = os.environ.get("SWEEP_GPU", "1")
PROMPT_NAME = os.environ.get("SWEEP_PROMPT", "engine")
BUDGETS = [int(x) for x in os.environ.get("SWEEP_BUDGETS", "4096").split(",")]
ATTEMPTS = int(os.environ.get("SWEEP_ATTEMPTS", "3"))
CALL_TIMEOUT_S = float(os.environ.get("SWEEP_CALL_TIMEOUT_S", "1800"))
REPO_SUBSTR = "gemma-4-31B-it"

# The sampler arms. `off` is the SHIPPED configuration and is re-run here rather than reused
# from the budget sweep, so the comparison is within one script, one server instance and one
# code path -- the repo has already paid once for a phantom effect that came from comparing
# two arms served from different places.
ARMS = {
    "off": {},
    "repeat_penalty_1.1": {"repeat_penalty": 1.1, "repeat_last_n": 256},
    "dry_0.8": {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2},
}

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

    from budget_sweep import analyse_code, ramble_stats
    from carnot.agentic import arc_executable_world_model as e3

    prompt = open(os.path.join(HERE, "prompts", f"prompt2_{PROMPT_NAME}.txt")).read()
    required = ("engine",) if PROMPT_NAME == "engine" else ("engine", "is_level_complete")

    prop = e3.LocalGGUFProposer(repo_substr=REPO_SUBSTR, port=PORT, mtp=False,
                                n_ctx=32768, ffn_cpu_layers=0, kv_quant="q8_0")
    if not prop._ensure_server():
        json.dump({"status": "blocked_generator_server_not_started"},
                  open(os.path.join(OUT, "sweep.json"), "w"), indent=2)
        return 3
    pid_out = subprocess.run(["ss", "-lptnH", f"sport = :{PORT}"],
                             capture_output=True, text=True, timeout=5).stdout
    m = re.search(r"pid=(\d+)", pid_out)
    spid = int(m.group(1)) if m else None
    witness = {"port": prop.port, "server_pid": spid,
               "observed_model_path": prop.observed_model_path(),
               "observed_n_ctx": prop.observed_n_ctx(), "gpu_requested": GPU}
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

    rows: list[dict] = []
    for arm, extra in ARMS.items():
        for budget in BUDGETS:
            for attempt in range(ATTEMPTS):
                payload = {"prompt": prompt, "n_predict": budget,
                           "temperature": 0.2 + 0.1 * attempt, "cache_prompt": True,
                           "stop": ["```"], **extra}
                seed = e3.LocalGGUFProposer.sampling_seed(attempt)
                if seed is not None:
                    payload["seed"] = seed
                t = time.monotonic()
                row: dict = {"arm": arm, "prompt": PROMPT_NAME, "budget": budget,
                             "attempt": attempt, "temperature": payload["temperature"],
                             "seed": seed, "sampler_extra": dict(extra)}
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{PORT}/completion",
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
                        resp = json.load(r)
                except Exception as exc:
                    row.update(status=f"http_error:{type(exc).__name__}",
                               error=str(exc)[:300], wall_s=round(time.monotonic() - t, 1))
                    rows.append(row)
                    log(f"  {arm} b={budget} a={attempt} :: {row['status']}")
                    continue
                text = resp.get("content", "")
                timings = resp.get("timings") or {}
                code = e3._extract_python(text) or text.strip()
                row.update(status="ok", stop_type=resp.get("stop_type"),
                           predicted_n=timings.get("predicted_n"),
                           wall_s=round(time.monotonic() - t, 1), n_chars=len(text),
                           **ramble_stats(text), **analyse_code(code))
                row["generate_would_accept"] = bool(
                    code and all(f"def {fn}" in code for fn in required) and row["parses"])
                row["usable_engine"] = bool(
                    row["generate_would_accept"] and row["engine_returns_on_all_paths"])
                fn = f"{arm}_{PROMPT_NAME}_b{budget}_a{attempt}.txt"
                with open(os.path.join(OUT, fn), "w") as fh:
                    fh.write(text)
                row["completion_file"] = fn
                rows.append(row)
                log(f"  {arm} b={budget} a={attempt} :: stop={row['stop_type']} "
                    f"pred_n={row['predicted_n']} ramble={row['ramble_frac']} "
                    f"code_ln={row['code_lines']} accept={row['generate_would_accept']} "
                    f"returns={row['engine_returns_on_all_paths']} "
                    f"USABLE={row['usable_engine']} wall={row['wall_s']}s")
                with open(os.path.join(OUT, "sweep.json"), "w") as fh:
                    json.dump({"status": "partial", "witness": witness, "rows": rows},
                              fh, indent=2, sort_keys=True)

    with open(os.path.join(OUT, "sweep.json"), "w") as fh:
        json.dump({"status": "ok", "witness": witness, "arms": ARMS, "budgets": BUDGETS,
                   "attempts": ATTEMPTS, "prompt": PROMPT_NAME, "rows": rows,
                   "wall_s": round(time.monotonic() - _T0, 1)}, fh, indent=2, sort_keys=True)
    log(f"DONE {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
