#!/usr/bin/env python3
"""PHASE 1 (confirm) -- does `repeat_penalty 1.1` + a PLAIN re-ask actually move the funnel?

THE TWO ARMS, and why exactly these two.

  CONTROL    the SHIPPED configuration: `generate()`'s own payload (no repetition control of any
             kind -- read from the server's /props on 2026-07-31: repeat_penalty 1.0,
             repeat_last_n 64, dry_multiplier 0.0, frequency_penalty 0.0, presence_penalty 0.0),
             round 0 only, and the shipped ACCEPT-FIRST bar (non-empty, defines `engine`, parses).
             The shipped path takes the first candidate clearing that bar, and Phase 2 measured it
             doing so 13 times in 15 while the code was mechanically broken.

  TREATMENT  `repeat_penalty 1.1` + `repeat_last_n 256`, and on a mechanically DEFECTIVE round 0,
             ONE plain re-ask -- the same neutral block Phase 2 ran as its control arm, which names
             NOTHING about what went wrong. The defect TEXT is deliberately excluded: Phase 2's
             paired sign test put repair-text against contentless-retry at p = 1.000 on 5
             discordant pairs, so the text is unwarranted and the contentless ask is cheaper.

TREATMENT ROUND 0 IS REPORTED AS ITS OWN COLUMN, at zero extra cost, because it is the
`repeat_penalty`-only sub-arm. Without it a compound arm that wins tells you nothing about WHICH
half won -- and the two halves have very different costs to wire in.

WHAT THIS RUN EXISTS TO RETIRE. Phase 1's `cell_recall 0.947` for repeat_penalty was n=3, ONE
game, and IN-SAMPLE -- all six changing transitions it was graded on were in the prompt. This run
is 6 games and grades OUT-OF-SAMPLE, on the split derived and PROVEN by `split.py` (rows whose
delta line does not occur in the prompt text). Scoring is a separate offline step; this file only
generates and records.

LEAK DISCIPLINE. The treatment's own accept/reject decision -- the dry run, the defect check, the
non-inert check -- sees ONLY the transitions the prompt showed. A pipeline that consulted held-out
rows to decide whether to re-ask would be training on its own test set, and the held-out score
would stop meaning anything. The held-out rows are opened for the first time by `score.py`.

ORDERING IS ATTEMPT-MAJOR ACROSS GAMES, deliberately. An earlier probe in this repo died at 3.2h
with 4 of 12 cells never launched, which left an unbalanced grid that could not be reported. Here
the outer loop is (seed_base, attempt) and the inner loop is the games on this GPU, so a
wall-clock cutoff at any point leaves every game with the SAME number of attempts and the run is
reportable as-is. Partial JSON is rewritten after every completed CELL (one game's control +
treatment round 0 + any re-ask), which is the smallest unit that is meaningful to score -- a
half-written cell would give the scorer a control with no treatment to pair it against.

BOTH ARMS SHARE ONE SERVER PROCESS. The sampler seed does not reach across server instances --
identical config on a second server gives different output, while within one process it holds
byte-exactly across a 4x budget range. Arms split over two processes would be confounded by
sampler variance alone, which this repo has already paid for once.

PINS, each paid for in an earlier incident: n_ctx=32768 (81920 does not fit a 24 GiB card and the
31B then silently binds the iGPU HIP build and runs LLM-OFF while REPORTING LLM-ON),
ffn_cpu_layers=0, MTP off, `CUDA_VISIBLE_DEVICES=""` on this parent, a NON-DEFAULT port (8919 is
the default and a stale server there is silently adopted), and the CUDA build PROVEN from
`/proc/<pid>/exe` plus a per-PID VRAM row rather than inferred from what was requested.
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

GAMES = [g for g in os.environ.get("CONFIRM_GAMES", "ft09,tn36,vc33").split(",") if g]
PORT = int(os.environ.get("CONFIRM_PORT", "8951"))
GPU = os.environ.get("CONFIRM_GPU", "0")
CALL_INDEX = int(os.environ.get("CONFIRM_CALL_INDEX", "2"))
SEED_BASES = [int(x) for x in os.environ.get("CONFIRM_SEED_BASES", "3003,3004").split(",")]
ATTEMPTS = int(os.environ.get("CONFIRM_ATTEMPTS", "3"))
BUDGET = int(os.environ.get("CONFIRM_BUDGET", "4096"))
CALL_TIMEOUT_S = float(os.environ.get("CONFIRM_CALL_TIMEOUT_S", "1800"))
DEADLINE_S = float(os.environ.get("CONFIRM_DEADLINE_S", "12600"))  # 3.5h hard stop
REPO_SUBSTR = "gemma-4-31B-it"
TAG = os.environ.get("CONFIRM_TAG", f"gpu{GPU}")

# The SHIPPED sampler sends no repetition control at all. The treatment adds exactly two fields.
CONTROL_SAMPLER: dict = {}
TREATMENT_SAMPLER: dict = {"repeat_penalty": 1.1, "repeat_last_n": 256}

OUT = os.path.join(HERE, "confirm", TAG)
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


# The instruction line the induce prompt ends with. A follow-up block is inserted BEFORE it so
# the output instruction stays last, which is where the model is trained to look for it.
_TAIL_MARKER = "Return ONLY one ```python code block"

# VERBATIM from `results/arc_engine_validation_20260731/harness/repair_ab.py`'s arm C. Reused
# unchanged so this run measures the SAME intervention that arm measured, at larger n and on a
# held-out split, rather than a differently-worded thing that would not be comparable.
PLAIN_REASK_BLOCK = """
YOUR PREVIOUS ANSWER WAS RUN AGAINST THE OBSERVED TRANSITIONS AND WAS NOT SATISFACTORY.
Please try again from the same evidence:

  * re-read the observed transitions above before writing anything
  * take the simplest rule that is consistent with all of them
  * prefer a general rule over a table of the specific cases you were shown

Write the function again. `engine(grid, action, data)` must return a numpy array of the SAME
SHAPE as `grid` on EVERY path, and must not raise on any observed transition.
"""


def _insert_block(prompt: str, block: str) -> str:
    idx = prompt.rfind(_TAIL_MARKER)
    if idx < 0:  # no marker: append before the trailing fence opener
        return prompt.rstrip().rsplit("```python", 1)[0] + block + "\n```python\n"
    return prompt[:idx] + block + "\n" + prompt[idx:]


def main() -> int:  # noqa: C901
    import urllib.request

    from carnot.agentic import arc_engine_static_validation as sv
    from carnot.agentic import arc_executable_world_model as e3
    from split import load_split

    for mod in (e3, sv):
        assert mod.__file__.startswith(REPO), f"wrong repo code imported: {mod.__file__}"

    # Prompts + the PROVEN split, loaded before the server so a missing capture costs no GPU.
    cells: dict = {}
    for game in GAMES:
        prompt_path = os.path.join(HERE, "capture", game, f"prompt{CALL_INDEX}_engine.txt")
        if not os.path.exists(prompt_path):
            log(f"SKIP {game}: no capture at {prompt_path}")
            continue
        s = load_split(game, CALL_INDEX)
        if not s["split_proven"]:
            log(f"SKIP {game}: split not proven -> {s['checks']}")
            continue
        cells[game] = {
            "prompt": open(prompt_path).read(),
            # ONLY the shown rows reach the pipeline's own decisions. See LEAK DISCIPLINE.
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
            open(os.path.join(OUT, "confirm.json"), "w"),
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
            open(os.path.join(OUT, "confirm.json"), "w"),
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
                open(os.path.join(OUT, "confirm.json"), "w"),
                indent=2,
            )
            prop.stop()
            return 4

    rows: list[dict] = []
    attempts: list[dict] = []

    def _live_server_pid() -> int | None:
        """Re-read the PID currently listening on PORT, at CALL time.

        RETROFIT 2026-07-31 (adversarial review). This run captured `server_pid` ONCE per GPU
        worker, in the witness block, and per-attempt rows carried the seed but no PID. That is
        not sufficient, and the Phase 3 pre-flight later PROVED the failure mode is real: a
        server WAS replaced mid-session there (1532116 -> 1562595), and Phase 3's own
        `same_server_process` guard excluded a game rather than score a confounded pair.

        This matters specifically because the sampler seed does NOT reach across server
        instances -- identical config on a second server gives different output, while within
        one process it holds byte-exactly. So an arm pair that straddles a server restart is
        confounded by sampler variance alone, which is exactly the comparison this harness
        exists to avoid. A once-per-worker witness cannot detect that; a per-call read can.

        An equivalent replacement during THIS run would have been silent, so the pairs already
        recorded are unverified in this respect. They are not retro-fixable -- the PIDs were
        never written down -- and the honest statement is that the shared-process property was
        ASSERTED for this run's pairs and is MEASURED for any future one.
        """

        try:
            out = subprocess.run(
                ["ss", "-lptnH", f"sport = :{PORT}"], capture_output=True, text=True, timeout=5
            ).stdout
        except Exception:  # noqa: BLE001
            return None
        found = re.search(r"pid=(\d+)", out)
        return int(found.group(1)) if found else None

    def call(prompt, *, seed, temperature, sampler, game, tag) -> dict:
        payload = {
            "prompt": prompt,
            "n_predict": BUDGET,
            "temperature": temperature,
            "cache_prompt": True,
            "stop": ["```"],  # codeonly_eligible=True on every induce call
            **sampler,
        }
        if seed is not None:
            payload["seed"] = seed
        # Read BEFORE the request, so a row records the process that actually served it.
        _call_pid = _live_server_pid()
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
                "tag": tag,
                "status": f"http_error:{type(exc).__name__}",
                "error": str(exc)[:300],
                "wall_s": round(time.monotonic() - t, 1),
                "server_pid": _call_pid,
            }
        text = resp.get("content", "")
        timings = resp.get("timings") or {}
        code = e3._extract_python(text) or text.strip()
        shown = cells[game]["shown"]
        defects = sv.validate_engine_code(
            code,
            transitions=shown,  # SHOWN ONLY -- no held-out row informs this decision
            stop_type=resp.get("stop_type"),
            required=("engine",),
            budget=BUDGET,
        )
        try:
            import ast as _ast

            _ast.parse(code)
            parses = True
        except SyntaxError:
            parses = False
        accepted = bool(code and "def engine" in code and parses)
        changes = sv.engine_changes_anything(code, shown)
        fn = f"{game}_{tag}.txt"
        with open(os.path.join(OUT, fn), "w") as fh:
            fh.write(text)
        row = {
            "game": game,
            "tag": tag,
            "status": "ok",
            "completion_file": fn,
            "sampler": dict(sampler),
            "n_predict_requested": BUDGET,
            "stop_type": resp.get("stop_type"),
            "predicted_n": timings.get("predicted_n"),
            "prompt_n": timings.get("prompt_n"),
            "prompt_truncated": bool(resp.get("truncated")),
            "wall_s": round(time.monotonic() - t, 1),
            "server_pid": _call_pid,
            "prompt_chars": len(prompt),
            "generate_would_accept": accepted,
            "defect_kinds": sorted({d.kind for d in defects}),
            "defect_details": [d.detail[:240] for d in defects],
            "engine_changes_anything": changes,
            # THE BAR: mechanically clean AND not inert. An identity engine fails this.
            "usable": bool(accepted and not defects and changes is True),
        }
        log(
            f"  {game} {tag}: stop={row['stop_type']} pred_n={row['predicted_n']} "
            f"accept={accepted} defects={row['defect_kinds']} changes={changes} "
            f"USABLE={row['usable']} wall={row['wall_s']}s"
        )
        return row

    def flush(status: str) -> None:
        payload = {
            "status": status,
            "tag": TAG,
            "games": list(cells),
            "call_index": CALL_INDEX,
            "budget": BUDGET,
            "seed_bases": SEED_BASES,
            "attempts_per_base": ATTEMPTS,
            "control_sampler": CONTROL_SAMPLER,
            "treatment_sampler": TREATMENT_SAMPLER,
            "reask_block_chars": len(PLAIN_REASK_BLOCK),
            "witness": witness,
            "splits": {g: cells[g]["split"] for g in cells},
            "attempts": attempts,
            "rows": rows,
            "wall_s": round(time.monotonic() - _T0, 1),
        }
        with open(os.path.join(OUT, "confirm.json"), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    stopped = None
    # ATTEMPT-MAJOR: every game gets attempt n before any game gets attempt n+1.
    for base in SEED_BASES:
        for attempt in range(ATTEMPTS):
            for game in list(cells):
                if time.monotonic() - _T0 > DEADLINE_S:
                    stopped = "deadline_reached"
                    break
                seed = base * 1000 + attempt
                temp = round(0.2 + 0.1 * attempt, 2)
                pfx = f"b{base}_a{attempt}"
                rec: dict = {
                    "game": game,
                    "seed_base": base,
                    "attempt": attempt,
                    "seed": seed,
                    "temperature": temp,
                }

                ctl = call(
                    cells[game]["prompt"],
                    seed=seed,
                    temperature=temp,
                    sampler=CONTROL_SAMPLER,
                    game=game,
                    tag=f"{pfx}_control",
                )
                rows.append(ctl)
                rec["CONTROL"] = ctl

                tr0 = call(
                    cells[game]["prompt"],
                    seed=seed,
                    temperature=temp,
                    sampler=TREATMENT_SAMPLER,
                    game=game,
                    tag=f"{pfx}_t_round0",
                )
                rows.append(tr0)
                rec["TREATMENT_round0"] = tr0

                # The re-ask fires on a MECHANICAL DEFECT -- exactly where the shipped path
                # accepts instead. A clean round 0 is already the treatment's answer.
                if tr0.get("status") == "ok" and tr0["defect_kinds"]:
                    tr1 = call(
                        _insert_block(cells[game]["prompt"], PLAIN_REASK_BLOCK),
                        seed=seed,
                        temperature=temp,
                        sampler=TREATMENT_SAMPLER,
                        game=game,
                        tag=f"{pfx}_t_reask",
                    )
                    rows.append(tr1)
                    rec["TREATMENT_reask"] = tr1
                    rec["TREATMENT_final"] = tr1
                    rec["reask_fired"] = True
                else:
                    rec["TREATMENT_final"] = tr0
                    rec["reask_fired"] = False

                attempts.append(rec)
                flush("partial")
            if stopped:
                break
        if stopped:
            break

    flush(stopped or "ok")
    log(f"DONE rows={len(rows)} attempts={len(attempts)} stopped={stopped}")
    prop.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
