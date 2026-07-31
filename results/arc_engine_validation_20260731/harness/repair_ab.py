#!/usr/bin/env python3
"""PHASE 2, STEP 4b -- does TELLING the model what broke produce a usable engine? THREE arms.

THE QUESTION. Phase 2's static + dry-run checks are proven to CATCH the four broken-code
failures. That is a diagnosis, not a fix. The funnel only moves if a caught defect can be turned
into a working engine, so the question is whether feeding the measured failure back -- "your
`engine` returned None on action 6", "`is_level_complete` raised UnboundLocalError on 'cell'" --
produces a returning, non-inert engine.

WHY THREE ARMS AND NOT TWO. A two-arm design (shipped vs repair) cannot distinguish "the repair
TEXT works" from "ANY second ask works": the repair arm changes the prompt, and a changed prompt
resamples. The third arm is the control that separates them.

  A  SHIPPED   round 0 only -- exactly what `generate()` does today.
  B  REPAIR    round 0, then re-ask with `repair_prompt_block` (names the defect + the exception
               text + the failing code).
  C  CONTROL   round 0, then re-ask with a NEUTRAL block of comparable length that says a
               previous attempt was unsatisfactory but names NOTHING about what went wrong.

B and C share arm A's round 0 byte-for-byte -- it is generated ONCE and both branch from it -- so
the only difference between them is the content of the second ask. Both second asks reuse round
0's seed and temperature, so sampler noise is not a confound either. If B and C perform the same,
the repair CONTENT is worth nothing and only the retry is; that is a real possible outcome and
this harness is built so it would be visible rather than hidden.

THE ACCEPTANCE BAR IS NOT "returns on all paths". Phase 1 found that the completions scoring best
on every structural check were the IDENTITY FUNCTION, which clears the return check trivially.
`usable` here therefore requires BOTH: no mechanical defect AND the engine actually changes the
grid on some observed transition. An inert engine is counted as a failure, in every arm.

PINS, each paid for in an earlier incident: n_ctx=32768 (81920 does not fit a 24 GiB card and the
31B then silently binds the iGPU HIP build and runs LLM-OFF while REPORTING LLM-ON),
ffn_cpu_layers=0, MTP off, `CUDA_VISIBLE_DEVICES=""` on this parent, a NON-DEFAULT port (8919 is
the default and a stale server there is silently adopted), and the CUDA build PROVEN from
`/proc/<pid>/exe` plus a per-PID VRAM row rather than inferred from what was requested.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = os.path.dirname(os.path.abspath(__file__))

GAME = os.environ.get("AB_GAME", "ft09")
PORT = int(os.environ.get("AB_PORT", "8941"))
GPU = os.environ.get("AB_GPU", "0")
CALL_INDEX = int(os.environ.get("AB_CALL_INDEX", "2"))
ATTEMPTS = int(os.environ.get("AB_ATTEMPTS", "3"))
BUDGET = int(os.environ.get("AB_BUDGET", "4096"))
# Escalated budget used ONLY for a `retryable` (truncation) defect, per REQ-ARC-WMTE-6052
# SCENARIO-4: a truncated completion is a missing observation, so it is re-asked with more room
# rather than repaired. Phase 1 measured that more room does not by itself help on ft09; this is
# here so the retryable path is EXERCISED and reported, not because it is expected to rescue.
RETRY_BUDGET = int(os.environ.get("AB_RETRY_BUDGET", "8192"))
CALL_TIMEOUT_S = float(os.environ.get("AB_CALL_TIMEOUT_S", "1800"))
REPO_SUBSTR = "gemma-4-31B-it"

CAP = os.path.join(HERE, "capture", GAME)
OUT = os.path.join(HERE, "ab", GAME)
os.makedirs(OUT, exist_ok=True)

os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = GPU
os.environ["CARNOT_ARC_INDUCE_N_CTX"] = "32768"
os.environ["CARNOT_ARC_FFN_CPU_LAYERS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("CARNOT_ARC_GENERATOR_SEED", "3003")
os.environ["CARNOT_ARC_E3_DIR"] = os.path.join(OUT, "e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)

sys.path.insert(0, os.path.join(REPO, "python"))

_T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{GAME} {round(time.monotonic() - _T0, 1):>7}s] {msg}", flush=True)


# The instruction line the induce prompt ends with. A follow-up block is inserted BEFORE it so
# the output instruction stays last, which is where the model is trained to look for it.
_TAIL_MARKER = "Return ONLY one ```python code block"

# The control block. Written to be the same SHAPE as a repair block -- a header, bullets, a
# closing instruction -- and deliberately CONTENTLESS about the defect. Its length is checked
# against the repair block's at runtime and recorded, so "the control was much shorter" can
# never be an unexamined explanation of a difference.
CONTROL_BLOCK = """
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


def main() -> int:
    import urllib.request

    from carnot.agentic import arc_engine_static_validation as sv
    from carnot.agentic import arc_executable_world_model as e3

    for mod in (e3, sv):
        assert mod.__file__.startswith(REPO), f"wrong repo code imported: {mod.__file__}"

    prompt_path = os.path.join(CAP, f"prompt{CALL_INDEX}_engine.txt")
    trans_path = os.path.join(CAP, f"transitions{CALL_INDEX}.pkl")
    if not (os.path.exists(prompt_path) and os.path.exists(trans_path)):
        json.dump(
            {"status": "blocked_capture_missing", "game": GAME, "call_index": CALL_INDEX},
            open(os.path.join(OUT, "ab.json"), "w"),
            indent=2,
        )
        return 2
    base_prompt = open(prompt_path).read()
    with open(trans_path, "rb") as fh:
        trans = pickle.load(fh)
    required = ("engine",)

    prop = e3.LocalGGUFProposer(
        repo_substr=REPO_SUBSTR, port=PORT, mtp=False, n_ctx=32768,
        ffn_cpu_layers=0, kv_quant="q8_0",
    )
    log("starting server ...")
    if not prop._ensure_server():
        json.dump(
            {"status": "blocked_generator_server_not_started",
             "selection_log": list(e3.GENERATOR_SELECTION_LOG)[-25:]},
            open(os.path.join(OUT, "ab.json"), "w"), indent=2)
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
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_bus_id",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10).stdout
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
                      open(os.path.join(OUT, "ab.json"), "w"), indent=2)
            return 4

    def call(prompt: str, *, seed, temperature: float, n_predict: int, tag: str) -> dict:
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "cache_prompt": True,
            "stop": ["```"],  # codeonly_eligible=True on every induce call
        }
        if seed is not None:
            payload["seed"] = seed
        t = time.monotonic()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/completion",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
                resp = json.load(r)
        except Exception as exc:
            return {"tag": tag, "status": f"http_error:{type(exc).__name__}",
                    "error": str(exc)[:300], "wall_s": round(time.monotonic() - t, 1)}
        text = resp.get("content", "")
        timings = resp.get("timings") or {}
        code = e3._extract_python(text) or text.strip()
        defects = sv.validate_engine_code(
            code, transitions=trans, stop_type=resp.get("stop_type"),
            required=required, budget=n_predict,
        )
        # `generate()`'s own acceptance, reproduced: non-empty, every required def, parses.
        try:
            import ast as _ast

            _ast.parse(code)
            parses = True
        except SyntaxError:
            parses = False
        accepted = bool(code and all(f"def {fn}" in code for fn in required) and parses)
        changes = sv.engine_changes_anything(code, trans)
        fn = f"{tag}.txt"
        with open(os.path.join(OUT, fn), "w") as fh:
            fh.write(text)
        row = {
            "tag": tag, "status": "ok", "completion_file": fn,
            "n_predict_requested": n_predict,
            "stop_type": resp.get("stop_type"),
            "predicted_n": timings.get("predicted_n"),
            "prompt_n": timings.get("prompt_n"),
            "prompt_truncated": bool(resp.get("truncated")),
            "wall_s": round(time.monotonic() - t, 1),
            "prompt_chars": len(prompt),
            "generate_would_accept": accepted,
            "defect_kinds": sorted({d.kind for d in defects}),
            "defect_details": [d.detail[:240] for d in defects],
            "any_repairable": any(d.repairable for d in defects),
            "any_retryable": any(d.retryable for d in defects),
            "engine_changes_anything": changes,
            # THE BAR: mechanically clean AND not inert. An identity engine fails this.
            "usable": bool(accepted and not defects and changes is True),
        }
        log(
            f"  {tag}: stop={row['stop_type']} pred_n={row['predicted_n']} "
            f"accept={accepted} defects={row['defect_kinds']} changes={changes} "
            f"USABLE={row['usable']} wall={row['wall_s']}s"
        )
        return row

    rows: list[dict] = []
    attempts: list[dict] = []
    for attempt in range(ATTEMPTS):
        seed = e3.LocalGGUFProposer.sampling_seed(attempt)
        temp = 0.2 + 0.1 * attempt
        r0 = call(base_prompt, seed=seed, temperature=temp, n_predict=BUDGET,
                  tag=f"a{attempt}_round0")
        rows.append(r0)
        rec: dict = {"attempt": attempt, "seed": seed, "temperature": round(temp, 2),
                     "A_shipped": r0}
        if r0.get("status") != "ok":
            attempts.append(rec)
            continue
        if not r0["defect_kinds"]:
            # Nothing to repair. Both follow-up arms are undefined here, and saying so is more
            # honest than re-asking anyway and pretending the comparison means something.
            rec["followup"] = "skipped_round0_had_no_defects"
            attempts.append(rec)
            continue

        # Reconstruct the defects to build the repair block (the row only carries strings).
        code0 = e3._extract_python(open(os.path.join(OUT, r0["completion_file"])).read())
        code0 = code0 or open(os.path.join(OUT, r0["completion_file"])).read().strip()
        d0 = sv.validate_engine_code(
            code0, transitions=trans, stop_type=r0["stop_type"],
            required=required, budget=BUDGET,
        )
        rec["round0_defects"] = sorted({d.kind for d in d0})

        if any(d.retryable for d in d0) and not any(d.repairable for d in d0):
            # TRUNCATION path: re-ask with more room, same prompt. Both B and C reduce to the
            # same action here (there is no defect content to feed back), so it is reported ONCE
            # as its own arm rather than double-counted as a repair win.
            rB = call(base_prompt, seed=seed, temperature=temp, n_predict=RETRY_BUDGET,
                      tag=f"a{attempt}_retry")
            rows.append(rB)
            rec["B_repair"] = rB
            rec["followup"] = "retry_more_budget"
            attempts.append(rec)
            continue

        repair_block = sv.repair_prompt_block(d0, code=code0)
        rB = call(_insert_block(base_prompt, repair_block), seed=seed, temperature=temp,
                  n_predict=BUDGET, tag=f"a{attempt}_repair")
        rC = call(_insert_block(base_prompt, CONTROL_BLOCK), seed=seed, temperature=temp,
                  n_predict=BUDGET, tag=f"a{attempt}_control")
        rows.append(rB)
        rows.append(rC)
        rec["B_repair"] = rB
        rec["C_control"] = rC
        rec["followup"] = "repair_vs_control"
        rec["repair_block_chars"] = len(repair_block)
        rec["control_block_chars"] = len(CONTROL_BLOCK)
        attempts.append(rec)

        with open(os.path.join(OUT, "ab.json"), "w") as fh:
            json.dump({"status": "partial", "game": GAME, "witness": witness,
                       "attempts": attempts, "rows": rows}, fh, indent=2, sort_keys=True)

    def _n_usable(key: str) -> int:
        return sum(1 for a in attempts if isinstance(a.get(key), dict) and a[key].get("usable"))

    summary = {
        "n_attempts": len(attempts),
        "A_shipped_usable": _n_usable("A_shipped"),
        "B_repair_usable": _n_usable("B_repair"),
        "C_control_usable": _n_usable("C_control"),
        "A_shipped_accepted": sum(
            1 for a in attempts
            if isinstance(a.get("A_shipped"), dict) and a["A_shipped"].get("generate_would_accept")
        ),
        "A_shipped_defective": sum(
            1 for a in attempts
            if isinstance(a.get("A_shipped"), dict) and a["A_shipped"].get("defect_kinds")
        ),
        "followups": [a.get("followup") for a in attempts],
    }
    payload = {
        "status": "ok",
        "game": GAME,
        "call_index": CALL_INDEX,
        "budget": BUDGET,
        "retry_budget": RETRY_BUDGET,
        "n_transitions": len(trans),
        "prompt_sha256_16": __import__("hashlib").sha256(base_prompt.encode()).hexdigest()[:16],
        "witness": witness,
        "summary": summary,
        "attempts": attempts,
        "rows": rows,
        "wall_s": round(time.monotonic() - _T0, 1),
    }
    with open(os.path.join(OUT, "ab.json"), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    log(f"SUMMARY {json.dumps(summary)}")
    prop.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
