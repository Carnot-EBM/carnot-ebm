#!/usr/bin/env python3
"""PHASE 2 driver -- run `worker.py` once per Phase-1 candidate and collect the raw records.

INPUTS ARE PHASE 1's, UNCHANGED. The 48 candidate completions in
`results/arc_induce_bestofn_20260731/harness/bon/gpu1/*.txt`, the captured transition tapes in
`.../harness/capture/<game>/`, and the PROVEN held-out split recomputed by importing Phase 1's
own `split.load_split`. Nothing is re-generated and no GPU is touched: every number this phase
reports is derived from artefacts that already exist on disk, so the whole phase is
reproducible offline and costs no inference.

WHY THE SPLIT IS RECOMPUTED RATHER THAN READ. Phase 1 wrote the split's SUMMARY to `split.json`
but `.gitignore`d the row pickles (`_shown_*.pkl`). Re-deriving them through the same
`load_split` that produced the summary keeps this phase on the identical, already-proven
partition -- the driver asserts the recomputed per-game counts match the recorded summary, so a
silent drift in the split cannot pass unnoticed.

EVERY WORKER IS HARD-TIMED OUT. `timeout=` on the subprocess is the only protection against a
non-terminating induced engine, which Phase 1 demonstrated is a real and frequent failure mode.
A worker killed by the timeout is recorded as `timeout` -- a MISSING OBSERVATION -- and never
folded into a rate as if it were a negative result.
"""

from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import pathlib
import pickle
import subprocess
import sys
import time

REPO = pathlib.Path("/home/ianblenke/github.com/ianblenke/carnot")
P1 = REPO / "results" / "arc_induce_bestofn_20260731" / "harness"
HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = pathlib.Path("/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot") / (
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/phase2"
)
SCRATCH.mkdir(parents=True, exist_ok=True)
PY = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"
CALL_INDEX = 1  # the induce call that carries all three gates (established in Phase 1)
WORKER_TIMEOUT_S = 240
N_PARALLEL = 6


def _load_split_module():
    spec = importlib.util.spec_from_file_location("p1_split", P1 / "split.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p1_split"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    split_mod = _load_split_module()
    bon = json.loads((P1 / "bon" / "gpu1" / "bon.json").read_text())
    recorded = {r["game"]: r for r in json.loads((HERE.parent.parent / "arc_induce_bestofn_20260731" / "split.json").read_text())["rows"]}

    # ---- rebuild the proven split, and assert it matches what Phase 1 recorded --------------
    splits: dict[str, dict] = {}
    for game in sorted({r["game"] for r in bon["rows"]}):
        s = split_mod.load_split(game, CALL_INDEX)
        rec = recorded[game]
        for field in ("n_full", "n_shown", "n_heldout", "heldout_n_changing", "n_prefix"):
            if int(s[field]) != int(rec[field]):
                raise SystemExit(
                    f"SPLIT DRIFT on {game}.{field}: recomputed {s[field]} != recorded {rec[field]}"
                )
        splits[game] = s
        hp = SCRATCH / f"{game}_heldout.pkl"
        with open(hp, "wb") as fh:
            pickle.dump(s["_heldout"], fh)
        # The FULL tape is what the first-visit / dedup analysis needs; carry it forward too.
        with open(SCRATCH / f"{game}_full.pkl", "wb") as fh:
            pickle.dump(s["_full"], fh)
    print(f"split verified against Phase 1 for {len(splits)} games", flush=True)

    # ---- one job per candidate --------------------------------------------------------------
    sys.path.insert(0, str(REPO / "python"))
    from carnot.agentic import arc_executable_world_model as e3

    jobs = []
    for r in bon["rows"]:
        game, k = r["game"], r["candidate"]
        text = (P1 / "bon" / "gpu1" / r["completion_file"]).read_text()
        # The SAME extractor `generate()` uses, so the code scored here is byte-identical to
        # the code Phase 1 scored and to what the live pipeline would have executed.
        code = e3._extract_python(text) or text.strip()
        cp = SCRATCH / f"{game}_k{k}.py"
        cp.write_text(code)
        job = {
            "code_path": str(cp),
            "heldout_pkl": str(SCRATCH / f"{game}_heldout.pkl"),
            "full_pkl": str(SCRATCH / f"{game}_full.pkl"),
            "root_pkl": str(P1 / "capture" / game / f"root_grid{CALL_INDEX}.pkl"),
        }
        jp = SCRATCH / f"{game}_k{k}.job.json"
        jp.write_text(json.dumps(job))
        jobs.append((game, k, r, str(jp)))

    def _run(item):
        game, k, r, jp = item
        t = time.monotonic()
        try:
            p = subprocess.run(  # noqa: S603
                [PY, str(HERE / "worker.py"), jp],
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_S,
                cwd=str(REPO),
            )
            res = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {
                "status": "no_output",
                "stderr": p.stderr[-400:],
            }
        except subprocess.TimeoutExpired:
            res = {"status": "timeout"}
        except Exception as exc:  # noqa: BLE001
            res = {"status": f"driver_error:{type(exc).__name__}", "error": str(exc)[:200]}
        res.update(
            {
                "game": game,
                "candidate": k,
                "seed": r["seed"],
                "code_sha256_16": r["code_sha256_16"],
                "usable": r["usable"],
                "generate_would_accept": r["generate_would_accept"],
                "driver_wall_s": round(time.monotonic() - t, 2),
            }
        )
        print(
            f"  {game} k{k:<2} {res.get('status'):<34} "
            f"roll={(res.get('rollout') or {}).get('status', '-'):<38} "
            f"depth={(res.get('rollout') or {}).get('goal_first_true_depth')}",
            flush=True,
        )
        return res

    print(f"running {len(jobs)} candidates, {N_PARALLEL} at a time", flush=True)
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
        results = list(ex.map(_run, jobs))

    outp = HERE.parent / "phase2_raw.json"
    outp.write_text(
        json.dumps(
            {
                "call_index": CALL_INDEX,
                "max_steps": 400,
                "worker_timeout_s": WORKER_TIMEOUT_S,
                "wall_s": round(time.monotonic() - t0, 1),
                "splits": {g: {k: v for k, v in s.items() if not k.startswith("_")} for g, s in splits.items()},
                "results": results,
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(f"\nwrote {outp}  ({round(time.monotonic() - t0, 1)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
