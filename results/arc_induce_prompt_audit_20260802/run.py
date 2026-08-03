#!/usr/bin/env python3
"""DRIVER: audit the induce PROMPT across the whole 25-game public roster. CPU ONLY.

ONE GAME PER KILLABLE SUBPROCESS. `build_progress_window` solves the game offline and is the
slow part; a game that does not return inside the bound is recorded as a COVERAGE GAP with its
reason, never as a zero. `arc_goal_predicate_anatomy_20260801` lost tr87 to exactly this and
said so; the same discipline applies here.

NO GPU, NO LLM, NO GENERATION, NO SUBMISSION, NO SCORED GAME. The tokenizer is loaded
vocab_only (no weights). CUDA_VISIBLE_DEVICES is emptied in the worker before any carnot
import, so this can never contend for a GPU another session on this shared machine owns.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKER = HERE / "worker.py"
PYBIN = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"

REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
GAMES = sorted(str(e["game"]) for e in yaml.safe_load(REGISTRY.read_text())["games"])

# The LIVE generator, per CLAUDE.md / project_arc_live_generator: gemma-4-31B-it-qat.
# ARC_LIVE_GENERATOR_MODEL_FILENAME in arc_executable_world_model.py names this exact file.
GGUF = (
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-qat-GGUF/"
    "snapshots/43cc1aeb31adf47ec06a854507ce552cd9862e6f/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
)
PER_GAME_TIMEOUT_S = 1200


def one(game: str, cells: Path, dump: Path) -> dict:
    out = cells / f"{game}.json"
    if out.exists():
        try:
            return json.loads(out.read_text())
        except Exception:
            out.unlink(missing_ok=True)
    job = cells / f"{game}.job.json"
    job.write_text(json.dumps({"game": game, "gguf": GGUF, "dump_dir": str(dump)}))
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_WAIT_POLICY": "PASSIVE",
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    t0 = time.time()
    try:
        subprocess.run(
            [PYBIN, str(WORKER), str(job), str(out)],
            timeout=PER_GAME_TIMEOUT_S,
            env=env,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        rec = {
            "game": game,
            "status": "timeout",
            "error": f"worker exceeded {PER_GAME_TIMEOUT_S}s",
            "elapsed_s": round(time.time() - t0, 2),
        }
        out.write_text(json.dumps(rec, indent=1))
        return rec
    if not out.exists():
        rec = {
            "game": game,
            "status": "no_output",
            "error": "worker produced no record",
            "elapsed_s": round(time.time() - t0, 2),
        }
        out.write_text(json.dumps(rec, indent=1))
        return rec
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games", default="")
    a = ap.parse_args()
    cells = HERE / "cells"
    dump = HERE / "out" / "prompts"
    cells.mkdir(parents=True, exist_ok=True)
    dump.mkdir(parents=True, exist_ok=True)
    games = [g.strip() for g in a.games.split(",") if g.strip()] or GAMES
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, g, cells, dump): g for g in games}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"  {r['game']:6s} {r.get('status'):12s} {r.get('elapsed_s')}s", flush=True)
    rows.sort(key=lambda r: r["game"])
    (HERE / "out" / "rows.json").write_text(json.dumps(rows, indent=1))
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n{len(ok)}/{len(rows)} ok, {len(rows) - len(ok)} coverage gaps")


if __name__ == "__main__":
    main()
