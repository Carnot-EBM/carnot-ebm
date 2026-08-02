#!/usr/bin/env python3
"""Driver: enumerate saved engines, group by game, score each game in its own killable process."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
CELLS = OUT / "cells"
CELLS.mkdir(exist_ok=True)
SCRATCH = pathlib.Path(os.environ["SCRATCH_E3"])
SCRATCH.mkdir(parents=True, exist_ok=True)
GAME_TIMEOUT_S = int(os.environ.get("GVS_GAME_TIMEOUT_S", "1800"))

OBJPERC = REPO / "results/arc_object_perception_ab_change_fidelity_20260801/engines"
INERT = REPO / "results/arc_inert_rejection_ab_20260801/out/engines"
E3 = REPO / "results/arc_e3"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def enumerate_jobs() -> tuple[dict, dict]:
    by_game: dict[str, list] = defaultdict(list)
    prov: dict = {"objperc": 0, "inert": 0, "e3store": 0, "skipped": []}
    if OBJPERC.is_dir():
        for cell_dir in sorted(OBJPERC.iterdir()):
            if not cell_dir.is_dir():
                continue
            game = cell_dir.name.split("__")[0]
            p = cell_dir / game / "world_model.py"
            if not p.exists():
                prov["skipped"].append({"cell": cell_dir.name, "why": "no world_model.py"})
                continue
            by_game[game].append(
                {"cell": cell_dir.name, "corpus": "objperc", "path": str(p), "sha256": sha(p)}
            )
            prov["objperc"] += 1
    if INERT.is_dir():
        for p in sorted(INERT.glob("*.py")):
            game = p.stem.split("__")[0]
            by_game[game].append(
                {"cell": p.stem, "corpus": "inert", "path": str(p), "sha256": sha(p)}
            )
            prov["inert"] += 1
    if E3.is_dir():
        for gdir in sorted(E3.iterdir()):
            p = gdir / "world_model.py"
            if not (gdir.is_dir() and p.exists()):
                continue
            by_game[gdir.name].append(
                {
                    "cell": f"{gdir.name}__e3store",
                    "corpus": "e3store",
                    "path": str(p),
                    "sha256": sha(p),
                }
            )
            prov["e3store"] += 1
    return by_game, prov


def main() -> int:
    by_game, prov = enumerate_jobs()
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None
    games = sorted(by_game) if only is None else [g for g in sorted(by_game) if g in only]
    print(
        f"corpora: {prov['objperc']} objperc + {prov['inert']} inert + "
        f"{prov['e3store']} e3store over {len(by_game)} games; running {len(games)}",
        flush=True,
    )
    (OUT / "jobs.json").write_text(
        json.dumps({"provenance": prov, "by_game": {g: by_game[g] for g in games}}, indent=1)
    )
    env = dict(os.environ, CARNOT_REPO=str(REPO), SCRATCH_E3=str(SCRATCH))
    status = {}
    for i, game in enumerate(games, 1):
        cell_out = CELLS / f"{game}.json"
        if cell_out.exists():
            print(f"[{i}/{len(games)}] {game} cached", flush=True)
            status[game] = "cached"
            continue
        jobfile = OUT / f".jobs_{game}.json"
        jobfile.write_text(json.dumps(by_game[game]))
        t0 = time.time()
        try:
            r = subprocess.run(
                [
                    str(REPO / ".venv/bin/python"),
                    str(HERE / "score_worker.py"),
                    game,
                    str(jobfile),
                    str(cell_out),
                ],
                env=env,
                timeout=GAME_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
            ok = r.returncode == 0 and cell_out.exists()
            status[game] = "ok" if ok else f"worker_rc{r.returncode}"
            if not ok:
                (OUT / f"err_{game}.log").write_text(
                    (r.stdout or "")[-4000:] + "\n---STDERR---\n" + (r.stderr or "")[-8000:]
                )
        except subprocess.TimeoutExpired:
            # DROPPED with its reason recorded. Never a zero.
            status[game] = "game_timeout"
        print(
            f"[{i}/{len(games)}] {game} {status[game]} "
            f"n_engines={len(by_game[game])} {time.time() - t0:.1f}s",
            flush=True,
        )
        jobfile.unlink(missing_ok=True)
    (OUT / "collect_status.json").write_text(json.dumps(status, indent=1))
    print("collection done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
