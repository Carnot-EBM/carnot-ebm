"""Score every collected engine, one bounded worker process per cell.

Separate from `run_ab.py` so the GPU-bound collection is never at risk from a hostile engine, and
so scoring can be re-run without re-generating. A worker that does not finish inside
`WORKER_TIMEOUT_S` is recorded `worker_timeout` -- a MISSING OBSERVATION, never a zero.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/ianblenke/github.com/ianblenke/carnot")
OUT = HERE / "out"
WORKER_TIMEOUT_S = float(os.environ.get("INERT_WORKER_TIMEOUT_S", "180"))


def main() -> int:
    rows = json.loads((OUT / "rows.json").read_text())
    scored: list[dict] = []
    t0 = time.time()
    job_path = OUT / "_job.json"
    for i, r in enumerate(rows, 1):
        cell_id = f"{r['game']}__r{r['replicate']}__{r['tag']}"
        eng = OUT / "engines" / f"{cell_id}.py"
        out: dict = {
            "cell_id": cell_id,
            "game": r["game"],
            "replicate": r["replicate"],
            "arm": r["arm"],
            "tag": r["tag"],
        }
        if not eng.exists():
            out["status"] = "no_engine_emitted"
            scored.append(out)
            continue
        job_path.write_text(
            json.dumps(
                {
                    "cell_id": cell_id,
                    "engine_path": str(eng),
                    "window_pkl": str(OUT / "windows" / f"{r['game']}.pkl"),
                }
            )
        )
        env = dict(os.environ)
        env["CARNOT_REPO"] = str(ROOT)
        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, str(HERE / "score_worker.py"), str(job_path)],
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_S,
                env=env,
            )
        except subprocess.TimeoutExpired:
            # A MISSING OBSERVATION. The engine could not be scored inside the bound, which is a
            # fact about the engine but not a change_fidelity of 0 -- scoring it as 0 would
            # silently convert "we could not measure this" into "this measured badly".
            out["status"] = "worker_timeout"
            out["worker_timeout_s"] = WORKER_TIMEOUT_S
            scored.append(out)
            print(f"[{i}/{len(rows)}] {cell_id} TIMEOUT")
            continue
        if proc.returncode != 0:
            out["status"] = "worker_crashed"
            out["stderr_tail"] = (proc.stderr or "")[-400:]
            scored.append(out)
            print(f"[{i}/{len(rows)}] {cell_id} CRASHED rc={proc.returncode}")
            continue
        try:
            out.update(json.loads((proc.stdout or "").strip().splitlines()[-1]))
        except Exception as exc:  # noqa: BLE001
            out["status"] = "worker_unreadable"
            out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        scored.append(out)
        h = out.get("heldout") or {}
        sg = out.get("state_graph") or {}
        print(
            f"[{i}/{len(rows)}] {cell_id} {out.get('status')} "
            f"cf={h.get('change_fidelity')} depth={sg.get('probe_depth_reached')}"
        )
        (OUT / "scored.json").write_text(json.dumps(scored, indent=2))
    job_path.unlink(missing_ok=True)
    (OUT / "scored.json").write_text(json.dumps(scored, indent=2))
    print(f"scored {len(scored)} cells in {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
