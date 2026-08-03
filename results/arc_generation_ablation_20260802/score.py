"""Drive `score_worker.py` over every emitted engine, one killable subprocess each.

MISSING IS NEVER ZERO. A worker that times out or crashes is recorded with its status and is
EXCLUDED from every aggregate. It is not scored 0 -- an engine we failed to measure carries no
evidence about the arm that produced it, and coercing it to 0 would let a slow arm look like a bad
one. The counts of each status are reported alongside every aggregate.

`--probe` runs only the reachability probes (hand-written oracle + identity floor) on every game,
through the SAME worker and the SAME scorer the arms are graded with.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
OUT = HERE / "out"
ENGINES = OUT / "engines"
SCORED = OUT / "scored"
SCORED.mkdir(parents=True, exist_ok=True)
PY = os.environ.get("ABL_PY", "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python")
WORKER_TIMEOUT_S = float(os.environ.get("ABL_WORKER_TIMEOUT_S", "600"))


def run_worker(job: dict) -> dict:
    env = dict(os.environ)
    env["CARNOT_REPO"] = str(ROOT)
    env["CARNOT_ARC_E3_DIR"] = job["tmp_e3"]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(job, fh)
        job_path = fh.name
    t0 = time.time()
    try:
        p = subprocess.run(
            [PY, str(HERE / "score_worker.py"), job_path],
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT_S,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "cell_id": job.get("cell_id"),
            "game": job["game"],
            "mode": job["mode"],
            "status": "worker_timeout",
            "worker_s": round(time.time() - t0, 2),
        }
    finally:
        os.unlink(job_path)
    line = ""
    for ln in p.stdout.splitlines():
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            line = ln
    if not line:
        return {
            "cell_id": job.get("cell_id"),
            "game": job["game"],
            "mode": job["mode"],
            "status": "worker_no_output",
            "returncode": p.returncode,
            "stderr_tail": p.stderr[-600:],
            "worker_s": round(time.time() - t0, 2),
        }
    try:
        return json.loads(line)
    except Exception as exc:  # noqa: BLE001
        return {
            "cell_id": job.get("cell_id"),
            "game": job["game"],
            "mode": job["mode"],
            "status": "worker_bad_json",
            "error": str(exc)[:200],
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    windows_pkl = str(OUT / "windows.pkl")
    prep = json.loads((OUT / "prep_meta.json").read_text())
    games = [g for g in prep if g != "_prep" and prep[g].get("built")]
    tmp_root = Path(tempfile.mkdtemp(prefix="abl_score_"))

    if args.probe:
        res = []
        for mode in ("oracle", "identity"):
            for g in games:
                job = {
                    "mode": mode,
                    "game": g,
                    "windows_pkl": windows_pkl,
                    "tmp_e3": str(tmp_root / f"{mode}_{g}"),
                    "cell_id": f"PROBE_{mode}_{g}",
                }
                r = run_worker(job)
                res.append(r)
                ta = (r.get("tail") or {}).get("change_accuracy")
                fa = (r.get("fresh") or {}).get("change_accuracy")
                fn = (r.get("fresh") or {}).get("n_changing")
                print(
                    f"  {mode:8} {g:6} tail_ca={ta} fresh_ca={fa} fresh_n_changing={fn} "
                    f"{r.get('status')}",
                    flush=True,
                )
        (OUT / "probe.json").write_text(json.dumps(res, indent=2))
        oracle = [r for r in res if r["mode"] == "oracle"]
        ident = [r for r in res if r["mode"] == "identity"]

        def _all(rs, blk, field, val):
            xs = [(r.get(blk) or {}).get(field) for r in rs if (r.get(blk) or {}).get("measurable")]
            return bool(xs) and all(x == val for x in xs), len(xs)

        o_tail_ok, n_ot = _all(oracle, "tail", "change_accuracy", 1.0)
        o_fresh_ok, n_of = _all(oracle, "fresh", "change_accuracy", 1.0)
        i_tail_0, n_it = _all(ident, "tail", "change_accuracy", 0.0)
        i_fresh_0, n_if = _all(ident, "fresh", "change_accuracy", 0.0)
        summary = {
            "oracle_reaches_1.0_on_every_tail": o_tail_ok,
            "n_tail_blocks": n_ot,
            "oracle_reaches_1.0_on_every_fresh": o_fresh_ok,
            "n_fresh_blocks": n_of,
            "identity_is_0.0_on_every_tail": i_tail_0,
            "n_identity_tail": n_it,
            "identity_is_0.0_on_every_fresh": i_fresh_0,
            "n_identity_fresh": n_if,
            "metric_is_reachable_and_discriminating": bool(
                o_tail_ok and o_fresh_ok and i_tail_0 and i_fresh_0
            ),
            "why_this_gate_exists": "a prior arm in this project 'measured' 0 plans while a "
            "hardcoded 0.5 threshold sat above an achievable maximum of "
            "0.0476, so its zero was arithmetically forced rather than "
            "observed. A zero reported below is a MEASURED zero only "
            "because this probe shows 1.0 was reachable on the same rows "
            "with the same scorer.",
        }
        (OUT / "probe_summary.json").write_text(json.dumps(summary, indent=2))
        print("\n" + json.dumps(summary, indent=2))
        return 0

    engines = sorted(ENGINES.glob("*.py"))
    print(f"{len(engines)} engines to score", flush=True)
    todo = []
    for p in engines:
        cell_id = p.stem
        outp = SCORED / f"{cell_id}.json"
        if outp.exists():
            continue
        game = cell_id.split("__")[0]
        todo.append(
            {
                "mode": "engine",
                "game": game,
                "engine_path": str(p),
                "windows_pkl": windows_pkl,
                "tmp_e3": str(tmp_root / cell_id),
                "cell_id": cell_id,
            }
        )
    print(f"{len(todo)} not yet scored", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for r in ex.map(run_worker, todo):
            (SCORED / f"{r['cell_id']}.json").write_text(json.dumps(r, indent=2))
            done += 1
            if done % 20 == 0 or r.get("status") != "ok":
                print(
                    f"  [{done}/{len(todo)}] {r['cell_id']} {r.get('status')} "
                    f"tail_ca={(r.get('tail') or {}).get('change_accuracy')} "
                    f"fresh_ca={(r.get('fresh') or {}).get('change_accuracy')}",
                    flush=True,
                )
    print("scoring done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
