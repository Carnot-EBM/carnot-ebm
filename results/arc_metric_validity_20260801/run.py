#!/usr/bin/env python3
"""DRIVER: score every frozen induced engine on the metric AND on plannability. CPU only.

No LLM, no GPU, no new generation. Every input is already on disk: engine text from the two
frozen runs, and splits rebuilt by the same deterministic calls those runs made.

TWO CORPORA, DELIBERATELY, because each covers the other's hole:

  objperc  116 engines over 20 games (the object-perception A/B). Wide game coverage, but no
           `root_grid` was recorded, so the plan root must be reconstructed.
  bestofn   48 candidates over 6 games (the frozen best-of-N). Narrow, but its split carries a
           row-by-row proof AND it has the REAL `E3AgentPolicy.root_grid` on disk -- which is
           what makes the reconstruction over on `objperc` checkable instead of assumed.

WHY THE BEST-OF-N CANDIDATES ARE RE-PLANNED RATHER THAN JOINED FROM THEIR ARTIFACT. Every one of
its 48 records carries `goal_max_depth: 40`. The shipped default moved to 80 on 2026-07-31, and
`plan_max_depth_default`'s own measurement records that the change turns 2 plannable candidates
into 6 ON THIS CORPUS. Joining the frozen `plan_found` would therefore answer the question at a
horizon the live agent no longer uses, and would do it at the n where the answer is least
determinable. So plannability is re-derived at the current shipped default for both corpora.

EVERY WORKER IS KILLABLE AND A TIMEOUT IS UNDETERMINED, NOT ZERO. A candidate whose worker does
not return leaves BOTH numerator and denominator -- the same rule the frozen best-of-N applied to
its one non-terminating candidate. Scoring a hang as a failed plan would bias exactly the
quantity being estimated.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SCRATCH = Path(
    os.environ.get(
        "METRIC_VALIDITY_SCRATCH",
        "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
        "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/metric_validity",
    )
)
OBJPERC = REPO / "results" / "arc_object_perception_ab_change_fidelity_20260801"
BESTOFN = REPO / "results" / "arc_induce_bestofn_20260731"

WINDOW_TIMEOUT_S = 300.0
# The planner is allowed 20000 engine calls from each of up to three roots, plus a goal gate of
# the same size from each. A slow-but-terminating engine can legitimately need minutes.
SCORE_TIMEOUT_S = 900.0


def run_worker(worker: str, job: dict, tag: str, timeout: float) -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    jp = SCRATCH / f"job_{tag}.json"
    jp.write_text(json.dumps(job))
    env = dict(os.environ, CARNOT_REPO=str(REPO), CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    t0 = time.time()
    try:
        pr = subprocess.run(  # noqa: S603
            [sys.executable, str(HERE / worker), str(jp)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "undetermined_worker_timeout", "worker_wall_s": round(time.time() - t0, 1)}
    lines = (pr.stdout or "").strip().splitlines()
    if not lines:
        return {"status": "worker_no_output", "stderr": (pr.stderr or "")[-400:]}
    try:
        r = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"status": "worker_bad_output", "stdout": lines[-1][:300]}
    r["worker_wall_s"] = round(time.time() - t0, 1)
    return r


def build_objperc_jobs() -> tuple[list[dict], dict]:
    rows = json.loads((OBJPERC / "rows.json").read_text())
    games = sorted({r["game"] for r in rows})
    print(f"[objperc] rebuilding {len(games)} windows")
    win_status: dict = {}
    windows: dict[str, Path] = {}
    for g in games:
        p = SCRATCH / f"objperc_{g}_window.pkl"
        r = run_worker(
            "window_worker.py", {"game": g, "window_pkl": str(p)}, f"win_op_{g}", WINDOW_TIMEOUT_S
        )
        win_status[g] = r
        if r.get("status") == "ok" and p.exists():
            windows[g] = p
            print(
                f"  {g}: shown={r['n_shown']} held={r['n_heldout']} "
                f"held_changing={r['n_heldout_changing']} root={r['window_root_sha256_16']}"
            )
        else:
            print(f"  {g}: WINDOW NOT REBUILT ({r.get('status')})")

    jobs = []
    for r in rows:
        game, rep, tag = r["game"], r["replicate"], r["tag"]
        cell = f"{game}__r{rep}__{tag}"
        wp = windows.get(game)
        if wp is None:
            continue
        store = OBJPERC / "engines" / cell
        cands = sorted(store.rglob("world_model.py"))
        if not cands:
            continue
        jobs.append(
            {
                "cell": cell,
                "corpus": "objperc",
                "game": game,
                "arm": tag,
                "replicate": rep,
                "code_path": str(cands[0]),
                "window_pkl": str(wp),
            }
        )
    return jobs, win_status


def build_bestofn_jobs() -> tuple[list[dict], dict]:
    scored = json.loads((BESTOFN / "bestofn_scored.json").read_text())
    cands = scored["candidates"]
    games = sorted({c["game"] for c in cands})
    harness = BESTOFN / "harness"
    print(f"[bestofn] rebuilding {len(games)} proven splits")
    win_status: dict = {}
    windows: dict[str, Path] = {}
    for g in games:
        p = SCRATCH / f"bestofn_{g}_window.pkl"
        r = run_worker(
            "bon_window_worker.py",
            {
                "game": g,
                "window_pkl": str(p),
                "harness_dir": str(harness),
                "frozen_split_json": str(BESTOFN / "split.json"),
            },
            f"win_bon_{g}",
            WINDOW_TIMEOUT_S,
        )
        win_status[g] = r
        if r.get("status") == "ok" and p.exists():
            windows[g] = p
            print(
                f"  {g}: shown={r['n_shown']} held={r['n_heldout']} "
                f"held_changing={r['heldout_n_changing']} proven={r['split_proven']}"
            )
        else:
            print(f"  {g}: SPLIT NOT REBUILT ({r.get('status')})")

    jobs = []
    for c in cands:
        game = c["game"]
        wp = windows.get(game)
        if wp is None:
            continue
        # The completion text is frozen; re-extract the code exactly as score_bon.py did.
        cp = SCRATCH / f"bon_{game}_k{c['candidate']}.py"
        if not cp.exists():
            tag = c.get("tag") or "gpu1"
            txt = (harness / "bon" / tag / f"{game}_k{c['candidate']}.txt").read_text(
                errors="replace"
            )
            cp.write_text(_extract_python(txt) or txt.strip())
        root = harness / "capture" / game / "root_grid1.pkl"
        jobs.append(
            {
                "cell": f"{game}__k{c['candidate']}",
                "corpus": "bestofn",
                "game": game,
                "arm": "bestofn",
                "replicate": int(c["candidate"]),
                "code_path": str(cp),
                "window_pkl": str(wp),
                "real_root_pkl": str(root) if root.exists() else None,
                "frozen_plan_found_at_depth40": c.get("plan_found"),
                "frozen_goal_satisfiable_at_depth40": c.get("goal_satisfiable"),
            }
        )
    return jobs, win_status


def _extract_python(text: str) -> str:
    sys.path.insert(0, str(REPO / "python"))
    from carnot.agentic import arc_executable_world_model as e3

    return e3._extract_python(text)  # noqa: SLF001


def main() -> int:
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("MV_ONLY_CELLS", "")
    jobs_a, win_a = build_objperc_jobs()
    jobs_b, win_b = build_bestofn_jobs()
    jobs = jobs_a + jobs_b
    if only:
        want = set(only.split(","))
        jobs = [j for j in jobs if j["cell"] in want]
    print(f"\nscoring {len(jobs)} engines")

    results = []
    for i, j in enumerate(jobs, 1):
        r = run_worker("score_worker.py", j, f"sc_{j['corpus']}_{j['cell']}", SCORE_TIMEOUT_S)
        for k in ("arm", "replicate", "frozen_plan_found_at_depth40",
                  "frozen_goal_satisfiable_at_depth40"):
            if k in j:
                r.setdefault(k, j[k])
        r.setdefault("cell", j["cell"])
        r.setdefault("corpus", j["corpus"])
        r.setdefault("game", j["game"])
        results.append(r)
        cf = ((r.get("heldout") or {}).get("change_fidelity"))
        pf = ((r.get("plan") or {}).get("window_root") or {}).get("plan_found")
        print(
            f"  [{i:3d}/{len(jobs)}] {j['corpus']:<8} {j['cell']:<20} "
            f"status={r.get('status')} cf={cf} plan={pf} {r.get('worker_wall_s')}s",
            flush=True,
        )
        (SCRATCH / "partial_scored.json").write_text(json.dumps(results, indent=1))

    out = {
        "what_this_is": (
            "per-engine join of held-out change_fidelity to plan_in_model plannability, over two "
            "frozen corpora, at the CURRENT shipped search defaults. No LLM, no GPU."
        ),
        "duration_s": round(time.time() - t0, 2),
        "n_engines": len(results),
        "window_rebuild": {"objperc": win_a, "bestofn": win_b},
        "engines": results,
    }
    (HERE / "scored.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'scored.json'}  {out['duration_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
