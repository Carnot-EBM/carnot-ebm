#!/usr/bin/env python3
"""Re-score ONE frozen best-of-N candidate under every mask arm, in a KILLABLE process.

THIS IS A DIFFERENT CORPUS FROM THE A/B AND THE DISTINCTION IS LOAD-BEARING. Q1 and Q3 are
about `results/arc_induce_bestofn_20260731` -- 40 stall-path candidates over 5 games (tn36,
tu93, ft09, sc25, lp85) -- because that is the only corpus that records `plan_found` per
candidate. Q2 is about the 116 A/B engines over 20 games. The six perfect-`change_fidelity`
tn36 bar-tickers live HERE, not in the A/B. Reporting one corpus's numbers under the other's
question would be the whole result.

The split is rebuilt through the best-of-N harness's own `load_split`, imported rather than
reimplemented, so "held out" cannot drift into two definitions. `_induce_transitions_k` is
pinned to 8 exactly as the join that produced `fidelity_vs_plan.json` pinned it -- the shown/
held-out boundary depends on it, so an unpinned value would silently grade a different split
and the unmasked reproduction check is what would catch it.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_hudms/e3_bon")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "results" / "arc_induce_bestofn_20260731" / "harness"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    out: dict = {"status": "ok", "game": job["game"], "cand": job["cand"]}

    import carnot.agentic.arc_executable_world_model as _m
    import numpy as np

    _m._induce_transitions_k = lambda: 8  # noqa: SLF001

    from hud_masks import masks_for
    from score_arms import score_all_arms
    from split import load_split

    s = load_split(job["game"], 1)
    held = list(s["_heldout"])
    full_corpus = list(s["_full"])

    src = pathlib.Path(job["code_path"]).read_text()
    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(src, job["code_path"], "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"unrunnable:{type(exc).__name__}"
        out["error"] = str(exc)[:200]
        print(json.dumps(out))
        return 0
    engine = ns.get("engine")
    if not callable(engine):
        out["status"] = "no_engine_symbol"
        print(json.dumps(out))
        return 0

    m = masks_for(job["game"])
    out["mask_meta"] = m["meta"]
    out.update(score_all_arms(engine, held, full_corpus, m))
    out["n_graded_transitions"] = len(held)
    out["n_full_corpus_transitions"] = len(full_corpus)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
