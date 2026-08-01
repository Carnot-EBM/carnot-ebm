"""ESCALATION: the same gate, on a REAL ARC frame instead of a synthetic block grid.

WHY THIS EXISTS. `fidelity.py` measures the CHARITABLE case by construction: an 8x8
arrangement of solid 8x8-cell blocks, in a maximally-separated nameable palette. That
design was chosen so a FAILURE would be conclusive (fail there, fail everywhere). It
is the wrong instrument for a PASS -- a real ARC frame is much finer-grained, so a
good score on solid blocks does not license "images can carry ARC grids".

The pre-registered escalation rule in `fidelity.py` (design decision 3) therefore
requires this run before any positive claim.

THE GRID IS NOT SYNTHETIC. It is the actual reset frame of lp85, pulled from the
OFFLINE arcade over the local `environment_files/` (zero quota, no network, no scored
play). Its real palette natively contains 4 AND 14, and 5 AND 15 -- the exact index
pairs that `arc_executable_world_model.to_ascii()` already collapses today by keeping
only the last decimal digit. So the confusion matrix here is measured on real
co-occurring colours, not on a synthetic pairing we invented to be hard.

WHY lp85 AND NOT ka59. ka59 was the first candidate and was rejected on inspection of
its actual cell counts, BEFORE any probe was run against it: its reset frame contains
exactly ONE cell of colour 0 and ONE of colour 5, and only 16 of colour 14. A
stratified probe set over that frame tops out near n=90 and would rest the entire
4<->14 / 5<->15 confusion question on a handful of cells -- too thin to say anything.
lp85's reset frame carries 11 distinct colours with a minimum of 32 cells each and
still contains both critical pairs, so it supports the full n=128 with real coverage
of the pairs of interest. This is a sampling-adequacy decision made on cell counts
alone, not a search for a frame that scores better.

Everything else -- palette, prompt, batching, probe count, scoring, the strict/parsed
split, the truncation retry -- is IDENTICAL to `fidelity.py`, so the two runs are
directly comparable and the only thing that changed is the grid.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np

SD = (
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/mmgate"
)
sys.path.insert(0, SD)

import fidelity as F  # noqa: E402, N812
from common import PALETTE, image_part, render_png, rle_grid, text_part  # noqa: E402

GAME = "lp85"


def main():
    grid = np.load(f"{SD}/real_{GAME}.npy").astype(np.uint8)
    present = sorted(int(v) for v in np.unique(grid))
    rng = np.random.default_rng(F.SEED)

    # TWO-STAGE STRATIFIED SAMPLE, without replacement.
    #   Stage 1 guarantees COVERAGE: min(per, available) cells of every colour the
    #     frame actually contains, so a rare-but-present colour cannot be missed and
    #     the 4<->14 / 5<->15 confusions have cells to be measured on.
    #   Stage 2 restores POWER: top up to the full N_PROBES by sampling uniformly from
    #     the cells not already chosen. Without this the probe count would be pinned to
    #     (n_colours * per) and the run would silently lose precision relative to the
    #     synthetic arm, breaking the comparison between them.
    # We do not invent or duplicate cells to balance the classes -- the frame is the
    # frame, and stage 2 samples the frame's own natural colour distribution.
    per = max(1, F.N_PROBES // len(present))
    chosen: set[tuple[int, int]] = set()
    probes: list[tuple[int, int, int]] = []
    for v in present:
        rs, cs = np.where(grid == v)
        k = min(per, len(rs))
        for i in rng.choice(len(rs), size=k, replace=False):
            probes.append((int(rs[i]), int(cs[i]), int(v)))
            chosen.add((int(rs[i]), int(cs[i])))
    all_cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    remaining = [rc for rc in all_cells if rc not in chosen]
    need = F.N_PROBES - len(probes)
    if need > 0 and remaining:
        for i in rng.choice(len(remaining), size=min(need, len(remaining)), replace=False):
            r, c = remaining[i]
            probes.append((r, c, int(grid[r, c])))
    rng.shuffle(probes)
    probes = probes[: (len(probes) // F.BATCH) * F.BATCH]  # whole batches only

    rle = rle_grid(grid)
    meta = {
        "grid_source": f"REAL {GAME} reset frame from the OFFLINE arcade over environment_files/",
        "game": GAME,
        "grid_shape": list(grid.shape),
        "colours_present": present,
        "contains_4_and_14": bool(4 in present and 14 in present),
        "contains_5_and_15": bool(5 in present and 15 in present),
        "grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
        "probes_sha256": hashlib.sha256(json.dumps(probes).encode()).hexdigest(),
        "n_probes": len(probes),
        "palette": [{"index": i, "name": n, "rgb": list(c)} for i, (n, c) in enumerate(PALETTE)],
        "rle_chars": len(rle),
    }
    print(json.dumps({k: v for k, v in meta.items() if k != "palette"}, indent=1))

    results = []
    t_all = time.time()
    print("  --- TEXT CONTROL (shipped RLE encoding), REAL frame ---", flush=True)
    txt = (
        "Here is the grid, run-length encoded. Each line is `r<row>:<value>x<count>,...` "
        "covering that row's columns left to right with no gaps."
        f"\n{rle}\n\n"
    )
    results.append(F.run_scheme("real_text_rle", lambda: [text_part(txt)], probes, grid))

    for px in F.PX_SCHEMES:
        png = render_png(grid, px)
        print(
            f"  --- REAL IMAGE px_per_cell={px} "
            f"({grid.shape[0] * px}x{grid.shape[1] * px}px, {len(png)}B) ---",
            flush=True,
        )
        r = F.run_scheme(f"real_image_px{px}", lambda p=png: [image_part(p)], probes, grid)
        r["px_per_cell"] = px
        r["image_pixels"] = [grid.shape[0] * px, grid.shape[1] * px]
        r["image_bytes"] = len(png)
        results.append(r)
        with open(f"{SD}/real_{GAME}_px{px}.png", "wb") as f:
            f.write(png)

    out = {"meta": meta, "results": results, "duration_s": round(time.time() - t_all, 2)}
    with open(f"{SD}/escalate_raw.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== ESCALATION SUMMARY (real lp85 frame) ===")
    for r in results:
        print(
            f"  {r['scheme']:18s} exact={r['exact_match']:.3f} "
            f"({r['n_correct']}/{r['cells_probed']}) unparsed={r['n_unparseable']} "
            f"top_conf={list(r['confusions'].items())[:5]}"
        )


if __name__ == "__main__":
    main()
