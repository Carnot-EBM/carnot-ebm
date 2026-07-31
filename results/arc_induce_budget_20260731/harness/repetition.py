#!/usr/bin/env python3
"""Post-hoc: how much of each completion is a VERBATIM repetition loop?

`ramble_frac` in the sweep counts only bare `#` lines, and that undercounts badly -- the 8192
engine completions are ~80% loop while scoring `ramble_frac` 0.008, because the repeated unit is
a full sentence (`# target_block_row = (y // 6) * 6` x204) rather than an empty comment.

`loop_frac` here is the general form: the share of emitted lines sitting inside a run of >=3
IDENTICAL consecutive lines. It is the number that decides Phase 1. If doubling the budget leaves
`code_lines` flat and raises `loop_frac`, then the extra tokens were spent looping, and "raise
max_tokens" is the wrong lever no matter how many attempts are run.

Runs on the saved completions -- no GPU, no re-generation.
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_BARE = re.compile(r"^\s*#\s*$")


def loop_stats(text: str, min_run: int = 3) -> dict:
    lines = text.split("\n")
    n = len(lines)
    inside = 0
    longest = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and lines[j + 1] == lines[i]:
            j += 1
        run = j - i + 1
        if run >= min_run and lines[i].strip():
            inside += run
            longest = max(longest, run)
        i = j + 1
    code = [ln for ln in lines
            if ln.strip() and not ln.strip().startswith("#")]
    # CYCLE_FRAC -- the metric that actually catches this generator's degeneration, and the
    # reason `loop_frac` alone is not enough. At 16384 the engine completion alternates TWO
    # lines (`# target_block_row_end = y + 4` / `# target_block_col_end = x + 4`, 376 and 375
    # times), so no run of IDENTICAL CONSECUTIVE lines is long and `loop_frac` reads 0.19 --
    # while 1055 lines contain only 67 distinct ones. An ABAB cycle is the same failure as an
    # AAAA run and has to be counted as one.
    from collections import Counter

    nonblank = [ln for ln in lines if ln.strip()]
    counts = Counter(nonblank)
    cycled = sum(c for c in counts.values() if c >= 5)
    return {
        "n_lines": n,
        "lines_in_repetition_runs": inside,
        "loop_frac": round(inside / max(1, n), 4),
        "longest_verbatim_run": longest,
        "n_nonblank_lines": len(nonblank),
        "n_distinct_nonblank_lines": len(counts),
        "cycle_frac": round(cycled / max(1, len(nonblank)), 4),
        "code_lines": len(code),
        "bare_comment_lines": sum(1 for ln in lines if _BARE.match(ln)),
    }


def main() -> int:
    dirs = sys.argv[1:] or ["sweep", "sweep_combined", "sweep_refactor", "sweep_sampler"]
    rows = []
    for d in dirs:
        p = os.path.join(HERE, d, "sweep.json")
        if not os.path.exists(p):
            continue
        doc = json.load(open(p))
        for r in doc.get("rows") or []:
            fn = r.get("completion_file")
            if not fn:
                continue
            fp = os.path.join(HERE, d, fn)
            if not os.path.exists(fp):
                continue
            st = loop_stats(open(fp).read())
            rows.append({
                "lane": d, "arm": r.get("arm", "shipped"), "prompt": r["prompt"],
                "budget": r["budget"], "attempt": r["attempt"],
                "predicted_n": r.get("predicted_n"), "stop_type": r.get("stop_type"),
                "usable_engine": r.get("usable_engine"),
                "generate_would_accept": r.get("generate_would_accept"),
                **st,
            })

    hdr = (f"{'lane':>16} {'arm':>18} {'bud':>6} {'a':>2} {'pred_n':>7} {'stop':>6} "
           f"{'code':>5} {'loop_lines':>11} {'loop_frac':>10} {'longest':>8} {'USABLE':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["lane"], r["arm"], r["budget"], r["attempt"])):
        print(f"{r['lane']:>16} {r['arm']:>18} {r['budget']:>6} {r['attempt']:>2} "
              f"{str(r['predicted_n']):>7} {str(r['stop_type']):>6} {r['code_lines']:>5} "
              f"{r['lines_in_repetition_runs']:>11} {r['loop_frac']:>10.3f} "
              f"{r['longest_verbatim_run']:>8} {str(r['usable_engine']):>7}")

    print()
    print("AGGREGATE per (lane, arm, budget)")
    agg: dict = {}
    for r in rows:
        agg.setdefault((r["lane"], r["arm"], r["budget"]), []).append(r)
    print(f"{'lane':>16} {'arm':>18} {'bud':>6} {'n':>2} {'mean code':>10} "
          f"{'mean loop_frac':>15} {'n_usable':>9}")
    for k in sorted(agg):
        v = agg[k]
        print(f"{k[0]:>16} {k[1]:>18} {k[2]:>6} {len(v):>2} "
              f"{sum(r['code_lines'] for r in v)/len(v):>10.1f} "
              f"{sum(r['loop_frac'] for r in v)/len(v):>15.3f} "
              f"{sum(1 for r in v if r['usable_engine']):>9}")

    with open(os.path.join(HERE, "repetition.json"), "w") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
