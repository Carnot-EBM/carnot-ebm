"""Build a REAL worst-case ARC induce prompt via the production induce_prompt().

64x64 logical grid == the largest in ops/arc_solve_registry.yaml, which is the shape
_INDUCE_WORST_CASE_PROMPT_TOKENS = 15734 was measured against. Grids are constructed
(not replayed from a game) but they go through the SAME production prompt builder, so
the token count and structure are faithful to what the induce path actually sends.
"""

import sys, json
import numpy as np

sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")
from carnot.agentic.arc_executable_world_model import induce_prompt, Transition  # noqa: E402

rng = np.random.default_rng(5900)
H = W = 64
NCOLORS = 10

base = rng.integers(0, NCOLORS, size=(H, W), dtype=np.int64)
trans = []
cur = base.copy()
# 25 transitions == what the collector gathers; k=8 is the production default shown to the LLM.
for i in range(25):
    nxt = cur.copy()
    # scatter a realistic number of changed cells in horizontal runs (the RLE delta format)
    for _ in range(rng.integers(6, 14)):
        r = int(rng.integers(0, H))
        c = int(rng.integers(0, W - 8))
        ln = int(rng.integers(2, 8))
        nxt[r, c : c + ln] = int(rng.integers(0, NCOLORS))
    trans.append(
        Transition(
            grid=cur.copy(),
            action=int(i % 6) + 1,
            data=None,
            next_grid=nxt.copy(),
            level_before=0,
            level_after=0,
        )
    )
    cur = nxt

p = induce_prompt("wc64", trans, cell=1, k=8)
out = "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/fitgrid/induce_prompt_worstcase.txt"
open(out, "w").write(p)
print(json.dumps({"chars": len(p), "path": out}))
