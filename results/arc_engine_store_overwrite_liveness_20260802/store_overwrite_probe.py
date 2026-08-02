"""Is the engine-store overwrite regression still live? CPU-only, no LLM, scratch store.

Drives the REAL shipped `execute_bounded_llm_reinduction` with a scripted proposer, and
instruments the store at the RECEIVING end (every write to world_model.py and every read
by `_load_engine_from`) so the write/read INTERLEAVING is observed, not inferred from
call sites. Corpus + engine sources are reused verbatim from
tests/python/test_arc_engine_retention_best_round.py so the signals are the measured ones.

EVIDENCE SAFETY: the store is redirected to a scratch dir before any call. Nothing here
can touch results/arc_e3.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRATCH = Path(os.environ["PROBE_SCRATCH"])
os.environ.setdefault("CARNOT_ARC_E3_DIR", str(SCRATCH / "store_default"))

sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")
sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot")

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic import arc_llm_reinduction as reind  # noqa: E402
from carnot.agentic.arc_executable_world_model import Transition  # noqa: E402

GAME = "retn"
_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTIONS = [1, 3, 0, 2, 1, 3, 0, 2, 3]


def _true_next(grid, action):
    g = grid.copy()
    pos = np.argwhere(g == 3)
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES[int(action) % 4]
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    if int(action) % 4 == 3:
        g[5, 5] = int(g[5, 5]) + 1
    return g


def _corpus():
    grid = np.zeros((6, 6), dtype=int)
    grid[2, 2] = 3
    root = grid.copy()
    rows = []
    for action in ACTIONS:
        nxt = _true_next(grid, action)
        rows.append(
            Transition(
                grid=grid.copy(),
                action=action,
                data=None,
                next_grid=nxt.copy(),
                level_before=0,
                level_after=0,
            )
        )
        grid = nxt
    return rows, root


GOOD_SRC = """
import numpy as np

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    if len(pos) == 0:
        return g
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES.get(int(action) % 4, (0, 0))
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def is_level_complete(grid):
    g = np.asarray(grid)
    pos = np.argwhere(g == 3)
    return bool(len(pos) and int(pos[0][0]) == 0 and int(pos[0][1]) == 0)
"""

WORSE_SRC = """
import numpy as np


def engine(grid, action, data):
    return np.asarray(grid).copy()


def is_level_complete(grid):
    return False
"""

WORST_SRC = """
import numpy as np


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    g[0, 0] = 1
    return g


def is_level_complete(grid):
    return False
"""

_LABEL = {
    hashlib.sha256(GOOD_SRC.encode()).hexdigest()[:12]: "GOOD",
    hashlib.sha256(WORSE_SRC.encode()).hexdigest()[:12]: "WORSE",
    hashlib.sha256(WORST_SRC.encode()).hexdigest()[:12]: "WORST",
}


def _label(text):
    if text is None:
        return "ABSENT"
    return _LABEL.get(hashlib.sha256(text.encode()).hexdigest()[:12], "UNKNOWN")


EVENTS: list[dict] = []


class _ScriptedProposer:
    """Writes one engine source per round, exactly as the real proposer does."""

    model_specs = "scripted-store-overwrite-probe"

    def __init__(self, store: Path, sources: list[str], tag: str) -> None:
        self.store = Path(store)
        self.sources = list(sources)
        self.tag = tag
        self.writes: list[int] = []

    def _write(self, index):
        src = self.sources[min(index, len(self.sources) - 1)]
        path = self.store / GAME / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        prior = path.read_text() if path.exists() else None
        # RECEIVING-END instrumentation: what was on disk immediately before this write.
        EVENTS.append(
            {
                "op": "WRITE",
                "call": self.tag,
                "round": index + 1,
                "incumbent_before_write": _label(prior),
                "writing": _label(src),
                "destroyed_incumbent": prior is not None and prior != src,
            }
        )
        path.write_text(src)
        self.writes.append(index)
        return True, f"wrote round {index + 1}"

    def induce(self, game, trans, cell, *, previous_level_complete_grid=None, **kw):
        return self._write(0)

    def refactor(self, game, vr):
        return self._write(len(self.writes))


def _instrument_reads(tag_holder):
    original = e3._load_engine_from

    def wrapped(root, game):
        p = Path(root) / game / "world_model.py"
        text = p.read_text() if p.exists() else None
        EVENTS.append(
            {
                "op": "READ",
                "call": tag_holder[0],
                "read_label": _label(text),
            }
        )
        return original(root, game)

    e3._load_engine_from = wrapped
    return original


def _store_text(store):
    p = Path(store) / GAME / "world_model.py"
    return p.read_text() if p.exists() else None


def _run_call(store, sources, tag, tag_holder, retention_env=None):
    tag_holder[0] = tag
    if retention_env is None:
        os.environ.pop("CARNOT_ARC_ENGINE_RETENTION", None)
    else:
        os.environ["CARNOT_ARC_ENGINE_RETENTION"] = retention_env
    e3.E3_DIR = Path(store)
    transitions, root = _corpus()
    proposer = _ScriptedProposer(store, sources, tag)
    result = reind.execute_bounded_llm_reinduction(
        game=GAME,
        transitions=transitions,
        cell=1,
        root_grid=root,
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
        load_engine=e3.load_engine,
        plan_in_model=lambda engine, goal, grid: None,
        max_rounds=3,
        min_heldout_accuracy=1.0,
    )
    return {
        "tag": tag,
        "retention_env": retention_env,
        "retention_enabled_reported": bool(result.engine_retention.get("enabled")),
        "best_round": result.engine_retention.get("best_round"),
        "rounds_seen": result.engine_retention.get("rounds_seen"),
        "best_round_signal": result.engine_retention.get("best_round_signal"),
        "restored": result.engine_retention.get("restored"),
        "store_after": _label(_store_text(store)),
        "planned": bool(result.planned),
        "skipped": result.skipped,
    }


def main():
    t0 = time.time()
    tag_holder = ["init"]
    _instrument_reads(tag_holder)
    out: dict = {"cells": {}}

    # ---- CELL 1: WITHIN-CALL, retention default (delivery check) --------------------
    s1 = SCRATCH / "cell1"
    out["cells"]["cell1_within_call_retention_default"] = _run_call(
        s1, [GOOD_SRC, WORSE_SRC, WORST_SRC], "cell1", tag_holder
    )

    # ---- CELL 2: WITHIN-CALL, retention OFF (axis-reachability control) -------------
    s2 = SCRATCH / "cell2"
    out["cells"]["cell2_within_call_retention_off"] = _run_call(
        s2, [GOOD_SRC, WORSE_SRC, WORST_SRC], "cell2", tag_holder, retention_env="0"
    )

    # ---- CELL 3: CROSS-CALL, retention default --------------------------------------
    s3 = SCRATCH / "cell3"
    call1 = _run_call(s3, [GOOD_SRC, WORSE_SRC, WORST_SRC], "cell3_call1", tag_holder)
    store_between = _label(_store_text(s3))
    call2 = _run_call(s3, [WORSE_SRC, WORST_SRC, WORST_SRC], "cell3_call2", tag_holder)
    out["cells"]["cell3_cross_call"] = {
        "call1": call1,
        "store_between_calls": store_between,
        "call2": call2,
        "incumbent_survived_call2": call2["store_after"] == "GOOD",
    }

    # ---- CELL 4: the LIVE single-shot path shape (agent line ~6497-6506) ------------
    # `self._proposer().induce(...)` then `e3.load_engine(...)`: no retention involved.
    s4 = SCRATCH / "cell4"
    e3.E3_DIR = Path(s4)
    (Path(s4) / GAME).mkdir(parents=True, exist_ok=True)
    (Path(s4) / GAME / "world_model.py").write_text(GOOD_SRC)
    tag_holder[0] = "cell4_singleshot"
    incumbent = _label(_store_text(s4))
    p4 = _ScriptedProposer(s4, [WORSE_SRC], "cell4_singleshot")
    ok, _msg = p4.induce(GAME, [], 1)
    eng, done = e3.load_engine(GAME)
    grid = np.zeros((6, 6), dtype=int)
    grid[2, 2] = 3
    moved = np.asarray(eng(grid.copy(), 1, None))
    out["cells"]["cell4_live_single_shot_induce_then_load"] = {
        "incumbent_before": incumbent,
        "induce_ok": bool(ok),
        "store_after": _label(_store_text(s4)),
        "loaded_engine_is_noop": bool(np.array_equal(moved, grid)),
        "note": "models arc_competition_agent.py:6497 induce() -> 6506 e3.load_engine()",
    }

    # ---- CELL 5: does the destroyed engine matter to the PLANNER? -------------------
    # Plan with GOOD on disk vs WORSE on disk, using the shipped e3.plan_in_model.
    s5 = SCRATCH / "cell5"
    e3.E3_DIR = Path(s5)
    (Path(s5) / GAME).mkdir(parents=True, exist_ok=True)
    root = np.zeros((6, 6), dtype=int)
    root[2, 2] = 3
    plan_results = {}
    for name, src in (("GOOD", GOOD_SRC), ("WORSE", WORSE_SRC)):
        (Path(s5) / GAME / "world_model.py").write_text(src)
        tag_holder[0] = f"cell5_{name}"
        eng, goal = e3.load_engine(GAME)
        try:
            plan = e3.plan_in_model(eng, goal, root.copy())
            plan_results[name] = {
                "plan_found": plan is not None and len(plan) > 0,
                "plan_length": (0 if not plan else len(plan)),
            }
        except Exception as exc:  # pragma: no cover
            plan_results[name] = {"error": repr(exc)[:200]}
    out["cells"]["cell5_planner_on_stored_engine"] = plan_results

    out["events"] = EVENTS
    out["duration_s"] = round(time.time() - t0, 4)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
