"""Tool registry for tool-assisted ARC world-model induction (REQ-ARC-WMTE-6460).

WHY THIS EXISTS. Single-shot induction on a reasoning generator burns 38k-94k decode
tokens per call (median ~61k). ~95% of that is the think channel, and the model spends
it MENTALLY SIMULATING grid transforms against the observed transitions. Decode is
15.4x more expensive than prefill on the scored card. These tools replace mental
simulation with actual execution: the model proposes an engine, a tool RUNS it on the
observed transitions and returns a mismatch report, and the model reasons only about
the report.

MCP AS VOCABULARY, NOT TRANSPORT. The tools are plain functions plus JSON schemas.
The live loop (arc_induction_tool_loop.py) calls them directly in-process. No stdio
subprocess, no asyncio, no MCP dependency on the scored path. `register_mcp_tools`
exposes the same registry through carnot.mcp.server's optional FastMCP import for
dev use only.

THE MEMORIZATION TRAP, MITIGATED HERE BY DESIGN. `run_engine_on_transitions`
optimises in-sample recall by construction, and hardcoding the layout coordinates is
the cheapest way to pass it (tr87's two ~1.0 cells hardcode window coordinates; its
0.354 cell does not). Two built-in mitigations: (a) every mismatch report includes
the hardcoded-coordinate AST scan (logic from experiment_5760's memorization_scan,
re-stated here so the scored path does not import an experiment module that mutates
os.environ at import time); (b) 2-3 transitions are HELD OUT of the tool's test set
and reported aggregate-only, so a memorizing engine shows a visible/held-out gap the
model can see and the loop can select against.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

# Thresholds from experiment_5760's pre-registered memorization detector: ignore small
# literals (action codes / colors, <10); >=3 window-matching large literals -> flagged.
MEMORIZATION_MIN_MATCHES = 3
MEMORIZATION_MIN_COORD = 10

# Bounded report sizes. The whole point of the tools is to SHRINK decode; an unbounded
# report would just move the burn from the think channel to the prompt.
MAX_MISMATCHES_REPORTED = 5
MAX_DIFF_CELLS = 200
MAX_REGION_CELLS = 400
MAX_GOAL_PROBE_GRIDS = 24


def _engine_source(full_source: str) -> str:
    """Source segment of the top-level `engine` function ('' if absent/unparseable)."""
    try:
        tree = ast.parse(full_source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "engine":
            return ast.get_source_segment(full_source, node) or ""
    return ""


def _int_literals(src: str) -> list[int]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            out.append(int(node.value))
    return out


def window_changed_coords(transitions: Sequence[Any]) -> set[int]:
    """Row/col indices (>= MEMORIZATION_MIN_COORD) of cells that changed in the window.

    These are the coordinate constants a memorizing engine would hardcode."""
    coords: set[int] = set()
    for t in transitions:
        try:
            g = np.asarray(t.grid)
            ng = np.asarray(t.next_grid)
        except Exception:
            continue
        if g.shape != ng.shape:
            continue
        rs, cs = np.where(g != ng)
        coords.update(int(v) for v in rs.tolist())
        coords.update(int(v) for v in cs.tolist())
    return {v for v in coords if v >= MEMORIZATION_MIN_COORD}


def memorization_scan(full_source: str, coord_set: set[int]) -> dict[str, Any]:
    """AST scan for hardcoded observed-window coordinates in engine() source.

    Same logic and thresholds as experiment_5760's detector (see module docstring for
    why it is re-stated rather than imported)."""
    eng_src = _engine_source(full_source or "")
    lits = [v for v in _int_literals(eng_src) if v >= MEMORIZATION_MIN_COORD]
    n_match = sum(1 for v in lits if v in coord_set)
    return {
        "engine_source_found": bool(eng_src),
        "n_large_int_literals": len(lits),
        "n_window_coord_literals": int(n_match),
        "is_memorizing": bool(n_match >= MEMORIZATION_MIN_MATCHES),
    }


def holdout_split(transitions: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """Split observed transitions into (visible, held_out).

    The tail is held out because it is the same convention the CEGIS acceptance split
    uses (the model never sees the future). 3 rows held out on a normal window, 2 on a
    short one, 1 below 5 rows, none below 3 -- a tiny window has nothing to spare."""
    rows = list(transitions)
    if len(rows) < 3:
        return rows, []
    if len(rows) < 5:
        h = 1
    elif len(rows) < 12:
        h = 2
    else:
        h = 3
    return rows[:-h], rows[-h:]


def _exec_candidate(code: str, func_name: str) -> tuple[Optional[Callable], Optional[str]]:
    """Compile + exec candidate code, return (function, error). Never raises."""
    try:
        ast.parse(code)
    except SyntaxError as se:
        return None, f"syntax error line {se.lineno}: {se.msg}"
    ns: dict[str, Any] = {}
    try:
        exec(compile(code, "<induction_tool_candidate>", "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 - the exception IS the report
        return None, f"module exec raised {type(exc).__name__}: {exc}"
    fn = ns.get(func_name)
    if not callable(fn):
        return None, f"no top-level `def {func_name}` found"
    return fn, None


@dataclass
class CandidateRecord:
    """One scored engine candidate, tracked for the loop's best-so-far accept."""

    code: str
    visible_mismatches: int
    visible_accuracy: float
    visible_cell_recall: float
    holdout_accuracy: Optional[float]
    holdout_cell_recall: Optional[float]
    is_memorizing: bool
    has_goal: bool


@dataclass
class InductionToolSession:
    """Holds the observed transitions and serves the four tools against them.

    One session per induction call. The session owns the visible/held-out split, the
    memorization coordinate set, and the candidate ledger the loop's best-so-far
    accept reads."""

    transitions: list[Any]
    cell: int = 1
    hud_mask: Any = None
    visible: list[Any] = field(init=False)
    held_out: list[Any] = field(init=False)
    coord_set: set[int] = field(init=False)
    candidates: list[CandidateRecord] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.transitions = list(self.transitions)
        self.visible, self.held_out = holdout_split(self.transitions)
        self.coord_set = window_changed_coords(self.transitions)

    # ---- tool: run_engine_on_transitions ---------------------------------------

    def run_engine_on_transitions(self, code: str) -> dict[str, Any]:
        """Run a candidate engine() on the visible transitions; return a mismatch report.

        Held-out rows are scored AGGREGATE-ONLY: the model sees the score, never the
        cells, so it cannot hardcode its way past them."""
        self.calls.append({"tool": "run_engine_on_transitions"})
        engine, err = _exec_candidate(code, "engine")
        if engine is None:
            return {"ok": False, "error": err}
        # Local imports keep this module import-light: a broken sibling must degrade
        # one tool call, never the whole induce path (same rule as _engine_defects).
        from carnot.agentic.arc_executable_world_model import WorldModelVerifier

        try:
            vr = WorldModelVerifier(list(self.visible), hud_mask=self.hud_mask).score(engine)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"verifier raised {type(exc).__name__}: {exc}"}
        from carnot.agentic.arc_executable_world_model import _bounded_mismatches

        report: dict[str, Any] = {
            "ok": True,
            "n_transitions_tested": int(vr.n),
            "n_correct": int(vr.n_correct),
            "accuracy": round(float(vr.accuracy), 4),
            "cell_recall": round(float(vr.cell_recall), 4),
            "n_engine_raised": int(vr.n_engine_raised),
            "engine_raise_kinds": dict(vr.engine_raise_kinds),
            "mismatches": _bounded_mismatches(list(vr.mismatches), limit=MAX_MISMATCHES_REPORTED),
        }
        # Static defect kinds (truncation, missing return, ...). Advisory; a failure of
        # the checker degrades to [] exactly like the shipped _engine_defects wrapper.
        try:
            from carnot.agentic import arc_engine_static_validation as _sv

            defects = _sv.validate_engine_code(
                code, transitions=list(self.visible), stop_type="eos", required=("engine",)
            )
            report["static_defects"] = sorted({d.kind for d in defects})
        except Exception:
            report["static_defects"] = []
        scan = memorization_scan(code, self.coord_set)
        report["memorization_scan"] = scan
        if scan["is_memorizing"]:
            report["warning"] = (
                "Your engine hardcodes coordinate constants that match observed changed "
                "cells. Hardcoded layouts fail on held-out transitions. Replace the "
                "special cases with a general rule."
            )
        holdout_acc: Optional[float] = None
        holdout_recall: Optional[float] = None
        if self.held_out:
            try:
                hv = WorldModelVerifier(list(self.held_out), hud_mask=self.hud_mask).score(engine)
                holdout_acc = round(float(hv.accuracy), 4)
                holdout_recall = round(float(hv.cell_recall), 4)
                # Aggregate ONLY -- no mismatch detail, by design (see docstring).
                report["held_out"] = {
                    "n_transitions": int(hv.n),
                    "accuracy": holdout_acc,
                    "cell_recall": holdout_recall,
                }
            except Exception as exc:  # noqa: BLE001
                report["held_out"] = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            report["held_out"] = {"n_transitions": 0, "note": "window too small to hold out"}
        self.candidates.append(
            CandidateRecord(
                code=code,
                visible_mismatches=int(vr.n) - int(vr.n_correct),
                visible_accuracy=float(vr.accuracy),
                visible_cell_recall=float(vr.cell_recall),
                holdout_accuracy=holdout_acc,
                holdout_cell_recall=holdout_recall,
                is_memorizing=bool(scan["is_memorizing"]),
                has_goal="def is_level_complete" in code,
            )
        )
        return report

    # ---- tool: query_region ------------------------------------------------------

    def list_transitions(self) -> dict[str, Any]:
        """Compact index of the VISIBLE transitions: what exists, without the grids.

        REQ-ARC-WMTE-6500. This is the retrieval half of the tool set. The other tools
        VERIFY a candidate against data the prompt already contained; this one lets the
        prompt stop containing it. Rendering every transition is the largest single
        driver of prompt size, and prefill is what the K=4 concurrency probe timed out
        on -- so a model that can ask "which transitions exist and which ones changed a
        lot" and then `query_region` only those is cheaper to serve at concurrency.

        Bounded to `self.visible` for the same reason every other tool is: the held-out
        tail is scored aggregate-only, and an index that revealed its shape would leak
        the thing the split exists to protect. Grids are deliberately NOT returned --
        this is an index, and returning them would rebuild the prompt it replaces.
        """
        self.calls.append({"tool": "list_transitions"})
        rows: list[dict[str, Any]] = []
        for i, tr in enumerate(self.visible):
            before = np.asarray(tr.grid)
            after = np.asarray(tr.next_grid)
            h, w = before.shape[:2]
            if before.shape == after.shape:
                changed = int(np.count_nonzero(before != after))
                ch = np.argwhere(before != after)
                bbox = (
                    [
                        int(ch[:, 0].min()),
                        int(ch[:, 1].min()),
                        int(ch[:, 0].max()),
                        int(ch[:, 1].max()),
                    ]
                    if ch.size
                    else None
                )
            else:
                changed, bbox = -1, None  # shape change: -1 means "differs structurally"
            rows.append(
                {
                    "t": i,
                    "action": int(getattr(tr, "action", -1)),
                    "shape": [int(h), int(w)],
                    "changed_cells": changed,
                    "changed_bbox": bbox,
                }
            )
        return {"ok": True, "n_visible": len(rows), "transitions": rows}

    def query_region(
        self, t: int, r0: int, r1: int, c0: int, c1: int, which: str = "before"
    ) -> dict[str, Any]:
        """Plain integer cells of one transition's grid region. Kills RLE-reconstruction burn."""
        self.calls.append({"tool": "query_region"})
        if not (0 <= t < len(self.visible)):
            return {
                "ok": False,
                "error": f"t must be in 0..{len(self.visible) - 1} (visible transitions)",
            }
        tr = self.visible[t]
        grid = np.asarray(tr.next_grid if which == "after" else tr.grid)
        h, w = grid.shape[:2]
        r0, r1 = max(0, int(r0)), min(h, int(r1))
        c0, c1 = max(0, int(c0)), min(w, int(c1))
        if r1 <= r0 or c1 <= c0:
            return {"ok": False, "error": f"empty region after clipping to {h}x{w}"}
        if (r1 - r0) * (c1 - c0) > MAX_REGION_CELLS:
            return {
                "ok": False,
                "error": f"region larger than {MAX_REGION_CELLS} cells; ask a smaller window",
            }
        return {
            "ok": True,
            "t": t,
            "which": "after" if which == "after" else "before",
            "action": int(tr.action),
            "shape": [int(h), int(w)],
            "r0": r0,
            "c0": c0,
            "rows": [[int(v) for v in row] for row in grid[r0:r1, c0:c1]],
        }

    # ---- tool: diff_grids --------------------------------------------------------

    def diff_grids(self, t: int) -> dict[str, Any]:
        """Changed cells of one visible transition: {r, c, before, after} per cell."""
        self.calls.append({"tool": "diff_grids"})
        if not (0 <= t < len(self.visible)):
            return {
                "ok": False,
                "error": f"t must be in 0..{len(self.visible) - 1} (visible transitions)",
            }
        tr = self.visible[t]
        g = np.asarray(tr.grid)
        ng = np.asarray(tr.next_grid)
        if g.shape != ng.shape:
            return {"ok": False, "error": f"shape change {g.shape} -> {ng.shape}"}
        rs, cs = np.where(g != ng)
        cells = [
            {"r": int(r), "c": int(c), "before": int(g[r, c]), "after": int(ng[r, c])}
            for r, c in zip(rs.tolist(), cs.tolist())
        ]
        out: dict[str, Any] = {
            "ok": True,
            "t": t,
            "action": int(tr.action),
            "data": tr.data if isinstance(tr.data, dict) else None,
            "n_changed": len(cells),
            "changed_cells": cells[:MAX_DIFF_CELLS],
        }
        if len(cells) > MAX_DIFF_CELLS:
            out["changed_cells_omitted"] = len(cells) - MAX_DIFF_CELLS
        return out

    # ---- tool: run_goal_on_states ------------------------------------------------

    def run_goal_on_states(self, code: str) -> dict[str, Any]:
        """Run a candidate is_level_complete() over observed grids; report values + failures."""
        self.calls.append({"tool": "run_goal_on_states"})
        goal, err = _exec_candidate(code, "is_level_complete")
        if goal is None:
            return {"ok": False, "error": err}
        from carnot.agentic.arc_engine_call_guard import guarded_call

        grids: list[np.ndarray] = []
        for tr in self.visible:
            grids.append(np.asarray(tr.grid))
            nxt = getattr(tr, "next_grid", None)
            if nxt is not None:
                grids.append(np.asarray(nxt))
        values: list[Any] = []
        n_raised = 0
        first_error = ""
        for g in grids[:MAX_GOAL_PROBE_GRIDS]:
            try:
                values.append(bool(guarded_call(goal, g.copy())))
            except Exception as exc:  # noqa: BLE001
                n_raised += 1
                values.append("raised")
                if not first_error:
                    first_error = f"{type(exc).__name__}: {exc}"[:200]
        distinct = {v for v in values if v != "raised"}
        report: dict[str, Any] = {
            "ok": True,
            "n_grids_probed": len(values),
            "values": values,
            "n_raised": n_raised,
        }
        if first_error:
            report["first_error"] = first_error
        if len(distinct) == 1 and not n_raised:
            # Constancy over everything observed carries no information for the search.
            # A correct-but-unreached win predicate also looks like this; say so.
            report["note"] = (
                f"constant {distinct.pop()} on every observed grid -- carries no signal; "
                "correct only if the game has genuinely never been in a won state"
            )
        return report

    # ---- best-so-far accept ------------------------------------------------------

    def best_candidate(self) -> Optional[CandidateRecord]:
        """The monotone-accept winner: fewest visible mismatches, then held-out accuracy.

        A memorizing candidate loses ties to a non-memorizing one -- the whole reason
        the scan exists is that in-sample fit alone rewards hardcoding."""
        if not self.candidates:
            return None
        return min(
            self.candidates,
            key=lambda c: (
                c.visible_mismatches,
                -(c.holdout_accuracy if c.holdout_accuracy is not None else 0.0),
                1 if c.is_memorizing else 0,
                -c.visible_cell_recall,
            ),
        )


# ---------------------------------------------------------------------------------
# JSON schemas (OpenAI /v1/chat/completions `tools` format; llama.cpp grammar-enforces
# the call JSON server-side, lazily, outside the think block).
# ---------------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_engine_on_transitions",
            "description": (
                "Execute a candidate engine(grid, action, data) on the observed transitions. "
                "Returns accuracy, cell_recall, up to 5 concrete mismatches (true change vs "
                "your prediction), static defects, a hardcoded-coordinate scan, and an "
                "aggregate-only held-out score. Use this instead of simulating grids in your "
                "head. Submit FULL standalone code (import numpy as np + def engine)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python source defining engine(grid, action, data)",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_region",
            "description": (
                "Read a rectangular region of one observed transition's grid as plain integer "
                "rows (no run-length decoding needed). which='before' is the grid the action "
                "was taken in; 'after' is the result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "t": {"type": "integer", "description": "transition index (0-based)"},
                    "r0": {"type": "integer", "description": "first row (inclusive)"},
                    "r1": {"type": "integer", "description": "last row (exclusive)"},
                    "c0": {"type": "integer", "description": "first column (inclusive)"},
                    "c1": {"type": "integer", "description": "last column (exclusive)"},
                    "which": {"type": "string", "enum": ["before", "after"]},
                },
                "required": ["t", "r0", "r1", "c0", "c1"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_grids",
            "description": (
                "List every cell one observed transition changed: {r, c, before, after}, plus "
                "the action taken. Cheap; prefer this over reconstructing deltas mentally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "t": {"type": "integer", "description": "transition index (0-based)"}
                },
                "required": ["t"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_goal_on_states",
            "description": (
                "Execute a candidate is_level_complete(grid) on every observed grid. Returns "
                "the value per grid and any exceptions. A predicate constant across all "
                "observed grids carries no signal for the search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python source defining is_level_complete(grid)",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transitions",
            "description": (
                "Index of the observed transitions WITHOUT their grids: for each, the index t, "
                "the action, the grid shape, how many cells changed, and the bounding box of the "
                "change (changed_cells -1 means the shape itself changed). Call this FIRST to see "
                "what evidence exists, then use query_region or diff_grids on the transitions that "
                "matter. Cheaper than reading every grid."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_NAMES = tuple(s["function"]["name"] for s in TOOL_SCHEMAS)


def dispatch_tool(session: InductionToolSession, name: str, arguments: str) -> dict[str, Any]:
    """Execute one named tool with JSON-encoded arguments. Never raises.

    Returns the tool's report dict, or {"ok": False, "error": ...} on a bad name or
    unparseable arguments -- the error text goes back to the model as the tool result,
    so a malformed call costs one turn, not the induction."""
    try:
        kwargs = json.loads(arguments) if arguments else {}
        if not isinstance(kwargs, dict):
            return {"ok": False, "error": "arguments must be a JSON object"}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"unparseable JSON arguments: {exc}"}
    fn = {
        "run_engine_on_transitions": session.run_engine_on_transitions,
        "query_region": session.query_region,
        "diff_grids": session.diff_grids,
        "run_goal_on_states": session.run_goal_on_states,
        "list_transitions": session.list_transitions,
    }.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown tool {name!r}; available: {list(TOOL_NAMES)}"}
    try:
        return fn(**kwargs)
    except TypeError as exc:
        return {"ok": False, "error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - a tool bug must cost a turn, not the induction
        return {"ok": False, "error": f"tool {name} raised {type(exc).__name__}: {exc}"}


def register_mcp_tools(session: InductionToolSession, server: Any = None) -> Any:
    """DEV-ONLY: expose this session's tools on a FastMCP server.

    Reuses carnot.mcp.server's optional FastMCP import (falls back to its stub when the
    MCP extra is absent), so this works in any dev venv and never adds a dependency to
    the scored path. The live loop does NOT go through this -- it calls dispatch_tool
    directly in-process."""
    from carnot.mcp.server import FastMCP

    srv = server if server is not None else FastMCP("carnot-arc-induction")

    @srv.tool()
    def run_engine_on_transitions(code: str) -> dict[str, Any]:
        return session.run_engine_on_transitions(code)

    @srv.tool()
    def query_region(
        t: int, r0: int, r1: int, c0: int, c1: int, which: str = "before"
    ) -> dict[str, Any]:
        return session.query_region(t, r0, r1, c0, c1, which)

    @srv.tool()
    def diff_grids(t: int) -> dict[str, Any]:
        return session.diff_grids(t)

    @srv.tool()
    def run_goal_on_states(code: str) -> dict[str, Any]:
        return session.run_goal_on_states(code)

    return srv
