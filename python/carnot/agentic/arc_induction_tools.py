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
import os
import re
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
MAX_FIND_OBJECTS = 32
MAX_PREDICATE_CODE_CHARS = 2048
MAX_FIND_OBJECT_RESPONSE_BYTES = 8192
FIND_OBJECT_PREDICATE_TIMEOUT_S = 0.25

# Tool-gap capture bound (REQ-ARC-WMTE-6770). Events feed an offline ledger,
# so the per-session record is capped and the overflow is counted, not lost.
MAX_TOOL_GAP_EVENTS = 20


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
    # Tool-gap evidence (REQ-ARC-WMTE-6770): the model asked for a capability
    # the active tool set does not serve. See record_tool_gap for the kinds.
    tool_gap_events: list[dict[str, Any]] = field(default_factory=list)
    tool_gap_events_dropped: int = 0
    # Candidate enablement FROZEN at session creation (adversarial review
    # 2026-08-29, F5/F7): dispatch, payload, prompt, and stats all read this
    # one snapshot, so a mid-run env or registry mutation — including one made
    # by model-executed code — cannot change what this session serves or
    # make the record disagree with what actually ran.
    enabled_candidates: tuple[str, ...] = field(init=False)
    rejected_candidates: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.transitions = list(self.transitions)
        self.visible, self.held_out = holdout_split(self.transitions)
        self.coord_set = window_changed_coords(self.transitions)
        self.enabled_candidates, self.rejected_candidates = _candidate_names_from_env()

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

    # ---- tool: find_objects -----------------------------------------------------

    def find_objects(
        self,
        t: int,
        which: str,
        predicate_code: str,
        max_objects: int,
    ) -> dict[str, Any]:
        """Find color components and filter them with bounded generated code.

        The model receives compact object facts instead of every object cell. The
        source, call time, object count, and JSON response all have explicit bounds.
        These bounds stop a retrieval call from rebuilding an unbounded grid prompt.
        """
        self.calls.append({"tool": "find_objects"})
        if not (0 <= int(t) < len(self.visible)):
            return {
                "ok": False,
                "error": f"t must be in 0..{len(self.visible) - 1} (visible transitions)",
            }
        if which not in {"before", "after"}:
            return {"ok": False, "error": "which must be 'before' or 'after'"}
        if len(predicate_code) > MAX_PREDICATE_CODE_CHARS:
            return {
                "ok": False,
                "error": f"predicate_code exceeds {MAX_PREDICATE_CODE_CHARS} characters",
            }

        from carnot.agentic.arc_engine_call_guard import guarded_call

        try:
            predicate, err = guarded_call(
                _exec_candidate,
                predicate_code,
                "accept",
                timeout_s=FIND_OBJECT_PREDICATE_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 - generated code failure is the tool result
            return {"ok": False, "error": f"predicate raised {type(exc).__name__}: {exc}"}
        if predicate is None:
            return {"ok": False, "error": err}

        from carnot.agentic.arc_color_blob_salience import connected_color_blobs

        transition = self.visible[int(t)]
        grid = np.asarray(transition.grid if which == "before" else transition.next_grid)
        try:
            blobs = connected_color_blobs(grid)
        except Exception as exc:  # noqa: BLE001 - malformed fixture becomes a bounded error
            return {"ok": False, "error": f"object scan raised {type(exc).__name__}: {exc}"}

        limit = min(MAX_FIND_OBJECTS, max(1, int(max_objects)))
        matches: list[dict[str, Any]] = []
        for blob in blobs:
            obj = {
                "color": int(blob.color),
                "pixel_count": int(blob.pixel_count),
                "bbox": [int(value) for value in blob.bbox],
                "centroid": [round(float(value), 4) for value in blob.centroid],
                "height": int(blob.height),
                "width": int(blob.width),
            }
            try:
                accepted = bool(
                    guarded_call(
                        predicate,
                        dict(obj),
                        timeout_s=FIND_OBJECT_PREDICATE_TIMEOUT_S,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - generated code failure is the tool result
                return {"ok": False, "error": f"predicate raised {type(exc).__name__}: {exc}"}
            if accepted:
                matches.append(obj)

        objects = matches[:limit]
        result: dict[str, Any] = {
            "ok": True,
            "t": int(t),
            "which": which,
            "predicate_applied": True,
            "n_components_scanned": len(blobs),
            "n_objects_matched": len(matches),
            "objects": objects,
            "truncated": len(matches) > len(objects),
            "response_bytes": 0,
        }
        # The normal shape is already below the byte cap. Keep a second bound so a
        # later object-field addition cannot silently make the response unbounded.
        while True:
            for _ in range(3):
                result["response_bytes"] = len(json.dumps(result).encode("utf-8"))
            if result["response_bytes"] <= MAX_FIND_OBJECT_RESPONSE_BYTES:
                break
            if not result["objects"]:
                return {"ok": False, "error": "find_objects response bound is too small"}
            result["objects"].pop()
            result["truncated"] = True
        return result

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
            "name": "find_objects",
            "description": (
                "Find same-color connected components in one observed before/after grid, then "
                "filter compact object facts with bounded Python predicate code. The source "
                "must define accept(obj) and may read color, pixel_count, bbox, centroid, "
                "height, and width. Results and generated-code execution are bounded."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "t": {"type": "integer", "description": "transition index (0-based)"},
                    "which": {"type": "string", "enum": ["before", "after"]},
                    "predicate_code": {
                        "type": "string",
                        "description": "Python source defining accept(obj) -> bool",
                    },
                    "max_objects": {
                        "type": "integer",
                        "description": f"requested result cap; clamped to {MAX_FIND_OBJECTS}",
                    },
                },
                "required": ["t", "which", "predicate_code", "max_objects"],
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

# ---------------------------------------------------------------------------------
# Tool-gap capture + curated candidate tools (REQ-ARC-WMTE-6770).
#
# WHY. TOOL_SCHEMAS is a closed, hand-authored set. Before this block, a model
# call naming a tool that does not exist was answered with an error and then
# FORGOTTEN: only a conflated counter survived (tool_call_parse_failures mixes
# unknown names with malformed JSON), so nobody could ask "what tool did the
# model need and not have?". record_tool_gap keeps the identity of that demand.
#
# INTRODUCING a tool stays a HUMAN act. The gap ledger names the demand; a
# human authors the tool and registers it here as a CANDIDATE, default off.
# CARNOT_ARC_INDUCE_CANDIDATE_TOOLS (comma-separated exact names) enables
# registered candidates for a measurement run; with it unset the active set is
# the same TOOL_SCHEMAS object as before, byte-identical. This mirrors the
# supervisor-arm rule: selection over a curated set, never model generation.
# ---------------------------------------------------------------------------------

CANDIDATE_TOOLS_ENV = "CARNOT_ARC_INDUCE_CANDIDATE_TOOLS"

# name -> {"schema": OpenAI-format tool schema, "factory": session -> callable}.
# Human-authored entries only. Empty is the honest starting state: no live run
# has yet demanded a tool that does not exist (measured 2026-08-29 over 633
# recorded calls; see docs/research-notes/arc-tool-gap-feedback-2026-08-29.md).
CANDIDATE_TOOLS: dict[str, dict[str, Any]] = {}


def register_candidate_tool(
    schema: dict[str, Any],
    factory: Callable[["InductionToolSession"], Callable[..., dict[str, Any]]],
) -> str:
    """Register one HUMAN-AUTHORED candidate tool. Returns its name.

    Refuses a name collision with a core tool: shadowing a shipped tool would
    silently change measured behaviour under a flag meant only to ADD."""
    name = str(schema["function"]["name"])
    if name in TOOL_NAMES:
        raise ValueError(f"candidate tool {name!r} collides with a core tool")
    CANDIDATE_TOOLS[name] = {"schema": schema, "factory": factory}
    return name


def _candidate_names_from_env() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(enabled, rejected) candidate names from the env flag.

    Rejected = named in the env but not registered. Returned rather than
    dropped so a typo in the flag is visible in every run's stats instead of
    silently disabling the tool it meant to enable."""
    raw = os.environ.get(CANDIDATE_TOOLS_ENV, "")
    wanted = tuple(n for n in (s.strip() for s in raw.split(",")) if n)
    enabled = tuple(n for n in wanted if n in CANDIDATE_TOOLS)
    rejected = tuple(n for n in wanted if n not in CANDIDATE_TOOLS)
    return enabled, rejected


def enabled_candidate_names() -> tuple[str, ...]:
    return _candidate_names_from_env()[0]


def active_tool_schemas() -> list[dict[str, Any]]:
    """The tool set this process serves: core plus enabled candidates.

    With the env flag unset this returns the TOOL_SCHEMAS object itself, so
    the default request payload and prompt text cannot drift by construction."""
    enabled = enabled_candidate_names()
    if not enabled:
        return TOOL_SCHEMAS
    return [*TOOL_SCHEMAS, *(CANDIDATE_TOOLS[n]["schema"] for n in enabled)]


def active_tool_names() -> tuple[str, ...]:
    return tuple(s["function"]["name"] for s in active_tool_schemas())


def active_tool_schemas_for(session: "InductionToolSession") -> list[dict[str, Any]]:
    """The tool set for ONE session, from its frozen enablement snapshot.

    The loop serves payload, prompt, and dispatch from this, so what a run
    advertises is what it dispatches, for the whole run — a mid-run env or
    registry mutation changes nothing (adversarial review 2026-08-29, F5/F7)."""
    enabled = tuple(n for n in getattr(session, "enabled_candidates", ()) if n in CANDIDATE_TOOLS)
    if not enabled:
        return TOOL_SCHEMAS
    return [*TOOL_SCHEMAS, *(CANDIDATE_TOOLS[n]["schema"] for n in enabled)]


def active_tool_names_for(session: "InductionToolSession") -> tuple[str, ...]:
    return tuple(s["function"]["name"] for s in active_tool_schemas_for(session))


# Event-content bounds (adversarial review 2026-08-29, F9). The name and the
# argument keys are MODEL-CONTROLLED text headed for a durable ledger and a
# human-pasted markdown heading; the error field was already capped and these
# were not — an asymmetry a single hostile call could exploit.
MAX_TOOL_GAP_NAME_CHARS = 120
MAX_TOOL_GAP_ARG_KEYS = 20


def _record_unknown_tool(
    session: "InductionToolSession", name: str, keys: Optional[list[str]]
) -> None:
    record_tool_gap(
        session,
        {
            "kind": "unknown_tool",
            "requested_tool": str(name)[:MAX_TOOL_GAP_NAME_CHARS],
            "argument_keys": (
                None
                if keys is None
                else [str(k)[:MAX_TOOL_GAP_NAME_CHARS] for k in keys[:MAX_TOOL_GAP_ARG_KEYS]]
            ),
        },
    )


def record_tool_gap(session: "InductionToolSession", event: dict[str, Any]) -> None:
    """Retain one tool-gap event on the session, bounded.

    Two kinds. "unknown_tool": the model called a name outside the active set
    -- the strongest mechanical signal that a tool is missing. "bad_arguments":
    the model imagined a different signature for a tool that exists."""
    if len(session.tool_gap_events) >= MAX_TOOL_GAP_EVENTS:
        session.tool_gap_events_dropped += 1
        return
    session.tool_gap_events.append(event)


# ---------------------------------------------------------------------------------
# SELFPARSE transport (REQ-ARC-WMTE-6730). The scored vLLM server is launched with no
# tool-parser flags: a request carrying a `tools` field returns HTTP 400, and with
# `--tool-call-parser hermes` the model's calls stay unlifted TEXT, because Qwen3.8
# emits the Qwen3-coder XML convention (measured, offplay_out5 tool_transport_probe).
# These two functions remove the server dependency: the schemas travel as PROMPT TEXT
# so no `tools` field is ever sent, and the loop parses the XML itself.
# ---------------------------------------------------------------------------------

# Parameter types per tool, derived from TOOL_SCHEMAS so the two can never disagree.
_PARAM_TYPES: dict[str, dict[str, str]] = {
    s["function"]["name"]: {
        p: str(spec.get("type", "string"))
        for p, spec in (s["function"].get("parameters", {}).get("properties", {}) or {}).items()
    }
    for s in TOOL_SCHEMAS
}

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_NAME_RE = re.compile(r"<function=([\w.\-]+)\s*>")
_PARAMETER_RE = re.compile(r"<parameter=([\w.\-]+)\s*>(.*?)</parameter>", re.DOTALL)


def _trim_param_value(value: str) -> str:
    """Drop the single wrapping newline the XML convention puts around a value.

    Only ONE leading and ONE trailing newline: a code parameter may legitimately
    begin with indented lines, so a full strip() would corrupt it."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _param_types_for(name: str) -> dict[str, str]:
    """Parameter types for one tool: the core table first, then the candidate
    registry, so an enabled candidate's XML calls coerce by its own schema.

    Gated on ENABLEMENT, not mere registration (adversarial review 2026-08-29,
    F6): a registered-but-dark candidate must not change coercion, or the
    default path stops being byte-identical the moment anything registers."""
    if name in _PARAM_TYPES:
        return _PARAM_TYPES[name]
    if name not in enabled_candidate_names():
        return {}
    entry = CANDIDATE_TOOLS.get(name)
    if entry is None:
        return {}
    props = entry["schema"]["function"].get("parameters", {}).get("properties", {}) or {}
    return {p: str(spec.get("type", "string")) for p, spec in props.items()}


def _coerce_param(name: str, param: str, value: str) -> Any:
    """Schema-typed coercion of one XML parameter value.

    The XML carries raw text; dispatch_tool expects JSON-typed arguments. A value
    that fails coercion is passed through as text so dispatch reports the bad
    argument back to the model -- a malformed call costs one turn, never the run."""
    kind = _param_types_for(name).get(param, "string")
    raw = _trim_param_value(value)
    if kind == "integer":
        try:
            return int(raw.strip())
        except ValueError:
            return raw
    if kind == "number":
        try:
            return float(raw.strip())
        except ValueError:
            return raw
    if kind == "boolean":
        return raw.strip().lower() in ("true", "1", "yes")
    return raw


def parse_xml_tool_calls(content: str) -> tuple[list[dict[str, Any]], int, int]:
    """Parse Qwen3-coder XML tool calls out of a plain text completion.

    Returns (tool_calls, n_blocks_seen, n_blocks_unparsed). Each parsed call is in
    the same OpenAI shape the server would have lifted -- {"id", "type", "function":
    {"name", "arguments": <JSON string>}} -- so the loop's dispatch path is byte-for-
    byte the one the server-lifted path already exercises.

    Only text AFTER the last </think> is scanned: a call the model merely sketched
    while reasoning is not a call it made. An unterminated block (length-truncated
    mid-call) counts as seen-but-unparsed rather than dispatching half a payload."""
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    calls: list[dict[str, Any]] = []
    n_blocks = 0
    n_unparsed = 0
    for i, m in enumerate(_TOOL_CALL_BLOCK_RE.finditer(content)):
        n_blocks += 1
        block = m.group(1)
        fname = _FUNCTION_NAME_RE.search(block)
        if not fname:
            n_unparsed += 1
            continue
        args = {
            pm.group(1): _coerce_param(fname.group(1), pm.group(1), pm.group(2))
            for pm in _PARAMETER_RE.finditer(block)
        }
        calls.append(
            {
                "id": f"selfparse_{i}",
                "type": "function",
                "function": {"name": fname.group(1), "arguments": json.dumps(args)},
            }
        )
    # A trailing <tool_call> with no closing tag is a truncated call: count it so the
    # parse-rate stat sees it, but never dispatch a possibly half-transmitted payload.
    if "<tool_call>" in (content.rsplit("</tool_call>", 1)[-1] if n_blocks else content):
        n_blocks += 1
        n_unparsed += 1
    return calls, n_blocks, n_unparsed


_LOOSE_NAME_RES = (
    # <function=NAME>, <function="NAME">, <function NAME>, <function=NAME()> --
    # the shapes the strict parser refuses but a demanded name still sits in.
    re.compile(r"<function[=\s\"']+([\w.\-]+)"),
    # Server-lifted failure shape left as text: {"name": "NAME", "arguments": ...}
    re.compile(r"\"name\"\s*:\s*\"([\w.\-]+)\""),
)

MAX_LOOSE_NAMES = 5


def loose_tool_call_names(content: str) -> list[str]:
    """Tool names sitting in UNPARSED tool-call text, best-effort and bounded.

    Adversarial review 2026-08-29 (F4): the strict transports count a block the
    model wrote malformed, and the demanded name — the tool-gap signal — was in
    the refused text the whole time. This never dispatches anything; it only
    lets the loop keep the name as gap evidence. Text before the final
    </think> is skipped for the same reason parse_xml_tool_calls skips it."""
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    names: list[str] = []
    for rx in _LOOSE_NAME_RES:
        for m in rx.finditer(content):
            name = m.group(1)[:MAX_TOOL_GAP_NAME_CHARS]
            if name and name not in names:
                names.append(name)
            if len(names) >= MAX_LOOSE_NAMES:
                return names
    return names


def render_tool_schemas_for_prompt(schemas: Optional[list[dict[str, Any]]] = None) -> str:
    """The tool schemas as PROMPT TEXT, plus the exact call format to write.

    The loop passes its session's FROZEN schema list so prompt and dispatch
    cannot disagree mid-run; with no argument this renders the env-derived
    active set (the pre-freeze behaviour, kept for direct callers).

    Rendered from TOOL_SCHEMAS so prompt and dispatch can never drift apart. This is
    what lets a selfparse request omit the `tools` field entirely -- the only payload
    shape the flag-less scored server accepts."""
    lines = ["AVAILABLE TOOLS (schemas):"]
    # Active set, not the bare core constant: an enabled candidate tool must be
    # announced in the prompt, or the model can never call it (REQ-ARC-WMTE-6770).
    for s in schemas if schemas is not None else active_tool_schemas():
        fn = s["function"]
        props = fn.get("parameters", {}).get("properties", {}) or {}
        required = set(fn.get("parameters", {}).get("required", []) or [])
        sig = ", ".join(
            f"{p}: {spec.get('type', 'string')}" + ("" if p in required else " (optional)")
            for p, spec in props.items()
        )
        lines.append(f"- {fn['name']}({sig})")
        lines.append(f"    {fn['description']}")
    lines.append(
        "\nTOOL CALL FORMAT. This backend lifts no structured tool calls; write the call "
        "as plain text, EXACTLY this shape, then stop and wait for the result:\n"
        "<tool_call>\n"
        "<function=TOOL_NAME>\n"
        "<parameter=PARAM_NAME>\n"
        "VALUE\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "One <parameter=...> block per argument. The result arrives in the next user "
        "message inside <tool_response>...</tool_response>."
    )
    return "\n".join(lines)


def dispatch_tool(session: InductionToolSession, name: str, arguments: str) -> dict[str, Any]:
    """Execute one named tool with JSON-encoded arguments. Never raises.

    Returns the tool's report dict, or {"ok": False, "error": ...} on a bad name or
    unparseable arguments -- the error text goes back to the model as the tool result,
    so a malformed call costs one turn, not the induction."""
    known = name in TOOL_NAMES or name in getattr(session, "enabled_candidates", ())
    try:
        kwargs = json.loads(arguments) if arguments else {}
        if not isinstance(kwargs, dict):
            if not known:
                _record_unknown_tool(session, name, None)
            return {"ok": False, "error": "arguments must be a JSON object"}
    except json.JSONDecodeError as exc:
        # The NAME needed no parsing: an unknown name with malformed JSON is
        # still tool demand, and improvising an unseen tool is exactly when a
        # model writes malformed arguments (adversarial review 2026-08-29, F3).
        if not known:
            _record_unknown_tool(session, name, None)
        return {"ok": False, "error": f"unparseable JSON arguments: {exc}"}
    fn = {
        "run_engine_on_transitions": session.run_engine_on_transitions,
        "query_region": session.query_region,
        "diff_grids": session.diff_grids,
        "run_goal_on_states": session.run_goal_on_states,
        "find_objects": session.find_objects,
        "list_transitions": session.list_transitions,
    }.get(name)
    if (
        fn is None
        and name in getattr(session, "enabled_candidates", ())
        and name in CANDIDATE_TOOLS
    ):
        # Curated candidate tool, enabled for this SESSION (REQ-ARC-WMTE-6770;
        # frozen snapshot, see enabled_candidates). The factory is
        # human-authored code: a bug in it must cost one turn, never the
        # induction -- same contract as a tool body below.
        try:
            fn = CANDIDATE_TOOLS[name]["factory"](session)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": f"candidate tool {name} setup raised {type(exc).__name__}: {exc}",
            }
    if fn is None:
        # THE tool-gap signal: the model wrote a call for a name the active set
        # does not serve. Keep its identity; the count alone cannot say what
        # tool was wanted (REQ-ARC-WMTE-6770).
        _record_unknown_tool(session, name, sorted(kwargs))
        return {
            "ok": False,
            "error": f"unknown tool {name!r}; available: {list(active_tool_names_for(session))}",
        }
    try:
        return fn(**kwargs)
    except TypeError as exc:
        # The model imagined a different signature for a real tool. Retained as
        # gap evidence; a TypeError raised INSIDE a tool body lands here too,
        # so the offline analyzer treats this kind as noisier than unknown_tool.
        record_tool_gap(
            session,
            {"kind": "bad_arguments", "tool": name, "error": str(exc)[:200]},
        )
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

    @srv.tool()
    def find_objects(t: int, which: str, predicate_code: str, max_objects: int) -> dict[str, Any]:
        return session.find_objects(t, which, predicate_code, max_objects)

    return srv
