#!/usr/bin/env python3
"""Lint: ARC world-model / solver modules MUST be reachable from the LIVE agent path.

Origin: 2026-06-22 operator directive. The hazard-aware nav world model (arc_nav_world_model) was built
in the OUTER LOOP to "solve" tu93's charger levels, but it was imported only by scripts/experiments/* and
its test -- unreachable from any live solve path -- AND it re-solved a level the live agent already
deep-solves (tu93 -> L3) via plain verifier-routed search. So the effort produced no live capability and
no live efficiency. The operator's directive: "make sure this doesn't happen again."

This lint catches that failure mode mechanically. A module under python/carnot/agentic/ is SOLVER-LIKE if
its name matches *world_model* OR it defines a recognized solver surface (escalating_deepen; a class with
BOTH an `engine` and an `is_lethal` method; or a `plan_in_model` function). Every solver-like module MUST
be in the transitive import closure of the LIVE entrypoints:

  * scripts/arc_loop_solve.py                     (offline development twin: GameAdapter + OfflineSolver)
  * python/carnot/agentic/arc_competition_agent.py (the SCORED Kaggle agent: E3AgentPolicy cascade)

Function-level imports count (the live path imports the hazard pruner lazily inside solve_adaptered). A
solver-like module that is NOT reachable AND not explicitly ALLOWLISTED (with a reason) is an ORPHAN --
the lint exits non-zero so the work is wired in (or consciously allow-listed) before it lands.

Run: python3 scripts/arc_orphan_solver_lint.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENTIC = REPO / "python" / "carnot" / "agentic"

# LIVE entrypoints whose transitive import closure defines "reachable from the live agent".
ENTRYPOINTS = [
    REPO / "scripts" / "arc_loop_solve.py",
    AGENTIC / "arc_competition_agent.py",
]

# Explicit allow-list: solver-like modules that are intentionally NOT on the live path (with a reason).
# Keep this SMALL and justified -- an entry here is an assertion that the orphan is deliberate, not the
# failure mode this lint guards against. Format: module_stem -> reason.
ALLOWLIST = {
    # A pre-existing offline research prototype surfaced when this lint was introduced (2026-06-22): it is
    # imported only by its own experiment + test (not on any live path). Allow-listed so the lint can gate
    # NEW orphans without first requiring a re-wire of this pre-existing experiment. Flagged for future
    # review in docs/research-notes/arc-mechanism-parity-and-hazard-salvage-2026-06-22.md.
    "arc_execution_guided_world_model": "exp3979 prototype; imported only by its experiment + test",
    # 2026-07-31. The Phase-2 induce-validation module: AST return-path check, truncation
    # detection, and a dry run of the generated engine over the observed transitions. It is
    # DELIBERATELY unwired -- wiring it edits `arc_executable_world_model.py`/
    # `arc_llm_reinduction.py` and marks 12 registered artifacts stale, a price that should be
    # paid for a PROVEN funnel gain and what is proven so far is a diagnosis (the shipped
    # `generate()` accepts a defective candidate 13 times in 15) plus an UNDECIDED repair arm
    # (repair 3 vs control 2, exact two-sided sign test p = 1.000). Listed here rather than left
    # undetected because the lint's other solver-like tests do not match this module's shape, so
    # before this entry its orphan status was INVISIBLE -- the guard returned a clean bill on a
    # module that is genuinely unreachable from both live entrypoints. Retire this entry when the
    # module is wired into the pre-gate path.
    #
    # RETIRED 2026-07-31, on the condition the entry itself names. The module is now imported by
    # `LocalGGUFProposer._engine_defects`, which `generate()` calls on the code-only induce path,
    # so it is reachable from BOTH live entrypoints on the ordinary closure and needs no
    # exemption. The funnel gain that entry was waiting for was measured out-of-sample over 36
    # paired attempts on 6 games: mechanically-usable engines 13/36 -> 22/36, p = 0.049. Note
    # what that does and does not license -- the gain is on VALIDITY and COST, not on engine
    # quality, whose comparison came back p = 1.000 and could not have reached significance at
    # that n. Wiring a defect gate is justified by the former alone.
}


def _module_to_path(mod: str) -> Path | None:
    """Map a carnot.agentic.X dotted module to its file, if it is an agentic module."""
    parts = mod.split(".")
    if parts[:2] == ["carnot", "agentic"] and len(parts) >= 3:
        return AGENTIC / (parts[2] + ".py")
    return None


def _imports_of(path: Path) -> set[str]:
    """All agentic module STEMS imported anywhere in `path` (module- or function-level)."""
    out: set[str] = set()
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level >= 1:
                # RELATIVE imports from within carnot.agentic:
                #   from .X import ...      (level=1, module="X")     -> sibling module X
                #   from . import X, Y      (level=1, module=None)    -> sibling modules X, Y
                # (deeper levels point outside carnot.agentic; ignore.)
                if node.level == 1 and node.module:
                    if (AGENTIC / (node.module.split(".")[0] + ".py")).exists():
                        out.add(node.module.split(".")[0])
                elif node.level == 1 and node.module is None:
                    for alias in node.names:
                        if (AGENTIC / (alias.name + ".py")).exists():
                            out.add(alias.name)
            elif node.module:
                # from carnot.agentic.X import ...   OR   from carnot.agentic import X, Y
                p = _module_to_path(node.module)
                if p is not None:
                    out.add(p.stem)
                elif node.module == "carnot.agentic":
                    for alias in node.names:
                        if (AGENTIC / (alias.name + ".py")).exists():
                            out.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                p = _module_to_path(alias.name)
                if p is not None:
                    out.add(p.stem)
    return out


def _closure(entrypoints: list[Path]) -> set[str]:
    """Transitive set of agentic module stems reachable from the entrypoints."""
    seen: set[str] = set()
    stack: list[Path] = []
    for ep in entrypoints:
        for stem in _imports_of(ep):
            stack.append(AGENTIC / (stem + ".py"))
    while stack:
        path = stack.pop()
        if not path.exists() or path.stem in seen:
            continue
        seen.add(path.stem)
        for stem in _imports_of(path):
            if stem not in seen:
                stack.append(AGENTIC / (stem + ".py"))
    return seen


def _is_solver_like(path: Path) -> str | None:
    """Return a reason string if the module is solver-like (and thus must be live-reachable), else None."""
    if "world_model" in path.stem:
        return "name matches *world_model*"
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "escalating_deepen",
            "go_explore_solve",
            "plan_in_model",
        ):
            return f"defines solver function {node.name}()"
        # INDUCE-PATH GATING SURFACE (2026-07-31). A module that validates or repairs a GENERATED
        # engine sits on the same critical path as a solver -- it decides whether the live agent
        # gets a usable world model at all -- but matches none of the tests above: no
        # `*world_model*` in its name, no solver function, no `.engine`/`.is_lethal` class. So
        # `arc_engine_static_validation` was invisible to this lint and the lint reported a clean
        # bill while the module was genuinely unreachable from BOTH live entrypoints. Detecting
        # the shape is what makes its orphan status a recorded, allow-listed decision instead of
        # a silent gap; the module itself is allow-listed above with its reason.
        if isinstance(node, ast.FunctionDef) and node.name in (
            "validate_engine_code",
            "repair_prompt_block",
        ):
            return f"defines induce-path gating function {node.name}()"
        if isinstance(node, ast.ClassDef):
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            if "engine" in methods and "is_lethal" in methods:
                return f"class {node.name} has both .engine and .is_lethal (a world-model solver surface)"
        # A module that consumes a world-model solver class (e.g. a standalone pruner/planner built on
        # HazardAwareNavWorldModel) is solver-adjacent and must itself be live-reachable.
        if isinstance(node, ast.ImportFrom) and node.module and "nav_world_model" in node.module:
            names = {a.name for a in node.names}
            if names & {"HazardAwareNavWorldModel", "InducedNavWorldModel"}:
                return "imports a NavWorldModel solver class"
    return None


def main() -> int:
    closure = _closure(ENTRYPOINTS)
    # The entrypoints themselves + their own stems are trivially reachable.
    closure |= {ep.stem for ep in ENTRYPOINTS}
    orphans: list[tuple[str, str]] = []
    for path in sorted(AGENTIC.glob("arc_*.py")):
        reason = _is_solver_like(path)
        if reason is None:
            continue
        if path.stem in closure or path.stem in ALLOWLIST:
            continue
        orphans.append((path.stem, reason))

    if orphans:
        print("ORPHANED ARC SOLVER / WORLD-MODEL MODULES (not reachable from the live agent path):")
        for stem, reason in orphans:
            print(f"  - {stem}  ({reason})")
        print()
        print("Each must be wired into the live path (reachable from scripts/arc_loop_solve.py or")
        print(
            "python/carnot/agentic/arc_competition_agent.py) OR added to ALLOWLIST with a reason."
        )
        print("See CLAUDE.md 'ARC Live-Path Reachability Discipline'.")
        return 1
    print(
        f"OK: all solver-like ARC modules are reachable from the live agent path "
        f"({len(closure)} modules in the live closure)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
