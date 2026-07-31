"""Static + dry-run validation of an LLM-INDUCED world-model engine, BEFORE the trust gate.

WHY THIS EXISTS
===============
The 2026-07-30 activation grid measured the ARC live agent's LLM tier firing on 6 of 6 games
while its output reached the policy on 1 of 6. The 2026-07-30 gate-rejection audit
(`docs/research-notes/arc-gate-rejection-audit-2026-07-30.md`) then read every rejected engine
and found that the rejections were CORRECT -- but that four of the five failures were BROKEN
CODE rather than a bad model of the game:

  * **ft09** -- `engine()` returns `grid.copy()` when `action != 6`, then on the `action == 6`
    path computes two locals and ends in a 1112-of-1144-line comment wall with NO `return`.
    Every click therefore evaluates to `None`. The gate saw "predicted wrong", not "returned
    nothing".
  * **tu93** -- both replicates' engines "scan for the player and fall off the end of `engine()`
    with no return" (audit section 3). Held-out 0/8, cell recall 0.0, in both arms.
  * **lp85** -- generated code raised
    `UnboundLocalError: cannot access local variable 'cell' where it is not associated with a
    value`. The audit established this is the GENERATED code's defect, not a harness miscall:
    a wrong arity or wrong grid type would raise `TypeError`/`AttributeError`. Note WHERE: the
    raise comes out of `_eval_goal` (`arc_llm_reinduction.py:750`), i.e. from
    **`is_level_complete`, not `engine`**, on the level's ROOT grid
    (`reachable_grids_evaluated: 1` pins it there). That is why `dry_run_defects` runs BOTH
    generated functions -- an engine-only dry run would miss lp85 entirely, which is exactly
    the kind of near-miss that makes a clean report untrustworthy.
  * **ft09 round 2** -- the completion hit the 4096-token output cap
    (`missing ('engine','is_level_complete') ... HIT n_predict=4096 OUTPUT LIMIT`). A completion
    that stops at the cap without its required symbols is a MISSING OBSERVATION, not evidence
    about the model.

None of these four is a statement about the model's understanding of the game. Each is
mechanically detectable from the source text alone, or by running the code once against
transitions the agent has already observed. That is what this module does.

WHAT THIS MODULE IS NOT
=======================
**It is not a gate, and it must never be used as one.** It runs strictly BEFORE the semantic
trust gate (`WorldModelVerifier` / `change_gate_decision`) and has no power to admit anything.
Its only outputs are:

  * a defect list, which turns UNUSABLE code into an honest retry, and
  * a repair prompt, which turns a crashing engine into a re-induce with the exception text.

An engine that passes every check here still faces the UNCHANGED trust gate. Nothing in this
module lowers, softens or bypasses a quality threshold. That separation is deliberate: the
Phase-1 sweep (2026-07-31) produced a completion that was accepted, parsed, and returned on
every path, and was the IDENTITY FUNCTION -- `return grid` on both branches. It would clear
every check in this file and it models nothing. Catching that is the change-aware gate's job,
downstream, and this module's clean bill of health is not evidence of quality. See
`engine_changes_anything()` for the one behavioural datum recorded here, which exists so that
callers can report it, NOT so that they can gate on it.

THE FALSE-POSITIVE DIRECTION
============================
A false `missing_return` would REJECT working code -- strictly worse than the status quo, since
a non-returning engine is already caught downstream (as a wrong prediction, with a misleading
reason). So `_falls_through` is deliberately asymmetric: for any construct whose termination
semantics are not obvious from the AST alone (`while True`, `try/finally`, `match`), it assumes
the path TERMINATES and reports nothing. It only flags a fall-through it is confident about.
The cost of that choice is missed detections; the benefit is that a clean report is trustworthy.
`tests/python/test_arc_engine_static_validation.py` pins the asymmetry with explicit cases.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

__all__ = [
    "EngineDefect",
    "missing_return_defects",
    "truncation_defect",
    "dry_run_defects",
    "engine_changes_anything",
    "validate_engine_code",
    "repair_prompt_block",
]


# The engine signature the whole ARC world-model apparatus assumes: `engine(grid, action, data)`
# returning a grid-shaped array. `WorldModelVerifier.score` calls it as
# `engine(t.grid.copy(), t.action, t.data)` and wraps the result in `np.asarray`, which is why a
# `None` return degrades silently into `array(None, dtype=object)` rather than raising.
_ENGINE_FN = "engine"
_GOAL_FN = "is_level_complete"


@dataclass(frozen=True)
class EngineDefect:
    """One mechanically-established defect in a generated engine.

    `kind` is a stable machine-readable token; `detail` is the human sentence that goes into a
    diagnostic or a repair prompt.

    The two booleans encode WHAT A CALLER MAY DO about it, and they are not the same thing:

      * `retryable` -- the observation is incomplete through no fault of the model (the output
        cap truncated it). The right response is to ask again with more room. Scoring a
        truncated completion would be measuring our own budget, not the model.
      * `repairable` -- the code is complete but wrong in a way the model can be TOLD about
        (an exception, a `None` return). The right response is to feed the evidence back and
        re-induce. This is the difference between a veto and a fix.

    A defect may be neither, in which case the caller's only sound move is to reject the
    candidate and let the ordinary retry/fallback path run.
    """

    kind: str
    detail: str
    line: Optional[int] = None
    retryable: bool = False
    repairable: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        where = f" (line {self.line})" if self.line else ""
        return f"{self.kind}{where}: {self.detail}"


# ---------------------------------------------------------------------------
# 1. STATIC: does every path of engine() return?
# ---------------------------------------------------------------------------


def _terminates(stmt: ast.stmt) -> bool:
    """Does control flow provably NOT continue past `stmt` to the next sibling statement?

    Deliberately biased toward answering True (see the module docstring's FALSE-POSITIVE
    DIRECTION section): every uncertain construct is treated as terminating, so a fall-through
    is only reported when the AST makes it unambiguous.
    """
    if isinstance(stmt, (ast.Return, ast.Raise)):
        return True
    # `break`/`continue` transfer control out of the current statement list. Inside a loop body
    # that is not a function-level return, but a body ending in one does not "fall through" to a
    # following sibling either, so for the purposes of a fall-through walk it terminates the list.
    if isinstance(stmt, (ast.Break, ast.Continue)):
        return True
    if isinstance(stmt, ast.If):
        # An `if` without an `else` can always be skipped, so it never terminates. With an
        # `else`, it terminates iff BOTH arms do.
        if not stmt.orelse:
            return False
        return not _falls_through(stmt.body) and not _falls_through(stmt.orelse)
    if isinstance(stmt, ast.With):
        return not _falls_through(stmt.body)
    if isinstance(stmt, (ast.While, ast.For, ast.AsyncFor)):
        # A loop may execute zero times, so in general it does NOT terminate the path -- EXCEPT
        # `while True:` with no `break`, which never exits normally. Anything more subtle
        # (a `while <expr>` that is always true, a `for` over a provably non-empty iterable) is
        # treated as non-terminating only when it is a plain loop; `while True` is special-cased
        # because generated engines really do use it and misreading it would be a false positive.
        if isinstance(stmt, ast.While) and _is_literal_true(stmt.test) and not _has_break(stmt):
            return True
        return False
    if isinstance(stmt, ast.Try):
        # Conservative: a `try` is a control-flow construct whose exit paths depend on which
        # exceptions fire at runtime. Treat it as terminating (i.e. do not flag) unless it is
        # the simple, unambiguous shape where the body and every handler fall through and there
        # is no finally -- in which case the whole statement plainly falls through.
        if stmt.finalbody:
            return True
        parts = [stmt.body + stmt.orelse] + [h.body for h in stmt.handlers]
        return not any(_falls_through(p) for p in parts)
    if isinstance(stmt, ast.Match):
        # Terminates only if there is a catch-all case AND every case body terminates.
        if not stmt.cases:
            return False
        last = stmt.cases[-1].pattern
        catch_all = isinstance(last, ast.MatchAs) and last.pattern is None
        if not catch_all:
            return False
        return not any(_falls_through(c.body) for c in stmt.cases)
    return False


def _is_literal_true(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _has_break(loop: ast.stmt) -> bool:
    """Is there a `break` bound to THIS loop (not to a nested one)?"""
    for child in ast.iter_child_nodes(loop):
        if isinstance(child, (ast.While, ast.For, ast.AsyncFor)):
            continue  # a break in there belongs to the inner loop
        if isinstance(child, ast.Break):
            return True
        if _has_break(child):
            return True
    return False


def _falls_through(body: Sequence[ast.stmt]) -> bool:
    """Can control reach the end of this statement list without returning or raising?"""
    for stmt in body:
        if _terminates(stmt):
            return False
    return True


def _find_function(tree: ast.AST, name: str) -> Optional[ast.FunctionDef]:
    """The definition of `name` that the CALLER WILL ACTUALLY GET: the last TOP-LEVEL one.

    Two rules, and both were mistakes waiting to happen:

    **TOP-LEVEL, not any.** `exec`ing the module puts only module-level definitions in the
    namespace, so a `def engine` nested inside another function is NOT what the verifier calls.
    An earlier version used a bare `ast.walk` and would have graded the nested one, because
    `ast.walk` is breadth-first: every module-level definition is visited before any nested one,
    so "last visited" reliably picked the nested definition whenever one existed. A helper's
    inner `engine` would then have decided the verdict for the real one, in either direction.

    **LAST, not first**, because Python binds the last definition -- and generated code really
    does redefine a function. The ft09 frozen engine carries two `import numpy as np` lines from
    `_combine_world_model`'s concatenation, and a model that writes a draft and then a final
    version leaves both in the file. Grading the first would grade a function that never runs.

    A nested definition is used only when there is no top-level one at all: in that case the
    module defines nothing callable under this name, and reporting on the nested body is more
    informative than reporting nothing.
    """
    top: Optional[ast.FunctionDef] = None
    nested: Optional[ast.FunctionDef] = None
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            top = node
    if top is not None:
        return top
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            nested = node
    return nested


def missing_return_defects(code: str, func_name: str = _ENGINE_FN) -> list[EngineDefect]:
    """Report a defect if `func_name` can reach the end of its body without returning.

    THE OBSERVED FAILURE THIS CATCHES. ft09's live engine (preserved at
    `results/arc_induce_budget_20260731/upstream_ft09_cells/on__ft09__s1__world_model.py.frozen`)
    is `if action != 6: return grid.copy()` followed by an `action == 6` path that assigns
    `r_start`/`c_start` and then ends in comments. `engine(grid, 6, data)` returns `None`, so
    every click in the game is modelled as "nothing observable happened, and also the prediction
    has no shape". tu93's two engines fail the same way.

    A `return` with no value (or an explicit `return None`) is reported under the same kind: the
    caller receives `None` either way, and `np.asarray(None)` is a 0-d object array, not a grid.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [
            EngineDefect(
                kind="syntax_error",
                detail=f"generated code does not parse: {exc.msg}",
                line=exc.lineno,
            )
        ]
    fn = _find_function(tree, func_name)
    if fn is None:
        return [
            EngineDefect(
                kind="missing_function",
                detail=f"no `def {func_name}` in the generated code",
            )
        ]
    out: list[EngineDefect] = []
    if _falls_through(fn.body):
        out.append(
            EngineDefect(
                kind="missing_return",
                detail=(
                    f"`{func_name}` can reach the end of its body without returning, so it "
                    f"returns None on that path. Every branch must return a grid-shaped array."
                ),
                line=fn.end_lineno,
                repairable=True,
                evidence={"function": func_name, "def_line": fn.lineno},
            )
        )
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and (
            node.value is None
            or (isinstance(node.value, ast.Constant) and node.value.value is None)
        ):
            out.append(
                EngineDefect(
                    kind="returns_none_literal",
                    detail=(
                        f"`{func_name}` has an explicit `return None` (or a bare `return`); the "
                        f"caller needs a grid-shaped array on every path."
                    ),
                    line=node.lineno,
                    repairable=True,
                    evidence={"function": func_name},
                )
            )
    return out


# ---------------------------------------------------------------------------
# 2. TRUNCATION: an incomplete observation, not a bad model
# ---------------------------------------------------------------------------


def truncation_defect(
    *,
    stop_type: Optional[str],
    code: str,
    required: Sequence[str],
    budget: Optional[int] = None,
) -> Optional[EngineDefect]:
    """Was this completion cut off by the output cap before it finished the required symbols?

    THE OBSERVED FAILURE THIS CATCHES. ft09's round-2 induction reported
    `missing ('engine','is_level_complete') in output ... HIT n_predict=4096 OUTPUT LIMIT`, and
    round 1 did the same. The shipped `generate()` loop already DETECTS this (it appends the
    limit diagnostic to its message) but then does the same thing it does for any other unusable
    output: retry at the same budget, then give up. That treats a truncated completion as
    evidence about the model when it is evidence about the budget.

    `stop_type` is llama-server's own `stop_type` field as recorded by
    `LocalGGUFProposer._record_completion_diagnostics` -- `"limit"` when generation stopped
    because `n_predict` was exhausted.

    Returns None when the completion stopped for any other reason, or when it stopped at the cap
    but nevertheless contains every required symbol (in which case nothing was lost that we
    needed, and the ordinary checks apply).
    """
    if stop_type != "limit":
        return None
    missing = [fn for fn in required if f"def {fn}" not in code]
    if not missing:
        return None
    return EngineDefect(
        kind="truncated_before_required_symbols",
        detail=(
            f"the completion hit the {budget if budget else 'n_predict'}-token output cap "
            f"before defining {tuple(missing)}. This is a missing observation, not a bad model: "
            f"re-ask with more room rather than scoring what arrived."
        ),
        retryable=True,
        evidence={"missing": list(missing), "budget": budget},
    )


# ---------------------------------------------------------------------------
# 3. DRY RUN: execute the engine once against transitions we have already seen
# ---------------------------------------------------------------------------


def _exec_namespace(code: str) -> tuple[Optional[dict], Optional[EngineDefect]]:
    """Execute the generated module body, returning its namespace or the defect that stopped it.

    Import-time failures are their own defect kind: a `NameError` at module scope means the
    engine never even existed, which is a different repair from an engine that exists and
    crashes on a particular transition.
    """
    import numpy as np

    ns: dict[str, Any] = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<induced_world_model>", "exec"), ns)  # noqa: S102
    except SyntaxError as exc:
        return None, EngineDefect(
            kind="syntax_error",
            detail=f"generated code does not parse: {exc.msg}",
            line=exc.lineno,
        )
    except Exception as exc:  # noqa: BLE001 - any import-time failure is the datum
        return None, EngineDefect(
            kind="module_exec_raised",
            detail=f"executing the generated module raised {type(exc).__name__}: {exc}",
            repairable=True,
            evidence={"exception": type(exc).__name__},
        )
    return ns, None


def dry_run_defects(
    code: str,
    transitions: Sequence[Any],
    *,
    limit: int = 25,
    func_name: str = _ENGINE_FN,
) -> list[EngineDefect]:
    """Run `engine` against transitions the agent has ALREADY observed and report what breaks.

    THE OBSERVED FAILURE THIS CATCHES. lp85's round-3 engine raised
    `UnboundLocalError: cannot access local variable 'cell' where it is not associated with a
    value`. Nothing about the source text reveals it -- the name is assigned on some path -- so
    only running the code finds it. Because the exception text names the variable, it is
    directly repairable: feeding it back is the difference between a veto and a fix.

    This is a DEFECT scan, not a scoring pass. It reports only things that are wrong regardless
    of how good the model is: an exception, a `None` return, a non-array return, a shape that
    does not match the input grid. It deliberately does NOT report a wrong prediction -- being
    wrong about the game is what the trust gate downstream is for, and reporting it here would
    make this module a second, weaker gate.

    `transitions` are `arc_executable_world_model.Transition`-shaped: `.grid`, `.action`,
    `.data`. Any object with those three attributes works, so callers can pass test doubles.
    """
    import numpy as np

    ns, defect = _exec_namespace(code)
    if defect is not None:
        return [defect]
    assert ns is not None
    engine = ns.get(func_name)
    if not callable(engine):
        return [
            EngineDefect(
                kind="missing_function",
                detail=f"the generated module defines no callable `{func_name}`",
            )
        ]

    out: list[EngineDefect] = []
    seen_kinds: set[str] = set()
    for i, t in enumerate(list(transitions)[: int(limit)]):
        grid = np.asarray(t.grid)
        try:
            pred = engine(grid.copy(), t.action, t.data)
        except Exception as exc:  # noqa: BLE001 - the exception IS the observation
            kind = "engine_raised"
            if kind not in seen_kinds:
                seen_kinds.add(kind)
                out.append(
                    EngineDefect(
                        kind=kind,
                        detail=(
                            f"`{func_name}(grid, action={t.action}, data={t.data!r})` raised "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        repairable=True,
                        evidence={
                            "exception": type(exc).__name__,
                            "message": str(exc)[:300],
                            "action": int(t.action),
                            "transition_index": i,
                        },
                    )
                )
            continue
        if pred is None:
            kind = "engine_returned_none"
            if kind not in seen_kinds:
                seen_kinds.add(kind)
                out.append(
                    EngineDefect(
                        kind=kind,
                        detail=(
                            f"`{func_name}(grid, action={t.action}, ...)` returned None. "
                            f"np.asarray(None) is a 0-d object array, so downstream this reads "
                            f"as a wrong prediction rather than as absent code."
                        ),
                        repairable=True,
                        evidence={"action": int(t.action), "transition_index": i},
                    )
                )
            continue
        arr = np.asarray(pred)
        if arr.shape != grid.shape:
            kind = "engine_wrong_shape"
            if kind not in seen_kinds:
                seen_kinds.add(kind)
                out.append(
                    EngineDefect(
                        kind=kind,
                        detail=(
                            f"`{func_name}(grid, action={t.action}, ...)` returned shape "
                            f"{arr.shape}; the input grid is {grid.shape}. The engine must "
                            f"return a grid of the same shape."
                        ),
                        repairable=True,
                        evidence={
                            "got": list(arr.shape),
                            "want": list(grid.shape),
                            "action": int(t.action),
                            "transition_index": i,
                        },
                    )
                )
    out.extend(_goal_defects(ns, transitions, limit=limit))
    return out


def _goal_defects(
    ns: dict, transitions: Sequence[Any], *, limit: int, func_name: str = _GOAL_FN
) -> list[EngineDefect]:
    """Run `is_level_complete` on observed grids and report mechanical failures.

    THE OBSERVED FAILURE THIS CATCHES. lp85's round-3 candidate raised
    `UnboundLocalError: cannot access local variable 'cell' ...` out of `_eval_goal` on the
    level's ROOT grid. The satisfiability search caught it and recorded it as
    `goal_predicate_error`, i.e. as a fact about SATISFIABILITY -- after a whole search had been
    set up around a predicate that could never be called. Catching it here, at induce time,
    turns it into a repair with the variable name in hand.

    Reported here: a raise, and a return value that is not usable as a truth value. NOT reported
    here: a predicate that is constantly True or constantly False, or one that is true at the
    root. Those are degeneracy judgements and belong to `_goal_satisfiability_check`, which
    already makes them with a counterfactual this module does not have.
    """
    import numpy as np

    goal = ns.get(func_name)
    if not callable(goal):
        # A missing goal predicate is a `required`-symbol matter for the caller, not a dry-run
        # defect: the split-induce path deliberately generates it in a second call.
        return []
    out: list[EngineDefect] = []
    seen: set[str] = set()
    grids = []
    for t in list(transitions)[: int(limit)]:
        grids.append(np.asarray(t.grid))
        nxt = getattr(t, "next_grid", None)
        if nxt is not None:
            grids.append(np.asarray(nxt))
    for j, g in enumerate(grids):
        try:
            res = goal(g.copy())
        except Exception as exc:  # noqa: BLE001 - the exception IS the observation
            if "goal_raised" not in seen:
                seen.add("goal_raised")
                out.append(
                    EngineDefect(
                        kind="goal_raised",
                        detail=(
                            f"`{func_name}(grid)` raised {type(exc).__name__}: {exc} on an "
                            f"observed grid. It must be callable on every grid the game "
                            f"produces."
                        ),
                        repairable=True,
                        evidence={
                            "exception": type(exc).__name__,
                            "message": str(exc)[:300],
                            "grid_index": j,
                        },
                    )
                )
            continue
        try:
            bool(res)
        except Exception as exc:  # noqa: BLE001 - e.g. a non-scalar numpy array
            if "goal_not_boolean" not in seen:
                seen.add("goal_not_boolean")
                out.append(
                    EngineDefect(
                        kind="goal_not_boolean",
                        detail=(
                            f"`{func_name}(grid)` returned {type(res).__name__} which cannot be "
                            f"used as a truth value ({exc}). Return a single True/False."
                        ),
                        repairable=True,
                        evidence={"returned_type": type(res).__name__, "grid_index": j},
                    )
                )
    return out


def engine_changes_anything(
    code: str, transitions: Sequence[Any], *, limit: int = 25, func_name: str = _ENGINE_FN
) -> Optional[bool]:
    """Does the engine EVER produce an output different from its input, on observed transitions?

    RECORDED, NOT GATED. The Phase-1 budget sweep (2026-07-31) found that the completions which
    scored best under every structural check -- accepted, parsing, returning on every path,
    19-of-25 held-out exact -- were the IDENTITY FUNCTION, and that 19 of ft09's 25 transitions
    are no-ops, so "nothing ever changes" gets every one of them right. An identity engine
    clears `missing_return_defects` and `dry_run_defects` trivially.

    This function exists so a caller can REPORT that fact honestly next to a clean defect
    report, so that "no defects found" is never mistaken for "the engine is any good". It is
    deliberately NOT part of `validate_engine_code`'s defect list: degeneracy is a quality
    judgement, and quality judgements belong to the trust gate, which measures it properly
    (`change_fidelity`, `cell_recall`, no-op hallucination rate) over a held-out split.

    Returns None when the engine could not be run at all.
    """
    import numpy as np

    ns, defect = _exec_namespace(code)
    if defect is not None:
        return None
    assert ns is not None
    engine = ns.get(func_name)
    if not callable(engine):
        return None
    for t in list(transitions)[: int(limit)]:
        grid = np.asarray(t.grid)
        try:
            pred = engine(grid.copy(), t.action, t.data)
        except Exception:  # noqa: BLE001, S112 - a raise is not a change
            continue
        if pred is None:
            continue
        arr = np.asarray(pred)
        if arr.shape == grid.shape and not np.array_equal(arr, grid):
            return True
    return False


# ---------------------------------------------------------------------------
# 4. The single entry point + the repair prompt
# ---------------------------------------------------------------------------


def validate_engine_code(
    code: str,
    *,
    transitions: Optional[Sequence[Any]] = None,
    stop_type: Optional[str] = None,
    required: Sequence[str] = (_ENGINE_FN, _GOAL_FN),
    budget: Optional[int] = None,
    dry_run_limit: int = 25,
) -> list[EngineDefect]:
    """Every check, cheapest first, ordered so the most actionable defect is reported first.

    Order matters for the caller's decision, not just for speed:

      1. **truncation** -- if the completion was cut off, nothing else about it is informative.
         A truncated file "has no return" because it has no end, and saying so would be wrong.
      2. **static** -- pure AST, no execution, catches ft09 and tu93.
      3. **dry run** -- executes the engine, catches lp85. Skipped when no transitions are
         supplied (the caller may not have them yet).

    Returns [] when nothing mechanically detectable is wrong. That is NOT a quality claim --
    see `engine_changes_anything`.
    """
    trunc = truncation_defect(stop_type=stop_type, code=code, required=required, budget=budget)
    if trunc is not None:
        return [trunc]
    defects = missing_return_defects(code, func_name=_ENGINE_FN)
    if any(d.kind in {"syntax_error", "missing_function"} for d in defects):
        return defects
    if transitions:
        defects = defects + dry_run_defects(code, transitions, limit=dry_run_limit)
    return defects


def repair_prompt_block(
    defects: Sequence[EngineDefect],
    *,
    code: Optional[str] = None,
    max_code_chars: int = 4000,
) -> str:
    """The text to append to a re-induce prompt so the model can FIX what we measured.

    Only `repairable` defects go in. A truncation is not repairable by telling the model about
    it -- the fix is more budget, and describing the truncation would just consume more of it.

    The block states the observation and asks for the corrected file; it deliberately does not
    hint at what the game's mechanic might be. Suggesting a mechanic would put our guess in the
    model's mouth and then grade the model on it.

    `max_code_chars` EXISTS BECAUSE OF THE FAILURE THIS MODULE IS FOR. The completions being
    repaired are frequently repetition-loop runaways -- ft09's live engine is 1112 of 1144 lines
    of duplicated comment, and the Phase-1 sweep measured that at matched seed a doubled budget
    leaves the set of DISTINCT emitted lines unchanged while the emitted length doubles. Echoing
    such a wall back verbatim would spend thousands of prompt tokens re-showing the model the
    exact text it is stuck repeating, which is the last thing a repetition loop needs. The head
    of the code carries the structure worth keeping; the tail is the wall. The truncation is
    marked in the prompt so the model is not told a partial file is the whole file.
    """
    actionable = [d for d in defects if d.repairable]
    if not actionable:
        return ""
    lines = [
        "",
        "YOUR PREVIOUS ANSWER WAS RUN AGAINST THE OBSERVED TRANSITIONS AND FAILED MECHANICALLY.",
        "These are execution facts, not opinions about the game:",
        "",
    ]
    for d in actionable:
        where = f" (line {d.line})" if d.line else ""
        lines.append(f"  * {d.kind}{where} -- {d.detail}")
    lines += [
        "",
        "Fix ONLY these defects. Keep whatever the previous answer got right about the game's",
        "mechanic. `engine(grid, action, data)` must return a numpy array of the SAME SHAPE as",
        "`grid` on EVERY path, and must not raise on any observed transition.",
    ]
    if code:
        body = code.strip()
        if len(body) > int(max_code_chars):
            body = (
                body[: int(max_code_chars)]
                + f"\n# ... [{len(body) - int(max_code_chars)} further characters omitted;"
                f" the answer was much longer than this] ..."
            )
        lines += ["", "The code that failed:", "```python", body, "```"]
    return "\n".join(lines) + "\n"
