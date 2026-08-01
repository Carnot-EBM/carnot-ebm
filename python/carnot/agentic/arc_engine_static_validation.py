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

*STATUS OF THAT "runs strictly BEFORE" CLAIM -- WIRED 2026-07-31, ordering PROVEN 2026-08-01.*
`LocalGGUFProposer._engine_defects` (`arc_executable_world_model.py`) calls
`validate_engine_code` on the live code-only induce path, and the orphan-lint allow-list entry
was removed in the same change. The separation remains correct as written -- this module returns
defects and a prompt, never an admit/reject decision, so it CANNOT bypass the gate no matter
where it is called from.

The paragraph this replaces said the opposite ("Nothing in the tree calls `validate_engine_code`
... deliberately orphaned and allow-listed"), and stayed false for a day after the wiring landed.
It also set an obligation -- "the wiring change MUST add an integration test asserting
`validate_engine_code` is called BEFORE `WorldModelVerifier.score` on the same candidate" -- that
the wiring change did not discharge. Both are fixed here rather than quietly dropped:
`tests/python/test_arc_dry_run_wallclock_bound_2026_08_01.py::
test_validate_engine_code_runs_BEFORE_the_trust_gate_scores_the_candidate` drives the real
proposer against a scripted server and asserts the call order, so the sequencing is now enforced
by a test rather than described by this paragraph. Worth noting WHY the order matters: this
module reports mechanical defects and the trust gate judges quality, so scoring first grades
broken code as merely WRONG -- exactly what the 2026-07-30 audit found, four of five rejections
being broken code reported as bad predictions.

*EXECUTION IS BOUNDED (2026-08-01).* `dry_run_defects` runs generated code in a KILLABLE
SUBPROCESS with a wall-clock bound, because a non-terminating induced engine wedged the
generation loop for 13 minutes on 2026-07-31 and the shipped induce path reaches the same code.
A signal alarm cannot substitute: the loop's own `except Exception` and `_engine_defects`'
broad catch would absorb it and record a false CLEAN. See `dry_run_defects` for the full
reasoning and `CARNOT_ARC_DRY_RUN_TIMEOUT_S` for the disable switch.

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
import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

__all__ = [
    "EngineDefect",
    "missing_return_defects",
    "truncation_defect",
    "dry_run_defects",
    "dry_run_timeout_default",
    "engine_changes_anything",
    "engine_changes_anything_bounded",
    "engine_inertness_defect",
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


_DRY_RUN_DEFAULT_TIMEOUT_S = 30.0


def dry_run_timeout_default() -> float:
    """Wall-clock bound for EXECUTING generated code, resolved at call time.

    `CARNOT_ARC_DRY_RUN_TIMEOUT_S` overrides it. A value <= 0 disables the bound and restores the
    pre-2026-07-31 in-process behaviour exactly -- kept as a true A/B switch, and as the escape
    hatch if subprocess isolation ever proves worse than the hazard it removes.

    30s is chosen against what a REAL dry run costs, not by guesswork: it executes at most
    `limit=25` transitions of pure Python over 64x64 grids, which is milliseconds. A legitimate
    engine has three orders of magnitude of headroom here. The bound exists to catch
    non-termination, not to race slow-but-finite code, so it is set far above any honest run
    rather than tuned tight.
    """

    raw = os.environ.get("CARNOT_ARC_DRY_RUN_TIMEOUT_S")
    if raw is None:
        return _DRY_RUN_DEFAULT_TIMEOUT_S
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return _DRY_RUN_DEFAULT_TIMEOUT_S


def dry_run_defects(
    code: str,
    transitions: Sequence[Any],
    *,
    limit: int = 25,
    func_name: str = _ENGINE_FN,
    timeout_s: Optional[float] = None,
) -> list[EngineDefect]:
    """Bounded wrapper around the dry run. EXECUTES LLM-WRITTEN CODE, so it is bounded by a
    process that can be killed from outside rather than by anything inside the interpreter.

    THE INCIDENT (2026-07-31). The best-of-N generation loop wedged for 13 minutes in state R at
    32% CPU, no open socket, both GPUs idle -- spinning inside this function's dry run of ft09
    candidate 5, a generated engine that does not terminate. The completion had already arrived,
    so no HTTP timeout applied, and the agent's own wall budget is checked BETWEEN actions rather
    than inside one. The shipped induce path reaches here identically, via
    `LocalGGUFProposer.generate` -> `_engine_defects` -> `validate_engine_code` -> here, so this
    was never a harness-only hazard: a non-terminating induced engine hangs a live episode, and
    on a scored submission that is the whole run.

    WHY NOT A SIGNAL ALARM, which is the cheap and obvious fix. It would be SWALLOWED. The loop
    below wraps every engine invocation in `except Exception` because the exception IS the
    observation it is looking for, so a SIGALRM-raised exception inside the engine call is caught
    and recorded as an ordinary `engine_raised` defect -- or worse, `_engine_defects` upstream
    catches broadly and returns `[]`, which means ACCEPT. That turns a hang into a false CLEAN,
    which is strictly worse than the hang: the hang is at least visible. A signal is also
    main-thread-only and cannot interrupt a tight loop inside a C-level numpy call. A subprocess
    killed from outside can be swallowed by nothing.

    A TIMEOUT IS A DEFECT, NOT AN ERROR, and that distinction is load-bearing. Returning
    `engine_nonterminating` in the ordinary defect list means the existing caller path rejects
    the candidate through machinery that already exists. Raising instead would hit
    `_engine_defects`'s broad `except Exception: return []` and be converted into acceptance --
    the precise failure this function is being changed to prevent. The honest reading is that an
    engine the pipeline cannot finish CHECKING is not an engine the pipeline could have used.

    RESIDUAL RISK, stated rather than hidden. If the subprocess cannot be started at all, this
    falls back to running in-process -- i.e. to the old hazard -- because
    `arc_engine_static_validation` is an optional quality gate and a detector that can take down
    the path it checks is worse than no detector (the same reasoning that justifies
    `_engine_defects`'s swallow). That fallback is recorded in the returned defect's evidence
    when it happens, so it cannot be silent.
    """

    if timeout_s is None:
        timeout_s = dry_run_timeout_default()
    if timeout_s is None or float(timeout_s) <= 0:
        return _dry_run_defects_inprocess(code, transitions, limit=limit, func_name=func_name)

    bounded = _dry_run_defects_subprocess(
        code, transitions, limit=limit, func_name=func_name, timeout_s=float(timeout_s)
    )
    if bounded is not None:
        return bounded
    # Subprocess unavailable -- see RESIDUAL RISK above.
    return _dry_run_defects_inprocess(code, transitions, limit=limit, func_name=func_name)


def _dry_run_defects_inprocess(
    code: str,
    transitions: Sequence[Any],
    *,
    limit: int = 25,
    func_name: str = _ENGINE_FN,
) -> list[EngineDefect]:
    """Run `engine` against transitions the agent has ALREADY observed and report what breaks.

    UNBOUNDED BY CONSTRUCTION -- it executes generated code in this interpreter. Callers should
    reach it through `dry_run_defects`, which supplies the wall-clock bound. It stays a separate
    function so the child process has something to call that does not recurse, and so the
    disable path (`CARNOT_ARC_DRY_RUN_TIMEOUT_S=0`) is exactly the old code rather than an
    approximation of it.

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


def _defect_to_dict(d: EngineDefect) -> dict:
    return {
        "kind": d.kind,
        "detail": d.detail,
        "line": d.line,
        "retryable": bool(d.retryable),
        "repairable": bool(d.repairable),
        "evidence": d.evidence,
    }


def _defect_from_dict(raw: dict) -> EngineDefect:
    return EngineDefect(
        kind=str(raw.get("kind", "unknown")),
        detail=str(raw.get("detail", "")),
        line=raw.get("line"),
        retryable=bool(raw.get("retryable")),
        repairable=bool(raw.get("repairable")),
        evidence=dict(raw.get("evidence") or {}),
    )


@dataclass(frozen=True)
class _ShimTransition:
    """What the child reconstructs a transition as: exactly the three attributes the dry run
    reads, and nothing else.

    WHY NOT PICKLE THE CALLER'S OBJECT (found by testing, not by review). This module's contract
    is duck-typed and says so: "Any object with those three attributes works, so callers can pass
    test doubles." Pickling the object itself silently breaks that -- the child unpickles by
    importing the class, so a locally-defined transition double dies with
    `AttributeError: Can't get attribute 'T'`, and because the child then exits non-zero the
    parent reports `engine_crashed_validator`: a defect blamed on the ENGINE for something that
    is purely an artifact of how the check ships work across a process boundary. The first
    version of this did exactly that, and the first test written against it caught it.

    Passing the three attributes instead preserves the duck-typed contract. It costs nothing in
    fidelity: `grid` is a numpy array and pickles bit-exactly, `action` is an int, and `data` is
    whatever the caller had. Attributes beyond these three are dropped, which is correct -- the
    dry run reads no others, so carrying them would only widen what has to be picklable.
    """

    grid: Any
    action: Any
    data: Any


def _run_isolated_job(
    code: str,
    transitions: Sequence[Any],
    *,
    limit: int,
    func_name: str,
    timeout_s: float,
    job_kind: str,
) -> tuple[str, Any]:
    """Stage a job, run it in a KILLABLE CHILD, and report what happened -- nothing more.

    Extracted 2026-08-01 so that a SECOND check which executes generated code
    (`engine_changes_anything_bounded`) gets the identical process boundary rather than a
    hand-copied approximation of it. Two near-duplicate spawn paths is exactly how one of them
    quietly loses the `CUDA_VISIBLE_DEVICES=""` line, or the `PYTHONPATH` line, and starts
    behaving differently from the one that was reasoned about.

    This function deliberately makes NO judgement about the engine. It returns a status and a
    payload; turning either into an `EngineDefect` is the caller's job, because what a timeout
    MEANS differs by check (for the dry run it is `engine_nonterminating`, a real defect; for
    the inertness probe it is "undetermined", which must not become a rejection).

    status is one of:
      ``"ok"``          -- `payload` is the child's decoded JSON dict.
      ``"timeout"``     -- the child was killed at the wall-clock bound; `payload` is None.
      ``"crashed"``     -- the child exited non-zero; `payload` is ``(returncode, stderr_tail)``.
      ``"unavailable"`` -- the isolation machinery itself could not be used (unpicklable input,
                           spawn refused, unreadable result). This is an INFRASTRUCTURE fact and
                           never evidence about the engine; `payload` is None.
    """

    tmp = None
    try:
        staged = [
            _ShimTransition(grid=t.grid, action=t.action, data=getattr(t, "data", None))
            for t in list(transitions)[: int(limit)]
        ]
        with tempfile.NamedTemporaryFile(prefix="arc_dry_run_", suffix=".pkl", delete=False) as fh:
            pickle.dump(
                {
                    "code": code,
                    "transitions": staged,
                    "limit": int(limit),
                    "func_name": str(func_name),
                    "job_kind": str(job_kind),
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            tmp = Path(fh.name)
    except Exception:  # noqa: BLE001 - cannot stage the job; not a fact about the engine
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        return "unavailable", None

    env = dict(os.environ)
    # The child imports this package; make sure it can, whatever the parent's install layout is.
    pkg_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    # The child executes untrusted code and needs no accelerator. Denying it one also stops a
    # generated engine from grabbing a GPU the live agent is using.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"

    try:
        proc = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "carnot.agentic.arc_engine_static_validation",
                "--dry-run-job",
                str(tmp),
            ],
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            env=env,
        )
    except subprocess.TimeoutExpired:
        # THE INCIDENT'S SIGNATURE. `subprocess.run` kills the child on timeout, so unlike a
        # signal there is nothing left spinning and nothing that can catch this.
        return "timeout", None
    except Exception:  # noqa: BLE001 - spawn refused; infrastructure, not the engine
        return "unavailable", None
    finally:
        tmp.unlink(missing_ok=True)

    if proc.returncode != 0:
        return "crashed", (int(proc.returncode), (proc.stderr or "")[-400:])

    try:
        return "ok", json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - unreadable result; do not guess a verdict
        return "unavailable", None


def _dry_run_defects_subprocess(
    code: str,
    transitions: Sequence[Any],
    *,
    limit: int,
    func_name: str,
    timeout_s: float,
) -> Optional[list[EngineDefect]]:
    """Run the dry run in a killable child. Returns None when the child could not be USED at all.

    The None-vs-list distinction is the whole contract, and it is deliberately narrow. A list --
    including an empty one -- means the check RAN and this is its verdict. `None` means the
    isolation machinery itself was unavailable (no interpreter, spawn refused, unreadable
    result, unpicklable input), which is an infrastructure fact and NOT evidence about the
    engine. Only `None` sends the caller to the in-process fallback; a timeout does not, because
    a timeout IS a real observation about the code.
    """

    status, payload = _run_isolated_job(
        code,
        transitions,
        limit=limit,
        func_name=func_name,
        timeout_s=timeout_s,
        job_kind="dry_run",
    )
    if status == "timeout":
        return [
            EngineDefect(
                kind="engine_nonterminating",
                detail=(
                    f"executing `{func_name}` over the observed transitions did not finish "
                    f"within {float(timeout_s):g}s and the check was killed. An engine whose "
                    f"dry run does not terminate cannot be validated, and cannot be planned "
                    f"with either -- the planner would call it thousands of times."
                ),
                # NOT repairable: there is no exception text to feed back. The model produced
                # code that runs forever, and the actionable response is to re-induce, not to
                # describe a failure we never observed the end of.
                evidence={
                    "timeout_s": float(timeout_s),
                    "transitions_attempted": min(int(limit), len(list(transitions))),
                    "isolation": "subprocess",
                },
            )
        ]
    if status == "crashed":
        # A CRASHED child is a fact about the engine (a segfault or OOM in generated code is a
        # real defect), but it is reported distinctly from a clean verdict so it can never be
        # mistaken for one.
        returncode, stderr_tail = payload
        return [
            EngineDefect(
                kind="engine_crashed_validator",
                detail=(
                    f"the isolated dry run exited with code {returncode} without "
                    f"reporting. Generated code that takes the interpreter down -- a segfault "
                    f"in a native call, or an allocation the OS refused -- is a defect, and is "
                    f"reported here rather than as a passing check."
                ),
                evidence={
                    "returncode": int(returncode),
                    "stderr_tail": stderr_tail,
                    "isolation": "subprocess",
                },
            )
        ]
    if status != "ok":
        return None
    try:
        return [_defect_from_dict(d) for d in payload["defects"]]
    except Exception:  # noqa: BLE001 - unreadable result; do not guess a verdict
        return None


def _dry_run_child_main(job_path: str) -> int:
    """Child entry point. Prints one JSON line: the job's verdict, or nothing on a crash.

    `job_kind` selects which check runs in the child. It defaults to ``"dry_run"`` so that a job
    staged by an older parent still means what it meant.
    """

    with open(job_path, "rb") as fh:
        job = pickle.load(fh)  # noqa: S301 - written by this process's parent, not untrusted input
    kind = str(job.get("job_kind") or "dry_run")
    if kind == "changes_anything":
        print(
            json.dumps(
                _change_census_inprocess(
                    job["code"],
                    job["transitions"],
                    limit=int(job["limit"]),
                    func_name=str(job["func_name"]),
                ),
                default=str,
            )
        )
        return 0
    defects = _dry_run_defects_inprocess(
        job["code"],
        job["transitions"],
        limit=int(job["limit"]),
        func_name=str(job["func_name"]),
    )
    print(json.dumps({"defects": [_defect_to_dict(d) for d in defects]}, default=str))
    return 0


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

    UNBOUNDED, AND MEASURED TO BE (2026-08-01). This executes generated code IN THIS
    INTERPRETER with no wall-clock bound, exactly as `_dry_run_defects_inprocess` does. On the
    ft09 candidate 5 that caused the 2026-07-31 13-minute wedge, `validate_engine_code` returns
    in 30.07s with `engine_nonterminating` while THIS function does not return at all
    (`results/arc_generation_taxonomy_20260801/probe_unbounded_inertness.json`). Any caller on a
    live path must use `engine_changes_anything_bounded` instead. This name is kept, and kept
    unbounded, because it is the reporting helper the docstring above describes and changing its
    semantics under existing callers would be a behaviour change smuggled in as a refactor.
    """
    return _engine_changes_anything_inprocess(code, transitions, limit=limit, func_name=func_name)


def _engine_changes_anything_inprocess(
    code: str, transitions: Sequence[Any], *, limit: int = 25, func_name: str = _ENGINE_FN
) -> Optional[bool]:
    """The unbounded worker. Separate so the child process has something to call that does not
    recurse, mirroring `_dry_run_defects_inprocess`."""
    census = _change_census_inprocess(code, transitions, limit=limit, func_name=func_name)
    return census["changes_anything"]


def _change_census_inprocess(
    code: str, transitions: Sequence[Any], *, limit: int = 25, func_name: str = _ENGINE_FN
) -> dict:
    """`changes_anything`, PLUS how many transitions the engine actually answered usably.

    THE SECOND NUMBER IS WHY THIS EXISTS, and it was found by a test rather than by review.
    `engine_changes_anything` `continue`s past a raise and past a `None` return, because from
    its own point of view neither is a change. So an engine that raises on EVERY transition
    returns False from it -- indistinguishable, at that return value, from an engine that ran
    cleanly and predicted nothing ever changes. Those are different failures with different
    repairs, and only the second one is inertness.

    At the wired call site the ordering already prevents the confusion (`dry_run_defects` reports
    `engine_raised` first, so the inertness probe never sees a raising engine). But
    `engine_inertness_defect` is a public function and must not depend on its caller getting the
    order right to avoid mislabelling. `n_usable_predictions` is what lets it require positive
    evidence that the engine RAN before it concludes the engine does NOTHING.

    COST NOTE: counting requires visiting every transition, so this loses the early exit
    `engine_changes_anything` had on its first observed change. The VERDICT is identical; only
    the work is. At `limit=25` transitions of pure Python that is milliseconds, and it is now
    wrapped in a wall-clock bound in any case.
    """
    import numpy as np

    out = {"changes_anything": None, "n_usable_predictions": 0, "n_tried": 0}
    ns, defect = _exec_namespace(code)
    if defect is not None:
        return out
    assert ns is not None
    engine = ns.get(func_name)
    if not callable(engine):
        return out
    changed = False
    n_usable = 0
    tried = list(transitions)[: int(limit)]
    for t in tried:
        grid = np.asarray(t.grid)
        try:
            pred = engine(grid.copy(), t.action, t.data)
        except Exception:  # noqa: BLE001, S112 - a raise is not a change
            continue
        if pred is None:
            continue
        arr = np.asarray(pred)
        if arr.shape != grid.shape:
            # A wrong-shaped return is not a usable prediction either -- `dry_run_defects`
            # reports it as `engine_wrong_shape`, and it is not evidence about inertness.
            continue
        n_usable += 1
        if not np.array_equal(arr, grid):
            changed = True
    out["changes_anything"] = changed
    out["n_usable_predictions"] = n_usable
    out["n_tried"] = len(tried)
    return out


def engine_changes_anything_bounded(
    code: str,
    transitions: Sequence[Any],
    *,
    limit: int = 25,
    func_name: str = _ENGINE_FN,
    timeout_s: Optional[float] = None,
) -> Optional[bool]:
    """`engine_changes_anything` with the SAME killable-subprocess bound the dry run uses.

    Returns True (something changed), False (nothing ever changed on the transitions tried), or
    None for UNDETERMINED -- the engine could not be run, the child could not be spawned, or the
    probe was killed at the bound.

    THE THREE-VALUED RETURN IS THE POINT. `engine_changes_anything` already returned None for
    "could not run", and a caller that treats None as False turns "we do not know" into "reject
    it". Every consumer of this must branch on `is False`, never on falsiness.
    """
    census = _change_census_bounded(
        code, transitions, limit=limit, func_name=func_name, timeout_s=timeout_s
    )
    if census is None:
        return None
    verdict = census.get("changes_anything")
    return verdict if isinstance(verdict, bool) else None


def _change_census_bounded(
    code: str,
    transitions: Sequence[Any],
    *,
    limit: int = 25,
    func_name: str = _ENGINE_FN,
    timeout_s: Optional[float] = None,
) -> Optional[dict]:
    """`_change_census_inprocess` behind the killable-subprocess bound. None = UNDETERMINED."""
    if timeout_s is None:
        timeout_s = dry_run_timeout_default()
    if timeout_s is None or float(timeout_s) <= 0:
        # Same escape hatch, same meaning, as CARNOT_ARC_DRY_RUN_TIMEOUT_S=0 on the dry run:
        # restore the unbounded in-process behaviour exactly.
        return _change_census_inprocess(code, transitions, limit=limit, func_name=func_name)
    status, payload = _run_isolated_job(
        code,
        transitions,
        limit=limit,
        func_name=func_name,
        timeout_s=float(timeout_s),
        job_kind="changes_anything",
    )
    if status != "ok":
        # timeout / crashed / unavailable all mean UNDETERMINED here. A crash or a hang is a real
        # defect, but it is the DRY RUN's defect to report -- reporting it a second time from the
        # inertness probe would double-count one broken engine as two findings.
        return None
    return payload if isinstance(payload, dict) else None


def engine_inertness_defect(
    code: str,
    transitions: Sequence[Any],
    *,
    limit: int = 25,
    func_name: str = _ENGINE_FN,
    timeout_s: Optional[float] = None,
) -> Optional[EngineDefect]:
    """An `engine_inert` defect when the engine is ESTABLISHED to change nothing; else None.

    WHY THIS IS A DEFECT AND NOT A QUALITY JUDGEMENT -- the distinction this module's docstring
    draws, and the reason inertness sat outside `validate_engine_code` until now. An engine that
    predicts NO action changes ANYTHING is not merely a poor model; it is unusable by the thing
    downstream that consumes it. `plan_in_model` searches for an action sequence that reaches a
    goal state, and against an engine whose successor function is the identity there are no
    distinct states to search, so the planner cannot return a plan no matter how much budget it
    is given. That is a mechanical property of the code, checkable by running it -- the same
    class of fact as "raises" or "returns None", and the same class of fact this module exists
    to report.

    THE MEASUREMENT BEHIND IT (`results/outer_loop_arc_generation_taxonomy_20260801.json`).
    Inertness is the largest single failure class at 26 of 172 gemma-4-31B candidates (15.1%),
    exceeding every code-validity class combined, and it was the only class the live induce path
    took no action on despite the detector being shipped and already imported at that call site.
    Rejecting it is safe in the direction that matters: 0 of 11 inert candidates in the frozen
    corpus were plannable, and all 15 inert cells carrying a held-out score had `change_fidelity`
    EXACTLY 0.0. Nothing measurable was lost by rejecting them.

    STILL NOT PART OF `validate_engine_code`. It is offered as a separate function that a caller
    opts into, and the caller that does so
    (`LocalGGUFProposer._engine_defects`, behind `CARNOT_ARC_INDUCE_REJECT_INERT`) is DEFAULT OFF.
    Two reasons, both load-bearing: `validate_engine_code` is the definition an A/B measures
    usable-engine yield WITH, so changing it would make the measurement circular; and an
    identity engine really does clear every other check in this file, so folding inertness in
    would blur the honest statement that a clean report here is not a quality claim.

    UNDETERMINED IS NOT INERT, and neither is BROKEN. This returns None -- fail-OPEN, i.e.
    accept -- in two distinct situations, and both matter:

      * the probe could not reach a verdict (killed at the bound, child unavailable, the code
        does not import). Rejecting on "we could not tell" would be a fabricated observation,
        and the two ways to get here that really are the engine's fault (it hangs, it crashes
        the validator) are ALREADY rejected upstream by `dry_run_defects`, which runs first.
      * the engine never produced a single usable prediction -- it raised on everything, or
        returned None, or returned a wrongly-shaped array. `engine_changes_anything` reports
        False for that, because from its point of view nothing changed, but it is NOT inertness:
        the engine did not run. Each of those has its own defect kind and its own repair, and
        relabelling one as inertness would hand the model the wrong instruction. Requiring at
        least one usable prediction is what keeps this a POSITIVE finding rather than the
        absence of one. Found by a test, not by review.
    """
    census = _change_census_bounded(
        code, transitions, limit=limit, func_name=func_name, timeout_s=timeout_s
    )
    if census is None or census.get("changes_anything") is not False:
        return None
    n_usable = int(census.get("n_usable_predictions") or 0)
    if n_usable < 1:
        return None
    return EngineDefect(
        kind="engine_inert",
        detail=(
            f"`{func_name}(grid, action, data)` ran on {n_usable} observed transitions and "
            f"returned a grid IDENTICAL to its input every time -- it predicts that no action "
            f"changes anything. A planner cannot search a state graph with one state, so this "
            f"engine cannot produce a plan whatever budget it is given. Model what the action "
            f"actually does to the grid."
        ),
        # Repairable: unlike a hang, we can say exactly what was observed, and the observation
        # ("your engine ran and never changed the grid") is a fact the model can act on.
        repairable=True,
        evidence={
            "transitions_tried": int(census.get("n_tried") or 0),
            "n_usable_predictions": n_usable,
            "changed_on_any": False,
        },
    )


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


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess, asserted by test
    # Sole purpose: be the killable child for `_dry_run_defects_subprocess`. This module is
    # otherwise a library; the entry point exists because executing LLM-written code needs a
    # process boundary that can be terminated from outside, and a boundary is only a boundary if
    # something lives on the far side of it.
    if len(sys.argv) == 3 and sys.argv[1] == "--dry-run-job":
        raise SystemExit(_dry_run_child_main(sys.argv[2]))
    print("usage: python -m carnot.agentic.arc_engine_static_validation --dry-run-job <path>")
    raise SystemExit(2)
