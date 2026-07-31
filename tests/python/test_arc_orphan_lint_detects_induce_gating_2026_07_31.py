"""The orphan lint must SEE induce-path gating modules, not just solver-shaped ones.

WHY. `scripts/arc_orphan_solver_lint.py` enforces CLAUDE.md's ARC Live-Path Reachability
Discipline: an ARC solver module the live agent cannot import produces no live capability, so it
must be reachable or explicitly allow-listed. Its `_is_solver_like` test keyed on three shapes --
a `*world_model*` filename, a solver function (`escalating_deepen` / `go_explore_solve` /
`plan_in_model`), or a class with both `.engine` and `.is_lethal`.

`python/carnot/agentic/arc_engine_static_validation.py` (2026-07-31, REQ-ARC-WMTE-6052) matches
NONE of those, and is imported by zero live-path files. So the guard returned a clean bill on a
module that is genuinely unreachable from both live entrypoints -- a false-negative in exactly the
guard that exists to prevent silent off-path work. The module IS deliberately unwired, which is a
legitimate state; what was wrong is that the decision was invisible rather than recorded.

The fix has two halves and BOTH are pinned here, because either alone is useless:

  * `_is_solver_like` now also matches a module defining `validate_engine_code` or
    `repair_prompt_block` -- the induce-path gating surface, which decides whether the live agent
    gets a usable world model at all.
  * the module is in `ALLOWLIST` with its reason, so the lint stays green while the orphan status
    is now an auditable entry instead of a gap.

The second test is the load-bearing one: it removes the allow-list entry in memory and asserts the
lint FLAGS the module. Without it, a future edit that quietly drops the detection rule would leave
this file green -- the whole point is that the module is detectable, not that the lint passes.
"""

from __future__ import annotations

import importlib.util
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
LINT = REPO / "scripts" / "arc_orphan_solver_lint.py"
TARGET = REPO / "python" / "carnot" / "agentic" / "arc_engine_static_validation.py"


def _lint_module():
    spec = importlib.util.spec_from_file_location("_arc_orphan_lint_under_test", LINT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_validation_module_is_recognised_as_something_the_lint_must_account_for():
    """It is not solver-SHAPED, so the pre-2026-07-31 rules missed it entirely."""
    mod = _lint_module()
    assert TARGET.exists(), "the module under discussion must exist for this test to mean anything"
    reason = mod._is_solver_like(TARGET)
    assert reason is not None, (
        "arc_engine_static_validation must be visible to the orphan lint; if this fails the lint "
        "is again reporting a clean bill on an unreachable induce-path gate"
    )
    assert "validate_engine_code" in reason


def test_it_is_genuinely_unreachable_so_the_allow_list_entry_is_not_decorative():
    """If the module were on the live path the allow-list entry would be stale and misleading.

    This asserts the FACT the entry claims -- deliberately unwired -- rather than trusting the
    comment. When the module is eventually wired, this test fails and the allow-list entry should
    be removed in the same change.
    """
    mod = _lint_module()
    closure = mod._closure(mod.ENTRYPOINTS) | {ep.stem for ep in mod.ENTRYPOINTS}
    assert "arc_engine_static_validation" not in closure


def test_the_allow_list_entry_is_what_keeps_the_lint_green():
    """Remove the entry and the lint must flag -- proving the pass is an explicit decision."""
    mod = _lint_module()
    assert "arc_engine_static_validation" in mod.ALLOWLIST
    reason = mod.ALLOWLIST["arc_engine_static_validation"]
    # The discipline demands a REASON, not a bare exemption.
    assert len(reason) > 40 and "unwired" in reason

    closure = mod._closure(mod.ENTRYPOINTS) | {ep.stem for ep in mod.ENTRYPOINTS}
    without = {k: v for k, v in mod.ALLOWLIST.items() if k != "arc_engine_static_validation"}
    flagged = [
        p.stem
        for p in sorted(mod.AGENTIC.glob("arc_*.py"))
        if mod._is_solver_like(p) is not None and p.stem not in closure and p.stem not in without
    ]
    assert flagged == ["arc_engine_static_validation"]


def test_the_new_rule_does_not_widen_into_a_false_positive_sweep():
    """A detection rule that matches half the package would just be noise.

    `validate_engine_code` / `repair_prompt_block` are defined in exactly one agentic module, so
    the rule adds precisely one module to the lint's scope. If a future refactor spreads those
    names around, this fails and the rule needs narrowing rather than a bigger allow-list.
    """
    mod = _lint_module()
    matched = [
        p.stem
        for p in sorted(mod.AGENTIC.glob("arc_*.py"))
        if (r := mod._is_solver_like(p)) is not None and "induce-path gating" in r
    ]
    assert matched == ["arc_engine_static_validation"]


def test_the_lint_passes_as_shipped():
    """The end state: green, with the orphan recorded rather than invisible."""
    mod = _lint_module()
    assert mod.main() == 0
