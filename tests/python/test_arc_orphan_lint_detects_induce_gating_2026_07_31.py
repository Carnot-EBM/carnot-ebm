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

The fix had two halves:

  * `_is_solver_like` now also matches a module defining `validate_engine_code` or
    `repair_prompt_block` -- the induce-path gating surface, which decides whether the live agent
    gets a usable world model at all.
  * the module went into `ALLOWLIST` with its reason, so the lint stayed green while the orphan
    status became an auditable entry instead of a gap.

SUPERSEDED LATER THE SAME DAY, and the detection rule is why the succession was safe. The module
was WIRED into `LocalGGUFProposer.generate()`'s code-only induce path once the funnel gain it was
waiting on had been measured out-of-sample (13/36 -> 22/36 mechanically-usable engines, p = 0.049;
`docs/research-notes/arc-induce-repeat-penalty-confirm-2026-07-31.md`). The allow-list entry was
removed in that same change, exactly as its own comment instructed. Two tests below therefore
INVERTED -- they now assert reachability and the ABSENCE of an exemption. Their predecessors are
described in their docstrings rather than deleted, because the reason the entry existed is part of
the record.

The detection-rule tests did NOT change, and are the durable half: whether the module is exempted
or reachable, the lint must be able to SEE it. A future edit that quietly drops the rule would
otherwise leave this file green.
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


def test_it_is_now_genuinely_reachable_from_the_live_path():
    """UPDATED 2026-07-31 (same day, later): the module got wired, so this test INVERTED.

    Its predecessor asserted the opposite -- that the module was unreachable, so that the
    allow-list entry recording its orphan status could not become decorative. That test carried
    its own succession plan: "When the module is eventually wired, this test fails and the
    allow-list entry should be removed in the same change." That is what happened. Wiring
    `LocalGGUFProposer._engine_defects` -> `validate_engine_code` into `generate()`'s code-only
    induce path put the module in the ordinary import closure of both live entrypoints, the
    predecessor test failed as designed, and the allow-list entry was removed in that change.

    What justified the wiring was a measurement, not a tidiness preference: 36 attempt-matched
    pairs across 6 games, mechanically-usable engines 13/36 -> 22/36 at p = 0.049, scored
    out-of-sample. Note the narrowness -- that is a VALIDITY and COST result. Engine quality came
    back p = 1.000 on 5 discordant pairs and is not claimed.

    This test is now the one that keeps the wiring honest: if a future refactor drops the import,
    the module silently becomes off-path work again and this fails.
    """
    mod = _lint_module()
    closure = mod._closure(mod.ENTRYPOINTS) | {ep.stem for ep in mod.ENTRYPOINTS}
    assert "arc_engine_static_validation" in closure, (
        "the induce-path defect gate is no longer reachable from the live agent -- either the "
        "import in LocalGGUFProposer._engine_defects was dropped, or the closure no longer "
        "follows it. Off-path gating code produces no live capability (CLAUDE.md ARC Live-Path "
        "Reachability Discipline)."
    )


def test_the_lint_is_green_on_reachability_and_not_on_an_exemption():
    """The green bill must be earned. An allow-list entry would now be a false record.

    This is the load-bearing half. `main() == 0` alone cannot distinguish "reachable" from
    "exempted", and those are opposite facts about whether the live agent can use this code. So
    the assertion is on the ABSENCE of the exemption, with the reachability asserted above.
    """
    mod = _lint_module()
    assert "arc_engine_static_validation" not in mod.ALLOWLIST, (
        "the module is on the live path now; an allow-list entry claiming it is deliberately "
        "unwired would be a stale record of a decision that has been reversed"
    )
    # With no exemption in play, nothing solver-like may be left unreachable.
    closure = mod._closure(mod.ENTRYPOINTS) | {ep.stem for ep in mod.ENTRYPOINTS}
    flagged = [
        p.stem
        for p in sorted(mod.AGENTIC.glob("arc_*.py"))
        if mod._is_solver_like(p) is not None
        and p.stem not in closure
        and p.stem not in mod.ALLOWLIST
    ]
    assert flagged == []


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
