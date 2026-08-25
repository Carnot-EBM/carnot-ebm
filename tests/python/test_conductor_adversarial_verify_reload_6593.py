"""The fabrication gate must not run a stale copy of the linter (exp6593 incident).

The conductor runs as ONE long-lived `--loop` process. A plain
`from adversarial_verify import verify_artifact` therefore binds whatever copy Python
cached at first use. On 2026-08-25 the running conductor (pid 4109305, started
2026-08-24 17:53:50) stamped
`flagged_adversarial` on exp6593 using a linter copy over 14 hours old -- quarantining an
honest 1.16s replay under a rule that the artifact's OWN commit had already fixed.

The dangerous direction is the other one: a fabrication check added to the linter is
equally inert until the process restarts, so the gate silently checks less than the code
on disk says it does.

Spec refs: REQ-VERIFY-6593, SCENARIO-VERIFY-6593-STALE-GATE.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONDUCTOR = PROJECT_ROOT / "scripts" / "research_conductor.py"
VERIFIER = PROJECT_ROOT / "scripts" / "adversarial_verify.py"


def test_req_verify_6593_cached_import_is_stale_but_reload_is_not(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-VERIFY-6593-STALE-GATE: reproduce the staleness, then defeat it.

    Proves the mechanism rather than asserting it: a second plain import returns the
    cached module, and only an explicit reload observes the edit on disk.
    """

    module_name = "_exp6593_stale_gate_probe"
    target = tmp_path / f"{module_name}.py"
    target.write_text("FLOOR_SECONDS = 60.0\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    probe = importlib.import_module(module_name)
    assert probe.FLOOR_SECONDS == 60.0

    # A linter fix lands while the long-lived process keeps running.
    target.write_text("FLOOR_SECONDS = 0.0001\n", encoding="utf-8")

    stale = importlib.import_module(module_name)
    assert stale.FLOOR_SECONDS == 60.0, "a plain re-import must return the cached copy"

    fresh = importlib.reload(probe)
    assert fresh.FLOOR_SECONDS == 0.0001, "reload must observe the landed fix"

    sys.modules.pop(module_name, None)


def test_req_verify_6593_cached_verifier_alias_refreshes_before_stamping(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-VERIFY-6593-STALE-GATE: the actual gate entrypoint refreshes itself.

    Load a private copy of the real verifier, retain exactly the stale function alias
    the conductor retains, then land a changed implementation on disk.  The stale
    alias must delegate to the new implementation.  Removing the source-refresh
    boundary makes this RED while leaving the production conductor untouched.
    """

    module_name = "_exp6593_adversarial_verify_refresh_probe"
    target = tmp_path / f"{module_name}.py"
    original = VERIFIER.read_text(encoding="utf-8")
    target.write_text(original, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    try:
        verifier = importlib.import_module(module_name)
        stale_alias = verifier.verify_artifact
        target.write_text(
            original
            + "\n\ndef _verify_artifact_impl(path, *, declared=None):\n"
            + "    return {'implementation': 'fresh-on-disk'}\n",
            encoding="utf-8",
        )
        importlib.invalidate_caches()

        assert stale_alias(tmp_path / "unused.json") == {"implementation": "fresh-on-disk"}
    finally:
        sys.modules.pop(module_name, None)


def test_req_verify_6593_partial_verifier_edit_keeps_last_known_good_alias(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-VERIFY-6593-STALE-GATE: incomplete source cannot disable the gate."""

    module_name = "_exp6593_adversarial_verify_partial_edit_probe"
    target = tmp_path / f"{module_name}.py"
    original = VERIFIER.read_text(encoding="utf-8")
    target.write_text(original, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    try:
        verifier = importlib.import_module(module_name)
        stale_alias = verifier.verify_artifact
        artifact = tmp_path / "artifact.json"
        artifact.write_text('{"schema": "blocked_precondition"}', encoding="utf-8")
        expected = stale_alias(artifact)

        target.write_text("def incomplete(:\n", encoding="utf-8")
        importlib.invalidate_caches()

        assert stale_alias(artifact) == expected
    finally:
        sys.modules.pop(module_name, None)


def _adversarial_pass_source() -> str:
    """The `research_step` region that runs the fabrication gate."""
    source = CONDUCTOR.read_text(encoding="utf-8")
    anchor = "import adversarial_verify as _av_module"
    start = source.find(anchor if anchor in source else "adversarial_verify import")
    assert start != -1, "could not locate the conductor's adversarial-verify pass"
    return source[start : start + 1600]


def test_req_verify_6593_conductor_reloads_the_linter_before_stamping() -> None:
    """SCENARIO-VERIFY-6593-STALE-GATE: the caller refreshes too, not only the callee.

    The verifier now refreshes its own implementation, which covers the code INSIDE
    `verify_artifact`. It cannot refresh anything the conductor binds at module scope,
    so the conductor's own reload is a separate guarantee and needs its own test.
    Structural, because the real call runs inside a loop a unit test cannot enter.
    """

    region = _adversarial_pass_source()
    assert "importlib.reload(_av_module)" in region, (
        "the conductor must reload adversarial_verify before stamping; a plain import "
        "binds a copy cached when the loop process started"
    )
    assert "_av_module.verify_artifact" in region, (
        "the reloaded module must be the one actually called, not a stale alias"
    )


def test_req_verify_6593_reload_failure_cannot_skip_the_whole_gate() -> None:
    """SCENARIO-VERIFY-6593-STALE-GATE: a partial file must not fail the gate open.

    A sibling agent writing the linter makes reload raise SyntaxError. Handled by the
    outer `except`, that would skip the gate AND the pre-existing-stamp fallback, so a
    transient partial file would silently pass every task. The reload carries its own
    handler and falls back to the cached module.
    """

    tree = ast.parse(CONDUCTOR.read_text(encoding="utf-8"))

    def _is_reload(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reload"
            and any(isinstance(a, ast.Name) and a.id == "_av_module" for a in node.args)
        )

    guarded = [
        try_node
        for try_node in ast.walk(tree)
        if isinstance(try_node, ast.Try)
        # Only the try's OWN body counts. A reload nested in a deeper statement of a
        # wider try would satisfy a naive walk while still failing the whole gate open.
        and any(_is_reload(n) for stmt in try_node.body for n in ast.walk(stmt))
        and len(try_node.body) <= 2
    ]
    assert guarded, (
        "importlib.reload(_av_module) must sit in its own narrow try, not the outer "
        "one: the outer handler would skip the gate AND the pre-existing-stamp fallback"
    )
    assert any(
        isinstance(h.name, str) and h.name == "_reload_exc" for t in guarded for h in t.handlers
    ), "reload failure must be caught locally and fall back to the cached module"


def test_req_verify_6593_conductor_source_parses_after_the_change() -> None:
    """SCENARIO-VERIFY-6593-STALE-GATE: the edited conductor still compiles."""

    ast.parse(CONDUCTOR.read_text(encoding="utf-8"))
