"""Regression tests for the sandbox's runtime import guard.

Spec coverage: REQ-AUTO-004, REQ-AUTO-009

Origin: 2026-07-31 security audit.  `run_in_sandbox` executed hypothesis code
with the real `__builtins__`, so the AST-based blocklist -- which only inspects
`ast.Import` / `ast.ImportFrom` nodes -- could be walked around by never writing
the word `import`:

    def run(d):
        m = __import__('subprocess')
        return {'out': m.run(['echo','sandbox-escape'], capture_output=True).stdout}

That ran a real shell command.  Every test below is the audit's PoC turned into
a regression test, so the hole cannot silently reopen.

Read `sandbox.py`'s module docstring before extending these: the sandbox is a
guardrail against accidental misuse, NOT a security boundary, and
`test_reflective_escape_is_still_open` deliberately asserts a known-unclosed
hole rather than pretending otherwise.
"""

from carnot.autoresearch.sandbox import SandboxConfig, run_in_sandbox


def _run(body: str):
    """Execute a hypothesis whose `run()` body is `body` (4-space indented)."""
    return run_in_sandbox(f"def run(d):\n{body}\n", {})


class TestDynamicImportBypassIsClosed:
    """REQ-AUTO-009: the blocklist must hold against `__import__`, not just `import`."""

    def test_static_import_still_blocked(self) -> None:
        """The original AST path must keep working -- this is the control."""
        result = _run("    import os\n    return {}")
        assert not result.success
        assert "os" in (result.error or "")

    def test_dunder_import_of_blocked_module_is_refused(self) -> None:
        """The audit's primary bypass: __import__ is not an ast.Import node."""
        result = _run("    return {'cwd': __import__('os').getcwd()}")
        assert not result.success, "__import__('os') bypassed the blocklist"
        assert "Blocked import" in (result.error or "")

    def test_dunder_import_cannot_reach_subprocess(self) -> None:
        """The audit demonstrated real command execution through this path."""
        result = _run(
            "    m = __import__('subprocess')\n"
            "    return {'out': m.run(['echo','x'], capture_output=True).returncode}"
        )
        assert not result.success
        assert "subprocess" in (result.error or "")

    def test_submodule_import_is_matched_on_root(self) -> None:
        """`os.path` must be refused because its root package `os` is blocked."""
        result = _run("    return {'x': __import__('os.path')}")
        assert not result.success
        assert "Blocked import" in (result.error or "")

    def test_builtins_module_is_refused(self) -> None:
        """`builtins.__import__` is the REAL importer -- fetching it undoes the guard."""
        result = _run("    return {'x': __import__('builtins').__import__('os').getcwd()}")
        assert not result.success
        assert "builtins" in (result.error or "")

    def test_sys_modules_is_refused(self) -> None:
        """`sys.modules['os']` reaches an imported module without any importer."""
        result = _run("    return {'x': str(__import__('sys').modules['os'])}")
        assert not result.success
        assert "sys" in (result.error or "")

    def test_escape_hatches_hold_even_if_caller_narrows_the_blocklist(self) -> None:
        """A caller-supplied blocklist must not be able to re-open `builtins`/`sys`.

        `_ESCAPE_HATCH_MODULES` is enforced unconditionally precisely so that a
        permissive `SandboxConfig` cannot hand back a trivial escape.
        """
        permissive = SandboxConfig(blocked_modules=frozenset())
        code = "def run(d):\n    return {'x': str(__import__('sys').modules.keys())[:5]}\n"
        result = run_in_sandbox(code, {}, permissive)
        assert not result.success
        assert "sys" in (result.error or "")


class TestDeniedBuiltins:
    """REQ-AUTO-009: builtins granting filesystem or code-exec access are removed."""

    def test_open_is_unavailable(self) -> None:
        """`open` reaches the filesystem without importing anything."""
        result = _run("    return {'x': open('/etc/passwd').read()}")
        assert not result.success
        assert "open" in (result.error or "")

    def test_eval_is_unavailable(self) -> None:
        """`eval` re-enters with a namespace the guard does not control."""
        result = _run("    return {'x': eval(\"1+1\")}")
        assert not result.success
        assert "eval" in (result.error or "")

    def test_exec_is_unavailable(self) -> None:
        """Same reasoning as eval."""
        result = _run("    exec('x=1')\n    return {}")
        assert not result.success
        assert "exec" in (result.error or "")


class TestLegitimateHypothesesStillRun:
    """The guard must not break the hypotheses the sandbox exists to run.

    This is the half that makes the fix safe to ship: `import jax` compiles to a
    `__import__` call, so a guard that merely deleted `__import__` would have
    blocked every real hypothesis, not just the malicious ones.
    """

    def test_stdlib_import_statement_works(self) -> None:
        code = "import json, math\ndef run(d):\n    return {'v': json.dumps(math.sqrt(16))}\n"
        result = run_in_sandbox(code, {})
        assert result.success, result.error
        assert result.metrics == {"v": "4.0"}

    def test_third_party_import_works(self) -> None:
        """numpy is the workhorse import for real hypotheses."""
        code = "import numpy as np\ndef run(d):\n    return {'mean': float(np.array([1,2,3]).mean())}\n"
        result = run_in_sandbox(code, {})
        assert result.success, result.error
        assert result.metrics == {"mean": 2.0}

    def test_from_import_works(self) -> None:
        code = "from math import sqrt\ndef run(d):\n    return {'v': sqrt(9)}\n"
        result = run_in_sandbox(code, {})
        assert result.success, result.error
        assert result.metrics == {"v": 3.0}

    def test_benchmark_data_still_reaches_run(self) -> None:
        code = "def run(d):\n    return {'n': len(d['dataset'])}\n"
        result = run_in_sandbox(code, {"dataset": [1, 2, 3]})
        assert result.success, result.error
        assert result.metrics == {"n": 3}

    def test_hypothesis_cannot_corrupt_host_builtins(self) -> None:
        """The namespace gets a COPY, so monkeypatching cannot leak out."""
        code = "def run(d):\n    __builtins__['len'] = lambda x: 999\n    return {'v': len([1])}\n"
        run_in_sandbox(code, {})
        assert len([1]) == 1, "hypothesis mutated the host interpreter's builtins"


class TestKnownResidual:
    """Assert a hole we have NOT closed, so the docstring cannot rot into a lie."""

    def test_reflective_escape_is_still_open(self) -> None:
        """`object.__subclasses__()` reaches `sys` without calling `__import__`.

        This asserts the CURRENT, HONEST state: the process-level sandbox is a
        guardrail, not a security boundary, and gVisor (`CARNOT_USE_SANDBOX=1`)
        is the real containment.  See sandbox.py's module docstring.

        If CPython ever closes this reflective route, this test FAILS -- which is
        the intended signal to revisit the docstring's residual-risk note rather
        than let it silently overstate the risk.  A failure here is good news
        that needs a doc update, not a regression to paper over.
        """
        result = _run(
            "    for c in ().__class__.__base__.__subclasses__():\n"
            "        g = getattr(getattr(c, '__init__', None), '__globals__', None)\n"
            "        if g and 'sys' in g:\n"
            "            return {'escaped': True}\n"
            "    return {'escaped': False}"
        )
        assert result.success, result.error
        assert result.metrics == {"escaped": True}, (
            "The reflective escape appears closed. This is not a regression -- "
            "update sandbox.py's residual-risk docstring and this test together."
        )
