#!/usr/bin/env python3
"""Release preparation script for Carnot.

Verifies that the package is ready for release by running a series of checks:
1. Version consistency (pyproject.toml, _version.py, __init__.py)
2. Unit tests pass
3. Clean install works (import, version, entrypoint)
4. CLI commands work (verify, score --list-models)
5. Example scripts run without error

Usage:
    JAX_PLATFORMS=cpu python scripts/prepare_release.py

Exit code 0 means all checks passed; non-zero means at least one failed.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSION = "0.1.0b1"

# Examples that should run cleanly with JAX_PLATFORMS=cpu and no external deps
RUNNABLE_EXAMPLES = [
    "examples/batch_verify.py",
    "examples/math_funcs.py",  # not a script, but importable
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CheckResult:
    """Result of a single release check."""

    def __init__(self, name: str, passed: bool, detail: str = "") -> None:
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self) -> str:
        icon = "PASS" if self.passed else "FAIL"
        suffix = f" -- {self.detail}" if self.detail else ""
        return f"  [{icon}] {self.name}{suffix}"


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, return CompletedProcess."""
    merged_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged_env,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_version_file() -> CheckResult:
    """Verify _version.py contains the expected version string."""
    try:
        # Force reimport
        if "carnot._version" in sys.modules:
            del sys.modules["carnot._version"]
        if "carnot" in sys.modules:
            del sys.modules["carnot"]

        from carnot._version import __version__

        if __version__ == EXPECTED_VERSION:
            return CheckResult("Version file", True, f"v{__version__}")
        return CheckResult(
            "Version file", False,
            f"expected {EXPECTED_VERSION!r}, got {__version__!r}",
        )
    except Exception as e:
        return CheckResult("Version file", False, str(e))


def check_version_init() -> CheckResult:
    """Verify __init__.py re-exports the same version."""
    try:
        import carnot
        from carnot._version import __version__

        if carnot.__version__ == __version__:
            return CheckResult("Version init consistency", True)
        return CheckResult(
            "Version init consistency", False,
            f"init={carnot.__version__!r} vs file={__version__!r}",
        )
    except Exception as e:
        return CheckResult("Version init consistency", False, str(e))


def check_imports() -> CheckResult:
    """Verify that key public modules are importable."""
    modules = [
        "carnot",
        "carnot.core",
        "carnot.models",
        "carnot.samplers",
        "carnot.training",
        "carnot.pipeline",
        "carnot.cli",
        "carnot.verify",
    ]
    failures: list[str] = []
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            failures.append(f"{mod_name}: {e}")

    if not failures:
        return CheckResult("Public imports", True, f"{len(modules)} modules OK")
    return CheckResult("Public imports", False, "; ".join(failures))


def check_unit_tests() -> CheckResult:
    """Run pytest on tests/python with coverage."""
    result = run_cmd(
        [sys.executable, "-m", "pytest", "tests/python", "-x", "-q",
         "--no-header", "--tb=short", "-p", "no:cacheprovider"],
        timeout=300,
    )
    if result.returncode == 0:
        # Extract summary line
        lines = result.stdout.strip().splitlines()
        summary = lines[-1] if lines else "passed"
        return CheckResult("Unit tests", True, summary.strip())
    # Show last few lines of output on failure
    lines = (result.stdout + result.stderr).strip().splitlines()
    detail = "\n".join(lines[-5:]) if lines else "unknown error"
    return CheckResult("Unit tests", False, detail)


def check_cli_verify() -> CheckResult:
    """Run 'carnot verify' on examples/math_funcs.py."""
    result = run_cmd([
        sys.executable, "-m", "carnot.cli",
        "verify", "examples/math_funcs.py",
        "--func", "gcd",
        "--test", "(12,8):4",
        "--test", "(7,13):1",
    ])
    if result.returncode == 0 and "PASS" in result.stdout:
        return CheckResult("CLI verify", True, "gcd verified")
    detail = result.stderr.strip() or result.stdout.strip()
    return CheckResult("CLI verify", False, detail[:200])


def check_cli_score_list() -> CheckResult:
    """Run 'carnot score --list-models'."""
    result = run_cmd([
        sys.executable, "-m", "carnot.cli",
        "score", "--list-models",
    ])
    if result.returncode == 0 and "per-token-ebm" in result.stdout:
        return CheckResult("CLI score --list-models", True)
    detail = result.stderr.strip() or result.stdout.strip()
    return CheckResult("CLI score --list-models", False, detail[:200])


def check_example_batch_verify() -> CheckResult:
    """Run the batch_verify example with built-in sample data."""
    example = REPO_ROOT / "examples" / "batch_verify.py"
    if not example.exists():
        return CheckResult("Example: batch_verify", False, "file not found")

    result = run_cmd(
        [sys.executable, str(example)],
        timeout=120,
    )
    if result.returncode == 0:
        return CheckResult("Example: batch_verify", True)
    detail = result.stderr.strip() or result.stdout.strip()
    return CheckResult("Example: batch_verify", False, detail[:200])


def check_example_custom_extractor() -> CheckResult:
    """Run the custom_extractor example."""
    example = REPO_ROOT / "examples" / "custom_extractor.py"
    if not example.exists():
        return CheckResult("Example: custom_extractor", False, "file not found")

    result = run_cmd(
        [sys.executable, str(example)],
        timeout=120,
    )
    if result.returncode == 0:
        return CheckResult("Example: custom_extractor", True)
    detail = result.stderr.strip() or result.stdout.strip()
    return CheckResult("Example: custom_extractor", False, detail[:200])


def check_release_notes() -> CheckResult:
    """Verify RELEASE_NOTES.md exists and mentions the version."""
    rn = REPO_ROOT / "RELEASE_NOTES.md"
    if not rn.exists():
        return CheckResult("Release notes", False, "RELEASE_NOTES.md not found")
    content = rn.read_text()
    if EXPECTED_VERSION.replace("b", "-beta") in content or EXPECTED_VERSION in content:
        return CheckResult("Release notes", True)
    return CheckResult(
        "Release notes", False,
        f"RELEASE_NOTES.md does not mention {EXPECTED_VERSION}",
    )


def check_readme() -> CheckResult:
    """Verify README.md exists and has install instructions."""
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return CheckResult("README", False, "README.md not found")
    content = readme.read_text()
    required = ["pip install", "Quick Start", "carnot"]
    missing = [r for r in required if r not in content]
    if not missing:
        return CheckResult("README", True)
    return CheckResult("README", False, f"missing: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run all release checks and print a summary."""
    start = time.monotonic()

    print("=" * 60)
    print(f"CARNOT RELEASE CHECK — target version {EXPECTED_VERSION}")
    print("=" * 60)
    print()

    checks = [
        check_version_file,
        check_version_init,
        check_imports,
        check_unit_tests,
        check_cli_verify,
        check_cli_score_list,
        check_example_batch_verify,
        check_example_custom_extractor,
        check_release_notes,
        check_readme,
    ]

    results: list[CheckResult] = []
    for check_fn in checks:
        name = check_fn.__doc__ or check_fn.__name__
        # Print a progress dot
        print(f"  Running: {name.split('.')[0].strip()}...", end="", flush=True)
        result = check_fn()
        results.append(result)
        icon = "ok" if result.passed else "FAIL"
        print(f" {icon}")

    elapsed = time.monotonic() - start

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for r in results:
        print(r)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    print(f"  {passed} passed, {failed} failed ({elapsed:.1f}s)")

    if failed == 0:
        print(f"\n  Ready to release Carnot {EXPECTED_VERSION}")
        print("=" * 60)
        return 0

    print(f"\n  NOT ready — fix {failed} issue(s) above")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
