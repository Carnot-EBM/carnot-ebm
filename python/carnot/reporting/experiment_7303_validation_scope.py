"""Run only validation commands named by the current experiment.

Historical helpers mixed an experiment's required checks with a repository-wide
pytest command. This module makes scope an input and keeps old repository health
observations in a separate result field.

Spec refs: REQ-REPORT-7303 and SCENARIO-REPORT-7303-*.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
from typing import Any


JsonDict = dict[str, Any]

REQUIRED_CHECK_NAMES = (
    "worktree_imports",
    "focused_pytest",
    "changed_module_coverage",
    "changed_module_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
)

_RESOLVE_IMPORTS = """\
import importlib
import json
from pathlib import Path
import sys

marker = sys.argv.index("--exp7303-resolve-imports")
root = Path(sys.argv[marker + 1]).resolve()
sys.path[:] = [str(root / "python"), str(root), *sys.path]
resolved = {}
for name in sys.argv[marker + 2:]:
    module = importlib.import_module(name)
    resolved[name] = str(Path(module.__file__).resolve())
print(json.dumps({"resolved_imports": resolved}, sort_keys=True), flush=True)
allowed = (root / "python").resolve()
raise SystemExit(any(not Path(value).is_relative_to(allowed) for value in resolved.values()))
"""


@dataclass(frozen=True)
class CommandSpec:
    """Describe one bounded subprocess before it starts."""

    name: str
    argv: tuple[str, ...]
    scope: str
    timeout_s: float = 900.0


def _sha256_file(path: Path) -> str:
    """Hash exact log bytes so a receipt cannot drift from its evidence."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> str:
    """Use stable repository paths while private fixtures remain readable."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _module_name(path: str) -> str:
    """Convert an implementation path to the import checked by the child process."""

    pure = Path(path)
    parts = pure.with_suffix("").parts
    if len(parts) < 3 or parts[:2] != ("python", "carnot"):
        raise ValueError(f"changed module must be below python/carnot: {path}")
    return ".".join(parts[1:])


def _validate_scope(
    root: Path,
    test_paths: Sequence[str],
    changed_modules: Sequence[str],
    static_paths: Sequence[str],
) -> None:
    """Reject directory targets and ambiguous inputs before a child can run."""

    if not test_paths:
        raise ValueError("explicit test paths are required")
    if not changed_modules:
        raise ValueError("explicit changed-module paths are required")
    for selector in test_paths:
        file_part = selector.split("::", 1)[0]
        normalized = file_part.rstrip("/")
        if normalized in {".", "tests", "tests/python"} or not normalized.endswith(".py"):
            raise ValueError(f"repository-wide pytest target is forbidden: {selector}")
        resolved = (root / normalized).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"test target leaves repository: {selector}")
    for path in (*changed_modules, *static_paths):
        resolved = (root / path).resolve()
        if not path.endswith(".py") or not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"changed path must be an in-repository Python file: {path}")
    for path in changed_modules:
        _module_name(path)


def build_scoped_commands(
    repo_root: Path,
    test_paths: Sequence[str],
    changed_modules: Sequence[str],
    *,
    static_paths: Sequence[str] = (),
    basetemp: Path,
    coverage_file: Path,
) -> list[CommandSpec]:
    """Build the fixed command set from explicit files with no broad fallback."""

    root = repo_root.resolve()
    tests = tuple(str(path) for path in test_paths)
    modules = tuple(str(path) for path in changed_modules)
    static = tuple(str(path) for path in static_paths)
    _validate_scope(root, tests, modules, static)
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    coverage = str(root / ".venv/bin/coverage")
    ruff = str(root / ".venv/bin/ruff")
    mypy = str(root / ".venv/bin/mypy")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    include = ",".join(f"*/{Path(path).relative_to('python/carnot')}" for path in modules)
    imports = tuple(_module_name(path) for path in modules)
    static_scope = (*modules, *static, *tests)
    return [
        CommandSpec(
            "worktree_imports",
            (
                python,
                "-u",
                "-c",
                _RESOLVE_IMPORTS,
                "--exp7303-resolve-imports",
                str(root),
                *imports,
            ),
            "changed_modules",
        ),
        CommandSpec(
            "focused_pytest",
            (pytest, *common, f"--basetemp={basetemp / 'focused'}", *tests, "-q"),
            "explicit_tests",
        ),
        CommandSpec(
            "changed_module_coverage",
            (
                coverage,
                "run",
                f"--data-file={coverage_file}",
                f"--include={include}",
                "-m",
                "pytest",
                *common,
                f"--basetemp={basetemp / 'coverage'}",
                *tests,
                "-q",
            ),
            "explicit_tests_and_changed_modules",
        ),
        CommandSpec(
            "changed_module_coverage_report",
            (
                coverage,
                "report",
                f"--data-file={coverage_file}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ),
            "changed_modules",
        ),
        CommandSpec("ruff_check", (ruff, "check", *static_scope), "changed_files"),
        CommandSpec("ruff_format", (ruff, "format", "--check", *static_scope), "changed_files"),
        CommandSpec("changed_module_mypy", (mypy, *modules), "changed_modules"),
        CommandSpec(
            "scoped_spec_coverage",
            (python, "-u", "scripts/check_spec_coverage.py", *tests),
            "explicit_tests",
        ),
    ]


def _progress(name: str, event: str, started: float, detail: str = "") -> None:
    """Emit one flushed boundary with monotonic elapsed time."""

    suffix = f" {detail}" if detail else ""
    print(
        f"[exp7303-validation] {event} {name} elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def run_commands(
    repo_root: Path,
    commands: Sequence[CommandSpec],
    *,
    log_dir: Path,
    extra_env: Mapping[str, str] | None = None,
    heartbeat_s: float = 60.0,
) -> list[JsonDict]:
    """Stream bounded children and retain every real exit, including failures."""

    root = repo_root.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(extra_env or {})
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    receipts: list[JsonDict] = []
    for index, spec in enumerate(commands):
        started = time.monotonic()
        _progress(spec.name, "before_subprocess", started, f"unit={index + 1}/{len(commands)}")
        process = subprocess.Popen(  # noqa: S603 - caller supplies an argument vector, never a shell.
            spec.argv,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        stop = threading.Event()
        timed_out = threading.Event()

        def monitor() -> None:
            while not stop.wait(heartbeat_s):  # pragma: no cover - production silence guard.
                elapsed = time.monotonic() - started
                _progress(spec.name, "subprocess_outstanding", started)
                if elapsed >= spec.timeout_s:
                    timed_out.set()
                    process.terminate()
                    return

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            print(f"[exp7303-validation:{spec.name}] {line.rstrip()}", flush=True)
        exit_code = process.wait()
        stop.set()
        monitor_thread.join(timeout=1.0)
        log_path = log_dir / f"{index:02d}_{spec.name}.log"
        log_path.write_text("".join(lines), encoding="utf-8")
        receipt: JsonDict = {
            "name": spec.name,
            "command": shlex.join(spec.argv),
            "command_argv": list(spec.argv),
            "scope": spec.scope,
            "exit_code": int(exit_code),
            "duration_s": time.monotonic() - started,
            "log_path": _relative_or_absolute(log_path, root),
            "log_sha256": _sha256_file(log_path),
            "passed": exit_code == 0 and not timed_out.is_set(),
            "timed_out": timed_out.is_set(),
            "output_tail": "".join(lines)[-4000:],
        }
        if spec.name == "worktree_imports":
            parsed = next(
                (
                    json.loads(line)["resolved_imports"]
                    for line in reversed(lines)
                    if line.lstrip().startswith("{") and "resolved_imports" in line
                ),
                {},
            )
            receipt["resolved_imports"] = parsed
            receipt["passed"] = receipt["passed"] and bool(parsed)
        receipts.append(receipt)
        _progress(
            spec.name,
            "after_subprocess",
            started,
            f"exit={exit_code} unit={index + 1}/{len(commands)}",
        )
    return receipts


def reduce_required_checks(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require one passing receipt for each fixed command name."""

    counts = {name: 0 for name in REQUIRED_CHECK_NAMES}
    by_name: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in receipts:
        name = str(row.get("name"))
        if name in counts:
            counts[name] += 1
            if counts[name] > 1:
                duplicates.append(name)
            by_name[name] = row
    missing = [name for name in REQUIRED_CHECK_NAMES if counts[name] == 0]
    failed = [
        name
        for name in REQUIRED_CHECK_NAMES
        if name in by_name
        and (
            by_name[name].get("passed") is not True
            or by_name[name].get("exit_code") != 0
            or by_name[name].get("timed_out") is True
        )
    ]
    return {
        "required_checks_passed": not missing and not failed and not duplicates,
        "missing_required_commands": missing,
        "failed_required_commands": failed,
        "duplicate_required_commands": sorted(set(duplicates)),
    }


def build_repository_health(historical_failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep prior failures open without converting them into current receipts."""

    failures = [dict(row) for row in historical_failures]
    error_count = sum(len(row.get("collection_errors") or []) for row in failures)
    unresolved = any(row.get("resolved") is not True for row in failures)
    return {
        "status": "degraded_open" if unresolved else "healthy",
        "incident_open": unresolved,
        "historical_failures": failures,
        "historical_failure_count": len(failures),
        "unresolved_collection_error_observation_count": error_count,
        "affects_required_checks": False,
    }


def validation_outcome(
    receipts: Sequence[Mapping[str, Any]],
    historical_failures: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Return required validation and unrelated health as separate values."""

    current = [dict(row) for row in receipts]
    return {
        "validation_receipts": current,
        **reduce_required_checks(current),
        "repository_health": build_repository_health(historical_failures),
    }


def run_scoped_validation(
    repo_root: Path,
    test_paths: Sequence[str],
    changed_modules: Sequence[str],
    *,
    static_paths: Sequence[str] = (),
    basetemp: Path,
    coverage_file: Path,
    log_dir: Path | None = None,
    historical_failures: Sequence[Mapping[str, Any]] = (),
    extra_env: Mapping[str, str] | None = None,
) -> JsonDict:
    """Execute the explicit command set and reduce only current required checks."""

    commands = build_scoped_commands(
        repo_root,
        test_paths,
        changed_modules,
        static_paths=static_paths,
        basetemp=basetemp,
        coverage_file=coverage_file,
    )
    receipts = run_commands(
        repo_root,
        commands,
        log_dir=log_dir or repo_root / "results/raw/experiment_7303/validation",
        extra_env=extra_env,
    )
    return validation_outcome(receipts, historical_failures)
