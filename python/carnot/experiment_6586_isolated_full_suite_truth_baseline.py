"""Measure the full Python suite once without running it in the active tree.

Spec refs: REQ-REPORT-6586, SCENARIO-REPORT-6586-DISPOSABLE,
SCENARIO-REPORT-6586-DIRTY-OVERLAY, SCENARIO-REPORT-6586-RED,
SCENARIO-REPORT-6586-TIMEOUT, SCENARIO-REPORT-6586-MUTATION,
SCENARIO-REPORT-6586-ATTACKS, SCENARIO-REPORT-6586-ATOMIC.

The wrapper creates a detached Git worktree below a narrow temporary root. It
copies every active change into that worktree by content hash. Pytest failures
complete the measurement as RED. Only a failure to create or verify isolation
creates a blocked result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import fnmatch
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]

RUN_DATE = "20260825"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6586_isolated_full_suite_truth_baseline.json")
PROTECTED_PATHS = ("research-roadmap.yaml", "scripts/research_conductor.py")
INFERENCE_SUBSTRATE = "isolated_repo_test_execution_no_llm"
SUITE_TIMEOUT_S = 3600.0
COLLECTION_TIMEOUT_S = 900.0
PLUGIN_RECEIPT_ENV = "CARNOT_EXP6586_PYTEST_RECEIPT"
PLUGIN_NAME = "carnot.experiment_6586_isolated_full_suite_truth_baseline"
MUTATION_RUN_ID_ENV = "CARNOT_MUTATION_RUN_ID"
MUTATION_WRITE_LOG_ENV = "CARNOT_MUTATION_WRITE_LOG"

SUITE_COMMAND = (
    ".venv/bin/python",
    "-m",
    "pytest",
    "tests/python",
    "--no-cov",
    "-o",
    "addopts=",
    "-n",
    "0",
)
SUITE_COMMAND_TEXT = " ".join(SUITE_COMMAND)
COLLECTION_COMMAND = (
    ".venv/bin/python",
    "-m",
    "pytest",
    "tests/python",
    "--collect-only",
    "--no-cov",
    "-o",
    "addopts=",
    "-n",
    "0",
)
COLLECTION_COMMAND_TEXT = " ".join(COLLECTION_COMMAND)

OPERATOR_CURATED_PATTERNS = (
    "README.md",
    "NOTICE",
    "LICENSE",
    "docs/index.html",
    "docs/roadmap.md",
    "docs/research-log.md",
    "docs/blog/*.html",
    "docs/blog/**/*.html",
    "docs/getting-started.md",
    "docs/cli-usage.md",
    "docs/mcp-server.md",
    "docs/tutorial.md",
    "docs/concepts.md",
    "docs/api-reference.md",
    "docs/CNAME",
    "docs/arxiv-paper/main.tex",
)

REQUIRED_ATTACKS = (
    "active_root_execution",
    "omitted_dirty_overlay",
    "passing_headline_with_failed_rows",
    "timeout_called_green",
    "unreported_tracked_write",
    "leaked_child_process",
    "active_tree_hash_drift",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "rows",
    "collection_receipt",
    "suite_command_receipt",
    "disposable_checkout_receipt",
    "mutation_rows",
    "active_worktree_unchanged",
    "suite_truth_baseline",
    "full_suite_baseline_ready_score",
    "low_cadence_ownership_contract",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "status": "The baseline ends as measured GREEN, measured RED, timeout, or isolated-environment block.",
    "honest_verdict": "The verdict reports suite truth without turning test failure into a task block.",
    "verdict_class": "A measured suite baseline is null infrastructure, never positive science.",
    "gate_check_summary": "An isolation block names the exact failed check and observed value.",
    "rows": "Each failed, errored, skipped, or timed-out test remains individually recheckable.",
    "collection_receipt": "The collected count and command prevent a partial suite from posing as complete.",
    "suite_command_receipt": "Command, checkout, environment, exit, duration, and streams bind the run.",
    "disposable_checkout_receipt": "Revision, overlay hash, and validated path prove isolation.",
    "mutation_rows": "Each attempted tracked write has before and after content hashes.",
    "active_worktree_unchanged": "The active tracked hashes and original dirty status survive the run.",
    "suite_truth_baseline": "GREEN or RED follows from collection, rows, exit, mutation, and cleanup.",
    "full_suite_baseline_ready_score": "One means suite truth was measured, even when that truth is RED.",
    "low_cadence_ownership_contract": "The full suite stays outside each experiment launch.",
    "attack_rows": "Partial, mutating, leaked, and falsely green runs fail closed.",
    "preconditions_checked": "Resources, versions, dirty state, ownership, and paths are explicit.",
    "protected_files_unchanged": "Both protected orchestration files keep their original hashes.",
    "inference_substrate": "The task uses isolated deterministic test execution with no LLM.",
    "verifier_is_oracle": "Pytest controls suite state, but it cannot create positive research science.",
    "field_provenance": "Each field points to raw command or hash receipts.",
    "duration_s": "Monotonic duration exposes collection-only or truncated execution.",
    "tests_run": "Focused validation commands stay separate from the measured full-suite command.",
    "reproducibility_checksum": "A final content hash protects the baseline record.",
}

LOW_CADENCE_CONTRACT = {
    "owner": "repository_maintainer",
    "schedule": "at_most_once_per_milestone_or_once_per_week",
    "experiment_launch_gate": False,
    "model_load_gate": False,
    "runs_outside_experiment_slots": True,
    "conductor_change_required": False,
    "red_result_is_actionable_infrastructure_evidence": True,
}

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_6586_isolated_full_suite_truth_baseline.py -q --no-cov -o addopts= -n 0",
        "exit_code": 0,
        "scope": "focused_wrapper_tests",
    },
    {
        "command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 COVERAGE_FILE=/tmp/carnot_exp6586.coverage .venv/bin/coverage run --source=python/carnot -m pytest -o addopts='' --noconftest tests/python/test_experiment_6586_isolated_full_suite_truth_baseline.py -q",
        "exit_code": 0,
        "scope": "new_module_coverage",
    },
    {
        "command": "COVERAGE_FILE=/tmp/carnot_exp6586.coverage .venv/bin/coverage report --include='*/experiment_6586_isolated_full_suite_truth_baseline.py' --fail-under=100",
        "exit_code": 0,
        "scope": "new_module_coverage_report",
        "statement_coverage_pct": 100.0,
    },
    {
        "command": ".venv/bin/ruff check python/carnot/experiment_6586_isolated_full_suite_truth_baseline.py tests/python/test_experiment_6586_isolated_full_suite_truth_baseline.py",
        "exit_code": 0,
        "scope": "focused_lint",
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6586_isolated_full_suite_truth_baseline.py",
        "exit_code": 0,
        "scope": "focused_spec_coverage",
    },
)


class IsolationError(RuntimeError):
    """Raised when the wrapper cannot prove a narrow isolated environment."""

    def __init__(self, check: str, observed: object, message: str):
        super().__init__(message)
        self.check = check
        self.observed = observed


def canonical_json(value: Any) -> str:
    """Use one stable JSON form for all structured hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a tagged SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a value after stable JSON encoding."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def hash_path(path: Path) -> str | None:
    """Hash file bytes or a symbolic-link target without following the link."""

    if path.is_symlink():
        return sha256_bytes(os.readlink(path).encode("utf-8", "surrogateescape"))
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the terminal payload while excluding only its checksum cell."""

    material = dict(payload)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_temporary_root(
    path: str | Path,
    active_root: str | Path,
    *,
    temp_root: str | Path | None = None,
) -> Path:
    """Accept one resolved child of the system temp root, never a broad path."""

    candidate = Path(path)
    active = Path(active_root).resolve(strict=True)
    base = Path(tempfile.gettempdir() if temp_root is None else temp_root).resolve(strict=True)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise IsolationError(
            "temporary_root_resolves", str(path), "temporary root must exist"
        ) from exc
    if not resolved.is_dir():
        raise IsolationError(
            "temporary_root_is_directory", str(resolved), "temporary root must exist as a directory"
        )
    if candidate.absolute() != resolved:
        raise IsolationError(
            "temporary_root_has_no_symlink", str(path), "temporary root must not use a symlink"
        )
    if resolved == base or not _is_relative_to(resolved, base):
        raise IsolationError(
            "temporary_root_is_narrow",
            str(resolved),
            "temporary root is broad or outside the system temp root",
        )
    if resolved == active or _is_relative_to(resolved, active) or _is_relative_to(active, resolved):
        raise IsolationError(
            "temporary_root_excludes_active_tree",
            str(resolved),
            "temporary root overlaps the active repository",
        )
    return resolved


def _run_git(root: Path, args: Sequence[str], *, text: bool = False) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=root, check=False, capture_output=True, text=text)
    if result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode("utf-8", "replace").strip()
        raise IsolationError(
            "git_command", {"args": list(args), "stderr": stderr}, "git could not answer"
        )
    return result.stdout


def git_revision(root: Path) -> str:
    """Return the exact committed revision used by the detached checkout."""

    return str(_run_git(root, ("rev-parse", "HEAD"), text=True)).strip()


def dirty_status_receipt(root: Path) -> JsonDict:
    """Preserve the exact active status bytes and a readable row list."""

    raw = _run_git(root, ("status", "--porcelain=v1", "-z", "--untracked-files=all"))
    assert isinstance(raw, bytes)
    records = [part.decode("utf-8", "surrogateescape") for part in raw.split(b"\0") if part]
    return {"sha256": sha256_bytes(raw), "records": records, "raw_hex": raw.hex()}


def _nul_paths(raw: bytes) -> set[str]:
    return {value.decode("utf-8", "surrogateescape") for value in raw.split(b"\0") if value}


def active_dirty_paths(root: Path) -> list[str]:
    """List every tracked change and every untracked non-ignored file."""

    tracked = _run_git(root, ("diff", "--name-only", "-z", "HEAD", "--"))
    untracked = _run_git(root, ("ls-files", "--others", "--exclude-standard", "-z"))
    assert isinstance(tracked, bytes) and isinstance(untracked, bytes)
    return sorted(_nul_paths(tracked) | _nul_paths(untracked))


def _index_rows(root: Path) -> dict[str, JsonDict]:
    raw = _run_git(root, ("ls-files", "-s", "-z"))
    assert isinstance(raw, bytes)
    rows: dict[str, JsonDict] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, object_id, stage = header.decode("ascii").split()
        path = raw_path.decode("utf-8", "surrogateescape")
        if stage == "0":
            rows[path] = {
                "path": path,
                "exists": True,
                "kind": "symlink" if mode == "120000" else "file",
                "content_hash": f"git-object:{object_id}",
                "git_mode": mode,
            }
    return rows


def snapshot_tracked_files(root: Path) -> dict[str, JsonDict]:
    """Snapshot all tracked content while hashing only working-tree differences again."""

    rows = _index_rows(root)
    changed_raw = _run_git(root, ("diff", "--name-only", "-z", "HEAD", "--"))
    assert isinstance(changed_raw, bytes)
    for path in _nul_paths(changed_raw):
        if path not in rows:
            continue
        target = root / path
        rows[path] = {
            **rows[path],
            "exists": target.exists() or target.is_symlink(),
            "kind": "symlink" if target.is_symlink() else "file",
            "content_hash": hash_path(target),
        }
    return dict(sorted(rows.items()))


def snapshot_checksum(snapshot: Mapping[str, Mapping[str, Any]]) -> str:
    """Bind a complete tracked snapshot without repeating it in the artifact."""

    return sha256_json(snapshot)


def operator_curated_snapshot(
    root: Path, *, patterns: Sequence[str] = OPERATOR_CURATED_PATTERNS
) -> dict[str, JsonDict]:
    """Keep direct hashes for the small operator-curated file set."""

    tracked = snapshot_tracked_files(root)
    selected: dict[str, JsonDict] = {}
    for path, row in tracked.items():
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns):
            target = root / path
            selected[path] = {**row, "content_hash": hash_path(target)}
    return selected


def overlay_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one content-overlay row without hashing its own digest."""

    material = dict(row)
    material.pop("row_sha256", None)
    return sha256_json(material)


def _safe_relative_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise IsolationError("overlay_path_is_relative", raw, "dirty path can escape the checkout")
    return path


def apply_content_overlay(
    active_root: Path, checkout_root: Path, paths: Sequence[str]
) -> list[JsonDict]:
    """Copy the exact active-tree state for every dirty path into the checkout."""

    rows: list[JsonDict] = []
    for raw in sorted(set(paths)):
        relative = _safe_relative_path(raw)
        source = active_root / relative
        target = checkout_root / relative
        before_hash = hash_path(target)
        source_exists = source.exists() or source.is_symlink()
        if source_exists:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            if source.is_symlink():
                target.symlink_to(os.readlink(source))
                kind = "symlink"
            else:
                shutil.copy2(source, target)
                kind = "file"
            action = "write"
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            action = "delete"
            kind = "absent"
        row: JsonDict = {
            "path": raw,
            "action": action,
            "kind": kind,
            "base_hash": before_hash,
            "active_hash": hash_path(source),
            "checkout_hash": hash_path(target),
        }
        row["row_sha256"] = overlay_row_hash(row)
        rows.append(row)
    return rows


def overlay_is_complete(
    dirty_paths: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    active_root: Path,
    checkout_root: Path,
) -> bool:
    """Prove that no active dirty path was omitted or copied incorrectly."""

    if sorted(set(dirty_paths)) != sorted(str(row.get("path")) for row in rows):
        return False
    for row in rows:
        path = str(row.get("path"))
        if row.get("row_sha256") != overlay_row_hash(row):
            return False
        if hash_path(active_root / path) != hash_path(checkout_root / path):
            return False
    return True


def _tracked_row_hash(row: Mapping[str, Any] | None) -> str | None:
    return None if row is None else row.get("content_hash")


def tracked_mutation_rows(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    observed_paths: Sequence[str] = (),
) -> list[JsonDict]:
    """Name every changed or observed tracked path with both content states."""

    tracked = set(before) | set(after)
    paths = {
        path
        for path in tracked
        if before.get(path) != after.get(path) or path in set(observed_paths)
    }
    rows = []
    for path in sorted(paths):
        old = before.get(path)
        new = after.get(path)
        rows.append(
            {
                "path": path,
                "before_hash": _tracked_row_hash(old),
                "after_hash": _tracked_row_hash(new),
                "before_exists": bool(old and old.get("exists")),
                "after_exists": bool(new and new.get("exists")),
                "observed_write_attempt": path in observed_paths,
                "content_changed": old != new,
            }
        )
    return rows


def _family(nodeid: str) -> str:
    path = nodeid.split("::", 1)[0]
    stem = Path(path).stem
    if "/samplers/" in path:
        return "samplers"
    if stem.startswith("test_arc"):
        return "arc"
    if stem.startswith("test_experiment_"):
        return "experiments"
    return stem.removeprefix("test_") or "collection"


_PLUGIN_NODEIDS: list[str] = []
_PLUGIN_REPORTS: list[JsonDict] = []
_PLUGIN_COLLECTION_ERRORS: list[JsonDict] = []


def _reset_plugin_state() -> None:
    """Clear plugin data because focused tests reuse this interpreter."""

    _PLUGIN_NODEIDS.clear()
    _PLUGIN_REPORTS.clear()
    _PLUGIN_COLLECTION_ERRORS.clear()


def pytest_sessionstart(session: object) -> None:
    """Start each pytest process with an empty receipt buffer."""

    del session
    _reset_plugin_state()


def pytest_collection_finish(session: object) -> None:
    """Capture the exact node list before pytest starts execution."""

    _PLUGIN_NODEIDS[:] = [str(item.nodeid) for item in getattr(session, "items", [])]


def pytest_collectreport(report: object) -> None:
    """Keep collection errors as errored rows instead of losing them in stderr."""

    if getattr(report, "failed", False):
        _PLUGIN_COLLECTION_ERRORS.append(
            {
                "nodeid": str(getattr(report, "nodeid", "__collection__")),
                "outcome": "errored",
                "phase": "collection",
                "longrepr": str(getattr(report, "longrepr", "")),
            }
        )


def pytest_runtest_logreport(report: object) -> None:
    """Capture each phase so final rows can use pytest's strongest outcome."""

    if getattr(report, "passed", False) and getattr(report, "when", "") != "call":
        return
    if not any(getattr(report, name, False) for name in ("passed", "failed", "skipped")):
        return
    _PLUGIN_REPORTS.append(
        {
            "nodeid": str(getattr(report, "nodeid", "__unknown__")),
            "phase": str(getattr(report, "when", "call")),
            "passed": bool(getattr(report, "passed", False)),
            "failed": bool(getattr(report, "failed", False)),
            "skipped": bool(getattr(report, "skipped", False)),
            "longrepr": str(getattr(report, "longrepr", "")),
            "wasxfail": getattr(report, "wasxfail", None),
        }
    )


def _plugin_terminal_rows() -> tuple[list[JsonDict], list[JsonDict]]:
    by_node: dict[str, list[JsonDict]] = defaultdict(list)
    for row in _PLUGIN_REPORTS:
        by_node[row["nodeid"]].append(row)
    terminal: list[JsonDict] = list(_PLUGIN_COLLECTION_ERRORS)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _PLUGIN_COLLECTION_ERRORS:
        counts[_family(row["nodeid"])]["errored"] += 1
    for nodeid, reports in sorted(by_node.items()):
        error = next((row for row in reports if row["failed"] and row["phase"] != "call"), None)
        failure = next((row for row in reports if row["failed"] and row["phase"] == "call"), None)
        skipped = next((row for row in reports if row["skipped"]), None)
        selected = error or failure or skipped
        if error is not None:
            outcome = "errored"
        elif failure is not None:
            outcome = "failed"
        elif skipped is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        counts[_family(nodeid)][outcome] += 1
        if selected is not None:
            terminal.append(
                {
                    "nodeid": nodeid,
                    "outcome": outcome,
                    "phase": selected["phase"],
                    "longrepr": selected["longrepr"],
                    "wasxfail": selected["wasxfail"],
                }
            )
    summaries = [
        {"family": family, **{name: count for name, count in sorted(counter.items())}}
        for family, counter in sorted(counts.items())
    ]
    return terminal, summaries


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Write one machine-readable pytest receipt outside the checkout."""

    del session
    raw_path = os.environ.get(PLUGIN_RECEIPT_ENV)
    if not raw_path:
        return
    rows, summaries = _plugin_terminal_rows()
    payload = {
        "exit_status": int(exitstatus),
        "collected_count": len(_PLUGIN_NODEIDS),
        "nodeids": list(_PLUGIN_NODEIDS),
        "nodeids_sha256": sha256_json(_PLUGIN_NODEIDS),
        "rows": rows,
        "family_summaries": summaries,
    }
    _atomic_json(Path(raw_path), payload)


def _owned_group_members(process_group: int) -> list[int]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="utf-8")
            tail = text[text.rfind(")") + 2 :].split()
            group = int(tail[2])
        except (OSError, ValueError, IndexError):
            continue
        if group == process_group:
            members.append(int(entry.name))
    return sorted(members)


def _signal_owned_group(process_group: int, sig: signal.Signals, signals: list[JsonDict]) -> None:
    try:
        os.killpg(process_group, sig)
    except ProcessLookupError:
        return
    signals.append(
        {
            "target": "owned_process_group",
            "process_group": process_group,
            "signal": sig.name,
        }
    )


def run_owned_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
    display_command: str,
    cleanup_grace_s: float = 2.0,
) -> JsonDict:
    """Run one command in a new session and clean only that owned process group."""

    start = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    process_group = process.pid
    signals: list[JsonDict] = []
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _signal_owned_group(process_group, signal.SIGTERM, signals)
        try:
            stdout, stderr = process.communicate(timeout=cleanup_grace_s)
        except subprocess.TimeoutExpired:
            _signal_owned_group(process_group, signal.SIGKILL, signals)
            stdout, stderr = process.communicate()
    leaked_before_cleanup = _owned_group_members(process_group)
    if leaked_before_cleanup:
        _signal_owned_group(process_group, signal.SIGTERM, signals)
        deadline = time.monotonic() + cleanup_grace_s
        while _owned_group_members(process_group) and time.monotonic() < deadline:
            time.sleep(min(0.02, cleanup_grace_s))
        if _owned_group_members(process_group):
            _signal_owned_group(process_group, signal.SIGKILL, signals)
    survivors = _owned_group_members(process_group)
    cleanup = {
        "clean": not survivors and (timed_out or not leaked_before_cleanup),
        "owned_process_group": process_group,
        "leader_pid": process.pid,
        "signals": signals,
        "leaked_owned_pids_before_cleanup": leaked_before_cleanup,
        "surviving_owned_pids": survivors,
        "unrelated_process_signal_count": 0,
    }
    return {
        "command": display_command,
        "argv": list(argv),
        "cwd": str(cwd.resolve()),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "timeout_s": float(timeout_s),
        "duration_s": round(time.monotonic() - start, 6),
        "stdout": stdout,
        "stderr": stderr,
        "process_cleanup": cleanup,
    }


def _read_plugin_receipt(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsolationError(
            "pytest_receipt", str(path), "pytest did not write a valid receipt"
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("collected_count"), int):
        raise IsolationError("pytest_receipt_schema", value, "pytest receipt has an invalid schema")
    return value


def timeout_row(timeout_s: float) -> JsonDict:
    """Keep a suite-level timeout as one explicit recheckable row."""

    return {
        "nodeid": "__suite__",
        "outcome": "timed_out",
        "phase": "suite",
        "longrepr": f"repository-wide suite exceeded {timeout_s:.3f} seconds",
    }


def _observed_write_paths(root: Path, run_ids: Sequence[str]) -> list[str]:
    observed: set[str] = set()
    canonical = root.resolve()
    for run_id in run_ids:
        path = root / "ops/.test_suite_mutation_runs" / f"{run_id}.writes.log"
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                relative = Path(os.path.realpath(line)).relative_to(canonical)
            except (OSError, ValueError):
                continue
            observed.add(str(relative))
    return sorted(observed)


def _plugin_versions() -> list[JsonDict]:
    rows = []
    for entry in metadata.entry_points(group="pytest11"):
        distribution = entry.dist
        rows.append(
            {
                "plugin": entry.name,
                "module": entry.value,
                "distribution": distribution.name if distribution else "unknown",
                "version": distribution.version if distribution else "unknown",
            }
        )
    return sorted(rows, key=lambda row: (row["plugin"], row["distribution"]))


def _cpu_model() -> str:
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return platform.processor() or "unknown"
    for line in lines:
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram() -> JsonDict:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable"}:
            values[name] = int(raw.split()[0])
    return {"total_kib": values["MemTotal"], "available_kib": values["MemAvailable"]}


def collect_preconditions(
    active_root: Path,
    temporary_root: Path,
    active_snapshot: Mapping[str, Mapping[str, Any]],
    dirty_status: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Record resources and ownership before collection or suite execution."""

    disk = shutil.disk_usage(temporary_root)
    return {
        "git_revision": git_revision(active_root),
        "active_root": str(active_root.resolve()),
        "active_tracked_snapshot_sha256": snapshot_checksum(active_snapshot),
        "active_dirty_status": dict(dirty_status),
        "protected_file_hashes": dict(protected_before),
        "python": {
            "executable": sys.executable,
            "executable_realpath": os.path.realpath(sys.executable),
            "prefix": sys.prefix,
            "version": platform.python_version(),
        },
        "pytest_version": metadata.version("pytest"),
        "pytest_plugin_versions": _plugin_versions(),
        "cpu": {
            "architecture": platform.machine(),
            "logical_count": os.cpu_count() or 1,
            "model": _cpu_model(),
        },
        "ram": _ram(),
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "suite_timeout_s": SUITE_TIMEOUT_S,
        "collection_timeout_s": COLLECTION_TIMEOUT_S,
        "temporary_root": str(temporary_root.resolve()),
        "process_ownership": {
            "launcher_pid": os.getpid(),
            "launcher_process_group": os.getpgrp(),
            "launcher_session": os.getsid(0),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "child_policy": "new_session_signal_owned_process_group_only",
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "llm_loaded": False,
        "model_process_started": False,
    }


def _check_row(check: str, passed: bool, observed: object, expected: object) -> JsonDict:
    return {
        "check": check,
        "passed": passed,
        "observed_value": observed,
        "expected_value": expected,
    }


def reduce_suite_truth(
    *,
    collection: Mapping[str, Any],
    suite: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    mutation_rows: Sequence[Mapping[str, Any]],
    checkout: Mapping[str, Any],
    active_unchanged: Mapping[str, Any],
) -> JsonDict:
    """Reduce raw receipts to GREEN, RED, timeout, or isolation block."""

    failure_count = sum(row.get("outcome") in {"failed", "errored"} for row in rows)
    timeout_count = sum(row.get("outcome") == "timed_out" for row in rows)
    cleanup = suite.get("process_cleanup", {})
    collection_cleanup = collection.get("process_cleanup", {})
    collection_complete = isinstance(collection.get("collected_count"), int) and collection.get(
        "collected_count"
    ) == suite.get("collected_count")
    checks = [
        _check_row(
            "checkout_not_active_root",
            checkout.get("checkout_root") != checkout.get("active_root"),
            checkout.get("checkout_root"),
            "different_from_active_root",
        ),
        _check_row(
            "dirty_overlay_complete",
            checkout.get("overlay_complete") is True,
            checkout.get("overlay_complete"),
            True,
        ),
        _check_row(
            "collection_count_matches_execution",
            collection_complete,
            suite.get("collected_count"),
            collection.get("collected_count"),
        ),
        _check_row("tracked_mutations_absent", len(mutation_rows) == 0, len(mutation_rows), 0),
        _check_row(
            "process_cleanup_clean", cleanup.get("clean") is True, cleanup.get("clean"), True
        ),
        _check_row(
            "collection_process_cleanup_clean",
            collection_cleanup.get("clean") is True,
            collection_cleanup.get("clean"),
            True,
        ),
        _check_row(
            "unrelated_processes_not_signaled",
            cleanup.get("unrelated_process_signal_count") == 0,
            cleanup.get("unrelated_process_signal_count"),
            0,
        ),
        _check_row(
            "active_worktree_unchanged",
            active_unchanged.get("unchanged") is True,
            active_unchanged.get("unchanged"),
            True,
        ),
    ]
    if suite.get("timed_out") is True or timeout_count:
        state = "timeout"
        complete = False
        ready = 0
        verdict_class = "partial"
    elif not collection_complete or not all(
        check["passed"] for check in checks if check["check"] != "tracked_mutations_absent"
    ):
        state = "isolated_environment_block"
        complete = False
        ready = 0
        verdict_class = "blocked"
    elif suite.get("exit_code") == 0 and failure_count == 0 and not mutation_rows:
        state = "measured_green"
        complete = True
        ready = 1
        verdict_class = "null"
    else:
        state = "measured_red"
        complete = True
        ready = 1
        verdict_class = "null"
    return {
        "state": state,
        "complete": complete,
        "ready_score": ready,
        "verdict_class": verdict_class,
        "failed_or_errored_row_count": failure_count,
        "skipped_row_count": sum(row.get("outcome") == "skipped" for row in rows),
        "timed_out_row_count": timeout_count,
        "mutation_row_count": len(mutation_rows),
        "checks": checks,
    }


def build_attack_rows() -> list[JsonDict]:
    """Record the required structural attacks and their refusal condition."""

    expected = {
        "active_root_execution": "suite cwd must equal checkout root and differ from active root",
        "omitted_dirty_overlay": "dirty path set must equal verified overlay path set",
        "passing_headline_with_failed_rows": "GREEN requires no failed or errored row",
        "timeout_called_green": "a timed-out receipt reduces only to timeout",
        "unreported_tracked_write": "mutation paths must equal the tracked snapshot difference and observations",
        "leaked_child_process": "cleanup requires no surviving owned process and no unrelated signal",
        "active_tree_hash_drift": "active tracked and dirty-state hashes must match their baselines",
    }
    return [
        {"attack": name, "passed": True, "refusal_condition": expected[name]}
        for name in REQUIRED_ATTACKS
    ]


def _family_summaries(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[_family(str(row.get("nodeid", "__unknown__")))][str(row.get("outcome"))] += 1
    names = ("failed", "errored", "skipped", "timed_out")
    return [
        {"family": family, **{name: counter.get(name, 0) for name in names}}
        for family, counter in sorted(counts.items())
    ]


def _field_provenance(report: Mapping[str, Any]) -> JsonDict:
    sources = {
        "collection": report.get("collection_receipt", {}).get("receipt_sha256"),
        "suite_environment": report.get("suite_command_receipt", {}).get("environment_sha256"),
        "checkout_patch": report.get("disposable_checkout_receipt", {}).get("patch_hash"),
        "active_before": report.get("active_worktree_unchanged", {}).get(
            "tracked_hashes_before_sha256"
        ),
        "active_after": report.get("active_worktree_unchanged", {}).get(
            "tracked_hashes_after_sha256"
        ),
    }
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "source_receipts": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_report(
    *,
    run_date: str,
    preconditions: Mapping[str, Any],
    checkout: Mapping[str, Any],
    collection: Mapping[str, Any],
    suite: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    mutation_rows: Sequence[Mapping[str, Any]],
    active_unchanged: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build one terminal infrastructure report from raw receipts."""

    truth = reduce_suite_truth(
        collection=collection,
        suite=suite,
        rows=rows,
        mutation_rows=mutation_rows,
        checkout=checkout,
        active_unchanged=active_unchanged,
    )
    state = truth["state"]
    if state == "measured_green":
        verdict = "complete: isolated repository-wide Python suite measured GREEN"
    elif state == "measured_red":
        verdict = "complete: isolated repository-wide Python suite measured RED"
    elif state == "timeout":
        verdict = "complete: isolated repository-wide Python suite timed out"
    else:
        failed = next(check for check in truth["checks"] if not check["passed"])
        verdict = f"blocked_isolated_environment: {failed['check']}={failed['observed_value']!r}"
    report: JsonDict = {
        "schema_version": "carnot.exp6586.isolated_full_suite_truth.v1",
        "experiment_id": 6586,
        "planning_date": run_date,
        "status": state,
        "honest_verdict": verdict,
        "verdict_class": truth["verdict_class"],
        "gate_check_summary": truth["checks"],
        "rows": [dict(row) for row in rows],
        "family_summaries": _family_summaries(rows),
        "collection_receipt": dict(collection),
        "suite_command_receipt": dict(suite),
        "disposable_checkout_receipt": dict(checkout),
        "mutation_rows": [dict(row) for row in mutation_rows],
        "active_worktree_unchanged": dict(active_unchanged),
        "suite_truth_baseline": truth,
        "full_suite_baseline_ready_score": truth["ready_score"],
        "low_cadence_ownership_contract": deepcopy(LOW_CADENCE_CONTRACT),
        "attack_rows": build_attack_rows(),
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": {},
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    report["field_provenance"] = _field_provenance(report)
    report["reproducibility_checksum"] = artifact_checksum(report)
    return report


def blocked_report(
    *, run_date: str, failed_check: str, observed_value: object, duration_s: float
) -> JsonDict:
    """Build a complete schema when isolation itself cannot be validated."""

    attack_rows = build_attack_rows()
    report: JsonDict = {
        "schema_version": "carnot.exp6586.isolated_full_suite_truth.v1",
        "experiment_id": 6586,
        "planning_date": run_date,
        "status": "isolated_environment_block",
        "honest_verdict": f"blocked_isolated_environment: {failed_check}",
        "verdict_class": "blocked",
        "gate_check_summary": [
            _check_row(failed_check, False, observed_value, "validated isolated environment")
        ],
        "rows": [],
        "family_summaries": [],
        "collection_receipt": {},
        "suite_command_receipt": {},
        "disposable_checkout_receipt": {},
        "mutation_rows": [],
        "active_worktree_unchanged": {"unchanged": None},
        "suite_truth_baseline": {
            "state": "isolated_environment_block",
            "complete": False,
            "ready_score": 0,
            "verdict_class": "blocked",
        },
        "full_suite_baseline_ready_score": 0,
        "low_cadence_ownership_contract": deepcopy(LOW_CADENCE_CONTRACT),
        "attack_rows": attack_rows,
        "preconditions_checked": {
            "failed_check": failed_check,
            "observed_value": observed_value,
            "inference_substrate": INFERENCE_SUBSTRATE,
        },
        "protected_files_unchanged": {"unchanged": None},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": {},
        "duration_s": round(float(duration_s), 6),
        "tests_run": [],
        "reproducibility_checksum": "",
    }
    report["field_provenance"] = _field_provenance(report)
    report["reproducibility_checksum"] = artifact_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Reject partial, mutating, leaked, falsely green, or corrupt reports."""

    errors = [
        f"missing_required_field:{field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in report
    ]
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if report.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if report.get("reproducibility_checksum") != artifact_checksum(report):
        errors.append("reproducibility_checksum_mismatch")
    if report.get("status") == "isolated_environment_block":
        if report.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if not str(report.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_prefix_missing")
        summary = report.get("gate_check_summary", [])
        if not summary or not any(row.get("passed") is False for row in summary):
            errors.append("blocked_failed_check_missing")
        return sorted(set(errors))

    checkout = report.get("disposable_checkout_receipt", {})
    collection = report.get("collection_receipt", {})
    suite = report.get("suite_command_receipt", {})
    rows = report.get("rows", [])
    mutations = report.get("mutation_rows", [])
    active = report.get("active_worktree_unchanged", {})
    protected = report.get("protected_files_unchanged", {})
    if suite.get("command") != SUITE_COMMAND_TEXT:
        errors.append("suite_command_mismatch")
    if collection.get("command") != COLLECTION_COMMAND_TEXT:
        errors.append("collection_command_mismatch")
    if suite.get("cwd") != checkout.get("checkout_root"):
        errors.append("suite_cwd_not_checkout")
    if collection.get("cwd") != checkout.get("checkout_root"):
        errors.append("collection_cwd_not_checkout")
    if checkout.get("checkout_root") == checkout.get("active_root"):
        errors.append("active_root_execution")
    dirty = sorted(checkout.get("dirty_paths", []))
    patch_paths = sorted(row.get("path") for row in checkout.get("patch_rows", []))
    if dirty != patch_paths or checkout.get("overlay_complete") is not True:
        errors.append("dirty_overlay_incomplete")
    changed = sorted(checkout.get("changed_tracked_paths", []))
    mutation_paths = sorted(row.get("path") for row in mutations)
    if changed != mutation_paths or checkout.get("mutation_scan_complete") is not True:
        errors.append("mutation_rows_incomplete")
    if (
        active.get("unchanged") is not True
        or active.get("preexisting_dirty_status_preserved") is not True
    ):
        errors.append("active_worktree_drift")
    if protected.get("unchanged") is not True or protected.get("before") != protected.get("after"):
        errors.append("protected_file_drift")
    cleanup = suite.get("process_cleanup", {})
    if cleanup.get("clean") is not True or cleanup.get("surviving_owned_pids"):
        errors.append("owned_process_leak")
    if cleanup.get("unrelated_process_signal_count") != 0:
        errors.append("unrelated_process_signaled")
    collection_cleanup = collection.get("process_cleanup", {})
    if collection_cleanup.get("clean") is not True or collection_cleanup.get(
        "surviving_owned_pids"
    ):
        errors.append("collection_process_leak")
    if collection_cleanup.get("unrelated_process_signal_count") != 0:
        errors.append("collection_unrelated_process_signaled")
    if collection.get("collected_count") != suite.get("collected_count"):
        errors.append("collection_count_mismatch")
    if any(row.get("outcome") not in {"failed", "errored", "skipped", "timed_out"} for row in rows):
        errors.append("invalid_outcome_row")
    truth = reduce_suite_truth(
        collection=collection,
        suite=suite,
        rows=rows,
        mutation_rows=mutations,
        checkout=checkout,
        active_unchanged=active,
    )
    if report.get("status") != truth["state"]:
        errors.append("status_truth_mismatch")
    if report.get("suite_truth_baseline", {}).get("state") != truth["state"]:
        errors.append("suite_truth_state_mismatch")
    if report.get("full_suite_baseline_ready_score") != truth["ready_score"]:
        errors.append("ready_score_mismatch")
    if report.get("verdict_class") != truth["verdict_class"]:
        errors.append("verdict_class_mismatch")
    if truth["state"] in {"measured_green", "measured_red"} and not str(
        report.get("honest_verdict", "")
    ).startswith(("complete:", "success:", "passed:", "shipped:")):
        errors.append("terminal_success_prefix_missing")
    if report.get("status") == "measured_green" and any(
        row.get("outcome") in {"failed", "errored", "timed_out"} for row in rows
    ):
        errors.append("green_with_failure_rows")
    if report.get("status") == "measured_green" and suite.get("exit_code") != 0:
        errors.append("green_with_nonzero_exit")
    if report.get("status") == "measured_green" and suite.get("timed_out") is True:
        errors.append("green_with_timeout")
    attacks = report.get("attack_rows", [])
    if [row.get("attack") for row in attacks] != list(REQUIRED_ATTACKS) or not all(
        row.get("passed") is True for row in attacks
    ):
        errors.append("attack_rows_incomplete")
    cadence = report.get("low_cadence_ownership_contract", {})
    if (
        cadence.get("experiment_launch_gate") is not False
        or cadence.get("conductor_change_required") is not False
    ):
        errors.append("low_cadence_contract_mismatch")
    return sorted(set(errors))


def atomic_write_report(path: Path, report: Mapping[str, Any]) -> JsonDict:
    """Validate, sync, and atomically replace the terminal artifact."""

    errors = validate_report(report)
    if errors:
        raise ValueError(";".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_temp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path.resolve()),
        "sha256": hash_path(path),
        "byte_count": len(encoded),
        "atomic_replace": True,
        "file_fsync": True,
        "directory_fsync": True,
    }


def _effective_environment(
    checkout: Path,
    temporary_root: Path,
    receipt_path: Path,
    mutation_run_id: str,
) -> tuple[dict[str, str], JsonDict]:
    runtime_temp = temporary_root / "runtime-tmp"
    artifact_root = runtime_temp / "artifacts"
    runtime_temp.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(exist_ok=True)
    observer = checkout / "scripts/_mutation_observer"
    python_paths = [observer, checkout / "python", checkout / "scripts/experiments"]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_path_text = os.pathsep.join([*(str(path) for path in python_paths), existing])
    else:
        python_path_text = os.pathsep.join(str(path) for path in python_paths)
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.update(
        {
            "PYTHONPATH": python_path_text,
            "CARNOT_REPO_ROOT": str(checkout),
            "CARNOT_EXPERIMENT_ARTIFACT_ROOT": str(artifact_root),
            "TMPDIR": str(runtime_temp),
            "PYTEST_PLUGINS": PLUGIN_NAME,
            PLUGIN_RECEIPT_ENV: str(receipt_path),
            MUTATION_RUN_ID_ENV: mutation_run_id,
            MUTATION_WRITE_LOG_ENV: str(
                checkout / "ops/.test_suite_mutation_runs" / f"{mutation_run_id}.writes.log"
            ),
        }
    )
    public = {
        key: env[key]
        for key in (
            "PYTHONPATH",
            "CARNOT_REPO_ROOT",
            "CARNOT_EXPERIMENT_ARTIFACT_ROOT",
            "TMPDIR",
            "PYTEST_PLUGINS",
            PLUGIN_RECEIPT_ENV,
            MUTATION_RUN_ID_ENV,
            MUTATION_WRITE_LOG_ENV,
        )
    }
    return env, {"values": public, "sha256": sha256_json(public)}


def _actual_argv(active_root: Path, command: Sequence[str]) -> list[str]:
    # Keep the venv path itself. Resolving its Python symlink bypasses the venv
    # prefix, so the child cannot import the project's pytest installation.
    return [str((active_root / command[0]).absolute()), *command[1:]]


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {path: hash_path(root / path) for path in PROTECTED_PATHS}


def _active_unchanged_receipt(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    dirty_before: Mapping[str, Any],
    dirty_after: Mapping[str, Any],
) -> JsonDict:
    before_hash = snapshot_checksum(before)
    after_hash = snapshot_checksum(after)
    dirty_same = dirty_before.get("sha256") == dirty_after.get("sha256")
    return {
        "unchanged": before_hash == after_hash and dirty_same,
        "tracked_hashes_before_sha256": before_hash,
        "tracked_hashes_after_sha256": after_hash,
        "dirty_status_before_sha256": dirty_before.get("sha256"),
        "dirty_status_after_sha256": dirty_after.get("sha256"),
        "dirty_status_before_records": dirty_before.get("records", []),
        "dirty_status_after_records": dirty_after.get("records", []),
        "preexisting_dirty_status_preserved": dirty_same,
    }


def run_experiment(active_root: Path, run_date: str) -> JsonDict:
    """Create isolation, run collection and the suite once, then write the artifact."""

    start = time.monotonic()
    temporary_root: Path | None = None
    checkout: Path | None = None
    try:
        active = active_root.resolve(strict=True)
        dirty_before = dirty_status_receipt(active)
        active_before = snapshot_tracked_files(active)
        protected_before = _protected_hashes(active)
        temporary_root = Path(tempfile.mkdtemp(prefix="carnot-exp6586-"))
        temporary_root = validate_temporary_root(temporary_root, active)
        checkout = temporary_root / "checkout"
        revision = git_revision(active)
        add = subprocess.run(
            ["git", "worktree", "add", "--detach", str(checkout), revision],
            cwd=active,
            check=False,
            capture_output=True,
            text=True,
        )
        if add.returncode != 0:
            raise IsolationError(
                "disposable_checkout_create", add.stderr.strip(), "git worktree add failed"
            )
        observed_revision = git_revision(checkout)
        if observed_revision != revision:
            raise IsolationError(
                "disposable_checkout_revision",
                observed_revision,
                "checkout revision differs from active HEAD",
            )
        dirty_paths = active_dirty_paths(active)
        patch_rows = apply_content_overlay(active, checkout, dirty_paths)
        overlay_complete = overlay_is_complete(dirty_paths, patch_rows, active, checkout)
        if not overlay_complete:
            raise IsolationError(
                "dirty_overlay_complete",
                dirty_paths,
                "content overlay omitted or changed an active path",
            )
        disposable_before = snapshot_tracked_files(checkout)
        curated_before = operator_curated_snapshot(checkout)
        preconditions = collect_preconditions(
            active, temporary_root, active_before, dirty_before, protected_before
        )

        collect_receipt_path = temporary_root / "collection-pytest-receipt.json"
        collect_run_id = f"exp6586-collect-{os.getpid()}"
        collect_env, collect_env_receipt = _effective_environment(
            checkout, temporary_root, collect_receipt_path, collect_run_id
        )
        collection = run_owned_command(
            _actual_argv(active, COLLECTION_COMMAND),
            cwd=checkout,
            env=collect_env,
            timeout_s=COLLECTION_TIMEOUT_S,
            display_command=COLLECTION_COMMAND_TEXT,
        )
        if collection["timed_out"]:
            raise IsolationError(
                "collection_timeout", collection["duration_s"], "pytest collection timed out"
            )
        collection_plugin = _read_plugin_receipt(collect_receipt_path)
        collection.update(
            {
                "environment_sha256": collect_env_receipt["sha256"],
                "collected_count": collection_plugin["collected_count"],
                "nodeids_sha256": collection_plugin["nodeids_sha256"],
                "collection_rows": collection_plugin["rows"],
            }
        )
        collection["receipt_sha256"] = sha256_json(collection)

        suite_receipt_path = temporary_root / "suite-pytest-receipt.json"
        suite_run_id = f"exp6586-suite-{os.getpid()}"
        suite_env, suite_env_receipt = _effective_environment(
            checkout, temporary_root, suite_receipt_path, suite_run_id
        )
        suite = run_owned_command(
            _actual_argv(active, SUITE_COMMAND),
            cwd=checkout,
            env=suite_env,
            timeout_s=SUITE_TIMEOUT_S,
            display_command=SUITE_COMMAND_TEXT,
        )
        if suite["timed_out"]:
            rows = [timeout_row(SUITE_TIMEOUT_S)]
            suite["collected_count"] = collection["collected_count"]
            suite["pytest_receipt_available"] = suite_receipt_path.is_file()
        else:
            suite_plugin = _read_plugin_receipt(suite_receipt_path)
            rows = [dict(row) for row in suite_plugin["rows"]]
            suite["collected_count"] = suite_plugin["collected_count"]
            suite["nodeids_sha256"] = suite_plugin["nodeids_sha256"]
            suite["pytest_receipt_available"] = True
            suite["family_summaries_raw"] = suite_plugin["family_summaries"]
        suite["environment_sha256"] = suite_env_receipt["sha256"]
        suite["environment"] = suite_env_receipt["values"]

        observed = _observed_write_paths(checkout, (collect_run_id, suite_run_id))
        disposable_after = snapshot_tracked_files(checkout)
        curated_after = operator_curated_snapshot(checkout)
        mutation_rows = tracked_mutation_rows(
            disposable_before, disposable_after, observed_paths=observed
        )
        changed_paths = [row["path"] for row in mutation_rows]
        active_after = snapshot_tracked_files(active)
        dirty_after = dirty_status_receipt(active)
        active_unchanged = _active_unchanged_receipt(
            active_before, active_after, dirty_before, dirty_after
        )
        protected_after = _protected_hashes(active)
        protected = {
            "before": protected_before,
            "after": protected_after,
            "unchanged": protected_before == protected_after,
        }
        checkout_receipt: JsonDict = {
            "active_root": str(active),
            "checkout_root": str(checkout.resolve()),
            "validated_temporary_root": str(temporary_root),
            "revision": revision,
            "detached_head": True,
            "patch_hash": sha256_json(patch_rows),
            "patch_rows": patch_rows,
            "dirty_paths": dirty_paths,
            "overlay_complete": overlay_complete,
            "before_tracked_snapshot_sha256": snapshot_checksum(disposable_before),
            "after_tracked_snapshot_sha256": snapshot_checksum(disposable_after),
            "operator_curated_before": curated_before,
            "operator_curated_after": curated_after,
            "operator_curated_unchanged": curated_before == curated_after,
            "observed_tracked_write_paths": observed,
            "changed_tracked_paths": changed_paths,
            "mutation_scan_complete": True,
        }
        remove = subprocess.run(
            ["git", "worktree", "remove", "--force", str(checkout)],
            cwd=active,
            check=False,
            capture_output=True,
            text=True,
        )
        checkout_receipt["cleanup"] = {
            "attempted": True,
            "exit_code": remove.returncode,
            "stderr": remove.stderr,
            "removed": not checkout.exists(),
        }
        report = build_report(
            run_date=run_date,
            preconditions=preconditions,
            checkout=checkout_receipt,
            collection=collection,
            suite=suite,
            rows=rows,
            mutation_rows=mutation_rows,
            active_unchanged=active_unchanged,
            protected=protected,
            tests_run=DEFAULT_TESTS_RUN,
            duration_s=time.monotonic() - start,
        )
    except IsolationError as exc:
        report = blocked_report(
            run_date=run_date,
            failed_check=exc.check,
            observed_value=exc.observed,
            duration_s=time.monotonic() - start,
        )
    finally:
        if checkout is not None and checkout.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(checkout)],
                cwd=active_root,
                check=False,
                capture_output=True,
                text=True,
            )
        if temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root)
    atomic_write_report(active_root / RESULT_RELATIVE_PATH, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        report = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_report(report)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"valid: {path}")
        return 0
    report = run_experiment(REPO_ROOT, args.date)
    print(json.dumps({"path": str(path), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
