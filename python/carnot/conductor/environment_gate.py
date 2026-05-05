"""Environment gate artifact builder for milestone .104 activation.

The conductor already records many failures after it tries to launch work. This
module exists one step earlier: it turns disk space, inode headroom, repeated
pre-test output, stale .103 skeletons, and roadmap audit health into one JSON
artifact that downstream tasks can gate on. The implementation is deliberately
read-only except for the requested Exp 1337 artifact, because this gate should
describe the environment rather than repair it by deleting files or rewriting
historical results.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


DEFAULT_OUTPUT_PATH = Path(
    "results/experiment_1337_environment_gate_disk_pretest_stale_skeleton_audit.json"
)
DEFAULT_FOCUSED_PRETEST_TARGET = Path("tests/python/test_conductor_pretest_cache.py")
EXP1337_TASK_ID = "exp1337-environment-gate-disk-pretest-stale-skeleton-audit"
DISK_QUOTA_SIGNATURE = "Codex CLI error: [Errno 122] Disk quota exceeded"
PRETEST_PREFIX = "Pre-tests failing, self-heal failed:"
PRETEST_COUNTS_RE = re.compile(r"1 failed,\s*86 passed,\s*1 warning")
MIN_DISK_FREE_GB = 5.0
MIN_INODE_FREE_PCT = 5.0


@dataclass(frozen=True)
class FileSystemStats:
    disk_free_gb: float
    inode_free_pct: float


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], Path, int], CommandResult]
WriteObserver = Callable[[Path, dict[str, Any]], None]


def collect_filesystem_stats(root: Path) -> FileSystemStats:
    """Measure free blocks and inodes without mutating the filesystem.

    Disk-quota incidents can be caused by either byte exhaustion or inode
    exhaustion. `statvfs` exposes both from the same project-root mount, which
    avoids probing by creating temporary files.
    """
    stat = os.statvfs(root)
    disk_free_gb = stat.f_bavail * stat.f_frsize / (1024**3)
    inode_free_pct = 100.0 if stat.f_files == 0 else stat.f_favail / stat.f_files * 100.0
    return FileSystemStats(
        disk_free_gb=round(disk_free_gb, 3),
        inode_free_pct=round(inode_free_pct, 3),
    )


def _run_subprocess(cmd: list[str], cwd: Path, timeout_s: int) -> CommandResult:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_pretest_signature(text: str) -> str:
    if PRETEST_COUNTS_RE.search(text):
        return f"{PRETEST_PREFIX} 1 failed, 86 passed, 1 warning"
    return ""


def extract_environment_signatures(root: Path) -> dict[str, Any]:
    """Extract exact known .103 signatures from local evidence files.

    The artifact should preserve the operator-visible failure wording. That
    keeps downstream pruning tied to the known .103 failure instead of a new
    paraphrase that the scheduler cannot compare reliably.
    """
    conductor_log = (root / "ops/conductor-log.md").read_text(encoding="utf-8")
    retro = _read_json(root / "results/operational_retro_2026_04_103.json")
    retro_text = json.dumps(retro, sort_keys=True)
    combined = f"{conductor_log}\n{retro_text}"
    disk_signature = DISK_QUOTA_SIGNATURE if DISK_QUOTA_SIGNATURE in combined else ""
    pretest_signature = _extract_pretest_signature(combined)
    return {
        "disk_quota_signature": disk_signature,
        "disk_quota_occurrences": conductor_log.count(DISK_QUOTA_SIGNATURE),
        "pretest_signature": pretest_signature,
        "pretest_signature_occurrences": len(PRETEST_COUNTS_RE.findall(conductor_log)),
        "focused_pretest_signature_active": False,
    }


def _is_bootstrap_skeleton(payload: dict[str, Any]) -> bool:
    if payload.get("status") == "in_progress":
        return True
    if payload.get("honest_verdict") == "in_progress":
        substantive_values = [
            value
            for key, value in payload.items()
            if key not in {"artifact_metadata", "experiment_id", "experiment", "run_date", "status", "title", "honest_verdict"}
        ]
        return not any(value not in (None, [], {}, "", False) for value in substantive_values)
    return False


def find_stale_103_artifacts(root: Path) -> list[str]:
    stale_paths: list[str] = []
    for path in sorted((root / "results").glob("experiment_*.json")):
        match = re.match(r"experiment_(\d+)", path.name)
        if not match:
            continue
        exp_id = int(match.group(1))
        if not 1323 <= exp_id <= 1336:
            continue
        if _is_bootstrap_skeleton(_read_json(path)):
            stale_paths.append(_relative_path(root, path))
    return stale_paths


def _command_output(result: CommandResult) -> str:
    return "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)


def run_focused_pretest(
    root: Path,
    target: Path,
    command_runner: CommandRunner,
    timeout_s: int = 120,
) -> dict[str, Any]:
    target_path = target if target.is_absolute() else root / target
    target_label = _relative_path(root, target_path)
    if not target_path.exists():
        return {
            "status": "not_available",
            "command": [],
            "missing_path": target_label,
            "returncode": None,
            "output_excerpt": "",
            "repeated_signature_active": False,
        }

    command_target = target_label
    cmd = [".venv/bin/pytest", command_target, "-q", "--no-cov"]
    result = command_runner(cmd, root, timeout_s)
    output = _command_output(result)
    repeated_active = bool(PRETEST_COUNTS_RE.search(output))
    status = "passed" if result.returncode == 0 and not repeated_active else "failed"
    return {
        "status": status,
        "command": cmd,
        "missing_path": None,
        "returncode": result.returncode,
        "output_excerpt": output[:1000],
        "repeated_signature_active": repeated_active,
    }


def _run_optional_roadmap_command(
    root: Path,
    script: Path,
    command_runner: CommandRunner,
    timeout_s: int = 120,
) -> dict[str, Any]:
    script_path = root / script
    if not script_path.exists():
        return {
            "status": "not_available",
            "command": [],
            "missing_path": script.as_posix(),
            "returncode": None,
            "actionable_failures": [],
        }

    cmd = ["python3", script.as_posix(), "research-roadmap-next.yaml"]
    result = command_runner(cmd, root, timeout_s)
    output = _command_output(result)
    failures = [line.strip() for line in output.splitlines() if line.strip()][:8]
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "command": cmd,
        "missing_path": None,
        "returncode": result.returncode,
        "actionable_failures": [] if result.returncode == 0 else failures,
        "output_excerpt": output[:1000],
    }


def _load_roadmap_tasks(root: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load((root / "research-roadmap.yaml").read_text(encoding="utf-8")) or {}
    tasks = data.get("tasks", []) or []
    return [task for task in tasks if isinstance(task, dict)]


def _environment_gate_directly_blocks(task: dict[str, Any]) -> bool:
    for gate in task.get("gated_on") or []:
        if not isinstance(gate, dict):
            continue
        if gate.get("upstream") == EXP1337_TASK_ID and gate.get("artifact_field") == "environment_ready":
            return True
    return False


def _tasks_to_prune_until_gate_fixed(root: Path) -> list[dict[str, str]]:
    tasks = _load_roadmap_tasks(root)
    blocked_ids = {str(task.get("id")) for task in tasks if _environment_gate_directly_blocks(task)}
    changed = True
    while changed:
        changed = False
        for task in tasks:
            task_id = str(task.get("id") or "")
            if not task_id or task_id in blocked_ids:
                continue
            gates = task.get("gated_on") or []
            if any(isinstance(gate, dict) and gate.get("upstream") in blocked_ids for gate in gates):
                blocked_ids.add(task_id)
                changed = True
    return [
        {
            "task_id": task_id,
            "reason": "blocked_until_exp1337_environment_ready_true",
        }
        for task_id in sorted(blocked_ids)
    ]


def _honest_verdict(
    disk_quota_ok: bool,
    focused_pretest: dict[str, Any],
    repeated_active: bool,
    stale_count: int,
) -> str:
    if not disk_quota_ok:
        return "blocked_disk_quota_or_inode_gate"
    if repeated_active:
        return "blocked_repeated_pretest_signature_active"
    if focused_pretest["status"] == "not_available":
        return "blocked_focused_pretest_not_available"
    if focused_pretest["status"] != "passed":
        return "blocked_focused_pretest_failed"
    if stale_count:
        return "environment_ready_stale_103_artifacts_classified"
    return "environment_ready"


def build_environment_gate_artifact(
    root: Path,
    run_date: str,
    focused_pretest_target: Path = DEFAULT_FOCUSED_PRETEST_TARGET,
    command_runner: CommandRunner = _run_subprocess,
    min_disk_free_gb: float = MIN_DISK_FREE_GB,
    min_inode_free_pct: float = MIN_INODE_FREE_PCT,
) -> dict[str, Any]:
    root = root.resolve()
    fs_stats = collect_filesystem_stats(root)
    disk_quota_ok = (
        fs_stats.disk_free_gb >= min_disk_free_gb
        and fs_stats.inode_free_pct >= min_inode_free_pct
    )
    signatures = extract_environment_signatures(root)
    stale_paths = find_stale_103_artifacts(root)
    focused_pretest = run_focused_pretest(root, focused_pretest_target, command_runner)
    repeated_active = focused_pretest["repeated_signature_active"]
    signatures["focused_pretest_signature_active"] = repeated_active
    roadmap_health = {
        "prior_failures": _run_optional_roadmap_command(
            root,
            Path("scripts/validate_prior_failures.py"),
            command_runner,
        ),
        "roadmap_gates": _run_optional_roadmap_command(
            root,
            Path("scripts/audit_roadmap_gates.py"),
            command_runner,
        ),
    }
    environment_ready = (
        disk_quota_ok
        and focused_pretest["status"] == "passed"
        and not repeated_active
    )
    recommended_task_pruning = [] if environment_ready else _tasks_to_prune_until_gate_fixed(root)

    return {
        "status": "complete",
        "artifact_metadata": {
            "project_root": str(root),
            "run_date": run_date,
            "spec": "REQ-INFRA-1337",
        },
        "disk_free_gb": fs_stats.disk_free_gb,
        "inode_free_pct": fs_stats.inode_free_pct,
        "disk_quota_ok": disk_quota_ok,
        "disk_quota_gate": {
            "min_disk_free_gb": min_disk_free_gb,
            "min_inode_free_pct": min_inode_free_pct,
        },
        "repeated_pretest_signature": signatures,
        "focused_pretest_status": focused_pretest["status"],
        "focused_pretest": focused_pretest,
        "stale_artifact_paths": stale_paths,
        "stale_skeleton_count": len(stale_paths),
        "stale_artifacts_classified": True,
        "roadmap_health": roadmap_health,
        "environment_ready": environment_ready,
        "recommended_task_pruning": recommended_task_pruning,
        "honest_verdict": _honest_verdict(
            disk_quota_ok,
            focused_pretest,
            repeated_active,
            len(stale_paths),
        ),
    }


def _write_payload(path: Path, payload: dict[str, Any], observer: WriteObserver | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if observer is not None:
        observer(path, payload)


def write_environment_gate_artifact(
    root: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    run_date: str = "20260505",
    focused_pretest_target: Path = DEFAULT_FOCUSED_PRETEST_TARGET,
    command_runner: CommandRunner = _run_subprocess,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    final_output_path = output_path if output_path.is_absolute() else root / output_path
    _write_payload(final_output_path, {"status": "in_progress"}, write_observer)
    artifact = build_environment_gate_artifact(
        root,
        run_date=run_date,
        focused_pretest_target=focused_pretest_target,
        command_runner=command_runner,
    )
    _write_payload(final_output_path, artifact, write_observer)
    return artifact
