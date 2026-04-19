"""ExpandedGPUReaper — broad-heuristic GPU VRAM reclaimer for stale out-of-tree processes.

**Why this module exists (RETRO-033, seven consecutive missed milestones):**
    GPUVRAMGateV2 (Exp 487) and JITVRAMCheck (Exp 513) both exist and run before every
    GPU experiment, yet RETRO-033 missed seven consecutive milestones with the same
    root cause: stale pytest subprocesses (parent='pytest tests/python', child=
    'python3 -u -c ...') accumulated across milestones and collectively pinned large
    chunks of GPU VRAM.

    The existing gates use pattern-matched whitelists that miss 'python3 -u -c ...'
    child processes because those are generic Python subprocess launchers — completely
    indistinguishable from legitimate conductor subagent children by name alone.

    The fix: switch from name-based whitelisting to **process-subtree membership**.
    Any GPU-holding process that (a) is NOT a descendant of the current conductor process,
    (b) is holding >= MIN_VRAM_MB of VRAM, AND (c) has been alive >= MIN_AGE_S seconds is
    a reaping candidate.  This catches the stale pytest children because pytest itself
    was spawned by a prior conductor run (now dead), so the orphaned children are not in
    the current subtree regardless of their name.

**Why the process-subtree check (REQ-INFRA-067):**
    Killing by name risks terminating the reaper's own legitimate subagent children
    (e.g. a model-loading worker spawned 45 seconds ago by the current conductor run).
    A subtree check uses the kernel's actual parent-child relationships, which are exact
    and cannot be spoofed by process name.

**Why the age threshold (REQ-INFRA-067):**
    A freshly-spawned subagent that legitimately allocated VRAM 30 seconds ago should not
    be reaped.  The age threshold (default 1800 s = 30 min) ensures we only touch processes
    that have been running far longer than any normal subagent lifespan.

**Why the dry_run flag (REQ-INFRA-069):**
    Reaping is destructive — a wrong kill in a live experiment is catastrophic.  dry_run=True
    lets CI, audit scripts (like Exp 525), and human operators inspect the candidate list
    without actually sending SIGKILL.  The honest_verdict field documents which mode ran.

**Why no pyxrt import (REQ-INFRA-069):**
    The NPU stack (pyxrt) is not always installed and should not be required for a generic
    GPU VRAM reclamation utility.  This module uses only subprocess calls to nvidia-smi and
    ps, which are universally available on any CUDA-enabled Linux host.

Spec: REQ-INFRA-067, REQ-INFRA-068, REQ-INFRA-069,
      SCENARIO-INFRA-076, SCENARIO-INFRA-077, SCENARIO-INFRA-078
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

__all__ = [
    "ExpandedGPUReaper",
    "ExpandedGPUReaperConfig",
    "ExpandedGPUReapResult",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ExpandedGPUReaperConfig:
    """Configuration for ExpandedGPUReaper.

    Parameters
    ----------
    min_vram_mb : int
        Minimum VRAM usage (in MiB) a process must be holding to be a reap
        candidate.  Default 1024 MiB (1 GiB).  Processes using less than this
        are almost certainly not responsible for VRAM exhaustion, and killing
        them would just add noise to the audit log.
    min_age_s : int
        Minimum process age in seconds before a process is eligible for reaping.
        Default 1800 s (30 min).  A freshly-spawned model-loading worker that
        legitimately holds 10 GiB of VRAM should not be reaped; anything running
        for 30+ minutes without being a child of the current conductor is an orphan.
    dry_run : bool
        When True, compute and log the candidate list but do NOT send SIGKILL.
        The honest_verdict is 'reap_dry_run_complete' instead of 'reap_complete'.
        Use dry_run=True for CI, audit scripts, and human review before enabling
        live reaping in production.
    """

    min_vram_mb: int = 1024
    min_age_s: int = 1800
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ExpandedGPUReapResult:
    """Per-run result from ExpandedGPUReaper.reap().

    Fields
    ------
    killed : list[dict]
        PIDs that were sent SIGKILL (empty when dry_run=True).
        Each entry: {'pid': int, 'used_memory_mb': int, 'process_name': str,
                     'age_s': int, 'action': 'killed'}.
    skipped : list[dict]
        GPU-holding processes that were examined but NOT reaped, with the
        reason.  Each entry: {'pid': int, 'used_memory_mb': int,
        'process_name': str, 'age_s': int, 'reason': str}.
        Reasons: 'in_our_subtree', 'below_min_vram', 'below_min_age',
                 'kill_error', 'dry_run_candidate'.
    total_vram_freed_mb : int
        Sum of used_memory_mb for all killed entries (0 if dry_run).
    honest_verdict : str
        One of:
          'reap_complete'        — live run, some or zero processes killed
          'reap_dry_run_complete' — dry_run=True, candidates identified only
          'no_nvidia_smi_no_reap' — nvidia-smi not in PATH; nothing was done
    """

    killed: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    total_vram_freed_mb: int = 0
    honest_verdict: str = "reap_complete"


# ---------------------------------------------------------------------------
# Reaper
# ---------------------------------------------------------------------------


class ExpandedGPUReaper:
    """Broad-heuristic GPU VRAM reclaimer.

    Uses nvidia-smi to enumerate GPU-holding processes, then kills any that
    are outside the current process subtree AND above the VRAM and age thresholds.

    This is intentionally NOT wired into any conductor path in this module.
    Integration into GPUVRAMGateV2 (REQ-INFRA-068) is a separate step requiring
    human review, tracked in openspec/change-proposals/env-hardening-and-reruns.md.

    Usage
    -----
    >>> cfg = ExpandedGPUReaperConfig(min_vram_mb=1024, min_age_s=1800, dry_run=True)
    >>> reaper = ExpandedGPUReaper(cfg)
    >>> result = reaper.reap()
    >>> print(result.honest_verdict)
    """

    def __init__(self, config: ExpandedGPUReaperConfig | None = None) -> None:
        self._cfg = config or ExpandedGPUReaperConfig()
        self._root_pid: int = os.getpid()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_gpu_processes(self) -> list[dict]:
        """Parse nvidia-smi --query-compute-apps output into a list of dicts.

        Returns an empty list when nvidia-smi is not installed (CI stub path).
        Each dict contains:
          pid (int), used_memory_mb (int), process_name (str)

        Why nvidia-smi --query-compute-apps (not pynvml):
            pynvml requires an extra Python package and initialization sequence.
            nvidia-smi ships with every CUDA driver installation and is available
            as a subprocess call without any Python binding overhead.  This keeps
            the reaper dependency-free (REQ-INFRA-069).
        """
        if not shutil.which("nvidia-smi"):
            _log.debug("nvidia-smi not in PATH; returning empty GPU process list")
            return []
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory,process_name",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            _log.warning("nvidia-smi failed (exit %d): %s", exc.returncode, exc.output)
            return []

        processes: list[dict] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                used_memory_mb = int(parts[1])
                process_name = parts[2]
                processes.append(
                    {
                        "pid": pid,
                        "used_memory_mb": used_memory_mb,
                        "process_name": process_name,
                    }
                )
            except ValueError:
                _log.debug("Could not parse nvidia-smi line: %r", line)
        return processes

    def _process_age_s(self, pid: int) -> int:
        """Return how long PID has been running, in seconds, using ps -o etimes.

        Returns -1 if the process no longer exists or ps fails.  A -1 age is
        treated as 'always eligible' rather than 'skip', because a process that
        disappeared between the nvidia-smi poll and the age check is no longer
        holding VRAM anyway — the subsequent kill attempt will harmlessly fail.

        Why etimes instead of etime:
            etime is a human-readable string ([DD-]HH:MM:SS) that requires
            parsing.  etimes is elapsed seconds as an integer — directly usable.
        """
        try:
            out = subprocess.check_output(
                ["ps", "-o", "etimes=", "-p", str(pid)],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return int(out.strip())
        except (subprocess.CalledProcessError, ValueError):
            return -1

    def _in_our_subtree(self, pid: int, root_pid: int) -> bool:
        """Return True iff pid is a descendant of root_pid.

        Walks the ps parent chain upward from pid until it either reaches
        root_pid (True) or PID 1 / an error (False).

        Why walk upward (not downward):
            Reading /proc/<pid>/stat for every descendent requires enumerating
            all processes first.  Walking upward from the candidate stops as soon
            as we find root_pid, which is O(depth) rather than O(n_processes).
        """
        current = pid
        visited: set[int] = set()
        while True:
            if current == root_pid:
                return True
            if current in visited or current <= 1:
                return False
            visited.add(current)
            try:
                stat_path = f"/proc/{current}/stat"
                with open(stat_path) as fh:
                    content = fh.read()
                # /proc/<pid>/stat format: pid (comm) state ppid ...
                # The comm field may contain spaces and parentheses; split after last ')'
                last_paren = content.rfind(")")
                fields = content[last_paren + 2 :].split()
                ppid = int(fields[1])  # field index 3 (0-based after comm/state)
                current = ppid
            except (OSError, IndexError, ValueError):
                return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reap(self) -> ExpandedGPUReapResult:
        """Walk GPU processes and kill eligible orphans.

        A process is eligible when ALL three conditions hold:
          1. used_memory_mb >= config.min_vram_mb
          2. NOT a descendant of the current process (self._root_pid)
          3. age_s >= config.min_age_s  (or age_s == -1 meaning already gone)

        When config.dry_run is True, candidates are logged but NOT killed.

        Returns an ExpandedGPUReapResult with per-pid action logs and a
        summary verdict in honest_verdict.
        """
        if not shutil.which("nvidia-smi"):
            return ExpandedGPUReapResult(honest_verdict="no_nvidia_smi_no_reap")

        gpu_procs = self._list_gpu_processes()
        killed: list[dict] = []
        skipped: list[dict] = []

        for proc in gpu_procs:
            pid = proc["pid"]
            vram_mb = proc["used_memory_mb"]
            name = proc["process_name"]

            if vram_mb < self._cfg.min_vram_mb:
                skipped.append(
                    {
                        "pid": pid,
                        "used_memory_mb": vram_mb,
                        "process_name": name,
                        "age_s": -1,
                        "reason": "below_min_vram",
                    }
                )
                continue

            if self._in_our_subtree(pid, self._root_pid):
                age_s = self._process_age_s(pid)
                skipped.append(
                    {
                        "pid": pid,
                        "used_memory_mb": vram_mb,
                        "process_name": name,
                        "age_s": age_s,
                        "reason": "in_our_subtree",
                    }
                )
                continue

            age_s = self._process_age_s(pid)
            # age_s == -1 means process is already gone; treat as eligible age
            if age_s != -1 and age_s < self._cfg.min_age_s:
                skipped.append(
                    {
                        "pid": pid,
                        "used_memory_mb": vram_mb,
                        "process_name": name,
                        "age_s": age_s,
                        "reason": "below_min_age",
                    }
                )
                continue

            if self._cfg.dry_run:
                skipped.append(
                    {
                        "pid": pid,
                        "used_memory_mb": vram_mb,
                        "process_name": name,
                        "age_s": age_s,
                        "reason": "dry_run_candidate",
                    }
                )
                _log.info(
                    "DRY RUN: would kill pid=%d name=%r vram=%d MiB age=%d s",
                    pid,
                    name,
                    vram_mb,
                    age_s,
                )
                continue

            # Live kill path
            try:
                os.kill(pid, signal.SIGKILL)
                _log.warning(
                    "REAPED pid=%d name=%r vram=%d MiB age=%d s",
                    pid,
                    name,
                    vram_mb,
                    age_s,
                )
                killed.append(
                    {
                        "pid": pid,
                        "used_memory_mb": vram_mb,
                        "process_name": name,
                        "age_s": age_s,
                        "action": "killed",
                    }
                )
            except (ProcessLookupError, PermissionError) as exc:
                _log.warning("Could not kill pid=%d: %s", pid, exc)
                skipped.append(
                    {
                        "pid": pid,
                        "used_memory_mb": vram_mb,
                        "process_name": name,
                        "age_s": age_s,
                        "reason": "kill_error",
                    }
                )

        total_freed = sum(e["used_memory_mb"] for e in killed)
        verdict = "reap_dry_run_complete" if self._cfg.dry_run else "reap_complete"
        return ExpandedGPUReapResult(
            killed=killed,
            skipped=skipped,
            total_vram_freed_mb=total_freed,
            honest_verdict=verdict,
        )
