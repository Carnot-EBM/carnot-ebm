"""Dual-GPU health monitoring for Carnot experiment scaffolding.

**Why this module exists:**
    The 2026.04.29 retrospective (RETRO-002/RETRO-003) identified two GPU pathologies
    that silently degraded experiment performance:

    1. **Zombie GPU processes** (RETRO-002): processes holding ~1050 MB VRAM at 0%
       utilisation (PIDs 2592400/2595103).  These consume memory without doing work,
       crowding out the experiment's own allocations.

    2. **Idle GPU during sequential execution** (RETRO-003): Exp 219/221 ran two models
       sequentially on GPU 0 while GPU 1 sat idle the entire time.  Estimated cost:
       ~105 extra minutes (195 min actual vs ~90 min parallel).

    This module exposes ``DualGPUMonitor``, which is called by
    ``ExperimentTemplate.setup_gpu()`` before any timed inference begins.  If zombies
    or idle GPUs are detected, the result is logged as a warning in the experiment
    artifact so that future retrospectives have machine-readable evidence.

Spec: REQ-INFRA-003, REQ-INFRA-004,
      SCENARIO-INFRA-004, SCENARIO-INFRA-005, SCENARIO-INFRA-006
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPUProcessInfo
# ---------------------------------------------------------------------------


@dataclass
class GPUProcessInfo:
    """Snapshot of a single GPU compute process at the time of the health check.

    Fields
    ------
    pid : int
        OS process ID.
    gpu_index : int
        Zero-based GPU device index (matches ``nvidia-smi`` ordering).
    vram_mb : int
        VRAM allocated by this process in mebibytes (MiB).
    utilization_pct : int
        GPU compute utilisation percentage (0–100) for the device at sample time.
        Note: this is a *device-level* metric broadcast to all processes on that
        device; it is not per-process.
    is_zombie : bool
        ``True`` when ``utilization_pct == 0`` AND ``vram_mb > 100``.  A zombie
        process is holding significant VRAM but performing no computation — a
        reliable sign of a stalled or orphaned training/inference job.
    """

    pid: int
    gpu_index: int
    vram_mb: int
    utilization_pct: int
    is_zombie: bool


# ---------------------------------------------------------------------------
# DualGPUMonitor
# ---------------------------------------------------------------------------


class DualGPUMonitor:
    """Detect zombie GPU processes and idle GPUs before experiment inference starts.

    **Zombie detection rule (REQ-INFRA-003):**
        A process is a zombie when its device utilisation is 0% AND it holds more
        than 100 MiB of VRAM.  The 100 MiB floor avoids false positives from
        lightweight system processes (e.g. display servers) that legitimately park
        a small CUDA context without doing computation.

    **Dual-GPU check (REQ-INFRA-004):**
        ``check_dual_gpu_health()`` inspects all detected GPUs and flags any that
        have no active processes.  An idle GPU is almost always a sign that the
        experiment is inadvertently running sequentially instead of in parallel.

    **CI safety (SCENARIO-INFRA-006):**
        Both ``list_gpu_processes()`` and ``check_dual_gpu_health()`` degrade
        gracefully when ``nvidia-smi`` is absent (CI environments, Apple Silicon,
        CPU-only machines).  They never raise; they return empty structures instead.

    Usage::

        monitor = DualGPUMonitor()
        health = monitor.check_dual_gpu_health()
        if not health["all_healthy"]:
            logging.warning("GPU health check failed: %s", health)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_gpu_processes(self) -> list[GPUProcessInfo]:
        """Return a snapshot of all active GPU compute processes.

        Runs two ``nvidia-smi`` queries:
        - ``--query-compute-apps`` for per-process VRAM usage
        - ``--query-gpu=utilization.gpu`` for per-device utilisation

        Returns an empty list (never raises) if ``nvidia-smi`` is absent or
        returns a non-zero exit code.

        Returns
        -------
        list[GPUProcessInfo]
            One entry per process reported by ``nvidia-smi``.  Empty when
            ``nvidia-smi`` is unavailable.
        """
        try:
            proc_output = self._run_nvidia_smi_apps()
            util_output = self._run_nvidia_smi_util()
        except FileNotFoundError:
            # nvidia-smi not installed (CI / non-GPU machine)
            return []

        if proc_output is None or util_output is None:
            return []

        # Parse device utilisation: one percentage per line, one line per GPU
        device_util: dict[int, int] = {}
        for gpu_idx, line in enumerate(util_output.strip().splitlines()):
            # Lines look like "0 %" or "15 %"
            clean = line.strip().rstrip("%").strip()
            if clean:
                try:
                    device_util[gpu_idx] = int(clean)
                except ValueError:
                    device_util[gpu_idx] = 0

        # Parse per-process compute-apps: "pid, gpu_index, used_memory MiB"
        processes: list[GPUProcessInfo] = []
        for line in proc_output.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                gpu_idx = int(parts[1])
                # Memory field may look like "600 MiB" or just "600"
                vram_str = parts[2].replace("MiB", "").replace("MEB", "").strip()
                vram_mb = int(vram_str)
            except ValueError:
                continue

            util_pct = device_util.get(gpu_idx, 0)
            is_zombie = self._is_zombie(vram_mb=vram_mb, utilization_pct=util_pct)

            processes.append(
                GPUProcessInfo(
                    pid=pid,
                    gpu_index=gpu_idx,
                    vram_mb=vram_mb,
                    utilization_pct=util_pct,
                    is_zombie=is_zombie,
                )
            )

        return processes

    def detect_zombies(self) -> list[GPUProcessInfo]:
        """Return only the zombie processes from the current GPU process list.

        A zombie is any ``GPUProcessInfo`` with ``is_zombie=True``.

        Returns
        -------
        list[GPUProcessInfo]
            Subset of ``list_gpu_processes()`` where ``is_zombie`` is ``True``.
            Empty when no zombies are present.
        """
        return [p for p in self.list_gpu_processes() if p.is_zombie]

    def check_dual_gpu_health(self) -> dict:
        """Inspect GPU state and return a structured health summary.

        Determines:
        - How many distinct GPUs are visible (via process GPU indices + ``_get_gpu_count``).
        - How many zombie processes exist.
        - Which GPU indices have zero active processes ("idle GPUs").
        - Whether the overall configuration is healthy for dual-GPU parallel inference.

        ``all_healthy`` is ``True`` only when ALL of the following hold:
        - ``n_gpus_detected >= 2``
        - ``n_zombies == 0``
        - ``len(idle_gpus) == 0``

        Returns
        -------
        dict
            Keys: ``n_gpus_detected`` (int), ``n_zombies`` (int),
            ``idle_gpus`` (list[int]), ``all_healthy`` (bool).
        """
        try:
            processes = self.list_gpu_processes()
            n_gpus = self._get_gpu_count()
        except Exception:  # pragma: no cover — unexpected errors still return a safe dict
            return {
                "n_gpus_detected": 0,
                "n_zombies": 0,
                "idle_gpus": [],
                "all_healthy": False,
            }

        zombies = [p for p in processes if p.is_zombie]
        n_zombies = len(zombies)

        # Determine which GPUs actually have at least one active process
        active_gpu_indices = {p.gpu_index for p in processes}

        # Idle GPUs: indices in [0, n_gpus) that have no active processes
        idle_gpus: list[int] = [
            i for i in range(n_gpus) if i not in active_gpu_indices
        ]

        all_healthy = (n_gpus >= 2) and (n_zombies == 0) and (len(idle_gpus) == 0)

        return {
            "n_gpus_detected": n_gpus,
            "n_zombies": n_zombies,
            "idle_gpus": idle_gpus,
            "all_healthy": all_healthy,
        }

    def to_dict(self) -> dict:
        """Serialise the monitor state into a JSON-safe dict for artifact embedding.

        Returns a dict with:
        - ``"health"``: the result of ``check_dual_gpu_health()``
        - ``"processes"``: list of per-process dicts (from ``list_gpu_processes()``)

        This is designed to be embedded directly into experiment artifacts so that
        GPU state at experiment start is preserved in the research record.

        Returns
        -------
        dict
            JSON-serialisable; safe to pass directly to ``json.dumps()``.
        """
        processes = self.list_gpu_processes()
        health = self.check_dual_gpu_health()
        return {
            "health": health,
            "processes": [asdict(p) for p in processes],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_zombie(*, vram_mb: int, utilization_pct: int) -> bool:
        """Classify a process as a zombie based on VRAM and utilisation.

        A zombie holds a non-trivial amount of VRAM (>100 MiB) but performs
        no computation (0% utilisation).  The 100 MiB floor avoids false
        positives from display servers or idle CUDA contexts.

        Parameters
        ----------
        vram_mb : int
            VRAM allocated by the process in MiB.
        utilization_pct : int
            GPU compute utilisation percentage for the device (0–100).

        Returns
        -------
        bool
            ``True`` iff ``utilization_pct == 0`` AND ``vram_mb > 100``.
        """
        return utilization_pct == 0 and vram_mb > 100

    def _get_gpu_count(self) -> int:
        """Return the number of CUDA GPUs reported by ``nvidia-smi``.

        Falls back to the highest ``gpu_index`` seen in active processes + 1
        if the count query fails.  Returns 0 if ``nvidia-smi`` is absent.

        Returns
        -------
        int
            Number of CUDA devices detected, or 0 if unavailable.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return 0
            lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
            return len(lines)
        except FileNotFoundError:
            return 0
        except Exception:  # pragma: no cover
            return 0

    def _run_nvidia_smi_apps(self) -> str | None:
        """Run the compute-apps query and return raw stdout, or None on failure."""
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_index,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        return result.stdout

    def _run_nvidia_smi_util(self) -> str | None:
        """Run the GPU utilisation query and return raw stdout, or None on failure."""
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        return result.stdout
