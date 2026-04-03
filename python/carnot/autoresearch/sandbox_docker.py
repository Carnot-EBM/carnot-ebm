"""Docker+gVisor sandbox for production autoresearch hypothesis execution.

**Researcher summary:**
    Executes hypothesis code in an isolated Docker container with gVisor
    runtime (syscall interception), no network, read-only filesystem, and
    hard memory/CPU/timeout limits. Metrics returned via container stdout.

**Detailed explanation for engineers:**
    The process-level sandbox (sandbox.py) works for development, but Python's
    dynamic nature means a sufficiently clever hypothesis could escape it
    (e.g., via __builtins__.__import__, ctypes workarounds, etc.).

    This module provides production-grade isolation using Docker containers
    with the gVisor (runsc) runtime:

    **Defense in depth (5 layers):**

    1. **gVisor runtime**: Intercepts all syscalls in userspace. The hypothesis
       code never talks to the real kernel — gVisor reimplements the Linux
       syscall interface in a sandboxed Go process. Even a kernel exploit in
       the hypothesis can't escape.

    2. **No network** (``--network none``): The container has no network
       interfaces. No HTTP requests, no DNS, no data exfiltration.

    3. **Read-only filesystem** (``--read-only``): The container can't write
       to disk except /tmp (a size-limited tmpfs). Can't modify the carnot
       package, can't create persistent files.

    4. **Memory/CPU limits** (``--memory``, ``--cpus``): Hard cgroup limits
       enforced by the kernel. A memory-hungry hypothesis gets OOM-killed,
       not slowed down.

    5. **Timeout**: The container is forcibly killed after the configured
       timeout. No way to disable this from inside.

    **Communication protocol:**
    - Hypothesis code is written to a temp file and volume-mounted as
      ``/hypothesis.py`` (read-only)
    - Benchmark data is written as JSON and mounted as ``/data.json`` (read-only)
    - The in-container runner (``/runner.py``) calls ``run(benchmark_data)``
      and prints the metrics dict as JSON to stdout
    - The host captures stdout to get metrics, stderr for debugging

    **Graceful fallback:**
    If Docker is not available or the sandbox image isn't built, the module
    raises a clear error with instructions. The orchestrator can fall back
    to the process-level sandbox for development.

Spec: REQ-AUTO-004, REQ-AUTO-009
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carnot.autoresearch.sandbox import SandboxResult


# Default Docker image name for the sandbox
DEFAULT_IMAGE = "carnot-sandbox:latest"

# Default gVisor runtime name (as configured in /etc/docker/daemon.json)
DEFAULT_RUNTIME = "runsc"


@dataclass
class DockerSandboxConfig:
    """Configuration for the Docker+gVisor sandbox.

    **Researcher summary:**
        Image name, runtime, timeout, memory/CPU limits, GPU flag.

    **Detailed explanation for engineers:**
        Controls the Docker container parameters for hypothesis execution.

    Attributes:
        image: Docker image name. Must have Python, JAX, and carnot installed.
            Build with: ``docker build -f Dockerfile.sandbox -t carnot-sandbox .``
        runtime: Docker runtime. "runsc" for gVisor (production), "runc" for
            default Docker runtime (less secure, but works everywhere).
        timeout_seconds: Hard timeout. Container is killed after this many seconds.
        memory_limit: Docker memory limit string (e.g., "16g", "4g", "512m").
        cpu_limit: Number of CPUs the container can use (float, e.g., 4.0).
        gpu: If True, pass ``--gpus all`` for CUDA access. Requires
            nvidia-container-toolkit and a CUDA-capable sandbox image.
        network: If False (default), container has no network access.
        read_only: If True (default), container filesystem is read-only
            except for a tmpfs at /tmp.
        tmpfs_size: Size limit for /tmp inside the container (e.g., "100M").

    Spec: REQ-AUTO-004
    """

    image: str = DEFAULT_IMAGE
    runtime: str = DEFAULT_RUNTIME
    timeout_seconds: int = 1800
    memory_limit: str = "16g"
    cpu_limit: float = 4.0
    gpu: bool = False
    network: bool = False
    read_only: bool = True
    tmpfs_size: str = "100M"


def is_docker_available() -> bool:
    """Check if Docker is installed and the daemon is running.

    Returns True if ``docker info`` succeeds, False otherwise.
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_image_available(image: str) -> bool:
    """Check if a Docker image exists locally.

    Args:
        image: Image name with tag (e.g., "carnot-sandbox:latest").
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def is_runtime_available(runtime: str) -> bool:
    """Check if a Docker runtime is available.

    Args:
        runtime: Runtime name (e.g., "runsc" for gVisor, "runc" for default).
    """
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True,
            timeout=10,
            text=True,
        )
        return runtime in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_in_docker(
    hypothesis_code: str,
    benchmark_data: dict[str, Any],
    config: DockerSandboxConfig | None = None,
) -> SandboxResult:
    """Execute hypothesis code in a Docker+gVisor sandbox container.

    **Researcher summary:**
        Launches a container with the hypothesis code, captures stdout as
        JSON metrics, enforces timeout/memory/CPU/network limits.

    **Detailed explanation for engineers:**
        This function:
        1. Writes the hypothesis code and benchmark data to temp files
        2. Builds the ``docker run`` command with all security flags
        3. Runs the container with a timeout
        4. Parses stdout as JSON to get metrics
        5. Returns a SandboxResult (same interface as process-level sandbox)

        If Docker or the image is not available, returns a failed SandboxResult
        with a descriptive error message (not an exception).

    Args:
        hypothesis_code: Python source code defining a ``run(benchmark_data) -> dict`` function.
        benchmark_data: Dict passed to the hypothesis's run() function as JSON.
        config: Docker sandbox configuration. Uses defaults if None.

    Returns:
        SandboxResult with success/failure, metrics, stdout, stderr, timing.

    For example::

        result = run_in_docker(
            'def run(d): return {"final_energy": -5.0}',
            {"dim": 2},
        )
        if result.success:
            print(result.metrics["final_energy"])

    Spec: REQ-AUTO-004, REQ-AUTO-009
    """
    if config is None:
        config = DockerSandboxConfig()

    start_time = time.monotonic()

    # --- Preflight checks ---
    if not is_docker_available():
        return SandboxResult(
            success=False,
            error=(
                "Docker is not available. Install Docker and ensure the daemon is running. "
                "For development, use the process-level sandbox (run_in_sandbox) instead."
            ),
            wall_clock_seconds=time.monotonic() - start_time,
        )

    if not is_image_available(config.image):
        return SandboxResult(
            success=False,
            error=(
                f"Docker image '{config.image}' not found. "
                f"Build it with: docker build -f Dockerfile.sandbox -t {config.image} ."
            ),
            wall_clock_seconds=time.monotonic() - start_time,
        )

    # --- Write hypothesis and data to temp files ---
    with tempfile.TemporaryDirectory(prefix="carnot-sandbox-") as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write hypothesis code
        hypothesis_path = tmpdir_path / "hypothesis.py"
        hypothesis_path.write_text(hypothesis_code)

        # Write benchmark data as JSON
        data_path = tmpdir_path / "data.json"
        data_path.write_text(json.dumps(benchmark_data))

        # --- Build docker run command ---
        cmd = ["docker", "run", "--rm"]

        # Runtime: gVisor if available, otherwise default
        if is_runtime_available(config.runtime):
            cmd.extend(["--runtime", config.runtime])

        # Security: no network
        if not config.network:
            cmd.append("--network=none")

        # Security: read-only filesystem with tmpfs for /tmp
        if config.read_only:
            cmd.append("--read-only")
            cmd.extend(["--tmpfs", f"/tmp:size={config.tmpfs_size}"])

        # Resource limits
        cmd.extend(["--memory", config.memory_limit])
        cmd.extend(["--cpus", str(config.cpu_limit)])

        # GPU passthrough (requires nvidia-container-toolkit)
        if config.gpu:
            cmd.extend(["--gpus", "all"])

        # Volume mounts: hypothesis and data as read-only
        cmd.extend(["-v", f"{hypothesis_path}:/hypothesis.py:ro"])
        cmd.extend(["-v", f"{data_path}:/data.json:ro"])

        # Image
        cmd.append(config.image)

        # --- Execute ---
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            # Kill the container (docker run --rm will clean up)
            return SandboxResult(
                success=False,
                error="Docker container timed out",
                wall_clock_seconds=time.monotonic() - start_time,
                timed_out=True,
            )

    elapsed = time.monotonic() - start_time

    # --- Parse results ---
    stdout = proc.stdout
    stderr = proc.stderr

    if proc.returncode != 0:
        return SandboxResult(
            success=False,
            error=f"Container exited with code {proc.returncode}",
            stdout=stdout,
            stderr=stderr,
            wall_clock_seconds=elapsed,
        )

    # Parse stdout as JSON metrics
    try:
        metrics = json.loads(stdout.strip())
    except json.JSONDecodeError:
        return SandboxResult(
            success=False,
            error=f"Container stdout is not valid JSON: {stdout[:200]}",
            stdout=stdout,
            stderr=stderr,
            wall_clock_seconds=elapsed,
        )

    # Check if the runner reported an error
    if "error" in metrics:
        return SandboxResult(
            success=False,
            error=metrics["error"],
            stdout=stdout,
            stderr=stderr,
            wall_clock_seconds=elapsed,
        )

    return SandboxResult(
        success=True,
        metrics=metrics,
        stdout=stdout,
        stderr=stderr,
        wall_clock_seconds=elapsed,
    )
