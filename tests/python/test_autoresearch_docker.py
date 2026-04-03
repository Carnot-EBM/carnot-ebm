"""Tests for the Docker+gVisor sandbox backend.

**Testing strategy:**
    These tests cover the Docker sandbox logic without requiring Docker to be
    running. We test the preflight checks (docker available? image built?),
    command construction, and result parsing. Integration tests that actually
    launch containers are marked with @pytest.mark.docker and skipped when
    Docker is not available.

Spec coverage: REQ-AUTO-004, REQ-AUTO-009
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from carnot.autoresearch.sandbox import SandboxResult
from carnot.autoresearch.sandbox_docker import (
    DockerSandboxConfig,
    is_docker_available,
    is_image_available,
    is_runtime_available,
    run_in_docker,
)


class TestDockerSandboxConfig:
    """Tests for REQ-AUTO-004: Docker sandbox configuration."""

    def test_defaults(self) -> None:
        """REQ-AUTO-004: sensible defaults for production use."""
        config = DockerSandboxConfig()
        assert config.image == "carnot-sandbox:latest"
        assert config.runtime == "runsc"
        assert config.timeout_seconds == 1800
        assert config.memory_limit == "16g"
        assert config.cpu_limit == 4.0
        assert config.gpu is False
        assert config.network is False
        assert config.read_only is True

    def test_custom_config(self) -> None:
        """REQ-AUTO-004: custom configuration."""
        config = DockerSandboxConfig(
            image="my-sandbox:v2",
            runtime="runc",
            timeout_seconds=60,
            memory_limit="4g",
            gpu=True,
        )
        assert config.image == "my-sandbox:v2"
        assert config.gpu is True


class TestPreflightChecks:
    """Tests for preflight check functions."""

    @patch("carnot.autoresearch.sandbox_docker.shutil.which", return_value=None)
    def test_docker_not_installed(self, mock_which: MagicMock) -> None:
        """REQ-AUTO-004: detect Docker not installed."""
        assert is_docker_available() is False

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    @patch("carnot.autoresearch.sandbox_docker.shutil.which", return_value="/usr/bin/docker")
    def test_docker_daemon_not_running(self, mock_which: MagicMock, mock_run: MagicMock) -> None:
        """REQ-AUTO-004: detect Docker daemon not running."""
        mock_run.return_value = MagicMock(returncode=1)
        assert is_docker_available() is False

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    @patch("carnot.autoresearch.sandbox_docker.shutil.which", return_value="/usr/bin/docker")
    def test_docker_available(self, mock_which: MagicMock, mock_run: MagicMock) -> None:
        """REQ-AUTO-004: detect Docker available."""
        mock_run.return_value = MagicMock(returncode=0)
        assert is_docker_available() is True

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    def test_image_available(self, mock_run: MagicMock) -> None:
        """REQ-AUTO-004: detect image exists."""
        mock_run.return_value = MagicMock(returncode=0)
        assert is_image_available("carnot-sandbox:latest") is True

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    def test_image_not_available(self, mock_run: MagicMock) -> None:
        """REQ-AUTO-004: detect image missing."""
        mock_run.return_value = MagicMock(returncode=1)
        assert is_image_available("carnot-sandbox:latest") is False

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    def test_runtime_available(self, mock_run: MagicMock) -> None:
        """REQ-AUTO-004: detect gVisor runtime available."""
        mock_run.return_value = MagicMock(returncode=0, stdout="runc runsc")
        assert is_runtime_available("runsc") is True

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    def test_runtime_not_available(self, mock_run: MagicMock) -> None:
        """REQ-AUTO-004: detect gVisor runtime missing."""
        mock_run.return_value = MagicMock(returncode=0, stdout="runc")
        assert is_runtime_available("runsc") is False


class TestPreflightExceptions:
    """Tests for exception handling in preflight checks."""

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run", side_effect=FileNotFoundError)
    @patch("carnot.autoresearch.sandbox_docker.shutil.which", return_value="/usr/bin/docker")
    def test_docker_info_file_not_found(self, mock_which: MagicMock, mock_run: MagicMock) -> None:
        """REQ-AUTO-004: FileNotFoundError in docker info returns False."""
        assert is_docker_available() is False

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run", side_effect=FileNotFoundError)
    def test_image_inspect_file_not_found(self, mock_run: MagicMock) -> None:
        """REQ-AUTO-004: FileNotFoundError in docker image inspect returns False."""
        assert is_image_available("test:latest") is False

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run", side_effect=FileNotFoundError)
    def test_runtime_info_file_not_found(self, mock_run: MagicMock) -> None:
        """REQ-AUTO-004: FileNotFoundError in docker info --format returns False."""
        assert is_runtime_available("runsc") is False


class TestRunInDocker:
    """Tests for REQ-AUTO-004: Docker sandbox execution."""

    @patch("carnot.autoresearch.sandbox_docker.is_docker_available", return_value=False)
    def test_no_docker_returns_error(self, mock_avail: MagicMock) -> None:
        """REQ-AUTO-004: graceful failure when Docker not available."""
        result = run_in_docker("def run(d): return {}", {})
        assert not result.success
        assert "Docker is not available" in (result.error or "")

    @patch("carnot.autoresearch.sandbox_docker.is_image_available", return_value=False)
    @patch("carnot.autoresearch.sandbox_docker.is_docker_available", return_value=True)
    def test_no_image_returns_error(self, mock_avail: MagicMock, mock_img: MagicMock) -> None:
        """REQ-AUTO-004: graceful failure when image not built."""
        result = run_in_docker("def run(d): return {}", {})
        assert not result.success
        assert "not found" in (result.error or "")
        assert "docker build" in (result.error or "")

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    @patch("carnot.autoresearch.sandbox_docker.is_runtime_available", return_value=True)
    @patch("carnot.autoresearch.sandbox_docker.is_image_available", return_value=True)
    @patch("carnot.autoresearch.sandbox_docker.is_docker_available", return_value=True)
    def test_successful_execution(
        self,
        mock_avail: MagicMock,
        mock_img: MagicMock,
        mock_runtime: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """REQ-AUTO-004: successful hypothesis execution in Docker."""
        # Mock docker run returning JSON metrics
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"final_energy": -5.0, "_wall_clock_seconds": 1.2}),
            stderr="",
        )
        result = run_in_docker("def run(d): return {}", {"dim": 2})

        assert result.success
        assert result.metrics["final_energy"] == -5.0

        # Verify docker command was constructed correctly
        cmd = mock_run.call_args[0][0]
        assert "docker" in cmd
        assert "--network=none" in cmd
        assert "--read-only" in cmd
        assert "--runtime" in cmd
        assert "runsc" in cmd
        assert "--memory" in cmd

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    @patch("carnot.autoresearch.sandbox_docker.is_runtime_available", return_value=False)
    @patch("carnot.autoresearch.sandbox_docker.is_image_available", return_value=True)
    @patch("carnot.autoresearch.sandbox_docker.is_docker_available", return_value=True)
    def test_no_gvisor_falls_back(
        self,
        mock_avail: MagicMock,
        mock_img: MagicMock,
        mock_runtime: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """REQ-AUTO-004: works without gVisor (falls back to default runtime)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"final_energy": -5.0}),
            stderr="",
        )
        result = run_in_docker("def run(d): return {}", {})

        assert result.success
        # Command should NOT include --runtime since gVisor is unavailable
        cmd = mock_run.call_args[0][0]
        assert "--runtime" not in cmd

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    @patch("carnot.autoresearch.sandbox_docker.is_runtime_available", return_value=False)
    @patch("carnot.autoresearch.sandbox_docker.is_image_available", return_value=True)
    @patch("carnot.autoresearch.sandbox_docker.is_docker_available", return_value=True)
    def test_container_failure(
        self,
        mock_avail: MagicMock,
        mock_img: MagicMock,
        mock_runtime: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """REQ-AUTO-004: container exit code != 0 is reported."""
        mock_run.return_value = MagicMock(
            returncode=137,  # OOM killed
            stdout="",
            stderr="Killed",
        )
        result = run_in_docker("def run(d): return {}", {})

        assert not result.success
        assert "137" in (result.error or "")

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    @patch("carnot.autoresearch.sandbox_docker.is_runtime_available", return_value=False)
    @patch("carnot.autoresearch.sandbox_docker.is_image_available", return_value=True)
    @patch("carnot.autoresearch.sandbox_docker.is_docker_available", return_value=True)
    def test_timeout(
        self,
        mock_avail: MagicMock,
        mock_img: MagicMock,
        mock_runtime: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """REQ-AUTO-004: timeout is enforced."""
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="docker run", timeout=60)

        config = DockerSandboxConfig(timeout_seconds=60)
        result = run_in_docker("def run(d): return {}", {}, config=config)

        assert not result.success
        assert result.timed_out

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    @patch("carnot.autoresearch.sandbox_docker.is_runtime_available", return_value=False)
    @patch("carnot.autoresearch.sandbox_docker.is_image_available", return_value=True)
    @patch("carnot.autoresearch.sandbox_docker.is_docker_available", return_value=True)
    def test_invalid_json_stdout(
        self,
        mock_avail: MagicMock,
        mock_img: MagicMock,
        mock_runtime: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """REQ-AUTO-004: non-JSON stdout is handled gracefully."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not valid json",
            stderr="",
        )
        result = run_in_docker("def run(d): return {}", {})

        assert not result.success
        assert "not valid JSON" in (result.error or "")

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    @patch("carnot.autoresearch.sandbox_docker.is_runtime_available", return_value=False)
    @patch("carnot.autoresearch.sandbox_docker.is_image_available", return_value=True)
    @patch("carnot.autoresearch.sandbox_docker.is_docker_available", return_value=True)
    def test_runner_error_in_json(
        self,
        mock_avail: MagicMock,
        mock_img: MagicMock,
        mock_runtime: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """REQ-AUTO-004: runner-reported errors are captured."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"error": "run() raised ValueError: bad stuff"}),
            stderr="",
        )
        result = run_in_docker("def run(d): return {}", {})

        assert not result.success
        assert "ValueError" in (result.error or "")

    @patch("carnot.autoresearch.sandbox_docker.subprocess.run")
    @patch("carnot.autoresearch.sandbox_docker.is_runtime_available", return_value=False)
    @patch("carnot.autoresearch.sandbox_docker.is_image_available", return_value=True)
    @patch("carnot.autoresearch.sandbox_docker.is_docker_available", return_value=True)
    def test_gpu_flag(
        self,
        mock_avail: MagicMock,
        mock_img: MagicMock,
        mock_runtime: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """REQ-AUTO-004: GPU flag adds --gpus all."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"final_energy": -5.0}),
            stderr="",
        )
        config = DockerSandboxConfig(gpu=True)
        run_in_docker("def run(d): return {}", {}, config=config)

        cmd = mock_run.call_args[0][0]
        assert "--gpus" in cmd
        assert "all" in cmd
