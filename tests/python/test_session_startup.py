"""Tests for carnot.pipeline.session_startup.

Spec coverage: REQ-INFRA-008,
               SCENARIO-INFRA-012, SCENARIO-INFRA-013

Written test-first per REQ-INFRA-002.  Tests validate:
- parse_session_startup_output: parses canonical summary line into dict.
- parse_session_startup_output: returns safe defaults when line absent.
- run_session_startup(dry_run=True): calls script with --dry-run; never kills.
- run_session_startup(dry_run=False): calls script with --kill-zombies.
- run_session_startup: handles FileNotFoundError (script absent or nvidia-smi absent).
- run_session_startup: handles TimeoutExpired.
- all_healthy logic: True iff n_gpus_detected >= 2 AND n_zombies_found == 0.
- CI safety: no exceptions raised in any code path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.session_startup import (
    _SCRIPT_PATH,
    _SUMMARY_RE,
    parse_session_startup_output,
    run_session_startup,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent


def _summary_line(n_gpus: int, zombies: int, killed: int, all_healthy: bool) -> str:
    """Build a canonical SESSION STARTUP summary line for tests."""
    return (
        f"SESSION STARTUP: n_gpus={n_gpus} zombies={zombies} "
        f"killed={killed} all_healthy={all_healthy}"
    )


# ---------------------------------------------------------------------------
# TestParseSessionStartupOutput
# REQ-INFRA-008 / SCENARIO-INFRA-012
# ---------------------------------------------------------------------------


class TestParseSessionStartupOutput:
    """parse_session_startup_output: correct parsing of script stdout."""

    def test_parses_healthy_two_gpu_no_zombies(self) -> None:
        """SCENARIO-INFRA-012: 2 GPUs, 0 zombies → all_healthy=True."""
        output = _summary_line(2, 0, 0, True)
        result = parse_session_startup_output(output)
        assert result["n_gpus_detected"] == 2
        assert result["n_zombies_found"] == 0
        assert result["n_zombies_killed"] == 0
        assert result["all_healthy"] is True

    def test_parses_one_gpu_no_zombies_unhealthy(self) -> None:
        """Only 1 GPU → all_healthy=False (needs >= 2)."""
        output = _summary_line(1, 0, 0, False)
        result = parse_session_startup_output(output)
        assert result["n_gpus_detected"] == 1
        assert result["all_healthy"] is False

    def test_parses_two_gpus_with_zombie_unhealthy(self) -> None:
        """2 GPUs but 1 zombie → all_healthy=False."""
        output = _summary_line(2, 1, 0, False)
        result = parse_session_startup_output(output)
        assert result["n_gpus_detected"] == 2
        assert result["n_zombies_found"] == 1
        assert result["all_healthy"] is False

    def test_parses_killed_count(self) -> None:
        """Killed count is captured from the summary line."""
        output = _summary_line(2, 2, 2, True)
        result = parse_session_startup_output(output)
        assert result["n_zombies_killed"] == 2

    def test_all_healthy_recomputed_from_values(self) -> None:
        """all_healthy is recomputed by Python, ignoring the string literal."""
        # Script says all_healthy=True but n_gpus=1 → Python overrides to False
        output = "SESSION STARTUP: n_gpus=1 zombies=0 killed=0 all_healthy=True"
        result = parse_session_startup_output(output)
        assert result["all_healthy"] is False

    def test_all_healthy_two_gpus_zero_zombies(self) -> None:
        """all_healthy=True exactly when n_gpus>=2 and zombies==0."""
        output = "SESSION STARTUP: n_gpus=2 zombies=0 killed=0 all_healthy=False"
        result = parse_session_startup_output(output)
        assert result["all_healthy"] is True

    def test_summary_embedded_in_longer_output(self) -> None:
        """SCENARIO-INFRA-012: parser tolerates surrounding noise lines."""
        output = (
            "Checking nvidia-smi...\n"
            "Found 2 GPUs\n"
            + _summary_line(2, 0, 0, True)
            + "\nDone.\n"
        )
        result = parse_session_startup_output(output)
        assert result["n_gpus_detected"] == 2

    def test_missing_summary_line_returns_defaults(self) -> None:
        """SCENARIO-INFRA-013: no summary line → n_gpus=0, all_healthy=False."""
        result = parse_session_startup_output("nvidia-smi not found\n")
        assert result["n_gpus_detected"] == 0
        assert result["n_zombies_found"] == 0
        assert result["n_zombies_killed"] == 0
        assert result["all_healthy"] is False

    def test_empty_output_returns_defaults(self) -> None:
        """SCENARIO-INFRA-013: empty output → safe defaults, no exception."""
        result = parse_session_startup_output("")
        assert result["n_gpus_detected"] == 0
        assert result["all_healthy"] is False

    def test_return_type_is_dict(self) -> None:
        """Return value is always a dict."""
        result = parse_session_startup_output(_summary_line(2, 0, 0, True))
        assert isinstance(result, dict)

    def test_all_required_keys_present(self) -> None:
        """Dict always contains all four required keys."""
        result = parse_session_startup_output("")
        for key in ("n_gpus_detected", "n_zombies_found", "n_zombies_killed", "all_healthy"):
            assert key in result, f"Missing key: {key}"

    def test_zero_gpus_is_unhealthy(self) -> None:
        """0 GPUs → all_healthy=False."""
        output = _summary_line(0, 0, 0, False)
        result = parse_session_startup_output(output)
        assert result["n_gpus_detected"] == 0
        assert result["all_healthy"] is False

    def test_three_gpus_no_zombies_is_healthy(self) -> None:
        """3 GPUs, 0 zombies → all_healthy=True (>= 2 is sufficient)."""
        output = _summary_line(3, 0, 0, True)
        result = parse_session_startup_output(output)
        assert result["all_healthy"] is True

    def test_values_are_int_not_string(self) -> None:
        """Numeric values must be int, not str."""
        result = parse_session_startup_output(_summary_line(2, 1, 1, False))
        assert isinstance(result["n_gpus_detected"], int)
        assert isinstance(result["n_zombies_found"], int)
        assert isinstance(result["n_zombies_killed"], int)

    def test_all_healthy_is_bool(self) -> None:
        """all_healthy must be a Python bool, not a string."""
        result = parse_session_startup_output(_summary_line(2, 0, 0, True))
        assert isinstance(result["all_healthy"], bool)


# ---------------------------------------------------------------------------
# TestRunSessionStartup
# REQ-INFRA-008 / SCENARIO-INFRA-012 / SCENARIO-INFRA-013
# ---------------------------------------------------------------------------


class TestRunSessionStartupDryRun:
    """run_session_startup(dry_run=True): calls script with --dry-run; never kills."""

    def _make_completed_process(self, stdout: str = "", stderr: str = "") -> MagicMock:
        proc = MagicMock()
        proc.stdout = stdout
        proc.stderr = stderr
        proc.returncode = 0
        return proc

    def test_calls_script_with_dry_run_flag(self) -> None:
        """SCENARIO-INFRA-012: --dry-run is passed when dry_run=True."""
        summary = _summary_line(2, 0, 0, True)
        fake_proc = self._make_completed_process(stdout=summary)

        with patch("subprocess.run", return_value=fake_proc) as mock_run:
            run_session_startup(dry_run=True)

        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert "--dry-run" in cmd
        assert "--kill-zombies" not in cmd

    def test_calls_script_with_kill_zombies_when_not_dry_run(self) -> None:
        """When dry_run=False, --kill-zombies is passed instead."""
        summary = _summary_line(2, 0, 0, True)
        fake_proc = self._make_completed_process(stdout=summary)

        with patch("subprocess.run", return_value=fake_proc) as mock_run:
            run_session_startup(dry_run=False)

        args, _ = mock_run.call_args
        cmd = args[0]
        assert "--kill-zombies" in cmd
        assert "--dry-run" not in cmd

    def test_returns_parsed_dict(self) -> None:
        """SCENARIO-INFRA-012: parsed dict is returned from run_session_startup."""
        summary = _summary_line(2, 0, 0, True)
        fake_proc = self._make_completed_process(stdout=summary)

        with patch("subprocess.run", return_value=fake_proc):
            result = run_session_startup(dry_run=True)

        assert result["n_gpus_detected"] == 2
        assert result["all_healthy"] is True

    def test_script_path_contains_session_startup(self) -> None:
        """The script invoked is scripts/session_startup.sh."""
        summary = _summary_line(2, 0, 0, True)
        fake_proc = self._make_completed_process(stdout=summary)

        with patch("subprocess.run", return_value=fake_proc) as mock_run:
            run_session_startup(dry_run=True)

        args, _ = mock_run.call_args
        cmd = args[0]
        assert "session_startup" in cmd[0]

    def test_dry_run_never_kills_n_zombies_killed_zero(self) -> None:
        """SCENARIO-INFRA-012: in dry-run mode, n_zombies_killed is always 0."""
        # Even if zombies are found, dry-run should report killed=0
        summary = _summary_line(2, 2, 0, False)
        fake_proc = self._make_completed_process(stdout=summary)

        with patch("subprocess.run", return_value=fake_proc):
            result = run_session_startup(dry_run=True)

        assert result["n_zombies_killed"] == 0

    def test_combines_stdout_and_stderr_for_parsing(self) -> None:
        """Summary line in stderr is also found (stderr appended to stdout)."""
        summary = _summary_line(2, 0, 0, True)
        fake_proc = self._make_completed_process(stdout="", stderr=summary)

        with patch("subprocess.run", return_value=fake_proc):
            result = run_session_startup(dry_run=True)

        assert result["n_gpus_detected"] == 2


class TestRunSessionStartupCISafe:
    """run_session_startup: degrades gracefully in CI / no-GPU environments."""

    def test_file_not_found_returns_unhealthy(self) -> None:
        """SCENARIO-INFRA-013: script absent → n_gpus=0, all_healthy=False, no exception."""
        with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
            result = run_session_startup(dry_run=True)

        assert result["n_gpus_detected"] == 0
        assert result["all_healthy"] is False

    def test_file_not_found_does_not_raise(self) -> None:
        """SCENARIO-INFRA-013: FileNotFoundError is swallowed."""
        with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
            try:
                run_session_startup(dry_run=True)
            except Exception as exc:
                pytest.fail(f"run_session_startup raised unexpectedly: {exc}")

    def test_timeout_returns_unhealthy(self) -> None:
        """TimeoutExpired → n_gpus=0, all_healthy=False, no exception."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            result = run_session_startup(dry_run=True)

        assert result["n_gpus_detected"] == 0
        assert result["all_healthy"] is False

    def test_timeout_does_not_raise(self) -> None:
        """TimeoutExpired is swallowed."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            try:
                run_session_startup(dry_run=True)
            except Exception as exc:
                pytest.fail(f"run_session_startup raised on timeout: {exc}")

    def test_no_summary_line_in_output_returns_unhealthy(self) -> None:
        """SCENARIO-INFRA-013: nvidia-smi absent → script prints fallback, no summary line."""
        fake_proc = MagicMock()
        fake_proc.stdout = "nvidia-smi not found\n"
        fake_proc.stderr = ""
        fake_proc.returncode = 0

        with patch("subprocess.run", return_value=fake_proc):
            result = run_session_startup(dry_run=True)

        assert result["n_gpus_detected"] == 0
        assert result["all_healthy"] is False

    def test_return_value_is_always_dict(self) -> None:
        """Return type is dict regardless of error path."""
        with patch("subprocess.run", side_effect=FileNotFoundError("x")):
            result = run_session_startup(dry_run=True)
        assert isinstance(result, dict)

    def test_all_required_keys_present_on_error(self) -> None:
        """All four keys present even when script errors out."""
        with patch("subprocess.run", side_effect=FileNotFoundError("x")):
            result = run_session_startup(dry_run=True)
        for key in ("n_gpus_detected", "n_zombies_found", "n_zombies_killed", "all_healthy"):
            assert key in result


# ---------------------------------------------------------------------------
# TestAllHealthyRule
# REQ-INFRA-008
# ---------------------------------------------------------------------------


class TestAllHealthyRule:
    """all_healthy is True iff n_gpus_detected >= 2 AND n_zombies_found == 0."""

    @pytest.mark.parametrize(
        "n_gpus,zombies,expected",
        [
            (2, 0, True),   # exactly 2 GPUs, no zombies → healthy
            (3, 0, True),   # more than 2 GPUs, no zombies → healthy
            (1, 0, False),  # only 1 GPU → unhealthy
            (0, 0, False),  # no GPUs → unhealthy
            (2, 1, False),  # 2 GPUs but zombie → unhealthy
            (2, 2, False),  # 2 GPUs but 2 zombies → unhealthy
        ],
    )
    def test_all_healthy_combinations(
        self, n_gpus: int, zombies: int, expected: bool
    ) -> None:
        """Parametrized: all_healthy follows the n_gpus>=2 AND zombies==0 rule."""
        output = _summary_line(n_gpus, zombies, 0, expected)
        result = parse_session_startup_output(output)
        assert result["all_healthy"] is expected


# ---------------------------------------------------------------------------
# TestModuleConstants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Module-level constants are correctly set."""

    def test_script_path_is_path_object(self) -> None:
        """_SCRIPT_PATH is a pathlib.Path."""
        assert isinstance(_SCRIPT_PATH, Path)

    def test_script_path_ends_with_session_startup_sh(self) -> None:
        """_SCRIPT_PATH points to scripts/session_startup.sh."""
        assert _SCRIPT_PATH.name == "session_startup.sh"

    def test_summary_re_is_compiled(self) -> None:
        """_SUMMARY_RE is a compiled regex pattern."""
        import re

        assert hasattr(_SUMMARY_RE, "search")

    def test_summary_re_matches_canonical_line(self) -> None:
        """_SUMMARY_RE matches the exact output format of session_startup.sh."""
        line = _summary_line(2, 0, 0, True)
        match = _SUMMARY_RE.search(line)
        assert match is not None
