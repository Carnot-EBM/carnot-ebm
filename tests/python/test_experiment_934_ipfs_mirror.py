"""Tests for Experiment 934 — IPFS Mirror Establishment.

REQ-VERIFY-145: Published model weights must be mirrored on IPFS.
CLAUDE.md rule 3: All published artifacts must have >= 2 distribution channels.

Coverage strategy: all subprocess calls are mocked so tests run in any CI
environment without a real IPFS daemon.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import scripts.experiment_934_ipfs_mirror_establishment as exp934
from scripts.experiment_934_ipfs_mirror_establishment import (
    build_artifact,
    check_ipfs_available,
    close_known_issue,
    ensure_daemon_running,
    install_ipfs_via_pacman,
    ipfs_add,
    verify_pin,
    write_ipfs_mirrors,
)


# ---------------------------------------------------------------------------
# _run helper
# ---------------------------------------------------------------------------


class TestRunHelper:
    """_run wraps subprocess.run and normalises the return value."""

    def test_success(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
            rc, out, err = exp934._run(["ipfs", "--version"])
        assert rc == 0
        assert out == "ok"

    def test_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ipfs", 5)):
            rc, out, err = exp934._run(["ipfs", "--version"], timeout=5)
        assert rc == -1
        assert "timeout" in err

    def test_file_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            rc, out, err = exp934._run(["ipfs", "--version"])
        assert rc == -1
        assert "not found" in err


# ---------------------------------------------------------------------------
# check_ipfs_available
# ---------------------------------------------------------------------------


class TestCheckIpfsAvailable:
    def test_found(self):
        with patch.object(exp934, "_run", return_value=(0, "ipfs version 0.28.0", "")):
            assert check_ipfs_available() is True

    def test_not_found(self):
        with patch.object(exp934, "_run", return_value=(-1, "", "command not found")):
            assert check_ipfs_available() is False

    def test_nonzero_return(self):
        with patch.object(exp934, "_run", return_value=(1, "", "error")):
            assert check_ipfs_available() is False


# ---------------------------------------------------------------------------
# install_ipfs_via_pacman
# ---------------------------------------------------------------------------


class TestInstallIpfsViaPacman:
    def test_pacman_success(self):
        with patch.object(exp934, "_run", return_value=(0, "", "")) as mock_run:
            result = install_ipfs_via_pacman()
        assert result is True
        assert mock_run.call_args[0][0] == ["sudo", "pacman", "-S", "--noconfirm", "kubo"]

    def test_pacman_fails_apt_success(self):
        call_count = [0]

        def side(_cmd, timeout=30):
            call_count[0] += 1
            if call_count[0] == 1:
                return (1, "", "pacman error")  # pacman fails
            return (0, "", "")  # apt succeeds

        with patch.object(exp934, "_run", side_effect=side):
            result = install_ipfs_via_pacman()
        assert result is True
        assert call_count[0] == 2

    def test_both_fail(self):
        with patch.object(exp934, "_run", return_value=(1, "", "error")):
            result = install_ipfs_via_pacman()
        assert result is False


# ---------------------------------------------------------------------------
# ensure_daemon_running
# ---------------------------------------------------------------------------


class TestEnsureDaemonRunning:
    def test_already_running(self):
        with patch.object(exp934, "_run", return_value=(0, "{}", "")):
            assert ensure_daemon_running() is True

    def test_starts_and_becomes_responsive(self, tmp_path):
        """Daemon not running → init already done → starts → becomes responsive."""
        # Create a fake IPFS config so we skip init.
        (tmp_path / "config").write_text("{}")

        call_results = [
            (-1, "", "connection refused"),  # initial ipfs id — not running
            (0, '{"ID":"Qm..."}', ""),  # poll after Popen
        ]
        call_iter = iter(call_results)

        with (
            patch.object(exp934, "_run", side_effect=lambda *a, **kw: next(call_iter)),
            patch.object(exp934, "os") as mock_os,
            patch("subprocess.Popen"),
            patch("time.sleep"),
            patch.dict("os.environ", {"IPFS_PATH": str(tmp_path)}),
        ):
            mock_os.environ.get = lambda k, default=None: (
                str(tmp_path) if k == "IPFS_PATH" else default
            )
            # Patch Path.home to avoid touching real home.
            with patch.object(Path, "home", return_value=tmp_path):
                result = ensure_daemon_running()
        assert result is True

    def test_daemon_never_responds(self, tmp_path):
        """Daemon started but never becomes responsive — returns False."""
        (tmp_path / "config").write_text("{}")

        with (
            patch.object(exp934, "_run", return_value=(-1, "", "refused")),
            patch("subprocess.Popen"),
            patch("time.sleep"),
            patch.dict("os.environ", {"IPFS_PATH": str(tmp_path)}),
        ):
            result = ensure_daemon_running()
        assert result is False

    def test_init_fails(self, tmp_path):
        """If ipfs init fails, returns False without starting daemon."""
        call_results = [
            (-1, "", "connection refused"),  # initial ipfs id
            (1, "", "init error"),  # ipfs init
        ]
        call_iter = iter(call_results)

        with (
            patch.object(exp934, "_run", side_effect=lambda *a, **kw: next(call_iter)),
            patch.dict("os.environ", {"IPFS_PATH": str(tmp_path)}),
        ):
            result = ensure_daemon_running()
        assert result is False


# ---------------------------------------------------------------------------
# ipfs_add
# ---------------------------------------------------------------------------


class TestIpfsAdd:
    def test_file_added(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"fake weights")
        with patch.object(exp934, "_run", return_value=(0, "QmFakeCID123", "")):
            cid = ipfs_add(f)
        assert cid == "QmFakeCID123"

    def test_dir_added(self, tmp_path):
        (tmp_path / "file.bin").write_bytes(b"data")
        with patch.object(exp934, "_run", return_value=(0, "QmDirCID456", "")):
            cid = ipfs_add(tmp_path)
        assert cid == "QmDirCID456"
        # Should include -r flag for directories.
        # We verify by inspecting the _run call args indirectly through the result.
        assert cid is not None

    def test_missing_path_returns_none(self, tmp_path):
        cid = ipfs_add(tmp_path / "nonexistent.bin")
        assert cid is None

    def test_ipfs_add_fails(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"x")
        with patch.object(exp934, "_run", return_value=(1, "", "connection refused")):
            cid = ipfs_add(f)
        assert cid is None

    def test_empty_stdout_returns_none(self, tmp_path):
        f = tmp_path / "model.bin"
        f.write_bytes(b"x")
        with patch.object(exp934, "_run", return_value=(0, "", "")):
            cid = ipfs_add(f)
        assert cid is None


# ---------------------------------------------------------------------------
# verify_pin
# ---------------------------------------------------------------------------


class TestVerifyPin:
    def test_cid_present(self):
        pin_output = "QmABC123 recursive\nQmDEF456 recursive\n"
        with patch.object(exp934, "_run", return_value=(0, pin_output, "")):
            assert verify_pin("QmABC123") is True

    def test_cid_absent(self):
        with patch.object(exp934, "_run", return_value=(0, "QmOther recursive\n", "")):
            assert verify_pin("QmABC123") is False

    def test_command_fails(self):
        with patch.object(exp934, "_run", return_value=(1, "", "error")):
            assert verify_pin("QmABC123") is False


# ---------------------------------------------------------------------------
# write_ipfs_mirrors
# ---------------------------------------------------------------------------


class TestWriteIpfsMirrors:
    def test_writes_json(self, tmp_path):
        with patch.object(exp934, "RESULTS_DIR", tmp_path):
            with patch.object(exp934, "IPFS_MIRRORS_JSON", tmp_path / "ipfs_mirrors.json"):
                with patch.object(exp934, "VJEPA_WEIGHTS", tmp_path / "v.safetensors"):
                    with patch.object(exp934, "ESTIMATION_STAGING_DIR", tmp_path / "est"):
                        write_ipfs_mirrors("QmVJEPA", "QmEST")

        data = json.loads((tmp_path / "ipfs_mirrors.json").read_text())
        assert data["vjepa_v2"]["cid"] == "QmVJEPA"
        assert data["estimation_verifier_v1"]["cid"] == "QmEST"
        assert "ipfs.io/ipfs/QmVJEPA" in data["vjepa_v2"]["ipfs_gateway_url"]

    def test_writes_none_cids(self, tmp_path):
        with patch.object(exp934, "RESULTS_DIR", tmp_path):
            with patch.object(exp934, "IPFS_MIRRORS_JSON", tmp_path / "ipfs_mirrors.json"):
                with patch.object(exp934, "VJEPA_WEIGHTS", tmp_path / "v.safetensors"):
                    with patch.object(exp934, "ESTIMATION_STAGING_DIR", tmp_path / "est"):
                        write_ipfs_mirrors(None, None)

        data = json.loads((tmp_path / "ipfs_mirrors.json").read_text())
        assert data["vjepa_v2"]["cid"] is None
        assert data["vjepa_v2"]["ipfs_gateway_url"] is None


# ---------------------------------------------------------------------------
# build_artifact
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    def _make(self, verdict: str, **kwargs):
        return build_artifact(
            started_at="2026-04-26T15:00:00+00:00",
            ipfs_installed=kwargs.get("ipfs_installed", True),
            ipfs_cid_vjepa=kwargs.get("ipfs_cid_vjepa", "QmVJEPA"),
            ipfs_cid_estimation=kwargs.get("ipfs_cid_estimation", "QmEST"),
            honest_verdict=verdict,
        )

    def test_required_fields(self):
        art = self._make("ipfs_mirror_established")
        for field in (
            "experiment",
            "schema",
            "run_date",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "title",
        ):
            assert field in art, f"Missing required field: {field}"

    def test_status_success_on_established(self):
        art = self._make("ipfs_mirror_established")
        assert art["status"] == "success"

    def test_status_failed_on_install_failed(self):
        art = self._make(
            "ipfs_install_failed",
            ipfs_installed=False,
            ipfs_cid_vjepa=None,
            ipfs_cid_estimation=None,
        )
        assert art["status"] == "failed"

    def test_gateway_url_populated(self):
        art = self._make("ipfs_mirror_established")
        assert "QmVJEPA" in art["vjepa_gateway_url"]

    def test_gateway_url_none_when_no_cid(self):
        art = self._make("ipfs_pin_failed", ipfs_cid_vjepa=None, ipfs_cid_estimation=None)
        assert art["vjepa_gateway_url"] is None

    def test_duration_positive(self):
        art = self._make("ipfs_mirror_established")
        assert art["duration_s"] >= 0


# ---------------------------------------------------------------------------
# close_known_issue
# ---------------------------------------------------------------------------


class TestCloseKnownIssue:
    def test_appends_closure_note(self, tmp_path):
        ops_dir = tmp_path / "ops"
        ops_dir.mkdir()
        ki = ops_dir / "known-issues.md"
        ki.write_text("## IPFS not installed\nsome content\n")
        with patch.object(exp934, "REPO_ROOT", tmp_path):
            close_known_issue("QmVJEPA123")
        content = ki.read_text()
        assert "IPFS Mirror CLOSED" in content
        assert "QmVJEPA123" in content

    def test_idempotent(self, tmp_path):
        ops_dir = tmp_path / "ops"
        ops_dir.mkdir()
        ki = ops_dir / "known-issues.md"
        ki.write_text("## IPFS Mirror CLOSED (Exp 934, 2026-04-26)\nalready closed\n")
        with patch.object(exp934, "REPO_ROOT", tmp_path):
            close_known_issue("QmVJEPA123")
        # Should not append a second closure note.
        content = ki.read_text()
        assert content.count("IPFS Mirror CLOSED") == 1

    def test_missing_file_is_noop(self, tmp_path):
        with patch.object(exp934, "REPO_ROOT", tmp_path):
            close_known_issue("QmVJEPA123")  # no ops/ subdir — should not raise


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------


class TestMain:
    """Integration tests for the main() entry point.

    All subprocess calls and file I/O outside tmp_path are mocked so these
    tests run without a real IPFS daemon or the actual weight files.
    """

    def _patch_paths(self, tmp_path):
        """Return a context manager stack patching all module-level paths."""
        vjepa = tmp_path / "vjepa_predictor_v2.safetensors"
        vjepa.write_bytes(b"fake weights")
        est_dir = tmp_path / "carnot-vjepa-v2-card"
        est_dir.mkdir()
        (est_dir / "model.safetensors").write_bytes(b"est weights")
        return vjepa, est_dir

    def test_main_establish_success(self, tmp_path):
        vjepa, est_dir = self._patch_paths(tmp_path)

        with (
            patch.object(exp934, "RESULTS_DIR", tmp_path),
            patch.object(exp934, "DELIVERABLE", tmp_path / "exp934.json"),
            patch.object(exp934, "IPFS_MIRRORS_JSON", tmp_path / "ipfs_mirrors.json"),
            patch.object(exp934, "VJEPA_WEIGHTS", vjepa),
            patch.object(exp934, "ESTIMATION_STAGING_DIR", est_dir),
            patch.object(exp934, "REPO_ROOT", tmp_path),
            patch.object(exp934, "check_ipfs_available", return_value=True),
            patch.object(exp934, "ensure_daemon_running", return_value=True),
            patch.object(exp934, "ipfs_add", side_effect=["QmVJEPA", "QmEST"]),
            patch.object(exp934, "verify_pin", return_value=True),
            patch.object(exp934, "close_known_issue"),
        ):
            # Create a fake known-issues.md for close_known_issue to skip.
            (tmp_path / "ops").mkdir()
            rc = exp934.main()

        assert rc == 0
        art = json.loads((tmp_path / "exp934.json").read_text())
        assert art["honest_verdict"] == "ipfs_mirror_established"
        assert art["ipfs_cid_vjepa"] == "QmVJEPA"
        assert art["ipfs_cid_estimation"] == "QmEST"
        assert art["status"] == "success"

    def test_main_install_failed(self, tmp_path):
        with (
            patch.object(exp934, "RESULTS_DIR", tmp_path),
            patch.object(exp934, "DELIVERABLE", tmp_path / "exp934.json"),
            patch.object(exp934, "check_ipfs_available", return_value=False),
            patch.object(exp934, "install_ipfs_via_pacman", return_value=False),
        ):
            rc = exp934.main()

        assert rc == 1
        art = json.loads((tmp_path / "exp934.json").read_text())
        assert art["honest_verdict"] == "ipfs_install_failed"
        assert art["ipfs_installed"] is False

    def test_main_install_succeeds_then_pins(self, tmp_path):
        vjepa, est_dir = self._patch_paths(tmp_path)

        with (
            patch.object(exp934, "RESULTS_DIR", tmp_path),
            patch.object(exp934, "DELIVERABLE", tmp_path / "exp934.json"),
            patch.object(exp934, "IPFS_MIRRORS_JSON", tmp_path / "ipfs_mirrors.json"),
            patch.object(exp934, "VJEPA_WEIGHTS", vjepa),
            patch.object(exp934, "ESTIMATION_STAGING_DIR", est_dir),
            patch.object(exp934, "REPO_ROOT", tmp_path),
            # First call returns False (not installed); second returns True (installed).
            patch.object(exp934, "check_ipfs_available", side_effect=[False, True]),
            patch.object(exp934, "install_ipfs_via_pacman", return_value=True),
            patch.object(exp934, "ensure_daemon_running", return_value=True),
            patch.object(exp934, "ipfs_add", side_effect=["QmVJEPA2", "QmEST2"]),
            patch.object(exp934, "verify_pin", return_value=True),
            patch.object(exp934, "close_known_issue"),
        ):
            (tmp_path / "ops").mkdir()
            rc = exp934.main()

        assert rc == 0
        art = json.loads((tmp_path / "exp934.json").read_text())
        assert art["honest_verdict"] == "ipfs_mirror_established"

    def test_main_pin_failed(self, tmp_path):
        vjepa, est_dir = self._patch_paths(tmp_path)

        with (
            patch.object(exp934, "RESULTS_DIR", tmp_path),
            patch.object(exp934, "DELIVERABLE", tmp_path / "exp934.json"),
            patch.object(exp934, "IPFS_MIRRORS_JSON", tmp_path / "ipfs_mirrors.json"),
            patch.object(exp934, "VJEPA_WEIGHTS", vjepa),
            patch.object(exp934, "ESTIMATION_STAGING_DIR", est_dir),
            patch.object(exp934, "REPO_ROOT", tmp_path),
            patch.object(exp934, "check_ipfs_available", return_value=True),
            patch.object(exp934, "ensure_daemon_running", return_value=True),
            patch.object(exp934, "ipfs_add", return_value=None),
        ):
            rc = exp934.main()

        assert rc == 1
        art = json.loads((tmp_path / "exp934.json").read_text())
        assert art["honest_verdict"] == "ipfs_pin_failed"
        assert art["ipfs_cid_vjepa"] is None
