"""Unit tests for EnvPropagationGuard.propagate() and write_state_file().

Spec: REQ-INFRA-080, REQ-INFRA-081,
      SCENARIO-INFRA-090, SCENARIO-INFRA-091
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from scripts.experiment_template import EnvPropagationGuard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_guard(tmp_path: Path) -> type[EnvPropagationGuard]:
    """Return a subclass of EnvPropagationGuard that uses tmp_path for both files.

    Why a subclass rather than monkeypatching: the class attributes are class-level
    Path objects, not instance attributes, so patching them directly would mutate
    the real class across test isolation boundaries.  A subclass overrides them
    cleanly and leaves the real class untouched.
    """

    class _Guard(EnvPropagationGuard):
        STATE_FILE: Path = tmp_path / ".carnot" / "conductor_state.sh"
        _path: Path = tmp_path / ".carnot_session_env"

    return _Guard


# ---------------------------------------------------------------------------
# write_state_file tests  (REQ-INFRA-081)
# ---------------------------------------------------------------------------


def test_write_state_file_creates_dir_and_file(tmp_path: Path) -> None:
    """write_state_file() creates ~/.carnot/ and the state file if absent."""
    Guard = _make_guard(tmp_path)
    with mock.patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}, clear=False):
        Guard.write_state_file()
    assert Guard.STATE_FILE.exists()


def test_write_state_file_contains_shebang(tmp_path: Path) -> None:
    """The state file must begin with #!/bin/sh (REQ-INFRA-081-3)."""
    Guard = _make_guard(tmp_path)
    with mock.patch.dict(os.environ, {}, clear=False):
        Guard.write_state_file()
    content = Guard.STATE_FILE.read_text()
    assert content.startswith("#!/bin/sh")


def test_write_state_file_always_includes_force_live(tmp_path: Path) -> None:
    """CARNOT_FORCE_LIVE=1 is always written even if absent from env (REQ-INFRA-081-2)."""
    Guard = _make_guard(tmp_path)
    env_without_live = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
    with mock.patch.dict(os.environ, env_without_live, clear=True):
        Guard.write_state_file()
    content = Guard.STATE_FILE.read_text()
    assert "export CARNOT_FORCE_LIVE=1" in content


def test_write_state_file_includes_other_carnot_vars(tmp_path: Path) -> None:
    """Additional CARNOT_* vars present in os.environ are also written."""
    Guard = _make_guard(tmp_path)
    with mock.patch.dict(os.environ, {"CARNOT_N_SPINS": "256"}, clear=False):
        Guard.write_state_file()
    content = Guard.STATE_FILE.read_text()
    assert "export CARNOT_N_SPINS=256" in content


# ---------------------------------------------------------------------------
# propagate() tests  (REQ-INFRA-080)
# ---------------------------------------------------------------------------


def test_propagate_sets_force_live_unconditionally(tmp_path: Path) -> None:
    """propagate() sets CARNOT_FORCE_LIVE=1 even when absent (REQ-INFRA-080-2)."""
    Guard = _make_guard(tmp_path)
    env_without_live = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
    with mock.patch.dict(os.environ, env_without_live, clear=True):
        result = Guard.propagate()
    assert os.environ.get("CARNOT_FORCE_LIVE") == "1"
    assert result.get("CARNOT_FORCE_LIVE") == "1"


def test_propagate_sources_state_file(tmp_path: Path) -> None:
    """propagate() reads export KEY=VALUE lines from STATE_FILE (REQ-INFRA-080-1)."""
    Guard = _make_guard(tmp_path)
    Guard.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    Guard.STATE_FILE.write_text("#!/bin/sh\nexport CARNOT_N_SPINS=128\n")

    env_without_spins = {k: v for k, v in os.environ.items() if k != "CARNOT_N_SPINS"}
    with mock.patch.dict(os.environ, env_without_spins, clear=True):
        result = Guard.propagate()
        # Assert inside the mock context — os.environ is restored on exit
        assert os.environ.get("CARNOT_N_SPINS") == "128"

    assert result.get("CARNOT_N_SPINS") == "128"


def test_propagate_returns_dict_with_carnot_prefix(tmp_path: Path) -> None:
    """propagate() returns a dict containing all CARNOT_* vars (REQ-INFRA-080-3)."""
    Guard = _make_guard(tmp_path)
    with mock.patch.dict(os.environ, {"CARNOT_FOO": "bar"}, clear=False):
        result = Guard.propagate()
    assert "CARNOT_FOO" in result
    assert "CARNOT_FORCE_LIVE" in result


def test_propagate_explicit_env_wins_over_state_file(tmp_path: Path) -> None:
    """An explicit os.environ value is not overwritten by the STATE_FILE."""
    Guard = _make_guard(tmp_path)
    Guard.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    Guard.STATE_FILE.write_text("export CARNOT_N_SPINS=99\n")

    with mock.patch.dict(os.environ, {"CARNOT_N_SPINS": "512"}, clear=False):
        Guard.propagate()
        # Assert inside mock context — shell export must not be overwritten by state file
        assert os.environ.get("CARNOT_N_SPINS") == "512"


def test_propagate_no_state_file_still_works(tmp_path: Path) -> None:
    """propagate() succeeds and sets CARNOT_FORCE_LIVE=1 when no STATE_FILE exists."""
    Guard = _make_guard(tmp_path)
    assert not Guard.STATE_FILE.exists()
    env_clean = {k: v for k, v in os.environ.items() if not k.startswith("CARNOT_")}
    with mock.patch.dict(os.environ, env_clean, clear=True):
        result = Guard.propagate()
    assert os.environ.get("CARNOT_FORCE_LIVE") == "1"
    assert "CARNOT_FORCE_LIVE" in result


def test_propagate_rocm_and_hsa_vars_included(tmp_path: Path) -> None:
    """ROCM_* and HSA_* vars are included in the returned dict (REQ-INFRA-080-3)."""
    Guard = _make_guard(tmp_path)
    with mock.patch.dict(os.environ, {"ROCM_HOME": "/opt/rocm", "HSA_OVERRIDE": "1"}, clear=False):
        result = Guard.propagate()
    assert result.get("ROCM_HOME") == "/opt/rocm"
    assert result.get("HSA_OVERRIDE") == "1"
