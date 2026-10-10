"""SCENARIO-HARNESS-5930-DISK-SCRATCH: quota-limited tmpfs must not break fixtures."""

import errno
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.testing import pytest_basetemp_isolation as isolation


def clear_temp_overrides(monkeypatch):
    for key in ["TMPDIR", "TEMP", "TMP"]:
        monkeypatch.delenv(key, raising=False)


def test_default_scratch_avoids_quota_limited_tmp(monkeypatch):
    """REQ-HARNESS-5930: default selection never asks a quota-limited tmpfs for storage."""
    clear_temp_overrides(monkeypatch)
    monkeypatch.setattr(Path, "is_dir", lambda path: path == Path("/var/tmp"))
    monkeypatch.setattr(isolation.os, "access", lambda path, mode: True)

    def quota_limited_tmp():
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(isolation, "gettempdir", quota_limited_tmp)
    assert isolation.basetemp_parent().parent == Path("/var/tmp")


@pytest.mark.parametrize("key", ["TMPDIR", "TEMP", "TMP"])
def test_explicit_temp_environment_is_respected(monkeypatch, tmp_path, key):
    """REQ-HARNESS-5930: a nonempty explicit temporary root keeps tempfile semantics."""
    clear_temp_overrides(monkeypatch)
    monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setattr(isolation, "gettempdir", lambda: str(tmp_path))
    assert isolation.basetemp_parent().parent == tmp_path


@pytest.mark.parametrize("available,writable", [(False, True), (True, False)])
def test_unavailable_disk_scratch_uses_platform_default(monkeypatch, tmp_path, available, writable):
    """REQ-HARNESS-5930: portable fallback retains platforms without writable /var/tmp."""
    clear_temp_overrides(monkeypatch)
    monkeypatch.setattr(Path, "is_dir", lambda path: available)
    monkeypatch.setattr(isolation.os, "access", lambda path, mode: writable)
    monkeypatch.setattr(isolation, "gettempdir", lambda: str(tmp_path))
    assert isolation.basetemp_parent().parent == tmp_path


def test_empty_temp_overrides_allow_disk_scratch(monkeypatch):
    """REQ-HARNESS-5930: tempfile treats empty environment values as unset."""
    for key in ["TMPDIR", "TEMP", "TMP"]:
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(Path, "is_dir", lambda path: True)
    monkeypatch.setattr(isolation.os, "access", lambda path, mode: True)
    assert isolation.basetemp_parent().parent == Path("/var/tmp")


def test_missing_option_does_not_mutate_config():
    """REQ-HARNESS-5930: configs without a basetemp option remain untouched."""
    config = SimpleNamespace(option=SimpleNamespace())
    assert isolation.install_isolated_basetemp(config) is None
    assert not hasattr(config.option, "basetemp")


@pytest.mark.parametrize("has_uid", [True, False])
def test_missing_username_uses_identity_fallback(monkeypatch, has_uid):
    """REQ-HARNESS-5930: sandboxes without passwd records still get a per-user base."""

    def missing_user():
        raise KeyError("no passwd entry")

    monkeypatch.setattr(isolation.getpass, "getuser", missing_user)
    if has_uid:
        monkeypatch.setattr(isolation.os, "getuid", lambda: 123)
    else:
        monkeypatch.delattr(isolation.os, "getuid")
    assert isolation.basetemp_parent().name == (
        "pytest-carnot-uid123" if has_uid else "pytest-carnot-unknown"
    )


def test_stale_pruning_preserves_files_and_handles_stat_errors(tmp_path, monkeypatch):
    """REQ-HARNESS-5930: janitor failures and non-directories cannot break test execution."""
    parent = tmp_path / "parent"
    parent.mkdir()
    ordinary = parent / "keep-file"
    ordinary.write_text("original data")
    inaccessible = parent / "cannot-stat"
    inaccessible.mkdir()
    original = Path.stat

    def stat(path, *args, **kwargs):
        if path == inaccessible:
            raise PermissionError("stat denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    assert isolation.prune_stale_bases(parent, now=10**12) == []
    assert ordinary.read_text() == "original data"


def test_default_invocation_writes_real_fixture(pytestconfig, tmp_path, monkeypatch):
    """REQ-HARNESS-5930: the real fixture remains writable under the configured isolated base."""
    monkeypatch.setattr(isolation, "prune_stale_bases", lambda **kwargs: [])
    assigned = isolation.install_isolated_basetemp(
        SimpleNamespace(option=SimpleNamespace(basetemp=None))
    )
    assert assigned.parent == isolation.basetemp_parent()
    assert Path(pytestconfig.option.basetemp) in tmp_path.parents
    fixture = tmp_path / "authority-bytes"
    content = bytes(range(256)) * 256
    fixture.write_bytes(content)
    assert fixture.read_bytes() == content
