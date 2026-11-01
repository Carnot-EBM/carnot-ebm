"""REQ-REPORT-8660 / REQ-VERIFY-8660: private ownership precedes cleanup."""

from pathlib import Path
import shutil

import pytest

from carnot.reporting import installed_checkout_8660 as e


def test_owned_entries_and_recursive_alias_rejection(tmp_path):
    """SCENARIO-REPORT-8660-OWNERSHIP: no shared alias can enter a cleanup root."""
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "results").mkdir(parents=True)
    (source / "results/input.json").write_text("{}")
    target.mkdir()
    receipt = e.copy_entry(source / "results", target / "results")
    assert receipt["passed"]
    assert (target / "results").is_dir() and not (target / "results").is_symlink()
    assert (target / "results/input.json").read_bytes() == b"{}"
    e.audit_owned(target)
    (target / "results/evil").symlink_to(source / "results", target_is_directory=True)
    with pytest.raises(ValueError, match="cleanup_alias"):
        e.audit_owned(target)
    (target / "results/evil").unlink()
    shutil.rmtree(target)
    assert (source / "results/input.json").read_bytes() == b"{}"


def test_overlay_retains_original_except_materialization(tmp_path):
    """SCENARIO-VERIFY-8660-CUSTODY: exact test/helper source is separately sealed."""
    value = e.overlay(tmp_path)
    assert value["original_protocol"]["sha256"].endswith(e.PROTOCOL_HASH)
    assert "copy_entry(entry, root / entry.name)" in Path(value["materializer"]["path"]).read_text()
    assert value["original_helper"]["sha256"] != value["materializer"]["sha256"]
    assert value["test_sources"]


def test_guarded_failure_is_private(tmp_path):
    """SCENARIO-REPORT-8660-OWNERSHIP: the unmodified guard rejects actual removal."""
    result = e.guard_control(tmp_path)
    assert result["passed"]
    assert result["guard_enabled"]
    assert result["source_unchanged"]


def test_copy_files_empty_work_roots_and_manifest(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8660-OWNERSHIP: work roots and file operands have private inodes."""
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "plain").write_text("operand")
    log = tmp_path / "manifests"
    monkeypatch.setenv("CARNOT_8660_MATERIALIZATION_LOG", str(log))
    e.copy_entry(source / "plain", target / "plain")
    assert (source / "plain").stat().st_ino != (target / "plain").stat().st_ino
    (source / "work").mkdir()
    e.copy_entry(source / "work", target / "work")
    e.pytest_runtest_teardown()
    assert (log / "cleanup-audit.receipt").exists()
    shutil.rmtree(target)
    e.pytest_runtest_teardown()
    monkeypatch.delenv("CARNOT_8660_MATERIALIZATION_LOG")
    e.pytest_runtest_teardown()


def test_copied_hash_and_overlay_identity_fail_closed(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8660-CUSTODY: hash mismatch and unknown helper shapes fail."""
    source = tmp_path / "source/results"
    target = tmp_path / "target/results"
    source.mkdir(parents=True)
    target.parent.mkdir()
    (source / "input.json").write_text("{}")
    original = e.shutil.copyfile

    def changed(src, dst, **kwargs):
        result = original(src, dst, **kwargs)
        Path(dst).write_text("changed copy")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(e.shutil, "copyfile", changed)
        with pytest.raises(ValueError, match="copied_operand_hash"):
            e.copy_entry(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(e.subprocess, "check_output", lambda *a, **kw: b"wrong protocol")
        with pytest.raises(ValueError, match="original_protocol_hash"):
            e.overlay(tmp_path / "bad-protocol")
    helper = tmp_path / e.HELPER
    helper.parent.mkdir(parents=True)
    helper.write_text("unexpected helper")
    monkeypatch.setattr(e, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="materializer_source_shape"):
        e.overlay(tmp_path / "bad-helper")


def test_original_test_bytes_match_independent_historical_seal(tmp_path):
    """SCENARIO-VERIFY-8660-CUSTODY: live assertions must match the sealed Exp8646 inventory."""
    import json
    from carnot.reporting.installed_lean_8633 import load
    from carnot.reporting.installed_report_8660 import HISTORY, HISTORY_PIN

    history = load(dict(path=str(e.ROOT / HISTORY), sha256=HISTORY_PIN))
    reference = next(
        r
        for r in history["raw_shard_hashes"]
        if r["path"].endswith("/validation/collected_nodes.json")
    )
    sealed = load(reference)
    original = {r["path"]: r["sha256"] for r in sealed["source_hashes"]}
    manifest = e.overlay(tmp_path)
    assert len(manifest["test_sources"]) == 8
    assert all(original[r["path"]] == r["sha256"] for r in manifest["test_sources"])
    assert len(history["original_node_union"]["collected"]) == 122
