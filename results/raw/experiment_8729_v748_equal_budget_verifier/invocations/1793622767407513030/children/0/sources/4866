"""REQ-REPORT-8660: private copies let historical cleanup keep the evidence guard.

Only the historical materializer is overlaid. Original tests, numerical helpers
and assertions retain their own byte identities in the child's namespace.
"""

from __future__ import annotations

import json
import os
import time
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_HASH = "7865070fe8b58fb21e301bd5b1a75ac7ba7923ab9d4ccc5b4fd0051da3c2f966"
HELPER = "python/carnot/reporting/lean_original_8609.py"
PROTOCOL = "python/carnot/inference/resident_protocol.py"
INPUT_NUMBERS = (8431, 8584, 8593, 8596)


def reference(path: Path) -> Json:
    """A path alone cannot authenticate an operand copied for historical tests."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def copy_entry(source: Path, target: Path) -> Json:
    """Give every non-source root its own inode, without copying unrelated studies.

    Nested checkout tests need the head and original resident publications.
    Other directories are empty owned work roots; they never alias shared data.
    """
    copied = []
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        if source.name == "results":
            selected = [
                p
                for p in source.glob("*.json")
                if any(p.name.startswith(f"experiment_{number}_") for number in INPUT_NUMBERS)
                or p.name == "input.json"
            ]
            for operand in selected:
                original = reference(operand)
                shutil.copyfile(operand, target / operand.name)
                if sha256_file(target / operand.name) != original["sha256"]:
                    raise ValueError("copied_operand_hash")
                copied.append(dict(original=original, copied=reference(target / operand.name)))
    else:
        shutil.copyfile(source, target)
        copied.append(dict(original=reference(source), copied=reference(target)))
    audit_owned(target.parent, names=[target.name])
    receipt = dict(passed=True, root=str(target), copied_operands=copied)
    atomic_json(target.parent / ("owned-" + target.name + ".json"), receipt)
    log_root = os.environ.get("CARNOT_8660_MATERIALIZATION_LOG")
    if log_root:
        atomic_json(Path(log_root) / (str(time.time_ns()) + ".json"), receipt)
    return receipt


def pytest_runtest_teardown() -> None:
    """Audit owned cleanup roots before pytest removes any historical fixture."""
    log_root = os.environ.get("CARNOT_8660_MATERIALIZATION_LOG")
    if log_root:
        for manifest in Path(log_root).glob("*.json"):
            value = json.loads(manifest.read_bytes())
            root = Path(value["root"])
            if root.exists():
                audit_owned(root.parent, names=[root.name])
        atomic_json(Path(log_root) / "cleanup-audit.receipt", dict(passed=True))


def audit_owned(root: Path, *, names: list[str] | None = None) -> Json:
    """Reject aliases recursively before launch and again before real cleanup."""
    roots = [root / name for name in names] if names is not None else [root]
    count = 0
    for owned in roots:
        for path in [owned, *owned.rglob("*")]:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("cleanup_alias:" + str(path))
            count += 1
    return dict(passed=True, paths_checked=count, root=str(root))


def overlay(folder: Path) -> Json:
    """Seal one explicit helper diff and the exact ae7e7115c6 protocol bytes."""
    folder.mkdir(parents=True, exist_ok=True)
    original = (ROOT / HELPER).read_text()
    old = "(root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())"
    if original.count(old) != 1:
        raise ValueError("materializer_source_shape")
    replacement = (
        "from carnot.reporting.installed_checkout_8660 import copy_entry\n"
        "            copy_entry(entry, root / entry.name)"
    )
    materializer = folder / "owned_lean_original_8609.py"
    materializer.write_text(original.replace(old, replacement))
    protocol = folder / "original_resident_protocol.py"
    protocol.write_bytes(
        subprocess.check_output(["git", "show", "ae7e7115c6:" + PROTOCOL], cwd=ROOT, timeout=10)
    )
    if sha256_file(protocol) != "sha256:" + PROTOCOL_HASH:
        raise ValueError("original_protocol_hash")
    history = json.loads(
        (ROOT / "results/experiment_8634_v741_installed_compatibility.json").read_bytes()
    )
    tests = list(
        dict.fromkeys(n.split("::")[0] for n in history["original_node_union"]["original_nodes"])
    )
    value = dict(
        original_helper=reference(ROOT / HELPER),
        materializer=reference(materializer),
        original_protocol=reference(protocol),
        test_sources=[reference(ROOT / p) for p in tests],
        authorized_change="Private root ownership only; original assertions unchanged",
    )
    atomic_json(folder / "overlay_manifest.json", value)
    return value


def guard_control(raw: Path) -> Json:
    """Use an actual private git-tracked operand to reproduce the same audit error."""
    raw.mkdir(parents=True, exist_ok=True)
    fixture = raw / "fixture"
    guard = fixture / "python/carnot/testing/tracked_results_guard.py"
    guard.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "python/carnot/testing/tracked_results_guard.py", guard)
    source = fixture / "results/input.json"
    source.parent.mkdir()
    source.write_text("private guarded evidence")
    before = sha256_file(source)
    alias = fixture / "cleanup/results"
    alias.parent.mkdir()
    alias.symlink_to(source.parent, target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(fixture)], check=True, timeout=10)
    subprocess.run(["git", "-C", str(fixture), "add", "results"], check=True, timeout=10)
    code = (
        f'import runpy;g=runpy.run_path({str(guard)!r});g["install"]();'
        f"from pathlib import Path;Path({str(alias / 'input.json')!r}).unlink()"
    )
    receipt = child(
        "guarded_cleanup_failure", [sys.executable, "-u", "-c", code], raw, expected=1, deadline=30
    )
    rejected = "TrackedResultWriteError" in Path(receipt["stderr_path"]).read_text()
    after = sha256_file(source)
    alias.unlink()
    audit = audit_owned(fixture)
    shutil.rmtree(fixture)
    return dict(
        passed=receipt["passed"] and rejected,
        guard_enabled=rejected,
        source_unchanged=before == after,
        audit=audit,
        receipt=receipt,
    )
