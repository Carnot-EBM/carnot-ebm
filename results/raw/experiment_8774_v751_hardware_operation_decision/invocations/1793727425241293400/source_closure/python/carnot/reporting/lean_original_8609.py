"""REQ-VERIFY-8609: replay historical tests with their exact original scheduling bytes."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from carnot.reporting import lean_resident_8608 as prototype
from carnot.reporting import resident_provenance_8593 as historical
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]


def checkout(root: Path) -> Json:
    """Copy source bytes and original authority without patching assertions or readers.

    A historical producer must see its original eleven tasks in fresh children.
    Shared result files remain read-only inputs; test outputs use private scratch.
    """
    source = prototype.ROOT
    root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(
        (source / "openspec/change-proposals/v739-methods-protocol.json").read_bytes()
    )
    ref = protocol["v738_activation_reference"]
    activation = prototype.authenticate(ref)
    for entry in source.iterdir():
        if entry.name not in {"python", "scripts", "tests", "openspec", "research-roadmap.yaml"}:
            (root / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    for name in ["python", "scripts", "openspec"]:
        shutil.copytree(
            source / name,
            root / name,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.profraw"),
        )
    selected = ["tests/python/conftest.py", historical.TEST]
    for name in selected:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    (root / "research-roadmap.yaml").write_bytes(activation.read_bytes())
    files = []
    for name in [
        *historical.OWNED,
        historical.TEST,
        historical.methods.PROTOCOL,
        "python/carnot/reporting/v738_contract_methods.py",
    ]:
        original, copied = prototype.reference(source / name), prototype.reference(root / name)
        if original["sha256"] != copied["sha256"]:
            raise ValueError("historical_source_copy")
        files.append(dict(original=original, copied=copied))
    receipt: Json = dict(
        root=str(root),
        activation=ref,
        live_activation=prototype.reference(source / "research-roadmap.yaml"),
        replay_activation=prototype.reference(root / "research-roadmap.yaml"),
        source_files=files,
        source_manifest_sha256=canonical_hash(files),
        rule="Unchanged historical test source, original dates, original eleven-task authority; no reader or assertion patches.",
    )
    atomic_json(root / "authority_receipt.json", receipt)
    return receipt
