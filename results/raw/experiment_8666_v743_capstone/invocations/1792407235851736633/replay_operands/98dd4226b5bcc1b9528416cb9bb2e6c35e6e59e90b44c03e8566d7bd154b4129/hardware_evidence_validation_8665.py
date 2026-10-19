"""REQ-VERIFY-8665: retain original child SQLite and isolate consumer coverage."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
from unittest.mock import patch

from coverage import CoverageData

from carnot.reporting import hardware_operation_evidence_8665 as e
from carnot.reporting import hardware_operation_runner_8625 as shared
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
SUITES = [
    e.TEST,
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    "tests/python/test_primary_publication_7928.py",
]


def validate(raw: Path) -> list[Json]:
    """Collect each suite once; independent consumers cannot supply producer statement coverage."""
    folder = raw / "validation"
    retained, copies = folder / "retained", folder / "combine_copies"
    inventories = []

    def retained_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        if name.startswith("tests"):
            directory = folder / "children" / name
            directory.mkdir(parents=True, exist_ok=True)
            owned = any(e.TEST in a or e.CLI in a for a in argv)
            include = (
                ",".join(str(e.ROOT / p) for p in e.OWNED)
                if owned
                else str(e.ROOT / "python/carnot/reporting/*")
            )
            config = directory / "coverage.ini"
            config.write_text(
                "[run]\nparallel = true\npatch = subprocess\ninclude = "
                + include
                + "\n[report]\nexclude_lines =\n"
            )
            argv = [
                f"COVERAGE_FILE={directory / '.coverage'}"
                if a.startswith("COVERAGE_FILE=")
                else f"COVERAGE_RCFILE={config}"
                if a.startswith("COVERAGE_RCFILE=")
                else a
                for a in argv
            ]
        if name == "combine":
            copies.mkdir(parents=True, exist_ok=True)
            for original in retained.rglob(".coverage*"):
                if (original.parent / "owned").exists():
                    shutil.copyfile(
                        original,
                        copies / (".coverage." + original.parent.name + "." + original.name),
                    )
            argv = [*argv[:-1], "--keep", str(copies)]
        result = child(name, argv, logs, **kwargs)
        if name.startswith("tests"):
            snapshots = retained / name
            snapshots.mkdir(parents=True, exist_ok=True)
            if owned:
                (snapshots / "owned").write_text("unit and CLI coverage only\n")
            inventory = []
            for database in sorted(directory.glob(".coverage*")):
                snapshot = snapshots / database.name
                shutil.copyfile(database, snapshot)
                snapshot.chmod(0o444)
                data = CoverageData(basename=str(snapshot))
                data.read()
                inventory.append(
                    dict(
                        e.reference(snapshot),
                        measured_source_hashes=[
                            e.reference(Path(p))
                            for p in sorted(data.measured_files())
                            if Path(p).is_file()
                        ],
                        measured_source_lines={
                            p: sorted(data.lines(p) or []) for p in sorted(data.measured_files())
                        },
                    )
                )
            atomic_json(snapshots / "inventory.json", dict(databases=inventory, owned=owned))
            inventories.append(e.reference(snapshots / "inventory.json"))
        return result

    with (
        patch.object(shared, "e", e),
        patch.object(shared, "SUITES", SUITES),
        patch.object(shared, "child", retained_child),
    ):
        receipts = list(shared.validate(raw))
    receipts.append(
        dict(name="original_coverage_retention", passed=bool(inventories), inventories=inventories)
    )
    return receipts
