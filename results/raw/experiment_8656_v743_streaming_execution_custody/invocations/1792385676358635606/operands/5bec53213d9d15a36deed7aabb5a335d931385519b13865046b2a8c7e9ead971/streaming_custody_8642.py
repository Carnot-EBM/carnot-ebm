"""REQ-VERIFY-8642: retain real child data before another suite can consume it."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from coverage import Coverage, CoverageData

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.streaming_source_8629 import reference
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def freeze(path: Path, raw: Path) -> Json:
    """Address retained bytes by their hash so replacing a file cannot repair custody."""
    digest = sha256_file(path)
    target = raw.absolute() / "operands" / digest.split(":")[1] / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != path.read_bytes():
        raise ValueError("immutable_collision")
    target.write_bytes(path.read_bytes()) if not target.exists() else None
    target.chmod(0o444)
    return reference(target)


def authenticate(ref: Json) -> bool:
    """Reject renamed aliases as well as changed bytes; both break the frozen address."""
    path = Path(ref["path"])
    return bool(
        path.is_file()
        and path.absolute() == path.resolve()
        and path.parent.name == ref["sha256"].split(":")[-1]
        and sha256_file(path) == ref["sha256"]
    )


def inventory(path: Path, sources: list[Path]) -> list[Json]:
    """Read SQLite statements and contexts rather than treating a filename as evidence."""
    data = CoverageData(basename=str(path))
    data.read()
    allowed = {str(p.absolute()) for p in sources}
    files = sorted(data.measured_files())
    if set(files) - allowed:
        raise ValueError("foreign_source_data")
    return [
        dict(
            source=f,
            source_sha256=sha256_file(Path(f)),
            lines=sorted(data.lines(f) or []),
            contexts={str(n): sorted(v) for n, v in data.contexts_by_lineno(f).items()},
        )
        for f in files
    ]


def capture(base: Path, raw: Path, sources: list[Path]) -> list[Json]:
    """Copy each child's actual unsuffixed or parallel data before starting more work."""
    paths = sorted(p for p in base.parent.glob(base.name + "*") if p.is_file())
    rows = [
        dict(
            reference=freeze(p, raw),
            original_path=str(p),
            unsuffixed=p == base,
            inventory=inventory(p, sources),
        )
        for p in paths
    ]
    if not any(f["lines"] for row in rows for f in row["inventory"]):
        raise ValueError("missing_child_statements")
    return rows


def combine(rows: list[Json], sources: list[Path], raw: Path) -> Json:
    """Combine disposable copies so even Coverage.py cleanup cannot consume raw evidence."""
    raw.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        ref = row["reference"]
        if not authenticate(ref) or inventory(Path(ref["path"]), sources) != row["inventory"]:
            raise ValueError("child_custody")
        (raw / f"aggregate.{i}").write_bytes(Path(ref["path"]).read_bytes())
    cov = Coverage(data_file=str(raw / "aggregate"), config_file=False)
    cov.combine(data_paths=[str(raw)], keep=True)
    cov.save()
    counts = []
    for source in sources:
        _, statements, _, missing, _ = cov.analysis2(str(source))
        counts.append(
            dict(
                source=str(source),
                statements=statements,
                missing=missing,
                statement_count=len(statements),
                covered_count=len(statements) - len(missing),
            )
        )
    return dict(
        passed=bool(counts) and all(not r["missing"] for r in counts),
        statement_count=sum(r["statement_count"] for r in counts),
        files=counts,
    )


def retention_regression(raw: Path) -> Json:
    """Run the late hook against a real unsuffixed child, retaining its failed verdict."""
    from carnot.reporting import streaming_source_runner_8629 as old
    from carnot.reporting import triviaplus_extraction_runner_8620 as shipped

    with TemporaryDirectory(prefix="exp8642-retention-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        source = scratch / "probe.py"
        source.write_text("print('actual numerical work', sum(range(16)), flush=True)\n")
        base = scratch / ".coverage"
        with patch.dict(os.environ, COVERAGE_FILE=str(base)):
            receipt = child(
                "retention_probe",
                [
                    os.sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=/dev/null",
                    "--include=" + str(source),
                    str(source),
                ],
                raw / "regression_logs",
            )
        retained = capture(base, raw, [source])

        # The original hook scans only suffixes at this late boundary.
        def small_runner(_: Path) -> Json:
            return dict(
                receipts=[
                    shipped.child(
                        "coverage_combine",
                        [os.sys.executable, "-m", "coverage", "combine", "--keep", str(scratch)],
                        raw / "regression_logs",
                        expected=1,
                    )
                ]
            )

        with patch.object(shipped, "run_checks", small_runner):
            original = old.run_checks(raw / "old_hook")["receipts"][-1]
        source_ref = freeze(source, raw)
    return dict(
        passed=receipt["passed"] and not original["passed"] and not scratch.exists(),
        original_coverage_retention=original["passed"],
        direct_capture=retained,
        scratch_cleanup=not scratch.exists(),
        source_reference=source_ref,
        receipt=receipt,
    )
