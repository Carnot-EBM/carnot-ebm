"""REQ-VERIFY-8777: private authority preserves original source and assertion bytes."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from types import SimpleNamespace
from typing import Any

from coverage import CoverageData

from carnot.reporting import evidence_qualification_8777 as q
from carnot.reporting import evidence_set_methods_8768 as m
from carnot.reporting import evidence_set_runner_8768 as original
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from carnot.reporting.v746_retained_evidence import reference

Json = dict[str, Any]
V751 = "openspec/change-proposals/research-roadmap-v751-preserved-20261103.md"
PRIMARY = "results/experiment_8768_v751_evidence_set_methods.json"
ORIGINAL_OWNED, ORIGINAL_TESTS, ORIGINAL_CLI = list(m.OWNED), list(m.TESTS), m.CLI


def freeze_archive(root: Path) -> Path:
    """Copy results instead of linking them so private cleanup cannot change historical evidence."""
    root.mkdir(parents=True)
    old = json.loads((q.ROOT / PRIMARY).read_bytes())
    paths = [
        Path(r["path"]) for k in ["source_artifact_hashes", "code_config_hashes"] for r in old[k]
    ]
    paths += [q.ROOT / p for p in [PRIMARY, V751, *ORIGINAL_OWNED, *ORIGINAL_TESTS]]
    for source in paths:
        target = root / source.relative_to(q.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (root / "openspec/change-proposals/research-roadmap-vNEXT.md").write_bytes(
        (q.ROOT / V751).read_bytes()
    )
    (root / ".venv").symlink_to(q.ROOT / ".venv", target_is_directory=True)
    (root / "historical_cli.py").write_text(
        "from pathlib import Path\nimport sys\n"
        f"sys.path[:0] = {[str(q.ROOT / 'python'), str(q.ROOT)]!r}\n"
        "from carnot.reporting.evidence_archive_8777 import historical_main\n"
        f"historical_main(Path({str(root)!r}))\n"
    )
    tests = root / "original_tests"
    tests.mkdir()
    shutil.copyfile(q.ROOT / ORIGINAL_TESTS[0], tests / Path(ORIGINAL_TESTS[0]).name)
    (tests / "conftest.py").write_text(
        "from pathlib import Path\nimport pytest\n"
        "from carnot.reporting.evidence_archive_8777 import configured\n"
        "@pytest.fixture(autouse=True)\ndef original_authority():\n"
        f"    with configured(Path({str(root)!r})):\n        yield\n"
    )
    (root / "conftest.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n"
        f"        if item.path.is_relative_to(Path({str(root)!r})):\n"
        f"            item._nodeid = os.path.relpath(item.path, {str(q.ROOT)!r}) + '::' + item.name\n"
    )
    (root / "test_failed_owned.py").write_text(
        '"""REQ-VERIFY-8777: genuine failed validation."""\n\n\ndef test_failed_owned():\n    assert False\n'
    )
    atomic_json(
        root / "archive_seal.json",
        dict(
            references=[
                reference(p) for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
            ]
        ),
    )
    return root


@contextmanager
def configured(root: Path, failure: bool = False) -> Iterator[None]:
    """Change only authority locations; the old CLI, comparisons and tests remain unchanged."""
    saved = (m.ROOT, m.CLI, m.OWNED, original.terminal_runner.CLI, original.validation_runner.a)
    m.ROOT, m.CLI = root, str(root / "historical_cli.py")
    m.OWNED = [str(q.ROOT / p) for p in ORIGINAL_OWNED]
    original.terminal_runner.CLI = m.CLI
    original.validation_runner.a = SimpleNamespace(
        ROOT=q.ROOT,
        OWNED=m.OWNED,
        TESTS=[os.path.relpath(root / "test_failed_owned.py", q.ROOT)] if failure else m.TESTS,
        CONSUMERS=[] if failure else m.CONSUMERS,
        reference=reference,
    )
    try:
        yield
    finally:
        m.ROOT, m.CLI, m.OWNED, original.terminal_runner.CLI, original.validation_runner.a = saved


def historical_main(root: Path) -> None:
    """Run the unchanged original entrypoint after binding its private authority."""
    requested = next(
        (Path(value) for flag, value in zip(sys.argv, sys.argv[1:]) if flag == "--output"), None
    )
    if (
        "--private-control" in sys.argv
        and requested is not None
        and requested.resolve().is_relative_to(q.ROOT)
    ):
        q.progress("unsafe_original_private_output_rejected")
        raise SystemExit(2)
    with configured(root, failure=True):
        runpy.run_path(str(q.ROOT / ORIGINAL_CLI), run_name="__main__")


def failure_control(root: Path, raw: Path) -> Json:
    """A real failing pytest child must reach line345 and publish disqualified evidence."""
    raw.mkdir(parents=True)
    output = raw / Path(PRIMARY).name
    config = raw / "cli_coverage.ini"
    sqlite = raw / "original_cli.sqlite"
    config.write_text(
        "[run]\ncore=sysmon\ndata_file="
        + str(sqlite)
        + "\ninclude=\n"
        + "\n".join("    " + str(q.ROOT / p) for p in ORIGINAL_OWNED + q.OWNED)
        + "\n[report]\nexclude_lines=\n"
    )
    receipt = child(
        "original_failed_validation",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            str(root / "historical_cli.py"),
            "--output",
            str(output),
        ],
        raw / "logs",
        expected=0,
        deadline=240,
    )
    value = json.loads(output.read_bytes())
    primitive = Path(value["primitive_reference"]["path"])
    data = CoverageData(basename=str(sqlite))
    data.read()
    executed = 345 in (
        data.lines(str(q.ROOT / "python/carnot/reporting/evidence_set_runner_8768.py")) or []
    )
    checks = json.loads(Path(value["validation_reference"]["path"]).read_bytes())
    actual_failure = any(
        r["name"].startswith("child_") and r["actual_exit"] == 1 and r["normal_exit"]
        for r in checks["receipts"]
    )
    terminal = json.loads((primitive.parent / "terminal_validation.json").read_bytes())
    result = dict(
        passed=receipt["passed"]
        and executed
        and actual_failure
        and terminal["passed"]
        and value["verdict_class"] == "disqualified"
        and value["evidence_set_methods_ready_score"] == 0,
        line345_executed=executed,
        published_verdict=value["verdict_class"],
        readiness=value["evidence_set_methods_ready_score"],
        receipt=receipt,
        primary_reference=reference(output),
        producer_sqlite_reference=reference(sqlite),
        nested_producer_sqlite_reference=reference(primitive.parent / "producer.sqlite"),
        coverage_config_reference=reference(config),
        validation_reference=reference(Path(value["validation_reference"]["path"])),
        terminal_reference=reference(primitive.parent / "terminal_validation.json"),
        actual_failed_check=actual_failure,
    )
    atomic_json(raw / "control.json", result)
    return result
