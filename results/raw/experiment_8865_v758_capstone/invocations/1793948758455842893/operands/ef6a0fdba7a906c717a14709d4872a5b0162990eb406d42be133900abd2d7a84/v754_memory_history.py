"""REQ-VERIFY-8801: retain assertions while supplying exact historical fixture paths."""

from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
from typing import Any

from coverage import Coverage
import yaml

from carnot.reporting import v753_qualification_runner as original
from carnot.reporting import v754_repaired_memory as a
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
original_failure = original.original_failure


def suite(raw: Path) -> str:
    """Copy the repaired test bytes; only their fixture roots and CLI arguments change."""
    directory = raw / "repaired_family"
    directory.mkdir()
    source = a.ROOT / a.previous.TESTS[0]
    target = directory / source.name
    target.write_bytes(source.read_bytes())
    root = raw / "v753_authority"
    design = root / a.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes((a.ROOT / a.ARCHIVES[1]).read_bytes())
    tasks = a.parse_design(design.read_text(), milestone=a.previous.MILESTONE)[1]
    for name in [a.ACTIVE, a.STAGED]:
        (root / name).write_text(yaml.safe_dump(dict(milestone=a.previous.MILESTONE, tasks=tasks)))
    wrapper = directory / "original_cli.py"
    wrapper.write_text(
        "import sys\nfrom carnot.reporting import v753_qualification_runner as runner\n"
        f"raise SystemExit(runner.main(sys.argv[1:] + ['--root', {str(root)!r}]))\n"
    )
    (directory / "conftest.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "from carnot.reporting import v753_branch_qualification as api\n"
        "@pytest.fixture(autouse=True)\ndef historical_bindings(monkeypatch):\n"
        "    snapshot, freeze = api.snapshot, api.freeze\n"
        f"    root = Path({str(root)!r})\n"
        "    monkeypatch.setattr(api, 'snapshot', lambda source, target: snapshot(root if source == api.ROOT else source, target))\n"
        "    monkeypatch.setattr(api, 'freeze', lambda source, raw, date: freeze(root if source == api.ROOT else source, raw, date))\n"
        f"    monkeypatch.setattr(api, 'CLI', {str(wrapper)!r})\n"
    )
    assert target.read_bytes() == source.read_bytes()
    return os.path.relpath(target, a.ROOT)


def repaired_control(raw: Path) -> Json:
    """Trace the actual original CLI, because a fixture wrapper cannot cover its entry point."""
    scratch = Path(tempfile.mkdtemp(prefix="exp8801-repaired-", dir="/var/tmp"))
    output = scratch / (a.previous.NAME + ".json")
    database = raw / "repaired_sqlite"
    database.mkdir()
    config = raw / "repaired_coverage.ini"
    config.write_text(
        f"[run]\nparallel=true\npatch=subprocess\ndata_file={database}/.coverage\n[report]\nexclude_lines=\n"
    )
    receipt = child(
        "repaired_original_private_cli",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            str(a.ROOT / a.previous.CLI),
            "--private-control",
            "--root",
            str(raw / "v753_authority"),
            "--output",
            str(output),
        ],
        raw / "repaired_checks",
        deadline=120,
    )
    value = json.loads(output.read_bytes())
    scores = {k: value[k] for k in a.previous.old_report.SCORES}
    cov = Coverage(data_file=str(database / ".coverage"), config_file=False)
    cov.combine(keep=True)
    cov.save()
    saved = raw / "repaired_failure_primary.json"
    shutil.copyfile(output, saved)
    shutil.copytree(scratch / "raw", raw / "repaired_publication")
    ref = a.reference(database / ".coverage")
    return dict(
        passed=receipt["passed"]
        and value["verdict_class"] == "disqualified"
        and not any(scores.values()),
        receipt=receipt,
        readiness_scores=scores,
        sqlite_reference=ref,
        references=[ref, a.reference(saved)]
        + [a.reference(Path(receipt[s + "_path"])) for s in ["stdout", "stderr"]],
    )
