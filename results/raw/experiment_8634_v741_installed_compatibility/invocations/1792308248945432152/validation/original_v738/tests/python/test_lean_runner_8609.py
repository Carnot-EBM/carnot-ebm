"""REQ-VERIFY-8609: private CLI validation and bounded command orchestration."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import lean_qualification_8609 as e
from carnot.reporting import lean_runner_8609 as runner
from carnot.reporting.current_work_receipt import atomic_json


def test_standalone_cli_imports_without_pythonpath(tmp_path):
    """SCENARIO-VERIFY-8609-PRIVATE-IMPORTS: direct entry bootstraps outside the repo."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(e.ROOT / e.CLI), "--date", "invalid"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert "[exp8609] error=" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_private_cli_and_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8609-CUSTODY: actual CLI bytes reach unchanged readers."""
    work = e.measure(tmp_path / "work", "20261017", root=tmp_path)
    monkeypatch.setattr(runner.e, "measure", lambda *a: work)
    monkeypatch.setattr(runner, "validate", lambda *a: [{"name": "unit_required", "passed": True}])
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--date", "20261017", "--output", str(output)]) == 0
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    assert runner.main(["--date", "invalid"]) == 1
    assert runner.main(["--date", "2026101"]) == 1
    for args, expected in [
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--date", "bad"], 1),
    ]:
        result = subprocess.run(
            [sys.executable, "-u", e.CLI, *args], capture_output=True, timeout=60
        )
        assert result.returncode == expected, result.stderr
    clock = iter([0.0, 3600.0])
    monkeypatch.setattr(runner, "monotonic", lambda: next(clock))
    assert runner.main(["--date", "20261017", "--output", str(output)]) == 1
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_validation_commands_and_union(tmp_path, monkeypatch):
    """REQ-VERIFY-8609: empty collection cannot expand into the repository suite."""
    nodes = [
        e.TESTS[0] + "::test_one",
        e.TESTS[1] + "::test_two",
        runner.historical.e.TEST + "::test_original",
    ]
    commands = []

    def child(name, argv, raw, **kwargs):
        commands.append(argv)
        raw.mkdir(parents=True, exist_ok=True)
        log = raw / (name + ".stdout")
        log.write_text("\n".join(nodes) if name == "collection" else "")
        for arg in argv:
            if arg.startswith("--junitxml="):
                Path(arg.split("=", 1)[1]).write_text("<testsuite/>")
        if name == "coverage_json":
            files = {
                str(e.ROOT / p): dict(
                    summary=dict(num_statements=1, covered_lines=1), missing_lines=[]
                )
                for p in e.OWNED
            }
            Path(argv[-1]).write_text(
                json.dumps(dict(files=files, totals=dict(percent_covered=100)))
            )
        return dict(name=name, passed=True, stdout_path=str(log))

    work = e.measure(tmp_path / "unit-work", "20261017", root=tmp_path)
    monkeypatch.setattr(runner, "child", child)
    monkeypatch.setattr(
        runner.e, "upstream", lambda *a: (dict(original_node_union=dict(collected=nodes[:])), [])
    )
    monkeypatch.setattr(runner, "checkout", lambda root: dict(root=str(tmp_path)))
    monkeypatch.setattr(runner, "xml_nodes", lambda p: [nodes[int(p.stem.split("-")[-1])]])
    receipts = runner.validate(work, tmp_path / "validation")
    assert all(r["passed"] for r in receipts)
    assert any("--strict" in c for c in commands)
    assert work["original_node_union"]["passed"]
    nodes.clear()
    assert any(not r["passed"] for r in runner.validate({}, tmp_path / "empty"))
    assert runner.terminal(tmp_path / "unused", tmp_path)["passed"]


def test_cli_rejects_rehashed_owned_failure(tmp_path):
    """REQ-VERIFY-8609: self-consistent hashes cannot conceal required failures."""
    work = e.measure(tmp_path / "work", "20261017", root=tmp_path)
    value = e.artifact(work, [{"passed": False}], tmp_path / (e.NAME + ".json"))
    value["lean_measurement_ready_score"] = 1
    value["reproducibility_checksum"] = e.checksum(value)
    path = tmp_path / "mutated.json"
    atomic_json(path, value)
    assert runner.main(["--cold-replay", str(path)]) == 1


def test_original_authority_checkout(tmp_path):
    """REQ-VERIFY-8609: original nodes execute unchanged against sealed V738 authority."""
    from carnot.reporting.lean_original_8609 import checkout
    from carnot.reporting import v738_contract_methods as methods

    receipt = checkout(tmp_path / "original")
    root = Path(receipt["root"])
    assert methods.authority(root)["ready"]
    assert (root / "tests/python/test_resident_provenance_8593.py").read_bytes() == (
        e.ROOT / "tests/python/test_resident_provenance_8593.py"
    ).read_bytes()
    assert receipt["activation"]["sha256"] != receipt["live_activation"]["sha256"]


def test_historical_copy_tampering(tmp_path, monkeypatch):
    """REQ-VERIFY-8609: the original authority cannot license changed source bytes."""
    from carnot.reporting import lean_original_8609 as original

    copy = original.shutil.copyfile

    def altered(source, target, **kwargs):
        result = copy(source, target, **kwargs)
        if str(source).endswith(original.historical.TEST):
            Path(target).write_text("altered historical assertions")
        return result

    monkeypatch.setattr(original.shutil, "copyfile", altered)
    with pytest.raises(ValueError, match="historical_source_copy"):
        original.checkout(tmp_path / "changed")
