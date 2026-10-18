"""REQ-VERIFY-8640: bounded real CLI and exact publication readers."""

import json
from pathlib import Path
import sys

from carnot.reporting import v742_contract_methods as e
from carnot.reporting import v742_contract_runner as runner
from carnot.reporting.v709_execution import child


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8640-REPLAY: fresh children publish and reject rehashed claims."""
    output = tmp_path / "results" / (e.NAME + ".json")
    command = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "private",
        command
        + [
            "--date",
            "20261018",
            "--root",
            str(tmp_path / "missing"),
            "--output",
            str(output),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=180,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    for mode, args, expected in [
        ("cold", ["--cold-replay", str(output)], 0),
        ("date", ["--date", "bad"], 2),
        ("error", ["--deliberate-error"], 7),
        ("unsafe", ["--private-control"], 2),
    ]:
        assert child(mode, command + args, tmp_path / "logs", deadline=120, expected=expected)[
            "passed"
        ]
    value["completed_count"] += 1
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    assert child(
        "tampered",
        command + ["--cold-replay", str(output)],
        tmp_path / "logs",
        deadline=120,
        expected=1,
    )["passed"]


def test_manifest(tmp_path, monkeypatch):
    """REQ-VERIFY-8640: original disjoint IDs and invocation-only coverage are frozen."""

    def collected(private, suite):
        name = "collect_" + Path(suite).stem
        e.atomic_json(private / "collection" / (name + ".receipt.json"), dict(passed=True))
        return [suite + "::private"], dict(name=name, argv=[])

    monkeypatch.setattr(runner.collection, "collect", collected)
    plan = runner.manifest(tmp_path)
    assert {"ruff_check", "ruff_format", "strict_mypy", "spec_coverage", "coverage_report"} <= {
        r["name"] for r in plan
    }
    assert all("tests/python" not in r["argv"] for r in plan)
    assert next(r for r in plan if r["name"] == "owned_tests")["deadline"] == 600


def test_collection_receipt(tmp_path):
    """REQ-VERIFY-8640: collection logs are authenticated without another collection."""
    out, err, path = tmp_path / "out", tmp_path / "err", tmp_path / "receipt.json"
    out.write_text("original")
    err.write_text("")
    e.atomic_json(
        path,
        dict(
            passed=True,
            normal_exit=True,
            stdout_path=str(out),
            stdout_sha256=e.sha256_file(out),
            stderr_path=str(err),
            stderr_sha256=e.sha256_file(err),
        ),
    )
    digest = e.sha256_file(path)
    assert runner.verify_collection(path, digest)
    out.write_text("forged")
    assert not runner.verify_collection(path, digest)
    assert not runner.verify_collection(tmp_path / "missing", digest)


def test_coverage_custody(tmp_path):
    """REQ-VERIFY-8640: read actual SQLite coverage before combining independent copies."""
    from coverage import CoverageData

    private, raw = tmp_path / "private", tmp_path / "raw"
    private.mkdir()
    data = CoverageData(basename=str(private / ".coverage.actual"))
    data.add_lines({str(e.ROOT / e.OWNED[0]): {1, 7}})
    data.write()
    digest = e.sha256_file(private / ".coverage.actual")
    receipt = runner.retain_coverage(private, raw)
    assert receipt["passed"]
    assert e.sha256_file(private / ".coverage.actual") == digest
    assert receipt["original_references"][0]["sha256"] == digest
    assert runner.retain_coverage(tmp_path / "missing", raw / "absent")["passed"] is False
    (private / ".coverage.corrupt").write_bytes(b"corrupt")
    assert not runner.retain_coverage(private, raw / "bad")["passed"]


def test_execute_capture(tmp_path):
    """REQ-VERIFY-8640: actual failed children disqualify and retain missing coverage."""
    private = tmp_path / "private"
    private.mkdir()
    plan = [
        dict(
            name="owned_tests",
            argv=[sys.executable, "-c", "raise SystemExit(3)"],
            deadline=10,
            expected=0,
            scope="owned",
        ),
        dict(
            name="coverage_combine",
            argv=[sys.executable, "-c", "pass", str(private / "coverage_copies")],
            deadline=10,
            expected=0,
            scope="owned",
        ),
    ]
    receipts = runner.execute(plan, tmp_path / "raw" / "checks")
    assert receipts[0]["exit_code"] == 3
    assert not receipts[-1]["passed"]


def test_empty_coverage(tmp_path):
    """SCENARIO-VERIFY-8640-REPLAY: an empty real coverage database cannot qualify."""
    private = tmp_path / "empty"
    private.mkdir()
    (private / ".coverage.empty").touch()
    assert not runner.retain_coverage(private, tmp_path / "raw")["passed"]
