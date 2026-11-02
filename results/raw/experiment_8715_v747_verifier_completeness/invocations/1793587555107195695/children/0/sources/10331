"""REQ-VERIFY-8695: retained child bytes, not mutable scratch, grant authority."""

from copy import deepcopy
from pathlib import Path
import shutil
import sys

import pytest

from carnot.reporting import v746_retained_evidence as e
from carnot.reporting.v709_execution import child


def real_child(tmp_path, name, text="assert True"):
    """Run actual pytest under coverage so custody tests cannot invent execution."""
    scratch = tmp_path / name
    scratch.mkdir()
    source = scratch / "test_control.py"
    source.write_text(f"def test_real():\n    {text}\n")
    config = scratch / "coverage.ini"
    config.write_text(f"[run]\nparallel=true\ndata_file={scratch}/.coverage\n")
    nodes = [str(source) + "::test_real"]
    receipt = child(
        name,
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            *nodes,
            "--junitxml=" + str(scratch / "junit.xml"),
        ],
        tmp_path / "logs",
        deadline=60,
    )
    return scratch, source, nodes, receipt


def test_intervening_combine_and_replacement(tmp_path):
    """SCENARIO-VERIFY-8695-RETENTION: combining copies cannot erase prior child evidence."""
    scratch, source, nodes, receipt = real_child(tmp_path, "first")
    first = e.retain(
        receipt, nodes, scratch, tmp_path / "retained1", [e.reference(source)], "producer"
    )
    assert e.combine([first], tmp_path / "intervening")["references"]
    shutil.rmtree(scratch)
    scratch.mkdir()
    (scratch / ".coverage.replaced").write_bytes(b"replacement is not SQLite")
    second_scratch, second_source, second_nodes, second_receipt = real_child(tmp_path, "second")
    second = e.retain(
        second_receipt,
        second_nodes,
        second_scratch,
        tmp_path / "retained2",
        [e.reference(second_source)],
        "producer",
    )
    # Retained source copies authenticate execution after the original scratch is removed.
    result = e.aggregate([first, second], nodes + second_nodes)
    assert result["passed"] and len(result["node_ids"]) == 2
    assert len(e.combine([first, second], tmp_path / "producer")["references"]) == 2
    assert e.combine([], tmp_path / "consumer")["references"] == []
    assert e.load(first)["source_hashes"][0]["sha256"] == e.sha256_file(
        Path(e.load(first)["source_hashes"][0]["path"])
    )


@pytest.mark.parametrize(
    "mode", ["missing", "tampered", "duplicate", "source", "failed", "rehashed", "junit", "empty"]
)
def test_invalid_child_evidence(tmp_path, mode):
    """SCENARIO-VERIFY-8695-RETENTION: each evidence failure independently rejects custody."""
    scratch, source, nodes, receipt = real_child(tmp_path, "child")
    row = e.retain(
        receipt, nodes, scratch, tmp_path / "retained", [e.reference(source)], "producer"
    )
    value = e.load(row)
    refs = [row]
    if mode == "missing":
        Path(value["sqlite_references"][0]["path"]).unlink()
    if mode == "tampered":
        Path(value["sqlite_references"][0]["path"]).write_bytes(b"changed")
    if mode == "duplicate":
        refs.append(row)
    if mode == "source":
        Path(value["source_hashes"][0]["path"]).write_text("source drift")
    if mode == "failed":
        value["command_receipt"]["normal_exit"] = False
        e.atomic_json(Path(row["path"]), value)
        row.update(e.reference(Path(row["path"])))
    if mode == "rehashed":
        value["node_ids"] = ["forged::test_real"]
        e.atomic_json(Path(row["path"]), value)
    if mode == "junit":
        Path(value["junit_reference"]["path"]).write_text("<changed/>")
    assert not e.aggregate([] if mode == "empty" else refs, nodes)["passed"]


def test_failed_and_absent_retention(tmp_path):
    """REQ-VERIFY-8695: failed commands and empty coverage never gain valid receipts."""
    scratch, source, nodes, receipt = real_child(tmp_path, "failed", "assert False")
    row = e.retain(
        receipt, nodes, scratch, tmp_path / "retained", [e.reference(source)], "producer"
    )
    assert not e.aggregate([row], nodes)["passed"]
    for p in scratch.glob(".coverage*"):
        p.unlink()
    row = e.retain(receipt, nodes, scratch, tmp_path / "empty", [e.reference(source)], "producer")
    assert not e.aggregate([row], nodes)["passed"]


def test_historical_failure_and_original_rejection(tmp_path):
    """SCENARIO-VERIFY-8695-RETENTION: original line113 must run in its original source."""
    history = e.historical_failure()
    assert history["original_node_count"] == 77
    assert history["retained_child_sqlite_count"] > 0
    assert history["final_retained_reference_count"] == 0
    assert history["original_rejection_line_executed"] is False
    assert history["flagged_adversarial"] is True
    e.reject_original(tmp_path)
    assert e.REJECTION_SOURCE.read_text().splitlines()[112].strip() == "return False"


def test_missing_reference(tmp_path):
    """REQ-VERIFY-8695: absence and changed bytes cannot authenticate as valid inputs."""
    with pytest.raises(OSError):
        e.load(dict(path=str(tmp_path / "missing"), sha256="missing"))
    source = tmp_path / "source"
    source.write_text("{}")
    ref = e.reference(source)
    source.write_text("changed")
    with pytest.raises(ValueError):
        e.load(ref)
    with pytest.raises(ValueError):
        e.combine([ref], tmp_path / "bad")


def test_real_rejection_sqlite_and_consumer_separation(tmp_path, monkeypatch):
    """REQ-VERIFY-8695: an actual frozen-source execution qualifies only a producer."""
    scratch, source, nodes, receipt = real_child(
        tmp_path,
        "rejection",
        "from pathlib import Path; from carnot.reporting.v746_retained_evidence import reject_original; reject_original(Path(__file__).parent)",
    )
    row = e.retain(
        receipt, nodes, scratch, tmp_path / "retained", [e.reference(source)], "producer_control"
    )
    assert e.rejection_executed([row])
    value = e.load(row)
    value["role"] = "consumer"
    e.atomic_json(Path(row["path"]), value)
    consumer = e.reference(Path(row["path"]))
    assert not e.rejection_executed([consumer])
    value["role"] = "producer"
    e.atomic_json(Path(row["path"]), value)
    changed = e.reference(Path(row["path"]))
    Path(value["sqlite_references"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="producer_sqlite"):
        e.rejection_executed([changed])
    with pytest.raises(ValueError, match="changed_sqlite"):
        e.combine([changed], tmp_path / "bad-combine")
    monkeypatch.setattr(e, "REJECTION_PIN", "wrong")
    with pytest.raises(ValueError, match="source_drift"):
        e.reject_original(tmp_path)
    fake = tmp_path / "wrong_statement.py"
    fake.write_text("\n" * 112 + "pass\n")
    monkeypatch.setattr(e, "REJECTION_SOURCE", fake)
    monkeypatch.setattr(e, "REJECTION_PIN", e.sha256_file(fake))
    with pytest.raises(ValueError, match="statement_drift"):
        e.reject_original(tmp_path)


def test_incomplete_child_operands(tmp_path):
    """SCENARIO-VERIFY-8695-RETENTION: corruption, source drift and absent JUnit stay visible."""
    scratch, source, nodes, receipt = real_child(tmp_path, "incomplete")
    original = e.reference(source)
    source.write_text("source changed after collection")
    (scratch / ".coverage.corrupt").write_bytes(b"not SQLite")
    (scratch / "junit.xml").unlink()
    row = e.retain(receipt, nodes, scratch, tmp_path / "retained", [original], "producer")
    assert set(e.load(row)["errors"]) == {"not_sqlite", "source_drift"}
    assert not e.aggregate([row], nodes)["passed"]


def test_rehashed_junit_inventory(tmp_path):
    """REQ-VERIFY-8695: even internally hashed receipts must match actual JUnit outcomes."""
    scratch, source, nodes, receipt = real_child(tmp_path, "junit")
    row = e.retain(
        receipt, nodes, scratch, tmp_path / "retained", [e.reference(source)], "producer"
    )
    value = e.load(row)
    junit = Path(value["junit_reference"]["path"])
    junit.write_text('<testsuite><testcase name="different"/></testsuite>')
    value["junit_reference"] = e.reference(junit)
    e.atomic_json(Path(row["path"]), value)
    row = e.reference(Path(row["path"]))
    e.atomic_json(Path(row["path"]).with_suffix(".seal.json"), row)
    assert "junit_node_or_outcome_mismatch" in e.aggregate([row], nodes)["errors"]


def test_historical_changed_sqlite(tmp_path, monkeypatch):
    """REQ-VERIFY-8695: historical custody authenticates bytes before reproducing its failure."""
    shard = tmp_path / "original.sqlite"
    shard.write_bytes(b"original")
    ref = e.reference(shard)
    e.atomic_json(tmp_path / "checks" / "child.custody.json", dict(original_references=[ref]))
    validation = tmp_path / "validation.json"
    e.atomic_json(
        validation,
        dict(
            receipts=[
                dict(
                    name="original_contract_node_authority",
                    original_node_ids=[],
                    rejection_line_executed=False,
                ),
                dict(name="original_coverage_retention", original_references=[]),
            ]
        ),
    )
    history = tmp_path / "history.json"
    e.atomic_json(history, dict(validation_reference=e.reference(validation)))
    monkeypatch.setattr(e, "HISTORY", history)
    monkeypatch.setattr(e, "HISTORY_PIN", e.sha256_file(history))
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="historical_child_sqlite"):
        e.historical_failure()


def test_killed_child_and_missing_source(tmp_path):
    """REQ-VERIFY-8695: actual signal death and deleted source cannot qualify a child."""
    scratch = tmp_path / "killed"
    scratch.mkdir()
    source = scratch / "source.py"
    source.write_text("original source")
    ref = e.reference(source)
    receipt = child(
        "killed",
        [sys.executable, "-c", "import os,signal; os.kill(os.getpid(),signal.SIGKILL)"],
        tmp_path / "logs",
        deadline=10,
    )
    assert receipt["actual_exit"] == -9 and not receipt["normal_exit"]
    source.unlink()
    row = e.retain(receipt, ["killed::test"], scratch, tmp_path / "retained", [ref], "producer")
    assert not e.aggregate([row], ["killed::test"])["passed"]
    assert "missing_source" in e.load(row)["errors"]


def test_producer_aggregate_survives_unrelated_failure(tmp_path):
    """REQ-VERIFY-8695: broken consumer custody cannot erase intact producer execution."""
    from carnot.reporting import v746_contract_runner as runner

    scratch, source, nodes, command = real_child(tmp_path, "producer")
    row = e.retain(
        command, nodes, scratch, tmp_path / "retained", [e.reference(source)], "producer"
    )
    plan = [
        dict(role="producer", nodes=nodes, argv=command["argv"]),
        dict(role="consumer", nodes=["consumer::missing"], argv=[]),
    ]
    broken = dict(path=str(tmp_path / "missing-consumer"), sha256="missing")
    history = dict(original_node_ids=nodes, original_test_source_hashes=[])
    qualified = runner.qualify_producer(plan, [row, broken], history)
    assert qualified["aggregate"]["passed"]
    assert qualified["aggregate"]["node_ids"] == nodes
    assert not qualified["passed"]  # One constructed node cannot stand in for77 originals.
