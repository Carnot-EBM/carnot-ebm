"""REQ-REPORT-8654, REQ-VERIFY-8654: preserve original bounded learning obligations."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from coverage import CoverageData

from carnot.reporting import learning_core_8654 as e
from carnot.reporting import learning_custody_8641 as custody


def test_registered_module_groups():
    """SCENARIO-REPORT-8654-PARTITION: fixtures share one child per original module."""
    p = e.methods.protocol()
    partition = e.methods.read(p["cohort_partition_reference"])
    groups = e.groups(partition, partition["original_node_ids"])
    assert len(groups) == 3
    assert sum(len(g["node_ids"]) for g in groups) == 39
    assert all({n.split("::")[0] for n in g["node_ids"]} == {g["module"]} for g in groups)
    assert [g["child_budget_s"] for g in groups] == [573, 1672, 292]
    for mode in ("duplicate", "omit", "foreign", "cross_module", "group_omit"):
        changed = deepcopy(partition)
        collected = list(changed["original_node_ids"])
        if mode == "duplicate":
            collected.append(collected[0])
        elif mode == "omit":
            collected.pop()
        elif mode == "foreign":
            collected[-1] = "tests/python/foreign.py::test"
        elif mode == "cross_module":
            changed["cohorts"][0]["groups"][0]["module"] = "wrong.py"
        else:
            changed["cohorts"][0]["groups"][0]["node_ids"].pop()
        with pytest.raises(ValueError, match="partition"):
            e.groups(changed, collected)


def test_launch_budget():
    """SCENARIO-REPORT-8654-PARTITION: admit a full child before the measurement boundary."""
    group = dict(module="original.py", child_budget_s=573)
    assert e.admission(group, 0, 3027 * 10**9)["admitted"]
    assert not e.admission(group, 0, 3028 * 10**9)["admitted"]
    assert not e.admission(group, 0, 4800 * 10**9)["admitted"]


@pytest.mark.parametrize("mode", ["valid", "changed_source", "missing", "deleted", "rehashed"])
def test_immutable_second_combine(tmp_path, mode):
    """SCENARIO-VERIFY-8654-REPLAY: real SQLite custody rejects repaired evidence."""
    source = tmp_path / "original.py"
    source.write_text("answer = 1\n")
    sources = [custody.ref(source)]
    namespace = tmp_path / "child"
    namespace.mkdir()
    for suffix in ("", ".worker"):
        data = CoverageData(basename=str(namespace / (".coverage" + suffix)))
        data.set_context("original-module")
        data.add_lines({str(source): {1}})
        data.write()
    out, err = namespace / "out", namespace / "err"
    out.write_text("passed")
    err.write_text("")
    receipt = dict(passed=True, timed_out=False, exit_code=0)
    for stream, path in [("stdout", out), ("stderr", err)]:
        receipt[stream + "_path"] = str(path)
        receipt[stream + "_sha256"] = custody.ref(path)["sha256"]
    manifest = custody.seal_child(namespace, sources, receipt, ["original.py::test"], "A")
    shard = Path(manifest["files"][0]["reference"]["path"])
    if mode == "changed_source":
        source.write_text("answer = 2\n")
    elif mode in {"missing", "deleted"}:
        shard.unlink()
    elif mode == "rehashed":
        shard.chmod(0o644)
        data = CoverageData(basename=str(shard))
        data.read()
        data.add_lines({str(source): {2}})
        data.write()
        manifest["files"][0]["reference"] = custody.ref(shard)
    if mode != "valid":
        with pytest.raises((ValueError, OSError)):
            e.attest([manifest], sources)
    else:
        assert e.attest([manifest], sources) == {str(source): [1]}
        combined = custody.combine_copies([manifest], tmp_path / "combine")
        assert combined["second_combine_invariance"]["passed"]
        assert e.attest([manifest], sources) == {str(source): [1]}


def test_history_diagnosis():
    """REQ-REPORT-8654: a boundary timeout never becomes a numerical failure."""
    partition = e.methods.read(e.methods.protocol()["cohort_partition_reference"])
    row = e.diagnosis(partition)
    assert row["exit_code"] == -9
    assert row["duration_s"] == 62.074893612
    assert row["nominal_deadline_rows_passed"] == 3
    assert row["cause"] == "measurement_budget_exhaustion"
    assert row["missing_mutations"]


def test_preconditions_and_external_states(tmp_path):
    """REQ-REPORT-8654: absent operands stay distinct from an authenticated zero."""
    for name, value, state in [
        ("missing", None, "missing_file"),
        ("field", {}, "missing_field"),
        ("zero", {"methods_ready_score": 0}, "observed_zero"),
    ]:
        path = tmp_path / name
        if value is not None:
            path.write_text(json.dumps(value))
        work = e.prepare(tmp_path / (name + "-raw"), "20261019", path)
        assert work["input_failures"][0]["operand_state"] == state
        assert work["preconditions"]["private_disk_backed_scratch"]
        assert work["preconditions"]["current_llm_calls"] == 0


def test_prepare_authentication_and_resource_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8654-REPLAY: resource, publication and operand drift fail closed."""
    original_ref = e.c.ref
    valid = e.prepare(tmp_path / "valid", "20261019")
    assert valid["numerical_inference"]["finite"]
    assert valid["grouping_rows"] and not valid["input_failures"]
    assert valid["mechanics_fixture"] in valid["input_references"]
    assert e.MODEL_SPECS == []
    renamed = tmp_path / "upstream.json"
    renamed.write_bytes(e.UPSTREAM.read_bytes())
    with pytest.raises(ValueError, match="upstream_identity"):
        e.prepare(tmp_path / "renamed", "20261019", renamed)
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    with pytest.raises(ValueError, match="upstream_publication"):
        e.prepare(tmp_path / "publication", "20261019")
    monkeypatch.undo()
    monkeypatch.setattr(
        e.c,
        "ref",
        lambda p: (
            dict(original_ref(p), sha256="wrong")
            if p.name == "trainer_operands.json"
            else original_ref(p)
        ),
    )
    with pytest.raises(ValueError, match="input_identity"):
        e.prepare(tmp_path / "operands", "20261019")
    monkeypatch.undo()
    monkeypatch.setattr(e.shutil, "disk_usage", lambda *a: type("Disk", (), {"free": 0})())
    assert e.prepare(tmp_path / "disk", "20261019")["input_failures"]


def test_diagnosis_rejects_changed_history(monkeypatch):
    """REQ-REPORT-8654: independently sealed timeout bytes remain original authority."""
    partition = e.methods.read(e.methods.protocol()["cohort_partition_reference"])
    original = e.c.read
    for mode in ("log", "duration"):

        def changed(reference):
            work = original(reference)
            receipt = next(r for r in work["receipts"] if r["name"] == "original25")
            receipt["stdout_sha256" if mode == "log" else "duration_s"] = "changed"
            return work

        monkeypatch.setattr(e.c, "read", changed)
        with pytest.raises(ValueError, match="historical_timeout"):
            e.diagnosis(partition)
    monkeypatch.undo()


def test_execute_admission_and_collection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8654-PARTITION: bounded grouping launches each module once."""
    p = e.methods.protocol()
    partition = e.methods.read(p["cohort_partition_reference"])
    work = dict(
        raw=str(tmp_path),
        partition=partition,
        started_monotonic_ns=e.time.monotonic_ns(),
        receipts=[],
        launch_budget_rows=[],
    )
    log = tmp_path / "collected"
    log.write_text("\n".join(partition["original_node_ids"]))
    commands = []

    def collected(*args, **kwargs):
        commands.append(args[1])
        return dict(stdout_path=str(log), passed=True)

    monkeypatch.setattr(e.c, "supervise", collected)
    calls = []
    monkeypatch.setattr(e, "instrument", lambda *a: calls.append(a))
    e.execute(work)
    assert len(calls) == 3
    assert commands[0][-64:] == partition["original_node_ids"]
    assert work["collected"] == partition["original_node_ids"]
    assert all(row["admitted"] for row in work["launch_budget_rows"])
    work["started_monotonic_ns"] = 0
    e.execute(work)
    assert not work["launch_budget_rows"][-1]["admitted"]


def test_transitive_operands(tmp_path):
    """SCENARIO-VERIFY-8654-REPLAY: nested immutable operands remain mandatory on replay."""
    leaf = tmp_path / "leaf.json"
    leaf.write_text('{"value": 1}')
    wrapper = tmp_path / "wrapper.json"
    wrapper.write_text(json.dumps(dict(child=e.c.ref(leaf))))
    refs = e.operand_references([dict(configuration=e.c.ref(wrapper))])
    assert e.c.ref(wrapper) in refs and e.c.ref(leaf) in refs
    leaf.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="operand_identity"):
        e.operand_references([e.c.ref(wrapper)])


def test_repeated_operand_conflict(tmp_path):
    """REQ-VERIFY-8654: one path cannot authenticate contradictory operand identities."""
    path = tmp_path / "leaf.json"
    path.write_text("{}")
    reference = e.c.ref(path)
    assert e.operand_references([reference, reference]) == [reference]
    with pytest.raises(ValueError, match="operand_identity"):
        e.operand_references([dict(reference, sha256="wrong"), reference])
