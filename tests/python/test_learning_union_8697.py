"""REQ-REPORT-8697, REQ-VERIFY-8697: exact producer evidence grants readiness."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import learning_union_8697 as e


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Read original authority once so tests cannot manufacture producer passes."""
    return e.prepare(tmp_path_factory.mktemp("union8697") / "raw", "20261101")


def test_original_authority(prepared):
    """SCENARIO-REPORT-8697-UNION: authenticate the original39 and registered25."""
    assert not prepared["input_failures"]
    assert len(prepared["partition"]["original_node_ids"]) == 64
    assert [len(c["node_ids"]) for c in prepared["partition"]["cohorts"]] == [39, 25]
    assert e.shared_identity(prepared)
    assert prepared["preconditions"]["private_disk_backed_scratch"]
    assert not prepared["preconditions"]["optional_hardware_required"]
    assert e.MODEL_SPECS == []
    assert prepared["methods"]["learning_authority_inputs_ready_score"] == 1
    assert prepared["core_children"]


@pytest.mark.parametrize("mode", ["missing", "field", "zero", "changed", "flagged"])
def test_gate_states(tmp_path, mode):
    """SCENARIO-REPORT-8697-UNION: never confuse absence, zero or changed bytes."""
    upstream = tmp_path / "upstream.json"
    if mode != "missing":
        value = {} if mode == "field" else dict(learning_core_ready_score=0)
        if mode in {"changed", "flagged"}:
            value = json.loads(e.UPSTREAM.read_bytes())
            value["flagged_adversarial"] = mode == "flagged"
        upstream.write_text(json.dumps(value))
    work = e.prepare(tmp_path / "raw", "20261101", upstream)
    states = {r["operand_state"] for r in work["input_failures"]}
    assert {"missing": "missing_file", "field": "missing_field", "zero": "observed_zero"}.get(
        mode, "changed_bytes"
    ) in states
    e.freeze(work)
    before = deepcopy(work)
    value = e.build(work, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["learning_execution_ready_score"] == 0 and work == before
    work["receipts"].append(dict(name="owned_failure", passed=False))
    assert e.build(work, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"


def test_union_rejects_duplicates_and_failed_nodes(prepared, tmp_path):
    """REQ-VERIFY-8697: node counts cannot stand in for source statements."""
    work = deepcopy(prepared)
    work["raw"] = str(tmp_path)
    e.recompute(work, tmp_path / "combine")
    assert len(work["outcomes"]) == 39
    assert len(work["statement_rows"]) == 83
    assert all(r["executed"] for r in work["statement_rows"])
    e.freeze(work)
    value = e.build(work, tmp_path / (e.NAME + ".json"))
    assert value["learning_execution_ready_score"] == 0
    assert not value["exact_union_receipt"]["passed"]
    work["core_children"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        e.recompute(work, tmp_path / "duplicate")


def test_shared_source_identity(prepared, monkeypatch):
    """REQ-VERIFY-8697: drift invalidates reuse even when other assertions pass."""
    monkeypatch.setattr(e.base, "check_bundle", lambda reference: False)
    assert not e.shared_identity(prepared)


def test_private_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-VERIFY-8697-REPLAY: actual CLI success and rejection retain exits."""
    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = e.c.supervise(
        "private",
        argv
        + [
            "--private-output",
            str(output),
            "--upstream",
            str(tmp_path / "missing"),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=180,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    original = json.loads(output.read_bytes())
    assert original["verdict_class"] == "blocked" and e.replay(output)
    for field in ("learning_execution_ready_score", "completed_count", "duration_s"):
        value = deepcopy(original)
        value[field] += 1
        e.atomic_json(output, value)
        assert not e.replay(output)
    work = e.c.read(original["primitive_reference"])
    work["duration_s"] += 1
    e.atomic_json(Path(original["primitive_reference"]["path"]), work)
    original["primitive_reference"] = e.c.ref(Path(original["primitive_reference"]["path"]))
    e.atomic_json(output, original)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
    for name, args, code in [
        ("invalid", ["--date", "bad"], 2),
        ("unsafe", ["--private-output", str(e.ROOT / "bad")], 2),
        ("owned", ["--deliberate-error"], 7),
        ("absent", ["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        assert e.c.supervise(name, argv + args, tmp_path / "logs", expected=code)["passed"]


def test_actual_child_and_immutable_receipts(prepared, tmp_path, monkeypatch):
    """REQ-VERIFY-8697: real SQLite, failed children and damaged evidence differ."""
    work = deepcopy(prepared)
    work.update(raw=str(tmp_path), core_children=[], children=[], receipts=[])
    source = e.c.ref(e.ROOT / "python/carnot/reporting/current_work_receipt.py")
    node = "tests/python/test_current_work_receipt.py::test_scenario_report_7395_current_keeps_sidecar_counts_external"
    assert e.instrument(work, "real", [node], [source], 90, "owned")["passed"]
    for mode in ("source", "operand", "config", "node", "combine"):
        changed = deepcopy(work)
        if mode == "source":
            monkeypatch.setattr(e, "shared_identity", lambda w: False)
        elif mode == "operand":
            changed["input_references"][0]["sha256"] = "wrong"
        elif mode == "config":
            changed["children"][0]["receipt"]["coverage_config_reference"]["sha256"] = "wrong"
        elif mode == "node":
            changed["children"][0]["node_ids"] = []
        else:
            monkeypatch.setattr(
                e.c, "combine_copies", lambda *a: dict(second_combine_invariance=dict(passed=False))
            )
        with pytest.raises(ValueError):
            e.recompute(changed, tmp_path / mode)
        monkeypatch.undo()
    e.recompute(work, tmp_path / "valid")
    e.freeze(work)
    output = tmp_path / (e.NAME + ".json")
    e.atomic_json(output, e.build(work, output))
    assert e.replay(output)
    Path(work["receipts"][0]["stdout_path"]).write_text("changed")
    assert not e.replay(output)
    assert e.instrument(work, "killed", [node], [source], 0.001, "owned")["timed_out"]
    assert work["child_errors"]


def test_dispatch_and_budget(prepared, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8697-UNION: preserve all25 nodes and complete child budgets."""
    work = deepcopy(prepared)
    work["raw"] = str(tmp_path)
    inventory = tmp_path / "inventory"
    inventory.write_text(
        "\n".join(work["partition"]["original_node_ids"] + [e.E2E020 + "::private"])
    )
    monkeypatch.setattr(
        e.c,
        "supervise",
        lambda *a, **k: dict(name="collection", passed=True, stdout_path=str(inventory)),
    )
    calls = []
    monkeypatch.setattr(e, "instrument", lambda *a: calls.append(a))
    e.execute(work)
    assert len(calls) == 5 and sum(len(r[2]) for r in calls[:4]) == 25
    assert [r[-1] for r in calls] == ["B", "B", "consumer", "consumer", "e2e"]
    work["started_monotonic_ns"] = 0
    e.execute(work)
    assert not work["launch_budget_rows"][-1]["admitted"]
    work["receipts"] = []
    e.validation(work)
    assert calls[-1][-1] == "owned" and len(work["receipts"]) == 4


def test_seal_boundaries(prepared, tmp_path, monkeypatch):
    """REQ-VERIFY-8697: source drift on either side invalidates producer reuse."""
    work = deepcopy(prepared)
    work["raw"] = str(tmp_path)
    monkeypatch.setattr(e, "shared_identity", lambda w: False)
    with pytest.raises(ValueError, match="before_child"):
        e.instrument(work, "before", [], [], 1, "B")
    answers = iter([True, False])
    monkeypatch.setattr(e, "shared_identity", lambda w: next(answers))
    monkeypatch.setattr(e.core, "instrument", lambda *a: dict(passed=True))
    e.instrument(work, "after", [], [], 1, "B")
    assert work["source_reuse_refusals"]


def test_registered_methods_and_resource_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8697: changed registration cannot inherit an original producer pass."""
    read = e.c.read

    def altered(reference):
        value = read(reference)
        if "inherited_operand_checks" in value:
            value["objective_updates"] = {}
        return value

    monkeypatch.setattr(e.c, "read", altered)
    with pytest.raises(ValueError, match="registered_method"):
        e.prepare(tmp_path / "method", "20261101")
    monkeypatch.undo()
    monkeypatch.setattr(e.shutil, "disk_usage", lambda *a: type("Disk", (), {"free": 0})())
    monkeypatch.setattr(e.base, "check_bundle", lambda r: False)
    work = e.prepare(tmp_path / "resources", "20261101")
    assert {"cpu_resources", "original_shared_source_seal"} <= {
        r["field"] for r in work["input_failures"]
    }


def test_positive_requires_every_obligation(prepared, tmp_path):
    """SCENARIO-REPORT-8697-UNION: a fabricated node count cannot bypass a gate."""
    work = deepcopy(prepared)
    work["raw"] = str(tmp_path)
    work.update(
        outcomes=[
            dict(node_id=n, outcome="passed") for n in work["partition"]["original_node_ids"]
        ],
        statement_rows=[dict(line=n, executed=True) for n in range(83)],
        owned_coverage=dict(complete=True),
        receipts=[dict(name=n, passed=True) for n in e.REQUIRED],
    )
    e.freeze(work)
    value = e.build(work, tmp_path / (e.NAME + ".json"))
    assert value["learning_execution_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["generalized_learning_benefit_score"] == 0
    for field, replacement in [
        ("outcomes", work["outcomes"][:-1]),
        ("statement_rows", []),
        ("owned_coverage", dict(complete=False)),
    ]:
        changed = deepcopy(work)
        changed[field] = replacement
        assert (
            e.build(changed, tmp_path / (e.NAME + ".json"))["learning_execution_ready_score"] == 0
        )
    work["outcomes"][0]["outcome"] = "failed"
    assert e.build(work, tmp_path / (e.NAME + ".json"))["failed_count"] == 1


def test_run_dispatch(prepared, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8697-REPLAY: production executes then validates before publication."""
    work = deepcopy(prepared)
    work.update(raw=str(tmp_path / "raw"), core_children=[])
    calls = []
    monkeypatch.setattr(e, "prepare", lambda *a: work)
    monkeypatch.setattr(e, "execute", lambda *a: calls.append("execute"))
    monkeypatch.setattr(e, "validation", lambda *a: calls.append("validate"))
    assert e.run(tmp_path / (e.NAME + ".json"), "20261101") == 0
    assert calls == ["execute", "validate"]
    monkeypatch.setattr(e, "run", lambda *a, **kw: 0)
    assert e.main([]) == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--date", "20260230"],
        ["--date", "20261102"],
        ["--private-control"],
        ["--upstream", "/var/tmp/missing-upstream"],
    ],
)
def test_cli_guards(args):
    """REQ-VERIFY-8697: date and injected inputs are checked before creating evidence."""
    assert e.main(args) == 2


@pytest.mark.parametrize("present", [False, True])
def test_original_operands_checked_before_measurement(tmp_path, monkeypatch, present):
    """SCENARIO-REPORT-8697-UNION: absent and changed inputs block before launch."""
    operand = tmp_path / "operand.json"
    if present:
        operand.write_text("{}")
    methods = e.authority.methods()
    methods["references"].append(dict(path=str(operand), sha256="sha256:original"))
    monkeypatch.setattr(e.authority, "methods", lambda: methods)
    work = e.prepare(tmp_path / "raw", "20261101")
    rows = [r for r in work["input_failures"] if r["path"] == str(operand)]
    assert rows and rows[0]["operand_state"] == ("changed_bytes" if present else "missing_file")
    e.recompute(work, tmp_path / "combine")
    e.freeze(work)
    assert e.build(work, tmp_path / (e.NAME + ".json"))["verdict_class"] == "blocked"
