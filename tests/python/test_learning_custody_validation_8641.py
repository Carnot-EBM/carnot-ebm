"""REQ-VERIFY-8641: node, statement and semantic unions need independent evidence."""

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from carnot.reporting import learning_custody_validation_8641 as v


def test_disjoint_partition_and_measurement_deadline():
    """REQ-VERIFY-8641: every collected node executes once in a bounded child."""
    nodes = [f"test::{i}" for i in range(13)]
    groups = v.partition(nodes)
    assert [n for group in groups for n in group] == nodes
    assert max(map(len, groups)) <= 6
    assert v.child_budget(0, 3599_000_000_000, "original") == 1
    assert v.child_budget(0, 3599_000_000_000, "owned") == 600
    assert v.child_budget(0, 4400_000_000_000, "owned") == 100
    assert v.child_budget(0, 4500_000_000_000, "owned") == 1
    with pytest.raises(TimeoutError):
        v.execute(dict(started_monotonic_ns=time.monotonic_ns() - 3601_000_000_000))
    with pytest.raises(KeyError):
        v.execute(
            dict(
                started_monotonic_ns=time.monotonic_ns() - 3601_000_000_000,
                original_execution_complete=True,
            )
        )


def test_attestation_rejects_unfinished_and_tampered():
    """SCENARIO-VERIFY-8641-REPLAY: booleans do not stand in for genuine execution."""
    work = dict(completed_execution=False)
    with pytest.raises(ValueError, match="unfinished"):
        v.attest(work)
    work["completed_execution"] = True
    with pytest.raises((ValueError, KeyError)):
        v.attest(work)


def test_execute_plan_and_independent_attestation(tmp_path, monkeypatch):
    """REQ-VERIFY-8641: synthetic checker operands never enter the measured original union."""
    from coverage import CoverageData
    from coverage.parser import PythonParser
    import xml.etree.ElementTree as xml

    e = v.e
    work = e.prepare(tmp_path / "raw")
    historical = e.read(work["statement_diagnosis"]["historical_work_reference"])
    nodes = historical["node_union"]["collected"] + [
        e.original.TESTS[0] + "::test_actual_removed_read_mutant",
        e.TESTS[0] + "::test_old_failure_is_executed",
    ]
    nodes = list(dict.fromkeys(nodes))
    old_rows = historical["deadline_case_rows"]
    for elapsed, row in zip((3599, 3600, 3601), old_rows, strict=True):
        e.atomic_json(tmp_path / "raw/deadline_cases" / f"case-{elapsed}" / "case.json", row)

    def fake_child(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        out, err = logs / (name + ".out"), logs / (name + ".err")
        out.write_text("\n".join(nodes) if name == "collection" else "")
        err.write_text("")
        receipt = dict(
            name=name,
            passed=True,
            timed_out=False,
            pid=123,
            exit_code=0,
            argv=argv,
            stdout_path=str(out),
            stderr_path=str(err),
            stdout_sha256=e.ref(out)["sha256"],
            stderr_sha256=e.ref(err)["sha256"],
        )
        e.atomic_json(logs / (name + ".receipt.json"), receipt)
        return receipt

    def fake_invoke(name, argv, root, coverage_file, **kwargs):
        config = (root / "coverage.ini").read_text()
        assert all(str(e.ROOT / n.split("::")[0]) in config for n in argv if n.startswith("tests/"))
        context = config.split("context=", 1)[1].splitlines()[0]
        data = CoverageData(basename=str(coverage_file))
        data.set_context(context)
        for path in [*e.OWNED, e.original.RUNNER]:
            parser = PythonParser(filename=str(e.ROOT / path))
            parser.parse_source()
            data.add_lines({str(e.ROOT / path): parser.statements})
        data.write()
        argument = next(a for a in argv if a.startswith("--junitxml="))
        suite = xml.Element("testsuite")
        for node in argv[argv.index(argument) + 1 :]:
            file, test = node.split("::", 1)
            xml.SubElement(suite, "testcase", classname=file[:-3].replace("/", "."), name=test)
        xml.ElementTree(suite).write(argument.split("=", 1)[1])
        receipt = fake_child(name, argv, root / "logs")
        receipt["coverage_file"] = str(coverage_file)
        return receipt

    monkeypatch.setattr(v, "child", fake_child)
    monkeypatch.setattr(v.c, "invoke", fake_invoke)
    monkeypatch.setattr(
        v.c, "controls", lambda root: [dict(mutation=str(i), rejected=True) for i in range(7)]
    )
    v.execute(work)
    v.attest(work)
    assert e.qualified(work)
    assert e.build(work, tmp_path / (e.NAME + ".json"))["learning_execution_ready_score"] == 1
    partial = deepcopy(work)
    partial["completed_execution"] = False
    partial["node_union"] = {}
    partial_value = e.build(partial, tmp_path / (e.NAME + ".json"))
    assert set(partial_value["executed_node_ids"]) == set(nodes)
    assert partial_value["learning_execution_ready_score"] == 0
    for attack in (
        "authority",
        "source",
        "union",
        "original27",
        "custody",
        "report",
        "original83",
        "owned",
        "semantic",
        "mapping",
        "collection",
        "receipts",
        "log",
    ):
        changed = deepcopy(work)
        if attack == "authority":
            changed["test_authority"] = {}
        elif attack == "source":
            path = str(e.ROOT / changed["original_test_files"][0])
            changed["test_authority"]["sources"][path]["sha256"] = "wrong"
        elif attack == "union":
            changed["node_union"]["collected"] = []
        elif attack == "original27":
            changed["original_nodes"] = []
        elif attack == "custody":
            changed["child_shard_manifest"][0]["context"] = "forged"
        elif attack == "report":
            changed["original_report"]["aggregate_reference"]["sha256"] = "wrong"
        elif attack == "original83":
            changed["original_report"]["files"][str(e.ROOT / e.original.RUNNER)]["statements"] = []
        elif attack == "owned":
            changed["owned_report"]["files"] = {}
        elif attack == "semantic":
            changed["semantic_mutation_rows"] = []
        elif attack == "collection":
            collection = e.read(changed["collection_reference"])
            collection["nodes"] = collection["nodes"][1:]
            e.atomic_json(tmp_path / "changed-collection.json", collection)
            changed["collection_reference"] = e.ref(tmp_path / "changed-collection.json")
        elif attack == "receipts":
            changed["receipts"] = changed["receipts"][:-1]
        elif attack == "log":
            changed["receipts"][-1]["stdout_sha256"] = "wrong"
        else:
            changed["current_mapping"] = {}
        with monkeypatch.context() as ctx:
            if attack == "source":
                ctx.setattr(v, "original_sources", lambda: changed["test_authority"])
            if attack in {"owned", "original83"}:
                ctx.setattr(
                    v.c,
                    "aggregate",
                    lambda selected, output: changed[
                        "original_report" if output.name == "cold_original" else "owned_report"
                    ],
                )
            with pytest.raises((ValueError, KeyError)):
                v.attest(changed)
        assert not e.qualified(changed)
