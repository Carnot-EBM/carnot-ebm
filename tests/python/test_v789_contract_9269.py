"""REQ-REPORT-9269-V789 / SCENARIO-REPORT-9269-V789: compact isolated replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v789_contract as c
from carnot.reporting import v789_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Authenticate original producers once; fixtures never perform model work."""
    return c.measure(e.ROOT, tmp_path_factory.mktemp("contract9269"), "20261121")


def test_compactness_guard():
    """SCENARIO-REPORT-9269-V789-CONTROLS: oversized primaries fail before publication."""
    c.compact_guard({})
    with pytest.raises(ValueError, match="primary_size"):
        c.compact_guard(dict(bloat="x" * c.MAX_PRIMARY_BYTES))


def test_owned_coverage_list_mismatch(work, monkeypatch):
    """SCENARIO-REPORT-9269-V789-CONTROLS: exact coverage ownership is mandatory."""
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    coverage = dict(files=[dict(path=p) for p in e.OWNED])
    assert c.receipts_pass(work, dict(owned_statement_coverage=coverage))
    for paths in [[], e.OWNED[:-1], [*e.OWNED, "extra.py"]]:
        bad = dict(work, scope=dict(work["scope"], owned_changed_paths=paths))
        assert not c.receipts_pass(bad, dict(owned_statement_coverage=coverage))
    bad = c.build(work, dict(owned_statement_coverage=dict(files=[])))
    assert not bad["required_checks_passed"]
    assert bad["contract_ready_score"] == bad["replay_authority_ready_score"] == 0


def test_authentic_contract_and_compact_tables(work):
    """SCENARIO-REPORT-9269-V789: eight shards and fourteen complete ordered objects."""
    inspected = work["inspection"]
    assert inspected["contract"]["ready"], [r for r in inspected["checks"] if not r["passed"]]
    assert all(r["passed"] for r in inspected["checks"])
    assert inspected["source"]["roles"] == dict(TRAIN=150, CAL=50, TEST=100)
    assert inspected["source"]["answer_count"] == 900
    assert len(inspected["historical_input_roots"]["suites"]) == 9
    value = c.build(work, {})
    assert len(value["full_task_objects"]) == 14
    assert len(value["consumer_shard_rows"]) == 8
    assert value["MODEL_SPECS"] == []
    assert all(n == 0 for n in value["model_invocation_counts"].values())
    assert value["claim_scope"]["independent_generalization_score"] == 0
    for key in e.TABLES:
        table = value[key]
        assert e.ref(Path(table["path"]))["sha256"] == table["sha256"]
        assert table["count"] > 0
    assert len(json.dumps(value).encode()) < c.MAX_PRIMARY_BYTES
    assert value["preserved_v788"]["9256"]["corpus_scan_ready_score"] == 0
    assert [s[0] for s in c.SHARDS] == [
        "direct_consumers_core",
        *[f"direct_consumers_v{v}" for v in range(782, 789)],
    ]
    for command in work["scope"]["commands"][:9]:
        assert "--unset=PYTHONPATH" in command["argv"]
    obligations = json.loads(Path(inspected["validation_obligations"]["path"]).read_bytes())
    assert len(obligations["node_bindings"]) == 674
    assert {
        "tests/python/test_causal_validation_9181.py",
        "tests/python/test_v788_corpus_prefilter_9256.py",
    }.issubset(obligations["affected_v789_paths"])


def test_partitioned_supervisor_and_deadline(work, monkeypatch):
    """REQ-REPORT-9269-V789: measured receives the identical owned list."""
    seen = []
    measured = []

    def child(name, *args, **kwargs):
        seen.append(name)
        return dict(name=name, passed=True, exit_code=0, duration_s=0.1)

    def coverage(paths, owned, output):
        measured.append(owned)
        return dict(complete=True)

    monkeypatch.setattr(c.legacy.lifecycle, "child", child)
    monkeypatch.setattr(c.legacy.lifecycle, "measured", coverage)
    validation = c.qualify(work)
    assert measured[0] is work["scope"]["owned_changed_paths"]
    assert set(seen) == {*c.CHECKS, "global_python_suite"}
    assert len(validation["checks"]) == 13
    deadlines = {}

    def bounded_child(name, *args, **kwargs):
        deadlines[name] = kwargs["deadline"]
        return {}

    def invoke(args):
        for name in ["owned_tests", "cold_replay"]:
            c.legacy.lifecycle.child(name, [], Path("/tmp"), deadline=9999)
        return 0

    monkeypatch.setattr(c.legacy.lifecycle, "child", bounded_child)
    monkeypatch.setattr(c, "_main", invoke)
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    monkeypatch.delenv("CARNOT_V789_DEADLINE", raising=False)
    assert c.main(["--date", "20261121"]) == 0
    assert deadlines == dict(owned_tests=3300, cold_replay=4500)


def child(directory, *args):
    """Cold processes start without checkout environment imports and retain coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9269] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=directory, env=env, capture_output=True, text=True, timeout=300
    )
    print("[test9269] child_after", result.returncode, flush=True)
    return result


def test_private_publication_tamper_and_interleaved_replay(work, tmp_path):
    """SCENARIO-REPORT-9269-V789-CONTROLS: private cold replay rejects forged evidence."""
    source = tmp_path / "work.json"
    output = tmp_path / (e.NAME + ".json")
    atomic_json(source, work)
    result = child(tmp_path, "--from-work", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    assert c.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    reference = value["validation_obligations"]
    table = Path(reference["path"])
    original = table.read_bytes()
    table.write_text("{}\n")
    with pytest.raises(ValueError, match="sidecar"):
        c.replay(output)
    table.write_bytes(original)
    value["contract_ready_score"] = 99
    value["reproducibility_checksum"] = c.legacy.checksum(value)
    atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    assert child(tmp_path, "--date", "bad").returncode == 1
    assert child(tmp_path, "--from-work", source).returncode == 1
    assert (
        child(
            tmp_path, "--checks-only", "--root", tmp_path / "absent", "--output", output
        ).returncode
        == 0
    )


def test_external_absence_and_bad_task(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9269-V789-CONTROLS: external absence is terminal blocked."""
    monkeypatch.setattr(e, "HISTORICAL_BASE", tmp_path / "history", raising=False)
    result = e.inspect(tmp_path / "absent")
    assert not result["replay_ready"]
    assert any(r["observed"] == "missing_file" for r in result["checks"])
    root = tmp_path / "bad"
    (root / e.DESIGN).parent.mkdir(parents=True)
    (root / e.DESIGN).write_text("## Full machine-readable contract\n```yaml\n[]\n```\n")
    assert not e.authority(root)["ready"]


def test_complete_task_mutations(tmp_path, monkeypatch):
    """REQ-REPORT-9269-V789: hashes alone cannot bless reordered or edited objects."""
    import yaml

    root = tmp_path / "authority"
    for name in [e.DESIGN, e.ACTIVE, *e.COLLISION_INPUTS]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
    frozen = deepcopy(e.FROZEN)
    monkeypatch.setattr(e, "FROZEN", frozen)
    queue = yaml.safe_load((root / e.ACTIVE).read_bytes())
    for mutation in ["order", "prompt", "missing", "id", "container", "count"]:
        bad = deepcopy(queue)
        if mutation == "order":
            bad["tasks"][0], bad["tasks"][1] = bad["tasks"][1], bad["tasks"][0]
        elif mutation == "prompt":
            bad["tasks"][0]["prompt"] += "changed"
        elif mutation == "missing":
            del bad["tasks"][0]["prompt"]
        elif mutation == "id":
            bad["tasks"][0]["id"] = 42
        elif mutation == "container":
            bad["tasks"][0] = None
        else:
            bad["tasks"].pop()
        (root / e.ACTIVE).write_text(yaml.safe_dump(bad))
        frozen["pins"][e.ACTIVE] = e.ref(root / e.ACTIVE)["sha256"]
        assert not e.authority(root)["ready"]


def test_external_drift_is_blocked(tmp_path, monkeypatch):
    """REQ-REPORT-9269-V789: unavailable original authority keeps its named operand."""
    monkeypatch.setattr(e, "HISTORICAL_BASE", tmp_path / "history", raising=False)

    def unavailable(*args, **kwargs):
        raise ValueError("historical_input_drift:" + str(tmp_path / "missing.json"))

    monkeypatch.setattr(e, "original_root", unavailable)
    assert not e.inspect(e.ROOT)["replay_ready"]

    def absent_field(*args, **kwargs):
        raise KeyError("named_field")

    monkeypatch.setattr(e, "original_root", absent_field)
    result = e.inspect(e.ROOT)
    assert any(r["observed"] == "missing_field" for r in result["checks"])


def test_all_qualified_scores_from_rows(work, monkeypatch):
    """REQ-REPORT-9269-V789: administrative readiness cannot claim generalization."""
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    value = c.build(
        work, dict(owned_statement_coverage=dict(files=[dict(path=p) for p in e.OWNED]))
    )
    assert value["required_checks_passed"]
    assert value["contract_ready_score"] == value["replay_authority_ready_score"] == 1
    assert (
        value["source_ready_score"]
        == value["static_methods_ready_score"]
        == value["online_methods_ready_score"]
        == 1
    )
    assert value["verdict_class"] == "null"
    assert all(r["numerator"] == r["denominator"] == 1 for r in value["rows"])


def test_replay_size_and_count_controls(tmp_path):
    """SCENARIO-REPORT-9269-V789-CONTROLS: reject oversize or mismatched table counts."""
    output = tmp_path / "large.json"
    output.write_bytes(b" " * c.MAX_PRIMARY_BYTES)
    with pytest.raises(ValueError, match="primary_size"):
        c.replay(output)
    sidecar = tmp_path / "table.json"
    atomic_json(sidecar, [])
    atomic_json(output, {e.TABLES[0]: dict(e.ref(sidecar), count=1)})
    with pytest.raises(ValueError, match="compact_sidecar"):
        c.replay(output)


def test_interleaved_generations(work, tmp_path):
    """SCENARIO-REPORT-9269-V789-CONTROLS: private replay releases large legacy heaps on exit."""
    source = tmp_path / "interleaved.py"
    source.write_text(
        "import sys, json\nfrom pathlib import Path\n"
        + "sys.path[:0] = "
        + repr([str(e.ROOT / "python"), str(e.ROOT)])
        + "\n"
        + "from carnot.reporting import v789_contract as c\n"
        + "from carnot.reporting.current_work_receipt import atomic_json\n"
        + "from carnot.reporting.historical_contract_inputs import original_root\n"
        + "root = Path("
        + repr(str(tmp_path))
        + ")\n"
        + "work = json.loads(Path("
        + repr(str(Path(work["raw"]) / "work.json"))
        + ").read_bytes())\n"
        + "modules = [c.previous.previous, c.previous, c]\noutputs = []\n"
        + "for module, generation in zip(modules, [787, 788, 789]):\n"
        + "    current = work if generation == 789 else module.measure(original_root(root / str(generation), generation), root / f'raw{generation}', '20261120')\n"
        + "    path = root / f'v{generation}.json'\n"
        + "    atomic_json(path, module.build(current, {}))\n    outputs.append(path)\n"
        + "for index in [2, 0, 1, 2]:\n    assert modules[index].replay(outputs[index])['passed']\n"
    )
    python = (
        [sys.executable, "-m", "coverage", "run"]
        if os.environ.get("COVERAGE_RCFILE")
        else [sys.executable, "-u"]
    )
    receipt = c.legacy.lifecycle.child(
        "interleaved_generations",
        ["env", "--unset=PYTHONPATH", "--chdir=" + str(tmp_path), *python, str(source)],
        tmp_path / "replay_logs",
        deadline=1200,
        heartbeat=30,
    )
    assert receipt["passed"], (
        Path(receipt["stdout_path"]).read_text() + Path(receipt["stderr_path"]).read_text()
    )


def test_rehashed_false_protocols_rejected(work, tmp_path):
    """REQ-REPORT-9269-V789: new hashes never turn false protocols into original owners."""
    for protocol in [
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
        "selective_deferral_protocol",
        "conformal_risk_protocol",
    ]:
        changed = deepcopy(work)
        sidecar = tmp_path / (protocol + ".json")
        atomic_json(sidecar, {"fabricated": True})
        changed["protocol_sidecars"][protocol + "_reference"] = e.ref(sidecar)
        source = tmp_path / (protocol + "_work.json")
        atomic_json(source, changed)
        value = c.build(work, {})
        value["work_reference"] = e.ref(source)
        value["reproducibility_checksum"] = c.legacy.checksum(value)
        output = tmp_path / (protocol + "_primary.json")
        atomic_json(output, value)
        result = child(tmp_path, "--cold-replay", output)
        assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("machine", ["[bad]", "{}", "tasks: []"])
def test_invalid_machine_contract(tmp_path, machine):
    """REQ-REPORT-9269-V789: malformed fenced YAML cannot become an execution contract."""
    target = tmp_path / e.DESIGN
    target.parent.mkdir(parents=True)
    target.write_text("## Full machine-readable contract\n```yaml\n" + machine + "\n```\n")
    assert not e.authority(tmp_path)["ready"]
