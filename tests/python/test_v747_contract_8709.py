"""REQ-REPORT-8709: current binding preserves independently sealed original methods."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v747_contract_methods as m
from carnot.reporting import v747_contract_runner as runner
from carnot.reporting.v709_execution import child


def fixture(root):
    """Keep mutation controls private so original design bytes cannot be replaced."""
    design = root / m.DESIGN
    design.parent.mkdir(parents=True, exist_ok=True)
    text = (m.ROOT / m.DESIGN).read_text()
    design.write_text(text)
    tasks = m.parse_design(text, milestone=m.MILESTONE)[1]
    (root / m.ACTIVE).write_text(yaml.safe_dump(dict(milestone=m.MILESTONE, tasks=tasks)))
    return text, tasks


def test_authority_and_controls(tmp_path):
    """SCENARIO-REPORT-8709-AUTHORITY: full objects and independent digest all bind."""
    assert m.authority(m.ROOT)["ready"]
    assert not m.authority(tmp_path)["ready"]
    rows = m.mutations()
    assert len(rows) >= 20 and all(r["passed"] for r in rows)
    fixture(tmp_path)
    assert m.authority(tmp_path)["ready"]


def test_retained_methods():
    """REQ-REPORT-8709: authenticate the original execution rather than rerun77 nodes."""
    retained = m.retained()
    assert retained["passed"] and retained["qualification"]["original_node_count"] == 77
    p = m.methods()
    assert p["learning_authority_inputs_ready_score"] == 1
    original = m.previous.original.protocol()
    for key in ["objective_updates", "objective_audit", "installed_batching_protocol"]:
        assert p[key] == original[key]
    assert [len(c["node_ids"]) for c in p["original_partition"]["cohorts"]] == [39, 25]
    assert len(p["future_dependencies"]) == 13
    assert len(p["registered_v747_changes"]) == 5
    assert all(r["disqualified"] for r in p["immutable_diagnostics"])


@pytest.mark.parametrize("mode", ["missing", "changed", "field", "zero", "flagged", "terminal"])
def test_retained_rejection(tmp_path, monkeypatch, mode):
    """REQ-REPORT-8709: invalid retained evidence cannot grant readiness."""
    p = tmp_path / "retained.json"
    v = m.e.load(m.RETAINED)
    if mode == "field":
        v.pop("contract_ready_score")
    if mode == "zero":
        v["contract_ready_score"] = 0
    if mode == "flagged":
        v["flagged_adversarial"] = True
    if mode == "terminal":
        v["terminal_validation_sidecar_path"] = str(tmp_path / "missing-terminal")
    m.e.atomic_json(p, v)
    ref = m.e.reference(p)
    if mode == "missing":
        p.unlink()
    if mode == "changed":
        p.write_text("{}")
    monkeypatch.setattr(m, "RETAINED", ref)
    result = m.retained()
    assert not result["passed"] and result["gate_check_summary"]


def test_private_cli_and_cold_rejection(tmp_path):
    """SCENARIO-VERIFY-8709-REPLAY: run actual subprocess success and rejection paths."""
    output = tmp_path / "results" / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    for name, args, exit_code in [
        ("private", ["--private-control", "--output", str(output)], 0),
        ("cold", ["--cold-replay", str(output)], 0),
        ("date", ["--date", "20261101"], 2),
        ("unsafe", ["--private-control"], 2),
        ("error", ["--deliberate-error"], 7),
        ("absent", ["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        r = child(name, command + args, tmp_path / "logs", expected=exit_code, deadline=120)
        assert r["passed"], Path(r["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 0
    assert value["duration_s"] >= 0 and runner.replay(output)
    value["contract_ready_score"] = 1
    m.e.atomic_json(output, value)
    assert not runner.replay(output)


def test_collection_and_full_path(tmp_path, monkeypatch):
    """REQ-VERIFY-8709: current collection is disjoint; real failed coverage disqualifies."""
    private, raw = tmp_path / "private", tmp_path / "raw"
    private.mkdir()
    raw.mkdir()
    plan, receipt = runner.collect(private, raw)
    nodes = [n for s in plan for n in s["nodes"]]
    assert receipt["passed"] and len(nodes) == len(set(nodes))
    assert all(s["role"] != "producer" for s in plan)
    stream = tmp_path / "empty"
    stream.write_text("")
    original_child = runner.child
    monkeypatch.setattr(
        runner, "child", lambda *a, **k: dict(passed=False, stdout_path=str(stream))
    )
    with pytest.raises(ValueError, match="collection"):
        runner.collect(private, raw)
    monkeypatch.setattr(runner, "child", lambda *a, **k: dict(passed=True, stdout_path=str(stream)))
    stream.write_text(m.TESTS[0] + "::test_fake\n")
    negative_private = tmp_path / "negative-private"
    negative_private.mkdir()
    with pytest.raises(ValueError, match="missing_collection_nodes"):
        runner.collect(negative_private, raw)
    monkeypatch.setattr(runner, "child", original_child)
    tiny = tmp_path / "test_tiny.py"
    tiny.write_text("def test_actual():\n    assert True\n")

    def tiny_plan(private, raw):
        specs, collection = plan[:1], deepcopy(receipt)
        spec = deepcopy(specs[0])
        scratch = private / "child"
        scratch.mkdir()
        cfg = raw / "config.ini"
        cfg.write_text(f"[run]\nparallel=true\ndata_file={scratch}/.coverage\n")
        spec.update(
            name="tiny",
            scratch=str(scratch),
            nodes=[str(tiny) + "::test_actual"],
            config=m.e.reference(cfg),
        )
        spec["argv"] = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(cfg),
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
            *spec["nodes"],
            "--junitxml=" + str(scratch / "junit.xml"),
        ]
        return [spec], collection

    monkeypatch.setattr(runner, "collect", tiny_plan)
    preconditions = runner.old.preconditions
    monkeypatch.setattr(
        runner.old, "preconditions", lambda p: dict(preconditions(p), disk_backed=False)
    )
    output = tmp_path / "full" / (m.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and runner.replay(output)
    monkeypatch.setattr(m.e, "aggregate", lambda *a: dict(passed=False))
    assert not runner.replay(output)


def test_direct_replay_controls(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8709-REPLAY: missing and rehashed primitives reject independently."""
    assert runner.main(["--deliberate-error"]) == 7
    with pytest.raises(SystemExit):
        runner.main(["--private-control"])
    output = tmp_path / "results" / (m.NAME + ".json")
    assert runner.main(["--private-control", "--failed-child", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["flagged_adversarial"]
    assert runner.replay(output)
    original_sha = m.e.sha256_file
    first_source = value["code_config_hashes"][0]["path"]
    monkeypatch.setattr(
        m.e, "sha256_file", lambda p: "changed" if str(p) == first_source else original_sha(p)
    )
    assert not runner.replay(output)
    monkeypatch.setattr(m.e, "sha256_file", original_sha)
    original_methods = m.methods
    monkeypatch.setattr(
        m, "methods", lambda: dict(original_methods(), learning_authority_inputs_ready_score=0)
    )
    assert not runner.replay(output)
    monkeypatch.setattr(m, "methods", original_methods)
    ref = value["primitive_reference"]
    work = m.e.load(ref)
    work["contract"]["ready"] = False
    m.e.atomic_json(Path(ref["path"]), work)
    value["primitive_reference"] = m.e.reference(Path(ref["path"]))
    m.e.atomic_json(output, value)
    assert not runner.replay(output)
    seal = Path(value["terminal_validation_sidecar_path"]).parent / "independent_seal.json"
    m.e.atomic_json(seal, {k: value[k] for k in ["primitive_reference", "validation_reference"]})
    assert not runner.replay(output)
    Path(ref["path"]).unlink()
    assert not runner.replay(output)


@pytest.mark.parametrize("mode", ["source", "qualification", "seal"])
def test_retained_deep_identity(monkeypatch, mode):
    """REQ-REPORT-8709: retained execution requires source, command and independent seal identity."""
    load = m.e.load

    def changed(ref):
        value = load(ref)
        if mode == "source" and "node_ids" in value and "source_hashes" in value:
            value["source_hashes"][0]["original_sha256"] = "sha256:" + "0" * 64
        if mode == "qualification" and "producer_coverage_union" in value:
            value["producer_coverage_union"]["passed"] = False
        if mode == "seal" and ref["path"].endswith("independent_seal.json"):
            value["primitive_reference"]["sha256"] = "forged"
        return value

    monkeypatch.setattr(m.e, "load", changed)
    assert not m.retained()["passed"]


def test_missing_original_methods(monkeypatch):
    """REQ-REPORT-8709: unavailable original protocol blocks with a diagnostic, never a crash."""

    def missing():
        raise FileNotFoundError("original protocol absent")

    monkeypatch.setattr(m.previous, "methods", missing)
    result = m.methods()
    assert result["learning_authority_inputs_ready_score"] == 0
    assert result["gate_check_summary"]


def test_replay_rejects_historical_child_substitution(tmp_path):
    """SCENARIO-VERIFY-8709-REPLAY: rehashed historical rows cannot qualify current children."""
    output = tmp_path / "results" / (m.NAME + ".json")
    assert runner.main(["--private-control", "--output", str(output)]) == 0
    assert runner.replay(output)
    value = json.loads(output.read_bytes())
    validation = m.e.load(value["validation_reference"])
    validation["retained_child_rows"] = m.retained()["qualification"]["aggregate"][
        "retained_child_rows"
    ][:1]
    validation_path = Path(value["validation_reference"]["path"])
    m.e.atomic_json(validation_path, validation)
    value["validation_reference"] = m.e.reference(validation_path)
    seal = validation_path.parent / "independent_seal.json"
    m.e.atomic_json(seal, {k: value[k] for k in ["primitive_reference", "validation_reference"]})
    m.e.atomic_json(output, value)
    assert not runner.replay(output)
