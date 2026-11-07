"""REQ-REPORT-8776 / REQ-VERIFY-8776: independent administrative custody."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from carnot.reporting import v752_branch_methods as api
from carnot.reporting import v752_branch_report as report
from carnot.reporting import v752_branch_runner as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture
def contract(tmp_path):
    """REQ-REPORT-8776: tests mutate private authorities, never production outputs."""
    root = tmp_path / "authority"
    design = root / api.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes((api.ROOT / api.DESIGN).read_bytes())
    tasks = api.parse_design(design.read_text(), milestone=api.MILESTONE)[1]
    for name in [api.ACTIVE, api.STAGED]:
        (root / name).write_text(yaml.safe_dump(dict(milestone=api.MILESTONE, tasks=tasks)))
    return root, tasks


def test_contract_mutations_and_activation(contract, tmp_path):
    """SCENARIO-REPORT-8776-BRANCHES: all complete fields and encoding bind independently."""
    root, tasks = contract
    value = api.authority(root)
    assert value["ready"] and len(value["rows"]) == 13
    assert api.digest(tasks) == api.PIN
    controls = api.mutations(root, tmp_path / "mutations")
    assert len(controls) >= 13 * 16 and all(r["passed"] for r in controls)
    for task in tasks:
        task["prompt"] = api.encode(task["prompt"])
    (root / api.ACTIVE).write_text(yaml.safe_dump(dict(milestone=api.MILESTONE, tasks=tasks)))
    assert api.authority(root)["ready"]
    (root / api.STAGED).unlink()
    assert api.authority(root)["ready"]
    (root / api.ACTIVE).write_text("milestone: wrong\ntasks: []\n")
    assert not api.authority(root)["ready"]
    assert not api.authority(tmp_path / "absent")["ready"]


def test_protocol_isolation_and_causal_requirements(contract, tmp_path):
    """REQ-REPORT-8776: memory readiness is a protocol predicate, never a learner certificate."""
    _, tasks = contract
    protocols = api.protocols(tasks)
    assert set(protocols) == {"evidence", "memory", "capacity", "cost"}
    memory = protocols["memory"]
    assert api.memory_ready(memory)
    assert memory["observed_features"] == [
        "checked_predicates",
        "satisfied_local_predicates",
        "missing_predicates",
        "shared_variables",
        "observed_sign_conflicts",
        "current_successful_bindings",
    ]
    assert memory["architecture"] == dict(
        input_dim=9, hidden_dims=[4], energy_normalization="exp(-energy)"
    )
    assert memory["old_learning_union_authorized"] is False
    assert memory["splits"] == dict(fit=40, tune=20, stream=120, retention=120, canary=12)
    for field in ["inputs", "split_isolation", "causal_gates"]:
        changed = deepcopy(memory)
        changed[field] = []
        assert not api.memory_ready(changed)
    refs = api.seal_protocols(tmp_path, protocols)
    assert all(api.load(ref) == protocols[name] for name, ref in refs.items())


def test_runtime_once_absence_and_provenance(tmp_path, monkeypatch):
    """REQ-REPORT-8776: GPU visibility alone never admits the mandated canary."""
    calls = []
    monkeypatch.setattr(
        api.runtime_reader, "runtime", lambda *a: (calls.append(a), dict(ready=False))[1]
    )
    absent = api.runtime_input(tmp_path, tmp_path / "raw")
    assert len(calls) == 1 and not absent["ready"]
    assert absent["gate_check_summary"][0]["observed"] == "missing_file"
    path = tmp_path / "ops/cuda-runtime-repair-receipt.json"
    path.parent.mkdir()
    atomic_json(path, dict(ready=True))
    monkeypatch.setattr(api.runtime_reader, "runtime", lambda *a: dict(ready=True, changes=[]))
    assert not api.runtime_input(tmp_path, tmp_path / "raw")["ready"]
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"private provenance fixture; no model loaded")
    supplied = dict(
        ready=True,
        changes=[dict(field="libraries", changed=True)],
        mandated_model_provenance=dict(api.reference(model), hf_id="unsloth/Qwen3.8-27B-GGUF"),
    )
    monkeypatch.setattr(api.runtime_reader, "runtime", lambda *a: supplied)
    accepted = api.runtime_input(tmp_path, tmp_path / "raw")
    assert accepted["ready"] and len(accepted["references"]) == 2
    assert accepted["model_loads"] == accepted["device_probes"] == 0


def test_private_cli_replay_and_owned_failure(tmp_path):
    """SCENARIO-VERIFY-8776-REPLAY: real private children publish and reject tampering."""
    output = tmp_path / (api.NAME + ".json")
    command = [sys.executable, str(api.ROOT / api.CLI)]
    for name, args, expected in [
        ("private", ["--private-control", "--output", str(output)], 0),
        ("replay", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "missing")], 1),
        ("date", ["--date", "20260101"], 2),
        ("unsafe", ["--private-control"], 2),
    ]:
        receipt = child(name, command + args, tmp_path / "logs", expected=expected, deadline=120)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert all(value[k] == 0 for k in report.SCORES)
    assert value["MODEL_SPECS"] == [] and report.replay(output)
    saved = output.read_bytes()
    value["full_task_objects"][0]["prompt"] += "forged"
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not report.replay(output)
    output.write_bytes(saved)
    raw = Path(value["primitive_reference"]["path"]).parent
    seal = raw / "independent_seal.json"
    sealed = seal.read_bytes()
    seal.write_text("{}")
    assert not report.replay(output)
    seal.write_bytes(sealed)
    primitive = Path(value["primitive_reference"]["path"])
    original_primitive = primitive.read_bytes()
    primitive.write_text("{}")
    assert not report.replay(output)
    changed = dict(value, primitive_reference=api.reference(primitive))
    atomic_json(output, changed)
    atomic_json(seal, {k: changed[k] for k in ["primitive_reference", "validation_reference"]})
    assert not report.replay(output)
    primitive.write_bytes(original_primitive)
    seal.write_bytes(sealed)
    output.write_bytes(saved)
    work = api.load(value["primitive_reference"])
    protocol = Path(work["branch_protocol_refs"]["memory"]["path"])
    protocol_bytes = protocol.read_bytes()
    protocol.write_bytes(b"changed protocol operand")
    assert not report.replay(output)
    protocol.write_bytes(protocol_bytes)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api, "authority", lambda root: {})
        assert not report.replay(output)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(api, "protocols", lambda tasks: {})
        assert not report.replay(output)
    with pytest.MonkeyPatch.context() as patch:
        original_reference = api.reference
        patch.setattr(api, "reference", lambda p: dict(original_reference(p), sha256="forged"))
        assert not report.replay(output)


def test_independent_scores_and_failed_owned_validation(contract, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8776-BRANCHES: optional absences cannot erase qualified CPU protocols."""
    root, _ = contract
    work = api.freeze(root, tmp_path / "raw", "20261103")
    work["mutation_rows"] = [dict(passed=True)]
    validation = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True, node_ids=[]),
        coverage=dict(complete=True),
        raw_references=[],
        ended_at_utc=work["started_at_utc"],
    )
    output = tmp_path / (api.NAME + ".json")
    atomic_json(tmp_path / "raw/work.json", work)
    atomic_json(tmp_path / "raw/validation.json", validation)
    good = report.build(work, validation, output)
    assert good["verdict_class"] == "positive"
    assert good["contract_ready_score"] == good["memory_protocol_ready_score"] == 1
    assert good["runtime_repair_ready_score"] == good["arc_new_outcomes_available_score"] == 0
    assert good["learning_execution_authorized"] is False
    bad = deepcopy(validation)
    bad["receipts"][0]["passed"] = False
    failed = report.build(work, bad, output)
    assert failed["verdict_class"] == "disqualified" and all(failed[k] == 0 for k in report.SCORES)
    work["contract"]["ready"] = False
    assert report.build(work, validation, output)["verdict_class"] == "blocked"


def test_validation_uses_repaired_private_archives(tmp_path, monkeypatch):
    """REQ-VERIFY-8776: original suites retain assertions and use real result directories."""
    operand = tmp_path / "operand"
    operand.write_bytes(b"source custody")
    assert runner.checks.a.reference(operand) == api.reference(operand)

    def validate(raw, tracer, suites):
        assert len(suites) == 3
        assert all(Path(s).is_file() for s in suites)
        assert all("original" in s for s in suites)
        return dict(passed=True)

    monkeypatch.setattr(runner.checks, "validate", validate)
    result = runner.validate(tmp_path, None)
    assert result["passed"]
    assert len(result["archive_test_references"]) == 6


def test_main_dispatch_without_recursive_validation(tmp_path, monkeypatch):
    """REQ-VERIFY-8776: production dispatch has the same sealing and publication path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "Coverage", lambda **kw: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(
        runner,
        "validate",
        lambda raw, tracer: dict(
            receipts=[dict(passed=False)],
            union=dict(passed=False),
            coverage=dict(complete=False),
            raw_references=[],
            ended_at_utc=runner.now(),
        ),
    )
    output = tmp_path / (api.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
