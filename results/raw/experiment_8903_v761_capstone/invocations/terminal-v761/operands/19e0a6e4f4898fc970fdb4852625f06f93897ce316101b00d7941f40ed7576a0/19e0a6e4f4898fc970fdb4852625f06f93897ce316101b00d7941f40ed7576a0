"""REQ-REPORT-8827; SCENARIO-VERIFY-8827-CUSTODY: retain falsifiable custody."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v756_branch_contract as c


def fixture(root, *, active=True):
    """An explicit staged authority lets tests run without activating a roadmap."""
    design = root / c.DESIGN
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_bytes((c.ROOT / c.DESIGN).read_bytes())
    tasks = c.parse_design(design.read_text(), milestone=c.MILESTONE)[1]
    planned = dict(milestone=c.MILESTONE, tasks=tasks)
    (root / c.STAGED).write_text(yaml.dump(planned, Dumper=yaml.CSafeDumper))
    (root / c.ACTIVE).write_text(
        yaml.dump(
            planned if active else dict(milestone="2026.11.755", tasks=[]), Dumper=yaml.CSafeDumper
        )
    )
    return root


def test_full_authority_and_every_field(tmp_path):
    """SCENARIO-REPORT-8827-CONTRACT: full objects include prompts and retirement."""
    root = fixture(tmp_path, active=False)
    good = c.authority(root)
    assert good["ready"] and good["state"] == "private_staged_v756"
    tasks = good["tasks"]
    assert len(tasks) == 13
    assert not any(r["artifact_field"] == "contract_ready_score" for r in c.edges(tasks))
    for i, task in enumerate(tasks):
        for key in task:
            changed = deepcopy(tasks)
            changed[i][key] = "forged"
            c.atomic_json(root / c.STAGED, dict(milestone=c.MILESTONE, tasks=changed))
            assert not c.authority(root)["ready"], (i, key)
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    fixture(root)
    assert c.authority(root)["state"] == "active_v756"
    (root / c.STAGED).unlink()
    assert c.authority(root)["ready"]
    (root / c.ACTIVE).unlink()
    assert not c.authority(root)["ready"]


def test_history_methods_and_optional_admission(tmp_path):
    """REQ-VERIFY-8827: admitted methods cannot inherit optional runtime failures."""
    h = c.history(c.ROOT)
    assert h["primary_count"] == 7 and h["structured_skip_count"] == 6
    assert len(h["rows"]) == 13
    assert len(c.methods(c.ROOT)) == 2
    optional = c.optional_admission(c.ROOT, tmp_path)
    assert optional["runtime"]["ready"] is False
    assert optional["arc"]["ready"] is False
    assert all(
        {"check", "upstream", "artifact_field", "expected", "observed", "path", "sha256"}
        <= r.keys()
        for r in optional["failures"]
    )
    assert c.arc_support([])["ready"] is False
    rows = [
        dict(game=g, arm=a, seed=s, selected=True, applied=True)
        for g in ["a", "b", "c"]
        for a in ["off", "on"]
        for s in range(5)
    ]
    assert c.arc_support(rows)["ready"] is True
    assert not c.arc_support(rows[:-1])["ready"]
    assert not c.arc_support(rows[:20])["ready"]


def test_build_replay_and_failures(tmp_path):
    """REQ-REPORT-8827: recompute counts and refuse rehashed candidate claims."""
    work = c.freeze(fixture(tmp_path / "tree"), tmp_path / "raw", "20261105")
    validation = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True, percent=100),
        original=dict(complete=True, rows=[]),
    )
    value = c.build(work, validation)
    path = tmp_path / "candidate.json"
    c.atomic_json(path, value)
    assert value["contract_ready_score"] == 1 and value["runtime_repair_ready_score"] == 0
    assert not c.replay(path)["passed"]
    validation["coverage"]["complete"] = False
    value = c.build(work, validation)
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    for field in ["completed_count", "contract_ready_score", "runtime_repair_ready_score"]:
        changed = deepcopy(value)
        changed[field] = 99
        changed["reproducibility_checksum"] = c.checksum(changed)
        c.atomic_json(path, changed)
        assert not c.replay(path)["passed"]
    c.atomic_json(path, value)
    Path(work["raw_reference"]["path"]).write_text("{}")
    assert not c.replay(path)["passed"]
    path.write_text("{")
    assert not c.replay(path)["passed"]
    validation["receipts"][0]["passed"] = False
    assert c.build(work, validation)["verdict_class"] == "disqualified"
    validation["receipts"][0]["passed"] = True
    validation["coverage"]["complete"] = True
    work["contract"]["ready"] = False
    assert c.build(work, validation)["verdict_class"] == "disqualified"


def test_private_cli_and_orchestration(tmp_path, monkeypatch):
    """REQ-VERIFY-8827: real scanners and cold CLI cannot accept altered custody."""
    from carnot.reporting import v756_branch_runner as runner
    from carnot.reporting import v756_branch_qualification as qualification
    import runpy

    assert runner.main(["--date", "bad"]) == 2
    work = c.freeze(fixture(tmp_path / "tree"), tmp_path / "raw", "20261105")
    validation = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=False, percent=0),
        original=dict(complete=True, rows=[]),
    )
    c.atomic_json(tmp_path / "raw/validation.json", validation)
    path = tmp_path / (c.NAME + ".json")
    c.atomic_json(path, c.build(work, validation))
    assert runner.terminal(path, tmp_path / "terminal")["passed"]
    assert runner.main(["--replay", str(path)]) == 0
    assert runner.main(["--replay", str(tmp_path / "missing.json")]) == 1
    monkeypatch.setattr(sys, "argv", [c.CLI, "--replay", str(path)])
    with pytest.raises(SystemExit) as end:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert end.value.code == 0
    monkeypatch.setattr(c, "freeze", lambda *args: deepcopy(work))
    monkeypatch.setattr(qualification, "qualify", lambda *args: validation["original"])

    def checked(raw, tracer, original):
        tracer.stop()
        tracer.save()
        return deepcopy(validation)

    monkeypatch.setattr(runner, "validate", checked)
    from coverage import Coverage

    parent = Coverage.current()
    with monkeypatch.context() as tracing:
        tracing.setenv("COVERAGE_CORE", "pytrace")
        tracer = Coverage(
            data_file=parent.config.data_file if parent else str(tmp_path / ".coverage"),
            data_suffix=True,
            include=[str(c.ROOT / p) for p in c.OWNED],
            config_file=False,
        )
        tracer.set_option("report:exclude_lines", [])
        tracer.start()
        assert runner.main(["--output", str(path), "--raw-dir", str(tmp_path / "raw")]) == 1
        assert (
            runner.main(
                ["--output", str(tmp_path / "invalid.json"), "--raw-dir", str(tmp_path / "raw")]
            )
            == 1
        )
        tracer.stop()
        tracer.save()


def test_original_fixture_and_actual_qualification(tmp_path):
    """REQ-VERIFY-8827: original assertions run unchanged against preserved V755."""
    from carnot.reporting import v756_branch_qualification as q
    from carnot.reporting import v755_branch_contract as old

    root = q.original_fixture(tmp_path / "authority")
    assert old.authority(root)["ready"]
    assert (root / old.DESIGN).read_bytes() == (c.ROOT / c.ARCHIVE).read_bytes()


@pytest.mark.parametrize("mode", ["pin", "task", "receipt"])
def test_historical_corruption(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8827-CUSTODY: rehashed historical forgeries remain rejected."""
    for name in c.HISTORY_PINS:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((c.ROOT / name).read_bytes())
    cap = json.loads((tmp_path / c.HISTORY).read_bytes())
    if mode == "pin":
        monkeypatch.setattr(c, "HISTORY_PINS", {c.ARCHIVE: "0" * 64})
    else:
        if mode == "task":
            cap["full_task_objects"][0]["title"] = "forged"
        else:
            cap["rows"][0]["primary_reference"]["sha256"] = "sha256:" + "0" * 64
        c.atomic_json(tmp_path / c.HISTORY, cap)
        monkeypatch.setattr(
            c, "HISTORY_PINS", {p: c.sha256_file(tmp_path / p)[7:] for p in c.HISTORY_PINS}
        )
    with pytest.raises(ValueError):
        c.history(tmp_path)


def test_missing_and_changed_operands(tmp_path, monkeypatch):
    """REQ-VERIFY-8827: absence is explicit and immutable copies cannot be replaced."""
    root = fixture(tmp_path / "tree")
    work = c.freeze(root, tmp_path / "raw", "20261105")
    target = Path(work["snapshots"][0]["path"])
    target.write_text("drift")
    with pytest.raises(ValueError, match="immutable_operand_drift"):
        c.freeze(root, tmp_path / "raw", "20261105")
    (root / c.ACTIVE).unlink()
    missing = c.freeze(root, tmp_path / "missing", "20261105")
    assert missing["operand_failures"][0]["observed"] == "missing_file"
    source = tmp_path / "results/raw" / c.NAME / "literature/2610.09541v1.html"
    source.parent.mkdir(parents=True)
    source.write_text("changed method")
    with pytest.raises(ValueError, match="primary_method_drift"):
        c.methods(tmp_path)
    altered = deepcopy(work)
    altered["contract"]["rows"][0]["metric"] = 999
    c.atomic_json(
        Path(work["raw_reference"]["path"]),
        {k: v for k, v in altered.items() if k != "raw_reference"},
    )
    altered["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    validation = dict(
        receipts=[], coverage=dict(complete=True), original=dict(complete=True, rows=[])
    )
    path = tmp_path / "changed.json"
    c.atomic_json(path, c.build(altered, validation))
    assert not c.replay(path)["passed"]


def test_fixture_hook_and_measured_rejection(tmp_path):
    """REQ-VERIFY-8827: missing measured statements fail without exclusions."""
    from types import SimpleNamespace
    from coverage import Coverage
    from carnot.reporting import v756_branch_qualification as q

    old = SimpleNamespace(__name__="test_v755_branch_contract_8814")
    other = SimpleNamespace(__name__="other")
    q.pytest_collection_modifyitems([SimpleNamespace(module=other), SimpleNamespace(module=old)])
    assert old.tree is q.original_fixture
    tracer = Coverage(data_file=str(tmp_path / ".coverage.input"), config_file=False)
    tracer.start()
    c.progress("real_coverage_control")
    tracer.stop()
    tracer.save()
    report = q.measured([tmp_path / ".coverage.input"], [c.OWNED[0]], tmp_path / "report")
    assert not report["complete"] and not report["files"][0]["excluded"]
    assert not q.verify_measured(report, [c.OWNED[0]])
    assert not q.verify_measured({}, [c.OWNED[0]])
    changed = deepcopy(report)
    changed["data_reference"]["sha256"] = "sha256:" + "0" * 64
    assert not q.verify_measured(changed, [c.OWNED[0]])


def test_validation_composition(tmp_path, monkeypatch):
    """REQ-VERIFY-8827: retained original checks cannot vanish during composition."""
    from carnot.reporting import v756_branch_runner as runner

    # This control checks composition only; qualification always runs real measured children.
    monkeypatch.setattr(runner.shared, "validate", lambda *args: dict(receipts=[]))
    original = dict(
        receipts=[dict(passed=False)],
        report_reference={},
        data_reference={},
        fixture_references=[],
        code_references=[],
    )
    result = runner.validate(tmp_path, None, original)
    assert result["receipts"] == original["receipts"]


def test_runtime_requires_exact_external_custody(tmp_path):
    """REQ-VERIFY-8827: an existing repair file cannot prove a mandated model ran."""
    runtime = dict(authority=dict(ready=True, changes=[dict(changed=True, field="libraries")]))
    assert not c.runtime_custody(tmp_path, runtime)["ready"]
    repair = tmp_path / "ops/cuda-runtime-repair-receipt.json"
    c.atomic_json(repair, {})
    assert not c.runtime_custody(tmp_path, runtime)["ready"]
    repair.write_text("{")
    assert not c.runtime_custody(tmp_path, runtime)["ready"]
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"explicit test fixture, not a model")
    build = tmp_path / "cuda-build"
    build.write_bytes(b"explicit test build")
    mandate = tmp_path / "mandate.json"
    expected = dict(
        hf_id="unsloth/Qwen3.8-27B-GGUF",
        revision="fixture-pinned-revision",
        sha256=c.sha256_file(model),
        cuda_build_sha256=c.sha256_file(build),
    )
    c.atomic_json(mandate, expected)
    validator_refs = {p: c.sha256_file(c.ROOT / p) for p in c.RUNTIME_VALIDATORS}
    c.atomic_json(
        repair,
        dict(
            mandated_model_contract_reference=c.reference(mandate),
            validator_code_hashes=validator_refs,
        ),
    )
    runtime["authority"]["transcript"] = dict(
        mandated_model_provenance=dict(expected, path=str(model)),
        cuda_build_reference=c.reference(build),
        offloaded_layers=1,
        actual_device="fixture-device",
        byte_copy_parity=True,
    )
    assert c.runtime_custody(tmp_path, runtime)["ready"]
    for key in expected:
        changed = deepcopy(runtime)
        changed["authority"]["transcript"]["mandated_model_provenance"][key] = "forged"
        assert not c.runtime_custody(tmp_path, changed)["ready"]
    mandate.write_text("{}")
    assert not c.runtime_custody(tmp_path, runtime)["ready"]


@pytest.mark.parametrize("mode", ["authority", "branch"])
def test_rehashed_work_and_manifest_semantics(tmp_path, mode):
    """SCENARIO-VERIFY-8827-CUSTODY: byte hashes cannot excuse changed task semantics."""
    work = c.freeze(fixture(tmp_path / "tree"), tmp_path / "raw", "20261105")
    if mode == "authority":
        work["contract"]["rows"][0]["metric"] = 999
    else:
        ref = work["manifests"][0]
        document = json.loads(Path(ref["path"]).read_bytes())
        document["task"]["title"] = "forged"
        c.atomic_json(Path(ref["path"]), document)
        work["manifests"][0] = c.reference(Path(ref["path"]))
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    work["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    candidate = tmp_path / "candidate.json"
    validation = dict(
        receipts=[], coverage=dict(complete=False), original=dict(complete=False, rows=[])
    )
    c.atomic_json(candidate, c.build(work, validation))
    assert not c.replay(candidate)["passed"]
