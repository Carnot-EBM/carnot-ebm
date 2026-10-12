"""REQ-REPORT-8443 and REQ-VERIFY-8443: independent boundaries precede science."""

from copy import deepcopy
import json
import os
import subprocess
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v727_contract_methods as e
from carnot.reporting import v727_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture
def full_root(tmp_path):
    """SCENARIO-REPORT-8443-AUTHORITY: independent fixture preserves complete objects."""
    root = tmp_path / "authority"
    plan = yaml.safe_load((e.ROOT / e.ACTIVE).read_bytes())
    text = "## Exact task contract\n"
    for i, t in enumerate(plan["tasks"], 1):
        text += f"| {i} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |\n"
    text += f"Canonical full-task SHA-256: `{e.TASKS_PIN}`\n"
    text += "<!-- V727_TASK_CONTRACT_START -->\n```json\n" + json.dumps(plan) + "\n```\n"
    for name, data in [(e.DESIGN, text.encode()), (e.ACTIVE, (e.ROOT / e.ACTIVE).read_bytes())]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    target = root / e.h.DESIGN
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((e.ROOT / e.h.DESIGN).read_bytes())
    (root / "results").symlink_to(e.ROOT / "results", target_is_directory=True)
    return root


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "drop",
        "reorder",
        "prompt",
        "field",
        "model",
        "routing",
        "retire",
        "stage",
        "table",
        "digest",
        "missing",
        "prior",
        "gate",
        "rehash",
    ],
)
def test_authority(full_root, tmp_path, mode):
    """SCENARIO-REPORT-8443-AUTHORITY: complete objects include gates and history."""
    active, design = full_root / e.ACTIVE, full_root / e.DESIGN
    plan = yaml.safe_load(active.read_bytes())
    t = plan["tasks"][0]
    if mode == "drop":
        plan["tasks"].pop()
    if mode == "reorder":
        plan["tasks"][:2] = plan["tasks"][1::-1]
    if mode in {"prompt", "routing"}:
        t["prompt" if mode == "prompt" else "model"] += " changed"
    if mode == "field":
        t["unknown_field"] = True
    if mode == "model":
        t["MODEL_SPECS"] = [{"name": "changed"}]
    if mode == "retire":
        t["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "prior":
        t["prior_failures"][0]["addressed_by"] += " changed"
    if mode == "gate":
        plan["tasks"][1]["gated_on"][0]["value"] = 0
    if mode == "stage":
        changed = deepcopy(plan)
        changed["tasks"][0]["prompt"] += " changed"
        (full_root / e.STAGED).write_text(yaml.safe_dump(changed))
    if mode == "table":
        design.write_text(design.read_text().replace("| 1 | exp8443", "| 2 | exp8443"))
    if mode == "digest":
        design.write_text(design.read_text().replace(e.TASKS_PIN, "0" * 64))
    if mode == "missing":
        design.unlink()
    if mode == "rehash":
        t["prompt"] += " altered independently"
        design.write_text(
            design.read_text()
            .split("<!-- V727")[0]
            .replace(e.TASKS_PIN, e.tasks_digest(plan["tasks"]))
            + "<!-- V727_TASK_CONTRACT_START -->\n```json\n"
            + json.dumps(plan)
            + "\n```\n"
        )
    active.write_text(yaml.safe_dump(plan))
    result = e.authority(full_root, tmp_path / "snapshots")
    assert result["activated"] == (mode == "valid")
    assert len(result["contract_rows"]) == 14
    if mode != "valid":
        assert result["gate_check_summary"]


def test_protocol_and_missing_design(tmp_path):
    """SCENARIO-REPORT-8443-FAMILIES: absent design is distinct from activated slots."""
    p, old = e.protocol(), e.legacy.protocol()
    for key in e.INHERITED:
        assert p[key] == old[key]
    assert p["objective_updates"]["brier_derivative"] == "2*(p-y)*p*(1-p)/T"
    assert p["journal"]["checkpoint_every_transactions"] == 32
    result = e.authority(e.ROOT, tmp_path / "actual")
    assert result["observed_count"] == 14
    assert not result["activated"]
    assert result["gate_check_summary"]


def test_family_isolation(full_root, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8443-FAMILIES: missing learning evidence cannot zero service."""
    monkeypatch.setattr(
        e,
        "historical",
        lambda root, raw: dict(
            families=[
                dict(experiment_id=n, qualified=n != 8431, gate_check_summary=[])
                for n in e.h.FIELDS
            ],
            references=[],
            receipts=[],
        ),
    )
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("pass", [sys.executable, "-c", "print('genuine')"], raw / "logs")
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    assert value["methods_ready_score"] == 1
    assert value["learning_inputs_ready_score"] == 0
    assert value["service_inputs_ready_score"] == 1
    assert value["runtime_repair_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    assert e.build(work, [], raw, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"


def test_real_private_cli(full_root, tmp_path):
    """SCENARIO-VERIFY-8443-REPLAY: real children publish and reject rehashed bytes."""
    output = tmp_path / (e.NAME + ".json")
    base = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    cases = [
        (
            [
                "--date",
                "20261012",
                "--private-control",
                "--root",
                str(full_root),
                "--output",
                str(output),
            ],
            0,
        ),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261010"], 2),
        (["--private-control"], 2),
    ]
    for i, (args, expected) in enumerate(cases):
        receipt = child(f"cli-{i}", base + args, tmp_path / "logs", expected=expected, deadline=600)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    outside = subprocess.run(
        base + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert outside.returncode == 0, outside.stderr
    value = json.loads(output.read_bytes())
    assert Path(value["terminal_validation_sidecar_path"]).is_file()
    assert value["learning_inputs_ready_score"] == value["service_inputs_ready_score"] == 1
    value["learning_inputs_ready_score"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    assert any(p["name"] == "private_E2E021" for p in r.manifest(tmp_path / "plan"))


def test_protocol_and_resource_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8443-FAMILIES: independent seals and resources fail closed."""
    original_hash, original_read = e.sha256_file, e.read
    monkeypatch.setattr(
        e, "sha256_file", lambda p: "changed" if p == e.ROOT / e.METHODS else original_hash(p)
    )
    with pytest.raises(ValueError, match="methods_hash"):
        e.protocol()
    monkeypatch.setattr(e, "sha256_file", original_hash)
    p = e.protocol()
    old = deepcopy(e.legacy.protocol())
    old["partition"]["reseeding"] = True
    monkeypatch.setattr(
        e,
        "read",
        lambda ref: old if ref == p["inherited_protocol_reference"] else original_read(ref),
    )
    with pytest.raises(ValueError, match="inherited_scientific_operand"):
        e.protocol()
    monkeypatch.setattr(e, "read", original_read)
    monkeypatch.setattr(
        e, "historical", lambda root, raw: dict(families=[], references=[], receipts=[])
    )
    resources = e.legacy.legacy.resources
    monkeypatch.setattr(
        e.legacy.legacy,
        "resources",
        lambda raw: dict(resources(raw), private_disk_backed_scratch=False),
    )
    work = e.measure(tmp_path / "absent", tmp_path / "raw")
    assert work["failures"]
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert (
        value["methods_ready_score"]
        == value["learning_inputs_ready_score"]
        == value["service_inputs_ready_score"]
        == 0
    )
    assert value["verdict_class"] == "blocked"


def test_history_missing_and_corrupt(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8443-FAMILIES: missing authority or producer stays blocked."""
    h = e.h
    with pytest.raises(FileNotFoundError):
        h.custody(tmp_path, tmp_path / "copies")
    root = tmp_path / "history"
    target = root / h.DESIGN
    target.parent.mkdir(parents=True)
    target.write_text("changed")
    with pytest.raises(ValueError, match="historical_authority_pin"):
        h.custody(root, tmp_path / "copies")
    missing = dict(path=str(tmp_path / "missing"), sha256=None)
    assert not h.qualify(8431, missing, [], tmp_path / "scratch")["qualified"]
    primary = dict(missing, sha256="sha256:" + h.PINS[8431])
    assert not h.qualify(8431, primary, [], tmp_path / "scratch")["qualified"]
    authority = [dict(snapshot_path=str(target)), dict(snapshot_path=str(target))]
    assert not h.qualify(8431, primary, authority, tmp_path / "scratch")["qualified"]
    monkeypatch.setattr(h, "sha256_file", lambda p: h.DESIGN_PIN if p == target else h.ACTIVE_PIN)
    authority[1]["snapshot_path"] = str(tmp_path / "second")
    monkeypatch.setattr(h.cap, "load", lambda ref: dict(head_ready_score=0))
    monkeypatch.setattr(h.yaml, "safe_load", lambda data: dict(tasks=[dict(id="exp8431-fixture")]))
    (tmp_path / "second").write_text("private")
    monkeypatch.setattr(h.cap, "outcome", lambda *args: dict(eligible=False))
    assert not h.qualify(8431, primary, authority, tmp_path / "scratch")["qualified"]
    monkeypatch.setattr(e, "child", lambda *args, **kwargs: dict(passed=False, actual_exit=9))
    result = e.historical(tmp_path / "absent", tmp_path / "raw")
    assert len(result["families"]) == 5
    assert all(not f["qualified"] for f in result["families"])


def test_history_argument_boundary(tmp_path):
    """SCENARIO-VERIFY-8443-REPLAY: child operands cannot be silently omitted."""
    with pytest.raises(SystemExit) as error:
        r.main(["--history-request", str(tmp_path / "request")])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "mode", ["protocol", "source", "authority", "receipt", "log", "runtime", "family"]
)
def test_rehashed_primitives(full_root, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8443-REPLAY: repaired envelopes cannot forge underlying facts."""
    fixed = dict(
        families=[dict(experiment_id=n, qualified=True, gate_check_summary=[]) for n in e.h.FIELDS],
        references=[],
        receipts=[],
    )
    monkeypatch.setattr(e, "historical", lambda root, raw: deepcopy(fixed))
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("actual", [sys.executable, "-c", "print('genuine log')"], raw / "logs")
    candidate = tmp_path / "candidate.json"
    if mode == "protocol":
        work["protocol"]["partition"]["reseeding"] = True
    if mode == "authority":
        work["contract"]["contract_rows"][0]["checks"]["full_object"] = False
    if mode == "runtime":
        work["runtime"]["ready"] = True
    if mode == "family":
        work["history"]["families"][0]["qualified"] = False
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    if mode == "source":
        value["code_config_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    if mode == "receipt":
        value["validation_receipts"][0]["duration_s"] += 1
    if mode == "log":
        Path(receipt["stdout_path"]).write_text("forged log")
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(candidate, value)
    assert not e.replay(candidate)


def test_repeated_build_preserves_primitive_operands(full_root, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8443-PURE-REDUCTION: summary custody cannot mutate sealed sources."""
    primary = e.cap.freeze(
        e.ROOT / "results/experiment_8407_v724_python_transaction_cost.json", tmp_path / "custody"
    )
    history = dict(
        families=[dict(experiment_id=n, qualified=True, gate_check_summary=[]) for n in e.h.FIELDS],
        references=[primary],
        receipts=[],
    )
    monkeypatch.setattr(e, "historical", lambda root, raw: deepcopy(history))
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    before = deepcopy(work)
    output = tmp_path / (e.NAME + ".json")
    e.build(work, [dict(passed=True)], raw, output)
    e.build(work, [dict(passed=True)], raw, output)
    assert work == before
