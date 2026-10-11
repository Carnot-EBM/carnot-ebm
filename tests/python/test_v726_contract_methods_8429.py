"""REQ-REPORT-8429 and REQ-VERIFY-8429: qualify private authority and fixed science."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v726_contract_methods as e
from carnot.reporting import v726_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture
def full_root(tmp_path):
    """SCENARIO-REPORT-8429-AUTHORITY: copies preserve actual activated objects."""
    root = tmp_path / "authority"
    for name in [e.DESIGN, e.ACTIVE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
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
    ],
)
def test_authority_mutations(full_root, tmp_path, mode):
    """SCENARIO-REPORT-8429-AUTHORITY: full objects and independent order must agree."""
    active = full_root / e.ACTIVE
    plan = yaml.safe_load(active.read_bytes())
    task = plan["tasks"][0]
    if mode == "drop":
        plan["tasks"].pop()
    if mode == "reorder":
        plan["tasks"][:2] = plan["tasks"][1::-1]
    if mode in {"prompt", "routing"}:
        task["prompt" if mode == "prompt" else "model"] += " changed"
    if mode == "field":
        task["unregistered_field"] = True
    if mode == "model":
        task["MODEL_SPECS"] = [{"name": "changed"}]
    if mode == "retire":
        task["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "stage":
        changed = deepcopy(plan)
        changed["tasks"][0]["prompt"] += " changed"
        (full_root / e.STAGED).write_text(yaml.safe_dump(changed))
    active.write_text(yaml.safe_dump(plan))
    design = full_root / e.DESIGN
    if mode == "table":
        design.write_text(design.read_text().replace("| 1 | exp8429", "| 2 | exp8429"))
    if mode == "digest":
        design.write_text(design.read_text().replace(e.TASKS_PIN, "0" * 64))
    if mode == "missing":
        design.unlink()
    result = e.authority(full_root, tmp_path / "snapshots")
    assert result["activated"] == (mode == "valid")
    assert len(result["contract_rows"]) == 14
    if mode != "valid":
        assert result["gate_check_summary"]


def test_protocol_and_optional_absence(full_root, tmp_path):
    """SCENARIO-REPORT-8429-PROTOCOL: authority changes cannot move the scientific target."""
    p = e.protocol()
    old = e.legacy.protocol()
    for key in e.SCIENCE:
        assert p[key] == old[key]
    assert p["prefit_clarifications"]["tune_source_permutation_seed"] == 7252
    assert any(x["source"] == "HalluScoring 2026" for x in p["literature_mapping"])
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("pass", [sys.executable, "-c", "print('actual')"], raw / "logs")
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    assert value["methods_ready_score"] == value["contract_ready_score"] == 1
    assert value["runtime_repair_ready_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    for field in ["completed_count", "methods_ready_score", "independent_generalization_score"]:
        bad = deepcopy(value)
        bad[field] += 1
        bad["reproducibility_checksum"] = e.checksum(bad)
        atomic_json(candidate, bad)
        assert not e.replay(candidate)
    failed = child("fail", [sys.executable, "-c", "raise SystemExit(3)"], raw / "logs")
    assert e.build(work, [failed], raw, candidate)["verdict_class"] == "disqualified"
    work = e.measure(tmp_path / "absent", tmp_path / "missing-raw")
    assert (
        e.build(work, [receipt], tmp_path / "missing-raw", candidate)["verdict_class"] == "blocked"
    )


def test_real_private_cli(full_root, tmp_path):
    """SCENARIO-VERIFY-8429-REPLAY: actual children check publication and failure exits."""
    output = tmp_path / (e.NAME + ".json")
    base = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for i, (args, expected) in enumerate(
        [
            (
                [
                    "--date",
                    "20261011",
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
    ):
        receipt = child(f"cli-{i}", base + args, tmp_path / "logs", expected=expected, deadline=90)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert e.replay(output)
    value = json.loads(output.read_bytes())
    assert Path(value["terminal_validation_sidecar_path"]).is_file()
    assert r.manifest(tmp_path / "manifest")


def test_protocol_seals_and_required_resources(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8429-PROTOCOL: seals and real preconditions precede readiness."""
    original_hash, original_read = e.sha256_file, e.read
    monkeypatch.setattr(
        e, "sha256_file", lambda p: "changed" if p == e.ROOT / e.METHODS else original_hash(p)
    )
    with pytest.raises(ValueError, match="methods_hash"):
        e.protocol()
    monkeypatch.setattr(e, "sha256_file", original_hash)
    p = e.protocol()
    old = deepcopy(e.legacy.protocol())
    old["fit"]["steps"] += 1
    monkeypatch.setattr(
        e,
        "read",
        lambda ref: old if ref == p["inherited_protocol_reference"] else original_read(ref),
    )
    with pytest.raises(ValueError, match="inherited_scientific_operand"):
        e.protocol()
    monkeypatch.setattr(e, "read", original_read)
    resources = e.legacy.resources
    monkeypatch.setattr(
        e.legacy, "resources", lambda raw: dict(resources(raw), private_disk_backed_scratch=False)
    )
    altered = deepcopy(p)
    altered["source_references"].append(
        dict(path=str(tmp_path / "missing-input"), sha256="sha256:" + "0" * 64)
    )
    monkeypatch.setattr(e, "protocol", lambda: altered)
    work = e.measure(tmp_path / "absent", tmp_path / "raw")
    assert {f["artifact_field"] for f in work["failures"]} == {
        "private_disk_memory_tools_deadline",
        "required_input_sha256",
    }
    assert work["failures"][1]["observed"] is None


@pytest.mark.parametrize(
    "mode", ["protocol", "source", "authority", "receipt", "log", "runtime", "audit"]
)
def test_rehashed_primitive_failures(full_root, tmp_path, mode):
    """SCENARIO-VERIFY-8429-REPLAY: repaired outer hashes cannot forge independent operands."""
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("actual", [sys.executable, "-c", "print('genuine log')"], raw / "logs")
    if mode == "protocol":
        work["protocol"]["partition"]["reseeding"] = True
    if mode == "authority":
        work["contract"]["contract_rows"][0]["checks"]["full_object"] = False
    if mode == "runtime":
        work["runtime"]["ready"] = True
    if mode == "audit":
        work["compatibility_audit"] = [
            dict(references=[dict(path=receipt["stdout_path"], sha256="sha256:" + "0" * 64)])
        ]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    if mode == "source":
        value["code_config_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    if mode == "receipt":
        value["validation_receipts"][0]["duration_s"] += 1
    if mode == "log":
        Path(receipt["stdout_path"]).write_text("forged log")
    value["reproducibility_checksum"] = e.checksum(value)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert not e.replay(candidate)
