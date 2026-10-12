"""REQ-REPORT-8457, REQ-VERIFY-8457: original evidence and complete task authority."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import yaml

from carnot.reporting import v728_contract_methods as e
from carnot.reporting import v728_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture
def full_root(tmp_path):
    """Independent shipped design bytes prevent tests from deriving authority from YAML."""
    root = tmp_path / "authority"
    for name in [e.DESIGN, e.ACTIVE, e.h.DESIGN]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
    (root / "results").symlink_to(e.ROOT / "results", target_is_directory=True)
    return root


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "id",
        "title",
        "phase",
        "deliverable",
        "MODEL_SPECS",
        "inference_substrate_class",
        "gated_on",
        "prior",
        "retire",
        "routing",
        "prompt",
        "optional",
        "drop",
        "reorder",
        "stage",
        "table",
        "digest",
        "missing",
        "start",
        "end",
        "truncated",
        "rehash",
    ],
)
def test_authority(full_root, tmp_path, mode):
    """SCENARIO-REPORT-8457-AUTHORITY: every full object field binds execution."""
    active, design = full_root / e.ACTIVE, full_root / e.DESIGN
    plan = yaml.safe_load(active.read_bytes())
    task = plan["tasks"][0]
    if mode in {"id", "title", "deliverable", "inference_substrate_class", "prompt"}:
        task[mode] += " changed"
    if mode == "phase":
        task[mode] = 4
    if mode == "MODEL_SPECS":
        task[mode] = [{"name": "changed"}]
    if mode == "gated_on":
        plan["tasks"][1][mode][0]["value"] = 0
    if mode == "prior":
        task["prior_failures"][0]["addressed_by"] += " changed"
    if mode == "retire":
        task["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "routing":
        task["model"] += " changed"
    if mode == "optional":
        task["unknown_optional_field"] = True
    if mode == "drop":
        plan["tasks"].pop()
    if mode == "reorder":
        plan["tasks"][:2] = plan["tasks"][1::-1]
    if mode == "stage":
        staged = deepcopy(plan)
        staged["tasks"][0]["prompt"] += " changed"
        (full_root / e.STAGED).write_text(yaml.safe_dump(staged))
    if mode == "table":
        design.write_text(design.read_text().replace("| 1 | `exp8457", "| 2 | `exp8457"))
    if mode == "digest":
        design.write_text(design.read_text().replace(e.TASKS_PIN, "0" * 64))
    if mode in {"start", "end"}:
        marker = f"<!-- V728_TASK_CONTRACT_{mode.upper()} -->"
        design.write_text(design.read_text().replace(marker, ""))
    if mode == "truncated":
        design.write_text(design.read_text().split("## Phase 2")[0])
    if mode == "rehash":
        task["prompt"] += " forged"
        design.write_text(
            design.read_text()
            .split("<!-- V728")[0]
            .replace(e.TASKS_PIN, e.tasks_digest(plan["tasks"]))
            + "<!-- V728_TASK_CONTRACT_START -->\n```json\n"
            + json.dumps(plan)
            + "\n```\n<!-- V728_TASK_CONTRACT_END -->\n"
        )
    if mode == "missing":
        design.unlink()
    active.write_text(yaml.safe_dump(plan))
    result = e.authority(full_root, tmp_path / "snapshots")
    assert result["activated"] == (mode == "valid")
    assert len(result["contract_rows"]) == 14
    assert result["contract_rows"][0]["unit_id"] == plan["tasks"][0]["id"]
    assert all(row["effective_independent_groups"] == 0 for row in result["contract_rows"])
    if mode != "valid":
        assert result["gate_check_summary"]


def test_protocol():
    """SCENARIO-REPORT-8457-PROTOCOL: only declared corrections change frozen science."""
    p = e.protocol()
    old = e.read(p["inherited_protocol_reference"])
    for key in e.INHERITED:
        assert p[key] == old[key]
    assert p["scalar_fit"] == dict(old["scalar_fit"], coefficients=13)
    assert e.scalar_dimension() == 13
    assert p["objective_qualification"]["absolute_error_limit"] == 1e-7
    assert p["arc_panel"]["intended_episodes"] == 16
    assert p["evidence_canary"]["calls"] == 12
    assert len(p["literature_mapping"]) >= 5


def test_manifest_under_runner_binding(tmp_path):
    """SCENARIO-VERIFY-8457-CONTROLS: production bindings must preserve the original manifest."""
    with patch.object(r.shipped, "manifest", r.manifest):
        plan = r.manifest(tmp_path / "production-plan")
    assert {s["name"] for s in plan} >= {"owned_tests", "coverage_report", "private_E2E021"}


@pytest.mark.parametrize("mode", ["note", "inherit", "dimension", "basis"])
def test_protocol_rejects_changes(monkeypatch, mode):
    """SCENARIO-REPORT-8457-PROTOCOL: a repaired envelope cannot change fixed operands."""
    original_hash, original_read = e.sha256_file, e.read
    if mode == "note":
        monkeypatch.setattr(e, "sha256_file", lambda _: "changed")
    elif mode == "basis":
        monkeypatch.setattr(
            e,
            "sha256_file",
            lambda p: "changed" if p.name == "sparse_energy_7996.py" else original_hash(p),
        )
    else:

        def changed(ref):
            value = deepcopy(original_read(ref))
            if ref["path"] == str(e.ROOT / e.PROTOCOL):
                if mode == "inherit":
                    value["source_controls"]["question_counts"][0] = 149
                else:
                    value["scalar_fit"]["coefficients"] = 109
            return value

        monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()
    monkeypatch.setattr(e, "sha256_file", original_hash)


@pytest.mark.parametrize("absent", [None, 8430, 8431, 8433, 8434, 8407])
def test_family_boundaries(full_root, tmp_path, monkeypatch, absent):
    """SCENARIO-REPORT-8457-PROTOCOL: absent families never gate unrelated methods."""
    monkeypatch.setattr(
        e.legacy,
        "historical",
        lambda root, raw: dict(
            families=[
                dict(experiment_id=n, qualified=n != absent, gate_check_summary=[])
                for n in e.h.FIELDS
            ],
            references=[],
            receipts=[],
        ),
    )
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("real", [sys.executable, "-c", "print('genuine')"], raw / "logs")
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [receipt], raw, output)
    assert value["methods_ready_score"] == 1
    assert value["source_inputs_ready_score"] == int(absent not in {8430, 8431})
    assert value["learning_inputs_ready_score"] == int(absent not in {8430, 8431, 8433, 8434})
    assert value["service_inputs_ready_score"] == int(absent != 8407)
    assert value["runtime_repair_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    assert value["future_dependencies"][0]["experiment_id"] == 8458
    assert value["canonical_tasks_sha256"] == e.TASKS_PIN
    before = deepcopy(work)
    assert e.build(work, [], raw, output)["verdict_class"] == "disqualified"
    assert work == before


def test_real_private_cli(full_root, tmp_path):
    """SCENARIO-VERIFY-8457-CONTROLS: real private publication qualifies historical roots."""
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
        (["--date", "20261011"], 2),
        (["--private-control"], 2),
        (["--history-request", str(tmp_path / "missing")], 2),
    ]
    for i, (args, expected) in enumerate(cases):
        receipt = child(f"cli-{i}", base + args, tmp_path / "logs", expected=expected, deadline=600)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "circular_positive"
    assert value["source_inputs_ready_score"] == value["learning_inputs_ready_score"] == 1
    assert value["service_inputs_ready_score"] == 1
    assert value["required_checks_passed"]
    assert Path(value["terminal_validation_sidecar_path"]).is_file()
    value["source_inputs_ready_score"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    assert any(p["name"] == "private_E2E021" for p in r.manifest(tmp_path / "plan"))


def test_missing_inputs_and_deadline(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8457-CONTROLS: owned clocks and absent external operands stay distinct."""
    monkeypatch.setattr(
        e.legacy, "historical", lambda root, raw: dict(families=[], references=[], receipts=[])
    )
    raw = tmp_path / "raw"
    work = e.measure(tmp_path / "missing", raw)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["methods_ready_score"] == value["source_inputs_ready_score"] == 0
    original = e.legacy.time.monotonic_ns
    calls = iter([0, 3601_000_000_000])
    monkeypatch.setattr(e, "measurement_clock", lambda: next(calls))
    work = e.measure(tmp_path / "missing", tmp_path / "late")
    assert any(f["artifact_field"] == "measurement_cap_s" for f in work["failures"])
    assert original() > 0


@pytest.mark.parametrize(
    "mode", ["protocol", "authority", "family", "runtime", "source", "receipt", "log"]
)
def test_rehashed_primitives(full_root, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8457-CONTROLS: repaired hashes cannot replace independently checked facts."""
    fixed = dict(
        families=[dict(experiment_id=n, qualified=True, gate_check_summary=[]) for n in e.h.FIELDS],
        references=[],
        receipts=[],
    )
    monkeypatch.setattr(e.legacy, "historical", lambda root, raw: deepcopy(fixed))
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("actual", [sys.executable, "-c", "print('genuine')"], raw / "logs")
    if mode == "protocol":
        work["protocol"]["partition"]["reseeding"] = True
    if mode == "authority":
        work["contract"]["contract_rows"][0]["checks"]["full_object"] = False
    if mode == "family":
        work["history"]["families"][0]["qualified"] = False
    if mode == "runtime":
        work["runtime"]["ready"] = True
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
