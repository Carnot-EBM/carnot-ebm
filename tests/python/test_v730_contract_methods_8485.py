"""REQ-REPORT-8485, REQ-VERIFY-8485: complete authority and sealed methods."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v730_contract_methods as e
from carnot.reporting import v730_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture
def full_root(tmp_path):
    """Private exact copies test authority without editing the activated queue."""
    root = tmp_path / "authority"
    (root / e.DESIGN).parent.mkdir(parents=True)
    for name in [e.DESIGN, e.ACTIVE]:
        (root / name).write_bytes((e.ROOT / name).read_bytes())
    return root


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "count",
        "order",
        "id",
        "title",
        "phase",
        "deliverable",
        "model",
        "substrate",
        "gate",
        "prompt",
        "retirement",
        "end",
        "truncated",
        "missing",
        "rehash",
        "stage",
    ],
)
def test_authority(full_root, tmp_path, mode):
    """SCENARIO-REPORT-8485-AUTHORITY: every independent authority operand binds."""
    active, design = full_root / e.ACTIVE, full_root / e.DESIGN
    plan = yaml.safe_load(active.read_bytes())
    task = plan["tasks"][0]
    if mode in {"id", "title", "deliverable", "model", "prompt"}:
        task[mode] += " changed"
    if mode == "phase":
        task[mode] = 4
    if mode == "substrate":
        task["inference_substrate_class"] = "model_bounded_generation"
    if mode == "gate":
        plan["tasks"][1]["gated_on"][0]["value"] = 0
    if mode == "retirement":
        task["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "count":
        plan["tasks"].pop()
    if mode == "order":
        plan["tasks"][:2] = plan["tasks"][1::-1]
    if mode == "end":
        design.write_text(design.read_text().replace("<!-- V730_TASK_CONTRACT_END -->", ""))
    if mode == "truncated":
        design.write_text(design.read_text().split("## Exact task contract")[0])
    if mode == "missing":
        design.unlink()
    if mode == "rehash":
        task["prompt"] += " forged"
        design.write_text(
            design.read_text()
            .split("<!-- V730_TASK_CONTRACT_START")[0]
            .replace(e.TASKS_PIN, e.tasks_digest(plan["tasks"]))
            + "<!-- V730_TASK_CONTRACT_START -->\n```json\n"
            + json.dumps(plan)
            + "\n```\n<!-- V730_TASK_CONTRACT_END -->\n"
        )
    if mode == "stage":
        staged = deepcopy(plan)
        staged["tasks"][0]["prompt"] += " changed"
        (full_root / e.STAGED).write_text(yaml.safe_dump(staged))
    active.write_text(yaml.safe_dump(plan))
    result = e.authority(full_root, tmp_path / "snapshots")
    assert result["activated"] == (mode == "valid")
    assert len(result["contract_rows"]) == 14
    assert all(row["effective_independent_groups"] == 0 for row in result["contract_rows"])
    if mode != "valid":
        assert result["gate_check_summary"]


def test_protocol():
    """SCENARIO-REPORT-8485-METHODS: scientific operands survive authority repair."""
    p = e.protocol()
    old = e.read(p["inherited_protocol_reference"])
    assert all(p[k] == old[k] for k in e.INHERITED)
    assert p["scalar_fit"]["coefficients"] == 13
    assert p["append_cost"]["bootstrap_seed"] == 7298480
    assert p["evidence_canary"]["calls"] == 16
    assert len(p["evidence_canary"]["views"]) == 4
    assert "successful" in p["validation_repairs"]["panel_registry_rejection"]
    assert p["canonical_tasks_sha256"] == e.TASKS_PIN


@pytest.mark.parametrize("mode", ["note", "inherit", "basis", "fixture", "view"])
def test_protocol_rejection(monkeypatch, mode):
    """SCENARIO-REPORT-8485-METHODS: repaired envelopes cannot move frozen methods."""
    original_read, original_hash = e.read, e.sha256_file
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
                    value["append_protocol"]["fsync_before_ack"] = False
                elif mode == "fixture":
                    value["validation_repairs"]["panel_registry_rejection"] = "omitted"
                else:
                    value["evidence_canary"]["calls"] = 12
            return value

        monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()


def fake_history(absent=None):
    """One varied family shows readiness gates do not silently depend on each other."""
    return dict(
        families=[
            dict(experiment_id=n, qualified=n != absent, gate_check_summary=[]) for n in e.FIELDS
        ],
        references=[],
        receipts=[],
    )


@pytest.mark.parametrize("absent", [None, 8430, 8433, 8465])
def test_family_boundaries(full_root, tmp_path, monkeypatch, absent):
    """SCENARIO-REPORT-8485-METHODS: optional and historical absences stay independent."""
    monkeypatch.setattr(e, "historical", lambda root, raw: fake_history(absent))
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("real", [sys.executable, "-c", "print('genuine')"], raw / "logs")
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    assert value["methods_ready_score"] == value["current_contract_ready_score"] == 1
    assert value["source_inputs_ready_score"] == int(absent != 8430)
    assert value["learning_inputs_ready_score"] == int(absent not in {8430, 8433})
    assert value["service_inputs_ready_score"] == int(absent != 8465)
    assert value["runtime_repair_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    assert value["future_dependencies"][0]["experiment_id"] == 8486
    assert e.build(work, [], raw, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"


def test_missing_and_deadline(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8485-CONTROLS: missing authority and expired work remain visible."""
    monkeypatch.setattr(e, "historical", lambda root, raw: fake_history())
    work = e.measure(tmp_path / "missing", tmp_path / "raw")
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["methods_ready_score"] == 0
    clock = iter([0, 3601_000_000_000])
    monkeypatch.setattr(e, "measurement_clock", lambda: next(clock))
    work = e.measure(tmp_path / "missing", tmp_path / "late")
    assert any(g["artifact_field"] == "measurement_cap_s" for g in work["failures"])


@pytest.mark.parametrize("mode", ["valid", "protocol", "authority", "family", "source", "receipt"])
def test_rehashed_primitives(full_root, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8485-CONTROLS: independent operands reject self-consistent forgery."""
    monkeypatch.setattr(e, "historical", lambda root, raw: fake_history())
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("real", [sys.executable, "-c", "print('genuine')"], raw / "logs")
    if mode == "protocol":
        work["protocol"]["partition"]["reseeding"] = True
    if mode == "authority":
        work["contract"]["contract_rows"][0]["checks"]["full_object"] = False
    if mode == "family":
        work["history"]["families"][0]["qualified"] = False
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    if mode == "source":
        value["code_config_hashes"][0]["sha256"] = "changed"
    if mode == "receipt":
        value["validation_receipts"][0]["duration_s"] += 1
    value["reproducibility_checksum"] = e.checksum(value)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate) == (mode == "valid")


def test_real_private_cli(tmp_path):
    """SCENARIO-VERIFY-8485-CONTROLS: real publication qualifies current and historical inputs."""
    output = tmp_path / (e.NAME + ".json")
    base = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    cases = [
        (["--date", "20261013", "--private-control", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261012"], 2),
        (["--private-control"], 2),
        (["--history-request", str(tmp_path / "missing")], 2),
        (
            [
                "--private-control",
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "absent" / (e.NAME + ".json")),
            ],
            0,
        ),
    ]
    for i, (args, expected) in enumerate(cases):
        receipt = child(f"cli-{i}", base + args, tmp_path / "logs", expected=expected, deadline=600)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "circular_positive"
    assert value["methods_ready_score"] == value["current_contract_ready_score"] == 1
    assert value["source_inputs_ready_score"] == value["learning_inputs_ready_score"] == 1
    assert value["service_inputs_ready_score"] == 1
    assert value["required_checks_passed"]
    value["source_inputs_ready_score"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)


def test_manifest(tmp_path):
    """SCENARIO-VERIFY-8485-CONTROLS: historical shards preserve every collected assertion."""
    plan = r.manifest(tmp_path / "plan")
    for collection in (
        p for p in plan if p["name"].startswith("collect_") and e.TEST not in p["argv"]
    ):
        suite = collection["argv"][-1]
        actual = [
            node
            for p in plan
            if p["name"].startswith("historical_")
            for node in p["collected_node_ids"]
            if node.startswith(suite + "::")
        ]
        assert actual == collection["collected_node_ids"]
        assert len(actual) == len(set(actual))


def test_preserved_v729():
    """SCENARIO-REPORT-8485-AUTHORITY: failed old design stays byte-identical and truncated."""
    p = e.protocol()
    ref = p["preserved_v729_design_reference"]
    assert e.sha256_file(Path(ref["path"])) == ref["sha256"]
    assert "## Exact task contract" not in Path(ref["path"]).read_text()
