"""REQ-REPORT-8471, REQ-VERIFY-8471: authority absence and separate inputs."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v729_contract_methods as e
from carnot.reporting import v729_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture
def full_root(tmp_path):
    """Synthetic positive control stays private; it cannot repair repository authority."""
    root = tmp_path / "authority"
    root.mkdir()
    active = (e.ROOT / e.ACTIVE).read_bytes()
    (root / e.ACTIVE).write_bytes(active)
    plan = yaml.safe_load(active)
    table = "\n".join(
        f"| {i + 1} | `{t['id']}` | {t['title']} | {t['phase']} | `{t['deliverable']}` |"
        for i, t in enumerate(plan["tasks"])
    )
    design = root / e.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(
        "# Synthetic V729 test control\n## Phase 1\n## Phase 2\n"
        + "## Exact task contract\n"
        + table
        + "\nCanonical complete-task SHA256: `"
        + e.TASKS_PIN
        + "`\n"
        + "<!-- V729_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(plan)
        + "\n```\n<!-- V729_TASK_CONTRACT_END -->\n"
    )
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
    """SCENARIO-REPORT-8471-AUTHORITY: every task field and independent seal binds."""
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
        design.write_text(design.read_text().replace("| 1 | `exp8471", "| 2 | `exp8471"))
    if mode == "digest":
        design.write_text(design.read_text().replace(e.TASKS_PIN, "0" * 64))
    if mode in {"start", "end"}:
        design.write_text(
            design.read_text().replace(f"<!-- V729_TASK_CONTRACT_{mode.upper()} -->", "")
        )
    if mode == "truncated":
        design.write_text(design.read_text().split("## Phase 2")[0])
    if mode == "rehash":
        task["prompt"] += " forged"
        design.write_text(
            design.read_text()
            .split("<!-- V729")[0]
            .replace(e.TASKS_PIN, e.tasks_digest(plan["tasks"]))
            + "<!-- V729_TASK_CONTRACT_START -->\n```json\n"
            + json.dumps(plan)
            + "\n```\n<!-- V729_TASK_CONTRACT_END -->\n"
        )
    if mode == "missing":
        design.unlink()
    active.write_text(yaml.safe_dump(plan))
    result = e.authority(full_root, tmp_path / "snapshots")
    assert result["activated"] == (mode == "valid")
    assert len(result["contract_rows"]) == 14
    assert result["observed_count"] == len(plan["tasks"])
    assert all(row["effective_independent_groups"] == 0 for row in result["contract_rows"])
    if mode != "valid":
        assert result["gate_check_summary"]


def test_actual_truncation(tmp_path):
    """SCENARIO-REPORT-8471-AUTHORITY: observed activation cannot invent missing design."""
    value = e.authority(e.ROOT, tmp_path)
    assert value["observed_count"] == 14
    assert not value["activated"]
    assert value["canonical_tasks_sha256"] is None
    fields = {g["artifact_field"] for g in value["gate_check_summary"]}
    assert fields >= {
        "independent_visible_task_table",
        "embedded_complete_task_objects",
        "printed_canonical_task_digest",
    }


def test_protocol():
    """SCENARIO-REPORT-8471-METHODS: repairs and append methods preserve frozen science."""
    p = e.protocol()
    old = e.read(p["inherited_protocol_reference"])
    assert all(p[k] == old[k] for k in e.INHERITED)
    assert p["scalar_fit"]["coefficients"] == 13
    assert p["append_protocol"]["fsync_before_ack"] is True
    assert p["append_cost"]["bootstrap_seed"] == 7298480
    assert p["validation_repairs"]["changed_reference_rejection"] == "Exp8461 line388"
    assert len(p["literature_mapping"]) >= 6


@pytest.mark.parametrize("mode", ["note", "inherit", "basis", "append", "repair"])
def test_protocol_rejections(monkeypatch, mode):
    """SCENARIO-REPORT-8471-METHODS: rehashed envelopes cannot move independent operands."""
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
                    value["source_controls"]["question_counts"][0] = 149
                elif mode == "append":
                    value["append_protocol"]["fsync_before_ack"] = False
                else:
                    value["validation_repairs"]["changed_reference_rejection"] = "omitted"
            return value

        monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()


def fake_history(absent=None):
    """Synthetic family controls vary one gate without importing any scientific metric."""
    return dict(
        families=[
            dict(experiment_id=n, qualified=n != absent, gate_check_summary=[]) for n in e.FIELDS
        ],
        references=[],
        receipts=[],
    )


@pytest.mark.parametrize("absent", [None, 8430, 8431, 8433, 8434, 8407, 8464, 8465, 8466])
def test_family_boundaries(full_root, tmp_path, monkeypatch, absent):
    """SCENARIO-REPORT-8471-METHODS: missing families do not cross readiness boundaries."""
    monkeypatch.setattr(e, "historical", lambda root, raw: fake_history(absent))
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("real", [sys.executable, "-c", "print('genuine')"], raw / "logs")
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    assert value["methods_ready_score"] == 1
    assert value["source_inputs_ready_score"] == int(absent not in {8430, 8431})
    assert value["learning_inputs_ready_score"] == int(absent not in {8430, 8431, 8433, 8434})
    assert value["service_inputs_ready_score"] == int(absent not in {8407, 8464, 8465, 8466})
    assert value["runtime_repair_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    assert value["future_dependencies"][0]["experiment_id"] == 8472
    before = deepcopy(work)
    assert e.build(work, [], raw, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"
    assert work == before


def test_missing_and_deadline(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8471-CONTROLS: owned deadlines and missing authority stay distinct."""
    monkeypatch.setattr(e, "historical", lambda root, raw: fake_history())
    work = e.measure(tmp_path / "missing", tmp_path / "raw")
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["methods_ready_score"] == 0
    clock = iter([0, 3601_000_000_000])
    monkeypatch.setattr(e, "measurement_clock", lambda: next(clock))
    work = e.measure(tmp_path / "missing", tmp_path / "late")
    assert any(g["artifact_field"] == "measurement_cap_s" for g in work["failures"])


@pytest.mark.parametrize(
    "mode", ["protocol", "authority", "family", "runtime", "source", "receipt", "log"]
)
def test_rehashed_primitives(full_root, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8471-CONTROLS: independent replay rejects self-consistent tampering."""
    monkeypatch.setattr(e, "historical", lambda root, raw: fake_history())
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


def test_real_private_cli(tmp_path):
    """SCENARIO-VERIFY-8471-CONTROLS: real blocked publication still qualifies independent inputs."""
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
    ]
    for i, (args, expected) in enumerate(cases):
        receipt = child(f"cli-{i}", base + args, tmp_path / "logs", expected=expected, deadline=600)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["methods_ready_score"] == 0
    assert value["required_checks_passed"]
    assert value["source_inputs_ready_score"] == value["learning_inputs_ready_score"] == 1
    assert value["service_inputs_ready_score"] == 1
    value["source_inputs_ready_score"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)


def test_historical_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8471-METHODS: independent producer pins and original roots reject forgery."""
    h = e.history
    path = next((e.ROOT / "results").glob("experiment_8464_*.json"))
    primary = h.service.freeze(path, tmp_path / "primary")
    p = e.protocol()
    authority = []
    for original in p["historical_authorities"]["V728"]:
        ref = h.service.freeze(Path(original["path"]), tmp_path / "authority")
        ref["source_path"] = original["source_path"]
        authority.append(ref)
    request = dict(number=8464, primary=primary, authority=authority)
    changed = deepcopy(request)
    changed["primary"]["sha256"] = "changed"
    assert not h.qualify(changed, tmp_path / "scratch")["qualified"]
    changed = deepcopy(request)
    changed["authority"][0]["sha256"] = "changed"
    assert not h.qualify(changed, tmp_path / "scratch")["qualified"]
    monkeypatch.setattr(h.service, "outcome", lambda *args: dict(eligible=False))
    assert not h.qualify(request, tmp_path / "scratch")["qualified"]
    monkeypatch.setattr(h, "child", lambda *args, **kwargs: dict(passed=False, actual_exit=7))
    absent = deepcopy(p)
    absent["historical_authorities"]["V728"][0]["sha256"] = "changed"
    result = h.historical(tmp_path / "missing", tmp_path / "absent", e.CLI, absent)
    assert len(result["families"]) == len(e.FIELDS)
    assert all(not f["qualified"] for f in result["families"])


def test_manifest_and_collection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8471-CONTROLS: shard union equals exact collected historical suites."""
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
    bad = tmp_path / "bad-collection"
    bad.write_text("")
    monkeypatch.setattr(
        r, "child", lambda *args, **kwargs: dict(passed=False, stdout_path=str(bad))
    )
    with pytest.raises(ValueError, match="required_test_collection"):
        r.collect(tmp_path, e.TEST)


def test_bounded_replay_child(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8471-CONTROLS: complete historical replay has a bounded budget."""
    monkeypatch.setattr(r, "child", lambda *args, **kwargs: kwargs)
    assert r.bounded_child("cold", ["--cold-replay"], tmp_path, deadline=60)["deadline"] == 240
    assert r.bounded_child("other", ["ordinary"], tmp_path, deadline=60)["deadline"] == 60


def test_referenced_model_provenance():
    """SCENARIO-REPORT-8471-METHODS: full cached receipts stay available without recursive copying."""
    ref = dict(path="original.json", snapshot_path="sealed.json", sha256="sha256:original")
    value = dict(
        historical_model_provenance=[
            dict(
                MODEL_SPECS=[dict(name="cached")],
                imported_counts=dict(loads=3),
                scope="historical",
                model_receipt=dict(records=["large cached evidence"]),
                primary=dict(rows=["original"]),
            )
        ]
    )
    before = deepcopy(value)
    rows = e.history.model_provenance(value, ref)
    assert value == before
    assert rows[0]["MODEL_SPECS"] == [dict(name="cached")]
    assert rows[0]["imported_counts"] == dict(loads=3)
    assert rows[0]["provenance_reference"] == dict(
        path="sealed.json", sha256="sha256:original", field="historical_model_provenance"
    )
    assert "model_receipt" not in rows[0] and "primary" not in rows[0]
