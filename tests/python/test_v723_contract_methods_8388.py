"""REQ-REPORT-8388 / REQ-VERIFY-8388: current authority and history qualify separately."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v723_contract_methods as e
from carnot.reporting import v723_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def cli(*args):
    """Fresh processes test the actual reader instead of an imported approximation."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.fixture(scope="module")
def natural(tmp_path_factory):
    private = tmp_path_factory.mktemp("v723")
    output = private / (e.NAME + ".json")
    result = cli("--date", "20261010", "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    return output, value, work


def test_natural(natural):
    """SCENARIO-REPORT-8388-HISTORY: historical absence cannot erase current custody."""
    output, value, _ = natural
    assert e.replay(output) and cli("--cold-replay", output).returncode == 0
    assert len(value["rows"]) == 14 and len(value["future_dependencies"]) == 13
    assert value["current_contract_ready_score"] == value["direct_inputs_ready_score"] == 1
    assert value["historical_replay_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["required_checks_passed"] and not value["flagged_adversarial"]
    assert value["MODEL_SPECS"] == [] and value["no_model_load"]
    assert not any(value["model_invocation_counts"].values())
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert set(value) <= set(value["field_principles"])


@pytest.mark.parametrize(
    "mutation", ["match", "prompt", "delete", "reorder", "table", "digest", "truncated", "stage"]
)
def test_private_e2e018(tmp_path, mutation):
    """SCENARIO-REPORT-8388-AUTHORITY: complete objects and visible order must match."""
    text = (e.ROOT / e.DESIGN).read_text()
    plan = yaml.safe_load((e.ROOT / e.ACTIVE).read_bytes())
    if mutation == "prompt":
        plan["tasks"][0]["prompt"] += " edited"
    if mutation == "delete":
        plan["tasks"].pop()
    if mutation == "reorder":
        plan["tasks"].reverse()
    if mutation == "table":
        text = text.replace(plan["tasks"][0]["title"], "edited", 1)
    if mutation == "digest":
        text = text.replace(e.tasks_digest(plan["tasks"]), "0" * 64)
    if mutation == "truncated":
        text = text.split("<!-- V723_TASK_CONTRACT_START -->")[0]
    p = tmp_path / e.DESIGN
    p.parent.mkdir(parents=True)
    p.write_text(text)
    (tmp_path / e.ACTIVE).write_text(yaml.safe_dump(plan))
    if mutation == "stage":
        plan["tasks"][0]["prompt"] += " changed stage"
        (tmp_path / e.STAGED).write_text(yaml.safe_dump(plan))
    checked = e.authority(tmp_path, tmp_path / "raw")
    assert checked["activated"] == (mutation == "match")
    assert len(checked["contract_rows"]) == 14


def test_protocol(natural):
    """SCENARIO-REPORT-8388-PROTOCOL: freeze choices before human target ingestion."""
    p = natural[1]["direct_service_protocol"]
    old = json.loads((e.ROOT / e.old.PROTOCOL).read_bytes())
    for field in [
        "head",
        "policy",
        "reference",
        "online",
        "panels",
        "transaction",
        "benchmark",
        "prototype",
    ]:
        assert p[field] == old[field]
    assert len(p["cost_cells"]) == 27
    assert p["label_audit"]["bootstrap_replicates"] == 10000
    assert p["label_audit"]["human_targets_opened"] is False


@pytest.mark.parametrize(
    "field", ["contract", "deployment", "kernel", "historical_replay", "source_alias"]
)
def test_rehashed_tamper(natural, tmp_path, field):
    """SCENARIO-VERIFY-8388-REPLAY: a fresh hash cannot authorize changed operands."""
    _, value, work = natural
    changed = deepcopy(work)
    if field == "contract":
        changed[field]["tasks"][0]["prompt"] += " forged"
    elif field == "deployment":
        changed[field]["policy"]["accept_below"] = 0.5
    elif field == "kernel" or field == "historical_replay":
        changed[field]["ready"] = False
    else:
        changed["refs"][0]["source_path"] = "/tmp/changed-authority-alias"
    raw = tmp_path / "raw"
    atomic_json(raw / "measurement.json", changed)
    forged = e.build(changed, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json"))
    path = tmp_path / "forged.json"
    atomic_json(path, forged)
    assert not e.replay(path) and cli("--cold-replay", path).returncode == 1


def test_missing_and_failure(natural, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8388-REPLAY: external absence differs from an actual child error."""
    output = tmp_path / (e.NAME + ".json")
    assert (
        cli("--root", tmp_path / "absent", "--output", output, "--private-fixture").returncode == 0
    )
    blocked = json.loads(output.read_bytes())
    assert blocked["verdict_class"] == "blocked" and e.replay(output)
    assert blocked["current_contract_ready_score"] == blocked["direct_inputs_ready_score"] == 0
    assert blocked["gate_check_summary"][0]["observed"] is None
    assert not e.replay(tmp_path / "absent.json")
    assert cli("--date", "wrong").returncode != 0 and cli("--date").returncode != 0
    assert cli("--private-fixture").returncode != 0
    from carnot.reporting.v709_execution import child

    monkeypatch.setattr(
        r.base.qualified,
        "execute",
        lambda plan, logs: [
            child("error", [sys.executable, "-c", "raise SystemExit(3)"], logs, deadline=10)
        ],
    )
    assert r.main(["--date", "20261010", "--output", str(output)]) == 0
    failed = json.loads(output.read_bytes())
    assert failed["verdict_class"] == "disqualified" and not failed["required_checks_passed"]


def test_global_health_separate(natural, tmp_path):
    """REQ-VERIFY-8388: global health never grants or erases owned qualification."""
    _, value, work = natural
    result = e.build(
        work,
        [*value["validation_receipts"], dict(scope="global", passed=False)],
        Path(value["work_reference"]["path"]).parent,
        tmp_path / (e.NAME + ".json"),
    )
    assert result["direct_inputs_ready_score"] == 1 and result["required_checks_passed"]
    altered = deepcopy(value)
    altered["source_artifact_hashes"] = []
    altered["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in altered.items() if k != "reproducibility_checksum"}
    )
    path = tmp_path / "forged.json"
    atomic_json(path, altered)
    assert not e.replay(path)


def test_complete_machine_and_empty_lineage(tmp_path):
    """SCENARIO-REPORT-8388-AUTHORITY: every optional field is still digest-bound."""
    text = (e.ROOT / e.DESIGN).read_text()
    tasks = e.parse_design(text, milestone=e.MILESTONE)[1]
    original = e.tasks_digest(tasks)
    tasks[0].pop("prior_failures", None)
    digest = e.tasks_digest(tasks)
    prefix = (
        text.split("<!-- V723_TASK_CONTRACT_START -->")[0]
        + "\nCanonical full-task SHA-256: `"
        + original
        + "`\n"
    )
    text = (
        prefix.replace(original, digest)
        + "<!-- V723_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(dict(milestone=e.MILESTONE, tasks=tasks))
        + "\n```\n"
    )
    path = tmp_path / e.DESIGN
    path.parent.mkdir(parents=True)
    path.write_text(text)
    plan = dict(milestone=e.MILESTONE, tasks=tasks)
    for name in [e.ACTIVE, e.STAGED]:
        (tmp_path / name).write_text(yaml.safe_dump(plan))
    assert e.authority(tmp_path, tmp_path / "good")["activated"]
    tasks[0]["prompt"] += " edited independently"
    path.write_text(
        prefix.replace(original, digest)
        + "<!-- V723_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(dict(milestone=e.MILESTONE, tasks=tasks))
        + "\n```\n"
    )
    assert not e.authority(tmp_path, tmp_path / "bad")["activated"]


def test_historical_source_alias_rejected(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8388-HISTORY: an altered alias cannot borrow a sealed hash."""
    value = json.loads((e.ROOT / "results/experiment_8374_v722_contract_methods.json").read_bytes())
    value["source_artifact_hashes"][0]["source_path"] = "/tmp/false-alias"
    path = tmp_path / "results/experiment_8374_v722_contract_methods.json"
    atomic_json(path, value)
    monkeypatch.setattr(e, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="historical_source_alias"):
        e.historical(tmp_path / "scratch")


def test_manifest_frozen(tmp_path):
    """REQ-VERIFY-8388: scoped coverage and both named rejection suites precede measurement."""
    plan = r.manifest(tmp_path)
    assert all(p["deadline"] <= 900 for p in plan)
    assert len([p for p in plan if p["scope"] == "global"]) == 1
    assert {"private_E2E018", "private_E2E021"} <= {p["name"] for p in plan}


def test_failed_receipt_hash(natural, tmp_path):
    """SCENARIO-VERIFY-8388-REPLAY: a forged log digest must fail before recomputation."""
    altered = deepcopy(natural[1])
    altered["validation_receipts"][0]["stdout_sha256"] = "sha256:" + "0" * 64
    path = tmp_path / "forged-log.json"
    atomic_json(path, altered)
    assert not e.replay(path)


def test_actual_consumers_and_outside_cli(natural, tmp_path):
    """REQ-VERIFY-8388: existing consumers select the same bytes as the independent CLI."""
    from carnot.reporting.primary_publication import reader_receipt

    output = natural[0]
    receipt = reader_receipt(
        e.TASK, output.parent, field="current_contract_ready_score", expected=1
    )
    assert receipt["gate_path"] == receipt["document_path"] == str(output)
    assert receipt["gate_sha256"] == receipt["document_sha256"] == e.sha256_file(output)
    result = subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
