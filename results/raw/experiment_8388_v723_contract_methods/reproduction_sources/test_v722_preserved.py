"""REQ-REPORT-8374 / REQ-VERIFY-8374: custody gives no semantic benefit."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v722_contract_methods as e
from carnot.reporting import v722_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def cli(*args):
    """Use a fresh process so imports and CLI statements are measured too."""
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.fixture(scope="module")
def natural(tmp_path_factory):
    private = tmp_path_factory.mktemp("v722")
    output = private / (e.NAME + ".json")
    result = cli("--date", "20261010", "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    return output, value, work


def test_natural(natural):
    """SCENARIO-REPORT-8374-METHODS: direct input custody survives absent design."""
    output, value, work = natural
    assert e.replay(output)
    assert len(value["rows"]) == 14 and len(value["future_dependencies"]) == 13
    assert value["direct_inputs_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["current_contract_ready_score"] == 0
    assert value["required_checks_passed"] and not value["flagged_adversarial"]
    assert value["MODEL_SPECS"] == [] and value["no_model_load"]
    assert not any(value["model_invocation_counts"].values())
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert set(value) <= set(value["field_principles"])
    assert work["inputs"]["heads"] and work["kernel"]["ready"]
    assert value["historical_dispositions"]["executed"] == 8
    assert value["historical_dispositions"]["pre_gate"] == 2
    assert value["historical_dispositions"]["absent"] == 4
    assert value["historical_dispositions"]["utility"]["H1_gain"] == -0.00390625
    assert value["historical_dispositions"]["utility"]["H2_gain"] == 0
    assert "20261010" in value["invocation_argv"]
    assert "20261008" in value["invocation_adapter_argv"]


def test_protocol(natural):
    """SCENARIO-REPORT-8374-METHODS: freeze decisions before future measurement."""
    _, value, work = natural
    p = work["deployment"]
    assert value["protocol_sha256"] == e.PROTOCOL_PIN
    assert p["scientific_protocol"]["sha256"] == e.inputs.PIN
    assert len(p["head"]["coefficients"]) == 34 and p["head"]["temperature"] == 2
    assert p["reference"]["probability"] == "scipy.special.expit(eta / T)"
    assert p["policy"]["accept_below"] == 0.25 and p["policy"]["reject_above"] == 0.75
    assert p["panels"]["random_vectors"] == 4096
    assert p["panels"]["stream_missing_slots"] == 22
    assert p["panels"]["update_fault_seeds"] == [11, 22, 33]
    assert p["benchmark"]["primary_nfr01_cell"] == dict(
        workload="natural", batch=1, predictions_per_update=8
    )
    assert p["benchmark"]["measured_repeats"] == 5
    assert p["prototype"]["positive_threshold_hex"] == "0x1.193ea7aad030bp+0"
    assert p["prototype"]["unchanged_scipy_policy"] is False
    assert "snapshot_pinning" in p["transaction"]["steps"]
    assert "directory_fsync" in p["transaction"]["steps"]
    assert p["direct_gates"]["requires_table_certificate"] is False


@pytest.mark.parametrize(
    "mutation",
    ["match", "prompt", "delete", "reorder", "lineage", "table", "digest", "missing_stage"],
)
def test_private_e2e018(tmp_path, mutation):
    """SCENARIO-REPORT-8374-AUTHORITY: full objects and visible rows must agree."""
    tasks = yaml.safe_load((e.ROOT / e.ACTIVE).read_bytes())["tasks"]
    table = "\n".join(
        f"| {i + 1} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
        for i, t in enumerate(tasks)
    )
    digest = e.tasks_digest(tasks)
    text = (
        "## Exact task contract\n"
        + table
        + "\nCanonical full-task SHA256: `"
        + digest
        + "`\n<!-- V722_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(dict(milestone=e.MILESTONE, tasks=tasks))
        + "\n```\n"
    )
    if mutation == "table":
        text = text.replace(tasks[0]["title"], "wrong", 1)
    if mutation == "digest":
        text = text.replace(digest, "0" * 64)
    design = tmp_path / e.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(text)
    plan = dict(milestone=e.MILESTONE, tasks=deepcopy(tasks))
    if mutation == "prompt":
        plan["tasks"][0]["prompt"] += " changed"
    if mutation == "delete":
        plan["tasks"].pop()
    if mutation == "reorder":
        plan["tasks"].reverse()
    if mutation == "lineage":
        plan["tasks"][0]["prior_failures"][0]["verdict"] += " changed"
    (tmp_path / e.ACTIVE).write_text(yaml.safe_dump(plan))
    if mutation != "missing_stage":
        (tmp_path / e.STAGED).write_text(yaml.safe_dump(plan))
    checked = e.authority(tmp_path, tmp_path / "raw")
    assert checked["activated"] == (mutation in ["match", "missing_stage"])
    assert len(checked["contract_rows"]) == 14
    if mutation == "missing_stage":
        assert checked["staging_disposition"] == "consumed_by_activation"


def test_missing_and_failure(tmp_path, natural):
    """SCENARIO-VERIFY-8374-CLI: owned failure differs from external absence."""
    _, value, work = natural
    failed = e.build(
        work,
        [dict(passed=False, name="deliberate_error")],
        Path(value["work_reference"]["path"]).parent,
        tmp_path / (e.NAME + ".json"),
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["direct_inputs_ready_score"] == 0
    output = tmp_path / (e.NAME + ".json")
    result = cli("--root", tmp_path / "absent", "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    blocked = json.loads(output.read_bytes())
    assert blocked["verdict_class"] == "blocked" and len(blocked["rows"]) == 14
    assert blocked["gate_check_summary"][0]["observed"] is None
    assert blocked["direct_inputs_ready_score"] == 0 and e.replay(output)
    assert not e.replay(tmp_path / "missing.json")
    assert cli("--date", "wrong").returncode != 0
    assert cli("--date").returncode != 0
    assert cli("--private-fixture").returncode != 0


@pytest.mark.parametrize(
    "field",
    [
        "contract",
        "inputs",
        "deployment",
        "methods",
        "historical_dispositions",
        "trajectory",
        "kernel",
    ],
)
def test_rehashed_primitives(natural, tmp_path, field):
    """SCENARIO-VERIFY-8374-REPLAY: hashing forged primitive meaning cannot qualify it."""
    _, value, work = natural
    changed = deepcopy(work)
    if field == "contract":
        changed[field]["tasks"][0]["prompt"] += " forged"
    elif field == "inputs":
        changed[field]["heads"]["manifest"]["heads"][0]["coefficients"][0] += 0.125
    elif field == "deployment":
        changed[field]["policy"]["accept_below"] = 0.5
    elif field == "methods":
        changed[field]["science_changed"] = True
    elif field == "historical_dispositions":
        changed[field]["utility"]["H2_gain"] = 0.02
    else:
        changed[field]["ready"] = False
    raw = tmp_path / "raw"
    atomic_json(raw / "measurement.json", changed)
    forged = e.build(changed, value["validation_receipts"], raw, tmp_path / (e.NAME + ".json"))
    path = tmp_path / "forged.json"
    atomic_json(path, forged)
    assert not e.replay(path)


def test_receipts_and_source_tamper(natural, tmp_path):
    """SCENARIO-VERIFY-8374-REPLAY: changed receipts and source arrays fail closed."""
    _, value, _ = natural
    altered = deepcopy(value)
    altered["source_artifact_hashes"] = []
    altered["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in altered.items() if k != "reproducibility_checksum"}
    )
    path = tmp_path / "forged.json"
    atomic_json(path, altered)
    assert not e.replay(path)
    altered = deepcopy(value)
    altered["validation_receipts"][0]["stdout_sha256"] = "sha256:" + "0" * 64
    atomic_json(path, altered)
    assert not e.replay(path)


def test_owned_child_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8374-CLI: a real failed child must disqualify readiness."""
    from carnot.reporting.v709_execution import child

    def fail(plan, logs):
        return [
            child(
                "deliberate_error",
                [sys.executable, "-u", "-c", "raise SystemExit(3)"],
                logs,
                deadline=10,
            )
        ]

    monkeypatch.setattr(r.qualified, "execute", fail)
    output = tmp_path / (e.NAME + ".json")
    assert r.main(["--date", "20261010", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["validation_receipts"][0]["actual_exit"] == 3
    assert value["direct_inputs_ready_score"] == 0
    assert not value["required_checks_passed"] and value["flagged_adversarial"]


def test_global_health_separate(natural, tmp_path):
    """REQ-VERIFY-8374: pre-existing global failures cannot erase qualified inputs."""
    _, value, work = natural
    receipts = [*value["validation_receipts"], dict(name="global", scope="global", passed=False)]
    result = e.build(
        work, receipts, Path(value["work_reference"]["path"]).parent, tmp_path / (e.NAME + ".json")
    )
    assert result["required_checks_passed"] and result["direct_inputs_ready_score"] == 1
    assert result["repository_health"][0]["passed"] is False


def test_method_error(natural, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8374-REPLAY: malformed owned method operands remain explicit."""
    original = e.inputs.bind

    def fail(ref, raw, refs, **kw):
        if ref["path"].endswith(e.METHODS):
            raise ValueError("deliberate_error")
        return original(ref, raw, refs, **kw)

    monkeypatch.setattr(e.inputs, "bind", fail)
    work = e.measure(e.ROOT, tmp_path)
    assert work["failures"][-1]["observed"] == "deliberate_error"
    assert work["deployment"] == {} and work["kernel"] == {}


def test_disk_budget_gate(tmp_path, monkeypatch):
    """REQ-VERIFY-8374: memory-backed scratch cannot qualify durable evidence."""
    read = Path.read_text

    def memory_mount(path, *args, **kwargs):
        if str(path) == "/proc/self/mountinfo":
            return "1 0 0:0 / / rw - tmpfs none rw\n"
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", memory_mount)
    work = e.measure(tmp_path / "missing", tmp_path / "raw")
    assert any(g["artifact_field"] == "private_disk_memory_budget" for g in work["failures"])


def test_qualified_administrative_control(natural, tmp_path):
    """REQ-REPORT-8374: a complete reference contract can grant only oracle custody."""
    _, value, work = natural
    changed = deepcopy(work)
    changed["contract"]["activated"] = True
    changed["failures"] = []
    result = e.build(
        changed,
        value["validation_receipts"],
        Path(value["work_reference"]["path"]).parent,
        tmp_path / (e.NAME + ".json"),
    )
    assert result["verdict_class"] == "circular_positive"
    assert result["current_contract_ready_score"] == 1
    assert result["independent_generalization_score"] == 0
