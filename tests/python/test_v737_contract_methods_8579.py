"""REQ-REPORT-8579, REQ-VERIFY-8579: authority and fresh-process rejection controls."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v737_contract_methods as e


@pytest.fixture
def authority(tmp_path):
    """A private complete contract exercises full objects without rewriting actual authority."""
    p = e.protocol()
    tasks = deepcopy(p["task_map_v737"])
    text = "## Exact task contract\n"
    text += "\n".join(
        f"| {i + 1} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
        for i, t in enumerate(tasks)
    )
    text += "\ncanonical_tasks_sha256: " + e.tasks_digest(tasks)
    text += "\n<!-- V737_TASK_CONTRACT_START -->\n```json\n"
    text += json.dumps(dict(milestone=e.MILESTONE, tasks=tasks))
    text += "\n```\n<!-- V737_TASK_CONTRACT_END -->\n"
    (tmp_path / e.DESIGN).parent.mkdir(parents=True)
    (tmp_path / e.DESIGN).write_text(text)
    (tmp_path / e.ACTIVE).write_text(yaml.safe_dump(dict(milestone=e.MILESTONE, tasks=tasks)))
    return tmp_path


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
        "missing",
        "rehash",
        "stage",
    ],
)
def test_contract(authority, mode):
    """SCENARIO-REPORT-8579-AUTHORITY: every independent mutation must fail."""
    active = authority / e.ACTIVE
    v = yaml.safe_load(active.read_bytes())
    t = v["tasks"][0]
    if mode in {"id", "title", "deliverable", "model", "prompt"}:
        t[mode] += " forged"
    if mode == "phase":
        t["phase"] = 99
    if mode == "substrate":
        t["inference_substrate_class"] = "live_llm_inference"
    if mode == "gate":
        v["tasks"][2]["gated_on"][0]["value"] = 99
    if mode == "retirement":
        t["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "count":
        v["tasks"].pop()
    if mode == "order":
        v["tasks"].reverse()
    if mode == "end":
        d = authority / e.DESIGN
        d.write_text(d.read_text().replace("<!-- V737_TASK_CONTRACT_END -->", ""))
    if mode == "missing":
        (authority / e.DESIGN).unlink()
    if mode == "rehash":
        t["prompt"] += " forged"
        d = authority / e.DESIGN
        text = d.read_text().replace(
            e.tasks_digest(e.protocol()["task_map_v737"]), e.tasks_digest(v["tasks"])
        )
        a, b = text.split("```json\n")
        d.write_text(a + "```json\n" + json.dumps(v) + "\n```\n<!-- V737_TASK_CONTRACT_END -->")
    if mode == "stage":
        (authority / e.STAGED).write_text(yaml.safe_dump(dict(v, tasks=[])))
    active.write_text(yaml.safe_dump(v))
    r = e.authority(authority)
    assert r["ready"] == (mode == "valid")
    assert len(r["rows"]) == 12
    assert all(x["effective_independent_groups"] == 0 for x in r["rows"])


def test_protocol():
    """SCENARIO-REPORT-8579-FAMILIES: original operands and new session grid are sealed."""
    p = e.protocol()
    old = e.read(p["inherited_v735_protocol_reference"])
    assert all(p[k] == old[k] for k in e.INHERITED)
    assert p["resident_session"]["requests_per_panel"] == 64
    assert p["resident_session"]["session_lengths_panels"] == [1, 4, 16]
    assert p["resident_session"]["p95_ratio_at_most"] == 1.05
    assert p["MODEL_SPECS"] == []
    assert p["preserved_v735_design_reference"]["sha256"].endswith(
        "f198170ac62aeb1074482f9b35dd71995f9c8f6ada4235b831958124a7dbfd15"
    )


def test_protocol_rejects(monkeypatch):
    """SCENARIO-REPORT-8579-FAMILIES: pin changes reject a consistently rehashed envelope."""
    monkeypatch.setattr(e, "PROTOCOL_PIN", "sha256:forged")
    with pytest.raises(ValueError):
        e.protocol()


def test_history_and_missing(tmp_path):
    """SCENARIO-REPORT-8579-FAMILIES: original family receipts qualify without a new controller."""
    from carnot.reporting import v737_contract_methods as h

    r = h.historical(e.protocol())
    assert r["learning_inputs_ready_score"] == 1
    assert r["resident_inputs_ready_score"] == 1
    assert r["canary_inputs_ready_score"] == 0
    assert any(g["field"] == "original_dependency_sha256" for g in r["gate_check_summary"])
    p = deepcopy(e.protocol())
    p["original_producers"]["8431"]["path"] = str(tmp_path / "absent")
    r = h.historical(p)
    assert r["learning_inputs_ready_score"] == 0
    assert r["gate_check_summary"]
    assert h.checked(dict(path=str(tmp_path / "absent"), sha256="missing")) is False


def test_primitives_and_replay(authority, tmp_path):
    """SCENARIO-VERIFY-8579-CONTROLS: immutable work and recomputation reject repaired summaries."""
    from carnot.reporting import v737_contract_report as report

    raw = tmp_path / "raw"
    work = e.measure(authority, raw)
    output = tmp_path / (e.NAME + ".json")
    checks = raw / "checks.json"
    e.atomic_json(checks, dict(receipts=[]))
    value = report.build(work, e.reference(checks), raw, output)
    e.atomic_json(output, value)
    assert report.replay(output)
    assert value["verdict_class"] == "circular_positive"
    value["methods_ready_score"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    assert not report.replay(output)
    value["primitive_reference"]["path"] = str(tmp_path / "missing")
    e.atomic_json(output, value)
    assert not report.replay(output)
    e.atomic_json(checks, dict(receipts=[dict(passed=False, name="owned_failure")]))
    value = report.build(work, e.reference(checks), raw, output)
    assert value["verdict_class"] == "disqualified"
    work["contract"]["ready"] = False
    e.atomic_json(checks, dict(receipts=[]))
    value = report.build(work, e.reference(checks), raw, output)
    assert value["verdict_class"] == "blocked"


def test_runtime_missing(tmp_path):
    """SCENARIO-VERIFY-8579-CONTROLS: optional absence never runs a device command."""
    from carnot.reporting import v737_contract_methods as h

    assert h.runtime(tmp_path)["ready"] is False


@pytest.mark.parametrize("mode", ["scientific", "task", "design"])
def test_protocol_operand_rejection(monkeypatch, mode):
    """SCENARIO-REPORT-8579-FAMILIES: independent seals survive envelope repairs."""
    original = e.read

    def changed(ref):
        p = deepcopy(original(ref))
        if ref["path"] == str(e.ROOT / e.PROTOCOL):
            if mode == "scientific":
                p["objective_updates"] = {}
            if mode == "task":
                p["task_map_v737"][0]["prompt"] += " tampered"
            if mode == "design":
                p["preserved_v735_design_reference"]["sha256"] = "forged"
        return p

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()


def test_family_rejections(tmp_path):
    """SCENARIO-REPORT-8579-FAMILIES: changed dependencies cannot reuse qualification."""
    from carnot.reporting import v737_contract_methods as h

    p = tmp_path / "family.json"
    source = tmp_path / "source"
    source.write_text("source")
    ref = e.reference(source)
    source.write_text("changed")
    e.atomic_json(p, dict(required_checks_passed=False, code_config_hashes=[ref]))
    row = h.family(e.reference(p), 1)
    assert not row["qualified"] and len(row["gate_check_summary"]) == 2
    assert not h.checked(dict(config={}, sha256="wrong"))


def test_replay_rejections(authority, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8579-CONTROLS: changed code, authority and logs reject replay."""
    from carnot.reporting import v737_contract_report as report
    from carnot.reporting import v737_contract_methods as h

    raw = tmp_path / "raw"
    work = e.measure(authority, raw)
    checks = raw / "checks.json"
    log = raw / "log"
    log.write_text("original")
    e.atomic_json(
        checks,
        dict(
            receipts=[
                dict(
                    passed=True,
                    name="log_check",
                    stdout_path=str(log),
                    stdout_sha256=e.sha256_file(log),
                )
            ]
        ),
    )
    output = tmp_path / (e.NAME + ".json")
    value = report.build(work, e.reference(checks), raw, output)
    e.atomic_json(output, value)
    log.write_text("modified")
    assert not report.replay(output)
    log.write_text("original")
    with monkeypatch.context() as m:
        original_checked = h.checked
        m.setattr(
            h,
            "checked",
            lambda ref: (
                False
                if ref.get("path", "").endswith("v737_contract_methods.py")
                else original_checked(ref)
            ),
        )
        assert not report.replay(output)
    with monkeypatch.context() as m:
        m.setattr(e, "authority", lambda root: {})
        assert not report.replay(output)
    value["reproducibility_checksum"] = "forged"
    e.atomic_json(output, value)
    assert not report.replay(output)


def test_runtime_supplied(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8579-CONTROLS: supplied receipts use a bounded reader child only."""
    from carnot.reporting import v737_contract_methods as h

    path = tmp_path / "ops/cuda-runtime-repair-receipt.json"
    path.parent.mkdir()
    path.write_text("{}")
    result = h.runtime(tmp_path, tmp_path / "runtime")
    assert result["ready"] is False
    assert result["reader_receipt"]["deadline_s"] == 60
    monkeypatch.setattr(
        h, "child", lambda *a, **kw: dict(passed=False, actual_exit=-9, deadline_s=60)
    )
    assert not h.runtime(tmp_path, tmp_path / "failed")["ready"]


def test_union_failure(monkeypatch):
    """SCENARIO-REPORT-8579-FAMILIES: an owned union mismatch cannot qualify learning."""
    from carnot.reporting import v737_contract_methods as h
    from carnot.reporting import historical_learning_8554 as original

    monkeypatch.setattr(original, "exact_union", lambda *args: dict(passed=False))
    r = h.historical(e.protocol())
    assert r["learning_inputs_ready_score"] == 0
    assert r["resident_inputs_ready_score"] == 1


def test_resource_absence(authority, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8579-CONTROLS: missing required tools are explicit external gates."""
    from carnot.reporting import v737_contract_report as report

    monkeypatch.setattr(e.shutil, "which", lambda name: None)
    from types import SimpleNamespace

    monkeypatch.setattr(e.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    raw = tmp_path / "raw"
    work = e.measure(authority, raw)
    checks = raw / "checks.json"
    e.atomic_json(checks, dict(receipts=[]))
    value = report.build(work, e.reference(checks), raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"]


def test_original_head_rejection(monkeypatch):
    """SCENARIO-REPORT-8579-FAMILIES: head/scaler/temperature bytes have their own seal."""
    from carnot.reporting import v737_contract_methods as h

    original = h.checked
    monkeypatch.setattr(
        h,
        "checked",
        lambda ref: False if ref.get("path", "").endswith("frozen_heads.json") else original(ref),
    )
    result = h.historical(e.protocol())
    assert result["learning_inputs_ready_score"] == 0
    assert result["resident_inputs_ready_score"] == 0


def test_manifest_cannot_drop_checks(authority, tmp_path):
    """SCENARIO-VERIFY-8579-CONTROLS: rehashed validation cannot omit required commands."""
    from carnot.reporting import v737_contract_report as report

    raw = tmp_path / "raw"
    work = e.measure(authority, raw)
    plan = raw / "execution_manifest.json"
    e.atomic_json(
        plan, dict(commands=[dict(name="required_check", argv=["real-check"], expected=0)])
    )
    work["execution_manifest_reference"] = e.reference(plan)
    e.atomic_json(raw / "measurement.json", work)
    checks = raw / "checks.json"
    e.atomic_json(checks, dict(receipts=[]))
    value = report.build(work, e.reference(checks), raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"


@pytest.mark.parametrize("number", [8554, 8571, 8572])
def test_missing_family_isolation(tmp_path, number):
    """SCENARIO-REPORT-8579-FAMILIES: only consumers of missing original bytes lose readiness."""
    p = deepcopy(e.protocol())
    p["original_producers"][str(number)]["path"] = str(tmp_path / "absent")
    r = e.historical(p)
    assert r["learning_inputs_ready_score"] == int(number != 8554)
    assert r["resident_inputs_ready_score"] == int(number == 8554)
    assert any(g["upstream"] == f"exp{number}" for g in r["gate_check_summary"])


@pytest.mark.parametrize("mode", ["resident", "operand_hash", "registration"])
def test_additional_protocol_seals(monkeypatch, mode):
    """SCENARIO-REPORT-8579-AUTHORITY: rehashed registration cannot change inherited operands."""
    original = e.read

    def changed(ref):
        p = deepcopy(original(ref))
        if ref["path"] == str(e.ROOT / e.PROTOCOL):
            if mode == "resident":
                p["resident_session"]["bootstrap_seed"] += 1
            if mode == "operand_hash":
                p["inherited_operand_checks"]["objective_updates"]["sha256"] = "forged"
            if mode == "registration":
                p["literature_reference"]["sha256"] = "forged"
        return p

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()


def test_frozen_authority_and_primitive_tamper(authority, tmp_path):
    """SCENARIO-VERIFY-8579-CONTROLS: mutable planning cannot change frozen authority or clocks."""
    from carnot.reporting import v737_contract_report as report

    raw = tmp_path / "raw"
    work = e.measure(authority, raw)
    checks = raw / "checks.json"
    e.atomic_json(checks, dict(receipts=[]))
    output = tmp_path / (e.NAME + ".json")
    value = report.build(work, e.reference(checks), raw, output)
    e.atomic_json(output, value)
    (authority / e.ACTIVE).unlink()
    assert report.replay(output)
    work["duration_s"] += 100
    e.atomic_json(raw / "measurement.json", work)
    value = report.build(work, e.reference(checks), raw, output)
    e.atomic_json(output, value)
    assert not report.replay(output)


def test_downstream_gate_spellings():
    """REQ-REPORT-8579: downstream gates and prompts consume the exact registered scalar names."""
    p = e.protocol()
    fields = {
        "methods_ready_score",
        "current_contract_ready_score",
        "learning_inputs_ready_score",
        "resident_inputs_ready_score",
        "canary_inputs_ready_score",
        "runtime_repair_ready_score",
    }
    for task in p["task_map_v737"]:
        for g in task["gated_on"]:
            if g["upstream"] == e.TASK:
                assert g["artifact_field"] in fields
                assert p["gate_field_map"][task["id"]] == task["gated_on"]
                assert e.tasks_digest(p["task_map_v737"]) == e.TASKS_PIN
    assert p["hallutruth_release_admissibility"]["learning_corpus_changed"] is False
