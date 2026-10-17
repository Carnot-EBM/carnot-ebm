"""REQ-REPORT-8591, REQ-VERIFY-8591: original operands and explicit rejections."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v738_contract_methods as e


@pytest.fixture
def authority(tmp_path):
    """Private complete objects let each mutation test retain the original authority."""
    p = e.protocol()
    for key, name in [("v738_design_reference", e.DESIGN), ("v738_activation_reference", e.ACTIVE)]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(p[key]["path"]).read_bytes())
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
        "old_active",
        "table",
        "digest",
    ],
)
def test_contract(authority, mode):
    """SCENARIO-REPORT-8591-AUTHORITY: every independent mutation must reject."""
    active, design = authority / e.ACTIVE, authority / e.DESIGN
    value = yaml.safe_load(active.read_bytes())
    t = value["tasks"][0]
    if mode in {"id", "title", "deliverable", "model", "prompt"}:
        t[mode] += " forged"
    if mode == "phase":
        t["phase"] = 99
    if mode == "substrate":
        t["inference_substrate_class"] = "live_llm_inference"
    if mode == "gate":
        value["tasks"][3]["gated_on"][0]["value"] = 99
    if mode == "retirement":
        t["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "count":
        value["tasks"].pop()
    if mode == "order":
        value["tasks"].reverse()
    if mode == "old_active":
        value["milestone"] = "2026.10.737"
    if mode in {"end", "table", "digest"}:
        text = design.read_text()
        text = text.replace("<!-- V738_TASK_CONTRACT_END -->", "") if mode == "end" else text
        text = text.replace("| 1 | exp8591-", "| 99 | exp8591-") if mode == "table" else text
        text = text.replace(e.TASKS_PIN, "0" * 64) if mode == "digest" else text
        design.write_text(text)
    if mode == "missing":
        design.unlink()
    if mode == "rehash":
        t["prompt"] += " forged"
        text = design.read_text().replace(e.TASKS_PIN, e.tasks_digest(value["tasks"]))
        before, _ = text.split("```json\n", 1)
        design.write_text(
            before + "```json\n" + json.dumps(value) + "\n```\n<!-- V738_TASK_CONTRACT_END -->"
        )
    if mode == "stage":
        (authority / e.STAGED).write_text(yaml.safe_dump(dict(value, tasks=[])))
    active.write_text(yaml.safe_dump(value))
    result = e.authority(authority)
    assert result["ready"] == (mode == "valid")
    assert len(result["rows"]) == 11
    assert all(r["effective_independent_groups"] == 0 for r in result["rows"])


def test_protocol():
    """SCENARIO-REPORT-8591-INPUTS: scientific operands remain equal by value and digest."""
    p = e.protocol()
    old = e.read(p["inherited_v737_protocol_reference"])
    for key in e.INHERITED:
        assert p[key] == old[key]
        assert p["inherited_operand_checks"][key]["sha256"] == e.canonical_hash(old[key])
    assert len(p["task_map_v738"]) == 11
    assert p["MODEL_SPECS"] == []
    assert len(p["literature_method_rows"]) == 5
    assert p["arc_v737_manifest"]["refinement_min_overlap_games"] == 3
    assert p["medexpert_audit_plan"]["train_detector"] is False


def test_history():
    """SCENARIO-REPORT-8591-INPUTS: learning and resident inputs qualify independently."""
    h = e.historical(e.protocol())
    assert h["learning_inputs_ready_score"] == h["resident_inputs_ready_score"] == 1
    assert h["canary_inputs_ready_score"] == 0
    assert {8580, 8581, 8431, 8554, 8572} <= {r["experiment_id"] for r in h["families"]}


@pytest.mark.parametrize("number", [8431, 8554, 8572, 8580, 8581])
def test_missing_family(tmp_path, number):
    """SCENARIO-REPORT-8591-INPUTS: missing families lower only affected CPU gates."""
    p = deepcopy(e.protocol())
    p["original_producers"][str(number)]["path"] = str(tmp_path / "absent")
    h = e.historical(p)
    assert h["learning_inputs_ready_score"] == int(number == 8572)
    assert h["resident_inputs_ready_score"] == int(number not in {8431, 8572})


def test_replay(authority, tmp_path):
    """SCENARIO-VERIFY-8591-REPLAY: repaired summaries and owned failures remain falsifiable."""
    from carnot.reporting import v738_contract_report as report

    raw = tmp_path / "raw"
    work = e.measure(authority, raw, "20261018")
    checks = raw / "checks.json"
    e.atomic_json(checks, dict(receipts=[]))
    output = tmp_path / (e.NAME + ".json")
    value = report.build(work, e.reference(checks), raw, output)
    e.atomic_json(output, value)
    assert report.replay(output)
    assert value["run_date"] == "20261018"
    assert value["intended_count"] == value["completed_count"] == 11
    assert value["independent_generalization_score"] == 0
    value["methods_ready_score"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    assert not report.replay(output)
    e.atomic_json(checks, dict(receipts=[dict(name="owned_failure", passed=False)]))
    assert report.build(work, e.reference(checks), raw, output)["verdict_class"] == "disqualified"


def test_seal_and_staged_old_queue(authority, monkeypatch):
    """SCENARIO-REPORT-8591-AUTHORITY: valid staging can precede live activation."""
    (authority / e.STAGED).write_bytes((authority / e.ACTIVE).read_bytes())
    (authority / e.ACTIVE).write_text("milestone: 2026.10.737\ntasks: []\n")
    assert e.authority(authority)["ready"]
    monkeypatch.setattr(e, "PROTOCOL_PIN", "sha256:wrong")
    with pytest.raises(ValueError):
        e.protocol()


def test_rehashed_protocol(monkeypatch):
    """SCENARIO-REPORT-8591-INPUTS: an envelope cannot change inherited numerical rules."""
    original = e.read

    def changed(ref):
        result = original(ref)
        if ref["path"] == str(e.ROOT / e.PROTOCOL):
            result["objective_updates"] = {}
        return result

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError, match="sealed"):
        e.protocol()


def test_canary_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8591-INPUTS: real revision evidence is mandatory for a new consumer."""
    from carnot.reporting import v738_contract_inputs as inputs

    p = e.protocol()
    path = tmp_path / "binding.json"
    descriptor = inputs.canary_descriptor(p)
    e.atomic_json(path, descriptor)
    assert inputs.accept_canary(path)
    assert not inputs.accept_canary(tmp_path / "absent")
    changed = dict(descriptor, evidence_canary={})
    e.atomic_json(path, changed)
    assert not inputs.accept_canary(path)
    e.atomic_json(path, descriptor)
    monkeypatch.setattr(inputs.old, "family", lambda *a: dict(gate_check_summary=[]))
    assert not inputs.accept_canary(path)


def test_score_rejection(monkeypatch):
    """SCENARIO-REPORT-8591-INPUTS: a zero differs from missing qualification evidence."""
    from carnot.reporting import v738_contract_inputs as inputs

    original = inputs.e.read

    def changed(ref):
        value = original(ref)
        if value.get("experiment_id") == 8580:
            value["current_validator_ready_score"] = 0
        return value

    monkeypatch.setattr(inputs.e, "read", changed)
    assert e.historical(e.protocol())["learning_inputs_ready_score"] == 0


def test_primitive_rejections(authority, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8591-REPLAY: cold reconstruction checks source, logs and primitives."""
    from carnot.reporting import v738_contract_report as report

    raw = tmp_path / "raw"
    work = e.measure(authority, raw, "20261017")
    validation = raw / "validation.json"
    e.atomic_json(validation, dict(receipts=[]))
    output = tmp_path / (e.NAME + ".json")
    original = deepcopy(work)

    def candidate():
        e.atomic_json(raw / "measurement.json", work)
        e.atomic_json(output, report.build(work, e.reference(validation), raw, output))

    candidate()
    assert report.replay(output)
    # Repeated negative cases use the already authenticated immutable family result.
    # Real standalone acceptance is exercised separately by the private CLI test.
    monkeypatch.setattr(e, "historical", lambda protocol: deepcopy(original["history"]))
    from carnot.reporting import v738_contract_inputs as inputs

    monkeypatch.setattr(inputs, "accept_canary", lambda path: True)
    for key in ["duration_s", "canary_revision_binding", "history", "code_refs"]:
        work = deepcopy(original)
        if key == "duration_s":
            work[key] += 1
        if key == "canary_revision_binding":
            work[key]["evidence_canary"] = {}
        if key == "history":
            work[key]["learning_inputs_ready_score"] = 0
        if key == "code_refs":
            work[key][0]["sha256"] = "wrong"
        candidate()
        assert not report.replay(output)
    work = deepcopy(original)
    work["canary_revision_binding"]["ready"] = False
    candidate()
    assert not report.replay(output)
    work = deepcopy(original)
    candidate()
    log = Path(work["canary_revision_binding"]["private_cli_receipts"][0]["stdout_path"])
    log.write_text("changed")
    assert not report.replay(output)
    assert not report.replay(tmp_path / "missing")
