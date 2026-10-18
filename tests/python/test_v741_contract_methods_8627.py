"""REQ-REPORT-8627, REQ-VERIFY-8627: authority and independent immutable inputs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v741_contract_methods as e


@pytest.fixture
def authority(tmp_path):
    """Synthetic private design exercises full parity without repairing real authority."""
    p = e.protocol()
    value = dict(milestone=e.MILESTONE, tasks=p["task_map_v741"])
    (tmp_path / e.ACTIVE).write_text(yaml.safe_dump(value))
    design = tmp_path / e.DESIGN
    design.parent.mkdir(parents=True)
    table = "\n".join(
        f"| {i + 1} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
        for i, t in enumerate(value["tasks"])
    )
    design.write_text(
        "## Exact task contract\n"
        + table
        + "\ncanonical_tasks_sha256: "
        + e.TASKS_PIN
        + "\n<!-- V741_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n<!-- V741_TASK_CONTRACT_END -->"
    )
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
    """SCENARIO-REPORT-8627-AUTHORITY: independent full-object mutations reject."""
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
        value["tasks"][1]["gated_on"][0]["value"] = 99
    if mode == "retirement":
        t["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "count":
        value["tasks"].pop()
    if mode == "order":
        value["tasks"].reverse()
    if mode == "old_active":
        value["milestone"] = "2026.10.739"
    if mode in {"end", "table", "digest"}:
        text = design.read_text()
        if mode == "end":
            text = text.replace("<!-- V741_TASK_CONTRACT_END -->", "")
        if mode == "table":
            text = text.replace("| 1 | exp8627-", "| 99 | exp8627-")
        if mode == "digest":
            text = text.replace(e.TASKS_PIN, "0" * 64)
        design.write_text(text)
    if mode == "missing":
        design.unlink()
    if mode == "rehash":
        t["prompt"] += " forged"
        before = (
            design.read_text()
            .split("```json\n")[0]
            .replace(e.TASKS_PIN, e.tasks_digest(value["tasks"]))
        )
        design.write_text(
            before + "```json\n" + json.dumps(value) + "\n```\n<!-- V741_TASK_CONTRACT_END -->"
        )
    if mode == "stage":
        (authority / e.STAGED).write_text(yaml.safe_dump(dict(value, tasks=[])))
    active.write_text(yaml.safe_dump(value))
    result = e.authority(authority)
    assert result["ready"] == (mode == "valid")
    assert len(result["rows"]) == 13


def test_protocol():
    """SCENARIO-REPORT-8627-INPUTS: unchanged numerical operands have separate pins."""
    p = e.protocol()
    old = e.read(p["inherited_v738_protocol_reference"])
    for key in e.INHERITED:
        assert p[key] == old[key]
        assert p["inherited_operand_checks"][key]["sha256"] == e.canonical_hash(old[key])
    assert len(p["task_map_v741"]) == 13
    assert len(p["literature_method_rows"]) == 6
    assert p["source_roster_reference"]["intended_responses"] == 190
    assert p["installed_entry_2x2"]["bootstrap_seed"] == 7418635
    assert p["arc_observer_protocol"]["horizon"] == 120


@pytest.mark.parametrize(
    "number", [None, 8431, 8580, 8581, 8616, 8617, 8620, 8621, 8622, 8624, 8588]
)
def test_history(tmp_path, number):
    """SCENARIO-REPORT-8627-INPUTS: missing inputs affect only their branch."""
    p = deepcopy(e.protocol())
    if number:
        p["current_producers"][str(number)]["path"] = str(tmp_path / "absent")
    h = e.historical(p)
    assert h["learning_inputs_ready_score"] == 0  # Exp8616 remains quarantined.
    assert h["extraction_inputs_ready_score"] == int(number not in {8617, 8620})
    assert h["deployment_inputs_ready_score"] == int(number not in {8431, 8621, 8622})
    assert h["arc_inputs_ready_score"] == int(number not in {8588, 8624})


def test_seal_and_staging(authority, monkeypatch):
    """SCENARIO-REPORT-8627-AUTHORITY: staging qualifies before an older queue."""
    (authority / e.STAGED).write_bytes((authority / e.ACTIVE).read_bytes())
    (authority / e.ACTIVE).write_text("milestone: 2026.10.738\ntasks: []\n")
    assert e.authority(authority)["ready"]
    monkeypatch.setattr(e, "PROTOCOL_PIN", "sha256:wrong")
    with pytest.raises(ValueError):
        e.protocol()


def test_rehashed_protocol(monkeypatch):
    """SCENARIO-REPORT-8627-INPUTS: repairing envelope hashes cannot change science."""
    original = e.read

    def changed(ref):
        value = original(ref)
        if ref["path"] == str(e.ROOT / e.PROTOCOL):
            value["objective_updates"] = {}
        return value

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError, match="sealed"):
        e.protocol()


def test_replay(authority, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8627-REPLAY: immutable reduction rejects repaired operands."""
    from carnot.reporting import v741_contract_report as report

    raw = tmp_path / "raw"
    original = e.measure(authority, raw, "20261018")
    checks = raw / "checks.json"
    e.atomic_json(checks, dict(receipts=[]))
    output = tmp_path / (e.NAME + ".json")
    work = deepcopy(original)

    def candidate():
        e.atomic_json(raw / "measurement.json", work)
        value = report.build(work, e.reference(checks), raw, output)
        e.atomic_json(output, value)
        return value

    value = candidate()
    assert value["completed_count"] == value["intended_count"] == 13
    assert value["run_date"] == "20261018"
    assert value["target_model"]["evaluated_this_invocation"] is False
    assert report.replay(output)
    value["methods_ready_score"] = 0
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(output, value)
    assert not report.replay(output)
    monkeypatch.setattr(e, "historical", lambda p: deepcopy(original["history"]))
    for mode in ["duration", "history", "code", "runtime", "runtime_ready", "control_tamper"]:
        work = deepcopy(original)
        if mode == "control_tamper":
            work["authority_mutation_rows"][0]["mutation"] = "forged"
        if mode == "duration":
            work["duration_s"] += 1
        if mode == "history":
            work["history"]["learning_inputs_ready_score"] = 1
        if mode == "code":
            work["code_refs"][0]["sha256"] = "wrong"
        if mode == "runtime":
            work["runtime"]["inventory_reference"]["sha256"] = "wrong"
        if mode == "runtime_ready":
            work["runtime"]["ready"] = True
        candidate()
        assert not report.replay(output)
    work = deepcopy(original)
    log = raw / "fake.stdout"
    log.write_text("original")
    e.atomic_json(
        checks,
        dict(
            receipts=[
                dict(
                    name="log", passed=True, stdout_path=str(log), stdout_sha256=e.sha256_file(log)
                )
            ]
        ),
    )
    candidate()
    log.write_text("changed")
    assert not report.replay(output)
    e.atomic_json(checks, dict(receipts=[dict(name="owned_failure", passed=False)]))
    assert candidate()["verdict_class"] == "disqualified"
    assert not report.replay(tmp_path / "absent")


def test_score_rejection(monkeypatch):
    """SCENARIO-REPORT-8627-INPUTS: observed zero differs from missing evidence."""
    original = e.read

    def changed(ref):
        value = original(ref)
        if value.get("experiment_id") == 8593:
            value["resident_measurement_ready_score"] = 0
        return value

    monkeypatch.setattr(e, "read", changed)
    assert e.historical(e.protocol())["resident_inputs_ready_score"] == 0


def test_preserved_authority_missing(tmp_path):
    """SCENARIO-REPORT-8627-INPUTS: activation is not permission to invent old bytes."""
    p = deepcopy(e.protocol())
    p["v738_activation_reference"]["path"] = str(tmp_path / "missing-original")
    h = e.historical(p)
    assert h["learning_inputs_ready_score"] == 0
    assert h["resident_inputs_ready_score"] == 0


def test_current_canary_revision():
    """SCENARIO-REPORT-8627-INPUTS: retain the qualified explicit V738 revision."""
    h = e.historical(e.protocol())
    assert h["canary_inputs_ready_score"] == 1


def test_optional_cached_reader_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8627-INPUTS: optional CUDA input failure cannot crash CPU work."""
    from carnot.verify import cuda_reader_8410 as reader

    def absent(*args):
        raise FileNotFoundError("missing cached evidence")

    monkeypatch.setattr(reader, "cached", absent)
    value = e.runtime(tmp_path, tmp_path / "runtime")
    assert not value["ready"]
    assert value["gate_check_summary"]
    assert value["probe_count"] == 0


def test_missing_canary_revision(tmp_path):
    """SCENARIO-REPORT-8627-INPUTS: missing optional revision leaves CPU gates alone."""
    p = deepcopy(e.protocol())
    p["branch_references"]["canary"][0]["path"] = str(tmp_path / "missing-canary")
    h = e.historical(p)
    assert h["canary_inputs_ready_score"] == 0
    assert h["extraction_inputs_ready_score"] == h["deployment_inputs_ready_score"] == 1


def test_current_branch_registration(authority, tmp_path):
    """SCENARIO-REPORT-8627-INPUTS: old failures never become qualified history."""
    work = e.measure(authority, tmp_path / "sealed-work", "20261018")
    p = work["protocol"]
    assert p["lean_worker_2x2"] == e.read(p["inherited_v739_protocol_reference"])["lean_worker_2x2"]
    assert p["lean_worker_comparisons"]["primary"] == "L16/B1 lean_native versus legacy_native"
    assert len(p["triviaplus_coverage_plan"]["fixed_lexical_scores"]) == 4
    assert work["history"]["learning_inputs_ready_score"] == 0
    assert work["history"]["extraction_inputs_ready_score"] == 1
    assert work["runtime"]["ready"] is False
    assert all(row["passed"] for row in work["authority_mutation_rows"])
    for n in ["8602", "8603", "8609"]:
        assert not p["preserved_v739_failure"][n]["readiness_promoted"]


def test_owned_control_failure(authority, tmp_path):
    """SCENARIO-VERIFY-8627-REPLAY: failed controls disqualify and clear acceptance."""
    from carnot.reporting import v741_contract_report as report

    raw = tmp_path / "owned-failure"
    work = e.measure(authority, raw, "20261018")
    work["authority_mutation_rows"][0]["passed"] = False
    checks = raw / "validation.json"
    e.atomic_json(checks, dict(receipts=[]))
    value = report.build(work, e.reference(checks), raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["methods_ready_score"] == 0
    assert value["acceptance_gates"]["authority_mutations"] is False
    assert value["gate_check_summary"]


@pytest.mark.parametrize(
    "mode", ["roster", "private_missing", "validator_missing", "validator_changed", "zero"]
)
def test_branch_rejection_paths(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8627-INPUTS: custody failures do not become numeric zero evidence."""
    p = deepcopy(e.protocol())
    original = e.read
    if mode == "private_missing":
        p["source_roster_reference"]["predictor_reference"]["path"] = str(tmp_path / "missing")

    def changed(ref):
        value = original(ref)
        if mode == "roster" and value.get("experiment_id") == 8620:
            value["extraction_rows"][0]["question_hash"] = "forged"
        if mode == "zero" and value.get("experiment_id") == 8621:
            value["lean_measurement_ready_score"] = 0
        validator = ref == original(p["current_producers"]["8580"])["current_validator_reference"]
        if validator and mode == "validator_missing":
            raise FileNotFoundError("missing original validator")
        if validator and mode == "validator_changed":
            value["ready"] = False
        return value

    monkeypatch.setattr(e, "read", changed)
    h = e.historical(p)
    branch = (
        "extraction"
        if mode in {"roster", "private_missing"}
        else "deployment"
        if mode == "zero"
        else "canary"
    )
    assert h[branch + "_inputs_ready_score"] == 0
    assert h["gate_check_summary"]


def test_consistent_clock_tamper(authority, tmp_path):
    """SCENARIO-VERIFY-8627-REPLAY: repaired clocks cannot replace the frozen measurement."""
    from carnot.reporting import v741_contract_report as report

    raw = tmp_path / "clock-tamper"
    work = e.measure(authority, raw, "20261018")
    checks = raw / "validation.json"
    e.atomic_json(checks, dict(receipts=[]))
    work["end_ns"] += 1000000000
    work["duration_s"] += 1
    e.atomic_json(raw / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    value = report.build(work, e.reference(checks), raw, output)
    e.atomic_json(output, value)
    assert not report.replay(output)
