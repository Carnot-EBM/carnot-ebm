"""REQ-REPORT-8615, REQ-VERIFY-8615: authority and independent immutable inputs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v740_contract_methods as e


@pytest.fixture
def authority(tmp_path):
    """Synthetic private design exercises full parity without repairing real authority."""
    p = e.protocol()
    value = dict(milestone=e.MILESTONE, tasks=p["task_map_v740"])
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
        + "\n<!-- V740_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(value)
        + "\n```\n<!-- V740_TASK_CONTRACT_END -->"
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
    """SCENARIO-REPORT-8615-AUTHORITY: independent full-object mutations reject."""
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
            text = text.replace("<!-- V740_TASK_CONTRACT_END -->", "")
        if mode == "table":
            text = text.replace("| 1 | exp8615-", "| 99 | exp8615-")
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
            before + "```json\n" + json.dumps(value) + "\n```\n<!-- V740_TASK_CONTRACT_END -->"
        )
    if mode == "stage":
        (authority / e.STAGED).write_text(yaml.safe_dump(dict(value, tasks=[])))
    active.write_text(yaml.safe_dump(value))
    result = e.authority(authority)
    assert result["ready"] == (mode == "valid")
    assert len(result["rows"]) == 12


def test_protocol():
    """SCENARIO-REPORT-8615-INPUTS: unchanged numerical operands have separate pins."""
    p = e.protocol()
    old = e.read(p["inherited_v738_protocol_reference"])
    for key in e.INHERITED:
        assert p[key] == old[key]
        assert p["inherited_operand_checks"][key]["sha256"] == e.canonical_hash(old[key])
    assert len(p["task_map_v740"]) == 12
    assert len(p["literature_method_rows"]) == 6
    assert p["triviaplus_coverage_plan"]["nine_input_head_allowed"] is False
    assert p["lean_worker_2x2"]["primary"] == dict(length=16, batch=1)
    assert p["action_observer_plan"]["horizon"] == 120


@pytest.mark.parametrize("number", [None, 8431, 8554, 8580, 8581, 8593, 8596, 8608, 8609])
def test_history(tmp_path, number):
    """SCENARIO-REPORT-8615-INPUTS: only affected CPU branches lose readiness."""
    p = deepcopy(e.protocol())
    if number:
        p["original_producers"][str(number)]["path"] = str(tmp_path / "absent")
    h = e.historical(p)
    assert h["learning_inputs_ready_score"] == int(number not in {8431, 8554, 8580, 8581})
    assert h["resident_inputs_ready_score"] == int(number not in {8431, 8593, 8596, 8608, 8609})


def test_seal_and_staging(authority, monkeypatch):
    """SCENARIO-REPORT-8615-AUTHORITY: staging qualifies before an older queue."""
    (authority / e.STAGED).write_bytes((authority / e.ACTIVE).read_bytes())
    (authority / e.ACTIVE).write_text("milestone: 2026.10.738\ntasks: []\n")
    assert e.authority(authority)["ready"]
    monkeypatch.setattr(e, "PROTOCOL_PIN", "sha256:wrong")
    with pytest.raises(ValueError):
        e.protocol()


def test_rehashed_protocol(monkeypatch):
    """SCENARIO-REPORT-8615-INPUTS: repairing envelope hashes cannot change science."""
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
    """SCENARIO-VERIFY-8615-REPLAY: immutable reduction rejects repaired operands."""
    from carnot.reporting import v740_contract_report as report

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
    assert value["completed_count"] == value["intended_count"] == 12
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
            work["history"]["learning_inputs_ready_score"] = 0
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
    """SCENARIO-REPORT-8615-INPUTS: observed zero differs from missing evidence."""
    original = e.read

    def changed(ref):
        value = original(ref)
        if value.get("experiment_id") == 8593:
            value["resident_measurement_ready_score"] = 0
        return value

    monkeypatch.setattr(e, "read", changed)
    assert e.historical(e.protocol())["resident_inputs_ready_score"] == 0


def test_preserved_authority_missing(tmp_path):
    """SCENARIO-REPORT-8615-INPUTS: activation is not permission to invent old bytes."""
    p = deepcopy(e.protocol())
    p["v738_activation_reference"]["path"] = str(tmp_path / "missing-original")
    h = e.historical(p)
    assert h["learning_inputs_ready_score"] == 1
    assert h["resident_inputs_ready_score"] == 0


def test_current_canary_revision():
    """SCENARIO-REPORT-8615-INPUTS: retain the qualified explicit V738 revision."""
    h = e.historical(e.protocol())
    assert h["canary_inputs_ready_score"] == 1


def test_optional_cached_reader_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8615-INPUTS: optional CUDA input failure cannot crash CPU work."""
    from carnot.verify import cuda_reader_8410 as reader

    def absent(*args):
        raise FileNotFoundError("missing cached evidence")

    monkeypatch.setattr(reader, "cached", absent)
    value = e.runtime(tmp_path, tmp_path / "runtime")
    assert not value["ready"]
    assert value["gate_check_summary"]
    assert value["probe_count"] == 0


def test_missing_canary_revision(tmp_path):
    """SCENARIO-REPORT-8615-INPUTS: missing optional revision leaves CPU gates alone."""
    p = deepcopy(e.protocol())
    p["original_producers"]["8591"]["path"] = str(tmp_path / "missing-canary")
    h = e.historical(p)
    assert h["canary_inputs_ready_score"] == 0
    assert h["learning_inputs_ready_score"] == h["resident_inputs_ready_score"] == 1


def test_current_branch_registration(authority, tmp_path):
    """SCENARIO-REPORT-8615-INPUTS: old failures never become qualified history."""
    work = e.measure(authority, tmp_path / "sealed-work", "20261017")
    p = work["protocol"]
    assert p["lean_worker_2x2"] == e.read(p["inherited_v739_protocol_reference"])["lean_worker_2x2"]
    assert p["lean_worker_comparisons"]["primary"] == "L16/B1 lean_native versus legacy_native"
    assert len(p["triviaplus_coverage_plan"]["fixed_lexical_scores"]) == 4
    assert work["history"]["learning_inputs_ready_score"] == 1
    assert work["history"]["resident_inputs_ready_score"] == 1
    assert work["runtime"]["ready"] is False
    assert all(row["passed"] for row in work["authority_mutation_rows"])
    for n in ["8602", "8603", "8609"]:
        assert not p["preserved_v739_failure"][n]["readiness_promoted"]


def test_owned_control_failure(authority, tmp_path):
    """SCENARIO-VERIFY-8615-REPLAY: failed controls disqualify and clear acceptance."""
    from carnot.reporting import v740_contract_report as report

    raw = tmp_path / "owned-failure"
    work = e.measure(authority, raw, "20261017")
    work["authority_mutation_rows"][0]["passed"] = False
    checks = raw / "validation.json"
    e.atomic_json(checks, dict(receipts=[]))
    value = report.build(work, e.reference(checks), raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["methods_ready_score"] == 0
    assert value["acceptance_gates"]["authority_mutations"] is False
    assert value["gate_check_summary"]
