"""REQ-REPORT-8653, REQ-VERIFY-8653: sealed authority and independent operands."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v743_contract_methods as e


@pytest.fixture
def authority(tmp_path):
    """Private copies let mutations test authority without changing the live queue."""
    p = e.protocol()
    design = tmp_path / e.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes(Path(p["v743_design_reference"]["path"]).read_bytes())
    (tmp_path / e.ACTIVE).write_text(
        yaml.safe_dump(dict(milestone=e.MILESTONE, tasks=p["task_map_v743"]))
    )
    return tmp_path


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "staged",
        "count",
        "order",
        "id",
        "title",
        "phase",
        "deliverable",
        "model",
        "MODEL_SPECS",
        "inference_substrate_class",
        "gate",
        "prompt",
        "retirement",
        "marker",
        "table",
        "digest",
        "missing",
        "rehashed",
        "old_active",
    ],
)
def test_authority(authority, mode):
    """SCENARIO-REPORT-8653-AUTHORITY: every complete-object mutation must reject."""
    active, design = authority / e.ACTIVE, authority / e.DESIGN
    value = yaml.safe_load(active.read_bytes())
    text = design.read_text()
    if mode in {
        "id",
        "title",
        "phase",
        "deliverable",
        "model",
        "prompt",
        "inference_substrate_class",
    }:
        value["tasks"][0][mode] = "forged"
    if mode == "MODEL_SPECS":
        value["tasks"][0][mode] = ["forged"]
    if mode == "count":
        value["tasks"].pop()
    if mode == "order":
        value["tasks"].reverse()
    if mode == "gate":
        value["tasks"][1]["gated_on"][0]["value"] = 99
    if mode == "retirement":
        value["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "marker":
        text = text.replace("<!-- V743_TASK_CONTRACT_END -->", "")
    if mode == "table":
        text = text.replace("| 1 |", "| 99 |", 1)
    if mode == "digest":
        text = text.replace(e.TASKS_PIN, "0" * 64)
    if mode == "rehashed":
        value["tasks"][0]["prompt"] += "forged"
        text = (
            text.split("```json")[0].replace(e.TASKS_PIN, e.tasks_digest(value["tasks"]))
            + "```json\n"
            + json.dumps(value)
            + "\n```\n<!-- V743_TASK_CONTRACT_END -->"
        )
    if mode == "old_active":
        value["milestone"] = "2026.10.742"
    active.write_text(yaml.safe_dump(value))
    design.write_text(text)
    if mode == "staged":
        (authority / e.STAGED).write_bytes(active.read_bytes())
        active.write_text("milestone: 2026.10.742\ntasks: []\n")
    if mode == "missing":
        design.unlink()
    got = e.authority(authority)
    assert got["ready"] == (mode in {"valid", "staged"})
    assert len(got["rows"]) == 14


def test_protocol_and_circular_gate():
    """SCENARIO-REPORT-8653-OPERANDS: old failure permits new execution qualification only."""
    p = e.protocol()
    old = e.read(p["inherited_v742_protocol_reference"])
    for key in e.INHERITED:
        assert p[key] == old[key]
        assert p["inherited_operand_checks"][key]["sha256"] == e.canonical_hash(old[key])
    h = e.historical(p)
    assert h["learning_inputs_ready_score"] == 0
    assert h["learning_operands_ready_score"] == 1
    assert "learning_execution_ready_score" not in h
    assert all(row["passed"] for row in h["circular_gate_regression_rows"])
    assert (
        h["extraction_inputs_ready_score"]
        == h["deployment_inputs_ready_score"]
        == h["arc_inputs_ready_score"]
        == 1
    )
    assert h["canary_inputs_ready_score"] == 1
    assert len(p["task_map_v743"]) == 14
    assert p["installed_entry_2x2"]["bootstrap_seed"] == 7418635
    assert p["streaming_extraction_protocol"]["p2_imputation"] is False


@pytest.mark.parametrize("branch", ["learning", "extraction", "deployment", "arc", "canary"])
def test_missing_operand(tmp_path, branch):
    """SCENARIO-REPORT-8653-OPERANDS: missing paths block only dependent research branches."""
    p = deepcopy(e.protocol())
    p["operand_references"][branch][0]["path"] = str(tmp_path / "missing")
    h = e.historical(p)
    field = (
        "learning_operands_ready_score" if branch == "learning" else branch + "_inputs_ready_score"
    )
    assert h[field] == 0
    assert h["gate_check_summary"]
    assert any(g["observed"] is None for g in h["gate_check_summary"])


@pytest.mark.parametrize("mode", ["head", "data", "hash", "zero", "inventory", "objective"])
def test_learning_rejects(tmp_path, mode):
    """SCENARIO-REPORT-8653-OPERANDS: checkpoint, labels and test inventory remain mandatory."""
    p = deepcopy(e.protocol())
    if mode in {"head", "data", "inventory"}:
        p[
            {
                "head": "head_checkpoint_reference",
                "data": "learning_data_reference",
                "inventory": "test_inventory_reference",
            }[mode]
        ]["path"] = str(tmp_path / "missing")
    if mode == "hash":
        p["learning_data_reference"]["sha256"] = "wrong"
    if mode == "objective":
        p["objective_updates"]["capacity"] = 99
    if mode == "zero":
        value = e.read(p["current_producers"]["8581"])
        value["online_adapter_ready_score"] = 0
        path = tmp_path / "zero.json"
        e.atomic_json(path, value)
        p["current_producers"]["8581"] = e.reference(path)
    h = e.historical(p)
    assert h["learning_operands_ready_score"] == 0
    assert h["learning_inputs_ready_score"] == 0


def test_protocol_tamper(monkeypatch):
    """SCENARIO-REPORT-8653-AUTHORITY: a repaired envelope cannot change pinned objectives."""
    original = e.read

    def changed(ref):
        value = original(ref)
        if ref["path"] == str(e.ROOT / e.PROTOCOL):
            value["objective_updates"] = {}
        return value

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()


def test_replay_and_owned_failures(authority, tmp_path):
    """SCENARIO-VERIFY-8653-REPLAY: repaired summaries and raw primitives must reject."""
    from carnot.reporting import v743_contract_report as report

    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    original = e.measure(authority, raw, "20261019")
    checks = raw / "checks.json"
    e.atomic_json(checks, dict(receipts=[]))

    def candidate(work):
        e.atomic_json(raw / "measurement.json", work)
        value = report.build(work, e.reference(checks), raw, output)
        e.atomic_json(output, value)
        return value

    value = candidate(original)
    assert value["learning_operands_ready_score"] == 1
    assert "learning_execution_ready_score" not in value
    assert report.replay(output)
    for mode in ["numerical", "history", "clock", "code", "runtime", "controls", "seal"]:
        work = deepcopy(original)
        if mode == "numerical":
            work["numerical"]["probability"] = 0
        if mode == "history":
            work["history"]["learning_operands_ready_score"] = 0
        if mode == "clock":
            work["duration_s"] += 1
        if mode == "code":
            work["code_refs"][0]["sha256"] = "wrong"
        if mode == "runtime":
            work["runtime"]["ready"] = True
        if mode == "controls":
            work["authority_mutation_rows"][0]["passed"] = False
        if mode == "seal":
            work["end_ns"] += 1000000000
            work["duration_s"] += 1
        candidate(work)
        assert not report.replay(output)
    assert not report.replay(tmp_path / "missing")
    shard = raw / "coverage.fixture"
    shard.write_bytes(b"unit fixture for hash custody")
    e.atomic_json(
        checks,
        dict(
            receipts=[
                dict(
                    name="original_coverage_retention",
                    passed=True,
                    original_references=[e.reference(shard)],
                )
            ]
        ),
    )
    candidate(original)
    assert report.replay(output)
    shard.unlink()
    assert not report.replay(output)

    e.atomic_json(checks, dict(receipts=[dict(name="owned_failure", passed=False)]))
    assert candidate(original)["verdict_class"] == "disqualified"
    work = deepcopy(original)
    work["authority_mutation_rows"][0]["passed"] = False
    assert candidate(work)["methods_ready_score"] == 0


@pytest.mark.parametrize("number", [8431, 8580, 8581, 8633, 8637])
def test_missing_current_producer(tmp_path, number):
    """SCENARIO-REPORT-8653-OPERANDS: missing producer fields cannot become observed zeros."""
    p = deepcopy(e.protocol())
    p["current_producers"][str(number)]["path"] = str(tmp_path / "missing")
    h = e.historical(p)
    field = (
        "learning_operands_ready_score"
        if number < 8600
        else "deployment_inputs_ready_score"
        if number == 8633
        else "arc_inputs_ready_score"
    )
    assert h[field] == 0
    assert h["gate_check_summary"]


def test_cohort_and_callback_registration():
    """REQ-REPORT-8653: exact source-bound cohorts retain every original node once."""
    p = e.protocol()
    partition = e.read(p["cohort_partition_reference"])
    cohorts = partition["cohorts"]
    nodes = [n for row in cohorts for n in row["node_ids"]]
    assert len(nodes) == len(set(nodes)) == 64
    assert set(nodes) == set(partition["original_node_ids"])
    assert len(partition["original_runner_node_ids"]) == 27
    assert set(partition["original_runner_node_ids"]) <= set(nodes)
    assert all(e.checked(ref) for ref in partition["source_references"])
    assert {r["cohort"] for r in cohorts} == {"A", "B"}
    assert all(r["measurement_budget_s"] <= 3600 for r in cohorts)
    assert len(p["literature_method_rows"]) == 6
    assert p["installed_cost_cells"] == 120
    assert p["source_study_seed"] == 7418632
    callback = p["reset_callback_payload_protocol"]
    assert callback["historical_error_type"] == "TypeError"
    assert callback["historical_unattributed_resets"] == 236
    assert all(e.checked(ref) for ref in callback["actual_public_frame_references"])
    assert callback["allowed_changes"] == ["serialization", "event_ownership"]
    assert p["reset_provenance_protocol"]["episodes"] == 16


def test_gate_operand_states(tmp_path):
    """SCENARIO-REPORT-8653-OPERANDS: null, missing field and zero are distinct."""
    for name, content, observed, state in [
        ("missing", None, None, "missing_file"),
        ("field", {}, None, "missing_field"),
        ("zero", {"ready_score": 0}, 0, "observed_zero"),
    ]:
        path = tmp_path / name
        if content is not None:
            e.atomic_json(path, content)
        row = e.gate(path, "ready_score", 1, observed)
        assert row["operand_state"] == state
        assert row["observed"] == observed


def test_partition_rehash_rejected(monkeypatch):
    """SCENARIO-VERIFY-8653-REPLAY: duplicated original nodes cannot be repaired by a digest."""
    original = e.read

    def changed(ref):
        value = original(ref)
        if ref["path"].endswith("cohort-partition-budgeted.json"):
            value["cohorts"][0]["node_ids"][0] = value["cohorts"][1]["node_ids"][0]
        return value

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError, match="original_cohort_partition"):
        e.protocol()


def test_measured_budget_and_compact_callback_inputs():
    """REQ-REPORT-8653: measured budgets fit reserves and reset specimens stay immutable."""
    p = e.protocol()
    partition = e.read(p["cohort_partition_reference"])
    for cohort in partition["cohorts"]:
        assert sum(g["child_budget_s"] for g in cohort["groups"]) <= 3300
        for group in cohort["groups"]:
            assert group["child_budget_s"] >= sum(group["measured_historical_child_durations_s"])
    fixture = e.read(p["reset_callback_payload_protocol"]["compact_fixture_reference"])
    assert fixture["recorder_errors"]
    assert fixture["public_observation"]
    assert fixture["source_reference"]["sha256"].startswith("sha256:")
    assert (
        Path(p["reset_callback_payload_protocol"]["compact_fixture_reference"]["path"])
        .stat()
        .st_size
        < 100000
    )
