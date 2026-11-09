"""REQ-REPORT-8957: exact authority and new live receipts are separate gates."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot.reporting import v766_branch_contract as c


def tree(root):
    """Keep queue mutations private so tests cannot activate research work."""
    path = root / c.DESIGN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((c.ROOT / c.DESIGN).read_bytes())
    tasks = c.parse_design(path.read_text(), milestone=c.MILESTONE)[1]
    (root / c.ACTIVE).write_text(
        yaml.safe_dump(dict(yaml.safe_load((c.ROOT / c.ACTIVE).read_bytes()), tasks=tasks))
    )
    return root


def validation(passed=True):
    """Synthetic checks test reduction only; they never qualify production."""
    return dict(
        receipts=[dict(name="fixture", passed=passed)],
        coverage=dict(complete=passed, percent=100 if passed else 0),
        global_test_debt=dict(passed=False, scope="global"),
    )


def test_authority_and_controls(tmp_path):
    """SCENARIO-REPORT-8957-CONTRACT: bind fourteen complete objects and reject edits."""
    root = tree(tmp_path / "root")
    result = c.authority(root)
    assert result["ready"] and len(result["tasks"]) == len(result["rows"]) == 14
    assert c.canonical_hash(result["tasks"]) == c.PIN
    assert all(r["passed"] for r in c.mutations(root, tmp_path / "controls"))
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    queue["milestone"] = "2026.11.765"
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert c.authority(root)["comparison_mode"] == "staged"
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    assert not c.authority(tmp_path / "missing")["ready"]


def test_arc_support_denominator():
    """REQ-REPORT-8957: thirty episodes, three held families, two real applied arms."""
    rows = [
        dict(
            event_identity=str(i),
            invocation_id="new",
            game=f"g{i % 3}",
            seed=i,
            arm=("drop_goal_bias" if i % 2 else "allow_reinduction"),
            status="completed",
            selected=True,
            applied=True,
        )
        for i in range(30)
    ]
    assert c.arc_support(rows)["ready"]
    assert not c.arc_support(rows[:29])["ready"]
    assert not c.arc_support(rows[:1] * 30)["ready"]
    assert not c.arc_support([dict(r, status="censored") for r in rows])["ready"]
    assert not c.arc_support([dict(r, selected=False) for r in rows])["ready"]
    assert not c.arc_support([dict(r, game="one") for r in rows])["ready"]
    assert not c.arc_support([dict(r, arm="drop_goal_bias") for r in rows])["ready"]


def test_freeze_build_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8957-CONTRACT: external zero blocks ARC without blocking custody."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261109")
    value = c.build(work, validation())
    assert value["contract_ready_score"] == 1
    assert value["arc_new_outcomes_ready_score"] == 0
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["intended_count"] == value["completed_count"] == 14
    assert value["independent_count"] == value["numerical_training_calls"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert value["MODEL_SPECS"] == value["model_specs"] == []
    assert value["execution_venue"] == "host"
    assert value["history"]["quarantine"]["flagged_adversarial"] is True
    assert value["history"]["graph_corpus"]["independent_error_count"] == 233
    assert any(r["check"] == "observed_zero" for r in value["gate_check_summary"])
    candidate = tmp_path / (c.NAME + ".json")
    c.atomic_json(candidate, value)
    assert not c.replay(candidate)["passed"]
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    assert c.replay(candidate)["passed"]
    forged = deepcopy(work)
    private_queue = Path(work["raw"]) / "controls" / "missing" / c.ACTIVE
    saved_queue = private_queue.read_bytes()
    private_queue.write_bytes((Path(work["raw"]) / "authority" / c.ACTIVE).read_bytes())
    forged["raw_operands"] = [c.reference(Path(r["path"])) for r in forged["raw_operands"]]
    c.atomic_json(
        Path(work["raw_reference"]["path"]),
        {k: v for k, v in forged.items() if k != "raw_reference"},
    )
    forged["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
    c.atomic_json(candidate, c.build(forged, validation()))
    assert not c.replay(candidate)["passed"]
    private_queue.write_bytes(saved_queue)
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    c.atomic_json(candidate, value)
    changed = deepcopy(value)
    changed["arc_new_outcomes_ready_score"] = 1
    changed["reproducibility_checksum"] = c.checksum(changed)
    c.atomic_json(candidate, changed)
    assert not c.replay(candidate)["passed"]
    assert c.build(work, validation(False))["verdict_class"] == "disqualified"
    changed_work = deepcopy(work)
    changed_work["authority_mutation_rows"][0]["passed"] = False
    assert c.build(changed_work, validation())["verdict_class"] == "disqualified"
    with pytest.raises(FileExistsError):
        c.freeze(tmp_path, Path(work["raw"]), "20261109")
    assert not c.replay(tmp_path / "absent")["passed"]
    checks = validation()
    c.atomic_json(Path(work["raw"]) / "validation.json", checks)
    value["raw_shard_hashes"].append(c.reference(Path(work["raw"]) / "validation.json"))
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(candidate, value)
    assert c.replay(candidate)["passed"]
    for section, field, replacement in [
        ("contract", "ready", False),
        ("history", "authenticated", False),
        ("arc", "ready", True),
    ]:
        forged = deepcopy(work)
        forged[section][field] = replacement
        c.atomic_json(
            Path(work["raw_reference"]["path"]),
            {k: v for k, v in forged.items() if k != "raw_reference"},
        )
        forged["raw_reference"] = c.reference(Path(work["raw_reference"]["path"]))
        forged_value = c.build(forged, checks)
        c.atomic_json(candidate, forged_value)
        assert not c.replay(candidate)["passed"]
    c.atomic_json(
        Path(work["raw_reference"]["path"]), {k: v for k, v in work.items() if k != "raw_reference"}
    )
    c.atomic_json(candidate, value)
    operand = Path(work["raw_operands"][0]["path"])
    operand.write_text("changed immutable control")
    assert not c.replay(candidate)["passed"]


def test_cli(tmp_path, monkeypatch):
    """REQ-REPORT-8957: direct invocation rejects invalid dates and supports cold replay."""
    from carnot.reporting import v755_branch_runner as runner

    monkeypatch.setattr(sys, "argv", [c.CLI, "--date", "bad"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert error.value.code == 2
    monkeypatch.setattr(runner, "run", lambda *args: 0)
    monkeypatch.setattr(
        sys, "argv", [c.CLI, "--date", "20261109", "--output", str(tmp_path / (c.NAME + ".json"))]
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert error.value.code == 0
    monkeypatch.setattr(sys, "argv", [c.CLI, "--cold-replay", str(tmp_path / "absent")])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert error.value.code == 1


def test_retained_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8957: absence, wrong gates and changed quarantine remain explicit failures."""
    from carnot.reporting import v766_branch_evidence as e

    assert not e.history(tmp_path)["authenticated"]
    values = {}
    for n in [8945, 8956, *e.GRAPH_GATES]:
        value = dict(
            experiment_id=n,
            task_id=f"exp{n}-fixture",
            honest_verdict="complete_null_fixture",
            flagged_adversarial=n == 8945,
            required_checks_passed=True,
            owned_statement_coverage=dict(complete=True),
            validation_receipts=[dict(passed=True)],
        )
        if n in e.GRAPH_GATES:
            value[e.GRAPH_GATES[n]] = 1
        if n == 8951:
            value.update(
                independent_count=320,
                independent_error_count=233,
                semantic_error_count=227,
                out_of_graph_error_count=6,
                format_error_count=0,
            )
        values[n] = value
    pins = {}
    for n, value in values.items():
        name = f"results/experiment_{n}_fixture.json"
        c.atomic_json(tmp_path / name, value)
        pins[name] = c.reference(tmp_path / name)["sha256"]
    monkeypatch.setattr(e, "PINS", pins)
    monkeypatch.setattr(
        e.retained,
        "PINS",
        {
            n: (f"experiment_{n}_fixture.json", pins[f"results/experiment_{n}_fixture.json"])
            for n in values
        },
    )
    assert e.history(tmp_path)["authenticated"]
    for n, field, observed in [
        (8946, "matched_source_ready_score", 0),
        (8945, "flagged_adversarial", False),
        (8951, "independent_error_count", 0),
    ]:
        name = f"results/experiment_{n}_fixture.json"
        changed = dict(values[n], **{field: observed})
        c.atomic_json(tmp_path / name, changed)
        pins[name] = c.reference(tmp_path / name)["sha256"]
        assert not e.history(tmp_path)["authenticated"]
        c.atomic_json(tmp_path / name, values[n])
        pins[name] = c.reference(tmp_path / name)["sha256"]
    name = "results/experiment_8946_fixture.json"
    del values[8946]["matched_source_ready_score"]
    c.atomic_json(tmp_path / name, values[8946])
    pins[name] = c.reference(tmp_path / name)["sha256"]
    assert e.history(tmp_path)["failures"][0]["observed"] == "missing_field"
    (tmp_path / name).write_text("changed")
    assert not e.history(tmp_path)["authenticated"]


def test_arc_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8957: absent or conflicting frontier authority cannot qualify ARC."""
    from carnot.reporting import v766_branch_evidence as e

    assert not e.arc_evidence(tmp_path)["ready"]
    producer = dict(frontier_after=dict(ledger_receipt_identities=["wrong"]))
    monkeypatch.setattr(e.arc, "qualified", lambda *args: (producer, []))
    monkeypatch.setattr(e.arc, "read", lambda *args: dict(entries={}))
    assert not e.arc_evidence(tmp_path)["ready"]
    producer["frontier_after"]["ledger_receipt_identities"] = []
    ledger = tmp_path / "ledger.json"
    c.atomic_json(ledger, dict(schema="wrong", entries={}))
    monkeypatch.setattr(e.arc, "authenticate", lambda: ({}, []))
    assert not e.arc_evidence(tmp_path, ledger)["ready"]
    monkeypatch.setattr(e.arc, "delta", lambda *args: ([], {}))
    assert e.arc_evidence(tmp_path, ledger)["failures"][0]["check"] == "observed_zero"
    monkeypatch.setattr(
        e.arc,
        "delta",
        lambda *args: (
            [
                dict(
                    status="completed",
                    selected=True,
                    applied=True,
                    invocation_id="new",
                    game="g",
                    seed=1,
                    arm="drop_goal_bias",
                )
            ],
            {},
        ),
    )
    assert e.arc_evidence(tmp_path, ledger)["failures"][0]["check"] == "insufficient_support"


def test_missing_authority_and_encoded_queue(tmp_path):
    """REQ-REPORT-8957: retired queues are not authority; encoded prompts preserve exact content."""
    root = tree(tmp_path / "root")
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    for task in queue["tasks"]:
        task["prompt"] = c.encode(task["prompt"])
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert c.authority(root)["ready"]
    (root / c.DESIGN).write_text("truncated")
    assert not c.authority(root)["ready"]
    work = c.freeze(tmp_path / "absent", tmp_path / "raw", "20261109")
    value = c.build(work, validation())
    assert value["contract_ready_score"] == 0 and value["gate_check_summary"]


def test_private_cli_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8957-CONTRACT: private CLI writes checked bytes and rejects rehashed claims."""
    from carnot.reporting import v755_branch_runner as runner

    def fixture_validation(raw, tracer):
        tracer.stop()
        tracer.save()
        return validation()

    monkeypatch.setattr(runner, "validate", fixture_validation)
    monkeypatch.setattr(runner, "terminal", lambda candidate, raw: c.replay(candidate))
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    output = tmp_path / (c.NAME + ".json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            c.CLI,
            "--date",
            "20261109",
            "--root",
            str(tree(tmp_path / "authority")),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert error.value.code == 0 and output.is_file()
    monkeypatch.setattr(sys, "argv", [c.CLI, "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert error.value.code == 0
    value = json.loads(output.read_bytes())
    value["full_task_objects"][0]["title"] = "forged"
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(output, value)
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert error.value.code == 1
