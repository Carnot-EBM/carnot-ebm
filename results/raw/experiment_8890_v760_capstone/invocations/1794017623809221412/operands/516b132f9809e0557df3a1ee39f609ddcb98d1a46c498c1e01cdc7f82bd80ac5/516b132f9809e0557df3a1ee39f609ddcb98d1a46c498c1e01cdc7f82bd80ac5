"""REQ-REPORT-8865; SCENARIO-REPORT-8865-CONTRACT: honest complete accounting."""

from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from coverage import Coverage

import pytest

from carnot.reporting import v758_capstone as c


def validation(passed=True):
    return dict(
        receipts=[dict(name="unit", passed=passed)], coverage=dict(complete=passed, percent=100)
    )


def test_custody_cache(tmp_path):
    """REQ-REPORT-8865: duplicate custody saves work without accepting changed bytes."""
    path = tmp_path / "input.json"
    path.write_text("original")
    first = c.reference(path)
    saved = c.retain(path, tmp_path / "copies")
    assert c.retain(path, tmp_path / "copies") == saved
    path.write_text("changed")
    assert c.reference(path) != first
    assert c.reference(tmp_path / "absent")["sha256"] is None
    Path(saved["path"]).write_text("forged")
    path.write_text("original")
    with pytest.raises(ValueError, match="immutable"):
        c.retain(path, tmp_path / "copies")


def test_admission(tmp_path):
    """REQ-REPORT-8865: identity, validation, missing custody and changed bytes all reject."""
    path = tmp_path / "experiment_8853_control.json"
    value = dict(
        experiment_id=8853,
        task_id="exp8853-control",
        honest_verdict="complete_null_control",
        verdict_class="null",
        required_checks_passed=True,
        owned_statement_coverage=dict(complete=True, percent=100),
        validation_receipts=[dict(passed=True)],
    )
    c.atomic_json(path, value)
    assert "custody_missing" in c.admit(path, value["task_id"])["errors"][0]
    source = tmp_path / "raw.json"
    source.write_text("raw")
    value["raw_shard_hashes"] = [c.reference(source)] * 2
    value["validation_receipts"].append(
        dict(passed=True, log_path=str(source), log_sha256=c.reference(source)["sha256"])
    )
    c.atomic_json(path, value)
    assert c.admit(path, value["task_id"])["admitted"]
    assert not c.admit(path, "exp8853-wrong")["admitted"]
    source.write_text("changed")
    assert not c.admit(path, value["task_id"])["admitted"]
    assert not c.admit(tmp_path / "missing", "missing")["admitted"]


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    raw = tmp_path_factory.mktemp("capstone8865")
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ninclude=\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in c.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    data = raw / ".coverage"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(config),
            "--data-file",
            str(data),
            "--parallel-mode",
            str(Path(__file__).absolute()),
            str(raw),
        ],
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        text=True,
        timeout=420,
    )
    assert child.returncode == 0, "frozen-custody child failed; see forwarded output"
    combined = Coverage(data_file=str(data), config_file=str(config))
    combined.combine(data_paths=[str(raw)], strict=True, keep=True)
    if current := Coverage.current():
        current.get_data().update(combined.get_data())
    work = json.loads((raw / "work.json").read_bytes())
    # Producer admission rechecks all raw bindings; repeated snapshot lists only slow ledger mutations.
    for slot in work["slots"]:
        slot["snapshots"] = [slot["primary_reference"]]
    c.atomic_json(raw / "work.json", work)
    by_id = {r["experiment_id"]: r for r in work["slots"]}

    class UnitDigests(dict):
        def clear(self):
            pass

    def measured(value):
        return deepcopy(by_id[value["experiment_id"]]["reduction"])

    def admitted(path, task_id):
        slot = next(r for r in work["slots"] if r["task_id"] == task_id)
        value = dict(slot["producer"], rows=[None] * slot["producer"]["reported_completed_rows"])
        return dict(admitted=slot["admitted"], value=value)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c, "_digests", UnitDigests())
        patch.setattr(c, "reduce", measured)
        patch.setattr(c, "admit", admitted)

        # Upstream raw trees are already immutable; the real producer also tests copying them.
        def existing_operand(path, directory):
            return dict(c.reference(path), source_path=str(path.absolute()))

        patch.setattr(c, "retain", existing_operand)
        patch.setattr(c.shared, "retain", existing_operand)
        yield work


def test_accounting(frozen):
    """SCENARIO-REPORT-8865-CONTRACT: all slots survive a failed scientific branch."""
    value = c.build(frozen, validation())
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8853, 8866))
    assert value["completed_count"] == 13
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["independent_stream_count"] == 24
    assert value["actual_llm_call_count"] == 0
    assert value["independent_generalization_score"] == 0
    assert not value["activation_authorized"] and not value["publication_authorized"]
    slots = {r["experiment_id"]: r for r in value["rows"]}
    assert slots[8863]["disposition"] in {"conductor_pre_gate", "conductor_cascade_skip"}
    assert slots[8863]["verdict_class"] == "blocked"
    assert slots[8860]["verdict_class"] == "null"
    assert slots[8865]["experiment_id"] == 8865
    assert value["historical_v757"]["verdict_class"] == "blocked"
    assert value["historical_v757"]["disqualified_task_ids"]
    assert set(value) <= set(value["field_principles"])
    assert all(
        r["next_decision"] and r["evidence_needed_to_reopen"] for r in value["branch_conclusions"]
    )
    assert c.build(frozen, validation(False))["verdict_class"] == "disqualified"


def test_replay(frozen, tmp_path):
    """REQ-REPORT-8865: repaired outer hashes cannot authenticate false reductions."""
    path = tmp_path / "candidate.json"
    value = c.build(frozen, validation())
    c.atomic_json(path, value)
    assert c.replay(path)["passed"]
    for key in ["independent_stream_count", "independent_draw_count", "completed_count"]:
        changed = deepcopy(value)
        changed[key] += 1
        changed["reproducibility_checksum"] = c.checksum(changed)
        c.atomic_json(path, changed)
        assert not c.replay(path)["passed"]
    path.write_text("{")
    assert not c.replay(path)["passed"]


def test_contract_and_validation(frozen, tmp_path, monkeypatch):
    """REQ-REPORT-8865: absent authority is blocked, owned failures disqualify."""
    bad = deepcopy(frozen)
    bad["contract"]["ready"] = False
    assert c.build(bad, validation())["reconciliation_ready_score"] == 0
    monkeypatch.setattr(c.authority, "PINS", {c.authority.CAPSTONE: "bad"})
    with pytest.raises(ValueError, match="historical_authority"):
        c.freeze(c.ROOT, tmp_path / "bad", "20261106")


def test_reductions(frozen):
    """REQ-REPORT-8865: independently reduced benefit stays separate from integrity."""
    slots = {r["experiment_id"]: r for r in frozen["slots"]}
    audit = slots[8860]["reduction"]["registered_estimands"]
    assert audit["future_gain_score"] == audit["addition_benefit_score"] == 1
    assert audit["retention_pass_score"] == 1
    assert audit["true_label_specificity_score"] == audit["causal_benefit_score"] == 0
    assert slots[8859]["reduction"]["registered_estimands"]["split_protocol_benefit_score"] == 0
    assert slots[8861]["reduction"]["cpu_request_count"] == 2560
    assert slots[8862]["reduction"]["cpu_cost_window_count"] == 60


def test_rehashed_work(frozen, tmp_path):
    """SCENARIO-REPORT-8865-CONTRACT: changed custody cannot repair itself."""
    for mode in ["tasks", "reduction", "summary", "code"]:
        work = deepcopy(frozen)
        if mode == "tasks":
            work["contract"]["tasks"][0]["title"] += "forged"
        elif mode == "reduction":
            next(r for r in work["slots"] if r["admitted"])["reduction"][
                "natural_generalization_count"
            ] = 1
        elif mode == "summary":
            work["slots"][0]["producer"]["reported_completed_rows"] += 1
        else:
            work["code"][0]["sha256"] = "sha256:bad"
        c.atomic_json(tmp_path / "work.json", work)
        value = c.build(frozen, validation())
        value["work_reference"] = c.reference(tmp_path / "work.json")
        c.atomic_json(tmp_path / "candidate.json", value)
        assert not c.replay(tmp_path / "candidate.json")["passed"]


def test_cli(frozen, tmp_path):
    """REQ-REPORT-8865: a real child reads private evidence outside the checkout."""
    import os
    import subprocess
    import sys
    from coverage import Coverage
    from carnot.reporting.v758_capstone_runner import main, evidence_path

    assert evidence_path("/tmp/exp8839-global/receipt.json").name == "repository_health.json"
    assert evidence_path(path := tmp_path / "candidate.json") == path
    assert main(["--date", "bad"]) == 2
    assert main(["--replay", str(tmp_path / "missing")]) == 1
    c.atomic_json(path, c.build(frozen, validation()))
    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ninclude=\n"
        + "".join("    " + str(c.ROOT / p) + "\n" for p in c.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    data = tmp_path / ".coverage"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(config),
            "--data-file",
            str(data),
            "--parallel-mode",
            str(c.ROOT / c.CLI),
            "--replay",
            str(path),
        ],
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        text=True,
        timeout=420,
    )
    assert result.returncode == 0, "outside-checkout CLI replay failed; see forwarded output"
    child = Coverage(data_file=str(data), config_file=str(config))
    child.combine(data_paths=[str(tmp_path)], strict=True, keep=True)
    if current := Coverage.current():
        current.get_data().update(child.get_data())


def test_fixed_child_deadlines(tmp_path, monkeypatch):
    """REQ-REPORT-8865: large custody readers get fixed budgets before they start."""
    from carnot.reporting import v758_capstone_runner as runner

    monkeypatch.setattr(
        runner, "base_check", lambda name, argv, raw, deadline, scope: dict(deadline=deadline)
    )
    assert runner.run_check("cold_replay", [], tmp_path)["deadline"] == 600
    assert runner.run_check("ruff_check", [], tmp_path)["deadline"] == 120


def test_reducer_rejection(tmp_path, monkeypatch):
    """REQ-REPORT-8865: each raw join and independently computed headline can fail."""
    from carnot.reporting import v758_capstone_reduction as r
    from carnot.reporting import linear_benefit_audit_8860 as audit
    from carnot.reporting import split_certificate_evaluation_8859 as split

    raw = tmp_path / "census.json"
    c.atomic_json(
        raw, dict(rows=[], window_rows=[], clocks=[], installation_s=0, preparation_cost_rows=[])
    )
    ref = c.reference(raw)
    draws = tmp_path / "draws.json"
    c.atomic_json(draws, dict(replicates=[]))
    monkeypatch.setattr(audit, "reduce", lambda work: dict(rows=[]))
    for n, message in [
        (8859, "certificate"),
        (8860, "learning"),
        (8861, "request"),
        (8862, "durable"),
    ]:
        with pytest.raises(ValueError, match=message):
            r.reduce(
                dict(
                    experiment_id=n,
                    rows=[dict(cluster_id="control")],
                    raw_shard_hashes=[ref],
                    work_reference=ref,
                    primitive_reference=ref,
                    draw_manifest=c.reference(draws),
                )
            )
    monkeypatch.setattr(split, "summarize", lambda rows: dict(split_protocol_benefit_score=0))
    with pytest.raises(ValueError, match="estimand"):
        r.reduce(
            dict(
                experiment_id=8859,
                rows=[],
                raw_shard_hashes=[ref],
                split_protocol_benefit_score=1,
                draw_manifest=dict(path=str(tmp_path / "draws.json")),
            )
        )


def test_missing_and_rejected_slots(frozen, tmp_path, monkeypatch):
    """REQ-REPORT-8865: rejected bytes and missing inputs retain all thirteen slots."""
    slots = {r["task_id"]: r for r in frozen["slots"]}
    monkeypatch.setattr(
        c.shared,
        "outcome",
        lambda task, *args: dict(
            deepcopy(slots[task["id"]]), producer={}, disposition="admitted_primary"
        ),
    )
    monkeypatch.setattr(
        c,
        "admit",
        lambda path, task: dict(
            admitted=task != "exp8853-branch-contract", errors=[], references=[]
        ),
    )

    def rejected(value):
        raise ValueError("independent_reduction_failed")

    monkeypatch.setattr(c, "reduce", rejected)
    original = c.authority.authority
    monkeypatch.setattr(c.authority, "authority", lambda root: dict(original(root), tasks=[]))
    work = c.freeze(c.ROOT, tmp_path / "rejected", "20261106")
    assert all(not r["admitted"] for r in work["slots"])
    assert any(r["missing_reason"] == "rejected_producer" for r in work["edges"])
    assert c.build(work, validation())["verdict_class"] == "blocked"


def test_replay_controls(frozen, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8865-CONTRACT: admission, authority and reductions must survive replay."""
    path = tmp_path / "candidate.json"
    value = c.build(frozen, validation())
    c.atomic_json(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(c.authority, "PIN", "bad")
        assert not c.replay(path)["passed"]
    with monkeypatch.context() as patch:
        patch.setattr(c.authority, "authority", lambda root: dict(tasks=[], ready=False))
        assert not c.replay(path)["passed"]
    with monkeypatch.context() as patch:
        patch.setattr(c, "admit", lambda *args: dict(admitted=False, value={}))
        assert not c.replay(path)["passed"]
    with monkeypatch.context() as patch:
        original = c.admit

        def summary_drift(*args):
            checked = original(*args)
            checked["value"]["rows"] = []
            return checked

        patch.setattr(c, "admit", summary_drift)
        assert not c.replay(path)["passed"]
    with monkeypatch.context() as patch:

        def invalid_reduction(value):
            raise ValueError("failed_control")

        patch.setattr(c, "reduce", invalid_reduction)
        assert not c.replay(path)["passed"]


def test_replay_bound_work(frozen, tmp_path):
    """REQ-REPORT-8865: newly hashed work still needs the original contract and operands."""
    import shutil

    shutil.copytree(Path(frozen["raw"]) / "authority_view", tmp_path / "authority_view")
    for mode in ["tasks", "code", "edge", "class"]:
        work = deepcopy(frozen)
        work["raw"] = str(tmp_path)
        if mode == "tasks":
            work["contract"]["tasks"][0]["title"] += "forged"
        elif mode == "code":
            work["code"][0]["sha256"] = "sha256:bad"
        elif mode == "edge":
            work["edges"][0]["observed"] = 999
        else:
            work["slots"][0]["verdict_class"] = "positive"
        c.atomic_json(tmp_path / "work.json", work)
        value = c.build(work, validation())
        c.atomic_json(tmp_path / "candidate.json", value)
        assert not c.replay(tmp_path / "candidate.json")["passed"]


def test_replay_rejected_gate(frozen, tmp_path, monkeypatch):
    """REQ-REPORT-8865: a positive scalar from a rejected producer cannot open a gate."""
    import shutil

    work = deepcopy(frozen)
    work["raw"] = str(tmp_path)
    slot = next(r for r in work["slots"] if r["experiment_id"] == 8854)
    slot.update(admitted=False, reduction={})
    for edge in work["edges"]:
        if edge["upstream"] == slot["task_id"]:
            edge.update(passed=False, missing_reason="rejected_producer")
    shutil.copytree(Path(frozen["raw"]) / "authority_view", tmp_path / "authority_view")
    original = c.admit

    def rejected(path, task):
        checked = original(path, task)
        if task == slot["task_id"]:
            checked["admitted"] = False
        return checked

    monkeypatch.setattr(c, "admit", rejected)
    c.atomic_json(tmp_path / "work.json", work)
    c.atomic_json(tmp_path / "candidate.json", c.build(work, validation()))
    assert c.replay(tmp_path / "candidate.json")["passed"]


if __name__ == "__main__":
    # The full historical census has a large temporary JSON decode; isolate it from pytest's RSS ledger.
    with pytest.MonkeyPatch.context() as patch:

        def retained(path, directory):
            return dict(c.reference(path), source_path=str(path.absolute()))

        patch.setattr(c, "retain", retained)
        patch.setattr(c.shared, "retain", retained)
        c.freeze(c.ROOT, Path(sys.argv[1]), "20261106")
