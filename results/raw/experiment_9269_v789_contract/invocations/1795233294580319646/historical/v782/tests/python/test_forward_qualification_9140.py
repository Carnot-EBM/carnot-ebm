"""REQ-REPORT-9140-V779 / SCENARIO-REPORT-9140-V779: no partial qualification."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import forward_qualification_9140 as q
from carnot.reporting.current_work_receipt import atomic_json


def test_current_receipt_has_own_revision_identity() -> None:
    value = q.build(
        dict(checks=[], references=[], receipts=[], code_config_hashes=[]), "20261116", 0, 1
    )
    assert value["current_work_receipts"][0]["current_run_id"] == "exp9140"


def test_build_does_not_mutate_gate_evidence() -> None:
    work = dict(
        checks=[],
        references=[],
        receipts=[],
        code_config_hashes=[],
        resource_checks=[dict(passed=False)],
    )
    q.build(work, "20261116", 0, 1)
    assert work["checks"] == []


def test_added_roster_roundtrip_order(tmp_path: Path) -> None:
    work = dict(
        checks=[],
        references=[],
        receipts=[],
        code_config_hashes=[],
        additions={"z.py": ["z.py::test_z"], "a.py": ["a.py::test_a"]},
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, q.build(work, "20261116", 0, 1))
    assert q.replay(path)["passed"]


def test_failed_host_prevents_dependency_reads(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        q, "host_checks", lambda p: [dict(check="disk_free_bytes", passed=False, observed=0)]
    )
    monkeypatch.setattr(
        q, "authenticate", lambda *a: pytest.fail("dependent read after host failure")
    )
    output = tmp_path / (q.NAME + ".json")
    assert q.main(["--output", str(output)]) == 0
    assert not json.loads(output.read_bytes())["gate_check_summary"][0]["passed"]


@pytest.fixture(scope="module")
def authenticated(tmp_path_factory: Any) -> dict[str, Any]:
    return q.authenticate(q.ROOT, tmp_path_factory.mktemp("qualification"))


def test_original_union_projection_and_quarantine(authenticated: dict[str, Any]) -> None:
    work = authenticated
    assert work["checks"][-1]["check"] == "complete_schedule_projection"
    assert work["checks"][-1]["observed"] == 27880.142389201996
    assert not work["checks"][-1]["passed"]
    assert len(work["schedule"]["original_node_manifest"]) == 621
    value = q.build(work, "20261116", 0, 100)
    assert value["verdict_class"] == "blocked"
    assert value["forward_validation_ready_score"] == 0
    assert value["sample_size_budget"]["censored"] == 621
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["qualified_revision_reference"]["original_quarantine_preserved"]
    missing = value["remaining_obligations"]["missing_statements"]
    assert any("forward_evidence_runner_9111.py" in r["path"] and r["missing"] for r in missing)
    assert all(r["source_sha256"] for r in missing)


@pytest.mark.parametrize("operand", ["missing_file", "missing_field", 0, True])
def test_gate_operands(tmp_path: Path, monkeypatch: Any, operand: Any) -> None:
    if operand != "missing_file":
        path = tmp_path / q.UPSTREAM
        atomic_json(path, {} if operand == "missing_field" else dict(schedule_ready_score=operand))
        monkeypatch.setattr(q, "PIN", q.s.ref(path)["sha256"])
    work = q.authenticate(tmp_path, tmp_path / "raw")
    row = work["checks"][-1]
    assert type(row["observed"]) is type(operand) and row["observed"] == operand
    assert row["passed"] is False
    value = q.build(work, "20261116", 0, 1)
    assert value["honest_verdict"] == "complete_blocked_upstream"


@pytest.mark.parametrize("mutation", ["headline", "checksum", "plan", "checks"])
def test_rehashed_tamper_refused(
    authenticated: dict[str, Any], tmp_path: Path, mutation: str
) -> None:
    value = q.build(deepcopy(authenticated), "20261116", 0, 1)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert q.replay(path)["passed"]
    if mutation == "headline":
        value["forward_validation_ready_score"] = 1
    elif mutation == "checksum":
        value["reproducibility_checksum"] = "bad"
    elif mutation == "plan":
        value["replay_input"]["work"]["schedule"]["original_node_manifest"].pop()
    else:
        value["replay_input"]["work"]["checks"][-1]["passed"] = True
    if mutation != "checksum":
        value["reproducibility_checksum"] = q.s.canonical_hash(value["replay_input"])
    atomic_json(path, value)
    with pytest.raises(ValueError):
        q.replay(path)


def test_private_cli_refusal_cold_replay_and_tamper(tmp_path: Path) -> None:
    output = tmp_path / (q.NAME + ".json")
    logs = tmp_path / "logs"
    argv = q.r.foreign_argv(
        q.CLI, "--input-root", str(tmp_path / "absent"), "--output", str(output)
    )
    assert q.s.prior.run_check("refusal", argv, logs)["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and not value["required_checks_passed"]
    replay = q.r.foreign_argv(q.CLI, "--replay", str(output))
    assert q.s.prior.run_check("cold", replay, logs)["passed"]
    value["forward_validation_ready_score"] = 1
    atomic_json(output, value)
    assert q.s.prior.run_check("tamper", replay, logs, expected=1)["passed"]
    assert q.main(["--date", "bad"]) == 1


def test_normal_main_and_checks(
    authenticated: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(q, "authenticate", lambda *a: deepcopy(authenticated))
    monkeypatch.setattr(q, "verify", lambda *a: None)
    real_check = q.s.prior.run_check

    def check(name: str, argv: list[str], raw: Path, *args: Any, **kwargs: Any) -> Any:
        if name in {"repository_health", "collect_additions"}:
            argv = [str(q.ROOT / ".venv/bin/python"), "-c", "print('unit orchestration control')"]
        return real_check(name, argv, raw, *args, **kwargs)

    monkeypatch.setattr(q.s.prior, "run_check", check)
    output = tmp_path / (q.NAME + ".json")
    assert q.main(["--output", str(output)]) == 0
    assert q.main(["--replay", str(output)]) == 0
    assert q.terminal(output, tmp_path / "terminal")["passed"]
    assert q.commands(tmp_path)


def test_verification_covers_only_owned_code(tmp_path: Path, monkeypatch: Any) -> None:
    class Tracer:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def combine(self, **kwargs: Any) -> None:
            pass

        def save(self) -> None:
            pass

        def analysis2(self, path: str) -> Any:
            return path, [1], [], [], ""

    monkeypatch.setattr(q, "Coverage", Tracer)
    monkeypatch.setattr(q.s.prior, "run_check", lambda *a, **k: dict(passed=True))
    work: dict[str, Any] = dict(receipts=[], references=[])
    (tmp_path / "owned_events.jsonl").write_text("")
    q.verify(work, tmp_path)
    assert work["coverage"]["statement_percent"] == 100
    assert work["coverage"]["excluded_statements"] == 0
    assert len(work["coverage"]["files"]) == len(q.OWNED)


def test_resource_limits_are_gates(tmp_path: Path, monkeypatch: Any) -> None:
    assert all(row["passed"] for row in q.host_checks(tmp_path))
    monkeypatch.setattr(q.r.shutil, "disk_usage", lambda p: type("Disk", (), {"free": 0})())
    rows = q.host_checks(tmp_path)
    assert rows[0]["observed"] == 0 and rows[0]["passed"] is False


def test_authority_and_projection_drift(
    authenticated: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    original = q.r.replay
    monkeypatch.setattr(q.r, "replay", lambda p: dict(passed=True))
    monkeypatch.setattr(q.s, "authenticate", lambda *a: dict(checks=[dict(passed=False)]))
    assert not q.authenticate(q.ROOT, tmp_path)["checks"][0]["passed"]
    altered = deepcopy(authenticated)
    altered["checks"] = [c for c in altered["checks"] if c["passed"]]
    altered["schedule"]["units"] = "minutes"
    monkeypatch.setattr(q.s, "authenticate", lambda *a: altered)
    with pytest.raises(ValueError, match="schedule_drift"):
        q.authenticate(q.ROOT, tmp_path)
    monkeypatch.setattr(q.r, "replay", original)


def test_replay_receipt_streams(authenticated: dict[str, Any], tmp_path: Path) -> None:
    work = deepcopy(authenticated)
    receipt = q.s.prior.run_check(
        "actual", [str(q.ROOT / ".venv/bin/python"), "-c", "print('actual')"], tmp_path
    )
    work["receipts"].append(receipt)
    path = tmp_path / "candidate.json"
    atomic_json(path, q.build(work, "20261116", 0, 1))
    assert q.replay(path)["passed"]


def test_added_nodes_do_not_replace_originals(authenticated: dict[str, Any]) -> None:
    work = deepcopy(authenticated)
    node = TEST_NODE = "revision.py::test_current"
    work["additions"] = {"revision.py": [node]}
    work["revision_events"] = [
        dict(nodeid=node, when=p, outcome="passed", duration=0.01)
        for p in ["setup", "call", "teardown"]
    ]
    value = q.build(work, "20261116", 0, 1)
    assert value["sample_size_budget"]["intended"] == 622
    assert value["sample_size_budget"]["completed"] == 1
    assert value["sample_size_budget"]["censored"] == 621
    assert value["rows"][-1]["unit_id"] == TEST_NODE
    assert value["forward_validation_ready_score"] == 0


def test_rejects_replacement_schedule_authority(
    authenticated: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    path = tmp_path / "candidate.json"
    atomic_json(path, q.build(authenticated, "20261116", 0, 1))
    monkeypatch.setattr(q, "PIN", "different_authority")
    with pytest.raises(ValueError, match="schedule_authority"):
        q.replay(path)


def test_a_feasible_schedule_needs_new_authority(
    authenticated: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    work = deepcopy(authenticated)
    work["schedule"]["budget_equation"].update(feasible=True, projected_total_s=1)
    work["checks"] = [c for c in work["checks"] if c["passed"]]
    value = json.loads((q.ROOT / q.UPSTREAM).read_bytes())
    value["schedule_reference"] = work["schedule"]
    source = tmp_path / q.UPSTREAM
    atomic_json(source, value)
    monkeypatch.setattr(q, "PIN", q.s.ref(source)["sha256"])
    monkeypatch.setattr(q.r, "replay", lambda p: dict(passed=True))
    monkeypatch.setattr(q.s, "authenticate", lambda *a: work)
    # The retained directory normally comes from the real authenticator.
    (tmp_path / "raw/inputs").mkdir(parents=True)
    with pytest.raises(ValueError, match="unexpected_feasible_revision"):
        q.authenticate(tmp_path, tmp_path / "raw")


def test_missing_original_source_is_a_gate(
    authenticated: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    work = deepcopy(authenticated)
    work["checks"] = [c for c in work["checks"] if c["passed"]]
    work["historical"]["replay_input"]["work"]["historical_work"]["evidence"]["snapshots"][
        "old_validator"
    ]["path"] = str(tmp_path / "absent.py")
    monkeypatch.setattr(q.s, "authenticate", lambda *a: work)
    monkeypatch.setattr(q.r, "replay", lambda p: dict(passed=True))
    refused = q.authenticate(q.ROOT, tmp_path)
    assert refused["checks"][-1]["observed"] == "missing_file"
