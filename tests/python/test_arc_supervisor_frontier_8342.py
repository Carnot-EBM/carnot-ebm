"""REQ-REPORT-8342 / REQ-VERIFY-8342: preserve reader assertions and real children."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_frontier_8342 as e
from carnot.reporting import arc_supervisor_execution_8328 as old
from carnot.reporting import v718_replay_history as h
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child
from test_arc_supervisor_frontier_8328 import (
    test_authentication as test_authentication,
    test_children_and_cli as test_children_and_cli,
    test_failure_boundaries as test_failure_boundaries,
    test_measure_build_replay_and_block as test_measure_build_replay_and_block,
    test_missing_tools_and_registry as test_missing_tools_and_registry,
    test_qualified_empty_and_native_delta as test_qualified_empty_and_native_delta,
)
from test_v718_contract_replay_8318 import test_finding_policy as test_finding_policy

Json = dict[str, Any]


@pytest.fixture(autouse=True)
def preserved_v718(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8342-EXECUTION: old assertions read their actual frozen authority."""
    value = old.m.json_document(old.OUTPUT)
    bound = old.m.read_bound_sidecar(
        old.OUTPUT,
        old.OUTPUT.parent
        / "raw"
        / old.NAME
        / "validators"
        / (sha256_file(old.OUTPUT)[7:] + ".json"),
    )
    assert bound["report"]["passed"] and value["verdict_class"] == "disqualified"
    work = old.m.json_document(Path(value["work_reference"]["path"]))
    root = tmp_path / "preserved_v718"
    for label in [old.m.authority.ACTIVE, old.m.authority.DESIGN]:
        digest = work["source_artifact_hashes"][str(old.m.ROOT / label)]
        saved = next(Path(p) for p, d in work["snapshots"].items() if d == digest)
        assert sha256_file(saved) == digest
        target = root / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(saved.read_bytes())
    original = old.m.authority.authority

    def historical(_: Path, raw: Path) -> Json:
        with monkeypatch.context() as context:
            context.setattr(h, "design", lambda r, milestone: root / old.m.authority.DESIGN)
            return dict(original(root, raw))

    monkeypatch.setattr(old.m.authority, "authority", historical)


def test_v719_measure_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8342-FRONTIER: new custody keeps a qualified empty frontier."""
    work = e.measure(tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / e.OUTPUT.name)
    assert not work["failures"]
    assert value["experiment_id"] == 8342 and value["milestone"] == "2026.10.719"
    assert value["honest_verdict"] == "complete_null_no_supervisor_outcomes"
    assert value["frontier_before"] == value["frontier_after"]
    assert value["arc_reader_ready_score"] == 1 and value["arc_outcome_support_score"] == 0
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert e.replay(value) and not e.replay(dict(value, solve_claims=["invented"]))
    assert all(r["passed"] for r in e.controls(value, tmp_path / "controls"))
    plan = e.plan(tmp_path)
    assert {"coverage_report", "private_E2E017", "qualified_consumers"} <= {r["name"] for r in plan}
    assert all(r["deadline"] <= 240 for r in plan)


def test_v719_cli_and_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8342-EXECUTION: CLI, actual failure and recovery paths are covered."""
    main = runpy.run_path(str(e.ROOT / e.CLI))["main"]
    for args in [["--date", "20261008"], ["--private-e2e", "--output", str(e.OUTPUT)]]:
        with pytest.raises(SystemExit):
            main(args)
    output = tmp_path / e.OUTPUT.name
    receipt = child(
        "v719_cli",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--private-e2e", "--output", str(output)],
        tmp_path / "children",
        deadline=60,
    )
    assert receipt["passed"]
    value = e.json_document(output)
    assert value["fixture_claim_scope"] and value["arc_reader_ready_score"] == 0
    assert main(["--cold-replay", str(output)]) == 0
    atomic_json(output, dict(value, new_outcome_count=1))
    assert main(["--cold-replay", str(output)]) == 1
    monkeypatch.setattr(
        e,
        "plan",
        lambda p: [
            dict(
                name="owned_failure",
                argv=[sys.executable, "-u", "-c", "raise SystemExit(7)"],
                deadline=5,
                expected=0,
                scope="owned",
            )
        ],
    )
    assert e.run(tmp_path / "failed" / e.OUTPUT.name, tmp_path) == 1
    monkeypatch.setattr(e, "plan", lambda p: [])
    assert e.run(tmp_path / "recovery" / e.OUTPUT.name, tmp_path) == 0


def test_preserved_publication_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8342-EXECUTION: rejected warnings survive recovery with zero readiness."""
    work = old.m.measure(tmp_path / "raw", tmp_path)
    output = tmp_path / old.OUTPUT.name
    value = old.build(work, [], tmp_path / "raw", output)
    original = old.audit
    calls: list[Json] = []

    def warning(candidate: Path, logs: Path, proof: Json) -> Json:
        report = original(candidate, logs, proof)
        flag = dict(kind="UNKNOWN_CONTROL", severity="warn", detail="private negative control")
        report.update(
            passed=False, findings=[flag], dispositions=[dict(finding=flag, resolved=False)]
        )
        report["receipt"]["passed"] = False
        calls.append(deepcopy(report))
        return report

    monkeypatch.setattr(old, "audit", warning)
    old.publish(value, work, output, tmp_path / "raw")
    published = json.loads(output.read_bytes())
    assert len(calls) == 2 and published["verdict_class"] == "disqualified"
    assert published["flagged_adversarial"] and published["arc_reader_ready_score"] == 0
    assert published["adversarial_findings"][0]["findings"] == calls[0]["findings"]


def test_frozen_protocol_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8342-FRONTIER: changed scientific bytes block current readiness."""
    original = e.sha256_file
    monkeypatch.setattr(
        e,
        "sha256_file",
        lambda p: "sha256:changed" if p == e.ROOT / e.authority.PROTOCOL else original(p),
    )
    work = e.measure(tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / e.OUTPUT.name)
    assert value["verdict_class"] == "blocked" and value["arc_reader_ready_score"] == 0
    assert work["failures"][-1]["artifact_field"] == "protocol_sha256"


def test_missing_private_and_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8342-FRONTIER: absent operands never become a measured zero."""
    work = e.measure(tmp_path / "raw", tmp_path / "absent")
    assert work["failures"][0]["artifact_field"] == "private_scratch_rw"
    monkeypatch.setattr(e.authority, "PROTOCOL", "absent-protocol.json")
    work = e.measure(tmp_path / "missing-protocol", tmp_path)
    assert work["failures"][-1]["observed"] is None
    with monkeypatch.context() as context:
        original = Path.read_bytes
        context.setattr(
            Path,
            "read_bytes",
            lambda p: b"changed" if p.name == "exp8342-private-probe" else original(p),
        )
        work = e.measure(tmp_path / "bad-probe", tmp_path)
        assert work["failures"][0]["artifact_field"] == "private_scratch_rw"
