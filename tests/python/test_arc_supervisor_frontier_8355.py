"""REQ-REPORT-8355 / REQ-VERIFY-8355: qualify unchanged readers and real children."""

from copy import deepcopy
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest
import test_arc_supervisor_frontier_8342 as historical_tests

from carnot.reporting import arc_supervisor_frontier_8355 as e
from carnot.reporting import arc_supervisor_frontier_8342 as prior
from carnot.reporting import v718_replay_history as history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child
from test_arc_supervisor_frontier_8342 import (
    test_authentication as test_authentication,
    test_children_and_cli as test_children_and_cli,
    test_failure_boundaries as test_failure_boundaries,
    test_finding_policy as test_finding_policy,
    test_frozen_protocol_failure as test_frozen_protocol_failure,
    test_measure_build_replay_and_block as test_measure_build_replay_and_block,
    test_missing_private_and_protocol as test_missing_private_and_protocol,
    test_missing_tools_and_registry as test_missing_tools_and_registry,
    test_preserved_publication_findings as test_preserved_publication_findings,
    test_qualified_empty_and_native_delta as test_qualified_empty_and_native_delta,
    test_v719_cli_and_recovery as test_v719_cli_and_recovery,
    test_v719_measure_and_replay as test_v719_measure_and_replay,
)


@pytest.fixture(autouse=True)
def preserved_v718(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """SCENARIO-VERIFY-8355-EXECUTION: freeze old authority only for old assertions."""
    if request.node.name in {
        "test_current_reader_and_replay",
        "test_real_cli_and_failure",
        "test_block_and_rehashed_history",
    }:
        return
    historical_tests.preserved_v718.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture(autouse=True)
def frozen_v719(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8355-EXECUTION: original assertions use authentic old bytes."""
    value = prior.json_document(prior.OUTPUT)
    work = prior.json_document(Path(value["work_reference"]["path"]))
    root = tmp_path / "frozen_v719"
    labels = [prior.authority.ACTIVE, prior.authority.DESIGN]
    for label in labels:
        digest = work["source_artifact_hashes"][str(prior.ROOT / label)]
        saved = next(Path(p) for p, d in work["snapshots"].items() if d == digest)
        assert sha256_file(saved) == digest
        target = root / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(saved.read_bytes())
    original = prior.authority.authority

    def historical(_: Path, raw: Path) -> dict[str, Any]:
        with monkeypatch.context() as context:
            context.setattr(history, "design", lambda r, milestone: root / labels[1])
            return dict(original(root, raw))

    monkeypatch.setattr(prior.authority, "authority", historical)


def test_current_reader_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8355-REPLAY: empty natural evidence stays an honest null."""
    raw = tmp_path / "raw"
    work = e.measure(raw, tmp_path)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / e.OUTPUT.name)
    assert not work["failures"]
    assert value["experiment_id"] == 8355 and value["milestone"] == "2026.10.720"
    assert value["honest_verdict"] == "complete_null_no_supervisor_outcomes"
    assert value["arc_reader_ready_score"] == 1 and value["arc_outcome_support_score"] == 0
    assert value["frontier_before"] == value["frontier_after"]
    assert value["current_game_calls"] == value["current_model_calls"] == 0
    assert value["solve_credit_claimed"] is False and value["MODEL_SPECS"] == []
    assert e.replay(value) and not e.replay(dict(value, current_model_calls=1))
    assert all(r["passed"] for r in e.controls(value, tmp_path / "controls"))
    assert [d["verdict_class"] for d in work["historical_dispositions"]] == ["disqualified"] * 2
    plan = e.plan(tmp_path)
    assert {"strict_mypy", "private_E2E017", "private_E2E018_consumers"} <= {
        r["name"] for r in plan
    }
    assert all(r["deadline"] <= 240 for r in plan)


def test_real_cli_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8355-EXECUTION: actual CLI branches and failures retain evidence."""
    main = runpy.run_path(str(e.ROOT / e.CLI))["main"]
    for args in [["--date", "20261008"], ["--private-e2e", "--output", str(e.OUTPUT)]]:
        with pytest.raises(SystemExit):
            main(args)
    output = tmp_path / e.OUTPUT.name
    receipt = child(
        "cli8355",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--private-e2e", "--output", str(output)],
        tmp_path / "children",
        deadline=60,
    )
    assert receipt["passed"]
    value = e.json_document(output)
    assert value["fixture_claim_scope"] and value["arc_reader_ready_score"] == 0
    assert main(["--cold-replay", str(output)]) == 0
    atomic_json(output, dict(value, solve_credit_claimed=True))
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


def test_block_and_rehashed_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8355-REPLAY: missing bytes block; invented history fails replay."""
    raw = tmp_path / "raw"
    work = e.measure(raw, tmp_path)
    value = e.build(work, [], raw, tmp_path / e.OUTPUT.name)
    changed = deepcopy(work)
    changed["historical_dispositions"][0]["verdict_class"] = "positive"
    atomic_json(Path(value["work_reference"]["path"]), changed)
    value["work_reference"]["sha256"] = sha256_file(Path(value["work_reference"]["path"]))
    value["reproducibility_checksum"] = canonical_hash(changed)
    assert not e.replay(value)
    assert not e.replay({})
    with monkeypatch.context() as context:
        context.setattr(e, "read_bound_sidecar", lambda p, s: dict(report=dict(passed=False)))
        bad = e.measure(tmp_path / "bad-terminal", tmp_path)
        assert bad["failures"][-1]["artifact_field"] == "historical_authenticated_bytes"
    monkeypatch.setattr(e, "HISTORICAL", [tmp_path / "missing.json"])
    blocked = e.measure(tmp_path / "blocked", tmp_path)
    assert blocked["failures"][-1]["observed"] is None
    assert (
        e.build(blocked, [], tmp_path / "blocked", tmp_path / e.OUTPUT.name)["verdict_class"]
        == "blocked"
    )
