"""REQ-REPORT-7936-IDENTITY/ORDER/TERMINAL: authenticate before counting."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_v688_receipts as reader
from carnot.reporting import arc_supervisor_v688_refinement as task
from carnot.reporting.arc_supervisor_delta import _source_rows
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7936_v688_arc_supervisor_refinement as cli


def episode(**extra: Any) -> dict[str, Any]:
    """Use explicit identity and clocks so fixture novelty has a clear boundary."""
    return dict(
        game="g1",
        seed=1,
        invocation_id="call1",
        receipt_id="receipt1",
        solve_provenance="live_agent_self_discovery",
        event_timestamp="2026-09-30T12:00:00Z",
        termination={"reason": "finished"},
        trajectory_supervisor={
            "enabled": True,
            "mode": "applied",
            "redirects": [
                {
                    "id": "r1",
                    "arm": "drop_goal_bias",
                    "fired": True,
                    "resolved_by_levelup": True,
                    "actions_to_levelup": 3,
                    "pending_at_resolution": 2,
                }
            ],
        },
        **extra,
    )


def producer(root: Path, rows: Any, **extra: Any) -> Path:
    """Producer hashes authenticate private raw bytes without live game calls."""
    raw = root / "results/raw/live/rows.json"
    atomic_json(raw, {"rows": rows})
    path = root / "results/experiment_9001_arc.json"
    atomic_json(
        path,
        dict(
            run_date="20260930",
            verdict_class="null",
            flagged_adversarial=False,
            source_artifact_hashes={raw.relative_to(root).as_posix(): sha256_file(raw)},
            **extra,
        ),
    )
    return path


def scan(root: Path, paths: list[Path], **extra: Any) -> dict[str, Any]:
    """Hold calendar and identity filters to the same authenticated corpus."""
    return reader.inspect(
        root,
        paths,
        {},
        "20260930",
        "20260930",
        {"event_timestamp": "2026-09-30T10:00:00Z"},
        **extra,
    )


# SCENARIO-REPORT-7936-INGEST; SCENARIO-ARC-7936-SAME-DAY.
def test_same_day_retry_and_clock(tmp_path: Path) -> None:
    row = episode()
    path = producer(tmp_path, [row, row])
    assert _source_rows(tmp_path, path, "20260930", set()) == []
    value = scan(tmp_path, [path])
    assert value["calendar_filter_count"] == 0 and value["identity_filter_count"] == 1
    assert value["rows"][1]["reason"] == "identical_retry"
    assert value["rows"][0]["chronology"] == "after_cutoff"
    assert value["rows"][0]["helped_share"] == 0.5
    assert value["rows"][0]["helped_sole"] is False
    row.pop("event_timestamp")
    path = producer(tmp_path, [row])
    value = scan(tmp_path, [path])
    assert value["identity_filter_count"] == value["chronology_unknown_count"] == 1
    assert not value["rows"][0]["prospective"]
    row["event_timestamp"] = "2026-09-29T08:00:00Z"
    path = producer(tmp_path, [row])
    assert scan(tmp_path, [path])["rows"][0]["chronology"] == "before_cutoff"


# SCENARIO-REPORT-7936-INGEST: every conflicting identity fails closed.
def test_identity_conflicts_and_inventory(tmp_path: Path) -> None:
    row = episode()
    path = producer(tmp_path, [row])
    fresh = scan(tmp_path, [path])
    seen = fresh["receipt_inventory"]
    assert (
        reader.inspect(tmp_path, [path], seen, "20260930", "20260930", {})["identity_filter_count"]
        == 0
    )
    changed = deepcopy(row)
    changed["trajectory_supervisor"]["redirects"][0]["actions_to_levelup"] = 4
    path = producer(tmp_path, [row, changed])
    value = scan(tmp_path, [path])
    assert value["identity_filter_count"] == 0
    assert all(r["reason"] == "conflicting_identity" for r in value["rows"])
    path = producer(tmp_path, [changed])
    assert (
        reader.inspect(tmp_path, [path], seen, "20260930", "20260930", {})["rows"][0]["reason"]
        == "changed_receipt"
    )


# SCENARIO-REPORT-7936-INGEST: unauthenticated and ineligible controls stay visible.
@pytest.mark.parametrize(
    "kind",
    [
        "disqualified",
        "future",
        "shadow",
        "empty",
        "proxy",
        "identity",
        "malformed",
        "unknown",
        "future_clock",
        "hash",
        "missing",
        "escape",
        "bad_producer",
        "bad_raw",
        "hash_type",
        "hashes",
        "sequence",
    ],
)
def test_controls(tmp_path: Path, kind: str) -> None:
    row = episode()
    doc: dict[str, Any] = {}
    if kind == "disqualified":
        doc["verdict_class"] = "disqualified"
    if kind == "future":
        doc["run_date"] = "20261001"
    if kind == "shadow":
        row["trajectory_supervisor"]["mode"] = "shadow"
    if kind == "empty":
        row["trajectory_supervisor"]["redirects"] = []
    if kind == "proxy":
        row["solve_provenance"] = "development_proxy"
    if kind == "identity":
        row.pop("invocation_id")
    if kind == "malformed":
        row["trajectory_supervisor"]["redirects"] = ["bad"]
    if kind == "unknown":
        row["trajectory_supervisor"]["redirects"][0].pop("resolved_by_levelup")
    if kind == "future_clock":
        row["event_timestamp"] = "2026-10-01T00:00:00Z"
    if kind == "sequence":
        row.pop("event_timestamp")
        row.update(event_sequence=3, sequence_scope="run")
    path = producer(tmp_path, [row])
    raw = tmp_path / "results/raw/live/rows.json"
    if kind in {"hash", "bad_raw"}:
        raw.write_text("invalid" if kind == "bad_raw" else "{}")
    if kind == "missing":
        raw.unlink()
    if kind == "escape":
        doc["source_artifact_hashes"] = {"results/raw/../../../escape.json": "sha256:a"}
    if kind == "hash_type":
        doc["source_artifact_hashes"] = {"results/raw/live/rows.json": 42}
    if kind == "hashes":
        doc["source_artifact_hashes"] = []
    base = json.loads(path.read_text())
    base.update(doc)
    if kind == "bad_raw":
        base["source_artifact_hashes"] = {"results/raw/live/rows.json": sha256_file(raw)}
    atomic_json(path, base)
    if kind == "bad_producer":
        path.write_text("[]")
    value = scan(tmp_path, [path])
    assert value["identity_filter_count"] == int(kind in {"unknown", "sequence"})
    if kind == "unknown":
        assert value["rows"][0]["status"] == "unknown"


# SCENARIO-REPORT-7936-REDUCE: censored and unknown observations preserve their fields.
def test_reduction_and_order(tmp_path: Path) -> None:
    row = episode()
    row["trajectory_supervisor"]["redirects"][0]["resolved_by_levelup"] = False
    row["termination"] = {"reason": "time_limit"}
    path = producer(tmp_path, [row])
    value = scan(tmp_path, [path])
    assert value["sample_size_budget"]["censored"] == 1
    assert reader.replay(value) == []
    value["identity_filter_count"] = 2
    assert reader.replay(value) == ["identity_filter_count"]
    assert (
        reader.event_order(
            {"event_sequence": 2, "sequence_scope": "a"},
            {"event_sequence": 1, "sequence_scope": "a"},
            "20260930",
        )
        == "after_cutoff"
    )
    assert (
        reader.event_order(
            {"event_sequence": 1, "sequence_scope": "a"},
            {"event_sequence": 2, "sequence_scope": "a"},
            "20260930",
        )
        == "before_cutoff"
    )
    for clock in ("invalid", "2026-09-30T12:00:00", None):
        assert reader.event_order({"event_timestamp": clock}, {}, "20260930") == "unknown"
    assert (
        reader.event_order(episode(), {"event_timestamp": "2026-09-30T10:00:00"}, "20260930")
        == "unknown"
    )
    path = producer(tmp_path, [episode()])
    doc = json.loads(path.read_text())
    digest = next(iter(doc["source_artifact_hashes"].values()))
    doc["source_artifact_hashes"] = {
        "ignored.txt": "bad",
        "outside.json": "bad",
        "results/raw/live/rows.json": {"sha256": digest[7:]},
    }
    atomic_json(path, doc)
    assert scan(tmp_path, [path])["identity_filter_count"] == 1
    assert (
        reader.inspect(tmp_path, [path], {"raw:" + digest: digest}, "20260930", "20260930", {})[
            "identity_filter_count"
        ]
        == 0
    )


# SCENARIO-REPORT-7936-REDUCE: proposals require the curated sample floor and held-out direction.
@pytest.mark.parametrize(
    "helped,n,decision",
    [
        (False, 60, "propose_deprioritization"),
        (True, 60, "propose_raise_priority"),
        (False, 20, "unchanged"),
    ],
)
def test_decisions(tmp_path: Path, helped: bool, n: int, decision: str) -> None:
    rows = []
    for i in range(n):
        row = episode()
        row.update(game="g" + str(i % 5), invocation_id=str(i), receipt_id=str(i))
        receipt = row["trajectory_supervisor"]
        receipt.update(
            arms_enabled=["drop_goal_bias"],
            arms_used=["drop_goal_bias"],
            stagnations_unredirected=1,
        )
        receipt["redirects"][0]["resolved_by_levelup"] = helped
        rows.append(row)
    value = scan(tmp_path, [producer(tmp_path, rows)])
    assert value["refinement_decisions"][0]["decision"] == decision
    assert bool(value["shared_method_gap"]) == (not helped)
    assert not value["refinement_decisions"][0]["defaults_changed"]


# SCENARIO-REPORT-7936-CLI: private success, required failure and cold replay are real routes.
def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = producer(tmp_path, [episode()])
    output = tmp_path / "delta.json"
    args = ["--reduce-ledger", str(tmp_path), "--producer", str(path)]
    with pytest.raises(SystemExit):
        cli.main(args)
    assert cli.main([*args, "--output", str(output)]) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    value = json.loads(output.read_text())
    value["identity_filter_count"] = 10
    atomic_json(output, value)
    assert cli.main(["--cold-replay", str(output)]) == 1
    monkeypatch.setattr(task, "execute", lambda *_args: 0)
    assert cli.main(["--output", str(output)]) == 0


# SCENARIO-REPORT-7936-PUBLISH: current custody preserves the accepted and seen inventories.
def test_inputs_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task, "ROOT", tmp_path)
    prior = tmp_path / "prior.json"
    inventory = tmp_path / "inventory.json"
    registry = tmp_path / "registry.yaml"
    registry.write_text("games: {}\n")
    atomic_json(inventory, {"seen_receipt_hashes": {}, "cutoff_receipt_hashes": {}})
    atomic_json(
        prior,
        {
            "receipt_inventory_path": str(inventory),
            "verdict_class": "null",
            "flagged_adversarial": False,
            "arc_delta_ready_score": 1,
            "source_artifact_hashes": {str(inventory): sha256_file(inventory)},
        },
    )
    monkeypatch.setattr(task, "PRIOR", prior)
    monkeypatch.setattr(task, "INVENTORY", inventory)
    monkeypatch.setattr(task, "REGISTRY", registry)
    monkeypatch.setattr(
        task, "PINNED", {str(p): sha256_file(p) for p in (prior, inventory, registry)}
    )
    checked = task.inputs()
    assert not checked["failures"]
    prior.write_text("{}")
    assert task.inputs()["failures"]
    registry.unlink()
    assert task.inputs()["failures"][-1]["observed"] == "missing"
    specs = task.commands(tmp_path / "private")
    includes = {arg for s in specs for arg in s["argv"] if arg.startswith("--include=")}
    assert includes == {"--include=" + task.INCLUDE}
    assert any(s["name"] == "e2e_017" for s in specs)
    for spec in specs:
        if spec["name"] in {"e2e_016_fixture-e2e", "e2e_016_cold-replay"}:
            assert spec["argv"][spec["argv"].index("--date") + 1] == "20260929"


# SCENARIO-REPORT-7936-PUBLISH: owned failures zero readiness and terminal bytes are rechecked.
@pytest.mark.parametrize("always_fail", [False, True])
def test_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, always_fail: bool) -> None:
    calls = 0

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return dict(
            spec,
            passed=calls > 2 and not always_fail,
            log_path="private",
            log_sha256="sha256:private",
        )

    monkeypatch.setattr(task, "run", run)
    value = dict(flagged_adversarial=False, gate_check_summary=[], acceptance_gate_results={})
    output = tmp_path / "experiment_7936_v688_arc_supervisor_refinement.json"
    if always_fail:
        with pytest.raises(ValueError, match="terminal_recheck_failed"):
            task.publish(value, output, tmp_path, tmp_path / "raw")
    else:
        task.publish(value, output, tmp_path, tmp_path / "raw")
        assert value["verdict_class"] == "disqualified"
        assert value["arc_evidence_ready_score"] == 0 and not value["flagged_adversarial"]
        assert sha256_file(output) == sha256_file(tmp_path / "terminal_candidate.json")


# SCENARIO-REPORT-7936-PUBLISH: actual readers select the primary with newer sidecars.
def test_resolution(tmp_path: Path) -> None:
    output = tmp_path / "experiment_7936_v688_arc_supervisor_refinement.json"
    atomic_json(
        output, dict(experiment_id=7936, verdict_class="null", honest_verdict="complete_null")
    )
    receipt = task.resolution(output, tmp_path / "raw")
    assert receipt["sha256"] == sha256_file(output)
    assert receipt["conductor_sha256"] == receipt["reconciliation_sha256"]


# SCENARIO-REPORT-7936-PUBLISH: execution reports null, blocked and owned failure separately.
@pytest.mark.parametrize("state", ["null", "blocked", "owned", "coverage"])
def test_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    monkeypatch.setattr(task, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    private = tmp_path / "private"
    private.mkdir()
    path = producer(tmp_path, [episode()])
    # Include eligible, malformed and unsupported producer documents in frozen custody.
    path.rename(path.with_name("experiment_7900_arc.json"))
    malformed = tmp_path / "results/experiment_7901_bad.json"
    malformed.write_text("bad")
    atomic_json(tmp_path / "results/experiment_7902_list.json", [])
    frozen = {"seen_receipt_hashes": {}, "cutoff_receipt_hashes": {}}
    checked = dict(
        prior={},
        inventory=frozen,
        registry={"games": {}},
        checks=[],
        failures=[dict(expected="present", observed="missing")] if state == "blocked" else [],
    )
    monkeypatch.setattr(task, "inputs", lambda: checked)
    monkeypatch.setattr(task.validation, "dependency_hashes", lambda *_args, **_kwargs: {})
    real_hash = task.sha256_file
    monkeypatch.setattr(
        task, "sha256_file", lambda p: real_hash(p) if p.is_file() else "sha256:fixture"
    )

    def commands(directory: Path) -> list[dict[str, Any]]:
        atomic_json(
            directory / "coverage.json",
            {
                "files": {
                    name: {
                        "summary": {"num_statements": 1, "covered_lines": 1},
                        "missing_lines": [],
                    }
                    for name in [*task.MODULES, task.CLI]
                }
            },
        )
        return [
            dict(
                name="required",
                argv=["true"],
                classification="required",
                deadline_s=10,
                expected_exit=0,
                expected_text=None,
            ),
            dict(
                name="health",
                argv=["true"],
                classification="repository_health",
                deadline_s=10,
                expected_exit=0,
                expected_text=None,
            ),
        ]

    monkeypatch.setattr(task, "commands", commands)
    monkeypatch.setattr(
        task,
        "run",
        lambda spec, *_args: dict(
            spec,
            passed=state != "owned",
            command_argv=spec["argv"],
            log_path="private",
            log_sha256="sha256:private",
            exit_code=int(state == "owned"),
        ),
    )
    if state == "coverage":
        monkeypatch.setattr(task.validation, "coverage_complete", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(task, "publish", lambda value, output, *_args: atomic_json(output, value))
    output = tmp_path / "results/experiment_7936_v688_arc_supervisor_refinement.json"
    code = task.execute(output, private)
    value = json.loads(output.read_text())
    assert code == int(state != "null")
    assert value["verdict_class"] == ("disqualified" if state in {"owned", "coverage"} else state)
    assert value["new_level_solves_claimed"] == 0 and value["MODEL_SPECS"] == []
    assert value["milestone"] == "2026.09.688"


# SCENARIO-REPORT-7936-CLI: expected exits also need their frozen failure reasons.
def test_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        task.validation, "run_check", lambda *_args: dict(passed=True, output_tail="wrong")
    )
    assert not task.run(dict(expected_text="missing"), tmp_path, tmp_path)["passed"]
    assert task.run(dict(expected_text=None), tmp_path, tmp_path)["passed"]
    absent = tmp_path / "absent.json"
    assert scan(tmp_path, [absent])["rows"][0]["reason"] == "malformed_producer"
    path = producer(tmp_path, [{"game": "g"}])
    assert scan(tmp_path, [path])["identity_filter_count"] == 0


# SCENARIO-REPORT-7936-PUBLISH: matching bytes do not override failed readiness operands.
def test_failed_prior_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = [tmp_path / name for name in ("prior.json", "inventory.json", "registry.yaml")]
    atomic_json(
        paths[0],
        dict(verdict_class="disqualified", flagged_adversarial=True, arc_delta_ready_score=0),
    )
    atomic_json(paths[1], {})
    paths[2].write_text("games: {}\n")
    monkeypatch.setattr(task, "PRIOR", paths[0])
    monkeypatch.setattr(task, "INVENTORY", paths[1])
    monkeypatch.setattr(task, "REGISTRY", paths[2])
    monkeypatch.setattr(task, "PINNED", {str(p): sha256_file(p) for p in paths})
    checked = task.inputs()
    assert len(checked["failures"]) == 3
