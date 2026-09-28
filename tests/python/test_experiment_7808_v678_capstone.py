"""Task-owned V678 capstone checks (REQ-REPORT-7808)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot.experiment_7808_v678_capstone import (
    account_tasks,
    build_artifact,
    check_authority,
    mechanism_decisions,
    replay_candidate,
)


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "docs/research-notes/v678-authority-snapshots/roadmap.yaml"


def test_authority_snapshot_rejects_mutated_table_and_yaml(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7808-CUSTODY: all three authorities must agree."""
    design = (ROOT / "docs/research-notes/v678-authority-snapshots/design.md").read_bytes()
    roadmap = SNAPSHOT.read_bytes()
    assert check_authority(design, roadmap)["passed"] is True
    changed = design.replace(
        b"Bind fourteen tasks and register source dependence methods", b"Forged contract title", 1
    )
    assert check_authority(changed, roadmap)["passed"] is False
    value = yaml.safe_load(roadmap)
    value["tasks"][0]["title"] = "forged title"
    assert check_authority(design, yaml.safe_dump(value).encode())["passed"] is False


def test_missing_science_is_distinct_from_conductor_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7808-CUSTODY: a queue OK cannot fill a science row."""
    tasks = yaml.safe_load(SNAPSHOT.read_text())["tasks"]
    receipt = tmp_path / "results/experiment_7801_qwen_counter_evidence.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"schema": "blocked_gate_check_v1", "experiment": 7801}))
    rows, sources, failures = account_tasks(tmp_path, tasks)
    assert len(rows) == len(sources) + 1 == 14
    row = rows[6]
    assert row["availability"] == "pre_gate_receipt"
    assert row["producer_eligible"] is False
    assert row["producer_hash"] is None
    assert row["pre_gate_receipt_hash"] is not None
    assert any(f["upstream_id"] == "Exp7801" and f["field"] == "producer_path" for f in failures)
    assert rows[-1]["availability"] == "planned_output"
    assert all(source["path"] != tasks[-1]["deliverable"] for source in sources)


def test_disqualified_producer_cannot_open_a_science_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7808-CUSTODY: a score in rejected evidence is ineligible."""
    tasks = yaml.safe_load(SNAPSHOT.read_text())["tasks"]
    path = tmp_path / tasks[1]["deliverable"]
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": 7796,
                "milestone": "2026.09.678",
                "run_date": "20260928",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_required_validation",
                "flagged_adversarial": False,
                "sentence_protocol_ready_score": 1,
            }
        )
    )
    rows, _, failures = account_tasks(tmp_path, tasks)
    assert rows[1]["producer_eligible"] is False
    assert any(f["field"] == "verdict_class" for f in failures)
    assert all(row.get("benefit") is None for row in rows)


def test_repeated_prior_retires_only_unchanged_scope() -> None:
    """SCENARIO-REPORT-7808-DECISIONS: same verdict has a narrow retirement."""
    tasks = yaml.safe_load(SNAPSHOT.read_text())["tasks"]
    rows = [
        {
            "task_id": task["id"],
            "producer_path": task["deliverable"],
            "producer_hash": None,
            "verdict_class": None,
            "honest_verdict": None,
        }
        for task in tasks
    ]
    rows[8]["verdict_class"] = "disqualified"
    rows[8]["honest_verdict"] = "complete_disqualified_required_runner_validation"
    decisions = mechanism_decisions(rows, tasks)
    arc = next(item for item in decisions if item["task_id"] == tasks[8]["id"])
    assert arc["decision"] == "retire"
    assert arc["trigger"]
    assert arc["matched_prior_ids"]
    source = next(item for item in decisions if item["task_id"] == tasks[3]["id"])
    assert source["decision"] == "await_named_prerequisite"


def test_cold_replay_rejects_changed_source_and_self_hash(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7808-TERMINAL: source bytes and output roles stay fixed."""
    source = tmp_path / "source.json"
    source.write_text("{}")
    candidate = {
        "source_artifact_hashes": [
            {"path": "source.json", "sha256": "sha256:bad", "eligible": False}
        ],
        "task_dispositions": [{"experiment_id": 7808}],
    }
    assert "source.json" in replay_candidate(candidate, tmp_path)
    candidate["source_artifact_hashes"][0]["path"] = "results/experiment_7808_v678_capstone.json"
    assert "self_input" in replay_candidate(candidate, tmp_path)


def test_snapshot_is_historical_fixture_independent_of_mutable_roadmap() -> None:
    """SCENARIO-REPORT-7808-CUSTODY: the test binds immutable V678 bytes."""
    assert SNAPSHOT.is_file()
    assert len(yaml.safe_load(SNAPSHOT.read_bytes())["tasks"]) == 14


def test_current_capstone_keeps_external_gaps_and_publication_separate() -> None:
    """SCENARIO-REPORT-7808-DECISIONS: current absence stays blocked."""
    gate = {name: True for name in ("G1", "G2", "G3", "G4")}
    gate.update(paper_ready=True, unmet_gates=[])
    validation = {"required_checks_passed": True}
    result = build_artifact(ROOT, gate, validation, [], 0.25)
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"] == "complete_blocked_required_v678_evidence"
    assert result["paper_ready"] is True
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert [row["experiment_id"] for row in result["rows"]] == list(range(7795, 7809))
    assert result["rows"][6]["availability"] == "pre_gate_receipt"
    assert result["rows"][12]["producer_eligible"] is False
    assert result["MODEL_SPECS"] == []
    assert result["model_invocation_counts"]["loads"] == 0
    assert replay_candidate(result, ROOT) == []


def test_failed_current_validation_disqualifies_even_if_external_inputs_missing() -> None:
    """SCENARIO-REPORT-7808-TERMINAL: owned checks control own validity."""
    gate = {name: False for name in ("G1", "G2", "G3", "G4")}
    gate.update(paper_ready=False, unmet_gates=list(gate))
    result = build_artifact(ROOT, gate, {"required_checks_passed": False}, [], 0.25)
    assert result["verdict_class"] == "disqualified"
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert any(row["field"] == "required_checks_passed" for row in result["gate_check_summary"])


def test_invalid_authority_and_producer_schema_are_named(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7808-CUSTODY: corrupt bytes retain a failed operand."""
    assert check_authority(b"ignored", b"[]")["passed"] is False
    assert check_authority(b"\xff", SNAPSHOT.read_bytes())["passed"] is False
    tasks = yaml.safe_load(SNAPSHOT.read_bytes())["tasks"]
    with pytest.raises(ValueError, match="fourteen-task"):
        account_tasks(tmp_path, tasks[:-1])
    bad = tmp_path / tasks[0]["deliverable"]
    bad.parent.mkdir(parents=True)
    bad.write_text("[]")
    _, _, failures = account_tasks(tmp_path, tasks)
    assert any(f["field"] == "schema" and f["upstream_id"] == "Exp7795" for f in failures)


def test_changed_authority_bytes_disqualify_a_private_capstone(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7808-CUSTODY: stale live design cannot replace a snapshot."""
    from carnot.experiment_7808_v678_capstone import CLI, DESIGN, MODULE, ROADMAP

    for label in (DESIGN, ROADMAP, MODULE, CLI):
        target = tmp_path / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / label).read_bytes())
    changed = (
        (tmp_path / DESIGN)
        .read_bytes()
        .replace(
            b"Bind fourteen tasks and register source dependence methods",
            b"Forged contract title",
            1,
        )
    )
    (tmp_path / DESIGN).write_bytes(changed)
    live = tmp_path / "openspec/change-proposals/research-roadmap-vNEXT.md"
    live.parent.mkdir(parents=True)
    live.write_bytes((ROOT / DESIGN).read_bytes())
    gate = {name: False for name in ("G1", "G2", "G3", "G4")}
    gate.update(paper_ready=False, unmet_gates=list(gate))
    result = build_artifact(tmp_path, gate, {"required_checks_passed": True}, [], 0.25)
    assert result["verdict_class"] == "disqualified"
    assert {"table_json_yaml_match", "authority_bytes"}.issubset(
        {row["field"] for row in result["gate_check_summary"]}
    )


def test_cli_entrypoint_runs_private_e2e_without_touching_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7808-TERMINAL: run actual CLI control flow on private bytes."""
    from scripts.experiments import experiment_7808_v678_capstone as cli
    from carnot.experiment_7808_v678_capstone import CLI, DESIGN, MODULE, ROADMAP

    for label in (DESIGN, ROADMAP, MODULE, CLI):
        target = tmp_path / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / label).read_bytes())
    producer = tmp_path / "results/experiment_7795_v678_contract_methods.json"
    producer.parent.mkdir(parents=True, exist_ok=True)
    producer.write_bytes((ROOT / producer.relative_to(tmp_path)).read_bytes())
    publication = {
        "gates": {name: {"pass": True} for name in ("G1", "G2", "G3", "G4")},
        "paper_ready": True,
        "unmet_gates": [],
    }
    failed: set[str] = set()

    def fake_run(root: Path, name: str, argv: tuple[str, ...], timeout: float) -> dict[str, object]:
        from carnot.reporting.current_work_receipt import sha256_file

        log = root / f"logs/{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(publication) if name == "publication_gate" else "ok\n")
        return {
            "name": name,
            "command_argv": list(argv),
            "exit_code": 0,
            "log_path": str(log.relative_to(root)),
            "log_sha256": sha256_file(log),
            "passed": name not in failed,
            "duration_s": 0.01,
        }

    monkeypatch.setattr(cli, "_run", fake_run)
    monkeypatch.setattr(
        cli,
        "run_scoped_validation",
        lambda *a, **k: {"required_checks_passed": True, "validation_receipts": []},
    )
    result = cli.run_experiment(tmp_path, "20260928")
    assert result["honest_verdict"] == "complete_blocked_required_v678_evidence"
    assert (tmp_path / cli.OUTPUT).is_file()
    assert [r["name"] for r in result["validation_receipts"]["terminal_readers"]] == [
        "cold_replay",
        "adversarial_verify",
        "strict_row_consistency",
    ]
    assert result["duration_s"] > 0
    failed.add("strict_row_consistency")
    disqualified = cli.run_experiment(tmp_path, "20260928")
    assert disqualified["verdict_class"] == "disqualified"
    assert all(value == 0 for value in disqualified["acceptance_gate_results"].values())
    with pytest.raises(ValueError, match="run date"):
        cli.run_experiment(tmp_path, "20260927")


def test_cli_main_cold_read_and_result_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7808-TERMINAL: both CLI exits report exact state."""
    from scripts.experiments import experiment_7808_v678_capstone as cli

    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "source_artifact_hashes": [],
                "task_dispositions": [{"experiment_id": n} for n in range(7795, 7809)],
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["exp7808", "--cold-replay", str(candidate)])
    assert cli.main() == 0
    assert '"errors": []' in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["exp7808", "--date", "20260928"])
    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda *a: {
            "honest_verdict": "complete_blocked_required_v678_evidence",
            "verdict_class": "blocked",
        },
    )
    assert cli.main() == 0
    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda *a: {
            "honest_verdict": "complete_disqualified_required_validation",
            "verdict_class": "disqualified",
        },
    )
    assert cli.main() == 1


def test_cli_failure_and_helper_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7808-TERMINAL: failed readers zero the result."""
    from scripts.experiments import experiment_7808_v678_capstone as cli
    from carnot.experiment_7808_v678_capstone import CLI, DESIGN, MODULE, ROADMAP

    monkeypatch.setattr(cli, "run_commands", lambda *a, **k: [{"name": "probe", "passed": True}])
    assert cli._run(tmp_path, "probe", ("true",), 1)["passed"] is True
    for label in (DESIGN, ROADMAP, MODULE, CLI):
        target = tmp_path / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / label).read_bytes())
    monkeypatch.setattr(cli, "check_authority", lambda *a: {"passed": False, "errors": ["forged"]})
    with pytest.raises(ValueError, match="authority mismatch"):
        cli.run_experiment(tmp_path, "20260928")


def test_cli_script_guard_cold_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7808-TERMINAL: the script guard reaches main."""
    import runpy
    from carnot.experiment_7808_v678_capstone import CLI

    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "source_artifact_hashes": [],
                "task_dispositions": [{"experiment_id": n} for n in range(7795, 7809)],
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["exp7808", "--cold-replay", str(candidate)])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(ROOT / CLI), run_name="__main__")
    assert caught.value.code == 0
