"""REQ-REPORT-7836: terminal V680 queue and validation custody."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7836_v680_capstone as capstone
from scripts.experiments import experiment_7836_v680_capstone as cli


ROOT = Path(__file__).resolve().parents[2]


def test_authority_and_queue_preserve_missing_science() -> None:
    """SCENARIO-REPORT-7836-QUEUE keeps every task and exact failed operands."""
    tasks, authority = capstone.load_authority(ROOT)
    assert authority["passed"] is True
    assert [task["id"].split("-", 1)[0] for task in tasks] == [
        f"exp{number}" for number in range(7823, 7837)
    ]
    rows, sources, failures = capstone.account(ROOT, tasks)
    assert len(rows) == 14
    assert rows[-1]["availability"] == "planned_output"
    assert rows[3]["availability"] == "pre_gate_receipt"
    assert rows[6]["availability"] == "pre_gate_receipt"
    assert rows[4]["availability"] == "absent"
    assert rows[10]["availability"] == "absent"
    assert rows[1]["producer_eligible"] is False
    assert any(f["field"] == "source_isolation_ready_score" for f in failures)
    assert any(s["role"] == "conductor_pre_gate_receipt" for s in sources)
    assert all(s["path"] != str(capstone.OUTPUT) for s in sources)


def test_current_result_is_terminal_blocked_and_scoped() -> None:
    """REQ-REPORT-7836 does not upgrade an audit or fixture to benefit."""
    result = capstone.build_result(ROOT, {"gates": {}, "paper_ready": False, "unmet_gates": []})
    assert result["honest_verdict"] == "complete_blocked_required_v680_evidence"
    assert result["verdict_class"] == "blocked"
    assert result["capstone_complete_score"] == 0
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert len(result["task_dispositions"]) == 14
    assert result["branch_outcomes"]["selective_utility"]["state"] == "blocked"
    assert result["claim_scope"]["fresh_generalization_eligible"] is False
    assert result["arc_selector_promoted"] is False
    assert result["arc_plain_rerun_scheduled"] is False
    assert result["gate_check_summary"]
    assert all(
        {"upstream_id", "path", "hash", "field", "op", "expected", "observed"} <= set(f)
        for f in result["gate_check_summary"]
    )
    assert result["source_artifact_hashes"]


def test_exact_prior_retirement_only() -> None:
    """SCENARIO-REPORT-7836-QUEUE retires a repeated scope, not a topic."""
    tasks, _ = capstone.load_authority(ROOT)
    rows, _, _ = capstone.account(ROOT, tasks)
    decisions = capstone.decisions(rows, tasks)
    assert decisions[1]["decision"] == "retire"
    assert decisions[1]["matched_prior_ids"]
    assert decisions[2]["decision"] == "continue"
    assert decisions[3]["decision"] == "await_named_prerequisite"
    assert all(item["trigger"] for item in decisions)


def test_dispatch_rejects_undeclared_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7836-REPLAY compares name, argv and class."""
    manifest = {"commands": [{"name": "one", "argv": ["true"], "classification": "required"}]}

    def good(_root: Path, spec: dict, _durable: Path) -> dict:
        return {
            "name": spec["name"],
            "command_argv": spec["argv"],
            "classification": spec["classification"],
        }

    observed: list[int] = []
    assert (
        len(capstone.dispatch(tmp_path, manifest, good, lambda index, _: observed.append(index)))
        == 1
    )
    assert observed == [0]
    for key, bad in (
        ("name", "extra"),
        ("command_argv", ["false"]),
        ("classification", "diagnostic"),
    ):

        def wrong(root: Path, spec: dict, durable: Path) -> dict:
            row = good(root, spec, durable)
            row[key] = bad
            return row

        with pytest.raises(ValueError, match="undeclared child"):
            capstone.dispatch(tmp_path, manifest, wrong)


def test_retry_and_mutation_fail_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7836-REPLAY seals once and detects changed bytes."""
    log = tmp_path / "sealed.log"
    log.write_bytes(b"first attempt")
    receipt = {"log_path": str(log), "log_sha256": capstone.sha256_file(log)}
    assert capstone.check_logs([receipt]) == []
    retry = tmp_path / "retry.log"
    retry.write_bytes(b"second attempt")
    assert capstone.check_logs([receipt]) == []
    log.write_bytes(b"mutated")
    assert capstone.check_logs([receipt]) == [str(log)]


def test_manifest_rejects_changed_argv(tmp_path: Path) -> None:
    """REQ-REPORT-7836 freezes the exact validation command bytes."""
    manifest = capstone.load_manifest(ROOT)
    assert {c["classification"] for c in manifest["commands"]} == {"required", "diagnostic"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest).replace("focused_pytest", "renamed_pytest"))
    with pytest.raises(ValueError, match="frozen"):
        capstone.load_manifest(ROOT, path)


def test_authority_mutation_and_invalid_producer(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7836-QUEUE rejects changed design and bad producer schema."""
    design = tmp_path / capstone.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes((ROOT / capstone.DESIGN).read_bytes())
    roadmap = tmp_path / capstone.ROADMAP
    roadmap.parent.mkdir(parents=True, exist_ok=True)
    value = yaml.safe_load((ROOT / capstone.ROADMAP).read_text())
    tasks = value["tasks"]
    value["tasks"][0]["title"] = "changed title"
    roadmap.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="authority mismatch"):
        capstone.load_authority(tmp_path)
    producer = tmp_path / tasks[0]["deliverable"]
    producer.parent.mkdir(parents=True)
    producer.write_text("[]")
    rows, _, failures = capstone.account(tmp_path, tasks)
    assert rows[0]["producer_eligible"] is False
    assert any(f["field"] == "verdict_class" for f in failures)


def test_replay_checks_all_custody_layers(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7836-REPLAY detects source, manifest and row mutation."""
    value = capstone.build_result(ROOT, {"gates": {}, "paper_ready": False, "unmet_gates": []})
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value))
    assert capstone.cold_replay(candidate, ROOT) == []
    value["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
    value["validation_command_manifest_sha256"] = "sha256:wrong"
    value["task_dispositions"][0]["producer_eligible"] = True
    candidate.write_text(json.dumps(value))
    errors = capstone.cold_replay(candidate, ROOT)
    assert value["source_artifact_hashes"][0]["path"] in errors
    assert str(capstone.MANIFEST) in errors
    assert "task_dispositions" in errors


def test_manifest_schema_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7836 rejects a changed roster even with a matching test digest."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "wrong", "commands": []}))
    monkeypatch.setattr(capstone, "MANIFEST_SHA", capstone.sha256_file(path).split(":", 1)[1])
    with pytest.raises(ValueError, match="schema"):
        capstone.load_manifest(ROOT, path)


def test_cli_prepare_replay_and_both_terminal_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7836-REPLAY drives the real CLI dispatcher branches."""
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "result.json"
    monkeypatch.setattr(cli, "CANDIDATE", candidate)
    monkeypatch.setattr(capstone, "OUTPUT", output)
    assert cli.main(["--prepare", str(candidate)]) == 0
    assert cli.main(["--prepare", str(candidate)]) == 0
    monkeypatch.setattr(capstone, "cold_replay", lambda *_: [])
    assert cli.main(["--cold-replay", str(candidate)]) == 0
    monkeypatch.setattr(capstone, "cold_replay", lambda *_: ["mutated"])
    assert cli.main(["--cold-replay", str(candidate)]) == 1
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260927"])
    changed = json.loads(candidate.read_text())
    changed["source_artifact_hashes"] = []
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="source bytes changed"):
        cli.main(["--prepare", str(candidate)])

    def dispatch(_root: Path, manifest: dict, _executor: object, before: object) -> list[dict]:
        receipts = []
        for index, command in enumerate(manifest["commands"][:2]):
            before(index, receipts)
            receipts.append(
                {
                    "name": command["name"],
                    "classification": command["classification"],
                    "passed": index == 0,
                    "exit_code": 0 if index == 0 else 1,
                    "log_path": str(tmp_path / "sealed.log"),
                    "log_sha256": "sha256:example",
                }
            )
        return receipts

    monkeypatch.setattr(capstone, "dispatch", dispatch)
    assert cli.main([]) == 1
    terminal = json.loads(output.read_text())
    assert terminal["verdict_class"] == "disqualified"
    assert terminal["validation_receipts"]["required_checks_passed"] is False
    assert cli.main(["--date", "20260928"]) == 1
