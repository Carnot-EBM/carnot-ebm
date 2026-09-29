"""REQ-REPORT-7850: current V681 capstone accounting and custody."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import v681_capstone as capstone
from scripts.experiments import experiment_7850_v681_capstone as cli


ROOT = Path(__file__).resolve().parents[2]


def test_scenario_report_7850_account_current_evidence() -> None:
    """SCENARIO-REPORT-7850-ACCOUNT keeps absent science and exact gates."""
    result = capstone.build_candidate(ROOT, "20260929", {"gates": {}, "paper_ready": False})
    rows = result["rows"]
    assert [row["experiment_id"] for row in rows] == list(range(7837, 7851))
    assert rows[-1]["availability"] == "planned_output"
    assert rows[3]["availability"] == "absent"
    assert rows[3]["verdict_class"] == "absent"
    assert rows[1]["verdict_class"] == "blocked"
    assert rows[2]["verdict_class"] == "disqualified"
    assert result["verdict_class"] == "blocked"
    assert result["milestone_evidence_ready_score"] == 0
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert any(
        f["artifact_field"] == "source_boundary_ready_score" and f["observed"] == 0
        for f in result["gate_check_summary"]
    )
    assert all(
        {"upstream_id", "path", "hash", "artifact_field", "op", "expected", "observed"} <= set(f)
        for f in result["gate_check_summary"]
    )
    assert len(result["continuation_decisions"]) == 14
    assert all(row["trigger"] for row in result["continuation_decisions"])
    assert not any(
        source["path"] == str(capstone.OUTPUT) for source in result["source_artifact_hashes"]
    )


def test_scenario_report_7850_replay_detects_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7850-REPLAY binds source, rows and sealed logs."""
    value = capstone.build_candidate(ROOT, "20260929", {"gates": {}, "paper_ready": False})
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value))
    assert capstone.cold_replay(ROOT, candidate) == []
    value["rows"][0]["producer_hash"] = "sha256:wrong"
    candidate.write_text(json.dumps(value))
    assert "rows" in capstone.cold_replay(ROOT, candidate)
    value = capstone.build_candidate(ROOT, "20260929", {"gates": {}, "paper_ready": False})
    log = tmp_path / "sealed.log"
    log.write_text("before")
    value["validation_receipts"] = [{"log_path": str(log), "log_sha256": capstone.sha256_file(log)}]
    candidate.write_text(json.dumps(value))
    assert capstone.cold_replay(ROOT, candidate) == []
    log.write_text("after")
    assert str(log) in capstone.cold_replay(ROOT, candidate)


def test_scenario_report_7850_validation_manifest_and_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7850-VALIDATION freezes commands and private CLI."""
    manifest = capstone.load_manifest(ROOT)
    assert {
        entry["name"] for entry in manifest["commands"] if entry["classification"] == "required"
    } == capstone.REQUIRED
    assert {
        entry["name"] for entry in manifest["commands"] if entry["classification"] == "diagnostic"
    } == {"repository_health_180s"}
    assert cli.main(["--date", "20260929", "--science-only", "--output-root", str(tmp_path)]) == 0
    candidate = tmp_path / "candidate.json"
    assert candidate.is_file()
    assert cli.main(["--cold-replay", str(candidate)]) == 0
    assert capstone.cold_replay(ROOT, candidate) == []


def test_scenario_report_7850_principle_and_retirement() -> None:
    """REQ-REPORT-7850 retires only a registered identical verdict."""
    result = capstone.build_candidate(ROOT, "20260929", {"gates": {}, "paper_ready": False})
    assert result["experiment_id"] == 7850
    assert result["task_id"] == "exp7850-capstone"
    assert result["field_principles"]["experiment_id"]
    assert result["MODEL_SPECS"] == result["model_specs"] == []
    assert result["model_invocation_counts"]["generation_calls_attempted"] == 0
    for row, decision in zip(result["rows"], result["continuation_decisions"], strict=True):
        prior = next(task for task in capstone.load_tasks(ROOT) if task["id"] == row["task_id"])[
            "prior_failures"
        ]
        same = [
            item
            for item in prior
            if item["retire_if_same_verdict"] and item["verdict"] == row["honest_verdict"]
        ]
        assert (decision["decision"] == "retire") == bool(same)


def test_scenario_report_7850_rejects_changed_authority_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7850-REPLAY rejects a changed authority or command set."""
    original_compare = capstone.compare_contract
    monkeypatch.setattr(
        capstone, "compare_contract", lambda *_: {"passed": False, "errors": ["changed"]}
    )
    with pytest.raises(ValueError, match="authority mismatch"):
        capstone.load_tasks(ROOT)
    monkeypatch.setattr(capstone, "compare_contract", original_compare)
    manifest = tmp_path / capstone.MANIFEST
    manifest.parent.mkdir(parents=True)
    value = capstone.load_manifest(ROOT)
    value["commands"][0]["name"] = "unexpected"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest mismatch"):
        capstone.load_manifest(tmp_path)


def test_scenario_report_7850_cold_source_manifest_and_relative_log(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7850-REPLAY detects three independent byte mutations."""
    value = capstone.build_candidate(ROOT, "20260929", {"gates": {}, "paper_ready": False})
    value["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
    value["validation_command_manifest_sha256"] = "sha256:wrong"
    value["validation_receipts"] = [{"log_path": "missing.log", "log_sha256": "sha256:wrong"}]
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(value))
    errors = capstone.cold_replay(ROOT, candidate)
    assert value["source_artifact_hashes"][0]["path"] in errors
    assert str(capstone.MANIFEST) in errors
    assert str(ROOT / "missing.log") in errors


def test_scenario_report_7850_malformed_producer_is_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7850-ACCOUNT excludes malformed producer bytes."""
    target = (ROOT / "results/experiment_7838_v681_source_boundary.json").resolve()
    original = Path.read_bytes

    def malformed(path: Path) -> bytes:
        return b"[not json" if path.resolve() == target else original(path)

    monkeypatch.setattr(Path, "read_bytes", malformed)
    rows, _, failures = capstone.account(ROOT, capstone.load_tasks(ROOT))
    assert not rows[1]["producer_eligible"]
    assert any(item["artifact_field"] == "experiment_id" for item in failures)
    assert capstone.artifact(ROOT, "results/experiment_7838_v681_source_boundary.json") == {}
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: b"[]" if path.resolve() == target else original(path)
    )
    rows, _, _ = capstone.account(ROOT, capstone.load_tasks(ROOT))
    assert rows[1]["verdict_class"] is None
    assert capstone.artifact(ROOT, "results/experiment_7838_v681_source_boundary.json") == {}


def test_scenario_report_7850_seals_and_reduces_owned_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7850-VALIDATION retains failed checks and separate health."""
    result = capstone.build_candidate(ROOT, "20260929", {"gates": {}, "paper_ready": False})
    entries = [
        {"name": name, "argv": ["true"], "classification": kind, "deadline_s": 1}
        for name, kind in (
            ("adversarial_verify", "required"),
            ("ruff_check", "required"),
            ("repository_health_180s", "diagnostic"),
        )
    ]
    monkeypatch.setattr(capstone, "load_manifest", lambda _root: {"commands": entries})
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    def fake_run(_root: Path, specs: list, *, log_dir: Path, heartbeat_s: float) -> list[dict]:
        name = specs[0].name
        log = log_dir / name / "raw.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(name)
        return [
            {
                "name": name,
                "log_path": str(log),
                "log_sha256": capstone.sha256_file(log),
                "output_tail": json.dumps({"flagged_count": 0})
                if name == "adversarial_verify"
                else "",
                "exit_code": 0 if name == "adversarial_verify" else 1,
                "passed": name == "adversarial_verify",
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run)
    validated = cli.validate(result, tmp_path / "owned", 0.0)
    assert validated["honest_verdict"] == "complete_disqualified_required_validation"
    assert validated["required_validation_failures"] == ["ruff_check"]
    assert validated["repository_health"]["status"] == "failed"
    assert all(Path(receipt["log_path"]).is_file() for receipt in validated["validation_receipts"])
    assert validated["flagged_adversarial"] is False
    assert validated["current_work_receipt"]["event_count"] == 0


def test_scenario_report_7850_checkpoint_does_not_replace_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7850-REPLAY refuses changed checkpoint content."""
    monkeypatch.setattr(cli, "publication", lambda _start: {"gates": {}, "paper_ready": False})
    value = cli.science(tmp_path, "20260929", 0.0)
    checkpoint = (
        tmp_path
        / "checkpoints"
        / value["reproducibility_checksum"].split(":", 1)[1]
        / "science.json"
    )
    checkpoint.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint_observation_changed"):
        cli.science(tmp_path, "20260929", 0.0)


def test_scenario_report_7850_malformed_verifier_flags_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7850-VALIDATION fails closed on unreadable verifier JSON."""
    result = capstone.build_candidate(ROOT, "20260929", {"gates": {}, "paper_ready": False})
    entry = {
        "name": "adversarial_verify",
        "argv": ["true"],
        "classification": "required",
        "deadline_s": 1,
    }
    monkeypatch.setattr(capstone, "load_manifest", lambda _root: {"commands": [entry]})
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    def fake_run(_root: Path, _specs: list, *, log_dir: Path, heartbeat_s: float) -> list[dict]:
        log = log_dir / "verifier.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("invalid")
        return [
            {
                "name": "adversarial_verify",
                "log_path": str(log),
                "log_sha256": capstone.sha256_file(log),
                "output_tail": "not JSON",
                "exit_code": 0,
                "passed": True,
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run)
    value = cli.validate(result, tmp_path / "owned", 0.0)
    assert value["flagged_adversarial"] is True
    assert value["verdict_class"] == "disqualified"


def test_scenario_report_7850_parent_terminal_and_bad_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7850-VALIDATION writes once after owned validation."""
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260930", "--science-only"])
    value = capstone.build_candidate(ROOT, "20260929", {"gates": {}, "paper_ready": False})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "science", lambda *_: value)
    monkeypatch.setattr(cli, "validate", lambda result, *_: result)
    assert cli.main(["--date", "20260929", "--output-root", str(tmp_path / "private")]) == 0
    terminal = json.loads((tmp_path / capstone.OUTPUT).read_text())
    assert terminal["experiment_id"] == 7850


def test_scenario_report_7850_external_absence_terminates_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7850-ACCOUNT yields blocked with all producers absent."""
    for path in (capstone.DESIGN, capstone.STAGED, capstone.ACTIVE, capstone.MANIFEST):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    value = capstone.build_candidate(tmp_path, "20260929", {"gates": {}, "paper_ready": False})
    assert value["verdict_class"] == "blocked"
    assert value["milestone_evidence_ready_score"] == 0
    assert value["rows"][0]["availability"] == "absent"
    assert value["gate_check_summary"]
