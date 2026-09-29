"""REQ-REPORT-7877: current bytes control the V683 audit."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v683_independent_audit as audit


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_7877_v683_independent_audit.py"


def test_current_block_and_gate_operands():
    """SCENARIO-REPORT-7877-BLOCKED: skip receipts cannot replace science."""
    tasks, failures, sources = audit.inspect_sources(ROOT)
    assert [r["upstream_id"] for r in tasks] == [f"Exp{n}" for n in range(7865, 7877)]
    assert {r["upstream_id"] for r in tasks if r["status"] == "missing_producer"} == {
        "Exp7869",
        "Exp7870",
        "Exp7871",
        "Exp7872",
        "Exp7873",
        "Exp7875",
    }
    assert all(
        set(f) >= {"upstream_id", "path", "hash", "artifact_field", "op", "expected", "observed"}
        for f in failures
    )
    assert any(
        f["artifact_field"] == "source_boundary_ready_score" and f["observed"] == 0
        for f in failures
    )
    assert any(
        f["artifact_field"] == "science_producer" and f["observed"] == "conductor_skip_receipt"
        for f in failures
    )
    assert len(sources) >= 12


def test_primitive_family_reduction():
    """SCENARIO-REPORT-7877-REPLAY: seeds share a family denominator."""
    rows = [
        {
            "family_id": "a",
            "seed": 1,
            "arm": "energy",
            "status": "completed",
            "probability": 0.2,
            "label": 0,
            "cost": 1,
        },
        {
            "family_id": "a",
            "seed": 2,
            "arm": "energy",
            "status": "completed",
            "probability": 0.4,
            "label": 0,
            "cost": 3,
        },
        {
            "family_id": "b",
            "seed": 1,
            "arm": "energy",
            "status": "completed",
            "probability": 0.8,
            "label": 1,
            "cost": 2,
        },
    ]
    comparison = audit.reduce_family_rows(rows)
    assert comparison["independent_family_count"] == 2
    assert comparison["seed_count"] == 3
    assert comparison["brier"] == pytest.approx(0.07)
    assert comparison["cost"] == pytest.approx(2)


def test_candidate_and_replay(tmp_path):
    """SCENARIO-REPORT-7877-BLOCKED: block is terminal but audit may finish."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"commands": []}')
    result = audit.build_candidate(ROOT, tmp_path, manifest, "20260929")
    assert result["experiment_id"] == 7877
    assert result["verdict_class"] == "blocked"
    assert result["milestone_evidence_complete_score"] == 0
    assert result["MODEL_SPECS"] == []
    assert result["target_model"] == "none"
    assert result["current_work_receipt"]["model_invoked"] is False
    assert result["resolved_imports"]["carnot.reporting.v683_independent_audit"] == str(
        Path(audit.__file__).resolve()
    )
    assert len(result["task_evidence_rows"]) == 12
    assert result["audit_execution_ready_score"] == 0
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(result))
    assert audit.cold_replay(path, ROOT) == []
    changed = copy.deepcopy(result)
    changed["task_evidence_rows"].pop()
    path.write_text(json.dumps(changed))
    assert "task_rows_changed" in audit.cold_replay(path, ROOT)
    changed = copy.deepcopy(result)
    changed["source_artifact_hashes"][0]["sha256"] = "sha256:forged"
    path.write_text(json.dumps(changed))
    assert "source_bytes_changed" in audit.cold_replay(path, ROOT)
    changed = copy.deepcopy(result)
    changed["honest_verdict"] = "partial_blocked"
    path.write_text(json.dumps(changed))
    assert "partial_terminal_verdict" in audit.cold_replay(path, ROOT)


def test_private_cli_success_failure(tmp_path):
    """SCENARIO-REPORT-7877-REPLAY: real CLI covers both exit paths."""
    base = [sys.executable, str(CLI)]
    run = subprocess.run(
        base + ["--date", "20260929", "--science-only", "--output-root", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    candidate = tmp_path / "candidate.json"
    assert candidate.is_file()
    ok = subprocess.run(
        base + ["--check-only", str(candidate)], cwd=ROOT, capture_output=True, text=True
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    value = json.loads(candidate.read_text())
    value["rows"] = []
    candidate.write_text(json.dumps(value))
    bad = subprocess.run(
        base + ["--check-only", str(candidate)], cwd=ROOT, capture_output=True, text=True
    )
    assert bad.returncode != 0


def load_cli():
    """Load the script module so focused coverage sees its owned branches."""
    spec = importlib.util.spec_from_file_location("exp7877_cli", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reader_mutations_and_invalid_authority(tmp_path):
    """SCENARIO-REPORT-7877-REPLAY: malformed bytes and roster drift fail closed."""
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert audit._data(bad) == {}
    bad.write_text("[]")
    assert audit._data(bad) == {}
    assert audit._receipts({"validation_receipts": 1}) == []
    assert audit._receipts({"validation_receipts": [{"name": "ok"}, None]}) == [{"name": "ok"}]
    (tmp_path / audit.AUTHORITY).write_text("tasks:\n- id: exp0-wrong\n")
    with pytest.raises(ValueError, match="authority_order_changed"):
        audit.inspect_sources(tmp_path)
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps({"rows": [None]}))
    tasks = [
        {
            "upstream_id": "Exp7869",
            "task_id": "exp7869-energy-fit",
            "path": str(raw),
            "hash": audit.sha256_file(raw),
            "status": "disqualified",
            "eligible": False,
            "source_exposure": "exposed_development",
            "label_authority": "original_human_label_only",
        }
    ]
    units, comparisons = audit._primitive_rows(tmp_path, tasks)
    assert units[0]["status"] == "malformed"
    assert comparisons[0]["row_count"] == 1


def test_replay_detects_unreadable_gates_rows_and_logs(tmp_path):
    """SCENARIO-REPORT-7877-REPLAY: every bound byte must still agree."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"commands": []}')
    value = audit.build_candidate(ROOT, tmp_path, manifest, "20260929")
    candidate = tmp_path / "candidate.json"
    assert audit.cold_replay(candidate, ROOT) == ["candidate_unreadable"]
    changed = copy.deepcopy(value)
    changed["gate_check_summary"] = []
    candidate.write_text(json.dumps(changed))
    assert "gate_operands_changed" in audit.cold_replay(candidate, ROOT)
    changed = copy.deepcopy(value)
    changed["recomputed_comparison_rows"] = []
    candidate.write_text(json.dumps(changed))
    assert "raw_reduction_changed" in audit.cold_replay(candidate, ROOT)
    changed = copy.deepcopy(value)
    changed["validation_receipts"] = [
        {"log_path": str(tmp_path / "absent.log"), "log_sha256": "sha256:missing"}
    ]
    candidate.write_text(json.dumps(changed))
    assert "validation_log_changed" in audit.cold_replay(candidate, ROOT)


def test_cli_seal_and_checkpoint_guards(tmp_path):
    """SCENARIO-REPORT-7877-REPLAY: closed logs and checkpoints are immutable."""
    cli = load_cli()
    log = tmp_path / "raw.log"
    log.write_text("child output")
    receipt = cli._seal({"name": "one", "log_path": str(log)}, tmp_path)
    assert Path(receipt["log_path"]).read_text() == "child output"
    log.write_text("child output")
    assert (
        cli._seal({"name": "one", "log_path": str(log)}, tmp_path)["log_sha256"]
        == receipt["log_sha256"]
    )
    log.write_text("child output")
    Path(receipt["log_path"]).write_text("forged")
    with pytest.raises(ValueError, match="sealed_log_collision"):
        cli._seal({"name": "one", "log_path": str(log)}, tmp_path)
    output = tmp_path / "science"
    result = cli.science(output, "20260929", 0)
    manifest = output / "validation_command_manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="frozen_validation_manifest_changed"):
        cli.science(output, "20260929", 0)
    cli.atomic_json(manifest, cli.command_manifest(output))
    checkpoint = (
        output
        / "checkpoints"
        / result["reproducibility_checksum"].split(":", 1)[1]
        / "science.json"
    )
    checkpoint.write_text("{}")
    with pytest.raises(ValueError, match="hash_bound_checkpoint_changed"):
        cli.science(output, "20260929", 0)


@pytest.mark.parametrize("passed", [True, False])
def test_cli_validation_receipts(tmp_path, monkeypatch, passed):
    """SCENARIO-REPORT-7877-REPLAY: required exits control readiness."""
    cli = load_cli()
    manifest = tmp_path / "commands.json"
    manifest.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "name": "fake",
                        "argv": ["fake"],
                        "classification": "required",
                        "deadline_s": 3,
                    }
                ]
            }
        )
    )
    result = audit.build_candidate(ROOT, tmp_path, manifest, "20260929")

    def fake_run(_root, specs, *, log_dir, heartbeat_s):
        assert specs[0].name == "fake" and heartbeat_s == 30
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "00_fake.log"
        log.write_text("finished")
        return [
            {
                "name": "fake",
                "exit_code": 0 if passed else 1,
                "passed": passed,
                "log_path": str(log),
                "log_sha256": cli.sha256_file(log),
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run)
    outcome = cli.validate(result, tmp_path, 0)
    assert outcome["audit_execution_ready_score"] == int(passed)
    assert outcome["verdict_class"] == ("blocked" if passed else "disqualified")
    assert outcome["repository_health"]["status"] == (
        "passed" if passed else "failed_owned_validation"
    )
    assert Path(outcome["validation_receipts"][0]["log_path"]).is_file()


@pytest.mark.parametrize(
    "tail,flagged",
    [("error", True), ('{"flagged_count": 1}', True), ('{"flagged_count": 0}', False)],
)
def test_adversarial_receipt_controls_class(tmp_path, monkeypatch, tail, flagged):
    """SCENARIO-REPORT-7877-REPLAY: parse actual verifier report, fail closed."""
    cli = load_cli()
    manifest = tmp_path / "commands.json"
    manifest.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "name": "adversarial_verify",
                        "argv": ["verify"],
                        "classification": "required",
                        "deadline_s": 3,
                    }
                ]
            }
        )
    )
    result = audit.build_candidate(ROOT, tmp_path, manifest, "20260929")

    def fake_run(_root, _specs, *, log_dir, heartbeat_s):
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "00_adversarial_verify.log"
        log.write_text(tail)
        return [
            {
                "name": "adversarial_verify",
                "exit_code": 0,
                "passed": True,
                "log_path": str(log),
                "log_sha256": cli.sha256_file(log),
                "output_tail": tail,
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run)
    outcome = cli.validate(result, tmp_path, 0)
    assert outcome["flagged_adversarial"] is flagged
    assert outcome["verdict_class"] == ("disqualified" if flagged else "blocked")


@pytest.mark.parametrize("replay_errors", [[], ["changed"]])
def test_cli_main_terminal_paths(tmp_path, monkeypatch, replay_errors):
    """SCENARIO-REPORT-7877-REPLAY: published class follows terminal replay."""
    cli = load_cli()
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"commands": []}')
    result = audit.build_candidate(ROOT, tmp_path, manifest, "20260929")
    monkeypatch.setattr(cli, "science", lambda _output, _date, _start: copy.deepcopy(result))
    monkeypatch.setattr(cli, "validate", lambda value, _output, _start: value)
    monkeypatch.setattr(cli.audit, "cold_replay", lambda _path, _root: replay_errors)
    monkeypatch.setattr(cli, "OUTPUT", tmp_path / "final.json")
    monkeypatch.setattr(sys, "argv", [str(CLI), "--output-root", str(tmp_path)])
    assert cli.main() == 0
    final = json.loads(cli.OUTPUT.read_text())
    assert final["verdict_class"] == ("disqualified" if replay_errors else "blocked")
    assert final["field_principles"].get("cold_replay_errors", "present")
    monkeypatch.setattr(sys, "argv", [str(CLI), "--check-only", str(cli.OUTPUT)])
    assert cli.main() == int(bool(replay_errors))
    monkeypatch.setattr(sys, "argv", [str(CLI), "--date", "20260928"])
    with pytest.raises(SystemExit):
        cli.main()
