"""REQ-REPORT-7891-V685: private authority lifecycle and CLI checks."""

from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from carnot.reporting.v685_authority_lifecycle import assess_authorities, cold_replay
from scripts.experiments import experiment_7891_v685_authority_lifecycle as cli

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/v685"
CLI = ROOT / "scripts/experiments/experiment_7891_v685_authority_lifecycle.py"


@pytest.fixture
def authority_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """SCENARIO-REPORT-7891-LIFECYCLE: test copies cannot follow live roadmap edits."""
    design, staged, active = (
        tmp_path / name for name in ("design.md", "staged.yaml", "active.yaml")
    )
    design.write_bytes((FIXTURE / "design.md").read_bytes())
    active.write_bytes(gzip.decompress((FIXTURE / "active.yaml.gz").read_bytes()))
    staged.write_bytes(active.read_bytes())
    return design, staged, active, tmp_path / "snapshots"


def test_matching_activation_and_consumed_stage(authority_paths):
    """SCENARIO-REPORT-7891-LIFECYCLE: only matching active bytes prove activation."""
    design, staged, active, snapshots = authority_paths
    result = assess_authorities(design, staged, active, snapshots)
    assert result["activated"] and result["planning_matched"]
    assert (
        result["canonical_tasks_sha256"]
        == "59704cb0b0513926d74c40683e7abb2657e7c6c73649291209afa2b3899eef96"
    )
    assert len(result["contract_rows"]) == 12
    assert all(row["matched"] for row in result["contract_rows"])
    staged.unlink()
    result = assess_authorities(design, staged, active, snapshots)
    assert result["activated"] and not result["planning_matched"]
    assert result["authority_snapshots"]["staged"]["exists"] is False
    staged.write_text("milestone: 2026.09.686\ntasks: []\n")
    assert assess_authorities(design, staged, active, snapshots)["activated"]


def test_old_active_and_changed_full_task_fail(authority_paths):
    """SCENARIO-REPORT-7891-LIFECYCLE: prompt and lineage belong to the digest."""
    design, staged, active, snapshots = authority_paths
    original = yaml.safe_load(active.read_text())
    old = deepcopy(original)
    old["milestone"] = "2026.09.684"
    active.write_text(yaml.safe_dump(old, sort_keys=False))
    assert not assess_authorities(design, staged, active, snapshots)["activated"]
    active.write_text(yaml.safe_dump(original, sort_keys=False))
    for mutate in (
        lambda t: t[0].__setitem__("prompt", t[0]["prompt"] + " changed"),
        lambda t: t[0]["prior_failures"][0].__setitem__("verdict", "changed"),
    ):
        changed = deepcopy(original)
        mutate(changed["tasks"])
        active.write_text(yaml.safe_dump(changed, sort_keys=False))
        result = assess_authorities(design, staged, active, snapshots)
        assert not result["activated"]
        assert any(
            f["artifact_field"] == "canonical_tasks_sha256" for f in result["gate_check_summary"]
        )


@pytest.mark.parametrize(
    "field",
    ("id", "title", "phase", "deliverable", "MODEL_SPECS", "inference_substrate_class", "gated_on"),
)
def test_core_mutations_fail(authority_paths, field):
    """SCENARIO-REPORT-7891-LIFECYCLE: every visible core operand is checked."""
    design, staged, active, snapshots = authority_paths
    value = yaml.safe_load(active.read_text())
    index = 3 if field == "gated_on" else 0
    value["tasks"][index][field] = "wrong"
    active.write_text(yaml.safe_dump(value, sort_keys=False))
    result = assess_authorities(design, staged, active, snapshots)
    assert not result["activated"]
    assert any(not row["matched"] for row in result["contract_rows"])


def test_missing_digest_and_inconsistent_design_count_fail(authority_paths):
    """SCENARIO-REPORT-7891-LIFECYCLE: both design bindings are required."""
    design, staged, active, snapshots = authority_paths
    original = design.read_text()
    design.write_text(
        original.replace("Canonical full-task SHA-256:", "Missing full-task SHA-256:")
    )
    with pytest.raises(ValueError, match="V685 design digest missing"):
        assess_authorities(design, staged, active, snapshots)
    design.write_text(original)
    lines = original.splitlines(keepends=True)
    first_task = next(i for i, line in enumerate(lines) if line.startswith("| 1 |"))
    design.write_text("".join(lines[:first_task] + lines[first_task + 1 :]))
    result = assess_authorities(design, staged, active, snapshots)
    assert not result["activated"]
    assert any(f["artifact_field"] == "task_count" for f in result["gate_check_summary"])


def test_wrong_staged_digest_and_active_count_fail(authority_paths):
    """SCENARIO-REPORT-7891-LIFECYCLE: each authority has its own task checks."""
    design, staged, active, snapshots = authority_paths
    staged_value = yaml.safe_load(staged.read_text())
    staged_value["tasks"][0]["prompt"] += " changed"
    staged.write_text(yaml.safe_dump(staged_value, sort_keys=False))
    result = assess_authorities(design, staged, active, snapshots)
    assert not result["activated"]
    assert any(
        f["artifact_path"] == str(staged) and f["artifact_field"] == "canonical_tasks_sha256"
        for f in result["gate_check_summary"]
    )
    active_value = yaml.safe_load(active.read_text())
    active_value["tasks"].pop()
    active.write_text(yaml.safe_dump(active_value, sort_keys=False))
    result = assess_authorities(design, staged, active, snapshots)
    assert any(
        f["artifact_path"] == str(active) and f["artifact_field"] == "task_count"
        for f in result["gate_check_summary"]
    )


def test_immutable_snapshot_refuses_corruption(authority_paths):
    """SCENARIO-REPORT-7891-LIFECYCLE: a prior snapshot cannot be overwritten."""
    design, staged, active, snapshots = authority_paths
    result = assess_authorities(design, staged, active, snapshots)
    Path(result["authority_snapshots"]["design"]["snapshot_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="immutable authority snapshot differs"):
        assess_authorities(design, staged, active, snapshots)


def test_cli_private_success_failure_and_replay(authority_paths, tmp_path):
    """SCENARIO-REPORT-7891-VALIDATION: real CLI rejects a changed candidate."""
    design, staged, active, _ = authority_paths
    output, raw = tmp_path / "candidate.json", tmp_path / "rows.json"
    base = [
        sys.executable,
        str(CLI),
        "--date",
        "20260929",
        "--design",
        str(design),
        "--staged",
        str(staged),
        "--active",
        str(active),
        "--output",
        str(output),
        "--raw",
        str(raw),
        "--no-validation",
    ]
    done = subprocess.run(base, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    value = json.loads(output.read_text())
    assert value["contract_ready_score"] == 0
    assert value["verdict_class"] == "partial"
    assert len(value["mutation_rows"]) >= 11
    assert all(row["passed"] for row in value["mutation_rows"])
    manifest = json.loads(Path(value["validation_command_manifest_path"]).read_text())
    assert any(
        item["name"] == "negative_cli_replay" and item["expected_exit"] == 1
        for item in manifest["commands"]
    )
    negative = next(item for item in manifest["commands"] if item["name"] == "negative_cli_replay")
    negative_run = subprocess.run(
        negative["argv"], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert negative_run.returncode == negative["expected_exit"]
    assert "cold_replay_mismatch" in negative_run.stdout
    assert cold_replay(output, raw)
    replay = subprocess.run(
        [sys.executable, str(CLI), "--cold-replay", str(output), "--raw", str(raw)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["rows"][0]["absolute_metric"] = 0
    output.write_text(json.dumps(value))
    bad = subprocess.run(
        [sys.executable, str(CLI), "--cold-replay", str(output), "--raw", str(raw)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert bad.returncode != 0
    assert "cold_replay_mismatch" in bad.stdout


def test_terminal_recheck_failure_refuses_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7891-VALIDATION: failed final bytes cannot publish."""
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    candidate = {
        "rows": [],
        "acceptance_gate_results": {"readiness": 1},
        "contract_ready_score": 1,
    }
    report = [{"name": "adversarial_verify", "actual_exit": 1}]
    monkeypatch.setattr(cli, "build_candidate", lambda *_: deepcopy(candidate))
    monkeypatch.setattr(cli, "terminal_validate", lambda *_: (False, report))
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp7891", "--output", str(output), "--raw", str(raw)],
    )
    with pytest.raises(ValueError, match="terminal_recheck_failed"):
        cli.main()
    assert not output.exists()


def test_child_deadline_seals_failed_receipt(tmp_path):
    """SCENARIO-REPORT-7891-VALIDATION: a timed-out check cannot pass."""
    receipt = cli.run_child(
        {
            "name": "deadline_probe",
            "argv": [sys.executable, "-c", "import time; time.sleep(10)"],
            "expected_exit": 0,
            "deadline_s": 0.01,
        },
        0,
        tmp_path,
    )
    assert receipt["timed_out"] is True
    assert receipt["passed"] is False
    assert Path(receipt["log_path"]).is_file()


def test_authority_and_required_check_verdicts(authority_paths, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7891-VALIDATION: readiness follows authority and checks."""
    design, staged, active, _ = authority_paths
    args = SimpleNamespace(
        design=design,
        staged=staged,
        active=active,
        output=tmp_path / "result.json",
        date="20260929",
        no_validation=True,
    )
    monkeypatch.setattr(cli, "mutation_controls", lambda *_: [])
    partial = cli.build_candidate(args, tmp_path / "partial")
    assert partial["verdict_class"] == "partial"
    assert partial["contract_ready_score"] == 0
    args.no_validation = False
    monkeypatch.setattr(cli, "run_child", lambda spec, *_: {"passed": False, "argv": spec["argv"]})
    disqualified = cli.build_candidate(args, tmp_path / "disqualified")
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["contract_ready_score"] == 0
    changed = yaml.safe_load(active.read_text())
    changed["milestone"] = "2026.09.684"
    active.write_text(yaml.safe_dump(changed, sort_keys=False))
    blocked = cli.build_candidate(args, tmp_path / "blocked")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["contract_ready_score"] == 0


def test_terminal_recovery_rechecks_disqualified_bytes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7891-VALIDATION: a repaired verdict gets checked again."""
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    candidate = {
        "rows": [],
        "acceptance_gate_results": {"readiness": 1},
        "contract_ready_score": 1,
    }
    checks = iter(
        [
            (False, [{"name": "adversarial_verify", "actual_exit": 1}]),
            (True, [{"name": "adversarial_verify", "actual_exit": 0}]),
        ]
    )
    monkeypatch.setattr(cli, "build_candidate", lambda *_: deepcopy(candidate))
    monkeypatch.setattr(cli, "terminal_validate", lambda *_: next(checks))
    monkeypatch.setattr(sys, "argv", ["exp7891", "--output", str(output), "--raw", str(raw)])
    assert cli.main() == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["contract_ready_score"] == 0
    assert value["flagged_adversarial"] is True
    assert json.loads((tmp_path / "terminal_validator_reports.json").read_text())["passed"]


def test_direct_replay_date_and_publication_hash_exits(tmp_path, monkeypatch, capsys):
    """SCENARIO-REPORT-7891-VALIDATION: CLI exits match replay and byte checks."""
    monkeypatch.setattr(cli, "cold_replay", lambda *_: False)
    monkeypatch.setattr(sys, "argv", ["exp7891", "--cold-replay", str(tmp_path / "bad")])
    assert cli.main() == 1
    assert "cold_replay_mismatch" in capsys.readouterr().out
    monkeypatch.setattr(cli, "cold_replay", lambda *_: True)
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", ["exp7891", "--date", "wrong"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    monkeypatch.setattr(cli, "build_candidate", lambda *_: {"rows": []})
    monkeypatch.setattr(cli, "sha256_file", lambda path: str(path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp7891", "--no-validation", "--output", str(output), "--raw", str(raw)],
    )
    with pytest.raises(ValueError, match="published bytes differ"):
        cli.main()
