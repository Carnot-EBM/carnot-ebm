"""V673 capstone custody and cold reduction (REQ-REPORT-7738)."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from carnot import experiment_7738_v673_capstone as capstone
from carnot.experiment_7738_v673_capstone import account, cold_replay, read_authority


ROOT = Path(__file__).resolve().parents[2]


def test_three_source_contract_and_real_custody() -> None:
    """SCENARIO-REPORT-7738-CUSTODY: all slots and both blocked receipt types survive."""
    authority = read_authority(ROOT)
    assert authority["comparison"]["passed"] is True
    rows, hashes, failures = account(ROOT, authority["tasks"])
    assert [row["task_id"] for row in rows] == [task["id"] for task in authority["tasks"]]
    assert len(rows) == 13
    assert rows[-1]["availability"] == "planned_output"
    assert rows[5]["availability"] == "pre_gate_receipt"
    assert rows[7]["availability"] == "pre_gate_receipt"
    assert rows[8]["verdict_class"] == "blocked"
    assert any(item["upstream_id"] == "Exp7731" for item in failures)
    assert any(item["upstream_id"] == "Exp7733" for item in failures)
    assert any(item["upstream_id"] == "Exp7734" for item in failures)
    assert hashes["pre_gate_receipts"]
    assert hashes["absent_sources"]


def test_required_eligible_null_and_flagged_are_distinct(tmp_path: Path) -> None:
    """REQ-REPORT-7738: a complete null is eligible; a flag remains blocked."""
    tasks = [
        {
            "id": f"exp{number}-case",
            "deliverable": f"results/experiment_{number}_v673_case.json",
            "gated_on": [],
        }
        for number in range(7726, 7739)
    ]
    (tmp_path / "results").mkdir()
    for number in (7731, 7733, 7734):
        path = tmp_path / f"results/experiment_{number}_v673_case.json"
        path.write_text(
            json.dumps(
                {
                    "honest_verdict": "complete_null_case",
                    "verdict_class": "null",
                    "flagged_adversarial": False,
                }
            )
        )
    _, _, failures = account(tmp_path, tasks)
    assert failures == []
    flagged = tmp_path / "results/experiment_7733_v673_case.json"
    flagged.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null_case",
                "verdict_class": "null",
                "flagged_adversarial": True,
            }
        )
    )
    _, _, failures = account(tmp_path, tasks)
    assert any(
        item["field"] == "flagged_adversarial" and item["upstream_id"] == "Exp7733"
        for item in failures
    )


def test_cold_replay_detects_row_and_source_changes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7738-TERMINAL: result and source bytes bind together."""
    tasks = [
        {
            "id": f"exp{number}-case",
            "deliverable": f"results/experiment_{number}_v673_case.json",
            "gated_on": [],
        }
        for number in range(7726, 7739)
    ]
    (tmp_path / "results").mkdir()
    source = tmp_path / "results/experiment_7726_v673_case.json"
    source.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null_case",
                "verdict_class": "null",
                "flagged_adversarial": False,
            }
        )
    )
    rows, hashes, failures = account(tmp_path, tasks)
    candidate = {"rows": rows, "source_artifact_hashes": hashes, "gate_check_summary": failures}
    assert cold_replay(candidate, tmp_path, tasks) == []
    changed = json.loads(json.dumps(candidate))
    changed["rows"][0]["verdict_class"] = "positive"
    assert "rows" in cold_replay(changed, tmp_path, tasks)
    source.write_text(source.read_text() + " ")
    assert "source_artifact_hashes" in cold_replay(candidate, tmp_path, tasks)


def test_reject_malformed_custody_and_verdict(tmp_path: Path) -> None:
    """REQ-REPORT-7738: malformed producer bytes and changed verdicts fail closed."""
    tasks = [
        {
            "id": f"exp{number}-case",
            "deliverable": f"results/experiment_{number}_v673_case.json",
            "gated_on": [],
        }
        for number in range(7726, 7739)
    ]
    (tmp_path / "results").mkdir()
    with pytest.raises(ValueError, match="thirteen-task"):
        account(tmp_path, tasks[:-1])
    planned = tmp_path / tasks[0]["deliverable"]
    planned.write_text("[]")
    with pytest.raises(ValueError, match="producer object"):
        account(tmp_path, tasks)
    planned.unlink()
    alternate = tmp_path / tasks[0]["deliverable"].replace("_v673_", "_")
    alternate.write_text("{}")
    with pytest.raises(ValueError, match="pre-gate schema"):
        account(tmp_path, tasks)
    alternate.unlink()
    rows, hashes, failures = account(tmp_path, tasks)
    candidate = {
        "rows": rows,
        "source_artifact_hashes": hashes,
        "gate_check_summary": failures,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive",
    }
    assert "verdict" in cold_replay(candidate, tmp_path, tasks)


def staged_root(tmp_path: Path) -> Path:
    """Copy only the authority and input bytes needed by the reducer."""
    for name in (
        "research-roadmap.yaml",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "ops/publication_gate_state.json",
        "research-hardware-wishlist.md",
        capstone.MODULE,
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    authority = read_authority(ROOT)
    for task in authority["tasks"][:-1]:
        path = ROOT / task["deliverable"]
        alternate = ROOT / task["deliverable"].replace("_v673_", "_")
        source = path if path.is_file() else alternate
        if source.is_file():
            target = tmp_path / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return tmp_path


def test_terminal_builder_and_cold_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7738-TERMINAL: raw rows and exact candidate agree."""
    root = staged_root(tmp_path)
    full_log = root / capstone.RAW / "validation/full/00_full_python_suite.log"
    full_log.parent.mkdir(parents=True, exist_ok=True)
    full_log.write_text("18 unrelated collection errors\n")
    pub = capstone.publication(ROOT)
    monkeypatch.setattr(capstone, "publication", lambda unused: pub)
    monkeypatch.setattr(
        capstone,
        "run_commands",
        lambda unused, commands, **kwargs: [
            {
                "name": command.name,
                "passed": True,
                "exit_code": 0,
                "command": "test",
                "log_sha256": "sha256:test",
            }
            for command in commands
        ],
    )
    with pytest.raises(ValueError, match="run date"):
        capstone.run_experiment(root, "20260926", capstone.OUTPUT)
    result = capstone.run_experiment(root, "20260927", capstone.OUTPUT)
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"] == "complete_blocked_required_v673_evidence"
    assert result["capstone_accounting_ready_score"] == 1
    assert result["acceptance_gate_results"]["brier_score"] is None
    assert result["publication_gates"]["headline_auroc"] == 0.9131
    assert result["validation_receipts"]["global_suite_debt"]["exit_code"] == 2
    candidate = root / capstone.RAW / "terminal_candidate.json"
    assert capstone.read_candidate(candidate, root) == []
    assert capstone.main(["--root", str(root), "--cold-validate", str(candidate)]) == 0
    value = json.loads(candidate.read_text())
    value["continuation_decisions"][0]["gate"] = "false"
    candidate.write_text(json.dumps(value))
    assert "continuation_decisions" in capstone.read_candidate(candidate, root)
    assert capstone.main(["--root", str(root), "--cold-validate", str(candidate)]) == 1
    (root / capstone.RAW / "rows.json").write_text("[]")
    assert "raw_rows" in capstone.read_candidate(candidate, root)
    assert capstone.main(["--root", str(root), "--date", "20260927"]) == 0


def test_owned_terminal_failure_disqualifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7738: a failed terminal reader cannot open readiness."""
    root = staged_root(tmp_path)
    monkeypatch.setattr(
        capstone,
        "publication",
        lambda unused: {"paper_ready": False, "gates": {}, "unmet_gates": ["G2"]},
    )

    def receipts(unused: Path, commands: list[object], **kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "name": command.name,
                "passed": command.name != "adversarial_verify",
                "exit_code": int(command.name == "adversarial_verify"),
                "command": "test",
                "log_sha256": "sha256:test",
            }
            for command in commands
        ]

    monkeypatch.setattr(capstone, "run_commands", receipts)
    result = capstone.run_experiment(root, "20260927", capstone.OUTPUT)
    assert result["verdict_class"] == "disqualified"
    assert result["flagged_adversarial"] is True
    assert result["acceptance_gate_results"]["readiness"] is False
