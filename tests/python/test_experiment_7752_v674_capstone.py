"""V674 capstone custody and fresh replay (REQ-REPORT-7752)."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from carnot import experiment_7752_v674_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


def fixture_root(tmp_path: Path) -> Path:
    """Give the reducer a real contract and private producer bytes."""
    for name in ("research-roadmap.yaml", str(capstone.DESIGN)):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    tasks = capstone.authority(tmp_path)["tasks"]
    for number in (7744, 7746, 7747):
        task = tasks[number - 7739]
        path = tmp_path / task["deliverable"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "honest_verdict": "complete_null_measured",
                    "verdict_class": "null",
                    "flagged_adversarial": False,
                    "rows": [{"unit_id": f"family-{number}", "denominators": {"families": 1}}],
                }
            )
        )
    return tmp_path


def test_contract_and_missing_required_sources() -> None:
    """SCENARIO-REPORT-7752-CUSTODY: all slots and exact blocks survive."""
    source = capstone.authority(ROOT)
    rows, hashes, failed = capstone.account(ROOT, source["tasks"])
    assert source["comparison"]["passed"]
    assert [r["experiment_id"] for r in rows] == list(range(7739, 7753))
    assert rows[-1]["availability"] == "planned_output"
    assert rows[2]["availability"] == "pre_gate_receipt"
    assert rows[5]["availability"] == "absent"
    assert rows[8]["verdict_class"] == "blocked"
    assert {f["upstream_id"] for f in failed} == {"Exp7744", "Exp7746", "Exp7747"}
    assert hashes["pre_gate_receipts"] and hashes["historical_disqualified_sources"]


def test_fixture_replay_rejects_changed_source_and_summary(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7752-REPLAY: exact source and row bytes bind conclusions."""
    root = fixture_root(tmp_path)
    value = capstone.build_artifact(
        root, {"gates": {}, "paper_ready": False, "unmet_gates": ["G2"]}
    )
    assert value["verdict_class"] == "null"
    assert value["capstone_complete_score"] == 1
    assert capstone.cold_replay(value, root) == []
    changed = json.loads(json.dumps(value))
    changed["task_dispositions"][0]["availability"] = "producer"
    assert "task_dispositions" in capstone.cold_replay(changed, root)
    task = capstone.authority(root)["tasks"][7744 - 7739]
    (root / task["deliverable"]).write_text((root / task["deliverable"]).read_text() + " ")
    assert "source_artifact_hashes" in capstone.cold_replay(value, root)


def test_malformed_contract_and_reader_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7752: malformed producer and wrong order cannot qualify."""
    root = fixture_root(tmp_path)
    tasks = capstone.authority(root)["tasks"]
    with pytest.raises(ValueError, match="fourteen"):
        capstone.account(root, tasks[:-1])
    path = root / tasks[7744 - 7739]["deliverable"]
    path.write_text("[]")
    with pytest.raises(ValueError, match="producer object"):
        capstone.account(root, tasks)


def test_flagged_alternate_and_contract_mismatch(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7752-CUSTODY: three distinct faults cannot qualify."""
    root = fixture_root(tmp_path)
    tasks = capstone.authority(root)["tasks"]
    planned = root / tasks[7744 - 7739]["deliverable"]
    value = json.loads(planned.read_text())
    value["flagged_adversarial"] = True
    planned.write_text(json.dumps(value))
    _, _, failures = capstone.account(root, tasks)
    assert any(f["field"] == "flagged_adversarial" for f in failures)
    planned.unlink()
    alternate = root / tasks[7744 - 7739]["deliverable"].replace("_v674_", "_")
    alternate.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="pre-gate schema"):
        capstone.account(root, tasks)
    alternate.unlink()
    design = root / capstone.DESIGN
    design.write_text(
        design.read_text().replace(
            "| 1 | exp7739-contract-methods | 1 | Bind fourteen tasks",
            "| 1 | exp7739-contract-methods | 1 | Bind thirteen tasks",
            1,
        )
    )
    result = capstone.build_artifact(root, {"gates": {}, "paper_ready": False})
    assert result["verdict_class"] == "disqualified"
    assert any(f["check"] == "independent_contract" for f in result["gate_check_summary"])


def test_fresh_cli_cold_reader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7752-REPLAY: a separate Python process checks raw bytes."""
    root = fixture_root(tmp_path)
    value = capstone.build_artifact(root, {"gates": {}, "paper_ready": False})
    raw = root / capstone.RAW / "rows.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(value["rows"]))
    candidate = raw.parent / "terminal_candidate.json"
    candidate.write_text(json.dumps(value))
    command = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / capstone.CLI),
        "--root",
        str(root),
        "--cold-validate",
        str(candidate),
    ]
    assert subprocess.run(command, cwd=ROOT, capture_output=True, check=False).returncode == 0
    raw.write_text("[]")
    assert subprocess.run(command, cwd=ROOT, capture_output=True, check=False).returncode == 1


def test_jsonl_raw_rows_are_hashed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7752-REPLAY: Qwen raw rows remain tied to source bytes."""
    root = fixture_root(tmp_path)
    task = capstone.authority(root)["tasks"][7745 - 7739]
    source = root / task["deliverable"]
    source.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null_pilot",
                "verdict_class": "null",
                "flagged_adversarial": False,
            }
        )
    )
    raw = (
        root
        / task["deliverable"].removesuffix(".json").replace("results/", "results/raw/")
        / "rows.jsonl"
    )
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text('{"family": 1}\n')
    value = capstone.build_artifact(root, {"gates": {}})
    assert value["rows"][7745 - 7739]["raw_rows_sha256"]
    raw.write_text('{"family": 2}\n')
    assert "source_artifact_hashes" in capstone.cold_replay(value, root)
