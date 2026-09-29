"""Private controls for REQ-REPORT-7902-V685."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v685_capstone as cap
from scripts.experiments import experiment_7902_v685_capstone as cli


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/v685"


def private_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A versioned contract makes mutations independent of live roadmaps."""
    root = tmp_path
    design = root / "design.md"
    active = root / "active.yaml"
    design.write_bytes((FIXTURES / "design.md").read_bytes())
    active.write_bytes(gzip.decompress((FIXTURES / "active.yaml.gz").read_bytes()))
    (root / "results").mkdir()
    return root, design, active


def producer(root: Path, task: dict, verdict: str, **extra: object) -> Path:
    """Write one private producer with an explicit task identity."""
    path = root / task["deliverable"]
    path.parent.mkdir(parents=True, exist_ok=True)
    number = int(task["id"][3:7])
    path.write_text(
        json.dumps(
            {
                "experiment_id": number,
                "task_id": task["id"],
                "milestone": "2026.09.685",
                "run_date": "20260929",
                "verdict_class": verdict,
                "honest_verdict": f"complete_{verdict}_fixture",
                "flagged_adversarial": False,
                "MODEL_SPECS": task["MODEL_SPECS"],
                "rows": [],
                **extra,
            }
        )
    )
    return path


def test_private_twelve_dispositions_and_gate_operands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7902-DISPOSITIONS: preserve each evidence state."""
    root, design, active = private_root(tmp_path)
    tasks = yaml.safe_load(active.read_text())["tasks"]
    producer(root, tasks[0], "circular_positive", contract_ready_score=1)
    producer(root, tasks[1], "blocked", source_boundary_ready_score=0)
    producer(root, tasks[2], "disqualified", intervention_protocol_ready_score=0)
    producer(root, tasks[8], "null", arc_delta_ready_score=1)
    candidate = cap.build_candidate(root, design, active, "20260929")
    assert len(candidate["outcome_rows"]) == 12
    assert candidate["outcome_rows"][-1]["status"] == "self_administrative"
    assert candidate["outcome_rows"][4]["status"] == "missing"
    assert candidate["outcome_rows"][1]["status"] == "blocked"
    assert candidate["outcome_rows"][2]["status"] == "disqualified"
    assert candidate["outcome_rows"][8]["status"] == "null"
    assert candidate["verdict_class"] == "blocked"
    assert candidate["honest_verdict"].startswith("complete_blocked_")
    assert candidate["capstone_execution_ready_score"] == 1
    assert any(
        item["upstream_id"] == "Exp7892"
        and item["artifact_field"] == "source_boundary_ready_score"
        and item["observed"] == 0
        and item["artifact_hash"]
        for item in candidate["gate_check_summary"]
    )
    assert candidate["sample_size_budget"]["intended"] == 12
    assert candidate["acceptance_gate_results"]["probability_quality"] is None


def test_authority_and_producer_identity_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7902-DISPOSITIONS: no relabeled positive passes."""
    root, design, active = private_root(tmp_path)
    tasks = yaml.safe_load(active.read_text())["tasks"]
    producer(root, tasks[0], "positive", task_id="wrong", MODEL_SPECS=["wrong"])
    candidate = cap.build_candidate(root, design, active, "20260929")
    assert candidate["outcome_rows"][0]["status"] == "inconsistent"
    assert {x["artifact_field"] for x in candidate["gate_check_summary"]} >= {
        "task_id",
        "MODEL_SPECS",
    }
    edited = yaml.safe_load(active.read_text())
    edited["tasks"][0]["gated_on"] = [{"upstream": "wrong", "op": "==", "value": 1}]
    active.write_text(yaml.safe_dump(edited))
    with pytest.raises(ValueError, match="authority"):
        cap.build_candidate(root, design, active, "20260929")


def test_primitive_reduction_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7902-REPLAY: families, not seeds, set sample size."""
    rows = [
        {
            "family_id": "f",
            "arm": arm,
            "seed": seed,
            "status": "completed",
            "label": 1,
            "probability": probability,
            "cost": cost,
            "false_accept": False,
        }
        for arm, probability, cost in (("head", 0.8, 0.2), ("control", 0.5, 0.5))
        for seed in (1, 2)
    ]
    reduced = cap.reduce_rows(rows)
    assert reduced["independent_families"] == 1
    assert reduced["completed"] == 4
    assert reduced["brier_by_arm"]["head"] == pytest.approx(0.04)
    assert reduced["cost_by_arm"]["control"] == pytest.approx(0.5)
    assert reduced["paired_cost_ci95"] == pytest.approx([0.3, 0.3])
    root, design, active = private_root(tmp_path)
    tasks = yaml.safe_load(active.read_text())["tasks"]
    producer(root, tasks[3], "positive", energy_fit_ready_score=1, rows=rows)
    candidate = cap.build_candidate(root, design, active, "20260929")
    path = root / "candidate.json"
    path.write_text(json.dumps(candidate))
    assert cap.cold_replay(path, root, design, active) == []
    changed = json.loads(path.read_text())
    changed["independent_reduction_rows"][3]["completed"] = 999
    path.write_text(json.dumps(changed))
    assert "independent_reduction_rows_changed" in cap.cold_replay(path, root, design, active)
    producer(root, tasks[3], "null", energy_fit_ready_score=1, rows=rows)
    assert "source_bytes_changed" in cap.cold_replay(path, root, design, active)


def test_private_cli_routes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7902-VALIDATION: real argv needs an explicit date."""
    root, design, active = private_root(tmp_path)
    script = ROOT / "scripts/experiments/experiment_7902_v685_capstone.py"
    output = root / "candidate.json"
    base = [
        sys.executable,
        str(script),
        "--root",
        str(root),
        "--design",
        str(design),
        "--active",
        str(active),
        "--output",
        str(output),
    ]
    missing = subprocess.run(
        [*base, "--evidence-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert missing.returncode != 0
    assert "--date" in missing.stderr
    success = subprocess.run(
        [*base, "--date", "20260929", "--evidence-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert success.returncode == 0, success.stderr
    replay = subprocess.run(
        [
            sys.executable,
            str(script),
            "--date",
            "20260929",
            "--root",
            str(root),
            "--design",
            str(design),
            "--active",
            str(active),
            "--cold-replay",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert replay.returncode == 0, replay.stderr


def test_reducer_rejects_invalid_rows_and_audits_order(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7902-REPLAY: raw contradictions stop evidence promotion."""
    with pytest.raises(ValueError, match="malformed_primitive_row"):
        cap.reduce_rows([None])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="family_label_conflict"):
        cap.reduce_rows([{"family_id": "f", "status": "completed", "label": value}
                         for value in (0, 1)])
    with pytest.raises(ValueError, match="human_label_changed"):
        cap.reduce_rows([{"family_id": "f", "status": "completed", "label": 1,
                          "original_human_label": 0}])
    counts = cap.reduce_rows([{"family_id": "f", "status": "unstarted"},
                              {"family_id": "f", "status": "excluded"},
                              {"family_id": "f", "status": "censored"},
                              {"family_id": "f", "status": "failed"},
                              {"family_id": "g", "status": "completed", "action": "accept",
                               "label": 0}])
    assert (counts["intended"], counts["failed"], counts["censored"],
            counts["false_accepts_by_arm"]["default"]) == (5, 1, 1, 1)
    assert cap.reduce_rows([{"family_id": "f", "arm": arm, "status": "completed"}
                            for arm in ("head", "control")])["paired_cost_ci95"] is None
    assert cap._science_audit(7897, [{"prediction_step": 4, "feedback_step": 3,
                                     "bank_write_step": 5, "later_decision_step": 6}]) == [
        "feedback_bank_decision_order"]
    assert cap._science_audit(7898, [
        {"family_id": "f", "buffer_role": "fit", "raw_confidence": .1,
         "transformed_confidence": .9},
        {"family_id": "f", "buffer_role": "test", "raw_confidence": .9,
         "transformed_confidence": .1}]) == [
            "calibration_buffer_overlap", "nonmonotone_confidence_transform"]
    unreadable = tmp_path / "bad.json"
    unreadable.write_text("[")
    assert cap._read(unreadable) is None
    assert cap.cold_replay(unreadable, tmp_path, unreadable, unreadable) == [
        "candidate_unreadable"]


def test_private_malformed_producer_and_date(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7902-DISPOSITIONS: malformed producer data is explicit."""
    root, design, active = private_root(tmp_path)
    tasks = yaml.safe_load(active.read_text())["tasks"]
    producer(root, tasks[0], "positive", rows="bad",
             validation_receipts={"checks": [{"name": "unit", "classification": "required",
                                               "passed": False, "exit_code": 1}]})
    producer(root, tasks[1], "positive", rows=[{"family_id": "x", "status": "completed",
                                               "label": 1, "original_human_label": 0}])
    candidate = cap.build_candidate(root, design, active, "20260929")
    fields = {x["artifact_field"] for x in candidate["gate_check_summary"]}
    assert "rows" in fields
    assert "human_label_changed" in fields
    assert "validation_receipts.unit" in fields
    assert candidate["outcome_rows"][1]["status"] == "inconsistent"
    with pytest.raises(ValueError, match="date"):
        cap.build_candidate(root, design, active, "20260928")
    changed = yaml.safe_load(active.read_text())
    changed["tasks"].pop()
    active.write_text(yaml.safe_dump(changed))
    monkeypatch.setattr(cap, "assess_authorities", lambda *args: {"activated": True})
    with pytest.raises(ValueError, match="task_order"):
        cap.build_candidate(root, design, active, "20260929")
