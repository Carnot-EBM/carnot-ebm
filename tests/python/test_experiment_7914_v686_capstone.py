"""REQ-REPORT-7914-V686: qualify current reduction with private authorities."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v686_capstone as cap
from scripts.experiments import experiment_7914_v686_capstone as cli

ROOT = Path(__file__).resolve().parents[2]


def fixture(root: Path) -> tuple[Path, Path]:
    """Frozen versions prevent a later roadmap from changing this test's evidence."""
    root.mkdir(parents=True, exist_ok=True)
    design, active = root / "design.md", root / "research-roadmap.yaml"
    for target, name in ((design, "design.md"), (active, "active.yaml")):
        target.write_bytes(gzip.decompress((ROOT / f"tests/fixtures/v686/{name}.gz").read_bytes()))
    return design, active


def producer(root: Path, task: dict, verdict: str = "null", **extra: object) -> Path:
    """Each fixture retains the identity of the producer whose bytes are measured."""
    path = root / task["deliverable"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": int(task["id"][3:7]),
                "task_id": task["id"],
                "milestone": "2026.09.686",
                "run_date": "20260930",
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


def test_dispositions_and_missing_authority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7914-DISPOSITIONS: absence cannot borrow conductor bytes."""
    design, active = fixture(tmp_path)
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    producer(tmp_path, tasks[0], "blocked")
    producer(tmp_path, tasks[1], "disqualified", training_runtime_ready_score=0)
    producer(tmp_path, tasks[8], "null")
    receipt = tmp_path / "results/experiment_7906_energy_fit.json"
    receipt.write_text(
        json.dumps({"blocked_at_layer": "conductor_pre_gate", "gates_evaluated": []})
    )
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    assert len(value["outcome_rows"]) == len(value["independent_reduction_rows"]) == 12
    assert [value["outcome_rows"][i]["status"] for i in (0, 1, 3, 8, 11)] == [
        "blocked",
        "disqualified",
        "skipped",
        "null",
        "self_administrative",
    ]
    assert value["verdict_class"] == "blocked"
    assert value["sample_size_budget"]["unit"] == "task_disposition"
    assert value["sample_size_budget"]["intended"] == 12
    assert value["capstone_execution_ready_score"] == 1
    assert len(value["gap_decisions"]) == 3
    assert value["claim_scope"]["gap_oracle_distinct"] == "open"
    assert any(
        x["artifact_field"] == "training_runtime_ready_score" and x["observed"] == 0
        for x in value["gate_check_summary"]
    )
    design.unlink()
    assert not cap.build_candidate(tmp_path, design, active, "20260930")["activation_confirmed"]
    with pytest.raises(ValueError, match="date"):
        cap.build_candidate(tmp_path, design, active, "20260929")


def test_probability_and_delayed_primitives() -> None:
    """SCENARIO-REPORT-7914-PRIMITIVES: repeated calls do not add families."""
    rows = [
        {
            "family_id": "f",
            "arm": arm,
            "seed": seed,
            "status": "completed",
            "label": 1,
            "probability": p,
            "cost": cost,
            "action": action,
        }
        for arm, p, cost, action in (("head", 0.8, 0.2, "abstain"), ("control", 0.5, 0.5, "accept"))
        for seed in (1, 2)
    ]
    reduced = cap.reduce_primitives(rows)
    assert reduced["independent_families"] == 1
    assert reduced["brier_by_arm"]["head"] == pytest.approx(0.04)
    assert reduced["abstention_by_arm"]["head"] == 1
    delayed = [
        {
            "family_id": "f",
            "status": "completed",
            "label": 1,
            "prediction_step": 1,
            "feedback_step": 2,
            "bank_write_step": 2,
            "later_decision_step": 3,
            "issued_alpha": 0.1,
            "feedback_issued_alpha": 0.1,
            "prediction_set": [1],
            "tau": 1,
            "release_step": 2,
            "window": 0,
        }
    ]
    value = cap.reduce_primitives(delayed)
    assert value["delayed_sets"]["1"]["coverage"] == 1
    for field, bad, reason in (
        ("feedback_step", 0, "feedback_order"),
        ("feedback_issued_alpha", 0.2, "issued_alpha_changed"),
        ("release_step", 1, "early_feedback"),
        ("feature_fields", ["label"], "label_feature"),
    ):
        with pytest.raises(ValueError, match=reason):
            cap.reduce_primitives([{**delayed[0], field: bad}])
    with pytest.raises(ValueError, match="stale_dependency"):
        cap.reduce_primitives(
            [
                {
                    **delayed[0],
                    "dependency_hashes": {"a": "old"},
                    "current_dependency_hashes": {"a": "new"},
                }
            ]
        )
    assert cap.reduce_primitives([])["delayed_sets"] == {}
    invalid = Path("/tmp/carnot-7914-invalid-json.txt")
    invalid.write_text("not json")
    assert cap.read(invalid)[0] == {}
    invalid.write_text("[]")
    assert cap.read(invalid)[0] == {}


def test_identity_replay_and_complete_null(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7914-PRIMITIVES: cold reduction rejects edited counters."""
    design, active = fixture(tmp_path)
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    for task in tasks[:-1]:
        ready = {
            g["artifact_field"]: g["value"]
            for t in tasks
            for g in t.get("gated_on", [])
            if g["upstream"] == task["id"] and g["op"] == "=="
        }
        producer(tmp_path, task, **ready)
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    assert value["verdict_class"] == "null"
    assert cap.cold_replay(value, tmp_path, design, active) == []
    value["sample_size_budget"]["completed"] = 999
    assert "sample_size_budget_changed" in cap.cold_replay(value, tmp_path, design, active)
    producer(tmp_path, tasks[0], "positive", task_id="wrong", rows=[None])
    bad = cap.build_candidate(tmp_path, design, active, "20260930")
    assert bad["outcome_rows"][0]["status"] == "disqualified"
    assert any(x["artifact_field"] == "task_id" for x in bad["gate_check_summary"])
    assert "source_bytes_changed" in cap.cold_replay(value, tmp_path, design, active)
    tasks[0]["title"] = "changed"
    active.write_text(yaml.safe_dump({"milestone": "2026.09.686", "tasks": tasks}))
    assert not cap.build_candidate(tmp_path, design, active, "20260930")["activation_confirmed"]


def test_owned_qualification_and_terminal_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7914-QUALIFICATION: failed checks suppress readiness."""
    design, active = fixture(tmp_path)
    counter = {"terminal": 0}

    def children(root: Path, commands: list, **kwargs: object) -> list[dict]:
        receipts = []
        for command in commands:
            log = tmp_path / f"{command.name}.log"
            log.write_text(
                json.dumps(
                    {
                        "flagged_count": int(counter["terminal"] == 0),
                        "gates": {"G1": {"pass": True}},
                    }
                )
            )
            receipts.append(
                {
                    "name": command.name,
                    "passed": command.name != "unit",
                    "exit_code": 1 if command.name == "unit" else 0,
                    "command_argv": list(command.argv),
                    "log_path": str(log),
                    "log_sha256": cap.sha256_file(log),
                    "duration_s": 0.01,
                }
            )
        if commands[0].name == "adversarial_verify":
            counter["terminal"] += 1
        return receipts

    monkeypatch.setattr(cap, "run_commands", children)
    monkeypatch.setattr(
        cap,
        "validation_commands",
        lambda p: [
            cap.CommandSpec("unit", ("false",), "required", 1),
            cap.CommandSpec("publication_gate", ("true",), "required", 1),
        ],
    )
    output = tmp_path / "terminal.json"
    cap.qualify(tmp_path, design, active, "20260930", output)
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["capstone_execution_ready_score"] == 0
    assert value["G1"] is True
    assert counter["terminal"] == 3
    assert value["flagged_adversarial"] is False
    sidecar = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert sidecar["candidate_sha256"] == cap.sha256_file(output)
    assert cap.cold_replay(value, tmp_path, design, active) == []


def test_cli_routes_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7914-QUALIFICATION: current private authority replaces old publication."""
    design, active = fixture(tmp_path)
    output = tmp_path / "cli.json"
    args = [
        "--date",
        "20260930",
        "--root",
        str(tmp_path),
        "--design",
        str(design),
        "--active",
        str(active),
        "--output",
        str(output),
    ]
    assert cli.main([*args, "--evidence-only"]) == 0
    assert cli.main([*args, "--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    value["rows"] = []
    output.write_text(json.dumps(value))
    assert cli.main([*args, "--cold-replay", str(output)]) == 1
    monkeypatch.setattr(cap, "qualify", lambda *args: 0)
    assert cli.main(args) == 0
    with pytest.raises(SystemExit):
        cli.main([])
    commands = cap.validation_commands(tmp_path)
    assert all(
        "--date" in c.argv and "20260930" in c.argv for c in commands if c.name.startswith("e2e016")
    )
    assert {"cli_success", "cli_block", "cli_failure", "cli_replay", "coverage_report"} <= {
        c.name for c in commands
    }
    assert "source_bytes_changed" not in capsys.readouterr().out
