"""REQ-REPORT-7927-V687: preserve evidence custody through private reductions."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v687_capstone as cap
from carnot.reporting import v687_capstone_validation as validation
from scripts.experiments import experiment_7927_v687_capstone as cli

ROOT = Path(__file__).resolve().parents[2]


def fixture(root: Path, complete: bool = False) -> tuple[Path, Path]:
    """Frozen current authorities keep tests independent of later conductor changes."""
    root.mkdir(parents=True, exist_ok=True)
    design, active = root / "design.md", root / "research-roadmap.yaml"
    for target, name in ((design, "design.md"), (active, "active.yaml")):
        target.write_bytes(gzip.decompress((ROOT / f"tests/fixtures/v687/{name}.gz").read_bytes()))
    if complete:
        tasks = yaml.safe_load(active.read_bytes())["tasks"]
        for task in tasks[:-1]:
            gates = {
                gate["artifact_field"]: gate["value"]
                for other in tasks
                for gate in other.get("gated_on", [])
                if gate["upstream"] == task["id"] and gate["op"] == "=="
            }
            producer(root, task, **gates)
    return design, active


def producer(root: Path, task: dict, **extra: object) -> Path:
    """Synthetic values test the audit without claiming that science executed."""
    path = root / task["deliverable"]
    cap.atomic_json(
        path,
        {
            "experiment_id": int(task["id"][3:7]),
            "task_id": task["id"],
            "milestone": "2026.09.687",
            "run_date": "20260930",
            "verdict_class": "null",
            "honest_verdict": "complete_null_fixture",
            "flagged_adversarial": False,
            "MODEL_SPECS": task["MODEL_SPECS"],
            "rows": [],
            **extra,
        },
    )
    return path


def test_thirteen_dispositions_and_authority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7927-CUSTODY: skipped receipts never become producers."""
    design, active = fixture(tmp_path)
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    producer(tmp_path, tasks[0], verdict_class="blocked")
    producer(tmp_path, tasks[1], verdict_class="disqualified", training_runtime_ready_score=0)
    producer(tmp_path, tasks[9])
    cap.atomic_json(
        tmp_path / "results/experiment_7918_energy_fit.json",
        dict(
            blocked_at_layer="conductor_pre_gate",
            gates_evaluated=[
                dict(
                    upstream=tasks[1]["id"],
                    artifact_path="actual-sidecar.json",
                    artifact_sha256="sha256:old",
                    artifact_field="training_runtime_ready_score",
                    op="==",
                    expected=1,
                    actual=None,
                    passed=False,
                )
            ],
        ),
    )
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    assert len(value["outcome_rows"]) == len(value["independent_reduction_rows"]) == 13
    assert [value["outcome_rows"][i]["status"] for i in (0, 1, 3, 4, 9, 12)] == [
        "blocked",
        "disqualified",
        "skipped",
        "absent",
        "null",
        "self_administrative",
    ]
    assert value["verdict_class"] == "blocked"
    assert value["capstone_execution_ready_score"] == 0
    assert value["sample_size_budget"]["completed"] == 13
    assert len(value["gap_decisions"]) == 3
    assert any(r["artifact_path"] == "actual-sidecar.json" for r in value["gate_check_summary"])
    design.unlink()
    assert not cap.build_candidate(tmp_path, design, active, "20260930")["activation_confirmed"]
    active.write_text("invalid: [")
    assert len(cap.build_candidate(tmp_path, design, active, "20260930")["rows"]) == 13
    with pytest.raises(ValueError, match="date"):
        cap.build_candidate(tmp_path, design, active, "20260929")


def test_complete_null_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7927-CUSTODY: audit readiness differs from scientific utility."""
    design, active = fixture(tmp_path, True)
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    assert value["verdict_class"] == "null"
    assert cap.cold_replay(value, tmp_path, design, active) == []
    value["sample_size_budget"]["completed"] = 99
    assert "sample_size_budget_changed" in cap.cold_replay(value, tmp_path, design, active)
    task = yaml.safe_load(active.read_bytes())["tasks"][0]
    producer(tmp_path, task, task_id="wrong", rows=[None])
    changed = cap.build_candidate(tmp_path, design, active, "20260930")
    assert changed["outcome_rows"][0]["status"] == "disqualified"
    assert "source_bytes_changed" in cap.cold_replay(value, tmp_path, design, active)
    value["G1"] = True
    assert "publication_operands_changed" in cap.cold_replay(value, tmp_path, design, active)
    task["prior_failures"] = []
    active.write_text(yaml.safe_dump(dict(milestone="2026.09.687", tasks=[task])))
    assert not cap.build_candidate(tmp_path, design, active, "20260930")["activation_confirmed"]


def test_primitive_channels_and_denominators() -> None:
    """SCENARIO-REPORT-7927-PRIMITIVES: independent groups cannot count seed repeats."""
    rows = [
        dict(
            family_id="f",
            source_group="g",
            status="completed",
            arm=arm,
            seed=seed,
            probability=p,
            label=1,
            cost=cost,
        )
        for arm, p, cost in (("head", 0.8, 0.2), ("control", 0.5, 0.5))
        for seed in (1, 2)
    ]
    reduced = cap.reduce_primitives(rows)
    assert reduced["brier_by_arm"]["head"] == pytest.approx(0.04)
    assert reduced["cost_by_arm"]["head"] == 0.2
    assert reduced["independent_families"] == reduced["source_groups"] == 1
    assert reduced["seed_count"] == 2
    extra = [
        dict(
            family_id="f",
            source_group="g",
            status="completed",
            arm=arm,
            probability=p,
            latency_ms=ms,
            score_draw_seed=1,
            stress_draw_seed=2,
            fragility_score=0.4,
            stress_failure=True,
        )
        for arm, p, ms in (("witness_neighbors", 0.7, 2), ("matched_control", 0.2, 3))
    ]
    reduced = cap.reduce_primitives(extra)
    assert reduced["source_sensitivity"]["mean_neighbors_minus_filler"] == pytest.approx(0.5)
    assert reduced["service_timing_ms"]["matched_control"] == 3
    assert reduced["fragility"]["stress_failure_rate"] == 1
    for bad, reason in (
        (dict(score_draw_seed=2), "score_stress"),
        (dict(feature_fields=["future_label"]), "future_label"),
        (dict(label_release_step=4, decision_step=2), "future_label"),
    ):
        with pytest.raises(ValueError, match=reason):
            cap.reduce_primitives([{**extra[0], **bad}])


def test_real_publication_receipt_and_invalid_outputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7927-QUALIFICATION: use the real read-only CLI receipt."""
    spec = dict(
        name="publication_gate",
        argv=[str(ROOT / ".venv/bin/python"), "scripts/publication_gate.py", "--json"],
        expected_exit=0,
        deadline_s=60,
        classification="required",
    )
    receipt = validation.run_check(ROOT, spec, tmp_path, tmp_path / "sealed")
    value = {}
    validation.publication_result(value, receipt)
    parsed = json.loads(Path(receipt["log_path"]).read_text())
    assert receipt["passed"]
    assert value["publication_gate_results"] == parsed
    assert value["paper_ready"] == parsed["paper_ready"]
    assert all(value[key] == parsed["gates"][key]["pass"] for key in cap.GATES)
    for content in ("not json", "[]", '{"gates":{}}'):
        Path(receipt["log_path"]).write_text(content)
        bad = {**receipt, "passed": True}
        validation.publication_result({}, bad)
        assert bad["passed"] is False
        assert bad["failure_reason"] == "invalid_publication_output"
    failed = {**receipt, "passed": False, "exit_code": 1}
    untouched = {}
    validation.publication_result(untouched, failed)
    assert untouched == {}


def test_frozen_commands_preserve_historical_date(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7927-QUALIFICATION: fixture dates differ from execution dates."""
    frozen = validation.manifest(tmp_path)
    by_name = {r["name"]: r for r in frozen["commands"]}
    for name in ("e2e016_fixture", "e2e016_replay"):
        argv = by_name[name]["argv"]
        assert argv[argv.index("--date") + 1] == "20260929"
    wrong = by_name["e2e016_wrong_date"]
    observed = validation.run_check(ROOT, wrong, tmp_path, tmp_path / "sealed")
    assert observed["passed"] and "run_date_mismatch" in Path(observed["log_path"]).read_text()
    assert frozen["coverage_includes"] == list(validation.OWNED)
    assert {
        "cli_success",
        "cli_block",
        "cli_failure",
        "cli_replay",
        "cli_terminal",
    } <= by_name.keys()
    assert len(frozen["dependency_hashes"]) > len(validation.OWNED)


@pytest.mark.parametrize("terminal_flags", ([0], [1, 0, 0], [1, 1, 1], [1, 0, 1]))
def test_terminal_recheck_binds_final_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_flags: list[int]
) -> None:
    """SCENARIO-REPORT-7927-QUALIFICATION: a verdict change needs a fresh byte check."""
    design, active = fixture(tmp_path)
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    value["capstone_execution_ready_score"] = 1
    value["acceptance_gate_results"]["readiness"] = 1
    calls = []

    def child(root: Path, spec: dict, private: Path, durable: Path) -> dict:
        index = len(calls) // 2
        flag = terminal_flags[min(index, len(terminal_flags) - 1)]
        log = tmp_path / f"{len(calls)}.log"
        log.write_text(
            json.dumps(dict(flagged_count=flag)) if spec["name"] == "adversarial_verify" else "OK\n"
        )
        calls.append(spec)
        return {
            **spec,
            "passed": True,
            "actual_exit": 0,
            "exit_code": 0,
            "log_path": str(log),
            "log_sha256": cap.sha256_file(log),
        }

    monkeypatch.setattr(validation, "run_check", child)
    output = tmp_path / "checked.json"
    if terminal_flags == [1, 0, 1]:
        with pytest.raises(ValueError, match="terminal_flags_unstable"):
            validation.terminal(value, output, tmp_path / "private", tmp_path / "durable")
        assert not output.exists()
        return
    validation.terminal(value, output, tmp_path / "private", tmp_path / "durable")
    actual = json.loads(output.read_text())
    assert actual["flagged_adversarial"] == bool(terminal_flags[-1])
    assert actual["capstone_execution_ready_score"] == int(not terminal_flags[0])
    sidecar = json.loads(Path(actual["terminal_validation_sidecar_path"]).read_text())
    assert sidecar["candidate_sha256"] == cap.sha256_file(output)
    assert sidecar["adversarial_report"]["flagged_count"] == terminal_flags[-1]


@pytest.mark.parametrize("fail", (False, True))
def test_owned_qualification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    """SCENARIO-REPORT-7927-QUALIFICATION: owned failures disqualify; absence blocks."""
    design, active = fixture(tmp_path)
    publication = dict(gates={k: dict(pass_=False) for k in cap.GATES})
    publication["gates"] = {k: {"pass": False} for k in cap.GATES}

    def child(root: Path, spec: dict, private: Path, durable: Path) -> dict:
        log = tmp_path / (spec["name"] + ".log")
        log.write_text(json.dumps(publication) if spec["name"] == "publication_gate" else "OK")
        return {
            **spec,
            "passed": not fail,
            "actual_exit": int(fail),
            "exit_code": int(fail),
            "log_path": str(log),
            "log_sha256": cap.sha256_file(log),
        }

    monkeypatch.setattr(validation, "run_check", child)
    monkeypatch.setattr(
        validation,
        "manifest",
        lambda p: dict(
            dependency_hashes={},
            coverage_includes=list(validation.OWNED),
            commands=[
                dict(
                    name="publication_gate",
                    argv=["true"],
                    expected_exit=0,
                    deadline_s=1,
                    classification="required",
                )
            ],
        ),
    )
    monkeypatch.setattr(validation, "coverage_complete", lambda p, **kw: not fail)
    monkeypatch.setattr(
        validation, "terminal", lambda value, output, *args: cap.atomic_json(output, value)
    )
    output = tmp_path / "final.json"
    assert validation.qualify(tmp_path, design, active, "20260930", output) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("disqualified" if fail else "blocked")
    assert value["capstone_execution_ready_score"] == int(not fail)
    assert Path(tmp_path / value["report_path"]).is_file()


def test_cli_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7927-QUALIFICATION: parameterized routes preserve producer IDs."""
    design, active = fixture(tmp_path)
    output = tmp_path / "candidate.json"
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
    monkeypatch.setattr(validation, "qualify", lambda *args: 0)
    assert cli.main(args) == 0
    monkeypatch.setattr(validation, "terminal", lambda *args: None)
    assert cli.main([*args, "--terminal-recheck", str(output)]) == 0
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260929"])


def test_expected_failure_reason_and_owned_replay_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7927-QUALIFICATION: cold drift is an owned disqualification."""
    design, active = fixture(tmp_path)
    monkeypatch.setattr(
        validation,
        "manifest",
        lambda p: dict(
            dependency_hashes={},
            commands=[
                dict(
                    name="cli_failure",
                    argv=["false"],
                    expected_exit=2,
                    failure_reason="--date",
                    deadline_s=1,
                    classification="required",
                )
            ],
        ),
    )

    def child(root: Path, spec: dict, private: Path, durable: Path) -> dict:
        log = tmp_path / "expected.log"
        log.write_text("--date required\n")
        return {
            **spec,
            "actual_exit": 2,
            "passed": True,
            "log_path": str(log),
            "log_sha256": cap.sha256_file(log),
        }

    monkeypatch.setattr(validation, "run_check", child)
    monkeypatch.setattr(validation, "coverage_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(cap, "cold_replay", lambda *args: ["source_bytes_changed"])
    monkeypatch.setattr(
        validation, "terminal", lambda value, output, *args: cap.atomic_json(output, value)
    )
    output = tmp_path / "final.json"
    assert validation.qualify(tmp_path, design, active, "20260930", output) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["capstone_execution_ready_score"] == 0
    assert value["validation_receipts"][0]["passed"]
    assert value["gate_check_summary"][-1]["observed"] == ["source_bytes_changed"]


def test_grouped_fragility_and_timestamp_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7927-PRIMITIVES: repeat draws cannot change group weight."""
    rows = [
        dict(
            family_id="a",
            source_group="a",
            status="completed",
            stress_failure=True,
            fragility_score=0.8,
            score_draw_seed=1,
            stress_draw_seed=2,
        )
        for _ in range(9)
    ]
    rows.append(
        dict(
            family_id="b",
            source_group="b",
            status="completed",
            stress_failure=False,
            fragility_score=0.2,
            score_draw_seed=1,
            stress_draw_seed=2,
        )
    )
    reduced = cap.reduce_primitives(rows)
    assert reduced["fragility"]["stress_failure_rate"] == 0.5
    assert reduced["fragility"]["mean_fragility_score"] == 0.5
    with pytest.raises(ValueError, match="family_source_group_conflict"):
        cap.reduce_primitives([rows[0], {**rows[0], "source_group": "other"}])
    design, active = fixture(tmp_path)
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    assert value["ended_monotonic_timestamp_ns"] >= value["started_monotonic_timestamp_ns"]
    assert value["acceptance_gate_results"]["readiness"] == 0
    assert not value["acceptance_gate_results"]["validity"]


@pytest.mark.parametrize("invalid", ("not json", "[]", "{}"))
def test_malformed_terminal_report_is_rechecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    """SCENARIO-REPORT-7927-QUALIFICATION: unreadable validators cannot open readiness."""
    design, active = fixture(tmp_path)
    value = cap.build_candidate(tmp_path, design, active, "20260930")
    calls = []

    def child(root: Path, spec: dict, private: Path, durable: Path) -> dict:
        log = tmp_path / f"bad-terminal-{len(calls)}.log"
        log.write_text(invalid if not calls else json.dumps(dict(flagged_count=0)))
        calls.append(spec)
        return {
            **spec,
            "passed": True,
            "actual_exit": 0,
            "log_path": str(log),
            "log_sha256": cap.sha256_file(log),
        }

    monkeypatch.setattr(validation, "run_check", child)
    output = tmp_path / "checked.json"
    validation.terminal(value, output, tmp_path / "private", tmp_path / "durable")
    actual = json.loads(output.read_text())
    assert actual["verdict_class"] == "disqualified"
    assert actual["capstone_execution_ready_score"] == 0
    assert not actual["flagged_adversarial"]
    assert actual["gate_check_summary"][-1]["observed"]["parse_error"]
