"""REQ-REPORT-7978-V691: exact custody, separate branches and checked publication."""

from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v691_capstone as cap
from carnot.reporting import v691_capstone_validation as validation
from scripts.experiments import experiment_7978_v691_capstone as cli

ROOT = Path(__file__).resolve().parents[2]


def fixture(root: Path, complete: bool = False) -> tuple[Path, Path]:
    """Frozen authority fixtures survive future roadmap activation."""
    root.mkdir(parents=True, exist_ok=True)
    design, active = root / "design.md", root / "research-roadmap.yaml"
    for path, name in ((design, "design.md"), (active, "active.yaml")):
        path.write_bytes(gzip.decompress((ROOT / f"tests/fixtures/v691/{name}.gz").read_bytes()))
    if complete:
        tasks = yaml.safe_load(active.read_bytes())["tasks"]
        for task in tasks[:-1]:
            gates = {
                g["artifact_field"]: g["value"]
                for t in tasks
                for g in t.get("gated_on", [])
                if g["upstream"] == task["id"] and g["op"] == "=="
            }
            producer(root, task, **gates)
    return design, active


def producer(root: Path, task: dict, **extra: object) -> Path:
    """A pre-midnight issue date with a post-midnight finish is honest custody."""
    path = root / task["deliverable"]
    cap.atomic_json(
        path,
        dict(
            experiment_id=int(task["id"][3:7]),
            task_id=task["id"],
            milestone="2026.10.691",
            run_date="20260930",
            execution_date="20260930",
            current_run_id=task["id"] + "-20260930",
            started_at="2026-09-30T23:59:59Z",
            finished_at="2026-10-01T00:00:01Z",
            MODEL_SPECS=[],
            honest_verdict="complete_null_fixture",
            verdict_class="null",
            flagged_adversarial=False,
            rows=[],
        )
        | extra,
    )
    return path


def test_midnight_and_frozen_tampering(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7978-DATES: freeze invocation, never substitute consumer date."""
    design, active = fixture(tmp_path, True)
    monkeypatch.setattr(cap.response, "reconstruct", lambda data: {})
    invocations = cap.freeze_producers(tmp_path, active)
    value = cap.build_candidate(tmp_path, design, active, "20261001", invocations=invocations)
    assert len(value["outcome_rows"]) == 13
    assert value["verdict_class"] == "null", value["gate_check_summary"]
    assert all(r["passed"] for r in value["producer_date_rows"])
    assert value["producer_date_rows"][0]["identity"]["run_date"] == "20260930"
    assert cap.cold_replay(value, tmp_path, design, active) == []
    path = Path(invocations[0]["path"])
    changed = json.loads(path.read_text())
    changed["run_date"] = "20261001"
    cap.atomic_json(path, changed)
    assert "source_bytes_changed" in cap.cold_replay(value, tmp_path, design, active)
    candidate = cap.build_candidate(tmp_path, design, active, "20261001", invocations=invocations)
    assert candidate["outcome_rows"][0]["status"] == "disqualified"
    with pytest.raises(ValueError, match="date"):
        cap.build_candidate(tmp_path, design, active, "20260930")


def test_dispositions_and_independent_branches(tmp_path):
    """SCENARIO-REPORT-7978-BRANCHES: dispatch evidence cannot fill absent science."""
    design, active = fixture(tmp_path)
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    producer(tmp_path, tasks[0], verdict_class="blocked")
    producer(tmp_path, tasks[1], verdict_class="disqualified")
    producer(tmp_path, tasks[2], flagged_adversarial=True)
    producer(tmp_path, tasks[3], rows=[None])
    producer(tmp_path, tasks[9], rows=[None])
    stub = tmp_path / "results/experiment_7970_energy_fit.json"
    cap.atomic_json(stub, dict(blocked_at_layer="conductor_pre_gate", gates_evaluated=[]))
    value = cap.build_candidate(tmp_path, design, active, "20261001")
    assert [r["status"] for r in value["rows"][:5]] == [
        "blocked",
        "disqualified",
        "disqualified",
        "null",
        "skipped",
    ]
    assert value["rows"][-1]["status"] == "self_administrative"
    assert value["verdict_class"] == "blocked"
    assert value["sample_size_budget"]["completed"] == 13
    assert len(value["gap_decisions"]) == 3
    assert value["scientific_branches"]["source_energy"]["decision"] == "blocked"
    assert all(not r["scientifically_eligible"] for r in value["independent_reduction_rows"])
    assert cap.cold_replay(value, tmp_path, design, active) == []
    changed = deepcopy(value)
    changed["rows"][0]["status"] = "positive"
    assert "outcome_rows_changed" in cap.cold_replay(changed, tmp_path, design, active)
    active.write_text("invalid: [")
    assert not cap.build_candidate(tmp_path, design, active, "20261001")["activation_confirmed"]


def test_checkpoint_readers(monkeypatch):
    """SCENARIO-REPORT-7978-BRANCHES: checkpoints, not readiness averages, define metrics."""
    monkeypatch.setattr(cap.response, "reconstruct", lambda data: {"targets": 3})
    monkeypatch.setattr(cap.calibration, "replay", lambda data: {"brier": 0.2})
    monkeypatch.setattr(cap.service, "replay", lambda data: [{"p95_s": 0.1}])
    assert cap.reduction(7968, {"public_role_manifests": {"fit": 1}, "rows": []})[
        "checkpoint_reduction"
    ] == {"targets": 3}
    assert cap.reduction(7972, {"calibrator_checkpoints": {"heads": 1}, "rows": []})[
        "checkpoint_reduction"
    ] == {"brier": 0.2}
    assert cap.reduction(
        7976, {"service_rows": [], "sample_size_budget": {"intended": 0}, "rows": []}
    )["checkpoint_reduction"] == [{"p95_s": 0.1}]


def test_manifest_and_validation_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7978-PUBLICATION: owned failures cannot hide in repository health."""
    plan = validation.manifest(tmp_path / "plan")
    assert plan["coverage_includes"] == list(validation.OWNED)
    assert all("results/raw/" not in r.get("scratch_path", "") for r in plan["commands"])
    assert all(
        "20260929" in r["argv"]
        for r in plan["commands"]
        if r["name"] in {"e2e016_fixture", "e2e016_replay"}
    )
    design, active = fixture(tmp_path / "root")
    value = cap.build_candidate(tmp_path / "root", design, active, "20261001")
    validation.disqualify(value, "owned_failure", active, 0, 1)
    assert value["verdict_class"] == "disqualified"
    assert value["capstone_execution_ready_score"] == 0
    assert value["acceptance_gate_results"]["validity"] is False


def test_cli_routes(tmp_path, monkeypatch, capsys):
    """SCENARIO-REPORT-7978-PUBLICATION: actual CLI has explicit private reduction routes."""
    design, active = fixture(tmp_path / "root")
    output = tmp_path / "pub/experiment_7978_v691_capstone.json"
    args = [
        "--date",
        "20261001",
        "--root",
        str(tmp_path / "root"),
        "--design",
        str(design),
        "--active",
        str(active),
    ]
    assert cli.main([*args, "--evidence-only", "--output", str(output)]) == 0
    assert cli.main([*args, "--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    value["G1"] = True
    cap.atomic_json(output, value)
    assert cli.main([*args, "--cold-replay", str(output)]) == 1
    monkeypatch.setattr(validation, "terminal", lambda *args: cap.atomic_json(args[1], args[0]))
    assert cli.main([*args, "--terminal-recheck", str(output), "--output", str(output)]) == 0
    monkeypatch.setattr(validation, "qualify", lambda *args: 0)
    assert cli.main(args) == 0
    with pytest.raises(SystemExit):
        cli.main([])
    assert "[exp7978]" in capsys.readouterr().out


def test_retirement_exact_evidence(tmp_path):
    """SCENARIO-REPORT-7978-BRANCHES: retirement compares actual terminal evidence."""
    path = tmp_path / "results/experiment_42_prior.json"
    cap.atomic_json(path, dict(task_id="exp42-prior", honest_verdict="complete_null_prior"))
    row = cap.prior_evidence(
        tmp_path, {"experiment_id": "exp42-prior", "verdict": "complete_null_prior"}
    )
    assert row["verified"] and row["sha256"] == cap.sha256_file(path)
    assert not cap.prior_evidence(
        tmp_path, {"experiment_id": "exp42-prior", "verdict": "invented"}
    )["verified"]
    assert not cap.prior_evidence(tmp_path, {"experiment_id": "exp404-missing", "verdict": "null"})[
        "verified"
    ]


def test_owned_qualification_paths(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7978-PUBLICATION: current failures and health have separate authority."""
    design, active = fixture(tmp_path / "root")
    template = cap.build_candidate(tmp_path / "root", design, active, "20261001")
    monkeypatch.setattr(cap, "build_candidate", lambda *args, **kwargs: deepcopy(template))
    monkeypatch.setattr(validation, "terminal", lambda *args: cap.atomic_json(args[1], args[0]))
    private_seen = []

    def plan(private, **kwargs):
        private_seen.append(private)
        cap.atomic_json(
            private / "coverage.json",
            {"files": {"module": {"summary": {"num_statements": 1, "covered_lines": 1}}}},
        )
        (private / "unit.coverage").write_text("immutable coverage receipt")
        cap.atomic_json(private / "success.json", {"sample_size_budget": {"completed": 13}})
        return {
            "commands": [
                dict(
                    name="publication_gate",
                    argv=["private-check"],
                    expected_exit=0,
                    failure_reason="expected_failure",
                    deadline_s=60,
                    classification="required",
                    fixture_mutation=dict(
                        source=str(private / "success.json"), output=str(private / "negative.json")
                    ),
                ),
                dict(
                    name="negative",
                    argv=["private-negative"],
                    expected_exit=1,
                    failure_reason="expected_failure",
                    deadline_s=60,
                    classification="required",
                ),
            ],
            "dependency_hashes": {},
            "coverage_includes": list(validation.OWNED),
        }

    monkeypatch.setattr(validation, "manifest", plan)

    def check(root, spec, private, durable):
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("expected_failure")
        return {**spec, "passed": True, "actual_exit": spec["expected_exit"], "log_path": str(log)}

    monkeypatch.setattr(validation, "run_check", check)
    monkeypatch.setattr(validation.prior, "publication_result", lambda *args: None)
    monkeypatch.setattr(validation.prior.prior, "coverage_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(cap, "cold_replay", lambda *args: [])
    output = tmp_path / "published/experiment_7978_v691_capstone.json"
    assert validation.qualify(tmp_path / "root", design, active, "20261001", output) == 0
    assert not private_seen[0].exists()
    monkeypatch.setattr(validation.prior.prior, "coverage_complete", lambda *args, **kwargs: False)
    monkeypatch.setattr(cap, "cold_replay", lambda *args: ["source_bytes_changed"])
    assert validation.qualify(tmp_path / "root", design, active, "20261001", output) == 0


def test_terminal_reports_and_consumers(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7978-PUBLICATION: recheck final bytes after verifier changes."""
    design, active = fixture(tmp_path / "root")
    template = cap.build_candidate(tmp_path / "root", design, active, "20261001")
    output = tmp_path / "published/experiment_7978_v691_capstone.json"
    calls = []

    def check(root, spec, private, durable):
        calls.append(spec["name"])
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        report = {"flagged_count": 1 if len(calls) == 1 else 0}
        log.write_text(
            json.dumps(report) if spec["name"] == "adversarial_verify" else "strict rows passed"
        )
        return {**spec, "passed": True, "actual_exit": 0, "log_path": str(log)}

    monkeypatch.setattr(validation, "run_check", check)
    value = deepcopy(template)
    validation.terminal(value, output, tmp_path / "terminal", tmp_path / "durable")
    assert value["verdict_class"] == "disqualified"
    assert value["flagged_adversarial"] is False
    report = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert report["candidate_sha256"] == cap.sha256_file(output)
    assert json.loads(Path(value["primary_resolution_receipt"]["path"]).read_text())["passed"]
    monkeypatch.setattr(validation, "reader_receipt", lambda *args, **kwargs: {"passed": False})
    with pytest.raises(ValueError, match="primary_reader_drift"):
        validation.terminal(deepcopy(template), output, tmp_path / "drift", tmp_path / "durable2")


def test_malformed_primary_and_dependency_eligibility(tmp_path):
    """SCENARIO-REPORT-7978-DATES: invalid producer custody excludes dependent science."""
    design, active = fixture(tmp_path, True)
    tasks = yaml.safe_load(active.read_text())["tasks"]
    frozen = cap.freeze_producers(tmp_path, active)
    path = tmp_path / tasks[4]["deliverable"]
    value = json.loads(path.read_text())
    value["run_date"] = "changed"
    cap.atomic_json(path, value)
    result = cap.build_candidate(tmp_path, design, active, "20261001", invocations=frozen)
    assert result["rows"][4]["status"] == "disqualified"
    assert not any(result["rows"][i]["eligible"] for i in (5, 7, 8))
    path.write_text("{")
    result = cap.build_candidate(tmp_path, design, active, "20261001")
    assert result["rows"][4]["status"] == "disqualified"
    assert len(result["rows"]) == 13
    with pytest.raises(ValueError, match="producer_invocation_roster"):
        cap.build_candidate(tmp_path, design, active, "20261001", invocations=[])


def test_missing_checkpoint_cannot_claim_readiness(monkeypatch):
    """SCENARIO-REPORT-7978-BRANCHES: an empty checkpoint cannot bypass cold guards."""
    with pytest.raises(ValueError, match="unsafe_readiness"):
        cap.reduction(
            7972, dict(rows=[], calibrator_checkpoints={}, qwen_calibration_ready_score=1)
        )


def test_missing_invocation_fields(tmp_path):
    """SCENARIO-REPORT-7978-DATES: missing identity cannot self-authenticate as None."""
    design, active = fixture(tmp_path, True)
    task = yaml.safe_load(active.read_text())["tasks"][0]
    path = tmp_path / task["deliverable"]
    value = json.loads(path.read_text())
    for key in ("run_date", "execution_date", "started_at", "finished_at"):
        value.pop(key)
    cap.atomic_json(path, value)
    result = cap.build_candidate(tmp_path, design, active, "20261001")
    assert result["rows"][0]["status"] == "disqualified"
    assert not result["producer_date_rows"][0]["passed"]


def test_authority_fallback_and_publication_operand_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7978-BRANCHES: failed authorities never grant scientific readiness."""
    design, active = fixture(tmp_path)
    monkeypatch.setattr(
        cap.methods, "assess", lambda *args: (_ for _ in ()).throw(ValueError("invalid"))
    )
    value = cap.build_candidate(tmp_path, design, active, "20261001")
    assert not value["activation_confirmed"]
    value["G1"] = True
    assert "publication_operands_changed" in cap.cold_replay(value, tmp_path, design, active)


@pytest.mark.parametrize("mode", ["parse", "count", "unstable"])
def test_terminal_invalid_or_unstable_reports(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-7978-PUBLICATION: validator report failures remove readiness."""
    design, active = fixture(tmp_path / "root")
    value = cap.build_candidate(tmp_path / "root", design, active, "20261001")
    calls = []

    def check(root, spec, private, durable):
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        if spec["name"] == "adversarial_verify":
            calls.append(spec["name"])
            text = (
                "invalid"
                if mode == "parse"
                else json.dumps(
                    {
                        "flagged_count": "invalid"
                        if mode == "count"
                        else (1 if len(calls) % 2 else 0)
                    }
                )
            )
        else:
            text = "strict rows passed"
        log.write_text(text)
        return {**spec, "passed": True, "actual_exit": 0, "log_path": str(log)}

    monkeypatch.setattr(validation, "run_check", check)
    output = tmp_path / "published/experiment_7978_v691_capstone.json"
    if mode == "unstable":
        with pytest.raises(ValueError, match="terminal_flags_unstable"):
            validation.terminal(value, output, tmp_path / "terminal", tmp_path / "durable")
    else:
        validation.terminal(value, output, tmp_path / "terminal", tmp_path / "durable")
        assert value["verdict_class"] == "disqualified"
        assert value["capstone_execution_ready_score"] == 0


def test_roster_timestamp_and_historical_receipts(tmp_path):
    """SCENARIO-REPORT-7978-DATES: ordering, timestamps and previous evidence are bound."""
    design, active = fixture(tmp_path)
    task = yaml.safe_load(active.read_text())["tasks"][0]
    producer(tmp_path, task, finished_at="2026-09-29T00:00:00Z")
    declared = task["prior_failures"][0]
    number = declared["experiment_id"].split("-")[0][3:]
    cap.atomic_json(
        tmp_path / f"results/experiment_{number}_historical.json",
        {"task_id": declared["experiment_id"], "honest_verdict": declared["verdict"]},
    )
    value = cap.build_candidate(tmp_path, design, active, "20261001")
    assert value["rows"][0]["status"] == "disqualified"
    assert value["retirement_decisions"][0]["prior_primary"]["verified"]
    assert (
        cap.decision([{"status": "disqualified", "gate_blocked": False, "eligible": 0}])
        == "disqualified"
    )
    active.write_text("tasks: []")
    assert len(cap.tasks_from(active)) == 13


def test_absent_authorities_cold_replay(tmp_path):
    """SCENARIO-REPORT-7978-PUBLICATION: complete external blocking remains replayable."""
    root = tmp_path / "absent"
    value = cap.build_candidate(
        root, root / "design.md", root / "research-roadmap.yaml", "20261001"
    )
    assert value["verdict_class"] == "blocked"
    assert cap.cold_replay(value, root, root / "design.md", root / "research-roadmap.yaml") == []


def test_failed_attempt_custody_and_health_reuse(tmp_path):
    """SCENARIO-REPORT-7978-PUBLICATION: preserve failed receipts without repeating global health."""
    output = tmp_path / "results/experiment_7978_v691_capstone.json"
    receipt = {"name": "repository_full_suite", "actual_exit": 1, "passed": False}
    cap.atomic_json(
        output,
        {
            "experiment_id": 7978,
            "honest_verdict": "complete_disqualified_required_validation",
            "validation_receipts": [
                {"classification": "required", "passed": False, "actual_exit": 1}
            ],
            "repository_health": {"current_full_suite": [receipt]},
        },
    )
    evidence = validation.archive_attempt(output)
    assert evidence["sha256"] == cap.sha256_file(output)
    assert evidence["value"]["repository_health"]["current_full_suite"][0]["passed"] is False
    assert Path(evidence["path"]).read_bytes() == output.read_bytes()
    plan = validation.manifest(tmp_path / "plan", health=evidence)
    assert not any(r["name"] == "repository_full_suite" for r in plan["commands"])
    assert plan["prior_attempt"]["sha256"] == evidence["sha256"]
