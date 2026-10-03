"""REQ-REPORT-7991-V692 and REQ-VERIFY-7991-V692: honest independent audit."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import os

import pytest
import yaml

from carnot.reporting import v692_capstone as cap
from carnot.reporting import v692_capstone_reduction as reduction
from carnot.reporting import v692_capstone_validation as validation
from scripts.experiments import experiment_7991_v692_capstone as cli

ROOT = Path(__file__).resolve().parents[2]


def fixture(root, complete=False):
    """Frozen inputs keep future authority activation from rewriting this test."""
    root.mkdir(parents=True, exist_ok=True)
    for name, target in (("design.md", "design.md"), ("active.yaml", "research-roadmap.yaml")):
        (root / target).write_bytes(
            gzip.decompress((ROOT / f"tests/fixtures/v692/{name}.gz").read_bytes())
        )
    tasks = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())["tasks"]
    if complete:
        for task in tasks[:-1]:
            gates = {
                g["artifact_field"]: g["value"]
                for other in tasks
                for g in other.get("gated_on", [])
                if g["upstream"] == task["id"] and g["op"] == "=="
            }
            cap.atomic_json(
                root / task["deliverable"],
                dict(
                    experiment_id=int(task["id"][3:7]),
                    task_id=task["id"],
                    milestone="2026.10.692",
                    run_date="20260930",
                    execution_date="20260930",
                    started_at="2026-09-30T23:59:59Z",
                    finished_at="2026-10-01T00:00:01Z",
                    honest_verdict="complete_null_fixture",
                    verdict_class="null",
                    flagged_adversarial=False,
                    rows=[],
                )
                | gates,
            )
    return tasks


def candidate(root):
    return cap.build_candidate(root, root / "design.md", root / "research-roadmap.yaml", "20261001")


def test_dispositions_rollover_and_independent_prerequisites(tmp_path):
    """SCENARIO-REPORT-7991-DISPOSITIONS: audit failure cannot block separate science."""
    tasks = fixture(tmp_path, True)
    path = tmp_path / tasks[0]["deliverable"]
    data = json.loads(path.read_bytes())
    data.update(verdict_class="disqualified", honest_verdict="complete_disqualified_fixture")
    cap.atomic_json(path, data)
    value = candidate(tmp_path)
    assert len(value["rows"]) == 13
    assert value["rows"][0]["status"] == "disqualified"
    assert value["rows"][3]["eligible"] == 1
    assert value["scientific_branches"]["fit"]["decision"] == "measured-null"
    assert value["verdict_class"] == "null"
    assert value["producer_date_rows"][1]["identity"]["execution_date"] == "20260930"
    assert (
        cap.cold_replay(value, tmp_path, tmp_path / "design.md", tmp_path / "research-roadmap.yaml")
        == []
    )
    (tmp_path / tasks[4]["deliverable"]).unlink()
    cap.atomic_json(
        tmp_path / "results/experiment_7983_reserved_decisions.json",
        dict(
            blocked_at_layer="conductor_pre_gate", gates_evaluated=[], honest_verdict="GATE_BLOCK"
        ),
    )
    value = candidate(tmp_path)
    assert value["verdict_class"] == "blocked" and not value["science_ready"]
    assert value["rows"][4]["status"] == "skipped"
    assert value["capstone_execution_ready_score"] == 0
    assert value["gate_check_summary"] and len(value["gap_decisions"]) == 3
    assert value["MODEL_SPECS"] == [] and value["model_invocation_counts"]["calls"] == 0


def test_primitive_metrics_unknowns_sources_and_invalid_rows():
    """SCENARIO-REPORT-7991-REDUCTION: seeds and unknown labels cannot inflate science."""
    rows = [
        dict(
            family_id="a",
            source_cluster_id="source",
            arm=arm,
            seed=seed,
            p=p,
            y=y,
            decision="accept",
            role="evaluation",
            status="completed",
        )
        for arm, p, y in (("energy", 0.2, 1), ("logistic", 0.3, 1), ("unknown", 0.9, -1))
        for seed in (1, 2)
    ]
    measured = reduction.reduce_rows(rows)
    assert measured["independent_sources"] == 1 and measured["seeds"] == 2
    assert measured["scored_rows"] == 4 and measured["unknown_label_rows"] == 2
    assert measured["arm_metrics"]["energy"]["brier"] == pytest.approx(0.64)
    assert measured["arm_metrics"]["energy"]["cost"] == 5
    with pytest.raises(ValueError, match="malformed"):
        reduction.reduce_rows([None])
    with pytest.raises(ValueError, match="probability"):
        reduction.reduce_rows([dict(rows[0], p=2)])
    with pytest.raises(ValueError, match="same_information"):
        reduction.reduce_rows([rows[0], dict(rows[2], family_id="b")])
    with pytest.raises(ValueError, match="causal"):
        reduction.reduce_rows([dict(rows[0], label_read_event=1, label_release_event=2)])
    with pytest.raises(ValueError, match="restart"):
        reduction.reduce_rows(
            [dict(rows[0], restart_identity="a"), dict(rows[0], restart_identity="b")]
        )


def test_missing_malformed_and_frozen_authority(tmp_path):
    """SCENARIO-REPORT-7991-REPLAY: legitimate absence differs from replay drift."""
    fixture(tmp_path, True)
    value = candidate(tmp_path)
    for field, changed in (("sample_size_budget", {}), ("G1", True)):
        altered = deepcopy(value)
        altered[field] = changed
        assert cap.cold_replay(
            altered, tmp_path, tmp_path / "design.md", tmp_path / "research-roadmap.yaml"
        )
    saved = Path(value["authority_snapshots"]["active"]["snapshot_path"])
    saved.write_text("drift")
    assert "source_bytes_changed" in cap.cold_replay(
        value, tmp_path, tmp_path / "design.md", tmp_path / "research-roadmap.yaml"
    )
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []")
    assert not candidate(tmp_path)["activation_confirmed"]
    absent = tmp_path / "absent"
    blocked = candidate(absent)
    assert blocked["verdict_class"] == "blocked"
    assert (
        cap.cold_replay(blocked, absent, absent / "design.md", absent / "research-roadmap.yaml")
        == []
    )
    with pytest.raises(ValueError, match="date"):
        cap.build_candidate(
            tmp_path, tmp_path / "design.md", tmp_path / "research-roadmap.yaml", "20260930"
        )


def test_real_private_cli_expected_negative_wrapper(tmp_path):
    """SCENARIO-REPORT-7991-REPLAY: real CLI negatives pass only through exit assertions."""
    root = tmp_path / "fixture"
    tasks = fixture(root, True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    coverage_dir = Path(env.get("CARNOT_7991_COVERAGE_DIR", str(tmp_path)))
    program = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        "--data-file=" + str(coverage_dir / "owned.coverage"),
        "--include=" + ",".join(str(ROOT / p) for p in validation.OWNED),
        str(ROOT / validation.OWNED[-1]),
    ]
    base = ["--date", "20261001", "--root", str(root), "--design", str(root / "design.md")]
    output = tmp_path / "experiment_7991_v692_capstone.json"
    attempts = []

    def record(name, result, expected):
        attempts.append(name)
        cap.atomic_json(
            coverage_dir / "cli-receipts" / f"{len(attempts):02d}-{name}.json",
            dict(
                name=name,
                argv=result.args,
                expected_exit=expected,
                actual_exit=result.returncode,
                passed=result.returncode == expected,
                classification="required",
                cwd=str(tmp_path),
                PYTHONPATH_removed=True,
                stdout=result.stdout,
                stderr=result.stderr,
                asserted_child_exit=1 if name.startswith("negative_") else None,
            ),
        )

    def run(args):
        result = subprocess.run(
            program + args, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120
        )
        record(
            "private_cli_" + str(len(attempts)), result, 2 if args == ["--date", "20260930"] else 0
        )
        return result

    issued = run(base + ["--evidence-only", "--output", str(output)])
    assert issued.returncode == 0, issued.stdout + issued.stderr
    assert run(base + ["--cold-replay", str(output)]).returncode == 0
    wrapper = "import subprocess,sys; p=subprocess.run(sys.argv[1:]); assert p.returncode == 1, p.returncode; print('expected_child_exit_1',flush=True)"

    def negative(name):
        result = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-c",
                wrapper,
                *program,
                *base,
                "--cold-replay",
                str(output),
            ],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0 and "expected_child_exit_1" in result.stdout, (
            name,
            result.stdout,
            result.stderr,
        )
        record("negative_" + name, result, 0)

    original = json.loads(output.read_bytes())
    tampered = deepcopy(original)
    tampered["sample_size_budget"]["completed"] = 99
    cap.atomic_json(output, tampered)
    negative("aggregate_drift")
    cap.atomic_json(output, original)
    producer = root / tasks[0]["deliverable"]
    bytes_before = producer.read_bytes()
    producer.unlink()
    negative("missing_upstream")
    producer.write_bytes(bytes_before)
    frozen = Path(original["authority_snapshots"]["active"]["snapshot_path"])
    before = frozen.read_bytes()
    frozen.write_text("drift")
    negative("authority_drift")
    frozen.write_bytes(before)
    malformed = json.loads(producer.read_bytes())
    malformed["rows"] = [None]
    cap.atomic_json(producer, malformed)
    negative("malformed_rows")
    producer.write_bytes(bytes_before)
    absent = tmp_path / "absent"
    blocked = tmp_path / "blocked.json"
    args = ["--date", "20261001", "--root", str(absent), "--design", str(absent / "design.md")]
    assert run(args + ["--evidence-only", "--output", str(blocked)]).returncode == 0
    assert run(args + ["--cold-replay", str(blocked)]).returncode == 0
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"
    assert run(["--date", "20260930"]).returncode == 2


def test_producer_faults_and_references(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7991-DISPOSITIONS: failed custody names the exact observed operand."""
    tasks = fixture(tmp_path, True)
    path = tmp_path / tasks[0]["deliverable"]
    data = json.loads(path.read_bytes())
    ref = tmp_path / "checkpoint.json"
    ref.write_text("observed")
    data.update(
        run_date="invalid",
        experiment_id=0,
        flagged_adversarial=True,
        source_artifact_hashes={str(ref): "sha256:wrong"},
        gate_check_summary=[
            dict(
                path=str(ref),
                upstream_id="external",
                field="external_ready",
                expected=1,
                observed=0,
            )
        ],
    )
    cap.atomic_json(path, data)
    assert any(
        r["artifact_field"] == "producer_invocation_dates"
        for r in candidate(tmp_path)["gate_check_summary"]
    )
    data["run_date"] = "20260930"
    data["finished_at"] = "2026-09-29T00:00:00Z"
    cap.atomic_json(path, data)
    assert candidate(tmp_path)["rows"][0]["eligible"] == 0
    path.write_text("{")
    assert candidate(tmp_path)["rows"][0]["status"] == "disqualified"
    with pytest.raises(ValueError, match="roster"):
        cap.build_candidate(
            tmp_path,
            tmp_path / "design.md",
            tmp_path / "research-roadmap.yaml",
            "20261001",
            invocations=[],
        )
    (tmp_path / "research-roadmap.yaml").write_text("invalid: [")
    assert not candidate(tmp_path)["activation_confirmed"]
    assert reduction.references(dict(source_artifact_hashes={str(ref): dict(sha256="hash")}))


def test_owned_qualification_and_terminal(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-7991-TERMINAL: failure disqualifies; blocked readiness stays zero."""
    root = tmp_path / "root"
    fixture(root)
    output = tmp_path / "publish/experiment_7991_v692_capstone.json"
    calls = []

    def plan(private):
        counts = {
            n: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
            for n in validation.OWNED
        }
        cap.atomic_json(private / "coverage.json", dict(files=counts))
        cap.atomic_json(
            private / "cli-receipts" / "negative.json",
            dict(
                name="negative_fixture",
                argv=["private"],
                expected_exit=0,
                actual_exit=0,
                passed=True,
                classification="required",
            ),
        )
        return dict(
            commands=[
                dict(
                    name="publication_gate",
                    argv=["private"],
                    expected_exit=0,
                    deadline_s=60,
                    classification="required",
                )
            ],
            dependency_hashes={},
        )

    def check(cwd, spec, private, durable):
        log = durable / (spec["name"] + str(len(calls)) + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        calls.append(spec["name"])
        report = (
            dict(flagged_count=0)
            if spec["name"] == "adversarial_verify"
            else dict(gates={g: dict(pass_=False) for g in ()})
        )
        log.write_text(json.dumps(report))
        return dict(spec, passed=True, actual_exit=0, log_path=str(log))

    monkeypatch.setattr(validation, "manifest", plan)
    monkeypatch.setattr(validation, "run_check", check)
    assert (
        validation.qualify(
            root, root / "design.md", root / "research-roadmap.yaml", "20261001", output
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["capstone_execution_ready_score"] == 0
    selected = json.loads(Path(value["primary_resolution_receipt"]["path"]).read_bytes())
    assert selected["passed"] and selected["document_sha256"] == cap.sha256_file(output)
    value["input_root"] = str(root)
    validation.disqualify(
        value, "owned_failure", dict(name="required", actual_exit=1, log_path=str(output))
    )
    assert value["verdict_class"] == "disqualified" and not value["science_ready"]
    value["validation_receipts"].append(dict(passed=False, classification="required"))
    assert not cap.cold_replay(value, root, root / "design.md", root / "research-roadmap.yaml")
    monkeypatch.setattr(validation, "coverage_complete", lambda *a, **k: False)
    assert (
        validation.qualify(
            root, root / "design.md", root / "research-roadmap.yaml", "20261001", output
        )
        == 0
    )
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    monkeypatch.setattr(validation, "coverage_complete", lambda *a, **k: True)
    fixture(root, True)
    null_output = tmp_path / "null/experiment_7991_v692_capstone.json"
    assert (
        validation.qualify(
            root, root / "design.md", root / "research-roadmap.yaml", "20261001", null_output
        )
        == 0
    )
    assert json.loads(null_output.read_bytes())["capstone_execution_ready_score"] == 1


def test_manifest_and_owned_cli_selection(tmp_path, monkeypatch):
    """REQ-VERIFY-7991-V692: freeze explicit includes and consumer checks."""
    plan = validation.manifest(tmp_path)
    assert plan["coverage_includes"] == validation.OWNED
    assert len(plan["required_negative_routes"]) == 4
    assert all(r["expected_exit"] == 0 for r in plan["commands"])
    assert not any("tests/python" in r["argv"] for r in plan["commands"])
    monkeypatch.setattr(validation, "qualify", lambda *a: 0)
    assert cli.main(["--date", "20261001"]) == 0
    with pytest.raises(SystemExit):
        cli.main([])


def test_checkpoint_routes_and_spans(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7991-REDUCTION: invoke each available producer's own reader."""
    from carnot import experiment_7980_v692_evidence_features as features
    from carnot import experiment_7982_v692_multivariate_energy as energy
    from carnot import experiment_7984_v692_evidence_ablation as ablation
    from carnot.verify import qwen_stream_capture_7981 as capture
    from carnot.reporting import service_cost_7989 as service
    from carnot.reporting import experiment_7990_v692_hardware_evidence as hardware
    from carnot.reporting import arc_supervisor_v692_delta as arc

    seen = []

    def reader(name):
        def reduce(value):
            seen.append(name)
            return dict(passed=True)

        return reduce

    for module, name in ((features, "features"), (energy, "energy"), (ablation, "ablation")):
        monkeypatch.setattr(module, "replay", reader(name))
    monkeypatch.setattr(capture, "reduce", reader("capture"))
    monkeypatch.setattr(service, "reduce", reader("service"))
    monkeypatch.setattr(service, "replay", lambda *a: None)
    monkeypatch.setattr(hardware, "cold_reduce", lambda *a: dict(passed=True))
    monkeypatch.setattr(arc, "replay", lambda *a: [])
    roster = tmp_path / "roster.json"
    cap.atomic_json(roster, dict(rows=[]))
    assert (
        reduction.audit(
            7980, dict(public_features=True, fresh_roster=dict(path=str(roster)), rows=[]), tmp_path
        )["checkpoint_reduction"]["reserved_sources"]
        == 0
    )
    cap.atomic_json(roster, dict(rows=[dict(family_id="reserved")]))
    assert (
        reduction.audit(
            7980, dict(public_features=True, fresh_roster=dict(path=str(roster)), rows=[]), tmp_path
        )["checkpoint_reduction"]["reserved_independence"]
        == "exposure audit required"
    )
    for number, data in (
        (7981, dict(raw_response_shards=True)),
        (7982, dict(heads_seal=True)),
        (7984, dict(checkpoints=True)),
        (7988, dict(new_event_rows=[])),
        (7990, dict(board_rows=[{}])),
    ):
        assert reduction.audit(number, dict(data, rows=[]), tmp_path)["checkpoint_reduction"]
    row = dict(
        arm="scalar",
        status="generated",
        ticks_ns=list(range(8)),
        exclusive_phase_spans={p: 1 for p in service.PHASES},
        wall_ns=7,
    )
    assert reduction.audit(7989, dict(service_rows=True, rows=[row]), tmp_path)[
        "checkpoint_reduction"
    ]["exclusive_spans_reconstructed"]
    row["wall_ns"] = 99
    with pytest.raises(ValueError, match="span"):
        reduction.audit(7989, dict(service_rows=True, rows=[row]), tmp_path)
    row["wall_ns"] = 7
    monkeypatch.setattr(service, "replay", lambda *a: (_ for _ in ()).throw(ValueError("changed")))
    assert not reduction.audit(7989, dict(rows=[row]), tmp_path)["checkpoint_reduction"][
        "service_replay"
    ]["passed"]
    assert set(seen) == {"features", "energy", "ablation", "capture", "service"}


def test_terminal_failure_flags_and_consumer_drift(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-7991-TERMINAL: failed reports cannot certify a primary."""
    root = tmp_path / "root"
    fixture(root)
    value = candidate(root)
    value["input_root"] = str(root)
    output = tmp_path / "publish/experiment_7991_v692_capstone.json"
    attempts = []

    def check(cwd, spec, private, durable):
        p = durable / (spec["name"] + str(len(attempts)) + ".log")
        p.parent.mkdir(parents=True, exist_ok=True)
        attempts.append(spec["name"])
        p.write_text(json.dumps(dict(flagged_count=int(len(attempts) == 1))))
        return dict(spec, actual_exit=0, passed=True, log_path=str(p))

    monkeypatch.setattr(validation, "run_check", check)
    validation.terminal(value, output, tmp_path / "private", tmp_path / "durable")
    assert value["verdict_class"] == "disqualified"
    assert not value["capstone_execution_ready_score"]
    monkeypatch.setattr(validation, "reader_receipt", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="reader_drift"):
        validation.terminal(value, output, tmp_path / "other", tmp_path / "durable")

    def failed(cwd, spec, private, durable):
        p = durable / (spec["name"] + ".log")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("invalid")
        return dict(spec, actual_exit=1, passed=False, log_path=str(p))

    monkeypatch.setattr(validation, "run_check", failed)
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        validation.terminal(value, output, tmp_path / "bad", tmp_path / "durable")


def test_replay_owned_readiness_and_retirement(tmp_path):
    """SCENARIO-REPORT-7991-REPLAY: exact previous scope is required for retirement."""
    tasks = fixture(tmp_path, True)
    task = tasks[9]
    path = tmp_path / task["deliverable"]
    data = json.loads(path.read_bytes())
    data["claim_scope"] = "observational"
    prior = task["prior_failures"][-1]
    data["honest_verdict"] = prior["verdict"]
    cap.atomic_json(path, data)
    old = tmp_path / f"results/experiment_{prior['experiment_id'][3:7]}_prior.json"
    cap.atomic_json(
        old,
        dict(
            task_id=prior["experiment_id"],
            honest_verdict=prior["verdict"],
            claim_scope="observational",
        ),
    )
    value = candidate(tmp_path)
    assert any(r["decision"] == "retire_unchanged_scope" for r in value["retirement_decisions"])
    value["verdict_class"] = "disqualified"
    assert "unsubstantiated_disqualification" in cap.cold_replay(
        value, tmp_path, tmp_path / "design.md", tmp_path / "research-roadmap.yaml"
    )
    value["verdict_class"] = "positive"
    assert "verdict_class_changed" in cap.cold_replay(
        value, tmp_path, tmp_path / "design.md", tmp_path / "research-roadmap.yaml"
    )
    value["verdict_class"] = "blocked"
    value["capstone_execution_ready_score"] = 1
    assert "unsafe_readiness" in cap.cold_replay(
        value, tmp_path, tmp_path / "design.md", tmp_path / "research-roadmap.yaml"
    )


def test_changed_invocation_and_malformed_rows(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7991-REDUCTION: preserve valid rows after checkpoint failure."""
    tasks = fixture(tmp_path, True)
    before = candidate(tmp_path)
    frozen = [
        {k: r[k] for k in ("path", "sha256", "identity", "historical_timestamps")}
        for r in before["producer_date_rows"]
    ]
    frozen[0]["identity"]["run_date"] = "forged"
    value = cap.build_candidate(
        tmp_path,
        tmp_path / "design.md",
        tmp_path / "research-roadmap.yaml",
        "20261001",
        invocations=frozen,
    )
    assert not value["producer_date_rows"][0]["passed"]
    path = tmp_path / tasks[0]["deliverable"]
    data = json.loads(path.read_bytes())
    data["rows"] = [None]
    cap.atomic_json(path, data)
    assert (
        candidate(tmp_path)["independent_reduction_rows"][0]["unavailable_reason"]
        == "malformed_primitive_row"
    )
    original = reduction.audit

    def broken(number, data, root):
        if number == 7982:
            raise ValueError("changed_checkpoint")
        return original(number, data, root)

    monkeypatch.setattr(reduction, "audit", broken)
    assert (
        candidate(tmp_path)["independent_reduction_rows"][3]["unavailable_reason"]
        == "changed_checkpoint"
    )
