"""REQ-REPORT-8004, REQ-VERIFY-8004, REQ-SELF-8004: private evidence routes."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import os

import pytest
import yaml

from carnot.reporting import v693_capstone as cap
from carnot.reporting import v693_capstone_reduction as reduction
from carnot.reporting import v693_capstone_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from scripts.experiments import experiment_8004_v693_capstone as cli

ROOT = Path(__file__).resolve().parents[2]


def fixture(root):
    """A separate authority and producer set keeps protocol evidence circular."""
    root.mkdir(parents=True, exist_ok=True)
    raw = gzip.decompress((ROOT / "tests/fixtures/v693/active.yaml.gz").read_bytes())
    (root / "research-roadmap.yaml").write_bytes(raw)
    roadmap = yaml.safe_load(raw)
    tasks = roadmap["tasks"]
    table = ["| Order | ID | Title | Phase | Deliverable |", "| --- | --- | --- | --- | --- |"]
    table += [
        f"| {i} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
        for i, t in enumerate(tasks, 1)
    ]
    design = "# Private V693 oracle\n\n## Exact task contract\n\n" + "\n".join(table)
    design += (
        "\nCanonical full-task SHA-256: `" + cap.authority.lifecycle.tasks_digest(tasks) + "`\n"
    )
    design += "\n<!-- V693_TASK_CONTRACT_START -->\n```json\n" + json.dumps(roadmap) + "\n```\n"
    (root / "design.md").write_text(design)
    for t in tasks[:-1]:
        p = root / t["deliverable"]
        atomic_json(
            p,
            dict(
                experiment_id=int(t["id"][3:7]),
                task_id=t["id"],
                verdict_class="null",
                honest_verdict="complete_null_private",
                execution_date="20261001",
                flagged_adversarial=False,
                rows=[],
            ),
        )
    return root / "research-roadmap.yaml", root / "design.md"


def test_source_reduction_and_controls():
    """SCENARIO-REPORT-8004-REDUCTION: seeds and unknowns cannot inflate support."""
    rows = [
        dict(
            source_cluster_id="a",
            family_id="a",
            arm="frozen",
            seed=s,
            probability=0.5,
            y=0,
            action="escalate",
        )
        for s in (1, 2)
    ]
    rows += [dict(rows[0], arm="learned", probability=0.01, action="accept")]
    rows += [dict(rows[0], source_cluster_id="unknown", family_id="unknown", y=None)]
    got = reduction.score(rows, {})
    assert got["independent_sources"] == 1 and got["unknown_labels"] == 1
    assert got["comparisons"][0]["cost_left_minus_right"] == 0.25
    assert reduction.controls()["passed"]
    for changed in ({"probability": float("nan")}, {"probability": 1.1}, {"actual_cost": 4}):
        with pytest.raises(ValueError):
            reduction.score([dict(rows[0], **changed)], {})
    with pytest.raises(ValueError):
        reduction.score([None], {})
    assert reduction.score([dict(rows[0], eligibility=False)], {})["scored_rows"] == 0


def test_service_and_confidence_primitives():
    """REQ-SELF-8004: service and confidence use their actual primitive denominators."""
    service = dict(
        arm="sparse",
        case="durable",
        wall_ns=10,
        ticks_ns=[0, 4, 10],
        exclusive_phase_spans={"parse": 4, "head_prediction": 6},
        acquisition={"duration_s": 2},
        cached_total_s=1e-8,
        complete_total_s=2.00000001,
    )
    got = reduction.audit(dict(rows=[service]), {})
    assert got["service"][0]["complete_p50_s"] == 2.00000001
    for changes in ({"wall_ns": 11}, {"complete_total_s": 7}):
        with pytest.raises(ValueError):
            reduction.audit(dict(rows=[dict(service, **changes)]), {})
    unknown = dict(service, acquisition=None, complete_total_s=None)
    assert reduction.audit(dict(rows=[unknown]), {})["service"][0]["missing_acquisition"] == 1
    conf = dict(
        arm="frozen",
        source_cluster_id="a",
        family_id="a",
        p=0.2,
        y=0,
        prediction_set=[0, 1],
        issue_slot=1,
        release_slot=21,
        delay=20,
        error=0,
        action="escalate",
    )
    assert reduction.audit(dict(rows=[conf]), {})["confidence"][0]["coverage"] == 1
    for changes in ({"error": 1}, {"release_slot": 0}):
        with pytest.raises(ValueError):
            reduction.audit(dict(rows=[dict(conf, **changes)]), {})


def test_missing_producer_and_history(tmp_path):
    """SCENARIO-REPORT-8004-CUSTODY: exact missing paths survive arbitrary neighbours."""
    active, design = fixture(tmp_path)
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    (tmp_path / tasks[2]["deliverable"]).unlink()
    atomic_json(tmp_path / "results/experiment_7994_unrelated.json", {"verdict_class": "positive"})
    value = cap.build(tmp_path, active, design, "20261002", tmp_path / "snapshots")
    assert len(value["task_dispositions"]) == 13
    assert value["verdict_class"] == "blocked" and value["capstone_execution_ready_score"] == 0
    assert value["task_dispositions"][2]["producer_status"] == "missing"
    assert value["generalized_learning_benefit_score"] == 0
    assert value["gate_check_summary"]
    assert all(
        {"upstream_id", "path", "hash", "artifact_field", "op", "expected", "observed"} <= set(r)
        for r in value["gate_check_summary"]
    )
    with pytest.raises(ValueError):
        cap.build(tmp_path, active, design, "20261001", tmp_path / "snapshots")


def test_replay_and_private_real_cli(tmp_path):
    """REQ-VERIFY-8004: real private success, missing input and cold replay exits."""
    root = tmp_path / "input"
    active, design = fixture(root)
    out = tmp_path / "experiment_8004_v693_capstone.json"
    prefix = [str(ROOT / ".venv/bin/python"), str(ROOT / cap.OWNED[-1]), "--date", "20261002"]
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    for args, expected in (
        (
            [
                "--root",
                str(root),
                "--active",
                str(active),
                "--design",
                str(design),
                "--output",
                str(out),
                "--evidence-only",
            ],
            0,
        ),
        (["--cold-replay", str(out)], 0),
    ):
        result = subprocess.run(
            prefix + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        assert result.returncode == expected, result.stdout + result.stderr
    value = json.loads(out.read_text())
    assert not cap.cold_replay(value)
    changed = deepcopy(value)
    changed["independent_reduction_rows"][0]["scored_rows"] += 1
    atomic_json(out, changed)
    result = subprocess.run(
        prefix + ["--cold-replay", str(out)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert cli.main(["--date", "20261002", "--cold-replay", str(out), "--expect-rejection"]) == 0
    assert cli.main(["--date", "20261002", "--cold-replay", str(tmp_path / "absent")]) == 1
    value["generalized_learning_benefit_score"] = 1
    assert "unsupported_generalization" in cap.cold_replay(value)
    value["capstone_execution_ready_score"] = 1
    assert "unsafe_readiness" in cap.cold_replay(value)
    value["canonical_tasks_sha256"] = "changed"
    assert "authority_drift" in cap.cold_replay(value)
    (root / yaml.safe_load(active.read_bytes())["tasks"][0]["deliverable"]).write_text("changed")
    assert cap.cold_replay(value) == ["source_bytes_changed"]


def test_manifest_owned_failure_and_publication_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8004-TERMINAL: owned failure and publication rejection remove readiness."""
    frozen = validation.manifest(tmp_path)
    assert frozen["coverage_includes"] == cap.OWNED
    assert {"private_success", "private_blocked", "private_cold_replay", "full_suite"} <= {
        r["name"] for r in frozen["commands"]
    }
    active, design = fixture(tmp_path / "root")
    value = cap.build(tmp_path / "root", active, design, "20261002", tmp_path / "snapshots")
    validation.apply_checks(
        value,
        [
            {
                "name": "owned",
                "passed": False,
                "actual_exit": 1,
                "expected_exit": 0,
                "classification": "required",
            }
        ],
        {},
    )
    assert value["verdict_class"] == "disqualified" and value["capstone_execution_ready_score"] == 0
    validation.apply_checks(
        value,
        [{"name": "health", "passed": False, "actual_exit": 2, "classification": "diagnostic"}],
        {},
    )
    assert value["verdict_class"] == "disqualified"
    out = tmp_path / "experiment_8004_v693_capstone.json"
    atomic_json(out, {"old": True})
    monkeypatch.setattr(
        validation, "run_check", lambda *a, **k: dict(passed=False, actual_exit=1, name="terminal")
    )
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        validation.publish(value, out, tmp_path / "private", tmp_path / "raw")
    assert json.loads(out.read_text()) == {"old": True}


def test_causal_checkpoint_equations(tmp_path):
    """REQ-SELF-8004: saved gradients, decay and delayed changes agree with primitive states."""
    states = tmp_path / "states"
    gradient = dict(
        due_slot=21, origin_slot=1, coefficient_ids=[0], data_gradient=[0.5], weight=2, pi=0.5, y=0
    )
    previous = dict(head=dict(parameters=[1.0], decay_scale=1.0))
    current = dict(head=dict(parameters=[1 - 0.01 / 0.99998], decay_scale=0.99998))
    for slot, state in ((20, previous), (21, current)):
        atomic_json(
            states / f"committed-{slot:04d}.json", dict(state=state, checksum=canonical_hash(state))
        )
    gradients = tmp_path / "gradients.json"
    atomic_json(gradients, dict(rows=[gradient]))
    bundle = tmp_path / "bundle.json"
    atomic_json(bundle, dict(trajectories={"targeted_ipw-101": dict(state_directory=str(states))}))
    data = dict(
        checkpoints=dict(bundle=dict(path=str(bundle), sha256=sha256_file(bundle))),
        independent_reduction_rows=[
            dict(
                arm="targeted_ipw",
                seed=101,
                gradients=dict(path=str(gradients), sha256=sha256_file(gradients)),
            )
        ],
    )
    got = reduction.causal(data)
    assert got["updates_checked"] == 1 and got["maximum_gradient_error"] < 1e-10
    for change in ({"due_slot": 2}, {"data_gradient": [0.7]}):
        atomic_json(gradients, dict(rows=[dict(gradient, **change)]))
        data["independent_reduction_rows"][0]["gradients"]["sha256"] = sha256_file(gradients)
        with pytest.raises(ValueError):
            reduction.causal(data)
    data["checkpoints"]["bundle"]["sha256"] = "changed"
    with pytest.raises(ValueError):
        reduction.causal(data)


def test_private_qualification_and_terminal_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8004: qualification uses actual receipts, preserves health and publishes exact bytes."""
    root = tmp_path / "input"
    cap.prepare_fixture(root)
    active, design = root / "research-roadmap.yaml", root / "design.md"
    original_manifest = validation.manifest
    complete = True

    def plan(private):
        frozen = original_manifest(private)
        if complete:
            atomic_json(
                private / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
                        for p in cap.OWNED
                    }
                ),
            )
        frozen["commands"] = [
            r for r in frozen["commands"] if r["name"] in {"publication_gate", "owned_tests"}
        ]
        return frozen

    def check(cwd, spec, scratch, durable):
        log = durable / (spec["name"] + ".json")
        payload = dict(flagged_count=0)
        if spec["name"] == "publication_gate":
            payload = dict(
                paper_ready=False,
                unmet_gates=["G2"],
                gates={f"G{i}": dict(pass_=i != 2) for i in range(1, 5)},
            )
            payload["gates"] = {k: {"pass": v["pass_"]} for k, v in payload["gates"].items()}
        atomic_json(log, payload)
        return dict(
            **spec, passed=True, actual_exit=0, log_path=str(log), log_sha256=sha256_file(log)
        )

    monkeypatch.setattr(validation, "manifest", plan)
    monkeypatch.setattr(validation, "run_check", check)
    out = tmp_path / "published/experiment_8004_v693_capstone.json"
    assert validation.qualify(root, active, design, "20261002", out) == 0
    value = json.loads(out.read_text())
    assert value["sample_size_budget"]["completed"] == 13
    assert value["task_dispositions"][-1]["completed"]
    assert not cap.cold_replay(value)
    complete = False
    assert validation.qualify(root, active, design, "20261002", out) == 0
    assert json.loads(out.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(cap, "append_retirements", lambda *a: None)
    assert validation.qualify(ROOT, active, design, "20261002", out) == 0
    monkeypatch.setattr(cli.validation, "qualify", lambda *a: 0)
    assert cli.main(["--date", "20261002", "--root", str(root)]) == 0
    assert (
        cli.main(
            [
                "--date",
                "20261002",
                "--root",
                str(root),
                "--active",
                str(active),
                "--design",
                str(design),
                "--output",
                str(tmp_path / "blocked.json"),
                "--evidence-only",
                "--missing-producer-fixture",
            ]
        )
        == 0
    )


def test_source_contract_errors_and_retirement(tmp_path):
    """SCENARIO-REPORT-8004-CUSTODY: malformed inputs, sidecars and exact retirement stay distinct."""
    root = tmp_path / "input"
    cap.prepare_fixture(root)
    active, design = root / "research-roadmap.yaml", root / "design.md"
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    p = root / tasks[0]["deliverable"]
    with pytest.raises(ValueError, match="producer_object_required"):
        p.write_text("[]")
        cap.read(p)
    p.write_text("bad json")
    data, refs, failures, _ = cap.collect(root, tasks)
    assert data[0]["verdict_class"] == "disqualified"
    assert failures and refs
    cap.prepare_fixture(root)
    d = json.loads(p.read_text())
    d["rows"] = [dict(probability=3)]
    d["raw_shard_hashes"] = [dict(path=str(root / "absent"), sha256="changed")]
    atomic_json(p, d)
    terminal = Path(d["terminal_validation_sidecar_path"])
    atomic_json(terminal, dict(primary_sha256="wrong", sidecar_path=str(root / "bound.json")))
    atomic_json(root / "bound.json", dict(primary_sha256="wrong", report=dict(passed=False)))
    data, _, failures, _ = cap.collect(root, tasks)
    assert {"primitive_reduction", "sha256", "bound_validation_passed"} <= {
        r["artifact_field"] for r in failures
    }
    rows = cap.retirements(
        tasks, data + [dict(honest_verdict="complete_null_different", path=None, sha256=None)]
    )
    assert not any(r["retire"] for r in rows)
    tasks.reverse()
    active.write_text(yaml.safe_dump(dict(tasks=tasks)))
    with pytest.raises(ValueError, match="thirteen_task_roster_required"):
        cap.build(root, active, design, "20261002", tmp_path / "snapshots")


def test_negative_owned_invariants(tmp_path, monkeypatch):
    """REQ-VERIFY-8004: target identity, span names, corrupt states and reader drift must fail."""
    with pytest.raises(ValueError, match="missing_source_identity"):
        reduction.score([dict(probability=0.1, y=0)], {})
    row = dict(family_id="a", arm="x", probability=0.1, y=0)
    with pytest.raises(ValueError, match="source_target_drift"):
        reduction.score([row, dict(row, y=1)], {})
    assert not reduction.score([dict(row, role="fit"), dict(row, role="evaluation")], {})[
        "comparisons"
    ]
    with pytest.raises(ValueError, match="exclusive_span_drift"):
        reduction.audit(
            dict(
                rows=[
                    dict(
                        wall_ns=10,
                        ticks_ns=[0, 4, 10],
                        exclusive_phase_spans=dict(parse=3, head_prediction=7),
                    )
                ]
            ),
            {},
        )
    states = tmp_path / "states"
    atomic_json(states / "committed-0020.json", dict(state={}, checksum="wrong"))
    gradients = tmp_path / "gradients.json"
    atomic_json(gradients, dict(rows=[dict(due_slot=21, origin_slot=1)]))
    bundle = tmp_path / "bundle.json"
    atomic_json(bundle, dict(trajectories={"x-1": dict(state_directory=str(states))}))
    with pytest.raises(ValueError, match="causal_state_checksum_drift"):
        reduction.causal(
            dict(
                checkpoints=dict(bundle=dict(path=str(bundle), sha256=sha256_file(bundle))),
                independent_reduction_rows=[
                    dict(
                        arm="x",
                        seed=1,
                        gradients=dict(path=str(gradients), sha256=sha256_file(gradients)),
                    )
                ],
            )
        )
    root = tmp_path / "input"
    cap.prepare_fixture(root)
    (root / "design.md").write_text("malformed authority")
    active = root / "research-roadmap.yaml"
    value = cap.build(root, active, root / "design.md", "20261002", tmp_path / "snapshots")
    assert value["gate_check_summary"]
    log = tmp_path / "validator.json"
    atomic_json(log, dict(flagged_count=1))
    monkeypatch.setattr(validation, "run_check", lambda *a: dict(passed=True, log_path=str(log)))
    with pytest.raises(ValueError, match="critical_adversarial_flag"):
        validation.publish(
            value,
            tmp_path / "experiment_8004_v693_capstone.json",
            tmp_path / "private",
            tmp_path / "raw",
        )
    atomic_json(log, dict(flagged_count=0))
    monkeypatch.setattr(validation, "reader_receipt", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="primary_reader_drift"):
        validation.publish(
            value,
            tmp_path / "experiment_8004_v693_capstone.json",
            tmp_path / "private",
            tmp_path / "raw",
        )

    def reader(task, directory, **kw):
        if directory.name == "readers":
            digest = sha256_file(directory / "experiment_8004_v693_capstone.json")
            return dict(passed=True, gate_sha256=digest, document_sha256=digest)
        return dict(gate_sha256="wrong", document_sha256="wrong")

    monkeypatch.setattr(validation, "reader_receipt", reader)
    with pytest.raises(ValueError, match="published_reader_drift"):
        validation.publish(
            value,
            tmp_path / "experiment_8004_v693_capstone.json",
            tmp_path / "private",
            tmp_path / "raw",
        )


def test_exact_retirement_append(tmp_path):
    """REQ-REPORT-8004: exact archive declarations retire narrow scopes without replacing history."""
    previous = tmp_path / "results/old.json"
    atomic_json(previous, dict(honest_verdict="complete_null_service_cost"))
    (tmp_path / "research-complete.yaml").write_text(
        yaml.safe_dump(
            dict(milestones=[dict(tasks=[dict(id="old-service", deliverable="results/old.json")])])
        )
    )
    manifest = tmp_path / "ops/exclusion_manifest.yaml"
    manifest.parent.mkdir()
    original = "retired_extras:\n- id: preserved\n"
    manifest.write_text(original)
    rows = [
        dict(
            task_id="exp8002-service-cost",
            prior_experiment_id="old-service",
            prior_verdict="complete_null_service_cost",
            repeated_exact_verdict=True,
            retire=True,
            scope="unchanged full-service affordability evidence",
            reopen_condition="new useful workload and complete costs",
            producer_path=str(previous),
            producer_sha256=sha256_file(previous),
        )
    ]
    cap.authenticate_retirements(tmp_path, rows)
    assert rows[0]["prior_sha256"] == sha256_file(previous)
    cap.append_retirements(tmp_path, rows)
    assert manifest.read_text().startswith(original)
    before = manifest.read_bytes()
    cap.append_retirements(tmp_path, rows)
    assert manifest.read_bytes() == before
    missing = [dict(rows[0], prior_experiment_id="absent")]
    cap.authenticate_retirements(tmp_path, missing)
    assert not missing[0]["prior_authenticated"]
    manifest.write_text(original + "other:\n- id: other\n")
    with pytest.raises(ValueError, match="retirement_append_schema"):
        cap.append_retirements(tmp_path, rows)
