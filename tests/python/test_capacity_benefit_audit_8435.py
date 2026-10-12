"""REQ-REPORT-8435, REQ-VERIFY-8435: conditional benefit and independent custody."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import capacity_benefit_audit_8435 as m
from carnot.verify import block_feedback_state_8433 as state
from carnot.verify import capacity_online_learning_8434 as stream
from carnot.reporting import capacity_benefit_report_8435 as report
from carnot.reporting import capacity_benefit_runner_8435 as runner


def panel():
    """Independent question effects make duplicated siblings and seeds visible."""
    rows, retention = [], []
    for capacity in (8, 16):
        for seed in m.SEEDS:
            for arm in m.ARMS:
                for block in range(5):
                    for q in range(20):
                        p = {m.ARMS[0]: 0.6, m.ARMS[1]: 0.5, m.ARMS[2]: 0.3, m.ARMS[3]: 0.7}[arm]
                        for sibling in range(3):
                            rows.append(
                                dict(
                                    question_hash=f"q{block}-{q}",
                                    block=block,
                                    seed=seed,
                                    capacity=capacity,
                                    arm=arm,
                                    probability=p,
                                    brier=p * p,
                                    decision_loss=0.2,
                                    status="completed",
                                )
                            )
                    for slot in range(150):
                        retention.append(
                            dict(
                                question_hash=f"t{slot // 3}",
                                boundary=block,
                                seed=seed,
                                capacity=capacity,
                                arm=arm,
                                brier=0.25 + block * 0.001,
                                baseline_brier=0.25,
                            )
                        )
    return dict(rows=rows, retention_rows=retention, admission_rows=[], release_rows=[])


def test_registered_metrics_and_simultaneous_retention():
    """SCENARIO-REPORT-8435-METRICS: equal siblings/seeds never multiply questions."""
    result = m.analyze(panel())
    assert result["fifo_minus_uniform_brier"] == pytest.approx(0.16)
    assert result["gain_vs_frozen"]["estimate"] == pytest.approx(0.27)
    assert result["conditional_ci95"] == pytest.approx([0.16, 0.16])
    assert result["decision_loss_delta"] == 0
    assert result["retention_simultaneous_upper95"] == pytest.approx(0.004)
    assert result["independent_count"] == 100
    assert result["bootstrap_draws"] == 10000
    assert len(result["block_seed_capacity_controls"]) == 200
    assert result["scientific_outcome"] == "exploratory_positive"
    assert result["later_question_count"] == 80


def test_missing_and_degenerate_controls():
    """SCENARIO-REPORT-8435-METRICS: intended bounds survive incomplete evidence."""
    data = panel()
    data["rows"][0].update(status="censored", brier=None, probability=None, decision_loss=None)
    assert m.analyze(data)["missing_data_bounds"][0]["missing"] == 1
    data = dict(rows=[], retention_rows=[], admission_rows=[], release_rows=[])
    result = m.analyze(data)
    assert result["conditional_ci95"] is None
    assert result["scientific_outcome"] == "unmeasured"
    assert result["retention_simultaneous_upper95"] is None
    data = panel()
    for row in data["rows"]:
        row["probability"] = 0.5
    assert m.analyze(data)["control_degeneracy"]["shuffled_matches_uniform"]


def test_inclusion_and_cluster_averaging():
    """SCENARIO-REPORT-8435-METRICS: empirical frequencies retain designed probabilities."""
    data = panel()
    data["admission_rows"] = [
        dict(
            question_hash="q",
            arm="uniform_updates",
            capacity=8,
            admitted=i < 2,
            inclusion_probability=0.4,
            seed=s,
            block=0,
        )
        for i, s in enumerate(m.SEEDS)
    ]
    result = m.analyze(data)
    assert result["inclusion_frequencies"][0]["empirical"] == 0.4
    # One question's fifteen paired records still make one bootstrap value.
    values = m.question_contrasts(data["rows"], 8)
    assert len(values) == 80
    assert np.asarray(values)[:, 0].mean() == pytest.approx(0.16)


def test_private_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8435-CLI: genuine fresh processes preserve blocked evidence."""
    output = tmp_path / "results" / (m.NAME + ".json")
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    child = subprocess.run(
        [
            *argv,
            "--date",
            "20261012",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--private-control",
        ],
        capture_output=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stderr.decode()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == 12000
    assert value["fifo_minus_uniform_brier"] is None
    assert value["current_llm_invocations"] == 0
    assert (
        subprocess.run(
            [*argv, "--cold-replay", str(output)], capture_output=True, timeout=60
        ).returncode
        == 0
    )
    value["learning_audit_ready_score"] = 1
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(output, value)
    assert (
        subprocess.run(
            [*argv, "--cold-replay", str(output)], capture_output=True, timeout=60
        ).returncode
        == 1
    )
    for args, code in [
        (["--deliberate-error"], 7),
        (["--date", "wrong"], 2),
        (["--private-control"], 2),
        (["--cold-replay", str(tmp_path / "missing")], 1),
    ]:
        assert subprocess.run([*argv, *args], capture_output=True, timeout=60).returncode == code


def primitive_fixture(tmp_path, arm="uniform_updates"):
    """Independent primitives exercise the same replay used for natural evidence."""
    blocks = [state.questions(b) for b in range(5)]
    labels = [
        dict(unit_id=a["id"], y=j % 2)
        for block in blocks
        for q in block
        for j, a in enumerate(q["answers"])
    ]
    tune = [
        dict(unit_id=a["id"], question_hash=q["id"], x=a["x"], y=j % 2)
        for block in blocks
        for q in block
        for j, a in enumerate(q["answers"])
    ][:150]
    atomic_json(tmp_path / "labels.json", dict(rows=labels))
    atomic_json(tmp_path / "tune.json", dict(rows=tune))
    panel = dict(
        evaluator_custody=m.reference(tmp_path / "labels.json"),
        trainer_view=m.reference(tmp_path / "tune.json"),
    )
    heads = [dict(state.head(), seed=s) for s in m.SEEDS]
    source = dict(heads=heads, blocks=blocks, panel=panel)
    cfg = stream.configuration(
        heads[0], blocks, 17, 8, arm, panel["evaluator_custody"], panel["trainer_view"]
    )
    before = state.initial(cfg)
    sealed = state.seal(before, cfg, 0, blocks[0])
    released = stream.released(cfg, sealed)
    after = state.release(sealed, cfg, 0, released)
    step = dict(
        block=0,
        before_state=before,
        sealed_state=sealed,
        after_state=after,
        predictions=stream.predictions(cfg, before, 0),
        released_feedback=released,
        updates=stream.update_trace(sealed, cfg, 0, released),
        issued_wall_ns=1,
        released_wall_ns=2,
        checkpoint_hash=canonical_hash(after),
        sealed_checkpoint_hash=canonical_hash(sealed),
    )
    for name, value in [
        ("cfg", cfg),
        ("step", step),
        ("final", after),
        ("retention", stream.retention(cfg, [step])),
    ]:
        atomic_json(tmp_path / (name + ".json"), value)
    cells = [
        dict(
            identity=f"s{s}-c{c}-{a}",
            seed=s,
            capacity=c,
            arm=a,
            status="censored",
            missing_reason="fixture_absence",
            steps=[],
        )
        for s in m.SEEDS
        for c in (8, 16)
        for a in m.ARMS
    ]
    selected = next(c for c in cells if c["seed"] == 17 and c["capacity"] == 8 and c["arm"] == arm)
    selected.update(
        status="completed",
        missing_reason=None,
        configuration=m.reference(tmp_path / "cfg.json"),
        steps=[m.reference(tmp_path / "step.json")],
        final=m.reference(tmp_path / "final.json"),
        retention=m.reference(tmp_path / "retention.json"),
    )
    return dict(cells=cells), source, selected


@pytest.mark.parametrize("arm", ["frozen", "uniform_updates"])
def test_independent_primitive_reconstruction(tmp_path, arm):
    """SCENARIO-VERIFY-8435-REPLAY: real sparse writes equal an independent dense gradient."""
    work, source, cell = primitive_fixture(tmp_path, arm)
    data = report.reconstruct(work, source)
    assert data["dense_update_controls"] == 24
    assert len(data["rows"]) == 12000
    assert sum(r["status"] == "completed" for r in data["rows"]) == 60
    assert len(data["admission_rows"]) == 4000
    with pytest.raises(ValueError, match="intended_roster"):
        report.reconstruct(dict(cells=[]), source)


@pytest.mark.parametrize(
    "mutation",
    [
        "cfg",
        "order",
        "final",
        "retention",
        "rng",
        "target",
        "hidden",
        "capacity",
        "duplicate",
        "future",
    ],
)
def test_rehashed_primitive_mutations(tmp_path, mutation):
    """SCENARIO-VERIFY-8435-REPLAY: rehashing cannot authorize forbidden causal changes."""
    work, source, cell = primitive_fixture(tmp_path)
    name = (
        "cfg"
        if mutation == "cfg"
        else "final"
        if mutation == "final"
        else "retention"
        if mutation == "retention"
        else "step"
    )
    path = tmp_path / (name + ".json")
    value = json.loads(path.read_bytes())
    if mutation == "cfg":
        value["seed"] = 99
    elif mutation == "final":
        value["cursor"] = 99
    elif mutation == "retention":
        value["rows"][0]["brier"] += 0.1
    elif mutation == "order":
        value["before_state"]["version"] = 4
    elif mutation == "rng":
        value["sealed_state"]["admission_rng"][1][0] += 1
    elif mutation == "target":
        value["updates"][0]["target"] ^= 1
    elif mutation == "hidden":
        value["sealed_state"]["pending"][0]["y"] = 1
    elif mutation == "capacity":
        value["sealed_state"]["pending"].append(value["sealed_state"]["pending"][0])
    elif mutation == "duplicate":
        value["after_state"]["applied_feedback_ids"].append(
            value["after_state"]["applied_feedback_ids"][0]
        )
    else:
        value["released_wall_ns"] = 0
    atomic_json(path, value)
    key = "configuration" if name == "cfg" else name
    if name == "step":
        cell["steps"] = [m.reference(path)]
    else:
        cell[key] = m.reference(path)
    with pytest.raises(ValueError):
        report.reconstruct(work, source)


def test_joint_retention_bound_prevents_window_selection():
    """SCENARIO-REPORT-8435-METRICS: opposite windows need a joint rather than marginal bound."""
    rows = [
        dict(
            question_hash=f"t{q}",
            boundary=b,
            seed=s,
            capacity=8,
            arm=m.ARMS[2],
            brier=0.25 + (0.04 if q % 2 else -0.04) * (1 if b == 0 else -1 if b == 1 else 0),
            baseline_brier=0.25,
        )
        for q in range(50)
        for b in range(5)
        for s in m.SEEDS
        for j in range(3)
    ]
    result = m.retention_bound(rows)
    assert result["upper95"] == pytest.approx(0.0112)
    assert result["upper95"] > result["windows"][0]["upper95"]


def test_real_numeric_child(tmp_path):
    """SCENARIO-VERIFY-8435-REPLAY: a fresh CPU process reconstructs natural primitives."""
    raw = tmp_path / "raw"
    raw.mkdir()
    work = report.measure(m.ROOT, raw, tmp_path)
    assert work["process_receipts"][0]["actual_exit"] == 0
    result = work["result"]
    assert len(result["data"]["rows"]) == 12000
    assert result["data"]["dense_update_controls"] == 7200
    assert result["analysis"]["later_question_count"] == 80
    assert result["analysis"]["scientific_outcome"] == "null"
    atomic_json(raw / "measurement.json", work)
    candidate = tmp_path / (m.NAME + ".json")
    atomic_json(candidate, report.build(work, [], raw, candidate))
    assert report.replay(candidate)
    assert (
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(m.ROOT / m.CLI),
                "--numeric",
                str(tmp_path),
                "--operand-output",
                str(tmp_path / "numeric-missing.json"),
            ],
            capture_output=True,
            timeout=60,
        ).returncode
        == 1
    )


def test_dense_write_negative_control(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8435-REPLAY: the dense control independently catches sparse trace corruption."""
    work, source, cell = primitive_fixture(tmp_path)
    path = tmp_path / "step.json"
    step = json.loads(path.read_bytes())
    step["updates"][0]["coefficient_delta"][0] += 0.1
    atomic_json(path, step)
    cell["steps"] = [m.reference(path)]
    monkeypatch.setattr(report.upstream.e, "check_step", lambda cfg, step: step["after_state"])
    with pytest.raises(ValueError, match="dense_write_control"):
        report.reconstruct(work, source)


def test_numeric_producer_gates(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8435-REPLAY: independent producer seals cannot be self-authorized."""
    primary = dict(
        learning_stream_ready_score=0, required_checks_passed=True, flagged_adversarial=False
    )
    monkeypatch.setattr(report, "read", lambda ref: primary)
    with pytest.raises(ValueError, match="unqualified_upstream"):
        report.numeric(tmp_path)
    primary.update(
        learning_stream_ready_score=1,
        terminal_validation_sidecar_path=str(tmp_path / "terminal.json"),
    )
    atomic_json(tmp_path / "terminal.json", dict(publication=dict(sidecar_path="sidecar")))
    monkeypatch.setattr(report, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    with pytest.raises(ValueError, match="upstream_publication"):
        report.numeric(tmp_path)
    monkeypatch.setattr(report, "read_bound_sidecar", lambda *args: dict(report=dict(passed=True)))
    primary.update(primitive_reference={"path": "work"})
    saved = dict(
        source_refs=[dict(path="missing", sha256="invalid")], code_refs=[], primitive_files=[]
    )
    monkeypatch.setattr(report, "read", lambda ref: saved if ref["path"] == "work" else primary)
    monkeypatch.setattr(report, "sha256_file", lambda p: "other")
    with pytest.raises(ValueError, match="pinned_primitive_hash"):
        report.numeric(tmp_path)


def test_precondition_resources_and_input_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8435-CLI: missing operands differ from measured zero and owned errors."""
    monkeypatch.setattr(
        report.methods.legacy,
        "resources",
        lambda p: dict(
            private_disk_backed_scratch=False, available_memory_bytes=0, minimum_memory_bytes=1
        ),
    )
    monkeypatch.setattr(report, "read", lambda ref: dict(learning_stream_ready_score=0))
    work = report.measure(tmp_path, tmp_path / "raw", tmp_path)
    assert any(g["field"] == "private_disk_memory_tools_deadline" for g in work["failures"])
    assert any(
        g["observed"] == 0 and g["field"] == "learning_stream_ready_score" for g in work["failures"]
    )

    def bad_hash(ref):
        raise ValueError("source_hash")

    monkeypatch.setattr(report, "read", bad_hash)
    work = report.measure(tmp_path, tmp_path / "raw2", tmp_path)
    assert any(g["observed"] == "source_hash" for g in work["failures"])


def test_owned_numeric_child_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8435-CLI: an actual exit7 disqualifies owned reconstruction."""
    from carnot.reporting.v709_execution import child

    monkeypatch.setattr(
        report,
        "child",
        lambda name, argv, logs, **kw: child(
            name,
            [sys.executable, "-u", str(m.ROOT / m.CLI), "--deliberate-error"],
            logs,
            deadline=30,
        ),
    )
    work = report.measure(m.ROOT, tmp_path / "raw", tmp_path)
    assert work["process_receipts"][0]["actual_exit"] == 7
    assert work["failures"][0]["field"] == "owned_numeric_reconstruction"
    atomic_json(tmp_path / "raw/measurement.json", work)
    value = report.build(work, [], tmp_path / "raw", tmp_path / (m.NAME + ".json"))
    assert value["verdict_class"] == "disqualified" and value["learning_audit_ready_score"] == 0


def test_scientific_outcome_and_administrative_qualification(tmp_path):
    """SCENARIO-REPORT-8435-METRICS: an honest zero metric survives administrative rejection."""
    raw = tmp_path / "raw"
    raw.mkdir()
    work = dict(
        result=dict(data=panel(), source_refs=[]),
        process_receipts=[],
        adversarial_findings=[],
        failures=[],
        preconditions={},
        started_monotonic_ns=1,
        ended_monotonic_ns=2,
    )
    receipts = [dict(name=n, passed=True, duration_s=0.1) for n in report.REQUIRED]
    atomic_json(raw / "measurement.json", work)
    value = report.build(work, receipts, raw, tmp_path / (m.NAME + ".json"))
    assert value["verdict_class"] == "circular_positive"
    assert value["decision_loss_delta"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    work["adversarial_findings"] = [
        dict(code="SUSPICIOUS_METRIC", severity="info", detail="honest zero")
    ]
    atomic_json(raw / "measurement.json", work)
    value = report.build(work, receipts, raw, tmp_path / (m.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert (
        value["scientific_outcome"] == "exploratory_positive" and value["decision_loss_delta"] == 0
    )


def test_replay_rejects_rehashed_custody_changes(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8435-REPLAY: outer rehashing cannot grant authority or repair receipts."""
    raw = tmp_path / "raw"
    work = report.measure(tmp_path, raw, tmp_path)
    path = tmp_path / (m.NAME + ".json")

    def candidate():
        atomic_json(raw / "measurement.json", work)
        value = report.build(work, [], raw, path)
        atomic_json(path, value)
        return value

    value = candidate()
    assert report.replay(path)
    value["code_config_hashes"][0]["sha256"] = "invalid"
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    assert not report.replay(path)
    candidate()
    with monkeypatch.context() as scoped:
        scoped.setattr(report.methods, "authority", lambda *args: dict(activated=True))
        assert not report.replay(path)
    work["result"]["data"]["rows"][0]["brier"] = 0
    work["result"]["data"]["rows"][0]["decision_loss"] = 0
    candidate()
    assert not report.replay(path)
    work["result"] = dict(data=report.empty_data(), source_refs=[])
    candidate()
    with monkeypatch.context() as scoped:
        scoped.setattr(report.upstream, "receipts_valid", lambda *args: False)
        assert not report.replay(path)
    work["result"]["source_refs"] = [dict(path="synthetic independent operand", sha256="test")]
    candidate()
    with monkeypatch.context() as scoped:
        scoped.setattr(report, "numeric", lambda root: dict(other=True))
        assert not report.replay(path)
    snapshot = tmp_path / "snapshot.yaml"
    snapshot.write_text("corruption")
    work["authority"]["authority_snapshots"]["active"] = dict(
        snapshot_path=str(snapshot), sha256="wrong"
    )
    candidate()
    assert not report.replay(path)


def test_production_path_with_private_child_shards(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8435-CLI: production wiring copies genuine child coverage and logs."""
    private = tmp_path / "private"
    plan = runner.manifest(private)
    assert {r["name"] for r in plan} == report.REQUIRED
    config = private / "coverage.ini"
    cov = str(m.ROOT / ".venv/bin/coverage")
    assert (
        subprocess.run(
            [cov, "run", "--rcfile=" + str(config), str(m.ROOT / m.CLI), "--deliberate-error"],
            capture_output=True,
            timeout=60,
        ).returncode
        == 7
    )
    for argv in [
        [cov, "combine", "--rcfile=" + str(config), "--keep"],
        [cov, "json", "--rcfile=" + str(config), "-o", str(private / "coverage.json")],
    ]:
        process = subprocess.run(argv, capture_output=True, timeout=60)
        assert process.returncode == 0, process.stdout.decode() + process.stderr.decode()
    monkeypatch.setattr(
        runner,
        "manifest",
        lambda p: [
            dict(
                name="actual_private_child",
                argv=[sys.executable, "-u", "-c", "print('private CPU child',flush=True)"],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ],
    )
    output = tmp_path / "results" / (m.NAME + ".json")
    assert runner.run(tmp_path, output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert any(r["name"] == "actual_private_child" for r in value["validation_receipts"])
    work = m.read(value["primitive_reference"])
    assert work["coverage_shards"]


def test_external_primitive_absence_is_blocked(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8435-CLI: a missing upstream operand is external, not an owned failure."""
    original = report.read
    primary = dict(
        learning_stream_ready_score=1,
        primitive_reference=dict(path=str(tmp_path / "missing.json"), sha256="pinned"),
    )
    monkeypatch.setattr(
        report,
        "read",
        lambda ref: primary if ref["path"].endswith(report.UPSTREAM) else original(ref),
    )
    work = report.measure(m.ROOT, tmp_path / "raw", tmp_path)
    assert not work["process_receipts"]
    assert any(
        g["observed"] is None and g["path"] == str(tmp_path / "missing.json")
        for g in work["failures"]
    )
    atomic_json(tmp_path / "raw/measurement.json", work)
    value = report.build(work, [], tmp_path / "raw", tmp_path / (m.NAME + ".json"))
    assert value["verdict_class"] == "blocked"


def test_nested_primitive_hash_gate(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8435-REPLAY: an absent nested operand is retained by its own path and hash."""
    missing = tmp_path / "nested-missing.json"
    atomic_json(
        tmp_path / "upstream-work.json",
        dict(
            source_refs=[dict(path=str(missing), sha256="sealed")], code_refs=[], primitive_files=[]
        ),
    )
    primary = dict(
        learning_stream_ready_score=1,
        primitive_reference=m.reference(tmp_path / "upstream-work.json"),
    )
    original = report.read
    monkeypatch.setattr(
        report,
        "read",
        lambda ref: primary if ref["path"].endswith(report.UPSTREAM) else original(ref),
    )
    work = report.measure(m.ROOT, tmp_path / "raw", tmp_path)
    assert work["failures"][0]["path"] == str(missing)
    assert work["failures"][0]["expected"] == "sealed"
    assert not work["process_receipts"]
