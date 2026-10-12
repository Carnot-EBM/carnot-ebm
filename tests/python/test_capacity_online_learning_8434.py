"""REQ-VERIFY-8434, REQ-REPORT-8434: causal bounded learning and honest custody."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import block_feedback_state_8433 as s
from carnot.verify import capacity_online_learning_8434 as e


def fixture(tmp_path, arm="uniform_updates", capacity=8):
    """Separate feature and evaluator files make accidental target reads testable."""
    blocks = [s.questions(b) for b in range(5)]
    labels = [
        dict(unit_id=a["id"], y=(i + j) % 2)
        for block in blocks
        for i, q in enumerate(block)
        for j, a in enumerate(q["answers"])
    ]
    tune = [
        dict(unit_id=a["id"], question_hash=q["id"], x=a["x"], y=j % 2)
        for block in blocks[:3]
        for q in block
        for j, a in enumerate(q["answers"])
    ][:150]
    atomic_json(tmp_path / "labels.json", dict(rows=labels))
    atomic_json(tmp_path / "tune.json", dict(rows=tune))
    cfg = e.configuration(
        s.head(),
        blocks,
        17,
        capacity,
        arm,
        e.reference(tmp_path / "labels.json"),
        e.reference(tmp_path / "tune.json"),
    )
    atomic_json(tmp_path / "configuration.json", cfg)
    return cfg


@pytest.mark.parametrize(
    "arm,capacity",
    [
        ("frozen", 8),
        ("FIFO_updates", 16),
        ("uniform_updates", 16),
        ("shuffled_released_feedback", 8),
    ],
)
def test_causal_stream_and_fixed_geometry(tmp_path, arm, capacity):
    """SCENARIO-VERIFY-8434-STREAM: admission cannot inspect evaluator labels."""
    cfg = fixture(tmp_path, arm, capacity)
    out = e.worker(cfg, tmp_path / "stream")
    assert out["cursor"] == 100 and out["version"] == 10
    assert out["pending_high_water"] == 3 * capacity
    assert out["head"]["scaler"] == cfg["head"]["scaler"]
    assert out["head"]["temperature"] == cfg["head"]["temperature"]
    assert out["parameter_updates"] == (0 if arm == "frozen" else 15 * capacity)
    assert (out["head"] == cfg["head"]) == (arm == "frozen")
    steps = e.steps(tmp_path / "stream")
    assert len(steps) == 5
    for step in steps:
        assert len(step["predictions"]) == 60
        assert step["issued_wall_ns"] < step["released_wall_ns"]
        assert len(step["released_feedback"]) == 3 * capacity
        assert step["sealed_state"]["head"] == step["before_state"]["head"]
        assert step["after_state"]["cursor"] == 20 * (step["block"] + 1)
        assert all("y" not in r for r in step["sealed_state"]["pending"])
        assert e.check_step(cfg, step) == step["after_state"]
    assert e.retention(cfg, steps)["rows"]


def test_uniform_labels_and_per_block_shuffle(tmp_path):
    """SCENARIO-VERIFY-8434-STREAM: shuffle changes only released label pairing."""
    cfg = fixture(tmp_path)
    sealed = s.seal(s.initial(cfg), cfg, 0, cfg["blocks"][0])
    labels = e.released(cfg, sealed)
    shuffled = dict(cfg, shuffled=True)
    left = e.update_trace(sealed, cfg, 0, labels)
    right = e.update_trace(sealed, shuffled, 0, labels)
    assert sorted(r["target"] for r in left) == sorted(r["target"] for r in right)
    assert [r["id"] for r in left] == [r["id"] for r in right]
    assert any(a["target"] != b["target"] for a, b in zip(left, right, strict=True))
    with pytest.raises(ValueError, match="unsealed"):
        e.released(cfg, s.initial(cfg))
    bad = dict(sealed, pending=sealed["pending"][:-1])
    with pytest.raises(ValueError):
        e.released(cfg, bad)
    assert e.independent_update()["passed"]


@pytest.mark.parametrize("arm", e.ARMS)
def test_real_process_exit_and_exact_restart(tmp_path, arm):
    """SCENARIO-VERIFY-8434-RESTART: real exit discards live state after block 2."""
    cfg = fixture(tmp_path, arm)
    expected = e.worker(cfg, tmp_path / "baseline")
    argv = [
        sys.executable,
        "-u",
        str(e.ROOT / e.CLI),
        "--worker",
        str(tmp_path / "resume"),
        "--configuration",
        str(tmp_path / "configuration.json"),
    ]
    p = subprocess.run([*argv, "--crash"], capture_output=True, timeout=60)
    assert p.returncode == 73
    p = subprocess.run(argv, capture_output=True, timeout=60)
    assert p.returncode == 0, p.stderr.decode()
    observed = json.loads((tmp_path / "resume/final.json").read_bytes())
    assert observed == expected
    resumed = e.steps(tmp_path / "resume")
    baseline = e.steps(tmp_path / "baseline")
    assert [e.semantic_step(x) for x in resumed] == [e.semantic_step(x) for x in baseline]


def test_rehashed_primitive_tampering(tmp_path):
    """SCENARIO-REPORT-8434-CUSTODY: unchanged operands defeat forged seals."""
    cfg = fixture(tmp_path)
    e.worker(cfg, tmp_path / "stream")
    for field in ("probability", "target", "coefficient", "rng"):
        step = deepcopy(e.steps(tmp_path / "stream")[0])
        if field == "probability":
            step["predictions"][0]["probability"] += 0.1
        elif field == "target":
            key = next(iter(step["released_feedback"]))
            step["released_feedback"][key] ^= 1
        elif field == "coefficient":
            step["after_state"]["head"]["parameters"][0] += 0.1
        else:
            step["after_state"]["admission_rng"][1][0] ^= 1
        step["checkpoint_hash"] = canonical_hash(step["after_state"])
        with pytest.raises(ValueError):
            e.check_step(cfg, step)


def test_blocked_rows_and_qualification(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8434-CUSTODY: absence stays blocked with all intended slots."""
    from carnot.reporting import capacity_learning_report_8434 as r

    work = r.measure(tmp_path / "absent", tmp_path / "raw", tmp_path / "scratch")
    assert work["failures"]
    value = r.build(work, [], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert len(value["rows"]) == 12000 and value["censored_count"] == 12000
    assert len(value["question_units"]) == 4000
    assert value["learning_stream_ready_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["MODEL_SPECS"] == []
    assert any(g["observed"] is None for g in value["gate_check_summary"])


def test_manifest_and_cli_failures(tmp_path):
    """SCENARIO-REPORT-8434-VALIDATION: real child failure modes remain observable."""
    from carnot.reporting import capacity_learning_runner_8434 as runner

    plan = runner.manifest(tmp_path)
    assert {"private_E2E018_consumers", "private_E2E020", "private_E2E021"} <= {
        x["name"] for x in plan
    }
    assert "patch=subprocess,_exit" in (tmp_path / "coverage.ini").read_text()
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--deliberate-error"], 7),
        (["--date", "20200101"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--worker", str(tmp_path / "no_config")], 1),
        (["--private-control"], 2),
    ]:
        p = subprocess.run([*argv, *args], capture_output=True, timeout=45)
        assert p.returncode == expected, p.stderr.decode()


def numerical_history(cfg, directory, stop=5):
    """Build deterministic test operands while preserving the shipped transition rules."""
    current = s.initial(cfg)
    stream = []
    for block in range(stop):
        sealed = s.seal(current, cfg, block, cfg["blocks"][block])
        labels = e.released(cfg, sealed)
        after = s.release(sealed, cfg, block, labels)
        step = dict(
            block=block,
            before_state=current,
            sealed_state=sealed,
            after_state=after,
            predictions=e.predictions(cfg, current, block),
            issued_wall_ns=10 + 2 * block,
            released_wall_ns=11 + 2 * block,
            released_feedback=labels,
            updates=e.update_trace(sealed, cfg, block, labels),
            checkpoint_hash=canonical_hash(after),
            sealed_checkpoint_hash=canonical_hash(sealed),
        )
        atomic_json(directory / "steps" / f"{block:02}.json", step)
        stream.append(step)
        current = after
    atomic_json(directory / "final.json", current)
    return stream


@pytest.fixture
def report_fixture(tmp_path, monkeypatch, tmp_path_factory):
    """Independent fixture operands let reporting branches run without repeated measurement."""
    from carnot.reporting import capacity_learning_report_8434 as r
    from carnot.reporting.v709_execution import child

    import shutil

    cached = tmp_path_factory.getbasetemp() / "report_fixture_cache"
    cached.mkdir(exist_ok=True)
    cfg = fixture(cached)
    source = dict(
        heads=[dict(cfg["head"], seed=seed) for seed in e.SEEDS],
        blocks=cfg["blocks"],
        panel=dict(evaluator_custody=cfg["evaluator_labels"], trainer_view=cfg["retention_labels"]),
        source_refs=[],
    )
    monkeypatch.setattr(r, "operands", lambda root: source)

    def command(name, argv, logs, *, expected=0, deadline=180):
        if "--worker" in argv:
            directory = Path(argv[argv.index("--worker") + 1])
            configuration = json.loads(Path(argv[argv.index("--configuration") + 1]).read_bytes())
            numerical_history(configuration, directory, 3 if "--crash" in argv else 5)
        if "--retention" in argv:
            directory = Path(argv[argv.index("--retention") + 1])
            configuration = json.loads(Path(argv[argv.index("--configuration") + 1]).read_bytes())
            atomic_json(
                Path(argv[argv.index("--operand-output") + 1]),
                e.retention(configuration, e.steps(directory)),
            )
        return child(
            name,
            [
                sys.executable,
                "-u",
                "-c",
                f"print('fixture receipt',flush=True); raise SystemExit({expected})",
            ],
            logs,
            expected=expected,
            deadline=30,
        )

    monkeypatch.setattr(r, "child", command)
    if (cached / "receipts.json").is_file():
        work = json.loads((cached / "raw/measurement.json").read_bytes())
        receipts = json.loads((cached / "receipts.json").read_bytes())["rows"]
    else:
        work = r.measure(r.ROOT, cached / "raw", cached / "private")
        atomic_json(cached / "raw/stream_rows.json", dict(rows=r.reductions(work)["rows"]))
        receipts = [
            child(
                name,
                [sys.executable, "-u", "-c", "print('fixture check',flush=True)"],
                cached / "checks",
                deadline=30,
            )
            for name in sorted(r.REQUIRED_CHECKS)
        ]
        atomic_json(cached / "raw/measurement.json", work)
        atomic_json(cached / "receipts.json", dict(rows=receipts))
    (tmp_path / "raw").mkdir(exist_ok=True)
    for name in ("measurement.json", "stream_rows.json"):
        shutil.copyfile(cached / "raw" / name, tmp_path / "raw" / name)
    return r, work, receipts


def test_qualified_reduction_and_replay(report_fixture, tmp_path):
    """SCENARIO-REPORT-8434-CUSTODY: qualification needs causal evidence, not favorable loss."""
    r, work, receipts = report_fixture
    value = r.build(work, receipts, tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["learning_stream_ready_score"] == 1
    assert value["coefficient_change_count"] == 5400
    assert value["pending_high_water"] == {"8": 24, "16": 48}
    assert len(value["retention_rows"]) == 30000
    assert value["completed_count"] == 12000 and value["independent_count"] == 100
    assert len(value["question_units"]) == 4000
    assert all(x["intended"] == 240 for x in value["seed_losses"])
    assert len(value["restart_rows"]) == 4 and all(x["passed"] for x in value["restart_rows"])
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert r.replay(path)
    value["learning_stream_ready_score"] = 999
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(path, value)
    assert not r.replay(path)
    failed = deepcopy(receipts)
    failed[0]["passed"] = False
    assert (
        r.build(work, failed, tmp_path / "raw", tmp_path / (e.NAME + ".json"))["verdict_class"]
        == "disqualified"
    )


def test_worker_error_controls(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8434-STREAM: invalid targets and stalled children cannot write."""
    cfg = fixture(tmp_path)
    with pytest.raises(ValueError, match="configuration"):
        e.configuration(cfg["head"], cfg["blocks"], 999, 8, "invalid", {}, {})
    sealed = s.seal(s.initial(cfg), cfg, 0, cfg["blocks"][0])
    bad = json.loads((tmp_path / "labels.json").read_bytes())
    for row in bad["rows"]:
        row["y"] = None
    atomic_json(tmp_path / "bad_labels.json", bad)
    with pytest.raises(ValueError, match="unavailable"):
        e.released(dict(cfg, evaluator_labels=e.reference(tmp_path / "bad_labels.json")), sealed)
    atomic_json(tmp_path / "bad_tune.json", dict(rows=[]))
    with pytest.raises(ValueError, match="retention_roster"):
        e.retention(dict(cfg, retention_labels=e.reference(tmp_path / "bad_tune.json")), [])
    monkeypatch.setattr(e, "child", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="child_failed"):
        e.worker(cfg, tmp_path / "child_failure")
    moments = iter([0, 241])
    monkeypatch.setattr(e.time, "monotonic", lambda: next(moments))
    with pytest.raises(TimeoutError):
        e.worker(cfg, tmp_path / "timeout")


def test_private_cli_publication_and_replay(tmp_path):
    """SCENARIO-REPORT-8434-VALIDATION: real private consumers accept honest blocked bytes."""
    from carnot.reporting.primary_publication import read_bound_sidecar

    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    p = subprocess.run(
        [*argv, "--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)],
        capture_output=True,
        timeout=180,
    )
    assert p.returncode == 0, p.stderr.decode()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["consumer"]["passed"]
    assert read_bound_sidecar(output, Path(terminal["publication"]["sidecar_path"]))["report"][
        "passed"
    ]
    outside = tmp_path / "outside"
    outside.mkdir()
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    p = subprocess.run(
        [*argv, "--cold-replay", str(output)],
        cwd=outside,
        env=env,
        capture_output=True,
        timeout=120,
    )
    assert p.returncode == 0 and b"replay_passed" in p.stdout
    value["completed_count"] = 1
    from carnot.reporting import capacity_learning_report_8434 as r

    value["reproducibility_checksum"] = r.checksum(value)
    tamper = tmp_path / "tamper.json"
    atomic_json(tamper, value)
    p = subprocess.run([*argv, "--cold-replay", str(tamper)], capture_output=True, timeout=120)
    assert p.returncode == 1


def test_upstream_resources_and_deadline_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8434-CUSTODY: required qualification and private disk gates fail closed."""
    from carnot.reporting import capacity_learning_report_8434 as r

    actual_read = r.read

    def unqualified(ref):
        value = actual_read(ref)
        if ref["path"].endswith(r.STORE_PATH):
            return dict(value, block_store_ready_score=0)
        return value

    monkeypatch.setattr(r, "read", unqualified)
    with pytest.raises(ValueError, match="unqualified_checkpoint"):
        r.operands(r.ROOT)
    unqualified_work = r.measure(r.ROOT, tmp_path / "unqualified", tmp_path / "unqualified_private")
    assert any(
        g["field"] == "block_store_ready_score" and g["observed"] == 0
        for g in unqualified_work["failures"]
    )
    monkeypatch.setattr(r, "read", actual_read)
    monkeypatch.setattr(r, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    with pytest.raises(ValueError, match="publication"):
        r.operands(r.ROOT)
    monkeypatch.undo()
    actual_resources = r.methods.legacy.resources
    monkeypatch.setattr(
        r.methods.legacy,
        "resources",
        lambda path: dict(actual_resources(path), private_disk_backed_scratch=False),
    )
    work = r.measure(r.ROOT, tmp_path / "resources", tmp_path / "private")
    assert any(g["field"] == "private_disk_memory_tools_deadline" for g in work["failures"])
    monkeypatch.undo()
    from types import SimpleNamespace
    import time

    monkeypatch.setattr(
        r, "time", SimpleNamespace(monotonic_ns=time.monotonic_ns, monotonic=lambda: 1000)
    )
    calls = iter(range(100, 100000, 300))
    monkeypatch.setattr(r.time, "monotonic", lambda: next(calls))
    work = r.measure(r.ROOT, tmp_path / "deadline", tmp_path / "private2")
    assert all(c["missing_reason"] == "measurement_deadline" for c in work["cells"])


def test_actual_evaluator_rejects_issued_tamper(tmp_path):
    """SCENARIO-REPORT-8434-CUSTODY: a real evaluator rejects forged issued versions."""
    cfg = fixture(tmp_path)
    sealed = s.seal(s.initial(cfg), cfg, 0, cfg["blocks"][0])
    issue = dict(state=sealed, predictions=e.predictions(cfg, s.initial(cfg), 0), issued_wall_ns=1)
    issue["predictions"][0]["probability"] += 0.1
    atomic_json(tmp_path / "issue.json", issue)
    argv = [
        sys.executable,
        "-u",
        str(e.ROOT / e.CLI),
        "--release",
        str(tmp_path / "issue.json"),
        "--configuration",
        str(tmp_path / "configuration.json"),
        "--operand-output",
        str(tmp_path / "release.json"),
    ]
    assert subprocess.run(argv, capture_output=True, timeout=30).returncode == 1
    assert not (tmp_path / "release.json").exists()


def test_receipt_tampering_is_not_a_pass(tmp_path):
    """SCENARIO-REPORT-8434-CUSTODY: flags and actual exits decide receipt qualification."""
    from carnot.reporting import capacity_learning_report_8434 as r
    from carnot.reporting.v709_execution import child

    receipt = child(
        "actual", [sys.executable, "-u", "-c", "print('receipt')"], tmp_path, deadline=30
    )
    assert r.receipts_valid([receipt], [])
    bad = dict(receipt, actual_exit=3)
    assert not r.receipts_valid([bad], [])
    bad = dict(receipt, passed=False)
    assert not r.receipts_valid([bad], [])
    Path(receipt["stdout_path"]).write_text("changed")
    assert not r.receipts_valid([receipt], [])
    scanner = child(
        "adversarial",
        [sys.executable, "-u", "-c", 'print(\'{"reports":[{"flags":[{"severity":"warning"}]}]}\')'],
        tmp_path / "scanner",
        deadline=30,
    )
    scanner["passed"] = False
    assert r.receipts_valid([scanner], [dict(severity="warning")])
    assert not r.receipts_valid([scanner], [])


def test_retention_child_failure_and_restart_mismatch(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8434-RESTART: failed retention or altered audit cannot look complete."""
    cfg = fixture(tmp_path)

    def evaluator(name, argv, logs, **kw):
        if name.startswith("release"):
            issue = json.loads(Path(argv[argv.index("--release") + 1]).read_bytes())
            atomic_json(
                Path(argv[argv.index("--operand-output") + 1]),
                dict(
                    labels=e.released(cfg, issue["state"]),
                    released_wall_ns=issue["issued_wall_ns"] + 1,
                ),
            )
            return dict(passed=True)
        return dict(passed=False)

    monkeypatch.setattr(e, "child", evaluator)
    with pytest.raises(ValueError, match="retention_child_failed"):
        e.worker(cfg, tmp_path / "retention_failure")
    with pytest.raises(ValueError, match="restart_audit_state"):
        numerical_history(cfg, tmp_path / "mismatched")
        e.worker(cfg, tmp_path / "mismatched")


def test_runner_preserves_genuine_child_coverage(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8434-VALIDATION: copied shards come from a real covered child."""
    from carnot.reporting import capacity_learning_runner_8434 as runner
    from carnot.reporting.v709_execution import child

    private = tmp_path / "private"
    private.mkdir()
    rc = private / "coverage.ini"
    rc.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(private / ".coverage")
        + "\n"
    )
    cov = str(e.ROOT / ".venv/bin/coverage")
    p = subprocess.run(
        [cov, "run", "--rcfile=" + str(rc), str(e.ROOT / e.CLI), "--deliberate-error"],
        capture_output=True,
        timeout=45,
    )
    assert p.returncode == 7
    p = subprocess.run(
        [cov, "combine", "--rcfile=" + str(rc), "--keep"], capture_output=True, timeout=45
    )
    assert p.returncode == 0, p.stderr.decode()
    p = subprocess.run(
        [cov, "json", "--rcfile=" + str(rc), "-o", str(private / "coverage.json")],
        capture_output=True,
        timeout=45,
    )
    assert p.returncode == 0
    monkeypatch.setattr(
        runner,
        "manifest",
        lambda path: [
            dict(
                name="private_validation",
                argv=[sys.executable, "-u", "-c", "print('private')"],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ],
    )
    output = tmp_path / (e.NAME + ".json")
    from carnot.reporting import capacity_learning_report_8434 as report

    measurement = tmp_path / "raw" / output.stem / "invocations" / "resume" / "measurement.json"
    report.measure(tmp_path / "absent", measurement.parent, private)
    atomic_json(measurement.parent / "execution_manifest.json", dict(unfinished=True))
    (measurement.parent / "validation").mkdir()
    assert runner.run(tmp_path / "absent", output, private, measurement=measurement) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert any("coverage_shards" in x["path"] for x in value["raw_shard_hashes"])


def test_replay_negative_paths(report_fixture, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8434-CUSTODY: independent operands reject repaired local reductions."""
    r, work, receipts = report_fixture
    value = r.build(work, receipts, tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    original_read = r.read
    for field in [
        "code_hash",
        "authority_hash",
        "authority_activation",
        "roster",
        "source_missing",
        "configuration",
        "chain",
        "final",
        "retention",
        "restart",
        "update_control",
        "receipts",
        "stream",
    ]:
        changed = deepcopy(work)
        if field == "code_hash":
            changed["code_refs"][0]["sha256"] = "sha256:bad"
        elif field == "authority_hash":
            changed["authority"]["authority_snapshots"]["design"]["sha256"] = "sha256:bad"
        elif field == "authority_activation":
            changed["authority"]["activated"] = False
        elif field == "roster":
            changed["cells"].reverse()
        elif field == "source_missing":
            changed["failures"] = [dict(check="missing")]
        elif field in {"configuration", "chain", "final", "retention"}:
            key = {
                "configuration": "configuration",
                "chain": "steps",
                "final": "final",
                "retention": "retention",
            }[field]
            ref = changed["cells"][0][key]
            if field == "chain":
                ref = ref[0]
            primitive = deepcopy(original_read(ref))
            if field == "configuration":
                primitive["head"]["parameters"][0] += 0.1
            elif field == "chain":
                primitive["before_state"]["version"] = 9
            elif field == "final":
                primitive["cursor"] = 99
            else:
                primitive["rows"][0]["brier"] += 0.1
            path = tmp_path / (field + ".json")
            atomic_json(path, primitive)
            if field == "chain":
                changed["cells"][0][key][0] = e.reference(path)
            else:
                changed["cells"][0][key] = e.reference(path)
        elif field == "restart":
            changed["restart_rows"][0]["passed"] = False
        elif field == "update_control":
            changed["independent_update"]["passed"] = False
        elif field == "receipts":
            changed["process_receipts"][0]["passed"] = False
        else:
            changed["cells"][0]["status"] = "failed"
        # Supply a repaired work envelope while all upstream operands remain independent.
        monkeypatch.setattr(
            r,
            "read",
            lambda ref, edited=changed: (
                edited if ref == value["primitive_reference"] else original_read(ref)
            ),
        )
        if field == "stream":
            stream_path = tmp_path / "raw/stream_rows.json"
            saved = stream_path.read_bytes()
            atomic_json(stream_path, dict(rows=[]))
        assert not r.replay(candidate), field
        if field == "stream":
            stream_path.write_bytes(saved)
        monkeypatch.setattr(r, "read", original_read)


def test_incomplete_slots_keep_missing_bounds(report_fixture, tmp_path):
    """SCENARIO-REPORT-8434-CUSTODY: a failed late cell cannot shrink the intended panel."""
    r, work, receipts = report_fixture
    changed = deepcopy(work)
    cell = changed["cells"][-1]
    cell.update(status="failed", missing_reason="owned_child_failed", steps=cell["steps"][:1])
    cell.pop("final")
    cell.pop("retention")
    value = r.build(changed, receipts, tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["intended_count"] == 12000
    assert value["failed_count"] == 240
    assert value["completed_count"] == 11760
    assert value["learning_stream_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"


def test_required_missing_feature_guard(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8434-CUSTODY: missing features cannot become invented predictions."""
    from carnot.reporting import capacity_learning_report_8434 as r

    head, panel, frozen, features = r.upstream.operands(r.ROOT)
    features = deepcopy(features)
    next(row for row in features["rows"] if row["role"] == "test")["x"] = None
    monkeypatch.setattr(r.upstream, "operands", lambda root: (head, panel, frozen, features))
    with pytest.raises(ValueError, match="required_test_feature_missing"):
        r.operands(r.ROOT)


def test_independent_validation_commands_keep_receipts(tmp_path):
    """SCENARIO-REPORT-8434-VALIDATION: independent children precede dependent coverage reduction."""
    from carnot.reporting import capacity_learning_runner_8434 as runner

    plan = [
        dict(
            name=name,
            argv=[sys.executable, "-u", "-c", "print('actual independent command',flush=True)"],
            expected=0,
            deadline=30,
            scope="owned",
        )
        for name in (
            "owned_tests",
            "private_E2E018_consumers",
            "private_E2E020",
            "private_E2E021",
            "dependent",
        )
    ]
    receipts = runner.qualify(plan, tmp_path)
    assert [r["name"] for r in receipts] == [p["name"] for p in plan]
    assert all(r["passed"] for r in receipts)
    assert receipts[-1]["started_monotonic_ns"] >= max(
        r["ended_monotonic_ns"] for r in receipts[:-1]
    )
