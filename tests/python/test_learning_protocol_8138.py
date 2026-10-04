"""REQ-VERIFY-8138 and REQ-REPORT-8138: private conformance never proves benefit."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.verify import learning_protocol_8138 as e


def test_protocol():
    """REQ-VERIFY-8138: numerical authority survives historical override wording."""
    p = e.protocol()
    assert p["feedback_delay_slots"] == 20
    assert p["learning_rate"] == 0.05
    assert p["opportunities"] == [64, 128, 192]
    assert p["step_grid"] == [1, 0.5, 0.25, 0.125]


def test_events_and_restart(tmp_path):
    """SCENARIO-VERIFY-8138-CONFORMANCE: original time survives delayed restart."""
    rows, labels = e.fixture()
    state = e.run(rows, labels, 101)
    partial = e.run(rows, labels, 101, stop=63)
    restored = e.run(rows, labels, 101, state=json.loads(json.dumps(partial)))
    assert state == restored
    assert len(state["pending"]) == 20
    assert len(state["issued"]) == 256
    assert max(state["released"]) == 236
    assert e.verify_events(state, rows)
    broken = deepcopy(state)
    broken["events"][0]["kind"] = "release_feedback"
    with pytest.raises(ValueError, match="event_reference"):
        e.verify_events(broken, rows)
    assert e.shard(tmp_path, state) == e.shard(tmp_path, state)


def test_slot_and_label_failures():
    """SCENARIO-VERIFY-8138-CONFORMANCE: missing slots and reused labels fail."""
    rows, labels = e.fixture()
    with pytest.raises(ValueError, match="original_slots"):
        e.run(rows[:-1], labels, 101)
    rows[1]["source_cluster_id"] = rows[0]["source_cluster_id"]
    with pytest.raises(ValueError, match="original_slots"):
        e.run(rows, labels, 101)
    rows, labels = e.fixture()
    state = e.run(rows, labels, 101, stop=22)
    state["consumed"].append(3)
    with pytest.raises(ValueError, match="reused_label"):
        e.run(rows, labels, 101, state=state)
    state["baseline_hash"] = "changed"
    with pytest.raises(ValueError, match="baseline_hash"):
        e.run(rows, labels, 101, state=state)


def test_overflow_and_missing_masks():
    """SCENARIO-VERIFY-8138-CONFORMANCE: evicted feedback cannot reappear."""
    rows, labels = e.fixture()
    rows[16]["values"] = None
    labels[27] = None
    state = e.run(rows, labels, 101, capacity=2)
    assert state["lost"]
    assert not state["released"]
    state = e.run(rows, labels, 101)
    assert 17 not in state["training"]
    assert 28 not in state["training"]
    assert not set(state["training"]) & set(state["used_admission"])


def test_scalar_prediction_and_training():
    """REQ-VERIFY-8138: independent arithmetic checks optimized head outputs."""
    rows, labels = e.fixture()
    state = e.genesis(rows, 101)
    head = state["arms"]["fixed_public_center"]
    pool = [dict(rows[i], y=labels[i]) for i in range(20)]
    trained = e.train(head, state["geometry"], pool)
    assert trained["optimizer_step"] == 4
    assert e.probability(trained, state["geometry"], rows[0]["values"]) == pytest.approx(
        e.scalar_probability(trained, state["geometry"], rows[0]["values"]), abs=1e-12
    )
    assert e.train(head, state["geometry"], pool, reference=True)["weights"] == pytest.approx(
        trained["weights"], abs=1e-12
    )


@pytest.fixture(scope="module")
def full_receipt(tmp_path_factory):
    """One complete measured fixture backs control-flow tests without repeating training."""
    raw = tmp_path_factory.mktemp("full8138")
    return e.measure(raw, raw, fixture_mode=True), raw


def test_measure_build_replay_and_retention(tmp_path, full_receipt):
    """SCENARIO-REPORT-8138-TERMINAL: full fixtures carry zero natural benefit."""
    work, raw = deepcopy(full_receipt[0]), full_receipt[1]
    assert work["full_size_validation_receipts"]["seeds"] == 20
    assert work["full_size_validation_receipts"]["prediction_count"] == 256 * 4 * 20
    value = e.build(work, raw, [dict(passed=True)])
    assert value["verdict_class"] == "circular_positive"
    assert value["learning_protocol_ready_score"] == 1
    assert value["generalized_learning_benefit_score"] == 0
    path = tmp_path / (e.NAME + ".json")
    e.atomic_json(path, value)
    assert e.replay(path)
    value["rows"][0]["numerator"] += 0.1
    e.atomic_json(path, value)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "missing.json")
    value = e.build(work, raw, [dict(passed=False)])
    assert value["verdict_class"] == "disqualified"
    assert value["learning_protocol_ready_score"] == 0


def test_external_block(tmp_path, monkeypatch, full_receipt):
    """REQ-REPORT-8138: a missing upstream has a terminal exact operand."""
    cached = json.loads(
        Path(full_receipt[0]["event_reference_rows"][0]["transcript"]["path"]).read_text()
    )
    monkeypatch.setattr(e, "run", lambda *a, **kw: deepcopy(cached))
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw)
    value = e.build(work, raw, [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["learning_protocol_ready_score"] == 0
    check = next(r for r in value["gate_check_summary"] if not r["passed"])
    assert check["expected"] is True
    assert check["observed"] is False


def test_reference_mutations_and_boundaries():
    """REQ-VERIFY-8138: mutations at release, admission and durable boundaries fail."""
    rows, labels = e.fixture()
    state = e.run(rows, labels, 101)
    for mutate in ["pending", "released", "admission"]:
        broken = deepcopy(state)
        if mutate == "pending":
            broken["events"][0]["pending"] = []
        elif mutate == "released":
            broken["released"].pop()
        else:
            event = next(r for r in broken["events"] if r["kind"] == "release_feedback")
            event["kind"] = "admit_once"
            event["labels"] = [1] * 12
        with pytest.raises(ValueError, match="event_reference"):
            e.verify_events(broken, rows)
    for boundary in [64, 90, 128, 192, 236]:
        restored = e.run(rows, labels, 101, stop=boundary)
        assert e.run(rows, labels, 101, state=json.loads(json.dumps(restored))) == state
    with pytest.raises(ValueError, match="evaluator_slots"):
        e.run(rows, labels[:-1], 101)
    partial = e.run(rows, labels, 101, stop=22)
    partial["used_admission"].append(3)
    with pytest.raises(ValueError, match="reused_label"):
        e.run(rows, labels, 101, state=partial)
    labels = [0] * 256
    rows[0]["values"] = None
    state = e.run(rows, labels, 101)
    assert e.verify_events(state, rows)


def test_numerical_corruption(monkeypatch):
    """REQ-VERIFY-8138: optimized numerical drift cannot receive readiness credit."""
    rows, labels = e.fixture()
    with monkeypatch.context() as patch:
        patch.setattr(e, "scalar_probability", lambda *args: 99.0)
        with pytest.raises(ValueError, match="event_reference"):
            e.run(rows, labels, 101)
    original = e.train

    def corrupt(head, geometry, pool, *, reference=False):
        result = original(head, geometry, pool, reference=reference)
        if reference:
            result["weights"][0] += 1
        return result

    monkeypatch.setattr(e, "train", corrupt)
    with pytest.raises(ValueError, match="gradient_reference"):
        e.run(rows, labels, 101)


def test_private_cli_and_script_coverage(tmp_path):
    """SCENARIO-REPORT-8138-TERMINAL: actual script runs outside the checkout."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["JAX_PLATFORMS"] = "cpu"
    if os.environ.get("COVERAGE_RCFILE"):
        env["COVERAGE_PROCESS_START"] = os.environ["COVERAGE_RCFILE"]
    output = tmp_path / (e.NAME + ".json")
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    e.progress("before_private_CLI_subprocess")
    done = subprocess.run(
        [*argv, "--fixture-output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    e.progress("after_private_CLI_subprocess", int(done.returncode == 0))
    assert done.returncode == 0, done.stdout + done.stderr
    e.progress("before_private_cold_replay_subprocess")
    done = subprocess.run(
        [*argv, "--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    e.progress("after_private_cold_replay_subprocess", int(done.returncode == 0))
    assert done.returncode == 0 and "replay_passed" in done.stdout
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    done = subprocess.run(
        [*argv, "--fixture-output", str(blocked), "--root", str(tmp_path / "missing")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    value = json.loads(output.read_text())
    ref = value["event_reference_rows"][0]["transcript"]
    Path(ref["path"]).write_text("{}")
    done = subprocess.run(
        [*argv, "--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert done.returncode == 1 and "reduction_drift" in done.stdout
    done = subprocess.run(
        [*argv, "--date", "20260101"], cwd=tmp_path, env=env, capture_output=True, timeout=30
    )
    assert done.returncode == 2


def test_publication_supervisor_routes(tmp_path, monkeypatch, full_receipt):
    """REQ-REPORT-8138: normal and failed owned exits remain distinct from science."""
    from carnot.reporting import learning_protocol_execution_8138 as runner

    work = deepcopy(full_receipt[0])
    monkeypatch.setattr(e, "measure", lambda *a, **kw: deepcopy(work))
    # The supervisor test isolates process/publication control; real cold replay is tested through the CLI.
    monkeypatch.setattr(
        e,
        "replay",
        lambda p: e.historical.reductions(json.loads(p.read_text())["rows"]) == work["reductions"],
    )
    real_check = runner.check
    monkeypatch.setattr(runner, "run_check", lambda *a, **kw: dict(actual_exit=0, passed=True))
    assert real_check(dict(name="normal"), tmp_path)["normal_exit"]

    def fake_check(spec, private):
        if spec["name"] == "measurement":
            e.atomic_json(Path(spec["argv"][-1]), work)
        return dict(name=spec["name"], actual_exit=0, passed=True, normal_exit=True)

    monkeypatch.setattr(runner, "check", fake_check)
    output = tmp_path / "production" / (e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["required_checks_passed"]
    monkeypatch.setattr(
        runner,
        "check",
        lambda s, p: (
            dict(passed=False, actual_exit=1)
            if s["name"] == "adversarial_verify"
            else fake_check(s, p)
        ),
    )
    output = tmp_path / "disqualified" / (e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert runner.main(["--worker-output", str(tmp_path / "worker.json")]) == 0
    with pytest.raises(ValueError, match="candidate_rejected"):
        runner.main(
            [
                "--fixture-output",
                str(tmp_path / "mutation" / (e.NAME + ".json")),
                "--mutation",
                "transcript",
            ]
        )
    with pytest.raises(SystemExit):
        runner.main(["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))])


def test_reference_rejects_removed_commits_and_state_hash():
    """REQ-VERIFY-8138: an omitted durable boundary or forged state is rejected."""
    rows, labels = e.fixture()
    state = e.run(rows, labels, 101)
    for target in ["durable_commit", "state_hash", "prediction"]:
        bad = deepcopy(state)
        if target == "durable_commit":
            bad["events"] = [r for r in bad["events"] if r["kind"] != target]
        elif target == "state_hash":
            bad["events"][0]["state_hash"] = "forged"
        else:
            bad["issued"][0]["predictions"][e.ARMS[0]] = 0.99
        with pytest.raises(ValueError, match="event_reference"):
            e.verify_events(bad, rows)


def test_direct_custody_and_successful_imports(tmp_path, monkeypatch, full_receipt):
    """REQ-VERIFY-8138: original role/feature and evaluator operands bind separately."""
    from test_independent_online_memory_8116 import world as historical_fixture

    class Factory:
        def mktemp(self, name):
            path = tmp_path / name
            path.mkdir()
            return path

    root = historical_fixture.__wrapped__(Factory())
    upstream = json.loads((root / e.historical.UPSTREAM).read_text())
    upstream["role_manifests"] = {"stream": {"role": "stream"}, "retention": {"role": "retention"}}
    monkeypatch.setattr(e.historical, "authenticate", lambda r, p, f, b: (upstream, b))
    original = e.methods.read_ref
    monkeypatch.setattr(e.methods, "read_ref", lambda b, r: r if "role" in r else original(b, r))
    direct = {k: upstream[k] for k in ["stream_feature_manifest", "retention_feature_manifest"]}
    direct["original_slot_mask"] = {"stream": list(range(1, 257))}
    monkeypatch.setattr(e.methods, "authenticate_stream", lambda *a: direct)
    binder = e.methods.Custody(tmp_path / "custody")
    value, binder = e.authenticate(root, tmp_path / "custody", binder)
    assert value["direct_stream_authentication"] == direct
    assert all(row["passed"] for row in binder.checks)
    old = root / "results/experiment_8116_v702_independent_online_memory.json"
    e.atomic_json(
        old, dict(honest_verdict="complete_disqualified_owned_validation", config={"delay": 8})
    )
    cached = json.loads(
        Path(full_receipt[0]["event_reference_rows"][0]["transcript"]["path"]).read_text()
    )
    monkeypatch.setattr(e, "run", lambda *a, **kw: deepcopy(cached))
    work = e.measure(root, tmp_path / "natural-import")
    assert work["input_ready"] == 1
    assert not work["cited_upstream_artifacts"][-1]["operator_override_accepted"]
    direct["stream_feature_manifest"] = {"path": "forged"}
    with pytest.raises(ValueError):
        e.authenticate(root, tmp_path / "bad", e.methods.Custody(tmp_path / "bad"))


def test_labels_open_only_at_release():
    """REQ-VERIFY-8138: the learner cannot inspect future targets during preflight."""
    rows, targets = e.fixture()

    class Labels(list):
        def __iter__(self):
            raise AssertionError("evaluator must remain opaque until release")

    labels = Labels(targets)
    state = e.run(rows, labels, 101, stop=20)
    assert not state["released"]
    labels[0] = 3
    with pytest.raises(ValueError, match="evaluator_label"):
        e.run(rows, labels, 101, state=state, stop=21)


def test_durable_crash_before_delayed_release(tmp_path):
    """REQ-VERIFY-8138: a real child crash resumes before opening delayed feedback."""
    import os
    import subprocess

    checkpoint = tmp_path / "checkpoint.json"
    program = "\n".join(
        [
            "import os,sys",
            "from pathlib import Path",
            f"sys.path[:0] = [{str(e.ROOT / 'python')!r},{str(e.ROOT)!r}]",
            "from carnot.verify import learning_protocol_8138 as e",
            "rows,labels=e.fixture()",
            "def seal(kind,state):",
            '    if kind=="durable_commit" and state["cursor"]==64:',
            f"        e.atomic_json(Path({str(checkpoint)!r}),state)",
            "        os._exit(73)",
            "e.run(rows,labels,101,seal=seal)",
        ]
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [str(e.ROOT / ".venv/bin/python"), "-c", program],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert done.returncode == 73, done.stderr
    rows, labels = e.fixture()
    state = json.loads(checkpoint.read_text())
    assert state["phase"] == "candidate"
    assert max(state["released"]) == 43
    assert e.run(rows, labels, 101, state=state) == e.run(rows, labels, 101)


def test_remaining_event_guards():
    """REQ-VERIFY-8138: each invalid causal or durable operand fails closed."""
    rows, labels = e.fixture()
    state = e.run(rows, labels, 101)
    for choice in ["order", "defer", "base", "label_time", "durable_heads", "update_boundary"]:
        bad = deepcopy(state)
        if choice == "order":
            bad["events"][0]["kind"] = "unknown"
        elif choice == "defer":
            bad["events"][0]["kind"] = "defer_candidate"
        elif choice == "base":
            item = next(r for r in bad["events"] if r["kind"] == "commit_candidate")
            item["candidates"]["error_center"]["base"]["weights"][0] = 999
        elif choice == "label_time":
            item = next(r for r in bad["events"] if r["kind"] == "issue_prediction")
            item["prediction_hash"] = "wrong"
        elif choice == "durable_heads":
            item = next(r for r in bad["events"] if r["kind"] == "durable_update")
            item["heads"] = {}
        else:
            bad["events"] = [r for r in bad["events"] if r["kind"] != "durable_update"]
        with pytest.raises(ValueError, match="event_reference"):
            e.verify_events(bad, rows)
    assert e.verify_events(e.run(rows, [None] * 256, 101), rows)
    # A candidate without enough future admissions is deferred as a whole block.
    rows, labels = e.fixture()
    labels[64:128] = [None] * 64
    state = e.run(rows, labels, 101)
    assert any(r["kind"] == "defer_candidate" for r in state["events"])


def test_replay_guards(tmp_path, monkeypatch, full_receipt):
    """SCENARIO-REPORT-8138-TERMINAL: rehashed fraud still fails independent replay."""
    work, raw = full_receipt
    original = e.build(work, raw, [dict(passed=True)])
    # Narrow each test to one measured seed; full panel conformance is tested separately.
    original["event_reference_rows"] = original["event_reference_rows"][:1]
    original["rows"] = [r for r in original["rows"] if r["seed"] == 101]
    original["reductions"] = e.historical.reductions(original["rows"])
    path = tmp_path / (e.NAME + ".json")

    def save(value):
        value.pop("reproducibility_checksum", None)
        value["reproducibility_checksum"] = e.canonical_hash(value)
        e.atomic_json(path, value)

    for choice in ["code", "reduction", "score", "readiness"]:
        value = deepcopy(original)
        if choice == "code":
            value["code_config_hashes"][e.MODULE] = "wrong"
        elif choice == "reduction":
            value["reductions"] = {}
        elif choice == "score":
            value["rows"][0]["numerator"] += 0.1
            value["reductions"] = e.historical.reductions(value["rows"])
        else:
            value["learning_protocol_ready_score"] = 0
        save(value)
        assert not e.replay(path)
    with monkeypatch.context() as patch:
        patch.setattr(e, "scalar_probability", lambda *a: 99.0)
        patch.setattr(e, "verify_events", lambda *a: True)
        save(deepcopy(original))
        assert not e.replay(path)
    real_run = e.run
    with monkeypatch.context() as patch:
        patch.setattr(e, "run", lambda *a, **kw: dict(real_run(*a, **kw), seed=999))
        save(deepcopy(original))
        assert not e.replay(path)
    journal = Path(original["event_reference_rows"][0]["durable_journal"]["path"])
    journal_copy = tmp_path / "journal.jsonl"
    journal_copy.write_bytes(journal.read_bytes())
    for choice in ["changed", "extra"]:
        data = journal.read_text().splitlines()
        if choice == "changed":
            row = json.loads(data[0])
            row["phase"] = "wrong"
            data[0] = json.dumps(row)
        else:
            data.append(data[-1])
        journal_copy.write_text("\n".join(data) + "\n")
        value = deepcopy(original)
        value["event_reference_rows"][0]["durable_journal"] = {"path": str(journal_copy)}
        save(value)
        assert not e.replay(path)


def test_import_exception_operand(tmp_path, monkeypatch, full_receipt):
    """REQ-REPORT-8138: an unexpected missing custody operand stays a named block."""
    cached = json.loads(
        Path(full_receipt[0]["event_reference_rows"][0]["transcript"]["path"]).read_text()
    )
    monkeypatch.setattr(e, "run", lambda *a, **kw: deepcopy(cached))

    def fail(*args):
        raise ValueError("missing_reference")

    monkeypatch.setattr(e, "authenticate", fail)
    work = e.measure(tmp_path, tmp_path / "raw")
    assert work["gate_check_summary"][0]["observed"] == "missing_reference"
