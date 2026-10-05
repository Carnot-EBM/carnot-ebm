"""REQ-REPORT-8144 / REQ-VERIFY-8144: independent source and causal checks."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import learning_audit_8144 as a
from carnot.verify import learning_protocol_8138 as producer


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    """Fixture truth qualifies replay mechanics without adding natural evidence."""
    raw = tmp_path_factory.mktemp("audit8144-input")
    features, labels, states = {}, {}, []
    for role, count in [("stream", 256), ("retention", 64)]:
        rows, targets = producer.fixture(count, role)
        features[role] = producer.shard(raw, dict(rows=rows))
        labels[role] = producer.shard(
            raw, dict(rows=[dict(r, y=y) for r, y in zip(rows, targets, strict=True)])
        )
    stream = json.loads(Path(features["stream"]["path"]).read_text())["rows"]
    for seed in [101, 102]:
        state = producer.run(stream, producer.fixture()[1], seed)
        states.append(dict(seed=seed, state=producer.shard(raw, state)))
    value = dict(
        experiment_id=8143,
        learning_trajectory_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        fixture_mode=True,
        input_manifests=dict(
            stream_feature_manifest=features["stream"],
            retention_feature_manifest=features["retention"],
            evaluator_label_manifests=labels,
        ),
        state_manifest=states,
    )
    root = raw / "root"
    atomic_json(root / a.UPSTREAM, value)
    return root, stream, raw, value


def test_independent_causal_reconstruction(panel):
    """SCENARIO-VERIFY-8144-1: all final heads and original tail clocks rebuild."""
    _, rows, _, value = panel
    state = json.loads(Path(value["state_manifest"][0]["state"]["path"]).read_text())
    labels = Path(value["input_manifests"]["evaluator_label_manifests"]["stream"]["path"])
    result = a.reconstruct(state, rows, labels)
    assert result["passed"]
    assert result["pending"] == list(range(237, 257))
    assert result["released_count"] == 236
    assert result["overflow_count"] == 0
    assert result["admission_reuse_count"] == 0


@pytest.mark.parametrize(
    "change", ["order", "tail", "prediction", "gradient", "pending", "consumed", "admission"]
)
def test_causal_mutations(panel, change):
    """SCENARIO-VERIFY-8144-1: rehashed producer mutations still fail equations."""
    _, rows, _, value = panel
    state = json.loads(Path(value["state_manifest"][0]["state"]["path"]).read_text())
    if change == "order":
        state["events"][0:2] = reversed(state["events"][0:2])
    elif change == "tail":
        state["events"].append(dict(kind="release_feedback", slot=257, label_slot=237))
    elif change == "prediction":
        state["issued"][90]["predictions"][a.ARMS[0]] += 0.1
    elif change == "gradient":
        candidate = next(
            e for e in state["events"] if e["kind"] == "commit_candidate" and e["candidates"]
        )
        candidate["candidates"][a.ARMS[1]]["head"]["intercept"] += 0.1
    elif change == "pending":
        state["pending"] = []
    elif change == "consumed":
        state["consumed"].append(1)
    else:
        state["used_admission"].append(state["used_admission"][0])
    labels = Path(value["input_manifests"]["evaluator_label_manifests"]["stream"]["path"])
    with pytest.raises((ValueError, KeyError)):
        a.reconstruct(state, rows, labels)


def test_nested_seeds_masks_and_positive_control():
    """SCENARIO-REPORT-8144-2: controls exercise actual source gates and masks."""
    rows = a.control_rows()
    result = a.statistics(rows)
    assert result["h2_passed"] and result["retention_passed"]
    assert result["completed_count"] == 192
    assert result["paired_gain_interval"]["valid_draws"] == 10000
    duplicate = [dict(r, seed=r["seed"] + 100) for r in rows]
    assert a.statistics(rows + duplicate)["completed_count"] == 192
    for row in rows:
        if row["slot"] == 70 and row["condition"] == "later_stream":
            row.update(status="excluded", exclusion_reason="missing", numerator=0)
    masked = a.statistics(rows)
    assert masked["completed_count"] == 191
    assert masked["per_source_results"][5]["gain"] is None
    assert a.positive_control()["passed"]


def test_terminal_block_and_owned_failure(tmp_path):
    """REQ-REPORT-8144: missing external operands finish blocked, not partial."""
    work = a.measure(tmp_path, tmp_path / "raw")
    value = a.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["learning_audit_ready_score"] == 0
    assert value["gate_check_summary"][0]["expected"] is True
    assert value["gate_check_summary"][0]["observed"] is False
    failed = a.build(work, tmp_path / "raw", [dict(passed=False)])
    assert failed["verdict_class"] == "disqualified"


@pytest.fixture(scope="module")
def measured(panel, tmp_path_factory):
    """REQ-VERIFY-8144: independently measure private known-target states."""
    raw = tmp_path_factory.mktemp("audit8144-measured")
    return a.measure(panel[0], raw, fixture=True), raw


def test_measurement_seals_and_replay(measured):
    """SCENARIO-REPORT-8144-1: cold reductions and all private primitives bind."""
    work, raw = measured
    value = a.build(work, raw, [dict(passed=True)])
    assert value["learning_audit_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["h2_development_signal_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["call_ledger"] == []
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    path = raw / (a.NAME + ".json")
    atomic_json(path, value)
    assert a.replay(path)
    assert not a.replay(raw / "absent.json")
    for key in ["rows", "completed_count", "code_config_hashes"]:
        bad = deepcopy(value)
        if key == "rows":
            bad[key][0]["numerator"] += 0.1
        elif key == "completed_count":
            bad[key] += 1
        else:
            bad[key][a.MODULE] = "sha256:invalid"
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not a.replay(path)
    atomic_json(path, value)
    work["positive_control"]["passed"] = False
    assert a.build(work, raw, [dict(passed=True)])["verdict_class"] == "disqualified"
    work["positive_control"]["passed"] = True


@pytest.mark.parametrize(
    "mutation", ["missing_state", "changed_label", "not_ready", "missing_manifests"]
)
def test_external_primitive_blocks(panel, tmp_path, mutation):
    """SCENARIO-REPORT-8144-1: exact blocked operands never become partial."""
    root, _, _, original = panel
    value = deepcopy(original)
    if mutation == "missing_state":
        value["state_manifest"][0]["state"]["path"] = str(tmp_path / "absent.json")
    elif mutation == "changed_label":
        path = tmp_path / "changed.json"
        atomic_json(path, dict(rows=[]))
        value["input_manifests"]["evaluator_label_manifests"]["retention"]["path"] = str(path)
    elif mutation == "not_ready":
        value["learning_trajectory_ready_score"] = 0
    else:
        del value["input_manifests"]
    atomic_json(tmp_path / a.UPSTREAM, value)
    work = a.measure(tmp_path, tmp_path / "raw", fixture=True)
    result = a.build(work, tmp_path / "raw", [dict(passed=True)])
    assert result["verdict_class"] == "blocked"
    failure = next(r for r in result["gate_check_summary"] if not r["passed"])
    assert failure["artifact_field"] and failure["expected"] != failure["observed"]
    assert root.is_dir()


def test_changed_release_label_and_equation_errors(panel, tmp_path):
    """SCENARIO-VERIFY-8144-1: label changes cannot reuse old trained heads."""
    _, rows, _, value = panel
    state = json.loads(Path(value["state_manifest"][0]["state"]["path"]).read_text())
    labels = Path(value["input_manifests"]["evaluator_label_manifests"]["stream"]["path"])
    changed = json.loads(labels.read_text())
    for r in changed["rows"]:
        r["y"] = 1 - r["y"]
    path = tmp_path / "changed.json"
    atomic_json(path, changed)
    with pytest.raises(ValueError):
        a.reconstruct(state, rows, path)
    for x, y in [({"a": 1}, {"b": 1}), ([1], [1, 2]), (0.1, 0.2)]:
        with pytest.raises(ValueError):
            a.equal(x, y)
    assert a.proposal({}, {}, [], [], 101, 64) == {}
    assert a.bootstrap([None] * 192, 16)["valid_draws"] == 0


def child(argv, cwd, expected=0):
    """Script-path tests remove ambient imports and wait with real heartbeats."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    if os.environ.get("COVERAGE_RCFILE"):
        env["COVERAGE_PROCESS_START"] = os.environ["COVERAGE_RCFILE"]
    a.progress("before_private_subprocess", 0, 1)
    with subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        while True:
            try:
                log, _ = process.communicate(timeout=30)
                break
            except subprocess.TimeoutExpired:
                a.progress("private_child_pending", 0, 1)
    a.progress("after_private_subprocess", 1)
    assert process.returncode == expected, log[-4000:]
    return log


def test_private_cli_success_block_mutation_cold(panel, tmp_path):
    """SCENARIO-REPORT-8144-1: real CLI works from outside checkout."""
    py, cli = str(a.ROOT / ".venv/bin/python"), str(a.ROOT / a.CLI)
    output = tmp_path / (a.NAME + ".json")
    child([py, cli, "--root", str(panel[0]), "--fixture-output", str(output)], tmp_path)
    assert json.loads(output.read_text())["learning_audit_ready_score"] == 1
    assert "replay_passed" in child([py, cli, "--cold-replay", str(output)], tmp_path)
    bad = json.loads(output.read_text())
    bad["per_source_results"] = []
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(output, bad)
    assert "reduction_drift" in child([py, cli, "--cold-replay", str(output)], tmp_path, 1)
    blocked = tmp_path / "blocked" / (a.NAME + ".json")
    child([py, cli, "--root", str(tmp_path / "absent"), "--fixture-output", str(blocked)], tmp_path)
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    child([py, cli, "--date", "bad"], tmp_path, 2)
    child([py, cli, "--fixture-output", str(a.ROOT / "results" / (a.NAME + ".json"))], tmp_path, 2)
    child(
        [
            py,
            cli,
            "--root",
            str(tmp_path / "missing"),
            "--worker-output",
            str(tmp_path / "worker/measurement.json"),
        ],
        tmp_path,
    )
    child(
        [
            py,
            cli,
            "--root",
            str(panel[0]),
            "--fixture-output",
            str(tmp_path / "mutated" / (a.NAME + ".json")),
            "--mutation",
        ],
        tmp_path,
        1,
    )


def test_sealed_natural_reconstruction_and_recovery(tmp_path):
    """REQ-VERIFY-8144: actual sealed custody and historical restart are audited."""
    work = a.measure(a.ROOT, tmp_path / "raw")
    assert work["input_ready"] == 1, work["gate_check_summary"][-1]
    assert len(work["causal_order_checks"]) == 20
    assert all(r["passed"] for r in work["restart_parity_rows"])
    value = a.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["learning_audit_ready_score"] == 1
    assert value["h2_development_signal_score"] == 0
    assert value["completed_count"] == 158
    assert value["verdict_class"] == "null"
    assert value["retention_passed"]
    assert value["opportunity_summary"]["error_center"] == dict(
        accepted=20, rejected=0, deferred=40
    )
    assert value["changed_source_predictions"]["later_stream"]["error_center"] == 0
    assert value["changed_source_predictions"]["retention"]["error_center"] == 61


def test_runner_owned_failures_and_historical_preservation(tmp_path, monkeypatch):
    """REQ-REPORT-8144: an owned failure zeroes readiness before publication."""
    from carnot.reporting import learning_audit_execution_8144 as runner

    work = a.measure(tmp_path, tmp_path / "work")
    private = tmp_path / "validation"
    private.mkdir()
    calls = []

    def fake_check(spec, scratch):
        calls.append(spec["name"])
        if spec["name"] == "measurement":
            measurement = Path(spec["argv"][-1])
            atomic_json(measurement, work)
        return dict(name=spec["name"], passed=spec["name"] != "cold_replay", normal_exit=True)

    monkeypatch.setattr(runner, "check", fake_check)
    monkeypatch.setattr(runner, "mkdtemp", lambda **kw: str(private))
    output = tmp_path / (a.NAME + ".json")
    historical = dict(experiment_id=8144, task_id="exp8144-learning-audit", verdict_class="blocked")
    atomic_json(output, historical)
    assert runner.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["learning_audit_ready_score"] == 0
    assert "repository_full_suite" in calls
    assert next(output.parent.glob("raw/*/invocations/*/historical_primary.json")).is_file()
    assert next(output.parent.glob("raw/*/invocations/*/failed_terminal_candidate.json")).is_file()


def test_real_supervisor_receipts(tmp_path):
    """REQ-REPORT-8144: normal child exit and byte hash bind actual validation."""
    from carnot.reporting import learning_audit_execution_8144 as runner

    spec = dict(
        name="bounded_child",
        argv=[str(a.ROOT / ".venv/bin/python"), "-c", "print('completed',flush=True)"],
        expected_exit=0,
        deadline_s=30,
        classification="required",
    )
    receipt = runner.check(spec, tmp_path)
    assert receipt["passed"] and receipt["normal_exit"]
    assert receipt["actual_exit"] == receipt["expected_exit"] == 0
    assert receipt["log_sha256"] == sha256_file(Path(receipt["log_path"]))
    assert receipt["duration_s"] >= 0


@pytest.mark.parametrize("field", ["rng_state", "reserved", "issued"])
def test_complete_state_identity(panel, field):
    """REQ-VERIFY-8144: final state includes randomness and reserved geometry."""
    _, rows, _, value = panel
    state = json.loads(Path(value["state_manifest"][0]["state"]["path"]).read_text())
    if field == "issued":
        state[field].append(deepcopy(state[field][-1]))
    else:
        state[field] = []
    labels = Path(value["input_manifests"]["evaluator_label_manifests"]["stream"]["path"])
    with pytest.raises(ValueError):
        a.reconstruct(state, rows, labels)


def test_owned_equation_failure_is_terminal(panel, tmp_path, monkeypatch):
    """REQ-REPORT-8144: authenticated owned failure disqualifies without retry."""

    def fail(*args, **kwargs):
        raise ValueError("independent_equation_failure")

    monkeypatch.setattr(a, "reconstruct", fail)
    work = a.measure(panel[0], tmp_path / "raw", fixture=True)
    value = a.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["verdict_class"] == "disqualified"
    assert value["honest_verdict"] == "complete_disqualified_owned_validation"
    assert value["learning_audit_ready_score"] == 0
    assert "independent_equation_failure" in value["owned_reconstruction_error"]
    path = tmp_path / (a.NAME + ".json")
    atomic_json(path, value)
    assert a.replay(path)


@pytest.mark.parametrize("mutation", ["missing_state", "changed_label", "malicious_order"])
def test_private_cli_custody_and_order_fixtures(panel, tmp_path, mutation):
    """SCENARIO-VERIFY-8144-1: terminal CLI fixtures exercise primitive attacks."""
    value = deepcopy(panel[3])
    if mutation == "missing_state":
        value["state_manifest"][0]["state"]["path"] = str(tmp_path / "missing.json")
    elif mutation == "changed_label":
        ref = value["input_manifests"]["evaluator_label_manifests"]["retention"]
        label = json.loads(Path(ref["path"]).read_text())
        label["rows"][0]["y"] = 1 - label["rows"][0]["y"]
        path = tmp_path / "changed_label.json"
        atomic_json(path, label)
        ref["path"] = str(path)
    else:
        state = json.loads(Path(value["state_manifest"][0]["state"]["path"]).read_text())
        state["events"][:2] = reversed(state["events"][:2])
        path = tmp_path / "malicious_state.json"
        atomic_json(path, state)
        value["state_manifest"][0]["state"] = a.reference(path)
    root = tmp_path / "root"
    atomic_json(root / a.UPSTREAM, value)
    output = tmp_path / (a.NAME + ".json")
    py, cli = str(a.ROOT / ".venv/bin/python"), str(a.ROOT / a.CLI)
    child([py, cli, "--root", str(root), "--fixture-output", str(output)], tmp_path)
    result = json.loads(output.read_text())
    assert result["learning_audit_ready_score"] == 0
    assert result["verdict_class"] == (
        "disqualified" if mutation == "malicious_order" else "blocked"
    )
    assert "replay_passed" in child([py, cli, "--cold-replay", str(output)], tmp_path)
