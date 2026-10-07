"""REQ-VERIFY-8225 / REQ-REPORT-8225: causal fixtures and actual private CLI."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import delayed_utility_learning_8225 as k
from carnot.verify import delayed_utility_execution_8225 as e


def cli(tmp_path, *args):
    """Cross the real script boundary so missing imports and exit handling are tested."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private8225 CLI", flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300
    )
    print("after private8225 CLI", result.returncode, flush=True)
    return result


def test_five_arms_and_missing_causal_windows(tmp_path):
    """SCENARIO-VERIFY-8225-CAUSAL: every arm uses only released update roles."""
    rows, path, retained = k.fixture(tmp_path)
    final = k.run(rows, path, 101, tmp_path / "run")
    assert list(final) == k.ARMS and len(retained) == 64
    for arm, state in final.items():
        assert len(state["issued"]) == 256 and len(state["pending"]) == 20
        assert set(state["training"]).isdisjoint(state["used_admission"])
        assert state["missing_mask"] == [r["p"] is None for r in rows]
        assert state["global_parameters"] and state["rng_state"]
        commits = [v for v in state["events"] if v["kind"] == "commit_candidate"]
        assert [v["slot"] for v in commits] == [64, 144]
        for event in commits:
            assert all(i + 20 <= event["slot"] for i in event["fit_ids"])
            assert all(k.kernel.roles.bucket(rows[i - 1]) for i in event["fit_ids"])
        for event in state["events"]:
            if event["kind"] == "admit_once":
                assert len(event["labels"]) == 12
                assert all(i + 20 <= event["slot"] for i in event["labels"])
                assert event["step"] in [0, 1, 0.5, 0.25, 0.125]
        if arm == "frozen":
            assert state["model"] == {"kind": "input"}
        assert [r["issued"] for r in k.journal(tmp_path / "run" / arm / "issued.jsonl")] == state[
            "issued"
        ]
    released = k.journal(tmp_path / "run/global_only/released.jsonl")
    assert len(released) == 236 and all(r["release_slot"] == r["label_slot"] + 20 for r in released)
    with pytest.raises(ValueError, match="rollback"):
        k.run(rows, path, 101, tmp_path / "run")
    labels = k.Labels(path, rows, tmp_path / "unsealed.jsonl")
    with pytest.raises(ValueError, match="unsealed_release"):
        labels[0]
    rows[0]["p"] = None
    with pytest.raises(ValueError, match="baseline_hash"):
        k.run(rows, path, 101, tmp_path / "changed", state=final)


def test_global_optimizer_and_persistent_tree(tmp_path, monkeypatch):
    """REQ-VERIFY-8225: numerical fitting reuses the qualified bounded optimizer."""
    rows, _, _ = k.fixture(tmp_path)
    pool = [dict(r, y=r["slot"] % 2) for r in rows[:64]]
    for arm in k.ARMS:
        model = k.propose(pool, {"kind": "input"}, arm, 101)
        assert len(model["patches"]) <= 4
        if arm == "global_only":
            assert all(p["group"]["name"] == "global" for p in model["patches"])
        if arm == "local_only":
            assert all(p["group"]["name"] != "global" for p in model["patches"])
        p = k.predict(model, rows[0])
        assert p is None or 1e-6 <= p <= 1 - 1e-6
    candidate = k.propose(pool, {"kind": "input"}, "global_plus_group", 101)
    mixture = dict(kind="mixture", base=dict(kind="input"), candidate=candidate, step=0.5)
    assert k.predict(mixture, rows[1]) == pytest.approx(
        (rows[1]["p"] + k.predict(candidate, rows[1])) / 2
    )
    next_model = k.propose(pool, mixture, "global_plus_group", 101)
    assert next_model["base"]["kind"] == "mixture"
    assert k.predict(next_model, dict(rows[1], p=None)) is None
    monkeypatch.setattr(k.calibration, "train", lambda *a, **kw: dict(fit=dict(converged=False)))
    assert not k.propose(pool, mixture, "global_only", 101)["patches"]
    assert not k.propose([dict(r, p=None) for r in pool], mixture, "global_only", 101)["patches"]


def test_cli_crashes_recovery_and_replay(tmp_path):
    """SCENARIO-REPORT-8225-CLI: genuine exits and exact resumed state are required."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["utility_trajectory_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert value["generalized_learning_benefit_score"] == 0
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert [r["actual_exit"] for r in value["child_exit_rows"]] == [0, 73, 0, 73, 0]
    assert all(r["passed"] for r in value["restart_state_hashes"])
    assert value["retention_labels_opened"] is False
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    atomic_json(output, dict(value, completed_count=999))
    assert not e.replay(output)
    changed = dict(value, utility_trajectory_ready_score=9)
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    assert not e.replay(tmp_path / "missing.json")
    ref = value["raw_shard_hashes"][0]
    path = Path(ref["path"])
    saved = path.read_bytes()
    path.write_bytes(saved + b" ")
    assert not e.replay(output)
    path.write_bytes(saved)
    assert e.replay(output)


def test_external_block_and_private_cli_failures(tmp_path):
    """SCENARIO-REPORT-8225-CLI: missing differs from zero and fixtures remain private."""
    output = tmp_path / (e.NAME + ".json")
    assert cli(tmp_path, "--root", tmp_path / "absent", "--fixture-output", output).returncode == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["utility_trajectory_ready_score"] == 0
    assert any(c["observed"] is None for c in value["gate_check_summary"])
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert cli(tmp_path, "--date", "19000101").returncode == 2
    assert (
        cli(tmp_path, "--fixture-output", e.ROOT / "results" / "never-write.json").returncode == 2
    )
    worker = tmp_path / "worker/measurement.json"
    assert cli(tmp_path, "--root", tmp_path / "absent", "--worker-output", worker).returncode == 0
    work = json.loads(worker.read_text())
    failed = e.build(work, worker.parent, [dict(passed=False)])
    assert (
        failed["verdict_class"] == "disqualified" and failed["utility_trajectory_ready_score"] == 0
    )


def test_manifest_freezes_owned_scope(tmp_path):
    """REQ-REPORT-8225: coverage includes real CLI and crash child statements."""
    specs = e.manifest(tmp_path, tmp_path / (e.NAME + ".json"))
    commands = {r["name"]: r for r in specs["commands"]}
    assert "--fail-under=100" in commands["coverage_report"]["argv"]
    assert "--strict" in commands["strict_mypy"]["argv"]
    assert "--files" in commands["spec_coverage"]["argv"]
    assert "patch = _exit" in (tmp_path / "coverage.ini").read_text()
    assert specs["repository_health"]["classification"] == "diagnostic"


def test_preparation_masks_costs_and_deadlines(tmp_path, monkeypatch):
    """REQ-VERIFY-8225: original slot masks and measured costs remain explicit."""
    rows, path, retained = k.fixture(tmp_path)
    original = [dict(r, values=None if r["p"] is None else [0.0] * 9) for r in rows]
    monkeypatch.setattr(k.frozen, "baseline_probability", lambda row, baseline: 0.08)
    prepared = k.prepare(original, {})
    assert prepared[0]["p"] == 0.5 and prepared[0]["baseline_action"] == "accept"
    assert prepared[4]["p"] is None and prepared[4]["baseline_action"] == "escalate"
    final = k.run(rows, path, 101, tmp_path / "final")
    evidence = e.evidence([final], rows, retained, tmp_path / "final")
    assert len(evidence["learning_rows"]) == 256 * 5
    assert len(evidence["retention_predictions"]) == 64 * 5
    assert evidence["memory_bytes"] > 0 and evidence["update_timing_rows"]
    units = [r for r in evidence["rows"] if r["arm"] == "frozen" and r["metric"] == "brier"]
    assert len(units) == 192 and any(r["status"] == "censored" for r in units)
    assert any(r["status"] == "excluded" for r in units)
    monkeypatch.setattr(k, "BUDGET_S", -1)
    with pytest.raises(TimeoutError, match="cpu_science_budget"):
        k.run(rows, path, 101, tmp_path / "deadline")


def test_owned_failure_and_rehashed_state(tmp_path, monkeypatch):
    """REQ-REPORT-8225: owned exceptions and rehashed primitives cannot pass."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, raw, [dict(passed=True)], fixture=True)
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(work)
    changed["learning_rows"][0]["action"] = "invented"
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)], fixture=True))
    assert not e.replay(output)
    monkeypatch.setattr(
        k, "run", lambda *a, **kw: (_ for _ in ()).throw(ValueError("owned failure"))
    )
    failed = e.measure(e.ROOT, tmp_path / "failure", fixture=True)
    assert (
        e.build(failed, tmp_path / "failure", [dict(passed=True)])["verdict_class"]
        == "disqualified"
    )


def test_natural_primitives_and_rehashed_receipt(tmp_path):
    """REQ-REPORT-8225: bounded real children replay natural primitives without retaining RSS."""
    raw = tmp_path / "natural"
    result = cli(tmp_path, "--worker-output", raw / "measurement.json")
    assert result.returncode == 0, result.stdout + result.stderr
    code = "\n".join(
        [
            "import json, sys",
            "from pathlib import Path",
            "sys.path[:0] = " + repr([str(e.ROOT / "python"), str(e.ROOT)]),
            "from carnot.verify import delayed_utility_execution_8225 as e",
            "from carnot.reporting.current_work_receipt import atomic_json",
            "raw = Path(sys.argv[1]); work = json.loads((raw / 'measurement.json').read_bytes())",
            "assert work['input_ready'] == 1 and not work['owned_failure']",
            "assert len(work['states']) == 20 and all(r['passed'] for r in work['restart_state_hashes'])",
            "assert any(r['kind'] == 'issue_lookup' for r in work['update_timing_rows'])",
            "receipt = e.run_check(e.ROOT, dict(name='pass', argv=['/bin/true'], expected_exit=0, deadline_s=5), raw, raw / 'owned_logs')",
            "value = e.build(work, raw, [receipt]); output = raw.parent / (e.NAME + '.json')",
            "atomic_json(output, value); assert value['utility_trajectory_ready_score'] == 1",
            "assert e.replay(output)",
            "Path(receipt['stdout_path']).write_bytes(b'altered'); assert not e.replay(output)",
            "print('natural primitives,20 seeds,exact recovery and receipt tamper passed', flush=True)",
        ]
    )
    receipt = e.run_check(
        e.ROOT,
        dict(
            name="natural_primitive_replay",
            argv=[str(e.ROOT / ".venv/bin/python"), "-u", "-c", code, str(raw)],
            deadline_s=240,
            expected_exit=0,
        ),
        tmp_path,
        tmp_path / "natural_logs",
        heartbeat_s=20,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()


def test_rng_draws_and_optimizer_only_candidate(tmp_path):
    """REQ-VERIFY-8225: saved random state records actual witness draws."""
    rows, _, _ = k.fixture(tmp_path)
    pool = [
        dict(r, p=0.5, baseline_p=0.2 if i < 16 else 0.8, baseline_action="reject", y=int(i < 16))
        for i, r in enumerate(rows[:32])
    ]
    model = k.propose(pool, dict(kind="input"), "global_plus_random", 102)
    assert model["fit_rng_state"] != k.kernel.json_rng(102)


def test_global_fit_can_admit_without_residual_patches(tmp_path):
    """REQ-VERIFY-8225: the equally adaptive global control retains useful calibration."""
    rows, path, _ = k.fixture(tmp_path)
    pool = [dict(r, y=r["slot"] % 2) for r in rows[:64]]
    proposed = k.propose(pool, dict(kind="input"), "global_only", 101)
    assert proposed["global_fit_changed"]
    final = k.run(rows, path, 101, tmp_path / "run")
    commits = [r for r in final["global_only"]["events"] if r["kind"] == "commit_candidate"]
    assert commits[0]["candidate"] is not None


def test_admission_selection_preserves_missing_original_rows(tmp_path):
    """SCENARIO-VERIFY-8225-CAUSAL: selection cannot replace a missing admission unit."""
    rows, path, _ = k.fixture(tmp_path)
    final = k.run(rows, path, 101, tmp_path / "run")
    for state in final.values():
        events = [r for r in state["events"] if r["kind"] == "admit_once"]
        if events:
            assert events[0]["labels"] == list(range(65, 77))
    records = [dict(r, y=None) for r in rows[64:76]]
    with e.patch.object(k.kernel, "predict", k.predict):
        unchanged, detail = k.kernel.admission(dict(kind="input"), dict(kind="input"), records)
    assert unchanged == dict(kind="input") and detail["step"] == 0


def test_stream_permission_uses_its_original_qwen_offset(tmp_path):
    """REQ-VERIFY-8225: stream membership does not borrow static fitted features."""
    rows, _, _ = k.fixture(tmp_path)
    public = [dict(r, values=None if r["p"] is None else [0.0] * 9) for r in rows]
    prepared = k.prepare(public)
    assert prepared[0]["p"] == prepared[0]["baseline_p"] == 0.5
    assert prepared[0]["baseline_action"] == "escalate"
    assert prepared[4]["p"] is None and prepared[4]["baseline_action"] == "escalate"


def test_schema_and_total_budget_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8225-CLI: malformed external schema blocks without an owned failure."""
    root = tmp_path / "root"
    for name in [e.qualified.PROTOCOL, k.calibration.PROTOCOL]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((e.ROOT / name).read_bytes())
    path = root / "results" / (e.qualified.NAME + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]")
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert (
        value["verdict_class"] == "blocked"
        and value["gate_check_summary"][-1]["check"] == "input_schema"
    )
    monkeypatch.setattr(k, "BUDGET_S", -1)
    work = e.measure(e.ROOT, tmp_path / "deadline", fixture=True)
    assert work["owned_failure"] == "cpu_science_budget"
    assert (
        e.build(work, tmp_path / "deadline", [dict(passed=True)], fixture=True)["verdict_class"]
        == "disqualified"
    )


def test_rehashed_complete_state_and_final_seal(tmp_path):
    """REQ-REPORT-8225: state and seal edits fail primitive replay after rehashing."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    output = tmp_path / (e.NAME + ".json")
    changed = deepcopy(work)
    changed["states"][0]["frozen"]["cursor"] = 0
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)], fixture=True))
    assert not e.replay(output)
    final = Path(work["final_states_path"])
    saved = final.read_bytes()
    altered = json.loads(saved)
    altered[0]["frozen"]["cursor"] = 0
    atomic_json(final, altered)
    changed = deepcopy(work)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(final):
            ref["sha256"] = e.reference(final)["sha256"]
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)], fixture=True))
    assert not e.replay(output)
    final.write_bytes(saved)


def test_rehashed_trajectory_timing_and_live_operand(tmp_path):
    """REQ-REPORT-8225: archived predictions, costs and live bytes bind cold replay."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    output = tmp_path / (e.NAME + ".json")
    trajectory = Path(work["trajectory_path"])
    original = trajectory.read_bytes()
    changed = deepcopy(work)
    altered = json.loads(original)
    altered["learning_rows"][0]["action"] = "invented"
    atomic_json(trajectory, altered)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(trajectory):
            ref["sha256"] = e.reference(trajectory)["sha256"]
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)], fixture=True))
    assert not e.replay(output)
    trajectory.write_bytes(original)
    changed = deepcopy(work)
    changed["update_timing_rows"][0]["cpu_ns"] += 1
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)], fixture=True))
    assert not e.replay(output)
    source = raw / "private_fixture/fixture-labels.json"
    saved = source.read_bytes()
    snapshot = tmp_path / "label-snapshot.bin"
    snapshot.write_bytes(saved)
    source.write_bytes(saved + b" ")
    changed = deepcopy(work)
    changed["refs"].append(dict(e.reference(snapshot), upstream_path=str(source)))
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(source):
            ref["sha256"] = e.reference(source)["sha256"]
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)], fixture=True))
    assert not e.replay(output)
    source.write_bytes(saved)
