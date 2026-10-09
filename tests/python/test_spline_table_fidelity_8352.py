"""REQ-VERIFY-8352 / REQ-REPORT-8352: constructed numerical engineering controls."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from carnot.verify import spline_table_fidelity_8352 as n
from carnot.reporting import spline_table_fidelity_8352 as e
from carnot.reporting import spline_table_execution_8352 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def authentic(tmp_path_factory):
    """Only authenticated original bytes supply the scientific coefficients."""
    raw = tmp_path_factory.mktemp("table8352-inputs")
    return e.authenticate(e.ROOT, raw)


def test_panel_reference_and_boundaries(authentic):
    """SCENARIO-VERIFY-8352-PANEL: construct controls without any evaluator inputs."""
    head = authentic["head"]
    x, kinds, available = n.panel(head)
    assert x.shape == (4174, 5)
    assert kinds.count("random") == 4096
    assert available
    assert np.max(np.abs(n.direct(head, x) - n.reference(head, x))) <= 1e-10
    assert np.all((x[:, 1:] >= 0) & (x[:, 1:] <= 1))
    assert np.allclose(n.reference(head, x[-6:])[1::3], [0.25, 0.75], atol=1e-14)
    zero = dict(head, coefficients=[0, *head["coefficients"][1:]])
    assert n.panel(zero)[2] is False
    assert len(n.panel(zero)[0]) == 4168


def test_encoding_and_configuration_gates(authentic):
    """REQ-VERIFY-8352: test signed range, even ties and readiness/candidate split."""
    values = np.array([-0.5, 0.5, 1.5, 2.5, -32769, 32768]) / 2048
    codes, saturation = n.encode(values, "int16")
    assert codes.tolist() == [0, 0, 2, 2, -32768, 32767]
    assert saturation == 2
    assert n.encode(values, "float64")[1] == 0
    head = authentic["head"]
    x = n.panel(head)[0]
    ref = n.reference(head, x)
    configs = []
    for i, (size, storage, interpolation) in enumerate(n.CONFIGS):
        table, sat = n.table(head, size, storage)
        actual = n.lookup(head, x, table, storage, interpolation)
        row = n.summarize(ref, actual, size, storage, interpolation, i, sat)
        assert row["table_bytes"] == size * 4 * (8 if storage == "float64" else 2)
        assert row["action_flip_count"] == row["boundary_flip_count"] + row["far_flip_count"]
        configs.append(row)
    chosen = n.select(configs)
    assert chosen["candidate_passed"]
    assert chosen["table_bytes"] == min(r["table_bytes"] for r in configs if r["candidate_passed"])
    assert n.select([dict(r, candidate_passed=False) for r in configs]) is None
    ref = np.array([0.1, 0.249999, 0.5, 0.9])
    row = n.summarize(ref, np.array([0.3, 0.250001, 0.5, 0.9]), 65, "float64", "nearest", 0, 0)
    assert row["far_flip_count"] == row["boundary_flip_count"] == 1
    assert not row["candidate_passed"]


def test_updates_scoped_refresh_and_missed_entry(authentic):
    """SCENARIO-VERIFY-8352-REFRESH: scoped bytes match full bytes after every update."""
    original = authentic["head"]
    for size, storage in [(65, "float64"), (257, "int16")]:
        head = deepcopy(original)
        table, _ = n.table(head, size, storage)
        for index, x in enumerate(n.events()):
            new = n.update(head, x, index % 2)
            scoped, entries = n.refresh(head, new, table, storage)
            full, _ = n.table(new, size, storage)
            assert scoped.tobytes() == full.tobytes()
            assert new["coefficients"][:2] == original["coefficients"][:2]
            assert new["temperature"] == original["temperature"]
            changed = np.asarray(head["coefficients"])[2:] != np.asarray(new["coefficients"])[2:]
            basis = n.basis(np.linspace(0, 1, size))
            for feature, ids in enumerate(entries):
                expected = np.flatnonzero(
                    np.any(basis[:, changed.reshape(4, 8)[feature]] != 0, axis=1)
                )
                assert ids == expected.tolist()
            head, table = new, scoped
    old = original
    new = n.update(old, n.events()[0], 0)
    table, _ = n.table(old, 65, "float64")
    broken, _ = n.refresh(old, new, table, "float64", miss=True)
    assert broken.tobytes() != n.table(new, 65, "float64")[0].tobytes()
    large = dict(original, coefficients=[*original["coefficients"][:2], *([4.0] * 32)])
    changed = n.update(large, np.array([0.0, 0.5, 0.5, 0.5, 0.5]), 0)
    assert max(changed["coefficients"][2:]) <= 4


def test_missing_authentication_and_real_child(tmp_path):
    """SCENARIO-REPORT-8352-TERMINAL: a missing external operand is distinct from zero."""
    with pytest.raises(e.OperandError) as error:
        e.authenticate(tmp_path, tmp_path / "inputs")
    assert error.value.finding["observed"] is None
    assert error.value.finding["expected"]
    receipt = child("real", [sys.executable, "-u", "-c", "print('real child')"], tmp_path)
    assert receipt["passed"] and receipt["stdout_sha256"]


def test_independent_update_oracle_and_nearest_ties(authentic):
    """REQ-VERIFY-8352: shared refresh agreement cannot excuse a wrong gradient."""
    for temperature in [0.5, 2.0]:
        head = dict(authentic["head"], temperature=temperature)
        x = np.array([100.0, 0.37, 0.42, 0.91, 0.0])
        phi = np.array(n.kernel.scalar_design(x.tolist())[2:])
        p = n.reference(head, x[None])[0]
        gradient = (p - 0) / temperature * phi
        scale = min(1.0, 1.0 / np.linalg.norm(gradient))
        expected = np.clip(np.array(head["coefficients"])[2:] - 0.01 * scale * gradient, -4, 4)
        assert np.allclose(n.update(head, x, 0)["coefficients"][2:], expected, atol=1e-15, rtol=0)
    head = dict(coefficients=[0.0, 0.0, *([0.0] * 32)], temperature=1.0)
    table = np.tile(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), (4, 1))
    x = np.array([[0.0, *([v] * 4)] for v in [0.0, 0.125, 0.375, 1.0]])
    actual = n.lookup(head, x, table, "float64", "nearest")
    assert np.allclose(actual, 1 / (1 + np.exp(-np.array([0.0, 0.0, 8.0, 16.0]))))
    rows = [
        dict(table_bytes=2, probability_error_max=0.0002, order=2, candidate_passed=True),
        dict(table_bytes=2, probability_error_max=0.0001, order=3, candidate_passed=True),
        dict(table_bytes=2, probability_error_max=0.0001, order=1, candidate_passed=True),
    ]
    assert n.select(rows) == rows[2]


@pytest.fixture(scope="module")
def measured(tmp_path_factory, authentic):
    """Real measurements stay private; test receipts do not authorize the final run."""
    parent = tmp_path_factory.mktemp("table8352-measured")
    raw = parent / "raw" / e.NAME / "invocations" / "test"
    raw.mkdir(parents=True)
    private = parent / "private"
    private.mkdir()
    runner.manifest(private)
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=[
                dict(
                    name="test_control",
                    argv=[sys.executable, "-u", "-c", "print('private test control')"],
                    expected=0,
                )
            ]
        ),
    )
    work = e.measure(authentic, raw)
    work.update(
        failures=[],
        duration_s=1.0,
        preconditions=dict(private_scratch=True),
        manifest_reference=e.reference(raw / "execution_manifest.json"),
        code_refs=[],
        coverage=dict(
            totals=dict(percent_covered=100),
            files={p: dict(summary=dict(missing_lines=0)) for p in e.OWNED},
        ),
    )
    receipt = child(
        "test_control",
        [sys.executable, "-u", "-c", "print('private test control')"],
        raw / "checks",
    )
    return work, raw, parent / (e.NAME + ".json"), [receipt]


def write_candidate(work, raw, output, receipts):
    """Private candidates exercise reduction without publishing repository primaries."""
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    path = raw / "candidate.json"
    atomic_json(path, value)
    return path, value


def test_cold_cli_and_summary_tamper(measured):
    """SCENARIO-REPORT-8352-TERMINAL: actual cold processes reject repaired summaries."""
    work, raw, output, receipts = measured
    path, value = write_candidate(work, raw, output, receipts)
    assert value["table_fidelity_ready_score"] == value["table_candidate_score"] == 1
    assert e.replay(path)
    checks = runner.controls(value, raw / "controls")
    assert all(r["passed"] for r in checks)
    assert all(r["exit_code"] == int(r["name"] != "cold_valid") for r in checks)
    assert (
        e.build(work, [dict(receipts[0], passed=False)], raw, output)["verdict_class"]
        == "disqualified"
    )
    assert (
        e.build(dict(work, failures=[dict(observed=None)]), receipts, raw, output)["verdict_class"]
        == "blocked"
    )
    rejected = deepcopy(work)
    rejected["coverage"]["totals"]["percent_covered"] = 99
    assert e.build(rejected, receipts, raw, output)["table_fidelity_ready_score"] == 0
    no_candidate = deepcopy(work)
    no_candidate["configurations"] = [
        dict(r, candidate_passed=False) for r in work["configurations"]
    ]
    result = e.build(no_candidate, receipts, raw, output)
    assert result["table_fidelity_ready_score"] == 1 and result["table_candidate_score"] == 0
    assert not e.replay(raw / "absent")
    atomic_json(path, {})
    assert not e.replay(path)
    path, value = write_candidate(work, raw, output, receipts)
    atomic_json(path, dict(value, work_reference=dict(value["work_reference"], sha256="bad")))
    assert not e.replay(path)


def test_primitive_reconstruction_mutations(measured, monkeypatch):
    """REQ-REPORT-8352: repaired primitive hashes cannot conceal semantic changes."""
    work, raw, output, receipts = measured
    for mutate in [
        lambda w: w["inputs"]["head"].update(temperature=99),
        lambda w: w["inputs"].update(authority_sha256="wrong"),
        lambda w: w.update(vector_count=1),
        lambda w: w.update(configurations=w["configurations"][:-1]),
        lambda w: w["configurations"][0].update(probability_error_max=99),
        lambda w: w["configurations"][0].update(table_ns_per_vector_median=-1),
    ]:
        changed = deepcopy(work)
        mutate(changed)
        assert not e.replay_numeric(changed, raw)
    path, value = write_candidate(work, raw, output, receipts)
    with monkeypatch.context() as patch:
        patch.setattr(e, "sha256_file", lambda _: "changed")
        assert not e.replay(path)
    bad = deepcopy(value)
    bad["validation_receipts"][0]["stdout_sha256"] = "changed"
    atomic_json(path, bad)
    assert not e.replay(path)
    for filename, mutation in [
        ("panel_metadata.json", lambda d: d.update(boundary_available=False)),
        ("refresh_rows.json", lambda d: d.update(rows=d["rows"][:-1])),
        ("final_tables.json", lambda d: d["rows"][0].update(grid_points=257)),
        ("refresh_rows.json", lambda d: d["rows"][0].update(target=1)),
        ("refresh_rows.json", lambda d: d["rows"][0].update(timings=[])),
        ("final_tables.json", lambda d: d["rows"][0].update(table_hex="bad")),
    ]:
        operand = raw / filename
        original = operand.read_bytes()
        data = json.loads(original)
        mutation(data)
        atomic_json(operand, data)
        try:
            assert not e.replay_numeric(work, raw)
            changed = deepcopy(work)
            changed["primitive_refs"] = [
                e.reference(Path(r["path"])) for r in changed["primitive_refs"]
            ]
            path, _ = write_candidate(changed, raw, output, receipts)
            assert not e.replay(path)
        finally:
            operand.write_bytes(original)
    with monkeypatch.context() as patch:
        patch.setattr(e.kernel, "scalar_design", lambda _: [0.0] * 34)
        assert not e.replay_numeric(work, raw)
    for filename in ["panel.npy", "direct.npy", "reference.npy"]:
        operand = raw / filename
        original = operand.read_bytes()
        array = np.load(operand)
        array.flat[0] += 0.1
        np.save(operand, array)
        try:
            assert not e.replay_numeric(work, raw)
        finally:
            operand.write_bytes(original)
    operand = raw / "per_vector_rows.jsonl"
    original = operand.read_bytes()
    first, rest = original.split(b"\n", 1)
    row = json.loads(first)
    row["table_probability"] += 0.01
    operand.write_bytes(json.dumps(row).encode() + b"\n" + rest)
    assert not e.replay_numeric(work, raw)
    operand.write_bytes(original + b"{}\n")
    assert not e.replay_numeric(work, raw)
    operand.write_bytes(original)
    write_candidate(work, raw, output, receipts)


def test_owned_main_paths_and_failure_publication(measured, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8352-TERMINAL: main owns real children and failure publication."""
    work, original_raw, _, _ = measured
    original_manifest = runner.manifest

    def private_plan(private):
        original_manifest(private)
        atomic_json(private / "coverage.json", work["coverage"])
        return [
            dict(
                name="private_validation",
                argv=[sys.executable, "-u", "-c", "print('owned control')"],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ]

    def reuse_measurement(inputs, raw):
        import shutil

        for p in original_raw.iterdir():
            if p.is_file() and p.name not in [
                "execution_manifest.json",
                "measurement.json",
                "candidate.json",
            ]:
                shutil.copyfile(p, raw / p.name)
        copied = deepcopy(work)
        copied["inputs"] = inputs
        copied["primitive_refs"] = [
            e.reference(raw / Path(r["path"]).name) for r in work["primitive_refs"]
        ]
        for row in copied["configurations"]:
            for key in ["table_reference", "probabilities_reference", "timings_reference"]:
                row[key] = e.reference(raw / Path(row[key]["path"]).name)
        return copied

    monkeypatch.setattr(runner, "manifest", private_plan)
    monkeypatch.setattr(e, "measure", reuse_measurement)
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--date", "20261009", "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["table_fidelity_ready_score"] == 1
    value = json.loads(output.read_bytes())
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert side["normal_process_completion"]
    assert side["attempts"][-1]["passed"]
    with pytest.raises(SystemExit):
        runner.main(["--date", "wrong"])
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with monkeypatch.context() as patch:
        patch.setattr(runner, "terminal_checks", lambda *_: dict(passed=False))
        with pytest.raises(ValueError):
            runner.publish({}, tmp_path / "invalid.json", tmp_path)
    raw = Path(value["work_reference"]["path"]).parent
    checks = side["checks"]
    original_checks = runner.terminal_checks
    calls = []

    def reject_once(candidate, logs):
        calls.append(candidate)
        if len(calls) == 1:
            return dict(
                passed=False, checks=[dict(checks[0], passed=False)], adversarial=dict(findings=[])
            )
        return original_checks(candidate, logs)

    monkeypatch.setattr(runner, "terminal_checks", reject_once)
    runner.publish(value, output, raw)
    assert (raw / "rejected_candidate.json").exists()
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_sidecar_rejection_and_blocked_direct_cli(tmp_path, monkeypatch):
    """REQ-REPORT-8352: byte drift and missing operands follow actual failure paths."""
    with monkeypatch.context() as patch:

        def stale(*_):
            raise ValueError("stale sidecar")

        patch.setattr(e, "read_bound_sidecar", stale)
        with pytest.raises(e.OperandError) as error:
            e.authenticate(e.ROOT, tmp_path / "stale")
        assert error.value.finding["field"] == "bound_terminal_sidecar"
    output = tmp_path / "blocked" / (e.NAME + ".json")
    receipt = child(
        "blocked_cli",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--root",
            str(tmp_path / "missing-root"),
            "--output",
            str(output),
        ],
        tmp_path / "logs",
        deadline=120,
        heartbeat=20,
    )
    assert receipt["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["observed"] is None
    assert value["table_fidelity_ready_score"] == value["table_candidate_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert e.replay(output)


def test_refresh_shared_fault_and_hash_bound_work(measured, monkeypatch):
    """REQ-VERIFY-8352: a shared scoped/full defect must fail independent replay."""
    work, raw, output, receipts = measured
    checkpoint_ref = next(r for r in work["inputs"]["refs"] if r["source_sha256"] == e.CHECKPOINT)
    changed_checkpoint = json.loads(Path(checkpoint_ref["path"]).read_bytes())
    changed_checkpoint["heads"][0]["temperature"] = 99
    tampered = raw / "rehashed_checkpoint.json"
    atomic_json(tampered, changed_checkpoint)
    rebound = deepcopy(work)
    rebound["inputs"]["refs"] = [
        dict(r, **e.reference(tampered)) if r["source_sha256"] == e.CHECKPOINT else r
        for r in rebound["inputs"]["refs"]
    ]
    assert not e.replay_numeric(rebound, raw)
    path, _ = write_candidate(work, raw, output, receipts)
    changed = deepcopy(work)
    changed["inputs"]["head"]["temperature"] += 1
    path, _ = write_candidate(changed, raw, output, receipts)
    assert not e.replay(path)
    write_candidate(work, raw, output, receipts)
    altered = deepcopy(work)
    altered["primitive_refs"][0]["sha256"] = "wrong"
    path, _ = write_candidate(altered, raw, output, receipts)
    assert not e.replay(path)
    write_candidate(work, raw, output, receipts)
    original_refresh = n.refresh

    def broken_refresh(*args, **kwargs):
        values, entries = original_refresh(*args, **kwargs)
        values[0, 0] += 0.1
        return values, entries

    with monkeypatch.context() as patch:
        patch.setattr(n, "refresh", broken_refresh)
        assert not e.replay_numeric(work, raw)
