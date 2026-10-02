"""REQ-PYBIND-8027: test the real loaded numerical boundary before publication."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_8027_v695_native_update_cost as e


@pytest.fixture(scope="session")
def native():
    """Load the same task-owned extension used for numerical measurements."""
    return e.extension()[0]


@pytest.fixture(autouse=True)
def binding_path(native, monkeypatch):
    """Every private test starts from the durable session binary."""
    monkeypatch.setenv("CARNOT_8027_EXTENSION", native.__file__)


def head():
    """A plain artificial state gives no natural learning credit."""
    return dict(
        arm="conditioned_energy",
        parameters=[0.01] * 110,
        decay_scale=0.7,
        calibration=[0.2, 1.3],
        geometry=dict(
            scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
            logit_center=0.0,
            logit_scale=1.0,
            knots=[0.0] * 4 + [i / 9 for i in range(1, 9)] + [1.0] * 4,
        ),
    )


def test_req_pybind_8027_parity_masks_ties_restart(native):
    """SCENARIO-PYBIND-8027-PARITY: masks, knots, ties and lazy state agree."""
    h = head()
    raw = np.random.default_rng(8027).uniform(-0.2, 1.2, (256, 9))
    raw[:10] = np.arange(10)[:, None] / 9
    x = e.python_design(h, raw)
    n = native.RustNumericalUpdate8027(json.dumps(h))
    assert np.max(np.abs(np.asarray(n.design(raw.tolist())) - x)) < 1e-10
    ys = [None if i % 3 == 0 else i % 2 for i in range(256)]
    expected, _, _ = e.python_batch(h, x, ys)
    observed, _, _ = n.update_batch(x.tolist(), ys)
    assert np.max(np.abs(np.asarray(expected) - observed)) < 1e-10
    restored = native.RustNumericalUpdate8027(n.state_json())
    assert np.max(np.abs(e.coefficients(h) - restored.effective())) < 1e-10
    assert json.loads(n.state_json())["decay_scale"] != 0.7
    assert [e.action(p) for p in [0.1, 0.5, 0.0, 1.0]] == [
        "escalate",
        "escalate",
        "accept",
        "reject",
    ]
    assert e.percentiles([1, 1, 1]) == dict(p50=1.0, p95=1.0)
    for target in (0.1, 0.5):
        for p in (np.nextafter(target, 0.0), target, np.nextafter(target, 1.0)):
            tie = head()
            tie["parameters"] = [0.0] * 110
            tie["calibration"] = [float(np.log(p / (1 - p))), 0.0]
            vector = np.zeros(110)
            expected = e.causal.probability(tie, vector)
            observed = native.RustNumericalUpdate8027(json.dumps(tie)).update_batch(
                [vector.tolist()], [None]
            )[0][0]
            assert abs(expected - observed) <= 1e-10 and e.action(expected) == e.action(observed)


def test_req_pybind_8027_reject_nan_and_invalid(native):
    """REQ-PYBIND-8027: invalid requests leave serialized state unchanged."""
    n = native.RustNumericalUpdate8027(json.dumps(head()))
    before = n.state_json()
    for x, ys in [
        ([[float("nan")] * 110], [1]),
        ([[0.0] * 109], [1]),
        ([[0.0] * 110], [2]),
        ([[0.0] * 110], []),
    ]:
        with pytest.raises(ValueError):
            n.update_batch(x, ys)
        assert n.state_json() == before
    with pytest.raises(ValueError):
        n.design([[float("nan")] * 9])
    bad = copy.deepcopy(head())
    bad["decay_scale"] = 0.0
    with pytest.raises(ValueError):
        native.RustNumericalUpdate8027(json.dumps(bad))


def fixture_data():
    """Small exact updates exercise producer paths without natural-data claims."""
    h = head()
    sources = []
    state = copy.deepcopy(h)
    updates = []
    for i in range(4):
        public = dict(
            family_id=str(i),
            source_bytes=b"The cat is black.".hex(),
            answer_bytes=b"The cat is black.".hex(),
        )
        f = e.features.extract(public)
        row = dict(
            public,
            q=0.2 + i / 10,
            features=f["values"],
            public_eligible=True,
            source_cluster_id=f["source_normalized_hash"],
            slot=i,
        )
        sources.append(row)
        before = e.coefficients(state).tolist()
        e.python_batch(state, e.causal.design(h, row)[None, :], [i % 2])
        updates.append(
            dict(
                arm="uniform",
                seed=101,
                origin_slot=i,
                update_index=i,
                y=i % 2,
                family_id=str(i),
                before_coefficients=before,
                after_coefficients=e.coefficients(state).tolist(),
            )
        )
    return dict(
        head=h,
        sources=sources,
        updates=updates,
        references=[],
        imported=dict(
            acquisition=[dict(source_hash=sources[0]["source_cluster_id"], duration_s=1.0)], load={}
        ),
    )


def test_req_report_8027_measurement_and_replay(tmp_path, native, monkeypatch):
    """REQ-REPORT-8027: natural equations and durable timing reduce cold."""
    data = fixture_data()
    parity = e.parity(data, native, tmp_path)
    assert parity["parity_passed"]
    cost = e.benchmark(data, native, tmp_path)
    assert len(cost["timing_rows"]) == 180
    assert all(r["acquisition_matched"] for r in cost["timing_rows"])
    value = e.base([])
    value.update(
        parity, **cost, raw_directory=str(tmp_path), loaded_extension_receipt=e.extension()[1]
    )
    value["acceptance_gate_results"]["measurement"] = True
    e.atomic_json(tmp_path / "replay_inputs.json", data)
    path = tmp_path / "candidate.json"
    e.atomic_json(path, value)
    assert e.replay(path)["passed"]
    for field, error in [("kernel_speedup", "reduction_drift"), ("parity_rows", "parity_drift")]:
        altered = copy.deepcopy(value)
        altered[field] = {} if field == "kernel_speedup" else []
        e.atomic_json(path, altered)
        with pytest.raises(ValueError, match=error):
            e.replay(path)
    altered = copy.deepcopy(value)
    altered["loaded_extension_receipt"]["sha256"] = "wrong"
    e.atomic_json(path, altered)
    with pytest.raises(ValueError, match="binary_drift"):
        e.replay(path)
    e.atomic_json(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(e, "parity", lambda *args: dict(parity_rows=[]))
        with pytest.raises(ValueError, match="cold_equation_drift"):
            e.replay(path)
    value["checkpoint_references"] = parity["checkpoint_references"]
    e.atomic_json(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(e, "coefficients", lambda h: np.ones(110) * 200)
        patch.setattr(e, "parity", lambda *args: parity)
        with pytest.raises(ValueError, match="restart_drift"):
            e.replay(path)
    unsafe = e.base([])
    unsafe["native_update_ready_score"] = 1
    e.atomic_json(path, unsafe)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(path)


def test_req_report_8027_prerequisite_custody(tmp_path):
    """SCENARIO-REPORT-8027-CUSTODY: missing external inputs are terminal blocked."""
    assert len(e.load_inputs(tmp_path, tmp_path / "raw")[1]) == 9
    data = fixture_data()
    for stem, field in [
        ("experiment_8020_v695_qualified_energy_fit", "energy_fit_ready_score"),
        (e.upstream.NAME, "learning_measurement_ready_score"),
        ("experiment_8002_v693_service_cost", "service_measurement_ready_score"),
    ]:
        value = dict(task_id=stem, verdict_class="null", flagged_adversarial=False, **{field: 1})
        if stem == e.upstream.NAME:
            value["checkpoint_references"] = []
        if stem.startswith("experiment_8002"):
            value.update(acquisition_receipt_joins=[], load_amortization={})
        e.atomic_json(tmp_path / "results" / (stem + ".json"), value)
    assert e.load_inputs(tmp_path, tmp_path / "raw")[1][0]["artifact_field"] == "trajectory_files"
    trajectory = tmp_path / "results/raw" / e.upstream.NAME
    e.atomic_json(trajectory / "inputs.json", dict(head=data["head"], sources=data["sources"]))
    db = e.sqlite3.connect(trajectory / "ledger.sqlite")
    db.execute("CREATE TABLE events(seq INTEGER PRIMARY KEY,kind TEXT,payload TEXT)")
    for r in data["updates"]:
        db.execute("INSERT INTO events(kind,payload) VALUES(?,?)", ("update", json.dumps(r)))
    db.commit()
    db.close()
    primary = tmp_path / "results" / (e.upstream.NAME + ".json")
    value = json.loads(primary.read_text())
    value["raw_shard_hashes"] = [e.reference(trajectory / "inputs.json")]
    value["checkpoint_references"] = [e.reference(trajectory / "inputs.json")]
    e.atomic_json(primary, value)
    loaded, failed = e.load_inputs(tmp_path, tmp_path / "raw")
    assert not failed and len(loaded["updates"]) == 4
    capture = tmp_path / "capture.json"
    e.atomic_json(
        capture,
        dict(
            execution_date="20261001",
            rows=[
                dict(
                    role="stream",
                    status="generated",
                    duration_s=1.0,
                    source_cluster_id="source",
                    public_hash="public",
                    raw_response={},
                )
            ],
        ),
    )
    service = tmp_path / "results/experiment_8002_v693_service_cost.json"
    value = json.loads(service.read_text())
    value["cited_upstream_artifacts"] = [
        dict(producer_id=7995, original_reference=e.reference(capture))
    ]
    e.atomic_json(service, value)
    loaded, failed = e.load_inputs(tmp_path, tmp_path / "raw")
    assert not failed and loaded["imported"]["acquisition"][0]["producer_date"] == "20261001"


def test_req_report_8027_cli_and_validation(tmp_path, native, monkeypatch):
    """SCENARIO-REPORT-8027-CUSTODY: direct CLI checks all owned branch outcomes."""
    import runpy
    import sys

    commands = e.validation_plan(tmp_path)
    assert any(c.name == "repository_health" for c in commands)
    assert all("research_conductor.py" not in c.argv for c in commands)
    data = fixture_data()
    fixture = tmp_path / "fixture.json"
    e.atomic_json(fixture, data)
    # Private unit fixtures use short budgets; current measurement keeps its full freeze.
    monkeypatch.setitem(e.CONFIG, "batches", [1])
    monkeypatch.setitem(e.CONFIG, "repetitions", 1)
    monkeypatch.setattr(e, "terminal", lambda p: dict(passed=True))
    monkeypatch.setattr(
        e, "native_coverage", lambda *a: ([], {p: dict(count=1, covered=1) for p in e.RUST[:2]})
    )

    def plan(scratch):
        e.atomic_json(
            scratch / "coverage.json",
            dict(files={p: dict(summary=dict(num_statements=1, missing_lines=0)) for p in e.OWNED}),
        )
        return [e.CommandSpec("private_fixture", ("true",), "owned", 1)]

    monkeypatch.setattr(e, "validation_plan", plan)
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(scope="owned", passed=True)])
    output = tmp_path / "good" / (e.NAME + ".json")
    health_log = tmp_path / "private-health.log"
    health_log.write_text("private bounded diagnostic\n")
    e.atomic_json(
        output.parent / "raw" / output.stem / "repository_health_once.json",
        dict(
            scope="repository_health",
            name="repository_health",
            passed=False,
            log_path=str(health_log),
            log_sha256=e.sha256_file(health_log),
        ),
    )
    assert e.main(["--fixture-input", str(fixture), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert (
        value["native_update_ready_score"] == 1 and value["generalized_learning_benefit_score"] == 0
    )
    assert e.main(["--cold-replay", str(output)]) == 0
    assert value["repository_health"][0]["current_task_diagnostic_reused"] is True
    monkeypatch.setattr(
        e,
        "run_commands",
        lambda *a, **k: [
            dict(scope="owned", passed=False),
            dict(
                scope="repository_health",
                passed=False,
                log_path=str(health_log),
                log_sha256=e.sha256_file(health_log),
            ),
        ],
    )
    failed = tmp_path / "failed" / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(fixture), "--output", str(failed)]) == 0
    assert json.loads(failed.read_text())["verdict_class"] == "disqualified"
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            e.OWNED[-1],
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(blocked),
            "--validation-worker",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(e.ROOT / e.OWNED[-1]), run_name="__main__")
    assert exc.value.code == 0 and json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert e.terminal_commands(blocked)["passed"] is False


def test_req_pybind_8027_build_and_failure_paths(tmp_path, native, monkeypatch):
    """REQ-PYBIND-8027: build, conversion and restart failures retain zero claim."""
    from types import SimpleNamespace

    module, receipt = e.extension()
    monkeypatch.delenv("CARNOT_8027_EXTENSION", raising=False)
    monkeypatch.setattr(e, "ROOT", tmp_path)
    binary = tmp_path / Path(receipt["path"]).name
    binary.write_bytes(Path(receipt["path"]).read_bytes())
    monkeypatch.setattr(e.build, "build_native_extension", lambda root: (binary, {}))
    assert e.extension()[1]["actual_loaded"]
    with monkeypatch.context() as patch:
        patch.setattr(e.build, "load_native_extension", lambda p: object())
        with pytest.raises(ValueError, match="missing_numerical_entrypoint"):
            e.extension()
    data = fixture_data()
    monkeypatch.setitem(e.CONFIG, "batches", [1])
    monkeypatch.setitem(e.CONFIG, "repetitions", 1)
    wrong = SimpleNamespace(
        numerical_echo_8027=lambda rows: [], RustNumericalUpdate8027=module.RustNumericalUpdate8027
    )
    with pytest.raises(ValueError, match="ffi_conversion"):
        e.benchmark(data, wrong, tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(e.features, "extract", lambda public: dict(values=[200.0] * 8))
        with pytest.raises(ValueError, match="public_feature_drift"):
            e.benchmark(data, module, tmp_path)
    with monkeypatch.context() as patch:
        original = e.python_batch

        def incorrect(h, x, y):
            result = original(h, x, y)
            h["parameters"][0] += 2
            return result

        patch.setattr(e, "python_batch", incorrect)
        with pytest.raises(ValueError, match="benchmark_parity"):
            e.benchmark(data, module, tmp_path)
    data["imported"]["acquisition"] = []
    assert not e.benchmark(data, module, tmp_path)["timing_rows"][0]["acquisition_matched"]

    class Proxy:
        def __init__(self, encoded):
            self.inner = module.RustNumericalUpdate8027(encoded)
            self.changed = json.loads(encoded)["decay_scale"] != head()["decay_scale"]

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def effective(self):
            values = self.inner.effective()
            if self.changed:
                values[0] += 1
            return values

    broken = SimpleNamespace(RustNumericalUpdate8027=Proxy)
    with pytest.raises(ValueError, match="restart_coefficients"):
        e.parity(data, broken, tmp_path)


def test_req_pybind_8027_negative_and_terminal(tmp_path, native, monkeypatch):
    """REQ-PYBIND-8027: stable negative logits and the terminal adapter run directly."""
    h = head()
    h["parameters"][0] = -100.0
    n = native.RustNumericalUpdate8027(json.dumps(h))
    x = e.python_design(h, np.full((1, 9), 0.5))
    assert 0 < n.update_batch(x.tolist(), [None])[0][0] < 1e-35
    for encoded in ("[]", "{}"):
        with pytest.raises(ValueError):
            native.RustNumericalUpdate8027(encoded)
    path = tmp_path / "candidate.json"
    e.atomic_json(path, {})
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=False)])
    assert e.terminal(path)["passed"] is False


def test_req_pybind_8027_natural_serialized_restart(tmp_path, native):
    """SCENARIO-PYBIND-8027-PARITY: natural JSON differences stay within the frozen float64 gate."""
    data = json.loads((e.ROOT / "results/raw" / e.upstream.NAME / "inputs.json").read_text())
    db = e.sqlite3.connect(
        "file:" + str(e.ROOT / "results/raw" / e.upstream.NAME / "ledger.sqlite") + "?mode=ro",
        uri=True,
    )
    updates = [
        json.loads(r[0])
        for r in db.execute("SELECT payload FROM events WHERE kind='update' ORDER BY seq")
    ]
    db.close()
    data["updates"] = [r for r in updates if r["arm"] == "uniform" and r["seed"] == 101]
    result = e.parity(data, native, tmp_path)
    assert result["parity_passed"]


def test_req_report_8027_native_coverage_plan(tmp_path, monkeypatch):
    """REQ-REPORT-8027: Rust coverage failures cannot qualify numerical readiness."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    raw = tmp_path / "raw"
    target = scratch / "native-target/debug/deps"
    target.mkdir(parents=True)
    for stem in ("carnot_core", "carnot_python"):
        (target / (stem + "-abc")).write_text("fixture")
    (scratch / "native-1.profraw").write_text("fixture")
    log = tmp_path / "export.log"
    exported = dict(
        data=[
            dict(
                files=[
                    dict(filename=str(e.ROOT / p), summary=dict(lines=dict(count=1, covered=1)))
                    for p in e.RUST[:2]
                ]
            )
        ]
    )
    log.write_text(json.dumps(exported))
    show = tmp_path / "show.log"
    show.write_text("12|0|#[pymethods]\n")

    def passing(*args, **kwargs):
        return [
            dict(passed=True, log_path=str(show if c.name.startswith("rust_source_lines") else log))
            for c in args[1]
        ]

    monkeypatch.setattr(e, "run_commands", passing)
    assert len(e.native_coverage(scratch, raw)[1]) == 2
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=False)])
    assert not e.native_coverage(scratch, raw)[1]
    calls = iter([[dict(passed=True)], [dict(passed=False)]])
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: next(calls))
    assert not e.native_coverage(scratch, raw)[1]
