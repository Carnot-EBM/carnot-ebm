"""REQ-REPORT-8858: qualify explicit records without a research runtime dependency."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify.indexed_witness import verify_indexed_witness
from carnot.verify import _indexed_witness_core as core
from carnot.verify import _indexed_witness_meter as meter

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = (
    ROOT
    / "results/raw/experiment_8858_v758_installed_indexed_verifier/authorities/semantic_fixtures.json"
)


def stable(row: dict[str, Any]) -> dict[str, Any]:
    """Elapsed cost is measured separately and cannot decide semantic parity."""
    return {k: v for k, v in row.items() if k not in {"wall_time_s", "cpu_time_s"}}


def test_complete_original_roster():
    """SCENARIO-REPORT-8858-CONTRACT: preserve every original decision and paid operation."""
    sealed = json.loads(FIXTURES.read_bytes())
    assert sealed["original_count"] == 3600
    for unit in sealed["rows"]:
        assert stable(verify_indexed_witness(unit["input"])) == unit["expected"], unit["unit_id"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        [],
        {"facts": [], "query": []},
        {"facts": (), "query": [["a", "r", "b"]]},
        {"facts": [], "query": "a"},
        {"facts": [[]], "query": [["a", "r", "b"]]},
        {"facts": [("a", "r", "b", True)], "query": [["a", "r", "b"]]},
        {"facts": [["a", "r", 1, True]], "query": [["a", "r", "b"]]},
        {"facts": [["a", "r", "", True]], "query": [["a", "r", "b"]]},
        {"facts": [[" a", "r", "b", True]], "query": [["a", "r", "b"]]},
        {"facts": [["a", "r", "b", 1]], "query": [["a", "r", "b"]]},
        {"facts": [["?x", "r", "b", True]], "query": [["a", "r", "b"]]},
        {"facts": [], "query": [["a", "?r", "b"]]},
        {"facts": [], "query": [["a", "r", "b"]], "label": "support"},
    ],
)
def test_full_schema_before_budget(payload):
    """REQ-REPORT-8858: even an empty budget cannot bypass input validation."""
    for cap in [0, None]:
        with pytest.raises(ValueError):
            verify_indexed_witness(payload, cap=cap)


@pytest.mark.parametrize("cap", [-1, True, False, 1.5, "3", float("inf")])
def test_invalid_limits(cap):
    """REQ-REPORT-8858: reject ambiguous or negative resource limits at entry."""
    with pytest.raises(ValueError, match="primitive_limit"):
        verify_indexed_witness({"facts": [], "query": [["a", "r", "b"]]}, cap=cap)


def test_all_budget_rejections_and_context_cleanup():
    """REQ-REPORT-8858: every charged primitive can exhaust without returning partial proof."""
    sealed = json.loads(FIXTURES.read_bytes())
    for unit in sealed["rows"][3600:]:
        payload = unit["input"]
        full = verify_indexed_witness(payload, cap=None)
        for cap in range(full["primitive_operations"]):
            row = verify_indexed_witness(payload, cap=cap)
            assert row["budget_exhausted"] and row["decision"] == "insufficient"
            assert row["witness"] is None and row["supporting_bindings"] == 0
            assert row["primitive_operations"] == cap == sum(row["operation_counts"].values())
        assert stable(verify_indexed_witness(payload, cap=full["primitive_operations"])) == dict(
            stable(full), primitive_limit=full["primitive_operations"]
        )
    assert core.REQUEST.get() is None
    with pytest.raises(LookupError):
        meter.CURRENT.get()


def test_immutable_snapshot_and_stale_ownership():
    """REQ-REPORT-8858: source mutation and a foreign or expired owner cannot change an index."""
    payload = {"facts": [["a", "r", "b", True]] * 2, "query": [["?x", "r", "b"]]}
    token = meter.CURRENT.set(meter.Counter(None))
    meter.CURRENT.get().counts.update(dict.fromkeys(core.EXTRA, 0))
    request = core.REQUEST.set(object())
    try:
        index, _, _ = core.build_index(payload)
        payload["facts"][0][0] = "changed"
        assert index.candidates(["?x", "r", "b"], {}) == (("a", "r", "b", True),) * 2
        with pytest.raises(TypeError):
            index.table[("r", None, "b")] = ()
        nested = core.REQUEST.set(object())
        try:
            with pytest.raises(ValueError, match="stale_request_index"):
                index.candidates(["?x", "r", "b"], {})
        finally:
            core.REQUEST.reset(nested)
        assert len(index.candidates(["?x", "r", "b"], {})) == 2
    finally:
        core.REQUEST.reset(request)
    try:
        with pytest.raises(ValueError, match="stale_request_index"):
            index.candidates(["?x", "r", "b"], {})
    finally:
        meter.CURRENT.reset(token)


def test_concurrent_and_nested_requests(monkeypatch):
    """REQ-REPORT-8858: nested and concurrent calls restore the caller's meter and owner."""
    units = json.loads(FIXTURES.read_bytes())["rows"][-12:]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda u: stable(verify_indexed_witness(u["input"])), units))
    assert actual == [u["expected"] for u in units]
    original = core.solve
    active = False

    def nested(payload):
        nonlocal active
        if not active:
            active = True
            owner, paid = core.REQUEST.get(), meter.CURRENT.get()
            assert stable(verify_indexed_witness(units[-1]["input"])) == units[-1]["expected"]
            assert core.REQUEST.get() is owner and meter.CURRENT.get() is paid
            active = False
        return original(payload)

    monkeypatch.setattr(core, "solve", nested)
    assert stable(verify_indexed_witness(units[0]["input"])) == units[0]["expected"]


def test_public_exports_and_failure_cleanup(monkeypatch):
    """REQ-REPORT-8858: public aliases agree and unexpected failures release request state."""
    import carnot.verify as verify

    assert verify.verify_indexed_witness is verify_indexed_witness
    assert "verify_indexed_witness" in verify.__all__
    for name in verify.__all__:
        assert getattr(verify, name) is not None
    with pytest.raises(AttributeError):
        _ = verify.unknown_export

    def fail(payload):
        raise RuntimeError("owned_failure")

    monkeypatch.setattr(core, "solve", fail)
    with pytest.raises(RuntimeError, match="owned_failure"):
        verify_indexed_witness({"facts": [], "query": [["a", "r", "b"]]})
    assert core.REQUEST.get() is None
    with pytest.raises(LookupError):
        meter.CURRENT.get()


def test_authority_pin_and_algorithm_closure():
    """REQ-REPORT-8858: freeze the old fixtures and pure algorithm without editing its authorities."""
    import ast
    import hashlib
    import inspect
    from carnot.reporting import indexed_witness_8845 as old

    assert (
        hashlib.sha256(FIXTURES.read_bytes()).hexdigest()
        == "525c73f8e13eb870e7e6e637f4858f04a19f43cba24b181f67509855e3187235"
    )
    for name in ["allocate", "copy_binding", "RequestIndex", "build_index", "solve"]:
        before = inspect.getsource(getattr(old, name)).replace(
            "parse()(payload)", "meter.parse(payload)"
        )
        after = inspect.getsource(getattr(core, name))
        assert ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))


def test_runner_preconditions_and_private_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8858-CONTRACT: installed children run from private scratch and replay exactly."""
    from carnot.reporting import installed_indexed_runner_8858 as r

    pre = r.preconditions(tmp_path / "pre")
    assert pre["external_ready"] and pre["resources"]["passed"], pre
    output = tmp_path / (r.NAME + ".json")
    assert (
        r.main(
            [
                "--date",
                "20261106",
                "--private-output",
                "--private-unit-limit",
                "3",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["installed_index_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"
    assert r.replay(output)
    value["rows"][0]["decision"] = "invented"
    from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert not r.replay(output)
    assert not r.replay(tmp_path / "missing")


def test_runner_cli_rejection(tmp_path):
    """REQ-REPORT-8858: private subsets cannot publish into the repository or use an invalid date."""
    from carnot.reporting import installed_indexed_runner_8858 as r

    for args in [
        ["--date", "20250101"],
        ["--private-output"],
        ["--private-unit-limit", "1"],
        ["--private-output", "--private-unit-limit", "0", "--output", str(tmp_path / "x.json")],
    ]:
        with pytest.raises(SystemExit):
            r.main(args)
    assert r.main(["--cold-replay", str(tmp_path / "missing")]) == 1


def test_direct_probe_and_cli_child(tmp_path):
    """REQ-REPORT-8858: actual Python children exercise the probe and outside-checkout CLI."""
    import os
    import subprocess
    import sys
    from carnot.reporting import installed_indexed_runner_8858 as r
    from carnot.reporting.current_work_receipt import atomic_json

    inputs = tmp_path / "requests.json"
    atomic_json(inputs, {"rows": json.loads(FIXTURES.read_bytes())["rows"][-14:]})
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    probe = ROOT / "scripts/experiments/installed_indexed_probe_8858.py"
    result = subprocess.run(
        [sys.executable, "-u", str(probe), str(inputs), str(tmp_path / "observed.json")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [sys.executable, "-u", str(ROOT / r.CLI), "--cold-replay", str(tmp_path / "absent")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1 and "replay_rejected" in result.stdout


def seal_runner_work(tmp_path, monkeypatch, *, count=2):
    """Create actual evaluator rows while isolating publication mutation tests from package builds."""
    from carnot.reporting import installed_indexed_runner_8858 as r
    from carnot.reporting.current_work_receipt import atomic_json

    units = json.loads(FIXTURES.read_bytes())["rows"][:count]
    package = tmp_path / "package.whl"
    package.write_bytes(b"unit test package boundary")
    proof = dict(
        rows=[verify_indexed_witness(u["input"]) for u in units],
        installed_import_rows=[
            dict(check="installed_outside_checkout_no_pythonpath", passed=True),
            dict(check="no_reporting_training_or_artifact_loaders", passed=True),
        ],
        receipts=[dict(passed=True)],
        package_hash=r.reference(package),
    )
    monkeypatch.setattr(r, "installed", lambda *args: deepcopy(proof))
    raw = tmp_path / "raw"
    work = r.prepare(raw, count)
    work.update(caller_inventory=[], code_config_hashes=[], raw_shard_hashes=[])
    validation = r.unqualified()
    output = tmp_path / (r.NAME + ".json")

    def seal():
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        value = r.artifact(
            work,
            validation,
            "20261106",
            r.reference(raw / "work.json"),
            r.reference(raw / "validation.json"),
        )
        atomic_json(output, value)
        return value

    return r, work, validation, proof, raw, output, seal


def test_precondition_failure_classes_and_resources(tmp_path, monkeypatch):
    """REQ-REPORT-8858: distinguish missing inputs, rejected producers and failed owned resources."""
    from carnot.reporting import installed_indexed_runner_8858 as r
    from carnot.reporting.current_work_receipt import atomic_json

    with monkeypatch.context() as patch:
        patch.setattr(r.cost, "NAME", "missing8858")
        assert r.preconditions(tmp_path / "missing")["diagnostics"][-1]["check"] == "missing_file"
    original = r.load(r.reference(r.cost.ROOT / "results" / (r.cost.NAME + ".json")))
    real_load = r.load
    for bad, check in [
        ({}, "missing_field"),
        ({"indexed_cost_ready_score": 0}, "observed_zero"),
        (dict(original, task_id="wrong"), "rejected_producer"),
    ]:
        with monkeypatch.context() as patch:
            patch.setattr(
                r,
                "load",
                lambda ref: bad if ref["path"].endswith(r.cost.NAME + ".json") else real_load(ref),
            )
            assert r.preconditions(tmp_path / check)["diagnostics"][-1]["check"] == check
    with monkeypatch.context() as patch:
        patch.setattr(r, "FIXTURE_PIN", "sha256:drift")
        assert r.preconditions(tmp_path / "drift")["diagnostics"][-1]["error"] == "operand_drift"
    with monkeypatch.context() as patch:
        patch.setattr(r.shutil, "which", lambda name: None)
        work = r.prepare(tmp_path / "resource", 2)
        assert not work["preconditions_checked"]["resources"]["passed"]
        assert all(u["status"] == "pending" for u in work["rows"])
    with monkeypatch.context() as patch:
        patch.setattr(r, "FIXTURES", tmp_path / "absent")
        assert r.prepare(tmp_path / "no_fixtures")["rows"] == []


def test_owned_build_install_and_measurement_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8858: failed builds and installed calls retain all scheduled work and disqualify."""
    from carnot.reporting import installed_indexed_runner_8858 as r

    units = json.loads(FIXTURES.read_bytes())["rows"][:2]
    with monkeypatch.context() as patch:
        patch.setattr(r, "stream", lambda *args, **kwargs: dict(passed=False))
        with pytest.raises(ValueError, match="package_build"):
            r.installed(tmp_path / "build", units)

    def fail_install(name, argv, logs, **kwargs):
        if name == "build":
            wheel = Path(argv[argv.index("--out-dir") + 1]) / "unit.whl"
            wheel.parent.mkdir(parents=True)
            wheel.write_bytes(b"unit wheel")
        return dict(passed=name not in {"install", "installed_probe"})

    with monkeypatch.context() as patch:
        patch.setattr(r, "stream", fail_install)
        with pytest.raises(ValueError, match="installed_execution"):
            r.installed(tmp_path / "install", units)

    def fail(*args):
        raise ValueError("owned_failure")

    monkeypatch.setattr(r, "installed", fail)
    work = r.prepare(tmp_path / "execution", 2)
    assert work["internal_errors"] == ["owned_failure"]
    assert len(work["rows"]) == 2 and all(u["status"] == "pending" for u in work["rows"])
    assert r.prepare(tmp_path / "no_measurement", 2, False)["package_hash"] is None


def test_replay_rejects_rehashed_semantic_and_ownership_attacks(tmp_path, monkeypatch):
    """REQ-REPORT-8858: rehashing cannot authorize changed witnesses, paid work or controls."""
    r, work, validation, proof, raw, output, seal = seal_runner_work(tmp_path, monkeypatch)
    original = deepcopy(work)
    seal()
    assert r.replay(output)
    for field in [
        "rows",
        "semantic_parity_rows",
        "operation_parity_rows",
        "installed_import_rows",
        "ownership_rows",
    ]:
        work.clear()
        work.update(deepcopy(original))
        if field == "rows":
            work[field][0]["witness"] = {"?x": "invented"}
        elif field in {"semantic_parity_rows", "operation_parity_rows"}:
            work[field][0]["metric"] = 0
        else:
            work[field] = []
        seal()
        assert not r.replay(output), field
    work.clear()
    work.update(deepcopy(original))
    seal()
    with monkeypatch.context() as patch:
        patch.setattr(r, "sha256_file", lambda path: "changed")
        assert not r.replay(output)
    with monkeypatch.context() as patch:
        patch.setattr(r, "validate_current_work_receipt", lambda *args: ["changed"])
        assert not r.replay(output)
    validation.update(
        receipts=[dict(passed=True)], union=dict(passed=True), coverage=dict(complete=True)
    )
    seal()
    with monkeypatch.context() as patch:
        patch.setattr(
            r,
            "namespace",
            lambda *args: __import__("types").SimpleNamespace(
                validation_evidence=lambda value: False
            ),
        )
        assert not r.replay(output)


def test_validation_binding_main_and_terminal_classes(tmp_path, monkeypatch):
    """REQ-REPORT-8858: owned checks use unchanged historical consumers and only full gates grant readiness."""
    from types import SimpleNamespace

    r, work, validation, proof, raw, output, seal = seal_runner_work(tmp_path, monkeypatch)
    validator = SimpleNamespace(validate=lambda raw, private: dict(configuration=True))
    with monkeypatch.context() as patch:
        patch.setattr(r, "namespace", lambda *args: validator)
        assert r.validate(tmp_path / "validation", tmp_path)["configuration"]
        assert validator.original.CONSUMERS == [
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
        assert validator.e.OWNED == r.OWNED
    validation.update(
        receipts=[dict(passed=True)], union=dict(passed=True), coverage=dict(complete=True)
    )
    work["private_unit_limit"] = None
    work["rows"] = work["rows"] * 1807
    assert seal()["installed_index_ready_score"] == 0
    full = json.loads(FIXTURES.read_bytes())["rows"]
    work["rows"] = [
        dict(u["expected"], unit_id=u["unit_id"], cluster_id=u["cluster_id"], status="completed")
        for u in full
    ]
    work["semantic_parity_rows"] = [dict(unit_id=u["unit_id"], passed=True) for u in full]
    work["operation_parity_rows"] = deepcopy(work["semantic_parity_rows"])
    assert seal()["installed_index_ready_score"] == 1
    work["ownership_rows"][0]["passed"] = False
    assert seal()["installed_index_ready_score"] == 0
    work["preconditions_checked"]["external_ready"] = False
    work["preconditions_checked"]["diagnostics"] = [
        dict(
            check="missing_file",
            upstream="original",
            artifact_field="authority",
            expected=1,
            observed=None,
            path="missing",
            sha256=None,
        )
    ]
    assert seal()["verdict_class"] == "blocked"
    monkeypatch.setattr(r, "prepare", lambda *args: work)
    monkeypatch.setattr(r, "validate", lambda *args: validation)
    monkeypatch.setattr(r, "publish_primary", lambda *args: dict(unit_boundary=True))
    assert r.main(["--output", str(output)]) == 0
    monkeypatch.setattr(r, "replay", lambda path: True)
    assert r.main(["--cold-replay", str(output)]) == 0
