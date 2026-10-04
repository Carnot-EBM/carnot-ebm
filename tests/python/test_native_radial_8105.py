"""REQ-VERIFY-8105 and REQ-REPORT-8105: qualify the actual language boundary."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import native_radial_8105 as k


@pytest.fixture(scope="session")
def native():
    """Build once so every assertion crosses the executing interpreter's binding."""
    return k.extension()[0]


def state(dim=9, count=16, seed=0):
    """Nonzero fitted coefficients expose restart bugs hidden by zero baselines."""
    rng = np.random.default_rng(seed)
    return dict(
        geometry=dict(mean=[0.0] * dim, std=[1.0] * dim, sigma=2.0),
        centers=[dict(x=rng.normal(size=dim).tolist(), source_id=f"c{i}") for i in range(count)],
        coefficients=rng.normal(size=count + 1).tolist(),
        version=3,
        commit_hash="fixture",
    )


@pytest.mark.parametrize("dim", [1, 9])
@pytest.mark.parametrize("count", [16, 20, 24, 28])
def test_req_verify_8105_seeded_numpy(native, dim, count):
    """REQ-VERIFY-8105: 64 systems compare independent probabilities and gradients."""
    for seed in range(64):
        s = state(dim, count, seed)
        x = np.random.default_rng(seed + 100).normal(size=(8, dim))
        y = np.arange(8) % 2
        n = native.RustRadial8105(json.dumps(s))
        phi, p, grad = k.reference(s, x, y)
        np.testing.assert_allclose(n.design(x.tolist()), phi, atol=1e-14)
        np.testing.assert_allclose(n.predict(x.tolist()), p, atol=1e-10, rtol=0)
        np.testing.assert_allclose(n.gradient(x.tolist(), y.tolist(), 0.01), grad, atol=1e-12)
        native_p, native_actions, _ = k.service(s, x, n)
        python_p, python_actions, _ = k.service(s, x)
        np.testing.assert_allclose(native_p, python_p, atol=1e-10, rtol=0)
        assert native_actions == python_actions


def test_req_verify_8105_boundaries(native):
    """SCENARIO-VERIFY-8105-BOUNDARY: ties, duplicates and saturation are explicit."""
    s = state(1)
    for center in s["centers"]:
        center["x"] = [0.0]
    x = np.zeros((3, 1))
    for p in [0.1 - 1e-9, 0.1, 0.1 + 1e-9, 0.5 - 1e-9, 0.5, 0.5 + 1e-9]:
        s["coefficients"] = [float(np.log(p / (1 - p)))] + [0.0] * 16
        n = native.RustRadial8105(json.dumps(s))
        a, b = k.service(s, x, n), k.service(s, x)
        assert a[1] == b[1] and a[2] == b[2] == [True] * 3
    for intercept in [-1000.0, 1000.0]:
        s["coefficients"][0] = intercept
        assert (
            native.RustRadial8105(json.dumps(s)).predict(x.tolist()) == [float(intercept > 0)] * 3
        )
    s["coefficients"] = [1.0] + [0.0] * 16
    assert native.RustRadial8105(json.dumps(s)).design([[1e308]]) == [[1.0] + [0.0] * 16]
    for change in [
        dict(coefficients=[1.0]),
        dict(centers=[]),
        dict(geometry={}),
        dict(coefficients=[float("nan")] * 17),
    ]:
        with pytest.raises(ValueError):
            native.RustRadial8105(json.dumps(dict(s, **change)))
    n = native.RustRadial8105(json.dumps(s))
    for xbad in [[[float("inf")]], [[1.0, 2.0]]]:
        with pytest.raises(ValueError):
            n.predict(xbad)
    for labels, ridge in [([], 0.01), ([2], 0.01), ([1], -1.0), ([1], float("nan"))]:
        with pytest.raises(ValueError):
            n.gradient([[0.0]], labels, ridge)
    with pytest.raises(ValueError):
        n.gradient([], [], 0.01)


def test_req_verify_8105_serialization(native):
    """SCENARIO-VERIFY-8105-BOUNDARY: nonzero baseline and zero growth round-trip."""
    s = state()
    x = np.ones((2, 9))
    n = native.RustRadial8105(json.dumps(s))
    encoded = n.checkpoint()
    restored = native.RustRadial8105.restore(encoded)
    assert json.loads(restored.state_json()) == s
    assert restored.predict(x.tolist()) == n.predict(x.tolist())
    extended = deepcopy(s)
    extended["centers"] += [dict(x=[0.0] * 9, source_id=f"extra-{i}") for i in range(4)]
    extended["coefficients"] += [0.0] * 4
    assert native.RustRadial8105(json.dumps(extended)).predict(x.tolist()) == n.predict(x.tolist())
    changed = json.loads(encoded)
    changed["payload"] += " "
    for bad in ["{}", "[]", json.dumps(changed)]:
        with pytest.raises(ValueError):
            native.RustRadial8105.restore(bad)


def test_req_report_8105_copy_and_extension(tmp_path, native, monkeypatch):
    """REQ-REPORT-8105: a mapped binary's old inode retains its original bytes."""
    source, destination = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    with destination.open("rb") as old:
        k.copy_extension(source, destination)
        assert old.read() == b"old"
    assert destination.read_bytes() == b"new"
    monkeypatch.setenv("CARNOT_8105_EXTENSION", native.__file__)
    assert k.extension()[1]["actual_loaded"]
    monkeypatch.setattr(k.build, "load_native_extension", lambda p: object())
    with pytest.raises(ValueError, match="missing_radial"):
        k.extension()


def test_req_report_8105_cli_routes(tmp_path, native):
    """SCENARIO-REPORT-8105-CLI: real children run outside the checkout and exit."""
    import os
    import subprocess
    import sys
    from carnot import experiment_8105_v701_native_radial_kernel as e

    env = dict(
        os.environ, CARNOT_8105_EXTENSION=native.__file__, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1"
    )
    env.pop("PYTHONPATH", None)
    success = tmp_path / "good" / (e.NAME + ".json")
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    mutated = tmp_path / "mutated" / (e.NAME + ".json")
    rejected = mutated.parent / "raw" / mutated.stem / "failed_mutation_candidate.json"
    calls = [
        (["--fixture-output", str(success)], 0),
        (["--cold-replay", str(success)], 0),
        (["--fixture-output", str(blocked), "--root", str(tmp_path / "absent")], 0),
        (["--fixture-output", str(mutated), "--mutate"], 1),
        (["--cold-replay", str(rejected)], 1),
        (["--fixture-output", str(success)], 1),
    ]
    import time

    receipts = []
    for args, expected in calls:
        began = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        log = tmp_path / f"cli-{len(receipts)}.log"
        log.write_text(result.stdout + result.stderr)
        receipts.append(
            dict(
                name=f"private_cli_{len(receipts)}",
                command_argv=result.args,
                scope="owned",
                actual_exit=result.returncode,
                expected_exit=expected,
                normal_exit=result.returncode >= 0,
                duration_s=time.monotonic() - began,
                log_path=str(log),
                log_sha256=e.sha256_file(log),
                passed=result.returncode == expected,
            )
        )
        assert result.returncode == expected, result.stdout + result.stderr
    e.atomic_json(tmp_path / "cli_receipts.json", dict(receipts=receipts))
    assert json.loads(success.read_text())["verdict_class"] == "circular_positive"
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert json.loads(rejected.read_text())["native_kernel_ready_score"] == 0
    value = json.loads(success.read_text())
    value["completed_count"] -= 1
    altered = tmp_path / "altered.json"
    e.atomic_json(altered, value)
    assert not e.replay(altered)
    assert not e.replay(tmp_path / "missing.json")


def test_req_report_8105_owned_failure(tmp_path, native, monkeypatch):
    """REQ-REPORT-8105: failed checks suppress readiness; consumer sees exact identity."""
    from carnot import experiment_8105_v701_native_radial_kernel as e

    monkeypatch.setenv("CARNOT_8105_EXTENSION", native.__file__)
    monkeypatch.setattr(e, "validate", lambda *args: [dict(passed=False, scope="owned")])
    output = tmp_path / (e.NAME + ".json")
    health_log = tmp_path / "health.log"
    health_log.write_text("bounded global diagnostic\n")
    e.atomic_json(
        output.parent / "raw" / output.stem / "repository_health_once.json",
        dict(
            receipts=[
                dict(
                    passed=False,
                    scope="repository_health",
                    log_path=str(health_log),
                    log_sha256=e.sha256_file(health_log),
                )
            ]
        ),
    )
    assert e.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert e.reader_receipt(e.TASK, tmp_path, field="native_kernel_ready_score", expected=0)[
        "passed"
    ]


def test_req_report_8105_worker_replay_rejection(tmp_path, native, monkeypatch):
    """REQ-REPORT-8105: durable operands and aggregate claims cannot drift independently."""
    from carnot import experiment_8105_v701_native_radial_kernel as e
    import runpy
    import sys

    monkeypatch.setenv("CARNOT_8105_EXTENSION", native.__file__)
    raw = tmp_path / "raw"
    assert e.main(["--worker-output", str(raw / "work.json")]) == 0
    work = json.loads((raw / "work.json").read_text())
    value = e.build(work, [dict(passed=True, scope="owned")], raw, True, False)
    output = tmp_path / (e.NAME + ".json")
    e.atomic_json(output, value)
    assert e.replay(output)
    for field, change in [
        ("code_config_hashes", {e.OWNED[0]: "wrong"}),
        ("source_artifact_hashes", [dict(path=str(raw / "work.json"), sha256="wrong")]),
        ("native_library_sha256", "wrong"),
        (
            "validation_receipts",
            [dict(passed=True, log_path=str(raw / "work.json"), log_sha256="wrong")],
        ),
    ]:
        altered = dict(value, **{field: change})
        e.atomic_json(output, altered)
        assert not e.replay(output)
    e.atomic_json(output, value)
    changed = deepcopy(work)
    changed["evidence"]["systems"][0]["numpy_probabilities"][0] = 100
    e.atomic_json(raw / "work.json", changed)
    assert not e.replay(output)
    e.atomic_json(raw / "evidence.json", changed["evidence"])
    changed["raw_shard_hashes"][0]["sha256"] = e.sha256_file(raw / "evidence.json")
    altered = e.build(changed, value["validation_receipts"], raw, True, False)
    e.atomic_json(raw / "work.json", changed)
    e.atomic_json(output, altered)
    assert not e.replay(output)
    monkeypatch.setattr(sys, "argv", [e.CLI, "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert exc.value.code == 1


def test_req_report_8105_preconditions_and_validation(tmp_path, native, monkeypatch):
    """SCENARIO-REPORT-8105-CLI: missing terminal custody and failed subprocesses stay visible."""
    from carnot import experiment_8105_v701_native_radial_kernel as e

    e.atomic_json(tmp_path / "results/experiment_8085_test.json", {})
    monkeypatch.setattr(e, "INPUTS", ["results/experiment_8085_test.json"])
    assert (
        e.prerequisites(tmp_path, tmp_path / "raw")["gate_check_summary"][0]["check"]
        == "terminal_binding"
    )
    assert e.native_coverage(tmp_path, tmp_path / "raw")[0]["passed"] is False
    (tmp_path / "native-test.profraw").write_text("profile fixture")
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=False)])
    assert e.native_coverage(tmp_path, tmp_path / "raw")[0]["passed"] is False
    log = tmp_path / "coverage.log"
    log.write_text("1|1|let x=1;\n2|0|#[pymethods]\n")
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=True, log_path=str(log))])
    assert e.native_coverage(tmp_path, tmp_path / "raw")[-1]["passed"]
    receipt_log = tmp_path / "cli-source.log"
    receipt_log.write_text("private CLI fixture\n")
    e.atomic_json(
        tmp_path / "cli_receipts.json",
        dict(receipts=[dict(name="fixture", passed=True, log_path=str(receipt_log))]),
    )
    e.atomic_json(tmp_path / "coverage.json", dict(files={}))
    assert e.validate([], tmp_path / "raw", tmp_path)
    monkeypatch.setenv("CARNOT_8105_EXTENSION", native.__file__)
    monkeypatch.delenv("CARNOT_8105_EXTENSION")
    monkeypatch.setattr(k, "ROOT", tmp_path)
    monkeypatch.setattr(k.build, "build_native_extension", lambda root: (Path(native.__file__), {}))
    assert k.extension()[1]["actual_loaded"]


def test_req_report_8105_child_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8105: an abnormal child cannot publish a readiness claim."""
    from carnot import experiment_8105_v701_native_radial_kernel as e

    monkeypatch.setattr(e, "run_check", lambda *a: dict(passed=False))
    monkeypatch.setattr(e, "terminal", lambda p: dict(passed=True))
    assert e.main(["--fixture-output", str(tmp_path / (e.NAME + ".json"))]) == 0
    assert (
        json.loads((tmp_path / (e.NAME + ".json")).read_text())["verdict_class"] == "disqualified"
    )


def test_req_report_8105_invalid_native_control(tmp_path, native, monkeypatch):
    """REQ-VERIFY-8105: a native arm accepting nonfinite operands cannot qualify."""
    from carnot import experiment_8105_v701_native_radial_kernel as e
    from types import SimpleNamespace

    class Unsafe:
        def __init__(self, encoded):
            self.inner = native.RustRadial8105(encoded)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def predict(self, rows):
            if np.isnan(np.asarray(rows)).any():
                return [0.0]
            return self.inner.predict(rows)

        restore = staticmethod(native.RustRadial8105.restore)

    evidence = e.measure(tmp_path, SimpleNamespace(RustRadial8105=Unsafe))
    assert not e.reduction(evidence)["passed"]


def test_req_report_8105_coverage_startup(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8105-CLI: startup instrumentation must not stack active collectors."""
    import coverage
    import runpy
    import sys
    from carnot import experiment_8105_v701_native_radial_kernel as e

    monkeypatch.setenv("CARNOT_8105_COVERAGE_START", str(tmp_path / "coverage.ini"))
    monkeypatch.setattr(coverage.Coverage, "current", lambda: None)
    monkeypatch.setattr(coverage, "process_startup", lambda: None)
    monkeypatch.setattr(sys, "argv", [e.CLI, "--cold-replay", str(tmp_path / "absent.json")])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert exc.value.code == 1
