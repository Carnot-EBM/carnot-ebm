"""REQ-VERIFY-8379 / REQ-REPORT-8379: qualify executed arithmetic and sealed rows."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import native_direct_8379 as k
from carnot.reporting import native_direct_8379 as e


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    """SCENARIO-VERIFY-8379-EXECUTION: assertions cross a real compiled binding."""
    private = tmp_path_factory.mktemp("native8379")
    return k.extension(private, private / "build")


@pytest.fixture(scope="module")
def measured(tmp_path_factory, native):
    """SCENARIO-VERIFY-8379-PARITY: retain actual primitive rows for replay."""
    raw = tmp_path_factory.mktemp("primitive8379")
    return e.measure(e.ROOT, raw, raw, native[1]["path"]), raw


def head():
    """REQ-VERIFY-8379: the frozen protocol is the sole deployed head authority."""
    return json.loads((e.ROOT / e.authority.PROTOCOL).read_bytes())["head"]


def test_req_verify_8379_python_coordinator(tmp_path, native):
    """REQ-VERIFY-8379: the existing durable writer executes the opt-in providers."""
    from carnot.verify import direct_atomic_state_8376 as original

    coordinator = k.coordinator(native[0])
    frozen = original.trace(11, head())
    frozen["events"] = [frozen["events"][i] for i in (0, 1, 9, 11)]
    store = coordinator.Store(tmp_path, frozen)
    store.initialize()
    for event in frozen["events"]:
        store.apply(event)
    result = store.read()
    assert result == coordinator.fold(frozen)
    python = original.fold(frozen)
    assert result["pending"] == python["pending"]
    assert result["applied"].keys() == python["applied"].keys()
    assert (
        max(
            abs(a - b)
            for a, b in zip(result["head"]["coefficients"], python["head"]["coefficients"])
        )
        <= 1e-12
    )
    assert all(
        result["issued"][i]["action"] == python["issued"][i]["action"] for i in result["issued"]
    )


def test_req_verify_8379_payload_copies(native):
    """REQ-VERIFY-8379: count actual binding operands, including rejected copies."""
    module, counts = k.tracked(native[0])
    module.direct_design_8379([0.0] * 5)
    with pytest.raises(ValueError):
        module.direct_design_8379([0.0])
    assert counts["native_invocation_count"] == 2
    assert counts["binding_copy_bytes"] == 8 * (5 + 34 + 1)
    assert counts["binding_conversion_bytes"] == counts["binding_copy_bytes"]


def test_req_verify_8379_basis_and_updates(native):
    module = native[0]
    h = head()
    for value in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.387]:
        x = [0.7, value, value, value, value]
        assert module.direct_design_8379(x) == k.kernel.design(x)
    for seed in (11, 22, 33):
        rng = np.random.default_rng(seed)
        py, rust = deepcopy(h), deepcopy(h)
        for slot in range(32):
            x = [float(rng.normal()), *rng.random(4).tolist()]
            expected = k.optimizer.learn(py, x, slot % 2, "online_sparse")
            actual = k.learn(rust, x, slot % 2, module)
            assert max(abs(a - b) for a, b in zip(actual, expected["coefficients"])) <= 1e-12
            py["coefficients"], rust["coefficients"] = expected["coefficients"], actual
    assert k.probabilities(h, [[0.0, 0.5, 0.5, 0.5, 0.5]], module).shape == (1,)


@pytest.mark.parametrize(
    "bad",
    [
        [0.0],
        [float("nan")] * 5,
        [float("inf")] * 5,
        [0.0, -1e-20, 0.0, 0.0, 0.0],
        [0.0, 1.00001, 0.0, 0.0, 0.0],
    ],
)
def test_req_verify_8379_reject_features(native, bad):
    for arm in (None, native[0]):
        with pytest.raises(ValueError):
            k.probabilities(head(), [bad], arm)
    with pytest.raises(ValueError):
        native[0].direct_design_8379(bad)


def test_req_verify_8379_invalid_heads(native):
    for change in [
        dict(coefficients=[0.0]),
        dict(coefficients=[float("nan")] * 34),
        dict(coefficients=[5.0] * 34),
        dict(temperature=0.0),
        dict(temperature=float("inf")),
    ]:
        h = dict(head(), **change)
        for arm in (None, native[0]):
            with pytest.raises(ValueError):
                k.probabilities(h, [[0.0] * 5], arm)
        with pytest.raises(ValueError):
            native[0].direct_logits_8379(h["coefficients"], [[0.0] * 5], h["temperature"])
    with pytest.raises(ValueError):
        native[0].direct_update_8379([0.0] * 34, [0.0] * 5, float("nan"))
    with pytest.raises(ValueError):
        k.learn(head(), [0.0] * 5, 2, native[0])


def test_req_report_8379_replay_and_dispositions(measured, tmp_path):
    work, _ = measured
    value = e.build(work, [dict(name="unit", passed=True, scope="owned")])
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    assert value["native_invocation_count"] > 0
    assert value["independent_generalization_score"] == 0
    assert value["binding_copy_bytes"] > 0
    value["max_probability_error"] += 0.1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "absent")
    assert e.build(work, [dict(passed=False, scope="owned")])["verdict_class"] == "disqualified"
    blocked = deepcopy(work)
    blocked["input_ready"] = False
    assert e.build(blocked, [dict(passed=True, scope="owned")])["verdict_class"] == "blocked"


def test_req_report_8379_missing_input(tmp_path, native):
    work = e.measure(tmp_path / "missing", tmp_path / "raw", tmp_path, native[1]["path"])
    value = e.build(work, [dict(passed=True, scope="owned")])
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["observed"] is None
    assert value["native_parity_ready_score"] == 0


def test_req_report_8379_cli(tmp_path, native):
    script = e.ROOT / e.CLI
    path = tmp_path / (e.NAME + ".json")
    env = dict(os.environ, CARNOT_8379_EXTENSION=native[1]["path"], PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    commands = [
        (["--output", str(path), "--private-fixture"], 1),
        (["--cold-replay", str(path)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--deliberate-error"], 1),
        (["--date", "20000101"], 2),
        (
            [
                "--output",
                str(tmp_path / "blocked" / path.name),
                "--root",
                str(tmp_path / "absent"),
                "--private-fixture",
            ],
            0,
        ),
    ]
    for index, (args, expected) in enumerate(commands):
        print(
            f"[test8379] before_cli completed={index} pending={len(commands) - index}", flush=True
        )
        began = time.monotonic()
        process = subprocess.Popen(
            [sys.executable, "-u", str(script), *args],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        while True:
            try:
                stdout, stderr = process.communicate(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(
                    f"[test8379] waiting_cli completed={index} pending={len(commands) - index}",
                    flush=True,
                )
                assert time.monotonic() - began < 300
        (tmp_path / f"cli-{index}.log").write_text(stdout + stderr)
        print(
            f"[test8379] after_cli completed={index + 1} pending={len(commands) - index - 1}",
            flush=True,
        )
        assert process.returncode == expected, stdout + stderr


def test_req_report_8379_frozen_plan_and_real_main(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8379-CLI: execute production orchestration on absent operands."""
    from carnot.reporting import native_direct_runner_8379 as runner

    for key in ("COVERAGE_RCFILE", "COVERAGE_FILE", "TMPDIR", "CARNOT_8379_EXTENSION"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    private = tmp_path / "plan"
    private.mkdir()
    frozen = runner.plan(private)
    assert {r["name"] for r in frozen} >= {
        "owned_tests",
        "coverage_100",
        "private_E2E018_consumers",
    }
    monkeypatch.setattr(runner, "SCRATCH", tmp_path / "scratch")

    def short_plan(directory):
        (directory / "coverage.json").write_text('{"private_control": true}')
        return [
            dict(
                name="real_child_pass",
                argv=[sys.executable, "-c", "raise SystemExit(0)"],
                deadline_s=30,
                scope="owned",
            )
        ]

    monkeypatch.setattr(runner, "plan", short_plan)
    output = tmp_path / "blocked" / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    with pytest.raises(SystemExit) as rejected:
        runner.main(["--private-fixture"])
    assert rejected.value.code == 2

    def failure(*args):
        raise ValueError("deliberate_owned_control")

    monkeypatch.setattr(e, "measure", failure)
    assert runner.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 1


def test_req_verify_8379_extension_failure_controls(tmp_path, native, monkeypatch):
    """SCENARIO-VERIFY-8379-EXECUTION: failure mocks cannot qualify native execution."""
    with pytest.raises(ValueError, match="extension_spec"):
        k.extension(tmp_path, tmp_path, str(tmp_path / "unsupported.extension"))
    with monkeypatch.context() as patch:
        from types import SimpleNamespace

        patch.setattr(
            k.importlib.util,
            "spec_from_file_location",
            lambda *a: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)),
        )
        patch.setattr(k.importlib.util, "module_from_spec", lambda s: SimpleNamespace())
        with pytest.raises(ValueError, match="native_entrypoints"):
            k.extension(tmp_path, tmp_path, native[1]["path"])
    with monkeypatch.context() as patch:
        patch.setattr(k.importlib.util, "spec_from_file_location", lambda *a: None)
        with pytest.raises(ValueError, match="coordinator_spec"):
            k.coordinator(native[0])
    from carnot.reporting.v709_execution import child

    with monkeypatch.context() as patch:
        patch.delenv("CARNOT_8379_EXTENSION", raising=False)
        patch.setattr(
            k,
            "child",
            lambda name, argv, logs, **kw: child(
                name, [sys.executable, "-c", "raise SystemExit(1)"], logs, deadline=30
            ),
        )
        with pytest.raises(ValueError, match="native_build_failed"):
            k.extension(tmp_path / "failure", tmp_path / "failed_logs")
    h = dict(head(), temperature=float(np.nextafter(0.0, 1.0)))
    with pytest.raises(ValueError, match="residual"):
        k.learn(h, [-100.0, 0.0, 0.0, 0.0, 0.0], 1, native[0])
    assert native[0].direct_update_8379([0.0] * 34, [0.0] * 5, 0.0) == [0.0] * 34


def test_req_report_8379_operand_and_resource_failures(tmp_path, native, monkeypatch):
    """REQ-REPORT-8379: changed or absent authenticated operands never become zero."""
    import shutil

    root = tmp_path / "root"
    for relative in (e.authority.PROTOCOL, e.authority.METHODS, "research-roadmap.yaml"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.ROOT / relative, target)
    protocol = json.loads((root / e.authority.PROTOCOL).read_bytes())
    protocol["checkpoint"]["path"] = str(tmp_path / "missing_checkpoint")
    atomic_json(root / e.authority.PROTOCOL, protocol)
    monkeypatch.setattr(
        e.authority, "PROTOCOL_PIN", e.reference(root / e.authority.PROTOCOL)["sha256"]
    )
    bound, _, gates = e.bind(root, tmp_path / "bound")
    assert bound is None and gates[0]["check"] == "direct_operand_hash"
    with monkeypatch.context() as patch:
        patch.setattr(e.shutil, "which", lambda name: None)
        work = e.measure(tmp_path / "absent", tmp_path / "resource", tmp_path, native[1]["path"])
        assert any(g["check"] == "resource_preconditions" for g in work["gate_check_summary"])
    with monkeypatch.context() as patch:
        patch.setattr(
            e.authority,
            "PROTOCOL_PIN",
            "sha256:ab877c98112dc1a1497bb9aa1f2c28751e1b78671de743a6e4c036eeeddcc47e",
        )
        patch.setattr(
            k, "extension", lambda *a: (_ for _ in ()).throw(ValueError("owned_build_control"))
        )
        work = e.measure(e.ROOT, tmp_path / "failed_build", tmp_path)
        assert work["owned_failure"] == "owned_build_control"
        assert e.build(work, [dict(passed=True)])["verdict_class"] == "disqualified"


def test_req_report_8379_rehashed_operands(measured, tmp_path):
    """SCENARIO-REPORT-8379-CLI: repaired hashes cannot authorize false primitive rows."""
    work, _ = measured
    receipts = [dict(passed=True)]
    candidate = tmp_path / (e.NAME + ".json")
    measurement = Path(work["primitive_reference"]["path"]).parent / "measurement.json"
    saved = measurement.read_bytes()
    primitive = Path(work["primitive_reference"]["path"])
    panel = Path(work["panel_reference"]["path"])
    originals = {primitive: primitive.read_bytes(), panel: panel.read_bytes()}
    try:
        for case in (
            "checksum",
            "measurement",
            "operand",
            "panel_hash",
            "binary_hash",
            "panel_semantics",
            "primitive_semantics",
        ):
            for path, data in originals.items():
                path.write_bytes(data)
            changed = deepcopy(work)
            if case == "operand":
                changed["code_config_hashes"][0]["sha256"] = "sha256:invalid"
            if case == "panel_hash":
                changed["panel_reference"]["sha256"] = "sha256:invalid"
            if case == "binary_hash":
                changed["extension"]["sha256"] = "sha256:invalid"
            if case == "panel_semantics":
                altered = json.loads(panel.read_bytes())
                altered["head"]["coefficients"][0] += 0.125
                atomic_json(panel, altered)
                changed["panel_reference"] = e.reference(panel)
            if case == "primitive_semantics":
                altered = json.loads(primitive.read_bytes())
                altered["rows"][0]["native_probability"] += 0.125
                atomic_json(primitive, altered)
                changed["primitive_reference"] = e.reference(primitive)
            atomic_json(measurement, changed)
            value = e.build(changed, receipts)
            if case == "checksum":
                value["reproducibility_checksum"] = "sha256:invalid"
            elif case == "measurement":
                value["measurement_reference"]["sha256"] = "sha256:invalid"
                value.pop("reproducibility_checksum")
                value["reproducibility_checksum"] = canonical_hash(value)
            atomic_json(candidate, value)
            assert not e.replay(candidate), case
    finally:
        measurement.write_bytes(saved)
        for path, data in originals.items():
            path.write_bytes(data)


def test_req_verify_8379_deliberately_broken_rejection(native, monkeypatch):
    """REQ-VERIFY-8379: a falsely admitted input fails the declared rejection gates."""
    from types import SimpleNamespace

    panel = dict(
        head=head(),
        rows=[
            dict(
                unit_id="invalid", kind="invalid", input_rejected=True, x=[0.0, -1.0, 0.0, 0.0, 0.0]
            )
        ],
        updates=[],
        traces=[],
    )
    monkeypatch.setattr(k, "probabilities", lambda *a: np.zeros(1))
    bad = SimpleNamespace(
        direct_design_8379=lambda x: [0.0] * 34,
        direct_logits_8379=lambda *a: [0.0],
        direct_update_8379=native[0].direct_update_8379,
    )
    result = e.execute(panel, bad)
    assert result["rows"][0]["action_mismatch"] == 1
    assert not any(r["passed"] for r in result["rejection_controls"])


def test_req_report_8379_prior_receipts(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8379-CLI: freeze and authenticate real reused validation logs."""
    from carnot.reporting import native_direct_runner_8379 as runner
    from carnot.reporting.v709_execution import child

    for key in ("COVERAGE_RCFILE", "COVERAGE_FILE", "TMPDIR", "CARNOT_8379_EXTENSION"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    prior = child(
        "repository_health_once",
        [sys.executable, "-c", "raise SystemExit(0)"],
        tmp_path / "logs",
        deadline=30,
        scope="global",
    )
    receipt = tmp_path / "logs/repository_health_once.receipt.json"
    assert prior["passed"]
    extra = dict(prior, name="additional_control")
    atomic_json(tmp_path / "extra.json", extra)
    monkeypatch.setenv(
        "CARNOT_8379_PRIOR_VALIDATION", json.dumps([str(receipt), str(tmp_path / "extra.json")])
    )
    planned = runner.plan(tmp_path)
    reused = [row for row in planned if "receipt_reference" in row]
    assert len(reused) == 2
    assert runner.reuse(reused[0]) == prior
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="prior_receipt_hash"):
        runner.reuse(reused[0])
    receipt.write_text(json.dumps(prior))
    row = dict(reused[0], receipt_reference=e.reference(receipt))
    Path(prior["stdout_path"]).write_text("changed log")
    with pytest.raises(ValueError, match="prior_log_hash"):
        runner.reuse(row)


def test_req_verify_8379_exact_source_build(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8379-EXECUTION: real compilation is never replaced by a mock."""
    for key in ("TMPDIR", "PYO3_PYTHON", "RUSTFLAGS", "LLVM_PROFILE_FILE"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.delenv("CARNOT_8379_EXTENSION", raising=False)
    module, receipt = k.extension(tmp_path, tmp_path / "build")
    assert receipt["actual_loaded"] and all(row["passed"] for row in receipt["receipts"])
    assert module.direct_design_8379([0.0] * 5) == k.kernel.design([0.0] * 5)
    assert module.direct_update_8379([0.0] * 34, [0.0] * 5, 0.0) == [0.0] * 34


def test_req_verify_8379_empty_batch_rejection(native):
    """REQ-VERIFY-8379: an undefined empty batch must reject in both consumers."""
    for arm in (None, native[0]):
        with pytest.raises(ValueError):
            k.probabilities(head(), [], arm)
    with pytest.raises(ValueError):
        native[0].direct_logits_8379(head()["coefficients"], [], head()["temperature"])
