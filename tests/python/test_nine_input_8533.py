"""REQ-VERIFY-8533, REQ-REPORT-8533: real finite native controls and frozen requests."""

from copy import deepcopy
import json
from pathlib import Path
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from carnot.verify import nine_input_8533 as n
from carnot.verify import native_nine_input_8533 as build
from carnot.reporting import nine_input_inputs_8533 as inputs
from carnot.reporting import nine_input_qualification_8533 as evidence
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child
from carnot.reporting import nine_input_runner_8533 as runner
from carnot.reporting import nine_input_validation_8533 as validation


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    """SCENARIO-VERIFY-8533-PARITY: never replace Rust with a Python mirror."""
    path = tmp_path_factory.mktemp("native-8533")
    return build.extension(path, path / "logs")


def head():
    return dict(
        arm="spline",
        parameters=np.linspace(-0.4, 0.4, 109).tolist(),
        decay_scale=0.7,
        temperature=0.5,
        scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
    )


def test_finite_controls_and_negative_inputs(native):
    """SCENARIO-VERIFY-8533-PARITY: all dimensions, decay and native rejection."""
    module, ref = native
    rows = n.qualify(head(), np.asarray([[0.31] * 9, [0.0] * 9, [1.0] * 9]), module)
    assert rows and all(row["passed"] for row in rows)
    assert ref["actual_loaded"] and Path(ref["module_file"]).is_file()
    assert {r["kind"] for r in rows} >= {"answer", "knot", "zero", "decay", "malformed"}
    assert sum(r["native_execution_count"] for r in rows) > 0


def test_boundary_fallback_and_shared_policy(native):
    """REQ-VERIFY-8533: exact typed thresholds use the Python reference near a boundary."""
    module, _ = native
    h = head()
    h.update(parameters=[0.0] * 109, temperature=1.0)
    raw = np.asarray([[q] + [0.5] * 8 for q in [0.2, 0.8, 0.5, 0.1, 0.9]])
    p, fallbacks = n.probabilities(h, raw, module)
    assert fallbacks.tolist() == [True, True, False, False, False]
    assert np.array_equal(p, n.probabilities(h, raw)[0])
    assert [n.action(float(q)) for q in p] == ["accept", "reject", "escalate", "accept", "reject"]


def test_authenticated_tape_and_request_boundary(native, tmp_path):
    """REQ-REPORT-8533: frozen 900 rows and first64 stable test requests use actual text."""
    h, tape, refs, upstream = inputs.operands(inputs.ROOT, tmp_path)
    assert len(tape["rows"]) == 900 and len(tape["timed_ids"]) == 64
    assert tape["timed_ids"] == [r["unit_id"] for r in tape["rows"] if r["role"] == "test"][:64]
    assert len(h["parameters"]) == 109 and upstream["nine_input_inputs_ready_score"] == 1
    assert refs
    row = tape["rows"][0]
    a = json.loads(n.request(h, row))
    b = json.loads(n.request(h, row, native[0]))
    assert a["decision"] == b["decision"]
    assert abs(a["probability"] - b["probability"]) <= 1e-12
    changed = deepcopy(row)
    changed["public"]["source_bytes"] = b"".hex()
    with pytest.raises(ValueError, match="request_features"):
        n.request(h, changed)
    with pytest.raises(FileNotFoundError):
        inputs.operands(tmp_path / "absent", tmp_path / "blocked")


def test_cold_replay_rejects_rehashed_tamper(native, tmp_path):
    """SCENARIO-REPORT-8533-REPLAY: recomputation, not a repaired envelope hash, decides."""
    work = evidence.measure(inputs.ROOT, tmp_path / "raw")
    assert work["rows"] and not work["failures"]
    value = evidence.artifact(work, [], tmp_path / "out.json")
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert evidence.replay(candidate)
    value["native_execution_count"] += 1
    value["reproducibility_checksum"] = evidence.checksum(value)
    atomic_json(candidate, value)
    assert not evidence.replay(candidate)
    assert not evidence.replay(tmp_path / "missing")
    assert canonical_hash(work["rows"])


def test_offline_build_and_loader_rejections(monkeypatch, tmp_path):
    """SCENARIO-VERIFY-8533-PARITY: real pinned build plus owned build/import failures."""
    monkeypatch.delenv("CARNOT_8533_EXTENSION", raising=False)
    receipt = child(
        "offline_build",
        [
            "env",
            "CARNOT_8533_EXTENSION=",
            sys.executable,
            "-u",
            "-c",
            "from pathlib import Path; import sys; from carnot.verify.native_nine_input_8533 import extension; "
            "m,r=extension(Path(sys.argv[1]),Path(sys.argv[2])); "
            "assert m.nine_input_residual_8533([0.]*109,[[.5]*9]) == [0.]",
            str(tmp_path / "compile"),
            str(tmp_path / "logs"),
        ],
        tmp_path / "child_logs",
        deadline=120,
    )
    assert receipt["passed"]
    ref = dict(path=str(tmp_path / "native_binary/libnine_input_8533.so"))
    monkeypatch.setattr(build, "child", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="native_build_failed"):
        build.extension(tmp_path / "failed", tmp_path / "failed_logs")
    monkeypatch.setattr(build.importlib.util, "spec_from_file_location", lambda *a: None)
    with pytest.raises(ValueError, match="extension_spec"):
        build.load(Path(ref["path"]))
    fake = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None))
    monkeypatch.setattr(build.importlib.util, "spec_from_file_location", lambda *a: fake)
    monkeypatch.setattr(build.importlib.util, "module_from_spec", lambda s: SimpleNamespace())
    with pytest.raises(ValueError, match="extension_entrypoint"):
        build.load(Path(ref["path"]))


def test_owned_numeric_failure_and_blocked_inputs(native, monkeypatch, tmp_path):
    """REQ-REPORT-8533: external absence blocks; owned measurement failure disqualifies."""
    blocked = evidence.measure(tmp_path / "absent", tmp_path / "blocked")
    assert evidence.artifact(blocked, [], tmp_path / "blocked.json")["verdict_class"] == "blocked"
    with monkeypatch.context() as m:
        m.setattr(
            inputs, "operands", lambda *a: (_ for _ in ()).throw(ValueError("present_seal_failure"))
        )
        present = evidence.measure(inputs.ROOT, tmp_path / "present")
        assert present["failures"][0]["observed"] == 1
        assert present["failures"][0]["reason"] == "present_seal_failure"
    monkeypatch.setattr(evidence.shutil, "which", lambda *a: None)
    work = evidence.measure(inputs.ROOT, tmp_path / "no_tools")
    assert any(r["field"] == "tools_memory_scratch" for r in work["failures"])
    monkeypatch.undo()
    monkeypatch.setattr(build, "extension", lambda *a: (_ for _ in ()).throw(ValueError("owned")))
    work = evidence.measure(inputs.ROOT, tmp_path / "owned")
    assert evidence.artifact(work, [], tmp_path / "owned.json")["verdict_class"] == "disqualified"


def test_private_cli_publish_replay_and_absence(native, tmp_path):
    """SCENARIO-REPORT-8533-REPLAY: actual private CLI, invalid dates and explicit owned error."""
    output = tmp_path / (evidence.NAME + ".json")
    argv = [sys.executable, "-u", str(evidence.ROOT / evidence.CLI)]
    env = ["env", "CARNOT_8533_EXTENSION=" + native[1]["path"]]
    for name, args, expected in [
        ("private", ["--private-control", "--output", str(output)], 0),
        ("replay", ["--cold-replay", str(output)], 0),
        ("consume", ["--consume", str(output)], 0),
        ("date", ["--date", "19000101"], 2),
        ("guard", ["--private-control"], 2),
        ("owned", ["--deliberate-error"], 7),
        (
            "absent",
            [
                "--private-control",
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "absent" / output.name),
            ],
            0,
        ),
    ]:
        receipt = child(name, env + argv + args, tmp_path / "cli", expected=expected, deadline=120)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert evidence.consume(output)["ready"]
    with pytest.raises(ValueError, match="not_ready"):
        evidence.consume(tmp_path / "absent" / output.name)


@pytest.mark.parametrize(
    "mutation",
    [
        "readiness",
        "publication",
        "authority",
        "task",
        "family",
        "head",
        "text",
        "tape",
        "source_hash",
    ],
)
def test_input_rejections(mutation, monkeypatch, tmp_path):
    """REQ-REPORT-8533: independent producer seals and full authority reject altered operands."""
    original = inputs.read

    def changed(ref):
        value = deepcopy(original(ref))
        if mutation == "readiness" and ref["path"].endswith(inputs.UPSTREAM):
            value["nine_input_inputs_ready_score"] = 0
        if mutation == "family" and ref["path"].endswith(inputs.UPSTREAM):
            value["historical_family_receipts"][0]["qualified"] = False
        if mutation == "head" and ref["path"].endswith("frozen_heads.json"):
            value["heads"]["spline9"][0]["seed"] = 99
        if mutation == "tape" and ref["path"].endswith("feature_panel.json"):
            value["rows"][0]["unit_id"] = value["rows"][1]["unit_id"]
        return value

    monkeypatch.setattr(inputs, "read", changed)
    if mutation == "publication":
        monkeypatch.setattr(inputs, "read_bound_sidecar", lambda *a: {})
    if mutation in {"authority", "task"}:
        original_authority = inputs.methods.authority

        def authority(*args):
            value = original_authority(*args)
            if mutation == "authority":
                value["activated"] = False
            else:
                next(t for t in value["tasks"] if t["id"] == evidence.TASK)["title"] = "changed"
            return value

        monkeypatch.setattr(inputs.methods, "authority", authority)
    if mutation in {"text", "source_hash"}:
        hasher = inputs.sha256_file
        monkeypatch.setattr(
            inputs,
            "sha256_file",
            lambda p: (
                "invalid"
                if mutation == "source_hash" or str(p).endswith("human_judge_release_master.csv")
                else hasher(p)
            ),
        )
    with pytest.raises(ValueError):
        inputs.operands(inputs.ROOT, tmp_path)


def test_validation_collection_and_coverage_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8533-REPLAY: require terminal results for the exact disjoint union."""

    def recorded(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        stdout = logs / (name + ".stdout")
        text = ""
        if name.startswith("collect_"):
            text = argv[-1] + "::test_control\n"
        if name.startswith("shard_"):
            scratch = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--basetemp=")))
            assert not scratch.is_relative_to(evidence.ROOT)
            text = next(a for a in argv if "::" in a) + " PASSED\n"
        if name == "llvm_export":
            text = json.dumps(
                dict(
                    data=[
                        dict(
                            files=[
                                dict(
                                    filename="core/src/nine_input_8533.rs",
                                    summary=dict(lines=dict(percent=100)),
                                ),
                                dict(
                                    filename="binding/src/nine_input_8533.rs",
                                    summary=dict(lines=dict(percent=100)),
                                ),
                            ]
                        )
                    ]
                )
            )
        stdout.write_text(text)
        return dict(name=name, passed=True, argv=argv, stdout_path=str(stdout))

    monkeypatch.setattr(validation, "child", recorded)
    work = dict(extension=dict(path="/tmp/test.so"))
    receipts = validation.validate(work, tmp_path)
    assert all(r["passed"] for r in receipts)
    assert json.loads((tmp_path / "node_union.json").read_bytes())["passed"]
    assert all(r["passed"] for r in validation.rust_coverage(work, tmp_path))
    monkeypatch.setattr(validation, "child", lambda *a, **kw: dict(passed=False))
    assert not validation.rust_coverage(work, tmp_path)[-1]["passed"]


def test_independent_primitive_tampering(native, tmp_path):
    """SCENARIO-REPORT-8533-REPLAY: repaired primitive and envelope hashes cannot change facts."""
    raw = tmp_path / "raw"
    baseline = evidence.measure(inputs.ROOT, raw)
    receipt = child(
        "log_binding", [sys.executable, "-c", "print('real receipt')"], tmp_path / "logs"
    )
    candidate = tmp_path / "candidate.json"
    tape_bytes = (raw / "request_tape.json").read_bytes()
    for mutation in ["checksum", "code", "tape", "binary", "rows", "response", "receipt"]:
        work = deepcopy(baseline)
        (raw / "request_tape.json").write_bytes(tape_bytes)
        if mutation == "code":
            work["code_refs"][0]["sha256"] = "invalid"
        if mutation == "tape":
            tape = json.loads(tape_bytes)
            tape["rows"][0]["question"] += " changed"
            atomic_json(raw / "request_tape.json", tape)
            work["request_tape_reference"] = inputs.reference(raw / "request_tape.json")
        if mutation == "binary":
            work["extension"]["sha256"] = "invalid"
        if mutation == "rows":
            work["rows"][0]["native_residual"] += 0.1
        if mutation == "response":
            work["timing_consumer_rows"][0]["response"] = "{}"
        atomic_json(raw / "measurement.json", work)
        value = evidence.artifact(work, [dict(receipt)], candidate)
        if mutation == "receipt":
            value["validation_receipts"][0]["stdout_sha256"] = "invalid"
            value["reproducibility_checksum"] = evidence.checksum(value)
        if mutation == "checksum":
            value["reproducibility_checksum"] = "invalid"
        atomic_json(candidate, value)
        assert not evidence.replay(candidate), mutation


def test_complete_consumer_mismatch_and_malformed_acceptance(native, monkeypatch, tmp_path):
    """REQ-VERIFY-8533: owned disagreement and an evaluator accepting invalid rows fail closed."""
    with monkeypatch.context() as m:
        m.setattr(
            evidence.numeric,
            "request",
            lambda h, r, provider: json.dumps(
                dict(decision="accept" if provider is None else "reject")
            ),
        )
        work = evidence.measure(inputs.ROOT, tmp_path / "mismatch")
        assert work["owned_failure"]
    accepting = SimpleNamespace(nine_input_residual_8533=lambda c, x: [0.0] * len(x))
    rows = n.qualify(head(), np.asarray([[0.4] * 9]), accepting)
    assert any(not r["passed"] for r in rows)


def test_historical_adapter_and_production_route(native, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8533-REPLAY: original authority reaches original CLI children."""
    commands = []
    with monkeypatch.context() as m:
        m.setattr(validation.subprocess, "Popen", lambda argv, *a, **kw: commands.append(argv))

        def pytest_main(argv):
            validation.subprocess.Popen([sys.executable, "-c", "pass"])
            validation.subprocess.Popen(
                [
                    sys.executable,
                    str(
                        evidence.ROOT
                        / "scripts/experiments/experiment_8431_v726_criterion_energy_fit.py"
                    ),
                    "--deliberate-error",
                ]
            )
            return 0

        m.setattr(pytest, "main", pytest_main)
        assert runner.main(["--historical-tests", "-q"]) == 0
        assert any("--historical-exec" in command for command in commands)
    for expression in ["pass", "raise SystemExit(7)", "raise SystemExit('reason')"]:
        script = tmp_path / "original_cli.py"
        script.write_text(expression)
        assert runner.main(["--historical-exec", str(script)]) == (
            7 if expression == "raise SystemExit(7)" else 0
        )
    with monkeypatch.context() as m:
        m.setattr(validation, "validate", lambda *a: [dict(name="unit_control", passed=True)])
        m.setattr(
            validation, "rust_coverage", lambda *a: [dict(name="unit_rust_control", passed=True)]
        )
        assert runner.run(inputs.ROOT, tmp_path / (evidence.NAME + ".json"), False) == 0
