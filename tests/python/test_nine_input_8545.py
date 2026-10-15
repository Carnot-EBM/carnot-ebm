"""REQ-VERIFY-8545 / REQ-REPORT-8545: real children must qualify the consumer."""

import json
import os
from pathlib import Path
import sys

import pytest

from carnot.reporting import nine_input_qualification_8545 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module", autouse=True)
def native_environment(tmp_path_factory):
    """REQ-VERIFY-8545: tests share real compiled bytes, never a Python substitute."""
    previous = os.environ.get("CARNOT_8533_EXTENSION")
    if not previous:
        private = tmp_path_factory.mktemp("native-8545")
        _, ref = e.legacy.build.extension(private, private / "logs")
        os.environ["CARNOT_8533_EXTENSION"] = ref["path"]
    yield
    if previous is None:
        os.environ.pop("CARNOT_8533_EXTENSION", None)
    else:
        os.environ["CARNOT_8533_EXTENSION"] = previous


def test_original_authority_and_complete_requests(tmp_path):
    """SCENARIO-VERIFY-8545-PARITY: original operands reach real serialized decisions."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert not work["failures"]
    assert len(work["complete_request_rows"]) == 1800
    assert all(r["passed"] for r in work["complete_request_rows"])
    assert sum(r.get("boundary_fallback", False) for r in work["rows"]) >= 6
    value = e.artifact(work, [], tmp_path / (e.NAME + ".json"))
    assert value["nine_input_native_ready_score"] == 0
    assert value["inference_substrate"] == "aggregation_from_upstream_artifacts"
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    value["complete_request_rows"][0]["response"] = "{}"
    value["reproducibility_checksum"] = e.legacy.checksum(value)
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    assert not e.replay(tmp_path / "absent")


def test_request_cli_and_rejection_contract(tmp_path):
    """SCENARIO-REPORT-8545-TERMINAL: full CLI accepts text and rejects malformed files."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    bundle = tmp_path / "bundle.json"
    atomic_json(
        bundle,
        dict(
            head=work["head_reference"],
            tape=work["request_tape_reference"],
            extension=work["extension"],
        ),
    )
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        (
            "request",
            ["--request-bundle", str(bundle), "--request-output", str(tmp_path / "responses.json")],
            0,
        ),
        (
            "bad_request",
            [
                "--request-bundle",
                str(tmp_path / "absent"),
                "--request-output",
                str(tmp_path / "bad.json"),
            ],
            1,
        ),
        ("missing_output", ["--request-bundle", str(bundle)], 2),
        ("bad_date", ["--date", "19000101"], 2),
        ("owned", ["--deliberate-error"], 7),
        ("missing_replay", ["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        receipt = child(name, argv + args, tmp_path / "cli", deadline=120, expected=expected)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    responses = json.loads((tmp_path / "responses.json").read_bytes())
    assert len(responses["rows"]) == 1800 and all(r["passed"] for r in responses["rows"])


def test_external_absence_and_failed_validation(tmp_path):
    """REQ-REPORT-8545: missing differs from zero; failed owned checks disqualify."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw")
    value = e.artifact(work, [], tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"]
    work["owned_failure"] = True
    assert e.artifact(work, [], tmp_path / "out.json")["verdict_class"] == "disqualified"
    with pytest.raises(ValueError, match="not_ready"):
        candidate = tmp_path / "blocked.json"
        atomic_json(candidate, value)
        e.consume(candidate)


def test_authority_and_methods_rejections(tmp_path, monkeypatch):
    """REQ-REPORT-8545: independent seals reject changed historical or current inputs."""
    original = e.inputs.reference
    with monkeypatch.context() as m:
        m.setattr(e.inputs, "reference", lambda p: dict(original(p), sha256="changed"))
        with pytest.raises(ValueError, match="authority_hash"):
            e.original_authority()
    original_read = e.inputs.read
    with monkeypatch.context() as m:
        m.setattr(
            e.inputs,
            "read",
            lambda ref: (
                dict(original_read(ref), methods_ready_score=0)
                if ref["path"].endswith(e.CURRENT)
                else original_read(ref)
            ),
        )
        work = e.measure(e.ROOT, tmp_path / "bad_gate")
        assert any(r["check"] == "methods_and_native_inputs" for r in work["failures"])


def test_replay_and_complete_request_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8545-TERMINAL: repaired hashes cannot change primitive responses."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    output = tmp_path / (e.NAME + ".json")
    for mutation in ["checksum", "legacy", "request", "primitive"]:
        from copy import deepcopy

        changed = deepcopy(work)
        if mutation == "legacy":
            changed["rows"][0]["native_residual"] += 1
        if mutation == "request":
            changed["complete_request_rows"][0]["response"] = "{}"
        if mutation == "primitive":
            atomic_json(Path(changed["complete_requests_reference"]["path"]), dict(rows=[]))
            changed["complete_requests_reference"] = e.inputs.reference(
                Path(changed["complete_requests_reference"]["path"])
            )
        atomic_json(Path(changed["raw"]) / "measurement.json", changed)
        value = e.artifact(changed, [], output)
        if mutation == "checksum":
            value["reproducibility_checksum"] = "wrong"
        atomic_json(output, value)
        assert not e.replay(output), mutation


def test_historical_adapter_and_validation_paths(tmp_path, monkeypatch):
    """REQ-VERIFY-8545: adapters retain original commands and basetemp parents."""
    from carnot.reporting import nine_input_validation_8545 as v

    commands = []
    with monkeypatch.context() as m:
        m.setattr(v.subprocess, "Popen", lambda argv, *a, **kw: commands.append(argv))

        def tests(argv):
            v.subprocess.Popen([sys.executable, str(e.ROOT / e.legacy.CLI), "--deliberate-error"])
            v.subprocess.Popen([sys.executable, "-c", "pass"])
            return 0

        m.setattr(pytest, "main", tests)
        assert (
            v.historical(e.legacy.TEST, ["--basetemp=" + str(tmp_path / "nested" / "pytest")]) == 0
        )
        assert (tmp_path / "nested").is_dir()
        assert any("--sealed-exec" in c for c in commands)
    for text, code in [("pass", 0), ("raise SystemExit(7)", 7), ("raise SystemExit('reason')", 0)]:
        script = tmp_path / "script.py"
        script.write_text(text)
        assert v.historical("fixtures", [str(script)], script=True) == code


def test_runner_private_and_normal_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8545-TERMINAL: publication handles absent and owned outcomes."""
    from carnot.reporting import nine_input_runner_8545 as r
    from carnot.reporting import nine_input_validation_8545 as v

    assert (
        r.main(
            [
                "--private-control",
                "--root",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path / "blocked" / (e.NAME + ".json")),
            ]
        )
        == 0
    )
    with pytest.raises(SystemExit) as error:
        r.main(["--private-control"])
    assert error.value.code == 2
    with monkeypatch.context() as m:
        m.setattr(v, "historical", lambda *a, **kw: 7)
        assert r.main(["--sealed-tests", "fixture", "-q"]) == 7
    with monkeypatch.context() as m:
        m.setattr(v, "reproduce", lambda *a: [dict(name="fixture_reproduction", passed=True)])
        m.setattr(v, "validate", lambda *a: [dict(name="fixture", passed=True)])
        m.setattr(v, "finish", lambda *a: [dict(name="fixture_finish", passed=True)])
        assert r.run(e.ROOT, tmp_path / "normal" / (e.NAME + ".json"), False) == 0


def test_measured_validation_orchestration(tmp_path, monkeypatch):
    """REQ-VERIFY-8545: union and coverage gates reject missing terminal outcomes."""
    from carnot.reporting import nine_input_validation_8545 as v

    def recorded(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        stdout = logs / (name + ".stdout")
        text = ""
        if name.startswith("collect_"):
            text = argv[-1] + "::test_control\n"
        if name.startswith("shard_"):
            base = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--basetemp=")))
            assert base.parent.is_dir() and not base.is_relative_to(e.ROOT)
            text = next(a for a in argv if "::" in a) + " PASSED\n"
        if name == "coverage_json":
            atomic_json(Path(argv[-1]), dict(totals=dict(percent_covered=100)))
        stdout.write_text(text)
        return dict(name=name, passed=True, argv=argv, stdout_path=str(stdout))

    monkeypatch.setattr(v, "child", recorded)

    def rust(work, raw):
        atomic_json(raw / "rust_coverage_summary.json", dict(passed=True))
        return [dict(name="owned_rust_coverage_100", passed=True)]

    monkeypatch.setattr(v.old, "rust_coverage", rust)
    work = dict(extension=dict(path="/var/tmp/test.so"))
    assert all(r["passed"] for r in v.validate(work, tmp_path))
    assert work["exact_union_receipt"]["passed"]
    assert all(r["passed"] for r in v.finish(work, tmp_path))


def test_request_extension_hash_and_consumer_contract(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8545-TERMINAL: an altered extension cannot enter the request CLI."""
    from carnot.reporting import nine_input_runner_8545 as r

    binary = tmp_path / "extension.so"
    binary.write_bytes(b"invalid")
    bundle = tmp_path / "bundle.json"
    atomic_json(bundle, dict(extension=dict(path=str(binary), sha256="changed")))
    assert r.request_bundle(bundle, tmp_path / "responses.json") == 1
    value = dict(
        nine_input_native_ready_score=1,
        required_checks_passed=True,
        extension_reference={},
        request_tape_reference={},
        head_reference={},
        measurement_kernel_reference={},
    )
    candidate = tmp_path / "ready.json"
    atomic_json(candidate, value)
    monkeypatch.setattr(e, "replay", lambda p: True)
    assert e.consume(candidate)["ready"]
    assert r.main(["--consume", str(candidate)]) == 0


def test_original_failure_reproduction_receipt(tmp_path, monkeypatch):
    """REQ-VERIFY-8545: expected old failures cannot be confused with qualified passes."""
    from carnot.reporting import nine_input_validation_8545 as v

    def recorded(name, argv, logs, **kw):
        assert kw["expected"] == 1
        assert "CARNOT_8533_EXTENSION=/var/tmp/real.so" in argv
        logs.mkdir(parents=True, exist_ok=True)
        output = logs / "out"
        output.write_text(
            "\n".join("FAILED " + a for a in argv if "::" in a)
            + "\nprivate control output must be outside repository"
        )
        return dict(name=name, passed=True, stdout_path=str(output))

    monkeypatch.setattr(v, "child", recorded)
    work = dict(extension=dict(path="/var/tmp/real.so"))
    assert v.reproduce(work, tmp_path)[0]["passed"]
    assert work["original_failure_reproduction"]
