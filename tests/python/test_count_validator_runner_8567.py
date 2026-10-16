"""REQ-REPORT-8567 and REQ-VERIFY-8567: real publication and fresh replay."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from carnot.reporting import count_validator_8567 as study
from carnot.reporting import count_validator_runner_8567 as runner
from carnot.reporting.current_work_receipt import atomic_json


def cli(*args):
    """Execute the public entry point so private tests cover genuine child paths."""
    return subprocess.run(
        [sys.executable, "-u", str(runner.CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_private_publication_replay_and_tampering(tmp_path):
    """SCENARIO-REPORT-8567-PUBLICATION: fixture checks never claim qualification."""
    output = tmp_path / (study.NAME + ".json")
    result = cli("--date", "20261016", "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["count_validator_ready_score"] == 0
    assert cli("--cold-replay", output).returncode == 0
    for key, bad in [
        ("count_validator_ready_score", 1),
        ("rows", []),
        ("required_checks_passed", True),
        ("old_new_classification_rows", []),
    ]:
        changed = dict(value, **{key: bad})
        changed["reproducibility_checksum"] = study.checksum(changed)
        atomic_json(output, changed)
        assert cli("--cold-replay", output).returncode == 1
    work_path = Path(value["primitive_reference"]["path"])
    original_work = json.loads(work_path.read_bytes())
    changed = deepcopy(original_work)
    changed["controls"][0]["new_findings"] = []
    atomic_json(work_path, changed)
    atomic_json(output, study.build(changed, output))
    assert cli("--cold-replay", output).returncode == 1
    atomic_json(work_path, original_work)
    atomic_json(output, value)
    work_path.unlink()
    assert cli("--cold-replay", output).returncode == 1
    assert cli("--cold-replay", tmp_path / "absent").returncode == 1
    assert cli("--date", "wrong").returncode == 2
    assert cli("--fixture-output", study.ROOT / "results" / (study.NAME + ".json")).returncode == 2


def test_owned_failures_and_missing_input(tmp_path):
    """SCENARIO-VERIFY-8567-REPLAY: required failures disqualify and missing differs from zero."""
    work = runner.prepare(tmp_path / "raw")
    output = tmp_path / (study.NAME + ".json")
    work["receipts"] = [dict(name="failed", passed=False)]
    assert study.build(work, output)["verdict_class"] == "disqualified"
    work["failures"] = [dict(check="absent", observed=None)]
    work["receipts"] = []
    assert study.build(work, output)["verdict_class"] == "blocked"
    work["receipts"] = [dict(name="failed", passed=False)]
    assert study.build(work, output)["verdict_class"] == "disqualified"
    assert not study.replay(tmp_path / "missing")


def test_authentication_and_terminal_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8567: private candidate validation keeps every critical finding."""
    with monkeypatch.context() as patch:
        patch.setattr(study, "UPSTREAM", tmp_path / "missing")
        work = runner.prepare(tmp_path / "raw")
        assert work["failures"] and work["failures"][0]["observed"] is None
    monkeypatch.setattr(
        runner, "verify_artifact", lambda p: dict(flags=[dict(severity="critical")])
    )
    monkeypatch.setattr(study, "replay", lambda p: True)
    monkeypatch.setattr(runner, "child", lambda *a, **k: dict(passed=True))
    assert not runner.terminal(tmp_path / "candidate", tmp_path)["passed"]


def test_validation_manifest_and_production_route(tmp_path, monkeypatch):
    """REQ-VERIFY-8567: manifest accounting rejects lost nodes independently of child work."""
    work = runner.prepare(tmp_path / "raw")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    called = []

    def controlled(name, argv, logs, **kwargs):
        called.append(name)
        logs.mkdir(parents=True, exist_ok=True)
        out, err = logs / (name + ".stdout"), logs / (name + ".stderr")
        out.write_text(
            "\n".join([runner.TESTS[0] + "::test_one", *runner.E2E]) if name == "collection" else ""
        )
        err.write_text("")
        junit = next((a.split("=", 1)[1] for a in argv if a.startswith("--junitxml=")), None)
        if junit:
            nodes = [a for a in argv if "::" in a]
            Path(junit).write_text(
                "<testsuites><testsuite>"
                + "".join(
                    '<testcase classname="'
                    + n.split("::")[0][:-3].replace("/", ".")
                    + '" name="'
                    + n.split("::")[1]
                    + '"/>'
                    for n in nodes
                )
                + "</testsuite></testsuites>"
            )
        if name == "coverage":
            atomic_json(
                Path(work["raw"]) / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(missing_lines=0), missing_lines=[])
                        for p in [*runner.OWNED, "scripts/adversarial_verify.py"]
                    }
                ),
            )
        receipt = dict(
            name=name,
            passed=True,
            log_path=str(out),
            stdout_path=str(out),
            stderr_path=str(err),
            stdout_sha256=runner.reference(out)["sha256"],
            stderr_sha256=runner.reference(err)["sha256"],
        )
        atomic_json(out.with_suffix(".receipt.json"), receipt)
        return receipt

    monkeypatch.setattr(runner, "child", controlled)
    runner.validate(work, scratch)
    assert study.REQUIRED <= set(called)
    assert work["union_passed"] and work["coverage_complete"]
    value = study.build(work, tmp_path / (study.NAME + ".json"))
    assert value["count_validator_ready_score"] == 1 and value["verdict_class"] == "null"
    monkeypatch.setattr(runner, "validate", lambda *a: None)
    monkeypatch.setattr(runner, "terminal", lambda *a: dict(passed=True))
    assert runner.run(tmp_path / "production" / (study.NAME + ".json")) == 0
    monkeypatch.setattr(runner, "run", lambda *a, **k: 0)
    assert runner.main([]) == 0


def test_replay_receipt_and_code_mutations(tmp_path):
    """SCENARIO-VERIFY-8567-REPLAY: self-consistent checksums cannot alter sealed operands."""
    work = runner.prepare(tmp_path / "raw")
    receipt = runner.child(
        "tiny", [sys.executable, "-u", "-c", "print('real child')"], tmp_path / "logs", deadline=15
    )
    work["receipts"] = [receipt]
    output = tmp_path / (study.NAME + ".json")
    work_path = Path(work["raw"]) / "work.json"

    def seal(changed):
        atomic_json(work_path, changed)
        atomic_json(output, study.build(changed, output))

    seal(work)
    assert study.replay(output)
    changed = deepcopy(work)
    changed["code"][0]["sha256"] = "wrong"
    seal(changed)
    assert not study.replay(output)
    changed = deepcopy(work)
    changed["receipts"][0]["duration_s"] += 1
    seal(changed)
    assert not study.replay(output)
    seal(work)
    Path(receipt["stdout_path"]).write_text("tampered")
    assert not study.replay(output)


def test_original_authority_rejections(monkeypatch):
    """REQ-REPORT-8567: original hash, publication and exact source bytes stay pinned."""
    upstream, original, corpus = study.authenticate()
    for mode in ["identity", "old", "bound", "source", "frozen"]:
        with monkeypatch.context() as patch:
            real_read = study.read
            changed = deepcopy(upstream)
            if mode == "identity":
                changed["count_controls_ready_score"] = 0
            patch.setattr(
                study,
                "read",
                lambda ref: changed if ref["path"] == str(study.UPSTREAM) else real_read(ref),
            )
            if mode == "old":
                real_reference = study.reference
                patch.setattr(
                    study,
                    "reference",
                    lambda p: (
                        dict(path=str(p), sha256="wrong")
                        if p.name == "old_adversarial_verify.py"
                        else real_reference(p)
                    ),
                )
            if mode == "bound":
                patch.setattr(
                    study, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False))
                )
            if mode == "source":
                real_reference = study.reference
                patch.setattr(
                    study,
                    "reference",
                    lambda p: (
                        dict(path=str(p), sha256="wrong")
                        if str(p) == original["evidence"]["sources"][0]["path"]
                        else real_reference(p)
                    ),
                )
            if mode == "frozen":
                real_reference = study.reference
                patch.setattr(
                    study,
                    "reference",
                    lambda p: (
                        dict(path=str(p), sha256="wrong")
                        if str(p) == upstream["code_config_hashes"][0]["path"]
                        else real_reference(p)
                    ),
                )
            import pytest

            with pytest.raises(ValueError):
                study.authenticate()
