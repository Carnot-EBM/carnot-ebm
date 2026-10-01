"""Private publication and replay checks for REQ-REPORT-7968."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_7968_v691_response_role_targets as producer
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from test_response_role_targets_7968 import fixture


def test_fixture_publication_replay_capture_and_actual_readers(tmp_path):
    """SCENARIO-REPORT-7968-TERMINAL: run real private CLI and exact readers."""
    path = tmp_path / "input.json"
    fixture(path)
    output = tmp_path / "success" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(path), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["response_roles_ready_score"] == 0
    assert value["verdict_class"] == "circular_positive"
    assert producer.reconstruct(value)["role_counts"]["evaluation"] == 64
    sidecar = Path(value["terminal_validation_sidecar_path"])
    assert read_bound_sidecar(output, sidecar)
    os.utime(sidecar, ns=(output.stat().st_mtime_ns + 10**9,) * 2)
    receipt = reader_receipt(
        producer.TASK, output.parent, field="response_roles_ready_score", expected=0
    )
    assert receipt["passed"] and receipt["gate_sha256"] == sha256_file(output)
    public = value["public_role_manifests"]["fit"]
    assert (
        producer.main(
            [
                "--capture-input",
                public["path"],
                "--role",
                "fit",
                "--expected-sha256",
                public["sha256"],
            ]
        )
        == 0
    )
    for tamper in (False, True):
        candidate = deepcopy(value)
        if tamper:
            candidate["class_counts_by_role"]["fit"]["1"] += 1
        replay = tmp_path / f"replay-{tamper}.json"
        atomic_json(replay, candidate)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        prefix = [sys.executable, "-u"]
        if env.get("CARNOT_7968_COVERAGE_FILE"):
            prefix += [
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--data-file=" + env["CARNOT_7968_COVERAGE_FILE"],
                "--include=" + producer.INCLUDE,
            ]
        child = subprocess.run(
            prefix + [str(producer.ROOT / producer.OWNED[-1]), "--cold-replay", str(replay)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == int(tamper), child.stdout + child.stderr


def test_blocked_date_access_and_hash_drift(tmp_path):
    """SCENARIO-REPORT-7968-TERMINAL: external absence blocks; drift rejects."""
    output = tmp_path / "blocked" / (producer.NAME + ".json")
    assert producer.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    with pytest.raises(SystemExit):
        producer.main(["--date", "20260930"])
    path = tmp_path / "input.json"
    data = fixture(path)
    value = producer.build(data, tmp_path / "raw", fixture_path=path)
    changed = deepcopy(value)
    changed["rows"][0]["y"] = 1
    with pytest.raises(ValueError, match="union_drift"):
        producer.reconstruct(changed)
    changed = deepcopy(value)
    changed["response_roles_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.reconstruct(changed)
    item = value["public_role_manifests"]["fit"]
    atomic_json(Path(item["path"]), {"role": "fit", "human_label": 0})
    assert (
        producer.main(
            [
                "--capture-input",
                item["path"],
                "--role",
                "fit",
                "--expected-sha256",
                sha256_file(Path(item["path"])),
            ]
        )
        == 1
    )
    assert (
        producer.main(
            ["--capture-input", item["path"], "--role", "fit", "--expected-sha256", "bad"]
        )
        == 1
    )
    producer.apply_validation(value, [{"passed": False, "required": True}])
    assert value["verdict_class"] == "disqualified"


def test_natural_union_preserves_evaluation_and_command_scope(tmp_path):
    """REQ-REPORT-7968: preserve original labels and upstream invocation dates."""
    value = producer.build_live(producer.ROOT, tmp_path / "natural")
    historical = json.loads((producer.ROOT / producer.HISTORY).read_text())
    assert value["rows"] == historical["response_union_rows"]
    assert value["role_counts"]["evaluation"] == 64
    assert value["class_counts_by_role"]["evaluation"] == {"0": 43, "1": 19, "unknown": 2}
    assert producer.reconstruct(value)["response_roles_ready_score"] == 1
    with producer.TemporaryDirectory(prefix="carnot-7968-test-") as scratch:
        manifest = producer.freeze_commands(tmp_path / "commands", Path(scratch))
        assert manifest["coverage_includes"] == producer.INCLUDE
        assert manifest["scratch_root"] == scratch


def test_rehashed_views_blocked_readiness_and_live_drift(tmp_path, monkeypatch):
    """REQ-REPORT-7968: a fresh digest cannot authorize changed source evidence."""
    path = tmp_path / "input.json"
    data = fixture(path)
    value = producer.build(data, tmp_path / "raw", fixture_path=path)
    with pytest.raises(ValueError, match="original_union_drift"):
        producer.build(data, tmp_path / "wrong", original_rows=[])
    blocked = producer.base([])
    blocked["response_roles_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.reconstruct(blocked)
    item = value["evaluator_role_manifests"]["fit"]
    saved = json.loads(Path(item["path"]).read_text())
    saved["rows"] = []
    atomic_json(Path(item["path"]), saved)
    item["sha256"] = sha256_file(Path(item["path"]))
    with pytest.raises(ValueError, match="role_view_drift"):
        producer.reconstruct(value)
    natural = producer.build_live(producer.ROOT, tmp_path / "natural")
    changed = deepcopy(natural)
    changed["rows"][0]["y"] = 1
    with pytest.raises(ValueError, match="original_union_drift"):
        producer.reconstruct(changed)
    monkeypatch.setattr(producer, "authenticate", lambda root: ([{"failed": True}], {}))
    with pytest.raises(ValueError, match="cold_custody"):
        producer.reconstruct(natural)


def test_runtime_validation_and_terminal_recheck(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7968-TERMINAL: owned failure must zero readiness."""
    path = tmp_path / "input.json"
    data = fixture(path)

    def live(root, raw):
        return producer.build(data, raw, fixture_path=path)

    def children(manifest, logs):
        scratch = Path(manifest["scratch_root"])
        atomic_json(
            scratch / "routes/coverage.json",
            {"files": {"owned": {"summary": {"covered_lines": 1, "missing_lines": 0}}}},
        )
        (scratch / ".coverage").write_bytes(b"fixture")
        return [{"passed": True, "required": True}]

    monkeypatch.setattr(producer, "build_live", live)
    monkeypatch.setattr(producer.prior, "execute_commands", children)
    output = tmp_path / "runtime" / (producer.NAME + ".json")
    assert producer.main(["--output", str(output)]) == 0
    real_check = producer.terminal_check
    attempts = []

    def first_failure(candidate):
        attempts.append(candidate)
        if len(attempts) == 1:
            return {"passed": False, "flagged_adversarial": True}
        return real_check(candidate)

    monkeypatch.setattr(producer, "terminal_check", first_failure)
    output = tmp_path / "terminal" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(producer, "reader_receipt", lambda *args, **kwargs: {"passed": False})
    assert (
        producer.main(
            [
                "--fixture-input",
                str(path),
                "--output",
                str(tmp_path / "reader" / (producer.NAME + ".json")),
            ]
        )
        == 1
    )
    monkeypatch.delenv("CARNOT_7968_COVERAGE_FILE", raising=False)
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)


def test_cli_label_access_and_missing_fixture(tmp_path):
    """SCENARIO-VERIFY-7968-ACCESS: real label CLI enforces fitting isolation."""
    path = tmp_path / "input.json"
    data = fixture(path)
    value = producer.build(data, tmp_path / "raw", fixture_path=path)
    for role, expected in [("fit", 0), ("evaluation", 1)]:
        item = value["evaluator_role_manifests"][role]
        assert (
            producer.main(
                ["--label-input", item["path"], "--expected-sha256", item["sha256"], "--role", role]
            )
            == expected
        )
    seal = tmp_path / "seal.json"
    atomic_json(seal, {"sealed": True})
    seals = tmp_path / "seals.json"
    atomic_json(seals, {key: producer.reference(seal) for key in ("heads", "policies")})
    item = value["evaluator_role_manifests"]["evaluation"]
    assert (
        producer.main(
            [
                "--label-input",
                item["path"],
                "--expected-sha256",
                item["sha256"],
                "--role",
                "evaluation",
                "--purpose",
                "evaluation",
                "--seals",
                str(seals),
            ]
        )
        == 0
    )
    assert (
        producer.main(
            [
                "--fixture-input",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "missing-output" / (producer.NAME + ".json")),
            ]
        )
        == 1
    )


def test_required_receipts_and_empty_coverage(tmp_path, monkeypatch):
    """REQ-REPORT-7968: empty coverage and altered validation receipts fail."""
    natural = producer.build_live(producer.ROOT, tmp_path / "natural")
    natural["validation_receipts"] = [{"passed": True}]
    monkeypatch.setattr(producer.prior, "check_receipts", lambda value: None)
    assert producer.reconstruct(natural)
    path = tmp_path / "input.json"
    data = fixture(path)
    monkeypatch.setattr(
        producer, "build_live", lambda root, raw: producer.build(data, raw, fixture_path=path)
    )

    def children(manifest, logs):
        atomic_json(Path(manifest["scratch_root"]) / "routes/coverage.json", {"files": {}})
        return [{"passed": True, "required": True}]

    monkeypatch.setattr(producer.prior, "execute_commands", children)
    output = tmp_path / "coverage-failure" / (producer.NAME + ".json")
    assert producer.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["response_roles_ready_score"] == 0
    monkeypatch.delenv("CARNOT_7968_COVERAGE_FILE", raising=False)
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)


def test_external_reconstruction_failure_is_terminal_blocked(monkeypatch):
    """REQ-REPORT-7968: external custody drift is a complete blocked operand."""

    def broken(value):
        raise ValueError("hash_drift:external")

    monkeypatch.setattr(producer.prior, "reconstruct", broken)
    failures, _ = producer.authenticate(producer.ROOT)
    assert failures[0]["field"] == "reconstruction_custody"
    assert failures[0]["observed"] == "hash_drift:external"
