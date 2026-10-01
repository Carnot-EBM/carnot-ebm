"""Private CLI and cold reconstruction checks for REQ-REPORT-7955."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_7955_v690_response_targets as producer
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from test_experiment_7942_v689_sentence_labels import fixture


def test_fixture_publication_replay_and_actual_readers(tmp_path):
    """SCENARIO-REPORT-7955-TERMINAL: nested evidence cannot hide the primary."""
    path = tmp_path / "input.json"
    fixture(path)
    output = tmp_path / "success" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(path), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["response_targets_ready_score"] == 0
    assert producer.reconstruct(value)["class_counts"] == {"0": 32, "1": 32}
    sidecar = Path(value["terminal_validation_sidecar_path"])
    assert read_bound_sidecar(output, sidecar)
    os.utime(sidecar, ns=(output.stat().st_mtime_ns + 10**9,) * 2)
    receipt = reader_receipt(
        producer.TASK, output.parent, field="response_targets_ready_score", expected=0
    )
    assert receipt["passed"] and receipt["gate_sha256"] == sha256_file(output)
    for tamper in (False, True):
        candidate = deepcopy(value)
        if tamper:
            candidate["class_counts"]["1"] += 1
        replay = tmp_path / ("tampered.json" if tamper else "candidate.json")
        atomic_json(replay, candidate)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        prefix = [sys.executable, "-u"]
        if env.get("CARNOT_7955_COVERAGE_FILE"):
            prefix += [
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--data-file=" + env["CARNOT_7955_COVERAGE_FILE"],
                "--include=" + producer.INCLUDE,
            ]
        child = subprocess.run(
            prefix
            + [
                str(producer.ROOT / "scripts/experiments" / (producer.NAME + ".py")),
                "--cold-replay",
                str(replay),
                "--output",
                str(tmp_path / str(tamper) / (producer.NAME + ".json")),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == int(tamper), child.stdout + child.stderr
        assert ("reduction_drift" if tamper else "replay_passed") in child.stdout


def test_blocked_date_and_owned_failure(tmp_path):
    """SCENARIO-REPORT-7955-TERMINAL: missing external inputs are terminal."""
    output = tmp_path / "blocked" / (producer.NAME + ".json")
    assert producer.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    with pytest.raises(SystemExit) as error:
        producer.main(["--date", "20260930"])
    assert error.value.code == 2
    producer.apply_validation(value, [{"passed": False, "required": True}])
    assert value["verdict_class"] == "disqualified"


def test_cold_hash_and_annotation_drift(tmp_path):
    """REQ-REPORT-7955: rehashed derived targets still need original authority."""
    path = tmp_path / "input.json"
    d = fixture(path)
    value = producer.build(d, tmp_path / "raw", fixture_path=path)
    changed = deepcopy(value)
    changed["public_manifest_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash_drift"):
        producer.reconstruct(changed)
    changed = deepcopy(value)
    changed["rows"][0]["y"] = 1
    with pytest.raises(ValueError, match="reduction_drift"):
        producer.reconstruct(changed)
    assert producer.reconstruct(value)


def test_manifest_current_and_historical_dates(tmp_path):
    """SCENARIO-REPORT-7955-TERMINAL: freeze includes and each private route."""
    manifest = producer.freeze_commands(tmp_path)
    assert manifest["coverage_includes"] == producer.INCLUDE
    historical = [c for c in manifest["commands"] if c["name"].startswith("e2e016")]
    assert len(historical) == 2 and all("20260929" in c["argv"] for c in historical)
    assert manifest["current_date"] == "20261001"


def test_natural_cold_reconstruction_and_independent_union(tmp_path, monkeypatch):
    """REQ-REPORT-7955: reconstruct natural counts directly from original spans."""
    value = producer.build_live(producer.ROOT, tmp_path / "natural")
    value["response_targets_ready_score"] = 0
    assert producer.reconstruct(value)["class_counts"] == {"0": 43, "1": 19}
    raw = {r["id"]: r for r in producer.read_jsonl(producer.ROOT / "data/ragtruth/response.jsonl")}
    independent = {"0": 0, "1": 0}
    for row in value["rows"]:
        if row["role"] == "evaluation" and row["status"] == "completed":
            response = raw[row["response_id"]]
            assert response["quality"] == "good"
            y = int(len(response["labels"]) > 0)
            assert y == row["y"]
            independent[str(y)] += 1
    assert independent == value["class_counts"]
    unsafe = deepcopy(value)
    unsafe["response_targets_ready_score"] = 1
    unsafe["validation_receipts"] = [{"name": "unfinished", "passed": True}]
    producer.freeze_commands(tmp_path / "frozen")
    unsafe["validation_command_manifest_path"] = str(
        tmp_path / "frozen/validation_command_manifest.json"
    )
    with pytest.raises(ValueError, match="receipt_drift"):
        producer.reconstruct(unsafe)
    monkeypatch.setattr(producer, "authenticate", lambda root: ([{"failed": True}], {}))
    with pytest.raises(ValueError, match="cold_custody"):
        producer.reconstruct(value)
    monkeypatch.setattr(
        producer, "authenticate", lambda root: ([], {"public_shards": [], "role_counts": {}})
    )
    with pytest.raises(ValueError, match="original_role_roster"):
        producer.build_live(tmp_path, tmp_path / "wrong")


def test_reconstruction_rejects_rehashed_manifests_and_unsafe_readiness(tmp_path):
    """SCENARIO-REPORT-7955-TERMINAL: a new hash cannot authorize changed claims."""
    path = tmp_path / "input.json"
    d = fixture(path)
    value = producer.build(d, tmp_path / "raw", fixture_path=path)
    changed = deepcopy(value)
    changed["response_targets_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.reconstruct(changed)
    blocked = producer.base([])
    blocked["response_targets_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.reconstruct(blocked)
    changed = deepcopy(value)
    changed["annotation_rows"].append({"changed": True})
    with pytest.raises(ValueError, match="primitive_rows"):
        producer.reconstruct(changed)
    saved_path = Path(value["public_manifest_path"])
    saved = json.loads(saved_path.read_text())
    saved["boundaries"] = []
    atomic_json(saved_path, saved)
    value["public_manifest_sha256"] = sha256_file(saved_path)
    with pytest.raises(ValueError, match="public_drift"):
        producer.reconstruct(value)
    value = producer.build(d, tmp_path / "raw", fixture_path=path)
    saved_path = Path(value["evaluator_manifest_path"])
    producer.write_jsonl(saved_path, [])
    value["evaluator_manifest_sha256"] = sha256_file(saved_path)
    with pytest.raises(ValueError, match="evaluator_drift"):
        producer.reconstruct(value)


def test_owned_runtime_and_terminal_failure_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7955-TERMINAL: exercise the owned validation boundary."""
    path = tmp_path / "input.json"
    d = fixture(path)
    monkeypatch.setattr(
        producer.prior,
        "execute_commands",
        lambda manifest, logs: [{"passed": True, "required": True}],
    )
    assert producer.execute_commands(
        {"coverage_file": str(tmp_path / ".coverage")}, tmp_path / "logs"
    )
    monkeypatch.delenv("CARNOT_7955_COVERAGE_FILE")

    def mock_live(root, raw):
        value = producer.build(d, raw, fixture_path=path)
        atomic_json(
            raw / "coverage.json", {"files": {"fixture": {"summary": {"covered_lines": 1}}}}
        )
        return value

    monkeypatch.setattr(producer, "build_live", mock_live)
    assert producer.main(["--output", str(tmp_path / "runtime" / (producer.NAME + ".json"))]) == 0
    real_check = producer.terminal_check
    attempts = []

    def first_failure(candidate):
        attempts.append(candidate)
        if len(attempts) == 1:
            return {"passed": False, "flagged_adversarial": True}
        return real_check(candidate)

    monkeypatch.setattr(producer, "terminal_check", first_failure)
    out = tmp_path / "terminal" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(path), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(producer, "reader_receipt", lambda *args, **kwargs: {"passed": False})
    assert (
        producer.main(
            [
                "--fixture-input",
                str(path),
                "--output",
                str(tmp_path / "reader-failure" / (producer.NAME + ".json")),
            ]
        )
        == 1
    )


def test_required_receipt_hash_custody(tmp_path):
    """REQ-REPORT-7955: readiness cannot trust an altered child receipt."""
    log = tmp_path / "child.log"
    log.write_text("passed\n")
    manifest = tmp_path / "commands.json"
    atomic_json(
        manifest,
        {
            "commands": [
                {
                    "name": "owned",
                    "argv": ["exact"],
                    "expected_exit": 0,
                    "failure_reason": None,
                    "required": True,
                }
            ]
        },
    )
    value = producer.base([])
    value.update(
        validation_command_manifest_path=str(manifest),
        validation_receipts=[
            dict(
                name="owned",
                command_argv=["exact"],
                exit_code=0,
                passed=True,
                timed_out=False,
                log_path=str(log),
                log_sha256=sha256_file(log),
            )
        ],
    )
    producer.check_receipts(value)
    manifest_value = json.loads(manifest.read_text())
    manifest_value["commands"].append({"required": False})
    atomic_json(manifest, manifest_value)
    value["validation_receipts"][0]["log_path"] = os.path.relpath(log, producer.ROOT)
    producer.check_receipts(value)
    for mutation in ({"exit_code": 1}, {"command_argv": ["different"]}, {"log_sha256": "bad"}):
        changed = deepcopy(value)
        changed["validation_receipts"][0].update(mutation)
        with pytest.raises(ValueError, match="receipt"):
            producer.check_receipts(changed)
    value["validation_receipts"] = []
    with pytest.raises(ValueError, match="receipt"):
        producer.check_receipts(value)
    assert (
        producer.main(
            [
                "--fixture-input",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "missing-input" / (producer.NAME + ".json")),
            ]
        )
        == 1
    )
