"""REQ-REPORT-8276 / REQ-VERIFY-8276: qualify current thin execution bindings."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import os

import pytest
import yaml

from carnot.reporting import current_contract_readiness_8276 as q
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v709_execution import child
from carnot.reporting.current_work_receipt import sha256_file
from unittest.mock import patch


def authorities(tmp_path):
    """Keep private authority variants outside the scientific artifact store."""
    root = tmp_path / "repo"
    (root / Path(q.DESIGN).parent).mkdir(parents=True)
    (root / q.DESIGN).write_bytes((q.ROOT / q.DESIGN).read_bytes())
    active = yaml.safe_load((q.ROOT / q.ACTIVE).read_bytes())
    for name in [q.ACTIVE, q.STAGED]:
        (root / name).write_text(yaml.safe_dump(active))
    (root / q.PROTOCOL).write_bytes((q.ROOT / q.PROTOCOL).read_bytes())
    return root


def checks():
    """These declared unit doubles test reduction; production requires real custody."""
    return [
        dict(name=name, passed=True, exit_code=0)
        for name in [
            "current_contract_tests",
            "coverage_custody_tests",
            "view_component",
            "admission_component",
        ]
    ]


def test_authority_and_gate_binding(tmp_path):
    """SCENARIO-REPORT-8276-AUTHORITY: all fourteen complete tasks bind in order."""
    root = authorities(tmp_path)
    work = q.authority_work(root, tmp_path / "raw")
    assert work["contract"]["activated"] and work["contract"]["planning_matched"]
    assert len(work["tasks"]) == 14
    assert work["tasks"][0]["id"] == q.TASK
    assert work["tasks"][-1]["id"] == "exp8289-capstone"
    assert tasks_digest(work["tasks"]) == work["contract"]["canonical_tasks_sha256"]
    paths = {t["id"]: t["deliverable"] for t in work["tasks"]}
    for producer in work["execution_contract"]["producers"]:
        for gate in producer["current_gate_fields"]:
            assert gate["artifact_path"] == paths[gate["upstream"]]
    active = yaml.safe_load((root / q.ACTIVE).read_bytes())
    active["tasks"][0]["prompt"] += " changed"
    (root / q.ACTIVE).write_text(yaml.safe_dump(active))
    assert not q.authority_work(root, tmp_path / "bad")["contract"]["activated"]
    (root / q.ACTIVE).unlink()
    staged = q.authority_work(root, tmp_path / "staged")
    assert staged["contract"]["planning_matched"] and not staged["contract"]["activated"]


def test_external_and_owned_component_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8276-COMPONENTS: component failures do not borrow scores."""
    work = q.measure(tmp_path, tmp_path / "raw")
    assert work["failures"] and not work["protocol_work"]["evidence"]
    result = q.reduce(work, checks(), True, True)
    assert result["verdict_class"] == "blocked"
    assert result["coverage_custody_ready_score"] == 0
    assert result["gate_check_summary"][0]["observed"] is not True
    bad = checks()
    bad[-1].update(passed=False, exit_code=1)
    assert q.reduce(work, bad, True, True)["verdict_class"] == "disqualified"
    (tmp_path / q.DESIGN).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / q.DESIGN).write_text("invalid design")
    assert not q.authority_work(tmp_path, tmp_path / "invalid")["contract"]["activated"]


def test_current_thin_crash_and_capture(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8276-COMPONENTS: unchanged controls use the current CLI."""
    from test_protocol_conformance_8263 import test_hard_exit_pending_resume, test_capture_real_peer

    monkeypatch.setattr(q.protocol, "CLI", q.CLI)
    test_hard_exit_pending_resume(tmp_path)
    test_capture_real_peer(tmp_path)


def cli(tmp_path, *args):
    """SCENARIO-REPORT-8276-REPLAY: execute the actual thin script outside the repo."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(q.ROOT / ".venv/bin/python"), "-u", str(q.ROOT / q.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_and_frozen_manifest(tmp_path):
    """REQ-VERIFY-8276: current commands freeze exact JSON custody and child coverage."""
    plan = q.manifest(tmp_path, tmp_path / "candidate.json")
    assert {"view_component", "admission_component", "consumer_E2E015_019"} <= {
        r["name"] for r in plan["commands"]
    }
    assert "patch=subprocess, _exit" in (tmp_path / "coverage.ini").read_text()
    assert cli(tmp_path, "--date", "20261007").returncode == 2
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    assert cli(tmp_path, "--coverage-replay", tmp_path / "absent").returncode == 1


@pytest.fixture(scope="module")
def actual_work(tmp_path_factory):
    """REQ-VERIFY-8276: one actual embedded vocabulary run supplies reusable primitives."""
    raw = tmp_path_factory.mktemp("current8276")
    work = q.measure(q.ROOT, raw)
    work.update(runtime_checks=[], invocation_argv=[q.CLI, "--date", "20261008"])
    atomic_json(raw / "measurement.json", work)
    atomic_json(raw / "validation_commands.json", {})
    atomic_json(raw / "validation_receipts.json", {})
    return work, raw


def test_actual_components_and_independent_failures(actual_work):
    """SCENARIO-VERIFY-8276-COMPONENTS: real current primitives qualify each component."""
    work, raw = actual_work
    assert not work["failures"]
    assert work["protocol_work"]["tokenizer"]["neural_weights_loaded"] is False
    assert len(work["source_role_manifest"]["fit"]) == 128
    missing = [r for r in work["history"] if r["disposition"] == "cascade_skip_absent_primary"]
    assert len(missing) == 7 and all("honest_verdict" not in r for r in missing)
    result = q.reduce(work, checks(), True, True)
    scores = [
        "current_contract_ready_score",
        "coverage_custody_ready_score",
        "view_kernel_ready_score",
        "admission_kernel_ready_score",
    ]
    assert [result[k] for k in scores] == [1, 1, 1, 1]
    assert result["intended_count"] == 14 + 5440 + 12 + 13
    assert result["censored_count"] == 7
    assert result["verdict_class"] == "circular_positive"
    bad = checks()
    bad[-1].update(passed=False, exit_code=1)
    result = q.reduce(work, bad, True, True)
    assert result["verdict_class"] == "disqualified"
    assert [result[k] for k in scores] == [1, 1, 1, 0]
    broken = deepcopy(work)
    broken["protocol_work"]["owned_failure"] = "hard_exit_resume_drift"
    assert q.reduce(broken, checks(), True, True)["verdict_class"] == "disqualified"
    broken = deepcopy(work)
    broken["failures"] = [
        dict(component="view", artifact_field="embedded_vocabulary_available", observed=None)
    ]
    result = q.reduce(broken, checks(), True, True)
    assert [result[k] for k in scores] == [1, 1, 0, 1]


def test_terminal_negative_fields(tmp_path):
    """SCENARIO-REPORT-8276-AUTHORITY: bound sidecars cannot hide exact failed flags."""
    primary = tmp_path / "results/experiment_8262_private.json"
    terminal = primary.parent / "raw" / primary.stem / "terminal.json"
    sidecar = terminal.parent / "sidecar.json"
    atomic_json(
        primary,
        dict(
            terminal_validation_sidecar_path=str(terminal),
            required_checks_passed=False,
            flagged_adversarial=True,
        ),
    )
    atomic_json(sidecar, dict(primary_sha256=sha256_file(primary), report=dict(passed=False)))
    atomic_json(terminal, dict(publication=dict(primary_sha256="wrong", sidecar_path=str(sidecar))))
    work = dict(refs=[], failures=[])
    assert q.bind_terminal(
        tmp_path, str(primary.relative_to(tmp_path)), work, tmp_path / "raw", "coverage"
    )
    assert {r["artifact_field"] for r in work["failures"]} == {
        "required_checks_passed",
        "flagged_adversarial",
        "report.passed",
        "publication.primary_sha256",
    }
    primary2 = primary.with_name("experiment_8263_private.json")
    sidecar2 = primary2.parent / "raw" / primary2.stem / "sidecar.json"
    atomic_json(
        primary2,
        dict(
            terminal_validation_sidecar_path=str(sidecar2),
            required_checks_passed=True,
            flagged_adversarial=False,
        ),
    )
    atomic_json(sidecar2, dict(primary_sha256=sha256_file(primary2), report=dict(passed=True)))
    assert q.bind_terminal(
        tmp_path,
        str(primary2.relative_to(tmp_path)),
        dict(refs=[], failures=[]),
        tmp_path / "raw2",
        "protocol",
    )


def candidate(actual_work, tmp_path):
    """SCENARIO-REPORT-8276-REPLAY: incomplete validation remains honestly disqualified."""
    work, raw = actual_work
    receipt = child(
        "private_current",
        [str(q.ROOT / ".venv/bin/python"), "-c", "print('private')"],
        tmp_path / "logs",
    )
    with patch.object(q.runner, "q", q):
        value = q.build(work, raw, [receipt], {}, [], tmp_path / "deleted", {})
    path = tmp_path / (q.NAME + ".json")
    atomic_json(path, value)
    return path, value


def test_frozen_replay_and_rehashed_negatives(actual_work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8276-REPLAY: replay derives fields even when summaries are rehashed."""
    path, value = candidate(actual_work, tmp_path)
    assert q.replay(path)["passed"]
    assert cli(tmp_path, "--cold-replay", path).returncode == 0
    for key, change in [
        ("reproducibility_checksum", "bad"),
        ("experiment_id", 8277),
        ("current_contract_ready_score", 1),
        ("scratch_removed", False),
        ("model_invocation_counts", {}),
        ("source_artifact_hashes", [dict(path="absent", sha256="bad")]),
    ]:
        bad = deepcopy(value)
        bad[key] = change
        if key != "reproducibility_checksum":
            bad.pop("reproducibility_checksum")
            bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        with pytest.raises((ValueError, OSError)):
            q.replay(path)
    atomic_json(path, value)
    with patch.object(q.runner, "q", q):
        assert q.runner.terminal(path, tmp_path / "terminal")["passed"]
    monkeypatch.setattr(q.runner, "run", lambda root, output: {})
    assert q.main(["--root", str(tmp_path), "--output", str(path)]) == 0


def test_rehashed_owned_primitives(actual_work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8276-REPLAY: independent reconstructions reject rewritten work."""
    path, value = candidate(actual_work, tmp_path)
    original, raw = actual_work
    receipt = value["validation_receipts"][0]
    changed = deepcopy(value)
    changed["validation_receipts"][0]["stdout_sha256"] = "bad"
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="validation_stream_hash"):
        q.replay(path)
    for case, reason in [
        ("authority", "authority_reduction"),
        ("tasks", "full_task_primitive"),
        ("history", "historical_primitive"),
    ]:
        work = deepcopy(original)
        if case == "authority":
            work["contract"]["planning_matched"] ^= True
        elif case == "tasks":
            work["tasks"][0]["prompt"] += " changed"
        else:
            work["history"][0]["honest_verdict"] = "invented"
        atomic_json(raw / "measurement.json", work)
        with patch.object(q.runner, "q", q):
            changed = q.build(work, raw, [receipt], {}, [], tmp_path / "deleted", {})
        atomic_json(path, changed)
        with pytest.raises(ValueError, match=reason):
            q.replay(path)
    atomic_json(raw / "measurement.json", original)
    with patch.object(q.runner, "q", q):
        restored = q.build(original, raw, [receipt], {}, [], tmp_path / "deleted", {})
    atomic_json(path, restored)
    monkeypatch.setattr(q.protocol, "replay", lambda _: False)
    with pytest.raises(ValueError, match="protocol_primitive_drift"):
        q.replay(path)
