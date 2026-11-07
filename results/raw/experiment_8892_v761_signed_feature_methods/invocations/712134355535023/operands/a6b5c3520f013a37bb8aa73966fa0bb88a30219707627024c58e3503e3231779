"""REQ-REPORT-8867: separate template exposure from released-label information."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import feedback_information_8867 as r
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Use the original events, never a synthetic substitute for historical evidence."""
    work, pre, refs = r.preflight(tmp_path_factory.mktemp("private"), r.SOURCE)
    with patch.object(r.a.c, "update", side_effect=AssertionError("historical refit")):
        historic = r.historical(work)
    controls = r.development(work)
    return dict(historical=historic, development=controls, preconditions=pre, refs=refs)


def test_history_and_controls(evidence: dict[str, Any]) -> None:
    """SCENARIO-REPORT-8867-CONTRACT: retain all streams and report real effective n."""
    value = r.reduce(evidence)
    assert value["diagnosis_ready_score"] == 1
    assert value["completed_count"] == value["intended_count"] == 28800
    assert len(value["matched_control_rows"]) == 6
    assert len({x["cluster_id"] for x in value["template_cardinality_rows"]}) == 24
    assert all(
        x["distinct_normalized_query_templates"] == 1 for x in value["template_cardinality_rows"]
    )
    assert value["effective_information"]["distinct_query_templates"] == 1
    assert value["effective_information"]["repeated_observations_are_independent"] is False
    assert value["effective_information"]["historical_shuffled_label_mismatches"] == 656
    assert value["effective_information"]["historical_shuffled_exposure_differences"] == 0
    assert value["effective_information"]["historical_shuffled_multiset_changes"] == 24
    assert all(x["passed"] for x in value["control_sensitivity"])
    assert value["forward_method"]["experiment_id"] == 8869
    assert any(x["label_multiset_preserved"] for x in value["matched_control_rows"])


def test_query_only_and_permutation() -> None:
    """REQ-REPORT-8867: query syntax cannot consult labels, facts or witnesses."""
    world = r.a.payload(r.a.stream(26)["online"][0])
    with (
        patch.object(r.a.c.m, "decide", side_effect=AssertionError("oracle")),
        patch.object(r.a.c, "binding_provenance", side_effect=AssertionError("witness")),
    ):
        key = r.query_template(world["query"])
        assert key == r.query_template(r.a.payload(r.a.stream(27)["online"][0])["query"])
    labels = list(r.a.c.m.LABELS) * 2
    shuffled = r.permute(labels, 7)
    assert sorted(shuffled) == sorted(labels) and shuffled == r.permute(labels, 7)
    assert r.permute([], 7) == []


def test_feedback_rejects() -> None:
    """REQ-REPORT-8867: learned controls keep verified evidence and seal validation."""
    world = r.a.payload(r.a.stream(26)["online"][0])
    truth = r.a.c.m.decide(world)["decision"]
    sealed = dict(evidence_hash=r.canonical_hash(world), ordinal=1)
    assert r.verify_release(world, sealed, truth, 14) == truth
    for pending, label, at in [
        (sealed, "bad", 14),
        (dict(sealed, evidence_hash="bad"), truth, 14),
        (sealed, truth, 1),
    ]:
        with pytest.raises(ValueError):
            r.verify_release(world, pending, label, at)


def test_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8867: original bytes and resource checks precede owned work."""
    with pytest.raises(FileNotFoundError):
        r.preflight(tmp_path / "missing", tmp_path / "absent")
    with (
        patch.object(r.old, "preflight", side_effect=RuntimeError("internal_resources")),
        pytest.raises(RuntimeError),
    ):
        r.preflight(tmp_path / "env", r.SOURCE)
    monkeypatch.setattr(r, "load", lambda _: dict(required_checks_passed=False))
    with pytest.raises(ValueError):
        r.preflight(tmp_path / "zero", r.SOURCE)


def test_replay(evidence: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8867-CONTRACT: repaired hashes cannot bless forged controls."""
    raw, candidate = tmp_path / "evidence.json", tmp_path / (r.NAME + ".json")
    atomic_json(raw, evidence)
    value = dict(
        r.reduce(evidence),
        verdict_class="null",
        acceptance_gates=r.GATES,
        raw_shard_hashes=[r.reference(raw)],
        source_artifact_hashes=[],
        code_config_hashes=[],
        validation_receipts=[],
    )
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    child = subprocess.run(
        [sys.executable, "-u", str(r.ROOT / r.CLI), "--replay", str(candidate)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    monkeypatch.setattr(r, "historical", lambda _: deepcopy(evidence["historical"]))
    expected_dev = deepcopy(evidence["development"])
    monkeypatch.setattr(r, "development", lambda _: deepcopy(expected_dev))
    value["diagnosis_ready_score"] = 0
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(candidate)
    value["diagnosis_ready_score"] = 1
    evidence = deepcopy(evidence)
    evidence["development"]["batches"][0]["permuted_labels"][0] = "bad"
    atomic_json(raw, evidence)
    value["raw_shard_hashes"] = [r.reference(raw)]
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="evidence_drift"):
        r.replay(candidate)
    value["acceptance_gates"] = {}
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="acceptance_drift"):
        r.replay(candidate)


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "missing",
        "field",
        "rejected",
        "zero",
        "integrity",
        "controls",
        "owned",
        "terminal",
        "deadline",
        "partial",
    ],
)
def test_runner_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """REQ-REPORT-8867: retain censoring and distinguish external from owned failures."""
    from carnot.reporting import feedback_information_runner_8867 as runner

    minimal = r.empty("fixture")
    minimal["diagnosis_ready_score"] = 1

    def preflight(raw: Path, path: Path) -> Any:
        if failure in {"missing", "field", "rejected", "zero", "integrity", "partial"}:
            raise {
                "missing": FileNotFoundError("absent"),
                "field": KeyError("authority"),
                "rejected": ValueError("rejected_producer"),
                "zero": PermissionError("observed_zero"),
                "integrity": RuntimeError("internal_resources"),
                "partial": TimeoutError("deadline"),
            }[failure]
        return {}, {}, []

    def historical(work: dict[str, Any]) -> dict[str, Any]:
        if failure == "controls":
            raise ValueError("owned_controls")
        return {}

    monkeypatch.setattr(r, "preflight", preflight)
    monkeypatch.setattr(r, "historical", historical)
    monkeypatch.setattr(r, "development", lambda _: {})
    monkeypatch.setattr(r, "reduce", lambda _: deepcopy(minimal))
    monkeypatch.setattr(
        runner, "validate", lambda _: dict(passed=failure != "owned", receipts=[], coverage={})
    )
    monkeypatch.setattr(
        runner, "terminal", lambda *args: dict(passed=failure != "terminal", receipts=[])
    )
    if failure == "deadline":
        monkeypatch.setattr(runner, "BUDGET", -1)
    output = tmp_path / (r.NAME + ".json")
    assert runner.main(["--private-output", str(output), "--raw-dir", str(tmp_path / "raw")]) == 0
    value = json.loads(output.read_bytes())
    expected = (
        "null"
        if not failure
        else "blocked"
        if failure in {"missing", "field", "rejected", "zero"}
        else "partial"
        if failure == "partial"
        else "disqualified"
    )
    assert value["verdict_class"] == expected
    assert value["diagnosis_ready_score"] == int(not failure)
    assert value["intended_count"] == 28800
    assert runner.main(["--date", "wrong"]) == 1
    assert runner.main(["--replay", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(r, "replay", lambda _: dict(passed=True))
    assert runner.main(["--replay", str(output)]) == 0


def test_runner_delegates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8867: preserve the existing validation and terminal supervisors."""
    from carnot.reporting import feedback_information_runner_8867 as runner

    monkeypatch.setattr(runner.existing, "validate", lambda _: dict(passed=True))
    monkeypatch.setattr(runner.existing, "terminal", lambda *args: dict(passed=True))
    assert runner.validate(tmp_path)["passed"]
    assert runner.terminal(tmp_path / "candidate", tmp_path)["passed"]


def test_cli_errors(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8867-CONTRACT: the real CLI rejects bad dates and missing replays."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for args, marker in [
        (["--date", "wrong"], "date_rejected"),
        (["--replay", str(tmp_path / "absent")], "replay_rejected"),
    ]:
        child = subprocess.run(
            [sys.executable, "-u", str(r.ROOT / r.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == 1 and marker in child.stdout


def test_blocked_replay_and_logs(tmp_path: Path) -> None:
    """REQ-REPORT-8867: blocked census and actual validation logs remain bound."""
    raw, log = tmp_path / "empty.json", tmp_path / "check.log"
    atomic_json(raw, {})
    log.write_text("actual log\n")
    value = dict(
        r.empty("missing_file"),
        verdict_class="blocked",
        acceptance_gates=r.GATES,
        raw_shard_hashes=[r.reference(raw)],
        source_artifact_hashes=[],
        code_config_hashes=[],
        validation_receipts=[dict(log_path=str(log), log_sha256=r.reference(log)["sha256"])],
    )
    candidate = tmp_path / "blocked.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    log.write_text("changed")
    with pytest.raises(ValueError, match="validation_log"):
        r.replay(candidate)
    value["raw_shard_hashes"][0]["sha256"] = "changed"
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="replay_operand"):
        r.replay(candidate)


def test_relative_receipt_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8867-CONTRACT: real relative log receipts replay from /tmp."""
    raw, log = tmp_path / "empty.json", tmp_path / "executed.log"
    atomic_json(raw, {})
    log.write_text("executed validation\n")
    value = dict(
        r.empty("missing_file"),
        verdict_class="blocked",
        acceptance_gates=r.GATES,
        raw_shard_hashes=[r.reference(raw)],
        source_artifact_hashes=[],
        code_config_hashes=[],
        validation_receipts=[
            dict(log_path=os.path.relpath(log, r.ROOT), log_sha256=r.reference(log)["sha256"])
        ],
    )
    candidate = tmp_path / "relative.json"
    atomic_json(candidate, value)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    child = subprocess.run(
        [sys.executable, "-u", str(r.ROOT / r.CLI), "--replay", str(candidate)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr


@pytest.mark.parametrize("retained", [False, True])
def test_reuse_global_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retained: bool
) -> None:
    """REQ-REPORT-8867: owned repairs reuse an authenticated health log without rerunning it."""
    from carnot.reporting import feedback_information_runner_8867 as runner

    log = tmp_path / "health.log"
    log.write_text("prior bounded full suite\n")
    receipt = dict(
        name="global_python_health",
        scope="repository_health",
        passed=False,
        log_path=str(log),
        log_sha256=r.reference(log)["sha256"],
    )
    monkeypatch.setattr(r, "ROOT", tmp_path)
    if retained:
        atomic_json(
            tmp_path / "results" / (r.NAME + ".json"), dict(repository_health_receipts=[receipt])
        )

    def commands(plan: list[Any], logs: Path) -> list[Any]:
        assert any(p[0] == "global_python_health" for p in plan) is not retained
        return []

    def validate(raw: Path) -> dict[str, Any]:
        receipts = runner.existing.validation.commands(
            [
                ("owned_pytest", [], "owned", 600),
                ("global_python_health", [], "repository_health", 180),
            ],
            raw,
        )
        assert len(receipts) == int(retained)
        return dict(passed=True)

    monkeypatch.setattr(runner.existing.validation, "commands", commands)
    monkeypatch.setattr(runner.existing, "validate", validate)
    assert runner.validate(tmp_path)["passed"]


@pytest.mark.parametrize("state", ["relative", "changed", "missing", "empty"])
def test_retained_health_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """REQ-REPORT-8867: authenticate retained health logs before reusing original failures."""
    from carnot.reporting import feedback_information_runner_8867 as runner

    log = tmp_path / "health.log"
    log.write_text("prior bounded full suite\n")
    receipt = dict(
        name="global_python_health",
        scope="repository_health",
        passed=False,
        log_path="health.log",
        log_sha256=r.reference(log)["sha256"],
    )
    monkeypatch.setattr(r, "ROOT", tmp_path)
    atomic_json(
        tmp_path / "results" / (r.NAME + ".json"),
        dict(repository_health_receipts=[] if state == "empty" else [receipt]),
    )
    if state == "changed":
        log.write_text("changed log\n")
    elif state == "missing":
        log.unlink()

    def commands(plan: list[Any], logs: Path) -> list[Any]:
        assert [p[0] for p in plan] == (
            ["owned_pytest", "global_python_health"] if state == "empty" else ["owned_pytest"]
        )
        assert logs == tmp_path
        return [dict(name="owned_pytest", scope="owned", passed=True)]

    def validate(raw: Path) -> dict[str, Any]:
        assert state in {"relative", "empty"}
        receipts = runner.existing.validation.commands(
            [
                ("owned_pytest", [], "owned", 600),
                ("global_python_health", [], "repository_health", 180),
            ],
            raw,
        )
        assert receipts == [dict(name="owned_pytest", scope="owned", passed=True)] + (
            [receipt] if state == "relative" else []
        )
        return dict(passed=True)

    monkeypatch.setattr(runner.existing.validation, "commands", commands)
    monkeypatch.setattr(runner.existing, "validate", validate)
    if state in {"changed", "missing"}:
        with pytest.raises(ValueError if state == "changed" else FileNotFoundError):
            runner.validate(tmp_path)
    else:
        assert runner.validate(tmp_path)["passed"]


def test_replay_resolves_repository_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8867-CONTRACT: repository-relative logs resolve outside the checkout."""
    root, outside = tmp_path / "repository", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    raw, log = root / "empty.json", root / "executed.log"
    atomic_json(raw, {})
    log.write_text("executed validation\n")
    candidate = root / "candidate.json"
    atomic_json(
        candidate,
        dict(
            r.empty("missing_file"),
            verdict_class="blocked",
            acceptance_gates=r.GATES,
            raw_shard_hashes=[r.reference(raw)],
            source_artifact_hashes=[],
            code_config_hashes=[],
            validation_receipts=[dict(log_path=log.name, log_sha256=r.reference(log)["sha256"])],
        ),
    )
    monkeypatch.setattr(r, "ROOT", root)
    monkeypatch.chdir(outside)
    assert r.replay(candidate)["passed"]
    log.write_text("changed validation\n")
    with pytest.raises(ValueError, match="validation_log"):
        r.replay(candidate)
