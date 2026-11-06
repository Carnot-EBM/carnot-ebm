"""REQ-REPORT-8854: matched controls and frozen causal methods use real operands."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import linear_ablation_methods_8854 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_protocol() -> None:
    """SCENARIO-REPORT-8854-CONTRACT: cohort identity survives entity renaming."""
    protocol = m.protocol()
    streams = protocol["cohort_manifest"]
    assert len(streams) == 24
    assert [s["seed"] for s in streams[:5]] == [11, 23, 37, 47, 59]
    assert all(len(s[k]) == 48 for s in streams for k in ("online", "retention"))
    assert all(s["delays"] == [8] * 48 for s in streams[:12])
    assert all(s["delays"] == [4, 8, 12] * 16 for s in streams[12:])
    rows = m.census(protocol, "later_execution")
    assert len(rows) == 27648
    assert len({(r["unit_id"], r["arm"]) for r in rows}) == 27648
    assert sum(r["split"] == "online" for r in rows) == 6912
    worlds = [w for s in streams for k in ("online", "retention") for w in s[k]]
    assert len({w["marker"] for w in worlds}) == 2304
    assert len({w["structure_id"] for w in worlds}) == 2304
    value = m.payload(worlds[0])
    assert len({t for q in value["query"] for t in (q[0], q[2])}) == 7
    assert m.protocol() == protocol


@pytest.mark.parametrize("arm", m.ARMS)
def test_release_boundaries(tmp_path: Path, arm: str) -> None:
    """REQ-REPORT-8854: frozen components, evidence gates and commits are atomic."""
    s = m.c.Store(tmp_path / "state.json", m.c.head(11, "linear"))
    value = m.c.m.world("fixture", "support")
    sealed = s.predict("q1", value, 0.5)
    before = s.path.read_bytes()
    delay = 0 if "immediate" in arm else 8
    assert m.release(s, "q1", value, 1 + delay, arm, delay, label="support")["status"] == "invalid"
    assert s.path.read_bytes() == before
    s.close()
    delay = 0 if "immediate" in arm else 8
    for event, v, ordinal, label in [
        ("missing", value, 9, "support"),
        ("q1", value, 0, "support"),
        ("q1", value, 99, "support"),
        ("q1", value, 1 + delay, "refute"),
        ("q1", value, 1 + delay, None),
        ("q1", m.c.m.world("changed", "support"), 1 + delay, "support"),
    ]:
        assert m.release(s, event, v, ordinal, arm, delay, label=label)["status"] == "invalid"
        assert s.path.read_bytes() == before
    with pytest.raises(ValueError, match="arm_or_delay"):
        m.release(s, "q1", value, 9, "unknown", 8, label="support")
    old = deepcopy(s.state)
    with patch.object(m.c, "_atomic_write", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            m.release(s, "q1", value, 1 + delay, arm, delay, label="support")
    assert s.state == old and s.path.read_bytes() == before
    row = m.release(s, "q1", value, 1 + delay, arm, delay, label="support")
    assert row["prediction_hash"] == canonical_hash(sealed)
    assert row["status"] == "complete" and s.state["acks"] == ["q1"]
    assert row["weights_changed"] == m.ARMS[arm]["calibration"]
    assert row["templates_changed"] == m.ARMS[arm]["additions"]
    restart = m.c.Store(s.path, {})
    assert restart.state == s.state
    assert (
        m.release(restart, "q1", value, 1 + delay, arm, delay, label="support")["status"]
        == "invalid"
    )


def test_controls(tmp_path: Path) -> None:
    """REQ-REPORT-8854: numerical and original combined transitions really execute."""
    rows = m.controls(tmp_path)
    assert all(r["passed"] for r in rows)
    assert {r["check"] for r in rows} >= {
        "finite_difference",
        "original_combined",
        "prefix_shuffle",
    }


def test_pilot_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8854-CONTRACT: real pilots preserve seals and retention state."""
    stream = m.stream(24)
    heads = {stream["seed"]: m.c.head(stream["seed"], "linear")}
    result = m.run_stream(tmp_path / "pilot", stream, heads)
    assert len(result["rows"]) == 1152
    assert len(result["gradient_rows"]) == 192
    assert all(r["status"] == "complete" for r in result["rows"])
    assert len(result["movement_intervals"]) == 12
    assert all(r["unchanged"] for r in result["restart_rows"])
    assert m.projection([1.0, 2.0], 1.0)["passed"]
    assert not m.projection([100.0, 200.0], 1.0)["passed"]
    assert m.replay_pilot(result, tmp_path / "replay")
    changed = deepcopy(result)
    changed["rows"][0]["metric"] = 99
    with pytest.raises(ValueError, match="pilot_drift"):
        m.replay_pilot(changed, tmp_path / "tamper")


@pytest.fixture(scope="module")
def authority(tmp_path_factory: Any) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """REQ-REPORT-8854: authenticate actual frozen external inputs once."""
    import subprocess

    from carnot.reporting import linear_ablation_runner_8854 as r

    raw = tmp_path_factory.mktemp("authority8854")
    script = (
        "from pathlib import Path; from carnot.reporting import linear_ablation_runner_8854 as r; "
        "from carnot.reporting.current_work_receipt import atomic_json; "
        f"e,p,refs=r.preflight(Path({str(raw)!r}),r.m.prior.AUTHORITY); "
        f"atomic_json(Path({str(raw / 'response.json')!r}),dict(evidence=dict(development_rows=e['development_rows'],fits=e['fits']),pre=p,refs=refs))"
    )
    child = subprocess.run(
        [str(r.ROOT / ".venv/bin/python"), "-u", "-c", script],
        cwd=r.ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads((raw / "response.json").read_bytes())
    return value["evidence"], value["pre"], value["refs"]


def test_authority_and_fits(tmp_path: Path, authority: Any, monkeypatch: Any) -> None:
    """REQ-REPORT-8854: reproduce all five original linear initializations."""
    from carnot.reporting import linear_ablation_runner_8854 as r

    evidence, pre, refs = authority
    assert pre["configuration"] == m.protocol()["configuration"] and refs
    assert pre["fresh_query_structure"] not in pre["exposed_structure_hashes"]
    work: dict[str, Any] = dict(protocol=m.protocol())
    r.execute(tmp_path, evidence, work)
    assert work["numerical_training_calls"] == 409
    assert len(work["initialization_rows"]) == 5
    assert len(work["pilot_refs"]) == 2
    monkeypatch.setattr(r, "preflight", lambda raw, source: authority)
    monkeypatch.setattr(r, "execute", lambda raw, evidence, target: target.update(work))
    coverage_file = tmp_path / "fixture.coverage"
    coverage_file.write_text("private coverage binding fixture\n")
    coverage = dict(
        complete=True,
        percent=100,
        combined_reference=r.reference(coverage_file),
        files=[dict(path=name, statements=1, excluded=[], missing=[]) for name in r.OWNED],
    )
    monkeypatch.setattr(m.prior.validation, "coverage_summary", lambda path, names: coverage)
    monkeypatch.setattr(
        r,
        "validate",
        lambda raw: dict(passed=True, receipts=[], coverage=coverage),
    )
    monkeypatch.setattr(r, "terminal", lambda a, b: dict(passed=True, receipts=[]))
    output = tmp_path / (r.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    assert r.replay(output)["pilot_count"] == 2
    changed = json.loads(output.read_bytes())
    changed["owned_statement_coverage"]["percent"] = 99
    atomic_json(output, changed)
    with pytest.raises(ValueError, match="coverage_drift"):
        r.replay(output)
    monkeypatch.undo()
    bad = deepcopy(evidence)
    for row in bad["fits"]:
        if row["objective"] == "linear":
            row["parameters"] = []
    with pytest.raises(ValueError, match="original_fit_drift"):
        r.execute(tmp_path / "drift", bad, dict(protocol=m.protocol()))


def test_preflight_rejections(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8854: external authority absence and changed bytes cannot pass."""
    from carnot.reporting import linear_ablation_runner_8854 as r

    with pytest.raises(FileNotFoundError):
        r.preflight(tmp_path / "missing", tmp_path / "absent.json")
    source = json.loads(m.prior.AUTHORITY.read_bytes())
    for key, value in (("memory_execution_ready_score", 0), ("required_checks_passed", False)):
        changed = dict(source, **{key: value})
        path = tmp_path / "changed.json"
        atomic_json(path, changed)
        monkeypatch.setattr(m.prior, "AUTHORITY_PIN", r.reference(path)["sha256"])
        with pytest.raises(PermissionError):
            r.preflight(tmp_path / key, path)


def test_validation_wrappers(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8854: shared validators receive only owned scope and real CLI."""
    from carnot.reporting import linear_ablation_runner_8854 as r

    def scoped(path: Path) -> dict[str, Any]:
        assert m.prior.validation.OWNED == r.OWNED
        assert m.prior.validation.TESTS == r.TESTS
        return dict(passed=True)

    monkeypatch.setattr(m.prior.validation, "validate", scoped)
    assert r.validate(tmp_path)["passed"]
    monkeypatch.setattr(
        m.prior.validation, "terminal", lambda a, b: dict(passed=m.prior.validation.CLI == r.CLI)
    )
    assert r.terminal(tmp_path / "candidate.json", tmp_path)["passed"]


def publish_fixture(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    failure: Exception | None = None,
    validation: bool = True,
    terminal: bool = True,
) -> Path:
    """REQ-REPORT-8854: exercise publication branches with private validation fixtures."""
    from carnot.reporting import linear_ablation_runner_8854 as r

    def inputs(raw: Path, source: Path) -> Any:
        if failure is not None:
            raise failure
        return {}, {}, []

    def execute(raw: Path, evidence: Any, work: dict[str, Any]) -> None:
        work.update(causal_control_rows=[], timing_projection={}, numerical_training_calls=0)

    monkeypatch.setattr(r, "preflight", inputs)
    monkeypatch.setattr(r, "execute", execute)
    monkeypatch.setattr(
        r,
        "validate",
        lambda raw: dict(passed=validation, receipts=[], coverage=dict(complete=validation)),
    )
    monkeypatch.setattr(r, "terminal", lambda a, b: dict(passed=terminal, receipts=[]))
    path = tmp_path / (r.NAME + ".json")
    assert r.main(["--private-output", str(path)]) == 0
    return path


@pytest.mark.parametrize(
    "failure,verdict",
    [
        (None, "circular_positive"),
        (FileNotFoundError("absent"), "blocked"),
        (KeyError("field"), "blocked"),
        (PermissionError("observed_zero"), "blocked"),
        (ValueError("producer"), "blocked"),
    ],
)
def test_private_publication(
    tmp_path: Path, monkeypatch: Any, failure: Exception | None, verdict: str
) -> None:
    """SCENARIO-REPORT-8854-CONTRACT: complete blocked artifacts retain all slots."""
    path = publish_fixture(tmp_path, monkeypatch, failure=failure)
    value = json.loads(path.read_bytes())
    assert value["verdict_class"] == verdict
    assert value["intended_count"] == value["censored_count"] == 27648
    assert value["linear_methods_ready_score"] == int(verdict == "circular_positive")


def test_failed_owned_validation(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8854: internal failures and failed checks disqualify readiness."""
    from carnot.reporting import linear_ablation_runner_8854 as r

    path = publish_fixture(tmp_path / "validation", monkeypatch, validation=False)
    assert json.loads(path.read_bytes())["verdict_class"] == "disqualified"
    path = publish_fixture(tmp_path / "terminal", monkeypatch, terminal=False)
    assert json.loads(path.read_bytes())["linear_methods_ready_score"] == 0

    def fail(*args: Any) -> None:
        raise RuntimeError("owned_control")

    monkeypatch.setattr(r, "execute", fail)
    monkeypatch.setattr(r, "terminal", lambda a, b: dict(passed=True, receipts=[]))
    path = tmp_path / "owned" / (r.NAME + ".json")
    assert r.main(["--output", str(path)]) == 0
    assert json.loads(path.read_bytes())["gate_check_summary"][0]["status"] == "owned_failure"


def test_replay_and_cli(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8854: private cold replay rejects aggregate and rehashed protocol edits."""
    import os
    import subprocess

    from carnot.reporting import linear_ablation_runner_8854 as r

    path = publish_fixture(tmp_path, monkeypatch, failure=FileNotFoundError("absent"))
    assert r.replay(path)["replay_passed"]
    assert r.main(["--replay", str(path)]) == 0
    assert r.main(["--date", "bad"]) == 1
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for argv, code in (
        (["--replay", str(path)], 0),
        (["--date", "bad"], 1),
        (["--replay", str(tmp_path / "absent.json")], 1),
    ):
        child = subprocess.run(
            [str(r.ROOT / ".venv/bin/python"), "-u", str(r.ROOT / r.CLI), *argv],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == code, child.stdout + child.stderr
    original = json.loads(path.read_bytes())
    for key, value in (
        ("completed_count", 1),
        ("linear_methods_ready_score", 1),
        ("causal_control_rows", [dict(passed=True)]),
    ):
        atomic_json(path, dict(original, **{key: value}))
        with pytest.raises(ValueError):
            r.replay(path)
    atomic_json(path, original)
    log = tmp_path / "receipt.log"
    log.write_text("actual private log\n")
    original["validation_receipts"] = [
        dict(log_path=str(log), log_sha256=r.reference(log)["sha256"])
    ]
    atomic_json(path, original)
    assert r.replay(path)["replay_passed"]
    log.write_text("tampered\n")
    assert r.main(["--replay", str(path)]) == 1
    original["validation_receipts"] = []
    work = r.bound(original["raw_shard_hashes"][0])
    work["protocol"]["future_window"] = [1, 48]
    evidence_path = Path(original["raw_shard_hashes"][0]["path"])
    atomic_json(evidence_path, work)
    original["raw_shard_hashes"][0] = r.reference(evidence_path)
    atomic_json(path, original)
    with pytest.raises(ValueError, match="protocol_drift"):
        r.replay(path)
