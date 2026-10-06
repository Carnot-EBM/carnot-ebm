"""REQ-VERIFY-8197, REQ-REPORT-8197: private independent audit evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import selective_decision_audit_8197 as e


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Real external callers must work without an ambient import path."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private subprocess", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after private subprocess", result.returncode, flush=True)
    return result


def test_private_cli_e2e019(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8197: success, missing, aggregate tamper and cold replay."""
    path = tmp_path / "experiment_8197_fixture.json"
    child = cli(tmp_path, "--fixture-output", path)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(path.read_text())
    assert value["selective_audit_ready_score"] == 1
    assert len(value["rows"]) == 896 and value["intended_count"] == 128
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["independent_generalization_score"] == 0
    assert cli(tmp_path, "--cold-replay", path).returncode == 0
    atomic_json(path, dict(value, completed_count=1))
    assert cli(tmp_path, "--cold-replay", path).returncode == 1
    atomic_json(path, value)
    primitive = Path(value["raw_shard_hashes"][0]["path"])
    saved = primitive.read_bytes()
    primitive.write_text("{}")
    assert not e.replay(path)
    primitive.write_bytes(saved)
    blocked = tmp_path / "blocked/experiment_8197_fixture.json"
    assert cli(tmp_path, "--fixture-output", blocked, "--mutation", "block").returncode == 0
    b = json.loads(blocked.read_text())
    assert b["honest_verdict"] == "complete_blocked_sealed_evaluation_ready_score"
    assert b["selective_audit_ready_score"] == 0 and e.replay(blocked)
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    assert cli(tmp_path, "--date", "19990101").returncode == 2


def test_original_labels_and_identity(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8197: independently reconstructed targets reject swaps."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    data = work["evidence"]
    reduced = e.reduce(data)
    assert reduced["eligible_count"] == 128
    assert reduced["equivalent_logistic_parity"]["passed"]
    assert reduced["common_logit_shift_control"]["passed"]
    for mutate in (
        lambda d: d["clock"].update(labels_opened_ns=0),
        lambda d: d["predictions"][0].update(p=0.7),
        lambda d: d["targets"][0].update(y=1),
        lambda d: d["targets"].pop(),
        lambda d: d["slots"][0].update(answer_bytes=b"wrong".hex()),
        lambda d: d["original_response_records"][0].update(source_id="wrong"),
    ):
        bad = deepcopy(data)
        mutate(bad)
        with pytest.raises((ValueError, KeyError)):
            e.reduce(bad)
    bad = e.build(work, tmp_path / "raw", [dict(passed=False)])
    assert bad["verdict_class"] == "disqualified"
    assert bad["selective_audit_ready_score"] == bad["h1_development_signal_score"] == 0


def panel(*, missing: bool = False, escalation: bool = False) -> list[dict[str, Any]]:
    """Known controls exercise scientific gates independently of a model head."""
    rows = []
    for i in range(128):
        y = i % 2
        for arm in e.n.ARMS:
            p, action = (0.99, "reject") if y else (0.01, "accept")
            if arm == "frozen_v707_radial":
                p, action = 0.3, "escalate"
            if arm == "always_escalate" or escalation or missing:
                action = "escalate"
            if arm == "always_escalate" or missing:
                p = None
            rows.append(
                e.old.score(
                    dict(
                        unit_id=str(i),
                        source_cluster_id=str(i),
                        arm=arm,
                        condition="all_original_reserved_slots",
                        p=p,
                        action=action,
                        prediction_set=[y] if p is not None and arm.endswith("set") else None,
                        status="failed" if missing else "completed",
                        exclusion_reason=None,
                    ),
                    y,
                )
            )
    return rows


def test_scientific_gates_and_exact_intervals() -> None:
    """REQ-VERIFY-8197: source draws and each denominator keep distinct meaning."""
    result = e.statistics(panel())
    assert result["h1_development_signal_score"] == 1
    assert result["paired_intervals"]["all_slot"]["valid_draws"] == 10000
    assert result["acceptance_operands"]["complete_sources"] == 128
    assert e.statistics(panel(missing=True))["h1_development_signal_score"] == 0
    assert e.statistics(panel(escalation=True))["h1_development_signal_score"] == 0
    assert e.statistics([])["eligible_count"] == 0
    assert e.binomial(0, 0)["interval"] == [None, None]
    assert e.binomial(0, 10)["interval"][1] == pytest.approx(1 - 0.025**0.1)
    assert e.binomial(10, 10)["interval"][0] == pytest.approx(0.025**0.1)
    assert e.binomial(5, 10)["interval"][0] < 0.5
    with pytest.raises(ValueError, match="duplicate_source_arm"):
        e.statistics(panel() + [panel()[0]])


def test_natural_custody_and_rehashed_tamper(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8197: real cached inputs stay private and replay rejects drift."""
    work = e.measure(e.ROOT, tmp_path / "natural")
    assert work["evidence"], work["checks"][-1]
    assert all(c["passed"] for c in work["checks"])
    value = e.build(work, tmp_path / "natural", [dict(passed=True)])
    assert value["completed_count"] == 97
    assert value["historical_v707_null"]["verdict_class"] == "null"
    assert value["selective_audit_ready_score"] == 1
    assert value["paired_intervals"]["all_slot"]["valid_draws"] == 10000
    path = tmp_path / "experiment_8197_natural.json"
    atomic_json(path, value)
    assert e.replay(path)
    log = tmp_path / "receipt.log"
    log.write_text("original")
    with_log = e.build(
        work,
        tmp_path / "natural",
        [dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))],
    )
    atomic_json(path, with_log)
    assert e.replay(path)
    log.write_text("changed")
    assert not e.replay(path)
    atomic_json(path, value)
    ref = value["measurement_reference"]
    evidence = Path(ref["path"])
    saved = evidence.read_bytes()
    forged = deepcopy(work)
    forged["evidence"]["targets"][0]["y"] = 1 - forged["evidence"]["targets"][0]["y"]
    atomic_json(evidence, forged)
    value["measurement_reference"] = e.reference(evidence)
    atomic_json(path, value)
    assert not e.replay(path)
    evidence.write_bytes(saved)
    missing = e.measure(tmp_path, tmp_path / "missing")
    b = e.build(missing, tmp_path / "missing", [dict(passed=True)])
    assert b["honest_verdict"] == "complete_blocked_upstream_exists"
    monkeypatch.setitem(e.PINS, e.UPSTREAM, "wrong")
    wrong = e.measure(e.ROOT, tmp_path / "wrong_hash")
    assert wrong["checks"][-1]["check"] == "input_sha256"
    specs = e.manifest(tmp_path, path)
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    assert specs["commands"][0]["deadline_s"] == 300


def test_missing_evidence_labels_and_costs(tmp_path: Path, monkeypatch: Any) -> None:
    """SCENARIO-VERIFY-8197: missing evidence and annotation custody stay explicit."""
    work = e.measure(tmp_path, tmp_path / "fixture", fixture=True)
    data = deepcopy(work["evidence"])
    data["sealed"]["features"][0].update(x=None, status="failed", exclusion_reason="transport")
    data["predictions"] = e.sealed.reduce(data["sealed"])["prediction_rows"]
    result = e.reduce(data)
    assert result["eligible_count"] == 127
    assert all(r["numerator"] == 0.5 for r in result["rows"][:7])
    data["original_response_records"][1]["quality"] = "incomplete"
    expected, _ = e.old.base.human.target(data["original_response_records"][1], b"First fact.")
    data["targets"][1].update(expected)
    assert e.reduce(data)["eligible_count"] == 126
    bad = deepcopy(data)
    bad["costs"]["escalate"] = 0
    with pytest.raises(ValueError, match="original_cost_matrix"):
        e.reduce(bad)
    bad = deepcopy(data)
    bad["slots"][0]["source_bytes"] = b"different source".hex()
    with pytest.raises(ValueError, match="original_byte_identity"):
        e.reduce(bad)

    def custody_failure(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("independent_custody_failure")

    monkeypatch.setattr(e.sealed, "reduce", custody_failure)
    blocked = e.measure(tmp_path, tmp_path / "drift", fixture=True)
    assert blocked["checks"][-1]["observed"] == "independent_custody_failure"
    assert not blocked["evidence"]
