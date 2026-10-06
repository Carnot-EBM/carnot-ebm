"""REQ-VERIFY-8194 / REQ-REPORT-8194: private selective-policy qualification."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import selective_methods_8194 as e
from carnot.verify import selective_rule_8194 as n


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Use the real entry point without caller import paths or public fixtures."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private subprocess", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after private subprocess", result.returncode, flush=True)
    return result


def test_scalar_quantiles_and_sets() -> None:
    """SCENARIO-VERIFY-8194: exact order statistics include ties and overflow."""
    for count in [0, 1, 18, 19, 20, 25, 28, 100]:
        scores = [(i % 7) / 7 for i in range(count)]
        q = n.quantile(scores)
        rank = math.ceil((count + 1) * 0.95)
        expected = sorted(scores)[rank - 1] if rank <= count else None
        assert q == dict(n=count, rank=rank, threshold=expected, infinity=rank > count)
    qs = [n.quantile([0.2] * 19), n.quantile([1 - 0.7] * 19)]
    assert n.prediction_set(0.2, qs) == [0]
    assert n.prediction_set(0.7, qs) == [1]
    assert n.prediction_set(0.5, qs) == []
    assert n.prediction_set(None, qs) == [0, 1]
    assert n.prediction_set(0.5, [n.quantile([])] * 2) == [0, 1]
    for bad in [-1.0, 2.0, float("nan")]:
        with pytest.raises(ValueError, match="probability"):
            n.prediction_set(bad, qs)
    for shift in [-1000, 0, 1000]:
        p = n.logit_probability([shift, shift + 2], 0.5)
        assert p == pytest.approx(1 / (1 + math.exp(-4)), abs=1e-14)
        assert p == pytest.approx(n.logit_probability([shift + 10, shift + 12], 0.5))
    assert n.logit_probability([0, -1000], 1) == 0
    with pytest.raises(ValueError, match="temperature"):
        n.logit_probability([0, 1], 0)


def test_roles_support_and_leakage(tmp_path: Path) -> None:
    """REQ-VERIFY-8194: identity sorting precedes evaluator access and gates stay fixed."""
    rows, reserved, frozen = e.fixture()
    manifest = n.freeze_roles(rows, reserved)
    assert {k: len(v) for k, v in manifest.items()} == dict(
        head_fit=96, temperature_fit=32, calibration=64, reserved=128
    )
    assert n.freeze_roles(list(reversed(rows)), list(reversed(reserved))) == manifest
    support = n.class_support(rows, manifest)
    assert support["calibration"]["completed"] == 64
    for mutate in [
        lambda rs: rs[0].update(evaluator_label=1),
        lambda rs: rs[0].update(role="evaluation"),
        lambda rs: rs[1].update(source_cluster_id=rs[0]["source_cluster_id"]),
        lambda rs: rs.pop(),
    ]:
        bad = deepcopy(rows)
        mutate(bad)
        with pytest.raises(ValueError):
            n.freeze_roles(bad, reserved)
    bad = deepcopy(reserved)
    bad[0]["y"] = 1
    with pytest.raises(ValueError, match="evaluator_label"):
        n.freeze_roles(rows, bad)
    fit = n.train(rows, manifest, tmp_path)
    assert len(fit["heads"]) == 2
    for h in fit["heads"]:
        assert 0.25 <= h["temperature"] <= 4
        assert set(h["head_fit_source_ids"]).isdisjoint(h["temperature_source_ids"])
        assert set(h["head_fit_source_ids"]).isdisjoint(h["calibration_source_ids"])
        assert all(q["n"] >= 20 for q in h["quantiles"])
    changed = deepcopy(rows)
    for r in changed:
        if r["role"] == "tune":
            r["y"] = 1 - r["y"]
    other = n.train(changed, manifest, tmp_path / "changed")
    for a, b in zip(fit["heads"], other["heads"], strict=True):
        assert a["weights"] == b["weights"]
        assert a["temperature"] == b["temperature"]
        assert a["quantiles"] != b["quantiles"]
    assert n.predict(reserved, fit["heads"], frozen)


def test_private_publication_replay_and_masks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8194: real private CLI rejects altered rows and headlines."""
    output = tmp_path / "experiment_8194_fixture.json"
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["selective_protocol_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == [] and value["trained_head_specs"]
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert len(value["rows"]) == 128 * 7
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for key, changed in [("completed_count", 1), ("H1", {}), ("rows", [])]:
        bad = deepcopy(value)
        bad[key] = changed
        atomic_json(output, bad)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    for ref in value["raw_shard_hashes"]:
        p = Path(ref["path"])
        saved = p.read_bytes()
        p.write_text("{}")
        assert not e.replay(output)
        p.write_bytes(saved)
    log = tmp_path / "log.txt"
    log.write_text("original")
    value["validation_receipts"] = [
        dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))
    ]
    atomic_json(output, value)
    assert e.replay(output)
    log.write_text("changed")
    assert not e.replay(output)
    blocked = tmp_path / "blocked/experiment_8194_fixture.json"
    assert cli(tmp_path, "--fixture-output", blocked, "--mutation", "block").returncode == 0
    b = json.loads(blocked.read_text())
    assert b["verdict_class"] == "blocked" and b["selective_protocol_ready_score"] == 0
    assert e.replay(blocked)
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/private.json").returncode == 2
    assert cli(tmp_path, "--mutation", "block").returncode == 2
    assert cli(tmp_path, "--date", "19990101").returncode == 2
    assert not e.replay(tmp_path / "absent")


def test_measurement_independent_reduction(tmp_path: Path) -> None:
    """REQ-VERIFY-8194: missing evidence escalates and forged scalar rows fail."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    d = work["evidence"]
    result = e.reduce(d)
    assert result["eligible_count"] == 128
    assert result["H1"]["interval"]["valid_draws"] == 10000
    assert result["equivalent_logistic_parity"]["passed"]
    for mutate in [
        lambda x: x["predictions"][0].update(p=0.7),
        lambda x: x["targets"][0].update(source_cluster_id="wrong"),
        lambda x: x["targets"].pop(),
        lambda x: x["clock"].update(labels_opened_ns=0),
        lambda x: x["features"][0].update(evaluator_label=1),
    ]:
        bad = deepcopy(d)
        mutate(bad)
        with pytest.raises(ValueError):
            e.reduce(bad)
    missing = deepcopy(d)
    for r in missing["features"]:
        r["x"] = None
        r["historical_x"] = None
    missing["predictions"] = n.predict(missing["features"], missing["heads"], missing["frozen"])
    result = e.reduce(missing)
    assert result["eligible_count"] == 0
    assert all(r["numerator"] == 0.5 for r in result["rows"])
    assert not result["H1"]["passed"]
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    rows = e.reduce(d)["rows"]
    with pytest.raises(ValueError, match="duplicate_source_arm"):
        n.statistics([*rows, rows[0]])
    assert n.statistics([])["eligible_count"] == 0
    scores = n.class_support([dict(r, x=None) for r in e.fixture()[0]], d["role_manifest"])
    assert scores["head_fit"]["completed"] == 0


def test_natural_inputs_and_validation_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8194: owned checks use file paths and upstream operands are explicit."""
    manifest = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert manifest["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    assert all("::" not in p for s in manifest["commands"][5:] for p in s["argv"])
    work = e.measure(e.ROOT, tmp_path / "natural")
    assert work["evidence"], work["checks"][-1]
    assert all(c["passed"] for c in work["checks"])
    assert e.reduce(work["evidence"])["eligible_count"] == 97
    output = tmp_path / "experiment_8194_natural.json"
    atomic_json(output, e.build(work, tmp_path / "natural", [dict(passed=True)]))
    assert e.replay(output)
    missing = e.measure(tmp_path, tmp_path / "missing")
    assert missing["checks"][-1]["passed"] is False
    assert e.build(missing, tmp_path / "missing", [dict(passed=True)])["verdict_class"] == "blocked"
    monkeypatch.setattr(e, "PROTOCOL_PIN", "wrong")
    wrong = e.measure(e.ROOT, tmp_path / "wrong")
    assert wrong["checks"][-1]["check"] == "input_sha256"
    monkeypatch.undo()
    monkeypatch.setattr(e, "measure", lambda *a, **k: deepcopy(work))

    def fake_manifest(private: Path, candidate: Path) -> dict[str, Any]:
        private.joinpath("coverage.json").write_text("{}")
        return dict(
            commands=[dict(name="owned")],
            repository_health=dict(name="health"),
            terminal_commands=[],
        )

    monkeypatch.setattr(e, "manifest", fake_manifest)
    monkeypatch.setattr(e.audit.base.producer, "checked", lambda *a: dict(passed=True))
    monkeypatch.setattr(e.execution, "publish", lambda *a: None)
    assert e.main(["--output", str(output)]) == 0


def test_owned_failures_and_input_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8194: solver failures disqualify; support failures name counts."""
    from types import SimpleNamespace

    rows, reserved, _ = e.fixture()
    roles = n.freeze_roles(rows, reserved)
    bad = deepcopy(rows)
    bad[0]["role"] = "tune"
    with pytest.raises(ValueError, match="role_count"):
        n.freeze_roles(bad, reserved)
    solve = n.base.solve
    for full in (False, True):

        def failed(phi, y, ridge, deadline):
            if not full or len(phi) == 96:
                return dict(converged=False)
            return solve(phi, y, ridge, deadline)

        monkeypatch.setattr(n.base, "solve", failed)
        with pytest.raises(ValueError, match="nonconvergence"):
            n.train(rows, roles, tmp_path)
    monkeypatch.undo()
    monkeypatch.setattr(n, "minimize_scalar", lambda *a, **k: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match="temperature_nonconvergence"):
        n.train(rows, roles, tmp_path)
    work = e.measure(tmp_path, tmp_path / "failed", fixture=True)
    assert work["owned_failure"] == "temperature_nonconvergence"
    value = e.build(work, tmp_path / "failed", [dict(passed=True)], fixture=True)
    assert value["verdict_class"] == "disqualified"
    assert value["selective_protocol_ready_score"] == 0
    monkeypatch.undo()
    monkeypatch.setattr(
        e, "fixture", lambda: (_ for _ in ()).throw(OSError("missing private input"))
    )
    bad_work = e.measure(tmp_path, tmp_path / "malformed", fixture=True)
    assert bad_work["checks"][-1]["check"] == "input_custody"
    monkeypatch.undo()
    blocked = e.measure(tmp_path, tmp_path / "support", fixture=True, mutation="block")
    assert blocked["checks"][-1]["observed"] == 0
    assert e.build(blocked, tmp_path / "support", [dict(passed=True)])["verdict_class"] == "blocked"
    supports = n.class_support(rows, roles)
    supports["head_fit"]["completed"] = 71
    monkeypatch.setattr(n, "class_support", lambda *a: supports)
    low = e.measure(tmp_path, tmp_path / "low_support", fixture=True)
    assert low["checks"][-1]["expected"] == 72
    assert low["checks"][-1]["observed"] == 71
    assert low["checks"][-1]["op"] == ">="


def test_rehashed_forgery_and_swapped_labels(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8194: refreshed hashes cannot repair contradictory primitives."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    data = work["evidence"]
    bad = deepcopy(data)
    bad["predictions"].pop()
    with pytest.raises(ValueError, match="prediction_count"):
        e.reduce(bad)
    output = tmp_path / "experiment_8194_fixture.json"
    value = e.build(work, tmp_path / "raw", [dict(passed=True)], fixture=True)
    wrong = dict(value, protocol_sha256="wrong")
    atomic_json(output, wrong)
    assert not e.replay(output)
    ref = next(
        r for r in work["raw_shard_hashes"] if Path(r["path"]).stem == "independent_reduction"
    )
    path = Path(ref["path"])
    forged = json.loads(path.read_text())
    forged["H1"]["passed"] = not forged["H1"]["passed"]
    atomic_json(path, forged)
    ref["sha256"] = sha256_file(path)
    atomic_json(tmp_path / "raw/measurement.json", work)
    value["raw_shard_hashes"] = work["raw_shard_hashes"]
    value["measurement_reference"] = e.reference(tmp_path / "raw/measurement.json")
    atomic_json(output, value)
    assert not e.replay(output)
    natural = e.measure(e.ROOT, tmp_path / "natural")["evidence"]
    bad = deepcopy(natural)
    bad["targets"] = deepcopy(bad["targets"])
    bad["targets"][0]["y"] = 1 - bad["targets"][0]["y"]
    with pytest.raises(ValueError, match="original_labels"):
        e.reduce(bad)
    for unit in {r["unit_id"] for r in data["predictions"]}:
        a, b = [
            r
            for r in data["predictions"]
            if r["unit_id"] == unit and r["arm"] in ("local_set", "equivalent_logistic_set")
        ]
        assert abs(a["p"] - b["p"]) <= 1e-10 and a["action"] == b["action"]


def test_positive_control_and_distinct_coverage_denominators() -> None:
    """REQ-VERIFY-8194: a known useful policy passes H1; coverage uses class counts."""
    rows = []
    for i in range(128):
        y = i % 2
        for arm in n.ARMS:
            p = 0.9 if y else 0.01
            if arm == "frozen_v707_radial":
                p = 0.3
            if arm == "always_escalate":
                p = None
            action = (
                "escalate"
                if p is None or arm == "frozen_v707_radial"
                else "reject"
                if y
                else "accept"
            )
            row = dict(
                unit_id=str(i),
                source_cluster_id=str(i),
                arm=arm,
                p=p,
                prediction_set=[y] if arm.endswith("set") else None,
                action=action,
                status="completed" if p is not None else "excluded",
                exclusion_reason=None,
            )
            rows.append(e.audit.score(row, y))
    result = n.statistics(rows)
    assert result["H1"]["passed"] and result["h1_development_signal_score"] == 1
    assert result["H1"]["interval"]["valid_draws"] == 10000
    metrics = next(r for r in result["all_slot_metrics"] if r["arm"] == "local_set")
    assert metrics["singleton_coverage"] == 1
    assert metrics["error_among_accepts"] == dict(numerator=0, denominator=64, rate=0.0)
    assert all(r["denominator"] == 64 and r["coverage"] == 1 for r in metrics["class_coverage"])
