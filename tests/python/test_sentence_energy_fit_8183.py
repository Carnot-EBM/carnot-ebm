"""REQ-VERIFY-8183, REQ-REPORT-8183: private calibrated heads and byte custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import sentence_energy_8183 as n
from carnot.verify import sentence_energy_fit_8183 as e


def cli(tmp_path, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private subprocess", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after private subprocess", result.returncode, flush=True)
    return result


def test_private_cli_success_block_tamper_and_replay(tmp_path):
    """SCENARIO-REPORT-8183: actual script outside checkout, no model or reserved labels."""
    output = tmp_path / "experiment_8183_fixture.json"
    r = cli(tmp_path, "--fixture-output", output)
    assert r.returncode == 0, r.stdout + r.stderr
    v = json.loads(output.read_text())
    assert v["energy_fit_ready_score"] == 1
    assert v["verdict_class"] == "circular_positive"
    assert v["MODEL_SPECS"] == [] and v["call_ledger"] == []
    assert all(p["passed"] for p in v["equivalent_logistic_parity"])
    assert v["comparator_id"] == "radial16"
    assert all(r["role"] in ("fit", "tune") for r in v["rows"])
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    saved = deepcopy(v)
    v["comparator_id"] = "linear12"
    atomic_json(output, v)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, saved)
    head = Path(v["frozen_head_manifest"]["path"])
    original = head.read_bytes()
    head.write_text("{}")
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    head.write_bytes(original)
    blocked = tmp_path / "blocked" / "experiment_8183_block.json"
    assert cli(tmp_path, "--fixture-output", blocked, "--mutation", "block").returncode == 0
    b = json.loads(blocked.read_text())
    assert b["honest_verdict"] == "complete_blocked_fit_trainable_score"
    assert b["energy_fit_ready_score"] == 0 and e.replay(blocked)
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/private.json").returncode == 2
    assert cli(tmp_path, "--mutation", "block").returncode == 2
    assert cli(tmp_path, "--date", "19990101").returncode == 2
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent.json").returncode == 1


def test_role_feature_and_missing_arm_guards(tmp_path):
    """SCENARIO-VERIFY-8183: E2E-015/019 leakage and degenerate signals fail closed."""
    rows, controls, protocol = e.fixture()
    n.validate_rows(rows)
    for change, error in [
        ("role", "future_role"),
        ("leak", "source_leak"),
        ("gold", "feature_schema"),
        ("constant", "degenerate_features"),
    ]:
        bad = deepcopy(rows)
        if change == "role":
            bad[0]["role"] = "evaluation"
        elif change == "leak":
            bad[-1]["source_cluster_id"] = bad[0]["source_cluster_id"]
        elif change == "gold":
            bad[0]["gold_label_feature"] = 1
        else:
            for r in bad:
                r["x"] = [0.0] * 16
        with pytest.raises(ValueError, match=error):
            n.validate_rows(bad)
    with pytest.raises(ValueError, match="missing_arm"):
        n.train(rows, controls[:1], protocol, tmp_path)
    assert n.select_comparator({"scalar_span": 1, "linear12": 1, "radial16": 1}) == "scalar_span"
    assert n.decision(0.1, [0.1, 0.5]) == "escalate"
    assert n.decision(None, [0.1, 0.5]) == "escalate"
    assert n.decision(0.9, [0.1, 0.5]) == "reject"
    assert n.decision(0.01, [0.1, 0.5]) == "accept"
    g = n.geometry(np.asarray([r["x"] for r in rows]), [r["unit_id"] for r in rows])
    assert n.design("linear12", np.asarray([rows[0]["x"]]), controls[1]["geometry"]).shape == (
        1,
        13,
    )
    assert n.design("local_max", np.asarray([rows[0]["x"]]), g).shape == (1, 2)


def upstream_fixture(tmp_path, monkeypatch):
    rows, heads, protocol = e.fixture()
    expanded = []
    for repetition in range(4):
        for r in rows:
            copy = deepcopy(r)
            copy.update(
                unit_id=r["unit_id"] + str(repetition),
                source_cluster_id=r["source_cluster_id"] + str(repetition),
            )
            copy["historical_paired_control"].update(
                unit_id=copy["unit_id"], source_cluster_id=copy["source_cluster_id"]
            )
            expanded.append(copy)
    rows = expanded
    features = [dict(r, slot=i + 1) for i, r in enumerate(rows)]
    originals = [r["historical_paired_control"] for r in rows]
    call = tmp_path / "primitive_calls.json"
    slots = tmp_path / "source_plan.json"
    matched = tmp_path / "matched_features.json"
    frozen = tmp_path / "frozen_heads.json"
    protocol_path = tmp_path / "protocol.json"
    for p, v in [
        (call, dict(rows=[])),
        (slots, dict(rows=[])),
        (matched, dict(rows=originals)),
        (frozen, dict(heads=heads)),
        (protocol_path, protocol),
    ]:
        atomic_json(p, v)
    values = {
        e.UPSTREAM: dict(
            experiment_id=8182,
            fit_trainable_score=1,
            feature_rows=features,
            raw_shard_hashes=[e.reference(call), e.reference(slots)],
            MODEL_SPECS=["historical_model"],
            model_invocation_counts=dict(generation_calls=99),
        ),
        e.METHODS: dict(
            experiment_id=8179,
            protocol_path=str(protocol_path),
            protocol_sha256=e.sha256_file(protocol_path),
            feature_schema=protocol["features"],
        ),
        e.CONTROL: dict(
            experiment_id=8154,
            raw_shard_hashes=[e.reference(matched)],
            frozen_head_manifest=e.reference(frozen),
            code_config_hashes=[e.reference(e.ROOT / e.historical.NUMERIC)],
        ),
    }
    for name, v in values.items():
        v.update(required_checks_passed=True, flagged_adversarial=False)
        atomic_json(tmp_path / name, v)
    monkeypatch.setattr(e, "PINS", {name: e.sha256_file(tmp_path / name) for name in values})
    monkeypatch.setattr(e.historical, "publication_sidecar", lambda v: tmp_path / "unused")
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=True)))
    reduced = dict(feature_rows=features, rows=[dict(r, slot=i + 1) for i, r in enumerate(rows)])
    monkeypatch.setattr(e.capture, "reduce", lambda *a: reduced)
    return values, reduced


def test_bound_inputs_missing_masks_and_external_operands(tmp_path, monkeypatch):
    """REQ-VERIFY-8183: reconstruct matched inputs; block changed operands explicitly."""
    values, reduced = upstream_fixture(tmp_path, monkeypatch)
    plan = e.inputs(tmp_path, tmp_path / "raw")
    assert all(c["passed"] for c in plan["checks"])
    assert len(plan["rows"]) == 192
    assert (
        plan["historical_model_provenance"]["imported_invocation_counts"]["generation_calls"] == 99
    )
    values[e.UPSTREAM]["feature_rows"] = reduced["feature_rows"] = reduced["feature_rows"][1:]
    atomic_json(tmp_path / e.UPSTREAM, values[e.UPSTREAM])
    monkeypatch.setitem(e.PINS, e.UPSTREAM, e.sha256_file(tmp_path / e.UPSTREAM))
    missing = e.inputs(tmp_path, tmp_path / "missing")
    assert missing["rows"][0]["x"] is None
    assert missing["rows"][0]["status"] == "excluded"
    values[e.UPSTREAM]["fit_trainable_score"] = 0
    atomic_json(tmp_path / e.UPSTREAM, values[e.UPSTREAM])
    monkeypatch.setitem(e.PINS, e.UPSTREAM, e.sha256_file(tmp_path / e.UPSTREAM))
    blocked = e.inputs(tmp_path, tmp_path / "gate_block")
    assert blocked["checks"][-1]["check"] == "fit_trainable_score"
    assert blocked["checks"][-1]["observed"] == 0
    (tmp_path / e.UPSTREAM).write_text("{}")
    assert e.inputs(tmp_path, tmp_path / "tamper")["checks"][-1]["check"] == "input_sha256"
    monkeypatch.setitem(e.PINS, e.UPSTREAM, e.sha256_file(tmp_path / e.UPSTREAM))
    bad = e.inputs(tmp_path, tmp_path / "structure")
    assert bad["checks"][-1]["check"] == "required_checks_passed"
    (tmp_path / e.UPSTREAM).write_text("{")
    monkeypatch.setitem(e.PINS, e.UPSTREAM, e.sha256_file(tmp_path / e.UPSTREAM))
    assert e.inputs(tmp_path, tmp_path / "malformed")["checks"][-1]["check"] == "input_structure"


def test_owned_failure_and_measurement_log_tamper(tmp_path, monkeypatch):
    """REQ-REPORT-8183: owned failures zero readiness; every custody class is checked."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, fixture_mode=True)
    good = [dict(passed=True)]
    value = e.build(work, raw, good, fixture=True)
    output = tmp_path / "experiment_8183_test.json"
    atomic_json(output, value)
    assert e.replay(output)
    log = tmp_path / "sealed.log"
    log.write_text("original")
    good[0].update(log_path=str(log), log_sha256=e.sha256_file(log))
    atomic_json(output, e.build(work, raw, good, fixture=True))
    assert e.replay(output)
    log.write_text("changed")
    assert not e.replay(output)
    good[0].clear()
    good[0]["passed"] = True
    work["result"]["rows"][0]["numerator"] = 99
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, raw, good, fixture=True))
    assert not e.replay(output)
    monkeypatch.setattr(
        e.energy, "train", lambda *a: (_ for _ in ()).throw(ValueError("owned_failure"))
    )
    failed = e.measure(tmp_path, tmp_path / "failed", fixture_mode=True)
    assert e.build(failed, tmp_path / "failed", good)["verdict_class"] == "disqualified"
    blocked = e.measure(tmp_path / "absent", tmp_path / "blocked")
    assert e.build(blocked, tmp_path / "blocked", good)["verdict_class"] == "blocked"


def test_runner_owned_receipts_and_preserved_history(tmp_path, monkeypatch):
    """REQ-REPORT-8183: real runner branches retain receipts and old primary bytes."""
    original = e.measure
    monkeypatch.setattr(
        e, "measure", lambda root, raw, **kw: original(root, raw, fixture_mode=True)
    )
    captured = []

    def check(root, spec, *args, **kwargs):
        if spec["name"] == "coverage_json":
            Path(spec["argv"][-1]).write_text("{}")
        return dict(passed=True, name=spec["name"], actual_exit=0)

    monkeypatch.setattr(e.execution, "run_check", check)
    monkeypatch.setattr(e.execution, "publish", lambda value, *args: captured.append(value))
    output = tmp_path / "experiment_8183_runner.json"
    output.write_text('{"historical":true}')
    assert e.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert captured[0]["energy_fit_ready_score"] == 1
    assert captured[0]["repository_health"]["passed"]
    assert any(
        Path(r["path"]).name == "changed_code_coverage.json"
        for r in captured[0]["raw_shard_hashes"]
    )
    assert (
        list((tmp_path / "raw").rglob("preserved_historical_primary.json"))[0].read_text()
        == '{"historical":true}'
    )
    specs = e.manifest(tmp_path, tmp_path / "candidate.json")
    for r in specs["commands"]:
        if r["name"] in ("ruff_check", "ruff_format", "strict_mypy"):
            assert not any("::" in a for a in r["argv"])
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]


def test_missing_masks_original_controls_and_exact_logistic(tmp_path):
    """REQ-VERIFY-8183: missing local signals escalate while historical inputs survive."""
    rows, controls, protocol = e.fixture()
    original = deepcopy(controls)
    rows[0].update(x=None, status="excluded", exclusion_reason="missing_local")
    rows[1].update(x=None, status="excluded", exclusion_reason="missing_local_and_control")
    rows[1]["historical_paired_control"].update(
        x=None, status="excluded", exclusion_reason="missing_control"
    )
    fitted = n.train(rows, controls, protocol, tmp_path)
    assert controls == original
    assert fitted["comparator_id"] == "radial16"
    result = n.evaluate(rows, fitted["heads"])
    local = next(
        r
        for r in result["rows"]
        if r["unit_id"] == rows[0]["unit_id"] and r["arm"] == n.NEW_ARMS[0]
    )
    old = next(
        r for r in result["rows"] if r["unit_id"] == rows[0]["unit_id"] and r["arm"] == "radial16"
    )
    assert local["action"] == "escalate" and local["numerator"] == 0.5
    assert local["status"] == "excluded" and local["denominator"] == 1
    assert old["status"] == "completed" and old["p"] is not None
    assert all(r["passed"] for r in result["energy_logistic_parity_rows"])
    with pytest.raises(ValueError, match="primitive_cost_drift"):
        with n.patch.object(n.reporting, "energy", n):
            n.reporting.independent_reduce([dict(local, numerator=88)])


def test_unfinished_calibration_is_never_ready(tmp_path, monkeypatch):
    """REQ-VERIFY-8183: an uncalibrated head remains failed owned work."""
    rows, controls, protocol = e.fixture()
    monkeypatch.setattr(
        n.base,
        "train",
        lambda *a: dict(
            heads=[dict(arm=n.NEW_ARMS[0])], failures=[dict(error="calibration_failure")]
        ),
    )
    fitted = n.train(rows, controls, protocol, tmp_path)
    assert "policy" not in fitted["heads"][0]
    assert fitted["failures"]
