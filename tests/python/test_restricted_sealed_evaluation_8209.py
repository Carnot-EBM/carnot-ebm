"""REQ-VERIFY-8209 / REQ-REPORT-8209: predictions precede evaluator access."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import restricted_sealed_evaluation_8209 as e
from carnot.verify import restricted_sealed_rule_8209 as n


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Real children prove imports and prediction custody outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private8209 CLI", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print("after private8209 CLI", result.returncode, flush=True)
    return result


def test_all_slots_and_missingness() -> None:
    """SCENARIO-VERIFY-8209-SEAL: missing slots escalate without changing128."""
    data = e.fixture()
    data["features"][0].update(x=None, status="failed", exclusion_reason="missing")
    data["roster"][0].update(status="failed", exclusion_reason="missing", numerator=0)
    data["comparator"][0].update(p=None, action="escalate")
    result = n.reduce(data)
    assert result["intended_count"] == 128
    assert result["completed_count"] == 127 and result["failed_count"] == 1
    assert len(result["prediction_rows"]) == 768
    missing = [r for r in result["prediction_rows"] if r["slot"] == 1]
    assert all(r["action"] == "escalate" and r["p_bad"] is None for r in missing)
    assert result["acceptance_subset_violations"] == []
    assert result["equivalent_logistic_parity"]["passed"]
    assert all(
        r["action"] != "accept" or r["baseline_action"] == "accept"
        for r in result["prediction_rows"]
    )
    assert all("head_sha256" in r and "energies" in r for r in result["prediction_rows"])


@pytest.mark.parametrize(
    "mutation",
    [
        "head",
        "source",
        "slot",
        "duplicate",
        "label",
        "derived",
        "feature",
        "baseline",
        "missingness",
    ],
)
def test_reject_custody(mutation: str) -> None:
    """SCENARIO-VERIFY-8209-REJECT: malformed or target-bearing evidence fails."""
    data = e.fixture()
    if mutation == "head":
        data["frozen"]["heads"][0]["weights"][0] = 3
    elif mutation == "source":
        data["features"][0]["source_cluster_id"] = "other"
    elif mutation == "slot":
        data["features"][0]["slot"] = 129
    elif mutation == "duplicate":
        data["comparator"][0] = data["comparator"][1]
    elif mutation == "label":
        data["features"][0]["nested"] = {"evaluator_label": 1}
    elif mutation == "derived":
        data["features"][0]["oracle_y"] = 1
    elif mutation == "feature":
        data["features"][0]["x"] = [float("nan")] * 16
    elif mutation == "baseline":
        data["comparator"][0]["p"] = 0.2
    else:
        data["features"][0]["status"] = "failed"
    with pytest.raises(ValueError):
        n.reduce(data)


def test_private_cli_replay_and_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8209-CLI: real seal, block, date rejection and cold replay."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--date", "20261006", "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["sealed_action_ready_score"] == 1
    assert value["evaluator_targets_opened"] is False
    assert value["verdict_class"] == "circular_positive"
    assert value["independent_generalization_score"] == 0
    assert value["model_invocation_counts"]["model_loads"] == 0
    assert value["label_access_ledger"][-1]["event"] == "predictions_sealed"
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    changed = deepcopy(value)
    changed["prediction_rows"][0]["p_bad"] = 0.2
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    blocked = tmp_path / "block" / output.name
    assert cli(tmp_path, "--fixture-output", blocked, "--mutation", "source").returncode == 0
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"
    assert cli(tmp_path, "--date", "20261005").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results" / output.name).returncode == 2


def test_owned_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8209-REJECT: owned policy violations disqualify readiness."""
    monkeypatch.setattr(n.rule, "action", lambda *a: "accept")
    work = e.measure(e.ROOT, tmp_path / "owned", fixture=True)
    value = e.build(work, tmp_path / "owned", [dict(passed=True)])
    assert value["verdict_class"] == "disqualified"
    assert value["sealed_action_ready_score"] == 0


def test_adversarial_permissions(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8209-SEAL: boundary, extreme and absent evidence stay safe."""
    rows = n.subset_probe()
    assert rows and all(r["passed"] for r in rows)
    for row in rows:
        assert row["action"] != "accept" or row["baseline_action"] == "accept"
    data = e.fixture()
    data["frozen"]["baseline"]["calibration"] = [-10.0, 1.0]
    data["frozen_content_sha256"] = canonical_hash(data["frozen"])
    for row in data["comparator"]:
        row.update(p=float(n.expit(-10)), action="accept")
    assert any(r["action"] == "accept" for r in n.reduce(data)["prediction_rows"])
    monkeypatch.setattr(n, "expit", lambda z: 0.4)
    with pytest.raises(ValueError, match="authentic_baseline"):
        n.reduce(data)
    monkeypatch.undo()
    original = n.rule.predict
    monkeypatch.setattr(n.rule, "predict", lambda *a: dict(original(*a), p=0.2))
    with pytest.raises(ValueError, match="energy_logistic_parity"):
        n.reduce(e.fixture())


def test_schema_and_external_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8209-REJECT: real cached primitives bind without evaluator access."""
    private = tmp_path / "validation"
    private.mkdir()
    e.manifest(private, tmp_path / "candidate.json")
    work = e.measure(e.ROOT, tmp_path / "natural")
    assert not work["owned_failure"] and all(r["passed"] for r in work["checks"])
    value = e.build(work, tmp_path / "natural", [dict(passed=True)])
    assert value["sealed_action_ready_score"] == 1
    assert (value["completed_count"], value["failed_count"], value["excluded_count"]) == (97, 30, 1)
    assert all(
        r["action"] == "escalate" for r in value["prediction_rows"] if r["status"] != "completed"
    )
    output = tmp_path / "natural.json"
    atomic_json(output, value)
    assert e.replay(output)
    assert (
        cli(
            tmp_path,
            "--root",
            tmp_path / "absent",
            "--worker-output",
            tmp_path / "worker/measurement.json",
        ).returncode
        == 0
    )
    monkeypatch.setattr(e.fit, "bind", lambda *a: {})
    failed = e.measure(e.ROOT, tmp_path / "invalid")
    assert e.build(failed, tmp_path / "invalid", [dict(passed=True)])["verdict_class"] == "blocked"
    monkeypatch.setattr(e.fit, "bind", lambda *a: [])
    assert e.measure(e.ROOT, tmp_path / "schema")["checks"][-1]["passed"] is False


def test_roster_and_comparator_failures() -> None:
    """SCENARIO-VERIFY-8209-REJECT: structural joins never silently lose a source."""
    for change in [
        "roster",
        "missingness",
        "roster_slot",
        "feature_schema",
        "feature_count",
        "comparator_source",
    ]:
        data = e.fixture()
        if change == "roster":
            data["roster"].pop()
        elif change == "missingness":
            data["roster"][0]["numerator"] = 0
        elif change == "roster_slot":
            data["roster"][0]["slot"] = 129
        elif change == "feature_schema":
            data["features"][0]["secret_metric"] = 1
        elif change == "feature_count":
            data["features"].pop()
        else:
            data["comparator"][0]["source_cluster_id"] = "other"
        with pytest.raises(ValueError):
            n.reduce(data)


def test_replay_rejects_rehashed_primitives(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8209-CLI: rehashed files cannot change authenticated origins."""
    private = tmp_path / "validation"
    private.mkdir()
    e.manifest(private, tmp_path / "candidate.json")
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    assert work["evidence"]
    output = tmp_path / "value.json"
    value = e.build(work, raw, [dict(passed=True)])
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(work)
    changed["evidence"]["frozen"]["heads"][0]["weights"][0] += 0.1
    changed["evidence"]["frozen_content_sha256"] = canonical_hash(changed["evidence"]["frozen"])
    atomic_json(raw / "measurement.json", changed)
    changed_value = e.build(changed, raw, [dict(passed=True)])
    atomic_json(output, changed_value)
    assert e.replay(output) is False
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, value)
    assert not e.replay(tmp_path / "absent.json")
    output.write_text("{}")
    assert e.replay(output) is False


def test_replay_failure_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8209-CLI: verify byte, receipt and recomputation failures."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    log = tmp_path / "log"
    log.write_text("complete stdout")
    receipts = [dict(passed=True, stdout_path=str(log), stdout_sha256=e.sha256_file(log))]
    output = tmp_path / "value.json"
    value = e.build(work, raw, receipts, fixture=True)
    atomic_json(output, value)
    assert e.replay(output)
    bad = deepcopy(value)
    bad["reproducibility_checksum"] = "invalid"
    atomic_json(output, bad)
    assert e.replay(output) is False
    atomic_json(output, value)
    log.write_text("modified")
    assert e.replay(output) is False
    log.write_text("complete stdout")
    prediction_path = Path(value["predictions_path"])
    original = prediction_path.read_bytes()
    prediction_path.chmod(0o644)
    prediction_path.write_text("{}")
    assert e.replay(output) is False
    changed_work = deepcopy(work)
    for ref in changed_work["raw_shard_hashes"]:
        if ref["path"] == str(prediction_path):
            ref["sha256"] = e.sha256_file(prediction_path)
    atomic_json(raw / "measurement.json", changed_work)
    atomic_json(output, e.build(changed_work, raw, receipts, fixture=True))
    assert e.replay(output) is False
    prediction_path.write_bytes(original)


def test_coherent_rewrite_cannot_replace_upstream(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8209-CLI: even rehashed consistent primitives bind source bytes."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    assert work["evidence"]
    data = work["evidence"]
    data["features"][0]["x"][15] += 0.1
    reduced = n.reduce(data)
    replacements = {
        "primitive_evidence": data,
        "independent_reduction": reduced,
        "sealed_predictions": dict(rows=reduced["prediction_rows"], labels_opened=False),
    }
    for name, content in replacements.items():
        atomic_json(raw / (name + ".json"), content)
    work["label_access_ledger"][-1]["predictions_sha256"] = e.sha256_file(
        raw / "sealed_predictions.json"
    )
    atomic_json(
        raw / "label_access_ledger.json",
        dict(rows=work["label_access_ledger"], evaluator_targets_opened=False),
    )
    work["raw_shard_hashes"] = [e.reference(Path(r["path"])) for r in work["raw_shard_hashes"]]
    atomic_json(raw / "measurement.json", work)
    output = tmp_path / "value.json"
    atomic_json(output, e.build(work, raw, [dict(passed=True)]))
    assert e.replay(output) is False


def test_subset_violation_and_serialized_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8209-REJECT: policy mutation cannot pass a sealed denominator."""
    monkeypatch.setattr(n.rule, "action", lambda *a: "accept")
    with pytest.raises(ValueError, match="acceptance_subset_violation"):
        n.reduce(e.fixture())
    monkeypatch.undo()
    private = tmp_path / "validation"
    private.mkdir()
    specs = e.manifest(private, tmp_path / "candidate.json")
    assert e.CLI in specs["terminal_commands"][0]["argv"][2]


def test_replay_source_pins_and_historical_hash(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8209-CLI: altered copies and provenance never replace origins."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    assert work["evidence"]
    output = tmp_path / "value.json"
    value = e.build(work, raw, [dict(passed=True)])
    historical = deepcopy(value)
    historical["historical_model_provenance"]["primitive_call_manifest"]["sha256"] = "invalid"
    historical.pop("reproducibility_checksum")
    historical["reproducibility_checksum"] = canonical_hash(historical)
    atomic_json(output, historical)
    assert e.replay(output) is False
    copied = next(r for r in work["refs"] if Path(r["path"]).name.endswith(Path(e.UPSTREAM).name))
    copied_path = Path(copied["path"])
    content = json.loads(copied_path.read_bytes())
    content["action_fit_ready_score"] = 0
    atomic_json(copied_path, content)
    copied["sha256"] = e.sha256_file(copied_path)
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, raw, [dict(passed=True)]))
    assert e.replay(output) is False


def test_missing_upstream_attestation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8209-REJECT: copied readiness needs a real publication receipt."""
    monkeypatch.setattr(
        e.fit,
        "bind",
        lambda *a: dict(
            required_checks_passed=True, flagged_adversarial=False, action_fit_ready_score=1
        ),
    )
    work = e.measure(e.ROOT, tmp_path / "missing-attestation")
    assert work["checks"][-1]["artifact_field"] == "input_structure"
    assert work["checks"][-1]["passed"] is False
    assert (
        e.build(work, tmp_path / "missing-attestation", [dict(passed=True)])["verdict_class"]
        == "blocked"
    )
