"""REQ-VERIFY-8196, REQ-REPORT-8196: private source and byte custody tests."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import selective_sealed_evaluation_8196 as e


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """A real caller outside checkout must not need ambient import paths."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private subprocess", flush=True)
    child = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after private subprocess", child.returncode, flush=True)
    return child


def test_private_cli_cold_and_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8196: private success, missing input and byte tamper."""
    output = tmp_path / "experiment_8196_fixture.json"
    child = cli(tmp_path, "--fixture-output", output)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["sealed_evaluation_ready_score"] == 1
    assert len(value["prediction_rows"]) == 128 * 7
    assert value["MODEL_SPECS"] == [] and value["call_ledger"] == []
    assert value["model_invocation_counts"]["live_model_calls"] == 0
    assert value["independent_generalization_score"] == 0
    assert value["evaluator_targets_opened"] is False
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    atomic_json(output, dict(value, completed_count=1))
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    frozen = Path(value["seal_receipt"]["heads"]["path"])
    saved = frozen.read_bytes()
    frozen.write_text("{}")
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    frozen.write_bytes(saved)
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent.json").returncode == 1
    blocked = tmp_path / "block/experiment_8196_block.json"
    assert cli(tmp_path, "--fixture-output", blocked, "--mutation", "block").returncode == 0
    b = json.loads(blocked.read_text())
    assert b["honest_verdict"] == "complete_blocked_selective_fit_ready_score"
    assert b["sealed_evaluation_ready_score"] == 0 and e.replay(blocked)
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/forbidden.json").returncode == 2
    assert cli(tmp_path, "--mutation", "block").returncode == 2
    assert cli(tmp_path, "--date", "19990101").returncode == 2


def test_permutation_masks_labels_and_comparator(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8196: no source order or missing case selects an arm."""
    work = e.measure(e.ROOT, tmp_path / "fixture", fixture_mode=True)
    data = work["evidence"]
    reduced = e.reduce(data)
    assert reduced["completed_count"] == 128
    assert reduced["equivalent_logistic_parity"]["passed"]
    permuted = deepcopy(data)
    permuted["features"].reverse()
    permuted["comparator"].reverse()
    assert e.reduce(permuted) == reduced
    missing = deepcopy(data)
    missing["features"][0].update(x=None, status="failed", exclusion_reason="missing_feature")
    result = e.reduce(missing)
    assert result["completed_count"] == 127 and len(result["missing_slot_rows"]) == 1
    assert all(r["action"] == "escalate" for r in result["prediction_rows"][:7])
    for field in ("y", "evaluator_label", "human_target"):
        bad = deepcopy(data)
        bad["features"][0][field] = 1
        with pytest.raises(ValueError, match="evaluator_label"):
            e.reduce(bad)
    for mutation in ("duplicate", "unmatched", "source", "head", "role"):
        bad = deepcopy(data)
        if mutation == "duplicate":
            bad["features"][0] = bad["features"][1]
        elif mutation == "unmatched":
            bad["comparator"].pop()
        elif mutation == "source":
            bad["comparator"][0]["source_cluster_id"] = "unmatched"
        elif mutation == "head":
            bad["frozen"]["heads"][0]["weights"][0] += 1
        else:
            bad["roles"]["reserved"][0]["unit_id"] = "unmatched"
        with pytest.raises(ValueError):
            e.reduce(bad)
    disqualified = e.build(work, tmp_path / "fixture", [dict(passed=False)], fixture=True)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["sealed_evaluation_ready_score"] == 0


def test_natural_and_external_gates(tmp_path: Path) -> None:
    """REQ-REPORT-8196: authenticate real upstream bytes without label access."""
    specs = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    assert all("::" not in a for s in specs["commands"][5:] for a in s["argv"])
    work = e.measure(e.ROOT, tmp_path / "natural")
    assert work["evidence"], work["checks"][-1]
    assert all(c["passed"] for c in work["checks"])
    output = tmp_path / "experiment_8196_natural.json"
    value = e.build(work, tmp_path / "natural", [dict(passed=True)])
    atomic_json(output, value)
    assert value["sealed_evaluation_ready_score"] == 1 and e.replay(output)
    assert value["completed_count"] == 97 and len(value["missing_slot_rows"]) == 31
    assert len(value["historical_model_provenance"]["call_ledger"]) == 155
    assert value["historical_model_provenance"]["original_baseline_call_manifest"][
        "sha256"
    ].startswith("sha256:")
    assert value["seal_receipt"]["labels_opened"] is False
    for suffix in ("-" + Path(e.UPSTREAM).name, "-role_manifest.json"):
        altered = deepcopy(work)
        ref = next(r for r in altered["refs"] if r["path"].endswith(suffix))
        copied = Path(ref["path"])
        original = copied.read_bytes()
        atomic_json(copied, dict(json.loads(original), forged=True))
        ref["sha256"] = sha256_file(copied)
        atomic_json(tmp_path / "natural/measurement.json", altered)
        atomic_json(output, e.build(altered, tmp_path / "natural", [dict(passed=True)]))
        assert not e.replay(output)
        copied.write_bytes(original)
    forged = deepcopy(work)
    forged["evidence"]["features"][0].update(x=None, status="failed", exclusion_reason="forged")
    rebuilt = e.reduce(forged["evidence"])
    for name, content in (
        ("primitive_evidence", forged["evidence"]),
        ("independent_reduction", rebuilt),
        ("sealed_predictions", dict(rows=rebuilt["prediction_rows"], labels_opened=False)),
    ):
        atomic_json(tmp_path / "natural" / (name + ".json"), content)
    forged["raw_shard_hashes"] = [e.reference(Path(r["path"])) for r in forged["raw_shard_hashes"]]
    atomic_json(tmp_path / "natural/measurement.json", forged)
    atomic_json(output, e.build(forged, tmp_path / "natural", [dict(passed=True)]))
    assert not e.replay(output)
    missing = e.measure(tmp_path, tmp_path / "missing")
    blocked = e.build(missing, tmp_path / "missing", [dict(passed=True)])
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][-1]["passed"] is False


def test_rejection_branches_and_low_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8196: arithmetic failures and low support stay explicit."""
    work = e.measure(e.ROOT, tmp_path / "fixture", fixture_mode=True)
    data = work["evidence"]
    for mutation in ("slot", "dimensions", "comparator_probability", "arms"):
        bad = deepcopy(data)
        if mutation == "slot":
            bad["features"][0]["slot"] = 2
        elif mutation == "dimensions":
            bad["features"][0]["x"] = [0.0]
        elif mutation == "comparator_probability":
            bad["comparator"][0]["p"] = None
        else:
            bad["frozen"]["arms"] = []
            bad["frozen_content_sha256"] = e.canonical_hash(bad["frozen"])
        with pytest.raises(ValueError):
            e.reduce(bad)
    missing = deepcopy(data)
    for row in missing["features"]:
        row.update(x=None, status="failed", exclusion_reason="missing_feature")
    work["evidence"] = missing
    low = e.build(work, tmp_path / "fixture", [dict(passed=True)], fixture=True)
    assert low["sealed_evaluation_ready_score"] == 1
    assert low["complete_pair_support_sufficient"] is False
    assert low["support_audit"]["insufficient_support_verdict"].startswith("complete_null_")
    assert all(r["action"] == "escalate" for r in low["prediction_rows"])
    predict = e.n.predict

    def energy_drift(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        rows = predict(*args, **kwargs)
        if not kwargs.get("scalar"):
            rows[0]["p"] += 0.01
        return rows

    monkeypatch.setattr(e.n, "predict", energy_drift)
    with pytest.raises(ValueError, match="energy_probability_parity"):
        e.reduce(data)

    def logistic_drift(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        rows = predict(*args, **kwargs)
        rows[5]["action"] = "accept"
        return rows

    monkeypatch.setattr(e.n, "predict", logistic_drift)
    with pytest.raises(ValueError, match="logistic_decision_parity"):
        e.reduce(data)
    monkeypatch.setattr(e.n, "predict", predict)
    with monkeypatch.context() as owned:
        owned.setattr(
            e, "reduce", lambda _: (_ for _ in ()).throw(ValueError("energy_probability_parity"))
        )
        failed = e.measure(e.ROOT, tmp_path / "owned_failure", fixture_mode=True)
        assert (
            e.build(failed, tmp_path / "owned_failure", [dict(passed=True)], fixture=True)[
                "verdict_class"
            ]
            == "disqualified"
        )
    monkeypatch.setattr(e, "fixture", lambda: (_ for _ in ()).throw(ValueError("evaluator_label")))
    malformed = e.measure(e.ROOT, tmp_path / "malformed", fixture_mode=True)
    assert malformed["checks"][-1]["observed"] == "evaluator_label"


def test_rehashed_primitives_and_logs(tmp_path: Path) -> None:
    """REQ-REPORT-8196: rehashing changed evidence cannot certify a forgery."""
    work = e.measure(e.ROOT, tmp_path / "fixture", fixture_mode=True)
    output = tmp_path / "experiment_8196_replay.json"
    log = tmp_path / "owned.log"
    log.write_text("normal exit\n")
    receipts = [dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))]
    value = e.build(work, tmp_path / "fixture", receipts, fixture=True)
    atomic_json(output, value)
    assert e.replay(output)
    log.write_text("modified\n")
    assert not e.replay(output)
    log.write_text("normal exit\n")
    shard = tmp_path / "fixture/independent_reduction.json"
    saved = shard.read_bytes()
    shard.write_text("{}")
    changed = deepcopy(value)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(shard):
            ref["sha256"] = sha256_file(shard)
    atomic_json(output, changed)
    assert not e.replay(output)
    shard.write_bytes(saved)
    primitive = tmp_path / "fixture/measurement.json"
    altered = json.loads(primitive.read_text())
    altered["evidence"]["features"][0]["source_cluster_id"] = "unmatched"
    atomic_json(primitive, altered)
    value["measurement_reference"]["sha256"] = sha256_file(primitive)
    atomic_json(output, value)
    assert not e.replay(output)
