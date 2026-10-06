"""REQ-VERIFY-8210 / REQ-REPORT-8210: original costs and honest null custody."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import restricted_decision_audit_8210 as e
from carnot.verify import restricted_decision_rule_8210 as n


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Actual children check imports and exits without checkout environment help."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private8210 CLI", flush=True)
    started = time.monotonic_ns()
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print(
        json.dumps(
            dict(
                event="after_private8210_CLI",
                argv=argv,
                exit_code=result.returncode,
                started_monotonic_ns=started,
                ended_monotonic_ns=time.monotonic_ns(),
                stdout=result.stdout,
                stderr=result.stderr,
                stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest(),
            )
        ),
        flush=True,
    )
    return result


def test_original_costs_missing_and_identity() -> None:
    """SCENARIO-VERIFY-8210-REDUCTION: human targets and missing costs are independent."""
    data = e.fixture()
    result = n.reduce(data)
    assert result["intended_count"] == 128 and len(result["rows"]) == 768
    assert result["H1"]["passed"] is False
    assert result["equivalent_logistic_parity"]["passed"]
    missing = deepcopy(data)
    missing["sealed"]["features"][0].update(x=None, status="failed", exclusion_reason="missing")
    missing["sealed"]["roster"][0].update(status="failed", exclusion_reason="missing", numerator=0)
    missing["sealed"]["comparator"][0].update(p=None, action="escalate")
    missing["predictions"] = e.sealed.n.reduce(missing["sealed"])["prediction_rows"]
    reduced = n.reduce(missing)
    assert reduced["failed_count"] == 1 and reduced["completed_count"] == 127
    assert all(
        r["numerator"] == 0.5 and r["denominator"] == 1 for r in reduced["rows"] if r["slot"] == 1
    )
    for mutation in ("clock", "costs", "slots", "source", "bytes", "labels", "prediction"):
        changed = deepcopy(data)
        if mutation == "clock":
            changed["clock"]["labels_opened_ns"] = 0
        elif mutation == "costs":
            changed["costs"]["escalate"] = 0
        elif mutation == "slots":
            changed["slots"].pop()
        elif mutation == "source":
            changed["slots"][0]["source_id"] = "different"
        elif mutation == "bytes":
            changed["slots"][0]["source_bytes"] = b"other".hex()
        elif mutation == "labels":
            changed["original_response_records"][0]["response"] = "other"
        else:
            changed["predictions"][0]["denominator"] = 9
        with pytest.raises(ValueError):
            n.reduce(changed)


def test_registered_gates_and_distinct_risk_denominators() -> None:
    """SCENARIO-VERIFY-8210-REDUCTION: positive control cannot waive any H1 condition."""
    rows = n.reduce(e.fixture())["rows"]
    for row in rows:
        row.update(numerator=0.5, brier=0.1, false_accept=0, action="escalate")
        if row["arm"] in ("energy", "equivalent_logistic_identity"):
            row.update(numerator=0, action="reject")
    result = n.statistics(rows, "logistic")
    assert result["H1"]["passed"] and result["energy_increment_gain"]["mean_gain"] == 0.5
    assert result["policy_gain"]["mean_gain"] == 0.5
    for row in rows:
        if row["arm"] == "energy":
            row.update(numerator=0.5, action="escalate")
    assert not n.statistics(rows, "logistic")["H1"]["passed"]
    assert n.statistics([], "logistic")["H1"]["passed"] is False
    risk = result["acceptance_risk_rows"][0]
    assert risk["acceptance_coverage"]["denominator"] == 128
    assert risk["conditional_accepted_error"]["denominator"] == 0
    assert risk["conditional_accepted_error"]["interval"] == [None, None]


def test_private_cli_and_rehashed_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8210-CLI: real success, block and four cold mutation failures."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--date", "20261006", "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["action_audit_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["independent_generalization_score"] == 0
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for field in ("rows", "H1", "acceptance_risk_rows", "action_audit_ready_score"):
        changed = deepcopy(value)
        changed[field] = [] if isinstance(value[field], list) else 999
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    assert (
        cli(
            tmp_path, "--fixture-output", tmp_path / "block" / output.name, "--mutation", "source"
        ).returncode
        == 0
    )
    blocked = json.loads((tmp_path / "block" / output.name).read_bytes())
    assert blocked["verdict_class"] == "blocked" and blocked["action_audit_ready_score"] == 0
    assert cli(tmp_path, "--date", "20261005").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results" / output.name).returncode == 2


def test_natural_custody_and_owned_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8210: natural cached audit is ready; failed owned checks disqualify."""
    private = tmp_path / "validation"
    private.mkdir()
    e.manifest(private, tmp_path / "candidate.json")
    work = e.measure(e.ROOT, tmp_path / "natural")
    assert work["evidence"] and not work["owned_failure"]
    value = e.build(work, tmp_path / "natural", [dict(passed=True)])
    assert value["action_audit_ready_score"] == 1
    assert (value["completed_count"], value["failed_count"], value["excluded_count"]) == (97, 30, 1)
    output = tmp_path / "natural.json"
    atomic_json(output, value)
    assert e.replay(output)
    measurement = Path(value["measurement_reference"]["path"])
    for mutation in ("row", "original_label", "permission_mask", "denominator", "pin"):
        changed_work = deepcopy(work)
        data = changed_work["evidence"]
        if mutation == "original_label":
            data["original_response_records"][0]["labels"] = []
            data["original_response_records"][0]["quality"] = "changed"
        elif mutation == "permission_mask":
            data["predictions"][0]["allowed_actions"] = ["accept"]
        elif mutation == "denominator":
            data["predictions"][0]["denominator"] = 128
        elif mutation == "pin":
            changed_work["refs"][0]["sha256"] = "sha256:" + "0" * 64
        else:
            row = data["predictions"][0]
            row["action"] = "reject" if row["action"] == "accept" else "accept"
        atomic_json(measurement, changed_work)
        changed = deepcopy(value)
        changed["measurement_reference"] = e.reference(measurement)
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        rejected = cli(tmp_path, "--cold-replay", output)
        assert rejected.returncode == 1, mutation + rejected.stdout + rejected.stderr
    atomic_json(measurement, work)
    atomic_json(output, value)
    assert e.replay(output)
    monkeypatch.setattr(e, "OPTIONAL", "results/optional_missing_8210.json")
    optional = e.measure(e.ROOT, tmp_path / "optional")
    assert optional["evidence"] and "absent" in optional["optional_sibling_disposition"]
    monkeypatch.undo()
    assert (
        e.build(work, tmp_path / "natural", [dict(passed=False)])["verdict_class"] == "disqualified"
    )
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
    monkeypatch.setattr(
        n, "reduce", lambda data: (_ for _ in ()).throw(ValueError("owned_reducer"))
    )
    failed = e.measure(e.ROOT, tmp_path / "failed", fixture=True)
    assert (
        e.build(failed, tmp_path / "failed", [dict(passed=True)])["verdict_class"] == "disqualified"
    )


def test_failure_receipts_and_primitive_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8210-CLI: changed bytes, logs and malformed inputs fail closed."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    log = tmp_path / "stdout"
    log.write_text("actual saved output")
    receipts = [dict(passed=True, stdout_path=str(log), stdout_sha256=e.reference(log)["sha256"])]
    value = e.build(work, raw, receipts, fixture=True)
    output = tmp_path / "candidate.json"
    atomic_json(output, value)
    assert e.replay(output)
    log.write_text("tampered log")
    assert not e.replay(output)
    log.write_text("actual saved output")
    primitive = raw / "independent_reduction.json"
    original = primitive.read_bytes()
    atomic_json(primitive, {})
    assert not e.replay(output)
    changed = deepcopy(value)
    changed["raw_shard_hashes"] = [e.reference(Path(r["path"])) for r in value["raw_shard_hashes"]]
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert not e.replay(output)
    primitive.write_bytes(original)
    changed = deepcopy(value)
    changed["run_date"] = "bad"
    atomic_json(output, changed)
    assert not e.replay(output)
    output.write_text("not JSON")
    assert not e.replay(output)
    monkeypatch.setattr(
        e.fit,
        "bind",
        lambda *a: dict(
            sealed_action_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            evaluator_targets_opened=False,
        ),
    )
    assert e.measure(e.ROOT, tmp_path / "schema")["checks"][-1]["passed"] is False


def test_unpaired_and_duplicate_source_failures() -> None:
    """SCENARIO-VERIFY-8210-REDUCTION: an absent arm or duplicate cannot alter support."""
    rows = n.reduce(e.fixture())["rows"]
    with pytest.raises(ValueError, match="duplicate_source_arm"):
        n.statistics([*rows, rows[0]], "logistic")
    with pytest.raises(ValueError, match="unpaired_source"):
        n.statistics(rows[1:], "logistic")
