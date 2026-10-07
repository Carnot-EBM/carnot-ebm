"""REQ-VERIFY-8224 / REQ-REPORT-8224: corrected costs and qualified custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import utility_audit_execution_8224 as e


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """External cwd and absent PYTHONPATH exercise the installed script itself."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private8224 CLI", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print("after private8224 CLI", result.returncode, result.stdout, result.stderr, flush=True)
    return result


@pytest.fixture
def natural(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """SCENARIO-VERIFY-8224-REDUCTION: natural inputs are copied into private scratch."""
    raw = tmp_path / "natural"
    work = e.measure(e.ROOT, raw)
    assert all(c["passed"] for c in work["checks"]), work["checks"]
    assert not work["owned_failure"]
    return work, raw


def test_original_labels_and_missing_costs(natural: tuple[dict[str, Any], Path]) -> None:
    """SCENARIO-VERIFY-8224-REDUCTION: all slots and every arm retain explicit costs."""
    work, _ = natural
    result = work["diagnostics"]
    assert (
        result["intended_count"],
        result["completed_count"],
        result["failed_count"],
        result["excluded_count"],
    ) == (128, 97, 30, 1)
    assert len(result["rows"]) == 128 * 74
    assert result["H1"]["protocol"] == e.numeric.H1
    assert result["bootstrap_diagnostics"]["valid_draws"] == 10000
    assert result["bootstrap_diagnostics"]["random_seed"] == 7108223
    assert len(result["per_source_deltas"]) == 128
    assert all(r["numerator"] == 0.5 for r in result["rows"] if r["status"] != "completed")
    assert result["comparator_sha256"] == canonical_hash(work["evidence"]["comparator"])
    assert e.numeric.reduce(work["evidence"]) == result
    for field in ["clock", "protocol", "slots", "source", "bytes", "label", "prediction"]:
        data = deepcopy(work["evidence"])
        if field == "clock":
            data["clock"]["labels_opened_ns"] = 0
        elif field == "protocol":
            data["protocol"]["H1"]["seed"] = 1
        elif field == "slots":
            data["slots"].pop()
        elif field == "source":
            data["slots"][0]["source_id"] = "wrong"
        elif field == "bytes":
            data["slots"][0]["source_bytes"] = "00"
        elif field == "label":
            data["original_response_records"][0]["response"] = "wrong"
        else:
            data["predictions"][0]["action"] = "reject"
        with pytest.raises(ValueError):
            e.numeric.reduce(data)


def test_every_registered_gate(natural: tuple[dict[str, Any], Path]) -> None:
    """SCENARIO-VERIFY-8224-REDUCTION: shared calibration cannot waive H1 safeguards."""
    work, _ = natural
    rows = deepcopy(work["diagnostics"]["rows"])
    for row in rows:
        row.update(numerator=0.5, brier=0.1, false_accept=0, action="escalate")
        if row["arm"] == "energy_group":
            row.update(numerator=0, action="reject")
    comparator = work["evidence"]["comparator"]
    result = e.numeric.statistics(rows, comparator)
    assert result["H1"]["passed"] and result["h1_development_signal_score"] == 1
    assert result["H1"]["complete_class_support"] and result["per_source_deltas"]
    assert not e.numeric.statistics([], comparator)["H1"]["passed"]
    for arm, field, value, gate in [
        ("energy_group", "brier", 0.9, "primary_brier_increase"),
        ("energy_group", "false_accept", 1, "primary_extra_false_accepts"),
        ("energy_group", "numerator", 0.5, "lower_gain"),
        ("additive_group", "numerator", -1, "additive_group_cost_increase"),
        ("logistic_group", "numerator", -1, "logistic_group_cost_increase"),
    ]:
        changed = deepcopy(rows)
        for row in changed:
            if row["arm"] == arm:
                row[field] = value
        result = e.numeric.statistics(changed, comparator)
        assert not result["H1"]["operands"][gate]["passed"]
        assert result["h1_development_signal_score"] == 0
    with pytest.raises(ValueError, match="duplicate_source_arm"):
        e.numeric.statistics([*rows, rows[0]], comparator)
    with pytest.raises(ValueError, match="unpaired_source"):
        e.numeric.statistics(rows[1:], comparator)


def test_actual_cli_and_frozen_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8224-CLI: private replay and rehashed aggregate rejection are real."""
    output = tmp_path / (e.NAME + ".json")
    private = tmp_path / "validation"
    private.mkdir()
    before = (e.execution.e, e.execution.OWNED, e.upstream.CLI, e.upstream.TEST)
    plan = e.manifest(private, output)
    assert before == (e.execution.e, e.execution.OWNED, e.upstream.CLI, e.upstream.TEST)
    commands = {r["name"]: r["argv"] for r in plan["commands"]}
    assert e.TEST in commands["owned_unit_and_private_CLI"]
    assert "--fail-under=100" in commands["coverage_report"]
    assert "--files" in commands["spec_coverage"]
    assert "--strict" in commands["strict_mypy"]
    assert "-k" in commands["owned_unit_and_private_CLI"]
    assert "-k" in commands["owned_custody_failure_batch"]
    assert (
        "tests/python/test_restricted_decision_audit_8210.py" in commands["consumer_and_E2E015_019"]
    )
    cold = next(r for r in plan["terminal_commands"] if r["name"] == "cold_replay")
    assert str(e.ROOT / e.CLI) in cold["argv"]
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["utility_audit_ready_score"] == 1
    assert "descriptive_utility_residuals" not in value
    imported = {r["path"]: r["fields_imported"] for r in value["cited_upstream_artifacts"]}
    assert "frozen_comparator" in imported[str(e.ROOT / e.UPSTREAM)]
    assert "evidence.original_response_records" in imported[str(e.ROOT / e.LABELS)]
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["H1"]["passed"] = not value["H1"]["passed"]
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    blocked = tmp_path / "blocked" / output.name
    assert cli(tmp_path, "--root", tmp_path / "absent", "--fixture-output", blocked).returncode == 0
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"
    assert cli(tmp_path, "--cold-replay", blocked).returncode == 0
    assert cli(tmp_path, "--date", "19000101").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/forbidden.json").returncode == 2
    worker = tmp_path / "worker/measurement.json"
    assert cli(tmp_path, "--worker-output", worker).returncode == 0


def test_owned_and_external_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8224-CLI: execution failures cannot be reported as benefit nulls."""
    raw = tmp_path / "blocked"
    work = e.measure(e.ROOT, raw, mutation="source")
    value = e.build(work, raw, [dict(passed=True)])
    assert value["verdict_class"] == "blocked" and value["utility_audit_ready_score"] == 0
    assert value["gate_check_summary"][-1]["observed"] is None
    monkeypatch.setattr(e, "inputs", lambda *a: (_ for _ in ()).throw(KeyError("schema")))
    work = e.measure(e.ROOT, tmp_path / "schema")
    assert work["checks"][-1]["artifact_field"] == "authenticated_input_schema"
    monkeypatch.undo()
    monkeypatch.setattr(e.numeric, "reduce", lambda *a: (_ for _ in ()).throw(ValueError("owned")))
    raw = tmp_path / "owned"
    work = e.measure(e.ROOT, raw)
    value = e.build(work, raw, [dict(passed=True)])
    assert value["verdict_class"] == "disqualified" and value["utility_audit_ready_score"] == 0
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    assert e.build(work, raw, [dict(passed=False)])["h1_development_signal_score"] == 0


def test_replay_all_custody_paths(natural: tuple[dict[str, Any], Path], tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8224-REDUCTION: outer hashes do not authorize changed inner evidence."""
    work, raw = natural
    receipt = e.run_check(
        e.ROOT,
        dict(name="true", argv=["/bin/true"], expected_exit=0, deadline_s=5),
        tmp_path,
        raw / "logs",
    )
    output = tmp_path / (e.NAME + ".json")
    original = e.build(work, raw, [receipt])
    atomic_json(output, original)
    assert e.replay(output)
    assert not e.replay(tmp_path / "absent")
    for field in ["reproducibility_checksum", "raw_shard_hashes"]:
        changed = deepcopy(original)
        if field == "reproducibility_checksum":
            changed[field] = "wrong"
        else:
            changed[field][0]["sha256"] = "wrong"
            changed["reproducibility_checksum"] = canonical_hash(
                {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
            )
        atomic_json(output, changed)
        assert not e.replay(output)
    atomic_json(output, original)
    log = Path(receipt["stdout_path"])
    log.write_bytes(b"changed")
    assert not e.replay(output)
    log.write_bytes(b"")
    for mutation in ["primitive", "prediction", "pin", "reduction"]:
        changed = deepcopy(work)
        if mutation == "primitive":
            atomic_json(raw / "primitive_evidence.json", {})
            changed["raw_shard_hashes"][0] = e.reference(raw / "primitive_evidence.json")
        elif mutation == "prediction":
            changed["evidence"]["predictions"][0]["numerator"] = 999
            atomic_json(raw / "primitive_evidence.json", changed["evidence"])
            changed["raw_shard_hashes"][0] = e.reference(raw / "primitive_evidence.json")
        elif mutation == "pin":
            ref = next(r for r in changed["refs"] if r["upstream_path"] == str(e.ROOT / e.UPSTREAM))
            forged = tmp_path / "forged.json"
            atomic_json(forged, {})
            ref.update(path=str(forged), sha256=sha256_file(forged))
        else:
            changed["diagnostics"]["H1"]["passed"] = True
            atomic_json(raw / "independent_reduction.json", changed["diagnostics"])
            changed["raw_shard_hashes"][1] = e.reference(raw / "independent_reduction.json")
        atomic_json(raw / "measurement.json", changed)
        atomic_json(output, e.build(changed, raw, [receipt]))
        assert not e.replay(output), mutation
        for name, content in [
            ("primitive_evidence", work["evidence"]),
            ("independent_reduction", work["diagnostics"]),
            ("measurement", work),
        ]:
            atomic_json(raw / (name + ".json"), content)
    atomic_json(output, original)
    assert e.replay(output)


def test_missing_labels_and_frozen_child(
    natural: tuple[dict[str, Any], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8224-CLI: frozen replay runs outside checkout without PYTHONPATH."""
    work, raw = natural
    monkeypatch.setattr(
        e.numeric.qualified.historical.old.base.human, "target", lambda *a: ({"y": None}, {})
    )
    with pytest.raises(ValueError, match="original_annotation_not_qualified"):
        e.numeric.reduce(work["evidence"])
    monkeypatch.undo()
    private = tmp_path / "private"
    private.mkdir()
    output = tmp_path / (e.NAME + ".json")
    plan = e.manifest(private, output)
    cold = next(r for r in plan["terminal_commands"] if r["name"] == "cold_replay")
    assert cold["argv"][:3] == ["/usr/bin/env", "-u", "PYTHONPATH"]
    atomic_json(output, e.build(work, raw, [dict(passed=True)]))
    receipt = e.run_check(e.ROOT, cold, private, raw / "cold_logs")
    assert receipt["passed"] and receipt["cwd"] == str(private)
    assert receipt["stdout_sha256"] and receipt["stderr_sha256"]
    assert receipt["started_monotonic_ns"] < receipt["ended_monotonic_ns"]
