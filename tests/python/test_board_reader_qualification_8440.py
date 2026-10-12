"""REQ-REPORT-8440 / REQ-VERIFY-8440: authentic readers grant no board speedup."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import board_reader_qualification_8440 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_roots_costs_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8440-ROOTS: reductions remain tied to independent operands."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["board_reader_ready_score"] == value["current_contract_ready_score"] == 1
    assert value["experiment_id"] == 8440 and value["run_date"] == "20261012"
    assert value["MODEL_SPECS"] == [] and value["device_commands_executed"] == []
    assert value["polarfire_workload_validated"]
    assert value["polarfire_graduation"]["dispatch_sha256"] == e.old.prior.DISPATCH_PIN
    assert value["kv260_supported_operations"][0]["k_max"] == 5
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert set(value) <= set(value["field_principles"])
    assert work["history_defects"]["missing_statements"] == {
        "python/carnot/reporting/board_operation_boundary_8413.py": [211],
        "python/carnot/reporting/board_operation_runner_8413.py": [59, 60, 61, 62],
    }
    assert work["branches"][0]["disposition"] == "qualified_primary_cell"
    atomic_json(output, value)
    assert e.replay(output)
    for key in ["board_reader_ready_score", "device_commands_executed", "operation_mapping"]:
        changed = deepcopy(value)
        changed[key] = "tampered"
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        atomic_json(output, changed)
        assert not e.replay(output)
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"
    Path(work["refs"][-1]["path"]).write_bytes(b"altered operand")
    atomic_json(output, value)
    assert not e.replay(output)


def test_absence_and_wrong_authority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8440-ROOTS: missing required operands block independently."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked" and value["board_reader_ready_score"] == 0
    atomic_json(output, value)
    assert e.replay(output) and not e.replay(tmp_path / "missing")
    with patch.object(e, "TASKS_PIN", "wrong"):
        assert not e.authority(e.ROOT, tmp_path / "authority")["activated"]
    mapped = e.operation_mapping([])
    assert all(row["cost_ns"] is None for row in mapped)


def test_cli_children_and_failures(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8440-CONTROLS: child shards cover actual CLI exit paths."""
    from carnot.reporting import board_reader_runner_8440 as r
    from carnot.reporting.primary_publication import reader_receipt

    cli, output = [sys.executable, "-u", str(e.ROOT / e.CLI)], tmp_path / (e.NAME + ".json")
    for argv, expected in [
        (["--private-e2e", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 1),
        (["--private-e2e"], 2),
        (["--date", "20261011"], 2),
    ]:
        ran = subprocess.run([*cli, *argv], capture_output=True, text=True, timeout=480)
        assert ran.returncode == expected, ran.stdout + ran.stderr
    assert reader_receipt(e.TASK, tmp_path, field="board_reader_ready_score")["passed"]
    failed = [
        dict(
            name="deliberate",
            argv=[sys.executable, "-c", "raise SystemExit(3)"],
            expected=0,
            deadline=10,
            scope="owned",
        )
    ]
    with patch.object(r, "manifest", return_value=failed):
        assert r.run(tmp_path / "absent", output, tmp_path) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert any("E2E018" in row["name"] for row in r.manifest(tmp_path))


def test_typed_cost_mapping() -> None:
    """SCENARIO-REPORT-8440-COSTS: host costs cannot become fabric capability."""
    rows = [dict(operation=op, cost_ns=1) for op in e.old.OPERATIONS]
    mapped = e.operation_mapping(rows)
    assert {row["operation"] for row in mapped} == set(e.OPERATION_GROUPS)
    assert all(not row["kv260_supported"] for row in mapped)
    assert next(row for row in mapped if row["operation"] == "scoring")["cost_ns"] == 1
    assert next(row for row in mapped if row["operation"] == "transport")["cost_ns"] == 1
    with pytest.raises(ValueError):
        e.operation_mapping([dict(operation="unknown", cost_ns=1)])


def test_original_missing_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-8440: execute the five named statements with real historical readers."""
    from carnot.reporting import board_operation_runner_8413 as r

    with e.historical_bindings(724), patch.object(e.old.base, "probe", return_value=None):
        result = e.old.operands(e.ROOT, tmp_path / "mismatched-dispatch")
    assert not result["reader_ready"]
    assert any(g["observed"] == "qualified_reader_dispatch_binding" for g in result["checks"])
    assert r.manifest(tmp_path)[-1]["name"] == "original_source_drift_and_tamper_controls"


def test_genuine_child_coverage_retention(tmp_path: Path) -> None:
    """REQ-VERIFY-8440: measured child databases survive the private scratch lifetime."""
    from carnot.reporting import board_reader_runner_8440 as r

    private = tmp_path / "private"
    private.mkdir()
    cov = str(e.ROOT / ".venv/bin/coverage")
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in [*e.OWNED, *e.old.OWNED[:2]])
        + "[report]\nexclude_lines=\n"
    )
    rc = "--rcfile=" + str(config)
    plan = [
        dict(
            name="real_error_child",
            argv=[cov, "run", rc, e.CLI, "--deliberate-error"],
            expected=1,
            deadline=60,
            scope="owned",
        ),
        dict(
            name="combine",
            argv=[cov, "combine", rc, "--keep"],
            expected=0,
            deadline=30,
            scope="owned",
        ),
        dict(
            name="json",
            argv=[cov, "json", rc, "-o", str(private / "coverage.json")],
            expected=0,
            deadline=30,
            scope="owned",
        ),
    ]
    output = tmp_path / (e.NAME + ".json")
    with patch.object(r, "manifest", return_value=plan):
        assert r.run(tmp_path / "absent", output, private) == 0
    value = json.loads(output.read_bytes())
    assert value["newly_owned_coverage"] and value["raw_shard_hashes"]
    assert list(output.parent.glob("raw/*/invocations/*/coverage_shards/.coverage*"))


def test_optional_rejections_and_operand_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8440-ROOTS: self-consistent hashes cannot authorize new operands."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    for mutation in [
        lambda w: w.update(reader_ready=False),
        lambda w: w["checks"][0].update(observed="changed"),
        lambda w: w["contract"].update(task_sha256="changed"),
        lambda w: w["history_defects"].update(missing_statements={}),
    ]:
        changed = deepcopy(work)
        mutation(changed)
        atomic_json(raw / "measurement.json", changed)
        atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
        assert not e.replay(output)
    malformed = tmp_path / e.OPTIONAL
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{}")
    assert not e.optional(tmp_path, raw, dict(refs=[]))["qualified"]
    with patch.object(e, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        assert not e.optional(e.ROOT, raw, dict(refs=[]))["qualified"]


def test_historical_cli_identity_and_nonexit(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8440-CONTROLS: a wrapper cannot authorize an unrelated script."""
    from carnot.reporting import board_reader_runner_8440 as r

    with pytest.raises(ValueError, match="historical_cli_identity"):
        r.historical_main(["wrong-script"])
    with patch.object(r.runpy, "run_path", return_value={}):
        with pytest.raises(RuntimeError, match="historical_cli_did_not_exit"):
            r.historical_main([str(e.ROOT / e.old.CLI)])


def test_historical_adapter_preserves_coverage_prefix() -> None:
    """REQ-VERIFY-8440: rooting preserves the genuine child's original coverage command."""
    from carnot.reporting import board_reader_runner_8440 as r

    observed = []
    original = lambda argv, **kwargs: observed.append(argv)
    prefix = [
        sys.executable,
        "-u",
        "-m",
        "coverage",
        "run",
        "--parallel-mode",
        "--data-file=/tmp/exp8440-private-coverage",
    ]
    r.bound_popen(original)([*prefix, str(e.ROOT / e.old.CLI), "--deliberate-error"])
    assert observed[-1][: len(prefix)] == prefix
    assert observed[-1][len(prefix)] == str(e.ROOT / e.CLI)
    r.bound_popen(original)(["/bin/echo", "unchanged"])
    assert observed[-1] == ["/bin/echo", "unchanged"]
