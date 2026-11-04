"""REQ-REPORT-8800 / REQ-VERIFY-8800: all slots and sealed private execution."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from carnot.reporting import v753_capstone as c


def fixture_root(root):
    """Use real authority bytes while all synthetic results stay in private scratch."""
    for name in [c.q.DESIGN, c.q.ACTIVE]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.ROOT / name, target)
    tasks = c.contract(root)["tasks"]
    log = root / "ops/conductor-log.md"
    log.parent.mkdir(parents=True)
    log.write_text(
        "| now | "
        + tasks[3]["title"][:48]
        + " | GATE_BLOCK | Pre-emptive skip: upstream retired (exp8790-delayed-memory-prototype) |\n"
    )
    return root


def valid_validation():
    """A unit input represents scoped checks; production obtains actual receipts."""
    return dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        union=dict(passed=True),
        ended_at_utc=c.now(),
    )


def seal(work, validation, output):
    """Freeze independent input copies before testing a candidate's reconstruction."""
    raw = Path(work["raw"])
    for name, value in [
        ("work", work),
        ("frozen_inputs", work),
        ("validation", validation),
        ("frozen_validation", validation),
    ]:
        c.atomic_json(raw / (name + ".json"), value)
    c.atomic_json(
        raw / "independent_seal.json",
        dict(
            primitive_reference=c.reference(raw / "work.json"),
            validation_reference=c.reference(raw / "validation.json"),
        ),
    )
    value = c.build(work, validation, output)
    c.atomic_json(output, value)
    return value


def test_accounting_and_rehashed_attacks(tmp_path):
    """SCENARIO-REPORT-8800-ACCOUNTING: a scheduling slot grants no scientific result."""
    root = fixture_root(tmp_path / "root")
    work = c.freeze(root, tmp_path / "raw", "20261104")
    work["publication_gates"] = dict(gates={}, paper_ready=False, unmet_gates=["G2"])
    output = tmp_path / (c.NAME + ".json")
    value = seal(work, valid_validation(), output)
    assert c.replay(output)
    assert len(value["rows"]) == value["completed_count"] == 12
    assert value["reconciliation_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["rows"][3]["scheduler_outcome"] == "conductor_cascade_skip"
    assert value["component_cost_ready_score"] == value["learning_benefit_ready_score"] == 0
    assert not value["publication_authorized"] and not value["activation_authorized"]
    for field in ["reconciliation_ready_score", "full_task_objects", "gate_edge_rows"]:
        changed = deepcopy(value)
        changed[field] = "forged"
        changed["reproducibility_checksum"] = c.digest(changed)
        c.atomic_json(output, changed)
        assert not c.replay(output)
    c.atomic_json(output, value)
    primitive = Path(work["raw"]) / "frozen_inputs.json"
    primitive.write_text("{}")
    assert not c.replay(output)
    failed = c.build(work, dict(valid_validation(), receipts=[dict(passed=False)]), output)
    assert failed["verdict_class"] == "disqualified"
    assert failed["reconciliation_ready_score"] == 0


def test_authority_rejection_and_operand_types(tmp_path):
    """REQ-REPORT-8800: no alias or coordinated hash can authenticate another task."""
    root = fixture_root(tmp_path / "root")
    design = root / c.q.DESIGN
    design.write_text(design.read_text().replace(c.q.PIN, "0" * 64))
    with pytest.raises(ValueError, match="full_object_authority"):
        c.contract(root)
    p = tmp_path / "array.json"
    p.write_text("[]")
    with pytest.raises(ValueError, match="producer_object"):
        c.load(c.reference(p))


def cli(*args):
    """SCENARIO-VERIFY-8800-REPLAY: invoke the thin script outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(c.ROOT / c.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_private_cli_and_failure(tmp_path):
    """REQ-VERIFY-8800: real CLI children publish checked bytes and preserve failure."""
    root = fixture_root(tmp_path / "root")
    for failed in [False, True]:
        output = tmp_path / str(failed) / (c.NAME + ".json")
        args = ["--root", root, "--output", output, "--private-control"]
        result = cli(*args, *(["--failed-child"] if failed else []))
        assert result.returncode == int(failed), result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert cli("--cold-replay", output).returncode == 0
    assert cli("--private-control").returncode == 2
    assert cli("--date", "bad").returncode == 2
    assert cli("--cold-replay", tmp_path / "absent").returncode == 1


def test_resource_failure_and_gate_missingness(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8800-RESOURCE: failed imports block without measurements."""
    root = fixture_root(tmp_path / "root")

    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr(c.importlib, "import_module", unavailable)
    work = c.freeze(root, tmp_path / "raw", "20261104")
    work["publication_gates"] = dict(gates={}, paper_ready=False, unmet_gates=["G2"])
    value = seal(work, valid_validation(), tmp_path / (c.NAME + ".json"))
    assert value["honest_verdict"] == "complete_blocked_resource"
    assert value["reconciliation_ready_score"] == 0
    assert value["n_samples"] == 0
    assert any(
        g["field"] == "dependency_imports_passed" and g["observed"] is False
        for g in value["gate_check_summary"]
    )
    ref = c.reference(tmp_path / "missing.json")
    assert c.observe(ref, "exp0-missing", "ready")["operand_state"] == "missing_file"


def test_current_sources_stay_quarantined_and_closed(tmp_path):
    """REQ-REPORT-8800: read genuine upstream bytes but write only private evidence."""
    root = fixture_root(tmp_path / "root")
    shutil.copyfile(c.ROOT / "ops/conductor-log.md", root / "ops/conductor-log.md")
    results = root / "results"
    results.mkdir()
    # Bound fixture size while retaining genuine current receipts and their assertions.
    for number in [8776, *range(8789, 8800)]:
        for source in (c.ROOT / "results").glob(f"experiment_{number}_*.json"):
            target = results / source.name
            shutil.copyfile(source, target)
            side = (
                source.parent
                / "raw"
                / source.stem
                / "validators"
                / (c.reference(source)["sha256"][7:] + ".json")
            )
            if side.is_file():
                report = json.loads(side.read_bytes())
                report["primary_path"] = str(target)
                c.atomic_json(results / "raw" / target.stem / "validators" / side.name, report)
    work = c.freeze(root, tmp_path / "raw", "20261104")
    work["publication_gates"] = dict(gates={}, paper_ready=False, unmet_gates=["G2"])
    value = seal(work, valid_validation(), tmp_path / (c.NAME + ".json"))
    assert c.replay(tmp_path / (c.NAME + ".json"))
    rows = {r["experiment_id"]: r for r in value["rows"]}
    assert rows[8789]["verdict_class"] == "disqualified"
    assert rows[8791]["excluded"] and rows[8791]["disposition"] == "quarantined_evidence"
    assert rows[8796]["verdict_class"] == "blocked"
    assert not any(r["headline_admitted"] for r in value["source_admission_rows"])
    assert any(
        g["field"] == "memory_protocol_ready_score" and g["observed"] == 0
        for g in value["gate_edge_rows"]
    )
    assert any(r["explicit_conductor_skip"] for r in value["retirement_recommendations"])
    assert all(not r["automatic_rerun_authorized"] for r in value["next_measurement_priorities"])
    assert value["n_samples"] == value["independent_count"] == 0


def test_production_orchestration_private_operands(tmp_path, monkeypatch):
    """REQ-VERIFY-8800: a full-health failure stays separate from scoped acceptance."""
    from carnot.reporting import v753_capstone_runner as runner

    root = fixture_root(tmp_path / "root")

    def bounded_child(name, argv, logs, **kwargs):
        if name == "full_python_suite":
            argv = [sys.executable, "-c", "raise SystemExit(1)"]
        return runner_child(name, argv, logs, **kwargs)

    runner_child = runner.child
    monkeypatch.setattr(runner, "child", bounded_child)
    monkeypatch.setattr(runner, "validate", lambda raw: valid_validation())
    output = tmp_path / (c.NAME + ".json")
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert not value["repository_health_reference"]["full_suite_pass"]
    assert value["required_checks_passed"]
    assert c.replay(output)


def test_private_resource_cli(tmp_path):
    """SCENARIO-VERIFY-8800-RESOURCE: a real isolated interpreter blocks missing imports."""
    root = fixture_root(tmp_path / "root")
    wrapper = tmp_path / "resource_control.py"
    output = tmp_path / (c.NAME + ".json")
    wrapper.write_text(
        "from carnot.reporting import v753_capstone as c\n"
        "from carnot.reporting.v753_capstone_runner import main\n"
        "def unavailable(name):\n    raise ImportError(name)\n"
        "c.importlib.import_module = unavailable\n"
        f"raise SystemExit(main(['--private-control', '--root', {str(root)!r}, '--output', {str(output)!r}]))\n"
    )
    result = subprocess.run(
        [sys.executable, str(wrapper)], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["honest_verdict"] == "complete_blocked_resource"
    assert value["reconciliation_ready_score"] == value["n_samples"] == 0
    assert value["preconditions_checked"]["source_hashes_checked"]
    assert (
        value["preconditions_checked"]["projected_memory_bytes"]
        >= value["preconditions_checked"]["measured_peak_rss_bytes"]
    )


def test_strict_validation_child(tmp_path):
    """REQ-VERIFY-8800: strict typing cannot inherit repository-wide disabled errors."""
    from carnot.reporting import v753_capstone_runner as runner

    source = tmp_path / "typed.py"
    source.write_text("def value() -> int:\n    return 'invalid'\n")
    checked = runner.validation_child(
        "strict_mypy",
        [str(c.ROOT / ".venv/bin/mypy"), "--strict", str(source)],
        tmp_path / "logs",
        expected=1,
    )
    assert checked["passed"] and checked["normal_exit"]
    assert "--config-file=/dev/null" in checked["argv"]
    checked = runner.validation_child("plain", [sys.executable, "-c", "pass"], tmp_path / "logs")
    assert checked["passed"]


def test_checkout_import_without_pythonpath(tmp_path):
    """SCENARIO-VERIFY-8800-CHECKOUT-IMPORT: fresh imports resolve this checkout's scripts."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from carnot.reporting import v753_capstone as c\n"
            "from carnot.reporting.v753_capstone_runner import main\n"
            "from scripts import conductor_gates\n"
            "assert callable(main)\n"
            "assert conductor_gates.__file__ == str(c.ROOT / 'scripts/conductor_gates.py')\n",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
