"""REQ-REPORT-9136-V778; SCENARIO-REPORT-9136-V778.

These fixtures test missing-evidence accounting, never scientific performance.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import complete_cost_9136 as d
from carnot.reporting.current_work_receipt import atomic_json


def primary(root: Path, identity: int, **fields: object) -> Path:
    """Keep fixture producer evidence separate from natural scientific inputs."""
    path = root / d.INPUTS[identity][0]
    atomic_json(path, dict(experiment_id=identity, required_checks_passed=True, **fields))
    return path


def test_missing_inputs_are_terminal(tmp_path: Path) -> None:
    """REQ-REPORT-9136-V778: no model operation or service clock follows absent evidence."""
    work = d.collect(tmp_path, tmp_path / "raw")
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["requested_inference_substrate_class"] == "model_load_no_generation"
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert value["model_invocation_counts"]["model_loads_attempted"] == 0
    assert value["cost_inventory_ready_score"] == 1
    assert value["service_cost_ready_score"] == value["adaptation_cost_ready_score"] == 0
    for key in [
        "paired_service_cost_rows",
        "scientific_parity_rows",
        "complete_adaptation_rows",
        "cold_load_rows",
    ]:
        assert value[key] == []
    assert value["amdahl_ceiling_or_null"] is value["energy_joules_or_null"] is None
    assert value["effective_independent_groups"] == 0
    assert any(c["observed"] == "missing_file" for c in value["gate_check_summary"])
    assert (
        d.build(work, dict(passed=False, receipts=[], coverage={}), "20261116")["verdict_class"]
        == "disqualified"
    )


def test_exact_fields_and_original_bindings(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9136-V778: zero, absent, bool and corrupted bytes stay different."""
    primary(tmp_path, 9127, logit_runtime_ready_score=0)
    primary(tmp_path, 9131, static_predictions_ready_score=True)
    code = tmp_path / "producer.py"
    code.write_text("original")
    primary(
        tmp_path,
        9133,
        online_predictions_ready_score=1,
        code_config_hashes=[dict(d.ref(code), sha256="sha256:original")],
    )
    work = d.collect(tmp_path, tmp_path / "raw")
    observed = [c["observed"] for c in work["checks"]]
    assert 0 in observed and "missing_field" in observed and True in observed
    assert not any(work["eligibility"].values())
    assert any(c["check"] == "producer_binding" and not c["passed"] for c in work["checks"])
    primary(tmp_path, 9131, static_predictions_ready_score=0, code_config_hashes=[d.ref(code)])
    primary(tmp_path, 9133, online_predictions_ready_score=0, experiment_alias="wrong")
    path = primary(tmp_path, 9130, admission_ready_score=1)
    atomic_json(path, dict(experiment_id=1, admission_ready_score=1))
    path = tmp_path / d.INPUTS[9125][0]
    path.write_text("{broken")
    work = d.collect(tmp_path, tmp_path / "other")
    assert any(c["check"] == "input_format" and not c["passed"] for c in work["checks"])
    assert any(c["check"] == "identity" and not c["passed"] for c in work["checks"])


def test_no_inventory_can_price_eligible_service(tmp_path: Path) -> None:
    """REQ-REPORT-9136-V778: static availability never requires online benefit."""
    assert d.eligibility({9124: True, 9125: True, 9127: True, 9130: True, 9131: True}) == dict(
        static=True, online=False
    )
    work = d.collect(tmp_path, tmp_path / "raw")
    work["eligibility"]["static"] = True
    with pytest.raises(ValueError, match="requires_measurement"):
        d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")


def test_replay_and_retained_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9136-V778: rehashing a fabricated headline cannot invent work."""
    primary(tmp_path, 9131, static_predictions_ready_score=0)
    work = d.collect(tmp_path, tmp_path / "raw")
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert d.replay(path)["passed"]
    changed = deepcopy(value)
    changed["completed_count"] = 999
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(path, changed)
    assert not d.replay(path)["passed"]
    value["reproducibility_checksum"] = "bad"
    atomic_json(path, value)
    assert not d.replay(path)["passed"]
    assert not d.replay(tmp_path / "absent")["passed"]
    frozen = Path(work["references"][0]["path"])
    frozen.chmod(0o600)
    frozen.write_text("{}")
    with pytest.raises(ValueError, match="hash_drift"):
        d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")


def test_cli_foreign_success_refusal_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9136-V778: E2E-018/021 private real children need no PYTHONPATH."""
    output = tmp_path / (d.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [str(d.ROOT / ".venv/bin/python"), str(d.ROOT / d.CLI)]

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv + list(args), cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )

    result = run("--date", "20261116", "--sources", str(tmp_path), "--private-output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert run("--cold-replay", str(output)).returncode == 0
    assert run("--date", "bad").returncode == 2
    assert run("--date", "20261116", "--private-output", str(d.ROOT / "bad.json")).returncode == 2
    value = json.loads(output.read_text())
    value["service_cost_ready_score"] = 1
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1


def test_original_cohort_and_unique_unit_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-9136-V778: first20 groups retain all siblings and three processes."""
    groups = [f"g{i}" for i in range(25)]
    units = [
        dict(unit_id=f"{g}/{s}", qid=f"q{g}", cluster=g, generator=str(s))
        for g in groups
        for s in range(3)
    ]
    paths = {}
    for kind, value in [
        ("manifest", dict(validated_rows=units)),
        ("roles", dict(partition=dict(roles=dict(test=groups)))),
    ]:
        path = tmp_path / (kind + ".json")
        atomic_json(path, value)
        paths[kind] = (path.name, d.ref(path)["sha256"])
    monkeypatch.setattr(d, "COHORT_INPUTS", paths)
    work = d.collect(tmp_path, tmp_path / "raw")
    assert len(work["units"]) == 60
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    rows = [r for r in value["rows"] if r["scope"] == "science"]
    assert len(rows) == len({r["unit_id"] for r in rows}) == 720
    assert {r["group"] for r in rows} == set(groups[:20])
    assert value["unstarted_count"] == value["scientific_request_counts"]["unstarted"] == 720
    assert all(r["latency_s"] is None for r in rows)
    assert value["protocol"]["arm_order_seed"] == 776108
    assert value["protocol"]["parity_atol"] == 1e-5
    assert value["protocol"]["parity_rtol"] == 1e-4


def test_replay_binds_code_and_logs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9136-V778: retained code and actual checker logs cannot drift."""
    work = d.collect(tmp_path, tmp_path / "raw")
    code, log = tmp_path / "code.py", tmp_path / "check.log"
    code.write_text("source")
    log.write_text("passed")
    work["sources"] = [d.ref(code)]
    atomic_json(tmp_path / "raw/work.json", work)
    work["work_reference"] = d.ref(tmp_path / "raw/work.json")
    validation = dict(
        passed=True,
        receipts=[dict(log_path=str(log), log_sha256=d.ref(log)["sha256"])],
        coverage={},
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, d.build(work, validation, "20261116"))
    assert d.replay(path)["passed"]
    code.write_text("changed")
    assert d.replay(path)["reason"] == "code_hash_drift"
    code.write_text("source")
    log.write_text("changed")
    assert d.replay(path)["reason"] == "validation_log_drift"


def test_validation_measures_real_sqlite_and_child_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-9136-V778: real child failure cannot qualify measured coverage."""
    from coverage import Coverage
    from carnot.reporting import complete_cost_runner_9136 as r
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    path = tmp_path / "fixture.py"
    path.write_text("value = 1\n")
    tracer = Coverage(data_file=str(tmp_path / ".coverage"), config_file=False)
    tracer.start()
    exec(compile(path.read_text(), str(path), "exec"), {})
    tracer.stop()
    tracer.save()
    monkeypatch.setattr(d, "OWNED", [str(path)])
    good = r.validate(
        [CommandSpec("fixture_exit", ("/bin/true",), "fixture", 10)], tmp_path, tmp_path / "good"
    )
    assert good["passed"] and good["coverage"]["percent"] == 100
    bad = r.validate(
        [CommandSpec("fixture_exit", ("/bin/false",), "fixture", 10)], tmp_path, tmp_path / "bad"
    )
    assert not bad["passed"]


def test_main_orchestration_rejects_owned_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9136-V778: publication follows actual replay and terminal children."""
    from carnot.reporting import complete_cost_runner_9136 as r
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    original = d.ROOT
    for label in [*d.OWNED, *d.TESTS, "ops/exclusion_manifest.yaml"]:
        target = tmp_path / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / label).read_bytes())
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(r, "validate", lambda *a: dict(passed=False, receipts=[], coverage={}))
    calls = []

    def children(root: Path, commands: list[CommandSpec], **kwargs: object) -> list[dict]:
        calls.extend(c.name for c in commands)
        # Fixture commands exercise supervision; production uses the frozen real argv.
        return run_commands(
            original,
            [CommandSpec(c.name, ("/bin/true",), "fixture", 10) for c in commands],
            log_dir=tmp_path / ("logs" + str(len(calls))),
        )

    monkeypatch.setattr(r, "run_commands", children)
    assert r.main(["--date", "20261116", "--sources", str(tmp_path)]) == 1
    output = tmp_path / "results" / (d.NAME + ".json")
    assert not output.exists()
    monkeypatch.setattr(r, "validate", lambda *a: dict(passed=True, receipts=[], coverage={}))
    assert r.main(["--date", "20261116", "--sources", str(tmp_path)]) == 0
    assert output.exists()
    assert calls == ["global_python_suite", "adversarial", "strict_rows"]
