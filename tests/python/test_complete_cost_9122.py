"""REQ-REPORT-9122-V777; SCENARIO-REPORT-9122-V777.

Fixtures verify evidence accounting only, never model performance or benefit.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import complete_cost_9122 as d
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def primary(root: Path, identity: int, **extra: object) -> Path:
    """Make small external operands so the controls cannot load a real model."""
    path = root / d.INPUTS[identity][0]
    atomic_json(path, dict(experiment_id=identity, required_checks_passed=True) | extra)
    return path


def test_missing_inputs_are_terminal(tmp_path: Path) -> None:
    """REQ-REPORT-9122-V777: absence is not zero cost and needs no model load."""
    work = d.collect(tmp_path, tmp_path / "raw")
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["requested_inference_substrate_class"] == "model_load_no_generation"
    assert value["model_invocation_counts"]["model_loads_attempted"] == 0
    assert value["generated_tokens"] == value["current_llm_invocations"] == 0
    assert value["per_request_timing_rows"] == []
    assert value["energy_joules_or_null"] is None
    assert value["amdahl_ceiling_or_null"] is None
    assert all(c["observed"] == "missing_file" for c in work["checks"])
    assert value["effective_independent_groups"] == 0
    assert value["cost_inventory_ready_score"] == 1
    assert value["online_cost_ready_score"] == 0
    assert not value["acceptance_gates"]["complete_service"]
    bad = d.build(work, dict(passed=False, receipts=[], coverage={}), "20261116")
    assert bad["verdict_class"] == "disqualified"
    assert bad["cost_inventory_ready_score"] == 0


def test_typed_gates_and_independent_energy(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9122-V777: energy eligibility does not depend on embeddings."""
    primary(tmp_path, 9112, embeddings_ready_score=0, logits_ready_score=True)
    primary(tmp_path, 9117)
    primary(tmp_path, 9121, energy_audit_ready_score=1)
    work = d.collect(tmp_path, tmp_path / "raw")
    assert not work["eligibility"]["embedding"]
    assert work["eligibility"]["energy"]
    assert any(c["observed"] == 0 and not c["passed"] for c in work["checks"])
    assert any(c["observed"] == "missing_field" for c in work["checks"])
    assert any(c["observed"] is True and not c["passed"] for c in work["checks"])
    with pytest.raises(ValueError, match="qualified_service_requires_measurement"):
        d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")


def test_identity_hash_and_retained_snapshot(tmp_path: Path) -> None:
    """REQ-REPORT-9122-V777: original bytes survive checkout edits; tampering fails."""
    p = primary(tmp_path, 9117, static_predictions_ready_score=0)
    work = d.collect(tmp_path, tmp_path / "raw")
    frozen = work["references"][0]
    p.write_text("{}")
    assert d.read_reference(frozen)["experiment_id"] == 9117
    Path(frozen["path"]).chmod(0o600)
    Path(frozen["path"]).write_text("{}")
    with pytest.raises(ValueError, match="hash_drift"):
        d.read_reference(frozen)
    atomic_json(p, dict(experiment_id=1, static_predictions_ready_score=1))
    work = d.collect(tmp_path, tmp_path / "other")
    assert not work["eligibility"]["embedding"]
    assert any(c["check"] == "identity" and not c["passed"] for c in work["checks"])


def test_bound_input_refusal(tmp_path: Path) -> None:
    """REQ-REPORT-9122-V777: a missing original binding cannot use mutable code."""
    primary(
        tmp_path,
        9117,
        static_predictions_ready_score=1,
        code_config_hashes=[dict(path=str(tmp_path / "missing.py"), sha256="sha256:bad")],
    )
    work = d.collect(tmp_path, tmp_path / "raw")
    assert not work["eligibility"]["embedding"]
    assert any(c["check"] == "producer_binding" and not c["passed"] for c in work["checks"])


def test_replay_recomputes_aggregates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9122-V777: even rehashed invented results fail cold replay."""
    work = d.collect(tmp_path, tmp_path / "raw")
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert d.replay(path)["passed"]
    changed = deepcopy(value)
    changed["completed_count"] = 600
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(path, changed)
    assert not d.replay(path)["passed"]
    value["reproducibility_checksum"] = canonical_hash({})
    atomic_json(path, value)
    assert not d.replay(path)["passed"]
    assert not d.replay(tmp_path / "absent.json")["passed"]


def test_cli_foreign_success_refusal_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9122-V777: actual CLI children need no checkout environment."""
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
    value["energy_service_ready_score"] = 1
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1


def test_original_cohort_and_three_repetitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-9122-V777: retain first20 original TEST groups, all siblings once."""
    groups = [f"g{n}" for n in range(25)]
    units = [dict(unit_id=f"{g}/{s}", qid=f"q{g}", cluster=g) for g in groups for s in range(3)]
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
    assert {r["group"] for r in work["units"]} == set(groups[:20])
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    assert all(
        r["unit_id"] == r["original_unit_id"] for r in value["rows"] if r["scope"] == "science"
    )
    assert value["sample_size_budget"]["process_repetitions"] == 3
    assert value["effective_independent_groups"] == 0
    assert len(value["protocol"]["process_arm_order"]) == 3
    assert value["protocol"]["process_arm_order"] == [["embedding", "energy"]] * 3
    assert value["protocol"]["arm_order_seed"] == 776108
    assert value["unstarted_count"] == sum(r["scope"] == "science" for r in value["rows"])


def test_replay_binds_code_and_validation_logs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9122-V777: copied code and command logs remain byte-bound."""
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


def test_bound_hash_match_does_not_qualify_zero(tmp_path: Path) -> None:
    """REQ-REPORT-9122-V777: authentic producer code cannot turn a zero into readiness."""
    code = tmp_path / "retained.py"
    code.write_text("original")
    primary(tmp_path, 9117, static_predictions_ready_score=0, code_config_hashes=[d.ref(code)])
    work = d.collect(tmp_path, tmp_path / "raw")
    assert any(c["check"] == "producer_binding" and c["passed"] for c in work["checks"])
    assert not work["eligibility"]["embedding"]


def test_validation_uses_real_sqlite_and_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-9122-V777: measured coverage and the child exit must both pass."""
    from coverage import Coverage
    from carnot.reporting import complete_cost_runner_9122 as r
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


def test_main_qualification_and_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9122-V777: owned failure prevents any primary publication."""
    from carnot.reporting import complete_cost_runner_9122 as r

    original_root = d.ROOT
    for label in [*d.OWNED, *d.TESTS, "ops/exclusion_manifest.yaml"]:
        target = tmp_path / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original_root / label).read_bytes())
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(r, "validate", lambda *a: dict(passed=False, receipts=[], coverage={}))
    assert r.main(["--date", "20261116", "--sources", str(tmp_path)]) == 1
    output = tmp_path / "results" / (d.NAME + ".json")
    assert not output.exists()
    monkeypatch.setattr(r, "validate", lambda *a: dict(passed=True, receipts=[], coverage={}))
    calls = []

    def terminal(root: Path, commands: list, **kwargs: object) -> list:
        calls.extend(commands)
        return [dict(passed=True)]

    monkeypatch.setattr(r, "run_commands", terminal)
    assert r.main(["--date", "20261116", "--sources", str(tmp_path)]) == 0
    assert output.exists()
    assert [c.name for c in calls] == ["adversarial", "strict_rows"]


def test_inventory_rows_are_not_service_clocks(tmp_path: Path) -> None:
    """REQ-REPORT-9122-V777: primitive gate observations support accounting only."""
    work = d.collect(tmp_path, tmp_path / "raw")
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    administrative = [r for r in value["rows"] if r["scope"] == "administrative"]
    assert len(administrative) == len(work["checks"])
    assert all(
        r["absolute_metric"] == int(c["passed"])
        for r, c in zip(administrative, work["checks"], strict=True)
    )
    assert all(r["latency_s"] is None for r in administrative)
    assert value["completed_count"] == len(administrative)
    assert value["per_request_timing_rows"] == []


def test_malformed_external_primary_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9122-V777: malformed external evidence is a terminal block."""
    path = tmp_path / d.INPUTS[9121][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken")
    work = d.collect(tmp_path, tmp_path / "raw")
    assert not work["eligibility"]["energy"]
    assert any(c["check"] == "input_format" and not c["passed"] for c in work["checks"])
    assert (
        d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")["verdict_class"]
        == "blocked"
    )


def test_disqualified_upstream_cannot_qualify(tmp_path: Path) -> None:
    """REQ-REPORT-9122-V777: readiness alone cannot override failed upstream checks."""
    primary(tmp_path, 9121, energy_audit_ready_score=1, required_checks_passed=False)
    work = d.collect(tmp_path, tmp_path / "raw")
    assert not work["eligibility"]["energy"]
    assert any(
        c["artifact_field"] == "required_checks_passed" and not c["passed"] for c in work["checks"]
    )


def test_publication_rejects_changed_retained_input(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9122-V777: publication rehashes inputs even after collection."""
    primary(tmp_path, 9117, static_predictions_ready_score=0)
    work = d.collect(tmp_path, tmp_path / "raw")
    path = Path(work["references"][0]["path"])
    path.chmod(0o600)
    path.write_text("{}")
    with pytest.raises(ValueError, match="retained_hash_drift"):
        d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")


def test_failed_work_omits_completed_inventory_checks(tmp_path: Path) -> None:
    """REQ-REPORT-9122-V777: completed gate evaluation is not failed service work."""
    work = d.collect(tmp_path, tmp_path / "raw")
    value = d.build(work, dict(passed=True, receipts=[], coverage={}), "20261116")
    assert all(r["status"] != "completed" for r in value["failed_censored_work"])


def test_recovery_validation_has_bounded_time(tmp_path: Path) -> None:
    """REQ-REPORT-9122-V777: real recovery checks need a bounded usable budget."""
    from carnot.reporting import complete_cost_runner_9122 as r

    commands = r.plan(tmp_path / "checks")
    direct = next(c for c in commands if c.name == "direct_consumers")
    assert 600 <= direct.timeout_s <= 1200
    assert "tests/python/test_hard_exit_learning_qualification_8206.py" in direct.argv
