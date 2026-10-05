"""REQ-REPORT-8149 and REQ-VERIFY-8149: private evidence and terminal controls."""

from copy import deepcopy
import ctypes
import gc
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v704_capstone as c
from carnot.reporting import v704_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.verify.learning_audit_8144 import control_rows, statistics


def fixture(tmp_path):
    """Use private activated bytes and oracle pairs, never natural result paths."""
    root = tmp_path / "private"
    receipt = json.loads((c.ROOT / e.INPUT).read_bytes())
    tasks = receipt["task_contract"]
    snaps = {}
    for role in ("active", "design"):
        source = Path(receipt["authority_snapshots"][role]["snapshot_path"])
        target = root / (role + ".bin")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        snaps[role] = dict(snapshot_path=str(target), sha256=sha256_file(target))
    for n, task in enumerate(tasks[:-1], 8136):
        path = root / task["deliverable"]
        side = path.parent / "raw" / path.stem / "terminal.json"
        value = dict(
            experiment_id=n,
            task_id=task["id"],
            honest_verdict="complete_null_private",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(side),
            gate_check_summary=[],
            MODEL_SPECS=[],
            trained_head_specs=[],
            raw_shard_hashes=[],
        )
        value.update(
            {
                g["artifact_field"]: 1
                for t in tasks
                for g in t["gated_on"]
                if g["upstream"] == task["id"]
            }
        )
        if n == 8136:
            value.update(
                task_contract=tasks,
                authority_snapshots=snaps,
                canonical_tasks_sha256=receipt["canonical_tasks_sha256"],
            )
        if n == 8144:
            rows = control_rows(improved=False)
            primitive = side.parent / "primitive_rows.json"
            atomic_json(primitive, dict(rows=rows))
            value.update(
                audit_statistics=statistics(rows),
                raw_shard_hashes=[dict(path=str(primitive), sha256=sha256_file(primitive))],
            )
        pub = publish_primary(path, value, lambda _: dict(passed=True))
        atomic_json(side, dict(publication=pub))
    return root, tasks


def test_mixed_missing_disqualified_and_null(tmp_path):
    """SCENARIO-REPORT-8149-MIXED: branch absence cannot erase a supported null."""
    root, tasks = fixture(tmp_path)
    path = root / tasks[1]["deliverable"]
    value = json.loads(path.read_bytes())
    value.update(
        verdict_class="disqualified",
        required_checks_passed=False,
        honest_verdict="complete_disqualified_owned_validation",
    )
    atomic_json(path, value)
    (root / tasks[3]["deliverable"]).unlink()
    data = e.load(root, tmp_path / "raw")
    result = e.reduce(data)
    assert result["verdict_class"] == "blocked"
    assert len(result["task_dispositions"]) == result["completed_count"] == 14
    assert result["H1"]["status"] == "blocked"
    assert result["H2"]["status"] == "completed_null"
    assert result["H2"]["completed_count"] == 192
    assert result["independent_generalization_score"] == 0
    assert data["dispositions"][1]["verdict_class"] == "disqualified"
    assert data["dispositions"][3]["primary_honest_verdict"] == "<missing>"
    assert all(not r["retire_exact_configuration"] for r in result["retirement_decisions"])


@pytest.mark.parametrize("mutation", ["prompt", "table", "digest", "live"])
def test_authority_mutations(tmp_path, mutation):
    """SCENARIO-REPORT-8149-CUSTODY: full prompts and visible rows bind authority."""
    root, tasks = fixture(tmp_path)
    receipt = json.loads((root / e.INPUT).read_bytes())
    if mutation == "digest":
        receipt["canonical_tasks_sha256"] = "wrong"
    elif mutation == "live":
        changed = deepcopy(tasks)
        changed[0]["prompt"] += " changed"
        (root / "research-roadmap.yaml").write_text(
            yaml.safe_dump(dict(milestone="2026.10.704", tasks=changed))
        )
    else:
        role = "active" if mutation == "prompt" else "design"
        target = Path(receipt["authority_snapshots"][role]["snapshot_path"])
        if mutation == "prompt":
            value = yaml.safe_load(target.read_bytes())
            value["tasks"][0]["prompt"] += " changed"
            target.write_text(yaml.safe_dump(value))
        else:
            target.write_text(target.read_text().replace("| 1 |", "| 9 |", 1))
        receipt["authority_snapshots"][role]["sha256"] = sha256_file(target)
    atomic_json(root / e.INPUT, receipt)
    assert not e.load(root, tmp_path / "raw")["authority"]["activated"]


def test_primitive_drift_and_flagged_source(tmp_path):
    """SCENARIO-VERIFY-8149-BRANCHES: drift and flags exclude benefits."""
    root, tasks = fixture(tmp_path)
    path = root / tasks[8]["deliverable"]
    value = json.loads(path.read_bytes())
    value["audit_statistics"]["completed_count"] = 99
    pub = publish_primary(path, value, lambda _: dict(passed=True))
    atomic_json(Path(value["terminal_validation_sidecar_path"]), dict(publication=pub))
    data = e.load(root, tmp_path / "raw")
    assert not data["dispositions"][8]["eligible"]
    value["flagged_adversarial"] = True
    atomic_json(path, value)
    assert not e.load(root, tmp_path / "raw2")["dispositions"][8]["eligible"]


def test_conductor_skip_and_missing_authority(tmp_path):
    """SCENARIO-REPORT-8149-CUSTODY: administrative gate text is not a producer."""
    root, tasks = fixture(tmp_path)
    (root / tasks[3]["deliverable"]).unlink()
    atomic_json(
        root / "results/experiment_8139_skip.json",
        dict(
            schema="blocked_gate_check_v1",
            gate_check_summary="gate-unsat",
            honest_verdict="blocked_gate_check_failed",
            failed_field="source_protocol_ready_score",
            failed_upstream=tasks[1]["id"],
            failed_evidence_path=str(root / tasks[1]["deliverable"]),
            failed_evidence_sha256=None,
            failed_operator="==",
            failed_expected=1,
            failed_observed=0,
        ),
    )
    data = e.load(root, tmp_path / "raw")
    assert data["dispositions"][3]["disposition"] == "conductor_skip"
    assert data["dispositions"][3]["primary_honest_verdict"] == "<missing>"
    (root / e.INPUT).unlink()
    assert e.load(root, tmp_path / "raw2")["failures"]


def fake_execute(plan, raw):
    """Return real private log bytes while isolating expensive command orchestration."""
    receipts = []
    for spec in plan:
        log = raw / (spec.name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        if spec.name == "independent_reduction":
            assert c.main(list(spec.argv[3:])) == 0
        value = (
            dict(
                paper_ready=False,
                unmet_gates=["G2"],
                gates={k: dict(passed=k != "G2") for k in ["G1", "G2", "G3", "G4"]},
            )
            if spec.name == "publication_gate"
            else {}
        )
        atomic_json(log, value)
        receipts.append(
            dict(
                name=spec.name,
                scope=spec.scope,
                passed=True,
                normal_exit=True,
                log_path=str(log),
                log_sha256=sha256_file(log),
                command_argv=list(spec.argv),
                argv=list(spec.argv),
                expected_exit=0,
                actual_exit=0,
                duration_s=0.001,
            )
        )
    return receipts


@pytest.mark.parametrize("failed", [False, True])
def test_terminal_main_and_replay(tmp_path, monkeypatch, failed):
    """SCENARIO-REPORT-8149-CUSTODY: owned failure zeros readiness; replay binds logs."""
    root, tasks = fixture(tmp_path)
    monkeypatch.setattr(c, "execute", fake_execute)
    rejected = []

    def terminal(path):
        c.replay(path)
        rejected.append(path)
        return dict(passed=not failed or len(rejected) > 1)

    monkeypatch.setattr(c, "terminal", terminal)
    output = tmp_path / (c.NAME + ".json")
    assert c.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["capstone_execution_ready_score"] == int(not failed)
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert c.main(["--cold-replay", str(output)]) == 0
    assert c.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    value["completed_count"] = 15
    atomic_json(output, value)
    assert c.main(["--cold-replay", str(output)]) == 1
    value["completed_count"] = 14
    atomic_json(output, value)
    Path(value["validation_receipts"][0]["log_path"]).write_text("changed")
    with pytest.raises(ValueError, match="validation_log_hash_drift"):
        c.replay(output)


def test_terminal_runner_and_private_guards(tmp_path, monkeypatch):
    """REQ-REPORT-8149: terminal argv and private-mode production guards are fixed."""
    monkeypatch.setattr(c, "execute", fake_execute)
    assert c.terminal(tmp_path / "candidate.json")["passed"]
    assert (
        c.main(
            ["--fixture-e2e", "--root", str(c.ROOT), "--output", str(tmp_path / (c.NAME + ".json"))]
        )
        == 1
    )
    assert (
        c.main(
            [
                "--worker-input",
                "/tmp/missing8149",
                "--output",
                str(c.ROOT / "results" / (c.NAME + ".json")),
            ]
        )
        == 1
    )

    def failed_child(plan, raw):
        result = fake_execute(plan, raw)
        result[0]["passed"] = False
        return result

    root, _ = fixture(tmp_path)
    monkeypatch.setattr(c, "execute", failed_child)
    assert c.main(["--root", str(root), "--output", str(tmp_path / (c.NAME + ".json"))]) == 1


def test_real_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8149-CUSTODY: script-path success/block/mutation needs no PYTHONPATH."""
    root, tasks = fixture(tmp_path)
    (root / tasks[3]["deliverable"]).unlink()
    output = tmp_path / (c.NAME + ".json")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [str(c.ROOT / ".venv/bin/python"), "-u", str(c.ROOT / c.CLI)]
    result = subprocess.run(
        [*command, "--fixture-e2e", "--root", str(root), "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] and value["verdict_class"] == "blocked"
    replay = subprocess.run(
        [*command, "--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["rows"][0]["numerator"] = 9
    atomic_json(output, value)
    assert (
        subprocess.run(
            [*command, "--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=30,
        ).returncode
        == 1
    )
    original = list(sys.argv)
    sys.argv = [str(c.ROOT / c.CLI), "--cold-replay", str(output)]
    try:
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
        assert error.value.code == 1
    finally:
        sys.argv = original


def _natural_reductions(tmp_path):
    """REQ-VERIFY-8149: independently reopen real primaries already read by the summarizer."""
    data = e.load(c.ROOT, tmp_path / "natural")
    result = e.reduce(data)
    assert result["H2"]["status"] == "completed_null"
    assert result["H2"]["completed_count"] == 158
    assert len(result["H2"]["per_source_results"]) == 192
    assert result["H2"]["retention_class_counts"] == {"0": 31, "1": 30}
    assert result["service_evidence_scope"]["natural_host_qualified"]
    assert not result["service_evidence_scope"]["whole_service_qualified"]
    assert result["arc_evidence"]["reader_ready"] == 1
    assert result["arc_evidence"]["solve_credit"] == 0
    assert len(result["board_obligations"]) == 3
    assert all(not r["current_hardware_execution"] for r in result["board_obligations"])
    for number in [8145, 8146, 8148]:
        task = data["tasks"][number - 8136]["id"]
        assert data["audits"][task]["available"]
        value = deepcopy(data["primaries"][task])
        if number == 8145:
            value["reduction"]["completed_count"] += 1
        elif number == 8148:
            value["board_rows"][0]["custody_valid"] = False
        else:
            continue
        with pytest.raises(ValueError, match="primitive_reduction_drift"):
            e.primitive(value, number)
    del data, result, value
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def test_authenticated_natural_reductions(tmp_path):
    """REQ-VERIFY-8149: preserve assertions in a private, memory-isolated interpreter."""
    import coverage

    driver = tmp_path / "natural_driver.py"
    driver.write_text(
        "import sys, runpy\nfrom pathlib import Path\n"
        + f"sys.path[:0] = [{str(c.ROOT / 'python')!r}, {str(c.ROOT)!r}]\n"
        + f"runpy.run_path({str(c.ROOT / c.TEST)!r})['_natural_reductions'](Path({str(tmp_path)!r}))\n"
    )
    current = coverage.Coverage.current()
    child_data_path = tmp_path / ".coverage.natural"
    command = [str(c.ROOT / ".venv/bin/python")]
    if current:
        command += [
            "-m",
            "coverage",
            "run",
            "--data-file=" + str(child_data_path),
            "--include=" + ",".join(str(c.ROOT / p) for p in c.OWNED),
        ]
    result = subprocess.run(
        [*command, str(driver)], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if current:
        child_data = coverage.CoverageData(basename=str(child_data_path))
        child_data.read()
        current.get_data().update(child_data)


def test_missing_terminal_report(tmp_path):
    """SCENARIO-REPORT-8149-CUSTODY: stale or unsuccessful publication excludes a task."""
    root, tasks = fixture(tmp_path)
    path = root / tasks[0]["deliverable"]
    value = json.loads(path.read_bytes())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    sidecar = Path(terminal["publication"]["sidecar_path"])
    report = json.loads(sidecar.read_bytes())
    report["report"]["passed"] = False
    atomic_json(sidecar, report)
    assert not e.load(root, tmp_path / "raw")["dispositions"][0]["qualified"]


def test_archive_status_is_not_a_scientific_verdict(tmp_path):
    """SCENARIO-REPORT-8149-CUSTODY: retain exact archive and conductor statuses."""
    root, _ = fixture(tmp_path)
    old_id = "exp8113-radial-decision-fit"
    (root / "research-complete.yaml").write_text(
        yaml.safe_dump(
            dict(
                milestones=[
                    dict(tasks=[dict(id=old_id, title="Authentic old task", result="GATE_BLOCKED")])
                ]
            )
        )
    )
    (root / "ops").mkdir()
    (root / "ops/conductor-log.md").write_text(
        "| time | Authentic old task | GATE_BLOCK | upstream retired |\n"
    )
    prior = e.load(root, tmp_path / "raw")["prior_evidence"][old_id]
    assert prior["honest_verdict"] == "<missing>"
    assert prior["conductor_statuses"] == ["GATE_BLOCK"]
    assert prior["archive_results"] == ["GATE_BLOCKED"]


def test_replay_primitive_input_and_owned_fixture(tmp_path, monkeypatch):
    """REQ-REPORT-8149: fixture execution and primitive/hash drift are measured separately."""
    root, tasks = fixture(tmp_path)
    monkeypatch.setattr(c, "execute", fake_execute)
    monkeypatch.setattr(c, "terminal", lambda path: dict(passed=c.replay(path)["passed"]))
    output = tmp_path / (c.NAME + ".json")
    assert c.main(["--fixture-e2e", "--root", str(root), "--output", str(output)]) == 0
    with monkeypatch.context() as patch:
        patch.setattr(e, "primitive", lambda value, number: dict(available=False))
        with pytest.raises(ValueError, match="primitive_reduction_drift"):
            c.replay(output)
    value = json.loads(output.read_bytes())
    primitive = next(
        Path(r["path"])
        for r in value["raw_shard_hashes"]
        if Path(r["path"]).name == "primitive_rows.json"
    )
    primitive.write_text("changed")
    with pytest.raises(ValueError, match="input_hash_drift"):
        c.replay(output)
    path = root / tasks[4]["deliverable"]
    primary = json.loads(path.read_bytes())
    primary["honest_verdict"] = "unfinished"
    atomic_json(path, primary)
    assert (
        e.load(root, tmp_path / "raw2")["dispositions"][4]["honest_verdict"]
        == "complete_blocked_upstream_qualification"
    )


def test_production_worker_uses_private_scratch(tmp_path, monkeypatch):
    """REQ-REPORT-8149: production publication must not block its own private worker."""
    root, _ = fixture(tmp_path)
    monkeypatch.setattr(c, "ROOT", root)
    monkeypatch.setattr(c, "execute", fake_execute)
    monkeypatch.setattr(c, "terminal", lambda path: dict(passed=c.replay(path)["passed"]))
    assert c.main(["--root", str(root)]) == 0
    value = json.loads((root / "results" / (c.NAME + ".json")).read_bytes())
    worker_output = Path(value["measurement_exit_receipts"][0]["argv"][-1])
    assert not worker_output.is_relative_to(root / "results")


def test_frozen_pytest_parent_exists(tmp_path):
    """REQ-REPORT-8149: frozen validation is runnable before measurement begins."""
    plan = c.commands(tmp_path)
    focused = next(s for s in plan if s.name == "focused_pytest")
    target = next(a.split("=", 1)[1] for a in focused.argv if a.startswith("--basetemp="))
    assert Path(target).parent.is_dir()
