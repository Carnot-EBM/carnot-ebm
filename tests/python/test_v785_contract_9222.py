"""REQ-REPORT-9222-V785 / SCENARIO-REPORT-9222-V785: partitioned replay and V785 contract."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import protocol_owners as owners
from carnot.reporting import v782_contract as v782
from carnot.reporting import v783_contract as v783
from carnot.reporting import v784_contract as v784
from carnot.reporting import v785_contract as c
from carnot.reporting import v785_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_contract_inputs import authenticated, original_root


def child_cli(directory, *args, cli=None):
    """Private interpreters remove PYTHONPATH and retain actual child coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / (cli or e.CLI)), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9222] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=directory, env=env, text=True, capture_output=True, timeout=180
    )
    print("[test9222] child_after", result.returncode, flush=True)
    return result


def build_v785_fixture_tree(root: Path) -> tuple[Path, dict]:
    """Construct a complete 14-task V785 fixture tree in tmp_path for positive/mutation controls."""
    for name in e.COLLISION_INPUTS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((e.ROOT / name).read_bytes())
    queue = yaml.safe_load((e.ROOT / e.ACTIVE).read_bytes())
    existing = list(queue["tasks"])
    tasks = []
    machine_tasks = []
    table_lines = [
        "## Task contract",
        "",
        "| # | Phase | ID | Deliverable |",
        "|---|---|---|---|",
    ]
    for index in range(14):
        if index < len(existing):
            task = deepcopy(existing[index])
        else:
            num = 9222 + index
            task = dict(
                id=f"exp{num}-task-{index + 1}",
                title=f"Exp{num}: Synthetic fixture task {index + 1}",
                phase=2 if index < 9 else 3 if index < 12 else 4,
                track="research",
                priority="high",
                requires_gpu=False,
                max_turns=50,
                estimated_wall_time_min=45,
                per_unit_rows=True,
                milestone="2026.11.785",
                deliverable=f"results/experiment_{num}_v785_task_{index + 1}.json",
                inference_substrate_class="no_model_load",
                MODEL_SPECS=[],
                model="sonnet",
                agent_type="claude",
                gated_on=[],
                prompt=f"Work in {{project_root}} on {{date}} for Exp{num}.",
            )
        tasks.append(task)
        raw_prompt = task["prompt"].replace("{{", "{").replace("}}", "}")
        machine_tasks.append(dict(task, prompt=raw_prompt))
        table_lines.append(
            f"| {index + 1} | {task['phase']} | `{task['id']}` | `{task['deliverable']}` |"
        )
    active_doc = dict(
        milestone="2026.11.785",
        milestone_title=queue.get("milestone_title", "V785"),
        milestone_doc=e.DESIGN,
        tasks=tasks,
    )
    machine_doc = dict(milestone="2026.11.785", tasks=machine_tasks)
    design_text = (
        "\n".join(table_lines)
        + "\n\n## Full machine-readable contract\n\n```yaml\n"
        + yaml.safe_dump(machine_doc, sort_keys=False)
        + "```\n"
    )
    design_path = root / e.DESIGN
    design_path.parent.mkdir(parents=True, exist_ok=True)
    design_path.write_text(design_text)
    (root / e.ACTIVE).write_text(yaml.safe_dump(active_doc, sort_keys=False))
    (root / "results").mkdir(exist_ok=True)
    frozen = deepcopy(e.FROZEN)
    frozen["pins"][e.DESIGN] = e.ref(design_path)["sha256"]
    frozen["pins"][e.ACTIVE] = e.ref(root / e.ACTIVE)["sha256"]
    return root, frozen


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Real original sources establish custody, without any new model work."""
    return c.measure(e.ROOT, tmp_path_factory.mktemp("contract9222"), "20261118")


def test_protocol_owners_all_three_and_false_rehash(tmp_path):
    """SCENARIO-REPORT-9222-V785: rebound evidence never owns inherited or V785 protocols."""
    assert set(owners.OWNERS) == {
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
        "selective_deferral_protocol",
    }
    work_obj = dict(protocol_sidecars={})
    for protocol in owners.OWNERS:
        payload, owner = owners.load(protocol)
        path = tmp_path / (protocol + ".json")
        atomic_json(path, payload)
        work_obj["protocol_sidecars"][protocol + "_reference"] = e.ref(path)
        assert owners.validate(work_obj, protocol)["owning_configuration"] == owner
        changed = dict(payload, primary_gate_rescue=True)
        atomic_json(path, changed)
        work_obj["protocol_sidecars"][protocol + "_reference"] = e.ref(path)
        with pytest.raises(ValueError, match="protocol_drift"):
            owners.validate(work_obj, protocol)
        path.write_text('{"invented": true}')
        work_obj["protocol_sidecars"][protocol + "_reference"] = e.ref(path)
        with pytest.raises(ValueError, match="protocol_drift"):
            owners.validate(work_obj, protocol)
        path.unlink()
        with pytest.raises(ValueError, match="protocol_drift"):
            owners.validate(work_obj, protocol)
    with pytest.raises(ValueError, match="protocol_drift"):
        owners.validate(dict(protocol_sidecars={}), "selective_deferral_protocol")


def test_historical_roots_v782_v783_v784_and_pollution(tmp_path):
    """SCENARIO-REPORT-9222-V785: isolated V782/V783/V784 roots reject successor pollution."""
    for generation, module in [(782, v782), (783, v783), (784, v784)]:
        root = original_root(tmp_path / str(generation), generation)
        assert module.e.authority(root)["ready"]
        if generation == 784:
            assert not (root / "results/experiment_9209_corpus_prefilter.json").exists()
            assert not (root / "results/experiment_9211_v784_label_ambiguity.json").exists()
            cap_root = original_root(tmp_path / "784_cap", 784, suite="capstone")
            assert (cap_root / "results/experiment_9209_corpus_prefilter.json").is_file()
            assert (cap_root / "results/experiment_9211_v784_label_ambiguity.json").is_file()
            health = (
                cap_root
                / "results/raw/experiment_9221_v784_capstone/development/global_python_suite.stdout"
            )
            assert health.is_file()
            health.write_bytes(b"forged log")
            with pytest.raises(ValueError, match="historical_input_drift"):
                original_root(cap_root, 784, suite="capstone")
        (root / module.e.ACTIVE).chmod(0o600)
        (root / module.e.ACTIVE).write_bytes((e.ROOT / e.ACTIVE).read_bytes())
        assert not module.e.authority(root)["ready"]
        with pytest.raises(ValueError, match="historical_input_drift"):
            original_root(root, generation)


def test_v784_protocol_roster_remains_original(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9222-V785: new protocol owners cannot expand old replay requirements."""
    sidecars = {}
    for protocol in ["entropy_reference_protocol", "label_ambiguity_protocol"]:
        payload, _ = owners.load(protocol)
        path = tmp_path / (protocol + ".json")
        atomic_json(path, payload)
        sidecars[protocol + "_reference"] = e.ref(path)
    monkeypatch.setattr(v784, "OWNERS", dict(owners.OWNERS))
    assert [r["protocol"] for r in v784.protocol_rows(dict(protocol_sidecars=sidecars))] == [
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
    ]


def test_private_shard_commands_and_full_obligations(work):
    """SCENARIO-REPORT-9222-V785: private shards preserve every deferred causal obligation."""
    scope = work["scope"]
    private_dirs = []
    for command in scope["commands"][:4]:
        argv = command["argv"]
        assert "--unset=PYTHONPATH" in argv
        private = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--chdir=")))
        assert private.is_dir() and not private.is_relative_to(e.ROOT)
        private_dirs.append(private)
        assert all(Path(a).is_absolute() for a in argv if a.endswith(".py"))
    assert len(set(private_dirs)) == 4
    assert scope["deferred_validation_obligations"] == [
        dict(
            node_file="tests/python/test_causal_validation_9181.py",
            producer_reference=e.ref(e.ROOT / "tests/python/test_causal_validation_9181.py"),
            required_by="exp9224-forward-qualification",
            current_qualification_credit=0,
        )
    ]


def test_malformed_planner_identity_is_blocked(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9222-V785-CONTROLS: malformed external objects name a failed operand."""
    root, frozen = build_v785_fixture_tree(tmp_path / "malformed")
    value = yaml.safe_load((root / e.ACTIVE).read_bytes())
    value["tasks"][0]["id"] = 9222
    (root / e.ACTIVE).write_text(yaml.safe_dump(value))
    monkeypatch.setattr(e, "FROZEN", frozen)
    result = e.authority(root)
    assert result["ready"] is False
    assert any(
        row["artifact_field"] == "tasks[0]" and not row["passed"] for row in result["checks"]
    )


def test_checkout_and_complete_contract_scope(work, tmp_path, monkeypatch):
    """REQ-REPORT-9222-V785: partitioned replay and method custody stay distinct from task count."""
    inspection = work["inspection"]
    non_contract = [r for r in inspection["checks"] if r not in inspection["contract"]["checks"]]
    assert all(r["passed"] for r in non_contract), [r for r in non_contract if not r["passed"]]
    assert inspection["source_ready"] and inspection["static_ready"] and inspection["online_ready"]
    assert inspection["replay_ready"]
    assert len(inspection["validation_obligations"]["node_bindings"]) == 674
    assert inspection["validation_obligations"]["union_count"] == 1040
    assert inspection["source"]["roles"] == dict(TRAIN=150, CAL=50, TEST=100)
    assert inspection["source"]["answer_count"] == 900
    assert [len(r["qids"]) for r in inspection["execution_rosters"]] == [100] * 3
    assert inspection["preserved_v784"]["readiness_scores"] == [0] * 5
    assert inspection["preserved_v784"]["owned_tests_receipt"]["exit_code"] == 0
    assert inspection["preserved_v784"]["direct_consumers_receipt"]["exit_code"] == -9
    assert inspection["preserved_v784"]["direct_consumers_receipt"]["timed_out"] is True
    assert inspection["preserved_v784"]["exp9211_scores"] == dict(
        target_identifiability_ready_score=1, label_ambiguity_ready_score=1
    )
    assert len(work["scope"]["consumer_shards"]) == 3
    for shard in work["scope"]["consumer_shards"]:
        assert "tests/python/test_causal_validation_9181.py" not in shard["test_files"]
        assert shard["coverage_shard_path"].endswith(".coverage." + shard["shard_name"])
    value = c.build(work, {})
    assert value["replay_authority_ready_score"] == 0
    assert value["contract_ready_score"] == 0
    assert value["MODEL_SPECS"] == []
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["claim_scope"]["independent_generalization_score"] == 0
    assert len(value["protocol_owner_rows"]) == 3
    assert len(value["consumer_shard_rows"]) == 3
    assert "ambiguity_masks_reference" in value["protocol_references"]
    assert "target_rubric_reference" in value["protocol_references"]
    assert "selective_deferral_protocol_reference" in value["protocol_references"]
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    qualified = c.build(work, {})
    assert qualified["replay_authority_ready_score"] == 1
    assert (
        qualified["source_ready_score"]
        == qualified["static_methods_ready_score"]
        == qualified["online_methods_ready_score"]
        == 1
    )
    assert qualified["contract_ready_score"] == int(inspection["contract"]["ready"])
    complete_work = deepcopy(work)
    complete_root, frozen = build_v785_fixture_tree(tmp_path / "complete_v785")
    monkeypatch.setattr(e, "FROZEN", frozen)
    complete_contract = e.authority(complete_root)
    assert complete_contract["ready"]
    complete_work["inspection"]["contract"] = complete_contract
    complete_work["inspection"]["checks"] = complete_contract["checks"] + non_contract
    complete_val = c.build(complete_work, {})
    assert complete_val["verdict_class"] == "null"
    assert complete_val["contract_ready_score"] == 1
    assert complete_val["replay_authority_ready_score"] == 1


def test_private_cli_and_interleaved_replays(work, tmp_path):
    """SCENARIO-REPORT-9222-V785-CONTROLS: E2E-018/021 private CLI and interleaved replay."""
    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (e.NAME + ".json")
    result = child_cli(tmp_path, "--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    v784_work = v784.measure(original_root(tmp_path / "v784", 784), tmp_path / "raw784", "20261117")
    v784_source = tmp_path / "v784.json"
    atomic_json(v784_source, v784_work)
    v784_output = tmp_path / (v784.e.NAME + ".json")
    assert (
        child_cli(
            tmp_path,
            "--from-work",
            str(v784_source),
            "--output",
            str(v784_output),
            cli=v784.e.CLI,
        ).returncode
        == 0
    )
    assert child_cli(tmp_path, "--cold-replay", str(v784_output), cli=v784.e.CLI).returncode == 0
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    code = "\n".join(
        [
            "import sys; from pathlib import Path",
            "sys.path[:0] = " + repr([str(e.ROOT / "python"), str(e.ROOT)]),
            "from carnot.reporting import v785_contract as cur, v784_contract as prev",
            "assert cur.replay(Path(" + repr(str(output)) + "))['passed']",
            "assert prev.replay(Path(" + repr(str(v784_output)) + "))['passed']",
            "assert cur.replay(Path(" + repr(str(output)) + "))['passed']",
        ]
    )
    res = c.legacy.lifecycle.child(
        "interleaved_replays",
        ["env", "--unset=PYTHONPATH", "--chdir=" + str(tmp_path), sys.executable, "-c", code],
        tmp_path / "interleaved",
        deadline=240,
        heartbeat=30,
    )
    assert res["passed"], (
        Path(res["stdout_path"]).read_text() + Path(res["stderr_path"]).read_text()
    )
    value = json.loads(output.read_bytes())
    value["replay_authority_ready_score"] = 99
    value["reproducibility_checksum"] = c.legacy.checksum(value)
    atomic_json(output, value)
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 1
    assert child_cli(tmp_path, "--date", "bad").returncode == 1
    assert child_cli(tmp_path, "--from-work", str(source)).returncode == 1
    assert (
        child_cli(
            tmp_path, "--checks-only", "--root", str(tmp_path / "absent"), "--output", str(output)
        ).returncode
        == 0
    )


def test_missing_inputs_and_changed_owner(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9222-V785-CONTROLS: missing evidence blocks and owner drift refuses."""
    changed = deepcopy(owners.OWNERS)
    changed["selective_deferral_protocol"]["sha256"] = "sha256:" + "0" * 64
    monkeypatch.setattr(owners, "OWNERS", changed)
    with pytest.raises(ValueError, match="protocol_owner_hash"):
        owners.load("selective_deferral_protocol")
    monkeypatch.undo()
    missing = c.measure(tmp_path / "absent", tmp_path / "raw", "20261118")
    value = c.build(missing, {})
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["censored_count"] == 14


def test_complete_object_mutations_and_collisions(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9222-V785-CONTROLS: new pins cannot bless changed gates or occupied IDs."""
    root, frozen = build_v785_fixture_tree(tmp_path / "tree")
    monkeypatch.setattr(e, "FROZEN", deepcopy(frozen))
    assert e.authority(root)["ready"]
    original = yaml.safe_load((root / e.ACTIVE).read_bytes())
    for field in [
        "prompt",
        "id",
        "gated_on",
        "MODEL_SPECS",
        "inference_substrate_class",
        "order",
        "count",
    ]:
        value = deepcopy(original)
        if field == "order":
            value["tasks"].reverse()
        elif field == "count":
            value["tasks"].pop()
        else:
            value["tasks"][1][field] = "forged"
        (root / e.ACTIVE).write_text(yaml.safe_dump(value))
        changed_frozen = deepcopy(frozen)
        changed_frozen["pins"][e.ACTIVE] = e.ref(root / e.ACTIVE)["sha256"]
        monkeypatch.setattr(e, "FROZEN", changed_frozen)
        assert not e.authority(root)["ready"]
    monkeypatch.setattr(e, "FROZEN", deepcopy(frozen))
    (root / e.ACTIVE).write_text(yaml.safe_dump(original))
    (root / "results/experiment_9223_existing.json").write_text("{}")
    assert not e.authority(root)["ready"]
    (root / e.ACTIVE).write_text("{}")
    assert e.authority(root)["checks"][-1]["observed"] == "missing_field"
    (root / e.ACTIVE).unlink()
    assert any(r["observed"] == "missing_file" for r in e.authority(root)["checks"])


def test_external_absence_and_protocol_failure_controls(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9222-V785-CONTROLS: absence blocks, forged sidecars and owner drift refuse."""
    with pytest.raises(ValueError, match="historical_input_drift"):
        authenticated(dict(path=str(tmp_path / "missing"), sha256="sha256:" + "0" * 64))

    def unavailable(*args, **kwargs):
        raise ValueError("historical_input_drift:" + str(tmp_path / "missing"))

    monkeypatch.setattr(e, "original_root", unavailable)
    with monkeypatch.context() as context:
        context.setattr(e, "HISTORICAL_BASE", tmp_path, raising=False)
        result = e.inspect(e.ROOT)
    assert not result["source_ready"] and not result["replay_ready"]
    assert any(r["observed"] == "missing_file" for r in result["checks"])
    changed = deepcopy(work)
    fake = tmp_path / "false.json"
    atomic_json(fake, dict(invented=True))
    changed["protocol_sidecars"]["selective_deferral_protocol_reference"] = e.ref(fake)
    with pytest.raises(ValueError, match="protocol_drift"):
        c.protocol_rows(changed)
    missing = deepcopy(owners.OWNERS)
    missing["selective_deferral_protocol"]["path"] = str(tmp_path / "absent-owner")
    monkeypatch.setattr(owners, "OWNERS", missing)
    monkeypatch.setattr(c, "OWNERS", missing)
    rows = c.protocol_rows(work)
    assert rows[-1]["gate"]["observed"] == "missing_file"
    assert not rows[-1]["gate"]["passed"]
    monkeypatch.undo()
    tampered_work = deepcopy(work)
    tampered_work["protocol_owner_rows"][0]["original_hash"] = "sha256:" + "0" * 64
    source = tmp_path / "false-work.json"
    atomic_json(source, tampered_work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, dict(work_reference=e.ref(source)))
    with pytest.raises(ValueError, match="protocol_owner_drift"):
        c.replay(output)
    res = child_cli(tmp_path, "--cold-replay", str(output))
    assert res.returncode == 1 and "protocol_owner_drift" in res.stdout


def test_orchestration_and_partitioned_supervisor(work, tmp_path, monkeypatch):
    """REQ-REPORT-9222-V785: partitioned shards run under one parent supervisor and deadline."""
    seen = []

    def child(name, *args, **kwargs):
        seen.append((name, kwargs["deadline"]))
        return dict(name=name, passed=True, duration_s=1.25, exit_code=0)

    monkeypatch.setattr(c.legacy.lifecycle, "child", child)
    monkeypatch.setattr(c.legacy.lifecycle, "measured", lambda *args: dict(measured=True))
    value = c.qualify(work)
    assert len(value["checks"]) == 8 and value["owned_statement_coverage"]["measured"]
    assert value["repository_health"]["passed"]
    assert dict(seen) == dict(
        owned_tests=1500,
        direct_consumers_core=600,
        direct_consumers_v782_v783=1200,
        direct_consumers_v784=1200,
        ruff_check=600,
        ruff_format=600,
        strict_mypy=600,
        spec_coverage=600,
        global_python_suite=600,
    )
    assert all(r["scope_sha256"] == c.canonical_hash(work["scope"]) for r in value["checks"])
    monkeypatch.setattr(c, "measure", lambda *_: work)
    monkeypatch.setattr(c, "qualify", lambda _: value)
    monkeypatch.setattr(c.previous.previous, "publish_primary", lambda *_: dict(passed=True))
    assert c.main(["--date", "20261118", "--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    monkeypatch.setenv("CARNOT_V785_DEADLINE", "0")
    assert c.main(["--date", "20261118", "--output", str(tmp_path / (e.NAME + ".json"))]) == 1
