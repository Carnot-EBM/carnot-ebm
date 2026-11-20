"""REQ-REPORT-9241-V787 / SCENARIO-REPORT-9241-V787: split consumer replay and V787 contract."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import historical_contract_inputs as historical
from carnot.reporting import protocol_owners as owners
from carnot.reporting import v782_contract as v782
from carnot.reporting import v783_contract as v783
from carnot.reporting import v784_contract as v784
from carnot.reporting import v785_contract as v785
from carnot.reporting import v786_capstone as v786_cap
from carnot.reporting import v787_contract as c
from carnot.reporting import v787_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_contract_inputs import authenticated, original_root


def child_cli(directory, *args, cli=None):
    """Private interpreters remove PYTHONPATH and retain actual child coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / (cli or e.CLI)), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9241] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=directory, env=env, text=True, capture_output=True, timeout=180
    )
    print("[test9241] child_after", result.returncode, flush=True)
    return result


def build_v787_fixture_tree(root: Path) -> tuple[Path, dict]:
    """Construct a complete 14-task V787 fixture tree in tmp_path for positive/mutation controls."""
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
        "| Order | Phase | Task ID | Deliverable |",
        "|---|---|---|---|",
    ]
    for index in range(14):
        if index < len(existing):
            task = deepcopy(existing[index])
        else:
            num = 9241 + index
            task = dict(
                id=f"exp{num}-task-{index + 1}",
                title=f"Exp{num}: Synthetic V787 fixture task {index + 1}",
                phase=2 if index < 9 else 3 if index < 12 else 4,
                track="research",
                priority="high",
                requires_gpu=False,
                max_turns=50,
                estimated_wall_time_min=45,
                per_unit_rows=True,
                milestone="2026.11.787",
                deliverable=f"results/experiment_{num}_v787_task_{index + 1}.json",
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
        milestone="2026.11.787",
        milestone_title=queue.get("milestone_title", "V787"),
        milestone_doc=e.DESIGN,
        tasks=tasks,
    )
    machine_doc = dict(milestone="2026.11.787", tasks=machine_tasks)
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
    return c.measure(e.ROOT, tmp_path_factory.mktemp("contract9241"), "20261120")


def test_large_historical_target_rejects_pollution(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9241-V787-CONTROLS: large private targets retain original hashes."""
    authority = tmp_path / "authority"
    authority.mkdir()
    source = authority / "large.json"
    source.write_bytes(b" " * 70000)
    config = authority / "openspec/change-proposals/v784-contract-inputs.json"
    atomic_json(config, dict(pins={}))
    work_file = authority / "work.json"
    atomic_json(
        work_file,
        dict(
            snapshots={"results/large.json": e.ref(source)},
            operands={},
            values={"0": dict(value=dict(code_config_hashes=[e.ref(config)]))},
        ),
    )
    capstone = authority / "results/capstone.json"
    atomic_json(capstone, dict(work_reference=e.ref(work_file)))
    monkeypatch.setattr(historical, "ROOT", authority)
    monkeypatch.setattr(historical, "CAPSTONES", {784: (capstone.name, e.ref(capstone)["sha256"])})
    root = original_root(tmp_path / "large", 784)
    target = root / "results/large.json"
    target.unlink()
    target.write_bytes(b"forged private evidence")
    with pytest.raises(ValueError, match="historical_input_drift"):
        original_root(root, 784)


def test_retained_cold_logs_reject_pollution(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9241-V787-CONTROLS: restoring missing logs never overwrites tampering."""
    terminal = tmp_path / "cold"
    monkeypatch.setattr(historical, "_COLD_TERMINAL", terminal)
    original_root(tmp_path / "inputs785", 785)
    original_root(tmp_path / "inputs785_again", 785)
    (terminal / "v785_research_complete.yaml").write_bytes(b"forged archive")
    with pytest.raises(ValueError, match="historical_input_drift"):
        original_root(tmp_path / "inputs785", 785)
    (terminal / "v785_research_complete.yaml").unlink()
    (terminal / "cold_replay.stdout").write_bytes(b"forged")
    with pytest.raises(ValueError, match="historical_input_drift"):
        original_root(tmp_path / "inputs", 784)


def test_historical_digest_invalidates_restored_mtime(tmp_path):
    """SCENARIO-REPORT-9241-V787-CONTROLS: restoring mtime cannot hide same-size tampering."""
    path = tmp_path / "evidence"
    path.write_bytes(b"true")
    reference = e.ref(path)
    assert historical.authenticated(reference) == b"true"
    stat = path.stat()
    path.write_bytes(b"fake")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="historical_input_drift"):
        historical.authenticated(reference)


def test_protocol_owners_all_four_and_false_rehash(tmp_path):
    """SCENARIO-REPORT-9241-V787: rebound evidence never owns inherited or V787 protocols."""
    assert set(owners.OWNERS) == {
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
        "selective_deferral_protocol",
    }
    assert set(owners.V787_OWNERS) == {
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
        "selective_deferral_protocol",
        "conformal_risk_protocol",
    }
    work_obj = dict(protocol_sidecars={})
    for protocol in owners.V787_OWNERS:
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
        owners.validate(dict(protocol_sidecars={}), "conformal_risk_protocol")


def test_historical_roots_v782_to_v786_and_pollution(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9241-V787: isolated V782-V786 roots reject successor pollution."""
    for generation, module in [(782, v782), (783, v783), (784, v784)]:
        root = original_root(tmp_path / str(generation), generation)
        assert module.e.authority(root)["ready"]
        if generation == 784:
            assert not (root / "results/experiment_9209_corpus_prefilter.json").exists()
            assert not (root / "results/experiment_9211_v784_label_ambiguity.json").exists()
            cap_root = original_root(tmp_path / "784_cap", 784, suite="capstone")
            assert (cap_root / "results/experiment_9209_corpus_prefilter.json").is_file()
            assert (cap_root / "results/experiment_9211_v784_label_ambiguity.json").is_file()
        (root / module.e.ACTIVE).chmod(0o600)
        (root / module.e.ACTIVE).write_bytes((e.ROOT / e.ACTIVE).read_bytes())
        assert not module.e.authority(root)["ready"]
        with pytest.raises(ValueError, match="historical_input_drift"):
            original_root(root, generation)
    v785_root = original_root(tmp_path / "785", 785)
    assert not (v785_root / "results/experiment_9223_corpus_prefilter.json").exists()
    assert not (
        v785_root / "results/experiment_9225_v785_selective_deferral_protocol.json"
    ).exists()
    v785_auth = v785.e.authority(v785_root)
    assert len(v785_auth["tasks"]) == 5 and len(v785_auth["rows"]) == 14
    (v785_root / v785.e.ACTIVE).chmod(0o600)
    (v785_root / v785.e.ACTIVE).write_bytes((e.ROOT / e.ACTIVE).read_bytes())
    with pytest.raises(ValueError, match="historical_input_drift"):
        original_root(v785_root, 785)
    v786_root = original_root(tmp_path / "786_cap", 786, suite="capstone")
    assert (v786_root / "results/experiment_9222_v785_contract.json").is_file()
    assert (v786_root / "results/experiment_9223_corpus_prefilter.json").is_file()
    assert (v786_root / "results/experiment_9225_v785_selective_deferral_protocol.json").is_file()
    assert (v786_root / "results/experiment_9240_v786_capstone.json").is_file()
    monkeypatch.chdir(tmp_path)
    v786_work = v786_cap.collect(tmp_path / "raw9240", "20261120", v786_root)
    assert all(r["passed"] for r in v786_work["checks"])
    (v786_root / "research-roadmap.yaml").chmod(0o600)
    (v786_root / "research-roadmap.yaml").write_bytes((e.ROOT / e.ACTIVE).read_bytes())
    with pytest.raises(ValueError, match="historical_input_drift"):
        original_root(v786_root, 786, suite="capstone")


def test_prior_protocol_rosters_remain_original(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9241-V787: new protocol owners cannot expand old replay requirements."""
    sidecars = {}
    for protocol in owners.V787_OWNERS:
        payload, _ = owners.load(protocol)
        path = tmp_path / (protocol + ".json")
        atomic_json(path, payload)
        sidecars[protocol + "_reference"] = e.ref(path)
    monkeypatch.setattr(v784, "OWNERS", dict(owners.V787_OWNERS))
    assert [r["protocol"] for r in v784.protocol_rows(dict(protocol_sidecars=sidecars))] == [
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
    ]
    assert [r["protocol"] for r in v785.protocol_rows(dict(protocol_sidecars=sidecars))] == [
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
        "selective_deferral_protocol",
    ]
    assert [r["protocol"] for r in c.protocol_rows(dict(protocol_sidecars=sidecars))] == [
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
        "selective_deferral_protocol",
        "conformal_risk_protocol",
    ]


def test_private_shard_commands_and_full_obligations(work):
    """SCENARIO-REPORT-9241-V787: six private shards preserve the deferred causal obligation."""
    scope = work["scope"]
    assert "python/carnot/reporting/v781_capstone.py" in scope["owned_changed_paths"]
    coverage_config = Path(scope["coverage_config_reference"]["path"]).read_text()
    assert str(e.ROOT / "python/carnot/reporting/v781_capstone.py") in coverage_config
    private_dirs = []
    for command in scope["commands"][:7]:
        argv = command["argv"]
        assert "--unset=PYTHONPATH" in argv
        private = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--chdir=")))
        assert private.is_dir() and not private.is_relative_to(e.ROOT)
        private_dirs.append(private)
        assert all(Path(a).is_absolute() for a in argv if a.endswith(".py"))
    assert len(set(private_dirs)) == 7
    assert [s["shard_name"] for s in scope["consumer_shards"]] == [
        "direct_consumers_core",
        "direct_consumers_v782",
        "direct_consumers_v783",
        "direct_consumers_v784",
        "direct_consumers_v785",
        "direct_consumers_v786",
    ]
    assert scope["deferred_validation_obligations"] == [
        dict(
            node_file="tests/python/test_causal_validation_9181.py",
            producer_reference=e.ref(e.ROOT / "tests/python/test_causal_validation_9181.py"),
            required_by="exp9243-forward-qualification",
            current_qualification_credit=0,
        )
    ]


def test_malformed_planner_identity_is_blocked(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9241-V787-CONTROLS: malformed external objects name a failed operand."""
    root, frozen = build_v787_fixture_tree(tmp_path / "malformed")
    value = yaml.safe_load((root / e.ACTIVE).read_bytes())
    value["tasks"][0]["id"] = 9241
    (root / e.ACTIVE).write_text(yaml.safe_dump(value))
    monkeypatch.setattr(e, "FROZEN", frozen)
    result = e.authority(root)
    assert result["ready"] is False
    assert any(
        row["artifact_field"] == "tasks[0]" and not row["passed"] for row in result["checks"]
    )


def test_missing_design_names_its_operand(tmp_path):
    """SCENARIO-REPORT-9241-V787-CONTROLS: missing files retain their own failed path."""
    result = e.authority(tmp_path)
    assert not result["ready"]
    assert result["checks"][-1]["path"] == str(tmp_path / e.DESIGN)
    assert result["checks"][-1]["observed"] == "missing_file"


@pytest.mark.parametrize("container", ["[]", "null", "7"])
def test_malformed_yaml_container_is_blocked(tmp_path, container):
    """SCENARIO-REPORT-9241-V787-CONTROLS: planner data must contain named task objects."""
    path = tmp_path / e.DESIGN
    path.parent.mkdir(parents=True)
    path.write_text("## Full machine-readable contract\n```yaml\n" + container + "\n```\n")
    result = e.authority(tmp_path)
    assert not result["ready"]
    assert any(not row["passed"] for row in result["checks"])


def test_shared_measurement_deadline_reserves_verification(monkeypatch):
    """REQ-REPORT-9241-V787: descendants leave time to verify and publish actual receipts."""
    deadlines = {}

    def child(name, *args, **kwargs):
        deadlines[name] = kwargs["deadline"]
        return dict(passed=True)

    def invoke(argv):
        for name in ["owned_tests", "cold_replay"]:
            c.legacy.lifecycle.child(name, [], Path("/tmp"), deadline=9999)
        return 0

    monkeypatch.delenv("CARNOT_V787_DEADLINE", raising=False)
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    monkeypatch.setattr(c.legacy.lifecycle, "child", child)
    monkeypatch.setattr(c, "_main", invoke)
    assert c.main(["--date", "20261120"]) == 0
    assert deadlines == dict(owned_tests=3300, cold_replay=4800)


def test_checkout_and_complete_contract_scope(work, tmp_path, monkeypatch):
    """REQ-REPORT-9241-V787: six split shards and method custody stay distinct from task count."""
    assert work["run_date"] == "20261120"
    inspection = work["inspection"]
    non_contract = [r for r in inspection["checks"] if r not in inspection["contract"]["checks"]]
    assert all(r["passed"] for r in non_contract), [r for r in non_contract if not r["passed"]]
    assert inspection["source_ready"] and inspection["static_ready"] and inspection["online_ready"]
    assert inspection["replay_ready"]
    assert len(inspection["historical_input_roots"]) == 7
    assert len(inspection["validation_obligations"]["node_bindings"]) == 674
    assert inspection["validation_obligations"]["union_count"] == 1040
    assert inspection["source"]["roles"] == dict(TRAIN=150, CAL=50, TEST=100)
    assert inspection["source"]["answer_count"] == 900
    assert [len(r["qids"]) for r in inspection["execution_rosters"]] == [100] * 3
    assert inspection["preserved_v785"]["exp9222_readiness_scores"] == dict(
        contract_ready_score=0,
        replay_authority_ready_score=1,
        source_ready_score=1,
        static_methods_ready_score=1,
        online_methods_ready_score=1,
    )
    assert inspection["preserved_v785"]["exp9225_scores"] == dict(
        selective_deferral_ready_score=1,
        spline_support_diagnostic_ready_score=1,
        scientific_effect_measured=False,
    )
    assert inspection["preserved_v786"]["exp9240_reconciliation_ready_score"] == 1
    assert inspection["preserved_v786"]["exp9240_honest_verdict"] == (
        "complete_blocked_declared_gate"
    )
    assert len(work["scope"]["consumer_shards"]) == 6
    for shard in work["scope"]["consumer_shards"]:
        assert "tests/python/test_causal_validation_9181.py" not in shard["test_files"]
        assert shard["coverage_shard_path"].endswith(".coverage." + shard["shard_name"])
    value = c.build(work, {})
    assert value["replay_authority_ready_score"] == 0
    assert value["contract_ready_score"] == 0
    assert value["MODEL_SPECS"] == []
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["claim_scope"]["independent_generalization_score"] == 0
    assert len(value["protocol_owner_rows"]) == 4
    assert len(value["consumer_shard_rows"]) == 6
    assert "ambiguity_masks_reference" in value["protocol_references"]
    assert "target_rubric_reference" in value["protocol_references"]
    assert "selective_deferral_masks_reference" in value["protocol_references"]
    assert "selective_deferral_protocol_reference" in value["protocol_references"]
    assert "spline_support_protocol_reference" in value["protocol_references"]
    assert "conformal_risk_protocol_reference" in value["protocol_references"]
    assert "ca_ocrc_protocol_reference" in value["protocol_references"]
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
    complete_root, frozen = build_v787_fixture_tree(tmp_path / "complete_v787")
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
    """SCENARIO-REPORT-9241-V787-CONTROLS: E2E-018/021 private CLI and interleaved replay."""
    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (e.NAME + ".json")
    result = child_cli(tmp_path, "--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    v785_work = v785.measure(original_root(tmp_path / "v785", 785), tmp_path / "raw785", "20261119")
    v785_source = tmp_path / "v785.json"
    atomic_json(v785_source, v785_work)
    v785_output = tmp_path / (v785.e.NAME + ".json")
    assert (
        child_cli(
            tmp_path,
            "--from-work",
            str(v785_source),
            "--output",
            str(v785_output),
            cli=v785.e.CLI,
        ).returncode
        == 0
    )
    assert child_cli(tmp_path, "--cold-replay", str(v785_output), cli=v785.e.CLI).returncode == 0
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    code = "\n".join(
        [
            "import sys; from pathlib import Path",
            "sys.path[:0] = " + repr([str(e.ROOT / "python"), str(e.ROOT)]),
            "from carnot.reporting import v787_contract as cur, v785_contract as prev",
            "assert cur.replay(Path(" + repr(str(output)) + "))['passed']",
            "assert prev.replay(Path(" + repr(str(v785_output)) + "))['passed']",
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
    """SCENARIO-REPORT-9241-V787-CONTROLS: missing evidence blocks and owner drift refuses."""
    changed = deepcopy(owners.V787_OWNERS)
    changed["conformal_risk_protocol"]["sha256"] = "sha256:" + "0" * 64
    monkeypatch.setattr(owners, "V787_OWNERS", changed)
    with pytest.raises(ValueError, match="protocol_owner_hash"):
        owners.load("conformal_risk_protocol")
    monkeypatch.undo()
    missing = c.measure(tmp_path / "absent", tmp_path / "raw", "20261120")
    value = c.build(missing, {})
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["censored_count"] == 14


def test_complete_object_mutations_and_collisions(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9241-V787-CONTROLS: new pins cannot bless changed gates or occupied IDs."""
    root, frozen = build_v787_fixture_tree(tmp_path / "tree")
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
    (root / "results/experiment_9242_existing.json").write_text("{}")
    assert not e.authority(root)["ready"]
    (root / e.ACTIVE).write_text("{}")
    assert e.authority(root)["checks"][-1]["observed"] == "missing_field"
    (root / e.ACTIVE).unlink()
    assert any(r["observed"] == "missing_file" for r in e.authority(root)["checks"])


def test_external_absence_and_protocol_failure_controls(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9241-V787-CONTROLS: absence blocks, forged sidecars and owner drift refuse."""
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
    changed["protocol_sidecars"]["conformal_risk_protocol_reference"] = e.ref(fake)
    with pytest.raises(ValueError, match="protocol_drift"):
        c.protocol_rows(changed)
    missing = deepcopy(owners.V787_OWNERS)
    missing["conformal_risk_protocol"]["path"] = str(tmp_path / "absent-owner")
    monkeypatch.setattr(owners, "V787_OWNERS", missing)
    monkeypatch.setattr(c, "V787_OWNERS", missing)
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
    """REQ-REPORT-9241-V787: six partitioned shards run under one parent supervisor and deadline."""
    seen = []

    def child(name, *args, **kwargs):
        seen.append((name, kwargs["deadline"]))
        return dict(name=name, passed=True, duration_s=1.25, exit_code=0)

    monkeypatch.setattr(c.legacy.lifecycle, "child", child)
    monkeypatch.setattr(c.legacy.lifecycle, "measured", lambda *args: dict(measured=True))
    value = c.qualify(work)
    assert len(value["checks"]) == 11 and value["owned_statement_coverage"]["measured"]
    assert value["repository_health"]["passed"]
    assert dict(seen) == dict(
        owned_tests=1500,
        direct_consumers_core=600,
        direct_consumers_v782=1200,
        direct_consumers_v783=1200,
        direct_consumers_v784=1200,
        direct_consumers_v785=1200,
        direct_consumers_v786=1200,
        ruff_check=600,
        ruff_format=600,
        strict_mypy=600,
        spec_coverage=600,
        global_python_suite=600,
    )
    assert all(r["scope_sha256"] == c.canonical_hash(work["scope"]) for r in value["checks"])
    monkeypatch.setattr(c, "measure", lambda *_: work)
    monkeypatch.setattr(c, "qualify", lambda _: value)
    monkeypatch.setattr(
        c.previous.previous.previous, "publish_primary", lambda *_: dict(passed=True)
    )
    assert c.main(["--date", "20261120", "--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    monkeypatch.setenv("CARNOT_V787_DEADLINE", "0")
    assert c.main(["--date", "20261120", "--output", str(tmp_path / (e.NAME + ".json"))]) == 1
