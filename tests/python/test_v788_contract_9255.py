"""REQ-REPORT-9255-V788 / SCENARIO-REPORT-9255-V788: split consumer replay, coverage-list parity and V788 contract."""

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
from carnot.reporting import v787_contract as v787
from carnot.reporting import v788_contract as c
from carnot.reporting import v788_contract_evidence as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_contract_inputs import authenticated, original_root


def child_cli(directory, *args, cli=None):
    """Private interpreters remove PYTHONPATH and retain actual child coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / (cli or e.CLI)), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9255] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=directory, env=env, text=True, capture_output=True, timeout=300
    )
    print("[test9255] child_after", result.returncode, flush=True)
    return result


def test_private_cli_budget_includes_publication_and_replay(tmp_path, monkeypatch):
    """REQ-REPORT-9255-V788: one private call includes checker and cold-replay phases."""
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert child_cli(tmp_path, "--date", "20261120").returncode == 0
    assert calls == [300]


def build_v788_fixture_tree(root: Path) -> tuple[Path, dict]:
    """Construct a complete 14-task V788 fixture tree in tmp_path for positive/mutation controls."""
    source_root = original_root(root.parent / (root.name + "_v788_src"), 788)
    for name in e.COLLISION_INPUTS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / name).read_bytes())
    queue = yaml.safe_load((source_root / e.ACTIVE).read_bytes())
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
            num = 9255 + index
            task = dict(
                id=f"exp{num}-task-{index + 1}",
                title=f"Exp{num}: Synthetic V788 fixture task {index + 1}",
                phase=2 if index < 9 else 3 if index < 12 else 4,
                track="research",
                priority="high",
                requires_gpu=False,
                max_turns=50,
                estimated_wall_time_min=45,
                per_unit_rows=True,
                milestone="2026.11.788",
                deliverable=f"results/experiment_{num}_v788_task_{index + 1}.json",
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
        milestone="2026.11.788",
        milestone_title=queue.get("milestone_title", "V788"),
        milestone_doc=e.DESIGN,
        tasks=tasks,
    )
    machine_doc = dict(
        milestone="2026.11.788",
        milestone_title=queue.get("milestone_title", "V788"),
        milestone_doc=e.DESIGN,
        tasks=machine_tasks,
    )
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
    return c.measure(
        original_root(tmp_path_factory.mktemp("inputs9255"), 788),
        tmp_path_factory.mktemp("contract9255"),
        "20261120",
    )


def test_large_historical_target_rejects_pollution(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9255-V788-CONTROLS: large private targets retain original hashes."""
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
    """SCENARIO-REPORT-9255-V788-CONTROLS: restoring missing logs never overwrites tampering."""
    terminal = tmp_path / "cold"
    monkeypatch.setattr(historical, "_COLD_TERMINAL", terminal)
    original_root(tmp_path / "inputs785", 785)
    original_root(tmp_path / "inputs785_again", 785)
    (terminal / "v785_research_complete.yaml").write_bytes(b"forged archive")
    with pytest.raises(ValueError, match="historical_input_drift"):
        original_root(tmp_path / "inputs785", 785)
    (terminal / "v785_research_complete.yaml").unlink()
    original_root(tmp_path / "inputs787", 787)
    original_root(tmp_path / "inputs787_again", 787)
    (terminal / "v787_research_complete.yaml").write_bytes(b"forged v787 archive")
    with pytest.raises(ValueError, match="historical_input_drift"):
        original_root(tmp_path / "inputs787", 787)
    (terminal / "v787_research_complete.yaml").unlink()
    (terminal / "cold_replay.stdout").write_bytes(b"forged")
    with pytest.raises(ValueError, match="historical_input_drift"):
        original_root(tmp_path / "inputs", 784)


def test_historical_digest_invalidates_restored_mtime(tmp_path):
    """SCENARIO-REPORT-9255-V788-CONTROLS: restoring mtime cannot hide same-size tampering."""
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
    """SCENARIO-REPORT-9255-V788: rebound evidence never owns inherited or V787/V788 protocols."""
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


def test_historical_roots_v782_to_v787_and_pollution(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9255-V788: isolated V782-V787 roots reject successor pollution."""
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
    v787_root = original_root(tmp_path / "787", 787)
    assert (v787_root / "results/experiment_9241_v787_contract.json").is_file()
    assert not (v787_root / "results/experiment_9242_corpus_prefilter.json").exists()
    assert not (v787_root / "results/experiment_9244_v787_conformal_risk_protocol.json").exists()
    v787_auth = v787.e.authority(v787_root)
    assert len(v787_auth["tasks"]) == 7 and len(v787_auth["rows"]) == 14
    v787_cap = original_root(tmp_path / "787_cap", 787, suite="capstone")
    assert (v787_cap / "results/experiment_9241_v787_contract.json").is_file()
    assert (v787_cap / "results/experiment_9242_corpus_prefilter.json").is_file()
    assert (v787_cap / "results/experiment_9244_v787_conformal_risk_protocol.json").is_file()
    (v787_root / v787.e.ACTIVE).chmod(0o600)
    (v787_root / v787.e.ACTIVE).write_bytes((e.ROOT / e.ACTIVE).read_bytes())
    with pytest.raises(ValueError, match="historical_input_drift"):
        original_root(v787_root, 787)


def test_prior_protocol_rosters_remain_original(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9255-V788: new protocol owners cannot expand old replay requirements."""
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
    assert [r["protocol"] for r in v787.protocol_rows(dict(protocol_sidecars=sidecars))] == [
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
        "selective_deferral_protocol",
        "conformal_risk_protocol",
    ]
    assert [r["protocol"] for r in c.protocol_rows(dict(protocol_sidecars=sidecars))] == [
        "entropy_reference_protocol",
        "label_ambiguity_protocol",
        "selective_deferral_protocol",
        "conformal_risk_protocol",
    ]


def test_private_shard_commands_and_full_obligations(work):
    """SCENARIO-REPORT-9255-V788: seven private shards preserve the deferred causal obligation."""
    scope = work["scope"]
    assert scope["owned_changed_paths"] == list(e.OWNED)
    coverage_config = Path(scope["coverage_config_reference"]["path"]).read_text()
    for owned_path in scope["owned_changed_paths"]:
        assert str(e.ROOT / owned_path) in coverage_config
    private_dirs = []
    for command in scope["commands"][:8]:
        argv = command["argv"]
        assert "--unset=PYTHONPATH" in argv
        private = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--chdir=")))
        assert private.is_dir() and not private.is_relative_to(e.ROOT)
        private_dirs.append(private)
        assert all(Path(a).is_absolute() for a in argv if a.endswith(".py"))
    assert len(set(private_dirs)) == 8
    assert [s["shard_name"] for s in scope["consumer_shards"]] == [
        "direct_consumers_core",
        "direct_consumers_v782",
        "direct_consumers_v783",
        "direct_consumers_v784",
        "direct_consumers_v785",
        "direct_consumers_v786",
        "direct_consumers_v787",
    ]
    assert scope["consumer_shards"][-1]["test_files"] == [
        "tests/python/test_v787_contract_9241.py",
        "tests/python/test_conformal_risk_9244.py",
    ]
    assert scope["deferred_validation_obligations"] == [
        dict(
            node_file="tests/python/test_causal_validation_9181.py",
            producer_reference=e.ref(e.ROOT / "tests/python/test_causal_validation_9181.py"),
            required_by="exp9257-forward-qualification",
            current_qualification_credit=0,
        )
    ]


def test_coverage_list_parity_and_mismatch_regression_control(work, monkeypatch):
    """SCENARIO-REPORT-9255-V788-CONTROLS: reject any owned_changed_paths vs measured coverage file-list mismatch."""
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    matching_cov = dict(
        complete=True,
        percent=100.0,
        files=[
            dict(path=p, statements=10, missing=[], excluded=[])
            for p in work["scope"]["owned_changed_paths"]
        ],
    )
    assert c.receipts_pass(work, dict(owned_statement_coverage=matching_cov))

    # Simulate the v787_contract.py line-238/290 6-vs-5 defect: extra file in owned_changed_paths
    mismatched_work = deepcopy(work)
    mismatched_work["scope"]["owned_changed_paths"] = [
        *e.OWNED,
        "python/carnot/reporting/v781_capstone.py",
    ]
    assert not c.receipts_pass(mismatched_work, dict(owned_statement_coverage=matching_cov))
    bad_val = c.build(mismatched_work, dict(owned_statement_coverage=matching_cov))
    assert bad_val["required_checks_passed"] is False
    assert bad_val["contract_ready_score"] == 0
    assert bad_val["replay_authority_ready_score"] == 0
    assert bad_val["verdict_class"] == "disqualified"

    # Simulate measured coverage missing a file present in owned_changed_paths
    truncated_cov = dict(
        complete=True,
        percent=100.0,
        files=[dict(path=p, statements=10, missing=[], excluded=[]) for p in e.OWNED[:-1]],
    )
    assert not c.receipts_pass(work, dict(owned_statement_coverage=truncated_cov))
    bad_val2 = c.build(work, dict(owned_statement_coverage=truncated_cov))
    assert bad_val2["required_checks_passed"] is False
    assert bad_val2["contract_ready_score"] == 0
    assert bad_val2["replay_authority_ready_score"] == 0
    assert bad_val2["verdict_class"] == "disqualified"


def test_owned_scope_excludes_unchanged_protocol_owner():
    """REQ-REPORT-9255-V788: coverage claims only files this experiment changes."""
    assert "python/carnot/reporting/protocol_owners.py" not in e.OWNED


@pytest.mark.parametrize("failure", ["missing_file", "missing_field", "changed_bytes"])
def test_external_failure_operand_kind(tmp_path, monkeypatch, failure):
    """SCENARIO-REPORT-9255-V788-CONTROLS: absent fields and changed bytes remain distinct."""
    path = tmp_path / "producer.json"
    if failure == "changed_bytes":
        path.write_text("{}")

    def reject(_):
        if failure == "missing_field":
            raise KeyError("conformal_risk_ready_score")
        raise ValueError("historical_input_drift:" + str(path))

    monkeypatch.setattr(e, "_inspect", reject)
    monkeypatch.setattr(e, "HISTORICAL_BASE", tmp_path / "historical", raising=False)
    result = e.inspect(tmp_path)
    gate = result["checks"][-1]
    expected = e.ref(path)["sha256"] if failure == "changed_bytes" else failure
    assert gate["observed"] == expected
    if failure == "missing_field":
        assert gate["artifact_field"] == "conformal_risk_ready_score"
    else:
        assert gate["path"] == str(path)
    assert not gate["passed"]


def test_malformed_planner_identity_is_blocked(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9255-V788-CONTROLS: malformed external objects name a failed operand."""
    root, frozen = build_v788_fixture_tree(tmp_path / "malformed")
    value = yaml.safe_load((root / e.ACTIVE).read_bytes())
    value["tasks"][0]["id"] = 9255
    (root / e.ACTIVE).write_text(yaml.safe_dump(value))
    monkeypatch.setattr(e, "FROZEN", frozen)
    result = e.authority(root)
    assert result["ready"] is False
    assert any(
        row["artifact_field"] == "tasks[0]" and not row["passed"] for row in result["checks"]
    )


def test_missing_design_names_its_operand(tmp_path):
    """SCENARIO-REPORT-9255-V788-CONTROLS: missing files retain their own failed path."""
    result = e.authority(tmp_path)
    assert not result["ready"]
    assert result["checks"][-1]["path"] == str(tmp_path / e.DESIGN)
    assert result["checks"][-1]["observed"] == "missing_file"


@pytest.mark.parametrize("container", ["[]", "null", "7"])
def test_malformed_yaml_container_is_blocked(tmp_path, container):
    """SCENARIO-REPORT-9255-V788-CONTROLS: planner data must contain named task objects."""
    path = tmp_path / e.DESIGN
    path.parent.mkdir(parents=True)
    path.write_text("## Full machine-readable contract\n```yaml\n" + container + "\n```\n")
    result = e.authority(tmp_path)
    assert not result["ready"]
    assert any(not row["passed"] for row in result["checks"])


def test_shared_measurement_deadline_reserves_verification(monkeypatch):
    """REQ-REPORT-9255-V788: descendants leave time to verify and publish actual receipts."""
    deadlines = {}

    def child(name, *args, **kwargs):
        deadlines[name] = kwargs["deadline"]
        return dict(passed=True)

    def invoke(argv):
        for name in ["owned_tests", "cold_replay"]:
            c.legacy.lifecycle.child(name, [], Path("/tmp"), deadline=9999)
        return 0

    monkeypatch.delenv("CARNOT_V788_DEADLINE", raising=False)
    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    monkeypatch.setattr(c.legacy.lifecycle, "child", child)
    monkeypatch.setattr(c, "_main", invoke)
    assert c.main(["--date", "20261120"]) == 0
    assert deadlines == dict(owned_tests=3300, cold_replay=4800)


def test_checkout_and_complete_contract_scope(work, tmp_path, monkeypatch):
    """REQ-REPORT-9255-V788: seven split shards, coverage parity and 14-task parity qualify V788."""
    assert work["run_date"] == "20261120"
    inspection = work["inspection"]
    non_contract = [r for r in inspection["checks"] if r not in inspection["contract"]["checks"]]
    assert all(r["passed"] for r in non_contract), [r for r in non_contract if not r["passed"]]
    assert inspection["contract"]["ready"], [
        r for r in inspection["contract"]["checks"] if not r["passed"]
    ]
    assert inspection["source_ready"] and inspection["static_ready"] and inspection["online_ready"]
    assert inspection["replay_ready"]
    assert len(inspection["historical_input_roots"]) == 8
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
    assert inspection["preserved_v786"]["exp9240_reconciliation_ready_score"] == 1
    assert inspection["preserved_v787"]["exp9241_verdict"] == "complete_blocked_v787_contract"
    assert inspection["preserved_v787"]["exp9241_readiness_scores"] == dict(
        contract_ready_score=0,
        replay_authority_ready_score=0,
        source_ready_score=0,
        static_methods_ready_score=0,
        online_methods_ready_score=0,
    )
    assert inspection["preserved_v787"]["exp9244_scores"] == dict(
        conformal_risk_ready_score=1,
        ca_ocrc_ready_score=1,
        provenance_audit_ready_score=1,
        required_checks_passed=True,
        scientific_effect_measured=False,
    )
    assert len(work["scope"]["consumer_shards"]) == 7
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
    assert len(value["consumer_shard_rows"]) == 7
    for key in [
        "ambiguity_masks_reference",
        "target_rubric_reference",
        "selective_deferral_masks_reference",
        "selective_deferral_protocol_reference",
        "spline_support_protocol_reference",
        "conformal_risk_protocol_reference",
        "qualified_conformal_risk_protocol_reference",
        "ca_ocrc_protocol_reference",
        "provenance_ledger_reference",
        "mycelium_provenance_ledger_reference",
        "conformal_risk_reducer_reference",
    ]:
        assert key in value["protocol_references"]
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    qualified = c.build(work, {})
    assert qualified["verdict_class"] == "null"
    assert qualified["honest_verdict"] == "complete_null_v788_contract"
    assert qualified["contract_ready_score"] == 1
    assert qualified["replay_authority_ready_score"] == 1
    assert (
        qualified["source_ready_score"]
        == qualified["static_methods_ready_score"]
        == qualified["online_methods_ready_score"]
        == 1
    )


def test_private_cli_and_interleaved_replays(work, tmp_path):
    """SCENARIO-REPORT-9255-V788-CONTROLS: E2E-018/021 private CLI and interleaved replay."""
    source = tmp_path / "work.json"
    atomic_json(source, work)
    output = tmp_path / (e.NAME + ".json")
    result = child_cli(tmp_path, "--from-work", str(source), "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    v787_work = v787.measure(original_root(tmp_path / "v787", 787), tmp_path / "raw787", "20261120")
    v787_source = tmp_path / "v787.json"
    atomic_json(v787_source, v787_work)
    v787_output = tmp_path / (v787.e.NAME + ".json")
    assert (
        child_cli(
            tmp_path,
            "--from-work",
            str(v787_source),
            "--output",
            str(v787_output),
            cli=v787.e.CLI,
        ).returncode
        == 0
    )
    assert child_cli(tmp_path, "--cold-replay", str(v787_output), cli=v787.e.CLI).returncode == 0
    assert child_cli(tmp_path, "--cold-replay", str(output)).returncode == 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    code = "\n".join(
        [
            "import sys; from pathlib import Path",
            "sys.path[:0] = " + repr([str(e.ROOT / "python"), str(e.ROOT)]),
            "from carnot.reporting import v788_contract as cur, v787_contract as prev",
            "assert cur.replay(Path(" + repr(str(output)) + "))['passed']",
            "assert prev.replay(Path(" + repr(str(v787_output)) + "))['passed']",
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
    """SCENARIO-REPORT-9255-V788-CONTROLS: missing evidence blocks and owner drift refuses."""
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
    """SCENARIO-REPORT-9255-V788-CONTROLS: new pins cannot bless changed gates or occupied IDs."""
    root, frozen = build_v788_fixture_tree(tmp_path / "tree")
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
    (root / "results/experiment_9256_existing.json").write_text("{}")
    assert not e.authority(root)["ready"]
    (root / e.ACTIVE).write_text("{}")
    assert e.authority(root)["checks"][-1]["observed"] == "missing_field"
    (root / e.ACTIVE).unlink()
    assert any(r["observed"] == "missing_file" for r in e.authority(root)["checks"])


def test_external_absence_and_protocol_failure_controls(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9255-V788-CONTROLS: absence blocks, forged sidecars and owner drift refuse."""
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
    """REQ-REPORT-9255-V788: seven partitioned shards run under one parent supervisor and deadline."""
    seen = []
    measured_args = []

    def child(name, *args, **kwargs):
        seen.append((name, kwargs["deadline"]))
        return dict(name=name, passed=True, duration_s=1.25, exit_code=0)

    def fake_measured(paths, owned_list, out_dir):
        measured_args.append(list(owned_list))
        return dict(
            complete=True,
            percent=100.0,
            files=[dict(path=p, statements=10, missing=[], excluded=[]) for p in owned_list],
        )

    monkeypatch.setattr(c.legacy.lifecycle, "child", child)
    monkeypatch.setattr(c.legacy.lifecycle, "measured", fake_measured)
    value = c.qualify(work)
    assert len(value["checks"]) == 12 and value["owned_statement_coverage"]["complete"]
    assert measured_args == [work["scope"]["owned_changed_paths"]]
    assert value["repository_health"]["passed"]
    assert dict(seen) == dict(
        owned_tests=2400,
        direct_consumers_core=600,
        direct_consumers_v782=1200,
        direct_consumers_v783=1200,
        direct_consumers_v784=1200,
        direct_consumers_v785=1200,
        direct_consumers_v786=1200,
        direct_consumers_v787=1200,
        ruff_check=600,
        ruff_format=600,
        strict_mypy=600,
        spec_coverage=600,
        global_python_suite=600,
    )
    assert all(r["scope_sha256"] == c.canonical_hash(work["scope"]) for r in value["checks"])
    monkeypatch.setattr(c, "measure", lambda *_: work)
    monkeypatch.setattr(c, "qualify", lambda _: value)
    monkeypatch.setattr(c, "receipts_pass", lambda *_: True)
    monkeypatch.setattr(
        c.previous.previous.previous.previous, "publish_primary", lambda *_: dict(passed=True)
    )
    assert c.main(["--date", "20261120", "--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    monkeypatch.setenv("CARNOT_V788_DEADLINE", "0")
    assert c.main(["--date", "20261120", "--output", str(tmp_path / (e.NAME + ".json"))]) == 1
