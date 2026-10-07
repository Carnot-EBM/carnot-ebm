"""REQ-REPORT-8217 and REQ-VERIFY-8217: private terminal accounting controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import v709_capstone as c
from carnot.reporting import v709_capstone_inputs as e
from carnot.reporting import v709_capstone_science as s
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary


def fixture(tmp_path):
    """Real frozen activation provides task identity while all producer claims stay private."""
    root = tmp_path / "private"
    source = json.loads((e.ROOT / e.INPUT).read_bytes())
    for role in ("design", "active"):
        ref = source["authority_snapshots"][role]
        path = root / (role + ".bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(ref["snapshot_path"]).read_bytes())
        ref.update(snapshot_path=str(path), sha256=sha256_file(path))
    tasks = json.loads(Path(source["work_reference"]["path"]).read_bytes())["contract"]["tasks"]
    for n, task in enumerate(tasks[:-1], 8205):
        value = dict(
            experiment_id=n,
            task_id=task["id"],
            honest_verdict="complete_null_private",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            gate_check_summary=[],
            code_config_hashes={},
            validation_receipts=[],
            rows=[],
        )
        value.update(
            {
                g["artifact_field"]: 1
                for t in tasks
                for g in t["gated_on"]
                if g["upstream"] == task["id"]
            }
        )
        if n == 8205:
            value.update(
                {
                    k: source[k]
                    for k in (
                        "authority_snapshots",
                        "canonical_tasks_sha256",
                        "consumer_reader_ready_score",
                        "contract_ready_score",
                        "task_dispositions",
                    )
                }
            )
        publish_primary(root / task["deliverable"], value, lambda _: dict(passed=True))
    return root, tasks


def test_custody_states(tmp_path):
    """SCENARIO-REPORT-8217-CUSTODY: every missing, skipped and failed task keeps its slot."""
    root, tasks = fixture(tmp_path)
    missing = root / tasks[7]["deliverable"]
    missing.unlink()
    skipped = root / tasks[6]["deliverable"]
    skipped.unlink()
    atomic_json(
        skipped.with_name("experiment_8211_gate_skip.json"),
        dict(
            schema="blocked_gate_check_v1",
            honest_verdict="blocked_gate_check_failed",
            failed_field="stream_input_ready_score",
            failed_upstream=tasks[1]["id"],
            failed_evidence_path=str(root / tasks[1]["deliverable"]),
            failed_evidence_sha256=None,
            failed_operator="==",
            failed_expected=1,
            failed_observed=0,
        ),
    )
    broken = root / tasks[5]["deliverable"]
    value = json.loads(broken.read_bytes())
    value.update(
        required_checks_passed=False,
        verdict_class="disqualified",
        honest_verdict="complete_disqualified_owned_validation",
    )
    publish_primary(broken, value, lambda _: dict(passed=True))
    data = e.load(root, tmp_path / "raw")
    value = s.reduce(data, s.measure(data, tmp_path / "science"))
    assert value["completed_count"] == len(value["task_dispositions"]) == 13
    assert value["task_dispositions"][5]["disposition"] == "disqualified"
    assert value["task_dispositions"][6]["disposition"] == "pre_gate_skip"
    assert value["task_dispositions"][7]["disposition"] == "missing"
    assert value["verdict_class"] == "blocked"
    assert value["H1"]["alpha"] == value["H2"]["alpha"] == 0.025
    assert len(value["gap_decisions"]) == 3
    assert not value["service_evidence_scope"]["nfr01_met"]
    assert all("upstream_id" in g for g in value["gate_check_summary"])
    assert value["gate_check_summary"][-1]["upstream_id"] == tasks[9]["id"]
    c.qualify(value, False)
    assert value["verdict_class"] == "disqualified"
    assert value["task_dispositions"][-1]["failed"]


def test_fallback_and_bad_authority(tmp_path):
    """SCENARIO-REPORT-8217-CUSTODY: staged or absent authority never earns activation."""
    root, tasks = fixture(tmp_path)
    source = json.loads((root / e.INPUT).read_bytes())
    design = root / e.q.DESIGN
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_bytes(Path(source["authority_snapshots"]["design"]["snapshot_path"]).read_bytes())
    (root / "research-roadmap.yaml").write_bytes(
        Path(source["authority_snapshots"]["active"]["snapshot_path"]).read_bytes()
    )
    (root / e.INPUT).unlink()
    data = e.load(root, tmp_path / "fallback")
    assert data["authority"]["activated"] and data["original_contract_failures"]
    (root / "research-roadmap.yaml").unlink()
    data = e.load(root, tmp_path / "missing")
    value = s.reduce(data, s.measure(data, tmp_path / "unavailable"))
    assert value["completed_count"] == 13 and not data["authority"]["activated"]


def test_prior_exact_scope(tmp_path):
    """SCENARIO-REPORT-8217-CUSTODY: untested prerequisite skips do not retire science."""
    root, tasks = fixture(tmp_path)
    prior = tasks[0]["prior_failures"][0]
    path = root / "results/experiment_8192_private_prior.json"
    atomic_json(path, dict(task_id=prior["experiment_id"], honest_verdict=prior["verdict"]))
    data = e.load(root, tmp_path / "raw")
    value = s.reduce(data, s.measure(data, tmp_path / "science"))
    assert value["retirement_decisions"][0]["prior_verdict_matches_artifact"]
    assert not any(r["retire_exact_configuration"] for r in value["retirement_decisions"])


def cli(argv, cwd):
    """The real subprocess runs outside the checkout with no ambient import path."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(e.ROOT / c.CLI), *map(str, argv)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_real_cli_and_tamper(tmp_path):
    """SCENARIO-REPORT-8217-CLI: private terminal bytes replay and altered aggregates fail."""
    root, _ = fixture(tmp_path)
    output = tmp_path / (c.NAME + ".json")
    result = cli(["--root", root, "--output", output, "--private-fixture"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["capstone_execution_ready_score"] == 1
    assert value["verdict_class"] == "blocked" and value["completed_count"] == 13
    assert cli(["--cold-replay", output], tmp_path).returncode == 0
    value["H1"]["status"] = "invented"
    atomic_json(output, value)
    assert cli(["--cold-replay", output], tmp_path).returncode == 1
    assert cli(["--date", "20261007"], tmp_path).returncode == 2
    assert cli(["--private-fixture"], tmp_path).returncode == 1


def test_manifest(tmp_path):
    """SCENARIO-REPORT-8217-CLI: frozen commands own only the new statements and private E2E."""
    plan = c.commands(tmp_path)
    text = json.dumps(plan)
    assert c.CLI in text and c.TEST in text and "--strict" in text
    assert any(r["scope"] == "repository_health" for r in plan)
    assert "test_experiment_7891_v685_authority_lifecycle.py" in text
    assert all(str(tmp_path) in " ".join(r["argv"]) for r in plan if "pytest" in r["name"])


def primitive_fixture(tmp_path):
    """SCENARIO-VERIFY-8217-BRANCHES: qualified reducers receive private authentic shapes."""
    from carnot.reporting import arc_authoritative_frontier_8215 as arc
    from carnot.reporting import hardware_workload_obligations_8216 as hardware

    root, tasks = fixture(tmp_path)
    data = e.load(root, tmp_path / "inputs")
    values = data["primaries"]
    evidence = s.action.fixture()
    ref_path = tmp_path / "evidence.json"
    atomic_json(ref_path, evidence)
    ref = dict(path=str(ref_path), sha256=sha256_file(ref_path))
    values[tasks[4]["id"]]["measurement_reference"] = ref
    legacy = s.memory.legacy.control_rows(improved=True)
    legacy += [dict(r, arm="calibration_only") for r in legacy if r["arm"] == "fixed_public_center"]
    reconstructed = dict(legacy_rows=legacy, causal_checks=dict(restart_equal=True))
    values[tasks[7]["id"]].update(reconstructed)
    values[tasks[6]["id"]].update(trajectory_path=str(ref_path), raw_shard_hashes=[ref])
    work = dict(requests=[], startup_s=1, start_ns=0, end_ns=1000000000)
    service_path = tmp_path / "service.json"
    atomic_json(service_path, dict(work=work))
    values[tasks[9]["id"]].update(
        s.service.reduce(work),
        result_path=str(service_path),
        raw_shard_hashes=[dict(path=str(service_path), sha256=sha256_file(service_path))],
    )
    source = json.loads(next((e.ROOT / "results").glob("experiment_8216_*.json")).read_bytes())
    values[tasks[11]["id"]].update(
        {k: source[k] for k in ("primitive_reference", "replay_input_reference")}
    )
    arc_path = tmp_path / "arc.json"
    atomic_json(arc_path, arc.reduction([], {}))
    values[tasks[10]["id"]].update(
        primitive_path=str(arc_path), raw_shard_hashes={str(arc_path): sha256_file(arc_path)}
    )
    return data, evidence, reconstructed, ref


def test_independent_branches_and_mutations(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8217-BRANCHES: H1 survives a disqualified audit and H2 remains separate."""
    data, evidence, reconstructed, ref = primitive_fixture(tmp_path)
    monkeypatch.setattr(
        s.action,
        "measure",
        lambda *args: dict(
            evidence=evidence, owned_failure="", refs=[ref], raw_shard_hashes=[], checks=[]
        ),
    )
    monkeypatch.setattr(s.memory, "reconstruct", lambda *args: reconstructed)
    data["dispositions"][5]["eligible"] = False
    data["dispositions"][5]["disposition"] = "disqualified"
    branches = s.measure(data, tmp_path / "science")
    assert branches["H1"] and branches["H2"] and branches["board"] and branches["arc"]
    value = s.reduce(data, branches)
    assert value["H1"]["status"] != "blocked" and value["H2"]["status"] == "completed_signal"
    assert value["H1"]["energy_increment"] != value["H1"]["policy_versus_baseline"]
    assert len(value["board_obligations"]) == 3 and len(value["access_obligations"]) == 2
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert not value["service_evidence_scope"]["nfr01_met"]
    bad = deepcopy(branches)
    bad["arc"]["new_outcome_count"] = 42
    with pytest.raises(ValueError, match="arc_primitive_drift"):
        s.reduce(data, bad)
    prior = data["prior_evidence"][0]
    prior.update(
        prior_verdict_matches_artifact=True, verdict=data["dispositions"][0]["honest_verdict"]
    )
    data["primaries"][prior["task_id"]]["configuration_scope_sha256"] = (
        "private unchanged configuration"
    )
    prior["evidence"] = [
        dict(configuration_scope=dict(configuration_scope_sha256="private unchanged configuration"))
    ]
    assert s.reduce(data, branches)["retirement_decisions"][0]["retire_exact_configuration"]
    values = data["primaries"]
    values[data["tasks"][7]["id"]]["causal_checks"] = {"tampered": True}
    values[data["tasks"][9]["id"]]["completed_count"] = 999
    monkeypatch.setattr(
        s.action,
        "measure",
        lambda *args: dict(
            evidence={},
            owned_failure="bad",
            checks=[dict(passed=False)],
            refs=[],
            raw_shard_hashes=[],
        ),
    )
    broken = s.measure(data, tmp_path / "broken")
    assert not broken["H1"] and not broken["H2"] and not broken["service"]
    assert broken["board"] and broken["arc"]


def test_typed_receipts_and_malformed_primary(tmp_path):
    """SCENARIO-REPORT-8217-CUSTODY: typed schemas and malformed receipts stay auditable."""
    root, tasks = fixture(tmp_path)
    (root / "ops").mkdir()
    (root / "ops/exclusion_manifest.yaml").write_text("retired:\n- experiment_id: 8208\n")
    (root / "ops/conductor-log.md").write_text(tasks[2]["id"])
    path = root / tasks[2]["deliverable"]
    value = json.loads(path.read_bytes())
    log = root / "stream.txt"
    log.write_text("actual private log")
    value.update(
        code_config_hashes={str(log): sha256_file(log)},
        validation_receipts={
            "typed": dict(passed=True, stdout_path=str(log), stdout_sha256=sha256_file(log)),
            "unrelated_health": dict(passed=False, scope="repository_health"),
        },
    )
    side = path.parent / "raw" / path.stem / "terminal.json"
    value["terminal_validation_sidecar_path"] = str(side)
    publication = publish_primary(path, value, lambda _: dict(passed=True))
    atomic_json(side, dict(publication=publication))
    (root / tasks[3]["deliverable"]).write_text("malformed")
    data = e.load(root, tmp_path / "raw")
    assert data["dispositions"][2]["qualified"]
    assert data["dispositions"][2]["repository_health_failures"]
    assert data["dispositions"][3]["retired_scope_matches"]
    assert any(g["artifact_field"] == "readable_primary" for g in data["failures"])


def test_unavailable_authority_and_valid_external_block(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8217-CUSTODY: unavailable authority retains thirteen unmeasured slots."""
    root, tasks = fixture(tmp_path)
    (root / e.INPUT).unlink()
    data = e.load(root, tmp_path / "missing_authorities")
    assert len(data["dispositions"]) == 12 and not data["authority"]["activated"]
    path = root / tasks[2]["deliverable"]
    value = json.loads(path.read_bytes())
    value.update(
        verdict_class="blocked",
        honest_verdict="complete_blocked_external",
        gate_check_summary=[e.operand(str(path), "external_label_resource", True, None)],
    )
    publish_primary(path, value, lambda _: dict(passed=True))
    data = e.load(root, tmp_path / "blocked")
    assert data["dispositions"][2]["verdict_class"] == "blocked"
    assert data["dispositions"][2]["qualified"]
    monkeypatch.setattr(
        c.qualified,
        "precondition_command",
        lambda: dict(
            name="negative_environment",
            argv=[
                sys.executable,
                "-c",
                'print("missing external environment"); raise SystemExit(1)',
            ],
            deadline=10,
            expected=0,
            scope="preconditions",
        ),
    )
    output = tmp_path / (c.NAME + ".json")
    assert c.main(["--root", str(root), "--output", str(output), "--private-fixture"]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert any(
        r["artifact_field"] == "python_environment_exit" for r in value["gate_check_summary"]
    )
    altered = deepcopy(value)
    altered["code_config_hashes"][0]["sha256"] = "wrong"
    atomic_json(output, altered)
    with pytest.raises(ValueError, match="evidence_hash_drift"):
        c.replay(output)
    altered = deepcopy(value)
    altered["validation_receipts"][0]["stdout_sha256"] = "wrong"
    atomic_json(output, altered)
    with pytest.raises(ValueError, match="validation_stream_drift"):
        c.replay(output)
