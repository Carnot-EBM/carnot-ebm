"""REQ-REPORT-8017: private custody, independent reductions and terminal CLI."""

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from carnot.reporting import v694_capstone as cap
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8017_v694_capstone as cli

ROOT = Path(__file__).resolve().parents[2]


def fixture(root):
    """Oracle fixtures test custody without supplying natural science."""
    root.mkdir(parents=True, exist_ok=True)
    for name, target in (("active.yaml", "research-roadmap.yaml"), ("design.md", "design.md")):
        (root / target).write_bytes(
            gzip.decompress((ROOT / "tests/fixtures/v694" / (name + ".gz")).read_bytes())
        )
    tasks = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())["tasks"]
    fields = {t["id"]: {} for t in tasks}
    for t in tasks:
        for g in t.get("gated_on", []):
            if g["artifact_field"] not in {"verdict_class", "flagged_adversarial"}:
                fields[g["upstream"]][g["artifact_field"]] = g["value"]
    for t in tasks[:-1]:
        path = root / t["deliverable"]
        side = root / "sidecars" / (t["id"] + ".json")
        atomic_json(
            path,
            dict(
                experiment_id=int(t["id"][3:7]),
                task_id=t["id"],
                honest_verdict="complete_null_private",
                verdict_class="null",
                rows=[],
                execution_date="20261002",
                flagged_adversarial=False,
                terminal_validation_sidecar_path=str(side),
                **fields[t["id"]],
            ),
        )
        atomic_json(side, dict(passed=True, primary_sha256=sha256_file(path)))
    return tasks


def test_exact_roster_missing_and_skip(tmp_path):
    """SCENARIO-REPORT-8017-CUSTODY: skip bytes cannot become fabricated rows."""
    tasks = fixture(tmp_path)
    path = tmp_path / tasks[4]["deliverable"]
    path.unlink()
    skip = tmp_path / "results/experiment_8009_development_decisions.json"
    atomic_json(
        skip,
        dict(
            experiment=8009,
            schema="blocked_gate_check_v1",
            status="blocked",
            honest_verdict="blocked_gate_check_failed",
            gates_evaluated=[],
        ),
    )
    rows, refs, failed, reduced = cap.collect(tmp_path, tasks)
    assert len(rows) == 12 and rows[4]["producer_status"] == "conductor_skip_receipt"
    assert rows[4]["verdict_class"] == "blocked" and failed
    assert reduced[4]["measurement_available"] is False
    assert any(r["path"] == str(skip) for r in refs)
    skip.unlink()
    assert cap.collect(tmp_path, tasks)[0][4]["producer_status"] == "missing"
    path.write_text("[]")
    assert cap.collect(tmp_path, tasks)[0][4]["verdict_class"] == "disqualified"


def test_gate_and_summary_drift(tmp_path):
    """REQ-REPORT-8017: zero and missing gates remain failed operands."""
    tasks = fixture(tmp_path)
    path = tmp_path / tasks[3]["deliverable"]
    data = cap.old.read(path)
    data.pop("conditioned_fit_ready_score")
    data["flagged_adversarial"] = True
    atomic_json(path, data)
    failed = cap.collect(tmp_path, tasks)[2]
    assert any(r["observed"] == "contract_error_missing_field" for r in failed)
    assert any(r["artifact_field"] == "terminal_primary_sha256" for r in failed)
    row = dict(
        source_cluster_id="a", family_id="a", y=0, probability=0.5, arm="frozen", action="escalate"
    )
    data = dict(rows=[row, dict(row, seed=2)], retention_rows=[], metrics=[])
    got = cap.reduce_primitives(data, 8009)
    assert got["static"]["independent_sources"] == 1
    assert got["action_counts"]["escalate"] == 2
    data["rows"][0]["actual_cost"] = 4
    with pytest.raises(ValueError, match="primitive_cost_drift"):
        cap.reduce_primitives(data, 8009)
    assert (
        cap.reduce_primitives(dict(rows=[]), 8012)["learning"]["causal_updates"]["updates_checked"]
        == 0
    )


def test_source_response_reduction(tmp_path):
    """REQ-REPORT-8017: source sensitivity uses original response JSON."""
    refs = []
    rows = []
    for arm, probability in (("original", 0.1), ("swap", 0.8), ("duplicate", 0.1)):
        p = tmp_path / (arm + ".json")
        row = dict(
            group_id="a",
            arm=arm,
            raw_response={
                "choices": [
                    {"message": {"content": json.dumps(dict(unsupported_probability=probability))}}
                ]
            },
            probability=probability,
        )
        atomic_json(p, row)
        refs.append(dict(path=str(p), sha256=sha256_file(p)))
        rows.append(row)
    data = dict(
        checkpoint_references=refs,
        rows=rows,
        sensitivity_rows=[
            dict(
                group_id="a",
                complete=True,
                swap_minus_original=0.7,
                duplicate_minus_original=0.0,
                swap_minus_duplicate=0.7,
            )
        ],
    )
    got = cap.reduce_primitives(data, 8011)
    assert got["source"]["independent_sources"] == 1
    assert got["source"]["mean_swap_minus_duplicate"] == pytest.approx(0.7)
    data["duplicate_control_rows"] = [
        dict(group_id="a", complete=True, duplicate_minus_original=0.0)
    ]
    assert cap.reduce_primitives(data, 8011)["source"][
        "mean_swap_minus_duplicate"
    ] == pytest.approx(0.7)
    changed = deepcopy(data)
    changed["rows"][0]["probability"] = 0.9
    with pytest.raises(ValueError, match="source_probability_drift"):
        cap.reduce_primitives(changed, 8011)
    changed = deepcopy(data)
    changed["sensitivity_rows"][0]["swap_minus_original"] = 0.4
    with pytest.raises(ValueError, match="source_summary_drift"):
        cap.reduce_primitives(changed, 8011)
    changed = deepcopy(data)
    changed["checkpoint_references"] = []
    with pytest.raises(ValueError, match="source_checkpoint_roster_drift"):
        cap.reduce_primitives(changed, 8011)
    changed = deepcopy(data)
    changed["rows"].append(changed["rows"][0])
    changed["checkpoint_references"].append(changed["checkpoint_references"][0])
    with pytest.raises(ValueError, match="source_duplicate_arm"):
        cap.reduce_primitives(changed, 8011)
    changed = deepcopy(data)
    changed["rows"] = changed["rows"][:1]
    changed["checkpoint_references"] = changed["checkpoint_references"][:1]
    assert cap.reduce_primitives(changed, 8011)["source"]["independent_sources"] == 0
    saved = cap.old.read(Path(refs[0]["path"]))
    saved["raw_response"]["choices"][0]["message"]["content"] = json.dumps(
        dict(unsupported_probability=2)
    )
    saved["probability"] = 2
    atomic_json(Path(refs[0]["path"]), saved)
    changed["rows"][0] = saved
    with pytest.raises(ValueError, match="source_probability_drift"):
        cap.reduce_primitives(changed, 8011)
    saved["status"] = "excluded"
    atomic_json(Path(refs[0]["path"]), saved)
    assert cap.reduce_primitives(changed, 8011)["source"]["independent_sources"] == 0
    saved.pop("status")
    saved.update(parsed={"completed": True}, visible_ids=[])
    atomic_json(Path(refs[0]["path"]), saved)
    with pytest.raises(ValueError, match="source_parse_drift"):
        cap.reduce_primitives(changed, 8011)


def test_build_replay_and_cli(tmp_path):
    """SCENARIO-REPORT-8017-TERMINAL: real null, blocked and tampered CLI bytes."""
    root = tmp_path / "input"
    tasks = fixture(root)
    output = tmp_path / "experiment_8017_v694_capstone.json"
    args = [
        "--date",
        "20261002",
        "--root",
        str(root),
        "--design",
        str(root / "design.md"),
        "--output",
        str(output),
        "--evidence-only",
    ]
    assert cli.main(args) == 0
    value = cap.old.read(output)
    assert len(value["rows"]) == 13 and value["science_ready"] is False
    assert not cap.cold_replay(value)
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / cap.CLI),
            "--date",
            "20261002",
            "--cold-replay",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value["independent_reduction_rows"][0]["measurement_available"] = True
    atomic_json(output, value)
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / cap.CLI),
            "--date",
            "20261002",
            "--cold-replay",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert cli.main(["--date", "20261002", "--cold-replay", str(output)]) == 1
    assert cli.main(["--date", "20261002", "--cold-replay", str(tmp_path / "absent")]) == 1
    (root / tasks[0]["deliverable"]).unlink()
    assert cli.main(args) == 0
    assert cap.old.read(output)["verdict_class"] == "blocked"


def test_validation_and_real_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8017-TERMINAL: final bytes pass actual private terminal readers."""
    root = tmp_path / "input"
    fixture(root)
    output = tmp_path / "experiment_8017_v694_capstone.json"
    real_run = cap.run_check
    real_manifest = cap.manifest
    real_qualify = cap.qualify
    health_calls = []

    def manifest(private):
        frozen = real_manifest(private)
        assert frozen["coverage_includes"] == cap.OWNED
        assert any(s["classification"] == "diagnostic" for s in frozen["commands"])
        frozen["commands"] = [
            s
            for s in frozen["commands"]
            if s["name"] in {"publication_gate", "coverage_json", "full_suite"}
        ]
        return frozen

    def run(repo, spec, private, durable):
        if spec["name"] not in {"publication_gate", "coverage_json", "full_suite"}:
            return real_run(repo, spec, private, durable)
        if spec["name"] == "full_suite":
            health_calls.append(spec["argv"])
        log = durable / (spec["name"] + ".json")
        atomic_json(
            log,
            dict(
                paper_ready=False,
                unmet_gates=["G2"],
                gates={f"G{i}": {"pass": i != 2} for i in range(1, 5)},
            ),
        )
        if spec["name"] == "coverage_json":
            atomic_json(
                private.parent / "coverage.json",
                dict(
                    files={
                        p: {"summary": {"num_statements": 1, "covered_lines": 1}} for p in cap.OWNED
                    }
                ),
            )
        return dict(
            spec,
            passed=spec["name"] != "full_suite",
            actual_exit=int(spec["name"] == "full_suite"),
            log_path=str(log),
            log_sha256=sha256_file(log),
        )

    monkeypatch.setattr(cap, "manifest", manifest)
    monkeypatch.setattr(cap, "run_check", run)
    assert cap.qualify(root, root / "design.md", "20261002", output) == 0
    value = cap.old.read(output)
    assert value["capstone_execution_ready_score"] == 1
    assert value["sample_size_budget"]["completed"] == 13
    assert value["generalized_learning_benefit_score"] == 0
    assert cap.old.read(Path(value["terminal_validation_sidecar_path"]))[
        "primary_sha256"
    ] == sha256_file(output)
    changed = deepcopy(value)
    changed["rows"][-1]["numerator"] = 0
    assert cap.cold_replay(changed) == ["owned_primitive_drift"]
    cap.apply_checks(value, [dict(passed=False)], {})
    assert value["verdict_class"] == "disqualified"
    cap.apply_checks(
        value,
        [dict(passed=True), dict(passed=False, classification="diagnostic")],
        {p: dict(num_statements=1, covered_lines=1) for p in cap.OWNED},
    )
    assert value["required_checks_passed"] and value["capstone_execution_ready_score"] == 0
    monkeypatch.setattr(cap, "qualify", lambda *a: 0)
    assert cli.main(["--date", "20261002"]) == 0
    monkeypatch.setattr(cap, "qualify", real_qualify)
    original = cap.old.read(output)
    original["checkpoint_references"] = [
        r
        for r in original["checkpoint_references"]
        if r["role"] not in {"command_freeze", "owned_reduction"}
    ]
    monkeypatch.setattr(cap, "build", lambda *a: deepcopy(original))
    retired = []
    monkeypatch.setattr(cap, "append_retirements", lambda rows: retired.extend(rows))
    assert cap.qualify(ROOT, root / "design.md", "20261002", output) == 0
    assert retired
    assert len(health_calls) == 1


def test_replay_rejects_each_custody_boundary(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8017-CUSTODY: snapshots, fields and checkpoint bytes are operands."""
    tasks = fixture(tmp_path)
    value = cap.build(tmp_path, tmp_path / "design.md", "20261002", tmp_path / "raw")
    with pytest.raises(ValueError, match="date_or_thirteen"):
        cap.build(tmp_path, tmp_path / "design.md", "20261001", tmp_path / "raw")
    value["science_ready"] = True
    value["verdict_class"] = "blocked"
    value["capstone_execution_ready_score"] = 1
    value["canonical_tasks_sha256"] = "changed"
    assert {"unsupported_generalization", "unsafe_readiness", "authority_drift"} <= set(
        cap.cold_replay(value)
    )
    snapshot = Path(value["authority_snapshots"]["active"]["snapshot_path"])
    snapshot.write_text("changed")
    assert cap.cold_replay(value) == ["authority_snapshot_changed"]
    producer = tmp_path / tasks[0]["deliverable"]
    producer.write_text("changed")
    assert cap.cold_replay(value) == ["source_bytes_changed"]
    data = cap.old.read(tmp_path / tasks[6]["deliverable"])
    data.update(
        raw_shard_hashes=[dict(path=str(tmp_path / "missing-shard"), sha256="original")],
        rows=[dict(raw_response={})],
        checkpoint_references=[],
    )
    atomic_json(tmp_path / tasks[6]["deliverable"], data)
    assert any(r["artifact_field"] == "primitive_sha256" for r in cap.collect(tmp_path, tasks)[2])
    assert any(
        r["artifact_field"] == "independent_reduction" for r in cap.collect(tmp_path, tasks)[2]
    )
    real_assess = cap.assess

    def assess(*a, **kwargs):
        result = real_assess(*a, **kwargs)
        result["gate_check_summary"] = [
            dict(artifact_field="activation", observed=False, expected=True)
        ]
        return result

    monkeypatch.setattr(cap, "assess", assess)
    producer.write_text("{}")
    assert (
        cap.build(tmp_path, tmp_path / "design.md", "20261002", tmp_path / "raw2")["verdict_class"]
        == "blocked"
    )


def test_terminal_failure_paths_and_retirement(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8017-TERMINAL: bad terminal flags and reader drift reject publication."""
    tasks = fixture(tmp_path)
    value = cap.build(tmp_path, tmp_path / "design.md", "20261002", tmp_path / "raw")
    cap.apply_checks(
        value, [dict(passed=True)], {p: dict(num_statements=1, covered_lines=1) for p in cap.OWNED}
    )
    mode = {"value": "terminal"}

    def run(repo, spec, private, durable):
        log = durable / (spec["name"] + ".json")
        atomic_json(log, {"flagged_count": int(mode["value"] == "terminal")})
        return dict(
            spec,
            passed=not (mode["value"] == "published" and spec["name"].startswith("published_")),
            log_path=str(log),
            log_sha256=sha256_file(log),
        )

    monkeypatch.setattr(cap, "run_check", run)
    output = tmp_path / "experiment_8017_v694_capstone.json"
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        cap.publish(value, output, tmp_path / "terminal", tmp_path / "raw")
    mode["value"] = "readers"
    real_reader = cap.reader_receipt
    monkeypatch.setattr(cap, "reader_receipt", lambda *a, **k: {"passed": False})
    with pytest.raises(ValueError, match="published_reader_drift"):
        cap.publish(value, output, tmp_path / "terminal", tmp_path / "raw")
    monkeypatch.setattr(cap, "reader_receipt", real_reader)
    mode["value"] = "published"
    with pytest.raises(ValueError, match="published_validation_failed"):
        cap.publish(value, output, tmp_path / "terminal", tmp_path / "raw")
    primary = tmp_path / tasks[1]["deliverable"]
    data = cap.old.read(primary)
    validator = tmp_path / "validator.json"
    side = Path(data["terminal_validation_sidecar_path"])
    atomic_json(side, dict(primary_sha256=sha256_file(primary), sidecar_path=str(validator)))
    atomic_json(validator, dict(report={"passed": False}))
    data["raw_shard_hashes"] = {str(validator): sha256_file(validator)}
    data["gate_check_summary"] = [
        dict(
            upstream_id="prior",
            path=str(primary),
            hash=sha256_file(primary),
            artifact_field="class_support",
            expected=True,
            observed=False,
            passed=False,
        )
    ]
    atomic_json(primary, data)
    assert any(
        r["artifact_field"] == "terminal_validation.passed" for r in cap.collect(tmp_path, tasks)[2]
    )
    ops = tmp_path / "ops"
    ops.mkdir()
    (ops / "exclusion_manifest.yaml").write_text("retired_extras:\n")
    monkeypatch.setattr(cap, "ROOT", tmp_path)
    row = dict(
        task_id="task",
        prior_experiment_id="prior",
        retire=True,
        prior_authenticated=True,
        scope="same scope",
        terminal_verdict="complete_null_private",
        prior_path="prior.json",
        prior_sha256="sha256:prior",
        producer_path="producer.json",
        producer_sha256="sha256:producer",
        reopen_condition="New independent targets must change decisions.",
    )
    cap.append_retirements([row])
    before = (ops / "exclusion_manifest.yaml").read_bytes()
    cap.append_retirements([row, dict(row, retire=False)])
    assert (ops / "exclusion_manifest.yaml").read_bytes() == before
