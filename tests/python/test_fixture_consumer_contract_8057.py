"""REQ-REPORT-8057: fixture admission cannot turn oracle checks into science."""

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v698_fixture_consumer_contract as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def authorities(tmp_path):
    for source, target in (("design.md.gz", "design.md"), ("active.yaml.gz", "active.yaml")):
        (tmp_path / target).write_bytes(
            gzip.decompress((e.ROOT / "tests/fixtures/v698" / source).read_bytes())
        )
    (tmp_path / "staged.yaml").write_bytes((tmp_path / "active.yaml").read_bytes())
    return tmp_path / "design.md", tmp_path / "staged.yaml", tmp_path / "active.yaml"


def test_authority_complete_bytes_and_absent_stage(tmp_path):
    """SCENARIO-REPORT-8057-CONSUMERS: full prompts and consumed stage stay distinct."""
    design, staged, active = authorities(tmp_path)
    value = e.assess(design, staged, active, tmp_path / "snapshots")
    assert value["activated"] and len(value["contract_rows"]) == 13
    staged.unlink()
    assert e.assess(design, staged, active, tmp_path / "snapshots")["activated"]
    original = yaml.safe_load(active.read_text())
    for field in ("prompt", "title", "phase"):
        changed = deepcopy(original)
        changed["tasks"][0][field] = "changed"
        active.write_text(yaml.safe_dump(changed))
        assert not e.assess(design, staged, active, tmp_path / "snapshots")["activated"]
    active.write_text(yaml.safe_dump(original))
    text = design.read_text()
    design.write_text(text.replace("Canonical task SHA-256:", "removed digest:"))
    assert not e.assess(design, staged, active, tmp_path / "snapshots")["activated"]


def test_gate_matrix_and_real_history(tmp_path):
    """SCENARIO-REPORT-8057-CONSUMERS: conductor evaluates every class and control."""
    rows = e.gate_matrix(tmp_path)
    assert all(r["matched"] for r in rows)
    assert {r["producer_class"] for r in rows} >= set(e.CLASSES)
    circular = [
        r for r in rows if r["producer_class"] == "circular_positive" and r["control"] == "base"
    ]
    assert [r["observed_passed"] for r in circular] == [True, False]
    history = e.historical(e.ROOT, tmp_path / "history")
    assert history["original_gate"]["passed"] is False
    assert (
        next(g["actual"] for g in history["original_gate"]["gates_evaluated"] if not g["passed"])
        == "circular_positive"
    )
    assert [r["evidence_kind"] for r in history["rows"]] == [
        "actual_skip_receipt",
        "log_only",
        "log_only",
        "log_only",
    ]


def test_authenticated_resolution_and_terminal(tmp_path):
    """SCENARIO-REPORT-8057-CONSUMERS: exact paths and IDs reject ambiguous alternatives."""
    task = {"id": "exp9000-fixture", "deliverable": "results/experiment_9000_declared.json"}
    target = tmp_path / task["deliverable"]
    target.parent.mkdir()
    with pytest.raises(ValueError, match="missing_declared"):
        e.resolve(tmp_path, task)
    atomic_json(target, {"task_id": "exp9000-other"})
    with pytest.raises(ValueError, match="task_identity"):
        e.resolve(tmp_path, task)
    atomic_json(target, {"task_id": task["id"]})
    assert e.resolve(tmp_path, task) == target
    atomic_json(target.with_name("experiment_9000_alternative.json"), {"task_id": task["id"]})
    with pytest.raises(ValueError, match="ambiguous"):
        e.resolve(tmp_path, task)
    with pytest.raises(ValueError):
        e.clean_terminal(target)
    assert e.clean_terminal(e.ROOT / "results/experiment_8045_v697_scorer_workspace.json")["passed"]


def invoke(tmp_path, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if env.get("CARNOT_8057_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["CARNOT_8057_COVERAGE_CONFIG"]]
    return subprocess.run(
        [*prefix, str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_real_cli_success_blocked_mutation(tmp_path):
    """SCENARIO-REPORT-8057-TERMINAL: actual outside-checkout exits qualify runner routes."""
    design, staged, active = authorities(tmp_path)
    out = tmp_path / "results" / (e.NAME + ".json")
    args = ["--fixture-output", out, "--design", design, "--staged", staged, "--active", active]
    success = invoke(tmp_path, *args)
    assert success.returncode == 0, success.stdout + success.stderr
    assert json.loads(out.read_text())["fixture_consumer_ready_score"] == 1
    assert invoke(tmp_path, "--cold-replay", out).returncode == 0
    assert invoke(tmp_path, *args, "--mutate").returncode == 0
    assert json.loads(out.read_text())["verdict_class"] == "disqualified"
    assert invoke(tmp_path, "--cold-replay", out).returncode == 0
    assert invoke(tmp_path, *args, "--root", tmp_path / "absent").returncode == 0
    assert json.loads(out.read_text())["verdict_class"] == "blocked"
    value = json.loads(out.read_text())
    value["fixture_consumer_ready_score"] = 1
    atomic_json(out, value)
    assert invoke(tmp_path, "--cold-replay", out).returncode == 1
    assert invoke(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert invoke(tmp_path, "--date", "bad").returncode == 2


def test_existing_reader_accepts_v698_digest_label(tmp_path):
    """REQ-REPORT-8057: the parameterized reader accepts the actual design heading."""
    design, staged, active = authorities(tmp_path)
    result = e.authority.assess_authorities(
        design, staged, active, tmp_path / "reader", milestone=e.MILESTONE, first_id=8057, count=13
    )
    assert result["activated"]


def test_fixture_reduction_and_workspace_are_required():
    """SCENARIO-REPORT-8057-SCORER: workspace and strict numeric operands cannot be skipped."""
    rows = e.scorer.fixture_rows()
    workspace = {
        "before": {"actual_exit_code": 1, "missing_parent_detected": True},
        "after": {"actual_exit_code": 0, "parent_exists_before_launch": True},
    }
    assert e.qualify_fixture(rows, workspace)
    for field in ("conditional_logit_positions", "normalization_max_error", "target_logprobs"):
        bad = deepcopy(rows)
        if field == "conditional_logit_positions":
            bad[0][field][0] += 1
        elif field == "target_logprobs":
            bad[0][field][0] += 2e-6
        else:
            bad[0][field] = 1e-4
        assert not e.qualify_fixture(bad, workspace)
    bad = deepcopy(workspace)
    bad["after"]["actual_exit_code"] = 1
    assert not e.qualify_fixture(rows, bad)


def test_terminal_hash_and_quarantine_controls(tmp_path):
    """SCENARIO-REPORT-8057-CONSUMERS: changed primary bytes fail authenticated sidecars."""
    primary = tmp_path / "experiment_9000_control.json"
    sidecar = tmp_path / "raw" / primary.stem / "validator.json"
    terminal = tmp_path / "terminal.json"
    value = {
        "task_id": "exp9000-control",
        "terminal_validation_sidecar_path": str(terminal),
        "flagged_adversarial": False,
    }
    atomic_json(primary, value)
    digest = sha256_file(primary)
    atomic_json(sidecar, {"primary_sha256": digest, "report": {"passed": True}})
    atomic_json(terminal, {"publication": {"primary_sha256": digest, "sidecar_path": str(sidecar)}})
    assert e.clean_terminal(primary)["passed"]
    value["changed"] = True
    atomic_json(primary, value)
    with pytest.raises(ValueError, match="terminal_hash"):
        e.clean_terminal(primary)
    value["flagged_adversarial"] = True
    atomic_json(primary, value)
    digest = sha256_file(primary)
    atomic_json(sidecar, {"primary_sha256": digest, "report": {"passed": True}})
    atomic_json(terminal, {"publication": {"primary_sha256": digest, "sidecar_path": str(sidecar)}})
    with pytest.raises(ValueError, match="unclean_terminal"):
        e.clean_terminal(primary)


def test_production_orchestration_and_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8057-TERMINAL: production manifests, reductions and publication bind bytes."""
    design, staged, active = authorities(tmp_path)
    out = tmp_path / "results" / (e.NAME + ".json")
    specs = e.manifest(tmp_path)
    assert {s["name"] for s in specs} >= {"coverage_report", "consumer_tests", "strict_mypy"}
    assert any("--fail-under=100" in s["argv"] for s in specs)
    from carnot.reporting import v686_contract_validation as validation

    def run_check(root, spec, private, durable, **kwargs):
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        if spec["name"] == "cold_replay":
            assert e.replay(Path(spec["argv"][-1]))
        if spec["name"] == "coverage_json":
            atomic_json(
                private / "coverage.json", {"totals": {"num_statements": 1, "covered_lines": 1}}
            )
        log.write_text("private orchestration control\n")
        return dict(
            name=spec["name"],
            passed=True,
            log_path=str(log),
            log_sha256=sha256_file(log),
            argv=spec["argv"],
            exit_code=0,
            duration_s=0.01,
        )

    monkeypatch.setattr(validation, "run_check", run_check)
    args = [
        "--design",
        str(design),
        "--staged",
        str(staged),
        "--active",
        str(active),
        "--output",
        str(out),
    ]
    assert e.main(args) == 0
    assert e.replay(out)
    value = json.loads(out.read_text())
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    work = json.loads((raw / "work.json").read_text())
    failed = deepcopy(value["validation_receipts"])
    failed[0]["passed"] = False
    bad = e.build(
        work,
        raw,
        value["current_work_receipt"]["started_monotonic_ns"],
        value["current_work_receipt"]["ended_monotonic_ns"],
        failed,
    )
    assert bad["fixture_consumer_ready_score"] == 0 and bad["verdict_class"] == "disqualified"
    assert bad["gate_check_summary"]
    log = Path(value["validation_receipts"][0]["log_path"])
    previous = log.read_bytes()
    log.write_text("tampered")
    assert not e.replay(out)
    log.write_bytes(previous)
    shard = raw / "work.json"
    original = shard.read_bytes()
    shard.write_text("{}")
    assert not e.replay(out)
    shard.write_bytes(original)
    value["code_config_hashes"][e.MODULE] = "sha256:bad"
    atomic_json(out, value)
    assert not e.replay(out)


def test_outside_checkout_production_blocked_publication(tmp_path):
    """SCENARIO-REPORT-8057-TERMINAL: real blocked publication passes independent strict readers."""
    out = tmp_path / "results" / (e.NAME + ".json")
    result = invoke(tmp_path, "--root", tmp_path / "absent", "--output", out)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(out.read_text())["verdict_class"] == "blocked"
    assert invoke(tmp_path, "--cold-replay", out).returncode == 0


def test_design_digest_tamper_and_snapshot_replay(tmp_path):
    """SCENARIO-REPORT-8057-CONSUMERS: embedded prompt drift cannot borrow an active digest."""
    design, staged, active = authorities(tmp_path)
    design.write_text(
        design.read_text().replace(
            '"title": "Qualify fixture consumers and bind the complete V698 contract"',
            '"title": "changed embedded title"',
        )
    )
    assert not e.assess(design, staged, active, tmp_path / "snapshots")["activated"]


def test_missing_terminal_and_changed_scorer_fail_closed(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8057-SCORER: stale terminals and scorer versions cannot qualify readiness."""
    result = tmp_path / "results"
    result.mkdir()
    for n in (8044, 8045, 8047, 8056):
        source = next((e.ROOT / "results").glob(f"experiment_{n}_*.json"))
        (result / source.name).write_bytes(source.read_bytes())
    primary = result / "experiment_8045_v697_scorer_workspace.json"
    value = json.loads(primary.read_text())
    value["terminal_validation_sidecar_path"] = str(tmp_path / "missing")
    value["scorer_fixture_ready_score"] = 0
    value["scorer_code_hashes"][0]["sha256"] = "sha256:changed"
    atomic_json(primary, value)
    failures, _ = e.preconditions(tmp_path)
    assert any(f["field"] == "clean_terminal" for f in failures)
    monkeypatch.setattr(e, "preconditions", lambda root: ([], []))
    design, staged, active = authorities(tmp_path)
    work = e.measure(
        tmp_path, design, staged, active, tmp_path / "raw", tmp_path / "private", False
    )
    assert {f["field"] for f in work["failures"]} >= {
        "scorer_dependency_hash",
        "scorer_fixture_ready_score",
    }
    old = json.loads((result / "experiment_8044_v697_contract_methods.json").read_text())
    old["authority_snapshots"]["active"]["sha256"] = "sha256:changed"
    atomic_json(result / "experiment_8044_v697_contract_methods.json", old)
    with pytest.raises(ValueError, match="prior_authority_hash"):
        e.historical(tmp_path, tmp_path / "history")


def test_frozen_input_mutation_rejected(tmp_path):
    """SCENARIO-REPORT-8057-TERMINAL: replay verifies saved input bytes as well as claims."""
    design, staged, active = authorities(tmp_path)
    out = tmp_path / "results" / (e.NAME + ".json")
    assert (
        invoke(
            tmp_path,
            "--fixture-output",
            out,
            "--design",
            design,
            "--staged",
            staged,
            "--active",
            active,
        ).returncode
        == 0
    )
    value = json.loads(out.read_text())
    snapshot = Path(value["source_artifact_hashes"][0]["snapshot_path"])
    snapshot.write_text("changed snapshot")
    assert not e.replay(out)


def test_missing_authority_is_external_block(tmp_path):
    """SCENARIO-REPORT-8057-TERMINAL: missing authority is absent evidence, not a retryable partial."""
    work = e.measure(
        e.ROOT,
        tmp_path / "missing-design",
        tmp_path / "staged",
        tmp_path / "missing-active",
        tmp_path / "raw",
        tmp_path,
        False,
    )
    assert all(f["field"] == "resource_exists" for f in work["failures"])
    assert work["fixture_rows"] == []


def test_real_fixture_consumer_is_scoped(tmp_path):
    """SCENARIO-REPORT-8057-CONSUMERS: authenticated Exp8045 admits measurements and rejects science."""
    result = e.fixture_consumer(
        e.ROOT / "results/experiment_8045_v697_scorer_workspace.json", tmp_path
    )
    assert result["fixture_passed"] and not result["science_passed"]
    assert result["terminal_receipt"]["passed"]
