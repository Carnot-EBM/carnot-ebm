"""REQ-REPORT-7903-V686: private authorities, source custody and terminal bytes."""

from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting import v686_contract_methods as methods
from carnot.reporting import v686_contract_validation as validation
from scripts.experiments import experiment_7903_v686_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/v686"
SOURCE = ROOT / "results/experiment_7892_v685_source_boundary.json"


@pytest.fixture
def authorities(tmp_path):
    """SCENARIO-REPORT-7903-AUTHORITY: tests cannot follow active roadmap edits."""
    design, staged, active = (
        tmp_path / name for name in ("design.md", "stage.yaml", "active.yaml")
    )
    design.write_bytes(gzip.decompress((FIXTURE / "design.md.gz").read_bytes()))
    active.write_bytes(gzip.decompress((FIXTURE / "active.yaml.gz").read_bytes()))
    staged.write_bytes(active.read_bytes())
    return design, staged, active, tmp_path / "snapshots"


def test_versioned_authority_and_private_mutations(authorities, tmp_path):
    """SCENARIO-REPORT-7903-AUTHORITY: lifecycle uses the complete versioned digest."""
    design, staged, active, snapshots = authorities
    result = methods.assess(design, staged, active, snapshots)
    assert result["activated"] and len(result["contract_rows"]) == 12
    assert result["canonical_tasks_sha256"] == methods.FIXTURE_DIGEST
    staged.unlink()
    assert methods.assess(design, staged, active, snapshots)["activated"]
    staged.write_text("milestone: 2026.10.687\ntasks: []\n")
    assert methods.assess(design, staged, active, snapshots)["activated"]
    rows = methods.mutations(design, active, SOURCE, tmp_path / "private")
    assert len(rows) == 12 and all(row["passed"] for row in rows)
    assert {row["unit_id"] for row in rows} >= {
        "source_hash_drift",
        "later_staging",
        "stale_active",
        "prior_failure_deletion",
        "prompt_edit",
        "wrong_model",
        "wrong_class",
    }


@pytest.mark.parametrize("missing", ["table", "json", "digest", "file", "active"])
def test_missing_authority_is_terminal_block(authorities, missing):
    """SCENARIO-REPORT-7903-AUTHORITY: missing evidence cannot be synthesized."""
    design, staged, active, snapshots = authorities
    if missing == "file":
        design.unlink()
    elif missing == "active":
        active.unlink()
    else:
        text = design.read_text()
        text = text.replace(
            {
                "table": "## Exact task contract",
                "json": "V686_TASK_CONTRACT_START",
                "digest": "Canonical full-task SHA-256",
            }[missing],
            "missing",
        )
        design.write_text(text)
    result = methods.assess(design, staged, active, snapshots)
    assert not result["activated"] and result["gate_check_summary"]
    assert len(result["contract_rows"]) == 12
    assert result["authority_snapshots"]["active"]["source_path"] == str(active)
    assert result["authority_snapshots"]["staged"]["source_path"] == str(staged)


def test_source_custody_and_hash_failure(tmp_path):
    """SCENARIO-REPORT-7903-SOURCE: reauthenticate shards without a producer run."""
    result = methods.source_custody(SOURCE, methods.SOURCE_SHA256)
    assert result["ready"] and not result["gate_check_summary"]
    assert result["budget"]["intended"] == 640
    assert result["budget"]["eligible"] == 604 and result["budget"]["excluded"] == 36
    assert len(result["rows"]) == 640 and len(result["hashes"]) >= 8
    bad = methods.source_custody(SOURCE, "sha256:wrong")
    assert not bad["ready"] and bad["gate_check_summary"][0]["artifact_field"] == "sha256"
    missing = methods.source_custody(tmp_path / "missing", methods.SOURCE_SHA256)
    assert not missing["ready"] and missing["gate_check_summary"][0]["observed"] is None


def test_manifest_and_method_freeze(tmp_path):
    """SCENARIO-REPORT-7903-VALIDATION: argv and science scope precede results."""
    manifest = validation.manifest(ROOT, tmp_path)
    assert "python/carnot/__init__.py" in manifest["dependency_hashes"]
    assert all(row["argv"] and row["deadline_s"] <= 900 for row in manifest["commands"])
    assert all(row["expected_exit"] in (0, 1) for row in manifest["commands"])
    e2e = [row for row in manifest["commands"] if row["name"].startswith("e2e_016")]
    assert len(e2e) == 2 and all("20260930" in row["argv"] for row in e2e)
    assert all(str(tmp_path) in row["argv"][-1] for row in e2e)
    freeze = methods.method_freeze(ROOT)
    assert freeze["delayed_aci"]["tau"] == [1, 4, 16]
    assert freeze["delayed_aci"]["bootstrap_unit"] == "contiguous_16_event_block"
    assert freeze["service_cost"]["denominators"] == ["processed", "accepted", "caught_error"]
    assert len(freeze["future_validation_scopes"]) == 12


def test_real_private_cli_and_cold_replay(authorities, tmp_path):
    """SCENARIO-REPORT-7903-VALIDATION: the real CLI and reducer reject drift."""
    design, staged, active, _ = authorities
    output, raw = tmp_path / "candidate.json", tmp_path / "rows.json"
    argv = [
        sys.executable,
        str(Path(cli.__file__)),
        "--date",
        "20260930",
        "--design",
        str(design),
        "--staged",
        str(staged),
        "--active",
        str(active),
        "--output",
        str(output),
        "--raw",
        str(raw),
        "--fixture-e2e",
    ]
    done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert methods.cold_replay(output, raw)
    replay = [
        sys.executable,
        str(Path(cli.__file__)),
        "--date",
        "20260930",
        "--cold-replay",
        str(output),
        "--raw",
        str(raw),
    ]
    assert subprocess.run(replay, cwd=ROOT, capture_output=True, timeout=60).returncode == 0
    value = json.loads(output.read_text())
    value["sample_size_budget"]["completed"] = 0
    atomic_json(output, value)
    assert not methods.cold_replay(output, raw)
    assert subprocess.run(replay, cwd=ROOT, capture_output=True, timeout=60).returncode == 1


def test_required_failures_disqualify_and_external_missing_blocks():
    """SCENARIO-REPORT-7903-VALIDATION: scope and outside prerequisites differ."""
    assert methods.verdict(False, True, True) == ("complete_blocked_authority", "blocked", 0)
    assert methods.verdict(True, False, True) == ("complete_blocked_source_custody", "blocked", 0)
    assert methods.verdict(True, True, False)[1:] == ("disqualified", 0)
    assert methods.verdict(True, True, True)[1:] == ("circular_positive", 1)
    assert methods.verdict(False, True, False)[1:] == ("disqualified", 0)


def test_source_mutated_receipt_operands(tmp_path):
    """SCENARIO-REPORT-7903-SOURCE: authenticated wrong fields still fail custody."""
    value = json.loads(SOURCE.read_text())
    value["experiment_id"] = 1
    value["validation_receipts"][0]["passed"] = False
    value["public_shards"][0]["sha256"] = "sha256:drift"
    output = tmp_path / "source.json"
    atomic_json(output, value)
    result = methods.source_custody(output, sha256_file(output))
    assert not result["ready"]
    assert {row["artifact_field"] for row in result["gate_check_summary"]} >= {
        "experiment_id",
        "sha256",
        "public_shards.family_ids",
        "validation_receipts.affected_pytest.passed",
    }


def test_malformed_external_inputs_and_design_digest(authorities, tmp_path):
    """SCENARIO-REPORT-7903-AUTHORITY: malformed external evidence ends as a block."""
    design, staged, active, snapshots = authorities
    text = design.read_text().replace('"prompt":', '"prompt_changed":', 1)
    design.write_text(text)
    result = methods.assess(design, staged, active, snapshots)
    assert not result["activated"]
    assert any(
        row["artifact_field"] == "design_tasks_sha256" for row in result["gate_check_summary"]
    )
    design.write_text(
        "## Exact task contract\nV686_TASK_CONTRACT_START\nCanonical full-task SHA-256: `"
        + "a" * 64
        + "`\n"
    )
    assert not methods.assess(design, staged, active, snapshots)["activated"]
    source = tmp_path / "malformed.json"
    source.write_text("{}")
    failure = methods.source_custody(source, sha256_file(source))
    assert (
        not failure["ready"]
        and failure["gate_check_summary"][0]["artifact_field"] == "source_schema"
    )


def test_candidate_fields_and_primitive_replay(authorities, tmp_path):
    """SCENARIO-REPORT-7903-VALIDATION: counters derive from rows and owned receipts."""
    assessment = methods.assess(*authorities)
    custody = methods.source_custody(SOURCE, methods.SOURCE_SHA256)
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, validation.manifest(ROOT, tmp_path))
    artifact = methods.candidate(
        ROOT,
        assessment,
        custody,
        methods.method_freeze(ROOT),
        [{"passed": True}],
        [
            {"passed": True, "argv": ["true"]},
            {
                "passed": False,
                "argv": ["pytest"],
                "classification": "diagnostic",
                "name": "repository_full_suite",
            },
        ],
        manifest_path,
        100,
        200,
    )
    assert artifact["contract_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert (
        artifact["inference_substrate_class"] == "no_model_load" and artifact["MODEL_SPECS"] == []
    )
    assert set(artifact) <= set(artifact["field_principles"])
    assert artifact["acceptance_gate_results"]["decision_benefit"] is None
    output, raw = tmp_path / "candidate.json", tmp_path / "rows.json"
    atomic_json(output, artifact)
    atomic_json(raw, methods.primitive_rows(artifact))
    assert methods.cold_replay(output, raw)
    snapshot = Path(artifact["authority_snapshots"]["active"]["snapshot_path"])
    original_snapshot = snapshot.read_bytes()
    snapshot.write_bytes(b"drift")
    assert not methods.cold_replay(output, raw)
    snapshot.write_bytes(original_snapshot)
    changed = deepcopy(artifact)
    changed["rows"][0]["absolute_metric"] = 2
    atomic_json(output, changed)
    assert not methods.cold_replay(output, raw)
    assert not methods.cold_replay(tmp_path / "missing", raw)
    atomic_json(output, {"experiment_id": 1})
    assert not methods.cold_replay(output, raw)


def test_child_logs_expected_failure_and_timeout(tmp_path):
    """SCENARIO-REPORT-7903-VALIDATION: expected exits and bounded waits remain measured."""
    spec = {
        "name": "expected_failure",
        "argv": [sys.executable, "-c", "raise SystemExit(1)"],
        "expected_exit": 1,
        "deadline_s": 10,
        "failure_reason": "private rejection",
    }
    result = validation.run_check(ROOT, spec, tmp_path, tmp_path / "sealed")
    assert result["passed"] and result["actual_exit"] == 1
    assert sha256_file(Path(result["log_path"])) == result["log_sha256"]
    spec.update(
        name="deadline",
        argv=[sys.executable, "-c", "import time; time.sleep(10)"],
        expected_exit=0,
        deadline_s=0.01,
    )
    result = validation.run_check(ROOT, spec, tmp_path, tmp_path / "sealed", heartbeat_s=0.01)
    assert result["timed_out"] and not result["passed"]


def test_terminal_recheck_binds_final_bytes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7903-VALIDATION: failed final checks cannot publish unchecked bytes."""
    output, private, sealed = tmp_path / "result.json", tmp_path / "private", tmp_path / "sealed"
    artifact = {
        "honest_verdict": "complete_circular_positive_contract_methods",
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "contract_ready_score": 1,
        "acceptance_gate_results": {"readiness": 1},
        "gate_check_summary": [],
    }

    def check(root, spec, scratch, durable):
        return {"name": spec["name"], "passed": True, "actual_exit": 0}

    monkeypatch.setattr(validation, "run_check", check)
    validation.publish(ROOT, artifact, output, private, sealed)
    sidecar = next(sealed.glob("terminal-*.json"))
    assert json.loads(sidecar.read_text())["candidate_sha256"] == sha256_file(output)
    outputs = iter([False, True, True, True])

    def recover(root, spec, scratch, durable):
        return {"name": spec["name"], "passed": next(outputs), "actual_exit": 0}

    monkeypatch.setattr(validation, "run_check", recover)
    validation.publish(ROOT, artifact, output, private, sealed)
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(validation, "run_check", lambda *a: {"passed": False, "actual_exit": 1})
    rejected = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="terminal_recheck_failed"):
        validation.publish(ROOT, artifact, rejected, private, sealed)
    assert not rejected.exists()


def test_nonempty_per_file_coverage_required(tmp_path):
    """SCENARIO-REPORT-7903-VALIDATION: absent files cannot hide in an aggregate."""
    report = tmp_path / "coverage.json"
    atomic_json(
        report,
        {
            "files": {
                name: {"summary": {"num_statements": 1, "covered_lines": 1}, "missing_lines": []}
                for name in validation.CHANGED
            }
        },
    )
    assert validation.coverage_complete(report)
    value = json.loads(report.read_text())
    value["files"].pop(validation.CHANGED[0])
    atomic_json(report, value)
    assert not validation.coverage_complete(report)
    assert not validation.coverage_complete(tmp_path / "missing.json")


def test_blocked_without_active_rows_and_owned_failure(authorities, tmp_path):
    """SCENARIO-REPORT-7903-AUTHORITY: a wholly absent authority still has twelve slots."""
    design, staged, active, snapshots = authorities
    design.unlink()
    active.unlink()
    assessment = methods.assess(design, staged, active, snapshots)
    assert len(assessment["contract_rows"]) == 12
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, {})
    custody = {
        "ready": True,
        "gate_check_summary": [],
        "hashes": [],
        "rows": [],
        "budget": methods.boundary.budget([], 640),
    }
    artifact = methods.candidate(
        ROOT,
        assessment,
        custody,
        methods.method_freeze(ROOT),
        [],
        [{"name": "owned_failure", "argv": ["false"], "passed": False}],
        manifest_path,
        0,
        1,
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["gate_check_summary"][-1]["artifact_field"] == "required_check.passed"


def test_execute_qualification_and_main_guards(authorities, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7903-VALIDATION: all orchestration runs on private frozen evidence."""
    design, staged, active, _ = authorities
    args = SimpleNamespace(
        design=design,
        staged=staged,
        active=active,
        source=SOURCE,
        output=tmp_path / "result.json",
        raw=tmp_path / "raw/rows.json",
        fixture_e2e=False,
    )
    original = validation.manifest

    def manifest(root, private):
        frozen = original(root, private)
        frozen["commands"] = [
            {"name": "private_check", "argv": ["true"], "expected_exit": 0, "deadline_s": 1}
        ]
        return frozen

    def child(root, spec, private, durable):
        atomic_json(
            private / "coverage.json",
            {
                "files": {
                    name: {
                        "summary": {"num_statements": 1, "covered_lines": 1},
                        "missing_lines": [],
                    }
                    for name in validation.CHANGED
                }
            },
        )
        (private / ".coverage.combined").write_bytes(b"private coverage fixture")
        return {**spec, "passed": True, "actual_exit": 0}

    monkeypatch.setattr(validation, "manifest", manifest)
    monkeypatch.setattr(validation, "run_check", child)
    monkeypatch.setattr(
        validation, "publish", lambda root, value, output, *a: atomic_json(output, value)
    )
    result = validation.execute(ROOT, args, tmp_path / "private")
    assert result["contract_ready_score"] == 1 and methods.cold_replay(args.output, args.raw)
    monkeypatch.setattr(cli, "execute", lambda *a: result)
    monkeypatch.setattr(sys, "argv", ["exp7903", "--output", str(args.output)])
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", ["exp7903", "--fixture-e2e"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    monkeypatch.setattr(sys, "argv", ["exp7903", "--date", "wrong"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_terminal_copy_hash_mismatch_rejects(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7903-VALIDATION: checked bytes cannot drift during publication."""
    monkeypatch.setattr(validation, "run_check", lambda *a: {"passed": True, "actual_exit": 0})
    original = validation.sha256_file
    monkeypatch.setattr(
        validation,
        "sha256_file",
        lambda p: "sha256:drift" if p.name.endswith(".checked") else original(p),
    )
    with pytest.raises(ValueError, match="published_bytes_differ"):
        validation.publish(
            ROOT, {}, tmp_path / "result.json", tmp_path / "private", tmp_path / "sealed"
        )
