"""Executable V681 audit requirements (REQ-REPORT-7849)."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
from carnot.reporting import v681_independent_audit as audit


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_7849_v681_independent_audit.py"


def test_exact_source_inventory_and_blocked_operands():
    """SCENARIO-REPORT-7849-ABSENT: retain all eight declared branches."""
    sources, failures = audit.inspect_sources(ROOT)
    assert [row["upstream_id"] for row in sources] == [
        f"Exp{number}" for number in (7838, 7840, 7841, 7842, 7843, 7844, 7846, 7848)
    ]
    assert sources[0]["state"] == "blocked"
    assert sources[-1]["state"] == "disqualified"
    assert {row["state"] for row in sources[1:-1]} == {"missing", "conductor_only"}
    assert any(row["artifact_field"] == "experiment_id" for row in failures)
    assert any(row["artifact_field"] == "science_producer" for row in failures)
    assert all(
        set(row) >= {"upstream_id", "path", "hash", "artifact_field", "op", "expected", "observed"}
        for row in failures
    )


def test_identity_mutation_and_qualified_null(tmp_path):
    """SCENARIO-REPORT-7849-MUTATIONS: integer and slug are separate."""
    source = {
        "experiment_id": 7841,
        "task_id": "exp7841-decision-measurement",
        "milestone": "2026.09.681",
        "run_date": "20260929",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "decision_evidence_ready_score": 1,
    }
    assert (
        audit.producer_failures(7841, "exp7841-decision-measurement", tmp_path / "x", source) == []
    )
    source["experiment_id"] = source["task_id"]
    failures = audit.producer_failures(7841, "exp7841-decision-measurement", tmp_path / "x", source)
    assert any(row["artifact_field"] == "experiment_id" for row in failures)


def test_primitive_negative_mutations():
    """SCENARIO-REPORT-7849-MUTATIONS: fail private, future and replay leaks."""
    clean = {
        "family_id": "one",
        "role": "online_admission",
        "seed": 68101,
        "features": {"length": 3.0},
        "label": 1,
        "probability": 0.7,
        "action": "escalate",
        "prediction_step": 1,
        "feedback_step": 2,
        "admission_family_id": "one",
        "status": "completed",
        "feedback_replayed": False,
        "shuffle_label_step": 0,
    }
    assert audit.check_primitive_rows([clean], intended=1) == []
    changes = [
        ({"features": {"gold_label": 1}}, "private_feature"),
        ({"feedback_step": 0}, "future_feedback"),
        ({"feedback_replayed": True, "status": "dropped"}, "dropped_feedback_replayed"),
        ({"shuffle_label_step": 3}, "future_shuffle_label"),
        ({"admission_family_id": "other"}, "admission_family"),
    ]
    for change, failure in changes:
        assert failure in audit.check_primitive_rows([{**clean, **change}], intended=1)
    assert "lost_censoring_rows" in audit.check_primitive_rows([clean], intended=2)
    assert "seed_pseudoreplication" in audit.check_primitive_rows(
        [clean, {**clean, "seed": 68102}], intended=2
    )


def test_length_raw_join_recomputes_ineligible_diagnostic():
    """SCENARIO-REPORT-7849-ABSENT: raw length values survive without readiness."""
    result = audit.reduce_length(ROOT)
    assert result["eligible"] is False
    assert result["independent_n"] == 64
    assert abs(result["means"]["length"]["brier"] - 0.22403651576351566) < 1e-12
    assert result["means"]["length"]["cost"] == 0.25
    assert len(result["rows"]) == 64
    assert all(row["status"] == "completed" for row in result["rows"])


def test_blocked_artifact_replay_and_mutated_source(tmp_path):
    """SCENARIO-REPORT-7849-VALIDATION: candidate is terminal and byte bound."""
    result = audit.build_candidate(ROOT, tmp_path, "20260929")
    assert result["experiment_id"] == 7849
    assert result["task_id"] == "exp7849-independent-audit"
    assert result["honest_verdict"] == "complete_blocked_required_v681_science"
    assert result["verdict_class"] == "blocked"
    assert result["independent_evidence_ready_score"] == 0
    assert len(result["rows"]) == 8
    assert result["acceptance_gate_results"]["decision_benefit"] is None
    assert result["gate_check_summary"]
    assert all(item["rejected"] for item in result["mutation_results"])
    assert {row["name"] for row in result["repository_health"]["historical_required_failures"]} == {
        "full_python_pytest",
        "coverage_report",
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(result))
    assert audit.cold_replay(path, ROOT) == []
    changed = copy.deepcopy(result)
    changed["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
    path.write_text(json.dumps(changed))
    assert "source_bytes_changed" in audit.cold_replay(path, ROOT)


def test_real_cli_rejects_slug_and_missing_date(tmp_path):
    """SCENARIO-REPORT-7849-MUTATIONS: real CLI rejects identity substitution."""
    run = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--date",
            "20260929",
            "--science-only",
            "--output-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    candidate = tmp_path / "candidate.json"
    data = json.loads(candidate.read_text())
    data["experiment_id"] = data["task_id"]
    candidate.write_text(json.dumps(data))
    check = subprocess.run(
        [sys.executable, str(CLI), "--check-only", str(candidate)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert check.returncode != 0 and "experiment_id" in check.stdout
    bad = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--date",
            "20260928",
            "--science-only",
            "--output-root",
            str(tmp_path / "wrong"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0


def test_reader_invalid_json_and_eligible_null(tmp_path):
    """REQ-REPORT-7849: malformed files fail and a qualified null is admitted."""
    authority = tmp_path / audit.AUTHORITY
    authority.parent.mkdir(parents=True)
    paths = {n: f"results/experiment_{n}_science.json" for n in audit.NUMBERS}
    authority.write_text(
        json.dumps(
            {"tasks": [{"id": f"exp{n}-science", "deliverable": path} for n, path in paths.items()]}
        )
    )
    producer = tmp_path / paths[7838]
    producer.parent.mkdir(parents=True)
    producer.write_text("{")
    receipt = tmp_path / "results/experiment_7840_skip.json"
    receipt.write_text("{")
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["state"] == "disqualified"
    assert sources[1]["state"] == "conductor_only"
    assert any(x["artifact_field"] == "science_producer" for x in failures)
    producer.write_text(
        json.dumps(
            {
                "experiment_id": 7838,
                "task_id": "exp7838-science",
                "milestone": "2026.09.681",
                "run_date": "20260929",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "source_boundary_ready_score": 1,
            }
        )
    )
    sources, failures = audit.inspect_sources(tmp_path)
    assert sources[0]["state"] == "eligible" and sources[0]["eligibility"]
    assert not any(x["upstream_id"] == "Exp7838" for x in failures)
    producer.write_text(json.dumps({"run_date": "wrong"}))
    failures = audit.producer_failures(
        7838, "exp7838-science", producer, json.loads(producer.read_text())
    )
    assert any(x["artifact_field"] == "run_date" for x in failures)


def test_length_rejects_bad_roster_label_and_probability(tmp_path):
    """REQ-REPORT-7849: raw primitive joins fail closed."""
    assert audit.reduce_length(tmp_path)["status"] == "missing_raw"
    predictions = (
        tmp_path
        / "results/raw/experiment_7848_v681_length_shortcut/current/evaluation_predictions.json"
    )
    labels = (
        tmp_path / "results/raw/experiment_7727_v673_development_corpus/evaluation_evaluator.jsonl"
    )
    predictions.parent.mkdir(parents=True)
    labels.parent.mkdir(parents=True)
    row = {
        "family_id": "f",
        "length_risk": 0.01,
        "prevalence_risk": 0.99,
        "answer_bytes": 1,
        "source_bytes": 2,
        "stratum": 0,
    }
    predictions.write_text(json.dumps({"rows": [row, row]}))
    labels.write_text(json.dumps({"family_id": "f", "label": 1}) + "\n")
    with pytest.raises(ValueError, match="family_join"):
        audit.reduce_length(tmp_path)
    predictions.write_text(json.dumps({"rows": [row]}))
    labels.write_text(json.dumps({"family_id": "f", "label": -1}) + "\n")
    with pytest.raises(ValueError, match="label_invalid"):
        audit.reduce_length(tmp_path)
    labels.write_text(json.dumps({"family_id": "f", "label": 1}) + "\n")
    row["length_risk"] = 2
    predictions.write_text(json.dumps({"rows": [row]}))
    with pytest.raises(ValueError, match="probability_invalid"):
        audit.reduce_length(tmp_path)
    row["length_risk"] = 0.01
    predictions.write_text(json.dumps({"rows": [row]}))
    out = audit.reduce_length(tmp_path)
    assert out["rows"][0]["arms"]["length"]["action"] == "accept"
    assert out["rows"][0]["arms"]["prevalence"]["action"] == "reject"
    assert "unknown_label_loss" in audit.check_primitive_rows(
        [{"family_id": "f", "label": -1}], intended=1
    )


def test_replay_rejects_missing_bad_rows_metrics_and_log(tmp_path):
    """SCENARIO-REPORT-7849-MUTATIONS: every sealed byte is checked."""
    assert audit.cold_replay(tmp_path / "absent.json", ROOT) == ["candidate_unreadable"]
    base = audit.build_candidate(ROOT, tmp_path, "20260929")
    path = tmp_path / "candidate.json"
    log = tmp_path / "sealed.log"
    log.write_text("first")
    for change, expected in (
        ({"rows": []}, "rows_changed"),
        ({"recomputed_metrics": {"length_control_ineligible": {}}}, "metric_changed"),
        (
            {"validation_receipts": [{"log_path": str(log), "log_sha256": "sha256:wrong"}]},
            "validation_log_changed",
        ),
        ({"milestone": "wrong"}, "milestone"),
    ):
        path.write_text(json.dumps({**base, **change}))
        assert expected in audit.cold_replay(path, ROOT)


def _cli_module():
    """Load the real CLI as a module for bounded dispatcher assertions."""
    spec = importlib.util.spec_from_file_location("experiment_7849_cli_test", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dispatch_seals_unique_logs_and_preserves_required_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7849-VALIDATION: real manifest names and exits survive."""
    cli = _cli_module()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    manifest = tmp_path / audit.MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes((ROOT / audit.MANIFEST).read_bytes())
    result = audit.build_candidate(ROOT, tmp_path, "20260929")

    def fake_run_commands(root, specs, *, log_dir, heartbeat_s):
        name = specs[0].name
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{name}.log"
        content = json.dumps({"flagged_count": 0}) if name == "adversarial_verify" else name
        log.write_text(content)
        return [
            {
                "name": name,
                "log_path": str(log.relative_to(root)),
                "exit_code": 2 if name == "ruff_check" else 0,
                "passed": name != "ruff_check",
                "output_tail": content,
                "command_argv": list(specs[0].argv),
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run_commands)
    validated = cli.validate(result, tmp_path / "owned", time.monotonic())
    assert validated["verdict_class"] == "disqualified"
    assert validated["required_validation_failures"] == ["ruff_check"]
    assert validated["flagged_adversarial"] is False
    assert validated["repository_health"]["status"] == "passed"
    assert len(validated["validation_receipts"]) == 12
    assert all(
        Path(row["log_path"]).name.endswith(".log") for row in validated["validation_receipts"]
    )
    assert audit.cold_replay(tmp_path / "owned/candidate.json", ROOT) == []


def test_dispatch_bad_manifest_and_verifier_output(tmp_path, monkeypatch):
    """REQ-REPORT-7849: a changed required list or unreadable oracle fails."""
    cli = _cli_module()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    manifest = tmp_path / audit.MANIFEST
    manifest.parent.mkdir(parents=True)
    data = json.loads((ROOT / audit.MANIFEST).read_text())
    data["commands"] = data["commands"][:-1]
    data["commands"][0]["name"] = "wrong"
    manifest.write_text(json.dumps(data))
    result = audit.build_candidate(ROOT, tmp_path, "20260929")
    with pytest.raises(ValueError, match="required_manifest_changed"):
        cli.validate(result, tmp_path / "owned", time.monotonic())
    data = json.loads((ROOT / audit.MANIFEST).read_text())
    manifest.write_text(json.dumps(data))

    def unreadable(root, specs, *, log_dir, heartbeat_s):
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "child.log"
        log.write_text("not json")
        return [
            {
                "name": specs[0].name,
                "log_path": str(log.relative_to(root)),
                "exit_code": 0,
                "passed": True,
                "output_tail": "not json",
            }
        ]

    monkeypatch.setattr(cli, "run_commands", unreadable)
    result = cli.validate(result, tmp_path / "owned2", time.monotonic())
    assert result["flagged_adversarial"] is True
    assert result["verdict_class"] == "blocked"


def test_checkpoint_and_main_error_branches(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7849-VALIDATION: retry cannot replace observed science."""
    cli = _cli_module()
    cli.science(tmp_path, "20260929", time.monotonic())
    cli.science(tmp_path, "20260929", time.monotonic())
    checkpoint = next((tmp_path / "checkpoints").glob("*/science.json"))
    checkpoint.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint_observation_changed"):
        cli.science(tmp_path, "20260929", time.monotonic())
    monkeypatch.setattr(sys, "argv", [str(CLI), "--date", "20260928"])
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr(sys, "argv", [str(CLI), "--check-only", str(tmp_path / "absent.json")])
    assert cli.main() == 1
    monkeypatch.setattr(cli, "OUTPUT", tmp_path / "final.json")
    monkeypatch.setattr(cli, "validate", lambda result, output_root, start: result)
    monkeypatch.setattr(cli.audit, "cold_replay", lambda path, root: ["mutation"])
    monkeypatch.setattr(sys, "argv", [str(CLI), "--output-root", str(tmp_path / "new")])
    assert cli.main() == 0
    assert json.loads((tmp_path / "final.json").read_text())["verdict_class"] == "disqualified"
