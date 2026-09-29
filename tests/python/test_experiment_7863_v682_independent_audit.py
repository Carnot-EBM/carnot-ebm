"""REQ-REPORT-7863: cold V682 evidence uses exact current bytes."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v682_independent_audit as audit


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_7863_v682_independent_audit.py"


def load_cli():
    """Load the real CLI so coverage includes its branch decisions."""
    spec = importlib.util.spec_from_file_location("exp7863_cli", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_and_terminal_block():
    """SCENARIO-REPORT-7863-BLOCKED: preserve every science branch."""
    rows, failures, sources = audit.inspect_sources(ROOT)
    assert [row["upstream_id"] for row in rows] == [f"Exp{n}" for n in range(7852, 7863)]
    assert any(row["status"] == "missing" for row in rows)
    assert any(row["status"] == "disqualified" for row in rows)
    assert any(row["status"] == "eligible_null" for row in rows)
    assert all(
        set(row) >= {"upstream_id", "path", "hash", "artifact_field", "op", "expected", "observed"}
        for row in failures
    )
    assert len(sources) >= 11


def test_mutation_guards():
    """SCENARIO-REPORT-7863-MUTATIONS: detect fabricated primitive evidence."""
    clean = {
        "family_id": "f1",
        "source_family": "s1",
        "arm": "treatment",
        "seed": 1,
        "status": "completed",
        "features": {"length": 2},
        "prediction_step": 1,
        "feedback_step": 2,
        "role": "evaluation",
    }
    assert audit.check_rows([clean], 1) == []
    mutations = [
        ([{**clean, "features": {"fixture_id": "f1"}}], 1, "fixture_id_leakage"),
        ([{**clean, "source_erased": True, "features": {"source_text": "x"}}], 1, "source_leakage"),
        ([{**clean, "feedback_step": 1}], 1, "future_feedback"),
        (
            [{**clean, "status": "dropped", "feedback_replayed": True}],
            1,
            "dropped_feedback_replayed",
        ),
        ([clean], 2, "lost_rows"),
        ([clean, {**clean, "seed": 2}], 2, "seed_pseudoreplication"),
        ([{**clean, "threshold_fit_role": "evaluation"}], 1, "evaluation_tuned_threshold"),
        ([{**clean, "control_value": 1, "treatment_value": 1}], 1, "identical_controls"),
    ]
    for rows, intended, expected in mutations:
        assert expected in audit.check_rows(rows, intended)
    assert "zero_rows" in audit.check_rows([], 1)


def test_candidate_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-7863-BLOCKED: audit readiness differs from benefit."""
    result = audit.build_candidate(ROOT, tmp_path, "20260929")
    assert result["experiment_id"] == 7863
    assert result["honest_verdict"].startswith("complete_blocked")
    assert result["milestone_evidence_complete_score"] == 0
    assert len(result["producer_status_rows"]) == 11
    assert any(row.get("family_id") for row in result["rows"])
    assert all("metric" in row for row in result["rows"])
    assert result["eligible_producer_count"] >= 2
    assert result["MODEL_SPECS"] == []
    assert result["model_invocation_counts"]["loads"] == 0
    assert result["resolved_imports"]
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(result))
    assert audit.cold_replay(path, ROOT) == []
    changed = copy.deepcopy(result)
    changed["rows"].pop()
    path.write_text(json.dumps(changed))
    assert "rows_changed" in audit.cold_replay(path, ROOT)
    changed = copy.deepcopy(result)
    changed["honest_verdict"] = "partial_blocked"
    path.write_text(json.dumps(changed))
    assert "partial_terminal_verdict" in audit.cold_replay(path, ROOT)
    changed = copy.deepcopy(result)
    changed["source_artifact_hashes"][0]["sha256"] = "sha256:forged"
    path.write_text(json.dumps(changed))
    assert "source_bytes_changed" in audit.cold_replay(path, ROOT)


def test_cli_private_routes(tmp_path):
    """SCENARIO-REPORT-7863-MUTATIONS: real CLI covers errors and replay."""
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
    path = tmp_path / "candidate.json"
    assert path.is_file()
    replay = subprocess.run(
        [sys.executable, str(CLI), "--check-only", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    path.write_text("{")
    bad = subprocess.run(
        [sys.executable, str(CLI), "--check-only", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
    date = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--date",
            "20260928",
            "--science-only",
            "--output-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert date.returncode != 0


def test_reader_negative_routes(tmp_path):
    """SCENARIO-REPORT-7863-MUTATIONS: each source defect has an operand."""
    authority = tmp_path / audit.AUTHORITY
    authority.parent.mkdir(parents=True)
    authority.write_text(
        json.dumps(
            {
                "tasks": [{"id": "admin", "deliverable": "unused"}]
                + [
                    {"id": f"exp{n}-science", "deliverable": f"results/experiment_{n}_science.json"}
                    for n in range(7852, 7863)
                ]
            }
        )
    )
    (tmp_path / "results").mkdir()
    bad = tmp_path / "results/experiment_7852_science.json"
    bad.write_text("{")
    rows, failures, sources = audit.inspect_sources(tmp_path)
    assert rows[0]["status"] == "invalid_json"
    assert failures[0]["artifact_field"] == "science_producer"
    assert audit.precondition_record(sources[0])["schema_version"] is None
    assert audit.reduce_primitives(tmp_path, [sources[0]])[0][0]["row_count"] == 0
    assert audit.primitive_unit_rows(tmp_path, [sources[0]])[0]["status"] == "invalid_json"
    authority.write_text(json.dumps({"tasks": [{"id": "admin"}, {"id": "wrong"}]}))
    try:
        audit.inspect_sources(tmp_path)
    except ValueError as exc:
        assert "authority_roster_changed" in str(exc)
    else:
        raise AssertionError("changed roster accepted")
    required = audit._required_failures(
        7852,
        bad,
        {
            "validation_receipts": [
                None,
                {"classification": "diagnostic", "exit_code": 1},
                {"classification": "required", "name": "x", "exit_code": 1},
            ]
        },
    )
    assert len(required) == 1 and required[0]["artifact_field"] == "validation_receipts.x"


def test_primitive_reduction_and_tamper(tmp_path):
    """REQ-REPORT-7863: losses use labels and bytes, not a producer mean."""
    path = tmp_path / "producer.json"
    path.write_text(
        json.dumps(
            {
                "sample_size_budget": {"intended": 4},
                "rows": [
                    {"family_id": "f1", "status": "completed", "probability": 0.8, "label": 1},
                    {
                        "family_id": "f2",
                        "status": "completed",
                        "source_erased": True,
                        "features": {"source_text": "leak"},
                    },
                    None,
                ],
            }
        )
    )
    source = {
        "upstream_id": "Exp7852",
        "path": str(path),
        "sha256": audit.sha256_file(path),
        "eligibility": False,
    }
    reduced, discrepancies = audit.reduce_primitives(tmp_path, [source])
    assert reduced[0]["row_count"] == 3
    assert abs(reduced[0]["brier"] - 0.04) < 1e-12
    assert {d["artifact_field"] for d in discrepancies} == {
        "rows",
        "features",
        "sample_size_budget.intended",
    }
    units = audit.primitive_unit_rows(tmp_path, [source])
    assert len(units) == 3 and units[-1]["status"] == "malformed"
    missing = {
        **source,
        "path": str(tmp_path / "absent.json"),
        "task_id": "exp7852-science",
        "status": "missing",
    }
    assert audit.primitive_unit_rows(tmp_path, [missing])[0]["status"] == "missing"
    result = audit.build_candidate(ROOT, tmp_path, "20260929")
    candidate = tmp_path / "candidate.json"
    result["experiment_id"] = "exp7863-independent-audit"
    result["gate_check_summary"] = []
    result["independently_reduced_rows"] = []
    log = tmp_path / "log"
    log.write_text("closed")
    result["validation_receipts"] = [{"log_path": str(log), "log_sha256": "sha256:forged"}]
    candidate.write_text(json.dumps(result))
    errors = audit.cold_replay(candidate, ROOT)
    assert {
        "experiment_id",
        "gate_operands_changed",
        "primitive_reduction_changed",
        "validation_log_changed",
    } <= set(errors)
    candidate.write_text("{")
    assert audit.cold_replay(candidate, ROOT) == ["candidate_unreadable"]


def test_cli_validation_receipts_and_checkpoint(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7863-BLOCKED: only passing required children yield audit readiness."""
    cli = load_cli()
    first = cli.science(tmp_path, "20260929", 0.0)
    assert (
        cli.science(tmp_path, "20260929", 0.0)["reproducibility_checksum"]
        == first["reproducibility_checksum"]
    )
    checkpoint = next((tmp_path / "checkpoints").rglob("science.json"))
    checkpoint.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint_observation_changed"):
        cli.science(tmp_path, "20260929", 0.0)
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(audit, "MANIFEST", str(manifest))
    calls = []

    def fake_run(_root, specs, *, log_dir, heartbeat_s):
        assert heartbeat_s <= 60
        log_dir.mkdir(parents=True, exist_ok=True)
        name = specs[0].name
        calls.append(name)
        log = log_dir / f"{name}.log"
        log.write_text("{}" if name == "adversarial_verify" else "closed")
        return [
            {
                "name": name,
                "log_path": str(log),
                "exit_code": int(name == "fail"),
                "passed": name != "fail",
                "output_tail": '{"flagged_count": 0}',
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run)
    manifest.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "name": "worktree_imports",
                        "argv": [sys.executable],
                        "classification": "required",
                        "deadline_s": 1,
                    },
                    {
                        "name": "adversarial_verify",
                        "argv": [sys.executable],
                        "classification": "required",
                        "deadline_s": 1,
                    },
                    {
                        "name": "repository_health_180s",
                        "argv": [sys.executable],
                        "classification": "diagnostic",
                        "deadline_s": 1,
                    },
                ]
            }
        )
    )
    good = cli.validate(audit.build_candidate(ROOT, tmp_path, "20260929"), tmp_path, 0.0)
    assert good["audit_execution_ready_score"] == 1
    assert good["repository_health"]["status"] == "passed"
    assert calls == ["worktree_imports", "adversarial_verify", "repository_health_180s"]
    assert all(Path(r["log_path"]).is_file() for r in good["validation_receipts"])
    manifest.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "name": "fail",
                        "argv": [sys.executable],
                        "classification": "required",
                        "deadline_s": 1,
                    }
                ]
            }
        )
    )
    bad = cli.validate(audit.build_candidate(ROOT, tmp_path, "20260929"), tmp_path, 0.0)
    assert bad["verdict_class"] == "disqualified"
    assert bad["required_validation_failures"] == ["fail"]
    manifest.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "name": "adversarial_verify",
                        "argv": [sys.executable],
                        "classification": "required",
                        "deadline_s": 1,
                    }
                ]
            }
        )
    )
    old_runner = cli.run_commands

    def malformed_report(*args, **kwargs):
        receipts = old_runner(*args, **kwargs)
        receipts[0]["output_tail"] = "not json"
        return receipts

    monkeypatch.setattr(cli, "run_commands", malformed_report)
    flagged = cli.validate(audit.build_candidate(ROOT, tmp_path, "20260929"), tmp_path, 0.0)
    assert flagged["flagged_adversarial"] is True
    assert flagged["required_validation_failures"] == ["adversarial_verify"]


def test_cli_main_branches(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7863-MUTATIONS: main preserves a failed cold replay."""
    cli = load_cli()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(CLI), "--date", "20260929", "--science-only", "--output-root", str(tmp_path)],
    )
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", [str(CLI), "--check-only", str(tmp_path / "candidate.json")])
    assert cli.main() == 0
    (tmp_path / "candidate.json").write_text("{")
    assert cli.main() == 1
    monkeypatch.setattr(sys, "argv", [str(CLI), "--date", "20260928"])
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr(
        sys, "argv", [str(CLI), "--date", "20260929", "--output-root", str(tmp_path)]
    )
    monkeypatch.setattr(
        cli, "science", lambda *_: audit.build_candidate(ROOT, tmp_path, "20260929")
    )
    monkeypatch.setattr(cli, "validate", lambda result, *_: result)
    monkeypatch.setattr(audit, "cold_replay", lambda *_: ["forged"])
    output = tmp_path / "terminal.json"
    monkeypatch.setattr(cli, "OUTPUT", output)
    assert cli.main() == 0
    assert json.loads(output.read_text())["honest_verdict"] == "complete_disqualified_cold_replay"
