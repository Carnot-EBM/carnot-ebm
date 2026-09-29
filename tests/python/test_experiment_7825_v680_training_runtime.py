"""Current fixture runtime qualification for REQ-VERIFY-7825."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7825_v680_training_runtime as exp
from carnot.reporting.current_work_receipt import sha256_file
from scripts.experiments import experiment_7825_v680_training_runtime as cli


def test_historical_format_receipt_and_frozen_scope() -> None:
    """SCENARIO-VERIFY-7825-DISPATCH: retain the real failed format operand."""
    prior = json.loads(
        (exp.ROOT / "results/experiment_7811_v679_training_runtime.json").read_text()
    )
    assert prior["verdict_class"] == "disqualified"
    failure = next(
        row for row in prior["gate_check_summary"] if row["field"] == "ruff_format.exit_code"
    )
    assert failure["observed"] == 1
    assert sha256_file(Path(failure["artifact_path"])) == failure["artifact_hash"]
    assert (
        b"experiment_7811_v679_training_runtime.py" in Path(failure["artifact_path"]).read_bytes()
    )
    scope = json.loads(exp.SCOPE.read_text())
    assert scope["requirement"] == "REQ-VERIFY-7825"
    assert scope["frozen_before_implementation"] is True
    assert [row["name"] for row in scope["commands"]] == exp.COMMAND_NAMES
    assert {row["name"] for row in scope["commands"] if row["classification"] == "diagnostic"} == {
        "repository_health"
    }
    assert "tests/python/test_experiment_7811_v679_training_runtime.py" in scope["transitive_tests"]


def test_missing_external_input_blocks_before_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7825-FIXTURE: external absence is terminal blocked."""
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(exp, "fit_one", lambda *args: pytest.fail("fit started"))
    row = exp.measure("20260928", tmp_path / "raw")
    assert row["experiment_id"] == 7825
    assert row["milestone"] == "2026.09.680"
    assert row["honest_verdict"].startswith("complete_blocked_")
    assert row["verdict_class"] == "blocked"
    assert row["gate_check_summary"]
    assert row["sample_size_budget"]["started"] == 0
    assert row["training_runtime_ready_score"] == row["online_runtime_ready_score"] == 0
    assert len(row["rows"]) == 162


def test_current_fixtures_and_bank_replay(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7825-FIXTURE: all new seeds train and bank controls pass."""
    row = exp.measure("20260928", tmp_path / "raw")
    assert row["acceptance_gate_results"]["validity"] is True
    assert len(row["fixture_training_rows"]) == 27
    assert len(row["rows"]) == 162
    assert {r["seed"] for r in row["fixture_training_rows"]} == {68001, 68002, 68003}
    assert {r["arm"] for r in row["fixture_training_rows"]} == set(exp.ARMS)
    assert all(
        r["gradient_norm"] > 0 and r["initial_hash"] != r["final_hash"]
        for r in row["fixture_training_rows"]
    )
    assert all(
        r["reload_decision_equal"] and r["normalization_error"] < 1e-8
        for r in row["fixture_training_rows"]
    )
    assert all(
        all(abs(value) < float("inf") for value in r["dual_variables"])
        for r in row["fixture_training_rows"]
    )
    assert row["online_fixture"]["valid"] is True
    protocol = json.loads(Path(row["training_protocol_path"]).read_text())
    assert protocol["natural_recipe"]["epochs"] == 16
    assert protocol["seeds"] == [68001, 68002, 68003]
    assert protocol["fixture_only"] is True
    assert "initial_weights" not in protocol["natural_recipe"]
    online = json.loads(Path(row["online_protocol_path"]).read_text())
    assert len(online["predicates"]) == 16
    assert online["proposal_budget_per_block"] == 1
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(row))
    assert exp.cold_reduce(candidate)["row_count"] == 162
    resumed = exp.measure("20260928", tmp_path / "raw")
    assert len(resumed["fixture_training_rows"]) == 27
    assert [r["head_hash"] for r in resumed["fixture_training_rows"]] == [
        r["head_hash"] for r in row["fixture_training_rows"]
    ]


def test_required_format_failure_zeroes_both_scores(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7825-DISPATCH: required format failure is disqualifying."""
    scope = {"direct_tests": [], "transitive_tests": []}
    manifest = {
        "commands": [
            {"name": "ruff_format", "argv": ["ruff", "format"], "classification": "required"}
        ]
    }
    log = tmp_path / "failure.log"
    log.write_text("Would reformat")
    receipt = {
        "name": "ruff_format",
        "argv": ["ruff", "format"],
        "classification": "required",
        "exit_code": 1,
        "timed_out": False,
        **exp.seal_log(log, tmp_path / "durable", "ruff_format"),
    }
    gate = exp.reduce_validation(scope, manifest, [receipt])
    assert gate["training_runtime_ready_score"] == gate["online_runtime_ready_score"] == 0
    assert any(r["field"] == "ruff_format.exit_code" for r in gate["gate_check_summary"])
    Path(receipt["log_path"]).write_text("mutated")
    assert any(
        r["field"] == "ruff_format.log_sha256"
        for r in exp.reduce_validation(scope, manifest, [receipt])["gate_check_summary"]
    )


def test_real_cli_identity_and_retry_seals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7825-DISPATCH: real dispatcher owns exact child identities."""
    scope = json.loads(exp.SCOPE.read_text())
    miniature = {
        **scope,
        "commands": [scope["commands"][index] for index in (4, 6, 10, 11, 13)],
    }
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(miniature))
    monkeypatch.setattr(exp, "SCOPE", path)
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "output.json")
    base = exp.base_record("20260928", [], [], 0.01)
    base["acceptance_gate_results"]["validity"] = True
    monkeypatch.setattr(exp, "measure", lambda *args: json.loads(json.dumps(base)))
    seen = []

    def owned(command, durable, private):
        seen.append((command["name"], command["argv"], command["classification"]))
        log = private / command["name"] / "child.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("ok")
        return {
            "name": command["name"],
            "argv": command["argv"],
            "classification": command["classification"],
            "exit_code": 1 if command["name"] == "ruff_format" else 0,
            "timed_out": False,
            **exp.seal_log(log, durable, command["name"]),
        }

    monkeypatch.setattr(cli, "execute", owned)
    first = cli.run("20260928", tmp_path / "attempt-one")
    assert seen == [
        (r["name"], r["argv"], r["classification"]) for r in first["observed_child_commands"]
    ]
    assert first["training_runtime_ready_score"] == first["online_runtime_ready_score"] == 0
    first_log = Path(first["validation_receipts"]["commands"][0]["log_path"])
    before = first_log.read_bytes()
    cli.run("20260928", tmp_path / "attempt-two")
    assert first_log.read_bytes() == before


def test_invalid_current_fixture_retains_failed_operand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7825-FIXTURE: an invalid current fit cannot qualify."""
    monkeypatch.setattr(
        exp,
        "fit_one",
        lambda arm, seed, records, names, folder: {
            "arm": arm,
            "seed": seed,
            "head_path": str(folder / "missing.json"),
            "head_hash": "sha256:missing",
            "rows": [],
        },
    )
    monkeypatch.setattr(
        exp.prior.prior,
        "exercise_online",
        lambda *args: {"valid": False, "query_rows": []},
    )
    result = exp.measure("20260928", tmp_path / "invalid")
    assert result["acceptance_gate_results"]["validity"] is False
    assert result["gate_check_summary"][0]["field"] == "fixture_valid"
    assert result["training_runtime_ready_score"] == 0


def test_current_cli_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7825-DISPATCH: all CLI modes use current evidence."""
    monkeypatch.setattr(exp, "cold_reduce", lambda path: {"valid": True, "row_count": 6})
    assert cli.main(["--cold-reduce", str(tmp_path / "candidate.json")]) == 0
    record = exp.base_record("20260928", [], [], 0.01)
    record["online_fixture"] = {"valid": True}
    record["fixture_training_rows"] = [{"reload_decision_equal": True}]
    monkeypatch.setattr(exp, "measure", lambda *args: record)
    assert cli.main(["--mini-e2e", "--private-root", str(tmp_path / "e2e")]) == 0
    record["fixture_training_rows"][0]["reload_decision_equal"] = False
    assert cli.main(["--mini-e2e", "--private-root", str(tmp_path / "bad")]) == 1
    record["verdict_class"] = "blocked"
    assert cli.main(["--mini-e2e", "--private-root", str(tmp_path / "blocked")]) == 1
    with pytest.raises(SystemExit):
        cli.main(["--mini-e2e"])
    monkeypatch.setattr(
        cli,
        "run",
        lambda *args: {
            "experiment_id": 7825,
            "honest_verdict": "complete_blocked_external_precondition",
        },
    )
    assert cli.main(["--date", "20260928"]) == 0
    script = exp.ROOT / "scripts/experiments/experiment_7825_v680_training_runtime.py"
    monkeypatch.setattr(
        sys, "argv", [str(script), "--cold-reduce", str(tmp_path / "candidate.json")]
    )
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(script), run_name="__main__")
    assert done.value.code == 0
