"""REQ-REPORT-7881-V684: prospective CPU intervention qualification."""

import json
from pathlib import Path

import pytest


def test_scenario_report_7881_manifest(tmp_path: Path) -> None:
    """A frozen command roster excludes the historical full-suite gate."""
    from carnot import experiment_7881_v684_intervention_protocol as exp

    plan = exp.command_manifest(tmp_path)
    names = [item["name"] for item in plan["commands"]]
    assert len(names) == len(set(names))
    assert "affected_pytest" in names
    assert "coverage_report" in names
    assert "reused_coverage_report" in names
    assert {"cli_replay_coverage", "cli_failure_coverage"} <= set(names)
    assert len(plan["coverage_files"]) == 4
    assert "e2e_016_fixture" in names
    assert "full_pytest" not in names
    assert all(item["classification"] == "required" for item in plan["commands"])
    assert all(item["timeout_s"] <= 300 for item in plan["commands"])
    assert (
        "tests/python/test_experiment_7868_v683_intervention_protocol.py" in plan["affected_tests"]
    )
    assert plan["repository_health_command"]["classification"] == "diagnostic"
    assert plan["historical_required_failures"][0]["name"] == "full_pytest"
    assert plan["prior_owned_attempt"]["path"].endswith("attempt1_candidate.json")


def test_scenario_report_7881_fixture_and_replay(tmp_path: Path) -> None:
    """Every arm is retained, and stale fixture bytes fail cold replay."""
    from carnot import experiment_7881_v684_intervention_protocol as exp

    path = tmp_path / "fixture.json"
    first = exp.write_fixtures(path, tmp_path / "checkpoints")
    assert len(first["cases"]) == 24
    assert first["checkpoint_hits"] == 0
    assert len(first["rows"]) == 96
    assert len({row["family_id"] for row in first["rows"]}) == 24
    assert {row["status"] for row in first["cases"]} >= {
        "completed",
        "invalid_parse",
        "invalid_witness",
        "excluded_empty_source",
        "excluded_no_complete_sentence",
        "excluded_no_matched_context",
    }
    assert all(row["semantic_sensitivity"] is None for row in first["rows"])
    assert exp.write_fixtures(path, tmp_path / "checkpoints")["checkpoint_hits"] == 24
    assert exp.cold_replay(path) == {"families": 24, "rows": 96}
    changed = json.loads(path.read_text())
    changed["rows"][0]["status"] = "forged"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fixture_row_drift"):
        exp.cold_replay(path)
    changed["cases"] = []
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fixture_count_drift"):
        exp.cold_replay(path)
    changed = first.copy()
    changed["cases"] = list(first["cases"])
    changed["cases"][0] = {**first["cases"][0], "status": "forged"}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fixture_row_drift"):
        exp.cold_replay(path)
    fixture, response, finish = exp.family(0)
    identity = exp.canonical_hash(
        {"row": fixture, "reply": response, "finish": finish, "seed": exp.SEED}
    )
    checkpoint = tmp_path / "checkpoints" / f"{identity[7:]}.json"
    saved = json.loads(checkpoint.read_text())
    saved["identity"] = "changed"
    checkpoint.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checkpoint_drift"):
        exp.write_fixtures(path, tmp_path / "checkpoints")


def test_scenario_report_7881_preflight_and_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent prerequisite blocks; an owned failure disqualifies."""
    from carnot import experiment_7881_v684_intervention_protocol as exp

    checks, hashes = exp.preflight()
    assert checks and all(check["passed"] for check in checks)
    assert hashes["prior_exp7868"]["role"] == "historical"
    plan = exp.command_manifest(tmp_path)
    checked, _ = exp.preflight(plan)
    assert all(check["passed"] for check in checked)
    assert any(check["artifact_field"] == "schema" for check in checked)
    changed_plan = {**plan, "source_hashes": dict(plan["source_hashes"])}
    first_path = next(iter(changed_plan["source_hashes"]))
    changed_plan["source_hashes"][first_path] = "sha256:wrong"
    drifted, _ = exp.preflight(changed_plan)
    assert any(not check["passed"] and check["artifact_field"] == "sha256" for check in drifted)
    monkeypatch.setattr(exp, "PRIOR", tmp_path / "missing.json")
    missing, _ = exp.preflight()
    assert any(not check["passed"] and check["artifact_field"] == "exists" for check in missing)
    assert exp.verdict(missing, [], False) == ("complete_blocked_required_source", "blocked", 0)
    assert exp.verdict(checks, [{"classification": "required", "passed": False}], False)[1:] == (
        "disqualified",
        0,
    )
    assert exp.verdict(checks, [{"classification": "required", "passed": True}], False)[1:] == (
        "circular_positive",
        1,
    )


def test_scenario_report_7881_fixture_cli(tmp_path: Path) -> None:
    """The new script owns fixture success and cold replay routes."""
    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/experiments/experiment_7881_v684_intervention_protocol.py"
    )
    path = tmp_path / "fixture.json"
    for option in ("--fixture-e2e", "--cold-replay"):
        result = subprocess.run(
            [sys.executable, "-u", str(script), "--date", "20260929", option, str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(path.read_text())["model_calls"] == 0


def test_scenario_report_7881_cross_device_publication(tmp_path: Path) -> None:
    """Candidate bytes survive publication through a sibling temporary file."""
    from carnot import experiment_7881_v684_intervention_protocol as exp

    candidate = tmp_path / "scratch" / "candidate.json"
    candidate.parent.mkdir()
    candidate.write_bytes(b'{"exact":true}\n')
    output = tmp_path / "results" / "artifact.json"
    exp.publish_exact(candidate, output)
    assert output.read_bytes() == b'{"exact":true}\n'
    assert not candidate.exists()


def test_scenario_report_7881_publication_copy_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed copy cannot replace the exact checked bytes."""
    from carnot import experiment_7881_v684_intervention_protocol as exp

    candidate = tmp_path / "candidate.json"
    candidate.write_text("candidate")
    output = tmp_path / "artifact.json"
    hashes = iter(["source", "different"])
    monkeypatch.setattr(exp, "sha256_file", lambda _path: next(hashes))
    with pytest.raises(ValueError, match="candidate_copy_drift"):
        exp.publish_exact(candidate, output)
    assert not output.exists()


def test_scenario_report_7881_repository_health_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded diagnostic from attempt one is authenticated on retry."""
    from carnot import experiment_7881_v684_intervention_protocol as exp

    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    plan = exp.command_manifest(tmp_path)
    assert exp.prior_repository_health(plan) is None
    log = tmp_path / "health.log"
    log.write_text("prior timeout")
    receipt = {
        "command_argv": plan["repository_health_command"]["argv"],
        "log_path": str(log),
        "log_sha256": exp.sha256_file(log),
        "passed": False,
    }
    candidate = exp.RAW / "attempts/attempt1_candidate.json"
    exp.atomic_json(candidate, {"repository_health": {"diagnostic_receipt": receipt}})
    assert exp.prior_repository_health(plan) == receipt
    receipt["command_argv"] = ["wrong"]
    exp.atomic_json(candidate, {"repository_health": {"diagnostic_receipt": receipt}})
    with pytest.raises(ValueError, match="diagnostic_command_drift"):
        exp.prior_repository_health(plan)
    receipt["command_argv"] = plan["repository_health_command"]["argv"]
    exp.atomic_json(candidate, {"repository_health": {"diagnostic_receipt": receipt}})
    log.write_text("changed")
    with pytest.raises(ValueError, match="diagnostic_log_drift"):
        exp.prior_repository_health(plan)


def test_scenario_report_7881_seals_and_child_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closed child log has one content address and argv drift fails."""
    import time

    from carnot import experiment_7881_v684_intervention_protocol as exp

    log = tmp_path / "child.log"
    log.write_text("done")
    receipt = {"name": "one", "log_path": str(log), "command_argv": ["one"], "passed": True}
    first = exp.seal(receipt, 0, tmp_path)
    assert exp.seal(receipt, 0, tmp_path) == first
    Path(first["log_path"]).write_text("changed")
    with pytest.raises(ValueError, match="sealed_log_drift"):
        exp.seal(receipt, 0, tmp_path)
    monkeypatch.setattr(exp, "seal", lambda row, _index, _scratch: row)
    command = {
        "name": "one",
        "argv": ["one", f"--basetemp={tmp_path / 'base' / 'x'}"],
        "classification": "required",
        "timeout_s": 2,
    }
    monkeypatch.setattr(
        exp,
        "run_commands",
        lambda *_args, **_kwargs: [{**receipt, "command_argv": command["argv"]}],
    )
    assert exp.execute({"commands": [command]}, tmp_path, time.monotonic())[0]["passed"]
    assert (tmp_path / "base").is_dir()
    monkeypatch.setattr(exp, "run_commands", lambda *_args, **_kwargs: [receipt])
    with pytest.raises(ValueError, match="child_command_drift"):
        exp.execute({"commands": [command]}, tmp_path, time.monotonic())


def test_scenario_report_7881_expected_cli_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named negative CLI check requires its exact exit and error text."""
    import time

    from carnot import experiment_7881_v684_intervention_protocol as exp

    log = tmp_path / "error.log"
    log.write_text("FileNotFoundError: absent fixture")
    command = {
        "name": "cli_failure_coverage",
        "argv": ["python", "absent"],
        "classification": "required",
        "timeout_s": 2,
        "expected_exit_code": 1,
        "expected_error_token": "FileNotFoundError",
    }
    monkeypatch.setattr(exp, "seal", lambda row, _index, _scratch: row)
    monkeypatch.setattr(
        exp,
        "run_commands",
        lambda *_args, **_kwargs: [
            {
                "name": command["name"],
                "command_argv": command["argv"],
                "log_path": str(log),
                "passed": False,
                "exit_code": 1,
                "timed_out": False,
                "output_tail": log.read_text(),
            }
        ],
    )
    assert exp.execute({"commands": [command]}, tmp_path, time.monotonic())[0]["passed"]
    command["expected_error_token"] = "WrongError"
    assert not exp.execute({"commands": [command]}, tmp_path, time.monotonic())[0]["passed"]


@pytest.mark.parametrize(
    "mode", ["success", "blocked", "missing", "flagged", "invalid_report", "strict_fail"]
)
def test_scenario_report_7881_owned_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The current driver handles terminal, blocked, and missing-child paths."""
    from carnot import experiment_7881_v684_intervention_protocol as exp

    scratch = tmp_path / "scratch"
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "result.json")
    if mode == "blocked":
        monkeypatch.setattr(exp, "PRIOR", tmp_path / "absent.json")
        assert exp.run("20260929", scratch)["verdict_class"] == "blocked"
        return
    plan = exp.command_manifest(scratch)
    receipts = [
        {
            "name": item["name"],
            "classification": "required",
            "passed": True,
            "command_argv": item["argv"],
            "log_path": str(tmp_path / "unused.log"),
        }
        for item in plan["commands"]
    ]
    monkeypatch.setattr(
        exp, "execute", lambda *_args: receipts[:-1] if mode == "missing" else receipts
    )
    if mode == "missing":
        with pytest.raises(ValueError, match="missing_required_command"):
            exp.run("20260929", scratch)
        return
    adverse = tmp_path / "adverse.log"
    adverse.write_text(
        "bad" if mode == "invalid_report" else json.dumps({"flagged_count": int(mode == "flagged")})
    )
    strict = tmp_path / "strict.log"
    strict.write_text("checked")
    health = tmp_path / "health.log"
    health.write_text("diagnostic")

    def children(_root: Path, specs: object, **_kwargs: object) -> list[dict]:
        names = [spec.name for spec in specs]
        if names == ["repository_full_pytest"]:
            return [{"name": names[0], "log_path": str(health), "passed": False}]
        if "terminal_retry_logs" in str(_kwargs.get("log_dir", "")):
            retry = tmp_path / "retry-adverse.json"
            retry.write_text(json.dumps({"flagged_count": 0}))
            return [
                {"name": names[0], "log_path": str(retry), "passed": True},
                {"name": names[1], "log_path": str(strict), "passed": True},
            ]
        return [
            {
                "name": names[0],
                "log_path": str(adverse),
                "passed": mode not in {"flagged", "invalid_report"},
            },
            {"name": names[1], "log_path": str(strict), "passed": mode != "strict_fail"},
        ]

    monkeypatch.setattr(exp, "run_commands", children)
    result = exp.run("20260929", scratch)
    assert result["verdict_class"] == (
        "disqualified"
        if mode in {"flagged", "invalid_report", "strict_fail"}
        else "circular_positive"
    )
    if mode == "strict_fail":
        assert result["flagged_adversarial"] is False
        assert result["intervention_protocol_ready_score"] == 0
    assert exp.OUTPUT.is_file()
    assert result["repository_health"]["status"] == "degraded_open"


def test_scenario_report_7881_date_and_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wrong dates and changed frozen commands fail before measurement."""
    from carnot import experiment_7881_v684_intervention_protocol as exp

    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.run("20260928", tmp_path)
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    exp.atomic_json(exp.RAW / "validation_command_manifest_attempt3.json", {"different": True})
    with pytest.raises(ValueError, match="validation_manifest_drift"):
        exp.run("20260929", tmp_path)
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.main(["--date", "20260928"])
    monkeypatch.setattr(exp, "run", lambda _date, _scratch: {"verdict_class": "disqualified"})
    assert exp.main(["--date", "20260929", "--scratch", str(tmp_path)]) == 1


@pytest.mark.parametrize("mode", ["pass", "invalid", "drift"])
def test_scenario_report_7893_revalidation_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The changed candidate needs a fresh passing check before exact publication."""
    from carnot import experiment_7881_v684_intervention_protocol as exp

    scratch = tmp_path / "scratch"
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    monkeypatch.setattr(exp, "OUTPUT", tmp_path / "result.json")
    plan = exp.command_manifest(scratch)
    receipts = [
        {
            "name": item["name"],
            "classification": "required",
            "passed": True,
            "command_argv": item["argv"],
            "log_path": str(tmp_path / "unused.log"),
        }
        for item in plan["commands"]
    ]
    monkeypatch.setattr(exp, "execute", lambda *_args: receipts)
    health = tmp_path / "health.log"
    health.write_text("historical diagnostic")
    seen: list[str] = []
    if mode == "drift":
        original_sha = exp.sha256_file
        candidate_hash_calls = 0

        def drifting_sha(path: Path) -> str:
            nonlocal candidate_hash_calls
            if path == scratch / "terminal_candidate.json":
                candidate_hash_calls += 1
                if candidate_hash_calls == 5:
                    return "sha256:changed_after_validation"
            return original_sha(path)

        monkeypatch.setattr(exp, "sha256_file", drifting_sha)

    def children(_root: Path, specs: object, **_kwargs: object) -> list[dict]:
        names = [item.name for item in specs]
        if names == ["repository_full_pytest"]:
            return [{"name": names[0], "log_path": str(health), "passed": False}]
        candidate_path = Path(specs[0].argv[-1])
        seen.append(exp.sha256_file(candidate_path))
        index = len(seen)
        adverse = tmp_path / f"adverse-{index}.json"
        adverse.write_text(
            "invalid"
            if index == 2 and mode == "invalid"
            else json.dumps({"flagged_count": int(index == 1)})
        )
        strict = tmp_path / f"strict-{index}.log"
        strict.write_text("checked")
        passed = index == 2 and mode != "invalid"
        return [
            {"name": names[0], "log_path": str(adverse), "passed": passed},
            {"name": names[1], "log_path": str(strict), "passed": passed},
        ]

    monkeypatch.setattr(exp, "run_commands", children)
    if mode != "pass":
        expected = (
            "terminal_revalidation_failed" if mode == "invalid" else "terminal_candidate_drift"
        )
        with pytest.raises(ValueError, match=expected):
            exp.run("20260929", scratch)
        assert not exp.OUTPUT.exists()
        return
    result = exp.run("20260929", scratch)
    assert len(seen) == 2 and seen[0] != seen[1]
    assert result["verdict_class"] == "disqualified"
    assert result["intervention_protocol_ready_score"] == 0
    assert exp.sha256_file(exp.OUTPUT) == seen[1]
    sidecar = json.loads((scratch / "terminal_validation_reports.json").read_text())
    assert [item["candidate_sha256"] for item in sidecar["chain"]] == seen
    assert sidecar["candidate_sha256"] == seen[1]
