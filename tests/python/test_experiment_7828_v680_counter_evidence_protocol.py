"""REQ-REPORT-7828 and SCENARIO-REPORT-7828-* custody tests."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import time

import pytest

from carnot import experiment_7828_v680_counter_evidence_protocol as task


def test_scenario_report_7828_shards_and_exact_dispatch(tmp_path: Path) -> None:
    """The real CLI names completed files and rejects undeclared children."""
    plan = task.load_plan()
    commands = plan["commands"]
    assert commands[4]["name"] == "coverage_combine"
    assert commands[4]["argv"][-4:] == [
        *plan["historical_coverage_files"],
        str(Path(plan["private_root"]) / "commands/02_coverage_protocol/.coverage.protocol"),
        str(Path(plan["private_root"]) / "commands/03_coverage_cli/.coverage.cli"),
    ]
    assert all(Path(path).is_file() for path in plan["historical_coverage_files"])
    observed = []

    def execute(command: dict, index: int, scope: dict) -> dict:
        observed.append((command["name"], command["argv"], command["classification"]))
        log = tmp_path / f"{index}.log"
        log.write_bytes(command["name"].encode())
        return dict(
            name=command["name"],
            command_argv=command["argv"],
            classification=command["classification"],
            log_path=str(log),
            log_sha256=task.sha256_file(log),
            exit_code=0,
            passed=True,
            timed_out=False,
        )

    assert task.main(["--date", "20260928", "--dispatch-check"], child_executor=execute) == 0
    assert observed == [(x["name"], x["argv"], x["classification"]) for x in commands]
    altered = json.loads(json.dumps(plan))
    altered["commands"].append({"name": "secret", "argv": ["true"], "classification": "required"})
    with pytest.raises(ValueError, match="manifest_drift"):
        task.dispatch(altered, execute)
    altered = json.loads(json.dumps(plan))
    altered["commands"][0]["argv"].append("--extra")
    with pytest.raises(ValueError, match="manifest_drift"):
        task.dispatch(altered, execute)
    log = tmp_path / "0.log"
    log.write_bytes(log.read_bytes() + b"x")
    with pytest.raises(ValueError, match="validation_log_drift"):
        task.validate_log_receipt(
            {"log_path": str(log), "log_sha256": task.digest(b"worktree_imports")}
        )


def test_scenario_report_7828_failed_or_missing_shards(tmp_path: Path) -> None:
    """A file is usable only when its completed child exited successfully."""
    plan = task.load_plan()
    checks = task.shard_checks(plan)
    assert len(checks) == 6
    assert all(item["passed"] for item in checks)
    changed = json.loads(json.dumps(plan))
    changed["historical_coverage_files"][0] = str(tmp_path / "missing")
    assert not task.shard_checks(changed)[0]["passed"]
    assert task.shard_checks(changed)[0]["observed"] is False
    copy = tmp_path / "mutated.coverage"
    copy.write_bytes(Path(plan["historical_coverage_files"][0]).read_bytes() + b"x")
    changed["historical_coverage_files"][0] = str(copy)
    assert not task.shard_checks(changed)[1]["passed"]


def test_scenario_report_7828_failed_historical_shard_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed coverage bytes cannot erase a nonzero source child exit."""
    plan = task.load_plan()
    old = json.loads(
        (task.ROOT / "results/experiment_7814_v679_counter_evidence_protocol.json").read_text()
    )
    next(x for x in old["validation_receipts"] if x["name"] == "coverage_protocol")["exit_code"] = 1
    path = tmp_path / "results/experiment_7814_v679_counter_evidence_protocol.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(old))
    monkeypatch.setattr(task, "ROOT", tmp_path)
    assert not task.shard_checks(plan)[2]["passed"]


def test_scenario_report_7828_event_and_bytes() -> None:
    """The selected event and source bytes survive a UTF-8 deletion."""
    source = "Café one here. Other two now. Third nice here."
    answer = "Café is here. A later sentence is wrong."
    row = dict(
        family_id="fixture",
        complete_source=source,
        complete_response=answer,
        source_sha256=task.digest(source.encode()),
        response_sha256=task.digest(answer.encode()),
    )
    offsets = task.prior.sentence_offsets(source.encode())
    chosen = task.select_control(source.encode(), offsets, 0, row["family_id"], lambda _: 4)
    assert chosen in {1, 2}
    edited, visible = task.prior.visible_after(source.encode(), offsets, 0)
    assert edited.decode() == "Other two now. Third nice here."
    assert [x["source_sentence_id"] for x in visible] == [1, 2]
    assert task.prior.target_span(answer.encode())["end_byte"] == len("Café is here. ".encode())
    assert task.prior.aligned_label(row, {"label": 1, "annotations": [{"start": 0, "end": 5}]}) == 1
    assert (
        task.prior.aligned_label(row, {"label": 1, "annotations": [{"start": 20, "end": 30}]})
        is None
    )
    assert task.prior.aligned_label(row, {"label": 1, "annotations": []}) is None


def test_scenario_report_7828_cli_file_is_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the actual small CLI file so its dispatch path enters coverage."""
    monkeypatch.setattr(task, "main", lambda: 0)
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(
            str(task.ROOT / "scripts/experiments" / f"{task.NAME}.py"), run_name="__main__"
        )


@pytest.mark.parametrize("flagged", [False, True])
def test_scenario_report_7828_current_run_and_cold_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flagged: bool
) -> None:
    """A real 48-family preparation reaches the owned terminal result."""
    plan = task.load_plan()
    plan["raw_root"] = str(tmp_path / "attempt")
    plan["candidate_path"] = str(tmp_path / "attempt/candidate.json")
    plan["private_root"] = str(tmp_path / "private")
    for index, command in enumerate(plan["commands"]):
        command["private_root"] = str(tmp_path / "private" / str(index))
    monkeypatch.setattr(task, "load_plan", lambda: plan)
    monkeypatch.setattr(task, "OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(task.prior, "resource_checks", lambda: [])

    class Counter:
        template_sha256 = "sha256:fixture"

        def __init__(self, path: Path, expected_hash: str) -> None:
            assert path == task.prior.GGUF_PATH and expected_hash == task.prior.GGUF_SHA256

        def __call__(self, text: str) -> int:
            return len(text.split())

        def count_messages(self, messages: list[dict]) -> int:
            return 20

    monkeypatch.setattr(task.prior, "GGUFTokenCounter", Counter)
    original_request = task.prior.make_request
    rejected = set()

    def budget_one(row: dict, *args: object) -> dict:
        if row["family_id"] != "fixture-e2e" and not rejected:
            rejected.add(row["family_id"])
            raise ValueError("context_budget")
        return original_request(row, *args)

    monkeypatch.setattr(task.prior, "make_request", budget_one)

    def execute(command: dict, index: int, scope: dict) -> dict:
        log = tmp_path / f"{index}.log"
        log.write_text(
            ("bad" if flagged else '{"flagged_count": 0}')
            if command["name"] == "adversarial_verify"
            else command["name"]
        )
        return dict(
            name=command["name"],
            command_argv=command["argv"],
            classification=command["classification"],
            log_path=str(log),
            log_sha256=task.sha256_file(log),
            exit_code=0,
            passed=True,
            timed_out=False,
        )

    result = task.run_experiment("20260928", execute)
    assert result["verdict_class"] == ("disqualified" if flagged else "circular_positive")
    assert result["counter_evidence_ready_score"] == (0 if flagged else 1)
    assert result["random_seed"] == 68001
    assert len(result["rows"]) == 144
    assert result["aligned_label_count"] <= 48
    assert any(row["disposition"] == "unstarted_context_budget" for row in result["rows"])
    assert task.cold_reduce(tmp_path / "result.json")["families"] == 48
    assert task.main(["--date", "20260928", "--cold-replay", str(tmp_path / "result.json")]) == 0
    assert (
        task.main(["--date", "20260928", "--fixture-e2e", str(tmp_path / "other-fixture.json")])
        == 0
    )
    with pytest.raises(ValueError, match="attempt_root_reused"):
        task.run_experiment("20260928", execute)
    log = tmp_path / "0.log"
    log.write_bytes(log.read_bytes() + b"x")
    with pytest.raises(ValueError, match="validation_log_drift"):
        task.cold_reduce(tmp_path / "result.json")
    changed = json.loads((tmp_path / "result.json").read_text())
    changed["experiment_id"] = "wrong-owner"
    (tmp_path / "wrong.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="wrong_result_owner"):
        task.cold_reduce(tmp_path / "wrong.json")


def test_scenario_report_7828_blocked_and_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent external bytes produce a terminal blocked result before a child."""
    plan = task.load_plan()
    plan["raw_root"] = str(tmp_path / "blocked")
    monkeypatch.setattr(task, "load_plan", lambda: plan)
    monkeypatch.setattr(task, "OUTPUT", tmp_path / "blocked.json")
    monkeypatch.setattr(
        task.prior,
        "preflight",
        lambda root: (
            [],
            [
                task.prior.prior.check(
                    "missing_science", tmp_path / "missing", "exists", True, False
                )
            ],
            {},
        ),
    )
    monkeypatch.setattr(task.prior, "resource_checks", lambda: [])
    result = task.run_experiment("20260928")
    assert result["verdict_class"] == "blocked"
    assert result["gate_check_summary"]
    assert result["counter_evidence_ready_score"] == 0
    with pytest.raises(ValueError, match="run_date_mismatch"):
        task.main(["--date", "wrong"])
    with pytest.raises(ValueError, match="run_date_mismatch"):
        task.run_experiment("wrong")
    with pytest.raises(ValueError, match="evaluation64_invalid"):
        task.freeze_families([])
    assert task.select_control(b"One.", [], 4, "f", lambda _: 1) is None
    monkeypatch.setattr(
        task, "run_experiment", lambda date, executor=None: {"verdict_class": "disqualified"}
    )
    assert task.main(["--date", "20260928"]) == 1


def test_scenario_report_7828_command_and_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The current command reader rejects a changed byte or observed argv."""
    plan = task.load_plan()
    monkeypatch.setattr(task, "COMMAND_MANIFEST_SHA256", "sha256:wrong")
    with pytest.raises(ValueError, match="manifest_drift"):
        task.load_plan()
    monkeypatch.setattr(task, "load_plan", lambda: plan)

    def wrong(command: dict, index: int, scope: dict) -> dict:
        log = tmp_path / "wrong.log"
        log.write_text("wrong")
        return dict(
            name="wrong",
            command_argv=command["argv"],
            classification=command["classification"],
            log_path=str(log),
            log_sha256=task.sha256_file(log),
        )

    with pytest.raises(ValueError, match="observed_child_command_drift"):
        task.dispatch(plan, wrong)
    runpy.run_path(str(task.ROOT / "scripts/experiments" / f"{task.NAME}.py"), run_name="module")


def test_scenario_report_7828_failed_required_receipt_is_not_relabelled(tmp_path: Path) -> None:
    """A zero exit with an invalid import receipt remains a failed gate."""
    log = tmp_path / "import.log"
    log.write_text("unstructured import output")
    receipt = {
        "name": "worktree_imports",
        "classification": "required",
        "passed": False,
        "exit_code": 0,
        "log_path": str(log),
        "command_argv": ["python", "-c", "import x"],
    }
    result = task.build_result([], {}, [], None, None, [], [], [receipt], [], time.monotonic())
    assert result["verdict_class"] == "disqualified"
    assert result["counter_evidence_ready_score"] == 0
    assert result["gate_check_summary"][-1]["field"] == "worktree_imports.passed"
    assert result["gate_check_summary"][-1]["observed"] is False
