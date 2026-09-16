"""Qualify the repaired ARC result-resume launcher with CPU evidence.

Spec refs: REQ-ARC-WMTE-7345 and SCENARIO-ARC-WMTE-7345-*.
"""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import runpy
import time

import pytest

from carnot import experiment_7345_v645_arc_resume_check as exp


REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _receipts(names: tuple[str, ...], *, passed: bool = True) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "command": f"fixture {name}",
            "command_argv": ["fixture", name],
            "scope": "fixture",
            "exit_code": 0 if passed else 1,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": f"sha256:{index:064x}",
            "passed": passed,
            "timed_out": False,
        }
        for index, name in enumerate(names, start=1)
    ]


def test_req_7345_spec_identity_and_preconditions(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7345 treats Exp7336 as diagnostic evidence, not a producer."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-ARC-WMTE-7345:" in spec
    assert "SCENARIO-ARC-WMTE-7345-PRIVATE-BASETEMP" in spec
    assert exp.RUN_DATE == "20260916"
    assert exp.MILESTONE == "2026.09.645"
    assert exp.MODEL_SPECS == []

    receipt = exp.check_preconditions(REPO)
    assert receipt["passed"] is True
    assert receipt["first_loss_receipt"]["reproduced"] is True
    assert receipt["historical_exp7336"]["status"] == "disqualified"
    assert receipt["historical_exp7336"]["producer_eligible"] is False
    assert receipt["historical_exp7336"]["diagnostic_only"] is True
    assert all(row["passed"] for row in receipt["checks"])

    missing = exp.check_preconditions(REPO / "does-not-exist")
    assert missing["passed"] is False
    assert missing["checks"]
    assert missing["first_failure"]["observed"] is False

    malformed = tmp_path / exp.HISTORICAL_RESULT_PATH
    malformed.parent.mkdir(parents=True)
    malformed.write_text("[]", encoding="utf-8")
    assert exp.check_preconditions(tmp_path)["historical_exp7336"]["status"] is None


def test_scenario_7345_private_basetemp_reproduction_and_concurrency(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7345-PRIVATE-BASETEMP proves the launcher repair."""

    receipt = exp.qualify_private_basetemps(tmp_path / "scratch")
    assert receipt == {
        "missing_parent_failure_reproduced": True,
        "parents_created_before_launch": True,
        "concurrent_roots_distinct": True,
        "first_run_cleaned": True,
        "second_run_survived_cleanup": True,
    }


def test_scenario_7345_actual_policy_and_required_controls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7345-LIVE-RESUME drives E3AgentPolicy and HTTP payloads."""

    receipt = exp.run_resume_qualification(tmp_path / "panel")
    rows = {row["arm"]: row for row in receipt["rows"]}
    assert set(rows) == set(exp.REQUIRED_ARMS)

    live = rows["tool_needed_changed_input"]
    assert live["policy_class"] == "E3AgentPolicy"
    assert live["completion_calls"] == live["completion_limit"] == 2
    assert live["generated_tokens"] <= live["generated_token_limit"] == 4096
    assert live["result_delivery_count"] == 1
    assert live["later_request_result_occurrences"] == 1
    assert live["receipt_captured"] is True
    assert live["verified_engine_installed"] is True
    assert live["plan_installed"] is True
    assert live["later_policy_action"] is True

    withheld = rows["result_withheld"]
    assert withheld["result_delivery_count"] == 0
    assert withheld["plan_installed"] is False
    for name in (
        "stale_episode",
        "duplicate_request",
        "malformed_result",
        "timeout_after_dispatch",
        "exhausted_budget",
    ):
        row = rows[name]
        assert row["passed"] is True
        assert row["plan_authorized"] is False
        assert row["stale_or_duplicate_consumption"] == 0
        assert row["completion_budget_overrun"] == 0
        assert row["token_budget_overrun"] == 0

    sidecar = Path(receipt["fixture_sidecar_path"])
    assert sidecar.is_file()
    assert receipt["fixture_sidecar_sha256"] == exp.sha256_file(sidecar)
    assert (
        json.loads(sidecar.read_text(encoding="utf-8"))["counts_as_current_model_invocation"]
        is False
    )

    raw = tmp_path / "raw.json"
    exp.atomic_write(raw, {"rows": receipt["rows"]})
    reduced = exp.independent_reduce(raw)
    assert reduced["arc_resume_ready_score"] == 1
    changed = json.loads(raw.read_text(encoding="utf-8"))
    changed["rows"][0]["completion_calls"] = 3
    exp.atomic_write(raw, changed)
    assert exp.independent_reduce(raw)["arc_resume_ready_score"] == 0
    raw.write_text("{", encoding="utf-8")
    assert exp.independent_reduce(raw)["row_count"] == 0


def test_scenario_7345_terminal_artifact_and_fail_closed_reduction(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7345-TERMINAL binds current checks and repair hashes."""

    preconditions = exp.check_preconditions(REPO)
    private_paths = exp.qualify_private_basetemps(tmp_path / "private-paths")
    panel = exp.run_resume_qualification(tmp_path / "panel")
    raw = tmp_path / "raw.json"
    exp.atomic_write(raw, {"rows": panel["rows"]})
    source_hashes = exp.source_hashes(REPO, panel)
    artifact = exp.build_terminal_artifact(
        preconditions=preconditions,
        private_paths=private_paths,
        panel=panel,
        validation_receipts=_receipts(exp.REQUIRED_VALIDATION_NAMES),
        e2e_receipts=_receipts(exp.REQUIRED_E2E_NAMES),
        terminal_lint_receipts=_receipts(exp.REQUIRED_TERMINAL_LINT_NAMES),
        source_hashes=source_hashes,
        raw_rows_path=raw,
        started_at_utc="2026-09-16T12:00:00+00:00",
        ended_at_utc="2026-09-16T12:00:01+00:00",
        duration_s=1.0,
        phase_spans=[
            {
                "phase": "evaluation",
                "duration_s": 1.0,
                "completed_units": len(panel["rows"]),
            }
        ],
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["arc_resume_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["execution_venue"] == "host"
    assert artifact["flagged_adversarial"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["solve_provenance"] == "no_game_solve_cpu_transport_fixture"
    assert artifact["seal_for_exp7354"] is True
    assert artifact["promotion_value"] == 0
    assert artifact["current_repair_hashes"]
    assert artifact["first_loss_receipt"]["reproduced"] is True
    assert set(artifact) <= set(artifact["field_principles"])

    for field, value, error in (
        ("schema", "bad", "schema or experiment identity mismatch"),
        ("run_date", "bad", "milestone or run date mismatch"),
        ("model_invoked", True, "current model declaration mismatch"),
        ("invocation_counts", {}, "current invocation counts mismatch"),
        ("inference_substrate", "bad", "inference substrate mismatch"),
        ("execution_venue", "board", "execution venue mismatch"),
        ("verdict_class", "unknown", "verdict class mismatch"),
        ("arc_resume_ready_score", 0, "readiness reduction mismatch"),
        ("solve_provenance", "live_agent_self_discovery", "solve provenance mismatch"),
        ("field_principles", {}, "field principles mismatch"),
        ("reproducibility_checksum", "sha256:" + "0" * 64, "checksum mismatch"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in exp.validate_artifact(changed)

    unsafe = deepcopy(artifact)
    unsafe["verdict_class"] = "disqualified"
    assert "unsafe readiness on non-ready artifact" in exp.validate_artifact(unsafe)

    failed_receipts = _receipts(exp.REQUIRED_VALIDATION_NAMES)
    failed_receipts[1]["passed"] = False
    failed_receipts[1]["exit_code"] = 1
    failed = exp.build_terminal_artifact(
        preconditions=preconditions,
        private_paths=private_paths,
        panel=panel,
        validation_receipts=failed_receipts,
        e2e_receipts=_receipts(exp.REQUIRED_E2E_NAMES),
        terminal_lint_receipts=_receipts(exp.REQUIRED_TERMINAL_LINT_NAMES),
        source_hashes=source_hashes,
        raw_rows_path=raw,
        started_at_utc="2026-09-16T12:00:00+00:00",
        ended_at_utc="2026-09-16T12:00:01+00:00",
        duration_s=1.0,
        phase_spans=[],
    )
    assert failed["status"] == "disqualified"
    assert failed["verdict_class"] == "disqualified"
    assert failed["arc_resume_ready_score"] == 0
    assert failed["promotion_value"] == 0
    assert failed["gate_check_summary"]["first_failure"]["check"] == "scoped_validation"
    assert exp.validate_artifact(failed) == []

    blocked = exp.build_blocked_artifact(
        preconditions=exp.check_preconditions(tmp_path / "missing"),
        started_at_utc="2026-09-16T12:00:00+00:00",
        ended_at_utc="2026-09-16T12:00:01+00:00",
        duration_s=1.0,
    )
    assert blocked["status"].startswith("blocked_")
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["arc_resume_ready_score"] == 0
    assert blocked["gate_check_summary"]["first_failure"]
    assert exp.validate_artifact(blocked) == []


def test_req_7345_validation_scope_and_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-ARC-WMTE-7345-VALIDATION creates parents before each child launch."""

    captured: dict[str, object] = {}

    def scoped(*args: object, **kwargs: object) -> dict[str, object]:
        captured["scoped_args"] = args
        captured["scoped_kwargs"] = kwargs
        assert Path(str(kwargs["basetemp"])).is_dir()
        return {"validation_receipts": [{"name": "focused_pytest"}]}

    monkeypatch.setattr(exp.validation_scope, "run_scoped_validation", scoped)
    private = tmp_path / "private"
    assert exp.run_scoped_validation(tmp_path, private) == [{"name": "focused_pytest"}]
    kwargs = captured["scoped_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["basetemp"] == private / "scoped"

    def commands(root: Path, specs: object, **kwargs: object) -> list[dict[str, object]]:
        captured["e2e_specs"] = list(specs)  # type: ignore[arg-type]
        assert private.is_dir()
        return [{"name": "e2e_009"}]

    monkeypatch.setattr(exp.validation_scope, "run_commands", commands)
    assert exp.run_e2e(tmp_path, private) == [{"name": "e2e_009"}]
    assert [row.name for row in captured["e2e_specs"]] == list(exp.REQUIRED_E2E_NAMES)  # type: ignore[union-attr]
    specs = exp.terminal_lint_specs(tmp_path, tmp_path / "candidate.json")
    assert [row.name for row in specs] == list(exp.REQUIRED_TERMINAL_LINT_NAMES)

    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    sidecar = panel_dir / "sidecar.json"
    sidecar.write_text("{}", encoding="utf-8")
    for relative in exp.HASH_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.as_posix(), encoding="utf-8")
    hashes = exp.source_hashes(tmp_path, {"fixture_sidecar_path": str(sidecar)})
    assert hashes[exp.HISTORICAL_RESULT_PATH.as_posix()]["role"] == "historical_diagnostic"
    assert hashes[str(sidecar)]["counts_as_current_model_invocation"] is False

    exp.progress(time.monotonic(), "test", "boundary", completed_units=1)
    assert "phase=test event=boundary" in capsys.readouterr().out


def test_req_7345_runner_blocked_success_and_thin_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7345 stops on blocked inputs and atomically publishes success."""

    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    writes: list[Path] = []
    original_atomic = exp.atomic_write

    def record(path: Path, value: dict[str, object]) -> None:
        writes.append(path)
        original_atomic(path, value)

    monkeypatch.setattr(exp, "atomic_write", record)
    monkeypatch.setattr(
        exp,
        "check_preconditions",
        lambda root: {
            "passed": False,
            "checks": [],
            "first_failure": {
                "check": "missing",
                "upstream": "input",
                "artifact_field": "exists",
                "expected": True,
                "observed": False,
                "passed": False,
            },
            "first_loss_receipt": {},
            "historical_exp7336": {},
        },
    )
    blocked = exp.run_experiment(Namespace(date=exp.RUN_DATE))
    assert blocked["verdict_class"] == "blocked"
    assert tmp_path / exp.RESULT_PATH in writes

    rows = [
        {
            "arm": arm,
            "passed": True,
            "completion_calls": 1,
            "completion_limit": 2,
            "generated_tokens": 1,
            "generated_token_limit": 4096,
            "plan_authorized": False,
            "stale_or_duplicate_consumption": 0,
            "completion_budget_overrun": 0,
            "token_budget_overrun": 0,
            "failures": [],
            "abstentions": 0,
            "censored": False,
        }
        for arm in exp.REQUIRED_ARMS
    ]
    rows[0].update(
        {
            "policy_class": "E3AgentPolicy",
            "result_delivery_count": 1,
            "later_request_result_occurrences": 1,
            "receipt_captured": True,
            "verified_engine_installed": True,
            "plan_installed": True,
            "later_policy_action": True,
        }
    )
    preconditions = {
        "passed": True,
        "checks": [],
        "first_failure": None,
        "first_loss_receipt": {"reproduced": True},
        "historical_exp7336": {"diagnostic_only": True},
    }
    monkeypatch.setattr(exp, "check_preconditions", lambda root: preconditions)
    monkeypatch.setattr(
        exp,
        "qualify_private_basetemps",
        lambda root: {
            "missing_parent_failure_reproduced": True,
            "parents_created_before_launch": True,
            "concurrent_roots_distinct": True,
            "first_run_cleaned": True,
            "second_run_survived_cleanup": True,
        },
    )
    monkeypatch.setattr(
        exp,
        "run_resume_qualification",
        lambda path: {"rows": rows, "fixture_sidecar_path": "", "fixture_sidecar_sha256": ""},
    )
    monkeypatch.setattr(
        exp, "run_scoped_validation", lambda root, private: _receipts(exp.REQUIRED_VALIDATION_NAMES)
    )
    monkeypatch.setattr(exp, "run_e2e", lambda root, private: _receipts(exp.REQUIRED_E2E_NAMES))
    monkeypatch.setattr(exp, "source_hashes", lambda root, panel: {})
    monkeypatch.setattr(
        exp.validation_scope,
        "run_commands",
        lambda *args, **kwargs: _receipts(exp.REQUIRED_TERMINAL_LINT_NAMES),
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda value: [])
    completed = exp.run_experiment(Namespace(date=exp.RUN_DATE))
    assert completed["arc_resume_ready_score"] == 1
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert exp.parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE

    def corrupt_candidate(path: Path, value: dict[str, object]) -> None:
        original_atomic(path, value)
        if path.name == "measured_terminal_candidate.json":
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(str(value["reproducibility_checksum"]), "bad"), encoding="utf-8"
            )

    monkeypatch.setattr(exp, "atomic_write", corrupt_candidate)
    with pytest.raises(RuntimeError, match="terminal candidate reload mismatch"):
        exp.run_experiment(Namespace(date=exp.RUN_DATE))

    monkeypatch.setattr(exp, "atomic_write", original_atomic)
    broken_rows = deepcopy(rows)
    broken_rows[0]["completion_calls"] = 3
    monkeypatch.setattr(
        exp,
        "run_resume_qualification",
        lambda path: {
            "rows": broken_rows,
            "fixture_sidecar_path": "",
            "fixture_sidecar_sha256": "",
        },
    )
    with pytest.raises(RuntimeError, match="independent raw-row reduction failed"):
        exp.run_experiment(Namespace(date=exp.RUN_DATE))

    monkeypatch.setattr(
        exp,
        "run_resume_qualification",
        lambda path: {"rows": rows, "fixture_sidecar_path": "", "fixture_sidecar_sha256": ""},
    )

    monkeypatch.setattr(exp, "validate_artifact", lambda value: ["injected"])
    with pytest.raises(RuntimeError, match="terminal artifact validation failed"):
        exp.run_experiment(Namespace(date=exp.RUN_DATE))

    monkeypatch.setattr(exp, "run_experiment", lambda args: {"status": "blocked_missing_input"})
    assert exp.main(["--date", exp.RUN_DATE]) == 1

    called: list[list[str] | None] = []
    monkeypatch.setattr(exp, "main", lambda argv=None: called.append(argv) or 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(REPO / exp.WRAPPER_PATH), run_name="__main__")
    assert stopped.value.code == 0
    assert called == [None]
