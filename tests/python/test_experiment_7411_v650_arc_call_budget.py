"""Experiment 7411 artifact and live-plumbing checks for REQ-ARC-WMTE-7411."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_7411_v650_arc_call_budget as exp7411
from carnot.experiment_7411_v650_arc_call_budget import (
    EXPERIMENT_ID,
    MILESTONE,
    RUN_DATE,
    ZERO_INVOCATION_COUNTS,
    affected_manifest,
    artifact_checksum,
    build_terminal_artifact,
    build_validation_plan,
    collect_preconditions,
    diagnose_exp7406_overflow,
    e2e_command_specs,
    freeze_game_rotation,
    independent_reduce_file,
    inspect_redirect_ledger,
    parse_args,
    run_callback_matrix,
    run_scored_plumbing,
    validate_artifact,
)

REPO = Path(__file__).resolve().parents[2]

# The real scored factory and offline arcade allocate their shared SDK state on
# first use. That module-level cache is not a per-test leak.
pytestmark = pytest.mark.memory_watchdog_skip


def test_historical_overflow_is_authenticated_and_explained() -> None:
    """SCENARIO-ARC-WMTE-7411-OVERFLOW keeps the failed panel diagnostic."""
    diagnosis = diagnose_exp7406_overflow(REPO)
    assert diagnosis["experiment_id"] == "exp7406-arc-generalization"
    assert diagnosis["generation_calls_attempted"] == 35
    assert diagnosis["planned_call_limit"] == 12
    assert diagnosis["overflow_calls"] == 23
    assert diagnosis["cause_reproduced"] is True
    assert diagnosis["boundary_that_bypassed_budget"] == "generation_dispatch"
    assert diagnosis["old_enforcement_boundary"] == "terminal_reducer_only"
    assert diagnosis["eligible_science"] is False
    assert diagnosis["scope"] == "historical"


def test_callback_matrix_covers_all_branches_and_never_exceeds_two(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7411-ATOMIC-RESERVATION covers every callback branch."""
    panel = run_callback_matrix(tmp_path)
    assert panel["overflow_diagnostic"]["dispatches"] > 2
    assert panel["overflow_diagnostic"]["eligible_science"] is False
    assert panel["all_controls_passed"] is True
    assert {
        "primary",
        "parser_retry",
        "repair",
        "refinement",
        "supervisor",
        "nested",
        "concurrent",
    } <= set(panel["branches_exercised"])
    assert all(row["attempted"] <= 2 for row in panel["control_rows"])
    assert all(row["accounting_valid"] for row in panel["control_rows"])
    assert panel["late_write_violations"] == 0
    assert panel["cold_restart_parity"] is True


def test_three_game_scored_plumbing_uses_real_factory_and_twelve_actions(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7411-SCORED-PLUMBING drives the real scored policy."""
    selection = freeze_game_rotation(REPO)
    assert len(selection["selected_games"]) == 3
    assert selection["label_blind"] is True
    panel = run_scored_plumbing(selection["selected_games"], tmp_path)
    assert len(panel["rows"]) == 3
    assert all(row["factory"] == "make_carnot_agent" for row in panel["rows"])
    assert all(row["policy_class"] == "E3AgentPolicy" for row in panel["rows"])
    assert all(row["environment_actions"] <= 12 for row in panel["rows"])
    assert all(row["observed_dispatches"] <= 2 for row in panel["rows"])
    assert all(row["adapter_disabled"] is True for row in panel["rows"])
    assert all(row["solve_provenance"] == "development_proxy" for row in panel["rows"])
    assert all(row["solve_credit"] == 0 for row in panel["rows"])
    assert panel["budget_violations"] == 0


def test_redirect_ledger_banks_only_supported_outcomes() -> None:
    """REQ-ARC-WMTE-7411 does not invent a supervisor refinement."""
    disposition = inspect_redirect_ledger(REPO)
    assert disposition["banked_only"] is True
    assert disposition["arm_order_changed"] is False
    assert disposition["new_arm_created"] is False
    assert disposition["recommendation"] in {"nothing_to_refine", "bank_existing_recommendations"}


def _artifact(tmp_path: Path) -> dict:
    callbacks = run_callback_matrix(tmp_path / "callbacks")
    selection = freeze_game_rotation(REPO)
    scored = run_scored_plumbing(selection["selected_games"], tmp_path / "scored")
    redirect = inspect_redirect_ledger(REPO)
    validations = [
        {
            "name": name,
            "command_argv": ["fixture", name],
            "environment": {"COVERAGE_FILE": str(tmp_path / ".coverage")},
            "exit_code": 0,
            "duration_s": 0.01,
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in (
            "worktree_imports",
            "focused_pytest",
            "changed_module_coverage",
            "changed_module_coverage_report",
            "ruff_check",
            "ruff_format",
            "changed_module_mypy",
            "scoped_spec_coverage",
            "e2e_009",
            "e2e_010",
            "e2e_offline_smoke",
            "independent_reducer",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        )
    ]
    return build_terminal_artifact(
        started_at_utc="2026-09-19T00:00:00+00:00",
        ended_at_utc="2026-09-19T00:00:03+00:00",
        duration_s=3.0,
        phase_spans=[{"phase": "measured", "start_s": 0.0, "end_s": 3.0}],
        preconditions_checked=[
            {
                "check": "fixture",
                "category": "precondition",
                "operator": "==",
                "expected": True,
                "observed": True,
                "passed": True,
                "upstream": "test",
                "artifact_field": "fixture",
            }
        ],
        source_artifact_hashes={"fixture": {"path": "fixture", "sha256": "sha256:" + "b" * 64}},
        selection=selection,
        diagnosis=diagnose_exp7406_overflow(REPO),
        callback_panel=callbacks,
        scored_panel=scored,
        redirect_disposition=redirect,
        validation_receipts=validations,
        provenance_sidecars={
            "historical": {"path": "private", "sha256": "sha256:" + "c" * 64},
            "scripted": {"path": "private", "sha256": "sha256:" + "d" * 64},
        },
    )


@pytest.fixture(scope="module")
def ready_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build the real scored fixture once for claim-drift mutations."""

    return _artifact(tmp_path_factory.mktemp("exp7411-ready-artifact"))


def test_terminal_artifact_is_ready_but_never_claims_live_efficacy(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7411-TERMINAL separates plumbing from efficacy."""
    artifact = _artifact(tmp_path)
    assert validate_artifact(artifact) == []
    assert artifact["experiment_id"] == EXPERIMENT_ID
    assert artifact["milestone"] == MILESTONE
    assert artifact["run_date"] == RUN_DATE
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["arc_budget_ready_score"] == 1
    assert artifact["live_efficacy_score"] == artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["flagged_adversarial"] is False
    assert artifact["reproducibility_checksum"] == artifact_checksum(artifact)
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["small_ebm_training"]["receipt_class"] == "small_ebm_training"
    assert artifact["future_live_model_prerequisite"]["required_model"] == (
        "unsloth/Qwen3.8-27B-GGUF"
    )

    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert independent_reduce_file(path)["matches_declared"] is True


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema", "wrong", "identity_invalid"),
        ("MODEL_SPECS", [{"model": "forbidden"}], "current_model_fields_invalid"),
        ("model_invoked", True, "current_model_fields_invalid"),
        ("invocation_counts", {}, "current_model_fields_invalid"),
        ("inference_substrate", 1, "substrate_invalid"),
        ("inference_substrate_class", "gpu", "substrate_invalid"),
        ("execution_venue", "remote", "substrate_invalid"),
        ("promotion_score", 1, "score_invalid"),
        ("live_efficacy_score", 1, "score_invalid"),
        ("arc_budget_ready_score", 0, "readiness_mismatch"),
        ("verdict_class", "positive", "verdict_invalid"),
        ("duration_s", 0, "duration_invalid"),
    ],
)
def test_validator_rejects_terminal_claim_drift(
    ready_artifact: dict, field: str, value: object, error: str
) -> None:
    """REQ-ARC-WMTE-7411 fails closed on terminal claim drift."""
    changed = deepcopy(ready_artifact)
    changed[field] = value
    changed["reproducibility_checksum"] = artifact_checksum(changed)
    assert error in validate_artifact(changed)


def test_validator_rejects_rows_gates_principles_and_checksum(ready_artifact: dict) -> None:
    """SCENARIO-ARC-WMTE-7411-TERMINAL rejects inconsistent raw evidence."""
    artifact = deepcopy(ready_artifact)
    artifact["callback_rows"][0]["disposition"] = "in_flight"
    assert "raw_reduction_mismatch" in validate_artifact(artifact)

    artifact = deepcopy(ready_artifact)
    artifact["acceptance_gate_results"][0]["passed"] = False
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    assert "readiness_mismatch" in validate_artifact(artifact)

    artifact = deepcopy(ready_artifact)
    artifact["field_principles"].pop("schema")
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    assert "field_principles_incomplete" in validate_artifact(artifact)

    artifact = deepcopy(ready_artifact)
    artifact["reproducibility_checksum"] = "sha256:bad"
    assert "checksum_mismatch" in validate_artifact(artifact)

    artifact = deepcopy(ready_artifact)
    artifact["milestone"] = "wrong"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    assert "identity_invalid" in validate_artifact(artifact)


def test_local_io_preconditions_and_gate_helpers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7411 authenticates exact bytes and measured boundaries."""
    target = tmp_path / "nested" / "value.json"
    exp7411.atomic_json(target, {"value": 7})
    assert exp7411.load_object(target) == {"value": 7}
    bad = tmp_path / "bad.json"
    bad.write_text("not-json", encoding="utf-8")
    assert exp7411.load_object(bad) == {}
    assert exp7411.load_object(tmp_path / "missing.json") == {}
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert exp7411.load_object(sequence) == {}

    assert (
        exp7411.gate_row(
            "member",
            "test",
            ("expected", "observed"),
            "observed",
            upstream="test",
            artifact_field="value",
            operator="in",
        )["passed"]
        is True
    )
    exp7411.progress(0.0, "test", "boundary", completed_units=1)
    assert "completed_units=1" in capsys.readouterr().out
    assert "+00:00" in exp7411.utc_now()

    checks, hashes = collect_preconditions(REPO)
    assert all(row["passed"] for row in checks)
    assert exp7411.HISTORICAL_PATH.as_posix() in hashes


def test_missing_local_precondition_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7411 blocks unavailable local input bytes explicitly."""
    spec = tmp_path / "capability.md"
    spec.write_text("REQ-ARC-WMTE-7411", encoding="utf-8")
    ops = tmp_path / "ops"
    ops.mkdir()
    (ops / "exclusion_manifest.yaml").write_text("retired: []\n", encoding="utf-8")
    monkeypatch.setattr(exp7411, "INPUT_PATHS", (Path("missing-input.json"),))
    monkeypatch.setattr(exp7411, "SPEC_PATH", Path("capability.md"))
    checks, hashes = collect_preconditions(tmp_path)
    assert checks[0]["observed"] is None
    assert checks[0]["passed"] is False
    assert hashes == {}


def test_environment_restore_and_scored_failure_are_truthful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7411-SCORED-PLUMBING retains failures without solve credit."""
    key = "CARNOT_EXP7411_RESTORE_TEST"
    monkeypatch.setenv(key, "before")
    _old, restore = exp7411._temporary_environment({key: "during", "CARNOT_EXP7411_NEW": "set"})
    assert os.environ[key] == "during"
    restore()
    assert os.environ[key] == "before"
    assert "CARNOT_EXP7411_NEW" not in os.environ

    import scripts.arc_leaderboard_eval as leaderboard

    def fail_run(*_args: object, **_kwargs: object) -> dict:
        raise RuntimeError("scripted environment failure")

    monkeypatch.setattr(leaderboard, "run_game", fail_run)
    panel = run_scored_plumbing(["r11l"], tmp_path / "failed-scored")
    assert panel["rows"][0]["disposition"] == "complete_error"
    assert "scripted environment failure" in panel["rows"][0]["error"]
    assert panel["rows"][0]["solve_credit"] == 0


def test_scoped_command_runner_and_terminal_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7411-TERMINAL preserves exact scoped command receipts."""
    spec = exp7411.validation_scope.CommandSpec(
        "e2e_offline_smoke", ("python", "-c", "pass"), "fixture"
    )
    regular = exp7411.validation_scope.CommandSpec("regular", ("python", "-c", "pass"), "fixture")

    def fake_run_commands(
        root: Path,
        specs: list,
        *,
        log_dir: Path,
        extra_env: dict[str, str],
    ) -> list[dict]:
        assert root == REPO
        current = specs[0]
        assert current in {spec, regular}
        if current is spec:
            assert log_dir.name == "00_e2e_offline_smoke"
            assert extra_env["CARNOT_ARC_DISABLE_INDUCTION"] == "1"
        else:
            assert log_dir.name == "01_regular"
            assert extra_env == {}
        return [{"name": current.name, "exit_code": 0, "passed": True}]

    monkeypatch.setattr(exp7411.validation_scope, "run_commands", fake_run_commands)
    receipts = exp7411._run_specs(REPO, [spec, regular], tmp_path / "logs")
    assert receipts[0]["started_at_utc"] <= receipts[0]["ended_at_utc"]
    assert receipts[0]["environment"]["CARNOT_ARC_DISABLE_INDUCTION"] == "1"

    candidate = tmp_path / "candidate.json"
    terminal = exp7411._terminal_specs(REPO, candidate)
    assert [row.name for row in terminal] == [
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    assert all(str(candidate) in row.argv for row in terminal)


def test_scoped_plans_entrypoint_and_arguments_use_private_parents(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7411 freezes affected and end-to-end command scope."""
    manifest = affected_manifest()
    assert manifest.experiment_id == EXPERIMENT_ID
    assert set(manifest.test_paths) >= {
        "tests/python/test_experiment_7411_v650_arc_call_budget.py",
        "tests/python/test_arc_request_budget.py",
    }
    commands = build_validation_plan(REPO, tmp_path / "validation")
    assert {row.name for row in commands} == {
        "worktree_imports",
        "focused_pytest",
        "changed_module_coverage",
        "changed_module_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
    }
    e2e = e2e_command_specs(REPO, tmp_path / "e2e")
    assert {row.name for row in e2e} == {"e2e_009", "e2e_010", "e2e_offline_smoke"}
    smoke = next(row for row in e2e if row.name == "e2e_offline_smoke")
    assert "12" in smoke.argv
    assert str(tmp_path) in " ".join(smoke.argv)

    wrapper = REPO / "scripts/experiments/experiment_7411_v650_arc_call_budget.py"
    text = wrapper.read_text(encoding="utf-8")
    assert text.count("from carnot.") == 1
    assert "main()" in text
    assert "research_conductor" not in text

    args = parse_args(["--date", RUN_DATE])
    assert args.date == RUN_DATE
    with pytest.raises(SystemExit):
        parse_args(["--date", "20260920"])


def test_main_delegates_to_the_current_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7411 keeps the public entrypoint thin and date-locked."""
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(exp7411, "run_experiment", lambda root, date: calls.append((root, date)))
    assert exp7411.main(["--date", RUN_DATE]) == 0
    assert calls == [(REPO, RUN_DATE)]
