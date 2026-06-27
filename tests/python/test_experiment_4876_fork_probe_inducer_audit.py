"""Tests for Exp 4876 A1/A1b adversarial fork-probe audit.

Spec refs: REQ-ARC-WMTE-4876,
SCENARIO-ARC-WMTE-4876-A1-A1B-AUDIT,
SCENARIO-ARC-WMTE-4876-BLOCKED-A1-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4876_fork_probe_inducer_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _row(game: str, *, accuracy: float, bucket: str) -> dict[str, Any]:
    return {
        "game": game,
        "engine_heldout_accuracy": accuracy,
        "planned_bucket": bucket,
        "migrated": bucket == "COVERED",
        "winning_prefix_len": 2,
        "planned_pool_size": 1 if bucket != "NEVER_ENUMERATED" else 0,
        "heldout_transition_count": 5,
        "live_path_methods_called": [
            "E3AgentPolicy._induce_and_plan",
            "arc_executable_world_model.load_engine",
            "arc_executable_world_model.plan_in_model",
        ],
    }


def _a1_artifact() -> dict[str, Any]:
    return {
        "experiment_id": 4871,
        "honest_verdict": "complete_generation_wall_guidance_wall_high_accuracy_migration",
        "generator_backend": "gpu0_cuda",
        "duration_s": 65.0,
        "flagged_adversarial": False,
        "inference_substrate": "live_llm_inference",
        "fork_verdict": "GUIDANCE_WALL",
        "per_game_fork": {
            "cd82": _row("cd82", accuracy=0.8, bucket="COVERED"),
            "cn04": _row("cn04", accuracy=0.7, bucket="NEVER_ENUMERATED"),
            "ls20": _row("ls20", accuracy=0.9, bucket="ENUMERATED_BUT_LOST"),
        },
        "coverage_migration_count": 1,
        "median_engine_heldout_accuracy": 0.8,
        "positive_control_game": "tu93",
        "positive_control_migrated": True,
        "positive_control_fork": _row("tu93", accuracy=0.9, bucket="COVERED"),
        "planner_blind_to_banked_answer": True,
        "n_games_measured": 3,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "preconditions_checked": {
            "generator": {
                "ok": True,
                "generator_backend": "gpu0_cuda",
                "detail": "ok",
                "launch_env_cuda_visible_devices": "0",
            }
        },
    }


def _a1_source(*, injected: bool = False) -> str:
    if injected:
        return """
def measure_game_with_live_induce_plan(game, winning_prefix, proposer):
    policy = seeded_frontier(winning_prefix)
    policy._induce_and_plan()
    engine, is_done = e3.load_engine(game)
    plan = e3.plan_in_model(engine, is_done, policy.root_grid, hint=winning_prefix)
    return classify_planned_pool(game, winning_prefix, plan)
"""
    return """
def measure_game_with_live_induce_plan(game, winning_prefix, proposer):
    policy = cold_policy(game, proposer)
    policy._induce_and_plan()
    engine, is_done = e3.load_engine(game)
    plan = e3.plan_in_model(engine, is_done, policy.root_grid)
    return classify_planned_pool(game, winning_prefix, plan)
"""


def _delta_row(game: str, *, delta: float = 0.5) -> dict[str, Any]:
    return {
        "game": game,
        "baseline": 0.1,
        "refined": round(0.1 + delta, 6),
        "delta": delta,
        "repair_transition_ids": [0, 2],
        "remeasure_transition_ids": [1, 3, 4],
        "repair_counterexample_count": 2,
        "remeasure_transition_count": 3,
        "counterexamples_fixed": 1,
        "accepted_repairs": 1,
        "cegis_rounds": 1,
    }


def _a1b_artifact() -> dict[str, Any]:
    return {
        "experiment_id": 4872,
        "honest_verdict": "success_cegis_engine_accuracy_lift_0.500000",
        "per_game_accuracy_delta": {
            "cd82": _delta_row("cd82"),
            "cn04": _delta_row("cn04"),
            "ls20": _delta_row("ls20"),
        },
        "cegis_heldout_accuracy_delta_median": 0.5,
        "cegis_heldout_accuracy_delta_ci95": [0.5, 0.5],
        "delta_on_truly_heldout_split": True,
        "positive_control_passed": True,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
    }


def _tool_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {"returncode": 0, "stdout": "LIVE re-check: clean", "stderr": ""},
        {"loaded": True, "flag_count": 0, "flags": []},
        {"loaded": True, "flag_count": 0, "flags": []},
        {"passed": True, "returncode": 0, "stdout_tail": "OK", "stderr_tail": ""},
    )


def test_req_arc_wmte_4876_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4876: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4876",
        "SCENARIO-ARC-WMTE-4876-A1-A1B-AUDIT",
        "SCENARIO-ARC-WMTE-4876-BLOCKED-A1-ARTIFACT",
        mod.RESULT_RELATIVE_PATH,
        mod.AUDIT_REPORT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4876_good_a1_and_a1b_pass_gates() -> None:
    """SCENARIO-ARC-WMTE-4876-A1-A1B-AUDIT: all load-bearing gates pass."""

    summary, a1_adv, a1b_adv, lint = _tool_results()
    audit = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )

    assert audit["honest_verdict"] == "complete_a1_a1b_audited"
    assert audit["a1_genuinely_diagnostic"] is True
    assert audit["a1_ran_live_on_gpu0"] is True
    assert audit["planner_blind_confirmed"] is True
    assert audit["positive_control_confirmed"] is True
    assert audit["numbers_match_fork"] is True
    assert audit["a1b_delta_trustworthy"] is True
    assert audit["a1_failure_reasons"] == []
    assert audit["a1b_failure_reasons"] == []
    assert audit["checks"]["a1_numbers_match_fork"]["computed_fork_verdict"] == "GUIDANCE_WALL"
    assert audit["checks"]["a1b_delta"]["computed_ci95"] == [0.5, 0.5]


def test_scenario_arc_wmte_4876_adversarial_variants_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-4876-A1-A1B-AUDIT: hostile variants are explicit non-tests."""

    summary, a1_adv, a1b_adv, lint = _tool_results()

    too_short = _a1_artifact()
    too_short["duration_s"] = 59.0
    too_short["generator_backend"] = "cpu"

    control_failed = _a1_artifact()
    control_failed["positive_control_migrated"] = False
    control_failed["positive_control_fork"] = _row("tu93", accuracy=0.0, bucket="NEVER_ENUMERATED")

    wrong_fork = _a1_artifact()
    wrong_fork["fork_verdict"] = "PLANNER_GAP"

    bad_split = _a1b_artifact()
    bad_split["per_game_accuracy_delta"]["cd82"]["remeasure_transition_ids"] = [0, 1]
    bad_split["delta_on_truly_heldout_split"] = False

    oracle = _a1b_artifact()
    oracle["verifier_is_oracle"] = True

    cases = [
        (too_short, _a1_source(), _a1b_artifact(), "a1_not_live_on_gpu0"),
        (_a1_artifact(), _a1_source(injected=True), _a1b_artifact(), "banked_answer_used_before_classification"),
        (control_failed, _a1_source(), _a1b_artifact(), "positive_control_not_migrated"),
        (wrong_fork, _a1_source(), _a1b_artifact(), "fork_verdict_mismatch"),
        (_a1_artifact(), _a1_source(), bad_split, "a1b_split_not_disjoint"),
        (_a1_artifact(), _a1_source(), oracle, "a1b_verifier_is_oracle"),
    ]

    for a1_artifact, source_text, a1b_artifact, reason in cases:
        audit = mod.audit_sources(
            a1_artifact=a1_artifact,
            a1_source_text=source_text,
            a1_summarizer_result=summary,
            a1_adversarial_result=a1_adv,
            a1b_artifact=a1b_artifact,
            a1b_adversarial_result=a1b_adv,
            live_lint_result=lint,
        )
        all_reasons = audit["a1_failure_reasons"] + audit["a1b_failure_reasons"]
        assert reason in all_reasons
        if reason.startswith("a1b_"):
            assert audit["a1b_delta_trustworthy"] is False
        else:
            assert audit["a1_genuinely_diagnostic"] is False


def test_req_arc_wmte_4876_a1b_gate_skip_is_trustworthy_but_missing_artifact_is_not() -> None:
    """REQ-ARC-WMTE-4876: only explicit A1b gate skips count as delta trustworthy."""

    summary, a1_adv, a1b_adv, lint = _tool_results()
    high_a1 = _a1_artifact()
    high_a1["fork_verdict"] = "PLANNER_GAP"
    high_a1["coverage_migration_count"] = 0
    for row in high_a1["per_game_fork"].values():
        row["planned_bucket"] = "NEVER_ENUMERATED"
        row["migrated"] = False

    skipped = mod.audit_sources(
        a1_artifact=high_a1,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=None,
        a1b_adversarial_result=None,
        live_lint_result=lint,
    )
    low_a1 = _a1_artifact()
    low_a1["fork_verdict"] = "INDUCER_CEILING"
    low_a1["coverage_migration_count"] = 0
    low_a1["median_engine_heldout_accuracy"] = 0.1
    for row in low_a1["per_game_fork"].values():
        row["engine_heldout_accuracy"] = 0.1
        row["planned_bucket"] = "NEVER_ENUMERATED"
        row["migrated"] = False
    missing = mod.audit_sources(
        a1_artifact=low_a1,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=None,
        a1b_adversarial_result=None,
        live_lint_result=lint,
    )

    assert skipped["a1b_delta_trustworthy"] is True
    assert skipped["checks"]["a1b_delta"]["status"] == "gate_skipped"
    assert missing["a1b_delta_trustworthy"] is False
    assert "a1b_artifact_missing_after_inducer_ceiling_a1" in missing["a1b_failure_reasons"]


def test_req_arc_wmte_4876_build_schema_write_and_blocked_report(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4876: artifacts are checksum-stable and report appends are idempotent."""

    source = tmp_path / mod.A1_ARTIFACT_RELATIVE_PATH
    script = tmp_path / mod.A1_SCRIPT_RELATIVE_PATH
    a1b = tmp_path / mod.A1B_ARTIFACT_RELATIVE_PATH
    source.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    source.write_text(json.dumps(_a1_artifact()), encoding="utf-8")
    script.write_text(_a1_source(), encoding="utf-8")
    a1b.write_text(json.dumps(_a1b_artifact()), encoding="utf-8")

    summary, a1_adv, a1b_adv, lint = _tool_results()
    audit = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )
    artifact = mod.build_artifact(
        root=tmp_path,
        a1_artifact=_a1_artifact(),
        a1b_artifact=_a1b_artifact(),
        audit=audit,
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
        preconditions_checked={"ok": True},
        duration_s=0.0,
    )

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["duration_s"] == mod.DURATION_FLOOR_S
    assert artifact["a1_artifact_checksum"] == mod.file_checksum(source)
    assert artifact["a1_script_checksum"] == mod.file_checksum(script)
    assert artifact["a1b_artifact_checksum"] == mod.file_checksum(a1b)

    result_path = mod.write_artifact(artifact, root=tmp_path)
    report_path = mod.append_markdown_report(artifact, root=tmp_path)
    mod.append_markdown_report(artifact, root=tmp_path)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert report_path.read_text(encoding="utf-8").count("## Experiment 4876 A1/A1b Audit") == 1

    blocked = mod.blocked_artifact({"ok": False}, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_a1_artifact_missing"
    assert mod.artifact_schema_errors(blocked) == []

    broken = dict(artifact)
    broken.update(
        {
            "a1_genuinely_diagnostic": "yes",
            "a1_ran_live_on_gpu0": "yes",
            "planner_blind_confirmed": "yes",
            "positive_control_confirmed": "yes",
            "numbers_match_fork": "yes",
            "a1b_delta_trustworthy": "yes",
            "field_principles": {},
            "inference_substrate": "live_llm_inference",
            "checks": [],
            "a1_failure_reasons": "none",
            "a1b_failure_reasons": "none",
            "duration_s": 0.0,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    errors = mod.artifact_schema_errors(broken)
    for expected in (
        "a1_genuinely_diagnostic_must_be_bool",
        "a1_ran_live_on_gpu0_must_be_bool",
        "planner_blind_confirmed_must_be_bool",
        "positive_control_confirmed_must_be_bool",
        "numbers_match_fork_must_be_bool",
        "a1b_delta_trustworthy_must_be_bool",
        "field_principles_mismatch",
        "inference_substrate_mismatch",
        "checks_must_be_dict",
        "a1_failure_reasons_must_be_list",
        "a1b_failure_reasons_must_be_list",
        "duration_below_aggregation_floor",
        "reproducibility_checksum_mismatch",
    ):
        assert expected in errors


def test_req_arc_wmte_4876_run_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4876: checked-in A1/A1b artifacts produce the requested audit."""

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_a1_a1b_audited"
    assert artifact["a1_ran_live_on_gpu0"] is True
    assert artifact["planner_blind_confirmed"] is True
    assert artifact["positive_control_confirmed"] is False
    assert artifact["numbers_match_fork"] is False
    assert artifact["a1_genuinely_diagnostic"] is False
    assert artifact["a1b_delta_trustworthy"] is True
    assert artifact["checks"]["a1_live_gpu"]["generator_backend"] == "gpu0_cuda"
    assert "positive_control_not_migrated" in artifact["a1_failure_reasons"]
    assert "fork_verdict_missing" in artifact["a1_failure_reasons"]
    assert artifact["checks"]["a1b_delta"]["positive_control_passed"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4876_defensive_branches(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-ARC-WMTE-4876: malformed inputs fail closed without fabricating passes."""

    summary, a1_adv, a1b_adv, lint = _tool_results()

    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None
    non_object = tmp_path / "not_object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._read_json(non_object)

    assert mod._call_name(__import__("ast").Constant(value=1)) == ""
    assert mod._find_function(__import__("ast").parse("x = 1"), "missing") is None
    assert mod._first_parent_call(__import__("ast").Name(id="winning_prefix")) is None

    syntax = mod.audit_sources(
        a1_artifact={**_a1_artifact(), "planner_blind_to_banked_answer": False},
        a1_source_text="def measure_game_with_live_induce_plan(",
        a1_summarizer_result={"returncode": 2},
        a1_adversarial_result={"loaded": True, "flag_count": 1, "flags": []},
        a1b_artifact=None,
        a1b_adversarial_result=None,
        live_lint_result={"passed": False},
    )
    assert "a1_source_not_parseable" in syntax["a1_failure_reasons"]
    assert "artifact_planner_blind_flag_false" in syntax["a1_failure_reasons"]
    assert "a1_summarizer_failed" in syntax["a1_failure_reasons"]
    assert "a1_adversarial_verify_flagged" in syntax["a1_failure_reasons"]
    assert "live_path_unreachable" in syntax["a1_failure_reasons"]

    missing_function = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text="def not_the_path():\n    return None\n",
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )
    assert "measure_game_with_live_induce_plan_missing" in missing_function["a1_failure_reasons"]

    missing_calls = mod.audit_sources(
        a1_artifact={**_a1_artifact(), "solve_provenance": "live_agent_self_discovery"},
        a1_source_text="def measure_game_with_live_induce_plan(game, winning_prefix, proposer):\n    return []\n",
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )
    for reason in (
        "winning_prefix_not_used_for_classification",
        "induce_and_plan_not_called",
        "load_engine_not_called",
        "plan_in_model_not_called",
        "solve_provenance_not_development_proxy",
    ):
        assert reason in missing_calls["a1_failure_reasons"]

    backend_from_model = _a1_artifact()
    backend_from_model.pop("generator_backend")
    backend_from_model["model_specs"] = {"backend": "igpu_hip"}
    assert mod.audit_sources(
        a1_artifact=backend_from_model,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_ran_live_on_gpu0"] is True

    flagged = _a1_artifact()
    flagged["flagged_adversarial"] = True
    assert "a1_flagged_adversarial" in mod.audit_sources(
        a1_artifact=flagged,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_failure_reasons"]

    bad_backend = _a1_artifact()
    bad_backend["generator_backend"] = "cpu"
    bad_backend.pop("preconditions_checked")
    assert "a1_generator_backend_not_gpu0_or_igpu" in mod.audit_sources(
        a1_artifact=bad_backend,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_failure_reasons"]

    bad_control = _a1_artifact()
    bad_control["positive_control_game"] = "not_tu93"
    bad_control.pop("positive_control_fork")
    control_reasons = mod.audit_sources(
        a1_artifact=bad_control,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_failure_reasons"]
    assert "positive_control_row_missing" in control_reasons
    assert "positive_control_not_tu93" in control_reasons

    invalid_numbers = _a1_artifact()
    invalid_numbers["per_game_fork"] = {
        "bad": {"planned_bucket": "NOPE", "engine_heldout_accuracy": 2.0, "migrated": "no"},
        "bad2": {"planned_bucket": "COVERED", "engine_heldout_accuracy": "x", "migrated": False},
    }
    invalid_numbers["n_games_measured"] = "x"
    invalid_numbers["median_engine_heldout_accuracy"] = "x"
    invalid_numbers["fork_verdict"] = "UNKNOWN"
    invalid_numbers["coverage_migration_count"] = 99
    number_reasons = mod.audit_sources(
        a1_artifact=invalid_numbers,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_failure_reasons"]
    for reason in (
        "invalid_planned_bucket",
        "invalid_engine_heldout_accuracy",
        "invalid_migrated_flag",
        "row_migrated_mismatch",
        "n_games_measured_not_integer",
        "n_games_measured_mismatch",
        "n_games_measured_below_3",
        "median_engine_heldout_accuracy_mismatch",
        "invalid_fork_verdict",
        "coverage_migration_count_mismatch",
    ):
        assert reason in number_reasons
    assert mod._computed_fork_verdict({"not_a_row": []}) is None

    assert mod._bootstrap_ci95([], iterations=2) == [None, None]
    varied_ci = mod._bootstrap_ci95([0.0, 1.0, 2.0], iterations=5, seed=1)
    assert varied_ci[0] <= varied_ci[1]
    assert mod._split_is_disjoint({"repair_transition_ids": ["bad"], "remeasure_transition_ids": [1]}) is False

    bad_a1b = _a1b_artifact()
    bad_a1b["cegis_heldout_accuracy_delta_median"] = 9.0
    bad_a1b["cegis_heldout_accuracy_delta_ci95"] = [9.0, 9.0]
    bad_a1b["live_path_reachable"] = False
    audit = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=bad_a1b,
        a1b_adversarial_result={
            "loaded": True,
            "flag_count": 1,
            "flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}],
        },
        live_lint_result=lint,
    )
    assert "a1b_delta_median_mismatch" in audit["a1b_failure_reasons"]
    assert "a1b_delta_ci95_mismatch" in audit["a1b_failure_reasons"]
    assert "a1b_circular_moat_overclaim" in audit["a1b_failure_reasons"]
    assert "a1b_live_path_unreachable" in audit["a1b_residual_reasons"]

    blocked = mod.run(root=tmp_path, write=True, now=iter([1.0, 1.1]).__next__)
    assert blocked["honest_verdict"] == "blocked_a1_artifact_missing"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    source = tmp_path / mod.A1_ARTIFACT_RELATIVE_PATH
    script = tmp_path / mod.A1_SCRIPT_RELATIVE_PATH
    a1b = tmp_path / mod.A1B_ARTIFACT_RELATIVE_PATH
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    for path, content in (
        (source, json.dumps(_a1_artifact())),
        (script, _a1_source()),
        (a1b, json.dumps(_a1b_artifact())),
        (spec, "REQ-ARC-WMTE-4876"),
        (tmp_path / "scripts/summarize_artifact.py", ""),
        (tmp_path / "scripts/adversarial_verify.py", ""),
        (tmp_path / "scripts/arc_orphan_solver_lint.py", ""),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(mod, "run_summarizer", lambda _path: summary)
    monkeypatch.setattr(mod, "run_adversarial_verify", lambda _path: a1b_adv)
    monkeypatch.setattr(mod, "run_arc_orphan_solver_lint", lambda _root: lint)
    complete = mod.run(root=tmp_path, write=True, now=iter([2.0, 2.1]).__next__)
    assert complete["honest_verdict"] == "complete_a1_a1b_audited"

    broken = dict(complete)
    broken.update(
        {
            "honest_verdict": "bad",
            "a1_genuinely_diagnostic": True,
            "a1_failure_reasons": ["still_bad"],
            "a1b_delta_trustworthy": True,
            "a1b_failure_reasons": ["still_bad"],
            "random_seed": 0,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    schema_errors = mod.artifact_schema_errors(broken)
    for expected in (
        "honest_verdict_missing_terminal_prefix",
        "diagnostic_artifact_has_a1_failure_reasons",
        "trustworthy_a1b_has_failure_reasons",
        "random_seed_mismatch",
    ):
        assert expected in schema_errors
    with pytest.raises(ValueError):
        mod.write_artifact(broken, root=tmp_path)

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError):
        mod.run(root=tmp_path, write=False, now=iter([3.0, 3.1]).__next__)
