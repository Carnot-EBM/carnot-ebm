"""Tests for Exp 4897 value-gap representation adversarial audit.

Spec refs: REQ-ARC-WMTE-4897,
SCENARIO-ARC-WMTE-4897-A1-A1B-AUDIT,
SCENARIO-ARC-WMTE-4897-A1B-LIVE-OR-GATE-SKIPPED,
SCENARIO-ARC-WMTE-4897-BLOCKED-A1-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import ast

import pytest

from carnot import experiment_4897_value_gap_representation_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _a1_row(
    game: str,
    *,
    baseline: float,
    decision_need: float,
    recall: float = 0.5,
    bucket: str = "NEVER_ENUMERATED",
    author_ids: list[str] | None = None,
    heldout_ids: list[str] | None = None,
) -> dict[str, Any]:
    delta = round(decision_need - baseline, 6)
    return {
        "game": game,
        "cell_recall": recall,
        "value_acc_code_baseline": round(baseline, 6),
        "value_acc_decision_need": round(decision_need, 6),
        "value_delta": delta,
        "planned_bucket": bucket,
        "migrated": bucket == "COVERED",
        "author_transition_ids": author_ids if author_ids is not None else ["author:0", "author:1"],
        "heldout_transition_ids": heldout_ids if heldout_ids is not None else ["heldout:0", "heldout:1"],
        "baseline_transition_ids": ["heldout:0", "heldout:1"],
        "author_transition_count": 2,
        "heldout_transition_count": 2,
        "cold_transition_count": 2,
        "target_table_row_count": 2,
        "decision_need_target_kinds": ["action_effect"],
        "live_path_methods_called": [
            "DecisionNeedTargetTable",
            "arc_executable_world_model.load_engine",
            "arc_executable_world_model.plan_in_model",
        ],
    }


def _a1_artifact(*, fork: str = "VALUE_GAP_REPRESENTATION_INVARIANT") -> dict[str, Any]:
    if fork == "REPRESENTATION_UNLOCKS_VALUE":
        rows = {
            "cd82": _a1_row("cd82", baseline=0.2, decision_need=0.5, bucket="COVERED"),
            "cn04": _a1_row("cn04", baseline=0.2, decision_need=0.5),
            "ls20": _a1_row("ls20", baseline=0.2, decision_need=0.5),
        }
        median_delta = 0.3
        ci95 = [0.3, 0.3]
        migrations = 1
    else:
        rows = {
            "cd82": _a1_row("cd82", baseline=0.2, decision_need=0.2),
            "cn04": _a1_row("cn04", baseline=0.3, decision_need=0.3),
            "ls20": _a1_row("ls20", baseline=0.4, decision_need=0.4),
        }
        median_delta = 0.0
        ci95 = [0.0, 0.0]
        migrations = 0
    return {
        "experiment_id": 4892,
        "honest_verdict": f"complete_decision_need_no_value_lift_{fork}",
        "fork_verdict": fork,
        "generator_backend": "gpu0_cuda",
        "duration_s": 75.0,
        "flagged_adversarial": False,
        "inference_substrate": "live_llm_inference",
        "per_game_value_gap": rows,
        "positive_control_game": "tu93",
        "positive_control_non_degenerate": True,
        "positive_control_value_gap": _a1_row(
            "tu93", baseline=0.2, decision_need=0.4, recall=0.2
        ),
        "delta_on_truly_heldout_split": True,
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "n_games_measured": len(rows),
        "coverage_migration_count": migrations,
        "engine_cell_recall_median": 0.5,
        "decision_need_value_accuracy_delta_median": median_delta,
        "decision_need_value_accuracy_delta_ci95": ci95,
        "decision_need_config": {"bootstrap_iterations": 25},
        "random_seed": 20260628,
        "model_specs": {"backend": "gpu0_cuda", "name": "Qwen3.5-9B-MTP"},
    }


def _a1_source(*, injected: bool = False) -> str:
    if injected:
        return """
def measure_game_with_decision_need_targets(game, winning_prefix, proposer):
    cold_transitions = a1._collect_cold_policy_transitions(game=game, proposer=proposer)
    table = DecisionNeedTargetTable.author(winning_prefix, game=game)
    planned, plan_error = _plan_with_decision_need_table(game=game, table=table)
    return a1.classify_planned_pool(game, winning_prefix, planned)
"""
    return """
def measure_game_with_decision_need_targets(game, winning_prefix, proposer):
    cold_transitions = a1._collect_cold_policy_transitions(game=game, proposer=proposer)
    table = DecisionNeedTargetTable.author(cold_transitions, game=game)
    planned, plan_error = _plan_with_decision_need_table(game=game, table=table)
    return a1.classify_planned_pool(game, winning_prefix, planned)
"""


def _a1b_artifact(a1: dict[str, Any] | None = None) -> dict[str, Any]:
    source = a1 or _a1_artifact()
    rows = {
        game: {
            "game": game,
            "cell_recall": row["cell_recall"],
            "value_acc_code_baseline": row["value_acc_code_baseline"],
            "value_acc_latent": row["value_acc_code_baseline"],
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "fit_transition_ids": ["fit:0", "fit:1"],
            "heldout_transition_ids": list(row["heldout_transition_ids"]),
            "baseline_transition_ids": list(row["heldout_transition_ids"]),
            "fit_transition_count": 2,
            "heldout_transition_count": len(row["heldout_transition_ids"]),
            "latent_delta_count": 2,
            "accepted_delta_count": 2,
            "live_path_methods_called": [
                "ActionPrefixLatentAdapter",
                "arc_executable_world_model.load_engine",
            ],
        }
        for game, row in source["per_game_value_gap"].items()
    }
    return {
        "experiment_id": 4893,
        "honest_verdict": "complete_action_prefix_latent_no_value_lift_representation_invariant_hard",
        "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_HARD",
        "generator_backend": "gpu0_cuda",
        "duration_s": 75.0,
        "flagged_adversarial": False,
        "ran_genuinely_live": True,
        "delta_on_truly_heldout_split": True,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "inference_substrate": "live_llm_inference",
        "per_game_value_gap": rows,
        "n_games_measured": len(rows),
        "action_prefix_value_accuracy_delta_median": 0.0,
        "action_prefix_value_accuracy_delta_ci95": [0.0, 0.0],
        "model_specs": {"backend": "gpu0_cuda", "name": "Qwen3.5-9B-MTP"},
    }


def _tool_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {"returncode": 0, "stdout": "LIVE re-check: clean", "stderr": ""},
        {"loaded": True, "flag_count": 0, "flags": []},
        {
            "loaded": True,
            "flag_count": 1,
            "flags": [{"kind": "IMPLAUSIBLE_PERFECT", "severity": "INFO"}],
        },
        {"passed": True, "returncode": 0, "stdout_tail": "OK", "stderr_tail": ""},
    )


def test_req_arc_wmte_4897_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4897: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4897",
        "SCENARIO-ARC-WMTE-4897-A1-A1B-AUDIT",
        "SCENARIO-ARC-WMTE-4897-A1B-LIVE-OR-GATE-SKIPPED",
        "SCENARIO-ARC-WMTE-4897-BLOCKED-A1-ARTIFACT",
        mod.RESULT_RELATIVE_PATH,
        mod.AUDIT_REPORT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4897_good_a1_and_a1b_pass_gates() -> None:
    """SCENARIO-ARC-WMTE-4897-A1-A1B-AUDIT: all load-bearing gates pass."""

    summary, a1_adv, a1b_adv, lint = _tool_results()
    a1 = _a1_artifact()
    audit = mod.audit_sources(
        a1_artifact=a1,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(a1),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )

    assert audit["honest_verdict"] == "complete_a1_a1b_audited"
    assert audit["a1_genuinely_diagnostic"] is True
    assert audit["a1_positive_control_non_degenerate_confirmed"] is True
    assert audit["a1_delta_on_heldout_disjoint_confirmed"] is True
    assert audit["planner_blind_confirmed"] is True
    assert audit["numbers_match_fork"] is True
    assert audit["a1b_ran_genuinely_live"] is True
    assert audit["a1_failure_reasons"] == []
    assert audit["a1b_failure_reasons"] == []
    assert audit["checks"]["a1_numbers_match_fork"]["computed_fork_verdict"] == (
        "VALUE_GAP_REPRESENTATION_INVARIANT"
    )
    assert audit["checks"]["a1b_live_and_split"]["duration_too_short_flagged"] is False


def test_scenario_arc_wmte_4897_hostile_variants_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-4897-A1-A1B-AUDIT: adversarial variants are explicit non-tests."""

    summary, a1_adv, a1b_adv, lint = _tool_results()
    cases: list[tuple[dict[str, Any], str, dict[str, Any] | None, dict[str, Any], str]] = []

    too_short = _a1_artifact()
    too_short["duration_s"] = 59.0
    too_short["generator_backend"] = "cpu"
    cases.append((too_short, _a1_source(), _a1b_artifact(too_short), a1b_adv, "a1_not_live_on_gpu0"))

    degenerate = _a1_artifact()
    degenerate["positive_control_non_degenerate"] = False
    degenerate["positive_control_value_gap"]["cell_recall"] = 0.0
    cases.append((degenerate, _a1_source(), _a1b_artifact(degenerate), a1b_adv, "a1_positive_control_degenerate"))

    overlapping = _a1_artifact()
    overlapping["per_game_value_gap"]["cd82"]["author_transition_ids"] = ["heldout:0"]
    overlapping["delta_on_truly_heldout_split"] = False
    cases.append((overlapping, _a1_source(), _a1b_artifact(overlapping), a1b_adv, "a1_delta_split_not_disjoint"))

    wrong_fork = _a1_artifact()
    wrong_fork["fork_verdict"] = "PLANNER_GAP"
    cases.append((wrong_fork, _a1_source(), _a1b_artifact(wrong_fork), a1b_adv, "fork_verdict_mismatch"))

    injected = _a1_artifact()
    cases.append((injected, _a1_source(injected=True), _a1b_artifact(injected), a1b_adv, "banked_answer_used_before_classification"))

    unfair_a1b = _a1b_artifact()
    unfair_a1b["per_game_value_gap"]["cd82"]["heldout_transition_ids"] = ["other:0"]
    cases.append((_a1_artifact(), _a1_source(), unfair_a1b, a1b_adv, "a1b_not_same_heldout_split_as_a1"))

    oracle_a1b = _a1b_artifact()
    oracle_a1b["verifier_is_oracle"] = True
    cases.append((_a1_artifact(), _a1_source(), oracle_a1b, a1b_adv, "a1b_verifier_is_oracle"))

    short_a1b = _a1b_artifact()
    short_a1b["duration_s"] = 13.7
    short_a1b["ran_genuinely_live"] = False
    cases.append(
        (
            _a1_artifact(),
            _a1_source(),
            short_a1b,
            {"loaded": True, "flag_count": 1, "flags": [{"kind": "DURATION_TOO_SHORT"}]},
            "a1b_duration_too_short_flagged",
        )
    )

    for a1, source, a1b, a1b_result, reason in cases:
        audit = mod.audit_sources(
            a1_artifact=a1,
            a1_source_text=source,
            a1_summarizer_result=summary,
            a1_adversarial_result=a1_adv,
            a1b_artifact=a1b,
            a1b_adversarial_result=a1b_result,
            live_lint_result=lint,
        )
        all_reasons = audit["a1_failure_reasons"] + audit["a1b_failure_reasons"]
        assert reason in all_reasons
        if reason.startswith("a1b_"):
            assert audit["a1b_ran_genuinely_live"] is False
        else:
            assert audit["a1_genuinely_diagnostic"] is False

    flagged = mod.audit_sources(
        a1_artifact={**_a1_artifact(), "verifier_is_oracle": True},
        a1_source_text="def measure_game_with_decision_need_targets(",
        a1_summarizer_result={"returncode": 2},
        a1_adversarial_result={"loaded": True, "flag_count": 1, "flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}]},
        a1b_artifact=None,
        a1b_adversarial_result=None,
        live_lint_result={"passed": False},
    )
    for reason in (
        "a1_source_not_parseable",
        "a1_summarizer_failed",
        "a1_adversarial_verify_flagged",
        "a1_verifier_is_oracle",
        "a1_circular_moat_overclaim",
        "live_path_unreachable",
    ):
        assert reason in flagged["a1_failure_reasons"]


def test_req_arc_wmte_4897_a1b_gate_skip_and_missing_artifact() -> None:
    """SCENARIO-ARC-WMTE-4897-A1B-LIVE-OR-GATE-SKIPPED: skipped A1b is explicit."""

    summary, a1_adv, _a1b_adv, lint = _tool_results()
    closed = _a1_artifact(fork="REPRESENTATION_UNLOCKS_VALUE")

    skipped = mod.audit_sources(
        a1_artifact=closed,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=None,
        a1b_adversarial_result=None,
        live_lint_result=lint,
    )
    missing = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=None,
        a1b_adversarial_result=None,
        live_lint_result=lint,
    )

    assert skipped["a1b_ran_genuinely_live"] is True
    assert skipped["checks"]["a1b_live_and_split"]["status"] == "gate_skipped"
    assert missing["a1b_ran_genuinely_live"] is False
    assert "a1b_artifact_missing_after_low_value_a1" in missing["a1b_failure_reasons"]


def test_req_arc_wmte_4897_build_schema_write_and_report(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4897: artifacts are checksum-stable and report appends are idempotent."""

    a1_path = tmp_path / mod.A1_ARTIFACT_RELATIVE_PATH
    script_path = tmp_path / mod.A1_SCRIPT_RELATIVE_PATH
    a1b_path = tmp_path / mod.A1B_ARTIFACT_RELATIVE_PATH
    a1_path.parent.mkdir(parents=True)
    script_path.parent.mkdir(parents=True)
    a1b_path.parent.mkdir(parents=True, exist_ok=True)
    a1_path.write_text(json.dumps(_a1_artifact()), encoding="utf-8")
    script_path.write_text(_a1_source(), encoding="utf-8")
    a1b_path.write_text(json.dumps(_a1b_artifact()), encoding="utf-8")

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
    assert artifact["a1_artifact_checksum"] == mod.file_checksum(a1_path)
    assert artifact["a1_script_checksum"] == mod.file_checksum(script_path)
    assert artifact["a1b_artifact_checksum"] == mod.file_checksum(a1b_path)

    result_path = mod.write_artifact(artifact, root=tmp_path)
    report_path = mod.append_markdown_report(artifact, root=tmp_path)
    mod.append_markdown_report(artifact, root=tmp_path)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert report_path.read_text(encoding="utf-8").count("## Experiment 4897 A1/A1b Audit") == 1

    blocked = mod.blocked_artifact({"ok": False}, duration_s=0.0)
    assert blocked["honest_verdict"] == "blocked_a1_artifact_missing"
    assert mod.artifact_schema_errors(blocked) == []

    broken = dict(artifact)
    broken.update(
        {
            "field_principles": {},
            "honest_verdict": "bad",
            "a1_genuinely_diagnostic": "yes",
            "a1_positive_control_non_degenerate_confirmed": "yes",
            "a1_delta_on_heldout_disjoint_confirmed": "yes",
            "planner_blind_confirmed": "yes",
            "numbers_match_fork": "yes",
            "a1b_ran_genuinely_live": "yes",
            "inference_substrate": "live_llm_inference",
            "checks": [],
            "a1_failure_reasons": "none",
            "a1b_failure_reasons": "none",
            "random_seed": 0,
            "duration_s": 0.0,
            "reproducibility_checksum": "sha256:bad",
        }
    )
    errors = mod.artifact_schema_errors(broken)
    for expected in (
        "field_principles_mismatch",
        "honest_verdict_missing_terminal_prefix",
        "a1_genuinely_diagnostic_must_be_bool",
        "a1_positive_control_non_degenerate_confirmed_must_be_bool",
        "a1_delta_on_heldout_disjoint_confirmed_must_be_bool",
        "planner_blind_confirmed_must_be_bool",
        "numbers_match_fork_must_be_bool",
        "a1b_ran_genuinely_live_must_be_bool",
        "inference_substrate_mismatch",
        "checks_must_be_dict",
        "a1_failure_reasons_must_be_list",
        "a1b_failure_reasons_must_be_list",
        "random_seed_mismatch",
        "duration_below_aggregation_floor",
        "reproducibility_checksum_mismatch",
    ):
        assert expected in errors
    with pytest.raises(ValueError):
        mod.write_artifact(broken, root=tmp_path)


def test_req_arc_wmte_4897_run_paths(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-ARC-WMTE-4897: run writes blocked and complete artifacts without fabrication."""

    blocked = mod.run(root=tmp_path, write=True, now=iter([1.0, 1.1]).__next__)
    assert blocked["honest_verdict"] == "blocked_a1_artifact_missing"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    for path, content in (
        (tmp_path / mod.A1_ARTIFACT_RELATIVE_PATH, json.dumps(_a1_artifact())),
        (tmp_path / mod.A1_SCRIPT_RELATIVE_PATH, _a1_source()),
        (tmp_path / mod.A1B_ARTIFACT_RELATIVE_PATH, json.dumps(_a1b_artifact())),
        (tmp_path / mod.SPEC_RELATIVE_PATH, "REQ-ARC-WMTE-4897"),
        (tmp_path / "scripts/summarize_artifact.py", ""),
        (tmp_path / "scripts/adversarial_verify.py", ""),
        (tmp_path / "scripts/arc_orphan_solver_lint.py", ""),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    summary, a1_adv, a1b_adv, lint = _tool_results()
    monkeypatch.setattr(mod, "run_summarizer", lambda _path: summary)
    monkeypatch.setattr(mod, "run_adversarial_verify", lambda _path: a1_adv if "4892" in str(_path) else a1b_adv)
    monkeypatch.setattr(mod, "run_arc_orphan_solver_lint", lambda _root: lint)

    complete = mod.run(root=tmp_path, write=True, now=iter([2.0, 2.1]).__next__)
    assert complete["honest_verdict"] == "complete_a1_a1b_audited"
    assert complete["a1_genuinely_diagnostic"] is True
    assert complete["a1b_ran_genuinely_live"] is True


def test_req_arc_wmte_4897_checked_in_artifacts_match_audit() -> None:
    """REQ-ARC-WMTE-4897: checked-in A1/A1b artifacts produce the requested audit."""

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_a1_a1b_audited"
    assert artifact["a1_genuinely_diagnostic"] is True
    assert artifact["a1_positive_control_non_degenerate_confirmed"] is True
    assert artifact["a1_delta_on_heldout_disjoint_confirmed"] is True
    assert artifact["planner_blind_confirmed"] is True
    assert artifact["numbers_match_fork"] is True
    assert artifact["a1b_ran_genuinely_live"] is True
    assert artifact["checks"]["a1_live_gpu"]["generator_backend"] == "gpu0_cuda"
    assert artifact["checks"]["a1b_live_and_split"]["duration_too_short_flagged"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4897_defensive_branches(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-ARC-WMTE-4897: malformed inputs fail closed without fabricated passes."""

    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None
    non_object = tmp_path / "not_object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._read_json(non_object)

    assert mod._full_call_name(ast.Constant(value=1)) == ""
    assert mod._call_name(None) == ""
    assert mod._call_name(ast.Name(id="f")) == "f"
    assert mod._call_name(ast.Attribute(value=ast.Name(id="obj"), attr="method")) == "method"
    assert mod._call_name(ast.Constant(value=1)) == ""
    assert mod._find_function(ast.parse("x = 1"), "missing") is None
    assert mod._first_parent_call(ast.Name(id="winning_prefix")) is None
    expr = ast.parse("winning_prefix + 1")
    mod._attach_ast_parents(expr)
    assert mod._first_parent_call(next(n for n in ast.walk(expr) if isinstance(n, ast.Name))) is None

    from scripts import adversarial_verify

    class _Proc:
        returncode = 1
        stdout = "flagged"
        stderr = "stderr"

    monkeypatch.setattr(
        adversarial_verify,
        "verify_artifact",
        lambda _path: (_ for _ in ()).throw(NameError("broken direct verifier")),
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: _Proc())
    fallback = mod.run_adversarial_verify(Path("artifact.json"))
    assert fallback["flag_count"] == 1
    assert "broken direct verifier" in fallback["fallback_error"]
    assert fallback["stdout_tail"] == "flagged"

    missing_fn = mod.audit_sources(
        a1_artifact={**_a1_artifact(), "planner_blind_to_banked_answer": False},
        a1_source_text="def not_the_path():\n    return None\n",
        a1_summarizer_result={"returncode": 0},
        a1_adversarial_result={"loaded": True, "flag_count": 0, "flags": []},
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result={"loaded": True, "flag_count": 0, "flags": []},
        live_lint_result={"passed": True},
    )
    assert "artifact_planner_blind_flag_false" in missing_fn["a1_failure_reasons"]
    assert "measure_game_with_decision_need_targets_missing" in missing_fn["a1_failure_reasons"]

    missing_calls = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text="def measure_game_with_decision_need_targets(game, winning_prefix, proposer):\n    return []\n",
        a1_summarizer_result={"returncode": 0},
        a1_adversarial_result={"loaded": True, "flag_count": 0, "flags": []},
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result={"loaded": True, "flag_count": 0, "flags": []},
        live_lint_result={"passed": True},
    )
    for reason in (
        "winning_prefix_not_used_for_classification",
        "decision_need_target_table_not_authored",
        "plan_in_model_path_not_called",
        "classification_path_not_called",
    ):
        assert reason in missing_calls["a1_failure_reasons"]

    backend_from_model = _a1_artifact()
    backend_from_model.pop("generator_backend")
    assert mod._generator_backend(backend_from_model) == "gpu0_cuda"
    backend_from_preconditions = _a1_artifact()
    backend_from_preconditions.pop("generator_backend")
    backend_from_preconditions.pop("model_specs")
    backend_from_preconditions["preconditions_checked"] = {"generator": {"backend": "igpu_hip"}}
    assert mod._generator_backend(backend_from_preconditions) == "igpu_hip"
    assert mod._generator_backend({"preconditions_checked": {"generator": {"backend": "cpu"}}}) is None

    bad_live = _a1_artifact()
    bad_live["flagged_adversarial"] = True
    bad_live["inference_substrate"] = "aggregation_from_upstream_artifacts"
    assert "a1_flagged_adversarial" in mod._a1_live_gpu_check(bad_live)[1]
    assert "a1_inference_substrate_not_live_llm" in mod._a1_live_gpu_check(bad_live)[1]
    cpu_live = _a1_artifact()
    cpu_live["generator_backend"] = "cpu"
    cpu_live.pop("model_specs")
    cpu_live.pop("preconditions_checked", None)
    assert "a1_generator_backend_not_gpu0_or_igpu" in mod._a1_live_gpu_check(cpu_live)[1]

    assert mod._positive_control_recall({"cell_recall_decision_need": 0.3}) == 0.3
    assert mod._positive_control_recall({"cell_recall_baseline": 0.4}) == 0.4
    assert mod._positive_control_recall({}) is None
    missing_control = _a1_artifact()
    missing_control["positive_control_game"] = "not_tu93"
    missing_control["positive_control_value_gap"] = {}
    pc_reasons = mod._positive_control_check(missing_control)[1]
    assert "a1_positive_control_not_tu93" in pc_reasons
    assert "a1_positive_control_row_missing" in pc_reasons

    malformed_numbers = _a1_artifact()
    malformed_numbers["per_game_value_gap"] = {
        "bad1": {
            "planned_bucket": "BAD",
            "value_acc_code_baseline": "x",
            "value_acc_decision_need": None,
            "value_delta": None,
            "cell_recall": None,
            "migrated": "yes",
        },
        "bad2": {
            "planned_bucket": "COVERED",
            "value_acc_code_baseline": 0.1,
            "value_acc_decision_need": 0.2,
            "value_delta": 0.0,
            "cell_recall": 0.1,
            "migrated": False,
        },
        "bad3": {
            "planned_bucket": "ENUMERATED_BUT_LOST",
            "value_acc_code_baseline": 0.1,
            "value_acc_decision_need": 0.1,
            "value_delta": 0.0,
            "cell_recall": 0.1,
            "migrated": False,
        },
    }
    malformed_numbers.update(
        {
            "n_games_measured": "bad",
            "decision_need_value_accuracy_delta_median": 9.0,
            "decision_need_value_accuracy_delta_ci95": [9.0, 9.0],
            "engine_cell_recall_median": 9.0,
            "coverage_migration_count": 9,
            "fork_verdict": "NOPE",
        }
    )
    number_reasons = mod._numbers_match_fork_check(malformed_numbers)[1]
    for reason in (
        "invalid_planned_bucket",
        "invalid_value_metric",
        "value_delta_mismatch",
        "invalid_cell_recall",
        "invalid_migrated_flag",
        "row_migrated_mismatch",
        "never_enumerated_games_below_3",
        "n_games_measured_not_integer",
        "n_games_measured_mismatch",
        "decision_need_delta_median_mismatch",
        "decision_need_delta_ci95_mismatch",
        "engine_cell_recall_median_mismatch",
        "coverage_migration_count_mismatch",
        "invalid_fork_verdict",
    ):
        assert reason in number_reasons
    two_game_numbers = _a1_artifact()
    two_game_numbers["per_game_value_gap"].pop("ls20")
    two_game_numbers["n_games_measured"] = 2
    assert "n_games_measured_below_3" in mod._numbers_match_fork_check(two_game_numbers)[1]

    assert mod._computed_fork_verdict({}, {}, []) is None
    planner_gap_rows = {
        "a": _a1_row("a", baseline=0.0, decision_need=0.5),
        "b": _a1_row("b", baseline=0.0, decision_need=0.5),
        "c": _a1_row("c", baseline=0.0, decision_need=0.5),
    }
    assert (
        mod._computed_fork_verdict(planner_gap_rows, _a1_artifact()["positive_control_value_gap"], [0.5, 0.5])
        == "PLANNER_GAP"
    )
    assert "solve_provenance_not_development_proxy" in mod._live_path_check(
        {**_a1_artifact(), "solve_provenance": "other"},
        {"passed": True},
        {"passed": True},
        [],
    )[1]

    a1b_bad = _a1b_artifact()
    a1b_bad["per_game_value_gap"]["cd82"]["baseline_transition_ids"] = ["other:0"]
    a1b_bad["per_game_value_gap"]["cd82"]["fit_transition_ids"] = ["heldout:0"]
    a1b_bad["per_game_value_gap"]["cd82"]["value_acc_latent"] = 0.5
    a1b_bad["per_game_value_gap"]["cd82"]["live_path_methods_called"] = []
    a1b_bad["live_path_reachable"] = False
    a1b_bad["solve_provenance"] = "live_agent_self_discovery"
    a1b_bad["flagged_adversarial"] = True
    a1b_reasons = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result={"returncode": 0},
        a1_adversarial_result={"loaded": True, "flag_count": 0, "flags": []},
        a1b_artifact=a1b_bad,
        a1b_adversarial_result={
            "loaded": True,
            "flag_count": 1,
            "flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}],
        },
        live_lint_result={"passed": False},
    )["a1b_failure_reasons"]
    for reason in (
        "a1b_not_same_heldout_split_as_a1",
        "a1b_delta_vs_baseline_mismatch",
        "a1b_circular_moat_overclaim",
        "a1b_live_path_unreachable",
        "a1b_live_path_methods_missing",
        "a1b_solve_provenance_not_development_proxy",
        "a1b_flagged_adversarial_stamp",
    ):
        assert reason in a1b_reasons

    a1b_missing_row = _a1b_artifact()
    a1b_missing_row["per_game_value_gap"].pop("cd82")
    assert "a1b_not_same_heldout_split_as_a1" in mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result={"returncode": 0},
        a1_adversarial_result={"loaded": True, "flag_count": 0, "flags": []},
        a1b_artifact=a1b_missing_row,
        a1b_adversarial_result={"loaded": True, "flag_count": 0, "flags": []},
        live_lint_result={"passed": True},
    )["a1b_failure_reasons"]

    artifact = mod.blocked_artifact({"ok": False}, duration_s=1.0)
    artifact["a1_genuinely_diagnostic"] = True
    artifact["a1_failure_reasons"] = ["x"]
    artifact["a1b_ran_genuinely_live"] = True
    artifact["a1b_failure_reasons"] = ["y"]
    errors = mod.artifact_schema_errors(artifact)
    assert "diagnostic_artifact_has_a1_failure_reasons" in errors
    assert "genuine_a1b_has_failure_reasons" in errors
