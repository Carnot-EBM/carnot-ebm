"""Tests for Exp 4887 value-gap and inducer-ceiling adversarial audit.

Spec refs: REQ-ARC-WMTE-4887,
SCENARIO-ARC-WMTE-4887-A1-A1B-AUDIT,
SCENARIO-ARC-WMTE-4887-BLOCKED-A1-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4887_value_gap_inducer_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _a1_row(
    game: str,
    *,
    baseline: float,
    adapted: float,
    bucket: str = "NEVER_ENUMERATED",
    recall: float = 0.5,
    fit_ids: list[str] | None = None,
    heldout_ids: list[str] | None = None,
) -> dict[str, Any]:
    delta = round(adapted - baseline, 6)
    return {
        "game": game,
        "cell_recall_baseline": recall,
        "cell_recall_adapted": recall,
        "value_acc_baseline": round(baseline, 6),
        "value_acc_adapted": round(adapted, 6),
        "value_delta": delta,
        "planned_bucket": bucket,
        "migrated": bucket == "COVERED",
        "fit_transition_ids": fit_ids if fit_ids is not None else ["fit:0", "fit:1"],
        "remeasure_transition_ids": (
            heldout_ids if heldout_ids is not None else ["heldout:0", "heldout:1"]
        ),
        "baseline_transition_ids": ["baseline:0", "baseline:1"],
        "adapter_fit_transition_count": 2,
        "heldout_transition_count": 2,
        "cold_transition_count": 2,
        "live_path_methods_called": [
            "arc_live_ttt.LiveTTTWorldModel",
            "DynamicsValueAdapter",
            "arc_executable_world_model.load_engine",
            "arc_executable_world_model.plan_in_model",
        ],
    }


def _a1_artifact() -> dict[str, Any]:
    rows = {
        "cd82": _a1_row("cd82", baseline=0.2, adapted=0.2),
        "cn04": _a1_row("cn04", baseline=0.6, adapted=0.5),
        "ls20": _a1_row("ls20", baseline=0.3, adapted=0.4),
    }
    return {
        "experiment_id": 4882,
        "honest_verdict": "complete_ttt_dynamics_no_value_lift_INDUCER_CEILING_HARD",
        "fork_verdict": "INDUCER_CEILING_HARD",
        "generator_backend": "gpu0_cuda",
        "duration_s": 65.0,
        "flagged_adversarial": False,
        "inference_substrate": "live_llm_inference",
        "per_game_value_gap": rows,
        "positive_control_game": "tu93",
        "positive_control_non_degenerate": True,
        "positive_control_value_gap": _a1_row(
            "tu93", baseline=0.2, adapted=0.3, bucket="COVERED", recall=0.2
        ),
        "delta_on_truly_heldout_split": True,
        "planner_blind_to_banked_answer": True,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "solve_provenance": "development_proxy",
        "n_games_measured": len(rows),
        "coverage_migration_count": 0,
        "engine_cell_recall_median": 0.5,
        "tta_changed_cell_value_accuracy_delta_median": 0.0,
        "tta_value_accuracy_delta_ci95": [-0.1, 0.1],
        "preconditions_checked": {
            "generator": {
                "ok": True,
                "generator_backend": "gpu0_cuda",
                "backend": "gpu0_cuda",
                "launch_env_cuda_visible_devices": "0",
            },
            "live_path": {"ok": True},
            "positive_control": {"game": "tu93", "non_degenerate": True},
        },
        "model_specs": {"backend": "gpu0_cuda", "name": "Qwen3.5-9B-MTP"},
    }


def _a1_source(*, injected: bool = False) -> str:
    if injected:
        return """
def measure_game_with_ttt_dynamics_adaptation(game, winning_prefix, proposer):
    cold_transitions = a1._collect_cold_policy_transitions(game=game, proposer=proposer)
    adapter = DynamicsValueAdapter.fit(winning_prefix)
    planned = _plan_with_adapted_engine(game=game, engine=adapter, hint=winning_prefix)
    return a1.classify_planned_pool(game, winning_prefix, planned)
"""
    return """
def measure_game_with_ttt_dynamics_adaptation(game, winning_prefix, proposer):
    cold_transitions = a1._collect_cold_policy_transitions(game=game, proposer=proposer)
    adapter = DynamicsValueAdapter.fit(cold_transitions)
    planned = _plan_with_adapted_engine(game=game, engine=adapter)
    return a1.classify_planned_pool(game, winning_prefix, planned)
"""


def _a1b_row(
    game: str,
    *,
    lane: str,
    a1_row: dict[str, Any],
    heldout_ids: list[str] | None = None,
) -> dict[str, Any]:
    ids = heldout_ids if heldout_ids is not None else list(a1_row["remeasure_transition_ids"])
    return {
        "game": game,
        "lane": lane,
        "value_acc": a1_row["value_acc_baseline"],
        "cell_recall": a1_row["cell_recall_baseline"],
        "delta_vs_baseline": 0.0,
        "ci95": [0.0, 0.0],
        "a1_baseline_value_acc": a1_row["value_acc_baseline"],
        "a1_heldout_transition_ids": list(a1_row["remeasure_transition_ids"]),
        "heldout_transition_ids": ids,
        "fit_transition_ids": ["fit:0", "fit:1"],
        "heldout_transition_count": len(ids),
        "live_path_methods_called": ["arc_executable_world_model.load_engine"],
    }


def _a1b_artifact(a1: dict[str, Any] | None = None) -> dict[str, Any]:
    source = a1 or _a1_artifact()
    rows = source["per_game_value_gap"]
    return {
        "experiment_id": 4883,
        "honest_verdict": "complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling",
        "inducer_ceiling_attribution": "METHOD_IS_CEILING",
        "per_lane_per_game": {
            "reference": {
                game: _a1b_row(game, lane="reference", a1_row=row)
                for game, row in rows.items()
            },
            "local": {
                game: _a1b_row(game, lane="local", a1_row=row)
                for game, row in rows.items()
            },
        },
        "delta_on_truly_heldout_split": True,
        "verifier_is_oracle": False,
        "live_path_reachable": True,
        "reference_lane_is_ceiling_only": True,
        "solve_provenance": "development_proxy",
        "flagged_adversarial": False,
        "duration_s": 65.0,
    }


def _tool_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {"returncode": 0, "stdout": "LIVE re-check: clean", "stderr": ""},
        {"loaded": True, "flag_count": 0, "flags": []},
        {"loaded": True, "flag_count": 0, "flags": []},
        {"passed": True, "returncode": 0, "stdout_tail": "OK", "stderr_tail": ""},
    )


def test_req_arc_wmte_4887_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4887: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4887",
        "SCENARIO-ARC-WMTE-4887-A1-A1B-AUDIT",
        "SCENARIO-ARC-WMTE-4887-BLOCKED-A1-ARTIFACT",
        mod.RESULT_RELATIVE_PATH,
        mod.AUDIT_REPORT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4887_good_a1_and_a1b_pass_gates() -> None:
    """SCENARIO-ARC-WMTE-4887-A1-A1B-AUDIT: all load-bearing gates pass."""

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
    assert audit["a1b_ab_trustworthy"] is True
    assert audit["a1_failure_reasons"] == []
    assert audit["a1b_failure_reasons"] == []
    assert audit["checks"]["a1_numbers_match_fork"]["computed_fork_verdict"] == "INDUCER_CEILING_HARD"
    assert audit["checks"]["a1b_ab_fairness"]["same_split_as_a1"] is True


def test_scenario_arc_wmte_4887_hostile_variants_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-4887-A1-A1B-AUDIT: adversarial variants are explicit non-tests."""

    summary, a1_adv, a1b_adv, lint = _tool_results()
    cases: list[tuple[dict[str, Any], str, dict[str, Any] | None, dict[str, Any], str]] = []

    too_short = _a1_artifact()
    too_short["duration_s"] = 59.0
    too_short["generator_backend"] = "cpu"
    cases.append((too_short, _a1_source(), _a1b_artifact(too_short), a1b_adv, "a1_not_live_on_gpu0"))

    degenerate = _a1_artifact()
    degenerate["positive_control_non_degenerate"] = False
    degenerate["positive_control_value_gap"]["cell_recall_baseline"] = 0.0
    cases.append((degenerate, _a1_source(), _a1b_artifact(degenerate), a1b_adv, "a1_positive_control_degenerate"))

    overlapping = _a1_artifact()
    overlapping["per_game_value_gap"]["cd82"]["remeasure_transition_ids"] = ["fit:0"]
    overlapping["delta_on_truly_heldout_split"] = False
    cases.append((overlapping, _a1_source(), _a1b_artifact(overlapping), a1b_adv, "a1_delta_split_not_disjoint"))

    wrong_fork = _a1_artifact()
    wrong_fork["fork_verdict"] = "PLANNER_GAP"
    cases.append((wrong_fork, _a1_source(), _a1b_artifact(wrong_fork), a1b_adv, "fork_verdict_mismatch"))

    unfair_a1b = _a1b_artifact()
    unfair_a1b["per_lane_per_game"]["local"]["cd82"]["heldout_transition_ids"] = ["other:0"]
    cases.append((_a1_artifact(), _a1_source(), unfair_a1b, a1b_adv, "a1b_not_same_heldout_split_as_a1"))

    oracle_a1b = _a1b_artifact()
    oracle_a1b["verifier_is_oracle"] = True
    cases.append((_a1_artifact(), _a1_source(), oracle_a1b, a1b_adv, "a1b_verifier_is_oracle"))

    reference_claim = _a1b_artifact()
    reference_claim["reference_lane_is_ceiling_only"] = False
    cases.append((_a1_artifact(), _a1_source(), reference_claim, a1b_adv, "a1b_reference_lane_not_ceiling_only"))

    flagged_a1b = _a1b_artifact()
    cases.append(
        (
            _a1_artifact(),
            _a1_source(),
            flagged_a1b,
            {"loaded": True, "flag_count": 1, "flags": [{"kind": "DURATION_TOO_SHORT"}]},
            "a1b_adversarial_verify_flagged",
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
            assert audit["a1b_ab_trustworthy"] is False
        else:
            assert audit["a1_genuinely_diagnostic"] is False

    injected = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(injected=True),
        a1_summarizer_result=summary,
        a1_adversarial_result={"loaded": True, "flag_count": 1, "flags": []},
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result={"passed": False},
    )
    for reason in (
        "banked_answer_used_before_classification",
        "a1_adversarial_verify_flagged",
        "live_path_unreachable",
    ):
        assert reason in injected["a1_failure_reasons"]


def test_req_arc_wmte_4887_a1b_gate_skip_and_missing_artifact() -> None:
    """REQ-ARC-WMTE-4887: skipped A1b is explicit; missing low-value A1b is not trusted."""

    summary, a1_adv, _a1b_adv, lint = _tool_results()
    closed = _a1_artifact()
    closed["fork_verdict"] = "PLANNER_GAP"
    closed["tta_value_accuracy_delta_ci95"] = [0.2, 0.3]
    closed["per_game_value_gap"] = {
        "cd82": _a1_row("cd82", baseline=0.0, adapted=0.25),
        "cn04": _a1_row("cn04", baseline=0.0, adapted=0.25),
        "ls20": _a1_row("ls20", baseline=0.0, adapted=0.25),
    }
    closed["tta_changed_cell_value_accuracy_delta_median"] = 0.25

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

    assert skipped["a1b_ab_trustworthy"] is True
    assert skipped["checks"]["a1b_ab_fairness"]["status"] == "gate_skipped"
    assert missing["a1b_ab_trustworthy"] is False
    assert "a1b_artifact_missing_after_low_value_a1" in missing["a1b_failure_reasons"]


def test_req_arc_wmte_4887_build_schema_write_and_report(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4887: artifacts are checksum-stable and report appends are idempotent."""

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
    assert report_path.read_text(encoding="utf-8").count("## Experiment 4887 A1/A1b Audit") == 1

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
            "a1b_ab_trustworthy": "yes",
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
        "a1b_ab_trustworthy_must_be_bool",
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


def test_req_arc_wmte_4887_run_paths(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-ARC-WMTE-4887: run writes blocked and complete artifacts without fabrication."""

    blocked = mod.run(root=tmp_path, write=True, now=iter([1.0, 1.1]).__next__)
    assert blocked["honest_verdict"] == "blocked_a1_artifact_missing"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    for path, content in (
        (tmp_path / mod.A1_ARTIFACT_RELATIVE_PATH, json.dumps(_a1_artifact())),
        (tmp_path / mod.A1_SCRIPT_RELATIVE_PATH, _a1_source()),
        (tmp_path / mod.A1B_ARTIFACT_RELATIVE_PATH, json.dumps(_a1b_artifact())),
        (tmp_path / mod.SPEC_RELATIVE_PATH, "REQ-ARC-WMTE-4887"),
        (tmp_path / "scripts/summarize_artifact.py", ""),
        (tmp_path / "scripts/adversarial_verify.py", ""),
        (tmp_path / "scripts/arc_orphan_solver_lint.py", ""),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    summary, a1_adv, a1b_adv, lint = _tool_results()
    monkeypatch.setattr(mod, "run_summarizer", lambda _path: summary)
    monkeypatch.setattr(mod, "run_adversarial_verify", lambda _path: a1_adv if "4882" in str(_path) else a1b_adv)
    monkeypatch.setattr(mod, "run_arc_orphan_solver_lint", lambda _root: lint)

    complete = mod.run(root=tmp_path, write=True, now=iter([2.0, 2.1]).__next__)
    assert complete["honest_verdict"] == "complete_a1_a1b_audited"
    assert complete["a1_genuinely_diagnostic"] is True
    assert complete["a1b_ab_trustworthy"] is True


def test_req_arc_wmte_4887_checked_in_artifacts_match_audit() -> None:
    """REQ-ARC-WMTE-4887: checked-in A1/A1b artifacts produce the requested audit."""

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "complete_a1_a1b_audited"
    assert artifact["a1_genuinely_diagnostic"] is True
    assert artifact["a1_positive_control_non_degenerate_confirmed"] is True
    assert artifact["a1_delta_on_heldout_disjoint_confirmed"] is True
    assert artifact["planner_blind_confirmed"] is True
    assert artifact["numbers_match_fork"] is True
    assert artifact["checks"]["a1_live_gpu"]["generator_backend"] == "gpu0_cuda"
    assert artifact["a1b_ab_trustworthy"] is False
    assert "a1b_adversarial_verify_flagged" in artifact["a1b_failure_reasons"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_4887_defensive_branches(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-ARC-WMTE-4887: malformed inputs fail closed without fabricated passes."""

    summary, a1_adv, a1b_adv, lint = _tool_results()

    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None
    non_object = tmp_path / "not_object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        mod._read_json(non_object)

    ast = __import__("ast")
    assert mod._full_call_name(ast.Constant(value=1)) == ""
    assert mod._call_name(None) == ""
    assert mod._call_name(ast.Constant(value=1)) == ""
    assert mod._find_function(ast.parse("x = 1"), "missing") is None
    assert mod._first_parent_call(ast.Name(id="winning_prefix")) is None
    assert mod._positive_control_recall({"cell_recall_baseline": 0.1}) == 0.1
    assert mod._bootstrap_ci95([], iterations=1, seed=1) == [None, None]

    from scripts import adversarial_verify

    class _Proc:
        returncode = 1
        stdout = "flagged"
        stderr = ""

    monkeypatch.setattr(
        adversarial_verify,
        "verify_artifact",
        lambda _path: (_ for _ in ()).throw(NameError("broken direct verifier")),
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: _Proc())
    fallback = mod.run_adversarial_verify(Path("artifact.json"))
    assert fallback["flag_count"] == 1
    assert "broken direct verifier" in fallback["fallback_error"]

    syntax_a1 = _a1_artifact()
    syntax_a1["planner_blind_to_banked_answer"] = False
    syntax_a1["verifier_is_oracle"] = True
    syntax_a1["live_path_reachable"] = False
    syntax_a1["solve_provenance"] = "live_agent_self_discovery"
    syntax = mod.audit_sources(
        a1_artifact=syntax_a1,
        a1_source_text="def measure_game_with_ttt_dynamics_adaptation(",
        a1_summarizer_result={"returncode": 2},
        a1_adversarial_result={
            "loaded": True,
            "flag_count": 1,
            "flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}],
        },
        a1b_artifact=None,
        a1b_adversarial_result=None,
        live_lint_result={"passed": False},
    )
    for reason in (
        "artifact_planner_blind_flag_false",
        "a1_source_not_parseable",
        "a1_summarizer_failed",
        "a1_adversarial_verify_flagged",
        "a1_verifier_is_oracle",
        "a1_circular_moat_overclaim",
        "live_path_unreachable",
        "solve_provenance_not_development_proxy",
    ):
        assert reason in syntax["a1_failure_reasons"]

    missing_function = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text="def not_the_path():\n    return None\n",
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )
    assert "measure_game_with_ttt_dynamics_adaptation_missing" in missing_function["a1_failure_reasons"]

    missing_calls = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text="def measure_game_with_ttt_dynamics_adaptation(game, winning_prefix, proposer):\n    return []\n",
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )
    for reason in (
        "winning_prefix_not_used_for_classification",
        "dynamics_value_adapter_not_called",
        "plan_in_model_path_not_called",
        "classification_path_not_called",
    ):
        assert reason in missing_calls["a1_failure_reasons"]

    backend_from_preconditions = _a1_artifact()
    backend_from_preconditions.pop("generator_backend")
    backend_from_preconditions.pop("model_specs")
    backend_from_preconditions["preconditions_checked"]["generator"]["generator_backend"] = "igpu_hip"
    assert mod.audit_sources(
        a1_artifact=backend_from_preconditions,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(backend_from_preconditions),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_ran_live_on_gpu0"] is True

    bad_backend = _a1_artifact()
    bad_backend.pop("model_specs")
    bad_backend.pop("preconditions_checked")
    bad_backend["generator_backend"] = "cpu"
    assert "a1_generator_backend_not_gpu0_or_igpu" in mod.audit_sources(
        a1_artifact=bad_backend,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(bad_backend),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_failure_reasons"]

    flagged_a1 = _a1_artifact()
    flagged_a1["flagged_adversarial"] = True
    assert "a1_flagged_adversarial" in mod.audit_sources(
        a1_artifact=flagged_a1,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(flagged_a1),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_failure_reasons"]

    nested_control = _a1_artifact()
    nested_control["positive_control_value_gap"] = {
        "game": "tu93",
        "adapted_score": {"cell_recall": 0.25},
    }
    assert mod.audit_sources(
        a1_artifact=nested_control,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(nested_control),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_positive_control_non_degenerate_confirmed"] is True

    bad_control = _a1_artifact()
    bad_control["positive_control_game"] = "not_tu93"
    bad_control.pop("positive_control_value_gap")
    control_reasons = mod.audit_sources(
        a1_artifact=bad_control,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=_a1b_artifact(bad_control),
        a1b_adversarial_result=a1b_adv,
        live_lint_result=lint,
    )["a1_failure_reasons"]
    assert "a1_positive_control_not_tu93" in control_reasons
    assert "a1_positive_control_row_missing" in control_reasons

    invalid_numbers = _a1_artifact()
    invalid_numbers["per_game_value_gap"] = {
        "bad": {
            "planned_bucket": "NOPE",
            "value_acc_baseline": 2.0,
            "value_acc_adapted": "x",
            "value_delta": "x",
            "cell_recall_baseline": -1.0,
            "cell_recall_adapted": "x",
            "migrated": "no",
            "fit_transition_ids": [],
            "remeasure_transition_ids": [],
        },
        "bad2": _a1_row("bad2", baseline=0.1, adapted=0.2),
    }
    invalid_numbers["per_game_value_gap"]["bad2"]["value_delta"] = 0.9
    invalid_numbers["per_game_value_gap"]["bad2"]["migrated"] = True
    invalid_numbers["n_games_measured"] = "x"
    invalid_numbers["tta_changed_cell_value_accuracy_delta_median"] = 9.0
    invalid_numbers["tta_value_accuracy_delta_ci95"] = [9.0, 9.0]
    invalid_numbers["engine_cell_recall_median"] = 9.0
    invalid_numbers["coverage_migration_count"] = 99
    invalid_numbers["fork_verdict"] = "UNKNOWN"
    number_reasons = mod.audit_sources(
        a1_artifact=invalid_numbers,
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=None,
        a1b_adversarial_result=None,
        live_lint_result=lint,
    )["a1_failure_reasons"]
    for reason in (
        "invalid_planned_bucket",
        "invalid_value_metric",
        "value_delta_mismatch",
        "invalid_cell_recall",
        "invalid_migrated_flag",
        "row_migrated_mismatch",
        "n_games_measured_below_3",
        "never_enumerated_games_below_3",
        "n_games_measured_not_integer",
        "n_games_measured_mismatch",
        "tta_delta_median_mismatch",
        "tta_delta_ci95_mismatch",
        "engine_cell_recall_median_mismatch",
        "coverage_migration_count_mismatch",
        "invalid_fork_verdict",
    ):
        assert reason in number_reasons

    beatable_rows = {
        "cd82": _a1_row("cd82", baseline=0.0, adapted=0.3, bucket="COVERED"),
        "cn04": _a1_row("cn04", baseline=0.0, adapted=0.3),
        "ls20": _a1_row("ls20", baseline=0.0, adapted=0.3),
    }
    assert (
        mod._computed_fork_verdict(beatable_rows, _a1_artifact()["positive_control_value_gap"], [0.1, 0.3])
        == "INDUCER_CEILING_BEATABLE"
    )
    assert mod._computed_fork_verdict({}, {}, []) is None

    unfair_a1b = _a1b_artifact()
    unfair_a1b["per_lane_per_game"]["reference"]["cd82"]["a1_heldout_transition_ids"] = ["other:0"]
    unfair_a1b["per_lane_per_game"]["local"]["cn04"]["a1_baseline_value_acc"] = 0.0
    unfair_a1b["verifier_is_oracle"] = True
    unfair_a1b["live_path_reachable"] = False
    unfair_a1b["reference_lane_is_ceiling_only"] = False
    unfair_a1b["solve_provenance"] = "live_agent_self_discovery"
    unfair_a1b["flagged_adversarial"] = True
    unfair_a1b["duration_s"] = 1.0
    a1b_reasons = mod.audit_sources(
        a1_artifact=_a1_artifact(),
        a1_source_text=_a1_source(),
        a1_summarizer_result=summary,
        a1_adversarial_result=a1_adv,
        a1b_artifact=unfair_a1b,
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
        "a1b_verifier_is_oracle",
        "a1b_circular_moat_overclaim",
        "a1b_live_path_unreachable",
        "a1b_reference_lane_not_ceiling_only",
        "a1b_solve_provenance_not_development_proxy",
        "a1b_flagged_adversarial_stamp",
        "a1b_adversarial_verify_flagged",
        "a1b_duration_below_live_floor",
    ):
        assert reason in a1b_reasons

    source = tmp_path / mod.A1_ARTIFACT_RELATIVE_PATH
    script = tmp_path / mod.A1_SCRIPT_RELATIVE_PATH
    a1b = tmp_path / mod.A1B_ARTIFACT_RELATIVE_PATH
    for path, content in (
        (source, json.dumps(_a1_artifact())),
        (script, _a1_source()),
        (a1b, json.dumps(_a1b_artifact())),
        (tmp_path / mod.SPEC_RELATIVE_PATH, "REQ-ARC-WMTE-4887"),
        (tmp_path / "scripts/summarize_artifact.py", ""),
        (tmp_path / "scripts/adversarial_verify.py", ""),
        (tmp_path / "scripts/arc_orphan_solver_lint.py", ""),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(mod, "run_summarizer", lambda _path: summary)
    monkeypatch.setattr(mod, "run_adversarial_verify", lambda _path: a1_adv)
    monkeypatch.setattr(mod, "run_arc_orphan_solver_lint", lambda _root: lint)
    original_schema_errors = mod.artifact_schema_errors
    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError):
        mod.run(root=tmp_path, write=False, now=iter([3.0, 3.1]).__next__)
    monkeypatch.setattr(mod, "artifact_schema_errors", original_schema_errors)

    artifact = mod.blocked_artifact({"ok": False}, duration_s=0.0)
    artifact["a1_genuinely_diagnostic"] = True
    artifact["a1_failure_reasons"] = ["still_bad"]
    artifact["a1b_ab_trustworthy"] = True
    artifact["a1b_failure_reasons"] = ["still_bad"]
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    errors = mod.artifact_schema_errors(artifact)
    assert "diagnostic_artifact_has_a1_failure_reasons" in errors
    assert "trustworthy_a1b_has_failure_reasons" in errors
