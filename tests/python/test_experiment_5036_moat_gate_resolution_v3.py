"""Tests for Exp 5036 Phase D5 off-ARC verifier-moat gate resolution v3.

Spec refs: REQ-REPORT-5036, SCENARIO-REPORT-5036.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5036_moat_gate_resolution_v3 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _baseline() -> dict[str, Any]:
    return {
        "experiment": "experiment_5015_genuine_sc_baseline_fix",
        "experiment_id": 5015,
        "honest_verdict": "success_genuine_sc_baseline_fixed",
        "flagged_adversarial": False,
        "genuine_tuned_sc_accuracy": 0.585,
        "genuine_headroom_present": True,
        "oracle_at_k": 0.865,
    }


def _accuracy_arm(
    *,
    experiment: str,
    experiment_id: int,
    accuracy_field: str,
    accuracy: float | None = 0.585,
    delta: float | None = 0.0,
    ci95: list[float] | None = None,
    mcnemar_p: float | None = 1.0,
    flagged: bool = False,
    headroom_present: bool = True,
    verifier_is_oracle: bool = False,
    honest_verdict: str | None = None,
    scorer_trained: bool | None = None,
    abstention_rate: float | None = None,
    n_questions: int = 200,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": experiment,
        "experiment_id": experiment_id,
        "honest_verdict": honest_verdict or ("success_fixture" if delta and delta > 0.0 else "complete_fixture_null"),
        "flagged_adversarial": flagged,
        "verifier_is_oracle": verifier_is_oracle,
        "headroom_present": headroom_present,
        "genuine_tuned_sc_accuracy": 0.585 if accuracy is not None else None,
        accuracy_field: accuracy,
        "delta_vs_tuned_sc": delta,
        "paired_ci95": ci95 if ci95 is not None else [-0.02, 0.02],
        "mcnemar_p": mcnemar_p,
        "oracle_at_k": 0.865,
        "n_questions": n_questions,
    }
    if scorer_trained is not None:
        payload["scorer_trained"] = scorer_trained
    if abstention_rate is not None:
        payload["abstention_rate"] = abstention_rate
    return payload


def _d1(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "experiment": "experiment_5031_lora_ebm_scorer_musr_v3",
        "experiment_id": 5031,
        "accuracy_field": "trained_scorer_accuracy",
        "scorer_trained": True,
    }
    defaults.update(overrides)
    return _accuracy_arm(**defaults)


def _d2(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "experiment": "experiment_5032_uprm_replication_v3",
        "experiment_id": 5032,
        "accuracy_field": "uprm_selection_accuracy",
    }
    defaults.update(overrides)
    return _accuracy_arm(**defaults)


def _d3(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "experiment": "experiment_5033_ebrm_uncertainty_verifier_v3",
        "experiment_id": 5033,
        "accuracy_field": "ebrm_selection_accuracy",
        "abstention_rate": 0.2,
    }
    defaults.update(overrides)
    return _accuracy_arm(**defaults)


def _d4(
    *,
    best_verifier_from: str = "D1",
    second_corpus: str | None = "MMLU-Pro-hard",
    delta: float | None = 0.04,
    ci95: list[float] | None = None,
    mcnemar_p_second: float | None = 0.02,
    flagged: bool = False,
    headroom_present: bool = True,
    honest_verdict: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_5035_moat_second_corpus_v3",
        "experiment_id": 5035,
        "honest_verdict": honest_verdict or ("success_moat_generalizes_fixture" if delta and delta > 0 else "complete_fixture"),
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "headroom_present": headroom_present,
        "best_verifier_from": best_verifier_from,
        "second_corpus": second_corpus,
        "second_corpus_accuracy": 0.625 if delta is not None else None,
        "genuine_tuned_sc_accuracy_second": 0.585 if delta is not None else None,
        "delta_vs_tuned_sc_second": delta,
        "paired_ci95_second": ci95 if ci95 is not None else [0.01, 0.07],
        "mcnemar_p_second": mcnemar_p_second,
        "oracle_at_k_second": 0.81 if delta is not None else None,
        "n_questions": 200 if delta is not None else 0,
    }


def _d6(
    *,
    cascade_accuracy: float | None = None,
    judge_only_accuracy: float | None = None,
    paired_ci95: list[float] | None = None,
    judge_call_fraction: float | None = None,
    flagged: bool = False,
    honest_verdict: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_5034_uncertainty_routed_cascade_v2",
        "experiment_id": 5034,
        "honest_verdict": honest_verdict or ("success_cascade_parity_fixture" if cascade_accuracy is not None else "blocked_judge_server"),
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "cascade_accuracy": cascade_accuracy,
        "judge_only_accuracy": judge_only_accuracy,
        "paired_ci95_cascade_vs_judge": paired_ci95,
        "judge_call_fraction": judge_call_fraction,
        "cascade_judge_calls": None if judge_call_fraction is None else int(judge_call_fraction * 200),
        "judge_only_calls": 200 if judge_call_fraction is not None else 0,
        "n_questions": 200 if judge_call_fraction is not None else 0,
    }


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_required(
    root: Path,
    *,
    d1: dict[str, Any],
    d2: dict[str, Any],
    d3: dict[str, Any],
    d6: dict[str, Any],
    d4: dict[str, Any] | None = None,
) -> None:
    _write_json(root / mod.BASELINE_ARTIFACT_RELATIVE_PATH, _baseline())
    _write_json(root / mod.D1_ARTIFACT_RELATIVE_PATH, d1)
    _write_json(root / mod.D2_ARTIFACT_RELATIVE_PATH, d2)
    _write_json(root / mod.D3_ARTIFACT_RELATIVE_PATH, d3)
    _write_json(root / mod.D6_ARTIFACT_RELATIVE_PATH, d6)
    if d4 is not None:
        _write_json(root / mod.D4_ARTIFACT_RELATIVE_PATH, d4)


def test_req_report_5036_spec_declares_gate_resolution_contract() -> None:
    """REQ-REPORT-5036: OpenSpec anchors the D5 v3 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5036",
        "SCENARIO-REPORT-5036",
        "experiment_5036_moat_gate_resolution_v3.py",
        "results/experiment_5036_moat_gate_resolution_v3.json",
        "flagged_adversarial=true",
        "complete_moat_execution_incomplete_<arm>",
        "complete_moat_retired_bounded_lora_ebm_and_uprm_both_null",
        "diffusiongemma_gate_conditions_satisfied_off_arc",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_5036_current_blocked_d4_and_d6_are_execution_incomplete(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5036-INCOMPLETE: blocked D4/D6 are failed executions, not D3 nulls."""

    d6_path = tmp_path / mod.D6_ARTIFACT_RELATIVE_PATH
    _write_required(
        tmp_path,
        d1=_d1(accuracy=0.665, delta=0.08, ci95=[0.0, 0.165], mcnemar_p=0.076369),
        d2=_d2(accuracy=0.475, delta=-0.11, ci95=[-0.195, -0.03], mcnemar_p=0.016853),
        d3=_d3(accuracy=0.665, delta=0.08, ci95=[0.0, 0.165], mcnemar_p=0.076369, abstention_rate=0.0),
        d6=_d6(),
        d4=_d4(
            best_verifier_from="D3",
            second_corpus=None,
            delta=None,
            ci95=None,
            mcnemar_p_second=None,
            headroom_present=False,
            honest_verdict="blocked_second_corpus_unavailable",
        ),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        write=True,
    )

    assert artifact["honest_verdict"] == "complete_moat_execution_incomplete_cascade"
    assert artifact["decision"] == "EXECUTION-INCOMPLETE"
    assert artifact["moat_realized"] is False
    assert artifact["moat_retired_bounded"] is False
    assert artifact["diffusiongemma_gate_conditions_satisfied_off_arc"] is False
    assert [item["arm_id"] for item in artifact["execution_incomplete_arms"]] == ["D6", "D4"]
    assert artifact["execution_incomplete_arms"][1]["arm"] == "second-corpus-confirmation"
    assert all(item["arm_id"] != "D3" for item in artifact["execution_incomplete_arms"])
    assert artifact["cited_upstream_artifacts"][4]["sha256"] == _sha(d6_path)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5036_accuracy_win_requires_second_corpus_confirmation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5036-POSITIVE: MuSR plus D4 confirmation realizes the accuracy moat."""

    _write_required(
        tmp_path,
        d1=_d1(accuracy=0.645, delta=0.06, ci95=[0.02, 0.1], mcnemar_p=0.01),
        d2=_d2(delta=0.0, ci95=[-0.02, 0.02], mcnemar_p=1.0),
        d3=_d3(delta=0.01, ci95=[-0.01, 0.03], mcnemar_p=0.4),
        d6=_d6(cascade_accuracy=0.7, judge_only_accuracy=0.82, paired_ci95=[-0.2, -0.05], judge_call_fraction=1.0),
        d4=_d4(best_verifier_from="D1", delta=0.05, ci95=[0.015, 0.085], mcnemar_p_second=0.02),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"].startswith("success_moat_realized_off_arc_lora_ebm_musr_")
    assert artifact["decision"] == "POSITIVE"
    assert artifact["moat_realized"] is True
    assert artifact["efficiency_win"] is False
    assert artifact["best_arm"]["arm"] == "LoRA-EBM"
    assert artifact["diffusiongemma_gate_conditions_satisfied_off_arc"] is True
    assert artifact["diffusiongemma_gate_status"] != "MET"
    assert artifact["diffusiongemma_activation"] == "operator_gated_not_flipped"
    assert "ARC's ~13pp headroom remains" in artifact["paper_summary"]
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5036_efficiency_pareto_win_realizes_the_moat(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5036-POSITIVE: judge-parity at fewer calls realizes the efficiency axis."""

    _write_required(
        tmp_path,
        d1=_d1(delta=0.0, ci95=[-0.03, 0.02], mcnemar_p=0.8),
        d2=_d2(delta=0.0, ci95=[-0.02, 0.02], mcnemar_p=1.0),
        d3=_d3(delta=0.0, ci95=[-0.02, 0.02], mcnemar_p=1.0),
        d6=_d6(cascade_accuracy=0.8, judge_only_accuracy=0.81, paired_ci95=[-0.03, 0.01], judge_call_fraction=0.25),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"].startswith("success_moat_realized_off_arc_cascade_musr_efficiency_")
    assert artifact["decision"] == "POSITIVE"
    assert artifact["moat_realized"] is True
    assert artifact["efficiency_win"] is True
    assert artifact["best_arm"]["arm"] == "cascade"
    assert artifact["best_arm"]["judge_call_fraction"] == pytest.approx(0.25)
    assert artifact["moat_retired_bounded"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5036_clean_d1_d2_nulls_bound_retirement(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5036-RETIRE: only properly executed D1 and D2 clean nulls retire."""

    _write_required(
        tmp_path,
        d1=_d1(delta=0.0, ci95=[-0.03, 0.02], mcnemar_p=0.9, scorer_trained=True),
        d2=_d2(delta=-0.01, ci95=[-0.04, 0.01], mcnemar_p=0.7),
        d3=_d3(delta=0.0, ci95=[-0.02, 0.02], mcnemar_p=1.0),
        d6=_d6(cascade_accuracy=0.7, judge_only_accuracy=0.82, paired_ci95=[-0.2, -0.05], judge_call_fraction=1.0),
        d4=_d4(best_verifier_from="D1", delta=0.0, ci95=[-0.02, 0.02], mcnemar_p_second=1.0),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["honest_verdict"] == "complete_moat_retired_bounded_lora_ebm_and_uprm_both_null"
    assert artifact["decision"] == "BOUNDED-RETIRE"
    assert artifact["moat_realized"] is False
    assert artifact["moat_retired_bounded"] is True
    assert artifact["execution_incomplete_arms"] == []
    assert artifact["diffusiongemma_gate_conditions_satisfied_off_arc"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5036_defensive_branches_do_not_fabricate_metrics(tmp_path: Path) -> None:
    """REQ-REPORT-5036: malformed, oracle, flagged, and degenerate inputs stay scoped."""

    assert mod._number(True) is None
    assert mod._number("not numeric") is None
    assert mod._ci95("bad") is None
    assert mod._ci95(["bad", 0.1]) is None
    assert mod._format_delta(None) == "unknown"
    assert mod._slug("") == "unknown"
    assert mod._baseline_context(None) == (0.585, True)
    assert mod._best_accuracy_row([]) is None
    assert mod._d1_d2_retired([]) is False
    assert mod._first_incomplete_slug([]) == "unknown"
    assert mod._metric_text(None) == "no clean numeric arm"

    _write_json(tmp_path / mod.BASELINE_ARTIFACT_RELATIVE_PATH, _baseline())
    _write_json(tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH, _d1(flagged=True, delta=0.2, ci95=[0.1, 0.3], mcnemar_p=0.001))
    _write_json(tmp_path / mod.D2_ARTIFACT_RELATIVE_PATH, ["not", "an", "object"])
    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH, write=False)
    assert artifact["honest_verdict"] == "blocked_no_moat_arms"
    assert artifact["decision"] == "BLOCKED-NO-MOAT-ARMS"
    assert artifact["per_arm_table"] == []
    assert artifact["flagged_arms_skipped"][0]["arm_id"] == "D1"
    assert artifact["missing_upstream_artifacts"][0]["error"] == "artifact is not a JSON object"
    assert artifact["duration_s"] >= 0.0001

    skeleton = mod._accuracy_row(
        mod.D1_SPEC,
        tmp_path / "d1.json",
        _d1(scorer_trained=False, delta=0.1, ci95=[0.05, 0.15], mcnemar_p=0.01),
    )
    assert skeleton["execution_status"] == "skeleton"
    oracle_row = dict(skeleton, execution_status="clean", verifier_is_oracle=True, headroom_present=True)
    assert mod._row_is_positive(oracle_row) is False

    degenerate = mod._accuracy_row(
        mod.D3_SPEC,
        tmp_path / "d3.json",
        _d3(abstention_rate=0.75, delta=0.1, ci95=[0.05, 0.15], mcnemar_p=0.01),
    )
    assert degenerate["execution_status"] == "degenerate"

    malformed_d4 = mod._d4_row(
        tmp_path / "d4.json",
        _d4(best_verifier_from="D3", delta=None, ci95=None, mcnemar_p_second=None),
    )
    assert malformed_d4["arm_id"] == "D4"
    assert malformed_d4["execution_status"] == "blocked"

    clean_null_row = mod._accuracy_row(
        mod.D3_SPEC,
        tmp_path / "d3-clean.json",
        _d3(delta=0.0, ci95=[-0.02, 0.02], mcnemar_p=1.0),
    )
    positive_row = mod._accuracy_row(
        mod.D1_SPEC,
        tmp_path / "d1-positive.json",
        _d1(delta=0.1, ci95=[0.04, 0.16], mcnemar_p=0.01),
    )
    no_headroom_row = dict(clean_null_row, headroom_present=False)
    assert mod._mixed_verdict([no_headroom_row]) == "complete_moat_scoped_no_headroom_present_false_negative_risk"
    assert mod._mixed_verdict([positive_row]) == "complete_moat_scoped_positive_musr_no_cross_corpus_confirm"
    assert mod._mixed_verdict([clean_null_row]) == "complete_moat_scoped_no_realized_no_bounded_retirement"
    decision, verdict, best, efficiency_win, incomplete = mod._decision([clean_null_row])
    assert (decision, verdict, best["arm_id"], efficiency_win, incomplete) == (
        "MIXED-SCOPED",
        "complete_moat_scoped_no_realized_no_bounded_retirement",
        "D3",
        False,
        [],
    )
    assert "mixed/scoped" in mod._paper_summary("MIXED-SCOPED", clean_null_row, [], [])

    errors = mod.artifact_schema_errors({})
    for field in (
        "honest_verdict",
        "moat_realized",
        "execution_incomplete_arms",
        "per_arm_table",
        "diffusiongemma_gate_conditions_satisfied_off_arc",
        "inference_substrate",
    ):
        assert field in errors
    assert "diffusiongemma_gate_status" in mod.artifact_schema_errors({"diffusiongemma_gate_status": "MET"})
    assert "best_arm" in mod.artifact_schema_errors({"best_arm": []})
