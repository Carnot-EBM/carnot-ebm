"""Tests for Exp 5050 .464 verifier-moat gate resolution.

Spec refs: REQ-REPORT-5050, SCENARIO-REPORT-5050.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5050_moat_gate_resolution_v464 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _musr_arm(
    experiment_id: int,
    *,
    delta: float = 0.0,
    ci: list[float] | None = None,
    p: float = 1.0,
    verdict: str = "complete_fixture_no_win",
    flagged: bool = False,
    blocked: bool = False,
    critical: bool = False,
    headroom: bool = True,
    oracle: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "honest_verdict": "blocked_fixture" if blocked else verdict,
        "flagged_adversarial": flagged,
        "verifier_is_oracle": oracle,
        "headroom_present": headroom,
        "genuine_tuned_sc_accuracy": 0.585,
        "delta_vs_tuned_sc": delta,
        "paired_ci95": ci if ci is not None else [-0.02, 0.02],
        "mcnemar_p": p,
        "n_questions": 200,
        "oracle_at_k": 0.865,
    }
    if experiment_id == 5045:
        payload.update(
            {
                "experiment": "experiment_5045_powered_lora_ebm_eorm_musr",
                "powered_scorer_available": not blocked,
                "scorer_trained": not blocked,
                "powered_lora_ebm_accuracy": round(0.585 + delta, 6),
            }
        )
    elif experiment_id == 5046:
        payload.update(
            {
                "experiment": "experiment_5046_vpr_process_reward_repair",
                "process_reward_available": not blocked,
                "process_reward_accuracy": round(0.585 + delta, 6),
            }
        )
    else:
        payload.update(
            {
                "experiment": "experiment_5047_kan_purm_energy_calibration",
                "calibration_available": not blocked,
                "calibrated_accuracy": round(0.585 + delta, 6),
                "delta_vs_powered_d1": delta,
                "degeneracy_guard_fired": False,
            }
        )
    if critical:
        payload["corrigendum_pending"] = [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}]
    return payload


def _d6(
    *,
    verdict: str = "complete_cascade_no_efficiency",
    blocked: bool = False,
    flagged: bool = False,
    fraction: float = 1.0,
    ci: list[float] | None = None,
    cascade_accuracy: float = 0.70,
    judge_only_accuracy: float = 0.82,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_5048_cross_model_cascade_repair",
        "experiment_id": 5048,
        "honest_verdict": "blocked_gate_check_failed" if blocked else verdict,
        "status": "blocked" if blocked else "complete",
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "cascade_accuracy": None if blocked else cascade_accuracy,
        "judge_only_accuracy": None if blocked else judge_only_accuracy,
        "paired_ci95_cascade_vs_judge": None if blocked else (ci if ci is not None else [-0.16, -0.05]),
        "judge_call_fraction": None if blocked else fraction,
        "cascade_judge_calls": None if blocked else int(200 * fraction),
        "judge_only_calls": 200,
        "n_questions": 0 if blocked else 200,
    }
    return payload


def _d4(
    *,
    best_arm: str = "D1",
    confirmed: bool = False,
    flagged: bool = False,
    blocked: bool = False,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_5049_second_corpus_confirmation",
        "experiment_id": 5049,
        "honest_verdict": "blocked_second_corpus_unavailable"
        if blocked
        else ("success_second_corpus_confirms_fixture" if confirmed else "complete_second_corpus_no_confirm"),
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "headroom_present": True,
        "second_corpus_confirmed": confirmed,
        "second_corpus_name": "FixtureBench",
        "best_arm": best_arm,
        "verifier_accuracy_second": 0.64 if confirmed else 0.58,
        "genuine_sc_accuracy_second": 0.585,
        "delta_vs_tuned_sc_second": 0.055 if confirmed else -0.005,
        "paired_ci95_second": [0.02, 0.09] if confirmed else [-0.04, 0.02],
        "mcnemar_p_second": 0.01 if confirmed else 0.8,
        "n_questions_second": 100,
    }


def _prior() -> dict[str, Any]:
    return {
        "experiment": "experiment_5036_moat_gate_resolution_v3",
        "experiment_id": 5036,
        "honest_verdict": "complete_moat_execution_incomplete_cascade",
    }


def _write_phase_d(
    root: Path,
    *,
    d1: dict[str, Any] | None = None,
    d2: dict[str, Any] | None = None,
    d3: dict[str, Any] | None = None,
    d6: dict[str, Any] | None = None,
    d6_available_only: bool = False,
    d4: dict[str, Any] | None = None,
) -> None:
    if d1 is not None:
        _write_json(root / mod.D1_ARTIFACT_RELATIVE_PATH, d1)
    if d2 is not None:
        _write_json(root / mod.D2_ARTIFACT_RELATIVE_PATH, d2)
    if d3 is not None:
        _write_json(root / mod.D3_ARTIFACT_RELATIVE_PATH, d3)
    if d6 is not None:
        rel = mod.D6_AVAILABLE_ARTIFACT_RELATIVE_PATH if d6_available_only else mod.D6_EXPECTED_ARTIFACT_RELATIVE_PATH
        _write_json(root / rel, d6)
    if d4 is not None:
        _write_json(root / mod.D4_ARTIFACT_RELATIVE_PATH, d4)
    _write_json(root / mod.PRIOR_GATE_ARTIFACT_RELATIVE_PATH, _prior())


def test_req_report_5050_spec_declares_v464_gate_contract() -> None:
    """REQ-REPORT-5050: OpenSpec anchors the .464 gate artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5050",
        "SCENARIO-REPORT-5050-REALIZED",
        "SCENARIO-REPORT-5050-MUSR-SCOPED",
        "SCENARIO-REPORT-5050-RETIRE",
        "SCENARIO-REPORT-5050-INCOMPLETE",
        "experiment_5050_moat_gate_resolution_v464.py",
        "results/experiment_5050_moat_gate_resolution_v464.json",
        "`moat_state`",
        "`execution_incomplete_reasons`",
    ):
        assert marker in spec


def test_scenario_report_5050_realized_by_confirmed_verifier_win(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5050-REALIZED: MuSR win plus second corpus realizes the moat."""

    _write_phase_d(
        tmp_path,
        d1=_musr_arm(5045, delta=0.07, ci=[0.03, 0.11], p=0.01, verdict="success_d1_fixture"),
        d2=_musr_arm(5046, delta=0.0),
        d3=_musr_arm(5047, delta=-0.01),
        d6=_d6(),
        d4=_d4(best_arm="D1", confirmed=True),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["moat_state"] == "moat_realized"
    assert artifact["honest_verdict"].startswith("success_moat_realized_v464_d1")
    assert artifact["best_arm"] == "D1"
    assert artifact["best_arm_delta"] == 0.07
    assert artifact["best_arm_ci"] == [0.03, 0.11]
    assert artifact["second_corpus_confirmed"] is True
    assert artifact["cascade_efficiency_win"] is False
    assert artifact["bounded_retirement_ok"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5050_musr_scoped_positive_without_confirmation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5050-MUSR-SCOPED: MuSR-only win is positive but scoped."""

    _write_phase_d(
        tmp_path,
        d1=_musr_arm(5045, delta=0.0),
        d2=_musr_arm(5046, delta=0.06, ci=[0.02, 0.10], p=0.02, verdict="success_d2_fixture"),
        d3=_musr_arm(5047, delta=0.01),
        d6=_d6(),
        d4=_d4(best_arm="D2", blocked=True),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["moat_state"] == "musr_scoped_positive"
    assert artifact["honest_verdict"].startswith("complete_moat_musr_scoped_positive_v464_d2")
    assert artifact["best_arm"] == "D2"
    assert artifact["second_corpus_confirmed"] is False
    assert any("D4" in reason for reason in artifact["execution_incomplete_reasons"])
    assert artifact["bounded_retirement_ok"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5050_retired_bounded_after_clean_no_wins(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5050-RETIRE: clean D1/D2/D3 no-wins and no cascade win retire."""

    _write_phase_d(
        tmp_path,
        d1=_musr_arm(5045, delta=0.0, ci=[-0.03, 0.03], p=0.9),
        d2=_musr_arm(5046, delta=-0.02, ci=[-0.06, 0.01], p=0.5),
        d3=_musr_arm(5047, delta=-0.01, ci=[-0.04, 0.02], p=0.7),
        d6=_d6(fraction=1.0, ci=[-0.16, -0.05]),
        d4=_d4(best_arm="D1", confirmed=False),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["moat_state"] == "retired_bounded"
    assert artifact["honest_verdict"] == "complete_moat_retired_bounded_v464_clean_d1_d2_d3_no_efficiency"
    assert artifact["bounded_retirement_ok"] is True
    assert artifact["cascade_efficiency_win"] is False
    assert artifact["execution_incomplete_reasons"] == []
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5050_execution_incomplete_lists_blocked_and_flagged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5050-INCOMPLETE: blocked arms are not scientific nulls."""

    _write_phase_d(
        tmp_path,
        d1=_musr_arm(5045, delta=0.08, ci=[0.0, 0.165], p=0.076, blocked=True),
        d2=_musr_arm(5046, delta=-0.03, flagged=True, critical=True),
        d3=_musr_arm(5047, delta=0.02, ci=[-0.105, -0.015], p=0.016),
        d6=_d6(blocked=True),
        d6_available_only=True,
        d4=_d4(best_arm="D1", confirmed=True, flagged=True),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["moat_state"] == "execution_incomplete"
    assert artifact["honest_verdict"] == "complete_moat_execution_incomplete_v464_blocked_or_missing_phase_d"
    assert artifact["best_arm"] == "D1"
    assert artifact["best_arm_delta"] == 0.08
    assert artifact["best_arm_ci"] == [0.0, 0.165]
    assert artifact["second_corpus_confirmed"] is False
    assert artifact["cascade_efficiency_win"] is False
    assert artifact["bounded_retirement_ok"] is False
    assert artifact["missing_upstream_artifacts"][0]["path"].endswith("experiment_5048_cross_model_cascade_repair.json")
    assert {item["arm_id"] for item in artifact["blocked_upstream_artifacts"]} == {"D1", "D6"}
    assert {item["arm_id"] for item in artifact["flagged_upstream_artifacts"]} == {"D2", "D4"}
    assert any("D1 blocked" in reason for reason in artifact["execution_incomplete_reasons"])
    assert any("D6 blocked" in reason for reason in artifact["execution_incomplete_reasons"])
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5050_defensive_branches_and_cascade_efficiency(tmp_path: Path) -> None:
    """REQ-REPORT-5050: defensive branches preserve the gate contract."""

    assert mod._number(True) is None
    assert mod._number("not-a-number") is None
    assert mod._ci95("bad") is None
    assert mod._ci95(["bad", 0.1]) is None
    assert mod._ci_includes_zero([-0.01, 0.02]) is True
    assert mod._format_delta(None) == "unknown"
    assert mod._critical_corrigendum({"corrigendum_pending": [{"severity": "critical"}]}) is True
    assert mod._status({"corrigendum_pending": [{"severity": "critical"}]}) == "critical_corrigendum"
    assert mod._accuracy_from({}, ("missing",)) is None
    assert mod._cascade_efficiency_win(None) is False
    assert mod._best_musr_row([]) is None

    cascade_root = tmp_path / "cascade"
    _write_phase_d(
        cascade_root,
        d1=_musr_arm(5045, delta=0.0),
        d2=_musr_arm(5046, delta=-0.01),
        d3=_musr_arm(5047, delta=0.0),
        d6=_d6(fraction=0.25, ci=[-0.02, 0.02], cascade_accuracy=0.80, judge_only_accuracy=0.81),
        d4=_d4(confirmed=False),
    )
    cascade_artifact = mod.run(root=cascade_root, artifact_path=cascade_root / mod.RESULT_RELATIVE_PATH)
    assert cascade_artifact["moat_state"] == "moat_realized"
    assert cascade_artifact["best_arm"] == "D6"
    assert cascade_artifact["cascade_efficiency_win"] is True
    assert cascade_artifact["best_arm_delta"] == -0.01

    missing_root = tmp_path / "missing"
    _write_json(missing_root / mod.D6_EXPECTED_ARTIFACT_RELATIVE_PATH, _d6(flagged=True))
    loaded_missing = mod.load_phase_d_artifacts(missing_root)
    assert {"D1", "D2", "D3", "D4", "D5-prior"} <= {
        item["arm_id"] for item in loaded_missing["missing_upstream_artifacts"]
    }
    assert loaded_missing["flagged_upstream_artifacts"][0]["arm_id"] == "D6"

    critical_root = tmp_path / "critical"
    _write_phase_d(
        critical_root,
        d1=_musr_arm(5045, delta=0.0),
        d2=_musr_arm(5046, delta=0.0, critical=True),
        d3=_musr_arm(5047, delta=0.0),
        d6={**_d6(), "corrigendum_pending": [{"severity": "critical"}]},
        d4={**_d4(), "corrigendum_pending": [{"severity": "critical"}]},
    )
    loaded_critical = mod.load_phase_d_artifacts(critical_root)
    assert {item["arm_id"] for item in loaded_critical["critical_corrigendum_artifacts"]} == {"D2", "D6", "D4"}

    incomplete_root = tmp_path / "incomplete"
    _write_phase_d(
        incomplete_root,
        d1=_musr_arm(5045, delta=0.0),
        d2=_musr_arm(5046, delta=0.0),
        d3={"experiment_id": 5047, "honest_verdict": "complete_missing_metrics"},
        d6={"experiment_id": 5048, "honest_verdict": "complete_missing_metrics"},
        d4={"experiment_id": 5049, "honest_verdict": "complete_missing_metrics"},
    )
    loaded_incomplete = mod.load_phase_d_artifacts(incomplete_root)
    assert {item["arm_id"] for item in loaded_incomplete["malformed_upstream_artifacts"]} == {"D3", "D6", "D4"}

    assert "honest_verdict" in mod.artifact_schema_errors({})

    schema_errors = mod.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "moat_state": "bad",
            "best_arm": None,
            "best_arm_delta": None,
            "best_arm_ci": None,
            "second_corpus_confirmed": "no",
            "cascade_efficiency_win": "no",
            "execution_incomplete_reasons": "bad",
            "bounded_retirement_ok": "no",
            "next_actions": "bad",
            "spec_refs": [],
            "inference_substrate": "live",
            "field_principles": {},
        }
    )
    for field in (
        "honest_verdict",
        "moat_state",
        "spec_refs",
        "inference_substrate",
        "field_principles",
        "second_corpus_confirmed",
        "cascade_efficiency_win",
        "bounded_retirement_ok",
        "execution_incomplete_reasons",
        "next_actions",
    ):
        assert field in schema_errors
