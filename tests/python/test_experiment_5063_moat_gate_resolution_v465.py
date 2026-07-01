"""Tests for Exp 5063 .465 verifier-moat gate resolution.

Spec refs: REQ-REPORT-5063, SCENARIO-REPORT-5063.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5063_moat_gate_resolution_v465 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _d1(
    *,
    delta: float = 0.0,
    ci: list[float] | None = None,
    p: float = 1.0,
    proper: bool = False,
    flagged: bool = False,
    blocked: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_5059_d1_sota_refresh_audit",
        "experiment_id": 5059,
        "honest_verdict": "blocked_sota_candidate_refresh_unavailable"
        if blocked
        else "complete_d1_fixture",
        "best_arm_available": not blocked,
        "proper_musr_win": proper,
        "accuracy": round(0.585 + delta, 6),
        "tuned_sc_accuracy": 0.585,
        "delta_vs_tuned_sc": delta,
        "paired_ci95": ci if ci is not None else [-0.02, 0.02],
        "mcnemar_p": p,
        "headroom_present": True,
        "verifier_is_oracle": False,
        "n_questions": 200,
    }
    if blocked:
        payload["status"] = "blocked"
    if flagged:
        payload["flagged_adversarial"] = True
        payload["corrigendum_pending"] = [
            {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}
        ]
    return payload


def _d4(
    *,
    audit_clean: bool = True,
    confirmed: bool = False,
    delta: float = 0.0,
    blocked: bool = False,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_5060_second_corpus_audit_v2",
        "experiment_id": 5060,
        "honest_verdict": "blocked_second_corpus_unavailable"
        if blocked
        else ("success_d4_fixture" if confirmed else "retired_d4_fixture"),
        "status": "blocked" if blocked else "complete",
        "d4_verdict_class": "confirmed" if confirmed else "retired",
        "second_corpus_confirmed": confirmed,
        "second_corpus_audit_clean": audit_clean,
        "duplicate_audit_passed": audit_clean,
        "leak_audit_passed": True,
        "oracle_provenance_passed": True,
        "train_test_overlap_passed": True,
        "delta_vs_tuned_sc_second": delta,
        "paired_ci95_second": [0.02, 0.09] if delta > 0 else [-0.08, 0.01],
        "mcnemar_p_second": 0.01 if delta > 0 else 0.8,
        "headroom_present": True,
        "verifier_is_oracle": False,
        "n_questions_second": 100,
    }


def _d6(
    *,
    efficiency: bool = False,
    delta: float = 0.0,
    ci: list[float] | None = None,
    blocked: bool = False,
    flagged: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_5061_tool_first_cascade",
        "experiment_id": 5061,
        "honest_verdict": "blocked_tool_first_verifier_unavailable"
        if blocked
        else ("success_tool_first_cascade_parity_at_0pct_judge_calls" if efficiency else "complete_tool_first_cascade_no_efficiency_win"),
        "status": "blocked" if blocked else "complete",
        "cascade_executed": not blocked,
        "cascade_accuracy": None if blocked else round(0.585 + delta, 6),
        "judge_only_accuracy": None if blocked else 0.585,
        "delta_vs_judge_only": None if blocked else delta,
        "paired_ci95": None if blocked else (ci if ci is not None else [-0.02, 0.02]),
        "judge_call_fraction": None if blocked else (0.0 if efficiency else 1.0),
        "efficiency_win": False if blocked else efficiency,
        "verifier_is_oracle": False,
    }
    if flagged:
        payload["flagged_adversarial"] = True
        payload["corrigendum_pending"] = [{"kind": "TAUTOLOGY", "severity": "critical"}]
    return payload


def _guided(*, delta: float = 0.0, blocked: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_5062_guided_decoding_cost_frontier",
        "experiment_id": 5062,
        "honest_verdict": "blocked_guided_decoding_precondition"
        if blocked
        else "complete_guided_fixture",
        "status": "blocked" if blocked else "complete",
        "guided_decoding_executed": not blocked,
        "arms_differentiated": not blocked,
        "guided_accuracy": None if blocked else round(0.444444 + delta, 6),
        "unguided_accuracy": None if blocked else 0.444444,
        "delta_guided_vs_unguided": None if blocked else delta,
        "nfe_by_arm": None if blocked else {"guided": 45.0, "unguided": 9.0, "rerank_only": 72.0},
        "verifier_is_oracle": False,
    }


def _write_v465(
    root: Path,
    *,
    d1: dict[str, Any] | None = None,
    d4: dict[str, Any] | None = None,
    d6: dict[str, Any] | None = None,
    guided: dict[str, Any] | None = None,
) -> None:
    if d1 is not None:
        _write_json(root / mod.D1_ARTIFACT_RELATIVE_PATH, d1)
    if d4 is not None:
        _write_json(root / mod.D4_ARTIFACT_RELATIVE_PATH, d4)
    if d6 is not None:
        _write_json(root / mod.D6_ARTIFACT_RELATIVE_PATH, d6)
    if guided is not None:
        _write_json(root / mod.GUIDED_ARTIFACT_RELATIVE_PATH, guided)


def test_req_report_5063_spec_declares_v465_gate_contract() -> None:
    """REQ-REPORT-5063: OpenSpec anchors the .465 gate artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5063",
        "SCENARIO-REPORT-5063-INCOMPLETE",
        "SCENARIO-REPORT-5063-REALIZED",
        "SCENARIO-REPORT-5063-RETIRE",
        "experiment_5063_moat_gate_resolution_v465.py",
        "results/experiment_5063_moat_gate_resolution_v465.json",
        "`second_corpus_audit_clean`",
        "`guided_decoding_frontier_state`",
    ):
        assert marker in spec


def test_scenario_report_5063_incomplete_refuses_flagged_d1_and_unclean_d4(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5063-INCOMPLETE: flagged and unclean positives are not counted."""

    _write_v465(
        tmp_path,
        d1=_d1(delta=0.08, ci=[0.0, 0.165], p=0.076369, flagged=True),
        d4=_d4(audit_clean=False, confirmed=False, delta=0.37),
        d6=_d6(efficiency=True, delta=0.08, ci=[0.0, 0.165]),
        guided=_guided(delta=0.111112),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["moat_state"] == "execution_incomplete"
    assert artifact["honest_verdict"] == "complete_moat_execution_incomplete_v465_blocked_flagged_or_unclean"
    assert artifact["best_arm"] == "D6"
    assert artifact["best_arm_delta"] == 0.08
    assert artifact["second_corpus_confirmed"] is False
    assert artifact["second_corpus_audit_clean"] is False
    assert artifact["cascade_efficiency_win"] is True
    assert artifact["bounded_retirement_ok"] is False
    assert artifact["guided_decoding_frontier_state"].startswith("guided_gain_observed")
    assert {item["artifact_id"] for item in artifact["flagged_upstream_artifacts"]} == {"D1"}
    assert {item["artifact_id"] for item in artifact["clean_upstream_artifacts"]} == {"D4", "D6", "G1"}
    assert any("D1 flagged" in reason for reason in artifact["execution_incomplete_reasons"])
    assert any("D4 audit not clean" in reason for reason in artifact["execution_incomplete_reasons"])
    assert all(row["artifact_id"] != "D4" or row["status"] == "clean" for row in artifact["per_arm_table"])
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5063_realized_by_clean_d1_and_d4(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5063-REALIZED: clean MuSR win plus clean D4 confirmation realizes."""

    _write_v465(
        tmp_path,
        d1=_d1(delta=0.07, ci=[0.03, 0.12], p=0.01, proper=True),
        d4=_d4(audit_clean=True, confirmed=True, delta=0.08),
        d6=_d6(efficiency=False, delta=-0.01),
        guided=_guided(delta=0.02),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["moat_state"] == "moat_realized"
    assert artifact["honest_verdict"].startswith("success_moat_realized_v465_d1")
    assert artifact["best_arm"] == "D1"
    assert artifact["second_corpus_confirmed"] is True
    assert artifact["second_corpus_audit_clean"] is True
    assert artifact["cascade_efficiency_win"] is False
    assert artifact["bounded_retirement_ok"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5063_musr_and_second_corpus_scoped_states(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5063: scoped positives preserve non-headline evidence."""

    musr_root = tmp_path / "musr"
    _write_v465(
        musr_root,
        d1=_d1(delta=0.06, ci=[0.02, 0.11], p=0.02, proper=True),
        d4=_d4(audit_clean=True, confirmed=False, delta=-0.02),
        d6=_d6(efficiency=False, delta=0.0),
        guided=_guided(delta=-0.01),
    )
    musr_artifact = mod.run(root=musr_root, artifact_path=musr_root / mod.RESULT_RELATIVE_PATH)
    assert musr_artifact["moat_state"] == "musr_scoped_positive"
    assert musr_artifact["best_arm"] == "D1"
    assert musr_artifact["second_corpus_confirmed"] is False

    second_root = tmp_path / "second"
    _write_v465(
        second_root,
        d1=_d1(blocked=True),
        d4=_d4(audit_clean=True, confirmed=True, delta=0.12),
        d6=_d6(efficiency=False, delta=0.0),
        guided=_guided(delta=0.0),
    )
    second_artifact = mod.run(root=second_root, artifact_path=second_root / mod.RESULT_RELATIVE_PATH)
    assert second_artifact["moat_state"] == "second_corpus_scoped_positive"
    assert second_artifact["best_arm"] == "D4"
    assert second_artifact["best_arm_delta"] == 0.12
    assert any("D1 blocked" in reason for reason in second_artifact["execution_incomplete_reasons"])
    assert mod.artifact_schema_errors(second_artifact) == []


def test_scenario_report_5063_retired_bounded_after_clean_nulls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5063-RETIRE: clean D1/D4/D6 nulls bound retirement."""

    _write_v465(
        tmp_path,
        d1=_d1(delta=-0.01, ci=[-0.06, 0.03], p=0.7, proper=False),
        d4=_d4(audit_clean=True, confirmed=False, delta=-0.04),
        d6=_d6(efficiency=False, delta=-0.02, ci=[-0.08, 0.01]),
        guided=_guided(delta=-0.02),
    )

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["moat_state"] == "retired_bounded"
    assert artifact["honest_verdict"] == "complete_moat_retired_bounded_v465_clean_d1_d4_d6_null"
    assert artifact["bounded_retirement_ok"] is True
    assert artifact["execution_incomplete_reasons"] == []
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5063_records_missing_blocked_malformed_flagged_and_schema_errors(tmp_path: Path) -> None:
    """REQ-REPORT-5063: status buckets and schema checks are defensive."""

    malformed_path = tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text("{not-json", encoding="utf-8")
    _write_json(tmp_path / mod.D4_ARTIFACT_RELATIVE_PATH, _d4(blocked=True))
    _write_json(tmp_path / mod.D6_ARTIFACT_RELATIVE_PATH, _d6(flagged=True))

    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH)

    assert {item["artifact_id"] for item in artifact["malformed_upstream_artifacts"]} == {"D1"}
    assert {item["artifact_id"] for item in artifact["blocked_upstream_artifacts"]} == {"D4"}
    assert {item["artifact_id"] for item in artifact["flagged_upstream_artifacts"]} == {"D6"}
    assert {item["artifact_id"] for item in artifact["missing_upstream_artifacts"]} == {"G1"}
    assert artifact["moat_state"] == "execution_incomplete"
    assert mod._ci95("bad") is None
    assert mod._ci95(["bad", 0.1]) is None
    assert mod._ci_excludes_zero_positive([0.0, 0.2]) is False
    assert mod._number(True) is None
    assert mod._format_delta(None) == "unknown"
    assert mod._flags_from_payload({"corrigendum_pending": "bad"}) == []
    assert mod._status_from_payload({"honest_verdict": "complete_missing"}, ("missing",)) == "malformed"
    assert mod._row_by_id([], "D1") is None
    assert mod._d1_proper_win(None) is False
    assert mod._guided_frontier_state(None) == "missing"
    assert mod._guided_frontier_state({"status": "clean", "guided_decoding_executed": False}) == "not_executed"
    assert (
        mod._guided_frontier_state(
            {"status": "clean", "guided_decoding_executed": True, "arms_differentiated": False}
        )
        == "controls_not_differentiated"
    )
    non_object_path = tmp_path / "list.json"
    non_object_path.write_text("[]", encoding="utf-8")
    assert mod._read_json_object(non_object_path) == (None, "top-level JSON value is not an object")

    schema_errors = mod.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "moat_state": "bad",
            "best_arm": None,
            "best_arm_delta": None,
            "best_arm_ci": None,
            "second_corpus_confirmed": "no",
            "second_corpus_audit_clean": "no",
            "cascade_efficiency_win": "no",
            "guided_decoding_frontier_state": None,
            "bounded_retirement_ok": "no",
            "execution_incomplete_reasons": "bad",
            "per_arm_table": "bad",
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
        "second_corpus_audit_clean",
        "cascade_efficiency_win",
        "bounded_retirement_ok",
        "execution_incomplete_reasons",
        "per_arm_table",
        "next_actions",
    ):
        assert field in schema_errors
