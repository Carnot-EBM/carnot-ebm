"""Tests for Exp 5007 off-ARC verifier-moat gate resolution.

Spec refs: REQ-VERIFY-5007, SCENARIO-VERIFY-5007.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5007_moat_gate_resolution as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _arm_payload(
    *,
    experiment: str,
    accuracy_field: str,
    accuracy: float,
    tuned_sc_accuracy: float,
    delta: float,
    ci95: list[float],
    mcnemar_p: float,
    flagged: bool = False,
    headroom_present: bool = True,
    verifier_is_oracle: bool = False,
    corpus: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": experiment,
        "honest_verdict": "success_fixture" if delta > 0 else "complete_fixture_null",
        "flagged_adversarial": flagged,
        "verifier_is_oracle": verifier_is_oracle,
        "headroom_present": headroom_present,
        accuracy_field: accuracy,
        "tuned_sc_accuracy": tuned_sc_accuracy,
        "delta_vs_tuned_sc": delta,
        "paired_ci95": ci95,
        "mcnemar_p": mcnemar_p,
        "oracle_at_k": 0.9,
        "n_questions": 200,
        "model_specs": {"fixture": experiment},
    }
    if corpus is not None:
        payload["corpus"] = corpus
    return payload


def _d1(
    *,
    delta: float,
    ci95: list[float],
    mcnemar_p: float,
    flagged: bool = False,
    headroom_present: bool = True,
) -> dict[str, Any]:
    return _arm_payload(
        experiment="experiment_5003_lora_ebm_scorer_musr",
        accuracy_field="trained_scorer_accuracy",
        accuracy=0.58 + delta,
        tuned_sc_accuracy=0.58,
        delta=delta,
        ci95=ci95,
        mcnemar_p=mcnemar_p,
        flagged=flagged,
        headroom_present=headroom_present,
    )


def _d2(*, delta: float, ci95: list[float], mcnemar_p: float) -> dict[str, Any]:
    return _arm_payload(
        experiment="experiment_5004_uprm_replication",
        accuracy_field="uprm_selection_accuracy",
        accuracy=0.58 + delta,
        tuned_sc_accuracy=0.58,
        delta=delta,
        ci95=ci95,
        mcnemar_p=mcnemar_p,
        corpus="MuSR",
    )


def _d3(*, delta: float, ci95: list[float], mcnemar_p: float) -> dict[str, Any]:
    return _arm_payload(
        experiment="experiment_5005_ebrm_uncertainty_verifier",
        accuracy_field="ebrm_selection_accuracy",
        accuracy=0.58 + delta,
        tuned_sc_accuracy=0.58,
        delta=delta,
        ci95=ci95,
        mcnemar_p=mcnemar_p,
    )


def _d4(
    *,
    best_verifier_from: str,
    second_corpus: str,
    delta: float,
    ci95: list[float],
    mcnemar_p: float,
    flagged: bool = False,
    headroom_present: bool = True,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_5006_moat_second_corpus",
        "honest_verdict": "success_moat_generalizes_fixture" if delta > 0 else "complete_fixture",
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "headroom_present": headroom_present,
        "best_verifier_from": best_verifier_from,
        "second_corpus": second_corpus,
        "second_corpus_accuracy": 0.6 + delta,
        "tuned_sc_accuracy_second": 0.6,
        "delta_vs_tuned_sc_second": delta,
        "paired_ci95_second": ci95,
        "mcnemar_p_second": mcnemar_p,
        "oracle_at_k_second": 0.9,
        "n_questions": 200,
        "model_specs": {"fixture": "experiment_5006_moat_second_corpus"},
    }


def _clean_audit(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_req_verify_5007_spec_declares_gate_resolution_contract() -> None:
    """REQ-VERIFY-5007: OpenSpec anchors the D5 aggregation artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5007",
        "SCENARIO-VERIFY-5007",
        "experiment_5007_moat_gate_resolution.py",
        "results/experiment_5007_moat_gate_resolution.json",
        "flagged_adversarial=true",
        "success_moat_realized_off_arc_<arm>_<corpus>_<delta>",
        "complete_moat_retired_bounded_lora_ebm_and_uprm_both_null",
        "diffusiongemma_gate_conditions_satisfied_off_arc",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_verify_5007_realizes_cross_corpus_positive(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5007: MuSR plus second-corpus positive realizes the moat."""

    d1_path = tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH
    _write_json(d1_path, _d1(delta=0.06, ci95=[0.02, 0.1], mcnemar_p=0.01))
    _write_json(
        tmp_path / mod.D2_ARTIFACT_RELATIVE_PATH,
        _d2(delta=0.0, ci95=[-0.02, 0.02], mcnemar_p=1.0),
    )
    _write_json(
        tmp_path / mod.D3_ARTIFACT_RELATIVE_PATH,
        _d3(delta=0.01, ci95=[-0.01, 0.03], mcnemar_p=0.4),
    )
    _write_json(
        tmp_path / mod.D4_ARTIFACT_RELATIVE_PATH,
        _d4(
            best_verifier_from="D1",
            second_corpus="MMLU-Pro-hard",
            delta=0.05,
            ci95=[0.015, 0.085],
            mcnemar_p=0.02,
        ),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        audit_runner=_clean_audit,
        summary_runner=lambda _path: 0,
        write=True,
    )

    assert artifact["honest_verdict"].startswith(
        "success_moat_realized_off_arc_lora_ebm_musr_"
    )
    assert artifact["decision"] == "POSITIVE"
    assert artifact["moat_realized"] is True
    assert artifact["moat_retired_bounded"] is False
    assert artifact["diffusiongemma_gate_conditions_satisfied_off_arc"] is True
    assert artifact["diffusiongemma_gate_status"] != "MET"
    assert artifact["diffusiongemma_activation"] == "operator_gated_not_flipped"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["flagged_arms_skipped"] == []
    assert artifact["best_arm"]["arm"] == "LoRA-EBM"
    assert artifact["best_arm"]["delta_vs_tuned_sc"] == pytest.approx(0.06)
    assert {(row["arm"], row["corpus"]) for row in artifact["per_arm_table"]} >= {
        ("LoRA-EBM", "MuSR"),
        ("LoRA-EBM", "MMLU-Pro-hard"),
    }
    assert artifact["cited_upstream_artifacts"][0]["sha256"] == _sha(d1_path)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5007_skips_flagged_artifacts(tmp_path: Path) -> None:
    """REQ-VERIFY-5007: flagged arms are cited but never aggregated."""

    _write_json(
        tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH,
        _d1(delta=0.2, ci95=[0.1, 0.3], mcnemar_p=0.001, flagged=True),
    )
    _write_json(
        tmp_path / mod.D2_ARTIFACT_RELATIVE_PATH,
        _d2(delta=0.0, ci95=[-0.02, 0.02], mcnemar_p=1.0),
    )
    _write_json(
        tmp_path / mod.D3_ARTIFACT_RELATIVE_PATH,
        _d3(delta=0.0, ci95=[-0.02, 0.02], mcnemar_p=1.0),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        audit_runner=_clean_audit,
        summary_runner=lambda _path: 0,
    )

    assert artifact["decision"] == "MIXED-SCOPED"
    assert artifact["moat_realized"] is False
    assert artifact["moat_retired_bounded"] is False
    assert [item["arm_id"] for item in artifact["flagged_arms_skipped"]] == ["D1"]
    assert "LoRA-EBM" not in {row["arm"] for row in artifact["per_arm_table"]}
    d1_citation = next(item for item in artifact["cited_upstream_artifacts"] if item["arm_id"] == "D1")
    assert d1_citation["fields_imported"] == ["flagged_adversarial", "honest_verdict"]
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5007_retires_only_when_clean_d1_d2_both_null(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5007: clean D1 and D2 nulls produce bounded retirement."""

    _write_json(
        tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH,
        _d1(delta=0.0, ci95=[-0.03, 0.02], mcnemar_p=1.0),
    )
    _write_json(
        tmp_path / mod.D2_ARTIFACT_RELATIVE_PATH,
        _d2(delta=-0.01, ci95=[-0.04, 0.01], mcnemar_p=0.7),
    )
    _write_json(
        tmp_path / mod.D3_ARTIFACT_RELATIVE_PATH,
        _d3(delta=0.0, ci95=[-0.02, 0.02], mcnemar_p=1.0),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        audit_runner=_clean_audit,
        summary_runner=lambda _path: 0,
    )

    assert artifact["honest_verdict"] == "complete_moat_retired_bounded_lora_ebm_and_uprm_both_null"
    assert artifact["decision"] == "ALL-NULL-RETIRE"
    assert artifact["moat_realized"] is False
    assert artifact["moat_retired_bounded"] is True
    assert artifact["diffusiongemma_gate_conditions_satisfied_off_arc"] is False
    assert "D1 and D2 both failed" in artifact["paper_summary"]
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5007_blocks_when_no_nonflagged_d_arm(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5007: all flagged or absent D arms block without a moat claim."""

    _write_json(
        tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH,
        _d1(delta=0.2, ci95=[0.1, 0.3], mcnemar_p=0.001, flagged=True),
    )
    _write_json(
        tmp_path / mod.D4_ARTIFACT_RELATIVE_PATH,
        _d4(
            best_verifier_from="D1",
            second_corpus="MMLU-Pro-hard",
            delta=0.2,
            ci95=[0.1, 0.3],
            mcnemar_p=0.001,
            flagged=True,
        ),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        audit_runner=_clean_audit,
        summary_runner=lambda _path: 0,
    )

    assert artifact["honest_verdict"] == "blocked_no_moat_arms"
    assert artifact["decision"] == "BLOCKED"
    assert artifact["moat_realized"] is False
    assert artifact["moat_retired_bounded"] is False
    assert artifact["per_arm_table"] == []
    assert [item["arm_id"] for item in artifact["flagged_arms_skipped"]] == ["D1", "D4"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5007_defensive_helpers_and_scoped_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-5007: defensive parsing stays scoped instead of fabricating metrics."""

    assert mod._number(True) is None
    assert mod._number("not numeric") is None
    assert mod._ci95("bad") is None
    assert mod._ci95(["bad", 1.0]) is None
    assert mod._format_delta(None) == "unknown"
    assert mod._metric_text(None) == "no clean numeric arm"
    assert mod._second_confirmation_text([], "D1") == "no D4 second-corpus row was available"

    no_headroom = [
        {
            "source_experiment_id": 5003,
            "arm_id": "D1",
            "verifier_is_oracle": False,
            "headroom_present": False,
            "delta_vs_tuned_sc": 0.1,
            "paired_ci95": [0.02, 0.18],
            "mcnemar_p": 0.01,
        }
    ]
    assert (
        mod._mixed_verdict(no_headroom)
        == "complete_moat_scoped_no_headroom_present_false_negative_risk"
    )
    musr_only = [dict(no_headroom[0], headroom_present=True)]
    assert mod._mixed_verdict(musr_only) == "complete_moat_scoped_positive_musr_no_cross_corpus_confirm"
    second_only = [dict(musr_only[0], source_experiment_id=5006)]
    assert (
        mod._mixed_verdict(second_only)
        == "complete_moat_scoped_second_corpus_unanchored_no_musr_positive"
    )

    assert mod._compact_adversarial_flags({"reports": [{"flags": [{"kind": "X"}]}]}) == [
        {"kind": "X"}
    ]
    assert mod._audit_is_clean({"max_severity": 0}) is True
    assert mod._audit_is_clean({"max_severity": 1}) is False
    assert mod._audit_is_clean({"flagged_count": 0}) is True
    assert mod._audit_is_clean({"flagged_count": 1}) is False
    assert mod._audit_is_clean({"flags": []}) is True

    _write_json(tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH, ["not", "an", "object"])
    artifact = mod.run(root=tmp_path, artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH, write=False)
    assert artifact["decision"] == "BLOCKED"
    assert artifact["missing_upstream_artifacts"][0]["error"] == "artifact is not a JSON object"

    empty_errors = mod.artifact_schema_errors({})
    for field in (
        "adversarial_verify_clean",
        "decision",
        "duration_s",
        "field_principles",
        "honest_verdict",
        "inference_substrate",
        "paper_summary",
        "per_arm_table",
        "spec_refs",
        "verifier_is_oracle",
    ):
        assert field in empty_errors
    assert "best_arm" in mod.artifact_schema_errors({"best_arm": []})
    assert "diffusiongemma_gate_status" in mod.artifact_schema_errors(
        {"diffusiongemma_gate_status": "MET"}
    )
