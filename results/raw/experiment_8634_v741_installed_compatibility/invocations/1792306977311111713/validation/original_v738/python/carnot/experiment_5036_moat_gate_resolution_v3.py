"""Exp 5036: resolve the Phase D5 off-ARC verifier-moat gate from v3 D artifacts.

Spec refs: REQ-REPORT-5036, SCENARIO-REPORT-5036.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5022_moat_gate_resolution_v2 as base


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
ArtifactSpec = base.ArtifactSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5036
RESULT_RELATIVE_PATH = "results/experiment_5036_moat_gate_resolution_v3.json"
BASELINE_ARTIFACT_RELATIVE_PATH = "results/experiment_5015_genuine_sc_baseline_fix.json"
D1_ARTIFACT_RELATIVE_PATH = "results/experiment_5031_lora_ebm_scorer_musr_v3.json"
D2_ARTIFACT_RELATIVE_PATH = "results/experiment_5032_uprm_replication_v3.json"
D3_ARTIFACT_RELATIVE_PATH = "results/experiment_5033_ebrm_uncertainty_verifier_v3.json"
D6_ARTIFACT_RELATIVE_PATH = "results/experiment_5034_uncertainty_routed_cascade_v2.json"
D4_ARTIFACT_RELATIVE_PATH = "results/experiment_5035_moat_second_corpus_v3.json"
SPEC_REFS = ["REQ-REPORT-5036", "SCENARIO-REPORT-5036"]
GENUINE_TUNED_SC_ACCURACY = 0.585
MATERIAL_JUDGE_FRACTION = 0.8

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a realized moat is success_moat_realized_off_arc_<arm>_<corpus>_<delta>, "
            "a bounded retirement is complete_moat_retired_bounded_lora_ebm_and_uprm_both_null, "
            "an unexecuted arm is complete_moat_execution_incomplete_<arm>."
        )
    },
    "moat_realized": {
        "principle": (
            "true iff >=1 oracle-distinct arm beats the GENUINE tuned-SC with CI95-excl-0 on a "
            "headroom-present corpus (cross-corpus confirmed), OR the cascade hits an efficiency Pareto win."
        )
    },
    "moat_retired_bounded": {
        "principle": (
            "true iff the PROPERLY-EXECUTED D1 (scorer_trained) AND D2 both clean-null on every "
            "headroom-present oracle-distinct corpus AND no efficiency win -- NOT triggered by a "
            "skeleton/degenerate/blocked arm."
        )
    },
    "execution_incomplete_arms": {
        "principle": (
            "arms that did NOT cleanly execute (D1 skeleton / D2 blocked / D3 degenerate) -- these are FAILED "
            "executions to re-run in .464, NOT nulls that bound the moat."
        )
    },
    "best_arm": {
        "principle": (
            "the construction with the strongest oracle-distinct delta (LoRA-EBM/uPRM/EBRM) OR the efficiency "
            "Pareto point + its corpus + delta + CI."
        )
    },
    "efficiency_win": {
        "principle": (
            "true iff the cascade reached accuracy parity (within CI of judge-only) at materially fewer judge "
            "calls (north-star §5)."
        )
    },
    "per_arm_table": {
        "principle": (
            "per arm per corpus: delta_vs_tuned_sc (vs GENUINE SC 0.585), paired_ci95, mcnemar_p, "
            "verifier_is_oracle, headroom_present, scorer_trained/abstention_rate (the audit + "
            "execution-quality trail)."
        )
    },
    "diffusiongemma_gate_conditions_satisfied_off_arc": {
        "principle": (
            "true iff a POSITIVE arm satisfies the gate's 3 conditions ON THE TESTED DOMAIN; activation stays "
            "operator-gated (do NOT autonomously flip to MET)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false across all aggregated arms (the non-circular discipline; flagged_adversarial arms skipped)."
        )
    },
    "flagged_arms_skipped": {
        "principle": "the list of arms skipped for flagged_adversarial=true (never aggregated into a headline)."
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads D1-D4 + cascade JSON, no LLM; 0.0001s floor)."
        )
    },
    "cited_upstream_artifacts": {
        "principle": (
            "the {experiment_id, fields_imported, sha256} for each arm so the verdict is traceable to a real "
            "measurement."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "moat_realized",
    "moat_retired_bounded",
    "execution_incomplete_arms",
    "best_arm",
    "efficiency_win",
    "per_arm_table",
    "diffusiongemma_gate_conditions_satisfied_off_arc",
    "verifier_is_oracle",
    "flagged_arms_skipped",
    "inference_substrate",
    "cited_upstream_artifacts",
    "paper_summary",
    "decision",
    "duration_s",
    "field_principles",
    "spec_refs",
    "diffusiongemma_gate_status",
    "diffusiongemma_activation",
    "missing_upstream_artifacts",
    "reproducibility_checksum",
)

BASELINE_SPEC = ArtifactSpec(
    "B1",
    "genuine-tuned-SC",
    5015,
    BASELINE_ARTIFACT_RELATIVE_PATH,
    accuracy_field="genuine_tuned_sc_accuracy",
    oracle_at_k_field="oracle_at_k",
)
D1_SPEC = ArtifactSpec(
    "D1",
    "LoRA-EBM",
    5031,
    D1_ARTIFACT_RELATIVE_PATH,
    "trained_scorer_accuracy",
    "genuine_tuned_sc_accuracy",
    "delta_vs_tuned_sc",
    "paired_ci95",
    "mcnemar_p",
    None,
    "oracle_at_k",
)
D2_SPEC = ArtifactSpec(
    "D2",
    "uPRM",
    5032,
    D2_ARTIFACT_RELATIVE_PATH,
    "uprm_selection_accuracy",
    "genuine_tuned_sc_accuracy",
    "delta_vs_tuned_sc",
    "paired_ci95",
    "mcnemar_p",
    None,
    "oracle_at_k",
)
D3_SPEC = ArtifactSpec(
    "D3",
    "EBRM",
    5033,
    D3_ARTIFACT_RELATIVE_PATH,
    "ebrm_selection_accuracy",
    "genuine_tuned_sc_accuracy",
    "delta_vs_tuned_sc",
    "paired_ci95",
    "mcnemar_p",
    None,
    "oracle_at_k",
)
D6_SPEC = ArtifactSpec(
    "D6",
    "cascade",
    5034,
    D6_ARTIFACT_RELATIVE_PATH,
    "cascade_accuracy",
    "genuine_tuned_sc_accuracy",
    None,
    "paired_ci95_cascade_vs_judge",
    None,
    None,
    None,
)
D4_SPEC = ArtifactSpec(
    "D4",
    "second-corpus-confirmation",
    5035,
    D4_ARTIFACT_RELATIVE_PATH,
    "second_corpus_accuracy",
    "genuine_tuned_sc_accuracy_second",
    "delta_vs_tuned_sc_second",
    "paired_ci95_second",
    "mcnemar_p_second",
    "second_corpus",
    "oracle_at_k_second",
)
ALL_SPECS = (BASELINE_SPEC, D1_SPEC, D2_SPEC, D3_SPEC, D6_SPEC, D4_SPEC)
ACCURACY_SPECS = (D1_SPEC, D2_SPEC, D3_SPEC)
ACCURACY_SOURCE_EXPERIMENT_IDS = {spec.experiment_id for spec in ACCURACY_SPECS}
ARM_BY_ID = {spec.arm_id: spec.arm for spec in (*ACCURACY_SPECS, D6_SPEC, D4_SPEC)}

_number = base._number
_rounded = base._rounded
_ci95 = base._ci95
_ci_excludes_zero_positive = base._ci_excludes_zero_positive
_ci_includes_zero = base._ci_includes_zero
_slug = base._slug
_format_delta = base._format_delta
_bool_or_false = base._bool_or_false
_payload_blocked = base._payload_blocked
_json_dumps = base._json_dumps
_read_json = base._read_json
_sha256_file = base._sha256_file
write_json = base.write_json
_accuracy_execution_status = base._accuracy_execution_status
_row_is_positive = base._row_is_positive
_row_is_clean_null = base._row_is_clean_null


def _fields_for_spec(spec: ArtifactSpec, payload: Mapping[str, Any], flagged: bool) -> list[str]:
    if flagged:
        return [field for field in ("flagged_adversarial", "honest_verdict") if field in payload]
    if spec.arm_id == "B1":
        wanted = (
            "honest_verdict",
            "flagged_adversarial",
            "genuine_tuned_sc_accuracy",
            "genuine_headroom_present",
            "oracle_at_k",
        )
    elif spec.arm_id == "D6":
        wanted = (
            "honest_verdict",
            "flagged_adversarial",
            "verifier_is_oracle",
            "cascade_accuracy",
            "judge_only_accuracy",
            "paired_ci95_cascade_vs_judge",
            "judge_call_fraction",
            "cascade_judge_calls",
            "judge_only_calls",
            "n_questions",
        )
    elif spec.arm_id == "D4":
        wanted = (
            "honest_verdict",
            "flagged_adversarial",
            "verifier_is_oracle",
            "headroom_present",
            "best_verifier_from",
            "second_corpus",
            "second_corpus_accuracy",
            "genuine_tuned_sc_accuracy_second",
            "delta_vs_tuned_sc_second",
            "paired_ci95_second",
            "mcnemar_p_second",
            "n_questions",
        )
    else:
        wanted = (
            "honest_verdict",
            "flagged_adversarial",
            "verifier_is_oracle",
            "headroom_present",
            spec.accuracy_field,
            spec.tuned_sc_field,
            spec.delta_field,
            spec.ci95_field,
            spec.mcnemar_field,
            "scorer_trained",
            "abstention_rate",
            "n_questions",
        )
    return [field for field in wanted if field and field in payload]


def _citation(spec: ArtifactSpec, path: Path, payload: Mapping[str, Any]) -> JsonDict:
    flagged = payload.get("flagged_adversarial") is True
    return {
        "arm_id": spec.arm_id,
        "arm": spec.arm,
        "experiment_id": spec.experiment_id,
        "path": path.as_posix(),
        "fields_imported": _fields_for_spec(spec, payload, flagged),
        "sha256": _sha256_file(path),
    }


def _flagged_skip(spec: ArtifactSpec, path: Path, payload: Mapping[str, Any]) -> JsonDict:
    return {
        "arm_id": spec.arm_id,
        "arm": spec.arm,
        "experiment_id": spec.experiment_id,
        "path": path.as_posix(),
        "honest_verdict": payload.get("honest_verdict"),
    }


def _accuracy_row(spec: ArtifactSpec, path: Path, payload: Mapping[str, Any]) -> JsonDict:
    return base._accuracy_row(spec, path, payload)


def _d4_row(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    delta = _number(payload.get(D4_SPEC.delta_field))
    ci95 = _ci95(payload.get(D4_SPEC.ci95_field))
    mcnemar_p = _number(payload.get(D4_SPEC.mcnemar_field))
    blocked = _payload_blocked(payload) or delta is None or ci95 is None or mcnemar_p is None
    arm_id = "D4" if blocked else str(payload.get("best_verifier_from") or "D4")
    row = {
        "arm_id": arm_id,
        "arm": ARM_BY_ID.get(arm_id, arm_id),
        "source_experiment_id": D4_SPEC.experiment_id,
        "source_artifact": path.as_posix(),
        "corpus": str(payload.get("second_corpus") or "second_corpus_unknown"),
        "selection_accuracy": _rounded(_number(payload.get(D4_SPEC.accuracy_field))),
        "genuine_tuned_sc_accuracy": _rounded(_number(payload.get(D4_SPEC.tuned_sc_field))),
        "delta_vs_tuned_sc": _rounded(delta),
        "paired_ci95": ci95,
        "mcnemar_p": _rounded(mcnemar_p),
        "verifier_is_oracle": _bool_or_false(payload.get("verifier_is_oracle")),
        "headroom_present": _bool_or_false(payload.get("headroom_present")),
        "oracle_at_k": _rounded(_number(payload.get(D4_SPEC.oracle_at_k_field))),
        "n_questions": int(_number(payload.get("n_questions")) or 0),
        "scorer_trained": None,
        "abstention_rate": None,
        "execution_status": "blocked" if blocked else "clean",
        "confirmation_source": "D4",
        "honest_verdict": payload.get("honest_verdict"),
    }
    row["win_vs_tuned_sc"] = _row_is_positive(row)
    return row


def _cascade_row(path: Path, payload: Mapping[str, Any], baseline_accuracy: float | None, baseline_headroom: bool) -> JsonDict:
    cascade_accuracy = _number(payload.get("cascade_accuracy"))
    judge_only_accuracy = _number(payload.get("judge_only_accuracy"))
    judge_call_fraction = _number(payload.get("judge_call_fraction"))
    ci95 = _ci95(payload.get("paired_ci95_cascade_vs_judge"))
    delta_vs_tuned_sc = (
        cascade_accuracy - baseline_accuracy
        if cascade_accuracy is not None and baseline_accuracy is not None
        else None
    )
    delta_vs_judge = (
        cascade_accuracy - judge_only_accuracy
        if cascade_accuracy is not None and judge_only_accuracy is not None
        else None
    )
    execution_status = (
        "blocked"
        if _payload_blocked(payload) or cascade_accuracy is None or judge_only_accuracy is None or judge_call_fraction is None
        else "clean"
    )
    row = {
        "arm_id": "D6",
        "arm": "cascade",
        "source_experiment_id": D6_SPEC.experiment_id,
        "source_artifact": path.as_posix(),
        "corpus": "MuSR",
        "selection_accuracy": _rounded(cascade_accuracy),
        "genuine_tuned_sc_accuracy": _rounded(baseline_accuracy),
        "delta_vs_tuned_sc": _rounded(delta_vs_tuned_sc),
        "delta_vs_judge_only": _rounded(delta_vs_judge),
        "paired_ci95": ci95,
        "mcnemar_p": None,
        "verifier_is_oracle": _bool_or_false(payload.get("verifier_is_oracle")),
        "headroom_present": bool(baseline_headroom),
        "oracle_at_k": None,
        "n_questions": int(_number(payload.get("n_questions")) or 0),
        "judge_call_fraction": _rounded(judge_call_fraction),
        "cascade_judge_calls": int(_number(payload.get("cascade_judge_calls")) or 0),
        "judge_only_calls": int(_number(payload.get("judge_only_calls")) or 0),
        "scorer_trained": None,
        "abstention_rate": None,
        "execution_status": execution_status,
        "honest_verdict": payload.get("honest_verdict"),
    }
    row["efficiency_win"] = _row_is_efficiency_win(row)
    row["win_vs_tuned_sc"] = False
    return row


def _row_is_efficiency_win(row: Mapping[str, Any]) -> bool:
    fraction = _number(row.get("judge_call_fraction"))
    return (
        row.get("execution_status") == "clean"
        and row.get("verifier_is_oracle") is False
        and fraction is not None
        and 0.0 <= fraction < MATERIAL_JUDGE_FRACTION
        and _ci_includes_zero(row.get("paired_ci95"))
    )


def _row_for_spec(
    spec: ArtifactSpec,
    path: Path,
    payload: Mapping[str, Any],
    *,
    baseline_accuracy: float | None,
    baseline_headroom: bool,
) -> JsonDict | None:
    if spec.arm_id == "B1":
        return None
    if spec.arm_id == "D4":
        return _d4_row(path, payload)
    if spec.arm_id == "D6":
        return _cascade_row(path, payload, baseline_accuracy, baseline_headroom)
    return _accuracy_row(spec, path, payload)


def _baseline_context(payload: Mapping[str, Any] | None) -> tuple[float | None, bool]:
    if not isinstance(payload, Mapping):
        return GENUINE_TUNED_SC_ACCURACY, True
    return (
        _number(payload.get("genuine_tuned_sc_accuracy")) or GENUINE_TUNED_SC_ACCURACY,
        _bool_or_false(payload.get("genuine_headroom_present")),
    )


def load_upstream_artifacts(root: Path) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    rows: list[JsonDict] = []
    flagged: list[JsonDict] = []
    citations: list[JsonDict] = []
    missing: list[JsonDict] = []
    loaded_payloads: dict[str, JsonMap] = {}

    for spec in ALL_SPECS:
        path = root / spec.relative_path
        if not path.exists():
            missing.append({"arm_id": spec.arm_id, "arm": spec.arm, "experiment_id": spec.experiment_id, "path": path.as_posix()})
            continue
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            missing.append(
                {
                    "arm_id": spec.arm_id,
                    "arm": spec.arm,
                    "experiment_id": spec.experiment_id,
                    "path": path.as_posix(),
                    "error": "artifact is not a JSON object",
                }
            )
            continue
        loaded_payloads[spec.arm_id] = payload
        citations.append(_citation(spec, path, payload))

    baseline_accuracy, baseline_headroom = _baseline_context(loaded_payloads.get("B1"))
    for spec in ALL_SPECS:
        payload = loaded_payloads.get(spec.arm_id)
        if payload is None:
            continue
        path = root / spec.relative_path
        if payload.get("flagged_adversarial") is True:
            if spec.arm_id != "B1":
                flagged.append(_flagged_skip(spec, path, payload))
            continue
        row = _row_for_spec(
            spec,
            path,
            payload,
            baseline_accuracy=baseline_accuracy,
            baseline_headroom=baseline_headroom,
        )
        if row is not None:
            rows.append(row)
    return rows, flagged, citations, missing


def _best_accuracy_row(rows: Sequence[JsonMap]) -> JsonDict | None:
    candidates = [
        row
        for row in rows
        if str(row.get("arm_id")) in {"D1", "D2", "D3"}
        and row.get("verifier_is_oracle") is False
        and _number(row.get("delta_vs_tuned_sc")) is not None
    ]
    if not candidates:
        return None
    return dict(max(candidates, key=lambda row: float(row["delta_vs_tuned_sc"])))


def _confirmed_accuracy_positive(rows: Sequence[JsonMap]) -> JsonDict | None:
    positive_musr = [
        row
        for row in rows
        if str(row.get("arm_id")) in {"D1", "D2", "D3"}
        and int(row.get("source_experiment_id") or 0) in ACCURACY_SOURCE_EXPERIMENT_IDS
        and _row_is_positive(row)
    ]
    positive_ids = {str(row.get("arm_id")) for row in positive_musr}
    positive_second = [
        row
        for row in rows
        if int(row.get("source_experiment_id") or 0) == D4_SPEC.experiment_id
        and str(row.get("arm_id")) in positive_ids
        and _row_is_positive(row)
    ]
    if not positive_second:
        return None
    confirmed_ids = {str(row.get("arm_id")) for row in positive_second}
    candidates = [row for row in (*positive_musr, *positive_second) if str(row.get("arm_id")) in confirmed_ids]
    return dict(max(candidates, key=lambda row: float(row["delta_vs_tuned_sc"])))


def _efficiency_positive(rows: Sequence[JsonMap]) -> JsonDict | None:
    wins = [row for row in rows if str(row.get("arm_id")) == "D6" and _row_is_efficiency_win(row)]
    return dict(wins[0]) if wins else None


def _d1_d2_retired(rows: Sequence[JsonMap]) -> bool:
    for arm_id in ("D1", "D2"):
        arm_rows = [row for row in rows if str(row.get("arm_id")) == arm_id]
        if not arm_rows or not all(_row_is_clean_null(row) for row in arm_rows):
            return False
    return True


def _is_incomplete_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("execution_status")) not in {"clean"}


def _execution_incomplete_arms(rows: Sequence[JsonMap]) -> list[JsonDict]:
    incomplete: list[JsonDict] = []
    for row in rows:
        if not _is_incomplete_row(row):
            continue
        incomplete.append(
            {
                "arm_id": row.get("arm_id"),
                "arm": row.get("arm"),
                "corpus": row.get("corpus"),
                "execution_status": row.get("execution_status"),
                "honest_verdict": row.get("honest_verdict"),
                "source_experiment_id": row.get("source_experiment_id"),
            }
        )
    return incomplete


def _first_incomplete_slug(incomplete: Sequence[JsonMap]) -> str:
    preferred = {str(item.get("arm_id")): item for item in incomplete}
    for arm_id in ("D1", "D2", "D3", "D6", "D4"):
        if arm_id in preferred:
            return _slug(preferred[arm_id].get("arm"))
    return _slug(incomplete[0].get("arm")) if incomplete else "unknown"


def _mixed_verdict(rows: Sequence[JsonMap]) -> str:
    clean_headroom = [
        row
        for row in rows
        if row.get("execution_status") == "clean"
        and row.get("verifier_is_oracle") is False
        and row.get("headroom_present") is True
    ]
    if not clean_headroom:
        return "complete_moat_scoped_no_headroom_present_false_negative_risk"
    if any(str(row.get("arm_id")) in {"D1", "D2", "D3"} and _row_is_positive(row) for row in clean_headroom):
        return "complete_moat_scoped_positive_musr_no_cross_corpus_confirm"
    return "complete_moat_scoped_no_realized_no_bounded_retirement"


def _metric_text(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "no clean numeric arm"
    if row.get("arm") == "cascade":
        return (
            f"cascade on {row.get('corpus')} judge_call_fraction={row.get('judge_call_fraction')}, "
            f"delta_vs_judge_only={_format_delta(_number(row.get('delta_vs_judge_only')))}, "
            f"CI95={row.get('paired_ci95')}"
        )
    return (
        f"{row.get('arm')} on {row.get('corpus')} "
        f"delta_vs_tuned_sc={_format_delta(_number(row.get('delta_vs_tuned_sc')))}, "
        f"CI95={row.get('paired_ci95')}, McNemar p={row.get('mcnemar_p')}"
    )


def _decision(rows: Sequence[JsonMap]) -> tuple[str, str, JsonDict | None, bool, list[JsonDict]]:
    incomplete = _execution_incomplete_arms(rows)
    if not rows:
        return "BLOCKED-NO-MOAT-ARMS", "blocked_no_moat_arms", None, False, []
    efficiency = _efficiency_positive(rows)
    if efficiency is not None:
        verdict = (
            "success_moat_realized_off_arc_cascade_musr_efficiency_"
            f"{_format_delta(_number(efficiency.get('delta_vs_judge_only')))}"
        )
        return "POSITIVE", verdict, efficiency, True, incomplete
    confirmed = _confirmed_accuracy_positive(rows)
    if confirmed is not None:
        verdict = (
            "success_moat_realized_off_arc_"
            f"{_slug(confirmed.get('arm'))}_{_slug(confirmed.get('corpus'))}_"
            f"{_format_delta(_number(confirmed.get('delta_vs_tuned_sc')))}"
        )
        return "POSITIVE", verdict, confirmed, False, incomplete
    if incomplete:
        verdict = f"complete_moat_execution_incomplete_{_first_incomplete_slug(incomplete)}"
        return "EXECUTION-INCOMPLETE", verdict, _best_accuracy_row(rows), False, incomplete
    if _d1_d2_retired(rows):
        return (
            "BOUNDED-RETIRE",
            "complete_moat_retired_bounded_lora_ebm_and_uprm_both_null",
            _best_accuracy_row(rows),
            False,
            incomplete,
        )
    return "MIXED-SCOPED", _mixed_verdict(rows), _best_accuracy_row(rows), False, []


def _paper_summary(decision: str, best: Mapping[str, Any] | None, incomplete: Sequence[JsonMap], flagged: Sequence[JsonMap]) -> str:
    if decision == "BLOCKED-NO-MOAT-ARMS":
        return (
            "D5 is blocked: every available moat arm was absent, malformed, or flagged_adversarial=true, "
            "so no non-fabricated verifier-moat result was aggregated."
        )
    if decision == "POSITIVE":
        return (
            "The off-ARC verifier moat is realized on the tested domain: "
            f"{_metric_text(best)} satisfies the non-circular gate. DiffusionGemma gate conditions are "
            "satisfied off-ARC, but ARC's ~13pp headroom remains the canonical uncaptured target and "
            "activation remains operator-gated."
        )
    if decision == "BOUNDED-RETIRE":
        return (
            "The off-ARC accuracy moat retires as bounded for the tested D1/D2 constructions: both "
            "properly executed LoRA-EBM and uPRM are clean nulls on headroom-present oracle-distinct evidence, "
            "and the cascade has no efficiency Pareto win."
        )
    if decision == "EXECUTION-INCOMPLETE":
        arms = ", ".join(str(item.get("arm")) for item in incomplete)
        skipped = ", ".join(str(item.get("arm_id")) for item in flagged) or "none"
        return (
            f"D5 is execution-incomplete: {arms} did not cleanly execute, while flagged artifacts skipped were "
            f"{skipped}. This is not a clean null and cannot bound-retire the moat."
        )
    return (
        "D5 is mixed/scoped: available clean evidence did not provide a cross-corpus accuracy win, an "
        f"efficiency Pareto win, or the D1/D2 clean-null retirement condition. Best row: {_metric_text(best)}."
    )


def reproducibility_checksum(payload: JsonMap) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def build_artifact(
    *,
    rows: Sequence[JsonDict],
    flagged: Sequence[JsonDict],
    citations: Sequence[JsonDict],
    missing: Sequence[JsonDict],
    duration_s: float,
) -> JsonDict:
    decision, honest_verdict, best, efficiency_win, incomplete = _decision(rows)
    moat_realized = decision == "POSITIVE"
    artifact: JsonDict = {
        "experiment": "experiment_5036_moat_gate_resolution_v3",
        "experiment_id": EXPERIMENT_ID,
        "schema": "carnot.experiment_5036_moat_gate_resolution_v3.v1",
        "honest_verdict": honest_verdict,
        "decision": decision,
        "moat_realized": moat_realized,
        "moat_retired_bounded": decision == "BOUNDED-RETIRE",
        "execution_incomplete_arms": list(incomplete),
        "best_arm": best,
        "efficiency_win": bool(efficiency_win),
        "per_arm_table": list(rows),
        "diffusiongemma_gate_conditions_satisfied_off_arc": moat_realized,
        "diffusiongemma_gate_status": "conditions_satisfied_off_arc_operator_gated" if moat_realized else "STILL-PENDING",
        "diffusiongemma_activation": "operator_gated_not_flipped" if moat_realized else "not_activated",
        "verifier_is_oracle": False,
        "flagged_arms_skipped": list(flagged),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "cited_upstream_artifacts": list(citations),
        "missing_upstream_artifacts": list(missing),
        "paper_summary": _paper_summary(decision, best, incomplete, flagged),
        "duration_s": round(max(float(duration_s), 0.0001), 6),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(
        {
            "decision": artifact["decision"],
            "honest_verdict": artifact["honest_verdict"],
            "rows": artifact["per_arm_table"],
            "flagged": artifact["flagged_arms_skipped"],
            "missing": artifact["missing_upstream_artifacts"],
            "citations": artifact["cited_upstream_artifacts"],
        }
    )
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    for field in (
        "moat_realized",
        "moat_retired_bounded",
        "efficiency_win",
        "diffusiongemma_gate_conditions_satisfied_off_arc",
        "verifier_is_oracle",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate")
    if artifact.get("diffusiongemma_gate_status") == "MET":
        errors.append("diffusiongemma_gate_status")
    if artifact.get("decision") not in {"POSITIVE", "BOUNDED-RETIRE", "EXECUTION-INCOMPLETE", "MIXED-SCOPED", "BLOCKED-NO-MOAT-ARMS"}:
        errors.append("decision")
    if not str(artifact.get("honest_verdict", "")).startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict")
    for field in (
        "execution_incomplete_arms",
        "per_arm_table",
        "flagged_arms_skipped",
        "cited_upstream_artifacts",
        "missing_upstream_artifacts",
    ):
        if not isinstance(artifact.get(field), list):
            errors.append(field)
    if artifact.get("best_arm") is not None and not isinstance(artifact.get("best_arm"), dict):
        errors.append("best_arm")
    if not isinstance(artifact.get("paper_summary"), str) or not artifact.get("paper_summary"):
        errors.append("paper_summary")
    if not isinstance(artifact.get("duration_s"), (int, float)) or float(artifact.get("duration_s") or 0.0) < 0.0001:
        errors.append("duration_s")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    now: Any = time.time,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    rows, flagged, citations, missing = load_upstream_artifacts(root)
    artifact = build_artifact(
        rows=rows,
        flagged=flagged,
        citations=citations,
        missing=missing,
        duration_s=float(now()) - start,
    )
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    _ = argv
    artifact_path = REPO_ROOT / RESULT_RELATIVE_PATH
    artifact = run(artifact_path=artifact_path)
    errors = artifact_schema_errors(artifact)
    if errors:
        print(f"schema_errors={errors}")
        return 1
    print(artifact_path.as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
