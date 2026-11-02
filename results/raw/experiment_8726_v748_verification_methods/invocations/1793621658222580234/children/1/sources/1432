"""Exp 5050: resolve the .464 verifier-moat gate from Phase D artifacts.

Spec refs: REQ-REPORT-5050, SCENARIO-REPORT-5050.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5050
EXPERIMENT_NAME = "experiment_5050_moat_gate_resolution_v464"
SCHEMA = "carnot.experiment_5050_moat_gate_resolution_v464.v1"
RESULT_RELATIVE_PATH = "results/experiment_5050_moat_gate_resolution_v464.json"
D1_ARTIFACT_RELATIVE_PATH = "results/experiment_5045_powered_lora_ebm_eorm_musr.json"
D2_ARTIFACT_RELATIVE_PATH = "results/experiment_5046_vpr_process_reward_repair.json"
D3_ARTIFACT_RELATIVE_PATH = "results/experiment_5047_kan_purm_energy_calibration.json"
D6_EXPECTED_ARTIFACT_RELATIVE_PATH = "results/experiment_5048_cross_model_cascade_repair.json"
D6_AVAILABLE_ARTIFACT_RELATIVE_PATH = "results/experiment_5048_d6.json"
D4_ARTIFACT_RELATIVE_PATH = "results/experiment_5049_second_corpus_confirmation.json"
PRIOR_GATE_ARTIFACT_RELATIVE_PATH = "results/experiment_5036_moat_gate_resolution_v3.json"
SPEC_REFS = ["REQ-REPORT-5050", "SCENARIO-REPORT-5050"]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
MATERIAL_JUDGE_FRACTION = 0.8
REQUIRED_FIELDS = (
    "honest_verdict",
    "moat_state",
    "best_arm",
    "best_arm_delta",
    "best_arm_ci",
    "second_corpus_confirmed",
    "cascade_efficiency_win",
    "execution_incomplete_reasons",
    "bounded_retirement_ok",
    "next_actions",
)
MOAT_STATES = {"moat_realized", "musr_scoped_positive", "retired_bounded", "execution_incomplete"}

MUSR_ARM_SPECS: dict[str, JsonDict] = {
    "D1": {
        "arm": "powered_lora_ebm_eorm",
        "experiment_id": 5045,
        "relative_path": D1_ARTIFACT_RELATIVE_PATH,
        "accuracy_fields": ("powered_lora_ebm_accuracy", "trained_scorer_accuracy"),
        "availability_field": "powered_scorer_available",
        "trained_field": "scorer_trained",
    },
    "D2": {
        "arm": "vpr_process_reward",
        "experiment_id": 5046,
        "relative_path": D2_ARTIFACT_RELATIVE_PATH,
        "accuracy_fields": ("process_reward_accuracy",),
        "availability_field": "process_reward_available",
    },
    "D3": {
        "arm": "kan_purm_energy_calibration",
        "experiment_id": 5047,
        "relative_path": D3_ARTIFACT_RELATIVE_PATH,
        "accuracy_fields": ("calibrated_accuracy",),
        "availability_field": "calibration_available",
    },
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix summarizing the .464 moat_state without treating blocked arms as nulls."
    },
    "moat_state": {
        "principle": "one of moat_realized, musr_scoped_positive, retired_bounded, execution_incomplete."
    },
    "best_arm": {
        "principle": "the strongest usable MuSR verifier arm, or cascade when the cascade efficiency gate wins."
    },
    "best_arm_delta": {
        "principle": "best_arm signed delta: verifier-vs-tuned-SC for D1/D2/D3 or cascade-vs-judge for D6."
    },
    "best_arm_ci": {
        "principle": "paired CI95 for best_arm on the gate metric."
    },
    "second_corpus_confirmed": {
        "principle": "true only for a non-flagged clean Exp 5049 confirmation matching the winning MuSR arm."
    },
    "cascade_efficiency_win": {
        "principle": "true only for clean judge-parity CI with materially fewer judge calls."
    },
    "execution_incomplete_reasons": {
        "principle": "concrete missing, blocked, flagged, critical-corrigendum, malformed, or unusable arms."
    },
    "bounded_retirement_ok": {
        "principle": "true only when clean D1/D2/D3 no-wins and a clean no-efficiency cascade are all present."
    },
    "next_actions": {
        "principle": "state-conditioned follow-up actions for the next conductor step."
    },
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> JsonDict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - exercised by missing path handling.
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _ci95(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    low = _number(value[0])
    high = _number(value[1])
    if low is None or high is None:
        return None
    return [round(low, 6), round(high, 6)]


def _ci_excludes_zero_positive(ci95: Sequence[float] | None) -> bool:
    return bool(ci95 and len(ci95) == 2 and float(ci95[0]) > 0.0 and float(ci95[1]) > 0.0)


def _ci_includes_zero(ci95: Sequence[float] | None) -> bool:
    return bool(ci95 and len(ci95) == 2 and float(ci95[0]) <= 0.0 <= float(ci95[1]))


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _blocked(payload: JsonMap) -> bool:
    return str(payload.get("honest_verdict", "")).startswith("blocked_") or payload.get("status") == "blocked"


def _critical_corrigendum(payload: JsonMap) -> bool:
    for item in payload.get("corrigendum_pending") or []:
        if isinstance(item, Mapping) and item.get("severity") == "critical":
            return True
    return False


def _status(payload: JsonMap, required_ok: bool = True) -> str:
    if payload.get("flagged_adversarial") is True:
        return "flagged"
    if _critical_corrigendum(payload):
        return "critical_corrigendum"
    if _blocked(payload):
        return "blocked"
    return "clean" if required_ok else "incomplete"


def _accuracy_from(payload: JsonMap, fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _number(payload.get(field))
        if value is not None:
            return value
    return None


def _musr_summary(arm_id: str, path: Path, payload: JsonMap) -> JsonDict:
    spec = MUSR_ARM_SPECS[arm_id]
    delta = _number(payload.get("delta_vs_tuned_sc"))
    ci95 = _ci95(payload.get("paired_ci95"))
    p_value = _number(payload.get("mcnemar_p"))
    availability_ok = payload.get(spec["availability_field"]) is not False
    trained_ok = payload.get(spec.get("trained_field", spec["availability_field"])) is not False
    required_ok = delta is not None and ci95 is not None and p_value is not None and availability_ok and trained_ok
    execution_status = _status(payload, required_ok)
    row = {
        "arm_id": arm_id,
        "arm": spec["arm"],
        "experiment_id": spec["experiment_id"],
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "honest_verdict": payload.get("honest_verdict"),
        "execution_status": execution_status,
        "accuracy": _rounded(_accuracy_from(payload, spec["accuracy_fields"])),
        "delta_vs_tuned_sc": _rounded(delta),
        "paired_ci95": ci95,
        "mcnemar_p": _rounded(p_value),
        "headroom_present": payload.get("headroom_present") is True,
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
        "n_questions": int(_number(payload.get("n_questions")) or 0),
    }
    row["proper_musr_win"] = _proper_musr_win(row)
    row["clean_no_win"] = _clean_no_win(row)
    return row


def _cascade_summary(path: Path, payload: JsonMap) -> JsonDict:
    cascade_accuracy = _number(payload.get("cascade_accuracy"))
    judge_only_accuracy = _number(payload.get("judge_only_accuracy"))
    judge_fraction = _number(payload.get("judge_call_fraction"))
    ci95 = _ci95(payload.get("paired_ci95_cascade_vs_judge"))
    required_ok = cascade_accuracy is not None and judge_only_accuracy is not None and judge_fraction is not None and ci95 is not None
    delta_vs_judge = cascade_accuracy - judge_only_accuracy if cascade_accuracy is not None and judge_only_accuracy is not None else None
    row = {
        "arm_id": "D6",
        "arm": "cross_model_cascade",
        "experiment_id": 5048,
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "honest_verdict": payload.get("honest_verdict"),
        "execution_status": _status(payload, required_ok),
        "cascade_accuracy": _rounded(cascade_accuracy),
        "judge_only_accuracy": _rounded(judge_only_accuracy),
        "delta_vs_judge_only": _rounded(delta_vs_judge),
        "paired_ci95": ci95,
        "judge_call_fraction": _rounded(judge_fraction),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
        "n_questions": int(_number(payload.get("n_questions")) or 0),
    }
    row["efficiency_win"] = _cascade_efficiency_win(row)
    return row


def _second_corpus_summary(path: Path, payload: JsonMap) -> JsonDict:
    delta = _number(payload.get("delta_vs_tuned_sc_second"))
    ci95 = _ci95(payload.get("paired_ci95_second"))
    p_value = _number(payload.get("mcnemar_p_second"))
    required_ok = delta is not None and ci95 is not None and p_value is not None
    return {
        "arm_id": "D4",
        "arm": "second_corpus_confirmation",
        "experiment_id": 5049,
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "honest_verdict": payload.get("honest_verdict"),
        "execution_status": _status(payload, required_ok),
        "best_arm": payload.get("best_arm"),
        "second_corpus_confirmed": payload.get("second_corpus_confirmed") is True,
        "second_corpus_name": payload.get("second_corpus_name"),
        "delta_vs_tuned_sc_second": _rounded(delta),
        "paired_ci95_second": ci95,
        "mcnemar_p_second": _rounded(p_value),
        "headroom_present": payload.get("headroom_present") is True,
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _artifact_status_record(arm_id: str, arm: str, path: Path, payload: JsonMap) -> JsonDict:
    return {
        "arm_id": arm_id,
        "arm": arm,
        "path": path.as_posix(),
        "honest_verdict": payload.get("honest_verdict"),
        "status": payload.get("status"),
    }


def load_phase_d_artifacts(root: Path) -> JsonDict:
    root = Path(root)
    musr_rows: list[JsonDict] = []
    missing: list[JsonDict] = []
    blocked: list[JsonDict] = []
    flagged: list[JsonDict] = []
    critical: list[JsonDict] = []
    malformed: list[JsonDict] = []
    citations: list[JsonDict] = []

    for arm_id, spec in MUSR_ARM_SPECS.items():
        path = root / str(spec["relative_path"])
        payload = _read_json_object(path) if path.exists() else None
        if payload is None:
            missing.append({"arm_id": arm_id, "arm": spec["arm"], "path": path.as_posix()})
            continue
        row = _musr_summary(arm_id, path, payload)
        musr_rows.append(row)
        citations.append({"arm_id": arm_id, "path": path.as_posix(), "sha256": row["sha256"]})
        if row["execution_status"] == "blocked":
            blocked.append(_artifact_status_record(arm_id, spec["arm"], path, payload))
        elif row["execution_status"] == "flagged":
            flagged.append(_artifact_status_record(arm_id, spec["arm"], path, payload))
        elif row["execution_status"] == "critical_corrigendum":
            critical.append(_artifact_status_record(arm_id, spec["arm"], path, payload))
        elif row["execution_status"] == "incomplete":
            malformed.append(_artifact_status_record(arm_id, spec["arm"], path, payload))

    expected_d6_path = root / D6_EXPECTED_ARTIFACT_RELATIVE_PATH
    available_d6_path = root / D6_AVAILABLE_ARTIFACT_RELATIVE_PATH
    d6_path = expected_d6_path if expected_d6_path.exists() else available_d6_path
    if not expected_d6_path.exists():
        missing.append({"arm_id": "D6", "arm": "cross_model_cascade", "path": expected_d6_path.as_posix()})
    d6_payload = _read_json_object(d6_path) if d6_path.exists() else None
    cascade_row: JsonDict | None = None
    if d6_payload is not None:
        cascade_row = _cascade_summary(d6_path, d6_payload)
        citations.append({"arm_id": "D6", "path": d6_path.as_posix(), "sha256": cascade_row["sha256"]})
        if cascade_row["execution_status"] == "blocked":
            blocked.append(_artifact_status_record("D6", "cross_model_cascade", d6_path, d6_payload))
        elif cascade_row["execution_status"] == "flagged":
            flagged.append(_artifact_status_record("D6", "cross_model_cascade", d6_path, d6_payload))
        elif cascade_row["execution_status"] == "critical_corrigendum":
            critical.append(_artifact_status_record("D6", "cross_model_cascade", d6_path, d6_payload))
        elif cascade_row["execution_status"] == "incomplete":
            malformed.append(_artifact_status_record("D6", "cross_model_cascade", d6_path, d6_payload))

    d4_path = root / D4_ARTIFACT_RELATIVE_PATH
    d4_payload = _read_json_object(d4_path) if d4_path.exists() else None
    second_row: JsonDict | None = None
    if d4_payload is None:
        missing.append({"arm_id": "D4", "arm": "second_corpus_confirmation", "path": d4_path.as_posix()})
    else:
        second_row = _second_corpus_summary(d4_path, d4_payload)
        citations.append({"arm_id": "D4", "path": d4_path.as_posix(), "sha256": second_row["sha256"]})
        if second_row["execution_status"] == "blocked":
            blocked.append(_artifact_status_record("D4", "second_corpus_confirmation", d4_path, d4_payload))
        elif second_row["execution_status"] == "flagged":
            flagged.append(_artifact_status_record("D4", "second_corpus_confirmation", d4_path, d4_payload))
        elif second_row["execution_status"] == "critical_corrigendum":
            critical.append(_artifact_status_record("D4", "second_corpus_confirmation", d4_path, d4_payload))
        elif second_row["execution_status"] == "incomplete":
            malformed.append(_artifact_status_record("D4", "second_corpus_confirmation", d4_path, d4_payload))

    prior_path = root / PRIOR_GATE_ARTIFACT_RELATIVE_PATH
    prior_payload = _read_json_object(prior_path) if prior_path.exists() else None
    if prior_payload is None:
        missing.append({"arm_id": "D5-prior", "arm": "prior_gate_resolution", "path": prior_path.as_posix()})
    else:
        citations.append({"arm_id": "D5-prior", "path": prior_path.as_posix(), "sha256": _sha256_file(prior_path)})

    return {
        "musr_rows": musr_rows,
        "cascade_row": cascade_row,
        "second_row": second_row,
        "missing_upstream_artifacts": missing,
        "blocked_upstream_artifacts": blocked,
        "flagged_upstream_artifacts": flagged,
        "critical_corrigendum_artifacts": critical,
        "malformed_upstream_artifacts": malformed,
        "cited_upstream_artifacts": citations,
    }


def _proper_musr_win(row: JsonMap) -> bool:
    delta = _number(row.get("delta_vs_tuned_sc"))
    p_value = _number(row.get("mcnemar_p"))
    return (
        row.get("execution_status") == "clean"
        and row.get("verifier_is_oracle") is False
        and row.get("headroom_present") is True
        and delta is not None
        and delta > 0.0
        and _ci_excludes_zero_positive(row.get("paired_ci95"))
        and p_value is not None
        and p_value < 0.05
    )


def _clean_no_win(row: JsonMap) -> bool:
    return (
        row.get("execution_status") == "clean"
        and row.get("verifier_is_oracle") is False
        and row.get("headroom_present") is True
        and _number(row.get("delta_vs_tuned_sc")) is not None
        and _ci95(row.get("paired_ci95")) is not None
        and _number(row.get("mcnemar_p")) is not None
        and not _proper_musr_win(row)
    )


def _cascade_efficiency_win(row: JsonMap | None) -> bool:
    if row is None:
        return False
    fraction = _number(row.get("judge_call_fraction"))
    return (
        row.get("execution_status") == "clean"
        and row.get("verifier_is_oracle") is False
        and fraction is not None
        and 0.0 <= fraction < MATERIAL_JUDGE_FRACTION
        and _ci_includes_zero(row.get("paired_ci95"))
    )


def _best_musr_row(rows: Sequence[JsonMap]) -> JsonDict | None:
    candidates = [
        dict(row)
        for row in rows
        if row.get("execution_status") not in {"flagged", "critical_corrigendum", "incomplete"}
        and _number(row.get("delta_vs_tuned_sc")) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["delta_vs_tuned_sc"]))


def _confirmed(second_row: JsonMap | None, arm_id: str | None) -> bool:
    if arm_id is None or second_row is None:
        return False
    p_value = _number(second_row.get("mcnemar_p_second"))
    return (
        second_row.get("execution_status") == "clean"
        and second_row.get("second_corpus_confirmed") is True
        and str(second_row.get("best_arm")) == arm_id
        and second_row.get("verifier_is_oracle") is False
        and second_row.get("headroom_present") is True
        and _number(second_row.get("delta_vs_tuned_sc_second")) is not None
        and float(second_row["delta_vs_tuned_sc_second"]) > 0.0
        and _ci_excludes_zero_positive(second_row.get("paired_ci95_second"))
        and p_value is not None
        and p_value < 0.05
    )


def _relevant_missing(missing: Sequence[JsonMap], proper_win: JsonMap | None) -> list[str]:
    relevant = []
    for item in missing:
        arm_id = str(item.get("arm_id"))
        if arm_id in {"D1", "D2", "D3", "D6"} or (arm_id == "D4" and proper_win is not None):
            relevant.append(f"{arm_id} missing: {item.get('path')}")
    return relevant


def _unusable_reasons(loaded: JsonMap, proper_win: JsonMap | None) -> list[str]:
    reasons: list[str] = []
    reasons.extend(_relevant_missing(loaded["missing_upstream_artifacts"], proper_win))
    for key, label in (
        ("blocked_upstream_artifacts", "blocked"),
        ("flagged_upstream_artifacts", "flagged"),
        ("critical_corrigendum_artifacts", "critical_corrigendum"),
        ("malformed_upstream_artifacts", "malformed"),
    ):
        for item in loaded[key]:
            arm_id = str(item.get("arm_id"))
            if arm_id in {"D1", "D2", "D3", "D6"} or (arm_id == "D4" and proper_win is not None):
                detail = item.get("honest_verdict") or item.get("status") or item.get("path")
                reasons.append(f"{arm_id} {label}: {detail}")
    if proper_win is not None and not _confirmed(loaded["second_row"], str(proper_win.get("arm_id"))):
        reasons.append(f"D4 confirmation unavailable for {proper_win.get('arm_id')}")
    return sorted(dict.fromkeys(reasons))


def _bounded_retirement_ok(rows: Sequence[JsonMap], cascade_row: JsonMap | None) -> bool:
    clean_by_arm = {str(row.get("arm_id")): _clean_no_win(row) for row in rows}
    return all(clean_by_arm.get(arm_id) is True for arm_id in ("D1", "D2", "D3")) and (
        cascade_row is not None
        and cascade_row.get("execution_status") == "clean"
        and not _cascade_efficiency_win(cascade_row)
    )


def classify(loaded: JsonMap) -> JsonDict:
    rows = list(loaded["musr_rows"])
    proper_wins = [dict(row) for row in rows if _proper_musr_win(row)]
    proper_win = max(proper_wins, key=lambda row: float(row["delta_vs_tuned_sc"])) if proper_wins else None
    best = proper_win or _best_musr_row(rows)
    cascade_win = _cascade_efficiency_win(loaded["cascade_row"])
    second_confirmed = _confirmed(loaded["second_row"], str(proper_win.get("arm_id")) if proper_win else None)
    bounded_ok = _bounded_retirement_ok(rows, loaded["cascade_row"])
    reasons = _unusable_reasons(loaded, proper_win)

    if cascade_win:
        cascade = dict(loaded["cascade_row"])
        return {
            "moat_state": "moat_realized",
            "honest_verdict": "success_moat_realized_v464_d6_cascade_efficiency",
            "best_arm": "D6",
            "best_arm_delta": cascade.get("delta_vs_judge_only"),
            "best_arm_ci": cascade.get("paired_ci95"),
            "second_corpus_confirmed": False,
            "cascade_efficiency_win": True,
            "bounded_retirement_ok": False,
            "execution_incomplete_reasons": reasons,
        }
    if proper_win is not None and second_confirmed:
        return {
            "moat_state": "moat_realized",
            "honest_verdict": (
                f"success_moat_realized_v464_{_slug(proper_win.get('arm_id'))}_"
                f"{_format_delta(_number(proper_win.get('delta_vs_tuned_sc')))}"
            ),
            "best_arm": proper_win.get("arm_id"),
            "best_arm_delta": proper_win.get("delta_vs_tuned_sc"),
            "best_arm_ci": proper_win.get("paired_ci95"),
            "second_corpus_confirmed": True,
            "cascade_efficiency_win": False,
            "bounded_retirement_ok": False,
            "execution_incomplete_reasons": reasons,
        }
    if proper_win is not None:
        return {
            "moat_state": "musr_scoped_positive",
            "honest_verdict": (
                f"complete_moat_musr_scoped_positive_v464_{_slug(proper_win.get('arm_id'))}_"
                f"{_format_delta(_number(proper_win.get('delta_vs_tuned_sc')))}_no_confirm"
            ),
            "best_arm": proper_win.get("arm_id"),
            "best_arm_delta": proper_win.get("delta_vs_tuned_sc"),
            "best_arm_ci": proper_win.get("paired_ci95"),
            "second_corpus_confirmed": False,
            "cascade_efficiency_win": False,
            "bounded_retirement_ok": False,
            "execution_incomplete_reasons": reasons,
        }
    if bounded_ok:
        return {
            "moat_state": "retired_bounded",
            "honest_verdict": "complete_moat_retired_bounded_v464_clean_d1_d2_d3_no_efficiency",
            "best_arm": best.get("arm_id") if best else None,
            "best_arm_delta": best.get("delta_vs_tuned_sc") if best else None,
            "best_arm_ci": best.get("paired_ci95") if best else None,
            "second_corpus_confirmed": False,
            "cascade_efficiency_win": False,
            "bounded_retirement_ok": True,
            "execution_incomplete_reasons": [],
        }
    return {
        "moat_state": "execution_incomplete",
        "honest_verdict": "complete_moat_execution_incomplete_v464_blocked_or_missing_phase_d",
        "best_arm": best.get("arm_id") if best else None,
        "best_arm_delta": best.get("delta_vs_tuned_sc") if best else None,
        "best_arm_ci": best.get("paired_ci95") if best else None,
        "second_corpus_confirmed": False,
        "cascade_efficiency_win": False,
        "bounded_retirement_ok": False,
        "execution_incomplete_reasons": reasons or ["phase_d_gate_did_not_satisfy_realized_or_bounded_conditions"],
    }


def _next_actions(moat_state: str, reasons: Sequence[str]) -> list[str]:
    if moat_state == "moat_realized":
        return ["Scale the realized verifier construction under operator-gated activation; keep ARC progress separate."]
    if moat_state == "musr_scoped_positive":
        return ["Run a non-flagged second-corpus confirmation for the winning MuSR arm.", "Run the cascade efficiency check if it is still missing or blocked."]
    if moat_state == "retired_bounded":
        return ["Pivot the next milestone to a new verifier direction; D1/D2/D3 are bounded retired for this gate."]
    actions = ["Repair or rerun every blocked, flagged, missing, or critical Phase D arm before claiming a null."]
    if any("D6" in reason for reason in reasons):
        actions.append("Re-run Exp 5048 cascade with the judge-server gate open and explicit cost accounting.")
    if any("D1" in reason for reason in reasons):
        actions.append("Re-run Exp 5045 with the SOTA candidate-refresh gate satisfied or keep cached D1 non-headline.")
    if any("D2" in reason or "D4" in reason for reason in reasons):
        actions.append("Re-run flagged/corrigendum artifacts without critical audit findings before aggregation.")
    return actions


def reproducibility_checksum(payload: JsonMap) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def build_artifact(loaded: JsonMap, duration_s: float) -> JsonDict:
    decision = classify(loaded)
    artifact: JsonDict = {
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(float(duration_s), 0.0001), 6),
        "field_principles": FIELD_PRINCIPLES,
        "per_arm_table": list(loaded["musr_rows"]),
        "cascade_artifact": loaded["cascade_row"],
        "second_corpus_artifact": loaded["second_row"],
        "missing_upstream_artifacts": list(loaded["missing_upstream_artifacts"]),
        "blocked_upstream_artifacts": list(loaded["blocked_upstream_artifacts"]),
        "flagged_upstream_artifacts": list(loaded["flagged_upstream_artifacts"]),
        "critical_corrigendum_artifacts": list(loaded["critical_corrigendum_artifacts"]),
        "malformed_upstream_artifacts": list(loaded["malformed_upstream_artifacts"]),
        "cited_upstream_artifacts": list(loaded["cited_upstream_artifacts"]),
        **decision,
    }
    artifact["next_actions"] = _next_actions(
        str(artifact["moat_state"]),
        artifact["execution_incomplete_reasons"],
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(
        {
            "moat_state": artifact["moat_state"],
            "best_arm": artifact["best_arm"],
            "best_arm_delta": artifact["best_arm_delta"],
            "best_arm_ci": artifact["best_arm_ci"],
            "second_corpus_confirmed": artifact["second_corpus_confirmed"],
            "cascade_efficiency_win": artifact["cascade_efficiency_win"],
            "bounded_retirement_ok": artifact["bounded_retirement_ok"],
            "execution_incomplete_reasons": artifact["execution_incomplete_reasons"],
            "citations": artifact["cited_upstream_artifacts"],
        }
    )
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("moat_state") not in MOAT_STATES:
        errors.append("moat_state")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    for field in ("second_corpus_confirmed", "cascade_efficiency_win", "bounded_retirement_ok"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in ("execution_incomplete_reasons", "next_actions"):
        if not isinstance(artifact.get(field), list):
            errors.append(field)
    if not str(artifact.get("honest_verdict", "")).startswith(("success_", "complete_", "blocked_")):
        errors.append("honest_verdict")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    loaded = load_phase_d_artifacts(root)
    artifact = build_artifact(loaded, float(now()) - start)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    _ = argv
    artifact = run(root=REPO_ROOT, artifact_path=REPO_ROOT / RESULT_RELATIVE_PATH)
    errors = artifact_schema_errors(artifact)
    if errors:
        print(f"schema_errors={errors}")
        return 1
    print((REPO_ROOT / RESULT_RELATIVE_PATH).as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
