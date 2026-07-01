#!/usr/bin/env python3
"""Exp 5063: resolve the .465 verifier-moat gate from audited artifacts.

Spec refs: REQ-REPORT-5063, SCENARIO-REPORT-5063.
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
EXPERIMENT_ID = 5063
EXPERIMENT_NAME = "experiment_5063_moat_gate_resolution_v465"
SCHEMA = "carnot.experiment_5063_moat_gate_resolution_v465.v1"
RESULT_RELATIVE_PATH = "results/experiment_5063_moat_gate_resolution_v465.json"
D1_ARTIFACT_RELATIVE_PATH = "results/experiment_5059_d1_sota_refresh_audit.json"
D4_ARTIFACT_RELATIVE_PATH = "results/experiment_5060_second_corpus_audit_v2.json"
D6_ARTIFACT_RELATIVE_PATH = "results/experiment_5061_tool_first_cascade.json"
GUIDED_ARTIFACT_RELATIVE_PATH = "results/experiment_5062_guided_decoding_cost_frontier.json"
SPEC_REFS = ["REQ-REPORT-5063", "SCENARIO-REPORT-5063"]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

MOAT_STATES = {
    "moat_realized",
    "musr_scoped_positive",
    "second_corpus_scoped_positive",
    "retired_bounded",
    "execution_incomplete",
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "moat_state",
    "best_arm",
    "best_arm_delta",
    "best_arm_ci",
    "second_corpus_confirmed",
    "second_corpus_audit_clean",
    "cascade_efficiency_win",
    "guided_decoding_frontier_state",
    "bounded_retirement_ok",
    "execution_incomplete_reasons",
    "per_arm_table",
    "next_actions",
)

UPSTREAM_SPECS: tuple[JsonDict, ...] = (
    {
        "artifact_id": "D1",
        "arm": "d1_sota_refresh_audit",
        "experiment_id": 5059,
        "relative_path": D1_ARTIFACT_RELATIVE_PATH,
        "required": (
            "honest_verdict",
            "best_arm_available",
            "proper_musr_win",
            "delta_vs_tuned_sc",
            "paired_ci95",
            "mcnemar_p",
            "headroom_present",
            "verifier_is_oracle",
        ),
    },
    {
        "artifact_id": "D4",
        "arm": "d4_second_corpus_audit",
        "experiment_id": 5060,
        "relative_path": D4_ARTIFACT_RELATIVE_PATH,
        "required": (
            "honest_verdict",
            "d4_verdict_class",
            "second_corpus_confirmed",
            "second_corpus_audit_clean",
            "delta_vs_tuned_sc_second",
        ),
    },
    {
        "artifact_id": "D6",
        "arm": "d6_tool_first_cascade",
        "experiment_id": 5061,
        "relative_path": D6_ARTIFACT_RELATIVE_PATH,
        "required": (
            "honest_verdict",
            "cascade_executed",
            "delta_vs_judge_only",
            "paired_ci95",
            "judge_call_fraction",
            "efficiency_win",
            "verifier_is_oracle",
        ),
    },
    {
        "artifact_id": "G1",
        "arm": "guided_decoding_cost_frontier",
        "experiment_id": 5062,
        "relative_path": GUIDED_ARTIFACT_RELATIVE_PATH,
        "required": (
            "honest_verdict",
            "guided_decoding_executed",
            "arms_differentiated",
            "delta_guided_vs_unguided",
            "nfe_by_arm",
        ),
    },
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix for the .465 moat gate without counting blocked, flagged, or unclean evidence as nulls."
    },
    "moat_state": {
        "principle": "one of moat_realized, musr_scoped_positive, second_corpus_scoped_positive, retired_bounded, execution_incomplete."
    },
    "best_arm": {
        "principle": "strongest clean countable evidence; D4 is eligible only when its audit is clean."
    },
    "best_arm_delta": {
        "principle": "signed delta for best_arm on its own gate metric, or null when no clean countable evidence exists."
    },
    "best_arm_ci": {"principle": "paired CI95 for best_arm when the upstream artifact reports one."},
    "second_corpus_confirmed": {
        "principle": "true only when D4 is clean, audit-clean, non-oracle, positive, and confirmed."
    },
    "second_corpus_audit_clean": {
        "principle": "true only when the D4 artifact is clean and reports second_corpus_audit_clean=true."
    },
    "cascade_efficiency_win": {
        "principle": "true only when D6 is clean, non-oracle, and reports efficiency_win=true."
    },
    "guided_decoding_frontier_state": {
        "principle": "status of the clean guided-decoding cost-frontier evidence, never a substitute for D1/D4/D6 moat proof."
    },
    "bounded_retirement_ok": {
        "principle": "true only when D1, D4, and D6 all execute cleanly and null/regress without a proper win."
    },
    "execution_incomplete_reasons": {
        "principle": "concrete missing, blocked, malformed, flagged, or unclean evidence preventing a headline conclusion."
    },
    "per_arm_table": {"principle": "one row per upstream .465 artifact with status and gate-relevant fields."},
    "next_actions": {"principle": "state-conditioned follow-up actions for the next conductor step."},
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> tuple[JsonDict | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - missing paths are handled before reading.
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(payload, Mapping):
        return None, "top-level JSON value is not an object"
    return dict(payload), None


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _round(value: Any) -> float | None:
    parsed = _number(value)
    return round(parsed, 6) if parsed is not None else None


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


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def _blocked(payload: JsonMap) -> bool:
    verdict = str(payload.get("honest_verdict") or "")
    return verdict.startswith("blocked_") or payload.get("status") == "blocked"


def _flags_from_payload(payload: JsonMap) -> list[JsonDict]:
    flags = payload.get("corrigendum_pending") or []
    if not isinstance(flags, Sequence) or isinstance(flags, (str, bytes)):
        return []
    return [dict(item) for item in flags if isinstance(item, Mapping)]


def _missing_required_fields(payload: JsonMap, required: Sequence[str]) -> list[str]:
    return [field for field in required if field not in payload or payload.get(field) is None]


def _status_from_payload(payload: JsonMap, required: Sequence[str]) -> str:
    if _blocked(payload):
        return "blocked"
    if payload.get("flagged_adversarial") is True or _flags_from_payload(payload):
        return "flagged"
    if _missing_required_fields(payload, required):
        return "malformed"
    return "clean"


def _base_row(spec: JsonMap, path: Path, status: str) -> JsonDict:
    return {
        "artifact_id": spec["artifact_id"],
        "arm": spec["arm"],
        "experiment_id": spec["experiment_id"],
        "path": path.as_posix(),
        "status": status,
    }


def _status_record(row: JsonMap) -> JsonDict:
    return {
        "artifact_id": row.get("artifact_id"),
        "arm": row.get("arm"),
        "path": row.get("path"),
        "honest_verdict": row.get("honest_verdict"),
        "status": row.get("status"),
    }


def _summarize_payload(spec: JsonMap, path: Path, payload: JsonMap, status: str) -> JsonDict:
    row = _base_row(spec, path, status)
    row.update(
        {
            "sha256": _sha256_file(path),
            "honest_verdict": payload.get("honest_verdict"),
            "flags": _flags_from_payload(payload),
            "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
        }
    )
    artifact_id = spec["artifact_id"]
    if artifact_id == "D1":
        row.update(
            {
                "accuracy": _round(payload.get("accuracy")),
                "tuned_sc_accuracy": _round(payload.get("tuned_sc_accuracy")),
                "delta": _round(payload.get("delta_vs_tuned_sc")),
                "ci95": _ci95(payload.get("paired_ci95")),
                "mcnemar_p": _round(payload.get("mcnemar_p")),
                "headroom_present": payload.get("headroom_present") is True,
                "proper_musr_win": payload.get("proper_musr_win") is True,
                "best_arm_available": payload.get("best_arm_available") is True,
                "n_questions": int(_number(payload.get("n_questions")) or 0),
            }
        )
    elif artifact_id == "D4":
        row.update(
            {
                "delta": _round(payload.get("delta_vs_tuned_sc_second")),
                "ci95": _ci95(payload.get("paired_ci95_second")),
                "mcnemar_p": _round(payload.get("mcnemar_p_second")),
                "headroom_present": payload.get("headroom_present") is True,
                "second_corpus_confirmed": payload.get("second_corpus_confirmed") is True,
                "second_corpus_audit_clean": payload.get("second_corpus_audit_clean") is True,
                "d4_verdict_class": payload.get("d4_verdict_class"),
                "duplicate_audit_passed": payload.get("duplicate_audit_passed"),
                "leak_audit_passed": payload.get("leak_audit_passed"),
                "oracle_provenance_passed": payload.get("oracle_provenance_passed"),
                "train_test_overlap_passed": payload.get("train_test_overlap_passed"),
            }
        )
    elif artifact_id == "D6":
        row.update(
            {
                "cascade_executed": payload.get("cascade_executed") is True,
                "cascade_accuracy": _round(payload.get("cascade_accuracy")),
                "judge_only_accuracy": _round(payload.get("judge_only_accuracy")),
                "delta": _round(payload.get("delta_vs_judge_only")),
                "ci95": _ci95(payload.get("paired_ci95")),
                "judge_call_fraction": _round(payload.get("judge_call_fraction")),
                "efficiency_win": payload.get("efficiency_win") is True,
            }
        )
    elif artifact_id == "G1":
        row.update(
            {
                "guided_decoding_executed": payload.get("guided_decoding_executed") is True,
                "arms_differentiated": payload.get("arms_differentiated") is True,
                "guided_accuracy": _round(payload.get("guided_accuracy")),
                "unguided_accuracy": _round(payload.get("unguided_accuracy")),
                "delta": _round(payload.get("delta_guided_vs_unguided")),
                "nfe_by_arm": dict(payload.get("nfe_by_arm") or {})
                if isinstance(payload.get("nfe_by_arm"), Mapping)
                else None,
            }
        )
    return row


def load_upstream_artifacts(root: Path) -> JsonDict:
    root = Path(root)
    rows: list[JsonDict] = []
    buckets: dict[str, list[JsonDict]] = {
        "missing": [],
        "blocked": [],
        "malformed": [],
        "flagged": [],
        "clean": [],
    }
    for spec in UPSTREAM_SPECS:
        path = root / str(spec["relative_path"])
        if not path.exists():
            row = _base_row(spec, path, "missing")
            rows.append(row)
            buckets["missing"].append(_status_record(row))
            continue
        payload, error = _read_json_object(path)
        if payload is None:
            row = _base_row(spec, path, "malformed")
            row["parse_error"] = error
            row["sha256"] = _sha256_file(path)
            rows.append(row)
            buckets["malformed"].append(_status_record(row))
            continue
        status = _status_from_payload(payload, tuple(spec["required"]))
        row = _summarize_payload(spec, path, payload, status)
        rows.append(row)
        buckets[status].append(_status_record(row))
    return {
        "per_arm_table": rows,
        "missing_upstream_artifacts": buckets["missing"],
        "blocked_upstream_artifacts": buckets["blocked"],
        "malformed_upstream_artifacts": buckets["malformed"],
        "flagged_upstream_artifacts": buckets["flagged"],
        "clean_upstream_artifacts": buckets["clean"],
    }


def _row_by_id(rows: Sequence[JsonMap], artifact_id: str) -> JsonDict | None:
    for row in rows:
        if row.get("artifact_id") == artifact_id:
            return dict(row)
    return None


def _d1_proper_win(row: JsonMap | None) -> bool:
    if row is None:
        return False
    p_value = _number(row.get("mcnemar_p"))
    delta = _number(row.get("delta"))
    return (
        row.get("status") == "clean"
        and row.get("proper_musr_win") is True
        and row.get("verifier_is_oracle") is False
        and row.get("headroom_present") is True
        and delta is not None
        and delta > 0.0
        and _ci_excludes_zero_positive(row.get("ci95"))
        and p_value is not None
        and p_value < 0.05
    )


def _d4_audit_clean(row: JsonMap | None) -> bool:
    return bool(row and row.get("status") == "clean" and row.get("second_corpus_audit_clean") is True)


def _d4_confirmed(row: JsonMap | None) -> bool:
    if not _d4_audit_clean(row):
        return False
    p_value = _number(row.get("mcnemar_p"))
    delta = _number(row.get("delta"))
    return (
        row.get("second_corpus_confirmed") is True
        and row.get("verifier_is_oracle") is False
        and row.get("headroom_present") is True
        and delta is not None
        and delta > 0.0
        and (row.get("ci95") is None or _ci_excludes_zero_positive(row.get("ci95")))
        and (p_value is None or p_value < 0.05)
    )


def _d6_efficiency_win(row: JsonMap | None) -> bool:
    return bool(
        row
        and row.get("status") == "clean"
        and row.get("efficiency_win") is True
        and row.get("verifier_is_oracle") is False
    )


def _d6_significant_accuracy_win(row: JsonMap | None) -> bool:
    delta = _number((row or {}).get("delta"))
    return bool(delta is not None and delta > 0.0 and _ci_excludes_zero_positive((row or {}).get("ci95")))


def _d1_clean_no_win(row: JsonMap | None) -> bool:
    return bool(
        row
        and row.get("status") == "clean"
        and row.get("verifier_is_oracle") is False
        and _number(row.get("delta")) is not None
        and row.get("ci95") is not None
        and _number(row.get("mcnemar_p")) is not None
        and not _d1_proper_win(row)
    )


def _d4_clean_no_confirm(row: JsonMap | None) -> bool:
    return bool(
        row
        and row.get("status") == "clean"
        and row.get("second_corpus_audit_clean") is True
        and not _d4_confirmed(row)
    )


def _d6_clean_no_efficiency(row: JsonMap | None) -> bool:
    return bool(row and row.get("status") == "clean" and not _d6_efficiency_win(row))


def _bounded_retirement_ok(d1: JsonMap | None, d4: JsonMap | None, d6: JsonMap | None) -> bool:
    return _d1_clean_no_win(d1) and _d4_clean_no_confirm(d4) and _d6_clean_no_efficiency(d6)


def _guided_frontier_state(row: JsonMap | None) -> str:
    if row is None:
        return "missing"
    status = str(row.get("status"))
    if status != "clean":
        return status
    if row.get("guided_decoding_executed") is not True:
        return "not_executed"
    if row.get("arms_differentiated") is not True:
        return "controls_not_differentiated"
    delta = _number(row.get("delta"))
    if delta is not None and delta > 0.0:
        return f"guided_gain_observed_{_format_delta(delta)}"
    return "no_guided_gain"


def _best_evidence_row(
    d1: JsonMap | None,
    d4: JsonMap | None,
    d6: JsonMap | None,
    *,
    d1_win: bool,
    d4_confirmed: bool,
    d6_efficiency: bool,
) -> JsonDict | None:
    if d1_win:
        return dict(d1 or {})
    if d6_efficiency:
        return dict(d6 or {})
    if d4_confirmed:
        return dict(d4 or {})
    clean_rows = [
        dict(row)
        for row in (d1, d6)
        if row and row.get("status") == "clean" and _number(row.get("delta")) is not None
    ]
    return max(clean_rows, key=lambda row: float(row["delta"])) if clean_rows else None


def _execution_reasons(
    loaded: JsonMap,
    *,
    d1: JsonMap | None,
    d4: JsonMap | None,
    d6: JsonMap | None,
    d1_win: bool,
    d4_confirmed: bool,
    d6_efficiency: bool,
) -> list[str]:
    reasons: list[str] = []
    for bucket_name, label in (
        ("missing_upstream_artifacts", "missing"),
        ("blocked_upstream_artifacts", "blocked"),
        ("malformed_upstream_artifacts", "malformed"),
        ("flagged_upstream_artifacts", "flagged"),
    ):
        for item in loaded[bucket_name]:
            detail = item.get("honest_verdict") or item.get("path")
            reasons.append(f"{item.get('artifact_id')} {label}: {detail}")
    if d4 and d4.get("status") == "clean" and d4.get("second_corpus_audit_clean") is not True:
        reasons.append("D4 audit not clean: second_corpus_audit_clean=false")
    if d1_win and not d4_confirmed and not d6_efficiency:
        reasons.append("D4/D6 confirmation unavailable for clean D1 MuSR win")
    if d4_confirmed and not d1_win and not d6_efficiency and (d1 or {}).get("status") != "clean":
        reasons.append("MuSR/D6 evidence did not execute cleanly enough to headline D4")
    if d6_efficiency and not _d6_significant_accuracy_win(d6) and not d1_win:
        reasons.append("D6 efficiency observed but accuracy CI does not exclude zero")
    return sorted(dict.fromkeys(reasons))


def classify(loaded: JsonMap) -> JsonDict:
    rows = list(loaded["per_arm_table"])
    d1 = _row_by_id(rows, "D1")
    d4 = _row_by_id(rows, "D4")
    d6 = _row_by_id(rows, "D6")
    guided = _row_by_id(rows, "G1")
    d1_win = _d1_proper_win(d1)
    d4_confirm = _d4_confirmed(d4)
    d6_eff = _d6_efficiency_win(d6)
    d6_realizes = d6_eff and _d6_significant_accuracy_win(d6)
    second_audit_clean = _d4_audit_clean(d4)
    second_confirmed = d4_confirm
    bounded_ok = _bounded_retirement_ok(d1, d4, d6)
    best = _best_evidence_row(d1, d4, d6, d1_win=d1_win, d4_confirmed=d4_confirm, d6_efficiency=d6_eff)
    reasons = _execution_reasons(
        loaded,
        d1=d1,
        d4=d4,
        d6=d6,
        d1_win=d1_win,
        d4_confirmed=d4_confirm,
        d6_efficiency=d6_eff,
    )
    guided_state = _guided_frontier_state(guided)

    if (d1_win and (d4_confirm or d6_eff)) or d6_realizes:
        winner = d6 if d6_realizes and not d1_win else d1
        return {
            "moat_state": "moat_realized",
            "honest_verdict": (
                f"success_moat_realized_v465_{_slug((winner or {}).get('artifact_id'))}_"
                f"{_format_delta(_number((winner or {}).get('delta')))}"
            ),
            "best_arm": (winner or {}).get("artifact_id"),
            "best_arm_delta": (winner or {}).get("delta"),
            "best_arm_ci": (winner or {}).get("ci95"),
            "second_corpus_confirmed": second_confirmed,
            "second_corpus_audit_clean": second_audit_clean,
            "cascade_efficiency_win": d6_eff,
            "guided_decoding_frontier_state": guided_state,
            "bounded_retirement_ok": False,
            "execution_incomplete_reasons": reasons,
        }
    if d1_win:
        return {
            "moat_state": "musr_scoped_positive",
            "honest_verdict": f"complete_moat_musr_scoped_positive_v465_d1_{_format_delta(_number(d1.get('delta')))}",
            "best_arm": "D1",
            "best_arm_delta": d1.get("delta"),
            "best_arm_ci": d1.get("ci95"),
            "second_corpus_confirmed": False,
            "second_corpus_audit_clean": second_audit_clean,
            "cascade_efficiency_win": d6_eff,
            "guided_decoding_frontier_state": guided_state,
            "bounded_retirement_ok": False,
            "execution_incomplete_reasons": reasons,
        }
    if d4_confirm:
        return {
            "moat_state": "second_corpus_scoped_positive",
            "honest_verdict": f"complete_moat_second_corpus_scoped_positive_v465_d4_{_format_delta(_number(d4.get('delta')))}",
            "best_arm": "D4",
            "best_arm_delta": d4.get("delta"),
            "best_arm_ci": d4.get("ci95"),
            "second_corpus_confirmed": True,
            "second_corpus_audit_clean": True,
            "cascade_efficiency_win": d6_eff,
            "guided_decoding_frontier_state": guided_state,
            "bounded_retirement_ok": False,
            "execution_incomplete_reasons": reasons,
        }
    if bounded_ok:
        return {
            "moat_state": "retired_bounded",
            "honest_verdict": "complete_moat_retired_bounded_v465_clean_d1_d4_d6_null",
            "best_arm": None,
            "best_arm_delta": None,
            "best_arm_ci": None,
            "second_corpus_confirmed": False,
            "second_corpus_audit_clean": second_audit_clean,
            "cascade_efficiency_win": False,
            "guided_decoding_frontier_state": guided_state,
            "bounded_retirement_ok": True,
            "execution_incomplete_reasons": [],
        }
    return {
        "moat_state": "execution_incomplete",
        "honest_verdict": "complete_moat_execution_incomplete_v465_blocked_flagged_or_unclean",
        "best_arm": (best or {}).get("artifact_id"),
        "best_arm_delta": (best or {}).get("delta"),
        "best_arm_ci": (best or {}).get("ci95"),
        "second_corpus_confirmed": False,
        "second_corpus_audit_clean": second_audit_clean,
        "cascade_efficiency_win": d6_eff,
        "guided_decoding_frontier_state": guided_state,
        "bounded_retirement_ok": False,
        "execution_incomplete_reasons": reasons or ["gate_conditions_unsatisfied_without_clean_realization_or_retirement"],
    }


def _next_actions(moat_state: str, reasons: Sequence[str]) -> list[str]:
    if moat_state == "moat_realized":
        return ["Promote only the clean winning construction; keep D4 audit status visible in capstone reporting."]
    if moat_state == "musr_scoped_positive":
        return ["Run a clean D4 audit or D6 efficiency confirmation before making a PRD-level moat claim."]
    if moat_state == "second_corpus_scoped_positive":
        return ["Treat D4 as transfer evidence only; rerun MuSR/D6 cleanly before headline promotion."]
    if moat_state == "retired_bounded":
        return ["Retire this verifier-moat scope and pivot the next milestone to a new verifier direction."]
    actions = ["Repair or rerun blocked, malformed, flagged, or unclean upstream artifacts before claiming realization or retirement."]
    if any(str(reason).startswith("D1 ") for reason in reasons):
        actions.append("Rerun Exp 5059 without adversarial flags and with paired statistics that satisfy the D1 gate.")
    if any("D4 audit not clean" in str(reason) or str(reason).startswith("D4 ") for reason in reasons):
        actions.append("Rerun Exp 5060 until duplicate/leak/oracle audits are clean before counting second-corpus evidence.")
    if any(str(reason).startswith("D6 ") for reason in reasons):
        actions.append("Rerun Exp 5061 or report it as efficiency-only if the accuracy CI still includes zero.")
    return actions


def reproducibility_checksum(payload: JsonMap) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def build_artifact(loaded: JsonMap, duration_s: float) -> JsonDict:
    decision = classify(loaded)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "result_path": RESULT_RELATIVE_PATH,
        "spec_refs": list(SPEC_REFS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(float(duration_s), 0.0001), 6),
        "field_principles": FIELD_PRINCIPLES,
        "per_arm_table": list(loaded["per_arm_table"]),
        "missing_upstream_artifacts": list(loaded["missing_upstream_artifacts"]),
        "blocked_upstream_artifacts": list(loaded["blocked_upstream_artifacts"]),
        "malformed_upstream_artifacts": list(loaded["malformed_upstream_artifacts"]),
        "flagged_upstream_artifacts": list(loaded["flagged_upstream_artifacts"]),
        "clean_upstream_artifacts": list(loaded["clean_upstream_artifacts"]),
        "source_artifacts": {
            str(spec["artifact_id"]): str(spec["relative_path"]) for spec in UPSTREAM_SPECS
        },
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
            "second_corpus_audit_clean": artifact["second_corpus_audit_clean"],
            "cascade_efficiency_win": artifact["cascade_efficiency_win"],
            "guided_decoding_frontier_state": artifact["guided_decoding_frontier_state"],
            "bounded_retirement_ok": artifact["bounded_retirement_ok"],
            "execution_incomplete_reasons": artifact["execution_incomplete_reasons"],
            "status_rows": artifact["per_arm_table"],
        }
    )
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = [field for field in REQUIRED_FIELDS if field not in artifact]
    if artifact.get("moat_state") not in MOAT_STATES:
        errors.append("moat_state")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    for field in (
        "second_corpus_confirmed",
        "second_corpus_audit_clean",
        "cascade_efficiency_win",
        "bounded_retirement_ok",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in ("execution_incomplete_reasons", "per_arm_table", "next_actions"):
        if not isinstance(artifact.get(field), list):
            errors.append(field)
    if not isinstance(artifact.get("guided_decoding_frontier_state"), str):
        errors.append("guided_decoding_frontier_state")
    if not str(artifact.get("honest_verdict", "")).startswith(("success_", "complete_", "blocked_")):
        errors.append("honest_verdict")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    loaded = load_upstream_artifacts(root)
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
