"""Exp 5007: resolve the off-ARC verifier-moat gate from D1-D4 artifacts.

Spec refs: REQ-VERIFY-5007, SCENARIO-VERIFY-5007.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]

EXPERIMENT_ID = 5007
RESULT_RELATIVE_PATH = "results/experiment_5007_moat_gate_resolution.json"
D1_ARTIFACT_RELATIVE_PATH = "results/experiment_5003_lora_ebm_scorer_musr.json"
D2_ARTIFACT_RELATIVE_PATH = "results/experiment_5004_uprm_replication.json"
D3_ARTIFACT_RELATIVE_PATH = "results/experiment_5005_ebrm_uncertainty_verifier.json"
D4_ARTIFACT_RELATIVE_PATH = "results/experiment_5006_moat_second_corpus.json"
SPEC_REFS = ["REQ-VERIFY-5007", "SCENARIO-VERIFY-5007"]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a realized moat is "
            "success_moat_realized_off_arc_<arm>_<corpus>_<delta>, a bounded "
            "retirement is complete_moat_retired_bounded_lora_ebm_and_uprm_both_null."
        )
    },
    "moat_realized": {
        "principle": (
            "true iff >=1 oracle-distinct arm beats tuned-SC with CI95-excl-0 on a "
            "headroom-present corpus, cross-corpus confirmed."
        )
    },
    "moat_retired_bounded": {
        "principle": (
            "true iff D1 AND D2 both null on every headroom-present oracle-distinct "
            "corpus (the retire_if_same_verdict outcome; a publishable bounded null)."
        )
    },
    "best_arm": {
        "principle": (
            "the construction with the strongest oracle-distinct delta "
            "(LoRA-EBM/uPRM/EBRM) + its corpus + delta + CI."
        )
    },
    "per_arm_table": {
        "principle": (
            "per arm per corpus: delta_vs_tuned_sc, paired_ci95, mcnemar_p, "
            "verifier_is_oracle, headroom_present (the audit trail)."
        )
    },
    "diffusiongemma_gate_conditions_satisfied_off_arc": {
        "principle": (
            "true iff a POSITIVE arm satisfies the gate's 3 conditions (headroom "
            "present + non-trivial oracle-distinct verifier + matched-control "
            "CI95-excl-0) ON THE TESTED DOMAIN; activation stays operator-gated "
            "(do NOT autonomously flip the gate to MET)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false across all aggregated arms (the non-circular discipline; "
            "flagged_adversarial arms are skipped)."
        )
    },
    "flagged_arms_skipped": {
        "principle": (
            "the list of D arms skipped for flagged_adversarial=true (never "
            "aggregated into a headline)."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts (reads D1-D4 JSON, no LLM; "
            "0.0001s floor)."
        )
    },
    "cited_upstream_artifacts": {
        "principle": (
            "the {experiment_id, fields_imported, sha256} for each D arm so the "
            "verdict is traceable to a real measurement."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "moat_realized",
    "moat_retired_bounded",
    "best_arm",
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
    "adversarial_verify_clean",
    "adversarial_verify_flags",
)


@dataclass(frozen=True)
class ArtifactSpec:
    """One upstream D artifact and the metric names needed for aggregation."""

    arm_id: str
    arm: str
    experiment_id: int
    relative_path: str
    accuracy_field: str | None
    tuned_sc_field: str | None
    delta_field: str | None
    ci95_field: str | None
    mcnemar_field: str | None
    corpus_field: str | None
    oracle_at_k_field: str | None


ARM_SPECS = (
    ArtifactSpec(
        "D1",
        "LoRA-EBM",
        5003,
        D1_ARTIFACT_RELATIVE_PATH,
        "trained_scorer_accuracy",
        "tuned_sc_accuracy",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "mcnemar_p",
        None,
        "oracle_at_k",
    ),
    ArtifactSpec(
        "D2",
        "uPRM",
        5004,
        D2_ARTIFACT_RELATIVE_PATH,
        "uprm_selection_accuracy",
        "tuned_sc_accuracy",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "mcnemar_p",
        "corpus",
        "oracle_at_k",
    ),
    ArtifactSpec(
        "D3",
        "EBRM",
        5005,
        D3_ARTIFACT_RELATIVE_PATH,
        "ebrm_selection_accuracy",
        "tuned_sc_accuracy",
        "delta_vs_tuned_sc",
        "paired_ci95",
        "mcnemar_p",
        None,
        "oracle_at_k",
    ),
)
D4_SPEC = ArtifactSpec(
    "D4",
    "second-corpus-confirmation",
    5006,
    D4_ARTIFACT_RELATIVE_PATH,
    "second_corpus_accuracy",
    "tuned_sc_accuracy_second",
    "delta_vs_tuned_sc_second",
    "paired_ci95_second",
    "mcnemar_p_second",
    "second_corpus",
    "oracle_at_k_second",
)
ALL_SPECS = (*ARM_SPECS, D4_SPEC)
ARM_BY_ID = {spec.arm_id: spec.arm for spec in ARM_SPECS}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _fields_for_spec(spec: ArtifactSpec, payload: Mapping[str, Any], flagged: bool) -> list[str]:
    if flagged:
        return [field for field in ("flagged_adversarial", "honest_verdict") if field in payload]
    wanted = [
        "honest_verdict",
        "flagged_adversarial",
        "verifier_is_oracle",
        "headroom_present",
        spec.accuracy_field,
        spec.tuned_sc_field,
        spec.delta_field,
        spec.ci95_field,
        spec.mcnemar_field,
        spec.corpus_field,
        spec.oracle_at_k_field,
        "best_verifier_from",
        "n_questions",
    ]
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


def _row_is_positive(row: Mapping[str, Any]) -> bool:
    delta = _number(row.get("delta_vs_tuned_sc"))
    mcnemar_p = _number(row.get("mcnemar_p"))
    return (
        row.get("verifier_is_oracle") is False
        and row.get("headroom_present") is True
        and delta is not None
        and delta > 0.0
        and _ci_excludes_zero_positive(row.get("paired_ci95"))
        and mcnemar_p is not None
        and mcnemar_p < 0.05
    )


def _row_is_headroom_oracle_distinct(row: Mapping[str, Any]) -> bool:
    return row.get("verifier_is_oracle") is False and row.get("headroom_present") is True


def _d_arm_row(spec: ArtifactSpec, path: Path, payload: Mapping[str, Any]) -> JsonDict:
    delta = _number(payload.get(spec.delta_field)) if spec.delta_field else None
    ci95 = _ci95(payload.get(spec.ci95_field)) if spec.ci95_field else None
    mcnemar_p = _number(payload.get(spec.mcnemar_field)) if spec.mcnemar_field else None
    accuracy = _number(payload.get(spec.accuracy_field)) if spec.accuracy_field else None
    tuned_sc = _number(payload.get(spec.tuned_sc_field)) if spec.tuned_sc_field else None
    oracle_at_k = _number(payload.get(spec.oracle_at_k_field)) if spec.oracle_at_k_field else None
    corpus = payload.get(spec.corpus_field) if spec.corpus_field else None
    row = {
        "arm_id": spec.arm_id,
        "arm": spec.arm,
        "source_experiment_id": spec.experiment_id,
        "source_artifact": path.as_posix(),
        "corpus": str(corpus or "MuSR"),
        "selection_accuracy": _rounded(accuracy),
        "tuned_sc_accuracy": _rounded(tuned_sc),
        "delta_vs_tuned_sc": _rounded(delta),
        "paired_ci95": ci95,
        "mcnemar_p": _rounded(mcnemar_p),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
        "headroom_present": payload.get("headroom_present"),
        "oracle_at_k": _rounded(oracle_at_k),
        "n_questions": int(payload.get("n_questions") or 0),
    }
    row["win_vs_tuned_sc"] = _row_is_positive(row)
    return row


def _d4_row(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    source_arm_id = str(payload.get("best_verifier_from") or "unknown")
    delta = _number(payload.get(D4_SPEC.delta_field))
    ci95 = _ci95(payload.get(D4_SPEC.ci95_field))
    mcnemar_p = _number(payload.get(D4_SPEC.mcnemar_field))
    accuracy = _number(payload.get(D4_SPEC.accuracy_field))
    tuned_sc = _number(payload.get(D4_SPEC.tuned_sc_field))
    oracle_at_k = _number(payload.get(D4_SPEC.oracle_at_k_field))
    row = {
        "arm_id": source_arm_id,
        "arm": ARM_BY_ID.get(source_arm_id, source_arm_id),
        "source_experiment_id": D4_SPEC.experiment_id,
        "source_artifact": path.as_posix(),
        "corpus": str(payload.get("second_corpus") or "second_corpus_unknown"),
        "selection_accuracy": _rounded(accuracy),
        "tuned_sc_accuracy": _rounded(tuned_sc),
        "delta_vs_tuned_sc": _rounded(delta),
        "paired_ci95": ci95,
        "mcnemar_p": _rounded(mcnemar_p),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
        "headroom_present": payload.get("headroom_present"),
        "oracle_at_k": _rounded(oracle_at_k),
        "n_questions": int(payload.get("n_questions") or 0),
        "confirmation_source": "D4",
    }
    row["win_vs_tuned_sc"] = _row_is_positive(row)
    return row


def _row_for_spec(spec: ArtifactSpec, path: Path, payload: Mapping[str, Any]) -> JsonDict:
    if spec.arm_id == "D4":
        return _d4_row(path, payload)
    return _d_arm_row(spec, path, payload)


def load_upstream_artifacts(root: Path) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], int]:
    rows: list[JsonDict] = []
    flagged: list[JsonDict] = []
    citations: list[JsonDict] = []
    missing: list[JsonDict] = []
    nonflagged_present = 0

    for spec in ALL_SPECS:
        path = root / spec.relative_path
        if not path.exists():
            missing.append(
                {
                    "arm_id": spec.arm_id,
                    "arm": spec.arm,
                    "experiment_id": spec.experiment_id,
                    "path": path.as_posix(),
                }
            )
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
        citations.append(_citation(spec, path, payload))
        if payload.get("flagged_adversarial") is True:
            flagged.append(_flagged_skip(spec, path, payload))
            continue
        nonflagged_present += 1
        rows.append(_row_for_spec(spec, path, payload))

    return rows, flagged, citations, missing, nonflagged_present


def _best_row(rows: Sequence[JsonMap]) -> JsonDict | None:
    candidates = [
        row
        for row in rows
        if row.get("verifier_is_oracle") is False
        and _number(row.get("delta_vs_tuned_sc")) is not None
    ]
    if not candidates:
        return None
    return dict(max(candidates, key=lambda row: float(row["delta_vs_tuned_sc"])))


def _confirmed_positive_row(rows: Sequence[JsonMap]) -> JsonDict | None:
    positive_musr = [
        row
        for row in rows
        if int(row.get("source_experiment_id") or 0) in {5003, 5004, 5005}
        and _row_is_positive(row)
    ]
    positive_arm_ids = {str(row.get("arm_id")) for row in positive_musr}
    positive_second = [
        row
        for row in rows
        if int(row.get("source_experiment_id") or 0) == 5006
        and _row_is_positive(row)
        and str(row.get("arm_id")) in positive_arm_ids
    ]
    if not positive_second:
        return None
    confirmed_arm_ids = {str(row.get("arm_id")) for row in positive_second}
    candidates = [
        row for row in (*positive_musr, *positive_second) if str(row.get("arm_id")) in confirmed_arm_ids
    ]
    return dict(max(candidates, key=lambda row: float(row["delta_vs_tuned_sc"])))


def _d1_d2_retired(rows: Sequence[JsonMap]) -> bool:
    for arm_id in ("D1", "D2"):
        arm_rows = [
            row
            for row in rows
            if str(row.get("arm_id")) == arm_id and _row_is_headroom_oracle_distinct(row)
        ]
        if not arm_rows or any(_row_is_positive(row) for row in arm_rows):
            return False
    return True


def _mixed_verdict(rows: Sequence[JsonMap]) -> str:
    if not any(_row_is_headroom_oracle_distinct(row) for row in rows):
        return "complete_moat_scoped_no_headroom_present_false_negative_risk"
    positive_musr = any(
        int(row.get("source_experiment_id") or 0) in {5003, 5004, 5005}
        and _row_is_positive(row)
        for row in rows
    )
    positive_second = any(
        int(row.get("source_experiment_id") or 0) == 5006 and _row_is_positive(row)
        for row in rows
    )
    if positive_musr and not positive_second:
        return "complete_moat_scoped_positive_musr_no_cross_corpus_confirm"
    if positive_second and not positive_musr:
        return "complete_moat_scoped_second_corpus_unanchored_no_musr_positive"
    return "complete_moat_scoped_no_realized_no_bounded_retirement"


def _metric_text(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "no clean numeric arm"
    return (
        f"{row.get('arm')} on {row.get('corpus')} "
        f"delta_vs_tuned_sc={_format_delta(_number(row.get('delta_vs_tuned_sc')))}, "
        f"CI95={row.get('paired_ci95')}, McNemar p={row.get('mcnemar_p')}"
    )


def _second_confirmation_text(rows: Sequence[JsonMap], arm_id: str) -> str:
    confirmations = [
        row
        for row in rows
        if int(row.get("source_experiment_id") or 0) == 5006 and str(row.get("arm_id")) == arm_id
    ]
    if not confirmations:
        return "no D4 second-corpus row was available"
    row = confirmations[0]
    return (
        f"D4 confirmed on {row.get('corpus')} with "
        f"delta_vs_tuned_sc={_format_delta(_number(row.get('delta_vs_tuned_sc')))}, "
        f"CI95={row.get('paired_ci95')}, McNemar p={row.get('mcnemar_p')}"
    )


def _paper_summary(
    *,
    decision: str,
    best: Mapping[str, Any] | None,
    rows: Sequence[JsonMap],
    flagged: Sequence[JsonMap],
) -> str:
    if decision == "BLOCKED":
        return (
            "D5 is blocked: every available D1-D4 input was absent or carried "
            "flagged_adversarial=true, so no verifier-moat arm was aggregated."
        )
    if decision == "POSITIVE" and best is not None:
        return (
            "The off-ARC verifier moat is realized: "
            f"{_metric_text(best)} beats TUNED-SC on a headroom-present "
            "oracle-distinct domain, and "
            f"{_second_confirmation_text(rows, str(best.get('arm_id')))}. "
            "DiffusionGemma's off-ARC gate conditions are satisfied on the tested "
            "domain, while ARC headroom remains the canonical uncaptured target and "
            "activation remains operator-gated."
        )
    if decision == "ALL-NULL-RETIRE":
        return (
            "D1 and D2 both failed to beat TUNED-SC with CI95 excluding zero on every "
            "clean headroom-present oracle-distinct corpus available to this "
            "aggregation, so the off-ARC verifier moat is retired as bounded for "
            "the tested D1/D2 constructions."
        )
    skipped = ", ".join(str(item.get("arm_id")) for item in flagged) or "none"
    return (
        "D5 is scoped rather than realized or retired: "
        f"best clean row is {_metric_text(best)}, skipped flagged artifacts were "
        f"{skipped}, and the evidence did not provide both a clean positive MuSR arm "
        "and a clean D4 second-corpus confirmation. This is not a bounded retirement "
        "unless clean D1 and D2 nulls are both present."
    )


def _decision(rows: Sequence[JsonMap], nonflagged_present: int) -> tuple[str, str, JsonDict | None]:
    if nonflagged_present == 0:
        return "BLOCKED", "blocked_no_moat_arms", None
    confirmed = _confirmed_positive_row(rows)
    if confirmed is not None:
        verdict = (
            "success_moat_realized_off_arc_"
            f"{_slug(confirmed.get('arm'))}_{_slug(confirmed.get('corpus'))}_"
            f"{_format_delta(_number(confirmed.get('delta_vs_tuned_sc')))}"
        )
        return "POSITIVE", verdict, confirmed
    if _d1_d2_retired(rows):
        return (
            "ALL-NULL-RETIRE",
            "complete_moat_retired_bounded_lora_ebm_and_uprm_both_null",
            _best_row(rows),
        )
    return "MIXED-SCOPED", _mixed_verdict(rows), _best_row(rows)


def reproducibility_checksum(payload: JsonMap) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def build_artifact(
    *,
    rows: Sequence[JsonDict],
    flagged: Sequence[JsonDict],
    citations: Sequence[JsonDict],
    missing: Sequence[JsonDict],
    nonflagged_present: int,
    duration_s: float,
) -> JsonDict:
    decision, honest_verdict, selected = _decision(rows, nonflagged_present)
    best = selected if decision == "POSITIVE" else _best_row(rows)
    moat_realized = decision == "POSITIVE"
    artifact = {
        "experiment": "experiment_5007_moat_gate_resolution",
        "schema": "carnot.experiment_5007_moat_gate_resolution.v1",
        "honest_verdict": honest_verdict,
        "decision": decision,
        "moat_realized": moat_realized,
        "moat_retired_bounded": decision == "ALL-NULL-RETIRE",
        "best_arm": best,
        "per_arm_table": list(rows),
        "diffusiongemma_gate_conditions_satisfied_off_arc": moat_realized,
        "diffusiongemma_gate_status": (
            "conditions_satisfied_off_arc_operator_gated" if moat_realized else "STILL-PENDING"
        ),
        "diffusiongemma_activation": (
            "operator_gated_not_flipped" if moat_realized else "not_activated"
        ),
        "verifier_is_oracle": False,
        "flagged_arms_skipped": list(flagged),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "cited_upstream_artifacts": list(citations),
        "missing_upstream_artifacts": list(missing),
        "paper_summary": _paper_summary(
            decision=decision,
            best=best,
            rows=rows,
            flagged=flagged,
        ),
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
            "citations": artifact["cited_upstream_artifacts"],
            "missing": artifact["missing_upstream_artifacts"],
        }
    )
    return artifact


def _compact_adversarial_flags(report: JsonMap) -> list[JsonDict]:
    if "reports" in report and isinstance(report["reports"], list) and report["reports"]:
        report = report["reports"][0]
    flags = report.get("flags", []) if isinstance(report, Mapping) else []
    return [dict(flag) for flag in flags if isinstance(flag, Mapping)]


def _audit_is_clean(report: JsonMap) -> bool:
    if "max_severity" in report:
        return int(report.get("max_severity") or 0) == 0
    if "flagged_count" in report:
        return int(report.get("flagged_count") or 0) == 0
    if "flag_count" in report:
        return int(report.get("flag_count") or 0) == 0
    return not _compact_adversarial_flags(report)


def run_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - import glue
    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5007", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/adversarial_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def run_summarize_artifact(path: Path) -> int:  # pragma: no cover - reviewer CLI glue
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5007", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/summarize_artifact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.summarize(path))


def attach_audit(
    artifact: JsonDict,
    *,
    artifact_path: Path,
    audit_runner: AuditRunner,
    summary_runner: SummaryRunner,
) -> JsonDict:
    write_json(artifact_path, artifact)
    audit_report = audit_runner(artifact_path)
    updated = dict(artifact)
    updated["adversarial_verify_clean"] = _audit_is_clean(audit_report)
    updated["adversarial_verify_flags"] = _compact_adversarial_flags(audit_report)
    updated["adversarial_verify_report"] = audit_report
    write_json(artifact_path, updated)
    updated["summarize_artifact_exit_code"] = int(summary_runner(artifact_path))
    write_json(artifact_path, updated)
    return updated


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    for field in (
        "moat_realized",
        "moat_retired_bounded",
        "diffusiongemma_gate_conditions_satisfied_off_arc",
        "verifier_is_oracle",
        "adversarial_verify_clean",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate")
    if artifact.get("diffusiongemma_gate_status") == "MET":
        errors.append("diffusiongemma_gate_status")
    if artifact.get("decision") not in {"BLOCKED", "POSITIVE", "ALL-NULL-RETIRE", "MIXED-SCOPED"}:
        errors.append("decision")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("blocked_", "complete_", "success_")
    ):
        errors.append("honest_verdict")
    for field in (
        "per_arm_table",
        "flagged_arms_skipped",
        "cited_upstream_artifacts",
        "missing_upstream_artifacts",
        "adversarial_verify_flags",
    ):
        if not isinstance(artifact.get(field), list):
            errors.append(field)
    if artifact.get("best_arm") is not None and not isinstance(artifact.get("best_arm"), dict):
        errors.append("best_arm")
    if not isinstance(artifact.get("paper_summary"), str) or not artifact.get("paper_summary"):
        errors.append("paper_summary")
    if not isinstance(artifact.get("duration_s"), (int, float)) or float(
        artifact.get("duration_s") or 0.0
    ) < 0.0001:
        errors.append("duration_s")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    audit_runner: AuditRunner = run_adversarial_verify,
    summary_runner: SummaryRunner = run_summarize_artifact,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    rows, flagged, citations, missing, nonflagged_present = load_upstream_artifacts(root)
    artifact = build_artifact(
        rows=rows,
        flagged=flagged,
        citations=citations,
        missing=missing,
        nonflagged_present=nonflagged_present,
        duration_s=float(now()) - start,
    )
    if not write:
        return artifact
    return attach_audit(
        artifact,
        artifact_path=artifact_path,
        audit_runner=audit_runner,
        summary_runner=summary_runner,
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    _ = argv
    artifact_path = REPO_ROOT / RESULT_RELATIVE_PATH
    artifact = run(artifact_path=artifact_path)
    errors = artifact_schema_errors(artifact)
    if errors:
        print(f"schema errors: {', '.join(errors)}", file=sys.stderr)
        return 2
    print(artifact_path.as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main(sys.argv[1:]))
