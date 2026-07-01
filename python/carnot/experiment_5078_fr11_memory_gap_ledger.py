#!/usr/bin/env python3
"""Exp 5078: FR-11 memory-gap blocker ledger.

Spec refs: REQ-LEARN-5078, SCENARIO-LEARN-5078-MEMORY-GAP-LEDGER.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]

EXPERIMENT_ID = 5078
EXPERIMENT_NAME = "experiment_5078_fr11_memory_gap_ledger"
SCHEMA = "carnot.experiment_5078_fr11_memory_gap_ledger.v466"
RESULT_RELATIVE_PATH = "results/experiment_5078_fr11_memory_gap_ledger_v466.json"
EXP5051_RESULT_RELATIVE_PATH = "results/experiment_5051_verifier_trace_self_learning.json"
EXP5064_RESULT_RELATIVE_PATH = "results/experiment_5064_audited_skillgraph_self_learning.json"
EXP5077_RESULT_RELATIVE_PATH = "results/experiment_5077_fr11_group_sc_memory_v466.json"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = ["REQ-LEARN-5078", "SCENARIO-LEARN-5078-MEMORY-GAP-LEDGER"]
HONEST_VERDICT = "complete_fr11_memory_gap_ledger_written_no_promotion"

CONTROLLED_FAILURE_MODES = (
    "data_contamination",
    "irrelevant_replay",
    "retrieval_mismatch",
    "overfitting",
    "nonforgetting_regression",
    "verifier_shortcut",
    "insufficient_evaluation_power",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "fr11_attempts_summarized",
    "recurring_failure_modes",
    "safe_next_mechanisms",
    "retired_mechanisms",
    "promotion_blockers",
    "docs_update_recommendations",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix that states this is a written FR-11 blocker ledger, not a promotion."
    },
    "duration_s": {
        "principle": "measured wall-clock time for deterministic aggregation over existing artifacts."
    },
    "inference_substrate": {
        "principle": "declares aggregation_from_upstream_artifacts because no live model or verifier pass runs here."
    },
    "fr11_attempts_summarized": {
        "principle": "one compact row per upstream self-learning attempt so planners can audit the source of each blocker."
    },
    "recurring_failure_modes": {
        "principle": "controlled-vocabulary classification of observed, guarded, or unobserved FR-11 failure modes."
    },
    "safe_next_mechanisms": {
        "principle": "planner-facing mechanisms that are safe because they preserve no-promotion and guard requirements."
    },
    "retired_mechanisms": {
        "principle": "mechanisms contradicted by the recent upstream evidence and unsafe to rerun unchanged."
    },
    "promotion_blockers": {
        "principle": "the concrete guard failures or evidence gaps that prevent FR-11 promotion."
    },
    "docs_update_recommendations": {
        "principle": "documentation reconciliation hints only; this experiment does not edit ops or BMAD docs."
    },
    "flagged_adversarial": {
        "principle": "false only when this ledger is internally schema-clean and preserves upstream flags separately."
    },
}


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def read_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def load_inputs(root: Path) -> tuple[JsonDict, JsonDict, JsonDict]:
    root = Path(root)
    return (
        read_json_object(root / EXP5051_RESULT_RELATIVE_PATH),
        read_json_object(root / EXP5064_RESULT_RELATIVE_PATH),
        read_json_object(root / EXP5077_RESULT_RELATIVE_PATH),
    )


def summarize_attempt(payload: JsonMap, path: str) -> JsonDict:
    experiment_id = int(payload.get("experiment_id") or 0)
    heldout_delta = number(payload.get("heldout_delta"))
    nonforgetting_delta = number(payload.get("nonforgetting_delta"))
    promotion = payload.get("promotion_decision")
    promotion_decision = dict(promotion) if isinstance(promotion, Mapping) else {}
    heldout = payload.get("heldout_evaluation")
    heldout_eval = dict(heldout) if isinstance(heldout, Mapping) else {}
    return {
        "experiment_id": experiment_id,
        "artifact_path": path,
        "artifact_present": bool(payload),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "inference_substrate": str(payload.get("inference_substrate") or ""),
        "heldout_delta": heldout_delta,
        "nonforgetting_delta": nonforgetting_delta,
        "contamination_guard_passed": payload.get("contamination_guard_passed"),
        "promotion_attempted": experiment_id in {5064, 5077},
        "promoted": bool(
            payload.get("promoted") is True
            or int(payload.get("promoted_count") or 0) > 0
            or promotion_decision.get("promoted") is True
        ),
        "no_promote_reason": str(
            payload.get("no_promote_reason") or promotion_decision.get("no_promote_reason") or ""
        ),
        "heldout_n": int(
            heldout_eval.get("heldout_n")
            or heldout_eval.get("n_rows")
            or payload.get("heldout_n")
            or 0
        ),
        "regressed_previously_correct_ids": list(
            heldout_eval.get("regressed_previously_correct_ids") or []
        ),
        "improved_previously_wrong_ids": list(
            heldout_eval.get("improved_previously_wrong_ids") or []
        ),
        "guard_that_worked": guard_that_worked(payload),
    }


def guard_that_worked(payload: JsonMap) -> str:
    if payload.get("contamination_guard_passed") is True and (
        payload.get("heldout_delta", 0) is not None
    ):
        if number(payload.get("heldout_delta")) is not None and number(payload.get("heldout_delta")) < 0:
            return "heldout_delta_gate_blocked_promotion"
    if payload.get("rollback_guard_passed") is True:
        return "rollback_guard_preserved_baseline"
    if payload.get("contamination_guard_passed") is True:
        return "contamination_guard_passed_no_leak"
    return "no_guard_evidence"


def _attempts_by_id(attempts: Sequence[JsonMap]) -> dict[int, JsonMap]:
    return {int(row.get("experiment_id") or 0): row for row in attempts}


def _mode(
    name: str,
    observed: bool,
    affected: Sequence[int],
    evidence: Sequence[str],
    guard: str,
) -> JsonDict:
    return {
        "mode": name,
        "observed": bool(observed),
        "affected_experiments": sorted(int(exp_id) for exp_id in affected),
        "evidence": list(evidence),
        "guard_or_blocker": guard,
    }


def classify_failure_modes(
    exp5051: JsonMap,
    exp5064: JsonMap,
    exp5077: JsonMap,
    attempts: Sequence[JsonMap],
) -> list[JsonDict]:
    by_id = _attempts_by_id(attempts)
    contamination_failed = [
        exp_id
        for exp_id, row in by_id.items()
        if row.get("artifact_present") and row.get("contamination_guard_passed") is False
    ]
    rejected = exp5051.get("trace_filter_diagnostics")
    rejected_count = int(rejected.get("rejected_trace_count") or 0) if isinstance(rejected, Mapping) else 0
    quarantined_count = int(exp5077.get("quarantined_count") or 0)
    dev_eval = exp5077.get("dev_evaluation")
    dev_delta = number(dev_eval.get("delta")) if isinstance(dev_eval, Mapping) else None
    heldout_delta_5077 = number(exp5077.get("heldout_delta"))
    upstream_flags = upstream_flagged_sources(exp5077)

    return [
        _mode(
            "data_contamination",
            bool(contamination_failed),
            contamination_failed,
            [
                "all loaded attempts report contamination_guard_passed=true"
                if not contamination_failed
                else "at least one contamination guard failed"
            ],
            "contamination_guard_passed blocked leakage checks before promotion",
        ),
        _mode(
            "irrelevant_replay",
            rejected_count > 0 or quarantined_count > 0,
            [exp_id for exp_id in (5051, 5077) if exp_id in by_id],
            [
                f"Exp5051 rejected_trace_count={rejected_count}",
                f"Exp5077 quarantined_count={quarantined_count}",
            ],
            "structural trace filter and group-SC quarantine kept replay evidence from auto-promotion",
        ),
        _mode(
            "retrieval_mismatch",
            any((by_id[exp_id].get("heldout_delta") or 0) < 0 for exp_id in (5051, 5077)),
            [exp_id for exp_id in (5051, 5077) if (by_id[exp_id].get("heldout_delta") or 0) < 0],
            [
                f"Exp5051 heldout_delta={by_id.get(5051, {}).get('heldout_delta')}",
                f"Exp5077 memory_policy={dict(exp5077.get('memory_policy') or {}).get('policy_signature', '')}",
                f"Exp5077 heldout_delta={heldout_delta_5077}",
            ],
            "heldout utility gate blocked retrieval policy promotion",
        ),
        _mode(
            "overfitting",
            bool(dev_delta is not None and dev_delta > 0 and heldout_delta_5077 is not None and heldout_delta_5077 < 0),
            [5077] if dev_delta is not None and heldout_delta_5077 is not None and dev_delta > 0 and heldout_delta_5077 < 0 else [],
            [
                f"Exp5077 dev_delta={dev_delta}",
                f"Exp5077 heldout_delta={heldout_delta_5077}",
            ],
            "held-out split exposed a dev-only gain",
        ),
        _mode(
            "nonforgetting_regression",
            any((by_id[exp_id].get("nonforgetting_delta") or 0) < 0 for exp_id in (5064, 5077)),
            [exp_id for exp_id in (5064, 5077) if (by_id[exp_id].get("nonforgetting_delta") or 0) < 0],
            [
                f"Exp5064 nonforgetting_delta={by_id.get(5064, {}).get('nonforgetting_delta')}",
                f"Exp5077 nonforgetting_delta={by_id.get(5077, {}).get('nonforgetting_delta')}",
            ],
            "nonforgetting gate and rollback guard prevented promotion",
        ),
        _mode(
            "verifier_shortcut",
            bool(upstream_flags),
            [5077] if upstream_flags else [],
            [
                "upstream flagged adversarial sources: " + ",".join(upstream_flags)
                if upstream_flags
                else "no upstream verifier-shortcut flag was present in the loaded artifacts"
            ],
            "ledger preserves upstream adversarial flags and does not convert them into promotion evidence",
        ),
        _mode(
            "insufficient_evaluation_power",
            True,
            [exp_id for exp_id in (5051, 5064, 5077) if exp_id in by_id],
            [
                f"heldout_n values={[by_id[exp_id].get('heldout_n') for exp_id in (5051, 5064, 5077)]}",
                "no paired CI or power calculation is present in these upstream artifacts",
            ],
            "promotion remains blocked until held-out effect size and nonforgetting survive stronger evaluation",
        ),
    ]


def safe_next_mechanisms(modes: Sequence[JsonMap]) -> list[JsonDict]:
    _ = modes
    return [
        {
            "mechanism": "retrieval_config_evolution",
            "safe_to_try": True,
            "why": "changes only retrieval triggers, thresholds, and fallback routing under the same held-out and nonforgetting gates.",
            "must_keep_guards": ["contamination_guard", "heldout_delta_gate", "nonforgetting_gate"],
        },
        {
            "mechanism": "skill_lifecycle_governance",
            "safe_to_try": True,
            "why": "keeps candidate skills quarantined until promotion, rollback, demotion, and external-audit receipts all agree.",
            "must_keep_guards": ["external_audit_receipts", "rollback_guard", "no_promote_on_negative_delta"],
        },
        {
            "mechanism": "process_verifier_replay",
            "safe_to_try": True,
            "why": "tests process-level evidence instead of answer-level replay and can directly target verifier shortcut risk.",
            "must_keep_guards": ["oracle_distinct_labels", "answer_leakage_check", "paired_heldout_eval"],
        },
        {
            "mechanism": "bounded_fr11_retirement_for_domain",
            "safe_to_try": True,
            "why": "turns repeated no-promote evidence into an explicit bounded retirement rather than another replay attempt.",
            "must_keep_guards": ["same_verdict_retirement_gate", "headroom_recheck", "docs_reconciliation"],
        },
    ]


def retired_mechanisms() -> list[JsonDict]:
    return [
        {
            "mechanism": "blind_replay_memory_insertion",
            "retired_for_domain": True,
            "evidence": "Exp5051 replay insertion reduced held-out accuracy by 0.05.",
        },
        {
            "mechanism": "promote_on_dev_or_consensus_without_heldout_gain",
            "retired_for_domain": True,
            "evidence": "Exp5077 dev gain did not transfer to held-out, and Exp5064 no-promoted verified skills.",
        },
        {
            "mechanism": "group_sc_disagreement_fallback_policy",
            "retired_for_domain": True,
            "evidence": "Exp5077 fallback_to_tuned_on_verifier_sc_disagreement regressed held-out and nonforgetting.",
        },
    ]


def promotion_blockers(attempts: Sequence[JsonMap], exp5077: JsonMap) -> list[JsonDict]:
    rows: list[JsonDict] = []
    heldout_blocked = [
        int(row["experiment_id"])
        for row in attempts
        if row.get("artifact_present") and (row.get("heldout_delta") or 0) <= 0
    ]
    nonforgetting_blocked = [
        int(row["experiment_id"])
        for row in attempts
        if row.get("nonforgetting_delta") is not None and (row.get("nonforgetting_delta") or 0) < 0
    ]
    if heldout_blocked:
        rows.append(
            {
                "blocker": "heldout_delta_nonpositive",
                "affected_experiments": heldout_blocked,
                "principle": "FR-11 updates cannot promote without positive held-out utility.",
            }
        )
    if nonforgetting_blocked:
        rows.append(
            {
                "blocker": "nonforgetting_regressed",
                "affected_experiments": nonforgetting_blocked,
                "principle": "A useful memory update must not erase previously correct held-out behavior.",
            }
        )
    rows.append(
        {
            "blocker": "no_new_promotion_executed",
            "affected_experiments": [EXPERIMENT_ID],
            "principle": "This aggregation task is explicitly read-only and cannot promote memory or skills.",
        }
    )
    upstream_flags = upstream_flagged_sources(exp5077)
    if upstream_flags:
        rows.append(
            {
                "blocker": "upstream_adversarial_source_preserved",
                "affected_experiments": [5077],
                "principle": "Upstream adversarial flags remain blockers until the flagged source is revalidated.",
                "sources": upstream_flags,
            }
        )
    rows.append(
        {
            "blocker": "insufficient_evaluation_power",
            "affected_experiments": [int(row["experiment_id"]) for row in attempts if row.get("artifact_present")],
            "principle": "Small held-out slices without paired confidence bounds are not enough to justify promotion.",
        }
    )
    return rows


def docs_update_recommendations() -> list[JsonDict]:
    return [
        {
            "target": "research-program.md",
            "recommendation": "Record FR-11 replay-memory for this domain as no-promote pending a stronger process-verifier or retirement gate.",
        },
        {
            "target": "ops/known-issues.md",
            "recommendation": "Add the Exp5051/5064/5077 memory-gap pattern and preserve the aggregation-substrate discipline.",
        },
        {
            "target": "_bmad/prd.md",
            "recommendation": "Clarify that FR-11 success requires positive held-out utility and nonforgetting, not merely a completed self-learning loop.",
        },
    ]


def upstream_flagged_sources(exp5077: JsonMap) -> list[str]:
    sources = exp5077.get("upstream_flagged_adversarial_sources") or []
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        return []
    return sorted(str(source) for source in sources)


def source_artifacts(root: Path) -> list[JsonDict]:
    paths = [
        EXP5051_RESULT_RELATIVE_PATH,
        EXP5064_RESULT_RELATIVE_PATH,
        EXP5077_RESULT_RELATIVE_PATH,
    ]
    return [
        {
            "path": path,
            "present": (Path(root) / path).exists(),
            "sha256": sha256_file(Path(root) / path),
        }
        for path in paths
    ]


def checksum(artifact: JsonMap) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def build_artifact(
    *,
    root: Path,
    exp5051: JsonMap,
    exp5064: JsonMap,
    exp5077: JsonMap,
    duration_s: float,
) -> JsonDict:
    attempts = [
        summarize_attempt(exp5051, EXP5051_RESULT_RELATIVE_PATH),
        summarize_attempt(exp5064, EXP5064_RESULT_RELATIVE_PATH),
        summarize_attempt(exp5077, EXP5077_RESULT_RELATIVE_PATH),
    ]
    modes = classify_failure_modes(exp5051, exp5064, exp5077, attempts)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": HONEST_VERDICT,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "fr11_attempts_summarized": attempts,
        "recurring_failure_modes": modes,
        "safe_next_mechanisms": safe_next_mechanisms(modes),
        "retired_mechanisms": retired_mechanisms(),
        "promotion_blockers": promotion_blockers(attempts, exp5077),
        "docs_update_recommendations": docs_update_recommendations(),
        "flagged_adversarial": False,
        "upstream_flagged_adversarial_sources": upstream_flagged_sources(exp5077),
        "promotion_executed": False,
        "promoted_updates": [],
        "source_artifacts": source_artifacts(root),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if not str(artifact.get("honest_verdict") or "").startswith(HONEST_VERDICT):
        errors.append("honest_verdict")
    if number(artifact.get("duration_s")) is None:
        errors.append("duration_s")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    for field in (
        "fr11_attempts_summarized",
        "recurring_failure_modes",
        "safe_next_mechanisms",
        "retired_mechanisms",
        "promotion_blockers",
        "docs_update_recommendations",
    ):
        if not isinstance(artifact.get(field), list) or not artifact.get(field):
            errors.append(field)
    if artifact.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not isinstance(principles.get(field), Mapping)
        or not str(principles.get(field, {}).get("principle") or "")
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles")
    if artifact.get("promotion_executed") is not False:
        errors.append("promotion_executed")
    if artifact.get("promoted_updates") != []:
        errors.append("promoted_updates")
    attempts = artifact.get("fr11_attempts_summarized")
    if isinstance(attempts, list) and [row.get("experiment_id") for row in attempts] != [5051, 5064, 5077]:
        errors.append("fr11_attempts_summarized")
    modes = artifact.get("recurring_failure_modes")
    if isinstance(modes, list) and {row.get("mode") for row in modes} != set(CONTROLLED_FAILURE_MODES):
        errors.append("recurring_failure_modes")
    return sorted(dict.fromkeys(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    start = now()
    exp5051, exp5064, exp5077 = load_inputs(root)
    duration_s = max(0.0, now() - start)
    artifact = build_artifact(
        root=Path(root),
        exp5051=exp5051,
        exp5064=exp5064,
        exp5077=exp5077,
        duration_s=duration_s,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError(f"Exp5078 artifact schema errors: {errors}")
    if write:
        write_json(Path(artifact_path) if artifact_path is not None else Path(root) / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
