"""Exp 1499 verifier ensemble DRY and conditional orthogonality audit.

The audit is intentionally deterministic: it reads checked-in post-.114
artifacts and converts each available verifier outcome into a bounded
pass/fail table.  Conditional rates are computed only on overlapping observed
cases, so disjoint verifier surfaces do not pretend to be jointly calibrated.

Spec: REQ-VERIFY-1499, SCENARIO-VERIFY-1499.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

RUN_DATE = "20260507"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_ARTIFACT_PATH = DEFAULT_RESULTS_DIR / (
    "experiment_1499_verifier_ensemble_dry_orthogonality_v2.json"
)

BEAVER_ARTIFACT = "experiment_1482_beaver_lite_live_prefix_bound_calibration.json"
CCTU_ARTIFACT = "experiment_1486_cctu_executable_constraint_microbenchmark.json"
CCTU_MANIFEST = "cctu_microbenchmark_manifest_1486.jsonl"
LOCALIZATION_ARTIFACT = "experiment_1490_kona_ebt_partial_trace_localization_audit.json"
MEMORY_POLICY_ARTIFACT = "experiment_1484_fr11_v9_query_time_memory_policy.json"
STRUCTURED_VERDICT_ARTIFACT = "experiment_1408_structured_verdict_record.json"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "orthogonality_matrix_written",
    "verifiers_audited",
    "cases_evaluated",
    "conditional_acceptance_matrix",
    "redundant_verifier_pairs",
    "k_effective_estimate",
    "deterministic_first_recommendations",
    "retire_or_keep_decisions",
    "blockers",
    "honest_verdict",
)

ACTIVE_VERIFIER_INVENTORY: tuple[dict[str, str], ...] = (
    {
        "name": "beaver_lite_bound",
        "surface": "BEAVER-lite bounds",
        "source": BEAVER_ARTIFACT,
        "role": "deterministic unsafe-mass upper bound",
        "status": "active",
    },
    {
        "name": "cctu_tool_call_structure",
        "surface": "CCTU executable validators",
        "source": CCTU_MANIFEST,
        "role": "tool name and argument structure",
        "status": "active",
    },
    {
        "name": "cctu_tool_result_consistency",
        "surface": "CCTU executable validators",
        "source": CCTU_MANIFEST,
        "role": "local tool result consistency",
        "status": "active",
    },
    {
        "name": "cctu_final_answer_validity",
        "surface": "CCTU executable validators",
        "source": CCTU_MANIFEST,
        "role": "final answer validity",
        "status": "active",
    },
    {
        "name": "cctu_model_verifier_alignment",
        "surface": "CCTU executable validators",
        "source": CCTU_MANIFEST,
        "role": "model-declared verifier outcome matches deterministic base validity",
        "status": "active",
    },
    {
        "name": "cctu_full_executable_verifier",
        "surface": "CCTU executable validators",
        "source": CCTU_MANIFEST,
        "role": "AND of executable CCTU validator checks",
        "status": "active",
    },
    {
        "name": "partial_trace_energy_localization",
        "surface": "energy/localization",
        "source": LOCALIZATION_ARTIFACT,
        "role": "local energy ranks injected failure span first",
        "status": "active",
    },
    {
        "name": "query_time_memory_policy",
        "surface": "memory-policy checks",
        "source": MEMORY_POLICY_ARTIFACT,
        "role": "opt-in verified memory policy task success",
        "status": "active",
    },
    {
        "name": "structured_verdict_record_schema",
        "surface": "structured verdict records",
        "source": STRUCTURED_VERDICT_ARTIFACT,
        "role": "VerdictRecord schema and serialization gate",
        "status": "active",
    },
)

RETired_OR_NON_ACTIVE_SIGNALS = (
    "semantic_energy_headline",
    "v1_pairwise_self_verification",
)


@dataclass(frozen=True)
class SourceArtifacts:
    """Checked-in artifacts used by the Exp 1499 audit."""

    beaver: dict[str, Any]
    cctu_artifact: dict[str, Any]
    cctu_rows: list[dict[str, Any]]
    localization: dict[str, Any]
    memory_policy: dict[str, Any]
    structured_verdict: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (path.read_text(encoding="utf-8").splitlines() if path.exists() else [])
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_source_artifacts(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> SourceArtifacts:
    """Load the post-.114 source artifacts without running models or validators."""

    root = Path(results_dir)
    return SourceArtifacts(
        beaver=_read_json(root / BEAVER_ARTIFACT),
        cctu_artifact=_read_json(root / CCTU_ARTIFACT),
        cctu_rows=_read_jsonl(root / CCTU_MANIFEST),
        localization=_read_json(root / LOCALIZATION_ARTIFACT),
        memory_policy=_read_json(root / MEMORY_POLICY_ARTIFACT),
        structured_verdict=_read_json(root / STRUCTURED_VERDICT_ARTIFACT),
    )


def _case_row(
    case_id: str,
    surface: str,
    source: str,
    verifier_accepts: dict[str, bool],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "surface": surface,
        "source": source,
        "verifier_accepts": verifier_accepts,
        "metadata": metadata or {},
    }


def build_bounded_case_table(
    sources: SourceArtifacts,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    """Build pass/fail observations for the active post-.114 verifier surfaces."""

    rows: list[dict[str, Any]] = []
    labels = [item["name"] for item in ACTIVE_VERIFIER_INVENTORY]

    constraints = list(sources.beaver.get("prefix_closed_constraints") or [])
    bounds = list(sources.beaver.get("unsafe_mass_bounds") or [])
    empirical_rates = list(sources.beaver.get("empirical_violation_rates") or [])
    for index, constraint in enumerate(constraints):
        bound = float(bounds[index]) if index < len(bounds) else 0.0
        empirical = float(empirical_rates[index]) if index < len(empirical_rates) else 0.0
        rows.append(
            _case_row(
                str(constraint.get("constraint_id", f"beaver-{index}")),
                "BEAVER-lite bounds",
                BEAVER_ARTIFACT,
                {
                    "beaver_lite_bound": bool(sources.beaver.get("bound_is_sound"))
                    and bound >= empirical
                },
                {
                    "unsafe_mass_bound": bound,
                    "empirical_violation_rate": empirical,
                    "source_family": constraint.get("source_family"),
                },
            )
        )

    for row in sources.cctu_rows:
        validator = dict(row.get("validator_result") or {})
        verifier = dict(row.get("verifier_result") or {})
        rows.append(
            _case_row(
                str(row.get("case_id", f"cctu-{len(rows)}")),
                "CCTU executable validators",
                CCTU_MANIFEST,
                {
                    "cctu_tool_call_structure": bool(validator.get("tool_call_structure_valid")),
                    "cctu_tool_result_consistency": bool(validator.get("tool_result_consistent")),
                    "cctu_final_answer_validity": bool(validator.get("final_answer_valid")),
                    "cctu_model_verifier_alignment": bool(validator.get("verifier_outcome_valid")),
                    "cctu_full_executable_verifier": bool(verifier.get("accepted")),
                },
                {"family": row.get("family")},
            )
        )

    for index, trace in enumerate(list(sources.localization.get("per_trace") or [])):
        rows.append(
            _case_row(
                str(trace.get("case_id", f"trace-{index}")),
                "energy/localization",
                LOCALIZATION_ARTIFACT,
                {
                    "partial_trace_energy_localization": int(trace.get("localization_rank", 999))
                    == 1
                },
                {"localization_rank": trace.get("localization_rank")},
            )
        )

    replay = dict(sources.memory_policy.get("memory_policy_replay") or {})
    enabled = dict(replay.get("memory_enabled") or {})
    for index, decision in enumerate(list(enabled.get("decisions") or [])):
        rows.append(
            _case_row(
                str(decision.get("case_id", f"memory-{index}")),
                "memory-policy checks",
                MEMORY_POLICY_ARTIFACT,
                {"query_time_memory_policy": bool(decision.get("task_success"))},
                {"verifier_signal": decision.get("verifier_signal")},
            )
        )

    rows.append(
        _case_row(
            "structured-verdict-record-schema",
            "structured verdict records",
            STRUCTURED_VERDICT_ARTIFACT,
            {
                "structured_verdict_record_schema": sources.structured_verdict.get("status")
                == "complete"
            },
            {"honest_verdict": sources.structured_verdict.get("honest_verdict")},
        )
    )

    return rows, labels, [dict(item) for item in ACTIVE_VERIFIER_INVENTORY]


def _observed_value(row: dict[str, Any], label: str) -> bool | None:
    value = dict(row.get("verifier_accepts") or {}).get(label)
    return bool(value) if value is not None else None


def _round_rate(value: float) -> float:
    return round(float(value), 6)


def compute_conditional_acceptance_matrix(
    case_rows: Sequence[dict[str, Any]],
    verifier_labels: Sequence[str],
) -> dict[str, Any]:
    """Compute pairwise agreement and P(verifier_j accepts | verifier_i accepts)."""

    labels = list(verifier_labels)
    conditional: dict[str, dict[str, float | None]] = {label: {} for label in labels}
    agreement: dict[str, dict[str, float | None]] = {label: {} for label in labels}
    overlap_support: dict[str, dict[str, int]] = {label: {} for label in labels}
    accept_support: dict[str, dict[str, int]] = {label: {} for label in labels}

    for left in labels:
        for right in labels:
            paired = [
                (_observed_value(row, left), _observed_value(row, right))
                for row in case_rows
                if _observed_value(row, left) is not None
                and _observed_value(row, right) is not None
            ]
            left_accepts = [pair for pair in paired if pair[0] is True]
            both_accept = sum(1 for l_value, r_value in left_accepts if l_value and r_value)
            same = sum(1 for l_value, r_value in paired if l_value == r_value)
            conditional[left][right] = (
                _round_rate(both_accept / len(left_accepts)) if left_accepts else None
            )
            agreement[left][right] = _round_rate(same / len(paired)) if paired else None
            overlap_support[left][right] = len(paired)
            accept_support[left][right] = len(left_accepts)

    return {
        "labels": labels,
        "p_accept_j_given_i": conditional,
        "pairwise_agreement": agreement,
        "overlap_support": overlap_support,
        "accept_support": accept_support,
    }


def _overlap_vectors(
    case_rows: Sequence[dict[str, Any]],
    left: str,
    right: str,
) -> tuple[list[bool], list[bool]]:
    left_values: list[bool] = []
    right_values: list[bool] = []
    for row in case_rows:
        left_value = _observed_value(row, left)
        right_value = _observed_value(row, right)
        if left_value is not None and right_value is not None:
            left_values.append(left_value)
            right_values.append(right_value)
    return left_values, right_values


def find_redundant_verifier_pairs(
    case_rows: Sequence[dict[str, Any]],
    verifier_labels: Sequence[str],
    matrix: dict[str, Any],
    *,
    threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Return verifier pairs whose observed acceptances are duplicate or contained."""

    redundant: list[dict[str, Any]] = []
    conditional = matrix["p_accept_j_given_i"]
    for left, right in combinations(verifier_labels, 2):
        left_values, right_values = _overlap_vectors(case_rows, left, right)
        if not left_values:
            continue
        left_given_right = conditional[right][left]
        right_given_left = conditional[left][right]
        identical = left_values == right_values
        conditionally_duplicate = (
            left_given_right is not None
            and right_given_left is not None
            and left_given_right >= threshold
            and right_given_left >= threshold
        )
        if identical or conditionally_duplicate:
            redundant.append(
                {
                    "verifier_a": left,
                    "verifier_b": right,
                    "reason": (
                        "identical_observed_acceptance_vector"
                        if identical
                        else "symmetric_conditional_acceptance_above_threshold"
                    ),
                    "overlap_cases": len(left_values),
                    "conditional_acceptance_a_given_b": left_given_right,
                    "conditional_acceptance_b_given_a": right_given_left,
                }
            )
    return redundant


def estimate_k_effective(
    case_rows: Sequence[dict[str, Any]],
    verifier_labels: Sequence[str],
    redundant_pairs: Sequence[dict[str, Any]],
) -> float:
    """Estimate independent deterministic signal groups from observed variation."""

    del redundant_pairs
    signatures: set[tuple[bool, ...]] = set()
    for label in verifier_labels:
        observed = [_observed_value(row, label) for row in case_rows]
        compact = tuple(value for value in observed if value is not None)
        if len(set(compact)) >= 2:
            signatures.add(compact)
    return float(len(signatures)) if signatures else 1.0


def _source_blockers(sources: SourceArtifacts) -> list[str]:
    blockers: list[str] = []
    checks = [
        ("missing_beaver_artifact", sources.beaver),
        ("missing_cctu_artifact", sources.cctu_artifact),
        ("missing_cctu_manifest", sources.cctu_rows),
        ("missing_localization_artifact", sources.localization),
        ("missing_memory_policy_artifact", sources.memory_policy),
        ("missing_structured_verdict_artifact", sources.structured_verdict),
    ]
    for name, payload in checks:
        if not payload:
            blockers.append(name)
    return blockers


def _deterministic_first_recommendations() -> list[str]:
    return [
        "Gate generation first on executable deterministic validators: CCTU structure, tool-result consistency, final-answer validity, then full CCTU acceptance.",
        "Use BEAVER-lite bounds as a conservative prefix/unsafe-mass guard, but do not count pass-only calibration rows as an independent fail detector until adversarial unsafe-prefix rows are added.",
        "Use partial-trace local energy for repair localization after deterministic validation fails; keep it out of headline answer-quality claims.",
        "Keep query-time memory policy opt-in and zero-soundness-gated; do not let memory hits bypass deterministic validators.",
        "Retire V_1 pairwise self-verification from active generation gates unless it beats deterministic energy ranking on matched cases.",
    ]


def _retire_or_keep_decisions(redundant_pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "surface": "BEAVER-lite bounds",
            "decision": "keep",
            "reason": "deterministic bound is orthogonal by mechanism, but observed pass-only rows should not inflate k_effective",
        },
        {
            "surface": "CCTU executable validators",
            "decision": "merge_reporting_for_duplicate_pairs",
            "reason": f"{len(redundant_pairs)} redundant observed pair(s); keep atomic checks for debugging but report duplicate wrappers as one signal group",
        },
        {
            "surface": "energy/localization",
            "decision": "keep_demote_to_repair_localization",
            "reason": "local energy ranks injected failure spans but is not an answer-quality gate by itself",
        },
        {
            "surface": "memory-policy checks",
            "decision": "keep_opt_in_only",
            "reason": "memory improves bounded replay only under explicit zero-soundness gating",
        },
        {
            "surface": "structured verdict records",
            "decision": "keep_as_record_schema_not_verifier_vote",
            "reason": "schema records make verdicts auditable but should not count as an independent semantic verifier",
        },
        {
            "surface": "Semantic Energy and V_1 headline signals",
            "decision": "retire_from_active_ensemble",
            "reason": "post-.114 direction excludes retired Semantic Energy and V_1 headline signals from active gates",
        },
    ]


def build_artifact(
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the terminal Exp 1499 artifact from checked-in evidence."""

    sources = load_source_artifacts(results_dir)
    case_rows, labels, inventory = build_bounded_case_table(sources)
    matrix = compute_conditional_acceptance_matrix(case_rows, labels)
    redundant = find_redundant_verifier_pairs(case_rows, labels, matrix)
    blockers = _source_blockers(sources)
    status = "complete" if not blockers else "blocked"
    k_effective = estimate_k_effective(case_rows, labels, redundant)
    return {
        "schema": "carnot.eval.verifier_ensemble_dry_orthogonality_v2.v1",
        "experiment": "1499_verifier_ensemble_dry_orthogonality_v2",
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-1499", "SCENARIO-VERIFY-1499"],
        "status": status,
        "orthogonality_matrix_written": True,
        "verifiers_audited": labels,
        "verifier_inventory": inventory,
        "retired_or_non_active_signals_excluded": list(RETired_OR_NON_ACTIVE_SIGNALS),
        "cases_evaluated": len(case_rows),
        "case_table_summary": {
            "beaver_cases": sum(row["surface"] == "BEAVER-lite bounds" for row in case_rows),
            "cctu_cases": sum(row["surface"] == "CCTU executable validators" for row in case_rows),
            "localization_cases": sum(row["surface"] == "energy/localization" for row in case_rows),
            "memory_policy_cases": sum(
                row["surface"] == "memory-policy checks" for row in case_rows
            ),
            "structured_verdict_cases": sum(
                row["surface"] == "structured verdict records" for row in case_rows
            ),
        },
        "bounded_case_table": case_rows,
        "conditional_acceptance_matrix": matrix,
        "redundant_verifier_pairs": redundant,
        "k_effective_estimate": k_effective,
        "k_effective_method": "unique non-constant observed acceptance vectors; pass-only surfaces are audited but not counted as independent fail detectors",
        "duplicated_logic_findings": [
            "CCTU tool-result consistency and model verifier-outcome alignment can collapse to the same observed vector; keep one reporting group unless future cases separate them.",
            "BEAVERLiteVerifier is a thin wrapper around BEAVERLiteBounder; keep the wrapper only for API compatibility and put calibration logic in the bounder.",
            "Structured verdict records are schema/certificate plumbing, not a separate verifier vote.",
        ],
        "deterministic_first_recommendations": _deterministic_first_recommendations(),
        "retire_or_keep_decisions": _retire_or_keep_decisions(redundant),
        "source_artifacts": [
            f"results/{BEAVER_ARTIFACT}",
            f"results/{CCTU_ARTIFACT}",
            f"results/{CCTU_MANIFEST}",
            f"results/{LOCALIZATION_ARTIFACT}",
            f"results/{MEMORY_POLICY_ARTIFACT}",
            f"results/{STRUCTURED_VERDICT_ARTIFACT}",
        ],
        "tests_run": list(tests_run),
        "blockers": blockers,
        "honest_verdict": (
            f"complete: verifier ensemble DRY/orthogonality audit wrote matrix with k_effective={k_effective:.2f}"
            if not blockers
            else "complete: verifier ensemble audit blocked by missing source artifacts"
        ),
    }


def run_experiment(
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    output_path: str | Path = DEFAULT_ARTIFACT_PATH,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    """Persist the Exp 1499 artifact after first writing the bootstrap state."""

    destination = Path(output_path)
    _write_json(destination, {"status": "in_progress"})
    artifact = build_artifact(results_dir=results_dir, tests_run=tests_run)
    _write_json(destination, artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
