"""Adjudicate archived acquisition evidence without rerunning acquisition.

This module treats the old artifact as evidence, not as a benchmark launcher.
It copies exact bytes, recomputes the frozen paired statistics, and preserves
the old disqualification. No model, executor acquisition, or board runs here.

Spec refs: REQ-REPORT-7364 and SCENARIO-REPORT-7364-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.experiment_7351_v645_acquisition_prototype import ARMS, RESAMPLING_SEED
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.646"
EXPERIMENT_ID = "exp7364-acquisition-adjudication"
SCHEMA = "carnot.exp7364.v646.acquisition_adjudication.v1"
RESULT_PATH = Path("results/experiment_7364_v646_acquisition_adjudication.json")
RAW_DIR = Path("results/raw/experiment_7364_v646_acquisition_adjudication")
MODULE_PATH = Path("python/carnot/experiment_7364_v646_acquisition_adjudication.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7364_v646_acquisition_adjudication.py")
TEST_PATH = Path("tests/python/test_experiment_7364_v646_acquisition_adjudication.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
VALIDATION_CONTRACT_PATH = Path("results/experiment_7358_v646_validation_contract.json")
SOURCE_ARTIFACT_PATH = Path("results/experiment_7351_v645_acquisition_prototype.json")
SOURCE_RAW_DIR = Path("results/raw/experiment_7351_v645_acquisition_prototype")
SOURCE_ROWS_PATH = SOURCE_RAW_DIR / "evidence/comparative_rows.json"

REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
FINITE_RELATION_VOCABULARY = (
    "unconstrained",
    "eq",
    "neq",
    "next",
    "prev",
    "eq_or_next",
    "eq_or_prev",
)
ZERO_CURRENT_INVOCATIONS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7351_v645_acquisition_prototype.py"),
    Path("tests/python/test_experiment_7351_v645_acquisition_prototype.py"),
    SOURCE_ARTIFACT_PATH,
    SOURCE_ROWS_PATH,
    VALIDATION_CONTRACT_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
AFFECTED_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


class AdjudicationError(RuntimeError):
    """Stop when archived evidence or terminal validation cannot be trusted."""


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print every phase boundary with real monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7364] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_object(path: Path) -> JsonDict:
    """Load a JSON object while representing missing external bytes as empty."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _canonical_hash(value: Any) -> str:
    """Hash stable JSON so mutations have one deterministic identity."""

    return validation_contract.canonical_hash(value)


def _precondition(
    check: str, upstream: str, field: str, expected: Any, observed: Any, passed: bool
) -> JsonDict:
    """Record one exact prerequisite instead of reducing it to a Boolean."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def check_upstream_contract(
    producer: Mapping[str, Any], available: bool, excluded: bool
) -> list[JsonDict]:
    """Check the three YAML gates and all terminal eligibility boundaries."""

    if not available:
        return [
            _precondition(
                "validation_contract_path",
                VALIDATION_CONTRACT_PATH.as_posix(),
                "path",
                "readable_nonempty_json",
                "missing",
                False,
            )
        ]
    verdict = producer.get("verdict_class")
    status = str(producer.get("status", "missing"))
    allowed_verdicts = ["positive", "circular_positive", "null"]
    return [
        _precondition(
            "validation_contract_ready",
            VALIDATION_CONTRACT_PATH.as_posix(),
            "validation_contract_ready_score",
            1,
            producer.get("validation_contract_ready_score", "missing"),
            producer.get("validation_contract_ready_score") == 1,
        ),
        _precondition(
            "validation_contract_verdict",
            VALIDATION_CONTRACT_PATH.as_posix(),
            "verdict_class",
            allowed_verdicts,
            verdict,
            verdict in allowed_verdicts,
        ),
        _precondition(
            "validation_contract_adversarial",
            VALIDATION_CONTRACT_PATH.as_posix(),
            "flagged_adversarial",
            False,
            producer.get("flagged_adversarial", "missing"),
            producer.get("flagged_adversarial") is False,
        ),
        _precondition(
            "validation_contract_terminal_status",
            VALIDATION_CONTRACT_PATH.as_posix(),
            "status",
            "terminal_not_blocked_partial_or_disqualified",
            status,
            not any(token in status for token in ("blocked", "partial", "disqualified")),
        ),
        _precondition(
            "validation_contract_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
            not excluded,
        ),
    ]


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, str]]:
    """Authenticate every declared path and Exp7358 gate before archive work."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else "missing_or_empty",
                available,
            )
        )
        if available:
            hashes[relative.as_posix()] = validation_contract.sha256_file(path)

    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7364",
            "REQ-REPORT-7364" if "REQ-REPORT-7364" in spec_text else "missing",
            "REQ-REPORT-7364" in spec_text,
        )
    )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    producer_path = root / VALIDATION_CONTRACT_PATH
    producer = _load_object(producer_path)
    producer_available = bool(producer) and producer_path.stat().st_size > 0
    checks.extend(
        check_upstream_contract(
            producer,
            producer_available,
            "experiment_id: 7358" in exclusion_text
            or "exp7358-validation-contract" in exclusion_text,
        )
    )
    current_excluded = "experiment_id: 7364" in exclusion_text or EXPERIMENT_ID in exclusion_text
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            current_excluded,
            not current_excluded,
        )
    )
    return checks, hashes


def hash_copy_files(root: Path, sources: Sequence[Path], destination: Path) -> list[JsonDict]:
    """Copy exact bytes and retain both hashes so the copy can be audited."""

    inventory: list[JsonDict] = []
    root_resolved = root.resolve()
    for index, source in enumerate(sources):
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(root_resolved)
        except ValueError:
            relative = Path("external") / f"{index:03d}_{resolved.name}"
        copied = destination / "archive" / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved, copied)
        source_hash = validation_contract.sha256_file(resolved)
        copy_hash = validation_contract.sha256_file(copied)
        inventory.append(
            {
                "source_path": relative.as_posix(),
                "copy_path": str(copied),
                "source_sha256": source_hash,
                "copy_sha256": copy_hash,
                "bytes": resolved.stat().st_size,
                "matched": source_hash == copy_hash,
            }
        )
    return inventory


def evidence_signatures(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Bind costs, paid calls, and identities separately for useful diagnostics."""

    return {
        "complete_costs": _canonical_hash(
            [
                [row.get("panel"), row.get("context_id"), row.get("arm"), row.get("full_cost_s")]
                for row in rows
            ]
        ),
        "paid_queries": _canonical_hash(
            [
                [row.get("panel"), row.get("context_id"), row.get("arm"), row.get("query_count")]
                for row in rows
            ]
        ),
        "context_ids": _canonical_hash(
            [[row.get("panel"), row.get("context_id"), row.get("arm")] for row in rows]
        ),
        "rows": _canonical_hash(list(rows)),
    }


def _paired_ratio_ci(rows: Sequence[Mapping[str, Any]], field: str, seed: int) -> JsonDict:
    """Repeat the original context-cluster ratio bootstrap without new samples."""

    by_context: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("panel") != "evaluation":
            continue
        by_context.setdefault(str(row["context_id"]), {})[str(row["arm"])] = float(row[field])
    pairs = [
        (values["finite_bias_elimination"], values["conservative_acquisition"])
        for values in by_context.values()
        if {"finite_bias_elimination", "conservative_acquisition"} <= set(values)
    ]
    observed = sum(left for left, _ in pairs) / sum(right for _, right in pairs)
    generator = random.Random(seed)
    samples: list[float] = []
    for _index in range(2_000):
        draw = [pairs[generator.randrange(len(pairs))] for _pair in pairs]
        samples.append(sum(left for left, _ in draw) / sum(right for _, right in draw))
    samples.sort()
    return {
        "estimate": observed,
        "lower": samples[50],
        "upper": samples[1950],
        "context_clusters": len(pairs),
        "resamples": 2_000,
        "resampling_seed": seed,
        "numerator_arm": "finite_bias_elimination",
        "denominator_arm": "conservative_acquisition",
        "field": field,
    }


def _paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep every context and all arm boundary costs beside their pair ratios."""

    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["panel"]), str(row["context_id"]))
        grouped.setdefault(key, {})[str(row["arm"])] = row
    paired: list[JsonDict] = []
    retained = (
        "full_cost_s",
        "query_count",
        "query_attempt_count",
        "coverage",
        "utility",
        "returned_infeasible_count",
        "censored",
    )
    for (panel, context_id), arms in sorted(grouped.items()):
        arm_values = {
            arm: {field: arms[arm].get(field) for field in retained} for arm in ARMS if arm in arms
        }
        finite = arms.get("finite_bias_elimination", {})
        conservative = arms.get("conservative_acquisition", {})
        paired.append(
            {
                "panel": panel,
                "context_id": context_id,
                "arms": arm_values,
                "complete_cost_ratio": float(finite.get("full_cost_s", 0.0))
                / float(conservative.get("full_cost_s", 1.0)),
                "paid_query_ratio": float(finite.get("query_count", 0.0))
                / float(conservative.get("query_count", 1.0)),
                "included_in_scientific_interval": panel == "evaluation",
            }
        )
    return paired


def reconcile_archived_evidence(
    source_artifact: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    expected_signatures: Mapping[str, Any],
) -> JsonDict:
    """Cold-reduce row copies and report each independently anchored invariant."""

    errors: list[str] = []
    inline_rows = source_artifact.get("rows", [])
    if list(raw_rows) != inline_rows:
        errors.append("row_copy")
    actual_signatures = evidence_signatures(raw_rows)
    for name in ("complete_costs", "paid_queries", "context_ids", "rows"):
        if actual_signatures[name] != expected_signatures.get(name):
            errors.append(name)

    panels = Counter(str(row.get("panel")) for row in raw_rows)
    contexts = {
        panel: {str(row.get("context_id")) for row in raw_rows if row.get("panel") == panel}
        for panel in ("evaluation", "challenge")
    }
    expected_counts = {"evaluation": 90, "challenge": 36}
    for panel, count in expected_counts.items():
        if panels[panel] != count:
            errors.append(f"{panel}_rows")
    expected_contexts = {"evaluation": 30, "challenge": 12}
    for panel, count in expected_contexts.items():
        if len(contexts[panel]) != count:
            errors.append(f"{panel}_contexts")
    arm_sets = {
        (str(row.get("panel")), str(row.get("context_id"))): {
            str(candidate.get("arm"))
            for candidate in raw_rows
            if candidate.get("panel") == row.get("panel")
            and candidate.get("context_id") == row.get("context_id")
        }
        for row in raw_rows
    }
    if any(arms != set(ARMS) for arms in arm_sets.values()):
        errors.append("arm_sets")

    development = source_artifact.get("development_controls", {}).get("rows", [])
    development_ids = [row.get("context_id") for row in development]
    if len(development) != 12 or len(set(development_ids)) != 12:
        errors.append("development_records")

    evaluation = [row for row in raw_rows if row.get("panel") == "evaluation"]
    finite = [row for row in evaluation if row.get("arm") == "finite_bias_elimination"]
    conservative = [row for row in evaluation if row.get("arm") == "conservative_acquisition"]
    cost_ci = _paired_ratio_ci(raw_rows, "full_cost_s", RESAMPLING_SEED)
    paid_ci = _paired_ratio_ci(raw_rows, "query_count", RESAMPLING_SEED)
    reduction = {
        "complete_cost_ratio_ci95": cost_ci,
        "paid_query_ratio_ci95": paid_ci,
        "complete_cost_gate_threshold": 0.90,
        "complete_cost_gate_operator": "upper < threshold",
        "complete_cost_gate_passed": cost_ci["upper"] < 0.90,
        "returned_infeasible_count": sum(
            int(row.get("returned_infeasible_count", 0)) for row in raw_rows
        ),
        "finite_bias_coverage": sum(int(row.get("coverage", 0)) for row in finite),
        "conservative_coverage": sum(int(row.get("coverage", 0)) for row in conservative),
        "finite_bias_utility": sum(float(row.get("utility", 0.0)) for row in finite),
        "conservative_utility": sum(float(row.get("utility", 0.0)) for row in conservative),
        "query_ratio_is_diagnostic_only": True,
        "rows_sha256": actual_signatures["rows"],
    }
    original_ci = source_artifact.get("independent_reduction", {}).get(
        "paired_context_clustered_ci95", {}
    )
    for field in ("estimate", "lower", "upper", "context_clusters", "resamples"):
        if cost_ci.get(field) != original_ci.get(field):
            errors.append("original_cost_interval")
            break
    return {
        "errors": sorted(set(errors)),
        "row_accounting": {
            "evaluation_rows": panels["evaluation"],
            "evaluation_contexts": len(contexts["evaluation"]),
            "challenge_rows": panels["challenge"],
            "challenge_contexts": len(contexts["challenge"]),
            "development_records": len(development),
            "arms_per_context": len(ARMS),
        },
        "development_records": deepcopy(development),
        "paired_cost_rows": _paired_rows(raw_rows),
        "independent_reduction": reduction,
        "evidence_signatures": actual_signatures,
    }


def diagnose_original_validation(source_artifact: Mapping[str, Any]) -> JsonDict:
    """Trace the false validation value to the broad receipt that caused it."""

    health = source_artifact.get("repository_health", {})
    broad = health.get("current_observation")
    causal = dict(broad) if isinstance(broad, Mapping) else None
    if causal is not None:
        causal["affects_required_checks"] = health.get("affects_required_checks")
    scoped_lists = {
        name: list(source_artifact.get(name, []))
        for name in (
            "missing_required_commands",
            "failed_required_commands",
            "duplicate_required_commands",
        )
    }
    supported = bool(
        source_artifact.get("required_checks_passed") is False
        and causal
        and causal.get("name") == "full_python_suite"
        and causal.get("passed") is False
        and (causal.get("exit_code") != 0 or causal.get("timed_out") is True)
        and health.get("affects_required_checks") is True
    )
    required_inventory = [*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES, "full_python_suite"]
    observed_names = [
        str(row.get("name")) for row in source_artifact.get("validation_receipts", [])
    ] + ([str(causal.get("name"))] if causal else [])
    return {
        "supported": supported,
        "diagnostic_class": "diagnostic_correction_not_new_experiment",
        "original_required_checks_passed": source_artifact.get("required_checks_passed"),
        "required_receipt_inventory": required_inventory,
        "observed_receipt_names": observed_names,
        "missing_receipt_names": sorted(set(required_inventory) - set(observed_names)),
        "scoped_error_lists": scoped_lists,
        "causal_receipt": causal,
        "provenance": (
            "Exp7351 main made the repository-wide full suite a required observation and set "
            "required_checks_passed=false when that child timed out. Empty scoped error lists "
            "describe only the eight scoped checks."
        ),
        "historical_disqualification_cleared": False,
    }


def build_authority_support_rows() -> list[JsonDict]:
    """State which method assumptions the archived vocabulary can support."""

    common = {"claim_extends_beyond_actual_vocabulary": False}
    return [
        {
            **common,
            "constraint_family": "archived_finite_pairwise_relations",
            "authority_source": "exact_private_executor_boolean_feedback",
            "assumptions": "Domain three, binary scopes, stable version token, existential feedback.",
            "actual_vocabulary": list(FINITE_RELATION_VOCABULARY),
            "supported": True,
            "counterexample_boundary": "A ternary rule or relation outside the seven tables is unsupported.",
        },
        {
            **common,
            "constraint_family": "cca_pairwise_separation",
            "authority_source": "paper_assumption_plus_exact_boolean_feasibility",
            "assumptions": "Pairwise separation belongs to the declared CCA schedule language.",
            "actual_vocabulary": ["neq"],
            "supported": True,
            "counterexample_boundary": "No arbitrary compound or hidden drift claim follows.",
        },
        {
            **common,
            "constraint_family": "cca_global_capacity",
            "authority_source": "paper_assumption_not_instantiated_in_exp7351_vocabulary",
            "assumptions": "CCA permits a named global capacity family under exact feedback.",
            "actual_vocabulary": [],
            "supported": False,
            "counterexample_boundary": "Exp7351 contains no global-capacity atom or measurement.",
        },
        {
            **common,
            "constraint_family": "higher_order_constraints",
            "authority_source": "exact_executor_challenge_feedback_without_supported_representation",
            "assumptions": "Ternary challenge answers test abstention, not binary-rule authority.",
            "actual_vocabulary": [],
            "supported": False,
            "counterexample_boundary": "A ternary not-all-equal rule cannot justify pairwise atoms.",
        },
        {
            **common,
            "constraint_family": "t_oracle_fastca_finite_bias",
            "authority_source": "paper_finite_bias_assumption_with_exp7351_exact_feedback_only",
            "assumptions": "The target must belong to the finite bias; query count and time stay separate.",
            "actual_vocabulary": list(FINITE_RELATION_VOCABULARY),
            "supported": True,
            "counterexample_boundary": "Support ends when the target is outside these seven tables.",
        },
        {
            **common,
            "constraint_family": "neural_oracle_approximation",
            "authority_source": "approximate_learned_labels_not_used_by_exp7351",
            "assumptions": "Learned labels can be wrong and are not exact executor authority.",
            "actual_vocabulary": [],
            "supported": False,
            "counterexample_boundary": "No neural-oracle accuracy or authority was measured.",
        },
    ]


def _field_principles() -> dict[str, str]:
    """Explain required fields without wrapping their ordinary values."""

    return {
        "schema": "Version this record and retain ordinary top-level experiment_id and milestone.",
        "status": "Terminal only after actual work and affected validation.",
        "run_date": "Use 20260917 and retain real UTC boundaries.",
        "preconditions_checked": "Record exact paths and producer fields before dependent work.",
        "MODEL_SPECS": "No current model work is intended, so this list is empty.",
        "model_invoked": "True would mean any attempted current model load or generation.",
        "invocation_counts": "Keep current zero calls separate from the historical source record.",
        "inference_substrate": "Name actual host hashing and deterministic evidence aggregation.",
        "inference_substrate_class": "Aggregation is the closed class for this archival task.",
        "execution_venue": "Host means no new board execution occurred.",
        "duration_s": "Measure real monotonic elapsed time without padding.",
        "phase_spans": "Keep measured load, generation, evaluation, validation, and write spans disjoint.",
        "random_seed": "Reuse the frozen Exp7351 development, evaluation, and resampling seeds.",
        "reproducibility_checksum": "Bind code, inputs, copied evidence, rows, reduction, and gates.",
        "source_artifact_hashes": "Retain exact producer and input byte hashes.",
        "rows": "Keep all 90 evaluation and 36 challenge arm rows.",
        "sample_size_budget": "Preserve planned, attempted, completed, censored, and stopping values.",
        "acceptance_gate_results": "Keep expected, observed, and passed distinct for every gate.",
        "gate_check_summary": "Name each failed gate and exact expected and observed value.",
        "verifier_is_oracle": "The archived exact executor defines truth, so circularity remains.",
        "honest_verdict": "Distinguish complete accounting, null value, and unavailable evidence.",
        "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "A current critical finding prevents completion and promotion.",
        "validation_receipts": "Retain current command, scope, exit, elapsed time, and log hash.",
        "repository_health": "Keep the dated historical broad-suite failure separate from current checks.",
        "field_principles": "Explain fields separately from their scalar or dictionary values.",
        "acquisition_adjudication_complete_score": "One means cold accounting completed; it is not readiness or value.",
        "original_validation_discrepancy": "Name the original required receipt and its observed failure.",
        "paired_cost_rows": "Keep every archived context, arm cost, paid query, and pair disposition.",
        "authority_support_rows": "Bound each authority claim to its constraint family and assumptions.",
        "unchanged_method_disposition": "Retire an unchanged benchmark until a named mechanism changes.",
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, str], started_at: str
) -> JsonDict:
    """Create a schema-complete shell before evidence changes classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 3,
        "status": "partial_owned_adjudication",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_CURRENT_INVOCATIONS),
            "historical": {
                "source_experiment_id": 7351,
                "model_invoked": False,
                "counts": deepcopy(ZERO_CURRENT_INVOCATIONS),
            },
        },
        "inference_substrate": "host_cpu_archived_evidence_hashing_and_deterministic_aggregation",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "development": 7_351_101,
            "evaluation": 7_351_211,
            "resampling": RESAMPLING_SEED,
            "source": "exp7351_frozen_before_outcomes",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": {},
        "acceptance_gate_results": {},
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "partial_owned_adjudication",
        "verdict_class": "partial",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "required_checks_passed": False,
        "missing_required_commands": list(REQUIRED_CHECK_NAMES),
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": {
            "status": "not_yet_copied",
            "affects_required_checks": False,
            "historical_failures": [],
        },
        "field_principles": _field_principles(),
        "acquisition_adjudication_complete_score": 0,
        "acquisition_readiness_score": 0,
        "acquisition_value_score": 0,
        "promotion_score": 0,
        "original_validation_discrepancy": {},
        "paired_cost_rows": [],
        "authority_support_rows": [],
        "unchanged_method_disposition": {},
        "archive_copy_inventory": [],
        "archive_copy_complete": False,
        "historical_inference_sidecars": [],
        "development_records": [],
        "row_accounting": {},
        "independent_reduction": {},
        "archived_evidence_signatures": {},
        "reconciliation_errors": [],
        "original_disposition": {},
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, str], duration_s: float
) -> JsonDict:
    """Publish an external prerequisite failure without dependent evidence work."""

    artifact = _base_artifact(checks, hashes, datetime.now(UTC).isoformat())
    failed = next(dict(row) for row in checks if row.get("passed") is not True)
    artifact.update(
        {
            "status": "blocked_acquisition_adjudication_precondition",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "duration_s": duration_s,
            "honest_verdict": "blocked_external_acquisition_adjudication_precondition",
            "verdict_class": "blocked",
            "missing_required_commands": [],
            "gate_check_summary": {
                "all_passed": False,
                "failed_count": sum(row.get("passed") is not True for row in checks),
                "first_failure": failed,
            },
            "sample_size_budget": {
                "planned_archived_rows": 126,
                "attempted_archived_rows": 0,
                "completed_archived_rows": 0,
                "censored_archived_rows": 126,
                "stopping_rule": "Stop before dependent work on an external prerequisite failure.",
            },
            "repository_health": {
                "status": "not_evaluated",
                "affects_required_checks": False,
                "historical_failures": [],
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _phase_span(phase: str, start: float, run_start: float, units: int) -> JsonDict:
    """Record one non-overlapping monotonic interval relative to run start."""

    end = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": start - run_start,
        "end_offset_s": end - run_start,
        "duration_s": end - start,
        "completed_units": units,
    }


def _receipt_integrity(root: Path, source_artifact: Mapping[str, Any]) -> list[str]:
    """Check that each historical receipt still names an exact archived log."""

    rows = list(source_artifact.get("validation_receipts", []))
    broad = source_artifact.get("repository_health", {}).get("current_observation")
    if isinstance(broad, Mapping):
        rows.append(broad)
    errors: list[str] = []
    names = [str(row.get("name")) for row in rows]
    expected = [*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES, "full_python_suite"]
    if Counter(names) != Counter(expected):
        errors.append("historical_receipt_inventory")
    for row in rows:
        path = root / str(row.get("log_path", ""))
        if not path.is_file() or validation_contract.sha256_file(path) != row.get("log_sha256"):
            errors.append(f"historical_receipt_log:{row.get('name')}")
    return sorted(set(errors))


def build_artifact(root: Path, raw_dir: Path) -> JsonDict:
    """Copy archived bytes and build a diagnostic-only cold adjudication."""

    run_start = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    checks, hashes = collect_preconditions(root)
    if any(row["passed"] is not True for row in checks):
        return build_blocked_artifact(checks, hashes, time.monotonic() - run_start)
    artifact = _base_artifact(checks, hashes, started_at)
    artifact["phase_spans"].append(_phase_span("load", run_start, run_start, len(checks)))
    zero = time.monotonic()
    artifact["phase_spans"].append(_phase_span("generation", zero, run_start, 0))

    copy_started = time.monotonic()
    source_files = [root / SOURCE_ARTIFACT_PATH]
    source_files.extend(
        sorted(path for path in (root / SOURCE_RAW_DIR).rglob("*") if path.is_file())
    )
    inventory = hash_copy_files(root, source_files, raw_dir)
    artifact["archive_copy_inventory"] = inventory
    artifact["archive_copy_complete"] = bool(inventory) and all(row["matched"] for row in inventory)
    artifact["source_artifact_hashes"].update(
        {row["source_path"]: row["source_sha256"] for row in inventory}
    )
    artifact["phase_spans"].append(
        _phase_span("archive_copy", copy_started, run_start, len(inventory))
    )

    evaluation_started = time.monotonic()
    copied_artifact = raw_dir / "archive" / SOURCE_ARTIFACT_PATH
    copied_rows = raw_dir / "archive" / SOURCE_ROWS_PATH
    source_artifact = _load_object(copied_artifact)
    row_payload = _load_object(copied_rows)
    rows = row_payload.get("rows", [])
    signatures = evidence_signatures(source_artifact.get("rows", []))
    reconciliation = reconcile_archived_evidence(source_artifact, rows, signatures)
    receipt_errors = _receipt_integrity(root, source_artifact)
    discrepancy = diagnose_original_validation(source_artifact)
    errors = [*reconciliation["errors"], *receipt_errors]
    if not discrepancy["supported"]:
        errors.append("original_validation_receipt")
    artifact.update(
        {
            "rows": deepcopy(rows),
            "sample_size_budget": deepcopy(source_artifact.get("sample_size_budget", {})),
            "original_validation_discrepancy": discrepancy,
            "paired_cost_rows": reconciliation["paired_cost_rows"],
            "authority_support_rows": build_authority_support_rows(),
            "development_records": reconciliation["development_records"],
            "row_accounting": reconciliation["row_accounting"],
            "independent_reduction": reconciliation["independent_reduction"],
            "archived_evidence_signatures": signatures,
            "reconciliation_errors": sorted(set(errors)),
            "original_disposition": {
                "experiment_id": source_artifact.get("experiment_id"),
                "status": source_artifact.get("status"),
                "verdict_class": source_artifact.get("verdict_class"),
                "honest_verdict": source_artifact.get("honest_verdict"),
                "required_checks_passed": source_artifact.get("required_checks_passed"),
                "flagged_adversarial": source_artifact.get("flagged_adversarial"),
                "acquisition_value_score": source_artifact.get("acquisition_value_score"),
                "promotion_score": source_artifact.get("promotion_score"),
                "preserved": True,
            },
            "repository_health": {
                "status": "degraded_open",
                "affects_required_checks": False,
                "historical_failures": [
                    {
                        "observed_at": "2026-09-16",
                        "source_experiment_id": 7351,
                        "classification": "historical_required_broad_suite_timeout",
                        "receipt": discrepancy.get("causal_receipt"),
                        "resolved": False,
                    }
                ],
                "current_observation": None,
            },
            "historical_inference_sidecars": [
                {
                    "path": SOURCE_RAW_DIR.joinpath(
                        "sidecars/model_shaped_evidence.json"
                    ).as_posix(),
                    "sha256": next(
                        (
                            row["source_sha256"]
                            for row in inventory
                            if row["source_path"].endswith("sidecars/model_shaped_evidence.json")
                        ),
                        None,
                    ),
                    "historical_only": True,
                    "current_model_work": False,
                    "label": "Historical model-shaped context; not a generation receipt.",
                }
            ],
            "unchanged_method_disposition": {
                "retired": True,
                "retired_proposal": "unchanged_finite_bias_elimination_complete_cost_benchmark",
                "reason": "The frozen complete-cost upper interval does not clear 0.90.",
                "future_evidence_required": (
                    "Name a mechanism change before freezing a new panel. Examples include a different "
                    "query-selection rule, a representation that supports a new constraint family, or a "
                    "measured reduction in complete service cost. A repeated unchanged benchmark is not authorized."
                ),
            },
        }
    )
    artifact["phase_spans"].append(
        _phase_span("evaluation", evaluation_started, run_start, len(rows))
    )
    artifact["duration_s"] = time.monotonic() - run_start
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    apply_terminal_state(artifact, require_terminal=False)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _has_receipts(artifact: Mapping[str, Any], names: Sequence[str]) -> bool:
    """Require exactly one passing current receipt for each declared check."""

    receipts = artifact.get("validation_receipts", [])
    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in receipts
        )
        == 1
        for name in names
    )


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Keep each gate's target, measurement, and decision distinct."""

    return {
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def apply_terminal_state(artifact: JsonDict, *, require_terminal: bool) -> None:
    """Classify accounting separately from historical readiness and value."""

    if artifact.get("verdict_class") == "blocked":
        return
    scoped = artifact.get("required_checks_passed") is True and _has_receipts(
        artifact, REQUIRED_CHECK_NAMES
    )
    terminal = _has_receipts(artifact, TERMINAL_CHECK_NAMES) if require_terminal else True
    archive = artifact.get("archive_copy_complete") is True
    reconciliation = not artifact.get("reconciliation_errors")
    diagnosis = artifact.get("original_validation_discrepancy", {}).get("supported") is True
    row_accounting = artifact.get("row_accounting") == {
        "evaluation_rows": 90,
        "evaluation_contexts": 30,
        "challenge_rows": 36,
        "challenge_contexts": 12,
        "development_records": 12,
        "arms_per_context": 3,
    }
    safety = artifact.get("independent_reduction", {}).get("returned_infeasible_count") == 0
    cost_value = artifact.get("independent_reduction", {}).get("complete_cost_gate_passed") is True
    adversarial = artifact.get("flagged_adversarial") is False
    gates = {
        "qualified_validation_contract": _gate(
            True,
            all(row.get("passed") is True for row in artifact.get("preconditions_checked", [])),
            all(row.get("passed") is True for row in artifact.get("preconditions_checked", [])),
            "The same-milestone validation contract must pass its exact YAML gates.",
        ),
        "archive_copy": _gate(True, archive, archive, "Every copied source hash must match."),
        "row_accounting": _gate(
            True,
            row_accounting,
            row_accounting,
            "All evaluation, challenge, development, and arm rows must exist.",
        ),
        "cold_reconciliation": _gate(
            [],
            artifact.get("reconciliation_errors"),
            reconciliation,
            "Independent signatures and reductions must agree.",
        ),
        "original_validation_diagnosed": _gate(
            True, diagnosis, diagnosis, "The original false value needs its causal receipt."
        ),
        "safety": _gate(
            0,
            artifact.get("independent_reduction", {}).get("returned_infeasible_count"),
            safety,
            "No archived returned plan may be infeasible.",
        ),
        "scientific_cost_value": _gate(
            "complete_cost_ratio_ci95.upper < 0.90",
            artifact.get("independent_reduction", {})
            .get("complete_cost_ratio_ci95", {})
            .get("upper"),
            cost_value,
            "The frozen complete-cost upper interval stays the scientific gate.",
        ),
        "affected_validation": _gate(True, scoped, scoped, "All current scoped checks must pass."),
        "terminal_validation": _gate(
            True, terminal, terminal, "All cold terminal readers must pass."
        ),
        "adversarial_clear": _gate(
            False,
            artifact.get("flagged_adversarial"),
            adversarial,
            "A critical finding prevents completion.",
        ),
    }
    artifact["acceptance_gate_results"] = gates
    accounting_names = (
        "qualified_validation_contract",
        "archive_copy",
        "row_accounting",
        "cold_reconciliation",
        "original_validation_diagnosed",
        "affected_validation",
        "terminal_validation",
        "adversarial_clear",
    )
    complete = all(gates[name]["passed"] for name in accounting_names)
    artifact["acquisition_adjudication_complete_score"] = int(complete)
    artifact["acquisition_readiness_score"] = 0
    artifact["acquisition_value_score"] = 0
    artifact["promotion_score"] = 0
    failed = [
        {
            "check": name,
            "artifact_field": f"acceptance_gate_results.{name}",
            "expected": row["expected"],
            "observed": row["observed"],
        }
        for name, row in gates.items()
        if not row["passed"]
    ]
    artifact["gate_check_summary"] = {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }
    if complete:
        artifact["status"] = "complete_acquisition_adjudication_null"
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "complete_null_acquisition_adjudication_preserves_disqualification_and_failed_cost_gate"
        )
    else:
        artifact["status"] = "complete_acquisition_adjudication_disqualified"
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = (
            "complete_disqualified_acquisition_adjudication_required_evidence_or_validation_failed"
        )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable evidence and decisions while excluding host timing metadata."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "random_seed",
        "source_artifact_hashes",
        "archive_copy_inventory",
        "rows",
        "sample_size_budget",
        "original_validation_discrepancy",
        "paired_cost_rows",
        "authority_support_rows",
        "development_records",
        "row_accounting",
        "independent_reduction",
        "archived_evidence_signatures",
        "reconciliation_errors",
        "original_disposition",
        "unchanged_method_disposition",
        "acceptance_gate_results",
        "acquisition_adjudication_complete_score",
        "acquisition_readiness_score",
        "acquisition_value_score",
        "promotion_score",
        "verdict_class",
        "honest_verdict",
        "flagged_adversarial",
        "validation_receipts",
    )
    return _canonical_hash({key: artifact.get(key) for key in keys})


def validate_artifact(value: object, *, require_current_validation: bool = True) -> list[str]:
    """Cold-check identity, inference, source disposition, and row signatures."""

    if not isinstance(value, Mapping):
        return ["artifact_object"]
    artifact = value
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_contract")
    counts = artifact.get("invocation_counts", {})
    if not isinstance(counts, Mapping) or counts.get("current") != ZERO_CURRENT_INVOCATIONS:
        errors.append("invocation_counts")
    if artifact.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_class")
    if artifact.get("execution_venue") != "host":
        errors.append("venue")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion")
    if artifact.get("verdict_class") != "blocked":
        disposition = artifact.get("original_disposition", {})
        if (
            disposition.get("verdict_class") != "disqualified"
            or disposition.get("preserved") is not True
        ):
            errors.append("original_disposition")
        signatures = evidence_signatures(artifact.get("rows", []))
        if signatures != artifact.get("archived_evidence_signatures"):
            errors.append("reconciliation")
        if (
            artifact.get("acquisition_readiness_score") != 0
            or artifact.get("acquisition_value_score") != 0
        ):
            errors.append("historical_readiness")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    if require_current_validation and artifact.get("verdict_class") != "blocked":
        if artifact.get("required_checks_passed") is not True:
            errors.append("required_validation")
        if not _has_receipts(artifact, (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)):
            errors.append("validation_receipts")
        if artifact.get("acquisition_adjudication_complete_score") != 1:
            errors.append("accounting_completion")
    return sorted(set(errors))


def scoped_command_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build and validate the explicit Exp7358 boundary for affected files."""

    commands = validation_contract.build_command_plan(root, AFFECTED_MANIFEST, private_root)
    errors = validation_contract.validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if errors:
        raise AdjudicationError("invalid_scoped_command_plan:" + ",".join(errors))
    return commands


def validate_scoped_command_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Expose Exp7358 plan validation for tests and other V646 consumers."""

    return validation_contract.validate_command_plan(root, AFFECTED_MANIFEST, commands)


def run_affected_validation(root: Path, raw_dir: Path) -> JsonDict:
    """Execute the Exp7358 command plan through the shipped streaming runner."""

    private_root = Path(tempfile.mkdtemp(prefix="exp7364-validation-", dir="/tmp"))
    commands = scoped_command_plan(root, private_root)
    planned = [
        validation_contract.PlannedCommand(command, "required_validation", True)
        for command in commands
    ]
    receipts = validation_contract.run_categorized_commands(
        root, planned, log_dir=raw_dir / "validation/affected"
    )
    reduced = validation_contract.reduce_affected_receipts(root, AFFECTED_MANIFEST, receipts)
    return {
        "validation_receipts": receipts,
        "required_checks_passed": reduced["passed"],
        "missing_required_commands": reduced["missing_required_commands"],
        "failed_required_commands": reduced["failed_required_commands"],
        "duplicate_required_commands": reduced["duplicate_required_commands"],
    }


def run_terminal_validation(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Run a cold reducer and both strict terminal artifact readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,sys;from pathlib import Path;"
        "from carnot.experiment_7364_v646_acquisition_adjudication import validate_artifact;"
        "value=json.loads(Path(sys.argv[1]).read_text());"
        "errors=validate_artifact(value,require_current_validation=False);"
        "print(errors,flush=True);raise SystemExit(bool(errors))"
    )
    commands = [
        validation_scope.CommandSpec(
            "independent_reducer", (python, "-u", "-c", reducer, str(candidate)), "raw_evidence"
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_candidate",
        ),
    ]
    return validation_scope.run_commands(root, commands, log_dir=raw_dir / "validation/terminal")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed-date entrypoint and optional cold-validation mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build, validate, classify, and atomically publish the adjudication."""

    args = _parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"Exp7364 requires --date {RUN_DATE}")
    if args.validate is not None:
        errors = validate_artifact(_load_object(args.validate), require_current_validation=True)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))

    run_started = time.monotonic()
    root = REPO_ROOT
    output_root = args.output_root.resolve() if args.output_root else root
    result_path = output_root / RESULT_PATH
    raw_dir = output_root / RAW_DIR
    progress(run_started, "preconditions", "start")
    checks, hashes = collect_preconditions(root)
    failed = next((row for row in checks if row["passed"] is not True), None)
    if failed is not None:
        artifact = build_blocked_artifact(checks, hashes, time.monotonic() - run_started)
        validation_contract.atomic_json(result_path, artifact)
        progress(run_started, "terminal_write", "end", path=result_path, verdict="blocked")
        return 0
    progress(run_started, "preconditions", "end", checks=len(checks))

    progress(run_started, "archive_reconciliation", "start")
    artifact = build_artifact(root, raw_dir)
    progress(
        run_started,
        "archive_reconciliation",
        "end",
        rows=len(artifact["rows"]),
        errors=len(artifact["reconciliation_errors"]),
    )

    validation_started = time.monotonic()
    progress(run_started, "affected_validation", "before_subprocess")
    affected = run_affected_validation(root, raw_dir)
    artifact.update(affected)
    apply_terminal_state(artifact, require_terminal=False)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    candidate = raw_dir / "terminal_candidate.json"
    validation_contract.atomic_json(candidate, artifact)
    progress(
        run_started,
        "affected_validation",
        "after_subprocess",
        passed=artifact["required_checks_passed"],
    )

    progress(run_started, "terminal_validation", "before_subprocess")
    terminal = run_terminal_validation(root, candidate, raw_dir)
    artifact["validation_receipts"].extend(terminal)
    if any(row["name"] == "adversarial_verify" and not row["passed"] for row in terminal):
        artifact["flagged_adversarial"] = True
    progress(run_started, "terminal_validation", "after_subprocess", units=len(terminal))
    artifact["phase_spans"].append(
        _phase_span(
            "validation", validation_started, run_started, len(artifact["validation_receipts"])
        )
    )

    apply_terminal_state(artifact, require_terminal=True)
    artifact["duration_s"] = time.monotonic() - run_started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(
        artifact, require_current_validation=artifact["verdict_class"] != "disqualified"
    )
    if errors:
        raise AdjudicationError("terminal_artifact_invalid:" + ",".join(errors))
    write_started = time.monotonic()
    artifact["phase_spans"].append(_phase_span("write", write_started, run_started, 1))
    artifact["duration_s"] = time.monotonic() - run_started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    progress(run_started, "terminal_write", "start", path=result_path)
    validation_contract.atomic_json(result_path, artifact)
    progress(run_started, "terminal_write", "end", path=result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - use the thin script entrypoint.
    raise SystemExit(main())
