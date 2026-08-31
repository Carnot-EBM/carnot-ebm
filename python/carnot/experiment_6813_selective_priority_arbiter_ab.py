"""Replay the Exp6812 proposals through two exact arbitration policies.

The module does not generate proposals. It uses immutable candidate bytes and
exact receipts from Exp6812. Selection can read only declared proposal-time
fields. A separate Exp6811 evaluator measures the selected transition.

Spec refs: REQ-CONSTRAINT-6813 and SCENARIO-CONSTRAINT-6813-*.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

from carnot import experiment_6812_sota_operational_handoff_corpus_v2 as source_exp
from carnot.durable_row_checkpoint import atomic_write_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_6813_selective_priority_arbiter_ab.json"
SOURCE_NAME = "experiment_6812_sota_operational_handoff_corpus_v2.json"
COMPILER_NAME = "experiment_6811_operational_obligation_automaton_v3.json"
SCHEMA = "carnot.experiment_6813.selective_priority_arbiter_ab.v1"
BLOCKED_STATUS = "complete_blocked_selective_priority_arbiter_ab"
SELECTIVE_ARM = "selective_priority"
FLAT_ARM = "flat_reject_retry"
ARMS = (SELECTIVE_ARM, FLAT_ARM)
MODEL_SPECS = source_exp.MODEL_SPECS
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)
RANDOM_SEED = {"tie_seed": 681301, "interval_seed": 681302}
FEATURE_ALLOWLIST = (
    "parsed_proposal_bytes",
    "frozen_candidate_index",
    "declared_obligation_contract",
    "observed_facts",
    "parse_state",
    "hard_violation_count",
    "binding_violation_vector",
    "legal_support",
    "authority_preserved",
    "soft_score",
)
FEATURE_DENYLIST = (
    "model_id",
    "model_family",
    "scenario_id",
    "exact_answer",
    "future_outcome",
    "post_selection_progress",
    "harmful_selection",
    "exact_utility",
)
ARM_DEFINITIONS: JsonDict = {
    SELECTIVE_ARM: (
        "Preserve a valid base byte string. Otherwise choose the exact "
        "hard-binding-soft lexicographic minimum with a stable index tie."
    ),
    FLAT_ARM: (
        "Inspect the same candidates in frozen order. Reject any failed "
        "constraint and accept the first fully valid candidate."
    ),
}
FIELD_DESCRIPTIONS: JsonDict = {
    "schema": "Versioned artifact contract.",
    "experiment_id": "Stable experiment identity.",
    "title": "Human-readable experiment name.",
    "run_date": "Caller-supplied execution date.",
    "status": "Terminal experiment state.",
    "field_principles": "One plain-language principle for every top-level field.",
    "inference_substrate": "Deterministic CPU replay of frozen bytes with no model call.",
    "duration_s": "Measured wall duration in seconds.",
    "random_seed": "Frozen seeds for stable ties and paired intervals.",
    "reproducibility_checksum": "Hash that binds source, manifest, and deterministic rows.",
    "source_artifact_sha256": "SHA-256 of the authentic Exp6812 proposal artifact.",
    "compiler_artifact_sha256": "SHA-256 of the Exp6811 compiler artifact.",
    "frozen_manifest": "Development and held splits, budgets, and public constants.",
    "arm_definitions": "The selective and flat policies compared on each candidate set.",
    "feature_allowlist": "Proposal-time fields that selection may read.",
    "feature_denylist": "Identity, outcome, and utility labels denied to selection.",
    "budget_match_receipt": "Evidence that both arms receive equal permissions and work.",
    "rows": "Every model-scenario-seed-handoff-arm replay unit.",
    "hard_violation_rate_by_arm": "Held exact accepted hard-violation rate.",
    "accepted_progress_by_arm": "Held post-selection exact progress rate.",
    "retry_cost_by_arm": "Held realized retry cost under equal retry caps.",
    "false_intervention_rate_by_arm": "Held changes to proposals already known to be valid.",
    "safe_action_identity_by_arm": "Held byte identity for already-valid base proposals.",
    "legal_support_by_arm": "Held legal selections and model-family support.",
    "certificate_completeness_by_arm": "Held first-conflict or no-op certificate coverage.",
    "harmful_selections_by_arm": "Held selections that fail the external exact transition check.",
    "paired_progress_delta": "Held paired selective-minus-flat progress interval.",
    "paired_retry_delta": "Held paired flat-minus-selective retry interval.",
    "acceptance_gate_positive": "Predeclared conjunction for a positive effect claim.",
    "selective_arbiter_ab_completed": "Downstream completion independent of effect sign.",
    "gate_check_summary": "Named precondition checks with expected and observed values.",
    "manifest_receipt": "Checks that the split and constants were frozen before reduction.",
    "reducer_receipt": "Checks that held reduction is deterministic and row-derived.",
    "attack_results": "Deterministic attacks on priority, features, certificates, and budgets.",
    "verifier_is_oracle": "False because the outcome evaluator runs after both selectors.",
    "verdict_class": "Closed terminal class supported by rows and gates.",
    "honest_verdict": "Terminal statement that reports positive, null, or blocked evidence.",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one value so equality and hashes use the same bytes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def b64_bytes(value: bytes) -> str:
    """Encode bytes for a JSON field without changing their identity."""

    return base64.b64encode(value).decode("ascii")


def sha256_bytes(value: bytes) -> str:
    """Return the repository's prefixed SHA-256 representation."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def _valid_hash(value: Any) -> bool:
    """Accept only a complete prefixed lower-case SHA-256 value."""

    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _candidate_action(candidate: Mapping[str, Any]) -> JsonDict | None:
    """Read only the parsed action inside one candidate receipt."""

    proposal = candidate.get("candidate")
    action = proposal.get("action") if isinstance(proposal, Mapping) else None
    return deepcopy(dict(action)) if isinstance(action, Mapping) else None


def candidate_conflict(candidate: Mapping[str, Any]) -> JsonDict | None:
    """Name the first failed item in the declared selection order."""

    candidate_id = str(candidate.get("candidate_id", candidate.get("candidate_index", "unknown")))
    if candidate.get("parse_state") != "complete" or _candidate_action(candidate) is None:
        return {
            "candidate_id": candidate_id,
            "priority": "syntax",
            "first_conflict": "response_schema",
        }
    hard_count = int(candidate.get("hard_violation_count", 0))
    if hard_count:
        first = candidate.get("first_conflict")
        if not first:
            certificates = candidate.get("conflict_certificates", [])
            first = next(
                (
                    item.get("first_conflict") or item.get("reason")
                    for item in certificates
                    if isinstance(item, Mapping)
                ),
                "hard_constraint",
            )
        return {"candidate_id": candidate_id, "priority": "hard", "first_conflict": first}
    if candidate.get("authority_preserved", True) is not True:
        return {
            "candidate_id": candidate_id,
            "priority": "binding",
            "first_conflict": "authority_spoof",
        }
    vector = [int(value) for value in candidate.get("binding_violation_vector", [])]
    if any(vector):
        index = next(position for position, value in enumerate(vector) if value)
        identifiers = candidate.get("binding_obligation_ids", [])
        first = identifiers[index] if index < len(identifiers) else f"binding-{index}"
        return {"candidate_id": candidate_id, "priority": "binding", "first_conflict": first}
    if candidate.get("legal_support") is not True:
        certificates = candidate.get("conflict_certificates", [])
        first = next(
            (
                item.get("first_conflict") or item.get("reason")
                for item in certificates
                if isinstance(item, Mapping)
            ),
            "exact_support",
        )
        return {"candidate_id": candidate_id, "priority": "support", "first_conflict": first}
    return None


def _base_is_valid(candidate: Mapping[str, Any]) -> bool:
    """Apply the frozen validity test used by false-intervention accounting."""

    return candidate_conflict(candidate) is None


def _selective_eligible(candidate: Mapping[str, Any]) -> bool:
    """Keep only syntax-valid, hard-safe, supported, authorized proposals."""

    return (
        candidate.get("parse_state") == "complete"
        and _candidate_action(candidate) is not None
        and int(candidate.get("hard_violation_count", 0)) == 0
        and candidate.get("authority_preserved", True) is True
        and candidate.get("legal_support") is True
    )


def _selection_evidence(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retain the exact proposal-time evidence used by either selector."""

    return [
        {
            "candidate_id": item.get("candidate_id"),
            "candidate_index": item.get("candidate_index"),
            "parse_state": item.get("parse_state"),
            "hard_violation_count": item.get("hard_violation_count"),
            "binding_violation_vector": deepcopy(item.get("binding_violation_vector", [])),
            "soft_score": item.get("soft_score"),
            "legal_support": item.get("legal_support"),
            "authority_preserved": item.get("authority_preserved", True),
            "raw_output_sha256": item.get("raw_output_sha256"),
            "row_sha256": item.get("row_sha256"),
        }
        for item in sorted(candidates, key=lambda value: int(value["candidate_index"]))
    ]


def _finish_decision(
    *,
    selected_action: Mapping[str, Any],
    selected_candidate_id: str | None,
    certificate: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    retry_count: int,
    abstention: bool,
    base_action: Mapping[str, Any] | None,
    base_valid: bool,
) -> JsonDict:
    """Add byte identity and accounting fields after policy selection."""

    selected_bytes = canonical_json_bytes(selected_action)
    base_bytes = canonical_json_bytes(base_action) if base_action is not None else None
    safe_identity = selected_bytes == base_bytes if base_valid and base_bytes is not None else None
    false_intervention = bool(base_valid and safe_identity is False)
    return {
        "selected_action": deepcopy(dict(selected_action)),
        "selected_action_bytes_b64": b64_bytes(selected_bytes),
        "selected_action_sha256": sha256_bytes(selected_bytes),
        "selected_candidate_id": selected_candidate_id,
        "certificate": deepcopy(dict(certificate)),
        "certificate_complete": bool(certificate.get("kind")),
        "candidate_evidence": _selection_evidence(candidates),
        "retry_count": retry_count,
        "abstention": abstention,
        "base_already_valid": base_valid,
        "false_intervention": false_intervention,
        "safe_action_identity": safe_identity,
    }


def select_arm(
    candidates: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    fallback_action: Mapping[str, Any],
    soft_score_clip: int,
    base_action: Mapping[str, Any] | None = None,
    base_valid: bool | None = None,
) -> JsonDict:
    """Select from one frozen set without reading identity or outcome labels."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    ordered = sorted(candidates, key=lambda value: int(value["candidate_index"]))
    inferred_base = ordered[0] if ordered else None
    if base_action is None and inferred_base is not None:
        base_action = _candidate_action(inferred_base)
    if base_valid is None:
        base_valid = bool(inferred_base is not None and _base_is_valid(inferred_base))

    if arm == SELECTIVE_ARM and base_valid and base_action is not None:
        return _finish_decision(
            selected_action=base_action,
            selected_candidate_id="base_proposal",
            certificate={
                "kind": "no_op_preserved",
                "first_higher_priority_conflict": None,
                "rejections": [],
            },
            candidates=ordered,
            retry_count=0,
            abstention=False,
            base_action=base_action,
            base_valid=True,
        )

    rejections = [conflict for item in ordered if (conflict := candidate_conflict(item))]
    selected: Mapping[str, Any] | None = None
    if arm == SELECTIVE_ARM:
        eligible = [item for item in ordered if _selective_eligible(item)]
        if eligible:
            selected = min(
                eligible,
                key=lambda item: (
                    int(item.get("hard_violation_count", 0)),
                    tuple(int(value) for value in item.get("binding_violation_vector", [])),
                    -max(
                        -soft_score_clip,
                        min(soft_score_clip, int(item.get("soft_score") or 0)),
                    ),
                    int(item["candidate_index"]),
                ),
            )
    else:
        selected = next((item for item in ordered if _base_is_valid(item)), None)

    if selected is None:
        return _finish_decision(
            selected_action=fallback_action,
            selected_candidate_id=None,
            certificate={
                "kind": "no_legal_candidate",
                "first_higher_priority_conflict": (
                    rejections[0]["first_conflict"] if rejections else "empty_candidate_set"
                ),
                "rejections": rejections,
            },
            candidates=ordered,
            retry_count=min(max(0, len(ordered) - 1), max(0, len(ordered) - 1)),
            abstention=True,
            base_action=base_action,
            base_valid=bool(base_valid),
        )

    selected_index = int(selected["candidate_index"])
    selected_action = _candidate_action(selected)
    assert selected_action is not None
    first_conflict = rejections[0]["first_conflict"] if rejections else None
    return _finish_decision(
        selected_action=selected_action,
        selected_candidate_id=str(selected.get("candidate_id")),
        certificate={
            "kind": "selected",
            "first_higher_priority_conflict": first_conflict,
            "rejections": rejections,
        },
        candidates=ordered,
        retry_count=selected_index if arm == FLAT_ARM else 0,
        abstention=False,
        base_action=base_action,
        base_valid=bool(base_valid),
    )


def _wilson_upper(numerator: int, denominator: int, alpha: float) -> float | None:
    """Return a deterministic Wilson upper bound for one binomial rate."""

    if denominator == 0:
        return None
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    rate = numerator / denominator
    scale = 1.0 + z * z / denominator
    center = rate + z * z / (2.0 * denominator)
    radius = z * math.sqrt(rate * (1.0 - rate) / denominator + z * z / (4 * denominator**2))
    return min(1.0, (center + radius) / scale)


def false_intervention_metric(rows: Sequence[Mapping[str, Any]], *, alpha: float) -> JsonDict:
    """Count changes only when the frozen validity test marked the base safe."""

    eligible = [row for row in rows if row.get("base_already_valid") is True]
    numerator = sum(row.get("false_intervention") is True for row in eligible)
    denominator = len(eligible)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
        "upper_bound": _wilson_upper(numerator, denominator, alpha),
        "alpha": alpha,
    }


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Reduce one Boolean row field with an explicit denominator."""

    values = [row.get(field) for row in rows if row.get(field) is not None]
    numerator = sum(value is True for value in values)
    return {
        "numerator": numerator,
        "denominator": len(values),
        "rate": numerator / len(values) if values else None,
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Reduce one numeric outcome without hiding its unit count."""

    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return {"mean": statistics.fmean(values) if values else None, "unit_count": len(values)}


def derive_budget_match_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Verify equal permissions and fixed observed work for every A/B pair."""

    fields = (
        "candidate_count",
        "exact_check_count",
        "outcome_check_count",
        "work_units",
        "retry_cap",
        "cpu_allowance_us",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    mismatches = [
        pair_id
        for pair_id, pair_rows in sorted(grouped.items())
        if len(pair_rows) != 2
        or {str(row.get("arm")) for row in pair_rows} != set(ARMS)
        or any(len({row.get(field) for row in pair_rows}) != 1 for field in fields)
    ]
    equal_observed_work = all(
        len({row.get("work_units") for row in pair_rows}) == 1 for pair_rows in grouped.values()
    )
    return {
        "passed": not mismatches and bool(grouped),
        "pair_count": len(grouped),
        "matched_fields": list(fields),
        "equal_observed_work": equal_observed_work,
        "mismatched_pair_ids": mismatches,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    """Use a fixed nearest-rank rule for reproducible bootstrap bounds."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.floor(probability * len(ordered)))))
    return ordered[index]


def _paired_interval(values: Sequence[float], *, seed: int, resamples: int) -> JsonDict:
    """Bootstrap paired means with one frozen pseudo-random stream."""

    if not values:
        return {"estimate": None, "lower_bound": None, "upper_bound": None, "pair_count": 0}
    generator = random.Random(seed)
    size = len(values)
    samples = [
        statistics.fmean(values[generator.randrange(size)] for _ in range(size))
        for _ in range(resamples)
    ]
    return {
        "estimate": statistics.fmean(values),
        "lower_bound": _quantile(samples, 0.025),
        "upper_bound": _quantile(samples, 0.975),
        "pair_count": size,
        "resamples": resamples,
        "seed": seed,
    }


def derive_paired_deltas(
    rows: Sequence[Mapping[str, Any]], *, seed: int, resamples: int
) -> tuple[JsonDict, JsonDict]:
    """Derive held paired progress and retry changes from arm rows."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("split") == "held":
            grouped[str(row["pair_id"])][str(row["arm"])] = row
    complete = [pair for pair in grouped.values() if set(pair) == set(ARMS)]
    progress_values = [
        float(pair[SELECTIVE_ARM]["accepted_progress"]) - float(pair[FLAT_ARM]["accepted_progress"])
        for pair in complete
    ]
    retry_values = [
        float(pair[FLAT_ARM]["retry_count"]) - float(pair[SELECTIVE_ARM]["retry_count"])
        for pair in complete
    ]
    progress = _paired_interval(progress_values, seed=seed, resamples=resamples)
    retry = _paired_interval(retry_values, seed=seed + 1, resamples=resamples)
    progress["direction"] = "selective_minus_flat"
    retry["direction"] = "flat_minus_selective"
    return progress, retry


def derive_acceptance_gate(
    *,
    selective_hard_rate: float | None,
    selective_false_upper: float | None,
    false_limit: float,
    progress_lower: float | None,
    retry_lower: float | None,
    no_family_support_loss: bool,
    selective_harmful: int,
    flat_harmful: int,
) -> JsonDict:
    """Apply the predeclared positive conjunction without changing readiness."""

    conditions = {
        "zero_accepted_hard_violations": selective_hard_rate == 0.0,
        "false_intervention_within_limit": (
            selective_false_upper is not None and selective_false_upper <= false_limit
        ),
        "positive_paired_lower_bound": bool(
            (progress_lower is not None and progress_lower > 0.0)
            or (retry_lower is not None and retry_lower > 0.0)
        ),
        "no_family_support_loss": no_family_support_loss,
        "no_harmful_selection_increase": selective_harmful <= flat_harmful,
    }
    return {"passed": all(conditions.values()), "conditions": conditions}


def _read_json(path: Path) -> JsonDict | None:
    """Return one JSON object, or None when a precondition cannot be read."""

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _precondition_summary(root: Path) -> tuple[JsonDict, JsonDict | None, JsonDict | None]:
    """Check every owned input before any comparison row is built."""

    source_path = root / "results" / SOURCE_NAME
    compiler_path = root / "results" / COMPILER_NAME
    source = _read_json(source_path)
    compiler = _read_json(compiler_path)
    rows = source.get("rows", []) if source else []
    raw_manifest = source.get("raw_output_manifest", []) if source else []
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, Mapping):
            grouped[
                (
                    row.get("model_id"),
                    row.get("scenario_id"),
                    row.get("random_seed"),
                    row.get("arm"),
                )
            ].append(row)
    models = sorted(set(source.get("models_used", []))) if source else []
    raw_hashes_complete = bool(rows) and all(
        isinstance(row, Mapping)
        and _valid_hash(row.get("raw_output_sha256"))
        and _valid_hash(row.get("row_sha256"))
        for row in rows
    )
    raw_hashes_complete = (
        raw_hashes_complete
        and bool(raw_manifest)
        and all(
            isinstance(item, Mapping)
            and _valid_hash(item.get("raw_output_sha256"))
            and _valid_hash(item.get("raw_api_response_sha256"))
            for item in raw_manifest
        )
    )
    pairs_complete = (
        bool(grouped)
        and len(grouped) == 288
        and all(
            len(pair) == 2
            and sorted(int(item.get("candidate_index", -1)) for item in pair) == [0, 1]
            for pair in grouped.values()
        )
    )
    compiler_manifest = compiler.get("compiler_manifest", {}) if compiler else {}
    checks = [
        {
            "check": "operational_handoff_corpus_ready",
            "expected": True,
            "observed": source.get("operational_handoff_corpus_ready") if source else None,
        },
        {
            "check": "all_three_model_families",
            "expected": sorted(MODEL_SPECS),
            "observed": models,
        },
        {"check": "raw_hashes_complete", "expected": True, "observed": raw_hashes_complete},
        {"check": "complete_paired_rows", "expected": True, "observed": pairs_complete},
        {
            "check": "exp6811_compiler",
            "expected": True,
            "observed": bool(
                compiler
                and compiler.get("operational_automaton_fixture_ready") is True
                and compiler_manifest.get("owned_code")
                == "python/carnot/agentic/arc_trajectory_supervisor.py"
                and int(compiler_manifest.get("obligation_count", 0)) > 0
            ),
        },
    ]
    for check in checks:
        check["passed"] = check["observed"] == check["expected"]
    failed = [check["check"] for check in checks if not check["passed"]]
    return (
        {"passed": not failed, "failed_checks": failed, "checks": checks},
        source,
        compiler,
    )


def _scenario_split(scenario_id: str) -> str:
    """Keep variants one through four for fitting and the rest for held claims."""

    variant = int(scenario_id.rsplit("-", 1)[1])
    return "development" if variant <= 4 else "held"


def _public_constants(source: Mapping[str, Any]) -> JsonDict:
    """Fit the only data-dependent constant from development rows."""

    development_ids = [
        scenario["scenario_id"]
        for scenario in source["frozen_manifest"]["scenarios"]
        if _scenario_split(str(scenario["scenario_id"])) == "development"
    ]
    development = [row for row in source["rows"] if row["scenario_id"] in development_ids]
    soft_values = [
        abs(int(row["soft_score"])) for row in development if row.get("soft_score") is not None
    ]
    return {
        "fit_split": "development",
        "fit_source_row_count": len(development),
        "soft_score_clip": max([1, *soft_values]),
        "false_intervention_upper_limit": 0.20,
        "false_intervention_alpha": 0.05,
        "paired_interval_resamples": 2000,
        "candidate_count": 2,
        "exact_check_allowance": 2,
        "outcome_check_allowance": 1,
        "retry_cap": 1,
        "cpu_allowance_us": 250_000,
    }


def _manifest(source: Mapping[str, Any], source_sha: str, compiler_sha: str) -> JsonDict:
    """Freeze split identities and constants before held reduction."""

    scenarios = [str(item["scenario_id"]) for item in source["frozen_manifest"]["scenarios"]]
    development = [item for item in scenarios if _scenario_split(item) == "development"]
    held = [item for item in scenarios if _scenario_split(item) == "held"]
    manifest: JsonDict = {
        "schema": "carnot.experiment_6813.frozen_manifest.v1",
        "source_artifact_sha256": source_sha,
        "compiler_artifact_sha256": compiler_sha,
        "development_scenario_ids": development,
        "held_scenario_ids": held,
        "public_constants": _public_constants(source),
        "arms": list(ARMS),
        "model_specs": list(MODEL_SPECS),
        "source_handoff_arms": list(source_exp.ARMS),
        "random_seed": deepcopy(RANDOM_SEED),
        "frozen_before_reduction": True,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def _binding_ids(candidate: Mapping[str, Any]) -> list[str]:
    """Recover authority-ordered binding identifiers from retained contracts."""

    binding = [
        contract
        for contract in candidate.get("contracts", [])
        if contract.get("priority", {}).get("class") == "binding"
    ]
    binding.sort(key=lambda contract: int(contract["authority"]["order"]))
    return [f"binding-{index}:{item['authority']['issuer']}" for index, item in enumerate(binding)]


def _normalize_candidates(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose only selection evidence and stable byte receipts from source rows."""

    normalized: list[JsonDict] = []
    for row in rows:
        item = deepcopy(dict(row))
        item["candidate_id"] = (
            item["candidate"].get("candidate_id")
            if isinstance(item.get("candidate"), Mapping)
            else f"candidate_{item['candidate_index']}"
        )
        item["binding_obligation_ids"] = _binding_ids(item)
        normalized.append(item)
    return sorted(normalized, key=lambda item: int(item["candidate_index"]))


def _external_outcome(
    scenario: Mapping[str, Any],
    decision: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    cache: dict[str, JsonDict],
) -> JsonDict:
    """Evaluate one selected transition after both policy decisions are sealed."""

    selected_id = decision.get("selected_candidate_id")
    selected_source = next(
        (item for item in candidates if item.get("candidate_id") == selected_id), None
    )
    if selected_source and isinstance(selected_source.get("candidate"), Mapping):
        proposal = deepcopy(dict(selected_source["candidate"]))
    else:
        issuers = sorted(
            {
                obligation["contract"]["authority"]["issuer"]
                for obligation in scenario["obligations"]
            }
        )
        proposal = {
            "action": deepcopy(decision["selected_action"]),
            "authority_chain": issuers,
            "candidate_id": "candidate_0",
            "soft_progress": 0,
        }
    cache_key = sha256_bytes(
        canonical_json_bytes({"scenario_id": scenario["scenario_id"], "proposal": proposal})
    )
    if cache_key in cache:
        return deepcopy(cache[cache_key])
    exact = source_exp.evaluate_candidate(scenario, proposal)
    abstention = decision["abstention"] is True
    legal = bool(
        abstention or (exact["hard_violation_count"] == 0 and exact["operational_preserved"])
    )
    outcome = {
        "legality": legal,
        "accepted_progress": 0 if abstention else int(exact["operational_preserved"]),
        "accepted_hard_violation": bool(not abstention and exact["hard_violation_count"] > 0),
        "post_selection_hard_violation_count": exact["hard_violation_count"],
        "harmful_selection": bool(not abstention and not legal),
        "outcome_certificate": deepcopy(exact["certificate"]),
        "outcome_evaluator": "Exp6811 exact transition replay after selection",
    }
    cache[cache_key] = deepcopy(outcome)
    return outcome


def _build_rows(source: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Replay every complete source candidate set through both matched arms."""

    scenarios = {str(item["scenario_id"]): item for item in source["frozen_manifest"]["scenarios"]}
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in source["rows"]:
        grouped[(row["model_id"], row["scenario_id"], row["random_seed"], row["arm"])].append(row)
    constants = manifest["public_constants"]
    result: list[JsonDict] = []
    outcome_cache: dict[str, JsonDict] = {}
    for key, source_rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        model_id, scenario_id, seed, handoff_arm = key
        scenario = scenarios[str(scenario_id)]
        candidates = _normalize_candidates(source_rows)
        scenario_base = scenario.get("safe_proposal")
        if scenario_base is not None:
            base_action = deepcopy(scenario_base)
            base_valid = True
        else:
            base_action = _candidate_action(candidates[0])
            base_valid = _base_is_valid(candidates[0])
        pair_id = f"{model_id}|{scenario_id}|seed-{seed}|{handoff_arm}"
        for arm in ARMS:
            started = time.perf_counter_ns()
            decision = select_arm(
                candidates,
                arm=arm,
                fallback_action=scenario["fallback_action"],
                soft_score_clip=int(constants["soft_score_clip"]),
                base_action=base_action,
                base_valid=base_valid,
            )
            outcome = _external_outcome(scenario, decision, candidates, outcome_cache)
            latency_us = (time.perf_counter_ns() - started) / 1000.0
            result.append(
                {
                    "row_id": f"{pair_id}|{arm}",
                    "pair_id": pair_id,
                    "split": _scenario_split(str(scenario_id)),
                    "model_id": model_id,
                    "scenario_id": scenario_id,
                    "random_seed": seed,
                    "handoff_arm": handoff_arm,
                    "arm": arm,
                    **decision,
                    **outcome,
                    "candidate_count": constants["candidate_count"],
                    "exact_check_count": constants["exact_check_allowance"],
                    "outcome_check_count": constants["outcome_check_allowance"],
                    "retry_cap": constants["retry_cap"],
                    "cpu_allowance_us": constants["cpu_allowance_us"],
                    "work_units": constants["candidate_count"]
                    + constants["outcome_check_allowance"],
                    "latency_us": latency_us,
                }
            )
    return result


def _held_arm_rows(rows: Sequence[Mapping[str, Any]], arm: str) -> list[Mapping[str, Any]]:
    """Return the frozen held units for one arm."""

    return [row for row in rows if row.get("split") == "held" and row.get("arm") == arm]


def _metrics(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> JsonDict:
    """Reduce held outcomes after all arm rows are complete."""

    alpha = float(manifest["public_constants"]["false_intervention_alpha"])
    hard: JsonDict = {}
    progress: JsonDict = {}
    retry: JsonDict = {}
    false: JsonDict = {}
    safe_identity: JsonDict = {}
    legal: JsonDict = {}
    certificates: JsonDict = {}
    harmful: JsonDict = {}
    for arm in ARMS:
        arm_rows = _held_arm_rows(rows, arm)
        hard[arm] = _rate(arm_rows, "accepted_hard_violation")
        progress[arm] = _mean(arm_rows, "accepted_progress")
        retry[arm] = _mean(arm_rows, "retry_count")
        false[arm] = false_intervention_metric(arm_rows, alpha=alpha)
        safe_identity[arm] = _rate(arm_rows, "safe_action_identity")
        legal[arm] = _rate(arm_rows, "legality")
        legal[arm]["by_model"] = {
            model_id: _rate([row for row in arm_rows if row["model_id"] == model_id], "legality")
            for model_id in MODEL_SPECS
        }
        certificates[arm] = _rate(arm_rows, "certificate_complete")
        harmful[arm] = _rate(arm_rows, "harmful_selection")
    return {
        "hard_violation_rate_by_arm": hard,
        "accepted_progress_by_arm": progress,
        "retry_cost_by_arm": retry,
        "false_intervention_rate_by_arm": false,
        "safe_action_identity_by_arm": safe_identity,
        "legal_support_by_arm": legal,
        "certificate_completeness_by_arm": certificates,
        "harmful_selections_by_arm": harmful,
    }


def _attacks(soft_score_clip: int) -> list[JsonDict]:
    """Run bounded attacks that must pass before completion becomes true."""

    def candidate(
        index: int,
        *,
        hard: int = 0,
        binding: list[int] | None = None,
        soft: int = 0,
        legal: bool = True,
    ) -> JsonDict:
        return {
            "candidate_index": index,
            "candidate_id": f"candidate_{index}",
            "candidate": {
                "action": {"data": index, "kind": "ACTION"},
                "authority_chain": ["authority"],
                "candidate_id": f"candidate_{index}",
                "soft_progress": soft,
            },
            "parse_state": "complete",
            "hard_violation_count": hard,
            "binding_violation_vector": binding or [],
            "binding_obligation_ids": [
                f"binding-{position}" for position in range(len(binding or []))
            ],
            "soft_score": soft,
            "legal_support": legal,
            "authority_preserved": True,
            "conflict_certificates": [],
        }

    fallback = {"data": None, "kind": "NOOP"}
    hard_candidates = [candidate(0, hard=1, soft=10**9, legal=False), candidate(1, soft=-1)]
    hard = select_arm(
        hard_candidates,
        arm=SELECTIVE_ARM,
        fallback_action=fallback,
        soft_score_clip=soft_score_clip,
    )
    tied = [candidate(1, binding=[1], soft=1), candidate(0, binding=[1], soft=1)]
    tie = select_arm(
        tied, arm=SELECTIVE_ARM, fallback_action=fallback, soft_score_clip=soft_score_clip
    )
    attacked = deepcopy(tied)
    attacked[0].update({"model_id": "preferred", "exact_utility": 10**30, "future_outcome": "win"})
    feature = select_arm(
        attacked, arm=SELECTIVE_ARM, fallback_action=fallback, soft_score_clip=soft_score_clip
    )
    no_legal = select_arm(
        [candidate(0, hard=1, legal=False), candidate(1, hard=1, legal=False)],
        arm=SELECTIVE_ARM,
        fallback_action=fallback,
        soft_score_clip=soft_score_clip,
    )
    safe = {"data": "safe", "kind": "SAFE"}
    no_op = select_arm(
        [candidate(0, soft=0), candidate(1, soft=99)],
        arm=SELECTIVE_ARM,
        fallback_action=fallback,
        soft_score_clip=soft_score_clip,
        base_action=safe,
        base_valid=True,
    )
    common = {
        "pair_id": "attack-pair",
        "split": "held",
        "candidate_count": 2,
        "exact_check_count": 2,
        "outcome_check_count": 1,
        "work_units": 3,
        "retry_cap": 1,
        "cpu_allowance_us": 250_000,
    }
    budget = derive_budget_match_receipt(
        [{**common, "arm": SELECTIVE_ARM}, {**common, "arm": FLAT_ARM}]
    )
    return [
        {
            "attack_id": "hard_priority_dominance",
            "passed": hard["selected_candidate_id"] == "candidate_1",
        },
        {"attack_id": "stable_index_tie", "passed": tie["selected_candidate_id"] == "candidate_0"},
        {
            "attack_id": "denied_feature_injection",
            "passed": feature["selected_candidate_id"] == "candidate_0",
        },
        {
            "attack_id": "no_legal_abstention",
            "passed": no_legal["abstention"] is True and no_legal["certificate_complete"] is True,
        },
        {
            "attack_id": "safe_base_no_op",
            "passed": no_op["safe_action_identity"] is True
            and no_op["false_intervention"] is False,
        },
        {"attack_id": "budget_parity", "passed": budget["passed"] is True},
    ]


def _field_principles(artifact: Mapping[str, Any]) -> JsonDict:
    """Require an explicit principle for every emitted top-level field."""

    return {
        key: FIELD_DESCRIPTIONS.get(key, "Deterministic row-derived experiment evidence.")
        for key in artifact
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic evidence while excluding measured wall clocks."""

    payload = {
        key: deepcopy(value)
        for key, value in artifact.items()
        if key not in {"duration_s", "field_principles", "reproducibility_checksum"}
    }
    if isinstance(payload.get("rows"), list):
        for row in payload["rows"]:
            row.pop("latency_us", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    gate: Mapping[str, Any],
    source_sha: str | None,
    compiler_sha: str | None,
) -> JsonDict:
    """Emit the complete blocked contract and no comparison rows."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": "6813",
        "title": "Selective priority arbiter A/B",
        "run_date": run_date,
        "status": BLOCKED_STATUS,
        "inference_substrate": "deterministic CPU replay; no LLM invoked",
        "duration_s": duration_s,
        "random_seed": deepcopy(RANDOM_SEED),
        "source_artifact_sha256": source_sha,
        "compiler_artifact_sha256": compiler_sha,
        "frozen_manifest": {},
        "arm_definitions": deepcopy(ARM_DEFINITIONS),
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "budget_match_receipt": {"passed": False, "mismatched_pair_ids": []},
        "rows": [],
        "hard_violation_rate_by_arm": {},
        "accepted_progress_by_arm": {},
        "retry_cost_by_arm": {},
        "false_intervention_rate_by_arm": {},
        "safe_action_identity_by_arm": {},
        "legal_support_by_arm": {},
        "certificate_completeness_by_arm": {},
        "harmful_selections_by_arm": {},
        "paired_progress_delta": {},
        "paired_retry_delta": {},
        "acceptance_gate_positive": {"passed": False, "conditions": {}},
        "selective_arbiter_ab_completed": False,
        "gate_check_summary": deepcopy(dict(gate)),
        "manifest_receipt": {"passed": False},
        "reducer_receipt": {"passed": False},
        "attack_results": [],
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_STATUS,
    }
    artifact["field_principles"] = _field_principles({**artifact, "field_principles": {}})
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(artifact)
    return artifact


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    repo_root: str | Path = REPO_ROOT,
) -> JsonDict:
    """Build one complete or blocked artifact from local immutable inputs."""

    root = Path(repo_root)
    gate, source, compiler = _precondition_summary(root)
    source_path = root / "results" / SOURCE_NAME
    compiler_path = root / "results" / COMPILER_NAME
    source_sha = sha256_bytes(source_path.read_bytes()) if source_path.is_file() else None
    compiler_sha = sha256_bytes(compiler_path.read_bytes()) if compiler_path.is_file() else None
    if not gate["passed"] or source is None or compiler is None:
        return _blocked_artifact(
            run_date=run_date,
            duration_s=duration_s,
            gate=gate,
            source_sha=source_sha,
            compiler_sha=compiler_sha,
        )

    assert source_sha is not None and compiler_sha is not None
    manifest = _manifest(source, source_sha, compiler_sha)
    rows = _build_rows(source, manifest)
    budget = derive_budget_match_receipt(rows)
    metrics = _metrics(rows, manifest)
    resamples = int(manifest["public_constants"]["paired_interval_resamples"])
    progress_delta, retry_delta = derive_paired_deltas(
        rows, seed=RANDOM_SEED["interval_seed"], resamples=resamples
    )
    selective_legal = metrics["legal_support_by_arm"][SELECTIVE_ARM]["by_model"]
    flat_legal = metrics["legal_support_by_arm"][FLAT_ARM]["by_model"]
    no_family_loss = all(
        selective_legal[model]["rate"] >= flat_legal[model]["rate"] for model in MODEL_SPECS
    )
    positive = derive_acceptance_gate(
        selective_hard_rate=metrics["hard_violation_rate_by_arm"][SELECTIVE_ARM]["rate"],
        selective_false_upper=metrics["false_intervention_rate_by_arm"][SELECTIVE_ARM][
            "upper_bound"
        ],
        false_limit=float(manifest["public_constants"]["false_intervention_upper_limit"]),
        progress_lower=progress_delta["lower_bound"],
        retry_lower=retry_delta["lower_bound"],
        no_family_support_loss=no_family_loss,
        selective_harmful=metrics["harmful_selections_by_arm"][SELECTIVE_ARM]["numerator"],
        flat_harmful=metrics["harmful_selections_by_arm"][FLAT_ARM]["numerator"],
    )
    attacks = _attacks(int(manifest["public_constants"]["soft_score_clip"]))
    manifest_receipt = {
        "passed": bool(
            len(manifest["development_scenario_ids"]) == 24
            and len(manifest["held_scenario_ids"]) == 24
            and set(manifest["development_scenario_ids"]).isdisjoint(manifest["held_scenario_ids"])
            and manifest["public_constants"]["fit_split"] == "development"
            and manifest["frozen_before_reduction"] is True
        ),
        "development_count": len(manifest["development_scenario_ids"]),
        "held_count": len(manifest["held_scenario_ids"]),
    }
    second_progress, second_retry = derive_paired_deltas(
        rows, seed=RANDOM_SEED["interval_seed"], resamples=resamples
    )
    reducer_receipt = {
        "passed": progress_delta == second_progress and retry_delta == second_retry,
        "headline_split": "held",
        "held_pair_count": progress_delta["pair_count"],
    }
    expected_rows = 3 * 48 * 2 * 2
    completion_conditions = {
        "complete_rows": len(rows) == expected_rows,
        "equal_budgets": budget["passed"] is True,
        "deterministic_reducer": reducer_receipt["passed"] is True,
        "attacks_passed": all(item["passed"] for item in attacks),
        "manifest_frozen": manifest_receipt["passed"] is True,
    }
    completed = all(completion_conditions.values())
    verdict_class = "positive" if completed and positive["passed"] else "null"
    verdict = (
        "complete: selective priority positive gate passed on held exact replay"
        if verdict_class == "positive"
        else "complete: selective priority comparison completed with a row-supported null"
    )
    artifact = {
        "schema": SCHEMA,
        "experiment_id": "6813",
        "title": "Selective priority arbiter A/B",
        "run_date": run_date,
        "status": "complete",
        "inference_substrate": "deterministic CPU replay; no LLM invoked",
        "duration_s": duration_s,
        "random_seed": deepcopy(RANDOM_SEED),
        "source_artifact_sha256": source_sha,
        "compiler_artifact_sha256": compiler_sha,
        "frozen_manifest": manifest,
        "arm_definitions": deepcopy(ARM_DEFINITIONS),
        "feature_allowlist": list(FEATURE_ALLOWLIST),
        "feature_denylist": list(FEATURE_DENYLIST),
        "budget_match_receipt": budget,
        "rows": rows,
        **metrics,
        "paired_progress_delta": progress_delta,
        "paired_retry_delta": retry_delta,
        "acceptance_gate_positive": positive,
        "selective_arbiter_ab_completed": completed,
        "gate_check_summary": {**gate, "completion_conditions": completion_conditions},
        "manifest_receipt": manifest_receipt,
        "reducer_receipt": reducer_receipt,
        "attack_results": attacks,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }
    artifact["field_principles"] = _field_principles({**artifact, "field_principles": {}})
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return all structural or row-derived errors without changing the artifact."""

    required = set(FIELD_DESCRIPTIONS)
    errors: list[str] = []
    errors.extend([f"missing field: {key}" for key in sorted(required - set(artifact))])
    errors.extend(
        ["field principles do not cover every top-level field"]
        if set(artifact.get("field_principles", {})) != set(artifact)
        else []
    )
    errors.extend(
        ["invalid verdict class"] if artifact.get("verdict_class") not in VERDICT_CLASSES else []
    )
    errors.extend(
        ["verifier cannot be an oracle"] if artifact.get("verifier_is_oracle") is not False else []
    )
    if artifact.get("verdict_class") == "blocked":
        errors.extend(["blocked artifact contains rows"] if artifact.get("rows") else [])
        errors.extend(
            ["blocked completion field is true"]
            if artifact.get("selective_arbiter_ab_completed") is not False
            else []
        )
    else:
        rows = artifact.get("rows", [])
        errors.extend(["complete artifact row count mismatch"] if len(rows) != 576 else [])
        budget = derive_budget_match_receipt(rows)
        errors.extend(
            ["budget receipt is not row-derived"]
            if budget != artifact.get("budget_match_receipt")
            else []
        )
        manifest = artifact.get("frozen_manifest", {})
        constants = manifest.get("public_constants", {})
        progress, retry = derive_paired_deltas(
            rows,
            seed=int(
                manifest.get("random_seed", {}).get("interval_seed", RANDOM_SEED["interval_seed"])
            ),
            resamples=int(constants.get("paired_interval_resamples", 1)),
        )
        errors.extend(
            ["paired progress delta is not row-derived"]
            if progress != artifact.get("paired_progress_delta")
            else []
        )
        errors.extend(
            ["paired retry delta is not row-derived"]
            if retry != artifact.get("paired_retry_delta")
            else []
        )
        errors.extend(
            ["accepted hard violation present"]
            if any(row.get("accepted_hard_violation") is True for row in rows)
            else []
        )
        errors.extend(
            ["completion evidence is incomplete"]
            if artifact.get("selective_arbiter_ab_completed") is not True
            else []
        )
        errors.extend(
            ["attack suite failed"]
            if not all(item.get("passed") is True for item in artifact.get("attack_results", []))
            else []
        )
    errors.extend(
        ["reproducibility checksum mismatch"]
        if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact)
        else []
    )
    return errors


def run(*, run_date: str, repo_root: str | Path = REPO_ROOT) -> JsonDict:  # pragma: no cover
    """Measure the replay, validate it, and atomically write its artifact."""

    started = time.perf_counter()
    artifact = build_artifact(run_date=run_date, duration_s=0.0, repo_root=repo_root)
    artifact["duration_s"] = time.perf_counter() - started
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    output = Path(repo_root) / "results" / RESULT_NAME
    atomic_write_json(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the deterministic experiment from the required date CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    artifact = run(run_date=args.date)
    print(json.dumps({"status": artifact["status"], "verdict": artifact["honest_verdict"]}))
    return 0 if artifact["selective_arbiter_ab_completed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
