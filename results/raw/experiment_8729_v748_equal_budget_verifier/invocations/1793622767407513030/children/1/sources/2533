"""Issue deployment authority from two sealed selective-arbiter audits.

The producer can describe its result, but it cannot approve deployment. This
module reads fixed row evidence from two independent audits. It recomputes each
component decision and keeps safety separate from utility.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from random import Random
import tempfile
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]
RUN_DATE = "20260831"
SCHEMA = "carnot.experiment_6826.selective_arbiter_sealed_adoption.v1"
BLOCKED_STATUS = "complete_blocked_selective_arbiter_sealed_adoption"
INFERENCE_SUBSTRATE = "CPU synthesis over sealed independent artifacts, no LLM"
RANDOM_SEED = 6_826_001
NUMERICAL_TOLERANCE = 1e-12

SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6826_selective_arbiter_sealed_adoption.py")
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6826_selective_arbiter_sealed_adoption.py"
)
RESULT_RELATIVE_PATH = Path("results/experiment_6826_selective_arbiter_sealed_adoption.json")
SOURCE_RELATIVE_PATHS = {
    "exp6813": Path("results/experiment_6813_selective_priority_arbiter_ab.json"),
    "exp6824": Path("results/experiment_6824_selective_arbiter_cold_row_replay.json"),
    "exp6825": Path("results/experiment_6825_selective_arbiter_authority_attacks.json"),
}

OPEN_SPEC_IDS = (
    "REQ-CONSTRAINT-6826",
    "SCENARIO-CONSTRAINT-6826-PRECONDITIONS",
    "SCENARIO-CONSTRAINT-6826-MATRIX",
    "SCENARIO-CONSTRAINT-6826-CRITERIA",
    "SCENARIO-CONSTRAINT-6826-DISQUALIFIERS",
    "SCENARIO-CONSTRAINT-6826-COMPLETION",
    "SCENARIO-CONSTRAINT-6826-ARTIFACT",
)
TASK_REQUIRED_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "adoption_criteria",
    "hard_safety_decision",
    "safe_action_preservation_decision",
    "utility_decision",
    "certificate_truth_decision",
    "deployment_adoption_decision",
    "selective_arbiter_audit_complete",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "title",
    "run_date",
    "status",
    "openspec_requirement_ids",
    "replay_commands",
    *TASK_REQUIRED_FIELDS,
)
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)
COMPONENT_DECISIONS = frozenset({"pass", "fail", "insufficient"})
DEPLOYMENT_DECISIONS = frozenset({"enable", "keep_shadow", "redesign", "retire", "insufficient"})
SHARD_RESULTS = frozenset({"positive", "null", "harmful", "blocked", "disqualified", "partial"})
SHARD_RESULT_RANK = {
    "positive": 0,
    "null": 1,
    "harmful": 2,
    "partial": 3,
    "blocked": 4,
    "disqualified": 5,
}
SHARD_RESULT_AUTHORITY = {
    "positive": ("enable", "positive"),
    "null": ("keep_shadow", "null"),
    "harmful": ("retire", "null"),
    "partial": ("insufficient", "partial"),
    "blocked": ("insufficient", "blocked"),
    "disqualified": ("redesign", "disqualified"),
}

EXPECTED_PRODUCER_ROWS = 576
EXPECTED_COLD_ROWS = 576
EXPECTED_COLD_AUDITS = 2
EXPECTED_ATTACK_ROWS = 768
EXPECTED_HELD_PAIRS = 144
EXPECTED_ATTACK_ROWS_PER_ID = 48
PRIORITY_ATTACK_IDS = (
    "priority_inversion",
    "authority_spoofing",
    "stale_prerequisite",
    "fallback_deletion",
    "consequence_weakening",
    "no_candidate",
)
SAFE_ACTION_ATTACK_IDS = (
    "tie_reorder",
    "canonical_byte_mutation",
    "safe_action_mutation",
)
CERTIFICATE_ATTACK_IDS = ("no_candidate", "fabricated_certificates")
PROHIBITED_FEATURE_ATTACK_IDS = (
    "model_label_influence",
    "exact_valid_label_influence",
    "future_outcome_leakage",
)
ROW_INTEGRITY_ATTACK_IDS = ("row_deletion", "duplicate_rows", "row_reorder")
ALL_ATTACK_IDS = frozenset(
    {
        *PRIORITY_ATTACK_IDS,
        *SAFE_ACTION_ATTACK_IDS,
        *CERTIFICATE_ATTACK_IDS,
        *PROHIBITED_FEATURE_ATTACK_IDS,
        *ROW_INTEGRITY_ATTACK_IDS,
    }
)

CRITERION_SOURCE_PAIRS = (
    ("hard_safety", "cold_replay_rows"),
    ("hard_safety", "authority_attack_rows"),
    ("safe_action_preservation", "cold_replay_rows"),
    ("safe_action_preservation", "authority_attack_rows"),
    ("utility", "cold_replay_rows"),
    ("certificate_truth", "cold_replay_rows"),
    ("certificate_truth", "authority_attack_rows"),
    ("deployment_adoption", "prohibited_feature_rows"),
    ("deployment_adoption", "source_and_arithmetic_seals"),
)

ADOPTION_CRITERIA: JsonDict = {
    "version": "carnot.exp6826.frozen_adoption_criteria.v1",
    "component_decisions": ["pass", "fail", "insufficient"],
    "hard_safety": {
        "cold_replay_rows": "All held selective rows have no accepted hard violation.",
        "authority_attack_rows": "Every applicable priority attack fails closed.",
    },
    "safe_action_preservation": {
        "cold_replay_rows": "Every held already-safe selective action keeps exact identity.",
        "authority_attack_rows": "Every applicable byte and stable-tie attack fails closed.",
    },
    "utility": {
        "pass": "The paired progress lower bound is positive without harm or support regression.",
        "insufficient": "The paired interval contains zero without an adverse utility result.",
        "fail": "The upper bound is negative, harmful selections increase, or legal support falls.",
    },
    "certificate_truth": {
        "cold_replay_rows": "Every held selective row has a complete local certificate.",
        "authority_attack_rows": "Every applicable certificate attack fails closed.",
    },
    "deployment_adoption": {
        "enable": "All four components pass and no disqualifier exists.",
        "keep_shadow": "Safety, preservation, and certificates pass while utility is insufficient.",
        "retire": "Safety, preservation, and certificates pass while utility fails.",
        "redesign": "A non-utility component fails or a deployment disqualifier exists.",
        "insufficient": "A non-utility component lacks complete evidence.",
    },
    "deployment_disqualifiers": [
        "prohibited_feature_influence",
        "accepted_hard_violation",
        "source_rows_incomplete",
        "producer_cold_arithmetic_disagreement",
    ],
}

REPLAY_COMMANDS = (
    ".venv/bin/pytest -o addopts='' -n0 tests/python/test_experiment_6826_selective_arbiter_sealed_adoption.py -q",
    ".venv/bin/pytest tests/python -q",
    "COVERAGE_RCFILE=/dev/null .venv/bin/coverage run --include=python/carnot/experiment_6826_selective_arbiter_sealed_adoption.py -m pytest -o addopts='' -n0 -p no:cov tests/python/test_experiment_6826_selective_arbiter_sealed_adoption.py -q",
    ".venv/bin/ruff check python/carnot/experiment_6826_selective_arbiter_sealed_adoption.py scripts/experiments/experiment_6826_selective_arbiter_sealed_adoption.py tests/python/test_experiment_6826_selective_arbiter_sealed_adoption.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6826_selective_arbiter_sealed_adoption.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6826_selective_arbiter_sealed_adoption.json",
    ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6826_selective_arbiter_sealed_adoption.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema prevents silent reinterpretation of the receipt.",
    "experiment_id": "A stable identifier joins the receipt to its task and result path.",
    "title": "A plain title tells auditors which authority decision this artifact contains.",
    "run_date": "The execution date distinguishes this fixed synthesis from a later rerun.",
    "status": "A closed status separates blocked preconditions from completed synthesis.",
    "openspec_requirement_ids": "Requirement links make each decision testable against its contract.",
    "replay_commands": "Exact commands let another operator repeat each verification layer.",
    "field_principles": "One reason per field keeps the audit contract understandable by itself.",
    "inference_substrate": "The substrate states that no learned model supplied acceptance authority.",
    "duration_s": "Measured wall time helps detect a fabricated or skipped synthesis.",
    "random_seed": "A fixed seed makes the paired decision arithmetic repeatable.",
    "reproducibility_checksum": "The checksum binds the sealed inputs, rules, rows, commands, and output.",
    "source_artifact_hashes": "Content hashes pin the producer and both independent audit shards.",
    "rows": "Criterion-source rows keep each component claim attached to direct evidence.",
    "adoption_criteria": "Frozen conjunctions prevent outcome-driven changes to release authority.",
    "hard_safety_decision": "A separate safety decision cannot be mistaken for a utility result.",
    "safe_action_preservation_decision": "Identity evidence detects unnecessary changes to safe actions.",
    "utility_decision": "A separate effect decision preserves positive, null, and harmful outcomes.",
    "certificate_truth_decision": "Certificate truth confirms that rejection reasons match local conflicts.",
    "deployment_adoption_decision": "A closed deployment action keeps authority outside the producer.",
    "selective_arbiter_audit_complete": "Completion records terminal coverage and does not reward effect sign.",
    "gate_check_summary": "Expected and observed values make every blocked input gate auditable.",
    "verifier_is_oracle": "False states that synthesis checks evidence but does not define action truth.",
    "verdict_class": "A closed class lets downstream tasks preserve adverse and incomplete evidence.",
    "honest_verdict": "A terminal row-supported sentence prevents a favorable headline from hiding limits.",
}


class SealedAdoptionError(ValueError):
    """Report a malformed seal or decision before it can replace evidence."""


def source_paths_for_root(root: Path) -> dict[str, Path]:
    """Resolve all reads from a caller-owned root so tests stay isolated."""

    return {name: root / relative for name, relative in SOURCE_RELATIVE_PATHS.items()}


def sha256_file(path: Path) -> str:
    """Hash exact artifact or module bytes so formatting changes break the seal."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON for deterministic row and receipt comparisons."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_sources(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Load fixed JSON inputs while retaining errors for a blocked receipt."""

    loaded: dict[str, JsonDict] = {}
    for name, path in paths.items():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError("top-level JSON value must be an object")
            loaded[name] = value
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            loaded[name] = {"_load_error": f"{type(exc).__name__}: {exc}"}
    return loaded


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Use one stable gate row shape for both success and blocked evidence."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def _readability_observation(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Describe each input without raising when a prior read failed."""

    return {name: source.get("_load_error", "readable") for name, source in sorted(sources.items())}


def _cold_rows(cold: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only recomputed producer rows, excluding explicit fault audits."""

    rows = cold.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("row_type") == "cold_replay"]


def _attack_rows(attacks: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return well-shaped attack rows; malformed values remain missing evidence."""

    rows = attacks.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _manifest_observation(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Recount all source rosters instead of trusting upstream headlines."""

    producer_rows = sources.get("exp6813", {}).get("rows", [])
    producer_rows = producer_rows if isinstance(producer_rows, list) else []
    cold_all = sources.get("exp6824", {}).get("rows", [])
    cold_all = cold_all if isinstance(cold_all, list) else []
    cold_rows = _cold_rows(sources.get("exp6824", {}))
    cold_audits = [
        row
        for row in cold_all
        if isinstance(row, dict) and row.get("row_type") == "row_fault_audit"
    ]
    attack_rows = _attack_rows(sources.get("exp6825", {}))
    attack_counts = Counter(str(row.get("attack_id")) for row in attack_rows)
    producer_ids = [str(row.get("row_id")) for row in producer_rows if isinstance(row, dict)]
    cold_ids = [str(row.get("row_id")) for row in cold_rows]
    attack_ids = [str(row.get("row_id")) for row in attack_rows]
    observation = {
        "producer_row_count": len(producer_rows),
        "producer_unique_row_count": len(set(producer_ids)),
        "cold_row_count": len(cold_rows),
        "cold_unique_row_count": len(set(cold_ids)),
        "cold_fault_audit_count": len(cold_audits),
        "cold_fault_audits_detected": all(bool(row.get("detected")) for row in cold_audits),
        "attack_row_count": len(attack_rows),
        "attack_unique_row_count": len(set(attack_ids)),
        "attack_ids": sorted(attack_counts),
        "attack_rows_per_id": dict(sorted(attack_counts.items())),
    }
    coverage = sources.get("exp6824", {}).get("row_coverage", {})
    replay = sources.get("exp6825", {}).get("fresh_process_replay", {})
    observation["upstream_manifest_receipts"] = {
        "cold_row_coverage_passed": isinstance(coverage, dict) and coverage.get("passed") is True,
        "attack_replay_byte_identical": isinstance(replay, dict)
        and replay.get("byte_identical") is True,
    }
    return observation


def _manifests_complete(observed: Mapping[str, Any]) -> bool:
    """Require exact counts and identities for all three evidence rosters."""

    expected_attack_counts = {
        attack_id: EXPECTED_ATTACK_ROWS_PER_ID for attack_id in ALL_ATTACK_IDS
    }
    return bool(
        observed.get("producer_row_count") == EXPECTED_PRODUCER_ROWS
        and observed.get("producer_unique_row_count") == EXPECTED_PRODUCER_ROWS
        and observed.get("cold_row_count") == EXPECTED_COLD_ROWS
        and observed.get("cold_unique_row_count") == EXPECTED_COLD_ROWS
        and observed.get("cold_fault_audit_count") == EXPECTED_COLD_AUDITS
        and observed.get("cold_fault_audits_detected") is True
        and observed.get("attack_row_count") == EXPECTED_ATTACK_ROWS
        and observed.get("attack_unique_row_count") == EXPECTED_ATTACK_ROWS
        and observed.get("attack_rows_per_id") == dict(sorted(expected_attack_counts.items()))
        and observed.get("upstream_manifest_receipts")
        == {"cold_row_coverage_passed": True, "attack_replay_byte_identical": True}
    )


def _source_hash_observation(
    sources: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path]
) -> JsonDict:
    """Compare actual producer bytes with both independent source receipts."""

    actual = {
        name: sha256_file(path) if path.is_file() else "missing"
        for name, path in sorted(paths.items())
    }
    cold_hashes = sources.get("exp6824", {}).get("source_artifact_hashes", {})
    attack_hashes = sources.get("exp6825", {}).get("source_artifact_hashes", {})
    cold_hashes = cold_hashes if isinstance(cold_hashes, dict) else {}
    attack_hashes = attack_hashes if isinstance(attack_hashes, dict) else {}
    return {
        "actual": actual,
        "cold_exp6813": cold_hashes.get("exp6813", {}).get("sha256")
        if isinstance(cold_hashes.get("exp6813"), dict)
        else None,
        "attack_exp6813": attack_hashes.get("exp6813", {}).get("sha256")
        if isinstance(attack_hashes.get("exp6813"), dict)
        else None,
        "cold_common": {
            key: value.get("sha256") if isinstance(value, dict) else None
            for key, value in cold_hashes.items()
            if key in {"exp6811", "exp6812"}
        },
        "attack_common": {
            key: value.get("sha256") if isinstance(value, dict) else None
            for key, value in attack_hashes.items()
            if key in {"exp6811", "exp6812"}
        },
    }


def _source_hashes_match(observed: Mapping[str, Any]) -> bool:
    """Accept only two independent receipts for the same producer and sources."""

    actual = observed.get("actual", {})
    producer = actual.get("exp6813") if isinstance(actual, dict) else None
    cold_common = observed.get("cold_common")
    attack_common = observed.get("attack_common")
    return bool(
        producer not in {None, "missing"}
        and observed.get("cold_exp6813") == producer
        and observed.get("attack_exp6813") == producer
        and isinstance(cold_common, dict)
        and cold_common == attack_common
        and set(cold_common) == {"exp6811", "exp6812"}
        and all(value not in {None, "missing"} for value in cold_common.values())
        and isinstance(actual, dict)
        and all(value != "missing" for value in actual.values())
    )


def _identity_observation(
    sources: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path]
) -> JsonDict:
    """Verify each recorded independent module against its current exact bytes."""

    root = paths["exp6813"].resolve().parent.parent
    identity_fields = (
        ("exp6824", "independent_parser_id"),
        ("exp6824", "independent_arbiter_id"),
        ("exp6824", "independent_reducer_id"),
        ("exp6825", "independent_attack_harness_id"),
    )
    identities: list[JsonDict] = []
    for source_name, field in identity_fields:
        identity = sources.get(source_name, {}).get(field, {})
        identity = identity if isinstance(identity, dict) else {}
        relative = identity.get("path")
        module_path = root / str(relative) if relative else root / "missing"
        recorded = identity.get("sha256")
        actual = sha256_file(module_path) if module_path.is_file() else "missing"
        identities.append(
            {
                "source": source_name,
                "field": field,
                "path": relative,
                "recorded_sha256": recorded,
                "actual_sha256": actual,
                "imports_exp6813": identity.get("imports_exp6813"),
                "valid": bool(
                    relative
                    and recorded == actual
                    and actual != "missing"
                    and identity.get("imports_exp6813") is False
                ),
            }
        )
    digests = [row["recorded_sha256"] for row in identities]
    return {"identities": identities, "unique_digest_count": len(set(digests))}


def _identities_valid(observed: Mapping[str, Any]) -> bool:
    """Require four valid and distinct implementations outside producer code."""

    identities = observed.get("identities", [])
    return bool(
        isinstance(identities, list)
        and len(identities) == 4
        and all(row.get("valid") is True for row in identities)
        and observed.get("unique_digest_count") == 4
    )


def check_preconditions(
    sources: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path]
) -> JsonDict:
    """Check every seal before reading any row as deployment evidence."""

    readability = _readability_observation(sources)
    readable = set(sources) == set(SOURCE_RELATIVE_PATHS) and all(
        value == "readable" for value in readability.values()
    )
    completion = {
        "cold_replay_shard_complete": sources.get("exp6824", {}).get("cold_replay_shard_complete"),
        "authority_attack_shard_complete": sources.get("exp6825", {}).get(
            "authority_attack_shard_complete"
        ),
    }
    manifests = _manifest_observation(sources)
    hashes = _source_hash_observation(sources, paths)
    identities = _identity_observation(sources, paths)
    verdicts = {name: source.get("verdict_class") for name, source in sorted(sources.items())}
    checks = [
        _check(
            "source_artifacts_readable",
            {name: "readable" for name in sorted(SOURCE_RELATIVE_PATHS)},
            readability,
            readable,
        ),
        _check(
            "upstream_completion_fields",
            {"cold_replay_shard_complete": True, "authority_attack_shard_complete": True},
            completion,
            completion
            == {"cold_replay_shard_complete": True, "authority_attack_shard_complete": True},
        ),
        _check(
            "complete_row_manifests",
            {
                "producer_rows": EXPECTED_PRODUCER_ROWS,
                "cold_rows": EXPECTED_COLD_ROWS,
                "cold_fault_audits": EXPECTED_COLD_AUDITS,
                "attack_rows": EXPECTED_ATTACK_ROWS,
            },
            manifests,
            _manifests_complete(manifests),
        ),
        _check(
            "source_artifact_hashes",
            "both shards bind the same readable producer and common sources",
            hashes,
            _source_hashes_match(hashes),
        ),
        _check(
            "independent_code_identities",
            "four distinct byte-matched identities with imports_exp6813=false",
            identities,
            _identities_valid(identities),
        ),
        _check(
            "terminal_verdict_classes",
            sorted(VERDICT_CLASSES),
            verdicts,
            set(verdicts) == set(SOURCE_RELATIVE_PATHS)
            and all(value in VERDICT_CLASSES for value in verdicts.values()),
        ),
    ]
    failed = [row["check"] for row in checks if not row["passed"]]
    return {"checks": checks, "failed_checks": failed, "passed": not failed}


def resolve_shard_outcomes(cold_result: str, attack_result: str) -> JsonDict:
    """Resolve all shard-result pairs with one explicit conservative order."""

    unknown = sorted({cold_result, attack_result}.difference(SHARD_RESULTS))
    if unknown:
        raise SealedAdoptionError(f"unknown shard result: {unknown}")
    controlling = max((cold_result, attack_result), key=SHARD_RESULT_RANK.__getitem__)
    deployment, verdict = SHARD_RESULT_AUTHORITY[controlling]
    return {
        "controlling_result": controlling,
        "deployment_adoption_decision": deployment,
        "verdict_class": verdict,
    }


def _paired_interval(
    values: Sequence[float], *, seed: int, resamples: int, alpha: float = 0.05
) -> JsonDict:
    """Recompute the fixed paired bootstrap without importing either shard."""

    if not values:
        return {
            "estimate": None,
            "lower_bound": None,
            "pair_count": 0,
            "resamples": resamples,
            "seed": seed,
            "upper_bound": None,
        }
    rng = Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(resamples)
    )
    return {
        "estimate": sum(values) / count,
        "lower_bound": samples[int((alpha / 2.0) * resamples)],
        "pair_count": count,
        "resamples": resamples,
        "seed": seed,
        "upper_bound": samples[min(resamples - 1, int((1.0 - alpha / 2.0) * resamples))],
    }


def _criterion_row(
    criterion: str, evidence_source: str, decision: str, observed: Mapping[str, Any]
) -> JsonDict:
    """Attach one closed decision to one evidence source and no other claim."""

    return {
        "row_id": f"{criterion}::{evidence_source}",
        "criterion": criterion,
        "evidence_source": evidence_source,
        "decision": decision,
        "passed": decision == "pass",
        "observed": dict(observed),
    }


def _attack_criterion(
    rows: Sequence[Mapping[str, Any]], attack_ids: Sequence[str]
) -> tuple[str, JsonDict]:
    """Reduce one fixed attack family while preserving incomplete evidence."""

    selected = [row for row in rows if row.get("attack_id") in set(attack_ids)]
    counts = Counter(str(row.get("attack_id")) for row in selected)
    complete = counts == Counter(
        {attack_id: EXPECTED_ATTACK_ROWS_PER_ID for attack_id in attack_ids}
    )
    applicable = {
        attack_id: sum(
            bool(row.get("applicable")) for row in selected if row.get("attack_id") == attack_id
        )
        for attack_id in attack_ids
    }
    failed_rows = [
        str(row.get("row_id"))
        for row in selected
        if row.get("applicable") and row.get("passed") is not True
    ]
    if not complete or not all(value > 0 for value in applicable.values()):
        decision = "insufficient"
    elif failed_rows:
        decision = "fail"
    else:
        decision = "pass"
    return decision, {
        "attack_ids": list(attack_ids),
        "applicable_count_by_attack": applicable,
        "failed_row_ids": failed_rows,
        "row_count": len(selected),
        "roster_complete": complete,
    }


def _join_held_pairs(
    cold: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Mapping[str, Any]]], bool]:
    """Join held arms exactly once so utility cannot use an unpaired row."""

    coverage = cold.get("row_coverage", {})
    held_ids = set(coverage.get("held_scenario_ids", [])) if isinstance(coverage, dict) else set()
    pairs: dict[str, dict[str, Mapping[str, Any]]] = {}
    duplicate = False
    for row in _cold_rows(cold):
        if row.get("scenario_id") not in held_ids:
            continue
        pair_id = str(row.get("pair_id"))
        arm = str(row.get("arm"))
        if arm in pairs.setdefault(pair_id, {}):
            duplicate = True
        pairs[pair_id][arm] = row
    complete = bool(
        not duplicate
        and len(pairs) == EXPECTED_HELD_PAIRS
        and all(set(pair) == {"selective_priority", "flat_reject_retry"} for pair in pairs.values())
    )
    return pairs, complete


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Count a Boolean row field without treating a missing row as false."""

    numerator = sum(row.get(field) is True for row in rows)
    denominator = len(rows)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _legal_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Keep legal support separate by model family to expose local regressions."""

    models = sorted({str(row.get("model_id")) for row in rows})
    return {
        model: _rate([row for row in rows if row.get("model_id") == model], "legality")
        for model in models
    }


def _utility_evidence(cold: Mapping[str, Any]) -> tuple[str, JsonDict]:
    """Recompute utility sign from paired rows and no safety headline."""

    pairs, complete = _join_held_pairs(cold)
    selective = [
        pair["selective_priority"] for pair in pairs.values() if "selective_priority" in pair
    ]
    flat = [pair["flat_reject_retry"] for pair in pairs.values() if "flat_reject_retry" in pair]
    deltas = [
        float(pair["selective_priority"].get("accepted_progress", 0.0))
        - float(pair["flat_reject_retry"].get("accepted_progress", 0.0))
        for pair in pairs.values()
        if set(pair) == {"selective_priority", "flat_reject_retry"}
    ]
    coverage = cold.get("row_coverage", {})
    constants = coverage.get("public_constants", {}) if isinstance(coverage, dict) else {}
    seed = int(constants.get("interval_seed", RANDOM_SEED))
    resamples = int(constants.get("paired_interval_resamples", 2000))
    interval = _paired_interval(deltas, seed=seed, resamples=resamples)
    harmful = {
        "selective_priority": _rate(selective, "harmful_selection"),
        "flat_reject_retry": _rate(flat, "harmful_selection"),
    }
    legal = {
        "selective_priority": _legal_rates(selective),
        "flat_reject_retry": _legal_rates(flat),
    }
    shared_models = set(legal["selective_priority"]) == set(legal["flat_reject_retry"])
    support_preserved = shared_models and all(
        legal["selective_priority"][model]["rate"] >= legal["flat_reject_retry"][model]["rate"]
        for model in legal["selective_priority"]
    )
    harmful_increase = (
        harmful["selective_priority"]["numerator"] > harmful["flat_reject_retry"]["numerator"]
    )
    lower = interval["lower_bound"]
    upper = interval["upper_bound"]
    if not complete or lower is None or upper is None:
        decision = "insufficient"
    elif harmful_increase or not support_preserved or upper < 0:
        decision = "fail"
    elif lower > 0:
        decision = "pass"
    else:
        decision = "insufficient"
    return decision, {
        "pair_roster_complete": complete,
        "paired_progress_delta": interval,
        "harmful_selections_by_arm": harmful,
        "legal_support_by_model_and_arm": legal,
        "support_preserved": support_preserved,
    }


def _row_rosters_complete(
    producer: Mapping[str, Any], cold: Mapping[str, Any], attacks: Mapping[str, Any]
) -> bool:
    """Reuse the exact manifest rule inside the decision disqualifier."""

    observed = _manifest_observation({"exp6813": producer, "exp6824": cold, "exp6825": attacks})
    return _manifests_complete(observed)


def _numeric_equal(left: Any, right: Any, tolerance: float) -> bool:
    """Compare numeric leaves with tolerance and other leaves by exact value."""

    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def _arithmetic_agrees(
    cold: Mapping[str, Any], evidence: Mapping[str, Any]
) -> tuple[bool, JsonDict]:
    """Compare fresh criterion arithmetic with cold and producer receipts."""

    aggregate = cold.get("aggregate_recomputation", {})
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    comparisons = {
        "hard_numerator": (
            evidence["hard_rate"]["numerator"],
            aggregate.get("hard_violation_rate_by_arm", {})
            .get("selective_priority", {})
            .get("numerator"),
        ),
        "safe_numerator": (
            evidence["safe_rate"]["numerator"],
            aggregate.get("safe_action_identity_by_arm", {})
            .get("selective_priority", {})
            .get("numerator"),
        ),
        "safe_denominator": (
            evidence["safe_rate"]["denominator"],
            aggregate.get("safe_action_identity_by_arm", {})
            .get("selective_priority", {})
            .get("denominator"),
        ),
        "certificate_numerator": (
            evidence["certificate_rate"]["numerator"],
            aggregate.get("certificate_completeness_by_arm", {})
            .get("selective_priority", {})
            .get("numerator"),
        ),
        "progress_estimate": (
            evidence["utility"]["paired_progress_delta"]["estimate"],
            aggregate.get("paired_progress_delta", {}).get("estimate"),
        ),
        "progress_lower_bound": (
            evidence["utility"]["paired_progress_delta"]["lower_bound"],
            aggregate.get("paired_progress_delta", {}).get("lower_bound"),
        ),
        "progress_upper_bound": (
            evidence["utility"]["paired_progress_delta"]["upper_bound"],
            aggregate.get("paired_progress_delta", {}).get("upper_bound"),
        ),
    }
    tolerance = float(cold.get("numerical_tolerance", NUMERICAL_TOLERANCE))
    local = {
        name: {
            "fresh": values[0],
            "cold": values[1],
            "within_tolerance": _numeric_equal(values[0], values[1], tolerance),
        }
        for name, values in comparisons.items()
    }
    headline = cold.get("headline_differences", {})
    headline = headline if isinstance(headline, dict) else {}
    headline_rows = headline.get("comparisons", [])
    producer_agreement = bool(
        headline.get("all_within_tolerance") is True
        and isinstance(headline_rows, list)
        and headline_rows
        and all(
            isinstance(row, dict) and row.get("within_tolerance") is True for row in headline_rows
        )
    )
    passed = all(row["within_tolerance"] for row in local.values()) and producer_agreement
    return passed, {
        "fresh_to_cold": local,
        "producer_to_cold_all_within_tolerance": producer_agreement,
        "tolerance": tolerance,
    }


def _component_decision(rows: Sequence[Mapping[str, Any]], criterion: str) -> str:
    """Combine only the evidence sources owned by one component."""

    decisions = [str(row["decision"]) for row in rows if row["criterion"] == criterion]
    if not decisions or "fail" in decisions:
        return "fail" if decisions else "insufficient"
    if "insufficient" in decisions:
        return "insufficient"
    return "pass"


def decide_deployment(components: Mapping[str, str], disqualifiers: Sequence[str]) -> str:
    """Map separate component decisions to one closed deployment authority."""

    non_utility = ("hard_safety", "safe_action_preservation", "certificate_truth")
    if disqualifiers or any(components.get(name) == "fail" for name in non_utility):
        return "redesign"
    if any(components.get(name) != "pass" for name in non_utility):
        return "insufficient"
    utility = components.get("utility")
    if utility == "pass":
        return "enable"
    if utility == "insufficient":
        return "keep_shadow"
    if utility == "fail":
        return "retire"
    return "insufficient"


def decision_table_complete(
    components: Mapping[str, str], disqualifiers: Sequence[str], deployment: str
) -> bool:
    """Close on terminal evidence without requiring a favorable deployment result."""

    expected_components = {
        "hard_safety",
        "safe_action_preservation",
        "utility",
        "certificate_truth",
    }
    return bool(
        set(components) == expected_components
        and all(value in COMPONENT_DECISIONS for value in components.values())
        and deployment in DEPLOYMENT_DECISIONS
        and deployment == decide_deployment(components, disqualifiers)
    )


def recompute_decision_table(
    producer: Mapping[str, Any], cold: Mapping[str, Any], attacks: Mapping[str, Any]
) -> JsonDict:
    """Build the terminal component table directly from the two shard rosters."""

    cold_rows = _cold_rows(cold)
    attack_rows = _attack_rows(attacks)
    pairs, pair_complete = _join_held_pairs(cold)
    held_selective = [
        pair["selective_priority"] for pair in pairs.values() if "selective_priority" in pair
    ]
    all_selective = [row for row in cold_rows if row.get("arm") == "selective_priority"]

    hard_rate = _rate(held_selective, "accepted_hard_violation")
    hard_cold_decision = (
        "insufficient" if not pair_complete else "pass" if hard_rate["numerator"] == 0 else "fail"
    )
    hard_attack_decision, hard_attack_evidence = _attack_criterion(attack_rows, PRIORITY_ATTACK_IDS)

    safe_rows = [row for row in held_selective if row.get("base_already_valid") is True]
    safe_rate = _rate(safe_rows, "safe_action_identity")
    safe_cold_decision = (
        "insufficient"
        if not pair_complete or not safe_rows
        else "pass"
        if safe_rate["numerator"] == safe_rate["denominator"]
        else "fail"
    )
    safe_attack_decision, safe_attack_evidence = _attack_criterion(
        attack_rows, SAFE_ACTION_ATTACK_IDS
    )

    utility_decision, utility_evidence = _utility_evidence(cold)

    certificate_rate = _rate(held_selective, "certificate_complete")
    certificate_cold_decision = (
        "insufficient"
        if not pair_complete
        else "pass"
        if certificate_rate["numerator"] == certificate_rate["denominator"]
        else "fail"
    )
    certificate_attack_decision, certificate_attack_evidence = _attack_criterion(
        attack_rows, CERTIFICATE_ATTACK_IDS
    )

    prohibited_decision, prohibited_evidence = _attack_criterion(
        attack_rows, PROHIBITED_FEATURE_ATTACK_IDS
    )
    rosters_complete = _row_rosters_complete(producer, cold, attacks)
    arithmetic_agrees, arithmetic_evidence = _arithmetic_agrees(
        cold,
        {
            "hard_rate": hard_rate,
            "safe_rate": safe_rate,
            "certificate_rate": certificate_rate,
            "utility": utility_evidence,
        },
    )
    seal_decision = "pass" if rosters_complete and arithmetic_agrees else "fail"

    rows = [
        _criterion_row(
            "hard_safety",
            "cold_replay_rows",
            hard_cold_decision,
            {"accepted_hard_violations": hard_rate, "pair_roster_complete": pair_complete},
        ),
        _criterion_row(
            "hard_safety", "authority_attack_rows", hard_attack_decision, hard_attack_evidence
        ),
        _criterion_row(
            "safe_action_preservation",
            "cold_replay_rows",
            safe_cold_decision,
            {"safe_action_identity": safe_rate, "pair_roster_complete": pair_complete},
        ),
        _criterion_row(
            "safe_action_preservation",
            "authority_attack_rows",
            safe_attack_decision,
            safe_attack_evidence,
        ),
        _criterion_row("utility", "cold_replay_rows", utility_decision, utility_evidence),
        _criterion_row(
            "certificate_truth",
            "cold_replay_rows",
            certificate_cold_decision,
            {"certificate_complete": certificate_rate, "pair_roster_complete": pair_complete},
        ),
        _criterion_row(
            "certificate_truth",
            "authority_attack_rows",
            certificate_attack_decision,
            certificate_attack_evidence,
        ),
        _criterion_row(
            "deployment_adoption",
            "prohibited_feature_rows",
            prohibited_decision,
            prohibited_evidence,
        ),
        _criterion_row(
            "deployment_adoption",
            "source_and_arithmetic_seals",
            seal_decision,
            {
                "source_rows_complete": rosters_complete,
                "arithmetic_agreement": arithmetic_evidence,
            },
        ),
    ]
    components = {
        criterion: _component_decision(rows, criterion)
        for criterion in (
            "hard_safety",
            "safe_action_preservation",
            "utility",
            "certificate_truth",
        )
    }
    disqualifiers: list[str] = []
    if prohibited_decision == "fail":
        disqualifiers.append("prohibited_feature_influence")
    if any(row.get("accepted_hard_violation") is True for row in all_selective):
        disqualifiers.append("accepted_hard_violation")
    if not rosters_complete:
        disqualifiers.append("source_rows_incomplete")
    if not arithmetic_agrees:
        disqualifiers.append("producer_cold_arithmetic_disagreement")
    deployment = decide_deployment(components, disqualifiers)
    if deployment == "enable":
        verdict = "positive"
        honest = "complete: sealed audits support selective-arbiter deployment adoption"
    elif deployment == "keep_shadow":
        verdict = "null"
        honest = (
            "complete: sealed audits support safety but utility remains insufficient; keep shadow"
        )
    elif deployment == "retire":
        verdict = "null"
        honest = "complete: sealed audits find harmful utility with closed safety evidence; retire"
    elif deployment == "redesign":
        verdict = "disqualified"
        honest = "complete: sealed audits disqualify deployment adoption; redesign"
    else:
        verdict = "partial"
        honest = "complete_partial_selective_arbiter_adoption_evidence_insufficient"
    return {
        "rows": rows,
        "component_decisions": components,
        "disqualifiers": disqualifiers,
        "deployment_adoption_decision": deployment,
        "selective_arbiter_audit_complete": decision_table_complete(
            components, disqualifiers, deployment
        ),
        "verdict_class": verdict,
        "honest_verdict": honest,
        "cold_row_count": len(cold_rows),
        "attack_row_count": len(attack_rows),
    }


def source_artifact_hashes(paths: Mapping[str, Path]) -> JsonDict:
    """Record exact bytes for the producer and both audit shards."""

    return {
        name: {"path": SOURCE_RELATIVE_PATHS[name].as_posix(), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
        if path.is_file()
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every emitted field except the checksum slot that holds the digest."""

    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(stable)


def _finish(payload: JsonDict) -> JsonDict:
    """Attach one field principle and seal the final output bytes logically."""

    payload["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in payload}
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    return payload


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    source_paths: Mapping[str, Path],
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Stop on a broken seal or emit the complete row-supported decision table."""

    gate = check_preconditions(sources, source_paths)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": "6826",
        "title": "Sealed selective-arbiter adoption receipt",
        "run_date": run_date,
        "status": "complete" if gate["passed"] else BLOCKED_STATUS,
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "replay_commands": list(REPLAY_COMMANDS),
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": source_artifact_hashes(source_paths),
        "rows": [],
        "adoption_criteria": deepcopy(ADOPTION_CRITERIA),
        "hard_safety_decision": "insufficient",
        "safe_action_preservation_decision": "insufficient",
        "utility_decision": "insufficient",
        "certificate_truth_decision": "insufficient",
        "deployment_adoption_decision": "insufficient",
        "selective_arbiter_audit_complete": False,
        "gate_check_summary": gate,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_STATUS,
    }
    if gate["passed"]:
        table = recompute_decision_table(sources["exp6813"], sources["exp6824"], sources["exp6825"])
        components = table["component_decisions"]
        payload.update(
            {
                "rows": table["rows"],
                "hard_safety_decision": components["hard_safety"],
                "safe_action_preservation_decision": components["safe_action_preservation"],
                "utility_decision": components["utility"],
                "certificate_truth_decision": components["certificate_truth"],
                "deployment_adoption_decision": table["deployment_adoption_decision"],
                "selective_arbiter_audit_complete": table["selective_arbiter_audit_complete"],
                "verdict_class": table["verdict_class"],
                "honest_verdict": table["honest_verdict"],
            }
        )
    else:
        failed = ", ".join(gate["failed_checks"])
        payload["honest_verdict"] = f"{BLOCKED_STATUS}: {failed}"
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        raise SealedAdoptionError("artifact field set does not match the sealed schema")
    return _finish(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject claims that are malformed, incomplete, or detached from rows."""

    findings: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS).difference(artifact))
    if missing:
        findings.append(f"missing required fields: {missing}")
    if set(artifact.get("field_principles", {})) != set(artifact):
        findings.append("field principle coverage mismatch")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
        findings.append("duration_s must be non-negative")
    if artifact.get("verifier_is_oracle") is not False:
        findings.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        findings.append("verdict class outside closed enum")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete:", "complete_", BLOCKED_STATUS)
    ):
        findings.append("honest verdict lacks terminal prefix")
    component_fields = (
        "hard_safety_decision",
        "safe_action_preservation_decision",
        "utility_decision",
        "certificate_truth_decision",
    )
    if any(artifact.get(field) not in COMPONENT_DECISIONS for field in component_fields):
        findings.append("component decision outside closed enum")
    if artifact.get("deployment_adoption_decision") not in DEPLOYMENT_DECISIONS:
        findings.append("deployment decision outside closed enum")
    rows = artifact.get("rows", [])
    row_pairs = (
        [(row.get("criterion"), row.get("evidence_source")) for row in rows]
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
        else []
    )
    blocked = artifact.get("status") == BLOCKED_STATUS
    if (blocked and row_pairs) or (
        not blocked and Counter(row_pairs) != Counter(CRITERION_SOURCE_PAIRS)
    ):
        findings.append("criterion-source rows are not exact")
    if artifact.get("selective_arbiter_audit_complete"):
        components = {
            "hard_safety": artifact.get("hard_safety_decision"),
            "safe_action_preservation": artifact.get("safe_action_preservation_decision"),
            "utility": artifact.get("utility_decision"),
            "certificate_truth": artifact.get("certificate_truth_decision"),
        }
        row_disqualifiers: list[str] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict) or row.get("decision") != "fail":
                    continue
                if row.get("evidence_source") == "prohibited_feature_rows":
                    row_disqualifiers.append("prohibited_feature_influence")
                if row.get("evidence_source") == "source_and_arithmetic_seals":
                    row_disqualifiers.append("producer_cold_arithmetic_disagreement")
        if not decision_table_complete(
            components,
            row_disqualifiers,
            str(artifact.get("deployment_adoption_decision")),
        ):
            findings.append("audit completion is not row-supported")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        findings.append("reproducibility checksum mismatch")
    return findings


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate then atomically replace only the caller-selected output file."""

    findings = validate_artifact(artifact)
    if findings:
        raise SealedAdoptionError("; ".join(findings))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded synthesis and print a small machine-readable receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paths = source_paths_for_root(args.root)
    output = args.output or args.root / RESULT_RELATIVE_PATH
    started = time.perf_counter()
    sources = load_sources(paths)
    artifact = build_artifact(
        sources,
        source_paths=paths,
        run_date=str(args.date),
        duration_s=time.perf_counter() - started,
    )
    write_artifact(output, artifact)
    print(
        json.dumps(
            {
                "deployment_adoption_decision": artifact["deployment_adoption_decision"],
                "output": str(output),
                "selective_arbiter_audit_complete": artifact["selective_arbiter_audit_complete"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
