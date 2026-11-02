"""Build a chronological fixture for risk-sensitive memory choice.

Spec refs: REQ-CL-6853 and SCENARIO-CL-6853-*.

The builder reads frozen JSON evidence. It does not import prior experiment
modules or run a learner. This separation prevents residual-memory decisions
from becoming features in the next selector's training stream.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6853_risk_sensitive_memory_opportunity_fixture.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6853_risk_sensitive_memory_opportunity_fixture.py"
)
WRAPPER_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6853_risk_sensitive_memory_opportunity_fixture.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6853_risk_sensitive_memory_opportunity_fixture.py"
)

SOURCE_RELATIVE_PATHS = {
    "evidence_contract": Path(
        "results/experiment_6848_v599_method_change_evidence_contract.json"
    ),
    "chronology": Path("results/experiment_6827_chronological_causal_edge_memory_stream.json"),
    "outcomes_a": Path("results/experiment_6840_residual_memory_chronological_shard_a.json"),
    "outcomes_b": Path(
        "results/experiment_6841_residual_memory_delayed_correction_shard_b.json"
    ),
}

SCHEMA = "carnot.experiment_6853.risk_sensitive_memory_opportunity_fixture.v1"
EXPERIMENT_ID = "exp6853-risk-sensitive-memory-opportunity-fixture"
INFERENCE_SUBSTRATE = "deterministic CPU chronological fixture construction"
RANDOM_SEED = 6_853_001
CAPACITY_BUDGET = 2
BLOCKED_VERDICT = "complete_blocked_risk_sensitive_memory_opportunity_fixture"
READY_VERDICT = "complete_null_risk_sensitive_memory_opportunity_fixture_ready"

FIRST_CLASS_ACTIONS = ("verified_memory", "no_memory", "abstain")
COMPARISON_BASELINES = ("random_admission", "always_memory")
MANDATED_FAMILIES = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
DECISION_CONTEXT_FIELDS = (
    "relevance",
    "uncertainty",
    "exact_compatibility",
    "age",
    "correction_status",
    "family",
    "capacity",
    "false_positive_risk",
    "cost",
)
DENIED_CONTEXT_FIELDS = frozenset(
    {
        "exact_later_outcome",
        "exact_outcome",
        "exact_outcome_hash",
        "outcome_identity",
        "signed_direction",
        "memory_effect_class",
        "safe_selection_headroom",
        "observed_potential_outcome_support",
        "predicted_direction",
        "memory_dose",
        "admission_decision",
        "negative_transfer",
        "negative_transfer_delta",
        "residual_pressure",
        "decision_correct",
        "regret",
    }
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "random_seed",
    "reproducibility_checksum",
    "rows",
    "decision_context_schema",
    "feature_provenance_manifest",
    "action_manifest",
    "outcome_authority_manifest",
    "chronological_split_manifest",
    "leakage_attack_results",
    "decision_headroom_rows",
    "helpful_memory_count",
    "harmful_memory_count",
    "abstention_opportunity_count",
    "memory_headroom_nonzero_score",
    "risk_sensitive_stream_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

OPEN_SPEC_IDS = (
    "REQ-CL-6853",
    "SCENARIO-CL-6853-PRECONDITIONS",
    "SCENARIO-CL-6853-LEAKAGE",
    "SCENARIO-CL-6853-DUPLICATES",
    "SCENARIO-CL-6853-ACTIONS",
    "SCENARIO-CL-6853-HEADROOM",
    "SCENARIO-CL-6853-FAMILY-BALANCE",
    "SCENARIO-CL-6853-READY",
)

REPLAY_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6853_risk_sensitive_memory_opportunity_fixture.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_6853_risk_sensitive_memory_opportunity_fixture.py' -m pytest tests/python/test_experiment_6853_risk_sensitive_memory_opportunity_fixture.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check python/carnot/experiment_6853_risk_sensitive_memory_opportunity_fixture.py scripts/experiments/experiment_6853_risk_sensitive_memory_opportunity_fixture.py tests/python/test_experiment_6853_risk_sensitive_memory_opportunity_fixture.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6853_risk_sensitive_memory_opportunity_fixture.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6853_risk_sensitive_memory_opportunity_fixture.json",
    ".venv/bin/python scripts/artifact_convention_audit.py results/experiment_6853_risk_sensitive_memory_opportunity_fixture.json",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6853_risk_sensitive_memory_opportunity_fixture.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

RELEVANCE_BY_SCENARIO = {
    "stale_prerequisites": 0.95,
    "competing_authorities": 0.85,
    "fallback": 0.70,
    "consequence": 0.75,
    "already_safe_proposals": 0.20,
    "soft_conflict": 0.60,
}
UNCERTAINTY_BY_SCENARIO = {
    "stale_prerequisites": 0.25,
    "competing_authorities": 0.80,
    "fallback": 0.65,
    "consequence": 0.45,
    "already_safe_proposals": 0.10,
    "soft_conflict": 0.90,
}
FALSE_POSITIVE_RISK_BY_SCENARIO = {
    "stale_prerequisites": 0.65,
    "competing_authorities": 0.85,
    "fallback": 0.35,
    "consequence": 0.25,
    "already_safe_proposals": 0.95,
    "soft_conflict": 0.90,
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable JSON bytes so the same fixture has the same identity."""

    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 value in the repository's explicit string form."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash structured content after stable serialization."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str | None:
    """Hash a readable file without loading its full contents into memory."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_paths_for_root(root: Path) -> dict[str, Path]:
    """Resolve frozen inputs relative to the selected checkout."""

    return {name: root / path for name, path in SOURCE_RELATIVE_PATHS.items()}


def load_sources(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Read source JSON while preserving a machine-readable failure marker."""

    sources: dict[str, JsonDict] = {}
    for name, path in paths.items():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            value = {"_load_error": type(error).__name__}
        sources[name] = value if isinstance(value, dict) else {"_load_error": "not_object"}
    return sources


def source_artifact_hashes(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Bind each authority file to its path, role, and content digest."""

    roles = {
        "evidence_contract": "precondition_authority",
        "chronology": "pre_outcome_decision_authority",
        "outcomes_a": "exact_later_outcome_authority_orders_1_to_3",
        "outcomes_b": "exact_later_outcome_authority_orders_4_to_5",
    }
    result: dict[str, JsonDict] = {}
    for name, path in sorted(paths.items()):
        try:
            display_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            display_path = str(path)
        result[name] = {
            "path": display_path,
            "sha256": sha256_file(path),
            "role": roles[name],
        }
    return result


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Return one uniform gate row for terminal and blocked artifacts."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def _order_index(order_id: str) -> int:
    """Read the numeric order suffix used by the frozen chronology."""

    try:
        return int(order_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def select_chronological_rows(source: Mapping[str, Any]) -> list[JsonDict]:
    """Select factual write-read edges without importing a prior producer."""

    source_rows = source.get("rows", [])
    if not isinstance(source_rows, list):
        return []
    rows = [
        deepcopy(row)
        for row in source_rows
        if isinstance(row, dict)
        and row.get("counterfactual_applicable") is True
        and row.get("action_identity")
        and row.get("outcome_identity")
        and row.get("causal_edge_id")
        and row.get("write_operation_id")
        and row.get("read_operation_id")
    ]
    return sorted(
        rows,
        key=lambda row: (
            _order_index(str(row.get("order_id", ""))),
            str(row.get("source_family", "")),
            int(row.get("chronological_position", 0)),
            str(row.get("counterfactual_kind", "")),
            str(row.get("row_id", "")),
        ),
    )


def build_outcome_index(
    outcomes_a: Mapping[str, Any], outcomes_b: Mapping[str, Any]
) -> tuple[dict[str, JsonDict], list[JsonDict]]:
    """Index exact outcomes and reject conflicting receipts for one decision."""

    index: dict[str, JsonDict] = {}
    conflicts: list[JsonDict] = []
    for source_name, source in (("outcomes_a", outcomes_a), ("outcomes_b", outcomes_b)):
        source_rows = source.get("rows", [])
        if not isinstance(source_rows, list):
            continue
        for row in source_rows:
            if not isinstance(row, dict):
                continue
            decision_id = str(row.get("source_event_row_id", ""))
            outcome = row.get("exact_outcome")
            if not decision_id or not isinstance(outcome, dict):
                continue
            receipt = {
                "outcome_identity": outcome.get("outcome_identity"),
                "exact_outcome_hash": outcome.get("exact_outcome_hash"),
                "signed_direction": outcome.get("signed_direction"),
                "source_artifact": source_name,
            }
            if decision_id in index and index[decision_id] != receipt:
                conflicts.append(
                    {
                        "decision_id": decision_id,
                        "first": index[decision_id],
                        "conflicting": receipt,
                    }
                )
            else:
                index[decision_id] = receipt
    return index, conflicts


def _scenario_family(event_id: str) -> str:
    """Extract the stable case family from an event identity."""

    parts = event_id.split("|")
    if len(parts) < 2:
        return "unknown"
    return parts[1].rsplit("-", 1)[0]


def _correction_status(scenario: str) -> str:
    """Map visible case types to a pre-outcome correction state."""

    if scenario in {"fallback", "consequence"}:
        return "delayed_correction_pending"
    if scenario == "stale_prerequisites":
        return "stale_correction_pending"
    if scenario in {"competing_authorities", "soft_conflict"}:
        return "conflicting_correction_pending"
    if scenario == "already_safe_proposals":
        return "no_correction_needed"
    return "correction_status_unknown"


def _operation_position(operation_id: Any) -> int:
    """Read a stable sequence suffix without consulting later outcomes."""

    try:
        return int(str(operation_id).rsplit("::", 1)[1])
    except (IndexError, ValueError):
        return 0


def decision_context(source_row: Mapping[str, Any]) -> JsonDict:
    """Build only fixed features that exist before the outcome boundary."""

    scenario = _scenario_family(str(source_row.get("event_id", "")))
    chronological_position = int(source_row.get("chronological_position", 0))
    write_position = _operation_position(source_row.get("write_operation_id"))
    exact_compatibility = bool(
        source_row.get("operation_admitted") is True
        and source_row.get("causal_edge_id")
        and source_row.get("action_identity")
        and source_row.get("write_operation_id")
        and source_row.get("read_operation_id")
    )
    return {
        "relevance": RELEVANCE_BY_SCENARIO.get(scenario, 0.0),
        "uncertainty": UNCERTAINTY_BY_SCENARIO.get(scenario, 1.0),
        "exact_compatibility": exact_compatibility,
        "age": max(0, chronological_position - write_position),
        "correction_status": _correction_status(scenario),
        "family": str(source_row.get("source_family", "")),
        "capacity": {
            "budget": CAPACITY_BUDGET,
            "factual_read_admitted": source_row.get("operation_admitted") is True,
        },
        "false_positive_risk": FALSE_POSITIVE_RISK_BY_SCENARIO.get(scenario, 1.0),
        "cost": {"verified_memory": 2, "no_memory": 1, "abstain": 0},
    }


def _random_baseline_action(decision_id: str) -> str:
    """Freeze the placebo baseline without using the later outcome."""

    digest = sha256_json({"decision_id": decision_id, "seed": RANDOM_SEED})
    return "verified_memory" if int(digest[7:15], 16) % 2 == 0 else "no_memory"


def _effect_class(direction: int) -> str:
    """Convert the authority's signed observed outcome into a class label."""

    if direction > 0:
        return "helpful"
    if direction < 0:
        return "harmful"
    return "ambiguous"


def _outcome_for_decision(
    decision_id: str, outcome_index: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    """Join plain shard-A identities or sealed shard-B identity hashes."""

    direct = outcome_index.get(decision_id)
    if direct is not None:
        return direct
    return outcome_index.get(sha256_json(str(decision_id)))


def build_rows(
    chronological_rows: Sequence[Mapping[str, Any]],
    outcome_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[JsonDict], list[str]]:
    """Join each pre-outcome decision to one exact later receipt."""

    rows: list[JsonDict] = []
    missing: list[str] = []
    for sequence_index, source_row in enumerate(chronological_rows, start=1):
        decision_id = str(source_row.get("row_id", ""))
        exact_outcome = _outcome_for_decision(decision_id, outcome_index)
        if exact_outcome is None:
            missing.append(decision_id)
            continue
        context = decision_context(source_row)
        direction = int(exact_outcome["signed_direction"])
        effect_class = _effect_class(direction)
        headroom = int(direction != 0)
        delayed = context["correction_status"] == "delayed_correction_pending"
        headroom_label = "nonzero" if headroom else "zero"
        correction_label = "delayed_correction" if delayed else str(context["correction_status"])
        row: JsonDict = {
            "decision_id": decision_id,
            "decision_sequence_index": sequence_index,
            "order_id": str(source_row.get("order_id", "")),
            "chronological_position": int(source_row.get("chronological_position", 0)),
            "family": str(source_row.get("source_family", "")),
            "stratum": f"{source_row.get('source_family', '')}|{correction_label}|{headroom_label}",
            "decision_context": context,
            "available_actions": list(FIRST_CLASS_ACTIONS),
            "action_availability": {action: True for action in FIRST_CLASS_ACTIONS},
            "observed_action": "verified_memory",
            "observed_potential_outcome_support": {
                "verified_memory": {
                    "observed": True,
                    "authority": str(exact_outcome["source_artifact"]),
                },
                "no_memory": {
                    "observed": False,
                    "reason": "no_exact_action_receipt",
                },
                "abstain": {
                    "observed": False,
                    "reason": "no_exact_action_receipt",
                },
            },
            "baseline_actions": {
                "random_admission": _random_baseline_action(decision_id),
                "always_memory": "verified_memory",
            },
            "exact_later_outcome": {
                "outcome_identity": exact_outcome["outcome_identity"],
                "exact_outcome_hash": exact_outcome["exact_outcome_hash"],
                "signed_direction": direction,
                "revealed_after_decision": True,
                "source_artifact": exact_outcome["source_artifact"],
            },
            "memory_effect_class": effect_class,
            "safe_selection_headroom": headroom,
            "abstention_opportunity": bool(
                effect_class == "ambiguous"
                or (
                    float(context["uncertainty"]) >= 0.8
                    and float(context["false_positive_risk"]) >= 0.8
                )
            ),
            "delayed_correction": delayed,
        }
        row["row_sha256"] = sha256_json(row)
        rows.append(row)
    return rows, missing


def walk_keys(value: Any) -> Iterable[str]:
    """Yield every nested mapping key for residual-field exclusion checks."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def find_leakage_paths(value: Any, prefix: str = "") -> list[str]:
    """Return paths where post-outcome or learner fields cross the boundary."""

    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in DENIED_CONTEXT_FIELDS:
                paths.append(path)
            paths.extend(find_leakage_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            paths.extend(find_leakage_paths(child, path))
    return paths


def leakage_attack_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Prove the detector rejects direct, nested, and learned-field injections."""

    clean_paths: list[str] = []
    for index, row in enumerate(rows):
        context = row.get("decision_context", {})
        for path in find_leakage_paths(context):
            clean_paths.append(f"rows[{index}].decision_context.{path}")
    attacks = (
        ("direct_outcome", {"outcome_identity": "injected"}),
        ("nested_outcome", {"nested": {"exact_outcome_hash": "injected"}}),
        ("learned_decision", {"predicted_direction": 1}),
    )
    injection_rows = []
    for name, injection in attacks:
        detected_paths = find_leakage_paths(injection)
        injection_rows.append(
            {"attack": name, "detected": bool(detected_paths), "detected_paths": detected_paths}
        )
    return {
        "clean_fixture_leakage_paths": clean_paths,
        "injection_attacks": injection_rows,
        "passed": not clean_paths and all(row["detected"] for row in injection_rows),
    }


def family_balance_check(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require equal nonzero representation from all mandated model families."""

    counts = Counter(str(row.get("family", "")) for row in rows)
    observed = dict(sorted(counts.items()))
    values = [counts[family] for family in MANDATED_FAMILIES]
    passed = set(counts) == set(MANDATED_FAMILIES) and len(set(values)) == 1 and values[0] > 0
    return _check(
        "family_balance",
        {"families": list(MANDATED_FAMILIES), "equal_nonzero_counts": True},
        observed,
        passed,
    )


def validate_opportunity_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Validate chronology, identity, action, outcome, leakage, and balance."""

    decision_ids = [str(row.get("decision_id", "")) for row in rows]
    duplicate_count = len(decision_ids) - len(set(decision_ids))
    missing_outcomes = [
        row.get("decision_id")
        for row in rows
        if not isinstance(row.get("exact_later_outcome"), Mapping)
        or not row.get("exact_later_outcome", {}).get("exact_outcome_hash")
        or not row.get("exact_later_outcome", {}).get("outcome_identity")
        or row.get("exact_later_outcome", {}).get("revealed_after_decision") is not True
    ]
    required_actions = set(FIRST_CLASS_ACTIONS)
    invalid_actions = [
        row.get("decision_id")
        for row in rows
        if set(row.get("available_actions", [])) != required_actions
        or set(row.get("action_availability", {})) != required_actions
        or not all(row.get("action_availability", {}).values())
    ]
    leakage_paths = []
    for index, row in enumerate(rows):
        leakage_paths.extend(
            f"rows[{index}].decision_context.{path}"
            for path in find_leakage_paths(row.get("decision_context", {}))
        )
    sequence = [row.get("decision_sequence_index") for row in rows]
    return [
        _check(
            "chronological_rows",
            list(range(1, len(rows) + 1)),
            sequence,
            sequence == list(range(1, len(rows) + 1)),
        ),
        _check(
            "unique_decision_identities",
            {"duplicate_count": 0, "blank_count": 0},
            {
                "duplicate_count": duplicate_count,
                "blank_count": sum(not decision_id for decision_id in decision_ids),
            },
            duplicate_count == 0 and all(decision_ids),
        ),
        _check(
            "exact_later_outcomes_complete",
            {"missing_count": 0},
            {"missing_count": len(missing_outcomes), "decision_ids": missing_outcomes[:10]},
            not missing_outcomes,
        ),
        _check(
            "first_class_actions_available",
            {"actions": list(FIRST_CLASS_ACTIONS), "invalid_count": 0},
            {"invalid_count": len(invalid_actions), "decision_ids": invalid_actions[:10]},
            not invalid_actions,
        ),
        _check("decision_context_nonleaking", [], leakage_paths, not leakage_paths),
        family_balance_check(rows),
    ]


def summarize_headroom(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Measure safe-selection opportunities overall and within each stratum."""

    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row.get("stratum", ""))].append(row)

    def summary(group: Sequence[Mapping[str, Any]]) -> JsonDict:
        effects = Counter(str(row.get("memory_effect_class", "")) for row in group)
        nonzero = sum(int(row.get("safe_selection_headroom", 0)) > 0 for row in group)
        return {
            "decision_count": len(group),
            "helpful_memory_count": effects["helpful"],
            "harmful_memory_count": effects["harmful"],
            "ambiguous_memory_count": effects["ambiguous"],
            "nonzero_headroom_count": nonzero,
            "zero_headroom_count": len(group) - nonzero,
        }

    return {**summary(rows), "by_stratum": {key: summary(value) for key, value in sorted(by_stratum.items())}}


def decision_headroom_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep one compact headroom record for every decision identity."""

    return [
        {
            "decision_id": row["decision_id"],
            "stratum": row["stratum"],
            "memory_effect_class": row["memory_effect_class"],
            "safe_selection_headroom": row["safe_selection_headroom"],
            "observed_action": row["observed_action"],
            "supported_action_count": sum(
                bool(value.get("observed"))
                for value in row["observed_potential_outcome_support"].values()
            ),
        }
        for row in rows
    ]


def _decision_context_schema() -> JsonDict:
    """Describe the fixed feature boundary for the next experiment."""

    return {
        "fields": {
            "relevance": "float_0_to_1",
            "uncertainty": "float_0_to_1",
            "exact_compatibility": "bool",
            "age": "nonnegative_integer_events",
            "correction_status": "closed_string",
            "family": "source_family_string",
            "capacity": "budget_and_factual_read_receipt",
            "false_positive_risk": "float_0_to_1",
            "cost": "fixed_action_cost_units",
        },
        "field_order": list(DECISION_CONTEXT_FIELDS),
        "denied_fields": sorted(DENIED_CONTEXT_FIELDS),
        "outcome_boundary": "exact_later_outcome_is_joined_after_decision_context_freeze",
    }


def _feature_provenance_manifest() -> JsonDict:
    """Record the source or fixed rule behind every decision feature."""

    return {
        "relevance": "fixed lookup from the pre-outcome event scenario family",
        "uncertainty": "fixed lookup from the pre-outcome event scenario family",
        "exact_compatibility": "Exp6827 admitted causal edge with write, read, and action identities",
        "age": "Exp6827 chronological position minus write operation position",
        "correction_status": "fixed lookup from the pre-outcome event scenario family",
        "family": "Exp6827 source_family",
        "capacity": "Exp6827 capacity contract and factual read admission receipt",
        "false_positive_risk": "fixed lookup from the pre-outcome event scenario family",
        "cost": "fixed action costs from the Exp6853 action manifest",
        "outcome_fields_used": [],
        "learned_fields_used": [],
    }


def _action_manifest() -> JsonDict:
    """Separate selectable actions from later comparison policies."""

    return {
        "first_class_actions": {
            "verified_memory": {
                "description": "Expose the verified factual memory edge.",
                "cost_units": 2,
            },
            "no_memory": {
                "description": "Continue without memory injection.",
                "cost_units": 1,
            },
            "abstain": {
                "description": "Do not inject memory or claim a task action.",
                "cost_units": 0,
            },
        },
        "comparison_baselines": {
            "random_admission": {
                "rule": "SHA-256 parity selects verified_memory or no_memory.",
                "random_seed": RANDOM_SEED,
            },
            "always_memory": {"rule": "Select verified_memory for every decision."},
        },
    }


def _outcome_authority_manifest() -> JsonDict:
    """Limit the outcome join to exact fields already present in source JSON."""

    return {
        "chronology_source": "experiment_6827 rows with factual causal-edge identities",
        "exact_outcome_sources": {
            "orders_1_to_3": "experiment_6840 rows",
            "orders_4_to_5": "experiment_6841 sealed-identity rows",
        },
        "accepted_outcome_fields": [
            "outcome_identity",
            "exact_outcome_hash",
            "signed_direction",
        ],
        "observed_action": "verified_memory",
        "unsupported_actions_have_null_outcomes": True,
        "counterfactual_outcomes_fabricated": False,
        "producer_modules_imported": [],
        "excluded_source_fields": [
            "memory_dose",
            "predicted_direction",
            "admission_decision",
            "negative_transfer_delta",
            "residual_pressure",
        ],
        "effect_label_rule": {
            "positive": "helpful",
            "negative": "harmful",
            "zero": "ambiguous",
        },
    }


def chronological_split_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize the frozen order and family coverage without changing splits."""

    by_order = Counter(str(row.get("order_id", "")) for row in rows)
    by_family = Counter(str(row.get("family", "")) for row in rows)
    return {
        "ordering_rule": "order number, family, chronological position, counterfactual kind, decision identity",
        "decision_count": len(rows),
        "by_order": dict(sorted(by_order.items(), key=lambda item: _order_index(item[0]))),
        "by_family": dict(sorted(by_family.items())),
        "first_decision_id": rows[0]["decision_id"] if rows else None,
        "last_decision_id": rows[-1]["decision_id"] if rows else None,
    }


def _field_principles() -> JsonDict:
    """Explain why every top-level artifact field exists."""

    return {
        "schema": "A versioned schema prevents silent fixture reinterpretation.",
        "experiment_id": "A stable identity binds the fixture to its task.",
        "title": "The title states that this artifact contains opportunities, not a learner.",
        "run_date": "The supplied date fixes the execution boundary.",
        "status": "Status distinguishes complete construction from a blocked gate.",
        "openspec_requirement_ids": "Requirement anchors connect behavior to tests.",
        "replay_commands": "Commands let another operator repeat each validation layer.",
        "field_principles": "Each top-level field has one plain-language purpose.",
        "preconditions_checked": "Source and boundary failures stop the fixture.",
        "inference_substrate": "The builder uses deterministic CPU JSON construction and no LLM.",
        "duration_s": "Measured wall time shows that construction ran.",
        "source_artifact_hashes": "Hashes bind the exact evidence bytes.",
        "random_seed": "One fixed seed controls only the random-admission baseline.",
        "reproducibility_checksum": "The checksum binds stable content and excludes wall time.",
        "rows": "Each row joins one decision, one observed action, and one exact later outcome.",
        "decision_context_schema": "The schema fixes the pre-outcome feature boundary.",
        "feature_provenance_manifest": "The manifest names the source of every feature.",
        "action_manifest": "The manifest separates actions from baseline policies.",
        "outcome_authority_manifest": "The manifest prevents invented action outcomes.",
        "chronological_split_manifest": "The manifest proves order and family coverage.",
        "leakage_attack_results": "Injection attacks prove denied fields are detected.",
        "decision_headroom_rows": "One compact row records safe-selection headroom per decision.",
        "headroom_summary": "Overall and stratum counts preserve zero and nonzero controls.",
        "helpful_memory_count": "A positive count proves observed useful memory cases exist.",
        "harmful_memory_count": "A positive count proves observed harmful memory cases exist.",
        "abstention_opportunity_count": "The count identifies ambiguous or high-risk choices.",
        "memory_headroom_nonzero_score": "The score requires both safe-selection directions.",
        "risk_sensitive_stream_ready_score": "Exp6854 consumes this fully conjunctive fixture gate.",
        "gate_check_summary": "Every failed check keeps its expected and observed value.",
        "verifier_is_oracle": "False because the builder does not define outcome correctness.",
        "verdict_class": "A closed class prevents fixture readiness from becoming benefit.",
        "honest_verdict": "A complete prefix gives the conductor a terminal result.",
    }


def _base_artifact(
    *,
    repo_root: Path,
    run_date: str,
    duration_s: float,
    rows: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Create stable artifact fields before final gate computation."""

    headroom = summarize_headroom(rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "Risk-Sensitive Memory Opportunity Fixture",
        "run_date": run_date,
        "status": "complete",
        "openspec_requirement_ids": list(OPEN_SPEC_IDS),
        "replay_commands": list(REPLAY_COMMANDS),
        "field_principles": _field_principles(),
        "preconditions_checked": list(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "source_artifact_hashes": source_artifact_hashes(source_paths_for_root(repo_root)),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "rows": list(rows),
        "decision_context_schema": _decision_context_schema(),
        "feature_provenance_manifest": _feature_provenance_manifest(),
        "action_manifest": _action_manifest(),
        "outcome_authority_manifest": _outcome_authority_manifest(),
        "chronological_split_manifest": chronological_split_manifest(rows),
        "leakage_attack_results": leakage_attack_results(rows),
        "decision_headroom_rows": decision_headroom_rows(rows),
        "headroom_summary": headroom,
        "helpful_memory_count": headroom["helpful_memory_count"],
        "harmful_memory_count": headroom["harmful_memory_count"],
        "abstention_opportunity_count": sum(
            bool(row.get("abstention_opportunity")) for row in rows
        ),
        "memory_headroom_nonzero_score": 0,
        "risk_sensitive_stream_ready_score": 0,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_VERDICT,
    }
    return artifact


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return all checks plus a direct first-failure receipt."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "checks": list(checks),
        "failed_checks": failed,
        "failed_check": failed[0]["check"] if failed else None,
        "observed": failed[0]["observed"] if failed else None,
        "passed": not failed,
    }


def build_artifact(
    repo_root: Path,
    *,
    run_date: str,
    duration_s: float,
    sources: Mapping[str, Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build a ready fixture or a schema-complete blocked artifact."""

    paths = source_paths_for_root(repo_root)
    loaded = dict(sources) if sources is not None else load_sources(paths)
    readable_observed = {
        name: source.get("_load_error", "readable")
        for name, source in sorted(loaded.items())
    }
    readable = set(loaded) == set(SOURCE_RELATIVE_PATHS) and all(
        value == "readable" for value in readable_observed.values()
    )
    contract_score = loaded.get("evidence_contract", {}).get(
        "v599_evidence_contract_ready_score"
    )
    preconditions = [
        _check("v599_evidence_contract_ready_score", 1, contract_score, contract_score == 1),
        _check(
            "source_artifacts_readable",
            {name: "readable" for name in sorted(SOURCE_RELATIVE_PATHS)},
            readable_observed,
            readable,
        ),
    ]
    if any(row["passed"] is not True for row in preconditions):
        artifact = _base_artifact(
            repo_root=repo_root,
            run_date=run_date,
            duration_s=duration_s,
            rows=[],
            preconditions=preconditions,
        )
        artifact["gate_check_summary"] = _gate_summary(preconditions)
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    chronological_rows = select_chronological_rows(loaded["chronology"])
    outcome_index, conflicts = build_outcome_index(loaded["outcomes_a"], loaded["outcomes_b"])
    rows, missing = build_rows(chronological_rows, outcome_index)
    decision_ids = [str(row.get("row_id", "")) for row in chronological_rows]
    stable_identities = bool(
        decision_ids
        and len(decision_ids) == len(set(decision_ids))
        and all(decision_ids)
        and all(str(row.get("action_identity", "")).startswith("sha256:") for row in chronological_rows)
        and all(str(row.get("outcome_identity", "")).startswith("sha256:") for row in chronological_rows)
    )
    preconditions.extend(
        [
            _check(
                "chronological_source_rows",
                {"readable_rows": 765},
                {"readable_rows": len(chronological_rows)},
                len(chronological_rows) == 765,
            ),
            _check(
                "stable_decision_identities",
                {"decision_count": len(chronological_rows), "duplicate_count": 0},
                {
                    "decision_count": len(decision_ids),
                    "duplicate_count": len(decision_ids) - len(set(decision_ids)),
                },
                stable_identities,
            ),
            _check(
                "exact_later_outcome_join",
                {"missing_count": 0, "conflict_count": 0},
                {
                    "missing_count": len(missing),
                    "conflict_count": len(conflicts),
                    "missing_decision_ids": missing[:10],
                },
                not missing and not conflicts and len(rows) == len(chronological_rows),
            ),
        ]
    )
    artifact = _base_artifact(
        repo_root=repo_root,
        run_date=run_date,
        duration_s=duration_s,
        rows=rows,
        preconditions=preconditions,
    )
    row_checks = validate_opportunity_rows(rows)
    headroom = artifact["headroom_summary"]
    headroom_check = _check(
        "nonzero_safe_selection_headroom",
        {
            "helpful_memory_count": ">0",
            "harmful_memory_count": ">0",
            "nonzero_headroom_count": ">0",
            "zero_headroom_count": ">0",
        },
        {
            "helpful_memory_count": headroom["helpful_memory_count"],
            "harmful_memory_count": headroom["harmful_memory_count"],
            "nonzero_headroom_count": headroom["nonzero_headroom_count"],
            "zero_headroom_count": headroom["zero_headroom_count"],
        },
        headroom["helpful_memory_count"] > 0
        and headroom["harmful_memory_count"] > 0
        and headroom["nonzero_headroom_count"] > 0
        and headroom["zero_headroom_count"] > 0,
    )
    leakage_check = _check(
        "leakage_attacks",
        True,
        artifact["leakage_attack_results"],
        artifact["leakage_attack_results"]["passed"] is True,
    )
    checks = [*preconditions, *row_checks, headroom_check, leakage_check]
    gate_summary = _gate_summary(checks)
    artifact["gate_check_summary"] = gate_summary
    artifact["preconditions_checked"] = preconditions
    artifact["memory_headroom_nonzero_score"] = int(headroom_check["passed"] is True)
    artifact["risk_sensitive_stream_ready_score"] = int(gate_summary["passed"] is True)
    if artifact["risk_sensitive_stream_ready_score"] == 1:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = READY_VERDICT
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable fixture content while excluding wall time and checksum recursion."""

    unsigned = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(unsigned)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and consistency errors before writing the deliverable."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if str(artifact.get("honest_verdict", "")).startswith("complete_") is False:
        errors.append("honest_verdict must start with complete_")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("risk_sensitive_stream_ready_score") == 1:
        if artifact.get("gate_check_summary", {}).get("passed") is not True:
            errors.append("ready score contradicts failed gates")
        if validate_opportunity_rows(artifact.get("rows", []))[-1]["passed"] is not True:
            errors.append("ready fixture is family-imbalanced")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write one validated result without touching any other tracked file."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run deterministic construction for the supplied execution date."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Execution date in YYYYMMDD form.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    artifact = build_artifact(
        REPO_ROOT,
        run_date=args.date,
        duration_s=0.0,
    )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_artifact(args.output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - the tested wrapper owns CLI execution.
    raise SystemExit(main())
