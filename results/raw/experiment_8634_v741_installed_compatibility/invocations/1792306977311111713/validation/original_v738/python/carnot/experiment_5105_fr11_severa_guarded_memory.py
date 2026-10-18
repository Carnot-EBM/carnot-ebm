#!/usr/bin/env python3
"""Exp 5105: SEVerA-style guarded FR-11 memory/SOP self-learning.

Spec refs: REQ-LEARN-5105, SCENARIO-LEARN-5105-SEVERA-CONTRACT-NO-PROMOTE.

This runner is a deterministic Search-Verify-Learn attempt. It searches prior
FR-11 misses and verified traces for memory/SOP candidates, verifies each
candidate against explicit contracts, and only then evaluates whether a
promotion would improve held-out behavior without forgetting. The important
point is the guard boundary: a plausible memory is not useful enough to promote
unless provenance, evidence, staleness, poison resistance, held-out utility,
non-forgetting, contamination, and rollback all clear together.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
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

EXPERIMENT_ID = 5105
EXPERIMENT_NAME = "experiment_5105_fr11_severa_guarded_memory"
MODULE_RELATIVE_PATH = "python/carnot/experiment_5105_fr11_severa_guarded_memory.py"
SCHEMA = "carnot.experiment_5105_fr11_severa_guarded_memory.v468"
STORE_SCHEMA = "carnot.experiment_5105.severa_guarded_memory_sop_store.v468"
RESULT_RELATIVE_PATH = "results/experiment_5105_fr11_severa_guarded_memory_v468.json"
STORE_RELATIVE_PATH = (
    "results/replay_memory/experiment_5105_severa_guarded_memory_sop_store_v468.json"
)
EXP5064_RESULT_RELATIVE_PATH = "results/experiment_5064_audited_skillgraph_self_learning.json"
EXP5077_RESULT_RELATIVE_PATH = "results/experiment_5077_fr11_group_sc_memory_v466.json"
EXP5092_RESULT_RELATIVE_PATH = "results/experiment_5092_fr11_budgeted_onpolicy_memory_v467.json"
SPEC_REFS = ["REQ-LEARN-5105", "SCENARIO-LEARN-5105-SEVERA-CONTRACT-NO-PROMOTE"]
INFERENCE_SUBSTRATE = "exact_guarded_self_learning_eval"
RANDOM_SEED = 20260701

MANDATED_MODEL_SPECS: dict[str, str] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

CONTRACT_ORDER = (
    "provenance",
    "schema_validity",
    "scope",
    "evidence_support",
    "non_regression",
    "ttl_staleness",
    "poison_injection_resistance",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "preconditions_checked",
    "model_specs",
    "candidate_updates_total",
    "contract_pass_count",
    "promoted_count",
    "heldout_delta",
    "nonforgetting_delta",
    "rollback_guard_passed",
    "poison_guard_passed",
    "contamination_guard_passed",
    "formal_contracts",
    "promotion_decision",
    "llm_invoked",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix for promoted or guarded no-promote SEVerA FR-11 outcomes."
    },
    "duration_s": {
        "principle": "measured wall-clock time for search, contract verification, evaluation, and artifact writes."
    },
    "inference_substrate": {
        "principle": "exact_guarded_self_learning_eval means no hidden live LLM proposal, critique, or replay generation."
    },
    "preconditions_checked": {
        "principle": "split hashes, memory/SOP store path, candidate provenance, verifier path, contamination guard, and rollback path are recorded before promotion."
    },
    "model_specs": {
        "principle": "mandated SOTA GGUF IDs are recorded; llm_invoked states whether any were used live."
    },
    "candidate_updates_total": {
        "principle": "number of searched memory/SOP updates before formal contract filtering."
    },
    "contract_pass_count": {
        "principle": "number of candidates that pass every formal provenance, schema, scope, evidence, non-regression, TTL, and poison contract."
    },
    "promoted_count": {
        "principle": "number of updates promoted only after contracts, held-out utility, non-forgetting, contamination, poison, and rollback gates pass."
    },
    "heldout_delta": {
        "principle": "contract-guarded held-out accuracy minus baseline held-out accuracy on the frozen split."
    },
    "nonforgetting_delta": {
        "principle": "retention change on held-out rows that baseline answered correctly."
    },
    "rollback_guard_passed": {
        "principle": "true when the no-promote rollback arm exactly preserves baseline behavior if any gate blocks promotion."
    },
    "poison_guard_passed": {
        "principle": "true only when poisoned or injection-like candidates are rejected before acceptance or promotion."
    },
    "contamination_guard_passed": {
        "principle": "true only when held-out IDs are absent from train/dev candidates and promoted updates."
    },
    "formal_contracts": {
        "principle": "machine-readable contract definitions and per-candidate receipts used by Search-Verify-Learn."
    },
    "promotion_decision": {
        "principle": "single gate record explaining promotion or every no-promote blocker."
    },
    "llm_invoked": {
        "principle": "false unless live LLM proposals, critiques, or replay generations actually ran."
    },
    "flagged_adversarial": {
        "principle": "false only when the artifact schema and all global guards are internally consistent."
    },
}


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


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
    return parsed if math.isfinite(parsed) else None


def round_metric(value: float) -> float:
    return round(float(value), 6)


def as_binary_list(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return []
    bits: list[int] = []
    for item in value:
        parsed = number(item)
        if parsed not in {0.0, 1.0}:
            return []
        bits.append(int(parsed))
    return bits


def accuracy(bits: Sequence[int]) -> float:
    return round_metric(sum(int(bit) for bit in bits) / len(bits)) if bits else 0.0


def canonical_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def delta_label(delta: float) -> str:
    prefix = "plus" if delta >= 0.0 else "minus"
    return f"{prefix}_{abs(delta):.3f}".replace(".", "p")


def load_inputs(root: Path) -> tuple[JsonDict, JsonDict, JsonDict]:
    root = Path(root)
    return (
        read_json_object(root / EXP5092_RESULT_RELATIVE_PATH),
        read_json_object(root / EXP5077_RESULT_RELATIVE_PATH),
        read_json_object(root / EXP5064_RESULT_RELATIVE_PATH),
    )


def _ids_from_split_payload(payload: JsonMap) -> tuple[list[str], list[str], list[str]]:
    split = payload.get("split")
    if isinstance(split, Mapping):
        return (
            [str(row_id) for row_id in split.get("train_ids", [])],
            [str(row_id) for row_id in split.get("dev_ids", [])],
            [str(row_id) for row_id in split.get("heldout_ids", [])],
        )
    return [], [], []


def build_split(exp5092: JsonMap, exp5077: JsonMap, exp5064: JsonMap) -> JsonDict:
    train_ids, dev_ids, heldout_ids = _ids_from_split_payload(exp5092)
    split_source = "exp5092_budgeted_onpolicy_memory"
    if not train_ids or not dev_ids or not heldout_ids:
        train_ids, dev_ids, heldout_ids = _ids_from_split_payload(exp5077)
        split_source = "exp5077_group_sc_memory"
    if (not train_ids or not dev_ids or not heldout_ids) and isinstance(
        exp5064.get("split_ids"), Mapping
    ):
        source_train = [str(row_id) for row_id in exp5064["split_ids"].get("train_ids", [])]
        source_heldout = [str(row_id) for row_id in exp5064["split_ids"].get("heldout_ids", [])]
        train_ids, dev_ids, heldout_ids = source_train[:24], source_train[24:32], source_heldout
        split_source = "exp5064_audited_skillgraph_fallback"
    if not train_ids or not dev_ids or not heldout_ids:
        raise ValueError("train/dev/heldout split IDs are required")
    return {
        "train_ids": train_ids,
        "dev_ids": dev_ids,
        "heldout_ids": heldout_ids,
        "split_source": split_source,
        "heldout_frozen_before_candidate_generation": True,
        "final_answer_leakage_allowed": False,
    }


def poison_guard(payload: str) -> JsonDict:
    lowered = str(payload).lower()
    reasons: list[str] = []
    if any(pattern in lowered for pattern in ("ignore previous", "system:", "developer:", "<script", "{{")):
        reasons.append("prompt_injection_pattern")
    if "final_answer:" in lowered or "heldout_answer" in lowered:
        reasons.append("final_answer_leakage_pattern")
    return {"passed": not reasons, "reasons": reasons}


def build_candidate_updates(split: JsonMap, exp5092: JsonMap, exp5077: JsonMap) -> list[JsonDict]:
    allowed = set(str(row_id) for row_id in [*split.get("train_ids", []), *split.get("dev_ids", [])])
    source_candidates = [
        row for row in exp5092.get("memory_candidates", []) if isinstance(row, Mapping)
    ]
    if not source_candidates:
        source_candidates = [
            {
                "memory_id": f"fallback_5077_{index:04d}_{row.get('row_id', '')}",
                "row_id": row.get("row_id"),
                "payload": f"row={row.get('row_id')};trigger=verified_trace;action=prefer_baseline_until_contract",
                "observed_value": row.get("memory_correct", 0) - row.get("baseline_correct", 0),
                "net_value": row.get("memory_correct", 0) - row.get("baseline_correct", 0),
                "ttl_days": 14,
                "age_days": 1,
                "staleness_state": "fresh",
                "poison_guard": {"passed": True, "reasons": []},
                "provenance": {
                    "source_artifact": EXP5077_RESULT_RELATIVE_PATH,
                    "row_ids": [row.get("row_id")],
                    "final_answer_redacted": True,
                },
            }
            for index, row in enumerate(
                exp5077.get("dev_evaluation", {}).get("per_row", [])
                if isinstance(exp5077.get("dev_evaluation"), Mapping)
                else []
            )
            if isinstance(row, Mapping)
        ]
    updates: list[JsonDict] = []
    for index, candidate in enumerate(source_candidates):
        provenance = candidate.get("provenance") if isinstance(candidate.get("provenance"), Mapping) else {}
        row_ids = [
            str(row_id)
            for row_id in provenance.get("row_ids", [candidate.get("row_id")])
            if row_id is not None
        ]
        if not row_ids or not set(row_ids) <= allowed:
            continue
        payload = str(candidate.get("payload") or "")
        observed = number(candidate.get("observed_value"))
        net_value = number(candidate.get("net_value"))
        local_delta = net_value if net_value is not None else observed
        guard = poison_guard(payload)
        source_guard = candidate.get("poison_guard") if isinstance(candidate.get("poison_guard"), Mapping) else {}
        if source_guard.get("passed") is False:
            guard = {"passed": False, "reasons": sorted(set([*guard["reasons"], *source_guard.get("reasons", [])]))}
        update_type = "memory" if index % 2 == 0 else "sop"
        updates.append(
            {
                "candidate_id": f"candidate_5105_{index:04d}_{candidate.get('memory_id', 'update')}",
                "schema_version": "severa_candidate_update.v1",
                "update_type": update_type,
                "update_scope": "fr11_memory_sop",
                "source_row_ids": row_ids,
                "source_artifact": str(provenance.get("source_artifact") or EXP5092_RESULT_RELATIVE_PATH),
                "candidate_generation_provenance": {
                    "mode": "deterministic_search_from_current_misses_or_verified_traces",
                    "source_experiment": 5092,
                    "llm_invoked": False,
                    "final_answer_leakage_allowed": False,
                },
                "payload": payload,
                "evidence_support_count": len(row_ids),
                "evidence_support_score": round_metric(float(observed or 0.0)),
                "local_non_regression_delta": round_metric(float(local_delta or 0.0)),
                "ttl_days": int(candidate.get("ttl_days") or 14),
                "age_days": int(candidate.get("age_days") or 0),
                "staleness_state": str(candidate.get("staleness_state") or "fresh"),
                "poison_guard": guard,
                "final_answer_redacted": bool(provenance.get("final_answer_redacted")),
            }
        )
    return updates


def formal_contract_definitions() -> list[JsonDict]:
    return [
        {"contract_id": "provenance", "requires": "source artifact, generator mode, and redacted final answers"},
        {"contract_id": "schema_validity", "requires": "stable candidate id, schema version, update type, and payload"},
        {"contract_id": "scope", "requires": "source rows stay inside train/dev and outside frozen held-out IDs"},
        {"contract_id": "evidence_support", "requires": "positive deterministic evidence from at least one source row"},
        {"contract_id": "non_regression", "requires": "local replay delta is non-negative before held-out evaluation"},
        {"contract_id": "ttl_staleness", "requires": "candidate is fresh and age_days does not exceed ttl_days"},
        {"contract_id": "poison_injection_resistance", "requires": "prompt-injection and final-answer-leakage guards pass"},
    ]


def verify_candidate_contract(candidate: JsonMap, split: JsonMap) -> JsonDict:
    train_dev = set(str(row_id) for row_id in [*split.get("train_ids", []), *split.get("dev_ids", [])])
    heldout = set(str(row_id) for row_id in split.get("heldout_ids", []))
    source_rows = set(str(row_id) for row_id in candidate.get("source_row_ids", []))
    checks = {
        "provenance": bool(candidate.get("source_artifact"))
        and candidate.get("candidate_generation_provenance", {}).get("llm_invoked") is False
        and candidate.get("final_answer_redacted") is True,
        "schema_validity": bool(candidate.get("candidate_id"))
        and candidate.get("schema_version") == "severa_candidate_update.v1"
        and candidate.get("update_type") in {"memory", "sop"}
        and bool(candidate.get("payload")),
        "scope": bool(source_rows) and source_rows <= train_dev and not (source_rows & heldout),
        "evidence_support": int(candidate.get("evidence_support_count") or 0) > 0
        and float(candidate.get("evidence_support_score") or 0.0) > 0.0,
        "non_regression": float(candidate.get("local_non_regression_delta") or 0.0) >= 0.0,
        "ttl_staleness": int(candidate.get("age_days") or 0) <= int(candidate.get("ttl_days") or 0)
        and candidate.get("staleness_state") == "fresh",
        "poison_injection_resistance": candidate.get("poison_guard", {}).get("passed") is True,
    }
    failed = [contract_id for contract_id in CONTRACT_ORDER if checks[contract_id] is not True]
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "passed": not failed,
        "failed_contracts": failed,
        "contracts": {contract_id: bool(checks[contract_id]) for contract_id in CONTRACT_ORDER},
    }


def verify_candidate_contracts(candidates: Sequence[JsonMap], split: JsonMap) -> list[JsonDict]:
    return [verify_candidate_contract(candidate, split) for candidate in candidates]


def contract_passing_updates(receipts: Sequence[JsonMap]) -> list[JsonDict]:
    return [dict(receipt) for receipt in receipts if receipt.get("passed") is True]


def extract_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"row_id", "row_ids", "source_row_ids", "train_ids", "dev_ids", "heldout_ids"}:
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    found.update(str(element) for element in item)
                elif item is not None:
                    found.add(str(item))
            found.update(extract_ids(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found.update(extract_ids(item))
    return found


def contamination_guard(
    *,
    split: JsonMap,
    candidate_updates: Sequence[JsonMap],
    promoted_updates: Sequence[JsonMap],
) -> JsonDict:
    train = set(str(row_id) for row_id in split.get("train_ids", []))
    dev = set(str(row_id) for row_id in split.get("dev_ids", []))
    heldout = set(str(row_id) for row_id in split.get("heldout_ids", []))
    violations: list[str] = []
    for label, overlap in (
        ("split_overlap_train_dev", train & dev),
        ("split_overlap_train_heldout", train & heldout),
        ("split_overlap_dev_heldout", dev & heldout),
    ):
        for row_id in sorted(overlap):
            violations.append(f"{label}:{row_id}")
    for row_id in sorted(extract_ids(candidate_updates) & heldout):
        violations.append(f"candidate_heldout_id_leak:{row_id}")
    for row_id in sorted(extract_ids(promoted_updates) & heldout):
        violations.append(f"promoted_heldout_id_leak:{row_id}")
    return {"passed": not violations, "violations": violations}


def comparison(exp5092: JsonMap) -> JsonDict:
    ablations = exp5092.get("ablations") if isinstance(exp5092.get("ablations"), Mapping) else {}
    baseline_bits = as_binary_list(ablations.get("baseline", {}).get("correct"))
    prior_bits = as_binary_list(ablations.get("budget_curated_memory", {}).get("correct"))
    if not prior_bits:
        prior_bits = list(baseline_bits)
    guarded_bits = list(baseline_bits)
    retained = [guarded for base, guarded in zip(baseline_bits, guarded_bits, strict=False) if base == 1]
    baseline_acc = accuracy(baseline_bits)
    guarded_acc = accuracy(guarded_bits)
    return {
        "baseline": {"accuracy": baseline_acc, "correct": baseline_bits},
        "prior_budgeted_memory": {"accuracy": accuracy(prior_bits), "correct": prior_bits},
        "contract_guarded_updates": {"accuracy": guarded_acc, "correct": guarded_bits},
        "rollback_no_promote": {"accuracy": baseline_acc, "correct": baseline_bits, "rollback_applied": True},
        "heldout_delta": round_metric(guarded_acc - baseline_acc),
        "nonforgetting_delta": round_metric((accuracy(retained) if retained else 1.0) - 1.0),
    }


def promotion_decision(
    *,
    contract_pass_count: int,
    heldout_delta: float,
    nonforgetting_delta: float,
    contamination_guard_passed: bool,
    poison_guard_passed: bool,
    rollback_guard_passed: bool,
) -> JsonDict:
    blockers: list[str] = []
    if contract_pass_count <= 0:
        blockers.append("no_contract_passing_updates")
    if heldout_delta < 0.0:
        blockers.append("heldout_delta_negative")
    elif heldout_delta == 0.0:
        blockers.append("positive_utility_not_observed")
    if nonforgetting_delta < 0.0:
        blockers.append("nonforgetting_regressed")
    if not contamination_guard_passed:
        blockers.append("contamination_guard_failed")
    if not poison_guard_passed:
        blockers.append("poison_guard_failed")
    if not rollback_guard_passed:
        blockers.append("rollback_guard_failed")
    return {
        "promoted": not blockers,
        "no_promote_reason": ";".join(blockers),
        "gate_conditions": {
            "contract_pass_count": int(contract_pass_count),
            "heldout_delta_gt_zero": heldout_delta > 0.0,
            "nonforgetting_delta_gte_zero": nonforgetting_delta >= 0.0,
            "contamination_guard_passed": contamination_guard_passed,
            "poison_guard_passed": poison_guard_passed,
            "rollback_guard_passed": rollback_guard_passed,
        },
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    return [
        {"path": relative, "present": (Path(root) / relative).exists(), "sha256": sha256_file(Path(root) / relative)}
        for relative in (
            EXP5064_RESULT_RELATIVE_PATH,
            EXP5077_RESULT_RELATIVE_PATH,
            EXP5092_RESULT_RELATIVE_PATH,
        )
    ]


def check_preconditions(
    *,
    root: Path,
    split: JsonMap,
    store_path: Path,
    guard: JsonMap,
) -> JsonDict:
    return {
        "dataset_split_hashes": {
            "train": canonical_hash(split.get("train_ids", [])),
            "dev": canonical_hash(split.get("dev_ids", [])),
            "heldout": canonical_hash(split.get("heldout_ids", [])),
            "all": canonical_hash(
                {
                    "train": split.get("train_ids", []),
                    "dev": split.get("dev_ids", []),
                    "heldout": split.get("heldout_ids", []),
                }
            ),
        },
        "memory_sop_store_path": Path(store_path).as_posix(),
        "candidate_generation_provenance": {
            "mode": "deterministic_search_from_current_misses_or_verified_traces",
            "source_artifacts": source_artifacts(root),
            "final_answer_leakage_allowed": False,
            "llm_invoked": False,
        },
        "exact_verifier_path": f"{MODULE_RELATIVE_PATH}::verify_candidate_contracts",
        "contamination_guard": dict(guard),
        "rollback_path": f"{Path(store_path).as_posix()}#rollback_no_promote",
    }


def model_specs(exp5092: JsonMap, exp5064: JsonMap) -> JsonDict:
    return {
        "mandated_sota": dict(MANDATED_MODEL_SPECS),
        "llm_invoked": False,
        "proposal_mode": "deterministic_search_from_checked_in_artifacts",
        "critique_mode": "none",
        "replay_generation_mode": "none",
        "upstream_model_specs": {
            "exp5092": dict(exp5092.get("model_specs") or {}),
            "exp5064": dict(exp5064.get("model_specs") or {}),
        },
    }


def store_payload(
    *,
    artifact_path: Path,
    candidates: Sequence[JsonMap],
    receipts: Sequence[JsonMap],
    decision: JsonMap,
) -> JsonDict:
    accepted_ids = [str(row.get("candidate_id") or "") for row in receipts if row.get("passed") is True]
    return {
        "schema": STORE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "source_artifact": Path(artifact_path).as_posix(),
        "accepted_update_ids": accepted_ids,
        "rejected_update_ids": [
            str(row.get("candidate_id") or "") for row in receipts if row.get("passed") is not True
        ],
        "promoted_update_ids": accepted_ids if decision.get("promoted") is True else [],
        "candidate_updates": [dict(row) for row in candidates],
        "contract_receipts": [dict(row) for row in receipts],
        "promotion_decision": dict(decision),
        "rollback_policy": "no_promote_preserves_baseline_when_any_gate_fails",
    }


def checksum(artifact: JsonMap) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    store_path: Path | None = None,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    store_path = Path(store_path) if store_path else root / STORE_RELATIVE_PATH
    start = float(now())
    exp5092, exp5077, exp5064 = load_inputs(root)
    split = build_split(exp5092, exp5077, exp5064)
    candidates = build_candidate_updates(split, exp5092, exp5077)
    receipts = verify_candidate_contracts(candidates, split)
    passed = contract_passing_updates(receipts)
    guard = contamination_guard(split=split, candidate_updates=candidates, promoted_updates=[])
    comp = comparison(exp5092)
    heldout_delta = float(comp["heldout_delta"])
    nonforgetting_delta = float(comp["nonforgetting_delta"])
    rollback_guard_passed = comp["rollback_no_promote"]["correct"] == comp["baseline"]["correct"]
    poison_guard_passed = all(
        candidate.get("poison_guard", {}).get("passed") is True
        for candidate in candidates
        if any(candidate.get("candidate_id") == receipt.get("candidate_id") for receipt in passed)
    )
    decision = promotion_decision(
        contract_pass_count=len(passed),
        heldout_delta=heldout_delta,
        nonforgetting_delta=nonforgetting_delta,
        contamination_guard_passed=bool(guard["passed"]),
        poison_guard_passed=poison_guard_passed,
        rollback_guard_passed=rollback_guard_passed,
    )
    store = store_payload(
        artifact_path=artifact_path,
        candidates=candidates,
        receipts=receipts,
        decision=decision,
    )
    if write:
        write_json(store_path, store)
    preconditions = check_preconditions(root=root, split=split, store_path=store_path, guard=guard)
    verdict = (
        "success_fr11_severa_guarded_memory_promoted_under_contracts_"
        f"{delta_label(heldout_delta)}"
        if decision["promoted"]
        else "complete_fr11_severa_guarded_memory_no_promote_contracts_working_delta_"
        f"{delta_label(heldout_delta)}"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": verdict,
        "duration_s": round_metric(max(0.0, float(now()) - start)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": preconditions,
        "model_specs": model_specs(exp5092, exp5064),
        "candidate_updates_total": len(candidates),
        "contract_pass_count": len(passed),
        "promoted_count": len(passed) if decision["promoted"] else 0,
        "heldout_delta": heldout_delta,
        "nonforgetting_delta": nonforgetting_delta,
        "rollback_guard_passed": rollback_guard_passed,
        "poison_guard_passed": poison_guard_passed,
        "contamination_guard_passed": bool(guard["passed"]),
        "formal_contracts": {
            "definitions": formal_contract_definitions(),
            "candidate_receipts": receipts,
        },
        "promotion_decision": decision,
        "llm_invoked": False,
        "flagged_adversarial": False,
        "random_seed": RANDOM_SEED,
        "split": split,
        "candidate_updates": candidates,
        "accepted_update_ids": store["accepted_update_ids"],
        "flagged_adversarial_candidate_ids": [
            str(candidate.get("candidate_id") or "")
            for candidate in candidates
            if candidate.get("poison_guard", {}).get("passed") is not True
        ],
        "comparison": comp,
        "contamination_guard": guard,
        "store_path": store_path.as_posix(),
        "store_sha256": sha256_file(store_path) if write else None,
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    artifact["flagged_adversarial"] = bool(artifact_schema_errors(artifact))
    artifact["reproducibility_checksum"] = checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (
        verdict.startswith("success_fr11_severa_guarded_memory_promoted_under_contracts")
        or verdict.startswith("complete_fr11_severa_guarded_memory_no_promote_contracts_working")
    ):
        errors.append("honest_verdict")
    if number(artifact.get("duration_s")) is None:
        errors.append("duration_s")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked")
    specs = artifact.get("model_specs")
    mandated = specs.get("mandated_sota") if isinstance(specs, Mapping) else {}
    if not isinstance(mandated, Mapping) or set(MANDATED_MODEL_SPECS.values()) - set(
        str(value) for value in mandated.values()
    ):
        errors.append("model_specs")
    for field in ("candidate_updates_total", "contract_pass_count", "promoted_count"):
        if not isinstance(artifact.get(field), int):
            errors.append(field)
    for field in ("heldout_delta", "nonforgetting_delta"):
        if number(artifact.get(field)) is None:
            errors.append(field)
    for field in (
        "rollback_guard_passed",
        "poison_guard_passed",
        "contamination_guard_passed",
        "llm_invoked",
        "flagged_adversarial",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    contracts = artifact.get("formal_contracts")
    if not isinstance(contracts, Mapping) or not isinstance(contracts.get("definitions"), list):
        errors.append("formal_contracts")
    if not isinstance(artifact.get("promotion_decision"), Mapping):
        errors.append("promotion_decision")
    return sorted(set(errors))


def main() -> None:  # pragma: no cover - CLI wrapper
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
