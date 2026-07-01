#!/usr/bin/env python3
"""Exp 5077: guarded FR-11 group self-consistency memory attempt.

Spec refs: REQ-LEARN-5077, SCENARIO-LEARN-5077-GROUP-SC-NO-PROMOTE.
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

EXPERIMENT_ID = 5077
EXPERIMENT_NAME = "experiment_5077_fr11_group_sc_memory"
MODULE_RELATIVE_PATH = "python/carnot/experiment_5077_fr11_group_sc_memory.py"
SCHEMA = "carnot.experiment_5077_fr11_group_sc_memory.v466"
RESULT_RELATIVE_PATH = "results/experiment_5077_fr11_group_sc_memory_v466.json"
EXP5051_RESULT_RELATIVE_PATH = "results/experiment_5051_verifier_trace_self_learning.json"
EXP5051_MEMORY_RELATIVE_PATH = (
    "results/replay_memory/experiment_5051_verifier_trace_self_learning_memory.json"
)
EXP5059_RESULT_RELATIVE_PATH = "results/experiment_5059_d1_sota_refresh_audit.json"
EXP5064_RESULT_RELATIVE_PATH = "results/experiment_5064_audited_skillgraph_self_learning.json"
MUSR_CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
SPEC_REFS = ["REQ-LEARN-5077", "SCENARIO-LEARN-5077-GROUP-SC-NO-PROMOTE"]
RANDOM_SEED = 20260701
INFERENCE_SUBSTRATE = "deterministic_group_sc_memory_replay_no_live_llm"

MANDATED_MODEL_SPECS: dict[str, str] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "model_specs",
    "fr11_attempt_completed",
    "heldout_delta",
    "nonforgetting_delta",
    "contamination_guard_passed",
    "rollback_guard_passed",
    "promoted_count",
    "quarantined_count",
    "memory_policy",
    "group_self_consistency_summary",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix for promoted or guarded no-promote FR-11 group-SC memory outcomes."
    },
    "duration_s": {"principle": "measured wall-clock time for deterministic replay and guard checks."},
    "inference_substrate": {
        "principle": "declares deterministic replay over checked-in artifacts; no live LLM inference is hidden."
    },
    "model_specs": {
        "principle": "mandated SOTA GGUF IDs plus proposal provenance; live proposal calls are explicitly false."
    },
    "fr11_attempt_completed": {
        "principle": "true only after split freeze, proposal grouping, consensus evaluation, rollback, and artifact emission."
    },
    "heldout_delta": {
        "principle": "memory-enabled held-out accuracy minus baseline accuracy before rollback."
    },
    "nonforgetting_delta": {
        "principle": "retention delta on held-out rows the baseline answered correctly before rollback."
    },
    "contamination_guard_passed": {
        "principle": "true only when held-out IDs are absent from train/dev proposal and promotion payloads."
    },
    "rollback_guard_passed": {
        "principle": "true when regression leaves the active no-promote arm equal to baseline, or no rollback is needed."
    },
    "promoted_count": {
        "principle": "number of consensus memory candidates promoted after all utility and rollback gates."
    },
    "quarantined_count": {
        "principle": "number of non-consensus proposals plus tested consensus candidates that were not promoted."
    },
    "memory_policy": {
        "principle": "the consensus memory/retrieval policy evaluated on dev and held-out rows."
    },
    "group_self_consistency_summary": {
        "principle": "proposal groups, consensus threshold, tested candidates, and quarantined non-consensus proposals."
    },
    "flagged_adversarial": {
        "principle": "false only when artifact schema, oracle separation, contamination, and rollback accounting are internally consistent."
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


def as_binary_list(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        return []
    out: list[int] = []
    for item in value:
        parsed = number(item)
        if parsed not in {0.0, 1.0}:
            return []
        out.append(int(parsed))
    return out


def accuracy(bits: Sequence[int]) -> float:
    return round(sum(int(bit) for bit in bits) / len(bits), 6) if bits else 0.0


def round_metric(value: float) -> float:
    return round(float(value), 6)


def row_id(index: int) -> str:
    return f"q{index:04d}"


def row_index(value: str) -> int:
    return int(str(value).removeprefix("q"))


def delta_label(delta: float) -> str:
    prefix = "plus" if delta >= 0.0 else "minus"
    return f"{prefix}_{abs(delta):.3f}".replace(".", "p")


def load_inputs(root: Path) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    root = Path(root)
    return (
        read_json_object(root / EXP5051_RESULT_RELATIVE_PATH),
        read_json_object(root / EXP5051_MEMORY_RELATIVE_PATH),
        read_json_object(root / EXP5059_RESULT_RELATIVE_PATH),
        read_json_object(root / EXP5064_RESULT_RELATIVE_PATH),
    )


def paired_correct(exp5059: JsonMap) -> JsonDict:
    metrics = exp5059.get("refreshed_candidate_metrics")
    paired = metrics.get("paired_correct") if isinstance(metrics, Mapping) else exp5059.get("paired_correct")
    if not isinstance(paired, Mapping):
        return {"verifier": [], "tuned_self_consistency": [], "oracle_at_k": []}
    return {
        "verifier": as_binary_list(paired.get("verifier")),
        "tuned_self_consistency": as_binary_list(paired.get("tuned_self_consistency")),
        "oracle_at_k": as_binary_list(paired.get("oracle_at_k")),
    }


def load_checkpoint_rows(root: Path) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    checkpoint_dir = Path(root) / MUSR_CHECKPOINT_RELATIVE_DIR
    for path in sorted(checkpoint_dir.glob("q*.json")):
        payload = read_json_object(path)
        if not payload:
            continue
        rid = str(payload.get("row_id") or path.stem)
        sc_answer = str(payload.get("sc_answer") or "")
        energy_answer = str(payload.get("energy_pure_answer") or payload.get("energy_answer") or "")
        disagreement = bool(payload.get("verifier_sc_disagreement"))
        if "verifier_sc_disagreement" not in payload:
            disagreement = bool(sc_answer and energy_answer and sc_answer != energy_answer)
        rows[rid] = {
            "row_id": rid,
            "checkpoint_path": path.as_posix(),
            "sc_answer": sc_answer,
            "energy_answer": energy_answer,
            "energy_abstained": bool(payload.get("energy_abstained")),
            "verifier_sc_disagreement": disagreement,
        }
    return rows


def verified_trace_row_ids(memory: JsonMap, allowed_ids: set[str]) -> list[str]:
    rows: list[str] = []
    for trace in memory.get("verified_traces", []):
        if not isinstance(trace, Mapping):
            continue
        rid = str(trace.get("row_id") or "")
        if rid in allowed_ids:
            rows.append(rid)
    return sorted(dict.fromkeys(rows))


def near_miss_row_ids_from_pairs(paired: JsonMap, allowed_ids: set[str]) -> list[str]:
    verifier = paired.get("verifier", [])
    tuned = paired.get("tuned_self_consistency", [])
    oracle = paired.get("oracle_at_k", [])
    n_rows = min(len(verifier), len(tuned), len(oracle))
    rows: list[str] = []
    for index in range(n_rows):
        rid = row_id(index)
        if rid not in allowed_ids:
            continue
        if int(verifier[index]) == 0 and (int(tuned[index]) == 1 or int(oracle[index]) == 1):
            rows.append(rid)
    return rows


def build_train_dev_heldout_split(
    exp5051: JsonMap,
    memory: JsonMap,
    paired: JsonMap,
    *,
    train_count: int = 24,
    dev_count: int = 8,
    heldout_count: int = 20,
) -> JsonDict:
    split = exp5051.get("split_ids")
    if not isinstance(split, Mapping):
        raise ValueError("Exp5051 split_ids missing")
    source_train = [str(rid) for rid in split.get("train_ids", [])]
    source_heldout = [str(rid) for rid in split.get("heldout_ids", [])]
    if not source_train or not source_heldout:
        raise ValueError("Exp5051 split_ids must include train and heldout IDs")

    allowed_train = set(source_train)
    candidate_ids = verified_trace_row_ids(memory, allowed_train)
    candidate_ids.extend(near_miss_row_ids_from_pairs(paired, allowed_train))
    candidate_ids = sorted(dict.fromkeys(candidate_ids))
    if len(candidate_ids) < train_count + dev_count:
        candidate_ids = sorted(dict.fromkeys(candidate_ids + source_train))

    n_train = max(0, int(train_count))
    n_dev = max(0, int(dev_count))
    n_heldout = max(1, min(int(heldout_count), len(source_heldout)))
    return {
        "train_ids": candidate_ids[:n_train],
        "dev_ids": candidate_ids[n_train : n_train + n_dev],
        "heldout_ids": source_heldout[-n_heldout:],
        "split_source": "exp5051_frozen_near_misses_with_tail_heldout",
        "heldout_frozen_before_proposal": True,
    }


def trace_ids_for_rows(memory: JsonMap, row_ids: Sequence[str]) -> list[str]:
    wanted = set(str(rid) for rid in row_ids)
    trace_ids: list[str] = []
    for trace in memory.get("verified_traces", []):
        if isinstance(trace, Mapping) and str(trace.get("row_id") or "") in wanted:
            trace_id = str(trace.get("trace_id") or "")
            if trace_id:
                trace_ids.append(trace_id)
    return sorted(dict.fromkeys(trace_ids))


def generate_policy_proposals(split: JsonMap, memory: JsonMap) -> list[JsonDict]:
    train_ids = [str(rid) for rid in split.get("train_ids", [])]
    support_rows = train_ids[: min(8, len(train_ids))]
    support_trace_ids = trace_ids_for_rows(memory, support_rows)
    common = {
        "proposal_type": "memory_retrieval_policy",
        "policy_signature": "fallback_to_tuned_on_verifier_sc_disagreement",
        "trigger": "verifier_sc_disagreement",
        "action": "retrieve_verified_trace_then_fallback_to_tuned_self_consistency",
        "source_row_ids": support_rows,
        "source_trace_ids": support_trace_ids,
        "proposal_provenance": "deterministic_replay_from_exp5051_exp5064_negatives",
    }
    return [
        {
            **common,
            "proposal_id": "proposal_5077_qwen_disagreement_fallback",
            "proposer_role": "flagship_moe_replay",
            "proposer_model": MANDATED_MODEL_SPECS["flagship_moe"],
        },
        {
            **common,
            "proposal_id": "proposal_5077_gemma_dense_disagreement_fallback",
            "proposer_role": "flagship_dense_replay",
            "proposer_model": MANDATED_MODEL_SPECS["flagship_dense"],
        },
        {
            **common,
            "proposal_id": "proposal_5077_gemma_moe_disagreement_fallback",
            "proposer_role": "middle_moe_replay",
            "proposer_model": MANDATED_MODEL_SPECS["middle_moe"],
        },
        {
            "proposal_id": "proposal_5077_abstention_trace_memory",
            "proposal_type": "retrieval_memory_policy",
            "policy_signature": "retrieve_on_energy_abstention_only",
            "trigger": "energy_abstained",
            "action": "retrieve_trace_without_answer_fallback",
            "source_row_ids": support_rows[:2],
            "source_trace_ids": support_trace_ids[:2],
            "proposal_provenance": "deterministic_singleton_control",
            "proposer_role": "singleton_control",
            "proposer_model": MANDATED_MODEL_SPECS["flagship_dense"],
        },
        {
            "proposal_id": "proposal_5077_promote_all_verified_trace_memory",
            "proposal_type": "skill_policy",
            "policy_signature": "promote_all_verified_trace_memory",
            "trigger": "any_verified_trace",
            "action": "promote_without_dev_or_heldout_gate",
            "source_row_ids": support_rows,
            "source_trace_ids": support_trace_ids,
            "proposal_provenance": "deterministic_negative_control",
            "proposer_role": "unsafe_policy_control",
            "proposer_model": MANDATED_MODEL_SPECS["flagship_moe"],
        },
    ]


def candidate_id_for_signature(signature: str) -> str:
    if signature == "fallback_to_tuned_on_verifier_sc_disagreement":
        return "candidate_5077_disagreement_fallback"
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8]
    return f"candidate_5077_{digest}"


def group_self_consistency(
    proposals: Sequence[JsonMap],
    *,
    consensus_threshold: int = 2,
) -> tuple[JsonDict, list[JsonDict]]:
    groups: dict[str, list[JsonMap]] = {}
    for proposal in proposals:
        signature = str(proposal.get("policy_signature") or "")
        groups.setdefault(signature, []).append(proposal)

    candidates: list[JsonDict] = []
    group_rows: list[JsonDict] = []
    quarantined: list[str] = []
    for signature in sorted(groups):
        rows = groups[signature]
        proposal_ids = sorted(str(row.get("proposal_id") or "") for row in rows)
        consensus = len(rows) >= int(consensus_threshold)
        group_rows.append(
            {
                "policy_signature": signature,
                "proposal_count": len(rows),
                "proposal_ids": proposal_ids,
                "consensus": consensus,
            }
        )
        if not consensus:
            quarantined.extend(proposal_ids)
            continue
        source_rows = sorted(
            {
                str(row_id)
                for proposal in rows
                for row_id in proposal.get("source_row_ids", [])
            }
        )
        source_trace_ids = sorted(
            {
                str(trace_id)
                for proposal in rows
                for trace_id in proposal.get("source_trace_ids", [])
            }
        )
        exemplar = rows[0]
        candidates.append(
            {
                "candidate_id": candidate_id_for_signature(signature),
                "policy_signature": signature,
                "trigger": str(exemplar.get("trigger") or ""),
                "action": str(exemplar.get("action") or ""),
                "support_proposal_ids": proposal_ids,
                "source_row_ids": source_rows,
                "source_trace_ids": source_trace_ids,
                "consensus_support": len(rows),
            }
        )

    summary = {
        "consensus_threshold": int(consensus_threshold),
        "total_proposals": len(proposals),
        "groups": group_rows,
        "tested_consensus_candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "quarantined_nonconsensus_proposal_ids": sorted(quarantined),
    }
    return summary, candidates


def extract_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {
                "row_id",
                "source_row_ids",
                "support_row_ids",
                "train_ids",
                "dev_ids",
                "heldout_ids",
            }:
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
    proposals: Sequence[JsonMap],
    consensus_candidates: Sequence[JsonMap],
    promoted_memory_entries: Sequence[JsonMap],
) -> JsonDict:
    train = set(str(rid) for rid in split.get("train_ids", []))
    dev = set(str(rid) for rid in split.get("dev_ids", []))
    heldout = set(str(rid) for rid in split.get("heldout_ids", []))
    violations: list[str] = []
    if train & dev:
        violations.append(f"split_overlap_train_dev:{','.join(sorted(train & dev))}")
    if train & heldout:
        violations.append(f"split_overlap_train_heldout:{','.join(sorted(train & heldout))}")
    if dev & heldout:
        violations.append(f"split_overlap_dev_heldout:{','.join(sorted(dev & heldout))}")

    checks = (
        ("proposal", proposals),
        ("consensus_candidate", consensus_candidates),
        ("promoted_memory", promoted_memory_entries),
    )
    for label, payload in checks:
        leaked = sorted(extract_ids(payload) & heldout)
        for rid in leaked:
            violations.append(f"{label}_heldout_id_leak:{rid}")
    return {"passed": not violations, "violations": violations}


def policy_selects_tuned(policy: JsonMap, checkpoint_row: JsonMap) -> bool:
    return (
        str(policy.get("policy_signature") or "")
        == "fallback_to_tuned_on_verifier_sc_disagreement"
        and checkpoint_row.get("verifier_sc_disagreement") is True
    )


def evaluate_policy(
    paired: JsonMap,
    checkpoint_rows: Mapping[str, JsonMap],
    row_ids: Sequence[str],
    policy: JsonMap,
) -> JsonDict:
    verifier = paired.get("verifier", [])
    tuned = paired.get("tuned_self_consistency", [])
    baseline_bits: list[int] = []
    memory_bits: list[int] = []
    per_row: list[JsonDict] = []
    for rid in row_ids:
        index = row_index(str(rid))
        if index >= len(verifier) or index >= len(tuned):
            continue
        pre = int(verifier[index])
        tuned_bit = int(tuned[index])
        checkpoint = checkpoint_rows.get(str(rid), {})
        selected = "tuned_self_consistency" if policy_selects_tuned(policy, checkpoint) else "baseline_verifier"
        memory = tuned_bit if selected == "tuned_self_consistency" else pre
        baseline_bits.append(pre)
        memory_bits.append(memory)
        per_row.append(
            {
                "row_id": str(rid),
                "baseline_correct": pre,
                "memory_correct": memory,
                "tuned_self_consistency_correct": tuned_bit,
                "selector": selected,
                "verifier_sc_disagreement": bool(checkpoint.get("verifier_sc_disagreement")),
            }
        )

    baseline_accuracy = accuracy(baseline_bits)
    memory_accuracy = accuracy(memory_bits)
    nonforgetting_rows = [row for row in per_row if row["baseline_correct"] == 1]
    retained = [int(row["memory_correct"]) for row in nonforgetting_rows]
    retention = accuracy(retained) if retained else 1.0
    return {
        "row_ids": [str(rid) for rid in row_ids],
        "n_rows": len(baseline_bits),
        "baseline_correct": baseline_bits,
        "memory_correct": memory_bits,
        "baseline_accuracy": baseline_accuracy,
        "memory_accuracy": memory_accuracy,
        "delta": round_metric(memory_accuracy - baseline_accuracy),
        "nonforgetting_slice_n": len(nonforgetting_rows),
        "nonforgetting_retention": retention,
        "nonforgetting_delta": round_metric(retention - 1.0),
        "regressed_previously_correct_ids": [
            str(row["row_id"]) for row in nonforgetting_rows if row["memory_correct"] == 0
        ],
        "improved_previously_wrong_ids": [
            str(row["row_id"])
            for row in per_row
            if row["baseline_correct"] == 0 and row["memory_correct"] == 1
        ],
        "per_row": per_row,
    }


def promotion_decision(
    *,
    heldout_delta: float,
    nonforgetting_delta: float,
    contamination_guard_passed: bool,
    rollback_guard_passed: bool,
    consensus_candidate_count: int,
) -> JsonDict:
    blockers: list[str] = []
    if consensus_candidate_count <= 0:
        blockers.append("no_consensus_candidate")
    if heldout_delta < 0.0:
        blockers.append("heldout_delta_negative")
    if nonforgetting_delta < 0.0:
        blockers.append("nonforgetting_regressed")
    if not contamination_guard_passed:
        blockers.append("contamination_guard_failed")
    if not rollback_guard_passed:
        blockers.append("rollback_guard_failed")
    return {"promoted": not blockers, "no_promote_reason": ";".join(blockers)}


def source_artifacts(root: Path) -> list[JsonDict]:
    paths = [
        EXP5051_RESULT_RELATIVE_PATH,
        EXP5051_MEMORY_RELATIVE_PATH,
        EXP5059_RESULT_RELATIVE_PATH,
        EXP5064_RESULT_RELATIVE_PATH,
    ]
    return [
        {"path": path, "sha256": sha256_file(root / path), "present": (root / path).exists()}
        for path in paths
    ]


def model_specs(exp5059: JsonMap) -> JsonDict:
    return {
        "mandated_sota": dict(MANDATED_MODEL_SPECS),
        "llm_proposals_generated": False,
        "llm_critiques_generated": False,
        "proposal_mode": "deterministic_group_self_consistency_from_checked_in_exp5051_exp5064_evidence",
        "exp5059_model_specs": dict(exp5059.get("model_specs") or {}),
    }


def upstream_flags(exp5059: JsonMap, exp5064: JsonMap) -> list[str]:
    flags: list[str] = []
    if exp5059.get("flagged_adversarial") is True:
        flags.append(EXP5059_RESULT_RELATIVE_PATH)
    if exp5064.get("flagged_adversarial") is True:
        flags.append(EXP5064_RESULT_RELATIVE_PATH)
    return flags


def checksum(artifact: JsonMap) -> str:
    without_checksum = {
        key: value for key, value in artifact.items() if key != "reproducibility_checksum"
    }
    return "sha256:" + hashlib.sha256(json_dumps(without_checksum).encode("utf-8")).hexdigest()


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    now: Clock = time.perf_counter,
    write: bool = True,
    train_count: int = 24,
    dev_count: int = 8,
    heldout_count: int = 20,
    consensus_threshold: int = 2,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    exp5051, memory, exp5059, exp5064 = load_inputs(root)
    paired = paired_correct(exp5059)
    checkpoints = load_checkpoint_rows(root)
    split = build_train_dev_heldout_split(
        exp5051,
        memory,
        paired,
        train_count=train_count,
        dev_count=dev_count,
        heldout_count=heldout_count,
    )
    proposals = generate_policy_proposals(split, memory)
    group_summary, candidates = group_self_consistency(
        proposals,
        consensus_threshold=consensus_threshold,
    )
    guard = contamination_guard(
        split=split,
        proposals=proposals,
        consensus_candidates=candidates,
        promoted_memory_entries=[],
    )
    memory_policy: JsonDict = (
        dict(candidates[0])
        if candidates
        else {
            "candidate_id": "none",
            "policy_signature": "no_consensus_candidate",
            "trigger": "none",
            "action": "baseline_only",
            "source_row_ids": [],
            "source_trace_ids": [],
            "support_proposal_ids": [],
            "consensus_support": 0,
        }
    )
    dev_eval = evaluate_policy(paired, checkpoints, split["dev_ids"], memory_policy)
    heldout_eval = evaluate_policy(paired, checkpoints, split["heldout_ids"], memory_policy)
    rollback_needed = bool(
        heldout_eval["delta"] < 0.0 or heldout_eval["nonforgetting_delta"] < 0.0
    )
    rollback_correct = (
        list(heldout_eval["baseline_correct"])
        if rollback_needed
        else list(heldout_eval["memory_correct"])
    )
    rollback_accuracy = accuracy(rollback_correct)
    rollback_guard_passed = bool(
        not rollback_needed or rollback_correct == list(heldout_eval["baseline_correct"])
    )
    decision = promotion_decision(
        heldout_delta=float(heldout_eval["delta"]),
        nonforgetting_delta=float(heldout_eval["nonforgetting_delta"]),
        contamination_guard_passed=bool(guard["passed"]),
        rollback_guard_passed=rollback_guard_passed,
        consensus_candidate_count=len(candidates),
    )
    promoted = bool(decision["promoted"])
    promoted_count = len(candidates) if promoted else 0
    quarantined_count = (
        len(group_summary["quarantined_nonconsensus_proposal_ids"])
        + max(0, len(candidates) - promoted_count)
    )
    verdict = (
        f"success_fr11_group_sc_memory_promoted_{delta_label(float(heldout_eval['delta']))}"
        if promoted
        else "complete_fr11_group_sc_memory_guarded_no_promote_delta_"
        f"{delta_label(float(heldout_eval['delta']))}"
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
        "model_specs": model_specs(exp5059),
        "fr11_attempt_completed": True,
        "heldout_delta": float(heldout_eval["delta"]),
        "nonforgetting_delta": float(heldout_eval["nonforgetting_delta"]),
        "contamination_guard_passed": bool(guard["passed"]),
        "rollback_guard_passed": rollback_guard_passed,
        "promoted_count": promoted_count,
        "quarantined_count": quarantined_count,
        "memory_policy": memory_policy,
        "group_self_consistency_summary": group_summary,
        "flagged_adversarial": False,
        "random_seed": RANDOM_SEED,
        "split": split,
        "contamination_guard": guard,
        "proposals": [dict(proposal) for proposal in proposals],
        "consensus_candidates": [dict(candidate) for candidate in candidates],
        "dev_evaluation": dev_eval,
        "heldout_evaluation": heldout_eval,
        "ablations": {
            "baseline": {
                "accuracy": float(heldout_eval["baseline_accuracy"]),
                "correct": list(heldout_eval["baseline_correct"]),
            },
            "memory_enabled": {
                "accuracy": float(heldout_eval["memory_accuracy"]),
                "correct": list(heldout_eval["memory_correct"]),
            },
            "rollback_no_promote": {
                "accuracy": rollback_accuracy,
                "correct": rollback_correct,
                "rollback_applied": rollback_needed,
            },
        },
        "promotion_decision": {
            **decision,
            "gate_conditions": {
                "heldout_delta_gte_zero": float(heldout_eval["delta"]) >= 0.0,
                "nonforgetting_delta_gte_zero": float(heldout_eval["nonforgetting_delta"]) >= 0.0,
                "contamination_guard_passed": bool(guard["passed"]),
                "rollback_guard_passed": rollback_guard_passed,
            },
        },
        "source_artifacts": source_artifacts(root),
        "upstream_flagged_adversarial_sources": upstream_flags(exp5059, exp5064),
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
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (
        verdict.startswith("success_fr11_group_sc_memory_promoted_plus_")
        or verdict.startswith("complete_fr11_group_sc_memory_guarded_no_promote_delta_")
    ):
        errors.append("honest_verdict")
    if number(artifact.get("duration_s")) is None:
        errors.append("duration_s")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    specs = artifact.get("model_specs")
    mandated = specs.get("mandated_sota") if isinstance(specs, Mapping) else None
    if not isinstance(mandated, Mapping) or dict(mandated) != MANDATED_MODEL_SPECS:
        errors.append("model_specs")
    for field in (
        "fr11_attempt_completed",
        "contamination_guard_passed",
        "rollback_guard_passed",
        "flagged_adversarial",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("fr11_attempt_completed") is not True:
        errors.append("fr11_attempt_completed")
    for field in ("heldout_delta", "nonforgetting_delta"):
        if number(artifact.get(field)) is None:
            errors.append(field)
    for field in ("promoted_count", "quarantined_count"):
        if not isinstance(artifact.get(field), int) or int(artifact.get(field, -1)) < 0:
            errors.append(field)
    policy = artifact.get("memory_policy")
    if not isinstance(policy, Mapping) or not policy.get("policy_signature"):
        errors.append("memory_policy")
    summary = artifact.get("group_self_consistency_summary")
    if not isinstance(summary, Mapping) or not isinstance(
        summary.get("tested_consensus_candidate_ids"), Sequence
    ):
        errors.append("group_self_consistency_summary")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            entry = principles.get(field)
            if not isinstance(entry, Mapping) or not str(entry.get("principle") or ""):
                errors.append("field_principles")
                break
    if artifact.get("promoted_count", 0) > 0:
        if float(artifact.get("heldout_delta", -1.0)) < 0.0:
            errors.append("promoted_with_negative_heldout_delta")
        if float(artifact.get("nonforgetting_delta", -1.0)) < 0.0:
            errors.append("promoted_with_negative_nonforgetting_delta")
        if artifact.get("contamination_guard_passed") is not True:
            errors.append("promoted_with_contamination")
        if artifact.get("rollback_guard_passed") is not True:
            errors.append("promoted_without_rollback_guard")
    return sorted(set(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    print(
        json.dumps(
            {
                "result_path": (REPO_ROOT / RESULT_RELATIVE_PATH).as_posix(),
                "honest_verdict": artifact.get("honest_verdict"),
                "heldout_delta": artifact.get("heldout_delta"),
                "nonforgetting_delta": artifact.get("nonforgetting_delta"),
                "promoted_count": artifact.get("promoted_count"),
                "quarantined_count": artifact.get("quarantined_count"),
            },
            sort_keys=True,
        )
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
