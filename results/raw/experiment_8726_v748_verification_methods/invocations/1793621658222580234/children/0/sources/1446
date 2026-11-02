#!/usr/bin/env python3
"""Exp 5064: audited skill-graph self-learning with no-promote gates.

Spec refs: REQ-VERIFY-5064, SCENARIO-VERIFY-5064.
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

EXPERIMENT_ID = 5064
EXPERIMENT_NAME = "experiment_5064_audited_skillgraph_self_learning"
SCHEMA = "carnot.experiment_5064_audited_skillgraph_self_learning.v1"
SKILL_GRAPH_SCHEMA = "carnot.experiment_5064.skill_graph.v1"
RESULT_RELATIVE_PATH = "results/experiment_5064_audited_skillgraph_self_learning.json"
SKILL_GRAPH_RELATIVE_PATH = (
    "results/replay_memory/experiment_5064_audited_skillgraph_self_learning_skill_graph.json"
)
EXP5051_RESULT_RELATIVE_PATH = "results/experiment_5051_verifier_trace_self_learning.json"
EXP5051_MEMORY_RELATIVE_PATH = (
    "results/replay_memory/experiment_5051_verifier_trace_self_learning_memory.json"
)
EXP5059_RESULT_RELATIVE_PATH = "results/experiment_5059_d1_sota_refresh_audit.json"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 5064
SPEC_REFS = ["REQ-VERIFY-5064", "SCENARIO-VERIFY-5064"]

MODEL_SPECS: dict[str, str] = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix for audited positive promotion or guarded no-promotion."
    },
    "continuous_self_learning_task": {
        "principle": "true because this is the FR-11 continuous self-learning loop for the milestone."
    },
    "model_specs": {
        "principle": "mandated SOTA GGUF ids are recorded; this run derives skills from checked-in evidence and does not invoke legacy small models."
    },
    "self_learning_loop_executed": {
        "principle": "true only after split freeze, near-miss mining, skill proposal, self-audit, external audit, held-out evaluation, and promotion gating."
    },
    "near_miss_count": {
        "principle": "number of train-split near misses mined from Exp5051 and Exp5059 before held-out evaluation."
    },
    "candidate_skill_count": {
        "principle": "number of proposed skill or memory entries derived from train-only evidence."
    },
    "verified_skill_count": {
        "principle": "number of candidates that pass self-audit and deterministic external verifier receipts."
    },
    "promoted": {
        "principle": "true only when held-out delta is positive, contamination guard passes, and nonforgetting does not regress."
    },
    "no_promote_reason": {
        "principle": "empty only for promotion; otherwise semicolon-separated gate failures."
    },
    "pre_update_accuracy": {
        "principle": "held-out accuracy before applying the proposed skill graph update."
    },
    "post_update_accuracy": {
        "principle": "held-out accuracy after applying the proposed update under evaluation."
    },
    "heldout_delta": {
        "principle": "post_update_accuracy minus pre_update_accuracy on frozen held-out IDs."
    },
    "nonforgetting_delta": {
        "principle": "retention change on held-out examples that the pre-update verifier answered correctly."
    },
    "contamination_guard_passed": {
        "principle": "true only when held-out IDs are absent from trace inputs, candidate skills, and promoted memory entries."
    },
    "skill_graph_path": {
        "principle": "path to the audited skill graph containing candidate skills, receipts, and promotion decision."
    },
    "legacy_models_smoke_only": {
        "principle": "true; legacy small models are not used for headline trace or skill proposal provenance."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return dict(payload)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _row_id(index: int) -> str:
    return f"q{index:04d}"


def _row_index(row_id: str) -> int:
    return int(str(row_id).removeprefix("q"))


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _accuracy(bits: Sequence[int]) -> float:
    return _round_metric(sum(int(bit) for bit in bits) / len(bits)) if bits else 0.0


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _delta_label(delta: float) -> str:
    prefix = "plus" if delta >= 0 else "minus"
    return f"{prefix}_{abs(delta):.3f}".replace(".", "p")


def load_inputs(root: Path) -> tuple[JsonDict, JsonDict, JsonDict]:
    root = Path(root)
    return (
        _read_json(root / EXP5051_RESULT_RELATIVE_PATH),
        _read_json(root / EXP5051_MEMORY_RELATIVE_PATH),
        _read_json(root / EXP5059_RESULT_RELATIVE_PATH),
    )


def freeze_split(exp5051: JsonMap) -> JsonDict:
    split = exp5051.get("split_ids")
    if not isinstance(split, Mapping):
        raise ValueError("Exp5051 split_ids missing")
    train_ids = [str(row_id) for row_id in split.get("train_ids", [])]
    heldout_ids = [str(row_id) for row_id in split.get("heldout_ids", [])]
    if not train_ids or not heldout_ids:
        raise ValueError("Exp5051 split_ids must include train and heldout ids")
    return {"train_ids": train_ids, "heldout_ids": heldout_ids}


def _paired_correct(exp5059: JsonMap) -> JsonDict:
    metrics = exp5059.get("refreshed_candidate_metrics")
    paired = metrics.get("paired_correct") if isinstance(metrics, Mapping) else None
    if not isinstance(paired, Mapping):
        raise ValueError("Exp5059 refreshed_candidate_metrics.paired_correct missing")
    return {
        "verifier": [int(bool(value)) for value in paired.get("verifier", [])],
        "tuned_self_consistency": [
            int(bool(value)) for value in paired.get("tuned_self_consistency", [])
        ],
        "oracle_at_k": [int(bool(value)) for value in paired.get("oracle_at_k", [])],
    }


def _known_trace_ids(near_misses: Sequence[JsonMap]) -> set[str]:
    return {str(row["source_trace_id"]) for row in near_misses if row.get("source_trace_id")}


def mine_near_misses(
    exp5051: JsonMap,
    memory: JsonMap,
    exp5059: JsonMap,
    split: JsonMap,
) -> list[JsonDict]:
    _ = exp5051
    train_ids = set(str(row_id) for row_id in split.get("train_ids", []))
    near_misses: list[JsonDict] = []
    for trace in memory.get("verified_traces", []):
        if not isinstance(trace, Mapping):
            continue
        row_id = str(trace.get("row_id") or "")
        trace_id = str(trace.get("trace_id") or "")
        if row_id not in train_ids or not trace_id:
            continue
        near_misses.append(
            {
                "row_id": row_id,
                "source_artifact": EXP5051_MEMORY_RELATIVE_PATH,
                "source_trace_id": trace_id,
                "source_experiment": int(trace.get("source_experiment") or 5051),
                "near_miss_reasons": list(trace.get("near_miss_reasons") or []),
                "proposal_signal": "verified_replay_trace",
            }
        )

    paired = _paired_correct(exp5059)
    n_rows = min(
        len(paired["verifier"]),
        len(paired["tuned_self_consistency"]),
        len(paired["oracle_at_k"]),
    )
    for index in range(n_rows):
        row_id = _row_id(index)
        if row_id not in train_ids:
            continue
        verifier_ok = paired["verifier"][index]
        tuned_ok = paired["tuned_self_consistency"][index]
        oracle_ok = paired["oracle_at_k"][index]
        reasons: list[str] = []
        if verifier_ok == 0 and oracle_ok == 1:
            reasons.append("verifier_wrong_oracle_recoverable")
        if verifier_ok == 0 and tuned_ok == 1:
            reasons.append("verifier_wrong_tuned_sc_correct")
        if not reasons:
            continue
        near_misses.append(
            {
                "row_id": row_id,
                "source_artifact": EXP5059_RESULT_RELATIVE_PATH,
                "source_trace_id": f"5059:refreshed:{row_id}",
                "source_experiment": 5059,
                "near_miss_reasons": reasons,
                "proposal_signal": "refreshed_d1_paired_correct_near_miss",
            }
        )
    return near_misses


def _extract_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"row_id", "source_row_ids", "support_row_ids", "train_ids", "heldout_ids"}:
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    found.update(str(element) for element in item)
                elif item is not None:
                    found.add(str(item))
            found.update(_extract_ids(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found.update(_extract_ids(item))
    return found


def _self_audit_skill(skill: JsonMap) -> JsonDict:
    serialized = _json_dumps(skill).lower()
    failed: list[str] = []
    if not skill.get("source_trace_ids"):
        failed.append("source_trace_ids_missing")
    if not skill.get("source_artifacts"):
        failed.append("source_artifacts_missing")
    leak_terms = ("gold answer", "correct answer", "final_answer_correct", "label_correct")
    if any(term in serialized for term in leak_terms):
        failed.append("final_answer_leakage")
    return {
        "auditor": "deterministic_self_audit_v1",
        "verdict": "pass" if not failed else "fail",
        "failed_checks": failed,
    }


def _external_verify_skill(
    skill: JsonMap,
    *,
    known_trace_ids: set[str],
    heldout_ids: Sequence[str],
) -> JsonDict:
    failed: list[str] = []
    for trace_id in skill.get("source_trace_ids", []):
        if str(trace_id) not in known_trace_ids:
            failed.append(f"unknown_source_trace:{trace_id}")
    serialized = _json_dumps(skill)
    for row_id in heldout_ids:
        if str(row_id) in serialized:
            failed.append(f"heldout_id_leak:{row_id}")
    if not skill.get("auditable_action"):
        failed.append("auditable_action_missing")
    return {
        "verifier": "deterministic_external_skill_audit_v1",
        "passed": not failed,
        "failed_checks": failed,
    }


def build_candidate_skills(
    near_misses: Sequence[JsonMap],
    memory: JsonMap,
    exp5059: JsonMap,
    heldout_ids: Sequence[str],
) -> list[JsonDict]:
    _ = (memory, exp5059)
    known = _known_trace_ids(near_misses)
    exp5051_rows = [row for row in near_misses if row.get("source_experiment") != 5059]
    exp5059_rows = [row for row in near_misses if row.get("source_experiment") == 5059]
    candidates: list[JsonDict] = []
    if exp5051_rows:
        candidates.append(
            {
                "skill_id": "skill_5064_replay_disagreement_fallback",
                "skill_type": "memory_routing_rule",
                "auditable_action": "fallback_to_genuine_tuned_sc_on_verifier_sc_disagreement",
                "source_artifacts": [EXP5051_MEMORY_RELATIVE_PATH],
                "source_trace_ids": sorted({str(row["source_trace_id"]) for row in exp5051_rows}),
                "source_row_ids": sorted({str(row["row_id"]) for row in exp5051_rows}),
                "source_summary": "Exp5051 verified traces proposed replay fallback from structural disagreement.",
            }
        )
    if exp5059_rows:
        candidates.append(
            {
                "skill_id": "skill_5064_positive_utility_promotion_gate",
                "skill_type": "promotion_guard",
                "auditable_action": "require_positive_heldout_delta_and_nonforgetting_before_promotion",
                "source_artifacts": [EXP5059_RESULT_RELATIVE_PATH],
                "source_trace_ids": sorted({str(row["source_trace_id"]) for row in exp5059_rows}),
                "source_row_ids": sorted({str(row["row_id"]) for row in exp5059_rows}),
                "source_summary": "Exp5059 paired-correct near misses define utility-gated promotion discipline.",
            }
        )
    audited: list[JsonDict] = []
    for candidate in candidates:
        self_audit = _self_audit_skill(candidate)
        external_receipt = _external_verify_skill(
            candidate,
            known_trace_ids=known,
            heldout_ids=heldout_ids,
        )
        audited.append(
            {
                **candidate,
                "self_audit": self_audit,
                "external_verifier_receipt": external_receipt,
                "verified": self_audit["verdict"] == "pass" and external_receipt["passed"] is True,
            }
        )
    return audited


def contamination_guard(
    *,
    train_ids: Sequence[str],
    heldout_ids: Sequence[str],
    trace_inputs: Sequence[JsonMap],
    candidate_skills: Sequence[JsonMap],
    promoted_memory_entries: Sequence[JsonMap],
) -> JsonDict:
    train = set(str(row_id) for row_id in train_ids)
    heldout = set(str(row_id) for row_id in heldout_ids)
    violations: list[str] = []
    overlap = sorted(train & heldout)
    if overlap:
        violations.append(f"split_overlap:{','.join(overlap)}")
    checked = {
        "trace_inputs": trace_inputs,
        "candidate_skills": candidate_skills,
        "promoted_memory_entries": promoted_memory_entries,
    }
    for name, payload in checked.items():
        leaked = sorted(_extract_ids(payload) & heldout)
        for row_id in leaked:
            violations.append(f"{name}_heldout_id_leak:{row_id}")
    return {"passed": not violations, "violations": violations}


def evaluate_heldout(exp5051: JsonMap, exp5059: JsonMap, split: JsonMap) -> JsonDict:
    paired = _paired_correct(exp5059)
    decisions = {
        str(row.get("row_id")): str(row.get("selector") or "pre_update_verifier")
        for row in (exp5051.get("heldout_evaluation") or {}).get("selector_decisions", [])
        if isinstance(row, Mapping)
    }
    pre_bits: list[int] = []
    proposed_bits: list[int] = []
    fallback_bits: list[int] = []
    per_row: list[JsonDict] = []
    for row_id in split.get("heldout_ids", []):
        index = _row_index(str(row_id))
        if index >= len(paired["verifier"]) or index >= len(paired["tuned_self_consistency"]):
            continue
        pre = int(paired["verifier"][index])
        tuned = int(paired["tuned_self_consistency"][index])
        selector = decisions.get(str(row_id), "pre_update_verifier")
        proposed = tuned if selector == "tuned_self_consistency" else pre
        pre_bits.append(pre)
        proposed_bits.append(proposed)
        fallback_bits.append(pre)
        per_row.append(
            {
                "row_id": str(row_id),
                "pre_correct": pre,
                "proposed_correct": proposed,
                "no_promote_correct": pre,
                "selector": selector,
            }
        )
    pre_accuracy = _accuracy(pre_bits)
    post_accuracy = _accuracy(proposed_bits)
    no_promote_accuracy = _accuracy(fallback_bits)
    nonforgetting_rows = [row for row in per_row if row["pre_correct"] == 1]
    retained = [int(row["proposed_correct"]) for row in nonforgetting_rows]
    nonforgetting_retention = _accuracy(retained) if retained else 1.0
    return {
        "heldout_ids": [str(row_id) for row_id in split.get("heldout_ids", [])],
        "heldout_n": len(pre_bits),
        "pre_update_accuracy": pre_accuracy,
        "proposed_update_accuracy": post_accuracy,
        "post_update_accuracy": post_accuracy,
        "no_promote_fallback_accuracy": no_promote_accuracy,
        "heldout_delta": _round_metric(post_accuracy - pre_accuracy),
        "nonforgetting_slice_n": len(nonforgetting_rows),
        "nonforgetting_retention": nonforgetting_retention,
        "nonforgetting_delta": _round_metric(nonforgetting_retention - 1.0),
        "regressed_previously_correct_ids": [
            row["row_id"] for row in nonforgetting_rows if row["proposed_correct"] == 0
        ],
        "improved_previously_wrong_ids": [
            row["row_id"]
            for row in per_row
            if row["pre_correct"] == 0 and row["proposed_correct"] == 1
        ],
        "per_row": per_row,
    }


def promotion_decision(
    *,
    heldout_delta: float,
    nonforgetting_delta: float,
    contamination_guard_passed: bool,
) -> JsonDict:
    blockers: list[str] = []
    if heldout_delta <= 0.0:
        blockers.append("heldout_delta_nonpositive")
    if nonforgetting_delta < 0.0:
        blockers.append("nonforgetting_regressed")
    if not contamination_guard_passed:
        blockers.append("contamination_guard_failed")
    return {
        "promoted": not blockers,
        "no_promote_reason": "" if not blockers else ";".join(blockers),
    }


def _source_artifacts(root: Path) -> list[JsonDict]:
    paths = [
        EXP5051_RESULT_RELATIVE_PATH,
        EXP5051_MEMORY_RELATIVE_PATH,
        EXP5059_RESULT_RELATIVE_PATH,
    ]
    return [
        {
            "path": relative_path,
            "sha256": _sha256_file(root / relative_path),
        }
        for relative_path in paths
    ]


def _model_specs(exp5059: JsonMap) -> JsonDict:
    return {
        "mandated_sota": dict(MODEL_SPECS),
        "exp5059_model_specs": dict(exp5059.get("model_specs") or {}),
        "llm_traces_generated": False,
        "trace_proposal_mode": "derived_from_checked_in_exp5051_exp5059_evidence",
        "legacy_models_smoke_only": True,
    }


def _checksum(payload: JsonMap) -> str:
    without_checksum = {
        key: value for key, value in payload.items() if key != "reproducibility_checksum"
    }
    return "sha256:" + hashlib.sha256(_json_dumps(without_checksum).encode("utf-8")).hexdigest()


def build_skill_graph(
    *,
    split: JsonMap,
    near_misses: Sequence[JsonMap],
    candidate_skills: Sequence[JsonMap],
    guard: JsonMap,
    heldout_evaluation: JsonMap,
    decision: JsonMap,
    source_artifacts: Sequence[JsonMap],
) -> JsonDict:
    verified_skill_ids = [
        str(skill["skill_id"]) for skill in candidate_skills if skill.get("verified") is True
    ]
    promoted_skill_ids = verified_skill_ids if decision.get("promoted") is True else []
    return {
        "schema": SKILL_GRAPH_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "split_freeze": {
            "train_ids": list(split.get("train_ids", [])),
            "heldout_ids": list(split.get("heldout_ids", [])),
            "heldout_frozen_before_proposal": True,
        },
        "near_miss_count": len(near_misses),
        "near_miss_examples": [dict(row) for row in near_misses[:10]],
        "candidate_skills": [dict(skill) for skill in candidate_skills],
        "verified_skill_ids": verified_skill_ids,
        "promoted_skill_ids": promoted_skill_ids,
        "contamination_guard": dict(guard),
        "heldout_evaluation": dict(heldout_evaluation),
        "promotion_decision": dict(decision),
        "source_artifacts": list(source_artifacts),
    }


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    for field in (
        "continuous_self_learning_task",
        "self_learning_loop_executed",
        "promoted",
        "contamination_guard_passed",
        "legacy_models_smoke_only",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("continuous_self_learning_task") is not True:
        errors.append("continuous_self_learning_task_true")
    if artifact.get("legacy_models_smoke_only") is not True:
        errors.append("legacy_models_smoke_only_true")
    for field in ("near_miss_count", "candidate_skill_count", "verified_skill_count"):
        if not isinstance(artifact.get(field), int) or int(artifact.get(field, -1)) < 0:
            errors.append(field)
    for field in (
        "pre_update_accuracy",
        "post_update_accuracy",
        "heldout_delta",
        "nonforgetting_delta",
    ):
        if _number(artifact.get(field)) is None:
            errors.append(field)
    if artifact.get("promoted") is False and not artifact.get("no_promote_reason"):
        errors.append("no_promote_reason")
    model_specs = artifact.get("model_specs")
    mandated = model_specs.get("mandated_sota") if isinstance(model_specs, Mapping) else None
    if not isinstance(mandated, Mapping) or dict(mandated) != MODEL_SPECS:
        errors.append("model_specs")
    if not str(artifact.get("skill_graph_path") or ""):
        errors.append("skill_graph_path")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    skill_graph_path: Path | None = None,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    skill_graph_path = (
        Path(skill_graph_path) if skill_graph_path else root / SKILL_GRAPH_RELATIVE_PATH
    )
    start = float(now())
    exp5051, memory, exp5059 = load_inputs(root)
    split = freeze_split(exp5051)
    near_misses = mine_near_misses(exp5051, memory, exp5059, split)
    candidate_skills = build_candidate_skills(
        near_misses,
        memory,
        exp5059,
        split["heldout_ids"],
    )
    guard = contamination_guard(
        train_ids=split["train_ids"],
        heldout_ids=split["heldout_ids"],
        trace_inputs=near_misses,
        candidate_skills=candidate_skills,
        promoted_memory_entries=[],
    )
    heldout_evaluation = evaluate_heldout(exp5051, exp5059, split)
    decision = promotion_decision(
        heldout_delta=float(heldout_evaluation["heldout_delta"]),
        nonforgetting_delta=float(heldout_evaluation["nonforgetting_delta"]),
        contamination_guard_passed=bool(guard["passed"]),
    )
    sources = _source_artifacts(root)
    skill_graph = build_skill_graph(
        split=split,
        near_misses=near_misses,
        candidate_skills=candidate_skills,
        guard=guard,
        heldout_evaluation=heldout_evaluation,
        decision=decision,
        source_artifacts=sources,
    )
    if write:
        write_json(skill_graph_path, skill_graph)

    promoted = bool(decision["promoted"])
    heldout_delta = float(heldout_evaluation["heldout_delta"])
    verdict_prefix = "success_promoted_positive_utility" if promoted else "complete_guarded_no_promote"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": f"{verdict_prefix}_{_delta_label(heldout_delta)}",
        "continuous_self_learning_task": True,
        "model_specs": _model_specs(exp5059),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "self_learning_loop_executed": True,
        "near_miss_count": len(near_misses),
        "candidate_skill_count": len(candidate_skills),
        "verified_skill_count": sum(1 for skill in candidate_skills if skill.get("verified") is True),
        "promoted": promoted,
        "promoted_skill_ids": list(skill_graph["promoted_skill_ids"]),
        "no_promote_reason": str(decision["no_promote_reason"]),
        "pre_update_accuracy": float(heldout_evaluation["pre_update_accuracy"]),
        "post_update_accuracy": float(heldout_evaluation["post_update_accuracy"]),
        "heldout_delta": heldout_delta,
        "nonforgetting_delta": float(heldout_evaluation["nonforgetting_delta"]),
        "contamination_guard_passed": bool(guard["passed"]),
        "skill_graph_path": skill_graph_path.as_posix(),
        "skill_graph_sha256": _sha256_file(skill_graph_path) if write else None,
        "legacy_models_smoke_only": True,
        "split_ids": split,
        "source_artifacts": sources,
        "external_verifier_audit_receipts": [
            dict(skill["external_verifier_receipt"]) for skill in candidate_skills
        ],
        "heldout_evaluation": heldout_evaluation,
        "promotion_decision": decision,
        "positive_control": {
            "source": EXP5059_RESULT_RELATIVE_PATH,
            "headroom_present": bool(
                (exp5059.get("refreshed_candidate_metrics") or {}).get("headroom_present")
            ),
            "oracle_at_k": (exp5059.get("refreshed_candidate_metrics") or {}).get(
                "oracle_at_k"
            ),
        },
        "duration_s": round(max(0.0001, float(now()) - start), 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "continuous_self_learning_task": artifact.get("continuous_self_learning_task"),
                "promoted": artifact.get("promoted"),
                "heldout_delta": artifact.get("heldout_delta"),
                "nonforgetting_delta": artifact.get("nonforgetting_delta"),
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
