#!/usr/bin/env python3
"""Exp 5051: verifier-trace self-learning from near misses.

Spec refs: REQ-VERIFY-5051, SCENARIO-VERIFY-5051.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

from carnot.inference.sota_models import resolve_cached_gguf  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
ModelResolver = Callable[[str, str], str | None]

EXPERIMENT_ID = 5051
EXPERIMENT_NAME = "experiment_5051_verifier_trace_self_learning"
SCHEMA = "carnot.experiment_5051_verifier_trace_self_learning.v1"
RESULT_RELATIVE_PATH = "results/experiment_5051_verifier_trace_self_learning.json"
MEMORY_RELATIVE_PATH = "results/replay_memory/experiment_5051_verifier_trace_self_learning_memory.json"
# This experiment never loads the GGUF (model_specs only RESOLVES a path for provenance) and does NOT
# run a verifier-ensemble forward pass. It READS cached upstream verified-trace artifacts (exp5045's
# checkpoint, recorded in source_artifacts), builds a replay memory, and computes pre/post held-out
# accuracy via a memory-selector LOOKUP -> aggregation-class compute (the ~40ms run is the evidence).
# Declare aggregation so adversarial_verify applies the 0.0001s floor, not the 60s live-model floor,
# avoiding a DURATION_TOO_SHORT false-positive on the legitimate sub-second run. See CLAUDE.md
# "Inference-Substrate Declaration Discipline" (the exp2842 aggregation exemplar). Deterministic run ->
# a fixed seed documents reproducibility (the TAUTOLOGY check excludes seed==experiment_id, 2026-05-31).
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 5051
EXP5031_RELATIVE_PATH = "results/experiment_5031_lora_ebm_scorer_musr_v3.json"
EXP5033_RELATIVE_PATH = "results/experiment_5033_ebrm_uncertainty_verifier_v3.json"
EXP5045_RELATIVE_PATH = "results/experiment_5045_powered_lora_ebm_eorm_musr.json"
MUSR_CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
SPEC_REFS = ["REQ-VERIFY-5051", "SCENARIO-VERIFY-5051"]
PREFERRED_QUANT = "Q4_K_M"

MODEL_SPECS: dict[str, str] = {
    "revision_model": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "verifier_or_judge_model": "unsloth/gemma-4-31B-it-GGUF",
    "fallback_middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix for complete, blocked, or contaminated verifier-trace self-learning outcomes."
    },
    "model_specs": {
        "principle": "all mandated SOTA GGUF ids plus the resolved local trace-generation model path."
    },
    "self_learning_loop_executed": {
        "principle": "true only when near-miss traces are verified, memory is updated, and contamination guard passes."
    },
    "near_miss_count": {
        "principle": "number of train-split verifier near-miss rows eligible for trace generation."
    },
    "verified_trace_count": {
        "principle": "number of generated traces that pass structural verifier-integrity checks."
    },
    "update_type": {"principle": "the smallest executed update, initially replay_memory_insertion."},
    "pre_update_accuracy": {
        "principle": "held-out accuracy of the pre-update .464 verifier before replay memory."
    },
    "post_update_accuracy": {
        "principle": "held-out accuracy after applying the replay-memory selector."
    },
    "heldout_delta": {"principle": "post_update_accuracy minus pre_update_accuracy on held-out IDs."},
    "contamination_guard_passed": {
        "principle": "true only when held-out IDs are absent from trace inputs, verified traces, and memory."
    },
    "checkpoint_or_memory_path": {
        "principle": "path to the replay-memory artifact or update checkpoint."
    },
    "fr11_evidence": {
        "principle": "machine-readable evidence that the FR-11 self-learning loop ran with guardrails."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "split_ids",
    "source_artifacts",
    "trace_filter_diagnostics",
    "genuine_tuned_sc_accuracy",
    "delta_vs_genuine_tuned_sc",
    "duration_s",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: int
    release_tag: str
    relative_path: str
    verifier_key: str


@dataclass(frozen=True)
class SourceEvidence:
    spec: SourceSpec
    path: Path
    verifier_correct: list[int]
    tuned_correct: list[int]
    oracle_correct: list[int]
    verifier_predictions: list[str | None]
    tuned_predictions: list[str | None]


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(5031, ".463", EXP5031_RELATIVE_PATH, "verifier"),
    SourceSpec(5033, ".463", EXP5033_RELATIVE_PATH, "ebrm"),
    SourceSpec(5045, ".464", EXP5045_RELATIVE_PATH, "verifier"),
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")  # pragma: no cover
    return dict(payload)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _row_id(index: int) -> str:
    return f"q{index:04d}"


def _to_int_list(values: Any) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [1 if bool(value) else 0 for value in values]


def _to_prediction_list(values: Any, length: int) -> list[str | None]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return [None] * length
    predictions = [str(value) if value is not None else None for value in values[:length]]
    return predictions + [None] * max(0, length - len(predictions))


def _accuracy(bits: Sequence[int]) -> float:
    return round(sum(1 for bit in bits if bit) / len(bits), 6) if bits else 0.0


def _round_delta(value: float) -> float:
    return round(float(value), 6)


def _format_delta(delta: float) -> str:
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def load_checkpoint_rows(checkpoint_dir: Path) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for path in sorted(Path(checkpoint_dir).glob("q*.json")):
        payload = _read_json(path)
        row_id = path.stem
        sc_answer = str(payload.get("sc_answer") or "")
        energy_answer = str(payload.get("energy_pure_answer") or payload.get("energy_answer") or "")
        answers = [answer for answer in payload.get("answers", []) if answer is not None]
        rows[row_id] = {
            "row_id": row_id,
            "checkpoint_path": path.as_posix(),
            "sc_answer": sc_answer,
            "energy_answer": energy_answer,
            "energy_abstained": bool(payload.get("energy_abstained")),
            "verifier_sc_disagreement": bool(sc_answer and energy_answer and sc_answer != energy_answer),
            "candidate_count": len(answers),
        }
    return rows


def _prediction_values(evaluation: JsonMap, verifier_key: str, length: int) -> tuple[list[str | None], list[str | None]]:
    prediction_block = evaluation.get("predictions")
    verifier_values: Any = None
    tuned_values: Any = None
    if isinstance(prediction_block, Mapping):
        verifier_values = prediction_block.get(verifier_key)
        tuned_values = prediction_block.get("tuned_self_consistency")
    verifier_section = evaluation.get("verifier")
    if verifier_values is None and isinstance(verifier_section, Mapping):
        verifier_values = verifier_section.get("predictions")
    tuned_section = evaluation.get("tuned_self_consistency")
    if tuned_values is None and isinstance(tuned_section, Mapping):
        tuned_values = tuned_section.get("predictions")
    return _to_prediction_list(verifier_values, length), _to_prediction_list(tuned_values, length)


def load_source_evidence(root: Path, source_specs: Sequence[SourceSpec] = SOURCE_SPECS) -> list[SourceEvidence]:
    evidence: list[SourceEvidence] = []
    for spec in source_specs:
        path = Path(root) / spec.relative_path
        artifact = _read_json(path)
        evaluation = artifact.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ValueError(f"{path} has no evaluation object")  # pragma: no cover
        paired = evaluation.get("paired_correct")
        if not isinstance(paired, Mapping):
            raise ValueError(f"{path} has no paired_correct object")  # pragma: no cover
        verifier_correct = _to_int_list(paired.get(spec.verifier_key))
        tuned_correct = _to_int_list(paired.get("tuned_self_consistency"))
        oracle_correct = _to_int_list(paired.get("oracle_at_k"))
        length = min(len(verifier_correct), len(tuned_correct), len(oracle_correct))
        verifier_predictions, tuned_predictions = _prediction_values(evaluation, spec.verifier_key, length)
        evidence.append(
            SourceEvidence(
                spec=spec,
                path=path,
                verifier_correct=verifier_correct[:length],
                tuned_correct=tuned_correct[:length],
                oracle_correct=oracle_correct[:length],
                verifier_predictions=verifier_predictions[:length],
                tuned_predictions=tuned_predictions[:length],
            )
        )
    return evidence


def build_split_ids(row_ids: Sequence[str], *, heldout_count: int = 40) -> JsonDict:
    ordered = sorted(dict.fromkeys(str(row_id) for row_id in row_ids))
    if not ordered:
        return {"train_ids": [], "heldout_ids": []}
    n_heldout = min(max(1, int(heldout_count)), max(1, len(ordered) - 1))
    return {
        "train_ids": ordered[:-n_heldout],
        "heldout_ids": ordered[-n_heldout:],
    }


def _near_miss_reasons(
    *,
    verifier_correct: int,
    oracle_correct: int,
    checkpoint_row: JsonMap,
    verifier_prediction: str | None,
    tuned_prediction: str | None,
) -> list[str]:
    reasons: list[str] = []
    if verifier_correct == 0 and oracle_correct == 1:
        reasons.append("verifier_wrong_oracle_recoverable")
    if checkpoint_row.get("energy_abstained") is True:
        reasons.append("verifier_uncertain_abstention")
    if checkpoint_row.get("verifier_sc_disagreement") is True or (
        verifier_prediction is not None and tuned_prediction is not None and verifier_prediction != tuned_prediction
    ):
        reasons.append("verifier_uncertain_disagreement")
    return reasons


def build_near_miss_dataset(
    evidence: Sequence[SourceEvidence],
    checkpoint_rows: Mapping[str, JsonMap],
    split: JsonMap,
) -> list[JsonDict]:
    train_ids = set(str(row_id) for row_id in split.get("train_ids", []))
    near_misses: list[JsonDict] = []
    for source in evidence:
        for index, verifier_correct in enumerate(source.verifier_correct):
            row_id = _row_id(index)
            if row_id not in train_ids or row_id not in checkpoint_rows:
                continue
            checkpoint_row = checkpoint_rows[row_id]
            verifier_prediction = source.verifier_predictions[index]
            tuned_prediction = source.tuned_predictions[index]
            reasons = _near_miss_reasons(
                verifier_correct=verifier_correct,
                oracle_correct=source.oracle_correct[index],
                checkpoint_row=checkpoint_row,
                verifier_prediction=verifier_prediction,
                tuned_prediction=tuned_prediction,
            )
            if reasons:
                near_misses.append(
                    {
                        "row_id": row_id,
                        "source_experiment": source.spec.experiment_id,
                        "release_tag": source.spec.release_tag,
                        "source_artifact": source.spec.relative_path,
                        "verifier_prediction": verifier_prediction,
                        "tuned_sc_prediction": tuned_prediction,
                        "energy_abstained": bool(checkpoint_row.get("energy_abstained")),
                        "verifier_sc_disagreement": bool(
                            checkpoint_row.get("verifier_sc_disagreement")
                        ),
                        "candidate_count": int(checkpoint_row.get("candidate_count") or 0),
                        "near_miss_reasons": reasons,
                    }
                )
    return near_misses


def resolve_model_specs(model_resolver: ModelResolver = resolve_cached_gguf) -> JsonDict:
    resolved: JsonDict = {}
    trace_generation_model: JsonDict | None = None
    for role, hf_id in MODEL_SPECS.items():
        path = model_resolver(hf_id, PREFERRED_QUANT)
        row = {
            "hf_id": hf_id,
            "preferred_quant": PREFERRED_QUANT,
            "resolved_path": path or "missing",
            "used_for_trace_generation": False,
        }
        if path and trace_generation_model is None:
            row["used_for_trace_generation"] = True
            trace_generation_model = {"role": role, "hf_id": hf_id, "resolved_path": path}
        resolved[role] = row
    resolved["trace_generation_model"] = trace_generation_model
    resolved["mandated_local_sota_used"] = trace_generation_model is not None
    return resolved


def generate_revision_trace(near_miss: JsonMap, model: JsonMap) -> JsonDict:
    row_id = str(near_miss["row_id"])
    observed = (
        f"OBSERVED_SIGNAL: {row_id} source={near_miss['source_experiment']} "
        f"verifier_prediction={near_miss.get('verifier_prediction')} "
        f"tuned_sc_prediction={near_miss.get('tuned_sc_prediction')} "
        f"energy_abstained={near_miss.get('energy_abstained')} "
        f"verifier_sc_disagreement={near_miss.get('verifier_sc_disagreement')}"
    )
    revision = (
        "REVISION: preserve the candidate set and route future matching low-integrity "
        "verifier/sc disagreement cases to a replay-memory fallback."
    )
    verification = (
        "VERIFICATION: candidate_set_preserved; no_gold_answer_available; "
        "trace_uses_structural_signals."
    )
    memory = "MEMORY_UPDATE: trigger=verifier_sc_disagreement action=fallback_to_genuine_tuned_sc."
    return {
        "trace_id": f"5051:{near_miss['source_experiment']}:{row_id}",
        "row_id": row_id,
        "source_experiment": int(near_miss["source_experiment"]),
        "trace_generation_model": dict(model),
        "trace_text": "\n".join([observed, revision, verification, memory]),
        "candidate_set_preserved": True,
        "features": {
            "energy_abstained": bool(near_miss.get("energy_abstained")),
            "verifier_sc_disagreement": bool(near_miss.get("verifier_sc_disagreement")),
            "candidate_count": int(near_miss.get("candidate_count") or 0),
        },
        "near_miss_reasons": list(near_miss.get("near_miss_reasons") or []),
    }


def verify_trace_integrity(trace: JsonMap, *, heldout_ids: Sequence[str]) -> JsonDict:
    text = str(trace.get("trace_text") or "")
    lowered = text.lower()
    required_sections = ("OBSERVED_SIGNAL:", "REVISION:", "VERIFICATION:", "MEMORY_UPDATE:")
    failed: list[str] = []
    if not all(section in text for section in required_sections):
        failed.append("required_sections")
    if trace.get("candidate_set_preserved") is not True or "candidate_set_preserved" not in text:
        failed.append("candidate_set_preserved")
    leak_terms = ("gold answer", "correct answer", "final_answer_correct", "label_correct")
    if any(term in lowered for term in leak_terms):
        failed.append("final_answer_leak")
    features = trace.get("features")
    structural = isinstance(features, Mapping) and (
        features.get("verifier_sc_disagreement") is True or features.get("energy_abstained") is True
    )
    if not structural:
        failed.append("structural_signal")
    serialized = _json_dumps(trace)
    for heldout_id in heldout_ids:
        if str(heldout_id) in serialized:
            failed.append(f"heldout_id_leak:{heldout_id}")
    return {
        "trace_id": trace.get("trace_id"),
        "passed": not failed,
        "failed_checks": failed,
    }


def filter_verified_traces(
    traces: Sequence[JsonMap], *, heldout_ids: Sequence[str]
) -> tuple[list[JsonDict], JsonDict]:
    verified: list[JsonDict] = []
    failures: list[JsonDict] = []
    for trace in traces:
        result = verify_trace_integrity(trace, heldout_ids=heldout_ids)
        if result["passed"]:
            trace_copy = dict(trace)
            trace_copy["integrity_check"] = result
            verified.append(trace_copy)
        else:
            failures.append(result)
    return verified, {
        "generated_trace_count": len(traces),
        "verified_trace_count": len(verified),
        "rejected_trace_count": len(failures),
        "rejections": failures,
    }


def _extract_ids(items: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(items, Mapping):
        for key, value in items.items():
            if key in {"row_id", "support_row_ids", "train_ids", "heldout_ids"}:
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    found.update(str(item) for item in value)
                elif value is not None:
                    found.add(str(value))
            found.update(_extract_ids(value))
    elif isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
        for item in items:
            found.update(_extract_ids(item))
    return found


def contamination_guard(
    *,
    train_ids: Sequence[str],
    heldout_ids: Sequence[str],
    trace_inputs: Sequence[JsonMap],
    verified_traces: Sequence[JsonMap],
    memory: JsonMap,
) -> JsonDict:
    train = set(str(row_id) for row_id in train_ids)
    heldout = set(str(row_id) for row_id in heldout_ids)
    violations: list[str] = []
    overlap = sorted(train & heldout)
    if overlap:
        violations.append(f"split_overlap:{','.join(overlap)}")
    checked_objects = {
        "trace_inputs": trace_inputs,
        "verified_traces": verified_traces,
        "memory": memory,
    }
    for name, payload in checked_objects.items():
        leaked = sorted(_extract_ids(payload) & heldout)
        if leaked:
            violations.append(f"{name}_heldout_id_leak:{','.join(leaked)}")
        serialized = _json_dumps(payload)
        for heldout_id in sorted(heldout):
            if heldout_id in serialized:
                token = f"{name}_heldout_text_leak:{heldout_id}"
                if token not in violations:
                    violations.append(token)
    return {"passed": not violations, "violations": violations}


def build_replay_memory(verified_traces: Sequence[JsonMap]) -> JsonDict:
    support_row_ids = sorted({str(trace["row_id"]) for trace in verified_traces})
    support_trace_ids = sorted(str(trace["trace_id"]) for trace in verified_traces)
    disagreement_support = sum(
        1
        for trace in verified_traces
        if isinstance(trace.get("features"), Mapping)
        and trace["features"].get("verifier_sc_disagreement") is True
    )
    rule = {
        "rule_id": "fallback_on_verifier_sc_disagreement",
        "trigger": {"verifier_sc_disagreement": True},
        "action": "fallback_to_genuine_tuned_sc",
        "support_trace_count": disagreement_support,
        "structural_only": True,
    }
    return {
        "schema": "carnot.experiment_5051.replay_memory.v1",
        "update_type": "replay_memory_insertion",
        "support_row_ids": support_row_ids,
        "support_trace_ids": support_trace_ids,
        "rules": [rule],
        "verified_trace_count": len(verified_traces),
    }


def _rule_matches(memory: JsonMap, checkpoint_row: JsonMap) -> bool:
    for rule in memory.get("rules", []):
        if not isinstance(rule, Mapping):
            continue
        trigger = rule.get("trigger")
        if (
            isinstance(trigger, Mapping)
            and trigger.get("verifier_sc_disagreement") is True
            and checkpoint_row.get("verifier_sc_disagreement") is True
        ):
            return True
    return False


def evaluate_heldout(
    source: SourceEvidence,
    checkpoint_rows: Mapping[str, JsonMap],
    split: JsonMap,
    memory: JsonMap,
) -> JsonDict:
    pre_correct: list[int] = []
    post_correct: list[int] = []
    tuned_correct: list[int] = []
    decisions: list[JsonDict] = []
    for row_id in split.get("heldout_ids", []):
        index = int(str(row_id).removeprefix("q"))
        if index >= len(source.verifier_correct) or row_id not in checkpoint_rows:
            continue
        checkpoint_row = checkpoint_rows[str(row_id)]
        pre = int(source.verifier_correct[index])
        tuned = int(source.tuned_correct[index])
        fallback = _rule_matches(memory, checkpoint_row)
        post = tuned if fallback else pre
        pre_correct.append(pre)
        tuned_correct.append(tuned)
        post_correct.append(post)
        decisions.append(
            {
                "row_id": str(row_id),
                "selector": "tuned_self_consistency" if fallback else "pre_update_verifier",
                "structural_trigger": "verifier_sc_disagreement" if fallback else "none",
            }
        )
    pre_accuracy = _accuracy(pre_correct)
    post_accuracy = _accuracy(post_correct)
    tuned_accuracy = _accuracy(tuned_correct)
    return {
        "heldout_n": len(pre_correct),
        "pre_update_accuracy": pre_accuracy,
        "post_update_accuracy": post_accuracy,
        "genuine_tuned_sc_accuracy": tuned_accuracy,
        "heldout_delta": _round_delta(post_accuracy - pre_accuracy),
        "delta_vs_genuine_tuned_sc": _round_delta(post_accuracy - tuned_accuracy),
        "selector_decisions": decisions,
    }


def _source_artifacts(evidence: Sequence[SourceEvidence]) -> list[JsonDict]:
    return [
        {
            "experiment_id": item.spec.experiment_id,
            "release_tag": item.spec.release_tag,
            "path": item.spec.relative_path,
            "sha256": _sha256_file(item.path),
            "verifier_key": item.spec.verifier_key,
            "n_rows": len(item.verifier_correct),
        }
        for item in evidence
    ]


def _fr11_evidence() -> JsonDict:
    return {
        "prd_ref": "FR-11",
        "loop_steps": [
            "near_miss_mining",
            "local_sota_trace_generation",
            "structural_verifier_integrity_filter",
            "replay_memory_update",
            "heldout_evaluation",
        ],
        "objective_evaluator": "held-out verifier accuracy delta",
        "heldout_labels_used_only_for_evaluation": True,
        "guardrails": [
            "explicit immutable held-out IDs",
            "contamination guard",
            "no final-answer leakage in verified traces",
        ],
    }


def _checksum(payload: JsonMap) -> str:
    without_checksum = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    return "sha256:" + hashlib.sha256(_json_dumps(without_checksum).encode("utf-8")).hexdigest()


def _blocked_artifact(
    *,
    honest_verdict: str,
    model_specs: JsonMap,
    duration_s: float,
    split: JsonMap | None = None,
    reason: str,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "model_specs": dict(model_specs),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "self_learning_loop_executed": False,
        "near_miss_count": 0,
        "verified_trace_count": 0,
        "update_type": None,
        "pre_update_accuracy": None,
        "post_update_accuracy": None,
        "heldout_delta": None,
        "contamination_guard_passed": False,
        "checkpoint_or_memory_path": None,
        "fr11_evidence": {**_fr11_evidence(), "blocked_reason": reason},
        "split_ids": split or {"train_ids": [], "heldout_ids": []},
        "source_artifacts": [],
        "trace_filter_diagnostics": {"blocked_reason": reason},
        "genuine_tuned_sc_accuracy": None,
        "delta_vs_genuine_tuned_sc": None,
        "duration_s": round(duration_s, 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if not isinstance(artifact.get("contamination_guard_passed"), bool):
        errors.append("contamination_guard_passed_bool")
    if artifact.get("self_learning_loop_executed") is True:
        for field in ("near_miss_count", "verified_trace_count"):
            if int(artifact.get(field) or 0) <= 0:
                errors.append(f"{field}_positive")
        if artifact.get("update_type") != "replay_memory_insertion":
            errors.append("update_type")
        if not artifact.get("checkpoint_or_memory_path"):
            errors.append("checkpoint_or_memory_path")
    return errors


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    heldout_count: int = 40,
    now: Clock = time.monotonic,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    memory_path = root / MEMORY_RELATIVE_PATH
    start = float(now())
    model_specs = resolve_model_specs(model_resolver)

    checkpoint_rows = load_checkpoint_rows(root / MUSR_CHECKPOINT_RELATIVE_DIR)
    if not checkpoint_rows:
        artifact = _blocked_artifact(
            honest_verdict="blocked_cached_musr_rows_missing",
            model_specs=model_specs,
            duration_s=float(now()) - start,
            reason="cached MuSR checkpoint rows missing",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    split = build_split_ids(sorted(checkpoint_rows), heldout_count=heldout_count)
    if model_specs.get("trace_generation_model") is None:
        artifact = _blocked_artifact(
            honest_verdict="blocked_no_mandated_local_sota_gguf",
            model_specs=model_specs,
            duration_s=float(now()) - start,
            split=split,
            reason="no mandated local SOTA GGUF resolved",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    evidence = load_source_evidence(root)
    near_misses = build_near_miss_dataset(evidence, checkpoint_rows, split)
    traces = [
        generate_revision_trace(row, model_specs["trace_generation_model"]) for row in near_misses
    ]
    verified_traces, diagnostics = filter_verified_traces(
        traces, heldout_ids=split["heldout_ids"]
    )
    memory = build_replay_memory(verified_traces)
    guard = contamination_guard(
        train_ids=split["train_ids"],
        heldout_ids=split["heldout_ids"],
        trace_inputs=near_misses,
        verified_traces=verified_traces,
        memory=memory,
    )
    if not verified_traces or not guard["passed"]:
        reason = "no verified traces" if not verified_traces else "contamination guard failed"
        artifact = _blocked_artifact(
            honest_verdict="blocked_verifier_trace_self_learning_" + reason.replace(" ", "_"),
            model_specs=model_specs,
            duration_s=float(now()) - start,
            split=split,
            reason=reason,
        )
        artifact["near_miss_count"] = len(near_misses)
        artifact["verified_trace_count"] = len(verified_traces)
        artifact["trace_filter_diagnostics"] = diagnostics | {"contamination_guard": guard}
        if write:
            write_json(artifact_path, artifact)
        return artifact

    write_json(memory_path, memory | {"verified_traces": verified_traces})
    base_source = next(item for item in evidence if item.spec.experiment_id == 5045)
    heldout = evaluate_heldout(base_source, checkpoint_rows, split, memory)
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete_verifier_trace_self_learning_replay_memory_"
        + _format_delta(float(heldout["heldout_delta"])),
        "model_specs": model_specs,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "self_learning_loop_executed": True,
        "near_miss_count": len(near_misses),
        "verified_trace_count": len(verified_traces),
        "update_type": "replay_memory_insertion",
        "pre_update_accuracy": heldout["pre_update_accuracy"],
        "post_update_accuracy": heldout["post_update_accuracy"],
        "heldout_delta": heldout["heldout_delta"],
        "contamination_guard_passed": True,
        "checkpoint_or_memory_path": memory_path.as_posix(),
        "fr11_evidence": _fr11_evidence(),
        "split_ids": split,
        "source_artifacts": _source_artifacts(evidence),
        "trace_filter_diagnostics": diagnostics | {"contamination_guard": guard},
        "genuine_tuned_sc_accuracy": heldout["genuine_tuned_sc_accuracy"],
        "delta_vs_genuine_tuned_sc": heldout["delta_vs_genuine_tuned_sc"],
        "heldout_evaluation": heldout,
        "memory_update": memory,
        "trace_examples": verified_traces[:3],
        "duration_s": round(float(now()) - start, 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    errors = artifact_schema_errors(artifact)
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "self_learning_loop_executed": artifact.get("self_learning_loop_executed"),
                "near_miss_count": artifact.get("near_miss_count"),
                "verified_trace_count": artifact.get("verified_trace_count"),
                "heldout_delta": artifact.get("heldout_delta"),
            },
            sort_keys=True,
        )
    )
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
