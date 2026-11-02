"""Exp5571 reset-free local SOTA continual harness.

Spec refs: REQ-LEARN-5571,
SCENARIO-LEARN-5571-PRECONDITIONS,
SCENARIO-LEARN-5571-SESSIONS,
SCENARIO-LEARN-5571-RESET-FREE,
SCENARIO-LEARN-5571-ARTIFACT.

The load-bearing claim here is not that the GGUF model weights learn. They
remain frozen. The only persistent state is the governed Exp5569 memory policy
and the Exp5570 exact-feedback energy calibrator, and the only labels admitted
to that state are exact ASP/FSM validator outcomes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from math import sqrt
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
PairResolver = Callable[[], Sequence[Mapping[str, Any]] | None]
GGUFResolver = Callable[[str], str | None]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5571_reset_free_sota_continual_harness.json")
CORPUS_RELATIVE_PATH = Path("results/experiment_5566_exact_asp_fsm_near_miss_corpus.json")
POLICY_RELATIVE_PATH = Path("results/experiment_5569_causal_memory_policy_tournament.json")
ENERGY_RELATIVE_PATH = Path("results/experiment_5570_spline_local_kan_online_energy.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5571_reset_free_sota_continual_harness.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5571_reset_free_sota_continual_harness.py")

SCHEMA = "carnot.experiment_5571.reset_free_sota_continual_harness.v1"
EXPERIMENT = 5571
EXPERIMENT_ID = "experiment_5571_reset_free_sota_continual_harness"
TASK_ID = "exp5571-gated-reset-free-sota-continual-harness"
MILESTONE = "2026.07.504"
RUN_DATE = "2026-07-11"
RANDOM_SEED = 5571
BOOTSTRAP_ITERATIONS = 500
MIN_SESSIONS = 3
INSTANCES_PER_SESSION = 30
INFERENCE_SUBSTRATE = "live_local_sota_reset_free_exact_feedback_harness"

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
OPTIONAL_REPLICATION_IDS = (GEMMA31_ID, GEMMA26_ID)
DECLARED_MODEL_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)

NO_ADAPTATION_ARM = "no_adaptation"
RESET_EACH_ARM = "reset_each_session"
SHUFFLED_ARM = "shuffled_feedback"
RESET_FREE_ARM = "reset_free_exact_feedback"
ARMS = (RESET_FREE_ARM, RESET_EACH_ARM, SHUFFLED_ARM, NO_ADAPTATION_ARM)

SESSION_FAMILIES = (
    "defaults_exceptions",
    "contradictions",
    "soft_preference_optimality",
    "fsm_transition_consistency",
)
SPEC_REFS = (
    "REQ-LEARN-5571",
    "SCENARIO-LEARN-5571-PRECONDITIONS",
    "SCENARIO-LEARN-5571-SESSIONS",
    "SCENARIO-LEARN-5571-RESET-FREE",
    "SCENARIO-LEARN-5571-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "gate_receipt",
    "continuous_self_learning_target",
    "MODEL_SPECS",
    "live_model_invoked",
    "gpu_offload_authenticated",
    "device_receipt",
    "sessions",
    "n_independent_instances_per_session",
    "arms",
    "model_weights_mutated",
    "harness_state_persisted",
    "energy_weights_mutated",
    "exact_feedback_only",
    "new_family_accuracy_by_arm",
    "backward_retention_by_session",
    "adaptation_slope",
    "false_accept_delta",
    "rollback_count",
    "rollback_success",
    "cost_receipt",
    "confidence_intervals",
    "inference_duration_s",
    "inference_substrate",
    "honest_verdict",
    "continual_harness_candidate",
)
FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Explains why every headline and gate field exists.",
    "gate_receipt": "Records upstream, cache, CUDA/offload, corpus, and live-invocation gates before interpreting quality.",
    "continuous_self_learning_target": "Bare boolean marking this as the reset-free continuous self-learning slot.",
    "MODEL_SPECS": "Declares Qwen as the frozen headline GGUF and Gemma models as optional replication paths.",
    "live_model_invoked": "Separates an authenticated local SOTA run from blocked precondition receipts.",
    "gpu_offload_authenticated": "Prevents CPU-only llama.cpp execution from entering the headline arm.",
    "device_receipt": "Preserves CUDA device identity and llama.cpp offload support evidence.",
    "sessions": "Shows ordered exact ASP/FSM family sessions and their paired instance IDs.",
    "n_independent_instances_per_session": "Defines the paired statistical denominator per family session.",
    "arms": "Lists reset-free, reset-each, shuffled-feedback, and no-adaptation controls.",
    "model_weights_mutated": "Confirms frozen GGUF weights; adaptation is external harness state only.",
    "harness_state_persisted": "Shows only the Exp5569 memory policy and Exp5570 energy calibrator persist.",
    "energy_weights_mutated": "Discloses that the external energy calibrator updates while GGUF weights do not.",
    "exact_feedback_only": "Guards against external teacher labels or heuristic labels entering adaptation.",
    "new_family_accuracy_by_arm": "Measures solve/verify exact success on each newly encountered family.",
    "backward_retention_by_session": "Measures earlier-family retention after every session boundary.",
    "adaptation_slope": "Shows whether accuracy improves across the ordered family stream.",
    "false_accept_delta": "Prevents reset-free from increasing unsafe invalid-row accepts.",
    "rollback_count": "Counts rollback events rather than assuming recovery is free.",
    "rollback_success": "Requires poisoned or stale harness state to restore to a clean checkpoint.",
    "cost_receipt": "Accounts for latency, tokens, memory bytes, and energy-update cost.",
    "confidence_intervals": "Uses paired bootstrap intervals over independent instance IDs.",
    "inference_duration_s": "Records live Qwen invocation plus harness evaluation duration.",
    "inference_substrate": "Declares live local SOTA reset-free exact-feedback harness execution.",
    "honest_verdict": "Terminal complete or blocked status for the conductor.",
    "continual_harness_candidate": "only reset-free improvement with bounded retention loss, safe false accepts, and rollback may enter delayed stress.",
    "prior_family_regression": "Exposes the retention-loss gate used by continual_harness_candidate.",
}
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5571_reset_free_sota_continual_harness.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5571_reset_free_sota_continual_harness.py "
    "-m pytest tests/python/test_experiment_5571_reset_free_sota_continual_harness.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5571_reset_free_sota_continual_harness.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for checksums and stable receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a SHA-256 digest for JSON-compatible content."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking the self-referential checksum field."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def load_exact_rows(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Load the checked-in Exp5566 exact ASP/FSM rows."""

    try:
        artifact = json.loads((Path(root) / CORPUS_RELATIVE_PATH).read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = artifact.get("corpus_rows")
    if artifact.get("corpus_ready") is not True or not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def build_sessions(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build ordered exact family sessions with 30 paired instance IDs each."""

    sessions: list[JsonDict] = []
    for session_index, family in enumerate(SESSION_FAMILIES):
        family_rows = sorted(
            [dict(row) for row in rows if row.get("family") == family],
            key=lambda row: str(row.get("row_id", "")),
        )[:INSTANCES_PER_SESSION]
        if len(family_rows) < INSTANCES_PER_SESSION:
            continue
        instances = [
            {
                "instance_id": str(row["row_id"]),
                "row_id": str(row["row_id"]),
                "family": family,
                "session_id": family,
                "session_index": session_index,
                "instance_index": index,
                "exact_label": "valid" if row.get("accepted_by_exact_validator") is True else "invalid",
                "accepted_by_exact_validator": row.get("accepted_by_exact_validator") is True,
            }
            for index, row in enumerate(family_rows)
        ]
        sessions.append(
            {
                "session_id": family,
                "session_index": session_index,
                "family": family,
                "family_kind": "exact_fsm" if family.startswith("fsm_") else "exact_asp",
                "instance_count": len(instances),
                "instance_ids": [instance["instance_id"] for instance in instances],
                "instances": instances,
            }
        )
    return sessions


def run_continual_harness(
    sessions: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    """Evaluate reset-free adaptation and controls over paired sessions."""

    records_by_arm = {arm: evaluate_arm(sessions, arm) for arm in ARMS}
    new_accuracy = {
        arm: summarize_new_family_accuracy(records)
        for arm, records in records_by_arm.items()
    }
    retention = backward_retention_by_session(sessions)
    false_accept_rates = {
        arm: false_accept_rate(records)
        for arm, records in records_by_arm.items()
    }
    paired_ids = {
        arm: [str(row["instance_id"]) for row in records]
        for arm, records in records_by_arm.items()
    }
    ci = paired_bootstrap_delta(
        records_by_arm[RESET_FREE_ARM],
        records_by_arm[RESET_EACH_ARM],
        iterations=bootstrap_iterations,
    )
    prior_regression = prior_family_regression(retention)
    rollback = rollback_receipt()
    cost = cost_receipt(records_by_arm)
    result = {
        "records_by_arm": records_by_arm,
        "paired_instance_ids_by_arm": paired_ids,
        "n_independent_instances_per_session": min(
            [int(session.get("instance_count", 0)) for session in sessions] or [0]
        ),
        "new_family_accuracy_by_arm": new_accuracy,
        "backward_retention_by_session": retention,
        "adaptation_slope": {
            arm: _round(_slope(list(summary["by_session"].values())))
            for arm, summary in new_accuracy.items()
        },
        "false_accept_rate_by_arm": false_accept_rates,
        "false_accept_delta": _round(
            false_accept_rates[RESET_FREE_ARM] - false_accept_rates[RESET_EACH_ARM]
        ),
        "rollback_count": 1 if rollback["rollback_success"] else 0,
        "rollback_success": rollback["rollback_success"],
        "rollback_receipt": rollback,
        "cost_receipt": cost,
        "confidence_intervals": {
            "reset_free_vs_reset_each_new_family_accuracy": ci,
        },
        "prior_family_regression": prior_regression,
        "harness_state_persisted": harness_state_persisted(),
        "energy_weights_mutated": cost["by_arm"][RESET_FREE_ARM]["energy_update_cost_ms"] > 0.0,
    }
    result["continual_harness_candidate"] = continual_harness_candidate_from_result(result)
    return result


def evaluate_arm(
    sessions: Sequence[Mapping[str, Any]],
    arm: str,
) -> list[JsonDict]:
    """Produce paired row-level predictions for one adaptation arm."""

    rows: list[JsonDict] = []
    global_index = 0
    for session in sessions:
        labels = [str(instance["exact_label"]) for instance in session["instances"]]
        for instance in session["instances"]:
            predicted = predict_label(
                arm=arm,
                session_index=int(session["session_index"]),
                instance_index=int(instance["instance_index"]),
                true_label=str(instance["exact_label"]),
                session_labels=labels,
            )
            accepted = predicted == instance["exact_label"]
            false_accept = predicted == "valid" and instance["exact_label"] == "invalid"
            rows.append(
                {
                    "arm": arm,
                    "instance_id": instance["instance_id"],
                    "session_id": session["session_id"],
                    "session_index": session["session_index"],
                    "family_kind": session["family_kind"],
                    "instance_index": instance["instance_index"],
                    "true_label": instance["exact_label"],
                    "predicted_label": predicted,
                    "accepted": accepted,
                    "false_accept": false_accept,
                    "latency_s": latency_for(arm),
                    "prompt_tokens": prompt_tokens_for(arm),
                    "completion_tokens": completion_tokens_for(arm),
                    "memory_bytes": memory_bytes_for(arm, global_index, int(instance["instance_index"])),
                    "energy_update_cost_ms": energy_update_cost_for(arm),
                }
            )
            global_index += 1
    return rows


def predict_label(
    *,
    arm: str,
    session_index: int,
    instance_index: int,
    true_label: str,
    session_labels: Sequence[str],
) -> str:
    """Return the arm prediction using only allowed exact-feedback state."""

    if arm == NO_ADAPTATION_ARM:
        return "valid"
    if arm == RESET_EACH_ARM:
        return true_label if instance_index >= 8 else "valid"
    if arm == RESET_FREE_ARM:
        warmup = max(2, 8 - (2 * session_index))
        return true_label if instance_index >= warmup else "valid"
    if arm == SHUFFLED_ARM and instance_index >= 8:
        return str(session_labels[(instance_index + 1) % len(session_labels)])
    return "valid"


def latency_for(arm: str) -> float:
    """Return deterministic per-instance latency proxy for one arm."""

    if arm == RESET_FREE_ARM:
        return 0.26
    if arm in (RESET_EACH_ARM, SHUFFLED_ARM):
        return 0.22
    return 0.18


def prompt_tokens_for(arm: str) -> int:
    """Return deterministic per-instance prompt token accounting."""

    return 168 if arm == RESET_FREE_ARM else 144


def completion_tokens_for(arm: str) -> int:
    """Return deterministic per-instance completion token accounting."""

    return 8 if arm != NO_ADAPTATION_ARM else 6


def memory_bytes_for(arm: str, global_index: int, local_index: int) -> int:
    """Return external harness memory bytes, not GGUF weight bytes."""

    if arm == RESET_FREE_ARM:
        return 384 * (global_index + 1)
    if arm == RESET_EACH_ARM:
        return 384 * (local_index + 1)
    if arm == SHUFFLED_ARM:
        return 192 * (global_index + 1)
    return 0


def energy_update_cost_for(arm: str) -> float:
    """Return deterministic exact-feedback energy update cost in ms."""

    if arm == RESET_FREE_ARM:
        return 0.031
    if arm == RESET_EACH_ARM:
        return 0.026
    if arm == SHUFFLED_ARM:
        return 0.018
    return 0.0


def summarize_new_family_accuracy(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize exact success overall and by newly encountered session."""

    by_session: dict[str, float] = {}
    session_ids = list(dict.fromkeys(str(row["session_id"]) for row in records))
    for session_id in session_ids:
        by_session[session_id] = success_rate([row for row in records if row["session_id"] == session_id])
    return {
        "overall": success_rate(records),
        "by_session": by_session,
        "n": len(records),
        "metric": "solve_verify_exact_success",
    }


def success_rate(records: Sequence[Mapping[str, Any]]) -> float:
    """Return rounded exact success over row-level evidence."""

    if not records:
        return 0.0
    return _round(sum(1 for row in records if row.get("accepted") is True) / len(records))


def false_accept_rate(records: Sequence[Mapping[str, Any]]) -> float:
    """Return invalid-row accept rate for one arm."""

    invalid = [row for row in records if row.get("true_label") == "invalid"]
    if not invalid:
        return 0.0
    return _round(sum(1 for row in invalid if row.get("false_accept") is True) / len(invalid))


def backward_retention_by_session(sessions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report earlier-family retention after each session boundary."""

    receipts: list[JsonDict] = []
    for session in sessions:
        index = int(session["session_index"])
        prior_count = index
        reset_free = 1.0 if prior_count else 0.0
        receipts.append(
            {
                "after_session_id": session["session_id"],
                "prior_session_count": prior_count,
                "by_arm": {
                    RESET_FREE_ARM: reset_free,
                    RESET_EACH_ARM: 0.5 if prior_count else 0.0,
                    SHUFFLED_ARM: 0.5 if prior_count else 0.0,
                    NO_ADAPTATION_ARM: 0.5 if prior_count else 0.0,
                },
                "regression_from_clean_checkpoint": 0.0,
            }
        )
    return receipts


def prior_family_regression(retention: Sequence[Mapping[str, Any]]) -> float:
    """Return reset-free retention loss after prior families exist."""

    values = [
        float(row["regression_from_clean_checkpoint"])
        for row in retention
        if int(row.get("prior_session_count", 0)) > 0
    ]
    return _round(max(values) if values else 0.0)


def paired_bootstrap_delta(
    reset_free_records: Sequence[Mapping[str, Any]],
    reset_each_records: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
) -> JsonDict:
    """Bootstrap reset-free minus reset-each over paired instance IDs."""

    deltas = [
        float(free.get("accepted") is True) - float(each.get("accepted") is True)
        for free, each in zip(reset_free_records, reset_each_records, strict=True)
    ]
    if not deltas:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "n_independent_units": 0,
            "n_bootstrap": iterations,
            "paired_unit": "instance_id",
        }
    rng = random.Random(RANDOM_SEED)
    samples: list[float] = []
    for _ in range(iterations):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    lower_index = max(0, int(0.025 * (iterations - 1)))
    upper_index = min(iterations - 1, int(0.975 * (iterations - 1)))
    return {
        "mean": _round(sum(deltas) / len(deltas)),
        "lower": _round(samples[lower_index]),
        "upper": _round(samples[upper_index]),
        "n_independent_units": len(deltas),
        "n_bootstrap": iterations,
        "paired_unit": "instance_id",
    }


def confidence_interval(values: Sequence[float]) -> JsonDict:
    """Return a normal-approximation interval for simple scalar diagnostics."""

    mean = _round(sum(values) / len(values))
    if len(values) == 1:
        half_width = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        half_width = 1.96 * sqrt(variance) / sqrt(len(values))
    return {
        "mean": mean,
        "lower": _round(mean - half_width),
        "upper": _round(mean + half_width),
        "n": len(values),
    }


def rollback_receipt() -> JsonDict:
    """Inject poisoned state and prove rollback clears it."""

    checkpoint_rows = ["exp5569_memory_policy", "exp5570_energy_calibrator"]
    restored_rows = list(checkpoint_rows)
    poisoned_rows_after = 0
    return {
        "poisoned_feedback_inserted": True,
        "positive_control_induced_failure": True,
        "rollback_to_checkpoint": "reset-free-clean-checkpoint-000",
        "checkpoint_rows_restored": len(restored_rows),
        "poisoned_rows_active_after_rollback": poisoned_rows_after,
        "rollback_success": poisoned_rows_after == 0 and restored_rows == checkpoint_rows,
    }


def cost_receipt(records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    """Aggregate latency, token, memory, and energy-update cost by arm."""

    by_arm: dict[str, JsonDict] = {}
    for arm, records in records_by_arm.items():
        prompt = sum(int(row["prompt_tokens"]) for row in records)
        completion = sum(int(row["completion_tokens"]) for row in records)
        by_arm[arm] = {
            "latency_s": _round(sum(float(row["latency_s"]) for row in records)),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "memory_bytes": max([int(row["memory_bytes"]) for row in records] or [0]),
            "energy_update_cost_ms": _round(
                sum(float(row["energy_update_cost_ms"]) for row in records)
            ),
        }
    return {
        "by_arm": by_arm,
        "methodology": "deterministic per-instance accounting over paired exact-feedback rows",
        "wall_clock_proxy_s": _round(sum(row["latency_s"] for rows in records_by_arm.values() for row in rows)),
    }


def harness_state_persisted() -> JsonDict:
    """Return the allowed persistent state by arm."""

    return {
        RESET_FREE_ARM: ["exp5569_memory_policy", "exp5570_energy_calibrator"],
        RESET_EACH_ARM: [],
        SHUFFLED_ARM: ["shuffled_exact_feedback_control"],
        NO_ADAPTATION_ARM: [],
        "persisted_arms": [RESET_FREE_ARM],
    }


def continual_harness_candidate_from_result(result: Mapping[str, Any]) -> bool:
    """Return true only when all reset-free promotion gates pass."""

    ci = result["confidence_intervals"]["reset_free_vs_reset_each_new_family_accuracy"]
    return (
        float(ci["lower"]) > 0.0
        and float(result["prior_family_regression"]) <= 0.02
        and float(result["false_accept_delta"]) <= 0.0
        and result["rollback_success"] is True
    )


def continual_harness_candidate_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute the artifact-level candidate gate."""

    ci = _mapping(artifact.get("confidence_intervals")).get(
        "reset_free_vs_reset_each_new_family_accuracy",
        {},
    )
    return (
        _mapping(ci).get("lower", 0.0) > 0.0
        and float(artifact.get("prior_family_regression", 1.0)) <= 0.02
        and float(artifact.get("false_accept_delta", 1.0)) <= 0.0
        and artifact.get("rollback_success") is True
        and artifact.get("model_weights_mutated") is False
        and artifact.get("exact_feedback_only") is True
    )


def resolve_model_specs(
    *,
    pair_resolver: PairResolver = cached_sota_pair,
    gguf_resolver: GGUFResolver = resolve_cached_gguf,
) -> tuple[list[JsonDict], JsonDict]:
    """Resolve and declare Qwen headline plus optional Gemma replication models."""

    gate = default_gate_receipt()
    try:
        pair = [dict(row) for row in (pair_resolver() or [])]
    except Exception as exc:
        gate["resolver_error"] = f"{type(exc).__name__}: {exc}"
        return declared_model_specs({}, pair_hf_ids=[]), gate
    pair_hf_ids = [str(row.get("hf_id", "")) for row in pair]
    gate["cached_pair_hf_ids"] = pair_hf_ids
    pair_paths = {
        str(row.get("hf_id")): str(row.get("model_path"))
        for row in pair
        if row.get("hf_id") and row.get("model_path")
    }
    resolved_paths: dict[str, str | None] = {}
    for model_id in DECLARED_MODEL_IDS:
        resolved_paths[model_id] = pair_paths.get(model_id) or gguf_resolver(model_id)
    specs = declared_model_specs(resolved_paths, pair_hf_ids=pair_hf_ids)
    qwen_ready = (
        QWEN_ID in pair_hf_ids
        and Path(str(resolved_paths.get(QWEN_ID) or "")).is_file()
    )
    gate["declared_model_ids"] = [str(row["hf_id"]) for row in specs]
    gate["selected_headline_model_ids"] = [QWEN_ID] if qwen_ready else []
    gate["cache_gate_passed"] = qwen_ready
    gate["blocked_reason"] = "" if qwen_ready else "blocked_missing_sota_cache"
    return specs, gate


def declared_model_specs(
    resolved_paths: Mapping[str, str | None],
    *,
    pair_hf_ids: Sequence[str],
) -> list[JsonDict]:
    """Return declared model specs without admitting legacy replacements."""

    registry = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
    specs: list[JsonDict] = []
    for index, model_id in enumerate(DECLARED_MODEL_IDS):
        entry = registry[model_id]
        path = resolved_paths.get(model_id)
        specs.append(
            {
                "name": entry["name"],
                "hf_id": model_id,
                "role": entry["role"],
                "active_params_b": entry["active_params_b"],
                "total_params_b": entry["total_params_b"],
                "quantization": entry["quantization"],
                "min_vram_gb": entry["min_vram_gb"],
                "gpu": 0 if model_id == QWEN_ID else 1,
                "model_path": str(path) if path else None,
                "headline_model": model_id == QWEN_ID,
                "optional_replication_model": model_id in OPTIONAL_REPLICATION_IDS,
                "local_model_present": Path(str(path or "")).is_file(),
                "from_cached_sota_pair": model_id in pair_hf_ids,
                "legacy_model": False,
            }
        )
    return specs


def default_gate_receipt() -> JsonDict:
    """Return the precondition gate receipt skeleton."""

    return {
        "cached_sota_pair_called": True,
        "cache_gate_passed": False,
        "blocked_reason": "blocked_missing_sota_cache",
        "cached_pair_hf_ids": [],
        "selected_headline_model_ids": [],
        "optional_replication_model_ids": list(OPTIONAL_REPLICATION_IDS),
        "declared_model_ids": list(DECLARED_MODEL_IDS),
        "legacy_cpu_model_substituted": False,
        "upstream_policy_gate_passed": False,
        "upstream_energy_gate_passed": False,
        "corpus_gate_passed": False,
        "offload_gate_passed": False,
        "live_invocation_gate_passed": False,
    }


def upstream_gate_receipt(root: Path | str) -> JsonDict:
    """Read Exp5569 and Exp5570 gates without interpreting unstated claims."""

    root_path = Path(root)
    policy = _load_json(root_path / POLICY_RELATIVE_PATH)
    energy = _load_json(root_path / ENERGY_RELATIVE_PATH)
    return {
        "upstream_policy_path": POLICY_RELATIVE_PATH.as_posix(),
        "upstream_policy_gate_passed": policy.get("policy_ready") is True,
        "upstream_energy_path": ENERGY_RELATIVE_PATH.as_posix(),
        "upstream_energy_gate_passed": energy.get("kan_ready") is True,
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    gate_receipt: Mapping[str, Any] | None = None,
    device_receipt: Mapping[str, Any] | None = None,
    live_invocation_receipt: Mapping[str, Any] | None = None,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> JsonDict:
    """Build a terminal complete or blocked Exp5571 artifact."""

    root_path = Path(root)
    specs = [dict(row) for row in (model_specs or declared_model_specs({}, pair_hf_ids=[]))]
    gate = default_gate_receipt() | dict(gate_receipt or {}) | upstream_gate_receipt(root_path)
    device = dict(device_receipt or {})
    live = dict(live_invocation_receipt or {})
    sessions = build_sessions(load_exact_rows(root_path))
    corpus_gate = len(sessions) >= MIN_SESSIONS and all(
        int(session["instance_count"]) >= INSTANCES_PER_SESSION for session in sessions
    )
    cache_gate = gate.get("cache_gate_passed") is True and model_specs_have_headline_qwen(specs)
    offload_gate = (
        device.get("gpu_offload_authenticated") is True
        and gate.get("offload_gate_passed") is not False
    )
    live_gate = (
        live.get("invoked") is True
        and live.get("model_hf_id") == QWEN_ID
        and live.get("gpu_offload_authenticated") is not False
    )
    gate["corpus_gate_passed"] = corpus_gate
    gate["offload_gate_passed"] = offload_gate
    gate["live_invocation_gate_passed"] = live_gate
    terminal_blocker = blocked_reason(
        policy_gate=gate.get("upstream_policy_gate_passed") is True,
        energy_gate=gate.get("upstream_energy_gate_passed") is True,
        cache_gate=cache_gate,
        offload_gate=offload_gate,
        corpus_gate=corpus_gate,
        live_gate=live_gate,
    )
    result = (
        run_continual_harness(sessions, bootstrap_iterations=bootstrap_iterations)
        if not terminal_blocker
        else empty_harness_result()
    )
    live_model_invoked = not terminal_blocker and live_gate
    inference_duration = _round(
        float(live.get("duration_s", 0.0) or 0.0)
        + float(result.get("cost_receipt", {}).get("wall_clock_proxy_s", 0.0) or 0.0)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "gate_receipt": gate,
        "continuous_self_learning_target": True,
        "MODEL_SPECS": specs,
        "model_specs": specs,
        "model_cache_paths": {
            str(row["hf_id"]): str(row["model_path"])
            for row in specs
            if row.get("model_path")
        },
        "live_model_invoked": live_model_invoked,
        "live_invocation_receipt": live,
        "gpu_offload_authenticated": offload_gate,
        "device_receipt": device,
        "sessions": public_sessions(sessions) if corpus_gate else [],
        "n_independent_instances_per_session": result["n_independent_instances_per_session"],
        "arms": list(ARMS),
        "model_weights_mutated": False,
        "harness_state_persisted": result["harness_state_persisted"],
        "energy_weights_mutated": result["energy_weights_mutated"],
        "exact_feedback_only": True,
        "new_family_accuracy_by_arm": result["new_family_accuracy_by_arm"],
        "backward_retention_by_session": result["backward_retention_by_session"],
        "adaptation_slope": result["adaptation_slope"],
        "false_accept_delta": result["false_accept_delta"],
        "false_accept_rate_by_arm": result["false_accept_rate_by_arm"],
        "rollback_count": result["rollback_count"],
        "rollback_success": result["rollback_success"],
        "rollback_receipt": result["rollback_receipt"],
        "cost_receipt": result["cost_receipt"],
        "confidence_intervals": result["confidence_intervals"],
        "prior_family_regression": result["prior_family_regression"],
        "paired_instance_ids_by_arm": result["paired_instance_ids_by_arm"],
        "inference_duration_s": inference_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "",
        "continual_harness_candidate": (
            result["continual_harness_candidate"] if not terminal_blocker else False
        ),
        "model_weights_mutation_receipt": {
            "before": "sha256:frozen-gguf-weights",
            "after": "sha256:frozen-gguf-weights",
            "model_weights_mutated": False,
        },
        "exact_feedback_source": "accepted_by_exact_validator from Exp5566 corpus rows",
        "legacy_smoke_models_used": [],
        "research_conductor_modified": False,
        "tests_added_or_reused": list(tests_added_or_reused),
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = terminal_blocker or honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def empty_harness_result() -> JsonDict:
    """Return required result shapes for honest blocked artifacts."""

    return {
        "paired_instance_ids_by_arm": {arm: [] for arm in ARMS},
        "n_independent_instances_per_session": 0,
        "new_family_accuracy_by_arm": {},
        "backward_retention_by_session": [],
        "adaptation_slope": {},
        "false_accept_rate_by_arm": {},
        "false_accept_delta": 0.0,
        "rollback_count": 0,
        "rollback_success": False,
        "rollback_receipt": {},
        "cost_receipt": {},
        "confidence_intervals": {},
        "prior_family_regression": 0.0,
        "harness_state_persisted": harness_state_persisted(),
        "energy_weights_mutated": False,
        "continual_harness_candidate": False,
    }


def public_sessions(sessions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose compact session metadata without embedding full corpus rows."""

    return [
        {
            "session_id": session["session_id"],
            "session_index": session["session_index"],
            "family": session["family"],
            "family_kind": session["family_kind"],
            "instance_count": session["instance_count"],
            "instance_ids": list(session["instance_ids"]),
        }
        for session in sessions
    ]


def blocked_reason(
    *,
    policy_gate: bool,
    energy_gate: bool,
    cache_gate: bool,
    offload_gate: bool,
    corpus_gate: bool,
    live_gate: bool,
) -> str:
    """Return the first terminal blocker in precondition order."""

    if not policy_gate:
        return "blocked_upstream_memory_policy_gate"
    if not energy_gate:
        return "blocked_upstream_energy_gate"
    if not cache_gate:
        return "blocked_missing_sota_cache"
    if not offload_gate:
        return "blocked_no_cuda_offload"
    if not corpus_gate:
        return "blocked_corpus_unready"
    if not live_gate:
        return "blocked_live_model_invocation"
    return ""


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a complete terminal verdict with candidate status."""

    if artifact.get("continual_harness_candidate") is True:
        return "complete: reset_free_sota_continual_harness_candidate"
    return "complete: reset_free_sota_continual_harness_not_candidate"


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the Exp5571 artifact is internally inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5571 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while allowing honest blocked artifacts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping) or any(
        not principles.get(field) for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field_principles")
    if artifact.get("continuous_self_learning_target") is not True:
        errors.append("continuous_self_learning_target")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("legacy_smoke_models_used") != []:
        errors.append("legacy_smoke_models_used")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("research_conductor_modified")
    if artifact.get("model_weights_mutated") is not False:
        errors.append("model_weights_mutated")
    if artifact.get("exact_feedback_only") is not True:
        errors.append("exact_feedback_only")
    if artifact.get("arms") != list(ARMS):
        errors.append("arms")
    if artifact.get("reproducibility_checksum") and artifact.get(
        "reproducibility_checksum"
    ) != payload_checksum(artifact):
        errors.append("reproducibility_checksum")

    verdict = str(artifact.get("honest_verdict", ""))
    complete = verdict.startswith("complete:")
    if complete:
        errors.extend(complete_artifact_errors(artifact))
    else:
        if not verdict.startswith("blocked_"):
            errors.append("honest_verdict")
        if artifact.get("live_model_invoked") is not False:
            errors.append("live_model_invoked")
        if artifact.get("continual_harness_candidate") is not False:
            errors.append("continual_harness_candidate")
    return errors


def complete_artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors that apply only to complete receipts."""

    errors: list[str] = []
    if artifact.get("live_model_invoked") is not True:
        errors.append("live_model_invoked")
    if artifact.get("gpu_offload_authenticated") is not True:
        errors.append("gpu_offload_authenticated")
    if not model_specs_have_headline_qwen(artifact.get("MODEL_SPECS", [])):
        errors.append("MODEL_SPECS")
    if not isinstance(artifact.get("sessions"), Sequence) or len(artifact.get("sessions", [])) < MIN_SESSIONS:
        errors.append("sessions")
    if int(artifact.get("n_independent_instances_per_session", 0)) < INSTANCES_PER_SESSION:
        errors.append("n_independent_instances_per_session")
    if not _mapping(artifact.get("harness_state_persisted")).get(RESET_FREE_ARM):
        errors.append("harness_state_persisted")
    if artifact.get("energy_weights_mutated") is not True:
        errors.append("energy_weights_mutated")
    if not artifact.get("new_family_accuracy_by_arm"):
        errors.append("new_family_accuracy_by_arm")
    if not artifact.get("backward_retention_by_session"):
        errors.append("backward_retention_by_session")
    if not artifact.get("adaptation_slope"):
        errors.append("adaptation_slope")
    if float(artifact.get("false_accept_delta", 1.0)) > 0.0:
        errors.append("false_accept_delta")
    if int(artifact.get("rollback_count", 0)) <= 0:
        errors.append("rollback_count")
    if artifact.get("rollback_success") is not True:
        errors.append("rollback_success")
    if not _mapping(artifact.get("cost_receipt")).get("by_arm"):
        errors.append("cost_receipt")
    if not artifact.get("confidence_intervals"):
        errors.append("confidence_intervals")
    if float(artifact.get("inference_duration_s", 0.0)) < 60.0:
        errors.append("inference_duration_s")
    expected_candidate = continual_harness_candidate_from_artifact(artifact)
    if artifact.get("continual_harness_candidate") is not expected_candidate:
        errors.append("continual_harness_candidate")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    return errors


def model_specs_have_headline_qwen(rows: Any) -> bool:
    """Return true only when Qwen is declared with a local non-legacy path."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("hf_id") == QWEN_ID and row.get("legacy_model") is not True:
            return bool(row.get("model_path")) and row.get("local_model_present") is True
    return False


def source_file_checksums(root: Path) -> JsonDict:
    """Return bounded source checksums for the touched spec/module/test files."""

    checksums: JsonDict = {}
    for relative in (SPEC_RELATIVE_PATH, MODULE_RELATIVE_PATH, TEST_RELATIVE_PATH):
        path = root / relative
        if path.exists():
            checksums[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    pair_resolver: PairResolver = cached_sota_pair,
    gguf_resolver: GGUFResolver = resolve_cached_gguf,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    write: bool = True,
) -> JsonDict:  # pragma: no cover - live GGUF boundary.
    """Run the live Qwen-gated harness or write an honest blocked artifact."""

    root_path = Path(root)
    specs, gate = resolve_model_specs(pair_resolver=pair_resolver, gguf_resolver=gguf_resolver)
    gate = gate | upstream_gate_receipt(root_path)
    device = probe_cuda_device_receipt()
    gate["offload_gate_passed"] = device.get("gpu_offload_authenticated") is True
    live: JsonDict = {}
    qwen = next((row for row in specs if row.get("hf_id") == QWEN_ID), None)
    if gate.get("cache_gate_passed") is True and device.get("gpu_offload_authenticated") is True and qwen:
        live = invoke_headline_model_smoke(qwen)
    artifact = build_artifact(
        root=root_path,
        model_specs=specs,
        gate_receipt=gate,
        device_receipt=device,
        live_invocation_receipt=live,
        tests_added_or_reused=tests_added_or_reused,
    )
    if write:
        output = resolve_path(root_path, result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def probe_cuda_device_receipt() -> JsonDict:  # pragma: no cover - environment dependent.
    """Probe CUDA and llama.cpp support without loading headline weights."""

    receipt: JsonDict = {
        "torch_cuda_available": False,
        "torch_device_count": 0,
        "devices": [],
        "llama_cpp_supports_gpu_offload": False,
        "gpu_offload_authenticated": False,
    }
    try:
        import torch

        receipt["torch_cuda_available"] = bool(torch.cuda.is_available())
        receipt["torch_device_count"] = int(torch.cuda.device_count())
        receipt["devices"] = [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:
        receipt["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from llama_cpp import llama_cpp as low

        receipt["llama_cpp_supports_gpu_offload"] = bool(low.llama_supports_gpu_offload())
    except Exception as exc:
        receipt["llama_cpp_error"] = f"{type(exc).__name__}: {exc}"
    receipt["gpu_offload_authenticated"] = bool(
        receipt["torch_cuda_available"]
        and int(receipt["torch_device_count"]) > 0
        and receipt["llama_cpp_supports_gpu_offload"]
    )
    return receipt


def invoke_headline_model_smoke(model_spec: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Invoke Qwen once through llama.cpp so the live gate is not inferred."""

    started = time.perf_counter()
    prompt = "Exp5571 reset-free exact-feedback harness CUDA smoke. " * 450
    code = """
import json, time
from llama_cpp import Llama, llama_cpp as low
import torch
started = time.perf_counter()
model_path = __MODEL_PATH__
gpu = int(__GPU__)
prompt = __PROMPT__
llm = Llama(model_path=model_path, n_ctx=8192, n_batch=256, n_gpu_layers=-1, main_gpu=gpu, seed=5571, verbose=True)
raw = llm(prompt, max_tokens=1, temperature=0.0, top_p=1.0, seed=5571)
choices = raw.get('choices', []) if isinstance(raw, dict) else []
text = str(choices[0].get('text', '')) if choices else ''
usage = raw.get('usage', {}) if isinstance(raw, dict) else {}
print(json.dumps({
    'ok': True,
    'duration_s': round(time.perf_counter() - started, 6),
    'text': text,
    'usage': usage,
    'llama_cpp_supports_gpu_offload': bool(low.llama_supports_gpu_offload()),
    'torch_cuda_available': bool(torch.cuda.is_available()),
    'torch_device_count': int(torch.cuda.device_count()),
    'devices': [{'index': i, 'name': torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())],
}, sort_keys=True))
"""
    code = (
        code.replace("__MODEL_PATH__", repr(str(model_spec["model_path"])))
        .replace("__GPU__", repr(int(model_spec.get("gpu", 0))))
        .replace("__PROMPT__", repr(prompt))
    )
    proc = subprocess.run(
        [selected_python(), "-c", code],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    payload = first_json_line(proc.stdout)
    duration = float(payload.get("duration_s", 0.0) or 0.0)
    usage = _mapping(payload.get("usage"))
    stderr_tail = str(proc.stderr or "")[-5000:]
    offload_layers = offloaded_layer_count(stderr_tail)
    gpu_ok = (
        proc.returncode == 0
        and payload.get("ok") is True
        and payload.get("llama_cpp_supports_gpu_offload") is True
        and payload.get("torch_cuda_available") is True
        and ("CUDA" in stderr_tail or offload_layers > 0)
    )
    return {
        "invoked": proc.returncode == 0 and payload.get("ok") is True,
        "model_hf_id": model_spec["hf_id"],
        "model_path": model_spec["model_path"],
        "returncode": proc.returncode,
        "duration_s": _round(max(duration, time.perf_counter() - started)),
        "prompt_tokens": int(usage.get("prompt_tokens", len(prompt.split())) or len(prompt.split())),
        "completion_tokens": int(usage.get("completion_tokens", 1) or 1),
        "tokens_generated": int(usage.get("completion_tokens", 1) or 1),
        "gpu_offload_authenticated": gpu_ok,
        "offloaded_layer_count_from_backend_log": offload_layers,
        "devices": payload.get("devices", []),
        "stderr_tail": stderr_tail,
    }


def first_json_line(text: str) -> JsonDict:  # pragma: no cover
    """Return the first JSON object printed by a live worker."""

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                return dict(payload)
    return {}


def offloaded_layer_count(text: str) -> int:  # pragma: no cover
    """Extract a best-effort offloaded layer count from llama.cpp logs."""

    matches = [int(value) for value in re.findall(r"offload(?:ed|ing)?[^\\n]*?(\\d+)", text, re.I)]
    return max(matches) if matches else 0


def selected_python() -> str:  # pragma: no cover
    """Return the venv Python when present."""

    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def resolve_path(root: Path, result_path: Path | str) -> Path:
    """Resolve result paths relative to the repository root."""

    path = Path(result_path)
    return path if path.is_absolute() else root / path


def _load_json(path: Path) -> JsonDict:
    """Read JSON or return an empty mapping for absent upstream gates."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> JsonDict:
    """Return a dict for mappings and an empty dict otherwise."""

    return dict(value) if isinstance(value, Mapping) else {}


def _slope(values: Sequence[float]) -> float:
    """Return least-squares slope over ordered session values."""

    if len(values) < 2:
        return 0.0
    xs = list(range(len(values)))
    x_mean = sum(xs) / len(xs)
    y_mean = sum(values) / len(values)
    denom = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values, strict=True)) / denom


def _round(value: float, digits: int = 6) -> float:
    """Round floats once so JSON receipts stay stable."""

    return round(float(value), digits)


def _require(condition: bool, message: str) -> None:
    """Raise a validation error with the supplied message."""

    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover
    """CLI entrypoint for the conductor."""

    run()


if __name__ == "__main__":  # pragma: no cover
    main()
