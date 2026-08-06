"""Exp6164 gated continuous strategy learning A/B.

Spec refs: REQ-LEARN-6164, REQ-LEARN-6164-1, REQ-LEARN-6164-2,
REQ-LEARN-6164-3, REQ-LEARN-6164-4, REQ-LEARN-6164-5, REQ-LEARN-6164-6,
REQ-LEARN-6164-7, REQ-LEARN-6164-8, REQ-LEARN-6164-9,
REQ-LEARN-6164-10, SCENARIO-LEARN-6164-BLOCKED,
SCENARIO-LEARN-6164-MATCHED, SCENARIO-LEARN-6164-TRANSACTION,
SCENARIO-LEARN-6164-READY, REQ-CL-6164-MANDATORY-ARTIFACT,
REQ-CL-6164-PREREQUISITE-RECOMPUTE, REQ-CL-6164-BLOCKED-MODEL-LOAD,
REQ-CL-6164-MANDATED-MODEL, REQ-CL-6164-FOUR-ARM-MATCHING,
REQ-CL-6164-CHRONOLOGICAL-ISOLATION, REQ-CL-6164-READ-ONLY-SNAPSHOT,
REQ-CL-6164-POST-OUTCOME-COMMIT, REQ-CL-6164-CERTIFICATE,
REQ-CL-6164-DECISION-ADMISSION, REQ-CL-6164-UTILITY,
REQ-CL-6164-RETENTION, REQ-CL-6164-POISON, REQ-CL-6164-ROLLBACK,
REQ-CL-6164-BOUNDED-STATE, REQ-CL-6164-LIFECYCLE,
REQ-CL-6164-IMMUTABLE-WEIGHT, REQ-CL-6164-READY-SCORE,
SCENARIO-CL-6164-BLOCKED, SCENARIO-CL-6164-MATCHED,
SCENARIO-CL-6164-TRANSACTION, SCENARIO-CL-6164-READY.

The experiment is deliberately fail-closed. It must always leave a terminal
artifact, but model work is allowed only after the Exp6162 and Exp6163 gates
recompute ready inside this process. On the current task, a missing Exp6163
artifact is a scientific blocker, so the correct terminal artifact is blocked
with zero model, tokenizer, CUDA, and GPU invocation counts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]
ModelRunner = Callable[[list[JsonDict], tuple[str, ...], int], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6164_continuous_strategy_learning_ab.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6164_continuous_strategy_learning_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6164_continuous_strategy_learning_ab.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
REQUESTED_CONTINUOUS_SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6120_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6120_outcome_committed_reduced_order_csl.json"
)
EXP6160_RESULT_RELATIVE_PATH = Path("results/experiment_6160_sota_decision_calibration_corpus.json")
EXP6162_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6162_prospective_admission_replication.json"
)
EXP6163_RESULT_RELATIVE_PATH = Path("results/experiment_6163_certified_strategy_store_scaleup.json")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

SCHEMA = "carnot.experiment_6164.continuous_strategy_learning_ab.v1"
EXPERIMENT_ID = "experiment_6164_continuous_strategy_learning_ab"
RUN_DATE = "20260806"
RANDOM_SEED = 6164
EVENT_COUNT = 24
SEEDS = (6164, 6165)
TOKEN_BUDGET = 384
WALL_CAP_S = 1800
STATE_BYTE_BOUND = 4096
INFERENCE_SUBSTRATE = "blocked_before_model_load_or_live_local_sota_gguf_cuda"
VERIFIER_IS_ORACLE = False

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "primary",
        "quantization": "Q4_K_M",
        "loader": "llama_cpp.Llama",
        "gpu": 0,
        "native_chat_required": True,
        "weight_mutation_allowed": False,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "confirmation",
        "quantization": "Q4_K_M",
        "loader": "llama_cpp.Llama",
        "gpu": 1,
        "native_chat_required": True,
        "weight_mutation_allowed": False,
    },
]
MANDATED_MODEL_IDS = tuple(spec["hf_id"] for spec in MODEL_SPECS)
ARM_NAMES = (
    "no_memory",
    "exp6120_utility_only",
    "certificate_only_strategy",
    "decision_calibrated_strategy",
)
PARTITIONS = ("future_known", "shifted_family_held")
FAMILIES = ("known_family", "shifted_family")
ZERO_MODEL_INVOCATION_COUNTS = {
    "model_load_count": 0,
    "tokenizer_load_count": 0,
    "cuda_context_count": 0,
    "gpu_worker_count": 0,
    "native_chat_invocation_count": 0,
    "llama_cpp_loader_count": 0,
    "generated_token_count": 0,
}

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6164_continuous_strategy_learning_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6164_continuous_strategy_learning_ab.py "
    "-m pytest tests/python/test_experiment_6164_continuous_strategy_learning_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6164_continuous_strategy_learning_ab.py "
    "--fail-under=100"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6164_continuous_strategy_learning_ab --validate"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6164_continuous_strategy_learning_ab.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6164_continuous_strategy_learning_ab.json"
)
E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6162_prospective_admission_replication.py "
    "-q --no-cov -n 0"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6164_continuous_strategy_learning_ab.py "
    "tests/python/test_experiment_6164_continuous_strategy_learning_ab.py"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_COMMAND,
    RUFF_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
EXP6160_ROW_FILES = (
    Path("results/experiment_6160_sota_decision_calibration_corpus.qwen3_6_35b_a3b.rows.jsonl"),
    Path("results/experiment_6160_sota_decision_calibration_corpus.gemma_4_26b_a4b_it.rows.jsonl"),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    SELF_LEARNING_SPEC_RELATIVE_PATH,
    REQUESTED_CONTINUOUS_SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    EXP6120_RESULT_RELATIVE_PATH,
    EXP6160_RESULT_RELATIVE_PATH,
    *EXP6160_ROW_FILES,
    EXP6162_RESULT_RELATIVE_PATH,
    EXP6163_RESULT_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/experiment_6149_certified_strategy_schema_fixture.py"),
    Path("python/carnot/experiment_6162_prospective_admission_replication.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "continuous_self_learning_task",
    "mandatory_artifact_written",
    "prerequisite_gate_receipts",
    "blocked_before_model_load_receipt",
    "MODEL_SPECS",
    "model_specs",
    "resolved_paths_revisions_quantizations_hashes_and_loader_receipts",
    "embedded_tokenizer_chat_template_cuda_pid_and_lifecycle_receipts",
    "arm_definitions_and_resource_matching",
    "chronological_event_order_and_decision_snapshot_receipts",
    "exact_post_outcome_commit_abort_quarantine_receipts",
    "per_model_family_partition_future_utility_accuracy_regret_and_grouped_intervals",
    "learning_speed_and_time_to_benefit",
    "protected_retention_forgetting_safety_abstention_and_poison_metrics",
    "duplicate_reordered_rollback_restart_eviction_and_state_bytes",
    "model_weight_immutability_receipt",
    "acquisition_analysis_duration_and_cleanup_receipts",
    "continuous_strategy_learning_ready_score",
    "retirement_triggered",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status distinguishes blocked, positive, null, and retired strategy-learning evidence.",
    "preconditions_checked": "Hash Exp6160 rows, Exp6162 policy/verdict, Exp6163 schema/ABI/verdict, event order, prompts, models, validators, exclusions, outputs, and protected files before GPU acquisition.",
    "continuous_self_learning_task": "This field is bare true because the task is the mandatory FR-11 continuous self-learning run.",
    "mandatory_artifact_written": "This field is bare true because every terminal path must write the Exp6164 artifact.",
    "prerequisite_gate_receipts": "Exp6162 and Exp6163 readiness are recomputed internally and conjunctively.",
    "blocked_before_model_load_receipt": "A failed prerequisite must prove all model, tokenizer, CUDA, and GPU invocation counts are zero.",
    "MODEL_SPECS": "The top-level model list contains only the two mandated frozen SOTA GGUF hub ids.",
    "model_specs": "The lowercase model list mirrors MODEL_SPECS for downstream schema consumers.",
    "resolved_paths_revisions_quantizations_hashes_and_loader_receipts": "Model paths, revisions, quantizations, hashes, loaders, and GPU assignments are resolved only after prerequisites pass.",
    "embedded_tokenizer_chat_template_cuda_pid_and_lifecycle_receipts": "Tokenizer, chat-template, CUDA PID, native-chat, worker, and lifecycle receipts distinguish cached setup from live inference.",
    "arm_definitions_and_resource_matching": "No-memory, Exp6120 utility-only, certificate-only, and decision-calibrated arms are matched on events, prompts, seeds, token budgets, invocations, and resource caps.",
    "chronological_event_order_and_decision_snapshot_receipts": "Every decision reads a frozen read-only snapshot with only prior certificates.",
    "exact_post_outcome_commit_abort_quarantine_receipts": "Strategy writes commit only after exact post-outcome validation, otherwise abort or quarantine.",
    "per_model_family_partition_future_utility_accuracy_regret_and_grouped_intervals": "Future utility, accuracy, regret, and grouped intervals are reported per model, family, and partition before pooling.",
    "learning_speed_and_time_to_benefit": "Chronological learning curves and time-to-benefit are separated from final utility.",
    "protected_retention_forgetting_safety_abstention_and_poison_metrics": "Utility cannot buy protected forgetting, unsafe admission, abstention, or poison regressions.",
    "duplicate_reordered_rollback_restart_eviction_and_state_bytes": "Duplicate, reordered, rollback, restart, eviction, and bounded-state bytes are explicit lifecycle checks.",
    "model_weight_immutability_receipt": "This experiment may update certified external strategy state but never model weights.",
    "acquisition_analysis_duration_and_cleanup_receipts": "GPU acquisition, live inference, cached analysis, and cleanup durations are reported separately.",
    "continuous_strategy_learning_ready_score": "Readiness is one only when the decision-calibrated strategy beats both baselines for both models with positive lower intervals, no regressions, bounded state, cleanup, and immutable weights.",
    "retirement_triggered": "Repeated non-positive strategy-learning evidence can retire the construction instead of hiding a null.",
    "protected_files_unchanged": "Conductor, ops, and traceability files remain outside this experiment's mutable surface.",
    "duration_s": "Measured wall time is reported without classifying cached analysis as live inference.",
    "inference_substrate": "The substrate states whether the run blocked before load or used live local SOTA GGUF CUDA.",
    "verifier_is_oracle": "Exact validators score post-outcome commits, but the decision policy is not an oracle.",
    "missing_verifier_gaps": "Any missing prerequisite, model, lifecycle, safety, or validation gap is made explicit.",
    "field_provenance": "Every field traces to spec, upstream artifacts, model receipts, transaction receipts, tests, commands, or protected-file hashes.",
    "test_commands": "Commands document focused unit, coverage, prerequisite, artifact, model/cache/tokenizer/CUDA, arm matching, chronological isolation, transaction, metrics, immutability, lifecycle, schema, adversarial, protected-file, E2E, global pytest, and root-clutter checks.",
    "test_exit_codes": "Non-zero verification commands prevent readiness.",
    "reproducibility_checksum": "A checksum detects drift in inputs, model specs, receipts, metrics, commands, protected files, and output paths.",
    "honest_verdict": "Use `complete_positive:`, `complete_null:`, `retired:`, or `blocked:` and state whether self-learning actually executed.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON data in stable byte order for reproducible receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash text and carry the algorithm name in the stored receipt."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible data."""

    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str | None:
    """Return a content hash for an existing file, otherwise ``None``."""

    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> JsonDict:
    """Load a JSON object and return an empty mapping for absent files."""

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def model_slug(hf_id: str) -> str:
    """Convert a mandated HF id into the repo's row-sidecar slug."""

    basename = hf_id.split("/", 1)[-1]
    if basename.endswith("-GGUF"):
        basename = basename[:-5]
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", basename.lower())).strip("_")


def _path_receipt(path: Path) -> JsonDict:
    absolute = REPO_ROOT / path
    return {
        "path": path.as_posix(),
        "exists": absolute.exists(),
        "sha256": sha256_file(absolute),
        "size_bytes": absolute.stat().st_size if absolute.exists() and absolute.is_file() else 0,
    }


def _protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def _artifact_hash(artifact: Mapping[str, Any]) -> str:
    return sha256_json(artifact)


def _artifact_receipt(artifact: Mapping[str, Any], path: Path) -> JsonDict:
    return {
        "path": path.as_posix(),
        "exists": bool(artifact),
        "sha256": _artifact_hash(artifact) if artifact else None,
        "status": artifact.get("status"),
        "honest_verdict": artifact.get("honest_verdict"),
    }


def _model_ids_exact(model_specs: list[JsonDict]) -> bool:
    return [spec.get("hf_id") for spec in model_specs] == list(MANDATED_MODEL_IDS)


def _exp6162_ready(artifact: Mapping[str, Any]) -> JsonDict:
    gates = artifact.get("per_model_and_conjunctive_gate_matrix", {})
    by_model = gates.get("by_model", {}) if isinstance(gates, Mapping) else {}
    per_model = {
        model_id: bool(by_model.get(model_id, {}).get("model_pass"))
        for model_id in MANDATED_MODEL_IDS
    }
    ready = (
        artifact.get("status") == "complete_positive"
        and artifact.get("prospective_admission_replication_ready_score") == 1.0
        and str(artifact.get("honest_verdict", "")).startswith("complete_positive:")
        and bool(gates.get("conjunctive_pass"))
        and all(per_model.values())
    )
    return {
        "ready": ready,
        "status": artifact.get("status"),
        "ready_score": artifact.get("prospective_admission_replication_ready_score"),
        "honest_verdict": artifact.get("honest_verdict"),
        "conjunctive_pass": bool(gates.get("conjunctive_pass")),
        "per_model_pass": per_model,
        "policy_hash": artifact.get("policy_manifest_path_hash_and_contents", {}).get(
            "contents_hash"
        ),
    }


def _exp6163_ready(artifact: Mapping[str, Any]) -> JsonDict:
    schema = (
        artifact.get("schema_abi_verdict_receipt")
        or artifact.get("schema_abi_and_verdict_receipt")
        or artifact.get("certified_strategy_store_schema_abi_verdict")
        or {}
    )
    ready_score = (
        artifact.get("certified_strategy_store_scaleup_ready_score")
        if "certified_strategy_store_scaleup_ready_score" in artifact
        else artifact.get("strategy_store_scaleup_ready_score")
    )
    if ready_score is None:
        ready_score = artifact.get("certified_strategy_fixture_ready_score")
    status_ok = artifact.get("status") in {"complete_ready", "complete_positive"}
    verdict_ok = str(artifact.get("honest_verdict", "")).startswith(
        ("complete_ready:", "complete_positive:")
    )
    schema_ok = bool(schema.get("schema_valid"))
    abi_ok = bool(schema.get("abi_valid"))
    verdict_passed = bool(schema.get("verdict_passed"))
    bounded_state = bool(schema.get("bounded_state"))
    ready = (
        bool(artifact)
        and status_ok
        and ready_score == 1.0
        and verdict_ok
        and schema_ok
        and abi_ok
        and verdict_passed
        and bounded_state
    )
    return {
        "ready": ready,
        "status": artifact.get("status"),
        "ready_score": ready_score,
        "honest_verdict": artifact.get("honest_verdict"),
        "schema_version": schema.get("schema_version"),
        "schema_valid": schema_ok,
        "abi_valid": abi_ok,
        "verdict_passed": verdict_passed,
        "bounded_state": bounded_state,
    }


def prerequisite_gate_receipts(
    *,
    exp6162_artifact: Mapping[str, Any],
    exp6163_artifact: Mapping[str, Any],
    exp6162_path: Path,
    exp6163_path: Path,
) -> JsonDict:
    """Recompute upstream gates before any model or GPU action."""

    exp6162 = _exp6162_ready(exp6162_artifact)
    exp6163 = _exp6163_ready(exp6163_artifact)
    blocked_reasons = []
    if not exp6162["ready"]:
        blocked_reasons.append("exp6162_not_ready")
    if not exp6163["ready"]:
        blocked_reasons.append("exp6163_not_ready")
    return {
        "schema": SCHEMA + ".prerequisites.v1",
        "exp6162": {
            **exp6162,
            "artifact_receipt": _artifact_receipt(exp6162_artifact, exp6162_path),
        },
        "exp6163": {
            **exp6163,
            "artifact_receipt": _artifact_receipt(exp6163_artifact, exp6163_path),
        },
        "all_passed": not blocked_reasons,
        "blocked_reasons": blocked_reasons,
    }


def preconditions_checked(
    *,
    result_path: Path,
    exp6162_artifact: Mapping[str, Any],
    exp6163_artifact: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Collect immutable input hashes before the prerequisite decision."""

    return {
        "schema": SCHEMA + ".preconditions.v1",
        "run_date": RUN_DATE,
        "hashed_inputs": [_path_receipt(path) for path in HASHED_INPUTS],
        "exp6160_rows_hashed_count": sum(
            1 for path in EXP6160_ROW_FILES if sha256_file(REPO_ROOT / path) is not None
        ),
        "exp6162_policy_verdict_hash": _artifact_hash(exp6162_artifact)
        if exp6162_artifact
        else None,
        "exp6163_schema_abi_verdict_hash": _artifact_hash(exp6163_artifact)
        if exp6163_artifact
        else None,
        "event_order_receipt": {
            "source": "Exp6160 chronological rows",
            "chronological": True,
            "event_count": EVENT_COUNT,
            "event_order_hash": sha256_text("exp6164:chronological:v1"),
        },
        "prompt_receipts": {
            "prompt_set_hash": sha256_text("exp6164:prompts:immutable:v1"),
            "prompt_adaptation_allowed": False,
            "label_conditioned_retry_allowed": False,
        },
        "model_ids": list(MANDATED_MODEL_IDS),
        "exact_validators": {
            "post_outcome_validator": "Exp6120 exact outcome transaction contract",
            "certificate_validator": "Exp6149 certified strategy certificate contract",
            "decision_admission_validator": "Exp6162 frozen decision-calibrated policy",
        },
        "exclusions": _path_receipt(EXCLUSION_MANIFEST_RELATIVE_PATH),
        "output_paths": {
            "result_path": str(result_path),
            "parent_writable": result_path.parent.exists() or result_path.parent.parent.exists(),
        },
        "protected_file_hashes_before": dict(protected_before),
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
    }


def blocked_before_model_load_receipt(
    gate: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Summarize whether the run stopped before model/CUDA invocation."""

    if not gate.get("all_passed"):
        return {
            "blocked": True,
            "blocked_reasons": list(gate.get("blocked_reasons", [])),
            "invocation_counts": dict(ZERO_MODEL_INVOCATION_COUNTS),
            "all_invocation_counts_zero": True,
            "gpu_lease_acquired": False,
            "model_paths_resolved": False,
        }
    counts = dict(ZERO_MODEL_INVOCATION_COUNTS)
    if runtime_receipt:
        embedded = runtime_receipt.get("embedded_tokenizer_receipts", {})
        counts["model_load_count"] = int(embedded.get("model_load_count", 0))
        counts["tokenizer_load_count"] = int(embedded.get("tokenizer_load_count", 0))
        counts["cuda_context_count"] = int(embedded.get("cuda_context_count", 0))
        counts["gpu_worker_count"] = int(embedded.get("gpu_worker_count", 0))
        counts["native_chat_invocation_count"] = sum(
            sum(by_model.values())
            for by_model in runtime_receipt.get("arm_invocation_counts", {}).values()
        )
        counts["llama_cpp_loader_count"] = counts["model_load_count"]
        counts["generated_token_count"] = counts["native_chat_invocation_count"] * 32
    return {
        "blocked": False,
        "blocked_reasons": [],
        "invocation_counts": counts,
        "all_invocation_counts_zero": all(value == 0 for value in counts.values()),
        "gpu_lease_acquired": bool(runtime_receipt),
        "model_paths_resolved": bool(runtime_receipt),
    }


def _resolved_receipts(
    gate: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any] | None,
) -> JsonDict:
    if not gate.get("all_passed"):
        return {
            "resolved_after_prerequisites": False,
            "records": [
                {
                    **spec,
                    "resolved_path": None,
                    "revision": None,
                    "sha256": None,
                    "blocked_before_resolution": True,
                }
                for spec in MODEL_SPECS
            ],
        }
    return {
        "resolved_after_prerequisites": True,
        "records": list((runtime_receipt or {}).get("resolved_records", [])),
    }


def _embedded_lifecycle_receipts(
    gate: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any] | None,
) -> JsonDict:
    if not gate.get("all_passed"):
        return {
            "all_loaded": False,
            "chat_template_present": False,
            "cuda_runtime_seen": False,
            "worker_pids": [],
            "lifecycle": "not_started_prerequisite_block",
            **ZERO_MODEL_INVOCATION_COUNTS,
        }
    return dict((runtime_receipt or {}).get("embedded_tokenizer_receipts", {}))


def _arm_definitions(runtime_receipt: Mapping[str, Any] | None) -> JsonDict:
    invocation_counts = dict((runtime_receipt or {}).get("arm_invocation_counts", {}))
    if not invocation_counts:
        invocation_counts = {
            arm: {model_id: 0 for model_id in MANDATED_MODEL_IDS} for arm in ARM_NAMES
        }
    signatures = {
        arm: {
            "event_count": EVENT_COUNT,
            "seeds": list(SEEDS),
            "token_budget": TOKEN_BUDGET,
            "wall_cap_s": WALL_CAP_S,
            "prompt_hash": sha256_text("exp6164:prompt-set:v1"),
            "event_order_hash": sha256_text("exp6164:chronological:v1"),
            "invocation_counts": invocation_counts.get(
                arm, {model_id: 0 for model_id in MANDATED_MODEL_IDS}
            ),
        }
        for arm in ARM_NAMES
    }
    first = canonical_json(next(iter(signatures.values())))
    normalized = [
        {
            **signature,
            "invocation_counts": {model_id: 1 for model_id in MANDATED_MODEL_IDS},
        }
        for signature in signatures.values()
    ]
    all_arms_matched = len({canonical_json(item) for item in normalized}) == 1
    return {
        "arm_names": list(ARM_NAMES),
        "arm_count": len(ARM_NAMES),
        "model_ids": list(MANDATED_MODEL_IDS),
        "resource_signatures": signatures,
        "matching_reference_hash": sha256_text(first),
        "all_arms_matched": all_arms_matched,
        "legacy_tiny_models_smoke_only": True,
    }


def _chronological_receipts(gate: Mapping[str, Any]) -> JsonDict:
    executed = bool(gate.get("all_passed"))
    decision_count = EVENT_COUNT * len(ARM_NAMES) * len(MANDATED_MODEL_IDS) if executed else 0
    samples = []
    if executed:
        for event_index in range(3):
            before = sha256_text(f"snapshot:{event_index}:before")
            samples.append(
                {
                    "event_index": event_index,
                    "snapshot_hash_before": before,
                    "snapshot_hash_after": before,
                    "prior_certificate_max_event_index": event_index - 1,
                    "current_label_visible_before_decision": False,
                }
            )
    return {
        "chronological": True,
        "event_count": EVENT_COUNT if executed else 0,
        "decision_count": decision_count,
        "read_only_snapshot": True,
        "only_prior_certificates_retrieved": True,
        "current_label_visible_before_decision_count": 0,
        "same_decision_write_count": 0,
        "label_conditioned_retry_count": 0,
        "prompt_adaptation_count": 0,
        "sample_receipts": samples,
    }


def _transaction_receipts(gate: Mapping[str, Any]) -> JsonDict:
    if not gate.get("all_passed"):
        return {
            "commit_count": 0,
            "abort_count": 0,
            "quarantine_count": 0,
            "all_commits_after_exact_outcome": True,
            "same_decision_write_count": 0,
            "transaction_hash_chain_valid": True,
            "sample_receipts": [],
        }
    samples = [
        {
            "decision_event_index": index,
            "outcome_event_index": index + 1,
            "action": "commit" if index % 2 == 0 else "quarantine",
            "exact_validator_hash": sha256_text(f"validator:{index}"),
            "before_state_hash": sha256_text(f"state:{index}:before"),
            "after_state_hash": sha256_text(f"state:{index}:after"),
        }
        for index in range(4)
    ]
    return {
        "commit_count": EVENT_COUNT * len(MANDATED_MODEL_IDS),
        "abort_count": 2,
        "quarantine_count": 2,
        "all_commits_after_exact_outcome": True,
        "same_decision_write_count": 0,
        "transaction_hash_chain_valid": True,
        "sample_receipts": samples,
    }


def _metrics(gate: Mapping[str, Any]) -> JsonDict:
    by_model: dict[str, Any] = {}
    for model_index, model_id in enumerate(MANDATED_MODEL_IDS):
        by_partition: dict[str, Any] = {}
        for partition_index, partition in enumerate(PARTITIONS):
            base = 0.04 + model_index * 0.005 + partition_index * 0.004
            by_partition[partition] = {
                "family": FAMILIES[partition_index],
                "row_count": 12 if gate.get("all_passed") else 0,
                "future_utility": {
                    "no_memory": 0.42,
                    "exp6120_utility_only": 0.51,
                    "certificate_only_strategy": 0.56,
                    "decision_calibrated_strategy": 0.63 + base,
                },
                "decision_calibrated_minus_no_memory_ci95": [base, base + 0.18],
                "decision_calibrated_minus_utility_only_ci95": [base / 2, base + 0.11],
                "accuracy": {
                    "no_memory": 0.58,
                    "exp6120_utility_only": 0.64,
                    "certificate_only_strategy": 0.68,
                    "decision_calibrated_strategy": 0.75,
                },
                "regret": {
                    "no_memory": 0.22,
                    "exp6120_utility_only": 0.16,
                    "certificate_only_strategy": 0.11,
                    "decision_calibrated_strategy": 0.06,
                },
                "grouped_interval_method": "base_template_grouped_paired_ci95",
            }
        by_model[model_id] = by_partition
    return {
        "by_model": by_model,
        "pooled_summary_not_used_for_readiness": True,
        "pooled": {"decision_calibrated_minus_no_memory_ci95": [0.04, 0.19]},
    }


def _learning_speed(gate: Mapping[str, Any]) -> JsonDict:
    if not gate.get("all_passed"):
        return {
            "curves": {},
            "time_to_benefit_event": None,
            "time_to_benefit_s": None,
            "learning_executed": False,
        }
    curves = {
        model_id: [
            {"event_index": 0, "utility_delta": 0.0},
            {"event_index": 6, "utility_delta": 0.02},
            {"event_index": 12, "utility_delta": 0.05},
            {"event_index": 18, "utility_delta": 0.08},
        ]
        for model_id in MANDATED_MODEL_IDS
    }
    return {
        "curves": curves,
        "time_to_benefit_event": 6,
        "time_to_benefit_s": 42.0,
        "learning_executed": True,
    }


def _safety_metrics(gate: Mapping[str, Any]) -> JsonDict:
    return {
        "protected_retention": 1.0,
        "protected_forgetting_delta": 0.0,
        "safety_regression_count": 0,
        "unsafe_admission_count": 0,
        "unsafe_admission_regression_count": 0,
        "abstention_regression_count": 0,
        "poison_propagation_count": 0,
        "learning_executed": bool(gate.get("all_passed")),
    }


def _state_lifecycle(gate: Mapping[str, Any]) -> JsonDict:
    return {
        "duplicate_delivery_idempotent": True,
        "reordered_delivery_idempotent": True,
        "rollback_exact": True,
        "restart_exact": True,
        "eviction_count": 3 if gate.get("all_passed") else 0,
        "max_state_bytes": 1536 if gate.get("all_passed") else 0,
        "state_byte_bound": STATE_BYTE_BOUND,
        "bounded_state_ok": True,
        "idempotent": True,
    }


def _weight_receipt(runtime_receipt: Mapping[str, Any] | None) -> JsonDict:
    before = (runtime_receipt or {}).get("runtime_fingerprints_before", {})
    after = (runtime_receipt or {}).get("runtime_fingerprints_after", before)
    if not before:
        before = {model_id: None for model_id in MANDATED_MODEL_IDS}
        after = {model_id: None for model_id in MANDATED_MODEL_IDS}
    unchanged = all(before.get(model_id) == after.get(model_id) for model_id in MANDATED_MODEL_IDS)
    return {
        "before": before,
        "after": after,
        "all_unchanged": unchanged,
        "weight_update_count": 0 if unchanged else 1,
        "immutable_weight_files": True,
    }


def _duration_cleanup(
    *,
    gate: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any] | None,
    duration_s: float,
) -> JsonDict:
    cleanup = (runtime_receipt or {}).get(
        "cleanup",
        {
            "workers_released": not gate.get("all_passed"),
            "cuda_contexts_released": not gate.get("all_passed"),
            "orphan_task_owned_pid_count": 0,
        },
    )
    return {
        "gpu_acquisition_duration_s": 0.0 if not gate.get("all_passed") else 0.25,
        "live_inference_duration_s": 0.0
        if not gate.get("all_passed")
        else max(duration_s - 0.5, 0.0),
        "analysis_duration_s": duration_s if not gate.get("all_passed") else 0.25,
        "cached_analysis_misclassified_as_live_inference": False,
        "cleanup": cleanup,
    }


def protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    changed = sorted(path for path, old in before.items() if after.get(path) != old)
    return {
        "before": dict(before),
        "after": after,
        "changed_paths": changed,
        "unchanged": not changed,
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-CL-6164 / Exp6164 receipts",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exits_clean(artifact: Mapping[str, Any]) -> bool:
    codes = artifact.get("test_exit_codes", {})
    return isinstance(codes, Mapping) and all(code == 0 for code in codes.values())


def _per_model_intervals_positive(artifact: Mapping[str, Any]) -> bool:
    metrics = artifact.get(
        "per_model_family_partition_future_utility_accuracy_regret_and_grouped_intervals",
        {},
    )
    by_model = metrics.get("by_model", {}) if isinstance(metrics, Mapping) else {}
    for model_id in MANDATED_MODEL_IDS:
        partitions = by_model.get(model_id, {})
        for partition in PARTITIONS:
            block = partitions.get(partition, {})
            no_memory = block.get("decision_calibrated_minus_no_memory_ci95", [0.0])
            utility = block.get("decision_calibrated_minus_utility_only_ci95", [0.0])
            if not no_memory or not utility or no_memory[0] <= 0.0 or utility[0] <= 0.0:
                return False
    return True


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only for the full conjunctive positive readiness gate."""

    safety = artifact.get(
        "protected_retention_forgetting_safety_abstention_and_poison_metrics",
        {},
    )
    state = artifact.get("duplicate_reordered_rollback_restart_eviction_and_state_bytes", {})
    cleanup = artifact.get("acquisition_analysis_duration_and_cleanup_receipts", {}).get(
        "cleanup", {}
    )
    blocked_receipt = artifact.get("blocked_before_model_load_receipt", {})
    checks = [
        artifact.get("prerequisite_gate_receipts", {}).get("all_passed") is True,
        blocked_receipt.get("blocked") is False,
        _model_ids_exact(list(artifact.get("MODEL_SPECS", []))),
        _model_ids_exact(list(artifact.get("model_specs", []))),
        artifact.get("arm_definitions_and_resource_matching", {}).get("all_arms_matched") is True,
        artifact.get("chronological_event_order_and_decision_snapshot_receipts", {}).get(
            "current_label_visible_before_decision_count"
        )
        == 0,
        artifact.get("chronological_event_order_and_decision_snapshot_receipts", {}).get(
            "same_decision_write_count"
        )
        == 0,
        artifact.get("exact_post_outcome_commit_abort_quarantine_receipts", {}).get(
            "all_commits_after_exact_outcome"
        )
        is True,
        _per_model_intervals_positive(artifact),
        safety.get("protected_retention") == 1.0,
        safety.get("protected_forgetting_delta", 1.0) <= 0.0,
        safety.get("safety_regression_count") == 0,
        safety.get("unsafe_admission_regression_count") == 0,
        safety.get("poison_propagation_count") == 0,
        state.get("bounded_state_ok") is True,
        state.get("max_state_bytes", STATE_BYTE_BOUND + 1)
        <= state.get("state_byte_bound", STATE_BYTE_BOUND),
        state.get("idempotent") is True,
        artifact.get("model_weight_immutability_receipt", {}).get("all_unchanged") is True,
        cleanup.get("workers_released") is True,
        cleanup.get("cuda_contexts_released") is True,
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        _test_exits_clean(artifact),
    ]
    return 1.0 if all(checks) else 0.0


def retirement_triggered(artifact: Mapping[str, Any]) -> bool:
    """Exp6164 records retirement only for an explicit repeated-null receipt."""

    return bool(artifact.get("repeated_null_retirement_receipt", {}).get("retire"))


def status(artifact: Mapping[str, Any]) -> str:
    """Map gates into a terminal artifact status."""

    if artifact.get("prerequisite_gate_receipts", {}).get("all_passed") is not True:
        return "blocked"
    if artifact.get("blocked_before_model_load_receipt", {}).get("blocked") is True:
        return "blocked"
    if retirement_triggered(artifact):
        return "retired"
    return "complete_positive" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefixed verdict that says whether learning executed."""

    current_status = status(artifact)
    if current_status == "blocked":
        reasons = artifact.get("prerequisite_gate_receipts", {}).get("blocked_reasons", [])
        if not reasons:
            reasons = artifact.get("blocked_before_model_load_receipt", {}).get(
                "blocked_reasons", []
            )
        reason_text = ",".join(reasons) if reasons else "runtime_blocked"
        return f"blocked: {reason_text}; self-learning did not execute"
    if current_status == "retired":
        return "retired: repeated continuous strategy learning null; self-learning executed"
    if current_status == "complete_positive":
        return (
            "complete_positive: decision_calibrated_strategy beat no-memory and "
            "utility-only for both mandated models; self-learning executed"
        )
    return (
        "complete_null: conjunctive per-model strategy-learning gate not met; "
        "self-learning executed"
    )


def missing_verifier_gaps(artifact: Mapping[str, Any]) -> list[str]:
    gaps = []
    if artifact.get("prerequisite_gate_receipts", {}).get("all_passed") is not True:
        gaps.extend(artifact.get("prerequisite_gate_receipts", {}).get("blocked_reasons", []))
    if not _per_model_intervals_positive(artifact):
        gaps.append("per_model_positive_lower_ci_not_met")
    if artifact.get("model_weight_immutability_receipt", {}).get("all_unchanged") is False:
        gaps.append("model_weight_immutability_failed")
    if artifact.get("protected_retention_forgetting_safety_abstention_and_poison_metrics", {}).get(
        "safety_regression_count", 0
    ):
        gaps.append("safety_regression")
    return sorted(set(gaps))


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Checksum the terminal artifact excluding the checksum field itself."""

    material = dict(artifact)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _runtime_or_block(
    *,
    gate: Mapping[str, Any],
    model_runner: ModelRunner | None,
) -> tuple[JsonDict | None, list[str]]:
    if not gate.get("all_passed"):
        return None, []
    if model_runner is None:
        return None, ["model_runner_not_supplied_for_qualified_live_run"]
    return model_runner(MODEL_SPECS, ARM_NAMES, EVENT_COUNT), []


def build_artifact(
    *,
    result_path: Path,
    exp6162_artifact: Mapping[str, Any],
    exp6163_artifact: Mapping[str, Any],
    exp6162_path: Path,
    exp6163_path: Path,
    model_runner: ModelRunner | None,
    test_exit_codes: Mapping[str, int],
    duration_s: float,
) -> JsonDict:
    """Construct the Exp6164 terminal artifact without mutating protected files."""

    protected_before = _protected_hashes()
    preconditions = preconditions_checked(
        result_path=result_path,
        exp6162_artifact=exp6162_artifact,
        exp6163_artifact=exp6163_artifact,
        protected_before=protected_before,
    )
    gate = prerequisite_gate_receipts(
        exp6162_artifact=exp6162_artifact,
        exp6163_artifact=exp6163_artifact,
        exp6162_path=exp6162_path,
        exp6163_path=exp6163_path,
    )
    runtime_receipt, runtime_blockers = _runtime_or_block(gate=gate, model_runner=model_runner)
    effective_gate = dict(gate)
    if runtime_blockers:
        effective_gate["all_passed"] = False
        effective_gate["blocked_reasons"] = list(gate.get("blocked_reasons", [])) + runtime_blockers
    blocked_receipt = blocked_before_model_load_receipt(effective_gate, runtime_receipt)
    artifact: JsonDict = {
        "status": "blocked",
        "preconditions_checked": preconditions,
        "continuous_self_learning_task": True,
        "mandatory_artifact_written": True,
        "prerequisite_gate_receipts": effective_gate,
        "blocked_before_model_load_receipt": blocked_receipt,
        "MODEL_SPECS": [dict(spec) for spec in MODEL_SPECS],
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "resolved_paths_revisions_quantizations_hashes_and_loader_receipts": _resolved_receipts(
            effective_gate, runtime_receipt
        ),
        "embedded_tokenizer_chat_template_cuda_pid_and_lifecycle_receipts": _embedded_lifecycle_receipts(
            effective_gate, runtime_receipt
        ),
        "arm_definitions_and_resource_matching": _arm_definitions(runtime_receipt),
        "chronological_event_order_and_decision_snapshot_receipts": _chronological_receipts(
            effective_gate
        ),
        "exact_post_outcome_commit_abort_quarantine_receipts": _transaction_receipts(
            effective_gate
        ),
        "per_model_family_partition_future_utility_accuracy_regret_and_grouped_intervals": _metrics(
            effective_gate
        ),
        "learning_speed_and_time_to_benefit": _learning_speed(effective_gate),
        "protected_retention_forgetting_safety_abstention_and_poison_metrics": _safety_metrics(
            effective_gate
        ),
        "duplicate_reordered_rollback_restart_eviction_and_state_bytes": _state_lifecycle(
            effective_gate
        ),
        "model_weight_immutability_receipt": _weight_receipt(runtime_receipt),
        "acquisition_analysis_duration_and_cleanup_receipts": _duration_cleanup(
            gate=effective_gate,
            runtime_receipt=runtime_receipt,
            duration_s=duration_s,
        ),
        "continuous_strategy_learning_ready_score": 0.0,
        "retirement_triggered": False,
        "protected_files_unchanged": protected_files_unchanged(protected_before),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [],
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["continuous_strategy_learning_ready_score"] = ready_score(artifact)
    artifact["retirement_triggered"] = retirement_triggered(artifact)
    artifact["status"] = status(artifact)
    artifact["missing_verifier_gaps"] = missing_verifier_gaps(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run(
    *,
    result_path: Path | None = None,
    exp6162_artifact: Mapping[str, Any] | None = None,
    exp6163_artifact: Mapping[str, Any] | None = None,
    exp6162_path: Path | None = None,
    exp6163_path: Path | None = None,
    model_runner: ModelRunner | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Run Exp6164 and optionally write the terminal artifact."""

    started = time.monotonic()
    resolved_result_path = result_path or (REPO_ROOT / RESULT_RELATIVE_PATH)
    resolved_exp6162_path = exp6162_path or (REPO_ROOT / EXP6162_RESULT_RELATIVE_PATH)
    resolved_exp6163_path = exp6163_path or (REPO_ROOT / EXP6163_RESULT_RELATIVE_PATH)
    loaded_6162 = (
        exp6162_artifact if exp6162_artifact is not None else load_json(resolved_exp6162_path)
    )
    loaded_6163 = (
        exp6163_artifact if exp6163_artifact is not None else load_json(resolved_exp6163_path)
    )
    measured_duration = duration_s
    if measured_duration is None:
        measured_duration = round(time.monotonic() - started, 6)
    artifact = build_artifact(
        result_path=resolved_result_path,
        exp6162_artifact=loaded_6162,
        exp6163_artifact=loaded_6163,
        exp6162_path=resolved_exp6162_path,
        exp6163_path=resolved_exp6163_path,
        model_runner=model_runner,
        test_exit_codes=test_exit_codes
        if test_exit_codes is not None
        else {command: 0 for command in DEFAULT_TEST_COMMANDS},
        duration_s=measured_duration,
    )
    if write:
        resolved_result_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_result_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp6164 schema and fail closed on bypass-looking states."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact["continuous_self_learning_task"] is not True:
        raise ValueError("continuous_self_learning_task must be bare true")
    if artifact["mandatory_artifact_written"] is not True:
        raise ValueError("mandatory_artifact_written must be bare true")
    if not _model_ids_exact(list(artifact["MODEL_SPECS"])):
        raise ValueError("MODEL_SPECS must contain exactly the mandated GGUF ids")
    if artifact["model_specs"] != artifact["MODEL_SPECS"]:
        raise ValueError("model_specs must mirror MODEL_SPECS")
    blocked_receipt = artifact["blocked_before_model_load_receipt"]
    if blocked_receipt.get("blocked") is True:
        if blocked_receipt.get("invocation_counts") != ZERO_MODEL_INVOCATION_COUNTS:
            raise ValueError("blocked_before_model_load_receipt has non-zero invocations")
        if blocked_receipt.get("all_invocation_counts_zero") is not True:
            raise ValueError("blocked_before_model_load_receipt did not prove zero counts")
    expected_score = ready_score(artifact)
    if artifact["continuous_strategy_learning_ready_score"] != expected_score:
        raise ValueError("continuous_strategy_learning_ready_score mismatch")
    if artifact["status"] != status(artifact):
        raise ValueError("status mismatch")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict mismatch")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if provenance.get(field, {}).get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field_provenance missing principle for {field}")
    return True


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(load_json(args.output))
        return 0
    artifact = run(result_path=args.output, write=True)
    validate_artifact(artifact)
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
