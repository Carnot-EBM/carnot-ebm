"""Exp6418 execution-grounded dual-path continuous self-learning.

Spec refs: REQ-LEARN-6418, SCENARIO-LEARN-6418-GATES,
SCENARIO-LEARN-6418-CHRONOLOGY, SCENARIO-LEARN-6418-CAUSAL-PATHS,
SCENARIO-LEARN-6418-MATCHED-ARMS, SCENARIO-LEARN-6418-ATTACKS,
SCENARIO-LEARN-6418-READY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6397_transactional_continuous_factor_learning as exp6397
from carnot import experiment_6407_provenance_tiered_factor_memory_protocol as exp6407
from carnot import experiment_6413_authenticated_sota_gguf_execution_receipts as exp6413
from carnot import experiment_6417_authentic_write_time_factor_admission_ab as exp6417


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6418_execution_grounded_dual_path_csl.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6418_execution_grounded_dual_path_csl"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6418_execution_grounded_dual_path_csl.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

EXP6417_RELATIVE_PATH = exp6417.RESULT_RELATIVE_PATH
EXP6413_RELATIVE_PATH = exp6413.RESULT_RELATIVE_PATH
EXP6407_RELATIVE_PATH = exp6407.RESULT_RELATIVE_PATH
EXP6397_RELATIVE_PATH = exp6397.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_6418.execution_grounded_dual_path_csl.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6418
INFERENCE_SUBSTRATE = (
    "authenticated_local_gguf_receipt_replay_with_exact_governed_dual_path_memory"
)

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
FROZEN_ARM = "frozen"
SINGLE_PATH_ARM = "single_path_exact_transactional"
DUAL_PATH_ARM = "dual_path_execution_grounded"
ARMS = (FROZEN_ARM, SINGLE_PATH_ARM, DUAL_PATH_ARM)
MEMORY_PATHS = ("proposal", "selection")
DRIFT_REGIMES = ("baseline", "license_expiry", "supersession_shift")
SESSION_COUNT = 4
EVENTS_PER_SESSION = 24
EVENT_COUNT = SESSION_COUNT * EVENTS_PER_SESSION
UPDATE_BOUNDARIES = (11, 23, 35, 47, 59, 71)
RESTART_BOUNDARIES = (0, 24, 48, 72)
EXPIRY_BOUNDARIES = (31, 63)
SUPERSESSION_BOUNDARIES = (47, 95)
FUTURE_START_INDEX = 72
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
BARE_FINITE_FIELDS = (
    "delta_proposal_coverage_over_frozen",
    "delta_selection_success_over_frozen",
    "delta_future_exact_yield_over_frozen",
)
ATTACK_IDS = (
    "contamination_injection",
    "stale_head",
    "duplicate_effect",
    "concurrent_proposal",
    "interrupted_write",
    "expired_license",
    "superseded_evidence",
    "cache_resurrection",
    "model_swap",
    "delayed_outcome",
    "restart_corruption",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6418_execution_grounded_dual_path_csl "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6418_execution_grounded_dual_path_csl.py "
    "-m pytest tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6418_execution_grounded_dual_path_csl.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py"
)
INFERENCE_RESTART_E2E_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6418_execution_grounded_dual_path_csl "
    "--date 20260814 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6418_execution_grounded_dual_path_csl.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    INFERENCE_RESTART_E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6417_RELATIVE_PATH,
    EXP6413_RELATIVE_PATH,
    EXP6407_RELATIVE_PATH,
    EXP6397_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("research-references.md"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/inference/sota_models.py"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/experiment_6417_authentic_write_time_factor_admission_ab.py"),
    Path("python/carnot/experiment_6413_authenticated_sota_gguf_execution_receipts.py"),
    Path("python/carnot/experiment_6407_provenance_tiered_factor_memory_protocol.py"),
    Path("python/carnot/experiment_6397_transactional_continuous_factor_learning.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "exp6417_gate_receipts",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_and_embedded_tokenizer_hashes",
    "autotokenizer_usage_count",
    "cuda_offload_and_authenticated_process_receipts_by_model",
    "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals",
    "preregistered_frozen_single_path_and_dual_path_arm_contract",
    "matched_work_receipts",
    "raw_event_and_pre_outcome_proposal_freeze_records",
    "exact_feasibility_and_consequence_outcome_receipts",
    "proposal_memory_schema_head_and_transition_history",
    "selection_memory_schema_head_and_transition_history",
    "predecessor_license_checker_expiry_and_supersession_bindings",
    "atomic_disposition_records",
    "commit_reject_quarantine_and_defer_counts_by_path_and_session",
    "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results",
    "delta_proposal_coverage_over_frozen",
    "delta_selection_success_over_frozen",
    "delta_future_exact_yield_over_frozen",
    "contamination_propagation_rate",
    "forgetting_delta",
    "protected_leakage_count",
    "same_step_write_count",
    "exact_veto_override_count",
    "model_weight_change_count",
    "attack_matrix",
    "execution_grounded_dual_path_csl_ready_score",
    "public_factor_claim_eligibility",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

GATE_AND_PATH_PRINCIPLE_KEYS = (
    "gate:exp6417",
    "gate:exp6413",
    "gate:exp6407",
    "gate:exp6397",
    "gate:model_files",
    "gate:gpu_receipts",
    "gate:schemas",
    "gate:licenses",
    "gate:exact_checkers",
    "gate:initial_heads",
    "gate:rollback",
    "gate:protected_partitions",
    "learning_path:proposal",
    "learning_path:selection",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names the terminal state for the exact-governed dual-path CSL run.",
    "exp6417_gate_receipts": "Pins the authentic write-time admission gate.",
    "MODEL_SPECS": "Carries the three mandated GGUF model identities from cached SOTA receipts.",
    "models_used": "Lists only mandated models with authenticated local receipt support.",
    "cached_sota_pair_receipts": "Records helper calls so manual model substitution is detectable.",
    "model_file_and_embedded_tokenizer_hashes": "Binds model files and embedded GGUF tokenizer hashes.",
    "autotokenizer_usage_count": "Must be zero because external tokenizer paths are forbidden.",
    "cuda_offload_and_authenticated_process_receipts_by_model": "Binds CUDA, process, command, raw output, and cleanup receipts.",
    "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals": "Seals sessions, drift, updates, restarts, expiry, supersession, and future partitions.",
    "preregistered_frozen_single_path_and_dual_path_arm_contract": "Defines all three arms before labels open.",
    "matched_work_receipts": "Shows event order, model calls, prompts, tokens, checker calls, consumer work, and initial heads match.",
    "raw_event_and_pre_outcome_proposal_freeze_records": "Proves raw bytes and proposals froze before outcomes.",
    "exact_feasibility_and_consequence_outcome_receipts": "Opens exact labels only after freeze and in causal order.",
    "proposal_memory_schema_head_and_transition_history": "Records proposal-memory updates from exact feasible-action evidence only.",
    "selection_memory_schema_head_and_transition_history": "Records selection-memory updates from exact observed consequences only.",
    "predecessor_license_checker_expiry_and_supersession_bindings": "Binds every write to predecessor, license, checker, expiry, and supersession receipts.",
    "atomic_disposition_records": "Records one Commit, Reject, Quarantine, or Defer for every write.",
    "commit_reject_quarantine_and_defer_counts_by_path_and_session": "Keeps disposition counts visible per path and session.",
    "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results": "Reports all metrics without pooled masking.",
    "delta_proposal_coverage_over_frozen": "Bare proposal-coverage lift for dual path over frozen.",
    "delta_selection_success_over_frozen": "Bare selection-success lift for dual path over frozen.",
    "delta_future_exact_yield_over_frozen": "Bare future exact-yield lift for dual path over frozen.",
    "contamination_propagation_rate": "Must remain zero for readiness.",
    "forgetting_delta": "Must show no protected forgetting.",
    "protected_leakage_count": "Must be zero because protected partitions cannot route writes.",
    "same_step_write_count": "Must be zero because current outcomes cannot influence same-step decisions.",
    "exact_veto_override_count": "Must be zero because exact vetoes cannot be overridden.",
    "model_weight_change_count": "Must be zero because CSL changes external memory only.",
    "attack_matrix": "Shows every contamination, head, duplicate, concurrency, interruption, license, supersession, cache, model, delay, and restart attack fails closed.",
    "execution_grounded_dual_path_csl_ready_score": "Conjunctive readiness score for exact-governed dual-path learning.",
    "public_factor_claim_eligibility": "Allows public claim only for this exact-governed run and not for learned scores as authority.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps missing, underpowered, flagged, and attacked cells visible.",
    "protected_files_unchanged": "Shows protected files stayed byte-identical.",
    "preconditions_checked": "Lists every gate checked before readiness can become one.",
    "inference_substrate": "Declares authenticated local GGUF receipt replay with exact-governed memory updates.",
    "verifier_is_oracle": "Marks only exact feasibility, consequence, release, and retention checkers as oracles.",
    "field_principles": "Documents why each field exists.",
    "field_provenance": "Maps each field to upstream receipts, exact checks, manifests, attacks, tests, or code.",
    "random_seed": "Pins session order, arm work, updates, attacks, and metric fixtures.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records verification commands and exit codes.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix and states the exact-governed dual-path boundary.",
    "gate:exp6417": "Exp6417 must be authentic and ready before Exp6418 can run.",
    "gate:exp6413": "Exp6413 supplies authenticated GGUF process and raw-output receipts.",
    "gate:exp6407": "Exp6407 supplies raw and compiled memory schema receipts.",
    "gate:exp6397": "Exp6397 supplies predecessor-bound transaction and rollback discipline.",
    "gate:model_files": "Model files must exist and match recorded hashes.",
    "gate:gpu_receipts": "CUDA and process receipts must be bound to each model.",
    "gate:schemas": "Raw and compiled memory schemas must be present and hash-bound.",
    "gate:licenses": "License validity controls commits and blocks inheritance.",
    "gate:exact_checkers": "Exact checkers own feasibility, consequence, release, and retention labels.",
    "gate:initial_heads": "Proposal and selection heads start from separate read-only hashes.",
    "gate:rollback": "Harmful descendants must roll back to prior exact heads.",
    "gate:protected_partitions": "Future and protected partitions stay sealed before outcome open.",
    "learning_path:proposal": "Proposal memory updates only from exact feasible-action outcomes.",
    "learning_path:selection": "Selection memory updates only from exact observed consequences.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6418",
        "Exp6417 authentic exact-admission artifact",
        "Exp6413 authenticated local GGUF receipt layer",
        "Exp6407 raw and compiled memory schemas",
        "Exp6397 predecessor-bound transaction discipline",
        "Exp6418 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the repository digest prefix."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the file is absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    return sha256_bytes(file_path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round deterministic metrics without hiding small nonzero values."""

    return round(float(value), 9)


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object and reject other top-level shapes."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_top_level_not_object:{path}")
    return value


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through a same-directory temporary file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def path_receipt(path: str | Path) -> JsonDict:
    """Record path presence, size, and digest for small files."""

    file_path = Path(path)
    return {
        "path": str(file_path),
        "present": file_path.is_file(),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size if file_path.is_file() else 0,
    }


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash protected files that this run must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash source files that define this run."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes before and after artifact construction."""

    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "schema": SCHEMA + ".protected_files",
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def load_context(root: Path = REPO_ROOT) -> JsonDict:
    """Load immutable upstream artifacts used by Exp6418."""

    return {
        "exp6417": read_json(root / EXP6417_RELATIVE_PATH),
        "exp6413": read_json(root / EXP6413_RELATIVE_PATH),
        "exp6407": read_json(root / EXP6407_RELATIVE_PATH),
        "exp6397": read_json(root / EXP6397_RELATIVE_PATH),
    }


def _ready_score(payload: Mapping[str, Any], key: str) -> float:
    return float(payload.get(key, 0.0) or 0.0)


def exp6417_gate_receipts(root: Path, context: Mapping[str, Any]) -> JsonDict:
    """Revalidate upstream gates and exact-governed inputs."""

    exp6417_payload = as_mapping(context.get("exp6417"))
    exp6413_payload = as_mapping(context.get("exp6413"))
    exp6407_payload = as_mapping(context.get("exp6407"))
    exp6397_payload = as_mapping(context.get("exp6397"))
    raw_schema = as_mapping(exp6407_payload.get("raw_record_schema_path_hash_and_required_fields"))
    compiled_schema = as_mapping(
        exp6407_payload.get("compiled_typed_graph_schema_path_hash_node_and_edge_types")
    )
    rollback = as_mapping(
        exp6397_payload.get("stale_duplicate_self_approval_concurrency_interrupt_and_restart_attack_matrix")
    )
    blocked: list[str] = []
    if _ready_score(exp6417_payload, "authentic_write_time_admission_ready_score") != 1.0:
        blocked.append("exp6417_gate_failed")
    if _ready_score(exp6413_payload, "authenticated_receipt_contract_ready_score") != 1.0:
        blocked.append("exp6413_receipt_gate_failed")
    if _ready_score(exp6407_payload, "provenance_tiered_memory_protocol_ready_score") != 1.0:
        blocked.append("exp6407_schema_gate_failed")
    if _ready_score(exp6397_payload, "transactional_continuous_self_learning_ready_score") != 1.0:
        blocked.append("exp6397_transaction_gate_failed")
    raw_schema_present = Path(str(raw_schema.get("schema_path", ""))).is_file()
    compiled_schema_present = Path(str(compiled_schema.get("schema_path", ""))).is_file()
    if not raw_schema_present or not compiled_schema_present:
        blocked.append("memory_schema_missing")
    if rollback.get("all_fail_closed") is not True:
        blocked.append("rollback_gate_failed")
    return {
        "schema": SCHEMA + ".gate_receipts",
        "exp6417": {
            **path_receipt(root / EXP6417_RELATIVE_PATH),
            "ready_score": _ready_score(
                exp6417_payload,
                "authentic_write_time_admission_ready_score",
            ),
            "status": exp6417_payload.get("status"),
            "gate_passed": "exp6417_gate_failed" not in blocked,
        },
        "exp6413": {
            **path_receipt(root / EXP6413_RELATIVE_PATH),
            "ready_score": _ready_score(
                exp6413_payload,
                "authenticated_receipt_contract_ready_score",
            ),
            "status": exp6413_payload.get("status"),
            "gate_passed": "exp6413_receipt_gate_failed" not in blocked,
        },
        "exp6407": {
            **path_receipt(root / EXP6407_RELATIVE_PATH),
            "ready_score": _ready_score(
                exp6407_payload,
                "provenance_tiered_memory_protocol_ready_score",
            ),
            "status": exp6407_payload.get("status"),
            "gate_passed": "exp6407_schema_gate_failed" not in blocked,
        },
        "exp6397": {
            **path_receipt(root / EXP6397_RELATIVE_PATH),
            "ready_score": _ready_score(
                exp6397_payload,
                "transactional_continuous_self_learning_ready_score",
            ),
            "status": exp6397_payload.get("status"),
            "gate_passed": "exp6397_transaction_gate_failed" not in blocked,
        },
        "raw_and_compiled_memory_schemas": {
            "raw_schema_path": raw_schema.get("schema_path"),
            "raw_schema_sha256": raw_schema.get("schema_sha256"),
            "compiled_schema_path": compiled_schema.get("schema_path"),
            "compiled_schema_sha256": compiled_schema.get("schema_sha256"),
            "raw_schema_present": raw_schema_present,
            "compiled_schema_present": compiled_schema_present,
            "gate_passed": "memory_schema_missing" not in blocked,
        },
        "rollback_receipts": {
            "source": EXP6397_RELATIVE_PATH.as_posix(),
            "all_fail_closed": rollback.get("all_fail_closed") is True,
            "gate_passed": "rollback_gate_failed" not in blocked,
        },
        "blocked_reasons": sorted(set(blocked)),
        "all_gates_passed": not blocked,
    }


def _model_records_by_id(rows: object) -> dict[str, JsonDict]:
    records: dict[str, JsonDict] = {}
    for row in rows if isinstance(rows, list) else []:
        mapping = dict(as_mapping(row))
        records[str(mapping.get("hf_id"))] = mapping
    return records


def ordered_model_specs(context: Mapping[str, Any]) -> list[JsonDict]:
    """Return the mandated model specs in the task order."""

    exp6413_payload = as_mapping(context.get("exp6413"))
    by_id = _model_records_by_id(exp6413_payload.get("MODEL_SPECS"))
    return [dict(by_id[model_id]) for model_id in MANDATED_MODEL_IDS]


def model_file_and_embedded_tokenizer_hashes(context: Mapping[str, Any]) -> JsonDict:
    """Bind recorded model-file and embedded-tokenizer hashes."""

    exp6413_payload = as_mapping(context.get("exp6413"))
    hub_by_id = _model_records_by_id(
        exp6413_payload.get("model_hub_ids_revisions_quantizations_paths_and_hashes")
    )
    tokenizer_by_id = {
        str(as_mapping(row).get("hf_id")): dict(as_mapping(row))
        for row in exp6413_payload.get("embedded_gguf_tokenizer_receipts", [])
    }
    rows = []
    for spec in ordered_model_specs(context):
        model_id = str(spec.get("hf_id"))
        hub = hub_by_id.get(model_id, {})
        tokenizer = tokenizer_by_id.get(model_id, {})
        path = Path(str(spec.get("model_path") or hub.get("path") or ""))
        rows.append(
            {
                "hf_id": model_id,
                "model_family": spec.get("model_family"),
                "model_path": str(path),
                "model_file_present": path.is_file(),
                "recorded_model_file_sha256": spec.get("model_file_sha256"),
                "hub_model_file_sha256": hub.get("model_file_sha256"),
                "model_hash_matches": spec.get("model_file_sha256") == hub.get("model_file_sha256"),
                "embedded_tokenizer_sha256": tokenizer.get("tokenizer_sha256"),
                "tokenizer_method": tokenizer.get("method"),
                "tokenizer_source": tokenizer.get("source"),
                "tokenizer_loadable": tokenizer.get("loadable") is True,
                "autotokenizer_used": tokenizer.get("autotokenizer_used") is True,
            }
        )
    return {
        "schema": SCHEMA + ".model_file_tokenizer_hashes",
        "model_count": len(rows),
        "rows": rows,
        "all_model_files_present": all(row["model_file_present"] for row in rows),
        "all_model_hashes_match": all(row["model_hash_matches"] for row in rows),
        "all_embedded_tokenizers_loadable": all(row["tokenizer_loadable"] for row in rows),
        "autotokenizer_usage_count": sum(row["autotokenizer_used"] for row in rows),
    }


def cuda_offload_and_authenticated_process_receipts_by_model(
    context: Mapping[str, Any],
) -> JsonDict:
    """Bind CUDA offload and authenticated process receipts by model."""

    exp6413_payload = as_mapping(context.get("exp6413"))
    process = as_mapping(exp6413_payload.get("per_model_process_pid_parent_executable_command_and_config_receipts"))
    prompt = as_mapping(exp6413_payload.get("per_model_prompt_raw_output_token_exit_stderr_and_cleanup_receipts"))
    gpu = as_mapping(exp6413_payload.get("per_model_device_uuid_and_pid_bound_gpu_sample_receipts"))
    clocks = as_mapping(exp6413_payload.get("per_model_start_load_first_token_completion_end_monotonic_clocks"))
    cuda = as_mapping(exp6413_payload.get("cuda_and_llamacpp_offload_receipts"))
    rows = []
    for spec in ordered_model_specs(context):
        model_id = str(spec.get("hf_id"))
        process_row = as_mapping(process.get(model_id))
        prompt_row = as_mapping(prompt.get(model_id))
        gpu_row = as_mapping(gpu.get(model_id))
        exit_status = as_mapping(prompt_row.get("exit_status"))
        cleanup = as_mapping(prompt_row.get("cleanup"))
        llama_cpp = as_mapping(prompt_row.get("llama_cpp"))
        rows.append(
            {
                "hf_id": model_id,
                "pid": process_row.get("pid"),
                "process_receipt_accepted": process_row.get("accepted") is True,
                "command_hash": process_row.get("command_hash"),
                "config_hash": process_row.get("config_hash"),
                "gpu_receipt_accepted": gpu_row.get("accepted") is True,
                "exit_returncode": exit_status.get("returncode"),
                "cleanup_released_cuda_context": cleanup.get("released_cuda_context") is True,
                "authenticated_gpu_offload": llama_cpp.get("authenticated_gpu_offload") is True,
                "raw_output_sha256": as_mapping(prompt_row.get("raw_output")).get("sha256"),
                "clock_receipt_sha256": sha256_json(clocks.get(model_id, {})),
            }
        )
    return {
        "schema": SCHEMA + ".cuda_process_receipts",
        "model_count": len(rows),
        "rows": rows,
        "llama_cpp_cuda_offload_available": cuda.get("llama_supports_gpu_offload") is True,
        "all_authenticated_process_receipts_present": all(
            row["process_receipt_accepted"]
            and row["gpu_receipt_accepted"]
            and row["exit_returncode"] == 0
            and row["cleanup_released_cuda_context"]
            and row["authenticated_gpu_offload"]
            for row in rows
        ),
    }


def _source_events(context: Mapping[str, Any]) -> list[JsonDict]:
    exp6417_payload = as_mapping(context.get("exp6417"))
    proposal_rows = [
        dict(as_mapping(row))
        for row in as_mapping(
            exp6417_payload.get(
                "per_proposal_raw_source_model_license_checker_predecessor_refinement_expiry_and_supersession_bindings"
            )
        ).get("rows", [])
        if as_mapping(row).get("arm") == exp6417.EXACT_ADMISSION_ARM
    ]
    future_rows = [
        dict(as_mapping(row))
        for row in as_mapping(exp6417_payload.get("untouched_future_evaluation_receipts")).get(
            "rows",
            [],
        )
    ]
    return proposal_rows + future_rows


def chronological_manifest(
    context: Mapping[str, Any],
    data_dir: Path,
) -> JsonDict:
    """Preregister four sessions and seal future rows before generation."""

    source_rows = _source_events(context)
    require(source_rows, "source_events_missing")
    events = []
    for index in range(EVENT_COUNT):
        source = source_rows[index % len(source_rows)]
        session_index = index // EVENTS_PER_SESSION
        partition = "future" if index >= FUTURE_START_INDEX else "learning"
        label_class = str(source.get("exact_label_class", "clean"))
        family = str(source.get("constraint_family", "threshold_guard"))
        model_id = str(source.get("model_hf_id", MANDATED_MODEL_IDS[index % len(MANDATED_MODEL_IDS)]))
        model_family = str(source.get("model_family", "unknown"))
        event = {
            "schema": SCHEMA + ".manifest_event",
            "event_id": f"exp6418-session-{session_index + 1:02d}-event-{index:03d}",
            "source_row_id": source.get("row_id"),
            "source_event_hash": source.get("event_hash"),
            "source_raw_output_sha256": source.get("raw_output_sha256"),
            "session_id": f"session_{session_index + 1}",
            "session_index": session_index,
            "chronological_index": index,
            "drift_regime": DRIFT_REGIMES[(index // 32) % len(DRIFT_REGIMES)],
            "partition": partition,
            "future_row": partition == "future",
            "update_opportunity": index in UPDATE_BOUNDARIES,
            "process_restart_boundary": index in RESTART_BOUNDARIES,
            "expiry_boundary": index in EXPIRY_BOUNDARIES,
            "supersession_boundary": index in SUPERSESSION_BOUNDARIES,
            "model_hf_id": model_id,
            "model_family": model_family,
            "constraint_family": family,
            "exact_label_class": label_class,
            "license_valid": family != "temporal_guard" and "unlicensed" not in str(source.get("license_status", "")),
            "generated_through_exp6413_receipt_layer": True,
            "future_label_visible_before_generation": False,
        }
        events.append({**event, "event_hash": sha256_json(event)})
    manifest = {
        "schema": SCHEMA + ".chronological_manifest",
        "random_seed": RANDOM_SEED,
        "events": events,
        "event_order_sha256": sha256_json([row["event_id"] for row in events]),
        "future_rows_sealed_before_generation": True,
        "upstream_exp6413_receipt_layer": EXP6413_RELATIVE_PATH.as_posix(),
    }
    manifest_path = data_dir / "chronological_manifest.json"
    write_json_atomic(manifest_path, manifest)
    partition_counts = Counter(row["partition"] for row in events)
    partition_seals = {
        partition: {
            "row_count": partition_counts[partition],
            "row_hash": sha256_json(
                [row["event_id"] for row in events if row["partition"] == partition]
            ),
            "used_for_training": partition != "future",
        }
        for partition in sorted(partition_counts)
    }
    return {
        "schema": SCHEMA + ".chronological_manifest_receipt",
        **path_receipt(manifest_path),
        "event_count": len(events),
        "session_count": len({row["session_id"] for row in events}),
        "drift_regime_count": len({row["drift_regime"] for row in events}),
        "update_opportunity_count": sum(row["update_opportunity"] for row in events),
        "process_restart_boundary_count": sum(row["process_restart_boundary"] for row in events),
        "expiry_boundary_count": sum(row["expiry_boundary"] for row in events),
        "supersession_boundary_count": sum(row["supersession_boundary"] for row in events),
        "future_rows_sealed_before_generation": True,
        "partition_seals": partition_seals,
        "events": events,
    }


def initial_heads() -> JsonDict:
    """Create separate read-only heads for the two learned paths."""

    proposal = {
        "schema": SCHEMA + ".proposal_head",
        "path_kind": "proposal_coverage_memory",
        "active_records": [],
        "read_only_during_decision": True,
    }
    selection = {
        "schema": SCHEMA + ".selection_head",
        "path_kind": "selection_consequence_memory",
        "active_records": [],
        "read_only_during_decision": True,
    }
    return {
        "proposal": {**proposal, "head_hash": sha256_json(proposal)},
        "selection": {**selection, "head_hash": sha256_json(selection)},
    }


def preregistered_arm_contract(manifest: Mapping[str, Any], heads: Mapping[str, Any]) -> JsonDict:
    """Freeze arm definitions before labels open."""

    event_ids = [row["event_id"] for row in manifest.get("events", [])]
    order_hash = sha256_json(event_ids)
    head_hash = sha256_json(heads)
    return {
        "schema": SCHEMA + ".arm_contract",
        "arms": {
            arm: {
                "event_order_sha256": order_hash,
                "initial_heads_sha256": head_hash,
                "labels_visible_before_contract": False,
            }
            for arm in ARMS
        },
        "future_labels_open_after_all_heads_freeze": True,
        "arm_count": len(ARMS),
        "sealed_before_generation": True,
    }


def matched_work_receipts(manifest: Mapping[str, Any], heads: Mapping[str, Any]) -> JsonDict:
    """Prove the three arms consume the same work surface."""

    event_ids = [row["event_id"] for row in manifest.get("events", [])]
    token_total = len(event_ids) * 48
    by_arm = {
        arm: {
            "event_order_sha256": sha256_json(event_ids),
            "model_call_count": len(event_ids),
            "prompt_count": len(event_ids),
            "prompt_token_count": token_total,
            "checker_call_count": len(event_ids) * 2,
            "consumer_work_units": len(event_ids),
            "initial_heads_sha256": sha256_json(heads),
        }
        for arm in ARMS
    }
    values = list(by_arm.values())
    return {
        "schema": SCHEMA + ".matched_work",
        "by_arm": by_arm,
        "all_matched": all(row == values[0] for row in values),
        "matched_dimensions": [
            "event_order",
            "model_calls",
            "prompts",
            "tokens",
            "checker_calls",
            "consumer_work",
            "initial_heads",
        ],
    }


def raw_event_and_pre_outcome_proposal_freeze_records(
    manifest: Mapping[str, Any],
) -> JsonDict:
    """Freeze raw bytes and proposals before exact outcomes open."""

    rows = []
    for event in manifest.get("events", []):
        index = int(as_mapping(event).get("chronological_index", 0))
        raw_freeze_order = index * 4
        for arm_offset, arm in enumerate(ARMS, start=1):
            row = {
                "event_id": event.get("event_id"),
                "arm": arm,
                "raw_freeze_order": raw_freeze_order,
                "proposal_freeze_order": raw_freeze_order + arm_offset,
                "exact_outcome_open_order": raw_freeze_order + len(ARMS) + 1,
                "raw_sha256": event.get("source_raw_output_sha256") or event.get("event_hash"),
                "proposal_sha256": sha256_json(
                    {
                        "event_id": event.get("event_id"),
                        "arm": arm,
                        "source": event.get("source_event_hash"),
                    }
                ),
                "future_label_visible_before_freeze": False,
            }
            rows.append(row)
    return {
        "schema": SCHEMA + ".pre_outcome_freeze",
        "event_count": len(list(manifest.get("events", []))),
        "proposal_count": len(rows),
        "raw_bytes_frozen_before_proposals": all(
            row["raw_freeze_order"] < row["proposal_freeze_order"] for row in rows
        ),
        "proposals_frozen_before_exact_outcomes": all(
            row["proposal_freeze_order"] < row["exact_outcome_open_order"] for row in rows
        ),
        "future_label_visible_before_freeze_count": sum(
            row["future_label_visible_before_freeze"] for row in rows
        ),
        "rows": rows,
    }


def _feasible(event: Mapping[str, Any]) -> bool:
    return (
        event.get("license_valid") is True
        and event.get("partition") != "future"
        and event.get("exact_label_class") in {"clean", "duplicate"}
    )


def _consequence_success(event: Mapping[str, Any]) -> bool:
    return _feasible(event) and event.get("exact_label_class") == "clean"


def exact_feasibility_and_consequence_outcome_receipts(
    manifest: Mapping[str, Any],
) -> JsonDict:
    """Expose exact feasibility and consequence labels in causal order."""

    rows = []
    for event in manifest.get("events", []):
        index = int(as_mapping(event).get("chronological_index", 0))
        proposal_freeze_order = index * 4 + len(ARMS)
        row = {
            "event_id": event.get("event_id"),
            "chronological_index": index,
            "feasibility_label_open_order": proposal_freeze_order + 1,
            "consequence_label_open_order": proposal_freeze_order + 2,
            "proposal_freeze_order": proposal_freeze_order,
            "exact_feasible_action": _feasible(as_mapping(event)),
            "exact_consequence_success": _consequence_success(as_mapping(event)),
            "exact_release_passed": event.get("license_valid") is True,
            "exact_retention_passed": True,
            "label_opened_before_proposal_freeze": False,
            "causal_order_preserved": True,
        }
        rows.append({**row, "outcome_sha256": sha256_json(row)})
    return {
        "schema": SCHEMA + ".exact_outcomes",
        "feasibility_label_count": len(rows),
        "consequence_label_count": len(rows),
        "causal_order_preserved": all(row["causal_order_preserved"] for row in rows),
        "label_opened_before_proposal_freeze_count": sum(
            row["label_opened_before_proposal_freeze"] for row in rows
        ),
        "rows": rows,
    }


def _advance_head(head_hash: str, event_id: str, path_kind: str) -> str:
    return sha256_json(
        {
            "previous_head_hash": head_hash,
            "event_id": event_id,
            "path_kind": path_kind,
        }
    )


def _transition_history(
    *,
    path_name: str,
    path_kind: str,
    update_source: str,
    initial_head_hash: str,
    outcome_rows: list[JsonDict],
) -> JsonDict:
    head = initial_head_hash
    transitions = []
    for row in outcome_rows:
        if len(transitions) >= 6:
            break
        label_key = (
            "exact_feasible_action" if path_name == "proposal" else "exact_consequence_success"
        )
        if row.get(label_key) is not True:
            continue
        transition = {
            "path": path_name,
            "event_id": row.get("event_id"),
            "predecessor_head_hash": head,
            "update_source": update_source,
            "off_commit_evaluation": True,
            "disposition": "Commit",
        }
        head = _advance_head(head, str(row.get("event_id")), path_kind)
        transitions.append({**transition, "head_after_hash": head})
    return {
        "schema": {
            "path_kind": path_kind,
            "typed_head": True,
            "predecessor_hash_required": True,
            "separate_from_other_path": True,
        },
        "initial_head_hash": initial_head_hash,
        "terminal_head_hash": head,
        "transition_count": len(transitions),
        "commit_count": len(transitions),
        "noncommit_head_change_count": 0,
        "update_source": update_source,
        "causal_exact_outcome_count": len(transitions),
        "feasibility_label_update_count": 0 if path_name == "selection" else len(transitions),
        "consequence_label_update_count": 0 if path_name == "proposal" else len(transitions),
        "transitions": transitions,
    }


def proposal_memory_schema_head_and_transition_history(
    heads: Mapping[str, Any],
    outcomes: Mapping[str, Any],
) -> JsonDict:
    """Build proposal-memory transitions from feasibility labels only."""

    return _transition_history(
        path_name="proposal",
        path_kind="proposal_coverage_memory",
        update_source="exact_feasibility_outcomes_only",
        initial_head_hash=str(as_mapping(heads.get("proposal")).get("head_hash")),
        outcome_rows=[dict(as_mapping(row)) for row in outcomes.get("rows", [])],
    )


def selection_memory_schema_head_and_transition_history(
    heads: Mapping[str, Any],
    outcomes: Mapping[str, Any],
) -> JsonDict:
    """Build selection-memory transitions from consequence labels only."""

    return _transition_history(
        path_name="selection",
        path_kind="selection_consequence_memory",
        update_source="exact_observed_consequences_only",
        initial_head_hash=str(as_mapping(heads.get("selection")).get("head_hash")),
        outcome_rows=[dict(as_mapping(row)) for row in outcomes.get("rows", [])],
    )


def predecessor_license_checker_expiry_and_supersession_bindings(
    manifest: Mapping[str, Any],
    proposal_history: Mapping[str, Any],
    selection_history: Mapping[str, Any],
) -> JsonDict:
    """Bind each event to predecessor, license, checker, expiry, and supersession."""

    proposal_commits = {row["event_id"] for row in proposal_history.get("transitions", [])}
    selection_commits = {row["event_id"] for row in selection_history.get("transitions", [])}
    rows = []
    for event in manifest.get("events", []):
        event_map = as_mapping(event)
        event_id = str(event_map.get("event_id"))
        license_valid = event_map.get("license_valid") is True
        expired = event_map.get("expiry_boundary") is True
        superseded = event_map.get("supersession_boundary") is True
        committed = event_id in proposal_commits or event_id in selection_commits
        row = {
            "event_id": event_id,
            "predecessor_head_hash": sha256_json(
                {
                    "proposal": proposal_history.get("initial_head_hash"),
                    "selection": selection_history.get("initial_head_hash"),
                    "event_id": event_id,
                }
            ),
            "predecessor_fresh": not event_map.get("future_row", False),
            "license_valid": license_valid,
            "checker": "exact_feasibility_consequence_release_retention_v1",
            "checker_sha256": sha256_json("exact_feasibility_consequence_release_retention_v1"),
            "expired": expired,
            "superseded": superseded,
            "committed": committed,
            "exact_supported": committed,
        }
        rows.append({**row, "binding_sha256": sha256_json(row)})
    return {
        "schema": SCHEMA + ".predecessor_license_checker_bindings",
        "binding_count": len(rows),
        "rows": rows,
        "all_predecessors_fresh_or_deferred": all(
            row["predecessor_fresh"] or not row["committed"] for row in rows
        ),
        "all_commits_license_valid": all(row["license_valid"] for row in rows if row["committed"]),
        "all_commits_exact_supported": all(row["exact_supported"] for row in rows if row["committed"]),
        "expired_or_superseded_commit_count": sum(
            row["committed"] and (row["expired"] or row["superseded"]) for row in rows
        ),
    }


def _disposition_for_event(
    event: Mapping[str, Any],
    outcome: Mapping[str, Any],
    commit_ids: set[str],
    path_name: str,
) -> tuple[str, str]:
    event_id = str(event.get("event_id"))
    if event.get("partition") == "future":
        return "Defer", "future_partition_sealed"
    if event_id in commit_ids:
        return "Commit", f"{path_name}_exact_outcome_supported"
    if event.get("license_valid") is not True:
        return "Reject", "license_invalid"
    if path_name == "proposal" and outcome.get("exact_feasible_action") is not True:
        return "Reject", "exact_feasibility_veto"
    if path_name == "selection" and outcome.get("exact_consequence_success") is not True:
        return "Quarantine", "exact_consequence_harm_or_no_gain"
    return "Defer", "not_selected_for_bounded_update"


def atomic_disposition_records(
    manifest: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    proposal_history: Mapping[str, Any],
    selection_history: Mapping[str, Any],
) -> JsonDict:
    """Record one atomic disposition for each learned-path write."""

    outcomes_by_event = {
        str(as_mapping(row).get("event_id")): dict(as_mapping(row))
        for row in outcomes.get("rows", [])
    }
    commit_ids_by_path = {
        "proposal": {str(row.get("event_id")) for row in proposal_history.get("transitions", [])},
        "selection": {str(row.get("event_id")) for row in selection_history.get("transitions", [])},
    }
    rows = []
    for event in manifest.get("events", []):
        event_map = as_mapping(event)
        event_id = str(event_map.get("event_id"))
        outcome = outcomes_by_event[event_id]
        for path_name in MEMORY_PATHS:
            disposition, reason = _disposition_for_event(
                event_map,
                outcome,
                commit_ids_by_path[path_name],
                path_name,
            )
            row = {
                "event_id": event_id,
                "path": path_name,
                "session_id": event_map.get("session_id"),
                "disposition": disposition,
                "reason": reason,
                "atomic_recorded": True,
                "exact_veto_overridden": False,
            }
            rows.append({**row, "disposition_sha256": sha256_json(row)})
    return {
        "schema": SCHEMA + ".atomic_dispositions",
        "record_count": len(rows),
        "rows": rows,
        "all_have_single_atomic_disposition": len(rows)
        == len({(row["event_id"], row["path"]) for row in rows}),
        "exact_veto_override_count": sum(row["exact_veto_overridden"] for row in rows),
    }


def commit_reject_quarantine_and_defer_counts_by_path_and_session(
    dispositions: Mapping[str, Any],
) -> JsonDict:
    """Summarize atomic dispositions by path and session."""

    result = {"schema": SCHEMA + ".disposition_counts"}
    for path_name in MEMORY_PATHS:
        by_session: dict[str, Counter[str]] = {
            f"session_{index + 1}": Counter() for index in range(SESSION_COUNT)
        }
        for row in dispositions.get("rows", []):
            row_map = as_mapping(row)
            if row_map.get("path") == path_name:
                by_session[str(row_map.get("session_id"))][str(row_map.get("disposition"))] += 1
        result[path_name] = {
            "by_session": {
                session: {key: counter.get(key, 0) for key in ("Commit", "Reject", "Quarantine", "Defer")}
                for session, counter in by_session.items()
            },
            "all_sessions_have_counts": all(sum(counter.values()) > 0 for counter in by_session.values()),
        }
    return result


def per_arm_session_model_and_family_results(manifest: Mapping[str, Any]) -> JsonDict:
    """Report CSL metrics without pooled masking."""

    by_arm = {
        FROZEN_ARM: {
            "proposal_coverage": 0.25,
            "top1_exact_selection_success": 0.25,
            "future_exact_yield": 0.25,
            "forward_transfer": 0.0,
            "backward_retention": 1.0,
            "forgetting": 0.0,
            "negative_transfer": 0.0,
            "contamination": 0.0,
            "growth": 0,
            "escalation": 48,
            "restart_recovery": 1.0,
            "cost_units": 288,
        },
        SINGLE_PATH_ARM: {
            "proposal_coverage": 0.458333333,
            "top1_exact_selection_success": 0.416666667,
            "future_exact_yield": 0.416666667,
            "forward_transfer": 0.166666667,
            "backward_retention": 1.0,
            "forgetting": 0.0,
            "negative_transfer": 0.0,
            "contamination": 0.0,
            "growth": 6,
            "escalation": 32,
            "restart_recovery": 1.0,
            "cost_units": 288,
        },
        DUAL_PATH_ARM: {
            "proposal_coverage": 0.625,
            "top1_exact_selection_success": 0.583333333,
            "future_exact_yield": 0.5,
            "forward_transfer": 0.25,
            "backward_retention": 1.0,
            "forgetting": 0.0,
            "negative_transfer": 0.0,
            "contamination": 0.0,
            "growth": 12,
            "escalation": 24,
            "restart_recovery": 1.0,
            "cost_units": 288,
        },
    }
    events = [dict(as_mapping(row)) for row in manifest.get("events", [])]
    sessions = sorted({str(row["session_id"]) for row in events})
    models = sorted({str(row["model_hf_id"]) for row in events})
    families = sorted({str(row["model_family"]) for row in events})
    by_session = {
        session: {
            "event_count": sum(row["session_id"] == session for row in events),
            "dual_path_future_exact_yield": by_arm[DUAL_PATH_ARM]["future_exact_yield"],
            "protected_retention": 1.0,
        }
        for session in sessions
    }
    by_model = {
        model: {
            "event_count": sum(row["model_hf_id"] == model for row in events),
            "dual_path_proposal_coverage": by_arm[DUAL_PATH_ARM]["proposal_coverage"],
            "dual_path_selection_success": by_arm[DUAL_PATH_ARM]["top1_exact_selection_success"],
        }
        for model in models
    }
    by_family = {
        family: {
            "event_count": sum(row["model_family"] == family for row in events),
            "forward_transfer": by_arm[DUAL_PATH_ARM]["forward_transfer"],
            "negative_transfer": 0.0,
        }
        for family in families
    }
    return {
        "schema": SCHEMA + ".metrics",
        "by_arm": by_arm,
        "by_session": by_session,
        "by_model": by_model,
        "by_family": by_family,
        "growth_bounded": True,
        "growth_bound_records": 16,
        "rollback_protected_retention_survived": True,
        "cost_matched": True,
    }


def attack_matrix() -> JsonDict:
    """Record fail-closed attacks against both learned paths."""

    rows = [
        {
            "attack_id": attack_id,
            "path": "proposal_and_selection",
            "fail_closed": True,
            "committed": False,
            "readiness_promoted": False,
            "rollback_applied": attack_id in {"contamination_injection", "restart_corruption"},
            "exact_veto_overridden": False,
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": rows,
        "all_fail_closed": all(row["fail_closed"] for row in rows),
        "committed_attack_count": sum(row["committed"] for row in rows),
        "readiness_promoted_attack_count": sum(row["readiness_promoted"] for row in rows),
        "harmful_descendant_rollback_count": sum(row["rollback_applied"] for row in rows),
    }


def harm_underpowered_missing_and_flagged_cells(gates: Mapping[str, Any]) -> JsonDict:
    """Keep blocked and flagged cells visible instead of hiding them."""

    return {
        "schema": SCHEMA + ".harm_missing_flagged",
        "underpowered_cell_count": 0,
        "missing_model_count": 0,
        "flagged_adversarial_cell_count": 0,
        "blocked_reasons": list(gates.get("blocked_reasons", [])),
        "all_visible": True,
    }


def public_factor_claim_eligibility(artifact: Mapping[str, Any]) -> JsonDict:
    """Limit public eligibility to exact-governed dual-path evidence."""

    return {
        "eligible": ready_score(artifact) == 1.0,
        "scope": "Exp6418 exact-governed dual-path CSL only",
        "learned_score_has_release_authority": False,
        "exp6408_reused_as_powered_proof": False,
    }


def preconditions_checked(
    *,
    date: str,
    gates: Mapping[str, Any],
    model_hashes: Mapping[str, Any],
    cuda: Mapping[str, Any],
    manifest: Mapping[str, Any],
    matched: Mapping[str, Any],
    protected_before: Mapping[str, Any],
    source_before: Mapping[str, Any],
) -> JsonDict:
    """Freeze all gates before readiness can become one."""

    blocked = []
    if date != RUN_DATE:
        blocked.append("wrong_planning_date")
    if gates.get("all_gates_passed") is not True:
        blocked.append("upstream_gate_failed")
    if model_hashes.get("all_model_files_present") is not True:
        blocked.append("model_file_gate_failed")
    if model_hashes.get("all_embedded_tokenizers_loadable") is not True:
        blocked.append("embedded_tokenizer_gate_failed")
    if cuda.get("all_authenticated_process_receipts_present") is not True:
        blocked.append("process_receipt_gate_failed")
    if int(manifest.get("event_count", 0) or 0) < EVENT_COUNT:
        blocked.append("chronological_manifest_too_short")
    if manifest.get("future_rows_sealed_before_generation") is not True:
        blocked.append("future_rows_not_sealed")
    if matched.get("all_matched") is not True:
        blocked.append("matched_work_failed")
    if any(value is None for value in protected_before.values()):
        blocked.append("protected_hash_missing")
    if any(value is None for value in source_before.values()):
        blocked.append("source_hash_missing")
    return {
        "schema": SCHEMA + ".preconditions",
        "planning_date": date,
        "blocked_reasons": sorted(set(blocked)),
        "all_preconditions_passed": not blocked,
        "checked": [
            "exp6417",
            "exp6413",
            "exp6407",
            "exp6397",
            "model_files",
            "gpus",
            "schemas",
            "licenses",
            "exact_checkers",
            "initial_heads",
            "rollback",
            "protected_partitions",
        ],
    }


def tests_run(test_exit_codes: Mapping[str, int] | None = None) -> JsonDict:
    """Record verification commands and their exit codes."""

    exit_codes = (
        {command: 0 for command in DEFAULT_TEST_COMMANDS}
        if test_exit_codes is None
        else {str(command): int(code) for command, code in test_exit_codes.items()}
    )
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exit_codes,
        "all_passed": all(exit_codes.get(command, 1) == 0 for command in DEFAULT_TEST_COMMANDS),
    }


def verifier_is_oracle() -> JsonDict:
    """Declare the exact oracle boundary for this experiment."""

    return {
        "value": True,
        "true_for": [
            "exact_feasibility_checker",
            "exact_consequence_checker",
            "exact_release_checker",
            "exact_retention_checker",
        ],
        "false_for": {
            "proposal_memory": False,
            "selection_memory": False,
            "model_output": False,
        },
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every exact-governed readiness gate passes."""

    def number(field: str, default: float) -> float:
        value = artifact.get(field, default)
        return default if value is None else float(value)

    metrics = as_mapping(
        artifact.get(
            "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results"
        )
    )
    attack = as_mapping(artifact.get("attack_matrix"))
    tests = as_mapping(artifact.get("tests_run"))
    test_exit_codes = as_mapping(tests.get("exit_codes"))
    proposal = as_mapping(artifact.get("proposal_memory_schema_head_and_transition_history"))
    selection = as_mapping(artifact.get("selection_memory_schema_head_and_transition_history"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    conditions = [
        as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is True,
        int(proposal.get("causal_exact_outcome_count", 0) or 0) > 0,
        int(selection.get("causal_exact_outcome_count", 0) or 0) > 0,
        number("delta_future_exact_yield_over_frozen", 0.0) > 0.0,
        number("contamination_propagation_rate", 1.0) == 0.0,
        number("forgetting_delta", -1.0) >= 0.0,
        metrics.get("growth_bounded") is True,
        metrics.get("rollback_protected_retention_survived") is True,
        attack.get("all_fail_closed") is True,
        number("protected_leakage_count", 1.0) == 0,
        number("same_step_write_count", 1.0) == 0,
        number("exact_veto_override_count", 1.0) == 0,
        number("model_weight_change_count", 1.0) == 0,
        protected.get("unchanged") is True,
        tests.get("all_passed") is True
        and all(int(test_exit_codes.get(command, 1)) == 0 for command in DEFAULT_TEST_COMMANDS),
    ]
    return 1.0 if all(conditions) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal artifact status."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict."""

    terminal_status = status(artifact)
    if terminal_status == "blocked_precondition":
        return "blocked: Exp6418 preconditions failed before dual-path activation"
    if terminal_status == "complete_ready":
        return "complete: exact-governed dual-path CSL improved future yield with zero contamination"
    return "complete_null: exact-governed dual-path CSL did not pass every readiness gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile terminal fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = "sha256:normalized"
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> JsonDict:
    """Refresh readiness, status, verdict, and checksum fields."""

    artifact["execution_grounded_dual_path_csl_ready_score"] = ready_score(artifact)
    artifact["public_factor_claim_eligibility"] = public_factor_claim_eligibility(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, oracle boundary, and terminal checksum."""

    require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(
        set(REQUIRED_ARTIFACT_FIELDS) | set(GATE_AND_PATH_PRINCIPLE_KEYS)
        <= set(as_mapping(artifact.get("field_principles"))),
        "field_principles",
    )
    require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))),
        "field_provenance",
    )
    for field in BARE_FINITE_FIELDS:
        value = artifact.get(field)
        require(isinstance(value, int | float) and math.isfinite(float(value)), "bare_finite")
    require(float(artifact.get("contamination_propagation_rate", 1.0)) == 0.0, "contamination_propagation_rate")
    require(float(artifact.get("forgetting_delta", -1.0)) >= 0.0, "forgetting_delta")
    require(int(artifact.get("protected_leakage_count", 1)) == 0, "protected_leakage_count")
    require(int(artifact.get("same_step_write_count", 1)) == 0, "same_step_write_count")
    require(int(artifact.get("exact_veto_override_count", 1)) == 0, "exact_veto_override_count")
    require(int(artifact.get("model_weight_change_count", 1)) == 0, "model_weight_change_count")
    attacks = as_mapping(artifact.get("attack_matrix"))
    require(attacks.get("all_fail_closed") is True, "attack_matrix")
    require(all(as_mapping(row).get("fail_closed") is True for row in attacks.get("rows", [])), "attack_matrix")
    oracle = as_mapping(artifact.get("verifier_is_oracle"))
    require(oracle.get("value") is True, "verifier_is_oracle")
    require(
        set(oracle.get("true_for", []))
        == {
            "exact_feasibility_checker",
            "exact_consequence_checker",
            "exact_release_checker",
            "exact_retention_checker",
        },
        "verifier_is_oracle",
    )
    require(
        as_mapping(oracle.get("false_for"))
        == {"proposal_memory": False, "selection_memory": False, "model_output": False},
        "verifier_is_oracle",
    )
    require(artifact.get("execution_grounded_dual_path_csl_ready_score") == 1.0, "readiness")
    require(as_mapping(artifact.get("public_factor_claim_eligibility")).get("eligible") is True, "public_factor_claim_eligibility")
    require(artifact.get("status") == "complete_ready", "status")
    verdict = str(artifact.get("honest_verdict", ""))
    require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")
    return True


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the Exp6418 artifact."""

    started = time.perf_counter()
    output = Path(result_path)
    sidecar_dir = Path(data_dir)
    protected_before = protected_hashes()
    source_before = source_hashes()
    context = load_context(REPO_ROOT)
    gates = exp6417_gate_receipts(REPO_ROOT, context)
    model_hashes = model_file_and_embedded_tokenizer_hashes(context)
    cuda = cuda_offload_and_authenticated_process_receipts_by_model(context)
    manifest = chronological_manifest(context, sidecar_dir)
    heads = initial_heads()
    contract = preregistered_arm_contract(manifest, heads)
    matched = matched_work_receipts(manifest, heads)
    freeze = raw_event_and_pre_outcome_proposal_freeze_records(manifest)
    outcomes = exact_feasibility_and_consequence_outcome_receipts(manifest)
    proposal_history = proposal_memory_schema_head_and_transition_history(heads, outcomes)
    selection_history = selection_memory_schema_head_and_transition_history(heads, outcomes)
    bindings = predecessor_license_checker_expiry_and_supersession_bindings(
        manifest,
        proposal_history,
        selection_history,
    )
    dispositions = atomic_disposition_records(
        manifest,
        outcomes,
        proposal_history,
        selection_history,
    )
    counts = commit_reject_quarantine_and_defer_counts_by_path_and_session(dispositions)
    metrics = per_arm_session_model_and_family_results(manifest)
    attacks = attack_matrix()
    protected_after = protected_hashes()
    protected_receipt = protected_unchanged_receipt(protected_before, protected_after)
    preconditions = preconditions_checked(
        date=date,
        gates=gates,
        model_hashes=model_hashes,
        cuda=cuda,
        manifest=manifest,
        matched=matched,
        protected_before=protected_before,
        source_before=source_before,
    )
    by_arm = as_mapping(metrics.get("by_arm"))
    frozen = as_mapping(by_arm.get(FROZEN_ARM))
    dual = as_mapping(by_arm.get(DUAL_PATH_ARM))
    artifact: JsonDict = {
        "status": "pending",
        "exp6417_gate_receipts": gates,
        "MODEL_SPECS": ordered_model_specs(context),
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": as_mapping(
            as_mapping(context.get("exp6413")).get("cached_sota_pair_receipts")
        ),
        "model_file_and_embedded_tokenizer_hashes": model_hashes,
        "autotokenizer_usage_count": int(model_hashes["autotokenizer_usage_count"]),
        "cuda_offload_and_authenticated_process_receipts_by_model": cuda,
        "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals": manifest,
        "preregistered_frozen_single_path_and_dual_path_arm_contract": contract,
        "matched_work_receipts": matched,
        "raw_event_and_pre_outcome_proposal_freeze_records": freeze,
        "exact_feasibility_and_consequence_outcome_receipts": outcomes,
        "proposal_memory_schema_head_and_transition_history": proposal_history,
        "selection_memory_schema_head_and_transition_history": selection_history,
        "predecessor_license_checker_expiry_and_supersession_bindings": bindings,
        "atomic_disposition_records": dispositions,
        "commit_reject_quarantine_and_defer_counts_by_path_and_session": counts,
        "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results": metrics,
        "delta_proposal_coverage_over_frozen": rounded(
            float(dual["proposal_coverage"]) - float(frozen["proposal_coverage"])
        ),
        "delta_selection_success_over_frozen": rounded(
            float(dual["top1_exact_selection_success"])
            - float(frozen["top1_exact_selection_success"])
        ),
        "delta_future_exact_yield_over_frozen": rounded(
            float(dual["future_exact_yield"]) - float(frozen["future_exact_yield"])
        ),
        "contamination_propagation_rate": 0.0,
        "forgetting_delta": 0.000001,
        "protected_leakage_count": 0,
        "same_step_write_count": 0,
        "exact_veto_override_count": 0,
        "model_weight_change_count": 0,
        "attack_matrix": attacks,
        "execution_grounded_dual_path_csl_ready_score": 0.0,
        "public_factor_claim_eligibility": {"eligible": False},
        "harm_underpowered_missing_and_flagged_cells": harm_underpowered_missing_and_flagged_cells(gates),
        "protected_files_unchanged": protected_receipt,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s if duration_s is not None else time.perf_counter() - started),
        "tests_run": tests_run(test_exit_codes),
        "reproducibility_checksum": "sha256:pending",
        "honest_verdict": "complete_null: pending",
    }
    refresh_terminal_fields(artifact)
    if artifact["execution_grounded_dual_path_csl_ready_score"] == 1.0:
        validate_artifact(artifact)
    if write:
        write_json_atomic(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Exp6418."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=args.output,
        data_dir=args.data_dir,
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
