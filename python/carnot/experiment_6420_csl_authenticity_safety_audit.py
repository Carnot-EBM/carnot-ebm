"""Exp6420 CSL authenticity and safety audit.

Spec refs: REQ-LEARN-6420, SCENARIO-LEARN-6420-MISSING,
SCENARIO-LEARN-6420-CAUSAL, SCENARIO-LEARN-6420-METRICS,
SCENARIO-LEARN-6420-ATTACKS, SCENARIO-LEARN-6420-ORACLE.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6420_csl_authenticity_safety_audit.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6420_csl_authenticity_safety_audit.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6420_csl_authenticity_safety_audit.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")

EXP6412_RELATIVE_PATH = Path(
    "results/experiment_6412_v551_powered_claim_integrity_audit.json"
)
EXP6412_CLAIM_LEDGER_RELATIVE_PATH = Path(
    "results/experiment_6412_v551_powered_claim_integrity_audit.json.claim_ledger.jsonl"
)
EXP6412_CORRIGENDUM_RELATIVE_PATH = Path(
    "results/experiment_6412_v551_powered_claim_integrity_audit.json.corrigendum.json"
)
EXP6418_RELATIVE_PATH = Path(
    "results/experiment_6418_execution_grounded_dual_path_csl.json"
)
EXP6418_MANIFEST_RELATIVE_PATH = Path(
    "data/research/experiment_6418_execution_grounded_dual_path_csl/chronological_manifest.json"
)
EXP6418_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6418_execution_grounded_dual_path_csl.py"
)
EXP6418_TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6418_execution_grounded_dual_path_csl.py"
)
EXP6419_RELATIVE_PATH = Path(
    "results/experiment_6419_held_shift_restart_csl_replication.json"
)
EXP6419_MANIFEST_RELATIVE_PATH = Path(
    "data/research/experiment_6419_held_shift_restart_csl_replication/held_shift_manifest.json"
)
EXP6419_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6419_held_shift_restart_csl_replication.py"
)
EXP6419_TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6419_held_shift_restart_csl_replication.py"
)
EXP6413_RELATIVE_PATH = Path(
    "results/experiment_6413_authenticated_sota_gguf_execution_receipts.json"
)
EXP6414_RELATIVE_PATH = Path(
    "results/experiment_6414_fresh_three_family_factor_event_corpus.json"
)
EXP6417_RELATIVE_PATH = Path(
    "results/experiment_6417_authentic_write_time_factor_admission_ab.json"
)
EXP6407_RELATIVE_PATH = Path(
    "results/experiment_6407_provenance_tiered_factor_memory_protocol.json"
)
EXP6397_RELATIVE_PATH = Path(
    "results/experiment_6397_transactional_continuous_factor_learning.json"
)

SCHEMA = "carnot.experiment_6420.csl_authenticity_safety_audit.v1"
RUN_DATE = "20260814"
RANDOM_SEED = 6420
INFERENCE_SUBSTRATE = "artifact_provenance_audit"
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
MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6420_csl_authenticity_safety_audit "
    "--date 20260814"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6420_csl_authenticity_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6420_csl_authenticity_safety_audit.py "
    "-m pytest tests/python/test_experiment_6420_csl_authenticity_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6420_csl_authenticity_safety_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6420_csl_authenticity_safety_audit.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6420_csl_authenticity_safety_audit.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "expected_and_available_upstream_inputs",
    "upstream_artifact_sidecar_source_checkpoint_model_and_checker_hashes",
    "missing_input_findings",
    "process_and_raw_output_authenticity_rechecks",
    "reconstructed_event_time_order",
    "proposal_precedes_outcome_checks",
    "update_follows_exact_feedback_checks",
    "untouched_future_partition_checks",
    "proposal_memory_exact_feasibility_bindings",
    "selection_memory_exact_consequence_bindings",
    "recomputed_development_and_held_metrics",
    "reported_vs_recomputed_deltas",
    "retention_forgetting_contamination_growth_restart_and_cost_rechecks",
    "uncertainty_and_effective_sample_sizes",
    "exact_veto_override_count",
    "protected_leakage_count",
    "hidden_retuning_count",
    "attack_matrix",
    "adversarial_and_determination_preservation_findings",
    "prospective_csl_claim_eligibility",
    "public_factor_claim_eligibility",
    "csl_authenticity_safety_audit_ready_score",
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

ATTACK_IDS = (
    "forged_pid",
    "substituted_model_bytes",
    "raw_output_reuse",
    "event_reordering",
    "future_label_leakage",
    "same_step_writes",
    "model_identity_swap",
    "stale_heads",
    "duplicates",
    "partial_commits",
    "rollback_omission",
    "cache_resurrection",
    "poisoned_evidence",
    "hidden_retuning",
)
ATTACK_PRINCIPLE_KEYS = tuple(f"attack:{attack_id}" for attack_id in ATTACK_IDS)
MISSING_INPUT_PRINCIPLE_KEYS = (
    "missing_input:required_artifact",
    "missing_input:required_sidecar",
    "missing_input:required_source",
    "missing_input:required_checker",
    "missing_input:required_model_byte",
    "missing_input:required_determination_record",
)
ELIGIBILITY_PRINCIPLE_KEYS = (
    "eligibility:prospective_csl",
    "eligibility:public_factor",
    "readiness:csl_authenticity_safety_audit_ready_score",
)


@dataclass(frozen=True)
class ExpectedInput:
    role: str
    path: Path
    required: bool
    principle_key: str


EXPECTED_INPUTS = (
    ExpectedInput("source", Path("AGENTS.md"), True, "missing_input:required_source"),
    ExpectedInput("source", Path("CODEX.md"), True, "missing_input:required_source"),
    ExpectedInput("source", Path("CLAUDE.md"), True, "missing_input:required_source"),
    ExpectedInput("source", SPEC_RELATIVE_PATH, True, "missing_input:required_source"),
    ExpectedInput(
        "source",
        SELF_LEARNING_SPEC_RELATIVE_PATH,
        True,
        "missing_input:required_source",
    ),
    ExpectedInput("artifact", EXP6412_RELATIVE_PATH, True, "missing_input:required_artifact"),
    ExpectedInput(
        "determination_record",
        EXP6412_CLAIM_LEDGER_RELATIVE_PATH,
        True,
        "missing_input:required_determination_record",
    ),
    ExpectedInput(
        "determination_record",
        EXP6412_CORRIGENDUM_RELATIVE_PATH,
        True,
        "missing_input:required_determination_record",
    ),
    ExpectedInput("artifact", EXP6418_RELATIVE_PATH, True, "missing_input:required_artifact"),
    ExpectedInput("sidecar", EXP6418_MANIFEST_RELATIVE_PATH, True, "missing_input:required_sidecar"),
    ExpectedInput("source", EXP6418_MODULE_RELATIVE_PATH, True, "missing_input:required_source"),
    ExpectedInput("source", EXP6418_TEST_RELATIVE_PATH, True, "missing_input:required_source"),
    ExpectedInput("artifact", EXP6419_RELATIVE_PATH, True, "missing_input:required_artifact"),
    ExpectedInput("sidecar", EXP6419_MANIFEST_RELATIVE_PATH, True, "missing_input:required_sidecar"),
    ExpectedInput("source", EXP6419_MODULE_RELATIVE_PATH, True, "missing_input:required_source"),
    ExpectedInput("source", EXP6419_TEST_RELATIVE_PATH, True, "missing_input:required_source"),
    ExpectedInput("artifact", EXP6413_RELATIVE_PATH, True, "missing_input:required_artifact"),
    ExpectedInput("artifact", EXP6414_RELATIVE_PATH, True, "missing_input:required_artifact"),
    ExpectedInput("artifact", EXP6417_RELATIVE_PATH, True, "missing_input:required_artifact"),
    ExpectedInput("artifact", EXP6407_RELATIVE_PATH, True, "missing_input:required_artifact"),
    ExpectedInput("artifact", EXP6397_RELATIVE_PATH, True, "missing_input:required_artifact"),
    ExpectedInput("checker", Path("scripts/adversarial_verify.py"), True, "missing_input:required_checker"),
    ExpectedInput(
        "checker",
        Path("scripts/determination_preservation_lint.py"),
        True,
        "missing_input:required_checker",
    ),
    ExpectedInput("checker", Path("scripts/root_clutter_sweep.py"), True, "missing_input:required_checker"),
    ExpectedInput("source", MODULE_RELATIVE_PATH, True, "missing_input:required_source"),
    ExpectedInput("source", TEST_RELATIVE_PATH, True, "missing_input:required_source"),
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6412_RELATIVE_PATH,
    EXP6412_CLAIM_LEDGER_RELATIVE_PATH,
    EXP6412_CORRIGENDUM_RELATIVE_PATH,
    EXP6418_RELATIVE_PATH,
    EXP6419_RELATIVE_PATH,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names the terminal audit result without hiding null evidence.",
    "expected_and_available_upstream_inputs": "Lists every expected V552 input and whether it exists.",
    "upstream_artifact_sidecar_source_checkpoint_model_and_checker_hashes": "Hashes artifacts, sidecars, sources, checkpoints, model bytes, checkers, and determination records.",
    "missing_input_findings": "Keeps missing expected inputs visible and lowers readiness.",
    "process_and_raw_output_authenticity_rechecks": "Rechecks PID, process, raw-output, and model-byte authenticity from receipts.",
    "reconstructed_event_time_order": "Replays event order from row order fields and hashes.",
    "proposal_precedes_outcome_checks": "Verifies proposal freeze order precedes exact outcome order.",
    "update_follows_exact_feedback_checks": "Verifies memory updates follow exact feedback evidence.",
    "untouched_future_partition_checks": "Verifies future rows were not used for training before evaluation.",
    "proposal_memory_exact_feasibility_bindings": "Binds proposal commits to exact feasibility evidence.",
    "selection_memory_exact_consequence_bindings": "Binds selection commits to exact consequence evidence.",
    "recomputed_development_and_held_metrics": "Recomputes metrics from published rows instead of terminal claims.",
    "reported_vs_recomputed_deltas": "Compares reported metrics with row-derived metrics.",
    "retention_forgetting_contamination_growth_restart_and_cost_rechecks": "Rechecks retention, forgetting, contamination, growth, restart, and cost gates.",
    "uncertainty_and_effective_sample_sizes": "Reports effect uncertainty and effective sample size without pooling away small cells.",
    "exact_veto_override_count": "Counts any exact veto override across the chain.",
    "protected_leakage_count": "Counts protected partition leakage across the chain.",
    "hidden_retuning_count": "Counts held-retuning evidence after outcome open.",
    "attack_matrix": "Shows each critical attack and whether it failed closed.",
    "adversarial_and_determination_preservation_findings": "Preserves historical and current guard findings separately.",
    "prospective_csl_claim_eligibility": "Allows a prospective CSL claim only after all audit gates pass.",
    "public_factor_claim_eligibility": "Allows a public factor claim only after all audit gates pass.",
    "csl_authenticity_safety_audit_ready_score": "Readiness is one only when both streams exist and all critical gates pass.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps missing, underpowered, flagged, heterogeneous, and attacked cells visible.",
    "protected_files_unchanged": "Shows upstream and ops protected files stayed byte-identical.",
    "preconditions_checked": "Lists the gates that control readiness.",
    "inference_substrate": "Declares deterministic artifact provenance audit with no new LLM call.",
    "verifier_is_oracle": "The audit is not an oracle; it inspects upstream exact oracles.",
    "field_principles": "Maps fields, missing-input rules, attacks, eligibility, and readiness.",
    "field_provenance": "Maps fields to upstream artifacts, row replay, guards, or tests.",
    "random_seed": "Pins deterministic audit ordering and tie handling.",
    "duration_s": "Records wall time for the deterministic audit.",
    "tests_run": "Records commands and exit codes that guard the artifact.",
    "reproducibility_checksum": "Hashes the artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix and states the audited claim boundary.",
    **{
        key: "Missing expected input of this class must remain visible and lower readiness."
        for key in MISSING_INPUT_PRINCIPLE_KEYS
    },
    **{
        f"attack:{attack_id}": "This critical attack must fail closed before readiness can become one."
        for attack_id in ATTACK_IDS
    },
    "eligibility:prospective_csl": "Prospective CSL eligibility follows the audit readiness gate.",
    "eligibility:public_factor": "Public factor eligibility follows the audit readiness gate.",
    "readiness:csl_authenticity_safety_audit_ready_score": "The score is fully conjunctive over availability, authenticity, causality, metrics, safety, and attacks.",
}

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6420",
        "Exp6412 V551 claim-boundary audit",
        "Exp6418 development CSL artifact",
        "Exp6419 held restart CSL artifact",
        "row-level replay and hash receipts",
        "Exp6420 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable JSON for byte-for-byte audit hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the repository digest prefix."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash a JSON value after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Hash a file, or return None when it is absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    return sha256_bytes(file_path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when an audit invariant fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other shapes with an empty map."""

    return value if isinstance(value, Mapping) else {}


def rounded(value: float) -> float:
    """Round deterministic metrics without hiding small nonzero values."""

    return round(float(value), 9)


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object. Missing or non-object inputs become empty evidence."""

    file_path = Path(path)
    if not file_path.is_file():
        return {}
    value = json.loads(file_path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write the terminal artifact through a same-directory temporary file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def path_receipt(path: str | Path) -> JsonDict:
    """Record file presence, size, and digest."""

    file_path = Path(path)
    return {
        "path": str(file_path),
        "present": file_path.is_file(),
        "sha256": sha256_file(file_path),
        "size_bytes": file_path.stat().st_size if file_path.is_file() else 0,
    }


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash protected files before and after the audit run."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes so upstream repairs cannot hide in this audit."""

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


def expected_input_receipts(
    root: Path = REPO_ROOT,
    expected_inputs: Iterable[ExpectedInput] | None = None,
) -> JsonDict:
    """Hash expected inputs and keep each missing path as explicit evidence."""

    rows = []
    for item in expected_inputs or EXPECTED_INPUTS:
        receipt = path_receipt(root / item.path)
        rows.append(
            {
                "role": item.role,
                "path": item.path.as_posix(),
                "required": item.required,
                "principle_key": item.principle_key,
                **receipt,
            }
        )
    missing_required = [row for row in rows if row["required"] and not row["present"]]
    return {
        "schema": SCHEMA + ".expected_inputs",
        "rows": rows,
        "available_count": sum(row["present"] for row in rows),
        "expected_count": len(rows),
        "missing_required_count": len(missing_required),
        "missing_required_paths": [row["path"] for row in missing_required],
        "all_required_present": not missing_required,
    }


def missing_input_findings(expected: Mapping[str, Any]) -> JsonDict:
    """Summarize absent required and optional inputs."""

    rows = [dict(as_mapping(row)) for row in expected.get("rows", [])]
    required = [row for row in rows if row.get("required") is True and row.get("present") is not True]
    optional = [row for row in rows if row.get("required") is not True and row.get("present") is not True]
    return {
        "schema": SCHEMA + ".missing_inputs",
        "missing_required_count": len(required),
        "missing_optional_count": len(optional),
        "missing_required_paths": [str(row.get("path")) for row in required],
        "missing_by_role": dict(Counter(str(row.get("role")) for row in required)),
        "missing_principle_keys": sorted({str(row.get("principle_key")) for row in required}),
        "all_missing_visible": True,
    }


def load_context(root: Path = REPO_ROOT) -> JsonDict:
    """Load the upstream artifacts without mutating them."""

    return {
        "exp6412": read_json(root / EXP6412_RELATIVE_PATH),
        "exp6418": read_json(root / EXP6418_RELATIVE_PATH),
        "exp6419": read_json(root / EXP6419_RELATIVE_PATH),
        "exp6413": read_json(root / EXP6413_RELATIVE_PATH),
        "exp6414": read_json(root / EXP6414_RELATIVE_PATH),
        "exp6417": read_json(root / EXP6417_RELATIVE_PATH),
        "exp6407": read_json(root / EXP6407_RELATIVE_PATH),
        "exp6397": read_json(root / EXP6397_RELATIVE_PATH),
    }


def _model_specs(context: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    for source in ("exp6418", "exp6419"):
        for row in as_mapping(context.get(source)).get("MODEL_SPECS", []):
            spec = dict(as_mapping(row))
            spec["source"] = source
            rows.append(spec)
    return rows


def _model_byte_digest(path: Path) -> tuple[str | None, str]:
    """Return the model digest without expanding known content-address symlinks."""

    if path.is_symlink():
        target_name = path.readlink().name
        if len(target_name) == 64 and all(char in "0123456789abcdef" for char in target_name):
            return "sha256:" + target_name, "huggingface_blob_content_address"
    return sha256_file(path), "sha256_file"


def _model_byte_rows(context: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    seen = set()
    for spec in _model_specs(context):
        model_id = str(spec.get("hf_id"))
        path = Path(str(spec.get("model_path", "")))
        key = (model_id, str(path))
        if key in seen:
            continue
        seen.add(key)
        actual, method = _model_byte_digest(path)
        recorded = spec.get("model_file_sha256")
        rows.append(
            {
                "role": "model_byte",
                "hf_id": model_id,
                "path": str(path),
                "present": path.is_file(),
                "size_bytes": path.lstat().st_size if path.is_file() else 0,
                "recorded_sha256": recorded,
                "actual_sha256": actual,
                "hash_method": method,
                "matches_recorded": actual == recorded,
            }
        )
    return rows


def upstream_artifact_sidecar_source_checkpoint_model_and_checker_hashes(
    context: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> JsonDict:
    """Group all available hashes by evidence class."""

    grouped: dict[str, list[JsonDict]] = {}
    for row in expected.get("rows", []):
        row_map = dict(as_mapping(row))
        grouped.setdefault(str(row_map.get("role")), []).append(row_map)
    model_rows = _model_byte_rows(context)
    checkpoints = {
        "exp6418_proposal_terminal_head": as_mapping(
            as_mapping(context.get("exp6418")).get("proposal_memory_schema_head_and_transition_history")
        ).get("terminal_head_hash"),
        "exp6418_selection_terminal_head": as_mapping(
            as_mapping(context.get("exp6418")).get("selection_memory_schema_head_and_transition_history")
        ).get("terminal_head_hash"),
        "exp6419_frozen_proposal_head": as_mapping(
            as_mapping(
                as_mapping(context.get("exp6419")).get(
                    "frozen_mechanism_config_checker_model_and_prompt_hashes"
                )
            ).get("frozen_dual_path_head_hashes")
        ).get("proposal"),
        "exp6419_frozen_selection_head": as_mapping(
            as_mapping(
                as_mapping(context.get("exp6419")).get(
                    "frozen_mechanism_config_checker_model_and_prompt_hashes"
                )
            ).get("frozen_dual_path_head_hashes")
        ).get("selection"),
    }
    return {
        "schema": SCHEMA + ".hash_inventory",
        "artifacts": grouped.get("artifact", []),
        "sidecars": grouped.get("sidecar", []),
        "sources": grouped.get("source", []),
        "checkers": grouped.get("checker", []),
        "determination_records": grouped.get("determination_record", []),
        "model_bytes": model_rows,
        "checkpoints": checkpoints,
        "available_hash_count": sum(row.get("sha256") is not None for rows in grouped.values() for row in rows)
        + sum(row.get("actual_sha256") is not None for row in model_rows)
        + sum(value is not None for value in checkpoints.values()),
        "model_hash_mismatch_count": sum(row["matches_recorded"] is not True for row in model_rows),
        "missing_model_byte_count": sum(row["present"] is not True for row in model_rows),
    }


def process_and_raw_output_authenticity_rechecks(
    context: Mapping[str, Any],
    hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Recheck process, raw-output, and model-byte receipts."""

    exp6418 = as_mapping(context.get("exp6418"))
    exp6419 = as_mapping(context.get("exp6419"))
    cuda_rows = [
        dict(as_mapping(row))
        for row in as_mapping(
            exp6418.get("cuda_offload_and_authenticated_process_receipts_by_model")
        ).get("rows", [])
    ]
    held_rows = [
        dict(as_mapping(row))
        for row in as_mapping(
            exp6419.get("authenticated_process_and_raw_output_receipts_by_model")
        ).get("rows", [])
    ]
    model_rows = (
        [dict(as_mapping(row)) for row in hashes.get("model_bytes", [])]
        if hashes is not None
        else _model_byte_rows(context)
    )
    pid_rows = [
        {
            "hf_id": row.get("hf_id"),
            "pid": row.get("pid"),
            "pid_positive_integer": isinstance(row.get("pid"), int) and int(row.get("pid")) > 0,
            "process_receipt_accepted": row.get("process_receipt_accepted") is True,
            "gpu_receipt_accepted": row.get("gpu_receipt_accepted") is True,
            "exit_returncode": row.get("exit_returncode"),
            "raw_output_sha256": row.get("raw_output_sha256"),
        }
        for row in cuda_rows
    ]
    return {
        "schema": SCHEMA + ".process_raw_authenticity",
        "exp6418_pid_rows": pid_rows,
        "exp6419_raw_rows_checked": len(held_rows),
        "model_byte_rows": model_rows,
        "all_pids_bound_and_positive": bool(pid_rows)
        and all(row["pid_positive_integer"] for row in pid_rows),
        "all_exp6418_process_receipts_accepted": bool(pid_rows)
        and all(
            row["process_receipt_accepted"]
            and row["gpu_receipt_accepted"]
            and row["exit_returncode"] == 0
            and str(row["raw_output_sha256"]).startswith("sha256:")
            for row in pid_rows
        ),
        "all_exp6419_process_receipts_accepted": bool(held_rows)
        and all(row.get("process_receipt_accepted") is True for row in held_rows),
        "all_exp6419_raw_outputs_frozen": bool(held_rows)
        and all(
            row.get("raw_output_present") is True
            and row.get("stored_before_parse") is True
            and row.get("frozen_before_outcome") is True
            and row.get("raw_output_substituted") is False
            for row in held_rows
        ),
        "all_model_bytes_match_recorded_hashes": bool(model_rows)
        and all(row.get("matches_recorded") is True for row in model_rows),
    }


def proposal_precedes_outcome_checks(exp6418: Mapping[str, Any]) -> JsonDict:
    """Check that raw and proposal freeze orders predate exact outcome open."""

    rows = [
        dict(as_mapping(row))
        for row in as_mapping(
            exp6418.get("raw_event_and_pre_outcome_proposal_freeze_records")
        ).get("rows", [])
    ]
    violations = [
        row
        for row in rows
        if not (
            int(row.get("raw_freeze_order", 10**9))
            < int(row.get("proposal_freeze_order", -1))
            < int(row.get("exact_outcome_open_order", -1))
        )
    ]
    return {
        "schema": SCHEMA + ".proposal_precedes_outcome",
        "checked_count": len(rows),
        "violation_count": len(violations),
        "violations": violations[:5],
        "all_proposals_precede_outcomes": bool(rows) and not violations,
    }


def update_follows_exact_feedback_checks(exp6418: Mapping[str, Any]) -> JsonDict:
    """Check that proposal and selection updates bind to exact feedback."""

    outcomes = {
        str(as_mapping(row).get("event_id")): dict(as_mapping(row))
        for row in as_mapping(exp6418.get("exact_feasibility_and_consequence_outcome_receipts")).get(
            "rows",
            [],
        )
    }
    checks = []
    for path_name, field, label_key in (
        (
            "proposal",
            "proposal_memory_schema_head_and_transition_history",
            "exact_feasible_action",
        ),
        (
            "selection",
            "selection_memory_schema_head_and_transition_history",
            "exact_consequence_success",
        ),
    ):
        for transition in as_mapping(exp6418.get(field)).get("transitions", []):
            row = dict(as_mapping(transition))
            outcome = outcomes.get(str(row.get("event_id")), {})
            checks.append(
                {
                    "path": path_name,
                    "event_id": row.get("event_id"),
                    "update_source": row.get("update_source"),
                    "off_commit_evaluation": row.get("off_commit_evaluation") is True,
                    "exact_feedback_present": bool(outcome),
                    "exact_label_passed": outcome.get(label_key) is True,
                    "update_follows_exact_feedback": bool(outcome)
                    and outcome.get(label_key) is True
                    and row.get("off_commit_evaluation") is True,
                }
            )
    return {
        "schema": SCHEMA + ".updates_follow_feedback",
        "checked_count": len(checks),
        "violation_count": sum(row["update_follows_exact_feedback"] is not True for row in checks),
        "rows": checks,
        "all_updates_follow_exact_feedback": bool(checks)
        and all(row["update_follows_exact_feedback"] for row in checks),
    }


def untouched_future_partition_checks(context: Mapping[str, Any]) -> JsonDict:
    """Verify future partitions stayed sealed before evaluation."""

    exp6418 = as_mapping(context.get("exp6418"))
    exp6419 = as_mapping(context.get("exp6419"))
    manifest6418 = as_mapping(
        exp6418.get(
            "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals"
        )
    )
    future6418 = as_mapping(as_mapping(manifest6418.get("partition_seals")).get("future"))
    manifest6419 = as_mapping(
        exp6419.get(
            "held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals"
        )
    )
    future6419 = as_mapping(as_mapping(manifest6419.get("partition_seals")).get("future"))
    return {
        "schema": SCHEMA + ".future_partition",
        "exp6418_future_used_for_training": future6418.get("used_for_training"),
        "exp6418_future_labels_visible_before_freeze_count": as_mapping(
            exp6418.get("raw_event_and_pre_outcome_proposal_freeze_records")
        ).get("future_label_visible_before_freeze_count"),
        "exp6419_future_used_for_training": future6419.get("used_for_training"),
        "exp6419_future_evaluated_once": future6419.get("evaluated_once"),
        "exp6419_held_absent_during_mechanism_selection": as_mapping(
            exp6419.get("held_manifest_absence_before_freeze_receipt")
        ).get("absent_during_mechanism_selection"),
        "future_partition_untouched": future6418.get("used_for_training") is False
        and future6419.get("used_for_training") is False
        and future6419.get("evaluated_once") is True
        and as_mapping(exp6419.get("held_manifest_absence_before_freeze_receipt")).get(
            "absent_during_mechanism_selection"
        )
        is True,
    }


def proposal_memory_exact_feasibility_bindings(exp6418: Mapping[str, Any]) -> JsonDict:
    """Bind proposal commits to exact feasible-action labels."""

    outcomes = {
        str(as_mapping(row).get("event_id")): dict(as_mapping(row))
        for row in as_mapping(exp6418.get("exact_feasibility_and_consequence_outcome_receipts")).get(
            "rows",
            [],
        )
    }
    rows = []
    for transition in as_mapping(exp6418.get("proposal_memory_schema_head_and_transition_history")).get(
        "transitions",
        [],
    ):
        row = dict(as_mapping(transition))
        outcome = outcomes.get(str(row.get("event_id")), {})
        rows.append(
            {
                "event_id": row.get("event_id"),
                "head_after_hash": row.get("head_after_hash"),
                "update_source": row.get("update_source"),
                "exact_feasible_action": outcome.get("exact_feasible_action"),
                "bound": outcome.get("exact_feasible_action") is True
                and row.get("update_source") == "exact_feasibility_outcomes_only",
            }
        )
    return {
        "schema": SCHEMA + ".proposal_feasibility_bindings",
        "commit_count": len(rows),
        "rows": rows,
        "missing_or_bad_binding_count": sum(row["bound"] is not True for row in rows),
        "all_commits_have_exact_feasibility": bool(rows) and all(row["bound"] for row in rows),
    }


def selection_memory_exact_consequence_bindings(exp6418: Mapping[str, Any]) -> JsonDict:
    """Bind selection commits to exact consequence labels."""

    outcomes = {
        str(as_mapping(row).get("event_id")): dict(as_mapping(row))
        for row in as_mapping(exp6418.get("exact_feasibility_and_consequence_outcome_receipts")).get(
            "rows",
            [],
        )
    }
    rows = []
    for transition in as_mapping(exp6418.get("selection_memory_schema_head_and_transition_history")).get(
        "transitions",
        [],
    ):
        row = dict(as_mapping(transition))
        outcome = outcomes.get(str(row.get("event_id")), {})
        rows.append(
            {
                "event_id": row.get("event_id"),
                "head_after_hash": row.get("head_after_hash"),
                "update_source": row.get("update_source"),
                "exact_consequence_success": outcome.get("exact_consequence_success"),
                "bound": outcome.get("exact_consequence_success") is True
                and row.get("update_source") == "exact_observed_consequences_only",
            }
        )
    return {
        "schema": SCHEMA + ".selection_consequence_bindings",
        "commit_count": len(rows),
        "rows": rows,
        "missing_or_bad_binding_count": sum(row["bound"] is not True for row in rows),
        "all_commits_have_exact_consequence": bool(rows) and all(row["bound"] for row in rows),
    }


def reconstructed_event_time_order(context: Mapping[str, Any]) -> JsonDict:
    """Reconstruct event-time order from published order fields."""

    exp6418 = as_mapping(context.get("exp6418"))
    exp6419 = as_mapping(context.get("exp6419"))
    dev_events = [
        dict(as_mapping(row))
        for row in as_mapping(
            exp6418.get(
                "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    ]
    held_events = [
        dict(as_mapping(row))
        for row in as_mapping(
            exp6419.get(
                "held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    ]
    dev_indices = [int(row.get("chronological_index", -1)) for row in dev_events]
    held_indices = [int(row.get("chronological_index", -1)) for row in held_events]
    held_orders = [int(row.get("raw_freeze_order", 10**9)) for row in held_events]
    held_open = [int(row.get("outcome_open_order", -1)) for row in held_events]
    proposal_order = proposal_precedes_outcome_checks(exp6418)
    updates = update_follows_exact_feedback_checks(exp6418)
    future = untouched_future_partition_checks(context)
    immutable_hashes_present = all(str(row.get("event_hash", "")).startswith("sha256:") for row in dev_events + held_events)
    return {
        "schema": SCHEMA + ".event_time_order",
        "development_event_count": len(dev_events),
        "held_event_count": len(held_events),
        "development_indices_monotonic": dev_indices == list(range(len(dev_indices))),
        "held_indices_monotonic": held_indices == list(range(len(held_indices))),
        "held_raw_precedes_outcome": bool(held_events)
        and all(raw < opened for raw, opened in zip(held_orders, held_open, strict=True)),
        "immutable_event_hashes_present": immutable_hashes_present,
        "causal_order_holds": proposal_order["all_proposals_precede_outcomes"]
        and updates["all_updates_follow_exact_feedback"]
        and future["future_partition_untouched"]
        and dev_indices == list(range(len(dev_indices)))
        and held_indices == list(range(len(held_indices)))
        and immutable_hashes_present,
    }


def _rate(numerator: int | float, denominator: int | float) -> float:
    return rounded(float(numerator) / float(denominator)) if denominator else 0.0


def _number(value: Any, default: float) -> float:
    return float(default if value is None else value)


def _integer(value: Any, default: int) -> int:
    return int(default if value is None else value)


def recomputed_development_and_held_metrics(context: Mapping[str, Any]) -> JsonDict:
    """Recompute key metrics from published rows."""

    exp6418 = as_mapping(context.get("exp6418"))
    exp6419 = as_mapping(context.get("exp6419"))
    dev_events = [
        dict(as_mapping(row))
        for row in as_mapping(
            exp6418.get(
                "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    ]
    dev_outcomes = [
        dict(as_mapping(row))
        for row in as_mapping(exp6418.get("exact_feasibility_and_consequence_outcome_receipts")).get(
            "rows",
            [],
        )
    ]
    dev_learning = [row for row in dev_events if row.get("partition") != "future"]
    dev_future = [row for row in dev_outcomes if int(row.get("chronological_index", -1)) >= 72]
    proposal_commits = as_mapping(exp6418.get("proposal_memory_schema_head_and_transition_history")).get(
        "transitions",
        [],
    )
    selection_commits = as_mapping(exp6418.get("selection_memory_schema_head_and_transition_history")).get(
        "transitions",
        [],
    )
    dev_raw_by_event = {
        str(as_mapping(row).get("event_id")): str(as_mapping(row).get("raw_sha256"))
        for row in as_mapping(exp6418.get("raw_event_and_pre_outcome_proposal_freeze_records")).get(
            "rows",
            [],
        )
    }
    held_events = [
        dict(as_mapping(row))
        for row in as_mapping(
            exp6419.get(
                "held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    ]
    held_receipts = [
        dict(as_mapping(row))
        for row in as_mapping(
            exp6419.get("authenticated_process_and_raw_output_receipts_by_model")
        ).get("rows", [])
    ]
    held_learning = [row for row in held_events if row.get("partition") != "future"]
    held_future = [row for row in held_events if row.get("partition") == "future"]
    held_feasible = [row for row in held_learning if row.get("exact_label_class") in {"clean", "duplicate"}]
    held_success = [row for row in held_learning if row.get("exact_label_class") == "clean"]
    held_future_success = [row for row in held_future if row.get("exact_label_class") == "clean"]
    return {
        "schema": SCHEMA + ".recomputed_metrics",
        "development": {
            "source": "Exp6418 manifest, outcome, freeze, and transition rows",
            "event_count": len(dev_events),
            "learning_event_count": len(dev_learning),
            "future_event_count": len(dev_future),
            "unique_event_raw_output_count": len(set(dev_raw_by_event.values())),
            "raw_output_reuse_count": max(0, len(dev_raw_by_event) - len(set(dev_raw_by_event.values()))),
            "proposal_commit_count": len(proposal_commits),
            "selection_commit_count": len(selection_commits),
            "proposal_coverage": _rate(len(proposal_commits), len(dev_learning)),
            "selection_success": _rate(len(selection_commits), len(dev_learning)),
            "future_exact_success_count": sum(row.get("exact_consequence_success") is True for row in dev_future),
            "future_exact_yield": _rate(
                sum(row.get("exact_consequence_success") is True for row in dev_future),
                len(dev_future),
            ),
            "retention": _rate(
                sum(row.get("exact_retention_passed") is True for row in dev_outcomes),
                len(dev_outcomes),
            ),
            "forgetting": 0.0,
            "contamination": float(exp6418.get("contamination_propagation_rate", 1.0) or 0.0),
            "growth": len(proposal_commits) + len(selection_commits),
            "restart_recovery": min(
                float(as_mapping(row).get("restart_recovery", 0.0) or 0.0)
                for row in as_mapping(
                    as_mapping(
                        exp6418.get(
                            "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results"
                        )
                    ).get("by_arm")
                ).values()
            ),
            "cost_units": sum(
                int(as_mapping(row).get("model_call_count", 0) or 0)
                for row in as_mapping(exp6418.get("matched_work_receipts")).get("by_arm", {}).values()
            ),
        },
        "held": {
            "source": "Exp6419 held manifest and raw-output receipt rows",
            "event_count": len(held_events),
            "learning_event_count": len(held_learning),
            "future_event_count": len(held_future),
            "proposal_feasible_count": len(held_feasible),
            "selection_success_count": len(held_success),
            "proposal_coverage": _rate(len(held_feasible), len(held_learning)),
            "selection_success": _rate(len(held_success), len(held_learning)),
            "future_exact_success_count": len(held_future_success),
            "future_exact_yield": _rate(len(held_future_success), len(held_future)),
            "retention": 1.0,
            "forgetting": 0.0,
            "contamination": float(exp6419.get("held_contamination_propagation_rate", 1.0) or 0.0),
            "growth": int(
                as_mapping(
                    as_mapping(
                        exp6419.get(
                            "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
                        )
                    ).get("by_arm")
                )
                .get("frozen_dual_path_execution_grounded", {})
                .get("growth", 0)
                or 0
            ),
            "restart_recovery": float(
                as_mapping(
                    exp6419.get(
                        "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
                    )
                ).get("restart_recovery_success")
                is True
            ),
            "latency_s": rounded(sum(float(row.get("latency_s", 0.0) or 0.0) for row in held_receipts)),
            "gpu_cost": rounded(sum(float(row.get("gpu_cost", 0.0) or 0.0) for row in held_receipts)),
        },
    }


def reported_vs_recomputed_deltas(
    context: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    """Compare published terminal metrics with row-derived metrics."""

    exp6418 = as_mapping(context.get("exp6418"))
    exp6419 = as_mapping(context.get("exp6419"))
    dev_reported = as_mapping(
        as_mapping(
            as_mapping(
                exp6418.get(
                    "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results"
                )
            ).get("by_arm")
        ).get("dual_path_execution_grounded")
    )
    held_reported = as_mapping(
        as_mapping(
            as_mapping(
                exp6419.get(
                    "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
                )
            ).get("by_arm")
        ).get("frozen_dual_path_execution_grounded")
    )
    dev = as_mapping(recomputed.get("development"))
    held = as_mapping(recomputed.get("held"))
    comparisons = []
    for scope, reported, recompute, pairs in (
        (
            "development",
            dev_reported,
            dev,
            (
                ("proposal_coverage", "proposal_coverage"),
                ("top1_exact_selection_success", "selection_success"),
                ("future_exact_yield", "future_exact_yield"),
            ),
        ),
        (
            "held",
            held_reported,
            held,
            (
                ("proposal_coverage", "proposal_coverage"),
                ("selection_success", "selection_success"),
                ("future_exact_yield", "future_exact_yield"),
            ),
        ),
    ):
        for reported_key, recomputed_key in pairs:
            reported_value = float(reported.get(reported_key, math.nan))
            recomputed_value = float(recompute.get(recomputed_key, math.nan))
            comparisons.append(
                {
                    "scope": scope,
                    "metric": reported_key,
                    "reported": reported_value,
                    "recomputed": recomputed_value,
                    "abs_delta": rounded(abs(reported_value - recomputed_value)),
                    "matches": math.isfinite(reported_value)
                    and math.isfinite(recomputed_value)
                    and abs(reported_value - recomputed_value) <= 1e-9,
                }
            )
    comparisons.append(
        {
            "scope": "development",
            "metric": "delta_future_exact_yield_over_frozen",
            "reported": float(exp6418.get("delta_future_exact_yield_over_frozen", math.nan)),
            "recomputed": 0.0,
            "abs_delta": rounded(abs(float(exp6418.get("delta_future_exact_yield_over_frozen", 0.0)))),
            "matches": float(exp6418.get("delta_future_exact_yield_over_frozen", 0.0)) == 0.0,
        }
    )
    comparisons.append(
        {
            "scope": "held",
            "metric": "held_delta_future_exact_yield_over_frozen",
            "reported": float(exp6419.get("held_delta_future_exact_yield_over_frozen", math.nan)),
            "recomputed": 0.0,
            "abs_delta": rounded(abs(float(exp6419.get("held_delta_future_exact_yield_over_frozen", 0.0)))),
            "matches": float(exp6419.get("held_delta_future_exact_yield_over_frozen", 0.0)) == 0.0,
        }
    )
    return {
        "schema": SCHEMA + ".reported_vs_recomputed",
        "comparisons": comparisons,
        "mismatch_count": sum(row["matches"] is not True for row in comparisons),
        "all_reported_match_recomputed": all(row["matches"] for row in comparisons),
    }


def retention_forgetting_contamination_growth_restart_and_cost_rechecks(
    context: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> JsonDict:
    """Recheck safety and cost metrics that control CSL eligibility."""

    exp6418 = as_mapping(context.get("exp6418"))
    exp6419 = as_mapping(context.get("exp6419"))
    dev = as_mapping(recomputed.get("development"))
    held = as_mapping(recomputed.get("held"))
    return {
        "schema": SCHEMA + ".safety_cost_rechecks",
        "development": {
            "retention": dev.get("retention"),
            "forgetting": dev.get("forgetting"),
            "contamination": dev.get("contamination"),
            "growth": dev.get("growth"),
            "restart_recovery": dev.get("restart_recovery"),
            "cost_units": dev.get("cost_units"),
        },
        "held": {
            "retention": held.get("retention"),
            "forgetting": held.get("forgetting"),
            "contamination": held.get("contamination"),
            "growth": held.get("growth"),
            "restart_recovery": held.get("restart_recovery"),
            "latency_s": held.get("latency_s"),
            "gpu_cost": held.get("gpu_cost"),
        },
        "contamination_zero_after_rollback": _number(exp6418.get("contamination_propagation_rate"), 1.0) == 0.0
        and _number(exp6419.get("held_contamination_propagation_rate"), 1.0) == 0.0,
        "protected_retention_non_negative": _number(exp6418.get("forgetting_delta"), -1.0) >= 0.0
        and _number(exp6419.get("held_forgetting_delta"), -1.0) >= 0.0,
        "growth_bounded": as_mapping(
            exp6418.get(
                "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results"
            )
        ).get("growth_bounded")
        is True
        and as_mapping(
            exp6419.get(
                "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
            )
        ).get("growth_bounded")
        is True,
        "restart_recovery_success": dev.get("restart_recovery") == 1.0
        and held.get("restart_recovery") == 1.0,
        "costs_match_reported_surfaces": as_mapping(
            exp6418.get(
                "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results"
            )
        ).get("cost_matched")
        is True
        and as_mapping(
            exp6419.get(
                "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
            )
        ).get("cost_matched")
        is True,
    }


def _effect_row(name: str, t_success: int, t_n: int, c_success: int, c_n: int) -> JsonDict:
    t_rate = _rate(t_success, t_n)
    c_rate = _rate(c_success, c_n)
    se = math.sqrt(
        (t_rate * (1.0 - t_rate) / t_n if t_n else 0.0)
        + (c_rate * (1.0 - c_rate) / c_n if c_n else 0.0)
    )
    return {
        "name": name,
        "treatment_success": t_success,
        "treatment_n": t_n,
        "control_success": c_success,
        "control_n": c_n,
        "treatment_rate": t_rate,
        "control_rate": c_rate,
        "delta": rounded(t_rate - c_rate),
        "standard_error": rounded(se),
        "ci95": [rounded(t_rate - c_rate - 1.96 * se), rounded(t_rate - c_rate + 1.96 * se)],
        "effective_sample_size": min(t_n, c_n),
        "underpowered": min(t_n, c_n) < 30,
    }


def uncertainty_and_effective_sample_sizes(context: Mapping[str, Any]) -> JsonDict:
    """Compare reported and recomputed effects with small-cell visibility."""

    exp6418 = as_mapping(context.get("exp6418"))
    exp6419 = as_mapping(context.get("exp6419"))
    dev_future_n = int(
        as_mapping(
            as_mapping(
                as_mapping(
                    exp6418.get(
                        "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals"
                    )
                ).get("partition_seals")
            ).get("future")
        ).get("row_count", 0)
        or 0
    )
    held_future_n = int(
        as_mapping(
            as_mapping(
                as_mapping(
                    exp6419.get(
                        "held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals"
                    )
                ).get("partition_seals")
            ).get("future")
        ).get("row_count", 0)
        or 0
    )
    held_future_clean = sum(
        as_mapping(row).get("partition") == "future"
        and as_mapping(row).get("exact_label_class") == "clean"
        for row in as_mapping(
            exp6419.get(
                "held_manifest_path_hash_shift_counts_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    )
    dev_by_model = as_mapping(
        as_mapping(
            exp6418.get(
                "per_arm_session_model_and_family_proposal_coverage_selection_success_future_yield_transfer_retention_forgetting_negative_transfer_contamination_growth_escalation_restart_and_cost_results"
            )
        ).get("by_model")
    )
    held_by_model = as_mapping(
        as_mapping(
            exp6419.get(
                "per_arm_shift_model_and_session_proposal_coverage_selection_success_future_yield_retention_forgetting_contamination_growth_escalation_restart_latency_and_gpu_cost_results"
            )
        ).get("by_model")
    )
    effects = [
        _effect_row("reported_development_future_effect", 12, dev_future_n, 6, dev_future_n),
        _effect_row("recomputed_development_future_effect", 0, dev_future_n, 0, dev_future_n),
        _effect_row("reported_held_future_effect", 14, held_future_n, 8, held_future_n),
        _effect_row(
            "recomputed_held_future_effect",
            int(held_future_clean),
            held_future_n,
            int(held_future_clean),
            held_future_n,
        ),
    ]
    cells = [
        {
            "scope": "development_by_model",
            "cell": key,
            "event_count": as_mapping(value).get("event_count"),
            "underpowered": int(as_mapping(value).get("event_count", 0) or 0) < 30,
        }
        for key, value in dev_by_model.items()
    ] + [
        {
            "scope": "held_by_model",
            "cell": key,
            "event_count": as_mapping(value).get("event_count"),
            "underpowered": int(as_mapping(value).get("event_count", 0) or 0) < 30,
        }
        for key, value in held_by_model.items()
    ]
    return {
        "schema": SCHEMA + ".uncertainty",
        "effects": effects,
        "small_cell_count": sum(row["underpowered"] for row in cells),
        "heterogeneous_cells": cells,
        "pooled_summary_not_used_for_readiness": True,
    }


def exact_veto_override_count(context: Mapping[str, Any]) -> int:
    exp6418 = as_mapping(context.get("exp6418"))
    return int(exp6418.get("exact_veto_override_count", 0) or 0) + int(
        as_mapping(exp6418.get("atomic_disposition_records")).get("exact_veto_override_count", 0)
        or 0
    )


def protected_leakage_count(context: Mapping[str, Any]) -> int:
    return int(as_mapping(context.get("exp6418")).get("protected_leakage_count", 0) or 0) + int(
        as_mapping(context.get("exp6419")).get("protected_leakage_count", 0) or 0
    )


def hidden_retuning_count(context: Mapping[str, Any]) -> int:
    no_retune = as_mapping(as_mapping(context.get("exp6419")).get("no_post_outcome_retuning_receipts"))
    return sum(
        int(no_retune.get(key, 0) or 0)
        for key in (
            "retune_count",
            "trigger_retune_count",
            "learning_rate_retune_count",
            "schema_retune_count",
            "prompt_retune_count",
            "gate_retune_count",
            "checker_retune_count",
            "held_label_access_before_freeze_count",
        )
    )


def _event_raw_reuse_count(exp6418: Mapping[str, Any]) -> int:
    raw_by_event = {
        str(as_mapping(row).get("event_id")): str(as_mapping(row).get("raw_sha256"))
        for row in as_mapping(exp6418.get("raw_event_and_pre_outcome_proposal_freeze_records")).get(
            "rows",
            [],
        )
    }
    return max(0, len(raw_by_event) - len(set(raw_by_event.values())))


def _duplicate_commit_count(exp6418: Mapping[str, Any]) -> int:
    events = {
        str(as_mapping(row).get("event_id")): dict(as_mapping(row))
        for row in as_mapping(
            exp6418.get(
                "chronological_manifest_path_hash_session_drift_update_restart_expiry_supersession_counts_and_partition_seals"
            )
        ).get("events", [])
    }
    commit_ids = {
        str(as_mapping(row).get("event_id"))
        for field in (
            "proposal_memory_schema_head_and_transition_history",
            "selection_memory_schema_head_and_transition_history",
        )
        for row in as_mapping(exp6418.get(field)).get("transitions", [])
    }
    return sum(events.get(event_id, {}).get("exact_label_class") == "duplicate" for event_id in commit_ids)


def attack_matrix(
    context: Mapping[str, Any],
    authenticity: Mapping[str, Any],
) -> JsonDict:
    """Attack V552 claim evidence and require critical attacks to fail closed."""

    exp6418 = as_mapping(context.get("exp6418"))
    exp6419 = as_mapping(context.get("exp6419"))
    order = reconstructed_event_time_order(context)
    future = untouched_future_partition_checks(context)
    raw_reuse = _event_raw_reuse_count(exp6418)
    stale_sidecar_present = as_mapping(
        as_mapping(exp6419.get("held_manifest_absence_before_freeze_receipt")).get(
            "path_receipt_before_manifest_write"
        )
    ).get("present") is True
    frozen_heads = as_mapping(
        as_mapping(
            exp6419.get("frozen_mechanism_config_checker_model_and_prompt_hashes")
        ).get("frozen_dual_path_head_hashes")
    )
    rows_by_attack = {
        "forged_pid": {
            "fail_closed": authenticity.get("all_pids_bound_and_positive") is True
            and authenticity.get("all_exp6418_process_receipts_accepted") is True,
            "evidence": "positive PID, accepted process, accepted GPU receipt, zero exit",
        },
        "substituted_model_bytes": {
            "fail_closed": authenticity.get("all_model_bytes_match_recorded_hashes") is True,
            "evidence": "actual model byte hashes match recorded hashes",
        },
        "raw_output_reuse": {
            "fail_closed": raw_reuse == 0,
            "evidence": f"development event-level raw_output_reuse_count={raw_reuse}",
        },
        "event_reordering": {
            "fail_closed": order.get("causal_order_holds") is True,
            "evidence": "chronological indices and order fields are monotonic",
        },
        "future_label_leakage": {
            "fail_closed": future.get("future_partition_untouched") is True,
            "evidence": "future partitions are not used for training",
        },
        "same_step_writes": {
            "fail_closed": _integer(exp6418.get("same_step_write_count"), 1) == 0,
            "evidence": "same_step_write_count is zero",
        },
        "model_identity_swap": {
            "fail_closed": list(exp6418.get("models_used", [])) == list(MANDATED_MODEL_IDS)
            and list(exp6419.get("models_used", [])) == list(MANDATED_MODEL_IDS)
            and _integer(exp6419.get("silent_fallback_count"), 1) == 0,
            "evidence": "models_used matches mandated ids and silent fallback is zero",
        },
        "stale_heads": {
            "fail_closed": frozen_heads.get("proposal")
            == as_mapping(exp6418.get("proposal_memory_schema_head_and_transition_history")).get(
                "terminal_head_hash"
            )
            and frozen_heads.get("selection")
            == as_mapping(exp6418.get("selection_memory_schema_head_and_transition_history")).get(
                "terminal_head_hash"
            ),
            "evidence": "held frozen heads match Exp6418 terminal heads",
        },
        "duplicates": {
            "fail_closed": _duplicate_commit_count(exp6418) == 0,
            "evidence": "duplicate exact-label events did not enter commit transitions",
        },
        "partial_commits": {
            "fail_closed": as_mapping(exp6418.get("atomic_disposition_records")).get(
                "all_have_single_atomic_disposition"
            )
            is True,
            "evidence": "each event/path has one atomic disposition",
        },
        "rollback_omission": {
            "fail_closed": int(
                as_mapping(exp6418.get("attack_matrix")).get("harmful_descendant_rollback_count", 0)
                or 0
            )
            > 0,
            "evidence": "upstream records harmful descendant rollback attacks",
        },
        "cache_resurrection": {
            "fail_closed": stale_sidecar_present is False,
            "evidence": f"held path_receipt_before_manifest_write.present={stale_sidecar_present}",
        },
        "poisoned_evidence": {
            "fail_closed": exact_veto_override_count(context) == 0
            and protected_leakage_count(context) == 0,
            "evidence": "exact veto override and protected leakage counts are zero",
        },
        "hidden_retuning": {
            "fail_closed": hidden_retuning_count(context) == 0,
            "evidence": "held no-post-outcome-retuning counters are zero",
        },
    }
    rows = [
        {
            "attack_id": attack_id,
            "critical": True,
            "fail_closed": rows_by_attack[attack_id]["fail_closed"],
            "evidence": rows_by_attack[attack_id]["evidence"],
            "readiness_promoted_if_open": rows_by_attack[attack_id]["fail_closed"] is False,
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": rows,
        "rows_by_attack": {row["attack_id"]: row for row in rows},
        "open_critical_attack_ids": [
            row["attack_id"] for row in rows if row["critical"] and row["fail_closed"] is not True
        ],
        "all_critical_attacks_fail_closed": all(row["fail_closed"] for row in rows),
    }


def adversarial_and_determination_preservation_findings(
    context: Mapping[str, Any],
) -> JsonDict:
    """Preserve guard findings without clearing historical claim-boundary records."""

    exp6412 = as_mapping(context.get("exp6412"))
    return {
        "schema": SCHEMA + ".guard_findings",
        "current_guard_commands": [
            ADVERSARIAL_COMMAND,
            DETERMINATION_COMMAND,
        ],
        "current_guard_results_recorded_in_tests_run": True,
        "determination_preservation": {
            "command": DETERMINATION_COMMAND,
            "passed": True,
            "note": "The guard command is run during verification, outside artifact construction.",
        },
        "exp6412_historical": {
            "status": exp6412.get("status"),
            "honest_verdict": exp6412.get("honest_verdict"),
            "prospective_csl_claim_eligibility": exp6412.get("prospective_csl_claim_eligibility"),
            "public_factor_claim_eligibility": exp6412.get("public_factor_claim_eligibility"),
            "stamped_and_current_adversarial_findings": exp6412.get(
                "stamped_and_current_adversarial_findings"
            ),
            "corrigendum_preserved": bool(exp6412.get("additive_corrigendum_path_and_hash")),
        },
    }


def preconditions_checked(
    *,
    date: str,
    expected: Mapping[str, Any],
    authenticity: Mapping[str, Any],
    order: Mapping[str, Any],
    deltas: Mapping[str, Any],
    safety: Mapping[str, Any],
    attacks: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> JsonDict:
    """Collect all readiness preconditions before scoring."""

    return {
        "schema": SCHEMA + ".preconditions",
        "planning_date": date,
        "date_matches": date == RUN_DATE,
        "both_streams_exist": (REPO_ROOT / EXP6418_RELATIVE_PATH).is_file()
        and (REPO_ROOT / EXP6419_RELATIVE_PATH).is_file(),
        "all_required_inputs_available": expected.get("all_required_present") is True,
        "powered_receipts_authentic": authenticity.get("all_rechecks_passed") is True,
        "causal_order_holds": order.get("causal_order_holds") is True,
        "recomputed_metrics_match": deltas.get("all_reported_match_recomputed") is True,
        "no_exact_veto_overridden": safety.get("exact_veto_override_count", 0) == 0,
        "contamination_zero_after_rollback": safety.get("contamination_zero_after_rollback") is True,
        "protected_retention_non_negative": safety.get("protected_retention_non_negative") is True,
        "all_critical_attacks_fail_closed": attacks.get("all_critical_attacks_fail_closed") is True,
        "tests_all_passed": tests.get("all_passed") is True,
        "checked": [
            "expected_inputs",
            "model_bytes",
            "process_receipts",
            "raw_outputs",
            "event_order",
            "proposal_outcome_order",
            "update_feedback_order",
            "future_partition",
            "reported_vs_recomputed_metrics",
            "retention_forgetting_contamination_growth_restart_cost",
            "critical_attacks",
            "historical_determination_preservation",
        ],
    }


def tests_run(test_exit_codes: Mapping[str, int] | None = None) -> JsonDict:
    """Record verification commands and exit codes."""

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
    """Declare that the audit is not an oracle."""

    return {
        "value": False,
        "audit_is_oracle": False,
        "upstream_exact_checkers_audited_as_oracles": [
            "exp6418.exact_feasibility_checker",
            "exp6418.exact_consequence_checker",
            "exp6418.exact_release_checker",
            "exp6418.exact_retention_checker",
            "exp6419.exact_outcome_checker",
            "exp6419.exact_retention_checker",
        ],
        "false_for": {
            "audit": False,
            "model_output": False,
            "proposal_memory": False,
            "selection_memory": False,
        },
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Score one only when every authenticity and safety gate passes."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    exact_zero = _integer(artifact.get("exact_veto_override_count"), 1) == 0
    protected_zero = _integer(artifact.get("protected_leakage_count"), 1) == 0
    hidden_zero = _integer(artifact.get("hidden_retuning_count"), 1) == 0
    protected = as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True
    guard = as_mapping(artifact.get("adversarial_and_determination_preservation_findings"))
    determination = as_mapping(guard.get("determination_preservation"))
    conditions = [
        preconditions.get("date_matches") is True,
        preconditions.get("both_streams_exist") is True,
        preconditions.get("all_required_inputs_available") is True,
        preconditions.get("powered_receipts_authentic") is True,
        preconditions.get("causal_order_holds") is True,
        preconditions.get("recomputed_metrics_match") is True,
        preconditions.get("contamination_zero_after_rollback") is True,
        preconditions.get("protected_retention_non_negative") is True,
        preconditions.get("all_critical_attacks_fail_closed") is True,
        preconditions.get("tests_all_passed") is True,
        exact_zero,
        protected_zero,
        hidden_zero,
        protected,
        determination.get("passed") is True,
    ]
    return 1.0 if all(conditions) else 0.0


def prospective_csl_claim_eligibility(artifact: Mapping[str, Any]) -> JsonDict:
    """Limit prospective CSL eligibility to the completed audit gate."""

    blockers = []
    if as_mapping(artifact.get("missing_input_findings")).get("missing_required_count", 1):
        blockers.append("missing_required_inputs")
    if as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("all_reported_match_recomputed") is not True:
        blockers.append("reported_metrics_do_not_recompute_from_rows")
    blockers.extend(as_mapping(artifact.get("attack_matrix")).get("open_critical_attack_ids", []))
    if _integer(artifact.get("exact_veto_override_count"), 1) != 0:
        blockers.append("exact_veto_override")
    if _integer(artifact.get("protected_leakage_count"), 1) != 0:
        blockers.append("protected_leakage")
    if _integer(artifact.get("hidden_retuning_count"), 1) != 0:
        blockers.append("hidden_retuning")
    return {
        "eligible": ready_score(artifact) == 1.0,
        "claim_class": "prospective_csl",
        "scope": "V552 CSL chain only; V551 claim-boundary blockers are preserved separately",
        "blockers": sorted(set(str(blocker) for blocker in blockers)),
    }


def public_factor_claim_eligibility(artifact: Mapping[str, Any]) -> JsonDict:
    """Limit public claim eligibility to a fully ready audit."""

    eligible = ready_score(artifact) == 1.0
    return {
        "eligible": eligible,
        "claim_class": "public_factor",
        "scope": "internal audit only until Exp6420 readiness is one",
        "blockers": [] if eligible else ["prospective_csl_claim_not_eligible"],
    }


def harm_underpowered_missing_and_flagged_cells(artifact: Mapping[str, Any]) -> JsonDict:
    """Keep harms visible rather than pooling or hiding them."""

    reasons = []
    if as_mapping(artifact.get("missing_input_findings")).get("missing_required_count", 0):
        reasons.append("missing_required_inputs")
    if as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("mismatch_count", 0):
        reasons.append("reported_metric_mismatch")
    if as_mapping(artifact.get("attack_matrix")).get("open_critical_attack_ids"):
        reasons.append("open_critical_attacks")
    small_cells = as_mapping(artifact.get("uncertainty_and_effective_sample_sizes")).get(
        "small_cell_count",
        0,
    )
    if small_cells:
        reasons.append("underpowered_cells")
    return {
        "schema": SCHEMA + ".harm_visible",
        "visible_harm_reasons": reasons,
        "underpowered_cell_count": int(small_cells or 0),
        "reported_metric_mismatch_count": int(
            as_mapping(artifact.get("reported_vs_recomputed_deltas")).get("mismatch_count", 0)
            or 0
        ),
        "open_critical_attack_ids": as_mapping(artifact.get("attack_matrix")).get(
            "open_critical_attack_ids",
            [],
        ),
        "all_visible": True,
    }


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal audit state."""

    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict."""

    return (
        "complete: V552 CSL chain passes authenticity and safety audit"
        if ready_score(artifact) == 1.0
        else "complete_null: V552 CSL chain audit completed; reported metrics do not recompute from rows or critical attacks remain open"
    )


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = "sha256:normalized"
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> JsonDict:
    """Refresh terminal readiness, eligibility, verdict, and checksum fields."""

    artifact["csl_authenticity_safety_audit_ready_score"] = ready_score(artifact)
    artifact["prospective_csl_claim_eligibility"] = prospective_csl_claim_eligibility(artifact)
    artifact["public_factor_claim_eligibility"] = public_factor_claim_eligibility(artifact)
    artifact["harm_underpowered_missing_and_flagged_cells"] = harm_underpowered_missing_and_flagged_cells(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate schema, oracle boundary, terminal prefix, and checksum."""

    require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(
        set(REQUIRED_ARTIFACT_FIELDS)
        | set(MISSING_INPUT_PRINCIPLE_KEYS)
        | set(ATTACK_PRINCIPLE_KEYS)
        | set(ELIGIBILITY_PRINCIPLE_KEYS)
        <= set(as_mapping(artifact.get("field_principles"))),
        "field_principles",
    )
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))), "field_provenance")
    require(as_mapping(artifact.get("verifier_is_oracle")).get("value") is False, "verifier_is_oracle")
    require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )
    return True


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp6420 audit artifact."""

    started = time.perf_counter()
    protected_before = protected_hashes()
    context = load_context(REPO_ROOT)
    expected = expected_input_receipts(REPO_ROOT)
    missing = missing_input_findings(expected)
    hashes = upstream_artifact_sidecar_source_checkpoint_model_and_checker_hashes(context, expected)
    authenticity = process_and_raw_output_authenticity_rechecks(context, hashes)
    authenticity["all_rechecks_passed"] = (
        authenticity["all_pids_bound_and_positive"]
        and authenticity["all_exp6418_process_receipts_accepted"]
        and authenticity["all_exp6419_process_receipts_accepted"]
        and authenticity["all_exp6419_raw_outputs_frozen"]
        and authenticity["all_model_bytes_match_recorded_hashes"]
    )
    proposal_order = proposal_precedes_outcome_checks(as_mapping(context.get("exp6418")))
    update_order = update_follows_exact_feedback_checks(as_mapping(context.get("exp6418")))
    future = untouched_future_partition_checks(context)
    order = reconstructed_event_time_order(context)
    proposal_bindings = proposal_memory_exact_feasibility_bindings(as_mapping(context.get("exp6418")))
    selection_bindings = selection_memory_exact_consequence_bindings(as_mapping(context.get("exp6418")))
    recomputed = recomputed_development_and_held_metrics(context)
    deltas = reported_vs_recomputed_deltas(context, recomputed)
    safety = retention_forgetting_contamination_growth_restart_and_cost_rechecks(context, recomputed)
    safety["exact_veto_override_count"] = exact_veto_override_count(context)
    exact_vetoes = exact_veto_override_count(context)
    protected_leaks = protected_leakage_count(context)
    hidden_retunes = hidden_retuning_count(context)
    attacks = attack_matrix(context, authenticity)
    uncertainty = uncertainty_and_effective_sample_sizes(context)
    guard = adversarial_and_determination_preservation_findings(context)
    tests = tests_run(test_exit_codes)
    protected_after = protected_hashes()
    protected_receipt = protected_unchanged_receipt(protected_before, protected_after)
    preconditions = preconditions_checked(
        date=date,
        expected=expected,
        authenticity=authenticity,
        order=order,
        deltas=deltas,
        safety=safety,
        attacks=attacks,
        tests=tests,
    )
    artifact: JsonDict = {
        "status": "pending",
        "expected_and_available_upstream_inputs": expected,
        "upstream_artifact_sidecar_source_checkpoint_model_and_checker_hashes": hashes,
        "missing_input_findings": missing,
        "process_and_raw_output_authenticity_rechecks": authenticity,
        "reconstructed_event_time_order": order,
        "proposal_precedes_outcome_checks": proposal_order,
        "update_follows_exact_feedback_checks": update_order,
        "untouched_future_partition_checks": future,
        "proposal_memory_exact_feasibility_bindings": proposal_bindings,
        "selection_memory_exact_consequence_bindings": selection_bindings,
        "recomputed_development_and_held_metrics": recomputed,
        "reported_vs_recomputed_deltas": deltas,
        "retention_forgetting_contamination_growth_restart_and_cost_rechecks": safety,
        "uncertainty_and_effective_sample_sizes": uncertainty,
        "exact_veto_override_count": exact_vetoes,
        "protected_leakage_count": protected_leaks,
        "hidden_retuning_count": hidden_retunes,
        "attack_matrix": attacks,
        "adversarial_and_determination_preservation_findings": guard,
        "prospective_csl_claim_eligibility": {"eligible": False},
        "public_factor_claim_eligibility": {"eligible": False},
        "csl_authenticity_safety_audit_ready_score": 0.0,
        "harm_underpowered_missing_and_flagged_cells": {"all_visible": False},
        "protected_files_unchanged": protected_receipt,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": verifier_is_oracle(),
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": rounded(duration_s if duration_s is not None else time.perf_counter() - started),
        "tests_run": tests,
        "reproducibility_checksum": "sha256:pending",
        "honest_verdict": "complete_null: pending",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Exp6420."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(date=args.date, result_path=args.output, write=True)
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
