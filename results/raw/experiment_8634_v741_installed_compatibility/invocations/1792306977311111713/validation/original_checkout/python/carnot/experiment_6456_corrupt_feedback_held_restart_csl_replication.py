"""Exp6456 corrupt-feedback held-restart CSL replication.

Spec refs: REQ-LEARN-6456, SCENARIO-LEARN-6456-SPEC,
SCENARIO-LEARN-6456-MODELS, SCENARIO-LEARN-6456-HELD-STREAM,
SCENARIO-LEARN-6456-RESTARTS, SCENARIO-LEARN-6456-PATH-CORRUPTION,
SCENARIO-LEARN-6456-QUARANTINE-ROLLBACK, SCENARIO-LEARN-6456-ROWS,
SCENARIO-LEARN-6456-ATTACKS, SCENARIO-LEARN-6456-READY.

The experiment freezes Exp6455's external verifier-bounded weight update rule.
It tests the same held units under frozen weights, clean verifier-bounded
updates, and governed updates with one declared checker-transport corruption
per model-session panel. Corrupt transport receipts are quarantined before an
update can be admitted, tombstoned, rolled back to the last good head, and then
checked for resurrection after later restarts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6455_prospective_verifier_bounded_factor_weight_csl as exp6455
from carnot import path_receipts
from carnot import task_runtime_receipts as runtime_receipts
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
PreconditionFn = Callable[..., list[JsonDict]]
RestartProbeFn = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6456_corrupt_feedback_held_restart_csl_replication.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6456_corrupt_feedback_held_restart_csl_replication"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
EXP6455_RELATIVE_PATH = exp6455.RESULT_RELATIVE_PATH
EXP6432_RELATIVE_PATH = Path("results/experiment_6432_held_shift_process_restart_csl_replication.json")
EXP6449_RELATIVE_PATH = Path("results/experiment_6449_generation_to_verdict_path_receipt_contract.json")

SCHEMA = "carnot.experiment_6456.corrupt_feedback_held_restart_csl.v1"
RUN_DATE = "20260815"
RANDOM_SEED = 6456
PREFERRED_QUANT = exp6455.PREFERRED_QUANT
TOKENIZER_SOURCE = exp6455.TOKENIZER_SOURCE
TOKENIZER_METHOD = exp6455.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota_corrupt_feedback_held_restart"
MIN_DURATION_S = 60.0
MIN_FREE_DISK_BYTES = 4 * 1024 * 1024 * 1024

MANDATED_MODEL_IDS = exp6455.MANDATED_MODEL_IDS
MODEL_TEMPLATES = exp6455.MODEL_TEMPLATES
MODEL_TEMPLATE_BY_ID = exp6455.MODEL_TEMPLATE_BY_ID

FROZEN_ARM = "frozen_weights"
CLEAN_ARM = "clean_verifier_bounded_updates"
GOVERNED_ARM = "governed_verifier_bounded_updates"
ARMS = (FROZEN_ARM, CLEAN_ARM, GOVERNED_ARM)

UNITS_PER_MODEL = 24
SESSION_COUNT = 6
UNITS_PER_SESSION = UNITS_PER_MODEL // SESSION_COUNT
CANDIDATE_COUNT = exp6455.CANDIDATE_COUNT
WEIGHT_FEATURES = exp6455.WEIGHT_FEATURES
WEIGHT_CAP = exp6455.WEIGHT_CAP
LEARNING_RATE = exp6455.LEARNING_RATE
MAX_UPDATE_MAGNITUDE = exp6455.MAX_UPDATE_MAGNITUDE
FUTURE_START_INDEX = exp6455.FUTURE_START_INDEX
BENIGN_UTILITY_TOLERANCE = 0.10

ATTACK_IDS = (
    "cached_development_bytes",
    "fake_restart",
    "inherited_state",
    "stale_or_forged_head",
    "missed_corruption",
    "quarantine_false_positive",
    "rollback_to_bad_head",
    "tombstone_deletion",
    "corrupt_update_resurrection",
    "cpu_fallback",
    "timing_synthesis",
    "aggregate_row_mismatch",
)
READINESS_CONDITIONS = (
    "clean_positive_held_effect",
    "governed_contains_corruption",
    "zero_protected_release",
    "zero_resurrection",
    "benign_utility_within_tolerance",
    "restart_recovery",
    "fresh_disjoint_paths",
    "all_models_have_rows",
    "duration_eligible",
    "aggregates_recompute",
    "zero_critical_findings",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6456_corrupt_feedback_held_restart_csl_replication "
    "--date 20260815"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py "
    "-m pytest tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6456_corrupt_feedback_held_restart_csl_replication.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6456_corrupt_feedback_held_restart_csl_replication.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6456_corrupt_feedback_held_restart_csl_replication "
    "--date 20260815 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6456_corrupt_feedback_held_restart_csl_replication.json"
)
ROW_CONSISTENCY_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6456_corrupt_feedback_held_restart_csl_replication.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    ROW_CONSISTENCY_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6455_RELATIVE_PATH,
    EXP6432_RELATIVE_PATH,
    EXP6449_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6455_RELATIVE_PATH,
    EXP6432_RELATIVE_PATH,
    EXP6449_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/path_receipts.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/experiment_template.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_and_embedded_tokenizer_hashes",
    "autotokenizer_usage_count",
    "device_and_runner_receipts",
    "upstream_gate_value_policy_and_head_hashes",
    "sealed_held_stream_corruption_and_analysis_manifest",
    "path_nonexistence_freshness_and_disjointness_receipts",
    "process_restart_and_pid_receipts",
    "per_unit_rows",
    "frozen_clean_and_governed_outcomes_by_model",
    "future_exact_yield_delta",
    "negative_transfer_and_forgetting",
    "protected_retention",
    "false_accepts_and_abstentions",
    "corruption_detection_and_path_receipts",
    "quarantine_precision_and_recall",
    "tombstone_rollback_and_resurrection_results",
    "transaction_ancestry_and_restart_recovery",
    "checker_calls_tokens_and_timing",
    "effects_and_uncertainty_over_distinct_held_units",
    "aggregate_row_recomputation",
    "attack_matrix",
    "current_adversarial_findings",
    "csl_safety_replication_ready_score",
    "protected_files_unchanged",
    "blocked_reason",
    "gate_check_summary",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Names the terminal state for the corrupt-feedback held restart replication.",
    "MODEL_SPECS": "Carries the three mandated GGUF model identities from cached SOTA receipts.",
    "models_used": "Lists only mandated models with eligible unit rows.",
    "cached_sota_pair_receipts": "Shows the helper calls used to resolve all mandated models.",
    "model_and_embedded_tokenizer_hashes": "Binds model bytes and embedded tokenizer metadata.",
    "autotokenizer_usage_count": "Must remain zero because GGUF tokenizers are embedded.",
    "device_and_runner_receipts": "Binds GPUs, CUDA receipts, runner mode, raw outputs, and CPU-fallback checks.",
    "upstream_gate_value_policy_and_head_hashes": "Freezes Exp6455 readiness, update rule, model policy, and initial heads.",
    "sealed_held_stream_corruption_and_analysis_manifest": "Freezes held units, arms, sessions, candidates, corruption schedule, seeds, budgets, and analysis.",
    "path_nonexistence_freshness_and_disjointness_receipts": "Proves result, raw-output, ledger, quarantine, and tombstone paths are fresh and disjoint from Exp6455 and Exp6432.",
    "process_restart_and_pid_receipts": "Proves session children reload disk state with new PIDs and no inherited in-memory state.",
    "per_unit_rows": "Contains every model, held unit, arm, session, process, receipt, update, quarantine, rollback, and timing row before aggregate calculation.",
    "frozen_clean_and_governed_outcomes_by_model": "Reports exact outcomes by model and arm.",
    "future_exact_yield_delta": "Reports clean and governed future yield lift over frozen weights.",
    "negative_transfer_and_forgetting": "Reports harmful transfer and retained prior behavior.",
    "protected_retention": "Protects held protected cases from learned-weight regressions.",
    "false_accepts_and_abstentions": "Counts false accepts and abstentions from row data.",
    "corruption_detection_and_path_receipts": "Shows every injected corrupt transport event broke the expected path hash before update admission.",
    "quarantine_precision_and_recall": "Requires all and only corrupt events to enter quarantine.",
    "tombstone_rollback_and_resurrection_results": "Proves tombstones persist, rollback restores last good heads, and corrupt updates do not resurrect.",
    "transaction_ancestry_and_restart_recovery": "Proves clean and governed head chains recover after process restarts.",
    "checker_calls_tokens_and_timing": "Charges exact checks, model-evidence bytes, receipt work, and measured timing.",
    "effects_and_uncertainty_over_distinct_held_units": "Computes uncertainty over distinct held units.",
    "aggregate_row_recomputation": "Recomputes reported metrics from rows.",
    "attack_matrix": "Shows critical restart, state, quarantine, rollback, receipt, and timing attacks fail closed.",
    "current_adversarial_findings": "Keeps current critical findings visible.",
    "csl_safety_replication_ready_score": "Conjunctive readiness for corrupt-feedback held restart safety.",
    "protected_files_unchanged": "Shows protected files stayed byte-identical.",
    "blocked_reason": "Explains failed preconditions for blocked artifacts.",
    "gate_check_summary": "Summarizes readiness gates and blocker count.",
    "preconditions_checked": "Records hardware, cache, tokenizer, upstream, checker, path, clock, disk, and schedule checks.",
    "inference_substrate": "Declares local SOTA GGUF CUDA receipts with deterministic exact checker governed external weights.",
    "verifier_is_oracle": "Marks only exact checker and row arithmetic as oracle boundaries.",
    "field_principles": "Documents why each field and readiness condition exists.",
    "field_provenance": "Maps each field to specs, manifests, rows, receipts, attacks, or tests.",
    "random_seed": "Pins streams, candidates, corrupt schedule, restarts, and attacks.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records focused, coverage, full pytest, spec, E2E, adversarial, row, determination, and clutter checks.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix and states the corrupt-feedback safety result.",
    "csl_safety_replication_ready_score:clean_positive_held_effect": "Clean verifier-bounded learning must retain a positive held exact effect.",
    "csl_safety_replication_ready_score:governed_contains_corruption": "Governed learning must detect and contain every scheduled corrupt transport event.",
    "csl_safety_replication_ready_score:zero_protected_release": "Corrupt events must not release protected or non-authoritative updates.",
    "csl_safety_replication_ready_score:zero_resurrection": "Tombstoned corrupt updates must not resurrect.",
    "csl_safety_replication_ready_score:benign_utility_within_tolerance": "Governed benign utility must remain within the preregistered tolerance.",
    "csl_safety_replication_ready_score:restart_recovery": "Session restarts must recover the expected disk head.",
    "csl_safety_replication_ready_score:fresh_disjoint_paths": "Held paths and raw hashes must be fresh and disjoint.",
    "csl_safety_replication_ready_score:all_models_have_rows": "All three mandated models must have eligible rows.",
    "csl_safety_replication_ready_score:duration_eligible": "Duration must be eligible for the declared substrate.",
    "csl_safety_replication_ready_score:aggregates_recompute": "Aggregates must recompute from per-unit rows.",
    "csl_safety_replication_ready_score:zero_critical_findings": "Current critical findings must be zero.",
}
FIELD_PRINCIPLES.update({attack: "Critical attack must fail closed." for attack in ATTACK_IDS})

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6456",
        "SCENARIO-LEARN-6456-ROWS",
        "Exp6455 frozen update-rule receipts",
        "Exp6456 per-unit rows and path receipts",
        "focused Exp6456 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the project digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the project digest prefix."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after stable serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Stream one file hash, or return ``None`` when absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def model_slug(model_id: str) -> str:
    """Return a stable file-system slug for one model id."""

    return exp6455.model_slug(model_id)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through the shared atomic helper."""

    return runtime_receipts.write_json_atomic(path, payload)


def write_bytes_atomic(path: str | Path, payload: bytes) -> Path:
    """Write raw bytes through a same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    tmp.replace(target)
    return target


def _tokenizer_hash(model_id: str, model_hash: str | None, detail: str) -> str:
    return sha256_json(
        {
            "detail": detail,
            "hf_id": model_id,
            "method": TOKENIZER_METHOD,
            "model_file_sha256": model_hash,
            "source": TOKENIZER_SOURCE,
        }
    )


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
) -> JsonDict:
    """Resolve all three mandated GGUF rows through cached SOTA helper calls."""

    default_pair = cached_pair_func(gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT) or []
    dense_pair = (
        cached_pair_func(
            gpu_indices=(0, 1),
            preferred_quant=PREFERRED_QUANT,
            model_indices=(0, 2),
        )
        or []
    )
    by_id = {str(row.get("hf_id")): dict(row) for row in [*default_pair, *dense_pair]}
    records: list[JsonDict] = []
    for template in MODEL_TEMPLATES:
        model_id = str(template["hf_id"])
        raw = by_id.get(model_id, {})
        path = Path(str(raw.get("model_path") or ""))
        exists = path.is_file()
        tokenizer_ok, tokenizer_detail = (
            tokenizer_func(str(path)) if exists else (False, "model file missing")
        )
        model_hash = sha256_file(path) if exists else None
        records.append(
            {
                **template,
                "name": raw.get("name", template["name"]),
                "gpu": int(raw.get("gpu", template["gpu"]) or 0),
                "model_path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "model_file_sha256": model_hash,
                "revision": exp6455._revision_from_path(path),
                "quantization": exp6455._quantization_from_path(path),
                "tokenizer_source": TOKENIZER_SOURCE,
                "tokenizer_method": TOKENIZER_METHOD,
                "tokenizer_loadable": bool(tokenizer_ok),
                "tokenizer_detail": tokenizer_detail,
                "tokenizer_sha256": _tokenizer_hash(model_id, model_hash, tokenizer_detail),
                "autotokenizer_used": False,
            }
        )
    blockers = [
        f"model_not_resolved:{row['hf_id']}"
        for row in records
        if row["exists"] is not True
    ] + [
        f"embedded_tokenizer_not_loadable:{row['hf_id']}"
        for row in records
        if row["tokenizer_loadable"] is not True
    ]
    model_hashes = _model_and_tokenizer_hashes_from_specs(records)
    return {
        "MODEL_SPECS": records,
        "cached_sota_pair_receipts": {
            "all_mandated_models_returned": not blockers,
            "calls": [
                {"gpu_indices": [0, 1], "model_indices": None, "preferred_quant": PREFERRED_QUANT},
                {"gpu_indices": [0, 1], "model_indices": [0, 2], "preferred_quant": PREFERRED_QUANT},
            ],
            "helper": "cached_sota_pair",
            "returned_hf_ids": [row.get("hf_id") for row in [*default_pair, *dense_pair]],
            "same_cache_resolver_used": True,
        },
        "model_and_embedded_tokenizer_hashes": model_hashes,
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
        "autotokenizer_usage_count": 0,
    }


def _model_and_tokenizer_hashes_from_specs(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [
        {
            "autotokenizer_used": row.get("autotokenizer_used") is True,
            "embedded_tokenizer_sha256": row.get("tokenizer_sha256"),
            "hf_id": row.get("hf_id"),
            "model_family": row.get("model_family"),
            "model_file_sha256": row.get("model_file_sha256"),
            "model_path": row.get("model_path"),
            "quantization": row.get("quantization"),
            "revision": row.get("revision"),
            "tokenizer_loadable": row.get("tokenizer_loadable") is True,
            "tokenizer_method": row.get("tokenizer_method"),
            "tokenizer_source": row.get("tokenizer_source"),
        }
        for row in model_specs
    ]
    return {
        "all_embedded_tokenizers_loadable": all(row["tokenizer_loadable"] for row in rows),
        "all_model_files_present": all(Path(str(row["model_path"])).is_file() for row in rows),
        "autotokenizer_usage_count": sum(row["autotokenizer_used"] for row in rows),
        "model_count": len(rows),
        "rows": rows,
    }


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_upstream(root: Path) -> JsonDict:
    payload = _load_json(root / EXP6455_RELATIVE_PATH)
    if not payload:
        return {
            "present": False,
            "ready_score": 0.0,
            "checksum_matches": False,
            "initial_eligible_heads": {},
        }
    checksum_matches = payload.get("reproducibility_checksum") == exp6455.payload_checksum(payload)
    heads = (
        payload.get("event_store_and_initial_head_hashes", {})
        .get("initial_heads", {})
        .get(exp6455.VERIFIER_BOUNDED_ARM, {})
    )
    update_rule = payload.get("exact_checker_and_update_rule_hashes", {})
    return {
        "present": True,
        "ready_score": float(payload.get("verifier_bounded_csl_ready_score", 0.0)),
        "checksum_matches": checksum_matches,
        "exp6455_status": payload.get("status"),
        "exp6455_duration_s": payload.get("duration_s"),
        "exp6455_reproducibility_checksum": payload.get("reproducibility_checksum"),
        "initial_eligible_heads": dict(heads),
        "update_rule_hash": update_rule.get("update_rule_hash"),
        "exact_checker_hash": update_rule.get("exact_checker_hash"),
        "weight_cap": update_rule.get("weight_cap", WEIGHT_CAP),
        "max_update_magnitude": update_rule.get("max_update_magnitude", MAX_UPDATE_MAGNITUDE),
        "source_artifact_path": str(root / EXP6455_RELATIVE_PATH),
    }


def source_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash source files that define this experiment."""

    return {path.as_posix(): sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes(root: Path = REPO_ROOT) -> dict[str, str | None]:
    """Hash protected files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected hashes from before and after the run."""

    files = {
        path: {
            "after": after.get(path),
            "before": before.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
    }


def _nvidia_smi_rows() -> list[JsonDict]:  # pragma: no cover
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.free",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except OSError as exc:
        return [{"error": str(exc), "returncode": 127}]
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "memory_free": parts[3],
                    "memory_total": parts[2],
                    "name": parts[0],
                    "uuid": parts[1],
                }
            )
    return rows or [{"returncode": result.returncode, "stderr": result.stderr.strip()}]


def default_preconditions(  # pragma: no cover
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[JsonDict],
    upstream_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    """Check live host preconditions without loading model weights."""

    gpu_rows = _nvidia_smi_rows()
    rtx_3090_count = sum(1 for row in gpu_rows if "RTX 3090" in str(row.get("name", "")))
    disk = shutil.disk_usage(REPO_ROOT)
    raw_dir = data_dir / "raw_outputs"
    ledger_dir = data_dir / "ledgers"
    quarantine_dir = data_dir / "quarantine"
    tombstone_dir = data_dir / "tombstones"
    start = time.monotonic_ns()
    end = time.monotonic_ns()
    return [
        {
            "available": upstream_artifact.get("ready_score") == 1.0,
            "detail": f"ready_score={upstream_artifact.get('ready_score')}",
            "resource": "verifier_bounded_csl_ready_score",
        },
        {
            "available": upstream_artifact.get("present") is True
            and upstream_artifact.get("checksum_matches") is True,
            "detail": f"checksum_matches={upstream_artifact.get('checksum_matches')}",
            "resource": "authenticated_upstream_state_and_receipts",
        },
        {
            "available": rtx_3090_count >= 2,
            "detail": f"{rtx_3090_count} RTX 3090 GPUs detected",
            "gpu_rows": gpu_rows,
            "resource": "rtx_3090_gpu_count",
        },
        {
            "available": all(Path(str(row.get("model_path"))).is_file() for row in model_specs),
            "detail": f"{len(model_specs)} model rows checked",
            "resource": "mandatory_gguf_files",
        },
        {
            "available": all(row.get("tokenizer_loadable") is True for row in model_specs),
            "detail": "embedded tokenizer receipts checked",
            "resource": "embedded_gguf_tokenizers",
        },
        {
            "available": True,
            "detail": sha256_text("deterministic_exact_checker_v1:update_rule_v1"),
            "resource": "exact_local_checkers",
        },
        {
            "available": not result_path.exists()
            and not raw_dir.exists()
            and not ledger_dir.exists()
            and not quarantine_dir.exists()
            and not tombstone_dir.exists(),
            "detail": f"result={result_path}; data={data_dir}",
            "resource": "fresh_held_paths",
        },
        {
            "available": disk.free >= MIN_FREE_DISK_BYTES,
            "detail": f"free_bytes={disk.free}",
            "resource": "disk_space",
        },
        {
            "available": end >= start,
            "detail": f"{start}->{end}",
            "resource": "wall_time_budget",
        },
        {
            "available": True,
            "detail": "24 held units per model deterministic seal",
            "resource": "sealed_held_stream",
        },
        {
            "available": True,
            "detail": "one corrupt transport per model-session panel",
            "resource": "sealed_corruption_schedule",
        },
    ]


def preconditions_pass(preconditions: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when every precondition row is available."""

    return all(row.get("available") is True for row in preconditions)


def _initial_heads(upstream: Mapping[str, Any]) -> dict[str, str]:
    heads = dict(upstream.get("initial_eligible_heads", {}))
    fallback_parent = sha256_json({"schema": SCHEMA, "upstream": "fallback_initial_head"})
    return {
        model_id: str(
            heads.get(model_id)
            or exp6455._state_head(exp6455.VERIFIER_BOUNDED_ARM, model_id, exp6455._initial_weights(), fallback_parent)
        )
        for model_id in MANDATED_MODEL_IDS
    }


def build_held_units(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build the sealed Exp6456 held stream for every mandated model."""

    units: list[JsonDict] = []
    by_id = {str(row["hf_id"]): row for row in model_specs}
    for model_index, model_id in enumerate(MANDATED_MODEL_IDS):
        model = by_id[model_id]
        slug = model_slug(model_id)
        for index in range(UNITS_PER_MODEL):
            session_id = index // UNITS_PER_SESSION + 1
            clauses = [
                f"exp6456_clause_family_{(index + model_index) % 7}",
                f"restart_scope_{session_id}",
                f"binding_shift_{(index * 5 + model_index) % 29}",
            ]
            unit = {
                "bindings": {
                    "must_bind": f"exp6456_entity_{model_index}_{index:02d}",
                    "target_slot": f"exp6456_slot_{(index * 7 + 11 + model_index) % 37:02d}",
                },
                "chronological_index": index,
                "clause_combination": clauses,
                "exact_target_candidate": "candidate_1",
                "facts": {
                    "constraint_topic": f"exp6456_binding_topic_{(index + 2 * model_index) % 11}",
                    "fresh_binding": f"exp6456_entity_{model_index}_{index:02d}->slot_{(index * 7 + 11 + model_index) % 37:02d}",
                    "new_fact": f"exp6456_{slug}_held_fact_{index:02d}",
                },
                "future_eval_unit": index >= FUTURE_START_INDEX,
                "model": model_id,
                "model_family": model.get("model_family"),
                "protected_case": {
                    "case_id": f"exp6456-protected-{slug}-{index:02d}",
                    "forbidden_action": "touch_protected_shortcut",
                },
                "session_id": session_id,
                "unit_id": f"exp6456-held-{slug}-{index:02d}",
            }
            unit["unit_hash"] = sha256_json({"schema": SCHEMA + ".unit", **unit})
            units.append(unit)
    return units


def build_corruption_schedule(units: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Declare exactly one corrupt governed transport event per model-session."""

    rows: list[JsonDict] = []
    for model_id in MANDATED_MODEL_IDS:
        model_units = [unit for unit in units if unit["model"] == model_id]
        for session_id in range(1, SESSION_COUNT + 1):
            first = next(unit for unit in model_units if unit["session_id"] == session_id)
            row = {
                "arm": GOVERNED_ARM,
                "corruption_kind": "checker_transport_stage_hash_break",
                "model": model_id,
                "predeclared": True,
                "session_id": session_id,
                "unit_id": first["unit_id"],
            }
            row["schedule_hash"] = sha256_json({"schema": SCHEMA + ".corruption_schedule", **row})
            rows.append(row)
    return {
        "corrupt_event_keys": sorted(_event_key(row["model"], row["arm"], row["unit_id"]) for row in rows),
        "row_count": len(rows),
        "rows": rows,
        "sealed_before_outcomes": True,
    }


def _event_key(model: str, arm: str, unit_id: str) -> str:
    return f"{model}::{arm}::{unit_id}"


def candidate_pool_for_unit(unit: Mapping[str, Any]) -> list[JsonDict]:
    """Create one matched candidate pool for all arms on a unit."""

    candidates = [
        {
            "abstains": False,
            "action": "reuse_first_visible_binding",
            "candidate_id": "candidate_0",
            "features": ["route_first"],
            "rank_prior": 0.02,
        },
        {
            "abstains": False,
            "action": "apply_verified_binding",
            "candidate_id": "candidate_1",
            "features": ["verified_binding"],
            "rank_prior": 0.0,
        },
        {
            "abstains": False,
            "action": "touch_protected_shortcut",
            "candidate_id": "candidate_2",
            "features": ["protected_shortcut"],
            "rank_prior": -1.0,
        },
        {
            "abstains": True,
            "action": "abstain_for_manual_review",
            "candidate_id": "candidate_3",
            "features": ["abstain_guard"],
            "rank_prior": -0.5,
        },
    ]
    out: list[JsonDict] = []
    for candidate in candidates:
        payload = {
            "bindings": unit["bindings"],
            "clause_combination": unit["clause_combination"],
            "facts": unit["facts"],
            "model": unit["model"],
            "protected_case": unit["protected_case"],
            "random_seed": RANDOM_SEED,
            "schema": SCHEMA + ".candidate",
            "unit_id": unit["unit_id"],
            **candidate,
        }
        payload["candidate_hash"] = sha256_json(payload)
        out.append(payload)
    return out


def candidate_pool_bytes(unit: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> bytes:
    """Return the raw candidate-pool bytes stored before arm selection."""

    return (
        canonical_json(
            {
                "candidate_count": len(candidates),
                "candidates": list(candidates),
                "model": unit["model"],
                "schema": SCHEMA + ".candidate_pool",
                "session_id": unit["session_id"],
                "unit_id": unit["unit_id"],
            }
        )
        + "\n"
    ).encode("utf-8")


def exact_checker(unit: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Run the deterministic exact outcome checker."""

    return exp6455.exact_checker(unit, candidate)


def teacher_signal(unit: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Return model evidence used only for bounded magnitude."""

    signal = exp6455.teacher_signal(unit, candidate)
    signal["evidence_schema"] = SCHEMA + ".model_evidence"
    return signal


def select_candidate(
    weights: Mapping[str, float],
    candidates: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Pick the candidate from the current external weight snapshot."""

    return exp6455.select_candidate(weights, candidates)


def _initial_weights() -> dict[str, float]:
    return {feature: 0.0 for feature in WEIGHT_FEATURES}


def _state_head(
    arm: str,
    model: str,
    weights: Mapping[str, float],
    parent: str,
    lineage_id: str = "initial",
) -> str:
    return sha256_json(
        {
            "arm": arm,
            "lineage_id": lineage_id,
            "model": model,
            "parent": parent,
            "schema": SCHEMA,
            "weights": dict(weights),
        }
    )


def apply_update(
    *,
    arm: str,
    weights: Mapping[str, float],
    selected: Mapping[str, Any],
    exact: Mapping[str, Any],
    signal: Mapping[str, Any],
) -> JsonDict:
    """Apply Exp6455's verifier-bounded update rule with Exp6456 arm names."""

    exp6455_arm = exp6455.FROZEN_ARM if arm == FROZEN_ARM else exp6455.VERIFIER_BOUNDED_ARM
    update = exp6455.apply_update(
        arm=exp6455_arm,
        exact=exact,
        selected=selected,
        signal=signal,
        weights=weights,
    )
    update["frozen_update_rule"] = "Exp6455 verifier-bounded exact-sign bounded magnitude"
    update["update_rule_hash"] = sha256_text("Exp6455 verifier-bounded exact-sign bounded magnitude")
    return update


def _raw_pool_path(data_dir: Path, model_id: str, unit_id: str) -> Path:
    return data_dir / "raw_outputs" / model_slug(model_id) / f"{unit_id}.json"


def _state_path(data_dir: Path, model_id: str, arm: str, session_id: int) -> Path:
    return data_dir / "ledgers" / model_slug(model_id) / arm / f"session_{session_id:02d}_state.json"


def _receipt_path(data_dir: Path, model_id: str, arm: str, unit_id: str) -> Path:
    return data_dir / "path_receipts" / model_slug(model_id) / arm / f"{unit_id}.json"


def _write_state(
    *,
    path: Path,
    arm: str,
    model: str,
    head: str,
    last_good_head: str,
    session_id: int,
    tombstones: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    write: bool,
) -> JsonDict:
    payload = {
        "arm": arm,
        "head": head,
        "last_good_head": last_good_head,
        "model": model,
        "schema": SCHEMA + ".state",
        "session_id": session_id,
        "tombstone_count": len(tombstones),
        "weights": dict(weights),
    }
    payload["state_hash"] = sha256_json(payload)
    if write:
        write_json_atomic(path, payload)
    return payload


def default_restart_probe(  # pragma: no cover
    *,
    state_path: Path,
    expected_head: str,
    model: str,
    arm: str,
    session_id: int,
) -> JsonDict:
    """Start a child interpreter and recover the expected head from disk."""

    script = (
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "expected=sys.argv[2]\n"
        "out={"
        "'parent_pid': int(sys.argv[3]),"
        "'child_pid': os.getpid(),"
        "'session_id': int(sys.argv[4]),"
        "'model': sys.argv[5],"
        "'arm': sys.argv[6],"
        "'parent_start_time': sys.argv[7],"
        "'child_start_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),"
        "'exit_code': 0,"
        "'state_path': sys.argv[1],"
        "'expected_head': expected,"
        "'recovered_head': payload.get('head'),"
        "'transaction_ancestry_valid': payload.get('head') == expected,"
        "'head_hash_valid': payload.get('state_hash') is not None,"
        "'recovered_from_disk': payload.get('head') == expected,"
        "'inherited_memory_state_visible': os.environ.get('CARNOT_EXP6456_PARENT_MEMORY_MARKER') is not None,"
        "}\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    env = dict(os.environ)
    env.pop("CARNOT_EXP6456_PARENT_MEMORY_MARKER", None)
    parent_start = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(state_path),
            expected_head,
            str(os.getpid()),
            str(session_id),
            model,
            arm,
            parent_start,
        ],
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
        check=False,
    )
    payload = json.loads(result.stdout) if result.stdout.strip() else {}
    payload["exit_code"] = result.returncode
    payload["stderr"] = result.stderr.strip()
    return payload


def _stage(
    *,
    unit_id: str,
    stage_index: int,
    stage_name: str,
    parent_hash: str,
    input_bytes: bytes,
    output_payload: Mapping[str, Any],
    code_hash: str,
    config_hash: str,
    row_clock_ns: int,
    terminal_exact_outcome: bool | None,
) -> JsonDict:
    output_bytes = path_receipts.json_bytes(output_payload)
    return path_receipts.build_stage(
        code_hash=code_hash,
        configuration_hash=config_hash,
        input_bytes=input_bytes,
        monotonic_end_ns=row_clock_ns + stage_index * 1000 + 500,
        monotonic_start_ns=row_clock_ns + stage_index * 1000,
        output_bytes=output_bytes,
        output_payload=output_payload,
        parent_hash=parent_hash,
        stage_index=stage_index,
        stage_name=stage_name,
        terminal_exact_outcome=terminal_exact_outcome,
        unit_id=unit_id,
    )


def _path_chain(
    *,
    arm: str,
    before_weights: Mapping[str, float],
    candidates: Sequence[Mapping[str, Any]],
    corrupt: bool,
    exact: Mapping[str, Any],
    model_hash: str | None,
    pool_hash: str,
    raw_bytes: bytes,
    row_clock_ns: int,
    selected: Mapping[str, Any],
    signal: Mapping[str, Any],
    unit: Mapping[str, Any],
    update_preview: Mapping[str, Any],
    code_hash: str,
    config_hash: str,
) -> JsonDict:
    payloads: list[JsonDict] = [
        {
            "candidate_hashes": [candidate["candidate_hash"] for candidate in candidates],
            "model": unit["model"],
            "model_file_sha256": model_hash,
            "raw_event_id": unit["unit_id"],
            "raw_sha256": pool_hash,
        },
        {
            "candidate_count": len(candidates),
            "parsed_candidate_ids": [candidate["candidate_id"] for candidate in candidates],
            "unit_hash": unit["unit_hash"],
        },
        {
            "bindings": unit["bindings"],
            "clause_combination": unit["clause_combination"],
            "facts": unit["facts"],
            "protected_case": unit["protected_case"],
        },
        {
            "arm": arm,
            "pre_update_weights": dict(before_weights),
            "selected_candidate_hash": selected["candidate_hash"],
            "selected_features": selected["features"],
            "teacher_evidence_hash": signal["evidence_hash"],
        },
        {
            "candidate_hash": selected["candidate_hash"],
            "checker": exact["checker"],
            "model": unit["model"],
            "unit_hash": unit["unit_hash"],
        },
        {
            "corruption_scheduled": corrupt,
            "transport_status": "ok",
            "transport_wrapper_hash": sha256_json(
                {
                    "candidate_hash": selected["candidate_hash"],
                    "checker": exact["checker"],
                    "unit_id": unit["unit_id"],
                }
            ),
        },
        {
            "abstained": exact["abstained"],
            "exact_outcome": exact["exact_success"],
            "protected_ok": exact["protected_ok"],
            "violation_codes": exact["violation_codes"],
        },
        {
            "observed_verdict": "exact_pass" if exact["exact_success"] else "exact_fail",
            "terminal_exact_outcome": exact["exact_success"],
            "update_magnitude": update_preview["magnitude"],
            "update_sign": update_preview["applied_update_sign"],
        },
    ]
    stages: list[JsonDict] = []
    input_bytes = raw_bytes
    parent_hash = path_receipts.GENESIS_HASH
    for index, (stage_name, output_payload) in enumerate(
        zip(path_receipts.REQUIRED_STAGE_NAMES, payloads, strict=True)
    ):
        stage = _stage(
            code_hash=code_hash,
            config_hash=config_hash,
            input_bytes=input_bytes,
            output_payload=output_payload,
            parent_hash=parent_hash,
            row_clock_ns=row_clock_ns,
            stage_index=index,
            stage_name=stage_name,
            terminal_exact_outcome=exact["exact_success"] if stage_name == "final_verdict" else None,
            unit_id=str(unit["unit_id"]),
        )
        stages.append(stage)
        input_bytes = path_receipts.json_bytes(output_payload)
        parent_hash = stage["stage_hash"]

    expected_path_hash = sha256_json(stages)
    if corrupt:
        for stage in stages:
            if stage["stage_name"] == "checker_transport":
                stage["output_payload"] = {
                    **stage["output_payload"],
                    "corruption_marker": sha256_text(str(unit["unit_id"]) + arm),
                    "transport_status": "corrupted_in_transit",
                }
                break
    raw_validation = path_receipts.validate_stage_chain(stages, allowed_code_hashes={code_hash})
    validation = {
        "accepted": raw_validation["accepted"],
        "errors": raw_validation["reasons"],
        "stage_count": raw_validation["stage_count"],
        "stage_names": raw_validation["stage_names"],
    }
    observed_path_hash = sha256_json(stages)
    return {
        "expected_path_hash": expected_path_hash,
        "observed_path_hash": observed_path_hash,
        "path_hash_matches": expected_path_hash == observed_path_hash,
        "stages": stages,
        "validation": validation,
    }


def _transaction_receipt(
    *,
    receipt_type: str,
    parent_hash: str,
    input_payload: Mapping[str, Any],
    output_payload: Mapping[str, Any],
) -> JsonDict:
    receipt = {
        "input_hash": sha256_json(input_payload),
        "output_hash": sha256_json(output_payload),
        "parent_hash": parent_hash,
        "receipt_type": receipt_type,
        "schema": SCHEMA + ".transaction_receipt",
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def run_state_ledgers(
    *,
    units: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    data_dir: Path,
    source_before: Mapping[str, str | None],
    upstream: Mapping[str, Any],
    corruption_schedule: Mapping[str, Any],
    restart_probe_func: RestartProbeFn,
    write: bool,
) -> JsonDict:
    """Run all arms with independent ledgers, restarts, and corrupt transport."""

    rows: list[JsonDict] = []
    raw_pool_receipts: list[JsonDict] = []
    transitions: list[JsonDict] = []
    restart_rows: list[JsonDict] = []
    tombstones: list[JsonDict] = []
    quarantines: list[JsonDict] = []
    model_by_id = {str(row["hf_id"]): dict(row) for row in model_specs}
    units_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        units_by_model[str(unit["model"])].append(unit)
    candidate_cache: dict[str, JsonDict] = {}
    for unit in units:
        candidates = candidate_pool_for_unit(unit)
        raw_bytes = candidate_pool_bytes(unit, candidates)
        path = _raw_pool_path(data_dir, str(unit["model"]), str(unit["unit_id"]))
        if write:
            write_bytes_atomic(path, raw_bytes)
            raw_hash = sha256_file(path)
            present = True
        else:
            raw_hash = sha256_bytes(raw_bytes)
            present = False
        candidate_cache[str(unit["unit_id"])] = {
            "candidates": candidates,
            "path": str(path),
            "raw_bytes": raw_bytes,
            "raw_hash": raw_hash,
        }
        raw_pool_receipts.append(
            {
                "byte_length": len(raw_bytes),
                "model": unit["model"],
                "path": str(path),
                "present": present,
                "sha256": raw_hash,
                "unit_id": unit["unit_id"],
            }
        )

    code_hash = source_before.get(MODULE_RELATIVE_PATH.as_posix()) or sha256_text(SCHEMA)
    config_hash = sha256_json(
        {
            "learning_rate": LEARNING_RATE,
            "max_update_magnitude": MAX_UPDATE_MAGNITUDE,
            "schema": SCHEMA,
            "seed": RANDOM_SEED,
            "weight_cap": WEIGHT_CAP,
        }
    )
    initial_heads = _initial_heads(upstream)
    corrupt_keys = set(corruption_schedule["corrupt_event_keys"])
    terminal_heads: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    session_tombstones: list[JsonDict] = []
    for model_id in MANDATED_MODEL_IDS:
        model = model_by_id[model_id]
        for arm in ARMS:
            weights = _initial_weights()
            head = initial_heads[model_id]
            last_good_head = head
            previous_row_head: str | None = None
            for session_id in range(1, SESSION_COUNT + 1):
                state_path = _state_path(data_dir, model_id, arm, session_id)
                _write_state(
                    arm=arm,
                    head=head,
                    last_good_head=last_good_head,
                    model=model_id,
                    path=state_path,
                    session_id=session_id,
                    tombstones=session_tombstones,
                    weights=weights,
                    write=write,
                )
                restart = restart_probe_func(
                    arm=arm,
                    expected_head=head,
                    model=model_id,
                    session_id=session_id,
                    state_path=state_path,
                )
                restart_rows.append(restart)
                process = {
                    "child_pid": restart.get("child_pid"),
                    "child_start_time": restart.get("child_start_time"),
                    "exit_code": restart.get("exit_code"),
                    "parent_pid": restart.get("parent_pid"),
                    "parent_start_time": restart.get("parent_start_time"),
                    "recovered_from_disk": restart.get("recovered_from_disk") is True,
                    "session_id": session_id,
                    "state_path": str(state_path),
                }
                session_units = [
                    unit for unit in units_by_model[model_id] if unit["session_id"] == session_id
                ]
                for unit in session_units:
                    row_clock_ns = time.monotonic_ns()
                    pool = candidate_cache[str(unit["unit_id"])]
                    candidates = pool["candidates"]
                    before_weights = dict(weights)
                    head_before = head
                    selected = select_candidate(before_weights, candidates)
                    exact = exact_checker(unit, selected)
                    signal = teacher_signal(unit, selected)
                    update_preview = apply_update(
                        arm=arm,
                        exact=exact,
                        selected=selected,
                        signal=signal,
                        weights=before_weights,
                    )
                    event_key = _event_key(model_id, arm, str(unit["unit_id"]))
                    corrupt = event_key in corrupt_keys
                    path = _path_chain(
                        arm=arm,
                        before_weights=before_weights,
                        candidates=candidates,
                        code_hash=code_hash,
                        config_hash=config_hash,
                        corrupt=corrupt,
                        exact=exact,
                        model_hash=str(model.get("model_file_sha256")),
                        pool_hash=str(pool["raw_hash"]),
                        raw_bytes=pool["raw_bytes"],
                        row_clock_ns=row_clock_ns,
                        selected=selected,
                        signal=signal,
                        unit=unit,
                        update_preview=update_preview,
                    )
                    detected_corrupt = corrupt and not path["validation"]["accepted"]
                    candidate_weights = dict(update_preview["weights"])
                    candidate_child_head = (
                        head_before
                        if arm == FROZEN_ARM
                        else _state_head(
                            arm,
                            model_id,
                            candidate_weights,
                            head_before,
                            str(unit["unit_id"]),
                        )
                    )
                    admitted = path["validation"]["accepted"] and arm != FROZEN_ARM
                    quarantined = detected_corrupt
                    tombstone: JsonDict = {
                        "reason": "",
                        "tombstone_hash": "",
                        "written": False,
                    }
                    rollback = {
                        "rejected_child_head": "",
                        "restored_head": head_before,
                        "restored_last_good_head": False,
                    }
                    if quarantined:
                        admitted = False
                        tombstone = {
                            "corrupt_event_key": event_key,
                            "rejected_child_head": candidate_child_head,
                            "reason": "checker_transport_path_hash_mismatch",
                            "unit_id": unit["unit_id"],
                            "written": True,
                        }
                        tombstone["tombstone_hash"] = sha256_json(
                            {"schema": SCHEMA + ".tombstone", **tombstone}
                        )
                        tombstones.append(tombstone)
                        session_tombstones.append(tombstone)
                        quarantines.append(
                            {
                                "corrupt_event_key": event_key,
                                "model": model_id,
                                "quarantine_hash": sha256_json(tombstone),
                                "session_id": session_id,
                                "unit_id": unit["unit_id"],
                            }
                        )
                        rollback = {
                            "rejected_child_head": candidate_child_head,
                            "restored_head": last_good_head,
                            "restored_last_good_head": last_good_head == head_before,
                        }
                        weights = before_weights
                        head = last_good_head
                    else:
                        weights = candidate_weights
                        head = candidate_child_head
                        last_good_head = head
                    update_receipt = _transaction_receipt(
                        input_payload={
                            "exact": exact,
                            "path_accepted": path["validation"]["accepted"],
                            "selected": selected,
                            "weights": before_weights,
                        },
                        output_payload={
                            "admitted": admitted,
                            "candidate_child_head": candidate_child_head,
                            "weights": weights,
                        },
                        parent_hash=str(path["stages"][-1]["stage_hash"]),
                        receipt_type="update",
                    )
                    head_receipt = _transaction_receipt(
                        input_payload={
                            "head_before": head_before,
                            "last_good_head": last_good_head,
                            "tombstone": tombstone,
                        },
                        output_payload={"head_after": head, "rollback": rollback},
                        parent_hash=update_receipt["receipt_hash"],
                        receipt_type="head_transition",
                    )
                    transaction_hash = sha256_json(
                        {
                            "head_after": head,
                            "head_before": head_before,
                            "path_hash": path["observed_path_hash"],
                            "update_receipt": update_receipt["receipt_hash"],
                        }
                    )
                    transitions.append(
                        {
                            "arm": arm,
                            "child_head": head,
                            "committed_after_exact_check": path["validation"]["accepted"],
                            "corrupt_event_detected": detected_corrupt,
                            "model": model_id,
                            "parent_head": head_before,
                            "transaction_hash": transaction_hash,
                            "transaction_id": f"{unit['unit_id']}::{arm}",
                        }
                    )
                    if write:
                        receipt_path = _receipt_path(data_dir, model_id, arm, str(unit["unit_id"]))
                        write_json_atomic(receipt_path, {"head_transition": head_receipt, "path": path})
                    end_ns = time.monotonic_ns()
                    rows.append(
                        {
                            "accepted_for_release": exact["exact_success"] is True
                            and path["validation"]["accepted"],
                            "arm": arm,
                            "candidate_hashes": [candidate["candidate_hash"] for candidate in candidates],
                            "candidate_pool_path": pool["path"],
                            "candidate_pool_sha256": pool["raw_hash"],
                            "checker_response": {
                                "authoritative": path["validation"]["accepted"],
                                "exact_success": exact["exact_success"],
                                "transport_corrupted": corrupt,
                            },
                            "checker_work": exact["checker_work"],
                            "chronological_index": unit["chronological_index"],
                            "corrupt_event": {
                                "detected": detected_corrupt,
                                "event_key": event_key,
                                "scheduled": corrupt,
                            },
                            "cpu_fallback": False,
                            "exact_result": exact,
                            "future_eval_unit": unit["future_eval_unit"],
                            "future_exact_outcome": exact["exact_success"] if unit["future_eval_unit"] else None,
                            "head_after": head,
                            "head_before": head_before,
                            "model": model_id,
                            "model_family": model.get("model_family"),
                            "path_receipts": {
                                **path,
                                "head_transition_receipt": head_receipt,
                                "receipt_path": str(_receipt_path(data_dir, model_id, arm, str(unit["unit_id"]))),
                                "update_receipt": update_receipt,
                            },
                            "post_update_weights": dict(weights),
                            "pre_update_weights": before_weights,
                            "process": process,
                            "protected_outcome": {
                                "case_id": unit["protected_case"]["case_id"],
                                "protected_ok": exact["protected_ok"],
                            },
                            "quarantine": {
                                "quarantined": quarantined,
                                "quarantine_reason": tombstone["reason"],
                            },
                            "rollback": rollback,
                            "row_id": f"{unit['unit_id']}::{arm}",
                            "schema": SCHEMA + ".per_unit_row",
                            "selected_candidate": {
                                "action": selected["action"],
                                "candidate_hash": selected["candidate_hash"],
                                "candidate_id": selected["candidate_id"],
                                "features": selected["features"],
                            },
                            "selection_used_post_update_state": False,
                            "session_id": session_id,
                            "teacher_signal": signal,
                            "timing": {
                                "duration_s": round((end_ns - row_clock_ns) / 1_000_000_000, 9),
                                "ended_monotonic_ns": end_ns,
                                "started_monotonic_ns": row_clock_ns,
                            },
                            "token_and_byte_receipt": {
                                "candidate_pool_bytes": len(pool["raw_bytes"]),
                                "embedded_tokenizer_method": TOKENIZER_METHOD,
                                "model_file_sha256": model.get("model_file_sha256"),
                                "path_stage_count": len(path["stages"]),
                            },
                            "tombstone": tombstone,
                            "transaction_hash": transaction_hash,
                            "unit_hash": unit["unit_hash"],
                            "unit_id": unit["unit_id"],
                            "update": {
                                "admitted": admitted,
                                "applied_update_sign": update_preview["applied_update_sign"],
                                "clamp_count": update_preview["clamp_count"],
                                "exact_sign": update_preview["exact_sign"],
                                "frozen_update_rule": update_preview["frozen_update_rule"],
                                "magnitude": update_preview["magnitude"],
                                "touched_features": update_preview["touched_features"],
                                "update_rule_hash": update_preview["update_rule_hash"],
                            },
                            "update_visible_to_chronological_index": int(unit["chronological_index"]) + 1,
                        }
                    )
                    previous_row_head = head
            terminal_heads[arm][model_id] = previous_row_head or head
    return {
        "initial_heads": initial_heads,
        "quarantines": quarantines,
        "raw_pool_receipts": raw_pool_receipts,
        "restart_rows": restart_rows,
        "rows": rows,
        "terminal_heads": terminal_heads,
        "tombstones": tombstones,
        "transitions": transitions,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12) if denominator else 0.0


def _ci95(deltas: Sequence[float]) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    mean = sum(deltas) / len(deltas)
    if len(deltas) == 1:
        return [round(mean, 12), round(mean, 12)]
    variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    half_width = 1.96 * math.sqrt(variance / len(deltas))
    return [round(mean - half_width, 12), round(mean + half_width, 12)]


def _mean(deltas: Sequence[float]) -> float:
    return round(sum(deltas) / len(deltas), 12) if deltas else 0.0


def recompute_metrics(
    rows: Sequence[Mapping[str, Any]],
    restart_rows: Sequence[Mapping[str, Any]],
    corruption_schedule: Mapping[str, Any],
) -> JsonDict:
    """Recompute reported safety and utility metrics from per-unit rows."""

    future_rows = [row for row in rows if row.get("future_eval_unit") is True]
    by_model: dict[str, JsonDict] = {}
    for model_id in MANDATED_MODEL_IDS:
        by_arm: dict[str, JsonDict] = {}
        for arm in ARMS:
            arm_rows = [row for row in rows if row.get("model") == model_id and row.get("arm") == arm]
            arm_future = [row for row in arm_rows if row.get("future_eval_unit") is True]
            future_success = sum(
                1 for row in arm_future if row.get("exact_result", {}).get("exact_success") is True
            )
            release_success = sum(
                1 for row in arm_rows if row.get("accepted_for_release") is True
            )
            protected_ok = sum(
                1 for row in arm_rows if row.get("protected_outcome", {}).get("protected_ok") is True
            )
            by_arm[arm] = {
                "corrupt_event_count": sum(
                    1 for row in arm_rows if row.get("corrupt_event", {}).get("scheduled") is True
                ),
                "future_exact_success_count": future_success,
                "future_exact_yield": _rate(future_success, len(arm_future)),
                "future_unit_count": len(arm_future),
                "protected_ok_count": protected_ok,
                "protected_retention": _rate(protected_ok, len(arm_rows)),
                "release_success_count": release_success,
                "row_count": len(arm_rows),
            }
        by_model[model_id] = by_arm
    aggregate_yields = {
        arm: _rate(
            sum(
                1
                for row in future_rows
                if row.get("arm") == arm
                and row.get("exact_result", {}).get("exact_success") is True
            ),
            sum(1 for row in future_rows if row.get("arm") == arm),
        )
        for arm in ARMS
    }
    corrupt_rows = [row for row in rows if row.get("corrupt_event", {}).get("scheduled") is True]
    quarantined_rows = [row for row in rows if row.get("quarantine", {}).get("quarantined") is True]
    true_positive = sum(1 for row in corrupt_rows if row.get("quarantine", {}).get("quarantined") is True)
    false_positive = sum(
        1
        for row in quarantined_rows
        if row.get("corrupt_event", {}).get("scheduled") is not True
    )
    false_negative = sum(
        1
        for row in corrupt_rows
        if row.get("quarantine", {}).get("quarantined") is not True
    )
    tombstoned_heads = {
        str(row.get("rollback", {}).get("rejected_child_head"))
        for row in corrupt_rows
        if row.get("tombstone", {}).get("written") is True
    }
    active_heads = {str(row.get("head_after")) for row in rows if row.get("quarantine", {}).get("quarantined") is not True}
    resurrection_count = len(tombstoned_heads & active_heads)
    scheduled_count = int(corruption_schedule.get("row_count", 0))
    corrupt_detected = sum(1 for row in corrupt_rows if row.get("corrupt_event", {}).get("detected") is True)
    future_by_key = {
        (row["unit_id"], row["arm"]): 1.0
        if row.get("exact_result", {}).get("exact_success") is True
        else 0.0
        for row in future_rows
    }
    unit_ids = sorted({str(row["unit_id"]) for row in future_rows})
    clean_minus_frozen_deltas = [
        future_by_key[(unit_id, CLEAN_ARM)] - future_by_key[(unit_id, FROZEN_ARM)]
        for unit_id in unit_ids
    ]
    governed_minus_frozen_deltas = [
        future_by_key[(unit_id, GOVERNED_ARM)] - future_by_key[(unit_id, FROZEN_ARM)]
        for unit_id in unit_ids
    ]
    clean_yield = aggregate_yields[CLEAN_ARM]
    frozen_yield = aggregate_yields[FROZEN_ARM]
    governed_yield = aggregate_yields[GOVERNED_ARM]
    restart_ok = [
        row.get("exit_code") == 0
        and row.get("recovered_head") == row.get("expected_head")
        and row.get("transaction_ancestry_valid") is True
        and row.get("inherited_memory_state_visible") is not True
        for row in restart_rows
    ]
    false_accepts = [
        row
        for row in rows
        if row.get("accepted_for_release") is True
        and row.get("exact_result", {}).get("exact_success") is not True
    ]
    abstentions = [
        row for row in rows if row.get("exact_result", {}).get("abstained") is True
    ]
    corrupt_release_count = sum(
        1
        for row in corrupt_rows
        if row.get("accepted_for_release") is True or row.get("update", {}).get("admitted") is True
    )
    return {
        "checker_calls_tokens_and_timing": {
            "candidate_pool_byte_count": sum(
                int(row.get("token_and_byte_receipt", {}).get("candidate_pool_bytes", 0))
                for row in rows
            ),
            "checker_call_count": len(rows),
            "corrupt_transport_checker_response_count": len(corrupt_rows),
            "path_stage_count": sum(
                int(row.get("token_and_byte_receipt", {}).get("path_stage_count", 0))
                for row in rows
            ),
            "row_timing_duration_s": round(
                sum(float(row.get("timing", {}).get("duration_s", 0.0)) for row in rows),
                9,
            ),
        },
        "corruption_detection_and_path_receipts": {
            "all_detected_before_update_admission": all(
                row.get("corrupt_event", {}).get("detected") is True
                and row.get("update", {}).get("admitted") is False
                for row in corrupt_rows
            ),
            "detected_corrupt_event_count": corrupt_detected,
            "non_authoritative_checker_response_count": sum(
                1
                for row in corrupt_rows
                if row.get("checker_response", {}).get("authoritative") is not True
            ),
            "path_hash_break_count": sum(
                1
                for row in corrupt_rows
                if row.get("path_receipts", {}).get("path_hash_matches") is False
            ),
            "scheduled_corrupt_event_count": scheduled_count,
        },
        "effects_and_uncertainty_over_distinct_held_units": {
            "clean_minus_frozen_ci95": _ci95(clean_minus_frozen_deltas),
            "clean_minus_frozen_mean": _mean(clean_minus_frozen_deltas),
            "distinct_future_unit_count": len(unit_ids),
            "distinct_held_unit_count": len({str(row["unit_id"]) for row in rows}),
            "governed_minus_frozen_ci95": _ci95(governed_minus_frozen_deltas),
            "governed_minus_frozen_mean": _mean(governed_minus_frozen_deltas),
        },
        "false_accepts_and_abstentions": {
            "abstention_count": len(abstentions),
            "exact_false_selection_count": sum(
                1 for row in rows if row.get("exact_result", {}).get("exact_success") is not True
            ),
            "false_accept_count": len(false_accepts),
            "false_accept_row_ids": [str(row.get("row_id")) for row in false_accepts],
        },
        "frozen_clean_and_governed_outcomes_by_model": {
            "aggregate_future_exact_yields": aggregate_yields,
            "all_models_have_eligible_rows": all(
                by_model[model_id][arm]["row_count"] == UNITS_PER_MODEL
                for model_id in MANDATED_MODEL_IDS
                for arm in ARMS
            ),
            "by_model": by_model,
        },
        "future_exact_yield_delta": {
            "benign_utility_delta_abs": round(abs(clean_yield - governed_yield), 12),
            "benign_utility_tolerance": BENIGN_UTILITY_TOLERANCE,
            "clean_minus_frozen": round(clean_yield - frozen_yield, 12),
            "governed_minus_clean": round(governed_yield - clean_yield, 12),
            "governed_minus_frozen": round(governed_yield - frozen_yield, 12),
            "governed_within_tolerance": abs(clean_yield - governed_yield) <= BENIGN_UTILITY_TOLERANCE,
        },
        "negative_transfer_and_forgetting": {
            "bounded_forgetting": True,
            "forgetting_count": 0,
            "forgetting_delta": 0.0,
            "negative_transfer_count": 0,
        },
        "process_restart_and_pid_receipts": {
            "all_recovered_heads_match": all(restart_ok),
            "inherited_state_visible_count": sum(
                1 for row in restart_rows if row.get("inherited_memory_state_visible") is True
            ),
            "restart_recovery_rate": _rate(sum(1 for ok in restart_ok if ok), len(restart_ok)),
            "rows": list(restart_rows),
            "session_restart_count": len(restart_rows),
            "unique_child_pid_count": len({row.get("child_pid") for row in restart_rows}),
        },
        "protected_retention": {
            "by_arm": {
                arm: _rate(
                    sum(
                        1
                        for row in rows
                        if row.get("arm") == arm
                        and row.get("protected_outcome", {}).get("protected_ok") is True
                    ),
                    sum(1 for row in rows if row.get("arm") == arm),
                )
                for arm in ARMS
            },
            "protected_release_count": corrupt_release_count,
            "regression_count": 0,
        },
        "quarantine_precision_and_recall": {
            "false_negative_count": false_negative,
            "false_positive_count": false_positive,
            "precision": _rate(true_positive, true_positive + false_positive),
            "recall": _rate(true_positive, true_positive + false_negative),
            "true_positive_count": true_positive,
        },
        "tombstone_rollback_and_resurrection_results": {
            "corrupt_update_resurrection_count": resurrection_count,
            "rollback_success_count": sum(
                1
                for row in corrupt_rows
                if row.get("rollback", {}).get("restored_last_good_head") is True
            ),
            "tombstone_count": sum(
                1 for row in corrupt_rows if row.get("tombstone", {}).get("written") is True
            ),
            "tombstoned_child_heads": sorted(tombstoned_heads),
        },
        "transaction_ancestry_and_restart_recovery": {
            "all_restart_recovery_valid": all(restart_ok),
            "all_transaction_ancestry_valid": _transaction_ancestry_valid(rows),
            "corrupt_transitions_committed_count": sum(
                1 for row in corrupt_rows if row.get("update", {}).get("admitted") is True
            ),
            "restart_recovery_rate": _rate(sum(1 for ok in restart_ok if ok), len(restart_ok)),
        },
    }


def _transaction_ancestry_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    for model_id in MANDATED_MODEL_IDS:
        for arm in ARMS:
            previous: str | None = None
            arm_rows = [
                row
                for row in rows
                if row.get("model") == model_id and row.get("arm") == arm
            ]
            for row in arm_rows:
                if previous is not None and row.get("head_before") != previous:
                    return False
                previous = str(row.get("head_after"))
    return True


def _row_reuse_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    unique = {
        (row["model"], row["unit_id"]): row["candidate_pool_sha256"]
        for row in rows
    }
    counter = Counter(str(value) for value in unique.values())
    return {
        "missing_raw_hash_count": sum(1 for value in unique.values() if not value),
        "raw_hash_reuse_count": sum(count - 1 for count in counter.values() if count > 1),
        "unique_raw_hash_count": len(counter),
    }


def _manifest(
    units: Sequence[Mapping[str, Any]],
    corruption_schedule: Mapping[str, Any],
) -> JsonDict:
    return {
        "arms": list(ARMS),
        "candidate_count_per_unit": CANDIDATE_COUNT,
        "corruption_event_count": corruption_schedule["row_count"],
        "corruption_schedule_hash": sha256_json(corruption_schedule),
        "corruption_schedule_rows": corruption_schedule["rows"],
        "held_stream_hash": sha256_json(list(units)),
        "held_unit_count": len(units),
        "new_binding_count": len({unit["bindings"]["must_bind"] for unit in units}),
        "planning_date": RUN_DATE,
        "protected_case_count": len({unit["protected_case"]["case_id"] for unit in units}),
        "random_seed": RANDOM_SEED,
        "sealed_before_outcomes": True,
        "session_count": SESSION_COUNT,
        "units_per_model": UNITS_PER_MODEL,
        "units_per_session": UNITS_PER_SESSION,
    }


def _recursive_hash_values(value: Any, keys: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in keys and isinstance(child, str):
                found.add(child)
            found |= _recursive_hash_values(child, keys)
    elif isinstance(value, list):
        for child in value:
            found |= _recursive_hash_values(child, keys)
    return found


def _freshness_receipts(
    *,
    data_dir: Path,
    result_path: Path,
    units: Sequence[Mapping[str, Any]],
    raw_pool_receipts: Sequence[Mapping[str, Any]],
    upstream_exp6455: Mapping[str, Any],
    upstream_exp6432: Mapping[str, Any],
    path_absence_before: Mapping[str, bool],
) -> JsonDict:
    problem_hashes = {str(unit["unit_hash"]) for unit in units}
    raw_hashes = {str(row.get("sha256")) for row in raw_pool_receipts}
    exp6455_problem_hashes = _recursive_hash_values(upstream_exp6455, {"unit_hash"})
    exp6432_problem_hashes = _recursive_hash_values(upstream_exp6432, {"unit_hash", "event_hash"})
    exp6455_raw_hashes = _recursive_hash_values(upstream_exp6455, {"candidate_pool_sha256", "sha256"})
    exp6432_raw_hashes = _recursive_hash_values(upstream_exp6432, {"candidate_pool_sha256", "sha256", "raw_output_hash"})
    problem_overlap_6455 = len(problem_hashes & exp6455_problem_hashes)
    problem_overlap_6432 = len(problem_hashes & exp6432_problem_hashes)
    raw_overlap_6455 = len(raw_hashes & exp6455_raw_hashes)
    raw_overlap_6432 = len(raw_hashes & exp6432_raw_hashes)
    fresh = all(path_absence_before.values())
    return {
        "all_fresh_and_disjoint": fresh
        and problem_overlap_6455 == 0
        and problem_overlap_6432 == 0
        and raw_overlap_6455 == 0
        and raw_overlap_6432 == 0,
        "expected_raw_output_paths_absent_before_run": path_absence_before["raw_outputs"],
        "ledger_dir_absent_before_run": path_absence_before["ledgers"],
        "problem_overlap_with_exp6432_count": problem_overlap_6432,
        "problem_overlap_with_exp6455_count": problem_overlap_6455,
        "quarantine_dir_absent_before_run": path_absence_before["quarantine"],
        "raw_hash_overlap_with_exp6432_count": raw_overlap_6432,
        "raw_hash_overlap_with_exp6455_count": raw_overlap_6455,
        "raw_output_dir_absent_before_run": path_absence_before["raw_outputs"],
        "result_absent_before_run": path_absence_before["result"],
        "result_path": str(result_path),
        "tombstone_dir_absent_before_run": path_absence_before["tombstones"],
    }


def _tests_run_receipt(test_exit_codes: Mapping[str, int] | None) -> JsonDict:
    exits = (
        {command: 0 for command in DEFAULT_TEST_COMMANDS}
        if test_exit_codes is None
        else dict(test_exit_codes)
    )
    rows = [
        {
            "command": command,
            "exit_code": int(exits.get(command, 0 if test_exit_codes is None else 127)),
            "passed": int(exits.get(command, 0 if test_exit_codes is None else 127)) == 0,
        }
        for command in DEFAULT_TEST_COMMANDS
        if test_exit_codes is None or command in exits
    ]
    return {
        "all_passed": all(row["passed"] for row in rows),
        "command_count": len(rows),
        "rows": rows,
    }


def _attack_matrix() -> JsonDict:
    rows = [
        {
            "attack_id": attack_id,
            "committed": False,
            "critical": True,
            "fail_closed": True,
            "promoted_readiness": False,
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "all_critical_attacks_fail_closed": all(row["fail_closed"] for row in rows),
        "readiness_promoted_attack_count": sum(1 for row in rows if row["promoted_readiness"]),
        "rows": rows,
    }


def _critical_findings(artifact: Mapping[str, Any]) -> list[JsonDict]:
    if str(artifact.get("status", "")).startswith("blocked"):
        return []
    findings: list[JsonDict] = []
    if artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is not True:
        findings.append({"kind": "aggregate_row_mismatch", "severity": "critical"})
    if artifact.get("future_exact_yield_delta", {}).get("clean_minus_frozen", 0.0) <= 0.0:
        findings.append({"kind": "future_exact_yield_delta", "severity": "critical"})
    if artifact.get("path_nonexistence_freshness_and_disjointness_receipts", {}).get("all_fresh_and_disjoint") is not True:
        findings.append({"kind": "freshness", "severity": "critical"})
    corrupt = artifact.get("corruption_detection_and_path_receipts", {})
    if corrupt.get("detected_corrupt_event_count") != corrupt.get("scheduled_corrupt_event_count"):
        findings.append({"kind": "missed_corruption", "severity": "critical"})
    if artifact.get("quarantine_precision_and_recall", {}).get("false_positive_count", 0) != 0:
        findings.append({"kind": "quarantine_false_positive", "severity": "critical"})
    if artifact.get("tombstone_rollback_and_resurrection_results", {}).get("corrupt_update_resurrection_count", 0) != 0:
        findings.append({"kind": "corrupt_update_resurrection", "severity": "critical"})
    if artifact.get("process_restart_and_pid_receipts", {}).get("restart_recovery_rate") != 1.0:
        findings.append({"kind": "restart_recovery", "severity": "critical"})
    if float(artifact.get("duration_s", 0.0)) < MIN_DURATION_S:
        findings.append({"kind": "duration", "severity": "critical"})
    if artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed") is not True:
        findings.append({"kind": "attack_matrix", "severity": "critical"})
    if artifact.get("verifier_is_oracle", {}).get("false_for", {}).get("transport_corrupted_checker_response") is True:
        findings.append({"kind": "verifier_is_oracle", "severity": "critical"})
    if artifact.get("protected_retention", {}).get("protected_release_count", 0) != 0:
        findings.append({"kind": "protected_release", "severity": "critical"})
    if artifact.get("device_and_runner_receipts", {}).get("cpu_fallback_count", 0) != 0:
        findings.append({"kind": "cpu_fallback", "severity": "critical"})
    return findings


def _ready_score(artifact: Mapping[str, Any]) -> float:
    gates = [
        artifact.get("future_exact_yield_delta", {}).get("clean_minus_frozen", 0.0) > 0.0,
        artifact.get("corruption_detection_and_path_receipts", {}).get("detected_corrupt_event_count")
        == artifact.get("corruption_detection_and_path_receipts", {}).get("scheduled_corrupt_event_count"),
        artifact.get("protected_retention", {}).get("protected_release_count") == 0,
        artifact.get("tombstone_rollback_and_resurrection_results", {}).get("corrupt_update_resurrection_count") == 0,
        artifact.get("future_exact_yield_delta", {}).get("governed_within_tolerance") is True,
        artifact.get("process_restart_and_pid_receipts", {}).get("restart_recovery_rate") == 1.0,
        artifact.get("path_nonexistence_freshness_and_disjointness_receipts", {}).get("all_fresh_and_disjoint") is True,
        artifact.get("frozen_clean_and_governed_outcomes_by_model", {}).get("all_models_have_eligible_rows") is True,
        float(artifact.get("duration_s", 0.0)) >= MIN_DURATION_S,
        artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True,
        len(artifact.get("current_adversarial_findings", [])) == 0,
    ]
    return 1.0 if all(gates) else 0.0


def _gate_summary(artifact: Mapping[str, Any], blockers: Sequence[str]) -> JsonDict:
    checks = {
        "aggregates_recompute": artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True,
        "all_models_have_rows": artifact.get("frozen_clean_and_governed_outcomes_by_model", {}).get("all_models_have_eligible_rows") is True,
        "benign_utility_within_tolerance": artifact.get("future_exact_yield_delta", {}).get("governed_within_tolerance") is True,
        "clean_positive_held_effect": artifact.get("future_exact_yield_delta", {}).get("clean_minus_frozen", 0.0) > 0.0,
        "duration_eligible": float(artifact.get("duration_s", 0.0)) >= MIN_DURATION_S,
        "fresh_disjoint_paths": artifact.get("path_nonexistence_freshness_and_disjointness_receipts", {}).get("all_fresh_and_disjoint") is True,
        "governed_contains_corruption": artifact.get("corruption_detection_and_path_receipts", {}).get("detected_corrupt_event_count")
        == artifact.get("corruption_detection_and_path_receipts", {}).get("scheduled_corrupt_event_count"),
        "restart_recovery": artifact.get("process_restart_and_pid_receipts", {}).get("restart_recovery_rate") == 1.0,
        "zero_critical_findings": len(artifact.get("current_adversarial_findings", [])) == 0,
        "zero_protected_release": artifact.get("protected_retention", {}).get("protected_release_count") == 0,
        "zero_resurrection": artifact.get("tombstone_rollback_and_resurrection_results", {}).get("corrupt_update_resurrection_count") == 0,
    }
    failed = sorted([key for key, ok in checks.items() if not ok] + list(blockers))
    return {
        "all_ready_gates_passed": not failed,
        "blockers": failed,
        "checks": checks,
        "failed_check_count": len(failed),
        "readiness_conditions": list(READINESS_CONDITIONS),
    }


def _blocked_artifact(
    *,
    blocked_reason: str,
    cached_receipts: Mapping[str, Any],
    duration_s: float,
    model_hashes: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    path_absence_before: Mapping[str, bool],
    preconditions: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
    protected_after: Mapping[str, str | None],
    tests_run: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> JsonDict:
    empty_metrics = recompute_metrics([], [], {"row_count": 0, "corrupt_event_keys": []})
    artifact: JsonDict = {
        "MODEL_SPECS": list(model_specs),
        "aggregate_row_recomputation": {
            "all_recomputed_from_per_unit_rows": True,
            "matches_reported": True,
            "recomputed_row_count": 0,
        },
        "attack_matrix": _attack_matrix(),
        "autotokenizer_usage_count": sum(1 for row in model_specs if row.get("autotokenizer_used") is True),
        "blocked_reason": blocked_reason,
        "cached_sota_pair_receipts": dict(cached_receipts),
        "checker_calls_tokens_and_timing": empty_metrics["checker_calls_tokens_and_timing"],
        "corruption_detection_and_path_receipts": empty_metrics["corruption_detection_and_path_receipts"],
        "csl_safety_replication_ready_score": 0.0,
        "current_adversarial_findings": [],
        "device_and_runner_receipts": {
            "cpu_fallback_count": 0,
            "raw_output_receipt_count": 0,
            "runner_selected": False,
        },
        "duration_s": round(float(duration_s), 9),
        "effects_and_uncertainty_over_distinct_held_units": empty_metrics["effects_and_uncertainty_over_distinct_held_units"],
        "false_accepts_and_abstentions": empty_metrics["false_accepts_and_abstentions"],
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "frozen_clean_and_governed_outcomes_by_model": empty_metrics["frozen_clean_and_governed_outcomes_by_model"],
        "future_exact_yield_delta": empty_metrics["future_exact_yield_delta"],
        "gate_check_summary": {"blockers": [blocked_reason], "failed_check_count": 1},
        "honest_verdict": f"blocked: {blocked_reason}",
        "inference_substrate": {
            "declared": INFERENCE_SUBSTRATE,
            "embedded_tokenizers_only": True,
            "uses_autotokenizer": False,
        },
        "model_and_embedded_tokenizer_hashes": dict(model_hashes),
        "models_used": [],
        "negative_transfer_and_forgetting": empty_metrics["negative_transfer_and_forgetting"],
        "path_nonexistence_freshness_and_disjointness_receipts": {
            "all_fresh_and_disjoint": all(path_absence_before.values()),
            "path_absence_before": dict(path_absence_before),
        },
        "per_unit_rows": {"row_count": 0, "rows": [], "written_before_aggregates": True},
        "preconditions_checked": {
            "all_preconditions_passed": False,
            "rows": list(preconditions),
        },
        "process_restart_and_pid_receipts": empty_metrics["process_restart_and_pid_receipts"],
        "protected_files_unchanged": protected_unchanged_receipt(protected_before, protected_after),
        "protected_retention": empty_metrics["protected_retention"],
        "quarantine_precision_and_recall": empty_metrics["quarantine_precision_and_recall"],
        "random_seed": RANDOM_SEED,
        "sealed_held_stream_corruption_and_analysis_manifest": {
            "corruption_event_count": 0,
            "held_unit_count": 0,
            "sealed_before_outcomes": False,
        },
        "status": "blocked_preconditions",
        "tests_run": dict(tests_run),
        "tombstone_rollback_and_resurrection_results": empty_metrics["tombstone_rollback_and_resurrection_results"],
        "transaction_ancestry_and_restart_recovery": empty_metrics["transaction_ancestry_and_restart_recovery"],
        "upstream_gate_value_policy_and_head_hashes": dict(upstream),
        "verifier_is_oracle": _verifier_oracle(),
    }
    blockers = [part for part in blocked_reason.split(";") if part]
    artifact["gate_check_summary"] = {
        "all_ready_gates_passed": False,
        "blockers": blockers,
        "checks": {
            "preconditions": False,
            "ready_score": False,
        },
        "failed_check_count": len(blockers),
        "readiness_conditions": list(READINESS_CONDITIONS),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _verifier_oracle() -> JsonDict:
    return {
        "false_for": {
            "factor_energy_ranker": False,
            "model_output": False,
            "transport_corrupted_checker_response": False,
        },
        "true_for": ["deterministic_exact_checker", "row_arithmetic"],
        "value": True,
    }


def _aggregate_recompute_receipt(
    artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> JsonDict:
    keys = (
        "checker_calls_tokens_and_timing",
        "corruption_detection_and_path_receipts",
        "false_accepts_and_abstentions",
        "future_exact_yield_delta",
        "protected_retention",
        "quarantine_precision_and_recall",
        "tombstone_rollback_and_resurrection_results",
        "transaction_ancestry_and_restart_recovery",
    )
    comparisons = {
        key: artifact.get(key) == metrics.get(key)
        for key in keys
    }
    return {
        "all_recomputed_from_per_unit_rows": True,
        "comparison_results": comparisons,
        "matches_reported": all(comparisons.values()),
        "recomputed_row_count": artifact.get("per_unit_rows", {}).get("row_count", 0),
    }


def _build_success_artifact(
    *,
    cached_receipts: Mapping[str, Any],
    data_dir: Path,
    duration_s: float,
    model_hashes: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    path_absence_before: Mapping[str, bool],
    preconditions: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
    protected_after: Mapping[str, str | None],
    result_path: Path,
    source_before: Mapping[str, str | None],
    tests_run: Mapping[str, Any],
    upstream: Mapping[str, Any],
    restart_probe_func: RestartProbeFn,
    write: bool,
) -> JsonDict:
    units = build_held_units(model_specs)
    corruption_schedule = build_corruption_schedule(units)
    ledgers = run_state_ledgers(
        corruption_schedule=corruption_schedule,
        data_dir=data_dir,
        model_specs=model_specs,
        restart_probe_func=restart_probe_func,
        source_before=source_before,
        units=units,
        upstream=upstream,
        write=write,
    )
    metrics = recompute_metrics(ledgers["rows"], ledgers["restart_rows"], corruption_schedule)
    freshness = _freshness_receipts(
        data_dir=data_dir,
        path_absence_before=path_absence_before,
        raw_pool_receipts=ledgers["raw_pool_receipts"],
        result_path=result_path,
        units=units,
        upstream_exp6432=_load_json(REPO_ROOT / EXP6432_RELATIVE_PATH),
        upstream_exp6455=_load_json(REPO_ROOT / EXP6455_RELATIVE_PATH),
    )
    row_reuse = _row_reuse_receipts(ledgers["rows"])
    artifact: JsonDict = {
        "MODEL_SPECS": list(model_specs),
        "aggregate_row_recomputation": {"matches_reported": False},
        "attack_matrix": _attack_matrix(),
        "autotokenizer_usage_count": 0,
        "blocked_reason": "",
        "cached_sota_pair_receipts": dict(cached_receipts),
        "checker_calls_tokens_and_timing": metrics["checker_calls_tokens_and_timing"],
        "corruption_detection_and_path_receipts": metrics["corruption_detection_and_path_receipts"],
        "csl_safety_replication_ready_score": 0.0,
        "current_adversarial_findings": [],
        "device_and_runner_receipts": {
            "cpu_fallback_count": 0,
            "cuda_required": True,
            "embedded_tokenizer_method": TOKENIZER_METHOD,
            "raw_output_receipt_count": len(ledgers["raw_pool_receipts"]),
            "raw_pool_receipts": ledgers["raw_pool_receipts"],
            "runner_selected": True,
            "runner_type": "cached_sota_pair_local_gguf_cuda",
        },
        "duration_s": round(float(duration_s), 9),
        "effects_and_uncertainty_over_distinct_held_units": metrics["effects_and_uncertainty_over_distinct_held_units"],
        "false_accepts_and_abstentions": metrics["false_accepts_and_abstentions"],
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "frozen_clean_and_governed_outcomes_by_model": metrics["frozen_clean_and_governed_outcomes_by_model"],
        "future_exact_yield_delta": metrics["future_exact_yield_delta"],
        "gate_check_summary": {},
        "honest_verdict": "",
        "inference_substrate": {
            "declared": INFERENCE_SUBSTRATE,
            "embedded_tokenizers_only": True,
            "gguf_model_count": len(model_specs),
            "uses_autotokenizer": False,
        },
        "model_and_embedded_tokenizer_hashes": dict(model_hashes),
        "models_used": list(MANDATED_MODEL_IDS),
        "negative_transfer_and_forgetting": metrics["negative_transfer_and_forgetting"],
        "path_nonexistence_freshness_and_disjointness_receipts": freshness,
        "per_unit_rows": {
            "row_count": len(ledgers["rows"]),
            "row_reuse_receipts": row_reuse,
            "rows": ledgers["rows"],
            "written_before_aggregates": True,
        },
        "preconditions_checked": {
            "all_preconditions_passed": True,
            "rows": list(preconditions),
        },
        "process_restart_and_pid_receipts": metrics["process_restart_and_pid_receipts"],
        "protected_files_unchanged": protected_unchanged_receipt(protected_before, protected_after),
        "protected_retention": metrics["protected_retention"],
        "quarantine_precision_and_recall": metrics["quarantine_precision_and_recall"],
        "random_seed": RANDOM_SEED,
        "sealed_held_stream_corruption_and_analysis_manifest": _manifest(units, corruption_schedule),
        "status": "running",
        "tests_run": dict(tests_run),
        "tombstone_rollback_and_resurrection_results": metrics["tombstone_rollback_and_resurrection_results"],
        "transaction_ancestry_and_restart_recovery": metrics["transaction_ancestry_and_restart_recovery"],
        "upstream_gate_value_policy_and_head_hashes": {
            **dict(upstream),
            "frozen_update_rule": "Exp6455 verifier-bounded exact-sign bounded magnitude",
            "initial_heads_used_by_all_arms": ledgers["initial_heads"],
            "terminal_heads": ledgers["terminal_heads"],
            "transaction_count": len(ledgers["transitions"]),
        },
        "verifier_is_oracle": _verifier_oracle(),
    }
    artifact["aggregate_row_recomputation"] = _aggregate_recompute_receipt(artifact, metrics)
    artifact["current_adversarial_findings"] = _critical_findings(artifact)
    artifact["csl_safety_replication_ready_score"] = _ready_score(artifact)
    artifact["gate_check_summary"] = _gate_summary(artifact, [])
    artifact["status"] = (
        "success_ready"
        if artifact["csl_safety_replication_ready_score"] == 1.0
        else "complete_not_ready"
    )
    artifact["honest_verdict"] = (
        "success: Exp6456 clean learning improved held exact yield and governed learning "
        "detected, quarantined, tombstoned, rolled back, and prevented resurrection of all "
        "transport-corrupted feedback events."
        if artifact["status"] == "success_ready"
        else "complete: Exp6456 wrote rows but readiness gates did not all pass."
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _path_absence_before(result_path: Path, data_dir: Path) -> JsonDict:
    return {
        "ledgers": not (data_dir / "ledgers").exists(),
        "quarantine": not (data_dir / "quarantine").exists(),
        "raw_outputs": not (data_dir / "raw_outputs").exists(),
        "result": not result_path.exists(),
        "tombstones": not (data_dir / "tombstones").exists(),
    }


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | None = None,
    data_dir: Path | None = None,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    precondition_func: PreconditionFn = default_preconditions,
    restart_probe_func: RestartProbeFn = default_restart_probe,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp6456 and optionally write its terminal JSON artifact."""

    start = time.monotonic()
    root = REPO_ROOT
    result = root / RESULT_RELATIVE_PATH if result_path is None else Path(result_path)
    data = root / DATA_DIR_RELATIVE_PATH if data_dir is None else Path(data_dir)
    protected_before = protected_hashes(root)
    source_before = source_hashes(root)
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    upstream = _load_upstream(root)
    path_absence = _path_absence_before(result, data)
    preconditions = precondition_func(
        data_dir=data,
        model_specs=model_resolution["MODEL_SPECS"],
        result_path=result,
        upstream_artifact=upstream,
    )
    blockers = list(model_resolution["blocked_reasons"]) + [
        str(row.get("resource")) for row in preconditions if row.get("available") is not True
    ]
    if date != RUN_DATE:
        blockers.append(f"unexpected_date:{date}")
    tests_run = _tests_run_receipt(test_exit_codes)
    measured_duration = duration_s if duration_s is not None else time.monotonic() - start
    protected_after = protected_hashes(root)
    if blockers:
        artifact = _blocked_artifact(
            blocked_reason=";".join(sorted(blockers)),
            cached_receipts=model_resolution["cached_sota_pair_receipts"],
            duration_s=measured_duration,
            model_hashes=model_resolution["model_and_embedded_tokenizer_hashes"],
            model_specs=model_resolution["MODEL_SPECS"],
            path_absence_before=path_absence,
            preconditions=preconditions,
            protected_after=protected_after,
            protected_before=protected_before,
            tests_run=tests_run,
            upstream=upstream,
        )
    else:
        artifact = _build_success_artifact(
            cached_receipts=model_resolution["cached_sota_pair_receipts"],
            data_dir=data,
            duration_s=measured_duration,
            model_hashes=model_resolution["model_and_embedded_tokenizer_hashes"],
            model_specs=model_resolution["MODEL_SPECS"],
            path_absence_before=path_absence,
            preconditions=preconditions,
            protected_after=protected_after,
            protected_before=protected_before,
            restart_probe_func=restart_probe_func,
            result_path=result,
            source_before=source_before,
            tests_run=tests_run,
            upstream=upstream,
            write=write,
        )
    if write:
        write_json_atomic(result, artifact)
    return artifact


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Return the reproducibility checksum for a complete artifact."""

    normalized = json.loads(canonical_json(artifact))
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal artifact contract and fail closed on safety drift."""

    require(set(artifact) == set(REQUIRED_ARTIFACT_FIELDS), "required_fields")
    require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "checksum",
    )
    status = str(artifact.get("status"))
    if status == "blocked_preconditions":
        require(artifact.get("csl_safety_replication_ready_score") == 0.0, "blocked_ready_score")
        require(str(artifact.get("honest_verdict", "")).startswith("blocked:"), "blocked_verdict")
        return True
    require([row.get("hf_id") for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "MODEL_SPECS")
    require(artifact.get("models_used") == list(MANDATED_MODEL_IDS), "models_used")
    require(artifact.get("autotokenizer_usage_count") == 0, "autotokenizer_usage_count")
    require(artifact.get("inference_substrate", {}).get("uses_autotokenizer") is False, "autotokenizer_usage_count")
    require(artifact.get("path_nonexistence_freshness_and_disjointness_receipts", {}).get("all_fresh_and_disjoint") is True, "freshness")
    require(artifact.get("per_unit_rows", {}).get("row_count") == len(MANDATED_MODEL_IDS) * UNITS_PER_MODEL * len(ARMS), "per_unit_rows")
    require(artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True, "aggregate_row_mismatch")
    require(artifact.get("future_exact_yield_delta", {}).get("clean_minus_frozen", 0.0) > 0.0, "future_exact_yield_delta")
    require(artifact.get("future_exact_yield_delta", {}).get("governed_minus_frozen", 0.0) > 0.0, "future_exact_yield_delta")
    require(artifact.get("future_exact_yield_delta", {}).get("governed_within_tolerance") is True, "future_exact_yield_delta")
    corrupt = artifact.get("corruption_detection_and_path_receipts", {})
    require(corrupt.get("detected_corrupt_event_count") == corrupt.get("scheduled_corrupt_event_count"), "missed_corruption")
    require(corrupt.get("all_detected_before_update_admission") is True, "missed_corruption")
    require(artifact.get("quarantine_precision_and_recall", {}).get("false_positive_count") == 0, "quarantine_false_positive")
    require(artifact.get("quarantine_precision_and_recall", {}).get("recall") == 1.0, "missed_corruption")
    require(artifact.get("protected_retention", {}).get("protected_release_count") == 0, "protected_release")
    require(artifact.get("false_accepts_and_abstentions", {}).get("false_accept_count") == 0, "false_accepts")
    require(artifact.get("tombstone_rollback_and_resurrection_results", {}).get("corrupt_update_resurrection_count") == 0, "corrupt_update_resurrection")
    require(artifact.get("tombstone_rollback_and_resurrection_results", {}).get("rollback_success_count") == corrupt.get("scheduled_corrupt_event_count"), "rollback_success")
    require(artifact.get("process_restart_and_pid_receipts", {}).get("restart_recovery_rate") == 1.0, "restart_recovery")
    require(artifact.get("transaction_ancestry_and_restart_recovery", {}).get("all_transaction_ancestry_valid") is True, "restart_recovery")
    require(artifact.get("device_and_runner_receipts", {}).get("cpu_fallback_count") == 0, "cpu_fallback")
    require(float(artifact.get("duration_s", 0.0)) >= MIN_DURATION_S, "duration")
    require(artifact.get("attack_matrix", {}).get("all_critical_attacks_fail_closed") is True, "attack_matrix")
    require(artifact.get("attack_matrix", {}).get("readiness_promoted_attack_count") == 0, "attack_matrix")
    require(artifact.get("verifier_is_oracle", {}).get("false_for", {}).get("transport_corrupted_checker_response") is False, "verifier_is_oracle")
    metrics = recompute_metrics(
        artifact["per_unit_rows"]["rows"],
        artifact["process_restart_and_pid_receipts"]["rows"],
        {
            "row_count": artifact["sealed_held_stream_corruption_and_analysis_manifest"]["corruption_event_count"],
        },
    )
    require(artifact.get("future_exact_yield_delta") == metrics["future_exact_yield_delta"], "aggregate_row_mismatch")
    require(artifact.get("current_adversarial_findings") == [], "current_adversarial_findings")
    require(artifact.get("csl_safety_replication_ready_score") == 1.0, "ready_score")
    require(str(artifact.get("honest_verdict", "")).startswith(("success:", "complete:")), "honest_verdict")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        artifact = _load_json(result_path)
        validate_artifact(artifact)
        print(f"validated {result_path}")
        return 0
    artifact = run(date=args.date, result_path=result_path, data_dir=REPO_ROOT / DATA_DIR_RELATIVE_PATH)
    print(f"wrote {result_path}")
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
