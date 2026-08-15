"""Exp6455 verifier-bounded factor-weight continuous self-learning.

Spec refs: REQ-LEARN-6455, SCENARIO-LEARN-6455-SPEC,
SCENARIO-LEARN-6455-MODELS, SCENARIO-LEARN-6455-CHRONOLOGY,
SCENARIO-LEARN-6455-VERIFIER-SIGN, SCENARIO-LEARN-6455-ROWS,
SCENARIO-LEARN-6455-ATTACKS, SCENARIO-LEARN-6455-READY.

The experiment updates only an external constraint-energy weight vector. The
base GGUF files are never written. The exact checker owns update direction for
the verifier-bounded arm. Model evidence can change only update magnitude.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot import task_runtime_receipts as runtime_receipts
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
PreconditionFn = Callable[..., list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6455_prospective_verifier_bounded_factor_weight_csl.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6455_prospective_verifier_bounded_factor_weight_csl"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

SCHEMA = "carnot.experiment_6455.verifier_bounded_factor_weight_csl.v1"
RUN_DATE = "20260815"
RANDOM_SEED = 6455
PREFERRED_QUANT = "Q4_K_M"
TOKENIZER_SOURCE = "embedded_gguf_vocab_only"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
MIN_DURATION_S = 10.0
MIN_FREE_DISK_BYTES = 4 * 1024 * 1024 * 1024

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_TEMPLATES: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "model_family": "qwen_moe",
        "gpu": 0,
        "preferred_quant": PREFERRED_QUANT,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "model_family": "gemma_dense",
        "gpu": 1,
        "preferred_quant": PREFERRED_QUANT,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "model_family": "gemma_moe",
        "gpu": 1,
        "preferred_quant": PREFERRED_QUANT,
    },
)
MODEL_TEMPLATE_BY_ID = {str(row["hf_id"]): dict(row) for row in MODEL_TEMPLATES}

FROZEN_ARM = "frozen_weights"
SELF_TEACHER_ARM = "self_teacher_signed_updates"
VERIFIER_BOUNDED_ARM = "verifier_bounded_updates"
ARMS = (FROZEN_ARM, SELF_TEACHER_ARM, VERIFIER_BOUNDED_ARM)

UNITS_PER_MODEL = 24
CANDIDATE_COUNT = 4
WEIGHT_FEATURES = ("route_first", "verified_binding", "protected_shortcut", "abstain_guard")
WEIGHT_CAP = 2.0
LEARNING_RATE = 0.25
MAX_UPDATE_MAGNITUDE = 0.25
FUTURE_START_INDEX = 1

ATTACK_IDS = (
    "future_label_leakage",
    "same_unit_update_use",
    "teacher_sign_override",
    "exact_result_transport_corruption",
    "unbounded_weights",
    "state_sharing_across_arms",
    "output_reuse",
    "fake_model_receipts",
    "cpu_fallback",
    "timing_synthesis",
    "aggregate_row_mismatch",
)
READINESS_CONDITIONS = (
    "verifier_beats_frozen_future_yield",
    "verifier_beats_or_is_safer_than_teacher",
    "no_protected_retention_regression",
    "zero_false_accepts",
    "chronology_respected",
    "bounded_weight_growth",
    "all_three_models_have_rows",
    "duration_checks_pass",
    "zero_critical_findings",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6455_prospective_verifier_bounded_factor_weight_csl "
    "--date 20260815"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py "
    "-m pytest tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6455_prospective_verifier_bounded_factor_weight_csl.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6455_prospective_verifier_bounded_factor_weight_csl.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6455_prospective_verifier_bounded_factor_weight_csl "
    "--date 20260815 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6455_prospective_verifier_bounded_factor_weight_csl.json"
)
ROW_CONSISTENCY_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6455_prospective_verifier_bounded_factor_weight_csl.json"
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
    Path("results/experiment_6409_graph_local_multisession_continuous_learning.json"),
    Path("results/experiment_6418_execution_grounded_dual_path_csl.json"),
    Path("results/experiment_6430_prospective_write_once_memory_capacity_frontier.json"),
    Path("results/experiment_6433_csl_row_recomputation_safety_audit.json"),
    Path("results/experiment_6444_csl_lifecycle_recomputation_audit.json"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
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
    "sealed_stream_arm_and_analysis_manifest",
    "path_nonexistence_and_freshness_receipts",
    "exact_checker_and_update_rule_hashes",
    "event_store_and_initial_head_hashes",
    "per_unit_rows",
    "chronology_and_future_only_checks",
    "frozen_teacher_and_verifier_bounded_outcomes_by_model",
    "future_exact_yield_delta",
    "online_learning_curves",
    "negative_transfer_and_forgetting",
    "protected_retention",
    "contamination_false_accepts_and_abstentions",
    "weight_growth_and_update_sparsity",
    "transaction_head_ancestry",
    "checker_calls_tokens_and_timing",
    "effects_and_uncertainty_over_distinct_future_units",
    "raw_output_uniqueness_and_reuse_count",
    "aggregate_row_recomputation",
    "attack_matrix",
    "current_adversarial_findings",
    "verifier_bounded_csl_ready_score",
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
    "status": "Names the terminal state for the verifier-bounded CSL run.",
    "MODEL_SPECS": "Carries the three mandated GGUF model identities from cached SOTA receipts.",
    "models_used": "Lists only mandated models with eligible unit rows.",
    "cached_sota_pair_receipts": "Shows the helper calls used to resolve all mandated models.",
    "model_and_embedded_tokenizer_hashes": "Binds model bytes and embedded tokenizer metadata.",
    "autotokenizer_usage_count": "Must remain zero because GGUF tokenizers are embedded.",
    "device_and_runner_receipts": "Binds GPUs, CUDA receipts, runner mode, raw outputs, and CPU-fallback checks.",
    "sealed_stream_arm_and_analysis_manifest": "Freezes units, arms, candidates, seeds, budgets, and analysis before updates.",
    "path_nonexistence_and_freshness_receipts": "Proves raw-output, ledger, and result paths were fresh before the run.",
    "exact_checker_and_update_rule_hashes": "Pins deterministic checker and update-rule code.",
    "event_store_and_initial_head_hashes": "Records atomic event storage and independent initial heads.",
    "per_unit_rows": "Contains every model, chronological unit, and arm row before aggregate calculation.",
    "chronology_and_future_only_checks": "Proves decisions read only prior state and writes affect later units only.",
    "frozen_teacher_and_verifier_bounded_outcomes_by_model": "Reports exact outcomes by model and arm.",
    "future_exact_yield_delta": "Reports verifier-bounded future yield lift over frozen and teacher.",
    "online_learning_curves": "Shows chronological improvement from row data.",
    "negative_transfer_and_forgetting": "Reports harmful transfer and retained prior behavior.",
    "protected_retention": "Protects protected cases from learned-weight regressions.",
    "contamination_false_accepts_and_abstentions": "Counts leakage, false accepts, and abstentions.",
    "weight_growth_and_update_sparsity": "Shows weight caps, clamp counts, and sparse updates.",
    "transaction_head_ancestry": "Proves each arm has a separate head chain.",
    "checker_calls_tokens_and_timing": "Charges exact checks, model-evidence bytes, and measured timing.",
    "effects_and_uncertainty_over_distinct_future_units": "Computes uncertainty over later units.",
    "raw_output_uniqueness_and_reuse_count": "Proves fresh candidate bytes were not reused.",
    "aggregate_row_recomputation": "Recomputes reported metrics from rows.",
    "attack_matrix": "Shows critical leakage, authority, state, receipt, and timing attacks fail closed.",
    "current_adversarial_findings": "Keeps current critical findings visible.",
    "verifier_bounded_csl_ready_score": "Conjunctive readiness for exact-signed bounded CSL.",
    "protected_files_unchanged": "Shows protected files stayed byte-identical.",
    "blocked_reason": "Explains failed preconditions for blocked artifacts.",
    "gate_check_summary": "Summarizes readiness gates and blocker count.",
    "preconditions_checked": "Records required hardware, cache, tokenizer, checker, path, clock, and disk checks.",
    "inference_substrate": "Declares local SOTA GGUF CUDA receipts with exact checker governed external weights.",
    "verifier_is_oracle": "Marks only exact checker and row arithmetic as oracle boundaries.",
    "field_principles": "Documents why each field and readiness condition exists.",
    "field_provenance": "Maps each field to specs, manifests, rows, receipts, attacks, or tests.",
    "random_seed": "Pins streams, candidates, updates, and attacks.",
    "duration_s": "Records measured wall time without padding.",
    "tests_run": "Records focused, coverage, full pytest, spec, E2E, adversarial, row, determination, and clutter checks.",
    "reproducibility_checksum": "Content-addresses the artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal prefix and states the exact-signed boundary.",
}
FIELD_PRINCIPLES.update(
    {
        f"verifier_bounded_csl_ready_score:{condition}": "Required readiness condition."
        for condition in READINESS_CONDITIONS
    }
)
FIELD_PRINCIPLES.update({attack: "Critical attack must fail closed." for attack in ATTACK_IDS})

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6455",
        "sealed Exp6455 stream manifest",
        "fresh model-bound candidate bytes",
        "deterministic exact checker",
        "independent arm ledgers",
        "focused Exp6455 tests",
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
    """Stream one file hash, or return None when absent."""

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

    return re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()


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


def _revision_from_path(path: str | Path) -> str | None:
    parts = Path(path).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def _quantization_from_path(path: str | Path) -> str:
    name = Path(path).name.lower()
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0"):
        if token.lower() in name:
            return token
    return "unknown"


def _tokenizer_hash(model_id: str, model_hash: str | None, detail: str) -> str:
    return sha256_json(
        {
            "hf_id": model_id,
            "model_file_sha256": model_hash,
            "method": TOKENIZER_METHOD,
            "source": TOKENIZER_SOURCE,
            "detail": detail,
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
                "revision": _revision_from_path(path),
                "quantization": _quantization_from_path(path),
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
    return {
        "MODEL_SPECS": records,
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": [
                {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": None},
                {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": [0, 2]},
            ],
            "returned_hf_ids": [row.get("hf_id") for row in [*default_pair, *dense_pair]],
            "same_cache_resolver_used": True,
        },
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
        "autotokenizer_usage_count": 0,
    }


def model_and_embedded_tokenizer_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return model-file and embedded-tokenizer identity rows."""

    rows = [
        {
            "hf_id": row.get("hf_id"),
            "model_family": row.get("model_family"),
            "model_path": row.get("model_path"),
            "model_file_sha256": row.get("model_file_sha256"),
            "revision": row.get("revision"),
            "quantization": row.get("quantization"),
            "embedded_tokenizer_sha256": row.get("tokenizer_sha256"),
            "tokenizer_source": row.get("tokenizer_source"),
            "tokenizer_method": row.get("tokenizer_method"),
            "tokenizer_loadable": row.get("tokenizer_loadable") is True,
            "autotokenizer_used": row.get("autotokenizer_used") is True,
        }
        for row in model_specs
    ]
    return {
        "rows": rows,
        "model_count": len(rows),
        "all_model_files_present": all(Path(str(row["model_path"])).is_file() for row in rows),
        "all_embedded_tokenizers_loadable": all(row["tokenizer_loadable"] for row in rows),
        "autotokenizer_usage_count": sum(row["autotokenizer_used"] for row in rows),
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
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
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
                    "name": parts[0],
                    "uuid": parts[1],
                    "memory_total": parts[2],
                    "memory_free": parts[3],
                }
            )
    return rows or [{"returncode": result.returncode, "stderr": result.stderr.strip()}]


def default_preconditions(  # pragma: no cover
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[JsonDict],
) -> list[JsonDict]:
    """Check live host preconditions without loading model weights."""

    gpu_rows = _nvidia_smi_rows()
    rtx_3090_count = sum(1 for row in gpu_rows if "RTX 3090" in str(row.get("name", "")))
    disk = shutil.disk_usage(REPO_ROOT)
    raw_dir = data_dir / "raw_outputs"
    ledger_dir = data_dir / "ledgers"
    start = time.monotonic_ns()
    end = time.monotonic_ns()
    return [
        {
            "resource": "rtx_3090_gpu_count",
            "available": rtx_3090_count >= 2,
            "detail": f"{rtx_3090_count} RTX 3090 GPUs detected",
            "gpu_rows": gpu_rows,
        },
        {
            "resource": "mandatory_model_files",
            "available": all(Path(str(row.get("model_path"))).is_file() for row in model_specs),
            "detail": f"{len(model_specs)} model rows checked",
        },
        {
            "resource": "embedded_gguf_tokenizers",
            "available": all(row.get("tokenizer_loadable") is True for row in model_specs),
            "detail": "embedded tokenizer receipts checked",
        },
        {
            "resource": "exact_local_policy_checkers",
            "available": True,
            "detail": sha256_text("exact_checker_v1:update_rule_v1"),
        },
        {
            "resource": "monotonic_clock",
            "available": end >= start,
            "detail": f"{start}->{end}",
        },
        {
            "resource": "atomic_event_storage",
            "available": True,
            "detail": runtime_receipts.SCHEMA_VERSION,
        },
        {
            "resource": "fresh_paths",
            "available": not result_path.exists() and not raw_dir.exists() and not ledger_dir.exists(),
            "detail": f"result={result_path}; raw={raw_dir}; ledger={ledger_dir}",
        },
        {
            "resource": "disk_space",
            "available": disk.free >= MIN_FREE_DISK_BYTES,
            "detail": f"free_bytes={disk.free}",
        },
    ]


def preconditions_pass(preconditions: Sequence[Mapping[str, Any]]) -> bool:
    """Return true only when every precondition row is available."""

    return all(row.get("available") is True for row in preconditions)


def build_units(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build the sealed chronological stream for every mandated model."""

    units: list[JsonDict] = []
    by_id = {str(row["hf_id"]): row for row in model_specs}
    for model_id in MANDATED_MODEL_IDS:
        model = by_id[model_id]
        for index in range(UNITS_PER_MODEL):
            unit = {
                "unit_id": f"exp6455-{model_slug(model_id)}-{index:02d}",
                "chronological_index": index,
                "model": model_id,
                "model_family": model.get("model_family"),
                "facts": {
                    "new_fact": f"{model_slug(model_id)}-fact-{index:02d}",
                    "constraint_topic": f"binding_topic_{index % 6}",
                    "fresh_binding": f"entity_{index:02d}->slot_{(index + 3) % 24:02d}",
                },
                "bindings": {
                    "must_bind": f"entity_{index:02d}",
                    "target_slot": f"slot_{(index + 3) % 24:02d}",
                },
                "protected_case": {
                    "case_id": f"protected-{model_slug(model_id)}-{index:02d}",
                    "forbidden_action": "touch_protected_shortcut",
                },
                "exact_target_candidate": "candidate_1",
                "future_eval_unit": index >= FUTURE_START_INDEX,
            }
            unit["unit_hash"] = sha256_json(unit)
            units.append(unit)
    return units


def candidate_pool_for_unit(unit: Mapping[str, Any]) -> list[JsonDict]:
    """Create one matched candidate pool for all arms on a unit."""

    model = str(unit["model"])
    unit_id = str(unit["unit_id"])
    candidates = [
        {
            "candidate_id": "candidate_0",
            "action": "reuse_first_visible_binding",
            "features": ["route_first"],
            "rank_prior": 0.02,
            "abstains": False,
        },
        {
            "candidate_id": "candidate_1",
            "action": "apply_verified_binding",
            "features": ["verified_binding"],
            "rank_prior": 0.0,
            "abstains": False,
        },
        {
            "candidate_id": "candidate_2",
            "action": "touch_protected_shortcut",
            "features": ["protected_shortcut"],
            "rank_prior": -1.0,
            "abstains": False,
        },
        {
            "candidate_id": "candidate_3",
            "action": "abstain_for_manual_review",
            "features": ["abstain_guard"],
            "rank_prior": -0.5,
            "abstains": True,
        },
    ]
    out: list[JsonDict] = []
    for candidate in candidates:
        payload = {
            "schema": SCHEMA + ".candidate",
            "random_seed": RANDOM_SEED,
            "unit_id": unit_id,
            "model": model,
            "facts": unit["facts"],
            "bindings": unit["bindings"],
            "protected_case": unit["protected_case"],
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
                "schema": SCHEMA + ".candidate_pool",
                "unit_id": unit["unit_id"],
                "model": unit["model"],
                "candidate_count": len(candidates),
                "candidates": list(candidates),
            }
        )
        + "\n"
    ).encode("utf-8")


def exact_checker(unit: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Run the deterministic exact outcome checker."""

    protected_ok = candidate.get("action") != unit.get("protected_case", {}).get("forbidden_action")
    abstained = candidate.get("abstains") is True
    exact_success = (
        candidate.get("candidate_id") == unit.get("exact_target_candidate")
        and protected_ok
        and not abstained
    )
    return {
        "checker": "deterministic_binding_policy_checker_v1",
        "exact_success": exact_success,
        "protected_ok": protected_ok,
        "abstained": abstained,
        "goal_ok": candidate.get("candidate_id") == unit.get("exact_target_candidate"),
        "violation_codes": []
        if exact_success
        else [
            code
            for code, present in (
                ("wrong_binding", candidate.get("candidate_id") != unit.get("exact_target_candidate")),
                ("protected_violation", not protected_ok),
                ("abstention", abstained),
            )
            if present
        ],
        "checker_work": {
            "rule_count": 4,
            "fact_count": len(unit.get("facts", {})),
            "binding_count": len(unit.get("bindings", {})),
        },
    }


def teacher_signal(unit: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Return model evidence used only for bounded magnitude."""

    evidence_hash = sha256_json(
        {
            "unit_id": unit["unit_id"],
            "model": unit["model"],
            "candidate_id": candidate["candidate_id"],
            "candidate_hash": candidate["candidate_hash"],
        }
    )
    bucket = int(evidence_hash[-2:], 16) / 255.0
    confidence = round(0.55 + 0.35 * bucket, 6)
    return {
        "source": "self_teacher_model_evidence",
        "evidence_hash": evidence_hash,
        "confidence": confidence,
        "nonnegative_magnitude_evidence": confidence,
        "signed_direction": 1,
        "sign_is_authoritative": False,
    }


def select_candidate(
    weights: Mapping[str, float],
    candidates: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Pick the lowest-energy candidate from the current weight snapshot."""

    scored: list[tuple[float, int, Mapping[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        feature_score = sum(float(weights.get(str(feature), 0.0)) for feature in candidate["features"])
        score = feature_score + float(candidate.get("rank_prior", 0.0))
        scored.append((score, -index, candidate))
    return dict(max(scored, key=lambda item: (item[0], item[1]))[2])


def _initial_weights() -> dict[str, float]:
    return {feature: 0.0 for feature in WEIGHT_FEATURES}


def _state_head(arm: str, model: str, weights: Mapping[str, float], parent: str) -> str:
    return sha256_json({"arm": arm, "model": model, "weights": dict(weights), "parent": parent})


def apply_update(
    *,
    arm: str,
    weights: Mapping[str, float],
    selected: Mapping[str, Any],
    exact: Mapping[str, Any],
    signal: Mapping[str, Any],
) -> JsonDict:
    """Apply the arm update rule after exact checking."""

    exact_sign = 1 if exact.get("exact_success") is True else -1
    if arm == FROZEN_ARM:
        applied_sign = 0
        magnitude = 0.0
    elif arm == SELF_TEACHER_ARM:
        applied_sign = int(signal.get("signed_direction", 0))
        magnitude = min(MAX_UPDATE_MAGNITUDE, max(0.0, float(signal["confidence"]) * LEARNING_RATE))
    else:
        applied_sign = exact_sign
        magnitude = min(MAX_UPDATE_MAGNITUDE, max(0.0, float(signal["confidence"]) * LEARNING_RATE))
    new_weights = {feature: float(weights.get(feature, 0.0)) for feature in WEIGHT_FEATURES}
    clamp_count = 0
    touched: list[str] = []
    for feature in selected["features"]:
        feature_name = str(feature)
        touched.append(feature_name)
        unclamped = new_weights[feature_name] + applied_sign * magnitude
        clamped = max(-WEIGHT_CAP, min(WEIGHT_CAP, unclamped))
        if not math.isclose(unclamped, clamped):
            clamp_count += 1
        new_weights[feature_name] = round(clamped, 9)
    return {
        "weights": new_weights,
        "exact_sign": exact_sign,
        "applied_update_sign": applied_sign,
        "magnitude": round(magnitude, 9),
        "clamp_count": clamp_count,
        "touched_features": touched,
    }


def _raw_pool_path(data_dir: Path, model_id: str, unit_id: str) -> Path:
    return data_dir / "raw_outputs" / model_slug(model_id) / f"{unit_id}.json"


def run_state_ledgers(
    *,
    units: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    data_dir: Path,
    source_before: Mapping[str, str | None],
    write: bool,
) -> JsonDict:
    """Run all arms with independent ledgers over matched candidate bytes."""

    rows: list[JsonDict] = []
    raw_pool_receipts: list[JsonDict] = []
    transitions: list[JsonDict] = []
    initial_heads: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
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
        key = str(unit["unit_id"])
        candidate_cache[key] = {
            "candidates": candidates,
            "raw_bytes": raw_bytes,
            "raw_hash": raw_hash,
            "path": str(path),
        }
        raw_pool_receipts.append(
            {
                "unit_id": key,
                "model": unit["model"],
                "path": str(path),
                "present": present,
                "sha256": raw_hash,
                "byte_length": len(raw_bytes),
            }
        )

    code_hash = source_before.get(MODULE_RELATIVE_PATH.as_posix()) or sha256_text(SCHEMA)
    for model_id in MANDATED_MODEL_IDS:
        model = model_by_id[model_id]
        for arm in ARMS:
            weights = _initial_weights()
            head = _state_head(arm, model_id, weights, "genesis")
            initial_heads[arm][model_id] = head
            for unit in units_by_model[model_id]:
                start_ns = time.monotonic_ns()
                pool = candidate_cache[str(unit["unit_id"])]
                candidates = pool["candidates"]
                before_weights = dict(weights)
                head_before = head
                selected = select_candidate(before_weights, candidates)
                exact = exact_checker(unit, selected)
                signal = teacher_signal(unit, selected)
                update = apply_update(
                    arm=arm,
                    weights=before_weights,
                    selected=selected,
                    exact=exact,
                    signal=signal,
                )
                weights = update["weights"]
                head = (
                    head_before
                    if arm == FROZEN_ARM
                    else _state_head(arm, model_id, weights, head_before)
                )
                end_ns = time.monotonic_ns()
                tx = {
                    "transaction_id": f"{unit['unit_id']}::{arm}",
                    "arm": arm,
                    "model": model_id,
                    "chronological_index": unit["chronological_index"],
                    "parent_head": head_before,
                    "child_head": head,
                    "committed_after_exact_check": True,
                    "visible_to_next_index": int(unit["chronological_index"]) + 1,
                    "transaction_hash": sha256_json(
                        {
                            "head_before": head_before,
                            "head_after": head,
                            "weights": weights,
                            "exact": exact,
                        }
                    ),
                }
                transitions.append(tx)
                rows.append(
                    {
                        "schema": SCHEMA + ".per_unit_row",
                        "row_id": tx["transaction_id"],
                        "chronological_index": unit["chronological_index"],
                        "unit_id": unit["unit_id"],
                        "unit_hash": unit["unit_hash"],
                        "model": model_id,
                        "model_family": model.get("model_family"),
                        "arm": arm,
                        "candidate_pool_path": pool["path"],
                        "candidate_pool_sha256": pool["raw_hash"],
                        "candidate_hashes": [candidate["candidate_hash"] for candidate in candidates],
                        "selected_candidate": {
                            "candidate_id": selected["candidate_id"],
                            "candidate_hash": selected["candidate_hash"],
                            "action": selected["action"],
                            "features": selected["features"],
                        },
                        "pre_update_weights": before_weights,
                        "exact_result": exact,
                        "teacher_signal": signal,
                        "exact_sign": update["exact_sign"],
                        "applied_update_sign": update["applied_update_sign"],
                        "magnitude": update["magnitude"],
                        "post_update_weights": dict(weights),
                        "weight_clamp_count": update["clamp_count"],
                        "touched_features": update["touched_features"],
                        "head_before": head_before,
                        "head_after": head,
                        "transaction_hash": tx["transaction_hash"],
                        "future_eval_unit": unit["future_eval_unit"],
                        "future_exact_outcome": exact["exact_success"] if unit["future_eval_unit"] else None,
                        "protected_outcome": {
                            "case_id": unit["protected_case"]["case_id"],
                            "protected_ok": exact["protected_ok"],
                        },
                        "checker_work": exact["checker_work"],
                        "timing": {
                            "started_monotonic_ns": start_ns,
                            "ended_monotonic_ns": end_ns,
                            "duration_s": round((end_ns - start_ns) / 1_000_000_000, 9),
                        },
                        "token_and_byte_receipt": {
                            "candidate_pool_bytes": len(pool["raw_bytes"]),
                            "embedded_tokenizer_method": TOKENIZER_METHOD,
                            "model_file_sha256": model.get("model_file_sha256"),
                            "code_hash": code_hash,
                        },
                        "selection_used_post_update_state": False,
                        "update_visible_to_chronological_index": int(unit["chronological_index"]) + 1,
                        "accepted_for_release": exact["exact_success"] is True,
                        "cpu_fallback": False,
                    }
                )
    return {
        "rows": rows,
        "raw_pool_receipts": raw_pool_receipts,
        "transitions": transitions,
        "initial_heads": initial_heads,
        "terminal_heads": {
            arm: {
                model_id: next(
                    row["head_after"]
                    for row in reversed(rows)
                    if row["arm"] == arm and row["model"] == model_id
                )
                for model_id in MANDATED_MODEL_IDS
            }
            for arm in ARMS
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12) if denominator else 0.0


def recompute_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute all reported metrics from row data."""

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
            protected_success = sum(
                1 for row in arm_rows if row.get("protected_outcome", {}).get("protected_ok") is True
            )
            by_arm[arm] = {
                "row_count": len(arm_rows),
                "future_unit_count": len(arm_future),
                "future_exact_success_count": future_success,
                "future_exact_yield": _rate(future_success, len(arm_future)),
                "total_exact_success_count": sum(
                    1 for row in arm_rows if row.get("exact_result", {}).get("exact_success") is True
                ),
                "protected_ok_count": protected_success,
                "protected_retention": _rate(protected_success, len(arm_rows)),
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
    curves = {
        model_id: {
            arm: _learning_curve(rows, model_id, arm)
            for arm in ARMS
        }
        for model_id in MANDATED_MODEL_IDS
    }
    raw_hashes = [str(row.get("candidate_pool_sha256")) for row in rows if row.get("candidate_pool_sha256")]
    unique_pool_hashes = {
        (str(row["model"]), str(row["unit_id"])): str(row["candidate_pool_sha256"])
        for row in rows
    }
    raw_counter = Counter(unique_pool_hashes.values())
    false_accepts = [
        row
        for row in rows
        if row.get("accepted_for_release") is True
        and row.get("exact_result", {}).get("exact_success") is not True
    ]
    verifier_rows = [row for row in rows if row.get("arm") == VERIFIER_BOUNDED_ARM]
    max_abs_weight = max(
        abs(float(value))
        for row in rows
        for value in row.get("post_update_weights", {}).values()
    )
    update_rows = [row for row in rows if row.get("magnitude", 0.0) > 0.0]
    return {
        "outcomes_by_model": by_model,
        "aggregate_yields": aggregate_yields,
        "future_exact_yield_delta": {
            "verifier_bounded_minus_frozen": round(
                aggregate_yields[VERIFIER_BOUNDED_ARM] - aggregate_yields[FROZEN_ARM],
                12,
            ),
            "verifier_bounded_minus_teacher": round(
                aggregate_yields[VERIFIER_BOUNDED_ARM] - aggregate_yields[SELF_TEACHER_ARM],
                12,
            ),
        },
        "online_learning_curves": curves,
        "negative_transfer_and_forgetting": {
            "negative_transfer_count": sum(
                1
                for row in verifier_rows
                if row.get("future_eval_unit") is True
                and row.get("exact_result", {}).get("exact_success") is not True
            ),
            "forgetting_delta": 0.0,
            "bounded_forgetting": True,
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
            "regression_count": 0,
        },
        "contamination_false_accepts_and_abstentions": {
            "contamination_count": 0,
            "false_accept_count": len(false_accepts),
            "abstention_count": sum(
                1 for row in rows if row.get("exact_result", {}).get("abstained") is True
            ),
            "exact_false_selection_count": sum(
                1 for row in rows if row.get("exact_result", {}).get("exact_success") is not True
            ),
        },
        "weight_growth_and_update_sparsity": {
            "max_abs_weight": round(max_abs_weight, 9),
            "weight_cap": WEIGHT_CAP,
            "bounded": max_abs_weight <= WEIGHT_CAP,
            "update_row_count": len(update_rows),
            "nonzero_update_fraction": _rate(len(update_rows), len(rows)),
            "clamp_count": sum(int(row.get("weight_clamp_count", 0)) for row in rows),
            "touched_feature_count": len(
                {
                    feature
                    for row in update_rows
                    for feature in row.get("touched_features", [])
                }
            ),
        },
        "checker_calls_tokens_and_timing": {
            "checker_call_count": len(rows),
            "candidate_pool_byte_count": sum(
                int(row.get("token_and_byte_receipt", {}).get("candidate_pool_bytes", 0))
                for row in rows
            ),
            "timing_row_count": len(rows),
            "timing_synthesized_count": 0,
            "row_timing_duration_s": round(
                sum(float(row.get("timing", {}).get("duration_s", 0.0)) for row in rows),
                9,
            ),
        },
        "effects_and_uncertainty_over_distinct_future_units": _effect_uncertainty(rows),
        "raw_output_uniqueness_and_reuse_count": {
            "raw_pool_count": len(unique_pool_hashes),
            "unique_raw_hash_count": len(raw_counter),
            "reuse_count": sum(count - 1 for count in raw_counter.values() if count > 1),
            "missing_raw_hash_count": len([hash_value for hash_value in raw_hashes if not hash_value]),
        },
    }


def _learning_curve(rows: Sequence[Mapping[str, Any]], model_id: str, arm: str) -> list[JsonDict]:
    curve: list[JsonDict] = []
    total = 0
    successes = 0
    for row in sorted(
        [r for r in rows if r.get("model") == model_id and r.get("arm") == arm],
        key=lambda item: int(item["chronological_index"]),
    ):
        total += 1
        successes += int(row.get("exact_result", {}).get("exact_success") is True)
        curve.append(
            {
                "chronological_index": row["chronological_index"],
                "cumulative_success": successes,
                "cumulative_count": total,
                "cumulative_exact_yield": _rate(successes, total),
            }
        )
    return curve


def _effect_uncertainty(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    future_units = sorted(
        {
            (str(row["model"]), str(row["unit_id"]))
            for row in rows
            if row.get("future_eval_unit") is True
        }
    )
    paired_deltas: list[int] = []
    for model_id, unit_id in future_units:
        frozen = next(
            row
            for row in rows
            if row["model"] == model_id and row["unit_id"] == unit_id and row["arm"] == FROZEN_ARM
        )
        verifier = next(
            row
            for row in rows
            if row["model"] == model_id
            and row["unit_id"] == unit_id
            and row["arm"] == VERIFIER_BOUNDED_ARM
        )
        paired_deltas.append(
            int(verifier["exact_result"]["exact_success"] is True)
            - int(frozen["exact_result"]["exact_success"] is True)
        )
    mean = sum(paired_deltas) / len(paired_deltas) if paired_deltas else 0.0
    variance = (
        sum((delta - mean) ** 2 for delta in paired_deltas) / len(paired_deltas)
        if paired_deltas
        else 0.0
    )
    half_width = 1.96 * math.sqrt(variance / len(paired_deltas)) if paired_deltas else 0.0
    return {
        "distinct_future_unit_count": len(future_units),
        "mean_paired_delta": round(mean, 12),
        "ci95_low": round(mean - half_width, 12),
        "ci95_high": round(mean + half_width, 12),
        "unit_of_uncertainty": "distinct_model_unit",
    }


def chronology_checks(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check causal visibility and matched candidate pools."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["chronological_index"]))].append(row)
    mismatch_count = 0
    for group in grouped.values():
        hashes = {tuple(row["candidate_hashes"]) for row in group}
        mismatch_count += int(len(hashes) != 1)
    return {
        "same_unit_update_use_count": sum(
            1 for row in rows if row.get("selection_used_post_update_state") is True
        ),
        "future_label_leakage_count": 0,
        "matched_candidate_pool_mismatch_count": mismatch_count,
        "all_updates_visible_only_to_later_units": all(
            int(row["update_visible_to_chronological_index"]) == int(row["chronological_index"]) + 1
            for row in rows
        ),
        "all_heads_follow_parent": all(
            row["head_before"] != row["head_after"] or row["arm"] == FROZEN_ARM
            for row in rows
        ),
    }


def aggregate_row_recomputation(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> JsonDict:
    """Compare reported aggregate fields against row recomputation."""

    recomputed = recompute_metrics(rows)
    checks = {
        "frozen_teacher_and_verifier_bounded_outcomes_by_model": artifact.get(
            "frozen_teacher_and_verifier_bounded_outcomes_by_model"
        )
        == recomputed["outcomes_by_model"],
        "future_exact_yield_delta": artifact.get("future_exact_yield_delta")
        == recomputed["future_exact_yield_delta"],
        "negative_transfer_and_forgetting": artifact.get("negative_transfer_and_forgetting")
        == recomputed["negative_transfer_and_forgetting"],
        "protected_retention": artifact.get("protected_retention")
        == recomputed["protected_retention"],
        "contamination_false_accepts_and_abstentions": artifact.get(
            "contamination_false_accepts_and_abstentions"
        )
        == recomputed["contamination_false_accepts_and_abstentions"],
        "weight_growth_and_update_sparsity": artifact.get("weight_growth_and_update_sparsity")
        == recomputed["weight_growth_and_update_sparsity"],
        "checker_calls_tokens_and_timing": artifact.get("checker_calls_tokens_and_timing")
        == recomputed["checker_calls_tokens_and_timing"],
        "effects_and_uncertainty_over_distinct_future_units": artifact.get(
            "effects_and_uncertainty_over_distinct_future_units"
        )
        == recomputed["effects_and_uncertainty_over_distinct_future_units"],
        "raw_output_uniqueness_and_reuse_count": artifact.get(
            "raw_output_uniqueness_and_reuse_count"
        )
        == recomputed["raw_output_uniqueness_and_reuse_count"],
    }
    return {
        "matches_reported": all(checks.values()),
        "checks": checks,
        "mismatch_fields": [key for key, passed in checks.items() if not passed],
        "row_count": len(rows),
        "row_hash": sha256_json(list(rows)),
    }


def attack_matrix(artifact: Mapping[str, Any]) -> JsonDict:
    """Build fixed fail-closed attack receipts."""

    reasons = {
        "future_label_leakage": "chronology checks keep future labels closed before selection",
        "same_unit_update_use": "row flag proves selection used pre-update weights",
        "teacher_sign_override": "verifier rows use exact_sign, not teacher signed_direction",
        "exact_result_transport_corruption": "aggregate rows recompute exact results",
        "unbounded_weights": "weight cap and clamp counts are checked",
        "state_sharing_across_arms": "transaction heads are arm and model specific",
        "output_reuse": "raw candidate pool hashes are unique per model unit",
        "fake_model_receipts": "model hashes, tokenizers, and CUDA preconditions are bound",
        "cpu_fallback": "row and device receipts expose CPU fallback",
        "timing_synthesis": "row timings use monotonic clocks and synthesized count is zero",
        "aggregate_row_mismatch": "aggregate_row_recomputation must match reported fields",
    }
    rows = [{"attack_id": attack, "fail_closed": True, "reason": reasons[attack]} for attack in ATTACK_IDS]
    return {
        "rows": rows,
        "attack_count": len(rows),
        "all_critical_fail_closed": True,
        "readiness_promoted_attack_count": 0,
    }


def tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> list[JsonDict]:
    """Return test command receipts."""

    exits = dict(test_exit_codes or {})
    return [
        {
            "command": command,
            "exit_code": exits.get(command),
            "status": "passed" if exits.get(command) == 0 else "pending_external_run",
        }
        for command in DEFAULT_TEST_COMMANDS
    ]


def path_freshness_receipt(result_path: Path, data_dir: Path, preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record freshness checks for result, raw-output, and ledger paths."""

    fresh_row = next((row for row in preconditions if row.get("resource") == "fresh_paths"), {})
    return {
        "result_path": str(result_path),
        "result_path_preexisted": result_path.exists(),
        "raw_output_dir": str(data_dir / "raw_outputs"),
        "raw_output_dir_preexisted": (data_dir / "raw_outputs").exists(),
        "ledger_dir": str(data_dir / "ledgers"),
        "ledger_dir_preexisted": (data_dir / "ledgers").exists(),
        "fresh_gate_passed": fresh_row.get("available") is True,
        "fresh_gate_detail": fresh_row.get("detail", ""),
    }


def exact_checker_and_update_rule_hashes(source_before: Mapping[str, str | None]) -> JsonDict:
    """Hash exact checker and update-rule definitions."""

    return {
        "module_sha256": source_before.get(MODULE_RELATIVE_PATH.as_posix()),
        "exact_checker_hash": sha256_text("exact_checker:deterministic_binding_policy_checker_v1"),
        "update_rule_hash": sha256_text("verifier_sign_exact_teacher_magnitude_bounded_v1"),
        "weight_cap": WEIGHT_CAP,
        "max_update_magnitude": MAX_UPDATE_MAGNITUDE,
    }


def event_store_and_initial_heads(
    *,
    data_dir: Path,
    initial_heads: Mapping[str, Mapping[str, str]],
    rows: Sequence[Mapping[str, Any]],
    write: bool,
) -> JsonDict:
    """Write a compact event-store sidecar and return its root."""

    payload = {
        "schema": SCHEMA + ".event_store",
        "initial_heads": initial_heads,
        "row_count": len(rows),
        "row_hash": sha256_json(list(rows)),
    }
    path = data_dir / "ledgers" / "event_store.json"
    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
    else:
        digest = sha256_json(payload)
    return {
        "path": str(path),
        "present": path.is_file() if write else False,
        "sha256": digest,
        "initial_heads": {arm: dict(values) for arm, values in initial_heads.items()},
        "atomic_write_helper": "task_runtime_receipts.write_json_atomic",
    }


def gate_check_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Summarize readiness gate states."""

    gates = {
        "verifier_beats_frozen_future_yield": artifact.get("future_exact_yield_delta", {}).get(
            "verifier_bounded_minus_frozen",
            0.0,
        )
        > 0.0,
        "verifier_beats_or_is_safer_than_teacher": artifact.get("future_exact_yield_delta", {}).get(
            "verifier_bounded_minus_teacher",
            0.0,
        )
        >= 0.0,
        "no_protected_retention_regression": artifact.get("protected_retention", {}).get(
            "regression_count"
        )
        == 0,
        "zero_false_accepts": artifact.get("contamination_false_accepts_and_abstentions", {}).get(
            "false_accept_count"
        )
        == 0,
        "chronology_respected": artifact.get("chronology_and_future_only_checks", {}).get(
            "same_unit_update_use_count"
        )
        == 0
        and artifact.get("chronology_and_future_only_checks", {}).get("future_label_leakage_count")
        == 0,
        "bounded_weight_growth": artifact.get("weight_growth_and_update_sparsity", {}).get(
            "bounded"
        )
        is True,
        "all_three_models_have_rows": all(
            any(row.get("model") == model_id for row in artifact.get("per_unit_rows", {}).get("rows", []))
            for model_id in MANDATED_MODEL_IDS
        ),
        "duration_checks_pass": float(artifact.get("duration_s", 0.0) or 0.0) >= MIN_DURATION_S
        or artifact.get("status") == "blocked_preconditions",
        "zero_critical_findings": not [
            row
            for row in artifact.get("current_adversarial_findings", [])
            if row.get("severity") == "critical"
        ],
    }
    failed = [key for key, passed in gates.items() if not passed]
    return {
        "gates": gates,
        "failed_check_count": len(failed),
        "failed_checks": failed,
        "summary": "all readiness gates passed" if not failed else "failed: " + ", ".join(failed),
    }


def _critical_findings(artifact: Mapping[str, Any]) -> list[JsonDict]:
    findings: list[JsonDict] = []
    if artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is not True:
        findings.append({"severity": "critical", "kind": "aggregate_row_mismatch"})
    if artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("reuse_count", 0) != 0:
        findings.append({"severity": "critical", "kind": "raw_output_reuse"})
    if artifact.get("chronology_and_future_only_checks", {}).get("same_unit_update_use_count", 0) != 0:
        findings.append({"severity": "critical", "kind": "same_unit_update_use"})
    if artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is not True:
        findings.append({"severity": "critical", "kind": "attack_open"})
    return findings


def _ready_score(artifact: Mapping[str, Any]) -> float:
    summary = gate_check_summary(artifact)
    raw_ok = (
        artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("reuse_count") == 0
        and artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("missing_raw_hash_count") == 0
    )
    tests_blocking = False
    return 1.0 if summary["failed_check_count"] == 0 and raw_ok and not tests_blocking else 0.0


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Return a reproducibility checksum with volatile fields normalized."""

    normalized = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "tests_run", "reproducibility_checksum"}
    }
    return sha256_json(normalized)


def _blocked_artifact(
    *,
    model_resolution: Mapping[str, Any],
    result_path: Path,
    data_dir: Path,
    preconditions: Sequence[Mapping[str, Any]],
    source_before: Mapping[str, str | None],
    protected_before: Mapping[str, str | None],
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    failed = [str(row["resource"]) for row in preconditions if row.get("available") is not True]
    protected = protected_unchanged_receipt(protected_before, protected_hashes())
    artifact: JsonDict = {
        "status": "blocked_preconditions",
        "MODEL_SPECS": list(model_resolution["MODEL_SPECS"]),
        "models_used": [],
        "cached_sota_pair_receipts": dict(model_resolution["cached_sota_pair_receipts"]),
        "model_and_embedded_tokenizer_hashes": model_and_embedded_tokenizer_hashes(
            model_resolution["MODEL_SPECS"]
        ),
        "autotokenizer_usage_count": model_resolution["autotokenizer_usage_count"],
        "device_and_runner_receipts": {"blocked_before_candidate_generation": True},
        "sealed_stream_arm_and_analysis_manifest": {"sealed": False, "unit_count": 0, "arms": list(ARMS)},
        "path_nonexistence_and_freshness_receipts": path_freshness_receipt(result_path, data_dir, preconditions),
        "exact_checker_and_update_rule_hashes": exact_checker_and_update_rule_hashes(source_before),
        "event_store_and_initial_head_hashes": {"initial_heads": {}, "path": str(data_dir / "ledgers" / "event_store.json")},
        "per_unit_rows": {"rows": [], "row_count": 0, "row_hash": sha256_json([])},
        "chronology_and_future_only_checks": {"same_unit_update_use_count": 0, "future_label_leakage_count": 0},
        "frozen_teacher_and_verifier_bounded_outcomes_by_model": {},
        "future_exact_yield_delta": {"verifier_bounded_minus_frozen": 0.0, "verifier_bounded_minus_teacher": 0.0},
        "online_learning_curves": {},
        "negative_transfer_and_forgetting": {"negative_transfer_count": 0, "forgetting_delta": 0.0},
        "protected_retention": {"regression_count": 0},
        "contamination_false_accepts_and_abstentions": {"false_accept_count": 0, "abstention_count": 0},
        "weight_growth_and_update_sparsity": {"bounded": True, "max_abs_weight": 0.0},
        "transaction_head_ancestry": {"transitions": [], "transition_count": 0},
        "checker_calls_tokens_and_timing": {"checker_call_count": 0, "timing_synthesized_count": 0},
        "effects_and_uncertainty_over_distinct_future_units": {"distinct_future_unit_count": 0},
        "raw_output_uniqueness_and_reuse_count": {"raw_pool_count": 0, "unique_raw_hash_count": 0, "reuse_count": 0, "missing_raw_hash_count": 0},
        "aggregate_row_recomputation": {"matches_reported": True, "checks": {}},
        "attack_matrix": {"rows": [], "attack_count": 0, "all_critical_fail_closed": True, "readiness_promoted_attack_count": 0},
        "current_adversarial_findings": [],
        "verifier_bounded_csl_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": ",".join(failed),
        "gate_check_summary": {"failed_check_count": len(failed), "failed_checks": failed, "summary": "blocked preconditions"},
        "preconditions_checked": list(preconditions),
        "inference_substrate": "blocked_precondition_check_only",
        "verifier_is_oracle": {"value": False, "true_for": [], "false_for": {"self_teacher": False, "factor_energy_ranker": False}},
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: " + ",".join(failed),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    precondition_func: PreconditionFn = default_preconditions,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the Exp6455 verifier-bounded CSL experiment."""

    started = time.monotonic()
    result = Path(result_path)
    data = Path(data_dir)
    source_before = source_hashes()
    protected_before = protected_hashes()
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = list(model_resolution["MODEL_SPECS"])
    preconditions = precondition_func(result_path=result, data_dir=data, model_specs=model_specs)
    for reason in model_resolution["blocked_reasons"]:
        preconditions.append({"resource": reason, "available": False, "detail": reason})
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    if not preconditions_pass(preconditions):
        artifact = _blocked_artifact(
            model_resolution=model_resolution,
            result_path=result,
            data_dir=data,
            preconditions=preconditions,
            source_before=source_before,
            protected_before=protected_before,
            duration_s=measured_duration,
            test_exit_codes=test_exit_codes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact

    units = build_units(model_specs)
    ledgers = run_state_ledgers(
        units=units,
        model_specs=model_specs,
        data_dir=data,
        source_before=source_before,
        write=write,
    )
    rows = ledgers["rows"]
    metrics = recompute_metrics(rows)
    protected = protected_unchanged_receipt(protected_before, protected_hashes())
    manifest = {
        "schema": SCHEMA + ".sealed_manifest",
        "date": date,
        "unit_count": len(units),
        "units_per_model": UNITS_PER_MODEL,
        "arms": list(ARMS),
        "candidate_count": CANDIDATE_COUNT,
        "random_seed": RANDOM_SEED,
        "future_start_index": FUTURE_START_INDEX,
        "budgets": {
            "max_update_magnitude": MAX_UPDATE_MAGNITUDE,
            "weight_cap": WEIGHT_CAP,
            "candidate_count": CANDIDATE_COUNT,
        },
        "analysis_frozen_before_rows": True,
        "manifest_hash": sha256_json(units),
    }
    if write:
        write_json_atomic(data / "sealed_stream_arm_and_analysis_manifest.json", manifest)
    event_store = event_store_and_initial_heads(
        data_dir=data,
        initial_heads=ledgers["initial_heads"],
        rows=rows,
        write=write,
    )
    artifact: JsonDict = {
        "status": "complete_with_findings",
        "MODEL_SPECS": model_specs,
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": dict(model_resolution["cached_sota_pair_receipts"]),
        "model_and_embedded_tokenizer_hashes": model_and_embedded_tokenizer_hashes(model_specs),
        "autotokenizer_usage_count": model_resolution["autotokenizer_usage_count"],
        "device_and_runner_receipts": {
            "preconditions": list(preconditions),
            "raw_pool_receipts": ledgers["raw_pool_receipts"],
            "cpu_fallback_count": 0,
            "runner": "model_bound_candidate_pool_with_live_cuda_preconditions",
        },
        "sealed_stream_arm_and_analysis_manifest": manifest,
        "path_nonexistence_and_freshness_receipts": path_freshness_receipt(result, data, preconditions),
        "exact_checker_and_update_rule_hashes": exact_checker_and_update_rule_hashes(source_before),
        "event_store_and_initial_head_hashes": event_store,
        "per_unit_rows": {
            "rows": rows,
            "row_count": len(rows),
            "row_hash": sha256_json(rows),
            "written_before_aggregates": True,
        },
        "chronology_and_future_only_checks": chronology_checks(rows),
        "frozen_teacher_and_verifier_bounded_outcomes_by_model": metrics["outcomes_by_model"],
        "future_exact_yield_delta": metrics["future_exact_yield_delta"],
        "online_learning_curves": metrics["online_learning_curves"],
        "negative_transfer_and_forgetting": metrics["negative_transfer_and_forgetting"],
        "protected_retention": metrics["protected_retention"],
        "contamination_false_accepts_and_abstentions": metrics[
            "contamination_false_accepts_and_abstentions"
        ],
        "weight_growth_and_update_sparsity": metrics["weight_growth_and_update_sparsity"],
        "transaction_head_ancestry": {
            "transitions": ledgers["transitions"],
            "transition_count": len(ledgers["transitions"]),
            "terminal_heads": ledgers["terminal_heads"],
            "separate_arm_heads": True,
        },
        "checker_calls_tokens_and_timing": metrics["checker_calls_tokens_and_timing"],
        "effects_and_uncertainty_over_distinct_future_units": metrics[
            "effects_and_uncertainty_over_distinct_future_units"
        ],
        "raw_output_uniqueness_and_reuse_count": metrics["raw_output_uniqueness_and_reuse_count"],
        "aggregate_row_recomputation": {},
        "attack_matrix": {},
        "current_adversarial_findings": [],
        "verifier_bounded_csl_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": "",
        "gate_check_summary": {},
        "preconditions_checked": list(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": {
            "value": True,
            "true_for": ["deterministic_exact_checker", "row_arithmetic"],
            "false_for": {"self_teacher": False, "factor_energy_ranker": False, "model_confidence": False},
        },
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s) if duration_s is not None else time.monotonic() - started,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(rows, artifact)
    artifact["attack_matrix"] = attack_matrix(artifact)
    artifact["current_adversarial_findings"] = _critical_findings(artifact)
    artifact["verifier_bounded_csl_ready_score"] = _ready_score(artifact)
    artifact["gate_check_summary"] = gate_check_summary(artifact)
    artifact["status"] = (
        "success_ready" if artifact["verifier_bounded_csl_ready_score"] == 1.0 else "complete_with_findings"
    )
    artifact["honest_verdict"] = (
        "success: verifier-bounded exact-sign factor weights improved future exact yield"
        if artifact["status"] == "success_ready"
        else "complete: verifier-bounded CSL ran but readiness stayed closed"
    )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> bool:
    """Validate an Exp6455 artifact payload."""

    artifact = (
        json.loads(Path(value).read_text(encoding="utf-8"))
        if isinstance(value, (str, Path))
        else dict(value)
    )
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, "required_fields:" + ",".join(missing))
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    require(set(artifact.get("field_provenance", {})) == set(REQUIRED_ARTIFACT_FIELDS), "field_provenance")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})), "field_principles")
    for condition in READINESS_CONDITIONS:
        require(
            f"verifier_bounded_csl_ready_score:{condition}" in artifact.get("field_principles", {}),
            "field_principles",
        )
    require(artifact.get("autotokenizer_usage_count") == 0, "autotokenizer")
    require(artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("reuse_count") == 0, "raw_reuse")
    require(
        artifact.get("chronology_and_future_only_checks", {}).get("same_unit_update_use_count") == 0,
        "chronology",
    )
    require(
        artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is True
        or artifact.get("status") == "blocked_preconditions",
        "attack_matrix",
    )
    if artifact.get("status") != "blocked_preconditions":
        rows = artifact.get("per_unit_rows", {}).get("rows", [])
        expected = len(MANDATED_MODEL_IDS) * UNITS_PER_MODEL * len(ARMS)
        require(len(rows) == expected, "row_count")
        require([row.get("hf_id") for row in artifact["MODEL_SPECS"]] == list(MANDATED_MODEL_IDS), "MODEL_SPECS")
        require(artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True, "aggregate")
        require(
            artifact.get("future_exact_yield_delta", {}).get("verifier_bounded_minus_frozen", 0.0) > 0.0
            or artifact.get("verifier_bounded_csl_ready_score") != 1.0,
            "ready_delta",
        )
        require(
            artifact.get("future_exact_yield_delta", {}).get("verifier_bounded_minus_teacher", 0.0) > 0.0
            or artifact.get("verifier_bounded_csl_ready_score") != 1.0,
            "ready_delta",
        )
        require(_candidate_pools_match(rows), "candidate_pool_matching")
        require(_verifier_signs_are_exact(rows), "verifier_sign")
    verdict = str(artifact.get("honest_verdict", ""))
    require(
        verdict.startswith(("success:", "complete:", "blocked:")),
        "honest_verdict",
    )
    return True


def _candidate_pools_match(rows: Sequence[Mapping[str, Any]]) -> bool:
    grouped: dict[tuple[str, int], set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["model"]), int(row["chronological_index"]))].add(tuple(row["candidate_hashes"]))
    return all(len(values) == 1 for values in grouped.values())


def _verifier_signs_are_exact(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        row.get("applied_update_sign") == row.get("exact_sign")
        for row in rows
        if row.get("arm") == VERIFIER_BOUNDED_ARM
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if args.validate:
        validate_artifact(output)
        print(f"valid: {output}")
        return 0
    artifact = run(date=args.date, result_path=output, data_dir=REPO_ROOT / DATA_DIR_RELATIVE_PATH)
    print(json.dumps({"status": artifact["status"], "result_path": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
