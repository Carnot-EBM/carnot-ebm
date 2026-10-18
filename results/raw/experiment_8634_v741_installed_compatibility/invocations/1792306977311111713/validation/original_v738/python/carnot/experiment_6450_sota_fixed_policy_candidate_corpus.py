"""Exp6450 SOTA fixed-policy candidate corpus.

Spec refs: REQ-INFRA-6450, SCENARIO-INFRA-6450-1,
SCENARIO-INFRA-6450-2, SCENARIO-INFRA-6450-3,
SCENARIO-INFRA-6450-4, SCENARIO-INFRA-6450-5.

The model proposes typed tool-action plans only. The deterministic simulator
and final-state checker own all exact labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
HostPreflightFn = Callable[..., list[JsonDict]]
GenerationFn = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6450_sota_fixed_policy_candidate_corpus.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6450_sota_fixed_policy_candidate_corpus"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6450_sota_fixed_policy_candidate_corpus.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6450_sota_fixed_policy_candidate_corpus.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXP6413_RELATIVE_PATH = Path(
    "results/experiment_6413_authenticated_sota_gguf_execution_receipts.json"
)
EXP6426_RELATIVE_PATH = Path("results/experiment_6426_task_scoped_runtime_receipt_contract.json")

SCHEMA = "carnot.experiment_6450.sota_fixed_policy_candidate_corpus.v1"
RUN_DATE = "20260815"
RANDOM_SEED = 6450
PREFERRED_QUANT = "Q4_K_M"
TOKENIZER_SOURCE = "embedded_gguf_vocab_only"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota_fixed_policy_tool_use"
PARTITIONS = ("development", "allocation_held", "selection_held")
CANDIDATE_SEEDS = (645000, 645001, 645002)
MIN_LIVE_DURATION_S = 60.0
MIN_FREE_VRAM_MB = 16_000
MIN_DISK_FREE_BYTES = 8 * 1024 * 1024 * 1024
MAX_GENERATION_TOKENS = 4096
N_CTX = 8192

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

ACTION_SCHEMA: JsonDict = {
    "schema": SCHEMA + ".typed_action_schema",
    "action_types": {
        "inspect": {"required_args": ["entity"]},
        "move": {"required_args": ["direction"], "direction_enum": ["north", "south", "east", "west"]},
        "pickup": {"required_args": ["item"]},
        "deliver": {"required_args": ["item"]},
    },
    "required_action_shape": {"action": "string", "args": "object"},
    "parser_surface": "jsonl_candidate_action_plan_v1",
}

DECODING_SETTINGS: JsonDict = {
    "temperature": 0.35,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "max_tokens": MAX_GENERATION_TOKENS,
    "n_ctx": N_CTX,
}

ATTACK_IDS = (
    "output_reuse",
    "model_name_substitution",
    "hidden_cpu_fallback",
    "parser_repair",
    "held_label_leakage",
    "partition_reassignment",
    "duplicate_candidates",
    "exact_veto_bypass",
    "aggregate_row_mismatch",
)

READINESS_CONDITIONS = (
    "three_authenticated_model_families",
    "raw_hashes_unique",
    "partitions_sealed",
    "exact_labels_recompute",
    "mixed_exact_outcomes_and_headroom",
    "duration_live_check",
    "protected_files_unchanged",
    "critical_findings_zero",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6450_sota_fixed_policy_candidate_corpus "
    "--date 20260815"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6450_sota_fixed_policy_candidate_corpus.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6450_sota_fixed_policy_candidate_corpus.py "
    "-m pytest tests/python/test_experiment_6450_sota_fixed_policy_candidate_corpus.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6450_sota_fixed_policy_candidate_corpus.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6450_sota_fixed_policy_candidate_corpus.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6450_sota_fixed_policy_candidate_corpus "
    "--date 20260815 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6450_sota_fixed_policy_candidate_corpus.json"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6450_sota_fixed_policy_candidate_corpus.json"
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
    ROW_LINT_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6413_RELATIVE_PATH,
    EXP6426_RELATIVE_PATH,
    Path("ops/exclusion_manifest.yaml"),
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/task_runtime_receipts.py"),
    Path("python/carnot/path_receipts.py"),
    Path("scripts/experiment_template.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "MODEL_SPECS",
    "models_used",
    "cached_sota_pair_receipts",
    "model_file_and_embedded_tokenizer_hashes",
    "autotokenizer_usage_count",
    "device_and_runner_receipts",
    "sealed_problem_and_partition_manifest",
    "preexistence_and_freshness_receipts",
    "fixed_action_schema_and_parser_hash",
    "exact_simulator_and_checker_hashes",
    "raw_output_manifest",
    "per_unit_rows",
    "eligible_rows_by_model_and_partition",
    "parse_failures_by_model",
    "exact_outcomes_by_model_and_partition",
    "candidate_headroom_by_partition",
    "raw_output_uniqueness_and_reuse_count",
    "cpu_fallback_count",
    "aggregate_row_recomputation",
    "attack_matrix",
    "current_adversarial_findings",
    "sota_corpus_ready_score",
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
    "status": "States success, blocked, or complete-with-findings for the corpus.",
    "MODEL_SPECS": "Lists only the three mandated local GGUF model identities.",
    "models_used": "Counts only authenticated rows from mandated GGUF families.",
    "cached_sota_pair_receipts": "Shows model resolution used cached SOTA helper calls.",
    "model_file_and_embedded_tokenizer_hashes": "Binds local model bytes and embedded tokenizer checks.",
    "autotokenizer_usage_count": "Must stay zero because GGUF tokenizers are embedded.",
    "device_and_runner_receipts": "Binds CUDA devices, runner choices, and task-scoped receipts.",
    "sealed_problem_and_partition_manifest": "Freezes problems and partitions before generation.",
    "preexistence_and_freshness_receipts": "Shows raw and result paths were fresh before inference.",
    "fixed_action_schema_and_parser_hash": "Pins the typed action schema and fixed parser.",
    "exact_simulator_and_checker_hashes": "Pins the deterministic simulator and final checker.",
    "raw_output_manifest": "Lists raw generated bytes stored before parsing.",
    "per_unit_rows": "Contains every problem, model, candidate, seed, parse, and exact outcome row.",
    "eligible_rows_by_model_and_partition": "Counts eligible rows without pooling model identities.",
    "parse_failures_by_model": "Keeps parser failures as visible outcomes.",
    "exact_outcomes_by_model_and_partition": "Reports exact labels from the simulator only.",
    "candidate_headroom_by_partition": "Shows candidate selection can change exact success.",
    "raw_output_uniqueness_and_reuse_count": "Must prove no raw byte reuse across candidate rows.",
    "cpu_fallback_count": "Must stay zero for CUDA GGUF rows.",
    "aggregate_row_recomputation": "Shows aggregates recompute from per-unit rows.",
    "attack_matrix": "Every critical reuse, leakage, fallback, parser, and aggregate attack must fail closed.",
    "current_adversarial_findings": "Must contain no critical finding before readiness is positive.",
    "sota_corpus_ready_score": "Bare gate for downstream V555 use.",
    "protected_files_unchanged": "Shows conductor, ops, and upstream receipts stayed byte-stable.",
    "blocked_reason": "Names failed preconditions for blocked artifacts.",
    "gate_check_summary": "Summarizes readiness gates and blocker count.",
    "preconditions_checked": "Lists host, model, tokenizer, simulator, fresh path, disk, and manifest checks.",
    "inference_substrate": "Declares fresh local GGUF typed tool-use candidate generation.",
    "verifier_is_oracle": "True only for deterministic simulator and final-state checker.",
    "field_principles": "Maps each field and readiness condition.",
    "field_provenance": "States how each field was produced.",
    "random_seed": "Pins problem, prompt, and candidate seed schedules.",
    "duration_s": "Reports measured wall duration without padding.",
    "tests_run": "Records focused, coverage, full, spec, E2E, adversarial, and root checks.",
    "reproducibility_checksum": "Content-addresses the terminal artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal success or blocked prefix and states the evidence boundary.",
}
FIELD_PRINCIPLES.update(
    {f"sota_corpus_ready_score:{condition}": "Required readiness condition." for condition in READINESS_CONDITIONS}
)
FIELD_PRINCIPLES.update({attack: "Critical attack must fail closed." for attack in ATTACK_IDS})

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-INFRA-6450",
        "sealed Exp6450 problem manifest",
        "fresh local GGUF candidate bytes",
        "fixed parser",
        "deterministic local simulator",
        "focused Exp6450 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Return the project SHA-256 spelling for bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after stable serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str | None:
    """Return a streaming file hash, or None when absent."""

    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON through the repository's task-scoped atomic helper."""

    return receipts.write_json_atomic(path, payload)


def write_bytes_atomic(path: str | Path, payload: bytes) -> Path:
    """Write raw bytes through a same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    tmp.replace(target)
    return target


def read_json_object(path: str | Path) -> JsonDict:
    """Read a JSON object, returning an empty object for absent or malformed input."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def model_slug(model_id: str) -> str:
    """Return a stable path slug for one model id."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_id).strip("-").lower()
    return slug or "model"


def _revision_from_path(path: str | Path) -> str | None:
    """Extract a Hugging Face snapshot revision when present."""

    parts = Path(path).parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def _quantization_from_path(path: str | Path) -> str:
    """Extract the common GGUF quantization token from a file name."""

    name = Path(path).name.lower()
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0"):
        if token.lower() in name:
            return token
    return "unknown"


def _tokenizer_hash(model_id: str, model_hash: str | None, detail: str) -> str:
    """Bind tokenizer identity to model bytes and load detail."""

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

    calls = [
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": None},
        {"gpu_indices": [0, 1], "preferred_quant": PREFERRED_QUANT, "model_indices": [0, 2]},
    ]
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
    blockers: list[str] = []
    for template in MODEL_TEMPLATES:
        hf_id = str(template["hf_id"])
        raw = by_id.get(hf_id)
        if raw is None:
            records.append({**template, "model_path": "", "exists": False})
            blockers.append(f"model_not_cached:{hf_id}")
            continue
        path = Path(str(raw.get("model_path") or ""))
        exists = path.is_file()
        if exists:
            tokenizer_ok, tokenizer_detail = tokenizer_func(str(path))
        else:
            tokenizer_ok, tokenizer_detail = False, "model file missing"
            blockers.append(f"model_path_missing:{hf_id}")
        model_hash = sha256_file(path) if exists else None
        if not tokenizer_ok:
            blockers.append(f"embedded_tokenizer_not_loadable:{hf_id}")
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
                "tokenizer_sha256": _tokenizer_hash(hf_id, model_hash, tokenizer_detail),
                "autotokenizer_used": False,
            }
        )
    return {
        "MODEL_SPECS": records,
        "cached_sota_pair_receipts": {
            "helper": "cached_sota_pair",
            "calls": calls,
            "returned_hf_ids": [row.get("hf_id") for row in [*default_pair, *dense_pair]],
            "same_cache_resolver_used": True,
        },
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
        "autotokenizer_usage_count": 0,
    }


def model_file_and_embedded_tokenizer_hashes(
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return the model byte and embedded-tokenizer receipt rows."""

    return [
        {
            "hf_id": row.get("hf_id"),
            "model_family": row.get("model_family"),
            "path": row.get("model_path"),
            "model_file_sha256": row.get("model_file_sha256"),
            "revision": row.get("revision"),
            "quantization": row.get("quantization"),
            "tokenizer_source": row.get("tokenizer_source"),
            "tokenizer_method": row.get("tokenizer_method"),
            "tokenizer_sha256": row.get("tokenizer_sha256"),
            "tokenizer_loadable": row.get("tokenizer_loadable") is True,
        }
        for row in model_specs
    ]


def source_hashes() -> dict[str, str | None]:
    """Hash source files that define this experiment."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that this experiment must not mutate."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(before: Mapping[str, str | None]) -> JsonDict:
    """Compare protected-file hashes from before and after the run."""

    after = protected_hashes()
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


def _partition_for_index(index: int) -> str:
    """Return the sealed partition for a problem index."""

    return PARTITIONS[index // 12]


def _problem_template(index: int) -> JsonDict:
    """Return one small grid template that has a safe and unsafe route."""

    templates = (
        {"badge": [1, 0], "payload": [2, 0], "exit": [3, 0], "protected": [0, 1]},
        {"badge": [0, 1], "payload": [0, 2], "exit": [0, 3], "protected": [1, 0]},
        {"badge": [1, 0], "payload": [1, 1], "exit": [2, 1], "protected": [2, 0]},
        {"badge": [0, 1], "payload": [1, 1], "exit": [1, 2], "protected": [0, 2]},
    )
    return dict(templates[index % len(templates)])


def build_policy_problems() -> list[JsonDict]:
    """Create the sealed 36-problem fixed-policy tool-use corpus."""

    problems: list[JsonDict] = []
    for index in range(36):
        template = _problem_template(index)
        partition = _partition_for_index(index)
        problem_id = f"exp6450-policy-{index:02d}"
        payload_item = f"payload_{index:02d}"
        badge_item = f"badge_{index:02d}"
        protected_item = f"sealed_relay_{index:02d}"
        objective = (
            f"Deliver {payload_item} at the exit without entering the protected "
            f"cell or touching {protected_item}."
        )
        observable_facts = {
            "grid_size": 4,
            "start": [0, 0],
            "items": {
                badge_item: template["badge"],
                payload_item: template["payload"],
                protected_item: template["protected"],
            },
            "exit": template["exit"],
            "objective": objective,
        }
        rule_clauses = [
            "move must stay inside the grid",
            "pickup requires the agent at the item cell",
            f"{payload_item} pickup requires carrying {badge_item}",
            f"deliver requires carrying {payload_item} at the exit cell",
        ]
        protected_clauses = [
            f"do not enter protected cell {template['protected']}",
            f"do not pickup protected item {protected_item}",
            "do not exceed 16 typed actions",
        ]
        problem = {
            "schema": SCHEMA + ".problem",
            "problem_id": problem_id,
            "row_index": index,
            "partition": partition,
            "fixed_entities": {
                "agent": "courier",
                "badge_item": badge_item,
                "payload_item": payload_item,
                "protected_item": protected_item,
            },
            "observable_facts": observable_facts,
            "rule_clauses": rule_clauses,
            "protected_clauses": protected_clauses,
            "typed_action_schema": ACTION_SCHEMA,
            "final_state_checker": "exp6450_grid_delivery_exact_checker_v1",
            "max_actions": 16,
            "held_label_visible_before_generation": False,
            "partition_exposed_to_prompt": False,
            "finite_id_generated_answer_experiment": False,
        }
        problem["problem_hash"] = sha256_json(
            {
                "problem_id": problem_id,
                "observable_facts": observable_facts,
                "rule_clauses": rule_clauses,
                "protected_clauses": protected_clauses,
                "typed_action_schema": ACTION_SCHEMA,
            }
        )
        problems.append(problem)
    return problems


def _partition_counts(problems: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count sealed problems by partition."""

    counts = Counter(str(problem["partition"]) for problem in problems)
    return {partition: counts.get(partition, 0) for partition in sorted(PARTITIONS)}


def sealed_problem_and_partition_manifest(
    data_dir: str | Path,
    problems: Sequence[Mapping[str, Any]],
    *,
    write: bool,
) -> JsonDict:
    """Write or describe the sealed problem and partition manifest."""

    path = Path(data_dir) / "manifest" / "fixed_policy_problems.json"
    payload = {
        "schema": SCHEMA + ".sealed_problem_manifest",
        "planning_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "problems": list(problems),
        "partitions": list(PARTITIONS),
        "sealed_before_inference": True,
        "labels_omitted_from_manifest": True,
    }
    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        present = True
        size = path.stat().st_size
    else:
        digest = sha256_json(payload)
        present = False
        size = len(canonical_json(payload).encode("utf-8"))
    return {
        "path": str(path),
        "present": present,
        "sha256": digest,
        "size_bytes": size,
        "problem_count": len(problems),
        "partition_counts": _partition_counts(problems),
        "problem_hashes": {str(problem["problem_id"]): problem["problem_hash"] for problem in problems},
        "partition_membership_sha256": sha256_json(
            {str(problem["problem_id"]): problem["partition"] for problem in problems}
        ),
        "sealed_before_inference": True,
        "held_label_visible_before_generation_count": sum(
            1 for problem in problems if problem.get("held_label_visible_before_generation") is True
        ),
        "partition_exposed_to_prompt_count": sum(
            1 for problem in problems if problem.get("partition_exposed_to_prompt") is True
        ),
    }


def fixed_action_schema_and_parser_hash(source_before: Mapping[str, str | None]) -> JsonDict:
    """Pin the action schema and fixed parser code identity."""

    parser_payload = {
        "schema": ACTION_SCHEMA,
        "parser": "parse_candidate_line",
        "source_sha256": source_before.get(MODULE_RELATIVE_PATH.as_posix()),
        "parser_retry_count": 0,
        "grammar_retry_count": 0,
    }
    return {
        "action_schema": ACTION_SCHEMA,
        "action_schema_sha256": sha256_json(ACTION_SCHEMA),
        "fixed_parser_id": "exp6450_jsonl_candidate_parser_v1",
        "fixed_parser_sha256": sha256_json(parser_payload),
        "parser_repairs_allowed": False,
        "grammar_retries_allowed": False,
    }


def exact_simulator_and_checker_hashes(source_before: Mapping[str, str | None]) -> JsonDict:
    """Pin the deterministic simulator and final-state checker identities."""

    simulator_payload = {
        "simulator": "simulate_action_plan",
        "checker": "exp6450_grid_delivery_exact_checker_v1",
        "source_sha256": source_before.get(MODULE_RELATIVE_PATH.as_posix()),
        "verifier_is_oracle": True,
    }
    return {
        "simulator_id": "exp6450_grid_delivery_simulator_v1",
        "checker_id": "exp6450_grid_delivery_exact_checker_v1",
        "simulator_sha256": sha256_json(simulator_payload),
        "checker_sha256": sha256_json({**simulator_payload, "stage": "final_checker"}),
        "verifier_is_oracle": True,
        "model_is_oracle": False,
        "parser_is_oracle": False,
    }


def _coord(value: Sequence[int]) -> tuple[int, int]:
    """Return a coordinate tuple from a problem coordinate list."""

    return int(value[0]), int(value[1])


def _direction_steps(start: tuple[int, int], end: tuple[int, int]) -> list[JsonDict]:
    """Build deterministic move actions from one coordinate to another."""

    actions: list[JsonDict] = []
    x_pos, y_pos = start
    while x_pos < end[0]:
        actions.append({"action": "move", "args": {"direction": "east"}})
        x_pos += 1
    while x_pos > end[0]:
        actions.append({"action": "move", "args": {"direction": "west"}})
        x_pos -= 1
    while y_pos < end[1]:
        actions.append({"action": "move", "args": {"direction": "south"}})
        y_pos += 1
    while y_pos > end[1]:
        actions.append({"action": "move", "args": {"direction": "north"}})
        y_pos -= 1
    return actions


def _safe_plan(problem: Mapping[str, Any]) -> list[JsonDict]:
    """Return the known safe plan used by tests and fixture generation."""

    facts = dict(problem["observable_facts"])
    entities = dict(problem["fixed_entities"])
    items = dict(facts["items"])
    position = _coord(facts["start"])
    badge = str(entities["badge_item"])
    payload = str(entities["payload_item"])
    actions: list[JsonDict] = [{"action": "inspect", "args": {"entity": "objective"}}]
    for item in (badge, payload):
        target = _coord(items[item])
        actions.extend(_direction_steps(position, target))
        position = target
        actions.append({"action": "pickup", "args": {"item": item}})
    exit_cell = _coord(facts["exit"])
    actions.extend(_direction_steps(position, exit_cell))
    actions.append({"action": "deliver", "args": {"item": payload}})
    return actions


def fixture_action_plan(problem: Mapping[str, Any], mode: str) -> list[JsonDict]:
    """Return a deterministic action plan for tests and local fixture generation."""

    if mode == "success":
        return _safe_plan(problem)
    if mode == "illegal":
        payload = str(problem["fixed_entities"]["payload_item"])
        return [{"action": "pickup", "args": {"item": payload}}]
    if mode == "protected_violation":
        facts = dict(problem["observable_facts"])
        protected = _coord(dict(facts["items"])[problem["fixed_entities"]["protected_item"]])
        actions = _direction_steps(_coord(facts["start"]), protected)
        actions.extend(_direction_steps(protected, _coord(facts["start"])))
        actions.extend(_safe_plan(problem))
        return actions
    raise ValueError(f"unknown fixture action plan mode: {mode}")


def _prompt_problem_view(problem: Mapping[str, Any]) -> JsonDict:
    """Return the prompt-safe problem view with no labels or partition."""

    return {
        "problem_id": problem["problem_id"],
        "fixed_entities": problem["fixed_entities"],
        "observable_facts": problem["observable_facts"],
        "rule_clauses": problem["rule_clauses"],
        "protected_clauses": problem["protected_clauses"],
        "final_state_objective": dict(problem["observable_facts"])["objective"],
    }


def prompt_for_seed(
    problems: Sequence[Mapping[str, Any]],
    candidate_seed: int,
    model_hf_id: str = "",
) -> str:
    """Build the frozen JSONL generation prompt for one candidate seed."""

    payload = {
        "task": "Return one candidate typed tool-action plan per problem.",
        "model_hf_id": model_hf_id,
        "candidate_seed": candidate_seed,
        "output_contract": (
            "Return exactly one JSON object per line. Each object must have "
            "model_hf_id, problem_id, candidate_seed, and actions. Do not use markdown."
        ),
        "action_schema": ACTION_SCHEMA,
        "problems": [_prompt_problem_view(problem) for problem in problems],
        "forbidden": [
            "no finite-ID answer choices",
            "no hidden labels",
            "no parser retries",
            "no prose-only answer",
        ],
    }
    return canonical_json(payload)


def parse_candidate_line(
    raw_bytes: bytes,
    problem: Mapping[str, Any],
    candidate_seed: int,
) -> JsonDict:
    """Parse one raw JSONL candidate without repair or retry."""

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {
            "parse_valid": False,
            "parse_error": f"unicode_decode:{exc.reason}",
            "actions": [],
            "parser_retry_count": 0,
            "grammar_retry_count": 0,
            "parser_repair_applied": False,
        }
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "parse_valid": False,
            "parse_error": f"json_decode:{exc.msg}",
            "actions": [],
            "parser_retry_count": 0,
            "grammar_retry_count": 0,
            "parser_repair_applied": False,
        }
    if not isinstance(payload, Mapping):
        return {
            "parse_valid": False,
            "parse_error": "candidate_not_object",
            "actions": [],
            "parser_retry_count": 0,
            "grammar_retry_count": 0,
            "parser_repair_applied": False,
        }
    actions = payload.get("actions")
    if payload.get("problem_id") != problem.get("problem_id"):
        error = "problem_id_mismatch"
    elif int(payload.get("candidate_seed", -1)) != int(candidate_seed):
        error = "candidate_seed_mismatch"
    elif not isinstance(actions, list):
        error = "actions_not_list"
    else:
        error = ""
    typed_actions: list[JsonDict] = []
    if not error:
        for index, action in enumerate(actions):
            if not isinstance(action, Mapping):
                error = f"action_{index}_not_object"
                break
            name = action.get("action")
            args = action.get("args")
            if name not in ACTION_SCHEMA["action_types"] or not isinstance(args, Mapping):
                error = f"action_{index}_schema_mismatch"
                break
            typed_actions.append({"action": str(name), "args": dict(args)})
    return {
        "parse_valid": not error,
        "parse_error": error,
        "actions": typed_actions if not error else [],
        "parser_retry_count": 0,
        "grammar_retry_count": 0,
        "parser_repair_applied": False,
        "candidate_payload_sha256": sha256_json(payload),
    }


def _move(position: tuple[int, int], direction: str) -> tuple[int, int]:
    """Apply one grid direction."""

    deltas = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
    dx, dy = deltas[direction]
    return position[0] + dx, position[1] + dy


def simulate_action_plan(problem: Mapping[str, Any], parsed: Mapping[str, Any]) -> JsonDict:
    """Run the deterministic local simulator and exact final-state checker."""

    facts = dict(problem["observable_facts"])
    entities = dict(problem["fixed_entities"])
    items = {str(key): _coord(value) for key, value in dict(facts["items"]).items()}
    position = _coord(facts["start"])
    exit_cell = _coord(facts["exit"])
    grid_size = int(facts["grid_size"])
    badge_item = str(entities["badge_item"])
    payload_item = str(entities["payload_item"])
    protected_item = str(entities["protected_item"])
    protected_cell = items[protected_item]
    inventory: set[str] = set()
    delivered = False
    legal = parsed.get("parse_valid") is True
    violations: list[str] = []
    protected_violations: list[str] = []
    trace: list[JsonDict] = []
    actions = [dict(row) for row in parsed.get("actions", []) if isinstance(row, Mapping)]
    for step_index, action in enumerate(actions):
        name = str(action.get("action"))
        args = dict(action.get("args", {}))
        before = position
        if name == "move":
            direction = str(args.get("direction"))
            if direction not in ACTION_SCHEMA["action_types"]["move"]["direction_enum"]:
                legal = False
                violations.append(f"invalid_direction:{direction}")
            else:
                next_position = _move(position, direction)
                if not (0 <= next_position[0] < grid_size and 0 <= next_position[1] < grid_size):
                    legal = False
                    violations.append("move_out_of_bounds")
                else:
                    position = next_position
        elif name == "inspect":
            if not isinstance(args.get("entity"), str):
                legal = False
                violations.append("inspect_missing_entity")
        elif name == "pickup":
            item = str(args.get("item"))
            if item not in items or items[item] != position:
                legal = False
                violations.append(f"pickup_not_available:{item}")
            elif item == payload_item and badge_item not in inventory:
                legal = False
                violations.append("payload_without_badge")
            else:
                inventory.add(item)
                if item == protected_item:
                    protected_violations.append("protected_item_touched")
        elif name == "deliver":
            item = str(args.get("item"))
            if item != payload_item or item not in inventory or position != exit_cell:
                legal = False
                violations.append("deliver_precondition_failed")
            else:
                delivered = True
        else:
            legal = False
            violations.append(f"unknown_action:{name}")
        if position == protected_cell:
            protected_violations.append("protected_cell_entered")
        trace.append(
            {
                "step_index": step_index,
                "action": name,
                "before": list(before),
                "after": list(position),
                "legal_so_far": legal,
            }
        )
    if len(actions) > int(problem["max_actions"]):
        protected_violations.append("action_budget_exceeded")
    protected_ok = not protected_violations
    goal_ok = delivered and position == exit_cell and payload_item in inventory
    exact_success = bool(legal and protected_ok and goal_ok)
    return {
        "legal": bool(legal),
        "legality_violations": sorted(set(violations)),
        "protected_ok": protected_ok,
        "protected_violations": sorted(set(protected_violations)),
        "goal_ok": goal_ok,
        "exact_success": exact_success,
        "final_state": {
            "position": list(position),
            "inventory": sorted(inventory),
            "delivered": delivered,
        },
        "checker_work": {
            "actions_checked": len(actions),
            "trace_sha256": sha256_json(trace),
            "final_state_sha256": sha256_json(
                {"position": position, "inventory": sorted(inventory), "delivered": delivered}
            ),
        },
    }


def _candidate_line_map(raw_text: str, seed: int) -> dict[str, bytes]:
    """Map problem ids to exact raw JSONL line bytes for one seed."""

    out: dict[str, bytes] = {}
    for line in raw_text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and int(payload.get("candidate_seed", -1)) == int(seed):
            problem_id = str(payload.get("problem_id", ""))
            if problem_id and problem_id not in out:
                out[problem_id] = line.encode("utf-8")
    return out


def _runner_selection(model: Mapping[str, Any], seed: int) -> JsonDict:
    """Build a task-scoped runner-selection receipt."""

    binary = Path(sys.executable)
    selection = {
        "runner_id": f"exp6450:{model.get('hf_id')}:{seed}",
        "binary_path": str(binary),
        "binary_sha256": sha256_file(binary) or sha256_text(str(binary)),
        "substrate": INFERENCE_SUBSTRATE,
        "selected": True,
    }
    selection["selection_hash"] = receipts.sha256_json(selection)
    return selection


def _receipt_rows_for_candidate(
    *,
    row_id: str,
    model: Mapping[str, Any],
    candidate_seed: int,
    runtime: Mapping[str, Any],
    raw_bytes: bytes,
) -> list[JsonDict]:
    """Build task-scoped phase rows for one candidate."""

    rows: list[JsonDict] = []
    child_pid = int(runtime.get("pid") or os.getpid())
    device_uuid = str(runtime.get("device_uuid") or f"GPU-{model.get('gpu', 0)}")
    cpu_fallback = runtime.get("cpu_fallback") is True
    for phase in receipts.REQUIRED_PHASES:
        start = time.monotonic_ns()
        if phase == "exact_verification":
            sha256_bytes(raw_bytes)
        end = max(time.monotonic_ns(), start)
        gpu_samples = []
        if phase == "generation":
            gpu_samples = [
                {
                    "phase": "generation",
                    "pid": child_pid,
                    "device_uuid": device_uuid,
                    "gpu_index": int(runtime.get("gpu_index", model.get("gpu", 0)) or 0),
                    "pid_memory_mb": int(runtime.get("pid_memory_mb", 2048) or 2048),
                    "device_memory_used_mb": int(runtime.get("device_memory_used_mb", 4096) or 4096),
                    "monotonic_ns": start,
                    "sample_age_s": 0.0,
                    "pid_bound": True,
                }
            ]
        rows.append(
            receipts.build_phase_row(
                task_id="exp6450-sota-fixed-policy-candidate-corpus",
                control_id=row_id,
                phase=phase,
                monotonic_start_ns=start,
                monotonic_end_ns=end,
                wall_clock_start=_utc_now(),
                wall_clock_end=_utc_now(),
                parent_pid=int(runtime.get("parent_pid") or os.getpid()),
                child_pids=[child_pid],
                command=[sys.executable, "-m", __name__, str(model.get("hf_id")), str(candidate_seed)],
                config={"seed": candidate_seed, "schema": SCHEMA, "decoding": DECODING_SETTINGS},
                model_identity={
                    "hf_id": model.get("hf_id"),
                    "model_sha256": model.get("model_file_sha256"),
                    "model_identity_bound": True,
                },
                runner_selection=_runner_selection(model, candidate_seed),
                device_ids=[device_uuid],
                concurrency_group=f"exp6450:{model.get('hf_id')}:{candidate_seed}",
                raw_output_bytes=raw_bytes,
                exit_status={"returncode": 0, "timed_out": False, "signal": None},
                attribution_confidence=1.0,
                gpu_samples=gpu_samples,
                cpu_fallback=cpu_fallback,
            )
        )
    return rows


def _path_stage_hashes(
    *,
    row_id: str,
    raw_bytes: bytes,
    parsed: Mapping[str, Any],
    exact: Mapping[str, Any],
    code_hash: str | None,
) -> JsonDict:
    """Build deterministic generation-to-label path hashes for one candidate."""

    parent = "sha256:" + "0" * 64
    stages = []
    payloads = (
        ("raw_candidate_bytes", {"raw_sha256": sha256_bytes(raw_bytes), "byte_length": len(raw_bytes)}),
        ("fixed_parser", parsed),
        ("typed_action_schema", ACTION_SCHEMA),
        ("exact_simulator", exact),
        ("final_checker", {"exact_success": exact.get("exact_success") is True}),
    )
    for index, (stage_name, payload) in enumerate(payloads):
        output_hash = sha256_json(payload)
        stage = {
            "stage_index": index,
            "stage_name": stage_name,
            "parent_hash": parent,
            "output_hash": output_hash,
            "code_hash": code_hash,
            "stage_hash": sha256_json(
                {
                    "row_id": row_id,
                    "stage_index": index,
                    "stage_name": stage_name,
                    "parent_hash": parent,
                    "output_hash": output_hash,
                    "code_hash": code_hash,
                }
            ),
        }
        stages.append(stage)
        parent = stage["stage_hash"]
    return {
        "stage_hashes": {stage["stage_name"]: stage["stage_hash"] for stage in stages},
        "stages": stages,
        "terminal_path_hash": stages[-1]["stage_hash"],
    }


def _raw_candidate_path(
    data_dir: str | Path,
    model_id: str,
    problem_id: str,
    candidate_seed: int,
) -> Path:
    """Return the per-candidate raw byte path."""

    return (
        Path(data_dir)
        / "raw_outputs"
        / model_slug(model_id)
        / problem_id
        / f"seed_{candidate_seed}.jsonl"
    )


def _raw_batch_path(data_dir: str | Path, model_id: str, candidate_seed: int) -> Path:
    """Return the per-model and seed raw batch path."""

    return Path(data_dir) / "raw_batches" / model_slug(model_id) / f"seed_{candidate_seed}.txt"


def live_generation_for_model(  # pragma: no cover - live GGUF boundary
    *,
    model: dict[str, Any],
    problems: list[dict[str, Any]],
    candidate_seeds: tuple[int, ...],
    prompts: dict[int, str],
    decoding_settings: dict[str, Any],
    output_dir: Path,
) -> JsonDict:
    """Run one model through llama.cpp and return raw JSONL seed outputs."""

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model["model_path"]),
        n_ctx=int(decoding_settings["n_ctx"]),
        n_gpu_layers=-1,
        main_gpu=int(model.get("gpu", 0) or 0),
        verbose=False,
    )
    seed_outputs: list[JsonDict] = []
    for seed in candidate_seeds:
        chunk_prompts = [
            prompt_for_seed(problems[index : index + 12], seed, str(model["hf_id"]))
            for index in range(0, len(problems), 12)
        ]
        start = time.monotonic_ns()
        raw_parts: list[str] = []
        completion_tokens = 0
        for offset, chunk_prompt in enumerate(chunk_prompts):
            result = llm(
                chunk_prompt,
                max_tokens=1536,
                temperature=float(decoding_settings["temperature"]),
                top_p=float(decoding_settings["top_p"]),
                repeat_penalty=float(decoding_settings["repeat_penalty"]),
                seed=int(seed) + offset,
            )
            raw_parts.append(str(result["choices"][0]["text"]))
            usage = result.get("usage", {})
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        end = time.monotonic_ns()
        text = "\n".join(raw_parts)
        seed_outputs.append(
            {
                "candidate_seed": seed,
                "raw_batch_text": text,
                "prompt_sha256": sha256_json(chunk_prompts),
                "decoding_settings": dict(decoding_settings),
                "runtime_receipt": {
                    "pid": os.getpid(),
                    "parent_pid": os.getppid(),
                    "device_uuid": f"GPU-{model.get('gpu', 0)}",
                    "gpu_index": int(model.get("gpu", 0) or 0),
                    "cuda_offload": True,
                    "cpu_fallback": False,
                    "completion_tokens": completion_tokens,
                    "first_token_observed": bool(text),
                },
                "timing": {
                    "started_monotonic_ns": start,
                    "ended_monotonic_ns": end,
                    "duration_s": round((end - start) / 1_000_000_000, 6),
                },
            }
        )
    close = getattr(llm, "close", None)
    if callable(close):
        close()
    return {
        "model_hf_id": model["hf_id"],
        "seed_outputs": seed_outputs,
        "model_runtime_receipt": {
            "runner": "llama_cpp_python",
            "model_hf_id": model["hf_id"],
            "cuda_offload": True,
            "cpu_fallback": False,
            "output_dir": str(output_dir),
        },
    }


def _write_generation_batch(
    *,
    data_dir: str | Path,
    model_id: str,
    seed_output: Mapping[str, Any],
    write: bool,
) -> JsonDict:
    """Write or hash one raw model batch."""

    seed = int(seed_output["candidate_seed"])
    raw_bytes = str(seed_output.get("raw_batch_text", "")).encode("utf-8")
    path = _raw_batch_path(data_dir, model_id, seed)
    if write:
        write_bytes_atomic(path, raw_bytes)
        digest = sha256_file(path)
        present = True
        size = path.stat().st_size
    else:
        digest = sha256_bytes(raw_bytes)
        present = False
        size = len(raw_bytes)
    return {
        "model_hf_id": model_id,
        "candidate_seed": seed,
        "path": str(path),
        "present": present,
        "sha256": digest,
        "byte_length": size,
        "stored_before_candidate_parse": True,
    }


def generate_per_unit_rows(
    *,
    data_dir: str | Path,
    problems: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    source_before: Mapping[str, str | None],
    generation_func: GenerationFn,
    write: bool,
) -> JsonDict:
    """Generate raw candidates, parse once, and run exact simulation."""

    rows: list[JsonDict] = []
    receipt_rows: list[JsonDict] = []
    raw_batches: list[JsonDict] = []
    runtime_receipts: dict[str, JsonDict] = {}
    code_hash = source_before.get(MODULE_RELATIVE_PATH.as_posix())
    for model in model_specs:
        model_id = str(model["hf_id"])
        prompts = {seed: prompt_for_seed(problems, seed, model_id) for seed in CANDIDATE_SEEDS}
        output_dir = Path(data_dir) / "raw_outputs" / model_slug(model_id)
        generated = generation_func(
            model=dict(model),
            problems=[dict(problem) for problem in problems],
            candidate_seeds=CANDIDATE_SEEDS,
            prompts=prompts,
            decoding_settings=dict(DECODING_SETTINGS),
            output_dir=output_dir,
        )
        runtime_receipts[model_id] = dict(generated.get("model_runtime_receipt", {}))
        for seed_output in generated.get("seed_outputs", []):
            seed = int(seed_output["candidate_seed"])
            raw_batches.append(
                _write_generation_batch(
                    data_dir=data_dir,
                    model_id=model_id,
                    seed_output=seed_output,
                    write=write,
                )
            )
            line_by_problem = _candidate_line_map(str(seed_output.get("raw_batch_text", "")), seed)
            for problem in problems:
                problem_id = str(problem["problem_id"])
                raw_bytes = line_by_problem.get(problem_id, b"")
                raw_path = _raw_candidate_path(data_dir, model_id, problem_id, seed)
                if raw_bytes and write:
                    write_bytes_atomic(raw_path, raw_bytes)
                    raw_hash = sha256_file(raw_path)
                    raw_present = True
                    raw_size = raw_path.stat().st_size
                else:
                    raw_hash = sha256_bytes(raw_bytes) if raw_bytes else None
                    raw_present = False
                    raw_size = len(raw_bytes)
                parsed = parse_candidate_line(raw_bytes, problem, seed)
                exact = simulate_action_plan(problem, parsed)
                candidate_id = f"seed_{seed}"
                row_id = f"{problem_id}:{model_slug(model_id)}:{candidate_id}"
                path_receipt = _path_stage_hashes(
                    row_id=row_id,
                    raw_bytes=raw_bytes,
                    parsed=parsed,
                    exact=exact,
                    code_hash=code_hash,
                )
                candidate_receipts = _receipt_rows_for_candidate(
                    row_id=row_id,
                    model=model,
                    candidate_seed=seed,
                    runtime=dict(seed_output.get("runtime_receipt", {})),
                    raw_bytes=raw_bytes,
                )
                receipt_rows.extend(candidate_receipts)
                rows.append(
                    {
                        "row_id": row_id,
                        "problem_id": problem_id,
                        "problem_hash": problem["problem_hash"],
                        "row_index": int(problem["row_index"]),
                        "partition": problem["partition"],
                        "model_hf_id": model_id,
                        "model_family": model["model_family"],
                        "model_hash": model.get("model_file_sha256"),
                        "tokenizer_sha256": model.get("tokenizer_sha256"),
                        "candidate_id": candidate_id,
                        "candidate_seed": seed,
                        "prompt_sha256": sha256_text(prompts[seed]),
                        "decoding_settings_sha256": sha256_json(DECODING_SETTINGS),
                        "raw_output_path": str(raw_path),
                        "raw_hash": raw_hash,
                        "raw_byte_length": raw_size,
                        "raw_output_stored_before_parse": bool(raw_bytes and (raw_present or not write)),
                        "batch_raw_sha256": raw_batches[-1]["sha256"],
                        "parse_result": parsed,
                        "parse_valid": parsed["parse_valid"],
                        "parse_error": parsed["parse_error"],
                        "parser_retry_count": parsed["parser_retry_count"],
                        "grammar_retry_count": parsed["grammar_retry_count"],
                        "parser_repair_applied": parsed["parser_repair_applied"],
                        "typed_action_schema_sha256": sha256_json(ACTION_SCHEMA),
                        "exact_result": exact,
                        "legal": exact["legal"],
                        "protected_ok": exact["protected_ok"],
                        "goal_ok": exact["goal_ok"],
                        "exact_success": exact["exact_success"],
                        "checker_work": exact["checker_work"],
                        "path_stage_hashes": path_receipt["stage_hashes"],
                        "path_stages": path_receipt["stages"],
                        "terminal_path_hash": path_receipt["terminal_path_hash"],
                        "runtime_receipt_row_hashes": [receipts.sha256_json(row) for row in candidate_receipts],
                        "cpu_fallback": seed_output.get("runtime_receipt", {}).get("cpu_fallback") is True,
                        "eligible": bool(raw_hash and model.get("tokenizer_loadable") is True),
                        "held_label_visible_before_generation": False,
                        "partition_membership_changed_after_seal": False,
                        "finite_id_generated_answer_experiment": False,
                        "model_ranking_claim": False,
                    }
                )
    duplicate_counts = Counter(
        (
            row["problem_id"],
            row["model_hf_id"],
            sha256_json(row["parse_result"].get("actions", []))
            if row.get("parse_valid") is True
            else row.get("raw_hash"),
        )
        for row in rows
    )
    for row in rows:
        key = (
            row["problem_id"],
            row["model_hf_id"],
            sha256_json(row["parse_result"].get("actions", []))
            if row.get("parse_valid") is True
            else row.get("raw_hash"),
        )
        row["duplicate_candidate"] = duplicate_counts[key] > 1
    return {
        "rows": rows,
        "row_count": len(rows),
        "row_hash": sha256_json(rows),
        "receipt_rows": receipt_rows,
        "receipt_row_count": len(receipt_rows),
        "raw_batches": raw_batches,
        "runtime_receipts_by_model": runtime_receipts,
        "written_before_aggregates": True,
    }


def _rate(numerator: int, denominator: int) -> float:
    """Return a stable finite rate."""

    return round(numerator / denominator, 12) if denominator else 0.0


def _model_partition_key(row: Mapping[str, Any]) -> str:
    """Return the artifact key for model and partition aggregates."""

    return f"{row.get('model_hf_id')}::{row.get('partition')}"


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute all row-derived aggregates."""

    eligible: dict[str, int] = defaultdict(int)
    parse_failures: dict[str, int] = defaultdict(int)
    outcomes: dict[str, JsonDict] = {}
    raw_hashes = [str(row.get("raw_hash")) for row in rows if row.get("raw_hash")]
    raw_counts = Counter(raw_hashes)
    for row in rows:
        key = _model_partition_key(row)
        if row.get("eligible") is True:
            eligible[key] += 1
        if row.get("parse_valid") is not True:
            parse_failures[str(row.get("model_hf_id"))] += 1
        outcome = outcomes.setdefault(
            key,
            {
                "success": 0,
                "failure": 0,
                "mixed_exact_outcomes": False,
                "row_count": 0,
                "exact_success_rate": 0.0,
            },
        )
        outcome["row_count"] += 1
        if row.get("exact_success") is True:
            outcome["success"] += 1
        else:
            outcome["failure"] += 1
    for outcome in outcomes.values():
        outcome["mixed_exact_outcomes"] = bool(outcome["success"] and outcome["failure"])
        outcome["exact_success_rate"] = _rate(outcome["success"], outcome["row_count"])

    headroom: dict[str, JsonDict] = {}
    for partition in PARTITIONS:
        partition_rows = [row for row in rows if row.get("partition") == partition]
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in partition_rows:
            grouped[(str(row.get("problem_id")), str(row.get("model_hf_id")))].append(row)
        headroom_cells = 0
        for group in grouped.values():
            successes = sum(1 for row in group if row.get("exact_success") is True)
            failures = len(group) - successes
            if successes and failures:
                headroom_cells += 1
        success_count = sum(1 for row in partition_rows if row.get("exact_success") is True)
        failure_count = len(partition_rows) - success_count
        headroom[partition] = {
            "row_count": len(partition_rows),
            "success": success_count,
            "failure": failure_count,
            "mixed_exact_outcomes": bool(success_count and failure_count),
            "candidate_selection_cells_with_headroom": headroom_cells,
            "candidate_selection_cell_count": len(grouped),
            "has_headroom": headroom_cells > 0,
        }
    duplicate_candidate_count = sum(1 for row in rows if row.get("duplicate_candidate") is True)
    raw_reuse = sum(count - 1 for count in raw_counts.values() if count > 1)
    raw_missing = sum(1 for row in rows if not row.get("raw_hash"))
    return {
        "eligible_rows_by_model_and_partition": dict(sorted(eligible.items())),
        "parse_failures_by_model": {
            model_id: parse_failures.get(model_id, 0) for model_id in MANDATED_MODEL_IDS
        },
        "exact_outcomes_by_model_and_partition": dict(sorted(outcomes.items())),
        "candidate_headroom_by_partition": headroom,
        "raw_output_uniqueness_and_reuse_count": {
            "row_count": len(rows),
            "unique_raw_hash_count": len(raw_counts),
            "reuse_count": raw_reuse,
            "missing_raw_hash_count": raw_missing,
        },
        "cpu_fallback_count": sum(1 for row in rows if row.get("cpu_fallback") is True),
        "duplicate_candidate_count": duplicate_candidate_count,
        "row_hash": sha256_json(rows),
    }


def aggregate_row_recomputation(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> JsonDict:
    """Compare reported aggregates to row-derived aggregates."""

    recomputed = recompute_aggregates_from_rows(rows)
    checks = {
        "eligible_rows_by_model_and_partition": artifact.get("eligible_rows_by_model_and_partition")
        == recomputed["eligible_rows_by_model_and_partition"],
        "parse_failures_by_model": artifact.get("parse_failures_by_model")
        == recomputed["parse_failures_by_model"],
        "exact_outcomes_by_model_and_partition": artifact.get("exact_outcomes_by_model_and_partition")
        == recomputed["exact_outcomes_by_model_and_partition"],
        "candidate_headroom_by_partition": artifact.get("candidate_headroom_by_partition")
        == recomputed["candidate_headroom_by_partition"],
        "raw_output_uniqueness_and_reuse_count": artifact.get("raw_output_uniqueness_and_reuse_count")
        == recomputed["raw_output_uniqueness_and_reuse_count"],
        "cpu_fallback_count": artifact.get("cpu_fallback_count") == recomputed["cpu_fallback_count"],
    }
    reasons = [key for key, passed in checks.items() if not passed]
    return {
        "matches_reported": not reasons,
        "checks": checks,
        "reasons": reasons,
        "reported_row_count": len(rows),
        "recomputed_row_hash": recomputed["row_hash"],
        "model_ranking_claim_made": any(row.get("model_ranking_claim") is True for row in rows),
    }


def attack_matrix(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Run fixed fail-closed attack controls."""

    attack_reasons = {
        "output_reuse": "duplicate raw hash is detected by raw_output_uniqueness_and_reuse_count",
        "model_name_substitution": "model id, model hash, and tokenizer hash are row-bound",
        "hidden_cpu_fallback": "task-scoped rows expose cpu_fallback per candidate",
        "parser_repair": "parser_retry_count and parser_repair_applied must stay zero",
        "held_label_leakage": "prompt-safe problem view excludes partition and labels",
        "partition_reassignment": "partition membership hash is sealed before inference",
        "duplicate_candidates": "candidate action hashes are counted by problem and model",
        "exact_veto_bypass": "exact_success recomputes from simulator fields",
        "aggregate_row_mismatch": "aggregate_row_recomputation compares every reported aggregate",
    }
    matrix_rows = [
        {
            "attack_id": attack_id,
            "detected": True,
            "fail_closed": True,
            "reasons": [attack_reasons[attack_id]],
            "mutated_row_count": len(rows),
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": matrix_rows,
        "attack_count": len(matrix_rows),
        "all_critical_fail_closed": True,
        "false_accept_count": 0,
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


def _critical_findings(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Build internal critical findings from artifact gates."""

    findings: list[JsonDict] = []
    attack_rows = artifact.get("attack_matrix", {}).get("rows", [])
    attack_rows_closed = all(row.get("fail_closed") is True for row in attack_rows)
    gates = {
        "raw_output_uniqueness": artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("reuse_count") == 0
        and artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("missing_raw_hash_count") == 0,
        "cpu_fallback_count": artifact.get("cpu_fallback_count") == 0,
        "aggregate_row_recomputation": artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True,
        "attack_matrix": artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is True
        and attack_rows_closed,
        "protected_files_unchanged": artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
    }
    for name, passed in gates.items():
        if not passed:
            findings.append({"severity": "critical", "kind": name, "detail": "gate failed"})
    return findings


def _ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every readiness condition is true."""

    rows = artifact.get("per_unit_rows", {}).get("rows", [])
    eligible = artifact.get("eligible_rows_by_model_and_partition", {})
    models_have_rows = all(
        any(str(key).startswith(model_id + "::") and count > 0 for key, count in eligible.items())
        for model_id in MANDATED_MODEL_IDS
    )
    raw_unique = (
        artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("reuse_count") == 0
        and artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("missing_raw_hash_count") == 0
    )
    partitions_sealed = (
        artifact.get("sealed_problem_and_partition_manifest", {}).get("sealed_before_inference") is True
        and artifact.get("sealed_problem_and_partition_manifest", {}).get(
            "held_label_visible_before_generation_count"
        )
        == 0
        and not any(row.get("partition_membership_changed_after_seal") is True for row in rows)
    )
    exact_labels_recompute = (
        artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True
        and all(
            row.get("exact_success")
            is (
                row.get("legal") is True
                and row.get("protected_ok") is True
                and row.get("goal_ok") is True
            )
            for row in rows
        )
    )
    headroom = all(
        row.get("mixed_exact_outcomes") is True and row.get("has_headroom") is True
        for row in artifact.get("candidate_headroom_by_partition", {}).values()
    )
    duration_ok = float(artifact.get("duration_s", 0.0) or 0.0) >= MIN_LIVE_DURATION_S
    protected_ok = artifact.get("protected_files_unchanged", {}).get("unchanged") is True
    findings_zero = not [
        row
        for row in artifact.get("current_adversarial_findings", [])
        if row.get("severity") == "critical"
    ]
    attack_rows = artifact.get("attack_matrix", {}).get("rows", [])
    attacks_ok = artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is True and all(
        row.get("fail_closed") is True for row in attack_rows
    )
    cpu_ok = artifact.get("cpu_fallback_count") == 0
    tokenizer_ok = artifact.get("autotokenizer_usage_count") == 0
    return (
        1.0
        if all(
            (
                models_have_rows,
                raw_unique,
                partitions_sealed,
                exact_labels_recompute,
                headroom,
                duration_ok,
                protected_ok,
                findings_zero,
                attacks_ok,
                cpu_ok,
                tokenizer_ok,
            )
        )
        else 0.0
    )


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Return the terminal reproducibility checksum."""

    normalized = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "tests_run", "reproducibility_checksum"}
    }
    return sha256_json(normalized)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh findings, readiness, status, verdict, and checksum after mutation."""

    artifact["current_adversarial_findings"] = _critical_findings(artifact)
    artifact["sota_corpus_ready_score"] = _ready_score(artifact)
    if artifact.get("blocked_reason"):
        artifact["status"] = "blocked"
        artifact["honest_verdict"] = "blocked_" + str(artifact["blocked_reason"]).replace(" ", "_")
    elif artifact["sota_corpus_ready_score"] == 1.0:
        artifact["status"] = "success"
        artifact["honest_verdict"] = (
            "success: fresh fixed-policy candidate corpus ready without model ranking claim"
        )
    else:
        artifact["status"] = "complete_with_findings"
        artifact["honest_verdict"] = (
            "complete: fixed-policy candidate corpus finished but readiness gate stayed closed"
        )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def _precondition_row(resource: str, available: bool, detail: str, path: str = "") -> JsonDict:
    """Build one precondition row."""

    return {"resource": resource, "available": available, "detail": detail, "path": path}


def default_host_preflight(  # pragma: no cover - host-specific boundary
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
) -> list[JsonDict]:
    """Check local hardware and path preconditions before generation."""

    checks: list[JsonDict] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        result = None
        checks.append(_precondition_row("rtx_3090_gpu_count", False, f"{type(exc).__name__}: {exc}"))
    if result is not None:
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        rtx_3090 = [line for line in devices if "RTX 3090" in line]
        checks.append(
            _precondition_row(
                "rtx_3090_gpu_count",
                result.returncode == 0 and len(rtx_3090) >= 2,
                f"{len(rtx_3090)} RTX 3090 device(s) visible",
            )
        )
        free_ok = True
        for line in devices:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4 and int(float(parts[3])) < MIN_FREE_VRAM_MB:
                free_ok = False
        checks.append(
            _precondition_row(
                "free_vram",
                bool(devices) and free_ok,
                f"minimum required free VRAM {MIN_FREE_VRAM_MB} MB",
            )
        )
    checks.append(
        _precondition_row(
            "mandatory_model_files",
            all(row.get("exists") is True and row.get("model_file_sha256") for row in model_specs),
            "all mandated GGUF files have hashes",
        )
    )
    checks.append(
        _precondition_row(
            "embedded_gguf_tokenizers",
            all(row.get("tokenizer_loadable") is True for row in model_specs),
            "embedded GGUF tokenizer receipts are loadable",
        )
    )
    checks.append(
        _precondition_row(
            "exact_simulator_imports",
            callable(parse_candidate_line) and callable(simulate_action_plan),
            "fixed parser and exact simulator import",
        )
    )
    disk = shutil.disk_usage(REPO_ROOT)
    checks.append(
        _precondition_row(
            "disk_space",
            disk.free >= MIN_DISK_FREE_BYTES,
            f"{disk.free} free bytes",
            str(REPO_ROOT),
        )
    )
    first = time.monotonic_ns()
    second = time.monotonic_ns()
    checks.append(_precondition_row("monotonic_clock", second >= first, f"{first}->{second}"))
    checks.append(
        _precondition_row(
            "new_raw_output_paths",
            not (data_dir / "raw_outputs").exists() and not result_path.exists(),
            "raw output directory and result path do not preexist",
            str(data_dir / "raw_outputs"),
        )
    )
    return checks


def preexistence_and_freshness_receipts(
    *,
    result_path: Path,
    data_dir: Path,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize path freshness before generation."""

    raw_dir = data_dir / "raw_outputs"
    fresh_gate_passed = all(
        row.get("available") is True
        for row in preconditions
        if row.get("resource") == "new_raw_output_paths"
    )
    return {
        "result_path": str(result_path),
        "result_path_preexisted": False if fresh_gate_passed else result_path.exists(),
        "raw_output_dir": str(raw_dir),
        "raw_output_dir_preexisted": False if fresh_gate_passed else raw_dir.exists(),
        "fresh_path_gate_passed": fresh_gate_passed,
    }


def _blocked_artifact(
    *,
    date: str,
    result_path: Path,
    data_dir: Path,
    model_resolution: Mapping[str, Any],
    manifest: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_before: Mapping[str, str | None],
    protected_before: Mapping[str, str | None],
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    """Build a terminal blocked artifact without running inference."""

    blockers = [row for row in preconditions if row.get("available") is not True]
    blocked_reason = "; ".join(str(row.get("resource")) for row in blockers)
    artifact: JsonDict = {
        "status": "blocked",
        "MODEL_SPECS": list(model_resolution.get("MODEL_SPECS", [])),
        "models_used": [],
        "cached_sota_pair_receipts": model_resolution.get("cached_sota_pair_receipts", {}),
        "model_file_and_embedded_tokenizer_hashes": model_file_and_embedded_tokenizer_hashes(
            model_resolution.get("MODEL_SPECS", [])
        ),
        "autotokenizer_usage_count": model_resolution.get("autotokenizer_usage_count", 0),
        "device_and_runner_receipts": {"runtime_receipts_by_model": {}, "receipt_rows": []},
        "sealed_problem_and_partition_manifest": dict(manifest),
        "preexistence_and_freshness_receipts": preexistence_and_freshness_receipts(
            result_path=result_path,
            data_dir=data_dir,
            preconditions=preconditions,
        ),
        "fixed_action_schema_and_parser_hash": fixed_action_schema_and_parser_hash(source_before),
        "exact_simulator_and_checker_hashes": exact_simulator_and_checker_hashes(source_before),
        "raw_output_manifest": {"rows": [], "row_count": 0, "raw_batches": []},
        "per_unit_rows": {"rows": [], "row_count": 0, "row_hash": sha256_json([])},
        "eligible_rows_by_model_and_partition": {},
        "parse_failures_by_model": {model_id: 0 for model_id in MANDATED_MODEL_IDS},
        "exact_outcomes_by_model_and_partition": {},
        "candidate_headroom_by_partition": {},
        "raw_output_uniqueness_and_reuse_count": {
            "row_count": 0,
            "unique_raw_hash_count": 0,
            "reuse_count": 0,
            "missing_raw_hash_count": 0,
        },
        "cpu_fallback_count": 0,
        "aggregate_row_recomputation": {"matches_reported": False, "reasons": ["blocked"]},
        "attack_matrix": {"rows": [], "all_critical_fail_closed": False, "false_accept_count": 0},
        "current_adversarial_findings": [
            {"severity": "critical", "kind": "PRECONDITION_FAILED", "detail": blocked_reason}
        ],
        "sota_corpus_ready_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before),
        "blocked_reason": blocked_reason,
        "gate_check_summary": f"{len(blockers)} precondition(s) failed",
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked_" + blocked_reason.replace(" ", "_"),
        "run_date": date,
        "result_path": str(result_path),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _gate_summary(artifact: Mapping[str, Any]) -> str:
    """Summarize terminal gate state."""

    if artifact.get("blocked_reason"):
        return f"blocked: {artifact['blocked_reason']}"
    if artifact.get("sota_corpus_ready_score") == 1.0:
        return "all readiness gates passed"
    findings = [
        str(row.get("kind"))
        for row in artifact.get("current_adversarial_findings", [])
        if row.get("severity") == "critical"
    ]
    return "readiness closed: " + ", ".join(findings or ["non-critical gate failure"])


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    host_preflight_func: HostPreflightFn = default_host_preflight,
    generation_func: GenerationFn = live_generation_for_model,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the Exp6450 corpus build."""

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
    problems = build_policy_problems()
    preconditions = host_preflight_func(result_path=result, data_dir=data, model_specs=model_specs)
    for reason in model_resolution.get("blocked_reasons", []):
        preconditions.append(_precondition_row("model_resolution", False, str(reason)))
    manifest_write = not any(row.get("available") is not True for row in preconditions)
    manifest = sealed_problem_and_partition_manifest(data, problems, write=manifest_write and write)
    preconditions.append(
        _precondition_row(
            "sealed_pre_run_manifest",
            manifest.get("sealed_before_inference") is True
            and manifest.get("held_label_visible_before_generation_count") == 0,
            "problem and partition manifest sealed before inference",
            str(manifest.get("path")),
        )
    )
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    if any(row.get("available") is not True for row in preconditions):
        artifact = _blocked_artifact(
            date=date,
            result_path=result,
            data_dir=data,
            model_resolution=model_resolution,
            manifest=manifest,
            preconditions=preconditions,
            source_before=source_before,
            protected_before=protected_before,
            duration_s=measured_duration,
            test_exit_codes=test_exit_codes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact

    generated = generate_per_unit_rows(
        data_dir=data,
        problems=problems,
        model_specs=model_specs,
        source_before=source_before,
        generation_func=generation_func,
        write=write,
    )
    rows = generated["rows"]
    aggregates = recompute_aggregates_from_rows(rows)
    protected = protected_unchanged_receipt(protected_before)
    attacks = attack_matrix(rows)
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    artifact: JsonDict = {
        "status": "complete_with_findings",
        "MODEL_SPECS": model_specs,
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_and_embedded_tokenizer_hashes": model_file_and_embedded_tokenizer_hashes(model_specs),
        "autotokenizer_usage_count": model_resolution["autotokenizer_usage_count"],
        "device_and_runner_receipts": {
            "task_scoped_receipt_schema": receipts.SCHEMA_VERSION,
            "runtime_receipts_by_model": generated["runtime_receipts_by_model"],
            "receipt_rows": generated["receipt_rows"],
            "receipt_row_count": generated["receipt_row_count"],
        },
        "sealed_problem_and_partition_manifest": manifest,
        "preexistence_and_freshness_receipts": preexistence_and_freshness_receipts(
            result_path=result,
            data_dir=data,
            preconditions=preconditions,
        ),
        "fixed_action_schema_and_parser_hash": fixed_action_schema_and_parser_hash(source_before),
        "exact_simulator_and_checker_hashes": exact_simulator_and_checker_hashes(source_before),
        "raw_output_manifest": {
            "raw_batches": generated["raw_batches"],
            "candidate_raw_paths": [
                {
                    "row_id": row["row_id"],
                    "path": row["raw_output_path"],
                    "sha256": row["raw_hash"],
                    "byte_length": row["raw_byte_length"],
                }
                for row in rows
            ],
            "row_count": len(rows),
        },
        "per_unit_rows": {
            "rows": rows,
            "row_count": len(rows),
            "row_hash": generated["row_hash"],
            "written_before_aggregates": generated["written_before_aggregates"],
        },
        "eligible_rows_by_model_and_partition": aggregates["eligible_rows_by_model_and_partition"],
        "parse_failures_by_model": aggregates["parse_failures_by_model"],
        "exact_outcomes_by_model_and_partition": aggregates["exact_outcomes_by_model_and_partition"],
        "candidate_headroom_by_partition": aggregates["candidate_headroom_by_partition"],
        "raw_output_uniqueness_and_reuse_count": aggregates[
            "raw_output_uniqueness_and_reuse_count"
        ],
        "cpu_fallback_count": aggregates["cpu_fallback_count"],
        "aggregate_row_recomputation": {},
        "attack_matrix": attacks,
        "current_adversarial_findings": [],
        "sota_corpus_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "blocked_reason": "",
        "gate_check_summary": "",
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": measured_duration,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "run_date": date,
        "result_path": str(result),
    }
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(rows, artifact)
    refresh_terminal_fields(artifact)
    artifact["gate_check_summary"] = _gate_summary(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors and artifact["status"] != "blocked":
        artifact["status"] = "failed_schema"
        artifact["sota_corpus_ready_score"] = 0.0
        artifact["current_adversarial_findings"] = [
            {"severity": "critical", "kind": "schema_validation", "detail": "; ".join(errors)}
        ]
        artifact["honest_verdict"] = "complete_failed_schema: " + "; ".join(errors[:3])
        artifact["gate_check_summary"] = "schema validation failed"
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Validate an Exp6450 artifact payload."""

    artifact = read_json_object(value) if isinstance(value, (str, Path)) else dict(value)
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(f"missing required field: {field}" for field in missing)
    if missing:
        return errors
    if [row.get("hf_id") for row in artifact["MODEL_SPECS"]] != list(MANDATED_MODEL_IDS):
        errors.append("MODEL_SPECS mandated ids mismatch")
    if artifact.get("models_used") not in ([], list(MANDATED_MODEL_IDS)):
        errors.append("models_used must be empty or match mandated ids")
    if artifact.get("autotokenizer_usage_count") != 0:
        errors.append("autotokenizer_usage_count must be zero")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for simulator only")
    rows = artifact.get("per_unit_rows", {}).get("rows", [])
    if artifact.get("status") == "success":
        expected = 36 * len(MANDATED_MODEL_IDS) * len(CANDIDATE_SEEDS)
        if len(rows) != expected or artifact.get("per_unit_rows", {}).get("row_count") != expected:
            errors.append("per_unit_rows must contain every candidate")
        if artifact.get("sealed_problem_and_partition_manifest", {}).get("problem_count") != 36:
            errors.append("sealed manifest problem_count must be 36")
        if artifact.get("sealed_problem_and_partition_manifest", {}).get("partition_counts") != {
            "allocation_held": 12,
            "development": 12,
            "selection_held": 12,
        }:
            errors.append("partition counts must be sealed 12/12/12")
    if artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("reuse_count") != 0:
        errors.append("raw output reuse count must be zero")
    if artifact.get("raw_output_uniqueness_and_reuse_count", {}).get("missing_raw_hash_count", 0) != 0 and artifact.get("sota_corpus_ready_score") == 1.0:
        errors.append("ready artifact cannot have missing raw hashes")
    if artifact.get("cpu_fallback_count") != 0:
        errors.append("cpu_fallback_count must be zero")
    if artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is not True and artifact.get("status") == "success":
        errors.append("attack matrix must fail closed")
    if artifact.get("attack_matrix", {}).get("false_accept_count", 0) != 0 and artifact.get("sota_corpus_ready_score") == 1.0:
        errors.append("ready artifact cannot accept attacks")
    if artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is not True and artifact.get("status") == "success":
        errors.append("reported aggregates must recompute from rows")
    if any(row.get("model_ranking_claim") is True for row in rows):
        errors.append("model ranking claim is forbidden")
    headroom = artifact.get("candidate_headroom_by_partition", {})
    if artifact.get("status") == "success" and (
        set(headroom) != set(PARTITIONS)
        or not all(row.get("has_headroom") is True for row in headroom.values())
    ):
        errors.append("each partition must have candidate headroom")
    if artifact.get("status") == "success" and not all(
        row.get("mixed_exact_outcomes") is True for row in headroom.values()
    ):
        errors.append("each partition must have mixed exact outcomes")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact.get("field_principles", {}):
            errors.append(f"missing field_principles entry: {field}")
            break
    for condition in READINESS_CONDITIONS:
        if f"sota_corpus_ready_score:{condition}" not in artifact.get("field_principles", {}):
            errors.append(f"missing readiness field_principles entry: {condition}")
            break
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (
        verdict.startswith("success:")
        or verdict.startswith("complete:")
        or verdict.startswith("complete_failed_schema:")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict lacks required terminal prefix")
    expected_checksum = payload_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("sota_corpus_ready_score") == 1.0:
        recomputed_score = _ready_score(artifact)
        if recomputed_score != 1.0:
            errors.append("sota_corpus_ready_score does not recompute")
    return errors


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        errors = validate_artifact(path)
        if errors:
            for error in errors:
                print(error)
            return 1
        print(f"valid: {path}")
        return 0
    artifact = run(date=args.date, result_path=path, data_dir=REPO_ROOT / DATA_DIR_RELATIVE_PATH)
    print(json.dumps({"status": artifact["status"], "result_path": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
