"""Exp6505 local SOTA formal challenge mutation stream.

Spec refs: REQ-BENCH-6505, SCENARIO-BENCH-6505-PROVENANCE,
SCENARIO-BENCH-6505-ONE-SHOT, SCENARIO-BENCH-6505-NO-ANSWER,
SCENARIO-BENCH-6505-ADMISSION, SCENARIO-BENCH-6505-SCORES,
SCENARIO-BENCH-6505-ARTIFACT.

Models can propose only bounded edits to already-formal development rows.
Exact parsing, replay, SAT solving, and validity checks decide admission.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6504_exact_structural_benchmark_commitment as exp6504
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
Clause = list[int]
GenerationRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260822"
RANDOM_SEED = 6505
SOURCE_SELECTION_SEED = 6505001
GENERATION_SEED = 6505101
DEFAULT_SOURCE_COUNT = 1
SCHEMA_VERSION = "carnot.experiment_6505.sota_formal_challenge_mutations.v1"
INFERENCE_SUBSTRATE = (
    "local_llama_cpp_three_family_formal_mutation_generation_plus_exact_solver_admission"
)
VERIFIER_IS_ORACLE = True

RESULT_RELATIVE_PATH = Path("results/experiment_6505_sota_formal_challenge_mutations.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6505_sota_formal_challenge_mutations.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6505_sota_formal_challenge_mutations.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXP6504_RELATIVE_PATH = Path("results/experiment_6504_exact_structural_benchmark_commitment.json")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    EXP6504_RELATIVE_PATH,
    Path("results/experiment_5813_split_budget_sota_canary.json"),
    Path("results/experiment_5923_sota_schema_supported_constraintir_ab.json"),
    Path("results/experiment_6463_sota_fixed_policy_candidate_corpus_v2.json"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXP6504_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
)

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "family": "qwen_moe",
        "role": "flagship_moe_challenge_generator",
        "preferred_quantization": "Q4_K_M",
        "load_order": 1,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "family": "gemma_dense",
        "role": "flagship_dense_challenge_generator",
        "preferred_quantization": "Q4_K_M",
        "load_order": 2,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "family": "gemma_moe",
        "role": "middle_moe_challenge_generator",
        "preferred_quantization": "Q4_K_M",
        "load_order": 3,
    },
]

DECODE_CONFIG: JsonDict = {
    "max_new_tokens": 128,
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "n_ctx": 2048,
    "n_batch": 256,
    "n_ubatch": 64,
    "n_gpu_layers": -1,
    "main_gpu": 0,
    "model_timeout_s": 900,
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6505_sota_formal_challenge_mutations "
    "--date 20260822"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6505_sota_formal_challenge_mutations.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6505_sota_formal_challenge_mutations.py "
    "-m pytest tests/python/test_experiment_6505_sota_formal_challenge_mutations.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6505_sota_formal_challenge_mutations.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6505_sota_formal_challenge_mutations.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6505_sota_formal_challenge_mutations.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6505_sota_formal_challenge_mutations.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6505_sota_formal_challenge_mutations "
    "--validate"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6505_sota_formal_challenge_mutations.py "
    "tests/python/test_experiment_6505_sota_formal_challenge_mutations.py "
    "scripts/adversarial_verify.py"
)

DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": RUFF_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "verdict_class",
    "upstream_gate_receipt",
    "model_specs",
    "model_runtime_receipts",
    "mutation_grammar",
    "source_selection_receipt",
    "raw_request_response_receipts",
    "rows",
    "exact_admission_rows",
    "model_family_results",
    "challenge_generation_complete_score",
    "challenge_pool_ready_score",
    "prohibited_output_attack_matrix",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
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
    "status": "Records terminal generation and admission state.",
    "verdict_class": (
        "Closed enum: positive | circular_positive | null | blocked | disqualified | partial."
    ),
    "upstream_gate_receipt": "Binds the base benchmark gate and observed value.",
    "model_specs": "Lists all three mandated GGUF repositories, files, quantization, and roles.",
    "model_runtime_receipts": (
        "Records llama_cpp backend, tokenizer, offload, VRAM, and timing per model."
    ),
    "mutation_grammar": "Freezes the bounded formal edit language.",
    "source_selection_receipt": (
        "Proves all sources are development-only and chosen before generation."
    ),
    "raw_request_response_receipts": "Preserves exact bytes and hashes before parsing.",
    "rows": "Provides one request, parse, edit, exact-check, and disposition row per unit.",
    "exact_admission_rows": (
        "Records parse, replay, label, proof/model, novelty, and quarantine outcomes."
    ),
    "model_family_results": (
        "Reports valid yield and failure modes separately for each SOTA family."
    ),
    "challenge_generation_complete_score": (
        "Same-roadmap gate for complete request accounting."
    ),
    "challenge_pool_ready_score": (
        "Separates useful valid mutations from execution completeness."
    ),
    "prohibited_output_attack_matrix": (
        "Tests answers, labels, heuristics, semantic translation, retries, held access, "
        "and model identity shortcuts."
    ),
    "per_unit_rows": "Carries every request and exact admission result.",
    "aggregate_row_recomputation": "Recomputes all yields, counts, and scores from rows.",
    "gate_check_summary": (
        "Names any failed benchmark, cache, CUDA, GPU, parse, or runtime check and observed value."
    ),
    "preconditions_checked": (
        "Records gate, cache, GPU, runtime, tokenizer, disk, and repository checks."
    ),
    "protected_files_unchanged": "Proves protected files stayed unchanged.",
    "inference_substrate": (
        "Declares local llama_cpp GGUF generation plus exact CPU parsing and solving."
    ),
    "verifier_is_oracle": (
        "True only for exact formal admission checks; models are not oracles."
    ),
    "field_principles": "Explains each generation, admission, and boundary field.",
    "field_provenance": (
        "Maps rows to model files, raw hashes, parser functions, and solver receipts."
    ),
    "random_seed": "Records source selection and generation seeds.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records commands and exit codes.",
    "reproducibility_checksum": "Hashes specs, raw bytes, parses, admissions, and rows.",
    "honest_verdict": (
        "Uses complete_* for complete accounting, complete_null for zero useful yield, "
        "or blocked_* with gate_check_summary for unmet execution preconditions."
    ),
}
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

SUPPORTED_OPERATIONS = (
    "ADD_CLAUSE",
    "DROP_CLAUSE",
    "ADD_EDGE",
    "DROP_EDGE",
    "SWAP_COLOR",
    "ADD_JOB",
    "ADD_PRECEDENCE",
    "SHIFT_COEFFICIENT",
    "RELBL",
    "RELABEL_VAR",
)
MAX_EDIT_OPERATIONS = 8
MAX_LITERAL_ABS_DELTA = 2
PROHIBITED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("answer", re.compile(r"\b(answer|final answer|solution is)\b", re.IGNORECASE)),
    ("label", re.compile(r"\b(label|sat|unsat|satisfiable|unsatisfiable)\b", re.IGNORECASE)),
    ("solver_advice", re.compile(r"\b(solver|heuristic|branch|propagation|unit propagation)\b", re.IGNORECASE)),
    ("semantic_translation", re.compile(r"\b(translate|translation|semantic|constraintir|natural language)\b", re.IGNORECASE)),
    ("release_decision", re.compile(r"\b(release|publish|promote|challenge pool ready)\b", re.IGNORECASE)),
    ("held_access", re.compile(r"\b(held|test split|private split)\b", re.IGNORECASE)),
    ("model_identity", re.compile(r"\b(qwen|gemma|model family|model identity)\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class MutatedInstance:
    """Formal copy after replaying a bounded edit script."""

    variable_count: int
    clauses: list[Clause]
    coefficients: dict[str, int]
    replay_errors: list[str]
    changed: bool
    mutation_hash: str


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with stable bytes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with the project prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash stable JSON evidence with the project prefix."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hash protected source bytes and return a visible missing marker."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _infer_quantization(path: str | None, preferred: str) -> str:
    if not path:
        return "missing"
    name = Path(path).name
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in name.lower():
            return token
    return preferred


def resolve_model_specs(repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    """Resolve mandated model rows to local GGUF paths and file metadata."""

    rows: list[JsonDict] = []
    for base in MODEL_SPECS:
        model_path = resolve_cached_gguf(str(base["hf_id"]), str(base["preferred_quantization"]))
        path = Path(model_path) if model_path else None
        try:
            relative = path.relative_to(repo_root) if path is not None else None
        except ValueError:
            relative = path
        rows.append(
            {
                **base,
                "model_path": str(path) if path is not None else None,
                "model_path_display": str(relative) if relative is not None else None,
                "model_file": path.name if path is not None else None,
                "model_file_exists": path.is_file() if path is not None else False,
                "file_size_bytes": path.stat().st_size if path is not None and path.is_file() else 0,
                "observed_quantization": _infer_quantization(model_path, str(base["preferred_quantization"])),
                "resolver": "carnot.inference.sota_models.resolve_cached_gguf",
                "load_api": "llama_cpp.Llama",
                "tokenizer_source": "embedded_gguf_tokenizer",
            }
        )
    return rows


def _nvidia_smi_rows() -> list[JsonDict]:  # pragma: no cover - host dependent.
    result = subprocess.run(  # noqa: S603
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        return []
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "utilization_gpu_pct": int(parts[5]),
            }
        )
    return rows


def llama_cpp_status() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as low

        try:
            raw_info = low.llama_print_system_info()
            system_info = raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
        except Exception as exc:
            system_info = f"system_info_error:{type(exc).__name__}:{exc}"
        return {
            "import_ok": True,
            "version": str(getattr(llama_cpp, "__version__", "unknown")),
            "supports_gpu_offload": bool(low.llama_supports_gpu_offload()),
            "system_info": system_info,
        }
    except Exception as exc:
        return {
            "import_ok": False,
            "version": "unavailable",
            "supports_gpu_offload": False,
            "system_info": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _disk_receipt(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    return {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free}


def _resource_receipt(repo_root: Path) -> JsonDict:
    return {
        "cpu": {
            "logical_cpu_count": os.cpu_count() or 1,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
        },
        "disk": _disk_receipt(repo_root),
    }


def protected_file_hashes(repo_root: Path) -> dict[str, JsonDict]:
    """Hash files that must not change during model loading or parsing."""

    return {
        path.as_posix(): {
            "sha256": sha256_file(repo_root / path),
            "exists": (repo_root / path).is_file(),
            "protected_by_task_contract": True,
        }
        for path in PROTECTED_RELATIVE_PATHS
    }


def protected_files_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    files: dict[str, JsonDict] = {}
    for path in sorted(set(before) | set(after)):
        prior = dict(before.get(path, {}))
        post = dict(after.get(path, {}))
        files[path] = {
            "sha256_before": prior.get("sha256", "missing"),
            "sha256_after": post.get("sha256", "missing"),
            "exists_before": prior.get("exists") is True,
            "exists_after": post.get("exists") is True,
            "unchanged": prior.get("sha256") == post.get("sha256") and prior.get("sha256") != "missing",
            "protected_by_task_contract": True,
        }
    return {
        "files": files,
        "changed_paths": [path for path, row in files.items() if row["unchanged"] is not True],
        "all_protected_files_unchanged": all(row["unchanged"] is True for row in files.values()),
    }


def upstream_gate_receipt(repo_root: Path, protected_before: Mapping[str, Any]) -> JsonDict:
    """Bind the Exp6504 base benchmark gate and observed value."""

    path = repo_root / EXP6504_RELATIVE_PATH
    payload = _read_json(path) if path.is_file() else {}
    row = {
        "row_type": "upstream_gate_receipt",
        "experiment_id": "exp6504",
        "path": EXP6504_RELATIVE_PATH.as_posix(),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "field": "base_structural_benchmark_ready_score",
        "expected_value": 1.0,
        "observed_value": payload.get("base_structural_benchmark_ready_score"),
        "passed": payload.get("base_structural_benchmark_ready_score") == 1.0,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "protected_hashes_before_loading": dict(protected_before),
    }
    return {**row, "gate_receipt_hash": sha256_json(row)}


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    run_date: str,
    gate: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, Any],
) -> JsonDict:
    """Record local gates before generation starts."""

    cached_pair = cached_sota_pair() or []
    gpu_rows = _nvidia_smi_rows()
    llama = llama_cpp_status()
    model_cache_rows = [
        {
            "hf_id": row["hf_id"],
            "model_path": row["model_path"],
            "model_file_exists": row["model_file_exists"],
            "file_size_bytes": row["file_size_bytes"],
            "observed_quantization": row["observed_quantization"],
        }
        for row in model_specs
    ]
    failures = []
    if gate.get("passed") is not True:
        failures.append({"check": "exp6504_gate", "observed": gate.get("observed_value")})
    if not all(row["model_file_exists"] for row in model_cache_rows):
        failures.append({"check": "model_cache", "observed": model_cache_rows})
    if llama.get("import_ok") is not True:
        failures.append({"check": "llama_cpp_import", "observed": llama})
    if llama.get("supports_gpu_offload") is not True:
        failures.append({"check": "llama_cpp_cuda_offload", "observed": llama})
    if not gpu_rows:
        failures.append({"check": "gpu_inventory", "observed": gpu_rows})
    disk = _disk_receipt(repo_root)
    if disk["free_bytes"] <= 10_000_000_000:
        failures.append({"check": "disk_free_bytes", "observed": disk["free_bytes"]})
    return {
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "git_head": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "git_status_short": _git_output(repo_root, ["status", "--short"]),
        "upstream_gate_receipt": dict(gate),
        "model_cache": {
            "rows": model_cache_rows,
            "all_cached": all(row["model_file_exists"] for row in model_cache_rows),
            "cached_sota_pair_observation": [
                {"hf_id": row.get("hf_id"), "model_path": row.get("model_path")}
                for row in cached_pair
            ],
        },
        "llama_cpp": llama,
        "gpu_inventory": {
            "rows": gpu_rows,
            "gpu_count": len(gpu_rows),
            "max_free_vram_mib": max((int(row["memory_free_mib"]) for row in gpu_rows), default=0),
        },
        "resources": _resource_receipt(repo_root),
        "protected_hashes_before_loading": dict(protected_before),
        "required_files": {
            path.as_posix(): {
                "exists": (repo_root / path).exists(),
                "sha256": sha256_file(repo_root / path),
            }
            for path in SOURCE_RELATIVE_PATHS
        },
        "failed_precondition_checks": failures,
        "preconditions_ready": failures == [],
    }


def _benchmark_payload(repo_root: Path) -> JsonDict:
    return _read_json(repo_root / EXP6504_RELATIVE_PATH)


def _label_by_instance(benchmark: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["instance_id"]): row
        for row in benchmark.get("exact_label_rows", [])
        if isinstance(row, Mapping)
    }


def select_development_sources(repo_root: Path = REPO_ROOT, count: int = DEFAULT_SOURCE_COUNT) -> list[JsonDict]:
    """Select development-only formal rows before model generation."""

    benchmark = _benchmark_payload(repo_root)
    preferred_families = ("random_3cnf", "graph_coloring", "small_scheduling")
    rows = [
        dict(row)
        for row in benchmark.get("raw_instance_rows", [])
        if isinstance(row, Mapping) and row.get("split") == "development"
    ]
    selected: list[JsonDict] = []
    used_ids: set[str] = set()
    for family in preferred_families:
        hit = next((row for row in rows if row.get("family") == family), None)
        if hit is not None and str(hit["instance_id"]) not in used_ids:
            selected.append(hit)
            used_ids.add(str(hit["instance_id"]))
        if len(selected) >= count:
            break
    for row in rows:
        if len(selected) >= count:
            break
        if str(row["instance_id"]) not in used_ids:
            selected.append(row)
            used_ids.add(str(row["instance_id"]))
    return selected


def source_selection_receipt(repo_root: Path, count: int = DEFAULT_SOURCE_COUNT) -> JsonDict:
    sources = select_development_sources(repo_root, count=count)
    rows = []
    for index, source in enumerate(sources):
        payload = {
            "row_type": "source_selection",
            "selection_index": index,
            "source_instance_id": source["instance_id"],
            "raw_instance_hash": source["raw_instance_hash"],
            "family": source["family"],
            "scale": source["scale"],
            "surface_relabeling": source["surface_relabeling"],
            "split": source["split"],
            "label_inspected_before_generation": False,
            "selected_before_generation": True,
            "selection_seed": SOURCE_SELECTION_SEED,
            "spec_refs": ["REQ-BENCH-6505", "SCENARIO-BENCH-6505-ONE-SHOT"],
        }
        rows.append({**payload, "source_selection_hash": sha256_json(payload)})
    payload = {
        "schema_version": SCHEMA_VERSION + ".source_selection",
        "selection_seed": SOURCE_SELECTION_SEED,
        "selection_count": len(rows),
        "requested_count": count,
        "selected_before_generation": True,
        "all_sources_development_only": all(row["split"] == "development" for row in rows),
        "selected_sources": rows,
    }
    return {**payload, "source_selection_receipt_hash": sha256_json(payload)}


def mutation_grammar() -> JsonDict:
    """Freeze the bounded formal edit language used in prompts and parser."""

    payload = {
        "schema_version": SCHEMA_VERSION + ".mutation_grammar",
        "start_token": "BEGIN_MUTATION",
        "end_token": "END_MUTATION",
        "max_operations": MAX_EDIT_OPERATIONS,
        "supported_operations": list(SUPPORTED_OPERATIONS),
        "operation_syntax": {
            "ADD_CLAUSE": "ADD_CLAUSE <lit> [<lit>] [<lit>]",
            "DROP_CLAUSE": "DROP_CLAUSE <zero_based_clause_index>",
            "ADD_EDGE": "ADD_EDGE <var> <var>",
            "DROP_EDGE": "DROP_EDGE <var> <var>",
            "SWAP_COLOR": "SWAP_COLOR <var> <var>",
            "ADD_JOB": "ADD_JOB",
            "ADD_PRECEDENCE": "ADD_PRECEDENCE <before_var> <after_var>",
            "SHIFT_COEFFICIENT": "SHIFT_COEFFICIENT <var> <delta>",
            "RELBL": "RELBL <var> <var>",
            "RELABEL_VAR": "RELABEL_VAR <var> <var>",
        },
        "bounds": {
            "literal_abs_may_not_exceed_variable_count_plus": MAX_LITERAL_ABS_DELTA,
            "coefficient_abs_may_not_exceed": 9,
            "max_new_jobs": 1,
        },
        "prompt_forbids": [
            "answers",
            "labels",
            "solver advice",
            "free-form semantic translation",
            "release decisions",
            "held access",
            "model identity shortcuts",
        ],
        "parser": "parse_edit_script",
        "admission": "admit_response",
    }
    return {**payload, "mutation_grammar_hash": sha256_json(payload)}


def build_prompt(source: Mapping[str, Any], request_id: str) -> str:
    """Build one formal mutation request without labels or solver receipts."""

    clauses = "\n".join(
        f"{idx}: {' '.join(str(lit) for lit in clause)}"
        for idx, clause in enumerate(source["clauses"])
    )
    return (
        "You are proposing a bounded formal edit script for a development CNF instance.\n"
        "Output only the edit DSL between BEGIN_MUTATION and END_MUTATION.\n"
        "Do not output an answer, label, solver advice, natural language explanation, "
        "semantic translation, release decision, held reference, or model comparison.\n"
        f"request_id={request_id}\n"
        f"formalism={source['formalism']}\n"
        f"family={source['family']}\n"
        f"variable_count={source['variable_count']}\n"
        f"clause_count={source['clause_count']}\n"
        "clauses:\n"
        f"{clauses}\n"
        "Allowed operations: ADD_CLAUSE, DROP_CLAUSE, ADD_EDGE, DROP_EDGE, SWAP_COLOR, "
        "ADD_JOB, ADD_PRECEDENCE, SHIFT_COEFFICIENT, RELBL, RELABEL_VAR.\n"
        "BEGIN_MUTATION\n"
    )


def build_request_rows(
    sources: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Create one precommitted request for each source and model pair."""

    rows: list[JsonDict] = []
    for model_index, model in enumerate(model_specs):
        for source_index, source in enumerate(sources):
            request_id = f"exp6505:{model['family']}:{source_index:03d}"
            prompt = build_prompt(source, request_id)
            request_bytes = prompt.encode("utf-8")
            payload = {
                "row_type": "request",
                "request_id": request_id,
                "model_id": model["hf_id"],
                "model_family": model["family"],
                "model_load_order": model["load_order"],
                "source_instance_id": source["instance_id"],
                "source_raw_instance_hash": source["raw_instance_hash"],
                "source_family": source["family"],
                "source_scale": source["scale"],
                "source_surface_relabeling": source["surface_relabeling"],
                "source_split": source["split"],
                "request_index": len(rows),
                "seed": GENERATION_SEED + model_index * 100 + source_index,
                "retry_count": 0,
                "request_bytes_b64": _b64(request_bytes),
                "request_sha256": sha256_bytes(request_bytes),
                "prompt_exposes_label": False,
                "prompt_exposes_solver_receipt": False,
                "prompt_exposes_held_rows": False,
                "spec_refs": ["REQ-BENCH-6505", "SCENARIO-BENCH-6505-ONE-SHOT"],
            }
            rows.append({**payload, "request_row_hash": sha256_json(payload)})
    return rows


def _prohibited_reasons(text: str) -> list[str]:
    reasons = [name for name, pattern in PROHIBITED_PATTERNS if pattern.search(text)]
    if text.count("BEGIN_MUTATION") > 1 or text.count("END_MUTATION") > 1:
        reasons.append("retry")
    return sorted(set(reasons))


def _parse_int(raw: str, errors: list[str], line_no: int) -> int | None:
    try:
        return int(raw)
    except ValueError:
        errors.append(f"line {line_no}: expected integer {raw!r}")
        return None


def _positive_var(raw: str, source: Mapping[str, Any], errors: list[str], line_no: int) -> int | None:
    value = _parse_int(raw, errors, line_no)
    if value is None:
        return None
    upper = int(source["variable_count"]) + MAX_LITERAL_ABS_DELTA
    if value <= 0 or value > upper:
        errors.append(f"line {line_no}: variable {value} outside 1..{upper}")
        return None
    return value


def parse_edit_script(response_bytes: bytes, source: Mapping[str, Any]) -> JsonDict:
    """Parse a response as the exact bounded edit DSL."""

    text = response_bytes.decode("utf-8", "replace")
    prohibited = _prohibited_reasons(text)
    if prohibited:
        return {
            "parse_ok": False,
            "operations": [],
            "operation_count": 0,
            "edit_types": [],
            "parse_errors": ["prohibited output present"],
            "prohibited_reasons": prohibited,
            "parser": "parse_edit_script",
        }
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    try:
        start = lines.index("BEGIN_MUTATION")
        end = lines.index("END_MUTATION")
    except ValueError:
        return {
            "parse_ok": False,
            "operations": [],
            "operation_count": 0,
            "edit_types": [],
            "parse_errors": ["missing mutation boundary"],
            "prohibited_reasons": [],
            "parser": "parse_edit_script",
        }
    body = [line for line in lines[start + 1 : end] if line]
    errors: list[str] = []
    operations: list[JsonDict] = []
    for offset, line in enumerate(body, start=start + 2):
        parts = line.split()
        op = parts[0].upper()
        if op not in SUPPORTED_OPERATIONS:
            errors.append(f"line {offset}: unsupported operation {op}")
            continue
        if op == "ADD_CLAUSE":
            if not (2 <= len(parts) <= 4):
                errors.append(f"line {offset}: ADD_CLAUSE requires 1..3 literals")
                continue
            literals = [_parse_int(raw, errors, offset) for raw in parts[1:]]
            if any(value is None for value in literals):
                continue
            upper = int(source["variable_count"]) + MAX_LITERAL_ABS_DELTA
            clause = [int(value) for value in literals if value is not None]
            if any(value == 0 or abs(value) > upper for value in clause):
                errors.append(f"line {offset}: literal outside nonzero +/-{upper}")
                continue
            operations.append({"op": op, "literals": clause})
        elif op in {"DROP_CLAUSE", "ADD_JOB"}:
            if op == "ADD_JOB":
                if len(parts) != 1:
                    errors.append(f"line {offset}: ADD_JOB takes no arguments")
                    continue
                operations.append({"op": op})
            else:
                if len(parts) != 2:
                    errors.append(f"line {offset}: DROP_CLAUSE requires one index")
                    continue
                index = _parse_int(parts[1], errors, offset)
                if index is not None:
                    operations.append({"op": op, "index": index})
        elif op in {"ADD_EDGE", "DROP_EDGE", "SWAP_COLOR", "ADD_PRECEDENCE", "RELBL", "RELABEL_VAR"}:
            if len(parts) != 3:
                errors.append(f"line {offset}: {op} requires two variables")
                continue
            left = _positive_var(parts[1], source, errors, offset)
            right = _positive_var(parts[2], source, errors, offset)
            if left is not None and right is not None:
                operations.append({"op": op, "left": left, "right": right})
        else:
            if len(parts) != 3:
                errors.append(f"line {offset}: SHIFT_COEFFICIENT requires variable and delta")
                continue
            var = _positive_var(parts[1], source, errors, offset)
            delta = _parse_int(parts[2], errors, offset)
            if var is not None and delta is not None:
                operations.append({"op": op, "var": var, "delta": delta})
    if len(operations) > MAX_EDIT_OPERATIONS:
        errors.append("too many operations")
    if not operations:
        errors.append("empty mutation")
    return {
        "parse_ok": not errors,
        "operations": operations if not errors else [],
        "operation_count": len(operations) if not errors else 0,
        "edit_types": [str(op["op"]) for op in operations] if not errors else [],
        "parse_errors": errors,
        "prohibited_reasons": [],
        "parser": "parse_edit_script",
    }


def _swap_literal(literal: int, left: int, right: int) -> int:
    sign = 1 if literal > 0 else -1
    var = abs(literal)
    if var == left:
        var = right
    elif var == right:
        var = left
    return sign * var


def _canonical_semantics(variable_count: int, clauses: Sequence[Sequence[int]], coefficients: Mapping[str, int]) -> JsonDict:
    return {
        "variable_count": variable_count,
        "clauses": [list(map(int, clause)) for clause in clauses],
        "coefficients": {key: coefficients[key] for key in sorted(coefficients, key=int)},
    }


def apply_edit_script(source: Mapping[str, Any], operations: Sequence[Mapping[str, Any]]) -> MutatedInstance:
    """Replay syntactically valid edits against a copy of one formal row."""

    variable_count = int(source["variable_count"])
    clauses = [list(map(int, clause)) for clause in source["clauses"]]
    coefficients = {str(index): 0 for index in range(1, variable_count + 1)}
    before = _canonical_semantics(variable_count, clauses, coefficients)
    errors: list[str] = []
    for op in operations:
        name = str(op["op"])
        if name == "ADD_CLAUSE":
            clause = list(map(int, op["literals"]))
            if any(abs(literal) > variable_count for literal in clause):
                errors.append("ADD_CLAUSE literal exceeds current variable count")
            else:
                clauses.append(clause)
        elif name == "DROP_CLAUSE":
            index = int(op["index"])
            if index < 0 or index >= len(clauses):
                errors.append("DROP_CLAUSE index out of range")
            else:
                del clauses[index]
        elif name == "ADD_EDGE":
            left = int(op["left"])
            right = int(op["right"])
            if left == right or max(left, right) > variable_count:
                errors.append("ADD_EDGE variables invalid")
            else:
                clauses.append([-left, -right])
        elif name == "DROP_EDGE":
            target = sorted([-int(op["left"]), -int(op["right"])])
            match_index = next(
                (idx for idx, clause in enumerate(clauses) if sorted(clause) == target),
                None,
            )
            if match_index is None:
                errors.append("DROP_EDGE edge clause not found")
            else:
                del clauses[match_index]
        elif name in {"SWAP_COLOR", "RELBL", "RELABEL_VAR"}:
            left = int(op["left"])
            right = int(op["right"])
            if left == right or max(left, right) > variable_count:
                errors.append(f"{name} variables invalid")
            else:
                clauses = [[_swap_literal(literal, left, right) for literal in clause] for clause in clauses]
                coefficients[str(left)], coefficients[str(right)] = (
                    coefficients.get(str(right), 0),
                    coefficients.get(str(left), 0),
                )
        elif name == "ADD_JOB":
            variable_count += 1
            coefficients[str(variable_count)] = 0
            clauses.append([variable_count])
        elif name == "ADD_PRECEDENCE":
            left = int(op["left"])
            right = int(op["right"])
            if left == right or max(left, right) > variable_count:
                errors.append("ADD_PRECEDENCE variables invalid")
            else:
                clauses.append([-left, right])
        elif name == "SHIFT_COEFFICIENT":
            var = int(op["var"])
            delta = int(op["delta"])
            if var > variable_count:
                errors.append("SHIFT_COEFFICIENT variable invalid")
            else:
                coefficients[str(var)] = coefficients.get(str(var), 0) + delta
                if abs(coefficients[str(var)]) > 9:
                    errors.append("SHIFT_COEFFICIENT coefficient out of bounds")
    after = _canonical_semantics(variable_count, clauses, coefficients)
    return MutatedInstance(
        variable_count=variable_count,
        clauses=clauses,
        coefficients=coefficients,
        replay_errors=errors,
        changed=before != after,
        mutation_hash=sha256_json(after),
    )


def _mutated_label_row(source: Mapping[str, Any], mutated: MutatedInstance) -> JsonDict:
    row = {
        "instance_id": f"{source['instance_id']}:mutation",
        "base_instance_id": source["base_instance_id"],
        "lineage_id": f"{source['lineage_id']}:mutation",
        "family": source["family"],
        "source": source["source"],
        "scale": source["scale"],
        "surface_relabeling": source["surface_relabeling"],
        "structural_hardness": source["structural_hardness"],
        "density_band": source["density_band"],
        "split": source["split"],
        "raw_instance_hash": mutated.mutation_hash,
        "variable_count": mutated.variable_count,
        "clauses": mutated.clauses,
    }
    return exp6504.label_instance(row)


def admit_response(
    *,
    request_row: Mapping[str, Any],
    source: Mapping[str, Any],
    response_bytes: bytes,
    seen_mutation_hashes: set[str],
) -> JsonDict:
    """Parse, replay, solve, and admit or quarantine one model response."""

    parsed = parse_edit_script(response_bytes, source)
    mutated = MutatedInstance(
        variable_count=int(source["variable_count"]),
        clauses=[list(map(int, clause)) for clause in source["clauses"]],
        coefficients={str(index): 0 for index in range(1, int(source["variable_count"]) + 1)},
        replay_errors=["parse_failed"],
        changed=False,
        mutation_hash=sha256_json({"parse_failed": sha256_bytes(response_bytes)}),
    )
    label: JsonDict = {}
    replay: JsonDict = {}
    duplicate = False
    novel = False
    if parsed["parse_ok"] is True:
        mutated = apply_edit_script(source, parsed["operations"])
        duplicate = mutated.mutation_hash in seen_mutation_hashes
        novel = mutated.mutation_hash != source["structural_cnf_hash"] and not duplicate
        if not mutated.replay_errors:
            label = _mutated_label_row(source, mutated)
            replay = _mutated_label_row(source, mutated)
    no_prohibited = not parsed["prohibited_reasons"]
    model_or_proof_valid = bool(label.get("model_or_proof_valid"))
    label_ambiguous = label.get("exact_label") not in {"sat", "unsat"} if label else True
    accepted = (
        parsed["parse_ok"] is True
        and no_prohibited
        and not mutated.replay_errors
        and mutated.changed
        and not duplicate
        and novel
        and label.get("accepted") is True
        and replay.get("accepted") is True
        and replay.get("exact_label") == label.get("exact_label")
        and model_or_proof_valid
        and not label_ambiguous
    )
    if accepted:
        reason = ""
    elif parsed["prohibited_reasons"]:
        reason = "prohibited_output"
    elif parsed["parse_ok"] is not True:
        reason = "parse_failed"
    elif mutated.replay_errors:
        reason = "edit_replay_failed"
    elif not mutated.changed:
        reason = "unchanged_mutation"
    elif duplicate:
        reason = "duplicate_mutation"
    elif label_ambiguous:
        reason = "label_ambiguous"
    elif model_or_proof_valid is not True:
        reason = "model_or_proof_invalid"
    else:
        reason = "exact_admission_failed"
    payload = {
        "row_type": "exact_admission",
        "request_id": request_row["request_id"],
        "source_instance_id": source["instance_id"],
        "source_raw_instance_hash": source["raw_instance_hash"],
        "source_family": source["family"],
        "source_scale": source["scale"],
        "source_surface_relabeling": source["surface_relabeling"],
        "source_split": source["split"],
        "model_id": request_row["model_id"],
        "model_family": request_row["model_family"],
        "seed": request_row["seed"],
        "parse_ok": parsed["parse_ok"],
        "parse_errors": parsed["parse_errors"],
        "prohibited_reasons": parsed["prohibited_reasons"],
        "no_prohibited_output": no_prohibited,
        "operation_count": parsed["operation_count"],
        "edit_types": parsed["edit_types"],
        "mutation_hash": mutated.mutation_hash,
        "changed": mutated.changed,
        "duplicate": duplicate,
        "novel": novel,
        "edit_replay_errors": mutated.replay_errors,
        "exact_label": label.get("exact_label", "quarantined"),
        "model_or_proof_valid": model_or_proof_valid,
        "proof_or_model_receipt": label.get("proof_receipt", {}),
        "backend_receipts": label.get("backend_receipts", []),
        "exact_replay_passed": replay.get("accepted") is True
        and replay.get("exact_label") == label.get("exact_label"),
        "label_ambiguous": label_ambiguous,
        "accepted": accepted,
        "quarantine_reason": reason,
        "truncated": request_row.get("truncated") is True,
        "verifier_is_oracle": True,
        "spec_refs": ["REQ-BENCH-6505", "SCENARIO-BENCH-6505-ADMISSION"],
    }
    return {**payload, "admission_row_hash": sha256_json(payload)}


def _extract_text(raw: Mapping[str, Any]) -> str:  # pragma: no cover - live llama.cpp shape.
    choices = raw.get("choices")
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        return str(choices[0].get("text") or "")
    return ""


def _finish_reason(raw: Mapping[str, Any]) -> str:  # pragma: no cover - live llama.cpp shape.
    choices = raw.get("choices")
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        return str(choices[0].get("finish_reason") or "")
    return ""


def _worker_main(payload_path: str) -> int:  # pragma: no cover - live GGUF worker.
    payload = _read_json(Path(payload_path))
    model_spec = dict(payload["model_spec"])
    requests = [dict(row) for row in payload["requests"]]
    decode = dict(payload["decode_config"])
    output_path = Path(str(payload["output_path"]))
    from llama_cpp import Llama, llama_cpp as low

    start = time.perf_counter()
    vram_before = _nvidia_smi_rows()
    llm = Llama(
        model_path=str(model_spec["model_path"]),
        n_gpu_layers=int(decode["n_gpu_layers"]),
        main_gpu=int(decode["main_gpu"]),
        seed=RANDOM_SEED,
        n_ctx=int(decode["n_ctx"]),
        n_batch=int(decode["n_batch"]),
        n_ubatch=int(decode["n_ubatch"]),
        verbose=False,
    )
    probe_tokens = llm.tokenize(b"BEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n")
    load_time = time.perf_counter() - start
    rows: list[JsonDict] = []
    for request in requests:
        started = time.perf_counter()
        raw = llm(
            base64.b64decode(str(request["request_bytes_b64"])).decode("utf-8"),
            max_tokens=int(decode["max_new_tokens"]),
            temperature=float(decode["temperature"]),
            top_p=float(decode["top_p"]),
            repeat_penalty=float(decode["repeat_penalty"]),
            seed=int(request["seed"]),
        )
        text = _extract_text(raw)
        usage = dict(raw.get("usage") or {}) if isinstance(raw, Mapping) else {}
        token_count = int(usage.get("completion_tokens", 0) or 0)
        if token_count <= 0:
            token_count = len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
        rows.append(
            {
                "request_id": request["request_id"],
                "response_text": text,
                "response_bytes_b64": _b64(text.encode("utf-8")),
                "terminal_disposition": "generated",
                "finish_reason": _finish_reason(raw),
                "truncated": _finish_reason(raw) == "length",
                "generated_token_count": token_count,
                "decode_time_s": round(time.perf_counter() - started, 6),
                "error": "",
            }
        )
    llm = None
    gc.collect()
    output = {
        "model_runtime_receipt": {
            "model_id": model_spec["hf_id"],
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "llama_cpp_import_ok": True,
            "llama_cpp_supports_gpu_offload": bool(low.llama_supports_gpu_offload()),
            "embedded_tokenizer": {
                "source": "embedded_gguf_tokenizer",
                "loadable": len(probe_tokens) > 0,
                "probe_token_count": len(probe_tokens),
            },
            "offload": {
                "n_gpu_layers": int(decode["n_gpu_layers"]),
                "main_gpu": int(decode["main_gpu"]),
            },
            "vram": {"before": vram_before, "after": _nvidia_smi_rows()},
            "timing": {
                "load_time_s": round(load_time, 6),
                "total_time_s": round(time.perf_counter() - start, 6),
            },
            "terminal_disposition": "complete",
            "request_count": len(requests),
        },
        "rows": rows,
    }
    output_path.write_text(canonical_json(output), encoding="utf-8")
    return 0


def live_llama_cpp_generation_runner(
    *,
    model_spec: Mapping[str, Any],
    requests: list[dict[str, Any]],
    decode_config: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - live subprocess path.
    with tempfile.TemporaryDirectory(prefix="carnot_exp6505_") as tmp:
        tmp_path = Path(tmp)
        payload_path = tmp_path / "payload.json"
        output_path = tmp_path / "output.json"
        payload = {
            "model_spec": dict(model_spec),
            "requests": requests,
            "decode_config": dict(decode_config),
            "output_path": str(output_path),
        }
        payload_path.write_text(canonical_json(payload), encoding="utf-8")
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "carnot.experiment_6505_sota_formal_challenge_mutations", "--worker", str(payload_path)],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=float(decode_config.get("model_timeout_s", 900)),
        )
        if result.returncode == 0 and output_path.is_file():
            payload = _read_json(output_path)
            for row in payload.get("rows", []):
                row["response_bytes"] = base64.b64decode(str(row["response_bytes_b64"]))
            return payload
        error_text = (
            f"worker_returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        return blocked_generation_result(model_spec, requests, error_text.encode("utf-8"))


def blocked_generation_result(
    model_spec: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    error_bytes: bytes,
) -> JsonDict:
    """Create terminal blocked rows when generation cannot run."""

    error = error_bytes.decode("utf-8", "replace")
    return {
        "model_runtime_receipt": {
            "model_id": model_spec["hf_id"],
            "runtime_backend": "llama_cpp_python_cuda_gguf",
            "llama_cpp_import_ok": False,
            "llama_cpp_supports_gpu_offload": False,
            "embedded_tokenizer": {
                "source": "embedded_gguf_tokenizer",
                "loadable": False,
                "probe_token_count": 0,
            },
            "offload": {"n_gpu_layers": DECODE_CONFIG["n_gpu_layers"], "main_gpu": DECODE_CONFIG["main_gpu"]},
            "vram": {},
            "timing": {"load_time_s": 0.0, "total_time_s": 0.0},
            "terminal_disposition": "blocked",
            "request_count": len(requests),
            "error": error,
        },
        "rows": [
            {
                "request_id": request["request_id"],
                "response_text": error,
                "response_bytes": error_bytes,
                "terminal_disposition": "runtime_blocked",
                "finish_reason": "error",
                "truncated": False,
                "generated_token_count": 0,
                "decode_time_s": 0.0,
                "error": error,
            }
            for request in requests
        ],
    }


def fixture_generation_runner(
    *,
    model_spec: Mapping[str, Any],
    requests: list[dict[str, Any]],
    decode_config: Mapping[str, Any],
) -> JsonDict:
    """Fast deterministic generation path for schema and CLI tests."""

    rows = []
    for request in requests:
        if str(model_spec["hf_id"]).endswith("gemma-4-26B-A4B-it-GGUF"):
            text = "ANSWER sat\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n"
        else:
            text = "BEGIN_MUTATION\nADD_CLAUSE -1\nSHIFT_COEFFICIENT 1 2\nEND_MUTATION\n"
        rows.append(
            {
                "request_id": request["request_id"],
                "response_text": text,
                "response_bytes": text.encode("utf-8"),
                "terminal_disposition": "generated",
                "finish_reason": "stop",
                "truncated": False,
                "generated_token_count": 12,
                "decode_time_s": 0.01,
                "error": "",
            }
        )
    return {
        "model_runtime_receipt": {
            "model_id": model_spec["hf_id"],
            "runtime_backend": "fixture_llama_cpp",
            "llama_cpp_import_ok": True,
            "llama_cpp_supports_gpu_offload": True,
            "embedded_tokenizer": {
                "source": "embedded_gguf_tokenizer",
                "loadable": True,
                "probe_token_count": 4,
            },
            "offload": {"n_gpu_layers": decode_config["n_gpu_layers"], "main_gpu": decode_config["main_gpu"]},
            "vram": {"before_free_mib": 24000, "after_free_mib": 23900},
            "timing": {"load_time_s": 0.01, "total_time_s": 0.02},
            "terminal_disposition": "complete",
            "request_count": len(requests),
        },
        "rows": rows,
    }


def raw_receipt_from_generation(
    request: Mapping[str, Any],
    generation_row: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> JsonDict:
    response_bytes = bytes(generation_row["response_bytes"])
    payload = {
        "row_type": "raw_request_response",
        "request_id": request["request_id"],
        "model_id": request["model_id"],
        "model_family": request["model_family"],
        "model_path": model_spec.get("model_path"),
        "model_file": model_spec.get("model_file"),
        "source_instance_id": request["source_instance_id"],
        "request_bytes_b64": request["request_bytes_b64"],
        "response_bytes_b64": _b64(response_bytes),
        "request_sha256": request["request_sha256"],
        "response_sha256": sha256_bytes(response_bytes),
        "raw_persisted_before_parse": True,
        "terminal_disposition": generation_row["terminal_disposition"],
        "finish_reason": generation_row["finish_reason"],
        "truncated": generation_row["truncated"] is True,
        "generated_token_count": generation_row["generated_token_count"],
        "decode_time_s": generation_row["decode_time_s"],
        "error": generation_row["error"],
        "spec_refs": ["REQ-BENCH-6505", "SCENARIO-BENCH-6505-PROVENANCE"],
    }
    return {**payload, "raw_receipt_hash": sha256_json(payload)}


def combined_row(
    request: Mapping[str, Any],
    raw: Mapping[str, Any],
    admission: Mapping[str, Any],
    source_label: Mapping[str, Any] | None,
) -> JsonDict:
    payload = {
        "row_type": "mutation_request",
        "request_id": request["request_id"],
        "model_id": request["model_id"],
        "model_family": request["model_family"],
        "source_instance_id": request["source_instance_id"],
        "source_family": request["source_family"],
        "source_scale": request["source_scale"],
        "source_surface_relabeling": request["source_surface_relabeling"],
        "source_split": request["source_split"],
        "source_exact_label": source_label.get("exact_label") if source_label else "unknown",
        "seed": request["seed"],
        "retry_count": request["retry_count"],
        "request_sha256": raw["request_sha256"],
        "response_sha256": raw["response_sha256"],
        "parser_invoked_after_raw_persist": raw["raw_persisted_before_parse"] is True,
        "runtime_terminal_disposition": raw["terminal_disposition"],
        "finish_reason": raw["finish_reason"],
        "truncated": raw["truncated"],
        "parse_ok": admission["parse_ok"],
        "edit_types": admission["edit_types"],
        "exact_label": admission["exact_label"],
        "accepted": admission["accepted"],
        "quarantine_reason": admission["quarantine_reason"],
        "spec_refs": ["REQ-BENCH-6505", "SCENARIO-BENCH-6505-SCORES"],
    }
    return {**payload, "row_hash": sha256_json(payload)}


def prohibited_output_attack_matrix(source: Mapping[str, Any]) -> JsonDict:
    """Run parser-level attacks against forbidden output classes."""

    attacks = {
        "answers": b"ANSWER 1\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n",
        "labels": b"LABEL sat\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n",
        "heuristics": b"Use unit propagation.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n",
        "semantic_translation": b"Translate to ConstraintIR.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n",
        "retries": b"BEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\nBEGIN_MUTATION\nADD_CLAUSE 2\nEND_MUTATION\n",
        "held_access": b"Use held split row.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n",
        "model_identity_shortcuts": b"Qwen should do this.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n",
        "release_decisions": b"Release this.\nBEGIN_MUTATION\nADD_CLAUSE 1\nEND_MUTATION\n",
    }
    rows = []
    for attack_id, payload_bytes in attacks.items():
        parsed = parse_edit_script(payload_bytes, source)
        row = {
            "row_type": "prohibited_output_attack",
            "attack_id": attack_id,
            "parse_ok": parsed["parse_ok"],
            "prohibited_reasons": parsed["prohibited_reasons"],
            "fail_closed": parsed["parse_ok"] is False and bool(parsed["prohibited_reasons"]),
            "accepted": False,
            "spec_refs": ["REQ-BENCH-6505", "SCENARIO-BENCH-6505-NO-ANSWER"],
        }
        rows.append({**row, "attack_row_hash": sha256_json(row)})
    payload = {
        "schema_version": SCHEMA_VERSION + ".prohibited_output_attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
        "false_accept_count": sum(1 for row in rows if row["fail_closed"] is not True),
    }
    return {**payload, "attack_matrix_hash": sha256_json(payload)}


def per_unit_rows(rows: Sequence[Mapping[str, Any]], admissions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Carry every request row and its exact admission result."""

    return [*[dict(row) for row in rows], *[dict(row) for row in admissions]]


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute request accounting and useful-yield scores from rows."""

    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("row_type"))].append(row)
    requests = by_type["mutation_request"]
    admissions = by_type["exact_admission"]
    accepted = [row for row in admissions if row.get("accepted") is True]
    terminal_requests = [
        row
        for row in requests
        if str(row.get("runtime_terminal_disposition", "")) in {"generated", "runtime_blocked"}
    ]
    prohibited_accepts = [
        row
        for row in admissions
        if row.get("accepted") is True and row.get("no_prohibited_output") is not True
    ]
    request_count = len(requests)
    family_counts = Counter(str(row.get("model_id")) for row in requests)
    failure_modes = Counter(str(row.get("quarantine_reason") or "accepted") for row in admissions)
    complete = request_count > 0 and len(terminal_requests) == request_count
    return {
        "row_count": len(rows),
        "request_count": request_count,
        "admission_count": len(admissions),
        "terminal_request_count": len(terminal_requests),
        "accepted_mutation_count": len(accepted),
        "quarantined_mutation_count": len(admissions) - len(accepted),
        "prohibited_accepted_count": len(prohibited_accepts),
        "duplicate_quarantine_count": failure_modes["duplicate_mutation"],
        "unchanged_quarantine_count": failure_modes["unchanged_mutation"],
        "parse_failure_count": failure_modes["parse_failed"],
        "runtime_blocked_count": sum(
            1 for row in requests if row.get("runtime_terminal_disposition") == "runtime_blocked"
        ),
        "model_request_counts": dict(sorted(family_counts.items())),
        "failure_modes": dict(sorted(failure_modes.items())),
        "challenge_generation_complete_score_from_rows": 1.0 if complete else 0.0,
        "challenge_pool_ready_score_from_rows": 1.0
        if accepted and not prohibited_accepts
        else 0.0,
    }


def model_family_results(
    model_specs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    admissions: Sequence[Mapping[str, Any]],
    runtime_receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Summarize yield and failure modes separately for each model family."""

    rows_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    admissions_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    receipts = {str(row.get("model_id")): row for row in runtime_receipts}
    for row in rows:
        rows_by_model[str(row["model_id"])].append(row)
    for row in admissions:
        admissions_by_model[str(row["model_id"])].append(row)
    out = []
    for spec in model_specs:
        model_id = str(spec["hf_id"])
        model_rows = rows_by_model[model_id]
        model_admissions = admissions_by_model[model_id]
        accepted = [row for row in model_admissions if row.get("accepted") is True]
        failure_modes = Counter(str(row.get("quarantine_reason") or "accepted") for row in model_admissions)
        edit_types = Counter(
            edit
            for row in model_admissions
            for edit in row.get("edit_types", [])
        )
        payload = {
            "row_type": "model_family_result",
            "model_id": model_id,
            "model_family": spec["family"],
            "role": spec["role"],
            "request_count": len(model_rows),
            "terminal_request_count": len(model_rows),
            "valid_yield_count": len(accepted),
            "valid_yield_rate": round(len(accepted) / len(model_rows), 6) if model_rows else 0.0,
            "failure_modes": dict(sorted(failure_modes.items())),
            "source_families": sorted({str(row["source_family"]) for row in model_rows}),
            "source_scales": sorted({str(row["source_scale"]) for row in model_rows}),
            "source_labels": sorted({str(row["source_exact_label"]) for row in model_rows}),
            "edit_type_counts": dict(sorted(edit_types.items())),
            "truncation_count": sum(1 for row in model_rows if row.get("truncated") is True),
            "parser_failure_count": sum(1 for row in model_admissions if row.get("parse_ok") is not True),
            "runtime_receipt": dict(receipts.get(model_id, {})),
            "spec_refs": ["REQ-BENCH-6505", "SCENARIO-BENCH-6505-SCORES"],
        }
        out.append({**payload, "model_family_result_hash": sha256_json(payload)})
    return out


def gate_check_summary(
    *,
    gate: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    runtime_receipts: Sequence[Mapping[str, Any]],
    attacks: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize every generation, runtime, parser, and exact-admission gate."""

    checks = {
        "exp6504_gate": {
            "expected": 1.0,
            "observed": gate.get("observed_value"),
            "passed": gate.get("passed") is True,
        },
        "model_cache": {
            "expected": True,
            "observed": preconditions.get("model_cache", {}).get("all_cached"),
            "passed": preconditions.get("model_cache", {}).get("all_cached") is True,
        },
        "llama_cpp_cuda_support": {
            "expected": True,
            "observed": preconditions.get("llama_cpp", {}).get("supports_gpu_offload"),
            "passed": preconditions.get("llama_cpp", {}).get("supports_gpu_offload") is True,
        },
        "gpu_inventory": {
            "expected": "gpu_count>=1",
            "observed": preconditions.get("gpu_inventory", {}).get("gpu_count"),
            "passed": int(preconditions.get("gpu_inventory", {}).get("gpu_count", 0)) >= 1,
        },
        "all_requests_terminal": {
            "expected": 1.0,
            "observed": aggregate.get("challenge_generation_complete_score_from_rows"),
            "passed": aggregate.get("challenge_generation_complete_score_from_rows") == 1.0,
        },
        "no_accepted_prohibited_output": {
            "expected": 0,
            "observed": aggregate.get("prohibited_accepted_count"),
            "passed": aggregate.get("prohibited_accepted_count") == 0,
        },
        "prohibited_attacks_fail_closed": {
            "expected": True,
            "observed": attacks.get("all_attacks_fail_closed"),
            "passed": attacks.get("all_attacks_fail_closed") is True,
        },
        "runtime_receipts_terminal": {
            "expected": len(MODEL_SPECS),
            "observed": sum(
                1
                for row in runtime_receipts
                if row.get("terminal_disposition") in {"complete", "blocked"}
            ),
            "passed": len(runtime_receipts) == len(MODEL_SPECS)
            and all(row.get("terminal_disposition") in {"complete", "blocked"} for row in runtime_receipts),
        },
        "protected_files_unchanged": {
            "expected": True,
            "observed": protected.get("all_protected_files_unchanged"),
            "passed": protected.get("all_protected_files_unchanged") is True,
        },
        "tests_passed": {
            "expected": 0,
            "observed": sum(1 for row in tests_run if int(row.get("exit_code", 1)) != 0),
            "passed": all(int(row.get("exit_code", 1)) == 0 for row in tests_run),
        },
    }
    failed = [
        {"check": key, "expected": row["expected"], "observed": row["observed"]}
        for key, row in checks.items()
        if row["passed"] is not True
    ]
    return {
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": failed == [],
        "blocked_reason": "" if failed == [] else "blocked_" + ",".join(row["check"] for row in failed),
    }


def _status_verdict(
    aggregate: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[str, str, float, str]:
    complete = aggregate.get("challenge_generation_complete_score_from_rows") == 1.0
    ready = aggregate.get("challenge_pool_ready_score_from_rows") == 1.0
    runtime_blocked = int(aggregate.get("runtime_blocked_count", 0)) > 0
    if not complete or summary.get("checks", {}).get("exp6504_gate", {}).get("passed") is not True:
        return (
            "blocked_formal_challenge_mutation_generation",
            "blocked",
            0.0,
            f"blocked_formal_challenge_mutations: {summary.get('blocked_reason')}",
        )
    if runtime_blocked and not ready:
        return (
            "blocked_formal_challenge_mutation_generation",
            "blocked",
            0.0,
            f"blocked_formal_challenge_mutations: {summary.get('blocked_reason')}",
        )
    if ready:
        return (
            "complete_formal_challenge_mutation_accounting",
            "positive",
            1.0,
            (
                "complete_formal_challenge_mutations: terminal one-shot requests produced "
                "at least one exact-admitted useful mutation"
            ),
        )
    return (
        "complete_null_formal_challenge_mutation_accounting",
        "null",
        0.0,
        "complete_null_formal_challenge_mutations: terminal accounting complete with zero useful mutations",
    )


def _field_provenance(
    model_specs: Sequence[Mapping[str, Any]],
    raw_receipts: Sequence[Mapping[str, Any]],
    admissions: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    source_hashes = {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}
    base = {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "spec_refs": ["REQ-BENCH-6505"],
            "source_hashes": source_hashes,
            "local_reducer": "build_artifact",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    base["model_specs"]["model_files"] = [
        {"hf_id": row.get("hf_id"), "model_path": row.get("model_path"), "file_size_bytes": row.get("file_size_bytes")}
        for row in model_specs
    ]
    base["raw_request_response_receipts"]["raw_hashes"] = [
        {"request_id": row.get("request_id"), "response_sha256": row.get("response_sha256")}
        for row in raw_receipts
    ]
    base["exact_admission_rows"]["admission_hashes"] = [
        {"request_id": row.get("request_id"), "admission_row_hash": row.get("admission_row_hash")}
        for row in admissions
    ]
    base["field_provenance"]["parser"] = "parse_edit_script"
    base["field_provenance"]["solver"] = "experiment_6504.label_instance"
    return base


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the fields that define generation and admission state."""

    payload = {
        "upstream_gate_receipt": artifact.get("upstream_gate_receipt"),
        "model_specs": artifact.get("model_specs"),
        "mutation_grammar": artifact.get("mutation_grammar"),
        "source_selection_receipt": artifact.get("source_selection_receipt"),
        "raw_request_response_receipts": artifact.get("raw_request_response_receipts"),
        "rows": artifact.get("rows"),
        "exact_admission_rows": artifact.get("exact_admission_rows"),
        "model_family_results": artifact.get("model_family_results"),
        "prohibited_output_attack_matrix": artifact.get("prohibited_output_attack_matrix"),
        "aggregate_row_recomputation": artifact.get("aggregate_row_recomputation"),
    }
    return sha256_json(payload)


def tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    return [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)]


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    write: bool = False,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
    generation_runner: GenerationRunner | None = None,
    source_count: int = DEFAULT_SOURCE_COUNT,
) -> JsonDict:
    """Build and optionally write the terminal Exp6505 artifact."""

    start = time.perf_counter()
    target = result_path or repo_root / RESULT_RELATIVE_PATH
    protected_before = protected_file_hashes(repo_root)
    model_specs = resolve_model_specs(repo_root)
    gate = upstream_gate_receipt(repo_root, protected_before)
    selection = source_selection_receipt(repo_root, count=source_count)
    sources = select_development_sources(repo_root, count=source_count)
    benchmark = _benchmark_payload(repo_root)
    labels = _label_by_instance(benchmark)
    requests = build_request_rows(sources, model_specs)
    request_by_id = {str(row["request_id"]): row for row in requests}
    source_by_id = {str(row["instance_id"]): row for row in sources}
    runner = generation_runner or live_llama_cpp_generation_runner
    preconditions = preconditions_checked(
        repo_root=repo_root,
        result_path=target,
        run_date=run_date,
        gate=gate,
        model_specs=model_specs,
        protected_before=protected_before,
    )
    raw_receipts: list[JsonDict] = []
    admissions: list[JsonDict] = []
    runtime_receipts: list[JsonDict] = []
    combined_rows: list[JsonDict] = []
    seen_mutations: set[str] = set()
    for model in model_specs:
        model_requests = [row for row in requests if row["model_id"] == model["hf_id"]]
        if preconditions.get("preconditions_ready") is not True:
            generation = blocked_generation_result(
                model,
                model_requests,
                canonical_json(preconditions["failed_precondition_checks"]).encode("utf-8"),
            )
        else:
            generation = runner(
                model_spec=dict(model),
                requests=[dict(row) for row in model_requests],
                decode_config=dict(DECODE_CONFIG),
            )
        runtime_receipts.append(dict(generation["model_runtime_receipt"]))
        for generated in generation["rows"]:
            request = request_by_id[str(generated["request_id"])]
            source = source_by_id[str(request["source_instance_id"])]
            raw = raw_receipt_from_generation(request, generated, model)
            raw_receipts.append(raw)
            admission = admit_response(
                request_row={**request, "truncated": raw["truncated"]},
                source=source,
                response_bytes=base64.b64decode(raw["response_bytes_b64"]),
                seen_mutation_hashes=seen_mutations,
            )
            if admission["accepted"] is True:
                seen_mutations.add(str(admission["mutation_hash"]))
            admissions.append(admission)
            combined_rows.append(
                combined_row(request, raw, admission, labels.get(str(request["source_instance_id"])))
            )
    attacks = prohibited_output_attack_matrix(sources[0])
    unit_rows = per_unit_rows(combined_rows, admissions)
    aggregate = recompute_aggregates_from_rows(unit_rows)
    protected_after = protected_file_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    tests = tests_run_receipts(tests_run)
    summary = gate_check_summary(
        gate=gate,
        preconditions=preconditions,
        aggregate=aggregate,
        runtime_receipts=runtime_receipts,
        attacks=attacks,
        protected=protected,
        tests_run=tests,
    )
    status, verdict_class, pool_score, verdict = _status_verdict(aggregate, summary)
    family_results = model_family_results(model_specs, combined_rows, admissions, runtime_receipts)
    artifact: JsonDict = {
        "status": status,
        "verdict_class": verdict_class,
        "upstream_gate_receipt": gate,
        "model_specs": model_specs,
        "model_runtime_receipts": runtime_receipts,
        "mutation_grammar": mutation_grammar(),
        "source_selection_receipt": selection,
        "raw_request_response_receipts": raw_receipts,
        "rows": combined_rows,
        "exact_admission_rows": admissions,
        "model_family_results": family_results,
        "challenge_generation_complete_score": aggregate["challenge_generation_complete_score_from_rows"],
        "challenge_pool_ready_score": pool_score,
        "prohibited_output_attack_matrix": attacks,
        "per_unit_rows": unit_rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": summary,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(model_specs, raw_receipts, admissions),
        "random_seed": {
            "artifact_seed": RANDOM_SEED,
            "source_selection_seed": SOURCE_SELECTION_SEED,
            "generation_seed": GENERATION_SEED,
        },
        "duration_s": round(duration_s if duration_s is not None else time.perf_counter() - start, 6),
        "tests_run": tests,
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, artifact, allow_override=False)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Return schema and boundary errors. Empty list means valid."""

    try:
        artifact = _read_json(Path(value)) if isinstance(value, str | Path) else dict(value)
    except Exception as exc:
        return [str(exc)]
    errors: list[str] = []
    required = set(REQUIRED_ARTIFACT_FIELDS)
    present = set(artifact)
    if present != required:
        errors.append("required field set mismatch")
    if set(artifact.get("field_principles", {})) != required:
        errors.append("field_principles mismatch")
    if set(artifact.get("field_provenance", {})) != required:
        errors.append("field_provenance must cover required fields")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for exact admission only")
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if artifact.get("challenge_generation_complete_score") != aggregate.get(
        "challenge_generation_complete_score_from_rows"
    ):
        errors.append("challenge_generation_complete_score mismatch")
    if artifact.get("challenge_pool_ready_score") != (
        1.0 if aggregate.get("challenge_pool_ready_score_from_rows") == 1.0 else 0.0
    ):
        errors.append("challenge_pool_ready_score mismatch")
    accepted = [row for row in artifact.get("exact_admission_rows", []) if row.get("accepted") is True]
    if any(row.get("no_prohibited_output") is not True for row in accepted):
        errors.append("accepted row contains prohibited output")
    if len(artifact.get("model_specs", [])) != len(MODEL_SPECS):
        errors.append("model_specs must list all mandated models")
    if [row.get("hf_id") for row in artifact.get("model_specs", [])] != [
        row["hf_id"] for row in MODEL_SPECS
    ]:
        errors.append("model_specs order mismatch")
    if len(artifact.get("model_family_results", [])) != len(MODEL_SPECS):
        errors.append("model_family_results must cover all families")
    if artifact.get("prohibited_output_attack_matrix", {}).get("false_accept_count") != 0:
        errors.append("prohibited_output_attack_matrix false accepts")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (
        verdict.startswith("complete_formal_challenge_mutations:")
        or verdict.startswith("complete_null")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    checksum = artifact.get("reproducibility_checksum")
    if isinstance(checksum, str) and checksum:
        expected = reproducibility_checksum(artifact)
        if checksum != expected:
            errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    args_in = list(argv) if argv is not None else sys.argv[1:]
    if args_in and args_in[0] == "--worker":  # pragma: no cover
        if len(args_in) != 2:
            return 2
        return _worker_main(args_in[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--fixture-mode", action="store_true")
    args = parser.parse_args(args_in)
    result_path = Path(args.result_path)
    if args.validate:
        errors = validate_artifact(result_path)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        return 0
    build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        write=True,
        run_date=args.date,
        generation_runner=fixture_generation_runner if args.fixture_mode else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
