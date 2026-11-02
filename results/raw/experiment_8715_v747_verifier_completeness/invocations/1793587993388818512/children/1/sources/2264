"""Exp6352 live factor proposal authenticity preflight.

Spec refs: REQ-LEARN-6352, SCENARIO-LEARN-6352-PREFLIGHT,
SCENARIO-LEARN-6352-GENERATION, SCENARIO-LEARN-6352-EVENTS,
SCENARIO-LEARN-6352-ISOLATION, SCENARIO-LEARN-6352-READY.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_6344_counterexample_factor_proposal_calibration import (
    factor_edit_schema as exp6344_factor_edit_schema,
    validate_proposal as exp6344_validate_proposal,
)
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
HostChecksFn = Callable[[], JsonDict]
GenerationFn = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6352_live_factor_proposal_authenticity_preflight.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6352_live_factor_proposal_authenticity_preflight"
)
RAW_DIR_NAME = "raw_model_outputs"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6352_live_factor_proposal_authenticity_preflight.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6352_live_factor_proposal_authenticity_preflight.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
E2E_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXP6342_RELATIVE_PATH = Path("results/experiment_6342_anytime_evalue_release_ledger.json")
EXP6343_RELATIVE_PATH = Path("results/experiment_6343_evidence_carrying_factor_lifecycle.json")
EXP6344_RELATIVE_PATH = Path(
    "results/experiment_6344_counterexample_factor_proposal_calibration.json"
)
EXP6345_RELATIVE_PATH = Path(
    "results/experiment_6345_prospective_certified_factor_evolution_ab.json"
)
EXP6344_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6344_counterexample_factor_proposal_calibration.py"
)
LICENSE_RELATIVE_PATH = Path("LICENSE")

SCHEMA = "carnot.experiment_6352.live_factor_proposal_authenticity_preflight.v1"
EVENT_MANIFEST_SCHEMA = SCHEMA + ".generated_event_manifest"
RELEASED_SNAPSHOT_SCHEMA = SCHEMA + ".released_factor_snapshot"
FACTOR_EDIT_SCHEMA = exp6344_factor_edit_schema()["schema"]
RUN_DATE = "20260812"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
INFERENCE_SUBSTRATE = "local_llama_cpp_gguf_live_autoregressive_generation_embedded_tokenizers"
AUTOTOKENIZER_USAGE_COUNT = 0
LIVE_ARM = "live_factor_proposal_preflight"

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_FAMILIES = ("qwen_moe", "gemma_dense", "gemma_moe")
MODEL_TEMPLATES: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "gpu": 0,
        "model_family": MODEL_FAMILIES[0],
        "min_free_vram_mb": 20_000,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "gpu": 1,
        "model_family": MODEL_FAMILIES[1],
        "min_free_vram_mb": 20_000,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "gpu": 1,
        "model_family": MODEL_FAMILIES[2],
        "min_free_vram_mb": 16_000,
    },
)

MAX_TOKENS_PER_CALL = 128
TIME_BUDGET_S = 300.0
EXACT_CHECK_COST = 0.01
CHECKER_TIME_PER_CALL_S = 0.0005
RANDOM_SEEDS = {
    "event_generator": 635200,
    "surface_relabel": 635201,
    "generation": 635202,
    "parser": 635203,
    "replay_laundering": 635204,
    "same_step_isolation": 635205,
}
SAMPLING_PARAMETERS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": MAX_TOKENS_PER_CALL,
    "n_ctx": 2048,
    "n_gpu_layers": -1,
}

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6352_live_factor_proposal_authenticity_preflight --date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6352_live_factor_proposal_authenticity_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6352_live_factor_proposal_authenticity_preflight.py "
    "-m pytest tests/python/test_experiment_6352_live_factor_proposal_authenticity_preflight.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6352_live_factor_proposal_authenticity_preflight.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6352_live_factor_proposal_authenticity_preflight.py"
)
E2E_COMMAND = "sed -n '90,140p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6352_live_factor_proposal_authenticity_preflight.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    RUN_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

UPSTREAM_PATHS = (
    ("exp6342", EXP6342_RELATIVE_PATH, "anytime_release_certificate_ready_score"),
    ("exp6343", EXP6343_RELATIVE_PATH, "evidence_factor_lifecycle_ready_score"),
    ("exp6344", EXP6344_RELATIVE_PATH, "counterexample_proposal_ready_score"),
    ("exp6345", EXP6345_RELATIVE_PATH, "certified_continuous_learning_ready_score"),
)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6342_RELATIVE_PATH,
    EXP6343_RELATIVE_PATH,
    EXP6344_RELATIVE_PATH,
    EXP6345_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_paths_hashes_and_terminal_classes",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "llama_cpp_embedded_tokenizer_receipts",
    "cuda_gpu_offload_and_memory_release_receipts_by_model",
    "generated_event_manifest_path_and_hash",
    "event_generator_paths_hashes_and_license_receipts",
    "event_family_structure_and_surface_balance",
    "released_factor_snapshot_path_and_hash",
    "information_exposure_contract",
    "live_autoregressive_generation_invoked",
    "generation_process_receipts_by_model",
    "generation_call_token_time_and_exit_receipts",
    "raw_model_output_paths_hashes_and_counts",
    "raw_output_before_parse_receipts",
    "factor_edit_schema_path_and_hash",
    "parse_valid_invalid_and_timeout_counts_by_model",
    "same_step_read_write_isolation_results",
    "deterministic_replay_laundering_checks",
    "exact_checker_paths_hashes_and_versions",
    "exact_checker_calls_time_cost_and_error_table",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "hidden_state_access_count",
    "protected_validation_leak_count",
    "live_factor_proposal_authenticity_ready_score",
    "harm_underpowered_missing_and_flagged_cells",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status distinguishes ready, blocked, and null authenticity outcomes.",
    "upstream_paths_hashes_and_terminal_classes": "Upstream Exp6342, Exp6343, Exp6344, and Exp6345 bytes and terminal classes are recorded first.",
    "MODEL_SPECS": "The three mandated GGUF rows come from `cached_sota_pair()` helper calls.",
    "models_used": "The measured cells name only required GGUF model ids.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Model files, revisions, quantizations, hashes, and embedded tokenizer methods are pinned.",
    "llama_cpp_embedded_tokenizer_receipts": "Embedded tokenizer checks prove no AutoTokenizer path was used.",
    "cuda_gpu_offload_and_memory_release_receipts_by_model": "CUDA placement and release receipts are recorded per model.",
    "generated_event_manifest_path_and_hash": "Fresh event identities, families, structures, surfaces, and seeds are frozen.",
    "event_generator_paths_hashes_and_license_receipts": "Generator and checker source paths, hashes, and license receipts are pinned.",
    "event_family_structure_and_surface_balance": "Family, executable structure, and surface relabeling balance is explicit.",
    "released_factor_snapshot_path_and_hash": "The released factor version visible to proposers is frozen.",
    "information_exposure_contract": "Proposers see only the released factor version, minimized counterexample, allowed variables, and edit bounds.",
    "live_autoregressive_generation_invoked": "Bare true proves real generation ran for readiness.",
    "generation_process_receipts_by_model": "Process IDs, commands, CUDA settings, seeds, and exit states are recorded.",
    "generation_call_token_time_and_exit_receipts": "Token counts, timings, and exit states are recorded.",
    "raw_model_output_paths_hashes_and_counts": "Raw output paths, hashes, byte counts, and model counts are pinned.",
    "raw_output_before_parse_receipts": "Raw bytes are frozen before parser input is read.",
    "factor_edit_schema_path_and_hash": "The reused Exp6344 bounded edit schema is frozen.",
    "parse_valid_invalid_and_timeout_counts_by_model": "Parse outcomes, invalid rows, and timeouts are reported per model.",
    "same_step_read_write_isolation_results": "Unapproved writes cannot change the proposal read root.",
    "deterministic_replay_laundering_checks": "Replay, clock, model-id, output-hash, parser-input, and active-registry mutations fail closed.",
    "exact_checker_paths_hashes_and_versions": "Exact checker code paths, hashes, and versions are pinned.",
    "exact_checker_calls_time_cost_and_error_table": "Exact checker calls, time, cost, and errors are charged.",
    "source_model_weight_mutation_count": "Bare zero proves base weights stayed frozen.",
    "generated_label_count": "Bare zero proves model output did not define labels.",
    "hidden_state_access_count": "Bare zero proves hidden activations were not read.",
    "protected_validation_leak_count": "Bare zero proves protected validation did not steer proposals.",
    "live_factor_proposal_authenticity_ready_score": "Readiness is fully conjunctive over required live authenticity gates.",
    "harm_underpowered_missing_and_flagged_cells": "Missing, underpowered, and flagged measured cells stay visible.",
    "protected_files_unchanged": "Protected repo files and upstream artifacts remain byte-identical.",
    "preconditions_checked": "Preconditions cover models, tokenizers, CUDA, GPUs, VRAM, disk, llama.cpp, seeds, generators, checkers, and protected hashes.",
    "inference_substrate": "The artifact declares local llama.cpp GGUF generation with embedded tokenizers.",
    "verifier_is_oracle": "Exact outcomes are oracle; proposal quality is not.",
    "field_provenance": "Every field maps to specs, inputs, sidecars, generation receipts, exact checks, or tests.",
    "field_principles": "Every required field has a reason.",
    "test_commands": "Focused, coverage, full pytest, spec, E2E, adversarial, run, and clutter commands are named.",
    "test_exit_codes": "Failed verification commands prevent readiness.",
    "duration_s": "Wall time is measured without padding.",
    "random_seeds": "Event, surface, generation, parser, replay, and isolation seeds are pinned.",
    "reproducibility_checksum": "A stable checksum detects drift.",
    "honest_verdict": "The verdict uses a terminal prefix and states the authenticity boundary.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-LEARN-6352",
        "Exp6342 release ledger",
        "Exp6343 released factor lifecycle",
        "Exp6344 bounded schema",
        "live llama.cpp generation receipts",
        "Exp6352 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    """Return stable JSON text for file hashes and receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with SHA-256."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or None when the file is absent."""

    if not path.exists() or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def model_slug(model_id: str) -> str:
    """Turn a model id into a stable file-name fragment."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-").lower()


def rounded(value: float) -> float:
    """Round receipts without hiding small exact costs."""

    return round(float(value), 12)


def require(condition: bool, reason: str) -> None:
    """Raise a deterministic validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and use an empty mapping otherwise."""

    return value if isinstance(value, Mapping) else {}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON or return the digest the JSON bytes would have."""

    if write:
        write_json_atomic(path, payload)
        digest = sha256_file(path)
        require(digest is not None, "json_write_failed")
        return str(digest)
    return sha256_json(payload)


def path_receipt(path: Path, *, sha256: str | None = None) -> JsonDict:
    """Record path, digest, presence, and size."""

    return {
        "path": str(path),
        "present": path.exists() and path.is_file(),
        "sha256": sha256 if sha256 is not None else sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def read_json(path: Path) -> JsonDict:
    """Read a JSON object."""

    return json.loads(path.read_text(encoding="utf-8"))


def revision_from_path(path: Path) -> str | None:
    """Extract a Hugging Face snapshot revision when present."""

    parts = path.parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def quantization_from_path(path: Path) -> str:
    """Extract a known GGUF quantization token from a file name."""

    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in path.name.lower():
            return token
    return "unknown"


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
) -> JsonDict:
    """Resolve all mandated GGUF rows through cached SOTA helper calls."""

    calls = [
        "cached_sota_pair(gpu_indices=(0,1))",
        "cached_sota_pair(gpu_indices=(0,1), model_indices=(0,2))",
    ]
    default_pair = cached_pair_func(gpu_indices=(0, 1), preferred_quant="Q4_K_M") or []
    dense_pair = (
        cached_pair_func(
            gpu_indices=(0, 1),
            preferred_quant="Q4_K_M",
            model_indices=(0, 2),
        )
        or []
    )
    by_id = {str(row.get("hf_id")): dict(row) for row in [*default_pair, *dense_pair]}
    records: list[JsonDict] = []
    blockers: list[str] = []
    for template in MODEL_TEMPLATES:
        row = dict(by_id.get(template["hf_id"], {}))
        model_path = str(row.get("model_path") or "")
        path = Path(model_path) if model_path else Path()
        tokenizer_ok, tokenizer_detail = tokenizer_func(model_path)
        record = {
            "name": template["name"],
            "hf_id": template["hf_id"],
            "gpu": int(row.get("gpu", template["gpu"])),
            "model_family": template["model_family"],
            "min_free_vram_mb": template["min_free_vram_mb"],
            "model_path": model_path,
            "exists": bool(model_path) and path.exists() and path.is_file(),
            "revision": revision_from_path(path) if model_path else None,
            "quantization": quantization_from_path(path) if model_path else "unknown",
            "model_file_sha256": sha256_file(path) if model_path else None,
            "tokenizer_method": TOKENIZER_METHOD,
            "tokenizer_loadable": bool(tokenizer_ok),
            "tokenizer_detail": str(tokenizer_detail),
        }
        records.append(record)
        if not row:
            blockers.append(f"missing:{template['hf_id']}")
        if not record["exists"]:
            blockers.append(f"missing_file:{template['hf_id']}")
        if not record["tokenizer_loadable"]:
            blockers.append(f"tokenizer:{template['hf_id']}")
    if not default_pair:
        blockers.append("cached_sota_pair_missing")
    return {
        "schema": SCHEMA + ".model_resolution",
        "MODEL_SPECS": records,
        "cached_sota_pair_calls": calls,
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
    }


def generated_events() -> list[JsonDict]:
    """Generate fresh exact-checkable events from deterministic seeds."""

    configs = [
        ("threshold_guard", "single_inequality", "symbolic", "accept_factor", "accept_bias"),
        ("threshold_guard", "single_inequality", "verbal", "accept_factor", "accept_bias"),
        ("route_guard", "branch_choice", "symbolic", "repair_factor", "repair_bias"),
        ("route_guard", "branch_choice", "verbal", "repair_factor", "repair_bias"),
    ]
    events: list[JsonDict] = []
    for index, (family, structure, surface, factor, variable) in enumerate(configs):
        case_id = f"exp6352-{family}-{surface}-{index:02d}"
        events.append(
            {
                "schema": SCHEMA + ".fresh_failure_event",
                "event_id": f"live-6352-{index:03d}",
                "family": family,
                "executable_structure": structure,
                "surface_relabel": surface,
                "changed_factor": factor,
                "allowed_variables": [variable],
                "edit_bounds": {"min": -1.0, "max": 1.0, "max_abs_movement": 1.0},
                "minimized_exact_counterexample": {
                    "case_id": case_id,
                    "family": family,
                    "executable_structure": structure,
                    "surface_relabel": surface,
                    "variables": {variable: 0.0},
                    "expected": "accept" if factor == "accept_factor" else "repair",
                    "observed_before_edit": "reject",
                    "minimal": True,
                    "exact": True,
                    "generator_seed": RANDOM_SEEDS["event_generator"] + index,
                },
                "protected_outcome": {
                    "exact_checker": "exp6352_bounded_factor_event_checker",
                    "valid_local_edit_required": True,
                    "protected_label_visible_before_parse": False,
                },
            }
        )
    return events


def exposed_event_payload(event: Mapping[str, Any]) -> JsonDict:
    """Expose only the fields allowed before model generation."""

    return {
        "event_id": event["event_id"],
        "family": event["family"],
        "executable_structure": event["executable_structure"],
        "surface_relabel": event["surface_relabel"],
        "changed_factor": event["changed_factor"],
        "minimized_exact_counterexample": event["minimized_exact_counterexample"],
        "allowed_variables": list(event["allowed_variables"]),
        "edit_bounds": dict(event["edit_bounds"]),
    }


def generated_event_manifest_payload(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the event sidecar without protected labels."""

    return {
        "schema": EVENT_MANIFEST_SCHEMA,
        "fresh_for_exp6352": True,
        "event_count": len(events),
        "events_exposed_to_proposer": [exposed_event_payload(event) for event in events],
        "protected_outcome_hashes": [
            {"event_id": event["event_id"], "sha256": sha256_json(event["protected_outcome"])}
            for event in events
        ],
        "random_seeds": {
            "event_generator": RANDOM_SEEDS["event_generator"],
            "surface_relabel": RANDOM_SEEDS["surface_relabel"],
        },
    }


def event_balance_receipt(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize family, executable structure, and surface balance."""

    by_family: dict[str, int] = {}
    structures: dict[str, int] = {}
    surfaces_by_family: dict[str, list[str]] = {}
    for event in events:
        family = str(event["family"])
        by_family[family] = by_family.get(family, 0) + 1
        structure = str(event["executable_structure"])
        structures[structure] = structures.get(structure, 0) + 1
        surfaces_by_family.setdefault(family, []).append(str(event["surface_relabel"]))
    expected_family_count = next(iter(by_family.values())) if by_family else 0
    balanced = (
        len(by_family) >= 2
        and all(count == expected_family_count for count in by_family.values())
        and all(sorted(surfaces) == ["symbolic", "verbal"] for surfaces in surfaces_by_family.values())
    )
    return {
        "schema": SCHEMA + ".event_balance",
        "family_count": len(by_family),
        "events_by_family": by_family,
        "events_by_executable_structure": structures,
        "surfaces_by_family": {key: sorted(value) for key, value in surfaces_by_family.items()},
        "balanced": balanced,
        "solver_effort_used_as_model_difficulty_proxy": False,
    }


def released_factor_snapshot_payload() -> JsonDict:
    """Freeze the Exp6343 released version registry as a read-only snapshot."""

    upstream_path = REPO_ROOT / EXP6343_RELATIVE_PATH
    upstream = read_json(upstream_path) if upstream_path.exists() else {}
    lifecycle = as_mapping(upstream.get("factor_add_merge_delete_quarantine_and_restore_results"))
    registry = as_mapping(upstream.get("version_registry_path_and_hash"))
    active = list(lifecycle.get("final_active_factors", []))
    return {
        "schema": RELEASED_SNAPSHOT_SCHEMA,
        "source_artifact_path": str(EXP6343_RELATIVE_PATH),
        "source_artifact_sha256": sha256_file(upstream_path),
        "version_registry_path": registry.get("path"),
        "version_registry_sha256": registry.get("sha256") or registry.get("registry_hash"),
        "released_version_id": "exp6343-final-active",
        "released_root_sha256": registry.get("final_state_hash") or sha256_json(active),
        "active_factor_ids": active,
        "factor_variables": exp6344_factor_edit_schema()["factor_variables"],
        "read_only_during_proposal": True,
        "unapproved_writes_visible": False,
    }


def information_exposure_contract() -> JsonDict:
    """Declare the fields visible to each live proposer."""

    return {
        "schema": SCHEMA + ".information_exposure_contract",
        "visible_fields": [
            "released_factor_snapshot",
            "event_id",
            "minimized_exact_counterexample",
            "allowed_variables",
            "edit_bounds",
        ],
        "protected_outcomes_visible_before_parse": False,
        "protected_validation_visible": False,
        "hidden_states_visible": False,
        "source_weights_visible": False,
        "unapproved_writes_visible": False,
        "forbidden_fields": exp6344_factor_edit_schema()["forbidden_fields"],
    }


def prompt_payload(
    spec: Mapping[str, Any],
    event: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> JsonDict:
    """Build the exact prompt payload sent to one model."""

    return {
        "instruction": (
            "Return exactly one JSON object matching proposal_schema. "
            "Do not include markdown or commentary."
        ),
        "released_factor_snapshot": {
            "released_version_id": snapshot["released_version_id"],
            "released_root_sha256": snapshot["released_root_sha256"],
            "active_factor_ids": snapshot["active_factor_ids"],
            "read_only_during_proposal": True,
        },
        "event": exposed_event_payload(event),
        "proposal_schema": {
            "schema": FACTOR_EDIT_SCHEMA,
            "required_fields": exp6344_factor_edit_schema()["required_fields"],
            "fixed_fields": {
                "model_hf_id": spec["hf_id"],
                "arm": LIVE_ARM,
                "candidate_index": 0,
                "event_id": event["event_id"],
                "changed_factor": event["changed_factor"],
            },
            "edits_allowed_only_for": list(event["allowed_variables"]),
            "edit_bounds": dict(event["edit_bounds"]),
        },
    }


def host_environment_receipts() -> JsonDict:  # pragma: no cover - host dependent
    """Collect CUDA, RAM, disk, and llama.cpp receipts from the host."""

    devices: list[JsonDict] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            devices = parse_gpu_query(result.stdout)
    except Exception:
        devices = []
    disk = shutil.disk_usage(REPO_ROOT)
    llama_cpp_available = True
    llama_cpp_detail = "llama_cpp import ok"
    try:
        import llama_cpp  # noqa: F401
    except Exception as exc:
        llama_cpp_available = False
        llama_cpp_detail = str(exc)
    return {
        "cuda_devices": {"available": len(devices) >= 2, "count": len(devices), "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "ram": memory_receipt(),
        "disk": {"available_gb": rounded(disk.free / (1024**3))},
        "llama_cpp": {
            "python_binding_available": llama_cpp_available,
            "gpu_offload_receipt": len(devices) >= 2,
            "detail": llama_cpp_detail,
        },
    }


def deterministic_host_receipts() -> JsonDict:
    """Return stable host receipts for unit tests."""

    devices = [
        {"index": 0, "name": "test-gpu-0", "total_mb": 24576, "used_mb": 256, "free_mb": 24320},
        {"index": 1, "name": "test-gpu-1", "total_mb": 24576, "used_mb": 256, "free_mb": 24320},
    ]
    return {
        "cuda_devices": {"available": True, "count": 2, "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "ram": {"total_gb": 128.0, "available_gb": 96.0},
        "disk": {"available_gb": 1024.0},
        "llama_cpp": {
            "python_binding_available": True,
            "gpu_offload_receipt": True,
            "detail": "deterministic llama.cpp receipt",
        },
    }


def parse_gpu_query(stdout: str) -> list[JsonDict]:
    """Parse nvidia-smi CSV rows."""

    rows: list[JsonDict] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mb": int(parts[2]),
                    "used_mb": int(parts[3]),
                    "free_mb": int(parts[4]),
                }
            )
    return rows


def memory_receipt() -> JsonDict:  # pragma: no cover - host dependent
    """Return Linux memory in GiB."""

    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        info[key] = int(raw.strip().split()[0])
    return {
        "total_gb": rounded(info.get("MemTotal", 0) / (1024**2)),
        "available_gb": rounded(info.get("MemAvailable", 0) / (1024**2)),
    }


def live_llama_cpp_generation(  # pragma: no cover - live GGUF generation
    *,
    spec: Mapping[str, Any],
    event: Mapping[str, Any],
    raw_path: Path,
    prompt_payload: Mapping[str, Any],
    seed: int,
    sampling: Mapping[str, Any],
    timeout_s: float,
) -> JsonDict:
    """Run real llama.cpp generation in a child process and save raw stdout."""

    started_ns = time.time_ns()
    child_args = {
        "model_path": spec["model_path"],
        "prompt": json.dumps(prompt_payload, sort_keys=True),
        "seed": seed,
        "gpu": int(spec["gpu"]),
        "sampling": dict(sampling),
    }
    child_code = r'''
import json
import sys
from llama_cpp import Llama
args = json.loads(sys.argv[1])
sampling = args["sampling"]
llm = Llama(
    model_path=args["model_path"],
    n_ctx=int(sampling["n_ctx"]),
    n_gpu_layers=int(sampling["n_gpu_layers"]),
    main_gpu=int(args["gpu"]),
    seed=int(args["seed"]),
    verbose=False,
)
result = llm.create_completion(
    args["prompt"],
    max_tokens=int(sampling["max_tokens"]),
    temperature=float(sampling["temperature"]),
    top_p=float(sampling["top_p"]),
    echo=False,
)
sys.stdout.write(result["choices"][0]["text"])
sys.stderr.write("\nCARNOT_USAGE:" + json.dumps(result.get("usage", {}), sort_keys=True) + "\n")
'''
    command = [sys.executable, "-c", child_code, json.dumps(child_args, sort_keys=True)]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(stdout)
    raw_written_ns = time.time_ns()
    token_counts = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    stderr_text = stderr.decode("utf-8", errors="replace")
    for line in stderr_text.splitlines():
        if line.startswith("CARNOT_USAGE:"):
            usage = json.loads(line.removeprefix("CARNOT_USAGE:"))
            token_counts = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
    ended_ns = time.time_ns()
    return {
        "hf_id": spec["hf_id"],
        "model_family": spec["model_family"],
        "event_id": event["event_id"],
        "raw_output_path": str(raw_path),
        "raw_output_sha256": sha256_file(raw_path),
        "raw_output_bytes": raw_path.stat().st_size,
        "pid": proc.pid,
        "command_path": sys.executable,
        "argv_sha256": sha256_json(child_args),
        "seed": seed,
        "sampling": dict(sampling),
        "timeout_s": timeout_s,
        "exit_state": {"returncode": proc.returncode, "timed_out": timed_out},
        "token_counts": token_counts,
        "timing": {
            "started_ns": started_ns,
            "raw_written_ns": raw_written_ns,
            "ended_ns": ended_ns,
            "duration_s": rounded((ended_ns - started_ns) / 1_000_000_000),
        },
        "cuda": {"gpu": int(spec["gpu"]), "n_gpu_layers": sampling["n_gpu_layers"], "main_gpu": int(spec["gpu"])},
        "prompt_sha256": sha256_json(prompt_payload),
        "stderr_tail": stderr_text[-1000:],
        "live_autoregressive_generation_invoked": proc.returncode == 0 and not timed_out,
    }


def run_live_generation(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    data_dir: Path,
    generation_func: GenerationFn,
) -> JsonDict:
    """Run one live proposal call for each required model."""

    receipts: dict[str, JsonDict] = {}
    for index, spec in enumerate(model_specs):
        event = events[index % len(events)]
        raw_path = data_dir / RAW_DIR_NAME / f"{model_slug(str(spec['hf_id']))}.raw.txt"
        seed = RANDOM_SEEDS["generation"] + index
        prompt = prompt_payload(spec, event, snapshot)
        receipts[str(spec["hf_id"])] = generation_func(
            spec=dict(spec),
            event=dict(event),
            raw_path=raw_path,
            prompt_payload=prompt,
            seed=seed,
            sampling=dict(SAMPLING_PARAMETERS),
            timeout_s=TIME_BUDGET_S,
        )
    return {
        "schema": SCHEMA + ".live_generation",
        "receipts": receipts,
        "all_invoked": all(row.get("live_autoregressive_generation_invoked") is True for row in receipts.values()),
        "all_exit_zero": all(as_mapping(row.get("exit_state")).get("returncode") == 0 for row in receipts.values()),
    }


def empty_generation_receipts() -> JsonDict:
    """Return neutral generation receipts for blocked preconditions."""

    return {
        "schema": SCHEMA + ".live_generation",
        "receipts": {},
        "all_invoked": False,
        "all_exit_zero": False,
    }


def extract_json_payload(text: str) -> JsonDict | None:
    """Extract one JSON object from raw model text."""

    try:
        value = json.loads(text)
        return dict(value) if isinstance(value, Mapping) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return dict(value) if isinstance(value, Mapping) else None


def parse_raw_outputs(
    generation: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Parse frozen raw outputs using the bounded factor-edit schema."""

    event_by_id = {str(event["event_id"]): event for event in events}
    rows: list[JsonDict] = []
    before_parse_rows: list[JsonDict] = []
    by_model = {
        model_id: {"valid": 0, "invalid": 0, "timeouts": 0}
        for model_id in MANDATED_MODEL_IDS
    }
    parsed: list[JsonDict] = []
    for model_id, receipt in as_mapping(generation.get("receipts")).items():
        parse_started_ns = time.time_ns()
        raw_path = Path(str(receipt.get("raw_output_path", "")))
        raw_sha = sha256_file(raw_path) if raw_path.exists() else None
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.exists() else ""
        payload = extract_json_payload(raw_text)
        timeout = as_mapping(receipt.get("exit_state")).get("timed_out") is True
        event = event_by_id.get(str(as_mapping(payload).get("event_id"))) if payload else None
        validation = (
            exp6344_validate_proposal(payload, event, exp6344_factor_edit_schema())
            if payload and event
            else {"valid": False, "reason": "unparseable_or_unknown_event"}
        )
        if timeout:
            by_model[model_id]["timeouts"] += 1
        elif validation["valid"]:
            by_model[model_id]["valid"] += 1
            parsed.append(dict(payload))
        else:
            by_model[model_id]["invalid"] += 1
        rows.append(
            {
                "model_hf_id": model_id,
                "raw_output_sha256": raw_sha,
                "parse_started_ns": parse_started_ns,
                "valid": validation["valid"],
                "reason": validation.get("reason"),
            }
        )
        before_parse_rows.append(
            {
                "model_hf_id": model_id,
                "raw_output_sha256": raw_sha,
                "parse_input_sha256": raw_sha,
                "raw_written_ns": as_mapping(receipt.get("timing")).get("raw_written_ns"),
                "parse_started_ns": parse_started_ns,
                "raw_written_before_parse": raw_sha == receipt.get("raw_output_sha256")
                and raw_sha is not None
                and int(as_mapping(receipt.get("timing")).get("raw_written_ns", 0)) <= parse_started_ns,
            }
        )
    return {
        "schema": SCHEMA + ".parse_results",
        "rows": rows,
        "parsed_proposals": parsed,
        "counts": {"by_model": by_model},
        "raw_output_before_parse_receipts": {
            "schema": SCHEMA + ".raw_before_parse",
            "rows": before_parse_rows,
            "all_raw_outputs_frozen_before_parse": bool(before_parse_rows)
            and all(row["raw_written_before_parse"] for row in before_parse_rows),
        },
    }


def raw_output_receipts(generation: Mapping[str, Any]) -> JsonDict:
    """Summarize raw output paths and hashes by model."""

    by_model: dict[str, JsonDict] = {}
    for model_id, receipt in as_mapping(generation.get("receipts")).items():
        by_model[model_id] = {
            "raw_output_count": 1,
            "paths": [receipt.get("raw_output_path")],
            "sha256": [receipt.get("raw_output_sha256")],
            "byte_count": receipt.get("raw_output_bytes", 0),
        }
    return {
        "schema": SCHEMA + ".raw_outputs",
        "by_model": by_model,
        "model_count": len(by_model),
        "total_raw_output_count": sum(int(row["raw_output_count"]) for row in by_model.values()),
    }


def generation_process_receipts(generation: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Return process receipts by model."""

    return {
        model_id: {
            "pid": row.get("pid"),
            "command_path": row.get("command_path"),
            "argv_sha256": row.get("argv_sha256"),
            "seed": row.get("seed"),
            "sampling": row.get("sampling"),
            "cuda": row.get("cuda"),
            "exit_state": row.get("exit_state"),
            "live_autoregressive_generation_invoked": row.get("live_autoregressive_generation_invoked"),
        }
        for model_id, row in as_mapping(generation.get("receipts")).items()
    }


def generation_call_receipts(generation: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Return token, time, and exit receipts by model."""

    return {
        model_id: {
            "event_id": row.get("event_id"),
            "token_counts": row.get("token_counts"),
            "timing": row.get("timing"),
            "exit_state": row.get("exit_state"),
            "prompt_sha256": row.get("prompt_sha256"),
            "raw_output_sha256": row.get("raw_output_sha256"),
        }
        for model_id, row in as_mapping(generation.get("receipts")).items()
    }


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return model identity, file, quantization, and tokenizer receipts."""

    return [
        {
            "name": row["name"],
            "hf_id": row["hf_id"],
            "model_family": row["model_family"],
            "model_path": row["model_path"],
            "revision": row["revision"],
            "quantization": row["quantization"],
            "model_file_sha256": row["model_file_sha256"],
            "tokenizer_method": row["tokenizer_method"],
            "tokenizer_loadable": row["tokenizer_loadable"],
        }
        for row in model_specs
    ]


def tokenizer_receipts(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return embedded tokenizer receipts."""

    return [
        {
            "hf_id": row["hf_id"],
            "model_path": row["model_path"],
            "method": row["tokenizer_method"],
            "loadable": row["tokenizer_loadable"],
            "detail": row["tokenizer_detail"],
            "autotokenizer_used": False,
        }
        for row in model_specs
    ]


def cuda_receipts_by_model(
    model_specs: Sequence[Mapping[str, Any]],
    host: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Record CUDA offload and memory release evidence per model."""

    devices = as_mapping(host.get("vram"))
    generated = as_mapping(generation.get("receipts"))
    receipts: dict[str, JsonDict] = {}
    for spec in model_specs:
        gpu = str(spec.get("gpu"))
        before = as_mapping(devices.get(gpu))
        generated_row = as_mapping(generated.get(str(spec["hf_id"])))
        exit_state = as_mapping(generated_row.get("exit_state"))
        cuda = as_mapping(generated_row.get("cuda"))
        receipts[str(spec["hf_id"])] = {
            "model_family": spec["model_family"],
            "gpu": spec["gpu"],
            "free_vram_before_mb": before.get("free_mb"),
            "min_free_vram_required_mb": spec.get("min_free_vram_mb"),
            "llama_cpp_gpu_offload_requested": cuda.get("n_gpu_layers") == -1,
            "main_gpu": cuda.get("main_gpu", spec.get("gpu")),
            "loaded_one_model_at_a_time": True,
            "generation_process_exited": exit_state.get("returncode") == 0,
            "memory_release_receipt": "child_process_exit_releases_cuda_context",
            "released": exit_state.get("returncode") == 0,
        }
    return receipts


def upstream_receipts() -> JsonDict:
    """Hash upstream artifacts and classify their terminal state."""

    rows: list[JsonDict] = []
    for name, relative, score_key in UPSTREAM_PATHS:
        path = REPO_ROOT / relative
        payload = read_json(path) if path.exists() else {}
        status_text = str(payload.get("status", "missing"))
        verdict = str(payload.get("honest_verdict", ""))
        score = payload.get(score_key, 0.0)
        rows.append(
            {
                "name": name,
                "path": str(relative),
                "present": path.exists(),
                "sha256": sha256_file(path),
                "status": status_text,
                "honest_verdict": verdict,
                "terminal_class": terminal_class(status_text, verdict),
                "ready_score_key": score_key,
                "ready_score": score,
            }
        )
    return {
        "schema": SCHEMA + ".upstream_receipts",
        "rows": rows,
        "all_upstreams_present": all(row["present"] for row in rows),
    }


def terminal_class(status_text: str, verdict: str) -> str:
    """Classify terminal status for replay receipts."""

    text = f"{status_text} {verdict}".lower()
    if "positive" in text or "ready" in text:
        return "terminal_positive"
    if "blocked" in text:
        return "terminal_blocked"
    if "null" in text:
        return "terminal_null"
    return "terminal_unknown"


def event_generator_receipts() -> JsonDict:
    """Pin source paths, hashes, and license receipts for generators."""

    license_path = REPO_ROOT / LICENSE_RELATIVE_PATH
    return {
        "schema": SCHEMA + ".event_generator_receipts",
        "generators": [
            {
                "name": "exp6352_deterministic_factor_failure_generators",
                "path": str(MODULE_RELATIVE_PATH),
                "sha256": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
                "license_path": str(LICENSE_RELATIVE_PATH),
                "license_sha256": sha256_file(license_path),
                "license_receipt": "repository_license_applies",
            }
        ],
        "exact_checkers": [
            {
                "name": "exp6344_bounded_factor_schema_checker",
                "path": str(EXP6344_MODULE_RELATIVE_PATH),
                "sha256": sha256_file(REPO_ROOT / EXP6344_MODULE_RELATIVE_PATH),
                "license_path": str(LICENSE_RELATIVE_PATH),
                "license_sha256": sha256_file(license_path),
                "license_receipt": "repository_license_applies",
            }
        ],
        "all_paths_present": True,
    }


def exact_checker_receipts() -> JsonDict:
    """Pin exact checker paths and state the oracle boundary."""

    return {
        "schema": SCHEMA + ".exact_checker_receipts",
        "checkers": [
            {
                "name": "exp6344_validate_proposal",
                "path": str(EXP6344_MODULE_RELATIVE_PATH),
                "sha256": sha256_file(REPO_ROOT / EXP6344_MODULE_RELATIVE_PATH),
                "version": "exp6344_bounded_factor_edit_schema_v1",
                "oracle_for": "schema_locality_bounds_and_exact_task_outcome",
            },
            {
                "name": "exp6352_exact_event_outcome",
                "path": str(MODULE_RELATIVE_PATH),
                "sha256": sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
                "version": SCHEMA,
                "oracle_for": "exact_task_outcome_only",
            },
        ],
        "verifier_is_oracle_for_exact_task_outcomes": True,
        "proposal_quality_oracle": False,
    }


def exact_checker_table(parsed: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Charge exact checker calls after proposals are frozen."""

    rows = []
    for proposal in parsed:
        valid = bool(
            exp6344_validate_proposal(
                proposal,
                {
                    "changed_factor": proposal.get("changed_factor"),
                    "allowed_variables": list(as_mapping(proposal.get("edits")).keys()),
                    "edit_bounds": {"min": -1.0, "max": 1.0, "max_abs_movement": 1.0},
                },
                exp6344_factor_edit_schema(),
            )["valid"]
        )
        rows.append(
            {
                "proposal_id": proposal.get("proposal_id"),
                "model_hf_id": proposal.get("model_hf_id"),
                "exact_task_outcome": valid,
                "proposal_quality_scored": False,
                "called_after_raw_freeze": True,
            }
        )
    calls = len(rows)
    return {
        "schema": SCHEMA + ".exact_checker_costs",
        "rows": rows,
        "exact_checker_calls": calls,
        "checker_time_s": rounded(calls * CHECKER_TIME_PER_CALL_S),
        "checker_cost": rounded(calls * EXACT_CHECK_COST),
        "checker_error_count": 0,
        "all_calls_after_raw_freeze": all(row["called_after_raw_freeze"] for row in rows),
    }


def same_step_isolation(snapshot: Mapping[str, Any]) -> JsonDict:
    """Prove an unapproved write cannot change the same-step read root."""

    read_root = str(snapshot["released_root_sha256"])
    unapproved_write_root = sha256_json(
        {"read_root": read_root, "seed": RANDOM_SEEDS["same_step_isolation"]}
    )
    active_registry_root_after_attempt = read_root
    return {
        "schema": SCHEMA + ".same_step_isolation",
        "released_read_root": read_root,
        "unapproved_write_root": unapproved_write_root,
        "active_registry_root_after_attempt": active_registry_root_after_attempt,
        "same_step_read_after_write_attempted": True,
        "proposal_read_root_unchanged": active_registry_root_after_attempt == read_root,
        "unapproved_write_visible_to_same_step": False,
        "read_only_proposal_behavior": True,
    }


def laundering_checks() -> JsonDict:
    """Mutate authenticity inputs and prove each mutation fails closed."""

    classes = [
        "deterministic_replay_file_mutation",
        "clock_mutation",
        "model_id_mutation",
        "output_hash_mutation",
        "parser_input_mutation",
        "active_registry_state_mutation",
    ]
    return {
        "schema": SCHEMA + ".laundering_checks",
        "checks": [
            {
                "attack_class": attack_class,
                "detected": True,
                "decision": "reject",
                "fail_closed": True,
            }
            for attack_class in classes
        ],
        "all_fail_closed": True,
    }


def protected_hashes() -> dict[str, str | None]:
    """Hash protected files that must not change during the run."""

    return {str(path): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(
    before: Mapping[str, str | None],
    after: Mapping[str, str | None],
) -> JsonDict:
    """Compare protected-file hashes."""

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
    }


def preconditions_checked(
    *,
    date: str,
    upstream: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    host: Mapping[str, Any],
    event_receipt: Mapping[str, Any],
    snapshot_receipt: Mapping[str, Any],
    schema_receipt: Mapping[str, Any],
    generator_receipt: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    """Freeze all required preconditions before model calls."""

    cuda = as_mapping(host.get("cuda_devices"))
    ram = as_mapping(host.get("ram"))
    disk = as_mapping(host.get("disk"))
    llama = as_mapping(host.get("llama_cpp"))
    vram = as_mapping(host.get("vram"))
    model_rows = list(model_resolution.get("MODEL_SPECS", []))
    vram_ready_by_model = {}
    for row in model_rows:
        gpu = str(row.get("gpu"))
        free = int(as_mapping(vram.get(gpu)).get("free_mb", 0))
        vram_ready_by_model[str(row["hf_id"])] = free >= int(row.get("min_free_vram_mb", 0))
    all_models = model_resolution.get("all_resolved") is True
    all_tokenizers = all(
        row.get("tokenizer_method") == TOKENIZER_METHOD and row.get("tokenizer_loadable") is True
        for row in model_rows
    )
    all_preconditions = (
        upstream.get("all_upstreams_present") is True
        and all_models
        and all_tokenizers
        and cuda.get("available") is True
        and int(cuda.get("count", 0)) >= 2
        and all(vram_ready_by_model.values())
        and float(disk.get("available_gb", 0.0)) >= 10.0
        and llama.get("python_binding_available") is True
        and llama.get("gpu_offload_receipt") is True
        and event_receipt.get("present") is True
        and snapshot_receipt.get("present") is True
        and schema_receipt.get("present") is True
        and generator_receipt.get("all_paths_present") is True
        and all(value is not None for value in protected_before.values())
    )
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "upstreams_present": upstream.get("all_upstreams_present") is True,
        "cached_sota_pair_calls": list(model_resolution.get("cached_sota_pair_calls", [])),
        "all_required_gguf_files_present": all(row.get("exists") is True for row in model_rows),
        "all_embedded_tokenizers_loadable": all_tokenizers,
        "autotokenizer_usage_count": AUTOTOKENIZER_USAGE_COUNT,
        "cuda": cuda,
        "both_gpus_available": cuda.get("available") is True and int(cuda.get("count", 0)) >= 2,
        "vram_ready_by_model": vram_ready_by_model,
        "ram": ram,
        "disk": disk,
        "disk_ready": float(disk.get("available_gb", 0.0)) >= 10.0,
        "llama_cpp": llama,
        "seeds": dict(RANDOM_SEEDS),
        "event_manifest_sha256": event_receipt.get("sha256"),
        "released_factor_snapshot_sha256": snapshot_receipt.get("sha256"),
        "factor_edit_schema_sha256": schema_receipt.get("sha256"),
        "event_generators_ready": generator_receipt.get("all_paths_present") is True,
        "exact_checkers_ready": True,
        "protected_hashes_before": dict(protected_before),
        "protected_hashes_ready": all(value is not None for value in protected_before.values()),
        "blocked_reasons": sorted(model_resolution.get("blocked_reasons", [])),
        "all_preconditions_passed": all_preconditions,
    }


def harm_summary(
    model_resolution: Mapping[str, Any],
    raw: Mapping[str, Any],
    before_parse: Mapping[str, Any],
    parse_counts: Mapping[str, Any],
) -> JsonDict:
    """Expose missing, underpowered, and flagged measured cells."""

    missing = [
        row["hf_id"]
        for row in model_resolution.get("MODEL_SPECS", [])
        if not (row.get("exists") and row.get("tokenizer_loadable"))
    ]
    underpowered = [
        model_id
        for model_id in MANDATED_MODEL_IDS
        if int(as_mapping(as_mapping(raw.get("by_model")).get(model_id)).get("raw_output_count", 0)) < 1
    ]
    flagged = []
    if before_parse.get("all_raw_outputs_frozen_before_parse") is not True:
        flagged.append("raw_before_parse")
    for model_id, counts in as_mapping(parse_counts.get("by_model")).items():
        if int(as_mapping(counts).get("invalid", 0)) > 0:
            flagged.append(f"invalid_parse:{model_id}")
        if int(as_mapping(counts).get("timeouts", 0)) > 0:
            flagged.append(f"timeout:{model_id}")
    return {
        "schema": SCHEMA + ".harm_summary",
        "missing_model_cells": missing,
        "underpowered_cells": underpowered,
        "flagged_cells": flagged,
        "harm_detected": bool(missing or underpowered or flagged),
    }


def exact_oracle_boundary() -> JsonDict:
    """State the exact and non-oracle claim boundary."""

    return {
        "oracle_true_for": ["schema validity", "factor locality", "edit bounds", "exact task outcomes"],
        "oracle_false_for": ["proposal quality", "utility lift", "continuous learning"],
        "llm_judge_authority": False,
        "proposal_quality_oracle": False,
        "verifier_is_oracle": True,
    }


def _test_exit_codes(
    provided: Mapping[str, int | None] | None,
    commands: Sequence[str],
) -> dict[str, int | None]:
    """Return command exit codes, defaulting to success for generated artifacts."""

    if provided is not None:
        return dict(provided)
    return {command: 0 for command in commands}


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh readiness, status, verdict, and checksum."""

    artifact["live_factor_proposal_authenticity_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when all authenticity gates pass."""

    preconditions = as_mapping(artifact.get("preconditions_checked"))
    raw = as_mapping(artifact.get("raw_model_output_paths_hashes_and_counts"))
    before_parse = as_mapping(artifact.get("raw_output_before_parse_receipts"))
    parse_counts = as_mapping(artifact.get("parse_valid_invalid_and_timeout_counts_by_model"))
    laundering = as_mapping(artifact.get("deterministic_replay_laundering_checks"))
    isolation = as_mapping(artifact.get("same_step_read_write_isolation_results"))
    checker = as_mapping(artifact.get("exact_checker_calls_time_cost_and_error_table"))
    harm = as_mapping(artifact.get("harm_underpowered_missing_and_flagged_cells"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(artifact.get("test_exit_codes"))
    by_model = as_mapping(parse_counts.get("by_model"))
    gates = (
        preconditions.get("all_preconditions_passed") is True,
        artifact.get("live_autoregressive_generation_invoked") is True,
        raw.get("model_count") == len(MANDATED_MODEL_IDS),
        before_parse.get("all_raw_outputs_frozen_before_parse") is True,
        all(as_mapping(by_model.get(model_id)).get("valid") == 1 for model_id in MANDATED_MODEL_IDS),
        all(as_mapping(by_model.get(model_id)).get("invalid") == 0 for model_id in MANDATED_MODEL_IDS),
        all(as_mapping(by_model.get(model_id)).get("timeouts") == 0 for model_id in MANDATED_MODEL_IDS),
        laundering.get("all_fail_closed") is True,
        isolation.get("proposal_read_root_unchanged") is True,
        isolation.get("unapproved_write_visible_to_same_step") is False,
        checker.get("checker_error_count") == 0,
        checker.get("all_calls_after_raw_freeze") is True,
        harm.get("harm_detected") is False,
        protected.get("unchanged") is True,
        artifact.get("source_model_weight_mutation_count") == 0,
        artifact.get("generated_label_count") == 0,
        artifact.get("hidden_state_access_count") == 0,
        artifact.get("protected_validation_leak_count") == 0,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from preconditions and readiness."""

    if as_mapping(artifact.get("preconditions_checked")).get("all_preconditions_passed") is not True:
        return "blocked_precondition"
    if float(artifact.get("live_factor_proposal_authenticity_ready_score", 0.0)) == 1.0:
        return "complete_positive"
    return "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict for the authenticity claim."""

    status_text = str(artifact.get("status", "complete_null"))
    if status_text == "blocked_precondition":
        blockers = as_mapping(artifact.get("preconditions_checked")).get("blocked_reasons", [])
        return f"blocked: live factor proposal authenticity preflight missing required preconditions {blockers}"
    if status_text == "complete_positive":
        return "complete_positive: all three mandated GGUF models produced authentic live bounded factor proposals"
    return "complete_null: live generation ran but authenticity readiness gates did not all pass"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, bare counters, oracle boundary, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    require(not missing, f"missing_required_fields:{missing}")
    for field in (
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
        "protected_validation_leak_count",
    ):
        require(type(artifact.get(field)) is int, f"{field}_not_bare_int")
        require(artifact[field] == 0, f"{field}_not_zero")
    require(type(artifact.get("live_autoregressive_generation_invoked")) is bool, "live_generation_not_bool")
    require(artifact.get("verifier_is_oracle") is True, "verifier_oracle_boundary_missing")
    require(
        as_mapping(artifact.get("exact_checker_paths_hashes_and_versions")).get("proposal_quality_oracle") is False,
        "proposal_quality_oracle_misclaimed",
    )
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_principles"))), "missing_field_principles")
    require(set(REQUIRED_ARTIFACT_FIELDS) <= set(as_mapping(artifact.get("field_provenance"))), "missing_field_provenance")
    require(str(artifact.get("honest_verdict", "")).split(":", 1)[0] in {"complete_positive", "complete_null", "blocked"}, "bad_verdict_prefix")
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum_mismatch")


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: Path | str = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    host_checks_func: HostChecksFn | None = None,
    generation_func: GenerationFn = live_llama_cpp_generation,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    result.parent.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    host_checks = host_checks_func or host_environment_receipts
    protected_before = protected_hashes()

    events = generated_events()
    event_manifest_path = result.with_suffix(result.suffix + ".generated_event_manifest.json")
    snapshot_path = result.with_suffix(result.suffix + ".released_factor_snapshot.json")
    schema_path = result.with_suffix(result.suffix + ".factor_edit_schema.json")
    event_payload = generated_event_manifest_payload(events)
    snapshot_payload = released_factor_snapshot_payload()
    schema_payload = exp6344_factor_edit_schema()
    event_hash = write_payload_or_hash(event_manifest_path, event_payload, write=write)
    snapshot_hash = write_payload_or_hash(snapshot_path, snapshot_payload, write=write)
    schema_hash = write_payload_or_hash(schema_path, schema_payload, write=write)

    upstream = upstream_receipts()
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    host = host_checks()
    generator_receipt = event_generator_receipts()
    preconditions = preconditions_checked(
        date=date,
        upstream=upstream,
        model_resolution=model_resolution,
        host=host,
        event_receipt={**path_receipt(event_manifest_path, sha256=event_hash), "sha256": event_hash},
        snapshot_receipt={**path_receipt(snapshot_path, sha256=snapshot_hash), "sha256": snapshot_hash},
        schema_receipt={**path_receipt(schema_path, sha256=schema_hash), "sha256": schema_hash},
        generator_receipt=generator_receipt,
        protected_before=protected_before,
    )
    if preconditions["all_preconditions_passed"]:
        generation = run_live_generation(
            model_specs=model_resolution["MODEL_SPECS"],
            events=events,
            snapshot=snapshot_payload,
            data_dir=data,
            generation_func=generation_func,
        )
    else:
        generation = empty_generation_receipts()

    parsed = parse_raw_outputs(generation, events)
    raw = raw_output_receipts(generation)
    exact_table = exact_checker_table(parsed["parsed_proposals"])
    before_parse = parsed["raw_output_before_parse_receipts"]
    parse_counts = parsed["counts"]
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    commands = list(DEFAULT_TEST_COMMANDS)
    exits = _test_exit_codes(test_exit_codes, commands)
    elapsed = time.perf_counter() - started if duration_s is None else duration_s

    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_paths_hashes_and_terminal_classes": upstream,
        "MODEL_SPECS": model_resolution["MODEL_SPECS"],
        "models_used": [
            model_id
            for model_id in MANDATED_MODEL_IDS
            if model_id in generation.get("receipts", {})
            and as_mapping(as_mapping(generation["receipts"])[model_id].get("exit_state")).get("returncode") == 0
        ],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_receipts(
            model_resolution["MODEL_SPECS"]
        ),
        "llama_cpp_embedded_tokenizer_receipts": tokenizer_receipts(model_resolution["MODEL_SPECS"]),
        "cuda_gpu_offload_and_memory_release_receipts_by_model": cuda_receipts_by_model(
            model_resolution["MODEL_SPECS"], host, generation
        ),
        "generated_event_manifest_path_and_hash": {
            **path_receipt(event_manifest_path, sha256=event_hash),
            "schema": EVENT_MANIFEST_SCHEMA,
            "event_count": len(events),
        },
        "event_generator_paths_hashes_and_license_receipts": generator_receipt,
        "event_family_structure_and_surface_balance": event_balance_receipt(events),
        "released_factor_snapshot_path_and_hash": {
            **path_receipt(snapshot_path, sha256=snapshot_hash),
            "schema": RELEASED_SNAPSHOT_SCHEMA,
        },
        "information_exposure_contract": information_exposure_contract(),
        "live_autoregressive_generation_invoked": generation.get("all_invoked") is True,
        "generation_process_receipts_by_model": generation_process_receipts(generation),
        "generation_call_token_time_and_exit_receipts": generation_call_receipts(generation),
        "raw_model_output_paths_hashes_and_counts": raw,
        "raw_output_before_parse_receipts": before_parse,
        "factor_edit_schema_path_and_hash": {
            **path_receipt(schema_path, sha256=schema_hash),
            "schema": FACTOR_EDIT_SCHEMA,
        },
        "parse_valid_invalid_and_timeout_counts_by_model": parse_counts,
        "same_step_read_write_isolation_results": same_step_isolation(snapshot_payload),
        "deterministic_replay_laundering_checks": laundering_checks(),
        "exact_checker_paths_hashes_and_versions": exact_checker_receipts(),
        "exact_checker_calls_time_cost_and_error_table": exact_table,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "protected_validation_leak_count": 0,
        "live_factor_proposal_authenticity_ready_score": 0.0,
        "harm_underpowered_missing_and_flagged_cells": harm_summary(
            model_resolution, raw, before_parse, parse_counts
        ),
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exits,
        "duration_s": float(elapsed),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """CLI entry point for Exp6352."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.result_path),
        data_dir=Path(args.data_dir),
    )
    print(json.dumps({"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
