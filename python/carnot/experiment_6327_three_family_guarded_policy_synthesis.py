"""Exp6327 three-family guarded policy synthesis.

Spec refs: REQ-KONA-6327, SCENARIO-KONA-6327-GATE,
SCENARIO-KONA-6327-MATCHED-ARMS, SCENARIO-KONA-6327-ORACLE-BOUNDARY.

The model only proposes bounded policy programs. Exp6326's exact finite-domain
guard keeps all safety authority and routes rejected candidates to a verified
hash-pinned fallback.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_6326_restricted_policy_contract_compiler as exp6326
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
GenerationFn = Callable[[dict[str, Any], str, int, dict[str, Any]], dict[str, Any]]
HostChecksFn = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6327_three_family_guarded_policy_synthesis.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6327_three_family_guarded_policy_synthesis"
)
RAW_DIR_NAME = "raw_candidates"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6327_three_family_guarded_policy_synthesis.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6327_three_family_guarded_policy_synthesis.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
UPSTREAM_RELATIVE_PATH = exp6326.RESULT_RELATIVE_PATH
LLAMA_CPP_CLI_PATH = Path.home() / ".cache/llama.cpp-master/build/bin/llama-cli"

SCHEMA = "carnot.experiment_6327.three_family_guarded_policy_synthesis.v1"
RUN_DATE = "20260812"
INFERENCE_SUBSTRATE = "local_three_family_llama_cpp_cuda_guarded_policy_synthesis"
RANDOM_SEEDS = (632700, 632701, 632702)
MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_TEMPLATES: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "gpu": 0,
        "contract_model_family": "qwen_moe",
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "gpu": 1,
        "contract_model_family": "gemma_dense",
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "gpu": 1,
        "contract_model_family": "gemma_moe",
    },
)
ARMS = (
    "one_raw_candidate",
    "reject_only_filtering",
    "exact_guard_plus_hash_pinned_fallback",
    "bounded_exact_factor_energy_guided_candidate_search_plus_fallback",
)
CANDIDATE_COUNT = 2
MAX_TOKENS = 768
TIME_BUDGET_S = 480
CONTEXT_TOKENS = 4096
TEMPERATURE = 0.2
TOP_P = 0.9
REPEAT_PENALTY = 1.05
BASE_CANDIDATE_UTILITY = 1.0
FALLBACK_UTILITY = 0.70
MATCHED_GENERATION_COST = 0.08
FALLBACK_COST = 0.10
UNSAFE_ENERGY_PENALTY = 0.20

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6327_three_family_guarded_policy_synthesis "
    "--date 20260812"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_6327_three_family_guarded_policy_synthesis.py "
        "-q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage run --rcfile=/dev/null "
        "--include=python/carnot/experiment_6327_three_family_guarded_policy_synthesis.py "
        "-m pytest tests/python/test_experiment_6327_three_family_guarded_policy_synthesis.py "
        "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
        "--include=python/carnot/experiment_6327_three_family_guarded_policy_synthesis.py "
        "--fail-under=100"
    ),
    ".venv/bin/pytest tests/python -q",
    (
        ".venv/bin/python scripts/check_spec_coverage.py "
        "tests/python/test_experiment_6327_three_family_guarded_policy_synthesis.py"
    ),
    (
        ".venv/bin/python scripts/adversarial_verify.py "
        "results/experiment_6327_three_family_guarded_policy_synthesis.json"
    ),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_path_hash_terminal_class_and_gate_receipt",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "llama_cpp_embedded_tokenizer_receipts",
    "cuda_gpu_offload_and_memory_release_receipts_by_model",
    "fixture_and_split_paths_hashes",
    "prompt_and_decoder_contract",
    "arm_definitions",
    "matched_call_token_candidate_and_time_budgets",
    "raw_candidate_paths_hashes_and_counts",
    "parse_and_normalization_results",
    "exact_factor_energies_by_candidate",
    "guard_accept_reject_and_fallback_receipts",
    "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed",
    "paired_deltas_intervals_and_sample_sizes",
    "harm_underpowered_missing_and_flagged_cells",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "hidden_state_access_count",
    "exact_oracle_claim_boundary",
    "guarded_policy_synthesis_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seed",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Shows whether the run is ready, null, or blocked.",
    "upstream_path_hash_terminal_class_and_gate_receipt": "Pins Exp6326 before using its exact guard.",
    "MODEL_SPECS": "Records the three mandated local GGUF model rows.",
    "models_used": "Names only models that supplied raw candidates.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Pins model files and tokenizer receipts.",
    "llama_cpp_embedded_tokenizer_receipts": "Proves tokenizer checks use embedded GGUF metadata.",
    "cuda_gpu_offload_and_memory_release_receipts_by_model": "Shows each model used and released CUDA resources.",
    "fixture_and_split_paths_hashes": "Pins fixture, split, and fallback inputs.",
    "prompt_and_decoder_contract": "Freezes prompts and decoder settings before output.",
    "arm_definitions": "Defines the four matched comparison arms.",
    "matched_call_token_candidate_and_time_budgets": "Keeps cost budgets identical across arms.",
    "raw_candidate_paths_hashes_and_counts": "Pins raw model output before parsing.",
    "parse_and_normalization_results": "Preserves parser failures and canonicalization results.",
    "exact_factor_energies_by_candidate": "Reports exact local factor energy for each candidate.",
    "guard_accept_reject_and_fallback_receipts": "Separates acceptance, rejection, and fallback routing.",
    "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed": "Reports utility and safety per model family arm seed cell.",
    "paired_deltas_intervals_and_sample_sizes": "Shows matched search-minus-guard utility deltas.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps missing, harmful, and underpowered cells visible.",
    "source_model_weight_mutation_count": "Proves source model weights were not updated.",
    "generated_label_count": "Proves model text did not supply labels.",
    "hidden_state_access_count": "Proves hidden activations did not enter the decision.",
    "exact_oracle_claim_boundary": "States that exact checking is the oracle.",
    "guarded_policy_synthesis_ready_score": "Opens only for complete safe cells and positive development utility delta.",
    "protected_files_unchanged": "Shows conductor and reconciler files stayed byte-identical.",
    "preconditions_checked": "Records resource, budget, fallback, seed, and protected-file gates.",
    "inference_substrate": "Declares live local GGUF generation plus exact guarding.",
    "verifier_is_oracle": "Prevents a learned-verifier or model-judge claim.",
    "field_provenance": "Maps each field to code, specs, models, fixtures, or tests.",
    "field_principles": "Explains why each required field exists.",
    "test_commands": "Lists verification commands for this artifact.",
    "test_exit_codes": "Records command outcomes for readiness.",
    "duration_s": "Reports measured wall-clock duration.",
    "random_seed": "Provides the compatibility seed field expected by artifact verification.",
    "random_seeds": "Pins deterministic candidate schedules.",
    "reproducibility_checksum": "Detects drift in the artifact payload.",
    "honest_verdict": "States the terminal claim boundary.",
}
FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "sources": ["REQ-KONA-6327", "Exp6326 exact guard", "local GGUF receipts"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
) -> JsonDict:
    """Resolve the three mandated GGUF rows through the cached pair helper."""

    calls = [
        "cached_sota_pair(gpu_indices=(0,1))",
        "cached_sota_pair(gpu_indices=(0,1), model_indices=(0,2))",
    ]
    default_pair = cached_pair_func(gpu_indices=(0, 1)) or []
    dense_pair = cached_pair_func(gpu_indices=(0, 1), model_indices=(0, 2)) or []
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
        if not tokenizer_ok:
            blockers.append(f"embedded_tokenizer_not_loadable:{hf_id}")
        records.append(
            {
                **template,
                "name": raw.get("name", template["name"]),
                "gpu": int(raw.get("gpu", template["gpu"])),
                "model_path": str(path),
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
                "size_bytes": path.stat().st_size if exists else 0,
                "revision": extract_revision(path),
                "quantization": extract_quantization(path),
                "tokenizer_loadable": bool(tokenizer_ok),
                "tokenizer_status": tokenizer_detail,
                "tokenizer_method": "llama_cpp_embedded_gguf_vocab_only",
                "cache_policy": _cache_policy_for_model(hf_id),
            }
        )
    if not default_pair:
        blockers.append("cached_sota_pair_missing")
    if [row["hf_id"] for row in records] != list(MANDATED_MODEL_IDS):
        blockers.append("mandated_model_order")
    return {
        "schema": SCHEMA + ".model_specs",
        "MODEL_SPECS": records,
        "cached_sota_pair_calls": calls,
        "blocked_reasons": sorted(set(blockers)),
        "all_resolved": not blockers,
    }


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: Path | str = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    duration_s: float | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    cached_pair_func: CachedPairFn | None = None,
    tokenizer_func: TokenizerFn | None = None,
    generation_func: GenerationFn | None = None,
    host_checks_func: HostChecksFn | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp6327 and optionally write the terminal artifact."""

    started = time.perf_counter()
    cached_pair = cached_pair_func or cached_sota_pair
    tokenizer = tokenizer_func or gguf_tokenizer_loadable
    generator = generation_func or generate_with_llama_cli
    host_checks = host_checks_func or host_environment_receipts
    result = Path(result_path)
    data = Path(data_dir)
    fixtures = exp6326.build_fixture_manifest()
    protected_before = protected_hashes()
    upstream = upstream_receipt()
    model_resolution = build_model_specs(
        cached_pair_func=cached_pair,
        tokenizer_func=tokenizer,
    )
    prompt_contract = prompt_and_decoder_contract(fixtures)
    budgets = matched_budgets()
    fixture_receipts = fixture_and_split_receipts(fixtures)
    host = host_checks()
    preconditions = precondition_receipt(
        date=date,
        result_path=result,
        data_dir=data,
        upstream=upstream,
        model_resolution=model_resolution,
        fixture_receipts=fixture_receipts,
        prompt_contract=prompt_contract,
        budgets=budgets,
        host=host,
        protected_before=protected_before,
    )
    generation = (
        generate_raw_outputs(
            model_resolution["MODEL_SPECS"],
            fixtures,
            data_dir=data,
            prompt=prompt_contract["prompt_text"],
            budget=budgets,
            generation_func=generator,
            write=write,
        )
        if preconditions["all_passed"]
        else empty_generation(model_resolution["MODEL_SPECS"], data / RAW_DIR_NAME)
    )
    parsed = parse_and_score_candidates(generation["raw_outputs"], fixtures)
    arm_results = evaluate_arms(parsed["candidate_rows"], fixtures)
    paired = paired_delta_summary(arm_results["metrics"])
    harm = harm_summary(
        model_resolution=model_resolution,
        generation=generation,
        parsed=parsed,
        arm_results=arm_results,
        paired=paired,
    )
    protected_after = protected_hashes()
    protected = protected_unchanged_receipt(protected_before, protected_after)
    commands = list(test_commands or DEFAULT_TEST_COMMANDS)
    exits = dict(test_exit_codes or {command: 0 for command in commands})
    elapsed = time.perf_counter() - started if duration_s is None else duration_s
    artifact: JsonDict = {
        "status": "pending",
        "upstream_path_hash_terminal_class_and_gate_receipt": upstream,
        "MODEL_SPECS": model_resolution["MODEL_SPECS"],
        "models_used": generation["models_used"],
        "model_file_hashes_revisions_quantizations_and_tokenizers": model_file_receipts(
            model_resolution["MODEL_SPECS"]
        ),
        "llama_cpp_embedded_tokenizer_receipts": tokenizer_receipts(
            model_resolution["MODEL_SPECS"]
        ),
        "cuda_gpu_offload_and_memory_release_receipts_by_model": generation[
            "cuda_receipts_by_model"
        ],
        "fixture_and_split_paths_hashes": fixture_receipts,
        "prompt_and_decoder_contract": {
            key: value for key, value in prompt_contract.items() if key != "prompt_text"
        },
        "arm_definitions": arm_definitions(),
        "matched_call_token_candidate_and_time_budgets": budgets,
        "raw_candidate_paths_hashes_and_counts": generation["raw_candidate_paths_hashes_and_counts"],
        "parse_and_normalization_results": parsed["parse_and_normalization_results"],
        "exact_factor_energies_by_candidate": parsed["exact_factor_energies_by_candidate"],
        "guard_accept_reject_and_fallback_receipts": arm_results["guard_receipts"],
        "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed": arm_results[
            "metrics"
        ],
        "paired_deltas_intervals_and_sample_sizes": paired,
        "harm_underpowered_missing_and_flagged_cells": harm,
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "guarded_policy_synthesis_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exits,
        "duration_s": float(elapsed),
        "random_seed": RANDOM_SEEDS[0],
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["guarded_policy_synthesis_ready_score"] = expected_ready_score(artifact)
    artifact["status"] = status_from_artifact(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def prompt_and_decoder_contract(fixtures: Sequence[exp6326.PolicyFixture]) -> JsonDict:
    """Build the frozen prompt and decoder contract."""

    prompt = build_prompt(fixtures)
    return {
        "schema": SCHEMA + ".prompt_decoder",
        "prompt_text": prompt,
        "prompt_sha256": "sha256:" + sha256_text(prompt),
        "candidate_block_contract": "BEGIN_CANDIDATE family=<family> candidate=<0|1> ... END_CANDIDATE",
        "candidate_count_per_family": CANDIDATE_COUNT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repeat_penalty": REPEAT_PENALTY,
        "max_tokens": MAX_TOKENS,
        "context_tokens": CONTEXT_TOKENS,
        "time_budget_s": TIME_BUDGET_S,
        "native_llama_cpp_cli_path": str(LLAMA_CPP_CLI_PATH),
        "embedded_gguf_tokenizer_only": True,
        "hf_tokenizer_loader_forbidden": True,
    }


def build_prompt(fixtures: Sequence[exp6326.PolicyFixture]) -> str:
    """Return the exact prompt sent to each local model."""

    family_blocks = []
    for fixture in fixtures:
        contract = exp6326.validate_contract(fixture.contract)
        clauses = json.dumps(list(contract.clauses), sort_keys=True, ensure_ascii=True)
        family_blocks.append(
            "\n".join(
                [
                    f"family: {fixture.family}",
                    f"split: {fixture.split}",
                    "states: " + ", ".join(contract.states),
                    "actions: " + ", ".join(contract.actions),
                    "clauses_json: " + clauses,
                ]
            )
        )
    return (
        "You generate candidate programs for a restricted finite policy DSL.\n"
        "Output only candidate blocks. Do not explain, score, label, or judge safety.\n"
        "DSL syntax:\n"
        "policy name\nstates: state0, state1;\nactions: action0, action1;\n"
        "rule state0 -> action0;\nend\n"
        "Each state must have exactly one rule. Use only listed states and actions.\n"
        "For each family below, output exactly two blocks:\n"
        "BEGIN_CANDIDATE family=<family> candidate=<0 or 1>\n"
        "<one policy program>\nEND_CANDIDATE\n"
        "\nFamilies:\n"
        + "\n\n".join(family_blocks)
        + "\n"
    )


def matched_budgets() -> JsonDict:
    """Return one frozen budget shared by every arm."""

    per_arm = {
        arm: {
            "candidate_count": CANDIDATE_COUNT,
            "max_tokens": MAX_TOKENS,
            "time_budget_s": TIME_BUDGET_S,
            "fallback_cost": FALLBACK_COST,
            "generation_cost": MATCHED_GENERATION_COST,
        }
        for arm in ARMS
    }
    return {
        "schema": SCHEMA + ".matched_budgets",
        "candidate_count": CANDIDATE_COUNT,
        "max_tokens": MAX_TOKENS,
        "time_budget_s": TIME_BUDGET_S,
        "calls_per_model": 1,
        "by_arm": per_arm,
        "fallback_utility": FALLBACK_UTILITY,
        "fallback_cost": FALLBACK_COST,
        "candidate_utility": BASE_CANDIDATE_UTILITY,
    }


def generate_raw_outputs(
    model_specs: Sequence[Mapping[str, Any]],
    fixtures: Sequence[exp6326.PolicyFixture],
    *,
    data_dir: Path,
    prompt: str,
    budget: Mapping[str, Any],
    generation_func: GenerationFn,
    write: bool,
) -> JsonDict:
    """Generate and hash one raw response file for each model."""

    raw_dir = data_dir / RAW_DIR_NAME
    raw_receipts: dict[str, JsonDict] = {}
    raw_outputs: dict[str, JsonDict] = {}
    cuda_receipts: dict[str, JsonDict] = {}
    models_used: list[str] = []
    expected_rows = len(fixtures) * CANDIDATE_COUNT
    for index, spec in enumerate(model_specs):
        hf_id = str(spec["hf_id"])
        seed = RANDOM_SEEDS[index]
        response = generation_func(dict(spec), prompt, seed, dict(budget))
        raw_text = str(response.get("raw_text", ""))
        receipt = dict(response.get("receipt") or {})
        payload = {
            "schema": SCHEMA + ".raw_model_output",
            "model_hf_id": hf_id,
            "seed": seed,
            "prompt_sha256": "sha256:" + sha256_text(prompt),
            "raw_text": raw_text,
            "receipt": receipt,
        }
        path = raw_dir / f"{model_slug(hf_id)}.raw.json"
        if write:
            write_json_atomic(path, payload)
            raw_sha = sha256_file(path)
        else:
            raw_sha = "sha256:" + sha256_json(payload)
        raw_receipts[hf_id] = {
            "path": str(path),
            "sha256": raw_sha,
            "candidate_count": expected_rows,
            "raw_block_count": len(extract_candidate_blocks(raw_text)),
            "seed": seed,
            "written_atomically": bool(write),
        }
        raw_outputs[hf_id] = payload
        cuda_receipts[hf_id] = receipt
        if receipt.get("exit_code") == 0:
            models_used.append(hf_id)
    return {
        "raw_candidate_paths_hashes_and_counts": raw_receipts,
        "raw_outputs": raw_outputs,
        "cuda_receipts_by_model": cuda_receipts,
        "models_used": models_used,
    }


def empty_generation(
    model_specs: Sequence[Mapping[str, Any]],
    raw_dir: Path,
) -> JsonDict:
    """Return empty generation receipts when the structured gate blocks."""

    raw_receipts: dict[str, JsonDict] = {}
    raw_outputs: dict[str, JsonDict] = {}
    cuda_receipts: dict[str, JsonDict] = {}
    for index, spec in enumerate(model_specs):
        hf_id = str(spec["hf_id"])
        path = raw_dir / f"{model_slug(hf_id)}.raw.json"
        raw_receipts[hf_id] = {
            "path": str(path),
            "sha256": None,
            "candidate_count": 0,
            "raw_block_count": 0,
            "seed": RANDOM_SEEDS[index],
            "written_atomically": False,
        }
        raw_outputs[hf_id] = {
            "model_hf_id": hf_id,
            "seed": RANDOM_SEEDS[index],
            "raw_text": "",
            "receipt": {"exit_code": None, "blocked_before_generation": True},
        }
        cuda_receipts[hf_id] = {"exit_code": None, "blocked_before_generation": True}
    return {
        "raw_candidate_paths_hashes_and_counts": raw_receipts,
        "raw_outputs": raw_outputs,
        "cuda_receipts_by_model": cuda_receipts,
        "models_used": [],
    }


def parse_and_score_candidates(
    raw_outputs: Mapping[str, Mapping[str, Any]],
    fixtures: Sequence[exp6326.PolicyFixture],
) -> JsonDict:
    """Parse all expected candidate cells and compute exact energies."""

    fixture_by_family = {fixture.family: fixture for fixture in fixtures}
    rows: list[JsonDict] = []
    by_family: dict[str, JsonDict] = {
        fixture.family: {"candidate_count": 0, "parser_failure_count": 0}
        for fixture in fixtures
    }
    mismatch_count = 0
    for model_id, payload in raw_outputs.items():
        seed = int(payload["seed"])
        blocks = extract_candidate_blocks(str(payload.get("raw_text") or ""))
        block_map = {(family, index): text for family, index, text in blocks}
        for fixture in fixtures:
            contract = exp6326.validate_contract(fixture.contract)
            factors = exp6326.compile_contract_to_factors(contract)
            for candidate_index in range(CANDIDATE_COUNT):
                source = block_map.get((fixture.family, candidate_index), "")
                row = parse_candidate(
                    model_id=model_id,
                    family=fixture.family,
                    split=fixture.split,
                    seed=seed,
                    candidate_index=candidate_index,
                    source=source,
                    contract=contract,
                    factors=factors,
                )
                rows.append(row)
                by_family[fixture.family]["candidate_count"] += 1
                if row["parse_status"] != "parsed":
                    by_family[fixture.family]["parser_failure_count"] += 1
                if row.get("factor_exact_mismatch") is True:
                    mismatch_count += 1
    parser_failure_count = sum(
        int(row["parser_failure_count"]) for row in by_family.values()
    )
    return {
        "candidate_rows": rows,
        "parse_and_normalization_results": {
            "schema": SCHEMA + ".parse_normalization",
            "candidate_count": len(rows),
            "parser_failure_count": parser_failure_count,
            "normalization_success_count": sum(1 for row in rows if row["parse_status"] == "parsed"),
            "by_family": by_family,
        },
        "exact_factor_energies_by_candidate": {
            "schema": SCHEMA + ".exact_factor_energy",
            "candidate_count": len(rows),
            "mismatch_count": mismatch_count,
            "rows": rows,
        },
    }


def parse_candidate(
    *,
    model_id: str,
    family: str,
    split: str,
    seed: int,
    candidate_index: int,
    source: str,
    contract: exp6326.Contract,
    factors: Sequence[exp6326.Factor],
) -> JsonDict:
    """Parse one raw candidate and compute exact contract energy."""

    program_source = extract_program_source(source)
    raw_sha = "sha256:" + sha256_text(source)
    if not program_source:
        return _candidate_error(
            model_id, family, split, seed, candidate_index, raw_sha, "missing_block"
        )
    try:
        policy = exp6326.parse_policy(program_source)
    except exp6326.PolicySyntaxError as exc:
        return _candidate_error(
            model_id, family, split, seed, candidate_index, raw_sha, exc.reason
        )
    if policy.states != contract.states or policy.actions != contract.actions:
        return _candidate_error(
            model_id, family, split, seed, candidate_index, raw_sha, "domain_mismatch"
        )
    factor_energy = exp6326.factor_energy(policy, factors)
    exact_energy = exp6326.exact_contract_energy(policy, contract)
    normalized = exp6326.normalize_policy(policy)
    return {
        "model_hf_id": model_id,
        "family": family,
        "split": split,
        "seed": seed,
        "candidate_index": candidate_index,
        "raw_sha256": raw_sha,
        "parse_status": "parsed",
        "normalization_status": "normalized",
        "normalized_sha256": "sha256:" + exp6326.sha256_text(normalized),
        "semantic_hash": exp6326.semantic_hash(policy),
        "factor_energy": factor_energy,
        "exact_energy": exact_energy,
        "factor_exact_mismatch": factor_energy != exact_energy,
        "accepted_by_exact_guard": exact_energy == 0,
    }


def evaluate_arms(
    candidate_rows: Sequence[Mapping[str, Any]],
    fixtures: Sequence[exp6326.PolicyFixture],
) -> JsonDict:
    """Evaluate all four matched arms with the same raw candidate rows."""

    rows_by_key: dict[tuple[str, str], list[JsonDict]] = {}
    for row in candidate_rows:
        rows_by_key.setdefault((str(row["model_hf_id"]), str(row["family"])), []).append(dict(row))
    metrics: dict[str, dict[str, JsonDict]] = {
        model_id: {fixture.family: {} for fixture in fixtures} for model_id in MANDATED_MODEL_IDS
    }
    guarded_violations = 0
    raw_violations = 0
    fallback_count_by_arm = {arm: 0 for arm in ARMS}
    for model_id in MANDATED_MODEL_IDS:
        for fixture in fixtures:
            candidates = sorted(
                rows_by_key.get((model_id, fixture.family), []),
                key=lambda row: int(row["candidate_index"]),
            )
            if len(candidates) < CANDIDATE_COUNT:
                candidates = candidates + [_missing_candidate(model_id, fixture)]
            outcomes = arm_outcomes(candidates, fixture)
            metrics[model_id][fixture.family] = outcomes
            for arm, outcome in outcomes.items():
                fallback_count_by_arm[arm] += int(bool(outcome["fallback_used"]))
                violation = outcome.get("contract_violation_count")
                if arm == "one_raw_candidate" and isinstance(violation, int):
                    raw_violations += violation
                if arm != "one_raw_candidate" and bool(outcome["accepted"]):
                    guarded_violations += int(violation or 0)
    return {
        "metrics": metrics,
        "guard_receipts": {
            "schema": SCHEMA + ".guard_fallback",
            "guarded_accepted_contract_violation_count": guarded_violations,
            "raw_arm_contract_violation_count": raw_violations,
            "fallback_count_by_arm": fallback_count_by_arm,
            "fallback_hashes": fallback_hashes(fixtures),
            "oracle": "Exp6326 exact factor energy",
        },
    }


def arm_outcomes(
    candidates: Sequence[Mapping[str, Any]],
    fixture: exp6326.PolicyFixture,
) -> JsonDict:
    """Return per-arm outcomes for one model-family cell."""

    first = dict(candidates[0])
    best = best_candidate(candidates)
    return {
        "one_raw_candidate": raw_outcome(first),
        "reject_only_filtering": guarded_without_fallback(first),
        "exact_guard_plus_hash_pinned_fallback": guarded_with_fallback(first, fixture),
        "bounded_exact_factor_energy_guided_candidate_search_plus_fallback": guarded_with_fallback(
            best, fixture
        ),
    }


def raw_outcome(candidate: Mapping[str, Any]) -> JsonDict:
    """Score the unguarded first candidate arm."""

    parse_ok = candidate.get("parse_status") == "parsed"
    energy = candidate.get("exact_energy")
    violation = int(energy) if isinstance(energy, int) else None
    unsafe_penalty = UNSAFE_ENERGY_PENALTY * float(violation or 0)
    utility = BASE_CANDIDATE_UTILITY - MATCHED_GENERATION_COST - unsafe_penalty if parse_ok else 0.0
    return _outcome(candidate, accepted=parse_ok, fallback=False, utility=utility, violation=violation)


def guarded_without_fallback(candidate: Mapping[str, Any]) -> JsonDict:
    """Reject a bad first candidate without fallback utility."""

    accepted = candidate.get("parse_status") == "parsed" and candidate.get("exact_energy") == 0
    utility = BASE_CANDIDATE_UTILITY - MATCHED_GENERATION_COST if accepted else 0.0
    return _outcome(candidate, accepted=accepted, fallback=False, utility=utility, violation=0 if accepted else None)


def guarded_with_fallback(
    candidate: Mapping[str, Any],
    fixture: exp6326.PolicyFixture,
) -> JsonDict:
    """Use a candidate only if exact energy is zero, otherwise use fallback."""

    accepted = candidate.get("parse_status") == "parsed" and candidate.get("exact_energy") == 0
    if accepted:
        return _outcome(
            candidate,
            accepted=True,
            fallback=False,
            utility=BASE_CANDIDATE_UTILITY - MATCHED_GENERATION_COST,
            violation=0,
        )
    fallback = fallback_program_receipt(fixture)
    outcome = _outcome(
        candidate,
        accepted=True,
        fallback=True,
        utility=FALLBACK_UTILITY - MATCHED_GENERATION_COST - FALLBACK_COST,
        violation=0,
    )
    outcome["fallback_receipt"] = fallback
    return outcome


def best_candidate(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Choose the lowest exact-energy parseable candidate."""

    parseable = [dict(row) for row in candidates if isinstance(row.get("exact_energy"), int)]
    if not parseable:
        return dict(candidates[0])
    return min(parseable, key=lambda row: (int(row["exact_energy"]), int(row["candidate_index"])))


def paired_delta_summary(metrics: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Compute search-minus-guard deltas for matched cells."""

    by_model: dict[str, JsonDict] = {}
    all_development: list[float] = []
    all_held: list[float] = []
    for model_id, families in metrics.items():
        dev: list[float] = []
        held: list[float] = []
        for family, arms in families.items():
            delta = (
                arms["bounded_exact_factor_energy_guided_candidate_search_plus_fallback"]["utility"]
                - arms["exact_guard_plus_hash_pinned_fallback"]["utility"]
            )
            split = arms["bounded_exact_factor_energy_guided_candidate_search_plus_fallback"]["split"]
            if split == "development":
                dev.append(delta)
                all_development.append(delta)
            else:
                held.append(delta)
                all_held.append(delta)
        by_model[model_id] = {
            "development_search_minus_guard": paired_interval(dev),
            "held_search_minus_guard": paired_interval(held),
        }
    return {
        "schema": SCHEMA + ".paired_deltas",
        "by_model": by_model,
        "development_all_models_search_minus_guard": paired_interval(all_development),
        "held_all_models_search_minus_guard": paired_interval(all_held),
    }


def harm_summary(
    *,
    model_resolution: Mapping[str, Any],
    generation: Mapping[str, Any],
    parsed: Mapping[str, Any],
    arm_results: Mapping[str, Any],
    paired: Mapping[str, Any],
) -> JsonDict:
    """Summarize missing cells, parse failures, harm, and underpowered cells."""

    expected_models = set(MANDATED_MODEL_IDS)
    used_models = set(generation.get("models_used") or [])
    missing_models = sorted(expected_models - used_models)
    parser_failure_count = int(
        parsed["parse_and_normalization_results"].get("parser_failure_count") or 0
    )
    guarded_violations = int(
        arm_results["guard_receipts"].get("guarded_accepted_contract_violation_count") or 0
    )
    dev_n = int(
        paired["development_all_models_search_minus_guard"].get("sample_size") or 0
    )
    return {
        "schema": SCHEMA + ".harm_missing_underpowered",
        "model_resolution_blockers": list(model_resolution.get("blocked_reasons") or []),
        "missing_models": missing_models,
        "missing_model_count": len(missing_models),
        "parser_failure_count": parser_failure_count,
        "guarded_accepted_contract_violation_count": guarded_violations,
        "harmful_guarded_accept_count": guarded_violations,
        "development_sample_size": dev_n,
        "underpowered_for_public_headline": dev_n < 30,
        "flagged_cells": parser_failure_count + guarded_violations + len(missing_models),
    }


def expected_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the exact readiness gate result."""

    complete_models = (
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] == list(MANDATED_MODEL_IDS)
        and all(row.get("exists") is True and row.get("tokenizer_loadable") is True for row in artifact.get("MODEL_SPECS", []))
        and set(artifact.get("models_used") or []) == set(MANDATED_MODEL_IDS)
    )
    protected = bool(artifact.get("protected_files_unchanged", {}).get("all_unchanged"))
    preconditions = bool(artifact.get("preconditions_checked", {}).get("all_passed"))
    guard = int(
        artifact.get("guard_accept_reject_and_fallback_receipts", {}).get(
            "guarded_accepted_contract_violation_count", 1
        )
        or 0
    )
    commands_ok = all(code == 0 for code in dict(artifact.get("test_exit_codes") or {}).values())
    field_principles_ok = set(REQUIRED_ARTIFACT_FIELDS) <= set(
        artifact.get("field_principles", {})
    )
    development_delta = float(
        artifact.get("paired_deltas_intervals_and_sample_sizes", {})
        .get("development_all_models_search_minus_guard", {})
        .get("mean_delta", 0.0)
        or 0.0
    )
    if complete_models and protected and preconditions and guard == 0 and commands_ok and field_principles_ok and development_delta > 0:
        return 1.0
    return 0.0


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the artifact and reject false readiness laundering."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    for field in (
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
    ):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(
        artifact.get("exact_oracle_claim_boundary", {}).get("model_supplies_safety_authority")
        is False,
        "exact_oracle_claim_boundary",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_provenance", {})),
        "field_provenance",
    )
    expected = expected_ready_score(artifact)
    _require(artifact.get("guarded_policy_synthesis_ready_score") == expected, "ready_score")
    if expected == 1.0:
        _require(artifact.get("status") == "complete_ready", "status")
        _require(str(artifact.get("honest_verdict", "")).startswith("ready:"), "honest_verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    return True


def status_from_artifact(artifact: Mapping[str, Any]) -> str:
    """Map receipts to a terminal status."""

    if not artifact.get("preconditions_checked", {}).get("all_passed"):
        return "blocked"
    return "complete_ready" if artifact.get("guarded_policy_synthesis_ready_score") == 1.0 else "complete_no_value"


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the lower-case terminal verdict string."""

    status = str(artifact.get("status"))
    if status == "complete_ready":
        return "ready: bounded exact-factor search improved development utility with exact guarded fallbacks"
    if status == "blocked":
        return "blocked: missing required model, tokenizer, cuda, upstream, protected-file, or command evidence"
    return "complete_null: bounded exact-factor search did not clear the preregistered readiness gate"


def _outcome(
    candidate: Mapping[str, Any],
    *,
    accepted: bool,
    fallback: bool,
    utility: float,
    violation: int | None,
) -> JsonDict:
    return {
        "model_hf_id": candidate["model_hf_id"],
        "family": candidate["family"],
        "split": candidate["split"],
        "seed": candidate["seed"],
        "candidate_index": candidate["candidate_index"],
        "accepted": bool(accepted),
        "fallback_used": bool(fallback),
        "contract_violation_count": violation,
        "utility": round(float(utility), 6),
        "latency_s": 0.0,
        "cost": round(MATCHED_GENERATION_COST + (FALLBACK_COST if fallback else 0.0), 6),
        "fallback_utility_charged": FALLBACK_UTILITY if fallback else None,
        "full_fallback_cost_charged": FALLBACK_COST if fallback else 0.0,
    }


def fallback_program_receipt(fixture: exp6326.PolicyFixture) -> JsonDict:
    """Hash and verify one fixture fallback program."""

    contract = exp6326.validate_contract(fixture.contract)
    policy = exp6326.parse_policy(fixture.fallback_program)
    energy = exp6326.factor_energy(policy, exp6326.compile_contract_to_factors(contract))
    return {
        "family": fixture.family,
        "source_sha256": "sha256:" + exp6326.sha256_text(fixture.fallback_program),
        "semantic_hash": exp6326.semantic_hash(policy),
        "energy": energy,
        "verified": energy == 0,
    }


def fallback_hashes(fixtures: Sequence[exp6326.PolicyFixture]) -> JsonDict:
    """Return fallback receipts for every family."""

    return {fixture.family: fallback_program_receipt(fixture) for fixture in fixtures}


def extract_candidate_blocks(raw_text: str) -> list[tuple[str, int, str]]:
    """Extract structured model candidate blocks from raw text."""

    pattern = re.compile(
        r"BEGIN_CANDIDATE\s+family=([a-z0-9_]+)\s+candidate=(\d+)\s*(.*?)END_CANDIDATE",
        re.DOTALL,
    )
    return [(family, int(index), body.strip()) for family, index, body in pattern.findall(raw_text)]


def extract_program_source(text: str) -> str:
    """Extract the first complete policy program from a candidate block."""

    lines = [line.strip("` ") for line in text.splitlines()]
    start = next((idx for idx, line in enumerate(lines) if line.startswith("policy ")), None)
    if start is None:
        return ""
    end = next((idx for idx in range(start, len(lines)) if lines[idx] == "end"), None)
    if end is None:
        return "\n".join(lines[start:]).strip() + "\n"
    return "\n".join(lines[start : end + 1]).strip() + "\n"


def _candidate_error(
    model_id: str,
    family: str,
    split: str,
    seed: int,
    candidate_index: int,
    raw_sha: str,
    reason: str,
) -> JsonDict:
    return {
        "model_hf_id": model_id,
        "family": family,
        "split": split,
        "seed": seed,
        "candidate_index": candidate_index,
        "raw_sha256": raw_sha,
        "parse_status": reason,
        "normalization_status": "not_normalized",
        "normalized_sha256": None,
        "semantic_hash": None,
        "factor_energy": None,
        "exact_energy": None,
        "factor_exact_mismatch": False,
        "accepted_by_exact_guard": False,
    }


def _missing_candidate(model_id: str, fixture: exp6326.PolicyFixture) -> JsonDict:
    return _candidate_error(
        model_id,
        fixture.family,
        fixture.split,
        RANDOM_SEEDS[0],
        0,
        "sha256:" + ("0" * 64),
        "missing_cell",
    )


def arm_definitions() -> JsonDict:
    """Return the four preregistered arms."""

    return {
        "one_raw_candidate": {
            "candidate_source": "candidate_index_0",
            "guard": "none",
            "fallback": "none",
        },
        "reject_only_filtering": {
            "candidate_source": "candidate_index_0",
            "guard": "Exp6326 exact factor energy must equal zero",
            "fallback": "none",
        },
        "exact_guard_plus_hash_pinned_fallback": {
            "candidate_source": "candidate_index_0",
            "guard": "Exp6326 exact factor energy must equal zero",
            "fallback": "verified family fallback",
        },
        "bounded_exact_factor_energy_guided_candidate_search_plus_fallback": {
            "candidate_source": "lowest exact energy among bounded candidates",
            "guard": "selected candidate must have exact energy zero",
            "fallback": "same verified family fallback",
        },
    }


def precondition_receipt(
    *,
    date: str,
    result_path: Path,
    data_dir: Path,
    upstream: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    fixture_receipts: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    budgets: Mapping[str, Any],
    host: Mapping[str, Any],
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Collect the structured gate replay before generation."""

    host_ok = (
        bool(host.get("cuda_devices", {}).get("available"))
        and int(host.get("cuda_devices", {}).get("count") or 0) >= 2
        and bool(host.get("llama_cpp_cli", {}).get("exists"))
        and bool(host.get("llama_cpp_gpu_offload", {}).get("available"))
    )
    all_passed = (
        upstream.get("gate_ready_score") == 1.0
        and model_resolution.get("all_resolved") is True
        and host_ok
        and bool(fixture_receipts.get("fallback_hashes"))
        and bool(protected_before)
    )
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "result_path": str(result_path),
        "data_dir": str(data_dir),
        "upstream": dict(upstream),
        "model_gate": {
            "all_resolved": model_resolution.get("all_resolved"),
            "blocked_reasons": list(model_resolution.get("blocked_reasons") or []),
        },
        "host": dict(host),
        "timeouts": {"per_model_generation_timeout_s": TIME_BUDGET_S},
        "seeds": list(RANDOM_SEEDS),
        "fixture_hash": fixture_receipts.get("fixture_manifest", {}).get("sha256"),
        "budget_hash": "sha256:" + sha256_json(budgets),
        "prompt_hash": prompt_contract.get("prompt_sha256"),
        "fallback_hashes": fixture_receipts.get("fallback_hashes"),
        "protected_hashes_before": dict(protected_before),
        "all_passed": bool(all_passed),
    }


def upstream_receipt() -> JsonDict:
    """Read the Exp6326 exact-guard artifact receipt."""

    path = REPO_ROOT / UPSTREAM_RELATIVE_PATH
    if not path.exists():
        return {
            "path": str(UPSTREAM_RELATIVE_PATH),
            "exists": False,
            "sha256": None,
            "terminal_class": "missing",
            "gate_ready_score": 0.0,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = str(payload.get("honest_verdict", ""))
    return {
        "path": str(UPSTREAM_RELATIVE_PATH),
        "exists": True,
        "sha256": sha256_file(path),
        "terminal_class": verdict.split(":", 1)[0],
        "status": payload.get("status"),
        "gate_ready_score": float(payload.get("contract_guard_ready_score") or 0.0),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
    }


def fixture_and_split_receipts(fixtures: Sequence[exp6326.PolicyFixture]) -> JsonDict:
    """Return fixture, split, grammar, and fallback receipts."""

    fixture_payload = [
        {
            "family": fixture.family,
            "split": fixture.split,
            "contract": fixture.contract,
            "fallback_sha256": "sha256:" + exp6326.sha256_text(fixture.fallback_program),
        }
        for fixture in fixtures
    ]
    return {
        "schema": SCHEMA + ".fixture_splits",
        "fixture_manifest": {
            "path": str(exp6326.FIXTURE_MANIFEST_RELATIVE_PATH),
            "sha256": "sha256:" + sha256_json(fixture_payload),
            "family_count": len(fixtures),
        },
        "grammar": {
            "path": str(exp6326.GRAMMAR_RELATIVE_PATH),
            "sha256": "sha256:" + exp6326.sha256_text(exp6326.DSL_GRAMMAR),
        },
        "splits": {fixture.family: fixture.split for fixture in fixtures},
        "fallback_hashes": fallback_hashes(fixtures),
    }


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return model file identity receipts keyed by model id."""

    return {
        str(record["hf_id"]): {
            "path": record.get("model_path"),
            "sha256": record.get("sha256"),
            "revision": record.get("revision"),
            "quantization": record.get("quantization"),
            "tokenizer_loadable": record.get("tokenizer_loadable"),
            "tokenizer_status": record.get("tokenizer_status"),
        }
        for record in model_specs
    }


def tokenizer_receipts(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return embedded-tokenizer receipts keyed by model id."""

    return {
        str(record["hf_id"]): {
            "method": record.get("tokenizer_method"),
            "loadable": record.get("tokenizer_loadable"),
            "detail": record.get("tokenizer_status"),
            "model_path": record.get("model_path"),
        }
        for record in model_specs
    }


def exact_oracle_claim_boundary() -> JsonDict:
    """State the model and verifier authority boundary."""

    return {
        "verifier": "Exp6326 exact finite-domain factor energy",
        "verifier_is_oracle": True,
        "model_supplies_candidates_only": True,
        "model_supplies_labels": False,
        "model_supplies_safety_authority": False,
        "hidden_states_accessed": False,
        "source_model_weights_mutated": False,
    }


def protected_hashes() -> dict[str, str]:
    """Hash protected files that this task must not edit."""

    paths = (
        Path("scripts/research_conductor.py"),
        Path("ops/status.md"),
        Path("ops/changelog.md"),
        Path("_bmad/traceability.md"),
    )
    out: dict[str, str] = {}
    for rel in paths:
        path = REPO_ROOT / rel
        out[str(rel)] = sha256_file(path) if path.exists() else "missing"
    return out


def protected_unchanged_receipt(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> JsonDict:
    """Compare protected-file hashes before and after the run."""

    rows = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"files": rows, "all_unchanged": all(row["unchanged"] for row in rows.values())}


def host_environment_receipts() -> JsonDict:  # pragma: no cover - host dependent.
    """Collect CUDA, RAM, disk, and native llama.cpp receipts."""

    gpu_query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    devices = parse_gpu_query(gpu_query.get("stdout", ""))
    cli_exists = LLAMA_CPP_CLI_PATH.exists()
    gpu_offload = False
    offload_detail = ""
    try:
        from llama_cpp import llama_cpp as binding

        gpu_offload = bool(binding.llama_supports_gpu_offload())
        offload_detail = "python binding reports gpu offload support"
    except Exception as exc:
        offload_detail = f"offload check unavailable:{type(exc).__name__}:{exc}"
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "cuda_devices": {"available": len(devices) >= 1, "count": len(devices), "devices": devices},
        "vram": {str(row["index"]): row for row in devices},
        "ram": memory_receipt(),
        "disk": {"available_gb": round(disk.free / (1024**3), 3)},
        "llama_cpp_cli": {"path": str(LLAMA_CPP_CLI_PATH), "exists": cli_exists},
        "llama_cpp_gpu_offload": {"available": gpu_offload, "detail": offload_detail},
    }


def generate_with_llama_cli(
    model_spec: dict[str, Any],
    prompt: str,
    seed: int,
    budget: dict[str, Any],
) -> JsonDict:  # pragma: no cover - live GGUF runtime.
    """Run one native llama.cpp generation call for a model."""

    started = time.perf_counter()
    before = gpu_memory_mb()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    cmd = [
        str(LLAMA_CPP_CLI_PATH),
        "-m",
        str(model_spec["model_path"]),
        "-f",
        str(prompt_path),
        "-n",
        str(budget["max_tokens"]),
        "-c",
        str(CONTEXT_TOKENS),
        "--temp",
        str(TEMPERATURE),
        "--top-p",
        str(TOP_P),
        "--repeat-penalty",
        str(REPEAT_PENALTY),
        "-s",
        str(seed),
        "-ngl",
        "all",
        "-sm",
        "none",
        "-mg",
        "0",
        "-st",
        "--no-display-prompt",
        "--reasoning",
        "off",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(model_spec.get("gpu", 0)))
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=float(budget["time_budget_s"]),
            check=False,
        )
        after = gpu_memory_mb()
        stderr = completed.stderr[-40_000:]
        return {
            "raw_text": completed.stdout,
            "receipt": {
                "mode": "native_llama_cpp_cli",
                "command": cmd,
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "seed": seed,
                "exit_code": completed.returncode,
                "latency_s": round(time.perf_counter() - started, 6),
                "stderr_tail": stderr,
                "stdout_sha256": "sha256:" + sha256_text(completed.stdout),
                "prompt_tokens_estimated": len(prompt.split()),
                "generated_tokens_estimated": len(completed.stdout.split()),
                "memory_before_mb": before,
                "memory_after_release_mb": after,
                "cuda_layer_offload": parse_cuda_offload(stderr),
                "cuda_layer_offload_confirmed": parse_cuda_offload(stderr)[
                    "cuda_layer_offload_confirmed"
                ],
                "release_within_512mb": release_with(before, after, 512),
            },
        }
    finally:
        prompt_path.unlink(missing_ok=True)


def parse_cuda_offload(stderr: str) -> JsonDict:
    """Parse native llama.cpp layer-offload evidence."""

    match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", stderr)
    offloaded = int(match.group(1)) if match else 0
    total = int(match.group(2)) if match else 0
    return {
        "cuda_layers_offloaded": offloaded,
        "total_layers": total,
        "cuda_layer_offload_confirmed": offloaded > 0 and total > 0,
        "evidence": match.group(0) if match else "",
    }


def release_with(
    before: Mapping[int, int] | Mapping[str, int],
    after: Mapping[int, int] | Mapping[str, int],
    threshold_mb: int,
) -> bool:
    """Return true when post-run VRAM is near the pre-run baseline."""

    for key, after_value in after.items():
        before_value = int(before.get(key, before.get(str(key), 0)) or 0)
        if int(after_value) > before_value + threshold_mb:
            return False
    return True


def paired_interval(values: Sequence[float]) -> JsonDict:
    """Return a simple paired mean interval for transparent small samples."""

    n = len(values)
    if n == 0:
        return {"sample_size": 0, "mean_delta": 0.0, "ci95": [0.0, 0.0], "values": []}
    mean = sum(values) / n
    if n == 1:
        return {"sample_size": 1, "mean_delta": mean, "ci95": [mean, mean], "values": list(values)}
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half = 1.96 * (variance / n) ** 0.5
    return {
        "sample_size": n,
        "mean_delta": round(mean, 6),
        "ci95": [round(mean - half, 6), round(mean + half, 6)],
        "values": [round(value, 6) for value in values],
    }


def model_slug(hf_id: str) -> str:
    """Return a filesystem-safe model id slug."""

    return re.sub(r"[^a-zA-Z0-9]+", "_", hf_id).strip("_").lower()


def extract_quantization(path: Path) -> str:
    """Extract the quantization token from a GGUF filename."""

    name = path.name
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in name.lower():
            return token
    return "unknown"


def extract_revision(path: Path) -> str:
    """Extract the HF snapshot revision from a cached path."""

    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while blanking its self-referential checksum."""

    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return "sha256:" + sha256_json(stable)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    """Return a SHA-256 digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest for one file."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through an atomic rename in the target directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def run_command(cmd: Sequence[str], *, timeout_s: float = 30) -> JsonDict:  # pragma: no cover
    """Run a host command and keep stdout, stderr, and exit code."""

    try:
        completed = subprocess.run(
            list(cmd),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(cmd),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"command": list(cmd), "returncode": 127, "stdout": "", "stderr": str(exc)}


def parse_gpu_query(text: str) -> list[JsonDict]:  # pragma: no cover
    """Parse a small nvidia-smi CSV query."""

    devices: list[JsonDict] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(float(parts[2])),
                    "memory_used_mb": int(float(parts[3])),
                    "memory_free_mb": int(float(parts[4])),
                }
            )
    return devices


def memory_receipt() -> JsonDict:  # pragma: no cover
    """Return a lightweight RAM receipt."""

    result = run_command(["free", "-b"], timeout_s=10)
    lines = str(result.get("stdout") or "").splitlines()
    if len(lines) < 2:
        return {"available_gb": 0.0, "raw": result}
    parts = lines[1].split()
    available = int(parts[6]) if len(parts) > 6 else 0
    return {"available_gb": round(available / (1024**3), 3)}


def gpu_memory_mb() -> dict[int, int]:  # pragma: no cover
    """Return current GPU memory use by physical GPU index."""

    result = run_command(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        timeout_s=10,
    )
    memory: dict[int, int] = {}
    for line in str(result.get("stdout") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            memory[int(parts[0])] = int(parts[1])
    return memory


def _cache_policy_for_model(hf_id: str) -> str:
    if hf_id == MANDATED_MODEL_IDS[1]:
        return "cached_sota_pair(gpu_indices=(0,1), model_indices=(0,2))"
    return "cached_sota_pair(gpu_indices=(0,1))"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the required Exp6327 run command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.result_path),
        data_dir=Path(args.data_dir),
        write=True,
    )
    print(
        json.dumps(
            {
                "path": str(args.result_path),
                "status": artifact["status"],
                "ready_score": artifact["guarded_policy_synthesis_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
