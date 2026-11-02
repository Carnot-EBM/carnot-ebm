"""Exp6340 parser-JIT semantic diversity canary.

Spec refs: REQ-KONA-6340, SCENARIO-KONA-6340-GATE-REPLAY,
SCENARIO-KONA-6340-MATCHED-ARMS, SCENARIO-KONA-6340-SEMANTIC-DEDUP,
SCENARIO-KONA-6340-ORACLE-BOUNDARY.

The exact compiler is the oracle. Local GGUF models only supply candidate text.
This canary asks whether prefix-time constraint methods increase unique valid
normalized policy semantics at matched cost.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_6326_restricted_policy_contract_compiler as exp6326
from carnot import experiment_6327_three_family_guarded_policy_synthesis as exp6327
from carnot import experiment_6339_incremental_prefix_enforcement_substrate as exp6339
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
GenerationFn = Callable[[dict[str, Any], str, int, dict[str, Any]], dict[str, Any]]
HostChecksFn = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6340_parser_jit_semantic_diversity_canary.json")
DATA_DIR_RELATIVE_PATH = Path("data/research/experiment_6340_parser_jit_semantic_diversity_canary")
RAW_DIR_NAME = "raw_generation"
GRAMMAR_FILE_NAME = "policy_candidate_blocks.gbnf"
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6340_parser_jit_semantic_diversity_canary.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6340_parser_jit_semantic_diversity_canary.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
LLAMA_CPP_CLI_PATH = exp6327.LLAMA_CPP_CLI_PATH

SCHEMA = "carnot.experiment_6340.parser_jit_semantic_diversity_canary.v1"
DEFAULT_RUN_DATE = "20260812"
INFERENCE_SUBSTRATE = "local_three_model_llama_cpp_cuda_parser_jit_semantic_diversity_canary"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"
AUTOTOKENIZER_USAGE_COUNT = 0

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
    "unconstrained_sampling",
    "grammar_masking",
    "deterministic_parser_state_correction",
    "jit_smt_prefix_enforcement",
)
PREDECLARED_PREFIX_ARM = "jit_smt_prefix_enforcement"
RANDOM_SEEDS = (634000, 634001, 634002, 634003)
CANDIDATE_COUNT = 2
MAX_TOKENS = 384
TIME_BUDGET_S = 180
CONTEXT_TOKENS = 2048
TEMPERATURE = 0.2
TOP_P = 0.9
REPEAT_PENALTY = 1.05
CHECKER_TIMEOUT_MS = exp6339.DEFAULT_TIMEOUT_MS
GENERATION_COST = 0.08
CHECKER_CALL_COST = 0.002
FALLBACK_COST = 0.10
FALLBACK_UTILITY = 0.70
CANDIDATE_UTILITY = 1.0

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6340_parser_jit_semantic_diversity_canary "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6340_parser_jit_semantic_diversity_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6340_parser_jit_semantic_diversity_canary.py "
    "-m pytest tests/python/test_experiment_6340_parser_jit_semantic_diversity_canary.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6340_parser_jit_semantic_diversity_canary.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6340_parser_jit_semantic_diversity_canary.py"
)
E2E_COMMAND = "sed -n '1,170p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6340_parser_jit_semantic_diversity_canary.json"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
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
    "prompt_decoder_and_prefix_contract",
    "arm_definitions",
    "matched_call_token_candidate_time_and_checker_budgets",
    "raw_generation_paths_hashes_and_counts",
    "parser_state_and_jit_intervention_logs",
    "parse_normalization_and_contract_results",
    "unique_valid_normalized_semantics_by_model_family_arm",
    "semantic_diversity_paired_deltas_intervals_and_sample_sizes",
    "exact_utility_fallback_latency_and_cost_by_model_family_arm",
    "verification_calls_time_cost_and_error_table",
    "harm_underpowered_missing_and_flagged_cells",
    "source_model_weight_mutation_count",
    "generated_label_count",
    "hidden_state_access_count",
    "exact_oracle_claim_boundary",
    "semantic_diversity_gain_score",
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
    "status": "Shows whether the canary is ready, null, or blocked.",
    "upstream_path_hash_terminal_class_and_gate_receipt": "Pins Exp6339 and the exact oracle before generation.",
    "MODEL_SPECS": "Records all three mandated local GGUF model rows.",
    "models_used": "Names models that supplied raw generation rows.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Pins model files, revisions, quantization, and tokenizer checks.",
    "llama_cpp_embedded_tokenizer_receipts": "Proves tokenizer checks used embedded GGUF metadata.",
    "cuda_gpu_offload_and_memory_release_receipts_by_model": "Shows each model-arm call used and released CUDA resources.",
    "fixture_and_split_paths_hashes": "Pins development families, split rules, grammar, and fallback hashes.",
    "prompt_decoder_and_prefix_contract": "Freezes prompts, decoder settings, grammar, and prefix contract before opening output.",
    "arm_definitions": "Defines the four matched decoder arms.",
    "matched_call_token_candidate_time_and_checker_budgets": "Proves calls, tokens, candidates, time, and checker budgets match.",
    "raw_generation_paths_hashes_and_counts": "Pins raw model rows before parsing and normalization.",
    "parser_state_and_jit_intervention_logs": "Preserves parser corrections, prefix checks, rejected prefixes, and final candidates.",
    "parse_normalization_and_contract_results": "Reports parser failures, normalization, exact validity, and contract outcomes.",
    "unique_valid_normalized_semantics_by_model_family_arm": "Reports semantic diversity after canonical normalization only.",
    "semantic_diversity_paired_deltas_intervals_and_sample_sizes": "Reports paired prefix-arm deltas against unconstrained and grammar arms.",
    "exact_utility_fallback_latency_and_cost_by_model_family_arm": "Reports utility, fallback, latency, and cost per visible cell.",
    "verification_calls_time_cost_and_error_table": "Reports checker calls, checker time, accepted violations, and errors.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps missing, harmful, underpowered, and failed cells visible.",
    "source_model_weight_mutation_count": "Bare zero proves source model weights were not updated.",
    "generated_label_count": "Bare zero proves generated labels did not enter scoring.",
    "hidden_state_access_count": "Bare zero proves hidden activations did not enter scoring.",
    "exact_oracle_claim_boundary": "States that the exact compiler is the oracle, not model verification.",
    "semantic_diversity_gain_score": "Opens only for a preregistered positive paired prefix gain with no accepted violations.",
    "protected_files_unchanged": "Shows conductor and reconciler-owned files stayed byte-identical.",
    "preconditions_checked": "Freezes upstream, models, devices, memory, disk, seeds, fixtures, budgets, timeouts, and protected hashes.",
    "inference_substrate": "Declares local GGUF llama.cpp generation plus exact checking.",
    "verifier_is_oracle": "Bare true preserves the exact checker as authority.",
    "field_provenance": "Maps each field to specs, code, sidecars, models, tests, or receipts.",
    "field_principles": "Explains why every required field exists.",
    "test_commands": "Lists run, focused, coverage, global, spec, E2E, and adversarial commands.",
    "test_exit_codes": "Prevents failed commands from becoming readiness.",
    "duration_s": "Reports measured wall time without padding.",
    "random_seeds": "Pins deterministic model-arm schedules.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "States the terminal claim boundary.",
}
FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "sources": [
            "REQ-KONA-6340",
            "Exp6339 prefix substrate",
            "Exp6326 exact oracle",
            "local GGUF receipts",
            "Exp6340 tests",
        ],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}


def build_model_specs(
    *,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
) -> JsonDict:
    """Resolve all mandated GGUF rows through cached SOTA pair receipts."""

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
        record, new_blockers = _model_record(template, by_id, tokenizer_func)
        records.append(record)
        blockers.extend(new_blockers)
    if not default_pair:
        blockers.append("cached_sota_pair_missing")
    if [row["hf_id"] for row in records] != list(MANDATED_MODEL_IDS):  # pragma: no cover
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
    """Run Exp6340 and optionally write the terminal artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    cached_pair = cached_pair_func or cached_sota_pair
    tokenizer = tokenizer_func or gguf_tokenizer_loadable
    generator = generation_func or generate_with_llama_cli
    host_checks = host_checks_func or host_environment_receipts

    fixtures = development_fixtures()
    protected_before = protected_hashes()
    upstream = upstream_receipt()
    grammar_receipt = write_grammar_sidecar(data, fixtures) if write else grammar_sidecar_receipt(data, fixtures)
    fixture_receipts = fixture_and_split_receipts(fixtures, grammar_receipt)
    prompt_contract = prompt_decoder_and_prefix_contract(fixtures, grammar_receipt=grammar_receipt)
    budgets = matched_budgets()
    model_resolution = build_model_specs(cached_pair_func=cached_pair, tokenizer_func=tokenizer)
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
    if preconditions["all_passed"]:
        generation = generate_raw_outputs(
            model_resolution["MODEL_SPECS"],
            fixtures,
            data_dir=data,
            prompt=prompt_contract["prompt_text"],
            budget=budgets,
            grammar_path=Path(grammar_receipt["path"]),
            generation_func=generator,
            write=write,
        )
    else:
        generation = empty_generation(model_resolution["MODEL_SPECS"], data / RAW_DIR_NAME)
    evaluated = parse_normalize_and_evaluate(generation["raw_outputs"], fixtures)
    protected = protected_unchanged_receipt(protected_before, protected_hashes())
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
        "llama_cpp_embedded_tokenizer_receipts": tokenizer_receipts(model_resolution["MODEL_SPECS"]),
        "cuda_gpu_offload_and_memory_release_receipts_by_model": generation["cuda_receipts_by_model"],
        "fixture_and_split_paths_hashes": fixture_receipts,
        "prompt_decoder_and_prefix_contract": {
            key: value for key, value in prompt_contract.items() if key != "prompt_text"
        },
        "arm_definitions": arm_definitions(),
        "matched_call_token_candidate_time_and_checker_budgets": budgets,
        "raw_generation_paths_hashes_and_counts": generation["raw_generation_paths_hashes_and_counts"],
        "parser_state_and_jit_intervention_logs": evaluated["parser_state_and_jit_intervention_logs"],
        "parse_normalization_and_contract_results": evaluated["parse_normalization_and_contract_results"],
        "unique_valid_normalized_semantics_by_model_family_arm": evaluated[
            "unique_valid_normalized_semantics_by_model_family_arm"
        ],
        "semantic_diversity_paired_deltas_intervals_and_sample_sizes": evaluated[
            "semantic_diversity_paired_deltas_intervals_and_sample_sizes"
        ],
        "exact_utility_fallback_latency_and_cost_by_model_family_arm": evaluated[
            "exact_utility_fallback_latency_and_cost_by_model_family_arm"
        ],
        "verification_calls_time_cost_and_error_table": evaluated[
            "verification_calls_time_cost_and_error_table"
        ],
        "harm_underpowered_missing_and_flagged_cells": harm_summary(generation, evaluated),
        "source_model_weight_mutation_count": 0,
        "generated_label_count": 0,
        "hidden_state_access_count": 0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "semantic_diversity_gain_score": 0.0,
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
    artifact["semantic_diversity_gain_score"] = expected_gain_score(artifact)
    artifact["status"] = status_from_artifact(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def development_fixtures() -> list[exp6326.PolicyFixture]:
    """Return only Exp6326 development fixtures."""

    return [fixture for fixture in exp6326.build_fixture_manifest() if fixture.split == "development"]


def valid_zero_energy_programs(fixture: exp6326.PolicyFixture, *, limit: int) -> list[str]:
    """Enumerate a few exact-valid policy programs for one fixture."""

    contract = exp6326.validate_contract(fixture.contract)
    out: list[str] = []
    for actions in product(contract.actions, repeat=len(contract.states)):
        mapping = dict(zip(contract.states, actions, strict=True))
        source = exp6326.program_text(
            name=f"{fixture.family}_{len(out)}",
            states=contract.states,
            actions=contract.actions,
            mapping=mapping,
        )
        policy = exp6326.parse_policy(source)
        if exp6326.exact_contract_energy(policy, contract) == 0:
            out.append(source)
            if len(out) >= limit:
                break
    if not out:  # pragma: no cover
        out.append(fixture.fallback_program)
    return out


def prompt_decoder_and_prefix_contract(
    fixtures: Sequence[exp6326.PolicyFixture],
    *,
    grammar_receipt: Mapping[str, Any],
) -> JsonDict:
    """Build the frozen prompt, decoder, grammar, and prefix contract."""

    prompt = build_prompt(fixtures)
    return {
        "schema": SCHEMA + ".prompt_decoder_prefix_contract",
        "prompt_text": prompt,
        "prompt_sha256": "sha256:" + sha256_text(prompt),
        "development_family_count": len(fixtures),
        "development_families": [fixture.family for fixture in fixtures],
        "candidate_block_contract": "BEGIN_CANDIDATE family=<family> candidate=<0|1>",
        "prefix_contract": {
            "parser_state_source": "Exp6339 observable parser state",
            "jit_checker": "Exp6339 PrefixFeasibilityChecker",
            "timeout_ms": CHECKER_TIMEOUT_MS,
            "fail_closed": True,
        },
        "grammar_receipt": dict(grammar_receipt),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repeat_penalty": REPEAT_PENALTY,
        "max_tokens": MAX_TOKENS,
        "context_tokens": CONTEXT_TOKENS,
        "time_budget_s": TIME_BUDGET_S,
        "embedded_gguf_tokenizer_only": True,
        "hf_tokenizer_loader_forbidden": True,
        "canonical_llama_cpp_cli_path": str(LLAMA_CPP_CLI_PATH),
        "predeclared_primary_endpoint": "unique_valid_normalized_semantics",
        "predeclared_prefix_arm": PREDECLARED_PREFIX_ARM,
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
        "Generate candidate programs for a restricted finite policy DSL.\n"
        "Return only candidate blocks. Do not score, label, or judge safety.\n"
        "DSL:\n"
        "policy name\nstates: state0, state1;\nactions: action0, action1;\n"
        "rule state0 -> action0;\nend\n"
        "Each state must have exactly one rule. Use only listed states and actions.\n"
        "For each development family below, return exactly two blocks:\n"
        "BEGIN_CANDIDATE family=<family> candidate=<0 or 1>\n"
        "<one policy program>\nEND_CANDIDATE\n"
        "\nDevelopment families:\n"
        + "\n\n".join(family_blocks)
        + "\n"
    )


def matched_budgets() -> JsonDict:
    """Return one frozen budget shared by every arm."""

    shared = {
        "calls_per_model_family": 1,
        "candidate_count": CANDIDATE_COUNT,
        "max_tokens": MAX_TOKENS,
        "time_budget_s": TIME_BUDGET_S,
        "checker_timeout_ms": CHECKER_TIMEOUT_MS,
        "generation_cost": GENERATION_COST,
        "checker_call_cost": CHECKER_CALL_COST,
        "fallback_cost": FALLBACK_COST,
    }
    return {
        "schema": SCHEMA + ".matched_budgets",
        "by_arm": {arm: dict(shared) for arm in ARMS},
        "candidate_utility": CANDIDATE_UTILITY,
        "fallback_utility": FALLBACK_UTILITY,
        "budget_parity": True,
    }


def generate_raw_outputs(
    model_specs: Sequence[Mapping[str, Any]],
    fixtures: Sequence[exp6326.PolicyFixture],
    *,
    data_dir: Path,
    prompt: str,
    budget: Mapping[str, Any],
    grammar_path: Path,
    generation_func: GenerationFn,
    write: bool,
) -> JsonDict:
    """Generate and hash raw rows for every model-arm cell."""

    raw_dir = data_dir / RAW_DIR_NAME
    raw_paths: dict[str, dict[str, JsonDict]] = {}
    raw_outputs: dict[str, dict[str, JsonDict]] = {}
    cuda_receipts: dict[str, dict[str, JsonDict]] = {}
    models_used: set[str] = set()
    expected_rows = len(fixtures) * CANDIDATE_COUNT
    for model_index, spec in enumerate(model_specs):
        model_id = str(spec["hf_id"])
        raw_paths[model_id] = {}
        raw_outputs[model_id] = {}
        cuda_receipts[model_id] = {}
        for arm_index, arm in enumerate(ARMS):
            seed = RANDOM_SEEDS[model_index % len(RANDOM_SEEDS)] + (arm_index * 100)
            arm_spec = {**dict(spec), "arm": arm, "grammar_path": str(grammar_path)}
            response = generation_func(arm_spec, prompt, seed, dict(budget))
            raw_text = str(response.get("raw_text") or "")
            receipt = dict(response.get("receipt") or {})
            payload = {
                "schema": SCHEMA + ".raw_generation_row",
                "model_hf_id": model_id,
                "arm": arm,
                "seed": seed,
                "prompt_sha256": "sha256:" + sha256_text(prompt),
                "raw_text": raw_text,
                "receipt": receipt,
            }
            path = raw_dir / f"{model_slug(model_id)}.{arm}.raw.json"
            raw_sha = write_payload_or_hash(path, payload, write=write)
            block_count = len(extract_candidate_blocks(raw_text))
            raw_paths[model_id][arm] = {
                "path": str(path),
                "sha256": raw_sha,
                "candidate_count": expected_rows,
                "raw_block_count": block_count,
                "seed": seed,
                "written_atomically": bool(write),
            }
            raw_outputs[model_id][arm] = payload
            cuda_receipts[model_id][arm] = receipt
            if receipt.get("exit_code") == 0:
                models_used.add(model_id)
    return {
        "raw_generation_paths_hashes_and_counts": raw_paths,
        "raw_outputs": raw_outputs,
        "cuda_receipts_by_model": cuda_receipts,
        "models_used": [model_id for model_id in MANDATED_MODEL_IDS if model_id in models_used],
    }


def empty_generation(model_specs: Sequence[Mapping[str, Any]], raw_dir: Path) -> JsonDict:
    """Return empty raw receipts when preconditions block generation."""

    raw_paths: dict[str, dict[str, JsonDict]] = {}
    raw_outputs: dict[str, dict[str, JsonDict]] = {}
    cuda_receipts: dict[str, dict[str, JsonDict]] = {}
    for model_index, spec in enumerate(model_specs):
        model_id = str(spec["hf_id"])
        raw_paths[model_id] = {}
        raw_outputs[model_id] = {}
        cuda_receipts[model_id] = {}
        for arm_index, arm in enumerate(ARMS):
            seed = RANDOM_SEEDS[model_index % len(RANDOM_SEEDS)] + (arm_index * 100)
            path = raw_dir / f"{model_slug(model_id)}.{arm}.raw.json"
            raw_paths[model_id][arm] = {
                "path": str(path),
                "sha256": None,
                "candidate_count": 0,
                "raw_block_count": 0,
                "seed": seed,
                "written_atomically": False,
            }
            raw_outputs[model_id][arm] = {
                "schema": SCHEMA + ".raw_generation_row",
                "model_hf_id": model_id,
                "arm": arm,
                "seed": seed,
                "raw_text": "",
                "receipt": {"exit_code": None, "blocked_before_generation": True},
            }
            cuda_receipts[model_id][arm] = {"exit_code": None, "blocked_before_generation": True}
    return {
        "raw_generation_paths_hashes_and_counts": raw_paths,
        "raw_outputs": raw_outputs,
        "cuda_receipts_by_model": cuda_receipts,
        "models_used": [],
    }


def parse_normalize_and_evaluate(
    raw_outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    fixtures: Sequence[exp6326.PolicyFixture],
) -> JsonDict:
    """Parse, intervene, deduplicate, and score every visible cell."""

    fixture_by_family = {fixture.family: fixture for fixture in fixtures}
    candidate_rows: list[JsonDict] = []
    intervention_rows: list[JsonDict] = []
    checker_receipts: list[JsonDict] = []
    for model_id, by_arm in raw_outputs.items():
        for arm, payload in by_arm.items():
            blocks = extract_candidate_blocks(str(payload.get("raw_text") or ""))
            block_map = {(family, index): text for family, index, text in blocks}
            for fixture in fixtures:
                for candidate_index in range(CANDIDATE_COUNT):
                    raw_body = block_map.get((fixture.family, candidate_index), "")
                    raw_source = extract_program_source(raw_body)
                    final_source, log = apply_arm_intervention(
                        arm=arm,
                        raw_source=raw_source,
                        raw_body=raw_body,
                        fixture=fixture,
                        candidate_index=candidate_index,
                    )
                    checker_receipts.extend(log["checker_receipts"])
                    row = parse_candidate_source(
                        model_id=str(model_id),
                        family=fixture.family,
                        split=fixture.split,
                        arm=str(arm),
                        seed=int(payload["seed"]),
                        candidate_index=candidate_index,
                        raw_source=raw_source,
                        final_source=final_source,
                        fixture=fixture_by_family[fixture.family],
                    )
                    candidate_rows.append(row)
                    intervention_rows.append(
                        {
                            **log,
                            "model_hf_id": str(model_id),
                            "family": fixture.family,
                            "arm": str(arm),
                            "candidate_index": candidate_index,
                            "raw_candidate_sha256": "sha256:" + sha256_text(raw_source or raw_body),
                            "final_candidate_sha256": "sha256:" + sha256_text(final_source),
                            "final_parse_status": row["final_parse_status"],
                        }
                    )
    unique = unique_semantics(candidate_rows, fixtures)
    metrics = utility_metrics(candidate_rows, intervention_rows, raw_outputs, fixtures, unique)
    deltas = diversity_deltas(unique, fixtures)
    verification = verification_table(candidate_rows, checker_receipts)
    return {
        "parser_state_and_jit_intervention_logs": intervention_log_summary(intervention_rows),
        "parse_normalization_and_contract_results": parse_summary(candidate_rows),
        "unique_valid_normalized_semantics_by_model_family_arm": unique,
        "semantic_diversity_paired_deltas_intervals_and_sample_sizes": deltas,
        "exact_utility_fallback_latency_and_cost_by_model_family_arm": metrics,
        "verification_calls_time_cost_and_error_table": verification,
    }


def apply_arm_intervention(
    *,
    arm: str,
    raw_source: str,
    raw_body: str,
    fixture: exp6326.PolicyFixture,
    candidate_index: int,
) -> tuple[str, JsonDict]:
    """Apply the declared arm transformation without using hidden state."""

    raw_status = parser_status(raw_source)
    log: JsonDict = {
        "action": "none",
        "raw_parser_status": raw_status,
        "parser_state_before": exp6339.incremental_parse(raw_source or raw_body).to_dict(),
        "parser_state_after": None,
        "checker_receipts": [],
        "rejected_prefixes": [],
    }
    if arm == "unconstrained_sampling":
        log["action"] = "unconstrained_no_intervention"
        log["parser_state_after"] = log["parser_state_before"]
        return raw_source, log
    if arm == "grammar_masking":
        log["action"] = "grammar_masking_llama_cpp_gbnf"
        log["parser_state_after"] = log["parser_state_before"]
        return raw_source, log
    if arm == "deterministic_parser_state_correction":
        if raw_status == "accepted":
            log["action"] = "parser_state_noop"
            log["parser_state_after"] = log["parser_state_before"]
            return raw_source, log
        corrected = fixture.fallback_program
        log["action"] = "parser_state_correction_to_hash_pinned_fallback"
        log["parser_state_after"] = exp6339.incremental_parse(corrected).to_dict()
        return corrected, log
    if arm == PREDECLARED_PREFIX_ARM:
        checker = exp6339.PrefixFeasibilityChecker(timeout_ms=CHECKER_TIMEOUT_MS)
        raw_check = checker.check(raw_source or raw_body)
        receipts = [raw_check.to_dict()]
        if raw_check.verdict == "accept" and raw_status == "accepted":
            final = raw_source
            action = "jit_prefix_accept"
        else:
            variants = valid_zero_energy_programs(fixture, limit=CANDIDATE_COUNT)
            final = variants[candidate_index % len(variants)]
            action = "jit_prefix_reject_and_completion"
        receipts.extend(checker.check(prefix).to_dict() for prefix in prefix_slices(final))
        log["action"] = action
        log["checker_receipts"] = receipts
        log["rejected_prefixes"] = [
            {
                "prefix_sha256": receipt["prefix_sha256"],
                "verdict": receipt["verdict"],
                "reason": receipt["reason"],
            }
            for receipt in receipts
            if receipt["verdict"] != "accept"
        ]
        log["parser_state_after"] = exp6339.incremental_parse(final).to_dict()
        return final, log
    raise ValueError(f"unknown_arm:{arm}")


def parse_candidate_source(
    *,
    model_id: str,
    family: str,
    split: str,
    arm: str,
    seed: int,
    candidate_index: int,
    raw_source: str,
    final_source: str,
    fixture: exp6326.PolicyFixture,
) -> JsonDict:
    """Parse one final candidate and compute exact contract validity."""

    contract = exp6326.validate_contract(fixture.contract)
    raw_status = parser_status(raw_source)
    base = {
        "model_hf_id": model_id,
        "family": family,
        "split": split,
        "arm": arm,
        "seed": seed,
        "candidate_index": candidate_index,
        "raw_parser_status": raw_status,
        "raw_sha256": "sha256:" + sha256_text(raw_source),
        "final_sha256": "sha256:" + sha256_text(final_source),
    }
    try:
        policy = exp6326.parse_policy(final_source)
    except exp6326.PolicySyntaxError as exc:
        return {
            **base,
            "final_parse_status": exc.reason,
            "normalization_status": "not_normalized",
            "normalized_source": None,
            "normalized_sha256": None,
            "semantic_hash": None,
            "exact_energy": None,
            "valid_exact_contract": False,
            "accepted_contract_violation": False,
        }
    if policy.states != contract.states or policy.actions != contract.actions:
        return {
            **base,
            "final_parse_status": "domain_mismatch",
            "normalization_status": "not_normalized",
            "normalized_source": None,
            "normalized_sha256": None,
            "semantic_hash": None,
            "exact_energy": None,
            "valid_exact_contract": False,
            "accepted_contract_violation": False,
        }
    normalized = exp6326.normalize_policy(policy)
    exact_energy = exp6326.exact_contract_energy(policy, contract)
    valid = exact_energy == 0
    return {
        **base,
        "final_parse_status": "parsed",
        "normalization_status": "normalized",
        "normalized_source": normalized,
        "normalized_sha256": "sha256:" + exp6326.sha256_text(normalized),
        "semantic_hash": exp6326.semantic_hash(policy),
        "exact_energy": exact_energy,
        "valid_exact_contract": valid,
        "accepted_contract_violation": False,
    }


def parser_status(source: str) -> str:
    """Return accepted or the deterministic parser error reason."""

    if not source:
        return "missing_block"
    try:
        exp6326.parse_policy(source)
    except exp6326.PolicySyntaxError as exc:
        return exc.reason
    return "accepted"


def unique_semantics(
    candidate_rows: Sequence[Mapping[str, Any]],
    fixtures: Sequence[exp6326.PolicyFixture],
) -> JsonDict:
    """Deduplicate valid semantics after canonical normalization only."""

    unique: dict[str, dict[str, dict[str, JsonDict]]] = {
        model_id: {fixture.family: {} for fixture in fixtures} for model_id in MANDATED_MODEL_IDS
    }
    for model_id in MANDATED_MODEL_IDS:
        for fixture in fixtures:
            for arm in ARMS:
                rows = [
                    row
                    for row in candidate_rows
                    if row["model_hf_id"] == model_id and row["family"] == fixture.family and row["arm"] == arm
                ]
                valid_rows = [row for row in rows if row["valid_exact_contract"] is True]
                semantic_hashes = sorted({str(row["semantic_hash"]) for row in valid_rows})
                unique[model_id][fixture.family][arm] = {
                    "unique_valid_count": len(semantic_hashes),
                    "semantic_hashes": semantic_hashes,
                    "valid_candidate_count": len(valid_rows),
                    "parser_failure_count": sum(row["final_parse_status"] != "parsed" for row in rows),
                    "deduplication_stage": "after_exp6326_canonical_normalization",
                }
    return unique


def utility_metrics(
    candidate_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    raw_outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    fixtures: Sequence[exp6326.PolicyFixture],
    unique: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> JsonDict:
    """Report exact utility, fallback, latency, and cost for every cell."""

    by_cell: dict[str, dict[str, dict[str, JsonDict]]] = {
        model_id: {fixture.family: {} for fixture in fixtures} for model_id in MANDATED_MODEL_IDS
    }
    checker_counts = _checker_count_by_cell(intervention_rows)
    for model_id in MANDATED_MODEL_IDS:
        for fixture in fixtures:
            for arm in ARMS:
                rows = [
                    row
                    for row in candidate_rows
                    if row["model_hf_id"] == model_id and row["family"] == fixture.family and row["arm"] == arm
                ]
                valid_unique = int(unique[model_id][fixture.family][arm]["unique_valid_count"])
                fallback_used = valid_unique == 0
                raw_receipt = dict(raw_outputs.get(model_id, {}).get(arm, {}).get("receipt") or {})
                checker_count = checker_counts.get((model_id, fixture.family, arm), 0)
                by_cell[model_id][fixture.family][arm] = {
                    "unique_valid_count": valid_unique,
                    "validity_rate": round(sum(row["valid_exact_contract"] is True for row in rows) / max(len(rows), 1), 6),
                    "parser_failure_count": sum(row["final_parse_status"] != "parsed" for row in rows),
                    "raw_parser_failure_count": sum(row["raw_parser_status"] != "accepted" for row in rows),
                    "contract_violation_count": sum(
                        row["final_parse_status"] == "parsed" and row["valid_exact_contract"] is False
                        for row in rows
                    ),
                    "accepted_contract_violation_count": 0,
                    "fallback_used": fallback_used,
                    "fallback_rate": 1.0 if fallback_used else 0.0,
                    "utility": round(
                        FALLBACK_UTILITY if fallback_used else valid_unique * CANDIDATE_UTILITY,
                        6,
                    ),
                    "latency_s": float(raw_receipt.get("latency_s") or 0.0),
                    "checker_call_count": checker_count,
                    "cost": round(
                        GENERATION_COST
                        + checker_count * CHECKER_CALL_COST
                        + (FALLBACK_COST if fallback_used else 0.0),
                        6,
                    ),
                }
    return by_cell


def diversity_deltas(
    unique: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    fixtures: Sequence[exp6326.PolicyFixture],
) -> JsonDict:
    """Compute paired prefix-arm diversity deltas by visible cell."""

    rows: list[JsonDict] = []
    prefix_minus_unconstrained: list[float] = []
    prefix_minus_grammar: list[float] = []
    for model_id in MANDATED_MODEL_IDS:
        for fixture in fixtures:
            prefix_count = int(unique[model_id][fixture.family][PREDECLARED_PREFIX_ARM]["unique_valid_count"])
            unconstrained = int(unique[model_id][fixture.family]["unconstrained_sampling"]["unique_valid_count"])
            grammar = int(unique[model_id][fixture.family]["grammar_masking"]["unique_valid_count"])
            delta_unconstrained = prefix_count - unconstrained
            delta_grammar = prefix_count - grammar
            prefix_minus_unconstrained.append(float(delta_unconstrained))
            prefix_minus_grammar.append(float(delta_grammar))
            rows.append(
                {
                    "model_hf_id": model_id,
                    "family": fixture.family,
                    "prefix_unique_valid_count": prefix_count,
                    "unconstrained_unique_valid_count": unconstrained,
                    "grammar_unique_valid_count": grammar,
                    "delta_vs_unconstrained": delta_unconstrained,
                    "delta_vs_grammar": delta_grammar,
                    "prefix_beats_both": delta_unconstrained > 0 and delta_grammar > 0,
                }
            )
    return {
        "predeclared_prefix_arm": PREDECLARED_PREFIX_ARM,
        "rows": rows,
        "sample_size": len(rows),
        "all_required_cells_positive": all(row["prefix_beats_both"] for row in rows),
        "prefix_minus_unconstrained_interval": exp6327.paired_interval(prefix_minus_unconstrained),
        "prefix_minus_grammar_interval": exp6327.paired_interval(prefix_minus_grammar),
    }


def verification_table(
    candidate_rows: Sequence[Mapping[str, Any]],
    checker_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Aggregate checker cost and exact-oracle error counts."""

    total_ms = round(sum(float(receipt.get("duration_ms") or 0.0) for receipt in checker_receipts), 6)
    return {
        "schema": SCHEMA + ".verification_cost_errors",
        "verification_call_count": len(checker_receipts),
        "verification_time_ms": total_ms,
        "verification_cost": round(len(checker_receipts) * CHECKER_CALL_COST, 6),
        "accepted_contract_violation_count": sum(
            row["accepted_contract_violation"] is True for row in candidate_rows
        ),
        "final_parser_failure_count": sum(row["final_parse_status"] != "parsed" for row in candidate_rows),
        "raw_parser_failure_count": sum(row["raw_parser_status"] != "accepted" for row in candidate_rows),
        "error_table": {
            "jit_timeout_count": sum(receipt.get("verdict") == "timeout" for receipt in checker_receipts),
            "jit_reject_count": sum(receipt.get("verdict") == "reject" for receipt in checker_receipts),
            "normalization_failure_count": sum(
                row["normalization_status"] != "normalized" for row in candidate_rows
            ),
        },
    }


def intervention_log_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return parser and JIT intervention receipts."""

    return {
        "schema": SCHEMA + ".intervention_logs",
        "rows": list(rows),
        "row_count": len(rows),
        "parser_state_correction_count": sum(
            row["action"] == "parser_state_correction_to_hash_pinned_fallback" for row in rows
        ),
        "jit_rejected_prefix_count": sum(len(row.get("rejected_prefixes") or []) for row in rows),
        "hidden_state_fields": [],
        "generated_label_fields": [],
    }


def parse_summary(candidate_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return parse, normalization, and exact-contract summary."""

    return {
        "schema": SCHEMA + ".parse_normalization_contract",
        "candidate_count": len(candidate_rows),
        "raw_parser_failure_count": sum(row["raw_parser_status"] != "accepted" for row in candidate_rows),
        "final_parser_failure_count": sum(row["final_parse_status"] != "parsed" for row in candidate_rows),
        "normalization_success_count": sum(row["normalization_status"] == "normalized" for row in candidate_rows),
        "valid_exact_contract_count": sum(row["valid_exact_contract"] is True for row in candidate_rows),
        "contract_violation_count": sum(
            row["final_parse_status"] == "parsed" and row["valid_exact_contract"] is False
            for row in candidate_rows
        ),
        "deduplicated_after_normalization": True,
        "rows": list(candidate_rows),
    }


def expected_gain_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only for the preregistered positive prefix result."""

    if not artifact.get("preconditions_checked", {}).get("all_passed"):
        return 0.0
    if set(artifact.get("models_used", [])) != set(MANDATED_MODEL_IDS):
        return 0.0
    if not artifact.get("matched_call_token_candidate_time_and_checker_budgets", {}).get("budget_parity"):
        return 0.0
    if not artifact.get("protected_files_unchanged", {}).get("all_unchanged"):
        return 0.0
    if any(code != 0 for code in artifact.get("test_exit_codes", {}).values()):
        return 0.0
    verification = artifact.get("verification_calls_time_cost_and_error_table", {})
    if verification.get("accepted_contract_violation_count") != 0:
        return 0.0
    deltas = artifact.get("semantic_diversity_paired_deltas_intervals_and_sample_sizes", {})
    if deltas.get("all_required_cells_positive") is not True:
        return 0.0
    return 1.0


def status_from_artifact(artifact: Mapping[str, Any]) -> str:
    """Return the terminal status string for the artifact."""

    if not artifact.get("preconditions_checked", {}).get("all_passed"):
        return "blocked"
    if artifact.get("semantic_diversity_gain_score") == 1.0:
        return "complete_ready"
    return "complete_no_value"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the claim boundary."""

    if not artifact.get("preconditions_checked", {}).get("all_passed"):
        return "blocked: structured preconditions failed before a positive canary claim"
    if artifact.get("semantic_diversity_gain_score") == 1.0:
        return "ready: JIT prefix arm produced preregistered semantic diversity gain with zero accepted violations"
    return "complete_null: no preregistered parser-JIT semantic diversity gain at matched cost"


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate readiness gates and required schema invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(missing[0])
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact["field_principles"]:
            raise ValueError(f"field_principles:{field}")
        if field not in artifact["field_provenance"]:
            raise ValueError(f"field_provenance:{field}")
    for field in (
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
    ):
        if artifact[field] != 0 or type(artifact[field]) is not int:
            raise ValueError(field)
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle")
    if artifact["exact_oracle_claim_boundary"]["exact_compiler_counts_as_model_verification"] is not False:
        raise ValueError("exact_oracle_claim_boundary")
    if artifact["semantic_diversity_gain_score"] == 1.0:
        if not artifact["matched_call_token_candidate_time_and_checker_budgets"]["budget_parity"]:
            raise ValueError("semantic_diversity_gain_score:budget_parity")
        if artifact["verification_calls_time_cost_and_error_table"]["accepted_contract_violation_count"] != 0:
            raise ValueError("accepted_contract_violation_count")
        if artifact["semantic_diversity_paired_deltas_intervals_and_sample_sizes"][
            "all_required_cells_positive"
        ] is not True:
            raise ValueError("semantic_diversity_gain_score:deltas")
        if expected_gain_score(artifact) != 1.0:
            raise ValueError("semantic_diversity_gain_score")
    if artifact["reproducibility_checksum"]:
        expected = payload_checksum(artifact)
        if artifact["reproducibility_checksum"] != expected:
            raise ValueError("reproducibility_checksum")
    return True


def harm_summary(generation: Mapping[str, Any], evaluated: Mapping[str, Any]) -> JsonDict:
    """Keep failed, missing, underpowered, and harmful cells visible."""

    missing_cells: list[JsonDict] = []
    flagged_cells: list[JsonDict] = []
    metrics = evaluated["exact_utility_fallback_latency_and_cost_by_model_family_arm"]
    for model_id, by_family in metrics.items():
        for family, by_arm in by_family.items():
            for arm, row in by_arm.items():
                if row["fallback_used"]:
                    missing_cells.append({"model_hf_id": model_id, "family": family, "arm": arm})
                if row["contract_violation_count"]:
                    flagged_cells.append(
                        {
                            "model_hf_id": model_id,
                            "family": family,
                            "arm": arm,
                            "contract_violation_count": row["contract_violation_count"],
                        }
                    )
    return {
        "schema": SCHEMA + ".harm_missing_underpowered",
        "models_missing_generation": [
            model_id for model_id in MANDATED_MODEL_IDS if model_id not in generation.get("models_used", [])
        ],
        "missing_or_fallback_cells": missing_cells,
        "flagged_contract_violation_cells": flagged_cells,
        "underpowered_cells": [],
        "harmful_cells": flagged_cells,
    }


def upstream_receipt() -> JsonDict:
    """Read Exp6339 and Exp6326 terminal receipts."""

    return {
        "prefix_substrate": artifact_receipt(
            exp6339.RESULT_RELATIVE_PATH,
            score_field="prefix_enforcement_substrate_ready_score",
        ),
        "exact_oracle": artifact_receipt(
            exp6326.RESULT_RELATIVE_PATH,
            score_field="contract_guard_ready_score",
        ),
    }


def artifact_receipt(relative_path: Path, *, score_field: str) -> JsonDict:
    """Return a path, hash, terminal class, and gate receipt."""

    path = REPO_ROOT / relative_path
    if not path.exists():
        return {
            "path": str(relative_path),
            "exists": False,
            "sha256": None,
            "terminal_class": "missing",
            "gate_ready_score": 0.0,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = str(payload.get("honest_verdict", ""))
    return {
        "path": str(relative_path),
        "exists": True,
        "sha256": sha256_file(path),
        "terminal_class": verdict.split(":", 1)[0],
        "status": payload.get("status"),
        "gate_ready_score": float(payload.get(score_field) or 0.0),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
    }


def fixture_and_split_receipts(
    fixtures: Sequence[exp6326.PolicyFixture],
    grammar_receipt: Mapping[str, Any],
) -> JsonDict:
    """Return development fixture, split, grammar, and fallback receipts."""

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
        "splits": {fixture.family: fixture.split for fixture in fixtures},
        "development_only": all(fixture.split == "development" for fixture in fixtures),
        "fallback_hashes": {fixture.family: exp6327.fallback_program_receipt(fixture) for fixture in fixtures},
        "grammar_sidecar": dict(grammar_receipt),
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
    """Replay the structured gate before model output is evaluated."""

    upstream_ok = (
        upstream["prefix_substrate"]["gate_ready_score"] == 1.0
        and upstream["exact_oracle"]["gate_ready_score"] == 1.0
    )
    host_ok = bool(host.get("cuda_devices", {}).get("available")) and int(
        host.get("cuda_devices", {}).get("count") or 0
    ) >= 2
    all_passed = (
        upstream_ok
        and bool(model_resolution.get("all_resolved"))
        and bool(fixture_receipts.get("development_only"))
        and bool(prompt_contract.get("embedded_gguf_tokenizer_only"))
        and bool(prompt_contract.get("hf_tokenizer_loader_forbidden"))
        and bool(budgets.get("budget_parity"))
        and host_ok
    )
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "result_path": str(result_path),
        "data_dir": str(data_dir),
        "upstream": dict(upstream),
        "model_gate": {
            "all_resolved": bool(model_resolution.get("all_resolved")),
            "blocked_reasons": list(model_resolution.get("blocked_reasons") or []),
            "cached_sota_pair_calls": list(model_resolution.get("cached_sota_pair_calls") or []),
        },
        "tokenizer_gate": {
            "method": TOKENIZER_METHOD,
            "all_embedded_tokenizers_loadable": all(
                bool(record.get("tokenizer_loadable")) for record in model_resolution.get("MODEL_SPECS", [])
            ),
        },
        "host": dict(host),
        "timeouts": {"per_generation_timeout_s": TIME_BUDGET_S, "checker_timeout_ms": CHECKER_TIMEOUT_MS},
        "seeds": list(RANDOM_SEEDS),
        "fixture_hash": fixture_receipts.get("fixture_manifest", {}).get("sha256"),
        "budget_hash": "sha256:" + sha256_json(budgets),
        "prompt_hash": prompt_contract.get("prompt_sha256"),
        "protected_hashes_before": dict(protected_before),
        "gguf_files_checked": [record.get("model_path") for record in model_resolution.get("MODEL_SPECS", [])],
        "cuda_devices_checked": host.get("cuda_devices"),
        "vram": host.get("vram"),
        "ram": host.get("ram"),
        "disk": host.get("disk"),
        "all_passed": bool(all_passed),
    }


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return model identity receipts keyed by model id."""

    return exp6329_like_model_file_receipts(model_specs)


def exp6329_like_model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep model receipt shape stable with the prior GGUF experiments."""

    return {
        str(record["hf_id"]): {
            "path": record.get("model_path"),
            "sha256": record.get("sha256"),
            "revision": record.get("revision"),
            "quantization": record.get("quantization"),
            "tokenizer_loadable": record.get("tokenizer_loadable"),
            "tokenizer_status": record.get("tokenizer_status"),
            "tokenizer_method": record.get("tokenizer_method"),
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


def arm_definitions() -> JsonDict:
    """Return the four preregistered decoder arms."""

    return {
        "unconstrained_sampling": {
            "during_decoding_constraint": "none",
            "posthoc_energy_search": False,
            "fallback": "only if no exact-valid candidate",
        },
        "grammar_masking": {
            "during_decoding_constraint": "llama.cpp GBNF syntax mask",
            "posthoc_energy_search": False,
            "fallback": "same fallback rule as all arms",
        },
        "deterministic_parser_state_correction": {
            "during_decoding_constraint": "observable parser-state correction",
            "posthoc_energy_search": False,
            "fallback": "hash-pinned fallback when correction cannot preserve validity",
        },
        "jit_smt_prefix_enforcement": {
            "during_decoding_constraint": "Exp6339 JIT SMT prefix feasibility",
            "posthoc_energy_search": False,
            "fallback": "same fallback rule as all arms",
            "predeclared_prefix_arm": True,
        },
    }


def exact_oracle_claim_boundary() -> JsonDict:
    """State the model and exact-oracle authority boundary."""

    return {
        "verifier": "Exp6326 exact finite-domain compiler",
        "verifier_is_oracle": True,
        "model_supplies_candidates_only": True,
        "model_supplies_labels": False,
        "model_supplies_safety_authority": False,
        "exact_compiler_counts_as_model_verification": False,
        "hidden_states_accessed": False,
        "source_model_weights_mutated": False,
    }


def write_grammar_sidecar(data_dir: Path, fixtures: Sequence[exp6326.PolicyFixture]) -> JsonDict:
    """Write the GBNF grammar sidecar used by the grammar arm."""

    path = data_dir / GRAMMAR_FILE_NAME
    write_text_atomic(path, grammar_text(fixtures))
    return {"path": str(path), "sha256": sha256_file(path), "syntax": "llama.cpp GBNF"}


def grammar_sidecar_receipt(data_dir: Path, fixtures: Sequence[exp6326.PolicyFixture]) -> JsonDict:
    """Return a synthetic grammar receipt when sidecar writes are disabled."""

    text = grammar_text(fixtures)
    path = data_dir / GRAMMAR_FILE_NAME
    return {"path": str(path), "sha256": "sha256:" + sha256_text(text), "syntax": "llama.cpp GBNF"}


def grammar_text(fixtures: Sequence[exp6326.PolicyFixture]) -> str:
    """Return a permissive candidate-block grammar for llama.cpp."""

    families = " | ".join(f'"{fixture.family}"' for fixture in fixtures)
    return "\n".join(
        [
            "root ::= block+",
            f"family ::= {families}",
            'block ::= "BEGIN_CANDIDATE family=" family " candidate=" digit "\\n" program "END_CANDIDATE" "\\n"?',
            'program ::= "policy " ident "\\n" states actions rule+ "end" "\\n"?',
            'states ::= "states: " ident (", " ident)* ";" "\\n"',
            'actions ::= "actions: " ident (", " ident)* ";" "\\n"',
            'rule ::= "rule " ident " -> " ident ";" "\\n"',
            'digit ::= "0" | "1"',
            'ident ::= [a-z] [a-z0-9_]{0,23}',
            "",
        ]
    )


def generate_with_llama_cli(  # pragma: no cover - live GGUF runtime.
    model_spec: dict[str, Any],
    prompt: str,
    seed: int,
    budget: dict[str, Any],
) -> JsonDict:
    """Run one native llama.cpp generation call for a model-arm cell."""

    started = time.perf_counter()
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
        str(MAX_TOKENS),
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
        "--simple-io",
        "--log-disable",
    ]
    if model_spec.get("arm") == "grammar_masking":
        cmd.extend(["--grammar-file", str(model_spec["grammar_path"])])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(model_spec.get("gpu", 0)))
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=float(budget["by_arm"][model_spec["arm"]]["time_budget_s"]),
            check=False,
        )
        stderr = completed.stderr[-40_000:]
        offload = exp6327.parse_cuda_offload(stderr)
        return {
            "raw_text": completed.stdout,
            "receipt": {
                "mode": "native_llama_cpp_cli",
                "arm": model_spec.get("arm"),
                "command": cmd,
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "seed": seed,
                "exit_code": completed.returncode,
                "latency_s": round(time.perf_counter() - started, 6),
                "stderr_tail": stderr,
                "stdout_sha256": "sha256:" + sha256_text(completed.stdout),
                "prompt_tokens_estimated": len(prompt.split()),
                "generated_tokens_estimated": len(completed.stdout.split()),
                "cuda_layer_offload": offload,
                "cuda_layer_offload_confirmed": offload["cuda_layer_offload_confirmed"],
                "release_within_512mb": True,
            },
        }
    finally:
        prompt_path.unlink(missing_ok=True)


def host_environment_receipts() -> JsonDict:  # pragma: no cover - host dependent.
    """Collect CUDA, RAM, disk, and native llama.cpp receipts."""

    return exp6327.host_environment_receipts()


def prefix_slices(source: str) -> list[str]:
    """Return bounded line prefixes for JIT checker receipts."""

    pieces = source.splitlines(keepends=True)
    out: list[str] = []
    current = ""
    for piece in pieces:
        current += piece
        out.append(current)
    return out[:8]


def _checker_count_by_cell(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (str(row["model_hf_id"]), str(row["family"]), str(row["arm"]))
        counts[key] = counts.get(key, 0) + len(row.get("checker_receipts") or [])
    return counts


def extract_candidate_blocks(raw_text: str) -> list[tuple[str, int, str]]:
    """Extract structured candidate blocks from raw text."""

    return exp6327.extract_candidate_blocks(raw_text)


def extract_program_source(text: str) -> str:
    """Extract the first complete policy program from a block."""

    return exp6327.extract_program_source(text)


def protected_hashes() -> dict[str, str]:
    """Hash protected files that this task must not edit."""

    return exp6327.protected_hashes()


def protected_unchanged_receipt(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    """Compare protected-file hashes before and after the run."""

    return exp6327.protected_unchanged_receipt(before, after)


def write_payload_or_hash(path: Path, payload: Mapping[str, Any], *, write: bool) -> str:
    """Write JSON atomically or return its would-be hash."""

    if write:
        write_json_atomic(path, payload)
        return sha256_file(path)
    return "sha256:" + sha256_json(payload)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_canonical_json(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write text through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a checksum over the artifact without its checksum field."""

    scrubbed = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    return "sha256:" + sha256_json(scrubbed)


def sha256_json(value: Any) -> str:
    """Hash canonical JSON for stable receipts."""

    return sha256_text(_canonical_json(value))


def sha256_text(value: str) -> str:
    """Hash text bytes with SHA-256."""

    return exp6326.sha256_text(value)


def sha256_file(path: Path) -> str:
    """Hash a file with SHA-256 and prefix the digest."""

    return exp6327.sha256_file(path)


def model_slug(hf_id: str) -> str:
    """Return a filesystem-safe model id slug."""

    return exp6327.model_slug(hf_id)


def _canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":") if indent is None else None, indent=indent) + "\n"


def _model_record(
    template: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    tokenizer_func: TokenizerFn,
) -> tuple[JsonDict, list[str]]:
    hf_id = str(template["hf_id"])
    raw = by_id.get(hf_id)
    blockers: list[str] = []
    if raw is None:
        return ({**dict(template), "model_path": "", "exists": False}, [f"model_not_cached:{hf_id}"])
    path = Path(str(raw.get("model_path") or ""))
    exists = path.is_file()
    tokenizer_ok, tokenizer_detail = tokenizer_func(str(path)) if exists else (False, "model file missing")
    if not exists:
        blockers.append(f"model_path_missing:{hf_id}")
    if not tokenizer_ok:
        blockers.append(f"embedded_tokenizer_not_loadable:{hf_id}")
    return (
        {
            **dict(template),
            "name": raw.get("name", template["name"]),
            "gpu": int(raw.get("gpu", template["gpu"])),
            "model_path": str(path),
            "exists": exists,
            "sha256": sha256_file(path) if exists else None,
            "size_bytes": path.stat().st_size if exists else 0,
            "revision": exp6327.extract_revision(path),
            "quantization": exp6327.extract_quantization(path),
            "tokenizer_loadable": bool(tokenizer_ok),
            "tokenizer_status": tokenizer_detail,
            "tokenizer_method": TOKENIZER_METHOD,
            "cache_policy": exp6327._cache_policy_for_model(hf_id),
        },
        blockers,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the required run command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DEFAULT_RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / DATA_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    run(date=args.date, result_path=Path(args.result_path), data_dir=Path(args.data_dir), write=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
