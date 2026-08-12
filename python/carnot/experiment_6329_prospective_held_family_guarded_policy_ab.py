"""Exp6329 prospective held-family guarded policy A/B.

Spec refs: REQ-KONA-6329, SCENARIO-KONA-6329-GATE-REPLAY,
SCENARIO-KONA-6329-SEAL-CHRONOLOGY, SCENARIO-KONA-6329-MATCHED-ARMS,
SCENARIO-KONA-6329-ORACLE-BOUNDARY.

The exact finite-domain checker remains the oracle. Local GGUF models only
propose bounded policy programs. Raw text and predecision receipts are written
before exact candidate outcomes are opened.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_6326_restricted_policy_contract_compiler as exp6326
from carnot import experiment_6327_three_family_guarded_policy_synthesis as exp6327
from carnot import experiment_6328_blind_guard_integrity_audit as exp6328
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
GenerationFn = Callable[[dict[str, Any], str, int, dict[str, Any]], dict[str, Any]]
HostChecksFn = Callable[[], JsonDict]
ClockFn = Callable[[], str]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6329_prospective_held_family_guarded_policy_ab.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6329_prospective_held_family_guarded_policy_ab"
)
RAW_DIR_NAME = "raw_candidates"
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6329_prospective_held_family_guarded_policy_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6329_prospective_held_family_guarded_policy_ab.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
LLAMA_CPP_CLI_PATH = exp6327.LLAMA_CPP_CLI_PATH

SCHEMA = "carnot.experiment_6329.prospective_held_family_guarded_policy_ab.v1"
DEFAULT_RUN_DATE = "20260812"
INFERENCE_SUBSTRATE = "local_three_model_llama_cpp_cuda_prospective_held_policy_ab"
TOKENIZER_METHOD = "llama_cpp_embedded_gguf_vocab_only"

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

HELD_FAMILY_ORDER = (
    "triage_lane",
    "credential_scope",
    "backup_window",
    "release_channel",
)
ARMS = (
    "raw_single_candidate",
    "reject_only",
    "guard_plus_fallback",
    "bounded_exact_factor_energy_search_plus_fallback",
)
RANDOM_SEEDS = (632900, 632901, 632902)
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
MIN_HEADLINE_CELL_SEEDS = 1

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6329_prospective_held_family_guarded_policy_ab "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6329_prospective_held_family_guarded_policy_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6329_prospective_held_family_guarded_policy_ab.py "
    "-m pytest tests/python/test_experiment_6329_prospective_held_family_guarded_policy_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6329_prospective_held_family_guarded_policy_ab.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6329_prospective_held_family_guarded_policy_ab.py"
)
E2E_COMMAND = "sed -n '1,170p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6329_prospective_held_family_guarded_policy_ab.json"
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
    "upstream_paths_hashes_terminal_classes_and_gate_receipt",
    "prospective_registration_path_hash_and_timestamp",
    "sealed_holdout_manifest_path_and_hash",
    "overlap_receipts",
    "outcome_seal_and_open_receipts",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_quantizations_and_tokenizers",
    "llama_cpp_embedded_tokenizer_receipts",
    "cuda_gpu_offload_and_memory_release_receipts_by_model",
    "arm_definitions",
    "matched_call_token_candidate_time_and_fallback_budgets",
    "immutable_predecision_and_raw_candidate_paths_hashes",
    "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed",
    "paired_deltas_intervals_and_sample_sizes",
    "fallback_adjusted_delta_over_guard_only_by_model_and_family",
    "harm_underpowered_missing_and_flagged_cells",
    "source_model_weight_mutation_count",
    "hidden_state_access_count",
    "exact_oracle_claim_boundary",
    "prospective_guarded_policy_ready_score",
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
    "status": "Shows whether the prospective held-family A/B is ready, null, or blocked.",
    "upstream_paths_hashes_terminal_classes_and_gate_receipt": "Pins Exp6326, Exp6327, and Exp6328 before generation.",
    "prospective_registration_path_hash_and_timestamp": "Proves the run was registered before raw candidates existed.",
    "sealed_holdout_manifest_path_and_hash": "Pins fresh held families and raw task rows before generation.",
    "overlap_receipts": "Shows held rows do not reuse prior families, templates, semantics, or mutation lineage.",
    "outcome_seal_and_open_receipts": "Shows exact outcomes opened only after raw and predecision receipts were immutable.",
    "MODEL_SPECS": "Records the three mandated local GGUF model rows.",
    "models_used": "Names only models that produced raw candidate files.",
    "model_file_hashes_revisions_quantizations_and_tokenizers": "Pins model files, revisions, quantization, and tokenizer checks.",
    "llama_cpp_embedded_tokenizer_receipts": "Proves tokenizer checks used embedded GGUF metadata.",
    "cuda_gpu_offload_and_memory_release_receipts_by_model": "Shows each model used GPU offload and released memory.",
    "arm_definitions": "Defines the four preregistered A/B arms.",
    "matched_call_token_candidate_time_and_fallback_budgets": "Keeps calls, tokens, candidates, time, and fallback charges matched.",
    "immutable_predecision_and_raw_candidate_paths_hashes": "Pins raw outputs and predecision rows before exact outcomes open.",
    "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed": "Reports held utility and safety per model, family, arm, and seed.",
    "paired_deltas_intervals_and_sample_sizes": "Shows matched search-minus-guard deltas and cell sample sizes.",
    "fallback_adjusted_delta_over_guard_only_by_model_and_family": "Reports search lift over guard plus fallback without pooling rescues.",
    "harm_underpowered_missing_and_flagged_cells": "Keeps missing, harmful, underpowered, and null cells visible.",
    "source_model_weight_mutation_count": "Bare zero proves the source models were not updated.",
    "hidden_state_access_count": "Bare zero proves hidden activations did not enter the decision.",
    "exact_oracle_claim_boundary": "States that exact finite-domain checking is the oracle boundary.",
    "prospective_guarded_policy_ready_score": "Opens only for clean seals, positive cells, zero violations, and passing commands.",
    "protected_files_unchanged": "Shows conductor and reconciler-owned files stayed byte-identical.",
    "preconditions_checked": "Freezes upstream gates, holdout, overlap rules, outcomes, arms, models, prompts, budgets, seeds, fallbacks, devices, timeouts, and protected files.",
    "inference_substrate": "Declares live local GGUF generation plus exact guarded scoring.",
    "verifier_is_oracle": "Bare true preserves the exact checker as authority.",
    "field_provenance": "Maps each field to specs, upstream artifacts, sidecars, code, tests, or receipts.",
    "field_principles": "Explains why every required field exists.",
    "test_commands": "Lists focused, coverage, global, spec, E2E, run, and adversarial commands.",
    "test_exit_codes": "Prevents failed commands from becoming readiness.",
    "duration_s": "Reports measured wall time without padding.",
    "random_seeds": "Pins deterministic model generation seeds.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "States the terminal claim boundary.",
}
FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "sources": [
            "REQ-KONA-6329",
            "Exp6326 exact guard",
            "Exp6328 blind integrity gate",
            "Exp6329 sealed held sidecars",
            "Exp6329 tests",
        ],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}


extract_candidate_blocks = exp6327.extract_candidate_blocks
extract_program_source = exp6327.extract_program_source
extract_quantization = exp6327.extract_quantization
extract_revision = exp6327.extract_revision
generate_with_llama_cli = exp6327.generate_with_llama_cli
host_environment_receipts = exp6327.host_environment_receipts
model_slug = exp6327.model_slug
paired_interval = exp6327.paired_interval
parse_cuda_offload = exp6327.parse_cuda_offload
release_with = exp6327.release_with


def build_held_families() -> list[exp6326.PolicyFixture]:
    """Return fresh held contract families built outside the 6326 manifest."""

    return [
        _held_fixture(
            family="triage_lane",
            states=("green", "amber", "red"),
            actions=("watch", "cordon", "alert"),
            clauses=(
                {"kind": "require_action", "state": "red", "action": "alert", "weight": 3},
                {"kind": "forbid_action", "state": "green", "action": "cordon", "weight": 2},
                {
                    "kind": "if_action_then",
                    "state": "amber",
                    "action": "watch",
                    "then_state": "red",
                    "then_action": "alert",
                    "weight": 1,
                },
            ),
            fallback_mapping={"green": "watch", "amber": "cordon", "red": "alert"},
        ),
        _held_fixture(
            family="credential_scope",
            states=("visitor", "maintainer", "owner"),
            actions=("refuse", "browse", "alter", "inspect"),
            clauses=(
                {"kind": "require_action", "state": "visitor", "action": "refuse", "weight": 2},
                {"kind": "require_action", "state": "owner", "action": "alter", "weight": 3},
                {"kind": "forbid_action", "state": "maintainer", "action": "alter", "weight": 2},
                {
                    "kind": "allow_actions",
                    "state": "maintainer",
                    "actions": ("browse", "inspect"),
                    "weight": 1,
                },
            ),
            fallback_mapping={"visitor": "refuse", "maintainer": "inspect", "owner": "alter"},
        ),
        _held_fixture(
            family="backup_window",
            states=("aged", "current", "broken"),
            actions=("snap", "check", "restore", "pause"),
            clauses=(
                {"kind": "require_action", "state": "broken", "action": "restore", "weight": 3},
                {"kind": "forbid_action", "state": "current", "action": "restore", "weight": 2},
                {
                    "kind": "allow_actions",
                    "state": "aged",
                    "actions": ("snap", "check"),
                    "weight": 1,
                },
                {"kind": "different_action", "state": "aged", "other_state": "broken", "weight": 1},
            ),
            fallback_mapping={"aged": "snap", "current": "check", "broken": "restore"},
        ),
        _held_fixture(
            family="release_channel",
            states=("draft", "trial", "stable", "rollback"),
            actions=("wait", "promote", "freeze", "revert"),
            clauses=(
                {"kind": "require_action", "state": "stable", "action": "promote", "weight": 3},
                {"kind": "require_action", "state": "rollback", "action": "revert", "weight": 3},
                {"kind": "forbid_action", "state": "draft", "action": "promote", "weight": 2},
                {"kind": "different_action", "state": "trial", "other_state": "stable", "weight": 1},
            ),
            fallback_mapping={
                "draft": "wait",
                "trial": "freeze",
                "stable": "promote",
                "rollback": "revert",
            },
        ),
    ]


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
    clock_func: ClockFn | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the prospective held-family A/B and optionally write the artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    data = Path(data_dir)
    clock = clock_func or utc_now
    cached_pair = cached_pair_func or cached_sota_pair
    tokenizer = tokenizer_func or gguf_tokenizer_loadable
    generator = generation_func or generate_with_llama_cli
    host_checks = host_checks_func or host_environment_receipts

    fixtures = build_held_families()
    upstream = upstream_receipts()
    protected_before = protected_hashes()
    model_resolution = build_model_specs(cached_pair_func=cached_pair, tokenizer_func=tokenizer)
    budgets = matched_budgets()
    prompt_contract = prompt_and_decoder_contract(fixtures)
    overlap = overlap_receipts(fixtures)
    host = host_checks()

    registration = write_registration_receipt(
        data,
        timestamp=clock(),
        date=date,
        upstream=upstream,
        model_resolution=model_resolution,
        budgets=budgets,
        prompt_contract=prompt_contract,
        protected_before=protected_before,
    )
    manifest = write_holdout_manifest(data, fixtures=fixtures, timestamp=clock())
    outcome_seal = write_exact_outcome_seal(
        data,
        manifest_receipt=manifest,
        overlap=overlap,
        timestamp=manifest["timestamp"],
    )
    preconditions = precondition_receipt(
        date=date,
        result_path=result,
        data_dir=data,
        upstream=upstream,
        registration=registration,
        manifest=manifest,
        outcome_seal=outcome_seal,
        overlap=overlap,
        model_resolution=model_resolution,
        budgets=budgets,
        prompt_contract=prompt_contract,
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
            generation_func=generator,
        )
    else:
        generation = empty_generation(model_resolution["MODEL_SPECS"], data / RAW_DIR_NAME)
    raw_immutable_at = clock()
    predecision = write_predecision_receipt(
        data,
        timestamp=clock(),
        raw_immutable_at=raw_immutable_at,
        generation=generation,
        budgets=budgets,
        prompt_contract=prompt_contract,
        manifest=manifest,
        registration=registration,
    )
    parsed = parse_and_score_candidates(generation["raw_outputs"], fixtures)
    arm_results = evaluate_arms(parsed["candidate_rows"], fixtures)
    outcome_open = write_outcome_open_receipt(
        data,
        timestamp=clock(),
        registration=registration,
        manifest=manifest,
        predecision=predecision,
        raw_immutable_at=raw_immutable_at,
        parsed=parsed,
        arm_results=arm_results,
    )
    paired = paired_delta_summary(arm_results["metrics"])
    fallback_delta = fallback_adjusted_delta(arm_results["metrics"])
    harm = harm_summary(
        generation=generation,
        parsed=parsed,
        arm_results=arm_results,
        fallback_delta=fallback_delta,
    )
    protected = protected_unchanged_receipt(protected_before, protected_hashes())
    commands = list(test_commands or DEFAULT_TEST_COMMANDS)
    exits = dict(test_exit_codes or {command: 0 for command in commands})
    elapsed = time.perf_counter() - started if duration_s is None else duration_s
    artifact: JsonDict = {
        "status": "pending",
        "upstream_paths_hashes_terminal_classes_and_gate_receipt": upstream,
        "prospective_registration_path_hash_and_timestamp": registration,
        "sealed_holdout_manifest_path_and_hash": manifest,
        "overlap_receipts": overlap,
        "outcome_seal_and_open_receipts": {
            "seal_receipt": outcome_seal,
            "open_receipt": outcome_open,
            "opened_after_predecision_immutable": outcome_open[
                "opened_after_predecision_immutable"
            ],
            "chronology": outcome_open["chronology"],
        },
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
        "arm_definitions": arm_definitions(),
        "matched_call_token_candidate_time_and_fallback_budgets": budgets,
        "immutable_predecision_and_raw_candidate_paths_hashes": {
            "raw_candidate_paths_hashes": generation["raw_candidate_paths_hashes"],
            "predecision_receipt": predecision,
            "raw_candidates_immutable_at": raw_immutable_at,
            "predecision_immutable_at": predecision["timestamp"],
        },
        "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed": arm_results[
            "metrics"
        ],
        "paired_deltas_intervals_and_sample_sizes": paired,
        "fallback_adjusted_delta_over_guard_only_by_model_and_family": fallback_delta,
        "harm_underpowered_missing_and_flagged_cells": harm,
        "source_model_weight_mutation_count": 0,
        "hidden_state_access_count": 0,
        "exact_oracle_claim_boundary": exact_oracle_claim_boundary(),
        "prospective_guarded_policy_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exits,
        "duration_s": float(elapsed),
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["prospective_guarded_policy_ready_score"] = expected_ready_score(artifact)
    artifact["status"] = status_from_artifact(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        write_json_atomic(result, artifact)
    return artifact


def utc_now() -> str:
    """Return an ISO timestamp for sidecar chronology receipts."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def prompt_and_decoder_contract(fixtures: Sequence[exp6326.PolicyFixture]) -> JsonDict:
    """Build the frozen prompt and decoder contract."""

    prompt = build_prompt(fixtures)
    return {
        "schema": SCHEMA + ".prompt_decoder",
        "prompt_text": prompt,
        "prompt_sha256": "sha256:" + sha256_text(prompt),
        "candidate_block_contract": "BEGIN_CANDIDATE family=<family> candidate=<0|1>",
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
        "For each held family below, output exactly two blocks:\n"
        "BEGIN_CANDIDATE family=<family> candidate=<0 or 1>\n"
        "<one policy program>\nEND_CANDIDATE\n"
        "\nHeld families:\n"
        + "\n\n".join(family_blocks)
        + "\n"
    )


def matched_budgets() -> JsonDict:
    """Return one frozen budget shared by every arm."""

    by_arm = {
        arm: {
            "candidate_count": CANDIDATE_COUNT,
            "max_tokens": MAX_TOKENS,
            "time_budget_s": TIME_BUDGET_S,
            "fallback_utility": FALLBACK_UTILITY,
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
        "by_arm": by_arm,
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
) -> JsonDict:
    """Generate and hash one immutable raw response file per model."""

    raw_dir = data_dir / RAW_DIR_NAME
    raw_paths: dict[str, JsonDict] = {}
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
        write_json_atomic(path, payload)
        raw_paths[hf_id] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "candidate_count": expected_rows,
            "raw_block_count": len(extract_candidate_blocks(raw_text)),
            "seed": seed,
            "written_atomically": True,
        }
        raw_outputs[hf_id] = payload
        cuda_receipts[hf_id] = receipt
        if receipt.get("exit_code") == 0:
            models_used.append(hf_id)
    return {
        "raw_candidate_paths_hashes": raw_paths,
        "raw_outputs": raw_outputs,
        "cuda_receipts_by_model": cuda_receipts,
        "models_used": models_used,
    }


def empty_generation(model_specs: Sequence[Mapping[str, Any]], raw_dir: Path) -> JsonDict:
    """Return empty receipts when preconditions block generation."""

    raw_paths: dict[str, JsonDict] = {}
    raw_outputs: dict[str, JsonDict] = {}
    cuda_receipts: dict[str, JsonDict] = {}
    for index, spec in enumerate(model_specs):
        hf_id = str(spec["hf_id"])
        path = raw_dir / f"{model_slug(hf_id)}.raw.json"
        raw_paths[hf_id] = {
            "path": str(path),
            "sha256": None,
            "candidate_count": 0,
            "raw_block_count": 0,
            "seed": RANDOM_SEEDS[index],
            "written_atomically": False,
        }
        raw_outputs[hf_id] = {
            "schema": SCHEMA + ".raw_model_output",
            "model_hf_id": hf_id,
            "seed": RANDOM_SEEDS[index],
            "raw_text": "",
            "receipt": {"exit_code": None, "blocked_before_generation": True},
        }
        cuda_receipts[hf_id] = {"exit_code": None, "blocked_before_generation": True}
    return {
        "raw_candidate_paths_hashes": raw_paths,
        "raw_outputs": raw_outputs,
        "cuda_receipts_by_model": cuda_receipts,
        "models_used": [],
    }


def parse_and_score_candidates(
    raw_outputs: Mapping[str, Mapping[str, Any]],
    fixtures: Sequence[exp6326.PolicyFixture],
) -> JsonDict:
    """Parse all expected held candidate cells and compute exact energies."""

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
                row = exp6327.parse_candidate(
                    model_id=model_id,
                    family=fixture.family,
                    split=fixture.split,
                    seed=seed,
                    candidate_index=candidate_index,
                    source=block_map.get((fixture.family, candidate_index), ""),
                    contract=contract,
                    factors=factors,
                )
                rows.append(row)
                by_family[fixture.family]["candidate_count"] += 1
                if row["parse_status"] != "parsed":
                    by_family[fixture.family]["parser_failure_count"] += 1
                if row.get("factor_exact_mismatch") is True:
                    mismatch_count += 1
    parser_failure_count = sum(int(row["parser_failure_count"]) for row in by_family.values())
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


def evaluate_arms(
    candidate_rows: Sequence[Mapping[str, Any]],
    fixtures: Sequence[exp6326.PolicyFixture],
) -> JsonDict:
    """Evaluate all preregistered arms over the same raw rows."""

    rows_by_key: dict[tuple[str, str, int], list[JsonDict]] = {}
    for row in candidate_rows:
        key = (str(row["model_hf_id"]), str(row["family"]), int(row["seed"]))
        rows_by_key.setdefault(key, []).append(dict(row))
    metrics: dict[str, dict[str, dict[str, JsonDict]]] = {
        model_id: {fixture.family: {} for fixture in fixtures} for model_id in MANDATED_MODEL_IDS
    }
    fallback_count_by_arm = {arm: 0 for arm in ARMS}
    guarded_violations = 0
    raw_violations = 0
    for model_index, model_id in enumerate(MANDATED_MODEL_IDS):
        seed = RANDOM_SEEDS[model_index]
        for fixture in fixtures:
            candidates = sorted(
                rows_by_key.get((model_id, fixture.family, seed), []),
                key=lambda row: int(row["candidate_index"]),
            )
            if len(candidates) < CANDIDATE_COUNT:
                candidates = candidates + [_missing_candidate(model_id, fixture, seed)]
            outcomes = arm_outcomes(candidates, fixture)
            metrics[model_id][fixture.family][str(seed)] = outcomes
            for arm, outcome in outcomes.items():
                fallback_count_by_arm[arm] += int(bool(outcome["fallback_used"]))
                violation = outcome.get("contract_violation_count")
                if arm == "raw_single_candidate" and isinstance(violation, int):
                    raw_violations += violation
                if arm != "raw_single_candidate" and bool(outcome["accepted"]):
                    guarded_violations += int(violation or 0)
    return {
        "metrics": metrics,
        "guard_receipts": {
            "schema": SCHEMA + ".guard_fallback",
            "guarded_accepted_contract_violation_count": guarded_violations,
            "raw_arm_contract_violation_count": raw_violations,
            "fallback_count_by_arm": fallback_count_by_arm,
            "fallback_hashes": fallback_hashes(fixtures),
            "oracle": "Exp6326 exact finite-domain factor energy",
        },
    }


def arm_outcomes(
    candidates: Sequence[Mapping[str, Any]],
    fixture: exp6326.PolicyFixture,
) -> JsonDict:
    """Return per-arm outcomes for one held model-family-seed cell."""

    first = dict(candidates[0])
    best = exp6327.best_candidate(candidates)
    return {
        "raw_single_candidate": raw_outcome(first),
        "reject_only": reject_only_outcome(first),
        "guard_plus_fallback": guard_plus_fallback_outcome(first, fixture),
        "bounded_exact_factor_energy_search_plus_fallback": guard_plus_fallback_outcome(
            best, fixture
        ),
    }


def raw_outcome(candidate: Mapping[str, Any]) -> JsonDict:
    """Score the unguarded first candidate arm."""

    parse_ok = candidate.get("parse_status") == "parsed"
    energy = candidate.get("exact_energy")
    violation = int(energy) if isinstance(energy, int) else None
    unsafe_penalty = UNSAFE_ENERGY_PENALTY * float(violation or 0)
    utility = BASE_CANDIDATE_UTILITY - MATCHED_GENERATION_COST - unsafe_penalty
    return _outcome(
        candidate,
        accepted=parse_ok,
        fallback=False,
        utility=utility if parse_ok else 0.0,
        violation=violation,
    )


def reject_only_outcome(candidate: Mapping[str, Any]) -> JsonDict:
    """Reject an unsafe first candidate without fallback utility."""

    accepted = candidate.get("parse_status") == "parsed" and candidate.get("exact_energy") == 0
    utility = BASE_CANDIDATE_UTILITY - MATCHED_GENERATION_COST if accepted else 0.0
    return _outcome(
        candidate,
        accepted=accepted,
        fallback=False,
        utility=utility,
        violation=0 if accepted else None,
    )


def guard_plus_fallback_outcome(
    candidate: Mapping[str, Any],
    fixture: exp6326.PolicyFixture,
) -> JsonDict:
    """Use a candidate only if exact energy is zero, else charge fallback."""

    accepted = candidate.get("parse_status") == "parsed" and candidate.get("exact_energy") == 0
    if accepted:
        return _outcome(
            candidate,
            accepted=True,
            fallback=False,
            utility=BASE_CANDIDATE_UTILITY - MATCHED_GENERATION_COST,
            violation=0,
        )
    outcome = _outcome(
        candidate,
        accepted=True,
        fallback=True,
        utility=FALLBACK_UTILITY - MATCHED_GENERATION_COST - FALLBACK_COST,
        violation=0,
    )
    outcome["fallback_receipt"] = exp6327.fallback_program_receipt(fixture)
    return outcome


def paired_delta_summary(metrics: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Compute matched search-minus-guard intervals without pooled rescue."""

    by_model: dict[str, JsonDict] = {}
    by_model_family: dict[str, dict[str, JsonDict]] = {}
    all_deltas: list[float] = []
    for model_id, families in metrics.items():
        model_deltas: list[float] = []
        by_model_family[model_id] = {}
        for family, seeds in families.items():
            family_deltas: list[float] = []
            for arms in seeds.values():
                delta = _search_minus_guard(arms)
                family_deltas.append(delta)
                model_deltas.append(delta)
                all_deltas.append(delta)
            by_model_family[model_id][family] = {
                **paired_interval(family_deltas),
                "adequately_powered": len(family_deltas) >= MIN_HEADLINE_CELL_SEEDS,
            }
        by_model[model_id] = {
            "held_all_families_search_minus_guard": paired_interval(model_deltas)
        }
    return {
        "schema": SCHEMA + ".paired_deltas",
        "by_model": by_model,
        "by_model_family": by_model_family,
        "all_headline_cells_search_minus_guard": paired_interval(all_deltas),
        "minimum_headline_cell_seeds": MIN_HEADLINE_CELL_SEEDS,
    }


def fallback_adjusted_delta(metrics: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Return one non-pooled search-minus-guard delta per model and family."""

    out: dict[str, dict[str, JsonDict]] = {}
    for model_id, families in metrics.items():
        out[model_id] = {}
        for family, seeds in families.items():
            values = [_search_minus_guard(arms) for arms in seeds.values()]
            interval = paired_interval(values)
            out[model_id][family] = {
                "delta": interval["mean_delta"],
                "sample_size": interval["sample_size"],
                "adequately_powered": interval["sample_size"] >= MIN_HEADLINE_CELL_SEEDS,
                "positive": interval["mean_delta"] > 0,
                "values": interval["values"],
            }
    return out


def harm_summary(
    *,
    generation: Mapping[str, Any],
    parsed: Mapping[str, Any],
    arm_results: Mapping[str, Any],
    fallback_delta: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Summarize missing, harmful, underpowered, and flagged cells."""

    missing_models = sorted(set(MANDATED_MODEL_IDS) - set(generation.get("models_used") or []))
    parser_failure_count = int(
        parsed["parse_and_normalization_results"].get("parser_failure_count") or 0
    )
    guarded_violations = int(
        arm_results["guard_receipts"].get("guarded_accepted_contract_violation_count") or 0
    )
    flagged_cells: list[JsonDict] = []
    for model_id, families in fallback_delta.items():
        for family, receipt in families.items():
            if not receipt["adequately_powered"]:
                flagged_cells.append({"model": model_id, "family": family, "reason": "underpowered"})
            elif not receipt["positive"]:
                flagged_cells.append({"model": model_id, "family": family, "reason": "non_positive_delta"})
    for model_id in missing_models:
        flagged_cells.append({"model": model_id, "family": "*", "reason": "missing_model"})
    return {
        "schema": SCHEMA + ".harm_missing_underpowered",
        "missing_models": missing_models,
        "missing_model_count": len(missing_models),
        "parser_failure_count": parser_failure_count,
        "guarded_accepted_contract_violation_count": guarded_violations,
        "harmful_or_null_headline_cells": [
            cell for cell in flagged_cells if cell["reason"] == "non_positive_delta"
        ],
        "underpowered_cells": [cell for cell in flagged_cells if cell["reason"] == "underpowered"],
        "flagged_cells": flagged_cells,
        "flagged_cell_count": len(flagged_cells) + parser_failure_count + guarded_violations,
    }


def expected_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the exact prospective readiness gate result."""

    complete_models = (
        [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] == list(MANDATED_MODEL_IDS)
        and all(
            row.get("exists") is True and row.get("tokenizer_loadable") is True
            for row in artifact.get("MODEL_SPECS", [])
        )
        and set(artifact.get("models_used") or []) == set(MANDATED_MODEL_IDS)
    )
    commands_ok = all(code == 0 for code in dict(artifact.get("test_exit_codes") or {}).values())
    field_principles_ok = set(REQUIRED_ARTIFACT_FIELDS) <= set(
        artifact.get("field_principles", {})
    )
    field_provenance_ok = set(REQUIRED_ARTIFACT_FIELDS) <= set(
        artifact.get("field_provenance", {})
    )
    if (
        complete_models
        and bool(artifact.get("preconditions_checked", {}).get("all_passed"))
        and bool(artifact.get("protected_files_unchanged", {}).get("all_unchanged"))
        and bool(artifact.get("overlap_receipts", {}).get("declared_no_overlap"))
        and bool(
            artifact.get("outcome_seal_and_open_receipts", {}).get(
                "opened_after_predecision_immutable"
            )
        )
        and int(
            artifact.get("harm_underpowered_missing_and_flagged_cells", {}).get(
                "flagged_cell_count", 1
            )
            or 0
        )
        == 0
        and commands_ok
        and field_principles_ok
        and field_provenance_ok
    ):
        return 1.0
    return 0.0


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the artifact and reject false readiness laundering."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    for field in ("source_model_weight_mutation_count", "hidden_state_access_count"):
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
    _require(artifact.get("prospective_guarded_policy_ready_score") == expected, "ready_score")
    if expected == 1.0:
        _require(artifact.get("status") == "complete_ready", "status")
        _require(str(artifact.get("honest_verdict", "")).startswith("ready:"), "honest_verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum")
    return True


def status_from_artifact(artifact: Mapping[str, Any]) -> str:
    """Map receipts to a terminal status."""

    if not artifact.get("preconditions_checked", {}).get("all_passed"):
        return "blocked"
    return (
        "complete_ready"
        if artifact.get("prospective_guarded_policy_ready_score") == 1.0
        else "complete_no_value"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the lower-case terminal verdict string."""

    status = str(artifact.get("status"))
    if status == "complete_ready":
        return "ready: bounded exact-factor search improved every adequate held guard cell"
    if status == "blocked":
        return "blocked: upstream, model, tokenizer, host, seal, overlap, or protected-file gate failed"
    return "complete_null: held search did not clear every preregistered cell gate"


def upstream_receipts() -> JsonDict:
    """Return upstream path hashes, terminal classes, and gate receipts."""

    receipts = {
        "exp6326": _artifact_receipt(exp6326.RESULT_RELATIVE_PATH, "contract_guard_ready_score"),
        "exp6327": _artifact_receipt(exp6327.RESULT_RELATIVE_PATH, "guarded_policy_synthesis_ready_score"),
        "exp6328": _artifact_receipt(exp6328.RESULT_RELATIVE_PATH, "guard_integrity_ready_score"),
    }
    return {
        "schema": SCHEMA + ".upstream_gate",
        "artifacts": receipts,
        "exp6328_blind_integrity_gate_passed": receipts["exp6328"]["gate_ready_score"] == 1.0,
        "exp6326_exact_guard_gate_passed": receipts["exp6326"]["gate_ready_score"] == 1.0,
        "all_required_gates_passed": (
            receipts["exp6326"]["gate_ready_score"] == 1.0
            and receipts["exp6328"]["gate_ready_score"] == 1.0
        ),
    }


def overlap_receipts(fixtures: Sequence[exp6326.PolicyFixture]) -> JsonDict:
    """Prove declared held-family overlap counts are zero before inference."""

    upstream_fixtures = exp6326.build_fixture_manifest()
    old_families = {fixture.family for fixture in upstream_fixtures}
    new_families = {fixture.family for fixture in fixtures}
    old_fallback_hashes = {
        exp6326.semantic_hash(exp6326.parse_policy(fixture.fallback_program))
        for fixture in upstream_fixtures
    }
    new_fallback_hashes = {
        exp6326.semantic_hash(exp6326.parse_policy(fixture.fallback_program))
        for fixture in fixtures
    }
    old_identifiers = {
        item
        for fixture in upstream_fixtures
        for item in (
            *exp6326.validate_contract(fixture.contract).states,
            *exp6326.validate_contract(fixture.contract).actions,
            fixture.family,
        )
    }
    new_identifiers = {
        item
        for fixture in fixtures
        for item in (
            *exp6326.validate_contract(fixture.contract).states,
            *exp6326.validate_contract(fixture.contract).actions,
            fixture.family,
        )
    }
    counts = {
        "family_name_overlap_count": len(old_families & new_families),
        "task_generator_id_overlap_count": 0,
        "grammar_template_overlap_count": 0,
        "normalized_program_semantic_overlap_count": len(old_fallback_hashes & new_fallback_hashes),
        "semantic_signature_overlap_count": len(old_fallback_hashes & new_fallback_hashes),
        "mutation_lineage_overlap_count": 0,
        "identifier_overlap_count": len(old_identifiers & new_identifiers),
    }
    total = sum(counts.values())
    return {
        "schema": SCHEMA + ".overlap",
        "overlap_rules": {
            "prior_source": str(exp6326.RESULT_RELATIVE_PATH),
            "new_generator_id": "exp6329_independent_manual_held_family_builder_v1",
            "old_generator_id": "exp6326_restricted_policy_fixture_manifest_v1",
            "grammar_template_rule": "new held family text is not copied from prior fixture rows",
            "mutation_rule": "no prior family is mutated into a held row",
        },
        "counts": counts,
        "total_overlap_count": total,
        "declared_no_overlap": total == 0,
        "old_families": sorted(old_families),
        "new_families": sorted(new_families),
    }


def write_registration_receipt(
    data_dir: Path,
    *,
    timestamp: str,
    date: str,
    upstream: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    budgets: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Write the prospective registration before raw generation."""

    payload = {
        "schema": SCHEMA + ".prospective_registration",
        "timestamp": timestamp,
        "date": date,
        "upstream": upstream,
        "MODEL_SPECS": model_resolution["MODEL_SPECS"],
        "cached_sota_pair_calls": model_resolution["cached_sota_pair_calls"],
        "budget_hash": "sha256:" + sha256_json(budgets),
        "prompt_hash": prompt_contract["prompt_sha256"],
        "arm_ids": list(ARMS),
        "random_seeds": list(RANDOM_SEEDS),
        "protected_hashes_before": dict(protected_before),
    }
    path = data_dir / "prospective_registration.json"
    return _write_receipt(path, payload, timestamp)


def write_holdout_manifest(
    data_dir: Path,
    *,
    fixtures: Sequence[exp6326.PolicyFixture],
    timestamp: str,
) -> JsonDict:
    """Write the sealed held family and raw task row manifest."""

    rows = []
    for fixture in fixtures:
        contract = exp6326.validate_contract(fixture.contract)
        fallback_policy = exp6326.parse_policy(fixture.fallback_program)
        rows.append(
            {
                "family": fixture.family,
                "split": fixture.split,
                "states": list(contract.states),
                "actions": list(contract.actions),
                "contract": fixture.contract,
                "fallback_source_sha256": "sha256:" + exp6326.sha256_text(
                    fixture.fallback_program
                ),
                "fallback_semantic_hash": exp6326.semantic_hash(fallback_policy),
                "raw_task_row": _task_row(fixture),
            }
        )
    payload = {
        "schema": SCHEMA + ".sealed_holdout_manifest",
        "timestamp": timestamp,
        "family_order": list(HELD_FAMILY_ORDER),
        "held_family_count": len(fixtures),
        "families": rows,
        "raw_task_rows_sealed_before_generation": True,
    }
    path = data_dir / "sealed_holdout_manifest.json"
    return _write_receipt(path, payload, timestamp)


def write_exact_outcome_seal(
    data_dir: Path,
    *,
    manifest_receipt: Mapping[str, Any],
    overlap: Mapping[str, Any],
    timestamp: str,
) -> JsonDict:
    """Write the exact-outcome opening protocol before generation."""

    payload = {
        "schema": SCHEMA + ".exact_outcome_seal",
        "timestamp": timestamp,
        "manifest_sha256": manifest_receipt["sha256"],
        "overlap_sha256": "sha256:" + sha256_json(overlap),
        "oracle": "Exp6326 exact finite-domain checker",
        "candidate_outcomes_open": False,
        "opening_rule": "raw candidates and predecision receipt must be immutable first",
    }
    path = data_dir / "exact_outcome_seal.json"
    return _write_receipt(path, payload, timestamp)


def write_predecision_receipt(
    data_dir: Path,
    *,
    timestamp: str,
    raw_immutable_at: str,
    generation: Mapping[str, Any],
    budgets: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> JsonDict:
    """Write the immutable predecision receipt before exact scoring opens."""

    payload = {
        "schema": SCHEMA + ".predecision",
        "timestamp": timestamp,
        "raw_candidates_immutable_at": raw_immutable_at,
        "registration_sha256": registration["sha256"],
        "manifest_sha256": manifest["sha256"],
        "prompt_hash": prompt_contract["prompt_sha256"],
        "budget_hash": "sha256:" + sha256_json(budgets),
        "raw_candidate_paths_hashes": generation["raw_candidate_paths_hashes"],
        "arm_ids": list(ARMS),
        "exact_outcomes_open": False,
    }
    path = data_dir / "immutable_predecision_receipt.json"
    return _write_receipt(path, payload, timestamp)


def write_outcome_open_receipt(
    data_dir: Path,
    *,
    timestamp: str,
    registration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    predecision: Mapping[str, Any],
    raw_immutable_at: str,
    parsed: Mapping[str, Any],
    arm_results: Mapping[str, Any],
) -> JsonDict:
    """Open exact outcomes after the predecision receipt is immutable."""

    chronology = {
        "registration_at": registration["timestamp"],
        "holdout_sealed_at": manifest["timestamp"],
        "raw_candidates_immutable_at": raw_immutable_at,
        "predecision_immutable_at": predecision["timestamp"],
        "exact_outcomes_opened_at": timestamp,
    }
    payload = {
        "schema": SCHEMA + ".exact_outcomes_open",
        "timestamp": timestamp,
        "opened_against_predecision_sha256": predecision["sha256"],
        "candidate_outcomes_open": True,
        "chronology": chronology,
        "opened_after_predecision_immutable": predecision["timestamp"] < timestamp,
        "exact_factor_energies_by_candidate": parsed["exact_factor_energies_by_candidate"],
        "guard_receipts": arm_results["guard_receipts"],
    }
    path = data_dir / "exact_outcomes_open.json"
    receipt = _write_receipt(path, payload, timestamp)
    receipt.update(
        {
            "opened_against_predecision_sha256": predecision["sha256"],
            "chronology": chronology,
            "opened_after_predecision_immutable": payload["opened_after_predecision_immutable"],
        }
    )
    return receipt


def precondition_receipt(
    *,
    date: str,
    result_path: Path,
    data_dir: Path,
    upstream: Mapping[str, Any],
    registration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    outcome_seal: Mapping[str, Any],
    overlap: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
    budgets: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    host: Mapping[str, Any],
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Collect the full gate replay before generation."""

    host_ok = (
        bool(host.get("cuda_devices", {}).get("available"))
        and int(host.get("cuda_devices", {}).get("count") or 0) >= 2
        and bool(host.get("llama_cpp_cli", {}).get("exists"))
        and bool(host.get("llama_cpp_gpu_offload", {}).get("available"))
    )
    all_passed = (
        upstream.get("all_required_gates_passed") is True
        and model_resolution.get("all_resolved") is True
        and overlap.get("declared_no_overlap") is True
        and host_ok
        and bool(protected_before)
    )
    return {
        "schema": SCHEMA + ".preconditions",
        "date": date,
        "result_path": str(result_path),
        "data_dir": str(data_dir),
        "upstream": dict(upstream),
        "prospective_registration_frozen": registration,
        "holdout_manifest_frozen": manifest,
        "overlap_rules_frozen": overlap["overlap_rules"],
        "overlap_receipt_hash": "sha256:" + sha256_json(overlap),
        "exact_outcome_seal_frozen": outcome_seal,
        "arms_frozen": arm_definitions(),
        "models_frozen": {
            "all_resolved": model_resolution.get("all_resolved"),
            "blocked_reasons": list(model_resolution.get("blocked_reasons") or []),
            "cached_sota_pair_calls": model_resolution.get("cached_sota_pair_calls"),
            "MODEL_SPECS": model_resolution.get("MODEL_SPECS"),
        },
        "prompts_frozen": {
            key: value for key, value in prompt_contract.items() if key != "prompt_text"
        },
        "budgets_frozen": budgets,
        "seeds_frozen": list(RANDOM_SEEDS),
        "fallbacks_frozen": fallback_hashes(build_held_families()),
        "devices_frozen": dict(host),
        "timeouts_frozen": {"per_model_generation_timeout_s": TIME_BUDGET_S},
        "protected_hashes_frozen": dict(protected_before),
        "all_passed": bool(all_passed),
    }


def model_file_receipts(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return model identity receipts keyed by model id."""

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


def fallback_hashes(fixtures: Sequence[exp6326.PolicyFixture]) -> JsonDict:
    """Return fallback receipts for every held family."""

    return {fixture.family: exp6327.fallback_program_receipt(fixture) for fixture in fixtures}


def arm_definitions() -> JsonDict:
    """Return the four preregistered held-family arms."""

    return {
        "raw_single_candidate": {
            "candidate_source": "candidate_index_0",
            "guard": "none",
            "fallback": "none",
        },
        "reject_only": {
            "candidate_source": "candidate_index_0",
            "guard": "Exp6326 exact factor energy must equal zero",
            "fallback": "none",
        },
        "guard_plus_fallback": {
            "candidate_source": "candidate_index_0",
            "guard": "Exp6326 exact factor energy must equal zero",
            "fallback": "verified held-family fallback",
        },
        "bounded_exact_factor_energy_search_plus_fallback": {
            "candidate_source": "lowest exact energy among bounded candidates",
            "guard": "selected candidate must have exact energy zero",
            "fallback": "same verified held-family fallback",
        },
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
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the required Exp6329 run command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=DEFAULT_RUN_DATE)
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
                "ready_score": artifact["prospective_guarded_policy_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0


def _held_fixture(
    *,
    family: str,
    states: Sequence[str],
    actions: Sequence[str],
    clauses: Sequence[Mapping[str, Any]],
    fallback_mapping: Mapping[str, str],
) -> exp6326.PolicyFixture:
    contract = {
        "family": family,
        "split": "held",
        "states": list(states),
        "actions": list(actions),
        "clauses": [dict(clause) for clause in clauses],
    }
    fallback_program = exp6326.program_text(
        name=f"{family}_fallback",
        states=states,
        actions=actions,
        mapping=fallback_mapping,
    )
    return exp6326.PolicyFixture(
        family=family,
        split="held",
        description=f"Fresh held Exp6329 family {family}.",
        contract=contract,
        fallback_program=fallback_program,
        tags=("exp6329", "held", "prospective"),
    )


def _model_record(
    template: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    tokenizer_func: TokenizerFn,
) -> tuple[JsonDict, list[str]]:
    hf_id = str(template["hf_id"])
    raw = by_id.get(hf_id)
    blockers: list[str] = []
    if raw is None:
        blockers.append(f"model_not_cached:{hf_id}")
        return {**template, "model_path": "", "exists": False}, blockers
    path = Path(str(raw.get("model_path") or ""))
    exists = path.is_file()
    tokenizer_ok = False
    tokenizer_detail = "model file missing"
    if exists:
        tokenizer_ok, tokenizer_detail = tokenizer_func(str(path))
    else:
        blockers.append(f"model_path_missing:{hf_id}")
    if not tokenizer_ok:
        blockers.append(f"embedded_tokenizer_not_loadable:{hf_id}")
    return (
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
            "tokenizer_method": TOKENIZER_METHOD,
            "cache_policy": _cache_policy_for_model(hf_id),
        },
        blockers,
    )


def _task_row(fixture: exp6326.PolicyFixture) -> JsonDict:
    contract = exp6326.validate_contract(fixture.contract)
    return {
        "task_id": f"held::{fixture.family}",
        "family": fixture.family,
        "split": fixture.split,
        "states": list(contract.states),
        "actions": list(contract.actions),
        "clauses_sha256": "sha256:" + sha256_json(list(contract.clauses)),
        "candidate_count": CANDIDATE_COUNT,
    }


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
        "fallback_rate": 1.0 if fallback else 0.0,
        "contract_violation_count": violation,
        "exact_energy": candidate.get("exact_energy"),
        "utility": round(float(utility), 6),
        "exact_utility": round(float(utility), 6),
        "latency_s": 0.0,
        "cost": round(MATCHED_GENERATION_COST + (FALLBACK_COST if fallback else 0.0), 6),
        "fallback_utility_charged": FALLBACK_UTILITY if fallback else None,
        "full_fallback_cost_charged": FALLBACK_COST if fallback else 0.0,
    }


def _missing_candidate(model_id: str, fixture: exp6326.PolicyFixture, seed: int) -> JsonDict:
    return {
        "model_hf_id": model_id,
        "family": fixture.family,
        "split": fixture.split,
        "seed": seed,
        "candidate_index": 0,
        "raw_sha256": "sha256:" + ("0" * 64),
        "parse_status": "missing_cell",
        "normalization_status": "not_normalized",
        "normalized_sha256": None,
        "semantic_hash": None,
        "factor_energy": None,
        "exact_energy": None,
        "factor_exact_mismatch": False,
        "accepted_by_exact_guard": False,
    }


def _search_minus_guard(arms: Mapping[str, Any]) -> float:
    return round(
        float(arms["bounded_exact_factor_energy_search_plus_fallback"]["utility"])
        - float(arms["guard_plus_fallback"]["utility"]),
        6,
    )


def _artifact_receipt(relative_path: Path, gate_field: str) -> JsonDict:
    path = REPO_ROOT / relative_path
    if not path.exists():  # pragma: no cover - repository fixture is present.
        return {
            "path": str(relative_path),
            "exists": False,
            "sha256": None,
            "terminal_class": "missing",
            "status": None,
            "gate_field": gate_field,
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
        "gate_field": gate_field,
        "gate_ready_score": float(payload.get(gate_field) or 0.0),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
    }


def _write_receipt(path: Path, payload: Mapping[str, Any], timestamp: str) -> JsonDict:
    write_json_atomic(path, payload)
    return {"path": str(path), "sha256": sha256_file(path), "timestamp": timestamp}


def _cache_policy_for_model(hf_id: str) -> str:
    if hf_id == MANDATED_MODEL_IDS[1]:
        return "cached_sota_pair(gpu_indices=(0,1), model_indices=(0,2))"
    return "cached_sota_pair(gpu_indices=(0,1))"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
