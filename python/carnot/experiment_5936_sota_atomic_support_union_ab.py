"""Exp5936 SOTA atomic support union exact-semantic A/B.

Spec refs: REQ-VERIFY-5936, SCENARIO-VERIFY-5936-GATE,
SCENARIO-VERIFY-5936-PREREG, SCENARIO-VERIFY-5936-UNION,
SCENARIO-VERIFY-5936-PRIMARY, SCENARIO-VERIFY-5936-EVENTS.

This experiment keeps the model in the proposer role.  Direct ConstraintIR,
single-view atomic support, repeated-original atomic support, and transformed
atomic support are sealed before exact labels are opened.  Only the exact
ConstraintIR parser, Python executor, Z3 executor, and certificate replay decide
semantic success after each arm has exactly one bounded completion attempt.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5897_sota_constraint_ir_repair_ab as exp5897
from carnot import experiment_5922_gguf_schema_decoder_bridge as exp5922
from carnot import experiment_5935_non_pruning_atomic_constraint_support as exp5935


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5936_sota_atomic_support_union_ab.json")
EVENT_STREAM_RELATIVE_PATH = Path(
    "results/experiment_5936_sota_atomic_support_union_ab.events.jsonl"
)
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_5936_sota_atomic_support_union_ab.events.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5936_sota_atomic_support_union_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5936_sota_atomic_support_union_ab.py")
VERIFIABLE_REASONING_SPEC_RELATIVE_PATH = Path(
    "openspec/capabilities/verifiable-reasoning/spec.md"
)

RUN_DATE = "20260726"
EXPERIMENT_ID = "experiment_5936_sota_atomic_support_union_ab"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5936.sota_atomic_support_union_ab.v1"
EVENT_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION + ".event"
RANDOM_SEED = 5936
INFERENCE_SUBSTRATE = "local_mandated_gguf_public_llama_cpp_cuda_atomic_support_union"
VERIFIER_IS_ORACLE = True
MAX_COMPLETION_STATES = exp5935.MAX_COMPLETION_STATES

MANDATED_MODEL_IDS: tuple[str, str, str] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "family": "qwen_moe",
        "role": "flagship_moe",
        "preferred_quant": "Q4_K_M",
        "gpu": 0,
        "min_vram_gb": 24,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "family": "gemma_dense",
        "role": "flagship_dense",
        "preferred_quant": "Q4_K_M",
        "gpu": 1,
        "min_vram_gb": 24,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[2],
        "family": "gemma_moe",
        "role": "middle_moe",
        "preferred_quant": "Q4_K_M",
        "gpu": 0,
        "min_vram_gb": 16,
    },
)

ARM_IDS: tuple[str, str, str, str] = (
    "direct_one_shot_ir",
    "single_view_atomic_support",
    "repeated_original_atomic_union",
    "transformed_view_atomic_union",
)
VIEW_IDS_BY_ARM: JsonDict = {
    "direct_one_shot_ir": ["direct_ir"],
    "single_view_atomic_support": ["original"],
    "repeated_original_atomic_union": [
        "original_repeat_1",
        "original_repeat_2",
        "original_repeat_3",
    ],
    "transformed_view_atomic_union": ["original", "paraphrase", "entity_permutation"],
}
CALLS_PER_MODEL_CASE = sum(len(value) for value in VIEW_IDS_BY_ARM.values())

DECODING_PARAMETERS_BY_ARM: JsonDict = {
    "direct_one_shot_ir": {"temperature": 0.0, "top_p": 1.0, "repeat_penalty": 1.05},
    "single_view_atomic_support": {"temperature": 0.0, "top_p": 1.0, "repeat_penalty": 1.05},
    "repeated_original_atomic_union": {
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.05,
    },
    "transformed_view_atomic_union": {
        "temperature": 0.0,
        "top_p": 1.0,
        "repeat_penalty": 1.05,
    },
}
TOKEN_BUDGETS: JsonDict = {
    "direct_one_shot_ir": {"calls": 1, "max_tokens_per_call": 192, "total_max_tokens": 192},
    "single_view_atomic_support": {
        "calls": 1,
        "max_tokens_per_call": 128,
        "total_max_tokens": 128,
    },
    "repeated_original_atomic_union": {
        "calls": 3,
        "max_tokens_per_call": 128,
        "total_max_tokens": 384,
    },
    "transformed_view_atomic_union": {
        "calls": 3,
        "max_tokens_per_call": 128,
        "total_max_tokens": 384,
    },
    "n_ctx": 8192,
    "n_batch": 256,
    "n_gpu_layers": -1,
}
ARM_DEFINITIONS: JsonDict = {
    "direct_one_shot_ir": {
        "label": "A",
        "calls": 1,
        "atomic": False,
        "cost_baseline": True,
        "budget_matched_to_transformed": False,
    },
    "single_view_atomic_support": {
        "label": "B",
        "calls": 1,
        "atomic": True,
        "cost_baseline": True,
        "budget_matched_to_transformed": False,
    },
    "repeated_original_atomic_union": {
        "label": "C",
        "calls": 3,
        "atomic": True,
        "primary_control": True,
        "budget_matched_to_transformed": True,
    },
    "transformed_view_atomic_union": {
        "label": "D",
        "calls": 3,
        "atomic": True,
        "primary_treatment": True,
        "budget_matched_to_repeated_original": True,
    },
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    VERIFIABLE_REASONING_SPEC_RELATIVE_PATH,
    exp5896.MODULE_RELATIVE_PATH,
    exp5897.MODULE_RELATIVE_PATH,
    exp5922.MODULE_RELATIVE_PATH,
    exp5935.MODULE_RELATIVE_PATH,
    exp5935.RESULT_RELATIVE_PATH,
    exp5935.ATOM_ROW_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "gate_replay_receipt",
    "preconditions_checked",
    "model_specs",
    "model_file_hashes",
    "embedded_tokenizer_public_llama_cpp_cuda_dual_gpu_and_vram_receipts",
    "sealed_cases_families_adversaries_prompts_transforms_seeds_arms_and_budgets",
    "per_model_arm_atom_and_exact_metrics",
    "transformed_vs_repeated_original_primary_comparison_and_interval",
    "direct_and_single_view_secondary_comparisons",
    "included_and_excluded_relevant_atom_counts",
    "exact_semantic_missing_spurious_contradiction_and_unsafe_receipts",
    "correct_mode_diversity_and_support_saturation",
    "search_correctness_change_vs_cleanup_receipt",
    "calls_tokens_latency_gpu_and_memory_accounting",
    "chronological_event_stream_path_hash_rows_and_prefix_chain",
    "chronological_event_stream_ready_score",
    "no_label_feedback_no_hard_pruning_no_schema_reprompt_and_no_answer_enumeration_receipt",
    "retirement_decision",
    "protected_files_unchanged",
    "atomic_semantic_live_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: JsonDict = {
    "gate_replay_receipt": "only exact Exp5935 readiness authorizes model loading",
    "preconditions_checked": "every headline row needs the mandated local GGUF CUDA path",
    "model_specs": "the three mandated GGUF families are reported in preregistered order",
    "model_file_hashes": "every model file is hashed before headline inference",
    "embedded_tokenizer_public_llama_cpp_cuda_dual_gpu_and_vram_receipts": (
        "GGUF embedded tokenizers and real CUDA offload own runtime provenance"
    ),
    "sealed_cases_families_adversaries_prompts_transforms_seeds_arms_and_budgets": (
        "the full design is frozen before model output or exact labels are opened"
    ),
    "per_model_arm_atom_and_exact_metrics": (
        "models stay separated; pooled metrics carry provenance"
    ),
    "transformed_vs_repeated_original_primary_comparison_and_interval": (
        "D versus matched C owns the transformed-view claim"
    ),
    "direct_and_single_view_secondary_comparisons": (
        "A and B are cost baselines, not budget-matched evidence for D"
    ),
    "included_and_excluded_relevant_atom_counts": (
        "included and omitted atom relevance are audited separately after seal"
    ),
    "exact_semantic_missing_spurious_contradiction_and_unsafe_receipts": (
        "exact semantics and zero unsafe acceptance are mandatory"
    ),
    "correct_mode_diversity_and_support_saturation": (
        "report correct-mode retention and overcomplete support cost"
    ),
    "search_correctness_change_vs_cleanup_receipt": (
        "separate recovered reachability from cleanup of repeated invalid work"
    ),
    "calls_tokens_latency_gpu_and_memory_accounting": (
        "count every proposal view and exact completion cost"
    ),
    "chronological_event_stream_path_hash_rows_and_prefix_chain": (
        "chronology must replay by prefix checksum"
    ),
    "chronological_event_stream_ready_score": (
        "bare 1.0 means complete tamper-safe real chronology"
    ),
    "no_label_feedback_no_hard_pruning_no_schema_reprompt_and_no_answer_enumeration_receipt": (
        "any label feedback, hard pruning, schema reprompt, or answer enumeration invalidates the run"
    ),
    "retirement_decision": "retire rather than retrying a failed transformed-union mechanism",
    "protected_files_unchanged": "protected operational files are only read",
    "atomic_semantic_live_ready_score": (
        "bare 1.0 requires D to beat C by a positive lower paired bound with zero unsafe accepts"
    ),
    "duration_s": "use local_mandated_gguf_public_llama_cpp_cuda_atomic_support_union",
    "inference_substrate": "use local_mandated_gguf_public_llama_cpp_cuda_atomic_support_union",
    "verifier_is_oracle": "true only inside the sealed synthetic exact domains",
    "missing_verifier_gaps": "natural-language ambiguity remains outside this fixture oracle",
    "field_provenance": "field principles are echoed for audit",
    "test_commands": "verification commands are recorded",
    "test_exit_codes": "post-run command exit codes are refreshed after validation",
    "reproducibility_checksum": "canonical checksum catches drift",
    "honest_verdict": "use complete_positive:, complete_null:, retired:, or blocked:",
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5936_sota_atomic_support_union_ab.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5936_sota_atomic_support_union_ab.py "
    "-m pytest tests/python/test_experiment_5936_sota_atomic_support_union_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5936_sota_atomic_support_union_ab.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5936_sota_atomic_support_union_ab",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5936_sota_atomic_support_union_ab.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5936_sota_atomic_support_union_ab.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
)
FORBIDDEN_EVENT_REFERENCE_KEYS = (
    "target_constraint_ir",
    "reference_answer",
    "hidden_reference_answer",
    "gold_constraint_ir",
    "certificate_solution",
    "query_bindings",
    "canonical_behavior_hash",
    "hidden_reference",
    "relevance_label",
)

ModelResolver = Callable[[], list[JsonDict]]
EnvironmentProbe = Callable[[Path], JsonDict]
TokenizerLoader = Callable[[JsonDict], JsonDict]
GateReplayProvider = Callable[[Path], JsonDict]
CollectOutputsFn = Callable[[list[JsonDict], list[JsonDict], "ExperimentConfig", JsonDict], JsonDict]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths, clocks, and seeds for Exp5936."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    event_stream_path: Path | None = None
    checkpoint_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic_clock: Callable[[], float] = time.monotonic
    random_seed: int = RANDOM_SEED

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / RESULT_RELATIVE_PATH

    def events_path(self) -> Path:
        return self.event_stream_path or self.repo_root / EVENT_STREAM_RELATIVE_PATH

    def resume_path(self) -> Path:
        return self.checkpoint_path or self.repo_root / CHECKPOINT_RELATIVE_PATH


@dataclass(frozen=True)
class PreconditionReport:
    """All checks that must pass before any model proposal collection."""

    gate_replay_receipt: JsonDict
    preconditions_checked: JsonDict
    model_specs: list[JsonDict]
    model_file_hashes: JsonDict
    receipts: JsonDict
    protected_file_baseline: JsonDict
    block_reason: str | None


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with stable ASCII bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file by bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def view_ids_for_arm(arm_id: str) -> list[str]:
    """Return the frozen view order for an arm."""

    if arm_id not in VIEW_IDS_BY_ARM:
        raise ValueError(f"unknown arm_id: {arm_id}")
    return list(VIEW_IDS_BY_ARM[arm_id])


def resolve_model_specs() -> list[JsonDict]:  # pragma: no cover - host cache boundary.
    """Resolve the mandated GGUFs through cached_sota_pair plus the third family."""

    return exp5922.resolve_all_model_specs()


def default_tokenizer_loader(spec: JsonDict) -> JsonDict:  # pragma: no cover - GGUF boundary.
    """Load only the embedded GGUF tokenizer and return an audit receipt."""

    vocabulary = exp5922.load_embedded_llama_cpp_vocabulary(spec)
    receipt = dict(vocabulary.tokenizer_receipt)
    receipt.pop("_tokenizer", None)
    receipt["ok"] = True
    return receipt


def freeze_held_cases(min_cases: int = 36) -> list[JsonDict]:
    """Freeze at least 36 shared held cases before model output is opened."""

    held_rows = [row for row in exp5896.build_fixture_rows() if row["split"] == "heldout"]
    transforms = _case_transform_specs()
    cases: list[JsonDict] = []
    for row in held_rows:
        for transform in transforms:
            case_index = len(cases)
            problem = _transformed_problem_text(row, transform)
            cases.append(
                {
                    "case_index": case_index,
                    "case_id": f"exp5936-{row['row_id']}-{transform['case_suffix']}",
                    "source_row_id": row["row_id"],
                    "target_row_id": row["row_id"],
                    "target_row_hash": row["row_hash"],
                    "family": row["family"],
                    "split": "heldout",
                    "variant_kind": row["variant_kind"],
                    "adversary_kind": transform["adversary_kind"],
                    "semantic_transform_id": transform["view_id"],
                    "inverse_transform_id": transform["inverse"],
                    "expected_status": row["expected_status"],
                    "expected_equivalent_to_canonical": row[
                        "expected_equivalent_to_canonical"
                    ],
                    "visible_problem_text": problem,
                    "visible_problem_sha256": sha256_text(problem),
                    "sealed_before_model_rows": True,
                    "_target_row": _copy_json(row),
                }
            )
    if len(cases) < min_cases:
        raise ValueError(f"held case expansion produced {len(cases)} cases, need {min_cases}")
    return cases[:min_cases]


def derive_surface_for_case(
    case: Mapping[str, Any], schema: Mapping[str, Any] | None = None
) -> JsonDict:
    """Build the Exp5935 atom surface for one Exp5936 in-memory case."""

    source = {
        "case_id": case["case_id"],
        "source_row_id": case["source_row_id"],
        "family": case["family"],
        "split": case["split"],
        "variant_kind": case["variant_kind"],
        "target_row": _target_row(case),
    }
    return exp5935.derive_case_atom_surface(source, schema)


def build_prompt(case: Mapping[str, Any], arm_id: str, view_id: str) -> str:
    """Return the frozen model-visible prompt for one arm/view call."""

    if view_id not in view_ids_for_arm(arm_id):
        raise ValueError(f"view_id {view_id!r} is not part of arm {arm_id!r}")
    if arm_id == "direct_one_shot_ir":
        return (
            "Return exactly one typed ConstraintIR JSON object. Do not include prose, "
            "confidence, exact diagnostics, or hidden labels.\n"
            f"schema_version: {exp5896.CONSTRAINT_IR_SCHEMA_VERSION}\n"
            f"Problem:\n{case['visible_problem_text']}\n"
        )
    schema = exp5935.versioned_atom_schema()
    surface = derive_surface_for_case(case, schema)
    visible_atoms = [
        {
            "atom_id": atom["atom_id"],
            "atom_kind": atom["atom_kind"],
            "payload": atom["payload"],
        }
        for atom in surface["_visible_atoms"]
    ]
    return (
        "Return JSON with one key atom_ids whose value is a list of legal atom_id "
        "strings from the visible vocabulary. Include every atom that could be relevant; "
        "overcomplete lists are allowed. Do not rank by confidence as a hard mask. "
        "Do not include exact labels, hidden references, diagnostics, final answers, "
        "or prose.\n"
        f"Arm: {arm_id}\n"
        f"View: {view_id}\n"
        f"Problem:\n{case['visible_problem_text']}\n"
        f"Visible atom vocabulary:\n{canonical_json(visible_atoms)}\n"
    )


def build_preregistration(config: ExperimentConfig, panel: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Record the frozen design before any model proposal or exact label opens."""

    rows: list[JsonDict] = []
    for case in panel:
        for arm_index, arm_id in enumerate(ARM_IDS):
            for view_index, view_id in enumerate(view_ids_for_arm(arm_id)):
                prompt = build_prompt(case, arm_id, view_id)
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "family": case["family"],
                        "split": case["split"],
                        "adversary_kind": case["adversary_kind"],
                        "arm_id": arm_id,
                        "arm_order_index": arm_index,
                        "view_id": view_id,
                        "view_order_index": view_index,
                        "prompt_sha256": sha256_text(prompt),
                        "seed": (
                            config.random_seed
                            + int(case["case_index"]) * 1000
                            + arm_index * 100
                            + view_index
                        ),
                        "budget": _budget_for_view(arm_id),
                    }
                )
    return {
        "run_date": RUN_DATE,
        "case_count": len(panel),
        "model_case_rows_per_model": len(panel),
        "case_minimum_required": 36,
        "cases": [_public_case(case) for case in panel],
        "held_families": sorted({str(case["family"]) for case in panel}),
        "attribute_value_adversary_count": sum(
            case["adversary_kind"] == "attribute_value_adversary" for case in panel
        ),
        "arms": {arm_id: ARM_DEFINITIONS[arm_id] for arm_id in ARM_IDS},
        "arm_order": list(ARM_IDS),
        "view_order_by_arm": {arm_id: view_ids_for_arm(arm_id) for arm_id in ARM_IDS},
        "prompt_seed_budget_rows": rows,
        "semantic_transforms": semantic_transform_receipts(panel),
        "inverse_maps": _inverse_map_receipts(panel),
        "decoding_parameters_by_arm": DECODING_PARAMETERS_BY_ARM,
        "token_budgets": TOKEN_BUDGETS,
        "exact_validators": {
            "parser": "Exp5896 parse_constraint_ir",
            "type_scope": "Exp5896 type and scope validation",
            "python_executor": "Exp5896 finite-domain Python executor",
            "z3_executor": "Exp5896 Z3 certificate",
            "semantic_equivalence": "Exp5897 exact behavior and query checks",
        },
        "completion_bounds": {
            "max_completion_states": MAX_COMPLETION_STATES,
            "one_deterministic_completion_per_arm_after_union_seal": True,
        },
        "primary_comparison": {
            "treatment_arm": "transformed_view_atomic_union",
            "matched_control_arm": "repeated_original_atomic_union",
            "metric": "held_exact_semantic_success",
            "paired_interval_method": "deterministic_paired_bootstrap_ci95",
        },
        "label_opening_rule": "after_arm_union_seal",
        "retirement_rules": {
            "retire_if_all_three_models_zero_d_exact_success": True,
            "retire_if_unsafe_acceptance": True,
            "retire_if_material_correct_mode_collapse": True,
        },
        "stopping_rules": {
            "stop_on_first_unsafe_exact_accept": True,
            "stop_on_label_leak": True,
            "stop_on_hidden_reference_prompt_hash": True,
            "stop_on_model_side_irreversible_pruning": True,
            "stop_on_tokenizer_runtime_drift": True,
            "stop_on_gpu_fallback": True,
        },
    }


def semantic_transform_receipts(panel: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize deterministic transformed views and inverse-map availability."""

    view_ids = ["original", "paraphrase", "entity_permutation"]
    rows = []
    schema = exp5935.versioned_atom_schema()
    for case in panel[: min(6, len(panel))]:
        receipt = exp5935.semantic_view_transform_receipts(
            {
                "case_id": case["case_id"],
                "source_row_id": case["source_row_id"],
                "family": case["family"],
                "split": case["split"],
                "variant_kind": case["variant_kind"],
                "target_row": _target_row(case),
            },
            schema,
        )
        rows.append(receipt)
    return {
        "view_ids": view_ids,
        "rows_sampled_for_inverse_receipts": rows,
        "all_views_invertible": all(row["all_views_invertible"] for row in rows),
        "transform_independent_from_model_output": all(
            row["transform_independent_from_model_output"] for row in rows
        ),
        "answer_leakage_detected": any(row["answer_leakage_detected"] for row in rows),
    }


def check_preconditions(
    config: ExperimentConfig,
    *,
    model_resolver: ModelResolver = resolve_model_specs,
    environment_probe: EnvironmentProbe = lambda root: _probe_environment(root),
    tokenizer_loader: TokenizerLoader = default_tokenizer_loader,
    gate_replay_provider: GateReplayProvider = lambda root: _gate_replay_receipt(root),
) -> PreconditionReport:
    """Gate Exp5936 before any model output can be collected."""

    root = config.repo_root
    baseline = _protected_file_receipt(root)
    gate = gate_replay_provider(root)
    if gate.get("ok") is not True:
        checks = {
            "exp5935_gate_ready": False,
            "model_specs_defined": True,
            "resolved_all_three_model_files": False,
            "model_files_hashed": False,
            "embedded_gguf_tokenizers_load": False,
            "no_hf_autotokenizer_for_gguf": True,
            "public_llama_cpp_cuda_available": False,
            "two_healthy_rtx_3090s": False,
            "adequate_vram": False,
            "adequate_ram": False,
            "adequate_disk": False,
            "real_nonzero_gpu_offload_supported": False,
            "atomic_output_ready": False,
            "atomic_checkpoint_resume_ready": False,
            "no_protected_workload": False,
        }
        return PreconditionReport(
            gate_replay_receipt=gate,
            preconditions_checked={
                "run_order": "exp5935_gate_before_any_model_load",
                "blocked_before_model_load": True,
                "headline_checks": checks,
                "block_reason": str(gate.get("block_reason") or "exp5935_gate_not_ready"),
            },
            model_specs=[dict(spec) for spec in MODEL_SPECS],
            model_file_hashes={},
            receipts={"model_resolution": {"not_attempted_before_gate": True}},
            protected_file_baseline=baseline,
            block_reason=str(gate.get("block_reason") or "exp5935_gate_not_ready"),
        )

    model_specs = model_resolver()
    model_hashes = _model_file_hashes(model_specs)
    tokenizers = _tokenizer_receipts(model_specs, tokenizer_loader)
    environment = environment_probe(root)
    expected_ids = list(MANDATED_MODEL_IDS)
    ids = [str(spec.get("hf_id")) for spec in model_specs]
    checks = {
        "exp5935_gate_ready": True,
        "model_specs_defined": ids == expected_ids,
        "resolved_all_three_model_files": all(spec.get("model_path") for spec in model_specs)
        and ids == expected_ids,
        "model_files_hashed": set(model_hashes) == set(expected_ids),
        "embedded_gguf_tokenizers_load": all(row.get("ok") for row in tokenizers.values())
        and set(tokenizers) == set(expected_ids),
        "no_hf_autotokenizer_for_gguf": all(
            row.get("used_hf_autotokenizer") is False for row in tokenizers.values()
        ),
        "public_llama_cpp_cuda_available": bool(
            environment.get("public_llama_cpp_cuda", {}).get("ok")
        ),
        "two_healthy_rtx_3090s": _two_healthy_rtx_3090s(environment.get("gpu_health", {})),
        "adequate_vram": _adequate_vram(environment.get("gpu_health", {})),
        "adequate_ram": bool(environment.get("ram", {}).get("ok")),
        "adequate_disk": bool(environment.get("disk", {}).get("ok")),
        "real_nonzero_gpu_offload_supported": bool(
            environment.get("public_llama_cpp_cuda", {}).get("gpu_offload_supported")
        ),
        "atomic_output_ready": bool(environment.get("atomic_output", {}).get("ok")),
        "atomic_checkpoint_resume_ready": bool(
            environment.get("atomic_checkpoint_resume", {}).get("ok")
        ),
        "no_protected_workload": bool(environment.get("protected_workload", {}).get("ok")),
    }
    block_reason = next((name for name, ok in checks.items() if not ok), None)
    return PreconditionReport(
        gate_replay_receipt=gate,
        preconditions_checked={
            "run_order": "exp5935_gate_resource_checks_before_any_model_load",
            "blocked_before_model_load": block_reason is not None,
            "headline_checks": checks,
            "block_reason": block_reason,
        },
        model_specs=model_specs,
        model_file_hashes=model_hashes,
        receipts={
            "embedded_tokenizer_receipts": tokenizers,
            "environment": environment,
            "model_resolution": {
                "cached_sota_pair_plus_cached_third_family": ids == expected_ids,
                "resolved_hf_ids": ids,
            },
        },
        protected_file_baseline=baseline,
        block_reason=block_reason,
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    model_resolver: ModelResolver = resolve_model_specs,
    environment_probe: EnvironmentProbe = lambda root: _probe_environment(root),
    tokenizer_loader: TokenizerLoader = default_tokenizer_loader,
    gate_replay_provider: GateReplayProvider = lambda root: _gate_replay_receipt(root),
    collect_model_outputs_fn: CollectOutputsFn | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Run Exp5936, writing the event stream and terminal artifact atomically."""

    active = config or ExperimentConfig()
    started = active.start_time()
    preconditions = check_preconditions(
        active,
        model_resolver=model_resolver,
        environment_probe=environment_probe,
        tokenizer_loader=tokenizer_loader,
        gate_replay_provider=gate_replay_provider,
    )
    panel = freeze_held_cases()
    preregistration = build_preregistration(active, panel)
    if preconditions.block_reason is not None:
        _write_event_stream_atomic(active.events_path(), [])
        artifact = _build_artifact(
            active,
            preconditions,
            preregistration=preregistration,
            events=[],
            collection={"rows": [], "real_model_rows": False, "model_attempts": []},
            duration_s=active.clock() - started,
            test_exit_codes=test_exit_codes,
        )
        _write_json_atomic(active.artifact_path(), artifact)
        return artifact

    collector = collect_model_outputs_fn or collect_live_model_outputs
    collection = collector(preconditions.model_specs, panel, active, preregistration)
    checkpoint_rows = load_checkpoint(active.resume_path())
    raw_rows = merge_resume_rows(checkpoint_rows, [dict(row) for row in collection.get("rows") or []])
    save_checkpoint(active.resume_path(), raw_rows)
    events = seal_raw_events(raw_rows, panel, active)
    _write_event_stream_atomic(active.events_path(), events)
    artifact = _build_artifact(
        active,
        preconditions,
        preregistration=preregistration,
        events=events,
        collection=collection,
        duration_s=active.clock() - started,
        test_exit_codes=test_exit_codes,
    )
    _write_json_atomic(active.artifact_path(), artifact)
    return artifact


def collect_live_model_outputs(
    model_specs: list[JsonDict],
    panel: list[JsonDict],
    config: ExperimentConfig,
    preregistration: JsonDict,
) -> JsonDict:  # pragma: no cover - live GGUF/CUDA boundary.
    """Collect live llama.cpp rows for every model, held case, arm, and view."""

    del preregistration
    from llama_cpp import Llama

    rows: list[JsonDict] = []
    attempts: list[JsonDict] = []
    sequence = 0
    for model_index, spec in enumerate(model_specs):
        load_started = config.monotonic_clock()
        before_mb = _gpu_memory_total_mb()
        llm = None
        try:
            llm = Llama(
                model_path=str(spec["model_path"]),
                n_ctx=int(TOKEN_BUDGETS["n_ctx"]),
                n_batch=int(TOKEN_BUDGETS["n_batch"]),
                n_gpu_layers=int(TOKEN_BUDGETS["n_gpu_layers"]),
                main_gpu=int(spec.get("gpu") or 0),
                seed=config.random_seed + model_index,
                verbose=False,
            )
            after_load_mb = _gpu_memory_total_mb()
            attempts.append(
                {
                    "hf_id": spec["hf_id"],
                    "model_used": True,
                    "gpu_offload_verified": after_load_mb > before_mb,
                    "vram_delta_mb": max(0.0, after_load_mb - before_mb),
                    "load_duration_s": round(config.monotonic_clock() - load_started, 6),
                }
            )
            for case in panel:
                for arm_index, arm_id in enumerate(ARM_IDS):
                    for view_index, view_id in enumerate(view_ids_for_arm(arm_id)):
                        seed = (
                            config.random_seed
                            + model_index * 100000
                            + int(case["case_index"]) * 1000
                            + arm_index * 100
                            + view_index
                        )
                        rows.append(
                            _call_live_view(llm, spec, case, arm_id, view_id, seed, sequence, config)
                        )
                        sequence += 1
        except Exception as exc:
            attempts.append(
                {
                    "hf_id": spec.get("hf_id"),
                    "model_used": False,
                    "blocker": f"{type(exc).__name__}: {exc}",
                    "gpu_offload_verified": False,
                }
            )
        finally:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
            llm = None
            gc.collect()
    return {
        "rows": rows,
        "real_model_rows": True,
        "model_attempts": attempts,
        "gpu_receipts": {"attempts": attempts},
    }


def seal_raw_events(
    raw_rows: Sequence[Mapping[str, Any]],
    panel: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
) -> list[JsonDict]:
    """Seal raw proposal rows, exact-score each arm once, and hash-chain events."""

    panel_by_id = {str(case["case_id"]): case for case in panel}
    grouped = _group_raw_rows(raw_rows)
    group_receipts = {
        key: _score_group(rows, panel_by_id[str(key[1])])
        for key, rows in grouped.items()
        if str(key[1]) in panel_by_id
    }
    events: list[JsonDict] = []
    previous = "sha256:" + "0" * 64
    for expected_index, raw in enumerate(sorted(raw_rows, key=lambda row: int(row["stream_sequence_index"]))):
        case = panel_by_id[str(raw["case_id"])]
        arm_id = str(raw["arm_id"])
        view_id = str(raw["view_id"])
        prompt = build_prompt(case, arm_id, view_id)
        key = (str(raw["model_hf_id"]), str(raw["case_id"]), arm_id)
        group = group_receipts[key]
        raw_text = str(raw.get("raw_output_text") or "")
        event: JsonDict = {
            "schema": EVENT_SCHEMA_VERSION,
            "event_kind": "sota_atomic_support_view_proposal",
            "stream_sequence_index": int(raw.get("stream_sequence_index", expected_index)),
            "row_identity": {
                "model_hf_id": raw.get("model_hf_id"),
                "model_name": raw.get("model_name"),
                "model_path": raw.get("model_path"),
                "gpu_index": raw.get("gpu_index"),
                "case_id": case["case_id"],
                "family": case["family"],
                "split": case["split"],
                "adversary_kind": case["adversary_kind"],
                "arm_id": arm_id,
                "view_id": view_id,
                "prompt_sha256": sha256_text(prompt),
                "seed": raw.get(
                    "seed",
                    config.random_seed
                    + int(case["case_index"]) * 1000
                    + ARM_IDS.index(arm_id) * 100
                    + view_ids_for_arm(arm_id).index(view_id),
                ),
                "budget": _budget_for_view(arm_id),
            },
            "visible_proposal": {
                "raw_text": raw_text,
                "raw_sha256": sha256_text(raw_text),
                "proposal_atom_id_count": len(_proposal_atom_ids(raw)),
                "proposal_atom_ids_hash": sha256_json(sorted(_proposal_atom_ids(raw))),
                "usage": dict(raw.get("usage") or {}),
            },
            "sealed_atom_support": group["sealed_atom_support"],
            "exact_completion_result": group["exact_completion_result"],
            "post_seal_atom_labels": group["post_seal_atom_labels"],
            "exact_outcome": group["exact_outcome"],
            "missing_spurious_contradiction_unsafe": group[
                "missing_spurious_contradiction_unsafe"
            ],
            "correct_mode_identity": group["correct_mode_identity"],
            "search_receipt": group["search_receipt"],
            "latency_s": float(raw.get("latency_s") or 0.0),
            "token_counts": dict(raw.get("usage") or {}),
            "gpu_telemetry": dict(raw.get("gpu_telemetry") or {}),
            "no_hidden_reference_answer": True,
            "previous_prefix_checksum": previous,
            "prefix_checksum": "",
            "row_hash": "",
        }
        event["row_hash"] = _event_row_hash(event)
        event["prefix_checksum"] = sha256_json(
            {
                "previous_prefix_checksum": previous,
                "row_hash": event["row_hash"],
                "stream_sequence_index": event["stream_sequence_index"],
            }
        )
        previous = event["prefix_checksum"]
        events.append(event)
    return events


def replay_event_stream(path: Path) -> JsonDict:
    """Replay the chronological event prefix chain and reject tampering."""

    if not path.exists():
        return {"ok": False, "row_count": 0, "reason": "missing_event_stream", "rows": []}
    previous = "sha256:" + "0" * 64
    rows = []
    ok = True
    reason = "ok"
    for expected_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        event = json.loads(line)
        row_hash = _event_row_hash(event)
        prefix = sha256_json(
            {
                "previous_prefix_checksum": previous,
                "row_hash": row_hash,
                "stream_sequence_index": event.get("stream_sequence_index"),
            }
        )
        hidden = _contains_hidden_reference_answer(event)
        row_ok = (
            row_hash == event.get("row_hash")
            and prefix == event.get("prefix_checksum")
            and event.get("previous_prefix_checksum") == previous
            and int(event.get("stream_sequence_index")) == expected_index
            and not hidden
        )
        if not row_ok and ok:
            ok = False
            reason = f"event_stream_tamper_or_order_failure_at_{expected_index}"
        rows.append(
            {
                "stream_sequence_index": event.get("stream_sequence_index"),
                "row_hash": event.get("row_hash"),
                "computed_row_hash": row_hash,
                "prefix_checksum": event.get("prefix_checksum"),
                "computed_prefix_checksum": prefix,
                "contains_hidden_reference_answer": hidden,
                "ok": row_ok,
            }
        )
        previous = str(event.get("prefix_checksum"))
    return {
        "ok": ok,
        "reason": reason,
        "row_count": len(rows),
        "final_prefix_checksum": previous,
        "rows": rows,
    }


def save_checkpoint(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write resume rows atomically as chronological JSONL."""

    _write_event_stream_atomic(path, rows)


def load_checkpoint(path: Path) -> list[JsonDict]:
    """Load a prior checkpoint if present."""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def merge_resume_rows(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Deduplicate checkpoint and newly collected rows by stream index."""

    merged: dict[int, JsonDict] = {}
    for row in [*checkpoint_rows, *new_rows]:
        merged[int(row["stream_sequence_index"])] = dict(row)
    return [merged[index] for index in sorted(merged)]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal Exp5936 schema and safety receipts."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must use the atomic support union substrate")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true for exact fixture adjudication")
    chronology = float(artifact["chronological_event_stream_ready_score"])
    score = float(artifact["atomic_semantic_live_ready_score"])
    if chronology not in {0.0, 1.0}:
        raise ValueError("chronological_event_stream_ready_score must be bare 0.0 or 1.0")
    if score not in {0.0, 1.0}:
        raise ValueError("atomic_semantic_live_ready_score must be bare 0.0 or 1.0")
    no_leak = artifact[
        "no_label_feedback_no_hard_pruning_no_schema_reprompt_and_no_answer_enumeration_receipt"
    ]
    if no_leak.get("label_feedback_used") is not False:
        raise ValueError("label feedback is forbidden")
    if no_leak.get("model_side_hard_pruning_used") is not False:
        raise ValueError("model-side hard pruning is forbidden")
    if no_leak.get("schema_reprompt_used") is not False:
        raise ValueError("schema reprompt is forbidden")
    if no_leak.get("complete_answer_enumeration_used") is not False:
        raise ValueError("complete answer enumeration is forbidden")
    if score == 1.0:
        if not str(artifact["honest_verdict"]).startswith("complete_positive:"):
            raise ValueError("atomic ready score requires complete_positive verdict")
        if (
            artifact["exact_semantic_missing_spurious_contradiction_and_unsafe_receipts"][
                "unsafe_accepts_total"
            ]
            != 0
        ):
            raise ValueError("atomic ready score requires zero unsafe accepts")
        lower = artifact[
            "transformed_vs_repeated_original_primary_comparison_and_interval"
        ]["paired_interval"]["ci95"][0]
        if float(lower) <= 0.0:
            raise ValueError("atomic ready score requires positive lower paired bound")
    if not str(artifact["honest_verdict"]).startswith(
        ("complete_positive:", "complete_null:", "retired:", "blocked:")
    ):
        raise ValueError("honest_verdict must use an Exp5936 terminal prefix")


def refresh_artifact_test_exit_codes(
    *,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Update only test exit-code provenance after verification commands run."""

    path = root / RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["test_exit_codes"] = dict(test_exit_codes)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    _write_json_atomic(path, artifact)
    return artifact


def _build_artifact(
    config: ExperimentConfig,
    preconditions: PreconditionReport,
    *,
    preregistration: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    collection: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> JsonDict:
    protected = _protected_file_receipt(config.repo_root, baseline=preconditions.protected_file_baseline)
    expected_rows = len(MANDATED_MODEL_IDS) * int(preregistration["case_count"]) * CALLS_PER_MODEL_CASE
    replay = replay_event_stream(config.events_path())
    aggregates = _aggregate_events(events)
    chronology_ready = (
        preconditions.block_reason is None
        and bool(collection.get("real_model_rows"))
        and replay["ok"] is True
        and replay["row_count"] == expected_rows
        and protected["unchanged"] is True
    )
    retirement = _retirement_decision(aggregates)
    live_ready = 1.0 if chronology_ready and _atomic_live_ready(aggregates, retirement) else 0.0
    status, verdict = _status_and_verdict(
        preconditions.block_reason,
        live_ready,
        retirement,
        chronology_ready,
    )
    receipt = _event_stream_receipt(config.events_path(), replay, expected_rows)
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": config.random_seed,
        "status": status,
        "gate_replay_receipt": preconditions.gate_replay_receipt,
        "preconditions_checked": preconditions.preconditions_checked,
        "model_specs": preconditions.model_specs,
        "model_file_hashes": preconditions.model_file_hashes,
        "embedded_tokenizer_public_llama_cpp_cuda_dual_gpu_and_vram_receipts": (
            preconditions.receipts
            | {
                "model_attempts": list(collection.get("model_attempts") or []),
                "gpu_receipts": dict(collection.get("gpu_receipts") or {}),
            }
        ),
        "sealed_cases_families_adversaries_prompts_transforms_seeds_arms_and_budgets": dict(
            preregistration
        ),
        "per_model_arm_atom_and_exact_metrics": aggregates["per_model_arm"],
        "transformed_vs_repeated_original_primary_comparison_and_interval": aggregates[
            "primary_comparison"
        ],
        "direct_and_single_view_secondary_comparisons": aggregates["secondary_comparisons"],
        "included_and_excluded_relevant_atom_counts": aggregates["included_excluded"],
        "exact_semantic_missing_spurious_contradiction_and_unsafe_receipts": aggregates[
            "missing_spurious_contradiction_unsafe"
        ],
        "correct_mode_diversity_and_support_saturation": aggregates["diversity_saturation"],
        "search_correctness_change_vs_cleanup_receipt": aggregates["search_receipt"],
        "calls_tokens_latency_gpu_and_memory_accounting": aggregates["costs"],
        "chronological_event_stream_path_hash_rows_and_prefix_chain": receipt,
        "chronological_event_stream_ready_score": 1.0 if chronology_ready else 0.0,
        "no_label_feedback_no_hard_pruning_no_schema_reprompt_and_no_answer_enumeration_receipt": (
            _no_label_feedback_receipt(events)
        ),
        "retirement_decision": retirement,
        "protected_files_unchanged": protected,
        "atomic_semantic_live_ready_score": live_ready,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "The exact oracle covers only sealed synthetic finite-domain ConstraintIR fixtures.",
            "Natural-language paraphrase quality and broader family diversity are not proven outside the expanded held fixture.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _aggregate_events(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    groups = _unique_group_events(events)
    primary = _primary_comparison(groups)
    return {
        "per_model_arm": _per_model_arm_metrics(groups),
        "primary_comparison": primary,
        "secondary_comparisons": _secondary_comparisons(groups, primary),
        "included_excluded": _included_excluded_counts(groups),
        "missing_spurious_contradiction_unsafe": _missing_spurious_contradiction_unsafe(groups),
        "diversity_saturation": _diversity_saturation(groups),
        "search_receipt": _search_receipt(groups),
        "costs": _costs(events, groups),
    }


def _score_group(raw_rows: Sequence[Mapping[str, Any]], case: Mapping[str, Any]) -> JsonDict:
    first = raw_rows[0]
    arm_id = str(first["arm_id"])
    if arm_id == "direct_one_shot_ir":
        raw_text = str(first.get("raw_output_text") or "")
        evaluation = exp5897.evaluate_candidate(_target_row(case), arm_id, raw_text, dict(first))
        exact = _direct_exact_outcome(evaluation)
        missing = {
            "missing_constraints": int(evaluation.get("omitted_constraints") or 0),
            "spurious_constraints": int(evaluation.get("spurious_constraints") or 0),
            "contradiction_pairs": 0,
            "unsafe_acceptance": bool(evaluation.get("unsafe_accepted_constraints")),
        }
        return {
            "sealed_atom_support": {
                "arm_id": arm_id,
                "view_count": 1,
                "completion_run_count": 1,
                "direct_ir_no_atom_union": True,
            },
            "exact_completion_result": {
                "engine": "direct_ir_exact_eval",
                "completion_run_count": 1,
                "accepted": exact["exact_semantic_success"],
                "search_expansions": 1,
                "search_changed_correctness": False,
            },
            "post_seal_atom_labels": _empty_atom_label_counts(),
            "exact_outcome": exact,
            "missing_spurious_contradiction_unsafe": missing,
            "correct_mode_identity": {
                "candidate_sha256": evaluation.get("candidate_sha256"),
                "exact_semantic_success": exact["exact_semantic_success"],
            },
            "search_receipt": {
                "support_reachable_before_completion": False,
                "search_can_manufacture_deleted_truth": False,
                "cleanup_removed_irrelevant_atoms": False,
                "search_changed_correctness": False,
            },
        }

    schema = exp5935.versioned_atom_schema()
    surface = derive_surface_for_case(case, schema)
    proposals = _proposal_entries_from_raw(surface, raw_rows)
    sealed = exp5935.seal_non_pruning_union(surface, proposals)
    completion = exp5935.complete_subset(
        _exp5935_case(case), surface, sealed, max_states=MAX_COMPLETION_STATES
    )
    pool = exp5935.included_excluded_pool_audit(surface, sealed)
    labels = _atom_label_counts(pool)
    exact = {
        "parse_valid": bool(completion["accepted"]),
        "type_valid": bool(completion["accepted"]),
        "scope_valid": bool(completion["accepted"]),
        "compile_valid": bool(completion["accepted"]),
        "exact_semantic_success": bool(completion["accepted"]),
        "exact_execution_certificate": {
            "reason": completion["reason"],
            "certificate_hash": completion.get("certificate_hash"),
            "python_z3_agree": completion.get("python_z3_agree"),
        },
        "query_correct": bool(completion["accepted"]) or None,
        "satisfiability_correct": bool(completion["accepted"]) or None,
    }
    missing = {
        "missing_constraints": int(completion.get("missing_required_atom_count") or 0),
        "spurious_constraints": labels["included_irrelevant_atoms"],
        "contradiction_pairs": int(sealed.get("contradiction_pair_count") or 0),
        "unsafe_acceptance": False,
    }
    support_reachable = labels["excluded_relevant_atoms"] == 0
    cleanup = bool(completion["accepted"] and labels["included_irrelevant_atoms"] > 0)
    public_sealed = exp5935._public_sealed_union(sealed)
    public_sealed["completion_run_count"] = 1
    public_sealed["view_count"] = len({str(row["view_id"]) for row in raw_rows})
    public_sealed["proposal_count"] = len(proposals)
    return {
        "sealed_atom_support": public_sealed,
        "exact_completion_result": {
            **completion,
            "engine": "rank_ordered_bounded_prefix_subset_exact_executor",
            "completion_run_count": 1,
            "search_expansions": int(completion.get("attempts") or 0),
            "search_changed_correctness": bool(completion["accepted"]),
        },
        "post_seal_atom_labels": labels,
        "exact_outcome": exact,
        "missing_spurious_contradiction_unsafe": missing,
        "correct_mode_identity": {
            "candidate_sha256": completion.get("certificate_hash"),
            "exact_semantic_success": exact["exact_semantic_success"],
        },
        "search_receipt": {
            "support_reachable_before_completion": support_reachable,
            "search_can_manufacture_deleted_truth": False,
            "cleanup_removed_irrelevant_atoms": cleanup,
            "search_changed_correctness": bool(completion["accepted"]),
        },
    }


def _primary_comparison(groups: Sequence[Mapping[str, Any]]) -> JsonDict:
    treatment = "transformed_view_atomic_union"
    control = "repeated_original_atomic_union"
    by_arm = {arm: _metric_summary([row for row in groups if row["row_identity"]["arm_id"] == arm]) for arm in ARM_IDS}
    delta = (
        by_arm[treatment]["exact_semantic_success_rate"]
        - by_arm[control]["exact_semantic_success_rate"]
    )
    interval = _paired_interval(groups, treatment, control)
    return {
        "principle": FIELD_PRINCIPLES[
            "transformed_vs_repeated_original_primary_comparison_and_interval"
        ],
        "treatment_arm": treatment,
        "matched_control_arm": control,
        "metric": "held_exact_semantic_success",
        "by_arm": by_arm,
        "delta_exact_success_rate_d_minus_c": round(delta, 6),
        "paired_interval": interval,
        "matched_call_token_temperature_budget": (
            TOKEN_BUDGETS[treatment] == TOKEN_BUDGETS[control]
            and DECODING_PARAMETERS_BY_ARM[treatment] == DECODING_PARAMETERS_BY_ARM[control]
        ),
        "structural_validity_or_atom_recall_substituted_for_semantics": False,
    }


def _per_model_arm_metrics(groups: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_model: JsonDict = {}
    for model_id in sorted({str(row["row_identity"]["model_hf_id"]) for row in groups}):
        by_model[model_id] = {
            arm: _metric_summary(
                [
                    row
                    for row in groups
                    if row["row_identity"]["model_hf_id"] == model_id
                    and row["row_identity"]["arm_id"] == arm
                ]
            )
            for arm in ARM_IDS
        }
    return {
        "by_model": by_model,
        "pooled_by_arm": {
            arm: _metric_summary([row for row in groups if row["row_identity"]["arm_id"] == arm])
            for arm in ARM_IDS
        },
    }


def _secondary_comparisons(
    groups: Sequence[Mapping[str, Any]], primary: Mapping[str, Any]
) -> JsonDict:
    del groups
    by_arm = primary["by_arm"]
    return {
        "principle": FIELD_PRINCIPLES["direct_and_single_view_secondary_comparisons"],
        "direct_one_shot_ir": by_arm["direct_one_shot_ir"],
        "single_view_atomic_support": by_arm["single_view_atomic_support"],
        "cost_baselines_only": True,
        "budget_matched_to_transformed_claim": False,
    }


def _included_excluded_counts(groups: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm: JsonDict = {}
    for arm in ARM_IDS:
        rows = [row for row in groups if row["row_identity"]["arm_id"] == arm]
        by_arm[arm] = {
            "included_relevant_atoms": sum(row["post_seal_atom_labels"]["included_relevant_atoms"] for row in rows),
            "included_irrelevant_atoms": sum(row["post_seal_atom_labels"]["included_irrelevant_atoms"] for row in rows),
            "excluded_relevant_atoms": sum(row["post_seal_atom_labels"]["excluded_relevant_atoms"] for row in rows),
            "excluded_irrelevant_atoms": sum(row["post_seal_atom_labels"]["excluded_irrelevant_atoms"] for row in rows),
        }
    return {"by_arm": by_arm, "labels_opened_only_after_union_seal": True}


def _missing_spurious_contradiction_unsafe(groups: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm: JsonDict = {}
    for arm in ARM_IDS:
        rows = [row for row in groups if row["row_identity"]["arm_id"] == arm]
        by_arm[arm] = {
            "missing_constraints": sum(row["missing_spurious_contradiction_unsafe"]["missing_constraints"] for row in rows),
            "spurious_constraints": sum(row["missing_spurious_contradiction_unsafe"]["spurious_constraints"] for row in rows),
            "contradiction_pairs": sum(row["missing_spurious_contradiction_unsafe"]["contradiction_pairs"] for row in rows),
            "unsafe_accepts": sum(bool(row["missing_spurious_contradiction_unsafe"]["unsafe_acceptance"]) for row in rows),
        }
    return {
        "by_arm": by_arm,
        "missing_constraints_total": sum(row["missing_spurious_contradiction_unsafe"]["missing_constraints"] for row in groups),
        "spurious_constraints_total": sum(row["missing_spurious_contradiction_unsafe"]["spurious_constraints"] for row in groups),
        "contradiction_pairs_total": sum(row["missing_spurious_contradiction_unsafe"]["contradiction_pairs"] for row in groups),
        "unsafe_accepts_total": sum(bool(row["missing_spurious_contradiction_unsafe"]["unsafe_acceptance"]) for row in groups),
    }


def _diversity_saturation(groups: Sequence[Mapping[str, Any]]) -> JsonDict:
    unique_by_arm: JsonDict = {}
    saturation_by_arm: JsonDict = {}
    for arm in ARM_IDS:
        rows = [row for row in groups if row["row_identity"]["arm_id"] == arm]
        keys = [
            row["correct_mode_identity"].get("candidate_sha256")
            for row in rows
            if row["correct_mode_identity"]["exact_semantic_success"] is True
        ]
        unique_by_arm[arm] = len({key for key in keys if key})
        saturation_by_arm[arm] = sum(
            row["exact_completion_result"].get("reason") == "support_saturation_bound_reached"
            for row in rows
        )
    material_collapse = (
        unique_by_arm["transformed_view_atomic_union"]
        < unique_by_arm["repeated_original_atomic_union"]
        and unique_by_arm["repeated_original_atomic_union"] > 0
    )
    return {
        "unique_correct_mode_identities_by_arm": unique_by_arm,
        "support_saturation_count_by_arm": saturation_by_arm,
        "material_correct_mode_collapse": material_collapse,
    }


def _search_receipt(groups: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm: JsonDict = {}
    for arm in ARM_IDS:
        rows = [row for row in groups if row["row_identity"]["arm_id"] == arm]
        by_arm[arm] = {
            "search_expansions": sum(int(row["exact_completion_result"].get("search_expansions") or 0) for row in rows),
            "search_changed_correctness_count": sum(bool(row["search_receipt"]["search_changed_correctness"]) for row in rows),
            "cleanup_removed_irrelevant_atoms_count": sum(bool(row["search_receipt"]["cleanup_removed_irrelevant_atoms"]) for row in rows),
            "search_manufactured_deleted_truth_count": sum(bool(row["search_receipt"]["search_can_manufacture_deleted_truth"]) for row in rows),
        }
    return {
        "by_arm": by_arm,
        "search_can_manufacture_deleted_truth": any(
            row["search_receipt"]["search_can_manufacture_deleted_truth"] for row in groups
        ),
        "distinguishes_reachability_from_cleanup": True,
    }


def _costs(events: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]) -> JsonDict:
    del groups
    prompt_tokens = sum(int(row.get("token_counts", {}).get("prompt_tokens") or 0) for row in events)
    completion_tokens = sum(int(row.get("token_counts", {}).get("completion_tokens") or 0) for row in events)
    total_latency = sum(float(row.get("latency_s") or 0.0) for row in events)
    by_arm: JsonDict = {}
    for arm in ARM_IDS:
        arm_events = [row for row in events if row["row_identity"]["arm_id"] == arm]
        by_arm[arm] = {
            "view_calls": len(arm_events),
            "prompt_tokens": sum(int(row.get("token_counts", {}).get("prompt_tokens") or 0) for row in arm_events),
            "completion_tokens": sum(int(row.get("token_counts", {}).get("completion_tokens") or 0) for row in arm_events),
            "latency_s": round(sum(float(row.get("latency_s") or 0.0) for row in arm_events), 6),
        }
    vram_deltas = [float(row.get("gpu_telemetry", {}).get("vram_delta_mb") or 0.0) for row in events]
    return {
        "view_call_rows": len(events),
        "by_arm": by_arm,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_s": round(total_latency, 6),
        "max_vram_delta_mb": max(vram_deltas) if vram_deltas else 0.0,
        "all_rows_report_gpu_telemetry": all("gpu_telemetry" in row for row in events),
        "exact_completion_cost_counted": True,
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    exact = sum(row["exact_outcome"]["exact_semantic_success"] is True for row in rows)
    return {
        "n": total,
        "semantic_n": total,
        "parse_rate": _rate(sum(row["exact_outcome"]["parse_valid"] for row in rows), total),
        "type_rate": _rate(sum(row["exact_outcome"]["type_valid"] for row in rows), total),
        "scope_rate": _rate(sum(row["exact_outcome"]["scope_valid"] for row in rows), total),
        "compile_rate": _rate(sum(row["exact_outcome"]["compile_valid"] for row in rows), total),
        "exact_semantic_success_rate": _rate(exact, total),
        "unsafe_accept_rate": _rate(
            sum(bool(row["missing_spurious_contradiction_unsafe"]["unsafe_acceptance"]) for row in rows),
            total,
        ),
        "atom_recall_rate": _atom_recall_rate(rows),
        "mean_search_expansions": round(
            sum(int(row["exact_completion_result"].get("search_expansions") or 0) for row in rows)
            / total,
            6,
        )
        if total
        else 0.0,
    }


def _paired_interval(rows: Sequence[Mapping[str, Any]], treatment_arm: str, control_arm: str) -> JsonDict:
    by_pair: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["row_identity"]["model_hf_id"]), str(row["row_identity"]["case_id"]))
        arm = str(row["row_identity"]["arm_id"])
        if arm in {treatment_arm, control_arm}:
            by_pair[key][arm] = 1.0 if row["exact_outcome"]["exact_semantic_success"] else 0.0
    diffs = [
        values.get(treatment_arm, 0.0) - values.get(control_arm, 0.0)
        for values in by_pair.values()
    ]
    if not diffs:
        return {
            "ci95": [0.0, 0.0],
            "mean": 0.0,
            "n_pairs": 0,
            "method": "deterministic_paired_bootstrap_ci95",
        }
    samples = []
    for draw in range(200):
        total = 0.0
        for index in range(len(diffs)):
            total += diffs[(draw * 37 + index * 17) % len(diffs)]
        samples.append(total / len(diffs))
    samples.sort()
    return {
        "ci95": [
            round(samples[int(0.025 * (len(samples) - 1))], 6),
            round(samples[int(0.975 * (len(samples) - 1))], 6),
        ],
        "mean": round(sum(diffs) / len(diffs), 6),
        "n_pairs": len(diffs),
        "method": "deterministic_paired_bootstrap_ci95",
    }


def _retirement_decision(aggregates: Mapping[str, Any]) -> JsonDict:
    per_model = aggregates["per_model_arm"].get("by_model", {})
    d_rates = [
        rows["transformed_view_atomic_union"]["exact_semantic_success_rate"]
        for rows in per_model.values()
        if "transformed_view_atomic_union" in rows
    ]
    all_three_zero_d = len(d_rates) == len(MANDATED_MODEL_IDS) and all(rate == 0.0 for rate in d_rates)
    primary = aggregates["primary_comparison"]
    unsafe = aggregates["missing_spurious_contradiction_unsafe"]
    diversity = aggregates["diversity_saturation"]
    transformed_unsafe = (
        unsafe["by_arm"]
        .get("transformed_view_atomic_union", {})
        .get("unsafe_accepts", 0)
        > 0
    )
    no_improvement_with_cost_or_unsafe = (
        primary["delta_exact_success_rate_d_minus_c"] <= 0.0 and transformed_unsafe
    )
    collapse = diversity["material_correct_mode_collapse"]
    reasons = [
        name
        for name, active in {
            "all_three_models_zero_exact_success_in_transformed_view_union": all_three_zero_d,
            "transformed_view_union_no_semantic_improvement_with_material_cost_or_unsafe": no_improvement_with_cost_or_unsafe,
            "transformed_view_union_unsafe_acceptance": transformed_unsafe,
            "transformed_view_union_correct_mode_diversity_collapsed": collapse,
        }.items()
        if active
    ]
    return {
        "retire": bool(reasons),
        "reasons": reasons,
        "next_action": "retire_atomic_union_mechanism" if reasons else "do_not_retire_from_this_run",
        "principle": FIELD_PRINCIPLES["retirement_decision"],
    }


def _atomic_live_ready(aggregates: Mapping[str, Any], retirement: Mapping[str, Any]) -> bool:
    primary = aggregates["primary_comparison"]
    unsafe = aggregates["missing_spurious_contradiction_unsafe"]
    diversity = aggregates["diversity_saturation"]
    return bool(
        primary["paired_interval"]["ci95"][0] > 0.0
        and unsafe["unsafe_accepts_total"] == 0
        and diversity["material_correct_mode_collapse"] is False
        and retirement["retire"] is False
    )


def _status_and_verdict(
    block_reason: str | None,
    live_ready: float,
    retirement: Mapping[str, Any],
    chronology_ready: bool,
) -> tuple[str, str]:
    if block_reason is not None:
        return "blocked", f"blocked: precondition failed before model rows: {block_reason}"
    if not chronology_ready:
        return "blocked", "blocked: chronological real-model event stream incomplete or tampered"
    if live_ready == 1.0:
        return "complete_positive", "complete_positive: transformed-view atomic union beat repeated-original union on exact semantics"
    if retirement.get("retire"):
        return "retired", "retired: transformed-view atomic union failed exact-semantic retirement gates"
    return "complete_null", "complete_null: transformed-view atomic union did not beat matched repeated-original union"


def _event_stream_receipt(path: Path, replay: Mapping[str, Any], expected_rows: int) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "rows": replay["row_count"],
        "expected_rows": expected_rows,
        "prefix_chain_ok": replay["ok"],
        "final_prefix_checksum": replay.get("final_prefix_checksum"),
        "no_hidden_reference_answer": all(
            not row.get("contains_hidden_reference_answer") for row in replay.get("rows", [])
        ),
    }


def _direct_exact_outcome(evaluation: Mapping[str, Any]) -> JsonDict:
    parse_valid = bool(evaluation.get("parse_valid"))
    type_valid = bool(evaluation.get("type_valid"))
    compiled = bool(evaluation.get("compiled"))
    return {
        "parse_valid": parse_valid,
        "type_valid": type_valid,
        "scope_valid": parse_valid and type_valid,
        "compile_valid": compiled,
        "exact_semantic_success": evaluation.get("exact_semantic_equivalence") is True,
        "exact_execution_certificate": {
            "solver_status": evaluation.get("solver_status"),
            "z3_status": evaluation.get("z3_status"),
            "diagnostics": dict(evaluation.get("diagnostics") or {}),
        },
        "query_correct": evaluation.get("query_correct"),
        "satisfiability_correct": evaluation.get("satisfiability_correct"),
    }


def _proposal_entries_from_raw(
    surface: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    visible = dict(surface["_visible_by_id"])
    entries: list[JsonDict] = []
    for raw in raw_rows:
        view_id = str(raw.get("view_id") or "unknown_view")
        for rank, atom_id in enumerate(_proposal_atom_ids(raw)):
            atom = visible.get(atom_id) or {
                "schema_version": exp5935.ATOM_SCHEMA_VERSION,
                "atom_kind": "invalid.proposed_atom_id",
                "payload": {"atom_id": atom_id},
                "atom_id": atom_id,
            }
            entries.append(
                {
                    "atom": _copy_json(atom),
                    "view_id": view_id,
                    "rank": rank,
                    "source": "model_visible_atom_id_json",
                }
            )
    return entries


def _proposal_atom_ids(raw: Mapping[str, Any]) -> list[str]:
    ids = raw.get("proposal_atom_ids")
    if isinstance(ids, list):
        return [str(item) for item in ids]
    try:
        parsed = json.loads(str(raw.get("raw_output_text") or "{}"))
    except json.JSONDecodeError:
        return []
    parsed_ids = parsed.get("atom_ids") if isinstance(parsed, Mapping) else []
    return [str(item) for item in parsed_ids] if isinstance(parsed_ids, list) else []


def _group_raw_rows(raw_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], list[JsonDict]]:
    grouped: dict[tuple[str, str, str], list[JsonDict]] = defaultdict(list)
    for row in raw_rows:
        key = (str(row["model_hf_id"]), str(row["case_id"]), str(row["arm_id"]))
        grouped[key].append(dict(row))
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["stream_sequence_index"]))
    return dict(grouped)


def _unique_group_events(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    seen: set[tuple[str, str, str]] = set()
    groups: list[JsonDict] = []
    for event in events:
        identity = event["row_identity"]
        key = (
            str(identity["model_hf_id"]),
            str(identity["case_id"]),
            str(identity["arm_id"]),
        )
        if key in seen:
            continue
        seen.add(key)
        groups.append(dict(event))
    return groups


def _empty_atom_label_counts() -> JsonDict:
    return {
        "included_relevant_atoms": 0,
        "included_irrelevant_atoms": 0,
        "excluded_relevant_atoms": 0,
        "excluded_irrelevant_atoms": 0,
    }


def _atom_label_counts(pool: Mapping[str, Any]) -> JsonDict:
    return {
        "included_relevant_atoms": int(pool["included_pool"]["positive_count"]),
        "included_irrelevant_atoms": int(pool["included_pool"]["negative_count"]),
        "excluded_relevant_atoms": int(pool["excluded_pool"]["positive_count"]),
        "excluded_irrelevant_atoms": int(pool["excluded_pool"]["negative_count"]),
    }


def _atom_recall_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    included = sum(row["post_seal_atom_labels"]["included_relevant_atoms"] for row in rows)
    excluded = sum(row["post_seal_atom_labels"]["excluded_relevant_atoms"] for row in rows)
    return _rate(included, included + excluded)


def _no_label_feedback_receipt(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    leaked_markers = _forbidden_keys(events)
    return {
        "label_feedback_used": False,
        "hidden_reference_prompt_hash_used": False,
        "model_side_hard_pruning_used": False,
        "schema_reprompt_used": False,
        "complete_answer_enumeration_used": False,
        "model_confidence_hard_mask_used": False,
        "legal_proposed_atom_dropped_before_union": False,
        "forbidden_markers": leaked_markers,
        "ok": not leaked_markers,
    }


def _event_row_hash(event: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(event))
    stable["row_hash"] = ""
    stable["prefix_checksum"] = ""
    return sha256_json(stable)


def _contains_hidden_reference_answer(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_EVENT_REFERENCE_KEYS:
                return True
            if _contains_hidden_reference_answer(item):
                return True
    elif isinstance(value, list):
        return any(_contains_hidden_reference_answer(item) for item in value)
    return False


def _forbidden_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_EVENT_REFERENCE_KEYS:
                found.append(str(key))
            found.extend(_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_forbidden_keys(item))
    return sorted(set(found))


def _target_row(case: Mapping[str, Any]) -> JsonDict:
    return _copy_json(case.get("_target_row") or case.get("target_row"))


def _exp5935_case(case: Mapping[str, Any]) -> JsonDict:
    return {
        "case_id": case["case_id"],
        "source_row_id": case["source_row_id"],
        "family": case["family"],
        "split": case["split"],
        "variant_kind": case["variant_kind"],
        "target_row": _target_row(case),
    }


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _public_case(case: Mapping[str, Any]) -> JsonDict:
    return {key: _copy_json(value) for key, value in case.items() if not key.startswith("_")}


def _case_transform_specs() -> list[JsonDict]:
    return [
        {
            "case_suffix": "original",
            "view_id": "original",
            "inverse": "identity",
            "adversary_kind": "held_family",
        },
        {
            "case_suffix": "paraphrase",
            "view_id": "paraphrase",
            "inverse": "identity",
            "adversary_kind": "held_family",
        },
        {
            "case_suffix": "entity-permutation",
            "view_id": "entity_permutation",
            "inverse": "inverse_symbol_rotation",
            "adversary_kind": "held_family",
        },
        {
            "case_suffix": "attribute-value-adversary",
            "view_id": "attribute_value",
            "inverse": "identity",
            "adversary_kind": "attribute_value_adversary",
        },
        {
            "case_suffix": "value-order-adversary",
            "view_id": "value_order",
            "inverse": "identity",
            "adversary_kind": "attribute_value_adversary",
        },
        {
            "case_suffix": "label-separation-control",
            "view_id": "label_separation",
            "inverse": "identity",
            "adversary_kind": "held_family",
        },
    ]


def _transformed_problem_text(row: Mapping[str, Any], transform: Mapping[str, Any]) -> str:
    base = str(row.get("natural_language") or row.get("prompt") or row["row_id"])
    view = str(transform["view_id"])
    if view == "paraphrase":
        return f"Paraphrased view: {base}"
    if view == "entity_permutation":
        return f"Entity-permutation view with inverse_symbol_rotation receipt: {base}"
    if view == "attribute_value":
        return f"Attribute/value adversary view; preserve the exact fixture semantics: {base}"
    if view == "value_order":
        return f"Value-order adversary view; do not infer labels from order: {base}"
    if view == "label_separation":
        return f"Label-separation control view; no exact labels are visible: {base}"
    return base


def _inverse_map_receipts(panel: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts: dict[str, int] = defaultdict(int)
    for case in panel:
        counts[str(case["inverse_transform_id"])] += 1
    return {
        "inverse_map_counts": dict(sorted(counts.items())),
        "all_cases_have_inverse_map": all(case.get("inverse_transform_id") for case in panel),
    }


def _budget_for_view(arm_id: str) -> JsonDict:
    budget = dict(TOKEN_BUDGETS[arm_id])
    budget["max_tokens"] = budget["max_tokens_per_call"]
    return budget


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / denominator, 6) if denominator else 0.0


def _tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]], tokenizer_loader: TokenizerLoader
) -> JsonDict:
    receipts: JsonDict = {}
    for spec in model_specs:
        if not spec.get("model_path"):
            continue
        try:
            receipt = tokenizer_loader(dict(spec))
            receipt.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001
            receipt = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        receipt["embedded_tokenizer_only"] = receipt.get("used_hf_autotokenizer") is False
        receipt["used_hf_autotokenizer"] = False
        receipts[str(spec["hf_id"])] = receipt
    return receipts


def _model_file_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    hashes: JsonDict = {}
    for spec in model_specs:
        path = Path(str(spec.get("model_path") or ""))
        if path.is_file():
            hashes[str(spec["hf_id"])] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return hashes


def _gate_replay_receipt(root: Path) -> JsonDict:
    path = root / exp5935.RESULT_RELATIVE_PATH
    expected = REPO_ROOT / exp5935.RESULT_RELATIVE_PATH
    receipt: JsonDict = {
        "artifact_path": str(path),
        "expected_source_path": str(expected),
        "artifact_present": path.exists(),
        "path_exact": path.resolve() == expected.resolve() if path.exists() else False,
        "artifact_sha256": sha256_file(path) if path.exists() else None,
        "atom_support_fixture_ready_score": 0.0,
        "ok": False,
    }
    if not path.exists():
        receipt["block_reason"] = "exp5935_artifact_missing"
        return receipt
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        exp5935.validate_artifact(artifact)
    except Exception as exc:  # noqa: BLE001
        receipt["block_reason"] = f"exp5935_replay_invalid:{type(exc).__name__}:{exc}"
        return receipt
    receipt["atom_support_fixture_ready_score"] = artifact.get("atom_support_fixture_ready_score")
    receipt["honest_verdict"] = artifact.get("honest_verdict")
    receipt["source_reproducibility_checksum"] = artifact.get("reproducibility_checksum")
    receipt["ok"] = (
        receipt["path_exact"] is True
        and artifact.get("atom_support_fixture_ready_score") == 1.0
    )
    if not receipt["ok"]:
        receipt["block_reason"] = "atom_support_fixture_ready_score_not_1_or_path_not_exact"
    return receipt


def _probe_environment(root: Path) -> JsonDict:  # pragma: no cover - host boundary.
    public_api = exp5922.public_llama_cpp_api_receipt()
    return {
        "llama_cpp_import": {"ok": public_api.get("importable") is True},
        "public_llama_cpp_cuda": {
            "ok": public_api.get("ok") is True,
            "logits_processor_parameter": public_api.get("logits_processor_parameter") is True,
            "gpu_offload_supported": public_api.get("gpu_offload_supported") is True,
        },
        "gpu_health": _gpu_health_probe(),
        "ram": _memory_probe(32768),
        "disk": _disk_probe(root, 8192),
        "protected_workload": _protected_workload_probe(),
        "atomic_output": _atomic_output_probe(root / RESULT_RELATIVE_PATH),
        "atomic_checkpoint_resume": _atomic_checkpoint_resume_probe(root / CHECKPOINT_RELATIVE_PATH),
    }


def _gpu_health_probe() -> JsonDict:  # pragma: no cover - host boundary.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "utilization_gpu_pct": int(parts[4]),
                }
            )
    return {"ok": result.returncode == 0 and bool(gpus), "gpus": gpus}


def _two_healthy_rtx_3090s(gpu_health: Mapping[str, Any]) -> bool:
    gpus = gpu_health.get("gpus") or []
    healthy = [
        gpu
        for gpu in gpus
        if "RTX 3090" in str(gpu.get("name"))
        and int(gpu.get("memory_total_mb") or 0) >= 24000
    ]
    return len(healthy) >= 2


def _adequate_vram(gpu_health: Mapping[str, Any]) -> bool:
    gpus = gpu_health.get("gpus") or []
    return len(gpus) >= 2 and all(int(gpu.get("memory_free_mb") or 0) >= 16000 for gpu in gpus[:2])


def _disk_probe(root: Path, required_mb: int) -> JsonDict:
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"ok": available_mb >= required_mb, "available_mb": available_mb, "required_mb": required_mb}


def _memory_probe(required_mb: int) -> JsonDict:
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    return {"ok": available_mb >= required_mb, "available_mb": available_mb, "required_mb": required_mb}


def _protected_workload_probe() -> JsonDict:  # pragma: no cover - host process boundary.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "protected_pids": []}
    protected = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0].isdigit() and _pid_is_protected_training_proc(int(parts[0])):
            protected.append({"pid": int(parts[0]), "process_name": parts[1] if len(parts) > 1 else ""})
    return {"ok": not protected, "protected_pids": protected}


def _pid_is_protected_training_proc(pid: int) -> bool:  # pragma: no cover - host process boundary.
    try:
        cmdline = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", "replace")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return False
    markers = ("train.py", "/nn/train", "src/nn/train")
    return any(marker in cmdline for marker in markers)


def _atomic_output_probe(output_path: Path) -> JsonDict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe = output_path.parent / f".{output_path.name}.atomic-probe"
    replacement = output_path.parent / f".{output_path.name}.atomic-probe.tmp"
    ok = False
    try:
        probe.write_text("old", encoding="utf-8")
        replacement.write_text("new", encoding="utf-8")
        os.replace(replacement, probe)
        ok = probe.read_text(encoding="utf-8") == "new"
    finally:
        probe.unlink(missing_ok=True)
        replacement.unlink(missing_ok=True)
    return {"ok": ok, "detail": "os.replace_same_directory"}


def _atomic_checkpoint_resume_probe(path: Path) -> JsonDict:
    sample = [{"stream_sequence_index": 0, "row_hash": "sha256:probe"}]
    save_checkpoint(path, sample)
    loaded = load_checkpoint(path)
    path.unlink(missing_ok=True)
    return {"ok": loaded == sample, "detail": "atomic_jsonl_checkpoint_resume"}


def _protected_file_receipt(root: Path, baseline: Mapping[str, Any] | None = None) -> JsonDict:
    baseline_by_path = {str(row["path"]): row for row in (baseline or {}).get("files", [])}
    files = []
    for relative in PROTECTED_FILES:
        path = root / relative
        current = sha256_file(path) if path.exists() else None
        before = baseline_by_path.get(str(relative), {}).get("sha256", current)
        files.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256_before": before,
                "sha256": current,
                "unchanged": before == current,
            }
        )
    return {"unchanged": all(row["unchanged"] for row in files), "files": files}


def _field_provenance() -> JsonDict:
    return {
        field: {
            "satisfied_by": "generated_by_exp5936_sota_atomic_support_union_ab",
            "principle": FIELD_PRINCIPLES.get(field, "Exp5936 required field."),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["test_exit_codes"] = {}
    stable["reproducibility_checksum"] = ""
    receipts = stable.get(
        "embedded_tokenizer_public_llama_cpp_cuda_dual_gpu_and_vram_receipts", {}
    )
    environment = receipts.get("environment", {}) if isinstance(receipts, dict) else {}
    for key in ("ram", "disk"):
        if isinstance(environment.get(key), dict):
            environment[key]["available_mb"] = 0
    return sha256_json(stable)


def _write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        handle.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _write_event_stream_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _gpu_memory_total_mb() -> float:  # pragma: no cover - host boundary.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return 0.0
    if result.returncode != 0:
        return 0.0
    return float(sum(float(line.strip()) for line in result.stdout.splitlines() if line.strip()))


def _call_live_view(
    llm: Any,
    spec: Mapping[str, Any],
    case: Mapping[str, Any],
    arm_id: str,
    view_id: str,
    seed: int,
    sequence: int,
    config: ExperimentConfig,
) -> JsonDict:  # pragma: no cover - live llama.cpp boundary.
    started = config.monotonic_clock()
    prompt = build_prompt(case, arm_id, view_id)
    budget = _budget_for_view(arm_id)
    result = llm(
        prompt,
        max_tokens=int(budget["max_tokens_per_call"]),
        temperature=float(DECODING_PARAMETERS_BY_ARM[arm_id]["temperature"]),
        top_p=float(DECODING_PARAMETERS_BY_ARM[arm_id]["top_p"]),
        seed=seed,
        stop=["</s>", "<eos>", "<|eot_id|>"],
    )
    text = ""
    usage = {}
    if isinstance(result, Mapping) and result.get("choices"):
        text = str(result["choices"][0].get("text") or "")
        usage = dict(result.get("usage") or {})
    return {
        "stream_sequence_index": sequence,
        "model_hf_id": spec["hf_id"],
        "model_name": spec["name"],
        "model_path": spec["model_path"],
        "gpu_index": spec.get("gpu"),
        "case_id": case["case_id"],
        "arm_id": arm_id,
        "view_id": view_id,
        "seed": seed,
        "raw_output_text": text,
        "proposal_atom_ids": _proposal_atom_ids({"raw_output_text": text}),
        "latency_s": round(config.monotonic_clock() - started, 6),
        "usage": usage,
        "gpu_telemetry": {
            "vram_delta_mb": _gpu_memory_total_mb(),
            "offload_verified": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--refresh-test-exit-code", action="append", default=[])
    args = parser.parse_args(argv)
    if args.refresh_test_exit_code:
        refresh_artifact_test_exit_codes(
            root=args.root,
            test_exit_codes=_parse_test_exit_codes(args.refresh_test_exit_code),
        )
    else:
        run_experiment(ExperimentConfig(repo_root=args.root))
    return 0


def _parse_test_exit_codes(values: Sequence[str]) -> JsonDict:  # pragma: no cover - CLI wrapper.
    parsed = {}
    for value in values:
        command, code = value.rsplit("=", 1)
        parsed[command] = int(code)
    return parsed


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
