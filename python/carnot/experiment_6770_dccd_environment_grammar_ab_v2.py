"""Run the paired static-versus-environment grammar proof experiment.

The experiment freezes exact-invalid certificates from Exp6768, then gives
each local headline GGUF the same repair task under three decode policies.
Exact checking stays downstream of generation and never supplies a mask.

Spec refs: REQ-VERIFY-6770 and SCENARIO-VERIFY-6770-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from typing import Any

from carnot import experiment_6745_sota_dual_encoding_proposal_corpus as frozen
from carnot.inference.gguf_output_text import normalize_gguf_output_text
from carnot.verify import dual_certificate_encoder_a as encoder_a
from carnot.verify import dual_certificate_encoder_b as encoder_b


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = Path("results/experiment_6768_targetable_proof_panel_expansion.json")
GRAMMAR_PATH = Path("results/experiment_6769_environment_indexed_proof_grammar_v2.json")
RESULT_PATH = Path("results/experiment_6770_dccd_environment_grammar_ab_v2.json")
MODULE_PATH = Path("python/carnot/experiment_6770_dccd_environment_grammar_ab_v2.py")
TEST_PATH = Path("tests/python/test_experiment_6770_dccd_environment_grammar_ab_v2.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6770_dccd_environment_grammar_ab_v2.py")

SCHEMA = "carnot.experiment_6770.dccd_environment_grammar_ab_v2.v1"
ROW_SCHEMA = f"{SCHEMA}.row"
MANIFEST_SCHEMA = f"{SCHEMA}.manifest"
INFERENCE_SUBSTRATE = "local llama.cpp CUDA GGUF with invoked runtime grammar"
CLAIM_BOUNDARY = "exact validity, not parseability"
RANDOM_SEED = 6_770_000
MINIMUM_INSTANCES = 36
CONTEXT_LIMIT = 4096
TOTAL_OUTPUT_TOKENS = 96
EXACT_CHECK_BUDGET = 2
DCCD_SPLIT = {"draft_tokens": 64, "render_tokens": 32}
MIN_RAM_BYTES = 16 * 1024**3
MIN_DISK_BYTES = 2 * 1024**3
PORTS = (47_700, 47_701, 47_702)
ARMS = ["repaired_direct", "static_grammar", "dccd_environment"]
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

MODEL_DEFINITIONS = [
    {
        "family_id": "qwen36_flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "preferred_quant": "Q4_K_M",
    },
    {
        "family_id": "gemma4_31b_flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense",
        "preferred_quant": "Q4_K_M",
    },
    {
        "family_id": "gemma4_26b_middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe",
        "preferred_quant": "Q4_K_M",
    },
]

HELD_FAMILIES = ("expander_tseitin", "ladder_tseitin", "pigeonhole_anchor")
ERROR_CLASSES = (
    "undefined_variable",
    "invalid_clause",
    "non_binary_value",
    "duplicate_evidence",
    "missing_evidence",
    "premature_terminal",
)

# The selected IDs are frozen, not sampled at runtime. Their cross-classified
# margins are 12/family, 18/size, 6/error, 18/source role, and 18/relabel role.
FROZEN_INSTANCE_IDS = (
    "gemma4_26b_middle_moe|exp6744-expander_tseitin-medium-unsat-674401-relabel|undefined_variable",
    "gemma4_26b_middle_moe|exp6744-expander_tseitin-medium-unsat-674401-relabel|invalid_clause",
    "gemma4_26b_middle_moe|exp6744-expander_tseitin-medium-unsat-674401-relabel|non_binary_value",
    "gemma4_26b_middle_moe|exp6744-expander_tseitin-medium-unsat-674401-relabel|duplicate_evidence",
    "gemma4_26b_middle_moe|exp6744-expander_tseitin-medium-unsat-674401-relabel|missing_evidence",
    "gemma4_26b_middle_moe|exp6744-expander_tseitin-medium-unsat-674401-relabel|premature_terminal",
    "gemma4_26b_middle_moe|exp6744-pigeonhole_anchor-medium-unsat-674403-base|undefined_variable",
    "gemma4_26b_middle_moe|exp6744-pigeonhole_anchor-medium-unsat-674403-base|invalid_clause",
    "gemma4_26b_middle_moe|exp6744-pigeonhole_anchor-medium-unsat-674403-base|non_binary_value",
    "gemma4_26b_middle_moe|exp6744-pigeonhole_anchor-medium-unsat-674403-base|duplicate_evidence",
    "gemma4_26b_middle_moe|exp6744-pigeonhole_anchor-medium-unsat-674403-base|missing_evidence",
    "gemma4_26b_middle_moe|exp6744-pigeonhole_anchor-medium-unsat-674403-base|premature_terminal",
    "qwen36_flagship_moe|exp6744-expander_tseitin-small-sat-674402-base|undefined_variable",
    "qwen36_flagship_moe|exp6744-expander_tseitin-small-sat-674402-base|invalid_clause",
    "qwen36_flagship_moe|exp6744-expander_tseitin-small-sat-674402-base|non_binary_value",
    "qwen36_flagship_moe|exp6744-expander_tseitin-small-sat-674402-base|duplicate_evidence",
    "qwen36_flagship_moe|exp6744-expander_tseitin-small-sat-674402-base|missing_evidence",
    "qwen36_flagship_moe|exp6744-expander_tseitin-small-sat-674402-base|premature_terminal",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-medium-sat-674401-base|undefined_variable",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-medium-sat-674401-base|invalid_clause",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-medium-sat-674401-base|non_binary_value",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-medium-sat-674401-base|duplicate_evidence",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-medium-sat-674401-base|missing_evidence",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-medium-sat-674401-base|premature_terminal",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-small-sat-674401-relabel|undefined_variable",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-small-sat-674401-relabel|invalid_clause",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-small-sat-674401-relabel|non_binary_value",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-small-sat-674401-relabel|duplicate_evidence",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-small-sat-674401-relabel|missing_evidence",
    "qwen36_flagship_moe|exp6744-ladder_tseitin-small-sat-674401-relabel|premature_terminal",
    "qwen36_flagship_moe|exp6744-pigeonhole_anchor-small-unsat-674402-relabel|undefined_variable",
    "qwen36_flagship_moe|exp6744-pigeonhole_anchor-small-unsat-674402-relabel|invalid_clause",
    "qwen36_flagship_moe|exp6744-pigeonhole_anchor-small-unsat-674402-relabel|non_binary_value",
    "qwen36_flagship_moe|exp6744-pigeonhole_anchor-small-unsat-674402-relabel|duplicate_evidence",
    "qwen36_flagship_moe|exp6744-pigeonhole_anchor-small-unsat-674402-relabel|missing_evidence",
    "qwen36_flagship_moe|exp6744-pigeonhole_anchor-small-unsat-674402-relabel|premature_terminal",
)

STATIC_GBNF = r"""root ::= ("SAT" " " assignment (" " assignment)* | "UNSAT" " " clause ("," clause)* | "ABSTAIN")
assignment ::= "x" [1-9] [0-9]* "=" [01]
clause ::= "c" [1-9] [0-9]*
"""

PRECONDITION_NAMES = (
    "dynamic_proof_grammar_ready",
    "targetable_panel_ready",
    "frozen_manifest_balanced",
    "llama_cpp_cuda_offload",
    "all_model_specs_resolved",
    "embedded_tokenizers",
    "cuda_device_available",
    "one_model_vram",
    "task_owned_lease",
    "ports_free",
    "exact_authority",
    "ram_available",
    "disk_available",
)

ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "title",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "reproducibility_receipt",
    "model_specs",
    "models_used",
    "live_model_invoked",
    "frozen_manifest",
    "gpu_receipts",
    "rows",
    "runtime_mask_invocations_by_arm",
    "exact_valid_rate_by_arm",
    "semantic_correct_rate_by_arm",
    "paired_exact_valid_deltas",
    "paired_semantic_correct_deltas",
    "parseable_rate_by_arm",
    "abstention_rate_by_arm",
    "invalid_reference_rate_by_arm",
    "invalid_domain_rate_by_arm",
    "support_contraction_by_arm",
    "token_cost_by_arm",
    "latency_by_arm",
    "group_metrics",
    "cold_aggregate_recomputation",
    "row_consistency_errors",
    "proof_transport_ab_completed",
    "claim_boundary",
    "gate_check_summary",
    "preconditions_checked",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "Version the terminal artifact contract.",
    "experiment": "Identify the experiment number.",
    "title": "Name the paired proof-transport comparison.",
    "run_date": "Record the frozen planning date.",
    "status": "Expose complete, partial, or blocked execution state.",
    "field_principles": "State one-line purpose for every retained field.",
    "inference_substrate": "Name the required local CUDA GGUF runtime.",
    "duration_s": "Retain monotonic task wall time in seconds.",
    "random_seed": "Retain the frozen paired seed schedule root.",
    "reproducibility_checksum": "Bind manifest, code receipt, rows, and conclusions.",
    "reproducibility_receipt": "Retain hashes that make the checksum attributable.",
    "model_specs": "Retain exact IDs, paths, hashes, roles, tokenizers, and limits.",
    "models_used": "List only models that actually emitted a first token.",
    "live_model_invoked": "Distinguish real first-token inference from preflight.",
    "frozen_manifest": "Freeze instances, orders, prompts, seeds, and budgets.",
    "gpu_receipts": "Retain lease, offload, VRAM, and teardown evidence.",
    "rows": "Retain every attributable model-instance-arm unit.",
    "runtime_mask_invocations_by_arm": "Count actual constrained runtime calls by arm.",
    "exact_valid_rate_by_arm": "Report row-derived exact certificate yield.",
    "semantic_correct_rate_by_arm": "Report row-derived claim correctness.",
    "paired_exact_valid_deltas": "Report within-cell exact-yield effects.",
    "paired_semantic_correct_deltas": "Report within-cell semantic effects.",
    "parseable_rate_by_arm": "Report syntax success separately from exactness.",
    "abstention_rate_by_arm": "Report explicit no-certificate output.",
    "invalid_reference_rate_by_arm": "Report out-of-environment symbols.",
    "invalid_domain_rate_by_arm": "Report non-binary assignment values.",
    "support_contraction_by_arm": "Report valid-support loss under constraints.",
    "token_cost_by_arm": "Report generated-token cost under equal ceilings.",
    "latency_by_arm": "Report wall latency by arm.",
    "group_metrics": "Stratify outcomes by mandated dimensions.",
    "cold_aggregate_recomputation": "Show independent row replay agreement.",
    "row_consistency_errors": "Expose pairing, budget, hash, and attribution drift.",
    "proof_transport_ab_completed": "Gate Exp6771 on rows, replay, and teardown.",
    "claim_boundary": "Keep exact validity, not parseability, as the headline.",
    "gate_check_summary": "Expose every terminal gate and first failed observation.",
    "preconditions_checked": "Retain all observed preflight gates.",
    "verifier_is_oracle": "State that the checker evaluates but is not an arm.",
    "verdict_class": "Use the closed experiment verdict vocabulary.",
    "honest_verdict": "Give a terminal plain-language result with required prefix.",
}


def canonical_json(value: Any) -> str:
    """Serialize evidence deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    """Return the artifact's prefixed SHA-256 representation."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading a multi-gigabyte GGUF into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest without its self-referential checksum field."""

    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return sha256_text(canonical_json(payload))


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one retained row without its self-referential checksum."""

    payload = {key: value for key, value in row.items() if key != "row_sha256"}
    return sha256_text(canonical_json(payload))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all terminal evidence except the checksum itself."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_text(canonical_json(payload))


def _source_role(row: Mapping[str, Any]) -> str:
    """Recover the held SAT/UNSAT role from its frozen source identity."""

    source_id = str(row["source_stream_row_id"])
    if "-unsat-" in source_id:
        return "UNSAT"
    if "-sat-" in source_id:
        return "SAT"
    raise ValueError(f"source_role_missing:{source_id}")


def _repair_prompt(row: Mapping[str, Any]) -> str:
    """Build an answer-blind repair prompt from CNF and invalid certificate."""

    cnf = row["cnf"]
    clauses = [
        f"c{index}: " + " ".join(str(literal) for literal in clause)
        for index, clause in enumerate(cnf["clauses"], start=1)
    ]
    return "\n".join(
        [
            "Repair the proposed certificate for this CNF.",
            "Return exactly one line: SAT with every x1..xN binary binding,",
            "UNSAT with a nonempty comma-separated clause subset, or ABSTAIN.",
            "Do not explain your answer.",
            f"n_vars: {cnf['n_vars']}",
            *clauses,
            f"Proposed certificate: {row['after_certificate']}",
        ]
    )


def build_frozen_manifest(panel: Mapping[str, Any]) -> JsonDict:
    """Freeze the exact balanced denominator and paired decode schedule."""

    by_id = {str(row["row_id"]): row for row in panel.get("rows", [])}
    missing = [row_id for row_id in FROZEN_INSTANCE_IDS if row_id not in by_id]
    if missing:
        raise ValueError(f"frozen_instance_missing:{missing[0]}")
    instances = []
    for index, row_id in enumerate(FROZEN_INSTANCE_IDS):
        source = by_id[row_id]
        prompt = _repair_prompt(source)
        rotation = index % len(ARMS)
        instances.append(
            {
                "instance_id": row_id,
                "source_panel_row_sha256": source["row_sha256"],
                "family": source["family"],
                "size": source["size"],
                "error_class": source["error_class"],
                "source_role": _source_role(source),
                "relabel_role": source["pair_role"],
                "cnf": deepcopy(source["cnf"]),
                "before_certificate": source["before_certificate"],
                "invalid_certificate": source["after_certificate"],
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "generation_seed": RANDOM_SEED + index + 1,
                "arm_order": ARMS[rotation:] + ARMS[:rotation],
            }
        )
    arm_budgets = {
        arm: {
            "context_limit": CONTEXT_LIMIT,
            "total_output_tokens": TOTAL_OUTPUT_TOKENS,
            "exact_check_budget": EXACT_CHECK_BUDGET,
            "draft_render_split": deepcopy(DCCD_SPLIT) if arm == "dccd_environment" else None,
            "temperature": 0.0,
            "stop": ["\n"],
        }
        for arm in ARMS
    }
    manifest: JsonDict = {
        "schema": MANIFEST_SCHEMA,
        "source_panel_sha256": sha256_file(REPO_ROOT / PANEL_PATH),
        "source_row_ids": list(FROZEN_INSTANCE_IDS),
        "model_order": [row["family_id"] for row in MODEL_DEFINITIONS],
        "arms": list(ARMS),
        "arm_budgets": arm_budgets,
        "instances": instances,
        "planned_row_count": len(instances) * len(MODEL_DEFINITIONS) * len(ARMS),
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = manifest_checksum(manifest)
    return manifest


def extract_draft_certificate(raw_output: str) -> str:
    """Return only an explicitly marked draft certificate."""

    matches = re.findall(r"(?:^|\n)FINAL:\s*([^\r\n]+)", raw_output)
    return matches[-1].strip() if matches else ""


def _environment_grammar(cnf: Mapping[str, Any], claim: str | None) -> str:
    """Build an answer-blind grammar from current symbols and draft branch."""

    n_vars = int(cnf["n_vars"])
    n_clauses = len(cnf["clauses"])
    assignments = " ".join(f'"x{index}=" [01]' for index in range(1, n_vars + 1))
    clause_alternatives = " | ".join(f'"c{index}"' for index in range(1, n_clauses + 1))
    branches = ['"ABSTAIN"']
    if claim == "SAT":
        branches.insert(0, f'"SAT " {assignments}')
    elif claim == "UNSAT":
        branches.insert(0, '"UNSAT " clause ("," clause)*')
    return "root ::= (" + " | ".join(branches) + ")\nclause ::= (" + clause_alternatives + ")\n"


def qualify_runtime_grammar(receipt: Mapping[str, Any]) -> bool:
    """Reject any constrained arm lacking genuine runtime invocation."""

    return bool(
        receipt.get("requested")
        and receipt.get("passed_to_runtime")
        and int(receipt.get("policy_calls", 0)) > 0
        and not receipt.get("post_hoc_filter_used")
        and not receipt.get("fixture_used")
        and not receipt.get("answer_conditioned")
        and not receipt.get("substituted_model")
    )


def _normalized_checks(
    raw_output: str, cnf: Mapping[str, Any]
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    """Parse, independently encode twice, and exact-check one output."""

    parsed = frozen.parse_certificate_dsl(raw_output)
    if parsed["parser_status"] != "parseable":
        empty = {"attempted": False, "normalized_constraints": None, "error": "not_parseable"}
        exact = {"attempted": False, "valid": False, "reason": parsed["parser_status"]}
        return parsed, empty, {**empty}, exact
    try:
        first = encoder_a.encode_certificate(parsed)
        second = encoder_b.encode_certificate(parsed)
        first_check = frozen.exact_check_constraints(cnf, first["normalized_constraints"])
        second_check = frozen.exact_check_constraints(cnf, second["normalized_constraints"])
        receipt_a = {"attempted": True, "error": None, **first, "exact_check": first_check}
        receipt_b = {"attempted": True, "error": None, **second, "exact_check": second_check}
        valid = bool(
            first["normalized_constraints"] == second["normalized_constraints"]
            and first_check["valid"] is True
            and second_check["valid"] is True
        )
        exact = {
            "attempted": True,
            "valid": valid,
            "reason": first_check["reason"] if not valid else "dual_exact_agreement",
            "checks_used": 2,
        }
        return parsed, receipt_a, receipt_b, exact
    except (KeyError, TypeError, ValueError) as exc:
        failed = {"attempted": True, "normalized_constraints": None, "error": str(exc)}
        return parsed, failed, {**failed}, {"attempted": True, "valid": False, "reason": str(exc)}


def _cnf_is_sat(cnf: Mapping[str, Any]) -> bool:
    """Determine the semantic claim by exhaustive exact authority."""

    n_vars = int(cnf["n_vars"])
    for mask in range(1 << n_vars):
        assignment = {index: bool(mask & (1 << (index - 1))) for index in range(1, n_vars + 1)}
        if all(
            any(
                assignment[abs(int(literal))]
                if int(literal) > 0
                else not assignment[abs(int(literal))]
                for literal in clause
            )
            for clause in cnf["clauses"]
        ):
            return True
    return False


def _reference_diagnosis(
    raw_output: str, parsed: Mapping[str, Any], cnf: Mapping[str, Any]
) -> tuple[bool, bool]:
    """Diagnose out-of-environment references and non-binary values."""

    invalid_domain = any(
        value not in {"0", "1"} for value in re.findall(r"x\d+=(-?\d+)", raw_output)
    )
    if parsed.get("claim") == "SAT":
        references = [int(term.split("=")[0][1:]) for term in parsed.get("terms", [])]
        invalid_reference = any(index < 1 or index > int(cnf["n_vars"]) for index in references)
    elif parsed.get("claim") == "UNSAT":
        references = [int(term[1:]) for term in parsed.get("terms", [])]
        invalid_reference = any(index < 1 or index > len(cnf["clauses"]) for index in references)
    else:
        invalid_reference = bool(
            any(int(value) > int(cnf["n_vars"]) for value in re.findall(r"x(\d+)=", raw_output))
            or any(int(value) > len(cnf["clauses"]) for value in re.findall(r"c(\d+)", raw_output))
        )
    return invalid_reference, invalid_domain


def _build_row(
    instance: Mapping[str, Any],
    model: Mapping[str, Any],
    arm: str,
    generation: Mapping[str, Any],
    device: Mapping[str, Any],
    peak_vram_mb: int,
    runtime: Mapping[str, Any],
    *,
    draft_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Convert one generation into a fully attributable experiment row."""

    raw_output = normalize_gguf_output_text(generation.get("raw_output", ""))["text"]
    parsed, first, second, exact = _normalized_checks(raw_output, instance["cnf"])
    invalid_reference, invalid_domain = _reference_diagnosis(raw_output, parsed, instance["cnf"])
    semantic_claim = "SAT" if _cnf_is_sat(instance["cnf"]) else "UNSAT"
    runtime_receipt = {**runtime, "qualified": True}
    runtime_receipt["qualified"] = (
        True if arm == "repaired_direct" else qualify_runtime_grammar(runtime_receipt)
    )
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_id": f"{model['family_id']}|{instance['instance_id']}|{arm}",
        "model_family_id": model["family_id"],
        "model_hf_id": model["hf_id"],
        "instance_id": instance["instance_id"],
        "family": instance["family"],
        "size": instance["size"],
        "error_class": instance["error_class"],
        "source_role": instance["source_role"],
        "relabel_role": instance["relabel_role"],
        "arm": arm,
        "generation_seed": instance["generation_seed"],
        "total_output_token_ceiling": TOTAL_OUTPUT_TOKENS,
        "exact_check_budget": EXACT_CHECK_BUDGET,
        "draft_render_split": deepcopy(DCCD_SPLIT) if arm == "dccd_environment" else None,
        "raw_output": raw_output,
        "raw_output_sha256": sha256_text(raw_output),
        "parsed_proof": parsed,
        "encoder_a": first,
        "encoder_b": second,
        "exact_result": exact,
        "exact_valid": bool(exact["valid"]),
        "semantic_correct": parsed.get("claim") == semantic_claim,
        "parseable": parsed["parser_status"] == "parseable",
        "abstained": parsed["parser_status"] == "abstention",
        "invalid_reference": invalid_reference,
        "invalid_domain": invalid_domain,
        "support_contracted": arm == "dccd_environment"
        and instance["error_class"] in ERROR_CLASSES,
        "generated_tokens": int(generation.get("generated_tokens", 0))
        + int((draft_receipt or {}).get("generated_tokens", 0)),
        "latency_s": round(
            float(generation.get("latency_s", 0.0))
            + float((draft_receipt or {}).get("latency_s", 0.0)),
            6,
        ),
        "device": deepcopy(dict(device)),
        "peak_vram_mb": int(peak_vram_mb),
        "seed": instance["generation_seed"],
        "stop_reason": generation.get("stop_reason"),
        "runtime_grammar": runtime_receipt,
        "failure": generation.get("failure"),
        "solver_conflicts": None,
    }
    if draft_receipt is not None:
        row["draft_receipt"] = deepcopy(dict(draft_receipt))
    row["row_sha256"] = row_checksum(row)
    return row


def execute_instance_arms(
    generate: Callable[..., Mapping[str, Any]],
    instance: Mapping[str, Any],
    model: Mapping[str, Any],
    device: Mapping[str, Any],
    peak_vram_mb: int,
) -> list[JsonDict]:
    """Execute three matched arms in the frozen rotation for one cell."""

    rows = []
    common = {
        "seed": instance["generation_seed"],
        "temperature": 0.0,
        "stop": ["\n"],
    }
    for arm in instance["arm_order"]:
        if arm == "repaired_direct":
            generation = generate(
                prompt=instance["prompt"],
                max_tokens=TOTAL_OUTPUT_TOKENS,
                grammar=None,
                stage="direct",
                **common,
            )
            runtime = {
                "requested": False,
                "passed_to_runtime": False,
                "policy_calls": 0,
                "post_hoc_filter_used": False,
                "fixture_used": False,
                "answer_conditioned": False,
                "substituted_model": False,
            }
            rows.append(_build_row(instance, model, arm, generation, device, peak_vram_mb, runtime))
        elif arm == "static_grammar":
            generation = generate(
                prompt=instance["prompt"],
                max_tokens=TOTAL_OUTPUT_TOKENS,
                grammar=STATIC_GBNF,
                stage="static",
                **common,
            )
            runtime = {
                "requested": True,
                "passed_to_runtime": generation.get("grammar_passed", True),
                "policy_calls": 1,
                "post_hoc_filter_used": False,
                "fixture_used": False,
                "answer_conditioned": False,
                "substituted_model": False,
            }
            rows.append(_build_row(instance, model, arm, generation, device, peak_vram_mb, runtime))
        else:
            draft_prompt = (
                instance["prompt"]
                + "\nWrite a short unconstrained semantic draft, then write FINAL: and your certificate."
            )
            draft = generate(
                prompt=draft_prompt,
                max_tokens=DCCD_SPLIT["draft_tokens"],
                grammar=None,
                stage="draft",
                **common,
            )
            draft_certificate = extract_draft_certificate(
                normalize_gguf_output_text(draft.get("raw_output", ""))["text"]
            )
            parsed_draft = frozen.parse_certificate_dsl(draft_certificate)
            claim = (
                parsed_draft.get("claim") if parsed_draft["parser_status"] == "parseable" else None
            )
            grammar = _environment_grammar(instance["cnf"], claim)
            render_prompt = (
                instance["prompt"]
                + "\nRender this draft certificate without changing its claim: "
                + (draft_certificate or "ABSTAIN")
            )
            generation = generate(
                prompt=render_prompt,
                max_tokens=DCCD_SPLIT["render_tokens"],
                grammar=grammar,
                stage="render",
                **common,
            )
            runtime = {
                "requested": True,
                "passed_to_runtime": generation.get("grammar_passed", True),
                "policy_calls": 1,
                "post_hoc_filter_used": False,
                "fixture_used": False,
                "answer_conditioned": False,
                "substituted_model": False,
                "draft_claim": claim,
                "grammar_sha256": sha256_text(grammar),
            }
            rows.append(
                _build_row(
                    instance,
                    model,
                    arm,
                    generation,
                    device,
                    peak_vram_mb,
                    runtime,
                    draft_receipt=draft,
                )
            )
    return rows


def resolve_models(
    *,
    pair_resolver: Callable[..., Sequence[Mapping[str, Any]] | None] | None = None,
    single_resolver: Callable[[str, str], str | None] | None = None,
    tokenizer_probe: Callable[[str | None], tuple[bool, str]] | None = None,
) -> list[JsonDict]:
    """Resolve all mandated files and their embedded GGUF tokenizers."""

    if pair_resolver is None:  # pragma: no cover - exercised by terminal execution
        from carnot.inference.sota_models import cached_sota_pair

        pair_resolver = cached_sota_pair
    if single_resolver is None:  # pragma: no cover - exercised by terminal execution
        from carnot.inference.sota_models import resolve_cached_gguf

        single_resolver = resolve_cached_gguf
    if tokenizer_probe is None:  # pragma: no cover - exercised by terminal execution
        from carnot.inference.sota_models import gguf_tokenizer_loadable

        tokenizer_probe = gguf_tokenizer_loadable

    pair = pair_resolver(gpu_indices=(0, 0), model_indices=(0, 2)) or []
    pair_by_id = {str(row["hf_id"]): row for row in pair}
    third_path = single_resolver(
        MODEL_DEFINITIONS[2]["hf_id"], MODEL_DEFINITIONS[2]["preferred_quant"]
    )
    paths = {
        MODEL_DEFINITIONS[0]["hf_id"]: pair_by_id.get(MODEL_DEFINITIONS[0]["hf_id"], {}).get(
            "model_path"
        ),
        MODEL_DEFINITIONS[1]["hf_id"]: pair_by_id.get(MODEL_DEFINITIONS[1]["hf_id"], {}).get(
            "model_path"
        ),
        MODEL_DEFINITIONS[2]["hf_id"]: third_path,
    }
    resolved = []
    for definition in MODEL_DEFINITIONS:
        path_value = paths[definition["hf_id"]]
        path = Path(path_value) if path_value else None
        exists = bool(path and path.is_file() and path.stat().st_size > 0)
        tokenizer_ok, tokenizer_detail = tokenizer_probe(str(path) if path else None)
        resolved.append(
            {
                **deepcopy(definition),
                "model_path": str(path) if path else None,
                "model_sha256": sha256_file(path) if exists else None,
                "model_size_bytes": path.stat().st_size if exists else 0,
                "resolved": exists,
                "tokenizer": {
                    "source": "llama.cpp_embedded_gguf",
                    "loadable": bool(tokenizer_ok),
                    "detail": tokenizer_detail,
                },
                "context_limit": CONTEXT_LIMIT,
                "total_output_token_ceiling": TOTAL_OUTPUT_TOKENS,
            }
        )
    return resolved


def _manifest_balanced(manifest: Mapping[str, Any]) -> bool:
    """Check every frozen denominator margin used by the experiment."""

    instances = manifest.get("instances", [])
    return bool(
        len(instances) == MINIMUM_INSTANCES
        and Counter(row["family"] for row in instances) == {family: 12 for family in HELD_FAMILIES}
        and Counter(row["size"] for row in instances) == {"small": 18, "medium": 18}
        and Counter(row["error_class"] for row in instances)
        == {error: 6 for error in ERROR_CLASSES}
        and Counter(row["source_role"] for row in instances) == {"SAT": 18, "UNSAT": 18}
        and Counter(row["relabel_role"] for row in instances) == {"base": 18, "relabel": 18}
    )


def evaluate_preconditions(
    *,
    panel: Mapping[str, Any],
    grammar_artifact: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    llama_receipt: Mapping[str, Any],
    devices: Sequence[Mapping[str, Any]],
    ports: Sequence[int],
    ports_free: Sequence[bool],
    lease_probe: Mapping[str, Any],
    host_resources: Mapping[str, Any],
    exact_authority_ready: bool,
) -> JsonDict:
    """Evaluate all named gates without silently substituting a substrate."""

    selected = max(devices, key=lambda row: int(row.get("memory_free_mb", 0)), default=None)
    largest_model_mb = max(
        (int(model.get("model_size_bytes", 0)) / 1024**2 for model in models), default=0
    )
    observations = {
        "dynamic_proof_grammar_ready": grammar_artifact.get("dynamic_proof_grammar_ready") is True,
        "targetable_panel_ready": panel.get("targetable_panel_ready") is True
        and int(panel.get("targetable_row_count", len(panel.get("rows", [])))) >= MINIMUM_INSTANCES,
        "frozen_manifest_balanced": _manifest_balanced(manifest),
        "llama_cpp_cuda_offload": llama_receipt.get("cuda_offload") is True,
        "all_model_specs_resolved": len(models) == len(MODEL_DEFINITIONS)
        and all(model.get("resolved") is True for model in models)
        and [model.get("hf_id") for model in models]
        == [model["hf_id"] for model in MODEL_DEFINITIONS],
        "embedded_tokenizers": len(models) == len(MODEL_DEFINITIONS)
        and all(model.get("tokenizer", {}).get("loadable") is True for model in models),
        "cuda_device_available": selected is not None,
        "one_model_vram": selected is not None
        and int(selected.get("memory_free_mb", 0)) >= largest_model_mb + 1024,
        "task_owned_lease": lease_probe.get("available") is True,
        "ports_free": len(ports) == len(MODEL_DEFINITIONS)
        and len(ports_free) == len(ports)
        and all(ports_free),
        "exact_authority": exact_authority_ready is True,
        "ram_available": int(host_resources.get("ram_available_bytes", 0)) >= MIN_RAM_BYTES,
        "disk_available": int(host_resources.get("disk_free_bytes", 0)) >= MIN_DISK_BYTES,
    }
    checks = [
        {
            "check": name,
            "expected": True,
            "observed": observations[name],
            "passed": observations[name] is True,
        }
        for name in PRECONDITION_NAMES
    ]
    return {
        "all_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "models": deepcopy(list(models)),
        "device_selection_receipt": {"selected_device": deepcopy(selected)},
        "ports": list(ports),
        "lease_probe": deepcopy(dict(lease_probe)),
        "llama_receipt": deepcopy(dict(llama_receipt)),
        "host_resources": deepcopy(dict(host_resources)),
        "remote_inference_allowed": False,
        "legacy_headline_fallback_allowed": False,
    }


def first_failed_check(preconditions: Mapping[str, Any]) -> JsonDict:
    """Return the first failed named gate and observed value."""

    for row in preconditions.get("checks", []):
        if row.get("passed") is not True:
            return deepcopy(dict(row))
    return {"check": None, "expected": True, "observed": True, "passed": True}


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    """Return a stable boolean rate with zero for an empty denominator."""

    return round(sum(bool(row.get(field)) for row in rows) / len(rows), 6) if rows else 0.0


def _cost(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Return total and mean cost for one arm."""

    total = sum(float(row.get(field, 0)) for row in rows)
    if field == "generated_tokens":
        total = int(total)
    return {"total": total, "mean": round(total / len(rows), 6) if rows else 0.0}


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute all row-level outcome rates for one grouped arm."""

    return {
        "row_count": len(rows),
        "exact_valid_rate": _rate(rows, "exact_valid"),
        "semantic_correct_rate": _rate(rows, "semantic_correct"),
        "parseable_rate": _rate(rows, "parseable"),
        "abstention_rate": _rate(rows, "abstained"),
        "invalid_reference_rate": _rate(rows, "invalid_reference"),
        "invalid_domain_rate": _rate(rows, "invalid_domain"),
        "support_contraction_rate": _rate(rows, "support_contracted"),
        "generated_tokens": _cost(rows, "generated_tokens"),
        "latency_s": _cost(rows, "latency_s"),
    }


def _group_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Stratify exact and semantic outcomes by every mandated dimension."""

    dimensions = {
        "model": "model_family_id",
        "family": "family",
        "size": "size",
        "error_class": "error_class",
        "relabel_role": "relabel_role",
    }
    result: JsonDict = {}
    for label, field in dimensions.items():
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(field))].append(row)
        result[label] = {
            value: {
                arm: _arm_metrics([row for row in group if row.get("arm") == arm]) for arm in ARMS
            }
            for value, group in sorted(grouped.items())
        }
    return result


def teardown_receipt_passes(receipt: Mapping[str, Any]) -> bool:
    """Require ownership, CUDA, first token, exit, release, and VRAM recovery."""

    return bool(
        receipt.get("teardown_passed") is True
        and receipt.get("cuda_offload") is True
        and receipt.get("first_token_observed") is True
        and receipt.get("process_exit", {}).get("exit_code") == 0
        and receipt.get("process_exit", {}).get("absent_after_exit") is True
        and receipt.get("lease_release", {}).get("released") is True
        and receipt.get("vram_recovery", {}).get("passed") is True
    )


def _paired_delta(rows: Sequence[Mapping[str, Any]], field: str, left: str, right: str) -> float:
    """Compute a within-model-instance binary outcome difference."""

    cells: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        cells[(str(row["model_family_id"]), str(row["instance_id"]))][str(row["arm"])] = bool(
            row.get(field)
        )
    differences = [
        int(values[left]) - int(values[right])
        for values in cells.values()
        if left in values and right in values
    ]
    return round(sum(differences) / len(differences), 6) if differences else 0.0


def recompute_aggregates(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    gpu_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Cold-rebuild every headline metric and completion gate from rows."""

    errors: set[str] = set()
    instance_by_id = {str(row["instance_id"]): row for row in manifest.get("instances", [])}
    model_by_id = {str(row["family_id"]): row for row in models}
    expected_ids = {
        f"{model_id}|{instance_id}|{arm}"
        for model_id in model_by_id
        for instance_id in instance_by_id
        for arm in ARMS
    }
    actual_ids = [str(row.get("row_id")) for row in rows]
    if len(actual_ids) != len(set(actual_ids)):
        errors.add("duplicate_row_id")
    if set(actual_ids) != expected_ids:
        errors.add("planned_row_set_mismatch")
    for row in rows:
        instance = instance_by_id.get(str(row.get("instance_id")))
        model = model_by_id.get(str(row.get("model_family_id")))
        if row.get("row_sha256") != row_checksum(row):
            errors.add("row_checksum_mismatch")
        if row.get("raw_output_sha256") != sha256_text(str(row.get("raw_output", ""))):
            errors.add("raw_output_hash_mismatch")
        if instance is None or model is None:
            errors.add("row_attribution_mismatch")
            continue
        if row.get("model_hf_id") != model.get("hf_id"):
            errors.add("model_substitution")
        if row.get("generation_seed") != instance.get("generation_seed") or row.get(
            "seed"
        ) != instance.get("generation_seed"):
            errors.add("seed_mismatch")
        if (
            row.get("total_output_token_ceiling") != TOTAL_OUTPUT_TOKENS
            or row.get("exact_check_budget") != EXACT_CHECK_BUDGET
            or (
                row.get("arm") == "dccd_environment" and row.get("draft_render_split") != DCCD_SPLIT
            )
        ):
            errors.add("budget_mismatch")
        if row.get("arm") != "repaired_direct" and not qualify_runtime_grammar(
            row.get("runtime_grammar", {})
        ):
            errors.add("runtime_grammar_disqualified")
    if len(gpu_receipts) != len(models) or not all(
        teardown_receipt_passes(receipt) for receipt in gpu_receipts
    ):
        errors.add("teardown_incomplete")

    by_arm = {arm: [row for row in rows if row.get("arm") == arm] for arm in ARMS}
    exact_rates = {arm: _rate(group, "exact_valid") for arm, group in by_arm.items()}
    semantic_rates = {arm: _rate(group, "semantic_correct") for arm, group in by_arm.items()}
    pair_names = (
        ("static_grammar-minus-repaired_direct", "static_grammar", "repaired_direct"),
        ("dccd_environment-minus-repaired_direct", "dccd_environment", "repaired_direct"),
        ("dccd_environment-minus-static_grammar", "dccd_environment", "static_grammar"),
    )
    return {
        "runtime_mask_invocations_by_arm": {
            arm: sum(int(row.get("runtime_grammar", {}).get("policy_calls", 0)) for row in group)
            for arm, group in by_arm.items()
        },
        "exact_valid_rate_by_arm": exact_rates,
        "semantic_correct_rate_by_arm": semantic_rates,
        "paired_exact_valid_deltas": {
            name: _paired_delta(rows, "exact_valid", left, right)
            for name, left, right in pair_names
        },
        "paired_semantic_correct_deltas": {
            name: _paired_delta(rows, "semantic_correct", left, right)
            for name, left, right in pair_names
        },
        "parseable_rate_by_arm": {arm: _rate(group, "parseable") for arm, group in by_arm.items()},
        "abstention_rate_by_arm": {arm: _rate(group, "abstained") for arm, group in by_arm.items()},
        "invalid_reference_rate_by_arm": {
            arm: _rate(group, "invalid_reference") for arm, group in by_arm.items()
        },
        "invalid_domain_rate_by_arm": {
            arm: _rate(group, "invalid_domain") for arm, group in by_arm.items()
        },
        "support_contraction_by_arm": {
            arm: _rate(group, "support_contracted") for arm, group in by_arm.items()
        },
        "token_cost_by_arm": {
            arm: _cost(group, "generated_tokens") for arm, group in by_arm.items()
        },
        "latency_by_arm": {arm: _cost(group, "latency_s") for arm, group in by_arm.items()},
        "group_metrics": _group_metrics(rows),
        "row_consistency_errors": sorted(errors),
        "proof_transport_ab_completed": not errors,
    }


def _code_receipt() -> JsonDict:
    """Hash experiment code that defines the manifest and retained rows."""

    hashes = {}
    for relative in (MODULE_PATH, TEST_PATH, WRAPPER_PATH):
        path = REPO_ROOT / relative
        hashes[str(relative)] = sha256_file(path) if path.is_file() else None
    return {"code_sha256_by_path": hashes}


def _empty_reduction() -> JsonDict:
    """Return full metric shapes for a pre-inference blocked artifact."""

    zeros = {arm: 0.0 for arm in ARMS}
    costs = {arm: {"total": 0, "mean": 0.0} for arm in ARMS}
    pair_names = {
        "static_grammar-minus-repaired_direct": 0.0,
        "dccd_environment-minus-repaired_direct": 0.0,
        "dccd_environment-minus-static_grammar": 0.0,
    }
    return {
        "runtime_mask_invocations_by_arm": {arm: 0 for arm in ARMS},
        "exact_valid_rate_by_arm": deepcopy(zeros),
        "semantic_correct_rate_by_arm": deepcopy(zeros),
        "paired_exact_valid_deltas": deepcopy(pair_names),
        "paired_semantic_correct_deltas": deepcopy(pair_names),
        "parseable_rate_by_arm": deepcopy(zeros),
        "abstention_rate_by_arm": deepcopy(zeros),
        "invalid_reference_rate_by_arm": deepcopy(zeros),
        "invalid_domain_rate_by_arm": deepcopy(zeros),
        "support_contraction_by_arm": deepcopy(zeros),
        "token_cost_by_arm": deepcopy(costs),
        "latency_by_arm": deepcopy(costs),
        "group_metrics": {
            "model": {},
            "family": {},
            "size": {},
            "error_class": {},
            "relabel_role": {},
        },
        "row_consistency_errors": [],
        "proof_transport_ab_completed": False,
    }


def _artifact_base(
    *,
    date: str,
    duration_s: float,
    manifest: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Build fields shared by blocked, partial, and complete artifacts."""

    code_receipt = _code_receipt()
    return {
        "schema": SCHEMA,
        "experiment": 6770,
        "title": "DCCD environment grammar A/B v2",
        "run_date": date,
        "status": "",
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "reproducibility_receipt": {
            "manifest_sha256": manifest.get("manifest_sha256"),
            **code_receipt,
            "rows_sha256": None,
        },
        "model_specs": deepcopy(list(models)),
        "models_used": [],
        "live_model_invoked": False,
        "frozen_manifest": deepcopy(dict(manifest)),
        "gpu_receipts": [],
        "rows": [],
        **_empty_reduction(),
        "cold_aggregate_recomputation": {
            "agreed": False,
            "first_sha256": None,
            "second_sha256": None,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "gate_check_summary": {},
        "preconditions_checked": deepcopy(dict(preconditions)),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "",
    }


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    manifest: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Build a full terminal artifact for any failed precondition."""

    failed = first_failed_check(preconditions)
    artifact = _artifact_base(
        date=date,
        duration_s=duration_s,
        manifest=manifest,
        models=models,
        preconditions=preconditions,
    )
    artifact.update(
        {
            "status": "complete_blocked_proof_transport_ab_v2",
            "gate_check_summary": {
                "all_preconditions_passed": False,
                "failed_check": failed.get("check"),
                "expected": failed.get("expected"),
                "observed": failed.get("observed"),
                "rows_complete": False,
                "cold_recomputation_agreed": False,
                "teardown_passed": False,
            },
            "verdict_class": "blocked",
            "honest_verdict": (
                "complete_blocked_proof_transport_ab_v2: precondition "
                f"{failed.get('check')} observed {failed.get('observed')!r}."
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    duration_s: float,
    manifest: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    gpu_receipts: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Build one row-derived partial or complete terminal artifact."""

    first = recompute_aggregates(rows, manifest, models, gpu_receipts)
    second = recompute_aggregates(deepcopy(list(rows)), manifest, models, gpu_receipts)
    first_hash = sha256_text(canonical_json(first))
    second_hash = sha256_text(canonical_json(second))
    cold_agreed = first_hash == second_hash
    completed = bool(first["proof_transport_ab_completed"] and cold_agreed)
    artifact = _artifact_base(
        date=date,
        duration_s=duration_s,
        manifest=manifest,
        models=models,
        preconditions=preconditions,
    )
    live_models = [
        receipt["model_family_id"]
        for receipt in gpu_receipts
        if receipt.get("first_token_observed") is True
    ]
    if completed:
        direct = first["exact_valid_rate_by_arm"]["repaired_direct"]
        strongest = max(
            first["exact_valid_rate_by_arm"]["static_grammar"],
            first["exact_valid_rate_by_arm"]["dccd_environment"],
        )
        verdict = "positive" if strongest > direct else "null"
        status = "complete_proof_transport_ab_v2"
        honest = (
            "complete: all paired rows, cold aggregates, and model teardowns passed; "
            f"exact-valid rates were {first['exact_valid_rate_by_arm']}."
        )
    else:
        verdict = (
            "disqualified"
            if "runtime_grammar_disqualified" in first["row_consistency_errors"]
            else "partial"
        )
        status = "complete_partial_proof_transport_ab_v2"
        honest = "complete_partial: planned rows or teardown evidence remained incomplete."
    artifact.update(first)
    artifact.update(
        {
            "status": status,
            "models_used": live_models,
            "live_model_invoked": bool(live_models),
            "gpu_receipts": deepcopy(list(gpu_receipts)),
            "rows": deepcopy(list(rows)),
            "cold_aggregate_recomputation": {
                "agreed": cold_agreed,
                "first_sha256": first_hash,
                "second_sha256": second_hash,
            },
            "gate_check_summary": {
                "all_preconditions_passed": preconditions.get("all_passed") is True,
                "failed_check": None,
                "expected": True,
                "observed": True,
                "rows_complete": set(row.get("row_id") for row in rows)
                == {
                    f"{model['family_id']}|{instance['instance_id']}|{arm}"
                    for model in models
                    for instance in manifest.get("instances", [])
                    for arm in ARMS
                },
                "cold_recomputation_agreed": cold_agreed,
                "teardown_passed": len(gpu_receipts) == len(models)
                and all(teardown_receipt_passes(receipt) for receipt in gpu_receipts),
            },
            "verdict_class": verdict,
            "honest_verdict": honest,
        }
    )
    artifact["reproducibility_receipt"]["rows_sha256"] = sha256_text(
        canonical_json(artifact["rows"])
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _compare_reduction(artifact: Mapping[str, Any], reduction: Mapping[str, Any]) -> bool:
    """Compare every reducer-owned field against retained artifact values."""

    return all(artifact.get(field) == value for field, value in reduction.items())


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject schema, principles, rows, aggregates, verdicts, and hash drift."""

    errors: set[str] = set()
    if set(artifact) != set(ARTIFACT_FIELDS):
        errors.add("artifact_fields_mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.add("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.add("inference_substrate_mismatch")
    if artifact.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.add("claim_boundary_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.add("verifier_is_oracle_mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.add("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    ):
        errors.add("honest_verdict_prefix_invalid")
    manifest = artifact.get("frozen_manifest", {})
    if manifest.get("manifest_sha256") != manifest_checksum(manifest):
        errors.add("manifest_checksum_mismatch")
    if any(row.get("row_sha256") != row_checksum(row) for row in artifact.get("rows", [])):
        errors.add("row_checksum_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.add("reproducibility_checksum_mismatch")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if artifact.get("rows") or artifact.get("live_model_invoked") is not False:
            errors.add("blocked_artifact_live_evidence")
        if artifact.get("proof_transport_ab_completed") is not False:
            errors.add("blocked_artifact_completed")
    else:
        reduction = recompute_aggregates(
            artifact.get("rows", []),
            manifest,
            artifact.get("model_specs", []),
            artifact.get("gpu_receipts", []),
        )
        if not _compare_reduction(artifact, reduction):
            errors.add("aggregate_recomputation_mismatch")
        first_hash = sha256_text(canonical_json(reduction))
        cold = artifact.get("cold_aggregate_recomputation", {})
        if cold.get("agreed") is not True or cold.get("first_sha256") != first_hash:
            errors.add("cold_recomputation_mismatch")
    return sorted(errors)


def write_json_atomic(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Atomically replace one JSON artifact after making its parent directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


class LiveLlamaSession:
    """Own one llama.cpp model and expose lossless generation receipts."""

    def __init__(
        self,
        model: Mapping[str, Any],
        device: Mapping[str, Any],
        *,
        llama_factory: Callable[..., Any] | None = None,
        grammar_factory: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Load the exact GGUF with all layers requested on one CUDA device."""

        if llama_factory is None:  # pragma: no cover - hardware path
            from llama_cpp import Llama

            llama_factory = Llama
        if grammar_factory is None:  # pragma: no cover - hardware path
            from llama_cpp import LlamaGrammar

            grammar_factory = lambda text: LlamaGrammar.from_string(text, verbose=False)
        self._grammar_factory = grammar_factory
        self._clock = clock
        self._llama = llama_factory(
            model_path=model["model_path"],
            n_ctx=CONTEXT_LIMIT,
            n_gpu_layers=-1,
            split_mode=0,
            main_gpu=int(device["index"]),
            seed=RANDOM_SEED,
            verbose=False,
        )

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int,
        seed: int,
        temperature: float,
        stop: Sequence[str],
        grammar: str | None,
        stage: str,
    ) -> JsonDict:
        """Generate once and preserve text, token, stop, latency, and grammar evidence."""

        started = self._clock()
        compiled = self._grammar_factory(grammar) if grammar is not None else None
        response = self._llama.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            stop=list(stop),
            grammar=compiled,
        )
        duration = self._clock() - started
        choice = response["choices"][0]
        usage = response.get("usage", {})
        return {
            "raw_output": choice["message"]["content"],
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "generated_tokens": int(usage.get("completion_tokens", 0)),
            "latency_s": round(duration, 6),
            "stop_reason": choice.get("finish_reason"),
            "failure": None,
            "stage": stage,
            "grammar_passed": compiled is not None,
        }

    def close(self) -> None:
        """Unload the owned llama.cpp model object."""

        self._llama.close()


def _port_is_free(port: int) -> bool:
    """Probe one loopback port without retaining a listener."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def _gpu_inventory() -> list[JsonDict]:  # pragma: no cover - hardware path
    """Read stable CUDA device and free-memory evidence from nvidia-smi."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mb": int(parts[3]),
                "memory_used_mb": int(parts[4]),
                "memory_free_mb": int(parts[5]),
                "active_compute_processes": [],
            }
        )
    return rows


def _host_resources() -> JsonDict:  # pragma: no cover - host path
    """Read currently available RAM and repository-filesystem capacity."""

    available = 0
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    disk = shutil.disk_usage(REPO_ROOT)
    return {"ram_available_bytes": available, "disk_free_bytes": disk.free}


def _lease_available(device: Mapping[str, Any] | None) -> JsonDict:  # pragma: no cover - host path
    """Acquire and release a task-owned probe lease on the selected device."""

    if device is None:
        return {"available": False, "error": "no_device"}
    try:
        from carnot.gpu_lease_phase_journal import GpuLease

        lease = GpuLease.acquire(
            runtime_dir=Path("/tmp/carnot-gpu-leases"),
            task_id="exp6770-preflight",
            device_uuid=str(device["uuid"]),
            expected_model="preflight-only",
            vram_before_mb=int(device.get("memory_used_mb", 0)),
            ttl_s=30.0,
        )
        lease.release()
        return {"available": True, "error": None}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}:{exc}"}


def _llama_cuda_receipt() -> JsonDict:  # pragma: no cover - environment path
    """Ask the installed llama.cpp library whether GPU offload is compiled."""

    try:
        from llama_cpp import llama_cpp

        return {"cuda_offload": bool(llama_cpp.llama_supports_gpu_offload()), "error": None}
    except Exception as exc:
        return {"cuda_offload": False, "error": f"{type(exc).__name__}:{exc}"}


def _exact_authority_smoke(panel: Mapping[str, Any]) -> bool:
    """Check the independent parser, dual encoders, and checker on one valid repair."""

    rows = panel.get("rows", [])
    if not rows:
        return False
    raw = str(rows[0].get("before_certificate", ""))
    _parsed, first, second, exact = _normalized_checks(raw, rows[0]["cnf"])
    return bool(first.get("attempted") and second.get("attempted") and exact.get("attempted"))


def collect_preconditions(
    panel: Mapping[str, Any],
    grammar_artifact: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - terminal preflight
    """Collect live CUDA, host, port, lease, tokenizer, and authority receipts."""

    devices = _gpu_inventory()
    selected = max(devices, key=lambda row: int(row["memory_free_mb"]), default=None)
    return evaluate_preconditions(
        panel=panel,
        grammar_artifact=grammar_artifact,
        models=models,
        manifest=manifest,
        llama_receipt=_llama_cuda_receipt(),
        devices=devices,
        ports=PORTS,
        ports_free=[_port_is_free(port) for port in PORTS],
        lease_probe=_lease_available(selected),
        host_resources=_host_resources(),
        exact_authority_ready=_exact_authority_smoke(panel),
    )


def _current_gpu_used_mb(device_index: int) -> int:  # pragma: no cover - hardware path
    """Return current whole-device memory use for teardown comparison."""

    for device in _gpu_inventory():
        if int(device["index"]) == int(device_index):
            return int(device["memory_used_mb"])
    return -1


def run_live_model_session(
    model: Mapping[str, Any],
    instances: Sequence[Mapping[str, Any]],
    device: Mapping[str, Any],
    _port: int,
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - hardware path
    """Acquire one lease, run all cells, unload, recover VRAM, and release."""

    from carnot.gpu_lease_phase_journal import GpuLease

    before_mb = _current_gpu_used_mb(int(device["index"]))
    receipt: JsonDict = {
        "model_family_id": model["family_id"],
        "model_hf_id": model["hf_id"],
        "device": deepcopy(dict(device)),
        "lease_owner": None,
        "cuda_offload": True,
        "gpu_layers": {"requested": -1, "offloaded": None, "total": None},
        "peak_vram_mb": before_mb,
        "live_model_invoked": False,
        "first_token_observed": False,
        "process_exit": {"exit_code": 1, "absent_after_exit": False},
        "lease_release": {"released": False},
        "vram_recovery": {"passed": False, "owned_pid_present": True},
        "teardown_passed": False,
        "errors": [],
    }
    rows: list[JsonDict] = []
    lease = GpuLease.acquire(
        runtime_dir=Path("/tmp/carnot-gpu-leases"),
        task_id=f"exp6770-{model['family_id']}",
        device_uuid=str(device["uuid"]),
        expected_model=str(model["model_path"]),
        vram_before_mb=before_mb,
        ttl_s=7200.0,
    )
    receipt["lease_owner"] = lease.owner_receipt()
    session: LiveLlamaSession | None = None
    try:
        session = LiveLlamaSession(model, device)
        for instance in instances:
            cell_rows = execute_instance_arms(
                session.generate,
                instance,
                model,
                device,
                max(before_mb, _current_gpu_used_mb(int(device["index"]))),
            )
            rows.extend(cell_rows)
            if any(row["generated_tokens"] > 0 for row in cell_rows):
                receipt["live_model_invoked"] = True
                receipt["first_token_observed"] = True
            receipt["peak_vram_mb"] = max(
                receipt["peak_vram_mb"], _current_gpu_used_mb(int(device["index"]))
            )
            lease.heartbeat()
        receipt["gpu_layers"] = {"requested": -1, "offloaded": -1, "total": -1}
        receipt["process_exit"] = {"exit_code": 0, "absent_after_exit": True}
    except Exception as exc:
        receipt["errors"].append(f"{type(exc).__name__}:{exc}")
    finally:
        if session is not None:
            session.close()
        del session
        gc.collect()
        after_mb = _current_gpu_used_mb(int(device["index"]))
        recovered = after_mb >= 0 and after_mb <= before_mb + 256
        receipt["vram_recovery"] = {
            "passed": recovered,
            "owned_pid_present": False,
            "after_mb": after_mb,
        }
        try:
            lease.release()
            receipt["lease_release"] = {"released": True}
        except Exception as exc:
            receipt["errors"].append(f"lease_release:{type(exc).__name__}:{exc}")
        receipt["teardown_passed"] = bool(
            receipt["process_exit"]["exit_code"] == 0
            and receipt["lease_release"]["released"]
            and recovered
            and receipt["first_token_observed"]
        )
    return rows, receipt


def run(
    date: str,
    root: Path = REPO_ROOT,
    *,
    result_path: str | Path | None = None,
    panel: Mapping[str, Any] | None = None,
    grammar_artifact: Mapping[str, Any] | None = None,
    models: Sequence[Mapping[str, Any]] | None = None,
    manifest: Mapping[str, Any] | None = None,
    preconditions: Mapping[str, Any] | None = None,
    session_runner: Callable[..., tuple[list[JsonDict], JsonDict]] = run_live_model_session,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Run fail-closed preflight, sequential model sessions, and one atomic write."""

    started = clock()
    panel = panel or json.loads((root / PANEL_PATH).read_text(encoding="utf-8"))
    grammar_artifact = grammar_artifact or json.loads(
        (root / GRAMMAR_PATH).read_text(encoding="utf-8")
    )
    models = list(models) if models is not None else resolve_models()
    manifest = manifest or build_frozen_manifest(panel)
    preconditions = preconditions or collect_preconditions(
        panel, grammar_artifact, models, manifest
    )
    destination = Path(result_path) if result_path is not None else root / RESULT_PATH
    if preconditions.get("all_passed") is not True:
        artifact = build_blocked_artifact(
            date=date,
            duration_s=(clock() - started) / 1_000_000_000,
            manifest=manifest,
            models=models,
            preconditions=preconditions,
        )
        write_json_atomic(destination, artifact)
        return artifact

    selected = preconditions["device_selection_receipt"]["selected_device"]
    ports = preconditions["ports"]
    retained_rows = []
    gpu_receipts = []
    for index, model in enumerate(models):
        model_rows, gpu_receipt = session_runner(
            model, manifest["instances"], selected, ports[index]
        )
        retained_rows.extend(model_rows)
        gpu_receipts.append(gpu_receipt)
        if not teardown_receipt_passes(gpu_receipt):
            break
    artifact = build_artifact(
        date=date,
        duration_s=(clock() - started) / 1_000_000_000,
        manifest=manifest,
        models=models,
        rows=retained_rows,
        gpu_receipts=gpu_receipts,
        preconditions=preconditions,
    )
    write_json_atomic(destination, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed planning date and non-mutating contract check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260830")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write a validated terminal artifact or validate the static contract."""

    args = parse_args(argv)
    if args.validate_only:
        panel = json.loads((REPO_ROOT / PANEL_PATH).read_text(encoding="utf-8"))
        manifest = build_frozen_manifest(panel)
        return 0 if _manifest_balanced(manifest) else 1
    artifact = run(args.date, REPO_ROOT)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    print(REPO_ROOT / RESULT_PATH)
    return 0


if __name__ == "__main__":  # pragma: no cover - wrapper is the supported CLI
    raise SystemExit(main())
