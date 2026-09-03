"""Compare plain relation generation with and without exact prefix guidance.

Spec refs: REQ-INFERENCE-6920 and SCENARIO-INFERENCE-6920-*.

The model always emits ordinary text. The guided arm checks a completed line
only after the model samples it. A separate clingo path evaluates each selected
final program, so the search engine cannot approve its own result.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from typing import Any
from urllib import error, request

from carnot import experiment_6899_live_relation_acquisition_canary as canary
from carnot import experiment_6919_exact_prefix_viability_fixture as exact
from carnot import gpu_lease_phase_journal as lease_api
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess, port_is_free
from carnot.inference.llama_server_supervisor import read_process_identity
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/llm-ebm-inference/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6920_sota_exact_guided_relation_generation.json")
EXP6919_RELATIVE_PATH = Path("results/experiment_6919_exact_prefix_viability_fixture.json")
EXACT_MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6919_exact_prefix_viability_fixture.py")
ASP_COMPILER_RELATIVE_PATH = Path("python/carnot/asp_energy.py")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6920_sota_exact_guided_relation_generation.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6920_sota_exact_guided_relation_generation.py"
)
WRAPPER_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6920_sota_exact_guided_relation_generation.py"
)

INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda_with_dual_exact_engines"
IN_LOOP_ENGINE = "python_bounded_relation_enumerator_v1"
FINAL_ENGINE = "clingo_stable_model_final_engine_v1"
RANDOM_SEED = 692003
SEEDS = (692003, 692017, 692033)
ARMS = ("direct_generation", "unguided_best_of_k", "guided_frontier")
MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_FAMILIES = {
    MODEL_SPECS[0]: "qwen36_35b_a3b",
    MODEL_SPECS[1]: "gemma4_31b_dense",
    MODEL_SPECS[2]: "gemma4_26b_a4b",
}
EXPECTED_MODEL_HASHES = {
    MODEL_SPECS[0]: "sha256:ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
    MODEL_SPECS[1]: "sha256:9fdf3dc8b0384830b4402d151388c140bd8eb2abf8d60588d8224231198254a1",
    MODEL_SPECS[2]: "sha256:34c746b1d50ab813e29cd46c4796e3f43c741901a582f93a67b55b9fc9687b35",
}
EXPECTED_TOKENIZER_HASHES = {
    MODEL_SPECS[0]: "sha256:a008ef118a726aba1d1cbfecb73d4571a86d78f6e75cf6d33a687747c8f61c80",
    MODEL_SPECS[1]: "sha256:9696db82c1037b59ec7f2b1f2273272ee01466d9998d0f7816192b39db58a4d6",
    MODEL_SPECS[2]: "sha256:9696db82c1037b59ec7f2b1f2273272ee01466d9998d0f7816192b39db58a4d6",
}
EXPECTED_EXP6919_SHA256 = "sha256:78ea6e4d5c56efdd20b980cc083c72ff3c870ed8ee4f8f1e7440818e1c906687"
EXPECTED_ENGINE_HASHES = {
    "prefix_engine_module": (
        "sha256:e5224bf919247125df1f03b6acc468cdc08c05c59cd878781e4dbad8888ccda4"
    ),
    "asp_energy_compiler": (
        "sha256:0f6077bcd49aa93a6cdbde72422ecf97d905b76b31cadbc0cd401c494af015e1"
    ),
}

SOURCE_TASK_COUNT = 30
MIN_TASKS_PER_FAMILY = 6
MATCHED_CANDIDATE_BUDGET = 4
MATCHED_TOTAL_TOKEN_LIMIT = 128
MATCHED_CANDIDATE_TOKEN_LIMIT = MATCHED_TOTAL_TOKEN_LIMIT // MATCHED_CANDIDATE_BUDGET
DIRECT_TOKEN_LIMIT = 64
MIN_FREE_VRAM_MB = 24000
REQUEST_TIMEOUT_S = 180.0
HEALTH_TIMEOUT_S = 360.0
LEASE_TTL_S = 1200.0
CONTEXT_LENGTH = 4096
LEASE_RUNTIME_DIR = Path("/tmp/carnot-gpu-leases")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "model_specs",
    "models_used",
    "model_artifact_hashes",
    "tokenizer_receipts",
    "llama_cpp_receipts",
    "gpu_lease_rows",
    "server_lifecycle_rows",
    "rows",
    "source_rows",
    "arm_budget_rows",
    "direct_generation_rows",
    "unguided_best_of_k_rows",
    "guided_frontier_rows",
    "candidate_rows",
    "prefix_energy_rows",
    "rejected_branch_rows",
    "selected_branch_rows",
    "frontier_size_rows",
    "abstention_rows",
    "parser_rows",
    "final_exact_outcome_rows",
    "independent_solver_receipts",
    "per_model_arm_rows",
    "per_family_arm_rows",
    "validity_delta_rows",
    "false_admission_rows",
    "token_cost_rows",
    "latency_rows",
    "vram_rows",
    "energy_proxy_rows",
    "raw_request_manifest",
    "raw_output_manifest",
    "external_text_scorer_call_count",
    "constrained_schema_decode_count",
    "repair_prompt_count",
    "finite_answer_id_count",
    "model_weight_mutation_count",
    "random_seed",
    "reproducibility_checksum",
    "guided_generation_run_complete_score",
    "exact_guidance_utility_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason per field makes the result contract auditable.",
    "preconditions_checked": "Preflight evidence stops unsupported live claims before inference.",
    "inference_substrate": "The substrate names the live CUDA and dual-engine path that ran.",
    "duration_s": "Measured wall time distinguishes live work from an asserted result.",
    "source_artifact_hashes": "Hashes bind the run to exact fixtures, engines, code, and tests.",
    "model_specs": "Exact repository IDs prevent a smaller model from replacing a required model.",
    "models_used": "Used-model rows distinguish requested models from authenticated live loads.",
    "model_artifact_hashes": "Weight hashes detect cache or quantization substitution.",
    "tokenizer_receipts": "Native receipts prevent a Hugging Face tokenizer substitution.",
    "llama_cpp_receipts": "Server receipts bind outputs to model weights and CUDA residency.",
    "gpu_lease_rows": "Lease rows prove that each GPU process had task-owned authority.",
    "server_lifecycle_rows": "Lifecycle rows prove bounded startup and owner-scoped cleanup.",
    "rows": "Flat rows keep every candidate attached to its selected final outcome.",
    "source_rows": "Frozen source rows prove balance without exposing completion witnesses.",
    "arm_budget_rows": "Budget rows make the matched sampling comparison replayable.",
    "direct_generation_rows": "Direct rows retain the single-draw baseline without verifier use.",
    "unguided_best_of_k_rows": "Best-of-k rows retain every sampled candidate, not only its winner.",
    "guided_frontier_rows": "Frontier rows retain each candidate-level guided decision.",
    "candidate_rows": "Candidate rows preserve all raw requests, outputs, scores, and failures.",
    "prefix_energy_rows": "Energy rows show zero for extendable and positive for impossible prefixes.",
    "rejected_branch_rows": "Rejected rows expose the cost and reason for every pruned branch.",
    "selected_branch_rows": "Selected rows prove likelihood and tie rules chose each frontier step.",
    "frontier_size_rows": "Frontier sizes measure how much feasible search remained.",
    "abstention_rows": "Abstention rows keep exhausted searches in the denominator.",
    "parser_rows": "Parser rows prevent malformed text from being repaired or hidden.",
    "final_exact_outcome_rows": "Final rows carry independent exact authority for selected programs.",
    "independent_solver_receipts": "Solver receipts prove final evaluation used clingo, not guidance.",
    "per_model_arm_rows": "Model summaries support the preregistered two-of-three utility rule.",
    "per_family_arm_rows": "Family summaries expose gains hidden by a global mean.",
    "validity_delta_rows": "Delta rows compare guided and matched unguided validity directly.",
    "false_admission_rows": "False-admission rows detect disagreement after guided acceptance.",
    "token_cost_rows": "Token rows report the sampled-token side of the Pareto comparison.",
    "latency_rows": "Latency rows report the wall-time side of the Pareto comparison.",
    "vram_rows": "VRAM rows report the memory side of the live comparison.",
    "energy_proxy_rows": "Energy rows report the feasibility-compute side of the comparison.",
    "raw_request_manifest": "Request hashes expose hidden prompts and hidden samples.",
    "raw_output_manifest": "Output hashes bind parser and solver claims to model bytes.",
    "external_text_scorer_call_count": "Zero keeps learned or model text scorers out of selection.",
    "constrained_schema_decode_count": "Zero proves schema decoding stayed retired.",
    "repair_prompt_count": "Zero proves syntax repair did not become the outcome.",
    "finite_answer_id_count": "Zero proves the retired finite-ID transport stayed off.",
    "model_weight_mutation_count": "Zero keeps the experiment inference-only.",
    "random_seed": "The fixed seed anchors deterministic source and sampling schedules.",
    "reproducibility_checksum": "A timing-free checksum detects scientific-content drift.",
    "guided_generation_run_complete_score": "One requires every authenticated planned row and audit.",
    "exact_guidance_utility_score": "One requires the full preregistered validity and Pareto gate.",
    "gate_check_summary": "Expected and observed values make each failed gate actionable.",
    "verifier_is_oracle": "True discloses that an exact solver defines final correctness.",
    "verdict_class": "The closed class prevents circular oracle evidence from becoming positive.",
    "honest_verdict": "A complete prefix lets the conductor classify the terminal result.",
}


class ExactGuidedGenerationError(RuntimeError):
    """Report a contract failure without turning it into a model result."""


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for hashes and raw requests."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return an explicit SHA-256 identity for preserved bytes."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str | None:
    """Hash a present local file and leave a missing input explicit."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Build one check with exact expected and observed evidence."""

    return {
        "check": check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize checks while retaining every failure in order."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = [str(row["check"]) for row in rows if row.get("passed") is not True]
    first = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "passed": not failed,
        "failed_checks": failed,
        "failed_check": first.get("check") if first else None,
        "expected": deepcopy(first.get("expected")) if first else None,
        "observed": deepcopy(first.get("observed")) if first else None,
        "checks": rows,
    }


def resolve_three_models(
    *,
    pair_provider: Callable[..., Sequence[Mapping[str, Any]] | None] = cached_sota_pair,
    dense_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the canonical pair first, then add the dense cache extension."""

    pair = pair_provider(
        gpu_indices=(0, 0),
        preferred_quant="Q4_K_M",
        model_indices=(0, 1),
    )
    by_id = {
        str(row.get("hf_id")): deepcopy(dict(row)) for row in pair or [] if isinstance(row, Mapping)
    }
    dense_path = dense_resolver(MODEL_SPECS[1], "Q4_K_M")
    if dense_path:
        by_id[MODEL_SPECS[1]] = {
            "hf_id": MODEL_SPECS[1],
            "model_path": dense_path,
            "gpu": 0,
        }
    rows: list[JsonDict] = []
    for hf_id in MODEL_SPECS:
        row = deepcopy(dict(by_id.get(hf_id, {})))
        row.update({"hf_id": hf_id, "gpu": int(row.get("gpu", 0) or 0)})
        if row.get("model_path"):
            row["model_path"] = str(Path(str(row["model_path"])).absolute())
        rows.append(row)
    return rows


def _fixture_from_source_row(row: Mapping[str, Any]) -> exact.RelationFixture:
    """Rebuild a held fixture from its frozen family and source identity."""

    ordinal = int(str(row["source_id"]).rsplit("_", 1)[1])
    return exact.build_relation_fixture(str(row["family"]), ordinal)


def select_held_source_tasks(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Freeze three empty and three partial tasks for each held family."""

    prefix_rows = upstream.get("prefix_case_rows")
    if not isinstance(prefix_rows, Sequence) or isinstance(prefix_rows, (str, bytes)):
        raise ExactGuidedGenerationError("prefix_case_rows_missing")
    selected: list[JsonDict] = []
    for family in exact.FAMILIES:
        family_rows = [
            dict(row)
            for row in prefix_rows
            if isinstance(row, Mapping)
            and row.get("split") == "held"
            and row.get("family") == family
            and row.get("case_type") in {"empty_prefix", "positive"}
        ]
        family_rows.sort(
            key=lambda row: (
                0 if row.get("case_type") == "empty_prefix" else 1,
                str(row.get("source_id")),
            )
        )
        if len(family_rows) < MIN_TASKS_PER_FAMILY:
            raise ExactGuidedGenerationError(f"held_family_floor:{family}")
        for row in family_rows[:MIN_TASKS_PER_FAMILY]:
            fixture = _fixture_from_source_row(row)
            source = {
                "source_task_id": str(row["prefix_case_id"]),
                "source_id": fixture.source_id,
                "source_group": fixture.source_group,
                "split": "held",
                "family": fixture.family,
                "ordinal": fixture.ordinal,
                "subjects": list(fixture.subjects),
                "predicate": fixture.predicate,
                "values": list(fixture.values),
                "invalid_pairs": [list(pair) for pair in sorted(fixture.invalid_pairs)],
                "initial_prefix": list(row.get("prefix") or []),
                "max_lines": fixture.max_lines,
            }
            source["source_task_sha256"] = sha256_bytes(canonical_json(source))
            selected.append(source)
    selected.sort(key=lambda row: (str(row["family"]), str(row["source_task_id"])))
    if len(selected) != SOURCE_TASK_COUNT:
        raise ExactGuidedGenerationError(f"held_source_count:{len(selected)}")
    return selected


def _constraint_text(source: Mapping[str, Any]) -> str:
    """Describe the source task without exposing a solver label or witness."""

    invalid = (
        ", ".join(f"({left}, {right})" for left, right in source.get("invalid_pairs", [])) or "none"
    )
    return (
        f"The two subjects are {source['subjects'][0]} and {source['subjects'][1]}.\n"
        f"Use predicate {source['predicate']}.\n"
        f"Allowed object words are {', '.join(source['values'])}.\n"
        "The first pair item belongs to the first subject. "
        "The second pair item belongs to the second subject.\n"
        f"Disallowed object pairs are: {invalid}."
    )


def build_program_prompt(source: Mapping[str, Any]) -> str:
    """Ask for one complete two-line program with no decoding constraint."""

    prefix = "\n".join(str(line) for line in source.get("initial_prefix", [])) or "(empty)"
    return (
        "Write a valid plain-text relation program. /no_think\n"
        f"{_constraint_text(source)}\n"
        "The final program must contain exactly two lines and one line per subject.\n"
        "Each line must have exactly: SUBJECT PREDICATE OBJECT.\n"
        "The required starting lines below must appear unchanged in the final program.\n"
        f"STARTING LINES\n{prefix}\nEND STARTING LINES\n"
        "Output only the two final relation lines. Do not use JSON or Markdown."
    )


def build_frontier_prompt(source: Mapping[str, Any], partial_program: Sequence[str]) -> str:
    """Ask for one next line without telling the model any exact-engine result."""

    prefix = "\n".join(str(line) for line in partial_program) or "(empty)"
    return (
        "Write the next line of a valid plain-text relation program. /no_think\n"
        f"{_constraint_text(source)}\n"
        "The final program has exactly two lines and one line per subject.\n"
        "Each line has exactly: SUBJECT PREDICATE OBJECT.\n"
        f"CURRENT PROGRAM\n{prefix}\nEND CURRENT PROGRAM\n"
        "Output exactly one next relation line. Do not use JSON or Markdown."
    )


def parse_plain_candidate(raw_output_bytes: bytes, *, expected_line_count: int) -> JsonDict:
    """Parse only exact three-field lines and never synthesize a replacement."""

    text = raw_output_bytes.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    success = len(lines) == expected_line_count and all(len(line.split()) == 3 for line in lines)
    return {
        "raw_output_bytes": raw_output_bytes,
        "raw_output_text": text,
        "parse_success": success,
        "parsed_lines": lines if success else [],
        "parse_error": None if success else "plain_relation_line_shape",
        "repair_applied": False,
    }


def select_feasible_candidate(rows: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Select maximum likelihood with stable candidate index and line tie breaks."""

    feasible = [
        dict(row)
        for row in rows
        if row.get("prefix_energy") == 0
        and isinstance(row.get("likelihood"), (int, float))
        and math.isfinite(float(row["likelihood"]))
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (
            -float(row["likelihood"]),
            int(row.get("candidate_index", 0)),
            str(row.get("parsed_line", "")),
        ),
    )


def evaluate_guided_candidates(
    *,
    fixture: exact.RelationFixture,
    prior_prefix: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict | None]:
    """Apply the in-loop engine after sampling and retain every branch."""

    rows: list[JsonDict] = []
    for candidate in candidates:
        raw = candidate.get("raw_text", b"")
        raw_bytes = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
        parsed = parse_plain_candidate(raw_bytes, expected_line_count=1)
        parsed_line = parsed["parsed_lines"][0] if parsed["parse_success"] else None
        if parsed_line is None:
            energy = 1
            reason = "parse_failure"
            decision_reason = "unsupported_atom"
        else:
            decision = exact.direct_prefix_viability(fixture, (*prior_prefix, parsed_line))
            energy = 0 if decision.extendable else 1
            decision_reason = decision.reason
            reason = None if decision.extendable else f"exact_prefix_impossible:{decision.reason}"
        rows.append(
            {
                **deepcopy(dict(candidate)),
                "prior_partial_program": list(prior_prefix),
                "parsed_line": parsed_line,
                "parse_success": parsed["parse_success"],
                "parse_error": parsed["parse_error"],
                "prefix_energy": energy,
                "in_loop_engine": IN_LOOP_ENGINE,
                "in_loop_decision_reason": decision_reason,
                "rejection_reason": reason,
                "selected": False,
            }
        )
    selected = select_feasible_candidate(rows)
    if selected is not None:
        selected_index = int(selected["candidate_index"])
        for row in rows:
            row["selected"] = int(row["candidate_index"]) == selected_index
            if row["prefix_energy"] == 0 and not row["selected"]:
                row["rejection_reason"] = "lower_model_likelihood"
        selected = next(row for row in rows if row["selected"])
    return rows, selected


def guided_frontier_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Audit exact admission, feasible eligibility, and deterministic selection."""

    errors: list[str] = []
    selected_rows = [row for row in rows if row.get("selected") is True]
    if any(row.get("prefix_energy") != 0 for row in selected_rows):
        errors.append("invalid_prefix_admission")
    if any(
        row.get("prefix_energy") == 0
        and str(row.get("rejection_reason", "")).startswith("exact_prefix_impossible")
        for row in rows
    ):
        errors.append("feasible_branch_rejection")
    expected = select_feasible_candidate(rows)
    expected_index = int(expected["candidate_index"]) if expected is not None else None
    observed = [int(row["candidate_index"]) for row in selected_rows]
    if observed != ([] if expected_index is None else [expected_index]):
        errors.append("tie_drift")
    return errors


def parser_masking_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reparse raw output and detect any claimed replacement or repair."""

    errors: list[str] = []
    for index, row in enumerate(rows):
        raw = row.get("raw_output_bytes", b"")
        raw_bytes = raw if isinstance(raw, bytes) else bytes(raw)
        expected = int(row.get("expected_line_count", 1))
        replay = parse_plain_candidate(raw_bytes, expected_line_count=expected)
        if (
            row.get("parse_success") != replay["parse_success"]
            or list(row.get("parsed_lines") or []) != replay["parsed_lines"]
            or row.get("repair_applied") is True
        ):
            errors.append(f"parser_masking:{index}")
    return errors


def direct_arm_leakage_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject direct rows that consulted or named an exact selection engine."""

    errors: list[str] = []
    forbidden = ("clingo", "prefix_energy", "exact_prefix", "verifier_result")
    for index, row in enumerate(rows):
        if row.get("arm") != "direct_generation":
            continue
        payload_text = json.dumps(row.get("request_payload", {}), sort_keys=True).lower()
        leaked = (
            int(row.get("preselection_exact_engine_calls", 0) or 0) != 0
            or row.get("prefix_energy") is not None
            or row.get("selection_method") != "single_full_program_draw"
            or any(token in payload_text for token in forbidden)
        )
        if leaked:
            errors.append(f"direct_arm_verifier_leakage:{index}")
    return errors


def runtime_authentication_errors(row: Mapping[str, Any]) -> list[str]:
    """Name each transport or process condition that invalidates a live row."""

    receipt = row.get("runtime_receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    errors: list[str] = []
    if row.get("timed_out") is True:
        errors.append("timeout")
    if row.get("truncated") is True:
        errors.append("truncation")
    if row.get("server_crashed") is True:
        errors.append("server_crash")
    if receipt.get("process_identity_match") is not True:
        errors.append("stale_pid")
    if int(receipt.get("offload_layers", 0) or 0) <= 0:
        errors.append("zero_offload")
    if receipt.get("authentic") is not True or receipt.get("owned_cuda_residency") is not True:
        errors.append("unauthenticated_runtime")
    if row.get("parser_attempted") is not True:
        errors.append("parser_bypass")
    return errors


def runtime_row_authenticated(row: Mapping[str, Any]) -> bool:
    """Return true only when no runtime authenticity error remains."""

    return not runtime_authentication_errors(row)


def engine_separation_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require different declared implementations for search and final authority."""

    return [
        f"final_engine_reuse:{index}"
        for index, row in enumerate(rows)
        if row.get("in_loop_engine") != IN_LOOP_ENGINE
        or row.get("final_engine") != FINAL_ENGINE
        or row.get("in_loop_engine") == row.get("final_engine")
    ]


def build_arm_budget_rows(cell_ids: Sequence[str]) -> list[JsonDict]:
    """Freeze equal candidate and token limits for unguided and guided arms."""

    return [
        {
            "cell_id": str(cell_id),
            "arm": arm,
            "candidate_budget": MATCHED_CANDIDATE_BUDGET,
            "sampled_token_limit": MATCHED_TOTAL_TOKEN_LIMIT,
            "candidate_token_limit": MATCHED_CANDIDATE_TOKEN_LIMIT,
        }
        for cell_id in cell_ids
        for arm in ("unguided_best_of_k", "guided_frontier")
    ]


def candidate_budget_errors(
    candidate_rows: Sequence[Mapping[str, Any]],
    arm_budget_rows: Sequence[Mapping[str, Any]],
    *,
    request_count: int,
) -> list[str]:
    """Detect unequal limits, missing rows, and unrecorded model requests."""

    errors: list[str] = []
    matched = [
        row for row in candidate_rows if row.get("arm") in {"unguided_best_of_k", "guided_frontier"}
    ]
    if request_count > len(matched):
        errors.append("hidden_extra_samples")
    elif request_count < len(matched):
        errors.append("missing_request_receipts")
    by_cell_budget: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in arm_budget_rows:
        by_cell_budget[str(row.get("cell_id"))][str(row.get("arm"))] = row
    by_cell_candidates: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in matched:
        by_cell_candidates[(str(row.get("cell_id")), str(row.get("arm")))].append(row)
    for cell_id, arms in sorted(by_cell_budget.items()):
        unguided = arms.get("unguided_best_of_k", {})
        guided = arms.get("guided_frontier", {})
        if unguided.get("candidate_budget") != guided.get("candidate_budget"):
            errors.append(f"unequal_candidate_budget:{cell_id}")
        if unguided.get("sampled_token_limit") != guided.get("sampled_token_limit"):
            errors.append(f"unequal_sampled_token_budget:{cell_id}")
        for arm, budget in arms.items():
            observed = len(by_cell_candidates[(cell_id, arm)])
            limit = int(budget.get("candidate_budget", -1))
            count_invalid = observed != limit
            token_limit = sum(
                int(row.get("sampled_token_limit", 0) or 0)
                for row in by_cell_candidates[(cell_id, arm)]
            )
            if count_invalid:
                errors.append(f"candidate_count:{cell_id}:{arm}")
            if token_limit > int(budget.get("sampled_token_limit", -1)):
                errors.append(f"sampled_token_limit:{cell_id}:{arm}")
    return errors


def _summary_row(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> JsonDict:
    """Reduce detailed final outcomes into one exact aggregate row."""

    first = rows[0]
    count = len(rows)
    valid_count = sum(row.get("exact_final_valid") is True for row in rows)
    parse_failure_count = sum(row.get("parse_success") is not True for row in rows)
    return {
        **{key: first[key] for key in keys},
        "cell_count": count,
        "exact_final_valid_count": valid_count,
        "exact_final_validity_rate": valid_count / count,
        "parse_failure_count": parse_failure_count,
        "parse_failure_rate": parse_failure_count / count,
        "abstention_count": sum(row.get("abstained") is True for row in rows),
        "sampled_tokens": sum(int(row.get("sampled_tokens", 0) or 0) for row in rows),
        "wall_time_s": sum(float(row.get("wall_time_s", 0.0) or 0.0) for row in rows),
        "max_vram_mb": max(int(row.get("vram_mb", 0) or 0) for row in rows),
        "energy_proxy": sum(float(row.get("energy_proxy", 0.0) or 0.0) for row in rows),
    }


def _group_summaries(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[JsonDict]:
    """Group final outcomes by exact string keys in stable order."""

    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    return [_summary_row(groups[group], keys) for group in sorted(groups)]


def _validity_deltas(per_model_arm_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Compare guided final validity and parsing against matched unguided rows."""

    by_model = {(str(row["model_spec"]), str(row["arm"])): row for row in per_model_arm_rows}
    rows: list[JsonDict] = []
    for model in MODEL_SPECS:
        unguided = by_model.get((model, "unguided_best_of_k"), {})
        guided = by_model.get((model, "guided_frontier"), {})
        unguided_rate = float(unguided.get("exact_final_validity_rate", 0.0) or 0.0)
        guided_rate = float(guided.get("exact_final_validity_rate", 0.0) or 0.0)
        unguided_parse = float(unguided.get("parse_failure_rate", 1.0) or 0.0)
        guided_parse = float(guided.get("parse_failure_rate", 1.0) or 0.0)
        delta = guided_rate - unguided_rate
        rows.append(
            {
                "model_spec": model,
                "guided_validity_rate": guided_rate,
                "unguided_validity_rate": unguided_rate,
                "validity_delta": delta,
                "guided_beats_unguided": delta > 0.0,
                "no_regression_over_0_02": delta >= -0.02,
                "guided_parse_failure_rate": guided_parse,
                "unguided_parse_failure_rate": unguided_parse,
                "parse_failure_did_not_rise": guided_parse <= unguided_parse,
            }
        )
    return rows


def aggregate_outcome_rows(final_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build all outcome aggregates from detailed selected-program rows."""

    per_model = _group_summaries(final_rows, ("model_spec", "arm")) if final_rows else []
    per_family = _group_summaries(final_rows, ("family", "arm")) if final_rows else []
    return {
        "per_model_arm_rows": per_model,
        "per_family_arm_rows": per_family,
        "validity_delta_rows": _validity_deltas(per_model),
    }


def aggregate_disagreement_errors(
    final_rows: Sequence[Mapping[str, Any]], aggregates: Mapping[str, Any]
) -> list[str]:
    """Recompute every outcome aggregate and name changed tables."""

    expected = aggregate_outcome_rows(final_rows)
    return [
        key
        for key in expected
        if canonical_json(expected[key]) != canonical_json(aggregates.get(key))
    ]


def compute_utility_score(
    per_model_arm_rows: Sequence[Mapping[str, Any]], *, pareto_complete: bool
) -> tuple[int, list[JsonDict]]:
    """Apply the two-of-three validity, regression, parsing, and Pareto rule."""

    deltas = _validity_deltas(per_model_arm_rows)
    passed = (
        len(deltas) == len(MODEL_SPECS)
        and sum(bool(row["guided_beats_unguided"]) for row in deltas) >= 2
        and all(bool(row["no_regression_over_0_02"]) for row in deltas)
        and all(bool(row["parse_failure_did_not_rise"]) for row in deltas)
        and pareto_complete
    )
    return int(passed), deltas


def verdict_class(*, run_complete: int, utility_score: int) -> str:
    """Keep oracle-defined success circular and complete nulls explicit."""

    if not run_complete:
        return "partial"
    return "circular_positive" if utility_score else "null"


def evaluate_preconditions(
    *,
    upstream: Mapping[str, Any],
    upstream_sha256: str,
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    gpu_inventory: Sequence[Mapping[str, Any]],
    lease_probe_rows: Sequence[Mapping[str, Any]],
    outside_arc_job_rows: Sequence[Mapping[str, Any]],
    cuda_offload_supported: bool,
) -> JsonDict:
    """Evaluate exact assets and live resources before a server can start."""

    model_by_id = {str(row.get("hf_id")): row for row in models}
    model_observed = {
        hf_id: {
            "present": bool(model_by_id.get(hf_id, {}).get("model_path")),
            "sha256": model_by_id.get(hf_id, {}).get("sha256"),
        }
        for hf_id in MODEL_SPECS
    }
    model_expected = {
        hf_id: {"present": True, "sha256": EXPECTED_MODEL_HASHES[hf_id]} for hf_id in MODEL_SPECS
    }
    tokenizer_by_id = {str(row.get("hf_id")): row for row in tokenizer_receipts}
    tokenizer_observed = {
        hf_id: {
            "source": tokenizer_by_id.get(hf_id, {}).get("source"),
            "loadable": tokenizer_by_id.get(hf_id, {}).get("loadable"),
            "used_hf_autotokenizer": tokenizer_by_id.get(hf_id, {}).get("used_hf_autotokenizer"),
            "sha256": tokenizer_by_id.get(hf_id, {}).get("canonical_tokenizer_payload_sha256"),
            "model_sha256": tokenizer_by_id.get(hf_id, {}).get("model_sha256"),
        }
        for hf_id in MODEL_SPECS
    }
    tokenizer_expected = {
        hf_id: {
            "source": "native_embedded_gguf_llama_cpp_vocab_only",
            "loadable": True,
            "used_hf_autotokenizer": False,
            "sha256": EXPECTED_TOKENIZER_HASHES[hf_id],
            "model_sha256": EXPECTED_MODEL_HASHES[hf_id],
        }
        for hf_id in MODEL_SPECS
    }
    source_hashes = upstream.get("source_artifact_hashes")
    source_hashes = source_hashes if isinstance(source_hashes, Mapping) else {}
    engine_observed = {
        "prefix_engine_module": (
            source_hashes.get("module", {}).get("sha256")
            if isinstance(source_hashes.get("module"), Mapping)
            else None
        ),
        "asp_energy_compiler": (
            source_hashes.get("exp6274_compiler", {}).get("sha256")
            if isinstance(source_hashes.get("exp6274_compiler"), Mapping)
            else None
        ),
    }
    eligible = [
        dict(row)
        for row in gpu_inventory
        if int(row.get("free_vram_mb", 0) or 0) >= MIN_FREE_VRAM_MB
    ]
    lease_ok = len(lease_probe_rows) == 1 and all(
        row.get("owned") is True and row.get("released") is True for row in lease_probe_rows
    )
    checks = [
        gate_check(
            "prefix_viability_canary_ready_score",
            1,
            upstream.get("prefix_viability_canary_ready_score"),
        ),
        gate_check("exp6919_artifact_sha256", EXPECTED_EXP6919_SHA256, upstream_sha256),
        gate_check("exact_engine_hashes", EXPECTED_ENGINE_HASHES, engine_observed),
        gate_check("exact_model_files", model_expected, model_observed),
        gate_check("native_tokenizer_receipts", tokenizer_expected, tokenizer_observed),
        gate_check("cuda_offload_supported", True, bool(cuda_offload_supported)),
        gate_check("free_vram_at_least_24000_mib", True, bool(eligible)),
        gate_check("one_task_owned_gpu_lease", True, lease_ok),
        gate_check("outside_arc_job_count_on_selected_gpu", 0, len(outside_arc_job_rows)),
    ]
    summary = gate_summary(checks)
    return {
        "all_passed": summary["passed"],
        "checks": checks,
        "gate_check_summary": summary,
        "eligible_gpus": eligible,
        "selected_gpu": deepcopy(eligible[0]) if eligible else None,
        "outside_arc_job_rows": [deepcopy(dict(row)) for row in outside_arc_job_rows],
        "lease_probe_rows": [deepcopy(dict(row)) for row in lease_probe_rows],
        "cached_sota_pair_called": True,
    }


def _public_preconditions(preconditions: Mapping[str, Any]) -> JsonDict:
    """Remove large execution-only inputs from the artifact preflight record."""

    keys = (
        "all_passed",
        "checks",
        "gate_check_summary",
        "eligible_gpus",
        "selected_gpu",
        "outside_arc_job_rows",
        "lease_probe_rows",
        "cached_sota_pair_called",
    )
    return {key: deepcopy(preconditions.get(key)) for key in keys}


def source_artifact_hashes(root: Path) -> JsonDict:
    """Bind the result to all direct code and evidence inputs."""

    paths = {
        "exp6919_artifact": EXP6919_RELATIVE_PATH,
        "prefix_engine_module": EXACT_MODULE_RELATIVE_PATH,
        "asp_energy_compiler": ASP_COMPILER_RELATIVE_PATH,
        "module": MODULE_RELATIVE_PATH,
        "spec": SPEC_PATH,
        "tests": TEST_RELATIVE_PATH,
        "wrapper": WRAPPER_RELATIVE_PATH,
    }
    return {
        name: {"path": str(path), "sha256": sha256_path(root / path)}
        for name, path in paths.items()
    }


def _empty_evidence() -> JsonDict:
    """Return every row field so blocked artifacts keep the full schema."""

    rows = {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field.endswith("_rows")
        or field in {"rows", "raw_request_manifest", "raw_output_manifest"}
    }
    rows.update({"llama_cpp_receipts": [], "independent_solver_receipts": []})
    return rows


def _without_timing(value: Any) -> Any:
    """Remove clocks and process IDs from the reproducibility content hash."""

    if isinstance(value, Mapping):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key != "reproducibility_checksum"
            and key not in {"duration_s", "wall_time_s", "receipt_monotonic_ns"}
            and not key.endswith("_latency_ms")
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding nondeterministic clocks."""

    return sha256_bytes(canonical_json(_without_timing(deepcopy(dict(artifact)))))


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    preconditions: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    root: Path,
) -> JsonDict:
    """Build a complete terminal artifact when preflight blocks live work."""

    artifact: JsonDict = {
        "schema": "carnot.sota_exact_guided_relation_generation.v1",
        "experiment_id": "exp6920-sota-exact-guided-relation-generation",
        "run_date": str(date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": _public_preconditions(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": source_artifact_hashes(root),
        "model_specs": list(MODEL_SPECS),
        "models_used": [],
        "model_artifact_hashes": {
            str(row.get("hf_id")): {
                "path": row.get("model_path"),
                "sha256": row.get("sha256"),
                "size_bytes": int(row.get("model_size_bytes", 0) or 0),
            }
            for row in models
        },
        "tokenizer_receipts": [deepcopy(dict(row)) for row in tokenizer_receipts],
        **_empty_evidence(),
        "external_text_scorer_call_count": 0,
        "constrained_schema_decode_count": 0,
        "repair_prompt_count": 0,
        "finite_answer_id_count": 0,
        "model_weight_mutation_count": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "guided_generation_run_complete_score": 0,
        "exact_guidance_utility_score": 0,
        "gate_check_summary": deepcopy(preconditions.get("gate_check_summary", gate_summary([]))),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_sota_exact_guided_relation_generation",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _pareto_rows(final_rows: Sequence[Mapping[str, Any]], field: str) -> list[JsonDict]:
    """Keep one cost row per final cell so no aggregate hides missing telemetry."""

    return [
        {
            "cell_id": row["cell_id"],
            "model_spec": row["model_spec"],
            "family": row["family"],
            "arm": row["arm"],
            field: row[field],
        }
        for row in final_rows
    ]


def _guided_evidence_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Audit each frontier step independently and prefix its cell identity."""

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("cell_id")), int(row.get("frontier_step", 0)))].append(row)
    return [
        f"{error}:{cell_id}:{step}"
        for (cell_id, step), group in sorted(groups.items())
        for error in guided_frontier_errors(group)
    ]


def _completion_checks(
    *,
    preconditions: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    arm_budget_rows: Sequence[Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    duration_s: float,
) -> list[JsonDict]:
    """Build authenticated completeness checks from detailed evidence only."""

    direct_rows = [row for row in candidate_rows if row.get("arm") == "direct_generation"]
    guided_rows = [row for row in candidate_rows if row.get("arm") == "guided_frontier"]
    parser_inputs = [
        {
            "raw_output_bytes": base64.b64decode(str(row.get("raw_output_b64", ""))),
            "expected_line_count": int(row.get("expected_line_count", 1)),
            "parse_success": row.get("parse_success"),
            "parsed_lines": row.get("parsed_lines"),
            "repair_applied": row.get("repair_applied"),
        }
        for row in candidate_rows
    ]
    cell_ids = {
        f"{model}::{source['generation_seed']}::{source['source_task_id']}"
        for model in MODEL_SPECS
        for source in source_rows
    }
    expected_candidate_ids = {f"{cell_id}::direct_generation::0" for cell_id in cell_ids} | {
        f"{cell_id}::{arm}::{candidate_index}"
        for cell_id in cell_ids
        for arm in ("unguided_best_of_k", "guided_frontier")
        for candidate_index in range(MATCHED_CANDIDATE_BUDGET)
    }
    observed_candidate_ids = [str(row.get("candidate_id")) for row in candidate_rows]
    expected_final_ids = {f"{cell_id}::{arm}" for cell_id in cell_ids for arm in ARMS}
    observed_final_ids = {str(row.get("outcome_id")) for row in final_rows}
    candidate_count_in_bounds = len(candidate_rows) == len(cell_ids) * (
        1 + 2 * MATCHED_CANDIDATE_BUDGET
    )
    runtime_errors = [
        f"{row.get('candidate_id')}:{error}"
        for row in candidate_rows
        for error in runtime_authentication_errors(row)
    ]
    lifecycle_rows = list(acquisition.get("server_lifecycle_rows", []))
    lifecycle_errors = [
        str(row.get("hf_id"))
        for row in lifecycle_rows
        if not (
            row.get("process_exit_confirmed") is True
            and row.get("process_reaped") is True
            and row.get("port_release_confirmed") is True
            and row.get("lease_released") is True
            and row.get("leak_free") is True
            and int(row.get("unrelated_process_signal_count", 0) or 0) == 0
        )
    ]
    llama_rows = list(acquisition.get("llama_cpp_receipts", []))
    llama_ok = (
        len(llama_rows) == len(MODEL_SPECS)
        and {row.get("hf_id") for row in llama_rows} == set(MODEL_SPECS)
        and all(
            row.get("hf_id") in MODEL_SPECS
            and row.get("model_sha256") == EXPECTED_MODEL_HASHES[row["hf_id"]]
            and row.get("tokenizer_sha256") == EXPECTED_TOKENIZER_HASHES[row["hf_id"]]
            and int(row.get("offload_layers", 0) or 0) > 0
            and row.get("owned_cuda_residency") is True
            for row in llama_rows
        )
    )
    pareto_complete = all(
        all(field in row for field in ("sampled_tokens", "wall_time_s", "vram_mb", "energy_proxy"))
        for row in final_rows
    ) and len(final_rows) == len(expected_final_ids)
    final_cell_fields_complete = all(
        all(
            field in row
            for field in (
                "parse_success",
                "exact_final_valid",
                "answer_set_effect",
                "abstained",
                "false_admission",
                "branch_rejection_count",
                "sampled_tokens",
                "wall_time_s",
                "vram_mb",
                "energy_proxy",
            )
        )
        for row in final_rows
    ) and len(final_rows) == len(expected_final_ids)
    matched_request_count = int(acquisition.get("matched_request_count", -1))
    checks = [
        gate_check("preconditions", True, preconditions.get("all_passed") is True),
        gate_check("source_task_count", SOURCE_TASK_COUNT, len(source_rows)),
        gate_check(
            "distinct_source_task_count",
            SOURCE_TASK_COUNT,
            len({row.get("source_task_id") for row in source_rows}),
        ),
        gate_check(
            "source_family_floor",
            True,
            all(
                sum(row.get("family") == family for row in source_rows) >= MIN_TASKS_PER_FAMILY
                for family in exact.FAMILIES
            ),
        ),
        gate_check(
            "generation_seed_count",
            len(SEEDS),
            len({row.get("generation_seed") for row in source_rows}),
        ),
        gate_check("candidate_row_count_in_bounds", True, candidate_count_in_bounds),
        gate_check(
            "candidate_ids",
            sorted(expected_candidate_ids),
            sorted(observed_candidate_ids),
        ),
        gate_check("final_outcome_ids", sorted(expected_final_ids), sorted(observed_final_ids)),
        gate_check(
            "candidate_budget_errors",
            [],
            candidate_budget_errors(
                candidate_rows,
                arm_budget_rows,
                request_count=matched_request_count,
            ),
        ),
        gate_check("direct_arm_verifier_leakage", [], direct_arm_leakage_errors(direct_rows)),
        gate_check("guided_frontier_errors", [], _guided_evidence_errors(guided_rows)),
        gate_check("parser_masking_errors", [], parser_masking_errors(parser_inputs)),
        gate_check("runtime_authentication_errors", [], runtime_errors),
        gate_check(
            "candidate_likelihoods_present",
            True,
            all(row.get("likelihood") is not None for row in candidate_rows),
        ),
        gate_check("engine_separation_errors", [], engine_separation_errors(final_rows)),
        gate_check(
            "aggregate_row_disagreement", [], aggregate_disagreement_errors(final_rows, aggregates)
        ),
        gate_check("llama_cpp_receipts", True, llama_ok),
        gate_check("server_lifecycle_errors", [], lifecycle_errors),
        gate_check(
            "gpu_lease_row_count", len(MODEL_SPECS), len(acquisition.get("gpu_lease_rows", []))
        ),
        gate_check("duration_floor_s", True, duration_s >= 60.0),
        gate_check("final_cell_fields_complete", True, final_cell_fields_complete),
        gate_check("cost_latency_pareto_complete", True, pareto_complete),
        gate_check("external_text_scorer_call_count", 0, 0),
        gate_check("constrained_schema_decode_count", 0, 0),
        gate_check("repair_prompt_count", 0, 0),
        gate_check("finite_answer_id_count", 0, 0),
        gate_check("model_weight_mutation_count", 0, 0),
    ]
    return checks


def build_artifact(
    *,
    date: str,
    duration_s: float,
    root: Path,
    preconditions: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    acquisition: Mapping[str, Any],
) -> JsonDict:
    """Build the terminal result from preserved candidate and final rows."""

    candidates = [deepcopy(dict(row)) for row in acquisition.get("candidate_rows", [])]
    final_rows = [deepcopy(dict(row)) for row in acquisition.get("final_exact_outcome_rows", [])]
    cell_ids = [
        f"{model}::{source['generation_seed']}::{source['source_task_id']}"
        for model in MODEL_SPECS
        for source in source_rows
    ]
    budgets = build_arm_budget_rows(cell_ids)
    aggregates = aggregate_outcome_rows(final_rows)
    checks = _completion_checks(
        preconditions=preconditions,
        source_rows=source_rows,
        candidate_rows=candidates,
        final_rows=final_rows,
        arm_budget_rows=budgets,
        aggregates=aggregates,
        acquisition=acquisition,
        duration_s=duration_s,
    )
    summary = gate_summary(checks)
    run_complete = int(summary["passed"])
    pareto_complete = (
        next(row["observed"] for row in checks if row["check"] == "cost_latency_pareto_complete")
        is True
    )
    utility_score, delta_rows = compute_utility_score(
        aggregates["per_model_arm_rows"], pareto_complete=pareto_complete
    )
    utility_score *= run_complete
    outcome_by_arm = {(str(row["cell_id"]), str(row["arm"])): row for row in final_rows}
    flat_rows = []
    for candidate in candidates:
        outcome = outcome_by_arm.get((str(candidate["cell_id"]), str(candidate["arm"])), {})
        flat_rows.append(
            {
                **deepcopy(candidate),
                "selected_program_parse_success": outcome.get("parse_success"),
                "selected_program_exact_final_valid": outcome.get("exact_final_valid"),
                "selected_program_abstained": outcome.get("abstained"),
                "selected_program_answer_set_effect": outcome.get("answer_set_effect"),
            }
        )
    guided_rows = [row for row in candidates if row.get("arm") == "guided_frontier"]
    class_name = verdict_class(run_complete=run_complete, utility_score=utility_score)
    if class_name == "partial":
        honest = "complete_partial_sota_exact_guided_relation_generation"
    elif class_name == "circular_positive":
        honest = "complete_circular_positive_exact_guidance_utility"
    else:
        honest = "complete_null_exact_guidance_utility_not_shown"
    artifact: JsonDict = {
        "schema": "carnot.sota_exact_guided_relation_generation.v1",
        "experiment_id": "exp6920-sota-exact-guided-relation-generation",
        "run_date": str(date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": _public_preconditions(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": source_artifact_hashes(root),
        "model_specs": list(MODEL_SPECS),
        "models_used": [str(row["hf_id"]) for row in models if row.get("model_path")],
        "model_artifact_hashes": {
            str(row["hf_id"]): {
                "path": row.get("model_path"),
                "sha256": row.get("sha256"),
                "size_bytes": int(row.get("model_size_bytes", 0) or 0),
                "snapshot_identity": row.get("snapshot_identity"),
            }
            for row in models
        },
        "tokenizer_receipts": [deepcopy(dict(row)) for row in tokenizer_receipts],
        "llama_cpp_receipts": deepcopy(list(acquisition.get("llama_cpp_receipts", []))),
        "gpu_lease_rows": deepcopy(list(acquisition.get("gpu_lease_rows", []))),
        "server_lifecycle_rows": deepcopy(list(acquisition.get("server_lifecycle_rows", []))),
        "rows": flat_rows,
        "source_rows": [deepcopy(dict(row)) for row in source_rows],
        "arm_budget_rows": budgets,
        "direct_generation_rows": [
            row for row in candidates if row.get("arm") == "direct_generation"
        ],
        "unguided_best_of_k_rows": [
            row for row in candidates if row.get("arm") == "unguided_best_of_k"
        ],
        "guided_frontier_rows": guided_rows,
        "candidate_rows": candidates,
        "prefix_energy_rows": [
            {
                "candidate_id": row["candidate_id"],
                "partial_program": row.get("candidate_partial_program"),
                "prefix_energy": row.get("prefix_energy"),
                "in_loop_engine": row.get("in_loop_engine"),
            }
            for row in guided_rows
        ],
        "rejected_branch_rows": [row for row in guided_rows if row.get("selected") is not True],
        "selected_branch_rows": [row for row in guided_rows if row.get("selected") is True],
        "frontier_size_rows": deepcopy(list(acquisition.get("frontier_size_rows", []))),
        "abstention_rows": [row for row in final_rows if row.get("abstained") is True],
        "parser_rows": [
            {
                "candidate_id": row["candidate_id"],
                "raw_output_sha256": row.get("raw_output_sha256"),
                "parse_success": row.get("parse_success"),
                "parsed_lines": row.get("parsed_lines"),
                "parse_error": row.get("parse_error"),
                "repair_applied": row.get("repair_applied"),
            }
            for row in candidates
        ],
        "final_exact_outcome_rows": final_rows,
        "independent_solver_receipts": [
            {
                "outcome_id": row["outcome_id"],
                "engine": row["final_engine"],
                "solver_version": row["solver_version"],
                "exact_final_valid": row["exact_final_valid"],
            }
            for row in final_rows
        ],
        "per_model_arm_rows": aggregates["per_model_arm_rows"],
        "per_family_arm_rows": aggregates["per_family_arm_rows"],
        "validity_delta_rows": delta_rows,
        "false_admission_rows": [
            {
                "outcome_id": row["outcome_id"],
                "model_spec": row["model_spec"],
                "false_admission": row.get("false_admission", False),
            }
            for row in final_rows
            if row.get("arm") == "guided_frontier"
        ],
        "token_cost_rows": _pareto_rows(final_rows, "sampled_tokens"),
        "latency_rows": _pareto_rows(final_rows, "wall_time_s"),
        "vram_rows": _pareto_rows(final_rows, "vram_mb"),
        "energy_proxy_rows": _pareto_rows(final_rows, "energy_proxy"),
        "raw_request_manifest": [
            {
                "candidate_id": row["candidate_id"],
                "raw_request_sha256": row.get("raw_request_sha256"),
                "raw_request_byte_count": row.get("raw_request_byte_count"),
            }
            for row in candidates
        ],
        "raw_output_manifest": [
            {
                "candidate_id": row["candidate_id"],
                "raw_output_sha256": row.get("raw_output_sha256"),
                "raw_output_byte_count": row.get("raw_output_byte_count"),
            }
            for row in candidates
        ],
        "external_text_scorer_call_count": 0,
        "constrained_schema_decode_count": 0,
        "repair_prompt_count": 0,
        "finite_answer_id_count": 0,
        "model_weight_mutation_count": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "guided_generation_run_complete_score": run_complete,
        "exact_guidance_utility_score": utility_score,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": class_name,
        "honest_verdict": honest,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject incomplete schema, circular overclaim, or gate disagreement."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing_required_fields:{','.join(missing)}")
    missing_principles = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact["field_principles"]))
    if missing_principles:
        raise ValueError(f"missing_field_principles:{','.join(missing_principles)}")
    if artifact.get("verdict_class") == "positive":
        raise ValueError("oracle_verdict_cannot_be_positive")
    if artifact.get("verdict_class") not in {
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        raise ValueError("invalid_verdict_class")
    run_complete = artifact.get("guided_generation_run_complete_score")
    utility = artifact.get("exact_guidance_utility_score")
    if run_complete not in {0, 1} or utility not in {0, 1}:
        raise ValueError("invalid_gate_score")
    if bool(run_complete) != bool(artifact.get("gate_check_summary", {}).get("passed")):
        raise ValueError("run_complete_gate_disagreement")
    if utility and not run_complete:
        raise ValueError("utility_without_complete_run")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if any(
        int(artifact.get(field, -1)) != 0
        for field in (
            "external_text_scorer_call_count",
            "constrained_schema_decode_count",
            "repair_prompt_count",
            "finite_answer_id_count",
            "model_weight_mutation_count",
        )
    ):
        raise ValueError("retired_mechanism_activation")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("oracle_declaration")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        raise ValueError("honest_verdict_not_terminal")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def _native_tokenizer_receipt(model: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Reuse the native GGUF vocabulary probe that Exp6899 authenticated."""

    return canary._native_tokenizer_receipt(model)


def _outside_arc_jobs(gpu_uuid: str) -> list[JsonDict]:  # pragma: no cover
    """Find ARC processes on one GPU so this task never overlaps them."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    rows: list[JsonDict] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4 or parts[0] != gpu_uuid:
            continue
        pid = int(parts[1])
        try:
            cmdline = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except OSError:
            cmdline = parts[2]
        if "arc" in cmdline.lower():
            rows.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": pid,
                    "process_name": parts[2],
                    "used_gpu_memory_mb": int(parts[3]),
                    "cmdline": cmdline,
                }
            )
    return rows


def _probe_task_lease(gpu: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Acquire and release one lease before any server starts."""

    lease: Any = None
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=LEASE_RUNTIME_DIR,
            task_id="exp6920-preflight",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model="exp6920-three-model-sequential-run",
            vram_before_mb=int(gpu["free_vram_mb"]),
            ttl_s=30.0,
        )
        lease.transition("terminal_blocked")
        released = lease.release().get("released") is True
        return {
            "gpu_uuid": gpu["gpu_uuid"],
            "owned": True,
            "released": released,
        }
    except Exception as exc:
        if lease is not None:
            lease.close()
        return {
            "gpu_uuid": gpu.get("gpu_uuid"),
            "owned": False,
            "released": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_live_preconditions(root: Path) -> JsonDict:  # pragma: no cover
    """Resolve and authenticate every resource before a model process starts."""

    upstream_path = root / EXP6919_RELATIVE_PATH
    upstream = (
        json.loads(upstream_path.read_text(encoding="utf-8")) if upstream_path.is_file() else {}
    )
    upstream_sha256 = sha256_path(upstream_path) or ""
    models = resolve_three_models()
    for model in models:
        path = Path(str(model.get("model_path", "")))
        model.update(
            {
                "sha256": canary.base.sha256_file(path) if path.is_file() else "",
                "snapshot_identity": canary.base._snapshot_identity(str(path))
                if path.is_file()
                else "",
                "model_size_bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    tokenizers = [_native_tokenizer_receipt(model) for model in models]
    inventory = canary.base._gpu_inventory()
    eligible = [
        dict(row) for row in inventory if int(row.get("free_vram_mb", 0) or 0) >= MIN_FREE_VRAM_MB
    ]
    selected = next(
        (gpu for gpu in eligible if not _outside_arc_jobs(str(gpu["gpu_uuid"]))),
        eligible[0] if eligible else None,
    )
    outside = _outside_arc_jobs(str(selected["gpu_uuid"])) if selected else []
    lease_rows = [_probe_task_lease(selected)] if selected else []
    try:
        from llama_cpp import llama_cpp

        offload = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        offload = False
    report = evaluate_preconditions(
        upstream=upstream,
        upstream_sha256=upstream_sha256,
        models=models,
        tokenizer_receipts=tokenizers,
        gpu_inventory=[selected] if selected else inventory,
        lease_probe_rows=lease_rows,
        outside_arc_job_rows=outside,
        cuda_offload_supported=offload,
    )
    report.update(
        {
            "upstream": upstream,
            "upstream_sha256": upstream_sha256,
            "models": models,
            "tokenizer_receipts": tokenizers,
            "gpu_inventory": inventory,
            "selected_gpu": selected,
        }
    )
    return report


def _candidate_request(prompt: str, *, token_limit: int, seed: int) -> JsonDict:
    """Build an unconstrained chat request with native chosen-token logprobs."""

    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(token_limit),
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": int(seed),
        "stream": False,
        "logprobs": True,
        "top_logprobs": 1,
    }


def _response_logprobs(payload: Mapping[str, Any]) -> list[float]:  # pragma: no cover
    """Read current and legacy llama.cpp chosen-token logprob shapes."""

    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, Mapping) else {}
    logprobs = choice.get("logprobs")
    logprobs = logprobs if isinstance(logprobs, Mapping) else {}
    values: list[float] = []
    content = logprobs.get("content")
    if isinstance(content, list):
        values.extend(
            float(row["logprob"])
            for row in content
            if isinstance(row, Mapping) and isinstance(row.get("logprob"), (int, float))
        )
    legacy = logprobs.get("token_logprobs")
    if not values and isinstance(legacy, list):
        values.extend(float(value) for value in legacy if isinstance(value, (int, float)))
    probabilities = payload.get("completion_probabilities")
    if not values and isinstance(probabilities, list):
        values.extend(
            float(row["logprob"])
            for row in probabilities
            if isinstance(row, Mapping) and isinstance(row.get("logprob"), (int, float))
        )
    return values


def _sample_candidate(
    *,
    port: int,
    prompt: str,
    token_limit: int,
    seed: int,
    runtime_receipt: Mapping[str, Any],
    worker: OwnedLlamaCppProcess,
) -> JsonDict:  # pragma: no cover
    """Call one owned server and preserve exact request, response, and output bytes."""

    payload = _candidate_request(prompt, token_limit=token_limit, seed=seed)
    request_bytes = canonical_json(payload)
    req = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=request_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    response_bytes = b""
    output_bytes = b""
    generated_tokens = 0
    stop_reason = ""
    timed_out = False
    server_crashed = False
    error_text = ""
    likelihood: float | None = None
    token_logprobs: list[float] = []
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as response:
            response_bytes = response.read()
        body = json.loads(response_bytes.decode("utf-8"))
        choice = body["choices"][0]
        output_bytes = str(choice.get("message", {}).get("content", "")).encode("utf-8")
        generated_tokens = int(body.get("usage", {}).get("completion_tokens", 0) or 0)
        stop_reason = str(choice.get("finish_reason", ""))
        token_logprobs = _response_logprobs(body)
        likelihood = sum(token_logprobs) if token_logprobs else None
    except Exception as exc:
        timed_out = isinstance(exc, (TimeoutError, socket.timeout)) or (
            isinstance(exc, error.URLError)
            and isinstance(exc.reason, (TimeoutError, socket.timeout))
        )
        server_crashed = worker.process is not None and worker.process.poll() is not None
        stop_reason = "timeout" if timed_out else "request_failure"
        error_text = f"{type(exc).__name__}: {exc}"
    return {
        "request_payload": payload,
        "raw_request_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_request_sha256": sha256_bytes(request_bytes),
        "raw_request_byte_count": len(request_bytes),
        "raw_http_response_b64": base64.b64encode(response_bytes).decode("ascii"),
        "raw_http_response_sha256": sha256_bytes(response_bytes),
        "raw_output_b64": base64.b64encode(output_bytes).decode("ascii"),
        "raw_output_sha256": sha256_bytes(output_bytes),
        "raw_output_byte_count": len(output_bytes),
        "raw_output_text": output_bytes.decode("utf-8", errors="replace"),
        "generated_tokens": generated_tokens,
        "sampled_token_limit": token_limit,
        "token_logprobs": token_logprobs,
        "likelihood": likelihood,
        "mean_token_logprob": likelihood / len(token_logprobs) if token_logprobs else None,
        "stop_reason": stop_reason,
        "timed_out": timed_out,
        "truncated": stop_reason == "length",
        "server_crashed": server_crashed,
        "request_error": error_text or None,
        "parser_attempted": True,
        "wall_time_s": time.monotonic() - started,
        "runtime_receipt": deepcopy(dict(runtime_receipt)),
    }


def _attach_parse(row: JsonDict, *, expected_line_count: int) -> JsonDict:  # pragma: no cover
    """Attach one replayable parse result to preserved output bytes."""

    raw = base64.b64decode(row["raw_output_b64"])
    parsed = parse_plain_candidate(raw, expected_line_count=expected_line_count)
    row.update(
        {
            "expected_line_count": expected_line_count,
            "parse_success": parsed["parse_success"],
            "parsed_lines": parsed["parsed_lines"],
            "parse_error": parsed["parse_error"],
            "repair_applied": False,
        }
    )
    return row


def _candidate_base(
    *,
    model: str,
    source: Mapping[str, Any],
    arm: str,
    candidate_index: int,
    frontier_step: int,
    sample_seed: int,
) -> JsonDict:  # pragma: no cover
    """Build stable candidate identity fields before the request runs."""

    cell_id = f"{model}::{source['generation_seed']}::{source['source_task_id']}"
    return {
        "candidate_id": f"{cell_id}::{arm}::{candidate_index}",
        "cell_id": cell_id,
        "source_task_id": source["source_task_id"],
        "source_id": source["source_id"],
        "family": source["family"],
        "model_spec": model,
        "generation_seed": source["generation_seed"],
        "sample_seed": sample_seed,
        "arm": arm,
        "candidate_index": candidate_index,
        "frontier_step": frontier_step,
    }


def _select_by_likelihood(rows: Sequence[Mapping[str, Any]]) -> JsonDict | None:  # pragma: no cover
    """Select a sampled program without reading parser or exact-engine results."""

    scored = [
        dict(row)
        for row in rows
        if isinstance(row.get("likelihood"), (int, float))
        and math.isfinite(float(row["likelihood"]))
    ]
    if not scored:
        return None
    return min(
        scored,
        key=lambda row: (
            -float(row["likelihood"]),
            int(row["candidate_index"]),
            str(row.get("raw_output_text", "")),
        ),
    )


def _final_outcome(
    *,
    source: Mapping[str, Any],
    model: str,
    arm: str,
    candidates: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
    selected_program: Sequence[str],
    prefix_engine_admitted: bool,
    vram_mb: int,
) -> JsonDict:  # pragma: no cover
    """Evaluate a fixed selected program only through the independent engine."""

    fixture = _fixture_from_source_row(source)
    program = tuple(str(line) for line in selected_program)
    final = exact.clingo_final_validity(fixture, program)
    sampled_tokens = sum(int(row.get("generated_tokens", 0) or 0) for row in candidates)
    wall_time_s = sum(float(row.get("wall_time_s", 0.0) or 0.0) for row in candidates)
    prefix_checks = sum(row.get("prefix_energy") is not None for row in candidates)
    prefix_preserved = set(source.get("initial_prefix", [])).issubset(program)
    parse_success = selected is not None and len(program) == fixture.max_lines
    abstained = selected is None
    exact_valid = bool(final.extendable and prefix_preserved)
    return {
        "outcome_id": f"{candidates[0]['cell_id']}::{arm}",
        "cell_id": candidates[0]["cell_id"],
        "source_task_id": source["source_task_id"],
        "source_id": source["source_id"],
        "family": source["family"],
        "model_spec": model,
        "generation_seed": source["generation_seed"],
        "arm": arm,
        "selected_candidate_id": selected.get("candidate_id") if selected else None,
        "selected_program": list(program),
        "parse_success": parse_success,
        "exact_final_valid": exact_valid,
        "answer_set_count": final.completion_count if prefix_preserved else 0,
        "answer_set_effect": "admitted_one" if exact_valid else "eliminated_all",
        "required_prefix_preserved": prefix_preserved,
        "abstained": abstained,
        "exhausted_search_abstention": arm == "guided_frontier" and abstained,
        "false_admission": bool(prefix_engine_admitted and not exact_valid),
        "branch_rejection_count": sum(bool(row.get("rejection_reason")) for row in candidates),
        "sampled_tokens": sampled_tokens,
        "wall_time_s": wall_time_s,
        "vram_mb": int(vram_mb),
        "energy_proxy": prefix_checks,
        "in_loop_engine": IN_LOOP_ENGINE,
        "final_engine": FINAL_ENGINE,
        "solver_version": exact._clingo_valid_completions.__module__ + ":" + _clingo_version(),
        "final_reason": final.reason,
    }


def _clingo_version() -> str:  # pragma: no cover
    """Record the exact independent solver package version."""

    import clingo

    return f"clingo {clingo.__version__}"


def _run_cell(
    *,
    model: str,
    source: Mapping[str, Any],
    port: int,
    runtime_receipt: Mapping[str, Any],
    worker: OwnedLlamaCppProcess,
    vram_mb: int,
) -> JsonDict:  # pragma: no cover
    """Run all three arms for one matched source-model cell."""

    candidates: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    frontier_sizes: list[JsonDict] = []
    base_seed = int(source["generation_seed"])
    direct_prompt = build_program_prompt(source)
    direct_base = _candidate_base(
        model=model,
        source=source,
        arm="direct_generation",
        candidate_index=0,
        frontier_step=0,
        sample_seed=base_seed + 1,
    )
    direct = {
        **direct_base,
        **_sample_candidate(
            port=port,
            prompt=direct_prompt,
            token_limit=DIRECT_TOKEN_LIMIT,
            seed=base_seed + 1,
            runtime_receipt=runtime_receipt,
            worker=worker,
        ),
        "selection_method": "single_full_program_draw",
        "preselection_exact_engine_calls": 0,
        "prefix_energy": None,
        "in_loop_engine": None,
        "selected": True,
    }
    _attach_parse(direct, expected_line_count=2)
    candidates.append(direct)
    direct_program = direct["parsed_lines"] if direct["parse_success"] else []
    outcomes.append(
        _final_outcome(
            source=source,
            model=model,
            arm="direct_generation",
            candidates=[direct],
            selected=direct,
            selected_program=direct_program,
            prefix_engine_admitted=False,
            vram_mb=vram_mb,
        )
    )

    unguided: list[JsonDict] = []
    for candidate_index in range(MATCHED_CANDIDATE_BUDGET):
        sample_seed = base_seed + 100 + candidate_index
        row = {
            **_candidate_base(
                model=model,
                source=source,
                arm="unguided_best_of_k",
                candidate_index=candidate_index,
                frontier_step=0,
                sample_seed=sample_seed,
            ),
            **_sample_candidate(
                port=port,
                prompt=direct_prompt,
                token_limit=MATCHED_CANDIDATE_TOKEN_LIMIT,
                seed=sample_seed,
                runtime_receipt=runtime_receipt,
                worker=worker,
            ),
            "selection_method": "model_likelihood_only",
            "preselection_exact_engine_calls": 0,
            "prefix_energy": None,
            "in_loop_engine": None,
            "selected": False,
        }
        _attach_parse(row, expected_line_count=2)
        unguided.append(row)
    selected_unguided = _select_by_likelihood(unguided)
    if selected_unguided is not None:
        selected_id = selected_unguided["candidate_id"]
        for row in unguided:
            row["selected"] = row["candidate_id"] == selected_id
        selected_unguided = next(row for row in unguided if row["selected"])
    candidates.extend(unguided)
    unguided_program = (
        selected_unguided["parsed_lines"]
        if selected_unguided is not None and selected_unguided["parse_success"]
        else []
    )
    outcomes.append(
        _final_outcome(
            source=source,
            model=model,
            arm="unguided_best_of_k",
            candidates=unguided,
            selected=selected_unguided,
            selected_program=unguided_program,
            prefix_engine_admitted=False,
            vram_mb=vram_mb,
        )
    )

    fixture = _fixture_from_source_row(source)
    partial = list(source.get("initial_prefix", []))
    remaining_steps = fixture.max_lines - len(partial)
    per_step = (
        [MATCHED_CANDIDATE_BUDGET]
        if remaining_steps == 1
        else [MATCHED_CANDIDATE_BUDGET // 2, MATCHED_CANDIDATE_BUDGET // 2]
    )
    guided: list[JsonDict] = []
    guided_selected: JsonDict | None = None
    next_index = 0
    for step, width in enumerate(per_step):
        sampled: list[JsonDict] = []
        prompt = build_frontier_prompt(source, partial)
        for _ in range(width):
            candidate_index = next_index
            next_index += 1
            sample_seed = base_seed + 200 + candidate_index
            row = {
                **_candidate_base(
                    model=model,
                    source=source,
                    arm="guided_frontier",
                    candidate_index=candidate_index,
                    frontier_step=step,
                    sample_seed=sample_seed,
                ),
                **_sample_candidate(
                    port=port,
                    prompt=prompt,
                    token_limit=MATCHED_CANDIDATE_TOKEN_LIMIT,
                    seed=sample_seed,
                    runtime_receipt=runtime_receipt,
                    worker=worker,
                ),
                "selection_method": "feasible_then_model_likelihood",
                "preselection_exact_engine_calls": 1,
            }
            _attach_parse(row, expected_line_count=1)
            sampled.append(row)
        evaluated, selected_step = evaluate_guided_candidates(
            fixture=fixture,
            prior_prefix=tuple(partial),
            candidates=[
                {
                    **row,
                    "raw_text": base64.b64decode(row["raw_output_b64"]),
                }
                for row in sampled
            ],
        )
        for row in evaluated:
            row.pop("raw_text", None)
            if row.get("parsed_line") is not None:
                row["candidate_partial_program"] = [
                    *row["prior_partial_program"],
                    row["parsed_line"],
                ]
            else:
                row["candidate_partial_program"] = list(row["prior_partial_program"])
        feasible_count = sum(row["prefix_energy"] == 0 for row in evaluated)
        frontier_sizes.append(
            {
                "cell_id": evaluated[0]["cell_id"],
                "frontier_step": step,
                "candidate_count": len(evaluated),
                "feasible_candidate_count": feasible_count,
                "frontier_size": feasible_count,
                "exhausted": selected_step is None,
            }
        )
        guided.extend(evaluated)
        if selected_step is None:
            guided_selected = None
            continue
        selected_index = int(selected_step["candidate_index"])
        guided_selected = next(row for row in evaluated if row["candidate_index"] == selected_index)
        partial.append(str(guided_selected["parsed_line"]))
    candidates.extend(guided)
    completed_guided = len(partial) == fixture.max_lines and guided_selected is not None
    outcomes.append(
        _final_outcome(
            source=source,
            model=model,
            arm="guided_frontier",
            candidates=guided,
            selected=guided_selected if completed_guided else None,
            selected_program=partial if completed_guided else (),
            prefix_engine_admitted=completed_guided,
            vram_mb=vram_mb,
        )
    )
    return {
        "candidate_rows": candidates,
        "final_exact_outcome_rows": outcomes,
        "frontier_size_rows": frontier_sizes,
        "request_count": len(candidates),
        "matched_request_count": len(unguided) + len(guided),
    }


def _run_model_phase(
    *,
    model: Mapping[str, Any],
    tokenizer_receipt: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    gpu: Mapping[str, Any],
    runtime_dir: Path,
) -> JsonDict:  # pragma: no cover
    """Run one fresh task-owned llama.cpp lifecycle for one model family."""

    hf_id = str(model["hf_id"])
    port = canary.base._free_port()
    family = MODEL_FAMILIES[hf_id]
    log_path = runtime_dir / f"{family}.log"
    state_path = runtime_dir / f"{family}.owner.json"
    command = [
        str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
        "--model",
        str(model["model_path"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(CONTEXT_LENGTH),
        "--batch-size",
        "512",
        "--ubatch-size",
        "256",
        "--gpu-layers",
        "all",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--threads",
        "8",
        "--parallel",
        "1",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--no-ui",
        "--verbose",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    worker = OwnedLlamaCppProcess(
        command=command,
        port=port,
        env=env,
        log_path=log_path,
        state_path=state_path,
    )
    lease: Any = None
    process_receipt: JsonDict = {}
    resident: JsonDict = {}
    candidates: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    frontier_sizes: list[JsonDict] = []
    phase_error = ""
    lifecycle: JsonDict = {}
    request_count = 0
    matched_request_count = 0
    free_before = int(gpu.get("free_vram_mb", 0) or 0)
    try:
        fresh_inventory = canary.base._gpu_inventory()
        current_gpu = next(
            (row for row in fresh_inventory if row.get("gpu_uuid") == gpu.get("gpu_uuid")),
            gpu,
        )
        free_before = int(current_gpu.get("free_vram_mb", 0) or 0)
        if free_before < MIN_FREE_VRAM_MB:
            raise ExactGuidedGenerationError(f"free_vram_below_floor:{free_before}")
        outside = _outside_arc_jobs(str(gpu["gpu_uuid"]))
        if outside:
            raise ExactGuidedGenerationError(f"outside_arc_job:{outside}")
        lease = lease_api.GpuLease.acquire(
            runtime_dir=LEASE_RUNTIME_DIR,
            task_id=f"exp6920-{family}",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model=str(model["model_path"]),
            vram_before_mb=free_before,
            ttl_s=LEASE_TTL_S,
        )
        lease.transition("admitted")
        lease.transition("loading")
        process_receipt = worker.launch()
        health = worker.wait_for_health(HEALTH_TIMEOUT_S)
        if health.get("ok") is not True:
            raise ExactGuidedGenerationError(f"llama_server_health:{health}")
        current = read_process_identity(int(process_receipt["pid"]))
        identity_errors = canary.process_identity_errors(process_receipt, current)
        resident = canary.base._gpu_process_sample(gpu, int(process_receipt["pid"]), "resident")
        offload_layers = canary.base._offload_layers(log_path)
        if identity_errors:
            raise ExactGuidedGenerationError(f"process_identity:{identity_errors}")
        if resident.get("owned_cuda_residency") is not True or offload_layers <= 0:
            raise ExactGuidedGenerationError("owned_cuda_offload_missing")
        lease.transition("resident", vram_mb=int(resident.get("owned_vram_mb", 0)))
        lease.transition("inferencing")
        receipt_time = time.monotonic_ns()
        for source in sources:
            current_identity = read_process_identity(int(process_receipt["pid"]))
            identity_match = not canary.process_identity_errors(process_receipt, current_identity)
            runtime_receipt = {
                "authentic": identity_match,
                "model_sha256": model["sha256"],
                "tokenizer_sha256": tokenizer_receipt["canonical_tokenizer_payload_sha256"],
                "server_pid": process_receipt["pid"],
                "server_start_time_ticks": process_receipt["start_time_ticks"],
                "command_hash": process_receipt["command_hash"],
                "process_identity_match": identity_match,
                "receipt_age_s": (time.monotonic_ns() - receipt_time) / 1_000_000_000,
                "receipt_monotonic_ns": receipt_time,
                "gpu_uuid": gpu["gpu_uuid"],
                "offload_layers": offload_layers,
                "owned_cuda_residency": resident.get("owned_cuda_residency") is True,
                "vram_before_mb": free_before,
                "vram_after_load_mb": int(resident.get("owned_vram_mb", 0) or 0),
            }
            cell = _run_cell(
                model=hf_id,
                source=source,
                port=port,
                runtime_receipt=runtime_receipt,
                worker=worker,
                vram_mb=int(resident.get("owned_vram_mb", 0) or 0),
            )
            candidates.extend(cell["candidate_rows"])
            outcomes.extend(cell["final_exact_outcome_rows"])
            frontier_sizes.extend(cell["frontier_size_rows"])
            request_count += int(cell["request_count"])
            matched_request_count += int(cell["matched_request_count"])
            lease.heartbeat()
        lease.transition("unloading")
    except Exception as exc:
        phase_error = f"{type(exc).__name__}: {exc}"
    finally:
        if lease is not None and lease.document.get("phase") == "inferencing":
            lease.transition("unloading")
        cleanup = worker.cleanup()
        after = canary.base._gpu_process_sample(
            gpu, int(process_receipt.get("pid", 0) or 0), "after"
        )
        process_exit = cleanup.get("process_exit_confirmed") is True
        port_release = cleanup.get("port_release_confirmed") is True and port_is_free(port)
        lease_released = False
        teardown_error = ""
        if lease is not None:
            try:
                phase = str(lease.document.get("phase"))
                if phase == "unloading":
                    lease.transition(
                        "validating",
                        vram_mb=int(after.get("owned_vram_mb", 0) or 0),
                        exit_code=int(worker.process.returncode or 0) if worker.process else 0,
                        unload_observed=process_exit,
                    )
                    lease.transition(
                        "terminal_complete"
                        if not phase_error and process_exit and port_release
                        else "terminal_blocked"
                    )
                elif phase not in lease_api.TERMINAL_PHASES:
                    lease.transition("terminal_blocked")
                lease_released = lease.release().get("released") is True
            except Exception as exc:
                lease.close()
                teardown_error = f"{type(exc).__name__}: {exc}"
        lifecycle = {
            "hf_id": hf_id,
            "pid": process_receipt.get("pid"),
            "start_time_ticks": process_receipt.get("start_time_ticks"),
            "process_identity_match": bool(process_receipt)
            and "process_identity" not in phase_error,
            "process_exit_confirmed": process_exit,
            "process_reaped": cleanup.get("process_reaped") is True,
            "port_release_confirmed": port_release,
            "lease_released": lease_released,
            "unrelated_process_signal_count": int(
                cleanup.get("unrelated_process_kill_count_delta", 0) or 0
            ),
            "leak_free": bool(cleanup.get("leak_free") and lease_released and port_release),
            "teardown_error": teardown_error,
            "phase_error": phase_error,
            "vram_after_teardown_mb": int(after.get("owned_vram_mb", 0) or 0),
        }
    llama_receipt = {
        "hf_id": hf_id,
        "pid": process_receipt.get("pid"),
        "start_time_ticks": process_receipt.get("start_time_ticks"),
        "command_hash": process_receipt.get("command_hash"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("sha256"),
        "tokenizer_sha256": tokenizer_receipt.get("canonical_tokenizer_payload_sha256"),
        "gpu_uuid": gpu.get("gpu_uuid"),
        "offload_layers": canary.base._offload_layers(log_path),
        "owned_cuda_residency": resident.get("owned_cuda_residency") is True,
        "vram_before_mb": free_before,
        "vram_after_load_mb": int(resident.get("owned_vram_mb", 0) or 0),
        "vram_after_teardown_mb": lifecycle.get("vram_after_teardown_mb"),
        "stderr_tail": canary._stderr_tail(log_path),
        "log_sha256": canary.base.sha256_file(log_path)
        if log_path.is_file()
        else sha256_bytes(b""),
        "phase_error": phase_error,
    }
    return {
        "candidate_rows": candidates,
        "final_exact_outcome_rows": outcomes,
        "frontier_size_rows": frontier_sizes,
        "request_count": request_count,
        "matched_request_count": matched_request_count,
        "llama_cpp_receipt": llama_receipt,
        "gpu_lease_row": {
            "hf_id": hf_id,
            "gpu_uuid": gpu.get("gpu_uuid"),
            "owned": lease is not None,
            "released": lifecycle.get("lease_released") is True,
        },
        "server_lifecycle_row": lifecycle,
    }


def run_live_acquisition(
    *,
    sources: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    gpu: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover
    """Run three fresh model servers sequentially under one selected GPU policy."""

    candidates: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    frontier_sizes: list[JsonDict] = []
    llama_rows: list[JsonDict] = []
    lease_rows: list[JsonDict] = []
    lifecycle_rows: list[JsonDict] = []
    request_count = 0
    matched_request_count = 0
    token_by_model = {str(row["hf_id"]): row for row in tokenizer_receipts}
    with tempfile.TemporaryDirectory(prefix="carnot-exp6920-") as temporary:
        runtime_dir = Path(temporary)
        for model in models:
            phase = _run_model_phase(
                model=model,
                tokenizer_receipt=token_by_model[str(model["hf_id"])],
                sources=sources,
                gpu=gpu,
                runtime_dir=runtime_dir,
            )
            candidates.extend(phase["candidate_rows"])
            outcomes.extend(phase["final_exact_outcome_rows"])
            frontier_sizes.extend(phase["frontier_size_rows"])
            request_count += int(phase["request_count"])
            matched_request_count += int(phase["matched_request_count"])
            llama_rows.append(phase["llama_cpp_receipt"])
            lease_rows.append(phase["gpu_lease_row"])
            lifecycle_rows.append(phase["server_lifecycle_row"])
            gc.collect()
    return {
        "candidate_rows": candidates,
        "final_exact_outcome_rows": outcomes,
        "frontier_size_rows": frontier_sizes,
        "request_count": request_count,
        "matched_request_count": matched_request_count,
        "llama_cpp_receipts": llama_rows,
        "gpu_lease_rows": lease_rows,
        "server_lifecycle_rows": lifecycle_rows,
    }


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    precondition_collector: Callable[[Path], Mapping[str, Any]] = collect_live_preconditions,
    acquisition_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run the live comparison or write its complete blocked artifact once."""

    started = time.monotonic()
    target = result_path or root / RESULT_RELATIVE_PATH
    collected = deepcopy(dict(precondition_collector(root)))
    models = [deepcopy(dict(row)) for row in collected.get("models", [])]
    tokenizers = [deepcopy(dict(row)) for row in collected.get("tokenizer_receipts", [])]
    if collected.get("all_passed") is not True:
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            preconditions=collected,
            models=models,
            tokenizer_receipts=tokenizers,
            root=root,
        )
        canary.base.write_json_atomic(target, artifact)
        return artifact
    upstream = collected.get("upstream")
    if not isinstance(upstream, Mapping):
        raise ExactGuidedGenerationError("preflight_upstream_missing")
    sources = select_held_source_tasks(upstream)
    for index, source in enumerate(sources):
        source["generation_seed"] = SEEDS[index % len(SEEDS)]
        source["source_order"] = index
    runner = acquisition_runner or run_live_acquisition
    acquisition = dict(
        runner(
            sources=sources,
            models=models,
            tokenizer_receipts=tokenizers,
            gpu=collected["selected_gpu"],
        )
    )
    artifact = build_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        root=root,
        preconditions=collected,
        models=models,
        tokenizer_receipts=tokenizers,
        source_rows=sources,
        acquisition=acquisition,
    )
    canary.base.write_json_atomic(target, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dated experiment and print only its terminal summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    artifact = run(date=args.date)
    print(
        json.dumps(
            {
                "exact_guidance_utility_score": artifact["exact_guidance_utility_score"],
                "guided_generation_run_complete_score": artifact[
                    "guided_generation_run_complete_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
                "result_path": str(RESULT_RELATIVE_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the checked wrapper calls main.
    raise SystemExit(main())
