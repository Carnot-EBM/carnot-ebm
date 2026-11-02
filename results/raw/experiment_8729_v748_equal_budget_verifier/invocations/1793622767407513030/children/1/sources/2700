"""Build a frozen three-family bank with exact instance-level labels.

Spec refs: REQ-VERIFY-7129 and SCENARIO-VERIFY-7129-*.

The bank keeps model generation separate from exact authority. Models produce
direct text. A small exhaustive checker then evaluates that text against the
specific SAT, coloring, or scheduling instance. Solver work is retained for
stratification only; it is not a label for model difficulty.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6966_gguf_load_envelope_canary import (
    build_vram_release_row,
    embedded_tokenizer_probe,
    gpu_inventory,
    llama_cpp_probe,
)
from carnot.experiment_7080_v620_three_family_entrance_bank import _lease_probe
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair
from carnot.task_runtime_receipts import sha256_file, write_json_atomic


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = "carnot.experiment_7129_v626_sota_constraint_bank"
EXPERIMENT_ID = "experiment_7129_v626_sota_constraint_bank"
SCHEMA = "carnot.experiment_7129.v626_sota_constraint_bank.v1"
RUN_DATE = "20260908"
RANDOM_SEED = 7_129_202_609_08
RESULT_PATH = REPO_ROOT / "results/experiment_7129_v626_sota_constraint_bank.json"
FIXTURE_PATH = REPO_ROOT / "results/fixtures/experiment_7129_v626_constraint_bank.json"
RAW_DIR = REPO_ROOT / "results/raw/experiment_7129_v626_sota_constraint_bank"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
INFERENCE_SUBSTRATE = "model_bounded_generation: three-family exact constraint bank"
PREFERRED_QUANT = "Q4_K_M"
SURFACE_BUDGET_CHARS = 1_200
MODEL_TIMEOUT_S = 7_200.0
VRAM_RELEASE_TIMEOUT_S = 180.0
VRAM_RELEASE_TOLERANCE_MB = 512

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SOTA_REGISTRY_IDS = tuple(row["hf_id"] for row in SOTA_GGUF_MODELS)
CONSTRAINT_FAMILIES = ("sat_logic", "graph_coloring", "bounded_scheduling")
VARIANT_KINDS = ("canonical", "relabel", "paraphrase")

GENERATION_CONFIG: JsonDict = {
    "n_ctx": 2_048,
    "n_gpu_layers": -1,
    "n_batch": 512,
    "n_ubatch": 512,
    "main_gpu": 0,
    "tensor_split": [0.5, 0.5],
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 96,
    "seed_policy": "frozen_per_cell",
    "chat_template": "embedded_gguf_auto",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "MODEL_SPECS",
    "models_used",
    "model_repository_rows",
    "model_path_rows",
    "model_hash_rows",
    "model_quantization_rows",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "gpu_telemetry_rows",
    "token_rows",
    "duration_s",
    "source_artifact_hashes",
    "raw_trace_manifest",
    "rows",
    "base_instance_rows",
    "variant_rows",
    "solver_receipt_rows",
    "model_output_rows",
    "parse_rows",
    "exact_outcome_rows",
    "family_rows",
    "hardness_stratum_rows",
    "relabel_sensitivity_rows",
    "paraphrase_consistency_rows",
    "model_identity_confound_rows",
    "planned_cell_count",
    "completed_cell_count",
    "finite_answer_id_transport_used",
    "schema_constraintir_reprompt_used",
    "sota_constraint_bank_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for every required field makes silent evidence loss visible.",
    "preconditions_checked": "Exact observed gates prevent fabricated model or solver evidence.",
    "run_date": "The requested execution date distinguishes this frozen acquisition from later runs.",
    "MODEL_SPECS": "The ordered declarations prevent a smaller or different model from substituting.",
    "models_used": "The invoked roster exposes any missing family.",
    "model_repository_rows": "Repository identities bind local files to the mandated public families.",
    "model_path_rows": "Absolute paths make the exact cached files auditable.",
    "model_hash_rows": "Full file hashes prevent silent weight replacement.",
    "model_quantization_rows": "Quantization records keep compute conditions comparable.",
    "inference_substrate": "The substrate states that bounded local model generation supplied outputs.",
    "inference_substrate_class": "A closed class separates live work from a stable preflight block.",
    "execution_venue": "The host venue rules out an unattributed remote execution.",
    "gpu_telemetry_rows": "Numeric GPU readings show device identity, use, and release.",
    "token_rows": "Per-call token counts expose truncation and budget drift.",
    "duration_s": "Measured wall time exposes interrupted or implausibly short acquisition.",
    "source_artifact_hashes": "Source hashes bind results to reviewed code, specs, and prior context.",
    "raw_trace_manifest": "Content-addressed raw files let interrupted runs resume without row loss.",
    "rows": "One joined row per cell prevents aggregate-only evidence.",
    "base_instance_rows": "Twelve frozen bases separate structural family from surface variants.",
    "variant_rows": "Relabel and paraphrase rows make surface transformations explicit.",
    "solver_receipt_rows": "Exact labels, witnesses, objectives, and effort remain independently replayable.",
    "model_output_rows": "Prompts and direct outputs are preserved before parsing.",
    "parse_rows": "Transport success stays separate from semantic correctness.",
    "exact_outcome_rows": "Every proposal remains subordinate to its instance-level exact checker.",
    "family_rows": "Disaggregated family metrics prevent pooled results from hiding failures.",
    "hardness_stratum_rows": "Solver effort is retained only as a descriptive stratum.",
    "relabel_sensitivity_rows": "Paired relabel outcomes measure symbol sensitivity.",
    "paraphrase_consistency_rows": "Paired paraphrase outcomes measure surface consistency.",
    "model_identity_confound_rows": "Identity receipts expose family, file, template, or quantization confounds.",
    "planned_cell_count": "The fixed denominator prevents scope reduction after seeing outputs.",
    "completed_cell_count": "A raw semantic-key count measures acquisition completeness.",
    "finite_answer_id_transport_used": "False prevents a small answer menu from replacing direct generation.",
    "schema_constraintir_reprompt_used": "False prevents a structured reprompt from repairing direct text.",
    "sota_constraint_bank_ready_score": "One means complete attributable evidence, independent of accuracy.",
    "random_seed": "A frozen controller seed fixes cell ordering and deterministic decoding.",
    "reproducibility_checksum": "A canonical digest detects later artifact mutation.",
    "gate_check_summary": "The first exact expected-observed failure makes blocks actionable.",
    "verifier_is_oracle": "False prevents exact labels from becoming a model-difficulty claim.",
    "verdict_class": "A closed class gives automation an unambiguous terminal state.",
    "honest_verdict": "A matching prefix states complete, retryable, or blocked status without inflation.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with one stable byte spelling."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash the exact UTF-8 bytes of a string."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the digest that contains the result."""

    return sha256_text(
        canonical_json(
            {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
        )
    )


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Record one precondition with exact expected and observed values."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return all checks and promote the first failure for automation."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def resolve_model_specs(
    *, cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair
) -> list[JsonDict]:
    """Resolve all three paths through two explicit cached-pair calls."""

    first = cached_pair_func(
        gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT, model_indices=(0, 2)
    ) or []
    second = cached_pair_func(
        gpu_indices=(0, 1), preferred_quant=PREFERRED_QUANT, model_indices=(1, 0)
    ) or []
    resolved = {
        str(row.get("hf_id")): str(row.get("model_path") or "") for row in [*first, *second]
    }
    return [
        {
            "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
            "hf_id": model_id,
            "model_path": resolved.get(model_id, ""),
            "gpu_indices": [0, 1],
            "quantization": PREFERRED_QUANT,
            "chat_template_source": "embedded_gguf",
            "resolution_method": "cached_sota_pair",
            "remote_allowed": False,
        }
        for model_id in REQUIRED_MODEL_IDS
    ]


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject model order, path, quantization, or chat-template substitution."""

    errors: list[str] = []
    if [row.get("hf_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_roster_mismatch")
    for row in rows:
        model_id = str(row.get("hf_id"))
        path = str(row.get("model_path") or "")
        if not path:
            errors.append(f"model_path_missing:{model_id}")
        elif Path(path).suffix.lower() != ".gguf" or "mmproj" in Path(path).name.lower():
            errors.append(f"model_path_not_language_gguf:{model_id}")
        if row.get("quantization") != PREFERRED_QUANT:
            errors.append(f"model_quantization_mismatch:{model_id}")
        if row.get("chat_template_source") != "embedded_gguf":
            errors.append(f"chat_template_source_mismatch:{model_id}")
        if row.get("resolution_method") != "cached_sota_pair":
            errors.append(f"model_resolution_method_mismatch:{model_id}")
        if row.get("gpu_indices") != [0, 1] or row.get("remote_allowed") is not False:
            errors.append(f"model_execution_policy_mismatch:{model_id}")
    return errors


MODEL_SPECS = resolve_model_specs()


def _sat_formals() -> list[JsonDict]:
    """Return four fixed five-variable, six-clause SAT cases."""

    variables = ["A", "B", "C", "D", "E"]
    clauses = [
        [["A", "B"], ["!A", "C"], ["!B", "D"], ["!C", "E"], ["!D", "E"], ["A", "!E"]],
        [["A"], ["!A"], ["B", "C"], ["!B", "D"], ["!C", "E"], ["D", "E"]],
        [["A", "B", "C"], ["!A", "D"], ["!B", "D"], ["!C", "E"], ["!D", "E"], ["A", "!E"]],
        [["A", "B"], ["!A", "!B"], ["C", "D"], ["!C", "E"], ["!D", "E"], ["!E", "A"]],
    ]
    return [{"variables": variables, "clauses": row} for row in clauses]


def _graph_formals() -> list[JsonDict]:
    """Return four fixed five-node, six-edge, three-color cases."""

    nodes = ["A", "B", "C", "D", "E"]
    edges = [
        [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"], ["E", "A"], ["A", "C"]],
        [["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"], ["C", "D"]],
        [["A", "D"], ["A", "E"], ["B", "D"], ["B", "E"], ["C", "D"], ["C", "E"]],
        [["A", "B"], ["B", "C"], ["C", "A"], ["A", "D"], ["B", "D"], ["C", "E"]],
    ]
    return [{"nodes": nodes, "edges": row, "color_count": 3} for row in edges]


def _schedule_formals() -> list[JsonDict]:
    """Return four fixed single-machine bounded scheduling cases."""

    jobs = ["A", "B", "C", "D"]
    durations = {"A": 2, "B": 1, "C": 2, "D": 1}
    precedence = [
        [["A", "C"], ["B", "D"], ["A", "D"]],
        [["A", "B"], ["B", "C"], ["C", "D"]],
        [["A", "D"], ["B", "D"], ["C", "D"]],
        [["A", "C"], ["B", "C"], ["B", "D"]],
    ]
    return [
        {
            "jobs": jobs,
            "durations": durations,
            "precedence": row,
            "machine_count": 1,
            "horizon": 8,
            "objective": "minimize_makespan",
        }
        for row in precedence
    ]


def _symbols(family: str, formal: Mapping[str, Any]) -> list[str]:
    """Return the symbols whose names can change without changing meaning."""

    return list(formal[{"sat_logic": "variables", "graph_coloring": "nodes", "bounded_scheduling": "jobs"}[family]])


def _renamed_formal(family: str, formal: Mapping[str, Any], mapping: Mapping[str, str]) -> JsonDict:
    """Apply a bijective symbol rename to every formal occurrence."""

    value = deepcopy(dict(formal))
    if family == "sat_logic":
        value["variables"] = [mapping[name] for name in value["variables"]]
        value["clauses"] = [
            [("!" if literal.startswith("!") else "") + mapping[literal.lstrip("!")] for literal in clause]
            for clause in value["clauses"]
        ]
    elif family == "graph_coloring":
        value["nodes"] = [mapping[name] for name in value["nodes"]]
        value["edges"] = [[mapping[left], mapping[right]] for left, right in value["edges"]]
    else:
        value["jobs"] = [mapping[name] for name in value["jobs"]]
        value["durations"] = {mapping[name]: duration for name, duration in value["durations"].items()}
        value["precedence"] = [[mapping[left], mapping[right]] for left, right in value["precedence"]]
    return value


def _prompt_body(family: str, formal: Mapping[str, Any], paraphrase: bool) -> str:
    """Render one direct-answer prompt without exposing labels or answer IDs."""

    if family == "sat_logic":
        clauses = " ; ".join("(" + " OR ".join(clause) + ")" for clause in formal["clauses"])
        task = (
            f"Boolean variables: {', '.join(formal['variables'])}. Clauses: {clauses}. "
            if not paraphrase
            else f"Decide the same conjunction of disjunctions. Symbols are {', '.join(formal['variables'])}; required clauses are {clauses}. "
        )
        schema = '{"status":"SAT","assignment":{"A":true,...}} or {"status":"UNSAT"}'
    elif family == "graph_coloring":
        edges = ", ".join(f"{left}-{right}" for left, right in formal["edges"])
        task = (
            f"Color nodes {', '.join(formal['nodes'])} with colors 1 through {formal['color_count']}. Adjacent nodes differ. Edges: {edges}. "
            if not paraphrase
            else f"Assign one of {formal['color_count']} integer colors to each vertex {', '.join(formal['nodes'])}; endpoints of every listed link must disagree: {edges}. "
        )
        schema = '{"status":"SAT","coloring":{"A":1,...}} or {"status":"UNSAT"}'
    else:
        durations = ", ".join(f"{name}={formal['durations'][name]}" for name in formal["jobs"])
        precedence = ", ".join(f"{left} before {right}" for left, right in formal["precedence"])
        task = (
            f"Schedule jobs {', '.join(formal['jobs'])} on one machine. Integer durations: {durations}. Precedence: {precedence}. Starts are nonnegative and all jobs finish by {formal['horizon']}. Minimize makespan. "
            if not paraphrase
            else f"Find the shortest integer-time, non-overlapping single-machine timetable for {', '.join(formal['jobs'])}. Processing times are {durations}; ordering rules are {precedence}; the deadline is {formal['horizon']}. "
        )
        schema = '{"status":"SAT","starts":{"A":0,...}} or {"status":"UNSAT"}'
    return (
        "Solve this exact finite constraint instance. Return only one JSON object and no prose. "
        + task
        + "Use this direct-answer shape with the actual displayed symbol names: "
        + schema
        + ". Do not return an answer ID."
    )


def _padded_prompt(family: str, formal: Mapping[str, Any], *, paraphrase: bool) -> str:
    """Pad prompts to one frozen character budget within every family."""

    body = _prompt_body(family, formal, paraphrase)
    if len(body) > SURFACE_BUDGET_CHARS:
        raise ValueError("surface_budget_exceeded")
    return body + " " * (SURFACE_BUDGET_CHARS - len(body))


def _literal_value(literal: str, assignment: Mapping[str, bool]) -> bool:
    """Evaluate one signed Boolean literal."""

    value = bool(assignment[literal.lstrip("!")])
    return not value if literal.startswith("!") else value


def _normalize_solution(solution: Mapping[str, int | bool], inverse: Mapping[str, str]) -> JsonDict:
    """Map variant symbols back to canonical names before set comparison."""

    return {inverse.get(name, name): value for name, value in solution.items()}


def _solve_sat(formal: Mapping[str, Any]) -> tuple[list[JsonDict], int]:
    """Enumerate every SAT assignment and count literal checks."""

    variables = list(formal["variables"])
    solutions: list[JsonDict] = []
    checks = 0
    for values in itertools.product((False, True), repeat=len(variables)):
        assignment = dict(zip(variables, values, strict=True))
        holds = True
        for clause in formal["clauses"]:
            clause_holds = False
            for literal in clause:
                checks += 1
                if _literal_value(str(literal), assignment):
                    clause_holds = True
                    break
            if not clause_holds:
                holds = False
                break
        if holds:
            solutions.append(assignment)
    return solutions, checks


def _solve_graph(formal: Mapping[str, Any]) -> tuple[list[JsonDict], int]:
    """Enumerate every coloring and count edge checks."""

    nodes = list(formal["nodes"])
    solutions: list[JsonDict] = []
    checks = 0
    for values in itertools.product(range(1, int(formal["color_count"]) + 1), repeat=len(nodes)):
        coloring = dict(zip(nodes, values, strict=True))
        holds = True
        for left, right in formal["edges"]:
            checks += 1
            if coloring[left] == coloring[right]:
                holds = False
                break
        if holds:
            solutions.append(coloring)
    return solutions, checks


def _schedule_violations(formal: Mapping[str, Any], starts: Mapping[str, int]) -> int:
    """Count domain, precedence, overlap, and objective-independent violations."""

    jobs = list(formal["jobs"])
    if set(starts) != set(jobs) or any(type(starts.get(job)) is not int for job in jobs):
        return max(1, len(set(jobs) ^ set(starts)))
    violations = sum(
        int(starts[job] < 0 or starts[job] + formal["durations"][job] > formal["horizon"])
        for job in jobs
    )
    violations += sum(
        int(starts[left] + formal["durations"][left] > starts[right])
        for left, right in formal["precedence"]
    )
    for index, left in enumerate(jobs):
        for right in jobs[index + 1 :]:
            left_end = starts[left] + formal["durations"][left]
            right_end = starts[right] + formal["durations"][right]
            violations += int(not (left_end <= starts[right] or right_end <= starts[left]))
    return violations


def _solve_schedule(formal: Mapping[str, Any]) -> tuple[list[JsonDict], int, int | None]:
    """Enumerate bounded integer starts and retain every optimal schedule."""

    jobs = list(formal["jobs"])
    domains = [range(int(formal["horizon"]) - int(formal["durations"][job]) + 1) for job in jobs]
    optimal: list[JsonDict] = []
    optimum: int | None = None
    checks = 0
    for values in itertools.product(*domains):
        starts = dict(zip(jobs, values, strict=True))
        checks += 1
        if _schedule_violations(formal, starts):
            continue
        makespan = max(starts[job] + formal["durations"][job] for job in jobs)
        if optimum is None or makespan < optimum:
            optimum = makespan
            optimal = [starts]
        elif makespan == optimum:
            optimal.append(starts)
    return optimal, checks, optimum


def solve_formal(
    family: str,
    formal: Mapping[str, Any],
    *,
    inverse_symbol_map: Mapping[str, str] | None = None,
) -> JsonDict:
    """Compute the complete exact solution set for one finite instance."""

    if family == "sat_logic":
        raw_solutions, effort = _solve_sat(formal)
        objective = None
    elif family == "graph_coloring":
        raw_solutions, effort = _solve_graph(formal)
        objective = None
    elif family == "bounded_scheduling":
        raw_solutions, effort, objective = _solve_schedule(formal)
    else:
        raise ValueError(f"unknown_constraint_family:{family}")
    inverse = dict(inverse_symbol_map or {})
    normalized = [_normalize_solution(row, inverse) for row in raw_solutions]
    normalized.sort(key=canonical_json)
    return {
        "feasible": bool(raw_solutions),
        "solution_count": len(raw_solutions),
        "solution_set_hash": sha256_text(canonical_json(normalized)),
        "witness": deepcopy(raw_solutions[0]) if raw_solutions else None,
        "objective": objective,
        "solver_effort": effort,
        "solver_effort_unit": "exact_candidate_checks",
    }


def _base_definitions() -> list[tuple[str, str, JsonDict, int, str]]:
    """Return ordered family, ID, formal payload, size, and density tuples."""

    rows: list[tuple[str, str, JsonDict, int, str]] = []
    for index, formal in enumerate(_sat_formals()):
        rows.append(("sat_logic", f"sat-{index}", formal, 5, "6/5"))
    for index, formal in enumerate(_graph_formals()):
        rows.append(("graph_coloring", f"graph-{index}", formal, 5, "6/10"))
    for index, formal in enumerate(_schedule_formals()):
        rows.append(("bounded_scheduling", f"schedule-{index}", formal, 4, "3/12"))
    return rows


def _receipt(
    *,
    instance_id: str,
    base_id: str,
    family: str,
    variant_kind: str,
    formal: Mapping[str, Any],
    inverse_symbol_map: Mapping[str, str] | None,
    solver_effort_stratum: str = "pending",
) -> JsonDict:
    """Build one exact receipt from formal bytes, never from model output."""

    solved = solve_formal(family, formal, inverse_symbol_map=inverse_symbol_map)
    return {
        "instance_id": instance_id,
        "base_id": base_id,
        "family": family,
        "variant_kind": variant_kind,
        "formal": deepcopy(dict(formal)),
        "formal_hash": sha256_text(canonical_json(formal)),
        "inverse_symbol_map": deepcopy(dict(inverse_symbol_map or {})),
        **solved,
        "solver_effort_stratum": solver_effort_stratum,
        "solver_effort_is_model_difficulty": False,
        "checker": "complete_finite_enumerator_v1",
    }


def build_frozen_fixture() -> JsonDict:
    """Freeze 12 bases plus one relabel and paraphrase for each base."""

    base_rows: list[JsonDict] = []
    variant_rows: list[JsonDict] = []
    receipts: list[JsonDict] = []
    canonical_receipts: list[JsonDict] = []
    for family, base_id, formal, size, density in _base_definitions():
        prompt = _padded_prompt(family, formal, paraphrase=False)
        receipt = _receipt(
            instance_id=f"{base_id}:canonical",
            base_id=base_id,
            family=family,
            variant_kind="canonical",
            formal=formal,
            inverse_symbol_map=None,
        )
        canonical_receipts.append(receipt)
        base_rows.append(
            {
                "instance_id": f"{base_id}:canonical",
                "base_id": base_id,
                "family": family,
                "variant_kind": "canonical",
                "formal": deepcopy(formal),
                "formal_hash": sha256_text(canonical_json(formal)),
                "prompt": prompt,
                "prompt_hash": sha256_text(prompt),
                "size": size,
                "density": density,
                "surface_budget_chars": SURFACE_BUDGET_CHARS,
                "generation_budget_tokens": GENERATION_CONFIG["max_tokens"],
                "feasible": receipt["feasible"],
                "solution_count": receipt["solution_count"],
                "solution_set_hash": receipt["solution_set_hash"],
                "objective": receipt["objective"],
            }
        )
    ranked = sorted(
        canonical_receipts, key=lambda row: (int(row["solver_effort"]), str(row["base_id"]))
    )
    stratum_by_base = {
        str(row["base_id"]): ("low" if index < 4 else "medium" if index < 8 else "high")
        for index, row in enumerate(ranked)
    }
    for base, canonical in zip(base_rows, canonical_receipts, strict=True):
        stratum = stratum_by_base[str(base["base_id"])]
        base["solver_effort"] = canonical["solver_effort"]
        base["solver_effort_stratum"] = stratum
        canonical["solver_effort_stratum"] = stratum
        receipts.append(canonical)
        symbols = _symbols(str(base["family"]), base["formal"])
        forward = {name: f"X{index + 1}" for index, name in enumerate(symbols)}
        inverse = {value: key for key, value in forward.items()}
        relabeled_formal = _renamed_formal(str(base["family"]), base["formal"], forward)
        relabel_prompt = _padded_prompt(str(base["family"]), relabeled_formal, paraphrase=False)
        relabel_id = f"{base['base_id']}:relabel"
        relabel_receipt = _receipt(
            instance_id=relabel_id,
            base_id=str(base["base_id"]),
            family=str(base["family"]),
            variant_kind="relabel",
            formal=relabeled_formal,
            inverse_symbol_map=inverse,
            solver_effort_stratum=stratum,
        )
        variant_rows.append(
            {
                "instance_id": relabel_id,
                "base_id": base["base_id"],
                "family": base["family"],
                "variant_kind": "relabel",
                "formal": relabeled_formal,
                "formal_hash": relabel_receipt["formal_hash"],
                "prompt": relabel_prompt,
                "prompt_hash": sha256_text(relabel_prompt),
                "forward_symbol_map": forward,
                "inverse_symbol_map": inverse,
                "proof_preserving": relabel_receipt["solution_set_hash"]
                == canonical["solution_set_hash"],
                "base_solution_set_hash": canonical["solution_set_hash"],
                "surface_budget_chars": SURFACE_BUDGET_CHARS,
                "generation_budget_tokens": GENERATION_CONFIG["max_tokens"],
                "solver_effort_stratum": stratum,
            }
        )
        receipts.append(relabel_receipt)
        paraphrase_formal = deepcopy(base["formal"])
        paraphrase_prompt = _padded_prompt(str(base["family"]), paraphrase_formal, paraphrase=True)
        paraphrase_id = f"{base['base_id']}:paraphrase"
        paraphrase_receipt = _receipt(
            instance_id=paraphrase_id,
            base_id=str(base["base_id"]),
            family=str(base["family"]),
            variant_kind="paraphrase",
            formal=paraphrase_formal,
            inverse_symbol_map=None,
            solver_effort_stratum=stratum,
        )
        variant_rows.append(
            {
                "instance_id": paraphrase_id,
                "base_id": base["base_id"],
                "family": base["family"],
                "variant_kind": "paraphrase",
                "formal": paraphrase_formal,
                "formal_hash": paraphrase_receipt["formal_hash"],
                "prompt": paraphrase_prompt,
                "prompt_hash": sha256_text(paraphrase_prompt),
                "forward_symbol_map": {},
                "inverse_symbol_map": {},
                "proof_preserving": paraphrase_receipt["solution_set_hash"]
                == canonical["solution_set_hash"],
                "base_solution_set_hash": canonical["solution_set_hash"],
                "surface_budget_chars": SURFACE_BUDGET_CHARS,
                "generation_budget_tokens": GENERATION_CONFIG["max_tokens"],
                "solver_effort_stratum": stratum,
                "independent_exact_recheck": True,
            }
        )
        receipts.append(paraphrase_receipt)
    fixture = {
        "schema": "carnot.experiment_7129.constraint_fixture.v1",
        "random_seed": RANDOM_SEED,
        "base_instance_rows": base_rows,
        "variant_rows": variant_rows,
        "solver_receipt_rows": receipts,
    }
    fixture["fixture_hash"] = sha256_text(canonical_json(fixture))
    return fixture


def _receipt_projection(row: Mapping[str, Any]) -> JsonDict:
    """Select exact fields that must match an independent replay."""

    fields = (
        "instance_id",
        "base_id",
        "family",
        "variant_kind",
        "formal",
        "formal_hash",
        "inverse_symbol_map",
        "feasible",
        "solution_count",
        "solution_set_hash",
        "witness",
        "objective",
        "solver_effort",
        "solver_effort_unit",
        "solver_effort_stratum",
        "solver_effort_is_model_difficulty",
        "checker",
    )
    return {field: deepcopy(row.get(field)) for field in fields}


def fixture_errors(fixture: Mapping[str, Any]) -> list[str]:
    """Replay exact labels and reject any non-preserving surface variant."""

    errors: list[str] = []
    bases = list(fixture.get("base_instance_rows", []))
    variants = list(fixture.get("variant_rows", []))
    receipts = list(fixture.get("solver_receipt_rows", []))
    if len(bases) != 12 or Counter(str(row.get("family")) for row in bases) != Counter(
        {family: 4 for family in CONSTRAINT_FAMILIES}
    ):
        errors.append("base_instance_count_or_family_mismatch")
    if len(variants) != 24:
        errors.append("variant_count_mismatch")
    if len(receipts) != 36:
        errors.append("solver_receipt_count_mismatch")
    base_by_id = {str(row.get("base_id")): row for row in bases}
    receipt_by_id = {str(row.get("instance_id")): row for row in receipts}
    for receipt in receipts:
        replay = _receipt(
            instance_id=str(receipt.get("instance_id")),
            base_id=str(receipt.get("base_id")),
            family=str(receipt.get("family")),
            variant_kind=str(receipt.get("variant_kind")),
            formal=dict(receipt.get("formal") or {}),
            inverse_symbol_map=dict(receipt.get("inverse_symbol_map") or {}),
            solver_effort_stratum=str(receipt.get("solver_effort_stratum")),
        )
        if _receipt_projection(receipt) != _receipt_projection(replay):
            errors.append(f"solver_receipt_mismatch:{receipt.get('instance_id')}")
    for base in bases:
        receipt = receipt_by_id.get(str(base.get("instance_id")), {})
        for field in ("feasible", "solution_count", "solution_set_hash", "objective"):
            if base.get(field) != receipt.get(field):
                errors.append(f"base_label_mismatch:{base.get('base_id')}:{field}")
        if len(str(base.get("prompt", ""))) != SURFACE_BUDGET_CHARS:
            errors.append(f"base_surface_budget_mismatch:{base.get('base_id')}")
        if base.get("prompt_hash") != sha256_text(str(base.get("prompt", ""))):
            errors.append(f"base_prompt_hash_mismatch:{base.get('base_id')}")
    for variant in variants:
        base = base_by_id.get(str(variant.get("base_id")), {})
        receipt = receipt_by_id.get(str(variant.get("instance_id")), {})
        if variant.get("variant_kind") == "paraphrase":
            if variant.get("formal") != base.get("formal"):
                errors.append(f"semantic_paraphrase_drift:{variant.get('instance_id')}")
            if variant.get("independent_exact_recheck") is not True:
                errors.append(f"paraphrase_recheck_missing:{variant.get('instance_id')}")
        elif variant.get("variant_kind") == "relabel":
            forward = dict(variant.get("forward_symbol_map") or {})
            inverse = dict(variant.get("inverse_symbol_map") or {})
            expected_inverse = {value: key for key, value in forward.items()}
            if inverse != expected_inverse or set(forward) != set(
                _symbols(str(base.get("family")), dict(base.get("formal") or {}))
            ):
                errors.append(f"non_preserving_relabel:{variant.get('instance_id')}")
        else:
            errors.append(f"unknown_variant_kind:{variant.get('instance_id')}")
        if (
            receipt.get("solution_set_hash") != base.get("solution_set_hash")
            or receipt.get("feasible") != base.get("feasible")
            or receipt.get("objective") != base.get("objective")
            or variant.get("proof_preserving") is not True
        ):
            errors.append(f"variant_exact_semantics_mismatch:{variant.get('instance_id')}")
        if len(str(variant.get("prompt", ""))) != SURFACE_BUDGET_CHARS:
            errors.append(f"variant_surface_budget_mismatch:{variant.get('instance_id')}")
        if variant.get("prompt_hash") != sha256_text(str(variant.get("prompt", ""))):
            errors.append(f"variant_prompt_hash_mismatch:{variant.get('instance_id')}")
    payload = {key: deepcopy(value) for key, value in fixture.items() if key != "fixture_hash"}
    if fixture.get("fixture_hash") != sha256_text(canonical_json(payload)):
        errors.append("fixture_hash_mismatch")
    return list(dict.fromkeys(errors))


def _instance_views(fixture: Mapping[str, Any]) -> list[JsonDict]:
    """Return canonical and surface rows in one stable per-base order."""

    bases = {str(row["base_id"]): deepcopy(dict(row)) for row in fixture["base_instance_rows"]}
    variants = defaultdict(dict)
    for row in fixture["variant_rows"]:
        variants[str(row["base_id"])][str(row["variant_kind"])] = deepcopy(dict(row))
    rows = []
    for base_id, base in bases.items():
        rows.extend([base, variants[base_id]["relabel"], variants[base_id]["paraphrase"]])
    return rows


def build_schedule(
    model_specs: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any]
) -> list[JsonDict]:
    """Build the fixed 108-cell direct-generation schedule."""

    rows = []
    for model_index, model in enumerate(model_specs):
        for instance_index, instance in enumerate(_instance_views(fixture)):
            seed = RANDOM_SEED + model_index * 1_000 + instance_index
            row = {
                "cell_key": f"{model['hf_id']}|{instance['instance_id']}",
                "model_id": model["hf_id"],
                "model_path": model["model_path"],
                "model_sha256": model.get("model_sha256"),
                "instance_id": instance["instance_id"],
                "base_id": instance["base_id"],
                "family": instance["family"],
                "variant_kind": instance["variant_kind"],
                "prompt": instance["prompt"],
                "prompt_hash": instance["prompt_hash"],
                "seed": seed,
                "generation_config": deepcopy(GENERATION_CONFIG),
                "proposal_source": "direct_model_text",
                "finite_answer_id_transport_used": False,
                "schema_constraintir_reprompt_used": False,
            }
            row["schedule_hash"] = sha256_text(canonical_json(row))
            rows.append(row)
    return rows


def _contains_answer_id(value: Any) -> bool:
    """Detect finite answer identifiers at any parsed JSON depth."""

    if isinstance(value, Mapping):
        return any(
            str(key).lower() in {"answer_id", "option_id", "choice_id"}
            or _contains_answer_id(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_answer_id(item) for item in value)
    return False


def parse_model_text(raw_text: str, family: str) -> JsonDict:
    """Parse one direct JSON object without repair or a schema reprompt."""

    decoder = json.JSONDecoder()
    parsed: Any = None
    for match in re.finditer(r"\{", raw_text):
        try:
            candidate, _end = decoder.raw_decode(raw_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    if parsed is None:
        return {"parse_success": False, "parsed": None, "parse_error": "json_object_missing"}
    if _contains_answer_id(parsed):
        return {
            "parse_success": False,
            "parsed": None,
            "parse_error": "finite_answer_id_transport_forbidden",
        }
    if parsed.get("status") not in {"SAT", "UNSAT"}:
        return {"parse_success": False, "parsed": None, "parse_error": "status_invalid"}
    if parsed["status"] == "SAT":
        required = {
            "sat_logic": "assignment",
            "graph_coloring": "coloring",
            "bounded_scheduling": "starts",
        }.get(family)
        if required is None or not isinstance(parsed.get(required), dict):
            return {
                "parse_success": False,
                "parsed": None,
                "parse_error": "direct_assignment_shape_invalid",
            }
    return {"parse_success": True, "parsed": parsed, "parse_error": None}


def _sat_violations(formal: Mapping[str, Any], assignment: Mapping[str, Any]) -> int:
    """Count SAT domain and clause violations for one proposed assignment."""

    variables = set(formal["variables"])
    if set(assignment) != variables or any(type(assignment.get(name)) is not bool for name in variables):
        return max(1, len(variables ^ set(assignment)))
    return sum(
        int(not any(_literal_value(str(literal), assignment) for literal in clause))
        for clause in formal["clauses"]
    )


def _graph_violations(formal: Mapping[str, Any], coloring: Mapping[str, Any]) -> int:
    """Count graph domain and edge violations for one proposed coloring."""

    nodes = set(formal["nodes"])
    if set(coloring) != nodes or any(type(coloring.get(node)) is not int for node in nodes):
        return max(1, len(nodes ^ set(coloring)))
    violations = sum(
        int(coloring[node] < 1 or coloring[node] > int(formal["color_count"])) for node in nodes
    )
    return violations + sum(int(coloring[left] == coloring[right]) for left, right in formal["edges"])


def verify_direct_answer(receipt: Mapping[str, Any], parsed: Mapping[str, Any]) -> JsonDict:
    """Check one parsed action against only its exact instance."""

    status = parsed.get("status")
    if status == "UNSAT":
        correct = receipt.get("feasible") is False
        return {
            "exact_correct": correct,
            "constraint_violation_count": 0 if correct else 1,
            "objective_observed": None,
            "objective_matches": receipt.get("objective") is None,
        }
    if status != "SAT" or receipt.get("feasible") is not True:
        return {
            "exact_correct": False,
            "constraint_violation_count": 1,
            "objective_observed": None,
            "objective_matches": False,
        }
    family = str(receipt["family"])
    formal = dict(receipt["formal"])
    if family == "sat_logic":
        answer = dict(parsed.get("assignment") or {})
        violations = _sat_violations(formal, answer)
        observed = None
    elif family == "graph_coloring":
        answer = dict(parsed.get("coloring") or {})
        violations = _graph_violations(formal, answer)
        observed = None
    else:
        answer = dict(parsed.get("starts") or {})
        violations = _schedule_violations(formal, answer)
        observed = (
            max(answer[job] + formal["durations"][job] for job in formal["jobs"])
            if violations == 0
            else None
        )
    objective_matches = observed == receipt.get("objective")
    return {
        "exact_correct": violations == 0 and objective_matches,
        "constraint_violation_count": violations + int(not objective_matches),
        "objective_observed": observed,
        "objective_matches": objective_matches,
    }


def raw_output_row(
    scheduled: Mapping[str, Any],
    *,
    raw_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_s: float,
    terminal_state: str,
    raw_response: Mapping[str, Any] | None = None,
    reasoning_text: str = "",
    error: str | None = None,
) -> JsonDict:
    """Create an unparsed row that can be written immediately after a call."""

    return {
        **deepcopy(dict(scheduled)),
        "raw_text": raw_text,
        "raw_output_hash": sha256_text(raw_text),
        "reasoning_text": reasoning_text,
        "reasoning_hash": sha256_text(reasoning_text),
        "raw_response": deepcopy(dict(raw_response or {})),
        "raw_response_hash": sha256_text(canonical_json(raw_response or {})),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "duration_s": float(duration_s),
        "terminal_state": terminal_state,
        "error": error,
        "raw_persisted_before_parse": True,
        "parsed_at_write_time": False,
        "labeled_at_write_time": False,
    }


def load_raw_rows(path: Path) -> list[JsonDict]:
    """Load durable rows and reject duplicates or changed direct output hashes."""

    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keys = [str(row.get("cell_key")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_cell_key")
    if any(row.get("raw_output_hash") != sha256_text(str(row.get("raw_text", ""))) for row in rows):
        raise ValueError("raw_output_hash_mismatch")
    return rows


def persist_raw_row(path: Path, row: Mapping[str, Any]) -> JsonDict:
    """Append one invocation row and make duplicate retries idempotent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_raw_rows(path)
    prior = next((item for item in existing if item.get("cell_key") == row.get("cell_key")), None)
    if prior is not None:
        if prior != dict(row):
            raise ValueError("raw_row_mismatch")
        return {"cell_key": row.get("cell_key"), "written": False, "path": str(path)}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"cell_key": row.get("cell_key"), "written": True, "path": str(path)}


MODEL_OUTPUT_FIELDS = (
    "cell_key",
    "model_id",
    "model_path",
    "model_sha256",
    "instance_id",
    "base_id",
    "family",
    "variant_kind",
    "prompt",
    "prompt_hash",
    "seed",
    "generation_config",
    "proposal_source",
    "finite_answer_id_transport_used",
    "schema_constraintir_reprompt_used",
    "schedule_hash",
    "raw_text",
    "raw_output_hash",
    "reasoning_text",
    "reasoning_hash",
    "raw_response_hash",
    "prompt_tokens",
    "completion_tokens",
    "duration_s",
    "terminal_state",
    "error",
    "raw_persisted_before_parse",
    "parsed_at_write_time",
    "labeled_at_write_time",
)


def _model_output_projection(row: Mapping[str, Any]) -> JsonDict:
    """Project raw evidence without parser or exact labels."""

    return {field: deepcopy(row.get(field)) for field in MODEL_OUTPUT_FIELDS}


def _reduce_rows(
    raw_rows: Sequence[Mapping[str, Any]], fixture: Mapping[str, Any]
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Parse and score durable rows while preserving separate projections."""

    receipt_by_id = {str(row["instance_id"]): row for row in fixture["solver_receipt_rows"]}
    outputs: list[JsonDict] = []
    parses: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    combined: list[JsonDict] = []
    for raw in sorted(raw_rows, key=lambda row: str(row.get("cell_key"))):
        output = _model_output_projection(raw)
        parsed = parse_model_text(str(raw.get("raw_text", "")), str(raw.get("family", "")))
        parse_row = {"cell_key": raw.get("cell_key"), **parsed}
        if parsed["parse_success"]:
            exact = verify_direct_answer(receipt_by_id[str(raw["instance_id"])], parsed["parsed"])
        else:
            exact = {
                "exact_correct": False,
                "constraint_violation_count": 1,
                "objective_observed": None,
                "objective_matches": False,
            }
        outcome = {
            "cell_key": raw.get("cell_key"),
            "instance_id": raw.get("instance_id"),
            "solver_effort_stratum": receipt_by_id[str(raw["instance_id"])][
                "solver_effort_stratum"
            ],
            "solver_receipt_hash": sha256_text(
                canonical_json(receipt_by_id[str(raw["instance_id"])])
            ),
            **exact,
        }
        outputs.append(output)
        parses.append(parse_row)
        outcomes.append(outcome)
        combined.append({**output, **parse_row, **outcome})
    return outputs, parses, outcomes, combined


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a rate only when the denominator is nonzero."""

    return numerator / denominator if denominator else None


def _metric_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Summarize model-family and model-effort cells without gating readiness."""

    family_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    effort_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        family_groups[(str(row["model_id"]), str(row["family"]))].append(row)
        effort_groups[(str(row["model_id"]), str(row["solver_effort_stratum"]))].append(row)

    def summary(key: tuple[str, str], group: list[Mapping[str, Any]], label: str) -> JsonDict:
        count = len(group)
        parsed = sum(row.get("parse_success") is True for row in group)
        correct = sum(row.get("exact_correct") is True for row in group)
        return {
            "model_id": key[0],
            label: key[1],
            "cell_count": count,
            "parse_rate": _rate(parsed, count),
            "accuracy": _rate(correct, count),
            "exact_constraint_violation_count": sum(
                int(row.get("constraint_violation_count", 0) or 0) for row in group
            ),
            "solver_effort_is_model_difficulty": False,
        }

    family_rows = [summary(key, group, "family") for key, group in sorted(family_groups.items())]
    effort_rows = [
        summary(key, group, "solver_effort_stratum")
        for key, group in sorted(effort_groups.items())
    ]
    return family_rows, effort_rows


def _pair_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Build canonical-versus-surface paired outcome rows."""

    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["base_id"]))][str(row["variant_kind"])] = row
    relabel_rows = []
    paraphrase_rows = []
    for (model_id, base_id), variants in sorted(grouped.items()):
        canonical = variants.get("canonical")
        relabel = variants.get("relabel")
        paraphrase = variants.get("paraphrase")
        relabel_rows.append(
            {
                "model_id": model_id,
                "base_id": base_id,
                "pair_complete": canonical is not None and relabel is not None,
                "canonical_exact_correct": canonical.get("exact_correct") if canonical else None,
                "relabel_exact_correct": relabel.get("exact_correct") if relabel else None,
                "relabel_sensitive": (
                    canonical.get("exact_correct") != relabel.get("exact_correct")
                    if canonical and relabel
                    else None
                ),
            }
        )
        paraphrase_rows.append(
            {
                "model_id": model_id,
                "base_id": base_id,
                "pair_complete": canonical is not None and paraphrase is not None,
                "canonical_exact_correct": canonical.get("exact_correct") if canonical else None,
                "paraphrase_exact_correct": paraphrase.get("exact_correct") if paraphrase else None,
                "paraphrase_consistent": (
                    canonical.get("exact_correct") == paraphrase.get("exact_correct")
                    if canonical and paraphrase
                    else None
                ),
            }
        )
    return relabel_rows, paraphrase_rows


def _identity_rows_valid(
    specs: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> bool:
    """Require exact path, hash, quantization, template, and family identity."""

    by_id = {str(row.get("model_id")): row for row in rows}
    return len(by_id) == len(REQUIRED_MODEL_IDS) and all(
        (identity := by_id.get(str(spec.get("hf_id")))) is not None
        and identity.get("identity_matches") is True
        and identity.get("model_path") == spec.get("model_path")
        and identity.get("model_sha256") == spec.get("model_sha256")
        and identity.get("quantization") == PREFERRED_QUANT
        and identity.get("chat_template_source") == "embedded_gguf"
        and str(identity.get("chat_template_hash", "")).startswith("sha256:")
        for spec in specs
    )


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    fixture: Mapping[str, Any] | None = None,
    raw_rows: Sequence[Mapping[str, Any]] = (),
    model_identity_rows: Sequence[Mapping[str, Any]] = (),
    raw_trace_manifest: Sequence[Mapping[str, Any]] = (),
    gpu_telemetry_rows: Sequence[Mapping[str, Any]] = (),
    source_artifact_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build blocked, retryable partial, or positive-ready evidence."""

    fixture_value = deepcopy(dict(fixture or {}))
    bases = list(fixture_value.get("base_instance_rows", []))
    variants = list(fixture_value.get("variant_rows", []))
    receipts = list(fixture_value.get("solver_receipt_rows", []))
    outputs: list[JsonDict] = []
    parses: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    combined: list[JsonDict] = []
    if receipts:
        outputs, parses, outcomes, combined = _reduce_rows(raw_rows, fixture_value)
    family_rows, effort_rows = _metric_rows(combined)
    relabel_rows, paraphrase_rows = _pair_rows(combined)
    specs = [deepcopy(dict(row)) for row in model_specs]
    identity = [deepcopy(dict(row)) for row in model_identity_rows]
    identity_by_id = {str(row.get("model_id")): row for row in identity}
    for spec in specs:
        if not spec.get("model_sha256"):
            spec["model_sha256"] = identity_by_id.get(str(spec.get("hf_id")), {}).get(
                "model_sha256"
            )
    expected_keys = (
        {str(row["cell_key"]) for row in build_schedule(specs, fixture_value)} if receipts else set()
    )
    observed_keys = {str(row.get("cell_key")) for row in combined}
    completed = len(observed_keys & expected_keys)
    preflight_passed = preconditions.get("all_passed") is True
    fixture_valid = bool(receipts) and not fixture_errors(fixture_value)
    identity_valid = _identity_rows_valid(specs, identity)
    manifest_complete = bool(raw_trace_manifest) and {
        str(row.get("model_id")) for row in raw_trace_manifest
    } == set(REQUIRED_MODEL_IDS) and sum(int(row.get("row_count", 0) or 0) for row in raw_trace_manifest) == len(
        expected_keys
    )
    ready = int(
        preflight_passed
        and fixture_valid
        and identity_valid
        and observed_keys == expected_keys
        and len(combined) == len(expected_keys) == 108
        and manifest_complete
    )
    if not preflight_passed:
        verdict_class = "blocked"
        honest_verdict = "blocked_" + str(
            gate_summary(list(preconditions.get("checks", []))).get("failed_check")
            or "preflight_not_evaluated"
        )
        substrate_class = "blocked_no_run"
    elif ready:
        verdict_class = "positive"
        honest_verdict = "positive_complete_three_family_exact_constraint_bank"
        substrate_class = "model_bounded_generation"
    else:
        verdict_class = "partial"
        honest_verdict = f"partial_retryable_constraint_bank_{completed}_of_108"
        substrate_class = "model_bounded_generation"
    completion_check = gate_row("completed_cell_count", 108, completed, completed == 108)
    checks = list(preconditions.get("checks", []))
    if preflight_passed:
        checks.append(completion_check)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "run_date": str(run_date),
        "MODEL_SPECS": specs,
        "models_used": [model_id for model_id in REQUIRED_MODEL_IDS if any(row.get("model_id") == model_id for row in combined)],
        "model_repository_rows": [
            {"model_id": row.get("hf_id"), "repository": row.get("hf_id")} for row in specs
        ],
        "model_path_rows": [
            {"model_id": row.get("hf_id"), "model_path": row.get("model_path")} for row in specs
        ],
        "model_hash_rows": [
            {"model_id": row.get("hf_id"), "sha256": row.get("model_sha256")} for row in specs
        ],
        "model_quantization_rows": [
            {"model_id": row.get("hf_id"), "quantization": row.get("quantization")}
            for row in specs
        ],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": substrate_class,
        "execution_venue": "host",
        "gpu_telemetry_rows": [deepcopy(dict(row)) for row in gpu_telemetry_rows],
        "token_rows": [
            {
                "cell_key": row["cell_key"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "max_tokens": row["generation_config"]["max_tokens"],
            }
            for row in outputs
        ],
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes or {})),
        "raw_trace_manifest": [deepcopy(dict(row)) for row in raw_trace_manifest],
        "rows": combined,
        "base_instance_rows": bases,
        "variant_rows": variants,
        "solver_receipt_rows": receipts,
        "model_output_rows": outputs,
        "parse_rows": parses,
        "exact_outcome_rows": outcomes,
        "family_rows": family_rows,
        "hardness_stratum_rows": effort_rows,
        "relabel_sensitivity_rows": relabel_rows,
        "paraphrase_consistency_rows": paraphrase_rows,
        "model_identity_confound_rows": identity,
        "planned_cell_count": 108,
        "completed_cell_count": completed,
        "finite_answer_id_transport_used": False,
        "schema_constraintir_reprompt_used": False,
        "sota_constraint_bank_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _expected_prefix(verdict_class: str) -> str:
    """Return the required honest-verdict terminal prefix."""

    return {"positive": "positive_", "partial": "partial_", "blocked": "blocked_"}.get(
        verdict_class, verdict_class + "_"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check schema, rows, exact labels, identity, readiness, and verdict."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
        return errors
    if any(not str(artifact["field_principles"].get(field, "")).strip() for field in REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("execution_venue") != "host" or artifact.get("verifier_is_oracle") is not False:
        errors.append("venue_or_oracle_mismatch")
    if artifact.get("finite_answer_id_transport_used") is not False:
        errors.append("finite_answer_id_transport_used")
    if artifact.get("schema_constraintir_reprompt_used") is not False:
        errors.append("schema_constraintir_reprompt_used")
    specs = list(artifact.get("MODEL_SPECS", []))
    spec_errors = model_spec_errors(specs)
    verdict_class = str(artifact.get("verdict_class"))
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(_expected_prefix(verdict_class)):
        errors.append("honest_verdict_prefix_mismatch")
    if artifact.get("planned_cell_count") != 108:
        errors.append("planned_cell_count_mismatch")
    blocked = verdict_class == "blocked"
    if blocked:
        if "model_roster_mismatch" in spec_errors:
            errors.append("model_roster_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        summary = dict(artifact.get("gate_check_summary") or {})
        if (
            not summary.get("failed_check")
            or "expected_value" not in summary
            or "observed_value" not in summary
        ):
            errors.append("blocked_gate_detail_missing")
        if artifact.get("sota_constraint_bank_ready_score") != 0:
            errors.append("blocked_ready_score_nonzero")
        return list(dict.fromkeys(errors))
    errors.extend(spec_errors)
    if artifact.get("inference_substrate_class") != "model_bounded_generation":
        errors.append("live_substrate_class_mismatch")
    fixture = {
        "base_instance_rows": artifact.get("base_instance_rows", []),
        "variant_rows": artifact.get("variant_rows", []),
        "solver_receipt_rows": artifact.get("solver_receipt_rows", []),
    }
    fixture["fixture_hash"] = sha256_text(canonical_json(fixture))
    errors.extend(fixture_errors(fixture))
    combined = list(artifact.get("rows", []))
    projected_outputs = [_model_output_projection(row) for row in combined]
    if projected_outputs != artifact.get("model_output_rows"):
        errors.append("model_output_projection_mismatch")
    output_by_key = {str(row.get("cell_key")): row for row in artifact.get("model_output_rows", [])}
    parse_by_key = {str(row.get("cell_key")): row for row in artifact.get("parse_rows", [])}
    exact_by_key = {str(row.get("cell_key")): row for row in artifact.get("exact_outcome_rows", [])}
    rebuilt = []
    for key in sorted(output_by_key):
        if key in parse_by_key and key in exact_by_key:
            rebuilt.append({**output_by_key[key], **parse_by_key[key], **exact_by_key[key]})
    if rebuilt != combined:
        errors.append("combined_row_projection_mismatch")
    expected_keys = {str(row["cell_key"]) for row in build_schedule(specs, fixture)}
    keys = [str(row.get("cell_key")) for row in combined]
    key_set = set(keys)
    if not combined and (
        int(artifact.get("completed_cell_count", 0) or 0) > 0
        or artifact.get("family_rows")
        or artifact.get("hardness_stratum_rows")
    ):
        errors.append("aggregate_only_or_cell_key_mismatch")
    if len(keys) != len(key_set) or not key_set.issubset(expected_keys):
        errors.append("aggregate_only_or_cell_key_mismatch")
    if artifact.get("completed_cell_count") != len(key_set):
        errors.append("completed_cell_count_mismatch")
    for row in combined:
        if row.get("proposal_source") != "direct_model_text":
            errors.append("proposal_source_not_direct_text")
        if row.get("finite_answer_id_transport_used") is not False:
            errors.append("row_answer_id_transport_used")
        if row.get("schema_constraintir_reprompt_used") is not False:
            errors.append("row_constraintir_reprompt_used")
        if row.get("raw_persisted_before_parse") is not True:
            errors.append("raw_not_persisted_before_parse")
        if row.get("raw_output_hash") != sha256_text(str(row.get("raw_text", ""))):
            errors.append("raw_output_hash_mismatch")
        if row.get("model_id") not in REQUIRED_MODEL_IDS:
            errors.append("model_substitution")
    identity_valid = _identity_rows_valid(specs, artifact.get("model_identity_confound_rows", []))
    ready_expected = int(
        not errors
        and key_set == expected_keys
        and len(combined) == 108
        and identity_valid
        and sum(int(row.get("row_count", 0) or 0) for row in artifact.get("raw_trace_manifest", []))
        == 108
    )
    if artifact.get("sota_constraint_bank_ready_score") != ready_expected:
        errors.append("ready_score_mismatch")
    if ready_expected and verdict_class != "positive":
        errors.append("complete_verdict_not_positive")
    if not ready_expected and verdict_class != "partial":
        errors.append("incomplete_verdict_not_partial")
    if verdict_class == "partial":
        summary = dict(artifact.get("gate_check_summary") or {})
        if summary.get("failed_check") != "completed_cell_count":
            errors.append("partial_gate_detail_mismatch")
    family_expected, effort_expected = _metric_rows(combined)
    relabel_expected, paraphrase_expected = _pair_rows(combined)
    if artifact.get("family_rows") != family_expected:
        errors.append("family_aggregate_mismatch")
    if artifact.get("hardness_stratum_rows") != effort_expected:
        errors.append("hardness_aggregate_mismatch")
    if artifact.get("relabel_sensitivity_rows") != relabel_expected:
        errors.append("relabel_aggregate_mismatch")
    if artifact.get("paraphrase_consistency_rows") != paraphrase_expected:
        errors.append("paraphrase_aggregate_mismatch")
    return list(dict.fromkeys(errors))


def _storage_probe(path: Path) -> JsonDict:  # pragma: no cover - host filesystem boundary.
    """Measure whether one destination can durably create a sibling file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7129-write-probe-", dir=path.parent)
        os.write(descriptor, b"exact-storage-probe")
        os.fsync(descriptor)
        os.close(descriptor)
        size = Path(name).stat().st_size
        Path(name).unlink()
        return {"writable": True, "probe_bytes": size, "path": str(path)}
    except OSError as exc:
        return {
            "writable": False,
            "probe_bytes": 0,
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _exact_solver_probe() -> JsonDict:  # pragma: no cover - installed solver boundary.
    """Run both the local enumerator and installed Z3 on a fixed canary."""

    try:
        import z3

        solver = z3.Solver()
        canary = z3.Bool("exp7129_canary")
        solver.add(canary)
        z3_result = str(solver.check())
        local = solve_formal(
            "sat_logic", {"variables": ["A"], "clauses": [["A"]]}
        )
        passed = z3_result == "sat" and local["solution_count"] == 1
        return {
            "available": passed,
            "z3_version": z3.get_version_string(),
            "z3_canary": z3_result,
            "local_solution_count": local["solution_count"],
            "local_solver_effort": local["solver_effort"],
        }
    except Exception as exc:  # noqa: BLE001 - exact failure belongs in the artifact.
        return {
            "available": False,
            "z3_version": None,
            "z3_canary": None,
            "local_solution_count": 0,
            "local_solver_effort": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_preconditions(  # pragma: no cover - live host and CUDA boundary.
    *,
    model_specs: Sequence[Mapping[str, Any]],
    result_path: Path,
    fixture_path: Path,
    raw_dir: Path,
) -> JsonDict:
    """Check every stable prerequisite before loading any model weights."""

    exact = _exact_solver_probe()
    fixture_storage = _storage_probe(fixture_path)
    raw_storage = _storage_probe(raw_dir / "write-probe.jsonl")
    result_storage = _storage_probe(result_path)
    gpu = gpu_inventory()
    devices = list(gpu.get("devices", []))
    processes = list(gpu.get("processes", []))
    leases = _lease_probe(devices)
    llama = llama_cpp_probe()
    model_rows = [
        {
            "model_id": row.get("hf_id"),
            "path": row.get("model_path"),
            "exists": Path(str(row.get("model_path") or "")).is_file(),
            "size_bytes": (
                Path(str(row.get("model_path"))).stat().st_size
                if Path(str(row.get("model_path") or "")).is_file()
                else 0
            ),
            "q4_k_m_name_match": "q4_k_m" in Path(str(row.get("model_path") or "")).name.lower(),
        }
        for row in model_specs
    ]
    idle_devices = [
        row
        for row in devices
        if "RTX 3090" in str(row.get("name", ""))
        and int(row.get("utilization_gpu_pct", 100)) == 0
        and int(row.get("memory_used_mb", 10_000)) <= 512
    ]
    checks = [
        gate_row(
            "terminal_blocked_artifact_storage",
            {"writable": True},
            result_storage,
            result_storage.get("writable") is True,
        ),
        gate_row(
            "exact_solver_available",
            {"available": True, "z3_canary": "sat", "local_solution_count": 1},
            exact,
            exact.get("available") is True,
        ),
        gate_row(
            "frozen_fixture_storage",
            {"writable": True},
            fixture_storage,
            fixture_storage.get("writable") is True,
        ),
        gate_row(
            "all_three_cached_q4_k_m_files",
            {model_id: {"exists": True, "q4_k_m": True} for model_id in REQUIRED_MODEL_IDS},
            {
                str(row["model_id"]): {
                    "exists": row["exists"],
                    "q4_k_m": row["q4_k_m_name_match"],
                    "size_bytes": row["size_bytes"],
                }
                for row in model_rows
            },
            len(model_rows) == 3
            and not model_spec_errors(model_specs)
            and all(row["exists"] and row["q4_k_m_name_match"] and row["size_bytes"] > 0 for row in model_rows),
        ),
        gate_row(
            "idle_rtx_3090_count",
            2,
            {
                "count": len(idle_devices),
                "devices": devices,
                "compute_process_count": len(processes),
            },
            gpu.get("query_ok") is True
            and len(devices) == 2
            and len(idle_devices) == 2
            and not processes,
        ),
        gate_row(
            "two_idle_gpu_leases",
            2,
            {
                "available_count": sum(row.get("classification") == "available" for row in leases),
                "lease_rows": leases,
            },
            len(leases) == 2 and all(row.get("classification") == "available" for row in leases),
        ),
        gate_row(
            "cuda_llama_cpp_health",
            {"importable": True, "gpu_offload": True},
            llama,
            llama.get("importable") is True and llama.get("gpu_offload") is True,
        ),
        gate_row(
            "raw_trace_storage",
            {"writable": True},
            raw_storage,
            raw_storage.get("writable") is True,
        ),
    ]
    telemetry = [
        {
            "phase": "preflight",
            "gpu_index": row.get("index"),
            "gpu_uuid": row.get("uuid"),
            "name": row.get("name"),
            "memory_total_mb": row.get("memory_total_mb"),
            "memory_used_mb": row.get("memory_used_mb"),
            "memory_free_mb": row.get("memory_free_mb"),
            "utilization_gpu_pct": row.get("utilization_gpu_pct"),
            "temperature_c": row.get("temperature_c"),
            "compute_process_count": len(processes),
        }
        for row in devices
    ]
    return {
        "all_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "exact_solver": exact,
        "model_file_rows": model_rows,
        "gpu_topology": gpu,
        "gpu_lease_preflight_rows": leases,
        "llama_cpp": llama,
        "gpu_telemetry_rows": telemetry,
    }


def _source_hashes() -> JsonDict:  # pragma: no cover - live repository bytes.
    """Hash the implementation, tests, specs, and named context artifacts."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("research-references.md"),
        Path("openspec/capabilities/constraint-verification/spec.md"),
        Path("openspec/capabilities/verifiable-reasoning/spec.md"),
        Path("openspec/capabilities/research-reporting/spec.md"),
        Path("python/carnot/experiment_7129_v626_sota_constraint_bank.py"),
        Path("scripts/experiments/experiment_7129_v626_sota_constraint_bank.py"),
        Path("tests/python/test_experiment_7129_v626_sota_constraint_bank.py"),
        Path("results/experiment_7080_v620_three_family_entrance_bank.json"),
        Path("results/experiment_7122_v625_sota_ingestion.json"),
    )
    return {
        str(path): sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None
        for path in paths
    }


def _identity_row(  # pragma: no cover - embedded GGUF tokenizer boundary.
    model: Mapping[str, Any]
) -> JsonDict:
    """Bind one cached file to its embedded template and expected family name."""

    probe = embedded_tokenizer_probe(model)
    metadata = dict(probe.get("metadata") or {})
    templates = {
        str(key): value for key, value in metadata.items() if "chat_template" in str(key).lower()
    }
    expected = re.sub(
        r"[^a-z0-9]",
        "",
        str(model["hf_id"]).rsplit("/", 1)[-1].removesuffix("-GGUF").lower(),
    )
    path_name = re.sub(r"[^a-z0-9]", "", Path(str(model["model_path"])).name.lower())
    return {
        "model_id": model["hf_id"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "quantization": model["quantization"],
        "chat_template_source": "embedded_gguf",
        "chat_template_hash": sha256_text(canonical_json(templates)),
        "chat_template_count": len(templates),
        "tokenizer_token_count": probe.get("token_count"),
        "metadata_hash": probe.get("metadata_hash"),
        "identity_matches": bool(probe.get("passed") and templates and expected in path_name),
        "probe_duration_s": probe.get("duration_s"),
        "probe_error": probe.get("error"),
    }


def _memory_telemetry(  # pragma: no cover - live NVIDIA boundary.
    phase: str, gpu: Mapping[str, Any]
) -> list[JsonDict]:
    """Flatten numeric device readings for one acquisition phase."""

    processes = list(gpu.get("processes", []))
    return [
        {
            "phase": phase,
            "gpu_index": row.get("index"),
            "gpu_uuid": row.get("uuid"),
            "name": row.get("name"),
            "memory_total_mb": row.get("memory_total_mb"),
            "memory_used_mb": row.get("memory_used_mb"),
            "memory_free_mb": row.get("memory_free_mb"),
            "utilization_gpu_pct": row.get("utilization_gpu_pct"),
            "temperature_c": row.get("temperature_c"),
            "compute_process_count": len(processes),
        }
        for row in gpu.get("devices", [])
    ]


def _wait_for_vram_release(  # pragma: no cover - live NVIDIA polling boundary.
    baseline: Mapping[str, Any], model_id: str
) -> tuple[JsonDict, JsonDict]:
    """Wait until both devices return to their measured baseline envelope."""

    baseline_rows = [
        {"index": row["index"], "memory_used_mb": row["memory_used_mb"]}
        for row in baseline.get("devices", [])
    ]
    deadline = time.monotonic() + VRAM_RELEASE_TIMEOUT_S
    while True:
        after = gpu_inventory()
        after_rows = [
            {"index": row["index"], "memory_used_mb": row["memory_used_mb"]}
            for row in after.get("devices", [])
        ]
        receipt = build_vram_release_row(
            model_id=model_id, baseline_rows=baseline_rows, after_rows=after_rows
        )
        if receipt.get("passed") is True or time.monotonic() >= deadline:
            return receipt, after
        time.sleep(1)


def _terminalize_leases(  # pragma: no cover - durable kernel-lock boundary.
    leases: Sequence[Any],
    *,
    resident: bool,
    complete: bool,
    release: Mapping[str, Any] | None,
) -> list[JsonDict]:
    """Publish terminal lease journals and release only locks owned here."""

    rows = []
    after = {
        str(row.get("uuid")): int(row.get("memory_used_mb", 0) or 0)
        for row in (release or {}).get("devices", [])
    }
    for lease in leases:
        try:
            if resident:
                lease.transition("unloading")
                lease.transition(
                    "validating",
                    vram_mb=after.get(lease.device_uuid, 0),
                    exit_code=0 if complete else 1,
                    unload_observed=bool(release),
                )
                lease.transition("terminal_complete" if complete else "terminal_blocked")
            else:
                lease.transition("terminal_blocked")
            receipt = lease.release()
            rows.append({**lease.owner_receipt(), "release_receipt": receipt})
        except Exception as exc:  # noqa: BLE001 - cleanup evidence must survive.
            lease.close()
            rows.append(
                {
                    "lease_id": getattr(lease, "lease_id", None),
                    "device_uuid": getattr(lease, "device_uuid", None),
                    "release_error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def _response_fields(response: Mapping[str, Any]) -> tuple[str, str, int, int]:
    """Extract direct content, reasoning, and token counts from llama.cpp output."""

    choice = (list(response.get("choices") or [{}]) or [{}])[0]
    message = dict(choice.get("message") or {})
    usage = dict(response.get("usage") or {})
    return (
        str(message.get("content") or ""),
        str(message.get("reasoning_content") or message.get("reasoning") or ""),
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
    )


def run_model(  # pragma: no cover - required live llama.cpp CUDA boundary.
    *,
    model: Mapping[str, Any],
    schedule_rows: Sequence[Mapping[str, Any]],
    devices: Sequence[Mapping[str, Any]],
    raw_path: Path,
) -> JsonDict:
    """Load one model, invoke missing cells, and release both owned GPUs."""

    from llama_cpp import Llama

    baseline = gpu_inventory()
    leases = []
    telemetry = _memory_telemetry(f"before:{model['hf_id']}", baseline)
    resident = False
    llm: Any = None
    error = None
    lease_rows: list[JsonDict] = []
    try:
        for device in devices:
            lease = lease_api.GpuLease.acquire(
                runtime_dir=LEASE_RUNTIME_DIR,
                task_id=EXPERIMENT_ID,
                device_uuid=str(device["uuid"]),
                expected_model=str(model["model_path"]),
                vram_before_mb=int(device.get("memory_used_mb", 0) or 0),
                ttl_s=MODEL_TIMEOUT_S + 300,
            )
            lease.transition("admitted")
            lease.transition("loading")
            leases.append(lease)
        llm = Llama(
            model_path=str(model["model_path"]),
            n_ctx=int(GENERATION_CONFIG["n_ctx"]),
            n_gpu_layers=int(GENERATION_CONFIG["n_gpu_layers"]),
            n_batch=int(GENERATION_CONFIG["n_batch"]),
            n_ubatch=int(GENERATION_CONFIG["n_ubatch"]),
            main_gpu=int(GENERATION_CONFIG["main_gpu"]),
            split_mode=1,
            tensor_split=list(GENERATION_CONFIG["tensor_split"]),
            seed=RANDOM_SEED,
            use_mmap=True,
            verbose=False,
        )
        loaded = gpu_inventory()
        telemetry.extend(_memory_telemetry(f"resident:{model['hf_id']}", loaded))
        memory = {str(row["uuid"]): int(row["memory_used_mb"]) for row in loaded["devices"]}
        used = {
            str(row.get("gpu_uuid"))
            for row in loaded.get("processes", [])
            if int(row.get("pid", -1)) == os.getpid()
        }
        expected = {str(row["uuid"]) for row in devices}
        resident = used == expected
        if not resident:
            raise RuntimeError(f"dual_gpu_residency_missing:{sorted(used)}")
        for lease in leases:
            lease.transition("resident", vram_mb=memory.get(lease.device_uuid, 0))
            lease.transition("inferencing")
        existing = {str(row["cell_key"]) for row in load_raw_rows(raw_path)}
        for scheduled in schedule_rows:
            if str(scheduled["cell_key"]) in existing:
                continue
            started = time.perf_counter()
            try:
                response = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only the requested direct JSON answer. Do not use tools or answer IDs.",
                        },
                        {"role": "user", "content": str(scheduled["prompt"])},
                    ],
                    max_tokens=int(GENERATION_CONFIG["max_tokens"]),
                    temperature=float(GENERATION_CONFIG["temperature"]),
                    top_p=float(GENERATION_CONFIG["top_p"]),
                    seed=int(scheduled["seed"]),
                )
                raw_text, reasoning, prompt_tokens, completion_tokens = _response_fields(response)
                raw = raw_output_row(
                    scheduled,
                    raw_text=raw_text,
                    reasoning_text=reasoning,
                    raw_response=response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration_s=time.perf_counter() - started,
                    terminal_state="complete",
                )
            except Exception as exc:  # noqa: BLE001 - failed calls remain durable rows.
                raw = raw_output_row(
                    scheduled,
                    raw_text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    duration_s=time.perf_counter() - started,
                    terminal_state="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            persist_raw_row(raw_path, raw)
    except Exception as exc:  # noqa: BLE001 - a load failure leaves retryable evidence.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
        llm = None
        gc.collect()
        release_receipt, after_gpu = _wait_for_vram_release(baseline, str(model["hf_id"]))
        telemetry.extend(_memory_telemetry(f"after:{model['hf_id']}", after_gpu))
        expected_keys = {str(row["cell_key"]) for row in schedule_rows}
        observed_keys = {str(row["cell_key"]) for row in load_raw_rows(raw_path)}
        complete = error is None and expected_keys.issubset(observed_keys) and release_receipt.get("passed") is True
        lease_rows = _terminalize_leases(
            leases, resident=resident, complete=complete, release=after_gpu
        )
    return {
        "model_id": model["hf_id"],
        "raw_path": str(raw_path),
        "raw_rows": load_raw_rows(raw_path),
        "gpu_telemetry_rows": telemetry,
        "gpu_lease_rows": lease_rows,
        "terminal_state": "complete" if error is None else "partial",
        "error": error,
    }


def _raw_path_for(model_id: str, raw_dir: Path) -> Path:  # pragma: no cover - path projection.
    """Map one fixed model identity to its durable JSONL shard."""

    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return raw_dir / f"{slug}.jsonl"


def _raw_manifest(  # pragma: no cover - live raw-file projection.
    model_specs: Sequence[Mapping[str, Any]], raw_dir: Path
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Return content-addressed file receipts and all durable rows."""

    manifest = []
    rows = []
    for model in model_specs:
        path = _raw_path_for(str(model["hf_id"]), raw_dir)
        shard = load_raw_rows(path)
        rows.extend(shard)
        if path.is_file():
            manifest.append(
                {
                    "model_id": model["hf_id"],
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "row_count": len(shard),
                }
            )
    return manifest, rows


def run(  # pragma: no cover - required live command boundary.
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    fixture_path: Path = FIXTURE_PATH,
    raw_dir: Path = RAW_DIR,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write first, preflight, freeze, acquire, reduce, validate, and finish."""

    started = time.perf_counter()
    specs = [deepcopy(dict(row)) for row in (model_specs or MODEL_SPECS)]
    initial = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        preconditions={
            "all_passed": False,
            "checks": [
                gate_row(
                    "preflight_evaluated",
                    "all stable checks pass",
                    "terminal artifact initialized before model setup",
                    False,
                )
            ],
        },
    )
    write_json_atomic(result_path, initial)
    preconditions = collect_preconditions(
        model_specs=specs,
        result_path=result_path,
        fixture_path=fixture_path,
        raw_dir=raw_dir,
    )
    sources = _source_hashes()
    if preconditions["all_passed"] is not True:
        blocked = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            model_specs=specs,
            preconditions=preconditions,
            gpu_telemetry_rows=preconditions["gpu_telemetry_rows"],
            source_artifact_hashes=sources,
        )
        write_json_atomic(result_path, blocked)
        return blocked
    fixture = build_frozen_fixture()
    fixture_failures = fixture_errors(fixture)
    if fixture_failures:
        preconditions["checks"].append(
            gate_row("frozen_fixture_exact_replay", [], fixture_failures, False)
        )
        preconditions["all_passed"] = False
        blocked = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            model_specs=specs,
            preconditions=preconditions,
            fixture=fixture,
            gpu_telemetry_rows=preconditions["gpu_telemetry_rows"],
            source_artifact_hashes=sources,
        )
        write_json_atomic(result_path, blocked)
        return blocked
    write_json_atomic(fixture_path, fixture)
    for spec in specs:
        spec["model_sha256"] = sha256_file(spec["model_path"])
    identity_rows = [_identity_row(spec) for spec in specs]
    identity_errors = [
        row["model_id"] for row in identity_rows if row.get("identity_matches") is not True
    ]
    preconditions["checks"].append(
        gate_row("embedded_model_identity_and_chat_templates", [], identity_errors, not identity_errors)
    )
    preconditions["all_passed"] = not identity_errors
    if identity_errors:
        blocked = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            model_specs=specs,
            preconditions=preconditions,
            fixture=fixture,
            model_identity_rows=identity_rows,
            gpu_telemetry_rows=preconditions["gpu_telemetry_rows"],
            source_artifact_hashes=sources,
        )
        write_json_atomic(result_path, blocked)
        return blocked
    pending = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        preconditions={
            "all_passed": False,
            "checks": [gate_row("model_generation_started", True, False, False)],
        },
        fixture=fixture,
        model_identity_rows=identity_rows,
        gpu_telemetry_rows=preconditions["gpu_telemetry_rows"],
        source_artifact_hashes=sources,
    )
    write_json_atomic(result_path, pending)
    schedule = build_schedule(specs, fixture)
    phase_rows = []
    devices = preconditions["gpu_topology"]["devices"]
    for model in specs:
        phase = run_model(
            model=model,
            schedule_rows=[row for row in schedule if row["model_id"] == model["hf_id"]],
            devices=devices,
            raw_path=_raw_path_for(str(model["hf_id"]), raw_dir),
        )
        phase_rows.append(phase)
    manifest, raw_rows = _raw_manifest(specs, raw_dir)
    telemetry = [
        *preconditions["gpu_telemetry_rows"],
        *(row for phase in phase_rows for row in phase["gpu_telemetry_rows"]),
    ]
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        preconditions=preconditions,
        fixture=fixture,
        raw_rows=raw_rows,
        model_identity_rows=identity_rows,
        raw_trace_manifest=manifest,
        gpu_telemetry_rows=telemetry,
        source_artifact_hashes=sources,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run or validate the exact artifact requested by the experiment contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--fixture-path", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(
        run_date=args.date,
        result_path=args.result_path,
        fixture_path=args.fixture_path,
        raw_dir=args.raw_dir,
    )
    errors = validate_artifact(artifact)
    print(
        canonical_json(
            {
                "result_path": str(args.result_path),
                "planned_cell_count": artifact["planned_cell_count"],
                "completed_cell_count": artifact["completed_cell_count"],
                "sota_constraint_bank_ready_score": artifact[
                    "sota_constraint_bank_ready_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - module command surface.
    raise SystemExit(main())
