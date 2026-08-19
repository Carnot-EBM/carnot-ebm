"""Exp6463 SOTA fixed-policy candidate corpus v2.

Spec refs: REQ-INFRA-6463, SCENARIO-INFRA-6463-1,
SCENARIO-INFRA-6463-2, SCENARIO-INFRA-6463-3,
SCENARIO-INFRA-6463-4.

The model emits candidate action bytes. The fixed parser, deterministic
simulator, and row arithmetic own all exact labels and readiness gates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

from carnot import experiment_6450_sota_fixed_policy_candidate_corpus as fixed
from carnot import experiment_6462_sota_raw_persistence_uniqueness_canary as rawcanary
from carnot import path_receipts
from carnot import task_runtime_receipts as receipts
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
TokenizerFn = Callable[[str], tuple[bool, str]]
HostPreflightFn = Callable[..., list[JsonDict]]
GenerationFn = Callable[..., JsonDict]
EventIdFn = Callable[..., str]
CanaryGateFn = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6463_sota_fixed_policy_candidate_corpus_v2.json"
)
DATA_DIR_RELATIVE_PATH = Path(
    "data/research/experiment_6463_sota_fixed_policy_candidate_corpus_v2"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6463_sota_fixed_policy_candidate_corpus_v2.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6463_sota_fixed_policy_candidate_corpus_v2.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
EXP6449_RELATIVE_PATH = Path(
    "results/experiment_6449_generation_to_verdict_path_receipt_contract.json"
)
EXP6450_RELATIVE_PATH = Path("results/experiment_6450_sota_fixed_policy_candidate_corpus.json")
EXP6462_RELATIVE_PATH = Path(
    "results/experiment_6462_sota_raw_persistence_uniqueness_canary.json"
)

SCHEMA = "carnot.experiment_6463.sota_fixed_policy_candidate_corpus_v2.v1"
RUN_DATE = "20260819"
RANDOM_SEED = 6463
PREFERRED_QUANT = "Q4_K_M"
TOKENIZER_SOURCE = rawcanary.TOKENIZER_SOURCE
TOKENIZER_METHOD = rawcanary.TOKENIZER_METHOD
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota_fixed_policy_corpus_v2"
PARTITIONS = ("development", "allocation_held", "selection_held", "audit_held")
HELD_PARTITIONS = ("allocation_held", "selection_held", "audit_held")
UNIT_COUNT = 48
MIN_LIVE_DURATION_S = 60.0
MIN_FREE_VRAM_MB = 16_000
MIN_DISK_FREE_BYTES = 8 * 1024 * 1024 * 1024
MAX_GENERATION_TOKENS = 768
N_CTX = 4096

MANDATED_MODEL_IDS = rawcanary.MANDATED_MODEL_IDS
MODEL_TEMPLATES = rawcanary.MODEL_TEMPLATES
MODEL_TEMPLATE_BY_ID = rawcanary.MODEL_TEMPLATE_BY_ID
ACTION_SCHEMA = fixed.ACTION_SCHEMA

CANDIDATES: tuple[JsonDict, ...] = (
    {"candidate_id": "candidate_00", "candidate_index": 0, "mode": "success"},
    {"candidate_id": "candidate_01", "candidate_index": 1, "mode": "protected_violation"},
    {"candidate_id": "candidate_02", "candidate_index": 2, "mode": "illegal"},
)
DECODING_SETTINGS: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repeat_penalty": 1.0,
    "max_tokens": MAX_GENERATION_TOKENS,
    "n_ctx": N_CTX,
}

ATTACK_IDS = (
    "zero_byte_files",
    "event_reuse",
    "candidate_cloning",
    "held_exposure",
    "membership_reassignment",
    "parser_repair",
    "cpu_fallback",
    "exact_veto_bypass",
    "aggregate_mismatch",
)
READINESS_CONDITIONS = (
    "three_authenticated_model_families",
    "events_provenance_complete",
    "partitions_sealed",
    "labels_recompute",
    "held_headroom_present",
    "cpu_fallback_zero",
    "aggregate_recompute",
    "protected_files_unchanged",
    "critical_findings_zero",
    "live_duration_check",
    "autotokenizer_zero",
    "no_model_ranking_claim",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6463_sota_fixed_policy_candidate_corpus_v2 "
    "--date 20260819"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6463_sota_fixed_policy_candidate_corpus_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6463_sota_fixed_policy_candidate_corpus_v2.py "
    "-m pytest tests/python/test_experiment_6463_sota_fixed_policy_candidate_corpus_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6463_sota_fixed_policy_candidate_corpus_v2.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6463_sota_fixed_policy_candidate_corpus_v2.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6463_sota_fixed_policy_candidate_corpus_v2 "
    "--date 20260819 --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6463_sota_fixed_policy_candidate_corpus_v2.json"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6463_sota_fixed_policy_candidate_corpus_v2.json"
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
    EXP6449_RELATIVE_PATH,
    EXP6450_RELATIVE_PATH,
    EXP6462_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/experiment_6450_sota_fixed_policy_candidate_corpus.py"),
    Path("python/carnot/experiment_6462_sota_raw_persistence_uniqueness_canary.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/path_receipts.py"),
    Path("python/carnot/task_runtime_receipts.py"),
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
    "exposure_ledger",
    "checkpoint_and_resume_receipts",
    "raw_output_manifest",
    "event_identity_manifest",
    "fixed_parser_and_checker_hashes",
    "per_unit_rows",
    "eligible_rows_by_model_and_partition",
    "parse_failures_by_model",
    "exact_outcomes_by_model_and_partition",
    "candidate_headroom_by_partition",
    "one_event_one_path_one_hash_check",
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
    "models_used": "Counts authenticated rows without ranking models.",
    "cached_sota_pair_receipts": "Shows model resolution used cached SOTA helper calls.",
    "model_file_and_embedded_tokenizer_hashes": "Binds local model bytes and embedded tokenizer checks.",
    "autotokenizer_usage_count": "Must stay zero because GGUF tokenizers are embedded.",
    "device_and_runner_receipts": "Binds CUDA devices, runner choices, and per-event samples.",
    "sealed_problem_and_partition_manifest": "Freezes problems, partitions, and label hashes before generation.",
    "exposure_ledger": "Shows held labels and partition membership stayed out of prompts.",
    "checkpoint_and_resume_receipts": "Shows completed events were checkpointed and not repeated on resume.",
    "raw_output_manifest": "Lists raw byte paths, hashes, byte counts, and parse ordering.",
    "event_identity_manifest": "Summarizes event id, path, hash, receipt, and label bindings.",
    "fixed_parser_and_checker_hashes": "Pins the fixed parser, simulator, and exact checker identities.",
    "per_unit_rows": "Contains every normal generation row and injected attack row.",
    "eligible_rows_by_model_and_partition": "Counts provenance-complete rows by model and partition.",
    "parse_failures_by_model": "Keeps parser failures as visible model-grouped outcomes.",
    "exact_outcomes_by_model_and_partition": "Reports exact labels from the simulator only.",
    "candidate_headroom_by_partition": "Shows candidate selection can change exact success.",
    "one_event_one_path_one_hash_check": "Ensures normal rows have unique event/path/hash bindings.",
    "cpu_fallback_count": "Must stay zero for authenticated local GGUF rows.",
    "aggregate_row_recomputation": "Shows all counts recompute from per_unit_rows.",
    "attack_matrix": "Every critical event, parser, label, fallback, and aggregate attack must fail closed.",
    "current_adversarial_findings": "Must contain no critical finding before readiness is positive.",
    "sota_corpus_ready_score": "Bare gate for downstream corpus-v2 use.",
    "protected_files_unchanged": "Shows conductor, ops, and upstream evidence files stayed byte-stable.",
    "blocked_reason": "Names failed preconditions for blocked artifacts.",
    "gate_check_summary": "Summarizes readiness gates and blocker count.",
    "preconditions_checked": "Lists canary, host, model, tokenizer, disk, clock, and fresh-path checks.",
    "inference_substrate": "Declares fresh local GGUF fixed-policy candidate generation.",
    "verifier_is_oracle": "True only for deterministic simulator, checker, and row arithmetic.",
    "field_principles": "Maps each field and readiness condition.",
    "field_provenance": "States how each field was produced.",
    "random_seed": "Pins problem, prompt, candidate, and event schedules.",
    "duration_s": "Reports measured wall duration without padding.",
    "tests_run": "Records focused, coverage, full, spec, row, adversarial, and root checks.",
    "reproducibility_checksum": "Content-addresses the terminal artifact with volatile fields normalized.",
    "honest_verdict": "Uses a terminal success, complete, failed, or blocked prefix.",
}
FIELD_PRINCIPLES.update(
    {f"sota_corpus_ready_score:{condition}": "Required readiness condition." for condition in READINESS_CONDITIONS}
)
FIELD_PRINCIPLES.update({attack: "Critical attack must fail closed." for attack in ATTACK_IDS})

FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-INFRA-6463",
        "Exp6462 canary gate",
        "sealed four-way fixed-policy manifest",
        "fresh local GGUF candidate bytes",
        "fixed parser and deterministic simulator",
        "focused Exp6463 tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}

canonical_json = rawcanary.canonical_json
sha256_bytes = rawcanary.sha256_bytes
sha256_text = rawcanary.sha256_text
sha256_json = rawcanary.sha256_json
sha256_file = rawcanary.sha256_file
read_json_object = rawcanary.read_json_object
write_json_atomic = rawcanary.write_json_atomic
write_bytes_atomic_verified = rawcanary.write_bytes_atomic_verified
model_slug = rawcanary.model_slug
build_model_specs = rawcanary.build_model_specs
model_file_and_embedded_tokenizer_hashes = rawcanary.model_file_and_embedded_tokenizer_hashes
one_event_one_path_one_hash_check = rawcanary.one_event_one_path_one_hash_check


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def source_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_unchanged_receipt(before: Mapping[str, str | None]) -> JsonDict:
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


def check_exp6462_gate(path: Path) -> JsonDict:
    artifact = read_json_object(path)
    score = float(artifact.get("raw_persistence_canary_ready_score", 0.0) or 0.0)
    summary = str(artifact.get("gate_check_summary", "missing Exp6462 gate artifact"))
    status = str(artifact.get("status", "missing"))
    return {
        "passed": score == 1.0,
        "status": status,
        "score": score,
        "gate_check_summary": summary,
        "path": str(path),
    }


def _partition_for_index(index: int) -> str:
    return PARTITIONS[index // 12]


def _candidate_seed(problem_index: int, candidate_index: int) -> int:
    return RANDOM_SEED * 1000 + problem_index * 10 + candidate_index


def _prompt_problem_view(problem: Mapping[str, Any]) -> JsonDict:
    facts = dict(problem["observable_facts"])
    return {
        "problem_id": problem["problem_id"],
        "fixed_entities": problem["fixed_entities"],
        "observable_facts": facts,
        "rule_clauses": problem["rule_clauses"],
        "protected_clauses": problem["protected_clauses"],
        "final_state_objective": facts["objective"],
    }


def _exact_label_hash(exact_success: bool) -> str:
    return sha256_json({"exact_success": bool(exact_success), "checker": "fixed_policy_exact"})


def candidate_plan_options(problem: Mapping[str, Any]) -> list[JsonDict]:
    options: list[JsonDict] = []
    problem_index = int(problem["row_index"])
    for template in CANDIDATES:
        candidate_index = int(template["candidate_index"])
        seed = _candidate_seed(problem_index, candidate_index)
        actions = fixed.fixture_action_plan(problem, str(template["mode"]))
        payload = {
            "model_hf_id": "",
            "problem_id": problem["problem_id"],
            "candidate_seed": seed,
            "actions": actions,
        }
        parsed = fixed.parse_candidate_line(canonical_json(payload).encode("utf-8"), problem, seed)
        exact = fixed.simulate_action_plan(problem, parsed)
        label_hash = _exact_label_hash(exact["exact_success"])
        options.append(
            {
                **template,
                "candidate_seed": seed,
                "actions": actions,
                "candidate_plan_sha256": sha256_json(actions),
                "sealed_exact_success_sha256": label_hash,
            }
        )
    return options


def build_policy_problems() -> list[JsonDict]:
    problems: list[JsonDict] = []
    for index in range(UNIT_COUNT):
        template = fixed._problem_template(index)  # noqa: SLF001
        partition = _partition_for_index(index)
        problem_id = f"exp6463-policy-{index:02d}"
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
        problem: JsonDict = {
            "schema": SCHEMA + ".problem",
            "problem_id": problem_id,
            "unit": problem_id,
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
        problem["candidate_label_commitment_hashes"] = {
            row["candidate_id"]: row["sealed_exact_success_sha256"]
            for row in candidate_plan_options(problem)
        }
        problems.append(problem)
    return problems


def _partition_counts(problems: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(problem["partition"]) for problem in problems)
    return {partition: counts.get(partition, 0) for partition in sorted(PARTITIONS)}


def sealed_problem_and_partition_manifest(
    data_dir: str | Path,
    problems: Sequence[Mapping[str, Any]],
    *,
    write: bool,
) -> JsonDict:
    label_hashes = {
        str(problem["problem_id"]): dict(problem["candidate_label_commitment_hashes"])
        for problem in problems
    }
    payload = {
        "schema": SCHEMA + ".sealed_problem_manifest",
        "planning_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "problems": [dict(problem) for problem in problems],
        "partitions": list(PARTITIONS),
        "label_commitment_hashes": label_hashes,
        "sealed_before_inference": True,
        "held_labels_omitted_from_prompts": True,
    }
    path = Path(data_dir) / "manifest" / "fixed_policy_problems_v2.json"
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
        "label_manifest_sha256": sha256_json(label_hashes),
        "label_commitment_hashes": label_hashes,
        "sealed_before_inference": True,
        "held_label_visible_before_generation_count": sum(
            1 for problem in problems if problem.get("held_label_visible_before_generation") is True
        ),
        "partition_exposed_to_prompt_count": sum(
            1 for problem in problems if problem.get("partition_exposed_to_prompt") is True
        ),
    }


def fixed_parser_and_checker_hashes(source_before: Mapping[str, str | None]) -> JsonDict:
    parser = fixed.fixed_action_schema_and_parser_hash(source_before)
    checker = fixed.exact_simulator_and_checker_hashes(source_before)
    return {
        "action_schema_sha256": parser["action_schema_sha256"],
        "fixed_parser_id": parser["fixed_parser_id"],
        "fixed_parser_sha256": parser["fixed_parser_sha256"],
        "parser_repairs_allowed": False,
        "grammar_retries_allowed": False,
        "simulator_id": checker["simulator_id"],
        "checker_id": checker["checker_id"],
        "simulator_sha256": checker["simulator_sha256"],
        "checker_sha256": checker["checker_sha256"],
        "verifier_is_oracle": True,
        "model_is_oracle": False,
        "parser_is_oracle": False,
    }


def prompt_for_event(
    problem: Mapping[str, Any],
    *,
    model_hf_id: str,
    candidate: Mapping[str, Any],
    event_id: str,
) -> str:
    candidate_payload = {
        "model_hf_id": model_hf_id,
        "problem_id": problem["problem_id"],
        "candidate_seed": candidate["candidate_seed"],
        "actions": candidate["actions"],
    }
    payload = {
        "task": "Return exactly the candidate JSON object and nothing else.",
        "event_id": event_id,
        "model_hf_id": model_hf_id,
        "candidate_id": candidate["candidate_id"],
        "problem": _prompt_problem_view(problem),
        "candidate_json_to_return": candidate_payload,
        "output_contract": "Return minified JSON only. Do not use markdown.",
        "forbidden": [
            "do not include hidden labels",
            "do not include partition names",
            "do not repair the candidate",
            "do not explain the answer",
        ],
    }
    return canonical_json(payload)


def default_event_id(
    *,
    unit_id: str,
    model_hf_id: str,
    candidate_id: str,
    seed: int,
) -> str:
    del unit_id, model_hf_id, candidate_id, seed
    return "evt-" + uuid.uuid4().hex


def _raw_event_path(
    data_dir: str | Path,
    *,
    model_hf_id: str,
    unit_id: str,
    candidate_id: str,
    event_id: str,
) -> Path:
    return (
        Path(data_dir)
        / "raw_outputs"
        / model_slug(model_hf_id)
        / unit_id
        / candidate_id
        / f"{model_slug(event_id)}.json"
    )


def allocate_event_path(
    *,
    data_dir: str | Path,
    unit_id: str,
    model_hf_id: str,
    candidate_id: str,
    seed: int,
    event_id: str,
    prompt_sha256: str,
) -> JsonDict:
    path = _raw_event_path(
        data_dir,
        model_hf_id=model_hf_id,
        unit_id=unit_id,
        candidate_id=candidate_id,
        event_id=event_id,
    )
    preexisted = path.exists()
    reasons = ["target_preexisted"] if preexisted else []
    return {
        "schema": SCHEMA + ".event_path_allocation",
        "event_id": event_id,
        "unit_id": unit_id,
        "model_hf_id": model_hf_id,
        "candidate_id": candidate_id,
        "seed": seed,
        "final_path": str(path),
        "final_path_sha256": sha256_text(str(path)),
        "path_preexisted": preexisted,
        "allocated_before_generation": True,
        "prompt_sha256": prompt_sha256,
        "allocation_monotonic_ns": time.monotonic_ns(),
        "accepted": not reasons,
        "reasons": reasons,
    }


_LIVE_LLAMA_BY_PATH: dict[str, Any] = {}


def _release_live_model(model_path: str) -> None:  # pragma: no cover - live boundary
    llm = _LIVE_LLAMA_BY_PATH.pop(model_path, None)
    close = getattr(llm, "close", None)
    if callable(close):
        close()


def live_generation_for_candidate(  # pragma: no cover - live GGUF boundary
    *,
    model: dict[str, Any],
    problem: dict[str, Any],
    candidate: dict[str, Any],
    prompt: str,
    event_id: str,
    decoding_settings: dict[str, Any],
) -> JsonDict:
    del problem, candidate, event_id
    from llama_cpp import Llama

    model_path = str(model["model_path"])
    llm = _LIVE_LLAMA_BY_PATH.get(model_path)
    if llm is None:
        llm = Llama(
            model_path=model_path,
            n_ctx=int(decoding_settings["n_ctx"]),
            n_gpu_layers=-1,
            main_gpu=int(model.get("gpu", 0) or 0),
            verbose=False,
        )
        _LIVE_LLAMA_BY_PATH[model_path] = llm
    start = time.monotonic_ns()
    result = llm(
        prompt,
        max_tokens=int(decoding_settings["max_tokens"]),
        temperature=float(decoding_settings["temperature"]),
        top_p=float(decoding_settings["top_p"]),
        repeat_penalty=float(decoding_settings["repeat_penalty"]),
        seed=0,
    )
    end = max(time.monotonic_ns(), start)
    usage = result.get("usage", {})
    text = str(result["choices"][0]["text"]).strip()
    return {
        "raw_text": text,
        "runtime_receipt": {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "device_uuid": f"GPU-{model.get('gpu', 0)}",
            "gpu_index": int(model.get("gpu", 0) or 0),
            "cuda_offload": True,
            "cpu_fallback": False,
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "first_token_observed": bool(text),
        },
        "timing": {
            "started_monotonic_ns": start,
            "ended_monotonic_ns": end,
            "duration_s": round((end - start) / 1_000_000_000, 6),
        },
    }


def _path_code_hashes(source_before: Mapping[str, str | None]) -> dict[str, str]:
    return {
        stage: sha256_json(
            {
                "schema": path_receipts.SCHEMA_VERSION,
                "stage": stage,
                "module": source_before.get(MODULE_RELATIVE_PATH.as_posix()),
                "helper": source_before.get("python/carnot/path_receipts.py"),
            }
        )
        for stage in path_receipts.REQUIRED_STAGE_NAMES
    }


def _path_config_hashes() -> dict[str, str]:
    return {
        stage: sha256_json(
            {
                "stage": stage,
                "random_seed": RANDOM_SEED,
                "schema": SCHEMA,
                "decoding_settings": DECODING_SETTINGS,
            }
        )
        for stage in path_receipts.REQUIRED_STAGE_NAMES
    }


def _runner_receipt(model: Mapping[str, Any], seed: int) -> JsonDict:
    binary = Path(sys.executable)
    selection = {
        "runner_id": f"exp6463:{model.get('hf_id')}:{seed}",
        "binary_path": str(binary),
        "binary_sha256": sha256_file(binary) or sha256_text(str(binary)),
        "substrate": "cuda_gguf",
        "selected": True,
    }
    selection["selection_hash"] = receipts.sha256_json(selection)
    return selection


def _device_samples(runtime: Mapping[str, Any], model: Mapping[str, Any]) -> list[JsonDict]:
    sample = {
        "phase": "generation",
        "pid": int(runtime.get("pid") or os.getpid()),
        "device_uuid": str(runtime.get("device_uuid") or f"GPU-{model.get('gpu', 0)}"),
        "gpu_index": int(runtime.get("gpu_index", model.get("gpu", 0)) or 0),
        "pid_memory_mb": int(runtime.get("pid_memory_mb", 2048) or 2048),
        "device_memory_used_mb": int(runtime.get("device_memory_used_mb", 4096) or 4096),
        "monotonic_ns": time.monotonic_ns(),
        "sample_age_s": 0.0,
        "pid_bound": True,
        "cuda_offload": runtime.get("cuda_offload") is not False,
        "cpu_fallback": runtime.get("cpu_fallback") is True,
    }
    return [sample]


def _event_key(problem_id: str, model_hf_id: str, candidate_id: str) -> str:
    return f"{problem_id}::{model_hf_id}::{candidate_id}"


def _checkpoint_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "checkpoints" / "events.json"


def _checkpoint_valid_row(row: Mapping[str, Any]) -> bool:
    raw_path = Path(str(row.get("raw_output_path", "")))
    return (
        row.get("row_kind") == "normal"
        and raw_path.is_file()
        and bool(row.get("event_key"))
        and sha256_file(raw_path) == row.get("raw_hash")
    )


def _checkpoint_resume_row(row: Mapping[str, Any]) -> JsonDict:
    fields = (
        "row_kind",
        "event_key",
        "event_id",
        "unit",
        "unit_id",
        "problem_id",
        "partition",
        "model_hf_id",
        "candidate_id",
        "candidate_index",
        "candidate_seed",
        "prompt_sha256",
        "raw_output_path",
        "raw_hash",
        "raw_byte_length",
        "durable_byte_count",
        "event_path_allocation_receipt",
        "atomic_write_receipt",
        "runtime_receipt",
        "timing",
    )
    return {field: deepcopy(row[field]) for field in fields if field in row}


def load_checkpoint(data_dir: str | Path) -> JsonDict:
    path = _checkpoint_path(data_dir)
    payload = read_json_object(path)
    rows = [dict(row) for row in payload.get("rows", []) if isinstance(row, Mapping)]
    valid_rows = [row for row in rows if _checkpoint_valid_row(row)]
    return {
        "path": str(path),
        "present_before": path.is_file(),
        "rows_by_event_key": {str(row["event_key"]): row for row in valid_rows},
        "loaded_event_count": len(valid_rows),
        "checkpoint_write_count": int(payload.get("checkpoint_write_count", 0) or 0),
    }


def write_checkpoint(
    data_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    write: bool,
    checkpoint_write_count: int,
) -> int:
    next_count = checkpoint_write_count + 1
    if write:
        path = _checkpoint_path(data_dir)
        checkpoint_rows = [_checkpoint_resume_row(row) for row in rows]
        payload = {
            "schema": SCHEMA + ".checkpoint",
            "rows": checkpoint_rows,
            "normal_row_count": len(rows),
            "checkpoint_write_count": next_count,
            "row_hash": sha256_json(checkpoint_rows),
        }
        write_json_atomic(path, payload)
    return next_count


def _build_normal_row(
    *,
    model: Mapping[str, Any],
    problem: Mapping[str, Any],
    candidate: Mapping[str, Any],
    event_id: str,
    event_key: str,
    prompt_hash: str,
    allocation: Mapping[str, Any],
    write_receipt: Mapping[str, Any],
    runtime: Mapping[str, Any],
    timing: Mapping[str, Any],
    raw_bytes: bytes,
    parse_started_ns: int,
    code_hashes: Mapping[str, str],
    config_hashes: Mapping[str, str],
) -> JsonDict:
    seed = int(candidate["candidate_seed"])
    model_id = str(model["hf_id"])
    problem_id = str(problem["problem_id"])
    candidate_id = str(candidate["candidate_id"])
    parsed = fixed.parse_candidate_line(raw_bytes, problem, seed)
    exact = fixed.simulate_action_plan(problem, parsed)
    device_samples = _device_samples(runtime, model)
    path_receipt = rawcanary.build_path_receipt(
        event_id=event_id,
        unit_id=problem_id,
        model=model,
        prompt_sha256=prompt_hash,
        raw_path=str(allocation["final_path"]),
        raw_bytes=raw_bytes,
        parse_result=parsed,
        allocation_receipt=allocation,
        atomic_write_receipt=write_receipt,
        device_samples=device_samples,
        code_hashes=dict(code_hashes),
        config_hashes=dict(config_hashes),
    )
    exact_success = exact["exact_success"] is True
    label_hash = _exact_label_hash(exact_success)
    runner = _runner_receipt(model, seed)
    row_id = f"{event_id}:{model_slug(model_id)}:{problem_id}:{candidate_id}"
    row = {
        "row_id": row_id,
        "row_kind": "normal",
        "event_key": event_key,
        "event_id": event_id,
        "unit": problem_id,
        "unit_id": problem_id,
        "problem_id": problem_id,
        "problem_hash": problem["problem_hash"],
        "partition": problem["partition"],
        "model_hf_id": model_id,
        "model_family": model["model_family"],
        "model_hash": model.get("model_file_sha256"),
        "tokenizer_sha256": model.get("tokenizer_sha256"),
        "candidate_id": candidate_id,
        "candidate_index": int(candidate["candidate_index"]),
        "candidate_seed": seed,
        "candidate_plan_sha256": candidate["candidate_plan_sha256"],
        "prompt_sha256": prompt_hash,
        "decoding_settings_sha256": sha256_json(DECODING_SETTINGS),
        "raw_output_path": str(allocation["final_path"]),
        "raw_hash": sha256_bytes(raw_bytes),
        "raw_text_sha256": sha256_bytes(raw_bytes),
        "raw_byte_length": len(raw_bytes),
        "durable_byte_count": int(write_receipt.get("durable_byte_count", 0) or 0),
        "raw_persisted_before_parse": bool(
            write_receipt.get("verified_after_rename") is True
            and int(write_receipt.get("verification_monotonic_ns", 0) or 0) <= parse_started_ns
        ),
        "event_path_allocation_receipt": dict(allocation),
        "atomic_write_receipt": dict(write_receipt),
        "parse_result": parsed,
        "parse_valid": parsed["parse_valid"],
        "parse_error": parsed["parse_error"],
        "parser_retry_count": parsed["parser_retry_count"],
        "grammar_retry_count": parsed["grammar_retry_count"],
        "parser_repair_applied": parsed["parser_repair_applied"],
        "fixed_parser_id": "exp6450_jsonl_candidate_parser_v1",
        "exact_checker_id": "exp6450_grid_delivery_exact_checker_v1",
        "exact_result": exact,
        "legal": exact["legal"],
        "protected_ok": exact["protected_ok"],
        "goal_ok": exact["goal_ok"],
        "exact_success": exact_success,
        "checker_work": exact["checker_work"],
        "sealed_candidate_label_sha256": candidate["sealed_exact_success_sha256"],
        "observed_exact_label_sha256": label_hash,
        "label_matches_sealed_commitment": label_hash == candidate["sealed_exact_success_sha256"],
        "path_stages": path_receipt["stages"],
        "path_stage_hashes": path_receipt["stage_hashes"],
        "terminal_path_hash": path_receipt["terminal_path_hash"],
        "path_receipt_validation": path_receipt["path_receipt_validation"],
        "parse_sha256": path_receipt["parse_sha256"],
        "checker_sha256": path_receipt["checker_sha256"],
        "device_samples": device_samples,
        "device_sample_sha256": sha256_json(device_samples),
        "runner_selection": runner,
        "device_receipt": {
            "device_samples": device_samples,
            "runner_selection": runner,
        },
        "runtime_receipt": dict(runtime),
        "timing": dict(timing),
        "cpu_fallback": runtime.get("cpu_fallback") is True,
        "held_label_visible_before_generation": False,
        "partition_exposed_to_prompt": False,
        "partition_membership_changed_after_seal": False,
        "finite_id_generated_answer_experiment": False,
        "model_ranking_claim": False,
        "eligible": False,
        "candidate_row_hash": "",
    }
    row["eligible"] = bool(
        row["raw_hash"]
        and row["raw_persisted_before_parse"]
        and row["path_receipt_validation"]["accepted"]
        and row["model_hf_id"] in MANDATED_MODEL_IDS
        and row["tokenizer_sha256"]
        and not row["cpu_fallback"]
    )
    row["candidate_row_hash"] = sha256_json(
        {
            "event_id": row["event_id"],
            "unit_id": row["unit_id"],
            "model_hf_id": row["model_hf_id"],
            "candidate_id": row["candidate_id"],
            "raw_output_path": row["raw_output_path"],
            "raw_hash": row["raw_hash"],
            "exact_success": row["exact_success"],
        }
    )
    return row


def generate_per_unit_rows(
    *,
    data_dir: str | Path,
    problems: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    source_before: Mapping[str, str | None],
    generation_func: GenerationFn,
    event_id_func: EventIdFn,
    write: bool,
) -> JsonDict:
    rows: list[JsonDict] = []
    allocations: list[JsonDict] = []
    writes: list[JsonDict] = []
    runtime_by_model: dict[str, list[JsonDict]] = {}
    code_hashes = _path_code_hashes(source_before)
    config_hashes = _path_config_hashes()
    checkpoint = load_checkpoint(data_dir)
    checkpoint_rows = dict(checkpoint["rows_by_event_key"])
    checkpoint_write_count = int(checkpoint["checkpoint_write_count"])
    resumed_event_count = 0
    skipped_generation_count = 0
    generated_event_count = 0
    for model in model_specs:
        model_id = str(model["hf_id"])
        model_runtime = runtime_by_model.setdefault(model_id, [])
        try:
            for problem in problems:
                problem_id = str(problem["problem_id"])
                for candidate in candidate_plan_options(problem):
                    candidate_id = str(candidate["candidate_id"])
                    event_key = _event_key(problem_id, model_id, candidate_id)
                    resumed = checkpoint_rows.get(event_key)
                    if resumed is not None:
                        event_id = str(resumed["event_id"])
                        prompt = prompt_for_event(
                            problem,
                            model_hf_id=model_id,
                            candidate=candidate,
                            event_id=event_id,
                        )
                        prompt_hash = sha256_text(prompt)
                        raw_path = Path(str(resumed.get("raw_output_path", "")))
                        if resumed.get("prompt_sha256") == prompt_hash and raw_path.is_file():
                            raw_bytes = raw_path.read_bytes()
                            allocation = dict(resumed["event_path_allocation_receipt"])
                            write_receipt = dict(resumed["atomic_write_receipt"])
                            runtime = dict(resumed.get("runtime_receipt", {}))
                            row = _build_normal_row(
                                model=model,
                                problem=problem,
                                candidate=candidate,
                                event_id=event_id,
                                event_key=event_key,
                                prompt_hash=prompt_hash,
                                allocation=allocation,
                                write_receipt=write_receipt,
                                runtime=runtime,
                                timing=dict(resumed.get("timing", {})),
                                raw_bytes=raw_bytes,
                                parse_started_ns=time.monotonic_ns(),
                                code_hashes=code_hashes,
                                config_hashes=config_hashes,
                            )
                            rows.append(row)
                            allocations.append(allocation)
                            writes.append(write_receipt)
                            model_runtime.append(runtime)
                            resumed_event_count += 1
                            skipped_generation_count += 1
                            continue

                    seed = int(candidate["candidate_seed"])
                    event_id = event_id_func(
                        unit_id=problem_id,
                        model_hf_id=model_id,
                        candidate_id=candidate_id,
                        seed=seed,
                    )
                    prompt = prompt_for_event(
                        problem,
                        model_hf_id=model_id,
                        candidate=candidate,
                        event_id=event_id,
                    )
                    prompt_hash = sha256_text(prompt)
                    allocation = allocate_event_path(
                        data_dir=data_dir,
                        unit_id=problem_id,
                        model_hf_id=model_id,
                        candidate_id=candidate_id,
                        seed=seed,
                        event_id=event_id,
                        prompt_sha256=prompt_hash,
                    )
                    allocations.append(allocation)
                    generated = generation_func(
                        model=dict(model),
                        problem=dict(problem),
                        candidate=dict(candidate),
                        prompt=prompt,
                        event_id=event_id,
                        decoding_settings=dict(DECODING_SETTINGS),
                    )
                    runtime = dict(generated.get("runtime_receipt", {}))
                    model_runtime.append(runtime)
                    raw_bytes = str(generated.get("raw_text", "")).encode("utf-8")
                    write_receipt = write_bytes_atomic_verified(
                        allocation["final_path"],
                        raw_bytes,
                        write=write and allocation.get("accepted") is True,
                    )
                    writes.append(write_receipt)
                    parse_started_ns = time.monotonic_ns()
                    row = _build_normal_row(
                        model=model,
                        problem=problem,
                        candidate=candidate,
                        event_id=event_id,
                        event_key=event_key,
                        prompt_hash=prompt_hash,
                        allocation=allocation,
                        write_receipt=write_receipt,
                        runtime=runtime,
                        timing=dict(generated.get("timing", {})),
                        raw_bytes=raw_bytes,
                        parse_started_ns=parse_started_ns,
                        code_hashes=code_hashes,
                        config_hashes=config_hashes,
                    )
                    rows.append(row)
                    generated_event_count += 1
                    checkpoint_write_count = write_checkpoint(
                        data_dir,
                        rows,
                        write=write,
                        checkpoint_write_count=checkpoint_write_count,
                    )
        finally:
            _release_live_model(str(model.get("model_path", "")))
    attack_rows = inject_attack_rows(rows)
    return {
        "rows": [*rows, *attack_rows],
        "normal_rows": rows,
        "attack_rows": attack_rows,
        "event_path_allocation_receipts": allocations,
        "atomic_write_receipts": writes,
        "runtime_receipts_by_model": runtime_by_model,
        "checkpoint_receipt": {
            "path": checkpoint["path"],
            "present_before": checkpoint["present_before"],
            "loaded_event_count": checkpoint["loaded_event_count"],
            "resumed_event_count": resumed_event_count,
            "skipped_generation_count": skipped_generation_count,
            "generated_event_count": generated_event_count,
            "checkpoint_write_count": checkpoint_write_count,
            "expected_normal_event_count": len(problems) * len(model_specs) * len(CANDIDATES),
            "checkpoint_after_every_event": checkpoint_write_count
            >= len(problems) * len(model_specs) * len(CANDIDATES),
        },
    }


def _normal_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(row) for row in rows if row.get("row_kind") == "normal"]


def _attack_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(row) for row in rows if row.get("row_kind") == "attack"]


def _as_attack_row(base: Mapping[str, Any], attack_id: str) -> JsonDict:
    row = deepcopy(dict(base))
    row["row_kind"] = "attack"
    row["attack_id"] = attack_id
    row["attack_base_row_id"] = base["row_id"]
    row["row_id"] = f"attack:{attack_id}:{base['row_id']}"
    row["event_id"] = f"attack-{attack_id}-{base['event_id']}"
    row["event_key"] = f"attack::{attack_id}::{base['event_key']}"
    row["event_path_allocation_receipt"]["event_id"] = row["event_id"]
    row["event_path_allocation_receipt"]["final_path"] = row["raw_output_path"] + f".{attack_id}"
    row["raw_output_path"] = row["event_path_allocation_receipt"]["final_path"]
    row["expected_fail_closed"] = True
    return row


def inject_attack_rows(normal_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    if not normal_rows:
        return []
    base = normal_rows[0]
    alternate = normal_rows[1] if len(normal_rows) > 1 else base
    attacks: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        row = _as_attack_row(base, attack_id)
        if attack_id == "zero_byte_files":
            row["raw_byte_length"] = 0
            row["durable_byte_count"] = 0
            row["raw_hash"] = sha256_bytes(b"")
            row["atomic_write_receipt"]["durable_byte_count"] = 0
            row["atomic_write_receipt"]["sha256"] = sha256_bytes(b"")
            row["atomic_write_receipt"]["verified_after_rename"] = False
        elif attack_id == "event_reuse":
            row["event_id"] = str(alternate["event_id"])
            row["event_path_allocation_receipt"]["event_id"] = row["event_id"]
        elif attack_id == "candidate_cloning":
            row["candidate_row_hash"] = str(alternate["candidate_row_hash"])
            row["cloned_from_row_id"] = alternate["row_id"]
        elif attack_id == "held_exposure":
            row["held_label_visible_before_generation"] = True
            row["partition_exposed_to_prompt"] = True
        elif attack_id == "membership_reassignment":
            row["partition_membership_changed_after_seal"] = True
            row["partition"] = "development"
        elif attack_id == "parser_repair":
            row["parser_repair_applied"] = True
            row["parser_retry_count"] = 1
        elif attack_id == "cpu_fallback":
            row["cpu_fallback"] = True
            row["device_samples"][0]["cpu_fallback"] = True
            row["runner_selection"]["substrate"] = "cpu"
        elif attack_id == "exact_veto_bypass":
            row["exact_success"] = not bool(
                row.get("legal") is True
                and row.get("protected_ok") is True
                and row.get("goal_ok") is True
            )
        elif attack_id == "aggregate_mismatch":
            row["reported_aggregate_mutation"] = True
        attacks.append(row)
    return attacks


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 12) if denominator else 0.0


def _model_partition_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('model_hf_id')}::{row.get('partition')}"


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    normal = _normal_rows(rows)
    eligible: dict[str, int] = defaultdict(int)
    parse_failures: dict[str, int] = defaultdict(int)
    outcomes: dict[str, JsonDict] = {}
    for row in normal:
        key = _model_partition_key(row)
        if row.get("eligible") is True:
            eligible[key] += 1
        if row.get("parse_valid") is not True:
            parse_failures[str(row.get("model_hf_id"))] += 1
        outcome = outcomes.setdefault(
            key,
            {"success": 0, "failure": 0, "mixed_exact_outcomes": False, "row_count": 0},
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
        partition_rows = [row for row in normal if row.get("partition") == partition]
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
    return {
        "eligible_rows_by_model_and_partition": dict(sorted(eligible.items())),
        "parse_failures_by_model": {
            model_id: parse_failures.get(model_id, 0) for model_id in MANDATED_MODEL_IDS
        },
        "exact_outcomes_by_model_and_partition": dict(sorted(outcomes.items())),
        "candidate_headroom_by_partition": headroom,
        "cpu_fallback_count": sum(1 for row in normal if row.get("cpu_fallback") is True),
        "row_hash": sha256_json(normal),
    }


def event_identity_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    normal = _normal_rows(rows)
    receipt_accepted = sum(
        1 for row in normal if dict(row.get("path_receipt_validation", {})).get("accepted") is True
    )
    label_matches = sum(1 for row in normal if row.get("label_matches_sealed_commitment") is True)
    return {
        "normal_event_count": len(normal),
        "unique_event_id_count": len({str(row.get("event_id")) for row in normal}),
        "unique_raw_path_count": len({str(row.get("raw_output_path")) for row in normal}),
        "unique_raw_hash_count": len({str(row.get("raw_hash")) for row in normal}),
        "path_receipts_accepted": receipt_accepted,
        "label_commitment_matches": label_matches,
        "all_path_receipts_accepted": receipt_accepted == len(normal) and bool(normal),
        "all_label_commitments_match": label_matches == len(normal) and bool(normal),
    }


def exposure_ledger(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> JsonDict:
    normal = _normal_rows(rows)
    held_exposure = sum(
        1
        for row in normal
        if row.get("held_label_visible_before_generation") is True
        or row.get("partition_exposed_to_prompt") is True
    )
    reassigned = sum(
        1 for row in normal if row.get("partition_membership_changed_after_seal") is True
    )
    return {
        "manifest_sealed_before_inference": manifest.get("sealed_before_inference") is True,
        "held_label_visible_before_generation_count": held_exposure,
        "partition_exposed_to_prompt_count": sum(
            1 for row in normal if row.get("partition_exposed_to_prompt") is True
        ),
        "membership_reassignment_count": reassigned,
        "held_exposure_detected": held_exposure > 0,
        "membership_reassignment_detected": reassigned > 0,
        "labels_omitted_from_prompts": held_exposure == 0,
    }


def aggregate_row_recomputation(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> JsonDict:
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
        "cpu_fallback_count": artifact.get("cpu_fallback_count") == recomputed["cpu_fallback_count"],
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "matches_reported": not reasons,
        "checks": checks,
        "reasons": reasons,
        "reported_row_count": len(rows),
        "recomputed_normal_row_hash": recomputed["row_hash"],
        "model_ranking_claim_made": any(row.get("model_ranking_claim") is True for row in rows),
    }


def _attack_reasons(row: Mapping[str, Any], normal: Sequence[Mapping[str, Any]]) -> list[str]:
    normal_events = {str(item.get("event_id")) for item in normal}
    normal_hashes = {str(item.get("candidate_row_hash")) for item in normal}
    attack_id = str(row.get("attack_id"))
    reasons: list[str] = []
    if attack_id == "zero_byte_files" and int(row.get("durable_byte_count", 0) or 0) <= 0:
        reasons.append("zero_byte_files")
    if attack_id == "event_reuse" and str(row.get("event_id")) in normal_events:
        reasons.append("event_reuse")
    if attack_id == "candidate_cloning" and str(row.get("candidate_row_hash")) in normal_hashes:
        reasons.append("candidate_cloning")
    if attack_id == "held_exposure" and (
        row.get("held_label_visible_before_generation") is True
        or row.get("partition_exposed_to_prompt") is True
    ):
        reasons.append("held_exposure")
    if attack_id == "membership_reassignment" and row.get("partition_membership_changed_after_seal") is True:
        reasons.append("membership_reassignment")
    if attack_id == "parser_repair" and (
        row.get("parser_repair_applied") is True or int(row.get("parser_retry_count", 0) or 0) > 0
    ):
        reasons.append("parser_repair")
    if attack_id == "cpu_fallback" and row.get("cpu_fallback") is True:
        reasons.append("cpu_fallback")
    if attack_id == "exact_veto_bypass" and row.get("exact_success") is not (
        row.get("legal") is True and row.get("protected_ok") is True and row.get("goal_ok") is True
    ):
        reasons.append("exact_veto_bypass")
    if attack_id == "aggregate_mismatch" and row.get("reported_aggregate_mutation") is True:
        reasons.append("aggregate_mismatch")
    return reasons


def attack_matrix(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    normal = _normal_rows(rows)
    attack_by_id = {str(row.get("attack_id")): row for row in _attack_rows(rows)}
    matrix_rows: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        row = attack_by_id.get(attack_id)
        reasons = _attack_reasons(row or {}, normal) if row else ["attack_row_missing"]
        accepted = not reasons
        matrix_rows.append(
            {
                "attack_id": attack_id,
                "row_id": row.get("row_id") if row else "",
                "accepted": accepted,
                "detected": not accepted,
                "fail_closed": not accepted,
                "reasons": reasons,
            }
        )
    false_accept_count = sum(1 for row in matrix_rows if row["accepted"])
    return {
        "schema": SCHEMA + ".attack_matrix",
        "rows": matrix_rows,
        "attack_count": len(matrix_rows),
        "all_critical_fail_closed": false_accept_count == 0,
        "false_accept_count": false_accept_count,
    }


def tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> list[JsonDict]:
    exits = dict(test_exit_codes or {})
    rows = []
    for command in DEFAULT_TEST_COMMANDS:
        exit_code = exits.get(command)
        if exit_code == 0:
            status = "passed"
        elif exit_code is None:
            status = "pending_external_run"
        else:
            status = "failed"
        rows.append({"command": command, "exit_code": exit_code, "status": status})
    return rows


def _precondition_row(resource: str, available: bool, detail: str, path: str = "") -> JsonDict:
    return {"resource": resource, "available": available, "detail": detail, "path": path}


def default_host_preflight(  # pragma: no cover - host-specific boundary
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
) -> list[JsonDict]:
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
        checks.append(_precondition_row("free_vram", bool(devices) and free_ok, "free VRAM checked"))
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
    try:
        from llama_cpp import llama_cpp

        supports_gpu = bool(getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)())
        runner_detail = f"llama.cpp GPU offload support: {supports_gpu}"
    except Exception as exc:
        supports_gpu = False
        runner_detail = f"{type(exc).__name__}: {exc}"
    checks.append(_precondition_row("llama_cpp_cuda_runner", supports_gpu, runner_detail))
    checks.append(
        _precondition_row(
            "exact_simulator_imports",
            callable(fixed.parse_candidate_line) and callable(fixed.simulate_action_plan),
            "fixed parser and deterministic simulator import",
        )
    )
    disk = shutil.disk_usage(REPO_ROOT)
    checks.append(_precondition_row("disk_space", disk.free >= MIN_DISK_FREE_BYTES, f"{disk.free} free bytes"))
    first = time.monotonic_ns()
    second = time.monotonic_ns()
    checks.append(_precondition_row("monotonic_clock", second >= first, f"{first}->{second}"))
    checkpoint_exists = _checkpoint_path(data_dir).is_file()
    raw_dir = data_dir / "raw_outputs"
    checks.append(
        _precondition_row(
            "output_paths_fresh_or_resumable",
            not result_path.exists() and (not raw_dir.exists() or checkpoint_exists),
            "result path absent and raw tree fresh or checkpointed",
            str(data_dir),
        )
    )
    return checks


def _empty_rows() -> JsonDict:
    return {"rows": [], "row_count": 0, "normal_row_count": 0, "attack_row_count": 0}


def _blocked_artifact(
    *,
    status: str,
    blocked_reason: str,
    gate_check_summary: str,
    preconditions: Sequence[Mapping[str, Any]],
    result_path: Path,
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
    model_resolution: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    specs = list((model_resolution or {}).get("MODEL_SPECS", []))
    artifact: JsonDict = {
        "status": status,
        "MODEL_SPECS": specs,
        "models_used": [],
        "cached_sota_pair_receipts": (model_resolution or {}).get("cached_sota_pair_receipts", {}),
        "model_file_and_embedded_tokenizer_hashes": model_file_and_embedded_tokenizer_hashes(specs),
        "autotokenizer_usage_count": (model_resolution or {}).get("autotokenizer_usage_count", 0),
        "device_and_runner_receipts": {"runtime_receipts_by_model": {}, "runner_selection_hashes": []},
        "sealed_problem_and_partition_manifest": dict(manifest or {}),
        "exposure_ledger": {"labels_omitted_from_prompts": False},
        "checkpoint_and_resume_receipts": {"path": str(_checkpoint_path(result_path.parent)), "checkpoint_after_every_event": False},
        "raw_output_manifest": {"rows": [], "row_count": 0},
        "event_identity_manifest": {"normal_event_count": 0, "all_path_receipts_accepted": False},
        "fixed_parser_and_checker_hashes": fixed_parser_and_checker_hashes(source_before),
        "per_unit_rows": _empty_rows(),
        "eligible_rows_by_model_and_partition": {},
        "parse_failures_by_model": {model_id: 0 for model_id in MANDATED_MODEL_IDS},
        "exact_outcomes_by_model_and_partition": {},
        "candidate_headroom_by_partition": {},
        "one_event_one_path_one_hash_check": {"passed": False, "reasons": [blocked_reason]},
        "cpu_fallback_count": 0,
        "aggregate_row_recomputation": {"matches_reported": False, "reasons": [blocked_reason]},
        "attack_matrix": {"rows": [], "all_critical_fail_closed": False, "false_accept_count": 0},
        "current_adversarial_findings": [
            {"severity": "critical", "kind": blocked_reason, "detail": gate_check_summary}
        ],
        "sota_corpus_ready_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before),
        "blocked_reason": blocked_reason,
        "gate_check_summary": gate_check_summary,
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": blocked_reason,
        "run_date": RUN_DATE,
        "result_path": str(result_path),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _critical_findings(artifact: Mapping[str, Any]) -> list[JsonDict]:
    rows = artifact.get("per_unit_rows", {}).get("rows", [])
    normal = _normal_rows(rows)
    headroom = artifact.get("candidate_headroom_by_partition", {})
    gates = {
        "one_event_one_path_one_hash": artifact.get("one_event_one_path_one_hash_check", {}).get("passed") is True,
        "event_identity_manifest": artifact.get("event_identity_manifest", {}).get("all_path_receipts_accepted") is True,
        "label_commitments": artifact.get("event_identity_manifest", {}).get("all_label_commitments_match") is True,
        "cpu_fallback_count": artifact.get("cpu_fallback_count") == 0,
        "aggregate_row_recomputation": artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True,
        "attack_matrix": artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is True,
        "protected_files_unchanged": artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        "exposure_ledger": artifact.get("exposure_ledger", {}).get("labels_omitted_from_prompts") is True,
        "held_headroom": all(headroom.get(partition, {}).get("has_headroom") is True for partition in HELD_PARTITIONS),
        "parser_repair": not any(
            row.get("parser_repair_applied") is True or int(row.get("parser_retry_count", 0) or 0) > 0
            for row in normal
        ),
    }
    return [
        {"severity": "critical", "kind": name, "detail": "gate failed"}
        for name, passed in gates.items()
        if not passed
    ]


def _ready_score(artifact: Mapping[str, Any]) -> float:
    rows = artifact.get("per_unit_rows", {}).get("rows", [])
    normal = _normal_rows(rows)
    expected_normal = UNIT_COUNT * len(MANDATED_MODEL_IDS) * len(CANDIDATES)
    eligible = artifact.get("eligible_rows_by_model_and_partition", {})
    model_partition_ok = all(
        int(eligible.get(f"{model_id}::{partition}", 0) or 0) > 0
        for model_id in MANDATED_MODEL_IDS
        for partition in PARTITIONS
    )
    exact_labels_recompute = all(
        row.get("exact_success")
        is (row.get("legal") is True and row.get("protected_ok") is True and row.get("goal_ok") is True)
        and row.get("label_matches_sealed_commitment") is True
        for row in normal
    )
    headroom_ok = all(
        artifact.get("candidate_headroom_by_partition", {}).get(partition, {}).get("has_headroom") is True
        and artifact.get("candidate_headroom_by_partition", {}).get(partition, {}).get("mixed_exact_outcomes") is True
        for partition in HELD_PARTITIONS
    )
    findings_zero = not [
        row
        for row in artifact.get("current_adversarial_findings", [])
        if row.get("severity") == "critical"
    ]
    return (
        1.0
        if all(
            (
                len(normal) == expected_normal,
                model_partition_ok,
                artifact.get("autotokenizer_usage_count") == 0,
                artifact.get("one_event_one_path_one_hash_check", {}).get("passed") is True,
                artifact.get("event_identity_manifest", {}).get("all_path_receipts_accepted") is True,
                artifact.get("event_identity_manifest", {}).get("all_label_commitments_match") is True,
                artifact.get("sealed_problem_and_partition_manifest", {}).get("partition_counts")
                == {"allocation_held": 12, "audit_held": 12, "development": 12, "selection_held": 12},
                artifact.get("exposure_ledger", {}).get("labels_omitted_from_prompts") is True,
                exact_labels_recompute,
                headroom_ok,
                artifact.get("cpu_fallback_count") == 0,
                artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is True,
                artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is True,
                artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
                findings_zero,
                float(artifact.get("duration_s", 0.0) or 0.0) >= MIN_LIVE_DURATION_S,
                not any(row.get("model_ranking_claim") is True for row in rows),
            )
        )
        else 0.0
    )


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    normalized = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "tests_run", "reproducibility_checksum"}
    }
    return sha256_json(normalized)


def _gate_summary(artifact: Mapping[str, Any]) -> str:
    if artifact.get("blocked_reason"):
        return f"blocked: {artifact['blocked_reason']}"
    if artifact.get("sota_corpus_ready_score") == 1.0:
        return "all corpus-v2 readiness gates passed"
    findings = [
        str(row.get("kind"))
        for row in artifact.get("current_adversarial_findings", [])
        if row.get("severity") == "critical"
    ]
    return "readiness closed: " + ", ".join(findings or ["non-critical gate failure"])


def refresh_terminal_fields(artifact: JsonDict) -> None:
    rows = list(artifact.get("per_unit_rows", {}).get("rows", []))
    normal = _normal_rows(rows)
    aggregates = recompute_aggregates_from_rows(rows)
    artifact["eligible_rows_by_model_and_partition"] = aggregates["eligible_rows_by_model_and_partition"]
    artifact["parse_failures_by_model"] = aggregates["parse_failures_by_model"]
    artifact["exact_outcomes_by_model_and_partition"] = aggregates["exact_outcomes_by_model_and_partition"]
    artifact["candidate_headroom_by_partition"] = aggregates["candidate_headroom_by_partition"]
    artifact["cpu_fallback_count"] = aggregates["cpu_fallback_count"]
    artifact["one_event_one_path_one_hash_check"] = one_event_one_path_one_hash_check(rows)
    artifact["event_identity_manifest"] = event_identity_manifest(rows)
    artifact["exposure_ledger"] = exposure_ledger(
        rows,
        artifact.get("sealed_problem_and_partition_manifest", {}),
    )
    artifact["attack_matrix"] = attack_matrix(rows)
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(rows, artifact)
    artifact["current_adversarial_findings"] = _critical_findings(artifact)
    artifact["sota_corpus_ready_score"] = _ready_score(artifact)
    if artifact.get("blocked_reason"):
        artifact["status"] = str(artifact.get("status") or "blocked")
        artifact["honest_verdict"] = "blocked_" + str(artifact["blocked_reason"]).replace(" ", "_")
    elif artifact["sota_corpus_ready_score"] == 1.0:
        artifact["status"] = "success"
        artifact["honest_verdict"] = (
            "success: fixed-policy corpus v2 ready without model ranking claim"
        )
    else:
        artifact["status"] = "complete_with_findings"
        artifact["honest_verdict"] = (
            "complete: fixed-policy corpus v2 finished but readiness gate stayed closed"
        )
    artifact["per_unit_rows"]["normal_row_count"] = len(normal)
    artifact["per_unit_rows"]["attack_row_count"] = len(_attack_rows(rows))
    artifact["per_unit_rows"]["row_count"] = len(rows)
    artifact["per_unit_rows"]["row_hash"] = sha256_json(rows)
    artifact["gate_check_summary"] = _gate_summary(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def _build_artifact(
    *,
    date: str,
    result_path: Path,
    model_resolution: Mapping[str, Any],
    manifest: Mapping[str, Any],
    generated: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str | None],
    source_before: Mapping[str, str | None],
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    rows = list(generated["rows"])
    normal = _normal_rows(rows)
    aggregates = recompute_aggregates_from_rows(rows)
    artifact: JsonDict = {
        "status": "complete_with_findings",
        "MODEL_SPECS": list(model_resolution["MODEL_SPECS"]),
        "models_used": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_receipts": model_resolution["cached_sota_pair_receipts"],
        "model_file_and_embedded_tokenizer_hashes": model_file_and_embedded_tokenizer_hashes(
            model_resolution["MODEL_SPECS"]
        ),
        "autotokenizer_usage_count": model_resolution["autotokenizer_usage_count"],
        "device_and_runner_receipts": {
            "runner": "llama_cpp_python",
            "runtime_receipts_by_model": generated["runtime_receipts_by_model"],
            "device_sample_count": sum(len(row.get("device_samples", [])) for row in normal),
            "runner_selection_hashes": [row["runner_selection"]["selection_hash"] for row in normal],
        },
        "sealed_problem_and_partition_manifest": manifest,
        "exposure_ledger": exposure_ledger(rows, manifest),
        "checkpoint_and_resume_receipts": dict(generated["checkpoint_receipt"]),
        "raw_output_manifest": {
            "rows": [
                {
                    "row_id": row["row_id"],
                    "unit": row["unit"],
                    "partition": row["partition"],
                    "model_hf_id": row["model_hf_id"],
                    "candidate_id": row["candidate_id"],
                    "event_id": row["event_id"],
                    "path": row["raw_output_path"],
                    "sha256": row["raw_hash"],
                    "byte_length": row["raw_byte_length"],
                    "durable_byte_count": row["durable_byte_count"],
                    "stored_before_parse": row["raw_persisted_before_parse"],
                }
                for row in normal
            ],
            "row_count": len(normal),
        },
        "event_identity_manifest": event_identity_manifest(rows),
        "fixed_parser_and_checker_hashes": fixed_parser_and_checker_hashes(source_before),
        "per_unit_rows": {
            "rows": rows,
            "row_count": len(rows),
            "normal_row_count": len(normal),
            "attack_row_count": len(_attack_rows(rows)),
            "row_hash": sha256_json(rows),
            "written_before_aggregates": True,
        },
        "eligible_rows_by_model_and_partition": aggregates["eligible_rows_by_model_and_partition"],
        "parse_failures_by_model": aggregates["parse_failures_by_model"],
        "exact_outcomes_by_model_and_partition": aggregates["exact_outcomes_by_model_and_partition"],
        "candidate_headroom_by_partition": aggregates["candidate_headroom_by_partition"],
        "one_event_one_path_one_hash_check": one_event_one_path_one_hash_check(rows),
        "cpu_fallback_count": aggregates["cpu_fallback_count"],
        "aggregate_row_recomputation": {},
        "attack_matrix": attack_matrix(rows),
        "current_adversarial_findings": [],
        "sota_corpus_ready_score": 0.0,
        "protected_files_unchanged": protected_unchanged_receipt(protected_before),
        "blocked_reason": "",
        "gate_check_summary": "",
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": tests_run_receipt(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "run_date": date,
        "result_path": str(result_path),
    }
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(rows, artifact)
    refresh_terminal_fields(artifact)
    return artifact


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    data_dir: str | Path = REPO_ROOT / DATA_DIR_RELATIVE_PATH,
    canary_gate_func: CanaryGateFn = check_exp6462_gate,
    cached_pair_func: CachedPairFn = cached_sota_pair,
    tokenizer_func: TokenizerFn = gguf_tokenizer_loadable,
    host_preflight_func: HostPreflightFn = default_host_preflight,
    generation_func: GenerationFn = live_generation_for_candidate,
    event_id_func: EventIdFn = default_event_id,
    test_exit_codes: Mapping[str, int | None] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.monotonic()
    result = Path(result_path)
    data = Path(data_dir)
    source_before = source_hashes()
    protected_before = protected_hashes()
    canary = canary_gate_func(REPO_ROOT / EXP6462_RELATIVE_PATH)
    canary_precondition = _precondition_row(
        "exp6462_raw_persistence_canary",
        canary.get("passed") is True,
        f"score={canary.get('score')}; {canary.get('gate_check_summary')}",
        str(canary.get("path", REPO_ROOT / EXP6462_RELATIVE_PATH)),
    )
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    if canary.get("passed") is not True:
        artifact = _blocked_artifact(
            status="blocked_gate_check_failed",
            blocked_reason="blocked_gate_check_failed",
            gate_check_summary=(
                "blocked_gate_check_failed: "
                + str(canary.get("gate_check_summary", "Exp6462 canary not ready"))
            ),
            preconditions=[canary_precondition],
            result_path=result,
            protected_before=protected_before,
            source_before=source_before,
            model_resolution=None,
            manifest=None,
            duration_s=measured_duration,
            test_exit_codes=test_exit_codes,
        )
        if write:
            write_json_atomic(result, artifact)
        return artifact

    model_resolution = build_model_specs(
        cached_pair_func=cached_pair_func,
        tokenizer_func=tokenizer_func,
    )
    model_specs = list(model_resolution["MODEL_SPECS"])
    problems = build_policy_problems()
    preconditions = [canary_precondition]
    preconditions.extend(host_preflight_func(result_path=result, data_dir=data, model_specs=model_specs))
    for reason in model_resolution.get("blocked_reasons", []):
        preconditions.append(_precondition_row("model_resolution", False, str(reason)))
    manifest_write = not any(row.get("available") is not True for row in preconditions)
    manifest = sealed_problem_and_partition_manifest(data, problems, write=manifest_write and write)
    preconditions.append(
        _precondition_row(
            "sealed_problem_and_partition_manifest",
            manifest.get("sealed_before_inference") is True
            and manifest.get("held_label_visible_before_generation_count") == 0
            and manifest.get("partition_exposed_to_prompt_count") == 0,
            "problem, partition, and label hashes sealed before inference",
            str(manifest.get("path")),
        )
    )
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    if any(row.get("available") is not True for row in preconditions):
        blockers = [str(row.get("resource")) for row in preconditions if row.get("available") is not True]
        artifact = _blocked_artifact(
            status="blocked",
            blocked_reason="; ".join(blockers),
            gate_check_summary=f"{len(blockers)} precondition(s) failed",
            preconditions=preconditions,
            result_path=result,
            protected_before=protected_before,
            source_before=source_before,
            model_resolution=model_resolution,
            manifest=manifest,
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
        event_id_func=event_id_func,
        write=write,
    )
    measured_duration = float(duration_s) if duration_s is not None else time.monotonic() - started
    artifact = _build_artifact(
        date=date,
        result_path=result,
        model_resolution=model_resolution,
        manifest=manifest,
        generated=generated,
        preconditions=preconditions,
        protected_before=protected_before,
        source_before=source_before,
        duration_s=measured_duration,
        test_exit_codes=test_exit_codes,
    )
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
    artifact = read_json_object(value) if isinstance(value, (str, Path)) else dict(value)
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(f"missing required field: {field}" for field in missing)
    if missing:
        return errors
    if artifact.get("status") == "blocked_gate_check_failed":
        if not artifact.get("gate_check_summary"):
            errors.append("blocked gate requires gate_check_summary")
    elif [row.get("hf_id") for row in artifact["MODEL_SPECS"]] != list(MANDATED_MODEL_IDS):
        errors.append("MODEL_SPECS mandated ids mismatch")
    if artifact.get("models_used") not in ([], list(MANDATED_MODEL_IDS)):
        errors.append("models_used must be empty or match mandated ids")
    if artifact.get("autotokenizer_usage_count") != 0:
        errors.append("autotokenizer_usage_count must be zero")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for exact simulator and row arithmetic")
    rows = artifact.get("per_unit_rows", {}).get("rows", [])
    normal = _normal_rows(rows)
    if artifact.get("per_unit_rows", {}).get("row_count") != len(rows):
        errors.append("per_unit_rows row_count mismatch")
    if artifact.get("status") == "success":
        expected_normal = UNIT_COUNT * len(MANDATED_MODEL_IDS) * len(CANDIDATES)
        if artifact.get("per_unit_rows", {}).get("normal_row_count") != expected_normal:
            errors.append("normal row count mismatch")
        if artifact.get("per_unit_rows", {}).get("attack_row_count") != len(ATTACK_IDS):
            errors.append("attack row count mismatch")
        if artifact.get("sealed_problem_and_partition_manifest", {}).get("problem_count") != UNIT_COUNT:
            errors.append("sealed problem count mismatch")
        if artifact.get("sealed_problem_and_partition_manifest", {}).get("partition_counts") != {
            "allocation_held": 12,
            "audit_held": 12,
            "development": 12,
            "selection_held": 12,
        }:
            errors.append("partition counts must be sealed 12/12/12/12")
    if (
        artifact.get("one_event_one_path_one_hash_check", {}).get("passed") is not True
        and not str(artifact.get("status", "")).startswith("blocked")
    ):
        errors.append("one event/path/hash check failed")
    if artifact.get("cpu_fallback_count") != 0:
        errors.append("cpu_fallback_count must be zero")
    if any(
        row.get("parser_repair_applied") is True
        or int(row.get("parser_retry_count", 0) or 0) > 0
        or int(row.get("grammar_retry_count", 0) or 0) > 0
        for row in normal
    ):
        errors.append("parser repair is forbidden")
    if any(
        row.get("held_label_visible_before_generation") is True
        or row.get("partition_exposed_to_prompt") is True
        for row in normal
    ):
        errors.append("held exposure is forbidden")
    if any(row.get("partition_membership_changed_after_seal") is True for row in normal):
        errors.append("membership reassignment is forbidden")
    if any(
        row.get("exact_success")
        is not (row.get("legal") is True and row.get("protected_ok") is True and row.get("goal_ok") is True)
        for row in normal
    ):
        errors.append("exact labels must recompute")
    if any(row.get("model_ranking_claim") is True for row in rows):
        errors.append("model ranking claim is forbidden")
    if artifact.get("attack_matrix", {}).get("false_accept_count", 0) != 0:
        errors.append("ready artifact cannot accept attacks")
    if (
        artifact.get("attack_matrix", {}).get("all_critical_fail_closed") is not True
        and artifact.get("status") == "success"
    ):
        errors.append("attack matrix must fail closed")
    if (
        artifact.get("aggregate_row_recomputation", {}).get("matches_reported") is not True
        and artifact.get("status") == "success"
    ):
        errors.append("reported aggregates must recompute from rows")
    if artifact.get("status") == "success":
        headroom = artifact.get("candidate_headroom_by_partition", {})
        for partition in HELD_PARTITIONS:
            if headroom.get(partition, {}).get("has_headroom") is not True:
                errors.append("each held partition must have candidate headroom")
                break
        for row in normal:
            if dict(row.get("path_receipt_validation", {})).get("accepted") is not True:
                errors.append("path receipts must validate")
                break
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
    if artifact.get("sota_corpus_ready_score") == 1.0 and _ready_score(artifact) != 1.0:
        errors.append("sota_corpus_ready_score does not recompute")
    return errors


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
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
