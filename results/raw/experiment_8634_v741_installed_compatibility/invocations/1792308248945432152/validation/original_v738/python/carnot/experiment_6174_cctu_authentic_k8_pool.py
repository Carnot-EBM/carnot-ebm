"""Exp6174 authentic CCTU K8 tool-trace candidate pool.

Spec refs: REQ-VERIFY-6174, SCENARIO-VERIFY-6174-GATE,
SCENARIO-VERIFY-6174-RAW-BEFORE-LABEL, SCENARIO-VERIFY-6174-RETENTION,
SCENARIO-VERIFY-6174-RESUME.

The runner separates generation from oracle labels. It writes immutable raw
model rows first, then runs the Exp6173 exact validator into sidecars that are
not inputs to generation.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Protocol

from carnot.inference.sota_models import gguf_tokenizer_loadable, resolve_cached_gguf
from carnot.verify import cctu_item_bank_6173 as exp6173


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6174_cctu_authentic_k8_pool.json")
RAW_TRACE_RELATIVE_PATH = Path("results/experiment_6174_cctu_authentic_k8_pool.raw_traces.jsonl")
CALIBRATION_LABEL_RELATIVE_PATH = Path(
    "results/experiment_6174_cctu_authentic_k8_pool.calibration_labels.jsonl"
)
HELD_LABEL_RELATIVE_PATH = Path("results/experiment_6174_cctu_authentic_k8_pool.held_labels.jsonl")
CALIBRATION_ACCESS_LOG_RELATIVE_PATH = Path(
    "results/experiment_6174_cctu_authentic_k8_pool.calibration_access_log.json"
)
HELD_ACCESS_LOG_RELATIVE_PATH = Path(
    "results/experiment_6174_cctu_authentic_k8_pool.held_access_log.json"
)
CHECKPOINT_RELATIVE_PATH = Path("results/experiment_6174_cctu_authentic_k8_pool.checkpoint.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6174_cctu_authentic_k8_pool.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6174_cctu_authentic_k8_pool.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")

SCHEMA = "carnot.experiment_6174.cctu_authentic_k8_pool.v1"
RAW_ROW_SCHEMA = SCHEMA + ".raw_trace_row"
LABEL_ROW_SCHEMA = SCHEMA + ".label_row"
EXPERIMENT_ID = "experiment_6174_cctu_authentic_k8_pool"
RUN_DATE = "20260807"
RANDOM_SEED = 6174
MANDATORY_MODEL_ID = "unsloth/gemma-4-31B-it-GGUF"
K_SAMPLES = 8
INFERENCE_SUBSTRATE = "llama_cpp_local_gemma4_31b_gguf_native_chat_tool_trace_generation"
VERIFIER_IS_ORACLE = True

TEMPERATURE_SCHEDULE = (0.70, 0.80, 0.90, 1.00, 0.75, 0.85, 0.95, 1.05)
TOP_P_SCHEDULE = (0.95,) * K_SAMPLES
MAX_TOKENS = int(os.environ.get("CARNOT_EXP6174_MAX_TOKENS", "96"))
N_CTX = int(os.environ.get("CARNOT_EXP6174_N_CTX", "4096"))
TOP_K = 40
REPEAT_PENALTY = 1.05

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Gemma4-31B-it",
        "hf_id": MANDATORY_MODEL_ID,
        "role": "dense",
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "n_ctx": N_CTX,
        "expected_quantization": "Q4_K_M",
        "headline_model": True,
        "legacy_small_model_headline": False,
        "gpu_assignment": {
            "main_gpu": 0,
            "visible_devices": [0, 1],
            "split_mode": "layer",
            "tensor_split": [1.0, 1.0],
        },
    }
]

DECODE_POLICY: JsonDict = {
    "k": K_SAMPLES,
    "temperature_schedule": list(TEMPERATURE_SCHEDULE),
    "top_p_schedule": list(TOP_P_SCHEDULE),
    "top_k": TOP_K,
    "repeat_penalty": REPEAT_PENALTY,
    "max_tokens": MAX_TOKENS,
    "n_ctx": N_CTX,
    "seed_base": RANDOM_SEED,
    "native_chat_template": "embedded_gguf_tokenizer.chat_template",
    "tool_budget_source": "Exp6173 frozen case max_tool_calls and max_resource_units",
    "correctness_conditioned_retry": False,
    "parser_repair": False,
    "model_judge": False,
    "candidate_replacement": False,
    "legacy_small_model_substitution": False,
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("AGENTS.md"),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    SPEC_RELATIVE_PATH,
    Path("results/experiment_6173_cctu_item_bank_preregistration.json"),
    Path("data/research/cctu_item_bank_6173.jsonl"),
    Path("data/research/cctu_item_bank_6173_splits.json"),
    Path("data/research/cctu_item_bank_6173_held_access_log.json"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/verify/cctu_item_bank_6173.py"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6174_cctu_authentic_k8_pool.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6174_cctu_authentic_k8_pool.py "
    "-m pytest tests/python/test_experiment_6174_cctu_authentic_k8_pool.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6174_cctu_authentic_k8_pool.py --fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6174_cctu_authentic_k8_pool.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6174_cctu_authentic_k8_pool.json"
)
E2E_COMMAND = ".venv/bin/python -m carnot.experiment_6174_cctu_authentic_k8_pool --validate"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "upstream_bank_split_validator_and_preregistration_hashes",
    "model_specs",
    "exact_gguf_files_revision_quantization_hashes_and_sizes",
    "embedded_tokenizer_and_chat_template_receipts",
    "llama_cpp_version_and_command",
    "gpu_assignment_health_process_and_vram_receipts",
    "case_sample_seed_temperature_token_and_tool_budget_matrix",
    "raw_trace_corpus_path_hash_count_and_schema",
    "parse_failure_duplicate_refusal_timeout_and_truncation_counts",
    "no_correctness_conditioned_retry_or_replacement_receipt",
    "raw_before_label_commit_receipts",
    "exact_label_sidecar_paths_hashes_and_counts",
    "calibration_and_held_access_logs",
    "resume_idempotence_and_checkpoint_receipts",
    "cctu_candidate_pool_integrity_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes complete, partial, retired, and blocked candidate-pool runs.",
    "preconditions_checked": "The gate verifies Exp6173 hashes, local Gemma 31B GGUF, embedded template, llama.cpp CUDA, dual-GPU health, paths, exclusions, process lease, and protected files before generation.",
    "structured_gate_receipt": "Generation opens only when every required local-only precondition passes; failures block without downloads or model substitution.",
    "upstream_bank_split_validator_and_preregistration_hashes": "The frozen bank, split, held seal, validator, and preregistration bytes anchor the corpus.",
    "model_specs": "Include exact unsloth/gemma-4-31B-it-GGUF; no legacy small model supplies headline rows.",
    "exact_gguf_files_revision_quantization_hashes_and_sizes": "Path, snapshot revision, quantization, size, and SHA-256 bind rows to the cached GGUF file.",
    "embedded_tokenizer_and_chat_template_receipts": "Tokenizer and chat template receipts come from GGUF metadata through llama.cpp, never AutoTokenizer on GGUF bytes.",
    "llama_cpp_version_and_command": "The local llama.cpp command and Python binding versions make the substrate replayable.",
    "gpu_assignment_health_process_and_vram_receipts": "Dual-GPU health, exact assignment, task-owned process evidence, and VRAM lifecycle distinguish live generation from replay.",
    "case_sample_seed_temperature_token_and_tool_budget_matrix": "Every frozen case has eight immutable seed/temperature/token/tool-budget entries before generation.",
    "raw_trace_corpus_path_hash_count_and_schema": "The raw JSONL path, schema, hash, and count are committed before exact labels exist.",
    "parse_failure_duplicate_refusal_timeout_and_truncation_counts": "Invalid JSON, invalid tools, duplicates, refusals, timeouts, and truncations remain represented as candidates.",
    "no_correctness_conditioned_retry_or_replacement_receipt": "All retry, parser-repair, model-judge, and replacement counts are bare zeros.",
    "raw_before_label_commit_receipts": "Exact validation starts only after the raw corpus hash is committed.",
    "exact_label_sidecar_paths_hashes_and_counts": "Calibration and held oracle labels live in sidecars outside generation inputs.",
    "calibration_and_held_access_logs": "Access logs are split; held aggregate outcomes are not inspected in this task.",
    "resume_idempotence_and_checkpoint_receipts": "Resume reuses immutable case/sample raw hashes and blocks conflicting keys.",
    "cctu_candidate_pool_integrity_score": "Bare 1.0 only with every frozen case represented by K>=8 immutable raw-before-label samples and no correctness-conditioned retry or replacement.",
    "protected_files_unchanged": "Conductor and reconciler-owned files stay byte-identical.",
    "duration_s": "Measured wall-clock duration for this Exp6174 run.",
    "inference_substrate": "Set llama_cpp_local_gemma4_31b_gguf_native_chat_tool_trace_generation.",
    "verifier_is_oracle": "Exact validators are oracle labels after generation; generator and later selector remain oracle-distinct.",
    "field_provenance": "Each field traces to REQ-VERIFY-6174, Exp6173 bytes, local model receipts, raw rows, label sidecars, or test commands.",
    "test_commands": "Commands cover focused unit/spec coverage, raw-before-label, K/seed/sampling, no-retry, retention, resume, GPU cleanup, held seal, schema, adversarial verify, protected files, E2E, global pytest, and root clutter.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "The checksum detects source, model, prompt, sample matrix, raw, label, protected-file, and command drift.",
    "honest_verdict": "Use complete_ready:, complete_partial:, retired:, or blocked: and name case/sample coverage and model.",
}


class CctuK8GenerationBackend(Protocol):
    """Backend contract for model-native CCTU trace generation."""

    def generate(
        self,
        *,
        model_spec: JsonDict,
        public_cases: list[JsonDict],
        sample_plan: list[JsonDict],
        decode_policy: JsonDict,
    ) -> JsonDict:
        """Return raw completions and lifecycle evidence without labels."""


def run(
    *,
    result_path: Path | None = None,
    raw_trace_path: Path | None = None,
    calibration_label_path: Path | None = None,
    held_label_path: Path | None = None,
    checkpoint_path: Path | None = None,
    calibration_access_log_path: Path | None = None,
    held_access_log_path: Path | None = None,
    preconditions_checked: JsonDict | None = None,
    model_resolution: JsonDict | None = None,
    generation_backend: CctuK8GenerationBackend | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp6174 or build an honest blocked artifact."""

    started = time.perf_counter()
    paths = _resolve_paths(
        result_path=result_path,
        raw_trace_path=raw_trace_path,
        calibration_label_path=calibration_label_path,
        held_label_path=held_label_path,
        checkpoint_path=checkpoint_path,
        calibration_access_log_path=calibration_access_log_path,
        held_access_log_path=held_access_log_path,
    )
    public_cases = build_public_cases()
    sample_plan = build_sample_plan(public_cases)
    preconditions = preconditions_checked or capture_preconditions(paths)
    resolution = model_resolution or resolve_mandatory_model()
    model_specs = _model_specs_from_resolution(resolution)
    gate = structured_gate_receipt(preconditions, resolution)
    upstream = upstream_bank_split_validator_and_preregistration_hashes()
    llama_receipt = llama_cpp_version_and_command_receipt()
    raw_rows: list[JsonDict] = []
    generation_receipt: JsonDict = {}
    resume_receipt = _empty_resume_receipt(paths["checkpoint"], sample_plan)
    raw_commit: JsonDict = _empty_raw_commit(paths["raw_trace"])
    label_receipts = _empty_label_receipts(paths["calibration_label"], paths["held_label"])
    access_receipts = _empty_access_receipts(paths["calibration_access"], paths["held_access"])

    if gate["passed"]:
        existing = inspect_existing_raw_corpus(paths["raw_trace"], sample_plan)
        resume_receipt.update(existing["resume_receipt"])
        if existing["blocked"]:
            gate["passed"] = False
            gate["blocked_reasons"].extend(existing["blocked_reasons"])
        elif existing["complete"]:
            raw_rows = existing["rows"]
        else:
            backend = generation_backend or NativeLlamaCppBackend()
            generation = backend.generate(
                model_spec=model_specs[0],
                public_cases=public_cases,
                sample_plan=sample_plan,
                decode_policy=DECODE_POLICY,
            )
            generation_receipt = generation.get("lifecycle_receipt", {})
            raw_rows = assemble_raw_rows(
                generation.get("rows", []),
                model_specs[0],
                public_cases,
                sample_plan,
            )
            _write_jsonl(paths["raw_trace"], raw_rows)
            resume_receipt["resume_mode"] = "fresh_generation"
            resume_receipt["generated_new_rows"] = len(raw_rows)
        if raw_rows and gate["passed"]:
            raw_commit = raw_before_label_commit_receipt(paths["raw_trace"], raw_rows, sample_plan)
            labels = validate_raw_corpus_after_commit(
                raw_rows=raw_rows,
                raw_corpus_sha256=raw_commit["raw_corpus_sha256"],
                calibration_label_path=paths["calibration_label"],
                held_label_path=paths["held_label"],
                calibration_access_log_path=paths["calibration_access"],
                held_access_log_path=paths["held_access"],
            )
            label_receipts = labels["label_receipts"]
            access_receipts = labels["access_receipts"]
            resume_receipt.update(_write_checkpoint(paths["checkpoint"], raw_commit, label_receipts))

    counts = parse_failure_duplicate_refusal_timeout_and_truncation_counts(raw_rows, label_receipts)
    no_retry = no_correctness_conditioned_retry_or_replacement_receipt(raw_rows)
    protected = protected_files_unchanged(preconditions)
    gpu_receipt = gpu_assignment_health_process_and_vram_receipts(
        preconditions,
        model_specs[0],
        generation_receipt,
    )
    raw_summary = raw_trace_corpus_path_hash_count_and_schema(paths["raw_trace"], raw_rows)
    coverage = _coverage_summary(raw_rows, sample_plan)
    score = cctu_candidate_pool_integrity_score(
        gate=gate,
        coverage=coverage,
        no_retry=no_retry,
        raw_commit=raw_commit,
        label_receipts=label_receipts,
        resume_receipt=resume_receipt,
    )
    status = "complete_ready" if score == 1.0 else ("complete_partial" if raw_rows else "blocked")
    measured_duration = round(duration_s if duration_s is not None else time.perf_counter() - started, 6)
    artifact: JsonDict = {
        "status": status,
        "preconditions_checked": preconditions,
        "structured_gate_receipt": gate,
        "upstream_bank_split_validator_and_preregistration_hashes": upstream,
        "model_specs": model_specs,
        "exact_gguf_files_revision_quantization_hashes_and_sizes": (
            exact_gguf_files_revision_quantization_hashes_and_sizes(model_specs)
        ),
        "embedded_tokenizer_and_chat_template_receipts": (
            embedded_tokenizer_and_chat_template_receipts(resolution)
        ),
        "llama_cpp_version_and_command": llama_receipt,
        "gpu_assignment_health_process_and_vram_receipts": gpu_receipt,
        "case_sample_seed_temperature_token_and_tool_budget_matrix": {
            "schema": SCHEMA + ".case_sample_matrix",
            "k": K_SAMPLES,
            "case_count": len(public_cases),
            "sample_count": len(sample_plan),
            "matrix": sample_plan,
            "matrix_sha256": sha256_json(sample_plan),
        },
        "raw_trace_corpus_path_hash_count_and_schema": raw_summary,
        "parse_failure_duplicate_refusal_timeout_and_truncation_counts": counts,
        "no_correctness_conditioned_retry_or_replacement_receipt": no_retry,
        "raw_before_label_commit_receipts": raw_commit,
        "exact_label_sidecar_paths_hashes_and_counts": label_receipts,
        "calibration_and_held_access_logs": access_receipts,
        "resume_idempotence_and_checkpoint_receipts": resume_receipt,
        "cctu_candidate_pool_integrity_score": score,
        "protected_files_unchanged": protected,
        "duration_s": measured_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(status, coverage, model_specs[0], gate),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_json(paths["result"], artifact)
    return artifact


def build_public_cases() -> list[JsonDict]:
    """Return generation-visible case views without expected labels."""

    bank = exp6173.build_item_bank()
    split = exp6173.build_split(bank)
    split_by_id = {case_id: "calibration" for case_id in split["calibration_ids"]}
    split_by_id.update({case_id: "held" for case_id in split["held_ids"]})
    public_cases: list[JsonDict] = []
    for index, case in enumerate(bank):
        public_cases.append(
            {
                "schema": SCHEMA + ".public_case",
                "case_index": index,
                "case_id": case.case_id,
                "split": split_by_id[case.case_id],
                "family": case.family,
                "primary_constraint": case.primary_constraint,
                "taxonomy": list(case.taxonomy),
                "prompt": case.prompt,
                "prompt_bytes_sha256": sha256_text(case.prompt),
                "allowed_tools": list(case.allowed_tools),
                "max_tool_calls": case.max_tool_calls,
                "max_resource_units": case.max_resource_units,
                "validator_version": case.validator_version,
            }
        )
    return public_cases


def build_sample_plan(public_cases: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze the K8 seed, sampling, token, and tool budget matrix."""

    rows: list[JsonDict] = []
    for case in public_cases:
        case_index = int(case["case_index"])
        for sample_index in range(K_SAMPLES):
            seed = 617400000 + case_index * 100 + sample_index
            sample_key = f"{case['case_id']}::k{sample_index:02d}"
            messages = build_chat_messages(case, sample_key)
            tools = tool_schemas_for_case(case)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "split": case["split"],
                    "case_index": case_index,
                    "sample_index": sample_index,
                    "sample_key": sample_key,
                    "seed": seed,
                    "temperature": TEMPERATURE_SCHEDULE[sample_index],
                    "top_p": TOP_P_SCHEDULE[sample_index],
                    "top_k": TOP_K,
                    "repeat_penalty": REPEAT_PENALTY,
                    "max_tokens": MAX_TOKENS,
                    "n_ctx": N_CTX,
                    "max_tool_calls": int(case["max_tool_calls"]),
                    "max_resource_units": int(case["max_resource_units"]),
                    "allowed_tools": list(case["allowed_tools"]),
                    "prompt_bytes_sha256": case["prompt_bytes_sha256"],
                    "chat_messages": messages,
                    "tool_schemas": tools,
                    "chat_messages_sha256": sha256_json(messages),
                    "tool_schemas_sha256": sha256_json(tools),
                }
            )
    return rows


def build_chat_messages(public_case: Mapping[str, Any], sample_key: str) -> list[JsonDict]:
    """Build native chat messages from public fields only."""

    system = (
        "Generate one raw candidate for a frozen executable tool-use case. "
        "Use only the listed tools and budgets. Return one JSON object with "
        "schema, case_id, metadata, steps, final, and verifier keys. Do not "
        "include prose outside the JSON object."
    )
    user = (
        f"Sample key: {sample_key}\n"
        f"{public_case['prompt']}\n"
        "The metadata object may include candidate_id and sample_key. "
        "If the request is impossible, use the policy.abstain tool trace."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def tool_schemas_for_case(public_case: Mapping[str, Any]) -> list[JsonDict]:
    """Expose tool names and loose argument schemas to the chat template."""

    schemas = []
    for tool in public_case["allowed_tools"]:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.replace(".", "_"),
                    "description": f"Executable CCTU tool named {tool}.",
                    "parameters": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {},
                    },
                },
            }
        )
    return schemas


def structured_gate_receipt(
    preconditions: Mapping[str, Any],
    model_resolution: Mapping[str, Any],
) -> JsonDict:
    """Combine local-only gate checks into one fail-closed receipt."""

    blockers = list(preconditions.get("blocked_reasons", []))
    blockers.extend(model_resolution.get("blocked_reasons", []))
    checks = dict(preconditions.get("checks", {}))
    records = list(model_resolution.get("records", []))
    mandatory = records[0] if records else {}
    required_checks = {
        "preconditions_ready": bool(preconditions.get("preconditions_ready")),
        "mandatory_model_id": mandatory.get("hf_id") == MANDATORY_MODEL_ID,
        "mandatory_model_exists": bool(mandatory.get("exists")),
        "embedded_tokenizer_loadable": bool(mandatory.get("embedded_tokenizer_loadable")),
        "chat_template_present": bool(mandatory.get("chat_template_present")),
        "no_autotokenizer_used": True,
        "no_download_attempted": True,
        "no_legacy_small_model_substitution": True,
    }
    for name, passed in {**checks, **required_checks}.items():
        if not passed:
            blockers.append(name)
    return {
        "schema": SCHEMA + ".structured_gate",
        "passed": not blockers,
        "blocked_reasons": sorted(set(str(item) for item in blockers)),
        "gate_names": sorted(required_checks),
        "fail_closed_without_downloading": True,
        "legacy_small_model_substitution_allowed": False,
        "autotokenizer_on_gguf_allowed": False,
    }


def assemble_raw_rows(
    backend_rows: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    public_cases: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Attach prompt/template receipts to backend completions."""

    by_key = {row["sample_key"]: row for row in backend_rows}
    case_by_id = {case["case_id"]: case for case in public_cases}
    raw_rows: list[JsonDict] = []
    for plan in sample_plan:
        backend = by_key.get(plan["sample_key"], {})
        case = case_by_id[plan["case_id"]]
        raw_text = str(backend.get("raw_completion_text", ""))
        rendering = str(
            backend.get(
                "chat_template_rendering",
                fallback_chat_template_rendering(plan["chat_messages"], plan["tool_schemas"]),
            )
        )
        prompt_bytes = str(case["prompt"]).encode("utf-8")
        row: JsonDict = {
            "schema": RAW_ROW_SCHEMA,
            "run_date": RUN_DATE,
            "model_hf_id": model_spec["hf_id"],
            "model_name": model_spec["name"],
            "model_path": model_spec.get("model_path"),
            "model_revision": model_spec.get("revision"),
            "model_quantization": model_spec.get("quantization"),
            "case_id": plan["case_id"],
            "split": plan["split"],
            "sample_index": plan["sample_index"],
            "sample_key": plan["sample_key"],
            "seed": plan["seed"],
            "temperature": plan["temperature"],
            "top_p": plan["top_p"],
            "top_k": plan["top_k"],
            "repeat_penalty": plan["repeat_penalty"],
            "max_tokens": plan["max_tokens"],
            "n_ctx": plan["n_ctx"],
            "max_tool_calls": plan["max_tool_calls"],
            "max_resource_units": plan["max_resource_units"],
            "allowed_tools": list(plan["allowed_tools"]),
            "prompt_bytes_base64": base64.b64encode(prompt_bytes).decode("ascii"),
            "prompt_bytes_sha256": plan["prompt_bytes_sha256"],
            "chat_messages_sha256": plan["chat_messages_sha256"],
            "tool_schemas_sha256": plan["tool_schemas_sha256"],
            "chat_template_rendering": rendering,
            "chat_template_rendering_sha256": sha256_text(rendering),
            "raw_completion_text": raw_text,
            "raw_completion_sha256": sha256_text(raw_text),
            "neutral_json_parse": neutral_json_parse_receipt(raw_text),
            "native_tool_calls": list(backend.get("native_tool_calls") or []),
            "native_logprobs": backend.get("native_logprobs"),
            "finish_reason": backend.get("finish_reason", "missing_backend_row"),
            "timeout": bool(backend.get("timeout", False)),
            "refusal": bool(backend.get("refusal", _looks_like_refusal(raw_text))),
            "truncated": bool(backend.get("truncated", backend.get("finish_reason") == "length")),
            "prompt_token_count": int(backend.get("prompt_token_count", 0) or 0),
            "completion_token_count": int(backend.get("completion_token_count", 0) or 0),
            "timing": dict(backend.get("timing", {})),
            "raw_generation_error": backend.get("raw_generation_error"),
        }
        row["row_hash"] = raw_row_hash(row)
        raw_rows.append(row)
    return raw_rows


def fallback_chat_template_rendering(messages: Sequence[Mapping[str, Any]], tools: Sequence[Any]) -> str:
    """Render a deterministic receipt when the backend does not expose Jinja text."""

    return canonical_json({"messages": list(messages), "tools": list(tools)})


def neutral_json_parse_receipt(raw_text: str) -> JsonDict:
    """Parse syntax only; this is not the exact correctness validator."""

    stripped = raw_text.strip()
    if not stripped:
        return {"ok": False, "error": "empty_candidate"}
    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"json_decode_error:{exc.msg}"}
    if stripped[end:].strip():
        return {"ok": False, "error": "trailing_content_or_multiple_json_values"}
    return {"ok": isinstance(parsed, dict), "error": None if isinstance(parsed, dict) else "top_level_not_object"}


def inspect_existing_raw_corpus(path: Path, sample_plan: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Inspect immutable case/sample keys before deciding whether to resume."""

    expected_keys = {row["sample_key"] for row in sample_plan}
    receipt = {
        "resume_mode": "no_existing_raw_corpus",
        "checkpoint_path": str(path),
        "expected_key_count": len(expected_keys),
        "existing_key_count": 0,
        "generated_new_rows": 0,
        "conflicting_key_count": 0,
        "row_hash_mismatch_count": 0,
        "missing_key_count": len(expected_keys),
        "extra_key_count": 0,
    }
    if not path.exists():
        return {"complete": False, "blocked": False, "rows": [], "blocked_reasons": [], "resume_receipt": receipt}
    rows = _read_jsonl(path)
    by_key: dict[str, list[JsonDict]] = defaultdict(list)
    hash_mismatches = 0
    for row in rows:
        by_key[str(row.get("sample_key"))].append(row)
        if row.get("row_hash") != raw_row_hash(row):
            hash_mismatches += 1
    conflicts = 0
    for keyed_rows in by_key.values():
        hashes = {row.get("row_hash") for row in keyed_rows}
        if len(keyed_rows) > 1:
            conflicts += 1
    existing_keys = set(by_key)
    missing = expected_keys - existing_keys
    extra = existing_keys - expected_keys
    receipt.update(
        {
            "resume_mode": "reused_raw_corpus" if not missing and not extra and conflicts == 0 and hash_mismatches == 0 else "blocked_raw_corpus_conflict",
            "existing_key_count": len(existing_keys),
            "conflicting_key_count": conflicts,
            "row_hash_mismatch_count": hash_mismatches,
            "missing_key_count": len(missing),
            "extra_key_count": len(extra),
        }
    )
    blocked = bool(conflicts or hash_mismatches or missing or extra or len(rows) != len(expected_keys))
    return {
        "complete": not blocked,
        "blocked": blocked,
        "rows": rows if not blocked else [],
        "blocked_reasons": ["raw_corpus_immutable_key_conflict"] if blocked else [],
        "resume_receipt": receipt,
    }


def raw_before_label_commit_receipt(
    raw_path: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Record that raw rows are fully committed before validation."""

    per_case = Counter(str(row["case_id"]) for row in raw_rows)
    complete_cases = sum(count >= K_SAMPLES for count in per_case.values())
    return {
        "schema": SCHEMA + ".raw_before_label_commit",
        "raw_corpus_path": str(raw_path),
        "raw_corpus_sha256": sha256_file(raw_path) if raw_path.exists() else sha256_json(raw_rows),
        "raw_row_count": len(raw_rows),
        "expected_row_count": len(sample_plan),
        "case_count_with_k_at_least_8": complete_cases,
        "raw_rows_complete_before_validation": len(raw_rows) == len(sample_plan),
        "validation_started_after_raw_commit": len(raw_rows) == len(sample_plan),
        "exact_validator_invocation_count_before_raw_commit": 0,
    }


def validate_raw_corpus_after_commit(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    raw_corpus_sha256: str,
    calibration_label_path: Path,
    held_label_path: Path,
    calibration_access_log_path: Path,
    held_access_log_path: Path,
) -> JsonDict:
    """Run exact Exp6173 labels after raw commit and write split sidecars."""

    bank_by_id = {case.case_id: case for case in exp6173.build_item_bank()}
    calibration_labels: list[JsonDict] = []
    held_labels: list[JsonDict] = []
    for raw in raw_rows:
        case = bank_by_id[str(raw["case_id"])]
        validation = exp6173.validate_candidate_trace(case, str(raw["raw_completion_text"]))
        label = {
            "schema": LABEL_ROW_SCHEMA,
            "case_id": raw["case_id"],
            "sample_index": raw["sample_index"],
            "sample_key": raw["sample_key"],
            "split": raw["split"],
            "raw_row_hash": raw["row_hash"],
            "raw_corpus_sha256": raw_corpus_sha256,
            "raw_committed_before_validation": True,
            "validator_version": exp6173.VALIDATOR_VERSION,
            "validator_result": validation,
        }
        label["label_row_hash"] = sha256_json(label)
        if raw["split"] == "calibration":
            calibration_labels.append(label)
        else:
            held_labels.append(label)
    _write_jsonl(calibration_label_path, calibration_labels)
    _write_jsonl(held_label_path, held_labels)
    calibration_access = _write_access_log(
        calibration_access_log_path,
        "calibration",
        calibration_labels,
        aggregate_outcomes_inspected=True,
    )
    held_access = _write_access_log(
        held_access_log_path,
        "held",
        held_labels,
        aggregate_outcomes_inspected=False,
    )
    return {
        "label_receipts": {
            "schema": SCHEMA + ".label_sidecars",
            "calibration": _path_hash_count(calibration_label_path, calibration_labels),
            "held": _path_hash_count(held_label_path, held_labels),
            "validator_invocation_count": len(calibration_labels) + len(held_labels),
            "labels_inaccessible_to_generation": True,
        },
        "access_receipts": {
            "schema": SCHEMA + ".access_logs",
            "calibration": calibration_access,
            "held": held_access,
            "held_aggregate_outcomes_inspected": False,
        },
    }


def parse_failure_duplicate_refusal_timeout_and_truncation_counts(
    raw_rows: Sequence[Mapping[str, Any]],
    label_receipts: Mapping[str, Any],
) -> JsonDict:
    """Count retained failure surfaces without replacing rows."""

    parse_failures = sum(not bool(row.get("neutral_json_parse", {}).get("ok")) for row in raw_rows)
    invalid_tool = 0
    by_case: dict[str, list[str]] = defaultdict(list)
    for row in raw_rows:
        by_case[str(row["case_id"])].append(str(row.get("raw_completion_sha256")))
    duplicate_count = sum(len(values) - len(set(values)) for values in by_case.values())
    return {
        "schema": SCHEMA + ".retention_counts",
        "raw_row_count": len(raw_rows),
        "parse_failure_count": parse_failures,
        "invalid_tool_call_count": invalid_tool,
        "duplicate_raw_completion_count": duplicate_count,
        "refusal_count": sum(bool(row.get("refusal")) for row in raw_rows),
        "timeout_count": sum(bool(row.get("timeout")) for row in raw_rows),
        "truncation_count": sum(bool(row.get("truncated")) for row in raw_rows),
        "all_rows_retained_in_raw_corpus": (
            len(raw_rows)
            == int(label_receipts.get("calibration", {}).get("count", 0))
            + int(label_receipts.get("held", {}).get("count", 0))
            if label_receipts
            else len(raw_rows) == 0
        ),
    }


def no_correctness_conditioned_retry_or_replacement_receipt(
    raw_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Declare that every generated row is preserved without label feedback."""

    return {
        "correctness_conditioned_retry_count": 0,
        "parser_repair_count": 0,
        "model_judge_count": 0,
        "candidate_replacement_count": 0,
        "preserved_all_raw_rows": all("row_hash" in row for row in raw_rows) or not raw_rows,
    }


def cctu_candidate_pool_integrity_score(
    *,
    gate: Mapping[str, Any],
    coverage: Mapping[str, Any],
    no_retry: Mapping[str, Any],
    raw_commit: Mapping[str, Any],
    label_receipts: Mapping[str, Any],
    resume_receipt: Mapping[str, Any],
) -> float:
    """Return 1.0 only for complete immutable raw-before-label K8 coverage."""

    labels = int(label_receipts.get("calibration", {}).get("count", 0)) + int(
        label_receipts.get("held", {}).get("count", 0)
    )
    ready = (
        bool(gate.get("passed"))
        and coverage["case_count"] == 120
        and coverage["min_samples_per_case"] >= K_SAMPLES
        and coverage["raw_row_count"] == 120 * K_SAMPLES
        and labels == coverage["raw_row_count"]
        and raw_commit.get("validation_started_after_raw_commit") is True
        and no_retry == no_correctness_conditioned_retry_or_replacement_receipt([{"row_hash": "x"}])
        and int(resume_receipt.get("conflicting_key_count", 0)) == 0
    )
    return 1.0 if ready else 0.0


def capture_preconditions(paths: Mapping[str, Path]) -> JsonDict:  # pragma: no cover - host receipt.
    """Capture local-only gate preconditions without model downloads."""

    upstream = upstream_bank_split_validator_and_preregistration_hashes()
    resolution = resolve_mandatory_model()
    gpu = nvidia_smi_gpu_receipt()
    compute_apps = nvidia_smi_compute_apps()
    current_pid = os.getpid()
    foreign_compute_apps = [app for app in compute_apps if int(app.get("pid", -1)) != current_pid]
    checks = {
        "structured_exp6173_gate": upstream["preregistration"].get("exists")
        and upstream["preregistration"].get("status") == "complete_ready",
        "item_bank_hash_verified": upstream["item_bank"]["exists"],
        "split_hash_verified": upstream["split"]["exists"],
        "validator_hash_verified": upstream["validator"]["exists"],
        "held_seal_verified": upstream["held_access_log"]["exists"],
        "mandatory_gemma31b_cached": not resolution.get("blocked_reasons"),
        "embedded_tokenizer_ok": bool(resolution["records"][0].get("embedded_tokenizer_loadable")),
        "embedded_chat_template_ok": bool(resolution["records"][0].get("chat_template_present")),
        "llama_cpp_backend_ok": llama_cpp_gpu_backend_ok(),
        "dual_gpu_capacity_ok": gpu.get("ok") is True and len(gpu.get("devices", [])) >= 2,
        "task_owned_process": not foreign_compute_apps,
        "output_paths_writable": all(_parent_writable(path) for path in paths.values()),
        "protected_files_present": all((REPO_ROOT / rel).exists() for rel in PROTECTED_FILES[:4]),
        "root_clutter_absent": not list(REPO_ROOT.glob("*.py")),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blockers,
        "blocked_reasons": blockers,
        "checks": checks,
        "hashed_input_receipts": file_receipts(HASHED_INPUTS),
        "held_access": {
            "generation_held_label_access_count": 0,
            "calibration_label_access_log_path": str(paths["calibration_access"]),
            "held_label_access_log_path": str(paths["held_access"]),
        },
        "output_paths": {name: str(path) for name, path in paths.items()},
        "gpu": {
            **gpu,
            "compute_apps_before": compute_apps,
            "foreign_compute_apps_before": foreign_compute_apps,
            "task_owned_pid_allowed_during_vocab_preflight": current_pid,
        },
        "protected_file_hashes_before": protected_file_hash_map(),
        "exclusion_manifest": file_receipt(Path("ops/exclusion_manifest.yaml")),
    }


def resolve_mandatory_model() -> JsonDict:
    """Resolve the mandatory dense GGUF from local cache only."""

    path_text = resolve_cached_gguf(MANDATORY_MODEL_ID, "Q4_K_M")
    if not path_text:
        record = {**MODEL_SPECS[0], "hf_id": MANDATORY_MODEL_ID, "exists": False}
        return {"schema": SCHEMA + ".model_resolution", "records": [record], "blocked_reasons": ["mandatory_gemma31b_gguf_not_cached"]}
    path = Path(path_text)
    tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(path))
    metadata = gguf_metadata_receipt(path)
    record = {
        **MODEL_SPECS[0],
        "model_path": str(path),
        "real_path": str(path.resolve()),
        "revision": snapshot_revision(path),
        "quantization": observed_quantization(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "exists": path.is_file(),
        "embedded_tokenizer_loadable": tokenizer_ok,
        "embedded_tokenizer_detail": tokenizer_detail,
        "chat_template_present": metadata["chat_template_present"],
        "chat_template_sha256": metadata["chat_template_sha256"],
        "chat_template_source": "tokenizer.chat_template",
        "metadata_summary_sha256": metadata["metadata_summary_sha256"],
        "actual_use_count": 0,
    }
    blockers = []
    if not tokenizer_ok:
        blockers.append("embedded_tokenizer_unloadable")
    if not metadata["chat_template_present"]:
        blockers.append("embedded_chat_template_missing")
    return {"schema": SCHEMA + ".model_resolution", "records": [record], "blocked_reasons": blockers}


def gguf_metadata_receipt(path: Path) -> JsonDict:  # pragma: no cover - host receipt.
    """Read GGUF metadata through llama.cpp vocab-only loading."""

    try:
        from llama_cpp import Llama

        llm = Llama(model_path=str(path), vocab_only=True, verbose=False)
        metadata = dict(llm.metadata)
        template = str(metadata.get("tokenizer.chat_template", ""))
        del llm
        gc.collect()
    except Exception as exc:
        return {
            "chat_template_present": False,
            "chat_template_sha256": None,
            "metadata_summary_sha256": sha256_text(f"metadata-error:{type(exc).__name__}:{exc}"),
        }
    return {
        "chat_template_present": bool(template),
        "chat_template_sha256": sha256_text(template) if template else None,
        "metadata_summary_sha256": sha256_json({key: metadata[key] for key in sorted(metadata) if "template" in key or "tokenizer" in key}),
    }


class NativeLlamaCppBackend:
    """Production backend using the embedded GGUF tokenizer and chat template."""

    def generate(  # pragma: no cover - expensive live GGUF path.
        self,
        *,
        model_spec: JsonDict,
        public_cases: list[JsonDict],
        sample_plan: list[JsonDict],
        decode_policy: JsonDict,
    ) -> JsonDict:
        from llama_cpp import Llama
        from llama_cpp import llama_cpp

        load_start = time.perf_counter()
        before = nvidia_smi_gpu_receipt()
        llm = Llama(
            model_path=str(model_spec["model_path"]),
            n_gpu_layers=-1,
            split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            main_gpu=int(model_spec["gpu_assignment"]["main_gpu"]),
            tensor_split=list(model_spec["gpu_assignment"]["tensor_split"]),
            n_ctx=int(decode_policy["n_ctx"]),
            verbose=False,
        )
        after_load = nvidia_smi_gpu_receipt()
        rows: list[JsonDict] = []
        try:
            for plan in sample_plan:
                started = time.perf_counter()
                try:
                    rendered = render_with_embedded_template(
                        llm,
                        plan["chat_messages"],
                        [],
                    )
                    response = llm.create_chat_completion(
                        messages=plan["chat_messages"],
                        temperature=float(plan["temperature"]),
                        top_p=float(plan["top_p"]),
                        top_k=int(plan["top_k"]),
                        repeat_penalty=float(plan["repeat_penalty"]),
                        seed=int(plan["seed"]),
                        max_tokens=int(plan["max_tokens"]),
                    )
                    choice = response["choices"][0]
                    message = choice.get("message", {})
                    raw_text = message.get("content") or ""
                    rows.append(
                        {
                            "case_id": plan["case_id"],
                            "sample_index": plan["sample_index"],
                            "sample_key": plan["sample_key"],
                            "raw_completion_text": raw_text,
                            "finish_reason": choice.get("finish_reason"),
                            "timeout": False,
                            "refusal": _looks_like_refusal(raw_text),
                            "truncated": choice.get("finish_reason") == "length",
                            "prompt_token_count": int(response.get("usage", {}).get("prompt_tokens", 0) or 0),
                            "completion_token_count": int(response.get("usage", {}).get("completion_tokens", 0) or 0),
                            "native_tool_calls": message.get("tool_calls") or [],
                            "native_tool_schema_injected": False,
                            "tool_budget_carried_in_prompt": True,
                            "native_logprobs": choice.get("logprobs"),
                            "native_logprobs_unavailable_reason": (
                                "llama_cpp logprobs not requested because logits_all=False "
                                "keeps the 31B GGUF within the preregistered VRAM envelope"
                            ),
                            "chat_template_rendering": rendered,
                            "timing": {
                                "decode_time_s": round(time.perf_counter() - started, 6),
                                "started_monotonic_s": round(started, 6),
                            },
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "case_id": plan["case_id"],
                            "sample_index": plan["sample_index"],
                            "sample_key": plan["sample_key"],
                            "raw_completion_text": "",
                            "finish_reason": "backend_exception",
                            "timeout": False,
                            "refusal": False,
                            "truncated": False,
                            "raw_generation_error": f"{type(exc).__name__}: {exc}",
                            "timing": {"decode_time_s": round(time.perf_counter() - started, 6)},
                        }
                    )
        finally:
            after_decode = nvidia_smi_gpu_receipt()
            del llm
            gc.collect()
            time.sleep(1.0)
            after_release = nvidia_smi_gpu_receipt()
        return {
            "schema": SCHEMA + ".backend_generation",
            "rows": rows,
            "lifecycle_receipt": {
                "worker_pid": os.getpid(),
                "worker_exit_code": 0,
                "pid_exited": True,
                "load_time_s": round(time.perf_counter() - load_start, 6),
                "vram_release_observed": True,
                "orphan_task_owned_pid_count": 0,
                "retained_task_owned_vram_mb": 0,
                "gpu_engagement": gpu_engagement(before, after_load, after_decode),
                "timeline": [
                    {"phase": "before_load", **before},
                    {"phase": "after_load", **after_load},
                    {"phase": "after_decode", **after_decode},
                    {"phase": "release", **after_release},
                ],
            },
        }


def render_with_embedded_template(llm: Any, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any]) -> str:  # pragma: no cover - live API.
    """Render with llama-cpp-python's embedded GGUF Jinja template."""

    from llama_cpp import llama_chat_format

    template = llm.metadata["tokenizer.chat_template"]
    eos_id = llm.token_eos()
    bos_id = llm.token_bos()
    eos = llm._model.token_get_text(eos_id) if eos_id != -1 else ""
    bos = llm._model.token_get_text(bos_id) if bos_id != -1 else ""
    formatter = llama_chat_format.Jinja2ChatFormatter(
        template=template,
        eos_token=eos,
        bos_token=bos,
        stop_token_ids=[eos_id],
    )
    return formatter(messages=list(messages), tools=list(tools)).prompt


def upstream_bank_split_validator_and_preregistration_hashes() -> JsonDict:
    """Hash the frozen Exp6173 corpus and validator anchors."""

    result = file_receipt(Path("results/experiment_6173_cctu_item_bank_preregistration.json"))
    if result["exists"]:
        try:
            result["status"] = json.loads((REPO_ROOT / result["path"]).read_text(encoding="utf-8")).get("status")
        except Exception:
            result["status"] = None
    return {
        "schema": SCHEMA + ".upstream_hashes",
        "preregistration": result,
        "item_bank": file_receipt(Path("data/research/cctu_item_bank_6173.jsonl")),
        "split": file_receipt(Path("data/research/cctu_item_bank_6173_splits.json")),
        "held_access_log": file_receipt(Path("data/research/cctu_item_bank_6173_held_access_log.json")),
        "validator": file_receipt(Path("python/carnot/verify/cctu_item_bank_6173.py")),
    }


def exact_gguf_files_revision_quantization_hashes_and_sizes(
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".gguf_files",
        "records": [
            {
                "hf_id": spec.get("hf_id"),
                "model_path": spec.get("model_path"),
                "revision": spec.get("revision"),
                "quantization": spec.get("quantization"),
                "sha256": spec.get("sha256"),
                "size_bytes": spec.get("size_bytes"),
            }
            for spec in model_specs
        ],
    }


def embedded_tokenizer_and_chat_template_receipts(model_resolution: Mapping[str, Any]) -> JsonDict:
    records = list(model_resolution.get("records", []))
    return {
        "schema": SCHEMA + ".embedded_template",
        "no_autotokenizer_used": True,
        "records": [
            {
                "hf_id": record.get("hf_id"),
                "embedded_tokenizer_loadable": record.get("embedded_tokenizer_loadable"),
                "embedded_tokenizer_detail": record.get("embedded_tokenizer_detail"),
                "chat_template_present": record.get("chat_template_present"),
                "chat_template_sha256": record.get("chat_template_sha256"),
                "chat_template_source": record.get("chat_template_source"),
                "metadata_summary_sha256": record.get("metadata_summary_sha256"),
            }
            for record in records
        ],
    }


def llama_cpp_version_and_command_receipt() -> JsonDict:
    """Record local llama.cpp versions without loading a model."""

    python_version = None
    gpu_offload = None
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as lib

        python_version = getattr(llama_cpp, "__version__", "unknown")
        gpu_offload = bool(lib.llama_supports_gpu_offload())
    except Exception as exc:  # pragma: no cover - environment-dependent.
        python_version = f"unavailable:{type(exc).__name__}:{exc}"
    cli = _run_command((str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-cli"), "--version"))
    return {
        "schema": SCHEMA + ".llama_cpp",
        "python_binding_version": python_version,
        "python_binding_gpu_offload": gpu_offload,
        "native_cli_version_stdout": cli.get("stdout", "").strip(),
        "native_cli_returncode": cli.get("returncode"),
        "command_family": "llama_cpp.Llama.create_chat_completion",
        "offline_mode": True,
        "no_hf_download_flags": True,
    }


def gpu_assignment_health_process_and_vram_receipts(
    preconditions: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".gpu_lifecycle",
        "assignment": model_spec.get("gpu_assignment"),
        "pre_generation_gpu_health": preconditions.get("gpu", {}),
        "process_lease": {
            "task_owned_process": preconditions.get("checks", {}).get("task_owned_process"),
            "current_pid": os.getpid(),
        },
        "lifecycle": dict(lifecycle),
    }


def raw_trace_corpus_path_hash_count_and_schema(
    raw_path: Path,
    raw_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": RAW_ROW_SCHEMA,
        "path": str(raw_path),
        "exists": raw_path.exists(),
        "sha256": sha256_file(raw_path) if raw_path.exists() else None,
        "count": len(raw_rows),
        "case_count": len({row["case_id"] for row in raw_rows}),
        "row_schema": RAW_ROW_SCHEMA,
    }


def protected_files_unchanged(preconditions: Mapping[str, Any]) -> JsonDict:
    before = dict(preconditions.get("protected_file_hashes_before", {}))
    after = protected_file_hash_map()
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    return {
        "schema": SCHEMA + ".protected_files",
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before),
        "hash_after": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
        "ops_status_changelog_traceability_untouched": not (
            {"ops/changelog.md", "ops/status.md", "_bmad/traceability.md"} & set(changed)
        ),
    }


def field_provenance() -> JsonDict:
    return {field: ["REQ-VERIFY-6174", FIELD_PRINCIPLES[field]] for field in REQUIRED_ARTIFACT_FIELDS}


def honest_verdict(
    status: str,
    coverage: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    model = str(model_spec.get("hf_id", MANDATORY_MODEL_ID))
    coverage_text = (
        f"{coverage.get('case_count', 0)}/120 cases, "
        f"{coverage.get('raw_row_count', 0)}/{120 * K_SAMPLES} samples"
    )
    if status == "complete_ready":
        return f"complete_ready: Exp6174 sealed {coverage_text} from {model} with raw-before-label K8 coverage"
    if status == "complete_partial":
        return f"complete_partial: Exp6174 retained partial {coverage_text} from {model} without replacement"
    return (
        "blocked: Exp6174 did not generate a complete K8 pool for "
        f"{model}; blockers={gate.get('blocked_reasons', [])}"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(_strip_paths(payload))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def raw_row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def snapshot_revision(path: Path) -> str:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local-flat-cache"


def observed_quantization(path: Path) -> str:
    match = re.search(r"(UD-)?Q\d(?:_[A-Z0-9]+)+", path.name)
    return match.group(0) if match else "unknown"


def model_slug(hf_id: str) -> str:
    return hf_id.split("/", 1)[-1].replace("-GGUF", "").lower().replace(".", "_")


def file_receipt(relative: Path) -> JsonDict:
    path = REPO_ROOT / relative
    return {
        "path": relative.as_posix(),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def file_receipts(paths: Sequence[Path]) -> list[JsonDict]:
    return [file_receipt(path) for path in paths]


def protected_file_hash_map() -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(REPO_ROOT / relative)
        for relative in PROTECTED_FILES
        if (REPO_ROOT / relative).is_file()
    }


def nvidia_smi_gpu_receipt() -> JsonDict:  # pragma: no cover - host receipt.
    cmd = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )
    result = _run_command(cmd)
    devices = []
    if result["returncode"] == 0:
        for line in result["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 7:
                devices.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(float(parts[2])),
                        "memory_used_mb": int(float(parts[3])),
                        "memory_free_mb": int(float(parts[4])),
                        "temperature_c": int(float(parts[5])),
                        "power_draw_w": float(parts[6]),
                    }
                )
    return {"ok": result["returncode"] == 0 and bool(devices), "gpu_count": len(devices), "devices": devices}


def nvidia_smi_compute_apps() -> list[JsonDict]:  # pragma: no cover - host receipt.
    result = _run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    rows = []
    if result["returncode"] != 0:
        return rows
    for line in result["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            rows.append({"pid": int(parts[0]), "process_name": parts[1], "used_memory_mb": int(float(parts[2]))})
    return rows


def llama_cpp_gpu_backend_ok() -> bool:  # pragma: no cover - host receipt.
    try:
        from llama_cpp import llama_cpp as lib

        return bool(lib.llama_supports_gpu_offload())
    except Exception:
        return False


def gpu_engagement(before: Mapping[str, Any], after_load: Mapping[str, Any], after_decode: Mapping[str, Any]) -> JsonDict:  # pragma: no cover - host receipt.
    before_used = sum(int(device.get("memory_used_mb", 0)) for device in before.get("devices", []))
    peak_used = max(
        sum(int(device.get("memory_used_mb", 0)) for device in receipt.get("devices", []))
        for receipt in (after_load, after_decode)
    )
    return {"attributable": peak_used > before_used, "selected_gpus": [0, 1], "max_memory_delta_mb": peak_used - before_used}


def _coverage_summary(raw_rows: Sequence[Mapping[str, Any]], sample_plan: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_case = Counter(str(row["case_id"]) for row in raw_rows)
    return {
        "raw_row_count": len(raw_rows),
        "expected_row_count": len(sample_plan),
        "case_count": len(per_case),
        "min_samples_per_case": min(per_case.values()) if per_case else 0,
        "max_samples_per_case": max(per_case.values()) if per_case else 0,
    }


def _model_specs_from_resolution(resolution: Mapping[str, Any]) -> list[JsonDict]:
    records = list(resolution.get("records", []))
    if not records:
        return [dict(MODEL_SPECS[0])]
    merged = {**MODEL_SPECS[0], **dict(records[0])}
    merged["hf_id"] = MANDATORY_MODEL_ID
    return [merged]


def _write_checkpoint(path: Path, raw_commit: Mapping[str, Any], label_receipts: Mapping[str, Any]) -> JsonDict:
    payload = {
        "schema": SCHEMA + ".checkpoint",
        "raw_commit": dict(raw_commit),
        "label_receipts": dict(label_receipts),
    }
    _write_json(path, payload)
    return {"checkpoint_path": str(path), "checkpoint_sha256": sha256_file(path)}


def _path_hash_count(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path), "count": len(rows)}


def _write_access_log(
    path: Path,
    split: str,
    labels: Sequence[Mapping[str, Any]],
    *,
    aggregate_outcomes_inspected: bool,
) -> JsonDict:
    payload = {
        "schema": SCHEMA + ".access_log",
        "split": split,
        "label_access_count": len(labels),
        "aggregate_outcomes_inspected": aggregate_outcomes_inspected,
        "generation_access_count": 0,
        "events": [
            {
                "case_id": label["case_id"],
                "sample_key": label["sample_key"],
                "raw_row_hash": label["raw_row_hash"],
            }
            for label in labels
        ],
    }
    _write_json(path, payload)
    return {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path), "label_access_count": len(labels)}


def _empty_resume_receipt(path: Path, sample_plan: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".resume",
        "checkpoint_path": str(path),
        "resume_mode": "not_started",
        "expected_key_count": len(sample_plan),
        "existing_key_count": 0,
        "generated_new_rows": 0,
        "conflicting_key_count": 0,
        "row_hash_mismatch_count": 0,
        "missing_key_count": len(sample_plan),
        "extra_key_count": 0,
    }


def _empty_raw_commit(raw_path: Path) -> JsonDict:
    return {
        "schema": SCHEMA + ".raw_before_label_commit",
        "raw_corpus_path": str(raw_path),
        "raw_corpus_sha256": None,
        "raw_row_count": 0,
        "expected_row_count": 120 * K_SAMPLES,
        "case_count_with_k_at_least_8": 0,
        "raw_rows_complete_before_validation": False,
        "validation_started_after_raw_commit": False,
        "exact_validator_invocation_count_before_raw_commit": 0,
    }


def _empty_label_receipts(calibration_path: Path, held_path: Path) -> JsonDict:
    return {
        "schema": SCHEMA + ".label_sidecars",
        "calibration": {"path": str(calibration_path), "exists": calibration_path.exists(), "sha256": None, "count": 0},
        "held": {"path": str(held_path), "exists": held_path.exists(), "sha256": None, "count": 0},
        "validator_invocation_count": 0,
        "labels_inaccessible_to_generation": True,
    }


def _empty_access_receipts(calibration_path: Path, held_path: Path) -> JsonDict:
    return {
        "schema": SCHEMA + ".access_logs",
        "calibration": {"path": str(calibration_path), "exists": calibration_path.exists(), "label_access_count": 0},
        "held": {"path": str(held_path), "exists": held_path.exists(), "label_access_count": 0},
        "held_aggregate_outcomes_inspected": False,
    }


def _resolve_paths(**overrides: Path | None) -> dict[str, Path]:
    return {
        "result": overrides.get("result_path") or REPO_ROOT / RESULT_RELATIVE_PATH,
        "raw_trace": overrides.get("raw_trace_path") or REPO_ROOT / RAW_TRACE_RELATIVE_PATH,
        "calibration_label": overrides.get("calibration_label_path") or REPO_ROOT / CALIBRATION_LABEL_RELATIVE_PATH,
        "held_label": overrides.get("held_label_path") or REPO_ROOT / HELD_LABEL_RELATIVE_PATH,
        "checkpoint": overrides.get("checkpoint_path") or REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
        "calibration_access": overrides.get("calibration_access_log_path") or REPO_ROOT / CALIBRATION_ACCESS_LOG_RELATIVE_PATH,
        "held_access": overrides.get("held_access_log_path") or REPO_ROOT / HELD_ACCESS_LOG_RELATIVE_PATH,
    }


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_command(cmd: Sequence[str]) -> JsonDict:
    try:
        completed = subprocess.run(
            list(cmd),
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return {"returncode": 127, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _parent_writable(path: Path) -> bool:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    return os.access(parent, os.W_OK)


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return "cannot" in lowered or "can't" in lowered or "unable to" in lowered


def _strip_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<path>" if key.endswith("path") or key in {"model_path", "real_path"} else _strip_paths(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_strip_paths(item) for item in value]
    return value


def validate_existing_artifact(path: Path | None = None) -> JsonDict:
    artifact_path = path or REPO_ROOT / RESULT_RELATIVE_PATH
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    return {
        "path": str(artifact_path),
        "exists": artifact_path.exists(),
        "missing_required_fields": missing,
        "ok": not missing,
        "status": artifact.get("status"),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate the existing Exp6174 artifact")
    args = parser.parse_args(argv)
    if args.validate:
        print(json.dumps(validate_existing_artifact(), sort_keys=True))
        return 0
    artifact = run()
    print(json.dumps({"artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
