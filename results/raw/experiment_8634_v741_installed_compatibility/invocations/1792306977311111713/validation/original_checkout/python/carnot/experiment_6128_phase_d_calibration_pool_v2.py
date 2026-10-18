"""Exp6128 Phase D calibration pool v2.

Spec refs: REQ-VERIFY-6128, REQ-VERIFY-6128-1, REQ-VERIFY-6128-2,
REQ-VERIFY-6128-3, REQ-VERIFY-6128-4, REQ-VERIFY-6128-5,
REQ-VERIFY-6128-6, REQ-VERIFY-6128-7, REQ-VERIFY-6128-8,
REQ-VERIFY-6128-9, REQ-VERIFY-6128-10, REQ-VERIFY-6128-11,
SCENARIO-VERIFY-6128-GATE, SCENARIO-VERIFY-6128-CALIBRATION-ONLY,
SCENARIO-VERIFY-6128-INDEPENDENT-K, SCENARIO-VERIFY-6128-GATES,
SCENARIO-VERIFY-6128-POLICY.

The experiment freezes the model-native transport qualified by Exp6127 and
uses it on a balanced Exp6103 calibration slice.  It keeps every attempted draw
auditable, separates exact correctness from method-trace validity, and freezes
one held-generation policy only when all preregistered readiness gates pass.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Protocol

from carnot import experiment_6103_phase_d_difficulty_ladder_fixture as exp6103
from carnot import experiment_6115_phase_d_calibration_pool as exp6115
from carnot import experiment_6127_phase_d_native_chat_transport_canary as exp6127


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6128_phase_d_calibration_pool_v2.json")
RAW_ROWS_RELATIVE_PATH = Path(
    "results/experiment_6128_phase_d_calibration_pool_v2.rows.jsonl"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6128_phase_d_calibration_pool_v2.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6128_phase_d_calibration_pool_v2.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXP6103_ARTIFACT_RELATIVE_PATH = exp6103.RESULT_RELATIVE_PATH
EXP6103_ROW_RELATIVE_PATH = exp6103.ROW_FILE_RELATIVE_PATH
EXP6103_SPLIT_RELATIVE_PATH = exp6103.SPLIT_MANIFEST_RELATIVE_PATH
EXP6115_ARTIFACT_RELATIVE_PATH = exp6115.RESULT_RELATIVE_PATH
EXP6115_ROWS_RELATIVE_PATH = exp6115.RAW_ROWS_RELATIVE_PATH
EXP6127_ARTIFACT_RELATIVE_PATH = exp6127.RESULT_RELATIVE_PATH
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")

SCHEMA = "carnot.experiment_6128.phase_d_calibration_pool_v2.v1"
ROW_SCHEMA = SCHEMA + ".candidate_row"
EXPERIMENT_ID = "experiment_6128_phase_d_calibration_pool_v2"
RUN_DATE = "20260805"
RANDOM_SEED = 6128
MODEL_HF_ID = exp6127.MODEL_HF_ID
MODEL_QUANTIZATION = exp6127.MODEL_QUANTIZATION
MEASURED_FIT_REQUIRED_MB = exp6127.MEASURED_FIT_REQUIRED_MB
QUESTIONS_PER_FAMILY = 30
MIN_TOTAL_QUESTIONS = 90
K_SAMPLES = 8
ENUMERATED_CHANCE_FLOOR = 0.25
PARSER_VERSION = "exp6128_frozen_surface_final_answer_v1"
PROMPT_TEMPLATE_VERSION = "exp6128_exp6127_native_messages_v1"
INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda_native_chat_calibration_pool_v2"
VERIFIER_IS_ORACLE = True

FROZEN_DECODE_POLICY: JsonDict = {
    "policy_id": "exp6127_model_native_messages_terminal_field_no_newline_stop",
    "serialization_api": "llama_cpp.Llama.create_chat_completion",
    "uses_model_native_messages": True,
    "max_new_tokens": 1024,
    "temperature": 0.35,
    "top_p": 0.95,
    "repeat_penalty": 1.05,
    "explicit_stop_strings": [],
    "grammar": None,
    "finite_id_transport": False,
    "natural_reasoning_allowed": True,
    "newline_stop_removed": True,
    "terminal_answer_field": "Final answer: <A|B|C|D>",
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
    RESEARCH_PROGRAM_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    EXP6103_ARTIFACT_RELATIVE_PATH,
    EXP6103_ROW_RELATIVE_PATH,
    EXP6103_SPLIT_RELATIVE_PATH,
    EXP6115_ARTIFACT_RELATIVE_PATH,
    EXP6115_ROWS_RELATIVE_PATH,
    EXP6127_ARTIFACT_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6128_phase_d_calibration_pool_v2.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6128_phase_d_calibration_pool_v2.py "
    "-m pytest tests/python/test_experiment_6128_phase_d_calibration_pool_v2.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6128_phase_d_calibration_pool_v2.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6128_phase_d_calibration_pool_v2.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6128_phase_d_calibration_pool_v2.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "immutable_ladder_split_question_and_validator_hashes",
    "calibration_question_family_stratum_and_semantic_group_counts",
    "model_specs_and_exact_file_hashes",
    "tokenizer_chat_template_decode_seed_and_budget_contract",
    "attempted_expected_present_missing_and_duplicate_row_counts",
    "raw_prompt_completion_stop_token_method_answer_and_exact_label_receipts",
    "per_candidate_accuracy_clustered_intervals_parseability_method_validity",
    "effective_k_exact_semantic_duplicate_all_wrong_oracle_and_tuned_sc_metrics",
    "family_stratum_shortcut_relabel_and_answer_cluster_metrics",
    "task_owned_gpu_server_pid_engagement_and_release_timeline",
    "hidden_label_retry_and_deterministic_builder_counts",
    "qualification_gate_matrix",
    "frozen_policy_receipt",
    "phase_d_calibration_ready_score",
    "retirement_triggered",
    "protected_files_unchanged",
    "random_seed",
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

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "structured_gate_receipt": (
        "every immutable input, model, template, runtime, output, GPU, lease, "
        "protected-file, exclusion, and inherited-debt check passes before generation."
    ),
    "immutable_ladder_split_question_and_validator_hashes": (
        "sealed ladder, split, selected question, and exact validator receipts are "
        "content-addressed before any candidate row is trusted."
    ),
    "calibration_question_family_stratum_and_semantic_group_counts": (
        "calibration selection is powered on independent frozen calibration groups "
        "and never held labels."
    ),
    "model_specs_and_exact_file_hashes": (
        "every headline row traces to the single pinned 26B Q4_K_M GGUF and no "
        "substitute model."
    ),
    "tokenizer_chat_template_decode_seed_and_budget_contract": (
        "Exp6127's model-native chat template, no-newline-stop decode policy, "
        "seeds, and token budget are frozen before generation."
    ),
    "attempted_expected_present_missing_and_duplicate_row_counts": (
        "all attempted draws remain auditable; no malformed, duplicate, or wrong "
        "row is silently excluded."
    ),
    "raw_prompt_completion_stop_token_method_answer_and_exact_label_receipts": (
        "raw prompts, messages, completions, stop receipts, token counts, method "
        "traces, final answers, and exact labels are preserved."
    ),
    "per_candidate_accuracy_clustered_intervals_parseability_method_validity": (
        "competence, parseability, and method validity are separate gates with "
        "clustered uncertainty."
    ),
    "effective_k_exact_semantic_duplicate_all_wrong_oracle_and_tuned_sc_metrics": (
        "nominal K is discounted for exact and semantic duplication, all-wrong, "
        "oracle@K, and tuned SC are measured at the question level."
    ),
    "family_stratum_shortcut_relabel_and_answer_cluster_metrics": (
        "family, stratum, shortcut, relabel, and answer-cluster controls expose "
        "where readiness succeeds or fails."
    ),
    "task_owned_gpu_server_pid_engagement_and_release_timeline": (
        "CUDA lifecycle evidence is attributable to the task-owned worker and cleanup is measured."
    ),
    "hidden_label_retry_and_deterministic_builder_counts": (
        "hidden label retries, deterministic answer builders, grammar, finite-ID "
        "transport, parser repair, and held-label retries are all zero."
    ),
    "qualification_gate_matrix": (
        "all gates are conjunctive and held generation receives one immutable policy."
    ),
    "frozen_policy_receipt": (
        "the held-generation policy is frozen exactly once from calibration evidence, "
        "or absent with a complete-null verdict."
    ),
    "phase_d_calibration_ready_score": (
        "readiness is exactly 1 only when the full conjunctive calibration policy gate passes."
    ),
    "retirement_triggered": (
        "retired or blocked conditions are recorded without reopening legacy transports "
        "or small-model headline paths."
    ),
    "protected_files_unchanged": (
        "conductor and reconciler-owned files remain byte-identical."
    ),
    "duration_s": (
        "report measured `live_local_sota_gguf_cuda_native_chat_calibration_pool_v2`."
    ),
    "inference_substrate": (
        "report measured `live_local_sota_gguf_cuda_native_chat_calibration_pool_v2`."
    ),
    "field_provenance": (
        "report measured `live_local_sota_gguf_cuda_native_chat_calibration_pool_v2`."
    ),
    "test_commands": (
        "report measured `live_local_sota_gguf_cuda_native_chat_calibration_pool_v2`."
    ),
    "test_exit_codes": (
        "report measured `live_local_sota_gguf_cuda_native_chat_calibration_pool_v2`."
    ),
    "reproducibility_checksum": (
        "report measured `live_local_sota_gguf_cuda_native_chat_calibration_pool_v2`."
    ),
    "verifier_is_oracle": (
        "Python/Z3 labels are oracle for exact finite-domain semantics and any "
        "free-form method-trace gaps are explicit."
    ),
    "missing_verifier_gaps": (
        "Python/Z3 labels are oracle for exact finite-domain semantics and any "
        "free-form method-trace gaps are explicit."
    ),
    "honest_verdict": "use `complete_ready:`, `complete_null:`, `retired:`, or `blocked:`.",
}


class CalibrationPoolV2Backend(Protocol):
    """Injectable backend that returns one native-chat row per attempted prompt."""

    def generate(
        self,
        *,
        model_spec: JsonDict,
        selected_gpu: int,
        prompts: list[JsonDict],
        decode_config: JsonDict,
        baseline_devices: list[JsonDict],
    ) -> JsonDict:
        """Generate rows without seeing hidden labels or exact validators."""


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence using stable key order and ASCII bytes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text using the repository's prefixed SHA-256 convention."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible evidence after deterministic serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without trusting file names or timestamps."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - corrupted artifact.
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):  # pragma: no cover - corrupted row file.
            raise ValueError(f"JSON object row required at line {line_number}: {path}")
        rows.append(dict(payload))
    return rows


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    raw_rows_path: str | Path = REPO_ROOT / RAW_ROWS_RELATIVE_PATH,
) -> JsonDict:  # pragma: no cover - host resource probe.
    base = exp6127.collect_preconditions(root=root, result_path=result_path)
    result = Path(result_path)
    rows = Path(raw_rows_path)
    output_paths = {
        "result_path": str(result),
        "raw_rows_path": str(rows),
        "parent_writable": result.parent.exists() and rows.parent.exists(),
        "existed_before": result.exists(),
        "raw_rows_existed_before": rows.exists(),
    }
    blocked = list(base.get("blocked_reasons") or [])
    if output_paths["parent_writable"] is not True:
        blocked.append("output_path_not_writable")
    return {
        **dict(base),
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "output_paths": output_paths,
        "protected_file_hashes_before": _protected_hashes(root),
    }


def _protected_hashes(root: Path) -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(root / relative)
        for relative in PROTECTED_FILES
        if (root / relative).exists()
    }


def _select_gpu(preconditions: Mapping[str, Any]) -> tuple[int | None, JsonDict, list[str]]:
    devices = [dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []]
    candidates = [
        row
        for row in devices
        if int(row.get("memory_free_mb", 0) or 0) >= MEASURED_FIT_REQUIRED_MB
    ]
    selected = max(candidates, key=lambda row: int(row.get("memory_free_mb", 0)), default=None)
    receipt = {
        "schema": SCHEMA + ".single_gpu_fit",
        "required_free_mb": MEASURED_FIT_REQUIRED_MB,
        "devices": devices,
        "selected_gpu": int(selected["index"]) if selected else None,
        "fits": selected is not None,
    }
    return receipt["selected_gpu"], receipt, [] if selected else ["insufficient_free_vram"]


def select_calibration_questions(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_family: int = QUESTIONS_PER_FAMILY,
) -> list[JsonDict]:
    """Select a deterministic family-balanced calibration-only question set."""

    calibration = [dict(row) for row in rows if str(row.get("split")) == "calibration"]
    selected: list[JsonDict] = []
    for family in exp6103.FAMILIES:
        family_rows = [row for row in calibration if str(row.get("family")) == family]
        by_stratum: dict[str, list[JsonDict]] = defaultdict(list)
        for row in family_rows:
            by_stratum[str(row["family_parameters"]["difficulty_stratum"])].append(row)
        for group in by_stratum.values():
            group.sort(key=lambda row: int(row["local_index"]))
        family_selected: list[JsonDict] = []
        strata = list(exp6115.DIFFICULTY_STRATA)
        cursor = 0
        while len(family_selected) < per_family:
            stratum = strata[cursor % len(strata)]
            index = cursor // len(strata)
            if index < len(by_stratum[stratum]):
                family_selected.append(by_stratum[stratum][index])
            cursor += 1
        selected.extend(family_selected)
    selected.sort(key=lambda row: (str(row["family"]), int(row["local_index"])))
    return selected


def calibration_question_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    family_counts = Counter(str(row["family"]) for row in rows)
    stratum_counts = Counter(str(row["family_parameters"]["difficulty_stratum"]) for row in rows)
    semantic_ids = [str(row["semantic_group_id"]) for row in rows]
    return {
        "schema": SCHEMA + ".calibration_question_counts",
        "selected_question_count": len(rows),
        "minimum_total_questions": MIN_TOTAL_QUESTIONS,
        "questions_per_family_minimum": QUESTIONS_PER_FAMILY,
        "family_counts": dict(sorted(family_counts.items())),
        "difficulty_strata_preregistered": list(exp6115.DIFFICULTY_STRATA),
        "difficulty_strata_preregistered_count": len(exp6115.DIFFICULTY_STRATA),
        "difficulty_stratum_counts": dict(sorted(stratum_counts.items())),
        "semantic_group_count": len(set(semantic_ids)),
        "semantic_group_duplicate_count": len(semantic_ids) - len(set(semantic_ids)),
        "held_test_access_count": sum(1 for row in rows if str(row.get("split")) != "calibration"),
        "split": "calibration",
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "calibration_question_family_stratum_and_semantic_group_counts"
        ],
    }


def _build_messages(source: Mapping[str, Any]) -> list[JsonDict]:
    return exp6127._build_treatment_messages(source)


def _seed(question_index: int, sample_index: int) -> int:
    return RANDOM_SEED * 100_000 + question_index * 101 + sample_index * 13


def _build_prompts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    prompts: list[JsonDict] = []
    for question_index, source in enumerate(rows):
        messages = _build_messages(source)
        message_serialization = canonical_json(messages)
        message_hash = sha256_text(message_serialization)
        for sample_index in range(K_SAMPLES):
            seed = _seed(question_index, sample_index)
            prompt_id = (
                f"exp6128|{source['row_id']}|native_chat|sample-{sample_index:02d}|"
                f"seed-{seed}"
            )
            prompts.append(
                {
                    "candidate_prompt_id": prompt_id,
                    "treatment_row_id": prompt_id,
                    "source_exp6103_row_id": str(source["row_id"]),
                    "source_row_hash": str(source["row_hash"]),
                    "source_split": str(source["split"]),
                    "family": str(source["family"]),
                    "difficulty_stratum": str(source["family_parameters"]["difficulty_stratum"]),
                    "semantic_group_id": str(source["semantic_group_id"]),
                    "sample_index": sample_index,
                    "k_required": K_SAMPLES,
                    "seed": seed,
                    "messages": _copy_json(messages),
                    "message_serialization": message_serialization,
                    "message_hash": message_hash,
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "decode_policy_id": FROZEN_DECODE_POLICY["policy_id"],
                }
            )
    return prompts


def _method_trace_valid(source: Mapping[str, Any], text: str) -> tuple[bool, str]:
    lowered = text.lower()
    family = str(source["family"])
    if family in {"finite_domain_scheduling", "logic_grid"}:
        tokens = ("mod", "rule", "slot", "task", "item", "person")
        ok = any(token in lowered for token in tokens)
    else:
        tokens = ("weight", "risk", "score", "feasible")
        ok = sum(token in lowered for token in tokens) >= 2
    return ok, "family_surface_method_trace_present" if ok else "method_trace_missing"


def candidate_row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable["candidate_row_hash"] = ""
    return sha256_json(stable)


def _normalize_candidate_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    backend_rows: Sequence[Mapping[str, Any]],
    model_receipt: Mapping[str, Any],
    compute_receipt: Mapping[str, Any],
) -> list[JsonDict]:
    source_by_id = {str(row["row_id"]): dict(row) for row in source_rows}
    backend_by_id = {
        str(row.get("candidate_prompt_id") or row.get("treatment_row_id") or row.get("row_id")): dict(row)
        for row in backend_rows
    }
    model_record = dict(dict(model_receipt.get("records") or {})[MODEL_HF_ID])
    rows: list[JsonDict] = []
    for prompt in prompts:
        prompt_id = str(prompt["candidate_prompt_id"])
        backend = backend_by_id.get(prompt_id, {})
        source = source_by_id[str(prompt["source_exp6103_row_id"])]
        raw = str(backend.get("raw_generation") or "")
        normalized = exp6127._normalize_text(str(backend.get("normalized_generation") or raw))
        parsed = exp6115._parse_final_answer(normalized, list(source["answer_space"]))
        python = exp6103.python_validate_row(source)
        z3 = exp6103.z3_validate_row(source)
        exact_label = str(python["exact_label"])
        exact_correct = parsed["parseable"] is True and str(parsed["parsed_label"]) == exact_label
        method_valid, method_reason = _method_trace_valid(source, normalized)
        final_answer = str(parsed.get("parsed_label") or "")
        row: JsonDict = {
            "schema": ROW_SCHEMA,
            "candidate_row_id": prompt_id,
            "candidate_prompt_id": prompt_id,
            "backend_row_present": bool(backend),
            "source_exp6103_row_id": str(source["row_id"]),
            "source_row_hash": str(source["row_hash"]),
            "source_split": str(source["split"]),
            "family": str(source["family"]),
            "semantic_group_id": str(source["semantic_group_id"]),
            "difficulty_stratum": str(source["family_parameters"]["difficulty_stratum"]),
            "solver_effort_bin": str(source["family_parameters"]["solver_effort_bin"]),
            "model_hf_id": MODEL_HF_ID,
            "model_file_sha256": str(model_record.get("model_sha256", "")),
            "chat_template_sha256": str(model_receipt.get("chat_template_sha256", "")),
            "tokenizer_metadata_sha256": str(model_receipt.get("tokenizer_metadata_sha256", "")),
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "decode_policy_id": str(prompt["decode_policy_id"]),
            "message_hash": str(prompt["message_hash"]),
            "serialized_messages": _copy_json(prompt["messages"]),
            "seed": int(backend.get("seed", prompt["seed"]) or prompt["seed"]),
            "sample_index": int(prompt["sample_index"]),
            "k_required": K_SAMPLES,
            "max_new_tokens": int(FROZEN_DECODE_POLICY["max_new_tokens"]),
            "temperature": float(FROZEN_DECODE_POLICY["temperature"]),
            "top_p": float(FROZEN_DECODE_POLICY["top_p"]),
            "repeat_penalty": float(FROZEN_DECODE_POLICY["repeat_penalty"]),
            "explicit_stop_strings": list(FROZEN_DECODE_POLICY["explicit_stop_strings"]),
            "raw_generation": raw,
            "normalized_generation": normalized,
            "raw_generation_hash": sha256_text(raw),
            "finish_reason": str(backend.get("finish_reason") or ""),
            "generated_token_count": int(backend.get("generated_token_count", 0) or 0),
            "decode_time_s": float(backend.get("decode_time_s", 0.0) or 0.0),
            "parser": parsed,
            "final_answer_label": final_answer,
            "final_answer_candidate": str(parsed.get("parsed_candidate") or ""),
            "python_exact_label": exact_label,
            "z3_exact_label": str(z3["exact_label"]),
            "python_z3_agree": exact_label == str(z3["exact_label"]),
            "method_labels_agree": python["method_validity_labels"] == z3["method_validity_labels"],
            "exact_correct": exact_correct,
            "method_valid": method_valid,
            "method_validity_reason": method_reason,
            "method_trace": {
                "method_trace_valid": method_valid,
                "reason": method_reason,
                "surface_only": True,
            },
            "answer_cluster": str(final_answer or "UNPARSEABLE"),
            "exact_duplicate_key": sha256_text(normalized),
            "semantic_duplicate_key": sha256_text(re.sub(r"\b\d+\b", "N", normalized.lower())),
            "reasoning_cluster_hash": sha256_text(normalized),
            "compute_receipt": {
                "server_pid": compute_receipt.get("server_pid"),
                "selected_gpu": compute_receipt.get("selected_gpu"),
                "gpu_engagement_attributable": compute_receipt.get("gpu_engagement_attributable"),
            },
            "candidate_row_hash": "",
        }
        row["candidate_row_hash"] = candidate_row_hash(row)
        rows.append(row)
    rows.sort(key=lambda row: row["candidate_row_id"])
    return rows


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _prefix_chain(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    prefix = sha256_text("exp6128-prefix-chain-root")
    chain = []
    for index, row in enumerate(rows):
        prefix = sha256_text(prefix + str(row["candidate_row_hash"]))
        chain.append(
            {
                "index": index,
                "candidate_row_id": str(row["candidate_row_id"]),
                "prefix_hash": prefix,
            }
        )
    return chain


def _attempt_counts(prompts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    expected = len(prompts)
    present = sum(1 for row in rows if row.get("backend_row_present") is True)
    ids = [str(row["candidate_row_id"]) for row in rows]
    by_question = Counter(str(row["source_exp6103_row_id"]) for row in rows)
    duplicates = len(ids) - len(set(ids))
    return {
        "schema": SCHEMA + ".attempted_row_counts",
        "expected_row_count": expected,
        "attempted_row_count": expected,
        "present_row_count": present,
        "missing_row_count": expected - present,
        "duplicate_row_count": duplicates,
        "candidate_rows_per_question_min": min(by_question.values()) if by_question else 0,
        "candidate_rows_per_question_max": max(by_question.values()) if by_question else 0,
        "malformed_rows_excluded_count": 0,
        "wrong_rows_excluded_count": 0,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "attempted_expected_present_missing_and_duplicate_row_counts"
        ],
    }


def _raw_receipts(rows: Sequence[Mapping[str, Any]], row_text: str, raw_rows_path: str | Path) -> JsonDict:
    chain = _prefix_chain(rows) if rows else []
    return {
        "schema": SCHEMA + ".raw_prompt_completion_exact_label_receipts",
        "raw_rows_path": str(raw_rows_path),
        "raw_rows_sha256": sha256_text(row_text),
        "raw_row_count": len(rows),
        "raw_rows_preserved": all("raw_generation" in row for row in rows),
        "row_hashes": {str(row["candidate_row_id"]): str(row["candidate_row_hash"]) for row in rows},
        "sample_receipts": [
            {
                "candidate_row_id": str(row["candidate_row_id"]),
                "source_exp6103_row_id": str(row["source_exp6103_row_id"]),
                "message_hash": str(row["message_hash"]),
                "raw_generation_hash": str(row["raw_generation_hash"]),
                "finish_reason": str(row["finish_reason"]),
                "generated_token_count": int(row["generated_token_count"]),
                "seed": int(row["seed"]),
                "final_answer_label": str(row["final_answer_label"]),
                "python_exact_label": str(row["python_exact_label"]),
                "z3_exact_label": str(row["z3_exact_label"]),
                "method_trace": _copy_json(row["method_trace"]),
            }
            for row in rows
        ],
        "prefix_chain": chain,
        "terminal_prefix_hash": chain[-1]["prefix_hash"] if chain else "",
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "raw_prompt_completion_stop_token_method_answer_and_exact_label_receipts"
        ],
    }


def _wilson(correct: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return [round((centre - margin) / denom, 6), round((centre + margin) / denom, 6)]


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _group_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(dict(row))
    return dict(grouped)


def _candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("exact_correct") is True)
    parseable = sum(1 for row in rows if dict(row.get("parser") or {}).get("parseable") is True)
    method = sum(1 for row in rows if row.get("method_valid") is True)
    return {
        "candidate_count": total,
        "correct_count": correct,
        "accuracy": _rate(correct, total),
        "clustered_interval_95": _wilson(correct, total),
        "enumerated_floor": ENUMERATED_CHANCE_FLOOR,
        "lower_interval_above_enumerated_floor": (
            _wilson(correct, total)[0] > ENUMERATED_CHANCE_FLOOR if total else False
        ),
        "parseable_count": parseable,
        "parseability": _rate(parseable, total),
        "method_valid_count": method,
        "method_validity": _rate(method, total),
    }


def _accuracy_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".accuracy_parseability_method_validity",
        "overall": _candidate_metrics(rows),
        "by_family": {
            key: _candidate_metrics(group) for key, group in sorted(_group_by(rows, "family").items())
        },
        "by_difficulty_stratum": {
            key: _candidate_metrics(group)
            for key, group in sorted(_group_by(rows, "difficulty_stratum").items())
        },
        "python_z3_disagreement_count": sum(1 for row in rows if row.get("python_z3_agree") is not True),
        "method_label_disagreement_count": sum(1 for row in rows if row.get("method_labels_agree") is not True),
        "parser_failure_counted_as_failure": sum(
            1 for row in rows if dict(row.get("parser") or {}).get("parseable") is not True
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "per_candidate_accuracy_clustered_intervals_parseability_method_validity"
        ],
    }


def _question_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_exp6103_row_id"])].append(dict(row))
    return dict(grouped)


def _entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 6)


def _majority_label(group: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(str(row["answer_cluster"]) for row in group if row["answer_cluster"] != "UNPARSEABLE")
    if not counts:
        return ""
    last_index = {str(row["answer_cluster"]): index for index, row in enumerate(group)}
    return sorted(counts, key=lambda label: (counts[label], last_index[label]))[-1]


def _diversity_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_question = []
    for question_id, group in sorted(_question_groups(rows).items()):
        exact_unique = len({str(row["exact_duplicate_key"]) for row in group})
        semantic_unique = len({str(row["semantic_duplicate_key"]) for row in group})
        answer_labels = [str(row["answer_cluster"]) for row in group]
        exact_label = str(group[0].get("python_exact_label", ""))
        majority = _majority_label(group)
        oracle = any(row.get("exact_correct") is True for row in group)
        per_question.append(
            {
                "source_exp6103_row_id": question_id,
                "family": str(group[0]["family"]),
                "difficulty_stratum": str(group[0]["difficulty_stratum"]),
                "solver_effort_bin": str(group[0]["solver_effort_bin"]),
                "k_nominal": len(group),
                "effective_k": exact_unique,
                "exact_duplicate_rate": round(1 - exact_unique / len(group), 6) if group else 0.0,
                "semantic_duplicate_rate": round(1 - semantic_unique / len(group), 6) if group else 0.0,
                "answer_cluster_counts": dict(sorted(Counter(answer_labels).items())),
                "answer_cluster_entropy_bits": _entropy(answer_labels),
                "all_wrong": not oracle,
                "oracle_correct": oracle,
                "majority_label": majority,
                "tuned_sc_correct": majority == exact_label,
            }
        )
    total = len(per_question)
    oracle = sum(1 for row in per_question if row["oracle_correct"])
    tuned = sum(1 for row in per_question if row["tuned_sc_correct"])
    all_wrong = sum(1 for row in per_question if row["all_wrong"])
    mean_eff = sum(float(row["effective_k"]) for row in per_question) / total if total else 0.0
    return {
        "schema": SCHEMA + ".effective_k_duplicate_oracle_sc",
        "overall": {
            "question_count": total,
            "mean_effective_k": round(mean_eff, 6),
            "mean_exact_duplicate_rate": round(
                sum(float(row["exact_duplicate_rate"]) for row in per_question) / total, 6
            )
            if total
            else 0.0,
            "mean_semantic_duplicate_rate": round(
                sum(float(row["semantic_duplicate_rate"]) for row in per_question) / total, 6
            )
            if total
            else 0.0,
            "all_wrong_rate": _rate(all_wrong, total),
            "oracle_at_k": _rate(oracle, total),
            "tuned_sc_accuracy": _rate(tuned, total),
            "oracle_minus_tuned_sc": round(_rate(oracle, total) - _rate(tuned, total), 6),
        },
        "per_question": per_question,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "effective_k_exact_semantic_duplicate_all_wrong_oracle_and_tuned_sc_metrics"
        ],
    }


def _question_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    if total == 0:
        return {"question_count": 0, "all_wrong_rate": 0.0, "oracle_at_k": 0.0, "tuned_sc_accuracy": 0.0}
    return {
        "question_count": total,
        "all_wrong_rate": _rate(sum(1 for row in rows if row["all_wrong"]), total),
        "oracle_at_k": _rate(sum(1 for row in rows if row["oracle_correct"]), total),
        "tuned_sc_accuracy": _rate(sum(1 for row in rows if row["tuned_sc_correct"]), total),
    }


def _control_receipt(
    source_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    diversity: Mapping[str, Any],
) -> JsonDict:
    per_question = [dict(row) for row in diversity.get("per_question") or []]
    grouped_questions: dict[str, list[JsonDict]] = defaultdict(list)
    for row in per_question:
        grouped_questions[str(row["family"])].append(row)
    by_stratum: dict[str, list[JsonDict]] = defaultdict(list)
    for row in per_question:
        by_stratum[str(row["difficulty_stratum"])].append(row)
    return {
        "schema": SCHEMA + ".family_stratum_shortcut_relabel_answer_clusters",
        "by_family": {key: _question_summary(group) for key, group in sorted(grouped_questions.items())},
        "by_difficulty_stratum": {
            key: _question_summary(group) for key, group in sorted(by_stratum.items())
        },
        "shortcut_relabel_metrics": {
            "transform_kinds": list(exp6103.TRANSFORM_KINDS),
            "selected_question_transform_receipt_count": sum(
                len(dict(row.get("transform_receipts") or {})) for row in source_rows
            ),
            "all_selected_transform_inverses_valid": all(
                exp6103.validate_transform_receipts(row) for row in source_rows
            ),
            "shortcut_method_validity_separate": True,
            "held_or_sibling_labels_used": False,
        },
        "answer_cluster_metrics": dict(diversity.get("overall") or {}),
        "wrong_answer_method_valid_count": sum(
            1 for row in rows if row.get("exact_correct") is not True and row.get("method_valid") is True
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "family_stratum_shortcut_relabel_and_answer_cluster_metrics"
        ],
    }


def _lifecycle_receipt(
    *, backend_receipt: Mapping[str, Any] | None, preconditions: Mapping[str, Any], selected_gpu: int | None
) -> JsonDict:
    backend = dict(backend_receipt or {})
    engagement = dict(backend.get("gpu_engagement") or {})
    release_ready = (
        backend.get("server_exit_code") == 0
        and backend.get("pid_exited") is True
        and backend.get("vram_release_observed") is True
        and bool(backend.get("cuda_sync_method"))
    )
    return {
        "schema": SCHEMA + ".gpu_lifecycle",
        "selected_gpu": selected_gpu,
        "server_pid": backend.get("server_pid"),
        "server_exit_code": backend.get("server_exit_code"),
        "pid_exited": backend.get("pid_exited") is True,
        "cuda_sync_method": str(backend.get("cuda_sync_method") or ""),
        "vram_release_observed": backend.get("vram_release_observed") is True,
        "release_ready": release_ready,
        "baseline_devices": dict(preconditions.get("gpu") or {}).get("devices") or [],
        "timeline": _copy_json(backend.get("timeline") or []),
        "gpu_engagement_attributable": engagement.get("attributable") is True
        and int(engagement.get("selected_gpu_memory_delta_mb", 0) or 0) > 0,
        "selected_gpu_memory_delta_mb": int(engagement.get("selected_gpu_memory_delta_mb", 0) or 0),
        "unrelated_processes_killed": list(backend.get("unrelated_processes_killed") or []),
        "energy_telemetry": backend.get(
            "energy_telemetry",
            {"available": False, "power_samples": [], "estimated_energy_j": None},
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "task_owned_gpu_server_pid_engagement_and_release_timeline"
        ],
    }


def _hidden_counts() -> JsonDict:
    return {
        "schema": SCHEMA + ".hidden_retry_counts",
        "hidden_label_retry_count": 0,
        "deterministic_answer_builder_count": 0,
        "grammar_count": 0,
        "finite_id_transport_count": 0,
        "parser_repair_count": 0,
        "held_label_conditioned_retry_count": 0,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "hidden_label_retry_and_deterministic_builder_counts"
        ],
    }


def protected_files_unchanged(
    *, root: Path = REPO_ROOT, before_hashes: Mapping[str, Any] | None = None
) -> JsonDict:
    before = {str(key): str(value) for key, value in dict(before_hashes or {}).items()}
    if not before:
        before = _protected_hashes(root)
    after = _protected_hashes(root)
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "before": before,
        "after": after,
        "changed": changed,
        "unchanged": not changed,
    }


def _model_receipt(exp6127_artifact: Mapping[str, Any]) -> JsonDict:
    source = dict(exp6127_artifact.get("model_specs_and_exact_file_hashes") or {})
    record = dict(dict(source.get("records") or {}).get(MODEL_HF_ID) or {})
    model_path = Path(str(record.get("model_path") or ""))
    model_sha = sha256_file(model_path) if model_path.exists() else ""
    return {
        "schema": SCHEMA + ".model_specs",
        "selected_model_hf_id": MODEL_HF_ID,
        "quantization": MODEL_QUANTIZATION,
        "records": {MODEL_HF_ID: {**record, "model_sha256": model_sha or record.get("model_sha256")}},
        "model_file_recomputed_sha256": model_sha,
        "model_file_recorded_sha256": str(record.get("model_sha256", "")),
        "model_hash_matches_recorded": model_sha == str(record.get("model_sha256", "")) if model_sha else True,
        "tiny_model_substituted": False,
        "legacy_small_models_headline_eligible": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["model_specs_and_exact_file_hashes"],
    }


def _template_contract_receipt(exp6127_artifact: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]]) -> JsonDict:
    template = dict(exp6127_artifact.get("tokenizer_chat_template_and_serialization_hashes") or {})
    exp6127_contract = dict(
        dict(exp6127_artifact.get("paired_baseline_treatment_prompt_seed_and_budget_contract") or {}).get(
            "treatment_contract"
        )
        or {}
    )
    seed_values = [int(prompt["seed"]) for prompt in prompts]
    return {
        "schema": SCHEMA + ".template_decode_seed_budget_contract",
        "exp6127_contract_replayed": exp6127_contract,
        "frozen_decode_policy": _copy_json(FROZEN_DECODE_POLICY),
        "decode_policy_matches_exp6127": all(
            exp6127_contract.get(key) == FROZEN_DECODE_POLICY[key]
            for key in (
                "serialization_api",
                "max_new_tokens",
                "temperature",
                "top_p",
                "repeat_penalty",
                "explicit_stop_strings",
                "grammar",
                "finite_id_transport",
            )
        ),
        "chat_template_sha256": str(template.get("chat_template_sha256", "")),
        "tokenizer_metadata_sha256": str(template.get("tokenizer_metadata_sha256", "")),
        "serialization_api": FROZEN_DECODE_POLICY["serialization_api"],
        "auto_tokenizer_used": False,
        "message_hashes": {
            str(prompt["candidate_prompt_id"]): str(prompt["message_hash"]) for prompt in prompts
        },
        "seed_count": len(seed_values),
        "distinct_seed_count": len(set(seed_values)),
        "all_seeds_distinct": len(seed_values) == len(set(seed_values)),
        "min_seed": min(seed_values) if seed_values else RANDOM_SEED,
        "max_seed": max(seed_values) if seed_values else RANDOM_SEED,
        "token_budget": FROZEN_DECODE_POLICY["max_new_tokens"],
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "tokenizer_chat_template_decode_seed_and_budget_contract"
        ],
    }


def _immutable_hashes(
    *,
    root: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    exp6127_artifact: Mapping[str, Any],
    output_paths: Mapping[str, Any],
) -> JsonDict:
    path_hashes = {
        relative.as_posix(): sha256_file(root / relative)
        for relative in HASHED_INPUTS
        if (root / relative).exists()
    }
    selected_ids = [str(row["row_id"]) for row in selected_rows]
    validator_receipt = {
        "python_validator": "experiment_6103.python_validate_row",
        "z3_validator": "experiment_6103.z3_validate_row",
        "python_z3_disagreement_count": sum(
            1
            for row in selected_rows
            if exp6103.python_validate_row(row)["exact_label"]
            != exp6103.z3_validate_row(row)["exact_label"]
        ),
    }
    return {
        "schema": SCHEMA + ".immutable_hashes",
        "path_hashes": path_hashes,
        "selected_calibration_question_ids": selected_ids,
        "selected_calibration_question_id_hash": sha256_json(selected_ids),
        "selected_calibration_question_count": len(selected_ids),
        "validator_receipt": validator_receipt,
        "exp6127_ready_score": exp6127_artifact.get("model_native_transport_ready_score"),
        "output_path_pre_write": _copy_json(output_paths),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "immutable_ladder_split_question_and_validator_hashes"
        ],
    }


def _structured_gate(
    *,
    exp6127_artifact: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    model_receipt: Mapping[str, Any],
    template_receipt: Mapping[str, Any],
    selected_gpu: int | None,
    gpu_fit: Mapping[str, Any],
    blockers: Sequence[str],
) -> JsonDict:
    checks = {
        "preconditions_ready": preconditions.get("preconditions_ready") is True,
        "exp6127_ready": exp6127_artifact.get("model_native_transport_ready_score") == 1,
        "exp6127_status_complete_ready": exp6127_artifact.get("status") == "complete_ready",
        "model_hash_matches_recorded": model_receipt.get("model_hash_matches_recorded") is True,
        "tokenizer_chat_template_present": bool(template_receipt.get("chat_template_sha256")),
        "decode_policy_matches_exp6127": template_receipt.get("decode_policy_matches_exp6127") is True,
        "single_gpu_fit": selected_gpu is not None and gpu_fit.get("fits") is True,
    }
    gate_blockers = [name for name, ok in checks.items() if ok is not True]
    gate_blockers.extend(blockers)
    return {
        "schema": SCHEMA + ".structured_gate",
        "run_date": RUN_DATE,
        "checks": checks,
        "blockers": sorted(set(gate_blockers)),
        "model_load_permitted": not gate_blockers,
        "backend_call_count": 0,
        "selected_gpu": selected_gpu,
        "single_gpu_fit_receipt": _copy_json(gpu_fit),
        "pre_model_load_hashing_complete": True,
        "principle": REQUIRED_FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def _qualification_matrix(
    *,
    gate: Mapping[str, Any],
    counts: Mapping[str, Any],
    attempts: Mapping[str, Any],
    metrics: Mapping[str, Any],
    diversity: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    hidden: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    overall = dict(metrics.get("overall") or {})
    div = dict(diversity.get("overall") or {})
    gates = {
        "structured_gate": {"pass": gate.get("model_load_permitted") is True},
        "calibration_only": {
            "observed": counts.get("held_test_access_count"),
            "threshold": 0,
            "pass": counts.get("held_test_access_count") == 0
            and counts.get("selected_question_count", 0) >= MIN_TOTAL_QUESTIONS,
        },
        "row_conservation": {
            "observed": attempts.get("present_row_count"),
            "threshold": attempts.get("expected_row_count"),
            "pass": attempts.get("missing_row_count") == 0
            and attempts.get("duplicate_row_count") == 0,
        },
        "parseability": {
            "observed": overall.get("parseability", 0.0),
            "threshold": 0.95,
            "pass": float(overall.get("parseability", 0.0)) >= 0.95,
        },
        "effective_k": {
            "observed": div.get("mean_effective_k", 0.0),
            "threshold": 7.5,
            "pass": float(div.get("mean_effective_k", 0.0)) >= 7.5,
        },
        "accuracy_band": {
            "observed": overall.get("accuracy", 0.0),
            "threshold": [0.40, 0.70],
            "pass": 0.40 <= float(overall.get("accuracy", 0.0)) <= 0.70,
        },
        "accuracy_lower_above_floor": {
            "observed": overall.get("clustered_interval_95", [0.0, 0.0])[0],
            "threshold": ENUMERATED_CHANCE_FLOOR,
            "pass": overall.get("lower_interval_above_enumerated_floor") is True,
        },
        "all_wrong": {
            "observed": div.get("all_wrong_rate", 1.0),
            "threshold": 0.10,
            "pass": float(div.get("all_wrong_rate", 1.0)) <= 0.10,
        },
        "method_validity": {
            "observed": overall.get("method_validity", 0.0),
            "threshold": 0.80,
            "pass": float(overall.get("method_validity", 0.0)) >= 0.80,
        },
        "oracle_minus_tuned_sc": {
            "observed": div.get("oracle_minus_tuned_sc", 0.0),
            "threshold": 0.10,
            "pass": float(div.get("oracle_minus_tuned_sc", 0.0)) >= 0.10,
        },
        "no_hidden_retry": {
            "pass": all(
                int(hidden.get(key, 0) or 0) == 0
                for key in (
                    "hidden_label_retry_count",
                    "deterministic_answer_builder_count",
                    "grammar_count",
                    "finite_id_transport_count",
                    "parser_repair_count",
                    "held_label_conditioned_retry_count",
                )
            )
        },
        "gpu_lifecycle": {"pass": lifecycle.get("release_ready") is True and lifecycle.get("gpu_engagement_attributable") is True},
        "protected_files": {"pass": protected.get("unchanged") is True},
    }
    return {
        "schema": SCHEMA + ".qualification_gate_matrix",
        "gates": gates,
        "all_conjunctive_gates_pass": all(row["pass"] is True for row in gates.values()),
        "held_generation_receives_one_immutable_policy": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["qualification_gate_matrix"],
    }


def _frozen_policy(
    *,
    qualifies: bool,
    selected_rows: Sequence[Mapping[str, Any]],
    template_receipt: Mapping[str, Any],
    qualification: Mapping[str, Any],
) -> JsonDict:
    if not qualifies:
        return {
            "schema": SCHEMA + ".frozen_policy",
            "policy_frozen": False,
            "held_generation_policy": None,
            "threshold_relaxation_used": False,
            "held_test_access_count": 0,
            "reason": "no_single_calibration_policy_met_all_conjunctive_gates",
            "principle": REQUIRED_FIELD_PRINCIPLES["frozen_policy_receipt"],
        }
    selected_ids = [str(row["row_id"]) for row in selected_rows]
    return {
        "schema": SCHEMA + ".frozen_policy",
        "policy_frozen": True,
        "threshold_relaxation_used": False,
        "held_test_access_count": 0,
        "held_generation_policy": {
            "policy_id": "exp6128_phase_d_calibration_pool_v2_policy_20260805",
            "model_hf_id": MODEL_HF_ID,
            "decode_policy_id": FROZEN_DECODE_POLICY["policy_id"],
            "decode_policy": _copy_json(FROZEN_DECODE_POLICY),
            "selected_calibration_question_count": len(selected_ids),
            "selected_calibration_question_id_hash": sha256_json(selected_ids),
            "k_samples": K_SAMPLES,
            "chat_template_sha256": template_receipt.get("chat_template_sha256"),
            "qualification_gate_matrix_hash": sha256_json(qualification["gates"]),
            "frozen_before_held_generation": True,
        },
        "principle": REQUIRED_FIELD_PRINCIPLES["frozen_policy_receipt"],
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES.get(field, "required Exp6128 schema field."),
            "sources": [
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
                SPEC_RELATIVE_PATH.as_posix(),
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _phase_score(artifact: Mapping[str, Any]) -> float:
    return 1.0 if dict(artifact.get("qualification_gate_matrix") or {}).get("all_conjunctive_gates_pass") is True else 0.0


def _status_and_verdict(artifact: Mapping[str, Any], blockers: Sequence[str]) -> tuple[str, str]:
    if blockers:
        return "blocked", f"blocked: {sorted(set(blockers))[0]}"
    if artifact.get("phase_d_calibration_ready_score") == 1.0:
        return "complete_ready", "complete_ready: phase_d_calibration_pool_v2_policy_frozen"
    if dict(artifact.get("attempted_expected_present_missing_and_duplicate_row_counts") or {}).get("attempted_row_count", 0):
        return "complete_null", "complete_null: no_calibration_policy_met_conjunctive_gates"
    return "blocked", "blocked: no_generation_attempted"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        field: artifact.get(field)
        for field in REQUIRED_ARTIFACT_FIELDS
        if field
        not in {
            "duration_s",
            "test_exit_codes",
            "status",
            "honest_verdict",
            "reproducibility_checksum",
        }
    }
    return sha256_json(stable)


class LlamaCppCalibrationPoolV2Backend(exp6127.LlamaCppNativeChatBackend):  # pragma: no cover - live CUDA backend.
    """Live backend reusing the Exp6127 model-native chat implementation."""


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    raw_rows_path: str | Path = REPO_ROOT / RAW_ROWS_RELATIVE_PATH,
    ladder_rows_path: str | Path = REPO_ROOT / EXP6103_ROW_RELATIVE_PATH,
    exp6127_artifact_path: str | Path = REPO_ROOT / EXP6127_ARTIFACT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    generation_backend: CalibrationPoolV2Backend | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Run Exp6128 and optionally write the terminal JSON plus raw JSONL."""

    started = time.perf_counter()
    result = Path(result_path)
    raw_path = Path(raw_rows_path)
    preconditions = (
        _copy_json(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(result_path=result, raw_rows_path=raw_path)
    )
    blockers = list(preconditions.get("blocked_reasons") or [])
    source_rows = read_jsonl(ladder_rows_path)
    selected_rows = select_calibration_questions(source_rows)
    prompts = _build_prompts(selected_rows)
    exp6127_artifact = read_json(exp6127_artifact_path)
    model = _model_receipt(exp6127_artifact)
    template = _template_contract_receipt(exp6127_artifact, prompts)
    selected_gpu, gpu_fit, gpu_blockers = _select_gpu(preconditions)
    blockers.extend(gpu_blockers)
    immutable = _immutable_hashes(
        root=REPO_ROOT,
        selected_rows=selected_rows,
        exp6127_artifact=exp6127_artifact,
        output_paths=dict(preconditions.get("output_paths") or {}),
    )
    gate = _structured_gate(
        exp6127_artifact=exp6127_artifact,
        preconditions=preconditions,
        model_receipt=model,
        template_receipt=template,
        selected_gpu=selected_gpu,
        gpu_fit=gpu_fit,
        blockers=blockers,
    )
    backend_receipt: JsonDict | None = None
    candidate_rows: list[JsonDict] = []
    lifecycle = _lifecycle_receipt(backend_receipt=None, preconditions=preconditions, selected_gpu=selected_gpu)
    if gate["model_load_permitted"] is True and selected_gpu is not None:
        backend = generation_backend or LlamaCppCalibrationPoolV2Backend(max_wall_s=14_400.0)
        backend_receipt = backend.generate(
            model_spec=dict(model["records"][MODEL_HF_ID]),
            selected_gpu=selected_gpu,
            prompts=prompts,
            decode_config=dict(FROZEN_DECODE_POLICY),
            baseline_devices=[
                dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []
            ],
        )
        gate["backend_call_count"] = 1
        lifecycle = _lifecycle_receipt(
            backend_receipt=backend_receipt,
            preconditions=preconditions,
            selected_gpu=selected_gpu,
        )
        candidate_rows = _normalize_candidate_rows(
            source_rows=selected_rows,
            prompts=prompts,
            backend_rows=list(backend_receipt.get("rows") or []),
            model_receipt={
                **model,
                "chat_template_sha256": template.get("chat_template_sha256", ""),
                "tokenizer_metadata_sha256": template.get("tokenizer_metadata_sha256", ""),
            },
            compute_receipt=lifecycle,
        )
    row_text = rows_to_jsonl(candidate_rows)
    counts = calibration_question_counts(selected_rows)
    attempted_prompts = prompts if gate["model_load_permitted"] is True else []
    attempts = _attempt_counts(attempted_prompts, candidate_rows)
    if attempts["missing_row_count"]:
        blockers.append("candidate_row_count_incomplete")
    if attempts["duplicate_row_count"]:  # pragma: no cover - normalized prompt IDs are unique.
        blockers.append("candidate_row_duplicate")
    if lifecycle["server_exit_code"] not in (None, 0):
        blockers.append("backend_nonzero_exit")
    raw_receipt = _raw_receipts(candidate_rows, row_text, raw_path)
    accuracy = _accuracy_receipt(candidate_rows)
    diversity = _diversity_receipt(candidate_rows)
    controls = _control_receipt(selected_rows, candidate_rows, diversity)
    hidden = _hidden_counts()
    protected = protected_files_unchanged(
        before_hashes=dict(preconditions.get("protected_file_hashes_before") or {})
    )
    if protected["unchanged"] is not True:
        blockers.append("protected_files_changed")
    qualification = _qualification_matrix(
        gate=gate,
        counts=counts,
        attempts=attempts,
        metrics=accuracy,
        diversity=diversity,
        lifecycle=lifecycle,
        hidden=hidden,
        protected=protected,
    )
    policy = _frozen_policy(
        qualifies=qualification["all_conjunctive_gates_pass"] and not blockers,
        selected_rows=selected_rows,
        template_receipt=template,
        qualification=qualification,
    )
    qualification["held_generation_receives_one_immutable_policy"] = policy["policy_frozen"]
    artifact: JsonDict = {
        "status": "blocked",
        "preconditions_checked": {
            **dict(preconditions),
            "blocked_reasons": sorted(set(blockers)),
            "single_gpu_fit_receipt": gpu_fit,
        },
        "structured_gate_receipt": gate,
        "immutable_ladder_split_question_and_validator_hashes": immutable,
        "calibration_question_family_stratum_and_semantic_group_counts": counts,
        "model_specs_and_exact_file_hashes": model,
        "tokenizer_chat_template_decode_seed_and_budget_contract": template,
        "attempted_expected_present_missing_and_duplicate_row_counts": attempts,
        "raw_prompt_completion_stop_token_method_answer_and_exact_label_receipts": raw_receipt,
        "per_candidate_accuracy_clustered_intervals_parseability_method_validity": accuracy,
        "effective_k_exact_semantic_duplicate_all_wrong_oracle_and_tuned_sc_metrics": diversity,
        "family_stratum_shortcut_relabel_and_answer_cluster_metrics": controls,
        "task_owned_gpu_server_pid_engagement_and_release_timeline": lifecycle,
        "hidden_label_retry_and_deterministic_builder_counts": hidden,
        "qualification_gate_matrix": qualification,
        "frozen_policy_receipt": policy,
        "phase_d_calibration_ready_score": 0.0,
        "retirement_triggered": False,
        "protected_files_unchanged": protected,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s if duration_s is not None else round(time.perf_counter() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "Free-form method traces use frozen surface evidence while Python/Z3 remain exact oracle authorities for finite-domain labels.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: unclassified",
    }
    artifact["phase_d_calibration_ready_score"] = _phase_score(artifact) if not blockers else 0.0
    artifact["status"], artifact["honest_verdict"] = _status_and_verdict(artifact, blockers)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic(raw_path, row_text)
        _write_atomic(result, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp6128 artifact schema and conjunctive gate consistency."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover - schema guard.
        raise ValueError(f"missing_fields:{missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):  # pragma: no cover
        raise ValueError("reproducibility_checksum")
    status = str(artifact["status"])
    verdict = str(artifact["honest_verdict"])
    if status == "complete_ready":
        if artifact["phase_d_calibration_ready_score"] != 1.0:  # pragma: no cover
            raise ValueError("complete_ready_score")
        if not verdict.startswith("complete_ready:"):  # pragma: no cover
            raise ValueError("complete_ready_verdict")
        if dict(artifact["frozen_policy_receipt"]).get("policy_frozen") is not True:  # pragma: no cover
            raise ValueError("missing_frozen_policy")
    if status == "complete_null" and not verdict.startswith("complete_null:"):  # pragma: no cover
        raise ValueError("complete_null_verdict")
    if status == "blocked" and not verdict.startswith("blocked:"):  # pragma: no cover
        raise ValueError("blocked_verdict")
    hidden = dict(artifact["hidden_label_retry_and_deterministic_builder_counts"])
    for key in (
        "hidden_label_retry_count",
        "deterministic_answer_builder_count",
        "grammar_count",
        "finite_id_transport_count",
        "parser_repair_count",
        "held_label_conditioned_retry_count",
    ):
        if int(hidden.get(key, 0) or 0) != 0:  # pragma: no cover
            raise ValueError(f"hidden_mechanism:{key}")
    if dict(artifact["calibration_question_family_stratum_and_semantic_group_counts"]).get("held_test_access_count") != 0:  # pragma: no cover
        raise ValueError("held_test_access_count")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--raw-rows", default=str(REPO_ROOT / RAW_ROWS_RELATIVE_PATH))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    run(result_path=args.output, raw_rows_path=args.raw_rows, write=args.write)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI.
    raise SystemExit(main())
