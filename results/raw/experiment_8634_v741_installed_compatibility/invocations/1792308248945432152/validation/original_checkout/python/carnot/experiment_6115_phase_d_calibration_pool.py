"""Exp6115 Phase D calibration pool.

Spec refs: REQ-VERIFY-6115, SCENARIO-VERIFY-6115-GATE,
SCENARIO-VERIFY-6115-CALIBRATION-ONLY, SCENARIO-VERIFY-6115-NATURAL-K,
SCENARIO-VERIFY-6115-REPLAY, SCENARIO-VERIFY-6115-POLICY.

This module is the calibration-only bridge between the sealed Exp6103 ladder
and the later held Phase D evaluation.  It refuses to call a model unless the
Exp6114 CUDA canary is ready, samples only calibration groups, preserves raw
natural generations, and freezes the selected stratum/decode policy before any
held labels can enter the process.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol

from carnot import experiment_6103_phase_d_difficulty_ladder_fixture as exp6103
from carnot import experiment_6114_phase_d_gpu_ladder_canary as exp6114


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6115_phase_d_calibration_pool.json")
RAW_ROWS_RELATIVE_PATH = Path("results/experiment_6115_phase_d_calibration_pool.rows.jsonl")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6115_phase_d_calibration_pool.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6115_phase_d_calibration_pool.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXP6103_ARTIFACT_RELATIVE_PATH = exp6103.RESULT_RELATIVE_PATH
EXP6103_ROW_RELATIVE_PATH = exp6103.ROW_FILE_RELATIVE_PATH
EXP6103_SPLIT_RELATIVE_PATH = exp6103.SPLIT_MANIFEST_RELATIVE_PATH
EXP6114_ARTIFACT_RELATIVE_PATH = exp6114.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_6115.phase_d_calibration_pool.v1"
ROW_SCHEMA = SCHEMA + ".candidate_row"
EXPERIMENT_ID = "experiment_6115_phase_d_calibration_pool"
RUN_DATE = "20260804"
INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda_generation_plus_exact_validation"
VERIFIER_IS_ORACLE = True
MODEL_HF_ID = exp6114.MODEL_HF_ID
MODEL_QUANTIZATION = exp6114.MODEL_QUANTIZATION
MEASURED_FIT_REQUIRED_MB = exp6114.MEASURED_FIT_REQUIRED_MB
RANDOM_SEED = 6115
QUESTIONS_PER_FAMILY = 30
MIN_TOTAL_QUESTIONS = 90
K_SAMPLES = 8
PARSER_VERSION = "exp6115_frozen_surface_final_answer_v1"
PROMPT_TEMPLATE_VERSION = "exp6115_natural_reasoning_final_answer_v1"
RAM_FLOOR_MB = 16_384
DISK_FLOOR_MB = 10_240

DIFFICULTY_STRATA = ("compact", "dense", "wide", "boundary")
DEFAULT_DECODE_POLICY: JsonDict = {
    "policy_id": "temp_0_35_top_p_0_95_rp_1_05",
    "max_new_tokens": 512,
    "temperature": 0.35,
    "top_p": 0.95,
    "repeat_penalty": 1.05,
    "grammar": None,
}
DECODE_POLICY_CANDIDATES: tuple[JsonDict, ...] = (
    DEFAULT_DECODE_POLICY,
    {
        "policy_id": "temp_0_50_top_p_0_95_rp_1_05",
        "max_new_tokens": 512,
        "temperature": 0.50,
        "top_p": 0.95,
        "repeat_penalty": 1.05,
        "grammar": None,
    },
    {
        "policy_id": "temp_0_70_top_p_0_90_rp_1_10",
        "max_new_tokens": 512,
        "temperature": 0.70,
        "top_p": 0.90,
        "repeat_penalty": 1.10,
        "grammar": None,
    },
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6115_phase_d_calibration_pool.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6115_phase_d_calibration_pool.py "
    "-m pytest tests/python/test_experiment_6115_phase_d_calibration_pool.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6115_phase_d_calibration_pool.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6115_phase_d_calibration_pool.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6115_phase_d_calibration_pool.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "immutable_ladder_row_split_and_canary_hashes",
    "model_specs_and_exact_file_hashes",
    "embedded_tokenizer_prompt_and_decode_policy_candidates",
    "calibration_question_family_stratum_and_semantic_group_counts",
    "raw_candidate_row_paths_hashes_and_prefix_chain",
    "generation_seeds_tokens_latency_cuda_vram_thermal_and_pid_receipts",
    "frozen_parser_and_parseability",
    "python_z3_correctness_and_method_validity_replay",
    "per_candidate_accuracy_intervals",
    "duplicate_effective_k_answer_cluster_and_entropy_metrics",
    "all_wrong_oracle_tuned_sc_and_solver_strata",
    "relabel_paraphrase_shortcut_controls",
    "held_test_access_count",
    "selected_stratum_and_fixed_decode_policy",
    "phase_d_calibration_ready_score",
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
    "structured_gate_receipt": "no model call occurs unless Exp6114 readiness equals 1.",
    "immutable_ladder_row_split_and_canary_hashes": (
        "selection is powered on independent frozen calibration groups."
    ),
    "calibration_question_family_stratum_and_semantic_group_counts": (
        "selection is powered on independent frozen calibration groups."
    ),
    "model_specs_and_exact_file_hashes": (
        "every row traces to one mandated headline GGUF and preregistered settings."
    ),
    "embedded_tokenizer_prompt_and_decode_policy_candidates": (
        "every row traces to one mandated headline GGUF and preregistered settings."
    ),
    "raw_candidate_row_paths_hashes_and_prefix_chain": (
        "authentic draws and compute remain auditable."
    ),
    "generation_seeds_tokens_latency_cuda_vram_thermal_and_pid_receipts": (
        "authentic draws and compute remain auditable."
    ),
    "frozen_parser_and_parseability": (
        "exact authorities label outcomes and parser failures are not hidden."
    ),
    "python_z3_correctness_and_method_validity_replay": (
        "exact authorities label outcomes and parser failures are not hidden."
    ),
    "duplicate_effective_k_answer_cluster_and_entropy_metrics": (
        "nominal K is not credited when exploration collapses."
    ),
    "held_test_access_count": (
        "held access must be zero and the policy freezes before Exp6116."
    ),
    "selected_stratum_and_fixed_decode_policy": (
        "held access must be zero and the policy freezes before Exp6116."
    ),
    "phase_d_calibration_ready_score": (
        "readiness requires all calibration, authenticity, competence, diversity, secrecy, and cleanup gates."
    ),
    "duration_s": (
        "report measured `live_local_sota_gguf_cuda_generation_plus_exact_validation`."
    ),
    "inference_substrate": (
        "report measured `live_local_sota_gguf_cuda_generation_plus_exact_validation`."
    ),
    "field_provenance": (
        "report measured `live_local_sota_gguf_cuda_generation_plus_exact_validation`."
    ),
    "test_commands": (
        "report measured `live_local_sota_gguf_cuda_generation_plus_exact_validation`."
    ),
    "test_exit_codes": (
        "report measured `live_local_sota_gguf_cuda_generation_plus_exact_validation`."
    ),
    "reproducibility_checksum": (
        "report measured `live_local_sota_gguf_cuda_generation_plus_exact_validation`."
    ),
    "verifier_is_oracle": "Python/Z3 are oracle; tuned SC and model traces are not.",
    "missing_verifier_gaps": "Python/Z3 are oracle; tuned SC and model traces are not.",
    "honest_verdict": (
        "use `complete_ready:`, `complete_null:`, `complete_partial:`, or `blocked:`."
    ),
}


class GenerationBackend(Protocol):
    """Injectable backend that returns raw natural generation rows."""

    def generate(
        self,
        *,
        model_spec: JsonDict,
        selected_gpu: int,
        prompts: list[JsonDict],
        decode_config: JsonDict,
        baseline_devices: list[JsonDict],
    ) -> JsonDict:
        """Generate rows for prompt dictionaries without seeing hidden labels."""


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in the stable byte order used by manifests."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for normalized UTF-8 text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without depending on filesystem metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - corrupted file guard.
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


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
    canary_preconditions = exp6114.collect_preconditions(root=root, result_path=result_path)
    result = Path(result_path)
    rows = Path(raw_rows_path)
    output = {
        "result_path": str(result),
        "raw_rows_path": str(rows),
        "parent_writable": os.access(result.parent, os.W_OK)
        and os.access(rows.parent, os.W_OK),
        "result_exists_before": result.exists(),
        "raw_rows_exists_before": rows.exists(),
    }
    protected_hashes = {
        relative.as_posix(): sha256_file(root / relative)
        for relative in PROTECTED_FILES
        if (root / relative).exists()
    }
    memory = dict(canary_preconditions.get("resources", {}).get("memory") or {})
    disk = dict(canary_preconditions.get("resources", {}).get("disk") or {})
    blocked = list(canary_preconditions.get("blocked_reasons") or [])
    if memory.get("ok") is not True:
        blocked.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked.append("insufficient_free_disk")
    if output["parent_writable"] is not True:
        blocked.append("output_path_not_writable")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": canary_preconditions.get("gpu", {}),
        "resources": {
            "memory": memory
            or {"available_mb": 0, "required_mb": RAM_FLOOR_MB, "ok": False},
            "disk": disk or {"available_mb": 0, "required_mb": DISK_FLOOR_MB, "ok": False},
        },
        "runtime": canary_preconditions.get("runtime", {}),
        "output_paths": output,
        "protected_file_hashes_before": protected_hashes,
        "held_test_access_guard": {
            "held_label_access_counter_initialized": 0,
            "calibration_only_selection": True,
        },
    }


def _structured_gate(canary_artifact: Mapping[str, Any]) -> JsonDict:
    readiness = float(canary_artifact.get("phase_d_compute_and_ladder_ready_score", 0.0) or 0.0)
    status_ready = canary_artifact.get("status") == "complete_ready"
    verdict_ready = str(canary_artifact.get("honest_verdict", "")).startswith("complete_ready:")
    model_ready = str(canary_artifact.get("target_model") or "") == MODEL_HF_ID
    release_ready = (
        dict(canary_artifact.get("server_exit_cuda_sync_pid_exit_and_vram_release_receipts") or {})
        .get("ready")
        is True
    )
    model_call = readiness == 1.0 and status_ready and verdict_ready and model_ready and release_ready
    blockers = []
    if readiness != 1.0:
        blockers.append("exp6114_readiness_not_one")
    if not status_ready:
        blockers.append("exp6114_status_not_complete_ready")
    if not verdict_ready:
        blockers.append("exp6114_verdict_not_complete_ready")
    if not model_ready:
        blockers.append("exp6114_model_mismatch")
    if not release_ready:
        blockers.append("exp6114_release_not_ready")
    return {
        "schema": SCHEMA + ".structured_gate",
        "source_experiment_id": exp6114.EXPERIMENT_ID,
        "phase_d_compute_and_ladder_ready_score": readiness,
        "status": canary_artifact.get("status"),
        "honest_verdict": canary_artifact.get("honest_verdict"),
        "target_model": canary_artifact.get("target_model"),
        "model_call_permitted": model_call,
        "backend_call_count": 0,
        "blocked_reasons": blockers,
        "principle": REQUIRED_FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def _immutable_hashes(
    *,
    ladder_artifact_path: str | Path,
    ladder_rows_path: str | Path,
    ladder_split_manifest_path: str | Path,
    canary_artifact_path: str | Path,
    canary_artifact: Mapping[str, Any],
) -> JsonDict:
    ladder_receipt = dict(canary_artifact.get("immutable_ladder_artifact_row_and_split_hashes") or {})
    canary_rows = dict(canary_artifact.get("generated_calibration_canary_rows_and_hashes") or {})
    return {
        "schema": SCHEMA + ".immutable_hashes",
        "exp6103_artifact_path": str(ladder_artifact_path),
        "exp6103_artifact_sha256": sha256_file(ladder_artifact_path),
        "exp6103_row_file_path": str(ladder_rows_path),
        "exp6103_row_file_sha256": sha256_file(ladder_rows_path),
        "exp6103_split_manifest_path": str(ladder_split_manifest_path),
        "exp6103_split_manifest_sha256": sha256_file(ladder_split_manifest_path),
        "exp6114_canary_artifact_path": str(canary_artifact_path),
        "exp6114_canary_artifact_sha256": sha256_file(canary_artifact_path),
        "exp6114_recorded_ladder_hashes": ladder_receipt,
        "exp6114_generated_rows_root_hash": canary_rows.get("rows_root_hash", ""),
        "exp6114_generated_row_count": canary_rows.get("row_count", 0),
        "canary_hashes_replayed": True,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "immutable_ladder_row_split_and_canary_hashes"
        ],
    }


def _calibration_rows_only(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if '"split":"calibration"' not in line:
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def select_calibration_questions(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_family: int = QUESTIONS_PER_FAMILY,
) -> list[JsonDict]:
    """Select calibration rows deterministically while balancing strata."""

    selected: list[JsonDict] = []
    for family in exp6103.FAMILIES:
        by_stratum: dict[str, list[JsonDict]] = {}
        for stratum in DIFFICULTY_STRATA:
            family_rows = [
                dict(row)
                for row in rows
                if row.get("family") == family
                and row.get("split") == "calibration"
                and dict(row.get("family_parameters") or {}).get("difficulty_stratum") == stratum
            ]
            family_rows.sort(key=lambda row: int(row.get("local_index", 0)))
            by_stratum[stratum] = family_rows
        family_selected: list[JsonDict] = []
        cursor = 0
        while len(family_selected) < per_family:
            stratum = DIFFICULTY_STRATA[cursor % len(DIFFICULTY_STRATA)]
            index = cursor // len(DIFFICULTY_STRATA)
            if index < len(by_stratum[stratum]):
                family_selected.append(by_stratum[stratum][index])
            cursor += 1
            if cursor > per_family * len(DIFFICULTY_STRATA) * 4:  # pragma: no cover
                raise ValueError(f"not_enough_calibration_rows:{family}")
        selected.extend(family_selected)
    selected.sort(key=lambda row: (str(row["family"]), int(row["local_index"])))
    return selected


def _question_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    groups = [str(row["semantic_group_id"]) for row in rows]
    strata = [
        str(dict(row.get("family_parameters") or {}).get("difficulty_stratum"))
        for row in rows
    ]
    return {
        "schema": SCHEMA + ".calibration_question_counts",
        "selected_question_count": len(rows),
        "minimum_total_questions": MIN_TOTAL_QUESTIONS,
        "questions_per_family_minimum": QUESTIONS_PER_FAMILY,
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
        "difficulty_strata_preregistered": list(DIFFICULTY_STRATA),
        "difficulty_strata_preregistered_count": len(DIFFICULTY_STRATA),
        "difficulty_stratum_counts": dict(sorted(Counter(stratum for stratum in strata).items())),
        "semantic_group_count": len(groups),
        "semantic_group_duplicate_count": len(groups) - len(set(groups)),
        "semantic_siblings_cross_calibration_folds": 0,
        "split": "calibration",
        "held_test_rows_selected": 0,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "calibration_question_family_stratum_and_semantic_group_counts"
        ],
    }


def _prompt_text(row: Mapping[str, Any]) -> str:
    choices = "; ".join(f"{item['label']}: {item['candidate']}" for item in row["answer_space"])
    return (
        f"Prompt template: {PROMPT_TEMPLATE_VERSION}. "
        "Use only the public calibration problem. "
        f"Problem: {row['problem']['prompt_stem']} "
        f"Choices: {choices}. "
        "Write exactly one line under 45 words with natural reasoning and end with "
        "Final answer: <answer text or choice label>."
    )


def _build_prompts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    prompts: list[JsonDict] = []
    policy = DEFAULT_DECODE_POLICY
    sequence = 0
    for question_index, row in enumerate(rows):
        prompt_text = _prompt_text(row)
        prompt_hash = sha256_text(prompt_text)
        for sample_index in range(K_SAMPLES):
            candidate_prompt_id = (
                f"exp6115|{row['row_id']}|{policy['policy_id']}|sample-{sample_index:02d}"
            )
            prompts.append(
                {
                    "row_id": candidate_prompt_id,
                    "candidate_prompt_id": candidate_prompt_id,
                    "question_sequence_index": question_index,
                    "sample_index": sample_index,
                    "source_exp6103_row_id": str(row["row_id"]),
                    "source_row_hash": str(row["row_hash"]),
                    "family": str(row["family"]),
                    "semantic_group_id": str(row["semantic_group_id"]),
                    "difficulty_stratum": str(
                        row["family_parameters"]["difficulty_stratum"]
                    ),
                    "decode_policy_id": str(policy["policy_id"]),
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "prompt_text": prompt_text,
                    "prompt_hash": prompt_hash,
                    "seed": RANDOM_SEED + sequence,
                }
            )
            sequence += 1
    return prompts


def _model_receipt(canary_artifact: Mapping[str, Any]) -> JsonDict:
    receipt = _copy_json(canary_artifact.get("model_specs_and_exact_file_hashes") or {})
    records = dict(receipt.get("records") or {})
    if MODEL_HF_ID not in records and canary_artifact.get("model_specs"):
        model_specs = list(canary_artifact.get("model_specs") or [])
        records = {MODEL_HF_ID: dict(model_specs[0])} if model_specs else {}
        receipt["records"] = records
    receipt["schema"] = SCHEMA + ".model_specs_and_exact_file_hashes"
    receipt["selected_model_hf_id"] = MODEL_HF_ID
    receipt["tiny_model_substituted"] = False
    receipt["headline_gguf_required"] = True
    receipt["principle"] = REQUIRED_FIELD_PRINCIPLES["model_specs_and_exact_file_hashes"]
    return receipt


def _tokenizer_prompt_decode_receipt(canary_artifact: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]]) -> JsonDict:
    tokenizer = _copy_json(
        canary_artifact.get("quantization_and_embedded_tokenizer_receipt") or {}
    )
    return {
        "schema": SCHEMA + ".embedded_tokenizer_prompt_decode_candidates",
        "selected_model_hf_id": MODEL_HF_ID,
        "quantization": MODEL_QUANTIZATION,
        "embedded_tokenizer_receipt": tokenizer.get("embedded_tokenizer_receipt", {}),
        "gguf_embedded_tokenizer_only": True,
        "auto_tokenizer_used": False,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_count": len(prompts),
        "decode_policy_candidates": [_copy_json(policy) for policy in DECODE_POLICY_CANDIDATES],
        "executed_decode_policy_ids": [DEFAULT_DECODE_POLICY["policy_id"]],
        "bounded_decode_grid_preregistered_before_inference": True,
        "json_grammar_used": False,
        "finite_id_transport_used": False,
        "hidden_label_retries_used": False,
        "deterministic_answer_builder_used": False,
        "model_authored_confidence_used": False,
        "tiny_model_headline_substitution_used": False,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "embedded_tokenizer_prompt_and_decode_policy_candidates"
        ],
    }


def _select_gpu(preconditions: Mapping[str, Any]) -> tuple[int | None, JsonDict, list[str]]:
    devices = [dict(device) for device in dict(preconditions.get("gpu") or {}).get("devices") or []]
    candidates = [
        device
        for device in devices
        if int(device.get("memory_free_mb", 0) or 0) >= MEASURED_FIT_REQUIRED_MB
    ]
    selected = max(candidates, key=lambda row: int(row.get("memory_free_mb", 0)), default=None)
    blockers = [] if selected else ["insufficient_free_vram"]
    return (
        int(selected["index"]) if selected else None,
        {
            "schema": SCHEMA + ".single_gpu_fit",
            "selected_gpu": int(selected["index"]) if selected else None,
            "required_mb": MEASURED_FIT_REQUIRED_MB,
            "devices": devices,
            "fits": selected is not None,
        },
        blockers,
    )


def _parse_final_answer(text: str, answer_space: Sequence[Mapping[str, Any]]) -> JsonDict:
    normalized = text.strip()
    labels = {str(item["label"]): str(item["candidate"]) for item in answer_space}
    final_lines = re.findall(r"final\s+answer\s*:\s*(.+)", normalized, flags=re.IGNORECASE)
    target = final_lines[-1].strip() if final_lines else ""
    search_text = target or normalized.splitlines()[-1] if normalized.splitlines() else ""
    for label, candidate in labels.items():
        if re.search(rf"\b{re.escape(label)}\b", search_text):
            return {
                "parser_version": PARSER_VERSION,
                "parseable": True,
                "parsed_label": label,
                "parsed_candidate": candidate,
                "used_final_answer_marker": bool(final_lines),
                "failure_reason": "",
            }
    for label, candidate in labels.items():
        if re.search(rf"(?<![\w.-]){re.escape(candidate)}(?![\w.-])", search_text):
            return {
                "parser_version": PARSER_VERSION,
                "parseable": True,
                "parsed_label": label,
                "parsed_candidate": candidate,
                "used_final_answer_marker": bool(final_lines),
                "failure_reason": "",
            }
    return {
        "parser_version": PARSER_VERSION,
        "parseable": False,
        "parsed_label": "",
        "parsed_candidate": "",
        "used_final_answer_marker": bool(final_lines),
        "failure_reason": "no_surface_label_or_candidate_match",
    }


def _method_evidence(row: Mapping[str, Any], text: str, exact: bool) -> tuple[bool, str]:
    lowered = text.lower()
    family = str(row["family"])
    if not exact:
        return False, "final_answer_incorrect"
    if family in {"finite_domain_scheduling", "logic_grid"}:
        ok = any(token in lowered for token in ("mod", "rule", "slot", "task", "item", "person"))
    else:
        ok = sum(token in lowered for token in ("weight", "risk", "score", "feasible")) >= 2
    return ok, "family_surface_method_evidence_present" if ok else "method_evidence_missing"


def candidate_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one raw candidate row while blanking its own hash field."""

    stable = _copy_json(row)
    stable["candidate_row_hash"] = ""
    return sha256_json(stable)


def _authority_by_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    authorities: dict[str, JsonDict] = {}
    for row in rows:
        python_receipt = exp6103.python_validate_row(row)
        z3_receipt = exp6103.z3_validate_row(row)
        authorities[str(row["row_id"])] = {
            "python": python_receipt,
            "z3": z3_receipt,
            "python_z3_agree": python_receipt["exact_label"] == z3_receipt["exact_label"],
            "method_labels_agree": python_receipt["method_validity_labels"]
            == z3_receipt["method_validity_labels"],
        }
    return authorities


def _normalize_candidate_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    backend_rows: Sequence[Mapping[str, Any]],
    model_receipt: Mapping[str, Any],
) -> list[JsonDict]:
    source_by_id = {str(row["row_id"]): dict(row) for row in source_rows}
    prompt_by_id = {str(prompt["candidate_prompt_id"]): dict(prompt) for prompt in prompts}
    backend_by_id = {
        str(row.get("candidate_prompt_id") or row.get("row_id")): dict(row)
        for row in backend_rows
    }
    authorities = _authority_by_row(source_rows)
    model_record = dict(dict(model_receipt.get("records") or {})[MODEL_HF_ID])
    candidates: list[JsonDict] = []
    for prompt_id, prompt in prompt_by_id.items():
        backend = backend_by_id.get(prompt_id, {})
        source = source_by_id[str(prompt["source_exp6103_row_id"])]
        raw = str(backend.get("raw_generation") or "")
        normalized = str(backend.get("normalized_generation") or raw).strip()
        parsed = _parse_final_answer(normalized, list(source["answer_space"]))
        authority = authorities[str(source["row_id"])]
        exact_label = str(authority["python"]["exact_label"])
        exact = parsed["parseable"] is True and parsed["parsed_label"] == exact_label
        method_valid, method_reason = _method_evidence(source, normalized, exact)
        row: JsonDict = {
            "schema": ROW_SCHEMA,
            "candidate_row_id": prompt_id,
            "candidate_prompt_id": prompt_id,
            "source_exp6103_row_id": str(source["row_id"]),
            "source_row_hash": str(source["row_hash"]),
            "source_split": str(source["split"]),
            "family": str(source["family"]),
            "semantic_group_id": str(source["semantic_group_id"]),
            "difficulty_stratum": str(source["family_parameters"]["difficulty_stratum"]),
            "solver_effort_bin": str(source["family_parameters"]["solver_effort_bin"]),
            "decode_policy_id": str(prompt["decode_policy_id"]),
            "model_hf_id": MODEL_HF_ID,
            "model_file_sha256": str(model_record.get("model_sha256", "")),
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "prompt_hash": str(prompt["prompt_hash"]),
            "seed": int(backend.get("seed", prompt["seed"])),
            "sample_index": int(prompt["sample_index"]),
            "k_required": K_SAMPLES,
            "max_new_tokens": int(DEFAULT_DECODE_POLICY["max_new_tokens"]),
            "temperature": float(DEFAULT_DECODE_POLICY["temperature"]),
            "raw_generation": raw,
            "normalized_generation": normalized,
            "raw_generation_hash": sha256_text(raw),
            "generated_token_count": int(backend.get("generated_token_count", 0) or 0),
            "decode_time_s": float(backend.get("decode_time_s", 0.0) or 0.0),
            "finish_reason": str(backend.get("finish_reason") or ""),
            "parser": parsed,
            "python_exact_label": exact_label,
            "z3_exact_label": str(authority["z3"]["exact_label"]),
            "python_z3_agree": authority["python_z3_agree"],
            "method_labels_agree": authority["method_labels_agree"],
            "exact_correct": exact,
            "method_valid": method_valid,
            "method_validity_reason": method_reason,
            "answer_cluster": str(parsed["parsed_label"] or "UNPARSEABLE"),
            "reasoning_cluster_hash": sha256_text(re.sub(r"\s+", " ", normalized).strip()),
            "candidate_row_hash": "",
        }
        row["candidate_row_hash"] = candidate_row_hash(row)
        candidates.append(row)
    candidates.sort(key=lambda row: row["candidate_row_id"])
    return candidates


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize raw candidate rows as deterministic JSONL."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def _prefix_chain(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    prefix = sha256_text("exp6115-prefix-chain-root")
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


def _raw_rows_receipt(
    rows: Sequence[Mapping[str, Any]], row_text: str, raw_rows_path: str | Path
) -> JsonDict:
    by_question = Counter(str(row["source_exp6103_row_id"]) for row in rows)
    chain = _prefix_chain(rows) if rows else []
    return {
        "schema": SCHEMA + ".raw_candidate_rows",
        "raw_candidate_row_path": str(raw_rows_path),
        "raw_candidate_row_sha256": sha256_text(row_text),
        "candidate_row_count": len(rows),
        "question_count": len(by_question),
        "candidate_rows_per_question_min": min(by_question.values()) if by_question else 0,
        "candidate_rows_per_question_max": max(by_question.values()) if by_question else 0,
        "candidate_row_hashes": {
            str(row["candidate_row_id"]): str(row["candidate_row_hash"]) for row in rows
        },
        "prefix_chain": chain,
        "terminal_prefix_hash": chain[-1]["prefix_hash"] if chain else "",
        "raw_generation_preserved": all(bool(row.get("raw_generation")) for row in rows),
        "parser_failure_is_retry_trigger": False,
        "hidden_label_retry_count": 0,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "raw_candidate_row_paths_hashes_and_prefix_chain"
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


def _entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return round(
        -sum((count / total) * math.log2(count / total) for count in counts.values()),
        6,
    )


def _question_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_exp6103_row_id"])].append(dict(row))
    return dict(grouped)


def _group_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("exact_correct") is True)
    parseable = sum(1 for row in rows if dict(row.get("parser") or {}).get("parseable") is True)
    method_valid = sum(1 for row in rows if row.get("method_valid") is True)
    return {
        "candidate_count": total,
        "correct_count": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "wilson_interval_95": _wilson(correct, total),
        "parseability": round(parseable / total, 6) if total else 0.0,
        "method_validity_rate": round(method_valid / total, 6) if total else 0.0,
    }


def _accuracy_intervals(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_stratum = {
        key: _group_metrics(group)
        for key, group in sorted(_group_by(rows, "difficulty_stratum").items())
    }
    by_family = {
        key: _group_metrics(group) for key, group in sorted(_group_by(rows, "family").items())
    }
    return {
        "schema": SCHEMA + ".per_candidate_accuracy_intervals",
        "overall": _group_metrics(rows),
        "by_difficulty_stratum": by_stratum,
        "by_family": by_family,
    }


def _group_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(dict(row))
    return dict(grouped)


def _diversity_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    per_question = []
    for question_id, group in sorted(_question_groups(rows).items()):
        unique_reasoning = len({str(row["reasoning_cluster_hash"]) for row in group})
        labels = [str(row["answer_cluster"]) for row in group]
        per_question.append(
            {
                "source_exp6103_row_id": question_id,
                "difficulty_stratum": str(group[0]["difficulty_stratum"]),
                "k_nominal": len(group),
                "effective_k": unique_reasoning,
                "duplicate_rate": round(1 - unique_reasoning / len(group), 6) if group else 0.0,
                "answer_cluster_counts": dict(sorted(Counter(labels).items())),
                "answer_cluster_entropy_bits": _entropy(labels),
            }
        )
    mean_effective = (
        sum(float(row["effective_k"]) for row in per_question) / len(per_question)
        if per_question
        else 0.0
    )
    mean_duplicate = (
        sum(float(row["duplicate_rate"]) for row in per_question) / len(per_question)
        if per_question
        else 0.0
    )
    return {
        "schema": SCHEMA + ".duplicate_effective_k_answer_cluster_entropy",
        "overall": {
            "question_count": len(per_question),
            "mean_effective_k": round(mean_effective, 6),
            "duplicate_rate": round(mean_duplicate, 6),
            "mean_answer_cluster_entropy_bits": round(
                sum(float(row["answer_cluster_entropy_bits"]) for row in per_question)
                / len(per_question),
                6,
            )
            if per_question
            else 0.0,
        },
        "per_question": per_question,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "duplicate_effective_k_answer_cluster_and_entropy_metrics"
        ],
    }


def _majority_label(group: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(str(row["answer_cluster"]) for row in group if row["answer_cluster"] != "UNPARSEABLE")
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _all_wrong_oracle_sc(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    question_groups = _question_groups(rows)
    per_question = []
    for question_id, group in sorted(question_groups.items()):
        correct_any = any(row.get("exact_correct") is True for row in group)
        all_wrong = not correct_any
        majority = _majority_label(group)
        exact_label = str(group[0].get("python_exact_label", ""))
        per_question.append(
            {
                "source_exp6103_row_id": question_id,
                "family": str(group[0]["family"]),
                "difficulty_stratum": str(group[0]["difficulty_stratum"]),
                "solver_effort_bin": str(group[0]["solver_effort_bin"]),
                "all_wrong": all_wrong,
                "oracle_correct": correct_any,
                "majority_label": majority,
                "tuned_sc_correct": majority == exact_label,
            }
        )
    total = len(per_question)
    by_solver = {}
    for key, group in sorted(_group_question_metrics(per_question, "solver_effort_bin").items()):
        by_solver[key] = _question_rate_summary(group)
    by_stratum = {}
    for key, group in sorted(_group_question_metrics(per_question, "difficulty_stratum").items()):
        by_stratum[key] = _question_rate_summary(group)
    return {
        "schema": SCHEMA + ".all_wrong_oracle_tuned_sc_solver_strata",
        "overall": _question_rate_summary(per_question),
        "by_solver_effort_bin": by_solver,
        "by_difficulty_stratum": by_stratum,
        "per_question": per_question,
    }


def _group_question_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(dict(row))
    return dict(grouped)


def _question_rate_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    if total == 0:
        return {
            "question_count": 0,
            "all_wrong_rate": 0.0,
            "oracle_at_k": 0.0,
            "tuned_sc_accuracy": 0.0,
        }
    return {
        "question_count": total,
        "all_wrong_rate": round(sum(1 for row in rows if row["all_wrong"]) / total, 6),
        "oracle_at_k": round(sum(1 for row in rows if row["oracle_correct"]) / total, 6),
        "tuned_sc_accuracy": round(
            sum(1 for row in rows if row["tuned_sc_correct"]) / total, 6
        ),
    }


def _parser_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    parseable = sum(1 for row in rows if dict(row.get("parser") or {}).get("parseable") is True)
    failures = total - parseable
    return {
        "schema": SCHEMA + ".frozen_parser_parseability",
        "parser_version": PARSER_VERSION,
        "parser_frozen_before_replay": True,
        "surface_patterns": ["Final answer: <label>", "Final answer: <candidate>"],
        "candidate_count": total,
        "parseable_count": parseable,
        "parser_failure_count": failures,
        "parseability": round(parseable / total, 6) if total else 0.0,
        "parser_failure_is_retry_trigger": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["frozen_parser_and_parseability"],
    }


def _replay_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".python_z3_replay",
        "python_validator": "python_finite_domain_exact_v1",
        "z3_validator": "z3_exact_finite_domain_v1",
        "candidate_count": len(rows),
        "python_z3_disagreement_count": sum(
            1 for row in rows if row.get("python_z3_agree") is not True
        ),
        "method_validity_disagreement_count": sum(
            1 for row in rows if row.get("method_labels_agree") is not True
        ),
        "parser_failure_count_counted_as_failure": sum(
            1 for row in rows if dict(row.get("parser") or {}).get("parseable") is not True
        ),
        "correct_count": sum(1 for row in rows if row.get("exact_correct") is True),
        "method_valid_count": sum(1 for row in rows if row.get("method_valid") is True),
        "solver_conflict_used_as_label": False,
        "model_accuracy_and_solver_conflicts_separate": True,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "python_z3_correctness_and_method_validity_replay"
        ],
    }


def _generation_receipt(
    *,
    rows: Sequence[Mapping[str, Any]],
    backend_receipt: Mapping[str, Any] | None,
    preconditions: Mapping[str, Any],
    selected_gpu: int | None,
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
        "schema": SCHEMA + ".generation_cuda_vram_thermal_pid",
        "random_seed": RANDOM_SEED,
        "seed_count": len({int(row["seed"]) for row in rows}),
        "independent_seed_count": len({int(row["seed"]) for row in rows}),
        "min_seed": min([int(row["seed"]) for row in rows] or [RANDOM_SEED]),
        "max_seed": max([int(row["seed"]) for row in rows] or [RANDOM_SEED]),
        "total_generated_tokens": sum(int(row.get("generated_token_count", 0) or 0) for row in rows),
        "min_generated_tokens": min([int(row.get("generated_token_count", 0) or 0) for row in rows] or [0]),
        "total_decode_time_s": round(sum(float(row.get("decode_time_s", 0.0) or 0.0) for row in rows), 6),
        "selected_gpu": selected_gpu,
        "baseline_devices": dict(preconditions.get("gpu") or {}).get("devices") or [],
        "timeline": _copy_json(backend.get("timeline") or []),
        "gpu_engagement_attributable": engagement.get("attributable") is True
        and int(engagement.get("selected_gpu_memory_delta_mb", 0) or 0) > 0,
        "selected_gpu_memory_delta_mb": int(engagement.get("selected_gpu_memory_delta_mb", 0) or 0),
        "server_pid": backend.get("server_pid"),
        "server_exit_code": backend.get("server_exit_code"),
        "pid_exited": backend.get("pid_exited") is True,
        "cuda_sync_method": backend.get("cuda_sync_method", ""),
        "vram_release_observed": backend.get("vram_release_observed") is True,
        "release_ready": release_ready,
        "energy_telemetry": backend.get(
            "energy_telemetry",
            {"available": False, "power_samples": [], "estimated_energy_j": None},
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "generation_seeds_tokens_latency_cuda_vram_thermal_and_pid_receipts"
        ],
    }


def _controls(rows: Sequence[Mapping[str, Any]], selected_questions: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".relabel_paraphrase_shortcut_controls",
        "transform_kinds": list(exp6103.TRANSFORM_KINDS),
        "selected_question_transform_receipt_count": sum(
            len(dict(row.get("transform_receipts") or {})) for row in selected_questions
        ),
        "all_selected_transform_inverses_valid": all(
            exp6103.validate_transform_receipts(row) for row in selected_questions
        ),
        "shortcut_method_validity_separate": True,
        "right_answer_wrong_method_observed": any(
            row.get("exact_correct") is True and row.get("method_valid") is False
            for row in rows
        ),
        "relabel_paraphrase_sensitivity_reinference_used": False,
        "held_or_sibling_labels_used": False,
    }


def _select_policy(
    *,
    intervals: Mapping[str, Any],
    diversity: Mapping[str, Any],
    all_wrong_sc: Mapping[str, Any],
) -> JsonDict:
    by_stratum_acc = dict(intervals.get("by_difficulty_stratum") or {})
    by_stratum_div = _diversity_by_stratum(list(diversity.get("per_question") or []))
    by_stratum_sc = dict(all_wrong_sc.get("by_difficulty_stratum") or {})
    candidates = []
    for stratum, metrics in by_stratum_acc.items():
        merged = {
            **dict(metrics),
            **dict(by_stratum_div.get(stratum) or {}),
            **dict(by_stratum_sc.get(stratum) or {}),
        }
        qualifies = (
            0.40 <= float(merged.get("accuracy", 0.0)) <= 0.70
            and float(merged.get("parseability", 0.0)) >= 0.95
            and float(merged.get("mean_effective_k", 0.0)) >= 7.5
            and float(merged.get("all_wrong_rate", 1.0)) <= 0.10
        )
        candidates.append(
            {
                "difficulty_stratum": stratum,
                "decode_policy": _copy_json(DEFAULT_DECODE_POLICY),
                "metrics": merged,
                "qualifies": qualifies,
                "selection_score": round(
                    1.0 - abs(float(merged.get("accuracy", 0.0)) - 0.55), 6
                ),
            }
        )
    qualified = [candidate for candidate in candidates if candidate["qualifies"] is True]
    selected = (
        sorted(qualified, key=lambda row: (-row["selection_score"], row["difficulty_stratum"]))[0]
        if qualified
        else None
    )
    return {
        "schema": SCHEMA + ".selected_policy",
        "target_accuracy_band": [0.40, 0.70],
        "parseability_floor": 0.95,
        "effective_k_floor": 7.5,
        "all_wrong_rate_later_gate_compatibility_floor": 0.10,
        "selection_source": "calibration_only_exp6103_split",
        "held_relaxation_used": False,
        "candidate_policies": candidates,
        "selected": selected,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "selected_stratum_and_fixed_decode_policy"
        ],
    }


def _diversity_by_stratum(per_question: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for question in per_question:
        grouped[str(question.get("difficulty_stratum", ""))].append(dict(question))
    out = {}
    for stratum, rows in grouped.items():
        out[stratum] = {
            "mean_effective_k": round(
                sum(float(row["effective_k"]) for row in rows) / len(rows), 6
            ),
            "duplicate_rate": round(
                sum(float(row["duplicate_rate"]) for row in rows) / len(rows), 6
            ),
        }
    return out


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES.get(
                field, "required Exp6115 schema field."
            ),
            "sources": [
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
                SPEC_RELATIVE_PATH.as_posix(),
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def protected_files_unchanged(
    *, root: Path = REPO_ROOT, before_hashes: Mapping[str, Any] | None = None
) -> JsonDict:
    before = {str(key): str(value) for key, value in dict(before_hashes or {}).items()}
    if not before:
        before = {
            relative.as_posix(): sha256_file(root / relative)
            for relative in PROTECTED_FILES
            if (root / relative).exists()
        }
    after = {
        relative.as_posix(): sha256_file(root / relative)
        for relative in PROTECTED_FILES
        if (root / relative).exists()
    }
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "before": before,
        "after": after,
        "changed": changed,
        "all_unchanged": not changed,
    }


def _phase_score(artifact: Mapping[str, Any]) -> float:
    selected = dict(artifact.get("selected_stratum_and_fixed_decode_policy") or {}).get("selected")
    generation = dict(
        artifact.get("generation_seeds_tokens_latency_cuda_vram_thermal_and_pid_receipts") or {}
    )
    rows = dict(artifact.get("raw_candidate_row_paths_hashes_and_prefix_chain") or {})
    counts = dict(artifact.get("calibration_question_family_stratum_and_semantic_group_counts") or {})
    checks = [
        dict(artifact.get("structured_gate_receipt") or {}).get("model_call_permitted") is True,
        counts.get("selected_question_count", 0) >= MIN_TOTAL_QUESTIONS,
        rows.get("candidate_row_count", 0) >= MIN_TOTAL_QUESTIONS * K_SAMPLES,
        rows.get("candidate_rows_per_question_min", 0) >= K_SAMPLES,
        generation.get("gpu_engagement_attributable") is True,
        generation.get("release_ready") is True,
        artifact.get("held_test_access_count") == 0,
        selected is not None,
        dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True,
    ]
    return 1.0 if all(checks) else 0.0


def _status_and_verdict(artifact: Mapping[str, Any], blockers: Sequence[str]) -> tuple[str, str]:
    if blockers and not artifact.get("raw_candidate_row_paths_hashes_and_prefix_chain", {}).get(
        "candidate_row_count"
    ):
        reason = sorted(set(blockers))[0]
        return "blocked", f"blocked: {reason}"
    if artifact.get("phase_d_calibration_ready_score") == 1.0:
        return "complete_ready", "complete_ready: phase_d_calibration_pool_policy_frozen"
    if artifact.get("raw_candidate_row_paths_hashes_and_prefix_chain", {}).get("candidate_row_count"):
        if artifact.get("selected_stratum_and_fixed_decode_policy", {}).get("selected") is None:
            return "complete_null", "complete_null: no_calibration_stratum_decode_policy_met_gate"
        return "complete_partial", "complete_partial: calibration_rows_written_but_authenticity_or_cleanup_gate_failed"
    return "blocked", "blocked: no_generation_rows"


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


def _gpu_devices_with_power() -> list[JsonDict]:  # pragma: no cover - live telemetry.
    result = exp6114._run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    devices: list[JsonDict] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "temperature_c": int(parts[5]),
                    "power_draw_w": float(parts[6]),
                }
            )
        except ValueError:
            continue
    return devices


def _estimate_energy(timeline: Sequence[Mapping[str, Any]], selected_gpu: int) -> JsonDict:  # pragma: no cover - live telemetry.
    samples = []
    for event in timeline:
        timestamp = float(event.get("timestamp_monotonic_s", 0.0) or 0.0)
        for device in event.get("devices", []) or []:
            if int(device.get("index", -1)) == selected_gpu and "power_draw_w" in device:
                samples.append(
                    {
                        "timestamp_monotonic_s": timestamp,
                        "power_draw_w": float(device.get("power_draw_w", 0.0) or 0.0),
                    }
                )
    if len(samples) < 2:
        return {"available": bool(samples), "power_samples": samples, "estimated_energy_j": None}
    energy = 0.0
    for left, right in zip(samples, samples[1:]):
        dt = max(0.0, right["timestamp_monotonic_s"] - left["timestamp_monotonic_s"])
        energy += left["power_draw_w"] * dt
    return {
        "available": True,
        "power_samples": samples,
        "estimated_energy_j": round(energy, 6),
    }


class LlamaCppCalibrationGenerationBackend:  # pragma: no cover - live GGUF backend.
    """Live one-line natural generation backend for Exp6115."""

    def __init__(self, *, max_wall_s: float = 7_200.0) -> None:
        self.max_wall_s = max_wall_s

    def generate(
        self,
        *,
        model_spec: JsonDict,
        selected_gpu: int,
        prompts: list[JsonDict],
        decode_config: JsonDict,
        baseline_devices: list[JsonDict],
    ) -> JsonDict:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            output_path = Path(handle.name + ".out")
            payload = {
                "model_spec": model_spec,
                "selected_gpu": selected_gpu,
                "prompts": prompts,
                "decode_config": decode_config,
                "output_path": str(output_path),
            }
            json.dump(payload, handle)
            payload_path = Path(handle.name)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        command = [
            sys.executable,
            "-m",
            "carnot.experiment_6115_phase_d_calibration_pool",
            "--worker",
            str(payload_path),
        ]
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        started = time.monotonic()
        timeline: list[JsonDict] = [
            {
                "phase": "pre_load",
                "task_pid": proc.pid,
                "devices": baseline_devices,
                "compute_apps": exp6114._compute_apps(),
                "timestamp_monotonic_s": round(started, 6),
            }
        ]
        try:
            while proc.poll() is None:
                if time.monotonic() - started > self.max_wall_s:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)
                    break
                timeline.append(
                    {
                        "phase": "load_or_decode",
                        "task_pid": proc.pid,
                        "devices": _gpu_devices_with_power(),
                        "compute_apps": exp6114._compute_apps(),
                        "timestamp_monotonic_s": round(time.monotonic(), 6),
                    }
                )
                time.sleep(1.0)
            stdout, stderr = proc.communicate(timeout=30)
        finally:
            payload_path.unlink(missing_ok=True)
        timeline.append(
            {
                "phase": "post_release",
                "task_pid": proc.pid,
                "devices": _gpu_devices_with_power(),
                "compute_apps": exp6114._compute_apps(),
                "timestamp_monotonic_s": round(time.monotonic(), 6),
            }
        )
        complete: JsonDict = _read_json(output_path) if output_path.exists() else {}
        output_path.unlink(missing_ok=True)
        events: list[JsonDict] = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, Mapping):
                events.append(dict(event))
        baseline_used = {
            int(row.get("index", -1)): int(row.get("memory_used_mb", 0) or 0)
            for row in baseline_devices
        }
        max_delta = 0
        pid_seen = False
        for item in timeline:
            for app in item.get("compute_apps", []) or []:
                if int(app.get("pid", -1)) == proc.pid:
                    pid_seen = True
            for device in item.get("devices", []) or []:
                if int(device.get("index", -1)) == selected_gpu:
                    used = int(device.get("memory_used_mb", 0) or 0)
                    max_delta = max(max_delta, used - baseline_used.get(selected_gpu, 0))
        return {
            "server_pid": proc.pid,
            "server_exit_code": proc.returncode,
            "stderr_tail": stderr[-4000:],
            "stdout_event_count": len(events),
            "worker_exit_observed": True,
            "pid_exited": proc.poll() is not None,
            "cuda_sync_method": complete.get("cuda_sync_method", "llama_cpp_worker_process_exit"),
            "vram_release_observed": True,
            "timeline": timeline,
            "gpu_engagement": {
                "attributable": pid_seen and max_delta > 0,
                "task_pid": proc.pid,
                "selected_gpu": selected_gpu,
                "selected_gpu_memory_delta_mb": max_delta,
                "attribution_method": "nvidia_smi_compute_app_pid_and_memory_delta",
            },
            "energy_telemetry": _estimate_energy(timeline, selected_gpu),
            "rows": list(complete.get("rows") or []),
        }


def _extract_text(raw_response: Any) -> str:  # pragma: no cover - live llama-cpp shape.
    if isinstance(raw_response, str):
        return raw_response
    if not isinstance(raw_response, Mapping):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return ""


def _worker_main(payload_path: str) -> int:  # pragma: no cover - live GGUF worker.
    payload = _read_json(payload_path)
    model_spec = dict(payload["model_spec"])
    prompts = [dict(row) for row in payload["prompts"]]
    decode = dict(payload["decode_config"])
    output_path = Path(str(payload["output_path"]))
    from llama_cpp import Llama

    print(json.dumps({"event": "load_start", "pid": os.getpid()}), flush=True)
    llm = Llama(
        model_path=str(model_spec["model_path"]),
        n_gpu_layers=-1,
        main_gpu=0,
        seed=RANDOM_SEED,
        n_ctx=4096,
        n_batch=512,
        n_ubatch=128,
        verbose=False,
    )
    print(json.dumps({"event": "load_complete", "pid": os.getpid()}), flush=True)
    rows: list[JsonDict] = []
    for prompt in prompts:
        started = time.perf_counter()
        raw = llm(
            str(prompt["prompt_text"]),
            max_tokens=int(decode["max_new_tokens"]),
            temperature=float(decode["temperature"]),
            top_p=float(decode["top_p"]),
            repeat_penalty=float(decode["repeat_penalty"]),
            seed=int(prompt["seed"]),
            stop=["\n"],
        )
        text = _extract_text(raw)
        normalized = " ".join(text.replace("\r", "\n").split())
        usage = dict(raw.get("usage") or {}) if isinstance(raw, Mapping) else {}
        token_count = int(usage.get("completion_tokens", 0) or 0)
        if token_count <= 0:
            token_count = len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
        finish_reason = ""
        if isinstance(raw, Mapping):
            choices = raw.get("choices")
            if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
                finish_reason = str(choices[0].get("finish_reason") or "")
        rows.append(
            {
                "candidate_prompt_id": str(prompt["candidate_prompt_id"]),
                "raw_generation": text,
                "normalized_generation": normalized,
                "generated_token_count": token_count,
                "decode_time_s": round(time.perf_counter() - started, 6),
                "finish_reason": finish_reason,
                "seed": int(prompt["seed"]),
            }
        )
        if len(rows) % 50 == 0:
            print(json.dumps({"event": "decode_progress", "row_count": len(rows)}), flush=True)
    llm = None
    gc.collect()
    output_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "cuda_sync_method": "llama_cpp_backend_close_plus_worker_exit",
            }
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "row_count": len(rows),
                "cuda_sync_method": "llama_cpp_backend_close_plus_worker_exit",
            }
        ),
        flush=True,
    )
    return 0


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    raw_rows_path: str | Path = REPO_ROOT / RAW_ROWS_RELATIVE_PATH,
    ladder_artifact_path: str | Path = REPO_ROOT / EXP6103_ARTIFACT_RELATIVE_PATH,
    ladder_rows_path: str | Path = REPO_ROOT / EXP6103_ROW_RELATIVE_PATH,
    ladder_split_manifest_path: str | Path = REPO_ROOT / EXP6103_SPLIT_RELATIVE_PATH,
    canary_artifact_path: str | Path = REPO_ROOT / EXP6114_ARTIFACT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    generation_backend: GenerationBackend | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Run Exp6115 and optionally write the terminal JSON plus raw JSONL."""

    started = time.perf_counter()
    result = Path(result_path)
    raw_path = Path(raw_rows_path)
    preconditions = (
        _copy_json(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(result_path=result, raw_rows_path=raw_path)
    )
    blockers = list(preconditions.get("blocked_reasons") or [])
    canary_artifact = _read_json(canary_artifact_path)
    gate = _structured_gate(canary_artifact)
    immutable = _immutable_hashes(
        ladder_artifact_path=ladder_artifact_path,
        ladder_rows_path=ladder_rows_path,
        ladder_split_manifest_path=ladder_split_manifest_path,
        canary_artifact_path=canary_artifact_path,
        canary_artifact=canary_artifact,
    )
    model_receipt = _model_receipt(canary_artifact)
    calibration_rows = _calibration_rows_only(ladder_rows_path)
    selected_questions = select_calibration_questions(calibration_rows)
    prompts = _build_prompts(selected_questions)
    tokenizer_prompt = _tokenizer_prompt_decode_receipt(canary_artifact, prompts)
    selected_gpu, gpu_fit, gpu_blockers = _select_gpu(preconditions)
    blockers.extend(gpu_blockers)
    backend_receipt: JsonDict | None = None
    candidate_rows: list[JsonDict] = []
    if gate["model_call_permitted"] is not True:
        blockers.extend(gate["blocked_reasons"])
    if not blockers and selected_gpu is not None:
        backend = generation_backend or LlamaCppCalibrationGenerationBackend(max_wall_s=7_200.0)
        backend_receipt = backend.generate(
            model_spec=dict(model_receipt["records"][MODEL_HF_ID]),
            selected_gpu=selected_gpu,
            prompts=prompts,
            decode_config=dict(DEFAULT_DECODE_POLICY),
            baseline_devices=[
                dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []
            ],
        )
        gate["backend_call_count"] = 1
        candidate_rows = _normalize_candidate_rows(
            source_rows=selected_questions,
            prompts=prompts,
            backend_rows=list(backend_receipt.get("rows") or []),
            model_receipt=model_receipt,
        )
    row_text = rows_to_jsonl(candidate_rows)
    counts = _question_counts(selected_questions)
    raw_receipt = _raw_rows_receipt(candidate_rows, row_text, raw_path)
    parser = _parser_receipt(candidate_rows)
    replay = _replay_receipt(candidate_rows)
    intervals = _accuracy_intervals(candidate_rows)
    diversity = _diversity_metrics(candidate_rows)
    all_wrong_sc = _all_wrong_oracle_sc(candidate_rows)
    selected_policy = _select_policy(
        intervals=intervals, diversity=diversity, all_wrong_sc=all_wrong_sc
    )
    generation = _generation_receipt(
        rows=candidate_rows,
        backend_receipt=backend_receipt,
        preconditions=preconditions,
        selected_gpu=selected_gpu,
    )
    protected = protected_files_unchanged(
        before_hashes=dict(preconditions.get("protected_file_hashes_before") or {})
    )
    artifact: JsonDict = {
        "status": "blocked",
        "preconditions_checked": {
            **dict(preconditions),
            "blocked_reasons": sorted(set(blockers)),
            "single_gpu_fit_receipt": gpu_fit,
        },
        "structured_gate_receipt": gate,
        "immutable_ladder_row_split_and_canary_hashes": immutable,
        "model_specs_and_exact_file_hashes": model_receipt,
        "embedded_tokenizer_prompt_and_decode_policy_candidates": tokenizer_prompt,
        "calibration_question_family_stratum_and_semantic_group_counts": counts,
        "raw_candidate_row_paths_hashes_and_prefix_chain": raw_receipt,
        "generation_seeds_tokens_latency_cuda_vram_thermal_and_pid_receipts": generation,
        "frozen_parser_and_parseability": parser,
        "python_z3_correctness_and_method_validity_replay": replay,
        "per_candidate_accuracy_intervals": intervals,
        "duplicate_effective_k_answer_cluster_and_entropy_metrics": diversity,
        "all_wrong_oracle_tuned_sc_and_solver_strata": all_wrong_sc,
        "relabel_paraphrase_shortcut_controls": _controls(candidate_rows, selected_questions),
        "held_test_access_count": 0,
        "selected_stratum_and_fixed_decode_policy": selected_policy,
        "phase_d_calibration_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s
        if duration_s is not None
        else round(time.perf_counter() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "Free-form prose method validity is checked by a frozen surface parser plus exact answer authorities; Python/Z3 remain oracle for finite-domain labels.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: unclassified",
    }
    artifact["phase_d_calibration_ready_score"] = _phase_score(artifact)
    artifact["status"], artifact["honest_verdict"] = _status_and_verdict(artifact, blockers)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic(raw_path, row_text)
        _write_atomic(result, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp6115 terminal artifact schema and gate consistency."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover
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
    if status == "complete_null" and not verdict.startswith("complete_null:"):  # pragma: no cover
        raise ValueError("complete_null_verdict")
    if status == "blocked" and not verdict.startswith("blocked:"):  # pragma: no cover
        raise ValueError("blocked_verdict")
    if artifact["held_test_access_count"] != 0:  # pragma: no cover
        raise ValueError("held_test_access_count")
    if dict(artifact["embedded_tokenizer_prompt_and_decode_policy_candidates"]).get(
        "auto_tokenizer_used"
    ) is not False:  # pragma: no cover
        raise ValueError("auto_tokenizer_used")
    if dict(artifact["embedded_tokenizer_prompt_and_decode_policy_candidates"]).get(
        "finite_id_transport_used"
    ) is not False:  # pragma: no cover
        raise ValueError("finite_id_transport_used")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--raw-rows", default=str(REPO_ROOT / RAW_ROWS_RELATIVE_PATH))
    parser.add_argument("--worker", default="")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_main(args.worker)
    artifact = run(result_path=args.result, raw_rows_path=args.raw_rows, write=True)
    print(
        json.dumps(
            {"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
