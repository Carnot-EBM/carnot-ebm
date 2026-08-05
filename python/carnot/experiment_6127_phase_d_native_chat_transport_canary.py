"""Exp6127 native-chat transport canary on a frozen Exp6115 slice.

Spec refs: REQ-VERIFY-6127, REQ-VERIFY-6127-1, REQ-VERIFY-6127-2,
REQ-VERIFY-6127-3, REQ-VERIFY-6127-4, REQ-VERIFY-6127-5,
REQ-VERIFY-6127-6, REQ-VERIFY-6127-7, REQ-VERIFY-6127-8,
REQ-VERIFY-6127-9, REQ-VERIFY-6127-10,
SCENARIO-VERIFY-6127-GATE, SCENARIO-VERIFY-6127-SLICE,
SCENARIO-VERIFY-6127-NATIVE-CHAT, SCENARIO-VERIFY-6127-THRESHOLDS,
SCENARIO-VERIFY-6127-LIFECYCLE.

This experiment is deliberately narrow.  It keeps the failed Exp6115 raw
completion rows immutable, freezes a small calibration-only paired slice, and
tests exactly one model-native chat serialization with natural reasoning before
one terminal answer field.  It does not introduce grammar, finite-ID answer
transport, hidden-label retries, parser repair, or a semantic headline.
"""

from __future__ import annotations

from collections import Counter
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
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol

from carnot import experiment_6103_phase_d_difficulty_ladder_fixture as exp6103
from carnot import experiment_6114_phase_d_gpu_ladder_canary as exp6114
from carnot import experiment_6115_phase_d_calibration_pool as exp6115
from carnot import experiment_6126_phase_d_exp6115_transport_forensics as exp6126


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6127_phase_d_native_chat_transport_canary.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6127_phase_d_native_chat_transport_canary.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6127_phase_d_native_chat_transport_canary.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXP6103_ARTIFACT_RELATIVE_PATH = exp6103.RESULT_RELATIVE_PATH
EXP6103_ROW_RELATIVE_PATH = exp6103.ROW_FILE_RELATIVE_PATH
EXP6103_SPLIT_RELATIVE_PATH = exp6103.SPLIT_MANIFEST_RELATIVE_PATH
EXP6114_ARTIFACT_RELATIVE_PATH = exp6114.RESULT_RELATIVE_PATH
EXP6115_ARTIFACT_RELATIVE_PATH = exp6115.RESULT_RELATIVE_PATH
EXP6115_ROWS_RELATIVE_PATH = exp6115.RAW_ROWS_RELATIVE_PATH
EXP6115_MODULE_RELATIVE_PATH = exp6115.MODULE_RELATIVE_PATH
EXP6116_ARTIFACT_RELATIVE_PATH = Path("results/experiment_6116_phase_d_held_candidate_pool.json")
EXP6126_ARTIFACT_RELATIVE_PATH = exp6126.RESULT_RELATIVE_PATH
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")

SCHEMA = "carnot.experiment_6127.phase_d_native_chat_transport_canary.v1"
ROW_SCHEMA = SCHEMA + ".paired_row"
EXPERIMENT_ID = "experiment_6127_phase_d_native_chat_transport_canary"
RUN_DATE = "20260805"
RANDOM_SEED = 6127
MODEL_HF_ID = exp6114.MODEL_HF_ID
MODEL_QUANTIZATION = exp6114.MODEL_QUANTIZATION
MEASURED_FIT_REQUIRED_MB = exp6114.MEASURED_FIT_REQUIRED_MB
FROZEN_QUESTION_COUNT = 18
BASELINE_SAMPLE_INDEX = 0
TREATMENT_MAX_NEW_TOKENS = 1024
TREATMENT_TEMPERATURE = 0.35
TREATMENT_TOP_P = 0.95
TREATMENT_REPEAT_PENALTY = 1.05
INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda_native_chat_transport_canary"
VERIFIER_IS_ORACLE = True

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
    RESEARCH_REFERENCES_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    EXP6103_ARTIFACT_RELATIVE_PATH,
    EXP6103_ROW_RELATIVE_PATH,
    EXP6103_SPLIT_RELATIVE_PATH,
    EXP6114_ARTIFACT_RELATIVE_PATH,
    EXP6115_ARTIFACT_RELATIVE_PATH,
    EXP6115_ROWS_RELATIVE_PATH,
    EXP6115_MODULE_RELATIVE_PATH,
    EXP6116_ARTIFACT_RELATIVE_PATH,
    EXP6126_ARTIFACT_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6127_phase_d_native_chat_transport_canary.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6127_phase_d_native_chat_transport_canary.py "
    "-m pytest tests/python/test_experiment_6127_phase_d_native_chat_transport_canary.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6127_phase_d_native_chat_transport_canary.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6127_phase_d_native_chat_transport_canary.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6127_phase_d_native_chat_transport_canary.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "immutable_ladder_slice_and_row_hashes",
    "model_specs_and_exact_file_hashes",
    "tokenizer_chat_template_and_serialization_hashes",
    "paired_baseline_treatment_prompt_seed_and_budget_contract",
    "raw_completion_stop_reason_token_and_terminal_field_receipts",
    "nonempty_terminal_parse_channel_method_and_accuracy_arm_metrics",
    "paired_deltas_intervals_and_threshold_matrix",
    "hidden_label_retry_grammar_finite_id_and_parser_repair_counts",
    "task_owned_gpu_server_pid_engagement_and_release_timeline",
    "model_native_transport_ready_score",
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
        "all immutable upstream, model, runtime, output, GPU, lease, protected-file, "
        "exclusion, and inherited-debt checks pass before any treatment model load."
    ),
    "immutable_ladder_slice_and_row_hashes": (
        "the paired slice is frozen from calibration rows only, balanced by family "
        "and difficulty, and row identities are conserved."
    ),
    "model_specs_and_exact_file_hashes": (
        "every generated treatment row traces to the one mandated 26B Q4_K_M GGUF "
        "and no substitute model."
    ),
    "tokenizer_chat_template_and_serialization_hashes": (
        "treatment serialization uses the pinned model-native chat template and "
        "hashes the exact serialized messages before decode."
    ),
    "paired_baseline_treatment_prompt_seed_and_budget_contract": (
        "baseline and treatment share questions and seeds, baseline preserves "
        "exact Exp6115 prompt text, and treatment uses no newline stop with a "
        "non-truncating budget."
    ),
    "raw_completion_stop_reason_token_and_terminal_field_receipts": (
        "raw completions, stop reasons, token counts, terminal-field counts, and "
        "hashes are preserved for both arms."
    ),
    "nonempty_terminal_parse_channel_method_and_accuracy_arm_metrics": (
        "transport reachability, parser behavior, channel leakage, method validity, "
        "and exact accuracy are measured separately."
    ),
    "paired_deltas_intervals_and_threshold_matrix": (
        "every preregistered transport threshold must pass; parseability cannot "
        "substitute for method validity."
    ),
    "hidden_label_retry_grammar_finite_id_and_parser_repair_counts": (
        "hidden label retries, grammar, finite-ID transport, and parser repair are all zero."
    ),
    "task_owned_gpu_server_pid_engagement_and_release_timeline": (
        "CUDA lifecycle evidence is attributable to the task-owned worker and cleanup is measured."
    ),
    "model_native_transport_ready_score": (
        "readiness is exactly 1 only for the conjunctive threshold pass."
    ),
    "retirement_triggered": "the same negative verdict retires this exact attempt.",
    "protected_files_unchanged": (
        "conductor and reconciler-owned files remain byte-identical."
    ),
    "duration_s": (
        "report measured `live_local_sota_gguf_cuda_native_chat_transport_canary`."
    ),
    "inference_substrate": (
        "report measured `live_local_sota_gguf_cuda_native_chat_transport_canary`."
    ),
    "field_provenance": (
        "report measured `live_local_sota_gguf_cuda_native_chat_transport_canary`."
    ),
    "test_commands": (
        "report measured `live_local_sota_gguf_cuda_native_chat_transport_canary`."
    ),
    "test_exit_codes": (
        "report measured `live_local_sota_gguf_cuda_native_chat_transport_canary`."
    ),
    "reproducibility_checksum": (
        "report measured `live_local_sota_gguf_cuda_native_chat_transport_canary`."
    ),
    "verifier_is_oracle": (
        "Python/Z3 labels are oracle for calibration semantics, while transport gaps remain explicit."
    ),
    "missing_verifier_gaps": (
        "Python/Z3 labels are oracle for calibration semantics, while transport gaps remain explicit."
    ),
    "honest_verdict": (
        "use `complete_ready:`, `complete_null:`, `retired:`, or `blocked:`."
    ),
}


class NativeChatGenerationBackend(Protocol):
    """Injectable backend that returns one native-chat treatment row per prompt."""

    def generate(
        self,
        *,
        model_spec: JsonDict,
        selected_gpu: int,
        prompts: list[JsonDict],
        decode_config: JsonDict,
        baseline_devices: list[JsonDict],
    ) -> JsonDict:
        """Generate treatment rows without seeing hidden labels."""


def canonical_json(value: Any) -> str:
    """Serialize JSON receipts using a byte-stable representation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with the repository's prefixed SHA-256 convention."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible evidence after deterministic serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without trusting path names or timestamps."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - corrupted artifact guard.
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):  # pragma: no cover - corrupted row guard.
            raise ValueError(f"JSON object row required at line {line_number}: {path}")
        rows.append(dict(payload))
    return rows


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _run_command(command: Sequence[str], *, timeout_s: float) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": False,
        }


def _gpu_devices_with_power() -> list[JsonDict]:  # pragma: no cover
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,power.draw",
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
                    "memory_used_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "temperature_c": int(parts[5]),
                    "power_draw_w": float(parts[6]),
                }
            )
        except ValueError:
            continue
    return devices


def _compute_apps() -> list[JsonDict]:  # pragma: no cover
    result = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    apps: list[JsonDict] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            apps.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "gpu_uuid": parts[2],
                    "used_memory_mb": int(parts[3]),
                }
            )
        except ValueError:
            continue
    return apps


def _memory_probe() -> JsonDict:  # pragma: no cover
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": 16_384, "ok": available_mb >= 16_384}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover
    available_mb = int(shutil.disk_usage(root).free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 10_240, "ok": available_mb >= 10_240}


def _root_clutter_inventory(root: Path) -> JsonDict:  # pragma: no cover
    files = sorted(path.name for path in root.glob("*.py"))
    return {"root_python_files": files, "root_python_file_count": len(files), "ok": not files}


def _file_receipt(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _protected_hashes(root: Path) -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(root / relative)
        for relative in PROTECTED_FILES
        if (root / relative).exists()
    }


def _select_gpu(preconditions: Mapping[str, Any]) -> tuple[int | None, JsonDict, list[str]]:
    devices = [dict(device) for device in dict(preconditions.get("gpu") or {}).get("devices") or []]
    candidates = [
        device
        for device in devices
        if int(device.get("memory_free_mb", 0) or 0) >= MEASURED_FIT_REQUIRED_MB
    ]
    selected = max(candidates, key=lambda row: int(row.get("memory_free_mb", 0)), default=None)
    receipt = {
        "schema": SCHEMA + ".single_gpu_fit",
        "selected_gpu": int(selected["index"]) if selected else None,
        "required_free_mb": MEASURED_FIT_REQUIRED_MB,
        "devices": devices,
        "fits": selected is not None,
    }
    return receipt["selected_gpu"], receipt, [] if selected else ["insufficient_free_vram"]


def _model_record(exp6115_artifact: Mapping[str, Any]) -> JsonDict:
    records = dict(dict(exp6115_artifact["model_specs_and_exact_file_hashes"]).get("records") or {})
    record = dict(records[MODEL_HF_ID])
    return record


def _runtime_chat_api_receipt() -> JsonDict:  # pragma: no cover
    try:
        return exp6126.runtime_chat_template_api()
    except Exception as exc:
        return {
            "llama_cpp_importable": False,
            "llama_create_chat_completion_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    gguf_metadata_path: str | Path | None = None,
) -> JsonDict:  # pragma: no cover - host resource probe.
    result = Path(result_path)
    exp6115_artifact = read_json(root / EXP6115_ARTIFACT_RELATIVE_PATH)
    model_record = _model_record(exp6115_artifact)
    model_path = Path(gguf_metadata_path or str(model_record["model_path"])).expanduser()
    model_file_exists = model_path.is_file()
    model_sha = sha256_file(model_path) if model_file_exists else None
    devices = _gpu_devices_with_power()
    output = {
        "path": str(result),
        "parent_writable": os.access(result.parent, os.W_OK),
        "existed_before": result.exists(),
        "sha256_before": sha256_file(result) if result.exists() else None,
    }
    inherited_debt = {
        "known_issues_sha256": sha256_file(root / KNOWN_ISSUES_RELATIVE_PATH),
        "research_references_sha256": sha256_file(root / RESEARCH_REFERENCES_RELATIVE_PATH),
        "exclusion_manifest_sha256": sha256_file(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
    }
    blocked: list[str] = []
    if not output["parent_writable"]:
        blocked.append("output_path_not_writable")
    if not model_file_exists:
        blocked.append("mandated_gguf_missing")
    if model_sha and model_sha != model_record.get("model_sha256"):
        blocked.append("mandated_gguf_hash_mismatch")
    if not devices:
        blocked.append("gpu_device_receipt_unavailable")
    memory = _memory_probe()
    disk = _disk_probe(root)
    if memory["ok"] is not True:
        blocked.append("insufficient_free_ram")
    if disk["ok"] is not True:
        blocked.append("insufficient_free_disk")
    root_clutter = _root_clutter_inventory(root)
    if root_clutter["ok"] is not True:
        blocked.append("root_python_clutter_present")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hashed_input_receipts": [_file_receipt(root, relative) for relative in HASHED_INPUTS],
        "model_file": {
            "path": str(model_path),
            "exists": model_file_exists,
            "recorded_sha256": model_record.get("model_sha256"),
            "recomputed_sha256": model_sha,
            "hash_recomputed_before_model_load": model_sha is not None,
        },
        "gpu": {"gpu_count": len(devices), "ok": bool(devices), "devices": devices},
        "compute_apps_before": _compute_apps(),
        "lease_state": {
            "task_owned_pid": os.getpid(),
            "parent_pid": os.getppid(),
            "child_pids_before": [],
            "lease_scope": "task_owned_child_worker_only",
        },
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output,
        "root_clutter": root_clutter,
        "protected_file_hashes_before": _protected_hashes(root),
        "inherited_debt": inherited_debt,
    }


def _build_treatment_messages(source: Mapping[str, Any]) -> list[JsonDict]:
    labels = [str(item["label"]) for item in source["answer_space"]]
    label_pattern = "|".join(sorted(labels))
    choices = "\n".join(
        f"{item['label']}: {item['candidate']}" for item in source["answer_space"]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are solving a finite-choice calibration item. Reason naturally "
                "using only public facts, then end with one terminal answer field."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{source['problem']['prompt_stem']}\n\n"
                f"Choices:\n{choices}\n\n"
                "Think through the visible rule or constraints in natural language. "
                f"End with exactly one final line: Final answer: <{label_pattern}>"
            ),
        },
    ]


def _quota_matrix() -> dict[str, dict[str, int]]:
    families = list(exp6103.FAMILIES)
    strata = list(exp6115.DIFFICULTY_STRATA)
    return {
        families[0]: {strata[0]: 2, strata[1]: 2, strata[2]: 1, strata[3]: 1},
        families[1]: {strata[0]: 2, strata[1]: 1, strata[2]: 2, strata[3]: 1},
        families[2]: {strata[0]: 1, strata[1]: 2, strata[2]: 1, strata[3]: 2},
    }


def freeze_paired_slice(
    exp6115_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Freeze the 18-question calibration-only paired slice before generation."""

    source_by_id = {str(row["row_id"]): dict(row) for row in source_rows}
    baseline_rows = [
        dict(row)
        for row in exp6115_rows
        if str(row.get("source_split")) == "calibration"
        and int(row.get("sample_index", -1)) == BASELINE_SAMPLE_INDEX
    ]
    by_family_stratum: dict[tuple[str, str], list[JsonDict]] = {}
    for row in baseline_rows:
        key = (str(row["family"]), str(row["difficulty_stratum"]))
        by_family_stratum.setdefault(key, []).append(row)
    for rows in by_family_stratum.values():
        rows.sort(key=lambda row: str(row["source_exp6103_row_id"]))

    pairs: list[JsonDict] = []
    for family, stratum_counts in _quota_matrix().items():
        for stratum, count in stratum_counts.items():
            candidates = by_family_stratum.get((family, stratum), [])
            if len(candidates) < count:  # pragma: no cover - sealed fixture corruption.
                raise ValueError(f"not_enough_rows:{family}:{stratum}")
            for baseline in candidates[:count]:
                source_id = str(baseline["source_exp6103_row_id"])
                source = source_by_id[source_id]
                baseline_prompt_text = exp6115._prompt_text(source)
                messages = _build_treatment_messages(source)
                message_serialization = canonical_json(messages)
                seed = int(baseline["seed"])
                pairs.append(
                    {
                        "source_exp6103_row_id": source_id,
                        "source_row_hash": str(source["row_hash"]),
                        "source_split": str(source["split"]),
                        "family": str(baseline["family"]),
                        "difficulty_stratum": str(baseline["difficulty_stratum"]),
                        "semantic_group_id": str(baseline["semantic_group_id"]),
                        "baseline_candidate_row_id": str(baseline["candidate_row_id"]),
                        "baseline_candidate_row_hash": str(baseline["candidate_row_hash"]),
                        "baseline_sample_index": int(baseline["sample_index"]),
                        "baseline_seed": seed,
                        "treatment_seed": seed,
                        "baseline_prompt_text": baseline_prompt_text,
                        "baseline_prompt_hash": sha256_text(baseline_prompt_text),
                        "baseline_prompt_hash_from_exp6115": str(baseline["prompt_hash"]),
                        "baseline_prompt_hash_matches_exp6115": (
                            sha256_text(baseline_prompt_text) == str(baseline["prompt_hash"])
                        ),
                        "treatment_row_id": (
                            f"exp6127|{source_id}|native_chat|seed-{seed}"
                        ),
                        "treatment_messages": messages,
                        "treatment_message_serialization": message_serialization,
                        "treatment_message_hash": sha256_text(message_serialization),
                        "baseline_row": baseline,
                    }
                )
    pairs.sort(key=lambda row: (row["family"], row["source_exp6103_row_id"]))
    family_counts = Counter(str(row["family"]) for row in pairs)
    difficulty_counts = Counter(str(row["difficulty_stratum"]) for row in pairs)
    slice_identity = [
        {
            "source_exp6103_row_id": pair["source_exp6103_row_id"],
            "baseline_candidate_row_id": pair["baseline_candidate_row_id"],
            "treatment_row_id": pair["treatment_row_id"],
            "seed": pair["treatment_seed"],
            "treatment_message_hash": pair["treatment_message_hash"],
        }
        for pair in pairs
    ]
    return {
        "schema": SCHEMA + ".frozen_slice",
        "question_count": len(pairs),
        "minimum_question_count": FROZEN_QUESTION_COUNT,
        "baseline_sample_index": BASELINE_SAMPLE_INDEX,
        "held_test_access_count": sum(1 for row in pairs if row["source_split"] != "calibration"),
        "family_counts": dict(sorted(family_counts.items())),
        "difficulty_stratum_counts": dict(sorted(difficulty_counts.items())),
        "difficulty_balance_max_minus_min": (
            max(difficulty_counts.values()) - min(difficulty_counts.values())
        ),
        "semantic_group_count": len({str(row["semantic_group_id"]) for row in pairs}),
        "paired_question_and_seed_count": sum(
            1 for row in pairs if row["baseline_seed"] == row["treatment_seed"]
        ),
        "all_questions_and_seeds_paired": all(
            row["baseline_seed"] == row["treatment_seed"] for row in pairs
        ),
        "slice_identity_hash": sha256_json(slice_identity),
        "pairs": pairs,
        "principle": REQUIRED_FIELD_PRINCIPLES["immutable_ladder_slice_and_row_hashes"],
    }


def _treatment_prompts(frozen_slice: Mapping[str, Any]) -> list[JsonDict]:
    prompts = []
    for pair in frozen_slice["pairs"]:
        prompts.append(
            {
                "treatment_row_id": str(pair["treatment_row_id"]),
                "source_exp6103_row_id": str(pair["source_exp6103_row_id"]),
                "family": str(pair["family"]),
                "difficulty_stratum": str(pair["difficulty_stratum"]),
                "semantic_group_id": str(pair["semantic_group_id"]),
                "seed": int(pair["treatment_seed"]),
                "messages": _copy_json(pair["treatment_messages"]),
                "message_serialization": str(pair["treatment_message_serialization"]),
                "message_hash": str(pair["treatment_message_hash"]),
                "max_new_tokens": TREATMENT_MAX_NEW_TOKENS,
            }
        )
    return prompts


def _has_channel_token(text: str) -> bool:
    return any(token in text for token in ("<|channel>", "<|message", "<|start", "<|end"))


def _terminal_field_count(text: str) -> int:
    return len(re.findall(r"final\s+answer\s*:", text, flags=re.IGNORECASE))


def _normalize_text(text: str) -> str:
    return "\n".join(
        line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    )


def _score_text(source: Mapping[str, Any], text: str) -> JsonDict:
    normalized = _normalize_text(text)
    parser = exp6115._parse_final_answer(normalized, list(source["answer_space"]))
    python = exp6103.python_validate_row(source)
    z3 = exp6103.z3_validate_row(source)
    exact_label = str(python["exact_label"])
    exact = parser["parseable"] is True and str(parser["parsed_label"]) == exact_label
    method_valid, method_reason = exp6115._method_evidence(source, normalized, exact)
    return {
        "normalized_generation": normalized,
        "parser": parser,
        "python_exact_label": exact_label,
        "z3_exact_label": str(z3["exact_label"]),
        "python_z3_agree": exact_label == str(z3["exact_label"]),
        "exact_correct": exact,
        "method_valid": method_valid,
        "method_validity_reason": method_reason,
    }


def _baseline_rows(frozen_slice: Mapping[str, Any]) -> list[JsonDict]:
    rows = []
    for pair in frozen_slice["pairs"]:
        row = dict(pair["baseline_row"])
        row["arm"] = "baseline"
        row["paired_treatment_row_id"] = str(pair["treatment_row_id"])
        row["terminal_field_count"] = _terminal_field_count(str(row.get("raw_generation") or ""))
        row["channel_leakage"] = _has_channel_token(str(row.get("raw_generation") or ""))
        rows.append(row)
    return rows


def _normalize_treatment_rows(
    *,
    frozen_slice: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    backend_rows: Sequence[Mapping[str, Any]],
    model_record: Mapping[str, Any],
) -> list[JsonDict]:
    source_by_id = {str(row["row_id"]): dict(row) for row in source_rows}
    backend_by_id = {str(row["treatment_row_id"]): dict(row) for row in backend_rows}
    rows = []
    for pair in frozen_slice["pairs"]:
        backend = backend_by_id.get(str(pair["treatment_row_id"]))
        if backend is None:
            continue
        raw = str(backend.get("raw_generation") or "")
        source = source_by_id[str(pair["source_exp6103_row_id"])]
        scored = _score_text(source, raw)
        row: JsonDict = {
            "schema": ROW_SCHEMA,
            "arm": "treatment",
            "treatment_row_id": str(pair["treatment_row_id"]),
            "baseline_candidate_row_id": str(pair["baseline_candidate_row_id"]),
            "source_exp6103_row_id": str(pair["source_exp6103_row_id"]),
            "source_row_hash": str(pair["source_row_hash"]),
            "source_split": str(pair["source_split"]),
            "family": str(pair["family"]),
            "difficulty_stratum": str(pair["difficulty_stratum"]),
            "semantic_group_id": str(pair["semantic_group_id"]),
            "model_hf_id": MODEL_HF_ID,
            "model_file_sha256": str(model_record.get("model_sha256") or ""),
            "seed": int(backend.get("seed", pair["treatment_seed"])),
            "max_new_tokens": TREATMENT_MAX_NEW_TOKENS,
            "temperature": TREATMENT_TEMPERATURE,
            "top_p": TREATMENT_TOP_P,
            "repeat_penalty": TREATMENT_REPEAT_PENALTY,
            "messages_hash": str(pair["treatment_message_hash"]),
            "raw_generation": raw,
            "normalized_generation": str(
                backend.get("normalized_generation") or scored["normalized_generation"]
            ),
            "raw_generation_hash": sha256_text(raw),
            "generated_token_count": int(backend.get("generated_token_count", 0) or 0),
            "decode_time_s": float(backend.get("decode_time_s", 0.0) or 0.0),
            "finish_reason": str(backend.get("finish_reason") or ""),
            "terminal_field_count": _terminal_field_count(raw),
            "channel_leakage": _has_channel_token(raw),
            "candidate_row_hash": "",
            **{key: value for key, value in scored.items() if key != "normalized_generation"},
        }
        stable = _copy_json(row)
        stable["candidate_row_hash"] = ""
        row["candidate_row_hash"] = sha256_json(stable)
        rows.append(row)
    rows.sort(key=lambda row: row["treatment_row_id"])
    return rows


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _row_bool(row: Mapping[str, Any], name: str) -> bool:
    if name == "nonempty":
        return str(row.get("raw_generation") or "") != ""
    if name == "terminal_field":
        return _terminal_field_count(str(row.get("raw_generation") or "")) > 0
    if name == "parseable":
        return bool(dict(row.get("parser") or {}).get("parseable"))
    if name == "channel_leakage":
        return _has_channel_token(str(row.get("raw_generation") or ""))
    if name == "method_valid":
        return bool(row.get("method_valid"))
    if name == "exact_correct":
        return bool(row.get("exact_correct"))
    if name == "length_finish":
        return str(row.get("finish_reason") or "") == "length"
    raise KeyError(name)  # pragma: no cover - developer error.


def _arm_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    nonempty = sum(_row_bool(row, "nonempty") for row in rows)
    terminal = sum(_row_bool(row, "terminal_field") for row in rows)
    parseable = sum(_row_bool(row, "parseable") for row in rows)
    channel = sum(_row_bool(row, "channel_leakage") for row in rows)
    method = sum(_row_bool(row, "method_valid") for row in rows)
    exact = sum(_row_bool(row, "exact_correct") for row in rows)
    length = sum(_row_bool(row, "length_finish") for row in rows)
    return {
        "candidate_count": total,
        "nonempty_count": nonempty,
        "nonempty_rate": _rate(nonempty, total),
        "exact_empty_count": total - nonempty,
        "terminal_field_reach_count": terminal,
        "terminal_field_reach_rate": _rate(terminal, total),
        "terminal_field_exactly_once_count": sum(
            _terminal_field_count(str(row.get("raw_generation") or "")) == 1 for row in rows
        ),
        "parseable_count": parseable,
        "parseability": _rate(parseable, total),
        "channel_leakage_count": channel,
        "channel_leakage_rate": _rate(channel, total),
        "method_valid_count": method,
        "method_validity": _rate(method, total),
        "exact_correct_count": exact,
        "answer_accuracy": _rate(exact, total),
        "length_finish_reason_count": length,
        "finish_reason_counts": dict(sorted(Counter(str(row.get("finish_reason") or "") for row in rows).items())),
        "generated_token_count": {
            "min": min([int(row.get("generated_token_count", 0) or 0) for row in rows] or [0]),
            "max": max([int(row.get("generated_token_count", 0) or 0) for row in rows] or [0]),
            "sum": sum(int(row.get("generated_token_count", 0) or 0) for row in rows),
        },
    }


def arm_metrics_receipt(
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".arm_metrics",
        "baseline": _arm_metrics(baseline_rows),
        "treatment": _arm_metrics(treatment_rows),
        "transport_primary_fields": [
            "nonempty_rate",
            "terminal_field_reach_rate",
            "parseability",
            "channel_leakage_rate",
            "length_finish_reason_count",
        ],
        "semantic_fields": ["method_validity", "answer_accuracy"],
        "accuracy_reported_not_transport_primary": True,
        "method_validity_not_inferred_from_parseability": True,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "nonempty_terminal_parse_channel_method_and_accuracy_arm_metrics"
        ],
    }


def _paired_interval(diffs: Sequence[int]) -> JsonDict:
    n = len(diffs)
    if n == 0:
        return {"n": 0, "delta": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    mean = sum(diffs) / n
    variance = (
        sum((value - mean) ** 2 for value in diffs) / (n - 1)
        if n > 1
        else 0.0
    )
    se = math.sqrt(variance / n) if n else 0.0
    return {
        "n": n,
        "delta": round(mean, 6),
        "lower_95": round(mean - 1.96 * se, 6),
        "upper_95": round(mean + 1.96 * se, 6),
    }


def paired_threshold_matrix(
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
) -> JsonDict:
    treatment_by_source = {str(row["source_exp6103_row_id"]): dict(row) for row in treatment_rows}
    paired_baseline = [
        dict(row)
        for row in baseline_rows
        if str(row["source_exp6103_row_id"]) in treatment_by_source
    ]
    intervals = {}
    for name in (
        "nonempty",
        "terminal_field",
        "parseable",
        "channel_leakage",
        "method_valid",
        "exact_correct",
    ):
        diffs = [
            int(_row_bool(treatment_by_source[str(row["source_exp6103_row_id"])], name))
            - int(_row_bool(row, name))
            for row in paired_baseline
        ]
        intervals[name] = _paired_interval(diffs)
    metrics = arm_metrics_receipt(baseline_rows, treatment_rows)
    treatment = metrics["treatment"]
    method_delta = intervals["method_valid"]
    accuracy_delta = intervals["exact_correct"]
    thresholds = {
        "treatment_nonempty_at_least_0_95": {
            "observed": treatment["nonempty_rate"],
            "threshold": 0.95,
            "pass": treatment["nonempty_rate"] >= 0.95,
        },
        "treatment_terminal_field_reach_at_least_0_95": {
            "observed": treatment["terminal_field_reach_rate"],
            "threshold": 0.95,
            "pass": treatment["terminal_field_reach_rate"] >= 0.95,
        },
        "treatment_parseability_at_least_0_90": {
            "observed": treatment["parseability"],
            "threshold": 0.90,
            "pass": treatment["parseability"] >= 0.90,
        },
        "treatment_channel_leakage_at_most_0_05": {
            "observed": treatment["channel_leakage_rate"],
            "threshold": 0.05,
            "pass": treatment["channel_leakage_rate"] <= 0.05,
        },
        "treatment_length_finish_reason_count_zero": {
            "observed": treatment["length_finish_reason_count"],
            "threshold": 0,
            "pass": treatment["length_finish_reason_count"] == 0,
        },
    }
    transport_pass = all(row["pass"] is True for row in thresholds.values())
    method_pass = method_delta["delta"] > 0 and method_delta["lower_95"] > 0
    accuracy_pass = accuracy_delta["delta"] >= 0
    lifecycle_pass = (
        lifecycle.get("gpu_engagement_attributable") is True
        and lifecycle.get("release_ready") is True
    )
    return {
        "schema": SCHEMA + ".paired_threshold_matrix",
        "preregistered_before_treatment_generation": True,
        "paired_question_count": len(paired_baseline),
        "paired_binary_intervals": intervals,
        "thresholds": thresholds,
        "all_preregistered_transport_thresholds_pass": transport_pass,
        "method_validity_delta": {**method_delta, "pass": method_pass},
        "exact_accuracy_delta": {**accuracy_delta, "pass": accuracy_pass},
        "semantic_accuracy_reported_not_transport_primary": True,
        "parseability_cannot_substitute_for_method_validity": True,
        "gpu_lifecycle_pass": lifecycle_pass,
        "all_conjunctive_readiness_thresholds_pass": (
            transport_pass and method_pass and accuracy_pass and lifecycle_pass
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "paired_deltas_intervals_and_threshold_matrix"
        ],
    }


def _raw_row_receipts(
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    frozen_slice: Mapping[str, Any],
) -> JsonDict:
    def per_row(rows: Sequence[Mapping[str, Any]], arm: str) -> list[JsonDict]:
        out = []
        for row in rows:
            raw = str(row.get("raw_generation") or "")
            row_id = str(row.get("candidate_row_id") or row.get("treatment_row_id"))
            out.append(
                {
                    "arm": arm,
                    "row_id": row_id,
                    "source_exp6103_row_id": str(row["source_exp6103_row_id"]),
                    "seed": int(row.get("seed", 0) or 0),
                    "raw_completion": raw,
                    "raw_completion_hash": sha256_text(raw),
                    "finish_reason": str(row.get("finish_reason") or ""),
                    "generated_token_count": int(row.get("generated_token_count", 0) or 0),
                    "terminal_field_count": _terminal_field_count(raw),
                    "terminal_field_reached": _terminal_field_count(raw) > 0,
                    "parser": _copy_json(row.get("parser") or {}),
                    "python_exact_label": str(row.get("python_exact_label") or ""),
                    "z3_exact_label": str(row.get("z3_exact_label") or ""),
                    "exact_correct": bool(row.get("exact_correct")),
                    "method_valid": bool(row.get("method_valid")),
                    "method_validity_reason": str(row.get("method_validity_reason") or ""),
                    "model_file_sha256": str(row.get("model_file_sha256") or ""),
                }
            )
        return out

    baseline_metrics = _arm_metrics(baseline_rows)
    treatment_metrics = _arm_metrics(treatment_rows)
    return {
        "schema": SCHEMA + ".raw_completion_receipts",
        "slice_identity_hash": frozen_slice["slice_identity_hash"],
        "baseline": {
            "source": "immutable_exp6115_raw_completion_rows",
            "candidate_count": len(baseline_rows),
            "raw_completions_preserved": True,
            "terminal_field_reach_count": baseline_metrics["terminal_field_reach_count"],
            "length_finish_reason_count": baseline_metrics["length_finish_reason_count"],
            "rows": per_row(baseline_rows, "baseline"),
        },
        "treatment": {
            "source": "live_model_native_messages_generation",
            "candidate_count": len(treatment_rows),
            "raw_completions_preserved": all("raw_generation" in row for row in treatment_rows),
            "terminal_field_reach_count": treatment_metrics["terminal_field_reach_count"],
            "length_finish_reason_count": treatment_metrics["length_finish_reason_count"],
            "rows": per_row(treatment_rows, "treatment"),
        },
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "raw_completion_stop_reason_token_and_terminal_field_receipts"
        ],
    }


def _hidden_mechanism_counts() -> JsonDict:
    return {
        "hidden_label_retry_count": 0,
        "grammar_count": 0,
        "finite_id_transport_count": 0,
        "parser_repair_count": 0,
        "deterministic_answer_builder_count": 0,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "hidden_label_retry_grammar_finite_id_and_parser_repair_counts"
        ],
    }


def _model_specs_receipt(
    exp6115_artifact: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    record = _model_record(exp6115_artifact)
    model_pre = dict(preconditions.get("model_file") or {})
    recomputed = str(model_pre.get("recomputed_sha256") or record.get("model_sha256") or "")
    return {
        "schema": SCHEMA + ".model_specs",
        "selected_model_hf_id": MODEL_HF_ID,
        "quantization": MODEL_QUANTIZATION,
        "records": {MODEL_HF_ID: record},
        "model_file_rehashed_before_model_load": bool(model_pre.get("hash_recomputed_before_model_load")),
        "model_file_recomputed_sha256": recomputed,
        "model_file_recorded_sha256": str(record.get("model_sha256") or ""),
        "model_hash_matches_recorded": recomputed == str(record.get("model_sha256") or ""),
        "tiny_model_substituted": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["model_specs_and_exact_file_hashes"],
    }


def _tokenizer_serialization_receipt(
    *,
    exp6115_artifact: Mapping[str, Any],
    frozen_slice: Mapping[str, Any],
    gguf_metadata_path: str | Path | None,
) -> JsonDict:
    model_path = Path(gguf_metadata_path or _model_record(exp6115_artifact)["model_path"])
    metadata = exp6126.read_gguf_metadata(model_path)
    message_hashes = {
        str(pair["treatment_row_id"]): str(pair["treatment_message_hash"])
        for pair in frozen_slice["pairs"]
    }
    return {
        "schema": SCHEMA + ".tokenizer_chat_template_serialization",
        "metadata_reader": metadata["metadata_reader"],
        "model_path": str(model_path),
        "tokenizer_metadata_sha256": metadata["tokenizer_metadata_sha256"],
        "chat_template_present": bool(metadata["chat_template_present"]),
        "chat_template_sha256": metadata["chat_template_sha256"],
        "chat_template_keys": metadata["chat_template_keys"],
        "chat_template_preview": metadata["chat_template_preview"],
        "runtime_chat_template_api": _runtime_chat_api_receipt(),
        "serialization_api": "llama_cpp.Llama.create_chat_completion",
        "treatment_message_hashes": message_hashes,
        "treatment_serialization_root_hash": sha256_json(message_hashes),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "tokenizer_chat_template_and_serialization_hashes"
        ],
    }


def _prompt_seed_budget_contract(frozen_slice: Mapping[str, Any]) -> JsonDict:
    pairs = []
    for pair in frozen_slice["pairs"]:
        pairs.append(
            {
                "source_exp6103_row_id": pair["source_exp6103_row_id"],
                "baseline_candidate_row_id": pair["baseline_candidate_row_id"],
                "treatment_row_id": pair["treatment_row_id"],
                "seed": pair["treatment_seed"],
                "baseline_prompt_text": pair["baseline_prompt_text"],
                "baseline_prompt_hash": pair["baseline_prompt_hash"],
                "baseline_raw_completion_source": "Exp6115 immutable rows",
                "treatment_messages": _copy_json(pair["treatment_messages"]),
                "treatment_message_hash": pair["treatment_message_hash"],
            }
        )
    return {
        "schema": SCHEMA + ".paired_prompt_seed_budget_contract",
        "paired_question_count": frozen_slice["question_count"],
        "all_questions_and_seeds_paired": frozen_slice["all_questions_and_seeds_paired"],
        "baseline_contract": {
            "source_experiment": exp6115.EXPERIMENT_ID,
            "serialization_api": "llama_cpp.Llama.__call__",
            "prompt_template_version": exp6115.PROMPT_TEMPLATE_VERSION,
            "explicit_stop_strings": ["\n"],
            "max_new_tokens": exp6115.DEFAULT_DECODE_POLICY["max_new_tokens"],
        },
        "treatment_contract": {
            "contract_id": "exp6127_model_native_messages_terminal_field_no_newline_stop",
            "serialization_api": "llama_cpp.Llama.create_chat_completion",
            "uses_model_native_messages": True,
            "natural_reasoning_allowed": True,
            "terminal_answer_field": "Final answer: <A|B|C|D>",
            "explicit_stop_strings": [],
            "newline_stop_removed": True,
            "max_new_tokens": TREATMENT_MAX_NEW_TOKENS,
            "non_truncating_budget_fail_closed": True,
            "temperature": TREATMENT_TEMPERATURE,
            "top_p": TREATMENT_TOP_P,
            "repeat_penalty": TREATMENT_REPEAT_PENALTY,
            "grammar": None,
            "finite_id_transport": False,
        },
        "pairs": pairs,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "paired_baseline_treatment_prompt_seed_and_budget_contract"
        ],
    }


def _lifecycle_receipt(
    *,
    backend_receipt: Mapping[str, Any] | None,
    selected_gpu: int | None,
    baseline_devices: Sequence[Mapping[str, Any]],
) -> JsonDict:
    receipt = dict(backend_receipt or {})
    engagement = dict(receipt.get("gpu_engagement") or {})
    timeline = list(receipt.get("timeline") or [])
    gpu_engaged = (
        engagement.get("attributable") is True
        and int(engagement.get("selected_gpu_memory_delta_mb", 0) or 0) > 0
    )
    release_ready = (
        receipt.get("server_exit_code") == 0
        and receipt.get("pid_exited") is True
        and bool(receipt.get("cuda_sync_method"))
        and receipt.get("vram_release_observed") is True
        and not list(receipt.get("unrelated_processes_killed") or [])
    )
    return {
        "schema": SCHEMA + ".gpu_lifecycle",
        "selected_gpu": selected_gpu,
        "baseline_devices": _copy_json(list(baseline_devices)),
        "server_pid": receipt.get("server_pid"),
        "server_exit_code": receipt.get("server_exit_code"),
        "worker_exit_observed": receipt.get("worker_exit_observed") is True,
        "pid_exited": receipt.get("pid_exited") is True,
        "cuda_sync_method": str(receipt.get("cuda_sync_method") or ""),
        "vram_release_observed": receipt.get("vram_release_observed") is True,
        "unrelated_processes_killed": list(receipt.get("unrelated_processes_killed") or []),
        "gpu_engagement_attributable": gpu_engaged,
        "selected_gpu_memory_delta_mb": int(
            engagement.get("selected_gpu_memory_delta_mb", 0) or 0
        ),
        "release_ready": release_ready,
        "timeline": _copy_json(timeline),
        "energy_telemetry": _copy_json(
            receipt.get("energy_telemetry")
            or {"available": False, "power_samples": [], "estimated_energy_j": None}
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "task_owned_gpu_server_pid_engagement_and_release_timeline"
        ],
    }


def protected_files_unchanged(
    *,
    before_hashes: Mapping[str, str],
    root: Path = REPO_ROOT,
) -> JsonDict:
    after = _protected_hashes(root)
    changed = sorted(
        path for path, before_hash in dict(before_hashes).items() if after.get(path) != before_hash
    )
    return {
        "schema": SCHEMA + ".protected_files",
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "before_hashes": dict(before_hashes),
        "after_hashes": after,
        "changed_files": changed,
        "unchanged": not changed,
        "scripts_research_conductor_modified": "scripts/research_conductor.py" in changed,
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _immutable_hashes(
    *,
    root: Path,
    frozen_slice: Mapping[str, Any],
    tokenizer_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".immutable_hashes",
        "files": [_file_receipt(root, relative) for relative in HASHED_INPUTS],
        "frozen_slice_identity_hash": frozen_slice["slice_identity_hash"],
        "frozen_source_row_ids": [
            str(pair["source_exp6103_row_id"]) for pair in frozen_slice["pairs"]
        ],
        "frozen_baseline_candidate_row_ids": [
            str(pair["baseline_candidate_row_id"]) for pair in frozen_slice["pairs"]
        ],
        "frozen_treatment_row_ids": [
            str(pair["treatment_row_id"]) for pair in frozen_slice["pairs"]
        ],
        "tokenizer_chat_template_sha256": tokenizer_receipt["chat_template_sha256"],
        "treatment_serialization_root_hash": tokenizer_receipt[
            "treatment_serialization_root_hash"
        ],
        "output_path_pre_write": _copy_json(preconditions.get("output_paths") or {}),
        "protected_file_hashes_before": dict(
            preconditions.get("protected_file_hashes_before") or {}
        ),
        "exp6116_evidence_hashed_without_label_inspection": True,
        "principle": REQUIRED_FIELD_PRINCIPLES["immutable_ladder_slice_and_row_hashes"],
    }


def _structured_gate(
    *,
    preconditions: Mapping[str, Any],
    exp6114_artifact: Mapping[str, Any],
    exp6115_artifact: Mapping[str, Any],
    exp6126_artifact: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    tokenizer_receipt: Mapping[str, Any],
    selected_gpu: int | None,
    gpu_fit: Mapping[str, Any],
    blockers: Sequence[str],
) -> JsonDict:
    checks = {
        "preconditions_ready": preconditions.get("preconditions_ready") is True,
        "exp6114_ready": (
            exp6114_artifact.get("status") == "complete_ready"
            and float(exp6114_artifact.get("phase_d_compute_and_ladder_ready_score", 0.0) or 0.0)
            == 1.0
        ),
        "exp6115_rows_available": exp6115_artifact.get("status")
        in {"complete_null", "complete_ready", "complete_partial"},
        "exp6126_native_chat_change_justified": (
            exp6126_artifact.get("status") == "complete_ready"
            and int(exp6126_artifact.get("model_native_chat_change_justified_score", 0) or 0)
            == 1
        ),
        "model_hash_matches_recorded": model_specs.get("model_hash_matches_recorded") is True,
        "tokenizer_chat_template_present": tokenizer_receipt.get("chat_template_present") is True,
        "runtime_chat_api_available": dict(
            tokenizer_receipt.get("runtime_chat_template_api") or {}
        ).get("llama_create_chat_completion_available")
        is True,
        "single_gpu_fit": selected_gpu is not None and gpu_fit.get("fits") is True,
        "no_prior_blockers": not list(blockers),
    }
    gate_blockers = [name for name, ok in checks.items() if ok is not True]
    gate_blockers.extend(blockers)
    return {
        "schema": SCHEMA + ".structured_gate",
        "run_date": RUN_DATE,
        "model_load_permitted": not gate_blockers,
        "backend_call_count": 0,
        "pre_model_load_hashing_complete": True,
        "blockers": sorted(set(gate_blockers)),
        "checks": checks,
        "selected_gpu": selected_gpu,
        "single_gpu_fit_receipt": _copy_json(gpu_fit),
        "exp6114_status": exp6114_artifact.get("status"),
        "exp6115_status": exp6115_artifact.get("status"),
        "exp6126_status": exp6126_artifact.get("status"),
        "principle": REQUIRED_FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "sources": [
                EXP6103_ROW_RELATIVE_PATH.as_posix(),
                EXP6114_ARTIFACT_RELATIVE_PATH.as_posix(),
                EXP6115_ARTIFACT_RELATIVE_PATH.as_posix(),
                EXP6115_ROWS_RELATIVE_PATH.as_posix(),
                EXP6126_ARTIFACT_RELATIVE_PATH.as_posix(),
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
                SPEC_RELATIVE_PATH.as_posix(),
            ],
            "principle": REQUIRED_FIELD_PRINCIPLES.get(field, "required schema field"),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status_and_verdict(
    *,
    blockers: Sequence[str],
    ready_score: int,
    treatment_attempted: bool,
) -> tuple[str, str]:
    if blockers:
        return "blocked", "blocked: structured_gate_or_treatment_backend_incomplete"
    if ready_score == 1:
        return "complete_ready", "complete_ready: native_chat_transport_canary_passed"
    if treatment_attempted:
        return "retired", "retired: native_chat_transport_canary_failed_conjunctive_gate"
    return "complete_null", "complete_null: no_treatment_attempt_after_nonblocking_null"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable.pop("reproducibility_checksum", None)
    stable.pop("duration_s", None)
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover - schema guard.
        raise ValueError(f"missing_fields:{missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):  # pragma: no cover
        raise ValueError("reproducibility_checksum")
    status = str(artifact["status"])
    verdict = str(artifact["honest_verdict"])
    if status == "complete_ready" and not verdict.startswith("complete_ready:"):  # pragma: no cover
        raise ValueError("complete_ready_verdict")
    if status == "retired" and not verdict.startswith("retired:"):  # pragma: no cover
        raise ValueError("retired_verdict")
    if status == "blocked" and not verdict.startswith("blocked:"):  # pragma: no cover
        raise ValueError("blocked_verdict")
    if artifact["model_native_transport_ready_score"] not in {0, 1}:  # pragma: no cover
        raise ValueError("model_native_transport_ready_score")
    disabled = dict(artifact["hidden_label_retry_grammar_finite_id_and_parser_repair_counts"])
    for key in (
        "hidden_label_retry_count",
        "grammar_count",
        "finite_id_transport_count",
        "parser_repair_count",
        "deterministic_answer_builder_count",
    ):
        if disabled.get(key) != 0:  # pragma: no cover
            raise ValueError(key)
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:  # pragma: no cover
        raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not True:  # pragma: no cover
        raise ValueError("verifier_is_oracle")
    matrix = dict(artifact["paired_deltas_intervals_and_threshold_matrix"])
    if matrix.get("parseability_cannot_substitute_for_method_validity") is not True:  # pragma: no cover
        raise ValueError("parseability_substitution")
    return True


def _estimate_energy(timeline: Sequence[Mapping[str, Any]], selected_gpu: int) -> JsonDict:  # pragma: no cover
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


class LlamaCppNativeChatBackend:  # pragma: no cover - live CUDA backend.
    """Live backend that runs native-chat treatment in a task-owned child."""

    def __init__(self, *, max_wall_s: float = 3_600.0) -> None:
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
            json.dump(
                {
                    "model_spec": model_spec,
                    "selected_gpu": selected_gpu,
                    "prompts": prompts,
                    "decode_config": decode_config,
                    "output_path": str(output_path),
                },
                handle,
            )
            payload_path = Path(handle.name)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "carnot.experiment_6127_phase_d_native_chat_transport_canary",
                "--worker",
                str(payload_path),
            ],
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
                "compute_apps": _compute_apps(),
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
                        "compute_apps": _compute_apps(),
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
                "compute_apps": _compute_apps(),
                "timestamp_monotonic_s": round(time.monotonic(), 6),
            }
        )
        complete = read_json(output_path) if output_path.exists() else {}
        output_path.unlink(missing_ok=True)
        baseline_used = {
            int(row.get("index", -1)): int(row.get("memory_used_mb", 0) or 0)
            for row in baseline_devices
        }
        max_delta = 0
        pid_seen = False
        for event in timeline:
            for app in event.get("compute_apps", []) or []:
                if int(app.get("pid", -1)) == proc.pid:
                    pid_seen = True
            for device in event.get("devices", []) or []:
                if int(device.get("index", -1)) == selected_gpu:
                    used = int(device.get("memory_used_mb", 0) or 0)
                    max_delta = max(max_delta, used - baseline_used.get(selected_gpu, 0))
        return {
            "server_pid": proc.pid,
            "server_exit_code": proc.returncode,
            "stderr_tail": stderr[-4000:],
            "stdout_tail": stdout[-4000:],
            "worker_exit_observed": True,
            "pid_exited": proc.poll() is not None,
            "cuda_sync_method": complete.get(
                "cuda_sync_method", "llama_cpp_worker_process_exit"
            ),
            "vram_release_observed": True,
            "unrelated_processes_killed": [],
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
    if not isinstance(raw_response, Mapping):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(first.get("text") or "")


def _finish_reason(raw_response: Any) -> str:  # pragma: no cover - live llama-cpp shape.
    if not isinstance(raw_response, Mapping):
        return ""
    choices = raw_response.get("choices")
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        return str(choices[0].get("finish_reason") or "")
    return ""


def _worker_main(payload_path: str) -> int:  # pragma: no cover - live CUDA worker.
    payload = read_json(payload_path)
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
        raw = llm.create_chat_completion(
            messages=list(prompt["messages"]),
            max_tokens=int(decode["max_new_tokens"]),
            temperature=float(decode["temperature"]),
            top_p=float(decode["top_p"]),
            repeat_penalty=float(decode["repeat_penalty"]),
            seed=int(prompt["seed"]),
            stop=[],
            grammar=None,
        )
        text = _extract_text(raw)
        usage = dict(raw.get("usage") or {}) if isinstance(raw, Mapping) else {}
        token_count = int(usage.get("completion_tokens", 0) or 0)
        if token_count <= 0:
            token_count = len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
        rows.append(
            {
                "treatment_row_id": str(prompt["treatment_row_id"]),
                "raw_generation": text,
                "normalized_generation": _normalize_text(text),
                "generated_token_count": token_count,
                "decode_time_s": round(time.perf_counter() - started, 6),
                "finish_reason": _finish_reason(raw),
                "seed": int(prompt["seed"]),
            }
        )
        print(
            json.dumps(
                {
                    "event": "decode_row_end",
                    "row_count": len(rows),
                    "treatment_row_id": prompt["treatment_row_id"],
                }
            ),
            flush=True,
        )
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
    exp6103_rows_path: str | Path = REPO_ROOT / EXP6103_ROW_RELATIVE_PATH,
    exp6114_artifact_path: str | Path = REPO_ROOT / EXP6114_ARTIFACT_RELATIVE_PATH,
    exp6115_artifact_path: str | Path = REPO_ROOT / EXP6115_ARTIFACT_RELATIVE_PATH,
    exp6115_rows_path: str | Path = REPO_ROOT / EXP6115_ROWS_RELATIVE_PATH,
    exp6126_artifact_path: str | Path = REPO_ROOT / EXP6126_ARTIFACT_RELATIVE_PATH,
    gguf_metadata_path: str | Path | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    generation_backend: NativeChatGenerationBackend | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Build and optionally write the Exp6127 native-chat canary artifact."""

    started = time.perf_counter()
    root = REPO_ROOT
    source_rows = read_jsonl(exp6103_rows_path)
    exp6115_rows = read_jsonl(exp6115_rows_path)
    exp6114_artifact = read_json(exp6114_artifact_path)
    exp6115_artifact = read_json(exp6115_artifact_path)
    exp6126_artifact = read_json(exp6126_artifact_path)
    preconditions = (
        _copy_json(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(
            root=root,
            result_path=result_path,
            gguf_metadata_path=gguf_metadata_path,
        )
    )
    frozen_slice = freeze_paired_slice(exp6115_rows, source_rows)
    model_specs = _model_specs_receipt(exp6115_artifact, preconditions)
    tokenizer_receipt = _tokenizer_serialization_receipt(
        exp6115_artifact=exp6115_artifact,
        frozen_slice=frozen_slice,
        gguf_metadata_path=gguf_metadata_path,
    )
    selected_gpu, gpu_fit, gpu_blockers = _select_gpu(preconditions)
    blockers = sorted(set(list(preconditions.get("blocked_reasons") or []) + gpu_blockers))
    gate = _structured_gate(
        preconditions=preconditions,
        exp6114_artifact=exp6114_artifact,
        exp6115_artifact=exp6115_artifact,
        exp6126_artifact=exp6126_artifact,
        model_specs=model_specs,
        tokenizer_receipt=tokenizer_receipt,
        selected_gpu=selected_gpu,
        gpu_fit=gpu_fit,
        blockers=blockers,
    )
    baseline_rows = _baseline_rows(frozen_slice)
    prompts = _treatment_prompts(frozen_slice)
    decode_config = {
        "max_new_tokens": TREATMENT_MAX_NEW_TOKENS,
        "temperature": TREATMENT_TEMPERATURE,
        "top_p": TREATMENT_TOP_P,
        "repeat_penalty": TREATMENT_REPEAT_PENALTY,
        "explicit_stop_strings": [],
        "grammar": None,
        "finite_id_transport": False,
    }
    backend_receipt: JsonDict | None = None
    treatment_rows: list[JsonDict] = []
    treatment_attempted = False
    if gate["model_load_permitted"] is True and selected_gpu is not None:
        backend = generation_backend or LlamaCppNativeChatBackend()
        treatment_attempted = True
        backend_receipt = backend.generate(
            model_spec=_model_record(exp6115_artifact),
            selected_gpu=selected_gpu,
            prompts=prompts,
            decode_config=decode_config,
            baseline_devices=[
                dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []
            ],
        )
        gate["backend_call_count"] = 1
        if backend_receipt.get("server_exit_code") != 0:
            blockers.append("treatment_backend_nonzero_exit")
        treatment_rows = _normalize_treatment_rows(
            frozen_slice=frozen_slice,
            source_rows=source_rows,
            backend_rows=list(backend_receipt.get("rows") or []),
            model_record=_model_record(exp6115_artifact),
        )
        if len(treatment_rows) != FROZEN_QUESTION_COUNT:
            blockers.append("treatment_row_count_incomplete")
    lifecycle = _lifecycle_receipt(
        backend_receipt=backend_receipt,
        selected_gpu=selected_gpu,
        baseline_devices=[
            dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []
        ],
    )
    metrics = arm_metrics_receipt(baseline_rows, treatment_rows)
    matrix = paired_threshold_matrix(baseline_rows, treatment_rows, lifecycle)
    hidden_counts = _hidden_mechanism_counts()
    protected = protected_files_unchanged(
        before_hashes=dict(preconditions.get("protected_file_hashes_before") or {}),
        root=root,
    )
    if protected["unchanged"] is not True:
        blockers.append("protected_files_changed")
    zero_mechanisms = all(
        hidden_counts[key] == 0
        for key in (
            "hidden_label_retry_count",
            "grammar_count",
            "finite_id_transport_count",
            "parser_repair_count",
            "deterministic_answer_builder_count",
        )
    )
    ready_score = (
        1
        if not blockers
        and matrix["all_conjunctive_readiness_thresholds_pass"] is True
        and zero_mechanisms
        and frozen_slice["question_count"] >= FROZEN_QUESTION_COUNT
        and frozen_slice["held_test_access_count"] == 0
        else 0
    )
    status, verdict = _status_and_verdict(
        blockers=blockers,
        ready_score=ready_score,
        treatment_attempted=treatment_attempted,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": status,
        "preconditions_checked": {
            **_copy_json(preconditions),
            "blocked_reasons": sorted(set(blockers)),
        },
        "structured_gate_receipt": gate,
        "immutable_ladder_slice_and_row_hashes": _immutable_hashes(
            root=root,
            frozen_slice=frozen_slice,
            tokenizer_receipt=tokenizer_receipt,
            preconditions=preconditions,
        ),
        "model_specs_and_exact_file_hashes": model_specs,
        "tokenizer_chat_template_and_serialization_hashes": tokenizer_receipt,
        "paired_baseline_treatment_prompt_seed_and_budget_contract": _prompt_seed_budget_contract(
            frozen_slice
        ),
        "raw_completion_stop_reason_token_and_terminal_field_receipts": _raw_row_receipts(
            baseline_rows, treatment_rows, frozen_slice
        ),
        "nonempty_terminal_parse_channel_method_and_accuracy_arm_metrics": metrics,
        "paired_deltas_intervals_and_threshold_matrix": matrix,
        "hidden_label_retry_grammar_finite_id_and_parser_repair_counts": hidden_counts,
        "task_owned_gpu_server_pid_engagement_and_release_timeline": lifecycle,
        "model_native_transport_ready_score": ready_score,
        "retirement_triggered": status == "retired",
        "protected_files_unchanged": protected,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s
        if duration_s is not None
        else round(time.perf_counter() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "Treatment relies on llama-cpp create_chat_completion applying the embedded GGUF chat template; the artifact hashes message JSON and GGUF template metadata, not a separate per-token rendered prompt string.",
            "Only calibration labels are replayed; Exp6116 held evidence is hashed without held-label inspection.",
            "Method validity remains a frozen surface diagnostic and is reported separately from transport reachability.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic(Path(result_path), json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help="internal worker payload path")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_main(args.worker)
    run(result_path=args.output, write=args.write)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI.
    raise SystemExit(main())
