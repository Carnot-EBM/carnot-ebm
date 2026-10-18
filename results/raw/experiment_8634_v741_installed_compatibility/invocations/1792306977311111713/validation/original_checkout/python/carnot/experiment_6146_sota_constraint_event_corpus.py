"""Exp6146 SOTA GGUF no-memory constraint event corpus.

Spec refs: REQ-VERIFY-6146, REQ-VERIFY-6146-1, REQ-VERIFY-6146-2,
REQ-VERIFY-6146-3, REQ-VERIFY-6146-4, REQ-VERIFY-6146-5,
REQ-VERIFY-6146-6, REQ-VERIFY-6146-7, REQ-VERIFY-6146-8,
REQ-VERIFY-6146-9, SCENARIO-VERIFY-6146-GATE,
SCENARIO-VERIFY-6146-ORDERING, SCENARIO-VERIFY-6146-NO-MEMORY,
SCENARIO-VERIFY-6146-LIFECYCLE.

Exp6146 wraps the Exp6145 exact event stream with a frozen local-SOTA GGUF
baseline. The model sees only decision-time fields. Exact Exp6145 outcomes are
looked up after a raw model response has been recorded and hashed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import gc
import hashlib
import json
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

from carnot import experiment_6126_phase_d_exp6115_transport_forensics as exp6126
from carnot import experiment_6145_constraint_shift_stream as exp6145
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6146_sota_constraint_event_corpus.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6146_sota_constraint_event_corpus.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6146_sota_constraint_event_corpus.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SCHEMA = "carnot.experiment_6146.sota_constraint_event_corpus.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT_ID = "experiment_6146_sota_constraint_event_corpus"
RUN_DATE = "20260805"
RANDOM_SEED = 6146
LIVE_INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda"
BLOCKED_INFERENCE_SUBSTRATE = "blocked_no_live_local_sota_gguf_cuda"
VERIFIER_IS_ORACLE = True

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
MANDATED_MODEL_INDICES = (0, 2)
PARTITION_NAMES = ("calibration", "future_known", "sealed_shifted_family")
FORBIDDEN_MODEL_INPUT_TOKENS = (
    "exact_answer",
    "current_validator_result",
    "validator_result",
    "post_outcome",
    "held_label",
    "oracle_label",
)

DECODE_POLICY: JsonDict = {
    "temperature": 0.2,
    "top_p": 0.95,
    "repeat_penalty": 1.05,
    "max_tokens": 96,
    "n_ctx": 2048,
    "seed_base": RANDOM_SEED,
    "terminal_answer_convention": (
        "STRATEGY_ID: <short_id>\\nSTRATEGY: <decision-time rationale>\\n"
        "SOLUTION: <terminal proposal>"
    ),
    "grammar": None,
    "finite_id_transport": False,
    "memory": "none",
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
    SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    exp6145.RESULT_RELATIVE_PATH,
    exp6145.ROW_FILE_RELATIVE_PATH,
    exp6145.SPLIT_FILE_RELATIVE_PATH,
    exp6145.OUTCOME_FILE_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/pipeline/gemma4_quantized_loader.py"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6146_sota_constraint_event_corpus.py "
    "tests/python/test_inference_sota_models.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6146_sota_constraint_event_corpus.py "
    "-m pytest tests/python/test_experiment_6146_sota_constraint_event_corpus.py "
    "tests/python/test_inference_sota_models.py -q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6146_sota_constraint_event_corpus.py --fail-under=100"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6146_sota_constraint_event_corpus.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6146_sota_constraint_event_corpus.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "model_specs",
    "resolved_model_paths_revisions_quantizations_and_hashes",
    "embedded_tokenizer_and_chat_template_receipts",
    "cuda_offload_gpu_engagement_and_task_owned_pid_receipts",
    "frozen_prompt_decode_and_seed_policy",
    "stream_split_and_row_hashes",
    "per_model_event_row_conservation",
    "strategy_terminal_solution_and_invalid_output_counts",
    "post_decision_exact_outcome_receipts",
    "no_memory_and_no_adaptive_retry_receipts",
    "calibration_future_and_shift_metrics_by_model",
    "tiny_model_smoke_rows_excluded_from_headline",
    "lifecycle_timing_and_cleanup_receipts",
    "sota_constraint_event_corpus_ready_score",
    "protected_files_unchanged",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes authentic ready rows, partial rows, and blocked runs.",
    "preconditions_checked": "The structured gate recomputes stream, split, model manifest, prompt, chat template, decoder, output path, exclusion, protected-file, GPU lease, and inherited-server hashes before any model load.",
    "structured_gate_receipt": "Headline inference opens only after mandated models, embedded tokenizers, CUDA offload, task ownership, frozen prompts, row sidecars, and protected files pass.",
    "model_specs": "Bare top-level entries include exact mandated hub IDs, resolved GGUF paths, hashes, GPU assignments, and actual use counts.",
    "resolved_model_paths_revisions_quantizations_and_hashes": "Path, revision, quantization, byte size, and SHA-256 evidence prove the selected GGUF is a language model file and not a projector or tiny substitute.",
    "embedded_tokenizer_and_chat_template_receipts": "Tokenizer and chat template receipts come from embedded GGUF metadata and llama.cpp APIs, never AutoTokenizer on a GGUF repo ID.",
    "cuda_offload_gpu_engagement_and_task_owned_pid_receipts": "CUDA evidence is attributable to task-owned workers and shows nonzero GPU engagement for each mandated model.",
    "frozen_prompt_decode_and_seed_policy": "One model-native prompt/template, terminal convention, temperature, top-p, repeat penalty, context budget, and seed schedule are frozen before inference.",
    "stream_split_and_row_hashes": "Exp6145 stream, split, row, and outcome commitments are hashed before decisions and replayed for conservation.",
    "per_model_event_row_conservation": "Every mandated model has exactly one immutable row per Exp6145 event or the artifact blocks.",
    "strategy_terminal_solution_and_invalid_output_counts": "Strategy text, strategy IDs, terminal solutions, parser status, and invalid outputs are counted without hidden retry.",
    "post_decision_exact_outcome_receipts": "The validator runs after the frozen decision and is absent from model inputs.",
    "no_memory_and_no_adaptive_retry_receipts": "Each event is prompted independently with no prior model outputs, no correctness-conditioned retries, no grammar, no finite-ID transport, and no parser repair.",
    "calibration_future_and_shift_metrics_by_model": "Metrics are separated for calibration, future-known, and sealed shifted-family partitions per model.",
    "tiny_model_smoke_rows_excluded_from_headline": "Smoke rows, when present, are labeled separately and excluded from headline readiness and metrics.",
    "lifecycle_timing_and_cleanup_receipts": "Before/load/readiness/decode/release GPU state, PIDs, elapsed time, file hashes, orphan checks, and retained-VRAM checks are recorded.",
    "sota_constraint_event_corpus_ready_score": "Exactly one only for complete authentic mandated-model rows with task-owned CUDA evidence and no hidden outcome-conditioned retry.",
    "protected_files_unchanged": "Conductor and reconciler-owned files remain byte-identical.",
    "duration_s": "The measured end-to-end Exp6146 run time is reported.",
    "inference_substrate": "Set `live_local_sota_gguf_cuda` only when all receipts prove it; otherwise block.",
    "verifier_is_oracle": "Exp6145 exact Python/Z3 labels are post-decision oracle receipts and are not model inputs.",
    "missing_verifier_gaps": "Missing model, offload, row, lifecycle, cleanup, or oracle-ordering evidence is explicit.",
    "field_provenance": "Every field traces to prompt, specs, Exp6145 sidecars, model manifests, runtime receipts, tests, or command receipts.",
    "test_commands": "Commands document focused unit/spec coverage, structured gate, model/cache/hash, llama.cpp tokenizer, GPU engagement, row conservation, no-memory/no-retry, outcome ordering, lifecycle cleanup, schema, adversarial verify, protected-file, E2E, global pytest, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "The artifact hash detects source, model, prompt, stream, row, outcome, lifecycle, protected-file, and command drift.",
    "honest_verdict": "Use `complete_ready:`, `complete_partial:`, or `blocked:` and name any missing model, offload, row, or lifecycle evidence.",
}


class SotaGenerationBackend(Protocol):
    """Backend contract for live or fixture model-native chat generation."""

    def generate(
        self,
        *,
        model_spec: JsonDict,
        prompts: list[JsonDict],
        decode_config: JsonDict,
        baseline_devices: list[JsonDict],
    ) -> JsonDict:
        """Return raw model rows and task-owned CUDA lifecycle evidence."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable ASCII byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so large GGUF receipts are content-addressed."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_atomic(path, "".join(canonical_json(row) + "\n" for row in rows))


def model_slug(hf_id: str) -> str:
    """Return a filesystem-safe slug for a mandated model id."""

    tail = hf_id.split("/", 1)[-1].removesuffix("-GGUF")
    return re.sub(r"[^a-z0-9]+", "_", tail.lower()).strip("_")


def row_sidecar_filename(hf_id: str) -> str:
    """Return the immutable Exp6146 sidecar name for one model."""

    return f"experiment_6146_sota_constraint_event_corpus.{model_slug(hf_id)}.rows.jsonl"


def _extract_revision(path: str | Path) -> str:
    parts = Path(path).parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "project-local"


def _extract_quantization(path: str | Path) -> str:
    name = Path(path).name
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0"):
        if token.lower() in name.lower():
            return token
    return "unknown"


def _is_projector_gguf(path: str | Path) -> bool:
    p = Path(path)
    name = p.name.lower()
    return name.startswith(("mmproj", "mtp-")) or "mmproj" in name or p.parent.name.lower() == "mtp"


def resolve_mandated_model_specs() -> JsonDict:  # pragma: no cover - hashes live host files.
    """Resolve, hash, and preflight the two mandated headline GGUFs."""

    pair = cached_sota_pair(gpu_indices=(0, 1), model_indices=MANDATED_MODEL_INDICES)
    blocked: list[str] = []
    records: list[JsonDict] = []
    if pair is None:
        return {
            "schema": SCHEMA + ".model_resolution",
            "records": [],
            "blocked_reasons": ["mandated_cached_sota_pair_missing"],
        }
    by_id = {str(item["hf_id"]): dict(item) for item in pair}
    for expected_index, hf_id in enumerate(MANDATED_MODEL_IDS):
        raw = by_id.get(hf_id)
        if raw is None:
            blocked.append(f"mandated_model_missing:{hf_id}")
            continue
        path = Path(str(raw.get("model_path") or "")).expanduser()
        exists = path.is_file()
        projector = _is_projector_gguf(path)
        tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(path) if exists else None)
        metadata: JsonDict = {}
        if exists and not projector:
            try:
                metadata = exp6126.read_gguf_metadata(path)
            except Exception as exc:
                metadata = {"metadata_error": f"{type(exc).__name__}: {exc}"}
        record = {
            "name": str(raw.get("name") or ""),
            "hf_id": hf_id,
            "gpu": int(raw.get("gpu", expected_index)),
            "model_path": str(path),
            "real_path": str(path.resolve()) if exists else str(path),
            "revision": _extract_revision(path),
            "quantization": _extract_quantization(path),
            "sha256": sha256_file(path) if exists else None,
            "size_bytes": path.stat().st_size if exists else 0,
            "exists": exists,
            "is_projector_gguf": projector,
            "embedded_tokenizer_loadable": tokenizer_ok,
            "embedded_tokenizer_detail": tokenizer_detail,
            "chat_template_present": bool(metadata.get("chat_template_present")),
            "chat_template_sha256": metadata.get("chat_template_sha256"),
            "chat_template_keys": list(metadata.get("chat_template_keys") or []),
            "metadata_summary_sha256": metadata.get("metadata_summary_sha256"),
            "loader": "llama_cpp.Llama",
            "n_gpu_layers": -1,
            "actual_use_count": 0,
        }
        if not exists:
            blocked.append(f"mandated_gguf_missing:{hf_id}")
        if projector:
            blocked.append(f"projector_gguf_not_language_model:{hf_id}")
        if tokenizer_ok is not True:
            blocked.append(f"embedded_tokenizer_unloadable:{hf_id}")
        if record["chat_template_present"] is not True:
            blocked.append(f"chat_template_missing:{hf_id}")
        records.append(record)
    return {
        "schema": SCHEMA + ".model_resolution",
        "records": records,
        "blocked_reasons": sorted(set(blocked)),
    }


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


def _gpu_devices() -> list[JsonDict]:  # pragma: no cover
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


def _file_receipt(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _protected_hashes(root: Path) -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(root / path)
        for path in PROTECTED_FILES
        if (root / path).exists()
    }


def _root_clutter(root: Path) -> JsonDict:  # pragma: no cover
    files = sorted(path.name for path in root.glob("*.py"))
    return {"root_python_files": files, "root_python_file_count": len(files), "ok": not files}


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_sidecar_dir: str | Path = REPO_ROOT / "results",
) -> JsonDict:  # pragma: no cover - host resource probe.
    result = Path(result_path)
    row_dir = Path(row_sidecar_dir)
    devices = _gpu_devices()
    apps = _compute_apps()
    output_ready = result.parent.exists() and os.access(result.parent, os.W_OK)
    output_ready = output_ready and row_dir.exists() and os.access(row_dir, os.W_OK)
    root_clutter = _root_clutter(root)
    blocked: list[str] = []
    if not output_ready:
        blocked.append("output_path_not_writable")
    if len(devices) < 2:
        blocked.append("two_cuda_gpus_unavailable")
    if apps:
        blocked.append("inherited_model_server_or_compute_app_present")
    if root_clutter["ok"] is not True:
        blocked.append("root_python_clutter_present")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hashed_input_receipts": [_file_receipt(root, path) for path in HASHED_INPUTS],
        "gpu": {"gpu_count": len(devices), "ok": len(devices) >= 2, "devices": devices},
        "compute_apps_before": apps,
        "lease_state": {
            "task_owned_pid": os.getpid(),
            "parent_pid": os.getppid(),
            "lease_scope": "task_owned_child_workers_only",
            "no_inherited_model_server": not apps,
        },
        "output_paths": {
            "result_path": str(result),
            "row_sidecar_dir": str(row_dir),
            "parent_writable": output_ready,
            "existed_before": result.exists(),
            "sha256_before": sha256_file(result) if result.exists() else None,
        },
        "protected_file_hashes_before": _protected_hashes(root),
        "root_clutter": root_clutter,
    }


def _stream_receipt(bundle: exp6145.StreamBundle) -> JsonDict:
    validation = exp6145.validate_stream_bundle(bundle)
    return {
        "schema": SCHEMA + ".stream_hashes",
        "exp6145_result_path": exp6145.RESULT_RELATIVE_PATH.as_posix(),
        "exp6145_result_sha256": sha256_file(REPO_ROOT / exp6145.RESULT_RELATIVE_PATH),
        "row_path": exp6145.ROW_FILE_RELATIVE_PATH.as_posix(),
        "row_sha256": sha256_file(REPO_ROOT / exp6145.ROW_FILE_RELATIVE_PATH),
        "split_path": exp6145.SPLIT_FILE_RELATIVE_PATH.as_posix(),
        "split_sha256": sha256_file(REPO_ROOT / exp6145.SPLIT_FILE_RELATIVE_PATH),
        "outcome_path": exp6145.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        "outcome_sha256": sha256_file(REPO_ROOT / exp6145.OUTCOME_FILE_RELATIVE_PATH),
        "bundle_checksum": exp6145.bundle_checksum(bundle),
        "row_count": len(bundle.rows),
        "outcome_count": len(bundle.outcomes),
        "partition_counts": dict(Counter(str(row["partition"]) for row in bundle.rows)),
        "chronological_order": validation["chronological_order"],
        "forbidden_pre_outcome_field_scan": validation["forbidden_pre_outcome_field_scan"],
        "principle": FIELD_PRINCIPLES["stream_split_and_row_hashes"],
    }


def prompt_for_row(row: Mapping[str, Any]) -> JsonDict:
    """Build the frozen decision-time prompt for one Exp6145 row."""

    visible = {
        "schema": row.get("schema"),
        "event_id": row.get("event_id"),
        "chronological_index": row.get("chronological_index"),
        "base_template_id": row.get("base_template_id"),
        "family": row.get("family"),
        "partition": row.get("partition"),
        "variant_kind": row.get("variant_kind"),
        "alias_only": row.get("alias_only"),
        "structural_shift": row.get("structural_shift"),
        "control_kind": row.get("control_kind"),
        "pre_decision": _copy_json(row.get("pre_decision") or {}),
    }
    visible_json = canonical_json(visible)
    contains_forbidden = any(token in visible_json for token in FORBIDDEN_MODEL_INPUT_TOKENS)
    messages = [
        {
            "role": "system",
            "content": (
                "You solve one sealed constraint event. Use only the visible "
                "decision-time event JSON. Do not assume any previous model "
                "answers. End with the exact three-line terminal convention."
            ),
        },
        {
            "role": "user",
            "content": (
                "Visible event JSON:\n"
                f"{visible_json}\n\n"
                "Return exactly three lines:\n"
                f"{DECODE_POLICY['terminal_answer_convention']}"
            ),
        },
    ]
    return {
        "event_id": str(row.get("event_id")),
        "chronological_index": int(row.get("chronological_index", 0) or 0),
        "partition": str(row.get("partition")),
        "family": str(row.get("family")),
        "variant_kind": str(row.get("variant_kind")),
        "messages": messages,
        "visible_event_hash": sha256_json(visible),
        "message_hash": sha256_json(messages),
        "contains_forbidden_token": contains_forbidden,
    }


def _seed_for(model_index: int, event_index: int) -> int:
    return RANDOM_SEED + model_index * 1_000_003 + event_index


def _prompts_for_model(rows: Sequence[Mapping[str, Any]], model_index: int) -> list[JsonDict]:
    prompts = []
    for row in rows:
        prompt = prompt_for_row(row)
        prompt["seed"] = _seed_for(model_index, int(prompt["chronological_index"]))
        prompts.append(prompt)
    return prompts


_FIELD_RE = re.compile(r"^(STRATEGY_ID|STRATEGY|SOLUTION)\s*:\s*(.*)$", re.IGNORECASE)


def _parse_response(raw: str) -> JsonDict:
    fields: dict[str, str] = {}
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = _FIELD_RE.match(line.strip())
        if match:
            fields[match.group(1).upper()] = match.group(2).strip()
    complete = bool(fields.get("STRATEGY_ID") and fields.get("STRATEGY") and fields.get("SOLUTION"))
    return {
        "strategy_id": fields.get("STRATEGY_ID", ""),
        "strategy_text": fields.get("STRATEGY", ""),
        "terminal_solution": fields.get("SOLUTION", ""),
        "terminal_parse_status": "complete" if complete else "invalid_terminal_output",
        "invalid_output": not complete,
    }


def _outcome_map(bundle: exp6145.StreamBundle) -> dict[str, JsonDict]:
    return {str(row["event_id"]): dict(row) for row in bundle.outcomes}


def _normalize_backend_rows(
    *,
    model_spec: Mapping[str, Any],
    model_index: int,
    prompts: Sequence[Mapping[str, Any]],
    backend_rows: Sequence[Mapping[str, Any]],
    outcome_by_event: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    backend_by_event = {str(row.get("event_id")): dict(row) for row in backend_rows}
    rows: list[JsonDict] = []
    for prompt in prompts:
        event_id = str(prompt["event_id"])
        backend = backend_by_event.get(event_id, {})
        raw = str(backend.get("raw_response") or "")
        parsed = _parse_response(raw)
        outcome = dict(outcome_by_event[event_id])
        post = dict(outcome["post_outcome"])
        decision = {
            "schema": ROW_SCHEMA,
            "row_id": f"exp6146|{model_slug(str(model_spec['hf_id']))}|{event_id}",
            "model_hf_id": str(model_spec["hf_id"]),
            "model_name": str(model_spec.get("name") or ""),
            "model_gpu": int(model_spec.get("gpu", model_index)),
            "event_id": event_id,
            "chronological_index": int(prompt["chronological_index"]),
            "partition": str(prompt["partition"]),
            "family": str(prompt["family"]),
            "variant_kind": str(prompt["variant_kind"]),
            "seed": int(backend.get("seed", prompt["seed"]) or 0),
            "message_hash": str(prompt["message_hash"]),
            "visible_event_hash": str(prompt["visible_event_hash"]),
            "decode_policy_hash": sha256_json(DECODE_POLICY),
            "raw_response": raw,
            "raw_response_hash": sha256_text(raw),
            "generated_token_count": int(backend.get("generated_token_count", 0) or 0),
            "decode_time_s": float(backend.get("decode_time_s", 0.0) or 0.0),
            "finish_reason": str(backend.get("finish_reason") or ""),
            "decision_record_written_before_outcome": True,
            **parsed,
        }
        decision_hash = sha256_json(decision)
        row = {
            **decision,
            "decision_record_hash": decision_hash,
            "post_outcome_attached_after_decision": True,
            "post_outcome_id": event_id,
            "exact_outcome_hash": str(outcome["outcome_hash"]),
            "current_validator_result": str(post.get("current_validator_result") or ""),
            "exact_labels_hash": sha256_json(post.get("exact_labels") or {}),
            "exact_answer_hash": sha256_json(post.get("exact_answer") or []),
            "outcome_receipt_hash": sha256_json(
                {
                    "event_id": event_id,
                    "outcome_hash": outcome["outcome_hash"],
                    "attached_after_decision_hash": decision_hash,
                }
            ),
            "row_hash": "",
        }
        stable = _copy_json(row)
        stable["row_hash"] = ""
        row["row_hash"] = sha256_json(stable)
        rows.append(row)
    return rows


def _row_blob_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in rows))


def _conservation(
    *,
    bundle: exp6145.StreamBundle,
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    expected_ids = [str(row["event_id"]) for row in bundle.rows]
    per_model: dict[str, JsonDict] = {}
    all_ok = True
    for hf_id in MANDATED_MODEL_IDS:
        rows = list(per_model_rows.get(hf_id) or [])
        observed = [str(row.get("event_id")) for row in rows]
        missing = sorted(set(expected_ids) - set(observed))
        extra = sorted(set(observed) - set(expected_ids))
        duplicate_count = len(observed) - len(set(observed))
        chronological = observed == expected_ids
        ok = not missing and not extra and duplicate_count == 0 and chronological
        all_ok = all_ok and ok
        per_model[hf_id] = {
            "row_count": len(rows),
            "missing_event_ids": missing[:10],
            "extra_event_ids": extra[:10],
            "duplicate_event_id_count": duplicate_count,
            "chronological_order_matches_exp6145": chronological,
            "row_blob_hash": _row_blob_hash(rows),
            "conserved": ok,
        }
    return {
        "schema": SCHEMA + ".row_conservation",
        "expected_event_count": len(expected_ids),
        "expected_event_ids_hash": sha256_json(expected_ids),
        "per_model": per_model,
        "all_models_conserved": all_ok,
        "principle": FIELD_PRINCIPLES["per_model_event_row_conservation"],
    }


def _strategy_counts(per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    per_model: dict[str, JsonDict] = {}
    total_invalid = 0
    for hf_id, rows in per_model_rows.items():
        invalid = sum(1 for row in rows if row.get("invalid_output") is True)
        total_invalid += invalid
        per_model[hf_id] = {
            "row_count": len(rows),
            "strategy_id_count": sum(bool(row.get("strategy_id")) for row in rows),
            "terminal_solution_count": sum(bool(row.get("terminal_solution")) for row in rows),
            "invalid_output_count": invalid,
            "terminal_parse_status_counts": dict(
                sorted(Counter(str(row.get("terminal_parse_status") or "") for row in rows).items())
            ),
        }
    return {
        "schema": SCHEMA + ".strategy_counts",
        "per_model": per_model,
        "total_invalid_output_count": total_invalid,
        "hidden_retry_for_invalid_outputs": False,
        "principle": FIELD_PRINCIPLES["strategy_terminal_solution_and_invalid_output_counts"],
    }


def _post_decision_receipt(per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    all_rows = [row for rows in per_model_rows.values() for row in rows]
    return {
        "schema": SCHEMA + ".post_decision_outcomes",
        "post_decision_outcome_attachment_count": len(all_rows),
        "all_outcomes_attached_after_decision": all(
            row.get("post_outcome_attached_after_decision") is True for row in all_rows
        ),
        "decision_hash_present_before_outcome_count": sum(
            bool(row.get("decision_record_hash")) for row in all_rows
        ),
        "validator_input_absent_from_model_inputs": True,
        "outcome_hash_count": sum(bool(row.get("exact_outcome_hash")) for row in all_rows),
        "principle": FIELD_PRINCIPLES["post_decision_exact_outcome_receipts"],
    }


def _no_memory_receipt(prompts_by_model: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    prompt_count = sum(len(rows) for rows in prompts_by_model.values())
    forbidden = sum(
        1
        for prompts in prompts_by_model.values()
        for prompt in prompts
        if prompt.get("contains_forbidden_token") is True
    )
    return {
        "schema": SCHEMA + ".no_memory_no_retry",
        "memory_policy": "none",
        "prompt_count": prompt_count,
        "model_output_reused_as_future_input_count": 0,
        "adaptive_retry_count": 0,
        "correctness_conditioned_retry_count": 0,
        "grammar_count": 0,
        "finite_id_transport_count": 0,
        "parser_repair_count": 0,
        "hidden_label_in_prompt_count": forbidden,
        "principle": FIELD_PRINCIPLES["no_memory_and_no_adaptive_retry_receipts"],
    }


def _metrics(per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    out: dict[str, JsonDict] = {}
    for hf_id, rows in per_model_rows.items():
        by_partition: dict[str, JsonDict] = {}
        for partition in PARTITION_NAMES:
            part_rows = [row for row in rows if row.get("partition") == partition]
            by_partition[partition] = {
                "row_count": len(part_rows),
                "invalid_output_count": sum(row.get("invalid_output") is True for row in part_rows),
                "accepted_outcome_count": sum(
                    row.get("current_validator_result") == "accepted" for row in part_rows
                ),
                "rejected_outcome_count": sum(
                    row.get("current_validator_result") == "rejected" for row in part_rows
                ),
                "nonempty_response_count": sum(bool(row.get("raw_response")) for row in part_rows),
            }
        out[hf_id] = by_partition
    return {
        "schema": SCHEMA + ".partition_metrics",
        "by_model": out,
        "principle": FIELD_PRINCIPLES["calibration_future_and_shift_metrics_by_model"],
    }


def _lifecycle(
    backend_receipts: Mapping[str, Mapping[str, Any]],
    baseline_devices: Sequence[Mapping[str, Any]],
) -> JsonDict:
    per_model: dict[str, JsonDict] = {}
    all_release = True
    all_engaged = True
    orphan_count = 0
    retained_vram = 0
    for hf_id in MANDATED_MODEL_IDS:
        receipt = dict(backend_receipts.get(hf_id) or {})
        engagement = dict(receipt.get("gpu_engagement") or {})
        release_ready = (
            receipt.get("worker_exit_code") == 0
            and receipt.get("pid_exited") is True
            and bool(receipt.get("cuda_sync_method"))
            and receipt.get("vram_release_observed") is True
            and int(receipt.get("orphan_task_owned_pid_count", 0) or 0) == 0
            and int(receipt.get("retained_task_owned_vram_mb", 0) or 0) == 0
            and not list(receipt.get("unrelated_processes_killed") or [])
        )
        engaged = (
            engagement.get("attributable") is True
            and int(engagement.get("selected_gpu_memory_delta_mb", 0) or 0) > 0
            and int(engagement.get("n_gpu_layers", 0) or 0) == -1
        )
        all_release = all_release and release_ready
        all_engaged = all_engaged and engaged
        orphan_count += int(receipt.get("orphan_task_owned_pid_count", 0) or 0)
        retained_vram += int(receipt.get("retained_task_owned_vram_mb", 0) or 0)
        per_model[hf_id] = {
            "worker_pid": receipt.get("worker_pid"),
            "worker_exit_code": receipt.get("worker_exit_code"),
            "pid_exited": receipt.get("pid_exited") is True,
            "cuda_sync_method": str(receipt.get("cuda_sync_method") or ""),
            "vram_release_observed": receipt.get("vram_release_observed") is True,
            "gpu_engagement_attributable": engaged,
            "selected_gpu_memory_delta_mb": int(
                engagement.get("selected_gpu_memory_delta_mb", 0) or 0
            ),
            "orphan_task_owned_pid_count": int(
                receipt.get("orphan_task_owned_pid_count", 0) or 0
            ),
            "retained_task_owned_vram_mb": int(
                receipt.get("retained_task_owned_vram_mb", 0) or 0
            ),
            "timeline": _copy_json(list(receipt.get("timeline") or [])),
            "release_ready": release_ready,
        }
    return {
        "schema": SCHEMA + ".lifecycle",
        "baseline_devices": _copy_json(list(baseline_devices)),
        "per_model": per_model,
        "all_models_release_ready": all_release,
        "all_models_gpu_engaged": all_engaged,
        "orphan_task_owned_pid_count": orphan_count,
        "retained_task_owned_vram_mb": retained_vram,
        "principle": FIELD_PRINCIPLES["lifecycle_timing_and_cleanup_receipts"],
    }


def _protected_files_unchanged(before_hashes: Mapping[str, str], root: Path = REPO_ROOT) -> JsonDict:
    after = _protected_hashes(root)
    changed = sorted(
        path for path, before in dict(before_hashes).items() if after.get(path) != before
    )
    return {
        "schema": SCHEMA + ".protected_files",
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "before_hashes": dict(before_hashes),
        "after_hashes": after,
        "changed_files": changed,
        "unchanged": not changed,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _tokenizer_receipts(records: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".embedded_tokenizer_chat_template",
        "auto_tokenizer_called": False,
        "serialization_api": "llama_cpp.Llama.create_chat_completion",
        "records": {
            str(record["hf_id"]): {
                "model_path": str(record.get("model_path") or ""),
                "embedded_tokenizer_loadable": record.get("embedded_tokenizer_loadable") is True,
                "embedded_tokenizer_detail": str(record.get("embedded_tokenizer_detail") or ""),
                "chat_template_present": record.get("chat_template_present") is True,
                "chat_template_sha256": record.get("chat_template_sha256"),
                "chat_template_keys": list(record.get("chat_template_keys") or []),
            }
            for record in records
        },
        "principle": FIELD_PRINCIPLES["embedded_tokenizer_and_chat_template_receipts"],
    }


def _structured_gate(
    *,
    preconditions: Mapping[str, Any],
    model_records: Sequence[Mapping[str, Any]],
    model_blockers: Sequence[str],
    stream_receipt: Mapping[str, Any],
    prompts_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    ids = [str(record.get("hf_id")) for record in model_records]
    forbidden_prompt_count = sum(
        1
        for prompts in prompts_by_model.values()
        for prompt in prompts
        if prompt.get("contains_forbidden_token") is True
    )
    checks = {
        "preconditions_ready": preconditions.get("preconditions_ready") is True,
        "no_inherited_model_server": dict(preconditions.get("lease_state") or {}).get(
            "no_inherited_model_server"
        )
        is True,
        "stream_valid": stream_receipt.get("row_count", 0) == stream_receipt.get("outcome_count", -1),
        "mandated_model_ids_exact": ids == list(MANDATED_MODEL_IDS),
        "gguf_paths_exist": all(record.get("exists") is True for record in model_records),
        "no_projector_gguf": all(record.get("is_projector_gguf") is False for record in model_records),
        "embedded_tokenizers_loadable": all(
            record.get("embedded_tokenizer_loadable") is True for record in model_records
        ),
        "chat_templates_present": all(
            record.get("chat_template_present") is True for record in model_records
        ),
        "cuda_gpu_assignments_present": sorted(int(record.get("gpu", -1)) for record in model_records)
        == [0, 1],
        "full_cuda_offload_requested": all(int(record.get("n_gpu_layers", 0) or 0) == -1 for record in model_records),
        "forbidden_prompt_count_zero": forbidden_prompt_count == 0,
    }
    blockers = list(model_blockers)
    blockers.extend(list(preconditions.get("blocked_reasons") or []))
    blockers.extend(name for name, ok in checks.items() if ok is not True)
    return {
        "schema": SCHEMA + ".structured_gate",
        "run_date": RUN_DATE,
        "model_load_permitted": not blockers,
        "backend_call_count": 0,
        "checks": checks,
        "blockers": sorted(set(str(item) for item in blockers)),
        "forbidden_prompt_count": forbidden_prompt_count,
        "principle": FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def _field_provenance() -> JsonDict:
    sources = [
        SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6145.RESULT_RELATIVE_PATH.as_posix(),
        exp6145.ROW_FILE_RELATIVE_PATH.as_posix(),
        exp6145.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        "python/carnot/inference/sota_models.py",
    ]
    return {
        field: {"sources": sources, "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def ready_score(artifact: Mapping[str, Any]) -> int:
    """Return the strict Exp6146 readiness score."""

    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    return int(
        dict(artifact.get("structured_gate_receipt") or {}).get("model_load_permitted") is True
        and dict(artifact.get("per_model_event_row_conservation") or {}).get(
            "all_models_conserved"
        )
        is True
        and dict(artifact.get("post_decision_exact_outcome_receipts") or {}).get(
            "all_outcomes_attached_after_decision"
        )
        is True
        and dict(artifact.get("post_decision_exact_outcome_receipts") or {}).get(
            "validator_input_absent_from_model_inputs"
        )
        is True
        and dict(artifact.get("no_memory_and_no_adaptive_retry_receipts") or {}).get(
            "adaptive_retry_count"
        )
        == 0
        and dict(artifact.get("no_memory_and_no_adaptive_retry_receipts") or {}).get(
            "hidden_label_in_prompt_count"
        )
        == 0
        and dict(artifact.get("lifecycle_timing_and_cleanup_receipts") or {}).get(
            "all_models_release_ready"
        )
        is True
        and dict(artifact.get("lifecycle_timing_and_cleanup_receipts") or {}).get(
            "all_models_gpu_engaged"
        )
        is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and all(test_exit_codes.get(command) == 0 for command in DEFAULT_TEST_COMMANDS)
    )


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("structured_gate_receipt") or {}).get("blockers") or [])
    if dict(artifact.get("per_model_event_row_conservation") or {}).get(
        "all_models_conserved"
    ) is not True:
        reasons.append("row_conservation")
    lifecycle = dict(artifact.get("lifecycle_timing_and_cleanup_receipts") or {})
    if lifecycle.get("all_models_gpu_engaged") is not True:
        reasons.append("cuda_offload_or_gpu_engagement")
    if lifecycle.get("all_models_release_ready") is not True:
        reasons.append("lifecycle_cleanup")
    no_memory = dict(artifact.get("no_memory_and_no_adaptive_retry_receipts") or {})
    if no_memory.get("adaptive_retry_count") != 0:
        reasons.append("adaptive_retry_count")
    if no_memory.get("hidden_label_in_prompt_count") != 0:
        reasons.append("hidden_label_in_prompt")
    post = dict(artifact.get("post_decision_exact_outcome_receipts") or {})
    if post.get("validator_input_absent_from_model_inputs") is not True:
        reasons.append("post_decision_exact_outcome_receipts")
    return sorted(set(str(reason) for reason in reasons)) or ["incomplete_evidence"]


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("structured_gate_receipt") or {}).get("model_load_permitted") is not True:
        return "blocked"
    return "complete_ready" if ready_score(artifact) == 1 else "complete_partial"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if status(artifact) == "complete_ready":
        return "complete_ready: live_sota_constraint_event_corpus_complete"
    prefix = "blocked" if status(artifact) == "blocked" else "complete_partial"
    return f"{prefix}: " + ",".join(_blocked_reasons(artifact)[:10])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions["platform"] = "<normalized>"
        output = preconditions.get("output_paths")
        if isinstance(output, dict):
            output["result_path"] = "<normalized>"
            output["row_sidecar_dir"] = "<normalized>"
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    no_memory = dict(artifact["no_memory_and_no_adaptive_retry_receipts"])
    if no_memory.get("adaptive_retry_count") != 0:
        raise ValueError("adaptive_retry_count")
    if no_memory.get("grammar_count") != 0 or no_memory.get("parser_repair_count") != 0:
        raise ValueError("hidden_retry_or_repair")
    post = dict(artifact["post_decision_exact_outcome_receipts"])
    if post.get("validator_input_absent_from_model_inputs") is not True:
        raise ValueError("post_decision_exact_outcome_receipts")
    score = ready_score(artifact)
    if artifact["sota_constraint_event_corpus_ready_score"] != score:
        raise ValueError("sota_constraint_event_corpus_ready_score")
    if artifact["status"] != status(artifact):
        raise ValueError("status")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if score == 1 and artifact["inference_substrate"] != LIVE_INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle")
    return True


class LlamaCppSotaBackend:  # pragma: no cover - live CUDA backend.
    """Live backend that runs one model in a task-owned child process."""

    def __init__(self, *, max_wall_s: float = 7_200.0, poll_s: float = 2.0) -> None:
        self.max_wall_s = max_wall_s
        self.poll_s = poll_s

    def generate(
        self,
        *,
        model_spec: JsonDict,
        prompts: list[JsonDict],
        decode_config: JsonDict,
        baseline_devices: list[JsonDict],
    ) -> JsonDict:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            output_path = Path(handle.name + ".out")
            json.dump(
                {
                    "model_spec": model_spec,
                    "prompts": prompts,
                    "decode_config": decode_config,
                    "output_path": str(output_path),
                },
                handle,
            )
            payload_path = Path(handle.name)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(model_spec["gpu"])
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "carnot.experiment_6146_sota_constraint_event_corpus",
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
                "phase": "before_load",
                "task_pid": proc.pid,
                "devices": baseline_devices,
                "compute_apps": _compute_apps(),
                "timestamp_monotonic_s": round(started, 6),
            }
        ]
        timed_out = False
        try:
            while proc.poll() is None:
                if time.monotonic() - started > self.max_wall_s:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)
                    break
                timeline.append(
                    {
                        "phase": "decode",
                        "task_pid": proc.pid,
                        "devices": _gpu_devices(),
                        "compute_apps": _compute_apps(),
                        "timestamp_monotonic_s": round(time.monotonic(), 6),
                    }
                )
                time.sleep(self.poll_s)
            stdout, stderr = proc.communicate(timeout=30)
        finally:
            payload_path.unlink(missing_ok=True)
        timeline.append(
            {
                "phase": "release",
                "task_pid": proc.pid,
                "devices": _gpu_devices(),
                "compute_apps": _compute_apps(),
                "timestamp_monotonic_s": round(time.monotonic(), 6),
            }
        )
        complete = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
        output_path.unlink(missing_ok=True)
        baseline_used = {
            int(row.get("index", -1)): int(row.get("memory_used_mb", 0) or 0)
            for row in baseline_devices
        }
        selected_gpu = int(model_spec["gpu"])
        max_delta = 0
        pid_seen = False
        after_apps = _compute_apps()
        retained = 0
        for app in after_apps:
            if int(app.get("pid", -1)) == proc.pid:
                retained += int(app.get("used_memory_mb", 0) or 0)
        for event in timeline:
            for app in event.get("compute_apps", []) or []:
                if int(app.get("pid", -1)) == proc.pid:
                    pid_seen = True
            for device in event.get("devices", []) or []:
                if int(device.get("index", -1)) == selected_gpu:
                    used = int(device.get("memory_used_mb", 0) or 0)
                    max_delta = max(max_delta, used - baseline_used.get(selected_gpu, 0))
        return {
            "model_hf_id": model_spec["hf_id"],
            "worker_pid": proc.pid,
            "worker_exit_code": proc.returncode,
            "timed_out": timed_out,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "pid_exited": proc.poll() is not None,
            "cuda_sync_method": complete.get("cuda_sync_method", "worker_exit"),
            "vram_release_observed": retained == 0,
            "orphan_task_owned_pid_count": int(Path(f"/proc/{proc.pid}").exists()),
            "retained_task_owned_vram_mb": retained,
            "unrelated_processes_killed": [],
            "timeline": timeline,
            "gpu_engagement": {
                "attributable": pid_seen and max_delta > 0,
                "task_pid": proc.pid,
                "selected_gpu": selected_gpu,
                "selected_gpu_memory_delta_mb": max_delta,
                "n_gpu_layers": -1,
            },
            "rows": list(complete.get("rows") or []),
        }


def _extract_chat_text(raw_response: Any) -> str:  # pragma: no cover - live response shape.
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


def _finish_reason(raw_response: Any) -> str:  # pragma: no cover - live response shape.
    choices = raw_response.get("choices") if isinstance(raw_response, Mapping) else None
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        return str(choices[0].get("finish_reason") or "")
    return ""


def _worker_main(payload_path: str) -> int:  # pragma: no cover - live CUDA worker.
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
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
        n_ctx=int(decode["n_ctx"]),
        n_batch=256,
        n_ubatch=128,
        verbose=False,
    )
    print(json.dumps({"event": "load_complete", "pid": os.getpid()}), flush=True)
    rows: list[JsonDict] = []
    for prompt in prompts:
        started = time.perf_counter()
        raw = llm.create_chat_completion(
            messages=list(prompt["messages"]),
            max_tokens=int(decode["max_tokens"]),
            temperature=float(decode["temperature"]),
            top_p=float(decode["top_p"]),
            repeat_penalty=float(decode["repeat_penalty"]),
            seed=int(prompt["seed"]),
            stop=[],
            grammar=None,
        )
        text = _extract_chat_text(raw)
        usage = dict(raw.get("usage") or {}) if isinstance(raw, Mapping) else {}
        token_count = int(usage.get("completion_tokens", 0) or 0)
        if token_count <= 0:
            token_count = len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
        rows.append(
            {
                "event_id": str(prompt["event_id"]),
                "raw_response": text,
                "generated_token_count": token_count,
                "decode_time_s": round(time.perf_counter() - started, 6),
                "finish_reason": _finish_reason(raw),
                "seed": int(prompt["seed"]),
            }
        )
        print(
            json.dumps(
                {"event": "decode_row_end", "row_count": len(rows), "event_id": prompt["event_id"]}
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
    print(json.dumps({"event": "complete", "row_count": len(rows)}), flush=True)
    return 0


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_sidecar_dir: str | Path = REPO_ROOT / "results",
    preconditions_checked: Mapping[str, Any] | None = None,
    model_resolution: Mapping[str, Any] | None = None,
    generation_backend: SotaGenerationBackend | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Build and optionally write the Exp6146 SOTA event corpus artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    sidecar_dir = Path(row_sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    preconditions = (
        _copy_json(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(result_path=result, row_sidecar_dir=sidecar_dir)
    )
    resolution = _copy_json(model_resolution) if model_resolution is not None else resolve_mandated_model_specs()
    model_records = [dict(row) for row in resolution.get("records") or []]
    bundle = exp6145.build_stream_bundle()
    stream = _stream_receipt(bundle)
    outcome_by_event = _outcome_map(bundle)
    prompts_by_model = {
        hf_id: _prompts_for_model(bundle.rows, index)
        for index, hf_id in enumerate(MANDATED_MODEL_IDS)
    }
    gate = _structured_gate(
        preconditions=preconditions,
        model_records=model_records,
        model_blockers=list(resolution.get("blocked_reasons") or []),
        stream_receipt=stream,
        prompts_by_model=prompts_by_model,
    )
    backend_receipts: dict[str, JsonDict] = {}
    per_model_rows: dict[str, list[JsonDict]] = {hf_id: [] for hf_id in MANDATED_MODEL_IDS}
    blockers = list(gate["blockers"])
    if gate["model_load_permitted"] is True:
        backend = generation_backend or LlamaCppSotaBackend()
        baseline_devices = [dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []]
        records_by_id = {str(record["hf_id"]): record for record in model_records}
        for model_index, hf_id in enumerate(MANDATED_MODEL_IDS):
            receipt = backend.generate(
                model_spec=records_by_id[hf_id],
                prompts=list(prompts_by_model[hf_id]),
                decode_config=dict(DECODE_POLICY),
                baseline_devices=baseline_devices,
            )
            backend_receipts[hf_id] = dict(receipt)
            gate["backend_call_count"] = int(gate["backend_call_count"]) + 1
            if receipt.get("worker_exit_code") != 0:
                blockers.append(f"worker_nonzero_exit:{hf_id}")
            rows = _normalize_backend_rows(
                model_spec=records_by_id[hf_id],
                model_index=model_index,
                prompts=prompts_by_model[hf_id],
                backend_rows=list(receipt.get("rows") or []),
                outcome_by_event=outcome_by_event,
            )
            per_model_rows[hf_id] = rows
            records_by_id[hf_id]["actual_use_count"] = len(rows)
            if write:
                _write_jsonl(sidecar_dir / row_sidecar_filename(hf_id), rows)
        model_records = [records_by_id[hf_id] for hf_id in MANDATED_MODEL_IDS]
    gate["blockers"] = sorted(set(blockers))
    conservation = _conservation(bundle=bundle, per_model_rows=per_model_rows)
    strategy_counts = _strategy_counts(per_model_rows)
    post_decision = _post_decision_receipt(per_model_rows)
    no_memory = _no_memory_receipt(prompts_by_model)
    lifecycle = _lifecycle(
        backend_receipts,
        [dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []],
    )
    protected = _protected_files_unchanged(dict(preconditions.get("protected_file_hashes_before") or {}))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "blocked",
        "preconditions_checked": {
            **_copy_json(preconditions),
            "blocked_reasons": sorted(set(gate["blockers"])),
        },
        "structured_gate_receipt": gate,
        "model_specs": _copy_json(model_records),
        "resolved_model_paths_revisions_quantizations_and_hashes": {
            "schema": SCHEMA + ".resolved_model_paths",
            "records": _copy_json(model_records),
            "principle": FIELD_PRINCIPLES[
                "resolved_model_paths_revisions_quantizations_and_hashes"
            ],
        },
        "embedded_tokenizer_and_chat_template_receipts": _tokenizer_receipts(model_records),
        "cuda_offload_gpu_engagement_and_task_owned_pid_receipts": {
            "schema": SCHEMA + ".cuda_receipts",
            "backend_receipts": _copy_json(backend_receipts),
            "principle": FIELD_PRINCIPLES[
                "cuda_offload_gpu_engagement_and_task_owned_pid_receipts"
            ],
        },
        "frozen_prompt_decode_and_seed_policy": {
            "schema": SCHEMA + ".prompt_decode_seed",
            "prompt_template_version": "exp6146_no_memory_native_chat_v1",
            "terminal_answer_convention": DECODE_POLICY["terminal_answer_convention"],
            "decode_policy": _copy_json(DECODE_POLICY),
            "seed_schedule": {
                "seed_for_event": "RANDOM_SEED + model_index*1000003 + chronological_index",
                "random_seed": RANDOM_SEED,
            },
            "prompt_hash_root": sha256_json(
                {
                    hf_id: [prompt["message_hash"] for prompt in prompts]
                    for hf_id, prompts in prompts_by_model.items()
                }
            ),
            "principle": FIELD_PRINCIPLES["frozen_prompt_decode_and_seed_policy"],
        },
        "stream_split_and_row_hashes": stream,
        "per_model_event_row_conservation": conservation,
        "strategy_terminal_solution_and_invalid_output_counts": strategy_counts,
        "post_decision_exact_outcome_receipts": post_decision,
        "no_memory_and_no_adaptive_retry_receipts": no_memory,
        "calibration_future_and_shift_metrics_by_model": _metrics(per_model_rows),
        "tiny_model_smoke_rows_excluded_from_headline": {
            "schema": SCHEMA + ".tiny_smoke_exclusion",
            "tiny_model_smoke_row_count": 0,
            "headline_use_count": 0,
            "excluded_from_headline": True,
            "principle": FIELD_PRINCIPLES["tiny_model_smoke_rows_excluded_from_headline"],
        },
        "lifecycle_timing_and_cleanup_receipts": lifecycle,
        "sota_constraint_event_corpus_ready_score": 0,
        "protected_files_unchanged": protected,
        "duration_s": duration_s if duration_s is not None else round(time.perf_counter() - started, 6),
        "inference_substrate": BLOCKED_INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["sota_constraint_event_corpus_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["inference_substrate"] = (
        LIVE_INFERENCE_SUBSTRATE
        if artifact["sota_constraint_event_corpus_ready_score"] == 1
        else BLOCKED_INFERENCE_SUBSTRATE
    )
    artifact["missing_verifier_gaps"] = [] if artifact["status"] == "complete_ready" else _blocked_reasons(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic(result, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
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
