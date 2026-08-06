"""Exp6160 fresh local-SOTA decision calibration corpus.

Spec refs: REQ-VERIFY-6160, REQ-VERIFY-6160-1, REQ-VERIFY-6160-2,
REQ-VERIFY-6160-3, REQ-VERIFY-6160-4, REQ-VERIFY-6160-5,
REQ-VERIFY-6160-6, REQ-VERIFY-6160-7, REQ-VERIFY-6160-8,
REQ-VERIFY-6160-9, REQ-VERIFY-6160-10,
SCENARIO-VERIFY-6160-GATE, SCENARIO-VERIFY-6160-ORDERING,
SCENARIO-VERIFY-6160-NO-MEMORY, SCENARIO-VERIFY-6160-NONOVERLAP.

Exp6160 is the live local-SOTA pass over the fresh Exp6159 stream. The model
sees only Exp6159 pre-outcome rows. Exact outcomes are appended after each raw
decision hash exists, making leakage and row loss detectable in the artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Protocol

from carnot import experiment_6126_phase_d_exp6115_transport_forensics as exp6126
from carnot import experiment_6146_sota_constraint_event_corpus as exp6146
from carnot import experiment_6159_decision_calibrated_stream as exp6159
from carnot.inference.sota_models import cached_sota_pair, gguf_tokenizer_loadable


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6160_sota_decision_calibration_corpus.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6160_sota_decision_calibration_corpus.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6160_sota_decision_calibration_corpus.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

SCHEMA = "carnot.experiment_6160.sota_decision_calibration_corpus.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT_ID = "experiment_6160_sota_decision_calibration_corpus"
RUN_DATE = "20260806"
RANDOM_SEED = 6160
LIVE_INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda"
BLOCKED_INFERENCE_SUBSTRATE = "blocked_no_live_local_sota_gguf_cuda"
VERIFIER_IS_ORACLE = True

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MANDATED_MODEL_INDICES = (0, 1)
PARTITION_NAMES = exp6159.PARTITIONS
LEGACY_SMOKE_IDS = ("Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it")
FORBIDDEN_MODEL_INPUT_TOKENS = (
    "exact_answer",
    "current_outcome",
    "current_validator_result",
    "future_label",
    "held_label",
    "post_outcome",
    "unsafe_label",
    "outcome_receipt",
)

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED_MODEL_IDS[0],
        "gpu": 0,
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "expected_offload": "full_cuda",
        "headline_model": True,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED_MODEL_IDS[1],
        "gpu": 1,
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "expected_offload": "full_cuda",
        "headline_model": True,
    },
]

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
    "label_conditioned_retry": False,
    "parser_repair": False,
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    exp6146.RESULT_RELATIVE_PATH,
    Path("results/experiment_6146_sota_constraint_event_corpus.qwen3_6_35b_a3b.rows.jsonl"),
    Path("results/experiment_6146_sota_constraint_event_corpus.gemma_4_31b_it.rows.jsonl"),
    exp6159.RESULT_RELATIVE_PATH,
    exp6159.ROW_FILE_RELATIVE_PATH,
    exp6159.SPLIT_FILE_RELATIVE_PATH,
    exp6159.OUTCOME_FILE_RELATIVE_PATH,
    exp6159.PREREGISTRATION_FILE_RELATIVE_PATH,
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    exp6159.RESULT_RELATIVE_PATH,
    exp6159.ROW_FILE_RELATIVE_PATH,
    exp6159.SPLIT_FILE_RELATIVE_PATH,
    exp6159.OUTCOME_FILE_RELATIVE_PATH,
    exp6159.PREREGISTRATION_FILE_RELATIVE_PATH,
    exp6146.RESULT_RELATIVE_PATH,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/pipeline/gemma4_quantized_loader.py"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6160_sota_decision_calibration_corpus.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6160_sota_decision_calibration_corpus.py "
    "-m pytest tests/python/test_experiment_6160_sota_decision_calibration_corpus.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6160_sota_decision_calibration_corpus.py --fail-under=100"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6160_sota_decision_calibration_corpus --validate"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6160_sota_decision_calibration_corpus.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6160_sota_decision_calibration_corpus.json"
)
E2E_APPLICABLE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6160_sota_decision_calibration_corpus --e2e-check"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_APPLICABLE_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "MODEL_SPECS",
    "model_specs",
    "resolved_model_paths_revisions_quantizations_and_hashes",
    "embedded_tokenizer_and_chat_template_receipts",
    "prompt_decoder_and_seed_freeze_manifest",
    "gpu_offload_pid_lifecycle_and_cleanup_receipts",
    "per_model_row_paths_hashes_and_counts",
    "raw_response_strategy_answer_and_invalid_output_counts",
    "exact_post_decision_outcome_receipts",
    "chronological_split_family_and_shift_counts",
    "row_conservation_and_prior_corpus_nonoverlap",
    "label_conditioned_retry_count",
    "memory_read_and_write_counts",
    "sota_decision_corpus_ready_score",
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
    "status": "A terminal state distinguishes ready, partial, retired, and blocked local-SOTA decision corpus runs.",
    "preconditions_checked": "The structured gate hashes Exp6159 source files, model/cache evidence, prompt settings, output paths, exclusions, protected files, GPU lease state, and inherited-server state before loading a model.",
    "structured_gate_receipt": "Headline inference opens only after Exp6159 readiness, mandated model resolution, embedded tokenizers, native chat templates, CUDA offload, task ownership, frozen prompts, row sidecars, and protected files pass.",
    "MODEL_SPECS": "The top-level manifest names exactly the mandated Qwen3.6 flagship MoE and Gemma-4-26B middle MoE GGUFs with resolved paths, hashes, loaders, GPU assignments, and expected offload.",
    "model_specs": "The lower-case compatibility manifest mirrors `MODEL_SPECS` so existing artifact readers see the same mandated models.",
    "resolved_model_paths_revisions_quantizations_and_hashes": "Path, revision, quantization, byte size, and SHA-256 evidence prove each GGUF is a language model file and not a projector or legacy smoke substitute.",
    "embedded_tokenizer_and_chat_template_receipts": "Tokenizer and chat template receipts come from embedded GGUF metadata and llama.cpp APIs, never AutoTokenizer on a GGUF repo ID.",
    "prompt_decoder_and_seed_freeze_manifest": "One native-chat prompt, terminal answer convention, temperature, top-p, repeat penalty, context budget, token budget, and deterministic seed schedule are frozen before inference.",
    "gpu_offload_pid_lifecycle_and_cleanup_receipts": "Before/load/decode/release GPU states, task-owned PIDs, offload deltas, orphan checks, and retained-VRAM checks prove real CUDA engagement and clean teardown.",
    "per_model_row_paths_hashes_and_counts": "One immutable sidecar path, row count, and content hash per model makes model rows replayable.",
    "raw_response_strategy_answer_and_invalid_output_counts": "Raw responses, strategy parse state, answer parse state, token counts, timings, and invalid outputs are counted without hidden retry.",
    "exact_post_decision_outcome_receipts": "Exact Exp6159 outcomes are attached only after each decision hash is recorded and are absent from model inputs.",
    "chronological_split_family_and_shift_counts": "Calibration, future-known, and shifted-family rows remain in Exp6159 chronological order with family and structural-shift counts visible.",
    "row_conservation_and_prior_corpus_nonoverlap": "Every Exp6159 event is conserved once per model, row IDs are unique, split hashes match, prompt leakage is zero, and Exp6146 row/event overlap is zero.",
    "label_conditioned_retry_count": "The bare value is zero because correctness-conditioned reruns would leak labels into the corpus.",
    "memory_read_and_write_counts": "Memory reads and writes are bare zeros because Exp6160 measures independent frozen model decisions.",
    "sota_decision_corpus_ready_score": "Exactly one only with both mandated models, real GPU offload, complete conserved rows, no leakage, and clean lifecycle teardown.",
    "protected_files_unchanged": "Conductor, ops, traceability, and prior-corpus protected files remain byte-identical.",
    "duration_s": "The measured end-to-end Exp6160 run time is reported.",
    "inference_substrate": "Set `live_local_sota_gguf_cuda` only when all receipts prove live local SOTA GGUF CUDA execution; otherwise block.",
    "verifier_is_oracle": "Exp6159 exact Python/Z3 labels are post-decision oracle receipts and are not model inputs.",
    "missing_verifier_gaps": "Missing model, cache, tokenizer, CUDA, row, lifecycle, leakage, nonoverlap, or cleanup evidence is explicit.",
    "field_provenance": "Every field traces to specs, Exp6159 sidecars, Exp6146 nonoverlap evidence, model manifests, runtime receipts, tests, or command receipts.",
    "test_commands": "Commands document focused unit/spec coverage, structured gate, model/cache/hash/tokenizer/chat/CUDA, prompt freeze, row conservation/nonoverlap, exact-outcome order, lifecycle cleanup, schema, adversarial verify, protected-file, E2E-applicable, global pytest, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "The artifact hash detects source, model, prompt, stream, row, outcome, lifecycle, protected-file, and command drift.",
    "honest_verdict": "Use `complete_ready:`, `complete_partial:`, `retired:`, or `blocked:` and name any model-specific parser or transport failure.",
}

canonical_json = exp6146.canonical_json
sha256_text = exp6146.sha256_text
sha256_json = exp6146.sha256_json
sha256_file = exp6146.sha256_file
model_slug = exp6146.model_slug


class SotaDecisionBackend(Protocol):
    """Backend contract for Exp6160 model-native chat generation."""

    def generate(
        self,
        *,
        model_spec: JsonDict,
        prompts: list[JsonDict],
        decode_config: JsonDict,
        baseline_devices: list[JsonDict],
    ) -> JsonDict:
        """Return raw model rows and task-owned CUDA lifecycle evidence."""


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_atomic(path, "".join(canonical_json(row) + "\n" for row in rows))


def _load_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def row_sidecar_filename(hf_id: str) -> str:
    """Return the immutable Exp6160 sidecar name for one model."""

    return f"experiment_6160_sota_decision_calibration_corpus.{model_slug(hf_id)}.rows.jsonl"


def source_stream_bundle() -> exp6159.StreamBundle:
    """Return the fresh Exp6159 stream consumed by Exp6160."""

    return exp6159.build_stream_bundle()


def resolve_mandated_model_specs() -> JsonDict:  # pragma: no cover - hashes live host files.
    """Resolve, hash, and preflight the two mandated Exp6160 headline GGUFs."""

    pair = cached_sota_pair(gpu_indices=(0, 1), model_indices=MANDATED_MODEL_INDICES)
    records: list[JsonDict] = []
    blocked: list[str] = []
    if pair is None:
        return {
            "schema": SCHEMA + ".model_resolution",
            "records": [],
            "blocked_reasons": ["mandated_cached_sota_pair_missing"],
        }
    by_id = {str(item["hf_id"]): dict(item) for item in pair}
    for expected_index, hf_id in enumerate(MANDATED_MODEL_IDS):
        template = dict(MODEL_SPECS[expected_index])
        raw = by_id.get(hf_id)
        if raw is None:
            blocked.append(f"mandated_model_missing:{hf_id}")
            continue
        path = Path(str(raw.get("model_path") or "")).expanduser()
        exists = path.is_file()
        projector = exp6146._is_projector_gguf(path)
        tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(path) if exists else None)
        metadata: JsonDict = {}
        if exists and not projector:
            try:
                metadata = exp6126.read_gguf_metadata(path)
            except Exception as exc:
                metadata = {"metadata_error": f"{type(exc).__name__}: {exc}"}
        record = {
            **template,
            "model_path": str(path),
            "real_path": str(path.resolve()) if exists else str(path),
            "revision": exp6146._extract_revision(path),
            "quantization": exp6146._extract_quantization(path),
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


def _file_receipt(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _path_hashes(paths: Sequence[Path], root: Path = REPO_ROOT) -> dict[str, str]:
    return {path.as_posix(): sha256_file(root / path) for path in paths if (root / path).exists()}


def collect_preconditions(  # pragma: no cover - host resource probe.
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_sidecar_dir: str | Path = REPO_ROOT / "results",
) -> JsonDict:
    """Hash source state and prove the host is ready before loading a model."""

    result = Path(result_path)
    row_dir = Path(row_sidecar_dir)
    devices = exp6146._gpu_devices()
    apps = exp6146._compute_apps()
    root_clutter = exp6146._root_clutter(root)
    exp6159_artifact = _read_json(root / exp6159.RESULT_RELATIVE_PATH)
    source_files = (
        exp6159.RESULT_RELATIVE_PATH,
        exp6159.ROW_FILE_RELATIVE_PATH,
        exp6159.SPLIT_FILE_RELATIVE_PATH,
        exp6159.OUTCOME_FILE_RELATIVE_PATH,
        exp6159.PREREGISTRATION_FILE_RELATIVE_PATH,
    )
    output_ready = result.parent.exists() and os.access(result.parent, os.W_OK)
    output_ready = output_ready and row_dir.exists() and os.access(row_dir, os.W_OK)
    checks = {
        "exp6159_ready": exp6159_artifact.get("decision_calibrated_stream_ready_score") == 1.0,
        "exp6159_source_files_present": all((root / path).exists() for path in source_files),
        "two_cuda_gpus_available": len(devices) >= 2,
        "no_inherited_model_server": not apps,
        "output_paths_writable": output_ready,
        "root_clutter_absent": root_clutter["ok"] is True,
        "exclusion_manifest_present": (root / EXCLUSION_MANIFEST_RELATIVE_PATH).exists(),
        "protected_files_present": all((root / path).exists() for path in PROTECTED_FILES),
    }
    blocked = [name for name, ok in checks.items() if ok is not True]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blocked,
        "blocked_reasons": blocked,
        "checks": checks,
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
        "protected_file_hashes_before": _path_hashes(PROTECTED_FILES, root),
        "root_clutter": root_clutter,
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def _source_stream_receipt(bundle: exp6159.StreamBundle) -> JsonDict:
    validation = exp6159.validate_stream_bundle(bundle)
    sidecar_receipt = exp6159.replay_sidecars(
        REPO_ROOT / exp6159.ROW_FILE_RELATIVE_PATH,
        REPO_ROOT / exp6159.SPLIT_FILE_RELATIVE_PATH,
        REPO_ROOT / exp6159.OUTCOME_FILE_RELATIVE_PATH,
        REPO_ROOT / exp6159.PREREGISTRATION_FILE_RELATIVE_PATH,
    )
    return {
        "schema": SCHEMA + ".exp6159_source_stream",
        "exp6159_result_path": exp6159.RESULT_RELATIVE_PATH.as_posix(),
        "exp6159_result_sha256": sha256_file(REPO_ROOT / exp6159.RESULT_RELATIVE_PATH),
        "row_path": exp6159.ROW_FILE_RELATIVE_PATH.as_posix(),
        "row_sha256": sidecar_receipt["row_sha256"],
        "split_path": exp6159.SPLIT_FILE_RELATIVE_PATH.as_posix(),
        "split_sha256": sidecar_receipt["split_sha256"],
        "outcome_path": exp6159.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        "outcome_sha256": sidecar_receipt["outcome_sha256"],
        "preregistration_path": exp6159.PREREGISTRATION_FILE_RELATIVE_PATH.as_posix(),
        "preregistration_sha256": sidecar_receipt["preregistration_sha256"],
        "bundle_checksum": exp6159.bundle_checksum(bundle),
        "validation_bundle_checksum": validation["bundle_checksum"],
        "row_count": len(bundle.rows),
        "outcome_count": len(bundle.outcomes),
        "partition_counts": dict(Counter(str(row["partition"]) for row in bundle.rows)),
        "chronological_order": validation["chronological_order"],
        "forbidden_field_scan": validation["forbidden_field_scan"],
        "preregistration_hash": bundle.preregistration["preregistration_hash"],
    }


def prompt_for_row(row: Mapping[str, Any]) -> JsonDict:
    """Build the frozen decision-time prompt for one Exp6159 row."""

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
                "You solve one fresh sealed constraint event. Use only the visible "
                "decision-time event JSON. Do not use memory or previous answers. "
                "End with the exact three-line terminal convention."
            ),
        },
        {
            "role": "user",
            "content": (
                "Visible Exp6159 event JSON:\n"
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
        "structural_shift": row.get("structural_shift") is True,
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


def _outcome_map(bundle: exp6159.StreamBundle) -> dict[str, JsonDict]:
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
        parsed = exp6146._parse_response(raw)
        outcome = dict(outcome_by_event[event_id])
        post = dict(outcome["post_outcome"])
        decision = {
            "schema": ROW_SCHEMA,
            "row_id": f"exp6160|{model_slug(str(model_spec['hf_id']))}|{event_id}",
            "model_hf_id": str(model_spec["hf_id"]),
            "model_name": str(model_spec.get("name") or ""),
            "model_gpu": int(model_spec.get("gpu", model_index)),
            "event_id": event_id,
            "chronological_index": int(prompt["chronological_index"]),
            "partition": str(prompt["partition"]),
            "family": str(prompt["family"]),
            "variant_kind": str(prompt["variant_kind"]),
            "structural_shift": prompt.get("structural_shift") is True,
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
            "strategy_id": parsed["strategy_id"],
            "strategy_text": parsed["strategy_text"],
            "answer": parsed["terminal_solution"],
            "strategy_parse_state": (
                "complete" if parsed["strategy_id"] and parsed["strategy_text"] else "invalid"
            ),
            "answer_parse_state": parsed["terminal_parse_status"],
            "terminal_solution": parsed["terminal_solution"],
            "terminal_parse_status": parsed["terminal_parse_status"],
            "invalid_output": parsed["invalid_output"],
        }
        decision_hash = sha256_json(decision)
        row = {
            **decision,
            "decision_record_hash": decision_hash,
            "post_outcome_attached_after_decision": True,
            "post_outcome_id": event_id,
            "exact_outcome_hash": str(outcome["outcome_hash"]),
            "current_outcome": str(post.get("current_outcome") or ""),
            "unsafe_label": int(post.get("unsafe_label", 0) or 0),
            "future_label_hash": sha256_json(post.get("future_label")),
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


def _prior_exp6146_nonoverlap(
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    prior_rows: list[JsonDict] = []
    for hf_id in exp6146.MANDATED_MODEL_IDS:
        prior_rows.extend(_load_jsonl(REPO_ROOT / "results" / exp6146.row_sidecar_filename(hf_id)))
    new_rows = [row for rows in per_model_rows.values() for row in rows]
    prior_event_ids = {str(row.get("event_id")) for row in prior_rows}
    prior_row_ids = {str(row.get("row_id")) for row in prior_rows}
    prior_decisions = {str(row.get("decision_record_hash")) for row in prior_rows}
    new_event_ids = {str(row.get("event_id")) for row in new_rows}
    new_row_ids = {str(row.get("row_id")) for row in new_rows}
    new_decisions = {str(row.get("decision_record_hash")) for row in new_rows}
    counts = {
        "event_id_overlap_count": len(new_event_ids & prior_event_ids),
        "row_id_overlap_count": len(new_row_ids & prior_row_ids),
        "decision_hash_overlap_count": len(new_decisions & prior_decisions),
    }
    return {
        **counts,
        "prior_exp6146_row_count": len(prior_rows),
        "new_row_count": len(new_rows),
        "all_overlap_counts_zero": all(value == 0 for value in counts.values()),
        "prior_rows_hash": sha256_json(
            [row.get("row_hash") for row in prior_rows if row.get("row_hash")]
        ),
        "new_rows_hash": sha256_json([row.get("row_hash") for row in new_rows]),
    }


def _conservation(
    *,
    bundle: exp6159.StreamBundle,
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_stream: Mapping[str, Any],
    prompts_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    expected_ids = [str(row["event_id"]) for row in bundle.rows]
    expected_partitions = {str(row["event_id"]): str(row["partition"]) for row in bundle.rows}
    prompt_leakage = sum(
        1
        for prompts in prompts_by_model.values()
        for prompt in prompts
        if prompt.get("contains_forbidden_token") is True
    )
    per_model: dict[str, JsonDict] = {}
    all_ok = True
    for hf_id in MANDATED_MODEL_IDS:
        rows = list(per_model_rows.get(hf_id) or [])
        observed = [str(row.get("event_id")) for row in rows]
        row_ids = [str(row.get("row_id")) for row in rows]
        missing = sorted(set(expected_ids) - set(observed))
        extra = sorted(set(observed) - set(expected_ids))
        duplicate_event_count = len(observed) - len(set(observed))
        duplicate_row_count = len(row_ids) - len(set(row_ids))
        chronological = observed == expected_ids
        split_match = all(
            str(row.get("partition")) == expected_partitions.get(str(row.get("event_id")))
            for row in rows
        )
        ok = (
            not missing
            and not extra
            and duplicate_event_count == 0
            and duplicate_row_count == 0
            and chronological
            and split_match
        )
        all_ok = all_ok and ok
        per_model[hf_id] = {
            "row_count": len(rows),
            "missing_event_ids": missing[:10],
            "extra_event_ids": extra[:10],
            "duplicate_event_id_count": duplicate_event_count,
            "duplicate_row_id_count": duplicate_row_count,
            "chronological_order_matches_exp6159": chronological,
            "split_assignment_matches_exp6159": split_match,
            "row_blob_hash": _row_blob_hash(rows),
            "conserved": ok,
        }
    nonoverlap = _prior_exp6146_nonoverlap(per_model_rows)
    return {
        "schema": SCHEMA + ".row_conservation_nonoverlap",
        "expected_event_count": len(expected_ids),
        "expected_event_ids_hash": sha256_json(expected_ids),
        "exp6159_split_sha256": source_stream["split_sha256"],
        "exp6159_preregistration_sha256": source_stream["preregistration_sha256"],
        "per_model": per_model,
        "all_models_conserved": all_ok,
        "prompt_outcome_leakage_count": prompt_leakage,
        "prior_exp6146_nonoverlap": nonoverlap,
        "all_checks_pass": all_ok and prompt_leakage == 0 and nonoverlap["all_overlap_counts_zero"],
        "principle": FIELD_PRINCIPLES["row_conservation_and_prior_corpus_nonoverlap"],
    }


def _strategy_counts(per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    per_model: dict[str, JsonDict] = {}
    total_invalid = 0
    for hf_id, rows in per_model_rows.items():
        invalid = sum(1 for row in rows if row.get("invalid_output") is True)
        total_invalid += invalid
        per_model[hf_id] = {
            "row_count": len(rows),
            "raw_response_count": sum(bool(row.get("raw_response")) for row in rows),
            "strategy_id_count": sum(bool(row.get("strategy_id")) for row in rows),
            "answer_count": sum(bool(row.get("answer")) for row in rows),
            "generated_token_count": sum(
                int(row.get("generated_token_count", 0) or 0) for row in rows
            ),
            "decode_time_s": round(
                sum(float(row.get("decode_time_s", 0.0) or 0.0) for row in rows),
                6,
            ),
            "invalid_output_count": invalid,
            "strategy_parse_state_counts": dict(
                sorted(Counter(str(row.get("strategy_parse_state") or "") for row in rows).items())
            ),
            "answer_parse_state_counts": dict(
                sorted(Counter(str(row.get("answer_parse_state") or "") for row in rows).items())
            ),
        }
    return {
        "schema": SCHEMA + ".raw_response_strategy_answer_counts",
        "per_model": per_model,
        "total_invalid_output_count": total_invalid,
        "label_conditioned_retry_count": 0,
        "principle": FIELD_PRINCIPLES["raw_response_strategy_answer_and_invalid_output_counts"],
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
        "principle": FIELD_PRINCIPLES["exact_post_decision_outcome_receipts"],
    }


def _chronological_counts(
    bundle: exp6159.StreamBundle,
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    expected_ids = [str(row["event_id"]) for row in bundle.rows]
    source_partitions = Counter(str(row["partition"]) for row in bundle.rows)
    source_families = Counter(str(row["family"]) for row in bundle.rows)
    per_model = {}
    for hf_id, rows in per_model_rows.items():
        per_model[hf_id] = {
            "row_count": len(rows),
            "partition_counts": {
                name: sum(str(row.get("partition")) == name for row in rows)
                for name in PARTITION_NAMES
            },
            "chronological_order_matches_exp6159": [str(row.get("event_id")) for row in rows]
            == expected_ids,
        }
    return {
        "schema": SCHEMA + ".chronological_counts",
        "source_event_count": len(bundle.rows),
        "source_partition_counts": {
            name: source_partitions.get(name, 0) for name in PARTITION_NAMES
        },
        "source_family_counts": dict(sorted(source_families.items())),
        "source_family_count": len(source_families),
        "structural_shift_event_count": sum(
            row.get("structural_shift") is True for row in bundle.rows
        ),
        "alias_counted_as_shift_count": sum(
            row.get("alias_only") is True and row.get("structural_shift") is True
            for row in bundle.rows
        ),
        "chronological_order_matches_exp6159": all(
            data["chronological_order_matches_exp6159"] for data in per_model.values()
        ),
        "per_model": per_model,
        "principle": FIELD_PRINCIPLES["chronological_split_family_and_shift_counts"],
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
    transport_failures: list[str] = []
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
        if receipt and receipt.get("worker_exit_code") != 0:
            transport_failures.append(f"worker_nonzero_exit:{hf_id}")
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
            "orphan_task_owned_pid_count": int(receipt.get("orphan_task_owned_pid_count", 0) or 0),
            "retained_task_owned_vram_mb": int(receipt.get("retained_task_owned_vram_mb", 0) or 0),
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
        "model_specific_transport_failures": transport_failures,
        "principle": FIELD_PRINCIPLES["gpu_offload_pid_lifecycle_and_cleanup_receipts"],
    }


def _protected_files_unchanged(
    before_hashes: Mapping[str, str], root: Path = REPO_ROOT
) -> JsonDict:
    after = _path_hashes(PROTECTED_FILES, root)
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
    source_stream: Mapping[str, Any],
    prompts_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    ids = [str(record.get("hf_id")) for record in model_records]
    prompt_forbidden = sum(
        1
        for prompts in prompts_by_model.values()
        for prompt in prompts
        if prompt.get("contains_forbidden_token") is True
    )
    legacy_present = any(legacy in canonical_json(model_records) for legacy in LEGACY_SMOKE_IDS)
    checks = {
        "preconditions_ready": preconditions.get("preconditions_ready") is True,
        "no_inherited_model_server": dict(preconditions.get("lease_state") or {}).get(
            "no_inherited_model_server"
        )
        is True,
        "exp6159_ready_source": source_stream.get("row_count") == 240
        and source_stream.get("outcome_count") == 240,
        "exp6159_preregistration_hashed": bool(source_stream.get("preregistration_sha256")),
        "mandated_model_ids_exact": ids == list(MANDATED_MODEL_IDS),
        "legacy_smoke_models_absent_from_headline": not legacy_present,
        "gguf_paths_exist": all(record.get("exists") is True for record in model_records),
        "no_projector_gguf": all(
            record.get("is_projector_gguf") is False for record in model_records
        ),
        "embedded_tokenizers_loadable": all(
            record.get("embedded_tokenizer_loadable") is True for record in model_records
        ),
        "chat_templates_present": all(
            record.get("chat_template_present") is True for record in model_records
        ),
        "cuda_gpu_assignments_present": sorted(
            int(record.get("gpu", -1)) for record in model_records
        )
        == [0, 1],
        "full_cuda_offload_requested": all(
            int(record.get("n_gpu_layers", 0) or 0) == -1 for record in model_records
        ),
        "frozen_prompt_forbidden_count_zero": prompt_forbidden == 0,
        "label_conditioned_retry_disabled": DECODE_POLICY["label_conditioned_retry"] is False,
        "memory_disabled": DECODE_POLICY["memory"] == "none",
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
        "forbidden_prompt_count": prompt_forbidden,
        "principle": FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def _row_sidecar_receipts(
    *,
    sidecar_dir: Path,
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    written_this_run: bool,
) -> JsonDict:
    per_model: dict[str, JsonDict] = {}
    for hf_id in MANDATED_MODEL_IDS:
        path = sidecar_dir / row_sidecar_filename(hf_id)
        rows = list(per_model_rows.get(hf_id) or [])
        per_model[hf_id] = {
            "path": str(path),
            "exists": path.exists(),
            "written_this_run": written_this_run and bool(rows),
            "sha256": sha256_file(path) if path.exists() else None,
            "row_count": len(rows),
            "row_blob_hash": _row_blob_hash(rows),
            "schema": ROW_SCHEMA,
        }
    return {
        "schema": SCHEMA + ".row_sidecars",
        "per_model": per_model,
        "total_row_count": sum(receipt["row_count"] for receipt in per_model.values()),
        "principle": FIELD_PRINCIPLES["per_model_row_paths_hashes_and_counts"],
    }


def _prompt_freeze_manifest(
    prompts_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".prompt_decoder_seed_freeze",
        "prompt_template_version": "exp6160_fresh_no_memory_native_chat_v1",
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
        "outcome_tokens_forbidden": list(FORBIDDEN_MODEL_INPUT_TOKENS),
        "principle": FIELD_PRINCIPLES["prompt_decoder_and_seed_freeze_manifest"],
    }


def _memory_receipt() -> JsonDict:
    return {
        "memory_read_count": 0,
        "memory_write_count": 0,
        "memory_policy": "none",
        "principle": FIELD_PRINCIPLES["memory_read_and_write_counts"],
    }


def _field_provenance() -> JsonDict:
    sources = [
        SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6159.RESULT_RELATIVE_PATH.as_posix(),
        exp6159.ROW_FILE_RELATIVE_PATH.as_posix(),
        exp6159.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        exp6159.PREREGISTRATION_FILE_RELATIVE_PATH.as_posix(),
        exp6146.RESULT_RELATIVE_PATH.as_posix(),
        "python/carnot/inference/sota_models.py",
    ]
    return {
        field: {"sources": sources, "principle": FIELD_PRINCIPLES[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the strict Exp6160 readiness score."""

    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    conservation = dict(artifact.get("row_conservation_and_prior_corpus_nonoverlap") or {})
    lifecycle = dict(artifact.get("gpu_offload_pid_lifecycle_and_cleanup_receipts") or {})
    post = dict(artifact.get("exact_post_decision_outcome_receipts") or {})
    memory = dict(artifact.get("memory_read_and_write_counts") or {})
    return float(
        dict(artifact.get("structured_gate_receipt") or {}).get("model_load_permitted") is True
        and conservation.get("all_checks_pass") is True
        and post.get("all_outcomes_attached_after_decision") is True
        and post.get("validator_input_absent_from_model_inputs") is True
        and artifact.get("label_conditioned_retry_count") == 0
        and memory.get("memory_read_count") == 0
        and memory.get("memory_write_count") == 0
        and lifecycle.get("all_models_release_ready") is True
        and lifecycle.get("all_models_gpu_engaged") is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and all(test_exit_codes.get(command) == 0 for command in DEFAULT_TEST_COMMANDS)
    )


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("structured_gate_receipt") or {}).get("blockers") or [])
    if (
        dict(artifact.get("row_conservation_and_prior_corpus_nonoverlap") or {}).get(
            "all_checks_pass"
        )
        is not True
    ):
        reasons.append("row_conservation_and_prior_corpus_nonoverlap")
    lifecycle = dict(artifact.get("gpu_offload_pid_lifecycle_and_cleanup_receipts") or {})
    if lifecycle.get("all_models_gpu_engaged") is not True:
        reasons.append("cuda_offload_or_gpu_engagement")
    if lifecycle.get("all_models_release_ready") is not True:
        reasons.append("lifecycle_cleanup")
    reasons.extend(list(lifecycle.get("model_specific_transport_failures") or []))
    if artifact.get("label_conditioned_retry_count") != 0:
        reasons.append("label_conditioned_retry_count")
    memory = dict(artifact.get("memory_read_and_write_counts") or {})
    if memory.get("memory_read_count") != 0 or memory.get("memory_write_count") != 0:
        reasons.append("memory_read_and_write_counts")
    post = dict(artifact.get("exact_post_decision_outcome_receipts") or {})
    if post.get("validator_input_absent_from_model_inputs") is not True:
        reasons.append("exact_post_decision_outcome_receipts")
    return sorted(set(str(reason) for reason in reasons)) or ["incomplete_evidence"]


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("structured_gate_receipt") or {}).get("model_load_permitted") is not True:
        return "blocked"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_partial"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if status(artifact) == "complete_ready":
        return "complete_ready: live_sota_decision_calibration_corpus_complete"
    prefix = "blocked" if status(artifact) == "blocked" else "complete_partial"
    return f"{prefix}: " + ",".join(_blocked_reasons(artifact)[:12])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions["platform"] = "<normalized>"
        output = preconditions.get("output_paths")
        if isinstance(output, dict):
            output["result_path"] = "<normalized>"
            output["row_sidecar_dir"] = "<normalized>"
            output["sha256_before"] = "<normalized>"
    rows = stable.get("per_model_row_paths_hashes_and_counts")
    if isinstance(rows, dict):
        for receipt in dict(rows.get("per_model") or {}).values():
            if isinstance(receipt, dict):
                receipt["path"] = "<normalized>"
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
    if [row.get("hf_id") for row in artifact["MODEL_SPECS"]] != list(MANDATED_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    if artifact["MODEL_SPECS"] != artifact["model_specs"]:
        raise ValueError("model_specs")
    if artifact["label_conditioned_retry_count"] != 0:
        raise ValueError("label_conditioned_retry_count")
    memory = dict(artifact["memory_read_and_write_counts"])
    if memory.get("memory_read_count") != 0 or memory.get("memory_write_count") != 0:
        raise ValueError("memory_read_and_write_counts")
    conservation = dict(artifact["row_conservation_and_prior_corpus_nonoverlap"])
    if conservation.get("prompt_outcome_leakage_count") != 0 or (
        artifact["sota_decision_corpus_ready_score"] == 1.0
        and conservation.get("all_checks_pass") is not True
    ):
        raise ValueError("row_conservation_and_prior_corpus_nonoverlap")
    post = dict(artifact["exact_post_decision_outcome_receipts"])
    if post.get("validator_input_absent_from_model_inputs") is not True:
        raise ValueError("exact_post_decision_outcome_receipts")
    if artifact["sota_decision_corpus_ready_score"] != ready_score(artifact):
        raise ValueError("sota_decision_corpus_ready_score")
    if artifact["status"] != status(artifact):
        raise ValueError("status")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact["sota_decision_corpus_ready_score"] == 1.0:
        if artifact["inference_substrate"] != LIVE_INFERENCE_SUBSTRATE:
            raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle")
    return True


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_sidecar_dir: str | Path = REPO_ROOT / "results",
    preconditions_checked: Mapping[str, Any] | None = None,
    model_resolution: Mapping[str, Any] | None = None,
    generation_backend: SotaDecisionBackend | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Build and optionally write the Exp6160 SOTA decision corpus artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    sidecar_dir = Path(row_sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    preconditions = (
        _copy_json(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(result_path=result, row_sidecar_dir=sidecar_dir)
    )
    resolution = (
        _copy_json(model_resolution)
        if model_resolution is not None
        else resolve_mandated_model_specs()
    )
    model_records = [dict(row) for row in resolution.get("records") or []]
    bundle = source_stream_bundle()
    source_stream = _source_stream_receipt(bundle)
    outcome_by_event = _outcome_map(bundle)
    prompts_by_model = {
        hf_id: _prompts_for_model(bundle.rows, index)
        for index, hf_id in enumerate(MANDATED_MODEL_IDS)
    }
    gate = _structured_gate(
        preconditions=preconditions,
        model_records=model_records,
        model_blockers=list(resolution.get("blocked_reasons") or []),
        source_stream=source_stream,
        prompts_by_model=prompts_by_model,
    )
    backend_receipts: dict[str, JsonDict] = {}
    per_model_rows: dict[str, list[JsonDict]] = {hf_id: [] for hf_id in MANDATED_MODEL_IDS}
    blockers = list(gate["blockers"])
    wrote_rows = False
    if gate["model_load_permitted"] is True:
        backend = generation_backend or exp6146.LlamaCppSotaBackend()
        baseline_devices = [
            dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []
        ]
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
                wrote_rows = True
        model_records = [records_by_id[hf_id] for hf_id in MANDATED_MODEL_IDS]
    gate["blockers"] = sorted(set(blockers))
    conservation = _conservation(
        bundle=bundle,
        per_model_rows=per_model_rows,
        source_stream=source_stream,
        prompts_by_model=prompts_by_model,
    )
    protected = _protected_files_unchanged(
        dict(preconditions.get("protected_file_hashes_before") or {})
    )
    lifecycle = _lifecycle(
        backend_receipts,
        [dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []],
    )
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
        "MODEL_SPECS": _copy_json(model_records),
        "model_specs": _copy_json(model_records),
        "resolved_model_paths_revisions_quantizations_and_hashes": {
            "schema": SCHEMA + ".resolved_model_paths",
            "records": _copy_json(model_records),
            "principle": FIELD_PRINCIPLES[
                "resolved_model_paths_revisions_quantizations_and_hashes"
            ],
        },
        "embedded_tokenizer_and_chat_template_receipts": _tokenizer_receipts(model_records),
        "prompt_decoder_and_seed_freeze_manifest": _prompt_freeze_manifest(prompts_by_model),
        "gpu_offload_pid_lifecycle_and_cleanup_receipts": lifecycle,
        "per_model_row_paths_hashes_and_counts": _row_sidecar_receipts(
            sidecar_dir=sidecar_dir,
            per_model_rows=per_model_rows,
            written_this_run=wrote_rows,
        ),
        "raw_response_strategy_answer_and_invalid_output_counts": _strategy_counts(per_model_rows),
        "exact_post_decision_outcome_receipts": _post_decision_receipt(per_model_rows),
        "chronological_split_family_and_shift_counts": _chronological_counts(
            bundle,
            per_model_rows,
        ),
        "row_conservation_and_prior_corpus_nonoverlap": conservation,
        "label_conditioned_retry_count": 0,
        "memory_read_and_write_counts": _memory_receipt(),
        "sota_decision_corpus_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "duration_s": duration_s
        if duration_s is not None
        else round(time.perf_counter() - started, 6),
        "inference_substrate": BLOCKED_INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["sota_decision_corpus_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["inference_substrate"] = (
        LIVE_INFERENCE_SUBSTRATE
        if artifact["sota_decision_corpus_ready_score"] == 1.0
        else BLOCKED_INFERENCE_SUBSTRATE
    )
    artifact["missing_verifier_gaps"] = (
        [] if artifact["status"] == "complete_ready" else _blocked_reasons(artifact)
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic(result, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def _validate_checked_artifact() -> JsonDict:  # pragma: no cover - CLI helper.
    artifact = _read_json(REPO_ROOT / RESULT_RELATIVE_PATH)
    validate_artifact(artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--e2e-check", action="store_true")
    args = parser.parse_args(argv)
    if args.validate or args.e2e_check:
        _validate_checked_artifact()
        return 0
    run(result_path=args.output, write=args.write)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI.
    raise SystemExit(main())
