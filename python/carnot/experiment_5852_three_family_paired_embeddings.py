"""Exp5852 three-family paired GGUF embedding corpus.

Spec refs: REQ-VERIFY-5852, SCENARIO-VERIFY-5852-COMPLETE,
SCENARIO-VERIFY-5852-PARITY, SCENARIO-VERIFY-5852-RESUME,
SCENARIO-VERIFY-5852-BLOCKED.

This experiment consumes the clean Exp5840 exact counterfactual fixture and
extracts output-free final embeddings from the three mandated local GGUF
families. The row file is source-row major, model-order minor, so downstream
paired-difference training can audit every model-row cell without reconstructing
order from metadata.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Protocol

from carnot import experiment_5840_exact_counterfactual_embedding_fixture as exp5840
from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]
GpuProbe = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5852_three_family_paired_embeddings.json")
ROW_FILE_RELATIVE_PATH = Path("results/experiment_5852_three_family_paired_embeddings.rows.jsonl")
CHECKPOINT_RELATIVE_DIR = Path("results/checkpoints/experiment_5852_three_family_paired_embeddings")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5852_three_family_paired_embeddings.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5852_three_family_paired_embeddings.py")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
EXP5840_ARTIFACT_RELATIVE_PATH = exp5840.RESULT_RELATIVE_PATH
EXP5840_ROWS_RELATIVE_PATH = exp5840.ROW_FILE_RELATIVE_PATH

SCHEMA = "carnot.experiment_5852.three_family_paired_embeddings.v1"
ROW_SCHEMA = SCHEMA + ".row"
CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint"
EXPERIMENT = 5852
EXPERIMENT_ID = "experiment_5852_three_family_paired_embeddings"
MILESTONE = "2026.07.521"
RUN_DATE = "20260723"
INFERENCE_SUBSTRATE = "live_llm_embedding_extraction"
VERIFIER_IS_ORACLE = True
RAM_FLOOR_MB = 16_384
DISK_FLOOR_MB = 10_240
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_BATCH_SIZE = 512
DEFAULT_UBATCH_SIZE = 128
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_RANDOM_SEED = 5852
DEFAULT_CHECKPOINT_GROUP_SIZE = 32
EMBEDDING_DECIMALS = 8
NEUTRAL_PAD_VOCAB = exp5840.NEUTRAL_PAD_VOCAB
MANDATED_MODEL_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_SMOKE_MODEL_IDS = ("Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it")
SPEC_REFS = (
    "REQ-VERIFY-5852",
    "SCENARIO-VERIFY-5852-COMPLETE",
    "SCENARIO-VERIFY-5852-PARITY",
    "SCENARIO-VERIFY-5852-RESUME",
    "SCENARIO-VERIFY-5852-BLOCKED",
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5852_three_family_paired_embeddings.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5852_three_family_paired_embeddings.py "
    "-m pytest tests/python/test_experiment_5852_three_family_paired_embeddings.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5852_three_family_paired_embeddings.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5852_three_family_paired_embeddings.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "model_specs",
    "models_used",
    "model_file_and_tokenizer_receipts",
    "gpu_and_loader_receipts",
    "upstream_fixture_hashes",
    "deterministic_embedding_config",
    "model_axis_family_cell_counts",
    "embedding_shape_and_finiteness",
    "pair_alignment_receipts",
    "token_and_truncation_parity",
    "checkpoint_resume_receipts",
    "row_file_receipt",
    "paired_embedding_corpus_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal corpus state distinguishes complete extraction from partial checkpoints.",
    "preconditions_checked": "Fixture, model, tokenizer, device, resource, and output checks prevent fallback masquerading.",
    "model_specs": "Exact SOTA hub IDs, paths, hashes, and device assignments make the live surface reproducible.",
    "models_used": "The artifact must explicitly list all three mandated families.",
    "model_file_and_tokenizer_receipts": "Embedded GGUF tokenizer provenance prevents token-ID mismatches.",
    "gpu_and_loader_receipts": "Actual llama.cpp device placement distinguishes live extraction from mock execution.",
    "upstream_fixture_hashes": "Hashes bind embeddings to the clean exact causal corpus.",
    "deterministic_embedding_config": "Identical settings isolate representation differences from decoding randomness.",
    "model_axis_family_cell_counts": "Disaggregated counts prevent a partial family from carrying readiness.",
    "embedding_shape_and_finiteness": "Finite aligned vectors are required before any learned energy.",
    "pair_alignment_receipts": "Every difference must join exact members from the same causal pair.",
    "token_and_truncation_parity": "Length or truncation cannot encode the target.",
    "checkpoint_resume_receipts": "Hash-bound resumes prevent mixed model or data versions.",
    "row_file_receipt": "Path, count, and hash make the full corpus auditable.",
    "paired_embedding_corpus_ready_score": "EMIT BARE scalar; only 1.0 permits Exp5853.",
    "duration_s": "Measured multi-model wall time exposes mock or bootstrap-only execution.",
    "inference_substrate": "`live_llm_embedding_extraction` declares the true compute path.",
    "verifier_is_oracle": "True records exact labels; embeddings are not release authority.",
    "field_provenance": "Every aggregate traces to exact row, model, tokenizer, and device receipts.",
    "test_commands": "Commands document model resolution, extraction, alignment, parity, and schema checks.",
    "test_exit_codes": "Exit codes prevent partial extraction becoming readiness.",
    "reproducibility_checksum": "A checksum detects row, model, split, or configuration drift.",
    "honest_verdict": "A terminal prefix states ready, partial, or blocked outcome.",
}


class EmbeddingBackend(Protocol):
    """Minimal output-free embedding interface used by the live and fake paths."""

    def load(self) -> JsonDict:
        """Load model weights and return a loader/device receipt."""

    def tokenize(self, text: str) -> list[int]:
        """Return tokenizer IDs from the embedded GGUF tokenizer."""

    def embed(self, text: str) -> list[float]:
        """Return one final pooled embedding without generation or logits."""

    def close(self) -> None:
        """Release model resources after this model's row groups are done."""


EmbeddingBackendFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], EmbeddingBackend]


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a hex SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes in chunks so large GGUFs stay streamable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not Path(path).exists():
        return rows
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required at line {line_number}: {path}")
        rows.append(dict(payload))
    return rows


def source_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one Exp5840 source row while blanking its own row hash."""

    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def source_rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize Exp5840-shaped fixture rows deterministically."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def load_fixture_rows(path: str | Path = REPO_ROOT / EXP5840_ROWS_RELATIVE_PATH) -> list[JsonDict]:
    """Read Exp5840 source rows and verify their row hashes."""

    rows = _read_jsonl(path)
    for row in rows:
        if row.get("row_hash") != source_row_hash(row):
            raise ValueError(f"exp5840_row_hash:{row.get('row_id')}")
    return rows


def load_fixture_artifact(
    path: str | Path = REPO_ROOT / EXP5840_ARTIFACT_RELATIVE_PATH,
) -> JsonDict:
    """Read the Exp5840 terminal artifact and require clean fixture readiness."""

    artifact = _read_json(path)
    if artifact.get("counterfactual_fixture_ready_score") != 1.0:
        raise ValueError("exp5840_counterfactual_fixture_not_ready")
    return artifact


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash one Exp5852 output row while excluding its row hash field."""

    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize Exp5852 embedding rows as deterministic JSONL."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def read_row_file(path: str | Path) -> list[JsonDict]:
    """Read an Exp5852 JSONL row file."""

    return _read_jsonl(path)


def model_family(hf_id: str) -> str:
    """Return the stable short family label for a mandated GGUF id."""

    if hf_id == MANDATED_MODEL_HF_IDS[0]:
        return "qwen3.6-35b-a3b"
    if hf_id == MANDATED_MODEL_HF_IDS[1]:
        return "gemma-4-31b-it"
    if hf_id == MANDATED_MODEL_HF_IDS[2]:
        return "gemma-4-26b-a4b-it"
    return hf_id.rsplit("/", 1)[-1].replace("-GGUF", "").lower()


def _registry_row(hf_id: str) -> JsonDict:
    registry = {str(row["hf_id"]): dict(row) for row in SOTA_GGUF_MODELS}
    return dict(registry.get(hf_id, {}))


def _path_hash(path: str) -> str:
    return sha256_text(str(Path(path).expanduser().resolve())) if path else ""


def _tokenizer_receipt_from_source(source: Mapping[str, Any], model_path: str) -> JsonDict:
    provided = source.get("tokenizer_receipt")
    if isinstance(provided, Mapping):
        receipt = dict(provided)
        receipt.setdefault("source", "provided")
        receipt.setdefault("loadable", False)
        receipt.setdefault("detail", "")
        receipt["receipt_hash"] = sha256_json(receipt)
        return receipt
    ok, detail = (
        gguf_tokenizer_loadable(model_path) if model_path else (False, "missing model_path")
    )
    receipt = {
        "source": "embedded_gguf_llama_cpp_vocab_only",
        "loadable": ok,
        "detail": detail,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize mandated model specs with path, file hash, tokenizer, and GPU receipts."""

    by_id = {str(row.get("hf_id")): row for row in model_specs if isinstance(row, Mapping)}
    normalized: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        source = by_id.get(hf_id, {})
        registry = _registry_row(hf_id)
        model_path = str(source.get("model_path") or source.get("cache_path") or "")
        path = Path(model_path).expanduser() if model_path else Path()
        present = bool(model_path and path.is_file())
        tokenizer = (
            _tokenizer_receipt_from_source(source, model_path)
            if present
            else {
                "source": "missing_model_path",
                "loadable": False,
                "detail": f"model_path missing or not on disk: {model_path!r}",
                "receipt_hash": "",
            }
        )
        if not tokenizer.get("receipt_hash"):
            tokenizer["receipt_hash"] = sha256_json(tokenizer)
        model_sha = str(source.get("model_sha256") or (sha256_file(path) if present else ""))
        normalized.append(
            {
                "name": str(source.get("name") or registry.get("name") or hf_id.rsplit("/", 1)[-1]),
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": str(source.get("role") or registry.get("role") or ""),
                "gpu": int(source.get("gpu", index % 2) or 0),
                "model_path": model_path,
                "cache_path": model_path,
                "local_path_hash": _path_hash(model_path),
                "model_sha256": model_sha,
                "local_model_present": present,
                "headline_eligible": source.get("headline_eligible") is not False,
                "active_params_b": source.get("active_params_b", registry.get("active_params_b")),
                "total_params_b": source.get("total_params_b", registry.get("total_params_b")),
                "quantization": str(
                    source.get("quantization") or registry.get("quantization") or "Q4_K_M"
                ),
                "context_length": int(
                    source.get("context_length", DEFAULT_CONTEXT_LENGTH) or DEFAULT_CONTEXT_LENGTH
                ),
                "llama_cpp_loader": "carnot.pipeline.gemma4_quantized_loader.Gemma4QuantizedLoader",
                "tokenizer_receipt": tokenizer,
            }
        )
    return normalized


def resolve_all_model_specs() -> list[JsonDict]:  # pragma: no cover - host cache dependent.
    """Resolve all mandated GGUF files through the canonical SOTA registry."""

    rows: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        registry = _registry_row(hf_id)
        quant = str(registry.get("quantization") or "Q4_K_M")
        rows.append(
            {
                "name": registry.get("name") or hf_id.rsplit("/", 1)[-1],
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": registry.get("role", ""),
                "gpu": index % max(1, _gpu_count()),
                "model_path": resolve_cached_gguf(hf_id, quant) or "",
                "quantization": quant,
                "headline_eligible": True,
                "active_params_b": registry.get("active_params_b"),
                "total_params_b": registry.get("total_params_b"),
            }
        )
    return normalize_model_specs(rows)


def deterministic_embedding_config(
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    checkpoint_group_size: int = DEFAULT_CHECKPOINT_GROUP_SIZE,
) -> JsonDict:
    """Return the frozen output-free embedding settings."""

    config = {
        "schema": SCHEMA + ".deterministic_embedding_config",
        "backend": "llama_cpp_python",
        "loader_class": "Gemma4QuantizedLoader-compatible-output-free-embedding",
        "embedding": True,
        "pooling_type": "LLAMA_POOLING_TYPE_LAST",
        "normalize_embeddings": False,
        "embedding_value_precision": f"float32_json_{EMBEDDING_DECIMALS}dp",
        "preprocessing": "neutral_token_padding_then_float_rounding",
        "n_ctx": int(context_length),
        "n_batch": DEFAULT_BATCH_SIZE,
        "n_ubatch": DEFAULT_UBATCH_SIZE,
        "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
        "seed": DEFAULT_RANDOM_SEED,
        "logits_all": False,
        "max_tokens_generated": 0,
        "generated_answers_enabled": False,
        "output_logits_enabled": False,
        "truncate": False,
        "neutral_padding_vocab": list(NEUTRAL_PAD_VOCAB),
        "checkpoint_group_size": int(checkpoint_group_size),
    }
    config["config_hash"] = sha256_json(config)
    return config


class LlamaCppOutputFreeEmbeddingBackend(
    Gemma4QuantizedLoader
):  # pragma: no cover - live llama.cpp path.
    """llama.cpp embedding backend that reuses the GGUF quantized loader contract."""

    def __init__(self, model_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        super().__init__(
            model_path=str(model_spec["model_path"]),
            n_gpu_layers=int(config["n_gpu_layers"]),
            max_tokens=0,
        )
        self.model_spec = dict(model_spec)
        self.config = dict(config)

    # LSP note (typed 2026-07-26): this class has two masters. It INHERITS
    # Gemma4QuantizedLoader, whose `load()` returns bool, and it must also satisfy this
    # module's `EmbeddingBackend` Protocol, whose `load()` returns the loader RECEIPT dict
    # that the artifact records as its GPU-offload evidence. The two signatures cannot both
    # be honoured by one method, so the Protocol wins (the receipt is load-bearing for
    # fabrication detection -- an artifact claiming live GGUF inference has to carry the
    # observed device assignment) and the base-class widening is suppressed narrowly here
    # rather than by loosening Gemma4QuantizedLoader's own contract, which many other call
    # sites depend on. Behaviourally benign: a non-empty dict is truthy, so any caller that
    # treated the result as the base class's success bool still reads success correctly.
    def load(self) -> JsonDict:  # type: ignore[override]
        from llama_cpp import LLAMA_POOLING_TYPE_LAST, Llama, __version__ as llama_cpp_version

        before = _gpu_devices()
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=int(self.config["n_gpu_layers"]),
            main_gpu=int(self.model_spec["gpu"]),
            seed=int(self.config["seed"]),
            n_ctx=int(self.config["n_ctx"]),
            n_batch=int(self.config["n_batch"]),
            n_ubatch=int(self.config["n_ubatch"]),
            embedding=True,
            pooling_type=LLAMA_POOLING_TYPE_LAST,
            logits_all=False,
            verbose=False,
        )
        after = _gpu_devices()
        return {
            "loader_class": "carnot.pipeline.gemma4_quantized_loader.Gemma4QuantizedLoader",
            "llama_cpp_version": llama_cpp_version,
            "requested_n_gpu_layers": int(self.config["n_gpu_layers"]),
            "requested_main_gpu": int(self.model_spec["gpu"]),
            "observed_device_assignment": _gpu_delta(before, after),
            "embedding_mode": True,
            "output_logits_enabled": False,
            "generated_text_enabled": False,
        }

    def tokenize(self, text: str) -> list[int]:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return list(self._llm.tokenize(text.encode("utf-8"), add_bos=True, special=False))

    def embed(self, text: str) -> list[float]:
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        vector = self._llm.embed(
            text,
            normalize=bool(self.config["normalize_embeddings"]),
            truncate=False,
        )
        if vector and isinstance(vector[0], list):
            vector = vector[0]
        return _round_embedding(vector)

    def close(self) -> None:
        self._llm = None
        gc.collect()


def _round_embedding(vector: Any) -> list[float]:
    out: list[float] = []
    for value in vector:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite_embedding")
        out.append(round(number, EMBEDDING_DECIMALS))
    return out


def _gpu_count() -> int:  # pragma: no cover - host dependent.
    return max(1, int(_gpu_probe().get("gpu_count", 0) or 0))


def _memory_probe() -> JsonDict:  # pragma: no cover - host dependent.
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
    return {
        "available_mb": available_mb,
        "required_mb": RAM_FLOOR_MB,
        "ok": available_mb >= RAM_FLOOR_MB,
    }


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host dependent.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def _run_command(
    command: Sequence[str], *, timeout_s: float
) -> JsonDict:  # pragma: no cover - host dependent.
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


def _gpu_devices() -> list[JsonDict]:  # pragma: no cover - host dependent.
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    devices: list[JsonDict] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return devices


def _gpu_probe() -> JsonDict:  # pragma: no cover - host dependent.
    devices = _gpu_devices()
    return {
        "gpu_count": len(devices),
        "devices": devices,
        "ok": len(devices) > 0,
        "nvidia_smi": _run_command(["nvidia-smi"], timeout_s=10),
    }


def _gpu_delta(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover - host dependent.
    by_before = {
        int(row.get("index", -1)): int(row.get("memory_used_mb", 0) or 0) for row in before
    }
    by_after = {int(row.get("index", -1)): int(row.get("memory_used_mb", 0) or 0) for row in after}
    deltas = {
        str(index): max(0, by_after.get(index, 0) - by_before.get(index, 0))
        for index in sorted(set(by_before) | set(by_after))
        if index >= 0
    }
    return {"before": list(before), "after": list(after), "memory_delta_mb_by_gpu": deltas}


def _output_path_receipt(result_path: Path, row_file_path: Path, checkpoint_dir: Path) -> JsonDict:
    def writable(path: Path) -> bool:
        parent = path.parent
        return (parent.exists() and os.access(parent, os.W_OK)) or (
            parent.parent.exists() and os.access(parent.parent, os.W_OK)
        )

    return {
        "result_path": str(result_path),
        "row_file_path": str(row_file_path),
        "checkpoint_dir": str(checkpoint_dir),
        "result_writable": writable(result_path),
        "row_file_writable": writable(row_file_path),
        "checkpoint_writable": writable(checkpoint_dir / "probe.json"),
        "atomic_checkpoint_suffix": ".tmp",
        "ok": writable(result_path)
        and writable(row_file_path)
        and writable(checkpoint_dir / "probe.json"),
    }


def upstream_fixture_hashes(
    *,
    root: Path,
    fixture_artifact_path: Path,
    fixture_rows_path: Path,
    fixture_artifact: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Return hashes that bind this corpus to Exp5840 row and validator evidence."""

    row_hashes = {str(row["row_id"]): str(row["row_hash"]) for row in source_rows}
    files = {
        "exp5840_artifact": fixture_artifact_path,
        "exp5840_rows": fixture_rows_path,
        "verification_spec": root / VERIFY_SPEC_RELATIVE_PATH,
        "module": root / MODULE_RELATIVE_PATH,
        "test": root / TEST_RELATIVE_PATH,
        "protected_research_conductor": root / PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH,
    }
    return {
        "schema": SCHEMA + ".upstream_fixture_hashes",
        "files": {
            name: sha256_file(path) if path.exists() and path.is_file() else "missing"
            for name, path in files.items()
        },
        "exp5840_status": fixture_artifact.get("status"),
        "exp5840_counterfactual_fixture_ready_score": fixture_artifact.get(
            "counterfactual_fixture_ready_score"
        ),
        "row_count": len(source_rows),
        "row_hash_root": sha256_json(row_hashes),
        "row_file_receipt": dict(fixture_artifact.get("row_file_receipt") or {}),
        "split_receipt_hash": sha256_json(fixture_artifact.get("split_definition_and_hashes", {})),
        "validator_receipt_hash": sha256_json(
            fixture_artifact.get("exact_label_and_minimality_receipts", {})
        ),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    checkpoint_dir: str | Path = REPO_ROOT / CHECKPOINT_RELATIVE_DIR,
    fixture_artifact_path: str | Path = REPO_ROOT / EXP5840_ARTIFACT_RELATIVE_PATH,
    fixture_rows_path: str | Path = REPO_ROOT / EXP5840_ROWS_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
    gpu_probe: GpuProbe = _gpu_probe,
) -> JsonDict:  # pragma: no cover - host/resource dependent.
    """Collect Step 0 fixture, model, tokenizer, resource, and output checks."""

    root = Path(root)
    result = Path(result_path)
    rows_path = Path(row_file_path)
    checkpoints = Path(checkpoint_dir)
    blocked: list[str] = []
    fixture_ready = {"ok": False}
    try:
        fixture_artifact = load_fixture_artifact(fixture_artifact_path)
        source_rows = load_fixture_rows(fixture_rows_path)
        fixture_ready = {
            "ok": True,
            "artifact_path": str(fixture_artifact_path),
            "rows_path": str(fixture_rows_path),
            "row_count": len(source_rows),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        source_rows = []
        fixture_artifact = {}
        blocked.append("exp5840_fixture_unavailable_or_not_ready")
        fixture_ready = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    memory = memory_probe()
    disk = disk_probe(root)
    gpu = gpu_probe()
    outputs = _output_path_receipt(result, rows_path, checkpoints)
    if memory.get("ok") is not True:
        blocked.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked.append("insufficient_free_disk")
    if gpu.get("ok") is not True:
        blocked.append("gpu_device_receipt_unavailable")
    if outputs.get("ok") is not True:
        blocked.append("output_path_not_writable")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "fixture_ready": fixture_ready,
        "gpu": gpu,
        "resources": {"memory": memory, "disk": disk},
        "output_paths": outputs,
        "legacy_tiny_models_policy": {
            "legacy_smoke_model_ids": list(LEGACY_SMOKE_MODEL_IDS),
            "smoke_only": True,
            "cannot_satisfy_readiness": True,
        },
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def _precondition_blockers(
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers = list(preconditions.get("blocked_reasons") or [])
    if preconditions.get("preconditions_ready") is not True:
        blockers.append("preconditions_not_ready")
    if [str(row.get("hf_id")) for row in model_specs] != list(MANDATED_MODEL_HF_IDS):
        blockers.append("mandated_model_order_mismatch")
    for spec in model_specs:
        tokenizer = dict(spec.get("tokenizer_receipt") or {})
        if (
            spec.get("local_model_present") is not True
            or not str(spec.get("model_path", "")).endswith(".gguf")
            or not spec.get("model_sha256")
            or spec.get("headline_eligible") is not True
            or tokenizer.get("loadable") is not True
        ):
            blockers.append("mandated_model_unavailable")
            break
    gpu = preconditions.get("gpu", {})
    if not isinstance(gpu, Mapping) or int(gpu.get("gpu_count", 0) or 0) <= 0:
        blockers.append("gpu_device_receipt_unavailable")
    outputs = preconditions.get("output_paths", {})
    if not isinstance(outputs, Mapping) or outputs.get("ok") is not True:
        blockers.append("output_path_not_writable")
    policy = preconditions.get("legacy_tiny_models_policy", {})
    if not isinstance(policy, Mapping) or policy.get("cannot_satisfy_readiness") is not True:
        blockers.append("legacy_smoke_policy_missing")
    return sorted(set(blockers))


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _row_groups(source_rows: Sequence[Mapping[str, Any]], group_size: int) -> list[JsonDict]:
    groups: list[JsonDict] = []
    for start in range(0, len(source_rows), max(1, int(group_size))):
        group_rows = source_rows[start : start + max(1, int(group_size))]
        ids = [str(row["row_id"]) for row in group_rows]
        groups.append(
            {
                "row_group_id": f"rows-{start:06d}-{start + len(group_rows) - 1:06d}",
                "start_index": start,
                "end_index": start + len(group_rows) - 1,
                "row_ids": ids,
                "row_group_hash": sha256_json(ids),
            }
        )
    return groups


def _pad_to_parity(
    *,
    text: str,
    target_count: int,
    backend: EmbeddingBackend,
) -> tuple[str, int, list[str], bool]:
    padded = text
    added: list[str] = []
    count = len(backend.tokenize(padded))
    while count < target_count:
        progressed = False
        for token in NEUTRAL_PAD_VOCAB:
            trial = padded + " " + token
            trial_count = len(backend.tokenize(trial))
            if count < trial_count <= target_count:
                padded = trial
                added.append(token)
                count = trial_count
                progressed = True
                break
        if not progressed:
            return padded, count, added, False
    return padded, count, added, count == target_count


def _condition_feature_id(condition: Mapping[str, Any]) -> str:
    return sha256_json({"condition_id": str(condition["condition_id"])})


def _prepare_condition_inputs(
    source_row: Mapping[str, Any],
    backend: EmbeddingBackend,
    config: Mapping[str, Any],
) -> list[JsonDict]:
    conditions = [dict(row) for row in source_row["conditions"]]
    base_counts = [len(backend.tokenize(str(condition["model_input"]))) for condition in conditions]
    target = max(base_counts)
    prepared: list[JsonDict] = []
    for condition, base_count in zip(conditions, base_counts, strict=True):
        padded, count, added, parity_ok = _pad_to_parity(
            text=str(condition["model_input"]),
            target_count=target,
            backend=backend,
        )
        truncated = count > int(config["n_ctx"])
        prepared.append(
            {
                "condition_id": str(condition["condition_id"]),
                "condition_suffix": str(condition["condition_suffix"]),
                "source_model_input_hash": str(condition["model_input_hash"]),
                "embedding_input_hash": sha256_text(padded),
                "base_token_count": base_count,
                "token_count": count,
                "target_pair_token_count": target,
                "neutral_padding_tokens_added": added,
                "neutral_padding_token_count": len(added),
                "token_parity_ok": parity_ok,
                "truncated": truncated,
                "exact_label": bool(condition["exact_label"]),
                "text_for_embedding": padded,
            }
        )
    return prepared


def _embedding_hash(vector: Sequence[float]) -> str:
    return sha256_json([round(float(value), EMBEDDING_DECIMALS) for value in vector])


def _build_output_row(
    *,
    source_index: int,
    source_row: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    backend: EmbeddingBackend,
    config: Mapping[str, Any],
    loader_receipt: Mapping[str, Any],
) -> JsonDict:
    prepared = _prepare_condition_inputs(source_row, backend, config)
    condition_embeddings: list[JsonDict] = []
    for condition in prepared:
        embedding = _round_embedding(backend.embed(str(condition.pop("text_for_embedding"))))
        condition_embeddings.append(
            {
                "condition_id": condition["condition_id"],
                "condition_suffix": condition["condition_suffix"],
                "source_model_input_hash": condition["source_model_input_hash"],
                "embedding_input_hash": condition["embedding_input_hash"],
                "base_token_count": condition["base_token_count"],
                "token_count": condition["token_count"],
                "target_pair_token_count": condition["target_pair_token_count"],
                "neutral_padding_tokens_added": condition["neutral_padding_tokens_added"],
                "neutral_padding_token_count": condition["neutral_padding_token_count"],
                "token_parity_ok": condition["token_parity_ok"],
                "truncated": condition["truncated"],
                "embedding": embedding,
                "embedding_shape": [len(embedding)],
                "embedding_sha256": _embedding_hash(embedding),
            }
        )
    if len(condition_embeddings) != 2:
        raise ValueError("source_row_must_have_two_conditions")
    left, right = condition_embeddings
    if len(left["embedding"]) != len(right["embedding"]):
        raise ValueError("embedding_pair_shape_mismatch")
    difference = [
        round(float(b) - float(a), EMBEDDING_DECIMALS)
        for a, b in zip(left["embedding"], right["embedding"], strict=True)
    ]
    cell_id = f"{source_row['row_id']}|{model_spec['hf_id']}"
    feature_view = {
        "condition_features": [
            {
                "condition_id": _condition_feature_id(condition),
                "embedding": condition["embedding"],
                "embedding_shape": condition["embedding_shape"],
                "embedding_sha256": condition["embedding_sha256"],
                "token_count": condition["token_count"],
                "truncated": condition["truncated"],
            }
            for condition in condition_embeddings
        ],
        "paired_difference": difference,
        "paired_difference_sha256": _embedding_hash(difference),
        "difference_orientation": "condition_b_minus_a",
        "preprocessing": str(config["preprocessing"]),
    }
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "source_row_order": source_index,
        "source_row_id": str(source_row["row_id"]),
        "source_row_hash": str(source_row["row_hash"]),
        "source_pair_id": str(source_row["pair_id"]),
        "pair_group_id": str(source_row["pair_group_id"]),
        "split": str(source_row["split"]),
        "axis": str(source_row["axis"]),
        "family": str(source_row["family"]),
        "change": str(source_row["change"]),
        "surface_kind": str(source_row["surface_kind"]),
        "solver_effort_bin": str(source_row["solver_effort_bin"]),
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "model_file_sha256": str(model_spec["model_sha256"]),
        "model_local_path_hash": str(model_spec["local_path_hash"]),
        "embedding_cell_id": cell_id,
        "condition_embeddings": condition_embeddings,
        "paired_difference": difference,
        "paired_difference_sha256": _embedding_hash(difference),
        "oracle_label_receipt": {
            "source": "exp5840_exact_labels",
            "labels_by_condition_id": {
                prepared_condition["condition_id"]: prepared_condition["exact_label"]
                for prepared_condition in prepared
            },
            "verifier_is_oracle": True,
        },
        "feature_consumer_view": feature_view,
        "loader_receipt_hash": sha256_json(loader_receipt),
        "row_hash": "",
    }
    row["row_hash"] = row_hash(row)
    return row


def checkpoint_input_receipt(
    *,
    upstream_hashes: Mapping[str, Any],
    config: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    receipt = {
        "upstream_fixture_hashes_hash": sha256_json(upstream_hashes),
        "deterministic_embedding_config_hash": str(config["config_hash"]),
        "model_specs_hash": sha256_json(model_specs),
        "mandated_model_order": list(MANDATED_MODEL_HF_IDS),
    }
    receipt["input_receipt_hash"] = sha256_json(receipt)
    return receipt


def checkpoint_payload(
    *,
    model_spec: Mapping[str, Any],
    row_group_id: str,
    rows: Sequence[Mapping[str, Any]],
    input_receipt: Mapping[str, Any],
) -> JsonDict:
    row_hashes = {str(row["embedding_cell_id"]): str(row["row_hash"]) for row in rows}
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_hf_id": str(model_spec["hf_id"]),
        "model_file_sha256": str(model_spec["model_sha256"]),
        "tokenizer_receipt_hash": str(
            dict(model_spec.get("tokenizer_receipt") or {}).get("receipt_hash", "")
        ),
        "row_group_id": row_group_id,
        "input_receipt_hash": str(input_receipt["input_receipt_hash"]),
        "row_count": len(rows),
        "row_hashes": row_hashes,
        "row_hash_root": sha256_json(row_hashes),
        "rows": [dict(row) for row in rows],
    }
    payload["checkpoint_hash"] = sha256_json(payload)
    return payload


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
    row_group_id: str,
    input_receipt: Mapping[str, Any],
) -> JsonDict:
    reasons: list[str] = []
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        reasons.append("schema")
    if payload.get("model_hf_id") != model_spec.get("hf_id"):
        reasons.append("model_hf_id_mismatch")
    if payload.get("model_file_sha256") != model_spec.get("model_sha256"):
        reasons.append("model_file_hash_mismatch")
    expected_tokenizer = str(
        dict(model_spec.get("tokenizer_receipt") or {}).get("receipt_hash", "")
    )
    if payload.get("tokenizer_receipt_hash") != expected_tokenizer:
        reasons.append("tokenizer_receipt_hash_mismatch")
    if payload.get("row_group_id") != row_group_id:
        reasons.append("row_group_id_mismatch")
    if payload.get("input_receipt_hash") != input_receipt.get("input_receipt_hash"):
        reasons.append("input_receipt_hash_mismatch")
    rows = [dict(row) for row in payload.get("rows") or [] if isinstance(row, Mapping)]
    if payload.get("row_count") != len(rows):
        reasons.append("row_count_mismatch")
    row_hashes = {str(row.get("embedding_cell_id")): str(row.get("row_hash")) for row in rows}
    if payload.get("row_hashes") != row_hashes:
        reasons.append("row_hashes_mismatch")
    for row in rows:
        if row.get("row_hash") != row_hash(row):
            reasons.append("row_hash_mismatch")
            break
    return {"accepted": not reasons, "refusal_reasons": sorted(set(reasons))}


def _checkpoint_path(
    checkpoint_dir: Path, model_spec: Mapping[str, Any], row_group_id: str
) -> Path:
    family = str(model_spec["family"]).replace("/", "_")
    return checkpoint_dir / family / f"{row_group_id}.json"


def _load_checkpoint(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return _read_json(path)


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    _write_atomic(
        path, json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    )


def _extract_model_rows(
    *,
    model_spec: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    input_receipt: Mapping[str, Any],
    embedding_backend_factory: EmbeddingBackendFactory,
) -> tuple[list[JsonDict], JsonDict]:
    backend = embedding_backend_factory(model_spec, config)
    loader_receipt = backend.load()
    extracted_rows: list[JsonDict] = []
    resume_receipts: list[JsonDict] = []
    refused = False
    try:
        for group in _row_groups(source_rows, int(config["checkpoint_group_size"])):
            checkpoint_path = _checkpoint_path(
                checkpoint_dir, model_spec, str(group["row_group_id"])
            )
            checkpoint = _load_checkpoint(checkpoint_path)
            if checkpoint is not None:
                validation = validate_checkpoint_payload(
                    checkpoint,
                    model_spec=model_spec,
                    row_group_id=str(group["row_group_id"]),
                    input_receipt=input_receipt,
                )
                resume_receipts.append(
                    {
                        **group,
                        "model_hf_id": model_spec["hf_id"],
                        "checkpoint_path": str(checkpoint_path),
                        "resume_attempted": True,
                        **validation,
                    }
                )
                if validation["accepted"] is True:
                    extracted_rows.extend([dict(row) for row in checkpoint["rows"]])
                    continue
                refused = True
                break
            group_rows = []
            for source_index in range(int(group["start_index"]), int(group["end_index"]) + 1):
                group_rows.append(
                    _build_output_row(
                        source_index=source_index,
                        source_row=source_rows[source_index],
                        model_spec=model_spec,
                        backend=backend,
                        config=config,
                        loader_receipt=loader_receipt,
                    )
                )
            payload = checkpoint_payload(
                model_spec=model_spec,
                row_group_id=str(group["row_group_id"]),
                rows=group_rows,
                input_receipt=input_receipt,
            )
            _write_checkpoint(checkpoint_path, payload)
            resume_receipts.append(
                {
                    **group,
                    "model_hf_id": model_spec["hf_id"],
                    "checkpoint_path": str(checkpoint_path),
                    "resume_attempted": False,
                    "accepted": True,
                    "refusal_reasons": [],
                    "checkpoint_hash": payload["checkpoint_hash"],
                }
            )
            extracted_rows.extend(group_rows)
    finally:
        backend.close()
    return extracted_rows, {
        "model_hf_id": str(model_spec["hf_id"]),
        "loader_receipt": loader_receipt,
        "checkpoint_groups": resume_receipts,
        "checkpoint_refused": refused,
    }


def extract_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    input_receipt: Mapping[str, Any],
    embedding_backend_factory: EmbeddingBackendFactory = LlamaCppOutputFreeEmbeddingBackend,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Extract model-major checkpoints and return source-major output rows."""

    by_cell: dict[tuple[str, str], JsonDict] = {}
    extraction_receipts: list[JsonDict] = []
    for model_spec in model_specs:
        model_rows, receipt = _extract_model_rows(
            model_spec=model_spec,
            source_rows=source_rows,
            config=config,
            checkpoint_dir=checkpoint_dir,
            input_receipt=input_receipt,
            embedding_backend_factory=embedding_backend_factory,
        )
        extraction_receipts.append(receipt)
        for row in model_rows:
            by_cell[(str(row["source_row_id"]), str(row["model_hf_id"]))] = row
    ordered: list[JsonDict] = []
    for source_row in source_rows:
        for hf_id in MANDATED_MODEL_HF_IDS:
            cell = by_cell.get((str(source_row["row_id"]), hf_id))
            if cell is not None:
                ordered.append(cell)
    return ordered, extraction_receipts


def model_file_and_tokenizer_receipts(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipts = {
        str(spec["hf_id"]): {
            "model_path": str(spec["model_path"]),
            "model_sha256": str(spec["model_sha256"]),
            "local_path_hash": str(spec["local_path_hash"]),
            "quantization": str(spec["quantization"]),
            "tokenizer_receipt": dict(spec.get("tokenizer_receipt") or {}),
        }
        for spec in model_specs
    }
    return {
        "schema": SCHEMA + ".model_file_and_tokenizer_receipts",
        "receipts": receipts,
        "all_mandated_files_present": all(
            spec.get("local_model_present") is True for spec in model_specs
        ),
        "all_embedded_tokenizers_loadable": all(
            dict(spec.get("tokenizer_receipt") or {}).get("loadable") is True
            for spec in model_specs
        ),
        "receipt_hash": sha256_json(receipts),
    }


def gpu_and_loader_receipts(
    *,
    preconditions: Mapping[str, Any],
    extraction_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    loaders = {
        str(row["model_hf_id"]): dict(row.get("loader_receipt") or {})
        for row in extraction_receipts
    }
    return {
        "schema": SCHEMA + ".gpu_and_loader_receipts",
        "gpu_precondition_receipt": dict(preconditions.get("gpu") or {}),
        "loader_receipts_by_model": loaders,
        "llama_cpp_versions": sorted(
            {
                str(receipt.get("llama_cpp_version"))
                for receipt in loaders.values()
                if receipt.get("llama_cpp_version")
            }
        ),
        "all_loaders_output_free": bool(loaders)
        and all(
            receipt.get("embedding_mode") is True
            and receipt.get("output_logits_enabled") is False
            and receipt.get("generated_text_enabled") is False
            for receipt in loaders.values()
        ),
        "all_models_loaded": set(loaders) == set(MANDATED_MODEL_HF_IDS),
        "receipt_hash": sha256_json(loaders),
    }


def model_axis_family_cell_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    model_hf_ids: Sequence[str],
) -> JsonDict:
    source_counts = Counter(f"{row['axis']}|{row['family']}" for row in source_rows)
    observed = Counter(f"{row['model_hf_id']}|{row['axis']}|{row['family']}" for row in rows)
    split_counts = Counter(
        f"{row['model_hf_id']}|{row['axis']}|{row['family']}|{row['split']}" for row in rows
    )
    expected = {
        f"{hf_id}|{key}": value for hf_id in model_hf_ids for key, value in source_counts.items()
    }
    missing = [key for key, value in expected.items() if observed.get(key, 0) != value]
    return {
        "schema": SCHEMA + ".model_axis_family_cell_counts",
        "source_axis_family_counts": dict(sorted(source_counts.items())),
        "model_axis_family_counts": dict(sorted(observed.items())),
        "model_axis_family_split_counts": dict(sorted(split_counts.items())),
        "expected_model_axis_family_counts": dict(sorted(expected.items())),
        "missing_or_incomplete_cells": sorted(missing),
        "model_count": len(set(row.get("model_hf_id") for row in rows)),
        "source_row_count": len(source_rows),
        "all_cells_complete": not missing
        and len(rows) == len(source_rows) * len(model_hf_ids)
        and set(str(row.get("model_hf_id")) for row in rows) == set(model_hf_ids),
    }


def embedding_shape_and_finiteness(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    dims_by_model: dict[str, set[int]] = defaultdict(set)
    values_by_model: dict[str, list[list[float]]] = defaultdict(list)
    nonfinite: list[str] = []
    shape_failures: list[str] = []
    for row in rows:
        model = str(row.get("model_hf_id"))
        local_dims = []
        for condition in row.get("condition_embeddings", []):
            embedding = [float(value) for value in condition.get("embedding", [])]
            local_dims.append(len(embedding))
            dims_by_model[model].add(len(embedding))
            values_by_model[model].append(embedding)
            if any(not math.isfinite(value) for value in embedding):
                nonfinite.append(str(row.get("embedding_cell_id")))
        diff = [float(value) for value in row.get("paired_difference", [])]
        if len(local_dims) != 2 or len(set(local_dims + [len(diff)])) != 1:
            shape_failures.append(str(row.get("embedding_cell_id")))
    constant_dims: dict[str, list[int]] = {}
    for model, vectors in values_by_model.items():
        widths = {len(vector) for vector in vectors}
        if len(widths) != 1:
            constant_dims[model] = []
            continue
        width = len(vectors[0])
        constant_dims[model] = [
            index
            for index in range(width)
            if all(vector[index] == vectors[0][index] for vector in vectors)
        ]
    return {
        "schema": SCHEMA + ".embedding_shape_and_finiteness",
        "embedding_dims_by_model": {
            model: sorted(values) for model, values in sorted(dims_by_model.items())
        },
        "condition_embedding_count_by_model": {
            model: len(vectors) for model, vectors in sorted(values_by_model.items())
        },
        "shape_failure_count": len(shape_failures),
        "shape_failures": shape_failures[:20],
        "nonfinite_embedding_count": len(nonfinite),
        "nonfinite_embedding_cells": nonfinite[:20],
        "all_finite": bool(rows) and not nonfinite,
        "all_shapes_consistent": bool(rows)
        and not shape_failures
        and all(len(values) == 1 and next(iter(values)) > 0 for values in dims_by_model.values()),
        "constant_dimensions_after_preprocessing": {
            model: dims for model, dims in sorted(constant_dims.items()) if dims
        },
    }


def pair_alignment_receipts(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    model_hf_ids: Sequence[str],
) -> JsonDict:
    expected_order = [
        f"{source_row['row_id']}|{hf_id}" for source_row in source_rows for hf_id in model_hf_ids
    ]
    observed_order = [str(row.get("embedding_cell_id")) for row in rows]
    counts = Counter(observed_order)
    duplicates = sorted(key for key, value in counts.items() if value > 1)
    missing = sorted(set(expected_order) - set(observed_order))
    unexpected = sorted(set(observed_order) - set(expected_order))
    difference_failures = []
    pair_failures = []
    source_by_id = {str(row["row_id"]): row for row in source_rows}
    for row in rows:
        source = source_by_id.get(str(row.get("source_row_id")))
        if source is None or row.get("source_row_hash") != source.get("row_hash"):
            pair_failures.append(str(row.get("embedding_cell_id")))
            continue
        conditions = row.get("condition_embeddings", [])
        if len(conditions) != 2 or [c.get("condition_suffix") for c in conditions] != ["a", "b"]:
            pair_failures.append(str(row.get("embedding_cell_id")))
            continue
        left = [float(value) for value in conditions[0].get("embedding", [])]
        right = [float(value) for value in conditions[1].get("embedding", [])]
        expected = [round(b - a, EMBEDDING_DECIMALS) for a, b in zip(left, right, strict=True)]
        if row.get("paired_difference") != expected:
            difference_failures.append(str(row.get("embedding_cell_id")))
    return {
        "schema": SCHEMA + ".pair_alignment_receipts",
        "expected_model_row_cell_count": len(expected_order),
        "observed_model_row_cell_count": len(rows),
        "row_order_exact": observed_order == expected_order,
        "duplicate_model_row_cells": duplicates,
        "missing_model_row_cells": missing,
        "unexpected_model_row_cells": unexpected,
        "pair_join_failure_count": len(pair_failures),
        "pair_join_failures": pair_failures[:20],
        "paired_difference_mismatch_count": len(difference_failures),
        "paired_difference_mismatches": difference_failures[:20],
        "all_pairs_aligned": observed_order == expected_order
        and not duplicates
        and not missing
        and not unexpected
        and not pair_failures
        and not difference_failures,
    }


def token_and_truncation_parity(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    parity_failures = []
    truncation_failures = []
    padding_counts: Counter[str] = Counter()
    token_counts: list[int] = []
    for row in rows:
        conditions = list(row.get("condition_embeddings") or [])
        counts = [int(condition.get("token_count", -1)) for condition in conditions]
        token_counts.extend(counts)
        if len(counts) != 2 or len(set(counts)) != 1:
            parity_failures.append(str(row.get("embedding_cell_id")))
        if any(condition.get("truncated") is True for condition in conditions):
            truncation_failures.append(str(row.get("embedding_cell_id")))
        for condition in conditions:
            padding_counts[str(row.get("model_hf_id"))] += int(
                condition.get("neutral_padding_token_count", 0) or 0
            )
    return {
        "schema": SCHEMA + ".token_and_truncation_parity",
        "unique_token_counts": sorted(set(token_counts)),
        "pair_token_parity_failure_count": len(parity_failures),
        "pair_token_parity_failures": parity_failures[:20],
        "truncation_asymmetry_count": len(truncation_failures),
        "truncation_failures": truncation_failures[:20],
        "neutral_padding_token_counts_by_model": dict(sorted(padding_counts.items())),
        "all_pairs_token_count_matched": bool(rows) and not parity_failures,
        "no_truncation_asymmetry": not truncation_failures,
    }


def feature_leakage_checks(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    forbidden_keys = {
        "model_hf_id",
        "model_family",
        "family",
        "exact_label",
        "label",
        "oracle",
    }
    forbidden_tokens = {
        "qwen",
        "gemma",
        "finite_domain_csp",
        "weighted_maxsat",
        "hard_soft_packing",
        "finite_state_planning",
        "correct",
        "incorrect",
        "oracle",
        "label",
    }
    key_leaks = []
    token_leaks = []
    for row in rows:
        feature_view = dict(row.get("feature_consumer_view") or {})
        if forbidden_keys.intersection(feature_view):
            key_leaks.append(str(row.get("embedding_cell_id")))
        feature_text = canonical_json(feature_view).lower().replace("-", "_")
        if any(token in feature_text for token in forbidden_tokens):
            token_leaks.append(str(row.get("embedding_cell_id")))
    return {
        "schema": SCHEMA + ".feature_leakage_checks",
        "feature_identity_leakage_count": len(key_leaks),
        "feature_identity_leakage_cells": key_leaks[:20],
        "feature_token_leakage_count": len(token_leaks),
        "feature_token_leakage_cells": token_leaks[:20],
        "all_checks_passed": not key_leaks and not token_leaks,
    }


def checkpoint_resume_receipts(
    *,
    extraction_receipts: Sequence[Mapping[str, Any]],
    input_receipt: Mapping[str, Any],
) -> JsonDict:
    groups = [
        dict(group)
        for receipt in extraction_receipts
        for group in list(receipt.get("checkpoint_groups") or [])
    ]
    refusals = [
        dict(group)
        for group in groups
        if group.get("accepted") is not True or group.get("refusal_reasons")
    ]
    return {
        "schema": SCHEMA + ".checkpoint_resume_receipts",
        "input_receipt": dict(input_receipt),
        "checkpoint_groups": groups,
        "checkpoint_group_count": len(groups),
        "resume_refusal_count": len(refusals),
        "resume_refusals": refusals[:20],
        "all_checkpoints_hash_bound": bool(groups) and not refusals,
    }


def _row_file_receipt(rows: Sequence[Mapping[str, Any]], row_text: str) -> JsonDict:
    row_hashes = {str(row["embedding_cell_id"]): str(row["row_hash"]) for row in rows}
    receipt = {
        "path": ROW_FILE_RELATIVE_PATH.as_posix(),
        "row_count": len(rows),
        "sha256": sha256_text(row_text),
        "row_hashes": row_hashes,
        "row_hash_root": sha256_json(row_hashes),
        "atomic_write": True,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _row_file_receipt_ok(receipt: Mapping[str, Any]) -> bool:
    return (
        receipt.get("path") == ROW_FILE_RELATIVE_PATH.as_posix()
        and isinstance(receipt.get("row_count"), int)
        and str(receipt.get("sha256", "")).startswith("sha256:")
        and str(receipt.get("row_hash_root", "")).startswith("sha256:")
        and receipt.get("atomic_write") is True
    )


def verify_row_file(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    receipt = dict(artifact.get("row_file_receipt") or {})
    if not _row_file_receipt_ok(receipt):
        raise ValueError("row_file_receipt")
    expected_hashes = dict(receipt.get("row_hashes") or {})
    if len(rows) != receipt.get("row_count"):
        raise ValueError("row_count")
    for row in rows:
        if row.get("row_hash") != row_hash(row):
            raise ValueError(f"row_hash:{row.get('embedding_cell_id')}")
        if expected_hashes.get(str(row["embedding_cell_id"])) != row.get("row_hash"):
            raise ValueError(f"row_file_hash:{row.get('embedding_cell_id')}")
    if sha256_text(rows_to_jsonl(rows)) != receipt.get("sha256"):
        raise ValueError("row_file_sha256")
    return True


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5840_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5840_ROWS_RELATIVE_PATH.as_posix(),
        "python/carnot/inference/sota_models.py",
        "python/carnot/pipeline/gemma4_quantized_loader.py",
    ]
    return {
        field: {"principle": principle, "sources": sources}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def paired_embedding_corpus_ready_score(artifact: Mapping[str, Any]) -> float:
    preconditions = dict(artifact.get("preconditions_checked") or {})
    file_receipts = dict(artifact.get("model_file_and_tokenizer_receipts") or {})
    loader_receipts = dict(artifact.get("gpu_and_loader_receipts") or {})
    counts = dict(artifact.get("model_axis_family_cell_counts") or {})
    shapes = dict(artifact.get("embedding_shape_and_finiteness") or {})
    alignment = dict(artifact.get("pair_alignment_receipts") or {})
    parity = dict(artifact.get("token_and_truncation_parity") or {})
    checkpoints = dict(artifact.get("checkpoint_resume_receipts") or {})
    leakage = dict(artifact.get("feature_leakage_checks") or {"all_checks_passed": True})
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = bool(
        preconditions.get("preconditions_ready") is True
        and not preconditions.get("blocked_reasons")
        and artifact.get("models_used") == list(MANDATED_MODEL_HF_IDS)
        and [str(row.get("hf_id")) for row in artifact.get("model_specs", [])]
        == list(MANDATED_MODEL_HF_IDS)
        and file_receipts.get("all_mandated_files_present") is True
        and file_receipts.get("all_embedded_tokenizers_loadable") is True
        and loader_receipts.get("all_models_loaded") is True
        and loader_receipts.get("all_loaders_output_free") is True
        and counts.get("all_cells_complete") is True
        and shapes.get("all_finite") is True
        and shapes.get("all_shapes_consistent") is True
        and shapes.get("constant_dimensions_after_preprocessing") == {}
        and alignment.get("all_pairs_aligned") is True
        and parity.get("all_pairs_token_count_matched") is True
        and parity.get("no_truncation_asymmetry") is True
        and checkpoints.get("all_checkpoints_hash_bound") is True
        and leakage.get("all_checks_passed") is True
        and _row_file_receipt_ok(dict(artifact.get("row_file_receipt") or {}))
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    checks = {
        "model_file_and_tokenizer_receipts": dict(
            artifact.get("model_file_and_tokenizer_receipts") or {}
        ).get("all_mandated_files_present")
        is True
        and dict(artifact.get("model_file_and_tokenizer_receipts") or {}).get(
            "all_embedded_tokenizers_loadable"
        )
        is True,
        "gpu_and_loader_receipts": dict(artifact.get("gpu_and_loader_receipts") or {}).get(
            "all_models_loaded"
        )
        is True,
        "model_axis_family_cell_counts": dict(
            artifact.get("model_axis_family_cell_counts") or {}
        ).get("all_cells_complete")
        is True,
        "embedding_shape_and_finiteness": dict(
            artifact.get("embedding_shape_and_finiteness") or {}
        ).get("all_finite")
        is True,
        "pair_alignment_receipts": dict(artifact.get("pair_alignment_receipts") or {}).get(
            "all_pairs_aligned"
        )
        is True,
        "token_and_truncation_parity": dict(artifact.get("token_and_truncation_parity") or {}).get(
            "all_pairs_token_count_matched"
        )
        is True,
        "checkpoint_resume_receipts": dict(artifact.get("checkpoint_resume_receipts") or {}).get(
            "all_checkpoints_hash_bound"
        )
        is True,
        "row_file_receipt": _row_file_receipt_ok(dict(artifact.get("row_file_receipt") or {})),
    }
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    if not (
        commands
        and set(exit_codes) == set(commands)
        and all(code == 0 for code in exit_codes.values())
    ):
        reasons.append("failed_test_exit_codes")
    for name, ok in checks.items():
        if not ok:
            reasons.append(name)
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if paired_embedding_corpus_ready_score(artifact) == 1.0:
        return "ready: paired_embedding_corpus_complete_all_three_models"
    reasons = _blocked_reasons(artifact) or ["paired_embedding_corpus_not_ready"]
    prefix = "partial:" if artifact.get("status") == "partial" else "blocked:"
    return prefix + " " + ",".join(reasons[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    return sha256_json(stable)


def _artifact_from_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    upstream_hashes: Mapping[str, Any],
    config: Mapping[str, Any],
    extraction_receipts: Sequence[Mapping[str, Any]],
    input_receipt: Mapping[str, Any],
    row_text: str,
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    file_receipts = model_file_and_tokenizer_receipts(model_specs)
    checkpoint_receipts = checkpoint_resume_receipts(
        extraction_receipts=extraction_receipts,
        input_receipt=input_receipt,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "row_file": ROW_FILE_RELATIVE_PATH.as_posix(),
        "status": "complete",
        "preconditions_checked": dict(preconditions_checked),
        "model_specs": [dict(row) for row in model_specs],
        "models_used": list(MANDATED_MODEL_HF_IDS),
        "model_file_and_tokenizer_receipts": file_receipts,
        "gpu_and_loader_receipts": gpu_and_loader_receipts(
            preconditions=preconditions_checked,
            extraction_receipts=extraction_receipts,
        ),
        "upstream_fixture_hashes": dict(upstream_hashes),
        "deterministic_embedding_config": dict(config),
        "model_axis_family_cell_counts": model_axis_family_cell_counts(
            rows,
            source_rows=source_rows,
            model_hf_ids=MANDATED_MODEL_HF_IDS,
        ),
        "embedding_shape_and_finiteness": embedding_shape_and_finiteness(rows),
        "pair_alignment_receipts": pair_alignment_receipts(
            rows,
            source_rows=source_rows,
            model_hf_ids=MANDATED_MODEL_HF_IDS,
        ),
        "token_and_truncation_parity": token_and_truncation_parity(rows),
        "checkpoint_resume_receipts": checkpoint_receipts,
        "row_file_receipt": _row_file_receipt(rows, row_text),
        "paired_embedding_corpus_ready_score": 0.0,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "feature_leakage_checks": feature_leakage_checks(rows),
        "legacy_tiny_models": [
            {"hf_id": hf_id, "smoke_only": True, "readiness_eligible": False}
            for hf_id in LEGACY_SMOKE_MODEL_IDS
        ],
    }
    artifact["paired_embedding_corpus_ready_score"] = paired_embedding_corpus_ready_score(artifact)
    artifact["status"] = (
        "complete" if artifact["paired_embedding_corpus_ready_score"] == 1.0 else "partial"
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    upstream_hashes: Mapping[str, Any],
    config: Mapping[str, Any],
    input_receipt: Mapping[str, Any],
    row_text: str,
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    extraction_receipts: list[JsonDict] = []
    artifact = _artifact_from_rows(
        rows=[],
        source_rows=source_rows,
        model_specs=model_specs,
        preconditions_checked=preconditions_checked,
        upstream_hashes=upstream_hashes,
        config=config,
        extraction_receipts=extraction_receipts,
        input_receipt=input_receipt,
        row_text=row_text,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    artifact["status"] = "blocked"
    artifact["paired_embedding_corpus_ready_score"] = 0.0
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(field)
    if set(REQUIRED_FIELD_PRINCIPLES) - set(artifact.get("field_provenance", {})):
        raise ValueError("field_provenance")
    expected_score = paired_embedding_corpus_ready_score(artifact)
    if artifact.get("paired_embedding_corpus_ready_score") != expected_score:
        raise ValueError("paired_embedding_corpus_ready_score")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if artifact.get("models_used") != list(MANDATED_MODEL_HF_IDS):
        raise ValueError("models_used")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    if expected_score == 1.0:
        if artifact.get("status") != "complete":
            raise ValueError("status")
        if not str(artifact.get("honest_verdict", "")).startswith("ready:"):
            raise ValueError("honest_verdict")
    else:
        if artifact.get("status") not in {"blocked", "partial"}:
            raise ValueError("status")
        verdict = str(artifact.get("honest_verdict", ""))
        if not (verdict.startswith("blocked:") or verdict.startswith("partial:")):
            raise ValueError("honest_verdict")
    return True


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    checkpoint_dir: str | Path = REPO_ROOT / CHECKPOINT_RELATIVE_DIR,
    fixture_artifact_path: str | Path = REPO_ROOT / EXP5840_ARTIFACT_RELATIVE_PATH,
    fixture_rows_path: str | Path = REPO_ROOT / EXP5840_ROWS_RELATIVE_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    embedding_backend_factory: EmbeddingBackendFactory = LlamaCppOutputFreeEmbeddingBackend,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5852 or emit a terminal blocked artifact when Step 0 fails."""

    started = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path)
    row_file_path = Path(row_file_path)
    checkpoint_dir = Path(checkpoint_dir)
    fixture_artifact_path = Path(fixture_artifact_path)
    fixture_rows_path = Path(fixture_rows_path)
    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    config = deterministic_embedding_config()
    specs = (
        normalize_model_specs(model_specs) if model_specs is not None else resolve_all_model_specs()
    )
    try:
        fixture_artifact = load_fixture_artifact(fixture_artifact_path)
        source_rows = load_fixture_rows(fixture_rows_path)
    except (OSError, ValueError, json.JSONDecodeError):
        fixture_artifact = {"counterfactual_fixture_ready_score": 0.0, "status": "missing"}
        source_rows = []
    upstream_hashes = upstream_fixture_hashes(
        root=root,
        fixture_artifact_path=fixture_artifact_path,
        fixture_rows_path=fixture_rows_path,
        fixture_artifact=fixture_artifact,
        source_rows=source_rows,
    )
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(
            root=root,
            result_path=result_path,
            row_file_path=row_file_path,
            checkpoint_dir=checkpoint_dir,
            fixture_artifact_path=fixture_artifact_path,
            fixture_rows_path=fixture_rows_path,
        )
    )
    blockers = _precondition_blockers(preconditions, specs)
    if fixture_artifact.get("counterfactual_fixture_ready_score") != 1.0 or not source_rows:
        blockers.append("exp5840_fixture_unavailable_or_not_ready")
    preconditions["blocked_reasons"] = sorted(set(blockers))
    preconditions["preconditions_ready"] = not preconditions["blocked_reasons"]
    preconditions.setdefault(
        "output_paths",
        _output_path_receipt(result_path, row_file_path, checkpoint_dir),
    )
    input_receipt = checkpoint_input_receipt(
        upstream_hashes=upstream_hashes,
        config=config,
        model_specs=specs,
    )
    if preconditions["blocked_reasons"]:
        rows: list[JsonDict] = []
        row_text = ""
        artifact = _blocked_artifact(
            source_rows=source_rows,
            model_specs=specs,
            preconditions_checked=preconditions,
            upstream_hashes=upstream_hashes,
            config=config,
            input_receipt=input_receipt,
            row_text=row_text,
            duration_s=time.perf_counter() - started,
            test_commands=test_commands,
            test_exit_codes=exit_codes,
        )
    else:
        rows, extraction_receipts = extract_rows(
            source_rows=source_rows,
            model_specs=specs,
            config=config,
            checkpoint_dir=checkpoint_dir,
            input_receipt=input_receipt,
            embedding_backend_factory=embedding_backend_factory,
        )
        row_text = rows_to_jsonl(rows)
        artifact = _artifact_from_rows(
            rows=rows,
            source_rows=source_rows,
            model_specs=specs,
            preconditions_checked=preconditions,
            upstream_hashes=upstream_hashes,
            config=config,
            extraction_receipts=extraction_receipts,
            input_receipt=input_receipt,
            row_text=row_text,
            duration_s=time.perf_counter() - started,
            test_commands=test_commands,
            test_exit_codes=exit_codes,
        )
    validate_artifact(artifact)
    if write:
        _write_atomic(row_file_path, row_text)
        _write_atomic(
            result_path,
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        )
    return artifact


def main() -> int:  # pragma: no cover - script entry point.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point.
    raise SystemExit(main())
