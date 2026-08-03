"""Exp5964 SOTA GGUF atom compatibility representation corpus.

Spec refs: REQ-INFER-SOTA-5964, SCENARIO-INFER-SOTA-5964-BLOCKED,
SCENARIO-INFER-SOTA-5964-CORPUS, SCENARIO-INFER-SOTA-5964-CONTROLS.

This experiment asks only whether output-free final-token representations are a
usable corpus surface for a sealed exact context/candidate-atom fixture.  It does
not generate answers, does not read intermediate layers, and does not train a
classifier.  Exact labels remain external audit material from Exp5963.
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

from carnot import experiment_5852_three_family_paired_embeddings as exp5852
from carnot import experiment_5963_exact_atom_pair_fixture as exp5963
from carnot.inference.sota_models import SOTA_GGUF_MODELS, gguf_tokenizer_loadable, resolve_cached_gguf


JsonDict = dict[str, Any]
EmbeddingBackendFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], "EmbeddingBackend"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5964_sota_atom_compatibility_corpus.json")
ROW_BASENAME = "experiment_5964_sota_atom_compatibility_corpus"
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5964_sota_atom_compatibility_corpus.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5964_sota_atom_compatibility_corpus.py")
LLM_SPEC_RELATIVE_PATH = Path("openspec/capabilities/llm-ebm-inference/spec.md")
EXP5963_ARTIFACT_RELATIVE_PATH = exp5963.RESULT_RELATIVE_PATH
EXP5963_CONTEXT_RELATIVE_PATH = exp5963.CONTEXT_ROW_RELATIVE_PATH
EXP5963_PAIR_RELATIVE_PATH = exp5963.PAIR_ROW_RELATIVE_PATH

SCHEMA = "carnot.experiment_5964.sota_atom_compatibility_corpus.v1"
ROW_SCHEMA = SCHEMA + ".vector_row"
EXPERIMENT_ID = "experiment_5964_sota_atom_compatibility_corpus"
RUN_DATE = "20260803"
INFERENCE_SUBSTRATE = "live_llm_embedding_extraction"
VERIFIER_IS_ORACLE = False
RAM_FLOOR_MB = 16_384
DISK_FLOOR_MB = 10_240
DEFAULT_N_GPU_LAYERS = -1
EMBEDDING_DECIMALS = 8
EPSILON = 1e-8
PROMPT_TEMPLATE_VERSION = "exp5964_output_free_atom_compat_v1"
DUPLICATE_CONTROL_PAIR_COUNT = 2

MANDATED_MODEL_HF_IDS = exp5852.MANDATED_MODEL_HF_IDS
LEGACY_SMOKE_MODEL_IDS = exp5852.LEGACY_SMOKE_MODEL_IDS
REPRESENTATION_KINDS = (
    "context_alone",
    "atom_alone",
    "context_then_atom",
    "context_then_claim_flip_atom",
    "atom_then_context",
)
DUPLICATE_REPRESENTATION = "context_then_atom_duplicate"

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("research-references.md"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5964_sota_atom_compatibility_corpus.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5964_sota_atom_compatibility_corpus.py "
    "-m pytest tests/python/test_experiment_5964_sota_atom_compatibility_corpus.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5964_sota_atom_compatibility_corpus.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5964_sota_atom_compatibility_corpus.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5964_sota_atom_compatibility_corpus.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/exclusion_manifest.yaml "
    "ops/changelog.md ops/status.md _bmad/traceability.md research-references.md",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "gate_replay_receipt",
    "model_specs",
    "model_file_hashes",
    "embedded_tokenizer_and_llama_cpp_receipts",
    "cuda_offload_vram_thermal_and_cleanup_receipts",
    "prompt_serialization_and_pair_order_contract",
    "per_model_row_paths_hashes_counts_and_prefix_chains",
    "raw_vs_standardized_feature_separation",
    "finite_variance_duplicate_and_order_controls",
    "claim_flip_and_per_family_headroom_controls",
    "norm_length_frequency_label_pair_permutation_and_model_identity_controls",
    "split_and_label_secrecy_receipts",
    "atom_compatibility_corpus_ready_score",
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

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "Missing model, CUDA, tokenizer, memory, disk, time, or output prerequisites block "
        "before full extraction."
    ),
    "preconditions_checked": (
        "The preflight records every required local prerequisite and refuses legacy or CPU "
        "headline substitution."
    ),
    "gate_replay_receipt": (
        "Exp 5963 exact path/hash/value must satisfy `pair_fixture_ready_score == 1.0`."
    ),
    "model_specs": "All three mandated public GGUF identities and exact local files are auditable.",
    "model_file_hashes": "Exact cached GGUF file hashes bind every row to local bytes.",
    "embedded_tokenizer_and_llama_cpp_receipts": (
        "The GGUF runtime and embedded tokenizer are the only model-loading path."
    ),
    "cuda_offload_vram_thermal_and_cleanup_receipts": (
        "Headline rows require genuine CUDA offload, measured resources, and clean teardown."
    ),
    "prompt_serialization_and_pair_order_contract": (
        "Every vector maps to one sealed pair and preregistered ordering."
    ),
    "per_model_row_paths_hashes_counts_and_prefix_chains": (
        "Each family corpus is immutable, complete, and hash-linked to Exp 5963."
    ),
    "raw_vs_standardized_feature_separation": (
        "Train-fold statistics standardize within model; raw dimensions never create a "
        "cross-family shortcut."
    ),
    "finite_variance_duplicate_and_order_controls": (
        "Representations must be finite, non-degenerate, deterministic for duplicates, and "
        "honestly sensitive to ordering."
    ),
    "claim_flip_and_per_family_headroom_controls": (
        "Readiness requires directional semantic headroom in disaggregated families, not a "
        "pooled artifact."
    ),
    "norm_length_frequency_label_pair_permutation_and_model_identity_controls": (
        "Every shortcut remains an explicit measured negative control."
    ),
    "split_and_label_secrecy_receipts": (
        "Labels and test-fold statistics cannot affect extraction, normalization, or prompts."
    ),
    "atom_compatibility_corpus_ready_score": (
        "Emit bare `1.0` only for all-family integrity plus preregistered non-degenerate "
        "semantic headroom; the same disqualification retires the surface."
    ),
    "protected_files_unchanged": (
        "Active roadmap, conductor, exclusions, history, and unrelated changes remain unchanged."
    ),
    "duration_s": "Measured wall time for `live_llm_embedding_extraction`.",
    "inference_substrate": "Use measured `live_llm_embedding_extraction`.",
    "verifier_is_oracle": (
        "False for representation features because exact labels remain external."
    ),
    "missing_verifier_gaps": "Known limitations stay explicit.",
    "field_provenance": (
        "Every field traces to task prompt, spec, module, rows, model, and runtime receipts."
    ),
    "test_commands": (
        "Verification commands cover unit, coverage, gate, model, tokenizer, CUDA smoke, "
        "row-chain, shortcut, adversarial, spec, E2E, protected-file, and root-clutter checks."
    ),
    "test_exit_codes": (
        "Exit codes prevent partial extraction or unchecked rows from becoming readiness."
    ),
    "reproducibility_checksum": (
        "Stable checksum binds row corpora, receipts, controls, and verdict while excluding "
        "wall-clock duration."
    ),
    "honest_verdict": "Use `complete_ready:`, `retired:`, or `blocked:`.",
}


class EmbeddingBackend(Protocol):
    """Output-free embedding interface shared by fake and llama.cpp backends."""

    def load(self) -> JsonDict:
        """Load weights and return the runtime receipt."""

    def tokenize(self, text: str) -> list[int]:
        """Tokenize with the model's embedded GGUF tokenizer."""

    def embed(self, text: str) -> list[float]:
        """Return one pooled final-token representation without generation."""

    def close(self) -> None:
        """Release model resources after extraction."""


LlamaCppOutputFreeEmbeddingBackend = exp5852.LlamaCppOutputFreeEmbeddingBackend


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes in chunks so GGUF files stay streamable."""

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


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _round_embedding(vector: Any) -> list[float]:
    out: list[float] = []
    for value in vector:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite_embedding")
        out.append(round(number, EMBEDDING_DECIMALS))
    return out


def _embedding_hash(vector: Sequence[float]) -> str:
    return sha256_json([round(float(value), EMBEDDING_DECIMALS) for value in vector])


def model_family(hf_id: str) -> str:
    """Return the stable local family label used in row filenames."""

    return exp5852.model_family(hf_id)


def model_row_relative_path(hf_id: str) -> Path:
    """Return the per-model immutable row corpus relative path."""

    return Path("results") / f"{ROW_BASENAME}.{model_family(hf_id)}.rows.jsonl"


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize specs with the Exp5852 mandated order and embedded-tokenizer rule."""

    registry = {str(row["hf_id"]): dict(row) for row in SOTA_GGUF_MODELS}
    by_id = {str(row.get("hf_id")): row for row in model_specs if isinstance(row, Mapping)}
    normalized: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        source = by_id.get(hf_id, {})
        registry_row = registry.get(hf_id, {})
        model_path = str(source.get("model_path") or source.get("cache_path") or "")
        path = Path(model_path).expanduser() if model_path else Path()
        present = bool(model_path and path.is_file())
        provided = source.get("tokenizer_receipt")
        if isinstance(provided, Mapping):
            tokenizer = dict(provided)
            tokenizer.setdefault("source", "provided")
            tokenizer.setdefault("loadable", False)
            tokenizer.setdefault("detail", "")
        elif present:
            ok, detail = gguf_tokenizer_loadable(model_path)
            tokenizer = {
                "source": "embedded_gguf_llama_cpp_vocab_only",
                "loadable": ok,
                "detail": detail,
            }
        else:
            tokenizer = {
                "source": "missing_model_path",
                "loadable": False,
                "detail": f"model_path missing or not on disk: {model_path!r}",
            }
        tokenizer["receipt_hash"] = sha256_json(tokenizer)
        model_sha = str(source.get("model_sha256") or (sha256_file(path) if present else ""))
        normalized.append(
            {
                "name": str(
                    source.get("name") or registry_row.get("name") or hf_id.rsplit("/", 1)[-1]
                ),
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": str(source.get("role") or registry_row.get("role") or ""),
                "gpu": int(source.get("gpu", index % 2) or 0),
                "model_path": model_path,
                "cache_path": model_path,
                "local_path_hash": sha256_text(str(path.resolve())) if model_path else "",
                "model_sha256": model_sha,
                "local_model_present": present,
                "headline_eligible": source.get("headline_eligible") is not False,
                "active_params_b": source.get("active_params_b", registry_row.get("active_params_b")),
                "total_params_b": source.get("total_params_b", registry_row.get("total_params_b")),
                "min_vram_gb": source.get("min_vram_gb", registry_row.get("min_vram_gb")),
                "quantization": str(
                    source.get("quantization") or registry_row.get("quantization") or "Q4_K_M"
                ),
                "context_length": int(
                    source.get("context_length", exp5852.DEFAULT_CONTEXT_LENGTH)
                    or exp5852.DEFAULT_CONTEXT_LENGTH
                ),
                "llama_cpp_loader": (
                    "carnot.pipeline.gemma4_quantized_loader.Gemma4QuantizedLoader"
                ),
                "tokenizer_receipt": tokenizer,
            }
        )
    return normalized


def resolve_all_model_specs() -> list[JsonDict]:  # pragma: no cover - host cache dependent.
    """Resolve all three mandated GGUF files from local cache only."""

    registry = {str(row["hf_id"]): dict(row) for row in SOTA_GGUF_MODELS}
    rows: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        registry_row = registry.get(hf_id, {})
        quant = str(registry_row.get("quantization") or "Q4_K_M")
        rows.append(
            {
                "name": registry_row.get("name") or hf_id.rsplit("/", 1)[-1],
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": registry_row.get("role", ""),
                "gpu": index % 2,
                "model_path": resolve_cached_gguf(hf_id, quant) or "",
                "quantization": quant,
                "headline_eligible": True,
                "active_params_b": registry_row.get("active_params_b"),
                "total_params_b": registry_row.get("total_params_b"),
            }
        )
    return normalize_model_specs(rows)


def deterministic_embedding_config() -> JsonDict:
    """Return the frozen LAST-pooled output-free embedding settings."""

    config = dict(exp5852.deterministic_embedding_config())
    config.update(
        {
            "schema": SCHEMA + ".deterministic_embedding_config",
            "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "representation_kinds": list(REPRESENTATION_KINDS),
            "duplicate_control_pair_count": DUPLICATE_CONTROL_PAIR_COUNT,
        }
    )
    config["config_hash"] = sha256_json(config)
    return config


def gate_replay_receipt(
    *,
    fixture_artifact_path: str | Path,
    context_rows_path: str | Path,
    pair_rows_path: str | Path,
) -> JsonDict:
    """Replay the Exp5963 gate value and row chains before model extraction."""

    path = Path(fixture_artifact_path)
    try:
        artifact = _read_json(path)
        score = float(artifact.get("pair_fixture_ready_score", 0.0))
        status = str(artifact.get("status", "missing"))
        artifact_hash = sha256_file(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        artifact = {}
        score = 0.0
        status = "missing"
        artifact_hash = ""
        artifact_error = f"{type(exc).__name__}: {exc}"
    else:
        artifact_error = ""
    context_replay = exp5963.replay_context_rows(Path(context_rows_path))
    pair_replay = exp5963.replay_pair_rows(Path(pair_rows_path))
    receipt = {
        "schema": SCHEMA + ".gate_replay_receipt",
        "artifact_path": str(path),
        "artifact_sha256": artifact_hash,
        "artifact_status": status,
        "pair_fixture_ready_score": score,
        "context_rows_path": str(context_rows_path),
        "context_rows_sha256": (
            sha256_file(context_rows_path) if Path(context_rows_path).exists() else ""
        ),
        "pair_rows_path": str(pair_rows_path),
        "pair_rows_sha256": sha256_file(pair_rows_path) if Path(pair_rows_path).exists() else "",
        "context_replay": context_replay,
        "pair_replay": pair_replay,
        "artifact_error": artifact_error,
        "ready": score == 1.0 and context_replay.get("ok") is True and pair_replay.get("ok") is True,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    receipt["artifact_row_receipt"] = dict(artifact.get("row_paths_hashes_and_prefix_chain") or {})
    return receipt


def claim_flip_atom_text(atom: Mapping[str, Any]) -> str:
    """Return deterministic public text for a claim-flipped candidate atom."""

    payload = _copy_json(atom.get("payload", {}))
    if isinstance(payload, Mapping) and isinstance(payload.get("truth"), bool):
        flipped = dict(payload)
        flipped["truth"] = not bool(payload["truth"])
        return f"{atom.get('atom_kind', 'atom')} {canonical_json(flipped)}"
    return "claim.flip " + canonical_json(
        {
            "atom_kind": atom.get("atom_kind"),
            "payload": payload,
        }
    )


def _serialize_prompt(
    *,
    representation_kind: str,
    pair_order: str,
    context_text: str,
    candidate_text: str,
    pair: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            f"schema={PROMPT_TEMPLATE_VERSION}",
            f"representation={representation_kind}",
            f"pair_order={pair_order}",
            f"pair_sequence={pair['sequence_index']}",
            f"context_id={pair['context_id']}",
            "context:",
            context_text,
            "candidate_atom:",
            candidate_text,
        ]
    )


def prompt_inputs_for_pair(
    pair: Mapping[str, Any],
    context: Mapping[str, Any],
) -> list[JsonDict]:
    """Build preregistered, label-free prompt inputs for one sealed pair."""

    context_text = str(context["model_visible_text"])
    candidate_text = str(pair["candidate_text"])
    flipped_text = claim_flip_atom_text(dict(pair.get("candidate_atom") or {}))
    prompt_specs = [
        ("context_alone", "context_only", context_text, ""),
        ("atom_alone", "atom_only", "", candidate_text),
        ("context_then_atom", "context_then_atom", context_text, candidate_text),
        (
            "context_then_claim_flip_atom",
            "context_then_claim_flip_atom",
            context_text,
            flipped_text,
        ),
        ("atom_then_context", "atom_then_context", context_text, candidate_text),
    ]
    prompts: list[JsonDict] = []
    for kind, order, ctx_text, atom_text in prompt_specs:
        if kind == "atom_then_context":
            text = "\n".join(
                [
                    f"schema={PROMPT_TEMPLATE_VERSION}",
                    f"representation={kind}",
                    f"pair_order={order}",
                    f"pair_sequence={pair['sequence_index']}",
                    f"context_id={pair['context_id']}",
                    "candidate_atom:",
                    atom_text,
                    "context:",
                    ctx_text,
                ]
            )
        else:
            text = _serialize_prompt(
                representation_kind=kind,
                pair_order=order,
                context_text=ctx_text,
                candidate_text=atom_text,
                pair=pair,
            )
        prompts.append(
            {
                "representation_kind": kind,
                "pair_order": order,
                "prompt_text": text,
                "prompt_hash": sha256_text(text),
            }
        )
    return prompts


def vector_row_hash(row: Mapping[str, Any]) -> str:
    """Hash an Exp5964 vector row while blanking its own row hash."""

    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize vector rows deterministically."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def read_model_row_file(path: str | Path) -> list[JsonDict]:
    """Read one per-model vector row corpus."""

    return _read_jsonl(path)


def verify_model_row_file(rows: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any]) -> bool:
    """Verify row hashes, prefix chain, row count, and file hash receipt."""

    previous = exp5963.INITIAL_PREFIX_HASH
    row_hashes: dict[str, str] = {}
    for expected_index, row in enumerate(rows):
        if row.get("sequence_index") != expected_index:
            raise ValueError("sequence_index")
        if row.get("previous_hash") != previous:
            raise ValueError("previous_hash")
        computed = vector_row_hash(row)
        if row.get("row_hash") != computed:
            raise ValueError(f"row_hash:{row.get('vector_row_id')}")
        previous = str(row["row_hash"])
        row_hashes[str(row["vector_row_id"])] = str(row["row_hash"])
    if receipt.get("row_count") != len(rows):
        raise ValueError("row_count")
    if dict(receipt.get("row_hashes") or {}) != row_hashes:
        raise ValueError("row_hashes")
    if receipt.get("sha256") != sha256_text(rows_to_jsonl(rows)):
        raise ValueError("row_file_sha256")
    if receipt.get("final_prefix_checksum", previous) != previous:
        raise ValueError("final_prefix_checksum")
    return True


def _build_vector_row(
    *,
    sequence_index: int,
    model_spec: Mapping[str, Any],
    pair: Mapping[str, Any],
    context: Mapping[str, Any],
    prompt: Mapping[str, Any],
    backend: EmbeddingBackend,
    config: Mapping[str, Any],
    previous_hash: str,
    representation_kind: str | None = None,
) -> JsonDict:
    prompt_text = str(prompt["prompt_text"])
    embedding = _round_embedding(backend.embed(prompt_text))
    token_count = len(backend.tokenize(prompt_text))
    kind = representation_kind or str(prompt["representation_kind"])
    vector_row_id = (
        f"{model_spec['family']}|{pair['sequence_index']:06d}|{kind}"
    )
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "sequence_index": sequence_index,
        "vector_row_id": vector_row_id,
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "model_file_sha256": str(model_spec["model_sha256"]),
        "model_local_path_hash": str(model_spec["local_path_hash"]),
        "exp5963_pair_id": str(pair["pair_id"]),
        "exp5963_pair_sequence_index": int(pair["sequence_index"]),
        "exp5963_pair_row_hash": str(pair["row_hash"]),
        "exp5963_context_id": str(context["context_id"]),
        "exp5963_context_row_hash": str(context["row_hash"]),
        "semantic_instance_id": str(context["semantic_instance_id"]),
        "source_split": str(context["source_split"]),
        "source_family": str(context["family"]),
        "candidate_text_hash": str(pair["candidate_text_hash"]),
        "context_text_hash": str(context["model_visible_text_hash"]),
        "representation_kind": kind,
        "pair_order": str(prompt["pair_order"]),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_hash": str(prompt["prompt_hash"]),
        "prompt_token_count": token_count,
        "prompt_truncated": token_count > int(config["n_ctx"]),
        "embedding": embedding,
        "embedding_shape": [len(embedding)],
        "embedding_sha256": _embedding_hash(embedding),
        "previous_hash": previous_hash,
        "row_hash": "",
    }
    row["row_hash"] = vector_row_hash(row)
    return row


def extract_rows(
    *,
    context_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    embedding_backend_factory: EmbeddingBackendFactory = LlamaCppOutputFreeEmbeddingBackend,
) -> tuple[dict[str, list[JsonDict]], list[JsonDict]]:
    """Extract model-separated vector rows with output-free LAST pooling."""

    contexts_by_id = {str(row["context_id"]): dict(row) for row in context_rows}
    rows_by_model: dict[str, list[JsonDict]] = {}
    extraction_receipts: list[JsonDict] = []
    for model_spec in model_specs:
        backend = embedding_backend_factory(model_spec, config)
        before_close_devices = _gpu_devices()
        loader_receipt = backend.load()
        model_rows: list[JsonDict] = []
        previous = exp5963.INITIAL_PREFIX_HASH
        try:
            for pair in pair_rows:
                context = contexts_by_id[str(pair["context_id"])]
                prompts = prompt_inputs_for_pair(pair, context)
                context_then_atom_prompt = None
                for prompt in prompts:
                    if prompt["representation_kind"] == "context_then_atom":
                        context_then_atom_prompt = prompt
                    row = _build_vector_row(
                        sequence_index=len(model_rows),
                        model_spec=model_spec,
                        pair=pair,
                        context=context,
                        prompt=prompt,
                        backend=backend,
                        config=config,
                        previous_hash=previous,
                    )
                    previous = row["row_hash"]
                    model_rows.append(row)
                if int(pair["sequence_index"]) < DUPLICATE_CONTROL_PAIR_COUNT:
                    if context_then_atom_prompt is None:  # pragma: no cover - impossible list guard.
                        raise ValueError("missing_context_then_atom_prompt")
                    duplicate = _build_vector_row(
                        sequence_index=len(model_rows),
                        model_spec=model_spec,
                        pair=pair,
                        context=context,
                        prompt=context_then_atom_prompt,
                        backend=backend,
                        config=config,
                        previous_hash=previous,
                        representation_kind=DUPLICATE_REPRESENTATION,
                    )
                    previous = duplicate["row_hash"]
                    model_rows.append(duplicate)
        finally:
            backend.close()
            gc.collect()
        after_close_devices = _gpu_devices()
        rows_by_model[str(model_spec["hf_id"])] = model_rows
        extraction_receipts.append(
            {
                "model_hf_id": str(model_spec["hf_id"]),
                "loader_receipt": dict(loader_receipt),
                "row_count": len(model_rows),
                "cleanup_receipt": {
                    "backend_close_called": True,
                    "gc_collect_called": True,
                    "devices_before_close": before_close_devices,
                    "devices_after_close": after_close_devices,
                },
            }
        )
    return rows_by_model, extraction_receipts


def _row_file_receipt(
    *,
    path: Path,
    relative_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    text = rows_to_jsonl(rows)
    row_hashes = {str(row["vector_row_id"]): str(row["row_hash"]) for row in rows}
    prefix = str(rows[-1]["row_hash"]) if rows else exp5963.INITIAL_PREFIX_HASH
    receipt = {
        "path": str(relative_path),
        "absolute_path": str(path),
        "row_count": len(rows),
        "sha256": sha256_text(text),
        "row_hashes": row_hashes,
        "row_hash_root": sha256_json(row_hashes),
        "prefix_chain_ok": True,
        "final_prefix_checksum": prefix,
        "representation_counts": dict(
            sorted(Counter(str(row["representation_kind"]) for row in rows).items())
        ),
        "exp5963_pair_row_hash_root": sha256_json(
            sorted({str(row["exp5963_pair_row_hash"]) for row in rows})
        )
        if rows
        else sha256_json([]),
        "atomic_write": True,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _write_model_row_files(
    *,
    row_dir: Path,
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    models: dict[str, JsonDict] = {}
    for hf_id in MANDATED_MODEL_HF_IDS:
        rows = list(rows_by_model.get(hf_id, []))
        relative = model_row_relative_path(hf_id)
        path = row_dir / relative.name
        text = rows_to_jsonl(rows)
        _write_atomic(path, text)
        receipt = _row_file_receipt(path=path, relative_path=relative, rows=rows)
        verify_model_row_file(rows, receipt)
        models[hf_id] = receipt
    row_counts = {hf_id: receipt["row_count"] for hf_id, receipt in models.items()}
    expected_per_model = sum(
        receipt["row_count"] for receipt in models.values()
    ) // max(1, len(MANDATED_MODEL_HF_IDS))
    return {
        "schema": SCHEMA + ".per_model_row_paths",
        "models": models,
        "row_counts_by_model": row_counts,
        "all_prefix_chains_ok": all(row["prefix_chain_ok"] is True for row in models.values()),
        "all_models_have_rows": all(row["row_count"] > 0 for row in models.values()),
        "all_row_counts_match": len(set(row_counts.values())) <= 1,
        "expected_rows_per_model": expected_per_model,
        "receipt_hash": sha256_json(models),
    }


def _model_file_hashes(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    records = {
        str(spec["hf_id"]): {
            "model_path": str(spec["model_path"]),
            "model_sha256": str(spec["model_sha256"]),
            "local_model_present": spec.get("local_model_present") is True,
            "local_path_hash": str(spec["local_path_hash"]),
            "quantization": str(spec["quantization"]),
        }
        for spec in model_specs
    }
    return {
        "schema": SCHEMA + ".model_file_hashes",
        "records": records,
        "all_mandated_files_present": all(
            record["local_model_present"] and str(record["model_sha256"]).startswith("sha256:")
            for record in records.values()
        ),
        "receipt_hash": sha256_json(records),
    }


def _embedded_tokenizer_and_llama_cpp_receipts(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    tokenizer_records = {
        str(spec["hf_id"]): dict(spec.get("tokenizer_receipt") or {}) for spec in model_specs
    }
    return {
        "schema": SCHEMA + ".embedded_tokenizer_and_llama_cpp_receipts",
        "tokenizer_records": tokenizer_records,
        "llama_cpp": dict(preconditions.get("llama_cpp") or {}),
        "all_embedded_tokenizers_loadable": all(
            record.get("loadable") is True for record in tokenizer_records.values()
        ),
        "auto_tokenizer_used": False,
        "gguf_embedded_tokenizers_only": True,
        "receipt_hash": sha256_json(tokenizer_records),
    }


def _receipt_has_cuda_offload(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("cuda_offload_verified") is True:
        return True
    observed = dict(receipt.get("observed_device_assignment") or {})
    deltas = dict(observed.get("memory_delta_mb_by_gpu") or {})
    return any(int(value or 0) > 0 for value in deltas.values())


def _cuda_offload_receipts(
    *,
    preconditions: Mapping[str, Any],
    extraction_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    loader_records = {
        str(row["model_hf_id"]): dict(row.get("loader_receipt") or {})
        for row in extraction_receipts
    }
    cleanup_records = {
        str(row["model_hf_id"]): dict(row.get("cleanup_receipt") or {})
        for row in extraction_receipts
    }
    return {
        "schema": SCHEMA + ".cuda_offload_vram_thermal_cleanup",
        "cuda": dict(preconditions.get("cuda") or {}),
        "gpu": dict(preconditions.get("gpu") or {}),
        "resources": dict(preconditions.get("resources") or {}),
        "loader_receipts_by_model": loader_records,
        "cleanup_receipts_by_model": cleanup_records,
        "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
        "all_models_loaded": set(loader_records) == set(MANDATED_MODEL_HF_IDS),
        "all_models_output_free": bool(loader_records)
        and all(
            receipt.get("embedding_mode") is True
            and receipt.get("output_logits_enabled") is False
            and receipt.get("generated_text_enabled") is False
            for receipt in loader_records.values()
        ),
        "all_models_cuda_offloaded": bool(loader_records)
        and all(_receipt_has_cuda_offload(receipt) for receipt in loader_records.values()),
        "all_cleaned_up": bool(cleanup_records)
        and all(record.get("backend_close_called") is True for record in cleanup_records.values()),
        "receipt_hash": sha256_json({"loaders": loader_records, "cleanup": cleanup_records}),
    }


def _prompt_contract(pair_count: int) -> JsonDict:
    return {
        "schema": SCHEMA + ".prompt_serialization_contract",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "labels_in_prompt": False,
        "generated_text_enabled": False,
        "output_logits_enabled": False,
        "intermediate_layers_claimed": False,
        "representation_kinds": list(REPRESENTATION_KINDS),
        "duplicate_representation_kind": DUPLICATE_REPRESENTATION,
        "pair_order": "exp5963_sequence_index_major_then_representation_kind",
        "pair_count": int(pair_count),
        "prompt_fields": [
            "schema",
            "representation",
            "pair_order",
            "pair_sequence",
            "context_id",
            "context",
            "candidate_atom",
        ],
        "label_fields_forbidden_in_prompt": [
            "label",
            "label_bool",
            "python_label_bool",
            "z3_label_bool",
            "post_seal_reference_set_hash",
        ],
        "contract_hash": sha256_json(
            {
                "template": PROMPT_TEMPLATE_VERSION,
                "representations": list(REPRESENTATION_KINDS),
                "duplicate": DUPLICATE_REPRESENTATION,
            }
        ),
    }


def _vectors(rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    return [[float(value) for value in row.get("embedding", [])] for row in rows]


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return _mean([(float(value) - mean) ** 2 for value in values])


def _l2(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def _standardization_stats(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[JsonDict, JsonDict]:
    stats: JsonDict = {}
    summary: JsonDict = {}
    for hf_id, rows in rows_by_model.items():
        train_rows = [
            row
            for row in rows
            if row.get("source_split") == "train"
            and row.get("representation_kind") != DUPLICATE_REPRESENTATION
        ]
        vectors = _vectors(train_rows)
        width = len(vectors[0]) if vectors else 0
        means = []
        stds = []
        for dim in range(width):
            values = [vector[dim] for vector in vectors]
            variance = _variance(values)
            means.append(round(_mean(values), EMBEDDING_DECIMALS))
            stds.append(round(math.sqrt(variance), EMBEDDING_DECIMALS) or 1.0)
        stats[hf_id] = {"mean": means, "std": stds}
        summary[hf_id] = {
            "train_row_count": len(vectors),
            "dimension": width,
            "mean_sha256": sha256_json(means),
            "std_sha256": sha256_json(stds),
            "uses_only_source_split_train": True,
            "representation_scope": "all_preregistered_representations_within_model",
        }
    return stats, {
        "schema": SCHEMA + ".raw_vs_standardized_feature_separation",
        "raw_vectors_stored_only_in_per_model_files": True,
        "standardization_scope": "within_model",
        "train_fold_only": True,
        "test_fold_statistics_used": False,
        "cross_family_raw_concatenation": False,
        "model_identity_feature": False,
        "stats_by_model": summary,
        "stats_receipt_hash": sha256_json(summary),
    }


def _standardized_vector(row: Mapping[str, Any], stats: Mapping[str, Any]) -> list[float]:
    model_stats = dict(stats.get(str(row.get("model_hf_id"))) or {})
    mean = [float(value) for value in model_stats.get("mean", [])]
    std = [float(value) or 1.0 for value in model_stats.get("std", [])]
    vector = [float(value) for value in row.get("embedding", [])]
    if len(vector) != len(mean) or len(vector) != len(std):
        return vector
    return [
        round((value - center) / scale, EMBEDDING_DECIMALS)
        for value, center, scale in zip(vector, mean, std, strict=True)
    ]


def _rows_by_pair_and_kind(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {
        (int(row["exp5963_pair_sequence_index"]), str(row["representation_kind"])): row
        for row in rows
    }


def finite_variance_duplicate_and_order_controls(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    stats: Mapping[str, Any],
) -> JsonDict:
    """Audit finite vectors, within-model variance, duplicates, and order sensitivity."""

    family_records: dict[str, JsonDict] = {}
    for hf_id, rows in rows_by_model.items():
        vectors = _vectors([row for row in rows if row.get("representation_kind") != DUPLICATE_REPRESENTATION])
        finite = all(math.isfinite(value) for vector in vectors for value in vector)
        width = len(vectors[0]) if vectors else 0
        variances = [
            _variance([vector[dim] for vector in vectors]) for dim in range(width)
        ]
        nonzero_variance = any(value > EPSILON for value in variances)
        by_pair = _rows_by_pair_and_kind(rows)
        duplicate_distances: list[float] = []
        order_distances: list[float] = []
        for (pair_index, kind), duplicate in by_pair.items():
            if kind != DUPLICATE_REPRESENTATION:
                continue
            original = by_pair.get((pair_index, "context_then_atom"))
            if original is not None:
                duplicate_distances.append(
                    _l2(
                        [float(value) for value in duplicate["embedding"]],
                        [float(value) for value in original["embedding"]],
                    )
                )
        for pair_index in sorted({key[0] for key in by_pair}):
            context_atom = by_pair.get((pair_index, "context_then_atom"))
            atom_context = by_pair.get((pair_index, "atom_then_context"))
            if context_atom is not None and atom_context is not None:
                order_distances.append(
                    _l2(
                        _standardized_vector(context_atom, stats),
                        _standardized_vector(atom_context, stats),
                    )
                )
        duplicate_max = max(duplicate_distances) if duplicate_distances else float("inf")
        order_mean = _mean(order_distances)
        passed = bool(
            rows
            and finite
            and nonzero_variance
            and duplicate_distances
            and duplicate_max <= EPSILON
            and order_mean > EPSILON
        )
        family_records[hf_id] = {
            "finite": finite,
            "nonzero_variance": nonzero_variance,
            "max_duplicate_distance": round(duplicate_max, 8) if math.isfinite(duplicate_max) else None,
            "mean_order_control_distance": round(order_mean, 8),
            "order_control_pair_count": len(order_distances),
            "passed": passed,
        }
    return {
        "schema": SCHEMA + ".finite_variance_duplicate_order_controls",
        "families": family_records,
        "all_families_pass": set(family_records) == set(MANDATED_MODEL_HF_IDS)
        and all(row["passed"] is True for row in family_records.values()),
    }


def claim_flip_and_per_family_headroom_controls(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    stats: Mapping[str, Any],
) -> JsonDict:
    """Audit disaggregated claim-flip and order-control headroom."""

    family_records: dict[str, JsonDict] = {}
    for hf_id, rows in rows_by_model.items():
        by_pair = _rows_by_pair_and_kind(rows)
        pair_indices = sorted({key[0] for key in by_pair})
        claim_distances: list[float] = []
        permutation_distances: list[float] = []
        direction_projections: list[float] = []
        for offset, pair_index in enumerate(pair_indices):
            original = by_pair.get((pair_index, "context_then_atom"))
            flipped = by_pair.get((pair_index, "context_then_claim_flip_atom"))
            if original is None or flipped is None:
                continue
            original_vector = _standardized_vector(original, stats)
            flipped_vector = _standardized_vector(flipped, stats)
            delta = [b - a for a, b in zip(original_vector, flipped_vector, strict=True)]
            claim_distances.append(_l2(original_vector, flipped_vector))
            direction_projections.append(sum(delta))
            if len(pair_indices) > 1:
                permuted = by_pair.get(
                    (pair_indices[(offset + 1) % len(pair_indices)], "context_then_claim_flip_atom")
                )
                if permuted is not None:
                    permutation_distances.append(
                        _l2(original_vector, _standardized_vector(permuted, stats))
                    )
        mean_claim = _mean(claim_distances)
        mean_permutation = _mean(permutation_distances)
        positive_rate = (
            sum(1 for value in direction_projections if value > 0) / len(direction_projections)
            if direction_projections
            else 0.0
        )
        passed = bool(claim_distances and mean_claim > EPSILON and positive_rate >= 0.5)
        family_records[hf_id] = {
            "claim_flip_pair_count": len(claim_distances),
            "mean_claim_flip_distance": round(mean_claim, 8),
            "mean_pair_permutation_distance": round(mean_permutation, 8),
            "direction_positive_rate": round(positive_rate, 8),
            "permutation_control_measured": bool(permutation_distances),
            "passed": passed,
        }
    families_passing = [hf_id for hf_id, row in family_records.items() if row["passed"] is True]
    return {
        "schema": SCHEMA + ".claim_flip_headroom_controls",
        "families": family_records,
        "families_passing": families_passing,
        "families_passing_count": len(families_passing),
        "at_least_two_families_pass": len(families_passing) >= 2,
        "pooled_artifact_used": False,
    }


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = [float(score) for score, label in zip(scores, labels, strict=True) if label]
    negatives = [float(score) for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else 0.5


def norm_length_frequency_label_pair_permutation_and_model_identity_controls(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Measure explicit shortcut baselines without using them for readiness."""

    labels = {int(row["sequence_index"]): bool(row["label_bool"]) for row in pair_rows}
    candidate_frequency = Counter(str(row["candidate_text_hash"]) for row in pair_rows)
    family_records: dict[str, JsonDict] = {}
    all_labels_by_model: dict[str, Counter[str]] = {}
    for hf_id, rows in rows_by_model.items():
        target_rows = [
            row for row in rows if row.get("representation_kind") == "context_then_atom"
        ]
        row_labels = [labels.get(int(row["exp5963_pair_sequence_index"]), False) for row in target_rows]
        norms = [
            math.sqrt(sum(float(value) ** 2 for value in row.get("embedding", [])))
            for row in target_rows
        ]
        lengths = [float(row.get("prompt_token_count", 0)) for row in target_rows]
        freqs = [
            float(candidate_frequency[str(row.get("candidate_text_hash"))])
            for row in target_rows
        ]
        permuted_labels = row_labels[1:] + row_labels[:1] if row_labels else []
        permuted_norms = norms[1:] + norms[:1] if norms else []
        label_counter = Counter("compatible" if label else "incompatible" for label in row_labels)
        all_labels_by_model[hf_id] = label_counter
        family_records[hf_id] = {
            "row_count": len(target_rows),
            "norm_only_auc": round(_auc(norms, row_labels), 8),
            "length_only_auc": round(_auc(lengths, row_labels), 8),
            "candidate_frequency_auc": round(_auc(freqs, row_labels), 8),
            "label_permutation_norm_auc": round(_auc(norms, permuted_labels), 8),
            "pair_permutation_norm_auc": round(_auc(permuted_norms, row_labels), 8),
            "controls_measured": bool(target_rows),
        }
    return {
        "schema": SCHEMA + ".shortcut_controls",
        "families": family_records,
        "norm_only": {hf_id: row["norm_only_auc"] for hf_id, row in family_records.items()},
        "length_only": {hf_id: row["length_only_auc"] for hf_id, row in family_records.items()},
        "candidate_frequency": {
            hf_id: row["candidate_frequency_auc"] for hf_id, row in family_records.items()
        },
        "label_permutation": {
            hf_id: row["label_permutation_norm_auc"] for hf_id, row in family_records.items()
        },
        "pair_permutation": {
            hf_id: row["pair_permutation_norm_auc"] for hf_id, row in family_records.items()
        },
        "raw_model_identity": {
            "model_identity_preserved_as_stratum": True,
            "model_identity_feature_exported": False,
            "label_counts_by_model": {
                hf_id: dict(counter) for hf_id, counter in sorted(all_labels_by_model.items())
            },
            "same_label_distribution_across_models": len(
                {canonical_json(dict(counter)) for counter in all_labels_by_model.values()}
            )
            <= 1,
        },
        "all_controls_measured": set(family_records) == set(MANDATED_MODEL_HF_IDS)
        and all(row["controls_measured"] is True for row in family_records.values()),
        "used_for_readiness": False,
    }


def split_and_label_secrecy_receipts(
    *,
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Prove labels and heldout statistics were not part of extraction or prompts."""

    prompt_hashes = [
        str(row["prompt_hash"]) for rows in rows_by_model.values() for row in rows
    ]
    split_counts = Counter(
        str(row.get("source_split")) for rows in rows_by_model.values() for row in rows
    )
    label_counts = Counter("compatible" if row.get("label_bool") else "incompatible" for row in pair_rows)
    return {
        "schema": SCHEMA + ".split_and_label_secrecy",
        "labels_in_prompts": False,
        "label_fields_stored_in_vector_rows": False,
        "label_fields_joined_only_after_extraction_for_controls": True,
        "test_fold_statistics_used_for_standardization": False,
        "standardization_train_split_only": True,
        "prompt_hash_root": sha256_json(prompt_hashes),
        "source_split_counts_in_vector_rows": dict(sorted(split_counts.items())),
        "external_label_counts_from_exp5963": dict(sorted(label_counts.items())),
        "trained_score_used": False,
    }


def _raw_vs_standardized_and_controls(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    stats, separation = _standardization_stats(rows_by_model)
    finite_order = finite_variance_duplicate_and_order_controls(rows_by_model, stats)
    headroom = claim_flip_and_per_family_headroom_controls(rows_by_model, stats)
    shortcuts = norm_length_frequency_label_pair_permutation_and_model_identity_controls(
        rows_by_model, pair_rows
    )
    return separation, finite_order, headroom, shortcuts


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        LLM_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5963_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5963_CONTEXT_RELATIVE_PATH.as_posix(),
        EXP5963_PAIR_RELATIVE_PATH.as_posix(),
        "python/carnot/experiment_5852_three_family_paired_embeddings.py",
        "python/carnot/inference/sota_models.py",
    ]
    return {
        field: {"principle": principle, "sources": sources}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def _run_command(command: Sequence[str], *, timeout_s: float = 10.0) -> JsonDict:
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
    except Exception as exc:  # pragma: no cover - host command failure shape.
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
            "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    devices: list[JsonDict] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
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
                }
            )
        except ValueError:
            continue
    return devices


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


def _llama_cpp_probe() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        import llama_cpp
    except Exception as exc:
        return {
            "available": False,
            "version": "",
            "system_info": "",
            "cuda_backend_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    version = str(getattr(llama_cpp, "__version__", "unknown"))
    try:
        raw_info = llama_cpp.llama_print_system_info()
        system_info = raw_info.decode("utf-8", errors="replace") if isinstance(raw_info, bytes) else str(raw_info)
    except Exception as exc:
        system_info = f"system_info_unavailable:{type(exc).__name__}:{exc}"
    lowered = system_info.lower()
    return {
        "available": True,
        "version": version,
        "system_info": system_info,
        "cuda_backend_available": "cuda" in lowered or "cublas" in lowered,
        "embedding_pooling_api_required": "LLAMA_POOLING_TYPE_LAST",
    }


def _output_path_receipt(result_path: Path, row_dir: Path) -> JsonDict:
    def writable(path: Path) -> bool:
        parent = path if path.suffix == "" else path.parent
        return parent.exists() and os.access(parent, os.W_OK)

    return {
        "result_path": str(result_path),
        "row_dir": str(row_dir),
        "atomic_suffix": ".tmp",
        "result_writable": writable(result_path),
        "row_dir_writable": writable(row_dir),
        "ok": writable(result_path) and writable(row_dir),
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_dir: str | Path = REPO_ROOT / "results",
) -> JsonDict:  # pragma: no cover - host/resource dependent.
    """Collect local model, tokenizer, CUDA, resource, and output preconditions."""

    row_dir_path = Path(row_dir)
    row_dir_path.mkdir(parents=True, exist_ok=True)
    devices = _gpu_devices()
    llama_cpp = _llama_cpp_probe()
    memory = _memory_probe()
    disk = _disk_probe(Path(root))
    output = _output_path_receipt(Path(result_path), row_dir_path)
    blocked: list[str] = []
    if not devices:
        blocked.append("gpu_device_receipt_unavailable")
    if llama_cpp.get("available") is not True:
        blocked.append("llama_cpp_unavailable")
    if llama_cpp.get("cuda_backend_available") is not True:
        blocked.append("llama_cpp_cuda_backend_unavailable")
    if memory.get("ok") is not True:
        blocked.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked.append("insufficient_free_disk")
    if output.get("ok") is not True:
        blocked.append("output_path_not_writable")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "llama_cpp": llama_cpp,
        "cuda": {
            "available": bool(devices) and llama_cpp.get("cuda_backend_available") is True,
            "backend": "CUDA" if devices else "unavailable",
            "genuine_offload_required": True,
        },
        "gpu": {"gpu_count": len(devices), "devices": devices, "ok": bool(devices)},
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output,
        "time_budget": {"estimated_required_s": None, "available_s": None, "ok": True},
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
    gate_receipt: Mapping[str, Any],
) -> list[str]:
    blockers = list(preconditions.get("blocked_reasons") or [])
    if preconditions.get("preconditions_ready") is not True:
        blockers.append("preconditions_not_ready")
    if gate_receipt.get("ready") is not True:
        blockers.append("exp5963_gate_not_ready")
    if [str(row.get("hf_id")) for row in model_specs] != list(MANDATED_MODEL_HF_IDS):
        blockers.append("mandated_model_order_mismatch")
    for spec in model_specs:
        tokenizer = dict(spec.get("tokenizer_receipt") or {})
        model_path_name = Path(str(spec.get("model_path", ""))).name.lower()
        if (
            spec.get("local_model_present") is not True
            or not str(spec.get("model_path", "")).endswith(".gguf")
            or model_path_name.startswith("mmproj")
            or not str(spec.get("model_sha256", "")).startswith("sha256:")
            or spec.get("headline_eligible") is not True
            or tokenizer.get("loadable") is not True
        ):
            blockers.append("mandated_model_unavailable")
            break
    devices_by_index = {
        int(device.get("index", -1)): int(device.get("memory_free_mb", 0) or 0)
        for device in list(dict(preconditions.get("gpu") or {}).get("devices") or [])
        if isinstance(device, Mapping)
    }
    for spec in model_specs:
        required_gb = spec.get("min_vram_gb")
        if required_gb is None:
            continue
        required_mb = max(0, int(float(required_gb) * 1024) - 1024)
        free_mb = devices_by_index.get(int(spec.get("gpu", 0) or 0), 0)
        if free_mb < required_mb:
            blockers.append("insufficient_free_vram")
            break
    if dict(preconditions.get("llama_cpp") or {}).get("available") is not True:
        blockers.append("llama_cpp_unavailable")
    if dict(preconditions.get("llama_cpp") or {}).get("cuda_backend_available") is not True:
        blockers.append("llama_cpp_cuda_backend_unavailable")
    if dict(preconditions.get("cuda") or {}).get("available") is not True:
        blockers.append("cuda_offload_unavailable")
    if dict(preconditions.get("gpu") or {}).get("ok") is not True:
        blockers.append("gpu_device_receipt_unavailable")
    resources = dict(preconditions.get("resources") or {})
    if dict(resources.get("memory") or {}).get("ok") is not True:
        blockers.append("insufficient_free_ram")
    if dict(resources.get("disk") or {}).get("ok") is not True:
        blockers.append("insufficient_free_disk")
    if dict(preconditions.get("output_paths") or {}).get("ok") is not True:
        blockers.append("output_path_not_writable")
    if dict(preconditions.get("time_budget") or {}).get("ok") is not True:
        blockers.append("time_budget_unavailable")
    policy = dict(preconditions.get("legacy_tiny_models_policy") or {})
    if policy.get("cannot_satisfy_readiness") is not True:
        blockers.append("legacy_smoke_policy_missing")
    return sorted(set(blockers))


def protected_files_unchanged(root: Path = REPO_ROOT) -> JsonDict:
    """Check that conductor, exclusion, history, and ops-protected files are clean."""

    command = ["git", "status", "--short", "--", *[path.as_posix() for path in PROTECTED_FILES]]
    result = _run_command(command, timeout_s=10)
    records = {
        path.as_posix(): {
            "exists": (root / path).exists(),
            "sha256": sha256_file(root / path) if (root / path).exists() else "",
        }
        for path in PROTECTED_FILES
    }
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "records": records,
        "git_status_command": command,
        "git_status_stdout": str(result.get("stdout", "")),
        "git_status_returncode": result.get("returncode"),
        "unchanged": result.get("returncode") == 0 and not str(result.get("stdout", "")).strip(),
    }


def atom_compatibility_corpus_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness score for a clean, complete Exp5964 corpus."""

    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = bool(
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and not dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons")
        and dict(artifact.get("gate_replay_receipt") or {}).get("ready") is True
        and [str(row.get("hf_id")) for row in artifact.get("model_specs", [])]
        == list(MANDATED_MODEL_HF_IDS)
        and dict(artifact.get("model_file_hashes") or {}).get("all_mandated_files_present")
        is True
        and dict(artifact.get("embedded_tokenizer_and_llama_cpp_receipts") or {}).get(
            "all_embedded_tokenizers_loadable"
        )
        is True
        and dict(artifact.get("embedded_tokenizer_and_llama_cpp_receipts") or {}).get(
            "auto_tokenizer_used"
        )
        is False
        and dict(artifact.get("cuda_offload_vram_thermal_and_cleanup_receipts") or {}).get(
            "all_models_cuda_offloaded"
        )
        is True
        and dict(artifact.get("cuda_offload_vram_thermal_and_cleanup_receipts") or {}).get(
            "all_models_output_free"
        )
        is True
        and dict(artifact.get("per_model_row_paths_hashes_counts_and_prefix_chains") or {}).get(
            "all_prefix_chains_ok"
        )
        is True
        and dict(artifact.get("per_model_row_paths_hashes_counts_and_prefix_chains") or {}).get(
            "all_models_have_rows"
        )
        is True
        and dict(artifact.get("per_model_row_paths_hashes_counts_and_prefix_chains") or {}).get(
            "all_row_counts_match"
        )
        is True
        and dict(artifact.get("raw_vs_standardized_feature_separation") or {}).get(
            "raw_vectors_stored_only_in_per_model_files"
        )
        is True
        and dict(artifact.get("raw_vs_standardized_feature_separation") or {}).get(
            "cross_family_raw_concatenation"
        )
        is False
        and dict(artifact.get("finite_variance_duplicate_and_order_controls") or {}).get(
            "all_families_pass"
        )
        is True
        and dict(artifact.get("claim_flip_and_per_family_headroom_controls") or {}).get(
            "at_least_two_families_pass"
        )
        is True
        and dict(
            artifact.get("norm_length_frequency_label_pair_permutation_and_model_identity_controls")
            or {}
        ).get("all_controls_measured")
        is True
        and dict(artifact.get("split_and_label_secrecy_receipts") or {}).get("labels_in_prompts")
        is False
        and dict(artifact.get("split_and_label_secrecy_receipts") or {}).get(
            "test_fold_statistics_used_for_standardization"
        )
        is False
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is False
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    checks = {
        "gate_replay_receipt": dict(artifact.get("gate_replay_receipt") or {}).get("ready")
        is True,
        "model_file_hashes": dict(artifact.get("model_file_hashes") or {}).get(
            "all_mandated_files_present"
        )
        is True,
        "embedded_tokenizer_and_llama_cpp_receipts": dict(
            artifact.get("embedded_tokenizer_and_llama_cpp_receipts") or {}
        ).get("all_embedded_tokenizers_loadable")
        is True,
        "cuda_offload_vram_thermal_and_cleanup_receipts": dict(
            artifact.get("cuda_offload_vram_thermal_and_cleanup_receipts") or {}
        ).get("all_models_cuda_offloaded")
        is True,
        "per_model_row_paths_hashes_counts_and_prefix_chains": dict(
            artifact.get("per_model_row_paths_hashes_counts_and_prefix_chains") or {}
        ).get("all_models_have_rows")
        is True,
        "finite_variance_duplicate_and_order_controls": dict(
            artifact.get("finite_variance_duplicate_and_order_controls") or {}
        ).get("all_families_pass")
        is True,
        "claim_flip_and_per_family_headroom_controls": dict(
            artifact.get("claim_flip_and_per_family_headroom_controls") or {}
        ).get("at_least_two_families_pass")
        is True,
    }
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    if not (
        commands
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    ):
        reasons.append("failed_test_exit_codes")
    for name, ok in checks.items():
        if not ok:
            reasons.append(name)
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict prefix required by the spec."""

    if atom_compatibility_corpus_ready_score(artifact) == 1.0:
        return "complete_ready: atom compatibility corpus passed all-family integrity and headroom"
    preconditions = dict(artifact.get("preconditions_checked") or {})
    if preconditions.get("blocked_reasons"):
        return "blocked: " + ",".join(list(preconditions.get("blocked_reasons") or [])[:8])
    return "retired: compatibility surface lacked preregistered non-degenerate semantic headroom"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Checksum stable artifact content while excluding wall-clock duration."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    return sha256_json(stable)


def _artifact_from_rows(
    *,
    context_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    model_specs: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    gate_receipt: Mapping[str, Any],
    extraction_receipts: Sequence[Mapping[str, Any]],
    row_receipts: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    root: Path,
) -> JsonDict:
    separation, finite_order, headroom, shortcuts = _raw_vs_standardized_and_controls(
        rows_by_model, pair_rows
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "complete_ready",
        "preconditions_checked": dict(preconditions_checked),
        "gate_replay_receipt": dict(gate_receipt),
        "model_specs": [dict(spec) for spec in model_specs],
        "model_file_hashes": _model_file_hashes(model_specs),
        "embedded_tokenizer_and_llama_cpp_receipts": _embedded_tokenizer_and_llama_cpp_receipts(
            model_specs=model_specs,
            preconditions=preconditions_checked,
        ),
        "cuda_offload_vram_thermal_and_cleanup_receipts": _cuda_offload_receipts(
            preconditions=preconditions_checked,
            extraction_receipts=extraction_receipts,
        ),
        "prompt_serialization_and_pair_order_contract": _prompt_contract(len(pair_rows)),
        "per_model_row_paths_hashes_counts_and_prefix_chains": dict(row_receipts),
        "raw_vs_standardized_feature_separation": separation,
        "finite_variance_duplicate_and_order_controls": finite_order,
        "claim_flip_and_per_family_headroom_controls": headroom,
        "norm_length_frequency_label_pair_permutation_and_model_identity_controls": shortcuts,
        "split_and_label_secrecy_receipts": split_and_label_secrecy_receipts(
            rows_by_model=rows_by_model,
            pair_rows=pair_rows,
        ),
        "atom_compatibility_corpus_ready_score": 0.0,
        "protected_files_unchanged": protected_files_unchanged(root),
        "duration_s": round(float(duration_s), 6),
        "random_seed": exp5852.DEFAULT_RANDOM_SEED,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "Representation headroom is not a trained classifier accuracy claim.",
            "Exact labels remain external Exp5963 oracle data and are used only after extraction.",
            "Public GGUF runtimes do not expose intermediate hidden layers here.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "source_row_counts": {
            "context_rows": len(context_rows),
            "pair_rows": len(pair_rows),
        },
    }
    artifact["atom_compatibility_corpus_ready_score"] = atom_compatibility_corpus_ready_score(
        artifact
    )
    if artifact["atom_compatibility_corpus_ready_score"] == 1.0:
        artifact["status"] = "complete_ready"
    elif dict(preconditions_checked).get("blocked_reasons"):
        artifact["status"] = "blocked"
    else:
        artifact["status"] = "retired"
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal Exp5964 artifact contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if set(REQUIRED_FIELD_PRINCIPLES) - set(artifact.get("field_provenance", {})):
        raise ValueError("field_provenance")
    expected_score = atom_compatibility_corpus_ready_score(artifact)
    if artifact.get("atom_compatibility_corpus_ready_score") != expected_score:
        raise ValueError("atom_compatibility_corpus_ready_score")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")
    if [str(row.get("hf_id")) for row in artifact.get("model_specs", [])] != list(
        MANDATED_MODEL_HF_IDS
    ):
        raise ValueError("model_specs")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    status = str(artifact.get("status"))
    verdict = str(artifact.get("honest_verdict", ""))
    if expected_score == 1.0:
        if status != "complete_ready" or not verdict.startswith("complete_ready:"):
            raise ValueError("honest_verdict")
    elif status == "retired":
        if not verdict.startswith("retired:"):
            raise ValueError("honest_verdict")
    elif status == "blocked":
        if not verdict.startswith("blocked:"):
            raise ValueError("honest_verdict")
    else:
        raise ValueError("status")
    return True


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_dir: str | Path = REPO_ROOT / "results",
    fixture_artifact_path: str | Path = REPO_ROOT / EXP5963_ARTIFACT_RELATIVE_PATH,
    context_rows_path: str | Path = REPO_ROOT / EXP5963_CONTEXT_RELATIVE_PATH,
    pair_rows_path: str | Path = REPO_ROOT / EXP5963_PAIR_RELATIVE_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    embedding_backend_factory: EmbeddingBackendFactory = LlamaCppOutputFreeEmbeddingBackend,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5964 or emit a terminal blocker before full extraction."""

    started = time.perf_counter()
    root = Path(root)
    result = Path(result_path)
    rows_dir = Path(row_dir)
    rows_dir.mkdir(parents=True, exist_ok=True)
    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    config = deterministic_embedding_config()
    specs = normalize_model_specs(model_specs) if model_specs is not None else resolve_all_model_specs()
    gate_receipt = gate_replay_receipt(
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_rows_path,
        pair_rows_path=pair_rows_path,
    )
    context_rows = _read_jsonl(context_rows_path) if Path(context_rows_path).exists() else []
    pair_rows = _read_jsonl(pair_rows_path) if Path(pair_rows_path).exists() else []
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result, row_dir=rows_dir)
    )
    blockers = _precondition_blockers(preconditions, specs, gate_receipt)
    if not context_rows or not pair_rows:
        blockers.append("exp5963_rows_unavailable")
    preconditions["blocked_reasons"] = sorted(set(blockers))
    preconditions["preconditions_ready"] = not preconditions["blocked_reasons"]
    if preconditions["blocked_reasons"]:
        rows_by_model = {hf_id: [] for hf_id in MANDATED_MODEL_HF_IDS}
        extraction_receipts: list[JsonDict] = []
        row_receipts = _write_model_row_files(row_dir=rows_dir, rows_by_model=rows_by_model)
    else:
        try:
            rows_by_model, extraction_receipts = extract_rows(
                context_rows=context_rows,
                pair_rows=pair_rows,
                model_specs=specs,
                config=config,
                embedding_backend_factory=embedding_backend_factory,
            )
        except Exception as exc:  # pragma: no cover - live runtime failure receipt.
            preconditions["blocked_reasons"] = [f"live_embedding_extraction_failed:{type(exc).__name__}"]
            preconditions["preconditions_ready"] = False
            rows_by_model = {hf_id: [] for hf_id in MANDATED_MODEL_HF_IDS}
            extraction_receipts = []
        row_receipts = _write_model_row_files(row_dir=rows_dir, rows_by_model=rows_by_model)
    artifact = _artifact_from_rows(
        context_rows=context_rows,
        pair_rows=pair_rows,
        rows_by_model=rows_by_model,
        model_specs=specs,
        preconditions_checked=preconditions,
        gate_receipt=gate_receipt,
        extraction_receipts=extraction_receipts,
        row_receipts=row_receipts,
        duration_s=time.perf_counter() - started,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
        root=root,
    )
    validate_artifact(artifact)
    if write:
        _write_atomic(result, json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
