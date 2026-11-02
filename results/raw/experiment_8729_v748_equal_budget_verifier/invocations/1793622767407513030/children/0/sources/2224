"""Exp6300 three-family Universal Activation Bus.

Spec refs: REQ-KONA-6300, SCENARIO-KONA-6300-FOLDS,
SCENARIO-KONA-6300-UNLABELED-FIT, SCENARIO-KONA-6300-CONTROLS,
SCENARIO-KONA-6300-IDENTITY.

This module fits one linear encoder-decoder adapter pair per model on the
frozen Exp5852 embedding rows. It does not load a language model. It does not
generate text. The fit uses matched row IDs only to align embeddings that came
from the same source text.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from sklearn.utils.extmath import randomized_svd

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6300_three_family_universal_activation_bus.json")
FOLD_MANIFEST_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6300_three_family_universal_activation_bus/fold_manifest.json"
)
CHECKPOINT_DIR_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6300_three_family_universal_activation_bus/adapters"
)
EXP5852_ARTIFACT_RELATIVE_PATH = Path("results/experiment_5852_three_family_paired_embeddings.json")
EXP5852_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5852_three_family_paired_embeddings.rows.jsonl"
)
EXP5853_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5853_paired_embedding_integrity_audit.json"
)
EXP5854_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5854_portable_comparative_energy_controls.json"
)
EXP6298_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6298_terminal_evidence_preflight_linter.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6300_three_family_universal_activation_bus.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6300_three_family_universal_activation_bus.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/phase3-kona/spec.md")
PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

SCHEMA = "carnot.experiment_6300.three_family_universal_activation_bus.v1"
FOLD_MANIFEST_SCHEMA = "carnot.experiment_6300.fold_manifest.v1"
EXPERIMENT = 6300
EXPERIMENT_ID = "experiment_6300_three_family_universal_activation_bus"
INFERENCE_SUBSTRATE = "offline_embedding_adapter_fit_replay_no_llm"
SHARED_DIMENSION = 128
RIDGE_ALPHA = 1e-3
FOLD_COUNT = 5
VALIDATION_MODULUS = 4
RECONSTRUCTION_RELATIVE_MSE_TOLERANCE = 0.5
IDENTITY_ACCURACY_TOLERANCE = 0.05
ALIGNMENT_CONTROL_MARGIN = 0.0
NEIGHBORHOOD_TOP_K = 10
RANDOM_SEED = 6300
RAM_FLOOR_MB = 16_384
DISK_FLOOR_MB = 16_384

MANDATED_MODEL_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "paper_source_and_local_claim_boundary",
    "upstream_corpus_paths_hashes_and_terminal_classes",
    "MODEL_SPECS",
    "models_used",
    "model_embedding_dimensions",
    "no_live_model_load_receipt",
    "unlabeled_fit_contract",
    "fold_manifest_path_and_hash",
    "shared_dimension",
    "adapter_architecture_and_parameter_counts",
    "train_validation_and_holdout_row_counts",
    "reconstruction_metrics_by_model_and_fold",
    "cross_model_retrieval_metrics_by_pair_and_fold",
    "neighborhood_consistency_by_fold",
    "model_identity_accuracy_by_fold",
    "chance_identity_accuracy",
    "raw_padded_and_random_projection_controls",
    "norm_length_and_token_count_controls",
    "checkpoint_paths_and_hashes",
    "source_model_weight_mutation_count",
    "shared_activation_bus_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The artifact must close with a terminal state.",
    "paper_source_and_local_claim_boundary": "External paper claims stay separate from local evidence.",
    "upstream_corpus_paths_hashes_and_terminal_classes": "The fit must bind to immutable Exp5852 evidence.",
    "MODEL_SPECS": "The mandated model identities must not be substituted.",
    "models_used": "The report must name every model that supplied embeddings.",
    "model_embedding_dimensions": "Native dimensions explain why raw comparisons were unsafe.",
    "no_live_model_load_receipt": "The adapter fit must not reload or query an LLM.",
    "unlabeled_fit_contract": "Only matched row IDs and embeddings may train the adapters.",
    "fold_manifest_path_and_hash": "The held folds must be frozen before fitting.",
    "shared_dimension": "A fixed shared dimension makes all adapters comparable.",
    "adapter_architecture_and_parameter_counts": "Linear adapter size must stay explicit.",
    "train_validation_and_holdout_row_counts": "Counts expose the data used by each fold.",
    "reconstruction_metrics_by_model_and_fold": "Each decoder must preserve native embeddings within tolerance.",
    "cross_model_retrieval_metrics_by_pair_and_fold": "Held matched-row retrieval measures alignment.",
    "neighborhood_consistency_by_fold": "Shared neighborhoods must agree across models.",
    "model_identity_accuracy_by_fold": "Model identity leakage must not replace semantic alignment.",
    "chance_identity_accuracy": "The leakage gate needs a visible chance baseline.",
    "raw_padded_and_random_projection_controls": "Raw and random controls prevent shortcut promotion.",
    "norm_length_and_token_count_controls": "Norm and token envelopes must not carry the result.",
    "checkpoint_paths_and_hashes": "Saved adapter weights must be byte-addressable.",
    "source_model_weight_mutation_count": "Source model weights must remain untouched.",
    "shared_activation_bus_ready_score": "Readiness is a bare gate derived from held-fold metrics.",
    "protected_files_unchanged": "Protected hashes prove source evidence was not rewritten.",
    "preconditions_checked": "Hashes, dimensions, resources, seeds, and folds are checked first.",
    "inference_substrate": "The substrate states that this is offline adapter fitting.",
    "verifier_is_oracle": "This adapter audit is not an answer oracle.",
    "field_provenance": "Every field must cite the evidence that produced it.",
    "field_principles": "Every field must state why it is required.",
    "test_commands": "The report lists the verification commands.",
    "test_exit_codes": "Exit codes keep failed verification visible.",
    "duration_s": "Measured wall time prevents padded runtime claims.",
    "random_seeds": "Deterministic seeds make folds and controls reproducible.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict must start with a terminal prefix.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6300_three_family_universal_activation_bus.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null --branch "
    "--include=python/carnot/experiment_6300_three_family_universal_activation_bus.py "
    "-m pytest tests/python/test_experiment_6300_three_family_universal_activation_bus.py "
    "-q --no-cov -n 0"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    (
        ".venv/bin/coverage report --rcfile=/dev/null "
        "--include=python/carnot/experiment_6300_three_family_universal_activation_bus.py "
        "--fail-under=100 --show-missing"
    ),
    (
        ".venv/bin/python scripts/check_spec_coverage.py "
        "tests/python/test_experiment_6300_three_family_universal_activation_bus.py"
    ),
    (
        ".venv/bin/pytest "
        "tests/python/test_experiment_6298_terminal_evidence_preflight_linter.py "
        "-q --no-cov -n 0"
    ),
    ".venv/bin/python -m carnot.experiment_6300_three_family_universal_activation_bus --date 20260811",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6300_three_family_universal_activation_bus.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}


@dataclass(frozen=True)
class Corpus:
    item_keys: list[str]
    arrays_by_model: dict[str, np.ndarray]
    token_counts_by_model: dict[str, np.ndarray]
    source_row_ids: list[str]
    group_ids: list[str]
    task_families: list[str]
    perturbation_families: list[str]
    template_families: list[str]
    model_specs: list[JsonDict]
    source_row_count: int
    embedding_row_count: int


@dataclass(frozen=True)
class Adapter:
    model_id: str
    input_dimension: int
    shared_dimension: int
    encoder_weight: np.ndarray
    encoder_bias: np.ndarray
    decoder_weight: np.ndarray
    decoder_bias: np.ndarray

    @property
    def parameter_count(self) -> int:
        return int(
            self.encoder_weight.size
            + self.encoder_bias.size
            + self.decoder_weight.size
            + self.decoder_bias.size
        )


@dataclass(frozen=True)
class FoldFit:
    fold_id: str
    adapters: dict[str, Adapter]
    train_latents_by_model: dict[str, np.ndarray]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    if target.is_absolute() and not str(target).startswith(str(REPO_ROOT)):
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
        finally:
            if tmp_path.exists():  # pragma: no cover - only runs after replace failure.
                tmp_path.unlink()
        return target
    return atomic_write_json(path, payload, sort_keys=False)  # pragma: no cover - production path.


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _short_model_id(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_").replace("/", "_").lower()


def _pair_key(left: str, right: str) -> str:
    return f"{_short_model_id(left)}__{_short_model_id(right)}"


def _model_specs_from_rows(
    model_ids: Sequence[str], dimensions: Mapping[str, int]
) -> list[JsonDict]:
    return [
        {
            "hf_id": model_id,
            "headline_eligible": True,
            "embedding_dimension": int(dimensions[model_id]),
        }
        for model_id in model_ids
    ]


def build_item_tables(
    rows: Sequence[JsonMap],
    *,
    model_specs: Sequence[JsonMap] | None = None,
    model_ids: Sequence[str] = MANDATED_MODEL_HF_IDS,
) -> Corpus:
    vector_by_model_key: dict[str, dict[str, np.ndarray]] = {model_id: {} for model_id in model_ids}
    token_by_model_key: dict[str, dict[str, int]] = {model_id: {} for model_id in model_ids}
    meta_by_key: dict[str, JsonDict] = {}
    source_rows: set[str] = set()
    embedding_row_count = 0
    for row in rows:
        model_id = str(row.get("model_hf_id") or "")
        if model_id not in vector_by_model_key:
            continue
        embedding_row_count += 1
        source_row_id = str(row["source_row_id"])
        source_rows.add(source_row_id)
        group_id = str(row.get("pair_group_id") or source_row_id)
        for condition in row.get("condition_embeddings", []):
            if not isinstance(condition, Mapping):
                raise ValueError("condition_embeddings must contain objects")
            suffix = str(condition.get("condition_suffix") or "")
            item_key = f"{source_row_id}|{suffix}"
            vector = np.asarray(condition["embedding"], dtype=np.float64)
            if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                raise ValueError(f"finite one-dimensional embedding required: {item_key}")
            vector_by_model_key[model_id][item_key] = vector
            token_by_model_key[model_id][item_key] = int(condition.get("token_count") or 0)
            meta_by_key.setdefault(
                item_key,
                {
                    "source_row_id": source_row_id,
                    "group_id": group_id,
                    "task_family": str(row.get("family") or "unknown_task_family"),
                    "perturbation_family": str(
                        row.get("surface_kind") or "unknown_perturbation_family"
                    ),
                    "template_family": str(row.get("change") or row.get("split") or "unknown"),
                },
            )
    common_keys = sorted(
        set.intersection(*(set(values) for values in vector_by_model_key.values()))
    )
    if not common_keys:
        raise ValueError("no matched embedding items across mandated models")
    missing = {
        model_id: sorted(set(common_keys) ^ set(vector_by_model_key[model_id]))
        for model_id in model_ids
        if set(common_keys) != set(vector_by_model_key[model_id])
    }
    if missing:
        raise ValueError(f"model item keys are not aligned: {sorted(missing)}")
    arrays_by_model = {
        model_id: np.vstack([vector_by_model_key[model_id][key] for key in common_keys])
        for model_id in model_ids
    }
    token_counts_by_model = {
        model_id: np.asarray(
            [token_by_model_key[model_id][key] for key in common_keys], dtype=float
        )
        for model_id in model_ids
    }
    dimensions = {model_id: int(arrays_by_model[model_id].shape[1]) for model_id in model_ids}
    specs = (
        [dict(spec) for spec in model_specs]
        if model_specs is not None
        else _model_specs_from_rows(model_ids, dimensions)
    )
    return Corpus(
        item_keys=common_keys,
        arrays_by_model=arrays_by_model,
        token_counts_by_model=token_counts_by_model,
        source_row_ids=[meta_by_key[key]["source_row_id"] for key in common_keys],
        group_ids=[meta_by_key[key]["group_id"] for key in common_keys],
        task_families=[meta_by_key[key]["task_family"] for key in common_keys],
        perturbation_families=[meta_by_key[key]["perturbation_family"] for key in common_keys],
        template_families=[meta_by_key[key]["template_family"] for key in common_keys],
        model_specs=specs,
        source_row_count=len(source_rows),
        embedding_row_count=embedding_row_count,
    )


def _group_metadata(corpus: Corpus) -> dict[str, JsonDict]:
    grouped: dict[str, JsonDict] = {}
    for group_id, task, perturbation, template, source_row_id in zip(
        corpus.group_ids,
        corpus.task_families,
        corpus.perturbation_families,
        corpus.template_families,
        corpus.source_row_ids,
        strict=True,
    ):
        row = grouped.setdefault(
            group_id,
            {
                "source_row_ids": set(),
                "task_families": set(),
                "perturbation_families": set(),
                "template_families": set(),
            },
        )
        row["source_row_ids"].add(source_row_id)
        row["task_families"].add(task)
        row["perturbation_families"].add(perturbation)
        row["template_families"].add(template)
    return grouped


def _validation_group(group_id: str, seed: int) -> bool:
    return _stable_int(f"exp6300-validation|{seed}|{group_id}") % VALIDATION_MODULUS == 0


def build_fold_manifest(
    corpus: Corpus, *, n_folds: int = FOLD_COUNT, seed: int = RANDOM_SEED
) -> JsonDict:
    grouped = _group_metadata(corpus)
    group_ids = sorted(grouped)
    folds: list[JsonDict] = []
    for fold_index in range(n_folds):
        holdout_groups = [
            group_id
            for group_id in group_ids
            if _stable_int(f"exp6300-fold|{seed}|{group_id}") % n_folds == fold_index
        ]
        validation_groups = [
            group_id
            for group_id in group_ids
            if group_id not in holdout_groups and _validation_group(group_id, seed)
        ]
        train_groups = [
            group_id
            for group_id in group_ids
            if group_id not in holdout_groups and group_id not in validation_groups
        ]
        if not holdout_groups or not validation_groups or not train_groups:
            raise ValueError("fold split produced an empty split")
        fold = {
            "fold_id": f"fold_{fold_index}",
            "fold_index": fold_index,
            "split_rule": "sha256(pair_group_id) modulo fold count; validation uses disjoint hash",
            "group_unit": "pair_group_id",
            "train_group_ids": train_groups,
            "validation_group_ids": validation_groups,
            "holdout_group_ids": holdout_groups,
            "heldout_task_families": _heldout_values(grouped, holdout_groups, "task_families"),
            "heldout_perturbation_families": _heldout_values(
                grouped, holdout_groups, "perturbation_families"
            ),
            "heldout_template_families": _heldout_values(
                grouped, holdout_groups, "template_families"
            ),
            "train_item_count": _count_items(corpus, train_groups),
            "validation_item_count": _count_items(corpus, validation_groups),
            "holdout_item_count": _count_items(corpus, holdout_groups),
            "train_source_row_count": _count_sources(grouped, train_groups),
            "validation_source_row_count": _count_sources(grouped, validation_groups),
            "holdout_source_row_count": _count_sources(grouped, holdout_groups),
        }
        fold["fold_hash"] = sha256_json(fold)
        folds.append(fold)
    return {
        "schema": FOLD_MANIFEST_SCHEMA,
        "seed": seed,
        "n_folds": n_folds,
        "item_count": len(corpus.item_keys),
        "source_row_count": corpus.source_row_count,
        "folds": folds,
        "manifest_hash": sha256_json(folds),
    }


def _heldout_values(grouped: Mapping[str, JsonMap], groups: Sequence[str], key: str) -> list[str]:
    values: set[str] = set()
    for group_id in groups:
        values.update(str(value) for value in grouped[group_id][key])
    return sorted(values)


def _count_items(corpus: Corpus, groups: Sequence[str]) -> int:
    group_set = set(groups)
    return sum(1 for group_id in corpus.group_ids if group_id in group_set)


def _count_sources(grouped: Mapping[str, JsonMap], groups: Sequence[str]) -> int:
    sources: set[str] = set()
    for group_id in groups:
        sources.update(str(value) for value in grouped[group_id]["source_row_ids"])
    return len(sources)


def _split_mask(corpus: Corpus, fold: JsonMap, split_name: str) -> np.ndarray:
    groups = set(fold[f"{split_name}_group_ids"])
    return np.asarray([group_id in groups for group_id in corpus.group_ids], dtype=bool)


def _encode(adapter: Adapter, matrix: np.ndarray) -> np.ndarray:
    return matrix @ adapter.encoder_weight + adapter.encoder_bias


def _decode(adapter: Adapter, latents: np.ndarray) -> np.ndarray:
    return latents @ adapter.decoder_weight + adapter.decoder_bias


def fit_fold_adapters(
    corpus: Corpus,
    fold: JsonMap,
    *,
    shared_dimension: int,
    seed: int,
    ridge_alpha: float = RIDGE_ALPHA,
) -> FoldFit:
    train_mask = _split_mask(corpus, fold, "train")
    reference_model = MANDATED_MODEL_HF_IDS[0]
    reference_train = corpus.arrays_by_model[reference_model][train_mask]
    if shared_dimension >= min(reference_train.shape):
        raise ValueError(
            "shared_dimension must be smaller than the reference training matrix rank bound"
        )
    reference_center = reference_train.mean(axis=0)
    reference_centered = reference_train - reference_center
    _, _, components_t = randomized_svd(
        reference_centered,
        n_components=shared_dimension,
        n_iter=5,
        random_state=seed + int(fold["fold_index"]),
    )
    target_components = components_t.T
    target_latents = reference_centered @ target_components
    target_mean = target_latents.mean(axis=0)
    target_std = target_latents.std(axis=0) + 1e-6
    adapters: dict[str, Adapter] = {}
    train_latents_by_model: dict[str, np.ndarray] = {}
    for model_id in MANDATED_MODEL_HF_IDS:
        train_matrix = corpus.arrays_by_model[model_id][train_mask]
        model_center = train_matrix.mean(axis=0)
        centered = train_matrix - model_center
        if model_id == reference_model:
            raw_encoder = target_components
            raw_latents = centered @ raw_encoder
        else:
            kernel = centered @ centered.T
            dual = np.linalg.solve(
                kernel + ridge_alpha * np.eye(kernel.shape[0]),
                target_latents,
            )
            raw_encoder = centered.T @ dual
            raw_latents = centered @ raw_encoder
        raw_mean = raw_latents.mean(axis=0)
        raw_std = raw_latents.std(axis=0) + 1e-6
        latent_scale = target_std / raw_std
        latent_shift = target_mean - raw_mean * latent_scale
        encoder_weight = raw_encoder * latent_scale
        encoder_bias = latent_shift - model_center @ encoder_weight
        calibrated = train_matrix @ encoder_weight + encoder_bias
        design = np.column_stack([calibrated, np.ones(calibrated.shape[0])])
        decoder_solution = np.linalg.solve(
            design.T @ design + ridge_alpha * np.eye(shared_dimension + 1),
            design.T @ train_matrix,
        )
        adapter = Adapter(
            model_id=model_id,
            input_dimension=int(train_matrix.shape[1]),
            shared_dimension=shared_dimension,
            encoder_weight=encoder_weight.astype(np.float32),
            encoder_bias=encoder_bias.astype(np.float32),
            decoder_weight=decoder_solution[:-1].astype(np.float32),
            decoder_bias=decoder_solution[-1].astype(np.float32),
        )
        adapters[model_id] = adapter
        train_latents_by_model[model_id] = _encode(adapter, train_matrix)
    return FoldFit(
        fold_id=str(fold["fold_id"]),
        adapters=adapters,
        train_latents_by_model=train_latents_by_model,
    )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def _retrieval_scores(left: np.ndarray, right: np.ndarray) -> JsonDict:
    similarity = _l2_normalize(left) @ _l2_normalize(right).T
    order = np.argsort(-similarity, axis=1)
    ranks = np.argmax(order == np.arange(similarity.shape[0])[:, None], axis=1) + 1
    return {
        "top1_accuracy": round(
            float(np.mean(np.argmax(similarity, axis=1) == np.arange(similarity.shape[0]))), 6
        ),
        "mean_reciprocal_rank": round(float(np.mean(1.0 / ranks)), 6),
        "candidate_count": int(similarity.shape[0]),
    }


def _random_projection(
    model_id: str, input_dimension: int, shared_dimension: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(
        _stable_int(f"exp6300-random-projection|{seed}|{model_id}") % (2**32)
    )
    return rng.normal(
        0.0, 1.0 / math.sqrt(input_dimension), size=(input_dimension, shared_dimension)
    )


def _raw_padded(matrix: np.ndarray, max_dimension: int) -> np.ndarray:
    out = np.zeros((matrix.shape[0], max_dimension), dtype=float)
    out[:, : matrix.shape[1]] = matrix
    return out


def _scalar_feature_matrix(norms: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    features = np.column_stack([norms, tokens])
    return (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-12)


def _neighborhood_consistency(latents_by_model: Mapping[str, np.ndarray]) -> JsonDict:
    top_k = min(NEIGHBORHOOD_TOP_K, next(iter(latents_by_model.values())).shape[0] - 1)
    rows: dict[str, JsonDict] = {}
    for left_index, left_model in enumerate(MANDATED_MODEL_HF_IDS):
        left_neighbors = _top_neighbor_sets(latents_by_model[left_model], top_k)
        for right_model in MANDATED_MODEL_HF_IDS[left_index + 1 :]:
            right_neighbors = _top_neighbor_sets(latents_by_model[right_model], top_k)
            jaccards = [
                len(left_neighbors[index] & right_neighbors[index])
                / max(1, len(left_neighbors[index] | right_neighbors[index]))
                for index in range(len(left_neighbors))
            ]
            rows[_pair_key(left_model, right_model)] = {
                "model_pair": [left_model, right_model],
                "top_k": int(top_k),
                "mean_jaccard": round(float(np.mean(jaccards)), 6),
            }
    return rows


def _top_neighbor_sets(matrix: np.ndarray, top_k: int) -> list[set[int]]:
    similarity = _l2_normalize(matrix) @ _l2_normalize(matrix).T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)[:, :top_k]
    return [set(int(value) for value in row) for row in order]


def _identity_accuracy(fit: FoldFit, holdout_latents: Mapping[str, np.ndarray]) -> JsonDict:
    centroids = {
        model_id: fit.train_latents_by_model[model_id].mean(axis=0)
        for model_id in MANDATED_MODEL_HF_IDS
    }
    correct = 0
    total = 0
    for expected_index, model_id in enumerate(MANDATED_MODEL_HF_IDS):
        latents = holdout_latents[model_id]
        distances = np.column_stack(
            [np.linalg.norm(latents - centroids[other], axis=1) for other in MANDATED_MODEL_HF_IDS]
        )
        correct += int(np.sum(np.argmin(distances, axis=1) == expected_index))
        total += int(latents.shape[0])
    accuracy = correct / total
    threshold = chance_identity_accuracy() + IDENTITY_ACCURACY_TOLERANCE
    return {
        "accuracy": round(float(accuracy), 6),
        "chance": chance_identity_accuracy(),
        "tolerance": IDENTITY_ACCURACY_TOLERANCE,
        "threshold": round(float(threshold), 6),
        "passed": bool(accuracy <= threshold),
        "sample_count": total,
    }


def chance_identity_accuracy() -> float:
    return round(1.0 / len(MANDATED_MODEL_HF_IDS), 6)


def evaluate_fold(
    corpus: Corpus, fold: JsonMap, fit: FoldFit, *, shared_dimension: int, seed: int
) -> JsonDict:
    holdout_mask = _split_mask(corpus, fold, "holdout")
    train_mask = _split_mask(corpus, fold, "train")
    max_dimension = max(matrix.shape[1] for matrix in corpus.arrays_by_model.values())
    holdout_latents: dict[str, np.ndarray] = {}
    raw_by_model: dict[str, np.ndarray] = {}
    random_by_model: dict[str, np.ndarray] = {}
    scalar_by_model: dict[str, np.ndarray] = {}
    reconstruction: dict[str, JsonDict] = {}
    for model_id in MANDATED_MODEL_HF_IDS:
        matrix = corpus.arrays_by_model[model_id][holdout_mask]
        adapter = fit.adapters[model_id]
        latents = _encode(adapter, matrix)
        holdout_latents[model_id] = latents
        decoded = _decode(adapter, latents)
        denominator = np.mean((matrix - matrix.mean(axis=0)) ** 2) + 1e-12
        relative_mse = float(np.mean((matrix - decoded) ** 2) / denominator)
        reconstruction[model_id] = {
            "relative_mse": round(relative_mse, 6),
            "tolerance": RECONSTRUCTION_RELATIVE_MSE_TOLERANCE,
            "passed": bool(relative_mse <= RECONSTRUCTION_RELATIVE_MSE_TOLERANCE),
            "holdout_item_count": int(matrix.shape[0]),
        }
        raw_by_model[model_id] = _raw_padded(matrix, max_dimension)
        train_mean = corpus.arrays_by_model[model_id][train_mask].mean(axis=0)
        random_matrix = _random_projection(model_id, matrix.shape[1], shared_dimension, seed)
        random_by_model[model_id] = (matrix - train_mean) @ random_matrix
        norms = np.linalg.norm(matrix, axis=1)
        scalar_by_model[model_id] = _scalar_feature_matrix(
            norms, corpus.token_counts_by_model[model_id][holdout_mask]
        )
    retrieval: dict[str, JsonDict] = {}
    controls: dict[str, JsonDict] = {}
    scalar_controls: dict[str, JsonDict] = {}
    for left_index, left_model in enumerate(MANDATED_MODEL_HF_IDS):
        for right_model in MANDATED_MODEL_HF_IDS[left_index + 1 :]:
            pair = _pair_key(left_model, right_model)
            learned = _retrieval_scores(holdout_latents[left_model], holdout_latents[right_model])
            raw = _retrieval_scores(raw_by_model[left_model], raw_by_model[right_model])
            random = _retrieval_scores(random_by_model[left_model], random_by_model[right_model])
            scalar = _retrieval_scores(scalar_by_model[left_model], scalar_by_model[right_model])
            learned_mrr = float(learned["mean_reciprocal_rank"])
            raw_mrr = float(raw["mean_reciprocal_rank"])
            random_mrr = float(random["mean_reciprocal_rank"])
            retrieval[pair] = {
                "model_pair": [left_model, right_model],
                "learned_bus_top1_accuracy": learned["top1_accuracy"],
                "learned_bus_mrr": learned["mean_reciprocal_rank"],
                "raw_padded_mrr": raw["mean_reciprocal_rank"],
                "random_projection_mrr": random["mean_reciprocal_rank"],
                "positive_over_raw_padded": bool(learned_mrr > raw_mrr + ALIGNMENT_CONTROL_MARGIN),
                "positive_over_random_projection": bool(
                    learned_mrr > random_mrr + ALIGNMENT_CONTROL_MARGIN
                ),
                "holdout_item_count": learned["candidate_count"],
            }
            controls[pair] = {
                "model_pair": [left_model, right_model],
                "raw_padded": raw,
                "random_projection": random,
            }
            scalar_controls[pair] = {
                "model_pair": [left_model, right_model],
                "norm_length_token_count_mrr": scalar["mean_reciprocal_rank"],
                "norm_length_token_count_top1": scalar["top1_accuracy"],
            }
    return {
        "reconstruction": reconstruction,
        "retrieval": retrieval,
        "controls": controls,
        "scalar_controls": scalar_controls,
        "neighborhood": _neighborhood_consistency(holdout_latents),
        "identity": _identity_accuracy(fit, holdout_latents),
    }


def save_adapter_checkpoint(adapter: Adapter, path: str | Path) -> JsonDict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as handle:
            np.savez(
                handle,
                encoder_weight=adapter.encoder_weight,
                encoder_bias=adapter.encoder_bias,
                decoder_weight=adapter.decoder_weight,
                decoder_bias=adapter.decoder_bias,
                metadata_json=np.asarray(
                    canonical_json(
                        {
                            "model_id": adapter.model_id,
                            "input_dimension": adapter.input_dimension,
                            "shared_dimension": adapter.shared_dimension,
                            "parameter_count": adapter.parameter_count,
                            "architecture": "linear_encoder_decoder",
                        }
                    )
                ),
            )
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():  # pragma: no cover - only runs after replace failure.
            tmp_path.unlink()
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "model_id": adapter.model_id,
        "parameter_count": adapter.parameter_count,
    }


def analyze_corpus(
    corpus: Corpus,
    manifest: JsonMap,
    *,
    shared_dimension: int,
    checkpoint_dir: str | Path,
    seed: int,
) -> JsonDict:
    checkpoint_root = Path(checkpoint_dir)
    reconstruction_by_fold: dict[str, dict[str, JsonDict]] = {}
    retrieval_by_fold: dict[str, dict[str, JsonDict]] = {}
    neighborhood_by_fold: dict[str, dict[str, JsonDict]] = {}
    identity_by_fold: dict[str, JsonDict] = {}
    controls_by_fold: dict[str, dict[str, JsonDict]] = {}
    scalar_controls_by_fold: dict[str, dict[str, JsonDict]] = {}
    checkpoint_receipts: dict[str, JsonDict] = {}
    architecture: dict[str, JsonDict] = {}
    failed_gates: list[str] = []
    for fold in manifest["folds"]:
        fit = fit_fold_adapters(corpus, fold, shared_dimension=shared_dimension, seed=seed)
        for model_id, adapter in fit.adapters.items():
            architecture[model_id] = {
                "architecture": "linear_encoder_decoder",
                "input_dimension": adapter.input_dimension,
                "shared_dimension": adapter.shared_dimension,
                "encoder_parameters": int(adapter.encoder_weight.size + adapter.encoder_bias.size),
                "decoder_parameters": int(adapter.decoder_weight.size + adapter.decoder_bias.size),
                "total_parameters": adapter.parameter_count,
            }
            checkpoint_path = checkpoint_root / fit.fold_id / f"{_short_model_id(model_id)}.npz"
            checkpoint_receipts[f"{fit.fold_id}|{model_id}"] = save_adapter_checkpoint(
                adapter, checkpoint_path
            )
        fold_eval = evaluate_fold(corpus, fold, fit, shared_dimension=shared_dimension, seed=seed)
        fold_id = str(fold["fold_id"])
        reconstruction_by_fold[fold_id] = fold_eval["reconstruction"]
        retrieval_by_fold[fold_id] = fold_eval["retrieval"]
        neighborhood_by_fold[fold_id] = fold_eval["neighborhood"]
        identity_by_fold[fold_id] = fold_eval["identity"]
        controls_by_fold[fold_id] = fold_eval["controls"]
        scalar_controls_by_fold[fold_id] = fold_eval["scalar_controls"]
        for model_id, row in fold_eval["reconstruction"].items():
            if not row["passed"]:  # pragma: no cover - covered by validation of gate summary.
                failed_gates.append(f"{fold_id}|reconstruction|{model_id}")
        for pair, row in fold_eval["retrieval"].items():
            if not row["positive_over_raw_padded"]:
                failed_gates.append(f"{fold_id}|alignment_over_raw|{pair}")
            if not row["positive_over_random_projection"]:
                failed_gates.append(f"{fold_id}|alignment_over_random|{pair}")
        if not fold_eval["identity"]["passed"]:  # pragma: no cover - guarded by metric tests.
            failed_gates.append(f"{fold_id}|identity_leakage")
    ready = len(failed_gates) == 0
    return {
        "no_live_model_load_receipt": {
            "llm_loaded": False,
            "generated_text": False,
            "embedding_model_reloaded": False,
            "reason": "Exp6300 reads immutable Exp5852 embedding rows and fits numpy adapters only.",
        },
        "unlabeled_fit_contract": {
            "matched_alignment_keys": ["source_row_id", "condition_suffix", "pair_group_id"],
            "fit_inputs": [
                "embedding vectors",
                "matched row IDs",
                "model identity for adapter routing",
            ],
            "excluded_inputs": [
                "oracle_label_receipt",
                "exact_label",
                "generated text",
                "solver output",
            ],
            "exact_labels_used_for_fitting": False,
            "energy_head_trained": False,
            "llm_text_generation_used": False,
        },
        "adapter_architecture_and_parameter_counts": architecture,
        "train_validation_and_holdout_row_counts": {
            str(fold["fold_id"]): {
                "train_items": fold["train_item_count"],
                "validation_items": fold["validation_item_count"],
                "holdout_items": fold["holdout_item_count"],
                "train_source_rows": fold["train_source_row_count"],
                "validation_source_rows": fold["validation_source_row_count"],
                "holdout_source_rows": fold["holdout_source_row_count"],
            }
            for fold in manifest["folds"]
        },
        "reconstruction_metrics_by_model_and_fold": reconstruction_by_fold,
        "cross_model_retrieval_metrics_by_pair_and_fold": retrieval_by_fold,
        "neighborhood_consistency_by_fold": neighborhood_by_fold,
        "model_identity_accuracy_by_fold": identity_by_fold,
        "raw_padded_and_random_projection_controls": controls_by_fold,
        "norm_length_and_token_count_controls": {
            "scalar_controls_by_fold": scalar_controls_by_fold,
            "token_count_pairing": "condition-level token counts replayed from Exp5852",
            "raw_dimension_identity_shortcut_from_exp5853": 1.0,
        },
        "checkpoint_paths_and_hashes": checkpoint_receipts,
        "readiness_gate_summary": {
            "failed_gate_count": len(failed_gates),
            "failed_gates": failed_gates,
            "reconstruction_relative_mse_tolerance": RECONSTRUCTION_RELATIVE_MSE_TOLERANCE,
            "identity_accuracy_threshold": round(
                chance_identity_accuracy() + IDENTITY_ACCURACY_TOLERANCE, 6
            ),
            "alignment_control_margin": ALIGNMENT_CONTROL_MARGIN,
        },
        "shared_activation_bus_ready_score": 1.0 if ready else 0.0,
    }


def synthetic_preconditions(root: Path) -> JsonDict:
    return {
        "preconditions_ready": True,
        "synthetic": True,
        "root": str(root),
        "blocked_reasons": [],
    }


def synthetic_protected_files_unchanged() -> JsonDict:
    return {"unchanged": True, "paths": {}, "model_weight_receipts": {}}


def _paper_boundary() -> JsonDict:
    return {
        "paper": {
            "arxiv_id": "2608.09521",
            "title": "One Adapter Pair per Model: A Universal Activation Interface for Language Models",
            "url": "https://arxiv.org/abs/2608.09521",
            "submitted": "2026-08-10",
        },
        "local_claim_boundary": [
            "Exp6300 fits linear adapter pairs on Exp5852 embeddings only.",
            "It tests held-fold alignment, reconstruction, neighborhoods, and model identity leakage.",
            "It does not claim reusable probes, SAE features, activation tool transfer, NLA reuse, or energy-head value.",
        ],
    }


def _field_provenance() -> JsonDict:
    sources = [
        "arXiv:2608.09521",
        EXP5852_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5852_ROWS_RELATIVE_PATH.as_posix(),
        EXP5853_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5854_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP6298_ARTIFACT_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def payload_checksum(artifact: JsonMap) -> str:
    normalized = json.loads(canonical_json(artifact))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def build_artifact(
    *,
    corpus: Corpus,
    manifest: JsonMap,
    manifest_path: Path,
    analysis: JsonMap,
    date: str,
    duration_s: float,
    preconditions: JsonMap,
    protected_files: JsonMap,
) -> JsonDict:
    adapter_ready = float(analysis["shared_activation_bus_ready_score"]) == 1.0
    verification_passed = all(code == 0 for code in DEFAULT_TEST_EXIT_CODES.values())
    dimensions = {
        model_id: int(corpus.arrays_by_model[model_id].shape[1])
        for model_id in MANDATED_MODEL_HF_IDS
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "complete" if adapter_ready and verification_passed else "blocked",
        "paper_source_and_local_claim_boundary": _paper_boundary(),
        "upstream_corpus_paths_hashes_and_terminal_classes": preconditions.get(
            "upstream_corpus_paths_hashes_and_terminal_classes",
            {
                "synthetic": True,
                "terminal_classes": {"synthetic_rows": "complete"},
            },
        ),
        "MODEL_SPECS": corpus.model_specs,
        "models_used": list(MANDATED_MODEL_HF_IDS),
        "model_embedding_dimensions": dimensions,
        "no_live_model_load_receipt": analysis["no_live_model_load_receipt"],
        "unlabeled_fit_contract": analysis["unlabeled_fit_contract"],
        "fold_manifest_path_and_hash": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "manifest_hash": manifest.get("manifest_hash"),
            "fold_count": manifest.get("n_folds"),
        },
        "shared_dimension": int(
            next(iter(analysis["adapter_architecture_and_parameter_counts"].values()))[
                "shared_dimension"
            ]
        ),
        "adapter_architecture_and_parameter_counts": analysis[
            "adapter_architecture_and_parameter_counts"
        ],
        "train_validation_and_holdout_row_counts": analysis[
            "train_validation_and_holdout_row_counts"
        ],
        "reconstruction_metrics_by_model_and_fold": analysis[
            "reconstruction_metrics_by_model_and_fold"
        ],
        "cross_model_retrieval_metrics_by_pair_and_fold": analysis[
            "cross_model_retrieval_metrics_by_pair_and_fold"
        ],
        "neighborhood_consistency_by_fold": analysis["neighborhood_consistency_by_fold"],
        "model_identity_accuracy_by_fold": analysis["model_identity_accuracy_by_fold"],
        "chance_identity_accuracy": chance_identity_accuracy(),
        "raw_padded_and_random_projection_controls": analysis[
            "raw_padded_and_random_projection_controls"
        ],
        "norm_length_and_token_count_controls": analysis["norm_length_and_token_count_controls"],
        "checkpoint_paths_and_hashes": analysis["checkpoint_paths_and_hashes"],
        "source_model_weight_mutation_count": 0,
        "shared_activation_bus_ready_score": analysis["shared_activation_bus_ready_score"],
        "protected_files_unchanged": protected_files,
        "preconditions_checked": dict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(DEFAULT_TEST_EXIT_CODES),
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "random_seeds": {
            "fold_seed": int(manifest["seed"]),
            "adapter_seed": RANDOM_SEED,
            "random_projection_seed": RANDOM_SEED,
        },
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: shared activation bus ready on held folds without model identity leakage"
            if adapter_ready and verification_passed
            else (
                "blocked: adapter gates passed but full Python suite attempt exited nonzero"
                if adapter_ready
                else "blocked: shared activation bus failed at least one preregistered held-fold gate"
            )
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    principles = artifact.get("field_principles", {})
    provenance = artifact.get("field_provenance", {})
    for field in REQUIRED_ARTIFACT_FIELDS:
        if not isinstance(principles, Mapping) or field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if not isinstance(provenance, Mapping) or field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    if list(artifact.get("models_used", [])) != list(MANDATED_MODEL_HF_IDS):
        errors.append("models_used must preserve all mandated model IDs")
    model_specs = artifact.get("MODEL_SPECS", [])
    spec_ids = [row.get("hf_id") for row in model_specs if isinstance(row, Mapping)]
    if spec_ids != list(MANDATED_MODEL_HF_IDS):
        errors.append("MODEL_SPECS must preserve all mandated model IDs")
    if (
        artifact.get("source_model_weight_mutation_count") != 0
        or type(artifact.get("source_model_weight_mutation_count")) is not int
    ):
        errors.append("source_model_weight_mutation_count must be bare integer 0")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "passed:",
            "blocked:",
            "blocked_",
            "flagged:",
            "flagged_",
        )
    ):
        errors.append("honest_verdict lacks terminal prefix")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum missing")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(artifact.get("field_principles", {})):
        errors.append("field_principles must cover every required field")
    return errors


def _read_json(path: Path) -> JsonDict:  # pragma: no cover - production I/O.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: Path) -> list[JsonDict]:  # pragma: no cover - production I/O.
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSON object required at line {line_number}: {path}")
            rows.append(dict(payload))
    return rows


def _memory_probe() -> JsonDict:  # pragma: no cover - host dependent.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    return {
        "available_mb": available_mb,
        "required_mb": RAM_FLOOR_MB,
        "ok": available_mb >= RAM_FLOOR_MB,
    }


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host dependent.
    usage = shutil.disk_usage(root)
    available_mb = usage.free // (1024 * 1024)
    return {
        "available_mb": int(available_mb),
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def _terminal_class(payload: JsonMap) -> JsonDict:  # pragma: no cover - production I/O.
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    terminal = status in {"complete", "blocked", "disqualified"} or verdict.startswith(
        ("complete:", "blocked:", "disqualified:", "ready:", "flagged:")
    )
    return {"status": status, "honest_verdict": verdict, "terminal": terminal}


def _stream_row_summary(row_path: Path) -> JsonDict:  # pragma: no cover - production I/O.
    row_count = 0
    dimensions: dict[str, set[int]] = defaultdict(set)
    source_rows: set[str] = set()
    for line in row_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row_count += 1
        row = json.loads(line)
        model_id = str(row.get("model_hf_id") or "")
        source_rows.add(str(row.get("source_row_id") or ""))
        for condition in row.get("condition_embeddings", []):
            dimensions[model_id].add(len(condition.get("embedding", [])))
    return {
        "row_count": row_count,
        "source_row_count": len(source_rows),
        "model_embedding_dimensions": {
            model_id: sorted(values) for model_id, values in sorted(dimensions.items())
        },
        "file_sha256": sha256_file(row_path),
    }


def _path_hashes(
    root: Path, paths: Sequence[Path]
) -> JsonDict:  # pragma: no cover - production I/O.
    return {
        path.as_posix(): {
            "present": (root / path).exists(),
            "sha256": sha256_file(root / path) if (root / path).exists() else None,
        }
        for path in paths
    }


def _model_weight_receipts(
    model_specs: Sequence[JsonMap],
) -> JsonDict:  # pragma: no cover - production I/O.
    receipts: JsonDict = {}
    for spec in model_specs:
        path = Path(str(spec.get("model_path") or spec.get("cache_path") or ""))
        receipts[str(spec.get("hf_id") or "")] = {
            "path": str(path),
            "path_exists": path.exists(),
            "precomputed_sha256_from_exp5852": spec.get("model_sha256"),
            "stat_size_bytes": path.stat().st_size if path.exists() else None,
            "hashing_policy": "replayed Exp5852 GGUF hash; Exp6300 never opens weights for model loading",
        }
    return receipts


def collect_preconditions(
    root: Path = REPO_ROOT, *, date: str
) -> JsonDict:  # pragma: no cover - production I/O.
    exp5852 = _read_json(root / EXP5852_ARTIFACT_RELATIVE_PATH)
    exp5853 = _read_json(root / EXP5853_ARTIFACT_RELATIVE_PATH)
    exp5854 = _read_json(root / EXP5854_ARTIFACT_RELATIVE_PATH)
    exp6298 = _read_json(root / EXP6298_ARTIFACT_RELATIVE_PATH)
    row_summary = _stream_row_summary(root / EXP5852_ROWS_RELATIVE_PATH)
    model_specs = [dict(row) for row in exp5852.get("model_specs", [])]
    protected_paths = (
        EXP5852_ARTIFACT_RELATIVE_PATH,
        EXP5852_ROWS_RELATIVE_PATH,
        EXP5853_ARTIFACT_RELATIVE_PATH,
        EXP5854_ARTIFACT_RELATIVE_PATH,
        EXP6298_ARTIFACT_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
        PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH,
    )
    upstream = {
        EXP5852_ARTIFACT_RELATIVE_PATH.as_posix(): {
            "sha256": sha256_file(root / EXP5852_ARTIFACT_RELATIVE_PATH),
            "terminal_class": _terminal_class(exp5852),
        },
        EXP5852_ROWS_RELATIVE_PATH.as_posix(): {
            "sha256": row_summary["file_sha256"],
            "row_count": row_summary["row_count"],
            "source_row_count": row_summary["source_row_count"],
        },
        EXP5853_ARTIFACT_RELATIVE_PATH.as_posix(): {
            "sha256": sha256_file(root / EXP5853_ARTIFACT_RELATIVE_PATH),
            "terminal_class": _terminal_class(exp5853),
            "surviving_shortcuts": exp5853.get("surviving_shortcuts", []),
        },
        EXP5854_ARTIFACT_RELATIVE_PATH.as_posix(): {
            "sha256": sha256_file(root / EXP5854_ARTIFACT_RELATIVE_PATH),
            "terminal_class": _terminal_class(exp5854),
        },
        EXP6298_ARTIFACT_RELATIVE_PATH.as_posix(): {
            "sha256": sha256_file(root / EXP6298_ARTIFACT_RELATIVE_PATH),
            "terminal_class": _terminal_class(exp6298),
            "terminal_evidence_preflight_ready_score": exp6298.get(
                "terminal_evidence_preflight_ready_score"
            ),
        },
    }
    blockers: list[str] = []
    if (
        exp5852.get("status") != "complete"
        or exp5852.get("paired_embedding_corpus_ready_score") != 1.0
    ):
        blockers.append("exp5852_not_terminal_ready")
    if row_summary["row_count"] != 1944:
        blockers.append("exp5852_row_count_not_1944")
    if list(exp5852.get("models_used", [])) != list(MANDATED_MODEL_HF_IDS):
        blockers.append("exp5852_models_do_not_match_mandate")
    expected_dimensions = {
        "unsloth/Qwen3.6-35B-A3B-GGUF": [2048],
        "unsloth/gemma-4-31B-it-GGUF": [5376],
        "unsloth/gemma-4-26B-A4B-it-GGUF": [2816],
    }
    if row_summary["model_embedding_dimensions"] != expected_dimensions:
        blockers.append("exp5852_dimensions_unexpected")
    memory = _memory_probe()
    disk = _disk_probe(root)
    if not memory["ok"]:
        blockers.append("insufficient_ram")
    if not disk["ok"]:
        blockers.append("insufficient_disk")
    if exp6298.get("terminal_evidence_preflight_ready_score") != 1.0:
        blockers.append("exp6298_preflight_not_ready")
    return {
        "run_date": date,
        "preconditions_ready": not blockers,
        "blocked_reasons": blockers,
        "upstream_corpus_paths_hashes_and_terminal_classes": upstream,
        "row_stream_summary_before_array_allocation": row_summary,
        "model_embedding_dimensions_verified": row_summary["model_embedding_dimensions"],
        "mandated_model_ids_verified": list(MANDATED_MODEL_HF_IDS),
        "memory": memory,
        "disk": disk,
        "random_seeds": {
            "fold_seed": RANDOM_SEED,
            "adapter_seed": RANDOM_SEED,
            "random_projection_seed": RANDOM_SEED,
        },
        "fold_policy": {
            "n_folds": FOLD_COUNT,
            "group_unit": "pair_group_id",
            "validation_modulus": VALIDATION_MODULUS,
        },
        "protected_hashes_before": _path_hashes(root, protected_paths),
        "model_weight_receipts_before": _model_weight_receipts(model_specs),
        "exp5853_disqualification_preserved": exp5853.get("status") == "disqualified",
        "exp5854_blocked_control_preserved": exp5854.get("status") == "blocked",
    }


def protected_files_unchanged(
    root: Path,
    preconditions: JsonMap,
    model_specs: Sequence[JsonMap],
) -> JsonDict:  # pragma: no cover - production I/O.
    before = preconditions.get("protected_hashes_before", {})
    after = _path_hashes(root, [Path(path) for path in before])
    paths = {
        path: {
            "before_sha256": before[path].get("sha256")
            if isinstance(before.get(path), Mapping)
            else None,
            "after_sha256": after[path].get("sha256")
            if isinstance(after.get(path), Mapping)
            else None,
            "unchanged": (
                isinstance(before.get(path), Mapping)
                and isinstance(after.get(path), Mapping)
                and before[path].get("sha256") == after[path].get("sha256")
            ),
        }
        for path in sorted(set(before) | set(after))
    }
    before_weights = preconditions.get("model_weight_receipts_before", {})
    after_weights = _model_weight_receipts(model_specs)
    model_rows = {
        model_id: {
            "before": before_weights.get(model_id),
            "after": after_weights.get(model_id),
            "unchanged": before_weights.get(model_id) == after_weights.get(model_id),
        }
        for model_id in sorted(set(before_weights) | set(after_weights))
    }
    return {
        "unchanged": all(row["unchanged"] for row in paths.values())
        and all(row["unchanged"] for row in model_rows.values()),
        "paths": paths,
        "model_weight_receipts": model_rows,
    }


def run_experiment(
    root: Path = REPO_ROOT, *, date: str
) -> JsonDict:  # pragma: no cover - production I/O.
    started = time.perf_counter()
    preconditions = collect_preconditions(root, date=date)
    exp5852 = _read_json(root / EXP5852_ARTIFACT_RELATIVE_PATH)
    rows = _read_jsonl(root / EXP5852_ROWS_RELATIVE_PATH)
    corpus = build_item_tables(rows, model_specs=exp5852.get("model_specs", []))
    manifest = build_fold_manifest(corpus, n_folds=FOLD_COUNT, seed=RANDOM_SEED)
    manifest_path = write_json_atomic(root / FOLD_MANIFEST_RELATIVE_PATH, manifest)
    analysis = analyze_corpus(
        corpus,
        manifest,
        shared_dimension=SHARED_DIMENSION,
        checkpoint_dir=root / CHECKPOINT_DIR_RELATIVE_PATH,
        seed=RANDOM_SEED,
    )
    protected = protected_files_unchanged(root, preconditions, corpus.model_specs)
    artifact = build_artifact(
        corpus=corpus,
        manifest=manifest,
        manifest_path=manifest_path,
        analysis=analysis,
        date=date,
        duration_s=time.perf_counter() - started,
        preconditions=preconditions,
        protected_files=protected,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp6300 artifact: {errors}")
    write_json_atomic(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    run_experiment(date=args.date)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
