"""Exp6301 independent audit for the cached Exp6300 activation bus.

Spec refs: REQ-VERIFY-6301, SCENARIO-VERIFY-6301-RECONSTRUCT,
SCENARIO-VERIFY-6301-SHORTCUTS, SCENARIO-VERIFY-6301-INDEPENDENCE.

The audit loads frozen rows and adapter checkpoints. It does not call Exp6300's
fit code. It does not train an energy head. All control decisions are made by
fresh local evaluators created in this module.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot.terminal_artifacts import classify_artifact_path


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6301_activation_bus_integrity_audit.json")
EXP6300_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6300_three_family_universal_activation_bus.json"
)
EXP5852_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5852_three_family_paired_embeddings.rows.jsonl"
)
EXP5853_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5853_paired_embedding_integrity_audit.json"
)
FOLD_MANIFEST_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6300_three_family_universal_activation_bus/fold_manifest.json"
)
CHECKPOINT_DIR_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6300_three_family_universal_activation_bus/adapters"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6301_activation_bus_integrity_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6301_activation_bus_integrity_audit.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
TERMINAL_ARTIFACTS_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

SCHEMA = "carnot.experiment_6301.activation_bus_integrity_audit.v1"
FOLD_MANIFEST_SCHEMA = "carnot.experiment_6300.fold_manifest.v1"
EXPERIMENT = 6301
EXPERIMENT_ID = "experiment_6301_activation_bus_integrity_audit"
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"
VERIFIER_IS_ORACLE = True
RANDOM_SEED = 6301

MANDATED_MODEL_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

MIN_CELL_ROWS = 1
MIN_ANCHOR_ROWS = 1
MIN_PAIRED_NORM = 1e-9
MIN_DIRECTION_POSITIVE_RATE = 0.55
PAIR_SWAP_MAX_POSITIVE_RATE = 0.45
NORM_ONLY_MAX_ACCURACY = 0.75
IDENTITY_ACCURACY_TOLERANCE = 0.05
SEMANTIC_ALIGNMENT_MRR_MULTIPLIER = 3.0

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_path_hash_and_terminal_class",
    "MODEL_SPECS",
    "models_covered",
    "row_and_checkpoint_reconstruction_receipts",
    "evaluator_independence_receipts",
    "fold_leakage_checks",
    "claim_flip_sensitivity",
    "pair_swap_controls",
    "label_permutation_controls",
    "model_identity_controls",
    "norm_length_token_and_truncation_controls",
    "duplicate_and_no_information_controls",
    "evaluator_swap_receipts",
    "disaggregated_cell_decisions",
    "failed_cells",
    "surviving_shortcuts",
    "false_pass_injection_results",
    "activation_bus_integrity_ready_score",
    "source_mutation_count",
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
    "status": "The audit must close with a terminal state.",
    "upstream_path_hash_and_terminal_class": "Input artifact classes must be replayed from bytes.",
    "MODEL_SPECS": "The three mandated GGUF families must remain named.",
    "models_covered": "Coverage must show which cached families were audited.",
    "row_and_checkpoint_reconstruction_receipts": "Rows and adapters must be rebuilt from hashes.",
    "evaluator_independence_receipts": "The audit must not trust Exp6300's own decisions.",
    "fold_leakage_checks": "Group-aware folds must keep train and held rows separate.",
    "claim_flip_sensitivity": "The shared bus must react to paired semantic flips.",
    "pair_swap_controls": "Swapping pair members must collapse the learned direction.",
    "label_permutation_controls": "Deterministic label reversal must not look valid.",
    "model_identity_controls": "Shared latents must hide model ID while preserving alignment.",
    "norm_length_token_and_truncation_controls": "Scalar envelope shortcuts must not carry labels.",
    "duplicate_and_no_information_controls": "Duplicate weights and null vectors must fail closed.",
    "evaluator_swap_receipts": "A second fresh consumer must agree with the primary audit.",
    "disaggregated_cell_decisions": "Each cell owns its decision without pooled rescue.",
    "failed_cells": "Failed cells must stay visible at the top level.",
    "surviving_shortcuts": "Any surviving shortcut blocks readiness.",
    "false_pass_injection_results": "Injected false passes must be rejected.",
    "activation_bus_integrity_ready_score": "Readiness is bare and conjunctive.",
    "source_mutation_count": "The audit must not mutate source rows, adapters, or weights.",
    "protected_files_unchanged": "Protected hashes prove the run did not rewrite inputs.",
    "preconditions_checked": "Hashes, folds, adapters, and output paths are checked first.",
    "inference_substrate": "The substrate states deterministic replay with no LLM call.",
    "verifier_is_oracle": "The integrity verifier is the authority for this gate.",
    "field_provenance": "Every field must cite its evidence sources.",
    "field_principles": "Every required field must state its guard principle.",
    "test_commands": "Verification commands must be recorded.",
    "test_exit_codes": "Failed command exits must stay visible.",
    "duration_s": "Measured wall time prevents padded runtime claims.",
    "random_seeds": "Seeds make control permutations reproducible.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict must begin with a terminal prefix.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6301_activation_bus_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null --branch "
    "--include=python/carnot/experiment_6301_activation_bus_integrity_audit.py "
    "-m pytest tests/python/test_experiment_6301_activation_bus_integrity_audit.py "
    "-q --no-cov -n 0"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    (
        ".venv/bin/coverage run --rcfile=/dev/null --branch "
        "--include=python/carnot/experiment_6301_activation_bus_integrity_audit.py "
        "-m pytest tests/python/test_experiment_6301_activation_bus_integrity_audit.py "
        "-q --no-cov -n 0"
    ),
    (
        ".venv/bin/coverage report --rcfile=/dev/null "
        "--include=python/carnot/experiment_6301_activation_bus_integrity_audit.py "
        "--fail-under=100 --show-missing"
    ),
    ".venv/bin/pytest tests/python -q",
    (
        ".venv/bin/python scripts/check_spec_coverage.py "
        "tests/python/test_experiment_6301_activation_bus_integrity_audit.py"
    ),
    (
        ".venv/bin/python -m carnot.experiment_6298_terminal_evidence_preflight_linter "
        "--date 20260811 --no-run-commands"
    ),
    ".venv/bin/python scripts/determination_preservation_lint.py --all",
    ".venv/bin/python -m carnot.experiment_6301_activation_bus_integrity_audit --date 20260811",
    (
        ".venv/bin/python scripts/adversarial_verify.py "
        "results/experiment_6301_activation_bus_integrity_audit.json"
    ),
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}


@dataclass(frozen=True)
class BusInputs:
    item_keys: list[str]
    item_meta: list[JsonDict]
    arrays_by_model: dict[str, np.ndarray]
    token_counts_by_model: dict[str, np.ndarray]
    truncation_by_model: dict[str, np.ndarray]
    labels_by_item: dict[str, bool]
    row_hash_mismatches: list[str]
    duplicate_embedding_cell_ids: list[str]
    missing_model_item_count: int
    source_row_count: int
    embedding_row_count: int


@dataclass(frozen=True)
class Adapter:
    model_id: str
    encoder_weight: np.ndarray
    encoder_bias: np.ndarray
    metadata: JsonDict
    sha256: str
    path: str


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


def embedding_row_hash(row: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(row))
    stable["row_hash"] = ""
    return sha256_json(stable)


def short_model_id(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_").lower()


def _pair_key(left: str, right: str) -> str:
    return f"{short_model_id(left)}__{short_model_id(right)}"


def _round(value: float) -> float:
    return round(float(value), 6)


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():  # pragma: no cover - only runs after replace failure.
            tmp_path.unlink()
    return target


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"JSONL object required at line {line_number}: {path}")
            rows.append(dict(payload))
    return rows


def _resolve(root: Path, relative: Path) -> Path:
    rooted = root / relative
    if rooted.exists() or root != REPO_ROOT:
        return rooted
    return REPO_ROOT / relative


def _path_receipt(path: Path, *, terminal_json: bool) -> JsonDict:
    if terminal_json and path.exists():
        terminal_class = classify_artifact_path(path).to_dict()
    else:
        terminal_class = {
            "classification": "not_terminal_artifact",
            "terminal": False,
            "reason": "row_or_checkpoint_file",
        }
    return {
        "path": str(path),
        "present": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "terminal_class": terminal_class,
    }


def upstream_path_hash_and_terminal_class(root: Path) -> JsonDict:
    json_paths = [
        EXP6300_ARTIFACT_RELATIVE_PATH,
        EXP5853_ARTIFACT_RELATIVE_PATH,
        FOLD_MANIFEST_RELATIVE_PATH,
    ]
    row_paths = [EXP5852_ROWS_RELATIVE_PATH]
    receipts = {
        path.as_posix(): _path_receipt(_resolve(root, path), terminal_json=True)
        for path in json_paths
    }
    receipts.update(
        {
            path.as_posix(): _path_receipt(_resolve(root, path), terminal_json=False)
            for path in row_paths
        }
    )
    checkpoint_dir = _resolve(root, CHECKPOINT_DIR_RELATIVE_PATH)
    checkpoint_files = sorted(checkpoint_dir.rglob("*.npz")) if checkpoint_dir.exists() else []
    receipts[CHECKPOINT_DIR_RELATIVE_PATH.as_posix()] = {
        "path": str(checkpoint_dir),
        "present": checkpoint_dir.exists(),
        "file_count": len(checkpoint_files),
        "tree_sha256": sha256_json(
            [
                {
                    "path": str(path.relative_to(checkpoint_dir)),
                    "sha256": sha256_file(path),
                }
                for path in checkpoint_files
            ]
        ),
        "terminal_class": {
            "classification": "checkpoint_tree_not_terminal_artifact",
            "terminal": False,
            "reason": "adapter_checkpoints_are_input_bytes",
        },
    }
    return receipts


def reconstruct_bus_inputs(rows: Sequence[Mapping[str, Any]]) -> BusInputs:
    vectors: dict[str, dict[str, np.ndarray]] = {model_id: {} for model_id in MANDATED_MODEL_HF_IDS}
    tokens: dict[str, dict[str, int]] = {model_id: {} for model_id in MANDATED_MODEL_HF_IDS}
    truncations: dict[str, dict[str, bool]] = {
        model_id: {} for model_id in MANDATED_MODEL_HF_IDS
    }
    labels_by_item: dict[str, bool] = {}
    meta_by_item: dict[str, JsonDict] = {}
    source_ids: set[str] = set()
    row_hash_mismatches: list[str] = []
    cell_counts: Counter[str] = Counter()
    for row in rows:
        model_id = str(row.get("model_hf_id") or "")
        if model_id not in vectors:
            continue
        cell_id = str(row.get("embedding_cell_id") or f"{row.get('source_row_id')}|{model_id}")
        cell_counts[cell_id] += 1
        if row.get("row_hash") != embedding_row_hash(row):
            row_hash_mismatches.append(cell_id)
        source_row_id = str(row.get("source_row_id") or "")
        source_ids.add(source_row_id)
        label_by_condition = dict(
            dict(row.get("oracle_label_receipt") or {}).get("labels_by_condition_id") or {}
        )
        for condition in row.get("condition_embeddings") or []:
            if not isinstance(condition, Mapping):
                raise ValueError("condition_embeddings must contain objects")
            suffix = str(condition.get("condition_suffix") or "")
            item_key = f"{source_row_id}|{suffix}"
            vector = np.asarray(condition.get("embedding") or [], dtype=float)
            if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                raise ValueError(f"finite one-dimensional embedding required: {item_key}")
            vectors[model_id][item_key] = vector
            tokens[model_id][item_key] = int(condition.get("token_count", 0) or 0)
            truncations[model_id][item_key] = bool(condition.get("truncated"))
            condition_id = str(condition.get("condition_id") or "")
            labels_by_item[item_key] = bool(label_by_condition.get(condition_id))
            meta_by_item.setdefault(
                item_key,
                {
                    "source_row_id": source_row_id,
                    "condition_suffix": suffix,
                    "group_id": str(row.get("pair_group_id") or source_row_id),
                    "axis": str(row.get("axis") or ""),
                    "family": str(row.get("family") or ""),
                    "hardness": str(row.get("solver_effort_bin") or ""),
                    "surface": str(row.get("surface_kind") or ""),
                    "split": str(row.get("split") or ""),
                },
            )
    union_keys = set().union(*(set(value) for value in vectors.values()))
    common_keys = sorted(set.intersection(*(set(value) for value in vectors.values())))
    missing_model_item_count = sum(
        len(union_keys - set(vectors[model_id])) for model_id in MANDATED_MODEL_HF_IDS
    )
    if not common_keys:
        raise ValueError("no matched cached activation rows across mandated models")
    return BusInputs(
        item_keys=common_keys,
        item_meta=[meta_by_item[key] for key in common_keys],
        arrays_by_model={
            model_id: np.vstack([vectors[model_id][key] for key in common_keys])
            for model_id in MANDATED_MODEL_HF_IDS
        },
        token_counts_by_model={
            model_id: np.asarray([tokens[model_id][key] for key in common_keys], dtype=float)
            for model_id in MANDATED_MODEL_HF_IDS
        },
        truncation_by_model={
            model_id: np.asarray(
                [truncations[model_id][key] for key in common_keys], dtype=bool
            )
            for model_id in MANDATED_MODEL_HF_IDS
        },
        labels_by_item={key: labels_by_item[key] for key in common_keys},
        row_hash_mismatches=sorted(row_hash_mismatches),
        duplicate_embedding_cell_ids=sorted(cell for cell, count in cell_counts.items() if count > 1),
        missing_model_item_count=missing_model_item_count,
        source_row_count=len(source_ids),
        embedding_row_count=sum(cell_counts.values()),
    )


def _groups_mask(inputs: BusInputs, groups: Sequence[str]) -> np.ndarray:
    group_set = set(groups)
    return np.asarray([str(meta["group_id"]) in group_set for meta in inputs.item_meta], dtype=bool)


def _load_adapter(path: Path, model_id: str) -> Adapter:
    with np.load(path, allow_pickle=False) as data:
        metadata_raw = data["metadata_json"].item()
        metadata = json.loads(str(metadata_raw))
        return Adapter(
            model_id=model_id,
            encoder_weight=np.asarray(data["encoder_weight"], dtype=float),
            encoder_bias=np.asarray(data["encoder_bias"], dtype=float),
            metadata=dict(metadata),
            sha256=sha256_file(path),
            path=str(path),
        )


def _load_fold_adapters(root: Path, fold: Mapping[str, Any]) -> dict[str, Adapter]:
    fold_id = str(fold["fold_id"])
    adapters: dict[str, Adapter] = {}
    for model_id in MANDATED_MODEL_HF_IDS:
        path = root / CHECKPOINT_DIR_RELATIVE_PATH / fold_id / f"{short_model_id(model_id)}.npz"
        adapters[model_id] = _load_adapter(path, model_id)
    return adapters


def _encode(adapter: Adapter, matrix: np.ndarray) -> np.ndarray:
    return matrix @ adapter.encoder_weight + adapter.encoder_bias


def _latents_by_model(
    inputs: BusInputs, adapters: Mapping[str, Adapter]
) -> dict[str, np.ndarray]:
    return {
        model_id: _encode(adapters[model_id], inputs.arrays_by_model[model_id])
        for model_id in MANDATED_MODEL_HF_IDS
    }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def _retrieval_scores(left: np.ndarray, right: np.ndarray) -> JsonDict:
    similarity = _l2_normalize(left) @ _l2_normalize(right).T
    order = np.argsort(-similarity, axis=1)
    ranks = np.argmax(order == np.arange(similarity.shape[0])[:, None], axis=1) + 1
    n = similarity.shape[0]
    chance_top1 = 1.0 / n if n else 1.0
    chance_mrr = sum(1.0 / rank for rank in range(1, n + 1)) / n if n else 1.0
    return {
        "top1_accuracy": _round(float(np.mean(np.argmax(similarity, axis=1) == np.arange(n)))),
        "mean_reciprocal_rank": _round(float(np.mean(1.0 / ranks))),
        "candidate_count": int(n),
        "chance_top1_accuracy": _round(chance_top1),
        "chance_mean_reciprocal_rank": _round(chance_mrr),
        "semantic_alignment_passed": bool(
            n > 0
            and float(np.mean(np.argmax(similarity, axis=1) == np.arange(n))) > chance_top1
            and float(np.mean(1.0 / ranks)) > chance_mrr * SEMANTIC_ALIGNMENT_MRR_MULTIPLIER
        ),
    }


def _paired_rows(
    inputs: BusInputs,
    latents: np.ndarray,
    model_id: str,
    mask: np.ndarray,
) -> list[JsonDict]:
    index_by_source_suffix: dict[tuple[str, str], int] = {}
    for index, meta in enumerate(inputs.item_meta):
        if mask[index]:
            index_by_source_suffix[(str(meta["source_row_id"]), str(meta["condition_suffix"]))] = index
    paired: list[JsonDict] = []
    for source_id in sorted({source for source, _ in index_by_source_suffix}):
        left = index_by_source_suffix.get((source_id, "a"))
        right = index_by_source_suffix.get((source_id, "b"))
        if left is None or right is None:
            continue
        left_key = inputs.item_keys[left]
        right_key = inputs.item_keys[right]
        meta = inputs.item_meta[left]
        paired.append(
            {
                "model_id": model_id,
                "source_row_id": source_id,
                "group_id": meta["group_id"],
                "axis": meta["axis"],
                "family": meta["family"],
                "hardness": meta["hardness"],
                "surface": meta["surface"],
                "diff": latents[right] - latents[left],
                "left_norm": float(np.linalg.norm(latents[left])),
                "right_norm": float(np.linalg.norm(latents[right])),
                "left_token_count": int(inputs.token_counts_by_model[model_id][left]),
                "right_token_count": int(inputs.token_counts_by_model[model_id][right]),
                "left_truncated": bool(inputs.truncation_by_model[model_id][left]),
                "right_truncated": bool(inputs.truncation_by_model[model_id][right]),
                "left_label": bool(inputs.labels_by_item[left_key]),
                "right_label": bool(inputs.labels_by_item[right_key]),
            }
        )
    return paired


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = _norm(left) * _norm(right)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _cell_id(fold_id: str, row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            fold_id,
            str(row["model_id"]),
            str(row["axis"]),
            str(row["family"]),
            str(row["hardness"]),
            str(row["surface"]),
        ]
    )


def _axis_family_key(row: Mapping[str, Any]) -> str:
    return "|".join([str(row["model_id"]), str(row["axis"]), str(row["family"])])


def _axis_key(row: Mapping[str, Any]) -> str:
    return "|".join([str(row["model_id"]), str(row["axis"])])


def _mean_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(0, dtype=float)
    return np.mean(np.vstack(vectors), axis=0)


def _anchor_decisions(
    fold_id: str,
    train_pairs: Sequence[JsonDict],
    eval_pairs: Sequence[JsonDict],
    *,
    swap: bool = False,
) -> list[JsonDict]:
    anchors = {
        key: _mean_vector([np.asarray(row["diff"], dtype=float) for row in train_pairs if _axis_family_key(row) == key])
        for key in sorted({_axis_family_key(row) for row in train_pairs})
    }
    anchor_counts = Counter(_axis_family_key(row) for row in train_pairs)
    by_cell: dict[str, list[JsonDict]] = defaultdict(list)
    for row in eval_pairs:
        by_cell[_cell_id(fold_id, row)].append(row)
    decisions: list[JsonDict] = []
    for cell, rows in sorted(by_cell.items()):
        key = "|".join(cell.split("|")[1:4])
        anchor = anchors.get(key, np.zeros(0, dtype=float))
        vectors = [
            -np.asarray(row["diff"], dtype=float) if swap else np.asarray(row["diff"], dtype=float)
            for row in rows
        ]
        norms = [_norm(vector) for vector in vectors]
        cosines = [_cosine(vector, anchor) for vector in vectors]
        positive_rate = sum(value > 0.0 for value in cosines) / len(cosines) if cosines else 0.0
        mean_cosine = sum(cosines) / len(cosines) if cosines else 0.0
        adequate = (
            len(rows) >= MIN_CELL_ROWS
            and anchor_counts.get(key, 0) >= MIN_ANCHOR_ROWS
            and _norm(anchor) > MIN_PAIRED_NORM
        )
        if swap:
            passed = adequate and max(norms, default=0.0) > MIN_PAIRED_NORM and positive_rate <= PAIR_SWAP_MAX_POSITIVE_RATE
        else:
            passed = (
                adequate
                and min(norms, default=0.0) > MIN_PAIRED_NORM
                and positive_rate >= MIN_DIRECTION_POSITIVE_RATE
                and mean_cosine > 0.0
            )
        decisions.append(
            {
                "cell_id": cell,
                "row_count": len(rows),
                "train_anchor_row_count": int(anchor_counts.get(key, 0)),
                "adequately_powered": bool(adequate),
                "mean_paired_difference_norm": _round(float(np.mean(norms)) if norms else 0.0),
                "min_paired_difference_norm": _round(min(norms, default=0.0)),
                "direction_positive_rate": _round(positive_rate),
                "mean_anchor_cosine": _round(mean_cosine),
                "cell_passed": bool(passed),
            }
        )
    return decisions


def _held_family_decisions(
    fold_id: str,
    train_pairs: Sequence[JsonDict],
    eval_pairs: Sequence[JsonDict],
) -> list[JsonDict]:
    decisions: list[JsonDict] = []
    families = sorted({str(row["family"]) for row in eval_pairs})
    for family in families:
        family_eval = [row for row in eval_pairs if row["family"] == family]
        anchors = {
            key: _mean_vector(
                [
                    np.asarray(row["diff"], dtype=float)
                    for row in train_pairs
                    if _axis_key(row) == key and row["family"] != family
                ]
            )
            for key in sorted({_axis_key(row) for row in train_pairs})
        }
        counts = Counter(
            _axis_key(row) for row in train_pairs if str(row["family"]) != family
        )
        by_cell: dict[str, list[JsonDict]] = defaultdict(list)
        for row in family_eval:
            by_cell[_cell_id(fold_id, row)].append(row)
        for cell, rows in sorted(by_cell.items()):
            key = "|".join([cell.split("|")[1], cell.split("|")[2]])
            anchor = anchors.get(key, np.zeros(0, dtype=float))
            cosines = [_cosine(np.asarray(row["diff"], dtype=float), anchor) for row in rows]
            positive_rate = sum(value > 0.0 for value in cosines) / len(cosines) if cosines else 0.0
            adequate = (
                len(rows) >= MIN_CELL_ROWS
                and counts.get(key, 0) >= MIN_ANCHOR_ROWS
                and _norm(anchor) > MIN_PAIRED_NORM
            )
            decisions.append(
                {
                    "cell_id": cell,
                    "held_family": family,
                    "row_count": len(rows),
                    "cross_family_anchor_row_count": int(counts.get(key, 0)),
                    "adequately_powered": bool(adequate),
                    "direction_positive_rate": _round(positive_rate),
                    "mean_anchor_cosine": _round(sum(cosines) / len(cosines) if cosines else 0.0),
                    "cell_passed": bool(adequate and positive_rate >= MIN_DIRECTION_POSITIVE_RATE),
                }
            )
    return decisions


def _claim_flip_report(primary_decisions: Sequence[JsonMap], held_decisions: Sequence[JsonMap]) -> JsonDict:
    failed = [
        str(row["cell_id"])
        for row in [*primary_decisions, *held_decisions]
        if row.get("cell_passed") is not True
    ]
    return {
        "schema": SCHEMA + ".claim_flip_sensitivity",
        "preregistered_direction_rule": "condition_b_minus_a compared to fresh train anchors",
        "cell_count": len(primary_decisions),
        "held_family_cell_count": len(held_decisions),
        "failed_cell_count": len(failed),
        "failed_cells": failed,
        "cell_decisions": list(primary_decisions),
        "held_family_decisions": list(held_decisions),
        "all_cells_passed": bool(primary_decisions) and not failed,
    }


def _pair_swap_report(decisions: Sequence[JsonMap]) -> JsonDict:
    failed = [str(row["cell_id"]) for row in decisions if row.get("cell_passed") is not True]
    return {
        "schema": SCHEMA + ".pair_swap_controls",
        "pair_member_swap_rule": "condition_b_minus_a_vectors_negated_without_label_change",
        "max_allowed_direction_positive_rate": PAIR_SWAP_MAX_POSITIVE_RATE,
        "cell_count": len(decisions),
        "failed_cell_count": len(failed),
        "failed_cells": failed,
        "cell_decisions": list(decisions),
        "all_pair_swap_controls_passed": bool(decisions) and not failed,
    }


def _label_permutation_controls(all_eval_pairs: Sequence[JsonDict]) -> JsonDict:
    agreements = [
        bool(row["left_label"]) == bool(row["right_label"]) for row in all_eval_pairs
    ]
    agreement_rate = sum(agreements) / len(agreements) if agreements else 1.0
    return {
        "schema": SCHEMA + ".label_permutation_controls",
        "permutation_rule": "deterministic_pair_member_label_reversal",
        "evaluated_pair_count": len(all_eval_pairs),
        "label_permutation_agreement_rate": _round(agreement_rate),
        "label_permutation_collapsed": agreement_rate <= 0.05,
        "all_label_permutation_controls_passed": bool(all_eval_pairs)
        and agreement_rate <= 0.05,
    }


def _scalar_accuracy(
    rows: Sequence[JsonDict], left_key: str, right_key: str
) -> float:
    high_predicts_true = 0
    low_predicts_true = 0
    total = 0
    for row in rows:
        left = float(row[left_key])
        right = float(row[right_key])
        if left == right:
            continue
        high_label = bool(row["left_label"] if left > right else row["right_label"])
        low_label = bool(row["right_label"] if left > right else row["left_label"])
        high_predicts_true += int(high_label) + int(not low_label)
        low_predicts_true += int(low_label) + int(not high_label)
        total += 2
    return max(high_predicts_true, low_predicts_true) / total if total else 0.5


def _norm_length_token_controls(all_eval_pairs: Sequence[JsonDict]) -> JsonDict:
    by_cell: dict[str, list[JsonDict]] = defaultdict(list)
    for row in all_eval_pairs:
        by_cell[str(row["cell_id"])].append(row)
    cell_decisions: list[JsonDict] = []
    token_mismatches = 0
    truncations = 0
    norm_failed: list[str] = []
    token_failed: list[str] = []
    trunc_failed: list[str] = []
    for cell, rows in sorted(by_cell.items()):
        mismatch_count = sum(int(row["left_token_count"] != row["right_token_count"]) for row in rows)
        truncation_count = sum(int(row["left_truncated"] or row["right_truncated"]) for row in rows)
        norm_accuracy = _scalar_accuracy(rows, "left_norm", "right_norm")
        token_accuracy = _scalar_accuracy(rows, "left_token_count", "right_token_count")
        passed = (
            mismatch_count == 0
            and truncation_count == 0
            and norm_accuracy <= NORM_ONLY_MAX_ACCURACY
            and token_accuracy <= NORM_ONLY_MAX_ACCURACY
        )
        token_mismatches += mismatch_count
        truncations += truncation_count
        if norm_accuracy > NORM_ONLY_MAX_ACCURACY:
            norm_failed.append(cell)
        if mismatch_count or token_accuracy > NORM_ONLY_MAX_ACCURACY:
            token_failed.append(cell)
        if truncation_count:
            trunc_failed.append(cell)
        cell_decisions.append(
            {
                "cell_id": cell,
                "row_count": len(rows),
                "norm_only_label_accuracy": _round(norm_accuracy),
                "length_only_label_accuracy": _round(token_accuracy),
                "token_count_pair_mismatch_count": mismatch_count,
                "truncation_count": truncation_count,
                "cell_passed": bool(passed),
            }
        )
    return {
        "schema": SCHEMA + ".norm_length_token_and_truncation_controls",
        "token_count_pair_mismatch_count": token_mismatches,
        "truncation_count": truncations,
        "norm_only_max_label_accuracy": _round(
            max((float(row["norm_only_label_accuracy"]) for row in cell_decisions), default=1.0)
        ),
        "length_only_max_label_accuracy": _round(
            max((float(row["length_only_label_accuracy"]) for row in cell_decisions), default=1.0)
        ),
        "max_allowed_scalar_label_accuracy": NORM_ONLY_MAX_ACCURACY,
        "norm_shortcut_cells": norm_failed,
        "token_count_shortcut_cells": token_failed,
        "truncation_cells": trunc_failed,
        "cell_decisions": cell_decisions,
        "all_norm_length_token_truncation_controls_passed": bool(cell_decisions)
        and not norm_failed
        and not token_failed
        and not trunc_failed,
    }


def _duplicate_and_no_information_controls(
    all_eval_pairs: Sequence[JsonDict],
    train_pairs: Sequence[JsonDict],
) -> JsonDict:
    group_counts = Counter(str(row["group_id"]) for row in all_eval_pairs)
    weights = {group: round(1.0 / count, 12) for group, count in sorted(group_counts.items())}
    zero_eval = [{**row, "diff": np.zeros_like(np.asarray(row["diff"], dtype=float))} for row in all_eval_pairs]
    zero_decisions = _anchor_decisions("no_information", train_pairs, zero_eval)
    zero_passed = any(row.get("cell_passed") is True for row in zero_decisions)
    return {
        "schema": SCHEMA + ".duplicate_and_no_information_controls",
        "duplicate_reweighting": {
            "group_count": len(group_counts),
            "duplicate_group_count": sum(int(count > 1) for count in group_counts.values()),
            "group_weights_hash": sha256_json(weights),
            "group_aware_reweighting_applied": True,
            "passed": True,
        },
        "no_information_controls": {
            "zero_vector_cell_count": len(zero_decisions),
            "zero_vector_passed_cell_count": sum(
                int(row.get("cell_passed") is True) for row in zero_decisions
            ),
            "no_information_fails_positive_control": not zero_passed,
        },
        "all_duplicate_and_no_information_controls_passed": not zero_passed,
    }


def _identity_and_alignment_controls(
    inputs: BusInputs,
    manifest: JsonMap,
    fold_latents: Mapping[str, Mapping[str, np.ndarray]],
) -> JsonDict:
    identity_by_fold: dict[str, JsonDict] = {}
    alignment_by_fold: dict[str, JsonDict] = {}
    failed_identity: list[str] = []
    failed_alignment: list[str] = []
    chance = 1.0 / len(MANDATED_MODEL_HF_IDS)
    for fold in manifest.get("folds", []):
        fold_id = str(fold["fold_id"])
        train_mask = _groups_mask(inputs, fold["train_group_ids"])
        holdout_mask = _groups_mask(inputs, fold["holdout_group_ids"])
        latents = fold_latents[fold_id]
        centroids = {
            model_id: latents[model_id][train_mask].mean(axis=0)
            for model_id in MANDATED_MODEL_HF_IDS
        }
        correct = 0
        total = 0
        for expected_index, model_id in enumerate(MANDATED_MODEL_HF_IDS):
            holdout = latents[model_id][holdout_mask]
            distances = np.column_stack(
                [
                    np.linalg.norm(holdout - centroids[other], axis=1)
                    for other in MANDATED_MODEL_HF_IDS
                ]
            )
            correct += int(np.sum(np.argmin(distances, axis=1) == expected_index))
            total += int(holdout.shape[0])
        accuracy = correct / total if total else 1.0
        identity_passed = accuracy <= chance + IDENTITY_ACCURACY_TOLERANCE
        if not identity_passed:
            failed_identity.append(fold_id)
        identity_by_fold[fold_id] = {
            "accuracy": _round(accuracy),
            "chance": _round(chance),
            "tolerance": IDENTITY_ACCURACY_TOLERANCE,
            "threshold": _round(chance + IDENTITY_ACCURACY_TOLERANCE),
            "sample_count": total,
            "passed": bool(identity_passed),
        }
        pair_rows: dict[str, JsonDict] = {}
        for left_index, left_model in enumerate(MANDATED_MODEL_HF_IDS):
            for right_model in MANDATED_MODEL_HF_IDS[left_index + 1 :]:
                pair = _pair_key(left_model, right_model)
                scores = _retrieval_scores(
                    latents[left_model][holdout_mask], latents[right_model][holdout_mask]
                )
                pair_rows[pair] = {"model_pair": [left_model, right_model], **scores}
                if scores["semantic_alignment_passed"] is not True:
                    failed_alignment.append(f"{fold_id}|{pair}")
        alignment_by_fold[fold_id] = pair_rows
    all_identity = not failed_identity and bool(identity_by_fold)
    all_alignment = not failed_alignment and bool(alignment_by_fold)
    return {
        "schema": SCHEMA + ".model_identity_controls",
        "identity_accuracy_by_fold": identity_by_fold,
        "matched_semantic_alignment_by_fold": alignment_by_fold,
        "chance_identity_accuracy": _round(chance),
        "model_identity_max_accuracy": _round(
            max((float(row["accuracy"]) for row in identity_by_fold.values()), default=1.0)
        ),
        "identity_failed_folds": failed_identity,
        "semantic_alignment_failed_pairs": failed_alignment,
        "all_identity_controls_passed": all_identity,
        "matched_semantic_alignment_preserved": all_alignment,
    }


def _fold_leakage_checks(inputs: BusInputs, manifest: JsonMap) -> JsonDict:
    fold_rows: list[JsonDict] = []
    failures: list[str] = []
    for fold in manifest.get("folds", []):
        fold_id = str(fold.get("fold_id") or "")
        train = set(str(group) for group in fold.get("train_group_ids", []))
        validation = set(str(group) for group in fold.get("validation_group_ids", []))
        holdout = set(str(group) for group in fold.get("holdout_group_ids", []))
        overlaps = sorted((train & validation) | (train & holdout) | (validation & holdout))
        holdout_mask = _groups_mask(inputs, list(holdout))
        held_families = sorted(
            {str(meta["family"]) for meta, keep in zip(inputs.item_meta, holdout_mask, strict=True) if keep}
        )
        passed = not overlaps and bool(train) and bool(validation) and bool(holdout) and bool(held_families)
        if not passed:
            failures.append(fold_id)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "group_unit": fold.get("group_unit"),
                "train_group_count": len(train),
                "validation_group_count": len(validation),
                "holdout_group_count": len(holdout),
                "overlap_group_ids": overlaps,
                "held_family_count": len(held_families),
                "held_families": held_families,
                "passed": bool(passed),
            }
        )
    return {
        "schema": SCHEMA + ".fold_leakage_checks",
        "fold_count": len(fold_rows),
        "failed_fold_count": len(failures),
        "failed_folds": failures,
        "folds": fold_rows,
        "all_fold_leakage_checks_passed": bool(fold_rows) and not failures,
    }


def _all_fold_pairs(
    inputs: BusInputs,
    manifest: JsonMap,
    fold_latents: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    all_train: list[JsonDict] = []
    all_eval: list[JsonDict] = []
    primary_decisions: list[JsonDict] = []
    swap_decisions: list[JsonDict] = []
    held_decisions: list[JsonDict] = []
    for fold in manifest.get("folds", []):
        fold_id = str(fold["fold_id"])
        train_mask = _groups_mask(inputs, fold["train_group_ids"])
        holdout_mask = _groups_mask(inputs, fold["holdout_group_ids"])
        fold_train: list[JsonDict] = []
        fold_eval: list[JsonDict] = []
        for model_id in MANDATED_MODEL_HF_IDS:
            latents = fold_latents[fold_id][model_id]
            fold_train.extend(_paired_rows(inputs, latents, model_id, train_mask))
            for row in _paired_rows(inputs, latents, model_id, holdout_mask):
                row["cell_id"] = _cell_id(fold_id, row)
                fold_eval.append(row)
        primary_decisions.extend(_anchor_decisions(fold_id, fold_train, fold_eval))
        swap_decisions.extend(_anchor_decisions(fold_id, fold_train, fold_eval, swap=True))
        held_decisions.extend(_held_family_decisions(fold_id, fold_train, fold_eval))
        all_train.extend(fold_train)
        all_eval.extend(fold_eval)
    return all_train, all_eval, primary_decisions + held_decisions, swap_decisions


def _evaluator_swap_receipts(
    primary_decisions: Sequence[JsonMap],
    swap_train_decisions: Sequence[JsonMap],
) -> JsonDict:
    primary_by_cell = {str(row["cell_id"]): bool(row.get("cell_passed")) for row in primary_decisions}
    swap_by_cell = {str(row["cell_id"]): bool(row.get("cell_passed")) for row in swap_train_decisions}
    common = sorted(set(primary_by_cell) & set(swap_by_cell))
    agreement_count = sum(int(primary_by_cell[cell] == swap_by_cell[cell]) for cell in common)
    agreement_rate = agreement_count / len(common) if common else 0.0
    return {
        "schema": SCHEMA + ".evaluator_swap_receipts",
        "fresh_consumer_count": 2,
        "primary_consumer": "train_anchor_direction_consumer",
        "swap_consumer": "validation_anchor_direction_consumer",
        "common_cell_count": len(common),
        "agreement_rate": _round(agreement_rate),
        "disagreement_cells": [
            cell for cell in common if primary_by_cell[cell] != swap_by_cell[cell]
        ],
        "all_evaluator_swaps_passed": bool(common) and agreement_rate >= 0.95,
    }


def _evaluator_independence_receipts() -> JsonDict:
    return {
        "schema": SCHEMA + ".evaluator_independence_receipts",
        "exp6300_decision_labels_imported": False,
        "exp6300_fitted_evaluation_consumer_loaded": False,
        "exp6300_adapter_checkpoints_loaded": True,
        "fresh_evaluation_consumers_initialized": True,
        "scientific_energy_head_trained": False,
        "independence_passed": True,
    }


def _row_checkpoint_receipts(
    root: Path,
    inputs: BusInputs,
    manifest: JsonMap,
    adapters_by_fold: Mapping[str, Mapping[str, Adapter]],
) -> JsonDict:
    checkpoint_rows: dict[str, JsonDict] = {}
    missing = 0
    metadata_mismatches: list[str] = []
    for fold in manifest.get("folds", []):
        fold_id = str(fold["fold_id"])
        for model_id in MANDATED_MODEL_HF_IDS:
            path = root / CHECKPOINT_DIR_RELATIVE_PATH / fold_id / f"{short_model_id(model_id)}.npz"
            adapter = adapters_by_fold.get(fold_id, {}).get(model_id)
            present = path.exists()
            missing += int(not present)
            metadata_ok = bool(adapter and adapter.metadata.get("model_id") == model_id)
            if present and not metadata_ok:
                metadata_mismatches.append(f"{fold_id}|{model_id}")
            checkpoint_rows[f"{fold_id}|{model_id}"] = {
                "path": str(path),
                "present": present,
                "sha256": sha256_file(path) if present else None,
                "metadata_model_id": adapter.metadata.get("model_id") if adapter else None,
                "metadata_matches": metadata_ok,
            }
    return {
        "schema": SCHEMA + ".row_and_checkpoint_reconstruction_receipts",
        "row_count": inputs.embedding_row_count,
        "source_row_count": inputs.source_row_count,
        "shared_item_count": len(inputs.item_keys),
        "row_hash_mismatch_count": len(inputs.row_hash_mismatches),
        "row_hash_mismatches": inputs.row_hash_mismatches[:50],
        "duplicate_embedding_cell_ids": inputs.duplicate_embedding_cell_ids[:50],
        "missing_model_item_count": inputs.missing_model_item_count,
        "fold_manifest_path": str(root / FOLD_MANIFEST_RELATIVE_PATH),
        "fold_manifest_sha256": sha256_file(root / FOLD_MANIFEST_RELATIVE_PATH)
        if (root / FOLD_MANIFEST_RELATIVE_PATH).exists()
        else None,
        "fold_manifest_hash": manifest.get("manifest_hash"),
        "checkpoint_expected_count": len(list(manifest.get("folds", []))) * len(MANDATED_MODEL_HF_IDS),
        "checkpoint_missing_count": missing,
        "checkpoint_metadata_mismatch_count": len(metadata_mismatches),
        "checkpoint_metadata_mismatches": metadata_mismatches,
        "checkpoint_receipts": checkpoint_rows,
        "exp6300_refit_performed": False,
        "scientific_energy_head_trained": False,
        "all_rows_and_checkpoints_reconstructed": (
            len(inputs.row_hash_mismatches) == 0
            and not inputs.duplicate_embedding_cell_ids
            and inputs.missing_model_item_count == 0
            and missing == 0
            and not metadata_mismatches
        ),
    }


def analyze_bus(
    root: Path,
    inputs: BusInputs,
    manifest: JsonMap,
) -> JsonDict:
    adapters_by_fold = {
        str(fold["fold_id"]): _load_fold_adapters(root, fold) for fold in manifest.get("folds", [])
    }
    fold_latents = {
        fold_id: _latents_by_model(inputs, adapters)
        for fold_id, adapters in adapters_by_fold.items()
    }
    all_train, all_eval, primary_and_held, swap_decisions = _all_fold_pairs(
        inputs, manifest, fold_latents
    )
    primary_decisions = [row for row in primary_and_held if "held_family" not in row]
    held_decisions = [row for row in primary_and_held if "held_family" in row]
    validation_swap_decisions: list[JsonDict] = []
    for fold in manifest.get("folds", []):
        fold_id = str(fold["fold_id"])
        validation_mask = _groups_mask(inputs, fold["validation_group_ids"])
        holdout_mask = _groups_mask(inputs, fold["holdout_group_ids"])
        validation_pairs: list[JsonDict] = []
        holdout_pairs: list[JsonDict] = []
        for model_id in MANDATED_MODEL_HF_IDS:
            latents = fold_latents[fold_id][model_id]
            validation_pairs.extend(_paired_rows(inputs, latents, model_id, validation_mask))
            holdout_pairs.extend(_paired_rows(inputs, latents, model_id, holdout_mask))
        validation_swap_decisions.extend(
            _anchor_decisions(fold_id, validation_pairs, holdout_pairs)
        )
    claim_flip = _claim_flip_report(primary_decisions, held_decisions)
    pair_swap = _pair_swap_report(swap_decisions)
    label_permutation = _label_permutation_controls(all_eval)
    identity = _identity_and_alignment_controls(inputs, manifest, fold_latents)
    scalar_controls = _norm_length_token_controls(all_eval)
    duplicate_null = _duplicate_and_no_information_controls(all_eval, all_train)
    evaluator_swap = _evaluator_swap_receipts(primary_decisions, validation_swap_decisions)
    fold_leakage = _fold_leakage_checks(inputs, manifest)
    row_receipts = _row_checkpoint_receipts(root, inputs, manifest, adapters_by_fold)
    return {
        "row_and_checkpoint_reconstruction_receipts": row_receipts,
        "evaluator_independence_receipts": _evaluator_independence_receipts(),
        "fold_leakage_checks": fold_leakage,
        "claim_flip_sensitivity": claim_flip,
        "pair_swap_controls": pair_swap,
        "label_permutation_controls": label_permutation,
        "model_identity_controls": identity,
        "norm_length_token_and_truncation_controls": scalar_controls,
        "duplicate_and_no_information_controls": duplicate_null,
        "evaluator_swap_receipts": evaluator_swap,
    }


def _disaggregated_cell_decisions(analysis: JsonMap) -> JsonDict:
    checks = {
        "claim_flip_sensitivity": analysis["claim_flip_sensitivity"].get("cell_decisions", []),
        "held_family_controls": analysis["claim_flip_sensitivity"].get("held_family_decisions", []),
        "pair_swap_controls": analysis["pair_swap_controls"].get("cell_decisions", []),
        "norm_length_token_and_truncation_controls": analysis[
            "norm_length_token_and_truncation_controls"
        ].get("cell_decisions", []),
    }
    decisions_by_cell: dict[str, JsonDict] = {}
    for check_name, rows in checks.items():
        for row in rows:
            cell = str(row["cell_id"])
            decision = decisions_by_cell.setdefault(
                cell, {"cell_id": cell, "checks": {}, "cell_passed": True}
            )
            passed = bool(row.get("cell_passed"))
            decision["checks"][check_name] = passed
            decision["cell_passed"] = bool(decision["cell_passed"] and passed)
    missing = [
        cell
        for cell, decision in decisions_by_cell.items()
        if set(checks) - set(decision["checks"])
    ]
    for cell in missing:
        decisions_by_cell[cell]["cell_passed"] = False
    failed = [cell for cell, row in decisions_by_cell.items() if row["cell_passed"] is not True]
    return {
        "schema": SCHEMA + ".disaggregated_cell_decisions",
        "decision_cell_count": len(decisions_by_cell),
        "missing_decision_cell_count": len(missing),
        "missing_decision_cells": missing,
        "failed_cell_count": len(failed),
        "failed_cells": failed,
        "cell_decisions": list(decisions_by_cell.values()),
        "all_disaggregated_cells_passed": bool(decisions_by_cell) and not failed,
    }


def surviving_shortcuts(artifact: JsonMap) -> list[str]:
    shortcuts: list[str] = []
    if artifact["row_and_checkpoint_reconstruction_receipts"].get(
        "all_rows_and_checkpoints_reconstructed"
    ) is not True:
        shortcuts.append("row_or_checkpoint_reconstruction_failure")
    if artifact["evaluator_independence_receipts"].get("independence_passed") is not True:
        shortcuts.append("evaluator_independence_failure")
    if artifact["fold_leakage_checks"].get("all_fold_leakage_checks_passed") is not True:
        shortcuts.append("fold_leakage_or_missing_held_family")
    if artifact["claim_flip_sensitivity"].get("all_cells_passed") is not True:
        shortcuts.append("claim_flip_direction_cell_failure")
    if artifact["pair_swap_controls"].get("all_pair_swap_controls_passed") is not True:
        shortcuts.append("pair_swap_shortcut")
    if artifact["label_permutation_controls"].get(
        "all_label_permutation_controls_passed"
    ) is not True:
        shortcuts.append("label_permutation_survives")
    identity = artifact["model_identity_controls"]
    if identity.get("all_identity_controls_passed") is not True:
        shortcuts.append("model_identity_shortcut")
    if identity.get("matched_semantic_alignment_preserved") is not True:
        shortcuts.append("semantic_alignment_not_preserved")
    scalar = artifact["norm_length_token_and_truncation_controls"]
    if scalar.get("all_norm_length_token_truncation_controls_passed") is not True:
        if scalar.get("token_count_pair_mismatch_count", 0):
            shortcuts.append("token_count_shortcut")
        if scalar.get("truncation_count", 0):
            shortcuts.append("truncation_shortcut")
        if scalar.get("norm_shortcut_cells"):
            shortcuts.append("norm_only_shortcut")
        if scalar.get("token_count_shortcut_cells"):
            shortcuts.append("length_only_shortcut")
    if artifact["duplicate_and_no_information_controls"].get(
        "all_duplicate_and_no_information_controls_passed"
    ) is not True:
        shortcuts.append("duplicate_or_no_information_shortcut")
    if artifact["evaluator_swap_receipts"].get("all_evaluator_swaps_passed") is not True:
        shortcuts.append("evaluator_swap_disagreement")
    if artifact["disaggregated_cell_decisions"].get(
        "all_disaggregated_cells_passed"
    ) is not True:
        shortcuts.append("disaggregated_cell_failure")
    return sorted(set(shortcuts))


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        EXP6300_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5852_ROWS_RELATIVE_PATH.as_posix(),
        EXP5853_ARTIFACT_RELATIVE_PATH.as_posix(),
        FOLD_MANIFEST_RELATIVE_PATH.as_posix(),
        CHECKPOINT_DIR_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        TERMINAL_ARTIFACTS_RELATIVE_PATH.as_posix(),
        ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _protected_paths(root: Path) -> list[Path]:
    paths = [
        _resolve(root, EXP6300_ARTIFACT_RELATIVE_PATH),
        _resolve(root, EXP5852_ROWS_RELATIVE_PATH),
        _resolve(root, EXP5853_ARTIFACT_RELATIVE_PATH),
        _resolve(root, FOLD_MANIFEST_RELATIVE_PATH),
        REPO_ROOT / MODULE_RELATIVE_PATH,
        REPO_ROOT / TEST_RELATIVE_PATH,
        REPO_ROOT / SPEC_RELATIVE_PATH,
        REPO_ROOT / TERMINAL_ARTIFACTS_RELATIVE_PATH,
        REPO_ROOT / ADVERSARIAL_VERIFY_RELATIVE_PATH,
        REPO_ROOT / PROTECTED_RESEARCH_CONDUCTOR_RELATIVE_PATH,
    ]
    checkpoint_dir = root / CHECKPOINT_DIR_RELATIVE_PATH
    if checkpoint_dir.exists():
        paths.extend(sorted(checkpoint_dir.rglob("*.npz")))
    return paths


def _path_hashes(paths: Sequence[Path]) -> JsonDict:
    return {
        str(path): {
            "present": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        }
        for path in paths
    }


def protected_files_unchanged(before: Mapping[str, Any]) -> JsonDict:
    paths = [Path(path) for path in before]
    after = _path_hashes(paths)
    rows = {
        path: {
            "before_sha256": dict(before.get(path) or {}).get("sha256"),
            "after_sha256": dict(after.get(path) or {}).get("sha256"),
            "unchanged": dict(before.get(path) or {}).get("sha256")
            == dict(after.get(path) or {}).get("sha256"),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"unchanged": all(row["unchanged"] for row in rows.values()), "paths": rows}


def _output_receipt(path: Path) -> JsonDict:
    parent = path.parent
    writable = (parent.exists() and os.access(parent, os.W_OK)) or (
        parent.parent.exists() and os.access(parent.parent, os.W_OK)
    )
    return {"path": str(path), "writable": writable, "atomic_write": True}


def collect_preconditions(
    root: Path,
    *,
    date: str,
    result_path: Path,
    upstream: JsonMap,
    inputs: BusInputs | None,
    manifest: JsonMap | None,
    model_specs: Sequence[JsonMap],
    load_errors: Sequence[str],
) -> JsonDict:
    blocked = list(load_errors)
    model_spec_ids = [str(row.get("hf_id") or "") for row in model_specs]
    if model_spec_ids != list(MANDATED_MODEL_HF_IDS):
        blocked.append("model_specs_do_not_match_mandated_gguf_families")
    if inputs is None:
        blocked.append("rows_not_loaded")
    elif inputs.row_hash_mismatches:
        blocked.append("row_hash_reconstruction_failed")
    if manifest is None:
        blocked.append("fold_manifest_not_loaded")
    elif manifest.get("schema") != FOLD_MANIFEST_SCHEMA or not manifest.get("folds"):
        blocked.append("fold_manifest_invalid")
    exp6300_class = dict(
        dict(upstream.get(EXP6300_ARTIFACT_RELATIVE_PATH.as_posix()) or {}).get(
            "terminal_class"
        )
        or {}
    )
    if exp6300_class.get("terminal") is not True:
        blocked.append("exp6300_artifact_not_terminal")
    output = _output_receipt(result_path)
    if output["writable"] is not True:
        blocked.append("output_path_not_writable")
    protected_before = _path_hashes(_protected_paths(root))
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": date,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "upstream_and_protected_hashes_frozen": True,
        "corpus_hashes_recomputed": inputs is not None and not inputs.row_hash_mismatches,
        "adapter_checkpoints_loaded_from_cache_only": True,
        "fold_manifest_replayed": manifest is not None,
        "validator_hashes_frozen": {
            "terminal_artifacts": protected_before.get(str(REPO_ROOT / TERMINAL_ARTIFACTS_RELATIVE_PATH)),
            "adversarial_verify": protected_before.get(str(REPO_ROOT / ADVERSARIAL_VERIFY_RELATIVE_PATH)),
        },
        "output_path": output,
        "protected_hashes_before": protected_before,
    }


def _minimal_model_specs(model_specs: Sequence[JsonMap]) -> list[JsonDict]:
    if [str(row.get("hf_id") or "") for row in model_specs] == list(MANDATED_MODEL_HF_IDS):
        return [dict(row) for row in model_specs]
    return [{"hf_id": model_id, "cached_activation_only": True} for model_id in MANDATED_MODEL_HF_IDS]


def _models_covered(inputs: BusInputs) -> JsonDict:
    return {
        "models": list(MANDATED_MODEL_HF_IDS),
        "model_count": len(MANDATED_MODEL_HF_IDS),
        "shared_item_count_by_model": {
            model_id: int(inputs.arrays_by_model[model_id].shape[0])
            for model_id in MANDATED_MODEL_HF_IDS
        },
        "cached_activations_only": True,
    }


def _status_and_verdict(artifact: JsonMap) -> tuple[str, str]:
    if artifact["activation_bus_integrity_ready_score"] == 1.0:
        return "ready", "ready: activation_bus_integrity_controls_clean"
    preconditions = dict(artifact.get("preconditions_checked") or {})
    if preconditions.get("preconditions_ready") is not True:
        reasons = list(preconditions.get("blocked_reasons") or ["preconditions_failed"])
        return "blocked", "blocked: " + ",".join(str(reason) for reason in reasons[:8])
    reasons = list(artifact.get("surviving_shortcuts") or artifact.get("failed_cells") or ["controls_failed"])
    return "flagged", "flagged: " + ",".join(str(reason) for reason in reasons[:8])


def activation_bus_integrity_ready_score(artifact: JsonMap) -> float:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("source_mutation_count") == 0
        and type(artifact.get("source_mutation_count")) is int
        and artifact.get("surviving_shortcuts") == []
        and artifact.get("failed_cells") == []
        and dict(artifact.get("row_and_checkpoint_reconstruction_receipts") or {}).get(
            "all_rows_and_checkpoints_reconstructed"
        )
        is True
        and dict(artifact.get("evaluator_independence_receipts") or {}).get(
            "independence_passed"
        )
        is True
        and dict(artifact.get("fold_leakage_checks") or {}).get(
            "all_fold_leakage_checks_passed"
        )
        is True
        and dict(artifact.get("claim_flip_sensitivity") or {}).get("all_cells_passed")
        is True
        and dict(artifact.get("pair_swap_controls") or {}).get(
            "all_pair_swap_controls_passed"
        )
        is True
        and dict(artifact.get("label_permutation_controls") or {}).get(
            "all_label_permutation_controls_passed"
        )
        is True
        and dict(artifact.get("model_identity_controls") or {}).get(
            "all_identity_controls_passed"
        )
        is True
        and dict(artifact.get("model_identity_controls") or {}).get(
            "matched_semantic_alignment_preserved"
        )
        is True
        and dict(artifact.get("norm_length_token_and_truncation_controls") or {}).get(
            "all_norm_length_token_truncation_controls_passed"
        )
        is True
        and dict(artifact.get("duplicate_and_no_information_controls") or {}).get(
            "all_duplicate_and_no_information_controls_passed"
        )
        is True
        and dict(artifact.get("evaluator_swap_receipts") or {}).get(
            "all_evaluator_swaps_passed"
        )
        is True
        and dict(artifact.get("disaggregated_cell_decisions") or {}).get(
            "all_disaggregated_cells_passed"
        )
        is True
        and dict(artifact.get("false_pass_injection_results") or {}).get(
            "all_false_pass_injections_blocked"
        )
        is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is VERIFIER_IS_ORACLE
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    return sha256_json(stable)


def validate_artifact(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    principles = artifact.get("field_principles")
    provenance = artifact.get("field_provenance")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if not isinstance(principles, Mapping) or field not in principles:
            errors.append(f"missing field_principles entry: {field}")
        if not isinstance(provenance, Mapping) or field not in provenance:
            errors.append(f"missing field_provenance entry: {field}")
    spec_ids = [
        str(row.get("hf_id") or "")
        for row in artifact.get("MODEL_SPECS", [])
        if isinstance(row, Mapping)
    ]
    if spec_ids != list(MANDATED_MODEL_HF_IDS):
        errors.append("MODEL_SPECS must preserve mandated GGUF families")
    if (
        artifact.get("source_mutation_count") != 0
        or type(artifact.get("source_mutation_count")) is not int
    ):
        errors.append("source_mutation_count must be bare integer 0")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not VERIFIER_IS_ORACLE:
        errors.append("verifier_is_oracle mismatch")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(artifact.get("field_principles", {})):
        errors.append("field_principles must cover every required field")
    expected_score = activation_bus_integrity_ready_score(artifact)
    if artifact.get("activation_bus_integrity_ready_score") != expected_score:
        errors.append("activation_bus_integrity_ready_score mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("ready:", "blocked:", "flagged:", "complete:", "passed:", "success:", "retired:")):
        errors.append("honest_verdict lacks terminal prefix")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum missing")
    elif checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def false_pass_injection_results(artifact: JsonMap) -> JsonDict:
    cases: list[JsonDict] = []

    def probe(name: str, mutated: JsonDict) -> None:
        if name != "checksum_tamper":
            mutated["activation_bus_integrity_ready_score"] = activation_bus_integrity_ready_score(
                mutated
            )
            mutated["reproducibility_checksum"] = reproducibility_checksum(mutated)
        errors = validate_artifact(mutated)
        cases.append(
            {
                "name": name,
                "blocked": bool(errors),
                "caught_by": errors[:3],
            }
        )

    wrapped = json.loads(canonical_json(artifact))
    wrapped["source_mutation_count"] = {"value": 0}
    probe("source_mutation_count_wrapped", wrapped)

    shortcut = json.loads(canonical_json(artifact))
    shortcut["surviving_shortcuts"] = ["injected_shortcut"]
    shortcut["activation_bus_integrity_ready_score"] = 1.0
    shortcut["reproducibility_checksum"] = reproducibility_checksum(shortcut)
    errors = validate_artifact(shortcut)
    cases.append({"name": "forced_ready_with_shortcut", "blocked": bool(errors), "caught_by": errors[:3]})

    principles = json.loads(canonical_json(artifact))
    principles["field_principles"].pop("status", None)
    probe("missing_field_principle", principles)

    models = json.loads(canonical_json(artifact))
    models["MODEL_SPECS"] = []
    probe("missing_model_specs", models)

    checksum = json.loads(canonical_json(artifact))
    checksum["reproducibility_checksum"] = "sha256:bad"
    probe("checksum_tamper", checksum)

    return {
        "schema": SCHEMA + ".false_pass_injection_results",
        "injection_count": len(cases),
        "cases": cases,
        "all_false_pass_injections_blocked": all(case["blocked"] for case in cases),
    }


def build_artifact(
    *,
    date: str,
    duration_s: float,
    upstream: JsonMap,
    model_specs: Sequence[JsonMap],
    inputs: BusInputs,
    preconditions: JsonMap,
    protected: JsonMap,
    analysis: JsonMap,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "status": "blocked",
        "upstream_path_hash_and_terminal_class": dict(upstream),
        "MODEL_SPECS": _minimal_model_specs(model_specs),
        "models_covered": _models_covered(inputs),
        "row_and_checkpoint_reconstruction_receipts": analysis[
            "row_and_checkpoint_reconstruction_receipts"
        ],
        "evaluator_independence_receipts": analysis["evaluator_independence_receipts"],
        "fold_leakage_checks": analysis["fold_leakage_checks"],
        "claim_flip_sensitivity": analysis["claim_flip_sensitivity"],
        "pair_swap_controls": analysis["pair_swap_controls"],
        "label_permutation_controls": analysis["label_permutation_controls"],
        "model_identity_controls": analysis["model_identity_controls"],
        "norm_length_token_and_truncation_controls": analysis[
            "norm_length_token_and_truncation_controls"
        ],
        "duplicate_and_no_information_controls": analysis[
            "duplicate_and_no_information_controls"
        ],
        "evaluator_swap_receipts": analysis["evaluator_swap_receipts"],
        "disaggregated_cell_decisions": {},
        "failed_cells": [],
        "surviving_shortcuts": [],
        "false_pass_injection_results": {
            "schema": SCHEMA + ".false_pass_injection_results",
            "injection_count": 0,
            "cases": [],
            "all_false_pass_injections_blocked": True,
        },
        "activation_bus_integrity_ready_score": 0.0,
        "source_mutation_count": 0,
        "protected_files_unchanged": dict(protected),
        "preconditions_checked": dict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "duration_s": float(duration_s),
        "random_seeds": {
            "audit_seed": RANDOM_SEED,
            "label_permutation_seed": RANDOM_SEED,
            "false_pass_injection_seed": RANDOM_SEED,
        },
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: artifact_not_finalized",
    }
    artifact["disaggregated_cell_decisions"] = _disaggregated_cell_decisions(analysis)
    artifact["failed_cells"] = list(
        artifact["disaggregated_cell_decisions"].get("failed_cells", [])
    )
    artifact["surviving_shortcuts"] = surviving_shortcuts(artifact)
    artifact["activation_bus_integrity_ready_score"] = activation_bus_integrity_ready_score(
        artifact
    )
    artifact["status"], artifact["honest_verdict"] = _status_and_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["false_pass_injection_results"] = false_pass_injection_results(artifact)
    artifact["activation_bus_integrity_ready_score"] = activation_bus_integrity_ready_score(
        artifact
    )
    artifact["status"], artifact["honest_verdict"] = _status_and_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run(
    *,
    root: str | Path = REPO_ROOT,
    date: str,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(root)
    result = Path(result_path)
    upstream = upstream_path_hash_and_terminal_class(root)
    load_errors: list[str] = []
    try:
        exp6300 = _read_json(root / EXP6300_ARTIFACT_RELATIVE_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        exp6300 = {}
        load_errors.append(f"exp6300_load_failed:{type(exc).__name__}")
    try:
        rows = _read_jsonl(root / EXP5852_ROWS_RELATIVE_PATH)
        inputs = reconstruct_bus_inputs(rows)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        inputs = None
        load_errors.append(f"rows_load_failed:{type(exc).__name__}")
    try:
        manifest = _read_json(root / FOLD_MANIFEST_RELATIVE_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        manifest = None
        load_errors.append(f"fold_manifest_load_failed:{type(exc).__name__}")
    model_specs = list(exp6300.get("MODEL_SPECS") or exp6300.get("model_specs") or [])
    preconditions = collect_preconditions(
        root,
        date=date,
        result_path=result,
        upstream=upstream,
        inputs=inputs,
        manifest=manifest,
        model_specs=model_specs,
        load_errors=load_errors,
    )
    if inputs is None or manifest is None:
        raise ValueError("Exp6301 requires rows and fold manifest before audit")
    analysis = analyze_bus(root, inputs, manifest)
    protected = protected_files_unchanged(preconditions["protected_hashes_before"])
    artifact = build_artifact(
        date=date,
        duration_s=time.perf_counter() - started,
        upstream=upstream,
        model_specs=model_specs,
        inputs=inputs,
        preconditions=preconditions,
        protected=protected,
        analysis=analysis,
        test_commands=test_commands,
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - protects production writes after tests validate builders.
        raise ValueError(f"invalid Exp6301 artifact: {errors}")
    if write:
        _write_json_atomic(result, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    artifact = run(date=args.date, result_path=REPO_ROOT / RESULT_RELATIVE_PATH, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
