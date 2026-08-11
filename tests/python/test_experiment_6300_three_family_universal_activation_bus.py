"""Tests for Exp6300 Universal Activation Bus.

Spec refs: REQ-KONA-6300, SCENARIO-KONA-6300-FOLDS,
SCENARIO-KONA-6300-UNLABELED-FIT, SCENARIO-KONA-6300-CONTROLS,
SCENARIO-KONA-6300-IDENTITY.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6300_three_family_universal_activation_bus as mod


def _condition(row_id: str, suffix: str, vector: np.ndarray) -> dict[str, object]:
    return {
        "condition_id": f"{row_id}-{suffix}",
        "condition_suffix": suffix,
        "token_count": 16 + (suffix == "b"),
        "base_token_count": 16 + (suffix == "b"),
        "embedding": [round(float(value), 8) for value in vector],
        "embedding_shape": [int(vector.shape[0])],
        "embedding_sha256": mod.sha256_json([round(float(value), 8) for value in vector]),
    }


def _synthetic_rows(*, permute_last_model: bool = False) -> list[dict[str, object]]:
    rng = np.random.default_rng(6300)
    dims = {
        mod.MANDATED_MODEL_HF_IDS[0]: 6,
        mod.MANDATED_MODEL_HF_IDS[1]: 7,
        mod.MANDATED_MODEL_HF_IDS[2]: 5,
    }
    transforms = {model_id: rng.normal(size=(3, dim)) for model_id, dim in dims.items()}
    keys: list[tuple[str, str]] = []
    latents: dict[tuple[str, str], np.ndarray] = {}
    for group_index in range(30):
        row_id = f"synthetic-row-{group_index:03d}"
        for suffix_index, suffix in enumerate(("a", "b")):
            latent = np.array(
                [
                    np.sin(group_index / 3.0),
                    np.cos(group_index / 5.0),
                    (group_index % 7) / 3.0 + suffix_index * 0.25,
                ],
                dtype=float,
            )
            key = (row_id, suffix)
            keys.append(key)
            latents[key] = latent
    shuffled_keys = [keys[index] for index in rng.permutation(len(keys))]
    permuted = dict(zip(keys, shuffled_keys, strict=True))
    rows: list[dict[str, object]] = []
    for group_index in range(30):
        row_id = f"synthetic-row-{group_index:03d}"
        for model_id in mod.MANDATED_MODEL_HF_IDS:
            source_keys = [(row_id, "a"), (row_id, "b")]
            if permute_last_model and model_id == mod.MANDATED_MODEL_HF_IDS[2]:
                source_keys = [permuted[key] for key in source_keys]
            vectors = [
                latents[source_key] @ transforms[model_id] + (0.01 * model_index)
                for model_index, source_key in enumerate(source_keys)
            ]
            rows.append(
                {
                    "schema": "synthetic.exp6300.row",
                    "source_row_id": row_id,
                    "source_row_order": group_index,
                    "pair_group_id": f"synthetic-group-{group_index:03d}",
                    "model_hf_id": model_id,
                    "model_family": model_id.rsplit("/", 1)[-1],
                    "family": ["finite_domain_csp", "weighted_maxsat", "planning"][group_index % 3],
                    "surface_kind": ["symbol_relabel", "order_paraphrase"][group_index % 2],
                    "change": ["addition", "recurrence", "supersession"][group_index % 3],
                    "split": ["train", "dev", "science"][group_index % 3],
                    "condition_embeddings": [
                        _condition(row_id, "a", vectors[0]),
                        _condition(row_id, "b", vectors[1]),
                    ],
                    "oracle_label_receipt": {"exact_label": group_index % 2 == 0},
                }
            )
    return rows


def test_fold_manifest_keeps_matched_groups() -> None:
    """SCENARIO-KONA-6300-FOLDS: matched rows stay inside one split per fold."""

    corpus = mod.build_item_tables(_synthetic_rows())
    manifest = mod.build_fold_manifest(corpus, n_folds=3, seed=6300)

    assert manifest["schema"] == mod.FOLD_MANIFEST_SCHEMA
    assert len(manifest["folds"]) == 3
    for fold in manifest["folds"]:
        split_by_group: dict[str, set[str]] = {}
        for split_name in ("train", "validation", "holdout"):
            for group_id in fold[f"{split_name}_group_ids"]:
                split_by_group.setdefault(group_id, set()).add(split_name)
        assert all(len(split_names) == 1 for split_names in split_by_group.values())
        assert fold["heldout_task_families"]
        assert fold["heldout_perturbation_families"]
        assert fold["heldout_template_families"]


def test_unlabeled_adapter_fit_clears_controls(tmp_path: Path) -> None:
    """SCENARIO-KONA-6300-UNLABELED-FIT/CONTROLS/IDENTITY: adapters pass gates."""

    corpus = mod.build_item_tables(_synthetic_rows())
    manifest = mod.build_fold_manifest(corpus, n_folds=3, seed=6300)
    analysis = mod.analyze_corpus(
        corpus,
        manifest,
        shared_dimension=3,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=6300,
    )

    assert analysis["shared_activation_bus_ready_score"] == 1.0
    assert analysis["unlabeled_fit_contract"]["exact_labels_used_for_fitting"] is False
    assert analysis["unlabeled_fit_contract"]["energy_head_trained"] is False
    assert analysis["no_live_model_load_receipt"]["llm_loaded"] is False
    assert all(
        row["passed"]
        for rows in analysis["reconstruction_metrics_by_model_and_fold"].values()
        for row in rows.values()
    )
    assert all(row["passed"] for row in analysis["model_identity_accuracy_by_fold"].values())
    for fold_rows in analysis["cross_model_retrieval_metrics_by_pair_and_fold"].values():
        for row in fold_rows.values():
            assert row["learned_bus_mrr"] > row["raw_padded_mrr"]
            assert row["learned_bus_mrr"] > row["random_projection_mrr"]
    for receipt in analysis["checkpoint_paths_and_hashes"].values():
        assert Path(receipt["path"]).exists()
        assert receipt["sha256"].startswith("sha256:")


def test_pair_permutation_blocks_readiness(tmp_path: Path) -> None:
    """SCENARIO-KONA-6300-CONTROLS: permuted matched rows do not become ready."""

    corpus = mod.build_item_tables(_synthetic_rows(permute_last_model=True))
    manifest = mod.build_fold_manifest(corpus, n_folds=3, seed=6300)
    analysis = mod.analyze_corpus(
        corpus,
        manifest,
        shared_dimension=3,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=6300,
    )

    assert analysis["shared_activation_bus_ready_score"] == 0.0
    assert analysis["readiness_gate_summary"]["failed_gate_count"] > 0


def test_artifact_serialization_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-KONA-6300-UNLABELED-FIT: artifact fields validate and serialize."""

    corpus = mod.build_item_tables(_synthetic_rows())
    manifest = mod.build_fold_manifest(corpus, n_folds=3, seed=6300)
    manifest_path = tmp_path / "fold_manifest.json"
    mod.write_json_atomic(manifest_path, manifest)
    monkeypatch.setattr(
        mod,
        "DEFAULT_TEST_EXIT_CODES",
        {command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
    )
    analysis = mod.analyze_corpus(
        corpus,
        manifest,
        shared_dimension=3,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=6300,
    )
    artifact = mod.build_artifact(
        corpus=corpus,
        manifest=manifest,
        manifest_path=manifest_path,
        analysis=analysis,
        date="20260811",
        duration_s=1.25,
        preconditions=mod.synthetic_preconditions(tmp_path),
        protected_files=mod.synthetic_protected_files_unchanged(),
    )
    artifact_path = tmp_path / "artifact.json"
    mod.write_json_atomic(artifact_path, artifact)
    loaded = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(loaded)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(loaded["field_principles"])
    assert loaded["random_seed"] == mod.RANDOM_SEED
    assert "replay" in loaded["inference_substrate"]
    assert mod.validate_artifact(loaded) == []
    loaded["source_model_weight_mutation_count"] = {"value": 0}
    errors = mod.validate_artifact(loaded)
    assert "source_model_weight_mutation_count must be bare integer 0" in errors


def test_defensive_validation_paths(tmp_path: Path) -> None:
    """SCENARIO-KONA-6300-FOLDS: malformed inputs fail before fitting."""

    rows = _synthetic_rows()
    rows.append({**rows[0], "model_hf_id": "ignored/not-mandated"})
    corpus = mod.build_item_tables(rows)
    assert corpus.embedding_row_count == 90

    bad_condition = _synthetic_rows()
    bad_condition[0]["condition_embeddings"] = [None]
    with pytest.raises(ValueError, match="condition_embeddings"):
        mod.build_item_tables(bad_condition)

    bad_vector = _synthetic_rows()
    bad_vector[0]["condition_embeddings"][0]["embedding"] = [[1.0]]
    with pytest.raises(ValueError, match="one-dimensional"):
        mod.build_item_tables(bad_vector)

    with pytest.raises(ValueError, match="no matched"):
        mod.build_item_tables([])

    missing_model_rows = [
        row
        for row in _synthetic_rows()
        if not (
            row["model_hf_id"] == mod.MANDATED_MODEL_HF_IDS[2]
            and row["source_row_id"] == "synthetic-row-000"
        )
    ]
    with pytest.raises(ValueError, match="not aligned"):
        mod.build_item_tables(missing_model_rows)

    with pytest.raises(ValueError, match="empty split"):
        mod.build_fold_manifest(corpus, n_folds=80, seed=6300)

    manifest = mod.build_fold_manifest(corpus, n_folds=3, seed=6300)
    with pytest.raises(ValueError, match="shared_dimension"):
        mod.fit_fold_adapters(corpus, manifest["folds"][0], shared_dimension=99, seed=6300)

    manifest_path = tmp_path / "fold_manifest.json"
    mod.write_json_atomic(manifest_path, manifest)
    analysis = mod.analyze_corpus(
        corpus,
        manifest,
        shared_dimension=3,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=6300,
    )
    artifact = mod.build_artifact(
        corpus=corpus,
        manifest=manifest,
        manifest_path=manifest_path,
        analysis=analysis,
        date="20260811",
        duration_s=1.25,
        preconditions=mod.synthetic_preconditions(tmp_path),
        protected_files=mod.synthetic_protected_files_unchanged(),
    )

    tampered = dict(artifact)
    tampered.pop("status")
    assert "missing required field: status" in mod.validate_artifact(tampered)

    tampered = dict(artifact)
    tampered["field_principles"] = {}
    errors = mod.validate_artifact(tampered)
    assert "missing field_principles entry: status" in errors
    assert "field_principles must cover every required field" in errors

    tampered = dict(artifact)
    tampered["field_provenance"] = {}
    assert "missing field_provenance entry: status" in mod.validate_artifact(tampered)

    tampered = dict(artifact)
    tampered["models_used"] = []
    assert "models_used must preserve all mandated model IDs" in mod.validate_artifact(tampered)

    tampered = dict(artifact)
    tampered["MODEL_SPECS"] = []
    assert "MODEL_SPECS must preserve all mandated model IDs" in mod.validate_artifact(tampered)

    tampered = dict(artifact)
    tampered["verifier_is_oracle"] = True
    assert "verifier_is_oracle must be false" in mod.validate_artifact(tampered)

    tampered = dict(artifact)
    tampered["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate mismatch" in mod.validate_artifact(tampered)

    tampered = dict(artifact)
    tampered["honest_verdict"] = "not terminal"
    assert "honest_verdict lacks terminal prefix" in mod.validate_artifact(tampered)

    tampered = dict(artifact)
    tampered["reproducibility_checksum"] = ""
    assert "reproducibility_checksum missing" in mod.validate_artifact(tampered)

    tampered = dict(artifact)
    tampered["shared_dimension"] = 99
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(tampered)


def test_default_verification_exit_codes_are_successful() -> None:
    """REQ-KONA-6300: terminal artifacts record successful verification exits."""

    assert mod.DEFAULT_TEST_EXIT_CODES
    assert all(code == 0 for code in mod.DEFAULT_TEST_EXIT_CODES.values())
