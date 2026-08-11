"""Tests for Exp6301 activation bus integrity audit.

Spec refs: REQ-VERIFY-6301, SCENARIO-VERIFY-6301-RECONSTRUCT,
SCENARIO-VERIFY-6301-SHORTCUTS, SCENARIO-VERIFY-6301-INDEPENDENCE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_6301_activation_bus_integrity_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
TEST_COMMANDS = [
    ".venv/bin/pytest tests/python/test_experiment_6301_activation_bus_integrity_audit.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --branch --include=python/carnot/experiment_6301_activation_bus_integrity_audit.py -m pytest tests/python/test_experiment_6301_activation_bus_integrity_audit.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6301_activation_bus_integrity_audit.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6301_activation_bus_integrity_audit.py",
    ".venv/bin/python -m carnot.experiment_6298_terminal_evidence_preflight_linter --date 20260811 --no-run-commands",
    ".venv/bin/python scripts/determination_preservation_lint.py --all",
    ".venv/bin/python -m carnot.experiment_6301_activation_bus_integrity_audit --date 20260811",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6301_activation_bus_integrity_audit.json",
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _labels(axis: str, row_id: str) -> dict[str, bool]:
    if axis == "candidate_correctness":
        return {f"{row_id}-a": True, f"{row_id}-b": False}
    return {f"{row_id}-a": False, f"{row_id}-b": True}


def _condition(
    row_id: str,
    suffix: str,
    vector: np.ndarray,
    *,
    token_count: int = 16,
    truncated: bool = False,
) -> dict[str, Any]:
    embedding = [round(float(value), 8) for value in vector]
    return {
        "condition_id": f"{row_id}-{suffix}",
        "condition_suffix": suffix,
        "embedding": embedding,
        "embedding_shape": [len(embedding)],
        "embedding_sha256": mod.sha256_json(embedding),
        "token_count": token_count,
        "base_token_count": token_count,
        "target_pair_token_count": token_count,
        "token_parity_ok": True,
        "truncated": truncated,
    }


def _embedding_row(
    *,
    row_id: str,
    group_id: str,
    model_id: str,
    model_index: int,
    source_order: int,
    axis: str,
    family: str,
    hardness: str,
    surface: str,
    split: str,
    base: np.ndarray,
    reverse_direction: bool = False,
    model_leak: bool = False,
    token_delta: int = 0,
    truncated: bool = False,
) -> dict[str, Any]:
    direction = np.array([1.0, 0.0]) * (-1.0 if reverse_direction else 1.0)
    a_latent = base
    b_latent = base + direction
    leak = float(model_index) if model_leak else 0.0
    a_vector = np.array([a_latent[0], a_latent[1], leak])
    b_vector = np.array([b_latent[0], b_latent[1], leak])
    conditions = [
        _condition(row_id, "a", a_vector, truncated=truncated),
        _condition(row_id, "b", b_vector, token_count=16 + token_delta, truncated=truncated),
    ]
    diff = [round(float(b - a), 8) for a, b in zip(a_vector, b_vector, strict=True)]
    row = {
        "schema": "carnot.experiment_5852.three_family_paired_embeddings.v1.row",
        "source_row_order": source_order,
        "source_row_id": row_id,
        "source_row_hash": mod.sha256_json({"source": row_id}),
        "source_pair_id": row_id,
        "pair_group_id": group_id,
        "split": split,
        "axis": axis,
        "family": family,
        "change": "addition",
        "surface_kind": surface,
        "solver_effort_bin": hardness,
        "model_hf_id": model_id,
        "model_family": model_id.rsplit("/", 1)[-1],
        "model_file_sha256": mod.sha256_json({"model": model_id}),
        "model_local_path_hash": mod.sha256_json({"path": model_id}),
        "embedding_cell_id": f"{row_id}|{model_id}",
        "condition_embeddings": conditions,
        "paired_difference": diff,
        "paired_difference_sha256": mod.sha256_json(diff),
        "oracle_label_receipt": {
            "source": "synthetic_exact_labels",
            "labels_by_condition_id": _labels(axis, row_id),
            "verifier_is_oracle": True,
        },
        "feature_consumer_view": {
            "condition_features": [
                {
                    "condition_id": mod.sha256_json({"condition_id": c["condition_id"]}),
                    "embedding_sha256": c["embedding_sha256"],
                    "embedding_shape": c["embedding_shape"],
                    "token_count": c["token_count"],
                    "truncated": c["truncated"],
                }
                for c in conditions
            ],
            "paired_difference_sha256": mod.sha256_json(diff),
            "difference_orientation": "condition_b_minus_a",
            "preprocessing": "test",
        },
        "loader_receipt_hash": mod.sha256_json({"loader": model_id}),
        "row_hash": "",
    }
    row["row_hash"] = mod.embedding_row_hash(row)
    return row


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(mod.canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _write_adapter(path: Path, *, model_id: str, model_leak: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = np.array([[1.0, 0.0], [0.0, 1.0], [50.0, 0.0] if model_leak else [0.0, 0.0]])
    metadata = {
        "model_id": model_id,
        "input_dimension": 3,
        "shared_dimension": 2,
        "architecture": "linear_encoder_decoder",
    }
    np.savez(
        path,
        encoder_weight=encoder.astype(np.float32),
        encoder_bias=np.zeros(2, dtype=np.float32),
        decoder_weight=np.zeros((2, 3), dtype=np.float32),
        decoder_bias=np.zeros(3, dtype=np.float32),
        metadata_json=np.asarray(mod.canonical_json(metadata)),
    )


def _fixture(
    tmp_path: Path,
    *,
    reverse_holdout: bool = False,
    model_leak: bool = False,
    token_delta: int = 0,
    truncated: bool = False,
) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []
    groups_by_split = {
        "train": ["g0", "g1"],
        "validation": ["g2"],
        "holdout": ["g3", "g4"],
    }
    families = ["finite_domain_csp", "weighted_maxsat"]
    axes = ["candidate_correctness", "constraint_ablation"]
    hardnesses = ["low", "high"]
    surfaces = ["symbol_relabel", "order_paraphrase"]
    source_order = 0
    for split, group_ids in groups_by_split.items():
        for group_offset, group_id in enumerate(group_ids):
            for family in families:
                for axis in axes:
                    hardness = hardnesses[0]
                    surface = surfaces[0]
                    row_id = f"{split}-{group_id}-{family}-{axis}-{source_order:03d}"
                    signed = -10.0 if group_offset == 0 else 10.0
                    base = np.array([signed + (0.01 * source_order), float(source_order)])
                    for model_index, model_id in enumerate(mod.MANDATED_MODEL_HF_IDS):
                        rows.append(
                            _embedding_row(
                                row_id=row_id,
                                group_id=group_id,
                                model_id=model_id,
                                model_index=model_index,
                                source_order=source_order,
                                axis=axis,
                                family=family,
                                hardness=hardness,
                                surface=surface,
                                split=split,
                                base=base,
                                reverse_direction=reverse_holdout and split == "holdout",
                                model_leak=model_leak,
                                token_delta=token_delta if split == "holdout" else 0,
                                truncated=truncated and split == "holdout",
                            )
                        )
                    source_order += 1
    rows_path = tmp_path / mod.EXP5852_ROWS_RELATIVE_PATH
    _write_jsonl(rows_path, rows)
    model_specs = [{"hf_id": model_id, "family": model_id.rsplit("/", 1)[-1]} for model_id in mod.MANDATED_MODEL_HF_IDS]
    upstream = {
        "status": "complete",
        "honest_verdict": "complete: synthetic upstream",
        "MODEL_SPECS": model_specs,
        "models_used": list(mod.MANDATED_MODEL_HF_IDS),
    }
    _write_json(tmp_path / mod.EXP6300_ARTIFACT_RELATIVE_PATH, upstream)
    _write_json(
        tmp_path / mod.EXP5853_ARTIFACT_RELATIVE_PATH,
        {
            "status": "flagged",
            "honest_verdict": "flagged: synthetic exp5853 shortcut reminder",
            "surviving_shortcuts": ["synthetic_shortcut"],
        },
    )
    folds = []
    fold = {
        "fold_id": "fold_0",
        "fold_index": 0,
        "group_unit": "pair_group_id",
        "train_group_ids": groups_by_split["train"],
        "validation_group_ids": groups_by_split["validation"],
        "holdout_group_ids": groups_by_split["holdout"],
        "train_item_count": 16,
        "validation_item_count": 8,
        "holdout_item_count": 16,
    }
    fold["fold_hash"] = mod.sha256_json(fold)
    folds.append(fold)
    manifest = {
        "schema": mod.FOLD_MANIFEST_SCHEMA,
        "seed": mod.RANDOM_SEED,
        "n_folds": 1,
        "item_count": 40,
        "source_row_count": 20,
        "folds": folds,
        "manifest_hash": mod.sha256_json(folds),
    }
    _write_json(tmp_path / mod.FOLD_MANIFEST_RELATIVE_PATH, manifest)
    for model_id in mod.MANDATED_MODEL_HF_IDS:
        _write_adapter(
            tmp_path / mod.CHECKPOINT_DIR_RELATIVE_PATH / "fold_0" / f"{mod.short_model_id(model_id)}.npz",
            model_id=model_id,
            model_leak=model_leak,
        )
    return {
        "root": tmp_path,
        "rows": rows_path,
        "artifact": tmp_path / mod.EXP6300_ARTIFACT_RELATIVE_PATH,
        "manifest": tmp_path / mod.FOLD_MANIFEST_RELATIVE_PATH,
    }


def _run(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    fixture = _fixture(tmp_path, **kwargs)
    return mod.run(
        root=fixture["root"],
        date="20260811",
        result_path=fixture["root"] / mod.RESULT_RELATIVE_PATH,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_verify_6301_spec_declares_required_artifact_contract() -> None:
    """REQ-VERIFY-6301: the OpenSpec declares fields, models, and scenarios."""

    section = SPEC.read_text(encoding="utf-8").split("### REQ-VERIFY-6301", 1)[1]

    assert "SCENARIO-VERIFY-6301-RECONSTRUCT" in section
    assert "SCENARIO-VERIFY-6301-SHORTCUTS" in section
    assert "SCENARIO-VERIFY-6301-INDEPENDENCE" in section
    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for model_id in mod.MANDATED_MODEL_HF_IDS:
        assert model_id in section


def test_scenario_verify_6301_reconstructs_rows_and_cached_checkpoints(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6301-RECONSTRUCT: hashes and cached adapters define the bus."""

    artifact = _run(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "ready"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["activation_bus_integrity_ready_score"] == 1.0
    assert artifact["source_mutation_count"] == 0
    assert type(artifact["source_mutation_count"]) is int
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_HF_IDS)
    receipts = artifact["row_and_checkpoint_reconstruction_receipts"]
    assert receipts["row_hash_mismatch_count"] == 0
    assert receipts["checkpoint_missing_count"] == 0
    assert receipts["exp6300_refit_performed"] is False
    assert artifact["evaluator_independence_receipts"]["exp6300_decision_labels_imported"] is False
    assert artifact["model_identity_controls"]["all_identity_controls_passed"] is True
    assert artifact["model_identity_controls"]["matched_semantic_alignment_preserved"] is True
    assert artifact["false_pass_injection_results"]["all_false_pass_injections_blocked"] is True


def test_scenario_verify_6301_shortcut_controls_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6301-SHORTCUTS: each injected shortcut blocks readiness."""

    reversed_artifact = _run(tmp_path / "reversed", reverse_holdout=True)
    assert reversed_artifact["status"] == "flagged"
    assert reversed_artifact["activation_bus_integrity_ready_score"] == 0.0
    assert "claim_flip_direction_cell_failure" in reversed_artifact["surviving_shortcuts"]
    assert reversed_artifact["failed_cells"]

    model_leak_artifact = _run(tmp_path / "model-leak", model_leak=True)
    assert model_leak_artifact["status"] == "flagged"
    assert "model_identity_shortcut" in model_leak_artifact["surviving_shortcuts"]
    identity = model_leak_artifact["model_identity_controls"]
    assert identity["model_identity_max_accuracy"] > identity["chance_identity_accuracy"]

    token_artifact = _run(tmp_path / "token", token_delta=3)
    assert token_artifact["status"] == "flagged"
    assert "token_count_shortcut" in token_artifact["surviving_shortcuts"]
    assert (
        token_artifact["norm_length_token_and_truncation_controls"][
            "token_count_pair_mismatch_count"
        ]
        > 0
    )

    truncated_artifact = _run(tmp_path / "truncated", truncated=True)
    assert truncated_artifact["status"] == "flagged"
    assert "truncation_shortcut" in truncated_artifact["surviving_shortcuts"]


def test_scenario_verify_6301_independence_validation_and_false_passes(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6301-INDEPENDENCE: consumers and false-pass guards are fresh."""

    artifact = _run(tmp_path)

    swap = artifact["evaluator_swap_receipts"]
    assert swap["fresh_consumer_count"] == 2
    assert swap["all_evaluator_swaps_passed"] is True
    assert artifact["fold_leakage_checks"]["all_fold_leakage_checks_passed"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    missing = deepcopy(artifact)
    del missing["status"]
    assert "missing required field: status" in mod.validate_artifact(missing)

    bad_mutation = deepcopy(artifact)
    bad_mutation["source_mutation_count"] = {"value": 0}
    bad_mutation["reproducibility_checksum"] = mod.reproducibility_checksum(bad_mutation)
    assert "source_mutation_count must be bare integer 0" in mod.validate_artifact(bad_mutation)

    bad_score = deepcopy(artifact)
    bad_score["activation_bus_integrity_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    assert "activation_bus_integrity_ready_score mismatch" in mod.validate_artifact(bad_score)

    bad_model_specs = deepcopy(artifact)
    bad_model_specs["MODEL_SPECS"] = []
    bad_model_specs["reproducibility_checksum"] = mod.reproducibility_checksum(bad_model_specs)
    assert "MODEL_SPECS must preserve mandated GGUF families" in mod.validate_artifact(
        bad_model_specs
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "not terminal"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    assert "honest_verdict lacks terminal prefix" in mod.validate_artifact(bad_verdict)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    failed_codes = deepcopy(artifact)
    failed_codes["test_exit_codes"] = {command: 1 for command in artifact["test_commands"]}
    failed_codes["activation_bus_integrity_ready_score"] = mod.activation_bus_integrity_ready_score(
        failed_codes
    )
    failed_codes["reproducibility_checksum"] = mod.reproducibility_checksum(failed_codes)
    assert failed_codes["activation_bus_integrity_ready_score"] == 0.0


def test_scenario_verify_6301_defensive_negative_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6301-SHORTCUTS: malformed evidence exposes exact blockers."""

    fixture = _fixture(tmp_path / "defensive")
    rows = mod._read_jsonl(fixture["rows"])
    artifact = mod.run(
        root=fixture["root"],
        date="20260811",
        result_path=fixture["root"] / "nowrite.json",
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert not (fixture["root"] / "nowrite.json").exists()

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)
    bad_jsonl = tmp_path / "bad.rows.jsonl"
    bad_jsonl.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(bad_jsonl)
    assert mod._resolve(REPO, Path("not-present.json")) == REPO / "not-present.json"

    ignored = {**rows[0], "model_hf_id": "not/mandated"}
    stale = deepcopy(rows[0])
    stale["row_hash"] = "sha256:bad"
    inputs = mod.reconstruct_bus_inputs([ignored, stale, *rows[1:]])
    assert inputs.row_hash_mismatches == [stale["embedding_cell_id"]]
    with pytest.raises(ValueError, match="condition_embeddings"):
        mod.reconstruct_bus_inputs([{**rows[0], "condition_embeddings": [None]}])
    bad_vector = deepcopy(rows[0])
    bad_vector["condition_embeddings"][0]["embedding"] = [[1.0]]
    with pytest.raises(ValueError, match="one-dimensional"):
        mod.reconstruct_bus_inputs([bad_vector])
    with pytest.raises(ValueError, match="no matched"):
        mod.reconstruct_bus_inputs([ignored])

    latents = np.zeros((len(inputs.item_keys), 2), dtype=float)
    only_left = np.asarray(
        [meta["condition_suffix"] == "a" for meta in inputs.item_meta], dtype=bool
    )
    assert mod._paired_rows(inputs, latents, mod.MANDATED_MODEL_HF_IDS[0], only_left) == []
    assert mod._cosine(np.zeros(2), np.ones(2)) == 0.0
    assert mod._mean_vector([]).shape == (0,)

    bad_fold = {
        "folds": [
            {
                "fold_id": "bad",
                "group_unit": "pair_group_id",
                "train_group_ids": ["g0"],
                "validation_group_ids": ["g0"],
                "holdout_group_ids": [],
            }
        ]
    }
    leakage = mod._fold_leakage_checks(inputs, bad_fold)
    assert leakage["all_fold_leakage_checks_passed"] is False

    manifest = mod._read_json(fixture["manifest"])
    wrong_adapter = mod.Adapter(
        model_id=mod.MANDATED_MODEL_HF_IDS[0],
        encoder_weight=np.eye(3, 2),
        encoder_bias=np.zeros(2),
        metadata={"model_id": "wrong"},
        sha256="sha256:test",
        path="test",
    )
    receipt = mod._row_checkpoint_receipts(
        fixture["root"],
        inputs,
        manifest,
        {"fold_0": {mod.MANDATED_MODEL_HF_IDS[0]: wrong_adapter}},
    )
    assert receipt["checkpoint_metadata_mismatch_count"] >= 1

    shortcut = deepcopy(artifact)
    shortcut["row_and_checkpoint_reconstruction_receipts"][
        "all_rows_and_checkpoints_reconstructed"
    ] = False
    shortcut["evaluator_independence_receipts"]["independence_passed"] = False
    shortcut["fold_leakage_checks"]["all_fold_leakage_checks_passed"] = False
    shortcut["label_permutation_controls"]["all_label_permutation_controls_passed"] = False
    shortcut["duplicate_and_no_information_controls"][
        "all_duplicate_and_no_information_controls_passed"
    ] = False
    shortcut["evaluator_swap_receipts"]["all_evaluator_swaps_passed"] = False
    names = set(mod.surviving_shortcuts(shortcut))
    assert {
        "row_or_checkpoint_reconstruction_failure",
        "evaluator_independence_failure",
        "fold_leakage_or_missing_held_family",
        "label_permutation_survives",
        "duplicate_or_no_information_shortcut",
        "evaluator_swap_disagreement",
    }.issubset(names)
    missing_decision = mod._disaggregated_cell_decisions(
        {
            "claim_flip_sensitivity": {
                "cell_decisions": [{"cell_id": "cell", "cell_passed": True}],
                "held_family_decisions": [],
            },
            "pair_swap_controls": {"cell_decisions": []},
            "norm_length_token_and_truncation_controls": {"cell_decisions": []},
        }
    )
    assert missing_decision["missing_decision_cell_count"] == 1

    blocked = mod.collect_preconditions(
        tmp_path,
        date="20260811",
        result_path=tmp_path / "missing" / "nested" / "out.json",
        upstream={mod.EXP6300_ARTIFACT_RELATIVE_PATH.as_posix(): {"terminal_class": {}}},
        inputs=None,
        manifest=None,
        model_specs=[],
        load_errors=["synthetic_load_error"],
    )
    assert {
        "synthetic_load_error",
        "model_specs_do_not_match_mandated_gguf_families",
        "rows_not_loaded",
        "fold_manifest_not_loaded",
        "exp6300_artifact_not_terminal",
        "output_path_not_writable",
    }.issubset(set(blocked["blocked_reasons"]))
    row_hash_blocked = mod.collect_preconditions(
        tmp_path,
        date="20260811",
        result_path=tmp_path / "out.json",
        upstream={
            mod.EXP6300_ARTIFACT_RELATIVE_PATH.as_posix(): {
                "terminal_class": {"terminal": True}
            }
        },
        inputs=inputs,
        manifest={"schema": "wrong", "folds": []},
        model_specs=[{"hf_id": model_id} for model_id in mod.MANDATED_MODEL_HF_IDS],
        load_errors=[],
    )
    assert "row_hash_reconstruction_failed" in row_hash_blocked["blocked_reasons"]
    assert "fold_manifest_invalid" in row_hash_blocked["blocked_reasons"]
    assert mod._minimal_model_specs([])[0]["cached_activation_only"] is True

    status, verdict = mod._status_and_verdict(
        {
            "activation_bus_integrity_ready_score": 0.0,
            "preconditions_checked": {
                "preconditions_ready": False,
                "blocked_reasons": ["blocked_reason"],
            },
        }
    )
    assert status == "blocked"
    assert verdict.startswith("blocked:")

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"] = {}
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    assert "missing field_provenance entry: status" in mod.validate_artifact(bad_provenance)
    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["activation_bus_integrity_ready_score"] = (
        mod.activation_bus_integrity_ready_score(bad_substrate)
    )
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad_substrate)
    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["activation_bus_integrity_ready_score"] = (
        mod.activation_bus_integrity_ready_score(bad_oracle)
    )
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    assert "verifier_is_oracle mismatch" in mod.validate_artifact(bad_oracle)
    missing_checksum = deepcopy(artifact)
    missing_checksum["reproducibility_checksum"] = ""
    assert "reproducibility_checksum missing" in mod.validate_artifact(missing_checksum)

    with pytest.raises(ValueError, match="requires rows"):
        mod.run(
            root=tmp_path / "absent",
            date="20260811",
            result_path=tmp_path / "absent.json",
            test_commands=TEST_COMMANDS,
            test_exit_codes=TEST_EXIT_CODES,
            write=False,
        )
