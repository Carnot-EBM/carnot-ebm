"""Tests for Exp5853 paired embedding integrity audit.

Spec refs: REQ-VERIFY-5853, SCENARIO-VERIFY-5853-PRECONDITIONS,
SCENARIO-VERIFY-5853-CAUSAL, SCENARIO-VERIFY-5853-SWAPS,
SCENARIO-VERIFY-5853-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5853_paired_embedding_integrity_audit as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5853_paired_embedding_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5853_paired_embedding_integrity_audit.py "
    "-m pytest tests/python/test_experiment_5853_paired_embedding_integrity_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5853_paired_embedding_integrity_audit.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
RUN_COMMAND = ".venv/bin/python -m carnot.experiment_5853_paired_embedding_integrity_audit --write"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5853_paired_embedding_integrity_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    RUN_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _condition(
    row_id: str,
    suffix: str,
    *,
    exact_label: bool,
    constraint_present: bool,
    candidate_hash: str,
    context_hash: str,
) -> dict[str, Any]:
    return {
        "condition_id": f"{row_id}-{suffix}",
        "condition_suffix": suffix,
        "model_input": f"masked input {row_id} {suffix}",
        "model_input_hash": mod.sha256_text(f"masked input {row_id} {suffix}"),
        "token_count": 16,
        "candidate_hash": candidate_hash,
        "context_hash": context_hash,
        "constraint_present": constraint_present,
        "exact_label": exact_label,
    }


def _source_row(
    *,
    axis: str,
    family: str,
    hardness: str,
    surface: str,
    split: str,
    index: int,
) -> dict[str, Any]:
    row_id = f"exp5840-{axis}-{family}-{hardness}-{surface}-{split}-{index:03d}"
    candidate_a = mod.sha256_json({"candidate": row_id, "side": "a"})
    candidate_b = mod.sha256_json({"candidate": row_id, "side": "b"})
    context_a = mod.sha256_json({"context": row_id, "side": "a"})
    context_b = mod.sha256_json({"context": row_id, "side": "b"})
    if axis == "candidate_correctness":
        conditions = [
            _condition(
                row_id,
                "a",
                exact_label=True,
                constraint_present=True,
                candidate_hash=candidate_a,
                context_hash=context_a,
            ),
            _condition(
                row_id,
                "b",
                exact_label=False,
                constraint_present=True,
                candidate_hash=candidate_b,
                context_hash=context_a,
            ),
        ]
        accepted = [candidate_a]
        rejected = [candidate_b]
    else:
        conditions = [
            _condition(
                row_id,
                "a",
                exact_label=False,
                constraint_present=True,
                candidate_hash=candidate_a,
                context_hash=context_a,
            ),
            _condition(
                row_id,
                "b",
                exact_label=True,
                constraint_present=False,
                candidate_hash=candidate_a,
                context_hash=context_b,
            ),
        ]
        accepted = []
        rejected = [candidate_a]
    proof = {
        "one_minimal_violation": True,
        "edit_distance": 1,
        "correct_assignment_hash": candidate_a,
        "violation_assignment_hash": candidate_b
        if axis == "candidate_correctness"
        else candidate_a,
        "correct_accepts_under_present_constraint": True,
        "violation_accepts_under_present_constraint": False,
        "validator_versions": [
            mod.PRIMARY_VALIDATOR_VERSION,
            mod.INDEPENDENT_VALIDATOR_VERSION,
        ],
    }
    exact_receipt = {
        "row_id": row_id,
        "primary_validator_version": mod.PRIMARY_VALIDATOR_VERSION,
        "independent_validator_version": mod.INDEPENDENT_VALIDATOR_VERSION,
        "validators_agree": True,
        "accepted_assignment_hashes": accepted,
        "rejected_assignment_hashes": rejected,
        "minimal_edit_distance": 1,
        "minimal_violation_proof": proof,
    }
    ablation_receipt = {
        "row_id": row_id,
        "axis": axis,
        "candidate_fixed": axis == "constraint_ablation",
        "context_changed": axis == "constraint_ablation",
        "only_target_constraint_changed": True,
        "base_domain_accepts_fixed_candidate": True,
        "present_label": conditions[0]["exact_label"],
        "ablated_label": conditions[1]["exact_label"] if axis == "constraint_ablation" else None,
    }
    row: dict[str, Any] = {
        "schema": mod.SOURCE_ROW_SCHEMA,
        "row_id": row_id,
        "pair_id": row_id,
        "pair_group_id": f"group-{family}-{index:03d}",
        "bootstrap_unit_id": mod.sha256_json({"bootstrap": family, "index": index}),
        "split": split,
        "axis": axis,
        "family": family,
        "change": "addition",
        "surface_kind": surface,
        "solver_effort_bin": hardness,
        "conditions": conditions,
        "exact_receipt": exact_receipt,
        "ablation_receipt": ablation_receipt,
        "surface_receipt": {"proof_preserving": True, "surface_kind": surface},
        "source_provenance": {"exp5826_row_id": row_id, "exp5826_row_hash": "source"},
        "row_hash": "",
    }
    row["row_hash"] = mod.source_row_hash(row)
    return row


def _embedding_row(
    source: dict[str, Any],
    *,
    source_order: int,
    model_hf_id: str,
    model_index: int,
    width: int,
    reverse_direction: bool = False,
    leak_identity: bool = False,
    token_delta: int = 0,
) -> dict[str, Any]:
    sign = -1.0 if reverse_direction else 1.0
    scale = 1.0 + model_index * 0.1 + (0.4 if source["axis"] == "constraint_ablation" else 0.0)
    diff = [round(sign * scale, 6)] + [round(sign * 0.05 * (i + 1), 6) for i in range(width - 1)]
    left_embedding = [round(-value / 2.0, 6) for value in diff]
    right_embedding = [round(value / 2.0, 6) for value in diff]
    conditions = []
    for index, condition in enumerate(source["conditions"]):
        embedding = left_embedding if index == 0 else right_embedding
        token_count = 16 + (token_delta if index == 1 else 0)
        conditions.append(
            {
                "condition_id": condition["condition_id"],
                "condition_suffix": condition["condition_suffix"],
                "source_model_input_hash": condition["model_input_hash"],
                "embedding_input_hash": mod.sha256_text(condition["model_input"]),
                "base_token_count": token_count,
                "token_count": token_count,
                "target_pair_token_count": 16,
                "neutral_padding_tokens_added": [],
                "neutral_padding_token_count": 0,
                "token_parity_ok": token_delta == 0,
                "truncated": False,
                "embedding": embedding,
                "embedding_shape": [len(embedding)],
                "embedding_sha256": mod.sha256_json(embedding),
            }
        )
    feature_view = {
        "condition_features": [
            {
                "condition_id": mod.sha256_json({"condition_id": condition["condition_id"]}),
                "embedding": condition["embedding"],
                "embedding_shape": condition["embedding_shape"],
                "embedding_sha256": condition["embedding_sha256"],
                "token_count": condition["token_count"],
                "truncated": False,
            }
            for condition in conditions
        ],
        "paired_difference": diff,
        "paired_difference_sha256": mod.sha256_json(diff),
        "difference_orientation": "condition_b_minus_a",
        "preprocessing": "test",
    }
    if leak_identity:
        feature_view["model_hf_id"] = model_hf_id
    row = {
        "schema": mod.EMBEDDING_ROW_SCHEMA,
        "source_row_order": source_order,
        "source_row_id": source["row_id"],
        "source_row_hash": source["row_hash"],
        "source_pair_id": source["pair_id"],
        "pair_group_id": source["pair_group_id"],
        "split": source["split"],
        "axis": source["axis"],
        "family": source["family"],
        "change": source["change"],
        "surface_kind": source["surface_kind"],
        "solver_effort_bin": source["solver_effort_bin"],
        "model_hf_id": model_hf_id,
        "model_family": mod.model_family(model_hf_id),
        "model_file_sha256": mod.sha256_json({"model": model_hf_id}),
        "model_local_path_hash": mod.sha256_json({"path": model_hf_id}),
        "embedding_cell_id": f"{source['row_id']}|{model_hf_id}",
        "condition_embeddings": conditions,
        "paired_difference": diff,
        "paired_difference_sha256": mod.sha256_json(diff),
        "oracle_label_receipt": {
            "source": "exp5840_exact_labels",
            "labels_by_condition_id": {
                condition["condition_id"]: condition["exact_label"]
                for condition in source["conditions"]
            },
            "verifier_is_oracle": True,
        },
        "feature_consumer_view": feature_view,
        "loader_receipt_hash": mod.sha256_json({"loader": model_hf_id}),
        "row_hash": "",
    }
    row["row_hash"] = mod.embedding_row_hash(row)
    return row


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(mod.canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    reverse_one_direction: bool = False,
    distinct_model_widths: bool = False,
    leak_identity: bool = False,
    token_delta: int = 0,
    validator_disagreement: bool = False,
) -> dict[str, Path | list[dict[str, Any]]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    index = 0
    for family in mod.REQUIRED_CONSTRAINT_FAMILIES:
        for axis in mod.REQUIRED_CAUSAL_AXES:
            for hardness in mod.REQUIRED_HARDNESS_BINS:
                for surface in mod.REQUIRED_PROOF_PRESERVING_SURFACES:
                    for split in ("train", "science"):
                        row = _source_row(
                            axis=axis,
                            family=family,
                            hardness=hardness,
                            surface=surface,
                            split=split,
                            index=index,
                        )
                        if validator_disagreement and index == 0:
                            row["exact_receipt"]["independent_validator_version"] = row[
                                "exact_receipt"
                            ]["primary_validator_version"]
                            row["row_hash"] = mod.source_row_hash(row)
                        rows.append(row)
                        index += 1
    embedding_rows: list[dict[str, Any]] = []
    widths = [2, 2, 2] if not distinct_model_widths else [2, 3, 4]
    for source_order, source in enumerate(rows):
        for model_index, model_hf_id in enumerate(mod.MANDATED_MODEL_HF_IDS):
            embedding_rows.append(
                _embedding_row(
                    source,
                    source_order=source_order,
                    model_hf_id=model_hf_id,
                    model_index=model_index,
                    width=widths[model_index],
                    reverse_direction=reverse_one_direction
                    and source["axis"] == "candidate_correctness"
                    and source["split"] == "science"
                    and model_index == 0,
                    leak_identity=leak_identity,
                    token_delta=token_delta,
                )
            )
    source_path = tmp_path / "exp5840.rows.jsonl"
    embed_path = tmp_path / "exp5852.rows.jsonl"
    _write_jsonl(source_path, rows)
    _write_jsonl(embed_path, embedding_rows)
    source_text = source_path.read_text(encoding="utf-8")
    embed_text = embed_path.read_text(encoding="utf-8")
    source_artifact = {
        "status": "complete",
        "counterfactual_fixture_ready_score": 1.0,
        "split_definition_and_hashes": {
            "label_blind": True,
            "row_split_hashes": {
                split: mod.sha256_json([row["row_id"] for row in rows if row["split"] == split])
                for split in ("train", "science")
            },
        },
        "exact_label_and_minimality_receipts": {
            "validator_versions": [
                mod.PRIMARY_VALIDATOR_VERSION,
                mod.INDEPENDENT_VALIDATOR_VERSION,
            ]
        },
        "row_file_receipt": {
            "path": "results/experiment_5840_exact_counterfactual_embedding_fixture.rows.jsonl",
            "row_count": len(rows),
            "sha256": mod.sha256_text(source_text),
            "row_hashes": {row["row_id"]: row["row_hash"] for row in rows},
            "row_hash_root": mod.sha256_json({row["row_id"]: row["row_hash"] for row in rows}),
            "atomic_write": True,
        },
    }
    embed_artifact = {
        "status": "complete",
        "paired_embedding_corpus_ready_score": 1.0,
        "inference_substrate": "live_llm_embedding_extraction",
        "verifier_is_oracle": True,
        "models_used": list(mod.MANDATED_MODEL_HF_IDS),
        "model_specs": [{"hf_id": model} for model in mod.MANDATED_MODEL_HF_IDS],
        "row_file_receipt": {
            "path": "results/experiment_5852_three_family_paired_embeddings.rows.jsonl",
            "row_count": len(embedding_rows),
            "sha256": mod.sha256_text(embed_text),
            "row_hashes": {row["embedding_cell_id"]: row["row_hash"] for row in embedding_rows},
            "row_hash_root": mod.sha256_json(
                {row["embedding_cell_id"]: row["row_hash"] for row in embedding_rows}
            ),
            "atomic_write": True,
        },
    }
    source_artifact_path = tmp_path / "exp5840.json"
    embed_artifact_path = tmp_path / "exp5852.json"
    _write_json(source_artifact_path, source_artifact)
    _write_json(embed_artifact_path, embed_artifact)
    return {
        "source_rows": rows,
        "embedding_rows": embedding_rows,
        "source_rows_path": source_path,
        "embedding_rows_path": embed_path,
        "source_artifact_path": source_artifact_path,
        "embedding_artifact_path": embed_artifact_path,
    }


def _run_fixture(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    fixture = _fixture(tmp_path, **kwargs)
    return mod.run(
        root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        exp5852_artifact_path=fixture["embedding_artifact_path"],
        exp5852_rows_path=fixture["embedding_rows_path"],
        exp5840_artifact_path=fixture["source_artifact_path"],
        exp5840_rows_path=fixture["source_rows_path"],
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        memory_probe=lambda: {"available_mb": 65536, "required_mb": 16384, "ok": True},
        disk_probe=lambda root: {"available_mb": 65536, "required_mb": 8192, "ok": True},
        write=True,
    )


def test_req_verify_5853_spec_declares_integrity_contract() -> None:
    """REQ-VERIFY-5853: OpenSpec names fields, principles, and scenarios."""

    spec = VERIFY_SPEC.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5853") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5853",
        "SCENARIO-VERIFY-5853-PRECONDITIONS",
        "SCENARIO-VERIFY-5853-CAUSAL",
        "SCENARIO-VERIFY-5853-SWAPS",
        "SCENARIO-VERIFY-5853-CONTROLS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`paired_embedding_integrity_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5853_preconditions_reconstruct_rows_from_ids(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5853-PRECONDITIONS: row joins are rebuilt from IDs."""

    artifact = _run_fixture(tmp_path)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text())

    assert written == artifact
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("ready:")
    assert artifact["paired_embedding_integrity_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["paired_embedding_integrity_ready_score"], float)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    reconstruction = artifact["upstream_hashes_and_row_reconstruction"]["row_reconstruction"]
    assert reconstruction["reconstructed_from_immutable_row_ids"] is True
    assert reconstruction["aggregate_counts_trusted"] is False
    assert reconstruction["expected_model_row_cell_count"] == 288
    assert reconstruction["all_rows_reconstructed"] is True
    assert reconstruction["duplicate_embedding_cell_ids"] == []
    assert reconstruction["missing_embedding_cell_ids"] == []


def test_scenario_verify_5853_causal_and_swap_controls_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5853-CAUSAL/SWAPS: bad directions or validators block readiness."""

    direction_artifact = _run_fixture(tmp_path / "direction", reverse_one_direction=True)
    assert direction_artifact["status"] == "disqualified"
    assert direction_artifact["paired_embedding_integrity_ready_score"] == 0.0
    assert direction_artifact["honest_verdict"].startswith("disqualified:")
    assert "claim_flip_direction_cell_failure" in direction_artifact["surviving_shortcuts"]
    assert direction_artifact["claim_flip_sensitivity"]["failed_cell_count"] > 0
    assert mod.validate_artifact(direction_artifact) is True

    evaluator_artifact = _run_fixture(tmp_path / "evaluator", validator_disagreement=True)
    assert evaluator_artifact["status"] == "disqualified"
    assert "evaluator_swap_disagreement" in evaluator_artifact["surviving_shortcuts"]
    assert evaluator_artifact["evaluator_swap_receipts"]["validator_disagreement_count"] == 1
    assert mod.validate_artifact(evaluator_artifact) is True


def test_scenario_verify_5853_controls_detect_identity_and_token_shortcuts(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5853-CONTROLS: identity and length shortcuts survive per cell."""

    identity_artifact = _run_fixture(tmp_path / "identity", distinct_model_widths=True)
    assert identity_artifact["status"] == "disqualified"
    assert "raw_model_dimension_identity_shortcut" in identity_artifact["surviving_shortcuts"]
    identity = identity_artifact["identity_masking_and_prediction_controls"]
    assert identity["raw_dimension_identity_accuracy"] == pytest.approx(1.0)
    assert identity["feature_consumer_identity_leakage_count"] == 0
    assert mod.validate_artifact(identity_artifact) is True

    token_artifact = _run_fixture(tmp_path / "token", token_delta=1)
    assert token_artifact["status"] == "disqualified"
    assert "token_count_pair_shortcut" in token_artifact["surviving_shortcuts"]
    length_norm = token_artifact["length_norm_and_truncation_controls"]
    assert length_norm["token_count_pair_mismatch_count"] > 0
    assert length_norm["all_length_norm_controls_passed"] is False

    leaked_artifact = _run_fixture(tmp_path / "leaked", leak_identity=True)
    assert leaked_artifact["status"] == "disqualified"
    assert "feature_consumer_identity_leakage" in leaked_artifact["surviving_shortcuts"]
    assert (
        leaked_artifact["identity_masking_and_prediction_controls"][
            "feature_consumer_identity_leakage_count"
        ]
        > 0
    )


def test_scenario_verify_5853_blocked_gate_and_validation_defensive(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5853-PRECONDITIONS/CONTROLS: bad gates cannot look ready."""

    fixture = _fixture(tmp_path)
    artifact_path = fixture["embedding_artifact_path"]
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    artifact["paired_embedding_corpus_ready_score"] = 0.0
    _write_json(Path(artifact_path), artifact)
    blocked = mod.run(
        root=REPO,
        result_path=tmp_path / "blocked.json",
        exp5852_artifact_path=artifact_path,
        exp5852_rows_path=fixture["embedding_rows_path"],
        exp5840_artifact_path=fixture["source_artifact_path"],
        exp5840_rows_path=fixture["source_rows_path"],
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        memory_probe=lambda: {"available_mb": 65536, "required_mb": 16384, "ok": True},
        disk_probe=lambda root: {"available_mb": 65536, "required_mb": 8192, "ok": True},
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "exp5852_corpus_not_ready" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    missing_field = deepcopy(blocked)
    del missing_field["status"]
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(missing_field)
    bad_score = deepcopy(blocked)
    bad_score["paired_embedding_integrity_ready_score"] = 1.0
    with pytest.raises(ValueError, match="paired_embedding_integrity_ready_score"):
        mod.validate_artifact(bad_score)
    bad_checksum = deepcopy(blocked)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)
    bad_substrate = deepcopy(blocked)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["paired_embedding_integrity_ready_score"] = (
        mod.paired_embedding_integrity_ready_score(bad_substrate)
    )
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)


def test_scenario_verify_5853_defensive_branch_coverage(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5853-CONTROLS: malformed controls expose exact failure modes."""

    fixture = _fixture(tmp_path / "branches")
    source_rows = deepcopy(fixture["source_rows"])
    embedding_rows = deepcopy(fixture["embedding_rows"])

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(bad_json)
    bad_jsonl = tmp_path / "bad.rows.jsonl"
    bad_jsonl.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(bad_jsonl)
    assert mod._mean_vector([]) == []

    tree = tmp_path / "tree"
    pycache = tree / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "ignored.py").write_text("ignored = True\n", encoding="utf-8")
    (tree / "kept.py").write_text("kept = True\n", encoding="utf-8")
    assert mod._tree_hash(tree).startswith("sha256:")

    malformed_source = deepcopy(source_rows[0])
    malformed_source["conditions"] = [malformed_source["conditions"][0]]
    assert mod._source_labels(malformed_source) == (False, False)
    malformed_axis = deepcopy(source_rows[1])
    malformed_axis["axis"] = "unknown_axis"
    malformed_axis["row_hash"] = mod.source_row_hash(malformed_axis)
    grounded = mod.grounded_vs_true_cross([malformed_source, malformed_axis])
    assert grounded["grounding_truth_disagreement_count"] == 2

    replay_bad_candidate = deepcopy(source_rows[0])
    replay_bad_candidate["conditions"][0]["exact_label"] = False
    replay_bad_ablation = next(
        row for row in deepcopy(source_rows) if row["axis"] == "constraint_ablation"
    )
    replay_bad_ablation["conditions"][1]["exact_label"] = False
    receipt = mod.evaluator_swap_receipts([replay_bad_candidate, replay_bad_ablation], [])
    assert receipt["exact_label_replay_disagreement_count"] == 2
    private_code = deepcopy(embedding_rows[0])
    private_code["feature_consumer_view"]["note"] = mod.PRIMARY_VALIDATOR_VERSION
    receipt = mod.evaluator_swap_receipts([], [private_code])
    assert receipt["feature_consumer_reinitialization"]["private_code_marker_count"] == 1

    token_leak = deepcopy(embedding_rows[0])
    token_leak["feature_consumer_view"]["note"] = "finite_domain_csp"
    assert mod._feature_identity_leaks([token_leak]) == [token_leak["embedding_cell_id"]]

    ragged = deepcopy(embedding_rows[0])
    ragged["embedding_cell_id"] = "bad"
    ragged["source_row_hash"] = "sha256:stale"
    ragged["row_hash"] = "sha256:bad"
    ragged["source_row_order"] = 999
    reconstruction = mod.row_reconstruction(source_rows[:1], [ragged])
    assert reconstruction["embedding_cell_id_mismatch_count"] == 1
    assert reconstruction["stale_source_hash_count"] == 1
    assert reconstruction["embedding_row_hash_mismatch_count"] == 1
    assert reconstruction["source_row_order_mismatch_count"] == 1

    norm_row = deepcopy(embedding_rows[0])
    norm_row["condition_embeddings"][0]["embedding"] = [5.0, 0.0]
    norm_row["condition_embeddings"][1]["embedding"] = [0.0, 0.0]
    trunc_row = deepcopy(embedding_rows[1])
    trunc_row["condition_embeddings"][0]["truncated"] = True
    malformed_embedding = deepcopy(embedding_rows[2])
    malformed_embedding["condition_embeddings"] = [malformed_embedding["condition_embeddings"][0]]
    length = mod.length_norm_and_truncation_controls([norm_row, trunc_row, malformed_embedding])
    assert length["truncation_count"] == 1
    assert length["norm_only_max_label_accuracy"] == pytest.approx(1.0)
    assert length["all_length_norm_controls_passed"] is False

    missing = mod.disaggregated_cell_decisions(
        source_rows=source_rows[:1],
        claim_flip={"cell_decisions": []},
        ablation={"cell_decisions": []},
        length_norm={"cell_decisions": []},
    )
    assert missing["missing_decision_cell_count"] == 3

    shortcut_artifact = {
        "claim_flip_sensitivity": {"all_cells_passed": True},
        "constraint_ablation_sensitivity": {"all_cells_passed": False},
        "evaluator_swap_receipts": {"all_evaluator_swaps_passed": True},
        "grounded_vs_true_cross": {"grounding_and_truth_distinguished": False},
        "label_and_pair_permutation_controls": {"all_label_and_pair_controls_passed": True},
        "identity_masking_and_prediction_controls": {
            "raw_dimension_identity_accuracy": 0.3,
            "chance_identity_accuracy": 0.3,
            "feature_consumer_identity_leakage_count": 0,
        },
        "length_norm_and_truncation_controls": {
            "token_count_pair_mismatch_count": 0,
            "truncation_count": 1,
            "all_length_norm_controls_passed": True,
        },
        "perturbation_duplicate_and_no_information_controls": {
            "all_perturbation_duplicate_no_information_controls_passed": False
        },
        "disaggregated_cell_decisions": {"all_disaggregated_cells_passed": False},
    }
    assert {
        "constraint_ablation_direction_cell_failure",
        "grounded_vs_true_cross_failure",
        "truncation_shortcut",
        "perturbation_duplicate_or_no_information_control_failure",
        "disaggregated_cell_failure",
    }.issubset(set(mod._surviving_shortcuts(shortcut_artifact)))

    bad_artifact_paths = tmp_path / "missing"
    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / "missing.json",
        exp5852_artifact_path=bad_artifact_paths / "missing-exp5852.json",
        exp5852_rows_path=bad_artifact_paths / "missing-exp5852.rows.jsonl",
        exp5840_artifact_path=bad_artifact_paths / "missing-exp5840.json",
        exp5840_rows_path=bad_artifact_paths / "missing-exp5840.rows.jsonl",
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        memory_probe=lambda: {"available_mb": 1, "required_mb": 2, "ok": False},
        disk_probe=lambda root: {"available_mb": 1, "required_mb": 2, "ok": False},
        write=False,
    )
    assert {
        "exp5852_artifact_load_failed:FileNotFoundError",
        "exp5840_artifact_load_failed:FileNotFoundError",
        "exp5840_rows_load_failed:FileNotFoundError",
        "exp5852_rows_load_failed:FileNotFoundError",
        "exp5852_mandated_model_set_mismatch",
        "exp5840_fixture_not_ready",
        "exp5852_verifier_not_oracle",
        "exp5852_wrong_inference_substrate",
        "exp5840_rows_receipt_replay_failed",
        "exp5852_rows_receipt_replay_failed",
        "insufficient_free_ram",
        "insufficient_free_disk",
    }.issubset(set(blocked["preconditions_checked"]["blocked_reasons"]))

    failed_preconditions = mod._collect_preconditions(
        root=tmp_path,
        result_path=tmp_path / "absent" / "nested" / "artifact.json",
        exp5852_artifact={
            "verifier_is_oracle": True,
            "inference_substrate": "live_llm_embedding_extraction",
        },
        exp5840_artifact={},
        upstream={
            "exp5852_gate": {
                "paired_embedding_corpus_ready_score": 1.0,
                "models_used": list(mod.MANDATED_MODEL_HF_IDS),
            },
            "exp5840_gate": {"counterfactual_fixture_ready_score": 1.0},
            "source_row_file_receipt_replay": {"all_receipt_checks_passed": True},
            "embedding_row_file_receipt_replay": {"all_receipt_checks_passed": True},
            "row_reconstruction": {"all_rows_reconstructed": False},
        },
        load_errors=[],
        memory_probe=lambda: {"available_mb": 65536, "required_mb": 16384, "ok": True},
        disk_probe=lambda root: {"available_mb": 65536, "required_mb": 8192, "ok": True},
    )
    assert {
        "row_reconstruction_failed",
        "output_path_not_writable",
    }.issubset(set(failed_preconditions["blocked_reasons"]))

    ready_tamper = _run_fixture(tmp_path / "ready-tamper")
    ready_tamper["status"] = "blocked"
    ready_tamper["reproducibility_checksum"] = mod.reproducibility_checksum(ready_tamper)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(ready_tamper)
    bad_provenance = _run_fixture(tmp_path / "bad-provenance")
    bad_provenance["field_provenance"] = {}
    bad_provenance["paired_embedding_integrity_ready_score"] = (
        mod.paired_embedding_integrity_ready_score(bad_provenance)
    )
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)
    bad_oracle = _run_fixture(tmp_path / "bad-oracle")
    bad_oracle["verifier_is_oracle"] = False
    bad_oracle["paired_embedding_integrity_ready_score"] = (
        mod.paired_embedding_integrity_ready_score(bad_oracle)
    )
    bad_oracle["reproducibility_checksum"] = mod.reproducibility_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)
    ready_tamper = _run_fixture(tmp_path / "ready-verdict")
    ready_tamper["honest_verdict"] = "blocked: wrong"
    ready_tamper["reproducibility_checksum"] = mod.reproducibility_checksum(ready_tamper)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(ready_tamper)
    blocked_bad_verdict = deepcopy(blocked)
    blocked_bad_verdict["honest_verdict"] = "ready: wrong"
    blocked_bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        blocked_bad_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked_bad_verdict)
    disqualified = _run_fixture(tmp_path / "disqualified", distinct_model_widths=True)
    disqualified["honest_verdict"] = "ready: wrong"
    disqualified["reproducibility_checksum"] = mod.reproducibility_checksum(disqualified)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(disqualified)
    weird_status = deepcopy(blocked)
    weird_status["status"] = "partial"
    weird_status["honest_verdict"] = "partial: wrong"
    weird_status["reproducibility_checksum"] = mod.reproducibility_checksum(weird_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(weird_status)
