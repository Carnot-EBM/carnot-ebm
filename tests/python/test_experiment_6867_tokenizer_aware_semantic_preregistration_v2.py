"""Tests for REQ-INFERENCE-6867 and SCENARIO-INFERENCE-6867-*.

The tests keep all artifacts in memory or under ``tmp_path``. They never write
the tracked research record.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6867_tokenizer_aware_semantic_preregistration_v2 as exp


def _group(index: int, family: str = "exact_energy") -> dict:
    group_id = f"group-{family}-{index:02d}"
    return {
        "semantic_identity": group_id,
        "group_id": f"source-{group_id}",
        "family": family,
        "program": {"family": family, "index": index},
        "candidates": [
            {
                "candidate_id": f"valid-{group_id}",
                "semantic_identity": f"valid-semantic-{group_id}",
                "expected_label": True,
                "content": {"answer": "valid", "index": index},
            },
            {
                "candidate_id": f"invalid-{group_id}",
                "semantic_identity": f"invalid-semantic-{group_id}",
                "expected_label": False,
                "content": {"answer": "invalid", "index": index},
            },
        ],
    }


def _cell() -> dict:
    candidate_hashes = {
        "slot_0": exp.sha256_text("candidate-valid"),
        "slot_1": exp.sha256_text("candidate-wrong"),
    }
    return {
        "candidate_ids": ["valid-id", "invalid-id"],
        "candidate_labels_by_slot": [True, False],
        "candidate_token_ids": {"slot_0": [1, 2], "slot_1": [3, 4]},
        "candidate_token_counts": {"slot_0": 2, "slot_1": 2},
        "candidate_character_counts": {"slot_0": 15, "slot_1": 15},
        "paired_prompt_token_ids": {"base": [10, 11], "label_swap": [10, 11]},
        "paired_prompt_token_counts": {"base": 2, "label_swap": 2},
        "label_position_contract": {"base": [0, 1], "swapped": [1, 0]},
        "presentation_order": {"base": [0, 1], "label_swap": [1, 0]},
        "normalization": "NFC",
        "surface_control": {
            "raw_prompt_template_sha256": exp.sha256_text(exp.RAW_PROMPT_TEMPLATE),
            "candidate_sequence_template_sha256": exp.sha256_text(exp.CANDIDATE_SEQUENCE_TEMPLATE),
        },
        "candidate_sequence_sha256": candidate_hashes,
        "sequence_identity": exp.sequence_identity(
            "model", "group", exp.sha256_text("prompt"), candidate_hashes
        ),
        "model_hf_id": "model",
        "semantic_group_identity": "group",
        "prompt_sequence_sha256": exp.sha256_text("prompt"),
    }


def test_req_inference_6867_spec_declares_complete_contract() -> None:
    """REQ-INFERENCE-6867 exists before implementation code."""

    text = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-INFERENCE-6867:", 1)[1]
    for marker in (
        "SCENARIO-INFERENCE-6867-BANK-IDENTITY",
        "SCENARIO-INFERENCE-6867-TOKENIZER-IDENTITY",
        "SCENARIO-INFERENCE-6867-NUISANCE-CONTROLS",
        "SCENARIO-INFERENCE-6867-GROUP-SPLIT",
        "SCENARIO-INFERENCE-6867-HELD-SEAL",
        "SCENARIO-INFERENCE-6867-SAMPLE-FLOOR",
        "SCENARIO-INFERENCE-6867-SCORE-FREEZE",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_6867_bank_mutation_and_group_drift_fail_closed() -> None:
    """SCENARIO-INFERENCE-6867-BANK-IDENTITY checks bytes and identities."""

    groups = [_group(index) for index in range(100)]
    raw = b"frozen-bank"
    expected_ids = [row["semantic_identity"] for row in groups]

    assert (
        exp.bank_identity_errors(
            raw, groups, frozen_hash=exp.sha256_bytes(raw), frozen_group_ids=expected_ids
        )
        == []
    )
    assert exp.bank_identity_errors(
        raw + b"!", groups, frozen_hash=exp.sha256_bytes(raw), frozen_group_ids=expected_ids
    ) == ["frozen_contrast_bank_hash_mismatch"]
    changed = deepcopy(groups)
    changed[0]["semantic_identity"] = "mutated"
    assert "semantic_group_identity_mismatch" in exp.bank_identity_errors(
        raw, changed, frozen_hash=exp.sha256_bytes(raw), frozen_group_ids=expected_ids
    )
    short = groups[:99] + [groups[0]]
    assert exp.bank_identity_errors(
        raw, short, frozen_hash=exp.sha256_bytes(raw), frozen_group_ids=expected_ids
    ) == [
        "semantic_group_identity_collision",
        "semantic_group_identity_mismatch",
    ]
    assert "semantic_group_count_mismatch" in exp.bank_identity_errors(
        raw, groups[:99], frozen_hash=exp.sha256_bytes(raw), frozen_group_ids=expected_ids
    )


def test_scenario_6867_tokenizer_substitution_is_typed() -> None:
    """SCENARIO-INFERENCE-6867-TOKENIZER-IDENTITY rejects substitutions."""

    frozen = {
        "path": "/models/exact.gguf",
        "sha256": "sha256:exact",
        "size_bytes": 10,
        "quantization": "Q4_K_M",
        "snapshot_identity": "snapshot",
        "canonical_tokenizer_payload_sha256": "sha256:tokenizer",
    }
    assert exp.tokenizer_binding_errors(dict(frozen), frozen) == []
    fields = {
        "path": "model_path_substitution",
        "sha256": "model_hash_drift",
        "size_bytes": "model_size_drift",
        "quantization": "quantization_drift",
        "snapshot_identity": "snapshot_identity_drift",
        "canonical_tokenizer_payload_sha256": "canonical_tokenizer_payload_drift",
    }
    for field, reason in fields.items():
        changed = dict(frozen)
        changed[field] = "different"
        assert reason in exp.tokenizer_binding_errors(changed, frozen)


def test_scenario_6867_unequal_candidate_counts_are_rejected() -> None:
    """SCENARIO-INFERENCE-6867-NUISANCE-CONTROLS requires two candidates."""

    cell = _cell()
    cell["candidate_ids"] = ["only-one"]
    cell["candidate_token_ids"] = {"slot_0": [1, 2]}
    assert "candidate_count_mismatch" in exp.cell_nuisance_errors(cell)


def test_scenario_6867_unequal_tokens_and_prompt_counts_are_rejected() -> None:
    """SCENARIO-INFERENCE-6867-NUISANCE-CONTROLS freezes lengths."""

    cell = _cell()
    cell["candidate_token_counts"]["slot_1"] = 3
    cell["paired_prompt_token_counts"]["label_swap"] = 3
    assert exp.cell_nuisance_errors(cell) == [
        "unequal_candidate_token_counts",
        "paired_prompt_count_mismatch",
    ]


def test_scenario_6867_all_surface_controls_are_checked() -> None:
    """SCENARIO-INFERENCE-6867-NUISANCE-CONTROLS checks every frozen control."""

    cell = _cell()
    assert exp.cell_nuisance_errors(cell) == []
    mutations = {
        "candidate_labels_by_slot": ([False, True], "candidate_label_order_mismatch"),
        "candidate_character_counts": (
            {"slot_0": 15, "slot_1": 16},
            "unequal_candidate_character_counts",
        ),
        "paired_prompt_token_ids": (
            {"base": [10], "label_swap": [11]},
            "paired_prompt_token_ids_mismatch",
        ),
        "label_position_contract": (
            {"base": [1, 0], "swapped": [0, 1]},
            "label_position_contract_failed",
        ),
        "presentation_order": (
            {"base": [1, 0], "label_swap": [0, 1]},
            "presentation_order_failed",
        ),
        "normalization": ("NFD", "normalization_control_failed"),
        "surface_control": ({}, "surface_template_control_failed"),
        "candidate_ids": (["same", "same"], "candidate_identifier_collision"),
    }
    for field, (value, reason) in mutations.items():
        changed = deepcopy(cell)
        changed[field] = value
        assert reason in exp.cell_nuisance_errors(changed)
    changed = deepcopy(cell)
    changed["sequence_identity"] = "wrong"
    assert "sequence_identity_mismatch" in exp.cell_nuisance_errors(changed)


def test_scenario_6867_split_is_seeded_family_stratified_and_disjoint() -> None:
    """SCENARIO-INFERENCE-6867-GROUP-SPLIT assigns whole semantic groups."""

    families = ["exact_energy", "satisfaction_predicate", "memory_guard"]
    groups = [_group(index, family) for family in families for index in range(20)]
    first = exp.freeze_group_splits(groups, random_seed=6867)
    second = exp.freeze_group_splits(reversed(groups), random_seed=6867)

    assert first == second
    assert set(first["calibration_group_ids"]).isdisjoint(first["held_group_ids"])
    assert len(first["calibration_group_ids"]) == 30
    assert len(first["held_group_ids"]) == 30
    assert first["split_checksum"].startswith("sha256:")
    assert all(
        counts == {"calibration": 10, "held": 10} for counts in first["family_counts"].values()
    )


def test_scenario_6867_split_overlap_and_row_leakage_are_detected() -> None:
    """SCENARIO-INFERENCE-6867-GROUP-SPLIT detects both leakage forms."""

    rows = [
        {"semantic_group_identity": "a", "split": "calibration"},
        {"semantic_group_identity": "a", "split": "held"},
    ]
    errors = exp.split_errors(["a"], ["a"], rows)
    assert errors == ["split_overlap:a", "group_split_leakage:a"]


def test_scenario_6867_non_disjoint_semantic_families_are_rejected() -> None:
    """SCENARIO-INFERENCE-6867-GROUP-SPLIT rejects one identity in two families."""

    groups = [_group(0, "family_a"), _group(0, "family_b")]
    groups[1]["semantic_identity"] = groups[0]["semantic_identity"]
    assert exp.semantic_family_errors(groups) == [
        f"semantic_family_overlap:{groups[0]['semantic_identity']}"
    ]


def test_scenario_6867_held_labels_are_sealed_and_access_is_denied() -> None:
    """SCENARIO-INFERENCE-6867-HELD-SEAL keeps labels from calibration."""

    held = _group(1)
    sealed = exp.seal_held_groups([held], secret_salt=b"not-published")
    serialized = json.dumps(sealed)
    assert "expected_label" not in serialized
    assert "not-published" not in serialized
    assert sealed[0]["label_commitment"].startswith("sha256:")

    loader = exp.CalibrationLabelLoader({"cal": (True, False)}, {held["semantic_identity"]})
    assert loader.load("cal") == (True, False)
    with pytest.raises(exp.HeldLabelAccessError, match="held label access denied"):
        loader.load(held["semantic_identity"])
    assert loader.held_label_access_count == 0
    assert loader.held_label_access_attempt_count == 1


def test_scenario_6867_low_sample_size_keeps_readiness_zero() -> None:
    """SCENARIO-INFERENCE-6867-SAMPLE-FLOOR requires 20 groups per split."""

    rows = [
        {
            "model_hf_id": model,
            "split": split,
            "accepted_group_count": 20,
            "minimum_group_count": 20,
            "floor_passed": True,
        }
        for model in exp.MODEL_SPECS
        for split in ("calibration", "held")
    ]
    assert exp.readiness_score(rows, all_other_checks_pass=True) == 1
    rows[0]["accepted_group_count"] = 19
    rows[0]["floor_passed"] = False
    assert exp.readiness_score(rows, all_other_checks_pass=True) == 0
    assert exp.readiness_score([], all_other_checks_pass=True) == 0


def test_scenario_6867_prior_token_scores_block_only_scored_identities() -> None:
    """SCENARIO-INFERENCE-6867-SCORE-FREEZE distinguishes IDs from scores."""

    target = "score-cell"
    unscored = {"rows": [{"score_identity": target, "token_likelihood_call_count": 0}]}
    scored = {"rows": [{"score_identity": target, "candidate_log_likelihoods": [-1.0, -2.0]}]}
    assert exp.find_scored_split_identities([unscored], [target]) == []
    assert exp.find_scored_split_identities([unscored, scored], [target]) == [target]
    assert exp.score_identity("model", "group", "split").startswith("sha256:")


def test_req_inference_6867_statistics_freeze_all_declared_methods() -> None:
    """REQ-INFERENCE-6867 freezes scoring methods before likelihoods exist."""

    manifest = exp.preregistered_statistics()
    assert manifest["frozen_before_token_likelihood"] is True
    assert manifest["within_group_paired_contrast"]
    assert set(manifest["nuisance_difference_in_differences"]) == set(exp.NUISANCE_CONTROLS)
    assert manifest["bootstrap_interval"]["cluster_unit"] == "semantic_group_identity"
    assert manifest["missing_cell_rule"]["imputation"] == "none"
    assert manifest["per_family_effect"]
    assert manifest["family_replication_rule"]
    assert manifest["pooled_aggregation"]
    assert manifest["retirement_thresholds"]


def test_req_inference_6867_blocked_artifact_names_exact_failure() -> None:
    """REQ-INFERENCE-6867 emits a complete blocked schema on precondition failure."""

    artifact = exp.blocked_artifact(
        run_date="20260902",
        checks=[exp.gate_check("canonical_tokenizer_binding_ready_score", 1, 0, False)],
        duration_s=0.25,
    )
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == (
        "canonical_tokenizer_binding_ready_score"
    )
    assert artifact["gate_check_summary"]["expected"] == 1
    assert artifact["gate_check_summary"]["observed"] == 0
    assert artifact["token_likelihood_call_count"] == 0
    assert artifact["generated_answer_count"] == 0
    assert exp.validate_artifact(artifact) == []


def test_req_inference_6867_artifact_validation_fails_closed() -> None:
    """REQ-INFERENCE-6867 validates readiness, counters, principles, and seal."""

    artifact = exp.blocked_artifact(
        run_date="20260902",
        checks=[exp.gate_check("source", True, False, False)],
        duration_s=0.1,
    )
    mutations = {
        "inference_substrate": "live_llm_inference",
        "token_likelihood_call_count": 1,
        "generated_answer_count": 1,
        "split_overlap_count": 1,
        "held_label_access_count": 1,
        "verifier_is_oracle": True,
        "verdict_class": "unknown",
        "honest_verdict": "blocked",
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert field in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed.pop("rows")
    assert "missing_field:rows" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["field_principles"].pop("rows")
    assert "field_principle_keys" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["frozen_contrast_bank_hash"] = "sha256:wrong"
    assert "frozen_contrast_bank_hash" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["semantic_contrast_preregistration_v2_ready_score"] = 2
    assert "semantic_contrast_preregistration_v2_ready_score" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["sealed_held_group_manifest"] = [{"nested": [{"expected_label": True}]}]
    assert "sealed_held_group_manifest" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"] = [{"split": "held", "candidate_labels_by_slot": [True, False]}]
    assert "held_rows_label_seal" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["semantic_contrast_preregistration_v2_ready_score"] = 1
    changed["gate_check_summary"] = []
    changed["models_used"] = []
    assert {"ready_gate_summary", "models_used"}.issubset(exp.validate_artifact(changed))
    assert exp._contains_key("plain-leaf", {"expected_label"}) is False


def test_req_inference_6867_atomic_writer_uses_requested_path(tmp_path: Path) -> None:
    """REQ-INFERENCE-6867 writes only the explicitly requested test path."""

    target = tmp_path / "exp6867.json"
    exp.write_json_atomic(target, {"experiment_id": 6867})
    assert json.loads(target.read_text(encoding="utf-8")) == {"experiment_id": 6867}


class _LengthTokenizer:
    def tokenize(self, value: bytes, *, add_bos: bool, special: bool) -> list[int]:
        assert special is False
        count = max(1, (len(value) + 9) // 10)
        return ([99] if add_bos else []) + list(range(count))


class _UnmatchedTokenizer:
    def tokenize(self, value: bytes, *, add_bos: bool, special: bool) -> list[int]:
        assert special is False
        if b'"answer":"invalid"' in value:
            return [1, 2]
        if b'"answer":"valid"' in value:
            return [1]
        return [99] if add_bos else [0]


def _tokenizer_receipt() -> dict:
    return {
        "canonical_tokenizer_payload_sha256": "sha256:payload",
        "special_token_settings": {"bos_token_id": 2},
    }


def test_scenario_6867_render_cell_records_exact_native_sequences() -> None:
    """SCENARIO-INFERENCE-6867-NUISANCE-CONTROLS records accepted cells."""

    row = exp.render_group_cell(
        tokenizer=_LengthTokenizer(),
        tokenizer_receipt=_tokenizer_receipt(),
        model_hf_id=exp.MODEL_SPECS[0],
        group=_group(0),
    )
    assert row["accepted"] is True
    assert row["rejection_reasons"] == []
    assert row["padding_receipt"]["matched"] is True
    assert row["candidate_token_counts"]["slot_0"] == row["candidate_token_counts"]["slot_1"]
    assert (
        row["candidate_character_counts"]["slot_0"] == row["candidate_character_counts"]["slot_1"]
    )
    assert row["paired_prompt_token_ids"]["base"] == row["paired_prompt_token_ids"]["label_swap"]
    assert row["canonical_tokenizer_payload_hash"] == "sha256:payload"
    assert row["token_likelihood_call_count"] == 0
    assert row["generated_answer_count"] == 0


def test_scenario_6867_render_cell_preserves_unmatched_rejection() -> None:
    """SCENARIO-INFERENCE-6867-NUISANCE-CONTROLS preserves failed matching."""

    row = exp.render_group_cell(
        tokenizer=_UnmatchedTokenizer(),
        tokenizer_receipt=_tokenizer_receipt(),
        model_hf_id=exp.MODEL_SPECS[1],
        group=_group(0),
    )
    assert row["accepted"] is False
    assert row["padding_receipt"] == {
        "method": "bounded_trailing_whitespace_exact_match",
        "matched": False,
        "maximum_extra_character_budget": exp.MAX_WHITESPACE_EXTRA,
    }
    assert "unequal_candidate_token_counts" in row["rejection_reasons"]
    malformed = exp.render_group_cell(
        tokenizer=_LengthTokenizer(),
        tokenizer_receipt=_tokenizer_receipt(),
        model_hf_id=exp.MODEL_SPECS[2],
        group={"semantic_identity": "bad", "family": "exact_energy", "candidates": []},
    )
    assert "candidate_count_mismatch" in malformed["rejection_reasons"]


def test_req_inference_6867_sample_rows_and_held_redaction() -> None:
    """REQ-INFERENCE-6867 counts accepted groups and removes held labels."""

    split = {
        "calibration_group_ids": [f"c-{index}" for index in range(20)],
        "held_group_ids": [f"h-{index}" for index in range(20)],
    }
    rows = [
        {
            "model_hf_id": model,
            "semantic_group_identity": group_id,
            "split": "calibration" if group_id.startswith("c-") else "held",
            "score_identity": f"score-{model}-{group_id}",
            "candidate_labels_by_slot": [True, False],
        }
        for model in exp.MODEL_SPECS
        for group_id in split["calibration_group_ids"] + split["held_group_ids"]
    ]
    sample_rows = exp._sample_size_rows(rows, split)
    assert len(sample_rows) == 6
    assert all(row["floor_passed"] for row in sample_rows)
    commitments = {group_id: f"seal-{group_id}" for group_id in split["held_group_ids"]}
    public = exp._redact_held_rows(rows, commitments)
    held_rows = [row for row in public if row["split"] == "held"]
    calibration_rows = [row for row in public if row["split"] == "calibration"]
    assert all("candidate_labels_by_slot" not in row for row in held_rows)
    assert all(row["candidate_label_commitment"].startswith("seal-") for row in held_rows)
    assert all(row["candidate_labels_by_slot"] == [True, False] for row in calibration_rows)
    assert all("score_identity" not in row for row in public)
    assert all(row["split_cell_identity"].startswith("score-") for row in public)


def test_req_inference_6867_source_reader_and_frozen_binding_reducer(tmp_path: Path) -> None:
    """REQ-INFERENCE-6867 preserves typed source and frozen-binding evidence."""

    for relative in exp.SOURCE_PATHS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    documents, raw_values, failures = exp._read_sources(tmp_path)
    assert set(documents) == set(exp.SOURCE_PATHS)
    assert set(raw_values) == set(exp.SOURCE_PATHS)
    assert failures == []

    (tmp_path / exp.SOURCE_PATHS["exp6862"]).write_text("[]", encoding="utf-8")
    (tmp_path / exp.SOURCE_PATHS["exp6866"]).write_text("not-json", encoding="utf-8")
    documents, raw_values, failures = exp._read_sources(tmp_path)
    assert documents == {}
    assert raw_values == {}
    assert [row["check"] for row in failures] == [
        "source.exp6862.readable",
        "source.exp6866.readable",
    ]

    hashes = {
        model: {
            "path": f"/{index}.gguf",
            "sha256": f"sha256:{index}",
            "size_bytes": index,
            "quantization": "Q4_K_M",
            "snapshot_identity": f"snapshot-{index}",
        }
        for index, model in enumerate(exp.MODEL_SPECS, 1)
    }
    payload_rows = [
        {"hf_id": model, "live_canonical_payload_sha256": f"sha256:payload-{index}"}
        for index, model in enumerate(exp.MODEL_SPECS, 1)
    ]
    frozen = exp._frozen_bindings(
        {"model_artifact_hashes": hashes, "canonical_payload_hash_rows": payload_rows}
    )
    assert frozen[exp.MODEL_SPECS[0]]["path"] == "/1.gguf"
    assert frozen[exp.MODEL_SPECS[2]]["canonical_tokenizer_payload_sha256"] == ("sha256:payload-3")
    empty = exp._frozen_bindings({"model_artifact_hashes": [], "canonical_payload_hash_rows": {}})
    assert all(row["path"] is None for row in empty.values())


class _ReceiptTokenizer:
    evidence = {
        "special_token_ids": {"bos_token_id": 2},
        "add_bos_metadata_default": True,
        "chat_template": "template",
        "vocabulary_size": 100,
        "token_pieces_sha256": "sha256:pieces",
    }


def test_scenario_6867_canonical_tokenizer_receipt_is_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFERENCE-6867-TOKENIZER-IDENTITY recomputes one payload."""

    calls: list[tuple[Any, ...]] = []

    def fake_probe(tokenizer: Any, manifest: Mapping[str, Any]) -> list[dict]:
        calls.append((tokenizer, manifest))
        return [{"probe_id": "one"}]

    def fake_reduce(
        evidence: Mapping[str, Any],
        outputs: list[dict],
        *,
        source_receipt: Mapping[str, Any],
        source_kind: str,
    ) -> dict:
        calls.append((evidence, outputs, source_receipt, source_kind))
        return {
            "canonical_payload_schema_version": "schema-v1",
            "canonical_payload_sha256": "sha256:canonical",
        }

    monkeypatch.setattr(exp.binding, "run_probe_matrix", fake_probe)
    monkeypatch.setattr(exp.binding, "reduce_tokenizer_receipt", fake_reduce)
    source = {
        "semantic_probe_manifest": {"probes": ["one"]},
        "live_receipt_rows": [{"hf_id": exp.MODEL_SPECS[0], "wrapper": "old"}],
    }
    receipt = exp._tokenizer_receipt(
        _ReceiptTokenizer(), model_hf_id=exp.MODEL_SPECS[0], exp6866=source
    )
    assert receipt["canonical_tokenizer_payload_sha256"] == "sha256:canonical"
    assert receipt["special_token_settings"]["prompt"]["add_bos"] is True
    assert receipt["probe_output_count"] == 1
    assert receipt["token_likelihood_call_count"] == 0
    assert len(calls) == 2
    with pytest.raises(ValueError, match="semantic probe manifest missing"):
        exp._tokenizer_receipt(_ReceiptTokenizer(), model_hf_id=exp.MODEL_SPECS[0], exp6866={})
