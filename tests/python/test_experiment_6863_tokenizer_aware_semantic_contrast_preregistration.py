"""Tests for REQ-INFERENCE-6863 and SCENARIO-INFERENCE-6863-*.

These tests use small receipts and token rows. They never load a GGUF or write
the research artifact.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6863_tokenizer_aware_semantic_contrast_preregistration as exp


ROOT = Path(__file__).resolve().parents[2]


def _receipt(*, metadata_present: bool = True) -> dict:
    """Return one native-tokenizer receipt for pure contract tests."""

    return exp.tokenizer_receipt_from_metadata(
        metadata={"tokenizer.ggml.model": "test"} if metadata_present else {},
        probe_token_ids=[1, 2, 3],
        special_tokens={"bos_token_id": 1, "eos_token_id": 2},
        chat_template="{{ messages }}",
    )


def _model(hf_id: str, *, path: str = "/cache/model.gguf", digest: str = "sha256:abc") -> dict:
    """Return one resolved model row with a frozen cache binding."""

    return {
        "hf_id": hf_id,
        "model_path": path,
        "model_sha256": digest,
        "local_model_present": True,
        "cached_sota_pair_called": True,
        "tokenizer_receipt": _receipt(),
    }


def _cell(group: str, *, split: str = "calibration") -> dict:
    """Return one nuisance-matched token cell."""

    return {
        "cell_id": exp.cell_identity(exp.MODEL_SPECS[0], group),
        "model_hf_id": exp.MODEL_SPECS[0],
        "semantic_group_identity": group,
        "split": split,
        "prompt_token_ids": [1, 4, 5],
        "prompt_token_count": 3,
        "candidate_token_ids": {"slot_0": [7, 8], "slot_1": [9, 10]},
        "candidate_token_counts": {"slot_0": 2, "slot_1": 2},
        "candidate_character_counts": {"slot_0": 14, "slot_1": 14},
        "label_position_contract": {"base": [0, 1], "swapped": [1, 0]},
        "paired_prompt_token_counts": {"base": 3, "label_swap": 3},
    }


def test_req_inference_6863_spec_declares_required_contract() -> None:
    """REQ-INFERENCE-6863: OpenSpec owns the tokenizer and seal contract."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-INFERENCE-6863", 1)[1]
    for anchor in exp.SPEC_REFS:
        assert anchor in text
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_6863_native_tokenizer_rejects_missing_metadata() -> None:
    """SCENARIO-INFERENCE-6863-NATIVE-TOKENIZER: metadata is mandatory."""

    receipt = _receipt(metadata_present=False)
    assert receipt["source"] == "embedded_gguf_llama_cpp_vocab_only"
    assert receipt["metadata_present"] is False
    assert receipt["loadable"] is False
    assert receipt["tokenizer_sha256"] == ""


def test_scenario_6863_native_probe_preserves_frozen_bos_setting() -> None:
    """SCENARIO-INFERENCE-6863-NATIVE-TOKENIZER: the Exp6850 probe uses BOS."""

    receipt = _receipt()
    assert receipt["probe_tokenize_settings"] == {"add_bos": True, "special": False}


def test_scenario_6863_cache_substitution_fails_closed() -> None:
    """SCENARIO-INFERENCE-6863-CACHE-AND-HASH: a new path is not the frozen file."""

    resolved = _model(exp.MODEL_SPECS[0], path="/cache/substitute.gguf")
    frozen = {"path": "/cache/frozen.gguf", "sha256": "sha256:abc"}
    assert exp.model_binding_errors(resolved, frozen) == ["cache_substitution"]


def test_scenario_6863_hash_drift_fails_closed() -> None:
    """SCENARIO-INFERENCE-6863-CACHE-AND-HASH: changed bytes fail the gate."""

    resolved = _model(exp.MODEL_SPECS[0], digest="sha256:changed")
    frozen = {"path": "/cache/model.gguf", "sha256": "sha256:abc"}
    assert exp.model_binding_errors(resolved, frozen) == ["model_hash_drift"]


def test_scenario_6863_binding_names_all_missing_receipts() -> None:
    """REQ-INFERENCE-6863: missing cache and tokenizer evidence stays explicit."""

    resolved = _model(exp.MODEL_SPECS[0])
    resolved.update(local_model_present=False, cached_sota_pair_called=False)
    resolved["tokenizer_receipt"] = _receipt(metadata_present=False)
    frozen = {"path": "/cache/model.gguf", "sha256": "sha256:abc"}
    assert exp.model_binding_errors(resolved, frozen) == [
        "cached_sota_pair_not_called",
        "model_file_missing",
        "tokenizer_metadata_missing",
    ]


def test_scenario_6863_unequal_token_counts_reject_cell() -> None:
    """SCENARIO-INFERENCE-6863-NUISANCE-MATCH: unequal candidates are rejected."""

    cell = _cell("group-a")
    cell["candidate_token_counts"]["slot_1"] = 3
    cell["candidate_token_ids"]["slot_1"].append(11)
    assert exp.cell_nuisance_errors(cell) == ["unequal_candidate_token_counts"]


def test_scenario_6863_prompt_and_label_controls_reject_cell() -> None:
    """SCENARIO-INFERENCE-6863-NUISANCE-MATCH: paired controls must match."""

    cell = _cell("group-a")
    cell["paired_prompt_token_counts"]["label_swap"] = 4
    cell["label_position_contract"]["swapped"] = [0, 1]
    assert exp.cell_nuisance_errors(cell) == [
        "paired_prompt_length_mismatch",
        "label_position_contract_failed",
    ]


def test_scenario_6863_split_overlap_and_group_leakage_are_detected() -> None:
    """SCENARIO-INFERENCE-6863-GROUP-SPLIT: identities use one split only."""

    rows = [_cell("group-a"), _cell("group-a", split="held")]
    errors = exp.split_errors(["group-a"], ["group-a"], rows)
    assert errors == ["split_overlap:group-a", "group_leakage:group-a"]


def test_scenario_6863_split_is_deterministic_and_group_disjoint() -> None:
    """SCENARIO-INFERENCE-6863-GROUP-SPLIT: the seed freezes whole groups."""

    groups = [f"group-{index:02d}" for index in range(40)]
    first = exp.freeze_group_splits(groups, random_seed=6863)
    second = exp.freeze_group_splits(reversed(groups), random_seed=6863)
    assert first == second
    assert len(first["calibration_group_ids"]) == 20
    assert len(first["held_group_ids"]) == 20
    assert set(first["calibration_group_ids"]).isdisjoint(first["held_group_ids"])
    assert first["calibration_group_hash"].startswith("sha256:")
    assert first["held_group_hash"].startswith("sha256:")


def test_scenario_6863_held_manifest_contains_commitments_not_labels() -> None:
    """SCENARIO-INFERENCE-6863-HELD-SEAL: held labels are absent from public rows."""

    group = {
        "semantic_identity": "group-held",
        "candidates": [
            {"candidate_id": "valid", "expected_label": True},
            {"candidate_id": "invalid", "expected_label": False},
        ],
    }
    sealed = exp.seal_held_groups([group])
    assert sealed[0]["semantic_group_identity"] == "group-held"
    assert sealed[0]["label_commitment"].startswith("sha256:")
    assert "expected_label" not in json.dumps(sealed)


def test_scenario_6863_calibration_loader_refuses_held_label_access() -> None:
    """SCENARIO-INFERENCE-6863-HELD-SEAL: calibration cannot load held labels."""

    loader = exp.CalibrationLabelLoader({"group-cal": (True, False)}, {"group-held"})
    assert loader.load("group-cal") == (True, False)
    with pytest.raises(exp.HeldLabelAccessError, match="group-held"):
        loader.load("group-held")
    with pytest.raises(KeyError, match="group-missing"):
        loader.load("group-missing")
    assert loader.held_label_access_count == 0
    assert loader.held_label_access_attempt_count == 1


def test_req_inference_6863_preexisting_cell_identity_blocks_gate() -> None:
    """REQ-INFERENCE-6863: prior score identity reuse is detected before tokenization."""

    identity = exp.cell_identity(exp.MODEL_SPECS[0], "group-a")
    artifacts = [b'{"old_score_cell":"' + identity.encode() + b'"}']
    assert exp.preexisting_score_identities(artifacts, [identity, "absent"]) == [identity]


def test_scenario_6863_sample_floor_returns_null_readiness() -> None:
    """SCENARIO-INFERENCE-6863-SAMPLE-FLOOR: one short model keeps readiness zero."""

    sample_rows = []
    for model_id in exp.MODEL_SPECS:
        sample_rows.extend(
            [
                {
                    "model_hf_id": model_id,
                    "family": exp.MODEL_FAMILIES[model_id],
                    "split": split,
                    "accepted_group_count": 20,
                    "minimum_group_count": 20,
                }
                for split in ("calibration", "held")
            ]
        )
    assert exp.readiness_score(sample_rows, all_other_checks_pass=True) == 1
    sample_rows[0]["accepted_group_count"] = 19
    assert exp.readiness_score(sample_rows, all_other_checks_pass=True) == 0
    assert exp.readiness_score(sample_rows, all_other_checks_pass=False) == 0


def test_req_inference_6863_artifact_validation_is_fail_closed() -> None:
    """REQ-INFERENCE-6863: schema, zero calls, fields, and terminal values are checked."""

    artifact = {field: [] for field in exp.REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        inference_substrate=exp.INFERENCE_SUBSTRATE,
        token_likelihood_call_count=0,
        split_overlap_count=0,
        held_label_access_count=0,
        verifier_is_oracle=False,
        verdict_class="null",
        honest_verdict="complete_null_tokenizer_aware_preregistration_ready",
        field_principles={field: "reason" for field in exp.REQUIRED_ARTIFACT_FIELDS},
    )
    assert exp.validate_artifact(artifact) == []
    broken = deepcopy(artifact)
    del broken["rows"]
    broken["token_likelihood_call_count"] = 1
    broken["honest_verdict"] = "not-terminal"
    broken["verdict_class"] = "unknown"
    broken["verifier_is_oracle"] = True
    assert exp.validate_artifact(broken) == [
        "missing_field:rows",
        "field_principle_keys",
        "token_likelihood_call_count",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
    ]
