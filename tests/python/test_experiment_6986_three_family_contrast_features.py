"""Tests for the label-blind teacher-forced feature bank.

Spec refs: REQ-INF-6986 and SCENARIO-INF-6986-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6986_three_family_contrast_features as exp


def _manifest_row(candidate_id: str = "candidate-a", text: str = '{"x":1}') -> dict:
    return {
        "candidate_id": candidate_id,
        "prompt_text": "Read the frozen candidate.\nCANDIDATE:\n",
        "candidate_text": text,
        "source_block": "exp6984",
        "source_candidate_hash": exp.sha256_text(text),
    }


class FakeLlama:
    """Small embedded-tokenizer stand-in for pure scoring tests."""

    def __init__(self) -> None:
        self.scores = np.zeros((0, 8), dtype=np.float64)
        self.reset_calls = 0
        self.eval_calls: list[list[int]] = []

    def tokenize(self, value: bytes, *, add_bos: bool, special: bool = True) -> list[int]:
        del special
        tokens = [2 + byte % 6 for byte in value]
        return ([1] if add_bos else []) + tokens

    def detokenize(self, tokens: list[int]) -> bytes:
        return bytes((token - 2) % 6 for token in tokens if token != 1)

    def reset(self) -> None:
        self.reset_calls += 1

    def eval(self, tokens: list[int]) -> None:
        self.eval_calls.append(list(tokens))
        rows = []
        for position, token in enumerate(tokens):
            row = np.arange(8, dtype=np.float64) / 10.0
            row[(token + position + 1) % 8] += 1.0
            rows.append(row)
        self.scores = np.asarray(rows)


def _feature_row(candidate_id: str, family: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_hash": exp.sha256_text(candidate_id),
        "model_id": family,
        "terminal": True,
        "live_cuda": True,
        "tokenizer_alignment": True,
        "raw_token_count": 2,
        "raw_token_hash": exp.sha256_json([candidate_id, family]),
    }


def _teardown_rows() -> list[dict]:
    return [
        {
            "model_id": family,
            "process_exit_code": 0,
            "owned_process_absent": True,
            "port_release_confirmed": True,
            "model_close_called": True,
            "signals_sent": [],
            "passed": True,
        }
        for family in exp.REQUIRED_MODEL_IDS
    ]


def _vram_rows() -> list[dict]:
    return [
        {"model_id": family, "passed": True, "max_residual_mb": 512}
        for family in exp.REQUIRED_MODEL_IDS
    ]


def test_model_specs_start_with_cached_pair_and_reject_legacy(tmp_path: Path) -> None:
    """REQ-INF-6986: only the three exact local GGUF families can enter rows."""
    paths = {}
    for index, family in enumerate(exp.REQUIRED_MODEL_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(b"gguf")
        paths[family] = str(path)

    calls: list[tuple[int, int]] = []

    def pair(*, gpu_indices: tuple[int, int]) -> list[dict]:
        calls.append(gpu_indices)
        return [
            {"hf_id": family, "model_path": paths[family], "gpu": index}
            for index, family in enumerate(exp.REQUIRED_MODEL_IDS[:2])
        ]

    rows = exp.resolve_model_specs(
        cached_pair_func=pair,
        resolver=lambda family, _quant: paths[family],
    )
    assert calls == [(0, 1)]
    assert [row["hf_id"] for row in rows] == list(exp.REQUIRED_MODEL_IDS)
    assert exp.model_spec_errors(rows) == []

    legacy = deepcopy(rows)
    legacy[0]["hf_id"] = "Qwen/Qwen3.5-0.8B"
    assert "model_ids_mismatch" in exp.model_spec_errors(legacy)


def test_model_spec_validation_rejects_every_unsafe_shape() -> None:
    """SCENARIO-INF-6986-MODELS: missing, auxiliary, and CPU specs fail closed."""
    rows = [
        {
            "hf_id": family,
            "model_path": "" if index == 0 else "/models/mmproj.gguf",
            "gpu_indices": [0],
            "headline_eligible": False,
        }
        for index, family in enumerate(exp.REQUIRED_MODEL_IDS)
    ]
    errors = exp.model_spec_errors(rows)

    assert any(error.startswith("model_path_missing:") for error in errors)
    assert any(error.startswith("model_path_not_primary_gguf:") for error in errors)
    assert sum(error.startswith("dual_gpu_indices_missing:") for error in errors) == 3
    assert sum(error.startswith("headline_eligibility_missing:") for error in errors) == 3


@pytest.mark.parametrize(
    "denied_field",
    sorted(exp.FORBIDDEN_SCORING_FIELDS),
)
def test_label_denial_mutations_fail_at_any_depth(denied_field: str) -> None:
    """SCENARIO-INF-6986-LABEL-DENIAL: oracle metadata fails closed."""
    row = _manifest_row()
    row["nested"] = {"deeper": {denied_field: "oracle-value"}}
    errors = exp.label_blind_input_errors(row)
    assert any(denied_field in error for error in errors)


def test_manifest_freezes_order_prompt_candidate_and_config_hashes() -> None:
    """REQ-INF-6986: label-denied manifest construction freezes every input."""
    rows = [_manifest_row("a", "alpha"), _manifest_row("b", "beta")]
    manifest = exp.build_scoring_manifest(rows)
    repeated = exp.build_scoring_manifest(rows)

    assert manifest == repeated
    assert manifest["scoring_manifest_hash"].startswith("sha256:")
    assert manifest["candidate_order_hash"].startswith("sha256:")
    assert manifest["scoring_config_hash"] == exp.sha256_json(exp.SCORING_CONFIG)
    assert [row["ordinal"] for row in manifest["rows"]] == [0, 1]
    assert not exp.label_blind_input_errors(manifest["rows"])
    assert all(key not in exp.FORBIDDEN_SCORING_FIELDS for row in manifest["rows"] for key in row)


def test_actual_source_projection_has_138_hash_matching_candidates() -> None:
    """REQ-INF-6986: frozen sources contribute 72, 48, and 18 rows."""
    sources = exp.read_source_artifacts(exp.REPO_ROOT)
    rows = exp.project_frozen_candidates(sources)
    counts = exp.candidate_source_count_rows(rows)

    assert counts == [
        {"source_block": "exp6984", "expected": 72, "observed": 72, "passed": True},
        {"source_block": "exp6985", "expected": 48, "observed": 48, "passed": True},
        {"source_block": "exp6976_transfer", "expected": 18, "observed": 18, "passed": True},
    ]
    assert len(rows) == 138
    assert len({row["candidate_id"] for row in rows}) == 138
    assert all(
        exp.sha256_text(row["candidate_text"]) == row["source_candidate_hash"] for row in rows
    )
    assert not exp.label_blind_input_errors(rows)


def test_transfer_projection_rejects_a_candidate_without_its_attempt() -> None:
    """SCENARIO-INF-6986-SOURCES: transfer text must come from the pinned bank."""
    sources = {
        "exp6984": {"per_candidate_rows": []},
        "exp6985": {"per_candidate_rows": []},
        "exp6975": {"per_attempt_rows": []},
        "exp6976": {"per_candidate_rows": [{"attempt_key": "lost", "parse_success": True}]},
    }
    with pytest.raises(exp.ManifestError, match="transfer_attempt_missing:lost"):
        exp.project_frozen_candidates(sources)


def test_manifest_rejects_duplicate_empty_and_hash_drift() -> None:
    """SCENARIO-INF-6986-MANIFEST: identity, text, and content hash fail closed."""
    rows = [
        _manifest_row("duplicate", "valid"),
        _manifest_row("duplicate", ""),
    ]
    rows[1]["prompt_text"] = ""
    rows[1]["source_candidate_hash"] = exp.sha256_text("not-empty")
    with pytest.raises(exp.ManifestError) as caught:
        exp.build_scoring_manifest(rows)

    message = str(caught.value)
    assert "candidate_id_invalid_or_duplicate:duplicate" in message
    assert "empty_prompt_or_candidate:duplicate" in message
    assert "candidate_hash_mismatch:duplicate" in message


def test_scorer_blocks_strip_controller_fields_and_denial_receipts_pass() -> None:
    """SCENARIO-INF-6986-PROCESS: workers see only identifier and two text fields."""
    manifest = exp.build_scoring_manifest(
        [
            _manifest_row("a", "one"),
            _manifest_row("b", "two") | {"source_block": "exp6985"},
        ]
    )
    blocks = exp.scoring_process_blocks(manifest["rows"])
    denial_rows = exp.runtime_label_denial_rows()

    assert [len(block) for block in blocks] == [1, 1, 0]
    assert all(
        set(row) == {"candidate_id", "prompt_text", "candidate_text"}
        for block in blocks
        for row in block
    )
    assert {row["mutation_field"] for row in denial_rows} == set(exp.FORBIDDEN_SCORING_FIELDS)
    assert all(row["passed"] is True for row in denial_rows)


def test_teacher_forced_tokenization_uses_embedded_tokenizer_suffix() -> None:
    """SCENARIO-INF-6986-TOKENIZER: the candidate is an exact token suffix."""
    model = FakeLlama()
    tokenized = exp.tokenize_teacher_forced(model, _manifest_row(text="abc"))

    assert tokenized["tokenizer_source"] == "embedded_gguf"
    assert tokenized["alignment_passed"] is True
    assert (
        tokenized["full_token_ids"][-len(tokenized["candidate_token_ids"]) :]
        == tokenized["candidate_token_ids"]
    )
    assert tokenized["candidate_token_count"] == 3


def test_distribution_features_keep_scalars_and_hash_only() -> None:
    """SCENARIO-INF-6986-TOKENS: raw rows omit full vocabulary vectors."""
    logits = np.asarray([0.0, 1.0, 2.0, -1.0], dtype=np.float64)
    row = exp.distribution_features(logits, selected_token_id=2, previous_surprisal=0.5)

    probabilities = np.exp(logits - np.log(np.exp(logits).sum()))
    assert row["selected_token_log_probability"] == pytest.approx(np.log(probabilities[2]))
    assert row["surprisal"] == pytest.approx(-np.log(probabilities[2]))
    assert row["token_entropy"] == pytest.approx(-(probabilities * np.log(probabilities)).sum())
    assert row["top_probability_margin"] > 0
    assert row["local_surprisal_change"] == pytest.approx(row["surprisal"] - 0.5)
    assert row["full_logit_vector_hash"].startswith("sha256:")
    assert "logits" not in row

    singleton = exp.distribution_features(
        np.asarray([3.0]), selected_token_id=0, previous_surprisal=None
    )
    assert singleton["top_probability_margin"] == 0.0
    assert singleton["local_surprisal_change"] == 0.0
    with pytest.raises(ValueError, match="selected_token_id_out_of_range"):
        exp.distribution_features(logits, selected_token_id=9, previous_surprisal=None)


def test_score_candidate_preserves_per_token_alignment_and_features() -> None:
    """REQ-INF-6986: scoring teacher-forces tokens and emits no answer."""
    model = FakeLlama()
    result = exp.score_candidate(
        model, _manifest_row(text="ab"), model_id=exp.REQUIRED_MODEL_IDS[0]
    )

    assert model.reset_calls == 1
    assert len(model.eval_calls) == 1
    assert result["sequence_row"]["terminal"] is True
    assert result["sequence_row"]["candidate_token_count"] == 2
    assert result["sequence_row"]["sequence_nll"] > 0
    assert result["sequence_row"]["length_normalized_nll"] > 0
    assert len(result["raw_token_rows"]) == 2
    assert [row["candidate_token_position"] for row in result["raw_token_rows"]] == [0, 1]
    assert all("full_vocabulary_logits" not in row for row in result["raw_token_rows"])
    assert result["parser_row"]["candidate_id"] == "candidate-a"


def test_score_candidate_rejects_alignment_context_and_missing_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INF-6986-TOKENS: all three token/logit alignment guards are terminal."""
    model = FakeLlama()
    row = _manifest_row(text="ab")
    original = exp.tokenize_teacher_forced

    monkeypatch.setattr(
        exp,
        "tokenize_teacher_forced",
        lambda _model, _row: {"alignment_passed": False},
    )
    with pytest.raises(ValueError, match="tokenizer_alignment_failed"):
        exp.score_candidate(model, row, model_id=exp.REQUIRED_MODEL_IDS[0])

    monkeypatch.setattr(exp, "tokenize_teacher_forced", original)
    monkeypatch.setitem(exp.SCORING_CONFIG, "n_ctx", 1)
    with pytest.raises(ValueError, match="sequence_exceeds_context"):
        exp.score_candidate(model, row, model_id=exp.REQUIRED_MODEL_IDS[0])

    monkeypatch.setitem(exp.SCORING_CONFIG, "n_ctx", 16_384)

    class MissingScores(FakeLlama):
        def eval(self, tokens: list[int]) -> None:
            self.eval_calls.append(tokens)
            self.scores = np.zeros((0, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="logit_alignment_unavailable"):
        exp.score_candidate(MissingScores(), row, model_id=exp.REQUIRED_MODEL_IDS[0])


def test_parser_features_are_structural_and_label_free() -> None:
    """REQ-INF-6986: parser features count structure without oracle semantics."""
    row = exp.parser_structure_features("c", '{"a":[1,{"b":true}]}')
    malformed = exp.parser_structure_features("d", "{bad")

    assert row["json_parseable"] is True
    assert row["object_count"] == 2
    assert row["array_count"] == 1
    assert row["key_count"] == 2
    assert malformed["json_parseable"] is False
    assert set(row).isdisjoint(exp.FORBIDDEN_SCORING_FIELDS)

    scalar_row = exp.parser_structure_features("e", '[null,"text"]')
    assert scalar_row["null_count"] == 1
    assert scalar_row["string_count"] == 1


def test_checkpoint_recovery_replays_manifest_and_rejects_duplicates(tmp_path: Path) -> None:
    """SCENARIO-INF-6986-CHECKPOINT: valid blocks resume without duplicates."""
    path = tmp_path / "checkpoint.json"
    row = _feature_row("a", exp.REQUIRED_MODEL_IDS[0])
    receipt = exp.write_checkpoint(path, manifest_hash="sha256:manifest", rows=[row])
    loaded = exp.load_checkpoint(path, manifest_hash="sha256:manifest")

    assert receipt["checkpoint_hash"].startswith("sha256:")
    assert loaded["rows"] == [row]
    assert exp.completed_feature_keys(loaded["rows"]) == {("a", exp.REQUIRED_MODEL_IDS[0])}
    with pytest.raises(exp.CheckpointError, match="manifest_hash_mismatch"):
        exp.load_checkpoint(path, manifest_hash="sha256:other")
    with pytest.raises(exp.CheckpointError, match="duplicate_feature_key"):
        exp.write_checkpoint(path, manifest_hash="sha256:manifest", rows=[row, row])


def test_completeness_requires_414_exact_cuda_terminal_rows() -> None:
    """SCENARIO-INF-6986-FAMILIES: all 414 exact family rows are required."""
    manifest = [
        {"candidate_id": f"c-{index}", "candidate_hash": exp.sha256_text(f"c-{index}")}
        for index in range(138)
    ]
    rows = [
        _feature_row(candidate["candidate_id"], family)
        | {"candidate_hash": candidate["candidate_hash"]}
        for candidate in manifest
        for family in exp.REQUIRED_MODEL_IDS
    ]
    assert (
        exp.feature_bank_completion_errors(
            manifest, rows, _teardown_rows(), _vram_rows(), label_denial_passed=True
        )
        == []
    )

    cpu_rows = deepcopy(rows)
    cpu_rows[0]["live_cuda"] = False
    assert "cuda_incomplete" in exp.feature_bank_completion_errors(
        manifest, cpu_rows, _teardown_rows(), _vram_rows(), label_denial_passed=True
    )
    assert "feature_row_count" in exp.feature_bank_completion_errors(
        manifest, rows[:-1], _teardown_rows(), _vram_rows(), label_denial_passed=True
    )

    duplicate_rows = deepcopy(rows)
    duplicate_rows[-1] = deepcopy(duplicate_rows[0])
    assert "duplicate_feature_key" in exp.feature_bank_completion_errors(
        manifest, duplicate_rows, _teardown_rows(), _vram_rows(), label_denial_passed=True
    )
    drifted_rows = deepcopy(rows)
    drifted_rows[0]["candidate_hash"] = "sha256:drifted"
    assert "candidate_hash_or_terminal_mismatch" in exp.feature_bank_completion_errors(
        manifest, drifted_rows, _teardown_rows(), _vram_rows(), label_denial_passed=True
    )


def test_unowned_or_incomplete_teardown_keeps_readiness_zero() -> None:
    """SCENARIO-INF-6986-TEARDOWN: only clean owned release permits handoff."""
    manifest = [{"candidate_id": "a", "candidate_hash": exp.sha256_text("a")}]
    rows = [
        _feature_row("a", family) | {"candidate_hash": exp.sha256_text("a")}
        for family in exp.REQUIRED_MODEL_IDS
    ]
    teardown = _teardown_rows()
    teardown[0]["signals_sent"] = ["SIGTERM"]
    errors = exp.feature_bank_completion_errors(
        manifest,
        rows,
        teardown,
        _vram_rows(),
        label_denial_passed=True,
        expected_candidate_count=1,
    )
    assert "teardown_incomplete" in errors


def test_labels_join_only_after_all_family_exit_and_do_not_change_evidence() -> None:
    """SCENARIO-INF-6986-LABEL-JOIN: labels stay in a separate late table."""
    feature_rows = [_feature_row("a", family) for family in exp.REQUIRED_MODEL_IDS]
    before = deepcopy(feature_rows)
    labels = {"a": {"exact_label": "equivalent", "exact_success": True}}

    with pytest.raises(exp.LabelJoinError, match="family_processes_not_exited"):
        exp.join_labels(feature_rows, labels, _teardown_rows()[:-1])
    joined = exp.join_labels(feature_rows, labels, _teardown_rows())
    assert feature_rows == before
    assert len(joined) == 3
    assert all(row["exact_label"] == "equivalent" for row in joined)
    with pytest.raises(exp.LabelJoinError, match="label_missing:a"):
        exp.join_labels(feature_rows, {}, _teardown_rows())


def test_blocked_artifact_has_required_bare_fields_and_failed_gate() -> None:
    """SCENARIO-INF-6986-GATES: a blocked result keeps the complete schema."""
    checks = [exp.gate_check("cuda_device_count", 2, 1, passed=False)]
    artifact = exp.build_artifact(
        run_date="20260904",
        duration_s=0.1,
        live_duration_s=0.0,
        preconditions={"all_passed": False, "checks": checks},
        model_specs=[],
        source_artifact_hashes={},
        model_file_hashes={},
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["three_family_feature_bank_complete_score"] == 0
    assert type(artifact["three_family_feature_bank_complete_score"]) is int
    assert type(artifact["expected_feature_row_count"]) is int
    assert artifact["verifier_fit_performed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_three_family_contrast_features"
    assert artifact["gate_check_summary"]["failed_check"] == "cuda_device_count"
    assert artifact["gate_check_summary"]["expected_value"] == 2
    assert artifact["gate_check_summary"]["observed_value"] == 1


def test_partial_artifact_exposes_all_incomplete_evidence_counts() -> None:
    """SCENARIO-INF-6986-GATES: successful preflight cannot hide partial evidence."""
    artifact = exp.build_artifact(
        run_date="20260904",
        duration_s=1.0,
        live_duration_s=0.5,
        preconditions={"all_passed": True, "checks": []},
        model_specs=[],
        source_artifact_hashes={},
        model_file_hashes={},
        per_candidate_model_rows=[{"candidate_id": "a", "raw_token_count": 2}],
    )

    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("partial_")
    failures = artifact["gate_check_summary"]["observed_value"]
    assert "sequence_feature_count" in failures
    assert "parser_feature_count" in failures
    assert "raw_token_count" in failures
    assert "joined_label_count" in failures
    assert "process_isolation_incomplete" in failures


def test_positive_artifact_validates_required_fields_and_principles() -> None:
    """SCENARIO-INF-6986-BARE: a complete artifact has bare downstream fields."""
    manifest_rows = [
        _manifest_row(f"c-{index}", f"candidate-{index}")
        | {
            "ordinal": index,
            "candidate_hash": exp.sha256_text(f"candidate-{index}"),
            "prompt_hash": exp.sha256_text("Read the frozen candidate.\nCANDIDATE:\n"),
        }
        for index in range(138)
    ]
    feature_rows = [
        _feature_row(row["candidate_id"], family) | {"candidate_hash": row["candidate_hash"]}
        for row in manifest_rows
        for family in exp.REQUIRED_MODEL_IDS
    ]
    raw_token_rows = [
        {
            "candidate_id": row["candidate_id"],
            "model_id": row["model_id"],
            "candidate_token_position": token_position,
        }
        for row in feature_rows
        for token_position in range(2)
    ]
    artifact = exp.build_artifact(
        run_date="20260904",
        duration_s=65.0,
        live_duration_s=60.0,
        preconditions={"all_passed": True, "checks": []},
        model_specs=[
            {"hf_id": family, "model_path": f"/{index}.gguf"}
            for index, family in enumerate(exp.REQUIRED_MODEL_IDS)
        ],
        source_artifact_hashes={"x": "sha256:" + "0" * 64},
        model_file_hashes={family: "sha256:" + "1" * 64 for family in exp.REQUIRED_MODEL_IDS},
        scoring_manifest_rows=manifest_rows,
        scoring_manifest_hash="sha256:" + "2" * 64,
        per_candidate_model_rows=feature_rows,
        raw_token_rows=raw_token_rows,
        sequence_feature_rows=feature_rows,
        parser_feature_rows=[
            {"candidate_id": row["candidate_id"], "model_id": row["model_id"]}
            for row in feature_rows
        ],
        label_denial_rows=[{"passed": True}],
        process_isolation_rows=[
            {"model_id": family, "passed": True} for family in exp.REQUIRED_MODEL_IDS
        ],
        teardown_rows=_teardown_rows(),
        vram_release_rows=_vram_rows(),
        joined_label_rows=[
            {"candidate_id": row["candidate_id"], "model_id": row["model_id"]}
            for row in feature_rows
        ],
        candidate_source_count_rows=[
            {"source_block": "all", "expected": 138, "observed": 138, "passed": True}
        ],
    )

    assert artifact["observed_feature_row_count"] == 414
    assert artifact["three_family_feature_bank_complete_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert exp.validate_artifact(artifact) == []


def test_artifact_validator_rejects_schema_types_vectors_and_verdict_prefixes() -> None:
    """SCENARIO-INF-6986-VALIDATION: every artifact integrity branch fails closed."""
    artifact = exp.build_artifact(
        run_date="20260904",
        duration_s=0.1,
        live_duration_s=0.0,
        preconditions={"all_passed": False, "checks": []},
        model_specs=[],
        source_artifact_hashes={},
        model_file_hashes={},
    )
    artifact.pop("rows")
    artifact["field_principles"] = {}
    artifact["inference_substrate"] = "synthetic"
    artifact["expected_feature_row_count"] = "414"
    artifact["observed_feature_row_count"] = "0"
    artifact["three_family_feature_bank_complete_score"] = "0"
    artifact["verifier_fit_performed"] = 0
    artifact["verifier_is_oracle"] = 0
    artifact["scoring_manifest_rows"] = [{"split": "oracle"}]
    artifact["raw_token_rows"] = [{"nested": {"full_logit_vector": [1.0]}}]
    artifact["honest_verdict"] = "wrong_prefix"
    artifact["reproducibility_checksum"] = "sha256:wrong"
    errors = exp.validate_artifact(artifact)

    assert len(errors) >= 11
    assert any(error.startswith("required_fields_missing:") for error in errors)
    assert "manifest_contains_denied_fields" in errors
    assert "full_vocabulary_vector_present" in errors
    assert "reproducibility_checksum_mismatch" in errors

    positive = deepcopy(artifact)
    positive["three_family_feature_bank_complete_score"] = 1
    positive["observed_feature_row_count"] = 0
    positive["verdict_class"] = "blocked"
    positive["honest_verdict"] = "blocked_wrong"
    positive_errors = exp.validate_artifact(positive)
    assert "positive_row_count_mismatch" in positive_errors
    assert "positive_verdict_class_mismatch" in positive_errors
    assert "positive_verdict_prefix_mismatch" in positive_errors

    partial = deepcopy(artifact)
    partial["three_family_feature_bank_complete_score"] = 0
    partial["verdict_class"] = "partial"
    partial["honest_verdict"] = "wrong_prefix"
    assert "partial_verdict_prefix_mismatch" in exp.validate_artifact(partial)


def test_gpu_memory_rows_are_normalized_for_release_receipts() -> None:
    """SCENARIO-INF-6986-VRAM: device baselines use stable bare numeric fields."""
    rows = exp._memory_rows(
        {
            "devices": [
                {"index": "1", "uuid": "GPU-b", "memory_used_mb": "12", "memory_free_mb": None}
            ]
        }
    )
    assert rows == [{"index": 1, "uuid": "GPU-b", "memory_used_mb": 12, "memory_free_mb": 0}]


def test_command_wrapper_targets_the_exp6986_module() -> None:
    """REQ-INF-6986: the required command has one stable wrapper."""
    text = (
        exp.REPO_ROOT / "scripts/experiments/experiment_6986_three_family_contrast_features.py"
    ).read_text()
    assert "carnot.experiment_6986_three_family_contrast_features import main" in text
    assert "AutoTokenizer" not in text
    assert "create_completion" not in text


def test_manifest_worker_round_trip_rejects_denied_payload(tmp_path: Path) -> None:
    """SCENARIO-INF-6986-LABEL-DENIAL: the process boundary keeps its receipt."""
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "manifest.json"
    input_path.write_text(json.dumps([_manifest_row()]))
    assert exp.manifest_worker(input_path, output_path) == 0
    manifest = json.loads(output_path.read_text())
    assert manifest["rows"][0]["candidate_id"] == "candidate-a"

    input_path.write_text(json.dumps([_manifest_row() | {"split": "held_out"}]))
    assert exp.manifest_worker(input_path, output_path) == 2
    receipt = json.loads(output_path.read_text())
    assert receipt["passed"] is False
    assert any("split" in error for error in receipt["errors"])
