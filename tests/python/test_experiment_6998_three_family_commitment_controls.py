"""Tests for the three-family self-commitment shortcut controls.

Spec refs: REQ-INF-6998 and SCENARIO-INF-6998-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6998_three_family_commitment_controls as exp


class FakeLlama:
    """Provide deterministic embedded-tokenizer and logit behavior for tests."""

    def __init__(self) -> None:
        self.scores = np.zeros((0, 32), dtype=np.float64)
        self.reset_calls = 0

    def tokenize(self, value: bytes, *, add_bos: bool, special: bool = True) -> list[int]:
        del special
        tokens = [2 + byte % 30 for byte in value]
        return ([1] if add_bos else []) + tokens

    def detokenize(self, tokens: list[int]) -> bytes:
        return bytes((token - 2) % 30 for token in tokens if token != 1)

    def reset(self) -> None:
        self.reset_calls += 1

    def eval(self, tokens: list[int]) -> None:
        rows = []
        for position, _token in enumerate(tokens):
            row = np.linspace(-1.0, 1.0, 32, dtype=np.float64)
            row[(position + 7) % 32] += 2.0
            rows.append(row)
        self.scores = np.asarray(rows)


def _candidate(candidate_id: str, pair_id: str, position: int, text: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "contrast_group_id": pair_id,
        "pair_position": position,
        "serialized_candidate": text,
        "serialization_hash": exp.sha256_text(text),
        "split": "held_out",
        "formulation_family": "bounded_integer_linear",
    }


def _pair_manifest(pair_count: int = 2) -> list[dict]:
    candidates = [
        _candidate(
            f"candidate-{pair}-{position}", f"pair-{pair}", position, f"text-{pair}-{position}"
        )
        for pair in range(pair_count)
        for position in range(2)
    ]
    return exp.freeze_pair_manifest(candidates, expected_pair_count=pair_count)["rows"]


def _unit_row(candidate_id: str, condition: str, family: str) -> dict:
    return {
        "pair_id": candidate_id.rsplit("-", 1)[0],
        "candidate_id": candidate_id,
        "candidate_hash": exp.sha256_text(candidate_id),
        "condition": condition,
        "model_id": family,
        "terminal": True,
        "terminal_choice": "VALID",
        "choice_extraction_passed": True,
        "prefix_order_passed": True,
        "tokenizer_parity_passed": True,
        "live_cuda": True,
        "prompt_hash": exp.sha256_text(f"{candidate_id}:{condition}"),
        "curve_hash": exp.sha256_text(f"curve:{candidate_id}:{condition}:{family}"),
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


def test_req_inf_6998_spec_precedes_implementation() -> None:
    """REQ-INF-6998: the OpenSpec contract exists before implementation."""
    spec = (exp.REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md").read_text()
    assert "### REQ-INF-6998:" in spec
    for scenario in (
        "GATES",
        "FREEZE",
        "DENIAL",
        "CONDITIONS",
        "TOKENIZER",
        "CHOICE",
        "FAMILIES",
        "CHECKPOINT",
        "TEARDOWN",
        "BARE",
    ):
        assert f"SCENARIO-INF-6998-{scenario}" in spec


def test_req_inf_6998_model_specs_start_with_cached_pair(tmp_path: Path) -> None:
    """REQ-INF-6998: only the exact three cached GGUF families are eligible."""
    paths = {}
    for index, family in enumerate(exp.REQUIRED_MODEL_IDS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(b"GGUF")
        paths[family] = str(path)
    calls: list[tuple[int, int]] = []

    def pair(*, gpu_indices: tuple[int, int]) -> list[dict]:
        calls.append(gpu_indices)
        return [
            {"hf_id": family, "model_path": paths[family]} for family in exp.REQUIRED_MODEL_IDS[:2]
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


def test_scenario_inf_6998_freeze_selects_actual_12_source_disjoint_pairs() -> None:
    """SCENARIO-INF-6998-FREEZE: labels are not needed to freeze all held-out pairs."""
    source = json.loads(exp.EXP6984_PATH.read_text(encoding="utf-8"))
    frozen = exp.freeze_pair_manifest(source["per_candidate_rows"])

    assert len(frozen["rows"]) == 24
    assert len({row["pair_id"] for row in frozen["rows"]}) == 12
    assert len({row["source_group_id"] for row in frozen["rows"]}) == 12
    assert all(row["selected_before_label_open"] is True for row in frozen["rows"])
    assert all(
        "exact_label" not in row and "certified_relation" not in row for row in frozen["rows"]
    )
    assert frozen["pair_manifest_hash"].startswith("sha256:")


def test_scenario_inf_6998_freeze_rejects_overlap_hash_drift_and_bad_pairs() -> None:
    """SCENARIO-INF-6998-FREEZE: roster errors fail before any label opens."""
    rows = [
        _candidate("a0", "pair-a", 0, "a0"),
        _candidate("a1", "pair-a", 1, "a1"),
        _candidate("b0", "pair-b", 0, "b0"),
        _candidate("b1", "pair-b", 1, "b1"),
    ]
    rows[0]["source_group_id"] = "shared"
    rows[1]["source_group_id"] = "shared"
    rows[2]["source_group_id"] = "shared"
    rows[3]["source_group_id"] = "shared"
    with pytest.raises(exp.ManifestError, match="source_group_overlap"):
        exp.freeze_pair_manifest(rows, expected_pair_count=2)

    rows[2]["source_group_id"] = rows[3]["source_group_id"] = "other"
    rows[0]["serialization_hash"] = "sha256:drift"
    with pytest.raises(exp.ManifestError, match="candidate_hash_mismatch"):
        exp.freeze_pair_manifest(rows, expected_pair_count=2)

    with pytest.raises(exp.ManifestError, match="pair_count"):
        exp.freeze_pair_manifest(rows[:2], expected_pair_count=2)


@pytest.mark.parametrize("field", sorted(exp.FORBIDDEN_MODEL_FIELDS))
def test_scenario_inf_6998_denial_rejects_every_nested_authority_field(field: str) -> None:
    """SCENARIO-INF-6998-DENIAL: model payloads reject authority data at any depth."""
    errors = exp.model_input_errors({"candidate_id": "c", "nested": {field: "secret"}})
    assert any(field in error for error in errors)


def test_scenario_inf_6998_conditions_build_true_and_deranged_decoy_hints() -> None:
    """SCENARIO-INF-6998-CONDITIONS: clean, true, and decoy prompts stay distinct."""
    manifest = _pair_manifest(3)
    provenance = {
        row["candidate_id"]: {
            "pair_id": row["pair_id"],
            "pair_position": row["pair_position"],
            "provenance_text": f"provenance-{row['candidate_id']}",
        }
        for row in manifest
    }
    conditions = exp.build_condition_manifest(manifest, provenance)
    derangement = conditions["derangement"]

    assert set(derangement) == {"pair-0", "pair-1", "pair-2"}
    assert all(source != target for source, target in derangement.items())
    assert len(conditions["rows"]) == 18
    for candidate in manifest:
        rows = [
            row for row in conditions["rows"] if row["candidate_id"] == candidate["candidate_id"]
        ]
        prompts = {row["condition"]: row["prompt_text"] for row in rows}
        assert "provenance-" not in prompts["clean"]
        assert (
            provenance[candidate["candidate_id"]]["provenance_text"]
            in prompts["true_provenance_hint"]
        )
        assert (
            provenance[candidate["candidate_id"]]["provenance_text"]
            not in prompts["permuted_decoy_hint"]
        )
        assert all("equivalent" not in prompt.casefold() for prompt in prompts.values())


def test_scenario_inf_6998_conditions_reject_label_text_and_bad_provenance() -> None:
    """SCENARIO-INF-6998-CONDITIONS: hints cannot contain explicit labels or key drift."""
    manifest = _pair_manifest(2)
    provenance = {
        row["candidate_id"]: {
            "pair_id": row["pair_id"],
            "pair_position": row["pair_position"],
            "provenance_text": "The exact label is invalid.",
        }
        for row in manifest
    }
    with pytest.raises(exp.ManifestError, match="exact_label_text_in_hint"):
        exp.build_condition_manifest(manifest, provenance)

    provenance.pop(manifest[0]["candidate_id"])
    with pytest.raises(exp.ManifestError, match="provenance_key_mismatch"):
        exp.build_condition_manifest(manifest, provenance)


def test_scenario_inf_6998_tokenizer_freezes_ordered_candidate_prefixes() -> None:
    """SCENARIO-INF-6998-TOKENIZER: prefix order and full-token parity replay."""
    model = FakeLlama()
    rows = exp.freeze_token_prefixes(model, "abcdef")

    assert [row["fraction"] for row in rows] == list(exp.PREFIX_FRACTIONS)
    assert len(rows) <= 8
    assert [row["prefix_token_count"] for row in rows] == sorted(
        row["prefix_token_count"] for row in rows
    )
    assert rows[-1]["prefix_token_ids"] == model.tokenize(b"abcdef", add_bos=False, special=True)
    assert all(row["tokenizer_source"] == "embedded_gguf" for row in rows)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("analysis\nFINAL_CHOICE: VALID", "VALID"),
        ("analysis\nFINAL_CHOICE: INVALID\n", "INVALID"),
    ],
)
def test_scenario_inf_6998_choice_extracts_one_terminal_marker(text: str, expected: str) -> None:
    """SCENARIO-INF-6998-CHOICE: one terminal marker yields the model's own choice."""
    assert exp.extract_terminal_choice(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "VALID",
        "FINAL_CHOICE: VALID trailing",
        "FINAL_CHOICE: VALID\nFINAL_CHOICE: INVALID",
        "FINAL_CHOICE: MAYBE",
    ],
)
def test_scenario_inf_6998_choice_rejects_ambiguous_or_nonterminal_text(text: str) -> None:
    """SCENARIO-INF-6998-CHOICE: unclear choices fail closed."""
    with pytest.raises(exp.ChoiceError):
        exp.extract_terminal_choice(text)


def test_scenario_inf_6998_tokenizer_teacher_forces_choice_without_vectors() -> None:
    """SCENARIO-INF-6998-TOKENIZER: both terminal choices use exact embedded tokens."""
    model = FakeLlama()
    result = exp.score_terminal_choices(
        model,
        context_text="prompt-prefix\nFINAL_CHOICE:",
        own_choice="VALID",
        prefix_ordinal=0,
    )

    assert result["tokenizer_parity_passed"] is True
    assert 0.0 <= result["own_choice_probability"] <= 1.0
    assert result["valid_probability"] + result["invalid_probability"] == pytest.approx(1.0)
    assert result["uncertainty"] >= 0.0
    assert result["choice_token_rows"]
    assert all(
        "logits" not in row and "full_vocabulary_vector" not in row
        for row in result["choice_token_rows"]
    )
    assert all(
        row["full_logit_vector_hash"].startswith("sha256:") for row in result["choice_token_rows"]
    )


def test_req_inf_6998_curve_metrics_use_frozen_threshold_and_flip_counts() -> None:
    """REQ-INF-6998: curve summaries preserve latency, range, mass, flips, and uncertainty."""
    curve = [
        {
            "fraction": 0.125,
            "own_choice_probability": 0.55,
            "predicted_choice": "VALID",
            "uncertainty": 0.69,
        },
        {
            "fraction": 0.5,
            "own_choice_probability": 0.45,
            "predicted_choice": "INVALID",
            "uncertainty": 0.68,
        },
        {
            "fraction": 0.75,
            "own_choice_probability": 0.85,
            "predicted_choice": "VALID",
            "uncertainty": 0.42,
        },
        {
            "fraction": 1.0,
            "own_choice_probability": 0.90,
            "predicted_choice": "VALID",
            "uncertainty": 0.32,
        },
    ]
    metrics = exp.summarize_curve(curve)

    assert metrics["first_commitment_latency"] == 0.75
    assert metrics["commitment_range"] == pytest.approx(0.45)
    assert metrics["mean_uncommitted_mass"] == pytest.approx(1 - np.mean([0.55, 0.45, 0.85, 0.90]))
    assert metrics["choice_flip_count"] == 2
    assert metrics["mean_uncertainty"] == pytest.approx(np.mean([0.69, 0.68, 0.42, 0.32]))

    never = exp.summarize_curve(curve[:2])
    assert never["first_commitment_latency"] == 1.0
    assert never["commitment_reached"] is False


def test_scenario_inf_6998_checkpoint_replays_and_rejects_drift(tmp_path: Path) -> None:
    """SCENARIO-INF-6998-CHECKPOINT: resume keeps unique pair-family blocks."""
    path = tmp_path / "checkpoint.json"
    rows = [_unit_row("candidate-0-0", "clean", exp.REQUIRED_MODEL_IDS[0])]
    receipt = exp.write_checkpoint(
        path,
        manifest_hash="sha256:manifest",
        block_key="pair-0::family-0",
        rows=rows,
    )
    loaded = exp.load_checkpoint(path, manifest_hash="sha256:manifest")

    assert receipt["passed"] is True
    assert loaded["blocks"][0]["rows"] == rows
    repeated = exp.write_checkpoint(
        path,
        manifest_hash="sha256:manifest",
        block_key="pair-0::family-0",
        rows=rows,
    )
    assert repeated["recovered"] is True
    with pytest.raises(exp.CheckpointError, match="manifest_hash_mismatch"):
        exp.load_checkpoint(path, manifest_hash="sha256:other")
    with pytest.raises(exp.CheckpointError, match="checkpoint_block_mismatch"):
        exp.write_checkpoint(
            path,
            manifest_hash="sha256:manifest",
            block_key="pair-0::family-0",
            rows=[rows[0] | {"terminal_choice": "INVALID"}],
        )


def test_scenario_inf_6998_families_require_exact_216_cuda_rows() -> None:
    """SCENARIO-INF-6998-FAMILIES: only the exact terminal cross product completes."""
    manifest = _pair_manifest(12)
    prompt_hashes = {
        (row["candidate_id"], condition): exp.sha256_text(f"{row['candidate_id']}:{condition}")
        for row in manifest
        for condition in exp.CONDITIONS
    }
    rows = [
        _unit_row(candidate["candidate_id"], condition, family)
        | {
            "candidate_hash": candidate["candidate_hash"],
            "prompt_hash": prompt_hashes[(candidate["candidate_id"], condition)],
        }
        for candidate in manifest
        for condition in exp.CONDITIONS
        for family in exp.REQUIRED_MODEL_IDS
    ]
    assert (
        exp.completion_errors(
            manifest,
            rows,
            _teardown_rows(),
            _vram_rows(),
            label_denial_passed=True,
        )
        == []
    )

    mutated = deepcopy(rows)
    mutated[0]["live_cuda"] = False
    assert "cuda_incomplete" in exp.completion_errors(
        manifest, mutated, _teardown_rows(), _vram_rows(), label_denial_passed=True
    )
    assert "unit_row_count" in exp.completion_errors(
        manifest, rows[:-1], _teardown_rows(), _vram_rows(), label_denial_passed=True
    )
    duplicated = deepcopy(rows)
    duplicated[-1] = deepcopy(duplicated[0])
    assert "duplicate_unit_key" in exp.completion_errors(
        manifest, duplicated, _teardown_rows(), _vram_rows(), label_denial_passed=True
    )


def test_scenario_inf_6998_teardown_controls_late_label_join() -> None:
    """SCENARIO-INF-6998-TEARDOWN: labels open only after all owned workers exit."""
    evidence = [_unit_row("candidate-0-0", "clean", family) for family in exp.REQUIRED_MODEL_IDS]
    before = deepcopy(evidence)
    labels = {"candidate-0-0": "equivalent"}
    with pytest.raises(exp.LabelJoinError, match="family_processes_not_exited"):
        exp.join_labels(evidence, labels, _teardown_rows()[:-1])

    joined = exp.join_labels(evidence, labels, _teardown_rows())
    assert evidence == before
    assert all(row["exact_label"] == "equivalent" for row in joined)
    assert all(row["joined_after_all_model_processes_exit"] is True for row in joined)


def test_req_inf_6998_shortcut_gate_requires_true_only_negative_interval() -> None:
    """REQ-INF-6998: decoy replication prevents a shortcut-positive score."""
    positive = [
        {"comparison": "true_provenance_hint_minus_clean", "ci_low": -0.3, "ci_high": -0.1},
        {"comparison": "permuted_decoy_hint_minus_clean", "ci_low": -0.1, "ci_high": 0.2},
    ]
    assert exp.shortcut_detected(positive) == 1
    replicated = deepcopy(positive)
    replicated[1]["ci_high"] = -0.01
    assert exp.shortcut_detected(replicated) == 0
    null = deepcopy(positive)
    null[0]["ci_high"] = 0.0
    assert exp.shortcut_detected(null) == 0


def test_scenario_inf_6998_gates_build_complete_blocked_schema() -> None:
    """SCENARIO-INF-6998-GATES: blocked artifacts retain all required fields."""
    artifact = exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.2,
        live_duration_s=0.0,
        preconditions={
            "all_passed": False,
            "checks": [exp.gate_check("cuda_device_count", 2, 1)],
        },
        model_specs=[],
        source_artifact_hashes={},
        model_file_hashes={},
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_three_family_commitment_controls"
    assert artifact["gate_check_summary"]["failed_check"] == "cuda_device_count"
    assert artifact["gate_check_summary"]["expected_value"] == 2
    assert artifact["gate_check_summary"]["observed_value"] == 1
    assert exp.validate_artifact(artifact) == []


def test_scenario_inf_6998_bare_fields_and_policy_flags() -> None:
    """SCENARIO-INF-6998-BARE: completion and policy fields stay bare values."""
    artifact = exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        live_duration_s=0.5,
        preconditions={"all_passed": True, "checks": []},
        model_specs=[],
        source_artifact_hashes={},
        model_file_hashes={},
    )
    for field in (
        "expected_unit_count",
        "observed_unit_count",
        "commitment_control_complete_score",
        "shortcut_commitment_detected_score",
    ):
        assert type(artifact[field]) is int
    assert artifact["audit_only_control"] is True
    assert artifact["learner_feature_allowed"] is False
    assert artifact["verifier_fit_performed"] is False
    assert artifact["self_commitment_paper_reproduction_claimed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("partial_")
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)


def test_req_inf_6998_exp6997_allowlist_has_no_commitment_feature() -> None:
    """REQ-INF-6998: the audit control never becomes an Exp6997 learner feature."""
    artifact = json.loads(exp.EXP6997_PATH.read_text(encoding="utf-8"))
    assert not any("commitment" in field.casefold() for field in artifact["feature_allowlist"])


def test_req_inf_6998_wrapper_targets_module() -> None:
    """REQ-INF-6998: the required dated command has a thin executable wrapper."""
    wrapper = (
        exp.REPO_ROOT / "scripts/experiments/experiment_6998_three_family_commitment_controls.py"
    )
    source = wrapper.read_text(encoding="utf-8")
    assert "carnot.experiment_6998_three_family_commitment_controls" in source
    assert "main" in source
