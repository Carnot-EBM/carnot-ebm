"""Tests for Exp6345 prospective certified factor evolution A/B.

Spec refs: REQ-LEARN-6345, SCENARIO-LEARN-6345-GATE-REPLAY,
SCENARIO-LEARN-6345-SEALS, SCENARIO-LEARN-6345-MODELS,
SCENARIO-LEARN-6345-MATCHED-ARMS,
SCENARIO-LEARN-6345-RELEASE-LIFECYCLE,
SCENARIO-LEARN-6345-READY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_6345_prospective_certified_factor_evolution_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _fake_model_paths(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + ".Q4_K_M.gguf")
        path.write_bytes((model_id + "\n").encode("utf-8"))
        paths[model_id] = path
    return paths


def _fake_cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def _cached_pair(
        *,
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "gpu_indices": gpu_indices,
                "preferred_quant": preferred_quant,
                "model_indices": model_indices,
            }
        )
        if model_indices == (0, 2):
            return [
                {
                    "name": "Qwen3.6-35B-A3B",
                    "hf_id": mod.MANDATED_MODEL_IDS[0],
                    "gpu": gpu_indices[0],
                    "model_path": str(paths[mod.MANDATED_MODEL_IDS[0]]),
                },
                {
                    "name": "Gemma4-31B-it",
                    "hf_id": mod.MANDATED_MODEL_IDS[1],
                    "gpu": gpu_indices[1],
                    "model_path": str(paths[mod.MANDATED_MODEL_IDS[1]]),
                },
            ]
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": mod.MANDATED_MODEL_IDS[0],
                "gpu": gpu_indices[0],
                "model_path": str(paths[mod.MANDATED_MODEL_IDS[0]]),
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": mod.MANDATED_MODEL_IDS[2],
                "gpu": gpu_indices[1],
                "model_path": str(paths[mod.MANDATED_MODEL_IDS[2]]),
            },
        ]

    return _cached_pair


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _fake_model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    return mod.run(
        date="20260812",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "data",
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        cached_pair_func=_fake_cached_pair(paths, calls),
        tokenizer_func=lambda path: (path.endswith(".gguf"), "embedded ok"),
        host_checks_func=mod.deterministic_host_receipts,
        write=write,
    )


def _read_json(receipt: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(str(receipt["path"])).read_text(encoding="utf-8"))


def _read_jsonl(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(receipt["path"]))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_req_learn_6345_spec_declares_contract() -> None:
    """REQ-LEARN-6345: OpenSpec owns the artifact fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6345") :]
    for token in (
        "SCENARIO-LEARN-6345-GATE-REPLAY",
        "SCENARIO-LEARN-6345-SEALS",
        "SCENARIO-LEARN-6345-MODELS",
        "SCENARIO-LEARN-6345-MATCHED-ARMS",
        "SCENARIO-LEARN-6345-RELEASE-LIFECYCLE",
        "SCENARIO-LEARN-6345-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_6345_model_specs_use_cached_pair_and_llama_tokenizer(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6345-MODELS: GGUF tokenizers are embedded llama.cpp checks."""

    paths = _fake_model_paths(tmp_path)
    calls: list[dict[str, Any]] = []
    resolution = mod.build_model_specs(
        cached_pair_func=_fake_cached_pair(paths, calls),
        tokenizer_func=lambda path: (path.endswith(".gguf"), "embedded ok"),
    )

    assert resolution["all_resolved"] is True
    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolution["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert all(row["tokenizer_method"] == mod.TOKENIZER_METHOD for row in resolution["MODEL_SPECS"])
    assert all(row["tokenizer_loadable"] is True for row in resolution["MODEL_SPECS"])
    assert mod.AUTOTOKENIZER_USAGE_COUNT == 0

    missing = mod.build_model_specs(
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda _: (False, "not checked"),
    )
    assert missing["all_resolved"] is False
    assert "cached_sota_pair_missing" in missing["blocked_reasons"]


def test_scenario_learn_6345_seals_chronology_and_no_peeking(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6345-SEALS: outcomes stay sealed until postdecision open."""

    artifact = _artifact(tmp_path)
    registration = _read_json(artifact["prospective_registration_path_and_hash"])
    chronological = _read_json(artifact["sealed_chronological_stream_manifest_path_and_hash"])
    protected = _read_json(artifact["sealed_protected_validation_manifest_path_and_hash"])
    snapshots = _read_jsonl(artifact["predecision_snapshot_paths_hashes_and_counts"])
    outcomes = _read_jsonl(artifact["postdecision_outcome_paths_hashes_and_counts"])

    assert registration["freeze_before_any_event_outcome_read"] is True
    assert chronological["sealed_before_stream_open"] is True
    assert protected["sealed_until_all_arms_stop"] is True
    assert [event["time_index"] for event in chronological["events"]] == list(
        range(len(mod.chronological_events()))
    )
    learner_visible = json.dumps(chronological["events"], sort_keys=True)
    assert "exact_outcome" not in learner_visible
    assert "protected_label" not in learner_visible
    assert "hidden_state" not in learner_visible

    assert artifact["protected_outcome_seal_and_single_open_receipt"]["open_count"] == 1
    assert artifact["protected_outcome_seal_and_single_open_receipt"]["opened_after_all_arms_stopped"] is True
    assert len(snapshots) == len(outcomes)
    for snapshot in snapshots:
        payload = json.dumps(snapshot, sort_keys=True)
        assert "exact_outcome" not in payload
        assert "protected_label" not in payload
        assert snapshot["written_before_outcome_reveal"] is True


def test_scenario_learn_6345_matched_budgets_release_lifecycle_and_rollback(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6345-RELEASE-LIFECYCLE: release and rollback are bounded."""

    artifact = _artifact(tmp_path)
    budgets = artifact["matched_call_token_candidate_time_checker_state_and_memory_budgets"]
    ledger = _read_json(artifact["eprocess_release_ledger_path_and_hash"])
    registry = _read_jsonl(artifact["version_and_lifecycle_registry_paths_hashes"])

    assert set(artifact["arm_definitions"]["arms"]) == set(mod.ARMS)
    assert budgets["budget_parity"] is True
    first_budget = budgets["by_arm"][mod.ARMS[0]]
    for arm in mod.ARMS:
        assert budgets["by_arm"][arm] == first_budget
    assert ledger["false_release_count"] == 0
    assert ledger["release_authority"] == "exact_checker_eprocess"
    assert all(row["approved_by_arm_it_updates"] is False for row in ledger["rows"])
    assert artifact["unsafe_commit_count"] == 0
    assert artifact["rollback_byte_identity"]["byte_identical"] is True
    assert artifact["rollback_byte_identity"]["rollback_count"] >= 1
    assert registry[-1]["capacity_bound"] == mod.FACTOR_CAPACITY
    assert registry[-1]["max_factor_count"] <= mod.FACTOR_CAPACITY


def test_scenario_learn_6345_ready_metrics_and_fail_closed_controls(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6345-READY: readiness is conjunctive and disaggregated."""

    artifact = _artifact(tmp_path)
    deltas = artifact["future_same_family_and_held_family_paired_deltas_intervals_and_sample_sizes"]
    cells = artifact[
        "exact_yield_false_release_rollback_factor_growth_and_latency_by_model_family_arm_and_time"
    ]

    assert artifact["certified_continuous_learning_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["verifier_is_oracle"] is True
    for field in (
        "protected_validation_leak_count",
        "unsafe_commit_count",
        "source_model_weight_mutation_count",
        "generated_label_count",
        "hidden_state_access_count",
    ):
        assert type(artifact[field]) is int
        assert artifact[field] == 0
    assert cells["cell_count"] == len(mod.MODEL_FAMILIES) * len(mod.ARMS) * len(
        mod.chronological_events()
    )
    for family, partitions in deltas["by_model_family"].items():
        assert family in mod.MODEL_FAMILIES
        for partition in ("future_same_family", "held_family"):
            assert partitions[partition]["delta_vs_frozen_lower"] > 0.0
            assert partitions[partition]["delta_vs_fixed_v544_lower"] > 0.0

    failing_tests = dict(_passing_exit_codes())
    failing_tests[mod.FOCUSED_TEST_COMMAND] = 1
    failed = mod.run(
        date="20260812",
        result_path=tmp_path / "failed.json",
        data_dir=tmp_path / "failed-data",
        duration_s=1.0,
        test_exit_codes=failing_tests,
        cached_pair_func=_fake_cached_pair(_fake_model_paths(tmp_path / "failed-models"), []),
        tokenizer_func=lambda path: (path.endswith(".gguf"), "embedded ok"),
        host_checks_func=mod.deterministic_host_receipts,
        write=False,
    )
    assert failed["certified_continuous_learning_ready_score"] == 0.0
    assert failed["status"] == "complete_null"

    blocked = mod.run(
        date="20260812",
        result_path=tmp_path / "blocked.json",
        data_dir=tmp_path / "blocked-data",
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        cached_pair_func=lambda **_: None,
        tokenizer_func=lambda _: (False, "not checked"),
        host_checks_func=mod.deterministic_host_receipts,
        write=False,
    )
    assert blocked["certified_continuous_learning_ready_score"] == 0.0
    assert blocked["status"] == "blocked_precondition"
    assert blocked["honest_verdict"].startswith("blocked:")


def test_req_learn_6345_artifact_schema_provenance_and_checksum(tmp_path: Path) -> None:
    """REQ-LEARN-6345: artifact fields, principles, oracle, and checksum validate."""

    artifact = _artifact(tmp_path)

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["exact_oracle_claim_boundary"]["verifier_is_oracle"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["protected_validation_leak_count"] == 0
    assert artifact["unsafe_commit_count"] == 0
    mod.validate_artifact(artifact)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_learn_6345_defensive_receipts_and_classifiers(tmp_path: Path) -> None:
    """REQ-LEARN-6345: defensive helpers fail closed for receipt edge cases."""

    assert mod.sha256_file(tmp_path / "missing.json") is None
    try:
        mod.require(False, "expected_failure")
    except ValueError as exc:
        assert str(exc) == "expected_failure"
    assert mod.quantization_from_path(Path("model-without-token.gguf")) == "unknown"
    assert mod.terminal_class("blocked_precondition", "") == "terminal_blocked"
    assert mod.terminal_class("complete_null", "") == "terminal_null"
    assert mod.terminal_class("other", "") == "terminal_unknown"
    assert mod._test_exit_codes(None, ["cmd"]) == {"cmd": 0}

    harmful_deltas = {
        "by_model_family": {
            "qwen_moe": {
                "future_same_family": {"n": 1, "positive_over_both_controls": False}
            }
        }
    }
    summary = mod.harm_summary(
        {"MODEL_SPECS": [{"hf_id": "missing", "exists": False, "tokenizer_loadable": False}]},
        harmful_deltas,
        {"complete_required_cells": False},
    )
    assert summary["missing_model_cells"] == ["missing"]
    assert summary["underpowered_cells"] == [
        {"model_family": "qwen_moe", "partition": "future_same_family"}
    ]
    assert summary["harmful_cells"] == [
        {"model_family": "qwen_moe", "partition": "future_same_family"}
    ]
    assert summary["harm_detected"] is True
