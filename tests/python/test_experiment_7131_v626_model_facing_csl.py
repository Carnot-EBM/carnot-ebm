"""Tests for the model-facing fixed-schema learning loop.

Spec refs: REQ-SELF-7131 and SCENARIO-SELF-7131-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7131_v626_model_facing_csl as mod


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"


def _source_rows() -> list[dict[str, object]]:
    """Build four groups per family so every immutable split has one group."""

    rows: list[dict[str, object]] = []
    index = 0
    for family in mod.FAMILIES:
        for group_index in range(4):
            for variant in mod.VARIANTS:
                rows.append(
                    {
                        "source_index": index,
                        "cell_key": f"{QWEN}|{family}-{group_index}:{variant}",
                        "model_id": QWEN,
                        "base_id": f"{family}-{group_index}",
                        "instance_id": f"{family}-{group_index}:{variant}",
                        "family": family,
                        "variant_kind": variant,
                        "prompt": f"solve {family} {group_index} {variant}",
                        "prompt_hash": mod.sha256_text(
                            f"solve {family} {group_index} {variant}"
                        ),
                        "raw_text": '{"status":"UNSAT"}',
                        "raw_output_hash": mod.sha256_text('{"status":"UNSAT"}'),
                        "parse_success": True,
                        "parsed": {"status": "UNSAT"},
                        "exact_correct": group_index % 2 == 0,
                        "constraint_violation_count": group_index % 2,
                        "objective_matches": group_index % 2 == 0,
                        "prompt_tokens": 8,
                        "completion_tokens": 4,
                    }
                )
                index += 1
    return rows


def _model_specs(tmp_path: Path) -> list[dict[str, object]]:
    """Return the exact cached-pair shape without requiring a large test file."""

    paths = []
    for name in ("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "gemma-4-31B-it-Q4_K_M.gguf"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths.append(path)
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": QWEN,
            "gpu_indices": [0, 1],
            "model_path": str(paths[0]),
            "quantization": "Q4_K_M",
        },
        {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "gpu_indices": [0, 1],
            "model_path": str(paths[1]),
            "quantization": "Q4_K_M",
        },
    ]


def _upstream(tmp_path: Path, score: int = 1) -> Path:
    """Write a compact upstream artifact with the same causal fields as Exp7129."""

    path = tmp_path / "upstream.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "experiment_7129_v626_sota_constraint_bank",
                "sota_constraint_bank_ready_score": score,
                "rows": _source_rows(),
                "raw_trace_manifest": [],
                "verifier_is_oracle": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_call(prompt: str, seed: int, max_tokens: int) -> dict[str, object]:
    """Return one deterministic model-shaped answer for CPU-only tests."""

    del seed, max_tokens
    return {
        "raw_text": '{"status":"UNSAT"}',
        "reasoning_text": "",
        "raw_response": {"choices": [{"message": {"content": '{"status":"UNSAT"}'}}]},
        "prompt_tokens": max(1, len(prompt) // 8),
        "completion_tokens": 4,
        "duration_s": 0.01,
    }


def _complete_artifact(tmp_path: Path) -> dict[str, object]:
    """Run the complete CPU seam so mutation tests start from valid evidence."""

    specs = _model_specs(tmp_path)
    upstream = _upstream(tmp_path)
    return mod.run_experiment(
        repo_root=tmp_path,
        upstream_path=upstream,
        artifact_path=tmp_path / "result.json",
        raw_dir=tmp_path / "raw",
        run_date="20260908",
        model_specs=specs,
        preflight_func=lambda **_: mod.test_preflight(specs),
        model_call=_fake_call,
        receipt_lookup=lambda row: {
            "instance_id": row["instance_id"],
            "family": row["family"],
            "feasible": False,
            "objective": None,
            "formal": {},
            "inverse_symbol_map": {},
        },
        exact_check=lambda receipt, parsed: {
            "exact_correct": parsed.get("status") == "UNSAT",
            "constraint_violation_count": 0,
            "objective_matches": True,
            "decision": "accept",
        },
    )


def test_req_self_7131_model_spec_uses_cached_qwen_first(tmp_path: Path) -> None:
    """REQ-SELF-7131 binds the live panel to cached Qwen Q4_K_M."""

    specs = _model_specs(tmp_path)
    resolved = mod.resolve_model_specs(cached_pair_func=lambda **_: specs)
    assert resolved[0]["hf_id"] == QWEN
    assert resolved[0]["quantization"] == "Q4_K_M"
    assert resolved[0]["chat_template_source"] == "embedded_gguf"
    assert resolved[0]["model_hash"] == mod.sha256_path(Path(resolved[0]["model_path"]))
    with pytest.raises(RuntimeError, match="headline_qwen"):
        mod.resolve_model_specs(cached_pair_func=lambda **_: list(reversed(specs)))


def test_scenario_self_7131_time_rejects_same_event_write() -> None:
    """SCENARIO-SELF-7131-TIME rejects records before the source event closes."""

    store = mod.FixedSchemaMemory("fixed")
    record = mod.signed_record(
        source_row=_source_rows()[0],
        outcome={"exact_correct": False},
        decision_sequence=1,
        outcome_sequence=2,
        close_sequence=3,
        admitted_sequence=4,
        admitted_for_event_index=1,
    )
    assert store.commit(record, current_event_index=1)["committed"] is True
    same_event = mod.signed_record(
        source_row=_source_rows()[1],
        outcome={"exact_correct": True},
        decision_sequence=11,
        outcome_sequence=12,
        close_sequence=13,
        admitted_sequence=14,
        admitted_for_event_index=1,
    )
    with pytest.raises(mod.MemoryProtocolError, match="same_event_write"):
        store.commit(same_event, current_event_index=1)


def test_scenario_self_7131_leakage_rejects_hindsight() -> None:
    """SCENARIO-SELF-7131-LEAKAGE blocks exact and future fields in decisions."""

    visible = {"prompt": "solve", "memory_view": "none"}
    assert mod.seal_decision_view(visible)["decision_hash"].startswith("sha256:")
    for forbidden in ("exact_correct", "future_label", "post_event_aggregate"):
        with pytest.raises(mod.MemoryProtocolError, match="hindsight_leakage"):
            mod.seal_decision_view({**visible, forbidden: True})


def test_scenario_self_7131_budget_rejects_unequal_context(tmp_path: Path) -> None:
    """SCENARIO-SELF-7131-BUDGET gives every arm identical fixed budgets."""

    artifact = _complete_artifact(tmp_path)
    assert mod.validate_artifact(artifact) == []
    changed = deepcopy(artifact)
    changed["context_budget_rows"][0]["context_budget_bytes"] += 1
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "unequal_context_budgets" in mod.validate_artifact(changed)


def test_scenario_self_7131_source_mutation_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-SELF-7131-SOURCE binds source content and chronology."""

    artifact = _complete_artifact(tmp_path)
    changed = deepcopy(artifact)
    changed["chronological_event_rows"][0]["source_prompt_hash"] = mod.sha256_text("mutated")
    changed["rows"] = (
        changed["chronological_event_rows"]
        + changed["future_episode_rows"]
        + changed["protected_retention_rows"]
    )
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "source_stream_hash_mismatch" in mod.validate_artifact(changed)


def test_scenario_self_7131_signature_rejects_forgery() -> None:
    """SCENARIO-SELF-7131-SIGNATURE covers field and signature forgery."""

    record = mod.signed_record(
        source_row=_source_rows()[0],
        outcome={"exact_correct": False},
        decision_sequence=1,
        outcome_sequence=2,
        close_sequence=3,
        admitted_sequence=4,
        admitted_for_event_index=1,
    )
    assert mod.signature_valid(record) is True
    forged = deepcopy(record)
    forged["applicability"] = "different-family"
    assert mod.signature_valid(forged) is False
    forged["signature"] = "sha256:" + "0" * 64
    store = mod.FixedSchemaMemory("fixed")
    with pytest.raises(mod.MemoryProtocolError, match="invalid_signature"):
        store.commit(forged, current_event_index=1)


def test_scenario_self_7131_rollback_restores_parent() -> None:
    """SCENARIO-SELF-7131-ROLLBACK preserves bytes after commit failure."""

    store = mod.FixedSchemaMemory("fixed")
    record = mod.signed_record(
        source_row=_source_rows()[0],
        outcome={"exact_correct": False},
        decision_sequence=1,
        outcome_sequence=2,
        close_sequence=3,
        admitted_sequence=4,
        admitted_for_event_index=1,
    )
    before = store.state_hash()
    receipt = store.commit(record, current_event_index=1, fail_commit=True)
    assert receipt["terminal_state"] == "rolled_back"
    assert receipt["parent_restored"] is True
    assert store.state_hash() == before
    failed = deepcopy(receipt)
    failed["parent_restored"] = False
    assert mod.rollback_errors([failed]) == ["rollback_parent_not_restored"]


def test_scenario_self_7131_alias_rejects_note_memory() -> None:
    """SCENARIO-SELF-7131-ALIAS keeps free notes outside signed storage."""

    note = mod.FreeNoteMemory("notes")
    note.write("sat_logic", "remember JSON")
    fixed = mod.FixedSchemaMemory("fixed")
    fixed.records.append(note.records[0])
    with pytest.raises(mod.MemoryProtocolError, match="note_memory_alias"):
        fixed.retrieve("sat_logic", current_event_index=2)
    with pytest.raises(mod.MemoryProtocolError, match="store_alias"):
        mod.assert_private_stores(note, mod.FreeNoteMemory("notes"), fixed)


def test_scenario_self_7131_recovery_completes_or_restores() -> None:
    """SCENARIO-SELF-7131-RECOVERY closes interrupted prepared receipts."""

    store = mod.FixedSchemaMemory("fixed")
    record = mod.signed_record(
        source_row=_source_rows()[0],
        outcome={"exact_correct": False},
        decision_sequence=1,
        outcome_sequence=2,
        close_sequence=3,
        admitted_sequence=4,
        admitted_for_event_index=1,
    )
    prepared = store.commit(record, current_event_index=1, crash_after_prepare=True)
    assert prepared["terminal_state"] == "prepared"
    recovered = store.recover(prepared, record, current_event_index=1)
    assert recovered["terminal_state"] == "recovered_commit"
    assert recovered["partial_state_visible"] is False


def test_scenario_self_7131_forgetting_uses_per_unit_baseline(tmp_path: Path) -> None:
    """SCENARIO-SELF-7131-FORGETTING detects changed protected arithmetic."""

    artifact = _complete_artifact(tmp_path)
    assert artifact["model_facing_csl_complete_score"] == 1
    changed = deepcopy(artifact)
    changed["forgetting_rows"][0]["forgetting"] += 1
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "forgetting_rows_mismatch" in mod.validate_artifact(changed)


def test_req_self_7131_blocked_gate_names_exact_producer_field(tmp_path: Path) -> None:
    """REQ-SELF-7131 writes the exact upstream gate failure before GPU setup."""

    upstream = _upstream(tmp_path, score=0)
    called = False

    def forbidden_preflight(**_: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("GPU preflight must not run")

    artifact = mod.run_experiment(
        repo_root=tmp_path,
        upstream_path=upstream,
        artifact_path=tmp_path / "blocked.json",
        raw_dir=tmp_path / "raw",
        run_date="20260908",
        model_specs=_model_specs(tmp_path),
        preflight_func=forbidden_preflight,
        model_call=_fake_call,
    )
    assert called is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"] == (
        "experiment_7129_v626_sota_constraint_bank.sota_constraint_bank_ready_score"
    )
    assert artifact["gate_check_summary"]["expected_value"] == 1
    assert artifact["gate_check_summary"]["observed_value"] == 0
    assert mod.validate_artifact(artifact) == []


def test_req_self_7131_complete_null_separates_uplift(tmp_path: Path) -> None:
    """SCENARIO-SELF-7131-VERDICT keeps complete zero uplift terminal null."""

    artifact = _complete_artifact(tmp_path)
    assert artifact["model_facing_csl_complete_score"] == 1
    assert artifact["later_value_delta"] == 0.0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("null_")
    assert artifact["weights_updated"] is False
    assert artifact["same_event_writes"] == 0
    assert set(artifact["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_artifact(artifact) == []


def test_req_self_7131_validator_rejects_safety_and_verdict_drift(tmp_path: Path) -> None:
    """REQ-SELF-7131 cold validation recomputes safety and terminal claims."""

    artifact = _complete_artifact(tmp_path)

    def errors(change: object) -> list[str]:
        changed = deepcopy(artifact)
        change(changed)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        return mod.validate_artifact(changed)

    assert "forged_signature" in errors(
        lambda value: value["signature_rows"][0].__setitem__("valid", False)
    )
    assert "rollback_failure" in errors(
        lambda value: value["rollback_rows"][0].__setitem__("parent_restored", False)
    )
    assert "note_memory_alias" in errors(
        lambda value: value["memory_operation_rows"][0].__setitem__(
            "store_type", "free_note"
        )
    )
    assert "same_event_writes_mismatch" in errors(
        lambda value: value.__setitem__("same_event_writes", 1)
    )
    assert "completion_score_mismatch" in errors(
        lambda value: value.__setitem__("model_facing_csl_complete_score", 0)
    )
    assert "verdict_class_mismatch" in errors(
        lambda value: value.__setitem__("verdict_class", "positive")
    )
    assert "weights_updated" in errors(
        lambda value: value.__setitem__("weights_updated", True)
    )


def test_req_self_7131_write_is_atomic(tmp_path: Path) -> None:
    """REQ-SELF-7131 publishes complete artifacts through one replacement."""

    path = tmp_path / "nested" / "artifact.json"
    mod.write_artifact(path, {"complete": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"complete": True}

