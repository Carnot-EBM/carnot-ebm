"""Tests for Exp5936 SOTA atomic support union A/B.

Spec refs: REQ-VERIFY-5936, SCENARIO-VERIFY-5936-GATE,
SCENARIO-VERIFY-5936-PREREG, SCENARIO-VERIFY-5936-UNION,
SCENARIO-VERIFY-5936-PRIMARY, SCENARIO-VERIFY-5936-EVENTS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5935_non_pruning_atomic_constraint_support as exp5935
from carnot import experiment_5936_sota_atomic_support_union_ab as exp5936


def _fake_model_files(root: Path) -> dict[str, str]:
    model_dir = root / "models"
    model_dir.mkdir(parents=True)
    paths: dict[str, str] = {}
    for index, hf_id in enumerate(exp5936.MANDATED_MODEL_IDS):
        path = model_dir / f"model-{index}.Q4_K_M.gguf"
        path.write_text(f"fake gguf for {hf_id}\n", encoding="utf-8")
        paths[hf_id] = str(path)
    return paths


def _model_resolver(paths: dict[str, str]) -> Any:
    def resolver() -> list[dict[str, Any]]:
        return [
            {
                **spec,
                "gpu": index % 2,
                "model_path": paths[str(spec["hf_id"])],
                "resolved_via": (
                    "cached_sota_pair"
                    if index < 2
                    else "resolve_cached_gguf_cached_third_family"
                ),
            }
            for index, spec in enumerate(exp5936.MODEL_SPECS)
        ]

    return resolver


def _passing_environment() -> dict[str, Any]:
    return {
        "llama_cpp_import": {"ok": True, "detail": "import_ok"},
        "public_llama_cpp_cuda": {
            "ok": True,
            "logits_processor_parameter": True,
            "gpu_offload_supported": True,
        },
        "gpu_health": {
            "ok": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 23000,
                    "utilization_gpu_pct": 0,
                },
                {
                    "index": 1,
                    "name": "NVIDIA GeForce RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 22900,
                    "utilization_gpu_pct": 0,
                },
            ],
        },
        "ram": {"ok": True, "available_mb": 131072, "required_mb": 32768},
        "disk": {"ok": True, "available_mb": 100000, "required_mb": 8192},
        "protected_workload": {"ok": True, "protected_pids": []},
        "atomic_output": {"ok": True, "detail": "os.replace"},
        "atomic_checkpoint_resume": {"ok": True, "detail": "resume_probe_ok"},
    }


def _tokenizer_loader(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "hf_id": spec["hf_id"],
        "model_path": spec["model_path"],
        "embedded_tokenizer_only": True,
        "used_hf_autotokenizer": False,
        "n_vocab": 128,
    }


def _ready_gate(root: Path) -> dict[str, Any]:
    source = exp5936.REPO_ROOT / exp5935.RESULT_RELATIVE_PATH
    return {
        "ok": True,
        "artifact_path": str(root / exp5935.RESULT_RELATIVE_PATH),
        "expected_source_path": str(source),
        "path_exact": True,
        "artifact_sha256": exp5936.sha256_file(source),
        "atom_support_fixture_ready_score": 1.0,
        "honest_verdict": "complete_ready: fixture ready",
    }


def _not_ready_gate(root: Path) -> dict[str, Any]:
    receipt = _ready_gate(root)
    receipt["ok"] = False
    receipt["atom_support_fixture_ready_score"] = 0.0
    receipt["block_reason"] = "atom_support_fixture_ready_score_not_1"
    return receipt


def _proposal_ids(case: dict[str, Any], *, complete: bool) -> list[str]:
    schema = exp5935.versioned_atom_schema()
    surface = exp5936.derive_surface_for_case(case, schema)
    reference = [str(atom["atom_id"]) for atom in surface["_hidden_reference_atoms"]]
    if complete or len(reference) < 2:
        return reference
    omitted = str(exp5935._first_dynamic_reference_id(surface["_hidden_reference_atoms"]))
    return [atom_id for atom_id in reference if atom_id != omitted]


def _raw_row(
    sequence: int,
    spec: dict[str, Any],
    case: dict[str, Any],
    arm_id: str,
    view_id: str,
    proposal_atom_ids: list[str],
) -> dict[str, Any]:
    return {
        "stream_sequence_index": sequence,
        "model_hf_id": spec["hf_id"],
        "model_name": spec["name"],
        "model_path": spec["model_path"],
        "gpu_index": spec["gpu"],
        "case_id": case["case_id"],
        "arm_id": arm_id,
        "view_id": view_id,
        "raw_output_text": json.dumps({"atom_ids": proposal_atom_ids}, sort_keys=True),
        "proposal_atom_ids": proposal_atom_ids,
        "latency_s": 0.2,
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        "gpu_telemetry": {
            "average_gpu_utilization_pct": 42,
            "vram_delta_mb": 2048,
            "offload_verified": True,
        },
    }


def _collector_with_transformed_gain(
    model_specs: list[dict[str, Any]],
    panel: list[dict[str, Any]],
    config: exp5936.ExperimentConfig,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    del config, preregistration
    rows: list[dict[str, Any]] = []
    sequence = 0
    for spec in model_specs:
        for case in panel:
            for arm_id in exp5936.ARM_IDS:
                for view_id in exp5936.view_ids_for_arm(arm_id):
                    complete = (
                        arm_id == "transformed_view_atomic_union"
                        and view_id == "entity_permutation"
                    )
                    proposal_ids = _proposal_ids(case, complete=complete)
                    rows.append(_raw_row(sequence, spec, case, arm_id, view_id, proposal_ids))
                    sequence += 1
    return {
        "rows": rows,
        "real_model_rows": True,
        "model_attempts": [
            {
                "hf_id": spec["hf_id"],
                "model_used": True,
                "gpu_offload_verified": True,
                "vram_delta_mb": 2048,
            }
            for spec in model_specs
        ],
        "gpu_receipts": {"mode": "stubbed_real_path_contract"},
    }


def _collector_all_null(
    model_specs: list[dict[str, Any]],
    panel: list[dict[str, Any]],
    config: exp5936.ExperimentConfig,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    del config, preregistration
    rows: list[dict[str, Any]] = []
    sequence = 0
    for spec in model_specs:
        for case in panel:
            for arm_id in exp5936.ARM_IDS:
                for view_id in exp5936.view_ids_for_arm(arm_id):
                    rows.append(
                        _raw_row(
                            sequence,
                            spec,
                            case,
                            arm_id,
                            view_id,
                            _proposal_ids(case, complete=False),
                        )
                    )
                    sequence += 1
    return {"rows": rows, "real_model_rows": True, "model_attempts": [], "gpu_receipts": {}}


def test_gate_blocks_before_model_rows_when_exp5935_is_not_ready(tmp_path: Path) -> None:
    # REQ-VERIFY-5936, SCENARIO-VERIFY-5936-GATE
    paths = _fake_model_files(tmp_path)

    def should_not_collect(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("collector must not run after Exp5935 gate failure")

    artifact = exp5936.run_experiment(
        exp5936.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 1.0),
        model_resolver=_model_resolver(paths),
        environment_probe=lambda root: _passing_environment(),
        tokenizer_loader=_tokenizer_loader,
        gate_replay_provider=_not_ready_gate,
        collect_model_outputs_fn=should_not_collect,
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["preconditions_checked"]["blocked_before_model_load"] is True
    assert artifact["chronological_event_stream_ready_score"] == 0.0
    assert artifact["atomic_semantic_live_ready_score"] == 0.0
    assert (tmp_path / exp5936.EVENT_STREAM_RELATIVE_PATH).read_text(encoding="utf-8") == ""
    exp5936.validate_artifact(artifact)


def test_preregistered_panel_has_36_cases_and_matched_union_budgets() -> None:
    # REQ-VERIFY-5936, SCENARIO-VERIFY-5936-PREREG
    panel = exp5936.freeze_held_cases()
    prereg = exp5936.build_preregistration(exp5936.ExperimentConfig(started_at=0.0), panel)
    atomic_prompt = exp5936.build_prompt(
        panel[0], "single_view_atomic_support", "original"
    )

    assert len(panel) >= 36
    assert len(atomic_prompt) < 12_000
    assert "sha256:" not in atomic_prompt
    assert all(case["split"] == "heldout" for case in panel)
    assert prereg["case_count"] == len(panel)
    assert prereg["model_case_rows_per_model"] == len(panel)
    assert prereg["arms"]["repeated_original_atomic_union"]["calls"] == 3
    assert prereg["arms"]["transformed_view_atomic_union"]["calls"] == 3
    assert (
        prereg["token_budgets"]["repeated_original_atomic_union"]
        == prereg["token_budgets"]["transformed_view_atomic_union"]
    )
    assert prereg["decoding_parameters_by_arm"]["repeated_original_atomic_union"] == (
        prereg["decoding_parameters_by_arm"]["transformed_view_atomic_union"]
    )
    assert prereg["primary_comparison"]["treatment_arm"] == "transformed_view_atomic_union"
    assert prereg["primary_comparison"]["matched_control_arm"] == "repeated_original_atomic_union"
    assert {"original", "paraphrase", "entity_permutation"}.issubset(
        set(prereg["semantic_transforms"]["view_ids"])
    )
    assert prereg["label_opening_rule"] == "after_arm_union_seal"
    assert prereg["stopping_rules"]["stop_on_gpu_fallback"] is True


def test_transformed_union_primary_gain_sets_live_ready_and_replay_is_tamper_safe(
    tmp_path: Path,
) -> None:
    # REQ-VERIFY-5936, SCENARIO-VERIFY-5936-UNION,
    # SCENARIO-VERIFY-5936-PRIMARY, SCENARIO-VERIFY-5936-EVENTS
    paths = _fake_model_files(tmp_path)
    artifact = exp5936.run_experiment(
        exp5936.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 2.0),
        model_resolver=_model_resolver(paths),
        environment_probe=lambda root: _passing_environment(),
        tokenizer_loader=_tokenizer_loader,
        gate_replay_provider=_ready_gate,
        collect_model_outputs_fn=_collector_with_transformed_gain,
    )

    event_path = tmp_path / exp5936.EVENT_STREAM_RELATIVE_PATH
    replay = exp5936.replay_event_stream(event_path)
    primary = artifact[
        "transformed_vs_repeated_original_primary_comparison_and_interval"
    ]

    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["chronological_event_stream_ready_score"] == 1.0
    assert artifact["atomic_semantic_live_ready_score"] == 1.0
    assert primary["delta_exact_success_rate_d_minus_c"] > 0.0
    assert primary["paired_interval"]["ci95"][0] > 0.0
    assert primary["matched_call_token_temperature_budget"] is True
    assert artifact[
        "no_label_feedback_no_hard_pruning_no_schema_reprompt_and_no_answer_enumeration_receipt"
    ]["ok"] is True
    assert artifact[
        "exact_semantic_missing_spurious_contradiction_and_unsafe_receipts"
    ]["unsafe_accepts_total"] == 0
    assert replay["ok"] is True
    assert replay["row_count"] == artifact[
        "chronological_event_stream_path_hash_rows_and_prefix_chain"
    ]["rows"]
    assert replay["row_count"] == (
        len(exp5936.MANDATED_MODEL_IDS)
        * len(exp5936.freeze_held_cases())
        * exp5936.CALLS_PER_MODEL_CASE
    )
    assert all(not row["contains_hidden_reference_answer"] for row in replay["rows"])
    exp5936.validate_artifact(artifact)

    tampered_lines = event_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(tampered_lines[0])
    tampered["visible_proposal"]["raw_text"] += " "
    tampered_lines[0] = json.dumps(tampered, sort_keys=True)
    tampered_path = tmp_path / "tampered.events.jsonl"
    tampered_path.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")
    assert exp5936.replay_event_stream(tampered_path)["ok"] is False


def test_retirement_triggers_when_transformed_union_is_zero_for_all_models(
    tmp_path: Path,
) -> None:
    # REQ-VERIFY-5936, SCENARIO-VERIFY-5936-PRIMARY
    paths = _fake_model_files(tmp_path)
    artifact = exp5936.run_experiment(
        exp5936.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 2.0),
        model_resolver=_model_resolver(paths),
        environment_probe=lambda root: _passing_environment(),
        tokenizer_loader=_tokenizer_loader,
        gate_replay_provider=_ready_gate,
        collect_model_outputs_fn=_collector_all_null,
    )

    assert artifact["status"] == "retired"
    assert artifact["honest_verdict"].startswith("retired:")
    assert artifact["retirement_decision"]["retire"] is True
    assert "all_three_models_zero_exact_success_in_transformed_view_union" in (
        artifact["retirement_decision"]["reasons"]
    )
    assert artifact["atomic_semantic_live_ready_score"] == 0.0
    exp5936.validate_artifact(artifact)


def test_checkpoint_refresh_and_validation_guards(tmp_path: Path) -> None:
    # REQ-VERIFY-5936, SCENARIO-VERIFY-5936-EVENTS
    checkpoint = tmp_path / "checkpoint.jsonl"
    first = {"stream_sequence_index": 0, "row_hash": "sha256:first", "payload": "a"}
    duplicate = {"stream_sequence_index": 0, "row_hash": "sha256:first", "payload": "a"}
    second = {"stream_sequence_index": 1, "row_hash": "sha256:second", "payload": "b"}

    exp5936.save_checkpoint(checkpoint, [first])
    loaded = exp5936.load_checkpoint(checkpoint)
    merged = exp5936.merge_resume_rows(loaded, [duplicate, second])

    assert loaded == [first]
    assert merged == [first, second]
    assert exp5936.replay_event_stream(tmp_path / "missing.events.jsonl")["reason"] == (
        "missing_event_stream"
    )
    assert exp5936._paired_interval([], "transformed_view_atomic_union", "repeated_original_atomic_union") == {
        "ci95": [0.0, 0.0],
        "mean": 0.0,
        "n_pairs": 0,
        "method": "deterministic_paired_bootstrap_ci95",
    }
    assert exp5936._contains_hidden_reference_answer({"x": [{"target_constraint_ir": {}}]}) is True
    assert exp5936._contains_hidden_reference_answer({"x": [{"safe": "value"}]}) is False
    assert exp5936._forbidden_keys({"target_constraint_ir": {"x": 1}}) == [
        "target_constraint_ir"
    ]
    assert exp5936._status_and_verdict(None, 0.0, {"retire": False}, False)[0] == "blocked"
    assert exp5936._status_and_verdict(None, 0.0, {"retire": False}, True)[0] == (
        "complete_null"
    )
    with pytest.raises(ValueError, match="unknown arm_id"):
        exp5936.view_ids_for_arm("unknown")
    with pytest.raises(ValueError, match="not part of arm"):
        exp5936.build_prompt(exp5936.freeze_held_cases()[0], "single_view_atomic_support", "bad")
    with pytest.raises(ValueError, match="held case expansion"):
        exp5936.freeze_held_cases(min_cases=37)
    assert exp5936._proposal_atom_ids({"raw_output_text": "{\"atom_ids\":[1,2]}"}) == [
        "1",
        "2",
    ]
    assert exp5936._proposal_atom_ids({"raw_output_text": "{"}) == []
    assert exp5936._proposal_atom_ids({"raw_output_text": "[]"}) == []

    case = exp5936.freeze_held_cases()[0]
    surface = exp5936.derive_surface_for_case(case, exp5935.versioned_atom_schema())
    aliases = exp5936._visible_atom_aliases(surface)
    assert aliases[0]["atom_alias"] == "a000"
    alias_entry = exp5936._proposal_entries_from_raw(
        surface,
        [{"view_id": "original", "raw_output_text": json.dumps({"atom_ids": ["a000"]})}],
    )
    assert alias_entry[0]["atom"]["atom_id"] == aliases[0]["atom_id"]

    tokenizer_edges = exp5936._tokenizer_receipts(
        [
            {"hf_id": "missing/path", "model_path": None},
            {"hf_id": "raises", "model_path": str(tmp_path / "fake.gguf")},
        ],
        lambda spec: (_ for _ in ()).throw(RuntimeError("tokenizer boom")),
    )
    assert "missing/path" not in tokenizer_edges
    assert tokenizer_edges["raises"]["ok"] is False

    assert exp5936._gate_replay_receipt(tmp_path)["block_reason"] == "exp5935_artifact_missing"
    gate_copy = tmp_path / exp5935.RESULT_RELATIVE_PATH
    gate_copy.parent.mkdir(parents=True)
    gate_copy.write_text(
        (exp5936.REPO_ROOT / exp5935.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    copied_gate = exp5936._gate_replay_receipt(tmp_path)
    assert copied_gate["atom_support_fixture_ready_score"] == 1.0
    assert copied_gate["ok"] is False
    assert exp5936._gate_replay_receipt(exp5936.REPO_ROOT)["ok"] is True
    invalid_gate = tmp_path / exp5935.RESULT_RELATIVE_PATH
    invalid_gate.write_text("{}", encoding="utf-8")
    assert exp5936._gate_replay_receipt(tmp_path)["block_reason"].startswith(
        "exp5935_replay_invalid"
    )

    assert exp5936._disk_probe(tmp_path, 1)["ok"] is True
    assert "available_mb" in exp5936._memory_probe(1)
    assert exp5936._atomic_output_probe(tmp_path / "out.json")["ok"] is True
    assert exp5936._atomic_checkpoint_resume_probe(tmp_path / "checkpoint.jsonl")["ok"] is True

    paths = _fake_model_files(tmp_path)
    artifact = exp5936.run_experiment(
        exp5936.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 2.0),
        model_resolver=_model_resolver(paths),
        environment_probe=lambda root: _passing_environment(),
        tokenizer_loader=_tokenizer_loader,
        gate_replay_provider=_ready_gate,
        collect_model_outputs_fn=_collector_with_transformed_gain,
    )
    refreshed = exp5936.refresh_artifact_test_exit_codes(
        root=tmp_path,
        test_exit_codes={"focused": 0, "coverage": 0},
    )
    assert refreshed["test_exit_codes"] == {"focused": 0, "coverage": 0}

    for key, value, message in [
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("atomic_semantic_live_ready_score", 0.5, "atomic_semantic_live_ready_score"),
        (
            "chronological_event_stream_ready_score",
            0.5,
            "chronological_event_stream_ready_score",
        ),
    ]:
        broken = json.loads(json.dumps(artifact))
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5936.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken[
        "no_label_feedback_no_hard_pruning_no_schema_reprompt_and_no_answer_enumeration_receipt"
    ]["label_feedback_used"] = True
    with pytest.raises(ValueError, match="label feedback"):
        exp5936.validate_artifact(broken)

    for key, message in [
        ("model_side_hard_pruning_used", "hard pruning"),
        ("schema_reprompt_used", "schema reprompt"),
        ("complete_answer_enumeration_used", "complete answer enumeration"),
    ]:
        broken = json.loads(json.dumps(artifact))
        broken[
            "no_label_feedback_no_hard_pruning_no_schema_reprompt_and_no_answer_enumeration_receipt"
        ][key] = True
        with pytest.raises(ValueError, match=message):
            exp5936.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken[
        "exact_semantic_missing_spurious_contradiction_and_unsafe_receipts"
    ]["unsafe_accepts_total"] = 1
    with pytest.raises(ValueError, match="zero unsafe"):
        exp5936.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["honest_verdict"] = "complete_null: wrong positive prefix"
    with pytest.raises(ValueError, match="complete_positive"):
        exp5936.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken[
        "transformed_vs_repeated_original_primary_comparison_and_interval"
    ]["paired_interval"]["ci95"][0] = 0.0
    with pytest.raises(ValueError, match="positive lower"):
        exp5936.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["honest_verdict"] = "bad"
    broken["atomic_semantic_live_ready_score"] = 0.0
    with pytest.raises(ValueError, match="honest_verdict"):
        exp5936.validate_artifact(broken)

    missing = dict(artifact)
    del missing["model_specs"]
    with pytest.raises(ValueError, match="missing required fields"):
        exp5936.validate_artifact(missing)
