"""Tests for REQ-ARC-7127 and its paired ARC cell scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from carnot import experiment_7127_v626_adapter_withheld_arc_loo as exp


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "openspec/capabilities/arc-agi/spec.md"


def _hash(tag: str) -> str:
    return "sha256:" + (tag.encode().hex() + "1" * 64)[:64]


def _phase_rows() -> list[dict[str, object]]:
    rows = []
    for index, phase in enumerate(exp.PHASES):
        cap = exp.cap_for_phase(phase)
        rows.append(
            {
                "phase": phase,
                "process_pid": 7000 + min(index, 2),
                "process_start_ticks": 100 + min(index, 2),
                "started_monotonic_ns": index * 2_000_000_000,
                "ended_monotonic_ns": index * 2_000_000_000 + 1_000_000_000,
                "duration_s": 1.0,
                "cap_s": cap,
                "exit_state": "completed",
                "timeout_state": "not_timed_out",
                "stop_reason": "completed",
                "evidence_hash": _hash(f"phase-{index}"),
            }
        )
    return rows


def _evidence(tmp_path: Path) -> dict[str, object]:
    adapters = ["a1", "r11l", "z9"]
    process_rows = [
        {
            "role": "setup",
            "pid": 7000,
            "start_ticks": 100,
            "fresh_process": True,
            "exit_code": 0,
            "timed_out": False,
            "stop_reason": "completed",
        },
        {
            "role": "adapter_withheld",
            "pid": 7001,
            "start_ticks": 101,
            "fresh_process": True,
            "exit_code": 0,
            "timed_out": False,
            "stop_reason": "completed",
            "entrypoint": "make_carnot_agent:E3AgentPolicy",
            "adapter_keys_before": adapters,
            "adapter_keys_after": ["a1", "z9"],
            "removed_adapters": ["r11l"],
        },
        {
            "role": "adapter_visible_control",
            "pid": 7002,
            "start_ticks": 102,
            "fresh_process": True,
            "exit_code": 0,
            "timed_out": False,
            "stop_reason": "completed",
            "entrypoint": "make_carnot_agent:E3AgentPolicy",
            "adapter_keys_before": adapters,
            "adapter_keys_after": adapters,
            "removed_adapters": [],
        },
    ]
    request_rows = []
    token_rows = []
    proposal_rows = []
    verifier_rows = []
    action_rows = []
    transition_rows = []
    raw_manifest = []
    for index, arm in enumerate(exp.ARM_NAMES):
        request_id = f"request-{index}"
        action_id = f"action-{index}"
        raw = tmp_path / f"{arm}.jsonl"
        raw.write_text(json.dumps({"arm": arm, "request_id": request_id}) + "\n")
        raw_manifest.append(exp.raw_trace_receipt(raw, arm=arm, event_count=1))
        request_rows.append(
            {
                "arm": arm,
                "request_id": request_id,
                "model_repository": exp.MODEL_REPOSITORY,
                "prompt_hash": _hash("common-prompt"),
                "budget": deepcopy(exp.COMMON_BUDGET),
                "seed": exp.RANDOM_SEED,
                "generation_invoked": True,
                "response_hash": _hash(f"response-{index}"),
            }
        )
        token_rows.append(
            {
                "arm": arm,
                "request_id": request_id,
                "prompt_tokens": 100,
                "generated_tokens": 20,
            }
        )
        proposal_rows.append(
            {
                "arm": arm,
                "request_id": request_id,
                "proposal_id": f"proposal-{index}",
                "action_id": action_id,
                "action": 6,
                "data": {"x": 10, "y": 10},
                "source": "E3AgentPolicy.next_move",
            }
        )
        verifier_rows.append(
            {
                "arm": arm,
                "proposal_id": f"proposal-{index}",
                "verifier": "E3 policy-visible action schema",
                "accepted": True,
                "oracle": False,
            }
        )
        action_rows.append(
            {
                "arm": arm,
                "action_id": action_id,
                "proposal_id": f"proposal-{index}",
                "executed": True,
                "action": 6,
                "data": {"x": 10, "y": 10},
            }
        )
        transition_rows.append(
            {
                "arm": arm,
                "action_id": action_id,
                "executed": True,
                "before_hash": _hash(f"before-{index}"),
                "after_hash": _hash(f"after-{index}"),
                "level_before": 0,
                "level_after": 0,
                "reward": 0.0,
            }
        )
    return {
        "preconditions_checked": [
            exp.gate_row("artifact_initialized_before_setup", True, True),
            exp.gate_row("all_runtime_preconditions", True, True),
        ],
        "source_artifact_hashes": {
            "ops/arc_solve_registry.yaml": _hash("registry"),
            "environment_files/r11l/495a7899/r11l.py": _hash("fixture"),
        },
        "model": {
            "model_repository": exp.MODEL_REPOSITORY,
            "model_path": "/cache/Qwen3.6-35B-A3B-Q4_K_M.gguf",
            "model_hash": _hash("model"),
            "model_quantization": "Q4_K_M",
        },
        "selected_game": {
            "game": "r11l",
            "target_level": 1,
            "registry_levels_reproduced": 6,
            "fixture_path": "environment_files/r11l/495a7899/r11l.py",
            "fixture_hash": _hash("fixture"),
        },
        "registry_rank_before_outcomes": 1,
        "registry_hash_after": _hash("registry"),
        "phase_receipt_rows": _phase_rows(),
        "process_rows": process_rows,
        "request_rows": request_rows,
        "token_rows": token_rows,
        "proposal_rows": proposal_rows,
        "verifier_rows": verifier_rows,
        "action_rows": action_rows,
        "transition_rows": transition_rows,
        "raw_trace_manifest": raw_manifest,
        "gpu_telemetry_rows": [
            {"gpu": 0, "model": "NVIDIA GeForce RTX 3090", "sample_ok": True},
            {"gpu": 1, "model": "NVIDIA GeForce RTX 3090", "sample_ok": True},
        ],
        "duration_s": 12.0,
    }


def _build(tmp_path: Path, **changes: object) -> dict[str, object]:
    evidence = _evidence(tmp_path)
    evidence.update(changes)
    return exp.build_artifact(run_date="20260907", **evidence)


def test_req_arc_7127_spec_precedes_implementation() -> None:
    """REQ-ARC-7127 owns the artifact fields and named failure scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-7127") :]
    for scenario in ("PREFLIGHT", "PAIR", "RUNTIME", "NULL", "ADVERSARIAL", "NONCLAIM"):
        assert f"SCENARIO-ARC-7127-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_7127_preflight_writes_complete_block_before_setup(tmp_path: Path) -> None:
    """SCENARIO-ARC-7127-PREFLIGHT publishes a terminal block first."""

    output = tmp_path / "artifact.json"
    artifact = exp.initialize_terminal_artifact(output, run_date="20260907")
    assert output.is_file()
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert exp.validate_artifact(artifact) == []


def test_model_resolution_uses_cached_pair_and_qwen_chat_template(tmp_path: Path) -> None:
    """REQ-ARC-7127 resolves only cached Qwen Q4_K_M bytes."""

    model = tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model.write_bytes(b"GGUFmodel")
    calls = []

    def provider(**kwargs: object) -> list[dict[str, object]]:
        calls.append(kwargs)
        return [
            {"hf_id": exp.MODEL_REPOSITORY, "model_path": str(model), "gpu": 0},
            {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "model_path": str(model) + ".2", "gpu": 1},
        ]

    receipt = exp.resolve_headline_model(provider)
    assert calls == [{"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 1)}]
    assert receipt["model_repository"] == exp.MODEL_REPOSITORY
    assert receipt["model_path"] == str(model.resolve())
    assert receipt["model_quantization"] == "Q4_K_M"
    assert receipt["llama_cpp_chat_template"] == "gguf_metadata_qwen3"
    assert receipt["download_attempted"] is False


def test_registry_rank_one_freezes_existing_public_fixture() -> None:
    """REQ-ARC-7127 freezes eligibility rank one before outcome evidence."""

    rows = [
        {"game": "skip", "reproducibility": "provisional", "levels_reproduced": 1},
        {"game": "r11l", "reproducibility": "reproduced", "levels_reproduced": 6, "full_game_clear": True},
        {"game": "next", "reproducibility": "reproduced", "levels_reproduced": 2, "full_game_clear": True},
    ]
    selected = exp.freeze_registry_rank_one(rows)
    assert selected == {
        "game": "r11l",
        "target_level": 1,
        "registry_levels_reproduced": 6,
        "registry_rank": 1,
    }


def test_scenario_arc_7127_complete_zero_pair_is_terminal_null(tmp_path: Path) -> None:
    """SCENARIO-ARC-7127-NULL accepts executed zero-level evidence."""

    artifact = _build(tmp_path)
    assert artifact["arc_loo_cell_complete_score"] == 1
    assert artifact["withheld_levels"] == 0
    assert artifact["control_levels"] == 0
    assert artifact["level_delta"] == 0
    assert artifact["verdict_class"] == "null"
    assert str(artifact["honest_verdict"]).startswith("complete_null_")
    assert artifact["solve_claim_made"] is False
    assert artifact["offline_reproduced"] is False
    assert all(row["solve_provenance"] == "development_proxy" for row in artifact["rows"])
    assert exp.validate_artifact(artifact, verify_raw_traces=True) == []


def test_adapter_leakage_and_wrong_removal_are_disqualified(tmp_path: Path) -> None:
    """SCENARIO-ARC-7127-ADVERSARIAL rejects target adapter leakage."""

    evidence = _evidence(tmp_path)
    withheld = evidence["process_rows"][1]
    withheld["adapter_keys_after"] = ["a1", "r11l", "z9"]
    withheld["removed_adapters"] = []
    artifact = exp.build_artifact(run_date="20260907", **evidence)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["adapter_withheld_exactly"] is False
    assert artifact["gate_check_summary"]["failed_check"] == "adapter_withheld_exactly"
    assert exp.validate_artifact(artifact) == []


def test_noop_arm_and_zero_attempt_cannot_be_positive(tmp_path: Path) -> None:
    """SCENARIO-ARC-7127-RUNTIME blocks fake and zero-attempt arms."""

    evidence = _evidence(tmp_path)
    evidence["transition_rows"][0]["executed"] = False
    evidence["action_rows"][0]["executed"] = False
    blocked = exp.build_artifact(run_date="20260907", **evidence)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["arc_loo_cell_complete_score"] == 0
    assert blocked["gate_check_summary"]["failed_check"] == "executed_transition_per_arm"

    empty = _evidence(tmp_path)
    for field in ("request_rows", "token_rows", "proposal_rows", "verifier_rows", "action_rows", "transition_rows"):
        empty[field] = []
    blocked = exp.build_artifact(run_date="20260907", **empty)
    assert blocked["verdict_class"] == "blocked"
    assert "positive" not in str(blocked["honest_verdict"])
    assert exp.validate_artifact(blocked) == []


def test_stale_process_reuse_and_fake_entrypoint_are_disqualified(tmp_path: Path) -> None:
    """SCENARIO-ARC-7127-ADVERSARIAL rejects reused workers and fake routes."""

    reused = _evidence(tmp_path)
    reused["process_rows"][2]["pid"] = reused["process_rows"][1]["pid"]
    reused["process_rows"][2]["start_ticks"] = reused["process_rows"][1]["start_ticks"]
    artifact = exp.build_artifact(run_date="20260907", **reused)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["fresh_process_per_arm"] is False

    fake = _evidence(tmp_path)
    fake["process_rows"][1]["entrypoint"] = "fake_policy"
    artifact = exp.build_artifact(run_date="20260907", **fake)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["real_e3_entrypoint_used"] is False


def test_missing_phase_cap_overrun_and_registry_mutation_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-7127-ADVERSARIAL rejects three protocol mutations."""

    missing = _evidence(tmp_path)
    missing["phase_receipt_rows"] = missing["phase_receipt_rows"][:-1]
    artifact = exp.build_artifact(run_date="20260907", **missing)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "complete_phase_receipts"

    overrun = _evidence(tmp_path)
    overrun["phase_receipt_rows"][2]["duration_s"] = 301.0
    artifact = exp.build_artifact(run_date="20260907", **overrun)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["gate_check_summary"]["failed_check"] == "phase_caps_respected"

    mutated = _evidence(tmp_path)
    mutated["registry_hash_after"] = _hash("changed")
    artifact = exp.build_artifact(run_date="20260907", **mutated)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["registry_mutated"] is True
    assert exp.validate_artifact(artifact) == []


def test_raw_trace_mutation_and_projection_forgery_are_detected(tmp_path: Path) -> None:
    """REQ-ARC-7127 binds raw bytes and recomputed aggregate fields."""

    artifact = _build(tmp_path)
    Path(artifact["raw_trace_manifest"][0]["path"]).write_text("changed\n")
    assert "raw_trace_hash_mismatch" in exp.validate_artifact(artifact, verify_raw_traces=True)

    forged = _build(tmp_path)
    forged["arc_loo_cell_complete_score"] = 0
    forged["verdict_class"] = "positive"
    forged["honest_verdict"] = "complete_positive_forged"
    forged["reproducibility_checksum"] = exp.artifact_checksum(forged)
    errors = exp.validate_artifact(forged)
    assert "artifact_projection_mismatch" in errors

