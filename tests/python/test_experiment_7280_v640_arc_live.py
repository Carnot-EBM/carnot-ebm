"""Tests for the post-identity-repair live ARC policy-use pilot.

Spec: REQ-ARC-WMTE-7280 and SCENARIO-ARC-WMTE-7280-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import carnot.experiment_7280_v640_arc_live as live


def _identity_rows(episode_id: str) -> list[dict[str, object]]:
    return [
        {
            "episode_id": episode_id,
            "obligation": name,
            "status": "supported",
            "authenticated_before_first_generation": True,
            "authenticated_at_utc": "2026-09-13T15:00:00+00:00",
            "evidence_source": "live_server_props",
            "observed": {"unit": episode_id},
        }
        for name in live.IDENTITY_OBLIGATIONS
    ]


def _episode(
    game: str,
    arm: str,
    *,
    heldout: float,
    identity: float,
    levels: int,
    consumed: bool = False,
) -> dict[str, object]:
    episode_id = f"{game}:{arm}"
    return {
        "episode_id": episode_id,
        "game": game,
        "arm": arm,
        "seed": live.RANDOM_SEED,
        "execution_order": 0,
        "disposition": "complete",
        "censored": False,
        "abstention": False,
        "model_invoked": True,
        "model_loaded": True,
        "generation_calls_attempted": 2,
        "generation_calls_completed": 2,
        "usable_answers": 1,
        "generated_tokens": 3000,
        "action_count": 192,
        "action_limit": 192,
        "elapsed_cap_s": live.SESSION_LIMIT_S,
        "identity_baseline_accuracy": identity,
        "heldout_accuracy": heldout,
        "non_identity_predictions": 1 if arm == "typed_witness_feedback" else 0,
        "candidate_valid": arm == "typed_witness_feedback",
        "trust_accepted": arm == "typed_witness_feedback",
        "installed_plans": 1 if arm == "typed_witness_feedback" else 0,
        "model_planned_actions": 1 if consumed else 0,
        "levels": levels,
        "action_cost": 192,
        "compute_cost": {"wall_s": 61.0, "prompt_tokens": 1000, "generated_tokens": 3000},
        "policy_consumption_rows": (
            [
                {
                    "episode_id": episode_id,
                    "engine_sha256": "sha256:engine",
                    "plan_sha256": "sha256:plan",
                    "action_index": 90,
                    "policy_action_executed": True,
                }
            ]
            if consumed
            else []
        ),
        "identity_obligation_rows": _identity_rows(episode_id),
        "identity_receipt_sha256": "sha256:identity",
        "adapter_disabled": True,
        "fresh_store": True,
        "raw_request_manifest": [],
        "error": None,
    }


def _positive_rows() -> list[dict[str, object]]:
    return [
        _episode("re86", "current_feedback", heldout=0.35, identity=0.25, levels=1),
        _episode(
            "re86",
            "typed_witness_feedback",
            heldout=0.70,
            identity=0.25,
            levels=1,
            consumed=True,
        ),
        _episode(
            "r11l",
            "typed_witness_feedback",
            heldout=0.65,
            identity=0.30,
            levels=2,
            consumed=True,
        ),
        _episode("r11l", "current_feedback", heldout=0.40, identity=0.30, levels=2),
    ]


def _complete_artifact(rows: list[dict[str, object]]) -> dict[str, object]:
    return live.build_terminal_artifact(
        started_at_utc="2026-09-13T15:00:00+00:00",
        ended_at_utc="2026-09-13T15:05:00+00:00",
        duration_s=300.0,
        preconditions=[live.gate_check("upstream", "exp7276", "ready", 1, 1)],
        source_hashes={"results/experiment_7276_v640_arc_identity.json": {"sha256": "x"}},
        selection_receipt={"games": ["re86", "r11l"]},
        episode_rows=rows,
        model_spec={
            "hf_id": live.MODEL_ID,
            "quantization": live.QUANTIZATION,
            "revision": "revision",
            "model_file_hash": "sha256:model",
            "model_path": "/cache/model.gguf",
        },
        runner_receipt={
            "runner": "LocalGGUFProposer_native_llama.cpp",
            "server_pid": 42,
            "server_pid_start_tick": 7,
            "measured_kv_headroom_mb": 1024,
            "dual_gpu_runner_used": False,
        },
        validation_receipts=[{"name": "focused", "passed": True, "exit_code": 0}],
        phase_spans=[{"phase": "generation", "duration_s": 250.0}],
    )


def test_identity_upstream_authentication_is_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7280-EXTERNAL-BLOCK: identity readiness is exact."""
    path = tmp_path / "identity.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "exp7276-arc-identity",
                "milestone": "2026.09.640",
                "status": "complete",
                "arc_identity_ready_score": 1,
                "verdict_class": "circular_positive",
            }
        )
        + "\n"
    )
    checks, payload = live.authenticate_identity_upstream(path, quarantined=False, retired=False)
    assert payload["arc_identity_ready_score"] == 1
    assert all(row["passed"] for row in checks)

    path.write_bytes(path.read_bytes() + b"\x00")
    checks, _ = live.authenticate_identity_upstream(path, quarantined=True, retired=True)
    assert {row["field"] for row in checks if not row["passed"]} == {
        "quarantine_state",
        "retirement_state",
        "clean_terminal_bytes",
    }
    missing, payload = live.authenticate_identity_upstream(
        tmp_path / "missing.json", quarantined=False, retired=False
    )
    assert payload == {}
    assert next(row for row in missing if row["field"] == "exists")["passed"] is False

    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff\n")
    _checks, payload = live.authenticate_identity_upstream(
        invalid, quarantined=False, retired=False
    )
    assert payload == {}


def test_registry_precheck_freezes_re86_r11l_and_reverses_second_game() -> None:
    """SCENARIO-ARC-WMTE-7280-COUNTERBALANCED-POLICY-USE: roster is fixed before play."""
    registry = {
        "games": [
            {"game": "r11l", "levels_reproduced": 6, "reproducibility": "reproduced"},
            {"game": "aa11", "levels_reproduced": 0},
            {"game": "re86", "levels_reproduced": 8, "reproducibility": "reproduced"},
        ]
    }
    receipt = live.freeze_public_roster(registry, adaptered_games={"re86", "r11l"})
    assert receipt["selection_steps"] == ["registry_precheck", "roster_freeze"]
    assert receipt["games"] == ["re86", "r11l"]
    assert receipt["re_solve_campaign"] is False
    assert all(row["adapter_available_but_withheld"] for row in receipt["game_rows"])
    assert all(row["registered_public_development_target"] for row in receipt["game_rows"])

    schedule = live.counterbalanced_schedule(receipt["games"], live.RANDOM_SEED)
    assert [(row["game"], row["arm"]) for row in schedule] == [
        ("re86", "current_feedback"),
        ("re86", "typed_witness_feedback"),
        ("r11l", "typed_witness_feedback"),
        ("r11l", "current_feedback"),
    ]
    assert len({row["episode_id"] for row in schedule}) == 4

    incomplete = live.freeze_public_roster(
        {"games": [{"game": "re86", "levels_reproduced": 8}]}, adaptered_games=set()
    )
    assert incomplete["games"] == ["re86"]
    assert incomplete["roster_complete"] is False


def test_cached_model_resolution_uses_string_snapshot_path(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7280: cache selection pins the actual Qwen snapshot revision."""
    model = (
        tmp_path
        / "models--unsloth--Qwen3.8-27B-GGUF"
        / "snapshots"
        / "revision-one"
        / "Qwen3.8-27B-Q4_K_M.gguf"
    )
    model.parent.mkdir(parents=True)
    model.write_bytes(b"gguf fixture")
    selected, path, valid = live.resolve_cached_model(
        [{"hf_id": live.MODEL_ID, "model_path": str(model)}]
    )
    assert valid is True
    assert path == model
    assert selected["revision"] == "revision-one"
    assert selected["quantization"] == live.QUANTIZATION

    selected, path, valid = live.resolve_cached_model([])
    assert valid is False
    assert path is None
    assert selected["revision"] is None


def test_episode_environment_keeps_equal_caps_and_fresh_stores(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7280: only the typed-witness switch differs between paired arms."""
    inherited = {"CARNOT_ARC_TRANSITION_WITNESS": "stale", "KEEP": "yes"}
    current = live.episode_environment(
        inherited,
        arm="current_feedback",
        episode_dir=tmp_path / "current",
        gpu_index=0,
        port=8123,
    )
    treatment = live.episode_environment(
        inherited,
        arm="typed_witness_feedback",
        episode_dir=tmp_path / "treatment",
        gpu_index=0,
        port=8123,
    )
    assert "CARNOT_ARC_TRANSITION_WITNESS" not in current
    assert treatment["CARNOT_ARC_TRANSITION_WITNESS"] == "1"
    for environment in (current, treatment):
        assert environment["CARNOT_FORCE_LIVE"] == "1"
        assert environment["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
        assert environment["CARNOT_ARC_MAX_REFINEMENT_ROUNDS"] == "2"
        assert environment["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "2048"
        assert environment["CARNOT_ARC_RANDOM_SEED"] == str(live.RANDOM_SEED)
        assert environment["CARNOT_ARC_E3_DIR"].startswith(str(tmp_path))
        assert environment["KEEP"] == "yes"


def test_identity_obligations_are_bound_to_each_episode_before_generation() -> None:
    """SCENARIO-ARC-WMTE-7280-IDENTITY-BEFORE-GENERATION: all obligations are retained."""
    receipt = {
        "identity_obligation_rows": [
            {
                "obligation": name,
                "status": "supported",
                "evidence_source": "live_server_props",
                "observed_value": {"name": name},
            }
            for name in live.IDENTITY_OBLIGATIONS
        ]
    }
    rows = live.bind_identity_obligations(
        "re86:current_feedback",
        receipt,
        authenticated_at_utc="2026-09-13T15:00:00+00:00",
    )
    assert [row["obligation"] for row in rows] == list(live.IDENTITY_OBLIGATIONS)
    assert all(row["episode_id"] == "re86:current_feedback" for row in rows)
    assert all(row["authenticated_before_first_generation"] for row in rows)

    broken = deepcopy(receipt)
    broken["identity_obligation_rows"][0]["status"] = "unsupported"
    assert (
        live.bind_identity_obligations("episode", broken, authenticated_at_utc="t")[0][
            "authenticated_before_first_generation"
        ]
        is False
    )


def test_reducer_requires_useful_consumed_plan_in_each_treatment_game() -> None:
    """SCENARIO-ARC-WMTE-7280-PER-GAME-VALUE: synthesis and policy use are separate gates."""
    reduced = live.reduce_episode_rows(_positive_rows())
    assert reduced["arc_capture_complete_score"] == 1
    assert reduced["arc_method_value_score"] == 1
    assert reduced["treatment_games_with_useful_consumed_plan"] == ["r11l", "re86"]
    assert reduced["treatment_accuracy_above_identity_by_game"] == {
        "r11l": True,
        "re86": True,
    }
    assert reduced["matched_game_level_regression"] is False
    assert reduced["population_confidence_interval"] is None
    assert reduced["invocation_counts"] == {
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "generation_calls_attempted": 8,
        "generation_calls_completed": 8,
        "usable_answers": 4,
    }

    one_unused = deepcopy(_positive_rows())
    one_unused[2]["policy_consumption_rows"] = []
    assert live.reduce_episode_rows(one_unused)["arc_method_value_score"] == 0

    identity_tie = deepcopy(_positive_rows())
    identity_tie[1]["heldout_accuracy"] = identity_tie[1]["identity_baseline_accuracy"]
    assert live.reduce_episode_rows(identity_tie)["arc_method_value_score"] == 0

    invalid = deepcopy(_positive_rows())
    invalid[1]["candidate_valid"] = False
    assert live.reduce_episode_rows(invalid)["arc_method_value_score"] == 0

    regression = deepcopy(_positive_rows())
    regression[2]["levels"] = 1
    assert live.reduce_episode_rows(regression)["arc_method_value_score"] == 0


def test_episode_validation_rejects_budget_identity_and_isolation_drift() -> None:
    """REQ-ARC-WMTE-7280: action, call, token, identity, and store limits fail closed."""
    row = _positive_rows()[0]
    assert live.validate_episode_row(row) == []
    for field, value, expected in (
        ("action_count", 193, "action_limit_exceeded"),
        ("generation_calls_attempted", 3, "generation_call_limit_exceeded"),
        ("generation_calls_completed", 3, "generation_completion_count_invalid"),
        ("generated_tokens", 4097, "generated_token_limit_exceeded"),
        ("adapter_disabled", False, "adapter_not_disabled"),
        ("fresh_store", False, "store_not_fresh"),
        ("arm", "bad", "arm_invalid"),
        ("disposition", "running", "disposition_not_terminal"),
    ):
        broken = deepcopy(row)
        broken[field] = value
        assert expected in live.validate_episode_row(broken)
    broken_identity = deepcopy(row)
    broken_identity["identity_obligation_rows"][0]["status"] = "unsupported"
    assert "identity_obligations_not_supported" in live.validate_episode_row(broken_identity)


def test_terminal_artifact_exposes_all_required_fields_and_unset_official_score() -> None:
    """SCENARIO-ARC-WMTE-7280-TERMINAL-PUBLICATION: terminal evidence is ordinary and sealed."""
    artifact = _complete_artifact(_positive_rows())
    assert live.validate_artifact(artifact) == []
    assert artifact["schema"] == "carnot.experiment_7280.v640.arc_live.v1"
    assert artifact["experiment_id"] == "exp7280-arc-live"
    assert artifact["milestone"] == "2026.09.640"
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["arc_capture_complete_score"] == 1
    assert artifact["arc_method_value_score"] == 1
    assert artifact["official_score"] is None
    assert artifact["verifier_is_oracle"] is True
    assert artifact["MODEL_SPECS"][0]["hf_id"] == live.MODEL_ID
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["inference_substrate_class"] == "model_full_generation"
    assert artifact["inference_mode"] == "live_gpu"
    assert artifact["execution_venue"] == "host"
    assert artifact["runner_receipt"]["dual_gpu_runner_used"] is False
    assert len(artifact["identity_obligation_rows"]) == 4 * len(live.IDENTITY_OBLIGATIONS)
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["population_confidence_interval"] is None

    unused = deepcopy(_positive_rows())
    unused[2]["policy_consumption_rows"] = []
    null = _complete_artifact(unused)
    assert null["arc_method_value_score"] == 0
    assert null["honest_verdict"] == "complete_null_typed_witness_policy_use_gates_not_met"
    assert null["verdict_class"] == "null"

    load_only_rows = deepcopy(_positive_rows())
    for row in load_only_rows:
        row.update(
            {
                "model_invoked": False,
                "generation_calls_attempted": 0,
                "generation_calls_completed": 0,
                "usable_answers": 0,
                "generated_tokens": 0,
            }
        )
    load_only = _complete_artifact(load_only_rows)
    assert load_only["inference_substrate"] == "model_load_no_generation"
    no_load_rows = deepcopy(load_only_rows)
    for row in no_load_rows:
        row["model_loaded"] = False
    no_load = _complete_artifact(no_load_rows)
    assert no_load["inference_substrate"] == "blocked_no_run"
    assert no_load["MODEL_SPECS"] == []

    wrong_spec = deepcopy(artifact)
    wrong_spec["MODEL_SPECS"] = [{"hf_id": "wrong", "quantization": "wrong"}]
    assert "model_specs_inconsistent" in live.validate_artifact(wrong_spec)
    forbidden_spec = deepcopy(no_load)
    forbidden_spec["MODEL_SPECS"] = [{"hf_id": live.MODEL_ID}]
    assert "model_specs_inconsistent" in live.validate_artifact(forbidden_spec)

    mutated = deepcopy(artifact)
    mutated["rows"][0]["heldout_accuracy"] = 0.0
    assert "reproducibility_checksum_mismatch" in live.validate_artifact(mutated)


def test_external_failure_is_terminal_blocked_without_success_shape() -> None:
    """SCENARIO-ARC-WMTE-7280-EXTERNAL-BLOCK: missing prerequisites are never partial."""
    failure = live.gate_check("upstream", "exp7276", "arc_identity_ready_score", 1, 0)
    artifact = live.build_blocked_artifact(
        started_at_utc="2026-09-13T15:00:00+00:00",
        ended_at_utc="2026-09-13T15:00:01+00:00",
        duration_s=1.0,
        preconditions=[failure],
        source_hashes={},
    )
    assert live.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"] == [failure]
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"]["generation_calls_attempted"] == 0
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert not artifact["rows"]
    assert "partial" not in artifact["honest_verdict"]


def test_independent_reducer_and_validation_plan_are_scoped(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7280-TERMINAL-PUBLICATION: validation is exact and reducible."""
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps({"rows": _positive_rows()}))
    assert live.independent_reduce(raw)["arc_method_value_score"] == 1

    commands = live.build_validation_commands(
        terminal_candidate=tmp_path / "terminal.json", raw_rows=raw
    )
    names = [row["name"] for row in commands]
    assert names == [
        "focused_exp7280",
        "affected_exp7276",
        "affected_arc_eval_provenance",
        "e2e_009_policy_memory",
        "e2e_009_offline_scored_smoke",
        "e2e_010_grammar_transport",
        "scoped_coverage_run",
        "scoped_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
        "independent_raw_row_reducer",
        "terminal_candidate_adversarial_verify",
        "terminal_candidate_row_consistency",
    ]
    joined = [" ".join(row["command"]) for row in commands]
    assert all("tests/python -q" not in command for command in joined)
    assert "--mechanism e3 --game r11l --max-actions 12" in joined[4]
    assert commands[4]["env"] == {"CARNOT_ARC_DISABLE_INDUCTION": "1"}
    assert all("-n 0" in joined[index] for index in (0, 1, 2, 3, 5))
    assert str(tmp_path / "terminal.json") in joined[-1]


def test_gate_checksum_atomic_write_and_validator_edges(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7280: terminal mutations and malformed shapes fail closed."""
    gate = live.gate_check("kind", "source", "field", True, True)
    assert gate["passed"] is True
    target = tmp_path / "nested" / "result.json"
    live.atomic_write(target, {"b": 2, "a": 1})
    assert target.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'

    valid = _complete_artifact(_positive_rows())
    path = tmp_path / "valid.json"
    path.write_text(json.dumps(valid))
    assert live.validate_artifact(path) == []
    assert live.validate_artifact(tmp_path / "absent.json")[0].startswith("artifact_unreadable")

    malformed = deepcopy(valid)
    malformed.update(
        {
            "schema": "wrong",
            "milestone": "wrong",
            "status": "running",
            "verdict_class": "wrong",
            "field_principles": {},
        }
    )
    assert {
        "schema_or_experiment_identity_mismatch",
        "milestone_or_run_date_mismatch",
        "status_not_terminal",
        "verdict_class_invalid",
        "field_principles_must_cover_every_top_level_field",
    } <= set(live.validate_artifact(malformed))

    oracle = deepcopy(valid)
    oracle["verdict_class"] = "positive"
    assert "oracle_forbids_positive" in live.validate_artifact(oracle)

    broken = deepcopy(valid)
    broken.update(
        {
            "honest_verdict": "wrong",
            "arc_capture_complete_score": 0,
            "arc_method_value_score": 0,
            "invocation_counts": {},
            "model_invoked": False,
            "inference_substrate": "wrong",
            "duration_s": 1.0,
            "population_confidence_interval": [0.0, 1.0],
            "official_score": 1.0,
        }
    )
    assert {
        "complete_verdict_prefix_invalid",
        "arc_capture_complete_score_inconsistent",
        "arc_method_value_score_inconsistent",
        "invocation_counts_inconsistent",
        "model_invoked_inconsistent",
        "live_substrate_inconsistent",
        "model_full_generation_duration_floor_failed",
        "population_confidence_interval_forbidden",
        "official_score_must_be_unset",
    } <= set(live.validate_artifact(broken))

    blocked = live.build_blocked_artifact(
        started_at_utc="s",
        ended_at_utc="e",
        duration_s=1.0,
        preconditions=[live.gate_check("x", "y", "z", True, False)],
        source_hashes={},
    )
    blocked.update(
        {
            "honest_verdict": "wrong",
            "rows": [{"bad": True}],
            "model_invoked": True,
            "inference_substrate": "live_llm_inference",
            "gate_check_summary": [],
        }
    )
    assert {
        "blocked_verdict_prefix_invalid",
        "blocked_artifact_contains_measurement",
        "blocked_substrate_invalid",
        "blocked_gate_summary_missing",
    } <= set(live.validate_artifact(blocked))
