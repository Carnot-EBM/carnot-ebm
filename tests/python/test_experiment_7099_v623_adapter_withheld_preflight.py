"""Tests for REQ-ARC-7099 adapter-withheld E3 action evidence."""

from __future__ import annotations

import builtins
from copy import deepcopy
import io
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_7099_v623_adapter_withheld_preflight as experiment


ROOT = Path(__file__).resolve().parents[2]


def _model_files(tmp_path: Path) -> list[dict[str, object]]:
    rows = []
    for index, template in enumerate(experiment.MODEL_SPECS):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(f"model-{index}".encode())
        rows.append({**template, "model_path": str(path), "gpu": index})
    return rows


def _registry_rows() -> list[dict[str, object]]:
    return [
        {"game": "aa01", "levels_reproduced": 1, "mechanic_class": "navigation"},
        {"game": "bb02", "levels_reproduced": 3, "mechanic_class": "click"},
        {"game": "cc03", "levels_reproduced": 6, "mechanic_class": "navigation"},
        {"game": "dd04", "levels_reproduced": 9, "mechanic_class": "program"},
        {"game": "ee05", "levels_reproduced": 4, "mechanic_class": "drag"},
    ]


def _action_row(
    model: dict[str, object],
    game_id: str,
    *,
    index: int,
    raw_path: Path,
) -> dict[str, object]:
    candidates = [
        {"action": 1, "data": None},
        {"action": 2, "data": None},
    ]
    selected = candidates[index % 2]
    forecast = {
        "selected_candidate_index": index % 2,
        "predicted_change": True,
        "rationale": "The action can change the visible state.",
    }
    invocation = {
        "invoked": True,
        "model_id": model["hf_id"],
        "prompt_hash": f"sha256:{index:064x}",
        "candidate_count": len(candidates),
    }
    row = {
        "cell_id": f"{model['key']}:{game_id}",
        "game_id": game_id,
        "model_id": model["hf_id"],
        "model_path": model["model_path"],
        "gpu": model["gpu"],
        "seed": 709900 + index,
        "budget": {"actions": 1, "tokens": 128, "timeout_s": 60},
        "observation_hash": f"sha256:{(index + 10):064x}",
        "candidate_actions": candidates,
        "simulation_invocation": invocation,
        "forecast": forecast,
        "interpretation": {
            "accepted": True,
            "selected_candidate_index": index % 2,
            "forecast_hash": experiment.sha256_json(forecast),
        },
        "selected_action": selected,
        "forecast_consumed": True,
        "policy_class": "E3AgentPolicy",
        "factory": "make_carnot_agent",
        "valid_action": True,
        "advice_only": False,
        "transition": {
            "executed": True,
            "action": selected,
            "observation_before_hash": f"sha256:{(index + 10):064x}",
            "observation_after_hash": f"sha256:{(index + 20):064x}",
            "environment_return_hash": f"sha256:{(index + 20):064x}",
            "level_before": 0,
            "level_after": 0,
            "exact": True,
            "fresh_replay_exact": True,
        },
        "latency_s": 1.25,
        "forbidden_import_rows": experiment.clean_forbidden_import_rows(),
        "forbidden_read_rows": experiment.clean_forbidden_read_rows(),
        "isolation": {
            "worker_start_method": "spawn",
            "policy_process_separate_from_environment": True,
            "passed": True,
        },
    }
    raw_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    row["raw_trace_path"] = str(raw_path)
    row["raw_trace_hash"] = experiment.sha256_file(raw_path)
    return row


def _positive_artifact(tmp_path: Path) -> dict[str, object]:
    models = _model_files(tmp_path)
    games = ["aa01", "bb02", "cc03", "dd04"]
    rows = []
    for model_index, model in enumerate(models):
        for game_index, game in enumerate(games[model_index * 2 : model_index * 2 + 2]):
            rows.append(
                _action_row(
                    model,
                    game,
                    index=model_index * 2 + game_index,
                    raw_path=tmp_path / f"raw-{model_index}-{game_index}.json",
                )
            )
    return experiment.build_completed_artifact(
        execution_date="20260907",
        duration_s=12.5,
        preconditions=[experiment.gate_row("all_preconditions", True, True)],
        source_hashes={"module": "sha256:" + "a" * 64},
        model_specs=models,
        frozen_game_ids=games,
        registry_rows=[
            {**row, "eligible": True, "excluded_reason": None} for row in _registry_rows()[:4]
        ],
        action_rows=rows,
        gpu_rows=[
            {"gpu": 0, "model": "NVIDIA GeForce RTX 3090", "inside_inference": True},
            {"gpu": 1, "model": "NVIDIA GeForce RTX 3090", "inside_inference": True},
        ],
        cleanup_rows=[{"gpu": 0, "released": True}, {"gpu": 1, "released": True}],
    )


def test_req_arc_7099_spec_exists_before_implementation() -> None:
    """REQ-ARC-7099 and all named scenarios exist before implementation."""

    text = (ROOT / "openspec/capabilities/arc-agi/spec.md").read_text(encoding="utf-8")
    assert "REQ-ARC-7099" in text
    for suffix in (
        "BLOCKED",
        "FREEZE",
        "ISOLATION",
        "E3-ACTION",
        "LEDGER",
        "TRANSITION",
        "ADVERSARIAL",
        "NONCLAIM",
    ):
        assert f"SCENARIO-ARC-7099-{suffix}" in text


def test_model_resolution_requires_exact_pinned_ids_without_substitution(tmp_path: Path) -> None:
    """SCENARIO-ARC-7099-E3-ACTION rejects a substitute for either pinned model."""

    models = _model_files(tmp_path)
    resolved = experiment.resolve_model_specs(
        qwen38_resolver=lambda _hf_id: str(models[0]["model_path"]),
        qwen36_resolver=lambda _hf_id: str(models[1]["model_path"]),
        gpu_indices=(3, 4),
    )

    assert [row["hf_id"] for row in resolved] == [
        experiment.QWEN38_HF_ID,
        experiment.QWEN36_HF_ID,
    ]
    assert [row["gpu"] for row in resolved] == [3, 4]
    assert experiment.resolve_model_specs(
        qwen38_resolver=lambda _hf_id: None,
        qwen36_resolver=lambda _hf_id: str(models[1]["model_path"]),
    ) == []
    attacked = deepcopy(resolved)
    attacked[1]["hf_id"] = "legacy/substitute"
    assert experiment.validate_model_specs(attacked)


def test_model_pin_and_hash_defensive_branches_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-7099 covers missing, duplicate, and wrong-placement model receipts."""

    models = _model_files(tmp_path)
    monkeypatch.setattr(
        experiment,
        "cached_sota_pair",
        lambda gpu_indices: [{"hf_id": experiment.QWEN36_HF_ID, "model_path": models[1]["model_path"]}],
    )
    assert experiment._default_qwen36_resolver(experiment.QWEN36_HF_ID) == models[1]["model_path"]
    monkeypatch.setattr(experiment, "cached_sota_pair", lambda gpu_indices: [])
    assert experiment._default_qwen36_resolver(experiment.QWEN36_HF_ID) is None
    assert experiment.sha256_file(tmp_path / "missing.gguf") is None

    missing = deepcopy(models)
    missing[0]["model_path"] = str(tmp_path / "missing.gguf")
    assert "model_file_missing" in experiment.validate_model_specs(missing)
    duplicate = deepcopy(models)
    duplicate[1]["model_path"] = duplicate[0]["model_path"]
    assert "model_path_not_distinct" in experiment.validate_model_specs(duplicate)
    duplicate[1]["gpu"] = duplicate[0]["gpu"]
    assert "model_gpu_not_distinct" in experiment.validate_model_specs(duplicate)


def test_registry_precheck_excludes_prior_credit_and_freezes_strata() -> None:
    """SCENARIO-ARC-7099-FREEZE fixes four strata before any model outcome exists."""

    frozen, rows = experiment.freeze_registry_games(
        _registry_rows(),
        credited_receipts=[{"game_id": "bb02", "level": 0, "credited": True}],
        minimum=4,
    )

    assert len(frozen) == 4
    assert "bb02" not in frozen
    assert len({row["mechanic_class"] for row in rows if row["eligible"]}) >= 2
    assert len({row["registry_depth"] for row in rows if row["selected"]}) >= 2
    assert next(row for row in rows if row["game_id"] == "bb02")["excluded_reason"]
    with pytest.raises(ValueError, match="four"):
        experiment.freeze_registry_games(_registry_rows()[:3], credited_receipts=[], minimum=4)


def test_registry_freeze_rejects_degenerate_strata() -> None:
    """SCENARIO-ARC-7099-FREEZE requires both mechanic and registry-depth variation."""

    one = [{"game": "aa01", "levels_reproduced": 1, "mechanic_class": "navigation"}]
    with pytest.raises(ValueError, match="mechanic"):
        experiment.freeze_registry_games(one, credited_receipts=[], minimum=1)
    with pytest.raises(ValueError, match="mechanic"):
        experiment.freeze_registry_games([*one, *one], credited_receipts=[], minimum=2)
    same_mechanic = [
        {"game": f"aa0{index}", "levels_reproduced": 1, "mechanic_class": "navigation"}
        for index in range(5)
    ]
    with pytest.raises(ValueError, match="mechanic"):
        experiment.freeze_registry_games(same_mechanic, credited_receipts=[], minimum=4)
    same_depth = [
        {"game": f"bb0{index}", "levels_reproduced": 1, "mechanic_class": f"m{index % 2}"}
        for index in range(5)
    ]
    with pytest.raises(ValueError, match="depth"):
        experiment.freeze_registry_games(same_depth, credited_receipts=[], minimum=4)


def test_isolation_monitor_blocks_imports_adapter_lookup_and_forbidden_reads(tmp_path: Path) -> None:
    """SCENARIO-ARC-7099-ISOLATION fails closed on every withheld knowledge path."""

    monitor = experiment.IsolationMonitor(ROOT)
    with pytest.raises(experiment.IsolationViolation, match="forbidden import"):
        monitor.check_import("carnot.agentic.arc_game_adapters")
    with pytest.raises(experiment.IsolationViolation, match="forbidden read"):
        monitor.check_read(ROOT / "ops/arc_solve_registry.yaml")
    with pytest.raises(experiment.IsolationViolation, match="forbidden read"):
        monitor.check_read(ROOT / "environment_files/r11l/fake.py")
    with pytest.raises(experiment.IsolationViolation, match="forbidden read"):
        monitor.check_read(ROOT / "results/arc_loop_solve_r11l.json")
    with pytest.raises(experiment.IsolationViolation, match="forbidden read"):
        monitor.check_read(ROOT / "models/arc_per_game/r11l.json")
    allowed = tmp_path / "observation.json"
    allowed.write_text("{}", encoding="utf-8")
    assert monitor.check_read(allowed) is None
    assert len(monitor.forbidden_import_rows) == 1
    assert len(monitor.forbidden_read_rows) == 4
    assert all(row["blocked"] is True for row in monitor.forbidden_read_rows)


def test_isolation_monitor_activated_guards_and_restores_process_hooks(tmp_path: Path) -> None:
    """SCENARIO-ARC-7099-ISOLATION enforces guards only during the decision interval."""

    allowed = tmp_path / "observation.json"
    allowed.write_text("{}", encoding="utf-8")
    solver = ROOT / "scripts/fake_solver.py"
    monitor = experiment.IsolationMonitor(ROOT)
    original_import, original_open, original_io_open = builtins.__import__, builtins.open, io.open

    class BrokenPath:
        def __fspath__(self) -> str:
            raise TypeError("not path-like")

    assert monitor.check_import("json") is None
    assert monitor.check_read(BrokenPath()) is None
    with monitor.activated():
        assert __import__("json") is json
        with builtins.open(allowed, encoding="utf-8") as handle:
            assert handle.read() == "{}"
        with io.open(allowed, encoding="utf-8") as handle:  # noqa: UP020 - exercises io.open guard
            assert handle.read() == "{}"
        with pytest.raises(experiment.IsolationViolation, match="forbidden import"):
            __import__("carnot.agentic.arc_game_adapters")
        with pytest.raises(experiment.IsolationViolation, match="forbidden read"):
            builtins.open(solver, encoding="utf-8")
    assert (builtins.__import__, builtins.open, io.open) == (original_import, original_open, original_io_open)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Try moving right because that often works.", "advice_only"),
        ('{"selected_candidate_index":9,"predicted_change":true,"rationale":"x"}', "invalid_action"),
    ],
)
def test_advice_only_and_invalid_action_outputs_fail_closed(text: str, reason: str) -> None:
    """SCENARIO-ARC-7099-ADVERSARIAL never turns prose or invalid output into an action."""

    result = experiment.interpret_forecast(
        text,
        candidate_actions=[{"action": 1, "data": None}],
    )

    assert result["accepted"] is False
    assert result["reason"] == reason


def test_valid_forecast_consumes_one_generated_candidate() -> None:
    """SCENARIO-ARC-7099-LEDGER binds a model forecast to the selected action."""

    candidates = [{"action": 1, "data": None}, {"action": 6, "data": {"x": 2, "y": 3}}]
    result = experiment.interpret_forecast(
        '<forecast>{"selected_candidate_index":1,"predicted_change":true,'
        '"rationale":"The marked object can react."}</forecast>',
        candidate_actions=candidates,
    )

    assert result["accepted"] is True
    assert result["selected_action"] == candidates[1]
    assert result["forecast_hash"] == experiment.sha256_json(result["forecast"])


def test_malformed_forecast_payloads_fail_closed() -> None:
    """SCENARIO-ARC-7099-ADVERSARIAL rejects broken JSON and incomplete forecasts."""

    candidates = [{"action": 1, "data": None}]
    assert experiment.interpret_forecast("{not-json", candidate_actions=candidates)["reason"] == "advice_only"
    malformed = '{"selected_candidate_index":0,"predicted_change":"yes","rationale":"x"}'
    assert experiment.interpret_forecast(malformed, candidate_actions=candidates)["reason"] == "malformed_forecast"


def test_action_row_rejects_model_substitution_trace_gaps_and_invalid_action(tmp_path: Path) -> None:
    """SCENARIO-ARC-7099-ADVERSARIAL recomputes every causal action receipt."""

    models = _model_files(tmp_path)
    row = _action_row(models[0], "aa01", index=0, raw_path=tmp_path / "raw.json")
    assert experiment.validate_action_row(row, models) == []

    attacks = {
        "model_substitution": lambda value: value.update(model_id="legacy/model"),
        "advice_only": lambda value: value.update(advice_only=True),
        "invalid_action": lambda value: value.update(selected_action={"action": 7, "data": None}),
        "forecast_not_consumed": lambda value: value.update(forecast_consumed=False),
        "missing_simulation": lambda value: value.update(simulation_invocation={"invoked": False}),
        "forecast_ledger_incomplete": lambda value: value.update(forecast=None),
        "forecast_hash_mismatch": lambda value: value["interpretation"].update(forecast_hash="sha256:bad"),
        "transition_not_executed": lambda value: value.update(transition={}),
        "inexact_transition": lambda value: value["transition"].update(exact=False),
        "e3_factory_path_missing": lambda value: value.update(policy_class="CarnotAgentPolicy"),
        "adapter_lookup": lambda value: value["forbidden_import_rows"].append(
            {"target": "arc_game_adapters", "blocked": False, "passed": False}
        ),
        "forbidden_read": lambda value: value.update(forbidden_read_rows=[]),
        "process_isolation_missing": lambda value: value.update(isolation={"passed": False}),
    }
    for expected, mutate in attacks.items():
        attacked = deepcopy(row)
        mutate(attacked)
        assert any(expected in error for error in experiment.validate_action_row(attacked, models))

    assert experiment._ready_score(row and [row], [{**models[0], "hf_id": "substitute"}, models[1]]) == 0
    broken = deepcopy(row)
    broken["forecast_consumed"] = False
    second = deepcopy(row)
    second["game_id"] = "bb02"
    assert experiment._ready_score([broken, second], models) == 0


def test_public_click_action_carries_generic_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-7099-E3-ACTION sends generic click data through the public action API."""

    class Member:
        payload = None

        @classmethod
        def set_data(cls, payload: object) -> None:
            cls.payload = payload

    class Environment:
        observation_space = SimpleNamespace(game_id="aa01-public")

        def step(self, member: object, *, data: object) -> tuple[object, object]:
            return member, data

    monkeypatch.setitem(sys.modules, "arcengine", SimpleNamespace(GameAction=SimpleNamespace(ACTION6=Member)))
    data = {"x": 2, "y": 3}
    assert experiment._apply_public_action(Environment(), {"action": 6, "data": data}) == (Member, data)
    assert Member.payload == {"game_id": "aa01-public", **data}


def test_real_valid_action_has_an_exact_fresh_environment_transition() -> None:
    """SCENARIO-ARC-7099-TRANSITION executes and replays one real offline ARC action."""

    from carnot.agentic.arc_solver_kit import offline_arcade

    arcade = offline_arcade()
    game_id = next(str(info.game_id) for info in arcade.available_environments if str(info.game_id).startswith("tu93-"))
    action = {"action": 1, "data": None}
    receipt = experiment.execute_and_replay_action(
        arcade=arcade,
        game_id=game_id,
        seed=7099,
        action=action,
    )

    assert receipt["executed"] is True
    assert receipt["valid_action"] is True
    assert receipt["exact"] is True
    assert receipt["fresh_replay_exact"] is True
    assert receipt["action"] == action
    assert receipt["environment_return_hash"].startswith("sha256:")


def test_ready_score_requires_two_games_for_each_model_and_complete_ledgers(tmp_path: Path) -> None:
    """REQ-ARC-7099 computes readiness only from complete rows for both model arms."""

    artifact = _positive_artifact(tmp_path)
    assert artifact["adapter_withheld_live_path_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["arc_registry_delta"] == 0
    assert artifact["offline_reproduced"] is True
    assert artifact["valid_action_row_count"] == 4
    assert artifact["advice_only_row_count"] == 0
    assert experiment.validate_artifact(artifact, verify_raw_traces=True) == []

    one_game_missing = deepcopy(artifact)
    one_game_missing["action_rows"] = one_game_missing["action_rows"][:-1]
    one_game_missing["rows"] = one_game_missing["rows"][:-1]
    one_game_missing["reproducibility_checksum"] = experiment.artifact_checksum(one_game_missing)
    assert "ready_score_mismatch" in experiment.validate_artifact(one_game_missing)


def test_raw_trace_tampering_is_detected_after_aggregation(tmp_path: Path) -> None:
    """SCENARIO-ARC-7099-ADVERSARIAL binds aggregate rows to raw bytes outside results."""

    artifact = _positive_artifact(tmp_path)
    raw_path = Path(artifact["raw_trace_paths"][0])
    raw_path.write_text('{"tampered":true}', encoding="utf-8")

    errors = experiment.validate_artifact(artifact, verify_raw_traces=True)
    assert "raw_trace_hash_mismatch" in errors


def test_blocked_artifact_is_schema_complete_and_names_exact_failed_gate() -> None:
    """SCENARIO-ARC-7099-BLOCKED preserves exact missing-resource evidence."""

    checks = [experiment.gate_row("two_idle_rtx3090_leases", 2, 1)]
    artifact = experiment.build_blocked_artifact(
        execution_date="20260907",
        duration_s=0.25,
        preconditions=checks,
        source_hashes={},
    )

    assert set(experiment.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(experiment.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == "two_idle_rtx3090_leases"
    assert artifact["gate_check_summary"]["expected_value"] == 2
    assert artifact["gate_check_summary"]["observed_value"] == 1
    assert experiment.validate_artifact(artifact) == []


def test_artifact_validator_rejects_schema_score_checksum_and_nonclaim_attacks(tmp_path: Path) -> None:
    """SCENARIO-ARC-7099-NONCLAIM keeps readiness separate from solve credit."""

    artifact = _positive_artifact(tmp_path)
    attacks = {
        "missing_field:rows": lambda value: value.pop("rows"),
        "field_principles_incomplete": lambda value: value.update(field_principles={}),
        "ready_score_mismatch": lambda value: value.update(adapter_withheld_live_path_ready_score=0),
        "solve_provenance_invalid": lambda value: value.update(solve_provenance="live_agent_self_discovery"),
        "arc_registry_delta_invalid": lambda value: value.update(arc_registry_delta=1),
        "verifier_is_oracle_invalid": lambda value: value.update(verifier_is_oracle=True),
        "verdict_class_invalid": lambda value: value.update(verdict_class="invented"),
        "honest_verdict_invalid": lambda value: value.update(honest_verdict="positive result"),
        "advice_only_count_mismatch": lambda value: value.update(advice_only_row_count=1),
        "offline_reproduced_mismatch": lambda value: value.update(offline_reproduced=False),
        "reproducibility_checksum_mismatch": lambda value: value.update(random_seed=2),
    }
    for expected, mutate in attacks.items():
        attacked = deepcopy(artifact)
        mutate(attacked)
        if expected != "reproducibility_checksum_mismatch":
            attacked["reproducibility_checksum"] = experiment.artifact_checksum(attacked)
        assert expected in experiment.validate_artifact(attacked)

    assert experiment.validate_artifact([]) == ["artifact_not_object"]
    blocked = experiment.build_blocked_artifact(
        execution_date="20260907",
        duration_s=0.1,
        preconditions=[experiment.gate_row("missing", True, False)],
        source_hashes={},
    )
    blocked["inference_substrate_class"] = "model_full_generation"
    blocked["reproducibility_checksum"] = experiment.artifact_checksum(blocked)
    assert "blocked_contract_invalid" in experiment.validate_artifact(blocked)


@pytest.mark.memory_watchdog_skip
def test_production_reachability_uses_factory_and_e3_policy() -> None:
    """SCENARIO-ARC-7099-E3-ACTION inspects the scored factory, not arc_loop advice."""

    rows = experiment.production_reachability_rows()

    assert rows
    assert all(row["passed"] is True for row in rows)
    assert {row["symbol"] for row in rows} >= {
        "make_carnot_agent",
        "E3AgentPolicy",
        "LocalGGUFProposer",
    }
    assert all("arc_loop_solve.needs_re" not in str(row) for row in rows)


def test_identity_router_override_is_confined_to_spawned_worker() -> None:
    """SCENARIO-ARC-7099-ISOLATION removes registry routing only in the isolated policy process."""

    worker_source = inspect.getsource(experiment._live_worker_entry)
    reachability_source = inspect.getsource(experiment.production_reachability_rows)
    assert "competition._recommend_live_approach = withheld_approach" in worker_source
    assert "registry_consulted" in worker_source
    assert "_recommend_live_approach" not in reachability_source
