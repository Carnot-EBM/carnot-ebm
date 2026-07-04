"""Tests for Exp 5241 ARC gated live patch attempt.

Spec refs: REQ-REPORT-5241,
SCENARIO-REPORT-5241-NO-BANK-LIVE-PATCH-ATTEMPT,
SCENARIO-REPORT-5241-SOLVE-CLAIM-GATE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5241_arc_gated_live_patch_attempt_v479 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _registry(total: int = 69) -> dict[str, object]:
    return {
        "present": True,
        "path": mod.REGISTRY_RELATIVE_PATH,
        "reproducible_total_levels": total,
        "games": {"lp85": 5, "tu93": 5},
    }


def _preconditions() -> dict[str, object]:
    return {
        "agents_read": True,
        "codex_read": True,
        "spec_has_req_5241": True,
        "exp5240_patch_candidate_tested": True,
        "registry_present": True,
        "registry_loadable": True,
        "patch_path_matches_exp5240": True,
        "read_hidden_game_source": False,
        "offline_ground_truth_bfs": False,
        "hand_per_game_adapter": False,
    }


def _validation_commands(*, registry_passed: bool = True) -> list[dict[str, object]]:
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5241_arc_gated_live_patch_attempt_v479.py -q",
            "passed": True,
        },
        {
            "command": "python scripts/arc_count_integrity_lint.py ops/arc_solve_registry.yaml --skip-replay --json",
            "passed": registry_passed,
        },
    ]


def _attempt(
    *,
    claimed_level: int = 0,
    reproduced: bool = False,
    registry_validation_passed: bool = False,
    solution_labels: list[str] | None = None,
) -> dict[str, object]:
    return {
        "attempt_id": "exp5241_zz99_exp5241_live_probe_seed_5241_budget_8",
        "target_game": mod.DEFAULT_TARGET_GAME,
        "target_level": 1,
        "prior_reproduced_level": 0,
        "budget": 8,
        "random_seed": mod.RANDOM_SEED,
        "runtime_s": 0.01,
        "exact_command": mod.DEFAULT_EXACT_COMMAND,
        "policy": "arc_competition_agent._recommend_live_approach",
        "self_discovery_lever": "exp5240_provenance_routing_guard",
        "live_agent_patch_enabled": True,
        "runtime_self_discovery_attempted": True,
        "solution_labels": solution_labels or [],
        "reproduction_gate": {
            "claimed_level": claimed_level,
            "reproduced": reproduced,
            "registry_validation_passed": registry_validation_passed,
            "reached_level": claimed_level if reproduced else 0,
        },
        "model_ids": [],
        "llm_proposer_used": False,
        "model_specs": None,
        "forbidden_methods": {
            "read_hidden_game_source": False,
            "offline_ground_truth_bfs": False,
            "hand_per_game_adapter": False,
        },
        "process_deltas": {
            "skill_selection": "selected provenance-routing guard",
            "skill_following": "honored live self-discovery gate",
            "composition": "composed guard with strategy fallback",
            "reflection": "no level banked in fixture",
        },
        "approach_recommendation": {
            "typed_memory_provenance_guard": {"enabled": True},
            "strategy": {"name": "graph_explore"},
        },
    }


def test_req_report_5241_spec_declares_live_patch_contract() -> None:
    """REQ-REPORT-5241: OpenSpec anchors the Exp 5241 artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5241") : spec.index("### REQ-REPORT-5162")]

    for marker in (
        "REQ-REPORT-5241",
        "SCENARIO-REPORT-5241-NO-BANK-LIVE-PATCH-ATTEMPT",
        "SCENARIO-REPORT-5241-SOLVE-CLAIM-GATE",
        mod.RESULT_RELATIVE_PATH,
        "arc_live_agent_self_discovery",
        "live_agent_self_discovery",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_report_5241_no_bank_artifact_keeps_registry_total() -> None:
    """SCENARIO-REPORT-5241-NO-BANK-LIVE-PATCH-ATTEMPT: no-bank stays honest."""

    artifact = mod.build_artifact(
        precondition_audit=_preconditions(),
        registry_summary=_registry(),
        live_attempt=_attempt(),
        arc_validation_commands=_validation_commands(),
        duration_s=0.25,
    )

    assert artifact["preconditions_checked"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["registry_precheck_done"] is True
    assert artifact["duplicate_solve_target_avoided"] is True
    assert artifact["reproducible_total_levels_before"] == 69
    assert artifact["reproducible_total_levels_after"] == 69
    assert artifact["reproducible_total_levels_delta"] == 0
    assert artifact["live_agent_patch_enabled"] is True
    assert artifact["model_specs"] is None
    assert artifact["model_ids"] == []
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["patch_recommendation"] == "no_solve_no_regression"
    assert artifact["inference_substrate"] == "arc_live_agent_self_discovery"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "level_delta=0" in artifact["honest_verdict"]
    assert artifact["process_deltas"]["skill_following"]
    mod.validate_artifact(artifact)


def test_scenario_report_5241_success_requires_live_registry_validation() -> None:
    """SCENARIO-REPORT-5241-SOLVE-CLAIM-GATE: registry validation gates banking."""

    unvalidated = mod.build_artifact(
        precondition_audit=_preconditions(),
        registry_summary=_registry(),
        live_attempt=_attempt(
            claimed_level=1,
            reproduced=True,
            registry_validation_passed=False,
            solution_labels=['{"action": 1, "data": null}'],
        ),
        arc_validation_commands=_validation_commands(registry_passed=False),
        duration_s=0.25,
    )

    assert unvalidated["honest_verdict"].startswith("complete:")
    assert unvalidated["reproducible_total_levels_delta"] == 0
    assert unvalidated["solve_claim"]["claimed"] is False
    assert unvalidated["patch_recommendation"] == "iterate"
    mod.validate_artifact(unvalidated)

    success = mod.build_artifact(
        precondition_audit=_preconditions(),
        registry_summary=_registry(),
        live_attempt=_attempt(
            claimed_level=1,
            reproduced=True,
            registry_validation_passed=True,
            solution_labels=['{"action": 1, "data": null}'],
        ),
        arc_validation_commands=_validation_commands(registry_passed=True),
        duration_s=0.25,
    )

    assert success["honest_verdict"].startswith("success:")
    assert success["reproducible_total_levels_before"] == 69
    assert success["reproducible_total_levels_after"] == 70
    assert success["reproducible_total_levels_delta"] == 1
    assert success["solve_claim"]["claimed"] is True
    assert success["duplicate_solve_target_avoided"] is True
    assert success["patch_recommendation"] == "keep"
    mod.validate_artifact(success)


def test_req_report_5241_validation_edges_fail_closed() -> None:
    """REQ-REPORT-5241: malformed artifacts fail schema validation."""

    artifact = mod.build_artifact(
        precondition_audit=_preconditions(),
        registry_summary=_registry(),
        live_attempt=_attempt(),
        arc_validation_commands=_validation_commands(),
        duration_s=0.25,
    )

    missing = dict(artifact)
    missing.pop("preconditions_checked")
    assert "missing required field: preconditions_checked" in mod.artifact_schema_errors(missing)

    bad = dict(
        artifact,
        preconditions_checked="true",
        solve_provenance="development_proxy",
        registry_precheck_done="true",
        duplicate_solve_target_avoided="true",
        reproducible_total_levels_before="69",
        reproducible_total_levels_after="70",
        reproducible_total_levels_delta=1,
        live_agent_patch_enabled="true",
        model_specs=[],
        random_seed=0,
        arc_validation_commands={},
        patch_recommendation="promote",
        inference_substrate="offline",
        honest_verdict="pending",
    )
    errors = mod.artifact_schema_errors(bad)
    for expected in (
        "preconditions_checked must be bare bool",
        "solve_provenance must be live_agent_self_discovery",
        "registry_precheck_done must be bare bool",
        "duplicate_solve_target_avoided must be bare bool",
        "reproducible_total_levels_before must be bare int",
        "live_agent_patch_enabled must be bare bool",
        "model_specs must be null when no LLM proposer was used",
        "random_seed mismatch",
        "arc_validation_commands must be a list",
        "patch_recommendation must be one of",
        "inference_substrate mismatch",
        "honest_verdict must use a terminal prefix",
    ):
        assert expected in errors

    mismatch = dict(artifact, reproducible_total_levels_after=70)
    mismatch["reproducibility_checksum"] = mod.reproducibility_checksum(mismatch)
    assert "reproducible_total_levels_after must equal before + delta" in (
        mod.artifact_schema_errors(mismatch)
    )

    with pytest.raises(ValueError, match="missing required field: preconditions_checked"):
        mod.validate_artifact(missing)


def test_req_report_5241_precondition_io_and_blocked_run(tmp_path: Path) -> None:
    """REQ-REPORT-5241: precondition checks read the gate inputs and block cleanly."""

    ready = tmp_path / "ready"
    (ready / "ops").mkdir(parents=True)
    (ready / "results").mkdir()
    (ready / "openspec" / "capabilities" / "research-reporting").mkdir(parents=True)
    (ready / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (ready / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (ready / mod.SPEC_RELATIVE_PATH).write_text("REQ-REPORT-5241\n", encoding="utf-8")
    (ready / mod.REGISTRY_RELATIVE_PATH).write_text(
        "schema_version: 1\nreproducible_total_levels: 2\ngames:\n- game: lp85\n  levels_reproduced: 2\n",
        encoding="utf-8",
    )
    (ready / mod.EXP5240_RESULT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "recommended_live_patch_available": True,
                "patch_test_ready": True,
                "patch_path": mod.PATCH_RELATIVE_PATH,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert mod.load_registry_summary(ready)["games"] == {"lp85": 2}
    assert mod.check_preconditions(ready)["exp5240_patch_candidate_tested"] is True

    missing = tmp_path / "missing"
    blocked = mod.run_experiment(
        root=missing,
        result_path=tmp_path / "blocked" / mod.RESULT_RELATIVE_PATH,
    )
    assert blocked["honest_verdict"].startswith("blocked_preconditions_failed")
    assert blocked["preconditions_checked"] is False
    assert blocked["patch_recommendation"] == "rollback"
    assert blocked["process_deltas"]["reflection"] == "preconditions blocked the live patch attempt"
    mod.validate_artifact(blocked)

    malformed = tmp_path / "malformed"
    (malformed / "ops").mkdir(parents=True)
    (malformed / mod.REGISTRY_RELATIVE_PATH).write_text("not: [valid", encoding="utf-8")
    assert mod.load_registry_summary(malformed)["present"] is False


def test_req_report_5241_residual_branches_stay_no_bank() -> None:
    """REQ-REPORT-5241: duplicate and residual branches do not inflate totals."""

    duplicate = mod.build_artifact(
        precondition_audit=_preconditions(),
        registry_summary={
            "present": True,
            "path": mod.REGISTRY_RELATIVE_PATH,
            "reproducible_total_levels": 69,
            "games": {mod.DEFAULT_TARGET_GAME: 1},
        },
        live_attempt=_attempt(),
        arc_validation_commands=_validation_commands(),
        duration_s=0.25,
    )
    assert duplicate["duplicate_solve_target_avoided"] is False

    patch_disabled = dict(_attempt(), live_agent_patch_enabled=False)
    rollback = mod.build_artifact(
        precondition_audit=_preconditions(),
        registry_summary=_registry(),
        live_attempt=patch_disabled,
        arc_validation_commands=_validation_commands(),
        duration_s=0.25,
    )
    assert rollback["patch_recommendation"] == "rollback"
    assert "patch_not_enabled" in rollback["honest_verdict"]

    forbidden_attempt = dict(_attempt())
    forbidden_attempt["forbidden_methods"] = {
        "read_hidden_game_source": False,
        "offline_ground_truth_bfs": True,
        "hand_per_game_adapter": False,
    }
    forbidden = mod.build_artifact(
        precondition_audit=_preconditions(),
        registry_summary=_registry(),
        live_attempt=forbidden_attempt,
        arc_validation_commands=_validation_commands(),
        duration_s=0.25,
    )
    assert "forbidden_method_used" in forbidden["honest_verdict"]
    assert "forbidden methods must be false" in mod.artifact_schema_errors(forbidden)

    missing_labels = mod.build_artifact(
        precondition_audit=_preconditions(),
        registry_summary=_registry(),
        live_attempt=_attempt(claimed_level=1, reproduced=True, registry_validation_passed=True),
        arc_validation_commands=_validation_commands(),
        duration_s=0.25,
    )
    assert missing_labels["patch_recommendation"] == "iterate"
    assert "missing_live_solution_labels" in missing_labels["honest_verdict"]


def test_req_report_5241_schema_edge_coverage() -> None:
    """REQ-REPORT-5241: schema diagnostics cover metadata and solve-claim edges."""

    artifact = mod.build_artifact(
        precondition_audit=_preconditions(),
        registry_summary=_registry(),
        live_attempt=_attempt(),
        arc_validation_commands=_validation_commands(),
        duration_s=0.25,
    )

    bad_meta = dict(
        artifact,
        schema="bad",
        experiment="bad",
        experiment_id=0,
        spec_refs=[],
        field_principles={},
        solve_claim=[],
        honest_verdict=123,
        reproducibility_checksum="bad",
    )
    meta_errors = mod.artifact_schema_errors(bad_meta)
    for expected in (
        "schema mismatch",
        "experiment mismatch",
        "experiment_id mismatch",
        "spec_refs mismatch",
        "field_principles mismatch",
        "solve_claim must be a mapping",
        "honest_verdict must use a terminal prefix",
        "reproducibility_checksum must be 64 hex chars",
    ):
        assert expected in meta_errors

    llm_bad = dict(artifact, llm_proposer_used=True, model_specs=[{"hf_id": "small"}])
    assert "model_specs must include a mandated SOTA GGUF when LLM proposer was used" in (
        mod.artifact_schema_errors(llm_bad)
    )

    llm_ok = dict(
        artifact,
        llm_proposer_used=True,
        model_specs=[{"hf_id": mod.MANDATED_SOTA_GGUFS[0]}],
    )
    llm_ok["reproducibility_checksum"] = mod.reproducibility_checksum(llm_ok)
    assert "model_specs must include a mandated SOTA GGUF when LLM proposer was used" not in (
        mod.artifact_schema_errors(llm_ok)
    )

    non_bool_claim = dict(artifact, solve_claim={"claimed": "yes"})
    non_bool_claim["reproducibility_checksum"] = mod.reproducibility_checksum(non_bool_claim)
    assert "solve_claim.claimed must be bare bool" in mod.artifact_schema_errors(non_bool_claim)

    weak_success = dict(
        artifact,
        honest_verdict="success: weak",
        solve_provenance="development_proxy",
        solve_claim={"claimed": False},
    )
    weak_success["reproducibility_checksum"] = mod.reproducibility_checksum(weak_success)
    weak_errors = mod.artifact_schema_errors(weak_success)
    assert "success requires live_agent_self_discovery provenance" in weak_errors
    assert "success requires positive level delta" in weak_errors
    assert "success requires solve_claim.claimed true" in weak_errors

    non_success_delta = dict(
        artifact,
        reproducible_total_levels_after=70,
        reproducible_total_levels_delta=1,
    )
    non_success_delta["reproducibility_checksum"] = mod.reproducibility_checksum(
        non_success_delta
    )
    assert "non-success artifacts must not change registry totals" in mod.artifact_schema_errors(
        non_success_delta
    )


@pytest.mark.memory_watchdog_skip
def test_req_report_5241_live_agent_attempt_reaches_exp5240_guard() -> None:
    """REQ-REPORT-5241: the live recommendation path reaches the enabled patch."""

    attempt = mod.run_live_agent_patch_attempt(
        root=REPO,
        target_game=mod.DEFAULT_TARGET_GAME,
        budget=2,
        random_seed=mod.RANDOM_SEED,
        exact_command=mod.DEFAULT_EXACT_COMMAND,
    )

    guard = attempt["approach_recommendation"]["typed_memory_provenance_guard"]
    assert attempt["live_agent_patch_enabled"] is True
    assert guard["enabled"] is True
    assert attempt["llm_proposer_used"] is False
    assert attempt["model_specs"] is None
    assert attempt["model_ids"] == []
    assert attempt["forbidden_methods"] == {
        "read_hidden_game_source": False,
        "offline_ground_truth_bfs": False,
        "hand_per_game_adapter": False,
    }


def test_req_report_5241_run_experiment_writes_terminal_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5241: runner writes the required artifact fields."""

    monkeypatch.setattr(mod, "check_preconditions", lambda _root: _preconditions())
    monkeypatch.setattr(mod, "load_registry_summary", lambda _root: _registry())
    monkeypatch.setattr(mod, "run_live_agent_patch_attempt", lambda **_kwargs: _attempt())
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run_experiment(
        root=tmp_path,
        result_path=result_path,
        arc_validation_commands=_validation_commands(),
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_report_5241_repository_artifact_is_valid() -> None:
    """REQ-REPORT-5241: checked-in artifact remains schema-valid and replayable."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(result)
    assert result["preconditions_checked"] is True
    assert result["solve_provenance"] == "live_agent_self_discovery"
    assert result["inference_substrate"] == "arc_live_agent_self_discovery"
