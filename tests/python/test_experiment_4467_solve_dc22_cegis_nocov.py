"""Tests for Exp 4467 dc22 CEGIS config-rule grounding.

Spec refs: REQ-REPORT-4467, SCENARIO-REPORT-4467.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest
import yaml

from carnot import experiment_4467_solve_dc22_cegis_nocov as mod
from carnot.agentic import arc_game_adapters, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _examples() -> list[dict[str, str]]:
    return [
        {
            "game": "s5i5",
            "rule_id": "config_toggle_marker_coverage",
            "predicate": "controlled marker sprites occupy target marker coordinates",
        },
        {
            "game": "ft09",
            "rule_id": "local_constraint_color_cycle",
            "predicate": "click cells until local equality and inequality constraints hold",
        },
        {
            "game": "vc33",
            "rule_id": "config_support_clearance",
            "predicate": "support-clearance clicks open the blocking target lane",
        },
        {
            "game": "g50t",
            "rule_id": "config_toggle_target_offset",
            "predicate": "player reaches the target offset and commits the visible toggle",
        },
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "target_game": "dc22",
        "recommended": [
            {
                "game": "vc33",
                "similarity": 6.0,
                "solver": "python/carnot/experiment_4446_drive_generic_first_contact_bank.py",
                "win_condition": "support-clearance config-rule replay",
                "action_model": "ACTION6 click toggles plus navigation",
            }
        ],
        "selected_generic_operators": [{"operator": "config_rule_verifier"}],
        "strategy": {"routed_mechanic": "graph_explore"},
    }


def _ok_preconditions() -> dict[str, Any]:
    return {
        "dc22_environment_files": True,
        "arc_solver_imports": True,
        "qwen_gguf_cache": True,
        "igpu_llama_server": False,
        "generator_resource_available": True,
        "baseline_command": ".venv/bin/pytest -k \"config_rule or arc_solver_kit\" -q --no-cov",
        "baseline_exit_code": 0,
        "baseline_pytest_nocov_green": True,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _write_fixture_repo(root: Path) -> None:
    (root / "environment_files" / "dc22" / "fdcac232").mkdir(parents=True)
    (root / "ops").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "games": [
                    {
                        "game": "s5i5",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "mechanic_class": "config_toggle_marker_coverage",
                        "win_condition": "marker coverage",
                    },
                    {
                        "game": "ft09",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "mechanic_class": "local_constraint_color_cycle",
                        "win_condition": "local color cycle",
                    },
                    {
                        "game": "vc33",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "mechanic_class": "config_support_clearance",
                        "win_condition": "support-clearance config",
                    },
                    {
                        "game": "g50t",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 1,
                        "mechanic_class": "config_toggle_target_offset",
                        "win_condition": "target offset toggle",
                    },
                    {
                        "game": "dc22",
                        "reproducibility": "unsolved",
                        "levels_reproduced": 0,
                        "solver": "scripts/arc_loop_solve.py --game dc22",
                        "dead_ends": [
                            {
                                "gap_id": mod.DC22_GAP_ID,
                                "status": "open",
                                "failure_mode": "needs_per_game_RE",
                            }
                        ],
                    },
                ],
                "reproducible_total_levels": 39,
                "reproducible_total_games": 20,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text(
        "<!-- exp4438-gap-4423-dc22-unselectable-first-contact:start -->\n"
        "### GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT: fixture\n"
        "- status: open\n"
        "<!-- exp4438-gap-4423-dc22-unselectable-first-contact:end -->\n",
        encoding="utf-8",
    )


def _clock() -> Any:
    ticks = iter([0.0, 0.2, 1.1])
    return lambda: next(ticks)


def test_req_report_4467_spec_declares_dc22_cegis_contract() -> None:
    """REQ-REPORT-4467: OpenSpec declares the dc22 CEGIS artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4467" in spec
    assert "SCENARIO-REPORT-4467" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "counterexample-guided grounding loop" in spec
    assert "--no-cov" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_report_4467_config_rule_verifier_grounds_dc22_toggle_navigation() -> None:
    """REQ-REPORT-4467: dc22 toggle-navigation digest grounds through config_rule_verifier."""

    result = kit.config_rule_verifier(
        game="dc22",
        object_digest=mod.dc22_toggle_navigation_digest(),
        few_shot_examples=_examples(),
    )

    assert result["grounded"] is True
    assert result["predicate_id"] == "dc22_toggle_navigation"
    assert result["solution"] == mod.DC22_L1_SOLUTION
    assert result["grounded_win_condition"]["fires_on_win"] is True
    assert result["verifier_is_oracle"] is True


def test_req_report_4467_dc22_adapter_is_registered() -> None:
    """REQ-REPORT-4467: the per-game dc22 action/state delta is registered as an adapter."""

    adapter = arc_game_adapters.get_adapter("dc22")

    assert adapter is not None
    assert adapter.game == "dc22"
    assert "dc22" in arc_game_adapters.adaptered_games()
    assert adapter.depth_caps[1] >= len(mod.DC22_L1_SOLUTION)


def test_scenario_report_4467_success_banks_dc22_and_fills_gap(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4467: reproduced dc22 L1 updates the artifact, registry, and gap ledger."""

    _write_fixture_repo(tmp_path)
    reproduced_calls: list[list[str]] = []

    def reproduce(solution: Sequence[str]) -> dict[str, Any]:
        reproduced_calls.append(list(solution))
        return {
            "game": "dc22",
            "claimed_level": 1,
            "reached_level": 1,
            "reproduced": True,
            "mode": "offline_reproduction_gate_no_quota",
        }

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        few_shot_examples=_examples(),
        reproduce_fn=reproduce,
        offline_solver_fn=lambda solution: {
            "solution": list(solution),
            "reached_level": 1,
            "states_expanded": 20,
            "solver": "OfflineSolver(dc22)",
        },
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )

    assert reproduced_calls == [mod.DC22_L1_SOLUTION]
    assert artifact["honest_verdict"] == "success: dc22_cegis_L1_offline_reproduced"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["target_game"] == "dc22"
    assert artifact["dc22_grounded"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["counterexample_rounds"] == 1
    assert artifact["baseline_pytest_nocov_green"] is True
    assert len(artifact["few_shot_examples_used"]) >= 3
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducible_total_levels"] == 40
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    dc22 = next(row for row in registry["games"] if row["game"] == "dc22")
    assert dc22["reproducibility"] == "reproduced"
    assert dc22["levels_reproduced"] == 1
    assert registry["reproducible_total_levels"] == 40
    assert registry["reproducible_total_games"] == 21
    assert dc22["dead_ends"][0]["status"] == "filled"

    gaps = (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "- status: filled (experiment_4467_solve_dc22_cegis_nocov)" in gaps
    assert f"- evidence: {mod.RESULT_RELATIVE_PATH}; target_game=dc22; offline_reproduced=True" in gaps


def test_req_report_4467_blocked_precondition_stops_before_routing(tmp_path: Path) -> None:
    """REQ-REPORT-4467: missing resources write blocked artifacts without routing or reproduction."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={**_ok_preconditions(), "baseline_pytest_nocov_green": False, "ok": False},
        recommend_fn=lambda game: calls.append(game) or _recommendation(),
        reproduce_fn=lambda _solution: pytest.fail("reproduce must not run"),
        offline_solver_fn=lambda _solution: pytest.fail("offline solver must not run"),
        now=lambda: 2.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_baseline_tests_red"
    assert artifact["dc22_grounded"] is False
    assert artifact["offline_reproduced"] is False
    assert artifact["counterexample_rounds"] == 0
    assert artifact["baseline_pytest_nocov_green"] is False
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"dc22_environment_files": False}, "offline_env_dc22"),
        ({"arc_solver_imports": False}, "arc_solver_imports"),
        ({"generator_resource_available": False}, "qwen_generator_resource"),
        ({"no_3090_inference": False}, "no_3090_inference_policy"),
        ({"leaderboard_submission": True}, "leaderboard_submission_policy"),
    ],
)
def test_req_report_4467_precondition_miss_names_resource(
    override: dict[str, Any],
    expected: str,
) -> None:
    """REQ-REPORT-4467: every precondition miss maps to an explicit blocked resource."""

    assert mod.first_precondition_miss({**_ok_preconditions(), **override}) == expected


def test_req_report_4467_registry_helpers_handle_missing_and_malformed_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-4467: registry helpers keep blocked/malformed inputs from fabricating totals."""

    assert mod._load_registry(tmp_path) == {"games": []}
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).parent.mkdir(parents=True)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text("[", encoding="utf-8")

    assert mod._load_registry(tmp_path) == {"games": []}
    assert mod._registry_games({"games": {}}) == []
    assert mod._target_entry({"games": []}) is None
    assert mod._registry_totals(
        {
            "games": [
                {"game": "a", "reproducibility": "reproduced", "levels_reproduced": 2},
                {"game": "b", "reproducibility": "unsolved", "levels_reproduced": 0},
            ]
        }
    ) == {"reproducible_total_levels": 2, "reproducible_total_games": 1}


def test_req_report_4467_cegis_loop_records_early_grounding_and_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-4467: CEGIS rounds distinguish immediate grounding from rejected candidates."""

    grounded = {
        "operator": "config_rule_verifier",
        "game": "dc22",
        "grounded": True,
        "solution": ["ok"],
        "verifier_is_oracle": True,
    }
    monkeypatch.setattr(mod.kit, "config_rule_verifier", lambda **_kwargs: grounded)
    assert mod.counterexample_grounding_loop(few_shot_examples=[], budget=2)["counterexample_rounds"] == 0

    ungrounded = {
        "operator": "config_rule_verifier",
        "game": "dc22",
        "grounded": False,
        "solution": [],
        "residual": "still_missing",
        "verifier_is_oracle": True,
    }
    monkeypatch.setattr(mod.kit, "config_rule_verifier", lambda **_kwargs: ungrounded)
    result = mod.counterexample_grounding_loop(few_shot_examples=[], budget=1)

    assert result["counterexample_rounds"] == 1
    assert result["operator_result"]["residual"] == "still_missing"


def test_req_report_4467_adapter_and_gap_helpers_cover_failure_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-4467: helper branches report missing adapters and no-bank residuals honestly."""

    monkeypatch.setattr(mod.arc_game_adapters, "get_adapter", lambda _game: None)
    with pytest.raises(RuntimeError, match="dc22 adapter"):
        mod._adapter()

    assert mod._missing_gap(operator_result={"grounded": False, "residual": "x"}, reproduction_result={})[
        "residual_delta"
    ] == "x"
    assert mod._missing_gap(
        operator_result={"grounded": True},
        reproduction_result={"reproduced": True},
    )["residual_delta"] == "none"
    assert mod._missing_gap(
        operator_result={"grounded": True},
        reproduction_result={"reproduced": False},
    )["residual_delta"] == "dc22_reproduction_gate_failed"
    assert mod._verdict(
        precondition_miss=None,
        offline_reproduced=False,
        reproduced_levels=0,
        dc22_grounded=True,
    ) == "complete: dc22_cegis_grounded_no_bank_gap_logged"
    assert mod._verdict(
        precondition_miss=None,
        offline_reproduced=False,
        reproduced_levels=0,
        dc22_grounded=False,
    ) == "complete: dc22_cegis_not_grounded_gap_logged"


def test_req_report_4467_schema_rejects_partial_or_fabricated_success(tmp_path: Path) -> None:
    """REQ-REPORT-4467: schema guards reject partial prefixes, type drift, and fake success."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        few_shot_examples=_examples(),
        reproduce_fn=lambda _solution: {
            "game": "dc22",
            "claimed_level": 1,
            "reached_level": 1,
            "reproduced": True,
        },
        offline_solver_fn=lambda solution: {"solution": list(solution), "reached_level": 1},
        write_registry=False,
        write_gaps=False,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )
    bad = {
        **artifact,
        "honest_verdict": "partial: fake",
        "inference_substrate": None,
        "target_game": "bp35",
        "dc22_grounded": "true",
        "reproduced_levels": "1",
        "offline_reproduced": "true",
        "counterexample_rounds": "1",
        "baseline_pytest_nocov_green": "true",
        "few_shot_examples_used": {},
        "missing_verifier_gaps": {},
        "verifier_is_oracle": False,
        "reproducible_total_levels": "40",
        "random_seed": "4467",
        "reproducibility_checksum": "bad",
    }
    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "inference_substrate must not be None" in errors
    assert "target_game must be dc22" in errors
    assert "dc22_grounded must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "counterexample_rounds must be bare int" in errors
    assert "baseline_pytest_nocov_green must be bare bool" in errors
    assert "few_shot_examples_used must be list" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors

    unsupported = {**artifact, "inference_substrate": "unknown"}
    short_cached = {**artifact, "duration_s": 0.1}
    short_live = {**artifact, "inference_substrate": mod.LIVE_LLM_SUBSTRATE, "duration_s": 1.0}
    fake_success = {
        **artifact,
        "honest_verdict": "success: fake",
        "dc22_grounded": False,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "missing_verifier_gaps": [{"gap_id": mod.DC22_GAP_ID}],
        "no_3090_inference": False,
        "submitted_to_leaderboard": True,
        "field_principles": {},
    }
    no_principles = {**artifact, "field_principles": None}
    missing = dict(artifact)
    missing.pop("target_game")

    assert "inference_substrate has unsupported value" in mod.artifact_schema_errors(unsupported)
    assert "cached verifier substrate requires duration_s >= 1.0" in mod.artifact_schema_errors(short_cached)
    assert "live_llm_inference requires duration_s >= 60.0" in mod.artifact_schema_errors(short_live)
    fake_errors = mod.artifact_schema_errors(fake_success)
    assert "success verdict requires dc22_grounded true" in fake_errors
    assert "success verdict requires offline_reproduced true" in fake_errors
    assert "success verdict requires reproduced_levels >= 1" in fake_errors
    assert "success verdict requires no missing_verifier_gaps" in fake_errors
    assert "no_3090_inference must be true" in fake_errors
    assert "submitted_to_leaderboard must be false" in fake_errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4467" in fake_errors
    assert "field_principles must be dict" in mod.artifact_schema_errors(no_principles)
    assert "missing target_game" in mod.artifact_schema_errors(missing)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, {"honest_verdict": "partial: bad"})


def test_req_report_4467_ledger_helpers_cover_append_and_fallback_paths(tmp_path: Path) -> None:
    """REQ-REPORT-4467: registry and gap writers handle new rows and absent sentinels."""

    _write_fixture_repo(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        recommend_fn=lambda _game: _recommendation(),
        few_shot_examples=_examples(),
        reproduce_fn=lambda _solution: {
            "game": "dc22",
            "claimed_level": 1,
            "reached_level": 1,
            "reproduced": True,
        },
        offline_solver_fn=lambda solution: {"solution": list(solution), "reached_level": 1},
        write_registry=False,
        write_gaps=False,
        now=_clock(),
        sleep_fn=lambda _seconds: None,
    )

    entry = mod._banked_entry({}, artifact)
    assert entry["dead_ends"][0]["gap_id"] == mod.DC22_GAP_ID
    mod.update_arc_registry(tmp_path, {**artifact, "offline_reproduced": False})

    empty_root = tmp_path / "empty"
    (empty_root / "ops").mkdir(parents=True)
    mod.update_arc_registry(empty_root, artifact)
    empty_registry = yaml.safe_load((empty_root / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert empty_registry["games"][0]["game"] == "dc22"

    no_total_root = tmp_path / "no_totals"
    (no_total_root / "ops").mkdir(parents=True)
    (no_total_root / mod.REGISTRY_RELATIVE_PATH).write_text(
        "games:\n- game: dc22\n  reproducibility: unsolved\n  levels_reproduced: 0\n",
        encoding="utf-8",
    )
    mod.update_arc_registry(no_total_root, artifact)
    assert "reproducible_total_levels: 1" in (
        no_total_root / mod.REGISTRY_RELATIVE_PATH
    ).read_text(encoding="utf-8")

    gaps_root = tmp_path / "gaps"
    (gaps_root / "ops").mkdir(parents=True)
    (gaps_root / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text("existing", encoding="utf-8")
    mod.update_verifier_gaps(
        gaps_root,
        {
            **artifact,
            "offline_reproduced": False,
            "missing_verifier_gaps": [{"residual_delta": "still_missing"}],
        },
    )
    gaps_text = (gaps_root / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "existing\n" in gaps_text
    assert "movement: still_open" in gaps_text
