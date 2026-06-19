"""Tests for Exp 4443 g50t example-conditioned win banking.

Spec refs: REQ-REPORT-4443, SCENARIO-REPORT-4443.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from carnot import experiment_4443_bank_g50t_example_conditioned_win as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_fixture_repo(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "environment_files/g50t/fixture").mkdir(parents=True, exist_ok=True)
    (root / "environment_files/g50t/fixture/frame.json").write_text("{}", encoding="utf-8")
    registry = {
        "schema_version": 1,
        "updated": "2026-06-19",
        "games": [
            {
                "game": "s5i5",
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
                "win_condition": "marker coverage",
            },
            {
                "game": "ft09",
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
                "win_condition": "local color cycle",
            },
            {
                "game": "g50t",
                "reproducibility": "unsolved",
                "levels_reproduced": 0,
                "solver": "scripts/arc_loop_solve.py --game g50t",
                "gotchas": [],
                "dead_ends": [
                    {
                        "gap_id": "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT",
                        "status": "open",
                        "game": "g50t",
                    }
                ],
            },
        ],
        "reproducible_total_levels": 37,
        "reproducible_total_games": 18,
    }
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )


def _examples() -> list[dict[str, str]]:
    return [
        {
            "game": "ka59",
            "source": "results/experiment_4414_config_rule_induction_solve.json",
            "rule_id": "editable_count_4_equals_reference_count_4_32",
            "predicate": "count_4 == 32",
        },
        {
            "game": "s5i5",
            "source": mod.REGISTRY_RELATIVE_PATH,
            "rule_id": "marker_coverage",
            "predicate": "controlled markers cover target markers",
        },
        {
            "game": "ft09",
            "source": mod.REGISTRY_RELATIVE_PATH,
            "rule_id": "local_color_cycle_constraint",
            "predicate": "local color-cycle constraint is satisfied",
        },
    ]


def _digest() -> dict[str, Any]:
    return {
        "game": "g50t",
        "components": {
            "player": {"x": 13, "y": 7, "width": 7, "height": 7},
            "target": {"x": 42, "y": 48, "width": 9, "height": 9},
            "goal_top_left": {"x": 43, "y": 49},
            "triggers": [{"x": 37, "y": 7, "width": 7, "height": 7}],
        },
        "available_actions": [1, 2, 3, 4, 5],
        "value_counts": {"0": 3006, "1": 9, "5": 880, "8": 82, "9": 119},
    }


def _ok_preconditions() -> dict[str, Any]:
    return {
        "offline_env_files_present": True,
        "target_env_present": True,
        "arc_solver_kit_importable": True,
        "qwen_gguf_cached": True,
        "igpu_llama_server_available": False,
        "generator_resource_available": True,
        "grounded_few_shot_examples": 3,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _reproduce(solution: Sequence[str]) -> dict[str, Any]:
    assert list(solution) == mod.G50T_L1_SOLUTION
    return {
        "game": "g50t",
        "reached_level": 1,
        "claimed_level": 1,
        "reproduced": True,
        "mode": "offline_reproduction_gate_no_quota",
    }


def _solve(_digest: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    return list(mod.G50T_L1_SOLUTION), {
        "driver": "OfflineSolver",
        "win_check": "player.x == target.x + 1 and player.y == target.y + 1",
        "states_expanded": 17,
    }


def test_req_report_4443_spec_declares_corrected_banking_contract() -> None:
    """REQ-REPORT-4443: OpenSpec declares the corrected substrate and banking fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4443" in spec
    assert "SCENARIO-REPORT-4443" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.INFERENCE_SUBSTRATE in spec
    assert "duration_s>=1.0" in spec
    assert "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4443_banks_cached_predicate_and_updates_registry(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4443: a cached grounded g50t solve banks only after reproduce()."""

    _write_fixture_repo(tmp_path)
    clock = {"t": 10.0}
    slept: list[float] = []

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["t"] += seconds

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_ok_preconditions(),
        few_shot_examples=_examples(),
        digest=_digest(),
        solve_fn=_solve,
        reproduce_fn=_reproduce,
        now=now,
        sleep_fn=sleep,
    )

    assert slept and slept[0] >= mod.VERIFIER_SCORING_MIN_DURATION_S
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= mod.VERIFIER_SCORING_MIN_DURATION_S
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["reproducible_total_levels"] == 38
    assert [row["game"] for row in artifact["few_shot_examples_used"]] == ["ka59", "s5i5", "ft09"]
    assert artifact["solver"]["solution"] == mod.G50T_L1_SOLUTION
    assert artifact["grounded_win_condition"]["grounded"] is True
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    g50t = next(row for row in registry["games"] if row["game"] == "g50t")
    assert g50t["reproducibility"] == "reproduced"
    assert g50t["levels_reproduced"] == 1
    assert "player.x == target.x + 1" in g50t["win_condition"]
    assert g50t["dead_ends"][0]["status"] == "filled"
    assert registry["reproducible_total_levels"] == 38
    assert registry["reproducible_total_games"] == 19


def test_req_report_4443_schema_rejects_missing_substrate_and_short_cached_duration() -> None:
    """REQ-REPORT-4443: the .410 missing-substrate failure is a schema error."""

    artifact = mod.build_artifact(
        root=Path("."),
        preconditions=_ok_preconditions(),
        few_shot_examples=_examples(),
        digest=_digest(),
        prompt="prompt",
        qwen_generation={"grounded": True},
        grounded_win_condition={"grounded": True},
        solution=mod.G50T_L1_SOLUTION,
        solver_metadata={"driver": "OfflineSolver"},
        reproduction_result={"reproduced": True, "reached_level": 1, "claimed_level": 1},
        registry_totals={"reproducible_total_levels": 38, "reproducible_total_games": 19},
        started_at=1.0,
        ended_at=1.5,
    )

    missing = {**artifact, "inference_substrate": None}
    short = {**artifact, "duration_s": 0.5}

    assert "missing inference_substrate" in mod.artifact_schema_errors(missing)
    assert "cached verifier substrate requires duration_s >= 1.0" in mod.artifact_schema_errors(short)


def test_req_report_4443_blocked_resource_does_not_reproduce_or_bank(tmp_path: Path) -> None:
    """REQ-REPORT-4443: missing generator resources stop before reproduction and registry edits."""

    _write_fixture_repo(tmp_path)
    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={
            **_ok_preconditions(),
            "qwen_gguf_cached": False,
            "generator_resource_available": False,
            "ok": False,
        },
        few_shot_examples=_examples(),
        digest=_digest(),
        solve_fn=lambda _digest: calls.append("solve") or ([], {}),
        reproduce_fn=lambda _solution: calls.append("reproduce") or {},
        now=lambda: 3.0,
        sleep_fn=lambda _seconds: None,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_qwen_generator_resource"
    assert artifact["inference_substrate"] == mod.BLOCKED_INFERENCE_SUBSTRATE
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0

    registry = yaml.safe_load((tmp_path / mod.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    g50t = next(row for row in registry["games"] if row["game"] == "g50t")
    assert g50t["reproducibility"] == "unsolved"
    assert registry["reproducible_total_levels"] == 37
