"""Tests for Exp 4426 ARC registry reproducibility audit.

Spec refs: REQ-REPORT-4426, SCENARIO-REPORT-4426.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from carnot import experiment_4426_arc_registry_repro_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_fixture_repo(root: Path) -> None:
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reproducible_total_levels": 99,
                "reproducible_total_games": 2,
                "games": [
                    "not-a-row",
                    {
                        "game": "alpha",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 2,
                    },
                    {
                        "game": "beta",
                        "reproducibility": "reproduced",
                        "levels_reproduced": 3,
                    },
                    {
                        "game": "gamma",
                        "reproducibility": "unsolved",
                        "levels_reproduced": 0,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "results" / "experiment_4421_config_rule_solve_unseen.json",
        {
            "experiment": "experiment_4421_config_rule_solve_unseen",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "new_levels_reproduced": 1,
            "reproduction_result": {"reproduced": True, "reached_level": 1},
            "flagged_adversarial": True,
        },
    )
    _write_json(
        root / "results" / "experiment_4422_glyph_rewrite_perception.json",
        {
            "experiment": "experiment_4422_glyph_rewrite_perception",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reproduction_result": {"reproduced": True, "reached_level": 1},
        },
    )
    _write_json(
        root / "results" / "experiment_4423_generic_first_contact_breadth.json",
        {
            "experiment": "experiment_4423_generic_first_contact_breadth",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "new_levels_reproduced": 0,
            "standing_loop_result": {
                "offline_reproduced": False,
                "reproduced_levels": 0,
                "mode": "standing_arc_loop_routing_only",
            },
        },
    )
    _write_json(
        root / "results" / "experiment_4424_deeper_solved_game.json",
        {
            "experiment": "experiment_4424_deeper_solved_game",
            "offline_reproduced": False,
            "reproduced_levels": 1,
            "new_levels_reproduced": 0,
            "reproduce_result": {"reproduced": False, "reached_level": 1, "claimed_level": 2},
        },
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_reproduce(entry: Mapping[str, Any], _root: Path) -> dict[str, Any]:
    game = str(entry["game"])
    if game == "alpha":
        return {"game": game, "claimed_level": 2, "reached_level": 2, "reproduced": True}
    if game == "beta":
        return {"game": game, "claimed_level": 3, "reached_level": 1, "reproduced": False}
    raise AssertionError(f"unexpected reproduced entry {game}")


def test_req_report_4426_spec_declares_required_audit_contract() -> None:
    """REQ-REPORT-4426: OpenSpec names the audit artifact and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4426" in spec
    assert "SCENARIO-REPORT-4426" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "scripts/arc3_replay_scorecard_metaharness.py" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4426_audits_registry_total_and_flags_downgrade(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4426: total comes from reproduction, not registry assertion."""

    _write_fixture_repo(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        reproduce_entry_fn=_fake_reproduce,
        metaharness_runner=lambda _root: {
            "returncode": 0,
            "artifact_path": "results/arc3_replay_aggregate_scorecard.json",
            "total_levels": 32,
            "games": 16,
        },
        now=lambda: 10.0,
    )

    assert artifact["reproducible_total_levels"] == 3
    assert artifact["registry_claimed_reproducible_total_levels"] == 99
    assert artifact["honest_verdict"] == "complete: registry_repro_audit_flagged_1_counted_entries"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["metaharness"]["total_levels"] == 32
    assert artifact["counted_entries_audited"] == 2
    assert artifact["entries_downgraded_to_provisional"] == ["beta"]
    by_game = {row["game"]: row for row in artifact["registry_entry_audits"]}
    assert by_game["alpha"]["effective_levels_reproduced"] == 2
    assert by_game["beta"]["downgraded_to_provisional"] is True
    assert by_game["beta"]["effective_reproducibility"] == "provisional"
    assert by_game["gamma"]["counted_by_registry"] is False
    gates = {row["experiment"]: row for row in artifact["milestone_409_reproduction_gates"]}
    assert gates["exp4421"]["reproduction_gated"] is True
    assert gates["exp4421"]["artifact_flagged_adversarial"] is True
    assert gates["exp4421"]["new_levels_counted"] == 1
    assert gates["exp4423"]["reproduction_gated"] is True
    assert gates["exp4423"]["new_levels_counted"] == 0
    assert gates["exp4424"]["reproduction_gated"] is True
    assert gates["exp4424"]["new_levels_counted"] == 0
    assert mod.artifact_schema_errors(artifact) == []
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducible_total_levels"] == 3


def test_req_report_4426_schema_rejects_fabricated_terminal_artifact() -> None:
    """REQ-REPORT-4426: required artifact fields are bare and terminal-prefixed."""

    assert mod._load_json(Path("/definitely/missing/artifact.json")) == {}
    assert mod.load_registry(Path("/definitely/missing/repo")) == {"games": []}
    assert mod._as_int("bad") == 0
    assert mod._normalize_action({"action": 4, "data": None}) == (4, None)
    assert mod._normalize_action({"action": 3}) == (3, None)
    assert mod._normalize_action({"x": 9, "y": 10}) == (6, {"x": 9, "y": 10})
    assert mod._honest_verdict(3, 99, []) == "complete: registry_repro_audit_total_99_asserted_3_audited"
    assert mod._honest_verdict(3, 3, []) == "success: registry_reproducible_total_levels_3_audited"
    assert mod.milestone_409_gate_rows(Path("/definitely/missing/repo"))[0]["reproduction_gated"] is False

    artifact = {
        "reproducible_total_levels": "3",
        "honest_verdict": "maybe",
        "inference_substrate": "",
        "registry_entry_audits": {},
        "milestone_409_reproduction_gates": {},
        "metaharness": "bad",
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "reproducible_total_levels must be bare int" in errors
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate must be non-empty string" in errors
    assert "registry_entry_audits must be list" in errors
    assert "milestone_409_reproduction_gates must be list" in errors
    assert "metaharness must be dict" in errors

    missing_errors = mod.artifact_schema_errors({})
    assert "missing reproducible_total_levels" in missing_errors
    try:
        mod.write_artifact(Path("/tmp"), artifact)
    except ValueError as exc:
        assert "reproducible_total_levels must be bare int" in str(exc)
    else:  # pragma: no cover - write_artifact must reject invalid artifacts
        raise AssertionError("invalid artifact unexpectedly wrote")
