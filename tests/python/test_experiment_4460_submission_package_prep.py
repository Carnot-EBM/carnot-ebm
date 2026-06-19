"""Tests for Exp 4460 ARC operator submission package prep.

Spec refs: REQ-REPORT-4460, SCENARIO-REPORT-4460.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from carnot import experiment_4460_submission_package_prep as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixture_repo(root: Path) -> None:
    for game in ("alpha", "beta", "gamma"):
        (root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "prior_submitted_baseline_levels": 13,
                "games": [
                    {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 9},
                    {"game": "beta", "reproducibility": "reproduced", "levels_reproduced": 3},
                    {"game": "gamma", "reproducibility": "reproduced", "levels_reproduced": 6},
                    {"game": "delta", "reproducibility": "unsolved", "levels_reproduced": 2},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(
        root / mod.PRIOR_SUBMISSION_RELATIVE_PATH,
        {
            "live_total_levels": 13,
            "leaderboard_submitted": True,
            "per_game": [
                {"game": "alpha", "claimed": 9, "live_level": 9, "env_match": True},
                {"game": "beta", "claimed": 3, "live_level": 0, "env_match": False},
            ],
        },
    )


def _fake_reproduce(entry: Mapping[str, Any], _root: Path) -> dict[str, Any]:
    game = str(entry["game"])
    if game == "alpha":
        return {
            "game": game,
            "claimed_level": 9,
            "reached_level": 9,
            "reproduced": True,
            "source": "fixtures/alpha.json",
            "action_sequence": ["a1", "a2"],
            "action_count": 2,
        }
    if game == "beta":
        return {
            "game": game,
            "claimed_level": 3,
            "reached_level": 1,
            "reproduced": False,
            "source": "fixtures/beta.json",
            "action_sequence": ["b1"],
            "action_count": 1,
        }
    if game == "gamma":
        return {
            "game": game,
            "claimed_level": 6,
            "reached_level": 6,
            "reproduced": True,
            "source": "fixtures/gamma.json",
            "action_sequence": ["g1", "g2", "g3"],
            "action_count": 3,
        }
    raise AssertionError(f"unexpected reproduced entry {game}")


def test_req_report_4460_spec_declares_submission_package_contract() -> None:
    """REQ-REPORT-4460: OpenSpec names the package artifact and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4460" in spec
    assert "SCENARIO-REPORT-4460" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "scripts/arc3_live_submit.py" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4460_quarantines_failed_replay_and_counts_ready_package(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4460: only revalidated offline replays enter the package count."""

    _write_fixture_repo(tmp_path)
    clock = {"t": 10.0}

    def now() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={
            "offline_env_files_present": True,
            "arc_solver_kit_import": True,
            "ok": True,
        },
        reproduce_entry_fn=_fake_reproduce,
        now=now,
        sleep_fn=sleep,
    )

    assert artifact["honest_verdict"] == "success: submission_package_ready_15_levels_beats_13_quarantined_1"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["submission_package_ready"] is True
    assert artifact["total_reproduced_levels_in_package"] == 15
    assert artifact["prior_submitted_baseline_levels"] == 13
    assert artifact["beats_prior_baseline"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["verifier_is_oracle"] is True

    validations = {row["game"]: row for row in artifact["per_game_replay_validation"]}
    assert validations["alpha"]["replays_ok"] is True
    assert validations["alpha"]["env_match_basis"] == "prior_live_submission_confirmed"
    assert validations["beta"]["replays_ok"] is False
    assert validations["beta"]["quarantined"] is True
    assert validations["gamma"]["env_matched"] is True
    assert validations["gamma"]["env_match_basis"] == "offline_env_file_present"
    assert [row["game"] for row in artifact["package_manifest"]] == ["alpha", "gamma"]
    assert artifact["quarantined_games"] == ["beta"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["total_reproduced_levels_in_package"] == 15


def test_req_report_4460_blocks_without_offline_env_files(tmp_path: Path) -> None:
    """REQ-REPORT-4460: precondition misses stop as blocked artifacts without replay calls."""

    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump({"prior_submitted_baseline_levels": 13, "games": []}),
        encoding="utf-8",
    )
    called = {"reproduce": False}

    def reproduce(_entry: Mapping[str, Any], _root: Path) -> dict[str, Any]:
        called["reproduce"] = True
        return {}

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked={
            "offline_env_files_present": False,
            "arc_solver_kit_import": True,
            "ok": False,
        },
        reproduce_entry_fn=reproduce,
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )

    assert called["reproduce"] is False
    assert artifact["honest_verdict"] == "complete: blocked_offline_env_files"
    assert artifact["submission_package_ready"] is False
    assert artifact["total_reproduced_levels_in_package"] == 0
    assert artifact["beats_prior_baseline"] is False
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4460_schema_rejects_non_bare_or_submitted_artifacts() -> None:
    """REQ-REPORT-4460: required fields are bare and the package never submits."""

    artifact = {
        "honest_verdict": "maybe",
        "inference_substrate": "",
        "submission_package_ready": "true",
        "total_reproduced_levels_in_package": "15",
        "prior_submitted_baseline_levels": 13,
        "beats_prior_baseline": True,
        "per_game_replay_validation": {},
        "submitted_to_leaderboard": True,
        "verifier_is_oracle": True,
        "random_seed": 4460,
        "reproducibility_checksum": "x",
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate must be non-empty string" in errors
    assert "submission_package_ready must be bare bool" in errors
    assert "total_reproduced_levels_in_package must be bare int" in errors
    assert "per_game_replay_validation must be list" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "reproducibility_checksum must be sha256 hex" in errors
    try:
        mod.write_artifact(Path("/tmp"), artifact)
    except ValueError as exc:
        assert "submitted_to_leaderboard must be false" in str(exc)
    else:  # pragma: no cover - write_artifact must reject invalid artifacts
        raise AssertionError("invalid artifact unexpectedly wrote")


def test_req_report_4460_defensive_helpers_and_note_writer(tmp_path: Path) -> None:
    """REQ-REPORT-4460: defensive parse branches stay honest and schema-checked."""

    assert mod._as_int("bad") == 0
    assert mod._load_json(tmp_path / "missing.json") == {}
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert mod._load_json(tmp_path / "bad.json") == {}
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    assert mod._load_json(tmp_path / "list.json") == {}
    assert mod.load_registry(tmp_path) == {"games": []}
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text("[", encoding="utf-8")
    assert mod.load_registry(tmp_path) == {"games": []}

    assert mod.first_precondition_miss({"offline_env_files_present": True, "arc_solver_kit_import": False})
    assert mod._reproduced_registry_entries({"games": {}}) == []
    assert mod._reproduced_registry_entries(
        {
            "games": [
                "bad-row",
                {"game": "zero", "reproducibility": "reproduced", "levels_reproduced": 0},
                {"game": "ok", "reproducibility": "reproduced", "levels_reproduced": "1"},
            ]
        }
    ) == [{"game": "ok", "reproducibility": "reproduced", "levels_reproduced": "1"}]

    _write_json(tmp_path / mod.PRIOR_SUBMISSION_RELATIVE_PATH, {"per_game": "bad"})
    assert mod._prior_env_match_map(tmp_path) == {}
    _write_json(
        tmp_path / mod.PRIOR_SUBMISSION_RELATIVE_PATH,
        {"per_game": ["bad", {}, {"game": "miss", "claimed": 2, "live_level": 1, "env_match": True}]},
    )
    assert mod._prior_env_match_map(tmp_path) == {"miss": False}
    assert mod._env_match_status(tmp_path, "miss", {"miss": False}) == (
        False,
        "prior_live_submission_mismatch",
    )
    assert mod._env_match_status(tmp_path, "absent", {}) == (False, "missing_offline_env_file")

    _write_json(tmp_path / "scorecard.json", {"rows": [{"game": "ok", "plan": [1, "2"]}]})
    assert mod._scorecard_plan(tmp_path, "scorecard.json", "rows", "ok") == ["1", "2"]
    assert mod._scorecard_plan(tmp_path, "scorecard.json", "rows", "missing") == []

    row = mod._validation_row(
        {"game": "fallback", "levels_reproduced": 1},
        {"reached_level": 1, "reproduced": True, "action_sequence": "not-list"},
        env_matched=False,
        env_match_basis="missing_offline_env_file",
    )
    assert row["action_sequence"] == []
    assert row["quarantined"] is True
    assert mod._honest_verdict(ready=False, total=1, baseline=13, quarantined_count=2).startswith("complete:")

    ready_bad = {
        "honest_verdict": "success: bad_ready",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "submission_package_ready": True,
        "total_reproduced_levels_in_package": 15,
        "prior_submitted_baseline_levels": "13",
        "beats_prior_baseline": "yes",
        "per_game_replay_validation": [],
        "submitted_to_leaderboard": True,
        "verifier_is_oracle": False,
        "random_seed": "4460",
        "reproducibility_checksum": "0" * 64,
    }
    ready_errors = mod.artifact_schema_errors(ready_bad)
    assert "prior_submitted_baseline_levels must be bare int" in ready_errors
    assert "beats_prior_baseline must be bare bool" in ready_errors
    assert "verifier_is_oracle must be true" in ready_errors
    assert "random_seed must be bare int" in ready_errors
    assert "ready package must beat prior baseline" in ready_errors
    assert "ready package must not submit" in ready_errors

    missing_errors = mod.artifact_schema_errors({})
    assert "missing honest_verdict" in missing_errors
    note_path = mod.write_operator_note(
        tmp_path,
        {
            "submission_package_ready": False,
            "total_reproduced_levels_in_package": 0,
            "prior_submitted_baseline_levels": 13,
            "submitted_to_leaderboard": False,
            "operator_checklist": ["operator-only"],
            "package_manifest": "bad",
        },
    )
    assert note_path.read_text(encoding="utf-8").startswith("# ARC-AGI-3")
