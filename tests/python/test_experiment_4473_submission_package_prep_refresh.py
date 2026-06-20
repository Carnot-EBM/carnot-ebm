"""Tests for Exp 4473 refreshed ARC operator submission package prep.

Spec refs: REQ-REPORT-4473, SCENARIO-REPORT-4473.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from carnot import experiment_4473_submission_package_prep_refresh as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_fixture_repo(root: Path) -> None:
    for game in ("alpha", "beta", "dc22", "sc25", "sb26"):
        (root / "environment_files" / game / "fixture").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "prior_submitted_baseline_levels": 13,
                "games": [
                    {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 34},
                    {"game": "sc25", "reproducibility": "reproduced", "levels_reproduced": 5},
                    {"game": "dc22", "reproducibility": "reproduced", "levels_reproduced": 1},
                    {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": 1},
                    {"game": "beta", "reproducibility": "reproduced", "levels_reproduced": 3},
                    {"game": "ignored", "reproducibility": "unsolved", "levels_reproduced": 9},
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
                {"game": "alpha", "claimed": 34, "live_level": 34, "env_match": True},
                {"game": "beta", "claimed": 3, "live_level": 1, "env_match": False},
            ],
        },
    )
    _write_json(
        root / mod.PRIOR_PACKAGE_412_RELATIVE_PATH,
        {"total_reproduced_levels_in_package": mod.PRIOR_PACKAGE_412_LEVELS},
    )


def _fake_reproduce(entry: Mapping[str, Any], _root: Path) -> dict[str, Any]:
    game = str(entry["game"])
    claimed = int(entry["levels_reproduced"])
    if game == "beta":
        return {
            "game": game,
            "claimed_level": claimed,
            "reached_level": 1,
            "reproduced": False,
            "source": "fixtures/beta.json",
            "action_sequence": ["b1"],
            "action_count": 1,
        }
    return {
        "game": game,
        "claimed_level": claimed,
        "reached_level": claimed,
        "reproduced": True,
        "source": f"fixtures/{game}.json",
        "action_sequence": [f"{game}-a1", f"{game}-a2"],
        "action_count": 2,
    }


def test_req_report_4473_spec_declares_refresh_contract() -> None:
    """REQ-REPORT-4473: OpenSpec names the refresh artifact and growth field."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4473" in spec
    assert "SCENARIO-REPORT-4473" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "grew_vs_412" in spec
    assert "scripts/arc3_live_submit.py" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4473_refresh_counts_only_replayed_growth_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4473: .413 replay rows must validate before package count growth."""

    _write_fixture_repo(tmp_path)
    clock = {"t": 100.0}

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

    assert artifact["honest_verdict"] == "success: submission_package_ready_41_levels_beats_13_grew_vs_412_quarantined_1"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 1.0
    assert artifact["submission_package_ready"] is True
    assert artifact["total_reproduced_levels_in_package"] == 41
    assert artifact["prior_package_412_levels"] == 39
    assert artifact["grew_vs_412"] is True
    assert artifact["beats_prior_baseline"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["verifier_is_oracle"] is True

    validations = {row["game"]: row for row in artifact["per_game_replay_validation"]}
    assert validations["alpha"]["env_match_basis"] == "prior_live_submission_confirmed"
    assert validations["sc25"]["replays_ok"] is True
    assert validations["dc22"]["env_matched"] is True
    assert validations["sb26"]["reproduced_levels"] == 1
    assert validations["beta"]["quarantined"] is True
    assert [row["game"] for row in artifact["package_manifest"]] == ["alpha", "sc25", "dc22", "sb26"]
    assert artifact["quarantined_games"] == ["beta"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert mod.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["grew_vs_412"] is True
    note = (tmp_path / mod.OPERATOR_NOTE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "Exp 4473" in note
    assert "Submitted by this task: `False`" in note


def test_req_report_4473_blocks_precondition_miss_without_replay(tmp_path: Path) -> None:
    """REQ-REPORT-4473: missing offline env files stop as blocked artifacts."""

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
    assert artifact["grew_vs_412"] is False
    assert artifact["beats_prior_baseline"] is False
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4473_schema_rejects_inconsistent_or_submitted_artifacts() -> None:
    """REQ-REPORT-4473: required fields are bare and no leaderboard submission is allowed."""

    artifact = {
        "honest_verdict": "maybe",
        "inference_substrate": "",
        "submission_package_ready": "true",
        "total_reproduced_levels_in_package": "41",
        "grew_vs_412": "yes",
        "prior_submitted_baseline_levels": 13,
        "beats_prior_baseline": True,
        "per_game_replay_validation": {},
        "submitted_to_leaderboard": True,
        "verifier_is_oracle": True,
        "random_seed": 4473,
        "reproducibility_checksum": "x",
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate must equal refresh substrate" in errors
    assert "submission_package_ready must be bare bool" in errors
    assert "total_reproduced_levels_in_package must be bare int" in errors
    assert "grew_vs_412 must be bare bool" in errors
    assert "per_game_replay_validation must be list" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "reproducibility_checksum must be sha256 hex" in errors
    assert "missing honest_verdict" in mod.artifact_schema_errors({})
    assert mod._honest_verdict(
        ready=False,
        total=12,
        baseline=13,
        grew=False,
        quarantined_count=2,
    ) == "complete: submission_package_not_ready_12_levels_vs_13_quarantined_2"

    ready_bad = {
        "honest_verdict": "success: bad_ready",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "submission_package_ready": True,
        "total_reproduced_levels_in_package": 41,
        "prior_package_412_levels": 39,
        "grew_vs_412": True,
        "prior_submitted_baseline_levels": "13",
        "beats_prior_baseline": "no",
        "per_game_replay_validation": [],
        "package_manifest": [{"game": "alpha", "levels": 41}],
        "submitted_to_leaderboard": True,
        "verifier_is_oracle": False,
        "random_seed": "4473",
        "reproducibility_checksum": "0" * 64,
    }
    ready_errors = mod.artifact_schema_errors(ready_bad)
    assert "prior_submitted_baseline_levels must be bare int" in ready_errors
    assert "beats_prior_baseline must be bare bool" in ready_errors
    assert "verifier_is_oracle must be true" in ready_errors
    assert "random_seed must be bare int" in ready_errors
    assert "ready package must beat prior baseline" in ready_errors
    assert "ready package must not submit" in ready_errors
    try:
        mod.write_artifact(Path("/tmp"), artifact)
    except ValueError as exc:
        assert "submitted_to_leaderboard must be false" in str(exc)
    else:  # pragma: no cover - write_artifact must reject invalid artifacts
        raise AssertionError("invalid artifact unexpectedly wrote")


def test_req_report_4473_schema_checks_growth_consistency() -> None:
    """REQ-REPORT-4473: grew_vs_412 is derived from the refreshed package total."""

    artifact = {
        "honest_verdict": "success: bad_growth",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "submission_package_ready": True,
        "total_reproduced_levels_in_package": 39,
        "grew_vs_412": True,
        "prior_submitted_baseline_levels": 13,
        "beats_prior_baseline": True,
        "per_game_replay_validation": [],
        "submitted_to_leaderboard": False,
        "verifier_is_oracle": True,
        "random_seed": 4473,
        "reproducibility_checksum": "0" * 64,
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "grew_vs_412 inconsistent with total and .412 baseline" in errors
    assert "ready package must include package_manifest rows" in errors
