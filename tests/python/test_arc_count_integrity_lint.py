"""Tests for ARC reproduced-count integrity linting.

Spec refs: REQ-REPORT-4462, SCENARIO-REPORT-4462, SCENARIO-REPORT-4462-SUBMISSION.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts import arc_count_integrity_lint as lint


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
PRE_COMMIT_PATH = REPO / ".pre-commit-config.yaml"


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _registry_payload(*, total: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reproducible_total_levels": total,
        "provisional_total_levels": 5,
        "games": [
            {
                "game": "sc25",
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
                "levels_live_recorded": 5,
            },
            {
                "game": "alpha",
                "reproducibility": "reproduced",
                "levels_reproduced": 2,
            },
        ],
    }


def test_req_report_4462_spec_declares_count_integrity_contract() -> None:
    """REQ-REPORT-4462: OpenSpec names the guard and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4462" in spec
    assert "SCENARIO-REPORT-4462" in spec
    assert "SCENARIO-REPORT-4462-SUBMISSION" in spec
    assert "experiment_4462_provisional_reproduced_count_integrity_lint.json" in spec
    for field in (
        "honest_verdict",
        "guard_shipped",
        "catches_provisional_inflation",
        "tests_pass",
        "inference_substrate",
    ):
        assert field in spec


def test_scenario_report_4462_flags_provisional_live_recorded_inflation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4462: live-recorded provisional levels never count as reproduced."""

    registry = _write_yaml(tmp_path / "ops" / "arc_solve_registry.yaml", _registry_payload(total=7))

    issues = lint.lint_registry_path(registry, replay_entry_fn=lambda _entry, _root: None)
    kinds = {issue.kind for issue in issues}

    assert "PROVISIONAL_INFLATION" in kinds
    assert any("levels_reproduced" in issue.detail for issue in issues)
    assert any("levels_live_recorded" in issue.detail for issue in issues)


def test_req_report_4462_flags_plain_registry_total_mismatch(tmp_path: Path) -> None:
    """REQ-REPORT-4462: non-provisional total drift is still a count error."""

    registry = _write_yaml(
        tmp_path / "ops" / "arc_solve_registry.yaml",
        {
            "schema_version": 1,
            "reproducible_total_levels": 9,
            "games": [
                {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 2}
            ],
        },
    )

    issues = lint.lint_registry_path(registry, replay_entry_fn=lambda _entry, _root: None)

    assert [issue.kind for issue in issues] == ["REGISTRY_TOTAL_MISMATCH"]
    assert issues[0].to_dict()["severity"] == "error"


def test_scenario_report_4462_accepts_reproduced_sum_without_counting_live_recorded(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4462: reproduced totals equal sum(levels_reproduced), not live rows."""

    registry = _write_yaml(tmp_path / "ops" / "arc_solve_registry.yaml", _registry_payload(total=3))

    issues = lint.lint_registry_path(registry, replay_entry_fn=lambda _entry, _root: None)

    assert issues == []


def test_req_report_4462_flags_registry_replay_overclaim(tmp_path: Path) -> None:
    """REQ-REPORT-4462: sampled registry rows cannot claim beyond reproduce()."""

    registry = _write_yaml(tmp_path / "ops" / "arc_solve_registry.yaml", _registry_payload(total=3))

    def replay(entry: Mapping[str, Any], _root: Path) -> Mapping[str, Any]:
        return {
            "game": entry["game"],
            "reached_level": 1,
            "claimed_level": entry["levels_reproduced"],
            "reproduced": entry["game"] == "sc25",
        }

    issues = lint.lint_registry_path(registry, replay_entry_fn=replay)

    assert [issue.kind for issue in issues] == ["REGISTRY_REPLAY_OVERCLAIM"]
    assert issues[0].game == "alpha"
    assert "claimed 2" in issues[0].detail
    assert issues[0].to_dict()["game"] == "alpha"


def test_req_report_4462_registry_uses_default_replay_and_sampling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-REPORT-4462: default registry replay path is injectable and sampled."""

    registry_payload = {
        "schema_version": 1,
        "reproducible_total_levels": 6,
        "games": [
            {"game": "zeta", "reproducibility": "reproduced", "levels_reproduced": 1},
            {"game": "sc25", "reproducibility": "reproduced", "levels_reproduced": 1, "levels_live_recorded": 5},
            {"game": "alpha", "reproducibility": "reproduced", "levels_reproduced": 4},
        ],
    }
    registry = _write_yaml(tmp_path / "ops" / "arc_solve_registry.yaml", registry_payload)
    called: list[str] = []

    def default_replay(entry: Mapping[str, Any], _root: Path) -> Mapping[str, Any]:
        called.append(str(entry["game"]))
        return {"reached_level": entry["levels_reproduced"], "reproduced": True}

    monkeypatch.setattr(lint, "_default_registry_replay", default_replay)

    issues = lint.lint_registry_path(registry, max_replay_games=2)

    assert issues == []
    assert called == ["sc25", "alpha"]
    assert [row["game"] for row in lint._registry_replay_sample(registry_payload["games"], None)] == [
        "sc25",
        "alpha",
        "zeta",
    ]


def test_scenario_report_4462_submission_rejects_non_replaying_package_counts(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-4462-SUBMISSION: package totals require replay-valid rows."""

    package = _write_json(
        tmp_path / "results" / "experiment_4460_submission_package_prep.json",
        {
            "submitted_to_leaderboard": False,
            "total_reproduced_levels_in_package": 6,
            "per_game_replay_validation": [
                {
                    "game": "alpha",
                    "replays_ok": True,
                    "env_matched": True,
                    "reproduced_levels": 2,
                    "action_sequence": ["a1", "a2"],
                    "reproduction_result": {"reached_level": 2, "reproduced": True},
                },
                {
                    "game": "beta",
                    "replays_ok": True,
                    "env_matched": True,
                    "reproduced_levels": 3,
                    "action_sequence": ["b1"],
                    "reproduction_result": {"reached_level": 3, "reproduced": True},
                },
                {
                    "game": "gamma",
                    "replays_ok": False,
                    "env_matched": True,
                    "reproduced_levels": 1,
                    "action_sequence": ["g1"],
                    "reproduction_result": {"reached_level": 1, "reproduced": True},
                },
            ],
        },
    )

    def replay(row: Mapping[str, Any], _root: Path) -> Mapping[str, Any]:
        reached = {"alpha": 2, "beta": 1, "gamma": 1}[str(row["game"])]
        return {
            "game": row["game"],
            "reached_level": reached,
            "claimed_level": row["reproduced_levels"],
            "reproduced": reached >= int(row["reproduced_levels"]),
        }

    issues = lint.lint_submission_package_path(package, replay_row_fn=replay)
    kinds = [issue.kind for issue in issues]

    assert "SUBMISSION_REPLAY_OVERCLAIM" in kinds
    assert "SUBMISSION_ROW_NOT_COUNTABLE" in kinds
    assert "SUBMISSION_TOTAL_MISMATCH" in kinds
    assert any(issue.game == "beta" for issue in issues)
    assert any(issue.game == "gamma" for issue in issues)


def test_req_report_4462_submission_defensive_branches_assert(tmp_path: Path) -> None:
    """REQ-REPORT-4462: malformed package rows fail closed with assertions."""

    package_path = tmp_path / "results" / "experiment_4460_submission_package_prep.json"
    not_list = {
        "submitted_to_leaderboard": True,
        "total_reproduced_levels_in_package": 1,
        "per_game_replay_validation": {},
    }
    issues = lint.lint_submission_package_payload(package_path, not_list)
    assert [issue.kind for issue in issues] == [
        "SUBMISSION_SUBMITTED_TO_LEADERBOARD",
        "SUBMISSION_ROWS_NOT_LIST",
    ]

    payload = {
        "submitted_to_leaderboard": False,
        "total_reproduced_levels_in_package": 4,
        "per_game_replay_validation": [
            "bad-row",
            {"game": "zero", "reproduced_levels": 0},
            {
                "game": "env",
                "replays_ok": True,
                "env_matched": False,
                "reproduced_levels": 1,
                "action_sequence": ["x"],
                "reproduction_result": {"reached_level": 1, "reproduced": True},
            },
            {
                "game": "empty",
                "replays_ok": True,
                "env_matched": True,
                "reproduced_levels": 1,
                "action_sequence": [],
                "reproduction_result": {"reached_level": 1, "reproduced": True},
            },
            {
                "game": "missing-result",
                "replays_ok": True,
                "env_matched": True,
                "reproduced_levels": 1,
                "action_sequence": ["x"],
            },
            {
                "game": "embedded-low",
                "replays_ok": True,
                "env_matched": True,
                "reproduced_levels": 1,
                "action_sequence": ["x"],
                "reproduction_result": {"reached_level": 0, "reproduced": False},
            },
            {
                "game": "mismatch",
                "replays_ok": True,
                "env_matched": True,
                "reproduced_levels": 1,
                "action_sequence": ["x"],
                "reproduction_result": {"reached_level": 1, "reproduced": True},
            },
        ],
    }

    def replay(row: Mapping[str, Any], _root: Path) -> Mapping[str, Any]:
        assert row["game"] == "mismatch"
        return {
            "reached_level": 1,
            "reproduced": True,
            "expected_action_sequence": ["different"],
        }

    issues = lint.lint_submission_package_payload(package_path, payload, replay_row_fn=replay)
    kinds = [issue.kind for issue in issues]

    assert "SUBMISSION_ROW_NOT_MAPPING" in kinds
    assert kinds.count("SUBMISSION_ROW_NOT_COUNTABLE") == 4
    assert "SUBMISSION_ACTION_SEQUENCE_MISMATCH" in kinds
    assert "SUBMISSION_TOTAL_MISMATCH" in kinds


def test_req_report_4462_path_helpers_and_parse_failures_assert(tmp_path: Path) -> None:
    """REQ-REPORT-4462: parse and path helpers fail closed."""

    assert lint._as_int("bad") == 0
    assert lint._read_yaml_mapping(tmp_path / "missing.yaml") == {}
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("[", encoding="utf-8")
    assert lint._read_yaml_mapping(bad_yaml) == {}
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("[]", encoding="utf-8")
    assert lint._read_yaml_mapping(list_yaml) == {}

    assert lint._read_json_mapping(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert lint._read_json_mapping(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert lint._read_json_mapping(list_json) == {}

    assert lint._reproduced_entries({"games": {}}) == []
    assert lint._reproduced_entries({"games": ["bad", {"game": "zero", "levels_reproduced": 0}]}) == []
    assert lint._looks_like_provisional_inflation({}, [], 0, 0) is False
    assert lint._action_sequence({"action_sequence": "bad"}) == []

    repo_like = tmp_path / "repo"
    (repo_like / "ops").mkdir(parents=True)
    (repo_like / "results").mkdir()
    assert lint._infer_repo_root(repo_like / "ops" / "arc_solve_registry.yaml") == repo_like
    assert lint._infer_repo_root(tmp_path / "elsewhere" / "file.txt") == lint.REPO_ROOT

    _write_yaml(
        repo_like / lint.REGISTRY_RELATIVE_PATH,
        {"games": [{"game": "alpha", "levels_reproduced": 1}]},
    )
    assert lint._registry_entry_for_game(repo_like, "alpha")["levels_reproduced"] == 1
    assert lint._registry_entry_for_game(repo_like, "missing") == {"game": "missing"}


def test_req_report_4462_precommit_hook_is_scoped_to_registry_and_package() -> None:
    """REQ-REPORT-4462: pre-commit runs the lint on registry/package edits."""

    config = PRE_COMMIT_PATH.read_text(encoding="utf-8")
    assert "- id: arc-count-integrity-lint" in config
    hook_block = config.split("- id: arc-count-integrity-lint", maxsplit=1)[1].split(
        "\n      - id:",
        maxsplit=1,
    )[0]
    files_match = re.search(r"files: '([^']+)'", hook_block)

    assert "scripts/arc_count_integrity_lint.py" in hook_block
    assert files_match is not None
    files_re = re.compile(files_match.group(1))
    assert files_re.search("ops/arc_solve_registry.yaml")
    assert files_re.search("results/experiment_4460_submission_package_prep.json")
    assert not files_re.search("results/experiment_4450_inference_substrate_emission_lint_guard.json")


def test_req_report_4462_cli_reports_json_issues(tmp_path: Path, capsys) -> None:
    """REQ-REPORT-4462: CLI emits machine-readable issue reports."""

    registry = _write_yaml(tmp_path / "ops" / "arc_solve_registry.yaml", _registry_payload(total=7))

    exit_code = lint.main(["--json", "--skip-replay", str(registry)])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["ok"] is False
    assert report["issue_count"] == 1
    assert report["issues"][0]["kind"] == "PROVISIONAL_INFLATION"


def test_req_report_4462_cli_default_paths_and_package_path_assert(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """REQ-REPORT-4462: CLI handles default paths and explicit package paths."""

    registry = _write_yaml(
        tmp_path / "ops" / "arc_solve_registry.yaml",
        {"reproducible_total_levels": 0, "games": []},
    )
    package = _write_json(
        tmp_path / "results" / "experiment_4460_submission_package_prep.json",
        {
            "submitted_to_leaderboard": False,
            "total_reproduced_levels_in_package": 1,
            "per_game_replay_validation": [
                {
                    "game": "alpha",
                    "replays_ok": True,
                    "env_matched": True,
                    "reproduced_levels": 1,
                    "action_sequence": ["a"],
                    "reproduction_result": {"reached_level": 1, "reproduced": True},
                }
            ],
        },
    )

    assert lint.lint_submission_package_path(package, replay_row_fn=lambda _row, _root: None) == []
    assert (
        lint.lint_submission_package_payload(
            package,
            json.loads(package.read_text(encoding="utf-8")),
            max_package_replays=0,
        )
        == []
    )
    assert lint.lint_paths([registry, package, tmp_path / "ignored.txt"], skip_replay=True) == []

    monkeypatch.setattr(lint, "REPO_ROOT", tmp_path)
    assert lint.main(["--json", "--skip-replay"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "issue_count": 0, "issues": []}
