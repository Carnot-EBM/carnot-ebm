"""Tests for Exp 4270 ARC family provenance recovery.

Spec refs: REQ-VERIFY-4270, SCENARIO-VERIFY-4270.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_family_provenance_recovery_4270 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _grid_hash(grid: Any) -> str:
    raw = json.dumps(grid, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _candidate(candidate_id: str, grid: list[list[int]], *, correct: bool) -> dict[str, Any]:
    return {
        "candidate_grid_hash": _grid_hash(grid),
        "candidate_id": candidate_id,
        "candidate_index": int(candidate_id.rsplit("candidate", 1)[1]),
        "grid": grid,
        "is_correct": correct,
        "source_kinds": ["gold_flag"] if correct else ["pool_candidate"],
        "votes": 1.0,
    }


def _write_fixture(root: Path, *, task_n: int = 6, taxonomy: bool = True) -> None:
    tasks: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    programs: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, str]] = []
    for index in range(task_n):
        raw_task_id = f"task-{index}"
        task_id = f"fixture:{raw_task_id}"
        correct_grid = [[index]]
        wrong_grid = [[index + 100]]
        candidates = [
            _candidate(f"{task_id}::candidate0", wrong_grid, correct=False),
            _candidate(f"{task_id}::candidate1", correct_grid, correct=True),
        ]
        if index == task_n - 1:
            for candidate in candidates:
                candidate["is_correct"] = False
                candidate["source_kinds"] = ["pool_candidate"]
        tasks.append(
            {
                "candidate_count": len(candidates),
                "candidates": candidates,
                "oracle_present": index != task_n - 1,
                "raw_task_id": raw_task_id,
                "source_id": "fixture",
                "task_id": task_id,
                "vote_top_candidate_id": f"{task_id}::candidate0",
                "wrong_majority": index != task_n - 1,
            }
        )
        entries.append({"task": raw_task_id, "candidates": [{"grid": wrong_grid}]})
        programs.append(
            {
                "entry_i": index,
                "task": raw_task_id,
                "pred_grid": correct_grid if index != task_n - 1 else None,
                "pred_hash": _grid_hash(correct_grid) if index != task_n - 1 else None,
                "demo_fit": 1.0,
            }
        )
        if taxonomy and index < 3:
            taxonomy_rows.append(
                {
                    "task_id": raw_task_id,
                    "family_id": f"arc_tgi_family_{index // 2}",
                    "game_id": f"game-{index // 2}",
                }
            )

    pool_path = root / mod.POOL_REL
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump({"schema": "fixture", "tasks": tasks}, handle)
    with gzip.open(root / "results" / "fixture_pool.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)
    _write_json(root / "results" / "fixture_programs.json", {"programs": programs})
    _write_json(
        root / "results" / "arc3_win_condition_survey.json",
        {"per_game_surveys": [{"game": "game-0"}, {"game": "game-1"}]},
    )
    if taxonomy:
        _write_json(root / "results" / "arc_tgi_family_taxonomy.json", {"families": taxonomy_rows})


def _source_specs() -> tuple[mod.SourceSpec, ...]:
    return (
        mod.SourceSpec(
            "fixture",
            Path("results/fixture_pool.json.gz"),
            Path("results/fixture_programs.json"),
            required=False,
        ),
    )


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flags": []}]}


def test_req_verify_4270_spec_declares_family_manifest_contract() -> None:
    """REQ-VERIFY-4270: OpenSpec declares the family provenance artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4270",
        "SCENARIO-VERIFY-4270",
        "python/carnot/reporting/arc_family_provenance_recovery_4270.py",
        "results/experiment_4270_arc_family_provenance_recovery.py",
        "results/experiment_4270_arc_family_manifest.json",
        "blocked_arc_source_taxonomy_unavailable",
        "family_split_feasible",
        "distinct_family_n",
        "per_family_task_count",
        "provenance_manifest_path",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4270_recovers_taxonomy_and_fallback_manifest_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4270: every pool row gets family provenance and a fold."""

    _write_fixture(tmp_path, task_n=6, taxonomy=True)

    artifact = mod.run(
        tmp_path,
        source_specs=_source_specs(),
        min_held_out_task_n=2,
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: arc_family_manifest_recovered_existing_pool_feasible"
    assert artifact["family_split_feasible"] is True
    assert artifact["distinct_family_n"] == 5
    assert artifact["per_family_task_count"]["arc_tgi_family_0"] == 2
    assert artifact["per_family_task_count"]["original_arc_task:task-3"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["adversarial_verify"]["status"] == "clean"

    manifest_path = tmp_path / artifact["provenance_manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "carnot.arc_family_manifest.v1"
    assert len(manifest["rows"]) == 6
    rows_by_task = {row["task_id"]: row for row in manifest["rows"]}
    assert rows_by_task["fixture:task-0"]["family_id"] == "arc_tgi_family_0"
    assert rows_by_task["fixture:task-0"]["game_id"] == "game-0"
    assert rows_by_task["fixture:task-0"]["source_kind"] == "induced"
    assert rows_by_task["fixture:task-4"]["family_id"] == "original_arc_task:task-4"
    assert rows_by_task["fixture:task-4"]["recovered_by"] == "original_arc_task_fallback"
    assert rows_by_task["fixture:task-5"]["target_hash"].startswith("unavailable:")
    assert isinstance(rows_by_task["fixture:task-0"]["fold"], int)
    assert artifact["fallback_rows_sample"] == [
        "fixture:task-3",
        "fixture:task-4",
        "fixture:task-5",
    ]


def test_scenario_4270_reports_infeasible_family_concentration(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4270: concentrated pools route to the fresh-pool fallback."""

    _write_fixture(tmp_path, task_n=5, taxonomy=False)

    artifact = mod.run(
        tmp_path,
        source_specs=_source_specs(),
        min_held_out_task_n=10,
        adversarial_runner=_adversarial_clean,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: arc_family_manifest_recovered_existing_pool_infeasible"
    assert artifact["family_split_feasible"] is False
    assert artifact["distinct_family_n"] == 5
    assert "held-out fold has fewer than 10 tasks" in artifact["infeasible_reason"]
    assert all(key.startswith("original_arc_task:") for key in artifact["per_family_task_count"])


def test_scenario_4270_blocks_when_source_taxonomy_unavailable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4270: no reconstructable join source blocks honestly."""

    pool_path = tmp_path / mod.POOL_REL
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "tasks": [
                    {
                        "task_id": "fixture:task-0",
                        "raw_task_id": "task-0",
                        "source_id": "fixture",
                        "candidates": [],
                    }
                ]
            },
            handle,
        )

    artifact = mod.run(tmp_path, source_specs=_source_specs(), adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BLOCKED_SOURCE_TAXONOMY_VERDICT
    assert artifact["family_split_feasible"] is False
    assert artifact["distinct_family_n"] == 0
    assert artifact["per_family_task_count"] == {}
    assert artifact["provenance_manifest_path"] == ""
    assert artifact["verifier_is_oracle"] is False


def test_req_verify_4270_validation_and_checksums_are_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-4270: bare gate fields and reproducibility checksum are enforced."""

    _write_fixture(tmp_path, task_n=6, taxonomy=True)
    artifact = mod.run(
        tmp_path,
        source_specs=_source_specs(),
        min_held_out_task_n=2,
        adversarial_runner=_adversarial_clean,
    )
    manifest = mod.load_manifest(tmp_path / artifact["provenance_manifest_path"])
    checksum = mod.reproducibility_checksum(
        source_paths=[tmp_path / mod.POOL_REL, tmp_path / "results" / "fixture_pool.json.gz"],
        manifest=manifest,
        random_seed=artifact["random_seed"],
    )
    assert checksum.startswith("sha256:")
    assert mod._family_task_counts(manifest.rows)["arc_tgi_family_0"] == 2

    invalid_cases = [
        ({key: value for key, value in artifact.items() if key != "family_split_feasible"}, "missing required"),
        ({**artifact, "honest_verdict": "done"}, "terminal-prefixed"),
        ({**artifact, "family_split_feasible": {"value": True}}, "bare bool"),
        ({**artifact, "distinct_family_n": {"value": 5}}, "bare int"),
        ({**artifact, "per_family_task_count": []}, "histogram"),
        ({**artifact, "provenance_manifest_path": 4270}, "provenance_manifest_path"),
        ({**artifact, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**artifact, "random_seed": "4270"}, "random_seed"),
        ({**artifact, "field_principles": {}}, "field_principles"),
        ({**artifact, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
