"""Tests for Exp 4243 ARC candidate pool growth.

Spec refs: REQ-CAPSTONE-4243, SCENARIO-CAPSTONE-4243.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import carnot.experiment_4243_arc_candidate_pool_grow as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_pool_pair(root: Path, *, source: str = "primary") -> tuple[Path, Path]:
    pool_path = root / "results" / f"{source}_pool.json.gz"
    programs_path = root / "results" / f"{source}_programs.json"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "task": "task-a",
            "test_input": [[0]],
            "candidates": [
                {"grid": [[0]], "votes": 10, "q_mean": 0.9, "correct": False},
                {"grid": [[1]], "votes": 1, "q_mean": 0.2, "correct": True},
                {"grid": [[1]], "votes": 3, "q_mean": 0.7, "correct": False},
            ],
        },
        {
            "task": "task-b",
            "test_input": [[7]],
            "candidates": [
                {"grid": [[5]], "votes": 8, "q_mean": 0.8, "correct": False},
                {"grid": [[6]], "votes": 2, "q_mean": 0.3, "correct": False},
            ],
        },
    ]
    programs = {
        "programs": [
            {
                "entry_i": 0,
                "task": "task-a",
                "pred_grid": [[2]],
                "demo_fit": 1.0,
                "n_calls": 2,
                "code": "def transform(grid): return [[2]]",
            },
            {
                "entry_i": 1,
                "task": "task-b",
                "pred_grid": [[5]],
                "demo_fit": 0.8,
                "n_calls": 1,
                "code": "def transform(grid): return [[5]]",
            },
        ]
    }
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)
    _write_json(programs_path, programs)
    return pool_path, programs_path


@pytest.fixture
def cheap_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exp, "_detector_row_count", lambda pool_path, programs_path: 99)


def test_req_capstone_4243_spec_declares_pool_growth_contract() -> None:
    """REQ-CAPSTONE-4243: OpenSpec declares the ARC pool deliverable."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4243",
        "SCENARIO-CAPSTONE-4243",
        "positive_candidate_n",
        "wrong_majority_n",
        "pool_artifact_path",
        "blocked_arc_gap4_pools_missing",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in exp.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_capstone_4243_assembles_deduped_gold_and_induced_labels(
    tmp_path: Path, cheap_detector: None
) -> None:
    """SCENARIO-CAPSTONE-4243: gold and induced grids become task-grouped positives."""

    pool_path, programs_path = _write_pool_pair(tmp_path)
    specs = (
        exp.PoolSpec(
            "mini",
            pool_path.relative_to(tmp_path),
            programs_path.relative_to(tmp_path),
            required=True,
        ),
    )

    assembly = exp.assemble_pool(tmp_path, pool_specs=specs)

    assert assembly.task_n == 2
    assert assembly.detector_row_n == 99
    assert assembly.candidate_n == 5
    assert assembly.raw_candidate_n == 5
    assert assembly.positive_candidate_n == 3
    assert assembly.wrong_majority_n == 1
    task_a = next(task for task in assembly.tasks if task["task_id"] == "mini:task-a")
    assert task_a["wrong_majority"] is True
    assert task_a["candidate_count"] == 3
    gold = next(candidate for candidate in task_a["candidates"] if candidate["grid"] == [[1]])
    induced = next(candidate for candidate in task_a["candidates"] if candidate["grid"] == [[2]])
    assert gold["is_correct"] is True
    assert gold["votes"] == 4.0
    assert gold["source_kinds"] == ["gold_flag", "pool_candidate"]
    assert induced["is_correct"] is True
    assert induced["source_kinds"] == ["induced_pred_grid"]
    assert set(exp.FEATURE_NAMES).issubset(induced["features"])


def test_scenario_capstone_4243_run_persists_complete_pool_artifact(
    tmp_path: Path, cheap_detector: None
) -> None:
    """SCENARIO-CAPSTONE-4243: complete artifacts expose bare gate fields."""

    pool_path, programs_path = _write_pool_pair(tmp_path)
    specs = (
        exp.PoolSpec(
            "mini",
            pool_path.relative_to(tmp_path),
            programs_path.relative_to(tmp_path),
            required=True,
        ),
    )

    artifact = exp.run(
        tmp_path,
        pool_specs=specs,
        baseline_positive_n=2,
        baseline_wrong_majority_n=0,
        write=True,
    )

    exp.validate_artifact(artifact, tmp_path)
    assert artifact["honest_verdict"] == "complete: arc_candidate_pool_grown_for_a2"
    assert artifact["arc_pool_grown"] is True
    assert artifact["positive_candidate_n"] == 3
    assert artifact["wrong_majority_n"] == 1
    assert artifact["held_out_task_n"] == 2
    assert artifact["verifier_is_oracle"] is False
    pool_artifact_path = tmp_path / artifact["pool_artifact_path"]
    with gzip.open(pool_artifact_path, "rt", encoding="utf-8") as handle:
        pool_artifact = json.load(handle)
    assert pool_artifact["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert len(pool_artifact["tasks"]) == 2
    written = json.loads(
        (tmp_path / "results" / "experiment_4243_arc_candidate_pool_grow.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["pool_artifact_path"] == artifact["pool_artifact_path"]


def test_scenario_capstone_4243_blocks_when_required_gap_pool_missing(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4243: missing required cached pools stop honestly."""

    artifact = exp.run(tmp_path)

    exp.validate_artifact(artifact, tmp_path)
    assert artifact["honest_verdict"] == "blocked_arc_gap4_pools_missing"
    assert artifact["arc_pool_grown"] is False
    assert artifact["positive_candidate_n"] == 0
    assert artifact["wrong_majority_n"] == 0
    assert artifact["held_out_task_n"] == 0
    assert artifact["pool_artifact_path"] == ""
    assert artifact["verifier_is_oracle"] is False


def test_scenario_capstone_4243_validation_rejects_wrapped_gate_fields(
    tmp_path: Path, cheap_detector: None
) -> None:
    """REQ-CAPSTONE-4243: A2 gate fields must stay bare."""

    pool_path, programs_path = _write_pool_pair(tmp_path)
    specs = (
        exp.PoolSpec(
            "mini",
            pool_path.relative_to(tmp_path),
            programs_path.relative_to(tmp_path),
            required=True,
        ),
    )
    artifact = exp.run(
        tmp_path,
        pool_specs=specs,
        baseline_positive_n=2,
        baseline_wrong_majority_n=0,
        write=True,
    )
    invalid_cases = [
        ({key: value for key, value in artifact.items() if key != "arc_pool_grown"}, "missing required"),
        ({**artifact, "honest_verdict": "done"}, "terminal-prefixed"),
        ({**artifact, "arc_pool_grown": {"value": True}}, "bare bool"),
        ({**artifact, "positive_candidate_n": {"value": 3}}, "bare int"),
        ({**artifact, "wrong_majority_n": {"value": 1}}, "bare int"),
        ({**artifact, "held_out_task_n": {"value": 2}}, "bare int"),
        ({**artifact, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**artifact, "pool_artifact_path": {"value": artifact["pool_artifact_path"]}}, "string"),
        ({**artifact, "model_specs": []}, "model_specs"),
        ({**artifact, "reproducibility_checksum": "bad"}, "sha256-prefixed"),
        ({**artifact, "pool_artifact_path": "results/missing.json.gz"}, "pool artifact"),
    ]

    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(payload, tmp_path)


def test_scenario_capstone_4243_edge_paths_and_optional_pool_skips(
    tmp_path: Path, cheap_detector: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-4243: malformed optional pools do not block the primary pool."""

    pool_path, programs_path = _write_pool_pair(tmp_path, source="primary")
    optional_bad_pool = tmp_path / "results" / "optional_bad_pool.json.gz"
    optional_bad_programs = tmp_path / "results" / "optional_bad_programs.json"
    optional_bad_pool.write_text("not gzip", encoding="utf-8")
    optional_bad_programs.write_text("{}", encoding="utf-8")
    optional_schema_pool = tmp_path / "results" / "optional_schema_pool.json.gz"
    optional_schema_programs = tmp_path / "results" / "optional_schema_programs.json"
    with gzip.open(optional_schema_pool, "wt", encoding="utf-8") as handle:
        json.dump({"entries": {"not": "a-list"}}, handle)
    optional_schema_programs.write_text(json.dumps({"programs": []}), encoding="utf-8")

    specs = (
        exp.PoolSpec(
            "primary",
            pool_path.relative_to(tmp_path),
            programs_path.relative_to(tmp_path),
            required=True,
        ),
        exp.PoolSpec("missing", Path("results/missing.json.gz"), Path("results/missing.json"), False),
        exp.PoolSpec(
            "bad",
            optional_bad_pool.relative_to(tmp_path),
            optional_bad_programs.relative_to(tmp_path),
            False,
        ),
        exp.PoolSpec(
            "schema",
            optional_schema_pool.relative_to(tmp_path),
            optional_schema_programs.relative_to(tmp_path),
            False,
        ),
    )

    assembly = exp.assemble_pool(tmp_path, pool_specs=specs)

    assert assembly.task_n == 2
    assert assembly.skipped_optional_pools == ["missing", "bad", "schema"]
    assert exp._merged_candidates([{}, "bad"], None, {}) == []
    assert exp._task_payload(source_id="x", entry_index=0, entry={"candidates": []}, program={}) is None
    assert (
        exp._task_payload(
            source_id="x",
            entry_index=0,
            entry={"task": "dup", "candidates": [{"grid": [[1]]}, {"grid": [[1]]}]},
            program={},
        )
        is None
    )

    non_dict_root = tmp_path / "non-dict-entry"
    non_dict_pool = non_dict_root / "results" / "pool.json.gz"
    non_dict_programs = non_dict_root / "results" / "programs.json"
    non_dict_pool.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(non_dict_pool, "wt", encoding="utf-8") as handle:
        json.dump({"entries": [None]}, handle)
    non_dict_programs.write_text(json.dumps({"programs": []}), encoding="utf-8")
    non_dict_assembly = exp.assemble_pool(
        non_dict_root,
        pool_specs=(
            exp.PoolSpec(
                "non-dict",
                non_dict_pool.relative_to(non_dict_root),
                non_dict_programs.relative_to(non_dict_root),
                True,
            ),
        ),
    )
    assert non_dict_assembly.task_n == 0

    duplicate_root = tmp_path / "duplicate-task"
    duplicate_pool = duplicate_root / "results" / "pool.json.gz"
    duplicate_programs = duplicate_root / "results" / "programs.json"
    duplicate_pool.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(duplicate_pool, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "entries": [
                    {"task": "same", "candidates": [{"grid": [[0]], "votes": 5}]},
                    {"task": "same", "candidates": [{"grid": [[1]], "votes": 1}]},
                ]
            },
            handle,
        )
    duplicate_programs.write_text(
        json.dumps({"programs": [{"entry_i": 1, "pred_grid": [[1]], "demo_fit": 1.0}]}),
        encoding="utf-8",
    )
    duplicate_assembly = exp.assemble_pool(
        duplicate_root,
        pool_specs=(
            exp.PoolSpec(
                "dup",
                duplicate_pool.relative_to(duplicate_root),
                duplicate_programs.relative_to(duplicate_root),
                True,
            ),
        ),
    )
    assert duplicate_assembly.task_n == 1
    assert duplicate_assembly.tasks[0]["task_id"] == "dup:same"
    assert duplicate_assembly.tasks[0]["candidate_count"] == 2
    assert duplicate_assembly.wrong_majority_n == 1

    monkeypatch.setattr(exp.agg4231, "_task_rows", lambda **_kwargs: [])
    assert (
        exp._task_payload(
            source_id="x",
            entry_index=0,
            entry={"task": "empty-rows", "candidates": [{"grid": [[1]]}, {"grid": [[2]]}]},
            program={"pred_grid": [[3]]},
        )
        is None
    )


def test_scenario_capstone_4243_malformed_required_inputs_block_and_write(
    tmp_path: Path, cheap_detector: None
) -> None:
    """SCENARIO-CAPSTONE-4243: malformed required pools write blocked artifacts."""

    bad_root = tmp_path / "bad-required"
    bad_pool = bad_root / "results" / "bad_pool.json.gz"
    bad_programs = bad_root / "results" / "bad_programs.json"
    bad_pool.parent.mkdir(parents=True, exist_ok=True)
    bad_pool.write_text("not gzip", encoding="utf-8")
    bad_programs.write_text(json.dumps({"programs": []}), encoding="utf-8")
    bad_specs = (
        exp.PoolSpec(
            "bad",
            bad_pool.relative_to(bad_root),
            bad_programs.relative_to(bad_root),
            required=True,
        ),
    )

    artifact = exp.run(bad_root, pool_specs=bad_specs, write=True)

    assert artifact["honest_verdict"] == "blocked_arc_gap4_pools_missing"
    assert (bad_root / "results" / "experiment_4243_arc_candidate_pool_grow.json").exists()

    schema_root = tmp_path / "bad-schema"
    schema_pool = schema_root / "results" / "schema_pool.json.gz"
    schema_programs = schema_root / "results" / "schema_programs.json"
    schema_pool.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(schema_pool, "wt", encoding="utf-8") as handle:
        json.dump({"entries": {"not": "a-list"}}, handle)
    schema_programs.write_text(json.dumps({"programs": []}), encoding="utf-8")
    schema_specs = (
        exp.PoolSpec(
            "schema",
            schema_pool.relative_to(schema_root),
            schema_programs.relative_to(schema_root),
            required=True,
        ),
    )

    schema_artifact = exp.run(schema_root, pool_specs=schema_specs)

    assert schema_artifact["honest_verdict"] == "blocked_arc_gap4_pools_missing"


def test_req_capstone_4243_entrypoint_exists() -> None:
    """REQ-CAPSTONE-4243: the required command path has a Python entrypoint."""

    entrypoint = REPO / "results" / "experiment_4243_arc_candidate_pool_grow.py"

    assert entrypoint.exists()
    assert "carnot.experiment_4243_arc_candidate_pool_grow" in entrypoint.read_text(
        encoding="utf-8"
    )
