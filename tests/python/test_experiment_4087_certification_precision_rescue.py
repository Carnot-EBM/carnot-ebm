"""Tests for Exp 4087 GAP-5 precision-rescue sweep.

Spec refs: REQ-LEARN-4087, SCENARIO-LEARN-4087,
SCENARIO-LEARN-4087-FAIL.
"""

from __future__ import annotations

import json
import gzip
from pathlib import Path

from carnot.agentic import arc_exp4087_certification_precision_rescue as exp4087


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _identity_task(task_id: str, value: int) -> exp4087.TaskRecord:
    return exp4087.TaskRecord(
        task_id=task_id,
        pool_name="fixture",
        demos=[{"input": [[value, 0], [0, value]], "output": [[value, 0], [0, value]]}],
        test_input=[[value + 1, 0], [0, value + 1]],
        gold_outputs=[[[value + 1, 0], [0, value + 1]]],
    )


def _program(task: exp4087.TaskRecord, program_id: str, code: str) -> exp4087.CandidateProgram:
    return exp4087.CandidateProgram(
        task_key=task.task_key,
        task_id=task.task_id,
        pool_name=task.pool_name,
        program_id=program_id,
        code=code,
        source="fixture",
    )


def _write_pool(path: Path, task_id: str = "task") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "entries": [
                    "skip",
                    {"task": "", "candidates": []},
                    {
                        "task": task_id,
                        "demos": [{"input": [[1]], "output": [[1]]}],
                        "test_input": [[2]],
                        "candidates": [{"grid": [[2]], "correct": True}],
                    },
                ]
            },
            handle,
        )


def _write_programs(path: Path, task_id: str = "task") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "programs": [
                    {"task": "", "code": "def transform(grid):\n    return grid\n"},
                    "skip",
                    {"task": task_id, "code": "def transform(grid):\n    return grid\n"},
                ]
            }
        ),
        encoding="utf-8",
    )


def test_req_learn_4087_spec_declares_contract() -> None:
    """REQ-LEARN-4087: OpenSpec declares the sweep and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4087" in spec
    assert "SCENARIO-LEARN-4087" in spec
    assert "SCENARIO-LEARN-4087-FAIL" in spec
    assert "offline_saved_gap4_program_replay_precision_rescue" in spec
    for field in exp4087.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4087_augmentation_invariance_rejects_hardcoded_demo() -> None:
    """REQ-LEARN-4087-3: D4/color augmentation catches a hardcoded demo fit."""

    task = _identity_task("hardcoded", 1)
    generic = _program(task, "generic", "def transform(grid):\n    return grid\n")
    hardcoded = _program(
        task,
        "hardcoded",
        "def transform(grid):\n    return np.array([[1, 0], [0, 1]])\n",
    )

    rows = exp4087.replay_dataset({task.task_key: task}, {task.task_key: [generic, hardcoded]})
    by_id = {row.program_id: row for row in rows}

    assert by_id["generic"].demo_perfect is True
    assert by_id["generic"].augmentation_invariant is True
    assert by_id["hardcoded"].demo_perfect is True
    assert by_id["hardcoded"].augmentation_invariant is False


def test_scenario_learn_4087_precision_rescue_succeeds_with_invariance() -> None:
    """SCENARIO-LEARN-4087: a stacked filter can pass the precision/recall gate."""

    tasks = [_identity_task(f"good-{idx}", idx + 1) for idx in range(4)]
    bad = _identity_task("bad", 5)
    all_tasks = {task.task_key: task for task in [*tasks, bad]}
    programs = {
        task.task_key: [_program(task, f"{task.task_id}-generic", "def transform(grid):\n    return grid\n")]
        for task in tasks
    }
    programs[bad.task_key] = [
        _program(
            bad,
            "bad-hardcoded",
            "def transform(grid):\n    return np.array([[5, 0], [0, 5]])\n",
        )
    ]

    rows = exp4087.replay_dataset(all_tasks, programs)
    frontier = exp4087.build_frontier(rows, n_tasks_scored=len(all_tasks))
    artifact = exp4087.build_artifact(frontier, n_tasks_scored=len(all_tasks))

    assert artifact["precision_rescue_succeeded"] is True
    assert artifact["honest_verdict"].startswith("complete: precision_rescue_succeeded_best_")
    assert artifact["best_certified_precision"] == 1.0
    assert artifact["best_op_point_recall"] >= 0.8
    assert artifact["n_codex_calls"] == 0
    assert exp4087.artifact_schema_errors(artifact) == []


def test_scenario_learn_4087_precision_rescue_fails_closed() -> None:
    """SCENARIO-LEARN-4087-FAIL: no qualifying point yields a failed verdict."""

    task_a = _identity_task("wrong-a", 1)
    task_b = _identity_task("wrong-b", 2)
    programs = {}
    for task in (task_a, task_b):
        programs[task.task_key] = [
            _program(
                task,
                f"{task.task_id}-wrong",
                f"def transform(grid):\n    return np.array([[{task.demos[0]['input'][0][0]}, 0], [0, {task.demos[0]['input'][0][0]}]])\n",
            )
        ]

    rows = exp4087.replay_dataset(
        {task_a.task_key: task_a, task_b.task_key: task_b},
        programs,
    )
    artifact = exp4087.build_artifact(
        exp4087.build_frontier(rows, n_tasks_scored=2),
        n_tasks_scored=2,
    )

    assert artifact["precision_rescue_succeeded"] is False
    assert artifact["honest_verdict"].startswith("complete: precision_rescue_FAILED_")
    assert artifact["best_certified_precision"] == 0.0
    assert artifact["n_codex_calls"] == 0
    assert exp4087.artifact_schema_errors(artifact) == []


def test_req_learn_4087_run_blocks_on_missing_cached_pools(tmp_path: Path) -> None:
    """REQ-LEARN-4087-1: missing offline pools produce a blocked artifact."""

    output_path = tmp_path / "artifact.json"
    artifact = exp4087.run_experiment(
        repo_root=tmp_path,
        output_path=output_path,
        verifier_gaps_path=tmp_path / "verifier_gaps.md",
        update_verifier_gaps=False,
    )

    assert artifact["honest_verdict"] == "blocked_arc1_induced_programs"
    assert artifact["precision_rescue_succeeded"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert exp4087.artifact_schema_errors(artifact) == []


def test_req_learn_4087_schema_and_gap_update_are_defensive(tmp_path: Path) -> None:
    """REQ-LEARN-4087-6: schema errors are explicit and GAP-5 update is idempotent."""

    bad = {
        "honest_verdict": "bad",
        "precision_rescue_succeeded": "false",
        "best_certified_precision": "1.0",
        "best_op_point_recall": 0.0,
        "frontier": [{"filter_stack": "x", "precision": 1.0}],
        "n_tasks_scored": False,
        "n_codex_calls": 1,
        "random_seed": 4087,
        "reproducibility_checksum": 123,
        "inference_substrate": "wrong",
    }

    errors = exp4087.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "precision_rescue_succeeded must be a bare bool" in errors
    assert "best_certified_precision must be a bare float" in errors
    assert "n_tasks_scored must be a bare int" in errors
    assert "n_codex_calls must be 0 for offline replay" in errors
    assert "frontier entries must include filter_stack, threshold, precision, recall, n_certified" in errors
    assert "reproducibility_checksum must be a string" in errors
    assert "inference_substrate must declare offline replay precision rescue" in errors

    artifact = exp4087.build_artifact(
        [{"filter_stack": "demo_perfect", "threshold": "k=1", "precision": 1.0, "recall": 1.0, "n_certified": 1}],
        n_tasks_scored=1,
    )
    gaps = tmp_path / "verifier_gaps.md"
    gaps.write_text("### GAP-5: demo-underdetermination detection\n", encoding="utf-8")
    exp4087.append_gap5_precision_rescue_entry(gaps, artifact)
    once = gaps.read_text(encoding="utf-8")
    exp4087.append_gap5_precision_rescue_entry(gaps, artifact)
    assert gaps.read_text(encoding="utf-8") == once
    assert "Exp 4087 certification precision-rescue update" in once


def test_req_learn_4087_loader_runner_and_error_branches(tmp_path: Path) -> None:
    """REQ-LEARN-4087-1: loaders recover cached pools and fail loudly on malformed inputs."""

    results = tmp_path / "results"
    _write_pool(results / "arc3_gap3_stage2_eval_pool.json.gz", "arc1-task")
    _write_pool(results / "arc3_gap4_arc2_eval_pool.json.gz", "arc2-task")
    _write_programs(results / "arc3_gap4_induced_programs.json", "arc1-task")
    _write_programs(results / "arc3_gap4_arc2_induced_programs.json", "arc2-task")
    (results / "arc3_gap4_arc2_consistency_ensemble.json").write_text(
        json.dumps(
            {
                "part_b_agreement": {
                    "per_task": [
                        "skip",
                        {
                            "task": "arc2-task",
                            "samples": [
                                {"source": "duplicate", "code": "def transform(grid):\n    return grid\n"},
                                {"source": "none", "code": "def transform(grid):\n    return None\n"},
                                "skip",
                            ],
                        },
                        {"task": "", "samples": []},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (results / "arc3_gap4_arc2_chain_ensemble.json").write_text(
        json.dumps(
            {
                "per_task": [
                    {
                        "task": "arc2-task",
                        "arms": [
                            {
                                "source": "chain",
                                "code": "def transform(grid):\n    return np.array([[2]])\n",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    checks, blocker = exp4087.check_preconditions(repo_root=tmp_path)
    tasks, programs = exp4087.load_default_dataset(repo_root=tmp_path)
    artifact = exp4087.run_experiment(
        repo_root=tmp_path,
        verifier_gaps_path=tmp_path / "ops" / "verifier_gaps.md",
    )

    assert blocker is None
    assert all(check.available for check in checks)
    assert sorted(tasks) == ["arc1:arc1-task", "arc2:arc2-task"]
    assert len(programs["arc2:arc2-task"]) == 3
    assert artifact["n_tasks_scored"] == 2
    assert (results / exp4087.RESULT_FILENAME).exists()
    assert "Exp 4087 certification precision-rescue update" in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")

    assert exp4087._grid_equal(None, [[1]]) is False
    assert exp4087._grid_hash(None) is None
    assert exp4087._cell_disagreement(None, [[1]]) == 1.0
    assert exp4087._cell_disagreement([[1, 2]], [[1]]) == 1.0
    assert exp4087._call_transform(lambda _grid: None, [[1]])[1] == "transform returned None"

    def raises(_grid: object) -> object:
        raise RuntimeError("boom")

    assert "RuntimeError" in exp4087._call_transform(raises, [[1]])[1]
    assert exp4087._augmentation_invariant(None, []) is False
    assert exp4087._best_point([])["filter_stack"] == "none"
    assert exp4087.load_ensemble_programs(results / "arc3_gap4_arc2_consistency_ensemble.json")
    list_payload = results / "list_payload.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert exp4087.load_ensemble_programs(list_payload) == {}

    malformed_pool = results / "malformed_pool.json"
    malformed_pool.write_text(json.dumps({"entries": {}}), encoding="utf-8")
    try:
        exp4087.load_arc_pool(malformed_pool, pool_name="bad")
    except ValueError as exc:
        assert "entries list" in str(exc)
    empty_pool = results / "empty_pool.json"
    empty_pool.write_text(json.dumps({"entries": [{"task": "x", "candidates": []}]}), encoding="utf-8")
    try:
        exp4087.load_arc_pool(empty_pool, pool_name="bad")
    except ValueError as exc:
        assert "no usable ARC tasks" in str(exc)
    malformed_programs = results / "malformed_programs.json"
    malformed_programs.write_text(json.dumps({"programs": {}}), encoding="utf-8")
    try:
        exp4087.load_induced_programs(malformed_programs, pool_name="bad")
    except ValueError as exc:
        assert "programs list" in str(exc)
    bad_json = results / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert exp4087._check_loadable(bad_json, "bad_json").available is False
    empty_object = results / "empty_object.json"
    empty_object.write_text("{}", encoding="utf-8")
    assert exp4087._check_loadable(empty_object, "empty_object").available is False
    no_match = tmp_path / "nomatch"
    (no_match / "results").mkdir(parents=True)
    _write_pool(no_match / "results" / "arc3_gap3_stage2_eval_pool.json.gz", "a")
    _write_pool(no_match / "results" / "arc3_gap4_arc2_eval_pool.json.gz", "b")
    _write_programs(no_match / "results" / "arc3_gap4_induced_programs.json", "x")
    _write_programs(no_match / "results" / "arc3_gap4_arc2_induced_programs.json", "y")
    try:
        exp4087.load_default_dataset(repo_root=no_match)
    except ValueError as exc:
        assert "no cached programs match" in str(exc)


def test_req_learn_4087_replay_and_selection_defensive_branches() -> None:
    """REQ-LEARN-4087-2/4/5: replay and task selection handle abstention cases."""

    task = _identity_task("branches", 1)
    compile_failed = exp4087.replay_program(
        task,
        _program(task, "bad-code", "x = 1\n"),
    )
    demo_error = exp4087.replay_program(
        task,
        _program(
            task,
            "demo-error",
            "def transform(grid):\n"
            "    if int(grid[0, 0]) == 1:\n"
            "        raise RuntimeError('demo')\n"
            "    return grid\n",
        ),
    )
    test_error = exp4087.replay_program(
        task,
        _program(
            task,
            "test-error",
            "def transform(grid):\n"
            "    if int(grid[0, 0]) == 2:\n"
            "        raise RuntimeError('test')\n"
            "    return grid\n",
        ),
    )

    assert compile_failed.compile_ok is False
    assert compile_failed.error == "compile_failed"
    assert "demo0:transform returned None" in demo_error.error
    assert "test:transform returned None" in test_error.error

    base = exp4087.CandidateReplay(
        task_key="fixture:t",
        task_id="t",
        pool_name="fixture",
        program_id="base",
        source="fixture",
        code_hash="base",
        compile_ok=True,
        demo_perfect=True,
        augmentation_invariant=True,
        prediction_hash="same",
        prediction=[[1]],
        test_gold=True,
        min_hamming_energy=0.0,
        error="",
    )
    none_pred = exp4087.CandidateReplay(
        **{**base.__dict__, "program_id": "none", "prediction_hash": None}
    )
    not_demo = exp4087.CandidateReplay(
        **{**base.__dict__, "program_id": "not-demo", "demo_perfect": False}
    )
    not_invariant = exp4087.CandidateReplay(
        **{**base.__dict__, "program_id": "not-invariant", "augmentation_invariant": False}
    )
    high_tau = exp4087.CandidateReplay(
        **{**base.__dict__, "program_id": "high-tau", "min_hamming_energy": 1.0}
    )
    other = exp4087.CandidateReplay(
        **{**base.__dict__, "program_id": "other", "prediction_hash": "other"}
    )

    assert exp4087._certified_for_task(
        [none_pred],
        require_demo_perfect=True,
        require_invariance=False,
        min_agreement=1,
        tau=None,
    ) == []
    assert exp4087._certified_for_task(
        [not_demo],
        require_demo_perfect=True,
        require_invariance=False,
        min_agreement=1,
        tau=None,
    ) == []
    assert exp4087._certified_for_task(
        [not_invariant],
        require_demo_perfect=True,
        require_invariance=True,
        min_agreement=1,
        tau=None,
    ) == []
    assert exp4087._certified_for_task(
        [high_tau],
        require_demo_perfect=True,
        require_invariance=False,
        min_agreement=1,
        tau=0.0,
    ) == []
    assert exp4087._certified_for_task(
        [base],
        require_demo_perfect=True,
        require_invariance=False,
        min_agreement=2,
        tau=None,
    ) == []
    certified_ids = {
        row.program_id
        for row in exp4087._certified_for_task(
            [base, other],
            require_demo_perfect=True,
            require_invariance=False,
            min_agreement=1,
            tau=None,
        )
    }
    assert certified_ids == {"base", "other"}

    missing = exp4087.artifact_schema_errors(
        {
            "honest_verdict": None,
            "precision_rescue_succeeded": False,
            "best_certified_precision": 0.0,
            "best_op_point_recall": 0.0,
            "frontier": "bad",
            "n_tasks_scored": 0,
            "n_codex_calls": 0,
            "random_seed": 4087,
            "reproducibility_checksum": "",
            "inference_substrate": exp4087.INFERENCE_SUBSTRATE,
        }
    )
    assert "honest_verdict must be a string" in missing
    assert "frontier must be a list" in missing
    assert "missing required field honest_verdict" in exp4087.artifact_schema_errors({})
