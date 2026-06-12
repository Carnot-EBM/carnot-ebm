"""Tests for Exp 4077 verifier-reward RFT corpus build.

Spec refs: REQ-LEARN-4077, SCENARIO-LEARN-4077,
SCENARIO-LEARN-4077-NMATCH.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic import arc_exp4077_verifier_reward_rft_corpus_build as exp4077


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _preconditions(available: bool = True) -> list[exp4077.PreconditionCheck]:
    return [
        exp4077.PreconditionCheck("hf_safetensors_qwen_qwen3_5_0_8b", available, "test"),
        exp4077.PreconditionCheck("hf_safetensors_openbmb_minicpm5_1b", available, "test"),
        exp4077.PreconditionCheck("trl_peft_trainers", available, "test"),
        exp4077.PreconditionCheck("cuda_visible", available, "test"),
        exp4077.PreconditionCheck("arc1_pool", available, "test"),
        exp4077.PreconditionCheck("arc2_pool", available, "test"),
    ]


def _eval(
    task_id: str,
    program_id: str,
    *,
    demo_perfect: bool,
    test_gold: bool,
    split: str = "heldin",
) -> exp4077.ProgramEvaluation:
    return exp4077.ProgramEvaluation(
        task_id=task_id,
        program_id=program_id,
        code=f"def transform(grid):\n    return grid  # {task_id}-{program_id}\n",
        source="fixture_generator_k8",
        split=split,
        demo_perfect=demo_perfect,
        test_gold=test_gold,
        verifier_energy=0.0 if demo_perfect else 1.0,
        error="",
    )


def _passing_evaluations() -> list[exp4077.ProgramEvaluation]:
    rows: list[exp4077.ProgramEvaluation] = []
    for task_id in ("train-a", "train-b"):
        rows.extend(
            [
                _eval(task_id, "correct-0", demo_perfect=True, test_gold=True),
                _eval(task_id, "correct-1", demo_perfect=True, test_gold=True),
                _eval(task_id, "ablate-0", demo_perfect=False, test_gold=False),
                _eval(task_id, "ablate-1", demo_perfect=False, test_gold=False),
                _eval(task_id, "gold-0", demo_perfect=False, test_gold=True),
                _eval(task_id, "gold-1", demo_perfect=False, test_gold=True),
            ]
        )
    rows.extend(
        [
            _eval("eval-heldout", "heldout-0", demo_perfect=True, test_gold=True, split="heldout"),
            _eval("eval-heldout", "heldout-1", demo_perfect=False, test_gold=False, split="heldout"),
        ]
    )
    return rows


def test_req_learn_4077_spec_declares_contract() -> None:
    """REQ-LEARN-4077: OpenSpec declares the gate, arms, and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4077" in spec
    assert "SCENARIO-LEARN-4077" in spec
    assert "SCENARIO-LEARN-4077-NMATCH" in spec
    assert "experiment_4077_verifier_reward_rft_corpus_build.json" in spec
    assert "Qwen/Qwen3.5-0.8B" in spec
    assert "openbmb/MiniCPM5-1B" in spec
    for field in exp4077.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4077_precision_gate_blocks_poisoned_labels() -> None:
    """SCENARIO-LEARN-4077: low certification precision blocks corpus building."""

    evaluations = [
        _eval("heldout", "p0", demo_perfect=True, test_gold=True, split="heldout"),
        _eval("heldout", "p1", demo_perfect=True, test_gold=False, split="heldout"),
        _eval("heldout", "p2", demo_perfect=True, test_gold=False, split="heldout"),
        _eval("heldout", "p3", demo_perfect=False, test_gold=True, split="heldout"),
    ]
    metrics = exp4077.compute_certification_metrics(evaluations)
    artifact = exp4077.build_precision_blocked_artifact(
        metrics,
        preconditions_checked=_preconditions(),
        n_heldout_tasks=1,
        duration_s=0.25,
    )

    assert metrics.precision == pytest.approx(1 / 3)
    assert metrics.recall == pytest.approx(1 / 2)
    assert artifact["honest_verdict"] == "blocked_precision_gate_unmet_0.3333_0.5000"
    assert artifact["n_rft_correct"] == 0
    assert artifact["n_rft_ablation"] == 0
    assert artifact["n_gold_sft"] == 0
    assert artifact["trainer_smoke_passed"] is False
    assert exp4077.artifact_schema_errors(artifact) == []


def test_scenario_learn_4077_nmatch_builds_three_equal_arms(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4077-NMATCH: passing gate yields task-matched corpora."""

    rows = _passing_evaluations()
    metrics = exp4077.compute_certification_metrics([row for row in rows if row.split == "heldout"])
    corpora = exp4077.build_n_matched_corpora(rows)
    manifest = exp4077.build_heldout_eval_manifest(rows)
    artifact = exp4077.build_complete_artifact(
        metrics,
        corpora=corpora,
        heldout_manifest=manifest,
        preconditions_checked=_preconditions(),
        trainer_smoke_passed=True,
        duration_s=0.5,
    )

    assert metrics.gate_passed is True
    assert len(corpora.rft_correct) == len(corpora.rft_ablation) == len(corpora.gold_sft) == 4
    assert artifact["n_rft_correct"] == artifact["n_rft_ablation"] == artifact["n_gold_sft"] == 4
    assert artifact["honest_verdict"] == "complete: rft_corpus_built_3arms_nA_4_nB_4_nC_4_precgate_PASS"
    assert artifact["trainer_smoke_passed"] is True
    assert artifact["n_heldout_tasks"] == 1
    assert {item["task_id"] for item in corpora.rft_correct}.isdisjoint(
        {item["task_id"] for item in manifest}
    )
    assert all(item["demo_perfect"] is False for item in corpora.rft_ablation)
    assert all(item["test_gold"] is True for item in corpora.gold_sft)
    assert exp4077.artifact_schema_errors(artifact) == []

    paths = exp4077.write_corpus_jsonl(corpora, tmp_path)
    assert set(paths) == {"rft_correct", "rft_ablation", "gold_sft"}
    assert paths["rft_correct"].read_text(encoding="utf-8").count("\n") == 4


def test_req_learn_4077_evaluates_programs_with_sandbox_and_verifier() -> None:
    """REQ-LEARN-4077-1: transform programs are executed and verifier-scored."""

    task = exp4077.ArcTask(
        task_id="identity",
        pool_name="fixture",
        demos=[
            {
                "input": [[1, 2], [3, 4]],
                "output": [[1, 2], [3, 4]],
            }
        ],
        test_input=[[5, 6]],
        candidates=[{"grid": [[5, 6]], "correct": True}],
    )
    program = exp4077.CandidateProgram(
        task_id="identity",
        program_id="identity-0",
        code="def transform(grid):\n    return grid\n",
        source="fixture_generator_k8",
    )

    evaluation = exp4077.evaluate_program(task, program)

    assert evaluation.demo_perfect is True
    assert evaluation.test_gold is True
    assert evaluation.verifier_energy == 0.0
    assert evaluation.error == ""

    broken = exp4077.evaluate_program(
        task,
        exp4077.CandidateProgram(
            task_id="identity",
            program_id="broken",
            code="def transform(grid):\n    raise RuntimeError('boom')\n",
            source="fixture_generator_k8",
        ),
    )
    assert broken.demo_perfect is False
    assert broken.test_gold is False
    assert "RuntimeError" in broken.error


def test_req_learn_4077_precondition_checks_are_honest(tmp_path: Path) -> None:
    """REQ-LEARN-4077: missing mandatory resources block before corpus work."""

    hf_root = tmp_path / "hub"
    qwen = hf_root / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / "sha"
    qwen.mkdir(parents=True)
    (qwen / "config.json").write_text("{}", encoding="utf-8")
    (qwen / "model-00001-of-00001.safetensors").write_text("weights", encoding="utf-8")

    missing = exp4077.check_hf_safetensors_model(
        "openbmb/MiniCPM5-1B",
        cache_root=hf_root,
        trust_remote_code=True,
    )
    present = exp4077.check_hf_safetensors_model("Qwen/Qwen3.5-0.8B", cache_root=hf_root)
    blocked = exp4077.build_precondition_blocked_artifact(
        [present, missing],
        duration_s=0.1,
    )

    assert present.available is True
    assert missing.available is False
    assert blocked["honest_verdict"] == "blocked_hf_safetensors_openbmb_minicpm5_1b"
    assert blocked["runner_ready"] is False
    assert exp4077.artifact_schema_errors(blocked) == []

    gguf_only = hf_root / "models--owner--GGUFOnly" / "snapshots" / "sha"
    gguf_only.mkdir(parents=True)
    (gguf_only / "model.gguf").write_text("weights", encoding="utf-8")
    assert exp4077.check_hf_safetensors_model("owner/GGUFOnly", cache_root=hf_root).detail.startswith(
        "gguf_only"
    )

    no_config = hf_root / "models--owner--NoConfig" / "snapshots" / "sha"
    no_config.mkdir(parents=True)
    (no_config / "model.safetensors").write_text("weights", encoding="utf-8")
    assert "config.json missing" in exp4077.check_hf_safetensors_model(
        "owner/NoConfig",
        cache_root=hf_root,
    ).detail


def test_req_learn_4077_run_experiment_writes_blocked_or_complete(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-4077: runner writes stable JSON and does not train on failed gate."""

    output_path = tmp_path / "artifact.json"
    artifact = exp4077.run_experiment(
        repo_root=tmp_path,
        output_path=output_path,
        preconditions_checker=lambda **_: _preconditions(),
        task_loader=lambda **_: (
            [
                exp4077.ArcTask("heldout", "arc1", [], [], []),
            ],
            [
                exp4077.ArcTask("heldout2", "arc1", [], [], []),
            ],
        ),
        program_loader=lambda **_: {
            "heldout": [
                exp4077.CandidateProgram("heldout", "p0", "code", "fixture"),
                exp4077.CandidateProgram("heldout", "p1", "code", "fixture"),
            ],
            "heldout2": [],
        },
        program_evaluator=lambda task, program: _eval(
            task.task_id,
            program.program_id,
            demo_perfect=True,
            test_gold=(program.program_id == "p0"),
            split="heldout",
        ),
        smoke_trainer=lambda *_, **__: True,
    )

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert artifact["honest_verdict"] == "blocked_precision_gate_unmet_0.5000_1.0000"
    assert artifact["trainer_smoke_passed"] is False

    blocked_output = tmp_path / "blocked.json"
    blocked = exp4077.run_experiment(
        repo_root=tmp_path,
        output_path=blocked_output,
        preconditions_checker=lambda **_: _preconditions(False),
    )
    assert blocked["honest_verdict"] == "blocked_hf_safetensors_qwen_qwen3_5_0_8b"
    assert json.loads(blocked_output.read_text(encoding="utf-8")) == blocked


def test_req_learn_4077_loaders_and_schema_defensive_branches(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-4077: loaders reject malformed pools and preserve k>=8 contracts."""

    pool_path = tmp_path / "pool.json.gz"
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "entries": [
                    "skip-me",
                    {"task": "", "demos": []},
                    {
                        "task": "task-a",
                        "demos": [{"input": [[1]], "output": [[1]]}],
                        "test_input": [[1]],
                        "candidates": [{"grid": [[1]], "correct": True}],
                    },
                ]
            },
            handle,
        )
    loaded = exp4077.load_arc_pool(pool_path, pool_name="arc1")
    assert [task.task_id for task in loaded] == ["task-a"]
    assert exp4077.check_arc_pool(pool_path, "arc1_pool").available is True
    assert exp4077.check_arc_pool(tmp_path / "missing.json", "arc1_pool").available is False

    bad_pool = tmp_path / "bad_pool.json"
    bad_pool.write_text(json.dumps({"not_entries": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="entries list"):
        exp4077.load_arc_pool(bad_pool, pool_name="bad")

    empty_pool = tmp_path / "empty_pool.json"
    empty_pool.write_text(json.dumps({"entries": [{"task": ""}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="no usable ARC tasks"):
        exp4077.load_arc_pool(empty_pool, pool_name="empty")

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "tasks": {
                    "task-a": [
                        {"code": "def transform(grid):\n    return grid\n", "draw_index": idx}
                        for idx in range(8)
                    ],
                    "too-short": [{"code": "def transform(grid):\n    return grid\n"}],
                    "bad-row": ["not a mapping"] * 8,
                }
            }
        ),
        encoding="utf-8",
    )
    programs = exp4077.load_program_checkpoint(checkpoint)
    assert len(programs["task-a"]) == 8
    assert programs["task-a"][0].program_id == "task-a:draw0"

    malformed_checkpoint = tmp_path / "malformed_checkpoint.json"
    malformed_checkpoint.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="tasks mapping"):
        exp4077.load_program_checkpoint(malformed_checkpoint)
    no_k_checkpoint = tmp_path / "no_k_checkpoint.json"
    no_k_checkpoint.write_text(json.dumps({"tasks": {"x": []}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no tasks"):
        exp4077.load_program_checkpoint(no_k_checkpoint)

    assert exp4077._normalize_grid(np.array([[1]])) == [[1]]
    assert exp4077._normalize_grid(((1, 2),)) == [[1, 2]]

    bad_artifact = {
        "honest_verdict": "bad",
        "certification_precision": "x",
        "certification_recall": 0.0,
        "n_rft_correct": False,
        "n_rft_ablation": 1,
        "n_gold_sft": 2,
        "n_heldout_tasks": 0,
        "runner_ready": "yes",
        "trainer_smoke_passed": False,
        "preconditions_checked": [{}],
        "inference_substrate": "wrong",
    }
    errors = exp4077.artifact_schema_errors(bad_artifact)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "certification_precision must be numeric" in errors
    assert "n_rft_correct must be a bare int" in errors
    assert "runner_ready must be a bare bool" in errors
    assert "preconditions_checked entries must include resource and available" in errors

    mismatched = dict(bad_artifact, honest_verdict="complete: x", n_rft_correct=1, runner_ready=True)
    assert "complete artifacts must be N-matched across all three corpora" in exp4077.artifact_schema_errors(
        mismatched
    )
    poisoned = dict(
        bad_artifact,
        honest_verdict="blocked_precision_gate_unmet_0.1_1.0",
        certification_precision=0.1,
        n_rft_correct=1,
        runner_ready=True,
        trainer_smoke_passed=True,
        preconditions_checked=[{"resource": "x", "available": True}],
        inference_substrate=exp4077.INFERENCE_SUBSTRATE,
    )
    poison_errors = exp4077.artifact_schema_errors(poisoned)
    assert "precision-gate blocked artifacts must not include corpus rows" in poison_errors
    assert "precision-gate blocked artifacts must not smoke train" in poison_errors

    missing_errors = exp4077.artifact_schema_errors({})
    assert "missing required field honest_verdict" in missing_errors
    assert "honest_verdict must be a string" in missing_errors
    assert "preconditions_checked must be a list" in missing_errors


def test_req_learn_4077_default_split_and_complete_runner_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-4077-NMATCH: runner can complete when the gate passes."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    task_ids = tuple(f"task-{idx:02d}" for idx in range(12))
    pool_entries = [
        {
            "task": task_id,
            "demos": [{"input": [[1]], "output": [[1]]}],
            "test_input": [[1]],
            "candidates": [{"grid": [[1]], "correct": True}],
        }
        for task_id in task_ids
    ]
    with gzip.open(results_dir / "arc3_gap3_stage2_eval_pool.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"entries": pool_entries}, handle)
    (results_dir / "experiment_4012_gap4_local_best_of_n.checkpoint.json").write_text(
        json.dumps(
            {
                "tasks": {
                    task_id: [
                        {"code": "def transform(grid):\n    return grid\n", "draw_index": idx}
                        for idx in range(2)
                    ]
                    for task_id in task_ids
                }
            }
        ),
        encoding="utf-8",
    )
    precision, heldin, eval_tasks = exp4077.load_default_task_splits(repo_root=tmp_path, k_required=2)
    assert [task.task_id for task in precision] == ["task-00", "task-01", "task-02", "task-03"]
    assert [task.task_id for task in heldin] == ["task-04", "task-05", "task-06", "task-07"]
    assert [task.task_id for task in eval_tasks] == ["task-08", "task-09", "task-10", "task-11"]

    small3 = tmp_path / "small3"
    (small3 / "results").mkdir(parents=True)
    with gzip.open(small3 / "results" / "arc3_gap3_stage2_eval_pool.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"entries": pool_entries[:3]}, handle)
    (small3 / "results" / "experiment_4012_gap4_local_best_of_n.checkpoint.json").write_text(
        json.dumps(
            {
                "tasks": {
                    task_id: [{"code": "x"}] * 2
                    for task_id in ("task-00", "task-01", "task-02")
                }
            }
        ),
        encoding="utf-8",
    )
    small_precision, small_heldin, small_eval = exp4077.load_default_task_splits(repo_root=small3, k_required=2)
    assert [task.task_id for task in small_precision] == ["task-00", "task-01", "task-02"]
    assert small_heldin == []
    assert small_eval == []

    monkeypatch.setattr(exp4077, "DEFAULT_PROGRAM_CHECKPOINT", Path("programs.json"))
    (tmp_path / "programs.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "precision": [{"code": "p"}] * 8,
                    "train": [{"code": "t"}] * 8,
                    "eval": [{"code": "e"}] * 8,
                }
            }
        ),
        encoding="utf-8",
    )
    precision_task = exp4077.ArcTask("precision", "arc1", [], [], [])
    train_task = exp4077.ArcTask("train", "arc1", [], [], [])
    eval_task = exp4077.ArcTask("eval", "arc1", [], [], [])

    def evaluator(task: exp4077.ArcTask, program: exp4077.CandidateProgram) -> exp4077.ProgramEvaluation:
        if task.task_id == "precision":
            return _eval(task.task_id, program.program_id, demo_perfect=True, test_gold=True, split="wrong")
        suffix = int(program.program_id.rsplit("draw", 1)[1])
        if task.task_id == "train":
            return _eval(
                task.task_id,
                program.program_id,
                demo_perfect=suffix in (0, 1),
                test_gold=suffix in (0, 1, 4, 5),
                split="wrong",
            )
        return _eval(task.task_id, program.program_id, demo_perfect=False, test_gold=False, split="wrong")

    artifact = exp4077.run_experiment(
        repo_root=tmp_path,
        preconditions_checker=lambda **_: _preconditions(),
        task_loader=lambda **_: ([precision_task], [train_task], [eval_task]),
        program_evaluator=evaluator,
        smoke_trainer=lambda corpora: exp4077.smoke_train_two_tasks(
            corpora,
            trainer_factory=lambda records, arms: len(records) == 2 and arms == ["rft_correct", "rft_ablation", "gold_sft"],
        ),
    )

    assert artifact["honest_verdict"] == "complete: rft_corpus_built_3arms_nA_2_nB_2_nC_2_precgate_PASS"
    assert artifact["trainer_smoke_passed"] is True
    assert (tmp_path / "results" / exp4077.RESULT_FILENAME).exists()
    assert (tmp_path / "results" / "experiment_4077_rft_correct.jsonl").exists()

    assert exp4077.smoke_train_two_tasks(exp4077.RftCorpora([], [], []), trainer_factory=lambda *_: True) is False

    too_small = tmp_path / "small"
    (too_small / "results").mkdir(parents=True)
    with gzip.open(too_small / "results" / "arc3_gap3_stage2_eval_pool.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"entries": pool_entries[:2]}, handle)
    (too_small / "results" / "experiment_4012_gap4_local_best_of_n.checkpoint.json").write_text(
        json.dumps({"tasks": {"a": [{"code": "x"}] * 2, "b": [{"code": "x"}] * 2}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp4077, "DEFAULT_PROGRAM_CHECKPOINT", Path("results/experiment_4012_gap4_local_best_of_n.checkpoint.json"))
    with pytest.raises(ValueError, match="fewer than three"):
        exp4077.load_default_task_splits(repo_root=too_small, k_required=2)
