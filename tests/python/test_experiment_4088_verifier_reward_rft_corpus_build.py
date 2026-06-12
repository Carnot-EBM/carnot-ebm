"""Tests for Exp 4088 trustworthy verifier-reward RFT corpus build.

Spec refs: REQ-LEARN-4088, SCENARIO-LEARN-4088-BLOCKED,
SCENARIO-LEARN-4088-NMATCH, SCENARIO-LEARN-4088-SMOKE.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.agentic import arc_exp4088_verifier_reward_rft_corpus_build as exp4088


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _checks(available: bool = True) -> list[exp4088.PreconditionCheck]:
    return [
        exp4088.PreconditionCheck("hf_safetensors_qwen_qwen3_5_0_8b", available, "test"),
        exp4088.PreconditionCheck("trl_peft_trainers", available, "test"),
        exp4088.PreconditionCheck("cuda_visible", available, "test"),
        exp4088.PreconditionCheck("exp4087_operating_point", available, "test"),
        exp4088.PreconditionCheck("arc1_pool", available, "test"),
        exp4088.PreconditionCheck("codex_cli", available, "test"),
    ]


def _task(task_id: str) -> exp4088.ArcTask:
    return exp4088.ArcTask(
        task_id=task_id,
        pool_name="fixture",
        demos=[{"input": [[1]], "output": [[1]]}],
        test_input=[[1]],
        candidates=[{"grid": [[1]], "correct": True}],
    )


def _program(task_id: str, program_id: str) -> exp4088.CandidateProgram:
    return exp4088.CandidateProgram(
        task_id=task_id,
        program_id=program_id,
        code=f"def transform(grid):\n    return grid  # {task_id}-{program_id}\n",
        source="fixture_codex_live",
        latency_s=1.0,
        raw_response_sha256="sha",
    )


def _eval(
    task_id: str,
    program_id: str,
    *,
    demo_perfect: bool,
    test_gold: bool,
    split: str = "heldin",
    prediction_hash: str | None = "pred",
    verifier_energy: float | None = None,
) -> exp4088.GeneratedProgramEvaluation:
    return exp4088.GeneratedProgramEvaluation(
        task_id=task_id,
        program_id=program_id,
        code=f"def transform(grid):\n    return grid  # {task_id}-{program_id}\n",
        source="fixture_codex_live",
        split=split,
        demo_perfect=demo_perfect,
        test_gold=test_gold,
        prediction_hash=prediction_hash,
        certified_correct=False,
        verifier_energy=(0.0 if demo_perfect else 1.0) if verifier_energy is None else verifier_energy,
        error="",
    )


def _operating_point() -> dict[str, object]:
    return {
        "filter_stack": "k_of_n_agreement",
        "threshold": "k=1",
        "precision": 0.8824,
        "recall": 0.7143,
        "n_certified": 17,
    }


def _matched_rows() -> list[exp4088.GeneratedProgramEvaluation]:
    rows: list[exp4088.GeneratedProgramEvaluation] = []
    for task_id in ("train-a", "train-b"):
        rows.extend(
            [
                _eval(task_id, "cert", demo_perfect=True, test_gold=True, prediction_hash=f"{task_id}-good"),
                _eval(task_id, "ablate", demo_perfect=False, test_gold=False, prediction_hash=f"{task_id}-bad"),
                _eval(task_id, "gold", demo_perfect=False, test_gold=True, prediction_hash=f"{task_id}-gold"),
            ]
        )
    rows.append(
        _eval(
            "heldout",
            "eval-only",
            demo_perfect=True,
            test_gold=True,
            split="heldout",
            prediction_hash="heldout-good",
        )
    )
    return rows


def test_req_learn_4088_spec_declares_contract() -> None:
    """REQ-LEARN-4088: OpenSpec declares live generation, matched arms, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4088" in spec
    assert "SCENARIO-LEARN-4088-BLOCKED" in spec
    assert "SCENARIO-LEARN-4088-NMATCH" in spec
    assert "SCENARIO-LEARN-4088-SMOKE" in spec
    assert "experiment_4088_verifier_reward_rft_corpus_build.json" in spec
    for field in exp4088.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4088_blocked_precondition_artifact_is_honest() -> None:
    """SCENARIO-LEARN-4088-BLOCKED: missing resources block before live compute."""

    checks = _checks()
    checks[2] = exp4088.PreconditionCheck("cuda_visible", False, "torch.cuda false")
    artifact = exp4088.build_precondition_blocked_artifact(checks, duration_s=0.5)

    assert artifact["honest_verdict"] == "blocked_cuda_visible"
    assert artifact["runner_ready"] is False
    assert artifact["trainer_smoke_passed"] is False
    assert artifact["n_rft_correct"] == artifact["n_rft_ablation"] == artifact["n_gold_sft"] == 0
    assert exp4088.artifact_schema_errors(artifact) == []


def test_req_learn_4088_loads_exp4087_operating_point(tmp_path: Path) -> None:
    """REQ-LEARN-4088-1: Exp 4087's precision rescue gate is inherited."""

    results = tmp_path / "results"
    results.mkdir()
    path = results / "experiment_4087_certification_precision_rescue.json"
    path.write_text(
        json.dumps(
            {
                "precision_rescue_succeeded": True,
                "best_operating_point": _operating_point(),
            }
        ),
        encoding="utf-8",
    )

    check = exp4088.check_exp4087_operating_point(repo_root=tmp_path)
    op = exp4088.load_exp4087_operating_point(tmp_path)

    assert check.available is True
    assert op["filter_stack"] == "k_of_n_agreement"
    assert op["precision"] == 0.8824

    path.write_text(json.dumps({"precision_rescue_succeeded": False}), encoding="utf-8")
    assert exp4088.check_exp4087_operating_point(repo_root=tmp_path).available is False

    path.write_text(json.dumps({"precision_rescue_succeeded": True}), encoding="utf-8")
    assert exp4088.check_exp4087_operating_point(repo_root=tmp_path).available is False

    bad_point = _operating_point() | {"precision": 0.84}
    path.write_text(
        json.dumps({"precision_rescue_succeeded": True, "best_operating_point": bad_point}),
        encoding="utf-8",
    )
    assert exp4088.check_exp4087_operating_point(repo_root=tmp_path).available is False


def test_scenario_learn_4088_nmatch_uses_rescued_certification_rule() -> None:
    """SCENARIO-LEARN-4088-NMATCH: corpora are task-matched by rescued labels."""

    rows = exp4088.apply_operating_point(_matched_rows(), _operating_point())
    corpora = exp4088.build_n_matched_corpora(rows)
    manifest = exp4088.build_heldout_eval_manifest(rows)

    assert len(corpora.rft_correct) == len(corpora.rft_ablation) == len(corpora.gold_sft) == 2
    assert {row["task_id"] for row in corpora.rft_correct} == {"train-a", "train-b"}
    assert {row["task_id"] for row in corpora.rft_ablation} == {"train-a", "train-b"}
    assert {row["task_id"] for row in corpora.gold_sft} == {"train-a", "train-b"}
    assert all(row["certified_correct"] is True for row in corpora.rft_correct)
    assert all(row["certified_correct"] is False for row in corpora.rft_ablation)
    assert all(row["test_gold"] is True for row in corpora.gold_sft)
    assert manifest == [{"split": "heldout", "task_id": "heldout"}]
    assert {row["task_id"] for row in corpora.rft_correct}.isdisjoint(
        {row["task_id"] for row in manifest}
    )
    assert _task("x").gold_outputs == [[[1]]]
    assert exp4088._resource_slug("Qwen/Qwen3.5-0.8B") == "qwen_qwen3_5_0_8b"
    assert exp4088._grid_hash(None) is None
    assert exp4088._grid_hash([[1]]) == exp4088._grid_hash([[1]])
    assert exp4088._grid_equal([[1]], [[1]]) is True
    assert exp4088._grid_equal(None, [[1]]) is False
    assert exp4088._grid_equal([[1]], [[1, 2]]) is False


def test_req_learn_4088_certification_and_evaluation_defensive_branches() -> None:
    """REQ-LEARN-4088-3: certification handles supported and unsupported points."""

    rows = [_eval("train", "p0", demo_perfect=False, test_gold=False, verifier_energy=0.0)]
    graded = exp4088.apply_operating_point(
        rows,
        {"filter_stack": "graded_min_hamming", "threshold": "tau=0.0000"},
    )
    assert graded[0].certified_correct is True

    try:
        exp4088.apply_operating_point(rows, {"filter_stack": "unknown"})
    except ValueError as exc:
        assert "unsupported Exp 4087 operating point" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unsupported operating point did not raise")

    task = _task("train")
    program = _program("train", "0")
    evaluated = exp4088.evaluate_programs(
        [task],
        {"train": [program]},
        split="heldin",
        program_evaluator=lambda _task, _program, _split: _eval(
            "train",
            "0",
            demo_perfect=True,
            test_gold=True,
            split="wrong",
        ),
    )
    assert evaluated[0].split == "heldin"


def test_scenario_learn_4088_smoke_requires_checkpoint_writes(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4088-SMOKE: each arm must write a checkpoint directory."""

    corpora = exp4088.build_n_matched_corpora(
        exp4088.apply_operating_point(_matched_rows(), _operating_point())
    )
    calls: list[tuple[str, int, str]] = []

    def trainer(arm: str, records: list[dict[str, object]], checkpoint: Path, config: exp4088.TrainingConfig) -> None:
        calls.append((arm, len(records), json.dumps(config.to_dict(), sort_keys=True)))
        marker = checkpoint / "checkpoint-1"
        marker.mkdir(parents=True)
        (marker / "trainer_state.json").write_text("{}", encoding="utf-8")

    result = exp4088.smoke_train_two_tasks(corpora, repo_root=tmp_path, trainer_factory=trainer)

    assert result.passed is True
    assert [call[0] for call in calls] == ["rft_correct", "rft_ablation", "gold_sft"]
    assert {call[1] for call in calls} == {2}
    assert len({call[2] for call in calls}) == 1
    for path in result.checkpoint_paths.values():
        assert (Path(path) / "checkpoint-1").is_dir()

    assert exp4088.smoke_train_two_tasks(exp4088.RftCorpora([], [], []), repo_root=tmp_path).passed is False

    incomplete = exp4088.RftCorpora(
        corpora.rft_correct,
        corpora.rft_ablation[:1],
        corpora.gold_sft,
    )
    missing_records = exp4088.smoke_train_two_tasks(
        incomplete,
        repo_root=tmp_path / "missing-records",
        trainer_factory=trainer,
    )
    assert missing_records.passed is False

    no_checkpoint = exp4088.smoke_train_two_tasks(
        corpora,
        repo_root=tmp_path / "no-checkpoint",
        trainer_factory=lambda _arm, _records, _checkpoint, _config: None,
    )
    assert no_checkpoint.passed is False


def test_req_learn_4088_complete_artifact_schema_and_sidecars(tmp_path: Path) -> None:
    """REQ-LEARN-4088: complete artifacts are N-matched and reproducible."""

    rows = exp4088.apply_operating_point(_matched_rows(), _operating_point())
    corpora = exp4088.build_n_matched_corpora(rows)
    manifest = exp4088.build_heldout_eval_manifest(rows)
    smoke = exp4088.SmokeTrainResult(
        passed=True,
        checkpoint_paths={
            "rft_correct": str(tmp_path / "rft"),
            "rft_ablation": str(tmp_path / "abl"),
            "gold_sft": str(tmp_path / "gold"),
        },
        training_config=exp4088.TrainingConfig().to_dict(),
    )
    artifact = exp4088.build_complete_artifact(
        operating_point=_operating_point(),
        corpora=corpora,
        heldout_manifest=manifest,
        preconditions_checked=_checks(),
        generation_records=[
            exp4088.GenerationRecord("train-a", 8, 8, 10.0),
            exp4088.GenerationRecord("train-b", 8, 8, 11.0),
        ],
        smoke_result=smoke,
        duration_s=61.0,
    )

    assert artifact["honest_verdict"] == "complete: rft_corpus_built_3arms_nA_2_nB_2_nC_2_at_prec_0.88"
    assert artifact["n_rft_correct"] == artifact["n_rft_ablation"] == artifact["n_gold_sft"] == 2
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["trainer_smoke_passed"] is True
    assert artifact["reproducibility_checksum"] == exp4088.reproducibility_checksum(artifact)
    assert exp4088.artifact_schema_errors(artifact) == []

    paths = exp4088.write_corpus_jsonl(corpora, tmp_path)
    eval_path = exp4088.write_heldout_eval_manifest(
        manifest,
        tmp_path,
        smoke.checkpoint_paths,
        exp4088.model_specs(),
    )
    assert paths["rft_correct"].read_text(encoding="utf-8").count("\n") == 2
    assert json.loads(eval_path.read_text(encoding="utf-8"))["heldout_task_ids"] == ["heldout"]


def test_req_learn_4088_run_experiment_complete_and_blocked_paths(tmp_path: Path) -> None:
    """REQ-LEARN-4088: runner writes either blocked or complete stable JSON."""

    output = tmp_path / "results" / exp4088.RESULT_FILENAME
    programs = {
        "train-a": [_program("train-a", str(idx)) for idx in range(8)],
        "train-b": [_program("train-b", str(idx)) for idx in range(8)],
        "heldout": [_program("heldout", str(idx)) for idx in range(8)],
    }

    def evaluator(task: exp4088.ArcTask, program: exp4088.CandidateProgram, split: str) -> exp4088.GeneratedProgramEvaluation:
        if split == "heldout":
            return _eval(task.task_id, program.program_id, demo_perfect=True, test_gold=True, split=split)
        idx = int(program.program_id)
        return _eval(
            task.task_id,
            program.program_id,
            demo_perfect=(idx == 0),
            test_gold=(idx in (0, 2)),
            split=split,
            prediction_hash=f"{task.task_id}-{idx}",
        )

    def smoke(corpora: exp4088.RftCorpora) -> exp4088.SmokeTrainResult:
        assert len(corpora.rft_correct) == 2
        return exp4088.SmokeTrainResult(
            True,
            {"rft_correct": "r", "rft_ablation": "a", "gold_sft": "g"},
            exp4088.TrainingConfig().to_dict(),
        )

    artifact = exp4088.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        preconditions_checker=lambda **_: _checks(),
        operating_point_loader=lambda _root: _operating_point(),
        task_loader=lambda **_: ([_task("train-a"), _task("train-b")], [_task("heldout")]),
        program_generator=lambda tasks, **_: (
            {task.task_id: programs[task.task_id] for task in tasks},
            [
                exp4088.GenerationRecord(task.task_id, 8, 8, 1.0)
                for task in tasks
            ],
        ),
        program_evaluator=evaluator,
        smoke_trainer=smoke,
        duration_floor_s=0.0,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"].startswith("complete:")
    assert (tmp_path / "results" / "experiment_4088_rft_correct.jsonl").exists()
    assert (tmp_path / "results" / "experiment_4088_heldout_eval_manifest.json").exists()

    derived = exp4088.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "derived_records.json",
        preconditions_checker=lambda **_: _checks(),
        operating_point_loader=lambda _root: _operating_point(),
        task_loader=lambda **_: ([_task("train-a"), _task("train-b")], [_task("heldout")]),
        program_generator=lambda tasks, **_: {task.task_id: programs[task.task_id] for task in tasks},
        program_evaluator=evaluator,
        smoke_trainer=smoke,
        duration_floor_s=0.0,
    )
    assert derived["methodology"]["generation_records"][0]["latency_s"] == 8.0

    blocked = exp4088.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "blocked.json",
        preconditions_checker=lambda **_: _checks(False),
    )
    assert blocked["honest_verdict"] == "blocked_hf_safetensors_qwen_qwen3_5_0_8b"
