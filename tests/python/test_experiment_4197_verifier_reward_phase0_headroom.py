"""Tests for Exp 4197 verifier-as-reward code operating point.

Spec refs: REQ-CODE-4197, SCENARIO-CODE-4197-PHASE0,
SCENARIO-CODE-4197-HARNESS.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from carnot import experiment_4197_verifier_reward_phase0_headroom as exp4197


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "code-verification" / "spec.md"


def _candidate(
    task_id: str,
    draw_index: int,
    *,
    visible: bool,
    hidden: bool,
    truncated: bool = False,
    status: str = "ok",
    code: str | None = None,
) -> exp4197.CodeCandidate:
    return exp4197.CodeCandidate(
        task_id=task_id,
        draw_index=draw_index,
        code=code or f"def solve_{task_id.replace('/', '_')}_{draw_index}():\n    return {draw_index}\n",
        visible_passes=(visible,),
        hidden_passes=(hidden,),
        status=status,
        truncated=truncated,
        error=None if status == "ok" else "boom",
        generation_seconds=0.25,
    )


def test_req_code_4197_spec_declares_artifact_contract() -> None:
    """REQ-CODE-4197: OpenSpec declares the phase-0 and harness contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-4197" in spec
    assert "SCENARIO-CODE-4197-PHASE0" in spec
    assert "SCENARIO-CODE-4197-HARNESS" in spec
    for field in exp4197.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4197.FIELD_PRINCIPLES


def test_scenario_code_4197_phase0_metrics_are_bare_numbers() -> None:
    """SCENARIO-CODE-4197-PHASE0: visible-perfect labels yield bare metrics."""

    candidates = [
        _candidate("HumanEval/0", 0, visible=True, hidden=True),
        _candidate("HumanEval/0", 1, visible=True, hidden=True),
        _candidate("HumanEval/1", 0, visible=True, hidden=False),
        _candidate("HumanEval/1", 1, visible=False, hidden=True),
        _candidate("HumanEval/2", 0, visible=False, hidden=False),
        _candidate("HumanEval/2", 1, visible=False, hidden=False),
    ]

    metrics = exp4197.compute_phase0_metrics(candidates)

    assert metrics.phase0_precision == pytest.approx(2 / 3)
    assert metrics.tpr == pytest.approx(2 / 3)
    assert metrics.fpr == pytest.approx(1 / 3)
    assert metrics.youden_j == pytest.approx(1 / 3)
    assert metrics.visible_perfect_count == 3
    assert metrics.hidden_pass_count == 3
    assert metrics.to_artifact_fields()["phase0_precision"] == pytest.approx(2 / 3)
    assert isinstance(metrics.to_artifact_fields()["phase0_precision"], float)
    assert isinstance(metrics.to_artifact_fields()["youden_j"], float)


def test_scenario_code_4197_headroom_and_truncation_guard() -> None:
    """REQ-CODE-4197: headroom is bare bool and generation suitability is explicit."""

    first_draws = [
        _candidate("HumanEval/0", 0, visible=True, hidden=True),
        _candidate("HumanEval/1", 0, visible=True, hidden=False),
        _candidate("HumanEval/2", 0, visible=False, hidden=False, truncated=True),
        _candidate("HumanEval/3", 0, visible=False, hidden=False, status="error"),
    ]
    suitability = exp4197.compute_generation_suitability(
        first_draws,
        headroom_upper_bound=0.60,
        max_allowed_truncation=0.30,
    )

    assert suitability.training_headroom_present is True
    assert suitability.base_passrate == pytest.approx(0.25)
    assert suitability.own_visible_perfect_rate == pytest.approx(0.5)
    assert suitability.truncation_rate == pytest.approx(0.25)
    assert suitability.no_answer_rate == pytest.approx(0.25)


def test_scenario_code_4197_three_arm_runner_builds_matched_corpora() -> None:
    """SCENARIO-CODE-4197-HARNESS: arm A/B are N-matched from one generator."""

    tasks = [
        exp4197.CodeTask(
            task_id="HumanEval/0",
            prompt="write add",
            entry_point="add",
            visible_tests=(),
            hidden_tests=(),
            candidates=[
                _candidate("HumanEval/0", 0, visible=True, hidden=True),
                _candidate("HumanEval/0", 1, visible=False, hidden=False),
                _candidate("HumanEval/0", 2, visible=False, hidden=True),
            ],
        ),
        exp4197.CodeTask(
            task_id="HumanEval/1",
            prompt="write sub",
            entry_point="sub",
            visible_tests=(),
            hidden_tests=(),
            candidates=[
                _candidate("HumanEval/1", 0, visible=True, hidden=False),
                _candidate("HumanEval/1", 1, visible=False, hidden=False),
                _candidate("HumanEval/1", 2, visible=True, hidden=True),
            ],
        ),
    ]

    corpora = exp4197.build_three_arm_corpora(tasks, seed=4197)

    assert len(corpora.arm_a_certified) == 3
    assert len(corpora.arm_b_random_control) == len(corpora.arm_a_certified)
    assert {ex.arm for ex in corpora.arm_a_certified} == {"A_certified"}
    assert {ex.arm for ex in corpora.arm_b_random_control} == {"B_random_same_generator"}
    assert len(corpora.arm_c_hidden_gold) == 3
    assert corpora.arm_d_cold_base == ()
    assert all(not ex.visible_perfect for ex in corpora.arm_b_random_control)


def test_req_code_4197_artifact_fields_and_checksum_are_stable(tmp_path: Path) -> None:
    """REQ-CODE-4197: artifact has bare gated fields and reproducibility checksum."""

    phase0_candidates = [
        _candidate("HumanEval/0", 0, visible=True, hidden=True),
        _candidate("HumanEval/0", 1, visible=True, hidden=True),
        _candidate("HumanEval/1", 0, visible=True, hidden=True),
        _candidate("HumanEval/2", 0, visible=False, hidden=False),
        _candidate("HumanEval/2", 1, visible=False, hidden=False),
        _candidate("HumanEval/3", 0, visible=False, hidden=True),
    ]
    suitability_candidates = [
        _candidate("HumanEval/0", 0, visible=True, hidden=True),
        _candidate("HumanEval/1", 0, visible=True, hidden=False),
        _candidate("HumanEval/2", 0, visible=False, hidden=False),
        _candidate("HumanEval/3", 0, visible=False, hidden=True),
    ]
    phase0 = exp4197.compute_phase0_metrics(phase0_candidates)
    suitability = exp4197.compute_generation_suitability(suitability_candidates)
    smoke = exp4197.smoke_three_arm_runner(
        [
            exp4197.CodeTask("HumanEval/0", "p0", "f", (), (), phase0_candidates[:2]),
            exp4197.CodeTask("HumanEval/1", "p1", "g", (), (), phase0_candidates[2:]),
        ],
        seed=4197,
    )

    artifact = exp4197.build_result_artifact(
        phase0=phase0,
        suitability=suitability,
        smoke=smoke,
        model_specs={
            "trainable_base": "google/gemma-4-E4B-it",
            "certification_reference": "unsloth/gemma-4-12B-it-GGUF",
        },
        operating_point={
            "base": "google/gemma-4-E4B-it",
            "corpus": "fixture",
            "K": 2,
            "max_new_tokens": 256,
            "base_passrate": suitability.base_passrate,
            "own_visible_perfect_rate": suitability.own_visible_perfect_rate,
            "truncation_rate": suitability.truncation_rate,
        },
        random_seed=4197,
        source_paths=["fixture.json"],
        duration_s=0.5,
    )

    assert isinstance(artifact["phase0_precision"], float)
    assert isinstance(artifact["youden_j"], float)
    assert isinstance(artifact["training_headroom_present"], bool)
    assert artifact["harness_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"]["phase0_precision"].startswith("BARE float")
    assert artifact["reproducibility_checksum"].startswith("sha256:")

    out = tmp_path / exp4197.RESULT_FILENAME
    exp4197.write_artifact(artifact, out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_code_4197_result_script_delegates_to_module() -> None:
    """REQ-CODE-4197: requested results script remains executable by path."""

    script = REPO / "results" / "experiment_4197_verifier_reward_phase0_headroom_harness_build.py"
    spec = importlib.util.spec_from_file_location("exp4197_result_script", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.main is exp4197.main


def test_req_code_4197_checkpoint_and_headroom_loaders(tmp_path: Path) -> None:
    """SCENARIO-CODE-4197-PHASE0: checkpoint loaders normalize raw vectors."""

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "evaluations_by_task": {
                    "HumanEval/2": [
                        "skip",
                        {
                            "task_id": "HumanEval/2",
                            "draw_index": 3,
                            "code": "def f():\n    return 1\n",
                            "visible_passes": [1, True],
                            "hidden_passes": [0, False],
                            "status": "ok",
                            "truncated": True,
                            "error": None,
                            "generation_seconds": "1.5",
                        },
                        {
                            "code": "",
                            "visible_passes": [],
                            "hidden_passes": [],
                            "status": "",
                            "error": "bad",
                            "generation_seconds": "nan",
                        },
                    ],
                    "bad": "not-a-candidate-list",
                }
            }
        ),
        encoding="utf-8",
    )
    tasks = exp4197.load_checkpoint_tasks(checkpoint)

    assert len(tasks) == 1
    assert tasks[0].task_id == "HumanEval/2"
    assert [c.draw_index for c in tasks[0].candidates] == [2, 3]
    assert exp4197.flatten_candidates(tasks)[1].generation_seconds == pytest.approx(1.5)
    assert exp4197.first_draw_candidates(tasks)[0].draw_index == 2

    with pytest.raises(ValueError, match="evaluations_by_task"):
        bad_checkpoint = tmp_path / "bad-checkpoint.json"
        bad_checkpoint.write_text("{}", encoding="utf-8")
        exp4197.load_checkpoint_tasks(bad_checkpoint)

    exp1999 = tmp_path / "exp1999.json"
    exp1999.write_text(
        json.dumps(
            {
                "results": [
                    {"task_id": "HumanEval/0", "baseline_passed": True},
                    {"baseline_passed": False},
                    "skip",
                ]
            }
        ),
        encoding="utf-8",
    )
    headroom = exp4197.load_exp1999_headroom_candidates(exp1999, limit=3)
    assert [c.task_id for c in headroom] == ["HumanEval/0", "HumanEval/1"]
    assert [c.hidden_pass for c in headroom] == [True, False]

    with pytest.raises(ValueError, match="results"):
        bad_exp1999 = tmp_path / "bad-exp1999.json"
        bad_exp1999.write_text("{}", encoding="utf-8")
        exp4197.load_exp1999_headroom_candidates(bad_exp1999)


def test_req_code_4197_branch_helpers_and_blocked_artifact(tmp_path: Path) -> None:
    """REQ-CODE-4197: branchy helpers stay deterministic and serializable."""

    assert exp4197.compute_generation_suitability([]).gen_suitable is False
    assert exp4197._float_or_none(0.25) == pytest.approx(0.25)
    assert exp4197._float_or_none(True) is None
    assert exp4197._float_or_none("bad") is None
    assert exp4197._float_or_none(object()) is None
    assert exp4197._model_cache_path("org/model").name == "models--org--model"
    assert exp4197._jsonable(Path("x")) == "x"
    assert exp4197._jsonable(exp4197.PreconditionCheck("r", True, "d")) == {
        "available": True,
        "detail": "d",
        "resource": "r",
    }

    class Scalar:
        def item(self) -> int:
            return 5

    assert exp4197._jsonable(Scalar()) == 5
    assert exp4197.reproducibility_checksum(1, [tmp_path / "missing"], {"a": 1}).startswith("sha256:")
    existing = tmp_path / "data.txt"
    existing.write_text("payload", encoding="utf-8")
    assert exp4197.reproducibility_checksum(1, [existing], {"a": 1}).startswith("sha256:")

    checks = [
        exp4197.PreconditionCheck("nonqwen_trainable_base_cached", False, "missing"),
        exp4197.PreconditionCheck("cuda_available", True, "ok"),
        exp4197.PreconditionCheck("code_corpus_loadable", True, "ok"),
        exp4197.PreconditionCheck("restricted_exec_sandbox_importable", True, "ok"),
    ]
    assert exp4197._blocked_verdict(checks) == "blocked_no_nonqwen_base_cached"
    checks[0] = exp4197.PreconditionCheck("nonqwen_trainable_base_cached", True, "ok")
    checks[1] = exp4197.PreconditionCheck("cuda_available", False, "missing")
    assert exp4197._blocked_verdict(checks) == "blocked_cuda_unavailable"
    checks[1] = exp4197.PreconditionCheck("cuda_available", True, "ok")
    checks[2] = exp4197.PreconditionCheck("code_corpus_loadable", False, "missing")
    assert exp4197._blocked_verdict(checks) == "blocked_code_corpus_or_sandbox_missing"
    checks[2] = exp4197.PreconditionCheck("code_corpus_loadable", True, "ok")
    checks[3] = exp4197.PreconditionCheck("restricted_exec_sandbox_importable", False, "missing")
    assert exp4197._blocked_verdict(checks) == "blocked_code_corpus_or_sandbox_missing"
    checks[3] = exp4197.PreconditionCheck("restricted_exec_sandbox_importable", True, "ok")
    assert exp4197._blocked_verdict(checks) is None

    blocked = exp4197.build_blocked_artifact("blocked_cuda_unavailable", checks=checks, random_seed=1, duration_s=0.1)
    assert blocked["honest_verdict"] == "blocked_cuda_unavailable"
    assert blocked["reproducibility_checksum"].startswith("sha256:")


def test_req_code_4197_arm_matching_edge_cases() -> None:
    """SCENARIO-CODE-4197-HARNESS: controls handle scarce or absent negatives."""

    mostly_certified = [
        exp4197.CodeTask(
            "HumanEval/0",
            "p",
            "f",
            (),
            (),
            [
                _candidate("HumanEval/0", 0, visible=True, hidden=True),
                _candidate("HumanEval/0", 1, visible=True, hidden=True),
                _candidate("HumanEval/0", 2, visible=False, hidden=False),
            ],
        )
    ]
    corpora = exp4197.build_three_arm_corpora(mostly_certified, seed=1)
    assert len(corpora.arm_a_certified) == 2
    assert len(corpora.arm_b_random_control) == 2
    assert {ex.source_draw_index for ex in corpora.arm_b_random_control} == {2}

    all_certified = [
        exp4197.CodeTask(
            "HumanEval/1",
            "p",
            "f",
            (),
            (),
            [_candidate("HumanEval/1", 0, visible=True, hidden=True)],
        )
    ]
    assert exp4197.build_three_arm_corpora(all_certified).arm_b_random_control == ()
    assert exp4197.select_smoke_tasks(all_certified, n_tasks=2) == all_certified
    failed_smoke = exp4197.smoke_three_arm_runner(all_certified)
    assert failed_smoke.harness_ready is False
    assert failed_smoke.detail == "matched arm smoke failed"


def test_req_code_4197_result_branches_and_run_with_temp_fixtures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CODE-4197: full run path writes the requested artifact from fixtures."""

    checkpoint = tmp_path / "phase0.checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "evaluations_by_task": {
                    "HumanEval/0": [
                        {
                            "task_id": "HumanEval/0",
                            "draw_index": 0,
                            "code": "def f():\n    return 1\n",
                            "visible_passes": [True],
                            "hidden_passes": [True],
                            "status": "ok",
                            "truncated": False,
                        },
                        {
                            "task_id": "HumanEval/0",
                            "draw_index": 1,
                            "code": "def f():\n    return 0\n",
                            "visible_passes": [False],
                            "hidden_passes": [False],
                            "status": "ok",
                            "truncated": False,
                        },
                    ],
                    "HumanEval/1": [
                        {
                            "task_id": "HumanEval/1",
                            "draw_index": 0,
                            "code": "def g():\n    return 1\n",
                            "visible_passes": [True],
                            "hidden_passes": [True],
                            "status": "ok",
                            "truncated": False,
                        },
                        {
                            "task_id": "HumanEval/1",
                            "draw_index": 1,
                            "code": "def g():\n    return 0\n",
                            "visible_passes": [False],
                            "hidden_passes": [False],
                            "status": "ok",
                            "truncated": False,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    headroom = tmp_path / "exp1999.json"
    headroom.write_text(
        json.dumps(
            {
                "results": [
                    {"task_id": "HumanEval/0", "baseline_passed": True},
                    {"task_id": "HumanEval/1", "baseline_passed": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp4197, "_cached_sota_specs", lambda: [{"name": "fixture", "model_path": "x.gguf"}])
    out = tmp_path / exp4197.RESULT_FILENAME

    artifact = exp4197.run(
        output_path=out,
        phase0_checkpoint=checkpoint,
        headroom_artifact=headroom,
        random_seed=123,
        check_runtime_preconditions=False,
    )

    assert out.is_file()
    assert artifact["harness_ready"] is True
    assert artifact["operating_point"]["K"] == 2
    assert artifact["measurement_sources"]["phase0_candidates"] == 4
    assert json.loads(out.read_text(encoding="utf-8"))["random_seed"] == 123

    blocked_checks = [exp4197.PreconditionCheck("nonqwen_trainable_base_cached", False, "missing")]
    monkeypatch.setattr(exp4197, "check_preconditions", lambda: blocked_checks)
    blocked = exp4197.run(output_path=tmp_path / "blocked.json", check_runtime_preconditions=True)
    assert blocked["honest_verdict"] == "blocked_no_nonqwen_base_cached"

    phase0_bad = exp4197.Phase0Metrics(tp=0, fp=1, fn=1, tn=0)
    good_suitability = exp4197.compute_generation_suitability(
        [_candidate("HumanEval/0", 0, visible=True, hidden=True), _candidate("HumanEval/1", 0, visible=False, hidden=False)]
    )
    good_smoke = exp4197.SmokeResult(True, 2, {"arm_a_certified": 1, "arm_b_random_control": 1, "arm_c_hidden_gold": 1, "arm_d_cold_base": 0}, "runner", "ok")
    no_phase0 = exp4197.build_result_artifact(
        phase0=phase0_bad,
        suitability=good_suitability,
        smoke=good_smoke,
        model_specs={},
        operating_point={},
        random_seed=1,
        source_paths=[],
        duration_s=0.0,
    )
    assert no_phase0["honest_verdict"].endswith("_phase0")

    bad_suitability = exp4197.compute_generation_suitability(
        [_candidate("HumanEval/0", 0, visible=True, hidden=True)], headroom_upper_bound=0.1
    )
    no_headroom = exp4197.build_result_artifact(
        phase0=exp4197.Phase0Metrics(tp=2, fp=0, fn=0, tn=1),
        suitability=bad_suitability,
        smoke=good_smoke,
        model_specs={},
        operating_point={},
        random_seed=1,
        source_paths=[],
        duration_s=0.0,
    )
    assert no_headroom["honest_verdict"].endswith("_headroom")

    bad_smoke = exp4197.SmokeResult(False, 2, {}, "runner", "bad")
    no_harness = exp4197.build_result_artifact(
        phase0=exp4197.Phase0Metrics(tp=2, fp=0, fn=0, tn=1),
        suitability=good_suitability,
        smoke=bad_smoke,
        model_specs={},
        operating_point={},
        random_seed=1,
        source_paths=[],
        duration_s=0.0,
    )
    assert no_harness["honest_verdict"].endswith("_harness")
