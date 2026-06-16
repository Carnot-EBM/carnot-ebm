"""Tests for Exp 4303 ARC verifier efficiency iso-FLOPs rerun.

Spec refs: REQ-VERIFY-4303, SCENARIO-VERIFY-4303.
"""

from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import verifier_efficiency_parity_isoflops_4303 as mod


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


class FakeJudge:
    def __init__(self, choices: list[int], *, latency_s: float = 3.0) -> None:
        self.choices = list(choices)
        self.latency_s = latency_s
        self.records: list[dict[str, Any]] = []
        self.calls = 0

    def judge(self, problem: str, candidates: list[str]) -> int:
        assert "ARC held-out-family selection" in problem
        assert any("Candidate 0" in candidate for candidate in candidates)
        self.calls += 1
        choice = self.choices.pop(0)
        self.records.append(
            {
                "chosen_index": choice,
                "latency_s": self.latency_s,
                "prompt_tokens": 180,
                "completion_tokens": 32,
                "total_tokens": 212,
                "raw_output": f"Grid reasoning: compare finalists. Final answer: {choice}",
                "parse_status": "parsed_final_answer",
            }
        )
        return choice


def _candidate(task_id: str, index: int, *, correct: bool, vote: float) -> dict[str, Any]:
    return {
        "candidate_id": f"{task_id}::candidate{index}",
        "candidate_index": index,
        "grid": [[index, 0], [0, index]],
        "is_correct": correct,
        "features": {
            "vote_weight": vote,
            "self_consistency_margin": vote - 0.5,
            "cell_confidence_mean": vote,
            "cell_confidence_margin": 0.1,
            "program_demo_fit": 1.0,
            "program_length": 10.0 + index,
        },
    }


def _make_repo(root: Path, *, task_n: int = 3) -> Path:
    tasks = []
    manifest_rows = []
    task_rows = []
    for idx in range(task_n):
        task_id = f"gap3_stage2:T{idx}"
        correct_index = 1 if idx != 1 else 2
        energy_index = 1 if idx != 1 else 0
        tasks.append(
            {
                "task_id": task_id,
                "raw_task_id": f"T{idx}",
                "source_id": "gap3_stage2",
                "candidate_count": 3,
                "candidates": [
                    _candidate(task_id, 0, correct=correct_index == 0, vote=0.7),
                    _candidate(task_id, 1, correct=correct_index == 1, vote=0.2),
                    _candidate(task_id, 2, correct=correct_index == 2, vote=0.1),
                ],
            }
        )
        manifest_rows.append(
            {
                "task_id": task_id,
                "raw_task_id": f"T{idx}",
                "source_id": "gap3_stage2",
                "source_kind": "induced",
                "family_id": f"family-{idx}",
                "fold": idx,
                "target_hash": "hash",
                "recovered_by": "fixture",
                "target_hash_recovered": True,
                "source_join_found": True,
            }
        )
        task_rows.append(
            {
                "task_id": task_id,
                "family_id": f"family-{idx}",
                "fold": idx,
                "oracle_hit": True,
                "vote_candidate_id": f"{task_id}::candidate0",
                "vote_correct": correct_index == 0,
                "set_encoder_candidate_id": f"{task_id}::candidate{energy_index}",
                "set_encoder_correct": correct_index == energy_index,
                "set_encoder_score_margin_vs_vote": 0.2,
                "matched_control_candidate_id": f"{task_id}::candidate0",
                "matched_control_correct": correct_index == 0,
                "online_adapt_candidate_id": f"{task_id}::candidate1",
                "online_adapt_correct": correct_index == 1,
            }
        )

    _write_gzip_json(
        root / mod.exp4284.POOL_REL,
        {
            "schema": "fixture",
            "task_n": len(tasks),
            "candidate_n": sum(len(task["candidates"]) for task in tasks),
            "tasks": tasks,
        },
    )
    _write_json(root / mod.exp4284.MANIFEST_REL, {"schema": "fixture", "rows": manifest_rows})
    _write_json(
        root / mod.exp4284.CROSS_FAMILY_REL,
        {
            "honest_verdict": "complete: fixture",
            "held_out_task_n": len(tasks),
            "pass_rates": {"set_encoder_at_1": 2 / 3},
            "task_rows": task_rows,
            "verifier_is_oracle": False,
        },
    )
    model_path = root / "results" / "fixture_set_encoder.json"
    _write_json(
        model_path,
        {
            "model": {
                "model_type": "constant_set_score",
                "constant_score": 0.5,
                "feature_names": [],
                "feature_means": [],
                "feature_scales": [],
                "hidden_dim": 1,
                "temperature": 1.0,
            },
            "model_type": "constant_set_score",
            "feature_names": [],
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / mod.exp4284.SET_ENCODER_BUILD_REL,
        {
            "aggregator_trained": True,
            "learned_verifier_path": str(model_path),
            "verifier_is_oracle": False,
            "model_specs": {"architecture": "fixture"},
        },
    )
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "models" / "qwen.gguf").write_bytes(b"qwen gguf")
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "adversarial_verify.py").write_text(
        "import json, sys\nprint(json.dumps({'flag_count': 0, 'artifact': sys.argv[-1]}))\n",
        encoding="utf-8",
    )
    return root


def _spec(hf_id: str, path: Path, active_params_b: float) -> dict[str, Any]:
    return {
        "name": hf_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
        "hf_id": hf_id,
        "model_path": str(path),
        "active_params_b": active_params_b,
    }


def test_req_4303_spec_declares_isoflops_contract() -> None:
    """REQ-VERIFY-4303: OpenSpec declares the iso-FLOPs strong-judge contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4303",
        "SCENARIO-VERIFY-4303",
        "python/carnot/reporting/verifier_efficiency_parity_isoflops_4303.py",
        "results/experiment_4303_verifier_efficiency_parity_isoflops.py",
        "results/experiment_4303_verifier_efficiency_parity_isoflops.json",
        "blocked_judge_models_not_cached",
        "blocked_window_exceeded",
        "efficiency_pareto_holds",
        "accuracy_energy_verifier",
        "accuracy_best_judge",
        "accuracy_delta_ci95",
        "cost_ratio",
        "iso_flops_curve",
        "verifier_is_oracle=false",
        mod.QWEN_JUDGE_ID,
        mod.GEMMA_JUDGE_ID,
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4303_accepts_one_checkpointed_strong_judge(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4303: one completed judge is enough and emits iso-FLOPs."""

    root = _make_repo(tmp_path)
    (root / mod.CHECKPOINT_REL).write_text("[]\n", encoding="utf-8")
    qwen_spec = _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)
    judge = FakeJudge([99, 99, 99], latency_s=2.0)

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec],
        llama_import_checker=lambda: True,
        judge_factory=lambda spec: judge,
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        min_tasks=3,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["accuracy_energy_verifier"] == pytest.approx(2 / 3)
    assert artifact["accuracy_best_judge"] == pytest.approx(0.0)
    assert artifact["efficiency_pareto_holds"] is True
    assert artifact["cost_ratio"] <= 0.1
    assert artifact["selection_task_n"] == 3
    assert artifact["best_judge_id"] == mod.QWEN_JUDGE_ID
    assert artifact["skipped_judge_ids"] == [mod.GEMMA_JUDGE_ID]
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert {point["arm"] for point in artifact["iso_flops_curve"]["points"]} == {
        "energy_verifier",
        mod.QWEN_JUDGE_ID,
    }
    assert artifact["iso_flops_curve"]["best_judge_id"] == mod.QWEN_JUDGE_ID
    assert artifact["model_specs"]["strong_prompt"]["version"] == mod.PROMPT_VERSION
    assert artifact["model_specs"]["cost_accounting_method"]
    assert all(row["judge_outputs"][mod.QWEN_JUDGE_ID]["judge_correct"] is False for row in artifact["per_task"])

    checkpoint = json.loads((root / mod.CHECKPOINT_REL).read_text(encoding="utf-8"))
    assert len(checkpoint["judges"][mod.QWEN_JUDGE_ID]["selections"]) == 3
    written = json.loads((root / mod.OUTPUT_REL).read_text(encoding="utf-8"))
    assert written == artifact


def test_scenario_4303_resumes_checkpoint_without_rejudging(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4303: completed judge calls are reused from checkpoint."""

    root = _make_repo(tmp_path)
    qwen_spec = _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)
    first_judge = FakeJudge([0, 0, 0], latency_s=2.0)
    mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec],
        llama_import_checker=lambda: True,
        judge_factory=lambda spec: first_judge,
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        min_tasks=3,
        adversarial_runner=lambda path: {"returncode": 0, "artifact": str(path)},
    )
    second_judge = FakeJudge([], latency_s=2.0)

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec],
        llama_import_checker=lambda: True,
        judge_factory=lambda spec: second_judge,
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        min_tasks=3,
        adversarial_runner=lambda path: {"returncode": 0, "artifact": str(path)},
    )

    assert second_judge.calls == 0
    assert artifact["selection_task_n"] == 3
    assert artifact["judge_metrics"][0]["checkpoint_resumed"] is True


def test_req_4303_blocks_without_cached_judges_before_llama_import(tmp_path: Path) -> None:
    """REQ-VERIFY-4303: missing GGUFs write blocked artifact without judge inference."""

    root = _make_repo(tmp_path)
    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [],
        llama_import_checker=lambda: pytest.fail("llama import check must not run without GGUF"),
        judge_factory=lambda _spec: pytest.fail("judge must not load without GGUF"),
        trm_stand_down_checker=lambda _root: (False, "active TRM training process observed"),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_judge_models_not_cached"
    assert artifact["efficiency_pareto_holds"] is False
    assert artifact["accuracy_energy_verifier"] == 0.0
    assert artifact["accuracy_best_judge"] == 0.0
    assert artifact["iso_flops_curve"]["points"] == []
    assert artifact["model_specs"]["status"] == "blocked"
    assert artifact["adversarial_verify"]["status"] == "not_run"
    assert any(row["resource"] == "cross_family_candidates" and row["available"] for row in artifact["preconditions_checked"])
    assert any(row["resource"] == "energy_verifier" and row["available"] for row in artifact["preconditions_checked"])
    assert any(row["resource"] == "trm_training_stand_down" and not row["available"] for row in artifact["preconditions_checked"])


def test_req_4303_blocks_when_llama_cpp_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-4303: llama.cpp is checked only after a real GGUF is available."""

    root = _make_repo(tmp_path)
    qwen_spec = _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec],
        llama_import_checker=lambda: False,
        judge_factory=lambda _spec: pytest.fail("judge must not load when llama_cpp is unavailable"),
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )

    assert artifact["honest_verdict"] == "blocked_llama_cpp_unavailable"
    assert any(row["resource"] == "llama_cpp" and not row["available"] for row in artifact["preconditions_checked"])


def test_req_4303_blocks_runtime_failures_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4303: judge runtime failures become honest blocked artifacts."""

    root = _make_repo(tmp_path)
    qwen_spec = _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec],
        llama_import_checker=lambda: True,
        judge_factory=lambda _spec: (_ for _ in ()).throw(RuntimeError("load failed")),
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )

    assert artifact["honest_verdict"] == "blocked_llm_judge_runtime"
    assert any(row["resource"].startswith("llm_judge_runtime:") for row in artifact["preconditions_checked"])


def test_req_4303_blocks_if_window_expires_before_first_judge(tmp_path: Path) -> None:
    """REQ-VERIFY-4303: no completed judge before the window gives blocked_window_exceeded."""

    root = _make_repo(tmp_path)
    qwen_spec = _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec],
        llama_import_checker=lambda: True,
        judge_factory=lambda _spec: pytest.fail("judge must not load after expired window"),
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
        window_s=0.000000000001,
    )

    assert artifact["honest_verdict"] == "blocked_window_exceeded"
    assert artifact["acceptance_gate"] is True


def test_req_4303_blocks_if_first_judge_expires_mid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-4303: first-judge mid-call timeout is blocked, not fabricated."""

    root = _make_repo(tmp_path)
    qwen_spec = _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)
    clock_values = itertools.chain([0.0] * 4, itertools.repeat(999.0))
    monkeypatch.setattr(mod.time, "perf_counter", lambda: next(clock_values))

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec],
        llama_import_checker=lambda: True,
        judge_factory=lambda _spec: FakeJudge([]),
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
        window_s=1.0,
    )

    assert artifact["honest_verdict"] == "blocked_window_exceeded"
    assert artifact["selection_task_n"] == 0


def test_scenario_4303_keeps_first_judge_when_second_hits_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-4303: a finished first judge is complete even if the second times out."""

    root = _make_repo(tmp_path)
    qwen_spec = _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)
    gemma_path = root / "models" / "gemma.gguf"
    gemma_path.write_bytes(b"gemma gguf")
    gemma_spec = _spec(mod.GEMMA_JUDGE_ID, gemma_path, 31.0)
    qwen_judge = FakeJudge([0, 0, 0], latency_s=2.0)
    gemma_judge = FakeJudge([], latency_s=2.0)
    clock_values = itertools.chain([0.0] * 7, itertools.repeat(999.0))
    monkeypatch.setattr(mod.time, "perf_counter", lambda: next(clock_values))

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec, gemma_spec],
        llama_import_checker=lambda: True,
        judge_factory=lambda spec: qwen_judge if spec["hf_id"] == mod.QWEN_JUDGE_ID else gemma_judge,
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        min_tasks=3,
        window_s=1.0,
        max_completed_judges=2,
        adversarial_runner=lambda path: {"returncode": 0, "artifact": str(path)},
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["best_judge_id"] == mod.QWEN_JUDGE_ID
    assert artifact["skipped_judge_ids"] == [mod.GEMMA_JUDGE_ID]
    assert gemma_judge.calls == 0


def test_scenario_4303_keeps_first_judge_when_second_expires_mid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-4303: a mid-judge timeout after one result still completes."""

    root = _make_repo(tmp_path)
    qwen_spec = _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)
    gemma_path = root / "models" / "gemma.gguf"
    gemma_path.write_bytes(b"gemma gguf")
    gemma_spec = _spec(mod.GEMMA_JUDGE_ID, gemma_path, 31.0)
    qwen_judge = FakeJudge([0, 0, 0], latency_s=2.0)
    gemma_judge = FakeJudge([], latency_s=2.0)
    clock_values = itertools.chain([0.0] * 8, itertools.repeat(999.0))
    monkeypatch.setattr(mod.time, "perf_counter", lambda: next(clock_values))

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec, gemma_spec],
        llama_import_checker=lambda: True,
        judge_factory=lambda spec: qwen_judge if spec["hf_id"] == mod.QWEN_JUDGE_ID else gemma_judge,
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        min_tasks=3,
        window_s=1.0,
        max_completed_judges=2,
        adversarial_runner=lambda path: {"returncode": 0, "artifact": str(path)},
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["skipped_judge_ids"] == [mod.GEMMA_JUDGE_ID]
    assert gemma_judge.calls == 0


def test_req_4303_blocks_missing_candidate_inputs(tmp_path: Path) -> None:
    """REQ-VERIFY-4303: candidate-load failures are recorded before inference."""

    root = tmp_path
    judge_path = root / "models" / "qwen.gguf"
    judge_path.parent.mkdir(parents=True)
    judge_path.write_bytes(b"qwen gguf")
    qwen_spec = _spec(mod.QWEN_JUDGE_ID, judge_path, 3.0)

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen_spec],
        llama_import_checker=lambda: pytest.fail("llama import check must wait for candidates"),
        judge_factory=lambda _spec: pytest.fail("judge must not load without candidates"),
        trm_stand_down_checker=lambda _root: (True, "no TRM training launched"),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )

    assert artifact["honest_verdict"] == "blocked_cross_family_candidates"


def test_run_checkpointed_judge_honors_deadline(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4303: per-task judge calls stop when the deadline expires."""

    root = _make_repo(tmp_path)
    cases, _, _, _, _ = mod.exp4284.load_selection_cases(root, min_tasks=3)

    with pytest.raises(mod.BlockedRun, match="blocked_window_exceeded"):
        mod.run_checkpointed_strong_llm_judge(
            cases,
            FakeJudge([]),
            judge_id=mod.QWEN_JUDGE_ID,
            checkpoint_path=root / mod.CHECKPOINT_REL,
            checkpoint={"judges": {}},
            deadline_monotonic=0.0,
        )


def test_run_adversarial_verify_preserves_non_json_output(tmp_path: Path) -> None:
    """REQ-VERIFY-4303: adversarial verifier output is recorded even if unparsable."""

    script = tmp_path / "scripts" / "adversarial_verify.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('not-json')\n", encoding="utf-8")

    report = mod._run_adversarial_verify(tmp_path, tmp_path / "artifact.json")

    assert report["stdout"] == "not-json\n"
    assert report["returncode"] == 0


def _manual_selection_rows(energy_hits: list[bool], judge_hits: list[bool]) -> list[dict[str, Any]]:
    rows = []
    for index, (energy_hit, judge_hit) in enumerate(zip(energy_hits, judge_hits, strict=True)):
        rows.append(
            {
                "task_id": f"T{index}",
                "family_id": f"family-{index}",
                "fold": index,
                "candidate_count": 2,
                "all_candidate_count": 3,
                "energy_candidate_id": f"T{index}::energy",
                "energy_finalist_index": 0,
                "energy_correct": energy_hit,
                "judge_id": mod.QWEN_JUDGE_ID,
                "judge_chosen_index": 1,
                "judge_candidate_id": f"T{index}::judge",
                "judge_correct": judge_hit,
                "judge_cost": {
                    "latency_s": 2.0,
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "raw_output": "Final answer: 1",
                    "parse_status": "parsed_final_answer",
                },
            }
        )
    return rows


@pytest.mark.parametrize(
    ("energy_hits", "judge_hits", "energy_dollars", "verdict_prefix"),
    [
        ([True, False, True], [True, False, True], 1e-12, "complete: parity_at_lower_cost"),
        ([False, False, False], [True, True, True], 1e-12, "complete: stronger_judge_closes_accuracy_gap"),
        ([True, True, True], [False, False, False], 1.0, "complete: no_cost_advantage"),
    ],
)
def test_complete_artifact_verdict_variants(
    tmp_path: Path,
    energy_hits: list[bool],
    judge_hits: list[bool],
    energy_dollars: float,
    verdict_prefix: str,
) -> None:
    """REQ-VERIFY-4303: every complete verdict is terminal-prefixed and valid."""

    artifact = mod._complete_artifact(
        judge_results=[
            {
                "judge_id": mod.QWEN_JUDGE_ID,
                "judge_spec": {"hf_id": mod.QWEN_JUDGE_ID, "model_path": "fixture.gguf", "active_params_b": 3.0},
                "selections": _manual_selection_rows(energy_hits, judge_hits),
                "judge_cost": {
                    "total_wall_clock_s": 6.0,
                    "total_tokens": 330,
                    "prompt_tokens": 300,
                    "completion_tokens": 30,
                },
            }
        ],
        energy_cost={
            "total_wall_clock_s": 0.1,
            "candidate_forward_passes": 9,
            "flops_proxy": 9.0,
            "score_checksum_component": 0.0,
            "estimated_dollars_per_1k_selections": energy_dollars,
        },
        checksums={"fixture": "sha256"},
        model_path=tmp_path / "model.json",
        build={"model_specs": {"architecture": "fixture"}},
        preconditions=[],
        skipped_judge_ids=[mod.GEMMA_JUDGE_ID],
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        duration_s=1.0,
        checkpoint_path=tmp_path / mod.CHECKPOINT_REL,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(verdict_prefix)


def _valid_blocked_artifact(tmp_path: Path) -> dict[str, Any]:
    return mod._blocked_artifact(
        "blocked_judge_models_not_cached",
        [{"resource": "fixture", "available": False, "detail": "fixture"}],
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
        checkpoint_path=tmp_path / mod.CHECKPOINT_REL,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda artifact: artifact.pop("cost_ratio"),
        lambda artifact: artifact.update({"honest_verdict": "not-terminal"}),
        lambda artifact: artifact.update({"efficiency_pareto_holds": 0}),
        lambda artifact: artifact.update({"accuracy_energy_verifier": 0}),
        lambda artifact: artifact.update({"verifier_is_oracle": True}),
        lambda artifact: artifact.update({"random_seed": 4303.0}),
        lambda artifact: artifact.update({"accuracy_delta_ci95": [0.0]}),
        lambda artifact: artifact.update({"iso_flops_curve": {"points": None}}),
        lambda artifact: artifact.update({"preconditions_checked": {}}),
        lambda artifact: artifact.update({"model_specs": []}),
        lambda artifact: artifact.update({"reproducibility_checksum": "short"}),
        lambda artifact: artifact.update({"field_principles": {}}),
        lambda artifact: artifact.update({"spec_refs": []}),
    ],
)
def test_validate_artifact_rejects_malformed_outputs(tmp_path: Path, mutate: Any) -> None:
    """REQ-VERIFY-4303: required bare fields are mechanically validated."""

    artifact = _valid_blocked_artifact(tmp_path)
    mutate(artifact)

    with pytest.raises(ValueError):
        mod.validate_artifact(artifact)
