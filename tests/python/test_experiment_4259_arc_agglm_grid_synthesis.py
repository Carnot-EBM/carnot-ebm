"""Tests for Exp 4259 ARC set-encoder grid synthesis.

Spec refs: REQ-VERIFY-4259, SCENARIO-VERIFY-4259.
"""

from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_agglm_grid_synthesis_4259 as mod
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adversarial_clean(_path: Path) -> dict:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def _features(vote: float) -> dict[str, float]:
    return {name: 0.0 for name in exp4244.FEATURE_NAMES} | {
        "vote_weight": float(vote),
        "self_consistency_margin": float(vote) - 0.5,
        "grid_height": 2.0,
        "grid_width": 2.0,
        "grid_cells": 4.0,
        "set_candidate_count": 3.0,
        "set_vote_max": float(vote),
        "set_vote_mean": 1.0 / 3.0,
    }


def _synthesis_candidates(target: list[list[int]]) -> list[list[list[int]]]:
    return [
        [[target[0][0], target[0][1]], [0, 0]],
        [[target[0][0], 0], [target[1][0], target[1][1]]],
        [[0, target[0][1]], [target[1][0], target[1][1]]],
    ]


def _write_synthesis_fixture(
    root: Path,
    *,
    hardened: bool = True,
    target_count: int = 4,
) -> Path:
    task_ids = [f"gap3_stage2:task-{index}" for index in range(target_count)]
    pool_tasks = []
    source_entries = []
    programs = []
    oof_rows = []
    task_rows_4245 = []
    for task_index, task_id in enumerate(task_ids):
        raw_task_id = task_id.split(":", 1)[1]
        target = [[task_index + 1, task_index + 2], [task_index + 3, task_index + 4]]
        grids = _synthesis_candidates(target)
        candidates = []
        source_candidates = []
        for candidate_index, grid in enumerate(grids):
            vote = 9.0 if candidate_index == 0 else 1.0
            score = [0.9, 0.8, 0.7][candidate_index]
            candidate_id = f"{task_id}::candidate{candidate_index}"
            candidates.append(
                {
                    "candidate_grid_hash": mod.grid_hash(grid),
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "features": _features(vote),
                    "grid": grid,
                    "is_correct": False,
                    "q_mean": 0.1,
                    "raw_candidate_indices": [candidate_index],
                    "source_kinds": ["pool_candidate"],
                    "votes": vote,
                }
            )
            source_candidates.append({"correct": False, "grid": grid, "q_mean": 0.1, "votes": vote})
            oof_rows.append(
                {
                    "candidate_id": candidate_id,
                    "correct": False,
                    "fold": task_index % 2,
                    "score": score,
                    "task_id": task_id,
                    "train_task_ids": [other for other in task_ids if other != task_id],
                }
            )
        pool_tasks.append(
            {
                "candidate_count": len(candidates),
                "candidates": candidates,
                "oracle_present": False,
                "raw_task_id": raw_task_id,
                "source_id": "gap3_stage2",
                "task_id": task_id,
                "vote_top_candidate_id": candidates[0]["candidate_id"],
                "wrong_majority": False,
            }
        )
        source_entries.append(
            {
                "candidates": source_candidates,
                "demos": [],
                "task": raw_task_id,
                "test_input": [[0, 0], [0, 0]],
            }
        )
        programs.append({"demo_fit": 1.0, "entry_i": task_index, "pred_grid": target, "task": raw_task_id})
        task_rows_4245.append(
            {
                "oracle_hit": False,
                "selector_only_candidate_id": candidates[0]["candidate_id"],
                "set_encoder_candidate_id": candidates[0]["candidate_id"],
                "set_encoder_correct": False,
                "task_id": task_id,
                "vote_candidate_id": candidates[0]["candidate_id"],
                "vote_correct": False,
            }
        )

    pool_rel = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
    source_pool_rel = Path("results/arc3_gap3_stage2_eval_pool.json.gz")
    source_programs_rel = Path("results/arc3_gap4_induced_programs.json")
    for rel, payload in (
        (
            pool_rel,
            {
                "candidate_n": sum(len(task["candidates"]) for task in pool_tasks),
                "positive_candidate_n": 0,
                "random_seed": 4243,
                "reproducibility_checksum": "sha256:" + "1" * 64,
                "schema": "carnot.arc_candidate_pool_grow.v1",
                "source_paths": [str(root / source_pool_rel), str(root / source_programs_rel)],
                "source_sha256": {},
                "spec_refs": ["REQ-CAPSTONE-4243"],
                "task_n": len(pool_tasks),
                "tasks": pool_tasks,
                "wrong_majority_n": 0,
            },
        ),
        (
            source_pool_rel,
            {
                "entries": source_entries,
                "experiment": "fixture",
                "n_candidates": sum(len(entry["candidates"]) for entry in source_entries),
                "n_entries": len(source_entries),
            },
        ),
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    _write_json(root / source_programs_rel, {"experiment": "fixture", "histories": [], "programs": programs})
    _write_json(
        root / "results" / "arc3_gap4_arc2_induced_programs.json",
        {"experiment": "fixture", "histories": [], "programs": []},
    )
    arc2_pool = root / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    with gzip.open(arc2_pool, "wt", encoding="utf-8") as handle:
        json.dump({"entries": [], "experiment": "fixture"}, handle)

    _write_json(
        root / "results" / "experiment_4243_arc_candidate_pool_grow.json",
        {
            "arc_pool_grown": True,
            "held_out_task_n": len(pool_tasks),
            "pool_artifact_path": str(pool_rel),
            "positive_candidate_n": 0,
            "random_seed": 4243,
            "reproducibility_checksum": "sha256:" + "2" * 64,
            "verifier_is_oracle": False,
            "wrong_majority_n": 0,
        },
    )
    model_path = root / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
    _write_json(
        model_path,
        {
            "feature_names": list(exp4244.FEATURE_NAMES),
            "held_out_task_n": len(pool_tasks),
            "model_specs": {"architecture": "fixture_set_encoder"},
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "3" * 64,
            "set_encoder_oof": {"fold_task_ids": [task_ids], "rows": oof_rows},
            "spec_refs": ["REQ-VERIFY-4244"],
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4244_arc_set_encoder_aggregator_build.json",
        {
            "aggregator_trained": True,
            "held_out_task_n": len(pool_tasks),
            "honest_verdict": "complete: fixture",
            "learned_verifier_path": str(model_path),
            "model_specs": {"architecture": "fixture_set_encoder"},
            "random_seed": 4244,
            "reproducibility_checksum": "sha256:" + "4" * 64,
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4245_arc_set_encoder_beats_vote.json",
        {
            "honest_verdict": "complete: arc_oracle_distinct_set_encoder_beats_vote",
            "oracle_at_k": 0.0,
            "pass_rates": {"set_encoder_at_1": 0.0, "vote_at_1": 0.0},
            "set_encoder_minus_vote_ci95": [0.1, 0.2],
            "set_encoder_minus_vote_delta": 0.1,
            "task_rows": task_rows_4245,
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "experiment_4256_arc_oracle_distinct_leak_audit.json",
        {
            "honest_verdict": "complete: fixture",
            "verifier_is_oracle": False,
            "win_survives_provenance_blind": hardened,
        },
    )
    _write_json(
        root / "results" / "experiment_4257_arc_oracle_distinct_multiseed_replication.json",
        {
            "honest_verdict": "complete: fixture",
            "oracle_distinct_win_replicates": hardened,
            "verifier_is_oracle": False,
        },
    )
    return root


def test_req_4259_spec_declares_grid_synthesis_contract() -> None:
    """REQ-VERIFY-4259: OpenSpec declares the synthesis gate fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4259",
        "SCENARIO-VERIFY-4259",
        "python/carnot/reporting/arc_agglm_grid_synthesis_4259.py",
        "results/experiment_4259_arc_agglm_grid_synthesis.py",
        "results/experiment_4259_arc_agglm_grid_synthesis.json",
        "complete_arc_synthesis_deferred_win_not_hardened",
        "synthesis_beats_selection",
        "synthesis_breaks_oracle_ceiling",
        "synthesis_minus_vote_delta",
        "synthesis_minus_oracle_delta",
        "exact_match_validated=true",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4259_synthesis_breaks_oracle_with_exact_grid_match(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4259: per-cell synthesis can exceed every candidate."""

    _write_synthesis_fixture(tmp_path)

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: arc_synthesis_breaks_oracle_ceiling"
    assert artifact["synthesis_beats_selection"] is True
    assert artifact["synthesis_breaks_oracle_ceiling"] is True
    assert artifact["exact_match_validated"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["held_out_task_n"] == 4
    assert artifact["pass_rates"]["synthesis_at_1"] == pytest.approx(1.0)
    assert artifact["pass_rates"]["vote_at_1"] == pytest.approx(0.0)
    assert artifact["pass_rates"]["selector_only_at_1"] == pytest.approx(0.0)
    assert artifact["pass_rates"]["no_synthesis_baseline_at_1"] == pytest.approx(0.0)
    assert artifact["oracle_at_k"] == pytest.approx(0.0)
    assert artifact["synthesis_minus_vote_delta"] == pytest.approx(1.0)
    assert artifact["synthesis_minus_oracle_delta"] == pytest.approx(1.0)
    assert artifact["synthesis_minus_selection_ci95"][0] > 0.0
    assert artifact["synthesis_minus_oracle_ci95"][0] > 0.0
    assert artifact["ceiling_break_task_n"] == 4
    assert artifact["target_grid_coverage_n"] == 4
    assert artifact["task_rows"][0]["synthesized_grid"] not in artifact["task_rows"][0]["top_k_candidate_grids"]
    assert artifact["adversarial_verify"]["status"] == "clean"


def test_scenario_4259_deferred_when_hardened_gates_fail(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4259: unhardened wins defer before synthesis scoring."""

    _write_synthesis_fixture(tmp_path, hardened=False)

    artifact = mod.run(tmp_path, adversarial_runner=_adversarial_clean)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.DEFERRED_VERDICT
    assert artifact["synthesis_beats_selection"] is False
    assert artifact["synthesis_breaks_oracle_ceiling"] is False
    assert artifact["exact_match_validated"] is True
    assert artifact["held_out_task_n"] == 0
    assert artifact["task_rows"] == []
    assert artifact["acceptance_gate"] is True

    replication_root = tmp_path / "replication-false"
    _write_synthesis_fixture(replication_root)
    _write_json(
        replication_root / "results" / "experiment_4257_arc_oracle_distinct_multiseed_replication.json",
        {
            "honest_verdict": "complete: fixture",
            "oracle_distinct_win_replicates": False,
            "verifier_is_oracle": False,
        },
    )
    assert mod.run(replication_root, adversarial_runner=_adversarial_clean)["honest_verdict"] == (
        mod.DEFERRED_VERDICT
    )

    oracle_root = tmp_path / "oracle-precondition"
    _write_synthesis_fixture(oracle_root)
    _write_json(
        oracle_root / "results" / "experiment_4256_arc_oracle_distinct_leak_audit.json",
        {
            "honest_verdict": "complete: fixture",
            "verifier_is_oracle": True,
            "win_survives_provenance_blind": True,
        },
    )
    assert mod.run(oracle_root, adversarial_runner=_adversarial_clean)["honest_verdict"] == (
        mod.DEFERRED_VERDICT
    )


def _manual_pool(root: Path, tasks: list[mod.SynthesisTask]) -> mod.SynthesisPool:
    pool_path = root / "pool.json.gz"
    model_path = root / "model.json"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump({"tasks": []}, handle)
    _write_json(model_path, {"model": "fixture"})
    return mod.SynthesisPool(
        tasks=tasks,
        candidate_pool_path=pool_path,
        candidate_pool_sha256="pool-sha",
        learned_verifier_path=model_path,
        learned_verifier_sha256="model-sha",
        target_source_paths=(),
        target_source_sha256={},
        model_specs={"architecture": "fixture"},
        dropped_task_n=0,
        dropped_candidate_n=0,
        score_source="fixture_scores",
    )


def _task_with_candidates(
    task_id: str,
    target: list[list[int]],
    candidates: list[mod.ScoredGridCandidate],
) -> mod.SynthesisTask:
    return mod.SynthesisTask(
        task_id=task_id,
        raw_task_id=task_id,
        source_id="fixture",
        candidates=candidates,
        target_hashes=frozenset({mod.grid_hash(target)}),
    )


def test_validation_edges_and_synthesis_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-4259: schema and deterministic synthesis helpers are explicit."""

    base = mod._deferred_artifact(
        mod.DEFERRED_VERDICT,
        random_seed=mod.RANDOM_SEED,
        checksum="sha256:abc",
        duration_s=0.1,
    )
    invalid_cases = [
        ({key: value for key, value in base.items() if key != "synthesis_minus_vote_delta"}, "missing required"),
        ({**base, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**base, "synthesis_beats_selection": {"value": False}}, "bare bool"),
        ({**base, "synthesis_minus_oracle_delta": None}, "bare float"),
        ({**base, "exact_match_validated": False}, "exact_match_validated"),
        ({**base, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**base, "random_seed": 4259.0}, "bare int"),
        ({**base, "field_principles": {}}, "field_principles"),
        ({**base, "spec_refs": []}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    family = [
        mod.ScoredGridCandidate("t", "c0", 0, 0.9, 0.9, False, [[1, 2], [0, 0]]),
        mod.ScoredGridCandidate("t", "c1", 1, 0.8, 0.1, False, [[1, 0], [3, 4]]),
        mod.ScoredGridCandidate("t", "c2", 2, 0.7, 0.0, False, [[0, 2], [3, 4]]),
    ]
    assert mod.synthesize_grid(family) == [[1, 2], [3, 4]]
    assert mod.synthesize_grid([]) == []
    assert mod.synthesize_grid(family, top_k=2) == [[1, 2], [0, 0]]
    uniform_family = [
        mod.ScoredGridCandidate("t", "a", 0, 0.0, 0.0, False, [[2]]),
        mod.ScoredGridCandidate("t", "b", 1, 0.0, 0.0, False, [[1]]),
    ]
    assert mod.synthesize_grid(uniform_family) == [[1]]
    assert mod._bootstrap_ci95([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert mod._bootstrap_ci95([1.0, -1.0], random_seed=1, resamples=0) == [0.0, 0.0]
    assert mod._ci_excludes_zero([0.1, 0.2]) is True
    assert mod._ci_excludes_zero([-0.1, 0.2]) is False
    assert mod._clean_adversarial_report({"reports": [{"flags": []}]})["status"] == "clean"
    flagged = mod._clean_adversarial_report(
        {"reports": [None, {"flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM"}]}], "returncode": 1}
    )
    assert flagged["status"] == "flagged"
    assert flagged["circular_moat_overclaim_clean"] is False
    assert mod._safe_float(True) == 0.0
    assert mod._safe_float("bad") == 0.0
    assert mod._safe_float(float("nan")) == 0.0
    assert mod._safe_int(False, 7) == 7
    assert mod._safe_int("bad", 7) == 7
    assert mod._grid("bad") is None
    assert mod._grid([1]) is None
    assert mod._grid([[True]]) is None
    assert mod._grid([[1], [1, 2]]) is None
    assert mod._grid([]) is None
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json_object(list_json)
    with pytest.raises(mod.BlockedRun, match=mod.MISSING_INPUT_VERDICT):
        mod._resolve_pool_path(tmp_path / "no-pool")
    fallback_pool = tmp_path / "fallback" / mod.POOL_REL
    fallback_pool.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(fallback_pool, "wt", encoding="utf-8") as handle:
        json.dump({"tasks": []}, handle)
    assert mod._resolve_pool_path(tmp_path / "fallback") == fallback_pool
    bad_pool = tmp_path / "bad-pool.json.gz"
    with gzip.open(bad_pool, "wt", encoding="utf-8") as handle:
        json.dump([], handle)
    with pytest.raises(mod.BlockedRun, match=mod.MISSING_INPUT_VERDICT):
        mod._load_pool_payload(bad_pool)
    assert mod._oof_score_map({"set_encoder_oof": {"rows": {}}}) == {}
    assert mod._oof_score_map(
        {"set_encoder_oof": {"rows": [None, {"candidate_id": 1}, {"candidate_id": "c", "task_id": "t"}]}}
    ) == {"c": (0.0, True, 0)}
    assert mod.run(tmp_path / "missing-inputs", adversarial_runner=_adversarial_clean)["honest_verdict"] == (
        mod.MISSING_INPUT_VERDICT
    )

    target = [[9, 9], [9, 9]]
    wrong_family = _synthesis_candidates(target)
    beats_task = _task_with_candidates(
        "beats",
        target,
        [
            mod.ScoredGridCandidate("beats", "c0", 0, 0.9, 0.9, False, wrong_family[0]),
            mod.ScoredGridCandidate("beats", "c1", 1, 0.8, 0.1, False, wrong_family[1]),
            mod.ScoredGridCandidate("beats", "c2", 2, 0.7, 0.0, False, wrong_family[2]),
            mod.ScoredGridCandidate("beats", "c3", 3, 0.01, 0.0, True, target),
        ],
    )
    beats = mod.measure_synthesis(
        _manual_pool(tmp_path / "manual-beats", [beats_task] * 4),
        repo_root=tmp_path / "manual-beats",
        top_k=3,
        random_seed=1,
        bootstrap_resamples=50,
    )
    assert beats["headline_outcome"] == "arc_synthesis_beats_selection"
    match_task = _task_with_candidates(
        "match",
        [[1]],
        [
            mod.ScoredGridCandidate("match", "c0", 0, 0.9, 1.0, True, [[1]]),
            mod.ScoredGridCandidate("match", "c1", 1, 0.1, 0.0, False, [[0]]),
        ],
    )
    matched = mod.measure_synthesis(
        _manual_pool(tmp_path / "manual-match", [match_task]),
        repo_root=tmp_path / "manual-match",
        top_k=1,
        random_seed=1,
        bootstrap_resamples=10,
    )
    assert matched["headline_outcome"] == "arc_synthesis_matches_selection_no_gain"
    under_task = _task_with_candidates(
        "under",
        [[1]],
        [
            mod.ScoredGridCandidate("under", "c0", 0, 0.9, 1.0, True, [[1]]),
            mod.ScoredGridCandidate("under", "c1", 1, 0.8, 0.0, False, [[0]]),
            mod.ScoredGridCandidate("under", "c2", 2, 0.7, 0.0, False, [[0]]),
        ],
    )
    under = mod.measure_synthesis(
        _manual_pool(tmp_path / "manual-under", [under_task]),
        repo_root=tmp_path / "manual-under",
        top_k=3,
        random_seed=1,
        bootstrap_resamples=10,
    )
    assert under["headline_outcome"] == "arc_synthesis_underperforms_selection"


def test_module_does_not_use_execution_or_candidate_correctness_for_ranking() -> None:
    """REQ-VERIFY-4259: synthesis remains oracle-distinct and score-guided."""

    source = inspect.getsource(mod)
    assert "arc_gap4_execution_verifier" not in source
    assert "Gap4ExecutionVerifier" not in source
    assert "extract_dsl_rules" not in source
    assert "apply_rule" not in source
    assert "get_consistency_energy" not in source
    assert "key=lambda candidate: (candidate.correct" not in source
    assert ".correct," not in inspect.getsource(mod.select_score_candidate)
