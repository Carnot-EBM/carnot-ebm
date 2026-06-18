"""Tests for Exp 4386 cross-domain detector generalization.

Spec refs: REQ-VERIFY-4386, SCENARIO-VERIFY-4386.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import pytest

from carnot import experiment_4386_cross_domain_detection_generalization as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_gzip_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_req_verify_4386_spec_declares_cross_domain_detection_contract() -> None:
    """REQ-VERIFY-4386: OpenSpec declares fields, controls, and blocked verdict."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4386",
        "SCENARIO-VERIFY-4386",
        "experiment_4386_cross_domain_detection_generalization.json",
        "detector_generalizes_cross_domain",
        "detection_by_domain",
        "domains_at_chance",
        "blocked_no_non_fover_cached_pool",
    ):
        assert marker in spec


def test_req_verify_4386_metrics_gate_and_random_control() -> None:
    """REQ-VERIFY-4386: AUROC CI lower bound is the bare generalization gate."""

    labels = [1] * 20 + [0] * 20
    scores = [0.9] * 20 + [0.1] * 20
    ci95 = mod.bootstrap_auroc_ci95(labels, scores, seed=4386, resamples=120)
    control = mod.random_score_auroc_control(labels, seed=4386, replicates=64)

    assert mod.compute_auroc(labels, scores) == pytest.approx(1.0)
    assert ci95 == [1.0, 1.0]
    assert mod.ci_lower_beats_chance(ci95) is True
    assert control["replicates"] == 64
    assert 0.35 <= control["auroc"] <= 0.65


def test_scenario_verify_4386_loads_arc_oof_scores_and_valid_restricted_control(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4386: ARC cached rows expose labels, scores, and valid-grid control."""

    model_path = tmp_path / "arc_model.json"
    pool_path = tmp_path / "arc_pool.json.gz"
    _write_json(
        model_path,
        {
            "verifier_is_oracle": False,
            "set_encoder_oof": {
                "rows": [
                    {"task_id": "t1", "candidate_id": "c0", "correct": True, "score": 0.9},
                    {"task_id": "t1", "candidate_id": "c1", "correct": False, "score": 0.1},
                    {"task_id": "t1", "candidate_id": "c2", "correct": False, "score": 0.8},
                    {"task_id": "t2", "candidate_id": "c3", "correct": True, "score": 0.7},
                ]
            },
            "model_specs": {"architecture": "test_set_encoder"},
        },
    )
    _write_gzip_json(
        pool_path,
        {
            "tasks": [
                {
                    "task_id": "t1",
                    "candidates": [
                        {"candidate_id": "c0", "is_correct": True, "grid": [[1]]},
                        {"candidate_id": "c1", "is_correct": False, "grid": [[2]]},
                        {"candidate_id": "c2", "is_correct": False, "grid": []},
                    ],
                },
                {
                    "task_id": "t2",
                    "candidates": [
                        {"candidate_id": "c3", "is_correct": True, "grid": [[3]]},
                    ],
                },
            ]
        },
    )

    rows = mod.load_arc_set_encoder_rows(model_path, pool_path)
    summary = mod.summarize_domain(
        "gap4_arc",
        rows,
        selection_headroom=0.129,
        seed=4386,
        bootstrap_resamples=120,
        random_control_replicates=16,
    )

    assert len(rows) == 4
    assert summary["domain"] == "gap4_arc"
    assert summary["detection_auroc"] == pytest.approx(0.75)
    assert summary["valid_but_wrong_restricted_auroc"] == pytest.approx(1.0)
    assert summary["valid_but_wrong_restricted_n"] == 3
    assert summary["selection_headroom"] == 0.129
    assert summary["base_rate"] == 0.5


def test_scenario_verify_4386_artifact_has_required_bare_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4386: complete artifact exposes capstone fields bare."""

    source = tmp_path / "pool.json"
    _write_json(source, {"rows": 4})
    domain = mod.summarize_domain(
        "gap4_arc",
        [
            mod.ScoredCandidate("gap4_arc", "t1", "c0", True, 0.9, True),
            mod.ScoredCandidate("gap4_arc", "t1", "c1", False, 0.1, True),
            mod.ScoredCandidate("gap4_arc", "t2", "c2", True, 0.8, True),
            mod.ScoredCandidate("gap4_arc", "t2", "c3", False, 0.2, True),
        ],
        selection_headroom=0.129,
        seed=4386,
        bootstrap_resamples=120,
        random_control_replicates=16,
    )

    artifact = mod.build_complete_artifact(
        domain_results=[domain],
        unavailable_domains=[{"domain": "gsm8k", "reason": "no_candidate_tasks"}],
        preconditions_checked=[
            mod.PreconditionCheck("verifier_registry", True, "loaded").as_dict()
        ],
        source_paths=[source],
        duration_s=1.25,
        bootstrap_resamples=120,
        random_control_replicates=16,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["detector_generalizes_cross_domain"] is True
    assert artifact["detection_by_domain"][0]["domain"] == "gap4_arc"
    assert artifact["domains_at_chance"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["trm_training"] == "stood_down_not_invoked"
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4386_blocked_artifact_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4386: missing non-FoVer pools block without positive metrics."""

    artifact = mod.build_blocked_artifact(
        preconditions_checked=[
            mod.PreconditionCheck("non_fover_cached_scored_pool", False, "none").as_dict()
        ],
        source_paths=[tmp_path / "missing.json"],
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "blocked_no_non_fover_cached_pool"
    assert artifact["detector_generalizes_cross_domain"] is False
    assert artifact["detection_by_domain"] == []
    assert artifact["domains_at_chance"] == []
    assert artifact["verifier_is_oracle"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_4386_run_experiment_writes_and_logs_chance_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4386: chance domains are surfaced as missing-verifier gaps."""

    registry = tmp_path / "ops" / "verifier_registry.yaml"
    fover = tmp_path / "results" / "experiment_4375.json"
    detector = tmp_path / "results" / "experiment_4381.json"
    headroom = tmp_path / "results" / "experiment_4175.json"
    arc_summary = tmp_path / "results" / "arc3_trm_verifier_rerank.json"
    artifact_path = tmp_path / "results" / "experiment_4386.json"
    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")
    for path, payload in (
        (fover, {"detector_auroc": 0.918304}),
        (detector, {"model_specs": {"fusion_method": "mean_l2r_r2l"}}),
        (
            headroom,
            {
                "per_domain_headroom": {
                    "code": {"selectable_headroom": 0.18},
                    "math": {"selectable_headroom": 0.0},
                    "sudoku": {"selectable_headroom": 0.129},
                }
            },
        ),
        (arc_summary, {"oracle_ceiling": {"pass@2": 0.6129}, "trm_vote_pass2": 0.4839}),
    ):
        _write_json(path, payload)

    chance_rows = [
        mod.ScoredCandidate("code_humaneval_mbpp", f"t{idx}", f"c{idx}", idx % 2 == 0, 0.5, True)
        for idx in range(40)
    ]

    monkeypatch.setattr(
        mod,
        "load_available_domain_rows",
        lambda _cfg: (
            {"code_humaneval_mbpp": chance_rows},
            [{"domain": "gsm8k", "reason": "no_candidate_tasks"}],
            [registry],
        ),
    )

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            registry_path=registry,
            fover_baseline_path=fover,
            detector_config_path=detector,
            headroom_census_path=headroom,
            arc_rerank_path=arc_summary,
            artifact_path=artifact_path,
            verifier_gaps_path=gaps_path,
            min_domain_candidates=2,
            bootstrap_resamples=120,
            random_control_replicates=16,
            started_at=1.0,
            clock=lambda: 3.0,
        ),
        adversarial_verify_runner=lambda _path: {"returncode": 0, "flags": []},
        write=True,
    )

    assert (
        artifact["honest_verdict"] == "complete: detector_fover_bound_non_fover_domains_at_chance"
    )
    assert artifact["detector_generalizes_cross_domain"] is False
    assert artifact["domains_at_chance"] == ["code_humaneval_mbpp"]
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert "GAP-4386-CODE-HUMANEVAL-MBPP-DETECTOR-CHANCE" in gaps_path.read_text(encoding="utf-8")
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_req_verify_4386_defensive_metric_and_grid_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4386: metric and grid validators fail closed on malformed inputs."""

    assert mod.round_float(None) is None
    assert mod.round_float(float("nan")) is None
    with pytest.raises(ValueError, match="same length"):
        mod.compute_auroc([1], [0.1, 0.2])
    with pytest.raises(ValueError, match="both positive and negative"):
        mod.compute_auroc([1, 1], [0.1, 0.2])
    assert mod.bootstrap_auroc_ci95([1, 1], [0.1, 0.2], seed=1, resamples=10) == [
        None,
        None,
    ]
    assert mod.bootstrap_auroc_ci95([1, 0], [0.9, 0.1], seed=1, resamples=0) == [
        None,
        None,
    ]

    assert mod._is_valid_grid("not a grid") is False
    assert mod._is_valid_grid([[]]) is False
    assert mod._is_valid_grid([[1], [1, 2]]) is False
    assert mod._is_valid_grid([[True]]) is False
    assert mod._is_valid_grid([[10]]) is False
    assert mod._is_valid_grid([[0, 1], [2, 3]]) is True

    bad_pool = tmp_path / "bad_pool.json"
    mixed_pool = tmp_path / "mixed_pool.json"
    _write_json(bad_pool, {"tasks": "not-list"})
    _write_json(
        mixed_pool,
        {
            "tasks": [
                "bad-task",
                {
                    "candidates": [
                        "bad-candidate",
                        {"grid": [[1]]},
                        {"candidate_id": "ok", "grid": [[1]]},
                    ]
                },
            ]
        },
    )
    assert mod.load_arc_valid_grid_map(bad_pool) == {}
    assert mod.load_arc_valid_grid_map(mixed_pool) == {"ok": True}


def test_req_verify_4386_loader_error_and_generic_pool_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-4386: malformed pools are rejected while generic scored rows load."""

    pool = tmp_path / "pool.json"
    _write_json(pool, {"tasks": []})
    for name, payload, message in (
        ("list_model.json", [], "JSON object"),
        ("oracle_model.json", {"verifier_is_oracle": True}, "verifier_is_oracle"),
        ("missing_rows.json", {"verifier_is_oracle": False}, "out-of-fold rows"),
        (
            "empty_rows.json",
            {"verifier_is_oracle": False, "set_encoder_oof": {"rows": ["bad", {"correct": True}]}},
            "no usable scored rows",
        ),
    ):
        path = tmp_path / name
        _write_json(path, payload)
        with pytest.raises(ValueError, match=message):
            mod.load_arc_set_encoder_rows(path, pool)

    generic = tmp_path / "generic.json"
    _write_json(
        generic,
        {
            "tasks": [
                "bad-task",
                {"task_id": "skip", "candidates": "bad"},
                {
                    "task": "t",
                    "cands": [
                        "bad-candidate",
                        {"correct": True},
                        {"correct": True, "score": 0.8},
                        {"is_correct": False, "verifier_score": 0.2, "candidate_id": "bad"},
                    ],
                },
            ]
        },
    )
    rows = mod._generic_scored_rows_from_pool("code_humaneval_mbpp", generic)
    not_tasks = tmp_path / "not_tasks.json"
    _write_json(not_tasks, {"tasks": "bad"})
    assert mod._generic_scored_rows_from_pool("code_humaneval_mbpp", not_tasks) == []
    assert [
        (row.task_id, row.candidate_id, row.is_correct, row.verifier_score) for row in rows
    ] == [
        ("t", "t::2", True, 0.8),
        ("t", "bad", False, 0.2),
    ]

    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{bad", encoding="utf-8")
    code_results = tmp_path / "code.json"
    _write_json(code_results, {"results": [{"baseline_passed": True, "repair_passed": False}]})
    assert mod._code_pool_unavailable_reason(tmp_path / "missing.json") == "missing"
    assert mod._code_pool_unavailable_reason(unreadable).startswith("unreadable:")
    assert "labeled_candidates=2" in mod._code_pool_unavailable_reason(code_results)
    assert mod._code_pool_unavailable_reason(generic) == "no_humaneval_mbpp_candidate_rows"
    gsm = tmp_path / "gsm.json"
    _write_json(gsm, {"datasets": {"control": []}})
    assert mod._gsm8k_pool_unavailable_reason(tmp_path / "missing_gsm.json") == "missing"
    assert mod._gsm8k_pool_unavailable_reason(unreadable).startswith("unreadable:")
    assert mod._gsm8k_pool_unavailable_reason(gsm) == (
        "datasets_present_but_no_multicandidate_verifier_scores"
    )
    assert mod._gsm8k_pool_unavailable_reason(generic) == "no_gsm8k_candidate_rows"

    empty_generic = tmp_path / "empty_generic.json"
    _write_json(empty_generic, {"tasks": []})
    cfg = mod.ExperimentConfig(
        arc_detector_model_path=tmp_path / "missing_arc_model.json",
        arc_candidate_pool_path=tmp_path / "missing_arc_pool.json.gz",
        code_pool_path=empty_generic,
        gsm8k_pool_path=empty_generic,
    )
    domains, unavailable, sources = mod.load_available_domain_rows(cfg)
    assert domains == {}
    assert sources == []
    assert {item["domain"] for item in unavailable} == {
        "gap4_arc",
        "code_humaneval_mbpp",
        "gsm8k",
    }


def test_req_verify_4386_real_loader_preconditions_and_headrooms(tmp_path: Path) -> None:
    """REQ-VERIFY-4386: available cached domains and preconditions summarize deterministically."""

    arc_model = tmp_path / "arc_model.json"
    arc_pool = tmp_path / "arc_pool.json.gz"
    code_pool = tmp_path / "code_pool.json"
    gsm_pool = tmp_path / "gsm_pool.json"
    registry = tmp_path / "ops" / "verifier_registry.yaml"
    fover = tmp_path / "results" / "experiment_4375.json"
    detector = tmp_path / "results" / "experiment_4381.json"
    headroom = tmp_path / "results" / "experiment_4175.json"
    arc_summary = tmp_path / "results" / "arc3.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")
    _write_json(
        arc_model,
        {
            "verifier_is_oracle": False,
            "set_encoder_oof": {
                "rows": [
                    {"task_id": "a", "candidate_id": "a0", "correct": True, "score": 0.9},
                    {"task_id": "a", "candidate_id": "a1", "correct": False, "score": 0.1},
                ]
            },
        },
    )
    _write_gzip_json(
        arc_pool,
        {
            "tasks": [
                {
                    "candidates": [
                        {"candidate_id": "a0", "grid": [[1]]},
                        {"candidate_id": "a1", "grid": [[2]]},
                    ]
                }
            ]
        },
    )
    _write_json(
        code_pool,
        {
            "tasks": [
                {
                    "task_id": "code",
                    "candidates": [
                        {"candidate_id": "code0", "is_correct": True, "verifier_score": 0.8},
                        {"candidate_id": "code1", "is_correct": False, "verifier_score": 0.2},
                    ],
                }
            ]
        },
    )
    _write_json(
        gsm_pool,
        {
            "tasks": [
                {
                    "task_id": "gsm",
                    "candidates": [
                        {"candidate_id": "gsm0", "correct": True, "score": 0.7},
                        {"candidate_id": "gsm1", "correct": False, "score": 0.3},
                    ],
                }
            ]
        },
    )
    _write_json(fover, {"detector_auroc": 0.918304})
    _write_json(detector, {"model_specs": {}})
    _write_json(
        headroom,
        {
            "per_domain_headroom": {
                "code": {"selectable_headroom": 0.18},
                "math": {"selectable_headroom": 0.02},
                "sudoku": {"selectable_headroom": 0.129},
            }
        },
    )
    _write_json(arc_summary, {"oracle_ceiling": {"pass@2": 0.6129}, "trm_vote_pass2": 0.4839})
    cfg = mod.ExperimentConfig(
        repo_root=tmp_path,
        registry_path=registry,
        fover_baseline_path=fover,
        detector_config_path=detector,
        headroom_census_path=headroom,
        arc_rerank_path=arc_summary,
        arc_detector_model_path=arc_model,
        arc_candidate_pool_path=arc_pool,
        code_pool_path=code_pool,
        gsm8k_pool_path=gsm_pool,
    )

    domains, unavailable, sources = mod.load_available_domain_rows(cfg)
    checks = mod.check_preconditions(cfg, domains, unavailable)
    headrooms = mod.load_selection_headrooms(headroom, arc_summary)

    assert sorted(domains) == ["code_humaneval_mbpp", "gap4_arc", "gsm8k"]
    assert unavailable == []
    assert arc_model in sources and code_pool in sources and gsm_pool in sources
    assert all(check.available for check in checks)
    assert headrooms == {"gap4_arc": 0.129, "code_humaneval_mbpp": 0.18, "gsm8k": 0.02}
    assert mod.load_selection_headrooms(tmp_path / "missing.json", tmp_path / "missing2.json") == {
        "gap4_arc": 0.0,
        "code_humaneval_mbpp": 0.0,
        "gsm8k": 0.0,
    }


def test_req_verify_4386_gap_append_schema_and_adversarial_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4386: gaps append idempotently and wrappers fail closed."""

    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    entry = {
        "gap_id": "GAP-4386-TEST",
        "status": "open",
        "domain": "test",
        "failure_mode": "chance",
        "missing_discriminator": "feature",
        "candidate_design": "design",
        "priority": "high",
    }
    mod.append_missing_verifier_gaps(gaps_path, [])
    assert not gaps_path.exists()
    mod.append_missing_verifier_gaps(gaps_path, [entry])
    once = gaps_path.read_text(encoding="utf-8")
    mod.append_missing_verifier_gaps(gaps_path, [entry])
    assert gaps_path.read_text(encoding="utf-8") == once

    missing = mod._json_artifact_check(tmp_path / "missing.json", "resource", "key")
    assert missing.available is False and missing.detail == "missing"
    unreadable = tmp_path / "bad.json"
    unreadable.write_text("{bad", encoding="utf-8")
    assert mod._json_artifact_check(unreadable, "resource", "key").detail.startswith("unreadable:")
    no_key = tmp_path / "no_key.json"
    _write_json(no_key, {"other": True})
    assert mod._json_artifact_check(no_key, "resource", "key").detail == "missing key"

    bad_artifact = {
        "detector_generalizes_cross_domain": "yes",
        "verifier_is_oracle": True,
        "detection_by_domain": {},
        "domains_at_chance": {},
        "inference_substrate": "wrong",
    }
    errors = mod.artifact_schema_errors(bad_artifact)
    assert "invalid:detector_generalizes_cross_domain" in errors
    assert "invalid:verifier_is_oracle" in errors
    assert "invalid:detection_by_domain" in errors
    assert "invalid:domains_at_chance" in errors
    assert "invalid:inference_substrate" in errors

    assert (
        mod.run_adversarial_verify(tmp_path / "artifact.json", repo_root=tmp_path)["returncode"]
        is None
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "adversarial_verify.py").write_text("# placeholder\n", encoding="utf-8")

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python"], returncode=0, stdout="clean\n", stderr=""
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    report = mod.run_adversarial_verify(tmp_path / "artifact.json", repo_root=tmp_path)
    assert report["returncode"] == 0
    assert report["stdout_tail"] == "clean\n"


def test_req_verify_4386_blocked_and_write_false_run_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4386: run_experiment covers blocked-write and complete-no-write paths."""

    registry = tmp_path / "ops" / "verifier_registry.yaml"
    fover = tmp_path / "results" / "experiment_4375.json"
    detector = tmp_path / "results" / "experiment_4381.json"
    headroom = tmp_path / "results" / "experiment_4175.json"
    arc_summary = tmp_path / "results" / "arc3.json"
    artifact_path = tmp_path / "results" / "experiment_4386.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")
    _write_json(fover, {"detector_auroc": 0.918304})
    _write_json(detector, {"model_specs": {}})
    _write_json(headroom, {"per_domain_headroom": {"sudoku": {"selectable_headroom": 0.129}}})
    _write_json(arc_summary, {"oracle_ceiling": {"pass@2": 0.6129}, "trm_vote_pass2": 0.4839})
    cfg = mod.ExperimentConfig(
        repo_root=tmp_path,
        registry_path=registry,
        fover_baseline_path=fover,
        detector_config_path=detector,
        headroom_census_path=headroom,
        arc_rerank_path=arc_summary,
        artifact_path=artifact_path,
        bootstrap_resamples=120,
        random_control_replicates=16,
        started_at=1.0,
        clock=lambda: 2.0,
    )

    monkeypatch.setattr(mod, "load_available_domain_rows", lambda _cfg: ({}, [], []))
    blocked = mod.run_experiment(cfg, write=True)
    assert blocked["honest_verdict"] == "blocked_no_non_fover_cached_pool"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == blocked

    good_rows = [
        mod.ScoredCandidate("gap4_arc", "t1", "c0", True, 0.9, True),
        mod.ScoredCandidate("gap4_arc", "t1", "c1", False, 0.1, True),
    ]
    monkeypatch.setattr(
        mod,
        "load_available_domain_rows",
        lambda _cfg: ({"gap4_arc": good_rows}, [], [tmp_path / "source.json"]),
    )
    no_write = mod.run_experiment(cfg, write=False)
    assert no_write["honest_verdict"].startswith("success:")
    assert no_write["adversarial_verify"]["skipped"] is True

    empty_complete = mod.build_complete_artifact(
        domain_results=[],
        unavailable_domains=[],
        preconditions_checked=[],
        source_paths=[],
        duration_s=0.0,
        bootstrap_resamples=120,
        random_control_replicates=16,
    )
    assert empty_complete["honest_verdict"] == "blocked_no_non_fover_cached_pool"
