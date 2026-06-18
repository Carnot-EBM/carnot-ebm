"""Tests for Exp 4397 calibrated cross-domain detector contract.

Spec refs: REQ-VERIFY-4397, SCENARIO-VERIFY-4397.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from carnot import experiment_4397_cross_domain_detection_calibration as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_req_verify_4397_spec_declares_calibrated_multi_domain_contract() -> None:
    """REQ-VERIFY-4397: OpenSpec declares cached-pool LODO calibration fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4397",
        "SCENARIO-VERIFY-4397",
        "experiment_4397_cross_domain_detection_calibration.json",
        "detection_calibrated_multi_domain",
        "pools_built",
        "ece_lodo_calibrated",
        "blocked_insufficient_cached_pools",
    ):
        assert marker in spec


def test_req_verify_4397_loads_code_and_gsm_cached_candidate_pools(tmp_path: Path) -> None:
    """REQ-VERIFY-4397: code/GSM pools are assembled from cached labels and metadata."""

    code_path = tmp_path / "experiment_1999_code_verification_humaneval.json"
    gsm_path = tmp_path / "adversarial_gsm8k_data_400.json"
    _write_json(
        code_path,
        {
            "results": [
                {
                    "task_id": "HumanEval/0",
                    "baseline_passed": False,
                    "repair_passed": True,
                    "extracted_constraints": 2,
                },
                {
                    "task_id": "HumanEval/1",
                    "baseline_passed": True,
                    "repair_passed": True,
                    "extracted_constraints": 0,
                },
            ]
        },
    )
    _write_json(
        gsm_path,
        {
            "datasets": {
                "control": [
                    {
                        "id": 0,
                        "correct_answer": 42,
                        "original_answer": 42,
                        "perturbation": "none",
                    }
                ],
                "number_swapped": [
                    {
                        "id": 0,
                        "correct_answer": 62,
                        "original_answer": 42,
                        "perturbation": "number_swap",
                    }
                ],
                "irrelevant_injected": [
                    {
                        "id": 0,
                        "correct_answer": 42,
                        "original_answer": 42,
                        "perturbation": "irrelevant_injection",
                    }
                ],
                "combined": [
                    {
                        "id": 0,
                        "correct_answer": 62,
                        "original_answer": 42,
                        "perturbation": "combined",
                    }
                ],
            }
        },
    )

    code_rows = mod.load_code_humaneval_rows(code_path)
    gsm_rows = mod.load_gsm8k_original_answer_rows(gsm_path)

    assert [(row.candidate_id, row.is_correct) for row in code_rows] == [
        ("HumanEval/0:baseline", False),
        ("HumanEval/0:repair", True),
        ("HumanEval/1:baseline", True),
        ("HumanEval/1:repair", True),
    ]
    assert code_rows[0].verifier_score < code_rows[1].verifier_score
    assert code_rows[2].verifier_score > code_rows[0].verifier_score
    assert [row.is_correct for row in gsm_rows] == [True, False, True, False]
    assert gsm_rows[0].verifier_score > gsm_rows[1].verifier_score
    assert gsm_rows[2].verifier_score > gsm_rows[3].verifier_score
    assert {row.source for row in gsm_rows} == {str(gsm_path)}


def test_scenario_verify_4397_lodo_calibration_improves_ece() -> None:
    """SCENARIO-VERIFY-4397: LODO Platt calibration reports ECE and risk-coverage."""

    domain_rows = {
        "gap4_arc": [
            mod.ScoredCandidate("gap4_arc", f"a{i}", f"c{i}", i < 20, 0.9 if i < 20 else 0.1)
            for i in range(40)
        ],
        "code_humaneval": [
            mod.ScoredCandidate(
                "code_humaneval", f"h{i}", f"c{i}", i < 20, 0.88 if i < 20 else 0.12
            )
            for i in range(40)
        ],
        "gsm8k": [
            mod.ScoredCandidate("gsm8k", f"g{i}", f"c{i}", i < 20, 0.86 if i < 20 else 0.14)
            for i in range(40)
        ],
    }

    reports = mod.leave_one_domain_out_calibration(
        domain_rows,
        seed=4397,
        n_steps=400,
        learning_rate=0.15,
    )

    assert sorted(reports) == ["code_humaneval", "gap4_arc", "gsm8k"]
    for report in reports.values():
        assert report["ece_lodo_calibrated"] < report["ece_uncalibrated"]
        assert report["platt_scaler"]["trained_on_domains"]
        assert report["risk_coverage"][0]["coverage"] == 1.0


def test_scenario_verify_4397_artifact_has_required_bare_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4397: complete artifact exposes capstone fields bare."""

    source = tmp_path / "pool.json"
    _write_json(source, {"rows": 4})
    rows = [
        mod.ScoredCandidate("gap4_arc", "t1", "c0", True, 0.9, True),
        mod.ScoredCandidate("gap4_arc", "t1", "c1", False, 0.1, True),
        mod.ScoredCandidate("gap4_arc", "t2", "c2", True, 0.8, True),
        mod.ScoredCandidate("gap4_arc", "t2", "c3", False, 0.2, True),
    ]
    calibration = {
        "ece_uncalibrated": 0.4,
        "ece_lodo_calibrated": 0.2,
        "risk_coverage": [{"coverage": 1.0, "risk": 0.0, "n_kept": 4}],
        "platt_scaler": {"trained_on_domains": ["code_humaneval"]},
    }
    domain = mod.summarize_domain(
        "gap4_arc",
        rows,
        selection_headroom=0.129,
        calibration_report=calibration,
        seed=4397,
        bootstrap_resamples=120,
        random_control_replicates=16,
    )

    artifact = mod.build_complete_artifact(
        domain_results=[domain, {**domain, "domain": "code_humaneval"}],
        pools_built=[
            {"domain": "gap4_arc", "source_cached_artifacts": [str(source)], "n": 4},
            {"domain": "code_humaneval", "source_cached_artifacts": [str(source)], "n": 4},
        ],
        unavailable_domains=[],
        preconditions_checked=[
            mod.PreconditionCheck("two_non_fover_cached_labeled_pools", True, "n=2").as_dict()
        ],
        source_paths=[source],
        duration_s=1.25,
        bootstrap_resamples=120,
        random_control_replicates=16,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["detection_calibrated_multi_domain"] is True
    assert artifact["detection_by_domain"][0]["ece_lodo_calibrated"] == 0.2
    assert artifact["pools_built"][0]["domain"] == "gap4_arc"
    assert artifact["domains_at_chance"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["calibration_method"] == "leave_one_domain_out_platt_scaling"
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4397_blocked_artifact_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4397: insufficient cached non-FoVer pools block honestly."""

    artifact = mod.build_blocked_artifact(
        preconditions_checked=[
            mod.PreconditionCheck(
                "two_non_fover_cached_labeled_pools", False, "only gap4_arc"
            ).as_dict()
        ],
        pools_built=[{"domain": "gap4_arc", "source_cached_artifacts": [], "n": 2}],
        unavailable_domains=[{"domain": "code_humaneval", "reason": "missing"}],
        source_paths=[tmp_path / "missing.json"],
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "blocked_insufficient_cached_pools"
    assert artifact["detection_calibrated_multi_domain"] is False
    assert artifact["detection_by_domain"] == []
    assert artifact["domains_at_chance"] == []
    assert artifact["verifier_is_oracle"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_4397_run_experiment_writes_and_logs_chance_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4397: chance domains are surfaced as missing-verifier gaps."""

    registry = tmp_path / "ops" / "verifier_registry.yaml"
    headroom = tmp_path / "results" / "experiment_4175.json"
    arc_summary = tmp_path / "results" / "arc3.json"
    artifact_path = tmp_path / "results" / "experiment_4397.json"
    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")
    _write_json(
        headroom,
        {
            "per_domain_headroom": {
                "code": {"selectable_headroom": 0.18},
                "math": {"selectable_headroom": 0.0},
                "sudoku": {"selectable_headroom": 0.129},
            }
        },
    )
    _write_json(arc_summary, {"oracle_ceiling": {"pass@2": 0.6129}, "trm_vote_pass2": 0.4839})
    source = tmp_path / "source.json"
    _write_json(source, {"ok": True})

    perfect_rows = [
        mod.ScoredCandidate("gap4_arc", f"a{idx}", f"c{idx}", idx < 20, 0.9 if idx < 20 else 0.1)
        for idx in range(40)
    ]
    chance_rows = [
        mod.ScoredCandidate("code_humaneval", f"h{idx}", f"c{idx}", idx % 2 == 0, 0.5, True)
        for idx in range(40)
    ]

    monkeypatch.setattr(
        mod,
        "load_available_domain_rows",
        lambda _cfg: (
            {"gap4_arc": perfect_rows, "code_humaneval": chance_rows},
            [
                {"domain": "gap4_arc", "source_cached_artifacts": [str(source)], "n": 40},
                {"domain": "code_humaneval", "source_cached_artifacts": [str(source)], "n": 40},
            ],
            [],
            [source],
        ),
    )

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            registry_path=registry,
            headroom_census_path=headroom,
            arc_rerank_path=arc_summary,
            artifact_path=artifact_path,
            verifier_gaps_path=gaps_path,
            min_non_fover_pools=2,
            min_domain_candidates=2,
            bootstrap_resamples=120,
            random_control_replicates=16,
            calibration_steps=300,
            started_at=1.0,
            clock=lambda: 3.0,
        ),
        adversarial_verify_runner=lambda _path: {"returncode": 0, "flags": []},
        write=True,
    )

    assert artifact["honest_verdict"] == "complete: calibrated_multi_domain_contract_false"
    assert artifact["detection_calibrated_multi_domain"] is False
    assert artifact["domains_at_chance"] == ["code_humaneval"]
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert "GAP-4397-CODE-HUMANEVAL-DETECTOR-CHANCE" in gaps_path.read_text(
        encoding="utf-8"
    )
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_req_verify_4397_defensive_edges_and_blocked_run_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4397: defensive branches fail closed and remain deterministic."""

    gz_path = tmp_path / "payload.json.gz"
    import gzip

    with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
        json.dump({"ok": True}, handle)
    assert mod._read_json(gz_path) == {"ok": True}
    assert mod._normalize_higher_correct([]) == []
    assert mod._normalize_higher_correct([2.0, 2.0]) == [0.5, 0.5]
    assert mod._normalize_higher_correct([2.0, 4.0]) == [0.0, 1.0]

    monkeypatch.setattr(
        mod,
        "read_labeled_fover_rows",
        lambda _path: [
            {"question_id": "q0", "label": "correct"},
            {"question_id": "q1", "label": "incorrect"},
        ],
    )
    monkeypatch.setattr(
        mod,
        "score_fover_production_ensemble",
        lambda _rows, _repo: SimpleNamespace(scores=[0.1, 0.9]),
    )
    fover_rows = mod.load_fover_rows(tmp_path / "fover.jsonl", tmp_path)
    assert [(row.candidate_id, row.is_correct, row.verifier_score) for row in fover_rows] == [
        ("fover:0", True, 1.0),
        ("fover:1", False, 0.0),
    ]

    malformed_code = tmp_path / "malformed_code.json"
    _write_json(malformed_code, {"results": "bad"})
    assert mod.load_code_humaneval_rows(malformed_code) == []
    skipped_code = tmp_path / "skipped_code.json"
    _write_json(
        skipped_code,
        {"results": ["bad", {"task_id": "skip"}, {"repair_passed": False}]},
    )
    assert [row.candidate_id for row in mod.load_code_humaneval_rows(skipped_code)] == [
        "HumanEval/2:repair"
    ]

    malformed_gsm = tmp_path / "malformed_gsm.json"
    _write_json(malformed_gsm, {"datasets": "bad"})
    assert mod.load_gsm8k_original_answer_rows(malformed_gsm) == []
    edge_gsm = tmp_path / "edge_gsm.json"
    _write_json(
        edge_gsm,
        {
            "datasets": {
                "bad": "not-list",
                "mystery": [
                    "bad-row",
                    {"correct_answer": 1},
                    {"correct_answer": 1, "original_answer": 2},
                ],
            }
        },
    )
    rows = mod.load_gsm8k_original_answer_rows(edge_gsm)
    assert len(rows) == 1
    assert rows[0].verifier_score == 0.5

    monkeypatch.setattr(
        mod,
        "load_fover_rows",
        lambda *_args: [
            mod.ScoredCandidate("fover", "f", "f0", True, 0.8),
            mod.ScoredCandidate("fover", "f", "f1", False, 0.2),
        ],
    )
    monkeypatch.setattr(
        mod,
        "load_arc_set_encoder_rows",
        lambda *_args: [
            mod.ScoredCandidate("gap4_arc", "a", "a0", True, 0.9),
            mod.ScoredCandidate("gap4_arc", "a", "a1", False, 0.1),
        ],
    )
    monkeypatch.setattr(
        mod,
        "load_code_humaneval_rows",
        lambda _path: [
            mod.ScoredCandidate("code_humaneval", "c", "c0", True, 0.8),
            mod.ScoredCandidate("code_humaneval", "c", "c1", False, 0.2),
        ],
    )
    monkeypatch.setattr(mod, "load_gsm8k_original_answer_rows", lambda _path: [])
    domains, pools, unavailable, sources = mod.load_available_domain_rows(
        mod.ExperimentConfig(repo_root=tmp_path)
    )
    assert sorted(domains) == ["code_humaneval", "fover", "gap4_arc"]
    assert [pool["domain"] for pool in pools] == ["fover", "gap4_arc", "code_humaneval"]
    assert unavailable == [{"domain": "gsm8k", "reason": "no_usable_cached_rows"}]
    assert sources

    with pytest.raises(ValueError, match="same length"):
        mod.expected_calibration_error([1], [0.5, 0.6])
    assert mod.expected_calibration_error([], []) == 0.0
    with pytest.raises(ValueError, match="same length"):
        mod.fit_platt_scaler([0.1], [1, 0], trained_on_domains=["x"])
    assert mod.fit_platt_scaler([], [], trained_on_domains=["x"]).n_train == 0
    with pytest.raises(ValueError, match="same length"):
        mod.risk_coverage_curve([1], [0.5, 0.6])
    assert mod.risk_coverage_curve([], []) == []

    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    mod.append_missing_verifier_gaps(gaps_path, [])
    assert not gaps_path.exists()
    entry = {
        "gap_id": "GAP-4397-DUP",
        "status": "open",
        "domain": "dup",
        "failure_mode": "chance",
        "missing_discriminator": "feature",
        "candidate_design": "design",
        "priority": "medium",
    }
    mod.append_missing_verifier_gaps(gaps_path, [entry])
    once = gaps_path.read_text(encoding="utf-8")
    mod.append_missing_verifier_gaps(gaps_path, [entry])
    assert gaps_path.read_text(encoding="utf-8") == once

    assert mod._json_has_key(tmp_path / "missing.json", "resource", "key").detail == "missing"
    unreadable = tmp_path / "bad.json"
    unreadable.write_text("{bad", encoding="utf-8")
    assert mod._json_has_key(unreadable, "resource", "key").detail.startswith("unreadable:")
    no_key = tmp_path / "no_key.json"
    _write_json(no_key, {"other": True})
    assert mod._json_has_key(no_key, "resource", "key").detail == "missing key"

    bad_artifact = {
        "detection_calibrated_multi_domain": "yes",
        "verifier_is_oracle": True,
        "detection_by_domain": {},
        "domains_at_chance": {},
        "pools_built": {},
        "inference_substrate": "wrong",
    }
    errors = mod.artifact_schema_errors(bad_artifact)
    assert "invalid:detection_calibrated_multi_domain" in errors
    assert "invalid:verifier_is_oracle" in errors
    assert "invalid:detection_by_domain" in errors
    assert "invalid:domains_at_chance" in errors
    assert "invalid:pools_built" in errors
    assert "invalid:inference_substrate" in errors

    registry = tmp_path / "ops" / "verifier_registry.yaml"
    artifact_path = tmp_path / "results" / "experiment_4397.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("verifiers:\n- verifier_id: fover_production_ensemble\n", encoding="utf-8")
    monkeypatch.setattr(
        mod,
        "load_available_domain_rows",
        lambda _cfg: ({}, [], [{"domain": "gap4_arc", "reason": "missing"}], []),
    )
    blocked = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            registry_path=registry,
            artifact_path=artifact_path,
            headroom_census_path=tmp_path / "missing_headroom.json",
            arc_rerank_path=tmp_path / "missing_arc.json",
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_insufficient_cached_pools"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == blocked

    good_rows = [
        mod.ScoredCandidate("gap4_arc", "a", "a0", True, 0.9),
        mod.ScoredCandidate("gap4_arc", "a", "a1", False, 0.1),
        mod.ScoredCandidate("code_humaneval", "c", "c0", True, 0.8),
        mod.ScoredCandidate("code_humaneval", "c", "c1", False, 0.2),
    ]
    monkeypatch.setattr(
        mod,
        "load_available_domain_rows",
        lambda _cfg: (
            {"gap4_arc": good_rows[:2], "code_humaneval": good_rows[2:]},
            [
                {"domain": "gap4_arc", "source_cached_artifacts": [], "n": 2},
                {"domain": "code_humaneval", "source_cached_artifacts": [], "n": 2},
            ],
            [],
            [],
        ),
    )
    _write_json(
        tmp_path / "headroom.json",
        {"per_domain_headroom": {"code": {"selectable_headroom": 0.18}}},
    )
    _write_json(
        tmp_path / "arc.json",
        {"oracle_ceiling": {"pass@2": 0.6}, "trm_vote_pass2": 0.5},
    )
    no_write = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            registry_path=registry,
            artifact_path=artifact_path,
            headroom_census_path=tmp_path / "headroom.json",
            arc_rerank_path=tmp_path / "arc.json",
            min_domain_candidates=1,
            bootstrap_resamples=20,
            random_control_replicates=4,
            calibration_steps=20,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        write=False,
    )
    assert no_write["adversarial_verify"]["skipped"] is True
