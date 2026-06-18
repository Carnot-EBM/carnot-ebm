"""Tests for Exp 4408 deconfounded detector-calibration repair.

Spec refs: REQ-VERIFY-4408, SCENARIO-VERIFY-4408.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from carnot import experiment_4408_cross_domain_detection_calibration_repair as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _rows(domain: str, n: int, *, high: float = 0.8, low: float = 0.2) -> list[mod.ScoredCandidate]:
    return [
        mod.ScoredCandidate(
            domain=domain,
            task_id=f"{domain}/task/{idx // 2}",
            candidate_id=f"{domain}:{idx}",
            is_correct=idx % 2 == 0,
            verifier_score=high if idx % 2 == 0 else low,
            valid_output=True,
            source=f"{domain}.fixture",
            semantic_key=f"answer-{idx}",
        )
        for idx in range(n)
    ]


def test_req_verify_4408_spec_declares_sca_repair_contract() -> None:
    """REQ-VERIFY-4408: OpenSpec declares proper-pool SCA repair fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4408",
        "SCENARIO-VERIFY-4408",
        "experiment_4408_cross_domain_detection_calibration_repair.json",
        "Semantic Confidence Aggregation",
        "base_rate_separation",
        "blocked_insufficient_pools_for_multi_domain_claim",
        "cited_upstream_artifacts",
    ):
        assert marker in spec


def test_req_verify_4408_sca_groups_semantic_answers_and_records_base_rates() -> None:
    """REQ-VERIFY-4408: SCA records raw/SCA base rates and answer cardinality."""

    rows = [
        mod.ScoredCandidate("code_humaneval", "HumanEval/0", "a0", True, 0.6, semantic_key="same"),
        mod.ScoredCandidate("code_humaneval", "HumanEval/0", "a1", True, 0.5, semantic_key="same"),
        mod.ScoredCandidate("code_humaneval", "HumanEval/0", "b0", False, 0.2, semantic_key="wrong"),
        mod.ScoredCandidate("code_humaneval", "HumanEval/1", "c0", False, 0.3, semantic_key="other"),
    ]

    result = mod.semantic_confidence_aggregation(rows)

    assert len(result.rows) == 3
    assert result.metadata["raw_n"] == 4
    assert result.metadata["n"] == 3
    assert result.metadata["raw_base_rate"] == 0.5
    assert result.metadata["base_rate"] == pytest.approx(1 / 3)
    assert result.metadata["answer_cardinality"]["max"] == 2
    grouped = {row.candidate_id: row for row in result.rows}
    assert grouped["HumanEval/0::same"].verifier_score == 0.8


def test_req_verify_4408_loads_powered_code_rows_from_cached_reward_corpus(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4408: HumanEval rows use hidden pass labels and non-oracle scores."""

    source = tmp_path / "arm_A.jsonl"
    _write_jsonl(
        source,
        [
            {
                "arm": "A_certified",
                "completion": "def f():\n    return 1\n",
                "hidden_pass": True,
                "source_draw_index": 0,
                "task_id": "HumanEval/0",
                "visible_perfect": True,
            },
            {
                "arm": "B_random_same_generator",
                "completion": "def f():\n    return 0\n",
                "hidden_pass": False,
                "source_draw_index": 1,
                "task_id": "HumanEval/0",
                "visible_perfect": False,
            },
        ],
    )
    exp4233 = tmp_path / "experiment_4233_oracle_distinct_code_beats_vote.json"
    _write_json(
        exp4233,
        {"candidate_pool": {"source_paths": [str(source)], "candidate_n": 2}},
    )

    rows = mod.load_code_humaneval_reward_rows(exp4233)

    assert [(row.task_id, row.is_correct) for row in rows] == [
        ("HumanEval/0", True),
        ("HumanEval/0", False),
    ]
    assert rows[0].verifier_score > rows[1].verifier_score
    assert rows[0].semantic_key != rows[1].semantic_key
    assert rows[0].source == str(source)


def test_scenario_verify_4408_complete_artifact_has_required_bare_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4408: powered proper pools emit the repair artifact."""

    source = tmp_path / "cached_pool.json"
    _write_json(source, {"ok": True})
    artifact_path = tmp_path / "results" / "experiment_4408.json"
    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    rows_by_domain = {
        "fover": _rows("fover", 4),
        "gap4_arc": _rows("gap4_arc", 6),
        "code_humaneval": _rows("code_humaneval", 6),
        "gsm8k": _rows("gsm8k", 6),
    }
    monkeypatch.setattr(
        mod,
        "load_raw_domain_rows",
        lambda _cfg: (
            rows_by_domain,
            [
                mod.pool_record(domain, [source], len(rows))
                for domain, rows in rows_by_domain.items()
            ],
            [],
            [source],
        ),
    )

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            artifact_path=artifact_path,
            verifier_gaps_path=gaps_path,
            min_powered_n=3,
            bootstrap_resamples=80,
            random_control_replicates=8,
            calibration_steps=120,
            started_at=10.0,
            clock=lambda: 12.5,
        ),
        adversarial_verify_runner=lambda _path: {"returncode": 0, "flags": []},
        write=True,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["detection_calibrated_multi_domain"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["base_rate_separation"]["method"] == "semantic_confidence_aggregation"
    assert artifact["detection_by_domain"][0]["answer_cardinality"]["mean"] == 2.0
    assert artifact["detection_by_domain"][0]["random_score_control"]["replicates"] == 8
    assert artifact["cited_upstream_artifacts"]
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_4408_blocked_when_fewer_than_two_non_fover_pools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4408: insufficient powered domains block honestly."""

    source = tmp_path / "cached_pool.json"
    _write_json(source, {"ok": True})
    monkeypatch.setattr(
        mod,
        "load_raw_domain_rows",
        lambda _cfg: (
            {"fover": _rows("fover", 4), "gap4_arc": _rows("gap4_arc", 4)},
            [
                mod.pool_record("fover", [source], 4),
                mod.pool_record("gap4_arc", [source], 4),
            ],
            [{"domain": "code_humaneval", "reason": "missing"}],
            [source],
        ),
    )

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            artifact_path=tmp_path / "results" / "experiment_4408.json",
            min_powered_n=5,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_insufficient_pools_for_multi_domain_claim"
    assert artifact["detection_calibrated_multi_domain"] is False
    assert artifact["detection_by_domain"] == []
    assert artifact["base_rate_separation"]["powered_non_fover_domains"] == []
    assert artifact["verifier_is_oracle"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_4408_schema_errors_and_chance_gap_entry() -> None:
    """REQ-VERIFY-4408: schema and missing-verifier gap guards fail closed."""

    bad = {
        "detection_calibrated_multi_domain": "true",
        "verifier_is_oracle": True,
        "detection_by_domain": {},
        "base_rate_separation": [],
        "preconditions_checked": {},
        "cited_upstream_artifacts": {},
        "inference_substrate": "wrong",
    }

    errors = mod.artifact_schema_errors(bad)

    assert "missing:honest_verdict" in errors
    assert "invalid:detection_calibrated_multi_domain" in errors
    assert "invalid:verifier_is_oracle" in errors
    assert "invalid:base_rate_separation" in errors
    gap = mod.missing_gap_entries(
        [{"domain": "gsm8k", "auroc_ci95": [0.45, 0.55], "n": 400}]
    )
    assert gap[0]["gap_id"] == "GAP-4408-GSM8K-DECONFOUNDED-DETECTOR-CHANCE"


def test_req_verify_4408_loader_defensive_edges_and_no_write_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4408: raw loaders, citation hashes, and no-write path are deterministic."""

    code_source = tmp_path / "arm_A.jsonl"
    _write_jsonl(
        code_source,
        [
            {"task_id": "skip", "completion": "bad"},
            {
                "completion": "def ok():\n    return True\n",
                "hidden_pass": True,
                "source_draw_index": "bad",
                "task_id": "HumanEval/9",
                "visible_perfect": True,
            },
        ],
    )
    code_artifact = tmp_path / "experiment_4233.json"
    _write_json(
        code_artifact,
        {"candidate_pool": {"source_paths": ["missing.jsonl", "arm_A.jsonl"]}},
    )
    no_sources = tmp_path / "no_sources.json"
    _write_json(no_sources, {"candidate_pool": {}})
    assert mod.load_code_humaneval_reward_rows(no_sources) == []

    loaded_code = mod.load_code_humaneval_reward_rows(code_artifact)
    assert len(loaded_code) == 1
    assert loaded_code[0].task_id == "HumanEval/9"

    def _ns(domain: str, correct: bool, score: float) -> SimpleNamespace:
        return SimpleNamespace(
            domain=domain,
            task_id=f"{domain}/task",
            candidate_id=f"{domain}/candidate",
            is_correct=correct,
            verifier_score=score,
            valid_output=True,
            source=f"{domain}.json",
        )

    monkeypatch.setattr(mod, "_load_fover_rows", lambda *_args: [_ns("fover", True, 0.8)])
    monkeypatch.setattr(mod, "_load_arc_set_encoder_rows", lambda *_args: [_ns("gap4_arc", False, 0.2)])
    monkeypatch.setattr(mod, "_load_gsm8k_original_answer_rows", lambda *_args: [_ns("gsm8k", True, 0.7)])
    domains, pools, unavailable, sources = mod.load_raw_domain_rows(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            code_reward_artifact_path=code_artifact,
            fover_corpus_path=tmp_path / "fover.jsonl",
            fover_baseline_path=tmp_path / "fover_baseline.json",
            fover_dual_condition_path=tmp_path / "fover_dual.json",
            arc_detector_model_path=tmp_path / "arc_model.json",
            arc_candidate_pool_path=tmp_path / "arc_pool.json.gz",
            arc_rerank_path=tmp_path / "arc_rerank.json",
            gsm8k_pool_path=tmp_path / "gsm.json",
            gsm8k_baseline_path=tmp_path / "gsm_base.json",
        )
    )
    assert sorted(domains) == ["code_humaneval", "fover", "gap4_arc", "gsm8k"]
    assert [pool["domain"] for pool in pools] == ["fover", "gap4_arc", "code_humaneval", "gsm8k"]
    assert unavailable == []
    assert sources

    monkeypatch.setattr(mod, "_load_gsm8k_original_answer_rows", lambda *_args: [])
    _domains, _pools, unavailable_empty, _sources = mod.load_raw_domain_rows(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            code_reward_artifact_path=code_artifact,
            fover_corpus_path=tmp_path / "fover.jsonl",
            fover_baseline_path=tmp_path / "fover_baseline.json",
            fover_dual_condition_path=tmp_path / "fover_dual.json",
            arc_detector_model_path=tmp_path / "arc_model.json",
            arc_candidate_pool_path=tmp_path / "arc_pool.json.gz",
            arc_rerank_path=tmp_path / "arc_rerank.json",
            gsm8k_pool_path=tmp_path / "gsm.json",
            gsm8k_baseline_path=tmp_path / "gsm_base.json",
        )
    )
    assert {"domain": "gsm8k", "reason": "no_usable_cached_rows"} in unavailable_empty

    empty_sca = mod.semantic_confidence_aggregation([])
    assert empty_sca.metadata["answer_cardinality"]["max"] == 0
    conflict_sca = mod.semantic_confidence_aggregation(
        [
            mod.ScoredCandidate("d", "t", "a", True, 0.4, semantic_key="same"),
            mod.ScoredCandidate("d", "t", "b", False, 0.3, semantic_key="same"),
        ]
    )
    assert conflict_sca.metadata["semantic_conflict_groups"] == 1

    missing = tmp_path / "missing.json"
    duplicate_cited = mod._cited_upstream_artifacts(
        mod.ExperimentConfig(exp4397_path=missing, sca_ingestion_path=missing),
        [missing],
    )
    assert duplicate_cited[0]["sha256"] == "missing"
    assert [item["path"] for item in duplicate_cited].count(str(missing)) == 1

    rows_by_domain = {
        "gap4_arc": _rows("gap4_arc", 6),
        "code_humaneval": _rows("code_humaneval", 6),
    }
    monkeypatch.setattr(
        mod,
        "load_raw_domain_rows",
        lambda _cfg: (
            rows_by_domain,
            [mod.pool_record(domain, [code_source], len(rows)) for domain, rows in rows_by_domain.items()],
            [],
            [code_source],
        ),
    )
    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            artifact_path=tmp_path / "no_write.json",
            min_powered_n=3,
            bootstrap_resamples=40,
            random_control_replicates=4,
            calibration_steps=80,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        write=False,
    )
    assert artifact["adversarial_verify"]["skipped"] is True
