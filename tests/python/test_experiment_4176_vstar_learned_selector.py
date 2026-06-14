"""Tests for Exp 4176 V-STaR learned selector.

Spec refs: REQ-VERIFY-4176, SCENARIO-VERIFY-4176.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import vstar_learned_selector_4176 as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_repo(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    source = results / "experiment_1999_code_verification_humaneval.json"
    _write_json(
        source,
        {
            "honest_verdict": "complete: fixture",
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
                {
                    "task_id": "HumanEval/2",
                    "baseline_passed": False,
                    "repair_passed": False,
                    "extracted_constraints": 3,
                },
                {
                    "task_id": "HumanEval/3",
                    "baseline_passed": False,
                    "repair_passed": True,
                    "extracted_constraints": 2,
                },
                {
                    "task_id": "HumanEval/4",
                    "baseline_passed": True,
                    "repair_passed": True,
                    "extracted_constraints": 0,
                },
                {
                    "task_id": "HumanEval/5",
                    "baseline_passed": False,
                    "repair_passed": True,
                    "extracted_constraints": 1,
                },
            ],
        },
    )
    _write_json(
        results / "experiment_4175_headroom_gate_executable_census.json",
        {
            "honest_verdict": "complete: headroom_present_domain_code",
            "headroom_present_domain": "code",
            "per_domain_headroom": {
                "code": {
                    "artifact_flags": {
                        "source": str(source),
                        "candidate_pool_detected": True,
                        "census_incomplete": False,
                    },
                    "selectable_headroom": 0.5,
                }
            },
        },
    )
    return tmp_path


def test_req_4176_spec_declares_selector_contract() -> None:
    """REQ-VERIFY-4176: OpenSpec declares the selector, metrics, and principles."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4176",
        "SCENARIO-VERIFY-4176",
        "python/carnot/reporting/vstar_learned_selector_4176.py",
        "results/experiment_4176_vstar_learned_selector.py",
        "selector_auroc_oof",
        "selector_pass1_vs_vote",
        "accepted_rejected_n",
        "cached_artifact_oof_vstar_selector",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_load_trace_corpus_uses_exp4175_code_pool(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4176: traces come from the selected executable code pool."""

    root = _make_repo(tmp_path)

    corpus = mod.load_trace_corpus(root)

    assert corpus.domain == "code"
    assert corpus.source_path.name == "experiment_1999_code_verification_humaneval.json"
    assert len(corpus.rows) == 12
    assert mod.accepted_rejected_counts(corpus.rows) == {"accepted": 7, "rejected": 5, "total": 12}
    first = corpus.rows[0]
    assert first.task_id == "HumanEval/0"
    assert first.candidate_id == "HumanEval/0::baseline"
    assert first.correct is False
    assert first.features["role_repair"] == 0.0
    assert first.features["vote_weight"] == 1.0
    assert "correct" not in first.features
    repair = corpus.rows[1]
    assert repair.features["role_repair"] == 1.0
    assert repair.features["candidate_index"] == 1.0


def test_oof_selector_reports_auroc_and_pass1_lift(tmp_path: Path) -> None:
    """REQ-VERIFY-4176: OOF training reports discrimination and ranker lift."""

    corpus = mod.load_trace_corpus(_make_repo(tmp_path))

    report = mod.train_oof_selector(corpus.rows, random_seed=mod.RANDOM_SEED, n_folds=3)

    assert report.selector_auroc_oof > 0.5
    assert report.sc_vote_pass1 == pytest.approx(2 / 6)
    assert report.selector_pass1 == pytest.approx(5 / 6)
    assert report.selector_pass1_vs_vote == pytest.approx(0.5)
    assert len(report.oof_scores) == len(corpus.rows)
    assert {row.fold for row in report.oof_rows} == {0, 1, 2}
    for row in report.oof_rows:
        assert row.task_id not in row.train_task_ids


def test_run_writes_terminal_artifact_and_deployable_selector(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4176: runner writes metrics and persisted ranker."""

    root = _make_repo(tmp_path)

    artifact = mod.run(root, random_seed=mod.RANDOM_SEED, n_folds=3)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["selector_auroc_oof"] > 0.5
    assert artifact["selector_pass1_vs_vote"] == pytest.approx(0.5)
    assert artifact["accepted_rejected_n"] == {"accepted": 7, "rejected": 5, "total": 12}
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == ["REQ-VERIFY-4176", "SCENARIO-VERIFY-4176"]
    assert artifact["inference_substrate"] == "cached_artifact_oof_vstar_selector"

    selector_path = Path(artifact["selector_path"])
    assert selector_path.exists()
    selector = mod.load_selector(selector_path)
    assert selector["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    baseline_score = mod.score_with_selector(
        selector,
        {
            "role_repair": 0.0,
            "vote_weight": 1.0,
            "candidate_index": 0.0,
            "extracted_constraints": 2.0,
        },
    )
    repair_score = mod.score_with_selector(
        selector,
        {
            "role_repair": 1.0,
            "vote_weight": 0.0,
            "candidate_index": 1.0,
            "extracted_constraints": 2.0,
        },
    )
    assert repair_score > baseline_score

    written = json.loads(
        (root / "results" / "experiment_4176_vstar_learned_selector.json").read_text(
            encoding="utf-8"
        )
    )
    assert written == artifact


def test_run_blocks_without_usable_headroom_domain(tmp_path: Path) -> None:
    """REQ-VERIFY-4176: missing cached selected pool writes a blocked artifact."""

    _write_json(
        tmp_path / "results" / "experiment_4175_headroom_gate_executable_census.json",
        {
            "honest_verdict": "complete: no_domain_clears_0.10",
            "headroom_present_domain": "",
            "per_domain_headroom": {},
        },
    )

    artifact = mod.run(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_no_headroom_present_domain"
    assert artifact["selector_auroc_oof"] == 0.0
    assert artifact["selector_pass1_vs_vote"] == 0.0
    assert artifact["accepted_rejected_n"] == {"accepted": 0, "rejected": 0, "total": 0}


def test_precondition_and_schema_errors_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-4176: blocked and malformed states are classified explicitly."""

    assert mod._as_float(True) == 0.0
    assert mod._as_float("not-a-number") == 0.0

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match="blocked_malformed_json_artifact"):
        mod._read_json_object(list_json)
    with pytest.raises(mod.BlockedRun, match="blocked_missing_candidate_pool_source"):
        mod._resolve_source_path(tmp_path, "")
    with pytest.raises(mod.BlockedRun, match="blocked_candidate_pool_missing"):
        mod._resolve_source_path(tmp_path, "missing.json")

    bad_pool = tmp_path / "bad_pool.json"
    _write_json(bad_pool, {"results": "not rows"})
    with pytest.raises(mod.BlockedRun, match="blocked_candidate_pool_missing_rows"):
        mod._code_trace_rows(bad_pool)
    empty_pool = tmp_path / "empty_pool.json"
    _write_json(
        empty_pool,
        {"results": [None, {"task_id": "x", "baseline_passed": "bad", "repair_passed": None}]},
    )
    with pytest.raises(mod.BlockedRun, match="blocked_no_labeled_candidate_traces"):
        mod._code_trace_rows(empty_pool)

    with pytest.raises(mod.BlockedRun, match="blocked_missing_headroom_gate"):
        mod.load_trace_corpus(tmp_path / "no-headroom")

    source = tmp_path / "results" / "pool.json"
    _write_json(
        source,
        {
            "results": [
                {"task_id": "x", "baseline_passed": True, "repair_passed": False},
                {"task_id": "y", "baseline_passed": False, "repair_passed": True},
            ]
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_4175_headroom_gate_executable_census.json",
        {"headroom_present_domain": "code", "per_domain_headroom": {"math": {}}},
    )
    with pytest.raises(mod.BlockedRun, match="blocked_headroom_domain_missing_stats"):
        mod.load_trace_corpus(tmp_path)
    _write_json(
        tmp_path / "results" / "experiment_4175_headroom_gate_executable_census.json",
        {
            "headroom_present_domain": "math",
            "per_domain_headroom": {"math": {"artifact_flags": {"source": str(source)}}},
        },
    )
    with pytest.raises(mod.BlockedRun, match="blocked_unsupported_headroom_domain_math"):
        mod.load_trace_corpus(tmp_path)

    one_class = [
        mod.TraceRow("t", "t::baseline", "baseline", 0, 1.0, True, dict.fromkeys(mod.FEATURE_NAMES, 0.0))
    ]
    with pytest.raises(mod.BlockedRun, match="blocked_lacks_accepted_rejected_traces"):
        mod._auroc([True], [0.5])
    with pytest.raises(mod.BlockedRun, match="blocked_lacks_accepted_rejected_traces"):
        mod.train_oof_selector(one_class)

    valid_blocked = mod._blocked_artifact("blocked_fixture", mod.RANDOM_SEED, 0.1)
    invalid_cases = [
        ({k: v for k, v in valid_blocked.items() if k != "selector_path"}, "missing required"),
        ({**valid_blocked, "honest_verdict": "not-terminal"}, "terminal-prefixed"),
        ({**valid_blocked, "selector_auroc_oof": {"value": 0.0}}, "bare float"),
        ({**valid_blocked, "selector_pass1_vs_vote": {"value": 0.0}}, "bare float"),
        ({**valid_blocked, "field_principles": {}}, "field_principles"),
        ({**valid_blocked, "inference_substrate": "live"}, "inference_substrate"),
        ({**valid_blocked, "accepted_rejected_n": {}}, "accepted_rejected_n"),
        (
            {
                **valid_blocked,
                "honest_verdict": "complete: missing_selector",
                "accepted_rejected_n": {"accepted": 1, "rejected": 1, "total": 2},
                "selector_path": str(tmp_path / "missing_selector.json"),
            },
            "persisted selector",
        ),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
