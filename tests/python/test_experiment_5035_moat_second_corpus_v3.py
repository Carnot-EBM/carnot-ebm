"""Tests for Exp 5035 D4 v3 cross-corpus moat generalization.

Spec refs: REQ-VERIFY-5035, SCENARIO-VERIFY-5035.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5035_moat_second_corpus_v3 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _audit_clean(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def _mmlu_rows(n: int) -> list[dict[str, Any]]:
    return [
        {
            "row_id": f"mmlu:{index}",
            "corpus": "MMLU-Pro-hard",
            "question": f"Question {index}?",
            "context": "",
            "choices": ["A", "B"],
            "gold": "A",
        }
        for index in range(n)
    ]


def _headroom_rows(n: int) -> list[dict[str, Any]]:
    rows = []
    for row in _mmlu_rows(n):
        rows.append(
            {
                **row,
                "candidates": [
                    {
                        "candidate_id": f"{row['row_id']}/wrong",
                        "answer": "B",
                        "cache_index": 0,
                        "temperature": "cached",
                        "d1_base_reward": 0.0,
                        "uprm_process_score": 0.0,
                    },
                    {
                        "candidate_id": f"{row['row_id']}/right",
                        "answer": "A",
                        "cache_index": 1,
                        "temperature": "cached",
                        "d1_base_reward": 1.0,
                        "uprm_process_score": 1.0,
                    },
                ],
            }
        )
    return rows


def _no_headroom_rows(n: int) -> list[dict[str, Any]]:
    rows = []
    for row in _mmlu_rows(n):
        rows.append(
            {
                **row,
                "candidates": [
                    {
                        "candidate_id": f"{row['row_id']}/right0",
                        "answer": "A",
                        "cache_index": 0,
                        "temperature": "cached",
                        "d1_base_reward": 1.0,
                        "uprm_process_score": 1.0,
                    },
                    {
                        "candidate_id": f"{row['row_id']}/right1",
                        "answer": "A",
                        "cache_index": 1,
                        "temperature": "cached",
                        "d1_base_reward": 0.8,
                        "uprm_process_score": 0.8,
                    },
                ],
            }
        )
    return rows


def _candidate_loader(rows: list[dict[str, Any]], path: Path):
    def load(**_kwargs: Any) -> tuple[list[dict[str, Any]], Path]:
        return rows, path

    return load


def _write_d3_artifact(root: Path, *, delta: float = 0.08, abstention: float = 0.0) -> None:
    _write_json(
        root / mod.D3_ARTIFACT_RELATIVE_PATH,
        {
            "verifier_is_oracle": False,
            "headroom_present": True,
            "abstention_rate": abstention,
            "ebrm_selection_accuracy": 0.665,
            "genuine_tuned_sc_accuracy": 0.665 - delta,
            "delta_vs_tuned_sc": delta,
            "paired_ci95": [0.0, 0.165],
            "uncertainty_calibration": {"selected_threshold": 1.0},
            "degeneracy_guard": {"degeneracy_flag": abstention > 0.5},
            "model_specs": {"base_scorer": "fixture-d1"},
        },
    )


def _write_d2_artifact(root: Path, *, delta: float = 0.03) -> None:
    _write_json(
        root / mod.D2_ARTIFACT_RELATIVE_PATH,
        {
            "experiment": "experiment_5032_uprm_replication_v3",
            "verifier_is_oracle": False,
            "headroom_present": True,
            "uprm_selection_accuracy": 0.62,
            "genuine_tuned_sc_accuracy": 0.62 - delta,
            "delta_vs_tuned_sc": delta,
            "scoring_path": "uprm_logprob",
            "model_specs": {"cached_candidate_generator": "gemma-4-12B-it-GGUF"},
        },
    )


def test_req_verify_5035_spec_declares_v3_cross_corpus_contract() -> None:
    """REQ-VERIFY-5035: OpenSpec anchors the v3 D4 cross-corpus contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5035",
        "SCENARIO-VERIFY-5035",
        "experiment_5035_moat_second_corpus_v3.py",
        "results/experiment_5035_moat_second_corpus_v3.json",
        "blocked_no_best_verifier",
        "oracle@K - GENUINE tuned-SC >= 0.10",
        "success_moat_generalizes_<corpus>_<delta>",
        "complete_moat_musr_scoped_<corpus>_no_confirm",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_verify_5035_all_d_arms_blocked_does_not_use_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5035: all blocked D arms emit blocked_no_best_verifier."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_loader=_candidate_loader(_headroom_rows(3), tmp_path / "rows.jsonl"),
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=3,
        limit=3,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_no_best_verifier"
    assert artifact["best_verifier_from"] is None
    assert artifact["verifier_is_oracle"] is False
    assert artifact["second_corpus_accuracy"] is None
    assert all(check["available"] is False for check in artifact["preconditions_checked"])
    assert "registry" not in json.dumps(artifact).lower()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5035_successful_second_corpus_generalization(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5035: a headroom-present second corpus can confirm D3."""

    _write_json(
        tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH,
        {
            "verifier_is_oracle": False,
            "scorer_trained": True,
            "trained_scorer_accuracy": 0.58,
            "delta_vs_tuned_sc": 0.01,
            "model_specs": {"base_model": "Qwen/Qwen3.5-2B"},
        },
    )
    _write_d2_artifact(tmp_path, delta=-0.02)
    _write_d3_artifact(tmp_path, delta=0.08)
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        corpus_loaders=[
            ("GPQA", lambda _limit: (_ for _ in ()).throw(mod.SecondCorpusUnavailable("no gpqa"))),
            ("MMLU-Pro-hard", _mmlu_rows),
        ],
        candidate_rows_loader=_candidate_loader(
            _headroom_rows(3), tmp_path / "mmlu_candidates.jsonl"
        ),
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=3,
        limit=3,
        bootstrap_samples=32,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("success_moat_generalizes_mmlu_pro_hard_")
    assert artifact["best_verifier_from"] == "D3"
    assert artifact["second_corpus"] == "MMLU-Pro-hard"
    assert artifact["headroom_present"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["oracle_distinctness_enforced"] is True
    assert artifact["second_corpus_accuracy"] == pytest.approx(1.0)
    assert artifact["genuine_tuned_sc_accuracy_second"] == pytest.approx(0.0)
    assert artifact["delta_vs_tuned_sc_second"] == pytest.approx(1.0)
    assert artifact["paired_ci95_second"] == [1.0, 1.0]
    assert artifact["n_questions"] == 3
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["non_degenerate"] is True
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5035_no_headroom_is_musr_scoped(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5035: no second-corpus headroom avoids a false null."""

    _write_d3_artifact(tmp_path, delta=0.08)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_loader=_candidate_loader(
            _no_headroom_rows(3), tmp_path / "mmlu_candidates.jsonl"
        ),
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=3,
        limit=3,
        bootstrap_samples=32,
        write=True,
    )

    assert artifact["honest_verdict"] == "complete_moat_musr_scoped_mmlu_pro_hard_no_confirm"
    assert artifact["headroom_present"] is False
    assert artifact["oracle_at_k_second"] == pytest.approx(1.0)
    assert artifact["genuine_tuned_sc_accuracy_second"] == pytest.approx(1.0)
    assert artifact["delta_vs_tuned_sc_second"] == pytest.approx(0.0)
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5035_schema_and_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5035: helper branches are deterministic and schema-visible."""

    assert mod._slug_corpus("MMLU-Pro-hard") == "mmlu_pro_hard"
    assert mod._ci_excludes_zero_positive([0.01, 0.2]) is True
    assert mod._ci_excludes_zero_positive([-0.01, 0.2]) is False
    assert mod._read_json_object(tmp_path / "missing.json") is None
    assert mod.candidate_cache_relative_path("GPQA").endswith(
        "experiment_5035_candidates_gpqa.jsonl"
    )
    assert mod.shared_b2_candidate_cache_relative_path("MMLU-Pro-hard").endswith(
        "experiment_5029_shared_logprob_candidate_cache_v2_mmlu_pro_hard.jsonl"
    )
    assert len(mod._candidate_cache_paths(tmp_path, "GPQA")) == 4

    shared_path = tmp_path / mod.shared_b2_candidate_cache_relative_path("MMLU-Pro-hard")
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    shared_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _headroom_rows(3)),
        encoding="utf-8",
    )
    rows, used_path = mod.default_candidate_rows_loader(
        root=tmp_path,
        corpus="MMLU-Pro-hard",
        corpus_rows=_mmlu_rows(3),
        candidate_cache_path=tmp_path / "unused.jsonl",
        limit=3,
        min_questions=3,
        k_candidates=5,
        random_seed=1,
        server_port=8919,
    )
    assert len(rows) == 3
    assert used_path == shared_path
    with pytest.raises(mod.SecondCorpusUnavailable):
        mod.default_candidate_rows_loader(
            root=tmp_path / "empty",
            corpus="MMLU-Pro-hard",
            corpus_rows=_mmlu_rows(3),
            candidate_cache_path=tmp_path / "unused.jsonl",
            limit=3,
            min_questions=3,
            k_candidates=5,
            random_seed=1,
            server_port=8919,
        )

    _write_d3_artifact(tmp_path, delta=0.08, abstention=0.9)
    _write_d2_artifact(tmp_path, delta=0.03)
    best, checks = mod.select_best_verifier(tmp_path)
    assert best is not None
    assert best.arm == "D2"
    assert checks[-1].available is False

    d2_eval = mod.evaluate_rows_with_verifier(
        _headroom_rows(3),
        verifier=best,
        seed=1,
        bootstrap_samples=32,
    )
    assert d2_eval["accuracy"] == pytest.approx(1.0)
    assert mod._d3_non_degenerate({"degeneracy_guard": {"degeneracy_flag": True}}) is False
    assert mod._d3_non_degenerate({"uncertainty_calibration": {"degeneracy_flag": True}}) is False
    assert mod._uprm_energy({"process_score": 0.25}) == pytest.approx(-0.25)
    with pytest.raises(mod.SecondCorpusUnavailable):
        mod._uprm_energy({})

    d1_eval = mod.evaluate_rows_with_verifier(
        [
            {
                **row,
                "candidates": [
                    {**row["candidates"][0], "trained_lora_ebm_energy": 1.0},
                    {**row["candidates"][1], "trained_lora_ebm_energy": 0.0},
                ],
            }
            for row in _headroom_rows(2)
        ],
        verifier=mod.VerifierSelection(
            arm="D1",
            scorer_kind="artifact_energy",
            delta_vs_tuned_sc=0.01,
            selection_accuracy=0.6,
            artifact_path=tmp_path / "d1.json",
            model_specs={},
        ),
        seed=1,
        bootstrap_samples=16,
    )
    assert d1_eval["accuracy"] == pytest.approx(1.0)

    no_candidate_attempt, no_candidate_checks = mod.select_second_corpus_attempt(
        root=tmp_path,
        verifier=best,
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_loader=lambda **_kwargs: ([], tmp_path / "empty.jsonl"),
        limit=3,
        min_questions=3,
        k_candidates=5,
        random_seed=1,
        server_port=8919,
        bootstrap_samples=32,
    )
    assert no_candidate_attempt is None
    assert no_candidate_checks[-1].resource == "candidate_cache_mmlu_pro_hard"

    fallback_rows, fallback_path = mod._candidate_loader_result(
        _headroom_rows(1), tmp_path / "fallback.jsonl"
    )
    assert len(fallback_rows) == 1
    assert fallback_path == tmp_path / "fallback.jsonl"

    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: False)
    oracle_attempt, oracle_checks = mod.select_second_corpus_attempt(
        root=tmp_path,
        verifier=best,
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_loader=_candidate_loader(_headroom_rows(3), tmp_path / "rows.jsonl"),
        limit=3,
        min_questions=3,
        k_candidates=5,
        random_seed=1,
        server_port=8919,
        bootstrap_samples=32,
    )
    assert oracle_attempt is None
    assert "OracleDistinctnessError" in oracle_checks[-1].detail

    blocked = mod.build_blocked_artifact(
        missing_resource="candidate_cache_unavailable",
        best_verifier=None,
        preconditions_checked=[],
        root=tmp_path,
        duration_s=0.0,
        blocked_error="missing",
    )
    assert blocked["blocked_error"] == "missing"
    assert mod.artifact_schema_errors(blocked) == []

    for mutated, field in (
        ({key: value for key, value in blocked.items() if key != "duration_s"}, "duration_s"),
        ({**blocked, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**blocked, "headroom_present": "no"}, "headroom_present"),
        ({**blocked, "paired_ci95_second": [0.0]}, "paired_ci95_second"),
        ({**blocked, "spec_refs": []}, "spec_refs"),
        ({**blocked, "field_principles": {}}, "field_principles"),
        ({**blocked, "genuine_tuned_sc_accuracy_second": 2.0}, "genuine_tuned_sc_accuracy_second"),
        ({**blocked, "delta_vs_tuned_sc_second": "0.1"}, "delta_vs_tuned_sc_second"),
        ({**blocked, "preconditions_checked": {}}, "preconditions_checked"),
        ({**blocked, "model_specs": []}, "model_specs"),
        ({**blocked, "honest_verdict": "maybe"}, "honest_verdict"),
    ):
        assert field in mod.artifact_schema_errors(mutated)

    assert mod._compact_adversarial_flags(
        {"reports": ["bad", {"flags": [{"kind": "WARN"}, "bad"]}]}
    ) == [{"kind": "WARN"}]
    assert mod._audit_is_clean({"flag_count": 0}) is True
    assert mod._audit_is_clean({"flagged_count": 1}) is False
    assert mod._audit_is_clean({"flags": [{"kind": "WARN"}]}) is False

    blocked_root = tmp_path / "blocked_second"
    _write_d2_artifact(blocked_root)
    second_blocked = mod.run(
        root=blocked_root,
        artifact_path=tmp_path / "blocked_second.json",
        corpus_loaders=[("MMLU-Pro-hard", lambda _limit: [])],
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=3,
        limit=3,
        write=True,
    )
    assert second_blocked["honest_verdict"] == "blocked_second_corpus_unavailable"
