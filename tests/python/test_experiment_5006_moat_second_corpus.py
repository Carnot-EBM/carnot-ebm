"""Tests for Exp 5006 second-corpus moat generalization.

Spec refs: REQ-VERIFY-5006, SCENARIO-VERIFY-5006.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5006_moat_second_corpus as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_d3_artifact(root: Path, *, delta: float = 0.0, threshold: float = 0.0) -> None:
    _write_json(
        root / mod.D3_ARTIFACT_RELATIVE_PATH,
        {
            "experiment": "experiment_5005_ebrm_uncertainty_verifier",
            "verifier_is_oracle": False,
            "headroom_present": True,
            "ebrm_selection_accuracy": 0.58 + delta,
            "tuned_sc_accuracy": 0.58,
            "delta_vs_tuned_sc": delta,
            "uncertainty_calibration": {"selected_threshold": threshold},
            "base_scorer_refined": "registry_quality_ensemble",
            "model_specs": {"base_scorer": "registry_quality_ensemble"},
        },
    )


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


def _headroom_candidate_rows(n: int) -> list[dict[str, Any]]:
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
                        "base_reward": 0.0,
                    },
                    {
                        "candidate_id": f"{row['row_id']}/right",
                        "answer": "A",
                        "cache_index": 1,
                        "base_reward": 1.0,
                    },
                ],
            }
        )
    return rows


def _no_headroom_candidate_rows(n: int) -> list[dict[str, Any]]:
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
                        "base_reward": 1.0,
                    },
                    {
                        "candidate_id": f"{row['row_id']}/right1",
                        "answer": "A",
                        "cache_index": 1,
                        "base_reward": 0.9,
                    },
                ],
            }
        )
    return rows


def test_req_verify_5006_spec_declares_second_corpus_contract() -> None:
    """REQ-VERIFY-5006: OpenSpec anchors the cross-corpus artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5006",
        "SCENARIO-VERIFY-5006",
        "experiment_5006_moat_second_corpus.py",
        "results/experiment_5006_moat_second_corpus.json",
        "oracle@K - tuned_sc >= 0.10",
        "success_moat_generalizes_<corpus>_<delta>",
        "complete_moat_musr_scoped_<corpus>_no_confirm",
        "gemma-4-12B-it-GGUF",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5006_selects_best_usable_d_arm_and_fallback(tmp_path: Path) -> None:
    """REQ-VERIFY-5006: best verifier is chosen by numeric MuSR delta."""

    _write_json(
        tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH,
        {
            "verifier_is_oracle": False,
            "trained_scorer_accuracy": 0.62,
            "tuned_sc_accuracy": 0.58,
            "delta_vs_tuned_sc": 0.04,
            "model_specs": {"base": "d1"},
        },
    )
    _write_json(
        tmp_path / mod.D2_ARTIFACT_RELATIVE_PATH,
        {"verifier_is_oracle": False, "uprm_selection_accuracy": None, "delta_vs_tuned_sc": None},
    )
    _write_d3_artifact(tmp_path, delta=0.07)

    best, checks = mod.select_best_verifier(tmp_path)

    assert best.arm == "D3"
    assert best.delta_vs_tuned_sc == pytest.approx(0.07)
    assert best.fallback_used is False
    assert [check.resource for check in checks] == ["d1_verifier", "d2_verifier", "d3_verifier"]

    empty_best, empty_checks = mod.select_best_verifier(tmp_path / "empty")
    assert empty_best.arm == "cheap_proxy_control"
    assert empty_best.fallback_used is True
    assert any(check.resource == "cheap_proxy_fallback" for check in empty_checks)


def test_scenario_verify_5006_successful_second_corpus_generalization(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5006: headroom-present MMLU-Pro-hard can confirm a win."""

    _write_d3_artifact(tmp_path, delta=0.0, threshold=0.0)
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        corpus_loaders=[
            ("GPQA", lambda _limit: (_ for _ in ()).throw(mod.SecondCorpusUnavailable("no gpqa"))),
            ("MMLU-Pro-hard", _mmlu_rows),
        ],
        candidate_rows_builder=lambda **kwargs: _headroom_candidate_rows(
            len(kwargs["corpus_rows"])
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
    assert artifact["tuned_sc_accuracy_second"] == pytest.approx(0.0)
    assert artifact["delta_vs_tuned_sc_second"] == pytest.approx(1.0)
    assert artifact["paired_ci95_second"] == [1.0, 1.0]
    assert artifact["n_questions"] == 3
    assert artifact["adversarial_verify_clean"] is True
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5006_no_headroom_is_scoped_not_null(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5006: no headroom records MuSR-scoped outcome."""

    _write_d3_artifact(tmp_path, delta=0.0, threshold=0.0)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_builder=lambda **kwargs: _no_headroom_candidate_rows(
            len(kwargs["corpus_rows"])
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
    assert artifact["second_corpus"] == "MMLU-Pro-hard"
    assert artifact["oracle_at_k_second"] == pytest.approx(1.0)
    assert artifact["tuned_sc_accuracy_second"] == pytest.approx(1.0)
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5006_schema_and_blocked_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5006: missing second corpus and schema errors fail closed."""

    _write_d3_artifact(tmp_path, delta=0.0)
    blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        corpus_loaders=[("MMLU-Pro-hard", lambda _limit: [])],
        candidate_rows_builder=lambda **_kwargs: [],
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=3,
        limit=3,
        write=True,
    )

    assert blocked["honest_verdict"] == "blocked_second_corpus_unavailable"
    assert blocked["second_corpus"] is None
    assert blocked["verifier_is_oracle"] is False
    assert blocked["second_corpus_accuracy"] is None
    assert mod.artifact_schema_errors(blocked) == []

    assert "verifier_is_oracle" in mod.artifact_schema_errors(
        {**blocked, "verifier_is_oracle": True}
    )
    assert "paired_ci95_second" in mod.artifact_schema_errors(
        {**blocked, "paired_ci95_second": [0.0]}
    )
    assert "spec_refs" in mod.artifact_schema_errors(
        {**blocked, "spec_refs": ["REQ-VERIFY-5006"]}
    )
    assert "honest_verdict" in mod.artifact_schema_errors({**blocked, "honest_verdict": "maybe"})


def test_req_verify_5006_defensive_helpers_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5006: helper branches stay deterministic and fail closed."""

    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._number("nan") is None
    assert mod._candidate_mean_logprob({"mean_logprob": "-0.5"}) == pytest.approx(-0.5)
    assert mod._slug_corpus(None) == "none"
    assert mod._ci_excludes_zero_positive([0.01, 0.2]) is True
    assert mod._ci_excludes_zero_positive([-0.01, 0.2]) is False

    non_object_path = tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH
    _write_json(non_object_path, ["bad"])
    best, checks = mod.select_best_verifier(tmp_path)
    assert best.fallback_used is True
    assert checks[0].detail.endswith("is not a JSON object")

    jsonl_path = tmp_path / "rows.jsonl"
    mod._write_jsonl(jsonl_path, [{"a": 1}, {"b": 2}])
    jsonl_path.write_text(jsonl_path.read_text(encoding="utf-8") + "\n[]\n", encoding="utf-8")
    assert mod._read_jsonl(jsonl_path) == [{"a": 1}, {"b": 2}]

    reward_rows = mod.attach_quality_rewards(
        [
            {
                "row_id": "r",
                "gold": "A",
                "candidates": [
                    {
                        "candidate_id": "r/a",
                        "answer": "A",
                        "token_logprobs": [-0.2, "-0.4", "bad"],
                        "reasoning": "short reason",
                        "cache_index": 0,
                    }
                ],
            },
            {"row_id": "empty", "candidates": []},
        ]
    )
    assert len(reward_rows) == 1
    assert reward_rows[0]["candidates"][0]["base_reward"] > 0.0
    assert mod._cheap_proxy_energy({"answer": "A"}) < 0.0
    assert mod._uprm_energy({"uprm_process_score": 0.25}) == pytest.approx(-0.25)
    assert math.isinf(mod._uprm_energy({}))

    cheap_eval = mod.evaluate_rows_with_verifier(
        _headroom_candidate_rows(2),
        verifier=mod.VerifierSelection(
            arm="cheap_proxy_control",
            scorer_kind="cheap_proxy_quality",
            delta_vs_tuned_sc=0.0,
            selection_accuracy=None,
            artifact_path=None,
            model_specs={},
            fallback_used=True,
        ),
        seed=1,
        bootstrap_samples=16,
    )
    assert cheap_eval["accuracy"] == pytest.approx(1.0)

    uprm_rows = _headroom_candidate_rows(2)
    for row in uprm_rows:
        row["candidates"][0]["uprm_process_score"] = 0.0
        row["candidates"][1]["uprm_process_score"] = 1.0
    uprm_eval = mod.evaluate_rows_with_verifier(
        uprm_rows,
        verifier=mod.VerifierSelection(
            arm="D2",
            scorer_kind="uprm_process_score",
            delta_vs_tuned_sc=0.01,
            selection_accuracy=0.5,
            artifact_path=None,
            model_specs={},
        ),
        seed=1,
        bootstrap_samples=16,
    )
    assert uprm_eval["delta"] == pytest.approx(1.0)

    with pytest.raises(mod.SecondCorpusUnavailable):
        mod.evaluate_rows_with_verifier(
            _headroom_candidate_rows(1),
            verifier=mod.VerifierSelection(
                arm="D1",
                scorer_kind="unsupported",
                delta_vs_tuned_sc=0.1,
                selection_accuracy=0.7,
                artifact_path=None,
                model_specs={},
            ),
            seed=1,
            bootstrap_samples=8,
        )

    assert mod._compact_adversarial_flags({"reports": [{"flags": [{"kind": "WARN"}, "bad"]}]}) == [
        {"kind": "WARN"}
    ]
    assert mod._audit_is_clean({"max_severity": 0}) is True
    assert mod._audit_is_clean({"flagged_count": 0}) is True
    assert mod._audit_is_clean({"flags": [{"kind": "WARN"}]}) is False

    _write_d3_artifact(tmp_path, delta=0.0)
    base_blocked = mod.build_blocked_artifact(
        missing_resource="x",
        best_verifier=best,
        preconditions_checked=[],
        duration_s=0.0,
        error="boom",
    )
    assert base_blocked["blocked_error"] == "boom"
    for mutated, field in (
        ({key: value for key, value in base_blocked.items() if key != "duration_s"}, "duration_s"),
        ({**base_blocked, "headroom_present": "no"}, "headroom_present"),
        ({**base_blocked, "second_corpus_accuracy": 2.0}, "second_corpus_accuracy"),
        ({**base_blocked, "delta_vs_tuned_sc_second": "0.1"}, "delta_vs_tuned_sc_second"),
        ({**base_blocked, "mcnemar_p_second": 2.0}, "mcnemar_p_second"),
        ({**base_blocked, "preconditions_checked": {}}, "preconditions_checked"),
        ({**base_blocked, "model_specs": []}, "model_specs"),
        ({**base_blocked, "field_principles": {}}, "field_principles"),
    ):
        assert field in mod.artifact_schema_errors(mutated)

    generation_error = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "generation_error.json",
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_builder=lambda **_kwargs: [],
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=2,
        limit=2,
        write=True,
    )
    assert generation_error["honest_verdict"] == "blocked_candidate_generation_or_scoring_error"

    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: False)
    oracle_error = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "oracle_error.json",
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_builder=lambda **kwargs: _headroom_candidate_rows(len(kwargs["corpus_rows"])),
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=2,
        limit=2,
        write=True,
    )
    assert oracle_error["honest_verdict"] == "blocked_oracle_distinctness_violation"
