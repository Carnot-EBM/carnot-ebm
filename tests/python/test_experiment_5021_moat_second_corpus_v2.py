"""Tests for Exp 5021 second-corpus moat generalization v2.

Spec refs: REQ-VERIFY-5021, SCENARIO-VERIFY-5021.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5021_moat_second_corpus_v2 as mod


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
                        "uprm_process_score": 0.0,
                    },
                    {
                        "candidate_id": f"{row['row_id']}/right",
                        "answer": "A",
                        "cache_index": 1,
                        "temperature": "cached",
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
                        "uprm_process_score": 1.0,
                    },
                    {
                        "candidate_id": f"{row['row_id']}/right1",
                        "answer": "A",
                        "cache_index": 1,
                        "temperature": "cached",
                        "uprm_process_score": 0.9,
                    },
                ],
            }
        )
    return rows


def _ebrm_rows(n: int) -> list[dict[str, Any]]:
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
                    },
                    {
                        "candidate_id": f"{row['row_id']}/right",
                        "answer": "A",
                        "cache_index": 1,
                        "temperature": "cached",
                        "cached_energy_selected": True,
                    },
                ],
            }
        )
    return rows


def _write_d2_artifact(root: Path, *, delta: float = 0.04) -> None:
    _write_json(
        root / mod.D2_ARTIFACT_RELATIVE_PATH,
        {
            "experiment": "experiment_5018_uprm_replication_v2",
            "verifier_is_oracle": False,
            "headroom_present": True,
            "uprm_selection_accuracy": 0.62,
            "genuine_tuned_sc_accuracy": 0.62 - delta,
            "delta_vs_tuned_sc": delta,
            "scoring_path": "uprm_logprob",
            "model_specs": {"cached_candidate_generator": "gemma-4-12B-it-GGUF"},
        },
    )


def test_req_verify_5021_spec_declares_v2_cross_corpus_contract() -> None:
    """REQ-VERIFY-5021: OpenSpec anchors the v2 second-corpus contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5021",
        "SCENARIO-VERIFY-5021",
        "experiment_5021_moat_second_corpus_v2.py",
        "results/experiment_5021_moat_second_corpus_v2.json",
        "blocked_no_best_verifier",
        "oracle@K - genuine tuned-SC >= 0.10",
        "success_moat_generalizes_<corpus>_<delta>",
        "complete_moat_musr_scoped_<corpus>_no_confirm",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_verify_5021_all_d_arms_blocked_does_not_use_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5021: all blocked D arms emit blocked_no_best_verifier."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_loader=lambda **_kwargs: _headroom_rows(3),
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
    assert "cheap" not in json.dumps(artifact).lower()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5021_successful_second_corpus_generalization(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5021: headroom-present MMLU-Pro-hard can confirm a win."""

    _write_json(
        tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH,
        {
            "verifier_is_oracle": False,
            "scorer_trained": True,
            "trained_scorer_accuracy": 0.58,
            "delta_vs_tuned_sc": 0.01,
            "model_specs": {"base_model": "Qwen/Qwen3.5-1.7B"},
        },
    )
    _write_d2_artifact(tmp_path, delta=0.07)
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        corpus_loaders=[
            ("GPQA", lambda _limit: (_ for _ in ()).throw(mod.SecondCorpusUnavailable("no gpqa"))),
            ("MMLU-Pro-hard", _mmlu_rows),
        ],
        candidate_rows_loader=lambda **kwargs: _headroom_rows(len(kwargs["corpus_rows"])),
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=3,
        limit=3,
        bootstrap_samples=32,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("success_moat_generalizes_mmlu_pro_hard_")
    assert artifact["best_verifier_from"] == "D2"
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


def test_scenario_verify_5021_no_headroom_is_musr_scoped(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5021: no second-corpus headroom avoids a false null."""

    _write_d2_artifact(tmp_path, delta=0.07)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_loader=lambda **kwargs: _no_headroom_rows(len(kwargs["corpus_rows"])),
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


def test_req_verify_5021_schema_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5021: helper branches are deterministic and schema-visible."""

    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._number("nan") is None
    assert mod._slug_corpus("MMLU-Pro-hard") == "mmlu_pro_hard"
    assert mod._ci_excludes_zero_positive([0.01, 0.2]) is True
    assert mod._ci_excludes_zero_positive([-0.01, 0.2]) is False
    assert mod.legacy_candidate_cache_relative_path("MMLU-Pro-hard").endswith(
        "experiment_5006_candidates_mmlu_pro_hard.jsonl"
    )
    assert mod.shared_cache_candidate_relative_path("GPQA").endswith(
        "experiment_5016_shared_logprob_candidate_cache_gpqa.jsonl"
    )
    assert len(mod._candidate_cache_paths(tmp_path, "GPQA")) == 3

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

    invalid_json = tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH
    invalid_json.parent.mkdir(parents=True, exist_ok=True)
    invalid_json.write_text("{", encoding="utf-8")
    assert mod._read_json_object(invalid_json) is None
    unusable_root = tmp_path / "unusable"
    _write_json(
        unusable_root / mod.D1_ARTIFACT_RELATIVE_PATH,
        {
            "verifier_is_oracle": False,
            "scorer_trained": False,
            "trained_scorer_accuracy": 0.4,
            "delta_vs_tuned_sc": 0.01,
        },
    )
    unusable_best, unusable_checks = mod.select_best_verifier(unusable_root)
    assert unusable_best is None
    assert unusable_checks[0].available is False

    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('\n{"a": 1}\nnot json\n[]\n{"b": 2}\n', encoding="utf-8")
    assert mod._read_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._read_jsonl(jsonl_path) == [{"a": 1}, {"b": 2}]

    assert mod._finite_candidate_energy({"trained_lora_ebm_energy": 0.2}) == pytest.approx(0.2)
    assert mod._finite_candidate_energy({"base_reward": 0.7}) == pytest.approx(-0.7)
    assert mod._finite_candidate_energy({"mean_logprob": -0.3}) == pytest.approx(0.3)
    assert mod._finite_candidate_energy({"cache_index": 4}) == pytest.approx(1.004)
    with pytest.raises(mod.SecondCorpusUnavailable):
        mod._uprm_energy({})
    artifact_rows = _headroom_rows(2)
    for row in artifact_rows:
        row["candidates"][0]["trained_lora_ebm_energy"] = 1.0
        row["candidates"][1]["trained_lora_ebm_energy"] = 0.0
    artifact_eval = mod.evaluate_rows_with_verifier(
        artifact_rows,
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
    assert artifact_eval["accuracy"] == pytest.approx(1.0)

    with pytest.raises(mod.SecondCorpusUnavailable):
        mod._load_corpus_rows(lambda _limit: _mmlu_rows(1), limit=1, min_questions=2)


def test_req_verify_5021_default_cache_loader_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5021: cache, D3, and oracle-distinct error paths are covered."""

    legacy_path = tmp_path / mod.legacy_candidate_cache_relative_path("MMLU-Pro-hard")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _headroom_rows(3)),
        encoding="utf-8",
    )
    loaded = mod.default_candidate_rows_loader(
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
    assert len(loaded) == 3
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

    _write_json(
        tmp_path / mod.D3_ARTIFACT_RELATIVE_PATH,
        {
            "verifier_is_oracle": False,
            "ebrm_selection_accuracy": 0.7,
            "delta_vs_tuned_sc": 0.08,
            "uncertainty_calibration": {"selected_threshold": 1.0},
            "model_specs": {"base_scorer": "fixture"},
        },
    )
    best, checks = mod.select_best_verifier(tmp_path)
    assert best is not None
    assert best.arm == "D3"
    assert best.ebrm_threshold == pytest.approx(1.0)
    assert checks[-1].available is True

    ebrm_eval = mod.evaluate_rows_with_verifier(
        _ebrm_rows(3),
        verifier=best,
        seed=1,
        bootstrap_samples=32,
    )
    assert ebrm_eval["accuracy"] == pytest.approx(1.0)
    assert ebrm_eval["headroom_present"] is True

    no_candidate_attempt, no_candidate_checks = mod.select_second_corpus_attempt(
        root=tmp_path,
        verifier=best,
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_loader=lambda **_kwargs: [],
        limit=3,
        min_questions=3,
        k_candidates=5,
        random_seed=1,
        server_port=8919,
        bootstrap_samples=32,
    )
    assert no_candidate_attempt is None
    assert no_candidate_checks[-1].resource == "candidate_cache_mmlu_pro_hard"

    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: False)
    oracle_attempt, oracle_checks = mod.select_second_corpus_attempt(
        root=tmp_path,
        verifier=best,
        corpus_loaders=[("MMLU-Pro-hard", _mmlu_rows)],
        candidate_rows_loader=lambda **_kwargs: _headroom_rows(3),
        limit=3,
        min_questions=3,
        k_candidates=5,
        random_seed=1,
        server_port=8919,
        bootstrap_samples=32,
    )
    assert oracle_attempt is None
    assert "OracleDistinctnessError" in oracle_checks[-1].detail

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
