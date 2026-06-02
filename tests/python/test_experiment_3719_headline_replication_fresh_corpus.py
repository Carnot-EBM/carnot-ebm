"""Tests for Exp 3719 fresh-corpus headline replication.

Spec refs: REQ-VERIFY-3719, SCENARIO-VERIFY-3719.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import headline_replication_fresh_corpus as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
REQUIRED_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "fresh_corpus_source",
    "fresh_corpus_auroc",
    "fresh_corpus_auroc_ci",
    "frozen_fover_auroc",
    "generalizes_beyond_fover",
    "n_seeds",
    "n_examples",
    "frozen_headline_unchanged_assert",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _exp235_fixture() -> dict[str, Any]:
    cases = [
        {
            "case_id": "gsm8k-a",
            "initial_correct": True,
            "correct": True,
            "history": [
                {"iteration": 0, "response": "1 + 1 = 2. Therefore the answer is 2."}
            ],
        },
        {
            "case_id": "gsm8k-b",
            "initial_correct": False,
            "correct": False,
            "history": [
                {"iteration": 0, "response": "2 + 2 = 5. Therefore the answer is 5."}
            ],
        },
        {
            "case_id": "gsm8k-c",
            "initial_correct": False,
            "correct": True,
            "history": [
                {"iteration": 1, "response": "3 + 3 = 6. Therefore the answer is 6."}
            ],
        },
        {
            "case_id": "gsm8k-d",
            "initial_correct": True,
            "correct": True,
            "history": [
                {"iteration": 0, "response": "4 + 4 = 9. Therefore the answer is 8."}
            ],
        },
    ]
    return {
        "paired_runs": [
            {
                "benchmark": "gsm8k_semantic",
                "mode": "verify_repair",
                "model_name": "Qwen3.5-0.8B",
                "cases": cases,
            }
        ]
    }


def _process_rows() -> list[dict[str, Any]]:
    base = {
        "benchmark": "gsm8k_semantic",
        "domain": "reasoning",
        "model": "Qwen3.5-0.8B",
        "source_artifact": "results/experiment_235_results.json",
        "source_experiment": 235,
    }
    return [
        {**base, "corpus_id": "row-a", "case_id": "gsm8k-a", "iteration": 0, "process_label": "clean"},
        {
            **base,
            "corpus_id": "row-b",
            "case_id": "gsm8k-b",
            "iteration": 0,
            "process_label": "wrong_answer_partially_sound_process",
        },
        {
            **base,
            "corpus_id": "row-c",
            "case_id": "gsm8k-c",
            "iteration": 1,
            "process_label": "repair_fixed_process_and_outcome",
        },
        {
            **base,
            "corpus_id": "row-d",
            "case_id": "gsm8k-d",
            "iteration": 0,
            "process_label": "right_answer_wrong_process",
        },
        {
            **base,
            "corpus_id": "row-code",
            "case_id": "human-eval-1",
            "iteration": 0,
            "domain": "code",
            "process_label": "unsupported_step",
        },
    ]


def _write_fresh_sources(root: Path) -> None:
    _write_json(root / "results" / "experiment_235_results.json", _exp235_fixture())
    _write_json(
        root / "results" / "experiment_248_results.json",
        {"corpus_path": "data/research/process_integrity_corpus_248.jsonl"},
    )
    _write_jsonl(root / "data" / "research" / "process_integrity_corpus_248.jsonl", _process_rows())
    _write_jsonl(
        root / "data" / "step_level_prm_training.jsonl",
        [{"question_id": "fover-derived", "partial_cot": "1 + 1 = 3", "step_label": "wrong"}],
    )
    _write_json(
        root / exp.EXP2850_REL_PATH,
        {"condition_a_production_auroc_mean": exp.FROZEN_FOVER_AUROC},
    )


def test_req_verify_3719_spec_anchor_exists() -> None:
    """REQ-VERIFY-3719: the fresh-corpus replication is OpenSpec anchored."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-3719" in spec
    assert "SCENARIO-VERIFY-3719" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "generalizes_beyond_fover" in spec


def test_scenario_verify_3719_assembles_distinct_process_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3719: Exp 235/248 rows become a non-FoVer corpus."""
    _write_fresh_sources(tmp_path)

    corpus = exp.assemble_fresh_corpus(tmp_path)

    assert corpus.blocked_reason is None
    assert corpus.source == exp.FRESH_CORPUS_SOURCE
    assert [row.corpus_id for row in corpus.rows] == ["row-a", "row-b", "row-c", "row-d"]
    assert [row.label for row in corpus.rows] == [0, 1, 0, 1]
    assert corpus.balance == {"correct": 2, "incorrect": 2}
    assert corpus.disqualified_sources[0]["reason"] == "fover_derived"


def test_scenario_verify_3719_blocks_without_distinct_corpus(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3719: FoVer-derived PRM rows alone do not count fresh."""
    _write_jsonl(
        tmp_path / "data" / "step_level_prm_training.jsonl",
        [
            {
                "question_id": "fover-1",
                "partial_cot": "1 + 1 = 3",
                "step_label": "wrong",
            }
        ],
    )

    artifact = exp.build_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / exp.OUTPUT_REL_PATH,
            started_at=10.0,
            clock=lambda: 12.0,
        )
    )

    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["fresh_corpus_auroc"] is None
    assert artifact["generalizes_beyond_fover"] is False
    assert artifact["adversarial_verify_clean"] is False


@pytest.mark.parametrize(
    ("seed_aurocs", "expected_verdict", "expected_generalizes"),
    [
        (
            [0.908, 0.911, 0.913, 0.915, 0.918],
            exp.GENERALIZES_VERDICT,
            True,
        ),
        (
            [0.39, 0.41, 0.43, 0.45, 0.47],
            exp.FOVER_SPECIFIC_VERDICT,
            False,
        ),
    ],
)
def test_scenario_verify_3719_parametrizes_honest_outcomes(
    seed_aurocs: list[float],
    expected_verdict: str,
    expected_generalizes: bool,
) -> None:
    """SCENARIO-VERIFY-3719: synthetic outcomes classify honestly."""
    corpus = exp.FreshCorpus(
        rows=[
            exp.FreshCorpusRow("a", "1 + 1 = 2", 0, "clean"),
            exp.FreshCorpusRow("b", "1 + 1 = 3", 1, "unsupported_step"),
        ],
        source="synthetic_processbench_style_fixture",
        source_paths=("fixture.jsonl",),
        source_sha256="fixture-sha",
        disqualified_sources=[],
    )
    seed_results = [
        exp.SeedScoreResult(
            seed=seed,
            auroc=auroc,
            n_examples=2,
            subset_sha256=f"subset-{seed}",
            per_verifier_auroc={
                "fr11_session_memory": 0.5,
                "tier0r_curry_howard": auroc,
                "tier0s_arithmetic_gap": 0.5,
                "tier0u_logical_consistency": 0.5,
            },
        )
        for seed, auroc in zip(exp.DEFAULT_RANDOM_SEEDS, seed_aurocs, strict=True)
    ]

    artifact = exp.build_artifact_from_seed_results(
        corpus=corpus,
        seed_results=seed_results,
        started_at=10.0,
        now=16.0,
        adversarial_verify_clean=True,
        frozen_headline_unchanged_assert=True,
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["generalizes_beyond_fover"] is expected_generalizes
    assert isinstance(artifact["generalizes_beyond_fover"], bool)
    assert artifact["frozen_fover_auroc"] == exp.FROZEN_FOVER_AUROC
    assert artifact["fresh_corpus_auroc"] != artifact["frozen_fover_auroc"]


def test_scenario_verify_3719_scores_seeded_subsets_with_injected_verifiers() -> None:
    """SCENARIO-VERIFY-3719: five seeded subsets produce AUROC rows and checksum."""
    rows = [
        exp.FreshCorpusRow(f"pos-{idx}", f"bad {idx}", 1, "unsupported_step")
        for idx in range(4)
    ] + [
        exp.FreshCorpusRow(f"neg-{idx}", f"good {idx}", 0, "clean")
        for idx in range(4)
    ]

    def score_texts(texts: list[str]) -> dict[str, list[float]]:
        return {
            "tier0r_curry_howard": [1.0 if text.startswith("bad") else 0.0 for text in texts],
            "tier0s_arithmetic_gap": [0.25 for _text in texts],
            "tier0u_logical_consistency": [0.0 for _text in texts],
        }

    seed_result = exp.score_fresh_seed(
        rows,
        seed=42,
        n_examples=6,
        text_scorer=score_texts,
        memory_scorer=lambda row: 0.0,
    )

    assert seed_result.auroc == 1.0
    assert seed_result.n_examples == 6
    assert seed_result.per_verifier_auroc["tier0r_curry_howard"] == 1.0
    assert seed_result.per_verifier_auroc["tier0s_arithmetic_gap"] == 0.5
    assert len(seed_result.subset_sha256) == 64


def test_scenario_verify_3719_build_artifact_success_with_injected_scoring(tmp_path: Path) -> None:
    """REQ-VERIFY-3719: build_artifact scores available rows without live inference."""
    _write_fresh_sources(tmp_path)

    def score_texts(texts: list[str]) -> dict[str, list[float]]:
        return {
            "tier0r_curry_howard": [1.0 if "5" in text or "9" in text else 0.0 for text in texts],
            "tier0s_arithmetic_gap": [0.5 for _text in texts],
            "tier0u_logical_consistency": [0.0 for _text in texts],
        }

    artifact = exp.build_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            random_seeds=(42,),
            n_examples=4,
            started_at=10.0,
            clock=lambda: 13.0,
        ),
        text_scorer=score_texts,
        memory_scorer=lambda row: 0.0,
        adversarial_verify_clean=True,
    )

    assert artifact["fresh_corpus_auroc"] is not None
    assert artifact["n_seeds"] == 1
    assert artifact["n_examples"] == 4
    assert artifact["frozen_headline_unchanged_assert"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE


def test_scenario_verify_3719_build_artifact_records_scoring_block(tmp_path: Path) -> None:
    """REQ-VERIFY-3719: scoring exceptions become blocked artifacts."""
    _write_fresh_sources(tmp_path)

    def broken_scorer(_texts: list[str]) -> dict[str, list[float]]:
        raise RuntimeError("synthetic scoring failure")

    artifact = exp.build_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            random_seeds=(42,),
            n_examples=4,
            started_at=10.0,
            clock=lambda: 11.0,
        ),
        text_scorer=broken_scorer,
        memory_scorer=lambda row: 0.0,
    )

    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["blocked_reason"] == "scoring_blocked"
    assert "synthetic scoring failure" in artifact["blocked_detail"]


def test_scenario_verify_3719_write_artifact_attaches_fake_adversarial_report(tmp_path: Path) -> None:
    """REQ-VERIFY-3719: write_artifact stores adversarial status in the JSON."""
    _write_fresh_sources(tmp_path)
    output = tmp_path / exp.OUTPUT_REL_PATH

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output,
            random_seeds=(42,),
            n_examples=4,
            started_at=10.0,
            clock=lambda: 12.0,
        ),
        adversarial_verify_runner=lambda _path: {"flag_count": 0, "flags": [], "returncode": 0},
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["adversarial_verify_clean"] is True
    assert persisted["acceptance_gate_passed"] is True
    assert exp.ExperimentConfig(repo_root=tmp_path).resolved_output_path() == tmp_path / exp.OUTPUT_REL_PATH


def test_scenario_verify_3719_run_adversarial_verify_parses_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-3719: adversarial subprocess JSON is normalized."""

    class GoodProc:
        stdout = json.dumps({"reports": [{"flag_count": 0, "flags": [], "loaded": True}]})
        stderr = ""
        returncode = 0

    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: GoodProc())
    report = exp.run_adversarial_verify(Path("artifact.json"))
    assert report["flag_count"] == 0
    assert report["returncode"] == 0


def test_scenario_verify_3719_run_adversarial_verify_handles_bad_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3719: invalid verifier output is represented as a warning."""

    class BadProc:
        stdout = "not-json"
        stderr = "boom"
        returncode = 2

    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: BadProc())
    report = exp.run_adversarial_verify(Path("artifact.json"))
    assert report["flag_count"] == 1
    assert report["flags"][0]["kind"] == "ADVERSARIAL_VERIFY_ERROR"


def test_scenario_verify_3719_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-3719: helper validation failures are deterministic."""
    assert exp._process_label_to_int("unknown") is None
    assert exp._disqualified_sources(tmp_path) == []
    assert exp.frozen_headline_still_unchanged(tmp_path) is False
    assert exp._fresh_memory_scorer(tmp_path)(exp.FreshCorpusRow("x", "1 + 1 = 2", 0, "clean")) == 0.0
    assert set(exp._score_text_verifiers(["1 + 1 = 2"])) == set(exp.VERIFIER_NAMES[1:])
    assert exp._seed_t_ci95([0.25]) == {"mean": 0.25, "low": 0.25, "high": 0.25}

    with pytest.raises(ValueError, match="missing verifier"):
        exp._assert_verifier_columns({}, 1)
    with pytest.raises(ValueError, match="returned 2 scores"):
        exp._assert_verifier_columns(
            {
                "tier0r_curry_howard": [0.0],
                "tier0s_arithmetic_gap": [0.0, 0.0],
                "tier0u_logical_consistency": [0.0],
            },
            1,
        )
    with pytest.raises(ValueError, match="class balance"):
        exp._select_balanced_rows([exp.FreshCorpusRow("a", "bad", 1, "unsupported_step")], seed=1, n_examples=2)
    with pytest.raises(ValueError, match="same length"):
        exp._compute_auroc([1], [])
    with pytest.raises(ValueError, match="both positive and negative"):
        exp._compute_auroc([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError, match="at least one value"):
        exp._seed_t_ci95([])
    with pytest.raises(ValueError, match="at least one seed"):
        exp.build_artifact_from_seed_results(
            corpus=exp.FreshCorpus([], "fixture", (), None, []),
            seed_results=[],
            started_at=0.0,
            now=1.0,
            adversarial_verify_clean=True,
            frozen_headline_unchanged_assert=True,
        )

    payload = exp.build_artifact_from_seed_results(
        corpus=exp.FreshCorpus(
            [
                exp.FreshCorpusRow("a", "good", 0, "clean"),
                exp.FreshCorpusRow("b", "bad", 1, "unsupported_step"),
            ],
            "fixture",
            (),
            None,
            [],
        ),
        seed_results=[
            exp.SeedScoreResult(
                seed=1,
                auroc=0.5,
                n_examples=2,
                subset_sha256="subset",
                per_verifier_auroc={name: 0.5 for name in exp.VERIFIER_NAMES},
            )
        ],
        started_at=0.0,
        now=1.0,
        adversarial_verify_clean=True,
        frozen_headline_unchanged_assert=True,
    )
    payload.pop("fresh_corpus_source")
    with pytest.raises(ValueError, match="missing required"):
        exp._validate_artifact(payload)
    payload["fresh_corpus_source"] = "fixture"
    payload["generalizes_beyond_fover"] = "false"
    with pytest.raises(ValueError, match="bare bool"):
        exp._validate_artifact(payload)
    payload["generalizes_beyond_fover"] = False
    payload["fresh_corpus_auroc"] = payload["frozen_fover_auroc"]
    with pytest.raises(ValueError, match="must not be copied"):
        exp._validate_artifact(payload)

    assert exp._report_clean({"flags": [{"severity": "info"}]}) is True
    assert exp._report_clean({"flags": [{"severity": "warn"}]}) is False
    compact = exp._compact_adversarial_report({"flags": [{"severity": "critical"}], "returncode": 9})
    assert compact["flag_count"] == 1

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp._read_json(bad_json)
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text("\n[]\n{\"ok\": true}\n", encoding="utf-8")
    assert exp._read_jsonl(jsonl) == [{"ok": True}]


def test_scenario_verify_3719_real_sources_are_assemblable_without_forced_success() -> None:
    """SCENARIO-VERIFY-3719: real sources assemble; verdict success is not hard-coded."""
    corpus = exp.assemble_fresh_corpus(REPO_ROOT)

    if corpus.blocked_reason is not None:
        assert corpus.blocked_reason == "no_distinct_process_corpus"
        return

    assert corpus.source == exp.FRESH_CORPUS_SOURCE
    assert len(corpus.rows) >= 30
    assert corpus.balance["correct"] > 0
    assert corpus.balance["incorrect"] > 0
