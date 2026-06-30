"""Tests for the Phase D shared moat benchmark harness.

Spec refs: REQ-KONA-5002, SCENARIO-KONA-5002-SMOKE,
SCENARIO-KONA-5002-ORACLE-DISTINCT, SCENARIO-KONA-5002-BLOCKED,
REQ-KONA-5015, SCENARIO-KONA-5015-GENUINE-SC,
SCENARIO-KONA-5015-DEGENERACY-GUARD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5002_moat_benchmark_harness as exp
from carnot import experiment_5015_genuine_sc_baseline_fix as exp5015
from carnot import moat_benchmark_harness as harness


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase3-kona" / "spec.md"


def _candidate(candidate_id: str, answer: str, energy: float, *, index: int = 0) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "answer": answer,
        "cache_index": index,
        "trivial_energy": energy,
    }


def _row(row_id: str, gold: str, answers: list[str], energies: list[float]) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "corpus": "synthetic",
        "question": f"question {row_id}",
        "context": "synthetic context",
        "choices": ["A", "B"],
        "gold": gold,
        "candidates": [
            _candidate(f"{row_id}-c{i}", answer, energies[i], index=i)
            for i, answer in enumerate(answers)
        ],
    }


def _synthetic_rows() -> list[dict[str, Any]]:
    return [
        _row("r1", "A", ["B", "B", "A"], [0.9, 0.8, 0.1]),
        _row("r2", "B", ["B", "A", "A"], [0.1, 0.8, 0.9]),
        _row("r3", "A", ["A", "B", "B"], [0.1, 0.8, 0.9]),
        _row("r4", "B", ["A", "B", "B"], [0.9, 0.1, 0.2]),
    ]


def _genuine_sc_rows() -> list[dict[str, Any]]:
    return [
        _row("g1", "A", ["B", "A", "A"], [0.9, 0.1, 0.2]),
        _row("g2", "B", ["A", "B", "B"], [0.9, 0.1, 0.2]),
        _row("g3", "A", ["A", "A", "B"], [0.1, 0.2, 0.9]),
        _row("g4", "B", ["B", "B", "A"], [0.1, 0.2, 0.9]),
        _row("g5", "A", ["B", "A", "B"], [0.9, 0.1, 0.2]),
    ]


def _write_checkpoint(path: Path, index: int, *, gold: str, answers: list[str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "q": index,
        "gold": gold,
        "answers": answers,
        "sc_answer": answers[0],
        "energy_answer": gold,
        "energy_pure_answer": gold,
    }
    target = path / f"q{index:04d}.json"
    target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return target


def test_req_kona_5002_spec_declares_shared_harness_contract() -> None:
    """REQ-KONA-5002: OpenSpec anchors paths, metrics, and guardrails."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-KONA-5002")
    end = spec.index("## Latent Symbol Bridge Falsification", start)
    section = spec[start:end]

    for marker in (
        "REQ-KONA-5002",
        "SCENARIO-KONA-5002-SMOKE",
        "SCENARIO-KONA-5002-ORACLE-DISTINCT",
        "SCENARIO-KONA-5002-BLOCKED",
        exp.HARNESS_MODULE_PATH,
        exp.MODULE_RELATIVE_PATH,
        exp.RESULT_RELATIVE_PATH,
        "tuned self-consistency",
        "oracle@K",
        "paired bootstrap CI95",
        "McNemar",
        "gold",
        "answer_index",
        "answer_choice",
        "model_id",
    ):
        assert marker in section
    for field, principle in exp.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_req_kona_5015_spec_declares_genuine_sc_and_degeneracy_guard() -> None:
    """REQ-KONA-5015: OpenSpec anchors the genuine SC baseline correction."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-KONA-5015")
    end = spec.index("## Latent Symbol Bridge Falsification", start)
    section = spec[start:end]

    for marker in (
        "REQ-KONA-5015",
        "SCENARIO-KONA-5015-GENUINE-SC",
        "SCENARIO-KONA-5015-DEGENERACY-GUARD",
        "SCENARIO-KONA-5015-SMOKE",
        "python/carnot/moat_benchmark_harness.py",
        "experiment_5015_genuine_sc_baseline_fix.py",
        "results/experiment_5015_genuine_sc_baseline_fix.json",
        "{1,3,5,7,...}",
        "abstain_rate > 0.50",
        "degeneracy_flag=true",
        "genuine_tuned_sc_accuracy",
        "sc_k_sweep",
        "tuned_k",
        "oracle_at_k",
    ):
        assert marker in section


def test_scenario_kona_5002_metrics_compute_tuned_sc_oracle_and_ci() -> None:
    """SCENARIO-KONA-5002-SMOKE: metrics use tuned SC, oracle@K, CI, and McNemar."""

    metrics = harness.evaluate_verifier(
        _synthetic_rows(),
        scorer=lambda candidate: float(candidate["trivial_energy"]),
        seed=123,
        bootstrap_samples=200,
    )

    assert metrics["n_rows"] == 4
    assert metrics["tuned_self_consistency"]["accuracy"] == pytest.approx(0.5)
    assert metrics["tuned_self_consistency"]["config"]["k"] == 1
    assert metrics["oracle_at_k"] == pytest.approx(1.0)
    assert metrics["n_flips_possible"] == 2
    assert metrics["headroom_present"] is True
    assert metrics["verifier"]["accuracy"] == pytest.approx(1.0)
    assert metrics["verifier_minus_tuned_sc_delta"] == pytest.approx(0.5)
    low, high = metrics["verifier_minus_tuned_sc_ci95"]
    assert low <= metrics["verifier_minus_tuned_sc_delta"] <= high
    assert 0.0 <= metrics["mcnemar_p"] <= 1.0


def test_scenario_kona_5015_genuine_sc_sweeps_odd_k_and_recomputes_headroom() -> None:
    """SCENARIO-KONA-5015-GENUINE-SC: K-way SC reports the full odd-K sweep."""

    tuned = harness.tuned_self_consistency(_genuine_sc_rows())
    metrics = harness.evaluate_verifier(
        _genuine_sc_rows(),
        scorer=lambda candidate: float(candidate["trivial_energy"]),
        seed=123,
        bootstrap_samples=64,
    )

    assert tuned["k_sweep"] == {"1": 0.4, "3": 0.8}
    assert tuned["config"]["k"] == 3
    assert tuned["accuracy"] == pytest.approx(0.8)
    assert tuned["candidates_per_question"] == 3
    assert tuned["degenerate_candidate_pool"] is False
    assert metrics["tuned_self_consistency"]["k_sweep"] == {"1": 0.4, "3": 0.8}
    assert metrics["oracle_at_k"] == pytest.approx(1.0)
    assert metrics["n_flips_possible"] == 1
    assert metrics["headroom_present"] is True


def test_scenario_kona_5015_single_candidate_pool_is_flagged_degenerate() -> None:
    """REQ-KONA-5015: single-candidate SC/oracle@K headroom is flagged honestly."""

    rows = [_row("s1", "A", ["A"], [0.1]), _row("s2", "B", ["A"], [0.1])]

    tuned = harness.tuned_self_consistency(rows)

    assert tuned["k_sweep"] == {"1": 0.5}
    assert tuned["config"]["k"] == 1
    assert tuned["candidates_per_question"] == 1
    assert tuned["degenerate_candidate_pool"] is True
    assert tuned["oracle_degenerate"] is True


def test_scenario_kona_5015_abstention_degeneracy_guard_flags_majority_abstain() -> None:
    """SCENARIO-KONA-5015-DEGENERACY-GUARD: >50% abstention is uninformative."""

    flagged = harness.abstention_degeneracy_guard(0.975)
    unflagged = harness.abstention_degeneracy_guard(0.5)

    assert flagged["degeneracy_flag"] is True
    assert flagged["verdict"].startswith("degenerate_abstaining_selector_")
    assert flagged["abstain_rate"] == pytest.approx(0.975)
    assert unflagged["degeneracy_flag"] is False
    assert unflagged["verdict"] == "nondegenerate_abstaining_selector"


def test_scenario_kona_5015_experiment_artifact_success_and_blocked(tmp_path: Path) -> None:
    """SCENARIO-KONA-5015-SMOKE: Exp 5015 re-emits the corrected MuSR baseline."""

    checkpoint_dir = tmp_path / "ckpt"
    for index, row in enumerate(_genuine_sc_rows()):
        _write_checkpoint(
            checkpoint_dir,
            index,
            gold=str(row["gold"]),
            answers=[str(candidate["answer"]) for candidate in row["candidates"]],
        )

    artifact = exp5015.build_artifact(
        checkpoint_dir=checkpoint_dir,
        smoke_limit=5,
        now=lambda: 0.0,
    )
    result_path = tmp_path / "result.json"
    written = exp5015.main(
        result_path=result_path,
        checkpoint_dir=checkpoint_dir,
        smoke_limit=5,
    )
    blocked = exp5015.build_artifact(
        checkpoint_dir=tmp_path / "missing",
        smoke_limit=5,
        now=lambda: 0.0,
    )

    exp5015.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp5015.HONEST_VERDICT
    assert artifact["genuine_tuned_sc_accuracy"] == pytest.approx(0.8)
    assert artifact["sc_k_sweep"] == {"1": 0.4, "3": 0.8}
    assert artifact["tuned_k"] == 3
    assert artifact["candidates_per_question"] == 3
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["genuine_headroom_present"] is True
    assert artifact["degeneracy_guard_fires"] is True
    assert artifact["corrected_musr_tuned_sc_baseline"]["k_sweep"] == {"1": 0.4, "3": 0.8}
    assert json.loads(result_path.read_text(encoding="utf-8")) == written
    assert blocked["honest_verdict"] == "blocked_candidate_cache_missing"
    assert blocked["sc_k_sweep"] == {}
    assert blocked["preconditions_checked"]["candidate_cache_present"] is False


def test_scenario_kona_5002_oracle_distinctness_blocks_gold_and_model_reads() -> None:
    """SCENARIO-KONA-5002-ORACLE-DISTINCT: forbidden candidate keys raise."""

    rows = _synthetic_rows()
    rows[0]["candidates"][0]["gold"] = rows[0]["gold"]
    rows[0]["candidates"][0]["model_id"] = "generator-a"

    with pytest.raises(harness.OracleDistinctnessError, match="gold"):
        harness.evaluate_verifier(rows, scorer=lambda candidate: candidate["gold"])
    with pytest.raises(harness.OracleDistinctnessError, match="model_id"):
        harness.evaluate_verifier(rows, scorer=lambda candidate: candidate.get("model_id"))


def test_req_kona_5002_musr_candidate_cache_reuses_checkpoint_answers(tmp_path: Path) -> None:
    """REQ-KONA-5002: MuSR checkpoint answers become reusable candidate pools."""

    checkpoint_dir = tmp_path / "ckpt"
    _write_checkpoint(checkpoint_dir, 0, gold="A", answers=["B", "A", None])
    corpus_rows = [
        {
            "row_id": "musr-0",
            "corpus": "MuSR/murder_mysteries",
            "question": "Who did it?",
            "context": "A short mystery.",
            "choices": ["A", "B"],
            "gold": "A",
        }
    ]

    rows = harness.attach_musr_cached_candidates(
        corpus_rows,
        checkpoint_dir=checkpoint_dir,
        limit=1,
    )

    assert rows[0]["row_id"] == "musr-0"
    assert [candidate["answer"] for candidate in rows[0]["candidates"]] == ["B", "A"]
    assert rows[0]["candidates"][1]["cached_energy_selected"] is True
    assert rows[0]["candidate_cache_path"].endswith("q0000.json")


def test_req_kona_5002_generation_path_records_logprobs_without_running_llm() -> None:
    """REQ-KONA-5002: fresh-generation path supports per-token logprobs for D arms."""

    calls: list[dict[str, Any]] = []

    def fake_generator(
        prompt: str, *, seed: int, config: harness.GenerationConfig
    ) -> dict[str, Any]:
        calls.append({"prompt": prompt, "seed": seed, "model": config.model})
        return {
            "text": "Reasoning... ANSWER: A",
            "token_logprobs": [-0.1, -0.2],
            "mean_logprob": -0.15,
        }

    row = _synthetic_rows()[0] | {"candidates": []}
    config = harness.GenerationConfig(k=2)
    candidates = harness.generate_candidates_with_logprobs(
        row,
        generator=fake_generator,
        config=config,
        seed=77,
    )

    assert config.model == "gemma-4-12B-it-GGUF"
    assert config.gpu == 0
    assert len(candidates) == 2
    assert candidates[0]["answer"] == "A"
    assert candidates[0]["token_logprobs"] == [-0.1, -0.2]
    assert calls[0]["model"] == "gemma-4-12B-it-GGUF"
    assert "ANSWER: <one choice verbatim>" in calls[0]["prompt"]


def test_scenario_kona_5002_experiment_artifact_success_and_blocked(tmp_path: Path) -> None:
    """SCENARIO-KONA-5002-SMOKE/BLOCKED: artifact gates success and missing resources."""

    checkpoint_dir = tmp_path / "results" / "distributional_energy_verifier_musr_checkpoints"
    for index, row in enumerate(_synthetic_rows()):
        _write_checkpoint(
            checkpoint_dir,
            index,
            gold=str(row["gold"]),
            answers=[str(candidate["answer"]) for candidate in row["candidates"]],
        )
    corpus_rows = [
        {key: row[key] for key in ("row_id", "corpus", "question", "context", "choices", "gold")}
        for row in _synthetic_rows()
    ]

    artifact = exp.build_artifact(
        repo_root=REPO,
        corpus_loader=lambda limit: corpus_rows[:limit],
        checkpoint_dir=checkpoint_dir,
        smoke_limit=4,
        bootstrap_samples=200,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp.HONEST_VERDICT
    assert artifact["harness_module_path"] == exp.HARNESS_MODULE_PATH
    assert artifact["tuned_sc_smoke"] == pytest.approx(0.5)
    assert artifact["oracle_at_k_smoke"] == pytest.approx(1.0)
    assert artifact["headroom_present_smoke"] is True
    assert artifact["oracle_distinctness_enforced"] is True
    assert artifact["smoke_metrics"]["verifier"]["accuracy"] == pytest.approx(1.0)

    blocked = exp.build_artifact(
        repo_root=REPO,
        corpus_loader=lambda limit: corpus_rows[:limit],
        checkpoint_dir=tmp_path / "missing",
        smoke_limit=4,
        bootstrap_samples=20,
    )
    exp.validate_artifact(blocked)
    assert blocked["honest_verdict"] == "blocked_candidate_cache_missing"
    assert blocked["preconditions_checked"]["candidate_cache_present"] is False
    assert blocked["smoke_metrics"] == {}


def test_req_kona_5002_main_writes_artifact(tmp_path: Path) -> None:
    """REQ-KONA-5002: main writes the supplied result path."""

    checkpoint_dir = tmp_path / "ckpt"
    for index, row in enumerate(_synthetic_rows()):
        _write_checkpoint(
            checkpoint_dir,
            index,
            gold=str(row["gold"]),
            answers=[str(candidate["answer"]) for candidate in row["candidates"]],
        )
    corpus_rows = [
        {key: row[key] for key in ("row_id", "corpus", "question", "context", "choices", "gold")}
        for row in _synthetic_rows()
    ]
    result_path = tmp_path / "artifact.json"

    artifact = exp.main(
        repo_root=REPO,
        result_path=result_path,
        corpus_loader=lambda limit: corpus_rows[:limit],
        checkpoint_dir=checkpoint_dir,
        smoke_limit=4,
        bootstrap_samples=200,
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
