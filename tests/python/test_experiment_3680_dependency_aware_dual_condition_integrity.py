"""Tests for Exp 3680 dependency-aware dual-condition integrity.

Spec: REQ-VERIFY-3680, SCENARIO-VERIFY-3680.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.eval.fover_memory_leakage_v3 import (
    CONDITION_ARCHITECTURE_ONLY,
    CONDITION_PRODUCTION,
    ConditionScoringError,
)
from carnot.verify import dependency_aware_dual_condition_integrity as exp3680
from carnot.verify import weaver_peer_comparison_v3 as exp3644


def _condition_rows(
    seed: int,
    *,
    n_examples: int = 1000,
) -> exp3680.ConditionScoreRows:
    """Build a FoVer-like dual-condition fixture with one useful anti-signal axis."""

    rng = np.random.default_rng(seed)
    half = n_examples // 2
    labels = np.asarray([0] * half + [1] * half, dtype=np.int64)
    direction = labels * 2 - 1
    shared = rng.normal(0.0, 0.04, n_examples)
    production_columns = [
        np.clip(0.50 + 0.09 * direction + shared + rng.normal(0.0, 0.16, n_examples), 0.0, 1.0),
        np.clip(0.50 + 0.08 * direction + shared + rng.normal(0.0, 0.16, n_examples), 0.0, 1.0),
        np.clip(0.50 - 0.15 * direction + rng.normal(0.0, 0.22, n_examples), 0.0, 1.0),
        np.clip(0.50 + 0.03 * direction + rng.normal(0.0, 0.18, n_examples), 0.0, 1.0),
    ]
    architecture_columns = [
        np.zeros(n_examples, dtype=np.float64),
        production_columns[1],
        production_columns[2],
        production_columns[3],
    ]
    return exp3680.ConditionScoreRows(
        seed=seed,
        labels=labels.tolist(),
        production_scores_by_verifier={
            name: column.tolist()
            for name, column in zip(exp3644.VERIFIER_NAMES, production_columns, strict=True)
        },
        architecture_scores_by_verifier={
            name: column.tolist()
            for name, column in zip(exp3644.VERIFIER_NAMES, architecture_columns, strict=True)
        },
        production_state_visible_count=3,
        architecture_state_visible_count=0,
        subset_sha256=f"subset-{seed}",
    )


@pytest.mark.parametrize(
    (
        "blocked",
        "dependency_aware_auroc",
        "delta_ci",
        "delong_p",
        "adversarial_clean",
        "leak_free",
        "expected_category",
        "expected_bool",
    ),
    [
        pytest.param(
            False,
            0.94,
            {"point": 0.03, "ci95": [0.01, 0.05]},
            0.01,
            True,
            True,
            "g1_rigor_confirmed",
            True,
            id="g1_rigor_confirmed",
        ),
        pytest.param(
            False,
            0.92,
            {"point": 0.01, "ci95": [-0.02, 0.04]},
            0.20,
            True,
            True,
            "no_significant_gain_under_protocol",
            False,
            id="no_significant_gain_under_protocol",
        ),
        pytest.param(
            True,
            None,
            None,
            None,
            False,
            False,
            "blocked",
            False,
            id="blocked",
        ),
    ],
)
def test_exp3680_classifies_honest_outcomes_without_single_success_string(
    blocked: bool,
    dependency_aware_auroc: float | None,
    delta_ci: dict[str, object] | None,
    delong_p: float | None,
    adversarial_clean: bool,
    leak_free: bool,
    expected_category: str,
    expected_bool: bool,
) -> None:
    """SCENARIO-VERIFY-3680: anti-poison outcomes include win, null, and blocked."""

    classification = exp3680.classify_outcome(
        blocked=blocked,
        dependency_aware_auroc=dependency_aware_auroc,
        frozen_headline_auroc=exp3680.FROZEN_HEADLINE_AUROC,
        delta_ci=delta_ci,
        delong_p=delong_p,
        adversarial_verify_clean=adversarial_clean,
        leak_free=leak_free,
    )

    assert classification.category == expected_category
    assert classification.terminal_verdict in exp3680.TERMINAL_VERDICTS
    assert classification.dependency_aware_g1_rigor_confirmed is expected_bool


def test_exp3680_builds_g1_dual_condition_artifact_from_synthetic_scores() -> None:
    """SCENARIO-VERIFY-3680: both weightings use identical production rows."""

    rows = [_condition_rows(seed) for seed in (42, 137, 271, 314, 1729)]
    artifact = exp3680.build_artifact_from_condition_rows(
        rows,
        started_s=0.0,
        now_s=8.0,
        bootstrap_seeds=(42, 137, 271, 314, 1729),
        n_bootstrap=8,
        adversarial_verify_clean=True,
    )

    exp3680.validate_artifact(artifact)
    assert set(exp3680.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3680.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3680.INFERENCE_SUBSTRATE
    assert artifact["frozen_headline_auroc"] == exp3680.FROZEN_HEADLINE_AUROC
    assert artifact["n_seeds"] == 5
    assert artifact["n_examples"] == 1000
    assert type(artifact["dependency_aware_g1_rigor_confirmed"]) is bool
    assert artifact["production_auroc_dependency_aware"] > artifact["production_auroc_carnot_current"]
    assert artifact["production_auroc_ci"]["ci95"][0] <= artifact["production_auroc_ci"]["point"]
    assert artifact["production_auroc_ci"]["point"] <= artifact["production_auroc_ci"]["ci95"][1]
    assert artifact["dependency_vs_carnot_delta_ci"]["ci95"][0] <= artifact[
        "dependency_vs_carnot_delta_ci"
    ]["point"]
    assert artifact["dependency_vs_carnot_delta_ci"]["point"] <= artifact[
        "dependency_vs_carnot_delta_ci"
    ]["ci95"][1]
    assert 0.0 <= artifact["delong_p_dependency_vs_carnot"] <= 1.0
    assert artifact["acceptance_gate"]["passed"] is True
    assert artifact["verifier_scoring_input_candidate_only"] is True
    assert artifact["leak_free"] is True
    assert len(artifact["per_seed_results"]) == 5
    assert all(row["architecture_state_visible_count"] == 0 for row in artifact["per_seed_results"])


def test_exp3680_leak_guard_rejects_ceiling_auroc() -> None:
    """REQ-VERIFY-3680: AUROC >=0.99 on n>=1000 is not leak-free."""

    assert exp3680.compute_leak_free(
        verifier_scoring_input_candidate_only=True,
        production_auroc_dependency_aware=None,
        n_examples=1000,
    ) is False
    assert exp3680.compute_leak_free(
        verifier_scoring_input_candidate_only=True,
        production_auroc_dependency_aware=0.9899,
        n_examples=1000,
    ) is True
    assert exp3680.compute_leak_free(
        verifier_scoring_input_candidate_only=True,
        production_auroc_dependency_aware=0.99,
        n_examples=1000,
    ) is False
    assert exp3680.compute_leak_free(
        verifier_scoring_input_candidate_only=False,
        production_auroc_dependency_aware=0.80,
        n_examples=1000,
    ) is False


def test_exp3680_blocks_when_preconditions_are_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-3680: missing corpus/dependency/G1 inputs writes blocked artifact."""

    artifact = exp3680.build_artifact(tmp_path, started_s=0.0, now_s=0.25)

    exp3680.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3680.BLOCKED_VERDICT
    assert artifact["dependency_aware_g1_rigor_confirmed"] is False
    assert artifact["production_auroc_dependency_aware"] is None
    assert artifact["production_auroc_carnot_current"] is None
    assert artifact["n_seeds"] == 0


def test_exp3680_build_artifact_uses_scoring_paths_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3680: build_artifact records scoring success and scoring failure."""

    rows = [_condition_rows(seed) for seed in (42, 137, 271, 314, 1729)]
    monkeypatch.setattr(
        exp3680,
        "probe_preconditions",
        lambda root, n_examples: [{"resource": "fixture", "available": True, "detail": "ok"}],
    )
    monkeypatch.setattr(
        exp3680,
        "load_exp2850_source_artifact",
        lambda root: {
            "condition_a_production_auroc_mean": 0.9131336,
            "condition_b_architecture_only_auroc_mean": 0.8946624,
            "n_seeds": 5,
            "n_examples": 1000,
            "reproducibility_checksum": "source",
        },
    )
    monkeypatch.setattr(exp3680, "discover_fr11_state_files", lambda root: [{"path": "state"}])

    def fake_score(
        root: Path,
        *,
        seed: int,
        n_examples: int,
        state_files: list[dict[str, str]],
    ) -> exp3680.ConditionScoreRows:
        return rows[(42, 137, 271, 314, 1729).index(seed)]

    monkeypatch.setattr(exp3680, "score_dual_condition_rows", fake_score)
    artifact = exp3680.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=4.0,
        n_bootstrap=4,
        adversarial_verify_clean=True,
    )
    exp3680.validate_artifact(artifact)
    assert artifact["n_seeds"] == 5
    assert artifact["preconditions_checked"] == [
        {"resource": "fixture", "available": True, "detail": "ok"}
    ]

    def broken_score(*args: object, **kwargs: object) -> exp3680.ConditionScoreRows:
        raise RuntimeError("scoring unavailable")

    monkeypatch.setattr(exp3680, "score_dual_condition_rows", broken_score)
    blocked = exp3680.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    exp3680.validate_artifact(blocked)
    assert blocked["honest_verdict"] == exp3680.BLOCKED_VERDICT
    assert blocked["preconditions_checked"][-1]["resource"] == "dual_condition_scoring"


def test_exp3680_preconditions_and_source_artifact_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3680: preconditions explain missing verifier, memory, and source inputs."""

    fover = tmp_path / "data" / "fover_corpus.jsonl"
    fover.parent.mkdir(parents=True)
    fover.write_text(
        "\n".join(
            json.dumps({"label": "correct" if idx % 2 == 0 else "incorrect", "step_text": str(idx)})
            for idx in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    source_path = tmp_path / exp3680.EXP2850_REL_PATH
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "condition_a_production_auroc_mean": 0.9131336,
                "condition_b_architecture_only_auroc_mean": 0.8946624,
                "n_seeds": 5,
                "n_examples": 1000,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp3680, "_score_text_verifiers", lambda texts: (_ for _ in ()).throw(RuntimeError("no verifier")))
    monkeypatch.setattr(exp3680, "discover_fr11_state_files", lambda root: (_ for _ in ()).throw(RuntimeError("no state")))
    checks = exp3680.probe_preconditions(tmp_path, n_examples=4)
    by_resource = {check["resource"]: check for check in checks}
    assert by_resource["fover_corpus"]["available"] is True
    assert by_resource["four_exp2837_scoring_verifiers"]["available"] is False
    assert by_resource["fr11_session_memory_state"]["available"] is False
    assert by_resource["exp2850_g1_source_artifact"]["available"] is True

    source_path.write_text(json.dumps({"n_seeds": 5}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing condition_a_production_auroc_mean"):
        exp3680.load_exp2850_source_artifact(tmp_path)


def test_exp3680_condition_scoring_rows_and_dual_condition_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3680: condition scoring preserves row parity and memory ablation."""

    fover = tmp_path / "data" / "fover_corpus.jsonl"
    fover.parent.mkdir(parents=True)
    rows = []
    for idx in range(3):
        rows.append({"question_id": f"ok_{idx}", "label": "correct", "step_text": f"{idx}+{idx}={2*idx}"})
        rows.append(
            {"question_id": f"bad_{idx}", "label": "incorrect", "step_text": f"{idx}+{idx}={2*idx+1}"}
        )
    fover.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        exp3680,
        "_score_text_verifiers",
        lambda texts: {
            "tier0r_curry_howard": [0.2 + 0.01 * idx for idx, _ in enumerate(texts)],
            "tier0s_arithmetic_gap": [0.3 + 0.01 * idx for idx, _ in enumerate(texts)],
            "tier0u_logical_consistency": [0.4 + 0.01 * idx for idx, _ in enumerate(texts)],
        },
    )
    monkeypatch.setattr(exp3680, "_load_fr11_memory_index", lambda root: {"question_ids": {"bad_0"}, "prompt_token_sets": []})
    monkeypatch.setattr(exp3680, "_fr11_memory_score", lambda row, memory: 1.0 if row["label"] == "incorrect" else 0.0)
    monkeypatch.setattr(exp3680, "discover_fr11_state_files", lambda root: [{"path": "state"}])
    production = exp3680.score_condition_verifier_rows(
        tmp_path,
        seed=42,
        n_examples=4,
        condition=CONDITION_PRODUCTION,
        require_no_state=False,
    )
    assert production.state_visible_count == 1
    assert set(production.scores_by_verifier) == set(exp3644.VERIFIER_NAMES)
    assert max(production.scores_by_verifier["fr11_session_memory"]) == 1.0

    monkeypatch.setattr(exp3680, "discover_fr11_state_files", lambda root: [])
    architecture = exp3680.score_condition_verifier_rows(
        tmp_path,
        seed=42,
        n_examples=4,
        condition=CONDITION_ARCHITECTURE_ONLY,
        require_no_state=True,
    )
    assert architecture.scores_by_verifier["fr11_session_memory"] == [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ConditionScoringError, match="unknown condition"):
        exp3680.score_condition_verifier_rows(
            tmp_path,
            seed=42,
            n_examples=4,
            condition="bad",
            require_no_state=False,
        )
    monkeypatch.setattr(exp3680, "discover_fr11_state_files", lambda root: [{"path": "state"}])
    with pytest.raises(ConditionScoringError, match="architecture-only condition saw"):
        exp3680.score_condition_verifier_rows(
            tmp_path,
            seed=42,
            n_examples=4,
            condition=CONDITION_ARCHITECTURE_ONLY,
            require_no_state=True,
        )

    prod = exp3680._ConditionVerifierScores([0, 1], {"x": [0.1, 0.2]}, 1, "same")
    arch = exp3680._ConditionVerifierScores([0, 1], {"x": [0.1, 0.2]}, 0, "same")
    calls: list[bool] = []

    def fake_condition(
        root: Path,
        *,
        seed: int,
        n_examples: int,
        condition: str,
        require_no_state: bool,
    ) -> exp3680._ConditionVerifierScores:
        calls.append(require_no_state)
        return arch if require_no_state else prod

    monkeypatch.setattr(exp3680, "score_condition_verifier_rows", fake_condition)
    monkeypatch.setattr(exp3680, "state_files_restored_sha_match", lambda root, state_files: True)
    panel = exp3680.score_dual_condition_rows(tmp_path, seed=42, n_examples=2, state_files=[])
    assert panel.labels == [0, 1]
    assert calls == [False, True]

    monkeypatch.setattr(
        exp3680,
        "score_condition_verifier_rows",
        lambda root, seed, n_examples, condition, require_no_state: exp3680._ConditionVerifierScores(
            [1, 0] if require_no_state else [0, 1],
            {"x": [0.1, 0.2]},
            0,
            "same",
        ),
    )
    with pytest.raises(ConditionScoringError, match="labels diverged"):
        exp3680.score_dual_condition_rows(tmp_path, seed=42, n_examples=2, state_files=[])

    monkeypatch.setattr(
        exp3680,
        "score_condition_verifier_rows",
        lambda root, seed, n_examples, condition, require_no_state: exp3680._ConditionVerifierScores(
            [0, 1],
            {"x": [0.1, 0.2]},
            0,
            "arch" if require_no_state else "prod",
        ),
    )
    with pytest.raises(ConditionScoringError, match="subsets diverged"):
        exp3680.score_dual_condition_rows(tmp_path, seed=42, n_examples=2, state_files=[])

    monkeypatch.setattr(
        exp3680,
        "score_condition_verifier_rows",
        lambda root, seed, n_examples, condition, require_no_state: exp3680._ConditionVerifierScores(
            [0, 1],
            {"x": [0.1, 0.2]},
            0,
            "same",
        ),
    )
    monkeypatch.setattr(exp3680, "state_files_restored_sha_match", lambda root, state_files: False)
    with pytest.raises(ConditionScoringError, match="restore SHA256 mismatch"):
        exp3680.score_dual_condition_rows(tmp_path, seed=42, n_examples=2, state_files=[])


def test_exp3680_write_artifact_stamps_adversarial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3680: writer reruns adversarial verification before final JSON."""

    measured = exp3680.build_artifact_from_condition_rows(
        [_condition_rows(seed) for seed in (42, 137, 271, 314, 1729)],
        started_s=0.0,
        now_s=4.0,
        n_bootstrap=4,
        adversarial_verify_clean=False,
    )
    monkeypatch.setattr(exp3680, "build_artifact", lambda root, started_s=None, now_s=None: dict(measured))
    monkeypatch.setattr(exp3680, "run_adversarial_verify_report", lambda path: {"flag_count": 0, "flags": []})
    output = exp3680.write_artifact(tmp_path, output_path="result.json", started_s=0.0, now_s=1.0)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["adversarial_verify_clean"] is True
    assert saved["acceptance_gate"]["passed"] is True
    assert saved["adversarial_verify_report"]["flag_count"] == 0

    blocked = exp3680._blocked_artifact(
        duration_s=0.1,
        random_seed=42,
        preconditions=[{"resource": "fixture", "available": False, "detail": "missing"}],
    )
    monkeypatch.setattr(exp3680, "build_artifact", lambda root, started_s=None, now_s=None: dict(blocked))
    blocked_output = exp3680.write_artifact(tmp_path, output_path="blocked.json")
    blocked_saved = json.loads(blocked_output.read_text(encoding="utf-8"))
    assert blocked_saved["honest_verdict"] == exp3680.BLOCKED_VERDICT


def test_exp3680_adversarial_report_cleanliness() -> None:
    """REQ-VERIFY-3680: adversarial clean ignores non-critical warns only."""

    assert exp3680.adversarial_report_is_clean({"flags": []}) is True
    assert exp3680.adversarial_report_is_clean(
        {"flags": [{"kind": "SAMPLE_SIZE", "severity": "warn"}]}
    ) is True
    assert exp3680.adversarial_report_is_clean(
        {"flags": [{"kind": "TAUTOLOGY", "severity": "warn"}]}
    ) is False
    assert exp3680.adversarial_report_is_clean(
        {"flags": [{"kind": "OTHER", "severity": "critical"}]}
    ) is False


def test_exp3680_validation_edges() -> None:
    """REQ-VERIFY-3680: schema guards enforce bare booleans and measured fields."""

    artifact = exp3680.build_artifact_from_condition_rows(
        [_condition_rows(seed) for seed in (42, 137, 271, 314, 1729)],
        started_s=0.0,
        now_s=4.0,
        n_bootstrap=4,
        adversarial_verify_clean=True,
    )
    with pytest.raises(ValueError, match="bare boolean"):
        exp3680.validate_artifact(dict(artifact, dependency_aware_g1_rigor_confirmed=1))
    with pytest.raises(ValueError, match="bare boolean"):
        exp3680.validate_artifact(dict(artifact, adversarial_verify_clean=1))
    with pytest.raises(ValueError, match="missing required artifact fields"):
        missing = dict(artifact)
        missing.pop("honest_verdict")
        exp3680.validate_artifact(missing)
    with pytest.raises(ValueError, match="field_principles must be present"):
        exp3680.validate_artifact(dict(artifact, field_principles=[]))
    with pytest.raises(ValueError, match="missing field principles"):
        exp3680.validate_artifact(dict(artifact, field_principles={}))
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3680.validate_artifact(dict(artifact, honest_verdict="complete: invented"))
    with pytest.raises(ValueError, match="duration_s"):
        exp3680.validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="n_seeds"):
        exp3680.validate_artifact(dict(artifact, n_seeds=4))
    with pytest.raises(ValueError, match="n_examples"):
        exp3680.validate_artifact(dict(artifact, n_examples=999))
    with pytest.raises(ValueError, match="production_auroc_dependency_aware"):
        exp3680.validate_artifact(dict(artifact, production_auroc_dependency_aware=1.2))
    with pytest.raises(ValueError, match="dependency_vs_carnot_delta_ci"):
        exp3680.validate_artifact(dict(artifact, dependency_vs_carnot_delta_ci=[]))
    with pytest.raises(ValueError, match="production_auroc_ci"):
        exp3680.validate_artifact(dict(artifact, production_auroc_ci={"point": 0.5}))
    with pytest.raises(ValueError, match="bounds"):
        exp3680.validate_artifact(
            dict(artifact, production_auroc_ci={"point": 0.5, "ci95": [float("nan"), 0.6]})
        )
    with pytest.raises(ValueError, match="contain"):
        exp3680.validate_artifact(
            dict(artifact, production_auroc_ci={"point": 0.7, "ci95": [0.5, 0.6]})
        )
    with pytest.raises(ValueError, match="delong_p_dependency_vs_carnot"):
        exp3680.validate_artifact(dict(artifact, delong_p_dependency_vs_carnot=-0.1))
    with pytest.raises(ValueError, match="at least one condition"):
        exp3680.build_artifact_from_condition_rows([], started_s=0.0, now_s=1.0)
    with pytest.raises(ValueError, match="binary classes"):
        exp3680.build_artifact_from_condition_rows(
            [
                exp3680.ConditionScoreRows(
                    seed=42,
                    labels=[0] * 1000,
                    production_scores_by_verifier={
                        name: [0.1] * 1000 for name in exp3644.VERIFIER_NAMES
                    },
                    architecture_scores_by_verifier={
                        name: [0.1] * 1000 for name in exp3644.VERIFIER_NAMES
                    },
                )
            ],
            started_s=0.0,
            now_s=1.0,
        )
    with pytest.raises(ValueError, match="same length"):
        bad = _condition_rows(42)
        exp3680.build_artifact_from_condition_rows(
            [
                exp3680.ConditionScoreRows(
                    seed=bad.seed,
                    labels=bad.labels[:-1],
                    production_scores_by_verifier=bad.production_scores_by_verifier,
                    architecture_scores_by_verifier=bad.architecture_scores_by_verifier,
                )
            ],
            started_s=0.0,
            now_s=1.0,
        )
    assert exp3680._round_metric(None) is None
    assert exp3680._round_p(None) is None
    assert exp3680._round_p(1e-8) == 1e-8
    assert exp3680._round_p(0.1234567) == 0.123457
    assert exp3680._duration(0.0, 1.5) == 1.5
    assert exp3680._is_finite_number(True) is False
    assert exp3680._is_finite_number(0.5) is True
