"""Test the Exp 2842 milestone .269 multi-corpus capstone synthesis.

References:
- REQ-BENCH-001
- REQ-BENCH-010
- REQ-PUBLISH-032
- SCENARIO-PUBLISH-032
- SCENARIO-PUBLISH-032C

These tests verify that the synthesis module:
  1. Honours the MANDATORY thesis operationalisations from the task spec.
  2. Produces a terminal-prefix honest_verdict.
  3. Correctly classifies blocked vs measured upstream artifacts.
  4. Never promotes blocked/missing data into the headline table.
  5. Recommends headline repin when exp2837 real data is present.
  6. Writes a valid JSON artifact to disk that satisfies all required schema fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v269_2842 as cap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_json(directory: Path, filename: str, data: Any) -> Path:
    """Helper: write a JSON file and return its path."""
    p = directory / filename
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def fover_real_artifact() -> dict[str, Any]:
    """Minimal well-formed exp2837 (FoVer dual-condition, REAL DATA) artifact."""
    return {
        "artifact": "experiment_2837_fover_memory_leakage_v3",
        "honest_verdict": (
            "complete: FoVer memory-leakage v3 measured with production state "
            "and architecture-only reset conditions"
        ),
        "condition_a_production_auroc_mean": cap.FOVER_PRODUCTION_AUROC,
        "condition_a_production_auroc_std": cap.FOVER_PRODUCTION_STD,
        "condition_b_architecture_only_auroc_mean": cap.FOVER_ARCHITECTURE_ONLY_AUROC,
        "condition_b_architecture_only_auroc_std": cap.FOVER_ARCHITECTURE_ONLY_STD,
        "learning_contribution": cap.FOVER_LEARNING_DELTA,
        "duration_s": 16.27,
        "n_examples": 1000,
        "n_seeds": 5,
    }


@pytest.fixture
def mbpp_blocked_artifact() -> dict[str, Any]:
    """Minimal blocked exp2838 artifact (MBPP dataset not accessible)."""
    return {
        "artifact": "experiment_2838_mbpp_dual_condition_v3",
        "honest_verdict": "blocked_mbpp_dataset",
        "condition_a_production_auroc_mean": None,
        "condition_b_architecture_only_auroc_mean": None,
        "learning_contribution": None,
        "duration_s": 0.3,
    }


@pytest.fixture
def humaneval_blocked_artifact() -> dict[str, Any]:
    """Minimal blocked exp2839 artifact (HumanEval dataset not accessible)."""
    return {
        "artifact": "experiment_2839_humaneval_dual_condition_v3",
        "honest_verdict": "blocked_humaneval_dataset",
        "condition_a_production_auroc_mean": None,
        "condition_b_architecture_only_auroc_mean": None,
        "learning_contribution": None,
        "duration_s": 0.2,
    }


@pytest.fixture
def truthfulqa_blocked_artifact() -> dict[str, Any]:
    """Minimal blocked exp2840 artifact (TruthfulQA generation split not accessible)."""
    return {
        "artifact": "experiment_2840_truthfulqa_dual_condition_v4",
        "honest_verdict": "blocked_truthfulqa_generation_split",
        "condition_a_production_auroc_mean": None,
        "condition_b_architecture_only_auroc_mean": None,
        "learning_contribution": None,
        "duration_s": 0.2,
    }


@pytest.fixture
def matrix_artifact() -> dict[str, Any]:
    """Minimal exp2840_cross_corpus_verifier_matrix_v3 artifact."""
    return {
        "honest_verdict": "complete: real upstream per-verifier AUROC matrix v3 built",
        "architecture_transfer_verifiers": [],
        "memory_augmented_verifiers": ["fr11_session_memory"],
        "corpus_specific_verifiers": ["tier0r_curry_howard"],
        "low_signal_verifiers": ["tier0s_arithmetic_gap", "tier0u_logical_consistency"],
        "duration_s": 0.001,
    }


@pytest.fixture
def preflight_artifact() -> dict[str, Any]:
    """Minimal exp2836 preflight artifact."""
    return {
        "artifact": "experiment_2836_sota_runtime_preflight",
        "honest_verdict": "success: .venv CUDA torch available and at least one mandated SOTA GGUF load-smoked",
        "duration_s": 8.2,
    }


@pytest.fixture
def pilot_artifact() -> dict[str, Any]:
    """Minimal exp2841 HaluEval/FEVER pilot artifact."""
    return {
        "artifact": "experiment_2841_halueval_fever_pilot",
        "honest_verdict": "complete: HaluEval/FEVER readiness pilot measured",
        "pilot_auroc_by_dataset": {
            "FEVER": {"auroc": 0.4327, "auroc_ci95": [0.22, 0.65]},
            "HaluEval": {"auroc": 0.61, "auroc_ci95": [0.49, 0.73]},
        },
        "n_examples": 50,
        "pilot_only": True,
        "duration_s": 12.0,
    }


def _make_all_artifacts(tmp_path: Path, overrides: dict[str, Any] | None = None) -> None:
    """Write all required .269 artifact files into *tmp_path*/results/."""
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)

    defaults: dict[str, tuple[str, Any]] = {
        "experiment_2835_archive_v268.json": (
            "archive",
            {"honest_verdict": "complete: archive_ready=true", "duration_s": 1.0},
        ),
        "experiment_2836_sota_runtime_preflight.json": (
            "exp2836_preflight",
            {
                "honest_verdict": "success: .venv CUDA torch available and at least one mandated SOTA GGUF load-smoked",
                "duration_s": 8.2,
            },
        ),
        "experiment_2836_fover_memory_leakage_isolation.json": (
            "exp2836_fover",
            {"honest_verdict": "blocked_model_cache: missing", "duration_s": 1.7},
        ),
        "experiment_2837_fover_memory_leakage_v3.json": (
            "exp2837_fover",
            {
                "honest_verdict": "complete: FoVer memory-leakage v3 measured",
                "condition_a_production_auroc_mean": cap.FOVER_PRODUCTION_AUROC,
                "condition_a_production_auroc_std": cap.FOVER_PRODUCTION_STD,
                "condition_b_architecture_only_auroc_mean": cap.FOVER_ARCHITECTURE_ONLY_AUROC,
                "condition_b_architecture_only_auroc_std": cap.FOVER_ARCHITECTURE_ONLY_STD,
                "learning_contribution": cap.FOVER_LEARNING_DELTA,
                "duration_s": 16.27,
                "n_examples": 1000,
                "n_seeds": 5,
            },
        ),
        "experiment_2837_mbpp_ensemble_eval.json": (
            "exp2837_mbpp",
            {"honest_verdict": "blocked_model_not_cached_qwen36_35b_a3b_gguf", "duration_s": 0.3},
        ),
        "experiment_2838_mbpp_dual_condition_v3.json": (
            "exp2838",
            {"honest_verdict": "blocked_mbpp_dataset", "duration_s": 0.3},
        ),
        "experiment_2839_humaneval_dual_condition_v3.json": (
            "exp2839_humaneval",
            {"honest_verdict": "blocked_humaneval_dataset", "duration_s": 0.2},
        ),
        "experiment_2839_truthfulqa_ensemble_eval.json": (
            "exp2839_truthfulqa",
            {"honest_verdict": "blocked_model_not_cached_qwen36_35b_a3b_gguf", "duration_s": 0.2},
        ),
        "experiment_2840_cross_corpus_verifier_matrix_v3.json": (
            "exp2840_matrix",
            {
                "honest_verdict": "complete: real upstream per-verifier AUROC matrix v3 built",
                "architecture_transfer_verifiers": [],
                "memory_augmented_verifiers": ["fr11_session_memory"],
                "corpus_specific_verifiers": ["tier0r_curry_howard"],
                "low_signal_verifiers": ["tier0s_arithmetic_gap", "tier0u_logical_consistency"],
                "duration_s": 0.001,
            },
        ),
        "experiment_2840_truthfulqa_dual_condition_v4.json": (
            "exp2840_truthfulqa",
            {"honest_verdict": "blocked_truthfulqa_generation_split", "duration_s": 0.2},
        ),
        "experiment_2841_halueval_fever_pilot.json": (
            "exp2841_pilot",
            {
                "honest_verdict": "complete: HaluEval/FEVER readiness pilot measured",
                "pilot_auroc_by_dataset": {
                    "FEVER": {"auroc": 0.4327, "auroc_ci95": [0.22, 0.65]},
                },
                "n_examples": 50,
                "pilot_only": True,
                "duration_s": 12.0,
            },
        ),
        "experiment_2841_paper_v6_multicorpus_table_v3.json": (
            "exp2841_table",
            {"honest_verdict": "complete: exp2836-2840 artifacts integrated honestly", "duration_s": 0.4},
        ),
        "experiment_2843_beaver_epr_bounded_probe.json": (
            "exp2843",
            {
                "honest_verdict": "complete: bounded-prefix/EPR proxy evaluated on local FoVer-style labels",
                "bounded_prefix_probe_auc": 0.7756,
                "entropy_production_summary": {"entropy_production_auc": 0.5812},
                "failure_modes": {"proxy_not_exact_beaver": True},
                "duration_s": 0.23,
            },
        ),
        "experiment_2844_loopus_fr11_self_learning_pilot.json": (
            "exp2844",
            {
                "honest_verdict": "blocked_live_recurrence_backend",
                "mean_energy_delta_loop0_to_final": None,
                "duration_s": 0.4,
            },
        ),
    }
    overrides = overrides or {}
    for filename, (key, data) in defaults.items():
        payload = overrides.get(key, data)
        _write_json(results, filename, payload)


# ---------------------------------------------------------------------------
# Unit tests for utility functions
# ---------------------------------------------------------------------------


def test_is_terminal_verdict_complete() -> None:
    """REQ-BENCH-001: complete: prefix is terminal."""
    assert cap.is_terminal_verdict("complete: some description")


def test_is_terminal_verdict_success() -> None:
    """REQ-BENCH-001: success: prefix is terminal."""
    assert cap.is_terminal_verdict("success: all good")


def test_is_terminal_verdict_underscore_form() -> None:
    """REQ-BENCH-001: complete_ underscore form is terminal."""
    assert cap.is_terminal_verdict("complete_fover_measured")


def test_is_terminal_verdict_blocked_is_not_terminal() -> None:
    """REQ-BENCH-001: blocked_ is NOT a terminal success verdict."""
    assert not cap.is_terminal_verdict("blocked_mbpp_dataset")


def test_is_blocked_verdict_identifies_blocked() -> None:
    """REQ-BENCH-001: blocked_ verdicts are correctly classified."""
    assert cap.is_blocked_verdict("blocked_mbpp_dataset")
    assert cap.is_blocked_verdict("blocked_humaneval_dataset")
    assert cap.is_blocked_verdict("blocked_truthfulqa_generation_split")


def test_is_blocked_verdict_rejects_complete() -> None:
    """REQ-BENCH-001: complete: is not a blocked verdict."""
    assert not cap.is_blocked_verdict("complete: measured")


def test_read_json_missing_file(tmp_path: Path) -> None:
    """REQ-BENCH-001: missing artifact returns empty dict, does not crash."""
    result = cap.read_json(tmp_path / "nonexistent.json")
    assert result == {}


def test_read_json_malformed_file(tmp_path: Path) -> None:
    """REQ-BENCH-001: malformed JSON returns empty dict, does not crash."""
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    result = cap.read_json(p)
    assert result == {}


# ---------------------------------------------------------------------------
# Thesis determination tests
# ---------------------------------------------------------------------------


def test_fover_overfit_false_when_no_non_fover_data() -> None:
    """SCENARIO-PUBLISH-032: overfit thesis cannot be confirmed without non-FoVer data.

    WHY: the thesis operationalisation requires at least one non-FoVer
    architecture-only AUROC to compare against FoVer's.  Blocked non-FoVer
    artifacts cannot contribute data.
    """
    fover_art = {
        "condition_b_architecture_only_auroc_mean": cap.FOVER_ARCHITECTURE_ONLY_AUROC,
    }
    non_fover = [
        {"honest_verdict": "blocked_mbpp_dataset"},
        {"honest_verdict": "blocked_humaneval_dataset"},
        {"honest_verdict": "blocked_truthfulqa_generation_split"},
    ]
    confirmed, rationale = cap._determine_fover_overfit(fover_art, non_fover)
    assert confirmed is False
    assert "Cannot evaluate" in rationale or "blocked" in rationale.lower()


def test_fover_overfit_false_when_fover_art_empty() -> None:
    """SCENARIO-PUBLISH-032: overfit thesis False when FoVer artifact is missing."""
    confirmed, _ = cap._determine_fover_overfit({}, [])
    assert confirmed is False


def test_fover_overfit_confirmed_when_gap_exceeds_threshold() -> None:
    """SCENARIO-PUBLISH-032: overfit thesis True when gap > 0.10.

    WHY: if FoVer arch-only = 0.90 and non-FoVer arch-only = 0.75, the gap
    is 0.15 > 0.10 — the thesis is confirmed.
    """
    fover_art = {"condition_b_architecture_only_auroc_mean": 0.90}
    non_fover = [{"condition_b_architecture_only_auroc_mean": 0.75, "honest_verdict": "complete: ok"}]
    confirmed, rationale = cap._determine_fover_overfit(fover_art, non_fover)
    assert confirmed is True
    assert "0.90" in rationale or "0.9000" in rationale


def test_self_learning_false_when_below_threshold() -> None:
    """SCENARIO-PUBLISH-032: self-learning not confirmed when delta < 0.05.

    WHY: the real exp2837 delta is 0.01847 — positive but below threshold.
    """
    art = {
        "honest_verdict": "complete: measured",
        "learning_contribution": cap.FOVER_LEARNING_DELTA,  # 0.01847, < 0.05
    }
    confirmed, rationale = cap._determine_self_learning_contribution(art)
    assert confirmed is False
    assert "0.05" in rationale or "threshold" in rationale


def test_self_learning_true_when_above_threshold() -> None:
    """SCENARIO-PUBLISH-032: self-learning confirmed when delta > 0.05."""
    art = {
        "honest_verdict": "complete: measured",
        "learning_contribution": 0.07,  # > 0.05
    }
    confirmed, _ = cap._determine_self_learning_contribution(art)
    assert confirmed is True


def test_self_learning_false_when_artifact_flagged() -> None:
    """SCENARIO-PUBLISH-032: adversarially flagged artifact cannot confirm self-learning."""
    art = {
        "honest_verdict": "complete: measured",
        "learning_contribution": 0.07,
        "flagged_adversarial": True,  # Adversarial flag present
    }
    confirmed, _ = cap._determine_self_learning_contribution(art)
    assert confirmed is False


def test_headline_repin_true_when_fover_real_data(fover_real_artifact: dict[str, Any]) -> None:
    """SCENARIO-PUBLISH-032C: headline repin recommended when exp2837 has real data.

    WHY: per CLAUDE.md, all headline results must have live GPU provenance.
    exp2837's 5-seed replicated measurement supersedes the carry-forward 0.9857.
    """
    recommended, rationale = cap._determine_headline_repin(fover_real_artifact)
    assert recommended is True
    assert str(cap.FOVER_PRODUCTION_AUROC)[:6] in rationale


def test_headline_repin_false_when_no_fover_data() -> None:
    """SCENARIO-PUBLISH-032C: no repin when fover artifact is missing."""
    recommended, _ = cap._determine_headline_repin({})
    assert recommended is False


# ---------------------------------------------------------------------------
# Integration tests using tmp_path artifacts
# ---------------------------------------------------------------------------


def test_build_artifact_required_fields_present(tmp_path: Path) -> None:
    """REQ-BENCH-010: all task-spec required fields are present in the artifact.

    WHY: the task spec explicitly lists 'REQUIRED ARTIFACT FIELDS' with
    principle annotations.  A schema-compliant artifact must contain all of them.
    """
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)

    required_fields = [
        "honest_verdict",
        "corpora_headline_table",
        "fover_shape_overfit_confirmed",
        "self_learning_contribution_confirmed",
        "supersedes_capstones",
        "architecture_transfer_verifiers",
        "memory_augmented_verifiers",
        "corpus_specific_verifiers",
        "low_signal_verifiers",
        "recommended_headline_repin",
        "gaps_for_270",
        "acceptance_criteria_met",
        "duration_s",
    ]
    missing = [f for f in required_fields if f not in artifact]
    assert not missing, f"Missing required fields: {missing}"


def test_honest_verdict_starts_with_terminal_prefix(tmp_path: Path) -> None:
    """REQ-BENCH-001: verdict must start with a terminal prefix.

    WHY: CLAUDE.md Verdict Terminal-Prefix Discipline.  Non-terminal verdicts
    are mis-classified as 'partial' by the conductor's reconciler.
    """
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    assert cap.is_terminal_verdict(artifact["honest_verdict"]), (
        f"honest_verdict does not start with terminal prefix: {artifact['honest_verdict']!r}"
    )


def test_supersedes_capstones_has_both_prior_capstones(tmp_path: Path) -> None:
    """REQ-PUBLISH-032: both prior capstones are listed as superseded."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    assert "exp2826" in artifact["supersedes_capstones"]
    assert "exp2834" in artifact["supersedes_capstones"]


def test_fover_table_row_has_real_data(tmp_path: Path) -> None:
    """REQ-BENCH-010: FoVer table row carries real AUROC values from exp2837."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    fover = artifact["corpora_headline_table"]["FoVer"]
    assert fover["production_mean"] == pytest.approx(cap.FOVER_PRODUCTION_AUROC, abs=1e-9)
    assert fover["architecture_only_mean"] == pytest.approx(cap.FOVER_ARCHITECTURE_ONLY_AUROC, abs=1e-9)
    assert fover["learning_delta"] == pytest.approx(cap.FOVER_LEARNING_DELTA, abs=1e-9)


def test_non_fover_table_rows_have_null_values(tmp_path: Path) -> None:
    """REQ-BENCH-010: blocked corpora must not carry fabricated AUROC values."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    for corpus in ["MBPP", "HumanEval", "TruthfulQA"]:
        row = artifact["corpora_headline_table"][corpus]
        assert row["production_mean"] is None, (
            f"{corpus} production_mean should be null (blocked corpus)"
        )
        assert row["architecture_only_mean"] is None, (
            f"{corpus} architecture_only_mean should be null (blocked corpus)"
        )


def test_fover_overfit_false_on_real_data(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-032: overfit thesis is False when non-FoVer is blocked."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    assert artifact["fover_shape_overfit_confirmed"] is False


def test_self_learning_false_on_real_delta(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-032: self-learning not confirmed (delta 0.0185 < 0.05)."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    assert artifact["self_learning_contribution_confirmed"] is False


def test_recommended_headline_repin_true_on_real_data(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-032C: headline repin recommended when exp2837 has real data."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    assert artifact["recommended_headline_repin"] is True


def test_acceptance_criteria_met_count(tmp_path: Path) -> None:
    """REQ-BENCH-010: exactly 6 acceptance criteria are met (3 corpus evals blocked)."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    # 6 of 10: FoVer production, arch-only, delta, matrix, pilot, preflight
    assert artifact["acceptance_criteria_met"] == 6


def test_gaps_for_270_covers_blocked_corpora(tmp_path: Path) -> None:
    """REQ-PUBLISH-032: gaps for .270 must identify all three blocked corpus evaluations."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    gap_titles = [g["title"] for g in artifact["gaps_for_270"]]
    assert any("MBPP" in t for t in gap_titles), "MBPP gap missing from gaps_for_270"
    assert any("HumanEval" in t for t in gap_titles), "HumanEval gap missing from gaps_for_270"
    assert any("TruthfulQA" in t for t in gap_titles), "TruthfulQA gap missing from gaps_for_270"


def test_memory_augmented_verifiers_from_matrix(tmp_path: Path) -> None:
    """REQ-BENCH-001: verifier classification is taken from the matrix artifact."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    assert "fr11_session_memory" in artifact["memory_augmented_verifiers"]


def test_duration_s_is_non_negative(tmp_path: Path) -> None:
    """REQ-BENCH-001: duration_s must be a non-negative float (no sleep padding)."""
    _make_all_artifacts(tmp_path)
    artifact = cap.build_artifact(repo_root=tmp_path)
    assert isinstance(artifact["duration_s"], float)
    assert artifact["duration_s"] >= 0.0


def test_write_artifact_creates_file(tmp_path: Path) -> None:
    """REQ-BENCH-010: write_artifact() produces a readable, schema-valid JSON file."""
    _make_all_artifacts(tmp_path)
    out = cap.write_artifact(repo_root=tmp_path)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["honest_verdict"].startswith("complete:")
    assert loaded["fover_shape_overfit_confirmed"] is False
    assert loaded["self_learning_contribution_confirmed"] is False
    assert loaded["recommended_headline_repin"] is True
