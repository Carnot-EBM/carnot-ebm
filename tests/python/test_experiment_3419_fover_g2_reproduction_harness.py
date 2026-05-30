"""Tests for Exp 3419 FoVer G2 reproduction harness.

Spec: REQ-VERIFY-2837 (the headline this harness reproduces),
      SCENARIO-VERIFY-2837.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers from the reproduce script under test
# ---------------------------------------------------------------------------
from scripts.reproduce_fover_headline import (
    CONDITION_A_CI_HIGH,
    CONDITION_A_CI_LOW,
    LEARNING_CONTRIB_CI_HIGH,
    LEARNING_CONTRIB_CI_LOW,
    RANDOM_SEEDS,
    N_EXAMPLES,
    build_reproducer_config,
    check_acceptance_ci,
    passthrough_precondition_probe,
    in_process_condition_runner,
)
from scripts.experiment_3419_fover_g2_reproduction_harness_v1 import (
    build_artifact,
    check_preconditions,
    collect_platform_info,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal FoVer corpus and fake experiment state
# ---------------------------------------------------------------------------

def _write_fover_corpus(path: Path, n_per_class: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx in range(n_per_class):
        rows.append({
            "question_id": f"ok_{idx}",
            "step_text": f"Step {idx}: compute {idx} + {idx} = {2 * idx}.",
            "label": "correct",
        })
        rows.append({
            "question_id": f"bad_{idx}",
            "step_text": f"Contradiction: state {idx} conflicts with rule {idx + 3}.",
            "label": "incorrect",
        })
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def _make_fake_result(
    *,
    cond_a: float = 0.9131,
    cond_b: float = 0.8947,
    lc: float = 0.0184,
    checksum: str = "abc123",
    blocked: bool = False,
) -> dict[str, Any]:
    if blocked:
        return {
            "honest_verdict": "blocked_fr11_state_files",
            "blocked_resources": ["fr11_state_files"],
            "condition_a_production_auroc_mean": None,
            "condition_b_architecture_only_auroc_mean": None,
            "learning_contribution": None,
            "learning_contribution_ci95": None,
        }
    return {
        "honest_verdict": "complete: dual-condition measured",
        "condition_a_production_auroc_mean": cond_a,
        "condition_b_architecture_only_auroc_mean": cond_b,
        "learning_contribution": lc,
        "learning_contribution_ci95": {"mean": lc, "low": lc - 0.006, "high": lc + 0.006},
        "condition_a_production_auroc_ci95": {
            "mean": cond_a, "low": cond_a - 0.010, "high": cond_a + 0.010
        },
        "per_seed_results": [],
        "per_verifier_condition_a_auroc": {},
        "per_verifier_condition_b_auroc": {},
        "reproducibility_checksum": checksum,
    }


# ---------------------------------------------------------------------------
# tests for check_acceptance_ci
# ---------------------------------------------------------------------------

class TestCheckAcceptanceCi:
    """SCENARIO-VERIFY-2837: CI bounds accepted/rejected correctly."""

    def test_both_in_ci(self) -> None:
        # REQ-VERIFY-2837: CI acceptance check passes when both in range.
        result = _make_fake_result(cond_a=0.9131, lc=0.0185)
        a_in, lc_in = check_acceptance_ci(result)
        assert a_in is True
        assert lc_in is True

    def test_cond_a_below_ci(self) -> None:
        result = _make_fake_result(cond_a=0.9026, lc=0.0185)
        a_in, lc_in = check_acceptance_ci(result)
        assert a_in is False
        assert lc_in is True

    def test_cond_a_above_ci(self) -> None:
        result = _make_fake_result(cond_a=0.9236, lc=0.0185)
        a_in, _ = check_acceptance_ci(result)
        assert a_in is False

    def test_lc_below_ci(self) -> None:
        result = _make_fake_result(cond_a=0.9131, lc=0.0124)
        _, lc_in = check_acceptance_ci(result)
        assert lc_in is False

    def test_lc_above_ci(self) -> None:
        result = _make_fake_result(cond_a=0.9131, lc=0.0246)
        _, lc_in = check_acceptance_ci(result)
        assert lc_in is False

    def test_none_values_not_in_ci(self) -> None:
        result = _make_fake_result(blocked=True)
        a_in, lc_in = check_acceptance_ci(result)
        assert a_in is False
        assert lc_in is False

    def test_ci_boundaries_inclusive(self) -> None:
        # REQ-VERIFY-2837: boundaries [low, high] are inclusive.
        result = _make_fake_result(cond_a=CONDITION_A_CI_LOW, lc=LEARNING_CONTRIB_CI_LOW)
        a_in, lc_in = check_acceptance_ci(result)
        assert a_in is True
        assert lc_in is True

        result2 = _make_fake_result(cond_a=CONDITION_A_CI_HIGH, lc=LEARNING_CONTRIB_CI_HIGH)
        a_in2, lc_in2 = check_acceptance_ci(result2)
        assert a_in2 is True
        assert lc_in2 is True

    def test_lc_from_dict_key(self) -> None:
        # learning_contribution_ci95 dict path
        result = _make_fake_result(cond_a=0.9131, lc=0.0185)
        result.pop("learning_contribution", None)
        result["learning_contribution_ci95"] = {"mean": 0.0185, "low": 0.013, "high": 0.024}
        _, lc_in = check_acceptance_ci(result)
        assert lc_in is True


# ---------------------------------------------------------------------------
# tests for passthrough_precondition_probe
# ---------------------------------------------------------------------------

class TestPassthroughPreconditionProbe:
    """REQ-VERIFY-2837: custom probe skips SOTA model check."""

    def test_corpus_missing_gives_blocked_check(self, tmp_path: Path) -> None:
        # REQ-VERIFY-2837: corpus missing → fover_corpus check fails.
        from carnot.eval.fover_memory_leakage_v3 import ExperimentConfig

        cfg = ExperimentConfig(repo_root=tmp_path, n_examples=10)
        checks = passthrough_precondition_probe(cfg, [], {})
        corpus_check = next(c for c in checks if c.resource == "fover_corpus")
        assert corpus_check.available is False

    def test_corpus_present_but_too_small(self, tmp_path: Path) -> None:
        # REQ-VERIFY-2837: corpus line count below n_examples → check fails.
        from carnot.eval.fover_memory_leakage_v3 import ExperimentConfig

        corpus = tmp_path / "data" / "fover_corpus.jsonl"
        _write_fover_corpus(corpus, n_per_class=2)  # 4 rows total
        cfg = ExperimentConfig(repo_root=tmp_path, n_examples=100)
        checks = passthrough_precondition_probe(cfg, [], {})
        corpus_check = next(c for c in checks if c.resource == "fover_corpus")
        assert corpus_check.available is False

    def test_corpus_sufficient(self, tmp_path: Path) -> None:
        from carnot.eval.fover_memory_leakage_v3 import ExperimentConfig

        corpus = tmp_path / "data" / "fover_corpus.jsonl"
        _write_fover_corpus(corpus, n_per_class=10)  # 20 rows
        cfg = ExperimentConfig(repo_root=tmp_path, n_examples=20)
        checks = passthrough_precondition_probe(cfg, [], {})
        corpus_check = next(c for c in checks if c.resource == "fover_corpus")
        assert corpus_check.available is True

    def test_fr11_state_files_empty_gives_blocked(self, tmp_path: Path) -> None:
        # REQ-VERIFY-2837: no FR-11 state files → fr11_state_files check fails.
        from carnot.eval.fover_memory_leakage_v3 import ExperimentConfig

        corpus = tmp_path / "data" / "fover_corpus.jsonl"
        _write_fover_corpus(corpus, n_per_class=10)
        cfg = ExperimentConfig(repo_root=tmp_path, n_examples=20)
        checks = passthrough_precondition_probe(cfg, [], {})
        fr11_check = next(c for c in checks if c.resource == "fr11_state_files")
        assert fr11_check.available is False

    def test_fr11_state_files_present_passes(self, tmp_path: Path) -> None:
        from carnot.eval.fover_memory_leakage_v3 import ExperimentConfig

        corpus = tmp_path / "data" / "fover_corpus.jsonl"
        _write_fover_corpus(corpus, n_per_class=10)
        cfg = ExperimentConfig(repo_root=tmp_path, n_examples=20)
        fake_state = [{"path": "data/fr11_fake.jsonl", "sha256": "abc", "n_bytes": 1}]
        checks = passthrough_precondition_probe(cfg, fake_state, {})
        fr11_check = next(c for c in checks if c.resource == "fr11_state_files")
        assert fr11_check.available is True

    def test_probe_does_not_check_sota_model(self, tmp_path: Path) -> None:
        # REQ-VERIFY-2837: probe must NOT add a mandated_sota_model_path check.
        from carnot.eval.fover_memory_leakage_v3 import ExperimentConfig

        corpus = tmp_path / "data" / "fover_corpus.jsonl"
        _write_fover_corpus(corpus, n_per_class=10)
        cfg = ExperimentConfig(repo_root=tmp_path, n_examples=20)
        fake_state = [{"path": "data/fr11_fake.jsonl", "sha256": "abc", "n_bytes": 1}]
        checks = passthrough_precondition_probe(cfg, fake_state, {})
        resources = {c.resource for c in checks}
        assert "mandated_sota_model_path" not in resources
        assert "exp2836_artifact" not in resources


# ---------------------------------------------------------------------------
# tests for build_reproducer_config
# ---------------------------------------------------------------------------

class TestBuildReproducerConfig:
    """REQ-VERIFY-2837: config does not point at exp2837 artifact."""

    def test_returns_config_with_correct_seeds(self, tmp_path: Path) -> None:
        cfg = build_reproducer_config(tmp_path, seeds=[42, 137], n_examples=200)
        assert cfg.random_seeds == (42, 137)
        assert cfg.n_examples == 200

    def test_exp2836_path_not_in_results(self, tmp_path: Path) -> None:
        # The stub path must NOT coincide with any real exp2837 artifact path,
        # ensuring the recompute is clean.
        cfg = build_reproducer_config(tmp_path)
        assert "2837" not in str(cfg.exp2836_path)

    def test_default_seeds_match_published(self, tmp_path: Path) -> None:
        cfg = build_reproducer_config(tmp_path)
        assert cfg.random_seeds == RANDOM_SEEDS


# ---------------------------------------------------------------------------
# tests for build_artifact
# ---------------------------------------------------------------------------

class TestBuildArtifact:
    """SCENARIO-VERIFY-2837: artifact schema and g2_status field."""

    def _fake_checks(self) -> dict[str, Any]:
        return {
            "fover_corpus_present": True,
            "eval_module_importable": True,
            "exp2836_present": True,
        }

    def _fake_platform(self) -> dict[str, Any]:
        return {"python_version": "3.11.0", "platform": "Linux"}

    def test_reproduced_in_ci_true_when_both_in_range(self) -> None:
        result = _make_fake_result(cond_a=0.9131, lc=0.0185)
        artifact = build_artifact(
            recompute_result=result,
            checks=self._fake_checks(),
            platform_info=self._fake_platform(),
            start_time=0.0,
            end_time=10.0,
        )
        assert artifact["reproduced_in_ci"] is True
        assert artifact["g2_status"] == "advanced_turnkey_harness_internal_confirmation"
        assert artifact["g2_independent_reproducer"] is False

    def test_reproduced_in_ci_false_when_outside_range(self) -> None:
        result = _make_fake_result(cond_a=0.80, lc=0.0050)
        artifact = build_artifact(
            recompute_result=result,
            checks=self._fake_checks(),
            platform_info=self._fake_platform(),
            start_time=0.0,
            end_time=10.0,
        )
        assert artifact["reproduced_in_ci"] is False

    def test_artifact_has_required_schema_fields(self) -> None:
        # REQ-VERIFY-2837: all required artifact fields must be present.
        result = _make_fake_result()
        artifact = build_artifact(
            recompute_result=result,
            checks=self._fake_checks(),
            platform_info=self._fake_platform(),
            start_time=0.0,
            end_time=5.0,
        )
        required = [
            "honest_verdict",
            "inference_substrate",
            "condition_a_auroc_reproduced",
            "learning_contribution_reproduced",
            "reproduced_in_ci",
            "harness_path",
            "g2_status",
            "reproducibility_checksum",
            "random_seed",
            "duration_s",
        ]
        for field in required:
            assert field in artifact, f"missing required field: {field}"

    def test_honest_verdict_starts_with_complete(self) -> None:
        # REQ-VERIFY-2837: verdict must use terminal prefix.
        result = _make_fake_result()
        artifact = build_artifact(
            recompute_result=result,
            checks=self._fake_checks(),
            platform_info=self._fake_platform(),
            start_time=0.0,
            end_time=5.0,
        )
        assert str(artifact["honest_verdict"]).startswith("complete:")

    def test_inference_substrate_value(self) -> None:
        result = _make_fake_result()
        artifact = build_artifact(
            recompute_result=result,
            checks=self._fake_checks(),
            platform_info=self._fake_platform(),
            start_time=0.0,
            end_time=5.0,
        )
        assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"

    def test_duration_s_matches_elapsed(self) -> None:
        result = _make_fake_result()
        artifact = build_artifact(
            recompute_result=result,
            checks=self._fake_checks(),
            platform_info=self._fake_platform(),
            start_time=100.0,
            end_time=115.0,
        )
        assert abs(artifact["duration_s"] - 15.0) < 0.001

    def test_harness_path_points_to_reproduce_script(self) -> None:
        result = _make_fake_result()
        artifact = build_artifact(
            recompute_result=result,
            checks=self._fake_checks(),
            platform_info=self._fake_platform(),
            start_time=0.0,
            end_time=5.0,
        )
        assert "reproduce_fover_headline" in str(artifact["harness_path"])


# ---------------------------------------------------------------------------
# tests for collect_platform_info
# ---------------------------------------------------------------------------

class TestCollectPlatformInfo:
    def test_has_python_version(self) -> None:
        info = collect_platform_info()
        assert "python_version" in info
        assert info["python_version"]

    def test_has_platform(self) -> None:
        info = collect_platform_info()
        assert "platform" in info

    def test_carnot_version_present(self) -> None:
        info = collect_platform_info()
        assert "carnot_version" in info


# ---------------------------------------------------------------------------
# tests for check_preconditions
# ---------------------------------------------------------------------------

class TestCheckPreconditions:
    def test_missing_corpus(self, tmp_path: Path) -> None:
        checks = check_preconditions(tmp_path)
        assert checks["fover_corpus_present"] is False

    def test_present_corpus(self, tmp_path: Path) -> None:
        corpus = tmp_path / "data" / "fover_corpus.jsonl"
        _write_fover_corpus(corpus, n_per_class=5)
        checks = check_preconditions(tmp_path)
        assert checks["fover_corpus_present"] is True

    def test_eval_module_importable(self, tmp_path: Path) -> None:
        # With a valid repo root, the eval module should be importable.
        repo_root = Path(__file__).resolve().parents[2]
        checks = check_preconditions(repo_root)
        assert checks["eval_module_importable"] is True
