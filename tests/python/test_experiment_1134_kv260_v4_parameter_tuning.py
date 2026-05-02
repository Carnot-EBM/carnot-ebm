"""Tests for Experiment 1134 — KV260 v4 parameter tuning + self-adaptive lambda.

These tests cover the pure-Python helpers introduced by the experiment
script. The full sweep is exercised by running the script once
end-to-end (which writes the artifact JSON); the tests validate the
artifact schema after the fact and unit-test the small helpers
(violations counter, verdict classifier, feasibility extrapolator,
spec-append) so future edits cannot silently break the contract.

Spec refs: REQ-HARDWARE-016, SCENARIO-HARDWARE-016, REQ-SAMPLE-012.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import experiment_1134_kv260_v4_parameter_tuning as exp1134

_REPO_ROOT = Path(__file__).parent.parent.parent
_ARTIFACT = _REPO_ROOT / "results" / "experiment_1134_kv260_v4_parameter_tuning.json"


def test_count_antiferro_violations_all_aligned():
    """All-+1 is the worst case for an antiferromagnetic ring: every
    edge is a violation, so the count must equal N. This is the start
    state the sampler boots into and the largest possible value the
    self-adaptive lambda update will see in a single sweep."""
    s = np.ones(8, dtype=np.int8)
    assert exp1134._count_antiferro_violations(s) == 8


def test_count_antiferro_violations_alternating_zero():
    """Alternating ±1 is the antiferromagnetic ground state on a ring of
    even length — zero violations. The self-adaptive update should
    plateau (lambda += 0) once the sampler reaches this configuration."""
    s = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)
    assert exp1134._count_antiferro_violations(s) == 0


def test_count_antiferro_violations_one_defect():
    """A single domain-wall defect should give exactly two violations on
    a periodic ring (the wall's two endpoints both fail)."""
    s = np.array([1, -1, 1, -1, 1, -1, 1, 1], dtype=np.int8)
    assert exp1134._count_antiferro_violations(s) == 2


def test_classify_verdict_below_threshold_wins():
    """If the best KL beats the threshold, that's the dream verdict
    regardless of whether the prior was beaten or self-adaptive helped."""
    v = exp1134._classify_verdict(
        kl_best=0.04, kl_prior=0.134, kl_threshold=0.05, self_adaptive_won=False
    )
    assert v == "kl_below_threshold"


def test_classify_verdict_self_adaptive_helped():
    """Improvement over prior plus self-adaptive winning the head-to-
    head must produce the self_adaptive_lambda_helped verdict so the
    operator can read the artifact and immediately know the
    contribution came from the adaptive branch."""
    v = exp1134._classify_verdict(
        kl_best=0.10, kl_prior=0.134, kl_threshold=0.05, self_adaptive_won=True
    )
    assert v == "self_adaptive_lambda_helped"


def test_classify_verdict_grid_only_improvement():
    """Improvement over prior but self-adaptive did NOT win — verdict
    is the plain ``kl_improved_not_below_threshold`` so the spec-append
    note focuses on the alpha/beta grid finding rather than the
    adaptive trajectory."""
    v = exp1134._classify_verdict(
        kl_best=0.10, kl_prior=0.134, kl_threshold=0.05, self_adaptive_won=False
    )
    assert v == "kl_improved_not_below_threshold"


def test_classify_verdict_no_improvement_full_sweep():
    """No improvement over prior — verdict reflects that the parameter
    space WAS mapped (not that the experiment failed); the spec gets
    the empirical-feasibility note so future planners don't re-propose
    the same sweep again. The 'doomed-rerun' discipline depends on
    this being recorded honestly."""
    v = exp1134._classify_verdict(
        kl_best=0.20, kl_prior=0.134, kl_threshold=0.05, self_adaptive_won=False
    )
    assert v == "kl_unchanged_parameter_space_mapped"


def test_extrapolate_feasibility_beta_decreasing():
    """When KL strictly decreases with beta, the log-linear fit must
    produce a non-trivial extrapolated beta_star and a fit_note that
    flags the result as an extrapolation, not a measurement."""
    fake = [
        {"beta": 2.0, "kl_v4_vs_gibbs": 0.5},
        {"beta": 3.0, "kl_v4_vs_gibbs": 0.2},
        {"beta": 4.0, "kl_v4_vs_gibbs": 0.08},
        {"beta": 5.0, "kl_v4_vs_gibbs": 0.04},
    ]
    out = exp1134._extrapolate_feasibility_beta(fake)
    assert out["feasibility_beta_estimate"] is not None
    assert out["feasibility_beta_estimate"] > 0.0
    assert out["feasibility_fit_slope_log_kl"] < 0.0
    assert "extrapolation" in out["feasibility_fit_note"]


def test_extrapolate_feasibility_beta_non_monotone():
    """When KL grows with beta (the actual exp1134 finding), the helper
    must report ``feasibility_beta_estimate = None`` and a falsifying
    note; we cannot fake an extrapolated beta in that case without
    misleading downstream readers."""
    fake = [
        {"beta": 2.0, "kl_v4_vs_gibbs": 0.5},
        {"beta": 3.0, "kl_v4_vs_gibbs": 1.0},
        {"beta": 4.0, "kl_v4_vs_gibbs": 5.0},
        {"beta": 5.0, "kl_v4_vs_gibbs": 30.0},
    ]
    out = exp1134._extrapolate_feasibility_beta(fake)
    assert out["feasibility_beta_estimate"] is None
    assert out["feasibility_fit_slope_log_kl"] > 0.0
    assert "falsified" in out["feasibility_fit_note"]


def test_extrapolate_feasibility_beta_insufficient_points():
    """The helper must refuse to fit on fewer than 2 points rather than
    raising — the rest of the pipeline must produce a clean artifact
    even if the beta sweep ever degenerates to a single measurement."""
    out = exp1134._extrapolate_feasibility_beta([{"beta": 2.0, "kl_v4_vs_gibbs": 0.5}])
    assert out["feasibility_beta_estimate"] is None
    assert out["feasibility_fit_note"] == "insufficient_points"


def test_append_feasibility_note_idempotent(tmp_path: Path):
    """Re-running the experiment must not duplicate the spec note. We
    write a synthetic spec file with the sentinel already present and
    confirm the helper returns False (no append)."""
    fake_spec = tmp_path / "spec.md"
    fake_spec.write_text("# v4\n\n## Empirical Feasibility (Exp 1134)\nalready here\n")
    feasibility = {"feasibility_beta_estimate": None, "feasibility_fit_note": "x"}
    appended = exp1134._append_feasibility_note_to_spec(
        feasibility, kl_best=0.1, best_beta=2.0, best_alpha=0.1, spec_path=fake_spec
    )
    assert appended is False
    # Content was not changed beyond the original.
    assert fake_spec.read_text().count("Empirical Feasibility (Exp 1134)") == 1


def test_append_feasibility_note_writes_paragraph(tmp_path: Path):
    """First-time append must succeed, return True, include the
    sentinel + the empirical numbers + the prior-best reference so a
    reader can locate exp1122's number for comparison."""
    fake_spec = tmp_path / "spec.md"
    fake_spec.write_text("# v4 spec\n\n(no feasibility section yet)\n")
    feasibility = {
        "feasibility_beta_estimate": 7.5,
        "feasibility_fit_note": "log-linear fit ok",
    }
    appended = exp1134._append_feasibility_note_to_spec(
        feasibility,
        kl_best=0.10,
        best_beta=2.0,
        best_alpha=0.1,
        spec_path=fake_spec,
    )
    assert appended is True
    body = fake_spec.read_text()
    assert "## Empirical Feasibility (Exp 1134)" in body
    assert "0.1000" in body  # current KL
    assert "0.134" in body  # exp1122 prior
    assert "beta ~ 7.50" in body


def test_append_feasibility_note_missing_spec(tmp_path: Path):
    """If the spec file is missing entirely, the helper must return
    False (not raise) so the experiment can still write the artifact —
    spec freshness is a soft-fail concern, not a hard-fail concern."""
    appended = exp1134._append_feasibility_note_to_spec(
        {"feasibility_beta_estimate": None, "feasibility_fit_note": "x"},
        kl_best=0.1,
        best_beta=2.0,
        best_alpha=0.1,
        spec_path=tmp_path / "does_not_exist.md",
    )
    assert appended is False


@pytest.mark.skipif(not _ARTIFACT.exists(), reason="artifact not yet generated")
def test_artifact_has_required_schema():
    """The deliverable JSON must carry every field listed in the
    milestone task. Missing any of them is a contract break with the
    conductor's reconciler — fail loudly here so we catch it in CI
    rather than discovering it from a stale ops/changelog entry."""
    art = json.loads(_ARTIFACT.read_text())
    required = [
        "experiment",
        "kl_v4_prior",
        "kl_v4_best",
        "kl_v4_below_05",
        "best_beta",
        "best_alpha",
        "self_adaptive_lambda_applied",
        "kl_v4_with_self_adaptive",
        "parameter_space_mapped",
        "kv260_v4_kl_below_05_or_feasibility_documented",
        "feasibility_notes",
        "honest_verdict",
    ]
    for key in required:
        assert key in art, f"missing field: {key}"
    assert art["experiment"] == 1134
    assert art["honest_verdict"] in {
        "kl_below_threshold",
        "kl_improved_not_below_threshold",
        "kl_unchanged_parameter_space_mapped",
        "self_adaptive_lambda_helped",
    }
    assert art["self_adaptive_lambda_applied"] is True
    assert art["parameter_space_mapped"] is True
    assert art["kv260_v4_kl_below_05_or_feasibility_documented"] is True


@pytest.mark.skipif(not _ARTIFACT.exists(), reason="artifact not yet generated")
def test_artifact_self_adaptive_trajectory_shape():
    """The 500-sweep adaptation loop checkpoints every 50 sweeps, so
    the trajectory list must have exactly 10 entries with monotone-
    non-decreasing lambda values (the update rule cannot decrement)."""
    art = json.loads(_ARTIFACT.read_text())
    traj = art["self_adaptive_trajectory"]
    assert len(traj) == 10
    lambdas = [t["lambda"] for t in traj]
    assert all(b >= a for a, b in zip(lambdas, lambdas[1:])), lambdas
