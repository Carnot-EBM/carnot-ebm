"""Tests for Exp 3882 Thesis-A part-b kill-gate beam-search rerun.

Spec refs: REQ-EBT-3882, SCENARIO-EBT-3882-IMPORT,
SCENARIO-EBT-3882-SCHEMA, SCENARIO-EBT-3882-ADJUDICATION.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "experiments" / "experiment_3882_thesis_a_partb_killgate.py"
SPEC_PATH = ROOT / "openspec" / "capabilities" / "ebt-nrgpt" / "spec.md"


def _load_exp3882():
    spec = importlib.util.spec_from_file_location("exp3882", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_req_ebt_3882_spec_anchor_exists() -> None:
    """REQ-EBT-3882: OpenSpec declares the rerun contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-EBT-3882" in spec
    assert "SCENARIO-EBT-3882-IMPORT" in spec
    assert "SCENARIO-EBT-3882-SCHEMA" in spec
    assert "SCENARIO-EBT-3882-ADJUDICATION" in spec
    assert SCRIPT_PATH.name in spec
    assert "results/experiment_3882_thesis_a_partb_killgate.json" in spec


def test_scenario_ebt_3882_import_uses_scripts_path() -> None:
    """SCENARIO-EBT-3882-IMPORT: scaled harness import avoids package import."""

    exp = _load_exp3882()
    source = inspect.getsource(exp.load_scaled_modules)

    assert 'REPO_ROOT / "scripts"' in source
    assert "thesis_a_part_b_scaled" in source
    assert "scripts.thesis_a_part_b_scaled" not in source


def test_scenario_ebt_3882_adjudication_thresholds() -> None:
    """SCENARIO-EBT-3882-ADJUDICATION: thresholds classify the kill-gate."""

    exp = _load_exp3882()

    artifact = exp.adjudicate(ar_best=0.84, ebt_argmin=0.0, ebt_beam=0.42)
    assert artifact.adjudication == "ARTIFACT"
    assert artifact.honest_verdict.startswith("complete: thesis_a_partb_ARTIFACT")

    fundamental = exp.adjudicate(ar_best=0.84, ebt_argmin=0.0, ebt_beam=0.0)
    assert fundamental.adjudication == "FUNDAMENTAL"
    assert fundamental.honest_verdict.startswith("complete: thesis_a_partb_FUNDAMENTAL")

    partial = exp.adjudicate(ar_best=0.84, ebt_argmin=0.0, ebt_beam=0.25)
    assert partial.adjudication == "INCONCLUSIVE"
    assert partial.honest_verdict.startswith("complete: thesis_a_partb_INCONCLUSIVE")


def test_scenario_ebt_3882_no_headroom_never_fundamental() -> None:
    """SCENARIO-EBT-3882-ADJUDICATION: failed headroom blocks strong labels."""

    exp = _load_exp3882()

    result = exp.adjudicate(ar_best=0.39, ebt_argmin=0.0, ebt_beam=0.0)

    assert result.positive_control_passed is False
    assert result.adjudication == "INCONCLUSIVE"
    assert result.honest_verdict == "complete: thesis_a_partb_INCONCLUSIVE_no_headroom_ar0.390"
    assert "FUNDAMENTAL" not in result.honest_verdict


def test_scenario_ebt_3882_schema_uses_bare_required_fields() -> None:
    """SCENARIO-EBT-3882-SCHEMA: required fields are not value/principle dicts."""

    exp = _load_exp3882()
    seed_eval = exp.SeedEvaluation(
        seed=1,
        checkpoint_path="results/experiment_3882_seed1.pt",
        checkpoint_reused=True,
        n_heldout=100,
        ar_best_accuracy=0.84,
        ar_best_of_n=1612,
        ar_forward_evals=644_800,
        ebt_argmin_accuracy=0.0,
        ebt_argmin_evals=103_200,
        ebt_beam_accuracy=0.0,
        ebt_beam_evals=645_000,
        matched_flops_ratio=1.0003,
        samples=[{"prompt": "043+359=", "true": "0402"}],
        training_diverged=False,
    )

    artifact = exp.build_artifact(
        seed_evaluations=[seed_eval],
        all_configured_seeds=[1, 2, 3],
        preconditions=exp.PreconditionReport(cuda=True, cuda_device_count=2, scaled_harness_import=True),
        model_specs={"dim": 768, "n_layers": 4, "beam": 8},
        started_s=10.0,
        finished_s=75.0,
        inference_substrate="live_llm_inference",
    )

    assert artifact["positive_control_passed"] is True
    assert artifact["ar_best_accuracy"] == 0.84
    assert artifact["ebt_argmin_accuracy"] == 0.0
    assert artifact["ebt_beam_accuracy"] == 0.0
    assert artifact["adjudication"] == "FUNDAMENTAL"
    assert artifact["matched_flops_ratio"] == 1.0003
    assert artifact["seeds_used"] == [1]
    assert artifact["random_seeds_used"] == [1]
    assert artifact["n_heldout"] == 100
    assert len(artifact["reproducibility_checksum"]) == 64
    for field in exp.BARE_REQUIRED_FIELDS:
        assert not (isinstance(artifact[field], dict) and {"value", "principle"} <= set(artifact[field]))
    assert exp.validate_artifact(artifact) == []


def test_scenario_ebt_3882_schema_handles_no_headroom_and_blocked(tmp_path: Path) -> None:
    """SCENARIO-EBT-3882-SCHEMA: no-headroom and blocked artifacts stay bare."""

    exp = _load_exp3882()
    no_headroom = exp.SeedEvaluation(
        seed=2,
        checkpoint_path="results/experiment_3882_seed2.pt",
        checkpoint_reused=False,
        n_heldout=100,
        ar_best_accuracy=0.12,
        ar_best_of_n=1612,
        ar_forward_evals=644_800,
        ebt_argmin_accuracy=None,
        ebt_argmin_evals=None,
        ebt_beam_accuracy=None,
        ebt_beam_evals=None,
        matched_flops_ratio=None,
        samples=[],
        training_diverged=False,
    )
    better_no_headroom = exp.SeedEvaluation(
        seed=3,
        checkpoint_path="results/experiment_3882_seed3.pt",
        checkpoint_reused=False,
        n_heldout=100,
        ar_best_accuracy=0.30,
        ar_best_of_n=1612,
        ar_forward_evals=644_800,
        ebt_argmin_accuracy=None,
        ebt_argmin_evals=None,
        ebt_beam_accuracy=None,
        ebt_beam_evals=None,
        matched_flops_ratio=None,
        samples=[],
        training_diverged=False,
    )

    artifact = exp.build_artifact(
        seed_evaluations=[no_headroom, better_no_headroom],
        all_configured_seeds=[1, 2, 3],
        preconditions=exp.PreconditionReport(cuda=True, cuda_device_count=1, scaled_harness_import=True),
        model_specs={"dim": 768, "n_layers": 4, "beam": 8},
        started_s=1.0,
        finished_s=2.0,
        inference_substrate="live_llm_inference",
    )

    assert artifact["positive_control_passed"] is False
    assert artifact["random_seed"] == 3
    assert artifact["honest_verdict"] == "complete: thesis_a_partb_INCONCLUSIVE_no_headroom_ar0.300"
    assert exp.validate_artifact(artifact) == []

    blocked = exp.build_blocked_artifact(
        honest_verdict="blocked_no_cuda",
        preconditions=exp.PreconditionReport(cuda=False, cuda_device_count=0, scaled_harness_import=False),
        model_specs={"dim": 768},
        started_s=3.0,
        finished_s=4.25,
    )
    assert blocked["honest_verdict"] == "blocked_no_cuda"
    assert blocked["positive_control_passed"] is False
    assert blocked["duration_s"] == 1.25
    assert exp.validate_artifact(blocked) == []

    out = tmp_path / "result.json"
    exp.write_artifact(out, blocked)
    assert out.read_text(encoding="utf-8").endswith("\n")

    with pytest.raises(ValueError, match="at least one seed evaluation"):
        exp.build_artifact(
            seed_evaluations=[],
            all_configured_seeds=[1, 2, 3],
            preconditions=exp.PreconditionReport(cuda=True, cuda_device_count=1, scaled_harness_import=True),
            model_specs={"dim": 768},
            started_s=1.0,
            finished_s=2.0,
            inference_substrate="live_llm_inference",
        )


def test_validate_artifact_rejects_wrapped_bare_fields() -> None:
    """SCENARIO-EBT-3882-SCHEMA: value/principle wrappers are invalid."""

    exp = _load_exp3882()
    bad = {
        "honest_verdict": "complete: bad",
        "positive_control_passed": True,
        "ar_best_accuracy": {"value": 0.84, "principle": "bad"},
    }

    errors = exp.validate_artifact(bad)

    assert "ar_best_accuracy must be a bare value, not a value/principle wrapper" in errors
    assert "positive_control_passed must be a bare bool" not in errors

    more_bad = {
        "honest_verdict": "not_terminal",
        "positive_control_passed": {"value": True, "principle": "bad"},
        "ar_best_accuracy": "bad",
        "ebt_argmin_accuracy": None,
        "ebt_beam_accuracy": None,
        "matched_flops_ratio": None,
        "adjudication": "BAD",
        "seeds_used": "bad",
        "random_seeds_used": "bad",
        "n_heldout": True,
        "preconditions_checked": [],
        "model_specs": [],
        "reproducibility_checksum": "short",
        "duration_s": -1.0,
        "inference_substrate": {},
    }

    more_errors = exp.validate_artifact(more_bad)

    assert "honest_verdict must start with complete: or blocked_" in more_errors
    assert "positive_control_passed must be a bare bool" in more_errors
    assert "ar_best_accuracy must be numeric or null" in more_errors
    assert "adjudication must be ARTIFACT, FUNDAMENTAL, or INCONCLUSIVE" in more_errors
    assert "seeds_used must be a list" in more_errors
    assert "random_seeds_used must be a list" in more_errors
    assert "n_heldout must be an integer" in more_errors
    assert "preconditions_checked must be an object" in more_errors
    assert "model_specs must be an object" in more_errors
    assert "reproducibility_checksum must be a sha256 hex string" in more_errors
    assert "duration_s must be non-negative" in more_errors
    assert "inference_substrate must be a string" in more_errors


def test_ebt_beam_generate_uses_cumulative_energy_beam_cost() -> None:
    """REQ-EBT-3882: global beam expands cumulative energy beams."""

    exp = _load_exp3882()

    class MockEBT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = torch.nn.Embedding(exp.VOCAB, 8)

        def forward(self, orig: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
            energy = torch.zeros((orig.shape[0], orig.shape[1], 1), dtype=pred.dtype)
            energy[:, -1, 0] = torch.arange(orig.shape[0], dtype=pred.dtype)
            return energy

    ids, n_energy_evals = exp.ebt_beam_generate(
        MockEBT(),
        pid=[2, 3, 4],
        ans_len=2,
        device=torch.device("cpu"),
        beam=2,
        topk=2,
    )

    assert len(ids) == 2
    assert n_energy_evals == exp.VOCAB * 3


def test_matched_ar_best_of_n_tracks_beam_flops() -> None:
    """REQ-EBT-3882: AR best-of-N is selected by Exp 3727 budget matcher."""

    exp = _load_exp3882()

    best_of_n = exp.matched_ar_best_of_n(n_heldout=100, ans_len=4, beam=8)

    assert best_of_n in {1612, 1613}
