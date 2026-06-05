"""Tests for Exp 3871 Thesis-A part-b DTP1 headroom-confirmed beam search.

Spec refs: REQ-EBT-3871, SCENARIO-EBT-3871-HEADROOM,
SCENARIO-EBT-3871-ADJUDICATION.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "experiments" / (
    "experiment_3871_thesis_a_partb_dtp1_headroom_confirmed.py"
)
SPEC_PATH = ROOT / "openspec" / "capabilities" / "ebt-nrgpt" / "spec.md"


def _load_exp3871():
    spec = importlib.util.spec_from_file_location("exp3871", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_req_ebt_3871_spec_anchor_exists() -> None:
    """REQ-EBT-3871: OpenSpec declares the experiment contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-EBT-3871" in spec
    assert "SCENARIO-EBT-3871-HEADROOM" in spec
    assert "SCENARIO-EBT-3871-ADJUDICATION" in spec
    assert SCRIPT_PATH.name in spec
    assert "results/experiment_3871_thesis_a_partb_dtp1_headroom_confirmed.json" in spec


def test_scenario_ebt_3871_adjudication_thresholds() -> None:
    """SCENARIO-EBT-3871-ADJUDICATION: beam and argmin thresholds decide labels."""

    exp = _load_exp3871()

    artifact = exp.adjudicate(ar_best=0.84, ebt_argmin=0.0, ebt_beam=0.43)
    assert artifact.adjudication == "ARTIFACT"
    assert artifact.honest_verdict.startswith("complete: thesis_a_partb_ARTIFACT")

    fundamental = exp.adjudicate(ar_best=0.84, ebt_argmin=0.0, ebt_beam=0.0)
    assert fundamental.adjudication == "FUNDAMENTAL"
    assert fundamental.honest_verdict.startswith("complete: thesis_a_partb_FUNDAMENTAL")

    partial = exp.adjudicate(ar_best=0.84, ebt_argmin=0.0, ebt_beam=0.25)
    assert partial.adjudication == "INCONCLUSIVE"
    assert partial.honest_verdict.startswith("complete: thesis_a_partb_INCONCLUSIVE")


def test_scenario_ebt_3871_no_headroom_is_never_fundamental() -> None:
    """SCENARIO-EBT-3871-HEADROOM: failed AR headroom gates the verdict."""

    exp = _load_exp3871()

    adjudication = exp.adjudicate(ar_best=0.29, ebt_argmin=0.0, ebt_beam=0.0)

    assert adjudication.positive_control_passed is False
    assert adjudication.adjudication == "INCONCLUSIVE"
    assert adjudication.honest_verdict == "complete: thesis_a_partb_INCONCLUSIVE_no_headroom_ar0.290"
    assert "FUNDAMENTAL" not in adjudication.honest_verdict


def test_build_artifact_keeps_positive_control_bare_bool_and_principles() -> None:
    """REQ-EBT-3871: required artifact fields are schema-stable and annotated."""

    exp = _load_exp3871()
    seed_eval = exp.SeedEvaluation(
        seed=1,
        checkpoint_path="results/experiment_3871_seed1.pt",
        checkpoint_reused=True,
        n_heldout=100,
        ar_best_accuracy=0.84,
        ar_best_of_n=258,
        ar_forward_evals=103_200,
        ebt_argmin_accuracy=0.0,
        ebt_argmin_evals=103_200,
        ebt_beam_accuracy=0.0,
        ebt_beam_evals=825_600,
        matched_flops_ratio=1.0,
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
    assert isinstance(artifact["positive_control_passed"], bool)
    assert artifact["ar_best_accuracy"]["value"] == 0.84
    assert "principle" in artifact["ar_best_accuracy"]
    assert artifact["ebt_argmin_accuracy"]["value"] == 0.0
    assert artifact["ebt_beam_accuracy"]["value"] == 0.0
    assert artifact["adjudication"]["value"] == "FUNDAMENTAL"
    assert artifact["matched_flops_ratio"]["value"] == 1.0
    assert artifact["seeds_used"]["value"] == [1]
    assert artifact["random_seeds_used"]["value"] == [1]
    assert artifact["n_heldout"]["value"] == 100
    assert len(artifact["reproducibility_checksum"]["value"]) == 64
    assert exp.validate_artifact(artifact) == []


def test_validate_artifact_rejects_wrapped_positive_control() -> None:
    """REQ-EBT-3871: positive_control_passed must be a bare bool."""

    exp = _load_exp3871()
    bad = {
        "honest_verdict": "complete: bad",
        "positive_control_passed": {"value": True, "principle": "bad"},
    }

    errors = exp.validate_artifact(bad)

    assert "positive_control_passed must be a bare bool" in errors


def test_ebt_beam_generate_uses_cumulative_energy_beam_cost() -> None:
    """REQ-EBT-3871: copied P1 v2 beam decode expands beams over cumulative energy."""

    exp = _load_exp3871()

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
