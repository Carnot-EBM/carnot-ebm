"""Tests for Exp 3893 Thesis-A fundamental negative replication.

Spec refs: REQ-EBT-3893, SCENARIO-EBT-3893-REPLICATION-GATE,
SCENARIO-EBT-3893-SCHEMA, SCENARIO-EBT-3893-REUSE.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "experiments" / "experiment_3893_ebt_fundamental_replication.py"
SPEC_PATH = ROOT / "openspec" / "capabilities" / "ebt-nrgpt" / "spec.md"


def _load_exp3893():
    spec = importlib.util.spec_from_file_location("exp3893", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed_eval(exp, *, seed: int, ar: float, argmin: float | None, beam: float | None, reused: bool = False):
    return SimpleNamespace(
        seed=seed,
        checkpoint_path=f"results/experiment_3893_seed{seed}.pt",
        checkpoint_reused=reused,
        n_heldout=100,
        ar_best_accuracy=ar,
        ar_best_of_n=1612,
        ar_forward_evals=644_800,
        ebt_argmin_accuracy=argmin,
        ebt_argmin_evals=103_200 if argmin is not None else None,
        ebt_beam_accuracy=beam,
        ebt_beam_evals=645_000 if beam is not None else None,
        matched_flops_ratio=1.0003 if beam is not None else None,
        samples=[{"prompt": "043+359=", "true": "0402"}] if beam is not None else [],
        training_diverged=False,
    )


def _preconditions(exp):
    return exp.PreconditionReport(
        cuda=True,
        cuda_device_count=2,
        scaled_harness_import=True,
        exp3882_pipeline_import=True,
        exp3882_reusable_functions=True,
    )


def test_req_ebt_3893_spec_anchor_exists() -> None:
    """REQ-EBT-3893: OpenSpec declares the replication contract before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-EBT-3893" in spec
    assert "SCENARIO-EBT-3893-REPLICATION-GATE" in spec
    assert "SCENARIO-EBT-3893-SCHEMA" in spec
    assert "SCENARIO-EBT-3893-REUSE" in spec
    assert SCRIPT_PATH.name in spec
    assert "results/experiment_3893_ebt_fundamental_replication.json" in spec


def test_scenario_ebt_3893_reuses_exp3882_measurement_path() -> None:
    """SCENARIO-EBT-3893-REUSE: seed evaluation delegates to Exp 3882."""

    exp = _load_exp3893()
    load_source = inspect.getsource(exp.load_exp3882_pipeline)
    eval_source = inspect.getsource(exp.evaluate_replication_seed)
    policy_source = inspect.getsource(exp.fresh_checkpoint_policy)

    assert exp.EXP3882_REL_PATH.name == "experiment_3882_thesis_a_partb_killgate.py"
    assert "EXP3882_REL_PATH" in load_source
    assert "pipeline._load_or_train_models" in eval_source
    assert "pipeline.evaluate_seed" in eval_source
    assert "pipeline._scaled_artifact_checkpoint" in policy_source
    assert "return None" in policy_source


def test_scenario_ebt_3893_replication_gate_replicates() -> None:
    """SCENARIO-EBT-3893-REPLICATION-GATE: two failed EBT seeds bank negative."""

    exp = _load_exp3893()
    artifact = exp.build_artifact(
        seed_evaluations=[
            _seed_eval(exp, seed=4, ar=0.88, argmin=0.0, beam=0.0),
            _seed_eval(exp, seed=5, ar=0.92, argmin=0.0, beam=0.0),
        ],
        all_configured_seeds=[4, 5],
        preconditions=_preconditions(exp),
        model_specs={"dim": 768, "beam": 8, "fresh_seed_policy": "no_checkpoint_reuse"},
        started_s=10.0,
        finished_s=75.0,
        inference_substrate="live_llm_inference",
    )

    assert artifact["replication_outcome"] == "REPLICATED"
    assert artifact["ebt_beam_accuracy_mean"] == 0.0
    assert artifact["ebt_argmin_accuracy_mean"] == 0.0
    assert artifact["ar_best_accuracy_mean"] == 0.90
    assert artifact["n_valid_seeds"] == 2
    assert artifact["retrained"] is True
    assert artifact["honest_verdict"] == (
        "complete: ebt_fundamental_REPLICATED_beam0.000_argmin0.000_vs_ar0.900_"
        "nseeds2_energy_as_generator_banked_negative"
    )
    assert exp.validate_artifact(artifact) == []


def test_scenario_ebt_3893_replication_gate_refutes() -> None:
    """SCENARIO-EBT-3893-REPLICATION-GATE: beam recovery refutes .359."""

    exp = _load_exp3893()
    artifact = exp.build_artifact(
        seed_evaluations=[
            _seed_eval(exp, seed=4, ar=0.80, argmin=0.05, beam=0.45),
            _seed_eval(exp, seed=5, ar=0.80, argmin=0.05, beam=0.35),
        ],
        all_configured_seeds=[4, 5],
        preconditions=_preconditions(exp),
        model_specs={"dim": 768, "beam": 8},
        started_s=1.0,
        finished_s=65.0,
        inference_substrate="live_llm_inference",
    )

    assert artifact["replication_outcome"] == "REFUTED"
    assert artifact["ebt_beam_accuracy_mean"] == 0.40
    assert artifact["honest_verdict"] == (
        "complete: ebt_fundamental_REFUTED_beam0.400_recovers_vs_ar0.800_"
        "359_verdict_was_artifact"
    )
    assert exp.validate_artifact(artifact) == []


def test_scenario_ebt_3893_invalid_ar_seed_is_inconclusive() -> None:
    """SCENARIO-EBT-3893-REPLICATION-GATE: collapsed AR seed is not valid."""

    exp = _load_exp3893()
    artifact = exp.build_artifact(
        seed_evaluations=[
            _seed_eval(exp, seed=4, ar=0.89, argmin=0.0, beam=0.0),
            _seed_eval(exp, seed=5, ar=0.0, argmin=None, beam=None),
        ],
        all_configured_seeds=[4, 5],
        preconditions=_preconditions(exp),
        model_specs={"dim": 768, "beam": 8},
        started_s=1.0,
        finished_s=65.0,
        inference_substrate="live_llm_inference",
    )

    assert artifact["replication_outcome"] == "INCONCLUSIVE"
    assert artifact["n_valid_seeds"] == 1
    assert artifact["ar_best_accuracy_mean"] == 0.89
    assert artifact["matched_flops_ratio"] == 1.0003
    assert artifact["invalid_seed_reasons"] == {"5": "ar_positive_control_failed_0.000"}
    assert artifact["honest_verdict"] == "complete: ebt_fundamental_INCONCLUSIVE_only1_valid_seeds_ar_control_collapsed"
    assert exp.validate_artifact(artifact) == []


def test_scenario_ebt_3893_edge_gate_paths() -> None:
    """SCENARIO-EBT-3893-REPLICATION-GATE: edge outcomes stay explicit."""

    exp = _load_exp3893()

    missing_metrics = exp.adjudicate_replication(
        ar_best_accuracy_mean=0.80,
        ebt_argmin_accuracy_mean=None,
        ebt_beam_accuracy_mean=0.0,
        n_valid_seeds=2,
    )
    assert missing_metrics.replication_outcome == "INCONCLUSIVE"
    assert missing_metrics.honest_verdict == "complete: ebt_fundamental_INCONCLUSIVE_only2_valid_seeds_metrics_missing"

    threshold = exp.adjudicate_replication(
        ar_best_accuracy_mean=0.80,
        ebt_argmin_accuracy_mean=0.10,
        ebt_beam_accuracy_mean=0.25,
        n_valid_seeds=2,
    )
    assert threshold.replication_outcome == "INCONCLUSIVE"
    assert threshold.honest_verdict == (
        "complete: ebt_fundamental_INCONCLUSIVE_beam0.250_argmin0.100_vs_ar0.800_"
        "thresholds_not_decisive_nseeds2"
    )

    artifact = exp.build_artifact(
        seed_evaluations=[
            _seed_eval(exp, seed=4, ar=0.0, argmin=None, beam=None),
            _seed_eval(exp, seed=5, ar=0.80, argmin=None, beam=None),
        ],
        all_configured_seeds=[4, 5],
        preconditions=_preconditions(exp),
        model_specs={"dim": 768, "beam": 8},
        started_s=1.0,
        finished_s=65.0,
        inference_substrate="live_llm_inference",
    )
    assert artifact["ar_best_accuracy_mean"] is None
    assert artifact["invalid_seed_reasons"] == {
        "4": "ar_positive_control_failed_0.000",
        "5": "ebt_metrics_missing",
    }
    assert artifact["n_valid_seeds"] == 0

    @dataclass(frozen=True)
    class Row:
        value: int

    assert exp._as_plain_dict(Row(value=3)) == {"value": 3}


def test_scenario_ebt_3893_schema_handles_blocked_and_malformed_fields(tmp_path: Path) -> None:
    """SCENARIO-EBT-3893-SCHEMA: blocked artifacts are bare and malformed fields fail."""

    exp = _load_exp3893()
    blocked = exp.build_blocked_artifact(
        honest_verdict="blocked_no_cuda",
        preconditions=exp.PreconditionReport(
            cuda=False,
            cuda_device_count=0,
            scaled_harness_import=False,
            exp3882_pipeline_import=True,
            exp3882_reusable_functions=True,
        ),
        model_specs={"dim": 768},
        started_s=3.0,
        finished_s=4.25,
    )

    assert blocked["replication_outcome"] == "INCONCLUSIVE"
    assert blocked["n_valid_seeds"] == 0
    assert blocked["retrained"] is False
    assert blocked["duration_s"] == 1.25
    assert exp.validate_artifact(blocked) == []
    for field in exp.BARE_REQUIRED_FIELDS:
        assert not (isinstance(blocked[field], dict) and {"value", "principle"} <= set(blocked[field]))

    out = tmp_path / "result.json"
    exp.write_artifact(out, blocked)
    assert out.read_text(encoding="utf-8").endswith("\n")

    bad = dict(blocked)
    bad["honest_verdict"] = "not_terminal"
    bad["replication_outcome"] = "BAD"
    bad["ebt_beam_accuracy_mean"] = {"value": 0.0, "principle": "bad"}
    bad["n_valid_seeds"] = True
    bad["n_heldout"] = True
    bad["retrained"] = "yes"
    bad["seeds_used"] = "bad"
    bad["random_seeds_used"] = "bad"
    bad["preconditions_checked"] = []
    bad["model_specs"] = []
    bad["random_seed"] = "bad"
    bad["reproducibility_checksum"] = "short"
    bad["duration_s"] = -1.0
    bad["inference_substrate"] = {}

    errors = exp.validate_artifact(bad)

    assert "honest_verdict must start with complete: or blocked_" in errors
    assert "replication_outcome must be REPLICATED, REFUTED, or INCONCLUSIVE" in errors
    assert "ebt_beam_accuracy_mean must be a bare value, not a value/principle wrapper" in errors
    assert "n_valid_seeds must be an integer" in errors
    assert "n_heldout must be an integer" in errors
    assert "retrained must be a bare bool" in errors
    assert "seeds_used must be a list" in errors
    assert "random_seeds_used must be a list" in errors
    assert "preconditions_checked must be an object" in errors
    assert "model_specs must be an object" in errors
    assert "random_seed must be an integer or null" in errors
    assert "reproducibility_checksum must be a sha256 hex string" in errors
    assert "duration_s must be non-negative" in errors
    assert "inference_substrate must be a string" in errors

    missing_errors = exp.validate_artifact({"honest_verdict": "complete: bad"})
    assert any(error.startswith("missing required fields:") for error in missing_errors)


def test_scenario_ebt_3893_fresh_checkpoint_policy(tmp_path: Path) -> None:
    """SCENARIO-EBT-3893-REUSE: fresh seed checkpoint paths avoid reuse."""

    exp = _load_exp3893()
    base = exp.fresh_checkpoint_path(seed=4, results_dir=tmp_path)
    assert base == tmp_path / "experiment_3893_ebt_fundamental_replication_seed4.pt"

    base.write_text("existing", encoding="utf-8")
    rerun = exp.fresh_checkpoint_path(seed=4, results_dir=tmp_path)
    assert rerun == tmp_path / "experiment_3893_ebt_fundamental_replication_seed4_rerun1.pt"

    def original_checkpoint_path(seed: int) -> Path:
        return tmp_path / f"old_seed{seed}.pt"

    def original_scaled_artifact_checkpoint(seed: int) -> Path:
        return tmp_path / f"scaled_seed{seed}.pt"

    pipeline = SimpleNamespace(
        _checkpoint_path=original_checkpoint_path,
        _scaled_artifact_checkpoint=original_scaled_artifact_checkpoint,
    )
    original_checkpoint_path = pipeline._checkpoint_path
    original_scaled_artifact_checkpoint = pipeline._scaled_artifact_checkpoint
    with exp.fresh_checkpoint_policy(pipeline, results_dir=tmp_path):
        assert pipeline._scaled_artifact_checkpoint(4) is None
        assert pipeline._checkpoint_path(4).name == "experiment_3893_ebt_fundamental_replication_seed4_rerun1.pt"
    assert pipeline._checkpoint_path is original_checkpoint_path
    assert pipeline._scaled_artifact_checkpoint is original_scaled_artifact_checkpoint
