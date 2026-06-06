"""Tests for Exp 3883 EBT System-2 K-curve.

Spec refs: REQ-EBT-3883, SCENARIO-EBT-3883-UPSTREAM,
SCENARIO-EBT-3883-SCHEMA, SCENARIO-EBT-3883-FALSIFICATION.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "experiments" / "experiment_3883_ebt_system2_kcurve.py"
SPEC_PATH = ROOT / "openspec" / "capabilities" / "ebt-nrgpt" / "spec.md"


def _load_exp3883():
    spec = importlib.util.spec_from_file_location("exp3883", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_req_ebt_3883_spec_anchor_exists() -> None:
    """REQ-EBT-3883: OpenSpec declares the K-curve contract before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-EBT-3883" in spec
    assert "SCENARIO-EBT-3883-UPSTREAM" in spec
    assert "SCENARIO-EBT-3883-SCHEMA" in spec
    assert "SCENARIO-EBT-3883-FALSIFICATION" in spec
    assert SCRIPT_PATH.name in spec
    assert "results/experiment_3883_ebt_system2_kcurve.json" in spec


def test_scenario_ebt_3883_import_uses_scripts_path() -> None:
    """REQ-EBT-3883: scaled harness import inserts scripts path directly."""

    exp = _load_exp3883()
    source = inspect.getsource(exp.load_scaled_harness)

    assert 'REPO_ROOT / "scripts"' in source
    assert "thesis_a_part_b_scaled" in source
    assert "scripts.thesis_a_part_b_scaled" not in source


def test_scenario_ebt_3883_falsification_classifies_curve_shapes() -> None:
    """SCENARIO-EBT-3883-FALSIFICATION: K=16 must beat K=1 for support."""

    exp = _load_exp3883()

    gain = exp.summarize_k_curve({1: 0.10, 2: 0.10, 4: 0.20, 8: 0.20, 16: 0.30})
    assert gain.k_curve_shape == "MONOTONE_GAIN"
    assert gain.best_k == 16
    assert gain.best_k_accuracy == 0.30
    assert gain.linear_slope > 0

    plateau = exp.summarize_k_curve({1: 0.10, 2: 0.10, 4: 0.10, 8: 0.10, 16: 0.10})
    assert plateau.k_curve_shape == "PLATEAU"
    assert plateau.best_k == 1
    assert plateau.best_k_accuracy == 0.10
    assert plateau.linear_slope == 0.0

    non_monotone = exp.summarize_k_curve({1: 0.10, 2: 0.30, 4: 0.20, 8: 0.30, 16: 0.30})
    assert non_monotone.k_curve_shape == "PLATEAU"
    assert non_monotone.best_k == 2

    degrading = exp.summarize_k_curve({1: 0.40, 2: 0.30, 4: 0.20, 8: 0.20, 16: 0.10})
    assert degrading.k_curve_shape == "DEGRADING"
    assert degrading.best_k == 1
    assert degrading.linear_slope < 0


def test_scenario_ebt_3883_schema_uses_bare_required_fields() -> None:
    """SCENARIO-EBT-3883-SCHEMA: complete artifacts use bare values."""

    exp = _load_exp3883()

    artifact = exp.build_artifact(
        accuracy_by_k={1: 0.10, 2: 0.10, 4: 0.20, 8: 0.20, 16: 0.30},
        n_heldout=100,
        seeds_used=[2],
        preconditions=exp.PreconditionReport(
            cuda=True,
            cuda_device_count=2,
            scaled_harness_import=True,
            upstream_positive_control=True,
            checkpoint_loaded=True,
        ),
        model_specs={"dim": 768, "selected_seed": 2, "k_values": [1, 2, 4, 8, 16]},
        random_seed=2,
        started_s=10.0,
        finished_s=75.0,
        inference_substrate="live_llm_inference",
    )

    assert artifact["accuracy_by_k"] == {"1": 0.10, "2": 0.10, "4": 0.20, "8": 0.20, "16": 0.30}
    assert artifact["k_curve_shape"] == "MONOTONE_GAIN"
    assert artifact["best_k"] == 16
    assert artifact["best_k_accuracy"] == 0.30
    assert artifact["honest_verdict"] == "complete: ebt_system2_SUPPORTED_acc_rises_with_k_best_k16_acc0.300"
    assert len(artifact["reproducibility_checksum"]) == 64
    for field in exp.BARE_REQUIRED_FIELDS:
        assert not (isinstance(artifact[field], dict) and {"value", "principle"} <= set(artifact[field]))
    assert exp.validate_artifact(artifact) == []


def test_scenario_ebt_3883_bounded_verdicts_and_blocked_schema() -> None:
    """SCENARIO-EBT-3883-FALSIFICATION: plateau/degrading curves are bounded."""

    exp = _load_exp3883()

    plateau = exp.build_artifact(
        accuracy_by_k={1: 0.0, 2: 0.0, 4: 0.0, 8: 0.0, 16: 0.0},
        n_heldout=100,
        seeds_used=[2],
        preconditions=exp.PreconditionReport(
            cuda=True,
            cuda_device_count=1,
            scaled_harness_import=True,
            upstream_positive_control=True,
            checkpoint_loaded=True,
        ),
        model_specs={"selected_seed": 2},
        random_seed=2,
        started_s=1.0,
        finished_s=64.0,
        inference_substrate="live_llm_inference",
    )
    assert plateau["k_curve_shape"] == "PLATEAU"
    assert plateau["honest_verdict"] == "complete: ebt_system2_BOUNDED_PLATEAU_no_usable_descent_signal_at_scale"
    assert exp.validate_artifact(plateau) == []

    degrading = exp.build_artifact(
        accuracy_by_k={1: 0.2, 2: 0.1, 4: 0.1, 8: 0.0, 16: 0.0},
        n_heldout=100,
        seeds_used=[2],
        preconditions=exp.PreconditionReport(
            cuda=True,
            cuda_device_count=1,
            scaled_harness_import=True,
            upstream_positive_control=True,
            checkpoint_loaded=True,
        ),
        model_specs={"selected_seed": 2},
        random_seed=2,
        started_s=1.0,
        finished_s=64.0,
        inference_substrate="live_llm_inference",
    )
    assert degrading["k_curve_shape"] == "DEGRADING"
    assert degrading["honest_verdict"] == "complete: ebt_system2_BOUNDED_DEGRADING_no_usable_descent_signal_at_scale"
    assert exp.validate_artifact(degrading) == []

    blocked = exp.build_blocked_artifact(
        honest_verdict="blocked_upstream_no_headroom",
        preconditions=exp.PreconditionReport(
            cuda=True,
            cuda_device_count=1,
            scaled_harness_import=True,
            upstream_positive_control=False,
            checkpoint_loaded=False,
            upstream_error="positive_control_passed was not true",
        ),
        model_specs={"selected_seed": None},
        started_s=1.0,
        finished_s=2.25,
    )
    assert blocked["accuracy_by_k"] == {}
    assert blocked["k_curve_shape"] is None
    assert blocked["best_k"] is None
    assert blocked["best_k_accuracy"] is None
    assert blocked["honest_verdict"] == "blocked_upstream_no_headroom"
    assert exp.validate_artifact(blocked) == []


def test_scenario_ebt_3883_upstream_checkpoint_resolution(tmp_path: Path) -> None:
    """SCENARIO-EBT-3883-UPSTREAM: selected checkpoint must load or block."""

    exp = _load_exp3883()
    checkpoint = tmp_path / "seed2.pt"
    torch.save({"ebt": {"weight": torch.tensor([1.0])}, "config": {"seed": 2}}, checkpoint)
    upstream_path = tmp_path / "experiment_3882.json"
    upstream_path.write_text(
        """
{
  "positive_control_passed": true,
  "random_seed": 2,
  "ebt_argmin_accuracy": 0.10,
  "ebt_beam_accuracy": 0.20,
  "model_specs": {"selected_seed": 2, "digits": 3},
  "seed_evaluations": [
    {"seed": 2, "checkpoint_path": "seed2.pt", "n_heldout": 100}
  ]
}
""",
        encoding="utf-8",
    )

    context, error = exp.load_upstream_context(upstream_path, tmp_path)

    assert error is None
    assert context is not None
    assert context.selected_seed == 2
    assert context.checkpoint_path == checkpoint
    assert context.decode_path == "beam"
    assert context.n_heldout == 100
    assert context.checkpoint_state["config"]["seed"] == 2

    upstream_path.write_text('{"positive_control_passed": false}', encoding="utf-8")
    bad_context, bad_error = exp.load_upstream_context(upstream_path, tmp_path)
    assert bad_context is None
    assert bad_error == "positive_control_passed was not true"

    upstream_path.write_text(
        """
{
  "positive_control_passed": true,
  "random_seed": 2,
  "seed_evaluations": [{"seed": 2, "checkpoint_path": "missing.pt"}]
}
""",
        encoding="utf-8",
    )
    missing_context, missing_error = exp.load_upstream_context(upstream_path, tmp_path)
    assert missing_context is None
    assert missing_error == "checkpoint did not load"


def test_validate_artifact_rejects_wrapped_or_malformed_fields() -> None:
    """SCENARIO-EBT-3883-SCHEMA: value/principle wrappers are invalid."""

    exp = _load_exp3883()
    bad = {
        "schema": exp.SCHEMA,
        "experiment": exp.EXPERIMENT_ID,
        "honest_verdict": "complete: bad",
        "accuracy_by_k": {"value": {"1": 0.1}, "principle": "bad"},
        "k_curve_shape": "BAD",
        "best_k": True,
        "best_k_accuracy": "bad",
        "n_heldout": True,
        "seeds_used": "bad",
        "preconditions_checked": [],
        "model_specs": [],
        "random_seed": "bad",
        "reproducibility_checksum": "short",
        "duration_s": -1.0,
        "inference_substrate": {},
    }

    errors = exp.validate_artifact(bad)

    assert "accuracy_by_k must be a bare value, not a value/principle wrapper" in errors
    assert "k_curve_shape must be MONOTONE_GAIN, PLATEAU, DEGRADING, or null" in errors
    assert "best_k must be an integer or null" in errors
    assert "best_k_accuracy must be numeric or null" in errors
    assert "n_heldout must be an integer" in errors
    assert "seeds_used must be a list" in errors
    assert "preconditions_checked must be an object" in errors
    assert "model_specs must be an object" in errors
    assert "random_seed must be an integer or null" in errors
    assert "reproducibility_checksum must be a sha256 hex string" in errors
    assert "duration_s must be non-negative" in errors
    assert "inference_substrate must be a string" in errors


def test_scenario_ebt_3883_error_branches_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-EBT-3883-UPSTREAM: malformed inputs fail closed."""

    exp = _load_exp3883()

    assert exp._as_float("bad", default=1.25) == 1.25
    assert exp._selected_seed({"random_seed": True, "model_specs": {"selected_seed": False}}) is None
    assert exp._checkpoint_for_seed({"n_heldout": 7}, selected_seed=2) == (None, 7)
    assert exp._checkpoint_for_seed(
        {"n_heldout": 7, "seed_evaluations": [{"seed": 1, "checkpoint_path": "a.pt"}]},
        selected_seed=2,
    ) == (None, 7)

    missing_context, missing_error = exp.load_upstream_context(tmp_path / "missing.json", tmp_path)
    assert missing_context is None
    assert missing_error == "experiment_3882 artifact missing"

    unreadable = tmp_path / "bad.json"
    unreadable.write_text("{", encoding="utf-8")
    unreadable_context, unreadable_error = exp.load_upstream_context(unreadable, tmp_path)
    assert unreadable_context is None
    assert unreadable_error == "experiment_3882 artifact unreadable"

    no_seed = tmp_path / "no_seed.json"
    no_seed.write_text('{"positive_control_passed": true}', encoding="utf-8")
    no_seed_context, no_seed_error = exp.load_upstream_context(no_seed, tmp_path)
    assert no_seed_context is None
    assert no_seed_error == "selected seed missing"

    no_checkpoint = tmp_path / "no_checkpoint.json"
    no_checkpoint.write_text(
        '{"positive_control_passed": true, "random_seed": 2, "seed_evaluations": []}',
        encoding="utf-8",
    )
    no_checkpoint_context, no_checkpoint_error = exp.load_upstream_context(no_checkpoint, tmp_path)
    assert no_checkpoint_context is None
    assert no_checkpoint_error == "selected checkpoint path missing"

    bad_checkpoint = tmp_path / "bad_checkpoint.pt"
    torch.save({"not_ebt": {}}, bad_checkpoint)
    bad_upstream = tmp_path / "bad_checkpoint.json"
    bad_upstream.write_text(
        """
{
  "positive_control_passed": true,
  "random_seed": 2,
  "seed_evaluations": [{"seed": 2, "checkpoint_path": "bad_checkpoint.pt"}]
}
""",
        encoding="utf-8",
    )
    bad_checkpoint_context, bad_checkpoint_error = exp.load_upstream_context(bad_upstream, tmp_path)
    assert bad_checkpoint_context is None
    assert bad_checkpoint_error == "checkpoint did not load"

    try:
        exp.summarize_k_curve({1: 0.0})
    except ValueError as exc:
        assert "missing K values" in str(exc)
    else:  # pragma: no cover - assertion branch.
        raise AssertionError("summarize_k_curve should reject incomplete curves")

    preconditions = exp.PreconditionReport(cuda=False, cuda_device_count=0, scaled_harness_import=False)
    updated = exp._with_precondition(preconditions, upstream_error="x")
    assert updated.upstream_error == "x"

    out = tmp_path / "nested" / "artifact.json"
    blocked = exp.build_blocked_artifact(
        honest_verdict="blocked_no_cuda",
        preconditions=updated,
        model_specs={},
        started_s=1.0,
        finished_s=2.0,
    )
    exp.write_artifact(out, blocked)
    assert out.read_text(encoding="utf-8").endswith("\n")

    malformed = dict(blocked)
    malformed.pop("schema")
    malformed["honest_verdict"] = "not_terminal"
    malformed["accuracy_by_k"] = []
    errors = exp.validate_artifact(malformed)
    assert any(error.startswith("missing required fields") for error in errors)
    assert "honest_verdict must start with complete: or blocked_" in errors
    assert "accuracy_by_k must be an object" in errors

    malformed["accuracy_by_k"] = {1: 0.0}
    assert "accuracy_by_k keys must be strings" in exp.validate_artifact(malformed)

    malformed["accuracy_by_k"] = {"1": True}
    assert "accuracy_by_k values must be numeric" in exp.validate_artifact(malformed)
