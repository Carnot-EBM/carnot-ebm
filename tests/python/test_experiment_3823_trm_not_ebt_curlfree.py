"""Tests for Exp 3823 TRM curl-free falsification.

Spec refs: REQ-3823, SCENARIO-3823, SCENARIO-3823-POSITIVE-CONTROL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.experiments import experiment_3823_trm_not_ebt_curlfree as mod


def test_spec_declares_req_3823() -> None:
    """REQ-3823: OpenSpec declares the curl-free diagnostic before implementation."""
    spec = Path("openspec/capabilities/phase3-kona/spec.md").read_text(encoding="utf-8")
    assert "REQ-3823" in spec
    assert "SCENARIO-3823" in spec
    assert "SCENARIO-3823-POSITIVE-CONTROL" in spec


def test_blocked_when_exp3821_source_is_not_loadable_checkpoint(tmp_path: Path) -> None:
    """SCENARIO-3823: missing loadable TRM checkpoint blocks curl claims."""
    source_artifact = tmp_path / "experiment_3821.json"
    source_artifact.write_text(
        json.dumps({"trm_checkpoint_source": "nano-trm tiny-train (0-epoch probe)"}),
        encoding="utf-8",
    )

    artifact = mod.build_artifact(source_artifact_path=source_artifact)

    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert mod.field_value(artifact["n_states_sampled"]) == 0
    assert mod.field_value(artifact["jacobian_antisymmetry_fraction"]) is None
    assert mod.field_value(artifact["scalar_potential_fit_residual"]) is None
    assert mod.field_value(artifact["positive_control_fit_residual"]) is None
    assert mod.field_value(artifact["preconditions_checked"])["trm_checkpoint_loadable"] is False
    assert mod.field_value(artifact["preconditions_checked"])["trm_checkpoint_source"] == (
        "nano-trm tiny-train (0-epoch probe)"
    )

    for field_name in mod.REQUIRED_PRINCIPLES:
        assert "principle" in artifact[field_name]
        assert artifact[field_name]["principle"] == mod.REQUIRED_PRINCIPLES[field_name]


def test_positive_control_has_low_curl_and_low_fit_residual() -> None:
    """SCENARIO-3823-POSITIVE-CONTROL: conservative fields are recognized."""
    states, deltas = mod.sample_quadratic_conservative_field(
        n_instances=64,
        latent_dim=3,
        steps=2,
        random_seed=7,
    )

    residual = mod.scalar_potential_fit_residual(states, deltas)
    asymmetry = mod.jacobian_antisymmetry_fraction(
        lambda h: mod.quadratic_conservative_delta(h, torch.eye(3, dtype=h.dtype)),
        states[:8],
    )

    assert residual < 1e-6
    assert asymmetry < 1e-6


def test_asymmetric_linear_field_has_large_curl_and_fit_residual() -> None:
    """REQ-3823: antisymmetric update fields cannot fit scalar-energy descent."""
    matrix = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    states = torch.randn(80, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(11))
    deltas = states @ matrix.T

    residual = mod.scalar_potential_fit_residual(states, deltas)
    asymmetry = mod.jacobian_antisymmetry_fraction(lambda h: h @ matrix.T, states[:10])

    assert residual > 0.95
    assert asymmetry > 0.95
    assert mod.classify_verdict(asymmetry, residual, positive_control_residual=1e-9).startswith(
        "complete: trm_not_ebt_curlfree_falsified"
    )


def test_synthetic_checkpoint_runs_full_nonblocked_path(tmp_path: Path) -> None:
    """REQ-3823: a loadable checkpoint produces measured curl metrics over M>=50 states."""
    checkpoint = tmp_path / "trm_linear.pt"
    torch.save({"update_matrix": [[0.0, -1.0], [1.0, 0.0]], "update_bias": [0.0, 0.0]}, checkpoint)
    source_artifact = tmp_path / "experiment_3821.json"
    source_artifact.write_text(json.dumps({"trm_checkpoint_source": str(checkpoint)}), encoding="utf-8")

    artifact = mod.build_artifact(
        source_artifact_path=source_artifact,
        n_instances=50,
        steps=2,
        latent_dim=2,
        random_seed=3823,
    )

    assert artifact["honest_verdict"].startswith("complete: trm_not_ebt_curlfree_falsified")
    assert mod.field_value(artifact["n_states_sampled"]) >= 50
    assert mod.field_value(artifact["jacobian_antisymmetry_fraction"]) > 0.95
    assert mod.field_value(artifact["scalar_potential_fit_residual"]) > 0.95
    assert mod.field_value(artifact["positive_control_fit_residual"]) < 1e-6
    assert mod.field_value(artifact["preconditions_checked"])["trm_checkpoint_loadable"] is True
    assert "linear_update_checkpoint" in mod.field_value(artifact["inference_substrate"])


def test_inconclusive_and_secretly_energy_descent_verdicts() -> None:
    """REQ-3823: verdict gate covers the positive-control failure and low-residual branches."""
    assert mod.classify_verdict(0.5, 0.8, positive_control_residual=0.2) == (
        "complete: INCONCLUSIVE_curlfree_positive_control_failed"
    )
    assert mod.classify_verdict(0.0, 1e-4, positive_control_residual=1e-9) == (
        "complete: trm_is_secretly_energy_descent_surprising_residual0.000100"
    )
    assert mod.classify_verdict(0.0, 0.10, positive_control_residual=1e-9) == (
        "complete: trm_is_secretly_energy_descent_surprising_residual0.100000"
    )


def test_write_artifact_persists_json(tmp_path: Path) -> None:
    """REQ-3823: writer persists the terminal artifact schema."""
    artifact = mod.blocked_artifact(
        preconditions={
            "torch_available": True,
            "numpy_available": True,
            "source_artifact_available": True,
            "trm_checkpoint_source": "missing.pt",
            "trm_checkpoint_loadable": False,
            "block_reason": mod.BLOCKED_VERDICT,
        },
        duration_s=0.01,
        random_seed=3823,
    )
    output_path = tmp_path / "experiment_3823.json"

    mod.write_artifact(artifact, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["schema"] == mod.SCHEMA
    assert loaded["honest_verdict"] == mod.BLOCKED_VERDICT


def test_load_source_artifact_rejects_malformed_json(tmp_path: Path) -> None:
    """REQ-3823: malformed checkpoint evidence stays fail-closed."""
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")

    preconditions = mod.run_preconditions_check(malformed)

    assert preconditions["source_artifact_available"] is False
    assert preconditions["trm_checkpoint_loadable"] is False
    assert preconditions["block_reason"] == mod.BLOCKED_VERDICT


def test_field_value_rejects_unwrapped_metric() -> None:
    """REQ-3823: principle-bearing fields are enforced by helpers."""
    with pytest.raises(TypeError):
        mod.field_value(1.0)


def test_linear_checkpoint_validation_and_call() -> None:
    """REQ-3823: synthetic checkpoint wrapper validates shapes and returns h_next."""
    with pytest.raises(ValueError, match="square"):
        mod.LinearUpdateCheckpoint([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="bias"):
        mod.LinearUpdateCheckpoint([[1.0, 0.0], [0.0, 1.0]], bias=[0.0])

    model = mod.LinearUpdateCheckpoint([[0.0, 1.0], [-1.0, 0.0]], bias=[0.5, 0.0])
    h = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    assert torch.allclose(model(h), h + model.forward_delta(h))


def test_import_json_and_source_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-3823: precondition helpers fail closed on malformed source inputs."""
    assert mod._import_available("definitely_missing_module_for_3823") is False

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json(non_object)

    assert mod.resolve_checkpoint_path(None) is None
    assert mod.resolve_checkpoint_path("https://example.test/model.pt") is None

    checkpoint = tmp_path / "relative.pt"
    torch.save({"update_matrix": [[1.0]]}, checkpoint)
    assert mod.resolve_checkpoint_path("relative.pt", base_dir=tmp_path) == checkpoint.resolve()

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"trm_checkpoint_source": str(checkpoint)}), encoding="utf-8")
    monkeypatch.setattr(mod, "_import_available", lambda name: False if name == "numpy" else True)
    preconditions = mod.run_preconditions_check(source)
    assert preconditions["checkpoint_load_error"] == "torch_or_numpy_unavailable"


def test_torch_load_fallback_and_loadable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-3823: checkpoint loadability records torch-load failures."""
    checkpoint = tmp_path / "x.pt"
    checkpoint.write_text("not a torch checkpoint", encoding="utf-8")
    loadable, error = mod.checkpoint_loadable(checkpoint)
    assert loadable is False
    assert "checkpoint" in str(error).lower() or "invalid" in str(error).lower()

    calls = {"count": 0}

    def fake_load(path, map_location=None, weights_only=None):
        calls["count"] += 1
        if weights_only is not None:
            raise TypeError("old torch")
        return {"ok": True}

    monkeypatch.setattr(torch, "load", fake_load)
    assert mod._torch_load(checkpoint) == {"ok": True}
    assert calls["count"] == 2


def test_load_update_model_module_nested_and_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-3823: supported torch module payloads load; unsupported payloads reject."""

    class TinyNextState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latent_dim = 2

        def forward(self, h):
            return h + 1.0

    direct = TinyNextState()
    monkeypatch.setattr(mod, "_torch_load", lambda path: direct)
    loaded_direct = mod.load_update_model(tmp_path / "direct.pt")
    assert loaded_direct.substrate_label == "torch_module_checkpoint"

    nested = TinyNextState()
    monkeypatch.setattr(mod, "_torch_load", lambda path: {"model": nested})
    loaded_nested = mod.load_update_model(tmp_path / "nested.pt")
    assert loaded_nested.substrate_label == "torch_module_checkpoint"

    module_without_dim = torch.nn.Identity()
    monkeypatch.setattr(mod, "_torch_load", lambda path: module_without_dim)
    with pytest.raises(ValueError, match="latent_dim"):
        mod.load_update_model(tmp_path / "bad_direct.pt")

    nested_without_dim = torch.nn.Identity()
    monkeypatch.setattr(mod, "_torch_load", lambda path: {"model": nested_without_dim})
    with pytest.raises(ValueError, match="nested"):
        mod.load_update_model(tmp_path / "bad_nested.pt")

    monkeypatch.setattr(mod, "_torch_load", lambda path: {"not": "a model"})
    with pytest.raises(ValueError, match="unsupported"):
        mod.load_update_model(tmp_path / "unsupported.pt")


def test_make_delta_fn_handles_next_state_shapes() -> None:
    """REQ-3823: generic model adapters handle tuple/dict outputs and reject bad shapes."""

    class TupleModel:
        def __call__(self, h):
            return (h + 2.0,)

    class DictModel:
        def __call__(self, h):
            return {"next_state": h + 3.0}

    class BadModel:
        def __call__(self, h):
            return torch.zeros(1, dtype=h.dtype)

    h = torch.zeros(2, dtype=torch.float64)
    assert torch.allclose(mod.make_delta_fn(TupleModel())(h), torch.full((2,), 2.0, dtype=torch.float64))
    assert torch.allclose(mod.make_delta_fn(DictModel())(h), torch.full((2,), 3.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="next latent"):
        mod.make_delta_fn(BadModel())(h)


def test_sample_and_jacobian_edge_cases() -> None:
    """REQ-3823: numerical helpers handle bounded edge cases."""
    model = mod.LinearUpdateCheckpoint([[0.0, 0.0], [0.0, 0.0]])
    states, deltas = mod.sample_update_pairs(
        model,
        n_instances=2,
        steps=1,
        latent_dim=2,
        random_seed=1,
    )
    assert states.shape[0] == 50
    assert torch.allclose(deltas, torch.zeros_like(deltas))

    with pytest.raises(ValueError, match="steps"):
        mod.sample_update_pairs(model, n_instances=50, steps=0, latent_dim=2, random_seed=1)

    assert mod.jacobian_antisymmetry_fraction(lambda h: h, torch.empty(0, 2, dtype=torch.float64)) == 0.0
    assert mod.jacobian_antisymmetry_fraction(lambda h: torch.zeros_like(h), states[:1]) == 0.0


def test_delta_single_fallback_and_shape_error() -> None:
    """REQ-3823: single-state Jacobian adapter falls back to batched functions."""

    def batched_only(h):
        if h.ndim == 1:
            raise RuntimeError("batch required")
        return h + 1.0

    assert torch.allclose(
        mod._call_delta_single(batched_only, torch.zeros(2, dtype=torch.float64)),
        torch.ones(2, dtype=torch.float64),
    )

    def wrong_shape(h):
        if h.ndim == 1:
            raise RuntimeError("batch required")
        return torch.zeros(1, 1, dtype=h.dtype)

    with pytest.raises(ValueError, match="delta_fn"):
        mod._call_delta_single(wrong_shape, torch.zeros(2, dtype=torch.float64))


def test_scalar_potential_edge_cases() -> None:
    """REQ-3823: scalar-potential residual handles invalid shapes and zero fields."""
    with pytest.raises(ValueError, match="rank-2"):
        mod.scalar_potential_fit_residual(torch.zeros(2), torch.zeros(2))

    states = torch.zeros(4, 2, dtype=torch.float64)
    deltas = torch.zeros(4, 2, dtype=torch.float64)
    assert mod.scalar_potential_fit_residual(states, deltas) == 0.0

    one_nonzero = torch.zeros(4, 2, dtype=torch.float64)
    one_nonzero[0, 0] = 1.0
    assert mod.scalar_potential_fit_residual(states, one_nonzero) > 0.0


def test_build_artifact_blocks_when_second_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-3823: a stale loadable precondition still blocks if model loading fails."""
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_text("x", encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"trm_checkpoint_source": str(checkpoint)}), encoding="utf-8")
    monkeypatch.setattr(
        mod,
        "run_preconditions_check",
        lambda path: {
            "torch_available": True,
            "numpy_available": True,
            "source_artifact_available": True,
            "trm_checkpoint_source": str(checkpoint),
            "trm_checkpoint_path": str(checkpoint),
            "trm_checkpoint_loadable": True,
            "checkpoint_load_error": None,
            "block_reason": None,
        },
    )
    monkeypatch.setattr(mod, "load_update_model", lambda path: (_ for _ in ()).throw(ValueError("bad")))

    artifact = mod.build_artifact(source_artifact_path=source)

    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert "ValueError" in mod.field_value(artifact["preconditions_checked"])["checkpoint_load_error"]


def test_main_uses_default_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-3823: CLI entrypoint builds and writes one artifact."""
    calls = {}

    def fake_build_artifact():
        calls["built"] = True
        return {"schema": mod.SCHEMA, "honest_verdict": mod.BLOCKED_VERDICT}

    def fake_write_artifact(artifact, output_path):
        calls["artifact"] = artifact
        calls["output_path"] = output_path

    monkeypatch.setattr(mod, "build_artifact", fake_build_artifact)
    monkeypatch.setattr(mod, "write_artifact", fake_write_artifact)

    mod.main()

    assert calls["built"] is True
    assert calls["artifact"]["schema"] == mod.SCHEMA
    assert calls["output_path"] == mod.OUTPUT_PATH
