"""Tests for Exp 4189 verifier-guided DiffusionGemma feasibility.

REQ-VERIFY-4189 / SCENARIO-VERIFY-4189: the runner must either produce a
feasible guided-vs-unguided smoke artifact, or block honestly when the
DiffusionGemma weights, CUDA, or per-step hook are unavailable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "results" / "experiment_4189_diffusiongemma_verifier_guided_decoding.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_4189", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_model_info() -> SimpleNamespace:
    return SimpleNamespace(
        id="google/diffusiongemma-26B-A4B-it",
        sha="abc123",
        gated=False,
        private=False,
        siblings=[
            SimpleNamespace(rfilename="README.md", size=100),
            SimpleNamespace(rfilename="config.json", size=100),
            SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=10),
            SimpleNamespace(rfilename="model-00002-of-00002.safetensors", size=10),
            SimpleNamespace(rfilename="model.safetensors.index.json", size=10),
        ],
    )


def _cuda_available() -> dict[str, object]:
    return {"available": True, "device_count": 1, "devices": ["NVIDIA GeForce RTX 3090"]}


def _hook_available() -> dict[str, object]:
    return {
        "available": True,
        "surface": "DiffusionGemmaGenerationMixin._denoising_step",
        "evidence": "logits_processor(input_ids, raw_logits, cur_step=cur_step) before sampler.accept_canvas",
    }


def _write_cached_shards(cache_dir: Path) -> None:
    snapshot = (
        cache_dir
        / "models--google--diffusiongemma-26B-A4B-it"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (snapshot / name).write_bytes(b"0123456789")


def test_req_verify_4189_spec_declares_diffusiongemma_contract() -> None:
    """REQ-VERIFY-4189: OpenSpec declares the feasibility artifact and fields."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4189",
        "SCENARIO-VERIFY-4189",
        "results/experiment_4189_diffusiongemma_verifier_guided_decoding.py",
        "results/experiment_4189_diffusiongemma_verifier_guided_decoding.json",
        "google/diffusiongemma-26B-A4B-it",
        "blocked_diffusiongemma_not_cached",
        "blocked_diffusiongemma_no_perstep_logit_hook",
        "diffusiongemma_feasible",
        "guided_vs_unguided_delta",
    ):
        assert marker in spec


def test_req_verify_4189_guidance_processor_reweights_candidate_logits() -> None:
    """REQ-VERIFY-4189: verifier energy is applied as logit -= lambda * energy."""

    exp = _load_module()

    def energy_fn(*, input_ids, candidate_token_ids, scores, cur_step):
        assert input_ids.shape == (1, 2)
        assert scores.shape == (1, 2, 4)
        assert int(cur_step) == 7
        return candidate_token_ids.float()

    processor = exp.VerifierEnergyLogitsProcessor(
        verifier_energy_fn=energy_fn,
        guidance_lambda=0.5,
        top_k=2,
    )
    scores = torch.tensor([[[1.0, 4.0, 3.0, -1.0], [0.0, 2.0, 1.0, 5.0]]])
    adjusted = processor(torch.tensor([[101, 102]]), scores, cur_step=torch.tensor(7))

    assert adjusted[0, 0, 1].item() == pytest.approx(3.5)
    assert adjusted[0, 0, 2].item() == pytest.approx(2.0)
    assert adjusted[0, 0, 0].item() == pytest.approx(1.0)
    assert adjusted[0, 1, 3].item() == pytest.approx(3.5)
    assert adjusted[0, 1, 1].item() == pytest.approx(1.5)
    assert adjusted[0, 1, 2].item() == pytest.approx(1.0)
    assert processor.call_count == 1
    assert processor.guidance_applied is True


def test_req_verify_4189_guidance_processor_defensive_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-4189: malformed guidance inputs fail loudly."""

    exp = _load_module()
    with pytest.raises(ValueError, match="guidance_lambda"):
        exp.VerifierEnergyLogitsProcessor(lambda **_: torch.zeros((1, 1, 1)), guidance_lambda=-0.1, top_k=1)
    with pytest.raises(ValueError, match="top_k"):
        exp.VerifierEnergyLogitsProcessor(lambda **_: torch.zeros((1, 1, 1)), guidance_lambda=0.1, top_k=0)

    processor = exp.VerifierEnergyLogitsProcessor(
        lambda **_: [[[0.0]]],
        guidance_lambda=1.0,
        top_k=1,
    )
    adjusted = processor(torch.tensor([[1]]), torch.ones((1, 1, 2)))
    assert adjusted.shape == (1, 1, 2)
    with pytest.raises(ValueError, match="shape"):
        processor(torch.tensor([[1]]), torch.ones((1, 2)))

    bad_shape = exp.VerifierEnergyLogitsProcessor(
        lambda **_: torch.zeros((1, 1)),
        guidance_lambda=1.0,
        top_k=1,
    )
    with pytest.raises(ValueError, match="energy shape"):
        bad_shape(torch.tensor([[1]]), torch.ones((1, 1, 2)))

    monkeypatch.setattr(exp, "torch", None)
    with pytest.raises(RuntimeError, match="torch is required"):
        processor(torch.tensor([[1]]), torch.ones((1, 1, 2)))


def test_req_verify_4189_preconditions_block_when_weights_missing(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4189: metadata-only cache does not count as model cached."""

    exp = _load_module()

    preconditions = exp.check_preconditions(
        cache_dir=tmp_path,
        model_info_fn=lambda repo_id: _fake_model_info(),
        cuda_info_fn=_cuda_available,
        hook_info_fn=_hook_available,
    )

    assert preconditions["verdict"] == "blocked_diffusiongemma_not_cached"
    cache_check = preconditions["checks"]["diffusiongemma_cache"]
    assert cache_check["repo_id"] == "google/diffusiongemma-26B-A4B-it"
    assert cache_check["full_weights_cached"] is False
    assert cache_check["expected_weight_shards"] == 2
    assert cache_check["present_weight_shards"] == 0
    assert preconditions["checks"]["cuda"]["available"] is True
    assert preconditions["checks"]["per_step_logit_hook"]["available"] is True


def test_req_verify_4189_precondition_verdict_priority_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4189: hook and CUDA failures get explicit blocked verdicts."""

    exp = _load_module()
    _write_cached_shards(tmp_path)

    no_hook = exp.check_preconditions(
        cache_dir=tmp_path,
        model_info_fn=lambda repo_id: _fake_model_info(),
        cuda_info_fn=_cuda_available,
        hook_info_fn=lambda: {"available": False, "evidence": "missing"},
    )
    assert no_hook["verdict"] == "blocked_diffusiongemma_no_perstep_logit_hook"

    no_cuda = exp.check_preconditions(
        cache_dir=tmp_path,
        model_info_fn=lambda repo_id: _fake_model_info(),
        cuda_info_fn=lambda: {"available": False, "device_count": 0, "devices": []},
        hook_info_fn=_hook_available,
    )
    assert no_cuda["verdict"] == "blocked_cuda_unavailable"

    repo_error = exp.check_preconditions(
        cache_dir=tmp_path / "missing-cache-root",
        model_info_fn=lambda repo_id: (_ for _ in ()).throw(RuntimeError("offline")),
        cuda_info_fn=_cuda_available,
        hook_info_fn=_hook_available,
    )
    assert repo_error["verdict"] == "blocked_diffusiongemma_not_cached"
    assert repo_error["checks"]["diffusiongemma_cache"]["repo_error"].startswith("RuntimeError")
    assert repo_error["checks"]["diffusiongemma_cache"]["cache_grep_diffusion_matches"] == []


def test_req_verify_4189_preconditions_pass_with_shards_cuda_and_hook(tmp_path: Path) -> None:
    """REQ-VERIFY-4189: all hard preconditions must pass before smoke is feasible."""

    exp = _load_module()
    _write_cached_shards(tmp_path)

    preconditions = exp.check_preconditions(
        cache_dir=tmp_path,
        model_info_fn=lambda repo_id: _fake_model_info(),
        cuda_info_fn=_cuda_available,
        hook_info_fn=_hook_available,
    )

    assert preconditions["verdict"] is None
    assert preconditions["all_passed"] is True
    assert preconditions["checks"]["diffusiongemma_cache"]["full_weights_cached"] is True


def test_scenario_4189_blocked_run_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4189: missing shards write a stable blocked JSON artifact."""

    exp = _load_module()
    artifact_path = tmp_path / "experiment_4189.json"

    artifact = exp.run(
        artifact_path=artifact_path,
        cache_dir=tmp_path,
        model_info_fn=lambda repo_id: _fake_model_info(),
        cuda_info_fn=_cuda_available,
        hook_info_fn=_hook_available,
    )

    exp.validate_artifact(artifact)
    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_diffusiongemma_not_cached"
    assert artifact["diffusiongemma_feasible"] is False
    assert artifact["guided_vs_unguided_delta"] == {
        "status": "blocked_diffusiongemma_not_cached",
        "n": 0,
        "guided_pass_rate": None,
        "unguided_pass_rate": None,
        "delta": None,
        "ci95": None,
    }
    assert artifact["model_specs"]["diffusiongemma"]["repo_id"] == "google/diffusiongemma-26B-A4B-it"
    assert artifact["model_specs"]["verifier_ensemble"]["name"] == "carnot_executable_code_verifier_energy"
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == ["REQ-VERIFY-4189", "SCENARIO-VERIFY-4189"]


def test_req_verify_4189_validation_rejects_schema_poison(tmp_path: Path) -> None:
    """REQ-VERIFY-4189: load-bearing artifact fields remain strongly typed."""

    exp = _load_module()
    artifact = exp.build_blocked_artifact(
        verdict="blocked_diffusiongemma_not_cached",
        preconditions=exp.check_preconditions(
            cache_dir=tmp_path,
            model_info_fn=lambda repo_id: _fake_model_info(),
            cuda_info_fn=_cuda_available,
            hook_info_fn=_hook_available,
        ),
        duration_s=0.1,
    )

    artifact["diffusiongemma_feasible"] = "false"
    with pytest.raises(ValueError, match="diffusiongemma_feasible"):
        exp.validate_artifact(artifact)

    artifact["diffusiongemma_feasible"] = False
    artifact["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(artifact)


def test_req_verify_4189_validation_defensive_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4189: schema validation blocks every load-bearing corruption."""

    exp = _load_module()
    base = exp.build_blocked_artifact(
        verdict="blocked_diffusiongemma_not_cached",
        preconditions=exp.check_preconditions(
            cache_dir=tmp_path,
            model_info_fn=lambda repo_id: _fake_model_info(),
            cuda_info_fn=_cuda_available,
            hook_info_fn=_hook_available,
        ),
        duration_s=0.1,
    )

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        (
            "cache/hook/cuda",
            lambda a: a.update(
                {
                    "preconditions_checked": [
                        {"resource": "cuda"},
                        {"resource": "cuda"},
                        {"resource": "cuda"},
                    ]
                }
            ),
        ),
        ("guided_vs_unguided_delta", lambda a: a.update({"guided_vs_unguided_delta": []})),
        ("model_specs", lambda a: a.update({"model_specs": []})),
        ("infeasible artifact", lambda a: a.update({"honest_verdict": "complete: bad"})),
        ("blocked delta status", lambda a: a.update({"guided_vs_unguided_delta": {"status": "other"}})),
    ]
    for message, mutate in corruptions:
        artifact = json.loads(json.dumps(base))
        mutate(artifact)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(artifact)

    feasible = json.loads(json.dumps(base))
    feasible["honest_verdict"] = "complete: diffusiongemma_guided_smoke_measured"
    feasible["diffusiongemma_feasible"] = True
    feasible["guided_vs_unguided_delta"] = {"status": "measured", "n": exp.SMOKE_N - 1}
    with pytest.raises(ValueError, match="measured smoke"):
        exp.validate_artifact(feasible)

    feasible["guided_vs_unguided_delta"] = {"status": "measured", "n": exp.SMOKE_N}
    with pytest.raises(ValueError, match="guidance hook fired"):
        exp.validate_artifact(feasible)

    feasible["model_specs"]["diffusiongemma"]["guidance_hook_fired"] = True
    exp.validate_artifact(feasible)


def test_req_verify_4189_checksum_bootstrap_and_smoke_helpers_are_deterministic() -> None:
    """REQ-VERIFY-4189: checksum, CI, and smoke helpers are reproducible."""

    exp = _load_module()
    task = exp.SmokeTask("id", "prompt", "fn", ("fn() == 1",))
    assert task.to_dict() == {
        "task_id": "id",
        "prompt": "prompt",
        "entry_point": "fn",
        "tests": ["fn() == 1"],
    }
    energy = exp.ExecutablePythonVerifierEnergy(tokenizer=object(), tasks=[task], neutral_energy=0.25)
    assert energy.tasks == [task]
    assert energy.neutral_energy == 0.25

    checksum = exp.reproducibility_checksum(exp.SMOKE_TASKS, exp.DEFAULT_GUIDANCE_CONFIG)
    assert checksum == exp.reproducibility_checksum(exp.SMOKE_TASKS, exp.DEFAULT_GUIDANCE_CONFIG)
    assert checksum != exp.reproducibility_checksum(
        exp.SMOKE_TASKS,
        {**exp.DEFAULT_GUIDANCE_CONFIG, "guidance_lambda": 0.25},
    )

    assert exp._percentile([], 0.5) == 0.0
    assert exp._percentile([7.0], 0.5) == 7.0
    with pytest.raises(ValueError, match="lengths"):
        exp.bootstrap_delta_ci([True], [], seed=123)
    assert exp.bootstrap_delta_ci([], [], seed=123) == [0.0, 0.0]
    ci = exp.bootstrap_delta_ci([True, False, True], [False, False, True], seed=123, resamples=200)
    assert ci == exp.bootstrap_delta_ci([True, False, True], [False, False, True], seed=123, resamples=200)
    assert len(ci) == 2
    assert ci[0] <= ci[1]
    assert exp._pass_rate([True, False, True]) == pytest.approx(2 / 3)
    assert exp._pass_rate([]) == 0.0
    assert exp._smoke_delta([True, False], [False, False], seed=123)["delta"] == 0.5
