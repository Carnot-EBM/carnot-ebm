"""Tests for Exp 4260 DiffusionGemma GGUF energy-guided preflight.

REQ-VERIFY-4260 / SCENARIO-VERIFY-4260: the runner must check the GGUF
cache, the llama.cpp vocab-only loader, and TRM stand-down before any smoke;
if those pass, the tiny denoising hook must actually reweight token selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4260_diffusiongemma_energy_guided_preflight as exp


class FakeTokenizer:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def tokenize(self, data: bytes) -> list[int]:
        text = data.decode("utf-8")
        if text not in self._ids:
            self._ids[text] = len(self._ids) + 100
        return [self._ids[text]]

    def detokenize(self, token_ids: list[int]) -> bytes:
        reverse = {value: key for key, value in self._ids.items()}
        return "".join(reverse.get(token_id, "?") for token_id in token_ids).encode("utf-8")


def _cache_root_with_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / exp.CACHE_REPO_DIRNAME
    repo_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    return tmp_path


def _fake_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
    path.write_bytes(b"fake gguf")
    return path


def _ok_loader(_path: str, _probe: str) -> exp.VocabLoadResult:
    return exp.VocabLoadResult(
        ok=True,
        backend="llama_cpp",
        mode="vocab_only",
        elapsed_s=0.01,
        token_count=3,
        token_ids=(1, 2, 3),
        detail="embedded GGUF tokenizer OK",
        tokenizer=FakeTokenizer(),
    )


def test_req_verify_4260_spec_declares_preflight_contract() -> None:
    """REQ-VERIFY-4260: OpenSpec declares the GGUF preflight fields."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4260",
        "SCENARIO-VERIFY-4260",
        "results/experiment_4260_diffusiongemma_energy_guided_preflight.py",
        "blocked_diffusiongemma_gguf_loader_failed",
        "preflight_go",
        "guidance_changes_selection",
        "full_run_cost_estimate_s",
        "verifier_is_oracle=false",
    ):
        assert marker in spec


def test_req_verify_4260_guidance_hook_changes_token_selection() -> None:
    """REQ-VERIFY-4260: logit -= lambda * energy can flip the chosen token."""

    hook = exp.VerifierGuidanceHook(guidance_lambda=0.7)
    candidates = (
        exp.GuidanceCandidate(token_id=1, token_text="wrong", base_logit=3.0, verifier_energy=1.0),
        exp.GuidanceCandidate(token_id=2, token_text="right", base_logit=2.6, verifier_energy=0.0),
    )

    selection = hook.select(candidates)

    assert selection.unguided.token_id == 1
    assert selection.guided.token_id == 2
    assert selection.changed is True
    assert selection.reweighted_token_count == 2
    assert selection.guided_score_by_token[1] == pytest.approx(2.3)
    assert selection.guided_score_by_token[2] == pytest.approx(2.6)


def test_scenario_4260_missing_cache_blocks_before_loader(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4260: missing cache writes blocked without loader use."""

    calls: list[str] = []

    def fail_loader(_path: str, _probe: str) -> exp.VocabLoadResult:
        calls.append("loader")
        raise AssertionError("loader should not run when cache is missing")

    preconditions = exp.check_preconditions(
        cache_root=tmp_path,
        resolve_gguf_fn=lambda **_: None,
        vocab_loader_fn=fail_loader,
        process_rows_fn=lambda: [],
    )

    assert calls == []
    assert preconditions["verdict"] == "blocked_diffusiongemma_not_cached"
    assert preconditions["all_passed"] is False
    assert preconditions["ordered_checks"][0]["resource"] == "diffusiongemma_cache"
    assert preconditions["ordered_checks"][1]["skipped"] is True
    assert preconditions["ordered_checks"][2]["skipped"] is True


def test_scenario_4260_active_trm_blocks_before_loader(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4260: active TRM training blocks and avoids the loader."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = _fake_gguf(tmp_path)
    calls: list[str] = []

    def fail_loader(_path: str, _probe: str) -> exp.VocabLoadResult:
        calls.append("loader")
        raise AssertionError("loader should not run when TRM is active")

    preconditions = exp.check_preconditions(
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=fail_loader,
        process_rows_fn=lambda: [{"pid": 123, "command": "python train_trm.py --resume"}],
    )

    assert calls == []
    assert preconditions["verdict"] == "blocked_trm_training_active"
    trm_check = preconditions["ordered_checks"][1]
    assert trm_check["resource"] == "trm_training_stand_down"
    assert trm_check["active_training_processes"]


def test_scenario_4260_loader_failure_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4260: loader failure is a terminal no-fabrication artifact."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = _fake_gguf(tmp_path)
    artifact_path = tmp_path / "experiment_4260.json"

    artifact = exp.run(
        artifact_path=artifact_path,
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: exp.VocabLoadResult(
            ok=False,
            backend="llama_cpp",
            mode="vocab_only",
            elapsed_s=0.02,
            token_count=0,
            token_ids=(),
            detail="Failed to load model from file",
        ),
        process_rows_fn=lambda: [],
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_diffusiongemma_gguf_loader_failed"
    assert artifact["preflight_go"] is False
    assert artifact["guidance_changes_selection"] is False
    assert artifact["full_run_cost_estimate_s"] == 0.0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["diffusiongemma"]["gguf_path"] == str(gguf_path)
    assert artifact["smoke_measurements"]["status"] == "blocked_diffusiongemma_gguf_loader_failed"


def test_scenario_4260_successful_smoke_reports_go_and_cost(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4260: injected loader path proves guidance reweights."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = _fake_gguf(tmp_path)
    artifact_path = tmp_path / "go.json"

    artifact = exp.run(
        artifact_path=artifact_path,
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=_ok_loader,
        process_rows_fn=lambda: [],
        full_benchmark_examples=10,
        full_benchmark_steps=8,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: diffusiongemma_energy_guided_preflight_go"
    assert artifact["preflight_go"] is True
    assert artifact["guidance_changes_selection"] is True
    assert artifact["full_run_cost_estimate_s"] > 0.0
    assert artifact["guidance_selection_change_count"] > 0
    assert artifact["guidance_reweighted_token_count"] > 0
    assert artifact["smoke_measurements"]["examples"] == len(exp.SMOKE_INPUTS)
    assert artifact["model_specs"]["denoising"]["smoke_steps"] == exp.DEFAULT_GUIDANCE_CONFIG.steps
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == exp.SPEC_REFS


def test_req_verify_4260_checksum_and_validation_are_stable(tmp_path: Path) -> None:
    """REQ-VERIFY-4260: checksum and required bare-field schema are stable."""

    checksum = exp.reproducibility_checksum(exp.SMOKE_INPUTS, exp.DEFAULT_GUIDANCE_CONFIG)
    assert checksum == exp.reproducibility_checksum(exp.SMOKE_INPUTS, exp.DEFAULT_GUIDANCE_CONFIG)
    assert checksum != exp.reproducibility_checksum(
        exp.SMOKE_INPUTS,
        exp.GuidanceConfig(steps=3, guidance_lambda=exp.DEFAULT_GUIDANCE_CONFIG.guidance_lambda),
    )

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = _fake_gguf(tmp_path)
    artifact = exp.run(
        artifact_path=tmp_path / "valid.json",
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=_ok_loader,
        process_rows_fn=lambda: [],
    )

    corruptions = [
        ("missing required fields", lambda a: a.pop("honest_verdict")),
        ("preflight_go", lambda a: a.update({"preflight_go": "true"})),
        ("guidance_changes_selection", lambda a: a.update({"guidance_changes_selection": 1})),
        ("full_run_cost_estimate_s", lambda a: a.update({"full_run_cost_estimate_s": "1.0"})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("model_specs", lambda a: a.update({"model_specs": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("infeasible artifact", lambda a: a.update({"preflight_go": True, "guidance_changes_selection": False})),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)
