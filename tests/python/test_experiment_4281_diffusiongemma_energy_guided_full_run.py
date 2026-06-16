"""Tests for Exp 4281 DiffusionGemma energy-guided full-run gate.

REQ-VERIFY-4281 / SCENARIO-VERIFY-4281: the runner must check the
DiffusionGemma PR binary and GGUF resources before inference, exercise the
energy-guidance hook, and honestly block the learned-verifier moat arm when the
verifier cannot score partial masked denoising states.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from carnot import experiment_4281_diffusiongemma_energy_guided_full_run as exp


class TinyTokenizer:
    """Tokenizer stub that is enough for the deterministic guidance smoke."""

    vocab = {
        "<unk>": 0,
        "4": 4,
        "5": 5,
        "3": 3,
        "9": 9,
        "8": 8,
        "0": 10,
        "return": 11,
        "pass": 12,
        "raise": 13,
    }

    def tokenize(self, data: bytes) -> list[int]:
        text = data.decode("utf-8", errors="replace")
        return [self.vocab.get(piece, 0) for piece in text.split()] or [0]

    def detokenize(self, token_ids: list[int]) -> bytes:
        inverse = {value: key for key, value in self.vocab.items()}
        return " ".join(inverse.get(int(token_id), "<unk>") for token_id in token_ids).encode(
            "utf-8"
        )


def _cache_root_with_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / exp.CACHE_REPO_DIRNAME
    repo_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    return tmp_path


def _binary(tmp_path: Path, payload: bytes = b"binary") -> Path:
    path = tmp_path / "llama-diffusion-gemma-eval"
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _loader_result() -> exp.VocabLoadResult:
    return exp.VocabLoadResult(
        ok=True,
        backend="test",
        mode="embedded_vocab_metadata",
        elapsed_s=0.001,
        token_count=1,
        token_ids=(4,),
        detail="test loader",
        tokenizer=TinyTokenizer(),
    )


def _energy_prior() -> dict[str, object]:
    return {
        "status": "extracted",
        "examples": 1,
        "score_shape": [exp.CANVAS_LEN, exp.VOCAB_SIZE],
        "score_finite_sample": True,
        "logits_file_size_bytes": exp.CANVAS_LEN * exp.VOCAB_SIZE * 4,
        "prompt_ids_count": 3,
    }


def test_req_verify_4281_spec_declares_full_run_gate_contract() -> None:
    """REQ-VERIFY-4281: OpenSpec declares the full-run gate fields."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4281",
        "SCENARIO-VERIFY-4281",
        "results/experiment_4281_diffusiongemma_energy_guided_full_run.py",
        "llama-diffusion-gemma-eval",
        "complete_diffusiongemma_learned_verifier_cannot_score_partial_states",
        "diffusiongemma_guidance_moat",
        "carnot_minus_rfg_delta",
        "guidance_moat_ci95",
        "execution_grounded_guidance_delta",
        "verifier_is_oracle=true",
        "verifier_is_oracle=false",
    ):
        assert marker in spec


def test_scenario_4281_missing_pr_binary_blocks_before_cache(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4281: missing PR binary stops before GGUF inspection."""

    calls: list[str] = []

    def fail_resolve(**_: object) -> str:
        calls.append("resolve")
        raise AssertionError("GGUF cache should not be inspected without PR binary")

    artifact = exp.run(
        artifact_path=tmp_path / "blocked.json",
        pr_binary_path=tmp_path / "missing-binary",
        cache_root=tmp_path,
        resolve_gguf_fn=fail_resolve,
        minimum_duration_s=0.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_pr_binary"
    assert artifact["diffusiongemma_guidance_moat"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "pr_binary"
    assert artifact["preconditions_checked"][1]["skipped"] is True


def test_scenario_4281_missing_cache_blocks_before_loader(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4281: missing GGUF cache stops before loader or inference."""

    calls: list[str] = []

    def fail_loader(_path: str, _probe: str) -> exp.VocabLoadResult:
        calls.append("loader")
        raise AssertionError("loader should not run when cache is missing")

    artifact = exp.run(
        artifact_path=tmp_path / "cache-blocked.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=tmp_path,
        resolve_gguf_fn=lambda **_: None,
        vocab_loader_fn=fail_loader,
        process_rows_fn=lambda: [],
        minimum_duration_s=0.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_diffusiongemma_not_cached"
    assert artifact["model_specs"]["diffusiongemma"]["model_loaded"] is False


def test_scenario_4281_trm_training_blocks_before_loader(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4281: active TRM training stops before inference."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    artifact = exp.run(
        artifact_path=tmp_path / "trm-blocked.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [{"pid": 88, "command": "torchrun train_trm.py"}],
        minimum_duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_trm_training_active"
    assert artifact["preconditions_checked"][2]["resource"] == "trm_training_stand_down"
    assert artifact["preconditions_checked"][2]["active_training_processes"]


def test_scenario_4281_partial_state_block_writes_decision_grade_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4281: partial-state verifier gap is complete and non-moat."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "diffusiongemma-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    artifact_path = tmp_path / "partial.json"

    artifact = exp.run(
        artifact_path=artifact_path,
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        energy_prior_fn=lambda **_: _energy_prior(),
        partial_state_support_fn=lambda: exp.PartialStateSupport(
            can_score=False,
            reason="learned verifier exposes complete_text_score only",
            inspected_symbols=("complete_text_score",),
        ),
        minimum_duration_s=0.0,
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == exp.PARTIAL_STATE_VERDICT
    assert artifact["diffusiongemma_guidance_moat"] is False
    assert artifact["guidance_changes_selection"] is True
    assert artifact["energy_prior_smoke"]["status"] == "extracted"
    assert artifact["headline_arm"]["status"] == "blocked_partial_state_verifier"
    assert artifact["headline_arm"]["verifier_is_oracle"] is False
    assert artifact["execution_grounded_arm"]["verifier_is_oracle"] is True
    assert (
        artifact["execution_grounded_arm"]["status"] == "not_run_after_headline_partial_state_block"
    )
    assert artifact["per_arm_verifier_is_oracle"] == {
        "headline_learned": False,
        "execution_grounded": True,
    }
    assert artifact["schema"].startswith("blocked_diffusiongemma_partial_state")
    assert artifact["model_specs"]["diffusiongemma"]["pr_binary"] == str(
        artifact["preconditions_checked"][0]["path"]
    )
    assert artifact["model_specs"]["denoising"]["conditions"] == [
        "unguided",
        "RFG",
        "EntRGi",
        "Carnot-verifier-guided",
    ]


def test_req_verify_4281_bootstrap_and_benchmark_summary() -> None:
    """REQ-VERIFY-4281: measured rows produce deltas, CI95, and moat bool."""

    rows = [
        {"task_id": f"t{i}", "unguided": i < 1, "rfg": i < 2, "entrgi": i < 3, "carnot": i < 9}
        for i in range(10)
    ]

    summary = exp.summarize_headline_rows(rows, resamples=2500, seed=4281)

    assert summary["condition_accuracy"] == {
        "unguided": 0.1,
        "rfg": 0.2,
        "entrgi": 0.3,
        "carnot": 0.9,
    }
    assert summary["carnot_minus_rfg_delta"] == pytest.approx(0.7)
    assert summary["carnot_minus_unguided_delta"] == pytest.approx(0.8)
    assert summary["guidance_moat_ci95"][0] > 0.0
    assert summary["diffusiongemma_guidance_moat"] is True
    assert summary["bootstrap_resamples"] == 2500


def test_req_verify_4281_execution_grounded_summary_is_circular() -> None:
    """REQ-VERIFY-4281: execution-grounded deltas are marked oracle-circular."""

    rows = [
        {"task_id": "s0", "unguided": False, "guided": True},
        {"task_id": "s1", "unguided": True, "guided": True},
        {"task_id": "s2", "unguided": False, "guided": False},
        {"task_id": "s3", "unguided": False, "guided": True},
    ]

    summary = exp.summarize_execution_grounded_rows(rows)

    assert summary["status"] == "measured_execution_grounded_circular"
    assert summary["verifier_is_oracle"] is True
    assert summary["execution_grounded_guidance_delta"] == pytest.approx(0.5)
    assert "NOT a moat" in summary["interpretation"]


def test_req_verify_4281_diagnostics_and_error_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-4281: diagnostics and explicit edge errors are covered."""

    no_tokenizer = exp.VocabLoadResult(
        ok=True,
        backend="test",
        mode="none",
        elapsed_s=0.0,
        token_count=0,
        token_ids=(),
        detail="no tokenizer",
        tokenizer=None,
    )
    with pytest.raises(RuntimeError, match="tokenizer"):
        exp.run_guidance_smoke(loader_result=no_tokenizer, config=exp.GuidanceConfig())

    with pytest.raises(ValueError, match="same length"):
        exp.bootstrap_delta_ci([True], [False, True], resamples=10, seed=1)
    with pytest.raises(ValueError, match="at least one task"):
        exp.bootstrap_delta_ci([], [], resamples=10, seed=1)
    with pytest.raises(ValueError, match="headline row"):
        exp.summarize_headline_rows([])
    with pytest.raises(ValueError, match="execution-grounded row"):
        exp.summarize_execution_grounded_rows([])

    sleeps: list[float] = []
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: sleeps.append(seconds))
    exp._maybe_sleep_for_live_floor(exp.time.perf_counter(), 1.0)
    assert sleeps and sleeps[0] > 0.0


def test_req_verify_4281_partial_state_diagnosis_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-4281: verifier API inspection distinguishes supported and blocked APIs."""

    fake_supported = types.SimpleNamespace(
        LearnedEnergyVerifier=type(
            "LearnedEnergyVerifier", (), {"score_partial_state": lambda self: 0.0}
        )
    )
    monkeypatch.setitem(sys.modules, "carnot.verify", fake_supported)
    supported = exp.diagnose_partial_state_support()
    assert supported.can_score is True
    assert "score_partial_state" in supported.inspected_symbols

    fake_blocked = types.SimpleNamespace(SemanticEnergyVerifier=object(), PlainObject=object())
    monkeypatch.setitem(sys.modules, "carnot.verify", fake_blocked)
    blocked = exp.diagnose_partial_state_support()
    assert blocked.can_score is False
    assert "No learned verifier" in blocked.reason
    assert "SemanticEnergyVerifier" in blocked.inspected_symbols

    import builtins

    original_import = builtins.__import__

    def fail_verify_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "carnot.verify":
            raise RuntimeError("verify boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_verify_import)
    failed = exp.diagnose_partial_state_support()
    assert failed.can_score is False
    assert "verify boom" in failed.reason


def test_scenario_4281_pr_binary_eval_failure_is_blocked(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4281: PR-binary eval failure stops before verifier claim."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    artifact = exp.run(
        artifact_path=tmp_path / "eval-blocked.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        energy_prior_fn=lambda **_: {"status": "blocked_pr_binary_eval_failed", "eval_rc": 2},
        minimum_duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_pr_binary_eval_failed"
    assert artifact["schema"] == "blocked_diffusiongemma_pr_binary_eval_v1"


def test_scenario_4281_supported_partial_api_without_runner_blocks(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4281: a partial API still needs a benchmark runner."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    artifact = exp.run(
        artifact_path=tmp_path / "runner-blocked.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        energy_prior_fn=lambda **_: _energy_prior(),
        partial_state_support_fn=lambda: exp.PartialStateSupport(
            True, "score_partial_state", ("Verifier",)
        ),
        benchmark_runner_fn=None,
        minimum_duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_diffusiongemma_benchmark_runner_unavailable"
    assert artifact["headline_arm"]["status"] == "blocked_benchmark_runner_unavailable"


def test_scenario_4281_measured_benchmark_path_writes_moat_when_ci_positive(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4281: measured rows populate headline and execution arms."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")

    def benchmark() -> dict[str, object]:
        return {
            "headline_rows": [
                {
                    "task_id": f"t{i}",
                    "unguided": i < 1,
                    "rfg": i < 2,
                    "entrgi": i < 3,
                    "carnot": i < 9,
                }
                for i in range(10)
            ],
            "execution_grounded_rows": [
                {"task_id": "s0", "unguided": False, "guided": True},
                {"task_id": "s1", "unguided": True, "guided": True},
            ],
        }

    artifact = exp.run(
        artifact_path=tmp_path / "measured.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        energy_prior_fn=lambda **_: _energy_prior(),
        partial_state_support_fn=lambda: exp.PartialStateSupport(
            True, "score_partial_state", ("Verifier",)
        ),
        benchmark_runner_fn=benchmark,
        minimum_duration_s=0.0,
    )

    assert artifact["schema"] == "diffusiongemma_guidance_full_run_v1"
    assert artifact["diffusiongemma_guidance_moat"] is True
    assert artifact["honest_verdict"] == "complete: diffusiongemma_guidance_moat_won"
    assert artifact["execution_grounded_guidance_delta"] == pytest.approx(0.5)


def test_req_verify_4281_validation_rejects_schema_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-4281: validator enforces required bare fields and arm honesty."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    artifact = exp.run(
        artifact_path=tmp_path / "valid.json",
        pr_binary_path=_binary(tmp_path),
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda _path, _probe: _loader_result(),
        process_rows_fn=lambda: [],
        energy_prior_fn=lambda **_: _energy_prior(),
        partial_state_support_fn=lambda: exp.PartialStateSupport(False, "no partial API", ()),
        minimum_duration_s=0.0,
    )

    corruptions = [
        ("missing required fields", lambda a: a.pop("diffusiongemma_guidance_moat")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        (
            "diffusiongemma_guidance_moat",
            lambda a: a.update({"diffusiongemma_guidance_moat": "false"}),
        ),
        ("carnot_minus_rfg_delta", lambda a: a.update({"carnot_minus_rfg_delta": "0.0"})),
        ("carnot_minus_unguided_delta", lambda a: a.update({"carnot_minus_unguided_delta": "0.0"})),
        (
            "execution_grounded_guidance_delta",
            lambda a: a.update({"execution_grounded_guidance_delta": None}),
        ),
        ("guidance_moat_ci95", lambda a: a.update({"guidance_moat_ci95": [0.1]})),
        (
            "per-arm verifier_is_oracle",
            lambda a: a.update({"per_arm_verifier_is_oracle": {"headline": True}}),
        ),
        ("top-level verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        (
            "moat cannot be true",
            lambda a: a.update(
                {"diffusiongemma_guidance_moat": True, "carnot_minus_rfg_delta": -0.1}
            ),
        ),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)
