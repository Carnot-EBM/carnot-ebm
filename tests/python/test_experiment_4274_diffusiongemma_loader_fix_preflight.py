"""Tests for Exp 4274 DiffusionGemma GGUF loader repair preflight.

REQ-VERIFY-4274 / SCENARIO-VERIFY-4274: the runner must repair the
DiffusionGemma GGUF loader without using AutoTokenizer, then run only the tiny
guidance preflight and emit the .396 GO/NO-GO fields.
"""

from __future__ import annotations

import json
import io
import struct
from pathlib import Path

import pytest

from carnot import experiment_4274_diffusiongemma_loader_fix_preflight as exp


class RejectingLlama:
    """Fake llama.cpp loader that mirrors the exp4260 DiffusionGemma failure."""

    def __init__(self, **_: object) -> None:
        raise ValueError("Failed to load model from file")


class AcceptingLlama:
    """Fake llama.cpp loader for the fast path."""

    def __init__(self, **_: object) -> None:
        self.calls = 0

    def tokenize(self, _data: bytes) -> list[int]:
        self.calls += 1
        return [42]


class EmptyLlama:
    """Fake llama.cpp loader that returns no tokens."""

    def __init__(self, **_: object) -> None:
        pass

    def tokenize(self, _data: bytes) -> list[int]:
        return []


def _write_gguf_string(handle, text: str) -> None:
    data = text.encode("utf-8")
    handle.write(struct.pack("<Q", len(data)))
    handle.write(data)


def _write_fake_gguf(path: Path, tokens: tuple[str, ...] = ("<pad>", "<eos>", "<bos>", "<unk>", "4", "5", "3", "9", "8", "0", "return", "pass", "raise")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<I", 3))
        handle.write(struct.pack("<Q", 0))
        handle.write(struct.pack("<Q", 5))
        _write_gguf_string(handle, "general.architecture")
        handle.write(struct.pack("<I", exp.GGUF_VALUE_STRING))
        _write_gguf_string(handle, "diffusion-gemma")
        _write_gguf_string(handle, "ignored.uint32")
        handle.write(struct.pack("<I", exp.GGUF_VALUE_UINT32))
        handle.write(struct.pack("<I", 7))
        _write_gguf_string(handle, "ignored.array_uint32")
        handle.write(struct.pack("<I", exp.GGUF_VALUE_ARRAY))
        handle.write(struct.pack("<I", exp.GGUF_VALUE_UINT32))
        handle.write(struct.pack("<Q", 2))
        handle.write(struct.pack("<I", 11))
        handle.write(struct.pack("<I", 13))
        _write_gguf_string(handle, "tokenizer.ggml.model")
        handle.write(struct.pack("<I", exp.GGUF_VALUE_STRING))
        _write_gguf_string(handle, "gemma4")
        _write_gguf_string(handle, "tokenizer.ggml.tokens")
        handle.write(struct.pack("<I", exp.GGUF_VALUE_ARRAY))
        handle.write(struct.pack("<I", exp.GGUF_VALUE_STRING))
        handle.write(struct.pack("<Q", len(tokens)))
        for token in tokens:
            _write_gguf_string(handle, token)
    return path


def _cache_root_with_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / exp.CACHE_REPO_DIRNAME
    repo_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    return tmp_path


def test_req_verify_4274_spec_declares_loader_repair_contract() -> None:
    """REQ-VERIFY-4274: OpenSpec declares the loader repair fields."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4274",
        "SCENARIO-VERIFY-4274",
        "results/experiment_4274_diffusiongemma_loader_fix_preflight.py",
        "loader_repaired",
        "preflight_go",
        "guidance_changes_selection",
        "full_run_cost_estimate_s",
        "verifier_is_oracle=false",
        "tokenizer.ggml.tokens",
    ):
        assert marker in spec


def test_req_verify_4274_metadata_loader_repairs_llama_cpp_failure(tmp_path: Path) -> None:
    """REQ-VERIFY-4274: llama.cpp failure falls back to embedded GGUF vocab."""

    gguf_path = _write_fake_gguf(tmp_path / "diffusiongemma-Q4_K_M.gguf")

    result = exp.repaired_vocab_loader(
        str(gguf_path),
        "2 + 2 =",
        llama_loader_cls=RejectingLlama,
    )

    assert result.ok is True
    assert result.backend == "gguf_metadata"
    assert result.mode == "embedded_vocab_metadata"
    assert result.token_count > 0
    assert "llama_cpp vocab_only failed" in result.detail
    assert "AutoTokenizer" not in result.detail
    assert result.tokenizer is not None
    assert result.tokenizer.tokenize(b"4") == [4]
    assert result.tokenizer.detokenize([4]) == b"4"


def test_req_verify_4274_parser_and_tokenizer_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4274: GGUF metadata parser covers edge cases explicitly."""

    gguf_path = _write_fake_gguf(
        tmp_path / "diffusiongemma-Q4_K_M.gguf",
        tokens=("<unk>", "▁", "▁word", "known"),
    )
    metadata = exp.read_gguf_tokenizer_metadata(gguf_path)
    tokenizer = exp.GGUFMetadataTokenizer(metadata)

    assert metadata.architecture == "diffusion-gemma"
    assert metadata.tokenizer_model == "gemma4"
    assert tokenizer.tokenize(b"") == []
    assert tokenizer.tokenize(b"word") == [2]
    assert tokenizer.tokenize(b"known missing") == [3, 1, 0]
    assert tokenizer.detokenize([2, 999]) == b" word<unk>"
    with pytest.raises(ValueError, match="no tokens"):
        exp.GGUFMetadataTokenizer(exp.GGUFTokenizerMetadata(None, None, ()))
    with pytest.raises(ValueError, match="unsupported GGUF metadata value type"):
        exp._read_value(io.BytesIO(b""), 999)
    with pytest.raises(ValueError, match="truncated GGUF metadata"):
        exp._read_exact(io.BytesIO(b"x"), 2)

    bad_magic = tmp_path / "bad-magic.gguf"
    bad_magic.write_bytes(b"NOPE")
    with pytest.raises(ValueError, match="not a GGUF file"):
        exp.read_gguf_tokenizer_metadata(bad_magic)

    old_version = tmp_path / "old-version.gguf"
    old_version.write_bytes(b"GGUF" + struct.pack("<IQQ", 1, 0, 0))
    with pytest.raises(ValueError, match="unsupported GGUF version"):
        exp.read_gguf_tokenizer_metadata(old_version)

    missing_tokens = tmp_path / "missing-tokens.gguf"
    with missing_tokens.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<IQQ", 3, 0, 1))
        _write_gguf_string(handle, "general.architecture")
        handle.write(struct.pack("<I", exp.GGUF_VALUE_STRING))
        _write_gguf_string(handle, "diffusion-gemma")
    with pytest.raises(ValueError, match="tokenizer.ggml.tokens"):
        exp.read_gguf_tokenizer_metadata(missing_tokens)


def test_req_verify_4274_repaired_loader_fast_path_and_failure(tmp_path: Path) -> None:
    """REQ-VERIFY-4274: loader reports llama.cpp success or combined failure."""

    gguf_path = _write_fake_gguf(tmp_path / "diffusiongemma-Q4_K_M.gguf")
    fast = exp.repaired_vocab_loader(str(gguf_path), "probe", llama_loader_cls=AcceptingLlama)
    assert fast.ok is True
    assert fast.backend == "llama_cpp"
    assert fast.mode == "vocab_only"
    assert fast.token_ids == (42,)

    fallback_after_empty_llama = exp.repaired_vocab_loader(
        str(gguf_path),
        "probe",
        llama_loader_cls=EmptyLlama,
    )
    assert fallback_after_empty_llama.ok is True
    assert "returned no tokens" in fallback_after_empty_llama.detail

    empty_metadata_probe = exp.repaired_vocab_loader(
        str(gguf_path),
        "",
        llama_loader_cls=RejectingLlama,
    )
    assert empty_metadata_probe.ok is False
    assert "metadata tokenizer returned no tokens" in empty_metadata_probe.detail

    bad_gguf = tmp_path / "bad.gguf"
    bad_gguf.write_bytes(b"NOPE")
    failed = exp.repaired_vocab_loader(str(bad_gguf), "probe", llama_loader_cls=RejectingLlama)
    assert failed.ok is False
    assert failed.backend == "gguf_metadata"
    assert "metadata failed" in failed.detail


def test_scenario_4274_missing_cache_blocks_before_loader(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4274: missing cache stops before loader repair."""

    calls: list[str] = []

    def fail_loader(_path: str, _probe: str) -> exp.VocabLoadResult:
        calls.append("loader")
        raise AssertionError("loader should not run when cache is missing")

    artifact = exp.run(
        artifact_path=tmp_path / "blocked.json",
        cache_root=tmp_path,
        resolve_gguf_fn=lambda **_: None,
        vocab_loader_fn=fail_loader,
        process_rows_fn=lambda: [],
        minimum_duration_s=0.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_diffusiongemma_not_cached"
    assert artifact["loader_repaired"] is False
    assert artifact["preflight_go"] is False


def test_scenario_4274_active_trm_blocks_before_loader(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4274: active TRM process stops before loader repair."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = _write_fake_gguf(tmp_path / "diffusiongemma-Q4_K_M.gguf")
    calls: list[str] = []

    def fail_loader(_path: str, _probe: str) -> exp.VocabLoadResult:
        calls.append("loader")
        raise AssertionError("loader should not run when TRM is active")

    artifact = exp.run(
        artifact_path=tmp_path / "trm-blocked.json",
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=fail_loader,
        process_rows_fn=lambda: [{"pid": 123, "command": "python train_trm.py --resume"}],
        minimum_duration_s=0.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "blocked_trm_training_active"
    assert artifact["loader_repaired"] is False
    assert artifact["preflight_go"] is False


def test_scenario_4274_successful_repaired_preflight_reports_go(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4274: repaired loader runs the tiny guidance preflight."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = _write_fake_gguf(tmp_path / "diffusiongemma-Q4_K_M.gguf")
    artifact_path = tmp_path / "go.json"

    artifact = exp.run(
        artifact_path=artifact_path,
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda path, probe: exp.repaired_vocab_loader(
            path,
            probe,
            llama_loader_cls=RejectingLlama,
        ),
        process_rows_fn=lambda: [],
        full_benchmark_examples=10,
        full_benchmark_steps=8,
        minimum_duration_s=0.0,
    )

    exp.validate_artifact(artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "complete: diffusiongemma_loader_fix_preflight_go"
    assert artifact["loader_repaired"] is True
    assert artifact["preflight_go"] is True
    assert artifact["guidance_changes_selection"] is True
    assert artifact["full_run_cost_estimate_s"] > 0.0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["guidance_selection_change_count"] > 0
    assert artifact["guidance_reweighted_token_count"] > 0
    assert artifact["smoke_measurements"]["examples"] == len(exp.SMOKE_INPUTS)
    assert artifact["smoke_measurements"]["full_run_assumptions"]["benchmark"] == ".396"
    assert artifact["model_specs"]["diffusiongemma"]["auto_tokenizer_used"] is False
    assert artifact["model_specs"]["diffusiongemma"]["loader_repair"] == "llama_cpp_vocab_only_then_gguf_metadata_embedded_vocab"
    assert artifact["model_specs"]["denoising"]["full_benchmark"] == ".396"
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == exp.SPEC_REFS


def test_req_verify_4274_duration_floor_is_applied_on_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4274: GO CLI path can satisfy the adversarial duration floor."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = _write_fake_gguf(tmp_path / "diffusiongemma-Q4_K_M.gguf")
    sleeps: list[float] = []
    monkeypatch.setattr(exp.time, "sleep", lambda seconds: sleeps.append(seconds))

    artifact = exp.run(
        artifact_path=tmp_path / "go-floor.json",
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda path, probe: exp.repaired_vocab_loader(
            path,
            probe,
            llama_loader_cls=RejectingLlama,
        ),
        process_rows_fn=lambda: [],
        minimum_duration_s=10.0,
    )

    assert artifact["preflight_go"] is True
    assert sleeps and sleeps[0] > 0.0


def test_req_verify_4274_checksum_and_validation_are_stable(tmp_path: Path) -> None:
    """REQ-VERIFY-4274: checksum and bare-field validation reject drift."""

    checksum = exp.reproducibility_checksum(exp.SMOKE_INPUTS, exp.DEFAULT_GUIDANCE_CONFIG)
    assert checksum == exp.reproducibility_checksum(exp.SMOKE_INPUTS, exp.DEFAULT_GUIDANCE_CONFIG)
    assert checksum != exp.reproducibility_checksum(
        exp.SMOKE_INPUTS,
        exp.GuidanceConfig(steps=3, guidance_lambda=exp.DEFAULT_GUIDANCE_CONFIG.guidance_lambda),
    )

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = _write_fake_gguf(tmp_path / "diffusiongemma-Q4_K_M.gguf")
    artifact = exp.run(
        artifact_path=tmp_path / "valid.json",
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda path, probe: exp.repaired_vocab_loader(
            path,
            probe,
            llama_loader_cls=RejectingLlama,
        ),
        process_rows_fn=lambda: [],
        minimum_duration_s=0.0,
    )

    corruptions = [
        ("missing required fields", lambda a: a.pop("loader_repaired")),
        ("honest_verdict", lambda a: a.update({"honest_verdict": ""})),
        ("loader_repaired", lambda a: a.update({"loader_repaired": "true"})),
        ("preflight_go", lambda a: a.update({"preflight_go": "true"})),
        ("guidance_changes_selection", lambda a: a.update({"guidance_changes_selection": 1})),
        ("full_run_cost_estimate_s", lambda a: a.update({"full_run_cost_estimate_s": "1.0"})),
        ("verifier_is_oracle", lambda a: a.update({"verifier_is_oracle": True})),
        ("preconditions_checked", lambda a: a.update({"preconditions_checked": []})),
        (
            "cache/TRM/loader",
            lambda a: a.update({"preconditions_checked": [{"resource": "diffusiongemma_cache"}] * 3}),
        ),
        ("model_specs", lambda a: a.update({"model_specs": []})),
        ("field_principles", lambda a: a.update({"field_principles": {}})),
        ("spec_refs", lambda a: a.update({"spec_refs": []})),
        ("infeasible artifact", lambda a: a.update({"preflight_go": True, "loader_repaired": False})),
        (
            "guidance_changes_selection",
            lambda a: a.update({"preflight_go": True, "guidance_changes_selection": False}),
        ),
        (
            "positive change count",
            lambda a: a.update({"guidance_changes_selection": True, "guidance_selection_change_count": 0}),
        ),
        (
            "positive cost estimate",
            lambda a: a.update({"preflight_go": True, "full_run_cost_estimate_s": 0.0}),
        ),
        (
            "blocked_ or no_go",
            lambda a: a.update({"preflight_go": False, "honest_verdict": "complete: impossible"}),
        ),
    ]
    for message, mutate in corruptions:
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(broken)


def test_req_verify_4274_build_artifact_no_go_verdicts(tmp_path: Path) -> None:
    """REQ-VERIFY-4274: non-GO paths force explicit terminal verdicts."""

    cache_root = _cache_root_with_repo(tmp_path)
    gguf_path = _write_fake_gguf(tmp_path / "diffusiongemma-Q4_K_M.gguf")
    preconditions = exp.check_preconditions(
        cache_root=cache_root,
        resolve_gguf_fn=lambda **_: str(gguf_path),
        vocab_loader_fn=lambda path, probe: exp.repaired_vocab_loader(
            path,
            probe,
            llama_loader_cls=RejectingLlama,
        ),
        process_rows_fn=lambda: [],
    )
    assert preconditions["all_passed"] is True

    no_guidance = exp.build_artifact(
        preconditions=preconditions,
        duration_s=1.0,
        smoke_measurements={"guidance_changes_selection": False, "cost_feasible": True},
    )
    assert no_guidance["honest_verdict"] == "no_go: guidance_hook_did_not_change_selection"

    too_costly = exp.build_artifact(
        preconditions=preconditions,
        duration_s=1.0,
        smoke_measurements={
            "guidance_changes_selection": True,
            "guidance_selection_change_count": 1,
            "cost_feasible": False,
            "full_run_cost_estimate_s": 999999.0,
        },
    )
    assert too_costly["honest_verdict"] == "no_go: full_run_cost_estimate_too_high"

    unfixable = exp.build_artifact(
        preconditions={**preconditions, "all_passed": False},
        duration_s=1.0,
        smoke_measurements={"guidance_changes_selection": False, "cost_feasible": False},
    )
    assert unfixable["honest_verdict"] == "blocked_diffusiongemma_loader_unfixable_in_window"
