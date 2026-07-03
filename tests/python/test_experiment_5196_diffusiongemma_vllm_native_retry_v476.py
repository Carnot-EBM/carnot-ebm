"""Tests for Exp 5196 DiffusionGemma vLLM-native retry + HF custom device_map.

Spec refs:
- REQ-VERIFY-5196 (the experiment must honestly report load success/failure and
  never claim diffusiongemma_loadable=true without a confirmed forward pass).
- SCENARIO-VERIFY-5196-VLLM-NATIVE (vLLM-native attempt outcomes classified).
- SCENARIO-VERIFY-5196-HF-DEVMAP (HF custom-device_map attempt outcomes).
- SCENARIO-VERIFY-5196-BARE-GATE-FIELD (diffusiongemma_loadable is a BARE bool).
- SCENARIO-VERIFY-5196-MEMORY-GAP (the ~10 GiB memory gap is investigated).

These tests exercise the PURE decision/analysis logic -- the heavyweight GPU
probes live in separate scripts and are not imported here.
"""

from __future__ import annotations

import json

import pytest

from carnot import experiment_5196_diffusiongemma_vllm_native_retry_v476 as exp


# --------------------------------------------------------------------------- #
# SCENARIO-VERIFY-5196-MEMORY-GAP: memory arithmetic                          #
# --------------------------------------------------------------------------- #
def test_gib_from_bytes_matches_binary_gib():
    # REQ-VERIFY-5196: report memory in binary GiB (what torch/nvidia-smi use).
    assert exp.gib_from_bytes(1024**3) == pytest.approx(1.0)
    assert exp.gib_from_bytes(exp.CHECKPOINT_BF16_BYTES) == pytest.approx(48.10, abs=0.05)


def test_fourbit_weight_gib_single_vs_double_copy():
    # A single 25.8B copy is ~12.9 GiB at 4-bit; both encoder+decoder ~doubles it.
    single = exp.fourbit_weight_gib(exp.ON_DISK_PARAMS_B)
    both = exp.fourbit_weight_gib(exp.ENCODER_PARAMS_B + exp.DECODER_PARAMS_B)
    assert 12.0 < single < 14.0
    assert both > 24.0  # exceeds one 24 GiB GPU -- the tie-break duplication story
    assert both == pytest.approx(2 * single, rel=0.02)


def test_fourbit_weight_gib_quant_state_overhead_increases_estimate():
    with_overhead = exp.fourbit_weight_gib(10.0, quant_state_bits=0.5)
    flat = exp.fourbit_weight_gib(10.0, quant_state_bits=0.0)
    assert with_overhead > flat


def test_diffusion_logit_buffer_default_is_huge_recipe_is_small():
    # SCENARIO-VERIFY-5196-MEMORY-GAP: the recipe forces max_num_seqs<=4 because
    # the max_seqs*canvas*vocab buffer explodes at the default.
    default = exp.diffusion_logit_buffer_gib(exp.VLLM_DEFAULT_MAX_NUM_SEQS)
    recipe = exp.diffusion_logit_buffer_gib(exp.RECIPE_MAX_NUM_SEQS)
    assert default > 30.0  # tens of GiB at the default -- guaranteed OOM
    assert recipe < 1.0  # sub-GiB at max_num_seqs=4
    assert default == pytest.approx(
        recipe * exp.VLLM_DEFAULT_MAX_NUM_SEQS / exp.RECIPE_MAX_NUM_SEQS, rel=1e-6
    )


def test_tied_embedding_gib_is_over_one_gib():
    # vocab 262144 x hidden 2816 bf16 embeddings stay un-quantised (>1 GiB).
    assert exp.tied_embedding_gib() > 1.0


def test_memory_gap_analysis_is_grounded_string():
    text = exp.memory_gap_analysis()
    assert isinstance(text, str) and len(text) > 200
    for token in ("tie", "max_num_seqs", "vocab", "encoder", "decoder"):
        assert token in text


# --------------------------------------------------------------------------- #
# SCENARIO-VERIFY-5196-*: loadability classification + verdict                #
# --------------------------------------------------------------------------- #
def _failed(mitigation="m", path_prefix="vllm_native"):
    return {
        "mitigation": f"{path_prefix}_{mitigation}",
        "outcome": "load_failed",
        "forward_pass_confirmed": False,
        "peak_vram_gib_per_gpu": {"gpu0": 0.0},
        "duration_s": 1.0,
    }


def test_classify_all_failed_is_both_failed():
    loadable, fwd, path = exp.classify_loadability([_failed(), _failed("m2")])
    assert (loadable, fwd, path) == (False, False, "both_failed")


def test_classify_vllm_forward_ok():
    good = {
        "mitigation": "vllm_native_fp8_tp2",
        "outcome": "forward_pass_ok",
        "forward_pass_confirmed": True,
    }
    assert exp.classify_loadability([_failed(), good]) == (True, True, "vllm_native")


def test_classify_hf_forward_ok():
    good = {
        "mitigation": "hf_custom_devmap_manual_split",
        "outcome": "forward_pass_ok",
        "forward_pass_confirmed": True,
    }
    assert exp.classify_loadability([good]) == (True, True, "hf_custom_device_map")


def test_classify_loaded_without_forward_does_not_count():
    # A load that returns but never confirms a forward pass must NOT flip the gate.
    loaded_only = {
        "mitigation": "hf_custom_devmap_x",
        "outcome": "loaded_no_forward",
        "forward_pass_confirmed": False,
    }
    assert exp.classify_loadability([loaded_only]) == (False, False, "both_failed")


def test_derive_verdict_blocked_and_success():
    assert exp.derive_verdict(False, False, "both_failed").startswith("blocked_")
    ok = exp.derive_verdict(True, True, "vllm_native")
    assert ok.startswith("success_") and "vllm_native" in ok


def test_peak_vram_across_takes_max():
    mits = [
        {"peak_vram_gib_per_gpu": {"gpu0": 5.0, "gpu1": 1.0}},
        {"peak_vram_gib_per_gpu": {"gpu0": 23.1}},
        {"peak_vram_gib_per_gpu": {"gpu0": "bad"}},  # ignored gracefully
        {},
    ]
    peaks = exp.peak_vram_across(mits)
    assert peaks["gpu0"] == pytest.approx(23.1)
    assert peaks["gpu1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# normalisation + ndjson parsing                                             #
# --------------------------------------------------------------------------- #
def test_normalise_mitigation_projects_required_keys():
    raw = {
        "mitigation": "x",
        "outcome": "load_failed",
        "error_if_any": "boom",
        "duration_s": "3.5",
        "forward_detail": "n/a",
    }
    n = exp.normalise_mitigation(raw)
    assert n["duration_s"] == 3.5 and n["forward_pass_confirmed"] is False
    assert n["mitigation"] == "x" and n["detail"] == "n/a"


def test_collect_attempts_from_ndjson(tmp_path):
    p = tmp_path / "probe.ndjson"
    lines = [
        json.dumps({"event": "registry", "vllm_version": "0.24.0"}),
        "not json at all",
        '{"truncated": "brace-started but invalid',  # exercises JSONDecodeError branch
        json.dumps({"event": "attempt", "mitigation": "a", "outcome": "load_failed",
                    "duration_s": 2.0}),
        json.dumps({"event": "attempt", "mitigation": "b", "outcome": "forward_pass_ok",
                    "forward_pass_confirmed": True}),
    ]
    p.write_text("\n".join(lines) + "\n")
    got = exp.collect_attempts_from_ndjson([p, tmp_path / "missing.ndjson"])
    assert [m["mitigation"] for m in got] == ["a", "b"]


# --------------------------------------------------------------------------- #
# SCENARIO-VERIFY-5196-BARE-GATE-FIELD + artifact assembly                    #
# --------------------------------------------------------------------------- #
def _precond():
    return [{"resource": "r", "available": True, "detail": "d"}]


def test_build_artifact_blocked_has_bare_gate_field():
    art = exp.build_artifact([_failed()], _precond(), duration_s=99.0)
    # The gate field MUST be a bare bool, not a {value, principle} wrapper.
    assert art["diffusiongemma_loadable"] is False
    assert isinstance(art["diffusiongemma_loadable"], bool)
    assert art["honest_verdict"].startswith("blocked_")
    assert art["loading_path_used"] == "both_failed"
    assert art["inference_substrate"] == "live_llm_inference"
    assert art["reproducibility_checksum"]
    assert "field_principles" in art and "diffusiongemma_loadable" in art["field_principles"]


def test_build_artifact_success_path_sets_gate_true():
    good = {
        "mitigation": "hf_custom_devmap_manual_split",
        "outcome": "forward_pass_ok",
        "forward_pass_confirmed": True,
        "duration_s": 120.0,
        "peak_vram_gib_per_gpu": {"gpu0": 18.0, "gpu1": 6.0},
    }
    art = exp.build_artifact([good], _precond(), duration_s=120.0)
    assert art["diffusiongemma_loadable"] is True
    assert art["forward_pass_confirmed"] is True
    assert art["honest_verdict"].startswith("success_")
    assert art["loading_path_used"] == "hf_custom_device_map"
    assert art["retirement"].startswith("N/A")


def test_build_artifact_records_all_required_and_is_valid():
    art = exp.build_artifact([_failed(), _failed("m2")], _precond(), 60.0)
    assert exp.validate_artifact(art) == []
    assert art["mitigations_tried"][0].keys() >= {
        "mitigation", "outcome", "error_if_any", "duration_s"
    }
    assert art["llama_cpp_pr_24427_status_checked"] == exp.LLAMA_CPP_PR_24427_STATUS


def test_checksum_is_deterministic_and_sensitive():
    a1 = exp.build_artifact([_failed()], _precond(), 10.0)
    a2 = exp.build_artifact([_failed()], _precond(), 10.0)
    assert a1["reproducibility_checksum"] == a2["reproducibility_checksum"]
    a3 = exp.build_artifact([_failed("different")], _precond(), 10.0)
    assert a3["reproducibility_checksum"] != a1["reproducibility_checksum"]


# --------------------------------------------------------------------------- #
# validate_artifact -- the guardrails                                         #
# --------------------------------------------------------------------------- #
def test_validate_rejects_wrapped_gate_field():
    art = exp.build_artifact([_failed()], _precond(), 10.0)
    art["diffusiongemma_loadable"] = {"value": False, "principle": "nope"}
    errs = exp.validate_artifact(art)
    assert any("BARE" in e for e in errs)


def test_validate_rejects_loadable_true_without_forward():
    art = exp.build_artifact([_failed()], _precond(), 10.0)
    art["diffusiongemma_loadable"] = True
    art["forward_pass_confirmed"] = False
    errs = exp.validate_artifact(art)
    assert any("requires forward_pass_confirmed" in e for e in errs)


def test_validate_rejects_missing_field_and_bad_verdict():
    art = exp.build_artifact([_failed()], _precond(), 10.0)
    del art["vllm_version"]
    art["honest_verdict"] = "diffusiongemma_blocked_no_prefix"
    errs = exp.validate_artifact(art)
    assert any("missing required field: vllm_version" in e for e in errs)
    assert any("terminal prefix" in e for e in errs)


def test_validate_rejects_wrong_substrate_and_bad_path():
    art = exp.build_artifact([_failed()], _precond(), 10.0)
    art["inference_substrate"] = "aggregation_from_upstream_artifacts"
    art["loading_path_used"] = "bogus"
    errs = exp.validate_artifact(art)
    assert any("inference_substrate" in e for e in errs)
    assert any("loading_path_used" in e for e in errs)


def test_validate_rejects_empty_mitigations_and_bad_peaks():
    art = exp.build_artifact([_failed()], _precond(), 10.0)
    art["mitigations_tried"] = []
    art["peak_vram_gib_per_gpu"] = ["not", "a", "dict"]
    errs = exp.validate_artifact(art)
    assert any("mitigations_tried must be a non-empty list" in e for e in errs)
    assert any("peak_vram_gib_per_gpu" in e for e in errs)


def test_validate_flags_mitigation_missing_keys_and_bad_seed():
    art = exp.build_artifact([_failed()], _precond(), 10.0)
    art["mitigations_tried"] = [{"mitigation": "x"}]  # missing outcome/duration_s
    art["random_seed"] = "not-an-int"
    errs = exp.validate_artifact(art)
    assert any("missing outcome" in e for e in errs)
    assert any("random_seed must be an int" in e for e in errs)


# --------------------------------------------------------------------------- #
# build_from_recorded + main                                                  #
# --------------------------------------------------------------------------- #
def test_build_from_recorded_includes_three_vllm_mitigations():
    art = exp.build_from_recorded([_failed(path_prefix="hf_custom_devmap")], 100.0)
    names = [m["mitigation"] for m in art["mitigations_tried"]]
    assert any(n.startswith("vllm_native_bnb4bit") for n in names)
    assert any(n.startswith("vllm_native_fp8") for n in names)
    assert exp.validate_artifact(art) == []


def test_build_from_recorded_default_uses_embedded_hf_and_is_valid():
    # With no explicit hf_mitigations, the embedded REAL HF outcomes are used, so
    # the artifact regenerates deterministically without the /tmp probe ndjson.
    art = exp.build_from_recorded()
    names = [m["mitigation"] for m in art["mitigations_tried"]]
    assert "hf_custom_devmap_manual_split_dec0_enc1_4bit" in names
    assert "hf_custom_devmap_colocate_gpu0_offload_vision_4bit" in names
    # duration_s defaults to the summed real wall-clock of every embedded attempt.
    assert art["duration_s"] > 60.0  # honest live-inference floor: real load attempts
    assert exp.validate_artifact(art) == []
    # The manual-split tie-confirmation finding is preserved in the detail.
    detail = {m["mitigation"]: m for m in art["mitigation_detail"]}
    assert "IGNORED" in detail["hf_custom_devmap_manual_split_dec0_enc1_4bit"]["detail"]


def test_build_from_recorded_default_is_deterministic():
    # Reproducibility: two default builds are byte-identical (same checksum).
    a1 = exp.build_from_recorded()
    a2 = exp.build_from_recorded()
    assert a1["reproducibility_checksum"] == a2["reproducibility_checksum"]


def test_main_print_mode(tmp_path, capsys):
    ndjson = tmp_path / "hf.ndjson"
    ndjson.write_text(
        json.dumps({"event": "attempt", "mitigation": "hf_custom_devmap_manual_split",
                    "outcome": "load_failed", "duration_s": 140.0,
                    "peak_vram_gib_per_gpu": {"gpu0": 23.1, "gpu1": 0.3}}) + "\n"
    )
    rc = exp.main(["--hf-ndjson", str(ndjson), "--print"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["diffusiongemma_loadable"] is False
    assert out["peak_vram_gib_per_gpu"]["gpu0"] == pytest.approx(23.1)


def test_main_handles_missing_hf_ndjson(tmp_path, capsys):
    rc = exp.main(["--hf-ndjson", str(tmp_path / "none.ndjson"), "--print"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # A missing probe ndjson falls back to the embedded REAL HF outcomes (so the
    # artifact stays reproducible), not a placeholder and not silently dropped.
    names = [m["mitigation"] for m in out["mitigations_tried"]]
    assert "hf_custom_devmap_manual_split_dec0_enc1_4bit" in names
    assert "hf_custom_devmap_colocate_gpu0_offload_vision_4bit" in names
    assert out["diffusiongemma_loadable"] is False


def test_main_writes_artifact(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    ndjson = tmp_path / "hf.ndjson"
    ndjson.write_text(
        json.dumps({"event": "attempt", "mitigation": "hf_custom_devmap_colocate",
                    "outcome": "load_failed", "duration_s": 5.0}) + "\n"
    )
    rc = exp.main(["--hf-ndjson", str(ndjson)])
    assert rc == 0
    written = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text())
    assert exp.validate_artifact(written) == []
    assert "wrote" in capsys.readouterr().out


def test_main_returns_1_on_invalid_artifact(monkeypatch, tmp_path):
    # If assembly ever produced an invalid artifact, main must fail loudly (rc=1).
    monkeypatch.setattr(exp, "build_from_recorded", lambda *a, **k: {"bad": True})
    rc = exp.main(["--hf-ndjson", str(tmp_path / "none.ndjson"), "--print"])
    assert rc == 1
