"""Tests for Exp 5182 DiffusionGemma meta-tensor root-cause fix (v475).

Spec refs: REQ-VERIFY-5182, SCENARIO-VERIFY-5182-ROOTCAUSE,
SCENARIO-VERIFY-5182-BLOCKED, SCENARIO-VERIFY-5182-BARE-GATE-FIELD.

These tests exercise the module's pure logic and its CPU-runnable helpers
(forward-pass confirmation with fake models, ladder orchestration with an
injected loader). The genuine 26B-model load is verified by the live GPU run
that writes the deliverable artifact, not by unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"

# This file legitimately imports torch / transformers / bitsandbytes (to test the
# CPU-runnable load-path helpers), which permanently bumps process RSS via the CUDA
# context + library import -- a one-time allocation, not a per-test leak. Opt out of the
# per-test RSS leak failure for the whole module; every assertion still runs.
pytestmark = pytest.mark.memory_watchdog_skip


def _outcome(label: str, confirmed: bool, outcome: str | None = None) -> mod.MitigationOutcome:
    """Build a MitigationOutcome for tests without touching a GPU."""
    return mod.MitigationOutcome(
        mitigation=label,
        outcome=outcome or ("forward_confirmed" if confirmed else "load_failed"),
        error_if_any=None if confirmed else "RuntimeError('boom')",
        duration_s=120.0,
        forward_confirmed=confirmed,
    )


def _versions() -> dict[str, str]:
    return {"transformers": "5.12.0", "accelerate": "1.14.0", "bitsandbytes": "0.49.2"}


def _preconds() -> list[dict[str, Any]]:
    return [
        {"resource": "gpu_free_for_4bit_load", "available": True, "detail": "gpu0: 23.5/24.0 GiB free"},
        {"resource": "diffusiongemma_weights_cached", "available": True, "detail": "8 shards"},
        {"resource": "transformers_accelerate_bitsandbytes", "available": True, "detail": "{}"},
    ]


# --- REQ-VERIFY-5182: the ladder never reuses .474's proven-fail device_map=auto loads ---


def test_ladder_has_no_474_overlap() -> None:
    """REQ-VERIFY-5182: every ladder mitigation is a genuinely new placement."""
    assert mod.ladder_overlap_with_474() == []


def test_ladder_first_mitigation_is_single_device() -> None:
    """SCENARIO-VERIFY-5182-ROOTCAUSE: mitigation 1 forces single-GPU placement."""
    m1 = mod.MITIGATION_LADDER[0]
    assert m1["device_map"] == {"": 0}
    assert m1["device_map"] != "auto"
    assert m1["bits"] == 4


def test_ladder_overlap_detects_a_plain_auto_ladder() -> None:
    """REQ-VERIFY-5182: the overlap guard actually fires on a .474-equivalent load."""
    bad = (
        {
            "label": "forblockdiffusion_4bit_nf4_devmap_auto_2gpu",
            "device_map": "auto",
            "no_split_override": None,
        },
    )
    assert mod.ladder_overlap_with_474(bad) == ["forblockdiffusion_4bit_nf4_devmap_auto_2gpu"]


def test_ladder_overlap_detects_maxmem_variant() -> None:
    """REQ-VERIFY-5182: the max_memory=24GiB .474 variant is also caught."""
    bad = (
        {
            "label": "x",
            "device_map": "auto",
            "no_split_override": None,
            "max_memory": "0:24GiB,1:24GiB",
        },
    )
    assert mod.ladder_overlap_with_474(bad) == ["x"]


# --- nf4 footprint precheck ---


def test_nf4_footprint_fits_single_gpu() -> None:
    """SCENARIO-VERIFY-5182-ROOTCAUSE: 26B at 4-bit is ~13 GiB and fits one 24 GiB GPU."""
    gib = mod.nf4_footprint_gib()
    assert 11.0 < gib < 15.0
    assert mod.fits_on_single_gpu() is True


def test_nf4_footprint_rejects_bad_inputs() -> None:
    """REQ-VERIFY-5182: footprint math guards invalid arguments."""
    with pytest.raises(ValueError):
        mod.nf4_footprint_gib(0)
    with pytest.raises(ValueError):
        mod.nf4_footprint_gib(26_000_000_000, bits_per_param=0)


def test_does_not_fit_on_tiny_gpu() -> None:
    """REQ-VERIFY-5182: a small GPU correctly reports the model does not fit."""
    assert mod.fits_on_single_gpu(gpu_gib=8.0) is False


# --- build_artifact success / blocked paths ---


def test_build_artifact_success_sets_bare_boolean() -> None:
    """SCENARIO-VERIFY-5182-BARE-GATE-FIELD: loadable is a bare True, verdict is complete."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(),
        versions=_versions(),
        mitigations=[_outcome("m1_single_device_gpu0_4bit_nf4", confirmed=True, outcome="forward_confirmed")],
        duration_s=140.0,
    )
    # The gate field is a raw JSON boolean, NOT a {value, principle} dict.
    assert art["diffusiongemma_loadable"] is True
    assert not isinstance(art["diffusiongemma_loadable"], dict)
    assert art["forward_pass_confirmed"] is True
    assert art["honest_verdict"].startswith("complete:")
    assert art["inference_substrate"] == "live_llm_inference"
    assert art["target_model"] == mod.MODEL_REPO
    mod.validate_artifact(art)  # must be schema-clean


def test_build_artifact_blocked_when_all_mitigations_fail() -> None:
    """SCENARIO-VERIFY-5182-BLOCKED: no confirmed forward -> loadable false, blocked verdict."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(),
        versions=_versions(),
        mitigations=[
            _outcome("m1_single_device_gpu0_4bit_nf4", confirmed=False),
            _outcome("m2_auto_explicit_no_split_4bit_nf4", confirmed=False),
        ],
        duration_s=300.0,
    )
    assert art["diffusiongemma_loadable"] is False
    assert art["forward_pass_confirmed"] is False
    assert art["honest_verdict"] == mod.BLOCKED_VERDICT
    mod.validate_artifact(art)


def test_loaded_no_forward_is_not_success() -> None:
    """REQ-VERIFY-5182: from_pretrained returning without a forward pass is NOT loadable."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(),
        versions=_versions(),
        mitigations=[_outcome("m1_single_device_gpu0_4bit_nf4", confirmed=False, outcome="loaded_no_forward")],
        duration_s=100.0,
    )
    assert art["diffusiongemma_loadable"] is False
    assert art["honest_verdict"] == mod.BLOCKED_VERDICT


def test_build_artifact_precondition_block_allows_empty_mitigations() -> None:
    """SCENARIO-VERIFY-5182-BLOCKED: a precondition block emits blocked_<resource> honestly."""
    art = mod.build_artifact(
        preconditions_checked=[
            {"resource": "gpu_free_for_4bit_load", "available": False, "detail": "gpu0 busy"},
        ],
        versions=_versions(),
        mitigations=[],
        duration_s=1.0,
        blocked_precondition="gpu_insufficient_free_memory",
    )
    assert art["diffusiongemma_loadable"] is False
    assert art["honest_verdict"] == "blocked_gpu_insufficient_free_memory"
    assert art["inference_substrate"] == "precondition_check_only"
    mod.validate_artifact(art)


# --- validate_artifact: the load-bearing bare-boolean + consistency checks ---


def test_validate_rejects_wrapped_gate_field() -> None:
    """SCENARIO-VERIFY-5182-BARE-GATE-FIELD: a {value, principle} wrapper is rejected."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    art["diffusiongemma_loadable"] = {"value": True, "principle": "wrapped"}
    with pytest.raises(ValueError, match="BARE top-level boolean"):
        mod.validate_artifact(art)


def test_validate_rejects_loadable_without_forward() -> None:
    """REQ-VERIFY-5182: loadable=true with forward_pass_confirmed=false is rejected."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    art["forward_pass_confirmed"] = False
    with pytest.raises(ValueError, match="requires forward_pass_confirmed"):
        mod.validate_artifact(art)


def test_validate_rejects_missing_field() -> None:
    """REQ-VERIFY-5182: a missing required field is reported."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    del art["root_cause"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(art)


def test_validate_rejects_bad_versions() -> None:
    """REQ-VERIFY-5182: versions dict must name all three libraries."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    art["transformers_accelerate_bitsandbytes_versions"] = {"transformers": "5.12.0"}
    with pytest.raises(ValueError, match="all three libs"):
        mod.validate_artifact(art)


def test_validate_rejects_474_overlap_in_recorded_mitigations() -> None:
    """REQ-VERIFY-5182: a recorded mitigation reusing a .474 label is rejected."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    art["mitigations_tried"] = [
        {"mitigation": "forblockdiffusion_4bit_nf4_devmap_auto_2gpu", "outcome": "load_failed",
         "error_if_any": "x", "duration_s": 148.0}
    ]
    with pytest.raises(ValueError, match="overlaps .474"):
        mod.validate_artifact(art)


def test_validate_rejects_bad_checksum() -> None:
    """REQ-VERIFY-5182: the reproducibility checksum must be a 64-char digest."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    art["reproducibility_checksum"] = "short"
    with pytest.raises(ValueError, match="64-char"):
        mod.validate_artifact(art)


def test_validate_rejects_empty_preconditions() -> None:
    """REQ-VERIFY-5182: preconditions_checked must be non-empty."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    art["preconditions_checked"] = []
    with pytest.raises(ValueError, match="preconditions_checked"):
        mod.validate_artifact(art)


def test_validate_rejects_empty_mitigations_without_block() -> None:
    """REQ-VERIFY-5182: an empty ladder with no precondition block is inconsistent."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    art["mitigations_tried"] = []
    with pytest.raises(ValueError, match="mitigations_tried is empty"):
        mod.validate_artifact(art)


def test_validate_rejects_non_terminal_verdict() -> None:
    """REQ-VERIFY-5182: honest_verdict must carry a terminal prefix."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    art["honest_verdict"] = "diffusiongemma_maybe_loaded"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(art)


# --- run_ladder orchestration (injected loader, no GPU) ---


def test_run_ladder_stops_at_first_success() -> None:
    """SCENARIO-VERIFY-5182-ROOTCAUSE: the ladder halts at the first confirmed forward."""
    calls: list[str] = []

    def fake_loader(spec: dict[str, Any]) -> mod.MitigationOutcome:
        calls.append(spec["label"])
        return _outcome(spec["label"], confirmed=True)

    outcomes = mod.run_ladder(loader=fake_loader)
    assert len(outcomes) == 1
    assert calls == [mod.MITIGATION_LADDER[0]["label"]]
    assert outcomes[0].forward_confirmed is True


def test_run_ladder_tries_all_when_all_fail() -> None:
    """SCENARIO-VERIFY-5182-BLOCKED: every mitigation runs when none confirms a forward."""
    def fake_loader(spec: dict[str, Any]) -> mod.MitigationOutcome:
        return _outcome(spec["label"], confirmed=False)

    outcomes = mod.run_ladder(loader=fake_loader)
    assert len(outcomes) == len(mod.MITIGATION_LADDER)
    assert all(not o.forward_confirmed for o in outcomes)


# --- forward-confirmation helpers (CPU tensors, no model load) ---


class _FakeOut:
    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)


class _FakeModel:
    """A minimal stand-in whose forward returns a real (CPU) tensor."""

    def __init__(self, out: Any, raise_on_call: bool = False) -> None:
        self._out = out
        self._raise = raise_on_call

    def parameters(self):  # noqa: ANN201 - test helper
        import torch

        yield torch.zeros(1)

    def eval(self) -> "_FakeModel":
        return self

    def __call__(self, **kwargs: Any) -> Any:
        if self._raise:
            raise RuntimeError("Tensor.item() cannot be called on meta tensors")
        return self._out


def test_confirm_forward_true_on_real_tensor() -> None:
    """SCENARIO-VERIFY-5182-ROOTCAUSE: a materialized output tensor confirms the forward."""
    import torch

    model = _FakeModel(_FakeOut(logits=torch.zeros(1, 5, 8)))
    ok, detail = mod._confirm_forward(model)
    assert ok is True
    assert "forward ok" in detail


def test_confirm_forward_false_when_model_raises() -> None:
    """SCENARIO-VERIFY-5182-BLOCKED: a meta-tensor error yields an unconfirmed forward."""
    model = _FakeModel(out=None, raise_on_call=True)
    ok, detail = mod._confirm_forward(model)
    assert ok is False
    assert "meta tensors" in detail


def test_extract_output_tensor_variants() -> None:
    """REQ-VERIFY-5182: the tensor extractor handles the diffusion output shapes."""
    import torch

    t = torch.zeros(1, 2)
    assert mod._extract_output_tensor(_FakeOut(logits=t)) is t
    assert mod._extract_output_tensor(_FakeOut(last_hidden_state=t)) is t
    assert mod._extract_output_tensor((t, "other")) is t

    class _Tupleable:
        def to_tuple(self) -> tuple[Any, ...]:
            return (t,)

    assert mod._extract_output_tensor(_Tupleable()) is t
    with pytest.raises(RuntimeError):
        mod._extract_output_tensor(_FakeOut())


# --- small helpers / serialization ---


def test_compute_verdict_mapping() -> None:
    """REQ-VERIFY-5182: verdict is success only when loadable AND forward confirmed."""
    assert mod.compute_verdict(True, True) == mod.SUCCESS_VERDICT
    assert mod.compute_verdict(False, False) == mod.BLOCKED_VERDICT
    assert mod.compute_verdict(True, False) == mod.BLOCKED_VERDICT


def test_mitigation_outcome_as_row() -> None:
    """REQ-VERIFY-5182: a mitigation row carries the four required keys."""
    row = _outcome("m1", confirmed=True).as_row()
    assert set(row) == {"mitigation", "outcome", "error_if_any", "duration_s"}


def test_checksum_is_deterministic_and_seed_sensitive() -> None:
    """REQ-VERIFY-5182: the checksum is stable for equal inputs, changes with the seed."""
    kw = dict(preconditions_checked=_preconds(), versions=_versions(),
              mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0)
    a = mod.build_artifact(**kw)
    b = mod.build_artifact(**kw)
    assert a["reproducibility_checksum"] == b["reproducibility_checksum"]
    c = mod.build_artifact(**kw, random_seed=999)
    assert c["reproducibility_checksum"] != a["reproducibility_checksum"]


def test_json_default_rejects_unknown_type() -> None:
    """REQ-VERIFY-5182: the checksum serializer refuses unserializable objects."""
    with pytest.raises(TypeError):
        mod.stable_checksum(object())


def test_write_result_roundtrip(tmp_path: Path) -> None:
    """REQ-VERIFY-5182: a written artifact reloads and revalidates cleanly."""
    art = mod.build_artifact(
        preconditions_checked=_preconds(), versions=_versions(),
        mitigations=[_outcome("m1", confirmed=True)], duration_s=120.0,
    )
    path = mod.write_result(art, tmp_path / "out.json")
    reloaded = json.loads(path.read_text())
    assert reloaded["diffusiongemma_loadable"] is True
    mod.validate_artifact(reloaded)


# --- cheap real helpers (no GPU) ---


def test_gather_versions_names_three_libs() -> None:
    """REQ-VERIFY-5182: the version probe reads all three installed libraries."""
    v = mod.gather_versions()
    assert {"transformers", "accelerate", "bitsandbytes"} <= set(v)
    assert all(isinstance(s, str) and s for s in v.values())


def test_check_preconditions_returns_records() -> None:
    """REQ-VERIFY-5182: the precondition probe returns a record per resource."""
    records, blocked = mod.check_preconditions()
    resources = {r["resource"] for r in records}
    assert {
        "gpu_free_for_4bit_load",
        "diffusiongemma_weights_cached",
        "transformers_accelerate_bitsandbytes",
    } <= resources
    # blocked is None (all pass) or a string naming the first failed resource.
    assert blocked is None or isinstance(blocked, str)


def test_bnb_config_supports_4bit_8bit_and_rejects_other() -> None:
    """REQ-VERIFY-5182: the quantization config supports the two ladder quant modes."""
    assert mod._bnb_config(4) is not None
    assert mod._bnb_config(8) is not None
    with pytest.raises(ValueError):
        mod._bnb_config(3)


def test_run_single_load_records_load_failure() -> None:
    """SCENARIO-VERIFY-5182-BLOCKED: a bad model class is recorded as a load_failed row."""
    bad_spec = {
        "label": "m_bad_class",
        "bits": 4,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "model_class_name": "NoSuchDiffusionGemmaClass",
        "no_split_override": None,
        "max_memory": None,
    }
    outcome = mod.run_single_load(bad_spec)
    assert outcome.outcome == "load_failed"
    assert outcome.forward_confirmed is False
    assert outcome.error_if_any is not None


# --- spec traceability + on-disk deliverable ---


def test_spec_declares_req_5182() -> None:
    """REQ-VERIFY-5182: the capability spec documents this experiment."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-5182" in text
    assert "SCENARIO-VERIFY-5182-BARE-GATE-FIELD" in text


def test_ondisk_deliverable_is_valid() -> None:
    """SCENARIO-VERIFY-5182-ROOTCAUSE: the produced deliverable is schema-clean on disk.

    The live GPU run writes this artifact before the suite is run for verification;
    the assertion confirms the real (non-faked) output honors the whole schema, and in
    particular that the gate field is a bare JSON boolean.
    """
    path = REPO / mod.RESULT_RELATIVE_PATH
    art = json.loads(path.read_text(encoding="utf-8"))
    mod.validate_artifact(art)
    assert isinstance(art["diffusiongemma_loadable"], bool)
    assert not isinstance(art["diffusiongemma_loadable"], dict)
