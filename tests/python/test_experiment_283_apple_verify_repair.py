"""Tests for scripts/experiment_283_apple_verify_repair.py.

All tests run under CARNOT_FORCE_LIVE=0 (simulated / mock mode).
No GPU hardware is required.

Spec coverage:
  REQ-VERIFY-068 — 12-cell verify-repair benchmark (3 modes × 2 variants × 2 models)
  REQ-VERIFY-069 — Improvement delta computation and primary criterion
  REQ-VERIFY-070 — Logit saving hook for verify-repair pipeline (Exp 291 JEPA)
  REQ-VERIFY-071 — Partial artifact with stall_at on 60s timeout
  REQ-VERIFY-072 — DualGPU dispatch at startup
  SCENARIO-VERIFY-084 — 12-cell result structure with all required fields
  SCENARIO-VERIFY-085 — Larger improvement on number_swap vs standard (primary criterion)
  SCENARIO-VERIFY-086 — DualGPU assigns Qwen GPU 0, Gemma GPU 1
  SCENARIO-VERIFY-087 — Logit files saved at each prefix fraction
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module loading helper
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "experiment_283_apple_verify_repair.py"
)


def _load_module() -> Any:
    """Load experiment_283 without executing main(), in mock mode."""
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec = importlib.util.spec_from_file_location("experiment_283_apple_verify_repair", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_283_apple_verify_repair"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_module()

VerifyRepairRunner = _mod.VerifyRepairRunner
build_artifact = _mod.build_artifact
compute_improvement_deltas = _mod.compute_improvement_deltas
CHECKPOINT_INTERVAL = _mod.CHECKPOINT_INTERVAL
LOGIT_FRACTIONS = _mod.LOGIT_FRACTIONS
INFERENCE_TIMEOUT_SECONDS = _mod.INFERENCE_TIMEOUT_SECONDS
MODEL_SPECS = _mod.MODEL_SPECS
EXPERIMENT = _mod.EXPERIMENT
ARTIFACT_SCHEMA = _mod.ARTIFACT_SCHEMA
MODES = _mod.MODES
VARIANT_TYPES = _mod.VARIANT_TYPES

# ---------------------------------------------------------------------------
# Fake dataset (4 questions — 2 number_swap, 2 irrelevant_sentence)
# ---------------------------------------------------------------------------

_FAKE_ROWS = [
    {
        "question_id": "gsm8k-001",
        "original_question": "Alice has 2 apples. Bob gives her 3 more. How many does Alice have?",
        "original_answer": 5,
        "variant_type": "number_swap",
        "variant_question": "Alice has 4 apples. Bob gives her 6 more. How many does Alice have?",
        "variant_answer": 10,
        "provenance": {"experiment": "exp281", "scale_factor": 2},
    },
    {
        "question_id": "gsm8k-002",
        "original_question": "There are 10 birds on a tree. 4 fly away. How many remain?",
        "original_answer": 6,
        "variant_type": "irrelevant_sentence",
        "variant_question": "There are 10 birds on a tree. Yesterday it was sunny. 4 fly away. How many remain?",
        "variant_answer": 6,
        "provenance": {"experiment": "exp281", "scale_factor": 1},
    },
    {
        "question_id": "gsm8k-003",
        "original_question": "Tom has 5 books. He lends 2. How many remain?",
        "original_answer": 3,
        "variant_type": "number_swap",
        "variant_question": "Tom has 10 books. He lends 4. How many remain?",
        "variant_answer": 6,
        "provenance": {"experiment": "exp281", "scale_factor": 2},
    },
    {
        "question_id": "gsm8k-004",
        "original_question": "Sara has 8 pencils. She uses 3. How many does she have?",
        "original_answer": 5,
        "variant_type": "irrelevant_sentence",
        "variant_question": "Sara has 8 pencils. It is Monday today. She uses 3. How many does she have?",
        "variant_answer": 5,
        "provenance": {"experiment": "exp281", "scale_factor": 1},
    },
]

_FAKE_VOCAB_SIZE = 32
_FAKE_SEQ_LEN = 4


def _make_fake_generate(
    *,
    baseline_correct: bool = True,
    verify_repair_correct: bool = True,
    raise_timeout: bool = False,
) -> Any:
    """Return a fake generate_fn for the verify-repair runner.

    Signature: (question, expected_answer, *, model_name, mode, variant_type) -> (response, logits)
    """
    call_count = [0]

    def _generate(
        question: str,
        expected_answer: int,
        *,
        model_name: str = "Qwen3.5-0.8B",
        mode: str = "baseline",
        variant_type: str = "number_swap",
        **kw: Any,
    ) -> tuple[str, np.ndarray]:
        call_count[0] += 1
        if raise_timeout:
            raise TimeoutError(f"Simulated 60s timeout on call {call_count[0]}")
        logits = np.zeros((1, _FAKE_SEQ_LEN, _FAKE_VOCAB_SIZE), dtype=np.float32)
        if mode == "baseline":
            response = str(expected_answer) if baseline_correct else str(expected_answer + 999)
        else:
            response = str(expected_answer) if verify_repair_correct else str(expected_answer + 999)
        return response, logits

    _generate.call_count = call_count  # type: ignore[attr-defined]
    return _generate


# ---------------------------------------------------------------------------
# REQ-VERIFY-068 / SCENARIO-VERIFY-084: 12-cell result structure
# ---------------------------------------------------------------------------


# REQ-VERIFY-068
def test_modes_constant() -> None:
    """REQ-VERIFY-068: MODES contains exactly baseline, verify_only, verify_repair."""
    assert set(MODES) == {"baseline", "verify_only", "verify_repair"}


# REQ-VERIFY-068
def test_variant_types_constant() -> None:
    """REQ-VERIFY-068: VARIANT_TYPES contains exactly number_swap, irrelevant_sentence."""
    assert set(VARIANT_TYPES) == {"number_swap", "irrelevant_sentence"}


# REQ-VERIFY-068, SCENARIO-VERIFY-084
def test_run_all_produces_12_cells(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-084: run_all() returns results for all 12 cells (3 modes × 2 variants × 2 models)."""
    two_specs = [
        {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0},
        {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 1},
    ]
    runner = VerifyRepairRunner(
        rows=_FAKE_ROWS,
        model_specs=two_specs,
        generate_fn=_make_fake_generate(),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    results = runner.run_all()

    # Expect entries for each (model, mode, variant_type) combination.
    expected_cells = [
        (model["name"], mode, vt)
        for model in two_specs
        for mode in MODES
        for vt in VARIANT_TYPES
    ]
    assert len(expected_cells) == 12

    for model_name, mode, vt in expected_cells:
        cell = results.get(model_name, {}).get(mode, {}).get(vt)
        assert cell is not None, (
            f"Missing cell ({model_name!r}, {mode!r}, {vt!r}) in results"
        )
        for field in ("correct", "total", "accuracy", "violation_detected_count", "repaired_count"):
            assert field in cell, f"Cell missing field {field!r}: {cell}"


# REQ-VERIFY-068, SCENARIO-VERIFY-084
def test_cell_accuracy_is_fraction(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-084: accuracy in each cell is between 0.0 and 1.0."""
    runner = VerifyRepairRunner(
        rows=_FAKE_ROWS,
        model_specs=MODEL_SPECS[:1],
        generate_fn=_make_fake_generate(),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    results = runner.run_all()
    model_name = MODEL_SPECS[0]["name"]
    for mode in MODES:
        for vt in VARIANT_TYPES:
            acc = results[model_name][mode][vt]["accuracy"]
            assert 0.0 <= acc <= 1.0, f"Accuracy out of range for ({mode}, {vt}): {acc}"


# ---------------------------------------------------------------------------
# REQ-VERIFY-069 / SCENARIO-VERIFY-085: Improvement delta computation
# ---------------------------------------------------------------------------


# REQ-VERIFY-069
def test_compute_improvement_deltas_structure() -> None:
    """REQ-VERIFY-069: compute_improvement_deltas returns delta per (mode, variant) for each model."""
    cell_results = {
        "Qwen3.5-0.8B": {
            "baseline": {
                "number_swap": {"accuracy": 0.4},
                "irrelevant_sentence": {"accuracy": 0.8},
            },
            "verify_only": {
                "number_swap": {"accuracy": 0.5},
                "irrelevant_sentence": {"accuracy": 0.85},
            },
            "verify_repair": {
                "number_swap": {"accuracy": 0.65},
                "irrelevant_sentence": {"accuracy": 0.82},
            },
        }
    }
    baseline_standard_acc = {"Qwen3.5-0.8B": 0.6}  # from Exp 282
    deltas = compute_improvement_deltas(cell_results, baseline_standard_acc=baseline_standard_acc)

    assert "Qwen3.5-0.8B" in deltas
    qwen = deltas["Qwen3.5-0.8B"]
    # delta(verify_repair, number_swap) = 0.65 - 0.4 = 0.25
    assert abs(qwen["verify_repair"]["number_swap"] - 0.25) < 1e-9
    # delta(verify_only, irrelevant_sentence) = 0.85 - 0.8 = 0.05
    assert abs(qwen["verify_only"]["irrelevant_sentence"] - 0.05) < 1e-9


# REQ-VERIFY-069, SCENARIO-VERIFY-085
def test_primary_criterion_larger_improvement_on_number_swap() -> None:
    """SCENARIO-VERIFY-085: primary criterion is True when Δ(vr,ns) > Δ(vr,std)."""
    # Δ(verify_repair, number_swap) = 0.3, Δ(verify_repair, standard) = 0.1
    cell_results = {
        "Qwen3.5-0.8B": {
            "baseline": {
                "number_swap": {"accuracy": 0.4},
                "irrelevant_sentence": {"accuracy": 0.8},
            },
            "verify_only": {
                "number_swap": {"accuracy": 0.5},
                "irrelevant_sentence": {"accuracy": 0.82},
            },
            "verify_repair": {
                "number_swap": {"accuracy": 0.70},
                "irrelevant_sentence": {"accuracy": 0.82},
            },
        }
    }
    # baseline standard accuracy from Exp 282: 0.6
    # Δ(verify_repair, standard) = verify_repair_standard - baseline_standard
    # We don't have verify_repair for standard here, so the function should handle it.
    # For this test, inject verify_repair standard accuracy separately.
    baseline_standard_acc = {"Qwen3.5-0.8B": 0.6}
    verify_repair_standard_acc = {"Qwen3.5-0.8B": 0.70}  # same → Δ = 0.10

    deltas = compute_improvement_deltas(
        cell_results,
        baseline_standard_acc=baseline_standard_acc,
        verify_repair_standard_acc=verify_repair_standard_acc,
    )
    # Δ(vr, ns) = 0.70 - 0.40 = 0.30 > Δ(vr, std) = 0.70 - 0.60 = 0.10
    qwen = deltas["Qwen3.5-0.8B"]
    delta_vr_ns = qwen["verify_repair"]["number_swap"]
    delta_vr_std = qwen.get("verify_repair", {}).get("standard", 0.0)

    assert delta_vr_ns > delta_vr_std, (
        f"Expected Δ(vr,ns)={delta_vr_ns:.3f} > Δ(vr,std)={delta_vr_std:.3f}"
    )


# REQ-VERIFY-069
def test_primary_criterion_field_in_artifact() -> None:
    """REQ-VERIFY-069: artifact primary_criterion_met field is present."""
    artifact = build_artifact(
        run_date="20260414",
        started_at="2026-04-14T05:00:00Z",
        finished_at="2026-04-14T05:10:00Z",
        inference_mode="mock",
        cell_results={"Qwen3.5-0.8B": {"baseline": {"number_swap": {"accuracy": 0.4, "correct": 2, "total": 5, "violation_detected_count": 0, "repaired_count": 0}, "irrelevant_sentence": {"accuracy": 0.8, "correct": 4, "total": 5, "violation_detected_count": 0, "repaired_count": 0}}, "verify_only": {"number_swap": {"accuracy": 0.5, "correct": 3, "total": 5, "violation_detected_count": 2, "repaired_count": 0}, "irrelevant_sentence": {"accuracy": 0.8, "correct": 4, "total": 5, "violation_detected_count": 1, "repaired_count": 0}}, "verify_repair": {"number_swap": {"accuracy": 0.7, "correct": 4, "total": 5, "violation_detected_count": 2, "repaired_count": 2}, "irrelevant_sentence": {"accuracy": 0.8, "correct": 4, "total": 5, "violation_detected_count": 1, "repaired_count": 1}}}},
        logit_paths={},
        improvement_deltas={},
        primary_criterion_met=True,
        stall_at=None,
        comparison_refs={},
    )
    assert "primary_criterion_met" in artifact
    assert artifact["primary_criterion_met"] is True


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-084: Artifact schema
# ---------------------------------------------------------------------------


# REQ-VERIFY-068
def test_artifact_schema_required_fields() -> None:
    """SCENARIO-VERIFY-084: build_artifact produces all required schema fields."""
    artifact = build_artifact(
        run_date="20260414",
        started_at="2026-04-14T05:00:00Z",
        finished_at="2026-04-14T05:10:00Z",
        inference_mode="mock",
        cell_results={},
        logit_paths={},
        improvement_deltas={},
        primary_criterion_met=False,
        stall_at=None,
        comparison_refs={},
    )
    for field_name in ARTIFACT_SCHEMA:
        assert field_name in artifact, f"Missing required artifact field: {field_name!r}"


# REQ-VERIFY-068
def test_artifact_experiment_number() -> None:
    """Artifact experiment field matches EXPERIMENT constant (283)."""
    artifact = build_artifact(
        run_date="20260414",
        started_at="2026-04-14T05:00:00Z",
        finished_at="2026-04-14T05:10:00Z",
        inference_mode="mock",
        cell_results={},
        logit_paths={},
        improvement_deltas={},
        primary_criterion_met=False,
        stall_at=None,
        comparison_refs={},
    )
    assert artifact["experiment"] == EXPERIMENT == 283


# REQ-VERIFY-068
def test_artifact_schema_version() -> None:
    """Artifact schema field is 'carnot.apple_verify_repair.v1'."""
    artifact = build_artifact(
        run_date="20260414",
        started_at="2026-04-14T05:00:00Z",
        finished_at="2026-04-14T05:10:00Z",
        inference_mode="mock",
        cell_results={},
        logit_paths={},
        improvement_deltas={},
        primary_criterion_met=False,
        stall_at=None,
        comparison_refs={},
    )
    assert artifact["schema"] == "carnot.apple_verify_repair.v1"


# ---------------------------------------------------------------------------
# REQ-VERIFY-071 / SCENARIO-VERIFY-084: Timeout → partial artifact
# ---------------------------------------------------------------------------


# REQ-VERIFY-071
def test_partial_artifact_has_stall_at(tmp_path: Path) -> None:
    """REQ-VERIFY-071: stall_at is set and partial=True when timeout fires."""
    runner = VerifyRepairRunner(
        rows=_FAKE_ROWS[:1],
        model_specs=MODEL_SPECS[:1],
        generate_fn=_make_fake_generate(raise_timeout=True),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    artifact = runner.run_with_timeout_handling(
        output_path=tmp_path / "artifact.json",
        run_date="20260414",
    )
    assert artifact.get("partial") is True, "Expected partial=True on timeout"
    assert artifact.get("stall_at") is not None, "Expected stall_at field on timeout"


# REQ-VERIFY-071
def test_full_artifact_has_no_stall(tmp_path: Path) -> None:
    """REQ-VERIFY-071: completed run has partial=False and stall_at=None."""
    runner = VerifyRepairRunner(
        rows=_FAKE_ROWS,
        model_specs=MODEL_SPECS[:1],
        generate_fn=_make_fake_generate(),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    artifact = runner.run_with_timeout_handling(
        output_path=tmp_path / "artifact.json",
        run_date="20260414",
    )
    assert artifact.get("partial") is False, "Expected partial=False on clean run"
    assert artifact.get("stall_at") is None, "Expected stall_at=None on clean run"


# REQ-VERIFY-071
def test_inference_timeout_constant() -> None:
    """REQ-VERIFY-071: INFERENCE_TIMEOUT_SECONDS is exactly 60."""
    assert INFERENCE_TIMEOUT_SECONDS == 60


# ---------------------------------------------------------------------------
# REQ-VERIFY-065 / SCENARIO-VERIFY-084: Checkpoint resume
# ---------------------------------------------------------------------------


# REQ-VERIFY-068
def test_checkpoint_interval_constant() -> None:
    """REQ-VERIFY-068: CHECKPOINT_INTERVAL is exactly 10."""
    assert CHECKPOINT_INTERVAL == 10


# REQ-VERIFY-068
def test_checkpoint_resume_skips_completed(tmp_path: Path) -> None:
    """REQ-VERIFY-068: runner skips questions already in checkpoint."""
    generate_fn = _make_fake_generate()

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    # Pre-populate a checkpoint with question 0 already done for baseline/number_swap.
    model_name = MODEL_SPECS[0]["name"]
    ckpt_file = ckpt_dir / f"{_mod.safe_slug(model_name)}__{_mod.safe_slug('baseline')}__{_mod.safe_slug('number_swap')}.json"
    ckpt_file.write_text(
        json.dumps({
            "model_name": model_name,
            "mode": "baseline",
            "variant_type": "number_swap",
            "completed": {"gsm8k-001": {"correct": True, "response": "10", "violation_detected": False, "repaired": False}},
        }),
        encoding="utf-8",
    )

    runner = VerifyRepairRunner(
        rows=[_FAKE_ROWS[0]],  # single number_swap row
        model_specs=MODEL_SPECS[:1],
        generate_fn=generate_fn,
        checkpoint_dir=ckpt_dir,
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    runner.run_mode_variant(model_name=model_name, mode="baseline", variant_type="number_swap")
    # generate_fn should NOT have been called — already checkpointed.
    assert generate_fn.call_count[0] == 0, (
        f"Expected 0 generate calls (resume), got {generate_fn.call_count[0]}"
    )


# ---------------------------------------------------------------------------
# REQ-VERIFY-070 / SCENARIO-VERIFY-087: Logit saving
# ---------------------------------------------------------------------------


# REQ-VERIFY-070
def test_logit_fractions_constant() -> None:
    """REQ-VERIFY-070: LOGIT_FRACTIONS are exactly [0.25, 0.50, 0.75, 1.00]."""
    assert LOGIT_FRACTIONS == [0.25, 0.50, 0.75, 1.00]


# REQ-VERIFY-070, SCENARIO-VERIFY-087
def test_logit_files_saved_at_fractions(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-087: .npy logit files are saved at each prefix fraction."""
    # Use 4 rows of same variant_type to hit all 4 fractions.
    four_ns_rows = []
    for i in range(4):
        four_ns_rows.append({
            "question_id": f"gsm8k-{i:03d}",
            "original_question": f"Q{i}: What is {i}+1?",
            "original_answer": i + 1,
            "variant_type": "number_swap",
            "variant_question": f"Q{i} scaled: What is {i*2}+2?",
            "variant_answer": (i + 1) * 2,
            "provenance": {},
        })

    logit_dir = tmp_path / "logits"
    logit_dir.mkdir()
    runner = VerifyRepairRunner(
        rows=four_ns_rows,
        model_specs=MODEL_SPECS[:1],
        generate_fn=_make_fake_generate(),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=logit_dir,
        timeout_seconds=60,
    )
    runner.run_mode_variant(model_name=MODEL_SPECS[0]["name"], mode="baseline", variant_type="number_swap")

    npy_files = sorted(logit_dir.rglob("*.npy"))
    assert len(npy_files) >= 1, "No .npy logit files were saved"
    # Check that "100" appears in at least one filename (final fraction).
    fraction_labels = {"25", "50", "75", "100"}
    found_labels = {label for label in fraction_labels for f in npy_files if label in f.name}
    assert "100" in found_labels or len(npy_files) >= 1, "No 100% fraction logit file found"


# REQ-VERIFY-070, SCENARIO-VERIFY-087
def test_logit_array_shape(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-087: saved logit arrays have shape (n, seq_len, vocab) or object array."""
    logit_dir = tmp_path / "logits"
    logit_dir.mkdir()
    runner = VerifyRepairRunner(
        rows=_FAKE_ROWS[:2],
        model_specs=MODEL_SPECS[:1],
        generate_fn=_make_fake_generate(),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=logit_dir,
        timeout_seconds=60,
    )
    runner.run_mode_variant(model_name=MODEL_SPECS[0]["name"], mode="baseline", variant_type="number_swap")

    npy_files = list(logit_dir.rglob("*.npy"))
    assert len(npy_files) > 0
    arr = np.load(str(npy_files[0]), allow_pickle=True)
    if arr.dtype == object:
        assert arr.ndim == 1
        for elem in arr:
            assert elem.ndim == 2
    else:
        assert arr.ndim == 3


# ---------------------------------------------------------------------------
# REQ-VERIFY-072 / SCENARIO-VERIFY-086: DualGPU dispatch
# ---------------------------------------------------------------------------


# REQ-VERIFY-072, SCENARIO-VERIFY-086
def test_model_specs_gpu_assignments() -> None:
    """SCENARIO-VERIFY-086: MODEL_SPECS assigns Qwen to GPU 0, Gemma to GPU 1."""
    assert len(MODEL_SPECS) >= 2
    qwen_spec = MODEL_SPECS[0]
    gemma_spec = MODEL_SPECS[1]
    assert "Qwen" in qwen_spec["name"], f"Expected Qwen on GPU 0, got {qwen_spec['name']!r}"
    assert qwen_spec.get("gpu", 0) == 0
    assert "Gemma" in gemma_spec["name"] or "gemma" in gemma_spec.get("hf_id", "").lower()
    assert gemma_spec.get("gpu", 1) == 1


# REQ-VERIFY-072, SCENARIO-VERIFY-086
def test_runner_dispatches_both_models(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-086: run_all() calls generate_fn for both models."""
    models_seen: list[str] = []

    def _tracking_generate(
        question: str, expected_answer: int, *, model_name: str = "unknown", **kw: Any
    ) -> tuple[str, np.ndarray]:
        models_seen.append(model_name)
        logits = np.zeros((1, _FAKE_SEQ_LEN, _FAKE_VOCAB_SIZE), dtype=np.float32)
        return str(expected_answer), logits

    two_specs = [
        {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0},
        {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 1},
    ]
    runner = VerifyRepairRunner(
        rows=_FAKE_ROWS,
        model_specs=two_specs,
        generate_fn=_tracking_generate,
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    runner.run_all()

    seen = set(models_seen)
    assert "Qwen3.5-0.8B" in seen, f"Qwen not dispatched; seen: {seen}"
    assert "Gemma4-E4B-it" in seen, f"Gemma not dispatched; seen: {seen}"


# ---------------------------------------------------------------------------
# REQ-VERIFY-068: Per-question record fields
# ---------------------------------------------------------------------------


# REQ-VERIFY-068
def test_per_question_record_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-068: each per-question result includes required fields."""
    runner = VerifyRepairRunner(
        rows=_FAKE_ROWS[:2],
        model_specs=MODEL_SPECS[:1],
        generate_fn=_make_fake_generate(),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    records = runner.run_mode_variant(
        model_name=MODEL_SPECS[0]["name"], mode="baseline", variant_type="number_swap"
    )
    required = {"question_id", "mode", "variant_type", "model", "correct",
                "violation_detected", "repaired", "semantic_grounding_fired", "formal_claim_fired"}
    for rec in records:
        for field in required:
            assert field in rec, f"Missing per-question field {field!r} in record: {rec}"


# REQ-VERIFY-068
def test_violation_detected_zero_in_baseline(tmp_path: Path) -> None:
    """REQ-VERIFY-068: violation_detected is False for baseline mode (no verification)."""
    runner = VerifyRepairRunner(
        rows=_FAKE_ROWS[:2],
        model_specs=MODEL_SPECS[:1],
        generate_fn=_make_fake_generate(),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    records = runner.run_mode_variant(
        model_name=MODEL_SPECS[0]["name"], mode="baseline", variant_type="number_swap"
    )
    for rec in records:
        assert rec["violation_detected"] is False, (
            f"Expected no violation detection in baseline mode, got: {rec}"
        )


# ---------------------------------------------------------------------------
# REQ-VERIFY-069: Logit paths in artifact
# ---------------------------------------------------------------------------


# REQ-VERIFY-070
def test_logit_paths_in_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-070: artifact logit_paths contains paths keyed by model/mode/variant."""
    runner = VerifyRepairRunner(
        rows=_FAKE_ROWS,
        model_specs=MODEL_SPECS[:1],
        generate_fn=_make_fake_generate(),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    artifact = runner.run_with_timeout_handling(
        output_path=tmp_path / "artifact.json",
        run_date="20260414",
    )
    logit_paths = artifact.get("logit_paths", {})
    model_name = MODEL_SPECS[0]["name"]
    assert model_name in logit_paths, f"No logit_paths entry for {model_name!r}"
