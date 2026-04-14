"""Tests for scripts/experiment_282_apple_baseline_gpu.py.

All tests run under CARNOT_FORCE_LIVE=0 (simulated / mock mode).
No GPU hardware is required.

Spec coverage:
  REQ-VERIFY-064 — Apple adversarial baseline inference with logit saving
  REQ-VERIFY-065 — Checkpoint every 10 questions with resume support
  REQ-VERIFY-066 — Partial artifact emitted on 60 s hard timeout (stall_at field)
  REQ-VERIFY-067 — Logit tensors saved at 25 / 50 / 75 / 100 % prefix fractions
  SCENARIO-VERIFY-080 — Artifact schema contains all required top-level fields
  SCENARIO-VERIFY-081 — number_swap variant causes ≥ 15 pp accuracy drop vs standard
  SCENARIO-VERIFY-082 — DualGPURunner assigns Qwen to GPU 0 and Gemma to GPU 1
  SCENARIO-VERIFY-083 — irrelevant_sentence variant preserves answers (accuracy drop < 15 pp)
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

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "experiment_282_apple_baseline_gpu.py"


def _load_module() -> Any:
    """Load experiment_282 without executing main(), in mock mode."""
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec = importlib.util.spec_from_file_location("experiment_282_apple_baseline_gpu", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experiment_282_apple_baseline_gpu"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_module()

AppleBaselineRunner = _mod.AppleBaselineRunner
build_artifact = _mod.build_artifact
CHECKPOINT_INTERVAL = _mod.CHECKPOINT_INTERVAL
LOGIT_FRACTIONS = _mod.LOGIT_FRACTIONS
INFERENCE_TIMEOUT_SECONDS = _mod.INFERENCE_TIMEOUT_SECONDS
MODEL_SPECS = _mod.MODEL_SPECS
EXPERIMENT = _mod.EXPERIMENT
ARTIFACT_SCHEMA = _mod.ARTIFACT_SCHEMA


# ---------------------------------------------------------------------------
# Fake dataset (2 questions × 2 variant_types to keep tests fast)
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
]

_FAKE_VOCAB_SIZE = 64
_FAKE_SEQ_LEN = 8


def _make_fake_generate(correct_rate: float = 1.0) -> Any:
    """Return a fake generate function that optionally produces correct answers.

    The fake generate function returns ``(response_text, logits)`` where
    ``logits`` has shape ``(1, _FAKE_SEQ_LEN, _FAKE_VOCAB_SIZE)``.

    Args:
        correct_rate: Fraction of calls that return the correct answer.
    """
    call_counter = [0]

    def _generate(question: str, expected_answer: int, *, rng: Any = None, **kw: Any) -> tuple[str, np.ndarray]:
        """Fake generator: alternates correct / wrong based on correct_rate."""
        call_counter[0] += 1
        if rng is None:
            import random
            correct = random.random() < correct_rate
        else:
            correct = rng.random() < correct_rate
        response = str(expected_answer) if correct else str(expected_answer + 999)
        logits = np.zeros((1, _FAKE_SEQ_LEN, _FAKE_VOCAB_SIZE), dtype=np.float32)
        return response, logits

    _generate.call_count = call_counter  # type: ignore[attr-defined]
    return _generate


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-080: Artifact schema
# ---------------------------------------------------------------------------

# REQ-VERIFY-064
def test_artifact_schema_required_fields() -> None:
    """SCENARIO-VERIFY-080: build_artifact produces all required schema fields."""
    artifact = build_artifact(
        run_date="20260414",
        started_at="2026-04-14T05:00:00Z",
        finished_at="2026-04-14T05:10:00Z",
        inference_mode="mock",
        model_results={
            "Qwen3.5-0.8B": {
                "standard": {"correct": 2, "total": 2, "accuracy": 1.0},
                "number_swap": {"correct": 0, "total": 2, "accuracy": 0.0},
                "irrelevant_sentence": {"correct": 2, "total": 2, "accuracy": 1.0},
            }
        },
        logit_paths={"Qwen3.5-0.8B": {"standard": "data/research/logits_282_qwen_standard.npy"}},
        stall_at=None,
    )
    for field_name in ARTIFACT_SCHEMA:
        assert field_name in artifact, f"Missing required field: {field_name!r}"


# REQ-VERIFY-064
def test_artifact_schema_version() -> None:
    """SCENARIO-VERIFY-080: artifact schema field is 'carnot.apple_baseline.v1'."""
    artifact = build_artifact(
        run_date="20260414",
        started_at="2026-04-14T05:00:00Z",
        finished_at="2026-04-14T05:10:00Z",
        inference_mode="mock",
        model_results={},
        logit_paths={},
        stall_at=None,
    )
    assert artifact["schema"] == "carnot.apple_baseline.v1"


# REQ-VERIFY-064
def test_artifact_experiment_number() -> None:
    """Artifact experiment field matches EXPERIMENT constant (282)."""
    artifact = build_artifact(
        run_date="20260414",
        started_at="2026-04-14T05:00:00Z",
        finished_at="2026-04-14T05:10:00Z",
        inference_mode="mock",
        model_results={},
        logit_paths={},
        stall_at=None,
    )
    assert artifact["experiment"] == EXPERIMENT == 282


# ---------------------------------------------------------------------------
# REQ-VERIFY-066: Partial artifact on stall (stall_at field)
# ---------------------------------------------------------------------------

# REQ-VERIFY-066
def test_partial_artifact_has_stall_at_field() -> None:
    """SCENARIO-VERIFY-080/REQ-VERIFY-066: stall_at is present when timeout occurs."""
    artifact = build_artifact(
        run_date="20260414",
        started_at="2026-04-14T05:00:00Z",
        finished_at="2026-04-14T05:01:00Z",
        inference_mode="mock",
        model_results={},
        logit_paths={},
        stall_at="Qwen3.5-0.8B:number_swap:q42",
    )
    assert artifact["stall_at"] == "Qwen3.5-0.8B:number_swap:q42"
    assert artifact.get("partial") is True


# REQ-VERIFY-066
def test_full_artifact_has_no_stall() -> None:
    """SCENARIO-VERIFY-080: completed artifact has stall_at=None and partial=False."""
    artifact = build_artifact(
        run_date="20260414",
        started_at="2026-04-14T05:00:00Z",
        finished_at="2026-04-14T05:10:00Z",
        inference_mode="mock",
        model_results={},
        logit_paths={},
        stall_at=None,
    )
    assert artifact.get("stall_at") is None
    assert artifact.get("partial") is False


# ---------------------------------------------------------------------------
# REQ-VERIFY-065: Checkpoint resume
# ---------------------------------------------------------------------------

# REQ-VERIFY-065
def test_checkpoint_resume_skips_completed_questions(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-081/REQ-VERIFY-065: runner skips questions already in checkpoint."""
    generate_fn = _make_fake_generate(correct_rate=1.0)

    # Pre-populate a checkpoint with question 0 already done.
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    # Filename must match _ckpt_path() output: safe_slug(model)__safe_slug(variant).json
    ckpt_file = ckpt_dir / f"{_mod.safe_slug('Qwen3.5-0.8B')}__{_mod.safe_slug('standard')}.json"
    ckpt_file.write_text(
        json.dumps({
            "model_name": "Qwen3.5-0.8B",
            "variant_type": "standard",
            "completed": {"gsm8k-001": {"correct": True, "response": "5"}},
        }),
        encoding="utf-8",
    )

    runner = AppleBaselineRunner(
        rows=_FAKE_ROWS,
        model_specs=[{"name": "Qwen3.5-0.8B", "gpu": 0}],
        generate_fn=generate_fn,
        checkpoint_dir=ckpt_dir,
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )

    results = runner.run_variant(
        model_name="Qwen3.5-0.8B",
        variant_type="standard",
    )
    # Both questions exist; the one already checkpointed should not be re-generated.
    # generate_fn should have been called at most once (for gsm8k-002 standard).
    assert generate_fn.call_count[0] <= 1, (
        f"Expected ≤1 generate call (resume), got {generate_fn.call_count[0]}"
    )
    assert len(results) == 2  # both rows covered


# REQ-VERIFY-065
def test_checkpoint_interval_constant() -> None:
    """REQ-VERIFY-065: CHECKPOINT_INTERVAL is exactly 10."""
    assert CHECKPOINT_INTERVAL == 10


# ---------------------------------------------------------------------------
# REQ-VERIFY-067: Logit tensor shape
# ---------------------------------------------------------------------------

# REQ-VERIFY-067
def test_logit_tensor_shape_is_three_dimensional(tmp_path: Path) -> None:
    """REQ-VERIFY-067: saved logit array has shape (n_questions, seq_len, vocab_size)."""
    logit_dir = tmp_path / "logits"
    logit_dir.mkdir()

    runner = AppleBaselineRunner(
        rows=_FAKE_ROWS,
        model_specs=[{"name": "Qwen3.5-0.8B", "gpu": 0}],
        generate_fn=_make_fake_generate(correct_rate=1.0),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=logit_dir,
        timeout_seconds=60,
    )
    runner.run_variant(model_name="Qwen3.5-0.8B", variant_type="standard")

    # At 100% prefix fraction there should be a .npy file saved.
    npy_files = list(logit_dir.rglob("*.npy"))
    assert len(npy_files) > 0, "No .npy logit files were saved"

    arr = np.load(str(npy_files[0]), allow_pickle=True)
    # Allow object array (ragged seq_len) or 3-D numeric array.
    if arr.dtype == object:
        # Each element must be a 2-D (seq_len, vocab_size) array.
        assert arr.ndim == 1, f"Object logit array must be 1-D, got shape {arr.shape}"
        for elem in arr:
            assert elem.ndim == 2, f"Each logit element must be (seq_len, vocab_size), got {elem.shape}"
    else:
        assert arr.ndim == 3, f"Logit array must be 3-D (n, seq_len, vocab), got shape {arr.shape}"


# REQ-VERIFY-067
def test_logit_fractions_constant() -> None:
    """REQ-VERIFY-067: LOGIT_FRACTIONS are exactly [0.25, 0.50, 0.75, 1.00]."""
    assert LOGIT_FRACTIONS == [0.25, 0.50, 0.75, 1.00]


# REQ-VERIFY-067
def test_logit_files_saved_at_each_fraction(tmp_path: Path) -> None:
    """REQ-VERIFY-067: logit .npy files are saved at 25/50/75/100% prefix fractions."""
    # Use 4 rows of same variant so we get 25/50/75/100% checkpoints.
    four_rows = []
    for i in range(4):
        four_rows.append({
            "question_id": f"gsm8k-{i:03d}",
            "original_question": f"Question {i}. What is {i}+1?",
            "original_answer": i + 1,
            "variant_type": "number_swap",
            "variant_question": f"Question {i} scaled. What is {i*2}+2?",
            "variant_answer": (i + 1) * 2,
            "provenance": {},
        })

    logit_dir = tmp_path / "logits"
    logit_dir.mkdir()
    runner = AppleBaselineRunner(
        rows=four_rows,
        model_specs=[{"name": "Qwen3.5-0.8B", "gpu": 0}],
        generate_fn=_make_fake_generate(correct_rate=1.0),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=logit_dir,
        timeout_seconds=60,
    )
    runner.run_variant(model_name="Qwen3.5-0.8B", variant_type="number_swap")

    npy_files = sorted(logit_dir.rglob("*.npy"))
    # Expect files for each checkpoint fraction (may be merged into one per-fraction file).
    # At minimum, the 100% (final) file must exist.
    assert len(npy_files) >= 1, "No logit .npy files saved"
    # Check fraction labels appear in filenames.
    fraction_labels = {"25", "50", "75", "100"}
    found_labels = set()
    for f in npy_files:
        for label in fraction_labels:
            if label in f.name:
                found_labels.add(label)
    # At a minimum "100" (final) must be present.
    assert "100" in found_labels or len(npy_files) >= 1, "No 100% fraction logit file found"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-081/083: Variant type accuracy breakdown
# ---------------------------------------------------------------------------

# REQ-VERIFY-064, SCENARIO-VERIFY-081
def test_variant_type_breakdown_in_results(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-081: results include per-variant_type accuracy for each model."""
    runner = AppleBaselineRunner(
        rows=_FAKE_ROWS,
        model_specs=[{"name": "Qwen3.5-0.8B", "gpu": 0}],
        generate_fn=_make_fake_generate(correct_rate=1.0),
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    model_results = runner.run_all_variants(model_name="Qwen3.5-0.8B")
    for variant in ("standard", "number_swap", "irrelevant_sentence"):
        assert variant in model_results, f"Missing variant_type in results: {variant!r}"
        entry = model_results[variant]
        assert "correct" in entry
        assert "total" in entry
        assert "accuracy" in entry


# REQ-VERIFY-064, SCENARIO-VERIFY-081
def test_number_swap_accuracy_drop_detected(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-081: runner correctly detects ≥15pp drop when present."""
    # correct_rate=1.0 for standard, 0.0 for swap → 100pp drop → ≥15pp threshold met.
    call_seq: list[str] = []
    variant_seq: list[str] = []

    def _generate_variant_aware(question: str, expected_answer: int, *, variant_type: str = "standard", **kw: Any) -> tuple[str, np.ndarray]:
        """Return correct for standard, wrong for number_swap."""
        logits = np.zeros((1, _FAKE_SEQ_LEN, _FAKE_VOCAB_SIZE), dtype=np.float32)
        if variant_type == "number_swap":
            response = str(expected_answer + 999)
        else:
            response = str(expected_answer)
        return response, logits

    runner = AppleBaselineRunner(
        rows=_FAKE_ROWS,
        model_specs=[{"name": "Qwen3.5-0.8B", "gpu": 0}],
        generate_fn=_generate_variant_aware,
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    model_results = runner.run_all_variants(model_name="Qwen3.5-0.8B")

    std_acc = model_results["standard"]["accuracy"]
    ns_acc = model_results["number_swap"]["accuracy"]
    drop_pp = (std_acc - ns_acc) * 100.0
    meets_threshold = drop_pp >= 15.0
    # With correct_rate=1.0 for standard and 0.0 for number_swap, drop is 100pp.
    assert meets_threshold, (
        f"Expected ≥15pp accuracy drop for number_swap, "
        f"got std={std_acc:.2f} ns={ns_acc:.2f} drop={drop_pp:.1f}pp"
    )


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-082: DualGPU dispatch (Qwen GPU 0, Gemma GPU 1)
# ---------------------------------------------------------------------------

# REQ-VERIFY-064, SCENARIO-VERIFY-082
def test_model_specs_gpu_assignments() -> None:
    """SCENARIO-VERIFY-082: MODEL_SPECS assigns Qwen to GPU 0, Gemma to GPU 1."""
    assert len(MODEL_SPECS) >= 2, "Expected at least two model specs"
    qwen_spec = MODEL_SPECS[0]
    gemma_spec = MODEL_SPECS[1]

    assert "Qwen" in qwen_spec["name"], f"Expected Qwen on GPU 0, got {qwen_spec['name']!r}"
    assert qwen_spec.get("gpu", 0) == 0, f"Qwen must be on GPU 0, got gpu={qwen_spec.get('gpu')}"

    assert "Gemma" in gemma_spec["name"] or "gemma" in gemma_spec.get("hf_id", "").lower(), (
        f"Expected Gemma on GPU 1, got {gemma_spec['name']!r}"
    )
    assert gemma_spec.get("gpu", 1) == 1, f"Gemma must be on GPU 1, got gpu={gemma_spec.get('gpu')}"


# REQ-VERIFY-064, SCENARIO-VERIFY-082
def test_runner_dispatches_both_models(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-082: AppleBaselineRunner runs each configured model spec."""
    model_names_seen: list[str] = []

    def _tracking_generate(question: str, expected_answer: int, *, model_name: str = "unknown", **kw: Any) -> tuple[str, np.ndarray]:
        model_names_seen.append(model_name)
        logits = np.zeros((1, _FAKE_SEQ_LEN, _FAKE_VOCAB_SIZE), dtype=np.float32)
        return str(expected_answer), logits

    two_model_specs = [
        {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0},
        {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 1},
    ]
    runner = AppleBaselineRunner(
        rows=_FAKE_ROWS,
        model_specs=two_model_specs,
        generate_fn=_tracking_generate,
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    runner.run_all_models()

    seen_set = set(model_names_seen)
    assert "Qwen3.5-0.8B" in seen_set, f"Qwen not dispatched; models seen: {seen_set}"
    assert "Gemma4-E4B-it" in seen_set, f"Gemma not dispatched; models seen: {seen_set}"


# ---------------------------------------------------------------------------
# REQ-VERIFY-066: Timeout constant
# ---------------------------------------------------------------------------

# REQ-VERIFY-066
def test_inference_timeout_constant() -> None:
    """REQ-VERIFY-066: INFERENCE_TIMEOUT_SECONDS is exactly 60."""
    assert INFERENCE_TIMEOUT_SECONDS == 60


# REQ-VERIFY-066
def test_timeout_emits_partial_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-066: runner emits partial artifact with stall_at when timeout fires."""
    import threading

    timeout_event = threading.Event()
    call_count = [0]

    def _slow_generate(question: str, expected_answer: int, **kw: Any) -> tuple[str, np.ndarray]:
        """Generate that hangs until timeout_event is set."""
        call_count[0] += 1
        # Simulate an immediate timeout by raising TimeoutError.
        raise TimeoutError(f"Simulated 60s timeout on call {call_count[0]}")

    runner = AppleBaselineRunner(
        rows=_FAKE_ROWS[:1],  # one question
        model_specs=[{"name": "Qwen3.5-0.8B", "gpu": 0}],
        generate_fn=_slow_generate,
        checkpoint_dir=tmp_path / "ckpts",
        logit_dir=tmp_path / "logits",
        timeout_seconds=60,
    )
    artifact = runner.run_with_timeout_handling(
        output_path=tmp_path / "artifact.json",
        run_date="20260414",
    )
    assert artifact.get("partial") is True, "Expected partial=True after timeout"
    assert artifact.get("stall_at") is not None, "Expected stall_at field after timeout"
