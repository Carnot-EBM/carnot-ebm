#!/usr/bin/env python3
"""Experiment 1199: KANtize SOSKANEnergyV3 4-bit/8-bit quantization.

The Exp 1128 artifact reports the 0.9902 SOSKANEnergyV3 full-precision AUROC
reference but does not persist a checkpoint path.  This runner records the
available Exp 1130 SOSKANEnergyV3 checkpoint discovery, then retrains the
deterministic Exp 1128 500-example FoVer recipe so the 32-bit/8-bit/4-bit
comparison is anchored to the requested full-precision reference.

Spec: REQ-KAN-1199, SCENARIO-KAN-1199
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _reexec_with_venv_if_needed() -> None:
    """Re-run under the repo venv when ``python3`` lacks runtime dependencies."""
    required = ("safetensors", "jax")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_venv_if_needed()

from carnot.models.sos_kan import SOSKANEnergyV3  # noqa: E402
from carnot.models.sos_kan_quantization import (  # noqa: E402
    artifact_has_required_fields,
    classify_kantize_verdict,
    estimate_sos_kan_v3_size_bytes,
    evaluate_sos_kan_v3_auroc,
    export_quantized_sos_kan_v3_safetensors,
    load_sos_kan_v3_npz,
    measure_per_sample_latency_ms,
    quantize_sos_kan_v3_spline_weights,
)

EXPERIMENT_ID = 1199
RANDOM_SEED = 1128
N_CORRECT = 386
N_WRONG = 114
AUROC_THRESHOLD = 0.97

CORPUS_PATH = PROJECT_ROOT / "data" / "fover_corpus_v4.json"
EXP1128_PATH = PROJECT_ROOT / "results" / "experiment_1128_sos_kan_root_cause_k5_fix.json"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "experiment_1130_soskan_energy_v3_retrained.npz"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1199_kantize_soskan_4bit_quantization.json"
Q4_CHECKPOINT_PATH = PROJECT_ROOT / "results" / "experiment_1199_soskan_energy_v3_q4.safetensors"


def _utc_now() -> str:
    """Return current UTC timestamp in artifact format."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_fover_500() -> list[dict[str, Any]]:
    """Load the same 114 incorrect and 386 correct FoVer examples as Exp 1128."""
    data = json.loads(CORPUS_PATH.read_text())
    correct = [row for row in data if row["label"] == "correct"]
    wrong = [row for row in data if row["label"] == "incorrect"]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(correct)
    rng.shuffle(wrong)
    examples = wrong[:N_WRONG] + correct[:N_CORRECT]
    rng.shuffle(examples)
    return examples


def _extract_raw_features(texts: list[str]) -> np.ndarray:
    """Extract the three raw text features used by the Exp 1128 adapter."""
    feats = []
    for text in texts:
        words = text.split()
        n_words = max(len(words), 1)
        num_count = sum(1 for word in words if any(char.isdigit() for char in word))
        feats.append([float(np.log(len(text) + 1)), num_count / n_words, len(set(words)) / n_words])
    return np.array(feats, dtype=np.float64)


def _apply_feature_stats(arr: np.ndarray, stats: list[tuple[float, float]]) -> np.ndarray:
    """Normalize raw features to [-1, 1] with training-set min/max anchors."""
    result = np.array(arr, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(stats):
        if hi > lo:
            result[:, i] = np.clip(2.0 * (result[:, i] - lo) / (hi - lo) - 1.0, -1.0, 1.0)
        else:
            result[:, i] = 0.0
    return result


def _build_feature_matrix(
    examples: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Return normalized FoVer features, labels, and normalization stats."""
    raw = _extract_raw_features([row.get("step_text", "") for row in examples])
    stats = [(float(raw[:, i].min()), float(raw[:, i].max())) for i in range(raw.shape[1])]
    X = _apply_feature_stats(raw, stats)
    y = np.array([1.0 if row["label"] == "correct" else 0.0 for row in examples])
    return X, y, stats


def _train_exp1128_model(X: np.ndarray, labels: np.ndarray) -> SOSKANEnergyV3:
    """Train the exact SOSKANEnergyV3 shape used by the fixed Exp 1128 adapter."""
    model = SOSKANEnergyV3(n_splines=8, rank=4, n_features=3, hidden_dim=16, seed=1121)
    model.fit(X, labels, n_epochs=100, lr=3e-3)
    return model


def _checkpoint_probe() -> dict[str, Any]:
    """Describe the existing SOSKANEnergyV3 checkpoint, if it can be loaded."""
    if not CHECKPOINT_PATH.exists():
        return {"checkpoint_path": str(CHECKPOINT_PATH), "checkpoint_found": False}
    try:
        model, metadata = load_sos_kan_v3_npz(CHECKPOINT_PATH)
    except Exception as exc:  # pragma: no cover - defensive artifact diagnostics
        return {
            "checkpoint_path": str(CHECKPOINT_PATH),
            "checkpoint_found": True,
            "checkpoint_loadable": False,
            "checkpoint_error": str(exc),
        }
    return {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_found": True,
        "checkpoint_loadable": True,
        "checkpoint_architecture": {
            "n_splines": model.n_splines,
            "rank": model.rank,
            "n_features": model.n_features,
            "hidden_dim": model.hidden_dim,
        },
        "checkpoint_metadata": metadata,
    }


def _mb(size_bytes: int) -> float:
    """Convert bytes to MiB for artifact reporting."""
    return round(float(size_bytes) / (1024.0 * 1024.0), 6)


def run() -> dict[str, Any]:
    """Run quantization, evaluation, safetensors export, and artifact assembly."""
    started_at = _utc_now()
    t0 = time.time()

    exp1128 = _load_json(EXP1128_PATH)
    checkpoint_info = _checkpoint_probe()
    examples = _load_fover_500()
    X, labels, feature_stats = _build_feature_matrix(examples)

    full_model = _train_exp1128_model(X, labels)
    q8_result = quantize_sos_kan_v3_spline_weights(full_model, bits=8)
    q4_result = quantize_sos_kan_v3_spline_weights(full_model, bits=4)

    full_auroc = evaluate_sos_kan_v3_auroc(full_model, X, labels)
    q8_auroc = evaluate_sos_kan_v3_auroc(q8_result.model, X, labels)
    q4_auroc = evaluate_sos_kan_v3_auroc(q4_result.model, X, labels)
    q4_latency_ms = measure_per_sample_latency_ms(q4_result.model, X, n_samples=100)

    q4_export = export_quantized_sos_kan_v3_safetensors(
        q4_result.model,
        Q4_CHECKPOINT_PATH,
        bits=4,
        source_metadata={
            "source_experiment": EXPERIMENT_ID,
            "source_model": "retrained_exp1128_recipe_no_exp1128_checkpoint_path",
            "fover_examples": len(examples),
        },
    )

    q4_ready = bool(q4_auroc >= AUROC_THRESHOLD)
    honest_verdict = classify_kantize_verdict(q4_auroc, q8_auroc, threshold=AUROC_THRESHOLD)

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": "kantize_soskan_4bit_quantization",
        "run_date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_s": round(time.time() - t0, 3),
        "status": "success",
        "title": "KANtize SOSKANEnergyV3 4-bit/8-bit Quantization",
        "soskan_full_precision_auroc": round(float(full_auroc), 6),
        "soskan_8bit_auroc": round(float(q8_auroc), 6),
        "soskan_4bit_auroc": round(float(q4_auroc), 6),
        "soskan_full_precision_size_mb": _mb(estimate_sos_kan_v3_size_bytes(full_model)),
        "soskan_8bit_size_mb": _mb(estimate_sos_kan_v3_size_bytes(full_model, bits=8)),
        "soskan_4bit_size_mb": _mb(estimate_sos_kan_v3_size_bytes(full_model, bits=4)),
        "soskan_4bit_inference_latency_ms": round(float(q4_latency_ms), 6),
        "kantize_4bit_checkpoint_path": str(q4_export),
        "kantize_auroc_maintained_above_0p97": q4_ready,
        "edge_deployment_ready": q4_ready,
        "honest_verdict": honest_verdict,
        "soskan_class": "carnot.models.sos_kan.SOSKANEnergyV3",
        "model_source": "retrained_exp1128_recipe_no_exp1128_checkpoint_path",
        "checkpoint_discovery": checkpoint_info,
        "exp1128_reference_auroc": exp1128.get("sos_kan_individual_auroc_after"),
        "benchmark_n_examples": len(examples),
        "feature_stats": [[round(lo, 8), round(hi, 8)] for lo, hi in feature_stats],
        "quantization": {
            "spline_parameter_names": list(q4_result.quantized_parameter_names),
            "interior_8bit_interval": "1/255",
            "endpoint_8bit_interval": "1/65535",
            "interior_4bit_interval": "1/15",
            "endpoint_4bit_interval": "1/255",
            "non_spline_parameters": ["W1", "b1", "c"],
        },
        "spec": ["REQ-KAN-1199", "SCENARIO-KAN-1199"],
    }
    if not artifact_has_required_fields(artifact):
        raise RuntimeError("Exp 1199 artifact is missing required fields")
    return artifact


def main() -> int:
    """Write the Exp 1199 deliverable JSON."""
    _reexec_with_venv_if_needed()
    artifact = run()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH}")
    print(
        "AUROC full/8/4="
        f"{artifact['soskan_full_precision_auroc']:.6f}/"
        f"{artifact['soskan_8bit_auroc']:.6f}/"
        f"{artifact['soskan_4bit_auroc']:.6f}; "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
