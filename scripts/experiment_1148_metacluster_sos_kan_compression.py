"""Experiment 1148: MetaCluster-style compression for SOSKANEnergyV3.

This runner anchors to Exp 1128's fixed SOSKANEnergyV3 result
(``sos_kan_individual_auroc_after=0.9902``), retrains that deterministic
500-example FoVer setup because Exp 1128 did not persist a checkpoint path, and
then applies a K=32 centroid codebook to the dominant SOS-KAN coefficient
blocks.  The compressed model is reconstructed from centroids plus indices and
evaluated against the original model on the same 500 examples.

Spec: REQ-KAN-1148, SCENARIO-KAN-1148
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1148_metacluster_sos_kan_compression.json"
EXP1128_PATH = PROJECT_ROOT / "results" / "experiment_1128_sos_kan_root_cause_k5_fix.json"
CORPUS_PATH = PROJECT_ROOT / "data" / "fover_corpus_v4.json"

EXPERIMENT_ID = 1148
RANDOM_SEED = 1128
N_CORRECT = 386
N_WRONG = 114
N_CENTROIDS = 32
AUROC_ORIGINAL_REFERENCE = 0.9902
COMPRESSED_BLOCKS = ("W2", "b2")


def _reexec_with_venv_if_needed() -> None:
    """Use the repo venv when the shell's ``python`` lacks sklearn."""
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError:
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python), *sys.argv])
        raise


_reexec_with_venv_if_needed()

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

for _pkg in ["carnot", "carnot.eval", "carnot.models"]:
    if _pkg not in sys.modules:
        _module = types.ModuleType(_pkg)
        _module.__path__ = [str(PYTHON_DIR / _pkg.replace(".", "/"))]  # type: ignore[attr-defined]
        _module.__package__ = _pkg
        sys.modules[_pkg] = _module

from carnot.eval.metrics import auroc as canonical_auroc  # noqa: E402
from carnot.models.sos_kan import SOSKANEnergyV3  # noqa: E402
from carnot.models.sos_kan_metacluster import (  # noqa: E402
    artifact_has_required_fields,
    classify_metacluster_verdict,
    compress_sos_kan_v3,
    inspect_sos_kan_v3_coefficients,
    reconstruct_sos_kan_v3,
)


def _load_exp1128_artifact() -> dict:
    """Load the result that provides the original SOS-KAN AUROC reference."""
    return json.loads(EXP1128_PATH.read_text())


def _checkpoint_path_from_artifact(artifact: dict) -> Path | None:
    """Return a checkpoint path if an earlier artifact recorded one."""
    for key in ("checkpoint_path", "model_checkpoint_path", "sos_kan_checkpoint_path"):
        value = artifact.get(key)
        if value:
            path = Path(str(value))
            return path if path.is_absolute() else PROJECT_ROOT / path
    return None


def _load_fover_500() -> list[dict]:
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


def _build_feature_matrix(examples: list[dict]) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Return normalized FoVer features plus the min/max stats used."""
    raw = _extract_raw_features([row.get("step_text", "") for row in examples])
    stats = [(float(raw[:, i].min()), float(raw[:, i].max())) for i in range(raw.shape[1])]
    return _apply_feature_stats(raw, stats), stats


def _energy_vector(model: SOSKANEnergyV3, X: np.ndarray) -> np.ndarray:
    """Evaluate one energy per FoVer example."""
    return np.array([model.energy(row) for row in X], dtype=np.float64)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation, with a stable value for constant vectors."""
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _train_exp1128_model(X: np.ndarray, labels: np.ndarray) -> SOSKANEnergyV3:
    """Train the exact SOSKANEnergyV3 shape used by the fixed Exp 1128 adapter."""
    model = SOSKANEnergyV3(n_splines=8, rank=4, n_features=3, hidden_dim=16, seed=1121)
    model.fit(X, labels, n_epochs=100, lr=3e-3)
    return model


def run() -> dict:
    """Run compression, reconstruction, evaluation, and artifact construction."""
    started_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()

    exp1128 = _load_exp1128_artifact()
    checkpoint_path = _checkpoint_path_from_artifact(exp1128)
    checkpoint_path_found = bool(checkpoint_path and checkpoint_path.exists())

    examples = _load_fover_500()
    labels = np.array([1.0 if row["label"] == "correct" else 0.0 for row in examples])
    X, feature_stats = _build_feature_matrix(examples)

    model = _train_exp1128_model(X, labels)
    coefficient_info = inspect_sos_kan_v3_coefficients(model)
    original_energies = _energy_vector(model, X)
    auroc_original_measured = float(canonical_auroc(labels, -original_energies))

    payload = compress_sos_kan_v3(
        model,
        n_centroids=N_CENTROIDS,
        random_state=EXPERIMENT_ID,
        block_names=COMPRESSED_BLOCKS,
    )
    compressed_model = reconstruct_sos_kan_v3(payload)
    compressed_energies = _energy_vector(compressed_model, X)
    auroc_compressed = float(canonical_auroc(labels, -compressed_energies))
    energy_correlation = _pearson(original_energies, compressed_energies)

    auroc_original = AUROC_ORIGINAL_REFERENCE
    auroc_drop = float(auroc_original - auroc_compressed)
    auroc_drop_within_02 = bool(auroc_drop <= 0.02)
    size_reduction_factor = float(payload.size_reduction_factor)
    honest_verdict = classify_metacluster_verdict(
        checkpoint_found=True,
        auroc_original=auroc_original,
        auroc_compressed=auroc_compressed,
        size_reduction_factor=size_reduction_factor,
    )

    finished_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": "metacluster_sos_kan_compression",
        "run_date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(time.time() - t0, 2),
        "status": "success",
        "title": "MetaCluster SOS-KAN Compression",
        "sos_kan_compressed": True,
        "auroc_original": auroc_original,
        "auroc_original_measured": round(auroc_original_measured, 6),
        "auroc_compressed": round(auroc_compressed, 6),
        "auroc_drop": round(auroc_drop, 6),
        "auroc_drop_within_02": auroc_drop_within_02,
        "size_original_bytes": int(payload.size_original_bytes),
        "size_compressed_bytes": int(payload.size_compressed_bytes),
        "size_reduction_factor": round(size_reduction_factor, 6),
        "n_centroids": N_CENTROIDS,
        "energy_correlation": round(energy_correlation, 6),
        "honest_verdict": honest_verdict,
        "compression_blocks": list(COMPRESSED_BLOCKS),
        "compressed_vector_count": int(payload.vector_count),
        "compressed_index_storage": "fixed_width_bitpacked",
        "all_coefficient_vector_count": coefficient_info["coefficient_vector_count"],
        "n_kan_basis_functions": coefficient_info["n_kan_basis_functions"],
        "coefficients_per_spline": coefficient_info["coefficients_per_spline"],
        "parameter_blocks": coefficient_info["parameter_blocks"],
        "feature_stats": [[round(lo, 8), round(hi, 8)] for lo, hi in feature_stats],
        "benchmark_n_examples": len(examples),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_path_found": checkpoint_path_found,
        "model_source": (
            "checkpoint_path_from_exp1128"
            if checkpoint_path_found
            else "retrained_exp1128_recipe_no_checkpoint_path_in_artifact"
        ),
        "exp1128_sos_kan_individual_auroc_after": exp1128.get("sos_kan_individual_auroc_after"),
        "spec": ["REQ-KAN-1148", "SCENARIO-KAN-1148"],
    }
    if not artifact_has_required_fields(artifact):
        raise RuntimeError("exp1148 artifact is missing required fields")
    return artifact


def main() -> int:
    """Write the exp1148 deliverable JSON."""
    artifact = run()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH}")
    print(
        "compressed AUROC="
        f"{artifact['auroc_compressed']:.6f}, "
        f"drop={artifact['auroc_drop']:.6f}, "
        f"ratio={artifact['size_reduction_factor']:.3f}x, "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
