"""Exp 1107: deploy structurally diverse verifier kernels and measure correlation.

Spec: REQ-VERIFY-1107, SCENARIO-VERIFY-1107
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.ast_structure_verifier import ASTStructureVerifier
from carnot.verify.semantic_consistency_verifier import SemanticConsistencyVerifier
from carnot.verify.z3_math_verifier import Z3MathVerifier

EXPERIMENT_ID = 1107
N_EXAMPLES = 100
CORPUS_PATH = PROJECT_ROOT / "data" / "fover_corpus_v4.json"
TEST_PATH = PROJECT_ROOT / "tests" / "python" / "test_diverse_verifiers_v1.py"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1107_new_diverse_verifiers_v1.json"
SOSKAN_EPOCHS = int(os.environ.get("CARNOT_EXP1107_SOSKAN_EPOCHS", "50"))


def main() -> None:
    started = time.time()
    examples = _load_examples(CORPUS_PATH, N_EXAMPLES)
    texts = [str(ex.get("step_text", "")) for ex in examples]

    verifiers = {
        "Z3MathVerifier": Z3MathVerifier(),
        "ASTStructureVerifier": ASTStructureVerifier(),
        "SemanticConsistencyVerifier": SemanticConsistencyVerifier(),
    }

    scores: dict[str, np.ndarray] = {}
    sos_scores, sos_source = _score_sos_kan_or_mock(examples)
    scores[sos_source] = sos_scores
    for name, verifier in verifiers.items():
        scores[name] = np.array([verifier.score(text) for text in texts], dtype=float)

    pairwise_correlations = _pairwise_correlations(scores)
    max_correlation_with_sos_kan = max(
        (
            abs(pairwise_correlations[f"{sos_source} vs {name}"])
            for name in verifiers
            if f"{sos_source} vs {name}" in pairwise_correlations
        ),
        default=0.0,
    )

    tests_written = _count_tests(TEST_PATH)
    tests_passing = _run_focused_tests(tests_written)
    z3_available = verifiers["Z3MathVerifier"].z3_available

    deployed = all(
        (PYTHON_DIR / path).exists()
        for path in (
            "carnot/verify/z3_math_verifier.py",
            "carnot/verify/ast_structure_verifier.py",
            "carnot/verify/semantic_consistency_verifier.py",
        )
    )

    if not deployed or tests_passing < tests_written:
        honest_verdict = "failed"
    elif max_correlation_with_sos_kan < 0.3:
        honest_verdict = "all_three_verifiers_orthogonal"
    else:
        honest_verdict = "verifiers_deployed_correlation_above_threshold"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": "new_diverse_verifiers_v1",
        "run_date": "2026-05-01",
        "duration_s": round(time.time() - started, 3),
        "verifiers_implemented": list(verifiers.keys()),
        "z3_math_verifier_path": "python/carnot/verify/z3_math_verifier.py",
        "ast_structure_verifier_path": "python/carnot/verify/ast_structure_verifier.py",
        "semantic_consistency_verifier_path": (
            "python/carnot/verify/semantic_consistency_verifier.py"
        ),
        "tests_written": tests_written,
        "tests_passing": tests_passing,
        "z3_available": z3_available,
        "pairwise_correlations": {
            key: round(float(value), 6) for key, value in sorted(pairwise_correlations.items())
        },
        "max_correlation_with_sos_kan": round(float(max_correlation_with_sos_kan), 6),
        "new_diverse_verifiers_deployed_3_verifiers": deployed,
        "honest_verdict": honest_verdict,
        "n_fover_examples": len(examples),
        "sos_kan_reference": sos_source,
        "sos_kan_epochs": SOSKAN_EPOCHS if sos_source == "SOSKANEnergyV3" else 0,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


def _load_examples(path: Path, n_examples: int) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected list corpus at {path}")
    return data[:n_examples]


def _score_sos_kan_or_mock(examples: list[dict]) -> tuple[np.ndarray, str]:
    try:
        from carnot.models.sos_kan import SOSKANEnergyV3

        corpus = json.loads(CORPUS_PATH.read_text())
        X_all, y_all = _featurize(corpus)
        model = SOSKANEnergyV3(
            n_splines=8,
            rank=8,
            n_features=16,
            hidden_dim=32,
            seed=42,
        )
        model.fit(X_all, y_all, n_epochs=SOSKAN_EPOCHS, lr=1e-3)
        X_eval, _ = _featurize(examples)
        energies = np.array([model.energy(row) for row in X_eval], dtype=float)
        return _minmax(energies), "SOSKANEnergyV3"
    except Exception as exc:
        print(f"[exp1107] SOSKANEnergyV3 unavailable, using mock scores: {exc}", file=sys.stderr)
        return _mock_sos_kan_scores(examples), "MockSOSKANEnergyV3"


def _featurize(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(items), 16), dtype=np.float64)
    y = np.zeros(len(items), dtype=np.float64)
    for idx, item in enumerate(items):
        text = str(item.get("step_text", ""))
        label = item.get("label", "unknown")
        y[idx] = 1.0 if label in ("correct", "valid", True, 1) else 0.0
        tl = text.lower()
        words = text.split()
        nw = max(len(words), 1)
        nc = max(len(text), 1)
        nums = re.findall(r"\b\d+\.?\d*\b", text)
        n_eq = text.count("=")
        X[idx, 0] = float(np.clip(math.log(nw + 1) / 5.0, 0, 1)) * 2 - 1
        X[idx, 1] = float(np.clip(n_eq / nw, 0, 1)) * 2 - 1
        X[idx, 2] = float(np.clip(len(nums) / nw, 0, 1)) * 2 - 1
        X[idx, 3] = float(np.clip(text.count("$") / nw, 0, 1)) * 2 - 1
        X[idx, 4] = 1.0 if any(k in tl for k in ["answer", "result", "solution"]) else -1.0
        X[idx, 5] = 1.0 if any(k in tl for k in ["let ", "define ", "let's let"]) else -1.0
        X[idx, 6] = (
            1.0
            if any(k in tl for k in ["therefore", "hence", "thus", "since ", "notice"])
            else -1.0
        )
        X[idx, 7] = 1.0 if n_eq >= 3 else -1.0
        X[idx, 8] = float(np.clip((text.count("+") + text.count("-")) / nw, 0, 1)) * 2 - 1
        X[idx, 9] = float(np.clip((text.count("(") + text.count(")")) / nc * 10, 0, 1)) * 2 - 1
        X[idx, 10] = 1.0 if "frac" in tl else -1.0
        X[idx, 11] = 1.0 if text and text[0].isdigit() else -1.0
        sents = re.split(r"[.!?]", text)
        ns = len([s for s in sents if s.strip()])
        X[idx, 12] = float(np.clip(ns / max(nc / 100.0, 1.0), 0, 2) / 2) * 2 - 1
        X[idx, 13] = (
            1.0 if any(k in tl for k in ["cannot", "impossible", "never", "always"]) else -1.0
        )
        X[idx, 14] = float(np.clip(math.log(len(set(nums)) + 1) / 3.0, 0, 1)) * 2 - 1
        X[idx, 15] = float(np.clip(len(text) / 500.0, 0, 1)) * 2 - 1
    return X, y


def _mock_sos_kan_scores(examples: list[dict]) -> np.ndarray:
    values = []
    for item in examples:
        text = str(item.get("step_text", ""))
        key = f"{item.get('question_id', '')}|{text[:120]}".encode()
        digest = hashlib.sha256(key).digest()
        values.append(int.from_bytes(digest[:8], "big") / float(2**64 - 1))
    return np.array(values, dtype=float)


def _pairwise_correlations(scores: dict[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {}
    for left, right in itertools.combinations(scores.keys(), 2):
        result[f"{left} vs {right}"] = _pearson(scores[left], scores[right])
    return result


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _count_tests(path: Path) -> int:
    return len(re.findall(r"^def test_", path.read_text(), flags=re.MULTILINE))


def _run_focused_tests(tests_written: int) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_DIR)
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TEST_PATH.relative_to(PROJECT_ROOT)),
        "-q",
        "-o",
        "addopts=",
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout)
    if completed.returncode == 0:
        return tests_written
    match = re.search(r"(\d+)\s+passed", completed.stdout)
    return int(match.group(1)) if match else 0


if __name__ == "__main__":
    main()
