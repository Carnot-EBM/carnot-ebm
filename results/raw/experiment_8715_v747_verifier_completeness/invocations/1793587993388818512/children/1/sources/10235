"""Q11 Transversal Spectral Synthesis diagnostic for verifier null spaces.

Q11 predicts that Carnot's Phase-3 sign(z) bottleneck becomes attackable in
polynomial time when the verifier ensemble contains SMT-trivial components:
verifiers that return exact zero energy for most inputs and therefore leave a
large null space. This script instruments the production k=5 AND-composed
verifier ensemble on FoVer v5 pairs, measuring exact-zero SMT triviality,
single-verifier orthant occupancy, AND occupancy, and whether the observed
triviality makes the TSS attack surface viable.

Spec: REQ-VERIFY-1252, SCENARIO-VERIFY-1252
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

ACCEPTANCE_THRESHOLD = 0.5


def _ensure_cli_runtime() -> None:
    """Re-exec under the repo venv when CLI python lacks verifier dependencies."""
    try:
        import jax  # noqa: F401
    except ModuleNotFoundError:
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python), *sys.argv])
        raise


def _build_default_verifier_ensemble() -> Any:
    """Build the production k=5 ensemble from the canonical verifier module."""
    from carnot.verify.and_composition_verifier import build_default_verifier_ensemble

    return build_default_verifier_ensemble()


def _load_pairs(corpus: Path, n_samples: int) -> list[dict[str, Any]]:
    """Load FoVer-style question/response pairs from a JSON corpus."""
    data = json.loads(corpus.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        pairs = data.get("pairs")
    else:
        pairs = data

    if not isinstance(pairs, list):
        raise ValueError(f"{corpus} must contain a top-level list or a 'pairs' list")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if len(pairs) < n_samples:
        raise ValueError(f"{corpus} contains {len(pairs)} pairs, need {n_samples}")

    return [dict(pair) for pair in pairs[:n_samples]]


def _pair_text(pair: dict[str, Any]) -> tuple[str, str]:
    """Extract question and response text from a FoVer corpus row."""
    question = str(pair.get("question", pair.get("prompt", "")))
    response = str(
        pair.get(
            "response",
            pair.get("step_text", pair.get("completion", pair.get("answer", ""))),
        )
    )
    return question, response


def _finite_energy(value: Any) -> float:
    """Convert verifier output to a JSON-safe floating-point energy."""
    energy = float(value)
    if not math.isfinite(energy):
        return 1.0
    return energy


def run_diagnostic(corpus: Path, n_samples: int, output: Path) -> dict[str, Any]:
    """Run the Q11 diagnostic and write the JSON report.

    REQ-VERIFY-1252 requires per-verifier exact-zero rates, per-verifier
    orthant occupancy with energy < 0.5, AND occupancy, and a TSS viability
    flag whenever any verifier is SMT-trivial on more than half the sample.
    """
    pairs = _load_pairs(Path(corpus), n_samples)
    ensemble = _build_default_verifier_ensemble()
    verifier_names = list(ensemble.verifier_names)
    if len(verifier_names) != 5:
        raise ValueError(f"default verifier ensemble must have k=5, got {len(verifier_names)}")

    energies_by_verifier: dict[str, list[float]] = {name: [] for name in verifier_names}
    and_accepts: list[bool] = []

    for pair in pairs:
        question, response = _pair_text(pair)
        result = ensemble.verify(question, response)
        row_scores = {
            name: _finite_energy(result.per_verifier_scores[name]) for name in verifier_names
        }
        for name, energy in row_scores.items():
            energies_by_verifier[name].append(energy)
        and_accepts.append(
            all(energy < ACCEPTANCE_THRESHOLD for energy in row_scores.values())
        )

    sample_count = len(pairs)
    smt_triviality_rates = {
        name: sum(energy == 0.0 for energy in energies) / sample_count
        for name, energies in energies_by_verifier.items()
    }
    orthant_occupancy = {
        name: sum(energy < ACCEPTANCE_THRESHOLD for energy in energies) / sample_count
        for name, energies in energies_by_verifier.items()
    }
    and_occupancy = sum(and_accepts) / sample_count
    tss_attack_viable = any(rate > 0.5 for rate in smt_triviality_rates.values())

    report: dict[str, Any] = {
        "experiment": "1252_q11_tss_instrumentation",
        "spec": "REQ-VERIFY-1252",
        "corpus": str(corpus),
        "n_samples": sample_count,
        "acceptance_threshold": ACCEPTANCE_THRESHOLD,
        "verifier_names": verifier_names,
        "per_verifier_energies": energies_by_verifier,
        "smt_triviality_rates": smt_triviality_rates,
        "orthant_occupancy": orthant_occupancy,
        "and_occupancy": and_occupancy,
        "tss_attack_viable": tss_attack_viable,
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure Q11 TSS SMT triviality and orthant occupancy."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        _ensure_cli_runtime()
    args = _build_parser().parse_args(argv)
    report = run_diagnostic(args.corpus, args.n_samples, args.output)
    print(f"wrote {args.output}")
    print(f"smt_triviality_rates={report['smt_triviality_rates']}")
    print(f"orthant_occupancy={report['orthant_occupancy']}")
    print(f"and_occupancy={report['and_occupancy']}")
    print(f"tss_attack_viable={report['tss_attack_viable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
