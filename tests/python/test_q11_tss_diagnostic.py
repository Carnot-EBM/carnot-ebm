"""Tests for the Q11 TSS diagnostic instrumentation.

Spec: REQ-VERIFY-1252, SCENARIO-VERIFY-1252
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import q11_tss_diagnostic as q11  # noqa: E402


class _FakeK5Ensemble:
    @property
    def verifier_names(self) -> list[str]:
        return ["v0", "v1", "v2", "v3", "v4"]

    def verify(self, question: str, response: str) -> SimpleNamespace:
        idx = int(response.rsplit(" ", 1)[-1])
        energy_rows = [
            [0.0, 0.25, 0.40, 0.40, 0.0],
            [0.0, 0.75, 0.40, 0.40, 0.0],
            [1.0, 0.25, 0.60, 0.40, 0.0],
            [0.0, 0.25, 0.40, 0.80, 0.0],
        ]
        scores = dict(zip(self.verifier_names, energy_rows[idx]))
        return SimpleNamespace(
            verified=all(energy < 0.5 for energy in scores.values()),
            per_verifier_scores=scores,
        )


def _write_corpus(path: Path) -> None:
    pairs = [
        {"question": f"question {idx}", "response": f"response {idx}"}
        for idx in range(4)
    ]
    path.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")


def test_diagnostic_runs_without_error(tmp_path, monkeypatch):
    """REQ-VERIFY-1252: CLI diagnostic writes a report without verifier errors."""
    corpus = tmp_path / "corpus.json"
    output = tmp_path / "report.json"
    _write_corpus(corpus)
    monkeypatch.setattr(q11, "_build_default_verifier_ensemble", _FakeK5Ensemble)

    rc = q11.main(
        [
            "--corpus",
            str(corpus),
            "--n_samples",
            "4",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    assert output.exists()


def test_output_has_required_fields(tmp_path, monkeypatch):
    """REQ-VERIFY-1252: report exposes Q11 triviality and occupancy fields."""
    corpus = tmp_path / "corpus.json"
    output = tmp_path / "report.json"
    _write_corpus(corpus)
    monkeypatch.setattr(q11, "_build_default_verifier_ensemble", _FakeK5Ensemble)

    report = q11.run_diagnostic(corpus, n_samples=4, output=output)

    assert report["smt_triviality_rates"] == {
        "v0": 0.75,
        "v1": 0.0,
        "v2": 0.0,
        "v3": 0.0,
        "v4": 1.0,
    }
    assert set(report) >= {
        "experiment",
        "verifier_names",
        "smt_triviality_rates",
        "orthant_occupancy",
        "and_occupancy",
        "tss_attack_viable",
    }
    assert report["tss_attack_viable"] is True


def test_and_occupancy_is_subset_of_single_verifier_occupancy(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-1252: AND occupancy cannot exceed any member occupancy."""
    corpus = tmp_path / "corpus.json"
    output = tmp_path / "report.json"
    _write_corpus(corpus)
    monkeypatch.setattr(q11, "_build_default_verifier_ensemble", _FakeK5Ensemble)

    report = q11.run_diagnostic(corpus, n_samples=4, output=output)

    assert report["and_occupancy"] == 0.25
    assert all(
        report["and_occupancy"] <= occupancy
        for occupancy in report["orthant_occupancy"].values()
    )
