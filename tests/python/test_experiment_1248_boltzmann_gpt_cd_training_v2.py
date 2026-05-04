"""Tests for Exp 1248 Boltzmann-GPT CD training v2.

Spec refs: REQ-KONA-022, SCENARIO-KONA-022.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.phase3 import boltzmann_gpt as bgpt  # noqa: E402


def _v2_rows(n_pairs: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n_pairs):
        rows.append(
            {
                "response": f"valid derivation exact total balanced proof {index}",
                "is_correct": True,
            }
        )
        rows.append(
            {
                "response": f"invalid derivation wrong mismatch arithmetic error {index}",
                "is_correct": False,
            }
        )
    return rows


def _write_v2_corpus(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"metadata": {}, "pairs": rows}), encoding="utf-8")
    return path


def test_balanced_fover_v5_loader_uses_minority_class_count(tmp_path: Path) -> None:
    """REQ-KONA-022: Exp 1248 builds a deterministic balanced FoVer v5 slice."""
    rows = _v2_rows(3) + [
        {"response": "extra incorrect distractor one", "is_correct": False},
        {"response": "extra incorrect distractor two", "is_correct": False},
    ]
    corpus_path = _write_v2_corpus(tmp_path / "fover_v5.json", rows)

    items_a, counts_a = bgpt.load_balanced_fover_v5_items(corpus_path, seed=1248)
    items_b, counts_b = bgpt.load_balanced_fover_v5_items(corpus_path, seed=1248)

    assert counts_a == counts_b
    assert counts_a["n_fover_v5_rows"] == 8
    assert counts_a["balanced_correct_count"] == 3
    assert counts_a["balanced_incorrect_count"] == 3
    assert [item.text for item in items_a] == [item.text for item in items_b]
    assert sum(item.label for item in items_a) == 3

    limited_items, limited_counts = bgpt.load_balanced_fover_v5_items(
        corpus_path,
        seed=1248,
        max_per_class=2,
    )
    assert limited_counts["balanced_correct_count"] == 2
    assert limited_counts["balanced_incorrect_count"] == 2
    assert len(limited_items) == 4


def test_run_cd_training_v2_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-KONA-022: Exp 1248 writes measured CD v2 artifact fields."""
    corpus_path = _write_v2_corpus(tmp_path / "fover_v5.json", _v2_rows(10))
    artifact_path = tmp_path / "experiment_1248.json"

    artifact = bgpt.run_cd_training_v2(
        corpus_path=corpus_path,
        artifact_path=artifact_path,
        n_cd_steps=8,
        lr=1e-2,
        visible_dim=8,
        hidden_dim=4,
        seed=1248,
        device="cpu",
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert bgpt.EXP1248_REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["experiment"] == "1248_boltzmann_gpt_cd_training_v2"
    assert payload["status"] == "complete"
    assert payload["forward_pass_ok"] is True
    assert payload["pre_cd_auroc"] == pytest.approx(0.65)
    assert payload["n_cd_steps"] == 8
    assert payload["balanced_correct_count"] == 10
    assert payload["balanced_incorrect_count"] == 10
    assert 0.0 <= payload["post_cd_auroc"] <= 1.0
    assert payload["honest_verdict"] == (
        f"boltzmann_gpt_cd_auroc_{payload['post_cd_auroc']:.2f}"
    )
