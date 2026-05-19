import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "python"))

import experiment_2548_real_corpus_validation as exp2548
from carnot.verify.real_corpus_validation import (
    compute_auroc,
    select_validation_corpus,
)


def _write_fover_rows(path: Path, n_rows: int = 60) -> None:
    rows = []
    for idx in range(n_rows):
        label = "incorrect" if idx % 2 else "correct"
        step_text = (
            f"Step {idx}: Compute {idx} + 1 = {idx + 1}. Therefore {idx + 1}."
            if label == "correct"
            else f"Step {idx}: Compute {idx} + 1 = {idx + 2}. Therefore {idx + 2}."
        )
        rows.append(
            {
                "question_id": f"case_{idx}",
                "step_text": step_text,
                "label": label,
                "confidence": 1.0,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows) + "\n", encoding="utf-8")


def test_req_verify_2548_auroc_uses_hallucination_positive_orientation() -> None:
    """REQ-VERIFY-2548: AUROC treats hallucination=1 as the positive class."""

    assert compute_auroc([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2]) == pytest.approx(1.0)
    assert compute_auroc([0, 1, 1, 0], [0.9, 0.1, 0.2, 0.8]) == pytest.approx(0.0)
    assert compute_auroc([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_req_verify_2548_selects_data_fover_before_results_fallback(tmp_path: Path) -> None:
    """REQ-VERIFY-2548: local data/FoVer files have priority over results fallbacks."""

    _write_fover_rows(tmp_path / "results" / "experiment_9999_fover_other.json", n_rows=80)
    _write_fover_rows(tmp_path / "data" / "fover_corpus_v4.json", n_rows=60)

    corpus = select_validation_corpus(repo_root=tmp_path, min_real_examples=50)

    assert corpus.corpus_type == "real"
    assert corpus.path == tmp_path / "data" / "fover_corpus_v4.json"
    assert corpus.n_real == 60
    assert corpus.label_counts == {0: 30, 1: 30}


def test_req_verify_2548_run_experiment_reports_citable_real_aurocs(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2548: real FoVer validation emits required citable fields."""

    _write_fover_rows(tmp_path / "data" / "fover_corpus_v4.json", n_rows=60)

    deliverable = exp2548.run_experiment(
        repo_root=tmp_path, results_dir=tmp_path / "results", write=False
    )

    required_fields = {
        "honest_verdict",
        "tier0r_real_auroc",
        "tier0s_real_auroc",
        "tier0u_real_auroc",
        "corpus_type",
        "n_real",
        "paper_citable",
        "preconditions_checked",
        "duration_s",
        "random_seed",
    }
    assert required_fields <= set(deliverable)
    assert deliverable["corpus_type"] == "real"
    assert deliverable["n_real"] == 60
    assert deliverable["random_seed"] == 42
    assert deliverable["acceptance_gates"]["corpus_type IS NOT NULL AND n_real >= 30"] is True
    for verifier in ("tier0r", "tier0s", "tier0u"):
        assert deliverable["paper_citable"][verifier] is True
        assert 0.0 <= deliverable[f"{verifier}_real_auroc"] <= 1.0


def test_req_verify_2548_writes_stable_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-2548: script writes the Exp 2548 artifact with field principles."""

    _write_fover_rows(tmp_path / "data" / "fover_corpus_v4.json", n_rows=60)

    deliverable = exp2548.run_experiment(
        repo_root=tmp_path, results_dir=tmp_path / "results", write=True
    )
    artifact_path = tmp_path / "results" / "experiment_2548_real_corpus_validation.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact == deliverable
    assert artifact["field_principles"]["paper_citable"].startswith("Dict of")
    assert artifact["preconditions_checked"]["selected_corpus"].endswith("fover_corpus_v4.json")
