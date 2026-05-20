from __future__ import annotations

import pytest

from carnot.reporting.sklearn_prereq_fix_2609 import (
    REQUIRED_FIELDS,
    build_artifact,
    discover_fover_corpus,
    validate_artifact,
)


def _write_lines(path, n_lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join('{"context":"c","claim":"k","label":"correct"}\n' for _ in range(n_lines)))


def test_build_artifact_has_required_gate_fields() -> None:
    # REQ-REPORT-2609: downstream verifier tasks need explicit gate fields.
    artifact = build_artifact(
        sklearn_available=True,
        sklearn_version="1.8.0",
        sklearn_already_installed=False,
        carnot_import_ok=True,
        fover_corpus_found=True,
        fover_corpus_path="/tmp/project/data/fover_corpus.jsonl",
        preconditions_checked=[
            {
                "resource": "sklearn.linear_model.LogisticRegression",
                "available": True,
                "check": 'python -c "import sklearn; from sklearn.linear_model import LogisticRegression"',
            }
        ],
        install_method="sudo -n pacman -S --needed --noconfirm python-scikit-learn",
        install_success=True,
        install_attempts=[],
        python_executable="/usr/bin/python",
        python_version="3.14.4",
    )

    validate_artifact(artifact)
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["sklearn_available"] is True
    assert artifact["sklearn_already_installed"] is False


def test_validate_artifact_rejects_nonterminal_verdict() -> None:
    # REQ-REPORT-2609: blocked-looking verdicts would cause false partial classification.
    artifact = build_artifact(
        sklearn_available=True,
        sklearn_version="1.8.0",
        sklearn_already_installed=True,
        carnot_import_ok=True,
        fover_corpus_found=False,
        fover_corpus_path="not_found",
        preconditions_checked=[{"resource": "sklearn", "available": True, "check": "import sklearn"}],
        install_method="not_needed",
        install_success=True,
        install_attempts=[],
        python_executable="/usr/bin/python",
        python_version="3.14.4",
    )
    artifact["honest_verdict"] = "blocked_sklearn"

    with pytest.raises(ValueError, match="honest_verdict"):
        validate_artifact(artifact)


def test_discover_fover_corpus_prefers_prompt_paths(tmp_path) -> None:
    # REQ-REPORT-2609: exact prompt paths win when they contain enough rows.
    prompt_path = tmp_path / "data" / "foVer_corpus.jsonl"
    fallback_path = tmp_path / "data" / "fover_corpus.jsonl"
    _write_lines(prompt_path, 101)
    _write_lines(fallback_path, 200)

    found, resolved_path, checks = discover_fover_corpus(tmp_path)

    assert found is True
    assert resolved_path == str(prompt_path)
    assert checks[1]["resource"] == "data/foVer_corpus.jsonl"
    assert checks[1]["n_lines"] == 101


def test_discover_fover_corpus_uses_lowercase_canonical_fallback(tmp_path) -> None:
    # REQ-REPORT-2609: the current repo corpus is lowercase, so record it explicitly.
    fallback_path = tmp_path / "data" / "fover_corpus.jsonl"
    _write_lines(fallback_path, 101)

    found, resolved_path, checks = discover_fover_corpus(tmp_path)

    assert found is True
    assert resolved_path == str(fallback_path)
    assert checks[0]["available"] is False
    assert checks[1]["available"] is False
    assert checks[2]["resource"] == "data/fover_corpus.jsonl"
    assert checks[2]["n_lines"] == 101
