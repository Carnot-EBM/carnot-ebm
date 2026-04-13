"""Tests for Experiment 262: predictive calibration corpus generation.

Verifies schema shape, label distribution reporting, and provenance field
presence under CARNOT_FORCE_LIVE=0 mock mode (no GPU / model required).

Spec: REQ-PRED-262-A (schema shape and required fields),
      REQ-PRED-262-B (label distribution reporting in summary artifact),
      REQ-PRED-262-C (provenance field presence),
      REQ-PRED-262-D (three prefix fractions per case),
      REQ-PRED-262-E (token pattern stats in summary)
SCENARIO-PRED-262-A (mock mode produces valid JSONL without GPU)
SCENARIO-PRED-262-B (summary artifact has all required top-level keys)
SCENARIO-PRED-262-C (pattern stats distinguish positive/negative cases)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Module loader helper
# ---------------------------------------------------------------------------


def _load_module(tmp_root: Path | None = None) -> Any:
    """Load experiment_262 script as a module, optionally overriding repo root."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_262_calibration_corpus.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_262_calibration_corpus", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if tmp_root is not None:
        # Patch CARNOT_REPO_ROOT so the module writes to a temp dir.
        os.environ["CARNOT_REPO_ROOT"] = str(tmp_root)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a .jsonl file into a list of dicts."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Required schema fields for each corpus row
# ---------------------------------------------------------------------------

_REQUIRED_ROW_FIELDS = {
    "case_id",
    "prefix_fraction",
    "token_feature_vector",
    "n_tokens_in_prefix",
    "token_pattern_features",
    "violation_label",
    "n_violations_final",
    "provenance_exp",
    "run_date",
    "experiment",
}

_REQUIRED_TOKEN_PATTERN_KEYS = {
    "digit_density",
    "operator_count",
    "equals_count",
    "sentence_count",
}

_REQUIRED_SUMMARY_KEYS = {
    "n_cases",
    "violation_rate",
    "prefix_fraction_feature_importance",
    "token_pattern_stats",
    "run_date",
    "experiment",
}

_VALID_PREFIX_FRACTIONS = {0.25, 0.50, 0.75}

# ---------------------------------------------------------------------------
# SCENARIO-PRED-262-A: mock mode produces valid JSONL without GPU
# ---------------------------------------------------------------------------


def test_mock_mode_produces_valid_jsonl(tmp_path: Any) -> None:
    """Mock mode (CARNOT_FORCE_LIVE=0) produces a readable JSONL with correct schema.

    Spec: REQ-PRED-262-A, SCENARIO-PRED-262-A
    """
    # Ensure live mode is disabled for this test.
    os.environ.pop("CARNOT_FORCE_LIVE", None)

    # Copy required source artifacts to tmp dir so the module can read them.
    repo_root = Path(__file__).resolve().parents[2]
    _mirror_source_artifacts(repo_root, tmp_path)

    module = _load_module(tmp_path)
    try:
        module.main(["--n-cases", "6", "--output-dir", str(tmp_path / "data" / "research")])
    finally:
        os.environ.pop("CARNOT_REPO_ROOT", None)

    corpus_path = tmp_path / "data" / "research" / "predictive_calibration_corpus_262.jsonl"
    assert corpus_path.exists(), f"Corpus file not written: {corpus_path}"

    rows = _read_jsonl(corpus_path)
    # 6 cases × 3 prefix fractions = 18 rows
    assert len(rows) == 18, f"Expected 18 rows (6 cases × 3 fractions), got {len(rows)}"


def test_schema_required_fields(tmp_path: Any) -> None:
    """Every corpus row has all required fields.

    Spec: REQ-PRED-262-A
    """
    os.environ.pop("CARNOT_FORCE_LIVE", None)
    repo_root = Path(__file__).resolve().parents[2]
    _mirror_source_artifacts(repo_root, tmp_path)

    module = _load_module(tmp_path)
    try:
        module.main(["--n-cases", "4", "--output-dir", str(tmp_path / "data" / "research")])
    finally:
        os.environ.pop("CARNOT_REPO_ROOT", None)

    corpus_path = tmp_path / "data" / "research" / "predictive_calibration_corpus_262.jsonl"
    rows = _read_jsonl(corpus_path)
    for i, row in enumerate(rows):
        missing = _REQUIRED_ROW_FIELDS - set(row.keys())
        assert not missing, f"Row {i} missing fields: {missing}"
        # Nested token_pattern_features
        tpf = row["token_pattern_features"]
        assert isinstance(tpf, dict), f"Row {i}: token_pattern_features must be dict"
        missing_tpf = _REQUIRED_TOKEN_PATTERN_KEYS - set(tpf.keys())
        assert not missing_tpf, f"Row {i} token_pattern_features missing: {missing_tpf}"
        # token_feature_vector is list of 9 floats
        vec = row["token_feature_vector"]
        assert isinstance(vec, list), f"Row {i}: token_feature_vector must be list"
        assert len(vec) == 9, f"Row {i}: token_feature_vector length {len(vec)} != 9"
        # prefix_fraction is one of the three expected values
        frac = row["prefix_fraction"]
        assert frac in _VALID_PREFIX_FRACTIONS, (
            f"Row {i}: prefix_fraction={frac!r} not in {_VALID_PREFIX_FRACTIONS}"
        )
        # violation_label is bool
        assert isinstance(row["violation_label"], bool), (
            f"Row {i}: violation_label must be bool, got {type(row['violation_label'])}"
        )


# ---------------------------------------------------------------------------
# SCENARIO-PRED-262-B: summary artifact has all required top-level keys
# ---------------------------------------------------------------------------


def test_summary_artifact_required_keys(tmp_path: Any) -> None:
    """Summary artifact contains all required top-level keys.

    Spec: REQ-PRED-262-B, SCENARIO-PRED-262-B
    """
    os.environ.pop("CARNOT_FORCE_LIVE", None)
    repo_root = Path(__file__).resolve().parents[2]
    _mirror_source_artifacts(repo_root, tmp_path)

    module = _load_module(tmp_path)
    try:
        module.main(["--n-cases", "4", "--output-dir", str(tmp_path / "data" / "research")])
    finally:
        os.environ.pop("CARNOT_REPO_ROOT", None)

    summary_path = tmp_path / "results" / "experiment_262_summary.json"
    assert summary_path.exists(), f"Summary not written: {summary_path}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = _REQUIRED_SUMMARY_KEYS - set(summary.keys())
    assert not missing, f"Summary missing keys: {missing}"


def test_summary_label_distribution(tmp_path: Any) -> None:
    """Summary reports a numeric violation_rate in [0, 1] and n_cases > 0.

    Spec: REQ-PRED-262-B
    """
    os.environ.pop("CARNOT_FORCE_LIVE", None)
    repo_root = Path(__file__).resolve().parents[2]
    _mirror_source_artifacts(repo_root, tmp_path)

    module = _load_module(tmp_path)
    try:
        module.main(["--n-cases", "6", "--output-dir", str(tmp_path / "data" / "research")])
    finally:
        os.environ.pop("CARNOT_REPO_ROOT", None)

    summary = json.loads(
        (tmp_path / "results" / "experiment_262_summary.json").read_text(encoding="utf-8")
    )
    assert summary["n_cases"] == 6, f"n_cases expected 6, got {summary['n_cases']}"
    vr = summary["violation_rate"]
    assert 0.0 <= vr <= 1.0, f"violation_rate {vr!r} not in [0,1]"


def test_summary_prefix_fraction_feature_importance(tmp_path: Any) -> None:
    """prefix_fraction_feature_importance dict has entries for all three fractions.

    Spec: REQ-PRED-262-B
    """
    os.environ.pop("CARNOT_FORCE_LIVE", None)
    repo_root = Path(__file__).resolve().parents[2]
    _mirror_source_artifacts(repo_root, tmp_path)

    module = _load_module(tmp_path)
    try:
        module.main(["--n-cases", "6", "--output-dir", str(tmp_path / "data" / "research")])
    finally:
        os.environ.pop("CARNOT_REPO_ROOT", None)

    summary = json.loads(
        (tmp_path / "results" / "experiment_262_summary.json").read_text(encoding="utf-8")
    )
    pfi = summary["prefix_fraction_feature_importance"]
    assert isinstance(pfi, dict), "prefix_fraction_feature_importance must be dict"
    for frac_key in ("0.25", "0.5", "0.75"):
        assert frac_key in pfi, f"prefix_fraction_feature_importance missing key {frac_key!r}"


# ---------------------------------------------------------------------------
# REQ-PRED-262-C: provenance field presence
# ---------------------------------------------------------------------------


def test_provenance_field_present_and_nonempty(tmp_path: Any) -> None:
    """Every row has a non-empty provenance_exp string.

    Spec: REQ-PRED-262-C, SCENARIO-PRED-262-A
    """
    os.environ.pop("CARNOT_FORCE_LIVE", None)
    repo_root = Path(__file__).resolve().parents[2]
    _mirror_source_artifacts(repo_root, tmp_path)

    module = _load_module(tmp_path)
    try:
        module.main(["--n-cases", "4", "--output-dir", str(tmp_path / "data" / "research")])
    finally:
        os.environ.pop("CARNOT_REPO_ROOT", None)

    corpus_path = tmp_path / "data" / "research" / "predictive_calibration_corpus_262.jsonl"
    rows = _read_jsonl(corpus_path)
    for i, row in enumerate(rows):
        prov = row.get("provenance_exp")
        assert prov and isinstance(prov, str), (
            f"Row {i}: provenance_exp must be non-empty string, got {prov!r}"
        )


# ---------------------------------------------------------------------------
# REQ-PRED-262-D: three prefix fractions per case
# ---------------------------------------------------------------------------


def test_three_prefix_fractions_per_case(tmp_path: Any) -> None:
    """Each case_id appears exactly three times (one row per prefix fraction).

    Spec: REQ-PRED-262-D
    """
    os.environ.pop("CARNOT_FORCE_LIVE", None)
    repo_root = Path(__file__).resolve().parents[2]
    _mirror_source_artifacts(repo_root, tmp_path)

    module = _load_module(tmp_path)
    try:
        module.main(["--n-cases", "5", "--output-dir", str(tmp_path / "data" / "research")])
    finally:
        os.environ.pop("CARNOT_REPO_ROOT", None)

    corpus_path = tmp_path / "data" / "research" / "predictive_calibration_corpus_262.jsonl"
    rows = _read_jsonl(corpus_path)

    from collections import Counter

    counts = Counter(row["case_id"] for row in rows)
    for case_id, cnt in counts.items():
        assert cnt == 3, (
            f"case_id {case_id!r} has {cnt} rows, expected 3 (one per prefix fraction)"
        )

    # Each case must have all three fractions
    from collections import defaultdict

    fractions_by_case: dict[str, set] = defaultdict(set)
    for row in rows:
        fractions_by_case[row["case_id"]].add(row["prefix_fraction"])
    for case_id, fracs in fractions_by_case.items():
        assert fracs == _VALID_PREFIX_FRACTIONS, (
            f"case_id {case_id!r} has fractions {fracs}, expected {_VALID_PREFIX_FRACTIONS}"
        )


# ---------------------------------------------------------------------------
# SCENARIO-PRED-262-C: token pattern stats in summary
# ---------------------------------------------------------------------------


def test_token_pattern_stats_in_summary(tmp_path: Any) -> None:
    """token_pattern_stats in summary contains positive/negative frequency dicts.

    Spec: REQ-PRED-262-E, SCENARIO-PRED-262-C
    """
    os.environ.pop("CARNOT_FORCE_LIVE", None)
    repo_root = Path(__file__).resolve().parents[2]
    _mirror_source_artifacts(repo_root, tmp_path)

    module = _load_module(tmp_path)
    try:
        module.main(["--n-cases", "8", "--output-dir", str(tmp_path / "data" / "research")])
    finally:
        os.environ.pop("CARNOT_REPO_ROOT", None)

    summary = json.loads(
        (tmp_path / "results" / "experiment_262_summary.json").read_text(encoding="utf-8")
    )
    tps = summary["token_pattern_stats"]
    assert isinstance(tps, dict), "token_pattern_stats must be dict"
    # Must have per-feature breakdown
    for feature in ("digit_density", "operator_count", "equals_count", "sentence_count"):
        assert feature in tps, f"token_pattern_stats missing feature {feature!r}"
        stat = tps[feature]
        assert "mean_positive" in stat, f"token_pattern_stats[{feature!r}] missing mean_positive"
        assert "mean_negative" in stat, f"token_pattern_stats[{feature!r}] missing mean_negative"


# ---------------------------------------------------------------------------
# Utility: mirror source artifacts to tmp_path
# ---------------------------------------------------------------------------


def _mirror_source_artifacts(repo_root: Path, tmp_path: Path) -> None:
    """Copy minimum required source files to tmp_path so the module resolves them."""
    import shutil

    # The module reads the Exp 235 GSM8K cohort.
    src = repo_root / "results" / "experiment_235_results.json"
    dst = tmp_path / "results" / "experiment_235_results.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)

    # Create the output directories.
    (tmp_path / "data" / "research").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
