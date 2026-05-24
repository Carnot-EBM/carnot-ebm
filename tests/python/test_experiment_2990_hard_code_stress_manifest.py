"""Tests for Exp 2990 verifier-backed hard-code stress manifest.

Spec: REQ-CODE-2990, SCENARIO-CODE-2990.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import hard_code_stress_manifest as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "code-verification" / "spec.md"
REQUIRED_FIELDS = {
    "hard_code_stress_set_ready",
    "manifest_path",
    "n_items",
    "all_items_have_tests",
    "all_baseline_candidates_fail",
    "all_reference_solutions_pass",
    "flaky_items",
    "verifier_transcript_paths",
    "hard_generation_sources",
    "honest_verdict",
}


def test_req_code_2990_spec_anchor_exists() -> None:
    """REQ-CODE-2990: the hard-code stress manifest is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-2990" in spec
    assert "SCENARIO-CODE-2990" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "datasets/repair_hard/manifest_v1.jsonl" in spec


def test_scenario_code_2990_writes_ready_manifest_and_transcript(tmp_path: Path) -> None:
    """SCENARIO-CODE-2990: every included hard item has executable pass/fail evidence."""

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            started_at=10.0,
            clock=lambda: 12.5,
            tests_run=("focused-exp2990",),
        )
    )
    manifest_path = tmp_path / artifact["manifest_path"]
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))
    rows = exp.load_manifest(manifest_path)
    transcript_path = tmp_path / artifact["verifier_transcript_paths"][0]
    transcript_rows = exp.load_jsonl(transcript_path)

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["hard_code_stress_set_ready"] is True
    assert artifact["honest_verdict"] == "ready: verifier-backed hard-code stress set validated"
    assert artifact["manifest_path"] == str(exp.DEFAULT_MANIFEST_REL_PATH)
    assert artifact["n_items"] == 24
    assert artifact["all_items_have_tests"] is True
    assert artifact["all_baseline_candidates_fail"] is True
    assert artifact["all_reference_solutions_pass"] is True
    assert artifact["flaky_items"] == []
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["manifest_sha256"] == exp.sha256_text(manifest_path.read_text(encoding="utf-8"))
    assert len(artifact["hard_generation_sources"]) >= 3
    assert "research-references.md:Post-.280 HardTests/HARDTESTGEN" in artifact[
        "hard_generation_sources"
    ]
    assert len(rows) == artifact["n_items"]
    assert len(transcript_rows) == artifact["n_items"]
    assert all(row["tests"] for row in rows)
    assert all(row["baseline_verification"]["passed"] is False for row in rows)
    assert all(row["reference_verification"]["passed"] is True for row in rows)
    assert all(row["baseline_verification"]["failing_test_ids"] for row in rows)
    assert all(row["transcript_sha256"] for row in rows)
    assert transcript_rows[0]["item_id"] == rows[0]["item_id"]
    assert transcript_rows[0]["baseline"]["passed"] is False
    assert transcript_rows[0]["reference"]["passed"] is True
    assert artifact["validation_commands"] == [
        ".venv/bin/pytest tests/python/test_experiment_2990_hard_code_stress_manifest.py -q",
        ".venv/bin/pytest tests/python -q",
        "python scripts/check_spec_coverage.py",
    ]


def test_req_code_2990_execution_distinguishes_baseline_from_reference() -> None:
    """REQ-CODE-2990: verifier execution records failing assertion evidence."""

    item = exp.default_items()[0]
    baseline = exp.run_candidate_tests(item, "baseline_candidate")
    reference = exp.run_candidate_tests(item, "reference_solution")
    missing_key = exp.run_candidate_tests(item, "not_a_candidate")
    broken_syntax = exp.run_candidate_tests(
        {**item, "baseline_candidate": "def clamp_score(:\n"},
        "baseline_candidate",
    )

    assert baseline.passed is False
    assert baseline.failing_test_ids
    assert baseline.error_count >= 1
    assert baseline.candidate_sha256 == exp.sha256_text(item["baseline_candidate"])
    assert reference.passed is True
    assert reference.failing_test_ids == []
    assert missing_key.passed is False
    assert missing_key.errors[0]["error_type"] == "missing_candidate"
    assert broken_syntax.passed is False
    assert broken_syntax.errors[0]["error_type"] == "SyntaxError"


def test_req_code_2990_blocks_invalid_or_nondistinguishing_items(tmp_path: Path) -> None:
    """REQ-CODE-2990: invalid, flaky, or non-hard rows cannot mark the set ready."""

    valid = exp.default_items()[0]
    no_tests = {**valid, "item_id": "no-tests", "tests": []}
    nondistinguishing = {
        **valid,
        "item_id": "nondistinguishing",
        "baseline_candidate": valid["reference_solution"],
    }
    flaky = {
        **valid,
        "item_id": "flaky",
        "tests": [
            {
                "test_id": "SCENARIO-CODE-2990-flaky",
                "code": "assert __import__('random').randint(0, 1) == 0",
            }
        ],
    }

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            manifest_items=(valid, no_tests, nondistinguishing, flaky),
            started_at=0.0,
            clock=lambda: 1.0,
        )
    )
    rows = exp.load_manifest(tmp_path / artifact["manifest_path"])

    assert artifact["hard_code_stress_set_ready"] is False
    assert artifact["honest_verdict"] == "blocked: hard-code stress set failed validation"
    assert artifact["n_items"] == 1
    assert artifact["all_items_have_tests"] is False
    assert artifact["all_baseline_candidates_fail"] is False
    assert artifact["all_reference_solutions_pass"] is False
    assert artifact["flaky_items"] == ["flaky"]
    assert [row["item_id"] for row in rows] == [valid["item_id"]]


def test_req_code_2990_manifest_loader_rejects_malformed_jsonl(tmp_path: Path) -> None:
    """REQ-CODE-2990: manifest evidence must remain inspectable JSONL."""

    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        exp.load_manifest(path)

    assert exp._relative_or_absolute(tmp_path, tmp_path / "inside" / "file.txt") == Path(
        "inside/file.txt"
    )
    assert exp._relative_or_absolute(tmp_path, tmp_path.parent / "outside.txt").is_absolute()
