"""Tests for Exp 2944 paper-v6 narrowing audit.

Spec refs: REQ-REPORT-2944, SCENARIO-REPORT-2944.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_narrowing_audit_2944 as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "files_scanned",
    "per_file_hits",
    "n_total_hits",
    "n_operator_curated_hits_left_for_operator",
    "n_autonomous_artifact_hits_auto_fixed",
    "suggested_lint_script_path",
    "cited_upstream_artifacts",
    "duration_s",
}


def _write(root: Path, rel_path: Path | str, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: object) -> None:
    _write(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_minimal_scan_tree(root: Path) -> None:
    _write(
        root,
        "CLAUDE.md",
        "## Paper-v6 Narrowing Discipline\n"
        "| # | Retracted claim | Forbidden phrasing |\n"
        "| **#2** | KV260 samples reach Boltzmann thermalization | "
        '"thermalization," "equilibrium samples," "Boltzmann-distributed energies" |\n',
    )
    _write(root, "docs/arxiv-paper/main.tex", "Hardware sovereignty remains unresolved.\n")
    _write(
        root,
        "docs/technical-report.md",
        "The verifier ensemble generalizes without a corpus boundary.\n",
    )
    _write(root, "docs/technical-report.html", "<html><body>clean</body></html>\n")
    _write(root, "docs/index.html", "<html><body>clean</body></html>\n")
    for experiment_id in range(1000, 1011):
        phrase = "clean"
        if experiment_id == 1010:
            phrase = (
                "KV260 hardware speedup was observed. "
                "The five-paper_ready streak proves paper readiness."
            )
        if experiment_id == 1000:
            phrase = "FPGA acceleration over CPU should not be scanned here."
        _write_json(
            root,
            f"results/experiment_{experiment_id}_capstone_v{experiment_id}.json",
            {"paper_v6_safe_claims": [phrase]},
        )


def test_req_report_2944_spec_anchor_exists() -> None:
    """REQ-REPORT-2944: OpenSpec declares the audit contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-2944" in spec
    assert "SCENARIO-REPORT-2944" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_req_report_2944_forbidden_regexes_cover_all_retracted_claims() -> None:
    """REQ-REPORT-2944: seven CLAUDE.md retractions become regex rules."""

    samples = {
        "#2": "Boltzmann-distributed energies",
        "#3": "Carnot's verifier ensemble runs faster on KV260",
        "#6": "exp2748 supports FPGA deployment",
        "#7": "Extropic Z1 is a future production target",
        "#8": "the verifier ensemble works on novel corpora",
        "#9": "Hardware sovereignty",
        "#10": ".271/.272/.273/.274/.275 paper_ready=true",
    }
    claim_ids = {pattern.retracted_claim_id for pattern in mod.FORBIDDEN_PATTERNS}

    assert claim_ids == set(samples)
    for claim_id, phrase in samples.items():
        assert any(
            re.search(pattern.regex, phrase, flags=pattern.flags)
            for pattern in mod.FORBIDDEN_PATTERNS
            if pattern.retracted_claim_id == claim_id
        ), claim_id


def test_scenario_report_2944_records_hits_and_rewrites_only_capstones(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2944: docs stay untouched while capstone strings are narrowed."""

    _write_minimal_scan_tree(tmp_path)
    original_doc = (tmp_path / "docs/arxiv-paper/main.tex").read_text(encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["n_total_hits"] == 4
    assert artifact["n_operator_curated_hits_left_for_operator"] == 2
    assert artifact["n_autonomous_artifact_hits_auto_fixed"] == 2
    assert artifact["suggested_lint_script_path"] == {
        "path": "scripts/paper_v6_narrowing_lint.py",
        "principle": "Path to a proposed pre-commit hook that would catch future violations. Just suggest; do not commit the hook.",
    }

    assert (tmp_path / "docs/arxiv-paper/main.tex").read_text(encoding="utf-8") == original_doc
    fixed_capstone = (tmp_path / "results/experiment_1010_capstone_v1010.json").read_text(
        encoding="utf-8"
    )
    excluded_capstone = (tmp_path / "results/experiment_1000_capstone_v1000.json").read_text(
        encoding="utf-8"
    )
    assert "KV260 hardware speedup" not in fixed_capstone
    assert "five-paper_ready streak" not in fixed_capstone
    assert "POC functional simulator anchoring future high-N deployment" in fixed_capstone
    assert "CI-loop discipline metric" in fixed_capstone
    assert "FPGA acceleration over CPU" in excluded_capstone

    files_scanned = artifact["files_scanned"]
    assert "results/experiment_1010_capstone_v1010.json" in files_scanned
    assert "results/experiment_1000_capstone_v1000.json" not in files_scanned
    assert len([path for path in files_scanned if "capstone" in path]) == 10
    assert all(
        set(hit) == {"file", "line", "matched_phrase", "retracted_claim_id", "suggested_fix"}
        for hit in artifact["per_file_hits"]
    )
    assert {hit["retracted_claim_id"] for hit in artifact["per_file_hits"]} == {
        "#3",
        "#9",
        "#10",
        "#8",
    }
    cited = {item["path"]: item for item in artifact["cited_upstream_artifacts"]}
    assert "CLAUDE.md" in cited
    assert "results/experiment_1010_capstone_v1010.json" in cited


def test_req_report_2944_write_artifact_persists_required_json(tmp_path: Path) -> None:
    """REQ-REPORT-2944: write_artifact emits the requested stable deliverable."""

    _write_minimal_scan_tree(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert REQUIRED_FIELDS <= saved.keys()
    assert saved["n_total_hits"] == 4
    assert saved["n_autonomous_artifact_hits_auto_fixed"] == 2


def test_req_report_2944_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-2944: missing and malformed inputs fail closed without rewrites."""

    malformed = tmp_path / "results" / "experiment_9999_capstone_bad.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{not json", encoding="utf-8")
    hit = {
        "file": "results/experiment_9999_capstone_bad.json",
        "line": 1,
        "matched_phrase": "Hardware sovereignty",
        "retracted_claim_id": "#9",
        "suggested_fix": 'Replace with "local edge deployability".',
    }

    assert mod.scan_paths(tmp_path, [Path("missing.txt")]) == []
    assert mod.auto_fix_capstone_hits(tmp_path, [hit]) == set()
    assert malformed.read_text(encoding="utf-8") == "{not json"
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod._narrow_json_strings(3) == (3, False)
    assert mod._experiment_number(Path("capstone_without_id.json")) == -1
    assert mod._honest_verdict(0, 0, 0) == "complete: paper_v6_narrowing_audit_no_matches"
