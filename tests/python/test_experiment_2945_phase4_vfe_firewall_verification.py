"""Tests for Exp 2945 Phase-4 VFE firewall verification.

Spec refs: REQ-REPORT-2945, SCENARIO-REPORT-2945.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from carnot.reporting import phase4_vfe_firewall_verification_2945 as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "files_scanned",
    "firewall_violations",
    "n_violations",
    "firewall_paragraph_draft",
    "cited_upstream_artifacts",
    "duration_s",
}


def _write(root: Path, rel_path: Path | str, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: object) -> None:
    _write(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _seed_scan_tree(root: Path) -> None:
    _write(
        root,
        "docs/arxiv-paper/main.tex",
        "Phase-4 active inference is reported for an RTX 3090 sampler.\n"
        "The KV260 hardware paragraph then cites exp2748 as deployment evidence.\n",
    )
    _write(
        root,
        "docs/arxiv-paper/sections/hardware-validation-v1.tex",
        "A variational free energy bound appears here.\n"
        "The nearby synchronous Glauber FPGA sentence creates a firewall hit.\n",
    )
    for experiment_id in range(1000, 1011):
        claim = "clean capstone"
        if experiment_id == 1010:
            claim = "The FEP aggregator supports KV260 hardware deployment."
        if experiment_id == 1000:
            claim = "exp2753 supports FPGA deployment but is too old for the ten latest capstones."
        _write_json(
            root,
            f"results/experiment_{experiment_id}_capstone_v{experiment_id}.json",
            {"paper_v6_safe_claims": [claim]},
        )


def test_req_report_2945_spec_anchor_exists() -> None:
    """REQ-REPORT-2945: OpenSpec declares the firewall verifier first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-2945" in spec
    assert "SCENARIO-REPORT-2945" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_req_report_2945_phase4_regexes_cover_mandatory_citations() -> None:
    """REQ-REPORT-2945: every mandatory Phase-4 citation has a regex."""

    samples = {
        "exp2550": "exp2550",
        "exp2748": "exp2748",
        "exp2753": "exp2753",
        "exp2766": "exp2766",
        "Phase-4 active inference": "Phase-4 active inference",
        "variational free energy": "variational free energy",
        "FEP factor graph": "FEP factor graph",
        "FEP aggregator": "FEP aggregator",
    }
    labels = {pattern.label for pattern in mod.PHASE4_PATTERNS}

    assert labels == set(samples)
    for label, phrase in samples.items():
        assert any(
            re.search(pattern.regex, phrase, flags=pattern.flags)
            for pattern in mod.PHASE4_PATTERNS
            if pattern.label == label
        ), label


def test_scenario_report_2945_records_only_hardware_context_hits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2945: Phase-4 citations become violations only near hardware."""

    text = (
        "The variational free energy estimate is continuous-sampler only.\n"
        "KV260 synchronous Glauber cannot inherit that bound.\n"
        "This separator has no deployment claim.\n"
        "Another neutral sentence keeps contexts distinct.\n"
        "The FEP factor graph is discussed for RTX 3090 sampling only.\n"
    )

    hits = mod.scan_text(text, Path("docs/arxiv-paper/main.tex"), context_radius=1)

    assert len(hits) == 1
    assert hits[0]["line"] == 1
    assert hits[0]["phase_4_citation"] == "variational free energy"
    assert "KV260 synchronous Glauber" in hits[0]["hardware_context_snippet"]
    assert "FEP factor graph" not in {
        hit["phase_4_citation"] for hit in hits
    }


def test_scenario_report_2945_builds_required_artifact_without_doc_edits(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2945: artifact records violations and leaves inputs untouched."""

    _seed_scan_tree(tmp_path)
    original_main = (tmp_path / "docs/arxiv-paper/main.tex").read_text(
        encoding="utf-8"
    )

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=13.25)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["n_violations"] == 4
    assert all(
        set(hit)
        == {"file", "line", "phase_4_citation", "hardware_context_snippet"}
        for hit in artifact["firewall_violations"]
    )
    assert "results/experiment_1010_capstone_v1010.json" in artifact["files_scanned"]
    assert (
        "results/experiment_1000_capstone_v1000.json"
        not in artifact["files_scanned"]
    )
    assert "docs/arxiv-paper/sections/hardware-validation-v1.tex" in artifact[
        "files_scanned"
    ]
    assert (
        tmp_path / "docs/arxiv-paper/main.tex"
    ).read_text(encoding="utf-8") == original_main

    paragraph = artifact["firewall_paragraph_draft"]
    assert paragraph["principle"].startswith("Operator-integrable LaTeX snippet")
    assert "RTX 3090 continuous-sampler deployment" in paragraph["latex"]
    assert "KV260" in paragraph["latex"]
    assert "synchronous Glauber" in paragraph["latex"]

    cited = {item["path"]: item for item in artifact["cited_upstream_artifacts"]}
    assert cited["docs/arxiv-paper/main.tex"]["sha256"] is not None
    assert cited["results/experiment_1010_capstone_v1010.json"]["sha256"] is not None


def test_req_report_2945_write_artifact_persists_schema_json(tmp_path: Path) -> None:
    """REQ-REPORT-2945: write_artifact emits the required stable deliverable."""

    _seed_scan_tree(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert REQUIRED_FIELDS <= saved.keys()
    assert saved["n_violations"] == 4


def test_req_report_2945_helper_edges_are_deterministic(tmp_path: Path) -> None:
    """REQ-REPORT-2945: missing files and capstone ordering fail closed."""

    long_text = "exp2550 hardware " + ("x" * 600)
    long_hits = mod.scan_text(long_text, Path("docs/arxiv-paper/main.tex"), context_radius=0)

    assert mod.select_latex_paths(tmp_path) == []
    assert mod.scan_paths(tmp_path, [Path("missing.tex")]) == []
    assert mod.sha256_file(tmp_path / "missing.tex") is None
    assert mod._experiment_number(Path("capstone_without_id.json")) == -1
    assert mod._honest_verdict(0) == "complete: phase4_vfe_firewall_no_violations"
    assert long_hits[0]["hardware_context_snippet"].endswith("...")
