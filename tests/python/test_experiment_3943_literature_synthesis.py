"""Tests for Exp 3943 verifier-efficiency literature synthesis.

Spec refs: REQ-REPORT-3943, SCENARIO-REPORT-3943,
SCENARIO-REPORT-3943-BLOCKED-REFERENCES.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import literature_synthesis_3943 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _seed_repo(root: Path, *, with_sources: bool = True, with_landed_results: bool = True) -> None:
    if with_sources:
        _write(
            root / "research-references.md",
            "\n".join(
                [
                    "## Existing verification-efficiency references",
                    "ProcessBench (Qwen, arXiv:2412.06559)",
                    "arXiv:2504.16828 (ThinkPRM)",
                    "arXiv:2408.15240 (Generative Verifiers / GenRM)",
                    "arXiv:2510.14913 (Budget-aware Discriminative Verification)",
                    "arXiv:2603.24621 (ARC-AGI-3)",
                    "arXiv:2605.05138 (Executable World Models for ARC-AGI-3)",
                ]
            )
            + "\n",
        )
        _write(root / "research-studying.md", "# Research Studying\n\nExisting queue.\n")
    _write(root / "README.md", "public readme before\n")
    _write(root / "docs" / "index.html", "<main>public index before</main>\n")
    _write(root / "docs" / "blog" / "post.md", "blog before\n")
    _write(root / "ops" / "changelog.md", "changelog before\n")
    _write(root / "ops" / "status.md", "status before\n")
    _write(root / "_bmad" / "traceability.md", "trace before\n")
    _write(root / "scripts" / "research_conductor.py", "# conductor before\n")
    if with_landed_results:
        _write_json(
            root / "results" / "experiment_3936_valid_efficiency_head_to_head.json",
            {
                "experiment": 3936,
                "honest_verdict": (
                    "complete: valid_efficiency_PARITY_PARETO_energy_12.4x_cheaper"
                ),
                "parity_or_pareto_landed": True,
                "energy_cheaper_than_competent_judge_x": 12.4,
                "energy_auroc": 0.842,
                "competent_judge_auroc": 0.839,
            },
        )
        _write_json(
            root / "results" / "experiment_3937_non_degenerate_cascade_router.json",
            {
                "experiment": 3937,
                "honest_verdict": "complete: non_degenerate_cascade_landed",
                "non_degenerate_cascade": True,
                "cascade_compute_saved_pct": 61.0,
            },
        )
        _write_json(
            root / "results" / "experiment_3938_moat_replication.json",
            {
                "experiment": 3938,
                "honest_verdict": "complete: independent_corpus_moat_replicated",
                "moat_replicates": True,
                "independent_corpus_moat": True,
            },
        )
        _write_json(
            root / "results" / "experiment_3939_arc_agi3_step2.json",
            {
                "experiment": 3939,
                "honest_verdict": "complete: arc_agi3_step2_router_still_helps",
                "action_efficiency_ratio": 2.18,
                "is_real_benchmark": False,
            },
        )
        _write_json(
            root / "results" / "experiment_3942_cross_domain_map.json",
            {
                "experiment": 3942,
                "honest_verdict": "complete: cross_domain_map_ready",
                "cross_domain_map_ready": True,
            },
        )


def test_req_report_3943_spec_anchor_exists() -> None:
    """REQ-REPORT-3943: OpenSpec declares the 3943 synthesis contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3943" in spec
    assert "SCENARIO-REPORT-3943" in spec
    assert "SCENARIO-REPORT-3943-BLOCKED-REFERENCES" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "new_references_added" in spec
    assert "public_docs_untouched" in spec


def test_scenario_report_3943_writes_note_ledger_and_terminal_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3943: landed proof is positioned without public-doc edits."""

    _seed_repo(tmp_path)
    before = {
        "references": (tmp_path / "research-references.md").read_text(encoding="utf-8"),
        "readme": (tmp_path / "README.md").read_text(encoding="utf-8"),
        "index": (tmp_path / "docs" / "index.html").read_text(encoding="utf-8"),
        "blog": (tmp_path / "docs" / "blog" / "post.md").read_text(encoding="utf-8"),
        "changelog": (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8"),
        "status": (tmp_path / "ops" / "status.md").read_text(encoding="utf-8"),
        "trace": (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8"),
        "conductor": (tmp_path / "scripts" / "research_conductor.py").read_text(
            encoding="utf-8"
        ),
    }

    output = mod.run(tmp_path, started_s=10.0, now_s=12.25)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    note = (tmp_path / artifact["synthesis_note_path"]).read_text(encoding="utf-8")
    studying = (tmp_path / "research-studying.md").read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == (
        "complete: literature_synthesis_positioned_0_new_refs_public_docs_untouched"
    )
    assert artifact["synthesis_note_path"] == mod.SYNTHESIS_NOTE_REL_PATH.as_posix()
    assert artifact["public_docs_untouched"] is True
    assert artifact["preconditions_checked"] is True
    assert artifact["new_references_added"] == 0
    assert artifact["duration_s"] == 2.25
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert "parity/Pareto at 12.400x lower judge cost" in (
        artifact["landscape_position_summary"]
    )
    assert "ProcessBench full-benchmark head-to-head" in (
        artifact["next_highest_leverage_experiments"]
    )
    assert "ARC-AGI-3 real agentic run" in artifact["next_highest_leverage_experiments"]
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)

    for term in (
        "ProcessBench",
        "ThinkPRM",
        "GenRM",
        "Budget-aware Discriminative Verification",
        "ARC-AGI-3",
        "Executable World Models",
    ):
        assert term in note
    assert "12.400x" in note
    assert "non-degenerate cascade" in note
    assert "independent-corpus moat" in note
    assert "does not claim an official ARC-AGI-3 score" in note
    assert mod.STUDYING_MARKER in studying
    assert "Score: 5 x 5 x 5 x 4 = 500" in studying
    assert "real ARC-AGI-3 agentic run" in studying

    assert (tmp_path / "research-references.md").read_text(encoding="utf-8") == before["references"]
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == before["readme"]
    assert (tmp_path / "docs" / "index.html").read_text(encoding="utf-8") == before["index"]
    assert (tmp_path / "docs" / "blog" / "post.md").read_text(encoding="utf-8") == before["blog"]
    assert (tmp_path / "ops" / "changelog.md").read_text(encoding="utf-8") == before["changelog"]
    assert (tmp_path / "ops" / "status.md").read_text(encoding="utf-8") == before["status"]
    assert (tmp_path / "_bmad" / "traceability.md").read_text(encoding="utf-8") == before["trace"]
    assert (tmp_path / "scripts" / "research_conductor.py").read_text(encoding="utf-8") == (
        before["conductor"]
    )


def test_scenario_report_3943_is_idempotent_and_records_source_gaps(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3943: repeat runs do not duplicate the scored candidate."""

    _seed_repo(tmp_path, with_landed_results=False)

    first = mod.cli_main(["--repo-root", str(tmp_path), "--started-s", "1.0", "--now-s", "2.0"])
    second = mod.cli_main(["--repo-root", str(tmp_path), "--started-s", "2.0", "--now-s", "3.0"])
    artifact = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    studying = (tmp_path / "research-studying.md").read_text(encoding="utf-8")
    note = (tmp_path / artifact["synthesis_note_path"]).read_text(encoding="utf-8")

    assert first == 0
    assert second == 0
    assert studying.count(mod.STUDYING_MARKER) == 1
    assert artifact["duration_s"] == 1.0
    assert artifact["new_references_added"] == 0
    assert artifact["source_gaps"]
    assert "requested .364 source artifacts were absent" in note
    assert "12.400x" not in note


def test_scenario_report_3943_blocked_references_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3943-BLOCKED-REFERENCES: missing ledgers block before edits."""

    _seed_repo(tmp_path, with_sources=False)
    output = mod.run(tmp_path, started_s=4.0, now_s=4.25)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_references_missing"
    assert artifact["synthesis_note_path"] == ""
    assert artifact["landscape_position_summary"] == ""
    assert artifact["next_highest_leverage_experiments"] == ""
    assert artifact["new_references_added"] == 0
    assert artifact["preconditions_checked"] is True
    assert artifact["public_docs_untouched"] is True
    assert not (tmp_path / mod.SYNTHESIS_NOTE_REL_PATH).exists()


def test_req_report_3943_helpers_and_schema_guards(tmp_path: Path) -> None:
    """REQ-REPORT-3943: helper and validation branches fail closed."""

    _seed_repo(tmp_path)
    artifact = json.loads(mod.run(tmp_path, started_s=1.0, now_s=1.1).read_text(encoding="utf-8"))

    assert mod.read_json_artifact(tmp_path / "results" / "missing.json") == {}
    assert mod.best_artifact_for_patterns(tmp_path, ("results/missing_*.json",)) == {}
    sources = mod.load_source_artifacts(tmp_path)
    assert sources["valid_efficiency"]["present"] is True
    assert "12.400x lower judge cost" in mod.landscape_position_summary(sources)
    qualitative_sources = dict(sources)
    qualitative_sources["valid_efficiency"] = {
        "present": True,
        "requested_present": True,
        "path": "results/experiment_3936_valid_efficiency_head_to_head.json",
        "payload": {
            "honest_verdict": "complete: valid_efficiency_PARITY_PARETO",
            "parity_or_pareto_landed": True,
        },
    }
    assert "parity/Pareto against the competent judge" in mod.landscape_position_summary(
        qualitative_sources
    )

    bad_cases = [
        (lambda data: data.pop("synthesis_note_path"), "missing required fields"),
        (lambda data: data.update({"synthesis_note_path": 7}), "synthesis_note_path"),
        (lambda data: data.update({"landscape_position_summary": 7}), "landscape_position"),
        (lambda data: data.update({"next_highest_leverage_experiments": []}), "next_highest"),
        (lambda data: data.update({"new_references_added": "0"}), "new_references_added"),
        (lambda data: data.update({"public_docs_untouched": "yes"}), "public_docs_untouched"),
        (lambda data: data.update({"preconditions_checked": "yes"}), "preconditions_checked"),
        (lambda data: data.update({"duration_s": "0.1"}), "duration_s"),
        (lambda data: data.update({"inference_substrate": "other"}), "inference_substrate"),
        (lambda data: data.update({"landscape_position_summary": "GGUF marker"}), "GGUF/CUDA"),
        (lambda data: data.update({"honest_verdict": "complete: wrong"}), "falsification"),
        (lambda data: data.update({"synthesis_note_path": ""}), "note path"),
        (lambda data: data.update({"public_docs_untouched": False}), "untouched public docs"),
        (lambda data: data.update({"honest_verdict": "partial"}), "honest_verdict"),
    ]
    for mutate, message in bad_cases:
        candidate = dict(artifact)
        mutate(candidate)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(candidate)
