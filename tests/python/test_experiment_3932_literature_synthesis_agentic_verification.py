"""Tests for Exp 3932 literature synthesis of agentic verification efficiency.

Spec refs: REQ-REPORT-3932, SCENARIO-REPORT-3932,
SCENARIO-REPORT-3932-BLOCKED-REFERENCES.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import literature_synthesis_agentic_verification_3932 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _seed_repo(root: Path, *, with_sources: bool = True) -> None:
    if with_sources:
        _write(
            root / "research-references.md",
            "\n".join(
                [
                    "## 2026-06-07 (.363 planning sweep - making the efficiency win CREDIBLE)",
                    "ProcessBench (Qwen, arXiv:2412.06559)",
                    "arXiv:2504.16828 (ThinkPRM)",
                    "arXiv:2408.15240 (Generative Verifiers / GenRM)",
                    "arXiv:2510.14913 (Budget-aware Discriminative Verification)",
                    "arXiv:2603.24621 (ARC-AGI-3)",
                    "arXiv:2502.11250 (Uncertainty-Aware Step-wise Verification)",
                ]
            )
            + "\n",
        )
        _write(root / "research-studying.md", "# Research Studying\n\nExisting queue.\n")
    _write(
        root / "ops" / "north-star.md",
        "## 5. STRATEGIC REFRAME\n"
        "The verifier earns its place if it is equally effective as the LM at lower cost.\n"
        "ARC-AGI-3 is the agentic proof venue after the offline verifier proof.\n",
    )
    _write(root / "README.md", "public readme before\n")
    _write(root / "docs" / "index.html", "<main>public index before</main>\n")
    _write(root / "docs" / "blog" / "post.md", "blog before\n")
    _write(root / "ops" / "changelog.md", "changelog before\n")
    _write(root / "ops" / "status.md", "status before\n")
    _write(root / "_bmad" / "traceability.md", "trace before\n")
    _write(root / "scripts" / "research_conductor.py", "# conductor before\n")
    _write_json(
        root / "results" / "experiment_3926_valid_efficiency_head_to_head.json",
        {
            "experiment": 3926,
            "honest_verdict": "blocked_upstream_competent_judge_not_ready",
            "flagged_adversarial": True,
            "judge_positive_control_passed": False,
            "energy_auroc": None,
            "llm_judge_auroc": None,
        },
    )
    _write_json(
        root / "results" / "experiment_3928_moat_scissor_replication.json",
        {
            "experiment": 3928,
            "honest_verdict": "blocked_all_gguf_inference_failed",
            "flagged_adversarial": True,
            "moat_replicates": False,
        },
    )
    _write_json(
        root / "results" / "experiment_3929_arc_agi3_action_efficiency.json",
        {
            "experiment": 3929,
            "honest_verdict": (
                "complete: arc_agi3_verifier_router_HELPS_ratio1.959_"
                "ci1.742-2.194_synthetic_first_agentic_step_real_benchmark_reachabletrue"
            ),
            "action_efficiency_ratio": 1.9591836734693875,
            "action_efficiency_ci95": {"low": 1.7420435510887773, "high": 2.193877551020408},
            "is_synthetic_not_real_benchmark": True,
            "real_benchmark_reachable": True,
        },
    )


def test_req_report_3932_spec_anchor_exists() -> None:
    """REQ-REPORT-3932: OpenSpec declares the literature synthesis contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3932" in spec
    assert "SCENARIO-REPORT-3932" in spec
    assert "SCENARIO-REPORT-3932-BLOCKED-REFERENCES" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "new_references_added" in spec
    assert "public_docs_untouched" in spec


def test_scenario_report_3932_writes_note_ledger_and_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3932: local synthesis writes the note and scored candidate."""

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

    output = mod.run(tmp_path, started_s=10.0, now_s=11.5)
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
    assert artifact["duration_s"] == 1.5
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert "cheap discriminative energy verifier" in artifact["landscape_position_summary"]
    assert "blocked locally" in artifact["landscape_position_summary"]
    assert "ProcessBench full-benchmark head-to-head" in (
        artifact["next_highest_leverage_experiments"]
    )
    assert "ARC-AGI-3 real-benchmark agentic run" in (
        artifact["next_highest_leverage_experiments"]
    )
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)

    for term in ("ProcessBench", "ThinkPRM", "GenRM", "Budget-aware", "ARC-AGI-3"):
        assert term in note
    assert "Exp 3926 is blocked/flagged" in note
    assert "1.959" in note
    assert "official ARC-AGI-3 score" in note
    assert mod.STUDYING_MARKER in studying
    assert "Score: 5 x 5 x 4 x 4 = 400" in studying
    assert "ProcessBench full-benchmark head-to-head" in studying

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


def test_scenario_report_3932_is_idempotent_and_cli_writes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3932: repeat runs do not duplicate ledger sections."""

    _seed_repo(tmp_path)

    first = mod.cli_main(["--repo-root", str(tmp_path), "--started-s", "1.0", "--now-s", "2.0"])
    second = mod.cli_main(["--repo-root", str(tmp_path), "--started-s", "2.0", "--now-s", "3.0"])
    artifact = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    studying = (tmp_path / "research-studying.md").read_text(encoding="utf-8")

    assert first == 0
    assert second == 0
    assert studying.count(mod.STUDYING_MARKER) == 1
    assert artifact["duration_s"] == 1.0
    assert artifact["new_references_added"] == 0


def test_scenario_report_3932_blocked_references_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3932-BLOCKED-REFERENCES: missing ledgers block before edits."""

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


def test_req_report_3932_helpers_and_schema_guards(tmp_path: Path) -> None:
    """REQ-REPORT-3932: helper and validation branches fail closed."""

    _seed_repo(tmp_path)
    artifact = json.loads(mod.run(tmp_path, started_s=1.0, now_s=1.1).read_text(encoding="utf-8"))

    assert mod.read_json_artifact(tmp_path, Path("results/missing.json")) == {}
    complete_summary = mod.landscape_position_summary(
        {
            "exp3926": {"honest_verdict": "complete: parity", "flagged_adversarial": False},
            "exp3929": {"action_efficiency_ratio": 2.0},
        }
    )
    assert "complete locally" in complete_summary

    bad_cases = [
        (lambda data: data.pop("synthesis_note_path"), "missing required fields"),
        (lambda data: data.update({"synthesis_note_path": 7}), "synthesis_note_path"),
        (lambda data: data.update({"landscape_position_summary": 7}), "landscape_position"),
        (lambda data: data.update({"next_highest_leverage_experiments": []}), "next_highest"),
        (lambda data: data.update({"new_references_added": "0"}), "new_references_added"),
        (lambda data: data.update({"public_docs_untouched": "yes"}), "public_docs_untouched"),
        (lambda data: data.update({"inference_substrate": "other"}), "inference_substrate"),
        (lambda data: data.update({"landscape_position_summary": "GGUF marker"}), "GGUF/CUDA"),
        (lambda data: data.update({"honest_verdict": "complete: wrong"}), "falsification"),
        (lambda data: data.update({"synthesis_note_path": ""}), "note path"),
        (lambda data: data.update({"honest_verdict": "partial"}), "honest_verdict"),
    ]
    for mutate, message in bad_cases:
        candidate = dict(artifact)
        mutate(candidate)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(candidate)
