"""Tests for REQ-REPORT-4162 / SCENARIO-REPORT-4162."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4162_sota_ingestion_verifier_moat_guidance as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/sota-ingestion-verifier-moat-guidance-2026-06-13.md"
)
ARTIFACT_PATH = Path(
    "results/experiment_4162_sota_ingestion_verifier_moat_guidance.json"
)
STUDYING_PATH = Path("research-studying.md")
REFERENCES_PATH = Path("research-references.md")


def _valid_methods() -> list[dict[str, str]]:
    return [
        {
            "name": "ARBITER reasoning-basin verifier-moat anchor",
            "arxiv_id_or_url": "2605.26172",
            "url": "https://arxiv.org/abs/2605.26172",
            "carnot_moat_implication": (
                "Majority vote can choose the largest wrong basin, so Carnot should "
                "measure external verifier recovery as an additive vote-orthogonal "
                "rerank signal, not as a vote replacement."
            ),
            "efficiency_implication": (
                "Keep the cheap verifier between fixed vote and LLM-judge: recover "
                "oracle headroom only when it improves accuracy per unit compute."
            ),
            "diffusiongemma_guidance_implication": (
                "Only guide DiffusionGemma after the verifier distinguishes correct "
                "minority basins from stable wrong basins on the current stack."
            ),
        },
        {
            "name": "ThinkPRM data-efficient process verifier",
            "arxiv_id_or_url": "2504.16828",
            "url": "https://arxiv.org/abs/2504.16828",
            "carnot_moat_implication": (
                "Process verification can beat self-consistency when it checks the "
                "reasoning path; Carnot should compare verifier-plus-vote selection "
                "against vote and judge baselines."
            ),
            "efficiency_implication": (
                "ThinkPRM sets the LLM-judge comparison bar: verifier compute must "
                "scale better than asking a large judge to rescore every candidate."
            ),
            "diffusiongemma_guidance_implication": (
                "Use process-style scores as intermediate guidance signals rather "
                "than waiting until the denoised candidate is complete."
            ),
        },
        {
            "name": "Optimal LLM+PRM aggregation",
            "arxiv_id_or_url": "2510.13918",
            "url": "https://arxiv.org/abs/2510.13918",
            "carnot_moat_implication": (
                "The verifier should be calibrated as a weighted aggregation term "
                "with vote evidence; replacing the vote can throw away useful LLM "
                "prior information."
            ),
            "efficiency_implication": (
                "Precompute aggregation weights so the verifier improves test-time "
                "scaling without multiplying candidate-generation cost."
            ),
            "diffusiongemma_guidance_implication": (
                "Expose guidance weights as an ablation knob for mixing base "
                "denoising confidence with Carnot verifier energy."
            ),
        },
        {
            "name": "RLV unified reasoner-verifier value head",
            "arxiv_id_or_url": "2505.04842",
            "url": "https://arxiv.org/abs/2505.04842",
            "carnot_moat_implication": (
                "Training a verifier/value capability alongside reasoning supports "
                "an external-value rerank arm, but the moat claim still requires a "
                "vote-plus-verifier head-to-head."
            ),
            "efficiency_implication": (
                "The next efficiency gate should compare a cheap verifier/value head "
                "against LLM-judge rescoring under matched parallel sampling."
            ),
            "diffusiongemma_guidance_implication": (
                "A learned value head is a plausible reward source for guidance, but "
                "it must be checked against executable verifier labels before use."
            ),
        },
        {
            "name": "EntRGi entropy-aware reward guidance",
            "arxiv_id_or_url": "2602.05000",
            "url": "https://arxiv.org/abs/2602.05000",
            "carnot_moat_implication": (
                "EntRGi is not moat evidence by itself; it becomes relevant only "
                "after Carnot proves the external reward/verifier is discriminative."
            ),
            "efficiency_implication": (
                "Guidance can spend verifier calls during denoising, so the .386 "
                "gate must report reward-call cost versus post-hoc judge rescoring."
            ),
            "diffusiongemma_guidance_implication": (
                "Use entropy-aware interpolation between soft token relaxations and "
                "hard tokens as the template for Carnot energy over DiffusionGemma."
            ),
        },
        {
            "name": "Executable World Models for ARC-AGI-3",
            "arxiv_id_or_url": "2605.05138",
            "url": "https://arxiv.org/abs/2605.05138",
            "carnot_moat_implication": (
                "Executable world models make verifier-grounded transitions the "
                "selection primitive; the moat is action recovery, not just answer "
                "selection."
            ),
            "efficiency_implication": (
                "Use RHAE/action efficiency as the cost axis when verifier-pruned "
                "planning replaces brute-force exploration."
            ),
            "diffusiongemma_guidance_implication": (
                "Treat generated world-model edits as candidates that can be guided "
                "or pruned by executable transition energy before acting."
            ),
        },
        {
            "name": "ARC-AGI-3 technical report",
            "arxiv_id_or_url": "2603.24621",
            "url": "https://arxiv.org/abs/2603.24621",
            "carnot_moat_implication": (
                "ARC-AGI-3 makes adaptive efficiency the benchmark target, so the "
                "verifier moat must improve actions-to-progress under real rules."
            ),
            "efficiency_implication": (
                "Report human-action-normalized efficiency and avoid claims that "
                "only improve raw solve count by spending more actions."
            ),
            "diffusiongemma_guidance_implication": (
                "Guided generation should target compact executable hypotheses and "
                "plans, not only fluent natural-language reasoning traces."
            ),
        },
    ]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v386="entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386",
    )


def test_req_report_4162_spec_anchor_exists() -> None:
    """REQ-REPORT-4162: OpenSpec declares verifier-moat guidance ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4162" in spec
    assert "SCENARIO-REPORT-4162" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert "research-references.md" in spec
    assert "flagged_for_v386" in spec
    for arxiv_id in mod.VERIFIED_ARXIV_IDS:
        assert f"arXiv:{arxiv_id}" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4162() -> None:
    """REQ-REPORT-4162: artifact exposes the required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": "complete: sota_ingestion_verifier_moat_guidance_mapped",
        "methods_mapped": _valid_methods(),
        "flagged_for_v386": (
            "entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386"
        ),
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method/source MUST carry a real arXiv ID/URL (verified); "
                "an ingestion note without verifiable citations is treated as fabrication."
            ),
            "flagged_for_v386": (
                "Closes the discover->ingest->plan loop: names the strongest method "
                "for the next planner."
            ),
        },
    }


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:4]}, "at least five"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "fake",
                        "arxiv_id_or_url": "9999.99999",
                        "url": "https://arxiv.org/abs/9999.99999",
                        "carnot_moat_implication": "fake",
                        "efficiency_implication": "fake",
                        "diffusiongemma_guidance_implication": "fake",
                    }
                ]
                + _valid_methods()[1:]
            },
            "verified arxiv ID",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"url": "https://example.com/2605.26172"}
                ]
                + _valid_methods()[1:]
            },
            "url",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [{"name": "ARBITER", "arxiv_id_or_url": "2605.26172"}] + _valid_methods()[1:]},
            "exactly",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate source",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"efficiency_implication": ""}
                ]
                + _valid_methods()[1:]
            },
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v386": ""}, "flagged_for_v386"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4162(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4162: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_bad_method_rows() -> None:
    """SCENARIO-REPORT-4162: artifact and method fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["inference_substrate"] = "manual_ingestion"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {
        "methods_mapped": ["not-a-dict"] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)


def test_validate_markdown_note_checks_scenario_report_4162_sections() -> None:
    """SCENARIO-REPORT-4162: note maps each source to the three required axes."""

    note = """
    ## Fresh-pass provenance
    reliable-channel sweeps and WebFetch verified the sources.
    ## SOTA -> experiment mapping
    ## ARBITER reasoning-basin verifier-moat anchor
    arXiv:2605.26172. Carnot moat implication. Efficiency implication.
    DiffusionGemma guidance implication.
    ## ThinkPRM data-efficient process verifier
    arXiv:2504.16828. Carnot moat implication. Efficiency implication.
    DiffusionGemma guidance implication.
    ## Optimal LLM+PRM aggregation
    arXiv:2510.13918. Carnot moat implication. Efficiency implication.
    DiffusionGemma guidance implication.
    ## RLV unified reasoner-verifier value head
    arXiv:2505.04842. Carnot moat implication. Efficiency implication.
    DiffusionGemma guidance implication.
    ## EntRGi entropy-aware reward guidance
    arXiv:2602.05000. Carnot moat implication. Efficiency implication.
    DiffusionGemma guidance implication.
    ## Executable World Models for ARC-AGI-3
    arXiv:2605.05138. Carnot moat implication. Efficiency implication.
    DiffusionGemma guidance implication.
    ## ARC-AGI-3 technical report
    arXiv:2603.24621. Carnot moat implication. Efficiency implication.
    DiffusionGemma guidance implication.
    ## Flagged for .386
    entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_sources_or_flag() -> None:
    """SCENARIO-REPORT-4162: note must cite each source and close the loop."""

    with pytest.raises(ValueError, match="Flagged for .386"):
        mod.validate_markdown_note("## Fresh-pass provenance\narXiv:2605.26172\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("arXiv:2602.05000", "EntRGi")
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4162(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4162: writer emits note, artifact, studying, and references."""

    note_path = tmp_path / "note.md"
    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    references_path = tmp_path / "research-references.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")
    references_path.write_text("# Research References\n\n## Existing\nBody.\n", encoding="utf-8")

    artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
        references_path=references_path,
    )
    second_artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
        references_path=references_path,
    )

    mod.validate_artifact(artifact)
    mod.validate_artifact(second_artifact)
    mod.validate_markdown_note(note_path.read_text(encoding="utf-8"))
    saved_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    studying = studying_path.read_text(encoding="utf-8")
    references = references_path.read_text(encoding="utf-8")

    assert saved_artifact == artifact
    assert studying.count("2026-06-13 Exp 4162") == 1
    assert references.count("2026-06-13 Exp 4162") == 1
    assert "flagged_for_v386" in references
    assert "Flagged for .386" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4162() -> None:
    """REQ-REPORT-4162: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    references_once = mod._with_references_section(without_marker)
    starts_with_heading = mod._with_references_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    references_refreshed = mod._with_references_section(references_once)
    marker_at_end = mod._with_references_section(
        references_once.split("\n## Existing")[0]
    )
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-13 Exp 4162") < studying_once.index("## Existing")
    assert references_once.index("2026-06-13 Exp 4162") < references_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-13 Exp 4162")
    assert studying_refreshed.count("2026-06-13 Exp 4162") == 1
    assert references_refreshed.count("2026-06-13 Exp 4162") == 1
    assert marker_at_end.count("2026-06-13 Exp 4162") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "## Existing\nBody." in references_refreshed
    assert no_heading.rstrip().endswith(
        "energy-verifier-vs-LLM-judge efficiency head-to-head first."
    )


def test_main_prints_terminal_verdict_for_req_report_4162(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4162: CLI entry point writes default repo-root outputs."""

    calls: dict[str, Path] = {}

    def fake_write_outputs(
        *,
        note_path: Path,
        artifact_path: Path,
        studying_path: Path,
        references_path: Path,
    ) -> dict[str, object]:
        calls["note_path"] = note_path
        calls["artifact_path"] = artifact_path
        calls["studying_path"] = studying_path
        calls["references_path"] = references_path
        return {"honest_verdict": "complete: sota_ingestion_verifier_moat_guidance_mapped"}

    monkeypatch.setattr(mod, "write_outputs", fake_write_outputs)

    assert mod.main() == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == "complete: sota_ingestion_verifier_moat_guidance_mapped"
    assert calls["note_path"].as_posix().endswith(NOTE_PATH.as_posix())
    assert calls["artifact_path"].as_posix().endswith(ARTIFACT_PATH.as_posix())
    assert calls["studying_path"].as_posix().endswith(STUDYING_PATH.as_posix())
    assert calls["references_path"].as_posix().endswith(REFERENCES_PATH.as_posix())


def test_module_main_guard_exits_zero_for_req_report_4162(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4162: direct script execution exits after writing outputs."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == (
        "complete: sota_ingestion_verifier_moat_guidance_mapped"
    )


def test_deliverable_files_validate_against_req_report_4162() -> None:
    """REQ-REPORT-4162: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")
    references = REFERENCES_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 5
    assert artifact["flagged_for_v386"] == (
        "entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386"
    )
    assert (
        "2026-06-13 Exp 4162 - .386 verifier-moat guidance SOTA ingestion ingested"
        in studying
    )
    assert (
        "Flagged for .386: "
        "`entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386`"
    ) in studying
    assert "2026-06-13 Exp 4162" in references
    assert "flagged_for_v386" in references
