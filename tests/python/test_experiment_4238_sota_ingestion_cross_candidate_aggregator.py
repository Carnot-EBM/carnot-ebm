"""Tests for REQ-REPORT-4238 / SCENARIO-REPORT-4238."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4238_sota_ingestion_cross_candidate_aggregator as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/sota-ingestion-cross-candidate-aggregator-v393-2026-06-15.md"
)
ARTIFACT_PATH = Path(
    "results/experiment_4238_sota_ingestion_cross_candidate_aggregator.json"
)
WRAPPER_PATH = Path(
    "results/experiment_4238_sota_ingestion_cross_candidate_aggregator.py"
)
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v393=mod.DEFAULT_FLAGGED_FOR_V393,
    )


def test_req_report_4238_spec_anchor_exists() -> None:
    """REQ-REPORT-4238: OpenSpec declares the .393 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4238" in spec
    assert "SCENARIO-REPORT-4238" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v393" in spec
    assert "Exp 4232 reports ARC headroom while the aggregator ties vote" in spec
    assert "disambiguation_read=ARC_null_is_data_sparsity" in spec
    for source in mod.VERIFIED_SOURCE_URLS:
        assert source in spec


def test_build_artifact_has_required_fields_for_req_report_4238() -> None:
    """REQ-REPORT-4238: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v393": mod.DEFAULT_FLAGGED_FOR_V393,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real arXiv ID/URL; an ingestion note "
                "without verifiable citations is treated as fabrication "
                "(adversarial_verify discipline)."
            ),
            "flagged_for_v393": (
                "Closes discover->ingest->plan: names the strongest method for "
                "the next planner, conditioned on the A2/A3 outcomes."
            ),
        },
    }


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (
            _valid_artifact() | {"field_principles": {"honest_verdict": "loose"}},
            "field_principles",
        ),
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:4]}, "five to eight"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "fake",
                        "arxiv_id_or_url": "9999.99999",
                        "url": "https://arxiv.org/abs/9999.99999",
                        "carnot_stack_mapping": "fake",
                        "a2_a3_mapping": "fake",
                        "failure_mode": "fake",
                        "experiment_mapping": "fake",
                    }
                ]
                + _valid_methods()[1:]
            },
            "verified source",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"url": "https://example.com/2404.06912"}
                ]
                + _valid_methods()[1:]
            },
            "url",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {"name": "Set-Encoder", "arxiv_id_or_url": "2404.06912"}
                ]
                + _valid_methods()[1:]
            },
            "exactly",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate source",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0] | {"failure_mode": ""}] + _valid_methods()[1:]},
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v393": ""}, "flagged_for_v393"),
        (
            _valid_artifact() | {"flagged_for_v393": "margin_trigger_only_v393"},
            "Set-Encoder",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4238(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4238: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4238: artifact fields are exact."""

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


def test_validate_markdown_note_checks_scenario_report_4238_sections() -> None:
    """SCENARIO-REPORT-4238: note maps sources to required axes."""

    note = """
    ## Fresh-pass provenance
    reliable-channel sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch HTTP 429.
    ## Exp 4231 A2 build, Exp 4232 ARC A3, and Exp 4233 code read
    oracle_distinct_auroc=0.7865558646 positive_candidate_n=20
    wrong_majority_n=9 no_learnable_gain_reason=too_few_positives_after_growth
    aggregator_minus_vote_delta=0.0 oracle_minus_vote=0.1730769231
    oracle_distinct_beats_vote=false held_out_task_n=52
    code_predictor_minus_vote_delta=0.03125 CI95 [0.00625, 0.0625]
    disambiguation_read=ARC_null_is_data_sparsity.
    ## SOTA -> experiment mapping
    arXiv:2404.06912 arXiv:2509.19681 arXiv:2606.04323 arXiv:2512.15146
    arXiv:2602.03975 arXiv:2603.03417 arXiv:2509.06870 arXiv:2602.09341.
    isolated scoring class imbalance margin-trigger under-power SCOPE
    adaptive allocation MSV AggLM AgentAuditor synthesize corrected grid
    bigger ARC pool full Set-Encoder.
    Carnot stack mapping. A2/A3 mapping. Failure mode. Experiment mapping.
    ## Flagged for .393
    bigger_arc_pool_full_set_encoder_agglm_aggregator_v393
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4238: note must cite sources and close the loop."""

    with pytest.raises(ValueError, match="Flagged for .393"):
        mod.validate_markdown_note("## Fresh-pass provenance\narXiv:2404.06912\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(mod.NOTE_MARKDOWN.replace("arXiv:2404.06912", "Set-Encoder"))

    with pytest.raises(ValueError, match="oracle_distinct_beats_vote=false"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("oracle_distinct_beats_vote=false", "pending")
        )

    with pytest.raises(ValueError, match="ARC_null_is_data_sparsity"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("ARC_null_is_data_sparsity", "bounded")
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4238(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4238: writer emits note, artifact, and studying entry."""

    note_path = tmp_path / "note.md"
    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")

    artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
    )
    second_artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
    )

    mod.validate_artifact(artifact)
    mod.validate_artifact(second_artifact)
    mod.validate_markdown_note(note_path.read_text(encoding="utf-8"))
    saved_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    studying = studying_path.read_text(encoding="utf-8")

    assert saved_artifact == artifact
    assert studying.count("2026-06-15 Exp 4238") == 1
    assert "flagged_for_v393" in studying
    assert "Flagged for .393" in studying
    assert "oracle_distinct_beats_vote=false" in studying
    assert "ARC_null_is_data_sparsity" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4238() -> None:
    """REQ-REPORT-4238: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-15 Exp 4238") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-15 Exp 4238")
    assert studying_refreshed.count("2026-06-15 Exp 4238") == 1
    assert marker_at_end.count("2026-06-15 Exp 4238") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert no_heading.rstrip().endswith(
        "build a bigger ARC pool before declaring the oracle-distinct selection thesis bounded."
    )


def test_main_prints_terminal_verdict_for_req_report_4238(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4238: CLI entry point writes default repo-root outputs."""

    calls: dict[str, Path] = {}

    def fake_write_outputs(
        *,
        note_path: Path,
        artifact_path: Path,
        studying_path: Path,
    ) -> dict[str, object]:
        calls["note_path"] = note_path
        calls["artifact_path"] = artifact_path
        calls["studying_path"] = studying_path
        return {"honest_verdict": mod.DEFAULT_HONEST_VERDICT}

    monkeypatch.setattr(mod, "write_outputs", fake_write_outputs)

    assert mod.main() == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert calls["note_path"].as_posix().endswith(NOTE_PATH.as_posix())
    assert calls["artifact_path"].as_posix().endswith(ARTIFACT_PATH.as_posix())
    assert calls["studying_path"].as_posix().endswith(STUDYING_PATH.as_posix())


def test_module_main_guard_exits_zero_for_req_report_4238(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4238: direct module execution exits after writing outputs."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT


def test_wrapper_script_runs_module_for_req_report_4238(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4238: required results/ wrapper delegates to the module."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT


def test_deliverable_files_validate_against_req_report_4238() -> None:
    """REQ-REPORT-4238: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v393"] == mod.DEFAULT_FLAGGED_FOR_V393
    assert "2026-06-15 Exp 4238 - .392 planning sweep SOTA ingestion ingested" in studying
    assert (
        "Flagged for .393: `bigger_arc_pool_full_set_encoder_agglm_aggregator_v393`"
        in studying
    )
